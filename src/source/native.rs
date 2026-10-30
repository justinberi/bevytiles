//! The worker pool: threads around one [`Fetcher`]. See the module docs in
//! [`super`].

use super::{Fetcher, TileDrop, TilePayload, TileRequest, CANCELLED};
use crate::config::{NetworkConfig, WorldConfig};
use crate::lod::TileKey;
use crossbeam_channel::{Receiver, Sender};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

struct Job {
    req: TileRequest,
    cancelled: Arc<AtomicBool>,
}

type InFlight = Arc<Mutex<HashMap<TileKey, Arc<AtomicBool>>>>;

/// The background tile fetcher. Owns the worker threads; hand out requests
/// and drain results once per frame. Dropping it disconnects the queue and
/// joins the workers (an in-flight HTTP read may delay that by its timeout).
pub struct TileSource {
    in_flight: InFlight,
    jobs_tx: Option<Sender<Job>>,
    payload_rx: Receiver<TilePayload>,
    drop_rx: Receiver<TileDrop>,
    workers: Vec<JoinHandle<()>>,
}

impl TileSource {
    /// Spawn the worker pool for a world.
    pub fn new(cfg: &NetworkConfig, world: &WorldConfig) -> Self {
        let fetcher = Arc::new(Fetcher::new(cfg));
        let world = Arc::new(world.clone());
        let (jobs_tx, jobs_rx) = crossbeam_channel::unbounded::<Job>();
        let (payload_tx, payload_rx) = crossbeam_channel::unbounded::<TilePayload>();
        let (drop_tx, drop_rx) = crossbeam_channel::unbounded::<TileDrop>();
        let in_flight: InFlight = Arc::new(Mutex::new(HashMap::new()));
        let workers = (0..cfg.threads.max(1))
            .map(|_| {
                let w = Worker {
                    fetcher: fetcher.clone(),
                    world: world.clone(),
                    in_flight: in_flight.clone(),
                    jobs: jobs_rx.clone(),
                    payloads: payload_tx.clone(),
                    drops: drop_tx.clone(),
                };
                std::thread::spawn(move || w.run())
            })
            .collect();
        Self {
            in_flight,
            jobs_tx: Some(jobs_tx),
            payload_rx,
            drop_rx,
            workers,
        }
    }

    /// Queue one tile fetch. Dedup by key: a no-op while the key is in flight
    /// (a cancelled-but-uncollected job counts). Every request is answered
    /// exactly once — payload or drop.
    pub fn request(&self, req: TileRequest) {
        let mut in_flight = self.in_flight.lock().unwrap();
        if in_flight.contains_key(&req.key) {
            return;
        }
        let flag = Arc::new(AtomicBool::new(false));
        in_flight.insert(req.key, flag.clone());
        if let Some(tx) = &self.jobs_tx {
            let _ = tx.send(Job {
                req,
                cancelled: flag,
            });
        }
    }

    /// Flag a job for cancellation (checked at pickup and between assets).
    pub fn cancel(&self, key: TileKey) {
        if let Some(flag) = self.in_flight.lock().unwrap().get(&key) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Collect everything finished since the last drain (non-blocking).
    pub fn drain(&self, ready: &mut Vec<TilePayload>, dropped: &mut Vec<TileDrop>) {
        ready.extend(self.payload_rx.try_iter());
        dropped.extend(self.drop_rx.try_iter());
    }
}

impl Drop for TileSource {
    fn drop(&mut self) {
        // disconnect the queue; workers exit once it is empty
        self.jobs_tx = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

struct Worker {
    fetcher: Arc<Fetcher>,
    world: Arc<WorldConfig>,
    in_flight: InFlight,
    jobs: Receiver<Job>,
    payloads: Sender<TilePayload>,
    drops: Sender<TileDrop>,
}

impl Worker {
    fn run(self) {
        while let Ok(job) = self.jobs.recv() {
            self.handle(job);
        }
    }

    fn handle(&self, job: Job) {
        let key = job.req.key;
        let cancelled = || job.cancelled.load(Ordering::Relaxed);
        let result: Result<TilePayload, String> = (|| {
            if cancelled() {
                return Err(CANCELLED.into());
            }
            let albedo = self.fetcher.texture(key.zoom, job.req.x, job.req.z)?;
            if cancelled() {
                return Err(CANCELLED.into());
            }
            let geometry = self.fetcher.geometry(&self.world, key)?;
            Ok(TilePayload {
                key,
                albedo,
                geometry,
            })
        })();

        self.in_flight.lock().unwrap().remove(&key);
        match result {
            _ if cancelled() => {
                let _ = self.drops.send(TileDrop::Cancelled(key));
            }
            Ok(payload) => {
                let _ = self.payloads.send(payload);
            }
            Err(e) if e == CANCELLED => {
                let _ = self.drops.send(TileDrop::Cancelled(key));
            }
            Err(e) => {
                let _ = self.drops.send(TileDrop::Failed(key, e));
            }
        }
    }
}
