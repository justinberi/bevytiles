//! The tile source: a tile's imagery and heightmaps from the disk cache, else
//! HTTP (written through to the cache) unless [`NetworkConfig::offline`],
//! decoded, and its CPU [`TileGeometry`] built from them.
//!
//! Two ways in, sharing one [`Fetcher`]:
//!
//! * [`TileSource`]: plain `std::thread` workers delivering whole-tile
//!   payloads through channels drained once per frame. Deliberately NOT
//!   bevy_tasks: the raytiles pool semantics — dedup by key, cancel flags
//!   checked between assets — are ported as-is.
//! * [`Fetcher`] directly, blocking on the caller's thread: tiles needed now
//!   (an app's own, pinned tiles with colliders), and preloading a cache.
//!
//! [`NetworkConfig::offline`]: crate::config::NetworkConfig::offline

use crate::config::{NetworkConfig, WorldConfig};
use crate::geometry::{self, Elevation, TileGeometry, TILE_PIXELS};
use crate::lod::TileKey;
use image::{ImageReader, RgbaImage};
use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

mod native;
pub use native::TileSource;

/// Decoded heightmaps kept in memory; past it the memo starts over.
const HEIGHTMAP_MEMO: usize = 64;

/// One tile fetch: anchor-relative identity + absolute provider coordinates
/// at `key.zoom` (the caller resolves the anchor).
#[derive(Clone, Copy, Debug)]
pub struct TileRequest {
    /// Anchor-relative identity, used for dedup/cancel and echoed in the
    /// payload/drop.
    pub key: TileKey,
    /// Absolute provider tile column at `key.zoom`.
    pub x: i32,
    /// Absolute provider tile row at `key.zoom`.
    pub z: i32,
}

/// A completed tile: its decoded imagery and its geometry (built on the
/// worker, so the main thread never pays for it).
pub struct TilePayload {
    /// The request's anchor-relative key.
    pub key: TileKey,
    /// Decoded satellite imagery.
    pub albedo: RgbaImage,
    /// The tile's CPU geometry.
    pub geometry: TileGeometry,
}

/// A tile that will not arrive. Every [`TileSource::request`] is answered by
/// exactly one [`TilePayload`] or one of these.
pub enum TileDrop {
    /// The job was cancelled before completing. If the tile is wanted again,
    /// re-request immediately (the camera came back).
    Cancelled(TileKey),
    /// Fetch or decode failed; the string is the reason. Callers should wait
    /// for the next desired-set rebuild rather than hot-retrying a failing
    /// provider.
    Failed(TileKey, String),
}

/// Which of a tile's two assets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The imagery (any format `image` reads: Esri's JPEG).
    Texture,
    /// A Terrarium heightmap (PNG).
    Heightmap,
}

pub(crate) const CANCELLED: &str = "\u{0}cancelled";

/// A template's `:zoom:`, `:x:` and `:y:` tokens filled in.
pub fn expand_url(template: &str, zoom: u8, x: i32, y: i32) -> String {
    template
        .replacen(":zoom:", &zoom.to_string(), 1)
        .replacen(":x:", &x.to_string(), 1)
        .replacen(":y:", &y.to_string(), 1)
}

pub(crate) fn decode_image(bytes: &[u8]) -> Result<image::DynamicImage, String> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image sniff: {e}"))?
        .decode()
        .map_err(|e| format!("image decode: {e}"))
}

/// A Terrarium PNG's heights (meters, row-major from the north-west).
pub fn decode_heightmap(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let img = decode_image(bytes)?.into_rgb8();
    if img.width() as usize != TILE_PIXELS || img.height() as usize != TILE_PIXELS {
        return Err(format!(
            "heightmap {} x {}, not {TILE_PIXELS} x {TILE_PIXELS}",
            img.width(),
            img.height()
        ));
    }
    Ok(crate::synth::decode_terrarium_floats(&img))
}

/// A provider tile: zoom, x, y.
type ProviderTile = (u8, i32, i32);

/// Fetches and decodes tiles, blocking: cache first, HTTP otherwise unless
/// offline. Shareable across threads; keeps recently decoded heightmaps.
pub struct Fetcher {
    cfg: Arc<NetworkConfig>,
    agent: ureq::Agent,
    heightmaps: Mutex<HashMap<ProviderTile, Arc<Vec<f32>>>>,
}

impl Fetcher {
    /// A fetcher with its own HTTP agent.
    pub fn new(cfg: &NetworkConfig) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_connect(Some(cfg.connect_timeout))
            .timeout_recv_response(Some(cfg.read_timeout))
            .timeout_recv_body(Some(cfg.read_timeout))
            .user_agent(cfg.user_agent.as_str())
            .build()
            .into();
        Self {
            cfg: Arc::new(cfg.clone()),
            agent,
            heightmaps: Mutex::new(HashMap::new()),
        }
    }

    /// The configuration it was made with.
    pub fn config(&self) -> &NetworkConfig {
        &self.cfg
    }

    /// Where an asset is kept in the cache.
    pub fn cache_path(&self, kind: Kind, zoom: u8, x: i32, y: i32) -> PathBuf {
        let template = match kind {
            Kind::Texture => &self.cfg.texture_cache,
            Kind::Heightmap => &self.cfg.heightmap_cache,
        };
        self.cfg.cache_dir.join(expand_url(template, zoom, x, y))
    }

    /// An asset's raw bytes: the cache's, else (unless offline) the
    /// provider's, written through to the cache.
    pub fn bytes(&self, kind: Kind, zoom: u8, x: i32, y: i32) -> Result<Vec<u8>, String> {
        let path = self.cache_path(kind, zoom, x, y);
        if path.is_file() {
            return std::fs::read(&path).map_err(|e| format!("cache read {path:?}: {e}"));
        }
        if self.cfg.offline {
            return Err(format!("{} not in the cache (offline)", path.display()));
        }
        let template = match kind {
            Kind::Texture => &self.cfg.texture_url,
            Kind::Heightmap => &self.cfg.heightmap_url,
        };
        let url = expand_url(template, zoom, x, y);
        let bytes = self
            .agent
            .get(&url)
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?
            .into_body()
            .with_config()
            .limit(32 * 1024 * 1024)
            .read_to_vec()
            .map_err(|e| format!("read {url}: {e}"))?;
        write_atomic(&path, &bytes)?;
        Ok(bytes)
    }

    /// A tile's decoded imagery.
    pub fn texture(&self, zoom: u8, x: i32, y: i32) -> Result<RgbaImage, String> {
        Ok(decode_image(&self.bytes(Kind::Texture, zoom, x, y)?)?.into_rgba8())
    }

    /// A decoded heightmap, from memory if decoded lately.
    pub fn heightmap(&self, zoom: u8, x: i32, y: i32) -> Result<Arc<Vec<f32>>, String> {
        if let Some(h) = self.heightmaps.lock().unwrap().get(&(zoom, x, y)) {
            return Ok(h.clone());
        }
        let heights = Arc::new(decode_heightmap(&self.bytes(
            Kind::Heightmap,
            zoom,
            x,
            y,
        )?)?);
        let mut memo = self.heightmaps.lock().unwrap();
        if memo.len() >= HEIGHTMAP_MEMO {
            memo.clear();
        }
        memo.insert((zoom, x, y), heights.clone());
        Ok(heights)
    }

    /// The heightmaps a tile's geometry reads: its home one (required), and
    /// the neighbours at its edges that can be had.
    pub fn elevation(&self, world: &WorldConfig, key: TileKey) -> Result<Elevation, String> {
        let zoom = world.elevation_zoom_for(key.zoom);
        let mut elevation = Elevation {
            zoom,
            ..Default::default()
        };
        for (i, (x, y)) in geometry::heightmaps_for(world, key).into_iter().enumerate() {
            match self.heightmap(zoom, x, y) {
                Ok(h) => {
                    elevation.tiles.insert((x, y), h);
                }
                Err(e) if i == 0 => return Err(e),
                Err(_) => {}
            }
        }
        Ok(elevation)
    }

    /// A tile's CPU geometry.
    pub fn geometry(&self, world: &WorldConfig, key: TileKey) -> Result<TileGeometry, String> {
        Ok(TileGeometry::build(
            world,
            key,
            &self.elevation(world, key)?,
        ))
    }

    /// The height function (absolute meters, times `height_scale`) at an
    /// absolute position, as tiles of `zoom` sample it.
    pub fn height_at(&self, world: &WorldConfig, zoom: u8, x: f64, z: f64) -> Result<f64, String> {
        let key = world.key_at(zoom, x, z);
        let elevation = self.elevation(world, key)?;
        let (fx, fy) = world.absolute_to_tile(elevation.zoom, x, z);
        Ok(elevation.metres(fx, fy) * f64::from(world.height_scale))
    }
}

/// Atomic cache write: unique tmp per writer (two workers may race on the
/// same path — a shared tmp name would interleave; the rename race itself is
/// benign, identical bytes).
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!(
        "{name}.tmp{}.{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp, bytes).map_err(|e| format!("write {tmp:?}: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {path:?}: {e}"))
}
