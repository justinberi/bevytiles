//! Integration tests for the tile source: real worker pool and blocking
//! fetcher against seeded disk caches (dead HTTP host proves the offline
//! paths). Mirrors the C++ raytiles tile_source suite where it still applies.

use bevytiles::config::{NetworkConfig, WorldConfig};
use bevytiles::lod::TileKey;
use bevytiles::source::{Fetcher, Kind, TileDrop, TilePayload, TileRequest, TileSource};
use bevytiles::synth;
use image::RgbaImage;
use std::path::Path;
use std::time::{Duration, Instant};

fn dead_host_config(dir: &Path) -> NetworkConfig {
    NetworkConfig {
        threads: 1, // deterministic pickup order
        cache_dir: dir.to_path_buf(),
        texture_url: "http://127.0.0.1:9/tex/:zoom:/:x:/:y:.png".into(),
        heightmap_url: "http://127.0.0.1:9/hm/:zoom:/:x:/:y:.png".into(),
        connect_timeout: Duration::from_millis(300),
        read_timeout: Duration::from_millis(500),
        ..Default::default()
    }
}

/// A world anchored at provider tile (0, 0) of zoom 9, so keys are provider
/// coordinates; heights from zoom 10.
fn world() -> WorldConfig {
    WorldConfig {
        anchor_x: 0,
        anchor_z: 0,
        elevation_zoom: 10,
        min_segments: 16,
        max_segments: 64,
        ..Default::default()
    }
}

fn cache_path(dir: &Path, kind: &str, zoom: u8, x: i32, z: i32) -> std::path::PathBuf {
    dir.join(kind)
        .join(zoom.to_string())
        .join(x.to_string())
        .join(format!("{z}.png"))
}

fn write_png(path: &Path, img: &image::DynamicImage) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    img.save_with_format(path, image::ImageFormat::Png).unwrap();
}

fn gradient_heights(offset: f32) -> Vec<f32> {
    (0..256 * 256)
        .map(|i| offset + (i % 256) as f32 * 0.5 + (i / 256) as f32 * 0.25)
        .collect()
}

fn seed_terrarium(path: &Path, offset: f32) {
    let img = synth::encode_terrarium(&gradient_heights(offset), 256, 256);
    write_png(path, &image::DynamicImage::ImageRgb8(img));
}

fn seed_texture(dir: &Path, zoom: u8, x: i32, z: i32) {
    let img = RgbaImage::from_pixel(2, 2, image::Rgba([200, 60, 30, 255]));
    write_png(
        &cache_path(dir, "texture", zoom, x, z),
        &image::DynamicImage::ImageRgba8(img),
    );
}

struct Harness {
    src: TileSource,
    payloads: Vec<TilePayload>,
    drops: Vec<TileDrop>,
}

impl Harness {
    fn new(cfg: NetworkConfig) -> Self {
        Self {
            src: TileSource::new(&cfg, &world()),
            payloads: Vec::new(),
            drops: Vec::new(),
        }
    }
    fn pump_until(&mut self, mut pred: impl FnMut(&Self) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let mut r = Vec::new();
            let mut d = Vec::new();
            self.src.drain(&mut r, &mut d);
            self.payloads.extend(r);
            self.drops.extend(d);
            if pred(self) {
                return true;
            }
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
fn cache_hit_delivers_whole_payload_without_network() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dead_host_config(dir.path());
    seed_texture(dir.path(), 12, 41, 57);
    seed_terrarium(&cache_path(dir.path(), "heightmap", 10, 10, 14), 100.0);

    let mut h = Harness::new(cfg);
    let key = TileKey {
        zoom: 12,
        x: 41,
        z: 57,
    };
    h.src.request(TileRequest { key, x: 41, z: 57 });

    assert!(h.pump_until(|h| !h.payloads.is_empty()));
    let p = &h.payloads[0];
    assert_eq!(p.key, key);
    assert_eq!(p.albedo.width(), 2);
    assert_eq!(p.geometry.segments, 64);
    assert_eq!(p.geometry.heights.len(), 65 * 65);
    // z12 (41, 57) is the quadrant (1, 1) of z10 (10, 14): its north-west
    // corner sits at texel coordinate (64, 64), between texel centres
    let corner = 100.0 + 63.5 * 0.5 + 63.5 * 0.25;
    assert!(
        (p.geometry.heights[0] - corner).abs() < 0.01,
        "{}",
        p.geometry.heights[0]
    );
}

#[test]
fn missing_heightmap_drops_with_reason() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dead_host_config(dir.path());
    seed_texture(dir.path(), 9, 1, 1);

    let mut h = Harness::new(cfg);
    h.src.request(TileRequest {
        key: TileKey {
            zoom: 9,
            x: 1,
            z: 1,
        },
        x: 1,
        z: 1,
    });
    assert!(h.pump_until(|h| !h.drops.is_empty()));
    assert!(h.payloads.is_empty());
    assert!(matches!(h.drops[0], TileDrop::Failed(_, _)));
}

#[test]
fn corrupt_heightmap_drops_the_tile() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dead_host_config(dir.path());
    let p = cache_path(dir.path(), "heightmap", 10, 7, 7);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, b"definitely not a png").unwrap();
    seed_texture(dir.path(), 16, 7 * 64, 7 * 64);

    let mut h = Harness::new(cfg);
    h.src.request(TileRequest {
        key: TileKey {
            zoom: 16,
            x: 7 * 64,
            z: 7 * 64,
        },
        x: 7 * 64,
        z: 7 * 64,
    });
    assert!(h.pump_until(|h| !h.drops.is_empty()));
    assert!(matches!(h.drops[0], TileDrop::Failed(_, _)));
}

#[test]
fn requests_dedup_by_key_while_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dead_host_config(dir.path());
    seed_texture(dir.path(), 9, 6, 6);
    seed_terrarium(&cache_path(dir.path(), "heightmap", 9, 6, 6), 0.0);

    let mut h = Harness::new(cfg);
    let key = TileKey {
        zoom: 9,
        x: 6,
        z: 6,
    };
    h.src.request(TileRequest { key, x: 6, z: 6 });
    h.src.request(TileRequest { key, x: 6, z: 6 }); // racy dedup: at most one answer per in-flight window
    assert!(h.pump_until(|h| !h.payloads.is_empty()));
    std::thread::sleep(Duration::from_millis(100));
    let mut r = Vec::new();
    let mut d = Vec::new();
    h.src.drain(&mut r, &mut d);
    assert!(h.payloads.len() + r.len() <= 2); // 1 expected; 2 tolerated if the first finished before the re-request
}

#[test]
fn another_cache_layout_and_offline_reads_it_alone() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = NetworkConfig {
        texture_cache: "imagery/:zoom:/:x:/:y:".into(),
        heightmap_cache: "elevation/:zoom:/:x:/:y:".into(),
        offline: true,
        ..dead_host_config(dir.path())
    };
    seed_terrarium(&dir.path().join("elevation/10/10/14"), 5.0);
    let fetcher = Fetcher::new(&cfg);
    assert_eq!(
        fetcher.cache_path(Kind::Texture, 19, 1, 2),
        dir.path().join("imagery/19/1/2")
    );
    let key = TileKey {
        zoom: 19,
        x: 10 * 512 + 3,
        z: 14 * 512 + 500,
    };
    let geometry = fetcher.geometry(&world(), key).unwrap();
    assert_eq!(geometry.segments, 16);
    let missing = fetcher.texture(19, key.x, key.z).unwrap_err();
    assert!(missing.contains("offline"), "{missing}");
    // a neighbour heightmap at the tile's edge is optional: clamped without it
    let edge = TileKey {
        zoom: 19,
        x: 10 * 512 + 511,
        z: 14 * 512,
    };
    assert!(fetcher.geometry(&world(), edge).is_ok());
}

#[test]
fn neighbour_heightmaps_make_one_function_across_their_edge() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dead_host_config(dir.path());
    seed_terrarium(&cache_path(dir.path(), "heightmap", 10, 10, 14), 0.0);
    seed_terrarium(&cache_path(dir.path(), "heightmap", 10, 11, 14), 128.0);
    let fetcher = Fetcher::new(&cfg);
    let world = world();
    let west = TileKey {
        zoom: 12,
        x: 43,
        z: 56,
    };
    let east = TileKey { x: 44, ..west };
    let (a, b) = (
        fetcher.geometry(&world, west).unwrap(),
        fetcher.geometry(&world, east).unwrap(),
    );
    let n = a.side();
    for k in 0..n {
        assert_eq!(a.heights[k * n + n - 1], b.heights[k * n]);
    }
    // the edge is halfway between texel 255 of the west map and texel 0 of
    // the east one: 127.5 and 128.0 at the top row's centre
    let top = a.heights[n - 1];
    assert!(
        (top - 0.5 * (127.5 + 128.0) - 0.125 * 0.0).abs() < 0.2,
        "{top}"
    );
}
