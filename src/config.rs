//! Public configuration resources. Mirrors raytiles' nested `config` — every
//! field defaulted; insert any of these before or during `Startup` to
//! override (the plugin reads them from the first `Update` on).

use crate::lod::{LodPolicy, TileKey};
use bevy::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

/// Lowest zoom level the engine supports; `WorldConfig::base_zoom` must not
/// go below it.
pub const MIN_ZOOM: u8 = 9;
/// Highest zoom level the engine supports; `WorldConfig::max_zoom` and every
/// ring's zoom must not exceed it.
pub const MAX_ZOOM: u8 = 22;
/// Number of zoom levels in `[MIN_ZOOM, MAX_ZOOM]`; sizes every per-zoom
/// array (`thresholds`). Slot `i` applies to zoom `base_zoom + i`; slots
/// beyond `max_zoom - base_zoom` are ignored.
pub const ZOOM_LEVELS: usize = (MAX_ZOOM - MIN_ZOOM + 1) as usize; // 14

/// The WGS84 equator, which web-mercator tiles divide.
pub const EQUATOR_CIRCUMFERENCE_M: f64 = 40_075_016.686;

/// World topology and the terrain's height function. Effectively immutable
/// once the plugin started.
#[derive(Resource, Clone)]
pub struct WorldConfig {
    /// Anchor tile X at `base_zoom`: the world origin sits at this tile's
    /// corner. Overwritten by [`WorldConfig::from_lat_lon`].
    pub anchor_x: i32,
    /// Anchor tile Z (slippy-map `y`) at `base_zoom`.
    pub anchor_z: i32,
    /// Lowest LOD zoom ever loaded (>= MIN_ZOOM).
    pub base_zoom: u8,
    /// Highest quadtree LOD zoom (<= MAX_ZOOM). Imagery must exist this deep;
    /// heights above `elevation_zoom` are sampled from it.
    pub max_zoom: u8,
    /// World size (meters) of one tile at `base_zoom`.
    pub tile_size: f32,
    /// World-space offset of the anchor point inside its anchor tile; filled
    /// by `from_lat_lon` so the origin sits exactly on the coordinate.
    pub origin_offset: Vec3,
    /// Zoom of the Terrarium heightmaps every tile's heights come from: a
    /// tile at or below it samples its own zoom's, a finer tile this one's.
    /// One continuous height function per zoom, read across tile edges, so
    /// neighbours meet exactly (see [`geometry`](crate::geometry)).
    pub elevation_zoom: u8,
    /// Terrain relief exaggeration applied to every height.
    pub height_scale: f32,
    /// Fewest mesh cells along a tile's side (a fine tile over coarse
    /// elevation still bends with it).
    pub min_segments: u32,
    /// Most mesh cells along a tile's side (one per elevation texel, capped).
    pub max_segments: u32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            anchor_x: 306,
            anchor_z: 207,
            base_zoom: MIN_ZOOM,
            max_zoom: 15,
            tile_size: 66_400.0,
            origin_offset: Vec3::ZERO,
            elevation_zoom: 15,
            height_scale: 1.0,
            min_segments: 4,
            max_segments: 64,
        }
    }
}

impl WorldConfig {
    /// Anchor the world at a geographic coordinate (degrees): derives the
    /// anchor tile, tile size, and origin offset (web-mercator, same math as
    /// raytiles).
    pub fn from_lat_lon(lat: f64, lon: f64) -> Self {
        Self::from_lat_lon_on(lat, lon, EQUATOR_CIRCUMFERENCE_M)
    }

    /// [`from_lat_lon`](Self::from_lat_lon) with the meters of a given
    /// sphere's circumference, to agree with another system's geodesy (a GPS
    /// model's sphere, say) on how far a degree is.
    pub fn from_lat_lon_on(lat: f64, lon: f64, circumference_m: f64) -> Self {
        let n = 2f64.powi(MIN_ZOOM as i32);
        let lat_rad = lat.to_radians();
        let x = (lon + 180.0) / 360.0 * n;
        let y = (1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n;
        let anchor_x = x.floor() as i32;
        let anchor_z = y.floor() as i32;
        let tile_size = circumference_m * lat_rad.cos() / n;
        Self {
            anchor_x,
            anchor_z,
            tile_size: tile_size as f32,
            origin_offset: Vec3::new(
                ((x - anchor_x as f64) * tile_size) as f32,
                0.0,
                ((y - anchor_z as f64) * tile_size) as f32,
            ),
            ..Default::default()
        }
    }

    /// A sensible initial camera position over the anchor, `altitude` m up.
    pub fn initial_position(&self, altitude: f32) -> Vec3 {
        self.origin_offset + Vec3::Y * altitude
    }

    /// World size (meters) of one tile at `zoom`: halves per level above base.
    pub fn zoom_size(&self, zoom: u8) -> f64 {
        f64::from(self.tile_size) / 2f64.powi(i32::from(zoom) - i32::from(self.base_zoom))
    }

    /// Absolute position (x, z meters) of a provider tile coordinate at
    /// `zoom` (fractional: `x + u` within tile `x`). f64 throughout.
    pub fn tile_to_absolute(&self, zoom: u8, x: f64, y: f64) -> (f64, f64) {
        let scale = 2f64.powi(i32::from(zoom) - i32::from(self.base_zoom));
        let size = f64::from(self.tile_size);
        (
            (x / scale - f64::from(self.anchor_x)) * size,
            (y / scale - f64::from(self.anchor_z)) * size,
        )
    }

    /// The provider tile coordinate at `zoom` (fractional) of an absolute
    /// position: the inverse of [`tile_to_absolute`](Self::tile_to_absolute).
    pub fn absolute_to_tile(&self, zoom: u8, x: f64, z: f64) -> (f64, f64) {
        let scale = 2f64.powi(i32::from(zoom) - i32::from(self.base_zoom));
        let size = f64::from(self.tile_size);
        (
            (x / size + f64::from(self.anchor_x)) * scale,
            (z / size + f64::from(self.anchor_z)) * scale,
        )
    }

    /// The anchor-relative key of the tile at `zoom` holding an absolute
    /// position.
    pub fn key_at(&self, zoom: u8, x: f64, z: f64) -> TileKey {
        let size = self.zoom_size(zoom);
        TileKey {
            zoom,
            x: (x / size).floor() as i32,
            z: (z / size).floor() as i32,
        }
    }

    /// Absolute provider tile (x, y) of an anchor-relative key.
    pub fn provider_tile(&self, key: TileKey) -> (i32, i32) {
        let scale = 1i32 << (key.zoom - self.base_zoom);
        (key.x + self.anchor_x * scale, key.z + self.anchor_z * scale)
    }

    /// The zoom of the heightmaps a tile at `zoom` samples.
    pub fn elevation_zoom_for(&self, zoom: u8) -> u8 {
        zoom.min(self.elevation_zoom)
    }
}

/// Which tiles are kept resident and how aggressively the set updates.
#[derive(Resource, Clone)]
pub struct StreamingConfig {
    /// Radius, in base-zoom tiles, of the disc loaded around the camera
    /// (quadtree policy).
    pub radius: i32,
    /// Per-zoom subdivision distance thresholds (meters),
    /// `thresholds[zoom - base_zoom]` (quadtree policy).
    pub thresholds: [f32; ZOOM_LEVELS],
    /// Which tiles the camera wants: the adaptive quadtree, or fixed rings.
    pub policy: LodPolicy,
    /// Camera travel (meters) that triggers a desired-set rebuild.
    pub update_distance: f32,
    /// Cap on tile promotions (entity spawns + asset creation) per frame.
    pub max_promotions_per_frame: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            radius: 6,
            thresholds: [
                100_000.0, 80_000.0, 40_000.0, 20_000.0, 10_000.0, 5_000.0, 2_500.0, 1_250.0,
                625.0, 312.0, 156.0, 78.0, 39.0, 20.0,
            ],
            policy: LodPolicy::Quadtree,
            update_distance: 500.0,
            max_promotions_per_frame: 8,
        }
    }
}

/// Rendering parameters, read when a tile is promoted. The terrain is drawn
/// with Bevy's `StandardMaterial`, so lights, shadows and `DistanceFog` are
/// the app's.
#[derive(Resource, Clone)]
pub struct RenderingConfig {
    /// Depth of the skirt hung from every tile's edges, in the tile's mesh
    /// cells (the skirt drops this many cell widths). Hides the cracks where
    /// zoom levels meet; 0 disables.
    pub skirt_cells: f32,
    /// Whether tiles cast shadows (they always receive them).
    pub cast_shadows: bool,
}

impl Default for RenderingConfig {
    fn default() -> Self {
        Self {
            skirt_cells: 0.5,
            cast_shadows: false,
        }
    }
}

/// Tile download / cache parameters.
#[derive(Resource, Clone)]
pub struct NetworkConfig {
    /// Number of background download/geometry worker threads. Downloads are
    /// I/O bound, so more threads than cores is fine.
    pub threads: usize,
    /// Root of the on-disk cache.
    pub cache_dir: PathBuf,
    /// Provider URL templates with `:zoom:`/`:x:`/`:y:` tokens. The Esri
    /// texture default uses `zoom/y/x` order — that swap is intentional.
    pub texture_url: String,
    /// Terrarium heightmap URL template (`zoom/x/y` order).
    pub heightmap_url: String,
    /// Where a texture's response body is kept, under `cache_dir`, with the
    /// same tokens. Another layout shares the cache with another program.
    pub texture_cache: String,
    /// Where a heightmap's response body is kept, under `cache_dir`.
    pub heightmap_cache: String,
    /// Read the cache alone, never HTTP: a tile missing from it fails. For
    /// runs that must repeat exactly, after a preload.
    pub offline: bool,
    /// The User-Agent header of every request.
    pub user_agent: String,
    /// HTTP connection timeout per request.
    pub connect_timeout: Duration,
    /// HTTP read timeout per request.
    pub read_timeout: Duration,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            threads: 4,
            cache_dir: PathBuf::from(".cache"),
            texture_url:
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/:zoom:/:y:/:x:"
                    .into(),
            heightmap_url: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/:zoom:/:x:/:y:.png".into(),
            texture_cache: "texture/:zoom:/:x:/:y:.png".into(),
            heightmap_cache: "heightmap/:zoom:/:x:/:y:.png".into(),
            offline: false,
            user_agent: concat!("bevytiles/", env!("CARGO_PKG_VERSION")).into(),
            connect_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(3),
        }
    }
}

/// Large-world shifting input: `absolute = user − world_offset`. The app owns
/// rebasing (shift camera + everything + this offset together); the plugin
/// rebakes tile transforms whenever it changes.
#[derive(Resource, Default, Clone)]
pub struct TerrainAnchor {
    /// The current user-space offset of the world origin. Mutate only as part
    /// of a rebase that shifts the camera and every user-space entity by the
    /// same amount.
    pub world_offset: Vec3,
}

/// Marker for the entity that drives streaming: a camera, or whatever the
/// terrain should follow (a vehicle). The first one found is used.
#[derive(Component)]
pub struct TerrainCamera;

/// Marker for an entity whose transform places the terrain: tiles spawn as
/// its children, and the camera is taken into its space. Without one, user
/// space is the world.
#[derive(Component)]
pub struct TerrainRoot;

/// Tiles the app draws itself (with [`geometry`](crate::geometry), say, to
/// give them colliders): the LOD policy never requests them, and the
/// quadtree subdivides down to them so nothing is drawn over them.
#[derive(Resource, Default, Clone)]
pub struct PinnedTiles(pub HashSet<TileKey>);

/// Initial-load status for splash screens. `loading` starts true and flips
/// only once a desired set exists and is fully serviced.
#[derive(Resource)]
pub struct TerrainStatus {
    /// True during the initial load only.
    pub loading: bool,
    /// Fraction of the desired set that is resident, [0, 1].
    pub progress: f32,
    /// Tiles drawn... resident count (debug convenience).
    pub resident: usize,
}

impl Default for TerrainStatus {
    fn default() -> Self {
        Self {
            loading: true,
            progress: 0.0,
            resident: 0,
        }
    }
}
