//! # bevytiles
//!
//! Geo-spatial terrain streaming for Bevy — a Rust port of the
//! [raytiles](https://github.com/ziv/raytiles) engine. Streams satellite
//! imagery, [Terrarium](https://registry.opendata.aws/terrain-tiles/)
//! heightmaps, and normal maps around a moving camera and renders them as
//! terrain meshed on the CPU, so physics and sensors can share it.
//!
//! ## Quick start
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevytiles::prelude::*;
//!
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .insert_resource(WorldConfig::from_lat_lon(46.206889, 9.497194))
//!     .add_plugins(TerrainPlugin)
//!     .add_systems(Startup, |mut c: Commands, w: Res<WorldConfig>| {
//!         c.spawn((Camera3d::default(), TerrainCamera,
//!                  Transform::from_translation(w.initial_position(5000.0))));
//!     })
//!     .run();
//! ```
//!
//! Configuration is four resources ([`WorldConfig`](config::WorldConfig),
//! [`StreamingConfig`](config::StreamingConfig),
//! [`RenderingConfig`](config::RenderingConfig),
//! [`NetworkConfig`](config::NetworkConfig)), all defaulted — insert any of
//! them before `run()` to override. Mark the streaming camera with
//! [`TerrainCamera`](config::TerrainCamera).
//!
//! ## Architecture
//!
//! One [`Entity`] per resident tile, each its own CPU mesh under a
//! `StandardMaterial` with the tile's imagery, so Bevy's visibility, lights,
//! shadows and fog apply as to anything else. Around that, the raytiles
//! design:
//!
//! | module | role |
//! |---|---|
//! | [`lod`] | pure desired-set policy (which tiles *should* exist for a camera position): raytiles' quadtree, or fixed rings |
//! | [`geometry`] | the CPU terrain: one continuous height function per elevation zoom, tile meshes that meet exactly, normals from its slope |
//! | [`source`] | tile fetching (threads + disk cache, or blocking) + decode + geometry, delivering whole-tile payloads through channels |
//! | [`store`] | ECS systems tying it together: eviction, budgeted promotion, desired-set upkeep, status (`render` feature) |
//! | [`height`] | the resident tiles' geometry answering [`ground_height`](height::ground_height) queries |
//!
//! Without the `render` feature the crate is the LOD policy, the source and
//! cache and the geometry alone: what a headless consumer (physics colliders,
//! depth sensors, a preload) needs, with the same vertices the plugin draws.
//!
//! ## Coordinate spaces
//!
//! Two frames, one convention: **`absolute = user − world_offset`**.
//! Tile math lives in absolute space (fixed origin at the anchor tile, x
//! east, z south); entity transforms live in user space (small floats near
//! the camera). A [`TerrainRoot`](config::TerrainRoot) entity's transform
//! places user space in the world (to turn it into another convention's
//! axes, say). For worlds larger than a few kilometers the app must
//! *rebase*: shift the camera, every user-space entity, and
//! [`TerrainAnchor::world_offset`](config::TerrainAnchor) by the same amount
//! — the plugin rebakes tile transforms whenever the offset changes. See
//! `examples/demo.rs` for the pattern.
//!
//! ## Frame loop
//!
//! Systems run in [`Update`], ordered by [`TerrainSet`]:
//! reconcile → promote → update-desired → status. The order is load-bearing;
//! see each variant's documentation.

#![warn(missing_docs)]

pub mod config;
pub mod geometry;
pub mod height;
pub mod lod;
pub mod source;
#[cfg(feature = "render")]
pub mod store;
pub mod synth;

/// The commonly needed surface: plugin, configs, markers, and queries.
pub mod prelude {
    pub use crate::config::{
        NetworkConfig, PinnedTiles, RenderingConfig, StreamingConfig, TerrainAnchor,
        TerrainCamera, TerrainRoot, TerrainStatus, WorldConfig, MAX_ZOOM, MIN_ZOOM, ZOOM_LEVELS,
    };
    pub use crate::geometry::TileGeometry;
    pub use crate::height::{ground_height, HeightGrids};
    pub use crate::lod::{LodPolicy, Ring, TileKey};
    pub use crate::source::Fetcher;
    #[cfg(feature = "render")]
    pub use crate::TerrainPlugin;
}

#[cfg(feature = "render")]
use bevy::prelude::*;

/// Ordering of the terrain systems within [`Update`]. Chained in declaration
/// order; the sequence mirrors raytiles' frame loop and is load-bearing —
/// in particular, [`Status`](TerrainSet::Status) must observe the desired set
/// *after* [`UpdateDesired`](TerrainSet::UpdateDesired) rebuilt it, or the
/// initial-loading flag flips before anything was ever requested.
#[cfg(feature = "render")]
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum TerrainSet {
    /// Evict resident tiles that are no longer needed (not desired and
    /// off-screen / beyond the horizon / covered by other zoom levels).
    Reconcile,
    /// Drain the [`source`](crate::source::TileSource) and spawn newly
    /// finished tiles, capped per frame
    /// ([`StreamingConfig::max_promotions_per_frame`](config::StreamingConfig)).
    Promote,
    /// Rebuild the desired tile set when the camera moved far enough; cancel
    /// stale downloads and request missing tiles.
    UpdateDesired,
    /// Update [`TerrainStatus`](config::TerrainStatus) and rebake transforms
    /// after a rebase.
    Status,
}

/// The terrain engine. Add after `DefaultPlugins`; insert any of the config
/// resources before or during `Startup` to override the defaults (the tile
/// source is created on the first `Update` that has a camera).
///
/// Streaming follows the first entity marked with
/// [`TerrainCamera`](config::TerrainCamera); with none present the terrain
/// systems idle. Light the scene as any other (a `DirectionalLight`, say).
#[cfg(feature = "render")]
pub struct TerrainPlugin;

#[cfg(feature = "render")]
impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<config::WorldConfig>()
            .init_resource::<config::StreamingConfig>()
            .init_resource::<config::RenderingConfig>()
            .init_resource::<config::NetworkConfig>()
            .init_resource::<config::TerrainAnchor>()
            .init_resource::<config::TerrainStatus>()
            .init_resource::<config::PinnedTiles>()
            .init_resource::<height::HeightGrids>()
            .init_resource::<store::TileIndex>()
            .init_resource::<store::PendingPayloads>()
            .init_resource::<store::LastDesiredPos>()
            .init_resource::<store::CoverageDirty>()
            .configure_sets(
                Update,
                (
                    TerrainSet::Reconcile,
                    TerrainSet::Promote,
                    TerrainSet::UpdateDesired,
                    TerrainSet::Status,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    store::reconcile.in_set(TerrainSet::Reconcile),
                    store::promote.in_set(TerrainSet::Promote),
                    store::update_desired.in_set(TerrainSet::UpdateDesired),
                    (store::status, store::rebase).in_set(TerrainSet::Status),
                ),
            );
    }
}
