//! CPU-side heights for `ground_height` queries: every resident tile's
//! [`TileGeometry`], sampled on its drawn triangles, so the answer is the
//! ground the camera sees (built on the worker threads along with the mesh).

use crate::config::{TerrainAnchor, WorldConfig, MAX_ZOOM};
use crate::geometry::TileGeometry;
use crate::lod::TileKey;
use bevy::prelude::*;
use std::collections::HashMap;

/// All resident tiles' geometry, keyed like the tile entities. Kept in
/// lockstep with residency by the [`store`](crate::store) systems; read it
/// via [`ground_height`].
#[derive(Resource, Default)]
pub struct HeightGrids(pub HashMap<TileKey, TileGeometry>);

/// Terrain altitude under `pos` (user space, in the terrain root's frame):
/// walks zooms finest→coarsest and samples the first resident tile holding
/// the XZ point, on its triangles (a ring's drop is not included).
pub fn ground_height(
    grids: &HeightGrids,
    world: &WorldConfig,
    anchor: &TerrainAnchor,
    pos: Vec3,
) -> Option<f32> {
    let abs = pos - anchor.world_offset;
    for zoom in (world.base_zoom..=MAX_ZOOM).rev() {
        let key = world.key_at(zoom, f64::from(abs.x), f64::from(abs.z));
        if let Some(geometry) = grids.0.get(&key) {
            let size = geometry.size;
            let u = ((f64::from(abs.x) - f64::from(key.x) * size) / size) as f32;
            let v = ((f64::from(abs.z) - f64::from(key.z) * size) / size) as f32;
            return Some(geometry.height_at(u, v));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_height_samples_the_finest_resident_tile() {
        let world = WorldConfig::default();
        let flat = |key: TileKey, h: f32| TileGeometry {
            key,
            size: world.zoom_size(key.zoom),
            segments: 1,
            heights: vec![h; 4],
            normals: vec![[0.0, 1.0, 0.0]; 4],
        };
        let mut grids = HeightGrids::default();
        let coarse = TileKey {
            zoom: 9,
            x: 0,
            z: 0,
        };
        let fine = TileKey {
            zoom: 10,
            x: 1,
            z: 0,
        };
        grids.0.insert(coarse, flat(coarse, 100.0));
        grids.0.insert(fine, flat(fine, 300.0));
        let anchor = TerrainAnchor::default();
        let at = |x: f32, z: f32| ground_height(&grids, &world, &anchor, Vec3::new(x, 0.0, z));
        assert_eq!(at(10.0, 10.0), Some(100.0));
        assert_eq!(at(40_000.0, 10.0), Some(300.0));
        assert_eq!(at(-10.0, 10.0), None);
    }
}
