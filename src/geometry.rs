//! CPU terrain geometry. Every tile's heights come from the Terrarium
//! heightmaps of one zoom ([`WorldConfig::elevation_zoom_for`]), bilinear
//! between texel centres and read across heightmap edges: one continuous
//! height function of position per zoom. Neighbouring tiles sample it at the
//! same points along a shared edge, so they meet exactly; the normals come
//! from its slope, not from each tile's mesh, so the lighting has no seam
//! either. The mesh drawn, a collider and a depth sensor all take the same
//! vertices ([`TileGeometry`]), and [`TileGeometry::height_at`] answers on the
//! drawn triangles.
//!
//! Frame: absolute space, x east, y up, z south (slippy-map `y`), meters.

use crate::config::WorldConfig;
use crate::lod::TileKey;
use std::collections::HashMap;
use std::sync::Arc;

/// Texels along a heightmap's side.
pub const TILE_PIXELS: usize = 256;

/// Offset (in texels) of the central difference giving the normals: a
/// property of the height function, so every zoom's normals agree.
const SLOPE_STEP_TEXELS: f64 = 0.5;

/// Mesh cells per elevation texel: a bilinear texel is a curved patch, which
/// one flat cell per texel (out of step with the texel centres) misses by
/// meters on rough ground; four follow it within a sixteenth of that.
pub const CELLS_PER_TEXEL: u32 = 4;

/// How far past a tile's edge (in texels) its samples reach: the slope's
/// step, the bilinear's next texel and the clamping's one texel.
const MARGIN_TEXELS: f64 = SLOPE_STEP_TEXELS + 1.0 + 1.0;

/// Decoded heightmaps of one zoom (meters, row-major from the north-west),
/// keyed by provider tile (x, y).
#[derive(Clone, Default)]
pub struct Elevation {
    /// Zoom of every heightmap here.
    pub zoom: u8,
    /// The heightmaps loaded (see [`Elevation::metres`] for one missing).
    pub tiles: HashMap<(i32, i32), Arc<Vec<f32>>>,
}

impl Elevation {
    /// Meters at a provider tile coordinate of this zoom (fractional: tile
    /// `x` spans `[x, x + 1)`), bilinear between texel centres. A texel of a
    /// heightmap that isn't loaded is clamped into the point's own heightmap
    /// (or the first loaded one within a texel of it, in a fixed order): a
    /// rule of the point alone, so every tile sampling it agrees. 0 (sea
    /// level, as no data) with none.
    pub fn metres(&self, fx: f64, fy: f64) -> f64 {
        let n = TILE_PIXELS as f64;
        let Some(anchor) = self.anchor(fx * n, fy * n) else {
            return 0.0;
        };
        let px = fx * n - 0.5;
        let py = fy * n - 0.5;
        let (x0, y0) = (px.floor() as i64, py.floor() as i64);
        let (tx, ty) = (px - x0 as f64, py - y0 as f64);
        let at = |gx: i64, gy: i64| self.texel(anchor, gx, gy);
        let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
        let bottom = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    /// The heightmap a point (global texel units) clamps into: its own if
    /// loaded, else the first loaded neighbour within a texel of it.
    fn anchor(&self, gx: f64, gy: f64) -> Option<(i32, i32)> {
        let n = TILE_PIXELS as f64;
        let (mx, my) = ((gx / n).floor() as i32, (gy / n).floor() as i32);
        for dy in [0, -1, 1] {
            for dx in [0, -1, 1] {
                let (x, y) = (mx + dx, my + dy);
                let off_x = (f64::from(x) * n - gx).max(gx - f64::from(x + 1) * n);
                let off_y = (f64::from(y) * n - gy).max(gy - f64::from(y + 1) * n);
                if off_x.max(off_y) <= 1.0 && self.tiles.contains_key(&(x, y)) {
                    return Some((x, y));
                }
            }
        }
        None
    }

    /// The texel at global texel coordinates, from its own heightmap if
    /// loaded, else clamped into `anchor`.
    fn texel(&self, anchor: (i32, i32), gx: i64, gy: i64) -> f64 {
        let n = TILE_PIXELS as i64;
        let tile = (gx.div_euclid(n) as i32, gy.div_euclid(n) as i32);
        if let Some(heights) = self.tiles.get(&tile) {
            return f64::from(heights[(gy.rem_euclid(n) * n + gx.rem_euclid(n)) as usize]);
        }
        let heights = &self.tiles[&anchor];
        let px = (gx - i64::from(anchor.0) * n).clamp(0, n - 1);
        let py = (gy - i64::from(anchor.1) * n).clamp(0, n - 1);
        f64::from(heights[(py * n + px) as usize])
    }
}

/// The heightmap (provider x, y at [`WorldConfig::elevation_zoom_for`]) a
/// tile lies in: its own, or its ancestor's.
pub fn home_heightmap(world: &WorldConfig, key: TileKey) -> (i32, i32) {
    let shift = key.zoom - world.elevation_zoom_for(key.zoom);
    let (x, y) = world.provider_tile(key);
    (x >> shift, y >> shift)
}

/// Every heightmap whose texels a tile's vertices and normals read, or that
/// a missing one clamps into: the home one first, then the neighbours within
/// [`MARGIN_TEXELS`] of its edges.
pub fn heightmaps_for(world: &WorldConfig, key: TileKey) -> Vec<(i32, i32)> {
    let home = home_heightmap(world, key);
    let (x, y) = world.provider_tile(key);
    let s = scale_to_elevation(world, key.zoom);
    let n = TILE_PIXELS as f64;
    let span = |t: i32| {
        let low = (f64::from(t) * s * n - MARGIN_TEXELS).floor() / n;
        let high = (f64::from(t + 1) * s * n + MARGIN_TEXELS).floor() / n;
        (low.floor() as i32, high.floor() as i32)
    };
    let (x0, x1) = span(x);
    let (y0, y1) = span(y);
    let mut out = vec![home];
    for hy in y0..=y1 {
        for hx in x0..=x1 {
            if (hx, hy) != home {
                out.push((hx, hy));
            }
        }
    }
    out
}

/// Mesh cells along the side of a tile at `zoom`: [`CELLS_PER_TEXEL`] per
/// elevation texel it spans, within `[min_segments, max_segments]`.
pub fn segments(world: &WorldConfig, zoom: u8) -> u32 {
    let finer = u32::from(zoom - world.elevation_zoom_for(zoom));
    let texels = (TILE_PIXELS as u32).checked_shr(finer).unwrap_or(0);
    (texels * CELLS_PER_TEXEL).clamp(
        world.min_segments,
        world.max_segments.max(world.min_segments),
    )
}

/// Provider tiles at the elevation zoom per provider tile at `zoom`.
fn scale_to_elevation(world: &WorldConfig, zoom: u8) -> f64 {
    2f64.powi(i32::from(world.elevation_zoom_for(zoom)) - i32::from(zoom))
}

/// A tile's terrain on the CPU: a `(segments + 1)²` grid of heights and
/// normals, row by row from the north-west corner (`u` east, `v` south),
/// each cell two triangles split along its north-east to south-west
/// diagonal. Heights are absolute meters (times `height_scale`).
#[derive(Clone, Debug)]
pub struct TileGeometry {
    /// The tile.
    pub key: TileKey,
    /// World size of the tile (meters).
    pub size: f64,
    /// Cells along each side.
    pub segments: u32,
    /// Vertex heights, `(segments + 1)²`.
    pub heights: Vec<f32>,
    /// Vertex normals (absolute frame), from the height function's slope.
    pub normals: Vec<[f32; 3]>,
}

impl TileGeometry {
    /// Samples the height function over a tile. `elevation` should hold the
    /// heightmaps of [`heightmaps_for`], the home one at least; neighbours it
    /// lacks are clamped.
    pub fn build(world: &WorldConfig, key: TileKey, elevation: &Elevation) -> Self {
        let segments = segments(world, key.zoom);
        let (x, y) = world.provider_tile(key);
        let s = scale_to_elevation(world, key.zoom);
        let scale = f64::from(world.height_scale);
        let step = SLOPE_STEP_TEXELS / TILE_PIXELS as f64;
        let run = 2.0 * step * world.zoom_size(elevation.zoom);
        let side = segments as usize + 1;
        let mut heights = Vec::with_capacity(side * side);
        let mut normals = Vec::with_capacity(side * side);
        for j in 0..side {
            let fy = (f64::from(y) + j as f64 / f64::from(segments)) * s;
            for i in 0..side {
                let fx = (f64::from(x) + i as f64 / f64::from(segments)) * s;
                let h = |fx: f64, fy: f64| elevation.metres(fx, fy) * scale;
                heights.push(h(fx, fy) as f32);
                let dx = (h(fx + step, fy) - h(fx - step, fy)) / run;
                let dz = (h(fx, fy + step) - h(fx, fy - step)) / run;
                let len = (dx * dx + 1.0 + dz * dz).sqrt();
                normals.push([(-dx / len) as f32, (1.0 / len) as f32, (-dz / len) as f32]);
            }
        }
        Self {
            key,
            size: world.zoom_size(key.zoom),
            segments,
            heights,
            normals,
        }
    }

    /// Vertices along a side.
    pub fn side(&self) -> usize {
        self.segments as usize + 1
    }

    /// Vertex positions about the tile's centre (x east, z south).
    pub fn local_positions(&self) -> Vec<[f32; 3]> {
        let side = self.side();
        let n = f64::from(self.segments);
        (0..side * side)
            .map(|k| {
                let (i, j) = ((k % side) as f64, (k / side) as f64);
                [
                    ((i / n - 0.5) * self.size) as f32,
                    self.heights[k],
                    ((j / n - 0.5) * self.size) as f32,
                ]
            })
            .collect()
    }

    /// Vertex positions in absolute space, in f64: what a consumer far from
    /// the anchor transforms to its own frame without losing precision.
    pub fn absolute_positions(&self) -> Vec<[f64; 3]> {
        let side = self.side();
        let n = f64::from(self.segments);
        let (x, z) = (f64::from(self.key.x), f64::from(self.key.z));
        (0..side * side)
            .map(|k| {
                let (i, j) = ((k % side) as f64, (k / side) as f64);
                [
                    (x + i / n) * self.size,
                    f64::from(self.heights[k]),
                    (z + j / n) * self.size,
                ]
            })
            .collect()
    }

    /// The tile's centre in absolute space (x, z).
    pub fn centre(&self) -> (f64, f64) {
        (
            (f64::from(self.key.x) + 0.5) * self.size,
            (f64::from(self.key.z) + 0.5) * self.size,
        )
    }

    /// Texture coordinates over the tile, `[0, 1]²` from the north-west.
    pub fn uvs(&self) -> Vec<[f32; 2]> {
        let side = self.side();
        let n = self.segments as f32;
        (0..side * side)
            .map(|k| [(k % side) as f32 / n, (k / side) as f32 / n])
            .collect()
    }

    /// Two triangles per cell, wound counter-clockwise seen from above.
    pub fn indices(&self) -> Vec<u32> {
        let side = self.side() as u32;
        let mut out = Vec::with_capacity(self.segments as usize * self.segments as usize * 6);
        for j in 0..self.segments {
            for i in 0..self.segments {
                let a = j * side + i;
                let (b, c) = (a + 1, a + side);
                out.extend_from_slice(&[a, c, b, b, c, c + 1]);
            }
        }
        out
    }

    /// The height of the drawn triangles at `(u, v)` in `[0, 1]²` (clamped).
    pub fn height_at(&self, u: f32, v: f32) -> f32 {
        let n = self.segments as f32;
        let fu = u.clamp(0.0, 1.0) * n;
        let fv = v.clamp(0.0, 1.0) * n;
        let (i, j) = (fu.floor().min(n - 1.0), fv.floor().min(n - 1.0));
        let (tu, tv) = (fu - i, fv - j);
        let side = self.side();
        let at = |di: usize, dj: usize| self.heights[(j as usize + dj) * side + i as usize + di];
        let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
        if tu + tv <= 1.0 {
            a + (b - a) * tu + (c - a) * tv
        } else {
            d + (c - d) * (1.0 - tu) + (b - d) * (1.0 - tv)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> WorldConfig {
        WorldConfig {
            elevation_zoom: 10,
            min_segments: 16,
            max_segments: 64,
            ..WorldConfig::from_lat_lon(-27.455, 153.102)
        }
    }

    fn hill(offset: f32) -> Arc<Vec<f32>> {
        Arc::new(
            (0..TILE_PIXELS * TILE_PIXELS)
                .map(|i| {
                    let (x, y) = ((i % 256) as f32, (i / 256) as f32);
                    offset + 0.3 * x + 0.1 * y + 5.0 * (x * 0.2).sin() * (y * 0.15).cos()
                })
                .collect(),
        )
    }

    fn elevation(world: &WorldConfig) -> Elevation {
        let home = home_heightmap(world, world.key_at(14, 0.0, 0.0));
        let mut elevation = Elevation {
            zoom: 10,
            ..Default::default()
        };
        elevation.tiles.insert(home, hill(20.0));
        elevation.tiles.insert((home.0 + 1, home.1), hill(96.7));
        elevation
    }

    #[test]
    fn the_height_function_is_bilinear_and_reads_across_edges() {
        let mut e = Elevation {
            zoom: 10,
            ..Default::default()
        };
        let ramp = |o: f32| Arc::new((0..256 * 256).map(|i| o + (i % 256) as f32).collect());
        e.tiles.insert((947, 593), ramp(0.0));
        e.tiles.insert((948, 593), ramp(256.0));
        let px = 1.0 / 256.0;
        assert!((e.metres(948.0, 593.5) - 255.5).abs() < 1e-9);
        assert!((e.metres(948.0 - 1e-12, 593.5) - 255.5).abs() < 1e-6);
        assert!((e.metres(948.0 + 0.5 * px, 593.5) - 256.0).abs() < 1e-9);
        assert!((e.metres(947.0 + 2.5 * px, 593.5) - 2.0).abs() < 1e-9);
        // a missing neighbour clamps into the point's map: its edge texel
        e.tiles.remove(&(948, 593));
        assert!((e.metres(948.0 - 0.5 * px, 593.5) - 255.0).abs() < 1e-9);
        assert!((e.metres(948.0 + 0.5 * px, 593.5) - 255.0).abs() < 1e-9);
        assert_eq!(e.metres(949.5, 593.5), 0.0);
    }

    #[test]
    fn neighbours_share_their_edges_heights_and_normals() {
        let world = world();
        let elevation = elevation(&world);
        let key = world.key_at(19, 0.0, 0.0);
        let here = TileGeometry::build(&world, key, &elevation);
        let east = TileGeometry::build(
            &world,
            TileKey {
                x: key.x + 1,
                ..key
            },
            &elevation,
        );
        let south = TileGeometry::build(
            &world,
            TileKey {
                z: key.z + 1,
                ..key
            },
            &elevation,
        );
        let n = here.side();
        let s = n - 1;
        for k in 0..n {
            assert_eq!(here.heights[k * n + s], east.heights[k * n]);
            assert_eq!(here.normals[k * n + s], east.normals[k * n]);
            assert_eq!(here.heights[s * n + k], south.heights[k]);
            assert_eq!(here.normals[s * n + k], south.normals[k]);
        }
        let ha = here.absolute_positions();
        let ea = east.absolute_positions();
        for k in 0..n {
            assert_eq!(ha[k * n + s], ea[k * n]);
        }
    }

    #[test]
    fn a_coarser_tile_agrees_with_a_finer_one_at_their_common_vertices() {
        let world = world();
        let elevation = elevation(&world);
        let fine = TileGeometry::build(&world, world.key_at(19, 0.0, 0.0), &elevation);
        let coarse_key = TileKey {
            zoom: 18,
            x: fine.key.x >> 1,
            z: fine.key.z >> 1,
        };
        let coarse = TileGeometry::build(&world, coarse_key, &elevation);
        let (fa, ca) = (fine.absolute_positions(), coarse.absolute_positions());
        let mut common = 0;
        for p in &fa {
            if let Some(q) = ca
                .iter()
                .find(|q| (q[0] - p[0]).abs() < 1e-6 && (q[2] - p[2]).abs() < 1e-6)
            {
                assert!((q[1] - p[1]).abs() < 1e-4, "{p:?} {q:?}");
                common += 1;
            }
        }
        assert!(common >= 9 * 9, "{common}");
    }

    #[test]
    fn segments_follow_the_elevation_texels_within_bounds() {
        let world = world();
        assert_eq!(segments(&world, 19), 16);
        assert_eq!(segments(&world, 16), 16);
        assert_eq!(segments(&world, 15), 32);
        assert_eq!(segments(&world, 14), 64);
        assert_eq!(segments(&world, 9), 64);
    }

    #[test]
    fn triangles_face_up_and_height_at_is_on_them() {
        let world = world();
        let g = TileGeometry::build(&world, world.key_at(17, 0.0, 0.0), &elevation(&world));
        let p = g.local_positions();
        for t in g.indices().as_chunks::<3>().0 {
            let [a, b, c] = [0, 1, 2].map(|k| p[t[k] as usize]);
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            assert!(ab[2] * ac[0] - ab[0] * ac[2] > 0.0);
        }
        let side = g.side();
        assert_eq!(g.height_at(0.0, 0.0), g.heights[0]);
        assert_eq!(g.height_at(1.0, 1.0), g.heights[side * side - 1]);
        let n = g.segments as f32;
        let mid = g.height_at(0.5 / n, 0.5 / n);
        let diagonal = 0.5 * (g.heights[1] + g.heights[side]);
        assert!((mid - diagonal).abs() < 1e-3);
        for normal in &g.normals {
            assert!(normal[1] > 0.5);
        }
    }
}
