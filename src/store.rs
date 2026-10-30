//! The tile store, ECS-native: one entity per resident tile; bookkeeping in
//! resources; frame steps as ordered systems (reconcile → promote →
//! update_desired → status). Each tile is its own CPU mesh (the worker's
//! [`TileGeometry`], skirted) under a `StandardMaterial` with its imagery, so
//! Bevy's visibility, lights, shadows and fog apply as to anything else.

use crate::config::*;
use crate::geometry::TileGeometry;
use crate::height::HeightGrids;
use crate::lod::{self, LodOptions, LodPolicy, TileKey};
use crate::source::{TileDrop, TilePayload, TileRequest, TileSource};
use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use std::collections::{HashMap, HashSet, VecDeque};

/// Marker + data component on every resident tile entity.
#[derive(Component)]
pub struct Tile {
    /// Anchor-relative identity (also the [`TileIndex`] / grid map key).
    pub key: TileKey,
    /// World size of this tile in meters.
    pub size: f32,
    /// Absolute tile center X (f64 — precision far from the anchor; the
    /// user-space transform is `abs + world_offset`, cast to f32 once).
    pub abs_x: f64,
    /// Absolute tile center Z.
    pub abs_z: f64,
}

/// The tile lifecycle bookkeeping (raytiles' `tile_store` maps): which tiles
/// exist, which should exist, which are being fetched.
#[derive(Resource, Default)]
pub struct TileIndex {
    /// Resident tiles → their entities. In lockstep with [`HeightGrids`] and
    /// the spawned [`Tile`] entities.
    pub resident: HashMap<TileKey, Entity>,
    /// What the LOD policy wants resident for the current camera position;
    /// rebuilt by [`update_desired`].
    pub desired: HashSet<TileKey>,
    /// Keys handed to the source and not yet answered (payload or drop).
    pub loading: HashSet<TileKey>,
}

/// The worker-pool [`TileSource`], wrapped as a resource. Created by
/// [`update_desired`] on first use, from the [`NetworkConfig`] and
/// [`WorldConfig`] of that moment.
#[derive(Resource)]
pub struct SourceRes(pub TileSource);

/// Payloads drained from the source but not yet promoted — the per-frame
/// promotion budget consumes from the front; the rest wait here.
#[derive(Resource, Default)]
pub struct PendingPayloads(pub VecDeque<TilePayload>);

/// Absolute camera position at the last desired-set rebuild; the next
/// rebuild triggers after [`StreamingConfig::update_distance`] of travel.
/// Initialized far away so the first frame always rebuilds.
#[derive(Resource)]
pub struct LastDesiredPos(pub Vec3);

impl Default for LastDesiredPos {
    fn default() -> Self {
        Self(Vec3::splat(-9_999_999.9))
    }
}

/// Coverage relations only change on promotion / desired rebuilds; the
/// (comparatively expensive) covered-by eviction checks are skipped while
/// this is clear. Exact, not a heuristic: evictions only remove cover.
#[derive(Resource)]
pub struct CoverageDirty(pub bool);

impl Default for CoverageDirty {
    fn default() -> Self {
        Self(true)
    }
}

/// World size (meters) of one tile at `zoom`: halves per level above base.
pub fn zoom_size(world: &WorldConfig, zoom: u8) -> f64 {
    world.zoom_size(zoom)
}

/// The streaming camera in absolute space: its global position taken into
/// the [`TerrainRoot`]'s frame (if any), less the world offset.
fn camera_absolute(
    cams: &Query<&GlobalTransform, With<TerrainCamera>>,
    roots: &Query<&GlobalTransform, With<TerrainRoot>>,
    anchor: &TerrainAnchor,
) -> Option<Vec3> {
    let cam = cams.iter().next()?.translation();
    let local = match roots.iter().next() {
        Some(root) => root.affine().inverse().transform_point3(cam),
        None => cam,
    };
    Some(local - anchor.world_offset)
}

// ---------------------------------------------------------------------------
// per-frame systems (chained: reconcile → promote → update_desired → status)

/// Evict resident tiles that are no longer needed. Rings: every tile not
/// desired. Quadtree: not desired AND (base-zoom stale | off-screen per last
/// frame's [`ViewVisibility`] | beyond the horizon | covered by a resident
/// parent / all four children). The covered-by rule protects against holes
/// in the ground; its checks are gated on [`CoverageDirty`].
#[allow(clippy::too_many_arguments)] // bevy system: each param is an injected resource
pub fn reconcile(
    mut commands: Commands,
    mut index: ResMut<TileIndex>,
    mut grids: ResMut<HeightGrids>,
    mut coverage_dirty: ResMut<CoverageDirty>,
    world: Res<WorldConfig>,
    streaming: Res<StreamingConfig>,
    anchor: Res<TerrainAnchor>,
    cams: Query<&GlobalTransform, With<TerrainCamera>>,
    roots: Query<&GlobalTransform, With<TerrainRoot>>,
    vis: Query<&ViewVisibility, With<Tile>>,
) {
    let Some(abs) = camera_absolute(&cams, &roots, &anchor) else {
        return;
    };
    let rings = matches!(streaming.policy, LodPolicy::Rings(_));

    let mut evict = Vec::new();
    for (&key, &entity) in index.resident.iter() {
        if index.desired.contains(&key) {
            continue;
        }
        let remove = if rings || key.zoom == world.base_zoom {
            // rings overlap by design; stale base tiles are the horizon
            true
        } else if !vis.get(entity).map(|v| v.get()).unwrap_or(true) {
            // not on screen last frame
            true
        } else if lod::out_of_horizon(abs, zoom_size(&world, key.zoom), key) {
            true
        } else if !coverage_dirty.0 {
            false
        } else {
            // visible and in range: evict only if parent/children cover the
            // area, otherwise keep it to avoid holes in the ground
            covered(&index.resident, &world, key)
        };
        if remove {
            evict.push((key, entity));
        }
    }
    for (key, entity) in evict {
        commands.entity(entity).despawn();
        index.resident.remove(&key);
        grids.0.remove(&key);
    }
    coverage_dirty.0 = false;
}

fn covered(resident: &HashMap<TileKey, Entity>, world: &WorldConfig, key: TileKey) -> bool {
    let has = |zoom: u8, x: i32, z: i32| resident.contains_key(&TileKey { zoom, x, z });
    if key.zoom > world.base_zoom && has(key.zoom - 1, key.x >> 1, key.z >> 1) {
        return true;
    }
    if key.zoom < MAX_ZOOM {
        let (cx, cz) = (key.x * 2, key.z * 2);
        if has(key.zoom + 1, cx, cz)
            && has(key.zoom + 1, cx + 1, cz)
            && has(key.zoom + 1, cx, cz + 1)
            && has(key.zoom + 1, cx + 1, cz + 1)
        {
            return true;
        }
    }
    // grandparent / grandchildren: rare, but happens when zoom levels are
    // skipped by distance-based loading during fast movement
    if key.zoom > world.base_zoom + 1 && has(key.zoom - 2, key.x >> 2, key.z >> 2) {
        return true;
    }
    if key.zoom + 1 < MAX_ZOOM {
        let (cx, cz) = (key.x * 4, key.z * 4);
        return (0..4).all(|ox| (0..4).all(|oz| has(key.zoom + 2, cx + ox, cz + oz)));
    }
    false
}

/// Drain the source (one non-blocking sweep) and promote finished tiles:
/// the skirted mesh, the imagery (sRGB, mipmapped) on a `StandardMaterial`,
/// the geometry into [`HeightGrids`], and the tile entity (a child of the
/// [`TerrainRoot`], if any). Capped at
/// [`StreamingConfig::max_promotions_per_frame`] per frame; the remainder
/// waits in [`PendingPayloads`]. Failed drops wait for the next desired
/// rebuild; cancelled-but-desired-again keys re-request immediately.
#[allow(clippy::too_many_arguments)]
pub fn promote(
    mut commands: Commands,
    mut index: ResMut<TileIndex>,
    mut pending: ResMut<PendingPayloads>,
    mut grids: ResMut<HeightGrids>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut coverage_dirty: ResMut<CoverageDirty>,
    source: Option<Res<SourceRes>>,
    world: Res<WorldConfig>,
    streaming: Res<StreamingConfig>,
    rendering: Res<RenderingConfig>,
    anchor: Res<TerrainAnchor>,
    roots: Query<Entity, With<TerrainRoot>>,
) {
    let Some(source) = source else { return };
    let mut ready = Vec::new();
    let mut dropped = Vec::new();
    source.0.drain(&mut ready, &mut dropped);
    for d in dropped {
        match d {
            TileDrop::Failed(key, reason) => {
                index.loading.remove(&key);
                warn!("tile {key:?} failed: {reason} - dropping until the next desired rebuild");
            }
            TileDrop::Cancelled(key) => {
                index.loading.remove(&key);
                // wanted again by the time the drop arrived — re-request now
                if index.desired.contains(&key) && !index.resident.contains_key(&key) {
                    request_tile(&source.0, &mut index, &world, key);
                }
            }
        }
    }
    pending.0.extend(ready);

    let root = roots.iter().next();
    for _ in 0..streaming.max_promotions_per_frame {
        let Some(payload) = pending.0.pop_front() else {
            break;
        };
        let key = payload.key;
        index.loading.remove(&key);
        if !index.desired.contains(&key) || index.resident.contains_key(&key) {
            continue; // no longer wanted (payload data just drops)
        }
        let material = materials.add(imagery_material(images.add(imagery_image(&payload.albedo))));
        let mesh = meshes.add(tile_mesh(&payload.geometry, &rendering));
        let (abs_x, abs_z) = payload.geometry.centre();
        let mut entity = commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            tile_transform(&payload.geometry, &anchor, streaming.policy.drop(key.zoom)),
            Tile {
                key,
                size: payload.geometry.size as f32,
                abs_x,
                abs_z,
            },
        ));
        if !rendering.cast_shadows {
            entity.insert(NotShadowCaster);
        }
        if let Some(root) = root {
            entity.insert(ChildOf(root));
        }
        index.resident.insert(key, entity.id());
        grids.0.insert(key, payload.geometry);
        coverage_dirty.0 = true; // a new resident can cover parent/children
    }
}

/// Movement-gated desired-set rebuild: runs the pure LOD policy (leaving out
/// [`PinnedTiles`]), cancels loading keys that fell out of the set (once,
/// here — the set only changes in this system), and requests missing keys
/// with their absolute provider coordinates resolved from the anchor.
#[allow(clippy::too_many_arguments)] // bevy system: each param is an injected resource
pub fn update_desired(
    mut commands: Commands,
    mut index: ResMut<TileIndex>,
    mut last: ResMut<LastDesiredPos>,
    mut coverage_dirty: ResMut<CoverageDirty>,
    source: Option<Res<SourceRes>>,
    world: Res<WorldConfig>,
    streaming: Res<StreamingConfig>,
    net: Res<NetworkConfig>,
    anchor: Res<TerrainAnchor>,
    pinned: Res<PinnedTiles>,
    cams: Query<&GlobalTransform, With<TerrainCamera>>,
    roots: Query<&GlobalTransform, With<TerrainRoot>>,
    mut scratch: Local<Vec<TileKey>>,
) {
    let Some(abs) = camera_absolute(&cams, &roots, &anchor) else {
        return;
    };
    let Some(source) = source else {
        assert!(
            world.base_zoom >= MIN_ZOOM
                && world.max_zoom <= MAX_ZOOM
                && world.max_zoom >= world.base_zoom
        );
        commands.insert_resource(SourceRes(TileSource::new(&net, &world)));
        return;
    };
    if abs.distance_squared(last.0) <= streaming.update_distance * streaming.update_distance
        && !pinned.is_changed()
    {
        return;
    }
    last.0 = abs;

    scratch.clear();
    let opts = LodOptions {
        base_zoom: world.base_zoom,
        max_zoom: world.max_zoom,
        base_tile_size: world.tile_size,
        radius: streaming.radius,
        thresholds: streaming.thresholds,
        policy: streaming.policy.clone(),
    };
    lod::desired_tiles(&opts, abs, &pinned.0, &mut scratch);
    index.desired = scratch.iter().copied().collect();
    coverage_dirty.0 = true;

    // cancel once, here — the desired set only changes in this system
    let stale: Vec<TileKey> = index
        .loading
        .iter()
        .filter(|k| !index.desired.contains(k))
        .copied()
        .collect();
    for key in stale {
        source.0.cancel(key);
    }

    let missing: Vec<TileKey> = index
        .desired
        .iter()
        .filter(|k| !index.resident.contains_key(k) && !index.loading.contains(k))
        .copied()
        .collect();
    for key in missing {
        request_tile(&source.0, &mut index, &world, key);
    }
}

fn request_tile(source: &TileSource, index: &mut TileIndex, world: &WorldConfig, key: TileKey) {
    let (x, z) = world.provider_tile(key);
    source.request(TileRequest { key, x, z });
    index.loading.insert(key);
}

/// Initial-loading contract: `loading` flips only once a desired set EXISTS
/// and is fully serviced. This system must run after `update_desired` — the
/// C++ engine shipped a bug from evaluating this before the first rebuild.
pub fn status(mut st: ResMut<TerrainStatus>, index: Res<TileIndex>, pending: Res<PendingPayloads>) {
    st.resident = index.resident.len();
    if index.desired.is_empty() {
        st.progress = 0.0;
        return;
    }
    let have = index
        .desired
        .iter()
        .filter(|k| index.resident.contains_key(k))
        .count();
    st.progress = have as f32 / index.desired.len() as f32;
    if st.loading && index.loading.is_empty() && pending.0.is_empty() {
        st.loading = false;
        info!(
            "initial load complete: {} tiles resident",
            index.resident.len()
        );
    }
}

/// Rebake every tile transform after a large-world rebase.
pub fn rebase(
    anchor: Res<TerrainAnchor>,
    streaming: Res<StreamingConfig>,
    mut tiles: Query<(&Tile, &mut Transform)>,
) {
    if !anchor.is_changed() {
        return;
    }
    for (tile, mut tf) in &mut tiles {
        tf.translation.x = (tile.abs_x + f64::from(anchor.world_offset.x)) as f32;
        tf.translation.y = anchor.world_offset.y - streaming.policy.drop(tile.key.zoom);
        tf.translation.z = (tile.abs_z + f64::from(anchor.world_offset.z)) as f32;
    }
}

// ---------------------------------------------------------------------------
// meshes, images and materials, public so an app draws its pinned tiles as
// the store draws the rest

/// A tile's transform in user space: its centre (`abs + world_offset`,
/// added in f64 before the single f32 cast), `drop` meters down.
pub fn tile_transform(geometry: &TileGeometry, anchor: &TerrainAnchor, drop: f32) -> Transform {
    let (abs_x, abs_z) = geometry.centre();
    Transform::from_xyz(
        (abs_x + f64::from(anchor.world_offset.x)) as f32,
        anchor.world_offset.y - drop,
        (abs_z + f64::from(anchor.world_offset.z)) as f32,
    )
}

/// The tile's mesh about its centre: the geometry's grid, and a skirt hung
/// from its edges ([`RenderingConfig::skirt_cells`] cells deep) facing out,
/// with the edge's normals and UVs so it lights and textures as the ground
/// above it. The skirt lies under the ground, seen only through a crack.
pub fn tile_mesh(geometry: &TileGeometry, rendering: &RenderingConfig) -> Mesh {
    let mut positions = geometry.local_positions();
    let mut normals = geometry.normals.clone();
    let mut uvs = geometry.uvs();
    let mut indices = geometry.indices();
    let depth = rendering.skirt_cells * (geometry.size / f64::from(geometry.segments)) as f32;
    if depth > 0.0 {
        let side = geometry.side() as u32;
        let s = side - 1;
        // the border counter-clockwise seen from above: south edge east,
        // east edge north, north edge west, west edge south
        let border: Vec<u32> = (0..s)
            .map(|i| s * side + i)
            .chain((0..s).map(|j| (s - j) * side + s))
            .chain((0..s).map(|i| s - i))
            .chain((0..s).map(|j| j * side))
            .collect();
        let first = positions.len() as u32;
        for &v in &border {
            let p = positions[v as usize];
            positions.push([p[0], p[1] - depth, p[2]]);
            normals.push(normals[v as usize]);
            uvs.push(uvs[v as usize]);
        }
        let n = border.len() as u32;
        for k in 0..n {
            let (top, next) = (border[k as usize], border[((k + 1) % n) as usize]);
            let (low, low_next) = (first + k, first + (k + 1) % n);
            indices.extend_from_slice(&[top, low, next, next, low, low_next]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// The imagery as a GPU image: sRGB, its whole mip chain, clamped at the
/// tile's edges, trilinear and anisotropic.
pub fn imagery_image(albedo: &image::RgbaImage) -> Image {
    let (data, levels) = mip_chain(albedo);
    let mut img = Image::new(
        Extent3d {
            width: albedo.width(),
            height: albedo.height(),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_descriptor.mip_level_count = levels;
    img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..Default::default()
    });
    img
}

/// The terrain's material: the imagery, fully rough and without specular
/// (the photograph already holds the ground's shading at noon).
pub fn imagery_material(albedo: Handle<Image>) -> StandardMaterial {
    StandardMaterial {
        base_color_texture: Some(albedo),
        perceptual_roughness: 1.0,
        reflectance: 0.0,
        ..Default::default()
    }
}

/// The image and its 2×2 box-filtered mips, concatenated, and their count.
fn mip_chain(img: &image::RgbaImage) -> (Vec<u8>, u32) {
    let (mut w, mut h) = (img.width() as usize, img.height() as usize);
    let mut level = img.as_raw().clone();
    let mut data = level.clone();
    let mut levels = 1;
    while w > 1 && h > 1 && w % 2 == 0 && h % 2 == 0 {
        let (nw, nh) = (w / 2, h / 2);
        let mut next = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let at = |dx: usize, dy: usize| u32::from(level[((2 * y + dy) * w + 2 * x + dx) * 4 + c]);
                    next[(y * nw + x) * 4 + c] = ((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8;
                }
            }
        }
        data.extend_from_slice(&next);
        level = next;
        (w, h) = (nw, nh);
        levels += 1;
    }
    (data, levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_skirt_hangs_from_every_edge_vertex_facing_out() {
        let geometry = TileGeometry {
            key: TileKey { zoom: 9, x: 0, z: 0 },
            size: 4.0,
            segments: 2,
            heights: vec![1.0; 9],
            normals: vec![[0.0, 1.0, 0.0]; 9],
        };
        let mesh = tile_mesh(&geometry, &RenderingConfig { skirt_cells: 0.5, cast_shadows: false });
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(p)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        assert_eq!(p.len(), 9 + 8);
        assert!(p[9..].iter().all(|v| (v[1] - 0.0).abs() < 1e-6));
        let Some(Indices::U32(idx)) = mesh.indices() else { panic!("indices") };
        assert_eq!(idx.len(), 2 * 2 * 6 + 8 * 6);
        for t in idx[24..].chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(p[t[k] as usize]));
            let normal = (b - a).cross(c - a);
            let centre = (a + b + c) / 3.0;
            assert!(normal.dot(Vec3::new(centre.x, 0.0, centre.z)) > 0.0, "{t:?}");
        }
    }

    #[test]
    fn mips_halve_down_to_one_texel() {
        let img = image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]));
        let (data, levels) = mip_chain(&img);
        assert_eq!(levels, 3);
        assert_eq!(data.len(), (16 + 4 + 1) * 4);
        assert_eq!(&data[data.len() - 4..], &[10, 20, 30, 255]);
    }
}
