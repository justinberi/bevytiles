<div align="center">
    <img src="assets/icon.png" alt="logo" width="150"/>
    <br />
    <strong>3D geospatial engine for Bevy</strong>
    <br />
    <br />
</div>


![GitHub Release](https://badgen.net/github/release/ziv/bevytiles)
![GitHub License](https://badgen.net/github/license/ziv/bevytiles)


[![Linux Build](https://github.com/ziv/bevytiles/actions/workflows/linux.yml/badge.svg)](https://github.com/ziv/bevytiles/actions/workflows/linux.yml)
[![macOS Build](https://github.com/ziv/bevytiles/actions/workflows/macos.yml/badge.svg)](https://github.com/ziv/bevytiles/actions/workflows/macos.yml)
[![Windows Build](https://github.com/ziv/bevytiles/actions/workflows/windows.yml/badge.svg)](https://github.com/ziv/bevytiles/actions/workflows/windows.yml)


**Bevytiles** is a 3D geospatial engine 🌎 for [Bevy](https://bevy.org/). Designed to stream and render the real
world in real time. It lets you visualize any location on Earth directly inside your Bevy games and applications.

Built for indie developers and professionals alike, Bevytiles is a perfect fit for UAV simulations, flight-planning
software, lightweight GIS analysis, presentations, digital sand tables, and any other geospatial visualization.

It provides precise, ground-truth altitude data, essential for accurate collision detection and spawning mechanics in
games, as well as for topographical analysis in GIS workflows.

Originally developed to power a flight simulator, Bevytiles was extracted into a lightweight, standalone library so it can be embedded seamlessly into any Bevy project.

## Demo

https://github.com/user-attachments/assets/c17c0c8b-c578-4573-8d11-23050c7aa4ba



## This fork

A fork of [ziv/bevytiles](https://github.com/ziv/bevytiles) 0.1 for a vehicle simulator, whose
physics colliders and depth sensors must see the terrain that is drawn:

- **Bevy 0.20**, ureq 3. The renderer is behind the `render` feature (default); without it the
  crate is the LOD policy, the tile source and cache, and the CPU geometry, for headless use.
- **Terrain meshed on the CPU** (`geometry`), not displaced in a vertex shader: every tile's
  heights come from the Terrarium heightmaps of one zoom (`WorldConfig::elevation_zoom`), bilinear
  between texel centres and read across heightmap edges, so neighbouring tiles of any zoom meet
  exactly; the normals come from that function's slope, so the lighting has no seam either.
  `TileGeometry` gives the drawn vertices (local or absolute f64) for colliders and sensors, and
  `ground_height` answers on the drawn triangles. Skirts (`RenderingConfig::skirt_cells`) hide the
  cracks where zoom levels meet, in place of the overlap factor.
- **Bevy's lighting**: tiles are `StandardMaterial`s with their (mipmapped, anisotropic) imagery,
  so lights, shadows and `DistanceFog` are the app's. The normal-map tiles and the custom shader
  are gone, as is heightmap synthesis above the native zoom (finer tiles sample the elevation zoom).
- **Rings** (`LodPolicy::Rings`): fixed, overlapping squares of tiles per zoom around the camera,
  each coarser one drawn lower, as a ground vehicle wants; `lod::rings_within` lists what a preload
  needs for a vehicle staying within a radius. **Pinned tiles** (`PinnedTiles`): tiles the app
  draws itself are never streamed or covered. **`TerrainRoot`**: tiles spawn under it, to place
  the terrain in another convention's axes. The streaming focus can be any entity.
- **Cache layout and offline**: `NetworkConfig::{texture,heightmap}_cache` path templates (to
  share another program's cache), `offline` to read the cache alone, a blocking `Fetcher` for tiles
  needed now and for preloading.
- The wasm backend and the web demo are removed (not ported).

## Run the demo

```sh
cargo run --release --features demo --example demo
```

Flies over the Grand Canyon. **A/D** roll, **Q/E** yaw, **W/S** pitch, **+/-** throttle, **R** reset after crashing
into the terrain. Tiles cache under `.cache/`; the first run downloads them (Esri imagery serves
JPEG despite the endpoint name — decoding handles both).

## Use as a library

```rust
use bevy::prelude::*;
use bevytiles::prelude::*;

App::new()
.add_plugins(DefaultPlugins)
.insert_resource(WorldConfig::from_lat_lon(46.206889, 9.497194)) // the Dolomites
.add_plugins(TerrainPlugin)
.add_systems(Startup, | mut c: Commands, w: Res<WorldConfig>| {
c.spawn((Camera3d::default(), TerrainCamera,
Transform::from_translation(w.initial_position(5000.0))));
c.spawn(DirectionalLight::default());
})
.run();
```

Configuration mirrors raytiles: `WorldConfig`, `StreamingConfig`, `RenderingConfig`,
`NetworkConfig`, inserted before or during `Startup`. Large worlds use the raytiles rebase
convention via the `TerrainAnchor` resource (`absolute = user − world_offset`); see the demo's
`rebase_large_world`.

## Layout

| module     | role                                                                                                    |
|------------|---------------------------------------------------------------------------------------------------------|
| `lod`      | pure desired-set policy: the quadtree (snapshot-tested — values match the C++ engine exactly), or rings |
| `geometry` | the CPU terrain: one height function per elevation zoom, tile grids, triangle-exact heights            |
| `source`   | tile fetching: worker threads or blocking, HTTP + disk cache, decode + geometry                        |
| `store`    | ECS systems: reconcile / promote / update_desired / status; one entity per tile (`render`)             |
| `synth`    | Terrarium float decode / quadrant upsample / carry-safe encode                                         |
| `height`   | the resident tiles' geometry for `ground_height`                                                       |

## Bevy compatibility

| bevytiles | bevy |
|-----------|------|
| 0.1       | 0.19 |
| 0.2 (fork)| 0.20 |

## Notes

- Data: imagery © Esri; elevation from the Mapzen/AWS terrain tiles. Mind their terms.
- `cargo test --no-default-features` runs the offline suite (lod snapshots, geometry, synthesis
  math, source integration against seeded caches); `cargo test` adds the store's mesh tests.

## License

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)
