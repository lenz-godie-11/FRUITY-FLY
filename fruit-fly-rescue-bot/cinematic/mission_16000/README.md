# AFTERLIGHT — a connectome rescue replay

A 25-second landscape cinematic built in Blender 5.2 through Blender MCP.
A stylized fruit fly searches a ruined building, recovers a beacon, and returns
home. A transparent brain inset in the top-right displays the recorded neural
activity throughout the mission.

## Deliverables

- `afterlight.mp4` — 1920×1080, 24 fps, H.264, original stereo atmosphere and event cues.
- `afterlight_silent.mp4` — the same video without audio, ready for your own music.
- `afterlight.blend` — editable scene, camera, fly, ruins, lighting, HUD, and animation.
- `soundtrack.wav` — separate original sound-design stem.
- `frames/` — all 600 full-resolution rendered frames, also usable for editing.
- `provenance.json`, `validation.json`, and `delivery.json` — data sources and verification.

In Blender, select the **AFTERLIGHT | 16000** scene, use the cinematic camera
(Numpad 0), and play frames 1–600. Final rendering uses Eevee. The original
startup scene is retained separately. The `.blend` contains packed fonts and
a packed neural-activity texture; playback does not require Python handlers,
the MCP connection, or access to the original dataset.

## Timing

| Time | Shot / event |
| --- | --- |
| 0–2 s | Close introduction to the fly and its neural activity |
| 2–19.5 s | Search through the recorded obstacle layout |
| 19.5 s | Beacon pickup; the payload attaches to the fly |
| 19.5–23 s | Return to the remembered departure point |
| 23–25 s | Arrival and mission-complete hold |

The camera movement, wings, vertical flight, pickup dip, lighting, fly scale,
and disaster props are cinematic staging. The horizontal route and action-time
neural states come from the saved mission. This is **a cinematic replay of a
grid-trained controller**, not a new flight-physics experiment or camera-based
navigation through the ruins. The fly is deliberately enlarged for legibility.

All 16 mission obstacle locations are retained. Small decorative floor shards
and distant building-shell geometry are visual dressing, not additional training
obstacles. Camera movement and vertical hover are authored independently of the
policy. The final hold displays the last recorded neural decision.

## Brain display and source integrity

The source is `runs/mission_16000/demo_success/replay.json`, seed **200089**:
42 moves, pickup after move 35, successful return, 31/33 traversable cells visited.
The selected success is illustrative. The larger model completed **39/128
unseen missions (30.5%)** and collected 96/128 beacons in the independent test.
See the main project README for the complete experiment and memory controls.

The model has 16,000 neurons. The inset plots the **15,619 neurons with measured
soma coordinates**; the 381 missing locations are omitted. The transparent shell
is a convex envelope of those measured positions, not a segmented neuropil mesh.
No anatomical positions or neural spikes were invented.

Cyan and amber denote positive and negative **simulated signed rate activity**,
not biological firing measurements or neurotransmitter identity. Brightness
uses the same fixed log scale across the clip:
`log1p(100 * abs(activity)) / log(101)`, followed by display emission and scene
color management. A packed 42-row texture atlas supplies one recorded activity
state per decision. Its playback is stepped, synchronized to the corresponding
movement; it does not generate random flickering.

Validation checks every one of the 43 horizontal route endpoints, every recorded
neural row, the pickup visibility transition, the 16 obstacle records, and the
count of displayed somas before rendering. Final encoding checks all 600 frame
sizes and decodes the whole finished MP4 to detect truncated or corrupt output.

## Reproduce

Run from the repository root. Python preparation and encoding use the existing
project venv. Blender scene creation and rendering use Blender's bundled Python.

```bash
venv/bin/python cinematic/mission_16000/scripts/prepare_assets.py

# Build in a fresh Blender session via MCP / Blender's Python console:
# import sys
# sys.path.insert(0, '/absolute/path/to/FruitFlyBrain/cinematic/mission_16000/scripts')
# import build_scene as cb
# cb.bootstrap(); cb.environment(); cb.fly(); cb.camera_and_animation(); cb.hud(); cb.finalize()
# exec(compile(open(str(cb.OUT/'scripts/polish_scene.py')).read(), 'polish_scene.py', 'exec'))
# exec(compile(open(str(cb.OUT/'scripts/final_touches.py')).read(), 'final_touches.py', 'exec'))
# exec(compile(open(str(cb.OUT/'scripts/fix_ending.py')).read(), 'fix_ending.py', 'exec'))

# Resume the PNG render; existing frames are retained.
/Applications/Blender.app/Contents/MacOS/Blender -b \
  cinematic/mission_16000/afterlight.blend \
  --python cinematic/mission_16000/scripts/render_movie.py

venv/bin/python cinematic/mission_16000/scripts/make_soundtrack.py
venv/bin/python cinematic/mission_16000/scripts/encode_movie.py
```

Only `imageio-ffmpeg` was added, inside `venv`, for local MP4 encoding. Its version
is recorded in the project requirements lock. No global Python packages or
external model/texture accounts are required. If you change the scene, render
into a new frame folder or deliberately clear the old frame output first;
resume skips existing frames and must not mix different scene versions.

## Attribution and suggested share caption

Fly, ruin geometry, materials, and sound design were created procedurally for
this project. No stock model, music track, or third-party texture was imported.

Anatomical data: **MaleCNS v1.0**, HHMI Janelia FlyEM, Cambridge, MRC LMB,
Google Research and collaborators, **CC BY 4.0**. Coordinates were selected,
reoriented and scaled for display; activity is from an engineered trained model.
[Official dataset](https://male-cns.janelia.org/download/).
See the repository's `DATA_LICENSE.md` for the full attribution.

Suggested caption:

> AFTERLIGHT: 16,000 neurons, one rescue mission. A trained model using measured
> fruit-fly wiring searches for a hidden beacon and brings it home. This Blender cinematic follows
> an actual successful policy replay, with synchronized simulated neural activity.
> Flight and disaster scenery are visual staging of a grid-based experiment.
> Selected success; independent larger-map result: 39/128 complete missions.
> Anatomy: MaleCNS v1.0, HHMI Janelia et al., CC BY 4.0 —
> https://male-cns.janelia.org/download/
