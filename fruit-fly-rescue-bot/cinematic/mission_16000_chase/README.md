# AFTERLIGHT — Into the Ruins

A new third-person version of the recorded 16,000-neuron rescue mission,
composed using the user's low, rear-following camera reference. The original
25-second version remains in `../mission_16000/`.

## Files

- `afterlight_chase.mp4`: 40 seconds, 1920×1080, 24 fps, H.264/AAC with original sound.
- `afterlight_chase_silent.mp4`: the same edit without sound.
- `afterlight_chase.blend`: editable Blender scene with packed activity and fonts.
- `frames/`: 960 rendered PNGs.
- `validation.json`: recorded path, neural timing, pickup and neuron-count checks.
- `camera_validation.json`: camera-to-subject visibility checks against ruin geometry.
- `delivery.json`: final frame count, duration, hash and full decode check.

## Visual changes

The fly is 28% of its former linear size. A low camera follows behind its
heading, with a short camera boom that contracts near recorded blocked cells.
The framing places the fly near the center, below a smaller top-right neural
display. Taller interior wall fragments, a surrounding damaged building shell,
structural columns, broken roof beams and hanging cables establish room scale.
The floor has thousands of irregular plaster and masonry fragments. Subtle
volumetric dust, directional light, finer surface relief, rounded wings and
additional body hairs support the closer viewpoint.

The scene remains a stylized procedural reconstruction, not a photoreal scan.
The reference supplies camera and scale direction; it is not used as a flat
background or as generated footage.

## Mission integrity

The source remains `runs/mission_16000/demo_success/replay.json`, seed 200089.
All 43 horizontal route endpoints and 42 neural states are retained. Presentation
time is slowed to 20 video frames per decision. No model retraining or new
mission result is implied.

- Search begins at frame 48 (2 seconds).
- Pickup occurs at frame 748 (about 31.17 seconds).
- Return is complete at frame 888 (37 seconds).
- The ending holds through frame 960 (40 seconds).

The top-right inset shows 15,619 neurons with measured soma coordinates; the
381 without coordinates remain omitted. Cyan/amber show positive/negative
simulated signed rates, using the same fixed brightness mapping as the original.
The transparent envelope is derived from soma positions, not a segmented brain
surface. Flight height, camera, architecture and decorative debris are staging.
The original 16 obstacle footprints remain unchanged.

This is a selected successful replay. The independent larger-map evaluation
completed 39/128 missions (30.5%). Full source attribution and activity-mapping
details are in `../mission_16000/README.md` and the repository `DATA_LICENSE.md`.
Anatomy: MaleCNS v1.0, HHMI Janelia and collaborators, CC BY 4.0.

## Rebuild and render

Open the original `../mission_16000/afterlight.blend`, then execute
`scripts/build_chase.py` followed by `scripts/refine_chase.py` in Blender.
Both scripts save into this new version directory. In the Blender Python console,
execute with a `__file__` value, for example:

```python
p = '/absolute/path/to/FruitFlyBrain/cinematic/mission_16000_chase/scripts/build_chase.py'
exec(compile(open(p).read(), p, 'exec'), {'__file__': p, '__name__': '__main__'})
```

From the repository root:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
  cinematic/mission_16000_chase/afterlight_chase.blend \
  --python cinematic/mission_16000_chase/scripts/render_movie.py
venv/bin/python cinematic/mission_16000_chase/scripts/make_soundtrack.py
venv/bin/python cinematic/mission_16000_chase/scripts/encode_movie.py
```

Rendering resumes by skipping existing frames. For scene changes, use a fresh
frame directory or deliberately remove affected generated frames before resuming.
No new packages were installed for this revision; encoding uses the project venv.
