"""Render or resume the 600-frame 1080p master, using Blender's bundled Python."""
import bpy,json,time,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'cinematic/mission_16000'
s=bpy.data.scenes['AFTERLIGHT | 16000'];bpy.context.window.scene=s
r=json.loads((ROOT/'runs/mission_16000/demo_success/replay.json').read_text())
root=bpy.data.objects['Fruit fly | replay actor']
row=bpy.data.materials['Measured signed neural emission'].node_tree.nodes['Recorded decision row']
# Validate every decision boundary before rendering, including return to base.
points=[f['position'] for f in r['frames']]+[r['final_position']]
for step,xy in enumerate(points):
    s.frame_set(48+step*12)
    assert np.allclose(root.location[:2],xy,atol=1e-5),(step,tuple(root.location),xy)
    expected=(min(step,41)+.5)/42
    assert abs(row.inputs['Y'].default_value-expected)<1e-6
s.frame_set(467);assert bpy.data.objects['Recovered beacon payload'].hide_render
s.frame_set(468);assert not bpy.data.objects['Recovered beacon payload'].hide_render
assert len([o for o in s.objects if 'mission_footprint' in o])==16
visible_neurons=len(bpy.data.objects['Measured neural activity / all positioned neurons'].data.vertices)//6
assert visible_neurons==15619
report={'passed':True,'checked_path_boundaries':43,'checked_activity_states':42,'mission_obstacles':16,
        'positioned_neurons':visible_neurons,'render_frames':600,'fps':24,'resolution':[1920,1080],
        'note':'Final arrival holds the last recorded neural decision. Flight height and wing movement are cinematic.'}
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('VALIDATION',json.dumps(report),flush=True)
s.render.resolution_percentage=100;s.eevee.taa_render_samples=32
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.render.image_settings.compression=20
started=time.perf_counter()
for frame in range(1,601):
    destination=OUT/f'frames/frame_{frame:04d}.png'
    if destination.exists():continue
    s.frame_set(frame);s.render.filepath=str(destination);bpy.ops.render.render(write_still=True)
    if frame%24==0 or frame==1:print('PROGRESS',frame,round(time.perf_counter()-started,1),flush=True)
(OUT/'render_summary.json').write_text(json.dumps({'frames':600,'seconds_this_pass':time.perf_counter()-started,'resolution':[1920,1080],'samples':32},indent=2)+'\n')
print('RENDER COMPLETE',flush=True)
