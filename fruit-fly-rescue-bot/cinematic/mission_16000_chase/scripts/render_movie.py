"""Validate and resume the 960-frame third-person master."""
import bpy,json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'cinematic/mission_16000_chase'
s=bpy.data.scenes['AFTERLIGHT | 16000'];bpy.context.window.scene=s
r=json.loads((ROOT/'runs/mission_16000/demo_success/replay.json').read_text())
actor=bpy.data.objects['Fruit fly | replay actor']
row=next(n for n in bpy.data.materials['Measured signed neural emission'].node_tree.nodes if n.type=='COMBXYZ')
points=[f['position'] for f in r['frames']]+[r['final_position']]
for step,xy in enumerate(points):
    s.frame_set(48+step*20)
    assert np.allclose(actor.location[:2],xy,atol=1e-5),(step,tuple(actor.location),xy)
    assert abs(row.inputs['Y'].default_value-(min(step,41)+.5)/42)<1e-6,step
s.frame_set(747);assert bpy.data.objects['Recovered beacon payload'].hide_render
s.frame_set(748);assert not bpy.data.objects['Recovered beacon payload'].hide_render
assert len([o for o in s.objects if 'mission_footprint' in o])==16
assert len(bpy.data.objects['Measured neural activity / all positioned neurons'].data.vertices)//6==15619
report={'passed':True,'path_boundaries_checked':43,'activity_states_checked':42,'obstacles':16,
        'visible_somas':15619,'frame_count':960,'resolution':[1920,1080],'seconds':40,
        'fly_relative_scale':.28,'pickup_frame':748,'arrival_frame':888,
        'note':'Recorded mission, cinematic flight and chase camera. Presentation timing slowed; policy unchanged.'}
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('VALIDATION',report,flush=True)
s.render.resolution_percentage=100;s.eevee.taa_render_samples=32
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.render.image_settings.compression=20
started=time.perf_counter()
for f in range(1,961):
    path=OUT/f'frames/frame_{f:04d}.png'
    if path.exists():continue
    s.frame_set(f);s.render.filepath=str(path);bpy.ops.render.render(write_still=True)
    if f%24==0 or f==1:print('PROGRESS',f,round(time.perf_counter()-started,1),flush=True)
(OUT/'render_summary.json').write_text(json.dumps({'frames':960,'seconds_this_pass':time.perf_counter()-started,'samples':32},indent=2)+'\n')
print('RENDER COMPLETE',flush=True)
