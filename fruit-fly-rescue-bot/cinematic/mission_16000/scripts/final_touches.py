import bpy, math
from mathutils import Vector
import build_scene as cb
cb.attach();s=bpy.context.scene
if s.get('final_touches'):raise RuntimeError('Already applied')
for group in [o for o in s.objects if o.name.startswith('Disaster tipped furniture')]:
    for o in group.children:o.location-=group.location
for o in [o for o in s.objects if o.name.startswith('Exposed wall rebar')]:
    x=o.data.splines[0].points[0].co.x
    index=round((x+4.6)/.62)
    if index in [4,5,11]:
        for point in o.data.splines[0].points:point.co.z*=.38
# Lift activity visibility without adding or inventing neural spikes.
m=bpy.data.materials['Measured signed neural emission']
next(n for n in m.node_tree.nodes if n.type=='EMISSION').inputs['Strength'].default_value=4.2
# Add a brief pickup dip, preserving all recorded horizontal positions.
root=bpy.data.objects['Fruit fly | replay actor']
for f in range(440,495):
    s.frame_set(f);root.location.z-=.33*math.exp(-((f-468)/10)**2);root.keyframe_insert('location',frame=f)
cam=s.camera
# Camera-local overlay coordinates at depth .49, matching the foreground HUD.
corners=cam.data.view_frame(scene=s);depth=.49
w=max(abs(v.x/v.z*depth) for v in corners);h=max(abs(v.y/v.z*depth) for v in corners)
for step in range(42):
    o=cb.text(f'Recorded decision {step:02}',f'DECISION {step+1:02d} / 42',((.75*2-1)*w,(.105*2-1)*h,-depth),h*.041,cb.MAT['White'],cam)
    start=1 if step==0 else cb.INTRO+step*cb.MOVE
    stop=cb.END if step==41 else cb.INTRO+(step+1)*cb.MOVE-1
    for f,hidden in [(1,start>1),(max(1,start-1),start>1),(start,False),(stop,False),(stop+1,True)]:
        o.hide_render=hidden;o.hide_viewport=hidden;o.keyframe_insert('hide_render',frame=f);o.keyframe_insert('hide_viewport',frame=f)
    o.visible_shadow=False
# Brief on-screen credit, with full attribution also supplied in the share caption.
o=cb.text('Anatomy credit','ANATOMY: MaleCNS v1.0 / HHMI Janelia et al. / CC BY 4.0',((.041*2-1)*w,(.827*2-1)*h,-depth),h*.022,cb.MAT['White'],cam)
for f,hidden in [(1,True),(551,True),(552,False),(600,False)]:o.hide_render=hidden;o.hide_viewport=hidden;o.keyframe_insert('hide_render',frame=f);o.keyframe_insert('hide_viewport',frame=f)
s['final_touches']=True;s.frame_set(156)
for area in bpy.context.screen.areas:
    if area.type=='VIEW_3D':
        area.spaces.active.camera=cam;area.spaces.active.region_3d.view_perspective='CAMERA'
        area.spaces.active.region_3d.view_camera_zoom=0
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(cb.OUT/'afterlight.blend'))
print('Final geometry, pickup dip, recorded decision counter, and attribution saved.')
