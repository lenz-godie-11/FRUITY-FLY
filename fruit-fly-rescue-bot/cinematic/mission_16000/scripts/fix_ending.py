"""Keep the fly centered as it descends during the closing camera move."""
import bpy
import numpy as np
from mathutils import Vector
import build_scene as cb
cb.attach();s=bpy.context.scene;cam=s.camera;root=bpy.data.objects['Fruit fly | replay actor']
if s.get('ending_fixed'):raise RuntimeError('Already applied')
for f in range(553,601):
    s.frame_set(f);blend=(f-552)/48
    oldfocus=np.array([root.location.x*.75,root.location.y*.75,.72])
    newfocus=oldfocus*(1-blend)+np.array(root.location)*blend
    cam.location+=Vector(newfocus-oldfocus)
    q=(Vector(newfocus)-cam.location).to_track_quat('-Z','Y')
    aim=Vector(newfocus)+q@Vector((.60,0,0))
    cam.rotation_euler=(aim-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.keyframe_insert('location',frame=f);cam.keyframe_insert('rotation_euler',frame=f)
s['ending_fixed']=True;s.frame_set(600)
bpy.ops.wm.save_as_mainfile(filepath=str(cb.OUT/'afterlight.blend'))
print('Closing shot reframed around the descending fly; movement and neural data unchanged.')
