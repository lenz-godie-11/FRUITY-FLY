"""Check close chase-camera visibility against actual static ruin geometry."""
import bpy,json
from pathlib import Path
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
OUT=Path(__file__).resolve().parents[1]
s=bpy.data.scenes['AFTERLIGHT | 16000'];bpy.context.window.scene=s
s.frame_set(1);deps=bpy.context.evaluated_depsgraph_get()
cam=s.camera;actor=bpy.data.objects['Fruit fly | replay actor']
verts=[];faces=[]
for o in s.objects:
    ancestor=o;skip=False
    while ancestor:
        if ancestor in [actor,cam]:skip=True;break
        ancestor=ancestor.parent
    if skip or o.type!='MESH' or o.hide_render or 'dust' in o.name.lower() or o.name.startswith('Airborne'):continue
    ev=o.evaluated_get(deps);mesh=ev.to_mesh();offset=len(verts)
    verts.extend([ev.matrix_world@v.co for v in mesh.vertices])
    faces.extend([tuple(offset+i for i in p.vertices) for p in mesh.polygons])
    ev.to_mesh_clear()
bvh=BVHTree.FromPolygons(verts,faces)
occluded=[];projected=[]
for f in range(1,961):
    s.frame_set(f)
    delta=cam.location-actor.location
    hit=bvh.ray_cast(actor.location,delta.normalized(),delta.length-.04)
    if hit[0] is not None:occluded.append({'frame':f,'distance_from_fly':hit[3]})
    p=world_to_camera_view(s,cam,actor.location);projected.append([p.x,p.y])
report={'frames_checked':960,'camera_to_fly_occlusions':occluded,
        'subject_screen_bounds':{'x':[min(p[0] for p in projected),max(p[0] for p in projected)],
                                 'y':[min(p[1] for p in projected),max(p[1] for p in projected)]}}
(OUT/'camera_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
