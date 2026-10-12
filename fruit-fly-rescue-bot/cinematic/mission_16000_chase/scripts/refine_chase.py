"""Macro-scale material cleanup and camera composition, after build_chase.py."""
import bpy,sys,math,random
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'cinematic/mission_16000_chase'
sys.path.insert(0,str(ROOT/'cinematic/mission_16000/scripts'))
import build_scene as cb
cb.attach();s=bpy.context.scene;R=random.Random(817)
assert not s.get('chase_refined')
# Generated coordinates were stretched per fragment; use physical world position.
for name in ['Concrete','Pale broken plaster','Brick','Brick pale','Wood','Fabric','Ash']:
    mat=bpy.data.materials[name];nodes=mat.node_tree.nodes
    geom=nodes.new('ShaderNodeNewGeometry')
    for n in list(nodes):
        if n.type=='TEX_NOISE':
            mat.node_tree.links.new(geom.outputs['Position'],n.inputs['Vector'])
            n.inputs['Scale'].default_value=55 if n.inputs['Scale'].default_value>20 else 3.2
        if n.type=='BUMP':
            n.inputs['Strength'].default_value=.25;n.inputs['Distance'].default_value=.012
    p=next(n for n in nodes if n.type=='BSDF_PRINCIPLED');p.inputs['Roughness'].default_value=.87
# Remove pyramid-shaped dressing and replace with fractured plate fragments.
for o in list(s.objects):
    if o.name.startswith(('Scattered fractured rubble','Irregular floor shard')):bpy.data.objects.remove(o,do_unlink=True)
verts=[[],[],[]];faces=[[],[],[]]
for i in range(4300):
    x,y=R.uniform(-4.7,4.7),R.uniform(-4.7,4.7)
    near=any(a-.20<x<c+.20 and b-.20<y<d+.20 for a,b,c,d in cb.REPLAY['obstacles'])
    size=R.uniform(.018,.075)*(1.8 if near else 1)
    height=R.uniform(.008,.038)*(1.7 if near else 1)
    k=i%3;idx=len(verts[k]);angle=R.random()*6.28
    polygon=[]
    for j in range(5):
        a=angle+j*6.28/5;rad=size*R.uniform(.6,1.3)
        polygon.append((x+rad*math.cos(a),y+rad*math.sin(a)))
    verts[k].extend([(a,b,.017) for a,b in polygon])
    verts[k].extend([(a+size*.08,b-size*.06,.022+height*R.uniform(.65,1.3)) for a,b in polygon])
    faces[k].extend([(idx+j,idx+(j+1)%5,idx+(j+1)%5+5,idx+j+5) for j in range(5)])
    faces[k].extend([(idx+5,idx+6,idx+7),(idx+5,idx+7,idx+8),(idx+5,idx+8,idx+9)])
for i,mat in enumerate(['Concrete','Brick pale','Pale broken plaster']):cb.mesh('Fractured flat rubble '+mat,verts[i],faces[i],cb.MAT[mat])
for o in s.objects:
    if o.name.startswith('Airborne dust'):o.scale*=.22
cb.enum(bpy.data.materials['Wing glass'],'surface_render_method','BLENDED')
bpy.data.materials['Wing glass'].use_transparency_overlap=False
p=next(n for n in bpy.data.materials['Wing glass'].node_tree.nodes if n.type=='BSDF_PRINCIPLED')
p.inputs['Metallic'].default_value=.05;p.inputs['Alpha'].default_value=.20
bpy.data.lights['Home cyan light'].energy=1.6
# Smooth the outer wing silhouette while retaining the authored vein network.
for o in [o for o in s.objects if o.name.startswith('Transparent veined wing')]:
    original=[v.co.copy() for v in o.data.vertices];coords=[]
    for i in range(len(original)):
        p0,p1,p2,p3=[original[j%len(original)] for j in [i-1,i,i+1,i+2]]
        for j in range(6):
            t=j/6
            coords.append(.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t))
    o.data.clear_geometry();o.data.from_pydata(coords,[],[tuple(range(len(coords)))]);o.data.update()
    border=next(ch for ch in o.parent.children if ch.name.startswith('Wing membrane outline'))
    border.data.splines.clear();sp=border.data.splines.new('POLY');sp.points.add(len(coords))
    for p,c in zip(sp.points,coords+[coords[0]]):p.co=(*c,1)
# Darker recesses, a warmer overhead key, and enough soft fill to read the fly.
next(n for n in s.world.node_tree.nodes if n.type=='BACKGROUND').inputs['Strength'].default_value=.075
bpy.data.lights['Cool sky fill'].energy=110
bpy.data.lights['Front softbox'].energy=60
bpy.data.lights['Golden hour breach'].energy=1450
bpy.data.lights['Camera soft bounce'].energy=4
bpy.data.lights['Sun through collapsed roof'].energy=2.8
s.view_settings.exposure=-.30
cam=s.camera;actor=bpy.data.objects['Fruit fly | replay actor']
for f in range(1,961):
    s.frame_set(f)
    forward=actor.location-cam.location;forward.z=0;forward.normalize()
    aim=actor.location+forward*.08+Vector((0,0,-.025))
    q=(aim-cam.location).to_track_quat('-Z','Y')
    aim+=q@Vector((.05,0,0))
    cam.rotation_euler=(aim-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.keyframe_insert('rotation_euler',frame=f)
s['chase_refined']=True;s.frame_set(140)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'afterlight_chase.blend'))
print('Macro materials, wing silhouette, shadow contrast and framing refined')
