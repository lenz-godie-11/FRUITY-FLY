"""Reframe the existing data-driven scene as an immersive third-person replay.
Run against the original afterlight.blend. Never overwrite that source.
"""
import bpy, sys, math, random, json
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'cinematic/mission_16000_chase'
sys.path.insert(0,str(ROOT/'cinematic/mission_16000/scripts'))
import build_scene as cb
cb.attach(); s=bpy.context.scene
assert not s.get('chase_version'), 'Load the original scene to rebuild'
OUT.mkdir(exist_ok=True);(OUT/'frames').mkdir(exist_ok=True)
R=random.Random(8317)
actor=bpy.data.objects['Fruit fly | replay actor'];cam=s.camera

def retime(t):
    if t<=48:return t
    if t<=552:return 48+(t-48)*20/12
    return 888+(t-552)*1.5

# Retime all recorded state, visibility, progress and animated geometry together.
for action in bpy.data.actions:
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    for kp in fc.keyframe_points:
                        kp.co.x=retime(kp.co.x)
                        kp.handle_left.x=retime(kp.handle_left.x)
                        kp.handle_right.x=retime(kp.handle_right.x)
for marker in s.timeline_markers:marker.frame=round(retime(marker.frame))
s.frame_end=960
actor.scale=(.28,)*3
cam.animation_data_clear()
# Keep the original lens so the camera-mounted neural display retains its layout.
cam.data.lens=40;cam.data.clip_start=.025
points=[Vector((*f['position'],0)) for f in cb.REPLAY['frames']]+[Vector((*cb.REPLAY['final_position'],0))]
obstacles=cb.REPLAY['obstacles']

def intersects(a,b,pad=.045):
    for i in range(1,25):
        p=a.lerp(b,i/24)
        if any(x0-pad<p.x<x1+pad and y0-pad<p.y<y1+pad for x0,y0,x1,y1 in obstacles):return True
    return False

last_angle=0.;camera_angle=0.;min_boom=9.;camera_positions=[]
for frame in range(1,961):
    progress=max(0,min(42,(frame-48)/20))
    step=min(41,int(progress));u=min(1,progress-step);u=u*u*(3-2*u)
    xy=points[step].lerp(points[step+1],u)
    delta=points[step+1]-points[step]
    angle=math.atan2(-delta.x,delta.y)
    while angle-last_angle>math.pi:angle-=2*math.pi
    while angle-last_angle<-math.pi:angle+=2*math.pi
    last_angle+=.20*(angle-last_angle)
    camera_angle+=.16*(last_angle-camera_angle)
    z=.52+.014*math.sin(frame*.17)
    z-=.20*math.exp(-((frame-748)/15)**2)
    if frame>888:z-=.25*min((frame-888)/48,1)
    actor.location=(xy.x,xy.y,z)
    actor.rotation_euler=(.025*math.sin(frame*.19),.035*math.sin(frame*.12),last_angle)
    actor.keyframe_insert('location',frame=frame);actor.keyframe_insert('rotation_euler',frame=frame)
    # A compact shoulder boom stays inside the recorded free space.
    forward=Vector((-math.sin(camera_angle),math.cos(camera_angle),0))
    boom=1.10
    while boom>.48 and intersects(xy,xy-forward*boom):boom-=.04
    min_boom=min(min_boom,boom)
    cam.location=actor.location-forward*boom+Vector((0,0,.36))
    aim=actor.location+forward*.18+Vector((0,0,.08))
    q=(aim-cam.location).to_track_quat('-Z','Y')
    aim+=q@Vector((.07,0,0))
    cam.rotation_euler=(aim-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.keyframe_insert('location',frame=frame);cam.keyframe_insert('rotation_euler',frame=frame)
    camera_positions.append(tuple(cam.location))
for name,side in [('Wing.L',-1),('Wing.R',1)]:
    wing=bpy.data.objects[name];wing.animation_data_clear()
    for f in range(1,961):
        wing.rotation_euler.y=side*.37*math.sin(2*math.pi*f/3.7)
        wing.keyframe_insert('rotation_euler',frame=f)

# Less dominant interface: preserve neural data and top-right placement.
for o in list(cam.children):
    if o.name.startswith('Brain'):
        pivot=Vector((.212,.117,o.location.z))
        o.location=pivot+(o.location-pivot)*.77;o.scale*=.77
    elif o.name in ['Brand','Subtitle','Anatomy credit']:
        o.scale*=.68
bpy.data.objects['Subtitle'].data.body='INTO THE RUINS / A CONNECTOME RESCUE'

# Towering fracture silhouettes at the same blocked footprints.
for o in list(s.objects):
    if o.name.startswith(('Exposed masonry','Collapsed concrete slab','Bent exposed reinforcement')):
        if o.type=='CURVE':
            for spl in o.data.splines:
                for p in spl.points:p.co.z*=1.9
        else:o.location.z*=1.9;o.scale.z*=1.9

def fractured_wall(name,x,y,width,height,depth,rotation=0):
    xs=[-width/2,-width*.29,-width*.09,width*.12,width*.33,width/2]
    heights=[height*R.uniform(.65,1.12) for _ in xs]
    front=[(xx,-depth/2,0) for xx in xs]+[(xx,-depth/2,h) for xx,h in zip(xs,heights)]
    verts=front+[(xx,depth/2,z) for xx,_,z in front]
    faces=[];n=len(xs)
    for i in range(n-1):
        faces.extend([(i,i+1,i+1+n,i+n),(i+2*n,i+3*n,i+1+3*n,i+1+2*n),
                      (i+n,i+n+1,i+3*n+1,i+3*n),(i,i+2*n,i+1+2*n,i+1)])
    faces.extend([(0,n,3*n,2*n),(n-1,2*n-1,4*n-1,3*n-1)])
    o=cb.mesh(name,verts,faces,cb.MAT['Pale broken plaster']);o.location=(x,y,.02);o.rotation_euler.z=rotation
    bevel=o.modifiers.new('Crumbled edges','BEVEL');bevel.width=.014;bevel.segments=2
    for xx,h in zip(xs[1:-1],heights[1:-1]):
        pts=[(xx,0,h-.15),(xx+.025,0,h+.20),(xx+.10,.03,h+.30)]
        bar=cb.line('Torn reinforcing steel',pts,.009,cb.MAT['Rebar'],o)
    return o

for i,(x0,y0,x1,y1) in enumerate(obstacles):
    x=(x0+x1)/2;y=(y0+y1)/2
    if i%4 in (0,1):
        fractured_wall('Interior wall fracture',x,y,.78,R.uniform(1.5,2.35),.17,0 if i%2 else math.pi/2)
    if i%4 in (2,3):
        # Furniture is retained as a familiar human-scale cue.
        cb.box('Broken suspended joist',(x,y,2.35),(.19,.72,.15),cb.MAT['Wood'],rot=(.4,.17,.12))

# The original cutaway becomes a ruined enclosure with tall piers and openings.
for side in range(4):
    for k in range(9):
        t=-4.4+k*1.1
        x,y,rot=[(t,-4.75,0),(4.75,t,math.pi/2),(t,4.86,0),(-4.86,t,math.pi/2)][side]
        if side==2:continue
        fractured_wall('Ragged outer building wall',x,y,1.08,R.uniform(2.5,3.9),.22,rot)
    for t in [-4.45,0,4.45]:
        x,y=[(t,-4.82),(4.82,t),(t,4.94),(-4.94,t)][side]
        cb.box('Concrete structural column',(x,y,1.9),(.27,.28,3.8),cb.MAT['Concrete'],bevel=.025)
for x in [-4.5,-1.8,1.8,4.5]:
    cb.box('Overhead torn roof beam',(x,0,3.55),(.21,9.5,.29),cb.MAT['Concrete'],rot=(0,.02,.01),bevel=.015)
    for y in [-3.5,-.3,2.8]:
        cb.box('Ceiling fragment',(x+.35,y,3.67),(.9,R.uniform(.7,1.8),.14),cb.MAT['Pale broken plaster'],rot=(0,R.uniform(-.12,.12),.03),bevel=.025)
        cb.line('Dangling electrical cable',[(x,y,3.55),(x+.08,y,2.9),(x-.10,y+.03,2.35)],.007,cb.MAT['Rebar'])

# Thousands of irregular rubble chips, batched into just three meshes.
verts=[[],[],[]];faces=[[],[],[]]
for i in range(3600):
    x,y=R.uniform(-4.7,4.7),R.uniform(-4.7,4.7)
    near=any(x0-.22<x<x1+.22 and y0-.22<y<y1+.22 for x0,y0,x1,y1 in obstacles)
    size=R.uniform(.018,.075)*(1.7 if near else 1)
    height=R.uniform(.015,.06)*(1.7 if near else 1)
    bank=i%3;start=len(verts[bank]);a=R.random()*6.28
    for j in range(5):
        t=a+j*2*math.pi/5;rad=size*R.uniform(.6,1.3)
        verts[bank].append((x+rad*math.cos(t),y+rad*math.sin(t),.018+R.uniform(0,.014)))
    verts[bank].append((x+size*.2,y-size*.1,height+.02))
    faces[bank].extend([(start+j,start+(j+1)%5,start+5) for j in range(5)])
for i,mat in enumerate(['Concrete','Brick pale','Pale broken plaster']):cb.mesh('Scattered fractured rubble '+mat,verts[i],faces[i],cb.MAT[mat])

# More delicate wings and a rougher body at macro viewing distance.
for name in ['Fly thorax','Fly amber']:
    p=next(n for n in cb.MAT[name].node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    p.inputs['Roughness'].default_value=.62;p.inputs['Metallic'].default_value=0
for side in [-1,1]:
    wing=bpy.data.objects['Wing.L' if side<0 else 'Wing.R']
    for k in range(13):
        t=k/14
        cb.line('Fine wing venation',[(side*(.12+.31*t),-.04-.13*t,.016),(side*(.18+.34*t),-.21-.18*t,.012)],.0011,cb.MAT['Wing veins'],wing)
for i in range(180):
    a=R.random()*6.28;y=R.uniform(-.14,.12)
    p=(.153*math.cos(a),y,.144*math.sin(a))
    cb.line('Fine thorax hair',[p,(p[0]*1.12,y-.018,p[2]*1.13)],.0009,cb.MAT['Fly dark'],actor)

# Atmospheric depth, shafts through the roof, and a restrained following fill.
bg=next(n for n in s.world.node_tree.nodes if n.type=='BACKGROUND')
bg.inputs['Strength'].default_value=.20
bpy.data.lights['Fly tracking glint'].energy=.13
bpy.data.lights['Cool sky fill'].energy=420
bpy.data.lights['Golden hour breach'].energy=1800
bpy.data.lights['Front softbox'].energy=150
bpy.data.lights['Sun through collapsed roof'].energy=2.2
fill=cb.light('Camera soft bounce','AREA',(0,0,0),(1,.83,.61),12,.6)
fill.parent=cam;fill.location=(0,.32,.1);fill.rotation_euler=(0,0,0)
fog=bpy.data.materials.new('Fine suspended plaster dust');fog.use_nodes=True
n=fog.node_tree.nodes;n.clear();vol=n.new('ShaderNodeVolumePrincipled');vol.inputs['Density'].default_value=.028
vol.inputs['Color'].default_value=(.61,.66,.7,1);vol.inputs['Anisotropy'].default_value=.38
out=n.new('ShaderNodeOutputMaterial');fog.node_tree.links.new(vol.outputs['Volume'],out.inputs['Volume'])
cb.box('Interior dust atmosphere',(0,0,2.3),(10.4,10.4,4.8),fog)
for i in range(260):
    cb.uvball('Sunlit fine dust',(R.uniform(-4.6,4.6),R.uniform(-4.6,4.6),R.uniform(.3,3.1)),(.0025,)*3,cb.MAT['Muted'],segments=4,rings=3)
s.view_settings.exposure=-.15;s.eevee.taa_render_samples=32
s['chase_version']=True;s['fly_scale_vs_original']=.28;s['camera_min_boom']=min_boom
s['presentation']='Low third-person chase. Recorded XY and neural rows unchanged. 40 seconds.'
s.frame_set(140);s.render.filepath=str(OUT/'frames/frame_')
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'afterlight_chase.blend'))
(OUT/'camera_positions.json').write_text(json.dumps(camera_positions))
print('CHASE SCENE READY',len(s.objects),'objects; minimum camera boom',min_boom)
