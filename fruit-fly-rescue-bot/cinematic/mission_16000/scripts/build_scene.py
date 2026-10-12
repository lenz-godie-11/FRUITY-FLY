"""Blender 5.x cinematic reconstruction. Run stages via Blender MCP or --python.

All activity is baked into a packed texture atlas, so playback needs no handlers.
"""
import bpy, math, json, random
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'cinematic/mission_16000'
REPLAY=json.loads((ROOT/'runs/mission_16000/demo_success/replay.json').read_text())
FPS=24; INTRO=48; MOVE=12; END=INTRO+42*MOVE+48
SCENE_NAME='AFTERLIGHT | 16000'
COL=None
R=random.Random(742)
MAT={}

def enum(obj,prop,value):
    options={i.identifier for i in obj.bl_rna.properties[prop].enum_items}
    if value not in options:raise ValueError(f'{prop}: {value} not in {options}')
    setattr(obj,prop,value)

def link(obj):
    COL.objects.link(obj);return obj

def mesh(name,verts,faces,mat=None):
    data=bpy.data.meshes.new(name);data.from_pydata(verts,[],faces);data.update()
    obj=link(bpy.data.objects.new(name,data))
    if mat:data.materials.append(mat)
    return obj

def box(name,loc,scale,mat,rot=(0,0,0),bevel=0):
    v=[(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]
    f=[(0,4,6,2),(1,3,7,5),(0,1,5,4),(2,6,7,3),(0,2,3,1),(4,5,7,6)]
    o=mesh(name,[(x*scale[0]/2,y*scale[1]/2,z*scale[2]/2) for x,y,z in v],f,mat)
    o.location=loc;o.rotation_euler=rot
    if bevel:
        mod=o.modifiers.new('Worn edges','BEVEL');mod.width=bevel;mod.segments=2
    return o

def uvball(name,loc,scale,mat,parent=None,segments=24,rings=12):
    vertices=[];faces=[]
    for j in range(rings+1):
        a=math.pi*j/rings
        for i in range(segments):
            b=2*math.pi*i/segments;vertices.append((math.sin(a)*math.cos(b)*scale[0],math.sin(a)*math.sin(b)*scale[1],math.cos(a)*scale[2]))
    for j in range(rings):
        for i in range(segments):
            a=j*segments+i;b=j*segments+(i+1)%segments;faces.append((a,b,b+segments,a+segments))
    o=mesh(name,vertices,faces,mat);o.location=loc;o.parent=parent
    for p in o.data.polygons:p.use_smooth=True
    return o

def line(name,points,radius,mat,parent=None):
    data=bpy.data.curves.new(name,'CURVE');enum(data,'dimensions','3D');data.bevel_depth=radius;data.bevel_resolution=2
    spl=data.splines.new('POLY');spl.points.add(len(points)-1)
    for p,co in zip(spl.points,points):p.co=(*co,1)
    o=link(bpy.data.objects.new(name,data));data.materials.append(mat);o.parent=parent;return o

def empty(name,loc=(0,0,0),parent=None):
    o=link(bpy.data.objects.new(name,None));o.location=loc;o.parent=parent;return o

def material(name,color,rough=.6,metal=0,emission=0,alpha=1,grit=0):
    m=bpy.data.materials.new(name);m.use_nodes=True
    p=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal
    if emission:p.inputs['Emission Color'].default_value=(*color,1);p.inputs['Emission Strength'].default_value=emission
    if alpha<1:
        p.inputs['Alpha'].default_value=alpha;enum(m,'surface_render_method','DITHERED')
    if grit:
        n=m.node_tree.nodes.new('ShaderNodeTexNoise');n.inputs['Scale'].default_value=35;n.inputs['Detail'].default_value=3
        b=m.node_tree.nodes.new('ShaderNodeBump');b.inputs['Strength'].default_value=.55;b.inputs['Distance'].default_value=grit
        m.node_tree.links.new(n.outputs['Factor'],b.inputs['Height']);m.node_tree.links.new(b.outputs['Normal'],p.inputs['Normal'])
    m.diffuse_color=(*color,alpha);MAT[name]=m;return m

def emission(name,color,strength=1):
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;n.clear()
    p=n.new('ShaderNodeEmission');p.inputs['Color'].default_value=(*color,1);p.inputs['Strength'].default_value=strength
    out=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(p.outputs['Emission'],out.inputs['Surface']);MAT[name]=m;return m

def light(name,kind,loc,color,power,size=1,target=None):
    d=bpy.data.lights.new(name,kind);d.energy=power;d.color=color
    if hasattr(d,'shape') and kind=='AREA':enum(d,'shape','DISK');d.size=size
    o=link(bpy.data.objects.new(name,d));o.location=loc
    if target:o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
    return o

def text(name,body,loc,size,mat,parent=None):
    d=bpy.data.curves.new(name,'FONT');d.body=body;d.size=size;d.space_character=1.1
    d.font=bpy.data.fonts.load('/System/Library/Fonts/Supplemental/Arial Bold.ttf',check_existing=True)
    o=link(bpy.data.objects.new(name,d));o.location=loc;o.parent=parent;d.materials.append(mat);return o

def bootstrap():
    global COL
    if bpy.data.scenes.get(SCENE_NAME):raise RuntimeError('Cinematic scene already exists; use the saved editable scene')
    s=bpy.data.scenes.new(SCENE_NAME);bpy.context.window.scene=s
    COL=bpy.data.collections.new('AFTERLIGHT — authored assets');s.collection.children.link(COL)
    s.render.engine='BLENDER_EEVEE'
    s.render.resolution_x=1920;s.render.resolution_y=1080;s.render.resolution_percentage=100;s.render.fps=FPS
    s.frame_start=1;s.frame_end=END
    s.eevee.taa_render_samples=32
    if hasattr(s.eevee,'use_gtao'):s.eevee.use_gtao=True
    s.world=bpy.data.worlds.new('Cold disaster dusk');s.world.use_nodes=True
    bg=next(n for n in s.world.node_tree.nodes if n.type=='BACKGROUND');bg.inputs['Color'].default_value=(.10,.16,.23,1);bg.inputs['Strength'].default_value=.32
    # Current color-management values are preserved; no guessed transform enum.
    material('Concrete',( .22,.245,.25),grit=.065)
    material('Pale broken plaster',(.48,.43,.34),grit=.045)
    material('Brick',(.28,.09,.047),grit=.035)
    material('Brick pale',(.38,.17,.09),grit=.025)
    material('Rebar',(.105,.13,.14),rough=.4,metal=.8)
    material('Ash',(.065,.081,.083),grit=.022)
    material('Wood',(.19,.092,.041),grit=.026)
    material('Fabric',(.075,.14,.135),grit=.035)
    material('Fly thorax',(.27,.17,.075),rough=.35,metal=.25)
    material('Fly amber',(.51,.265,.062),rough=.3)
    material('Fly dark',(.022,.018,.013),rough=.32)
    material('Fly eyes',(.51,.018,.025),rough=.19,metal=.2,grit=.003)
    material('Wing glass',(.43,.70,.77),rough=.18,metal=.35,alpha=.27)
    material('Wing veins',(.17,.27,.26),rough=.35,alpha=.70)
    material('Glass',(.23,.41,.42),rough=.16,metal=.5,alpha=.3)
    emission('Cyan',(.055,.85,1),2)
    emission('Gold',(1,.32,.055),2.5)
    emission('White',(.83,.92,1),1)
    emission('Muted',(.30,.44,.53),1)
    emission('HUD black',(.006,.014,.022),1)
    emission('HUD line',(.035,.18,.23),1)
    print('Scene created',s.name,END,'frames')

def environment():
    s=bpy.context.scene
    box('Foundation',(0,0,-.25),(10.2,10.2,.5),MAT['Concrete'],bevel=.10)
    # Irregular floor tiles, seams, fissures and dust; passable cells remain clear.
    for ix in range(15):
        for iy in range(15):
            box('Scorched floor tile',((ix-7)*.66,(iy-7)*.66,-.018),(.647,.647,.035),MAT['Concrete' if R.random()>.22 else 'Ash'])
    for k in range(36):
        x,y=R.uniform(-4.8,4.8),R.uniform(-4.8,4.8)
        line('Floor fracture',[(x,y,.01),(x+.17,y+.09,.012),(x+.36,y-.05,.012),(x+.5,y+.03,.012)],.006,MAT['Ash'])
    # Cutaway shell: high fractured walls at the back, no front wall to hide action.
    for x in np.arange(-4.6,5,.62):
        h=R.uniform(1.5,3.1)
        box('Ruptured north wall',(x,4.8,h/2),(.60,.24,h),MAT['Pale broken plaster'],bevel=.035)
        for j in range(3):
            line('Exposed wall rebar',[(x-.17+j*.15,4.8,h-.15),(x-.15+j*.15,4.8,h+R.uniform(.15,.55))],.012,MAT['Rebar'])
    for y in [-4.2,-3,-1.8,-.6,.6,1.8,3,4.2]:
        box('West structural pier',(-4.85,y,1.4),(.26,.3,2.8),MAT['Concrete'],bevel=.025)
    box('Broken lintel',(-4.85,2,2.6),(.3,3.7,.22),MAT['Concrete'],rot=(.1,.07,0))
    for z in [.55,1.05,1.55]:line('Window frame',[(-4.8,-3,z),(-4.8,.6,z)],.018,MAT['Rebar'])
    for i,wall in enumerate(REPLAY['obstacles']):
        xmin,ymin,xmax,ymax=wall;x=(xmin+xmax)/2;y=(ymin+ymax)/2
        root=empty(f'Obstacle {i+1:02d} | recorded blocked cell',(x,y,0))
        root['mission_footprint']=wall
        kind=i%4
        if kind in (0,1):
            # Bricked fragments stay inside each original blocked-cell footprint.
            for course in range(3+(i%3)):
                for b in range(3):
                    xx=x+(b-1)*.25+R.uniform(-.03,.03)
                    box('Exposed masonry',(xx,y+R.uniform(-.13,.13),.11+course*.16),(.23,.28,.145),MAT['Brick' if b%2 else 'Brick pale'],rot=(0,0,R.uniform(-.08,.08)),bevel=.009)
            box('Collapsed concrete slab',(x,y,.70 if kind==0 else .38),(.74,.62,.14),MAT['Pale broken plaster'],rot=(.38 if kind==0 else .13,.2,R.uniform(-.2,.2)),bevel=.025)
            for j in range(3):line('Bent exposed reinforcement',[(x-.25+j*.22,y-.3,.83),(x-.25+j*.22,y,.96),(x-.16+j*.22,y+.36,.83)],.011,MAT['Rebar'])
        elif kind==2:
            # Broken desk with a displaced drawer.
            box('Overturned desk top',(x,y,.5),(.77,.63,.1),MAT['Wood'],rot=(.28,.12,.1),bevel=.024)
            for dx in [-.27,.27]:
                for dy in [-.20,.20]:box('Desk leg',(x+dx,y+dy,.25),(.055,.055,.5),MAT['Rebar'],rot=(.12,0,.1))
            box('Desk drawer',(x+.1,y-.15,.12),(.42,.34,.20),MAT['Wood'],rot=(0,0,-.23),bevel=.018)
        else:
            box('Abandoned armchair seat',(x,y,.34),(.71,.65,.25),MAT['Fabric'],rot=(0,.12,-.08),bevel=.07)
            box('Abandoned armchair back',(x,y+.25,.70),(.7,.16,.65),MAT['Fabric'],rot=(.25,0,0),bevel=.055)
            for dx in [-.32,.32]:box('Armchair arm',(x+dx,y,.51),(.105,.55,.13),MAT['Wood'],bevel=.025)
        for k in range(11):
            dx,dy=R.uniform(-.37,.37),R.uniform(-.37,.37)
            box('Masonry debris',(x+dx,y+dy,R.uniform(.04,.10)),(R.uniform(.07,.18),R.uniform(.05,.15),R.uniform(.06,.12)),MAT['Brick' if k%3 else 'Concrete'],rot=(R.uniform(-.3,.3),R.uniform(-.3,.3),R.random()*3))
    # Background rubble lies outside the traversable arena.
    for i in range(80):
        side=R.choice([-1,1]);x=side*R.uniform(4.3,4.7);y=R.uniform(-4.6,4.6)
        box('Perimeter rubble',(x,y,.09),(R.uniform(.12,.3),R.uniform(.12,.4),R.uniform(.07,.22)),MAT['Concrete'],rot=(0,R.uniform(-.3,.3),R.random()*3))
    base=REPLAY['base'];target=REPLAY['target']
    box('Rescue docking plinth',(*base,.08),(.66,.66,.16),MAT['Rebar'],bevel=.065)
    for radius in [.32,.39]:
        pts=[(base[0]+radius*math.cos(t),base[1]+radius*math.sin(t),.17) for t in np.linspace(0,2*math.pi,65)]
        line('Home luminous ring',pts,.016,MAT['Cyan'])
    light('Home cyan light','POINT',(*base,.35),(.04,.65,1),28)
    beacon=empty('Beacon at recorded target',(*target,.16))
    b=box('Rescue beacon case',(0,0,0),(.23,.2,.25),MAT['Rebar'],bevel=.025);b.parent=beacon
    for z in [-.045,.025,.095]:
        b=box('Beacon luminous band',(0,-.103,z),(.18,.008,.018),MAT['Gold']);b.parent=beacon
    light('Golden hour breach','AREA',(2.5,4.0,7.0),(1,.64,.32),1700,5,(0,0,0))
    light('Cool sky fill','AREA',(-3,-2,7),(.34,.61,1),1100,6,(0,0,0))
    light('Front softbox','AREA',(3,-7,5),(.75,.89,1),850,5,(0,0,1))
    sun=light('Sun through collapsed roof','SUN',(0,0,5),(1,.70,.43),1.8,target=(-3,-4,0));sun.data.angle=.12
    # Sparse motes, kept as presentation decoration above the navigation plane.
    for i in range(70):
        uvball('Airborne dust',(R.uniform(-4.8,4.8),R.uniform(-4.5,4.5),R.uniform(.7,3.3)),(.008,)*3,MAT['Muted'],segments=6,rings=4)
    print('Ruins built',len(s.objects),'objects; exactly',len(REPLAY['obstacles']),'mission obstacles')

def fly():
    root=empty('Fruit fly | replay actor')
    uvball('Thorax',(0,0,0),(.16,.20,.145),MAT['Fly thorax'],root)
    uvball('Head',(0,.21,.035),(.155,.125,.13),MAT['Fly amber'],root)
    for side in [-1,1]:
        eye=uvball('Ruby compound eye',(side*.119,.246,.065),(.087,.091,.104),MAT['Fly eyes'],root,32,16)
        # Small raised facets catch the key light.
        for j in range(22):
            a=R.uniform(-1.2,1.2);b=R.uniform(-1.1,1.1)
            uvball('Eye facet',(side*(.13+.058*math.cos(a)*math.cos(b)),.247+.064*math.sin(a),.065+.078*math.sin(b)),(.008,)*3,MAT['Fly eyes'],root,6,4)
        line('Antenna',[(side*.046,.305,.087),(side*.063,.375,.11),(side*.12,.39,.15)],.009,MAT['Fly dark'],root)
        for j in range(3):
            y=.10-j*.12
            pts=[(side*.12,y,-.065),(side*(.23+.04*j),y-.08,-.17),(side*(.26+.04*j),y-.16,-.31)]
            line('Articulated leg',pts,.011,MAT['Fly dark'],root)
            line('Claw',[pts[-1],(pts[-1][0]+side*.035,pts[-1][1]-.035,pts[-1][2]+.01)],.005,MAT['Fly amber'],root)
        wing=empty('Wing.L' if side<0 else 'Wing.R',(side*.11,-.025,.11),root)
        outline=[(0,0,0),(.22*side,.04,.014),(.55*side,-.15,.022),(.58*side,-.34,.012),(.41*side,-.47,.006),(.13*side,-.30,0)]
        ob=mesh('Transparent veined wing',outline,[(0,1,2,3,4,5)],MAT['Wing glass']);ob.parent=wing
        line('Wing membrane outline',outline+[outline[0]],.004,MAT['Wing veins'],wing)
        for tip in outline[2:5]:line('Radial wing vein',[(0,0,.003),(tip[0]*.45,tip[1]*.25,.014),tip],.003,MAT['Wing veins'],wing)
        line('Wing cross vein',[(side*.15,-.075,.013),(side*.35,-.17,.016),(side*.36,-.35,.009)],.0025,MAT['Wing veins'],wing)
        for f in range(1,END+1):
            wing.rotation_euler.y=side*.48*math.sin(2*math.pi*f/3.7)
            wing.keyframe_insert('rotation_euler',frame=f)
    # Tapered abdomen with visible alternating dark bands.
    for i in range(7):
        t=i/6;size=.146*(1-.75*t)
        uvball('Abdomen segment %d'%i,(0,-.17-i*.054,-.013-i*.009),(size,.052,size*.74),MAT['Fly dark' if i%2 else 'Fly amber'],root)
    for i in range(32):
        a=R.uniform(0,2*math.pi);y=R.uniform(-.08,.12)
        start=(.15*math.cos(a),y,.135*math.sin(a))
        line('Thoracic bristle',[start,(start[0]*1.2,y-.027,start[2]*1.25)],.002,MAT['Fly dark'],root)
    payload=box('Recovered beacon payload',(0,.01,-.30),(.17,.14,.17),MAT['Gold'],bevel=.023);payload.parent=root
    payload.hide_render=True;payload.hide_viewport=True
    lamp=light('Fly tracking glint','POINT',(0,0,.4),(.5,.85,1),8);lamp.parent=root
    print('Fruit fly modeled: paired wings, six legs, segmented abdomen, compound eyes')

def camera_and_animation():
    s=bpy.context.scene
    data=bpy.data.cameras.new('Cinematic camera');cam=link(bpy.data.objects.new('Cinematic camera',data));s.camera=cam
    data.lens=40;data.clip_start=.05;data.clip_end=200
    root=bpy.data.objects['Fruit fly | replay actor'];beacon=bpy.data.objects['Beacon at recorded target'];payload=bpy.data.objects['Recovered beacon payload']
    points=np.array([f['position'] for f in REPLAY['frames']]+[REPLAY['final_position']])
    last_angle=0.
    for frame in range(1,END+1):
        progress=np.clip((frame-INTRO)/MOVE,0,42)
        step=min(int(progress),41);fraction=min(progress-step,1)
        # Smooth only along each recorded segment: never cut across obstacle corners.
        smooth=fraction*fraction*(3-2*fraction)
        xy=points[step]*(1-smooth)+points[step+1]*smooth
        z=.93+.055*math.sin(frame*.19)
        if frame>INTRO+42*MOVE:z-=.52*min((frame-(INTRO+42*MOVE))/36,1)
        root.location=(*xy,z)
        delta=points[step+1]-points[step];angle=math.atan2(-delta[0],delta[1])
        while angle-last_angle>math.pi:angle-=2*math.pi
        while angle-last_angle<-math.pi:angle+=2*math.pi
        last_angle+=.28*(angle-last_angle)
        root.rotation_euler=(.025*math.sin(frame*.19),.035*math.sin(frame*.12),last_angle)
        root.keyframe_insert('location',frame=frame);root.keyframe_insert('rotation_euler',frame=frame)
        opening=np.clip((frame-1)/INTRO,0,1);opening=opening*opening*(3-2*opening)
        end_zoom=np.clip((frame-(INTRO+42*MOVE))/48,0,1)
        focus=np.array([xy[0]*.75,xy[1]*.75,.72])
        macro=np.array([xy[0],xy[1],.95])
        focus=macro*(1-opening)+focus*opening
        offset=np.array([1.15,-1.8,1.05])*(1-opening)+np.array([4.2,-5.7,5.3])*opening
        offset*=1-.35*end_zoom
        cam.location=focus+offset
        # Place the tracked subject left of center to leave room for the brain HUD.
        aim=Vector(focus);q=(aim-cam.location).to_track_quat('-Z','Y')
        aim+=q@Vector((.60*opening,0,0))
        cam.rotation_euler=(aim-cam.location).to_track_quat('-Z','Y').to_euler()
        cam.keyframe_insert('location',frame=frame);cam.keyframe_insert('rotation_euler',frame=frame)
    pickup_frame=INTRO+REPLAY['pickup_step']*MOVE
    for o,inverse in [(payload,False),(beacon,True)]+[(ch,True) for ch in beacon.children]:
        for f,hidden in [(1,not inverse),(pickup_frame-1,not inverse),(pickup_frame,inverse),(END,inverse)]:
            o.hide_render=hidden;o.hide_viewport=hidden;o.keyframe_insert('hide_render',frame=f);o.keyframe_insert('hide_viewport',frame=f)
    for name,frame in [('Departure',INTRO),('Beacon acquired',pickup_frame),('Returned to base',INTRO+42*MOVE)]:s.timeline_markers.new(name,frame=frame)
    print('Flight / camera baked',END,'frames; pickup',pickup_frame)

def hud():
    s=bpy.context.scene;cam=s.camera
    # Camera-local HUD at depth 2.0. These objects remain fixed on screen.
    corners=cam.data.view_frame(scene=s)
    z=2.0;w=max(abs(v.x/v.z*z) for v in corners);h=max(abs(v.y/v.z*z) for v in corners)
    def pos(u,v,depth=z):return ((u*2-1)*w,(v*2-1)*h,-depth)
    def rect(name,u,v,width,height,mat,depth=2.02):
        o=mesh(name,[(-width*w,-height*h,0),(width*w,-height*h,0),(width*w,height*h,0),(-width*w,height*h,0)],[(0,1,2,3)],mat)
        o.parent=cam;o.location=pos(u,v,depth);return o
    def txt(name,body,u,v,size,mat=MAT['White']):return text(name,body,pos(u,v,1.98),h*size,mat,cam)
    rect('Brain HUD backing',.835,.74,.28,.44,MAT['HUD black'],2.08)
    rect('Brain HUD accent',.695,.74,.002,.44,MAT['Cyan'],2.06)
    txt('Brand','AFTERLIGHT',.04,.90,.105)
    txt('Subtitle','A FRUIT FLY CONNECTOME / A RESCUE MISSION',.041,.864,.025,MAT['Cyan'])
    txt('Brain title','16,000 NEURONS',.72,.914,.046,MAT['Cyan'])
    txt('Brain subtitle','RECORDED MODEL ACTIVITY',.72,.88,.021,MAT['Muted'])
    txt('Brain source','MaleCNS / measured soma positions',.72,.555,.020,MAT['Muted'])
    txt('Brain source missing','15,619 shown / 381 lack coordinates',.72,.53,.018,MAT['Muted'])
    txt('Brain legend','+ CYAN     /     - AMBER',.72,.585,.024,MAT['White'])
    # The transparent shell is derived from measured positions, not invented lobes.
    with np.load(OUT/'brain_visual.npz') as g:p=g['positions'];faces=g['hull_faces'];colors=g['colors']
    brain=empty('Brain HUD anatomical group',pos(.835,.735,1.9),cam)
    brain.scale=(w*.44,)*3;brain.rotation_euler=(.20,0,-.04)
    shellmat=material('Brain envelope glass',(.08,.45,.67),rough=.2,metal=.4,emission=.2,alpha=.085)
    shell=mesh('Transparent soma envelope',p.tolist(),faces.tolist(),shellmat);shell.parent=brain
    for poly in shell.data.polygons:poly.use_smooth=True
    # Octahedra: six vertices per measured soma, one atlas UV per neuron.
    n=len(p);r=.0031
    template=np.array([[r,0,0],[-r,0,0],[0,r,0],[0,-r,0],[0,0,r],[0,0,-r]])
    vertices=(p[:,None,:]+template[None,:,:]).reshape(-1,3)
    tri=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    triangles=(tri[None,:,:]+np.arange(n)[:,None,None]*6).reshape(-1,3)
    neurons=mesh('Measured neural activity / all positioned neurons',vertices.tolist(),triangles.tolist());neurons.parent=brain
    uv=neurons.data.uv_layers.new(name='Neuron address')
    uvdata=np.zeros((len(neurons.data.loops),2),'float32');uvdata[:,0]=np.repeat((np.arange(n)+.5)/n,24)
    uv.data.foreach_set('uv',uvdata.ravel())
    atlas=bpy.data.images.new('Recorded 42-step neural activity',width=n,height=42,alpha=True,float_buffer=False)
    atlas.pixels.foreach_set(colors.ravel());atlas.pack()
    m=bpy.data.materials.new('Measured signed neural emission');m.use_nodes=True;nodes=m.node_tree.nodes;nodes.clear()
    tex=nodes.new('ShaderNodeTexImage');tex.image=atlas;enum(tex,'interpolation','Closest')
    uvnode=nodes.new('ShaderNodeTexCoord');offset=nodes.new('ShaderNodeVectorMath');enum(offset,'operation','ADD')
    row=nodes.new('ShaderNodeCombineXYZ');row.name='Recorded decision row'
    for step in range(42):
        row.inputs['Y'].default_value=(step+.5)/42
        row.inputs['Y'].keyframe_insert('default_value',frame=INTRO+step*MOVE)
    row.inputs['Y'].default_value=.5/42;row.inputs['Y'].keyframe_insert('default_value',frame=1)
    shader=nodes.new('ShaderNodeEmission');shader.inputs['Strength'].default_value=2.3
    out=nodes.new('ShaderNodeOutputMaterial');links=m.node_tree.links
    links.new(uvnode.outputs['UV'],offset.inputs[0]);links.new(row.outputs['Vector'],offset.inputs[1]);links.new(offset.outputs['Vector'],tex.inputs['Vector'])
    links.new(tex.outputs['Color'],shader.inputs['Color']);links.new(shader.outputs['Emission'],out.inputs['Surface']);neurons.data.materials.append(m)
    # Stepped atlas sampling: one exact recorded state per policy decision.
    action=m.node_tree.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    for kp in fc.keyframe_points:enum(kp,'interpolation','CONSTANT')
    rect('Status backing',.5,.087,.94,.10,MAT['HUD black'],2.05)
    phases=[('SEARCH / LOCATE THE HIDDEN BEACON',1,INTRO+35*MOVE-1),('BEACON SECURED / RETURN TO BASE',INTRO+35*MOVE,INTRO+42*MOVE-1),('MISSION COMPLETE / BACK AT BASE',INTRO+42*MOVE,END)]
    for i,(label,start,stop) in enumerate(phases):
        o=txt('Mission phase '+str(i),label,.055,.092,.043,MAT['Gold'] if i else MAT['Cyan'])
        for f,hidden in [(1,start>1),(max(1,start-1),start>1),(start,False),(stop,False),(stop+1,True)]:
            o.hide_render=hidden;o.hide_viewport=hidden;o.keyframe_insert('hide_render',frame=f);o.keyframe_insert('hide_viewport',frame=f)
    txt('Replay disclosure','RECORDED POLICY / CINEMATIC FLIGHT REPLAY',.055,.061,.020,MAT['Muted'])
    txt('Episode identity','SEED 200089   /   7 x 7   /   16 OBSTACLES',.655,.071,.022,MAT['Muted'])
    rect('Timeline bed',.50,.035,.89,.004,MAT['HUD line'],1.96)
    progress=rect('Mission timeline',.055,.035,.89,.004,MAT['Cyan'],1.94)
    # Shift origin to left edge, then animate its length.
    for vertex in progress.data.vertices:vertex.co.x+=.89*w
    for f,scale in [(1,.0001),(INTRO,.0001),(INTRO+42*MOVE,1),(END,1)]:progress.scale.x=scale;progress.keyframe_insert('scale',frame=f)
    print('Brain overlay baked:',n,'measured somas / packed 42-state atlas')

def finalize():
    s=bpy.context.scene
    s.frame_set(156)
    for area in bpy.context.screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_perspective='CAMERA'
            area.spaces.active.overlay.show_overlays=False
    s['replay_source']=str(ROOT/'runs/mission_16000/demo_success/replay.json')
    s['staging_disclosure']='Recorded grid policy. Flight, ruins, fly scale and camera are cinematic staging. Brain envelope from soma positions.'
    s['neuron_count']=16000;s['visible_somas']=15619
    textblock=bpy.data.texts.new('READ ME — AFTERLIGHT')
    textblock.write((OUT/'provenance.json').read_text())
    enum(s.render.image_settings,'file_format','PNG');enum(s.render.image_settings,'color_mode','RGB')
    s.render.filepath=str(OUT/'frames/frame_')
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'afterlight.blend'))
    print('Saved',OUT/'afterlight.blend')


def attach():
    global COL,MAT
    COL=bpy.data.collections['AFTERLIGHT — authored assets']
    MAT={m.name:m for m in bpy.data.materials}
    bpy.context.window.scene=bpy.data.scenes[SCENE_NAME]
