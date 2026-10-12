"""Visual refinement, applied once after build_scene stages."""
import bpy,math,random
import numpy as np
from mathutils import Vector
import build_scene as cb
cb.attach();s=bpy.context.scene
if s.get('polished'):raise RuntimeError('Already polished')
R=random.Random(778)
# Moody cold interior, directional amber light through the broken roof.
next(n for n in s.world.node_tree.nodes if n.type=='BACKGROUND').inputs['Strength'].default_value=.13
bpy.data.lights['Golden hour breach'].energy=1000
bpy.data.lights['Cool sky fill'].energy=260
bpy.data.lights['Front softbox'].energy=260
bpy.data.lights['Sun through collapsed roof'].energy=1.4
s.view_settings.exposure=-.55;s.eevee.taa_render_samples=48
for name in ['Concrete','Pale broken plaster','Brick','Brick pale','Wood','Fabric']:
    m=cb.MAT[name];p=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED');base=np.array(p.inputs['Base Color'].default_value[:3])
    noise=m.node_tree.nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=4.2;noise.inputs['Detail'].default_value=4
    ramp=m.node_tree.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position=.23;ramp.color_ramp.elements[0].color=(*base*.17,1)
    ramp.color_ramp.elements[1].position=.8;ramp.color_ramp.elements[1].color=(*base*.82,1)
    m.node_tree.links.new(noise.outputs['Factor'],ramp.inputs['Factor']);m.node_tree.links.new(ramp.outputs['Color'],p.inputs['Base Color'])
# Make rear panels into a connected broken wall, with two broad ruptures.
for i,o in enumerate([o for o in s.objects if o.name.startswith('Ruptured north wall')]):
    o.scale.x=1.065
    for v in o.data.vertices:
        if v.co.z>0:v.co.z+=R.uniform(-.25,.05);v.co.y+=R.uniform(-.06,.06)
    if i in [4,5,11]:o.scale.z=.38;o.location.z*=.38
cb.box('Continuous ruined wall footing',(0,4.8,.20),(9.8,.27,.4),cb.MAT['Concrete'],bevel=.02)
for i in range(25):
    x=R.uniform(-4.5,4.5);h=R.uniform(.3,1.3)
    cb.line('Cracked plaster scar',[(x,4.658,.28),(x+.08,4.655,h*.55),(x-.04,4.654,h),(x+.09,4.656,h+.27)],.009,cb.MAT['Ash'])
# Small ground-level rubble shards add clutter without changing navigation.
for i in range(190):
    x,y=R.uniform(-4.4,4.4),R.uniform(-4.4,4.4)
    size=R.uniform(.025,.095)
    v=[(-size,-size*.6,0),(size,-size*.55,0),(size*.55,size,0),(-size*.65,size*.8,0),(0,0,R.uniform(.025,.08))]
    o=cb.mesh('Irregular floor shard',v,[(0,1,4),(1,2,4),(2,3,4),(3,0,4),(0,3,2,1)],cb.MAT['Brick' if i%3 else 'Concrete'])
    o.location=(x,y,.019);o.rotation_euler.z=R.random()*6.28
# Bend and tip furniture together, retaining the recorded obstacle centers.
for i,wall in enumerate(cb.REPLAY['obstacles']):
    if i%4!=3:continue
    x=(wall[0]+wall[2])/2;y=(wall[1]+wall[3])/2
    parts=[o for o in s.objects if ('armchair' in o.name.lower()) and abs(o.location.x-x)<.5 and abs(o.location.y-y)<.5]
    group=cb.empty('Disaster tipped furniture',(x,y,.28))
    for o in parts:
        world=o.matrix_world.copy();o.parent=group;o.matrix_world=world
        # final_touches.py converts these retained world positions to group-local coordinates.
    group.rotation_euler=(.63 if i==7 else -.22,.20,-.12)
# Camera HUD must be closer than the fly in the opening macro shot.
cam=s.camera
hudroots=[o for o in s.objects if o.parent==cam]
for o in hudroots:
    o.location*=.25;o.scale*=.25;o.visible_shadow=False
    if o.animation_data and o.animation_data.action:
        for layer in o.animation_data.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        if fc.data_path=='scale':
                            for kp in fc.keyframe_points:
                                kp.co.y*=.25;kp.handle_left.y*=.25;kp.handle_right.y*=.25
    if o.type=='FONT':
        o.data.size*=1.4 if o.name not in ['Brand','Subtitle'] else 1.12
brain=bpy.data.objects['Brain HUD anatomical group'];brain.scale*=.80
# Center the measured envelope, avoiding median-to-boundary offset in the inset.
with np.load(cb.OUT/'brain_visual.npz') as g:p=g['positions']
center=(p.min(0)+p.max(0))/2
for child in brain.children:
    if child.type=='MESH':
        for v in child.data.vertices:v.co-=Vector(center)
    child.visible_shadow=False
# Clean transparent hull: emission + transparency, with no reflected scenery.
m=cb.MAT.get('Brain envelope glass',bpy.data.materials['Brain envelope glass']);nodes=m.node_tree.nodes;nodes.clear()
em=nodes.new('ShaderNodeEmission');em.inputs['Color'].default_value=(.025,.28,.47,1);em.inputs['Strength'].default_value=.8
tr=nodes.new('ShaderNodeBsdfTransparent');mix=nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=.08
out=nodes.new('ShaderNodeOutputMaterial');m.node_tree.links.new(tr.outputs['BSDF'],mix.inputs[1]);m.node_tree.links.new(em.outputs['Emission'],mix.inputs[2]);m.node_tree.links.new(mix.outputs[0],out.inputs['Surface'])
cb.enum(m,'surface_render_method','BLENDED');m.use_transparency_overlap=False
s['polished']=True
s.frame_set(156)
# Fonts and atlas are packed; the saved animation has no external handlers.
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(cb.OUT/'afterlight.blend'))
print('Cinematic lighting, damage, typography and foreground HUD refined.')
