import bpy
from pathlib import Path
out=Path(__file__).resolve().parents[1]
s=bpy.data.scenes['AFTERLIGHT | 16000'];bpy.context.window.scene=s
s.render.resolution_percentage=60;s.eevee.taa_render_samples=24
for f in [1,140,285,500,748,940]:
    s.frame_set(f);s.render.filepath=str(out/f'preview_{f:04}.png')
    bpy.ops.render.render(write_still=True)
    print('PREVIEW',f,flush=True)
