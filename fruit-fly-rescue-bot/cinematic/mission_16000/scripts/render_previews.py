import bpy,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=bpy.data.scenes['AFTERLIGHT | 16000'];bpy.context.window.scene=s
s.render.resolution_percentage=50;s.eevee.taa_render_samples=24
for frame in [1,156,468,575]:
    s.frame_set(frame);s.render.filepath=str(root/f'preview_{frame:04}.png')
    started=time.perf_counter();bpy.ops.render.render(write_still=True)
    print('PREVIEW',frame,round(time.perf_counter()-started,2),flush=True)
