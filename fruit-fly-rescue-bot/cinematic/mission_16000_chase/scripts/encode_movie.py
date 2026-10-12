"""Encode the rendered master without applying Blender color management twice."""
from pathlib import Path
import json,subprocess,hashlib
import imageio_ffmpeg
from PIL import Image
OUT=Path(__file__).resolve().parents[1]
frames=[OUT/f'frames/frame_{i:04d}.png' for i in range(1,961)]
missing=[str(p) for p in frames if not p.exists()]
if missing:raise SystemExit(f'{len(missing)} frames are still missing; render first.')
for p in frames:
    with Image.open(p) as im:
        if im.size!=(1920,1080):raise ValueError(f'Wrong frame size: {p}')
        im.verify()
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
base=[ffmpeg,'-hide_banner','-loglevel','warning','-y']
silent=OUT/'afterlight_chase_silent.mp4';movie=OUT/'afterlight_chase.mp4'
tmp=OUT/'afterlight_chase_silent.partial.mp4'
subprocess.run(base+['-framerate','24','-start_number','1','-i',str(OUT/'frames/frame_%04d.png'),
    '-frames:v','960','-c:v','libx264','-preset','medium','-crf','18',
    '-vf','scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
    '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
    '-movflags','+faststart','-an',str(tmp)],check=True)
tmp.replace(silent)
tmp=OUT/'afterlight_chase.partial.mp4'
subprocess.run(base+['-i',str(silent),'-i',str(OUT/'soundtrack.wav'),'-map','0:v:0','-map','1:a:0',
    '-c:v','copy','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',str(tmp)],check=True)
tmp.replace(movie)
# Decode the full final movie once to catch damaged/truncated output.
subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(movie),'-f','null','-'],check=True)
decoded_frames,decoded_seconds=imageio_ffmpeg.count_frames_and_secs(movie)
assert decoded_frames==960 and abs(decoded_seconds-40)<.1,(decoded_frames,decoded_seconds)
report={'decoded_frames':decoded_frames,'decoded_seconds':decoded_seconds,'video':movie.name,'silent_video':silent.name,'seconds':40,'fps':24,'frames':960,'resolution':[1920,1080],
        'video_codec':'H.264 / yuv420p','audio':'Original synthesized stereo atmosphere / AAC',
        'sha256':hashlib.sha256(movie.read_bytes()).hexdigest(),'bytes':movie.stat().st_size,
        'full_decode_check':'passed'}
(OUT/'delivery.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
