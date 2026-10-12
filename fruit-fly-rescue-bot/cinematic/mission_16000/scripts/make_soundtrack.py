"""Original synthesized atmosphere and event sounds; no third-party recordings."""
from pathlib import Path
import wave
import numpy as np
from scipy.signal import butter,sosfilt
OUT=Path(__file__).resolve().parents[1]
sr=48000;seconds=25;n=sr*seconds;t=np.arange(n)/sr;rng=np.random.default_rng(200089)
noise=rng.normal(0,1,n)
wind=sosfilt(butter(2,550,fs=sr,output='sos'),noise)
wind/=np.std(wind)
wind*=.017*(.75+.20*np.sin(2*np.pi*.13*t))
rumble=.014*np.sin(2*np.pi*48*t)+.012*np.sin(2*np.pi*72*t)
# Stylized wing hum, deliberately quieter behind the atmosphere.
frequency=185+10*np.sin(2*np.pi*.7*t)
phase=2*np.pi*np.cumsum(frequency)/sr
buzz=.008*np.sin(phase)+.003*np.sin(phase*2)
buzz*=.7+.3*np.sin(2*np.pi*8*t)**2
mono=wind+rumble+buzz
for event,pitches in [(19.5,[523.25,783.99]),(23,[392,523.25,659.25,783.99])]:
    for j,f in enumerate(pitches):
        tau=t-(event+j*.11);active=tau>=0;env=np.where(active,np.exp(-np.maximum(tau,0)*3),0)
        mono+=.055*env*np.sin(2*np.pi*f*np.maximum(tau,0))
fade=np.minimum(t/.6,1)*np.minimum((seconds-t)/1.4,1)
left=mono*fade;right=(mono*.94+np.roll(wind,37)*.12)*fade
stereo=np.stack([left,right],axis=1)
stereo=np.clip(stereo,-.9,.9)
with wave.open(str(OUT/'soundtrack.wav'),'wb') as f:
    f.setnchannels(2);f.setsampwidth(2);f.setframerate(sr);f.writeframes((stereo*32767).astype('<i2').tobytes())
print('Original stereo atmosphere: 25 seconds; peak',float(np.max(np.abs(stereo))))
