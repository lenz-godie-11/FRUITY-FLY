"""Prepare measured anatomy and recorded activity; no synthetic neural signals."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'cinematic/mission_16000'
graph=ROOT/'data/processed/malecns_16000/graph.npz'
replay_path=ROOT/'runs/mission_16000/demo_success/replay.json'
r=json.loads(replay_path.read_text())
assert r['success'] and hashlib.sha256(graph.read_bytes()).hexdigest()==r['graph_sha256']
with np.load(graph) as g:
    coords=g['soma_um'];ids=g['neuron_ids']
valid=np.isfinite(coords).all(1)
p=coords[valid].astype(float)
p-=np.median(p,axis=0)
# Preserve anatomical axes. X horizontal, original Z vertical, original Y depth.
p=p[:,[0,2,1]];p[:,1]*=-1
p/=np.max(np.ptp(p,axis=0))
activity=np.array([f['activity'] for f in r['frames']],np.float32)[:,valid]
mag=np.log1p(100*np.abs(activity))/np.log(101)
positive=np.array([.05,.85,1.0]);negative=np.array([1.0,.31,.055]);base=np.array([.012,.035,.055])
colors=base[None,None,:]+mag[:,:,None]*(np.where(activity[:,:,None]>=0,positive,negative)-base)
rgba=np.concatenate([colors,np.ones((*colors.shape[:2],1))],axis=2).astype('float32')
hull=ConvexHull(p)
# Two convex anatomical envelopes illustrate transparency; not traced neuropil surfaces.
np.savez_compressed(OUT/'brain_visual.npz',positions=p.astype('float32'),colors=rgba,
                    hull_faces=hull.simplices.astype('int32'),source_indices=np.flatnonzero(valid),neuron_ids=ids[valid])
metadata={'replay':str(replay_path.relative_to(ROOT)),'replay_sha256':hashlib.sha256(replay_path.read_bytes()).hexdigest(),
          'graph_sha256':r['graph_sha256'],'neurons':len(coords),'positioned_neurons':int(valid.sum()),
          'missing_positions_omitted':int((~valid).sum()),'seed':r['seed'],'decisions':len(r['frames']),
          'pickup_step':r['pickup_step'],'success':r['success'],
          'activity_mapping':'Recorded signed activity; cyan positive / amber negative; magnitude log1p(100*abs(a))/log(101), fixed global scale.',
          'staging':'XY path and 16 obstacle footprints from the mission. Flight height, wings, fly size, ruins, camera, dust, and light are presentation only.',
          'brain_shell':'Convex envelope of measured soma positions, not a segmented anatomical surface.'}
(OUT/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
