"""Evaluate full missions and independently disable each neural-memory bank."""
import argparse
from dataclasses import asdict
from .mission import MissionSpec, DEFAULT_SPEC
import hashlib
import json
from pathlib import Path
import torch
from .obstacle_train import GRAPH
from .memory_model import MemoryConnectomePolicy
from .mission_train import rollout
from .train import resolve_device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,default=GRAPH)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/mission_8000/best.pt'))
    p.add_argument('--output-dir',type=Path,default=Path('runs/mission_8000/test'))
    p.add_argument('--seed',type=int,default=200000)
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--modes',nargs='+',choices=['policy','no_memory','no_search_memory','erase_home_on_pickup','silenced','teacher'],
                   default=['policy','no_memory','no_search_memory','erase_home_on_pickup','silenced','teacher'])
    args=p.parse_args()
    if args.episodes<1:p.error('Positive episode count required')
    device=resolve_device(args.device);torch.set_num_threads(6)
    checkpoint_bytes=args.checkpoint.read_bytes()
    checkpoint_hash=hashlib.sha256(checkpoint_bytes).hexdigest()
    import io
    checkpoint=torch.load(io.BytesIO(checkpoint_bytes),map_location=device,weights_only=False)
    graph_hash=hashlib.sha256(args.graph.read_bytes()).hexdigest()
    if checkpoint['graph_sha256']!=graph_hash:raise ValueError('Graph/checkpoint mismatch')
    spec=MissionSpec(**checkpoint['config'].get('mission',asdict(DEFAULT_SPEC)))
    model=MemoryConnectomePolicy(args.graph,ticks=checkpoint['ticks'],spec=spec).to(device)
    model.load_state_dict(checkpoint['model'])
    report={'mission':asdict(spec),'neurons':model.n,'stage':'hidden_mission','graph_sha256':graph_hash,'checkpoint_sha256':checkpoint_hash,
            'checkpoint_step':checkpoint['step'],'device':device,
            'memory':'Engineered gated home and visit banks with trained encodings, graph gains and readouts',
            'limitations':['Discrete cardinal grid motion with legal-action masking; not continuous steering',
                           'Known obstacle grid and ideal cell localization',
                           'Local beacon sensor, fixed obstacles, no sensor noise',
                           'Memory gate operations are engineered, not a biological learning rule']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for mode in args.modes:
        result,replay,_=rollout(model,device,args.seed,count=args.episodes,mode=mode,record=mode=='policy')
        report[mode]=result
        print(mode,json.dumps({k:v for k,v in result.items() if k not in ('failed_seeds','episodes_detail')}),flush=True)
        if replay:
            replay.update(graph_sha256=graph_hash,checkpoint_sha256=checkpoint_hash)
            (args.output_dir/'replay.json').write_text(json.dumps(replay)+'\n')
    if 'policy' in report and 'erase_home_on_pickup' in report:
        if report['policy']['collected']!=report['erase_home_on_pickup']['collected']:
            raise AssertionError('Home ablation changed pre-pickup behavior')
    (args.output_dir/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
