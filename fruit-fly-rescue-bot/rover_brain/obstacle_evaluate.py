"""Independent obstacle evaluation and replay, with no teacher in policy mode."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .obstacle_train import GRAPH, rollout
from .model import ConnectomePolicy
from .train import resolve_device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,default=GRAPH)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/obstacles_8000/best.pt'))
    p.add_argument('--output-dir',type=Path,default=Path('runs/obstacles_8000/test'))
    p.add_argument('--seed',type=int,default=80000)
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--obstacles',type=int,default=6)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--modes',nargs='+',choices=['policy','silenced','direct','teacher'],default=['policy','silenced','direct','teacher'])
    args=p.parse_args()
    if args.episodes<1:p.error('--episodes must be positive')
    device=resolve_device(args.device);torch.set_num_threads(6)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    digest=hashlib.sha256(args.graph.read_bytes()).hexdigest()
    if digest!=ckpt['graph_sha256']:raise ValueError('Checkpoint graph mismatch')
    model=ConnectomePolicy(args.graph,ticks=ckpt['ticks'],observations=ckpt['observations']).to(device)
    model.load_state_dict(ckpt['model'])
    report={'stage':'obstacles','obstacles_per_layout':args.obstacles,'graph_sha256':digest,
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'checkpoint_training_step':ckpt['step'],'device':device,
            'limitations':['Visible beacon and simulator-supplied local obstacle map',
                           'Imitation learning; no hidden-target search or return-to-base yet',
                           'No matched trained ordinary-network or rewired baseline']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for mode in args.modes:
        result,replay,_=rollout(model,device,args.seed,count=args.episodes,obstacles=args.obstacles,
                                mode=mode,record=mode=='policy')
        report[mode]=result
        print(mode,json.dumps(result),flush=True)
        if replay:
            replay.update(graph_sha256=digest,checkpoint_sha256=report['checkpoint_sha256'])
            (args.output_dir/'replay.json').write_text(json.dumps(replay)+'\n')
    (args.output_dir/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
