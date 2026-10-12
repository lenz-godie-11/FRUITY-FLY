"""Train search/home policy on exact recomputed neural-memory histories.

The max/write memory rules permit exact history recomputation for training,
avoiding long sequence backpropagation. Deployment updates memory online.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .mission import (MissionBatch,teacher_actions,neighbour,ACTION_NAMES,
                      MissionSpec,DEFAULT_SPEC)
from .memory_model import MemoryConnectomePolicy
from .obstacle_train import GRAPH
from .train import resolve_device


def legal_mask(observations,spec=DEFAULT_SPEC):
    """Known grid motion constraints only; no beacon/home/history information."""
    cells=observations[:,spec.position].argmax(axis=1)
    walls=observations[:,spec.walls].astype(bool)
    mask=np.zeros((len(cells),4),bool)
    for i,c in enumerate(cells):
        for a in range(4):
            n=neighbour(int(c),a,spec)
            mask[i,a]=n>=0 and not walls[i,n]
    return mask


def statistics(env,seed):
    return {'episodes':len(env.cell),'seed':seed,'discovered':int(env.discovered.sum()),
            'collected':int(env.carrying.sum()),'returned_to_base':int(env.success.sum()),
            'mission_success_rate':float(env.success.mean()),
            'return_rate_after_pickup':float(env.success.sum()/env.carrying.sum()) if env.carrying.any() else None,
            'mean_decisions':float(env.steps.mean()),'collision_attempts':int(env.collision_attempts.sum()),
            'timeouts':int((~env.success).sum()),'max_steps':env.max_steps,
            'mean_success_decisions':float(env.steps[env.success].mean()) if env.success.any() else None,
            'mean_visited_cells':float(env.visited.sum(axis=1).mean()),
            'failed_before_pickup':int((~env.carrying).sum()),
            'failed_after_pickup':int((env.carrying & ~env.success).sum()),
            'failed_seeds':(np.flatnonzero(~env.success)+seed).tolist(),
            'mission':asdict(env.spec),
            'episodes_detail':[{'seed':seed+i,'collected':bool(env.carrying[i]),
                 'success':bool(env.success[i]),'decisions':int(env.steps[i]),
                 'pickup_step':int(env.pickup_step[i]),'visited_cells':int(env.visited[i].sum())}
                 for i in range(len(env.cell))]}


@torch.no_grad()
def rollout(model,device,seed,count=64,mode='policy',collect=False,record=False,obstacles=None):
    spec=model.spec
    positions=spec.positions
    env=MissionBatch(count=count,seed=seed,obstacles=obstacles,spec=spec)
    model.eval()
    state=model.initial_state(count,device)
    weights=None if mode=='teacher' else model.weights()
    xs,ys,bs,vs,frames=[],[],[],[],[]
    home_correct,home_count=0,0
    while not env.done.all():
        obs=env.observation()
        if collect:
            active=~env.done
            labels=teacher_actions(env)
            xs.append(obs[active]);ys.append(labels[active]);bs.append(env.base[active].copy());vs.append(env.visited[active].copy())
        if mode=='teacher':
            actions=teacher_actions(env)
        else:
            if mode=='no_memory':state=torch.zeros_like(state)
            if mode=='erase_home_on_pickup':
                state=state.clone()
                carrying=torch.from_numpy(env.carrying).to(device)[:,None]
                state[:,model.home_indices]=torch.where(carrying,0.,state[:,model.home_indices])
            logits,state,home_logits,visit_logits=model.step(torch.from_numpy(obs).to(device),state,weights=weights,
                       forget_visits=mode=='no_search_memory',silence=mode=='silenced')
            mask=torch.from_numpy(legal_mask(obs,spec)).to(device)
            logits=logits.masked_fill(~mask,-1e9)
            actions=logits.argmax(1).cpu().numpy()
            recall_mask=(~env.done)&(env.steps>0)
            recalled=home_logits.argmax(1).cpu().numpy()
            home_correct+=int(((recalled==env.base)&recall_mask).sum())
            home_count+=int(recall_mask.sum())
            if record and not env.done[0]:
                frames.append({'step':int(env.steps[0]),'position':positions[env.cell[0]].tolist(),
                               'cell':int(env.cell[0]),'carrying':bool(env.carrying[0]),
                               'local_signal':obs[0,spec.signal].tolist(),'action':int(actions[0]),
                               'probabilities':logits[0].softmax(0).cpu().tolist(),
                               'activity':state[0].cpu().numpy().round(5).tolist(),
                               'decoded_home_cell':int(home_logits[0].argmax()),
                               'visited_count':int(env.visited[0].sum())})
        env.step(actions)
    report=statistics(env,seed)
    if home_count:report['home_decoder_accuracy_after_departure']=home_correct/home_count
    data=None
    if collect:data={'x':np.concatenate(xs),'y':np.concatenate(ys),'base':np.concatenate(bs),'visited':np.concatenate(vs)}
    replay=None
    if record:
        boxes=[]
        for cell in np.flatnonzero(env.walls[0]):
            pos=positions[cell];boxes.append([*(pos-.43),*(pos+.43)])
        replay={'mission':asdict(env.spec),'neurons':model.n,'stage':'hidden_mission','seed':seed,'action_names':ACTION_NAMES,'frames':frames,
                'base':positions[env.base[0]].tolist(),'target':positions[env.beacon[0]].tolist(),
                'obstacles':boxes,'success':bool(env.success[0]),'collected':bool(env.carrying[0]),
                'final_position':positions[env.cell[0]].tolist(),'pickup_step':int(env.pickup_step[0]),
                'note':'Grid mission; local beacon only. Target location in replay is for the audience, never policy input.'}
    return report,replay,data


def make_demonstrations(model,device,layouts,seed=100000):
    chunks=[]
    for offset in range(0,layouts,64):
        report,_,data=rollout(model,device,seed+offset,count=min(64,layouts-offset),mode='teacher',collect=True)
        if report['returned_to_base']!=report['episodes']:raise RuntimeError('Teacher failed a mission')
        chunks.append(data)
    return {key:np.concatenate([c[key] for c in chunks]) for key in chunks[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,default=GRAPH)
    p.add_argument('--run-dir',type=Path,default=Path('runs/mission_8000'))
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--layouts',type=int,default=2048)
    p.add_argument('--steps',type=int,default=2400)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--freeze-steps',type=int,default=400)
    p.add_argument('--dagger-rounds',type=int,default=2)
    p.add_argument('--dagger-layouts',type=int,default=256)
    p.add_argument('--dagger-steps',type=int,default=1200)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=29)
    p.add_argument('--warm-start',type=Path,default=Path('runs/obstacles_8000/best.pt'))
    p.add_argument('--resume',type=Path)
    p.add_argument('--scratch',action='store_true',help='Initialize gains instead of loading warm-start weights')
    p.add_argument('--size',type=int,default=5)
    p.add_argument('--obstacles',type=int,default=6)
    p.add_argument('--max-steps',type=int,default=96)
    p.add_argument('--minimum-distance',type=int,default=3)
    args=p.parse_args()
    if min(args.layouts,args.steps,args.batch_size,args.dagger_layouts,args.dagger_steps)<1 or args.dagger_rounds<0:
        p.error('Positive counts required; rounds may be zero')
    if args.lr<=0 or args.freeze_steps<0:p.error('Learning rate must be positive and freeze steps nonnegative')
    if args.layouts+args.dagger_rounds*args.dagger_layouts>=40000:p.error('Preserve held-out seed ranges')
    if args.resume and args.resume.parent.resolve()!=args.run_dir.resolve():p.error('Resume within the original run directory')
    torch.manual_seed(args.seed);torch.set_num_threads(6)
    rng=np.random.default_rng(args.seed);device=resolve_device(args.device)
    spec=MissionSpec(args.size,args.obstacles,args.max_steps,args.minimum_distance)
    model=MemoryConnectomePolicy(args.graph,spec=spec).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    graph_hash=hashlib.sha256(args.graph.read_bytes()).hexdigest()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if (args.run_dir/'latest.pt').exists() and not args.resume:raise FileExistsError('Use another run directory or --resume')
    best_key=(-1.,-1.,-999.);step=0;next_seed=100000
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device,weights_only=False)
        if checkpoint['graph_sha256']!=graph_hash:raise ValueError('Graph mismatch')
        if checkpoint['config'].get('mission',asdict(DEFAULT_SPEC))!=asdict(spec):raise ValueError('Mission config mismatch')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state=checkpoint['rng'];step=checkpoint['step'];next_seed=checkpoint['next_seed']
        best_key=tuple(checkpoint['best_key'])
        with np.load(args.run_dir/checkpoint['data_file']) as archive:data={k:archive[k] for k in archive.files}
    else:
        if not args.scratch:
            warm=torch.load(args.warm_start,map_location=device,weights_only=False)
            if warm['graph_sha256']!=graph_hash:raise ValueError('Warm-start graph mismatch; use --scratch for a new graph')
            model.edge_log_gain.data.copy_(warm['model']['edge_log_gain'])
        data=make_demonstrations(model,device,args.layouts,next_seed);next_seed+=args.layouts
    for group in optimizer.param_groups:group['lr']=args.lr
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(device=device,graph_sha256=graph_hash,stage='hidden_mission',observations=spec.observations,mission=asdict(spec),neurons=model.n,
                  development_seed=150000,development_episodes=64,
                  memory='Learned home-write / visit-max neural banks; engineered gates; online deployment',
                  navigation='Four cardinal grid moves; observed legal-action mask; no inference planner')
    (args.run_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    baseline,_,_=rollout(model,device,150000,count=64)
    print('Before:',json.dumps({k:v for k,v in baseline.items() if k not in ('episodes_detail','failed_seeds')}),flush=True)
    if not args.resume:(args.run_dir/'untrained_evaluation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    initial_gains=model.edge_log_gain.detach().clone();started=time.perf_counter()
    for round_index in range(args.dagger_rounds+1):
        if round_index:
            if next_seed+args.dagger_layouts>=140000:
                raise RuntimeError('Training seed range exhausted; preserve development and test layouts')
            additions=[]
            for offset in range(0,args.dagger_layouts,64):
                _,_,chunk=rollout(model,device,next_seed+offset,count=min(64,args.dagger_layouts-offset),collect=True)
                additions.append(chunk)
            next_seed+=args.dagger_layouts
            for key in data:data[key]=np.concatenate([data[key]]+[c[key] for c in additions])
        data_file=f'histories_{next_seed}.npz'
        np.savez_compressed(args.run_dir/data_file,**data)
        masks=legal_mask(data['x'],spec)
        updates=args.steps if round_index==0 else args.dagger_steps
        print(f'Round {round_index}; histories {len(data["y"])}',flush=True)
        for local in range(1,updates+1):
            step+=1;model.train();model.edge_log_gain.requires_grad_(step>args.freeze_steps)
            index=rng.integers(len(data['y']),size=args.batch_size)
            obs=torch.from_numpy(data['x'][index]).to(device)
            bases=torch.from_numpy(data['base'][index]).long().to(device)
            visited=torch.from_numpy(data['visited'][index]).to(device)
            state=model.state_from_history(bases,visited)
            optimizer.zero_grad(set_to_none=True)
            logits,_,home_logits,visit_logits=model.step(obs,state)
            logits=logits.masked_fill(~torch.from_numpy(masks[index]).to(device),-1e9)
            action_loss=F.cross_entropy(logits,torch.from_numpy(data['y'][index]).long().to(device))
            home_loss=F.cross_entropy(home_logits,bases)
            visit_loss=F.binary_cross_entropy_with_logits(visit_logits,visited.float())
            loss=action_loss+.15*home_loss+.1*visit_loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if local%100==0:
                print(json.dumps({'step':step,'round':round_index,'action_loss':float(action_loss.detach()),
                                  'home_loss':float(home_loss.detach()),'visit_loss':float(visit_loss.detach()),
                                  'seconds':time.perf_counter()-started}),flush=True)
            if local%400==0 or local==updates:
                report,_,_=rollout(model,device,150000,count=64)
                key=(report['mission_success_rate'],report['collected']/64,-report['mean_decisions'])
                improved=key>best_key
                if improved:best_key=key
                checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,
                            'step':step,'next_seed':next_seed,'best_key':best_key,'data_file':data_file,
                            'graph_sha256':graph_hash,'ticks':model.ticks,'config':config}
                tmp=args.run_dir/'latest.tmp';torch.save(checkpoint,tmp);tmp.replace(args.run_dir/'latest.pt')
                if improved:
                    tmp=args.run_dir/'best.tmp';torch.save(checkpoint,tmp);tmp.replace(args.run_dir/'best.pt')
                entry={'step':step,'development':report,'best':improved,'seconds':time.perf_counter()-started}
                with (args.run_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                print('Development:',json.dumps({'step':step,'success':report['mission_success_rate'],'collected':report['collected'],'best':improved,'seconds':entry['seconds']}),flush=True)
        best=torch.load(args.run_dir/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(best['model']);optimizer.load_state_dict(best['optimizer'])
        for group in optimizer.param_groups:group['lr']=args.lr
    delta=(model.edge_log_gain.detach()-initial_gains).abs()
    summary={'seconds':time.perf_counter()-started,'best_development_success_rate':best_key[0],
             'histories':len(data['y']),'next_training_seed':next_seed,
             'selected_policy_edges_changed_since_start':int((delta>1e-7).sum()),
             'graph_sha256':graph_hash}
    if device=='mps':summary['mps_driver_allocated_bytes_at_end']=torch.mps.driver_allocated_memory()
    (args.run_dir/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
