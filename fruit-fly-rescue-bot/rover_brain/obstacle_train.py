"""Learn obstacle navigation by demonstrations and policy-state relabelling.

No path planner or action correction is used in learned-policy evaluation.
Development layouts select the checkpoint; final test seeds are separate.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import ConnectomePolicy
from .obstacles import ObstacleBatch, teachers, teacher_actions, OBSERVATIONS, ACTION_NAMES
from .prepare import ROOT
from .train import resolve_device

GRAPH = ROOT / 'data/processed/malecns_8000/graph.npz'


def stats(env, seed):
    solved = env.success
    ratio = env.path_length / np.linalg.norm(env.target-env.base,axis=1)
    return {'episodes': len(solved), 'seed': seed, 'successes': int(solved.sum()),
            'collisions': int(env.collided.sum()), 'timeouts': int((~solved & ~env.collided).sum()),
            'success_rate': float(solved.mean()), 'mean_steps': float(env.steps.mean()),
            'mean_successful_path_ratio': float(ratio[solved].mean()) if solved.any() else None,
            'collision_layout_seeds': (np.flatnonzero(env.collided)+seed).tolist(),
            'timeout_layout_seeds': (np.flatnonzero(~solved & ~env.collided)+seed).tolist(),
            'max_steps': env.max_steps}


@torch.no_grad()
def rollout(model, device, seed, count=32, obstacles=6, mode='policy', record=False,
            collect=False, teacher_probability=0., rng=None):
    model.eval()
    env = ObstacleBatch(count=count, seed=seed, obstacles=obstacles)
    # The expert is never constructed for policy/direct/silenced evaluation.
    planners = teachers(env) if collect or mode == 'teacher' else None
    weights = model.weights() if mode in ['policy', 'silenced'] else None
    x, y, frames = [], [], []
    while not env.done.all():
        obs = env.observation()
        labels = teacher_actions(env, planners) if planners is not None else None
        if mode == 'teacher':
            action = labels
        elif mode == 'direct':
            angle = np.arctan2(obs[:, -6], obs[:, -5])
            action = np.where(angle > .16, 1, np.where(angle < -.16, 2, 0))
        else:
            logits, activity = model(torch.from_numpy(obs).to(device), weights=weights,
                                     silence=mode == 'silenced', return_activity=True)
            action = logits.argmax(1).cpu().numpy()
        if collect:
            active = ~env.done
            x.append(obs[active]); y.append(labels[active])
            if teacher_probability:
                if rng is None:
                    raise ValueError('Teacher mixing requires an explicit RNG')
                action = np.where(rng.random(count) < teacher_probability, labels, action)
        if record and not env.done[0]:
            if mode not in ['policy', 'silenced']:
                raise ValueError('Neural replay requires a learned-policy mode')
            frames.append({'step': int(env.steps[0]), 'position': env.position[0].tolist(),
                           'heading': float(env.heading[0]), 'observation': obs[0].tolist(),
                           'action': int(action[0]), 'probabilities': logits[0].softmax(0).cpu().tolist(),
                           'activity': activity[0].cpu().numpy().round(5).tolist()})
        env.step(action)
    replay = None
    if record:
        replay = {'stage': 'obstacles', 'seed': seed, 'episode_index': 0, 'action_names': ACTION_NAMES,
                  'base': env.base[0].tolist(), 'target': env.target[0].tolist(),
                  'obstacles': env.layouts[0].boxes.tolist(), 'rover_radius': .34,
                  'success': bool(env.success[0]), 'collided': bool(env.collided[0]), 'frames': frames,
                  'final_position': env.position[0].tolist(), 'final_heading': float(env.heading[0]),
                  'activity_note': 'Decision-time rate-model activity; geometry interpolated only for playback.'}
    dataset = (np.concatenate(x), np.concatenate(y)) if collect else None
    return stats(env,seed), replay, dataset


def collect_demonstrations(model, device, seed, layouts, obstacles):
    xs, ys, solved = [], [], 0
    for offset in range(0, layouts, 32):
        report, _, data = rollout(model,device,seed+offset,count=min(32,layouts-offset),
                                  obstacles=obstacles,mode='teacher',collect=True)
        xs.append(data[0]); ys.append(data[1]); solved += report['successes']
    print(f'Teacher demonstrations: {solved}/{layouts} solved', flush=True)
    if solved < layouts*.95:
        raise RuntimeError('Teacher solved fewer than 95% of generated layouts; inspect geometry before training')
    return np.concatenate(xs), np.concatenate(ys)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, default=GRAPH)
    p.add_argument('--run-dir', type=Path, default=Path('runs/obstacles_8000'))
    p.add_argument('--device', choices=['auto','cpu','mps'], default='auto')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--teacher-layouts', type=int, default=320)
    p.add_argument('--obstacles', type=int, default=6)
    p.add_argument('--steps', type=int, default=1600, help='Optimizer updates in the initial imitation round')
    p.add_argument('--dagger-rounds', type=int, default=3)
    p.add_argument('--dagger-layouts', type=int, default=96)
    p.add_argument('--dagger-steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=96)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--ticks', type=int, default=6)
    p.add_argument('--resume', type=Path)
    p.add_argument('--restore-best', action='store_true', help='On resume, refine best weights using the latest recovery pool')
    p.add_argument('--forward-fraction', type=float, default=0., help='Optional forward-action fraction per batch; zero uses pool frequencies')
    args = p.parse_args()
    if min(args.teacher_layouts,args.steps,args.batch_size,args.dagger_layouts,args.dagger_steps) < 1 or args.dagger_rounds < 0:
        p.error('Counts must be positive; rounds may be zero')
    if not 1 <= args.obstacles <= 10:
        p.error('--obstacles must be between 1 and 10')
    if args.ticks < 2 or args.lr <= 0 or args.teacher_layouts + 1000 >= 40000:
        p.error('Use at least two ticks, positive learning rate, and training layout seeds below 40000')
    if not 0 <= args.forward_fraction < 1 or (args.restore_best and not args.resume):
        p.error('Use a forward fraction in [0,1); --restore-best requires --resume')
    if args.resume and args.resume.parent.resolve() != args.run_dir.resolve():
        p.error('--run-dir must be the directory containing the resumed checkpoint')
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    model = ConnectomePolicy(args.graph, ticks=args.ticks, observations=OBSERVATIONS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    graph_hash = hashlib.sha256(args.graph.read_bytes()).hexdigest()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.run_dir / 'training_data.npz'
    total_steps, next_seed, best_key = 0, 1000, (-1., -999, -999.)
    if (args.run_dir/'latest.pt').exists() and not args.resume:
        raise FileExistsError('Run already exists; use --resume or another --run-dir')
    if args.resume:
        ckpt = torch.load(args.resume,map_location=device,weights_only=False)
        if ckpt['graph_sha256'] != graph_hash or ckpt['observations'] != OBSERVATIONS or ckpt['ticks'] != args.ticks:
            raise ValueError('Checkpoint graph/observation/tick mismatch')
        if ckpt['config']['obstacles'] != args.obstacles:
            raise ValueError('Resuming with different obstacle count is not supported')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
        rng.bit_generator.state = ckpt['rng']
        total_steps, next_seed, best_key = ckpt['step'],ckpt['next_seed'],tuple(ckpt['best_key'])
        data_path = args.resume.parent / ckpt.get('training_data_file', 'training_data.npz')
        with np.load(data_path) as data:
            x, y = data['x'],data['y']
        if args.restore_best:
            best = torch.load(args.run_dir/'best.pt',map_location=device,weights_only=False)
            if (best['graph_sha256'],best['observations'],best['ticks']) != (graph_hash,OBSERVATIONS,args.ticks):
                raise ValueError('Best checkpoint does not match the current graph and model')
            model.load_state_dict(best['model']);optimizer.load_state_dict(best['optimizer'])
            for group in optimizer.param_groups:group['lr']=args.lr
        # Make recovery data immutable and local to this run for future resumes.
        local_data = args.run_dir/f'training_data_{next_seed}.npz'
        if local_data != data_path:
            np.savez_compressed(local_data,x=x,y=y)
            data_path=local_data
    else:
        x,y = collect_demonstrations(model,device,next_seed,args.teacher_layouts,args.obstacles)
        next_seed += args.teacher_layouts
        data_path = args.run_dir / f'training_data_{next_seed}.npz'
        np.savez_compressed(data_path,x=x,y=y)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(device=device,observations=OBSERVATIONS,stage='obstacles',graph_sha256=graph_hash,
                  torch_version=torch.__version__,method='Imitation + DAgger; no inference planner',
                  development_seed=50000,development_episodes=32)
    (args.run_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config),flush=True)
    baseline,_,_=rollout(model,device,50000,obstacles=args.obstacles)
    print('Before:',json.dumps(baseline),flush=True)
    if not args.resume:
        (args.run_dir/'untrained_evaluation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    initial = model.edge_log_gain.detach().clone()
    started = time.perf_counter()
    for round_index in range(args.dagger_rounds+1):
        if round_index:
            if next_seed+args.dagger_layouts >= 40000:
                raise RuntimeError('Training seed range exhausted; preserve held-out seeds')
            additions_x, additions_y = [], []
            for offset in range(0,args.dagger_layouts,32):
                report,_,data=rollout(model,device,next_seed+offset,count=min(32,args.dagger_layouts-offset),
                                      obstacles=args.obstacles,collect=True,
                                      teacher_probability=.15 if round_index == 1 else 0.,rng=rng)
                additions_x.append(data[0]); additions_y.append(data[1])
            next_seed += args.dagger_layouts
            new_x,new_y = np.concatenate(additions_x),np.concatenate(additions_y)
            # Revisit recovery examples more often than the easy demonstrations.
            x=np.concatenate([x,new_x,new_x]); y=np.concatenate([y,new_y,new_y])
            data_path = args.run_dir / f'training_data_{next_seed}.npz'
            np.savez_compressed(data_path,x=x,y=y)
            print(f'DAgger round {round_index}: added {len(new_y)} on-policy labels; pool={len(y)}',flush=True)
        updates = args.steps if round_index == 0 else args.dagger_steps
        class_indices=[np.flatnonzero(y==action) for action in range(3)]
        for local_step in range(1,updates+1):
            model.train()
            if args.forward_fraction:
                forward_count=int(args.batch_size*args.forward_fraction)
                left_count=(args.batch_size-forward_count)//2
                counts=[forward_count,left_count,args.batch_size-forward_count-left_count]
                if any(len(indices)==0 for indices in class_indices):
                    raise ValueError('Action-balanced sampling requires all three classes')
                selected=np.concatenate([rng.choice(indices,size=count) for indices,count in zip(class_indices,counts)])
                rng.shuffle(selected)
            else:
                selected=rng.integers(len(y),size=args.batch_size)
            xb=torch.from_numpy(x[selected]).to(device)
            yb=torch.from_numpy(y[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(xb)
            loss=F.cross_entropy(logits,yb)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            total_steps += 1
            if local_step % 100 == 0 or local_step == updates:
                print(json.dumps({'step':total_steps,'round':round_index,'loss':float(loss.detach()),
                                  'seconds':time.perf_counter()-started}),flush=True)
            if local_step % 400 == 0 or local_step == updates:
                report,_,_=rollout(model,device,50000,obstacles=args.obstacles)
                key=(report['success_rate'],-report['collisions'],-report['mean_steps'])
                is_best=key>best_key
                if is_best: best_key=key
                state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':total_steps,
                       'ticks':args.ticks,'observations':OBSERVATIONS,'graph_sha256':graph_hash,
                       'training_data_file':data_path.name,
                       'rng':rng.bit_generator.state,'next_seed':next_seed,'best_key':best_key,'config':config}
                tmp=args.run_dir/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(args.run_dir/'latest.pt')
                if is_best:
                    tmp=args.run_dir/'best.tmp';torch.save(state,tmp);tmp.replace(args.run_dir/'best.pt')
                entry={'step':total_steps,'round':round_index,'loss':float(loss.detach()),'development':report,
                       'best':is_best,'seconds':time.perf_counter()-started}
                with (args.run_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                print('Development:',json.dumps(entry),flush=True)
        # Continue DAgger from the best policy rather than a regressed final batch.
        state=torch.load(args.run_dir/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.param_groups: group['lr']=args.lr
    delta=(model.edge_log_gain.detach()-initial).abs()
    summary={'training_seconds':time.perf_counter()-started,'optimizer_updates_this_invocation':args.steps+args.dagger_rounds*args.dagger_steps,
             'training_examples_in_pool':len(y),'next_training_layout_seed':next_seed,
             'edges_changed':int((delta>1e-7).sum()),'mean_absolute_edge_log_gain_change':float(delta.mean()),
             'best_development_success_rate':best_key[0], 'graph_sha256':graph_hash}
    if device=='mps':summary['mps_driver_allocated_bytes_at_end']=torch.mps.driver_allocated_memory()
    (args.run_dir/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
