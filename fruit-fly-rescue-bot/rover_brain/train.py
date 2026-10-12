"""Supervised warm start for visible-beacon steering; records held-out rollouts."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .environment import RoverBatch, training_batch, teacher, ACTION_NAMES
from .model import ConnectomePolicy
from .prepare import DEFAULT_GRAPH


def resolve_device(requested):
    if requested == 'auto':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process; run in a normal Mac terminal or choose --device cpu')
    return requested


@torch.no_grad()
def evaluate(model, device, seed=10000, episodes=32, silence=False, expert=False, record=False):
    model.eval()
    env = RoverBatch(count=episodes, seed=seed)
    weights = model.weights()
    frames = []
    while not env.done.all():
        observation = env.observation()
        logits, activity = model(torch.from_numpy(observation).to(device), weights=weights,
                                 silence=silence, return_activity=True)
        actions = teacher(observation) if expert else logits.argmax(1).cpu().numpy()
        if record and not env.done[0]:
            frames.append({'step': int(env.steps[0]), 'position': env.position[0].tolist(),
                           'heading': float(env.heading[0]), 'observation': observation[0].tolist(),
                           'action': int(actions[0]), 'probabilities': logits[0].softmax(0).cpu().tolist(),
                           'activity': activity[0].cpu().numpy().round(5).tolist()})
        env.step(actions)
    report = {'episodes': episodes, 'seed': seed, 'successes': int(env.success.sum()),
              'success_rate': float(env.success.mean()), 'mean_steps': float(env.steps.mean()),
              'max_steps': env.max_steps}
    replay = None
    if record:
        replay = {'stage': 'visible_beacon', 'seed': seed, 'episode_index': 0,
                  'action_names': ACTION_NAMES, 'base': env.base[0].tolist(), 'target': env.target[0].tolist(),
                  'success': bool(env.success[0]), 'frames': frames,
                  'final_position': env.position[0].tolist(), 'final_heading': float(env.heading[0]),
                  'activity_note': 'Signed rate-model activations in graph neuron-ID order; not biological recordings.'}
    return report, replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, default=DEFAULT_GRAPH)
    p.add_argument('--run-dir', type=Path, default=Path('runs/beacon_4000'))
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--eval-episodes', type=int, default=32)
    p.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    p.add_argument('--ticks', type=int, default=6)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_episodes < 1 or args.ticks < 2:
        p.error('Positive steps, batch size and episodes, and at least two ticks required')
    device = resolve_device(args.device)
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ConnectomePolicy(args.graph, ticks=args.ticks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    graph_hash = hashlib.sha256(args.graph.read_bytes()).hexdigest()
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt['graph_sha256'] != graph_hash or ckpt['ticks'] != args.ticks:
            raise ValueError('Checkpoint graph or tick count does not match')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        rng.bit_generator.state = ckpt['numpy_rng']
        start_step = ckpt['step']
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / 'checkpoint.pt').exists() and not args.resume:
        raise FileExistsError('Run already has a checkpoint; use --resume or a different --run-dir')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=device, torch_version=torch.__version__, platform=platform.platform(),
                  graph_sha256=graph_hash, trainable_parameters=sum(x.numel() for x in model.parameters()),
                  method='Teacher imitation, visible beacon; all measured edge gains trainable; no carry-over memory')
    (args.run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config, indent=2), flush=True)
    baseline, _ = evaluate(model, device, args.eval_seed, args.eval_episodes)
    print('Before training:', baseline, flush=True)
    x_val, y_val = training_batch(np.random.default_rng(90001), 1024)
    x_val, y_val = torch.from_numpy(x_val).to(device), torch.from_numpy(y_val).to(device)
    start = time.perf_counter()
    metrics = []
    initial_gain = model.edge_log_gain.detach().clone()
    for step in range(start_step + 1, start_step + args.steps + 1):
        model.train()
        x, y = training_batch(rng, args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        logits = model(torch.from_numpy(x).to(device))
        loss = F.cross_entropy(logits, torch.from_numpy(y).to(device))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == start_step + 1 or step % 50 == 0 or step == start_step + args.steps:
            model.eval()
            with torch.no_grad():
                weights = model.weights()
                correct = sum((model(xb, weights=weights).argmax(1) == yb).sum().item()
                              for xb, yb in zip(x_val.split(64), y_val.split(64)))
            entry = {'step': step, 'loss': float(loss.detach()), 'validation_accuracy': correct / len(y_val),
                     'elapsed_seconds': time.perf_counter() - start}
            metrics.append(entry)
            print(json.dumps(entry), flush=True)
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'step': step, 'ticks': args.ticks, 'graph_sha256': graph_hash,
                          'numpy_rng': rng.bit_generator.state, 'config': config}
            tmp = args.run_dir / 'checkpoint.tmp'
            torch.save(checkpoint, tmp)
            tmp.replace(args.run_dir / 'checkpoint.pt')
            with (args.run_dir / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(entry) + '\n')
    trained, replay = evaluate(model, device, args.eval_seed, args.eval_episodes, record=True)
    silenced, _ = evaluate(model, device, args.eval_seed, args.eval_episodes, silence=True)
    expert, _ = evaluate(model, device, args.eval_seed, args.eval_episodes, expert=True)
    delta = (model.edge_log_gain.detach() - initial_gain).abs()
    results = {'before_training': baseline, 'trained': trained, 'silenced_activity': silenced,
               'teacher': expert, 'mean_absolute_edge_log_gain_change': float(delta.mean()),
               'edges_changed': int((delta > 1e-7).sum()), 'training_seconds': metrics[-1]['elapsed_seconds'],
               'graph_sha256': graph_hash,
               'limitations': ['Stage 1 visible beacon only', 'Imitation learning, not reinforcement learning',
                               'No obstacles, hidden-target search, return-to-base, or temporal memory yet',
                               'No matched trained MLP or rewired baseline yet; no claim wiring is superior']}
    if device == 'mps':
        results['mps_allocated_bytes_at_end'] = torch.mps.current_allocated_memory()
        results['mps_driver_allocated_bytes_at_end'] = torch.mps.driver_allocated_memory()
    replay['graph_sha256'] = graph_hash
    replay['checkpoint_sha256'] = hashlib.sha256((args.run_dir / 'checkpoint.pt').read_bytes()).hexdigest()
    (args.run_dir / 'evaluation.json').write_text(json.dumps(results, indent=2) + '\n')
    (args.run_dir / 'replay.json').write_text(json.dumps(replay) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
