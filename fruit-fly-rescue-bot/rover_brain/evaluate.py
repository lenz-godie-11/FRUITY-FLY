"""Evaluate a saved checkpoint on new episodes without further training."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .prepare import DEFAULT_GRAPH
from .model import ConnectomePolicy
from .train import evaluate, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/beacon_4000/checkpoint.pt'))
    p.add_argument('--graph', type=Path, default=DEFAULT_GRAPH)
    p.add_argument('--output-dir', type=Path, default=Path('runs/beacon_4000/fresh_evaluation'))
    p.add_argument('--episodes', type=int, default=128)
    p.add_argument('--seed', type=int, default=20000)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be positive')
    device = resolve_device(args.device)
    torch.set_num_threads(6)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    graph_hash = hashlib.sha256(args.graph.read_bytes()).hexdigest()
    if graph_hash != checkpoint['graph_sha256']:
        raise ValueError('Checkpoint was trained on a different graph')
    model = ConnectomePolicy(args.graph, checkpoint['ticks']).to(device)
    model.load_state_dict(checkpoint['model'])
    report, replay = evaluate(model, device, args.seed, args.episodes, record=True)
    silenced, _ = evaluate(model, device, args.seed, args.episodes, silence=True)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result = {'trained': report, 'silenced_activity': silenced, 'graph_sha256': graph_hash,
              'checkpoint_sha256': checkpoint_hash, 'device': device, 'stage': 'visible_beacon'}
    replay.update(graph_sha256=graph_hash, checkpoint_sha256=checkpoint_hash)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output_dir / 'replay.json').write_text(json.dumps(replay) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
