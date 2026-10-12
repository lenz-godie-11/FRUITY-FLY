"""Extract a deterministic neuron subgraph from official data."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/malecns_v1'
DEFAULT_GRAPH = ROOT / 'data/processed/malecns_4000/graph.npz'


def reachable(adjacency, seeds, reverse=False):
    a = adjacency.T.tocsr() if reverse else adjacency.tocsr()
    seen = np.zeros(a.shape[0], bool)
    seen[seeds] = True
    distance = np.full(a.shape[0], -1, np.int32)
    distance[seeds] = 0
    frontier = np.asarray(seeds)
    level = 0
    while len(frontier):
        level += 1
        nxt = np.unique(a[frontier].indices)
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        distance[nxt] = level
        frontier = nxt
    return distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neurons', type=int, default=4000)
    parser.add_argument('--include-graph', type=Path, help='Retain every neuron in a previous graph when expanding')
    args = parser.parse_args()
    if args.neurons < 1000:
        parser.error('Use at least 1000 neurons for the specified role quotas')
    manifest = json.loads((RAW / 'manifest.json').read_text())
    for meta in manifest['files'].values():
        with (RAW / meta['filename']).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != meta['sha256']:
                raise ValueError(f"Source checksum changed: {meta['filename']}")
    ann = pd.read_feather(RAW / manifest['files']['annotations']['filename'])
    allowed = ['cb_intrinsic', 'visual_projection', 'cb_sensory', 'descending_neuron']
    ann = ann[ann.superclass.isin(allowed) & ann.status.eq('Traced')].sort_values('bodyId').reset_index(drop=True)
    ids = pd.Index(ann.bodyId.astype('int64'))
    pieces = []
    # Stream compressed Arrow record batches: never materialize 151M raw rows.
    with pa.memory_map(str(RAW / manifest['files']['connections']['filename']), 'r') as f:
        reader = pa.ipc.open_file(f)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pre = ids.get_indexer(batch.column('body_pre').to_numpy())
            post = ids.get_indexer(batch.column('body_post').to_numpy())
            weight = batch.column('weight').to_numpy()
            valid = (pre >= 0) & (post >= 0) & (pre != post) & (weight > 0)
            pieces.append((pre[valid].astype('int32'), post[valid].astype('int32'), weight[valid].astype('float32')))
    pre, post, weight = [np.concatenate([p[j] for p in pieces]) for j in range(3)]
    del pieces
    graph = sparse.csr_matrix((weight, (pre, post)), shape=(len(ids), len(ids)))
    graph.sum_duplicates()
    _, labels = connected_components(graph, directed=True, connection='strong')
    largest = np.bincount(labels).argmax()
    eligible = labels == largest
    strength = np.asarray(graph.sum(axis=0)).ravel() + np.asarray(graph.sum(axis=1)).ravel()
    def top(mask, n):
        candidates = np.flatnonzero(mask & eligible)
        return candidates[np.lexsort((ids.to_numpy()[candidates], -strength[candidates]))[:n]]
    # Seed complementary populations, then grow by total measured contact strength.
    seed_groups = [top(ann.superclass.eq('visual_projection').to_numpy(), 256),
                   top(ann.superclass.eq('cb_sensory').to_numpy(), 128),
                   top(ann.superclass.eq('descending_neuron').to_numpy(), 128),
                   top(ann['class'].eq('CX').to_numpy(), min(2048, args.neurons // 2))]
    selected = np.zeros(len(ids), bool)
    selected[np.concatenate(seed_groups)] = True
    included_hash = None
    if args.include_graph:
        with np.load(args.include_graph, allow_pickle=False) as previous:
            previous_indices = ids.get_indexer(previous['neuron_ids'])
        if (previous_indices < 0).any() or not eligible[previous_indices].all():
            raise ValueError('Previous graph contains neurons outside the eligible source component')
        selected[previous_indices] = True
        included_hash = hashlib.sha256(args.include_graph.read_bytes()).hexdigest()
    if selected.sum() >= args.neurons or eligible.sum() < args.neurons:
        raise ValueError('Requested size incompatible with seed quotas/eligible component')
    undirected = (graph + graph.T).tocsr()
    bias = np.where(ann['class'].eq('CX'), 2., 1.)
    score = np.asarray(undirected[:, selected].sum(axis=1)).ravel()
    heap = [(-float(score[i] * bias[i]), int(i)) for i in np.flatnonzero(eligible & ~selected & (score > 0))]
    heapq.heapify(heap)
    while selected.sum() < args.neurons:
        if not heap:
            raise RuntimeError('Connected frontier exhausted')
        neg, node = heapq.heappop(heap)
        if selected[node] or not np.isclose(-neg, score[node] * bias[node]):
            continue
        selected[node] = True
        start, end = undirected.indptr[node:node + 2]
        for nb, w in zip(undirected.indices[start:end], undirected.data[start:end]):
            if eligible[nb] and not selected[nb]:
                score[nb] += w
                heapq.heappush(heap, (-float(score[nb] * bias[nb]), int(nb)))
    neurons = ann.loc[selected].reset_index(drop=True)
    sub = graph[selected][:, selected].tocsr()
    inputs = np.flatnonzero(neurons.superclass.isin(['visual_projection', 'cb_sensory']))
    distance = reachable(sub, inputs)
    outputs = np.flatnonzero(neurons.superclass.eq('descending_neuron') & (distance > 0) & (distance <= 5))
    if len(outputs) < 16:
        raise RuntimeError('Too few descending readouts reachable in five synaptic hops')
    back = reachable(sub, outputs, reverse=True)
    nt = pd.read_feather(RAW / manifest['files']['neurotransmitters']['filename'])
    nt = nt.set_index('body').reindex(neurons.bodyId)
    neurons['neurotransmitter'] = nt.consensus_nt.to_numpy()
    # Pragmatic rate-model assumption, not receptor-specific biological ground truth.
    signs = np.where(neurons.neurotransmitter.isin(['gaba', 'glutamate']), -1., 1.).astype('float32')
    coo = sub.tocoo()
    incoming = np.asarray(sub.sum(axis=0)).ravel().clip(1)
    base_weight = (coo.data / incoming[coo.col] * signs[coo.row]).astype('float32')
    coords = np.full((len(neurons), 3), np.nan, 'float32')
    for i, value in enumerate(neurons.somaLocation):
        if value is not None and hasattr(value, '__len__') and len(value) == 3:
            coords[i] = np.asarray(value, dtype='float32') * .008
    out = ROOT / f'data/processed/malecns_{args.neurons}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'graph.npz', neuron_ids=neurons.bodyId.to_numpy(dtype='int64'),
                        pre=coo.row.astype('int32'), post=coo.col.astype('int32'),
                        synapse_count=coo.data.astype('int64'), base_weight=base_weight,
                        input_indices=inputs, output_indices=outputs, soma_um=coords,
                        source_sign=signs)
    neurons['input'] = np.isin(np.arange(len(neurons)), inputs)
    neurons['output'] = np.isin(np.arange(len(neurons)), outputs)
    neurons.drop(columns=['somaLocation', 'tosomaLocation']).to_csv(out / 'neurons.csv', index=False)
    report = {
        'dataset': 'MaleCNS v1.0', 'neurons': len(neurons), 'directed_edges': sub.nnz,
        'synaptic_contacts': int(coo.data.sum()), 'input_neurons': len(inputs), 'output_neurons': len(outputs),
        'weak_components': connected_components(sub, directed=True, connection='weak')[0],
        'strong_components': connected_components(sub, directed=True, connection='strong')[0],
        'reachable_from_inputs': int((distance >= 0).sum()),
        'on_input_to_output_paths': int(((distance >= 0) & (back >= 0)).sum()),
        'maximum_input_to_readout_shortest_path': int(distance[outputs].max()),
        'missing_soma_coordinates': int(np.isnan(coords).any(axis=1).sum()),
        'superclasses': neurons.superclass.value_counts().to_dict(),
        'neurotransmitters': neurons.neurotransmitter.fillna('unknown').value_counts().to_dict(),
        'graph_sha256': hashlib.sha256((out / 'graph.npz').read_bytes()).hexdigest(),
        'selection': f'Traced eligible brain classes; largest full candidate SCC; strongest 256 visual + 128 central sensory + 128 descending + {min(2048, args.neurons // 2)} CX seeds; retain include-graph neurons if supplied; grow by contact strength with 2x CX priority; body-ID tie break; induced edges; autapses omitted.',
        'included_graph_sha256': included_hash,
        'sign_policy': 'GABA/glutamate negative; all other/unknown transmitters positive. Modeling assumption; no receptor dynamics.',
        'weight_policy': 'Synapse count / retained incoming synapse total; presynaptic sign; pre -> post.',
        'source_manifest': manifest,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_manifest'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
