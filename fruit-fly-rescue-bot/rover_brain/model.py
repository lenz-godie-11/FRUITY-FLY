"""Trainable rate network restricted to measured pre -> post connections."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .environment import OBSERVATIONS


class ConnectomePolicy(nn.Module):
    def __init__(self, graph_path: str | Path, ticks=6, observations=OBSERVATIONS, actions=3):
        super().__init__()
        with np.load(graph_path, allow_pickle=False) as graph:
            self.n = len(graph['neuron_ids'])
            self.register_buffer('pre', torch.from_numpy(graph['pre'].astype('int64')))
            self.register_buffer('post', torch.from_numpy(graph['post'].astype('int64')))
            self.register_buffer('base_weight', torch.from_numpy(graph['base_weight']))
            self.register_buffer('inputs', torch.from_numpy(graph['input_indices'].astype('int64')))
            self.register_buffer('outputs', torch.from_numpy(graph['output_indices'].astype('int64')))
        if torch.isin(self.inputs, self.outputs).any():
            raise ValueError('Sensory inputs and descending readouts must be disjoint')
        self.ticks = ticks
        self.edge_log_gain = nn.Parameter(torch.zeros(len(self.pre)))
        self.encoder = nn.Linear(observations, len(self.inputs))
        self.readout = nn.Sequential(nn.LayerNorm(len(self.outputs)), nn.Linear(len(self.outputs), actions))

    def weights(self):
        # Bounded positive gains retain transmitter signs and the measured mask.
        values = self.base_weight * self.edge_log_gain.clamp(-2, 2).exp()
        if values.device.type == 'mps':
            # Dense MPS matrix: 256 MB at 8k, 1.024 GB at 16k, plus gradients.
            flat = values.new_zeros(self.n * self.n)
            return flat.scatter(0, self.pre * self.n + self.post, values).reshape(self.n, self.n)
        # CPU fallback avoids doing dense work for absent biological edges.
        return torch.sparse_coo_tensor(torch.stack([self.post, self.pre]), values,
                                       (self.n, self.n), check_invariants=True).coalesce()

    def forward(self, observations, weights=None, silence=False, return_activity=False):
        w = self.weights() if weights is None else weights
        drive = observations.new_zeros((len(observations), self.n))
        drive = drive.index_copy(1, self.inputs, self.encoder(observations))
        activity = torch.zeros_like(drive)
        # Stage 1 starts fresh per decision: no claim of temporal memory yet.
        for _ in range(self.ticks):
            recurrent = torch.sparse.mm(w, activity.T).T if w.is_sparse else activity @ w
            activity = .5 * activity + .5 * torch.tanh(recurrent + drive)
        if silence:
            activity = torch.zeros_like(activity)
        logits = self.readout(activity[:, self.outputs])
        return (logits, activity) if return_activity else logits
