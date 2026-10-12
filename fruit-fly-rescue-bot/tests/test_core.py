import json
from pathlib import Path
import numpy as np
import pytest
import torch
from rover_brain.environment import RoverBatch, observe, teacher, training_batch
from rover_brain.model import ConnectomePolicy
from rover_brain.prepare import DEFAULT_GRAPH, reachable
from scipy import sparse


@pytest.fixture
def tiny_graph(tmp_path):
    path = tmp_path / 'graph.npz'
    np.savez(path, neuron_ids=np.array([100, 200, 300, 400], dtype='int64'),
             pre=np.array([0, 1]), post=np.array([1, 2]), base_weight=np.array([.8, -.8], 'float32'),
             input_indices=np.array([0]), output_indices=np.array([2, 3]))
    return path


def test_propagates_pre_to_post_and_preserves_inhibitory_sign(tiny_graph):
    model = ConnectomePolicy(tiny_graph, ticks=3)
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.fill_(1)
    _, activity = model(torch.zeros(1, 11), return_activity=True)
    assert activity[0, 0] > 0
    assert activity[0, 1] > 0
    assert activity[0, 2] < 0
    assert activity[0, 3] == 0
    _, early = ConnectomePolicy(tiny_graph, ticks=2)(torch.zeros(1, 11), return_activity=True)
    assert early[0, 2] == 0  # Two-edge path requires three update ticks.


def test_measured_edges_receive_gradients_without_new_connections(tiny_graph):
    model = ConnectomePolicy(tiny_graph, ticks=4)
    _, activity = model(torch.ones(2, 11), return_activity=True)
    activity[:, 2].sum().backward()
    assert torch.isfinite(model.edge_log_gain.grad).all()
    assert torch.count_nonzero(model.edge_log_gain.grad) == 2
    assert model.weights()._nnz() == 2
    _, silenced = model(torch.ones(2, 11), silence=True, return_activity=True)
    assert torch.count_nonzero(silenced) == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU not visible')
def test_mps_dense_and_cpu_sparse_agree(tiny_graph):
    torch.manual_seed(42)
    cpu = ConnectomePolicy(tiny_graph, ticks=4)
    gpu = ConnectomePolicy(tiny_graph, ticks=4).to('mps')
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(8, 11)
    a, ha = cpu(x, return_activity=True)
    b, hb = gpu(x.to('mps'), return_activity=True)
    torch.testing.assert_close(a, b.cpu(), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ha, hb.cpu(), atol=1e-6, rtol=1e-4)
    ha[:, 2].sum().backward()
    hb[:, 2].sum().backward()
    torch.testing.assert_close(cpu.edge_log_gain.grad, gpu.edge_log_gain.grad.cpu(), atol=1e-6, rtol=1e-4)


def test_geometry_turns_and_ranges():
    obs = observe(np.array([[0., 0.]]), np.array([0.]), np.array([[0., 2.]]))
    assert teacher(obs)[0] == 1
    assert np.all((obs[:, :8] >= 0) & (obs[:, :8] <= 1))
    env = RoverBatch(count=1)
    env.position[:] = 0
    env.heading[:] = 0
    env.target[:] = [0, 2]
    env.step(np.array([1]))
    assert env.heading[0] > 0
    assert np.all(env.position == 0)


def test_teacher_solves_heldout_env_and_done_agents_stop():
    env = RoverBatch(count=64, seed=456)
    while not env.done.all():
        env.step(teacher(env.observation()))
    assert env.success.all()
    final = env.position.copy()
    env.step(np.zeros(64, dtype=int))
    np.testing.assert_array_equal(final, env.position)


def test_seeded_training_data():
    a, y = training_batch(np.random.default_rng(7), 1000)
    b, z = training_batch(np.random.default_rng(7), 1000)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(y, z)
    assert set(y) == {0, 1, 2}
    assert np.isfinite(a).all()


@pytest.mark.skipif(not DEFAULT_GRAPH.exists(), reason='Run connectome preparation first')
def test_prepared_real_connectome_integrity():
    with np.load(DEFAULT_GRAPH) as g:
        n = len(g['neuron_ids'])
        assert n == 4000
        assert len(np.unique(g['neuron_ids'])) == n
        assert g['neuron_ids'].dtype == np.int64
        assert (g['synapse_count'] > 0).all()
        assert (g['pre'] != g['post']).all()
        assert len(np.unique(g['pre'].astype('int64') * n + g['post'])) == len(g['pre'])
        assert not np.intersect1d(g['input_indices'], g['output_indices']).size
        assert np.isfinite(g['base_weight']).all()
        a = sparse.csr_matrix((np.ones(len(g['pre'])), (g['pre'], g['post'])), shape=(n, n))
        assert (reachable(a, g['input_indices']) >= 0).all()
        assert (reachable(a, g['output_indices'], reverse=True) >= 0).all()
        np.testing.assert_array_equal(np.sign(g['base_weight']), g['source_sign'][g['pre']])
