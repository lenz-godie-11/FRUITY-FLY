import numpy as np
import pytest
import torch
from rover_brain.obstacles import (ObstacleBatch, occupied, segment_blocked, ranges,
                                  make_layout, teachers, teacher_actions, OBSERVATIONS, RADIUS)
from rover_brain.prepare import ROOT, reachable
from scipy import sparse


def test_swept_collision_catches_thin_wall_and_parallel_clearance():
    box=np.array([[0.,-1.,.02,1.]])
    assert segment_blocked([-1,0],[1,0],box)
    assert not segment_blocked([-1,2],[1,2],box)
    assert segment_blocked([0,1+RADIUS],[1,1+RADIUS],box)
    assert segment_blocked([3.5,0],[3.9,0],np.empty((0,4)))
    np.testing.assert_array_equal(segment_blocked([[-1,0],[-1,2]],[[1,0],[1,2]],box),[True,False])


def test_range_sensor_sees_inflated_obstacle_not_wall():
    distances=ranges(np.array([-2.,0.]),0.,np.array([[0.,-1.,1.,1.]]))
    assert distances[16]==pytest.approx(2-RADIUS)
    assert distances[0]==pytest.approx(2-RADIUS)
    assert np.isfinite(distances).all()


def test_collision_terminates_episode_without_tunnelling():
    env=ObstacleBatch(count=1,seed=100)
    env.layouts[0].boxes=np.array([[0.,-1.,1.,1.]])
    env.position[0]=[-RADIUS-.05,0]
    env.heading[0]=0
    before=env.position.copy()
    env.step(np.array([0]))
    assert env.collided[0] and env.done[0] and not env.success[0]
    np.testing.assert_array_equal(env.position,before)
    env.step(np.array([1]))
    assert env.steps[0]==1
    assert env.heading[0]==0


def test_layout_seed_independent_of_batch_size_and_direct_path_blocked():
    a=ObstacleBatch(count=1,seed=321)
    b=ObstacleBatch(count=3,seed=321)
    np.testing.assert_array_equal(a.layouts[0].boxes,b.layouts[0].boxes)
    np.testing.assert_array_equal(a.observation()[0],b.observation()[0])
    assert segment_blocked(a.base[0],a.target[0],a.layouts[0].boxes)
    assert not occupied(np.array([a.base[0],a.target[0]]),a.layouts[0].boxes).any()
    assert a.observation().shape==(1,OBSERVATIONS)
    assert np.isfinite(a.observation()).all()


def test_teacher_navigates_varied_layouts_without_collision():
    env=ObstacleBatch(count=16,seed=61234,obstacles=8)
    planners=teachers(env)
    while not env.done.all():env.step(teacher_actions(env,planners))
    assert env.success.all()
    assert not env.collided.any()


def test_policy_rollout_does_not_request_expert_actions(monkeypatch):
    import rover_brain.obstacle_train as training
    def forbidden(*args,**kwargs):raise AssertionError('Teacher used in policy inference')
    monkeypatch.setattr(training,'teachers',forbidden)
    monkeypatch.setattr(training,'teacher_actions',forbidden)
    class Policy(torch.nn.Module):
        def weights(self):return None
        def forward(self,x,**kwargs):
            return torch.tensor([[1.,0,0]]).repeat(len(x),1),torch.zeros(len(x),4)
    report,_,_=training.rollout(Policy(),'cpu',seed=61234,count=2,mode='policy')
    assert report['episodes']==2


@pytest.mark.skipif(not (ROOT/'data/processed/malecns_8000/graph.npz').exists(),reason='Prepare 8000-neuron graph')
def test_expanded_graph_preserves_old_neurons_and_all_directed_paths():
    with np.load(ROOT/'data/processed/malecns_8000/graph.npz') as g, np.load(ROOT/'data/processed/malecns_4000/graph.npz') as old:
        ids=g['neuron_ids']
        assert len(ids)==8000 and len(np.unique(ids))==8000
        assert np.isin(old['neuron_ids'],ids).all()
        a=sparse.csr_matrix((np.ones(len(g['pre'])),(g['pre'],g['post'])),shape=(8000,8000))
        assert (reachable(a,g['input_indices'])>=0).all()
        assert (reachable(a,g['output_indices'],reverse=True)>=0).all()
        np.testing.assert_array_equal(np.sign(g['base_weight']),g['source_sign'][g['pre']])
