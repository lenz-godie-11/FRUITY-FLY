import numpy as np
import pytest
import torch
from rover_brain.mission import (MissionBatch,teacher_actions,teacher_action,distances,
                                POSITION,SIGNAL,DEPARTURE,CELLS,OBSERVATIONS)
from rover_brain.mission_train import legal_mask
from rover_brain.memory_model import MemoryConnectomePolicy


@pytest.fixture
def memory_graph(tmp_path):
    # Synthetic fixture only: three memory/sensory groups feed distinct readouts.
    path=tmp_path/'memory.npz'
    pre=np.arange(90)
    np.savez(path,neuron_ids=np.arange(180,dtype='int64'),pre=pre,post=pre+90,
             base_weight=np.full(90,.8,dtype='float32'),input_indices=pre,output_indices=pre+90)
    return path


def test_hidden_beacon_does_not_leak_outside_local_sensor():
    a=MissionBatch(count=1,seed=301)
    before=a.observation().copy()
    far=np.flatnonzero((distances(int(a.cell[0]),a.walls[0])>=3)&~a.walls[0])
    other=next(int(c) for c in far if c!=a.beacon[0])
    first=teacher_action(before[0],a.visited[0],a.base[0])
    a.beacon[0]=other
    np.testing.assert_array_equal(before,a.observation())
    assert teacher_action(a.observation()[0],a.visited[0],a.base[0])==first
    assert not before[0,SIGNAL].any()


def test_base_not_in_observation_after_departure():
    a=MissionBatch(count=1,seed=701)
    a.step(teacher_actions(a))
    before=a.observation().copy()
    a.base[0]=(a.base[0]+1)%CELLS
    np.testing.assert_array_equal(before,a.observation())
    assert before[0,DEPARTURE]==0


def test_full_mission_requires_pickup_and_return():
    a=MissionBatch(count=32,seed=444000)
    assert not a.success.any()
    while not a.done.all():
        actions=teacher_actions(a)
        assert legal_mask(a.observation())[np.arange(32),actions][~a.done].all()
        a.step(actions)
        assert not (a.success & ~a.carrying).any()
    assert a.carrying.all() and a.success.all()
    assert (a.cell==a.base).all()
    assert (a.pickup_step>0).all()
    assert a.collision_attempts.sum()==0
    positions=a.cell.copy();a.step(np.zeros(32,int));np.testing.assert_array_equal(a.cell,positions)


def test_online_memory_matches_training_history_recomputation(memory_graph):
    model=MemoryConnectomePolicy(memory_graph)
    env=MissionBatch(count=3,seed=500)
    state=model.initial_state(3)
    for _ in range(8):
        obs=torch.from_numpy(env.observation())
        rebuilt=model.state_from_history(torch.from_numpy(env.base),torch.from_numpy(env.visited))
        online,new,_,_=model.step(obs,state)
        reconstructed,_,_,_=model.step(obs,rebuilt)
        torch.testing.assert_close(online,reconstructed,atol=1e-6,rtol=1e-5)
        state=new
        env.step(teacher_actions(env))


def test_identical_current_input_retains_different_departure_memories(memory_graph):
    model=MemoryConnectomePolicy(memory_graph)
    obs=torch.zeros(2,OBSERVATIONS)
    obs[0,0]=1;obs[1,24]=1;obs[:,DEPARTURE]=1
    _,state,_,_=model.step(obs,model.initial_state(2))
    obs.zero_();obs[:,12]=1
    for _ in range(20):_,state,_,_=model.step(obs,state)
    assert not torch.equal(state[0,model.home_indices],state[1,model.home_indices])
    _,forgotten,_,_=model.step(obs,state,forget_home=True)
    assert torch.count_nonzero(forgotten[:,model.home_indices])==0


def test_departure_resets_old_episode_memory(memory_graph):
    model=MemoryConnectomePolicy(memory_graph)
    obs=torch.zeros(1,OBSERVATIONS);obs[0,3]=1;obs[:,DEPARTURE]=1
    dirty=torch.randn(1,model.n)
    a,sa,_,_=model.step(obs,dirty)
    b,sb,_,_=model.step(obs,model.initial_state(1))
    torch.testing.assert_close(a,b);torch.testing.assert_close(sa,sb)


def test_policy_has_no_teacher_at_inference(memory_graph,monkeypatch):
    import rover_brain.mission_train as training
    def forbidden(*args,**kwargs):raise AssertionError('Teacher called during deployment')
    monkeypatch.setattr(training,'teacher_actions',forbidden)
    result,_,_=training.rollout(MemoryConnectomePolicy(memory_graph),'cpu',seed=765,count=2)
    assert result['episodes']==2


def test_larger_world_connectivity_sensors_and_reference_rescue():
    from rover_brain.mission import MissionSpec, neighbours
    spec=MissionSpec(size=7,obstacles=16,max_steps=192,minimum_distance=5)
    env=MissionBatch(count=32,seed=450000,spec=spec)
    assert env.observation().shape==(32,110)
    assert (env.walls.sum(axis=1)==16).all()
    for i in range(32):
        dist=distances(int(env.base[i]),env.walls[i],spec)
        assert (dist[~env.walls[i]]>=0).all()
        assert dist[env.beacon[i]]>=5
    # Changing a remote beacon must not change observation or teacher choice.
    before=env.observation().copy()
    far=np.flatnonzero(distances(int(env.cell[0]),env.walls[0],spec)>=5)
    action=teacher_action(before[0],env.visited[0],env.base[0],spec)
    env.beacon[0]=next(int(c) for c in far if c!=env.beacon[0])
    np.testing.assert_array_equal(before,env.observation())
    assert teacher_action(env.observation()[0],env.visited[0],env.base[0],spec)==action
    # Beacon sensor follows dynamic offsets even at the upper end of the grid.
    cell=int(np.flatnonzero(~env.walls[0])[-1]);env.cell[0]=cell
    adjacent=neighbours(cell,env.walls[0],spec)[0]
    env.beacon[0]=adjacent[1]
    assert env.observation()[0,spec.signal][1+adjacent[0]]==1
    env=MissionBatch(count=32,seed=450000,spec=spec)
    while not env.done.all():
        actions=teacher_actions(env)
        assert legal_mask(env.observation(),spec)[np.arange(32),actions][~env.done].all()
        env.step(actions)
    assert env.success.all() and (env.cell==env.base).all()
    assert not env.collision_attempts.any()


def test_large_grid_online_memory_and_deployment(tmp_path,monkeypatch):
    from rover_brain.mission import MissionSpec
    import rover_brain.mission_train as training
    spec=MissionSpec(7,16,192,5)
    path=tmp_path/'large_memory.npz';pre=np.arange(180)
    np.savez(path,neuron_ids=np.arange(360,dtype='int64'),pre=pre,post=pre+180,
             base_weight=np.full(180,.8,dtype='float32'),input_indices=pre,output_indices=pre+180)
    model=MemoryConnectomePolicy(path,spec=spec)
    env=MissionBatch(count=2,seed=450100,spec=spec);state=model.initial_state(2)
    for _ in range(10):
        obs=torch.from_numpy(env.observation())
        history=model.state_from_history(torch.from_numpy(env.base),torch.from_numpy(env.visited))
        online,state,_,_=model.step(obs,state)
        rebuilt,_,_,_=model.step(obs,history)
        torch.testing.assert_close(online,rebuilt)
        env.step(teacher_actions(env))
    def forbidden(*args,**kwargs):raise AssertionError('Teacher called during inference')
    monkeypatch.setattr(training,'teacher_actions',forbidden)
    report,replay,_=training.rollout(model,'cpu',450200,count=1,record=True)
    assert report['mission']['size']==7 and len(replay['obstacles'])==16
    assert len(replay['frames'][0]['activity'])==360


def test_expanded_graph_preserves_neurons_and_synapses():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'data/processed'
    if not (root/'malecns_16000/graph.npz').exists():pytest.skip('Expanded graph not prepared')
    with np.load(root/'malecns_8000/graph.npz') as old,np.load(root/'malecns_16000/graph.npz') as new:
        assert len(new['neuron_ids'])==16000
        index=np.searchsorted(new['neuron_ids'],old['neuron_ids'])
        np.testing.assert_array_equal(new['neuron_ids'][index],old['neuron_ids'])
        # Every old measured edge is retained with exactly the same contact count.
        keys=new['pre'].astype('int64')*16000+new['post']
        old_keys=index[old['pre']]*16000+index[old['post']]
        slots=np.searchsorted(keys,old_keys)
        np.testing.assert_array_equal(keys[slots],old_keys)
        np.testing.assert_array_equal(new['synapse_count'][slots],old['synapse_count'])
