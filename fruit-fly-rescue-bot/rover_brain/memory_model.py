"""A measured-connectome policy with explicit gated neural memory banks.

The memory mechanism is engineered, not a claim of biological emulation. Learned
home and visit encodings occupy disjoint sensory-neuron state slots and reach
the action readout only through measured synaptic edges. Long-term slots persist;
transient neural processing settles anew each decision. No search algorithm is
used by the policy.
"""
import torch
from torch import nn
from .model import ConnectomePolicy
from .mission import DEFAULT_SPEC


class MemoryConnectomePolicy(ConnectomePolicy):
    def __init__(self,graph_path,ticks=3,spec=DEFAULT_SPEC):
        self.spec=spec
        CELLS=spec.cells
        super().__init__(graph_path,ticks=ticks,observations=spec.observations,actions=4)
        if len(self.inputs)<3*CELLS:raise ValueError('Insufficient sensory neurons for mission memory')
        bank=min(max(100,2*CELLS),len(self.inputs)//3)
        self.register_buffer('home_indices',self.inputs[:bank].clone())
        self.register_buffer('visit_indices',self.inputs[bank:2*bank].clone())
        self.register_buffer('sensor_indices',self.inputs[2*bank:].clone())
        memory_mask=torch.zeros(self.n)
        memory_mask[self.home_indices]=1
        memory_mask[self.visit_indices]=1
        self.register_buffer('memory_mask',memory_mask)
        self.encoder=nn.Linear(spec.observations,len(self.sensor_indices))
        self.home_encoder=nn.Linear(CELLS,bank,bias=False)
        self.visit_encoder=nn.Linear(CELLS,bank,bias=False)
        # A stable addressable starting code, subsequently trainable. Memory
        # itself is not learned from scratch; action selection and encodings are.
        with torch.no_grad():
            code=torch.eye(CELLS).repeat((bank+CELLS-1)//CELLS,1)[:bank]
            self.home_encoder.weight.copy_(code*.9)
            self.visit_encoder.weight.copy_(code*.9)
        self.home_decoder=nn.Linear(len(self.outputs),CELLS)
        self.visit_decoder=nn.Linear(len(self.outputs),CELLS)

    def initial_state(self,batch,device=None):
        return torch.zeros(batch,self.n,device=device or self.base_weight.device)

    def state_from_history(self,base,visited):
        """Training-only exact recomputation of gated memory from a past history.

        This is algebraically equivalent to replaying home-write and visit-max
        updates. The deployed policy receives observations and its own state,
        never this history table or the true base after departure.
        """
        home=torch.tanh(self.home_encoder(torch.nn.functional.one_hot(base,self.spec.cells).float()))
        codes=torch.relu(torch.tanh(self.visit_encoder(torch.eye(self.spec.cells,device=base.device))))
        visits=torch.where(visited[:,:,None],codes[None,:,:],0.).amax(dim=1)
        state=self.initial_state(len(base),base.device)
        state=state.index_copy(1,self.home_indices,home)
        return state.index_copy(1,self.visit_indices,visits)

    def step(self,obs,state,weights=None,forget_home=False,forget_visits=False,silence=False):
        w=self.weights() if weights is None else weights
        reset=obs[:,self.spec.departure:self.spec.departure+1]>0
        state=torch.where(reset,torch.zeros_like(state),state)
        home=torch.where(reset,torch.tanh(self.home_encoder(obs[:,self.spec.position])),state[:,self.home_indices])
        previous=state[:,self.visit_indices]
        if forget_visits:previous=torch.zeros_like(previous)
        visits=torch.maximum(previous,torch.relu(torch.tanh(self.visit_encoder(obs[:,self.spec.position]))))
        if forget_home:home=torch.zeros_like(home)
        sensory=torch.tanh(self.encoder(obs))
        # Long-lived memory slots persist; transient processing settles anew.
        activity=state*self.memory_mask
        activity=activity.index_copy(1,self.home_indices,home)
        activity=activity.index_copy(1,self.visit_indices,visits)
        for _ in range(self.ticks):
            recurrent=torch.sparse.mm(w,activity.T).T if w.is_sparse else activity@w
            activity=.25*activity+.75*torch.tanh(recurrent)
            activity=activity.index_copy(1,self.sensor_indices,sensory)
            activity=activity.index_copy(1,self.home_indices,home)
            activity=activity.index_copy(1,self.visit_indices,visits)
        if silence:activity=torch.zeros_like(activity)
        readout=activity[:,self.outputs]
        normalized=self.readout[0](readout)
        logits=self.readout[1](normalized)
        return logits,activity,self.home_decoder(normalized),self.visit_decoder(normalized)
