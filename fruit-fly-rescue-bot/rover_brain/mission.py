"""Partially observed grid mission: discover, collect, return to remembered base.

This is a discrete mission curriculum, not the previous continuous steering task.
The rover has ideal cell localization, a known obstacle map, and a beacon sensor
limited to its cell and four neighbours. No beacon coordinate or home vector is
given. Collision attempts stay in place and are counted; success requires pickup
and return. The teacher explores using visit history, never the hidden target.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MissionSpec:
    size: int = 5
    obstacles: int = 6
    max_steps: int = 96
    minimum_distance: int = 3

    def __post_init__(self):
        if self.size < 3 or not 0 <= self.obstacles < self.cells-2:
            raise ValueError('Invalid grid size or obstacle count')
        if self.max_steps < 1 or not 1 <= self.minimum_distance < self.cells-self.obstacles:
            raise ValueError('Invalid mission budget or minimum distance')

    @property
    def cells(self): return self.size*self.size
    @property
    def position(self): return slice(0,self.cells)
    @property
    def walls(self): return slice(self.cells,2*self.cells)
    @property
    def signal(self): return slice(2*self.cells,2*self.cells+5)
    @property
    def carrying(self): return 2*self.cells+5
    @property
    def previous(self): return slice(2*self.cells+6,2*self.cells+10)
    @property
    def blocked(self): return 2*self.cells+10
    @property
    def departure(self): return 2*self.cells+11
    @property
    def observations(self): return 2*self.cells+12
    @property
    def positions(self):
        return np.array([[(i%self.size-(self.size-1)/2)*SPACING,
                          (i//self.size-(self.size-1)/2)*SPACING] for i in range(self.cells)])


SPACING = 1.2
DEFAULT_SPEC = MissionSpec()
SIZE, CELLS = DEFAULT_SPEC.size, DEFAULT_SPEC.cells
POSITIONS = DEFAULT_SPEC.positions
MOVES = np.array([[0,1],[1,0],[0,-1],[-1,0]])
ACTION_NAMES = ['north','east','south','west']
# Backward-compatible observation indices for the original 5x5 checkpoints.
POSITION, WALLS, SIGNAL = DEFAULT_SPEC.position, DEFAULT_SPEC.walls, DEFAULT_SPEC.signal
CARRYING, PREVIOUS = DEFAULT_SPEC.carrying, DEFAULT_SPEC.previous
BLOCKED, DEPARTURE, OBSERVATIONS = DEFAULT_SPEC.blocked, DEFAULT_SPEC.departure, DEFAULT_SPEC.observations


def neighbour(cell, action, spec=DEFAULT_SPEC):
    SIZE=spec.size
    x,y = cell % SIZE, cell//SIZE
    dx,dy = MOVES[action]
    x,y = x+int(dx),y+int(dy)
    return y*SIZE+x if 0 <= x < SIZE and 0 <= y < SIZE else -1


def neighbours(cell, walls, spec=DEFAULT_SPEC):
    return [(a,n) for a in range(4) if (n:=neighbour(cell,a,spec)) >= 0 and not walls[n]]


def distances(start, walls, spec=DEFAULT_SPEC):
    dist=np.full(spec.cells,-1,int);dist[start]=0
    queue=deque([start])
    while queue:
        cell=queue.popleft()
        for _,n in neighbours(cell,walls,spec):
            if dist[n]<0:dist[n]=dist[cell]+1;queue.append(n)
    return dist


@dataclass
class MissionLayout:
    walls: np.ndarray
    base: int
    beacon: int
    seed: int


def make_layout(seed, obstacles=None, spec=DEFAULT_SPEC):
    from dataclasses import replace
    if obstacles is not None:spec=replace(spec,obstacles=obstacles)
    CELLS,obstacles=spec.cells,spec.obstacles
    rng=np.random.default_rng(seed)
    for _ in range(1000):
        walls=np.zeros(CELLS,bool)
        walls[rng.choice(CELLS,obstacles,replace=False)]=True
        free=np.flatnonzero(~walls)
        if (distances(int(free[0]),walls,spec)[free]<0).any():continue
        base=int(rng.choice(free))
        far=free[distances(base,walls,spec)[free]>=spec.minimum_distance]
        if not len(far):continue
        return MissionLayout(walls,base,int(rng.choice(far)),int(seed))
    raise RuntimeError(f'Cannot create connected mission for seed {seed}')


class MissionBatch:
    def __init__(self,count=32,seed=1000,obstacles=None,max_steps=None,spec=DEFAULT_SPEC):
        from dataclasses import replace
        if obstacles is not None:spec=replace(spec,obstacles=obstacles)
        if max_steps is not None:spec=replace(spec,max_steps=max_steps)
        if count<1:raise ValueError("Positive batch size required")
        self.spec=spec
        self.layouts=[make_layout(seed+i,spec=spec) for i in range(count)]
        self.cell=np.array([l.base for l in self.layouts])
        self.base=self.cell.copy()
        self.beacon=np.array([l.beacon for l in self.layouts])
        self.walls=np.array([l.walls for l in self.layouts])
        self.steps=np.zeros(count,int)
        self.carrying=np.zeros(count,bool)
        self.discovered=np.zeros(count,bool)
        self.success=np.zeros(count,bool)
        self.done=np.zeros(count,bool)
        self.last_action=np.full(count,-1,int)
        self.blocked=np.zeros(count,bool)
        self.collision_attempts=np.zeros(count,int)
        self.pickup_step=np.full(count,-1,int)
        self.visited=np.zeros((count,spec.cells),bool)
        self.visited[np.arange(count),self.cell]=True
        self.max_steps=spec.max_steps

    def observation(self):
        spec=self.spec
        obs=np.zeros((len(self.cell),spec.observations),'float32')
        obs[np.arange(len(self.cell)),self.cell]=1
        obs[:,spec.walls]=self.walls
        for i,cell in enumerate(self.cell):
            if not self.carrying[i]:
                obs[i,spec.signal.start]=float(cell==self.beacon[i])
                for a,n in neighbours(int(cell),self.walls[i],spec):
                    if n==self.beacon[i]:obs[i,spec.signal.start+1+a]=1
        obs[:,spec.carrying]=self.carrying
        active_previous=self.last_action>=0
        obs[np.flatnonzero(active_previous),spec.previous.start+self.last_action[active_previous]]=1
        obs[:,spec.blocked]=self.blocked
        obs[:,spec.departure]=(self.steps==0)
        return obs

    def step(self,actions):
        actions=np.asarray(actions)
        if actions.shape!=self.cell.shape or not np.isin(actions,range(4)).all():
            raise ValueError('One cardinal action (0..3) is required per rover')
        for i in np.flatnonzero(~self.done):
            self.blocked[i]=False
            n=neighbour(int(self.cell[i]),int(actions[i]),self.spec)
            if n<0 or self.walls[i,n]:
                self.blocked[i]=True;self.collision_attempts[i]+=1
            else:
                self.cell[i]=n
            self.last_action[i]=actions[i]
            self.steps[i]+=1
            self.visited[i,self.cell[i]]=True
            if self.cell[i]==self.beacon[i] and not self.carrying[i]:
                self.carrying[i]=True;self.pickup_step[i]=self.steps[i]
            self.discovered[i] |= self.carrying[i] or any(n==self.beacon[i] for _,n in neighbours(int(self.cell[i]),self.walls[i],self.spec))
            self.success[i]=self.carrying[i] and self.cell[i]==self.base[i]
            self.done[i]=self.success[i] or self.steps[i]>=self.max_steps


def teacher_action(observation, visited, remembered_base, spec=DEFAULT_SPEC):
    """Uses only observation/history. Hidden beacon coordinates are unavailable."""
    cell=int(observation[spec.position].argmax())
    walls=observation[spec.walls].astype(bool)
    if observation[spec.carrying]:
        goals={int(remembered_base)}
    else:
        signal=observation[spec.signal][1:]
        if signal.any():return int(signal.argmax())
        goals=set(np.flatnonzero(~visited & ~walls).tolist())
    # Find a nearest frontier or the remembered base, with fixed cardinal ties.
    queue=deque([(cell,None)]);seen={cell}
    while queue:
        current,first=queue.popleft()
        if current in goals and first is not None:return first
        for action,n in neighbours(current,walls,spec):
            if n not in seen:
                seen.add(n);queue.append((n,action if first is None else first))
    return neighbours(cell,walls,spec)[0][0]


def teacher_actions(env):
    obs=env.observation()
    return np.array([teacher_action(o,v,b,env.spec) if not done else 0 for o,v,b,done
                     in zip(obs,env.visited,env.base,env.done)],dtype='int64')
