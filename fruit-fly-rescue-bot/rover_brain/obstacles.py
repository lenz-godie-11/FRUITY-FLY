"""Obstacle curriculum with collision geometry, local maps, and training teacher.

Policy observations: 32 processed range rays, 15x15 robot-centred occupancy
samples (0.4 m spacing), beacon bearing/distance, and previous-action one-hot.
The map is supplied by the simulator, not learned camera perception. Obstacles
are inflated by the rover's conservative circular footprint for both sensing
and collision tests. The global shortest-path teacher is training-only.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

RADIUS = .34
STEP = .15
TURN = .24
LIMIT = 4.0
RAYS = np.linspace(-np.pi, np.pi, 32, endpoint=False)
MAP_AXIS = np.arange(-7, 8) * .4
MAP_POINTS = np.stack(np.meshgrid(MAP_AXIS, MAP_AXIS), axis=-1).reshape(-1, 2)
OBSERVATIONS = 32 + len(MAP_POINTS) + 3 + 3
GRID_AXIS = np.linspace(-3.6, 3.6, 37)
GRID = np.stack(np.meshgrid(GRID_AXIS, GRID_AXIS), axis=-1).reshape(-1, 2)
GRID_SIZE = len(GRID_AXIS)
ACTION_NAMES = ['forward', 'turn_left', 'turn_right']


def occupied(points, boxes, margin=RADIUS):
    points = np.asarray(points)
    hit = (np.abs(points) >= LIMIT - margin).any(axis=-1)
    for xmin, ymin, xmax, ymax in boxes:
        hit |= ((points >= [xmin-margin, ymin-margin]) &
                (points <= [xmax+margin, ymax+margin])).all(axis=-1)
    return hit


def segment_blocked(start, end, boxes, margin=RADIUS):
    """Exact line-vs-expanded-AABB slab test, including grazing contact."""
    start, end = np.broadcast_arrays(np.asarray(start, float), np.asarray(end, float))
    direction = end - start
    hit = (np.abs(start) >= LIMIT-margin).any(axis=-1) | (np.abs(end) >= LIMIT-margin).any(axis=-1)
    parallel = np.abs(direction) < 1e-12
    safe = np.where(parallel, 1., direction)
    for xmin, ymin, xmax, ymax in boxes:
        lo, hi = np.array([xmin, ymin])-margin, np.array([xmax, ymax])+margin
        a, b = (lo-start)/safe, (hi-start)/safe
        near = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=-1)
        far = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=-1)
        outside_parallel = (parallel & ((start < lo) | (start > hi))).any(axis=-1)
        hit |= ~outside_parallel & (np.maximum(near, 0) <= np.minimum(far, 1))
    return hit


def ranges(position, heading, boxes):
    angle = heading + RAYS
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    parallel = np.abs(direction) < 1e-12
    safe = np.where(parallel, 1., direction)
    wall = np.where(direction > 0, LIMIT-RADIUS, -LIMIT+RADIUS)
    distance = np.where(parallel, np.inf, (wall-position)/safe).min(axis=1)
    for xmin, ymin, xmax, ymax in boxes:
        lo, hi = np.array([xmin, ymin])-RADIUS, np.array([xmax, ymax])+RADIUS
        a, b = (lo-position)/safe, (hi-position)/safe
        near = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=1)
        far = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=1)
        outside = (parallel & ((position < lo) | (position > hi))).any(axis=1)
        hit = ~outside & (far >= np.maximum(near, 0))
        distance = np.minimum(distance, np.where(hit, np.maximum(near, 0), np.inf))
    return np.clip(distance, 0, 8)


@dataclass
class Layout:
    boxes: np.ndarray
    start: np.ndarray
    target: np.ndarray
    initial_heading: float
    seed: int


def make_layout(seed, obstacles=6):
    if not 1 <= obstacles <= 10:
        raise ValueError('Use between 1 and 10 obstacles')
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        start = np.array([-3.1, rng.uniform(-2.5, 2.5)])
        target = np.array([3.1, rng.uniform(-2.5, 2.5)])
        middle = start + rng.uniform(.4, .6)*(target-start)
        half = rng.uniform(.4, .8, 2)
        boxes = [np.r_[middle-half, middle+half]]
        for _ in range(300):
            if len(boxes) == obstacles:
                break
            center = rng.uniform(-2.8, 2.8, 2)
            half = rng.uniform(.25, .65, 2)
            box = np.r_[center-half, center+half]
            if occupied(np.array([start, target]), [box], margin=.7).any():
                continue
            # Keep obstacles separate; free-space topology is checked below.
            if any(np.all(box[:2] < other[2:]+.2) and np.all(box[2:] > other[:2]-.2) for other in boxes):
                continue
            boxes.append(box)
        if len(boxes) != obstacles:
            continue
        boxes = np.asarray(boxes)
        # Rotations and reflections avoid a fixed left-to-right task shortcut.
        if rng.random() < .5:
            start, target = start[::-1].copy(), target[::-1].copy()
            boxes = boxes[:, [1, 0, 3, 2]]
        if rng.random() < .5:
            start *= -1
            target *= -1
            boxes = -boxes[:, [2, 3, 0, 1]]
        layout = Layout(boxes, start, target, float(rng.uniform(-np.pi, np.pi)), int(seed))
        planner = Teacher(layout)
        start_index = planner.nearest(start)
        if np.isfinite(planner.distance[start_index]) and segment_blocked(start, target, boxes):
            return layout
    raise RuntimeError(f'Could not generate traversable layout for seed {seed}')


class Teacher:
    """Dijkstra cost-to-go, used only for demonstrations and training labels."""
    def __init__(self, layout):
        self.layout = layout
        free = ~occupied(GRID, layout.boxes, margin=RADIUS+.08)
        rows, cols, weights = [], [], []
        indices = np.arange(len(GRID)).reshape(GRID_SIZE, GRID_SIZE)
        for dy, dx in [(0,1), (1,0), (1,1), (1,-1)]:
            y, x = np.indices(indices.shape)
            ny, nx = y+dy, x+dx
            valid = (ny >= 0) & (ny < GRID_SIZE) & (nx >= 0) & (nx < GRID_SIZE)
            a = indices[valid]
            b = indices[ny[valid], nx[valid]]
            good = free[a] & free[b] & ~segment_blocked(GRID[a], GRID[b], layout.boxes, margin=RADIUS+.08)
            a, b = a[good], b[good]
            rows.extend([a, b]); cols.extend([b, a])
            weights.extend([np.full(len(a), np.hypot(dx,dy)*.2)]*2)
        adjacency = sparse.csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(len(GRID),len(GRID)))
        self.free = free
        goal = self.nearest(layout.target)
        self.distance, self.predecessor = dijkstra(adjacency, indices=goal, return_predecessors=True)

    def nearest(self, point):
        distance = np.linalg.norm(GRID-point, axis=1)
        distance[~self.free] = np.inf
        return int(distance.argmin())

    def action(self, position, heading):
        target = self.layout.target
        if not segment_blocked(position, target, self.layout.boxes, margin=RADIUS+.07):
            waypoint = target
        else:
            current = self.nearest(position)
            waypoint = GRID[current]
            for _ in range(12):
                nxt = int(self.predecessor[current])
                if nxt < 0:
                    break
                candidate = GRID[nxt]
                if segment_blocked(position, candidate, self.layout.boxes, margin=RADIUS+.05):
                    break
                waypoint = candidate
                current = nxt
                if np.linalg.norm(waypoint-position) >= 1.0:
                    break
        delta = waypoint-position
        error = (np.arctan2(delta[1], delta[0])-heading+np.pi) % (2*np.pi)-np.pi
        return 1 if error > .16 else 2 if error < -.16 else 0


class ObstacleBatch:
    def __init__(self, count=32, seed=1000, obstacles=6, max_steps=240):
        # Each layout uses its own seed, so batching never changes an episode.
        self.layouts = [make_layout(seed+i, obstacles) for i in range(count)]
        self.position = np.array([l.start for l in self.layouts])
        self.base = self.position.copy()
        self.target = np.array([l.target for l in self.layouts])
        self.heading = np.array([l.initial_heading for l in self.layouts])
        self.done = np.zeros(count, bool)
        self.success = self.done.copy()
        self.collided = self.done.copy()
        self.steps = np.zeros(count, int)
        self.path_length = np.zeros(count)
        self.last_action = np.full(count, -1, int)
        self.max_steps = max_steps

    def observation(self):
        values = []
        for i, layout in enumerate(self.layouts):
            c, s = np.cos(self.heading[i]), np.sin(self.heading[i])
            points = MAP_POINTS @ np.array([[c, s], [-s, c]]) + self.position[i]
            grid = occupied(points, layout.boxes).astype('float32')
            delta = self.target[i]-self.position[i]
            angle = np.arctan2(delta[1],delta[0])-self.heading[i]
            previous = np.zeros(3)
            if self.last_action[i] >= 0:
                previous[self.last_action[i]] = 1
            values.append(np.r_[ranges(self.position[i], self.heading[i], layout.boxes)/8,
                                grid, np.sin(angle), np.cos(angle), np.linalg.norm(delta)/12, previous])
        return np.asarray(values, 'float32')

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != self.done.shape or not np.isin(actions, [0,1,2]).all():
            raise ValueError('Expected one action (0, 1 or 2) per rover')
        active = ~self.done
        self.heading += active*np.where(actions == 1, TURN, np.where(actions == 2, -TURN, 0))
        self.heading = (self.heading+np.pi) % (2*np.pi)-np.pi
        proposed = self.position + (active & (actions == 0))[:,None]*STEP*np.column_stack([np.cos(self.heading),np.sin(self.heading)])
        for i in np.flatnonzero(active):
            if segment_blocked(self.position[i], proposed[i], self.layouts[i].boxes):
                self.collided[i] = True
            else:
                self.path_length[i] += np.linalg.norm(proposed[i]-self.position[i])
                self.position[i] = proposed[i]
        self.last_action[active] = actions[active]
        self.steps += active
        self.success |= active & ~self.collided & (np.linalg.norm(self.target-self.position,axis=1) < .35)
        self.done |= self.success | self.collided | (self.steps >= self.max_steps)


def teachers(env):
    return [Teacher(layout) for layout in env.layouts]


def teacher_actions(env, planners):
    return np.array([p.action(pos,head) if not done else 0 for p,pos,head,done
                     in zip(planners,env.position,env.heading,env.done)], dtype='int64')
