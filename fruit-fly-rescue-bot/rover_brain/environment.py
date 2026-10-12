"""Stage 1: batched planar rover with a visible beacon and eight range sensors.

Geometry is independent of rendering. Hidden-beacon search and obstacles are
later curriculum stages, not capabilities of this environment.
"""
import numpy as np

OBSERVATIONS = 11
ACTION_NAMES = ['forward', 'turn_left', 'turn_right']
RAY_ANGLES = np.linspace(-np.pi, np.pi, 8, endpoint=False)


def observe(position, heading, target):
    delta = target - position
    distance = np.linalg.norm(delta, axis=1)
    angle = np.arctan2(delta[:, 1], delta[:, 0]) - heading
    rays = heading[:, None] + RAY_ANGLES
    direction = np.stack([np.cos(rays), np.sin(rays)], axis=-1)
    safe = np.where(np.abs(direction) < 1e-7, 1e-7, direction)
    wall = np.where(direction >= 0, 4.0, -4.0)
    intersections = (wall - position[:, None, :]) / safe
    ranges = np.clip(intersections.min(axis=-1), 0, 8) / 8
    return np.column_stack([ranges, np.sin(angle), np.cos(angle), distance / 12]).astype('float32')


def teacher(observations):
    angle = np.arctan2(observations[:, 8], observations[:, 9])
    return np.where(angle > .18, 1, np.where(angle < -.18, 2, 0)).astype('int64')


def training_batch(rng, count):
    # Balance forward and turning examples; all are valid geometric states.
    position = rng.uniform(-2.5, 2.5, (count, 2))
    target = rng.uniform(-2.5, 2.5, (count, 2))
    bearing = np.arctan2(*(target - position)[:, ::-1].T)
    offset = rng.uniform(-np.pi, np.pi, count)
    forward = rng.random(count) < .35
    offset[forward] = rng.uniform(-.18, .18, forward.sum())
    obs = observe(position, bearing + offset, target)
    return obs, teacher(obs)


class RoverBatch:
    def __init__(self, count=32, seed=10000, max_steps=160):
        rng = np.random.default_rng(seed)
        self.position = rng.uniform(-2.5, 2.5, (count, 2))
        self.base = self.position.copy()
        self.target = rng.uniform(-2.5, 2.5, (count, 2))
        close = np.linalg.norm(self.target - self.position, axis=1) < 1
        while close.any():
            self.target[close] = rng.uniform(-2.5, 2.5, (close.sum(), 2))
            close = np.linalg.norm(self.target - self.position, axis=1) < 1
        self.heading = rng.uniform(-np.pi, np.pi, count)
        self.done = np.zeros(count, dtype=bool)
        self.success = self.done.copy()
        self.steps = np.zeros(count, dtype=int)
        self.max_steps = max_steps

    def observation(self):
        return observe(self.position, self.heading, self.target)

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != self.done.shape or not np.isin(actions, [0, 1, 2]).all():
            raise ValueError('Expected one action (0, 1 or 2) per rover')
        active = ~self.done
        self.heading += active * np.where(actions == 1, .22, np.where(actions == 2, -.22, 0))
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        forward = active & (actions == 0)
        direction = np.column_stack([np.cos(self.heading), np.sin(self.heading)])
        self.position = np.clip(self.position + .15 * forward[:, None] * direction, -3.85, 3.85)
        self.steps += active
        self.success |= np.linalg.norm(self.position - self.target, axis=1) < .35
        self.done |= self.success | (self.steps >= self.max_steps)
        return self.observation()
