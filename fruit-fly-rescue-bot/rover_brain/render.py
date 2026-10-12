"""Render recorded model decisions as a 3D GIF; never generate new actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import SymLogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from .prepare import DEFAULT_GRAPH

BG, TEXT, MUTED, CYAN, GOLD = '#0b1422', '#edf5ff', '#8c9eb6', '#5ce1dc', '#ffcd70'


def box_faces(position, heading, half_width=.18, half_length=.28, bottom=.09, top=.35):
    vertices = np.array([[x, y, z] for z in [bottom, top] for y in [-half_width, half_width] for x in [-half_length, half_length]])
    c, s = np.cos(heading), np.sin(heading)
    vertices[:, :2] = vertices[:, :2] @ np.array([[c, s], [-s, c]]) + position
    return [vertices[idx] for idx in [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                                     [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, default=Path('runs/beacon_4000/fresh_evaluation/replay.json'))
    p.add_argument('--graph', type=Path, default=DEFAULT_GRAPH)
    p.add_argument('--output', type=Path, default=Path('runs/beacon_4000/rover_preview.gif'))
    p.add_argument('--subframes', type=int, default=2, help='Interpolated playback frames per decision')
    args = p.parse_args()
    if args.subframes < 1:
        p.error('--subframes must be positive')
    replay = json.loads(args.replay.read_text())
    if hashlib.sha256(args.graph.read_bytes()).hexdigest() != replay['graph_sha256']:
        raise ValueError('Replay and anatomy refer to different graphs')
    frames = replay['frames']
    if not frames:
        raise ValueError('Replay has no recorded decisions')
    with np.load(args.graph) as graph:
        coordinates = graph['soma_um']
    visible = np.isfinite(coordinates).all(axis=1)
    points = coordinates[visible]
    activity = np.asarray([f['activity'] for f in frames])
    if activity.shape[1] != len(coordinates):
        raise ValueError('Activity count differs from graph neuron count')
    positions = np.array([f['position'] for f in frames] + [replay['final_position']])
    headings = np.array([f['heading'] for f in frames] + [replay['final_heading']])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': TEXT, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'font.size': 10})
    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    obstacle_stage = replay.get('stage') == 'obstacles'
    fig.text(.05, .92, 'A fly-wired brain navigates obstacles' if obstacle_stage else 'A fly-wired brain learns to steer', fontsize=24, weight='bold')
    fig.text(.05, .875, 'RESCUE ROVER  /  STAGE 02: OBSTACLE NAVIGATION' if obstacle_stage else 'RESCUE ROVER  /  STAGE 01: VISIBLE BEACON', color=CYAN, fontsize=11)
    arena = fig.add_axes([.025, .15, .57, .67], projection='3d', facecolor=BG, computed_zorder=False)
    arena.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(0, 1.8))
    arena.set_box_aspect((8, 8, 2))
    arena.view_init(elev=48, azim=-65)
    arena.set_axis_off()
    floor = Poly3DCollection([[[-4,-4,0],[4,-4,0],[4,4,0],[-4,4,0]]], facecolors='#14253b', edgecolors='#3a536d', zorder=0)
    arena.add_collection3d(floor)
    for value in np.arange(-4, 5):
        arena.plot([-4, 4], [value, value], [0.005]*2, color='#233950', lw=.6)
        arena.plot([value, value], [-4, 4], [0.005]*2, color='#233950', lw=.6)
    for xmin, ymin, xmax, ymax in replay.get('obstacles', []):
        center=np.array([(xmin+xmax)/2,(ymin+ymax)/2])
        obstacle=Poly3DCollection(box_faces(center,0,half_width=(ymax-ymin)/2,half_length=(xmax-xmin)/2,
                                            bottom=0,top=.65),facecolors='#4e6078',edgecolors='#8c9eb6',linewidths=.6,zorder=3)
        arena.add_collection3d(obstacle)
    base, target = np.asarray(replay['base']), np.asarray(replay['target'])
    arena.scatter(*base, .025, s=130, marker='s', c=MUTED, zorder=3)
    arena.text(*base, .5, 'START', color=MUTED, fontsize=8)
    arena.plot([target[0]]*2, [target[1]]*2, [0, 1], color=GOLD, lw=3)
    arena.scatter(*target, 1, s=90, c=GOLD, zorder=5)
    theta = np.linspace(0, 2*np.pi, 80)
    arena.plot(target[0]+.35*np.cos(theta), target[1]+.35*np.sin(theta), np.full(80, .025), color=GOLD)
    arena.text(*target, 1.25, 'BEACON', color=GOLD, fontsize=9)
    rover = Poly3DCollection(box_faces(positions[0], headings[0]), facecolors=CYAN, edgecolors='#adfff4', linewidths=.8, zorder=6)
    arena.add_collection3d(rover)
    wheel_offsets=np.array([[-.18,-.19],[-.18,.19],[.18,-.19],[.18,.19]])
    wheels=[]
    for offset in wheel_offsets:
        wheel=Poly3DCollection(box_faces(positions[0]+offset,0,half_width=.035,half_length=.065,
                                         bottom=.025,top=.18),facecolors='#182130',edgecolors='#8c9eb6',linewidths=.6,zorder=7)
        arena.add_collection3d(wheel)
        wheels.append(wheel)
    nose, = arena.plot([], [], [], color='white', lw=3, zorder=7)
    trail, = arena.plot([], [], [], color=CYAN, lw=2, alpha=.7)
    brain = fig.add_axes([.60, .31, .35, .49], projection='3d', facecolor=BG)
    brain.set_axis_off()
    brain.set_box_aspect(np.ptp(points, axis=0))
    brain.view_init(elev=15, azim=-75)
    dots = brain.scatter(*points.T, c=activity[0, visible], s=3, cmap='coolwarm',
                         norm=SymLogNorm(linthresh=.005, vmin=-1, vmax=1), alpha=.85, depthshade=False)
    fig.text(.62, .79, f'{len(coordinates):,} neurons · measured wiring', fontsize=12, weight='bold')
    fig.text(.62, .755, f'{visible.sum():,} recorded soma positions shown', color=MUTED, fontsize=9)
    color_ax = fig.add_axes([.945, .40, .01, .24])
    bar = fig.colorbar(dots, cax=color_ax, ticks=[-1, -.01, 0, .01, 1])
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    bar.set_label('Signed model activity', fontsize=8)
    bar.ax.yaxis.set_label_position('left')
    probs = fig.add_axes([.65, .20, .28, .12], facecolor=BG)
    bars = probs.barh(['Forward', 'Left', 'Right'], frames[0]['probabilities'], color=CYAN, height=.5)
    probs.set_xlim(0, 1)
    probs.set_xticks([0, .5, 1], ['0', '50%', '100%'])
    probs.invert_yaxis()
    probs.tick_params(labelsize=8, length=0)
    for spine in probs.spines.values():
        spine.set_visible(False)
    status = fig.text(.05, .16, '', color=CYAN, fontsize=12)
    fig.text(.05, .10, 'Recorded policy decisions + neural activity. Geometry is interpolated for playback.', color=MUTED, fontsize=9)
    fig.text(.05, .065, 'MaleCNS v1.0 · HHMI Janelia / Cambridge / MRC LMB / Google · CC BY 4.0', color=MUTED, fontsize=8)
    scope='Visible beacon + local map; no inference planner; no return-to-base yet.' if obstacle_stage else 'Visible target; no obstacles or return-to-base yet.'
    fig.text(.05, .035, f'{scope} Soma coordinates missing for {(~visible).sum()} neurons.', color=MUTED, fontsize=8)

    def update(index):
        step = min(index // args.subframes, len(frames)-1)
        fraction = min((index - step*args.subframes)/args.subframes, 1.)
        position = positions[step]*(1-fraction) + positions[step+1]*fraction
        delta = (headings[step+1]-headings[step]+np.pi) % (2*np.pi) - np.pi
        heading = headings[step]+fraction*delta
        rover.set_verts(box_faces(position, heading))
        c,s=np.cos(heading),np.sin(heading)
        wheel_positions=wheel_offsets@np.array([[c,s],[-s,c]])+position
        for wheel,center in zip(wheels,wheel_positions):
            wheel.set_verts(box_faces(center,heading,half_width=.035,half_length=.065,bottom=.025,top=.18))
        front = position + .38*np.array([np.cos(heading), np.sin(heading)])
        nose.set_data_3d([position[0], front[0]], [position[1], front[1]], [.37, .37])
        route = np.vstack([positions[:step+1], position])
        trail.set_data_3d(route[:, 0], route[:, 1], np.full(len(route), .03))
        dots.set_array(activity[step, visible])
        for rect, prob in zip(bars, frames[step]['probabilities']):
            rect.set_width(prob)
        action = replay['action_names'][frames[step]['action']].replace('_', ' ').upper()
        finished = index >= len(frames)*args.subframes
        ending='BEACON REACHED' if replay['success'] else 'COLLISION' if replay.get('collided') else 'TIME LIMIT REACHED'
        status.set_text(ending if finished
                        else f'Decision {step+1:02d} / {len(frames):02d}   ·   {action}')
        return []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    update(len(frames)*args.subframes//2)
    fig.savefig(args.output.with_suffix('.png'), dpi=110, facecolor=BG)
    animation = FuncAnimation(fig, update, frames=len(frames)*args.subframes+20, interval=80, blit=False)
    animation.save(args.output, writer=PillowWriter(fps=12), dpi=90)
    plt.close(fig)
    print(f'Saved {args.output} and {args.output.with_suffix(".png")}')


if __name__ == '__main__':
    main()
