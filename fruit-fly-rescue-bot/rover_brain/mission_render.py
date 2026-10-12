"""Animate actual hidden-mission actions, payload phase, and neural memory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .render import plt, FuncAnimation, PillowWriter, SymLogNorm, Poly3DCollection, box_faces, BG, TEXT, MUTED, CYAN, GOLD
from .mission import MOVES, MissionSpec, SPACING
from .obstacle_train import GRAPH


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,default=Path('runs/mission_8000/test/replay.json'))
    p.add_argument('--graph',type=Path,default=GRAPH)
    p.add_argument('--output',type=Path,default=Path('runs/mission_8000/hidden_mission.gif'))
    args=p.parse_args()
    replay=json.loads(args.replay.read_text())
    if hashlib.sha256(args.graph.read_bytes()).hexdigest()!=replay['graph_sha256']:raise ValueError('Replay graph mismatch')
    spec=MissionSpec(**replay.get('mission',{}))
    POSITIONS=spec.positions
    extent=spec.size*SPACING/2
    frames=replay['frames']
    if not frames:raise ValueError('Empty replay')
    with np.load(args.graph) as g:coords=g['soma_um']
    visible=np.isfinite(coords).all(axis=1);points=coords[visible]
    positions=np.array([f['position'] for f in frames]+[replay['final_position']])
    activity=np.array([f['activity'] for f in frames])
    if activity.shape[1]!=len(coords):raise ValueError('Replay activity count mismatch')
    base,target=np.array(replay['base']),np.array(replay['target'])
    base_cell=int(np.linalg.norm(POSITIONS-base,axis=1).argmin())
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':TEXT,'xtick.color':MUTED,'ytick.color':MUTED,'font.size':10})
    fig=plt.figure(figsize=(12,7),facecolor=BG)
    fig.text(.05,.92,'Find it. Remember. Bring it home.',fontsize=26,weight='bold')
    fig.text(.05,.875,f'{len(coords):,}-NEURON ROVER  /  {spec.size}×{spec.size} HIDDEN-BEACON MISSION',fontsize=11,color=CYAN)
    arena=fig.add_axes([.025,.17,.56,.64],projection='3d',facecolor=BG,computed_zorder=False)
    arena.set(xlim=(-extent-.1,extent+.1),ylim=(-extent-.1,extent+.1),zlim=(0,1.5));arena.set_box_aspect((2*extent,2*extent,1.6))
    arena.view_init(elev=52,azim=-65);arena.set_axis_off()
    arena.add_collection3d(Poly3DCollection([[[-extent,-extent,0],[extent,-extent,0],[extent,extent,0],[-extent,extent,0]]],facecolors='#14253b',edgecolors='#3a536d',zorder=0))
    for v in np.linspace(-extent,extent,spec.size+1):
        arena.plot([-extent,extent],[v,v],[.01,.01],color='#2a4159',lw=.6)
        arena.plot([v,v],[-extent,extent],[.01,.01],color='#2a4159',lw=.6)
    for xmin,ymin,xmax,ymax in replay['obstacles']:
        center=np.array([(xmin+xmax)/2,(ymin+ymax)/2])
        arena.add_collection3d(Poly3DCollection(box_faces(center,0,half_width=(ymax-ymin)/2,half_length=(xmax-xmin)/2,bottom=0,top=.6),
                                                 facecolors='#4e6078',edgecolors='#8297b0',linewidths=.6,zorder=3))
    circle=np.linspace(0,2*np.pi,60)
    arena.plot(base[0]+.38*np.cos(circle),base[1]+.38*np.sin(circle),np.full(60,.02),color=CYAN,lw=2)
    arena.text(*base,.8,'BASE',color=CYAN,fontsize=9)
    beacon=arena.scatter(*target,.45,s=90,color=GOLD,zorder=4)
    beacon_label=arena.text(*target,.85,'HIDDEN BEACON',color=GOLD,fontsize=8)
    rover=Poly3DCollection(box_faces(positions[0],0),facecolors=CYAN,edgecolors='#b4fff1',zorder=6)
    arena.add_collection3d(rover)
    nose,=arena.plot([],[],[],color='white',lw=3,zorder=7)
    payload=arena.scatter(*positions[0],.55,s=42,color=GOLD,zorder=8)
    search_trail,=arena.plot([],[],[],color=CYAN,lw=2,zorder=2)
    return_trail,=arena.plot([],[],[],color=GOLD,lw=2,zorder=2)
    brain=fig.add_axes([.60,.37,.35,.40],projection='3d',facecolor=BG)
    brain.set_axis_off();brain.set_box_aspect(np.ptp(points,axis=0));brain.view_init(elev=15,azim=-75)
    dots=brain.scatter(*points.T,c=activity[0,visible],s=2.3,cmap='coolwarm',norm=SymLogNorm(.005,vmin=-1,vmax=1),depthshade=False,alpha=.85)
    fig.text(.62,.79,'Persistent neural memory',fontsize=13,weight='bold')
    fig.text(.62,.755,f'{visible.sum():,} measured soma positions · signed model activity',fontsize=8,color=MUTED)
    info=fig.text(.63,.35,'',fontsize=10,color=CYAN)
    probs=fig.add_axes([.67,.17,.26,.14],facecolor=BG)
    bars=probs.barh(['North','East','South','West'],frames[0]['probabilities'],color=CYAN,height=.5)
    probs.set_xlim(0,1);probs.set_xticks([0,.5,1],['0','50%','100%']);probs.invert_yaxis();probs.tick_params(length=0,labelsize=8)
    for spine in probs.spines.values():spine.set_visible(False)
    phase=fig.text(.05,.17,'',fontsize=13,color=CYAN)
    fig.text(.05,.12,'Cyan: search path     Gold: return path     Beacon marker is shown to the viewer only.',fontsize=9,color=MUTED)
    fig.text(.05,.075,'Grid curriculum · known obstacle map · proximity-only beacon readings · engineered neural memory gates',fontsize=8,color=MUTED)
    fig.text(.05,.04,'MaleCNS v1.0 · HHMI Janelia / Cambridge / MRC LMB / Google · CC BY 4.0',fontsize=8,color=MUTED)
    pickup=replay['pickup_step']
    def update(index):
        step=min(index//4,len(frames)-1);fraction=min((index-step*4)/4,1.)
        frame=frames[step];position=positions[step]*(1-fraction)+positions[step+1]*fraction
        heading=float(np.arctan2(MOVES[frame['action'],1],MOVES[frame['action'],0]))
        rover.set_verts(box_faces(position,heading))
        front=position+.38*np.array([np.cos(heading),np.sin(heading)])
        nose.set_data_3d([position[0],front[0]],[position[1],front[1]],[.37,.37])
        payload._offsets3d=([position[0]],[position[1]],[.55])
        payload.set_visible(frame['carrying'])
        beacon.set_visible(not frame['carrying']);beacon_label.set_visible(not frame['carrying'])
        route=np.vstack([positions[:step+1],position])
        search=route[:pickup+1] if pickup>=0 else route
        search_trail.set_data_3d(search[:,0],search[:,1],np.full(len(search),.03))
        returning=route[pickup:] if pickup>=0 and step>=pickup else np.empty((0,2))
        return_trail.set_data_3d(returning[:,0],returning[:,1],np.full(len(returning),.035))
        dots.set_array(activity[step,visible])
        for bar,prob in zip(bars,frame['probabilities']):bar.set_width(prob)
        recall='correctly recalled' if frame['decoded_home_cell']==base_cell else 'uncertain'
        signal='sample collected' if frame['carrying'] else 'nearby beacon' if any(frame['local_signal']) else 'no local signal'
        info.set_text(f'Home: {recall}   ·   Explored: {frame["visited_count"]}/{spec.cells-spec.obstacles}\nSensor: {signal}')
        if index>=len(frames)*4:
            phase.set_text('MISSION COMPLETE · BACK AT BASE' if replay['success'] else 'TIME LIMIT · MISSION INCOMPLETE')
        else:
            phase.set_text(f'{"RETURN TO BASE" if frame["carrying"] else "SEARCH"}   ·   Decision {step+1}/{len(frames)}')
        phase.set_color(GOLD if frame['carrying'] else CYAN)
        return []
    args.output.parent.mkdir(parents=True,exist_ok=True)
    update((pickup+1)*4 if pickup>=0 else len(frames)*2)
    fig.savefig(args.output.with_suffix('.png'),dpi=110,facecolor=BG)
    animation=FuncAnimation(fig,update,frames=len(frames)*4+24,interval=100,blit=False)
    animation.save(args.output,writer=PillowWriter(fps=10),dpi=90)
    plt.close(fig);print(f'Saved {args.output}')


if __name__=='__main__':main()
