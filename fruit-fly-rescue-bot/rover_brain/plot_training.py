"""Plot measured development outcomes from an obstacle or hidden-mission run."""
import argparse
import json
import os
from pathlib import Path
from .prepare import ROOT

os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=Path('runs/obstacles_8000'))
    args=p.parse_args()
    entries=[json.loads(line) for line in (args.run_dir/'metrics.jsonl').read_text().splitlines() if line]
    steps=[e['step'] for e in entries]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    if 'mission_success_rate' in entries[0]['development']:
        config=json.loads((args.run_dir/'config.json').read_text())
        spec=config.get('mission',{'size':5,'obstacles':6})
        ax.plot(steps,[e['development']['collected']/e['development']['episodes'] for e in entries],
                's-',color='#7960a6',label='Collected beacon')
        ax.plot(steps,[e['development']['mission_success_rate'] for e in entries],
                'o-',color='#087f8c',label='Returned with beacon')
        title=f"{config.get('neurons',8000):,} neurons · {spec['size']}×{spec['size']} grid · {spec['obstacles']} obstacles"
    else:
        ax.plot(steps,[e['development']['success_rate'] for e in entries],'o-',color='#087f8c',label='Reached beacon')
        ax.plot(steps,[e['development']['collisions']/e['development']['episodes'] for e in entries],
                's-',color='#b84634',label='Collision')
        ax.plot(steps,[e['development']['timeouts']/e['development']['episodes'] for e in entries],
                '^-',color='#7960a6',label='Time limit')
        title='8,000-neuron rover: navigation through six obstacles'
    ax.set(xlabel='Optimizer updates',ylabel='Fraction of development episodes',ylim=(-.02,1.02),
           title=title)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.14))
    fig.savefig(args.run_dir/'learning_curve.png',dpi=160)
    plt.close(fig)
    print(args.run_dir/'learning_curve.png')


if __name__=='__main__':main()
