"""Render frozen automated intervention results; no participant inference."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent

def build():
    data=json.loads((ROOT/'dsge_prediction_baselines.json').read_text())['automated_baselines']['automated']
    families=['h1_attenuation','incoming_weight','h2_attenuation','sham']
    names=['First-layer attenuation','Incoming weight edit','Final-layer attenuation','Sham (0% attenuation)']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10.8,5.3),facecolor='#f8f6ef');ax.set_facecolor('#f8f6ef')
    y=np.arange(4)
    for offset,key,color,label in [(-.16,'zero_change','#a5a39a','Predict no change'),(.16,'frozen_gate_jacobian','#b94c32','Keep baseline gates fixed')]:
        vals=[data[f][key]['mae_sd'] for f in families]
        ax.barh(y+offset,vals,height=.28,color=color,label=label)
        for yy,v in zip(y+offset,vals): ax.text(v+.003,yy,'0' if v<1e-12 else f'{v:.4f}',va='center',fontsize=10)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,.24);ax.set_xlabel('Mean absolute prediction error / TRAIN output standard deviation')
    ax.legend(loc='lower right',frameon=False);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.xaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('A local calculation can miss finite intervention effects',x=.035,ha='left',fontsize=18,fontweight='bold')
    fig.text(.035,.89,'Frozen 387-parameter DSGE policy network · 300 fresh cases per edit family',fontsize=11)
    fig.text(.035,.045,'Exact network evaluation: 100% within 0.02 SD on all 1,200 cases. Human participants: 0.',fontsize=11)
    fig.subplots_adjust(left=.28,right=.96,top=.79,bottom=.18)
    out=ROOT/'output/dsge_prediction';out.mkdir(parents=True,exist_ok=True)
    for ext in ['svg','png']: fig.savefig(out/f'intervention-errors.{ext}',dpi=160)
    plt.close(fig)
if __name__=='__main__': build()
