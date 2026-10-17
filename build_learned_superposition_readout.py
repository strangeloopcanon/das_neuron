"""Plot frozen learned-model results; no fitting or changes to study evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/learned_v2'
COLORS=['#087f78','#315c9b','#9a5236','#744a86','#6d742c','#a04162']


def main():
    result=json.loads((ROOT/'learned_superposition_v2_results.json').read_text())
    confirmation=json.loads((ROOT/'learned_superposition_v2_confirmation_results.json').read_text())
    selection=json.loads((ROOT/'learned_superposition_v2_confirmation_protocol.json').read_text())
    default=result['default_seed'];model=next(m for m in result['models'] if m['stage']=='compressed' and m['seed']==default)
    fresh=next(m for m in confirmation['models'] if m['seed']==default)
    keep=[q['feature'] for q in fresh['per_feature'] if q['selected_on_validation']]
    w=np.asarray(model['W']);OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#152d38','axes.labelcolor':'#536974','xtick.color':'#536974','ytick.color':'#536974','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.8),gridspec_kw={'width_ratios':[1,1.3]})
    ax=axes[0]
    ax.axhline(0,color='#dce4e7',lw=.8);ax.axvline(0,color='#dce4e7',lw=.8)
    for i in range(6):
        retained=i in keep
        ax.annotate('',xy=w[:,i],xytext=(0,0),arrowprops={'arrowstyle':'->','color':COLORS[i] if retained else '#a9b3b8','lw':2 if retained else 1})
        ax.scatter(*w[:,i],color=COLORS[i] if retained else '#a9b3b8',s=26)
        xy=w[:,i]*1.12
        if retained:ax.text(*xy,f'S{i+1}',color=COLORS[i],ha='center',va='center',weight='bold')
        else:ax.annotate(f'S{i+1}: weak code',xy=w[:,i],xytext=(12,-24),textcoords='offset points',color='#536974',fontsize=9,arrowprops={'arrowstyle':'-','color':'#a9b3b8'})
    ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),xlabel='Hidden coordinate 1',ylabel='Hidden coordinate 2')
    ax.set_aspect('equal');ax.set_title('Learned feature directions',loc='left',fontweight='bold',pad=14)
    ax.text(0,-.25,'Five large directions form a pentagon.\nThe sixth response is reconstructed poorly.',transform=ax.transAxes,fontsize=9,color='#536974',va='top')
    ax=axes[1];indices=np.arange(6)
    active=[q['active_relative_rmse']*100 for q in fresh['per_feature']]
    leak=[q['combined_inactive_relative_rmse']*100 for q in fresh['per_feature']]
    ax.bar(indices-.18,active,width=.34,color='#315c9b',label='Active-value error')
    ax.bar(indices+.18,leak,width=.34,color='#087f78',label='Inactive-output leakage')
    ax.axhline(25,color='#536974',ls='--',lw=1,label='Declared per-feature limit')
    for i,v in enumerate(active):ax.text(i-.18,v+1.7,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
    ax.set(xticks=indices,xticklabels=[f'S{i+1}' for i in range(6)],ylim=(0,100),ylabel='% of active-response RMS')
    ax.set_title('Fresh holdout: all six sectors tested',loc='left',fontweight='bold',pad=14)
    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax.text(0,-.25,'Features were selected on VALIDATION before this fresh test.\nDefault seed 1103 was fixed before training.',transform=ax.transAxes,fontsize=9,color='#536974',va='top')
    fig.subplots_adjust(left=.07,right=.99,top=.87,bottom=.27,wspace=.34)
    fig.suptitle('Five response features learned in two coordinates',x=.07,ha='left',fontsize=16,fontweight='bold')
    fig.savefig(OUT/'learned-features.svg',metadata={'Date':None})
    svg=OUT/'learned-features.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(OUT/'learned-features.png',dpi=180,metadata={'Software':'Matplotlib'})
    plt.close(fig)
    selected=[q for m in confirmation['models'] for q in m['per_feature'] if q['selected_on_validation']]
    active_range=min(q['active_relative_rmse'] for q in selected),max(q['active_relative_rmse'] for q in selected)
    leak_range=min(q['combined_inactive_relative_rmse'] for q in selected),max(q['combined_inactive_relative_rmse'] for q in selected)
    r2_range=min(q['active_amplitude_r2'] for q in selected),max(q['active_amplitude_r2'] for q in selected)
    lines=['# Five learned features in two hidden coordinates','',
      '**The full-width training problem is fixed. All five 42-parameter controls learn all six responses. After compression and fine-tuning, all five 18-parameter networks retain five response features in two coordinates on singleton inputs.**','',
      f"The fresh confirmation test gives retained-feature active-value errors of **{100*active_range[0]:.2f}–{100*active_range[1]:.2f}%**, combined inactive-output leakage of **{100*leak_range[0]:.2f}–{100*leak_range[1]:.2f}%**, and active-amplitude R² of **{r2_range[0]:.3f}–{r2_range[1]:.3f}**. It tests 4,096 new amplitudes for every sector, including the poorly reconstructed sector.",'',
      '[Open the learned circuit](learned_superposition.html) · [Primary frozen protocol](learned_superposition_v2_protocol.json) · [All original outcomes](learned_superposition_v2_report.md) · [Fresh confirmation protocol](learned_superposition_v2_confirmation_protocol.json)','',
      '![Learned five-direction code and fresh per-feature reconstruction errors, including the poorly reconstructed sixth sector.](output/learned_v2/learned-features.svg)','',
      '## What was trained','',
      'The economic teacher has logarithmic utility, Cobb–Douglas production and full capital depreciation. Each of six separate growth economies has the exact policy c=(1−αβ)zk^α and k′=αβzk^α. At fixed capital one, its normalized consumption response is x=(c−c₀)/c₀=z−1. The networks take the solved response vectors as data and reconstruct them. They are response-compression models, not new policy solvers. The underlying economic responses are additive.','',
      'The repaired network uses h=Wx and x̂=ReLU(Wᵀh+b). Sharing the encoder and decoder weights gives 42 trainable parameters for six hidden coordinates, then 18 for two coordinates. For each fixed seed, we train the six-coordinate model, project its learned encoder using an independently sampled orthogonal projection, reset the decoder bias to zero, then fine-tune on TRAIN. No hexagon, pentagon, feature angles, or feature-aligned initialization is supplied.','',
      'The full-width VALIDATION gate passed for every seed before any compressed training began. All five full-width TEST controls then passed. Their worst overall TEST relative RMSE is 5.35×10⁻⁸. Previous failed untied-weight runs remain intact.','',
      '## Original target and fresh confirmation','',
      '**The predeclared all-six-feature target failed: 0/5 compressed models passed it.** Every model reconstructed one feature poorly while retaining the other five. The five-feature finding was discovered after the original test. It is not reported as success on the original six-feature criterion.','',
      'We then froze a separate confirmation protocol. It fixes every model by hash and selects retained features using only the original VALIDATION criteria: active error ≤25%, combined inactive-output leakage ≤25%, and positive active-amplitude R². No model was retrained. The fresh dataset uses a new seed and 24,576 positive rows disjoint from every original partition. All five selected features passed for all five models. All six features were evaluated and remain in the results.','',
      '| Seed | Full-width TEST | Confirmed features | Poorly reconstructed sector | Retained-feature active error | Retained-feature amplitude R² |','|---:|---|---:|---|---:|---:|']
    for m in confirmation['models']:
        pf=[q for q in m['per_feature'] if q['selected_on_validation']];weak=[q['feature']+1 for q in m['per_feature'] if not q['selected_on_validation']]
        full=next(q for q in result['models'] if q['stage']=='full' and q['seed']==m['seed'])
        lines.append(f"| {m['seed']} | {full['test_pass']} | {m['confirmed_count']}/6 | S{weak[0]} | {100*min(q['active_relative_rmse'] for q in pf):.2f}–{100*max(q['active_relative_rmse'] for q in pf):.2f}% | {min(q['active_amplitude_r2'] for q in pf):.3f}–{max(q['active_amplitude_r2'] for q in pf):.3f} |")
    lines+=['','The denominator is active-response RMS, not the level of economic consumption. Leakage combines all five inactive outputs. Amplitude R² is computed only on active examples, so an output that merely identifies the sector and emits a constant cannot pass.','',
      '## What the two neurons contain','',
      'For the default seed, the hidden coordinates are approximately:','','```text',
      'h1 = 0.271 S1 − 1.052 S2 + 0.485 S3 − 0.003 S4 − 0.919 S5 + 1.222 S6',
      'h2 = 1.196 S1 + 0.624 S2 − 1.126 S3 − 0.005 S4 − 0.809 S5 + 0.113 S6',
      '```','',
      'Here S1–S6 are the known sector-response inputs. The retained features use five distinct directions in the same two-dimensional space. Their consecutive angular gaps lie close to 72 degrees in every seed, a learned pentagon. These directions and angles are descriptive observations after training, not an imposed template or original acceptance criterion. The sixth direction is very small; its decoder mostly emits a poor constant response.','',
      'Five distinct, independently controlled response features remain selectively recoverable through two shared coordinates. That is the evidence for learned superposition in this experiment. Individual hidden coordinates mix those economic features. The result does not depend on calling a correlation a complete neuron meaning.','',
      '## Changes to the circuit','',
      'The saved intervention tests insert and remove each feature direction in latent space, decode the edited vector, measure target reconstruction, and measure changes to unaffected outputs. Separate coordinate ablations show how removing either hidden coordinate affects several sector responses. Encoding equality alone follows from the linear encoder; the measured decoder selectivity is the substantive result.','',
      'On simultaneous shocks, both the full-width and compressed models can fail because their training support contains only zero and singleton responses. Compressed pair-response RMSE is about 74%, versus 25–35% for matched full-width controls. These are distribution-shift results. The economic teacher remains additive. Pair failures should not all be attributed to compression.','',
      '## Verification and execution','',
      f"The training run took {result['training_seconds']:.2f} seconds of local CPU time. All five seeds are shown. The original protocol, numerical source, data archive and weights are hashed. The independent verifier imports no training module and reconstructs outputs, accuracy, geometry, projection ancestry, baselines, intervention effects, flags and the fresh confirmation. No old experiment or public deployment was replaced.",'','```sh','python -B verify_learned_superposition_v2.py','node verify_learned_superposition_ui.js','```','',
      'This establishes learned feature sharing in a small circuit. Difficulty of human understanding is a separate measurement; this study did not test it.','']
    (ROOT/'learned_superposition_v2_readout.md').write_text('\n'.join(lines))
    print('Built frozen-result readout and scientific figure; no fitting.')


if __name__=='__main__':main()
