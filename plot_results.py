"""Plots from accepted CSV/JSON only; descriptive ranges, no fabricated CIs."""
import argparse
import csv
import json
import os
import statistics
from config import OUT
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'plot-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A=OUT/'analysis'
COLORS={'F':'#2868a3','N':'#d58826','Q':'#218b73'}


def output(fig,name):
 fig.tight_layout()
 for suffix in ('pdf','svg'):
  path=A/(name+'.'+suffix)
  if path.exists():raise FileExistsError(path)
  fig.savefig(path,bbox_inches='tight',metadata={'Creator':'zkDRL artifact'})
 plt.close(fig)


def numeric():
 data=json.loads((A/'NUMERICAL_ANALYSIS.json').read_text());assert data['status']=='complete'
 rows={(r['algorithm'],r['condition'],r['kind']):r for r in data['conditions']}
 conditions=['main_F','main_N','main_Q','residual','precision_S256'];labels=['F','N','Q','Residual','S256-Q']
 fig,axes=plt.subplots(1,2,figsize=(9,3.4))
 values=[rows['dqn',c,0]['greedy_success_mean']*5 for c in conditions]
 axes[0].bar(labels,values,color=['#2868a3','#d58826','#218b73','#aa567c','#7859a6'])
 axes[0].set(ylim=(0,5.6),yticks=range(6),ylabel='Successful seeds out of 5',title='DQN: Boolean login')
 for i,v in enumerate(values):axes[0].text(i,v+.08,f'{int(v)}/5',ha='center')
 means=[rows['dqn',c,1]['greedy_return_mean'] for c in conditions]
 lows=[m-rows['dqn',c,1]['greedy_return_min'] for m,c in zip(means,conditions)]
 highs=[rows['dqn',c,1]['greedy_return_max']-m for m,c in zip(means,conditions)]
 axes[1].errorbar(range(5),means,yerr=[lows,highs],fmt='o',capsize=4,color='#2868a3')
 axes[1].axhline(5.375,color='gray',linestyle='--',linewidth=1,label='oracle')
 axes[1].set(xticks=range(5),xticklabels=labels,ylabel='Final greedy return',title='DQN: four-item exposure')
 axes[1].legend(frameon=False)
 for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.suptitle('110-run paired numerical study: DQN endpoints; whiskers are seed min–max',fontsize=10)
 output(fig,'numerical-dqn-endpoints')
 with (A/'numerical-checkpoint-curves.csv').open() as f:curves=list(csv.DictReader(f))
 fig,axes=plt.subplots(1,2,figsize=(9,3.4))
 for kind,ax,oracle,title in [(0,axes[0],.875,'A2C: Boolean login'),(1,axes[1],5.375,'A2C: four-item exposure')]:
  for cond,label in [('main_F','F'),('main_N','N'),('main_Q','Q')]:
   xs=[0,1000,2000,3000,4000];means=[];lo=[];hi=[]
   for ep in xs:
    v=[float(r['stochastic_return']) for r in curves if r['algorithm']=='a2c' and r['condition']==cond and int(r['kind'])==kind and int(r['episode'])==ep and r['evaluation_status']=='available']
    if len(v)!=5:raise RuntimeError('plot must disclose missing n instead of silently assuming5')
    means.append(statistics.mean(v));lo.append(min(v));hi.append(max(v))
   ax.fill_between(xs,lo,hi,color=COLORS[label],alpha=.1)
   ax.plot(xs,means,marker={'F':'o','N':'s','Q':'^'}[label],markersize=3,label=label,color=COLORS[label])
  ax.axhline(oracle,color='gray',linestyle='--',linewidth=1)
  ax.set(title=title,xlabel='Training episodes',ylabel='Frozen-policy stochastic return',xticks=[0,1000,2000,3000,4000])
  ax.grid(alpha=.2);ax.legend(frameon=False)
 fig.suptitle('A2C: five stored checkpoints, five paired seeds; shading is min–max, not CI',fontsize=10)
 output(fig,'numerical-a2c-checkpoints')


def proof():
 data=json.loads((A/'PROOF_ANALYSIS.json').read_text());assert data['status']=='complete'
 rows=data['configurations'];fig,ax=plt.subplots(figsize=(10,3.5))
 main=sorted([r for r in rows if r['axis']=='main'],key=lambda r:(r['algorithm'],r['kind'],r['seed']))
 labels=[r['algorithm'].upper()+' '+('login' if r['kind']==0 else 'exp')+' '+str(r['seed']) for r in main]
 y=[r['prove_median_seconds']/60 for r in main]
 ax.bar(range(len(main)),y,color=['#2868a3' if r['algorithm']=='dqn' else '#218b73' for r in main])
 ax.errorbar(range(len(main)),y,yerr=[[r['prove_median_seconds']/60-r['prove_min_seconds']/60 for r in main],[r['prove_max_seconds']/60-r['prove_median_seconds']/60 for r in main]],fmt='none',ecolor='black',capsize=3)
 ax.set(xticks=range(len(main)),xticklabels=labels,ylabel='Prove-call minutes',title='Main cohort: 3 timing requests per fixed initialization/challenge')
 ax.tick_params(axis='x',rotation=45);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 output(fig,'proof-main-cohort')
 fig,axes=plt.subplots(1,3,figsize=(12,3.4))
 def draw(ax,selected,x,label,color):
  y=[r['prove_median_seconds']/60 for r in selected]
  ax.errorbar(x,y,yerr=[[r['prove_median_seconds']/60-r['prove_min_seconds']/60 for r in selected],[r['prove_max_seconds']/60-r['prove_median_seconds']/60 for r in selected]],marker='o',capsize=3,label=label,color=color)
 for alg,color in [('dqn','#2868a3'),('a2c','#218b73')]:
  selected=sorted([r for r in rows if r['algorithm']==alg and r['kind']==1 and r['seed']==5000 and r['width']==16 and r['capacity']==(64 if alg=='dqn' else None) and r['episodes'] in [1000,2000,3000,4000]],key=lambda r:r['episodes'])
  assert len(selected)==4
  draw(axes[0],selected,[r['episodes'] for r in selected],alg.upper(),color)
 base=next(r for r in rows if r['algorithm']=='dqn' and r['kind']==1 and r['seed']==5000 and r['width']==16 and r['capacity']==64 and r['episodes']==4000)
 width=[base]+sorted([r for r in rows if r['axis']=='width'],key=lambda r:r['width'])
 capacity=[base]+sorted([r for r in rows if r['axis']=='capacity'],key=lambda r:r['capacity'])
 draw(axes[1],width,[r['width'] for r in width],'DQN','#2868a3')
 draw(axes[2],capacity,[r['capacity'] for r in capacity],'DQN','#2868a3')
 for ax,title,xlabel in zip(axes,['Episodes','Hidden width','Replay capacity'],['Episodes','Hidden units','Stored transitions']):
  ax.set(title=title,xlabel=xlabel,ylabel='Prove-call minutes');ax.grid(alpha=.2);ax.legend(frameon=False)
 fig.suptitle('One-factor scaling on one host; shared baseline counted once; min–max over 3 requests',fontsize=10)
 output(fig,'proof-scaling')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['numeric','proof','all']);a=p.parse_args()
 if a.mode in ('numeric','all'):numeric()
 if a.mode in ('proof','all'):proof()
