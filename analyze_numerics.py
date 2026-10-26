"""Descriptive paired-five-seed outputs, preserving NA and failed arms."""
import csv
import json
import statistics
from pathlib import Path
from config import OUT,save
from design import numerical_jobs,CHECKPOINTS


def values(algorithm,checkpoint):
 if not checkpoint or checkpoint.get('evaluation_status')!='available':return dict(greedy_return=None,stochastic_return=None,greedy_success=None,stochastic_success=None)
 if algorithm=='dqn':
  return dict(greedy_return=checkpoint['return'],stochastic_return=None,greedy_success=checkpoint.get('success'),stochastic_success=None)
 return dict(greedy_return=checkpoint['greedy']['undiscounted_return'],
  stochastic_return=checkpoint['stochastic']['undiscounted_return'],
  greedy_success=checkpoint['greedy']['success_probability'],stochastic_success=checkpoint['stochastic']['success_probability'])


def write_csv(path,rows):
 with path.open('x',newline='') as out:
  writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
 gate=json.loads((OUT/'NUMERICAL_GATE.json').read_text());assert gate['status']=='passed'
 dest=OUT/'analysis';dest.mkdir(exist_ok=True)
 endpoints=[];curves=[];diagnostics=[];groups={}
 for job in numerical_jobs():
  record=json.loads((OUT/'numerical-runs'/job['id']/'record.json').read_text())
  checkpoints={row['episode']:row for row in record['checkpoints']}
  common={k:job[k] for k in ('id','algorithm','condition','kind','seed')}
  for episode in CHECKPOINTS:
   cp=checkpoints.get(episode)
   row=dict(**common,episode=episode,training_status=record['status'],
    evaluation_status=cp.get('evaluation_status','not_reached') if cp else 'not_reached',**values(job['algorithm'],cp))
   curves.append(row)
   if episode==4000:endpoints.append(row);groups.setdefault((job['algorithm'],job['condition'],job['kind']),[]).append(row)
  diagnostics.append(dict(**common,training_status=record['status'],completed_episodes=record['completed_episodes'],
   evaluation_failure_count=record['evaluation_failure_count'],tape_sha256=record['tape_sha256'],initial_grid_sha256=record['initial_grid_sha256'],
   diagnostics=record['diagnostics'],numerical_maxima=record.get('numerical_maxima'),failed_guard=record.get('failed_guard')))
 summaries=[]
 for (algorithm,condition,kind),rows in groups.items():
  summary=dict(algorithm=algorithm,condition=condition,kind=kind,n_attempted=5,
   n_training_completed=sum(r['training_status']=='completed' for r in rows),
   n_final_evaluation_available=sum(r['evaluation_status']=='available' for r in rows))
  for metric in ('greedy_return','stochastic_return','greedy_success','stochastic_success'):
   observed=[r[metric] for r in rows if r[metric] is not None]
   summary[metric+'_mean']=statistics.mean(observed) if observed else None
   summary[metric+'_min']=min(observed) if observed else None
   summary[metric+'_max']=max(observed) if observed else None
   summary[metric+'_n']=len(observed)
  summaries.append(summary)
 write_csv(dest/'numerical-endpoints.csv',endpoints);write_csv(dest/'numerical-checkpoint-curves.csv',curves);write_csv(dest/'numerical-condition-summary.csv',summaries)
 save(dest/'NUMERICAL_ANALYSIS.json',dict(status='complete',attempted_runs=110,
  completed_training_runs=sum(r['training_status']=='completed' for r in endpoints),
  unavailable_final_evaluations=sum(r['evaluation_status']!='available' for r in endpoints),
  checkpoint_rows=550,conditions=summaries,diagnostics=diagnostics,
  statistical_unit='five predeclared paired seed groups peralgorithm, separate tasks; no population success guarantee or significance claim',
  intervals='observed five-seed minima/maxima, not confidence intervals',
  missing_rule='NA with explicit available n; no imputation or exclusion from attempted denominator'))
 print(json.dumps(dict(training_completed=sum(r['training_status']=='completed' for r in endpoints),conditions=len(summaries),outputs=str(dest))))

if __name__=='__main__':main()
