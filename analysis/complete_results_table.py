import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'registry/scores.csv').open()))
from publication_data import scores
lookup=scores()
order=['S','S+D12.dial','S+D25.dial','S+D50.dial','S+D80.dial','S+X80','S+D25.std','S+D50.std','S+D80.std','D80.dial','D80.std','S+D80-A.std','S+D80-D.std','S+D80-S.std','S+D80-Z.std']
runs=[r for r in csv.DictReader((ROOT/'registry/runs.csv').open()) if r['status']=='completed' and r['hub_artifact_id'] and '/smoke/' not in r['run_id']]
specs=[('dial.test.v1','wer_dialect'),('dial.test.v1','wer_standard'),('dial.test.v1','wer_either'),('dial.seen.v1','wer_either'),('phone.test.v1','wer_dialect'),('calls.inhouse.v1','wer_dialect'),('fleurs.lt.test.v1','wer_dialect'),('cv.lt.test.v1','wer_dialect')]
def label(key):
 if key=='S':return 'Telephone only'
 recipe=json.loads((ROOT/'recipes'/f'{key}.json').read_text())
 hours=sum(p['hours'] for p in recipe['parts'] if p['code']!='S')
 if key=='S+X80':return f'Telephone + other spontaneous speech, {hours:.2f} h'
 spelling='dialect' if key.endswith('.dial') else 'standard'
 if key.startswith('D80'):return f'Dialect only, {hours:.2f} h; {spelling} spelling'
 if '-' in key:
  region={'A':'Aukštaitija','D':'Dzūkija','S':'Suvalkija','Z':'Žemaitija'}[key.split('-')[1].split('.')[0]]
  return f'Telephone + dialect excluding {region}; standard spelling'
 return f'Telephone + dialect, {hours:.2f} h; {spelling} spelling'
lines=[r'\begin{tabular}{>{\raggedright\arraybackslash}p{83mm} r *{8}{r}}',r'\toprule',r'& & \multicolumn{3}{c}{Dialect test: accepted spelling} & Training & LIEPA-3 & Customer & & Common\\',r'\cmidrule(lr){3-5}',r'Model and training data & Run & Dialect & Standard & Either & speakers & telephone & service & FLEURS & Voice\\',r'\midrule']
def add(run,lab,seed):
 vals=[lookup[(run,t,m)] for t,m in specs]
 lines.append(lab+' & '+seed+' & '+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
add('B/parakeet-tdt-0.6b-v3','Stock Parakeet-TDT','--')
add('B/whisper-large-v3','Stock Whisper large-v3','--')
for key in order:
 for run in sorted((r for r in runs if r['run_id'].split('/')[1]==key),key=lambda r:r['run_id']):
  add(run['run_id'],label(key),run['run_id'].split('/')[2][1:])
lines +=[r'\bottomrule',r'\end{tabular}']
(ROOT/'paper/generated/tab_complete_results.tex').write_text('\n'.join(lines)+'\n')
print('Generated full results for two stock models and',len(runs),'fine-tuned runs.')
