import copy
import csv
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUNCT = re.compile(r'[^\w\s]|_', re.UNICODE)

def words(text):
    return len(PUNCT.sub(' ',unicodedata.normalize('NFC',text or '').lower()).split())

@lru_cache(None)
def standard_denominators():
    wanted={name:set((ROOT/'splits/v1'/f'{name}.txt').read_text().split()) for name in ['dial.test','dial.seen']}
    counts={name:0 for name in wanted}
    for line in (ROOT/'layers/dial.standard.v2.jsonl').open():
        row=json.loads(line)
        for name,ids in wanted.items():
            if row['id'] in ids: counts[name]+=words(row.get('std',row.get('text','')))
    return counts

def exact_wer(value,denominator):
    n=int(denominator)
    if not 0<n<100000: raise ValueError(f'Cannot unambiguously recover an error count with denominator {n}')
    e=round(float(value)*n/100)
    exact=100*e/n
    if abs(exact-float(value))>0.00050001:
        raise ValueError(f'Registered WER {value} incompatible with denominator {n}')
    return exact

@lru_cache(None)
def scores():
    out={}
    for row in csv.DictReader((ROOT/'registry/scores.csv').open()):
        run,test=row['eval_id'].split('@'); metric=row['metric'];v=float(row['value'])
        if metric in ['wer_dialect','wer_standard','wer_either'] and row['n_words']:
            n=int(row['n_words'])
            if metric=='wer_standard' and test.startswith('dial.'):
                n=standard_denominators()[test.removesuffix('.v1')]
            v=exact_wer(v,n)
        out[(run,test,metric)]=v
    return out

@lru_cache(None)
def table():
    out={}
    for (run,test,metric),value in scores().items():
        if metric not in ['wer_dialect','wer_standard','wer_either']:continue
        model=run.replace('ltd26/','').rsplit('/a',1)[0];name=test.removesuffix('.v1')
        if name.startswith('dial.'):
            out.setdefault(model,{}).setdefault(name,{})[metric.removeprefix('wer_')]=value
        elif metric=='wer_dialect':out.setdefault(model,{})[name]=value
    return out

def customer_results():
    r=json.loads((ROOT/'analysis/calls_results.json').read_text());sc=scores()
    for run,values in r['seed_wer'].items(): values['all']=sc[(run,'calls.inhouse.v1','wer_dialect')]
    for group,values in r['wer'].items():
        if group.startswith('stock '):
            run='B/parakeet-tdt-0.6b-v3' if group=='stock parakeet' else 'B/whisper-large-v3'
            vals=[sc[(run,'calls.inhouse.v1','wer_dialect')]]
        else:
            vals=[v['calls.inhouse'] for k,v in table().items() if k.rsplit('/s',1)[0]==group]
        if not vals:raise KeyError(group)
        values['all']['wer']=sum(vals)/len(vals)
    for key,values in r['contrasts'].items():
        a,b=key.split(' - '); values['all']['delta']=r['wer'][a]['all']['wer']-r['wer'][b]['all']['wer']
    return r
