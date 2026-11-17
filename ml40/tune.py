"""Bounded six-parameter search; date-disjoint validation; sealed WSPR final test.

python -m ml40.tune fit
python -m ml40.tune evaluate
Existing released artifacts are never overwritten.
"""
import argparse
import copy
import datetime as dt
import itertools
import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .plan import ROOT,DATA,RESULTS
from .models import Encoder,ModelBundle,StationEffects,sample_weights,crossfit_residuals
from .prepare import annotate
from .train import calibrate
from .test500 import digest
from .tune_collect import DATES

OUT=RESULTS/'tuning';CACHE=DATA/'tuning';OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
SPACE={'max_iter':[180,360,600],'learning_rate':[.03,.07,.10],
    'max_leaf_nodes':[15,31,63],'min_samples_leaf':[30,100,300],
    'l2_regularization':[1,10,50],'station_penalty':[10,50,200]}
REFERENCE={'max_iter':180,'learning_rate':.07,'max_leaf_nodes':23,'min_samples_leaf':60,
    'l2_regularization':10,'station_penalty':50}

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

def parameter_trials():
    combinations=[dict(zip(SPACE,v)) for v in itertools.product(*SPACE.values())]
    indexes=np.random.default_rng(40).choice(len(combinations),36,replace=False)
    return [REFERENCE]+[combinations[i] for i in indexes]

def screen_sample(d):
    parts=[]
    for _,g in d.groupby(['source','mode'],sort=True):
        cap=math.ceil(40000/g.date_key.nunique())
        part=g.sort_values('row_key').groupby('date_key',sort=True).head(cap)
        parts.append(part.sort_values('row_key').head(40000))
    return pd.concat(parts).sort_values('row_key').reset_index(drop=True)

class RankPanel:
    """Fixed station paths, observed hours only, reported-power normalized to 1 W."""
    def __init__(self,d):
        keys=['source','mode','date_key','tx_sign','rx_sign','tx_grid','rx_grid']
        self.d=d.reset_index(drop=True)
        self.shift=np.where(d.power_dbm.notna(),30-d.power_dbm.fillna(30),0)
        w=self.d[keys+['hour']].copy();w['actual']=self.d.snr2500.to_numpy()+self.shift
        w['hour_index']=w.groupby(keys+['hour'],sort=True).ngroup()
        self.hour_index=w.hour_index.to_numpy()
        h=w.groupby('hour_index',sort=True).agg({**{k:'first' for k in keys},'hour':'first','actual':'median'}).reset_index()
        h['case']=h.groupby(keys,sort=True).ngroup()
        eligible=h.groupby('case').actual.agg(['size','nunique'])
        good=eligible[(eligible['size']>=6)&(eligible['nunique']>=3)].index
        h=h[h['case'].isin(good)].copy()
        self.h=h
        # Hours are local to a day: use UTC hour rather than full timestamps in columns.
        h['utc_hour']=h.hour.dt.hour
        self.actual=h.pivot(index='case',columns='utc_hour',values='actual')
        self.ranks=self.actual.rank(axis=1);self.ranks=self.ranks.sub(self.ranks.mean(axis=1),axis=0)
        self.days=h.groupby('case').date_key.first().reindex(self.actual.index)
    def score(self,prediction):
        p=pd.Series(np.asarray(prediction)+self.shift).groupby(self.hour_index).median()
        h=self.h[['case','utc_hour','hour_index']].copy();h['pred']=h.hour_index.map(p)
        m=h.pivot(index='case',columns='utc_hour',values='pred').reindex_like(self.actual)
        ranks=m.rank(axis=1);ranks=ranks.sub(ranks.mean(axis=1),axis=0)
        corr=(self.ranks*ranks).sum(axis=1)/np.sqrt((self.ranks**2).sum(axis=1)*(ranks**2).sum(axis=1))
        corr=corr.replace([np.inf,-np.inf],np.nan)
        losses=[]
        for a,b in zip(self.actual.to_numpy(),m.to_numpy()):
            losses.append(float(np.nanmax(a)-a[np.nanargmax(b)]))
        return pd.DataFrame({'date':self.days,'spearman':corr,'best_hour_shortfall_db':losses}).reset_index(drop=True)

def score(d,model,rankpanel):
    p=model.predict(d,station_correction=False);err=abs(d.snr2500.to_numpy()-p);weights=sample_weights(d)
    result={'balanced_mae_db':float(np.average(err,weights=weights)),
        'mode_mae_db':{mode:float(np.average(err[d['mode'].eq(mode)],weights=weights[d['mode'].eq(mode)])) for mode in sorted(set(d['mode']))}}
    ranks=rankpanel.score(model.predict(rankpanel.d,station_correction=False))
    result['rank_median']=float(ranks.spearman.median());result['rank_cases']=len(ranks)
    result['quarter_rank_median']={str(k):float(v) for k,v in ranks.groupby(pd.to_datetime(ranks.date).dt.quarter).spearman.median().items()}
    result['quarter_mae_db']={str(q):float(np.average(err[d.hour.dt.quarter.eq(q)],weights=weights[d.hour.dt.quarter.eq(q)])) for q in sorted(set(d.hour.dt.quarter))}
    return result

def resources(d,label):
    path=CACHE/f'{label}-resources.joblib';support=__import__('hashlib').sha256('\n'.join(d.row_key).encode()).hexdigest()
    if path.exists():
        r=joblib.load(path)
        if r['support']==support:return r
    encoder=Encoder().fit(d);x=encoder.transform(d);y=d.snr2500.to_numpy();w=sample_weights(d)
    print('Cross-fitting training-only station residuals:',label,len(d),flush=True)
    residual=crossfit_residuals(d,x,y,w)
    r={'support':support,'encoder':encoder,'residual':residual,'weights':w}
    joblib.dump(r,path);return r

def fit_one(d,r,params):
    effects=StationEffects(penalty=params['station_penalty']).fit(d,r['residual'],r['weights'])
    target=d.snr2500.to_numpy()-effects.predict(d)
    settings={k:v for k,v in params.items() if k!='station_penalty'}
    model=HistGradientBoostingRegressor(loss='absolute_error',early_stopping=False,random_state=40,**settings)
    model.fit(r['encoder'].transform(d),target,sample_weight=r['weights'])
    return ModelBundle(r['encoder'],model,effects)

def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))

def eligible(score,reference):
    # Predeclared non-regression guard. Rank is the primary objective.
    return (score['balanced_mae_db']<=reference['balanced_mae_db']*1.02 and
        all(score['mode_mae_db'][m]<=v*1.05 for m,v in reference['mode_mae_db'].items()) and
        sum(score['quarter_rank_median'][q]>=v-.02 for q,v in reference['quarter_rank_median'].items())>=3)

def fit():
    fitpath=DATA/'fit_features.parquet';artifact=ROOT/'artifacts/gocap-40m.joblib'
    plan={'version':1,'seed':40,'parameter_space':SPACE,'trials':parameter_trials(),
        'fit_sha256':digest(fitpath),'reference_sha256':digest(artifact),
        'screen_cap_per_source_mode':40000,'full_training_finalists':3,
        'fresh_test_dates':DATES,'fresh_source':'WSPR only; event-hash SNR sample plus pair/day-hash ranking cohort',
        'selection':'Highest median validation WSPR hourly correlation, subject to balanced MAE <= reference *1.02; each mode MAE <= reference *1.05; >=3 quarters rank correlation >= reference -0.02. Screen uses its own reference; full stage uses released reference.',
        'promotion':'Fresh WSPR ranking paired date-bootstrap 95% lower bound >0 and fresh SNR MAE <= reference *1.02; otherwise keep reference.',
        'test_policy':'Do not read fresh processed test data or predict until selected model and checksums frozen. Published 500 cases are regression only.',
        'positive_only':True}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen tuning plan changed')
    save_json(path,plan)
    d=pd.read_parquet(fitpath);assert 'test' not in set(d.split)
    tr=d[d.split.eq('train')].reset_index(drop=True);val=d[d.split.eq('validation')].reset_index(drop=True)
    cal=d[d.split.eq('calibration')].reset_index(drop=True)
    assert not set(d.date_key)&set(DATES)
    rankfiles=sorted((DATA/'ranking'/'processed').glob('cohort-2024-*-08.parquet'))
    rd=annotate(pd.concat([pd.read_parquet(p) for p in rankfiles],ignore_index=True))
    assert rd.split.eq('validation').all();panel=RankPanel(rd)
    incumbent=joblib.load(artifact)
    with threadpool_limits(limits=4):
        reference=score(val,incumbent['model'],panel)
        screen=screen_sample(tr);r=resources(screen,'screen')
        results=[]
        for i,params in enumerate(plan['trials']):
            checkpoint=CACHE/f'screen-{i:02d}.joblib'
            if checkpoint.exists():
                saved=joblib.load(checkpoint);assert saved['params']==params
                sc=saved['score']
            else:
                model=fit_one(screen,r,params);sc=score(val,model,panel)
                joblib.dump({'params':params,'score':sc},checkpoint)
            results.append({'trial':i,'params':params,'score':sc})
            save_json(OUT/'screen_results.json',results)
            print('Screen',i,'rank',round(sc['rank_median'],4),'MAE',round(sc['balanced_mae_db'],4),params,flush=True)
        candidates=[x for x in results[1:] if eligible(x['score'],results[0]['score'])]
        if len(candidates)<3:
            # Do not loosen gates; extra unconstrained finalists are diagnostic only.
            candidates+=sorted([x for x in results[1:] if x not in candidates],key=lambda x:-x['score']['rank_median'])[:3-len(candidates)]
        finalists=sorted(candidates,key=lambda x:(-x['score']['rank_median'],x['score']['balanced_mae_db']))[:3]
        full=resources(tr,'full');full_results=[]
        for candidate in finalists:
            i=candidate['trial'];path=CACHE/f'full-{i:02d}.joblib'
            if path.exists():
                saved=joblib.load(path);assert saved['params']==candidate['params']
                model=saved['model'];sc=saved['score']
            else:
                print('Full training finalist',i,candidate['params'],flush=True)
                model=fit_one(tr,full,candidate['params']);sc=score(val,model,panel)
                joblib.dump({'params':candidate['params'],'model':model,'score':sc},path)
            full_results.append({'trial':i,'params':candidate['params'],'score':sc,'eligible':eligible(sc,reference)})
            print('Full',i,'rank',sc['rank_median'],'MAE',sc['balanced_mae_db'],'eligible',eligible(sc,reference),flush=True)
        acceptable=[x for x in full_results if x['eligible'] and x['score']['rank_median']>reference['rank_median']]
        chosen=max(acceptable,key=lambda x:(x['score']['rank_median'],-x['score']['balanced_mae_db'])) if acceptable else None
        run={'phase':'frozen_before_fresh_test','frozen_utc':utc(),'reference_validation':reference,
            'screen_rows':len(screen),'training_rows':len(tr),'validation_rows':len(val),
            'screen_train_days':screen.date_key.nunique(),'finalists':full_results,'chosen':chosen,
            'reference_sha256':digest(artifact),'fit_sha256':digest(fitpath),'strict_validation':{}}
        if chosen:
            model=joblib.load(CACHE/f'full-{chosen["trial"]:02d}.joblib')['model']
            bundle={'model':copy.copy(model),'calibration':calibrate(cal,model),
                'training_domain':incumbent['training_domain'],'target':incumbent['target'],
                'scenario':'tuned_standard','source_features_sha256':digest(fitpath)}
            bundle['model'].effects=None
            bundle['calibration']={'population_core':bundle['calibration']['population_core']}
            output=ROOT/'artifacts/gocap-40m-tuned.joblib';joblib.dump(bundle,output)
            run['tuned_sha256']=digest(output)
            # Strict station/field stress models are fitted and assessed on validation only.
            for scenario,column in [('station_holdout','station_held'),('geographic_holdout','geographic_held')]:
                st=tr[~tr[column]].reset_index(drop=True);sv=val[val[column]].reset_index(drop=True)
                sr=rd[rd[column]].reset_index(drop=True);sp=RankPanel(sr)
                res=resources(st,scenario)
                current=fit_one(st,res,REFERENCE);challenger=fit_one(st,res,chosen['params'])
                run['strict_validation'][scenario]={'train_rows':len(st),'validation_rows':len(sv),
                    'reference':score(sv,current,sp),'tuned':score(sv,challenger,sp)}
                save_json(OUT/'run-progress.json',run)
        # Frozen timestamp follows every validation decision; final test has not been loaded.
        run['frozen_utc']=utc();save_json(OUT/'run.json',run)
    print('Tuning frozen:',chosen,'Fresh test remains sealed.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','evaluate']);a=p.parse_args()
    if a.action=='fit':fit()
    else:
        from .tune_evaluate import evaluate
        evaluate()
