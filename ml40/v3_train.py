"""Ablations, validation-selected SNR/ranking components and split-calibrated quantiles."""
import json,time,datetime as dt
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import ROOT,DATA
from .v3_plan import BASE,OUT,TRAIN_DATES,TEST_DATES,plan
from .v3_features import enrich,RICH_NUMERIC
from .v3_models import SNRModel,HourRanker,QuantileIntervals,rank_training,profile_predict
from .prepare import annotate
from .models import sample_weights
from .tune import RankPanel
from .test500 import digest


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))

def usable(d):
    return d[~d['mode'].eq('WSPR')|d.power_dbm.notna()].reset_index(drop=True)


def ranking_score(panel,prediction,is_ranker=False):
    p=np.asarray(prediction,dtype=float)
    if is_ranker:p=p+panel.d.power_dbm.fillna(30).to_numpy()-30
    return panel.score(p)


def error_summary(d,prediction):
    w=sample_weights(d);e=abs(d.snr2500.to_numpy()-prediction)
    out={'balanced_mae_db':float(np.average(e,weights=w)), 'mae_db':float(e.mean()),'by_mode':{}}
    for mode in sorted(set(d['mode'])):
        use=d['mode'].eq(mode).to_numpy()
        out['by_mode'][mode]={'balanced_mae_db':float(np.average(e[use],weights=w[use])),'rows':int(use.sum())}
    return out


def cached_features(name,d):
    path=BASE/f'{name}.parquet';meta=path.with_suffix('.meta.json')
    schema=digest(ROOT/'ml40/v3_features.py')
    # A fresh source signature also covers actual targets and all historical inputs.
    source=__import__('hashlib').sha256(pd.util.hash_pandas_object(d,index=False).values.tobytes()).hexdigest()
    if path.exists() and meta.exists() and json.loads(meta.read_text())=={'schema':schema,'source':source}:
        return pd.read_parquet(path)
    enriched=enrich(d);enriched.to_parquet(path,index=False)
    meta.write_text(json.dumps({'schema':schema,'source':source}));return enriched


def fit():
    definition=plan()
    for date in TRAIN_DATES:
        p=BASE/'processed'/f'train-coherent-{date}.meta.json'
        while not p.exists():
            print('Waiting for coherent training collection:',date,flush=True);time.sleep(20)
    main=pd.read_parquet(DATA/'fit_features.parquet');assert 'test' not in set(main.split)
    parts=[pd.read_parquet(BASE/'processed'/f'train-coherent-{d}.parquet') for d in TRAIN_DATES]
    coherent=usable(pd.concat([p for p in parts if len(p)],ignore_index=True))
    assert coherent.split.eq('train').all() and not set(coherent.date_key)&set(TEST_DATES)
    scattered=usable(main[main.split.eq('train')&main['mode'].eq('WSPR')&main.date_key.isin(TRAIN_DATES)])
    other=usable(main[main.split.eq('train')&~main['mode'].eq('WSPR')])
    validation=usable(main[main.split.eq('validation')]);calibration=usable(main[main.split.eq('calibration')])
    rv=annotate(pd.concat([pd.read_parquet(p) for p in sorted((DATA/'ranking'/'processed').glob('cohort-2024-*-08.parquet'))],ignore_index=True))
    coherent=cached_features('coherent_train_features',coherent)
    scattered=cached_features('scattered_train_features',scattered)
    other=cached_features('other_train_features',other)
    validation=cached_features('validation_features',validation)
    calibration=cached_features('calibration_features',calibration)
    rv=cached_features('ranking_validation_features',usable(rv))
    pooled=pd.concat([coherent,other],ignore_index=True)
    assert not set(pooled.date_key)&set(validation.date_key)
    assert not set(pooled.date_key)&set(calibration.date_key)
    assert not (set(pooled.date_key)|set(validation.date_key)|set(calibration.date_key))&set(TEST_DATES)
    rt=rank_training(coherent);rt.to_parquet(BASE/'rank_training.parquet',index=False)
    panel=RankPanel(rv);models={'reference':joblib.load(ROOT/'artifacts/gocap-40m.joblib')['model']}
    configurations=[('scattered_wspr_base',scattered,False,False),('coherent_wspr_base',coherent,False,False),
        ('coherent_wspr_rich',coherent,True,False),('pooled_rich',pooled,True,False),('pooled_source_heads',pooled,True,True)]
    validation_scores={};rank_scores={};support={}
    with threadpool_limits(limits=4):
        for name,train,rich,heads in configurations:
            path=BASE/f'{name}.joblib'
            training_hash=__import__('hashlib').sha256(pd.util.hash_pandas_object(train,index=False).values.tobytes()).hexdigest()
            signature={'training_hash':training_hash,'models_code':digest(ROOT/'ml40/v3_models.py'),'rich':rich,'heads':heads}
            cached=joblib.load(path) if path.exists() else None
            if cached and cached['signature']==signature:model=cached['model']
            else:
                print('Fitting SNR ablation',name,len(train),'rows',flush=True)
                model=SNRModel(rich,heads).fit(train);joblib.dump({'signature':signature,'model':model},path)
            models[name]=model;support[name]={'rows':len(train),'days':train.date_key.nunique(),
                'source_mode_rows':train.groupby(['source','mode']).size().rename('rows').reset_index().to_dict('records')}
        for name,model in models.items():
            vd=validation if name in ['reference','pooled_rich','pooled_source_heads'] else validation[validation['mode'].eq('WSPR')]
            validation_scores[name]=error_summary(vd,model.predict(vd))
            r=ranking_score(panel,model.predict(rv));rank_scores[name]={'median_spearman':float(r.spearman.median()),
                'best_hour_shortfall_db':float(r.best_hour_shortfall_db.mean()),'cases':len(r)}
            print('Validation',name,validation_scores[name]['balanced_mae_db'],'rank',rank_scores[name]['median_spearman'],flush=True)
        # Choose profile models on validation only. No final score feeds this decision.
        profiles={mode:min([name for name,s in validation_scores.items() if mode in s['by_mode']],
            key=lambda name:validation_scores[name]['by_mode'][mode]['balanced_mae_db']) for mode in ['WSPR','FT8','FT4','CW']}
        pooled_choice=min(['pooled_rich','pooled_source_heads'],key=lambda name:validation_scores[name]['balanced_mae_db'])
        rank_models={};rank_trials=[]
        # Base-feature ranker is an ablation, richer variants are the six prespecified candidates.
        for rich,iterations,depth in [(False,240,3)]+[(True,i,d) for i in [120,240,400] for d in [3,5]]:
            name=f'ranker_{"rich" if rich else "base"}_{iterations}_{depth}';path=BASE/f'{name}.joblib'
            if path.exists():ranker=joblib.load(path)
            else:
                print('Fitting positive-only ranker',name,len(rt),'hours',rt.qid.nunique(),'groups',flush=True)
                ranker=HourRanker().fit(rt,rich,iterations,depth);joblib.dump(ranker,path)
            ranks=ranking_score(panel,ranker.predict(rv),True)
            score={'name':name,'rich':rich,'iterations':iterations,'depth':depth,
                'median_spearman':float(ranks.spearman.median()),'mean_best_hour_shortfall_db':float(ranks.best_hour_shortfall_db.mean()),'cases':len(ranks)}
            rank_trials.append(score);rank_models[name]=ranker
            print('Ranking validation',score,flush=True)
        chosen_rank=max([x for x in rank_trials if x['rich']],key=lambda s:(s['median_spearman'],-s['mean_best_hour_shortfall_db']))
        interval_path=BASE/'quantile_intervals.joblib'
        # Quantile calibration must track the selected point models.
        if interval_path.exists():
            intervals=joblib.load(interval_path)
            assert intervals.profile_selection==profiles
        else:
            print('Fitting conditional 10th/90th quantiles and independent calibration',flush=True)
            intervals=QuantileIntervals().fit(pooled,calibration,models,profiles);joblib.dump(intervals,interval_path)
        selected_models={name:models[name] for name in set(profiles.values())}
        domain={}
        for mode in ['WSPR','FT8','FT4','CW']:
            g=pooled[pooled['mode'].eq(mode)]
            domain[mode]={c:[float(g[c].min()),float(g[c].max())] for c in RICH_NUMERIC if g[c].notna().any()}
        bundle={'version':3,'models':selected_models,'profile_selection':profiles,
            'ranker':rank_models[chosen_rank['name']],'intervals':intervals,'training_domain':domain,
            'target':'SNR2500 among positive receptions; rank scores compare observed-hour strengths',
            'calibration_limit':'Unknown station antennas/noise; not independently calibrated for 100W dipoles'}
        artifact=ROOT/'artifacts/gocap-40m-v3.joblib';joblib.dump(bundle,artifact)
    run={'phase':'frozen_before_fresh_test','frozen_utc':utc(),'definition_sha256':digest(OUT/'plan.json'),
        'fit_sha256':digest(DATA/'fit_features.parquet'),'reference_sha256':definition['reference_sha256'],
        'artifact_sha256':digest(artifact),'training_support':support,'rank_training_hours':len(rt),
        'rank_training_groups':rt.qid.nunique(),'rank_training_days':rt.date_key.nunique(),
        'validation':validation_scores,'validation_ranking':rank_scores,'profile_selection':profiles,
        'pooled_choice':pooled_choice,'ranking_trials':rank_trials,'selected_ranker':chosen_rank,
        'interval_calibration':intervals.corrections,
        'training_features_sha256':{p.name:digest(p) for p in [BASE/'coherent_train_features.parquet',BASE/'scattered_train_features.parquet',BASE/'other_train_features.parquet']},
        'code_sha256':{name:digest(ROOT/'ml40'/name) for name in ['v3_models.py','v3_features.py','v3_train.py','igrf_degree1.json']}}
    save('run.json',run)
    print('All v3 components frozen before new test scoring:',profiles,chosen_rank['name'],flush=True)

if __name__=='__main__':fit()
