"""Fit and select on absolute hourly errors only; freeze before new-date scoring."""
import copy,json,time,datetime as dt
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import DATA,ROOT
from .hourly_plan import BASE,OUT,CORRECTION_DATES,TEST_DATES,plan,digest
from .hourly_models import NativeModel,ResidualModel,HourlyIntervals,metric
from .v3_features import enrich


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))


def fit():
    definition=plan();d=pd.read_parquet(DATA/'fit_features.parquet')
    assert 'test' not in set(d.split)
    tr=d[d.split.eq('train')].reset_index(drop=True)
    vd=enrich(d[d.split.eq('validation')].reset_index(drop=True))
    cd=enrich(d[d.split.eq('calibration')].reset_index(drop=True))
    vd.to_parquet(BASE/'validation_features.parquet',index=False)
    cd.to_parquet(BASE/'calibration_features.parquet',index=False)
    trpath=BASE/'original_train_features.parquet'
    if trpath.exists():tr=pd.read_parquet(trpath)
    else:tr=enrich(tr);tr.to_parquet(trpath,index=False)
    assert not set(CORRECTION_DATES)&set(d.date_key)
    assert not set(TEST_DATES)&set(d.date_key)
    reference=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
    baseline=joblib.load(DATA/'models/standard-boosted_baseline.joblib')['model']
    models={'reference':reference['model'],'existing_boosted_baseline':baseline}
    results={};selection={};support={}
    with threadpool_limits(limits=4):
        for mode in ['WSPR','FT8','FT4','CW']:
            train=tr[tr['mode'].eq(mode)].reset_index(drop=True)
            val=vd[vd['mode'].eq(mode)].reset_index(drop=True)
            variants=[('native_base_day',False,'day',False),('native_rich_day',True,'day',False),
                ('native_rich_hour',True,'hour',False),('native_rich_observation',True,'observation',False),
                ('native_rich_small',True,'hour',True)]
            for name,rich,weighting,small in variants:
                key=mode+'_'+name;path=BASE/f'{key}.joblib'
                if path.exists():model=joblib.load(path)
                else:
                    print('Fitting native hourly SNR',key,len(train),'rows',flush=True)
                    model=NativeModel(rich,weighting,small).fit(train);joblib.dump(model,path)
                models[key]=model
            if mode=='WSPR':
                for date in CORRECTION_DATES:
                    while not (BASE/'processed'/f'correction-{date}.meta.json').exists():
                        print('Waiting for independent correction date',date,flush=True);time.sleep(20)
                correction=pd.concat([pd.read_parquet(BASE/'processed'/f'correction-{date}.parquet') for date in CORRECTION_DATES],ignore_index=True)
                assert not set(correction.date_key)&(set(d.date_key)|set(TEST_DATES))
                correction.to_parquet(BASE/'correction_train_features.parquet',index=False)
                support['correction']={'rows':len(correction),'days':int(correction.date_key.nunique())}
                path=BASE/'independent_residual_full.joblib'
                if path.exists():full=joblib.load(path)
                else:
                    print('Fitting correction on genuinely out-of-reference-training predictions',len(correction),'rows',flush=True)
                    full=ResidualModel(reference['model']).fit(correction);joblib.dump(full,path)
                half=copy.copy(full);half.strength=.5
                models['independent_residual_full']=full;models['independent_residual_half']=half
                joblib.dump(half,BASE/'independent_residual_half.joblib')
            results[mode]={}
            candidate_names=['reference','existing_boosted_baseline']+[mode+'_'+v[0] for v in variants]
            if mode=='WSPR':candidate_names+=['independent_residual_half','independent_residual_full']
            for name in candidate_names:
                sc=metric(val,models[name].predict(val,station_correction=False));results[mode][name]=sc
                print('Absolute validation',mode,name,sc,flush=True)
            ref=results[mode]['reference']
            valid=[name for name,sc in results[mode].items() if sc['mae_db']<=ref['mae_db']*1.02 and sc['rmse_db']<=ref['rmse_db']*1.03]
            chosen=min(valid,key=lambda n:results[mode][n]['hour_date_mae_db'])
            selection[mode]=chosen
            support[mode]={'training_rows':len(train),'training_days':int(train.date_key.nunique())}
            save('validation_progress.json',{'results':results,'selection':selection,'support':support})
        print('Selected exclusively by absolute hourly errors:',selection,flush=True)
        intervals=HourlyIntervals().fit(cd,models,selection)
        chosen_models={name:models[name] for name in set(selection.values())}
        package={'version':'hourly-snr-1','models':chosen_models,'profile_selection':selection,'intervals':intervals,
            'training_domain':reference['training_domain'],
            'calibration_limit':'Positive reported-reception population; actual 100W/dipole antennas and receiver noise remain uncalibrated.',
            'selection_objective':'Absolute hourly SNR MAE, not hourly ranking'}
        artifact=ROOT/'artifacts/gocap-40m-hourly-experimental.joblib';joblib.dump(package,artifact)
    run={'phase':'frozen_before_new_test','frozen_utc':utc(),'selection':selection,'validation':results,'support':support,
        'interval_calibration':intervals.half_width,'reference_sha256':definition['reference_sha256'],
        'fit_sha256':definition['fit_sha256'],'plan_sha256':digest(OUT/'plan.json'),
        'artifact_sha256':digest(artifact),'training_feature_sha256':digest(trpath),
        'correction_features_sha256':digest(BASE/'correction_train_features.parquet'),
        'code_sha256':{name:digest(ROOT/'ml40'/name) for name in ['hourly_models.py','hourly_train.py','v3_features.py']}}
    save('run.json',run);print('Hourly SNR candidates frozen before new test scoring.',flush=True)
if __name__=='__main__':fit()
