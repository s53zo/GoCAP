"""Untouched-date comparison of absolute SNR errors for every observed hour."""
import copy,json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import DATA,ROOT
from .hourly_plan import BASE,OUT,TEST_DATES,HARC_TEST_DATES,CORRECTION_DATES,digest,active_harc_dates
from .hourly_models import metric,predict_profiles,HourlyIntervals
from .hourly_train import save,utc
from .v3_features import enrich


def interval_metrics(d,p,lo,hi):
    y=d.snr2500.to_numpy();r=metric(d,p)
    score=hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)
    r.update({'coverage':float(((y>=lo)&(y<=hi)).mean()),'mean_interval_width_db':float((hi-lo).mean()),
        'mean_interval_score':float(score.mean())});return r


def bootstrap_cells(d,p0,p,replicates=5000):
    w=pd.DataFrame({'date':d.date_key.to_numpy(),'hour':d.hour.dt.hour.to_numpy(),
        'difference':abs(p-d.snr2500.to_numpy())-abs(p0-d.snr2500.to_numpy())})
    cells=w.groupby(['date','hour']).difference.mean().reset_index()
    grouped=cells.groupby('date').difference.agg(['sum','size'])
    rng=np.random.default_rng(40);index=rng.integers(0,len(grouped),(replicates,len(grouped)))
    values=grouped['sum'].to_numpy()[index].sum(axis=1)/grouped['size'].to_numpy()[index].sum(axis=1)
    return {'hour_date_mae_difference_db':float(cells.difference.mean()),
        '95_percent_date_bootstrap_interval':list(map(float,np.quantile(values,[.025,.975]))),
        'date_clusters':len(grouped),'observed_hour_date_cells':len(cells),'replicates':replicates}


def evaluate():
    r=json.loads((OUT/'run.json').read_text());assert r['phase'] in ['frozen_before_new_test','evaluated']
    assert digest(DATA/'fit_features.parquet')==r['fit_sha256']
    assert digest(ROOT/'artifacts/gocap-40m.joblib')==r['reference_sha256']
    assert digest(ROOT/'artifacts/gocap-40m-hourly-experimental.joblib')==r['artifact_sha256']
    assert (OUT/'stress_freeze.json').exists(),'Finish strict SNR fits first'
    assert (OUT/'fresh_harc_preparation.json').exists(),'Resolve all preselected fresh archives first'
    preparation=json.loads((OUT/'fresh_harc_preparation.json').read_text())
    assert set(active_harc_dates())<=set(preparation),'Wait for the availability-amended archive set'
    for date in TEST_DATES:assert (BASE/'processed'/f'test-{date}.meta.json').exists()
    ws=pd.concat([pd.read_parquet(BASE/'processed'/f'test-{date}.parquet') for date in TEST_DATES],ignore_index=True)
    hp=[BASE/'processed'/f'harc-test-{date}.parquet' for date in active_harc_dates()]
    hs=pd.concat([pd.read_parquet(p) for p in hp if p.exists()],ignore_index=True) if any(p.exists() for p in hp) else pd.DataFrame()
    d=pd.concat([ws,hs],ignore_index=True)
    assert set(d.date_key)<=set(TEST_DATES+active_harc_dates()) and set(ws.date_key)==set(TEST_DATES)
    old=pd.read_parquet(DATA/'features.parquet',columns=['date_key']);assert not set(d.date_key)&set(old.date_key)
    assert not set(d.date_key)&set(CORRECTION_DATES)
    from .tune_collect import DATES as oldtuning
    from .v3_plan import TEST_DATES as oldv3
    assert not set(d.date_key)&(set(oldtuning)|set(oldv3))
    d.to_parquet(BASE/'fresh_features.parquet',index=False)
    original=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
    package=joblib.load(ROOT/'artifacts/gocap-40m-hourly-experimental.joblib')
    with threadpool_limits(limits=4):
        p0=original['model'].predict(d,station_correction=False)
        p=predict_profiles(d,package['models'],package['profile_selection'])
        low,high=package['intervals'].predict(d,package['models'])
    half=np.array([original['calibration']['population_core'][x.source+'|'+x.mode]['half_width_db'] for x in d.itertuples()])
    low0=p0-half;high0=p0+half
    results={};decisions={};rows=[]
    for mode in ['WSPR','FT8','FT4','CW']:
        mask=d['mode'].eq(mode).to_numpy();part=d.loc[mask].reset_index(drop=True)
        if not len(part):results[mode]={'status':'no_fresh_source_data'};decisions[mode]=False;continue
        ref=interval_metrics(part,p0[mask],low0[mask],high0[mask]);candidate=interval_metrics(part,p[mask],low[mask],high[mask])
        delta=bootstrap_cells(part,p0[mask],p[mask])
        decisions[mode]=bool(delta['95_percent_date_bootstrap_interval'][1]<0 and candidate['mae_db']<=ref['mae_db'] and candidate['rmse_db']<=ref['rmse_db']*1.02)
        results[mode]={'reference':ref,'candidate':candidate,'difference':delta,'candidate_name':r['selection'][mode],
            'passes_per_mode_promotion':decisions[mode]}
        for hour in sorted(set(part.hour.dt.hour)):
            ix=mask & d.hour.dt.hour.eq(hour).to_numpy();sub=d.loc[ix].reset_index(drop=True)
            rows.append({'mode':mode,'utc_hour':int(hour),'rows':len(sub),'dates':int(sub.date_key.nunique()),
                'reference_mae_db':float(abs(p0[ix]-sub.snr2500.to_numpy()).mean()),
                'candidate_mae_db':float(abs(p[ix]-sub.snr2500.to_numpy()).mean()),
                'reference_bias_db':float((p0[ix]-sub.snr2500.to_numpy()).mean()),
                'candidate_bias_db':float((p[ix]-sub.snr2500.to_numpy()).mean())})
    # The WSPR promotion also requires bounded fresh non-regression in other modes,
    # as declared before the test. A failing component is retained as a challenger.
    other_guard=all(s.get('candidate',{}).get('mae_db',0)<=s.get('reference',{}).get('mae_db',0)*1.05
        for mode,s in results.items() if mode!='WSPR' and 'candidate' in s)
    decisions['WSPR']=decisions['WSPR'] and other_guard
    hybrid_selection={mode:r['selection'][mode] if decisions.get(mode,False) else 'reference' for mode in ['WSPR','FT8','FT4','CW']}
    hybrid_models={name:package['models'][name] for name in set(hybrid_selection.values()) if name!='reference'}
    hybrid_models['reference']=original['model']
    bands=copy.copy(package['intervals']);bands.selection=hybrid_selection;bands.half_width=copy.deepcopy(package['intervals'].half_width)
    for mode in hybrid_selection:
        if hybrid_selection[mode]=='reference':
            for key,ci in original['calibration']['population_core'].items():
                if key.endswith('|'+mode):bands.half_width[key]={'db':ci['half_width_db'],'rows':ci['calibration_rows']}
    recommended={'version':'hourly-snr-1','models':hybrid_models,'profile_selection':hybrid_selection,'intervals':bands,
        'training_domain':original['training_domain'],'calibration_limit':package['calibration_limit'],
        'selection_objective':'Absolute SNR for each hour; no ranking model or strongest-hour output'}
    output=ROOT/'artifacts/gocap-40m-hourly.joblib'
    predicted=d[['source','mode','date_key','hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm','snr2500','distance_km',
        'tx_local_hour' if 'tx_local_hour' in d else 'tx_lon','kp_previous','sfi_previous_day']].copy()
    for name,val in [('reference_prediction',p0),('candidate_prediction',p),('reference_lower',low0),('reference_upper',high0),('candidate_lower',low),('candidate_upper',high)]:predicted[name]=val
    predicted.to_parquet(OUT/'fresh_predictions.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'hour_by_hour_errors.csv',index=False)
    stress={}
    with threadpool_limits(limits=4):
        for name,column in [('station_holdout','station_held'),('geographic_holdout','geographic_held')]:
            s=joblib.load(BASE/f'stress-{name}.joblib');sub=ws[ws[column]].reset_index(drop=True)
            a=s['reference'].predict(sub,station_correction=False);b=s['model'].predict(sub,station_correction=False)
            stress[name]={'reference':metric(sub,a),'candidate':metric(sub,b),'difference':bootstrap_cells(sub,a,b)}
    if (OUT/'deployment_guard.json').exists():
        stress_guard=all(s['candidate']['mae_db']<=s['reference']['mae_db']*1.05 for s in stress.values())
        decisions['WSPR']=decisions['WSPR'] and stress_guard
        if not decisions['WSPR']:
            hybrid_selection['WSPR']='reference'
            recommended['profile_selection']=hybrid_selection
            recommended['models']['reference']=original['model']
            recommended['intervals'].selection=hybrid_selection
            for key,ci in original['calibration']['population_core'].items():
                if key.endswith('|WSPR'):recommended['intervals'].half_width[key]={'db':ci['half_width_db'],'rows':ci['calibration_rows']}
    for mode,decision in decisions.items():
        if 'candidate' in results[mode]:results[mode]['passes_per_mode_promotion']=decision
    joblib.dump(recommended,output)
    r.update({'phase':'evaluated','evaluated_utc':utc(),'fresh_results':results,'strict_snr_tests':stress,
        'promoted_by_mode':decisions,'recommended_selection':hybrid_selection,'recommended_artifact_sha256':digest(output),
        'fresh_feature_sha256':digest(BASE/'fresh_features.parquet'),'fresh_rows':len(d),
        'fresh_test_file_sha256':{p.name:digest(p) for p in (BASE/'processed').glob('*test*.parquet')},
        'evaluation_code_sha256':digest(ROOT/'ml40/hourly_evaluate.py')})
    save('run.json',r)
    output.with_suffix('.json').write_text(json.dumps({'sha256':digest(output),'bytes':output.stat().st_size,
        'objective':'Hour-by-hour SNR estimates, not ranking','profile_selection':hybrid_selection,
        'promoted_by_mode':decisions,'physical_calibration_verified':False,'query_module':'ml40.hourly_query'},indent=2))
    print(json.dumps({'fresh_results':results,'recommended_selection':hybrid_selection,'stress':stress},indent=2))
if __name__=='__main__':evaluate()
