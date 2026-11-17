"""Independent reconciliation of frozen inputs, holdouts, scores and ranking traces."""
import hashlib
import json
import math
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import DATA,RESULTS

def manual_spearman(x,y):
    def ranks(a):
        a=np.asarray(a,dtype=float);order=np.argsort(a,kind='stable');out=np.empty(len(a))
        start=0
        while start<len(a):
            stop=start+1
            while stop<len(a) and a[order[stop]]==a[order[start]]:stop+=1
            out[order[start:stop]]=(start+stop-1)/2+1;start=stop
        return out
    if len(set(y))<2:return np.nan
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])

def verify():
    run=json.loads((RESULTS/'training_run.json').read_text());assert run['phase']=='evaluated'
    audit=json.loads((RESULTS/'data_audit.json').read_text());assert audit['status']=='complete'
    assert run['frozen_utc']<run['evaluated_utc']
    assert hashlib.sha256((DATA/'fit_features.parquet').read_bytes()).hexdigest()==run['fit_features_sha256']
    fitting=pd.read_parquet(DATA/'fit_features.parquet');full=pd.read_parquet(DATA/'features.parquet')
    assert 'test' not in set(fitting.split)
    assert len(set(fitting.date_key)&set(full.loc[full.split.eq('test'),'date_key']))==0
    assert full.row_key.nunique()==len(full)
    assert full.tx_grid.str.fullmatch('[A-R]{2}[0-9]{2}').all()
    assert full.rx_grid.str.fullmatch('[A-R]{2}[0-9]{2}').all()
    assert full.snr2500.notna().all()
    assert audit['actual_dates']>=300
    assert (full.available_boundary<=full.hour).all()
    predictions=pd.read_parquet(DATA/'test_predictions.parquet')
    lookup=full.set_index('row_key')
    target=predictions.row_key.map(lookup.snr2500)
    np.testing.assert_array_equal(predictions.snr2500,target)
    assert predictions.date_key.str.startswith('2025-').all()
    assert ~predictions.date_key.str.endswith('-15').any()
    metrics=pd.read_csv(RESULTS/'test_metrics.csv')
    metric_checks=0;artifact_checks=0;calibration_checks=0
    for scenario,settings in run['scenarios'].items():
        tr=fitting[fitting.split.eq('train')]
        cal=fitting[fitting.split.eq('calibration')]
        if scenario=='station_holdout':tr=tr[~tr.station_held];cal=cal[~cal.station_held]
        if scenario=='geographic_holdout':tr=tr[~tr.geographic_held];cal=cal[~cal.geographic_held]
        support=json.loads((DATA/'models'/f'{scenario}-training-support.json').read_text())
        assert set(support['stations'])==set(tr.tx_sign)|set(tr.rx_sign)
        assert set(support['fields'])==set(tr.tx_grid.str[:2])|set(tr.rx_grid.str[:2])
        assert len(tr)==settings['train_rows']
        for name,info in settings['files'].items():
            path=DATA/info['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
            bundle=joblib.load(path);model=bundle['model'];artifact_checks+=1
            if model.effects:
                assert set(model.effects.tx)<=set(tr.tx_key)
                assert set(model.effects.rx)<=set(tr.rx_key)
            for view,correct in [('population_core',False),('station_aware',True)]:
                with threadpool_limits(limits=4):p=model.predict(cal,station_correction=correct)
                errors=np.abs(cal.snr2500.to_numpy()-p)
                groups=(cal.source+'|'+cal['mode']).to_numpy()
                for group,ci in bundle['calibration'][view].items():
                    e=np.sort(errors[groups==group]);k=min(len(e),math.ceil((len(e)+1)*.8))
                    assert np.isclose(e[k-1],ci['half_width_db'],atol=1e-9)
                    calibration_checks+=1
        if scenario=='station_holdout':
            g=predictions[predictions.scenario.eq(scenario)]
            unseen1=~g.tx_sign.isin(support['stations']);unseen2=~g.rx_sign.isin(support['stations'])
            assert (unseen1|unseen2).all()
            both=g.selection.eq('both_endpoints_unseen')
            assert (unseen1[both]&unseen2[both]).all()
        if scenario=='geographic_holdout':
            g=predictions[predictions.scenario.eq(scenario)]
            assert g.row_key.map(lookup.geographic_held).all()
    keys=['scenario','selection','model','view','source','mode']
    grouped=predictions.groupby(keys,sort=False)
    for row in metrics.itertuples():
        key=tuple(getattr(row,k) for k in keys);g=grouped.get_group(key)
        err=g.prediction.to_numpy()-g.snr2500.to_numpy()
        assert len(g)==row.rows and g.date_key.nunique()==row.days
        assert np.isclose(abs(err).mean(),row.mae_db,atol=1e-10)
        assert np.isclose(np.sqrt(np.mean(err**2)),row.rmse_db,atol=1e-10)
        valid=g.lower.notna()&g.upper.notna()
        if valid.any():
            coverage=((g.loc[valid,'snr2500']>=g.loc[valid,'lower'])&(g.loc[valid,'snr2500']<=g.loc[valid,'upper'])).mean()
            assert np.isclose(coverage,row.interval_coverage)
        metric_checks+=1
    rank=pd.read_csv(RESULTS/'test_hour_ranking.csv')
    traced=rank[rank.cohort.eq('pair_day_cohort')]
    ranking_checks=0
    for row in traced.itertuples():
        actual=json.loads(row.actual_snr_by_hour);pred=json.loads(row.predicted_snr_by_hour)
        assert set(actual)==set(pred) and len(actual)==row.observed_hours and len(actual)>=6
        hours=sorted(actual);a=[actual[h] for h in hours];b=[pred[h] for h in hours]
        corr=manual_spearman(a,b)
        assert (np.isnan(corr) and np.isnan(row.spearman)) or np.isclose(corr,row.spearman,atol=1e-10)
        best=max(range(len(b)),key=lambda i:b[i])
        assert np.isclose(max(a)-a[best],row.best_observed_hour_shortfall_db)
        ranking_checks+=1
    assert ranking_checks>100
    result={'status':'verified','positive_reception_days':audit['actual_dates'],
        'hourly_rows':len(full),'prediction_rows':len(predictions),'metric_groups':metric_checks,
        'artifacts_checked':artifact_checks,'calibration_quantiles_recomputed':calibration_checks,
        'ranking_traces_recomputed':ranking_checks,
        'test_metrics_sha256':hashlib.sha256((RESULTS/'test_metrics.csv').read_bytes()).hexdigest(),
        'ranking_sha256':hashlib.sha256((RESULTS/'test_hour_ranking.csv').read_bytes()).hexdigest(),
        'checks':['fitting file has no final test rows','disjoint fitting/test calendar dates',
            'frozen-before-test timestamp','artifact checksums','source target reconciliation',
            'station/geographic exclusion in both roles','training-only nuisance keys',
            'independently recomputed calibration quantiles','MAE/RMSE and interval coverage',
            'manual tied-rank correlation and best-hour shortfall','locator and solar-boundary validity']}
    (RESULTS/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

def verify_ranking_sources():
    files=sorted((DATA/'ranking'/'processed').glob('cohort-2025-*.parquet'))
    d=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    d['date_key']=d.hour.dt.strftime('%Y-%m-%d')
    known=d.power_dbm.notna()
    d.loc[known,'snr2500']+=30-d.loc[known,'power_dbm']
    keys=['source','mode','date_key','tx_sign','rx_sign','tx_grid','rx_grid']
    actual={}
    for key,g in d.groupby(keys,sort=False):
        values=g.groupby('hour').snr2500.median()
        actual[key]={str(t):float(v) for t,v in values.items()}
    cases=pd.read_csv(RESULTS/'test_hour_ranking.csv')
    cases=cases[cases.cohort.eq('pair_day_cohort')&cases.unit.eq('station_pair')]
    n=0
    for row in cases.itertuples():
        key=(row.source,row.mode,row.date,row.endpoint1,row.endpoint2,row.tx_grid,row.rx_grid)
        expected=actual[key];recorded=json.loads(row.actual_snr_by_hour)
        assert set(expected)==set(recorded)
        for hour,value in expected.items():assert np.isclose(value,recorded[hour],atol=1e-12)
        n+=1
    path=RESULTS/'verification.json';result=json.loads(path.read_text())
    result['ranking_source_curves_checked']=n
    result['test_metrics_sha256']=hashlib.sha256((RESULTS/'test_metrics.csv').read_bytes()).hexdigest()
    result['ranking_sha256']=hashlib.sha256((RESULTS/'test_hour_ranking.csv').read_bytes()).hexdigest()
    result['checks'].append('primary actual ranking curves reconciled to independently sampled source observations')
    path.write_text(json.dumps(result,indent=2))
    print('Verified',n,'primary ranking curves against source observations')

if __name__=='__main__':verify();verify_ranking_sources()
