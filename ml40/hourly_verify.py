"""Independent source reconciliation and absolute-hourly metric verification."""
import json,math
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .plan import ROOT,DATA
from .hourly_plan import BASE,OUT,CORRECTION_DATES,TEST_DATES,active_harc_dates,digest
from .hourly_models import predict_profiles
from .hourly_train import save


def independent_metric(d,p):
    error=[float(b)-float(a) for a,b in zip(d.snr2500,p)]
    groups={}
    for date,hour,e in zip(d.date_key,d.hour.dt.hour,error):groups.setdefault((date,int(hour)),[]).append(abs(e))
    return {'hour_date_mae_db':sum(sum(e)/len(e) for e in groups.values())/len(groups),
        'mae_db':sum(map(abs,error))/len(error),'rmse_db':math.sqrt(sum(e*e for e in error)/len(error)),
        'bias_db':sum(error)/len(error),'observed_hour_date_cells':len(groups)}


def verify():
    r=json.loads((OUT/'run.json').read_text());assert r['phase']=='evaluated'
    assert r['frozen_utc']<r['evaluated_utc']
    assert digest(ROOT/'artifacts/gocap-40m.joblib')==r['reference_sha256']
    assert digest(ROOT/'artifacts/gocap-40m-hourly-experimental.joblib')==r['artifact_sha256']
    assert digest(ROOT/'artifacts/gocap-40m-hourly.joblib')==r['recommended_artifact_sha256']
    assert digest(DATA/'fit_features.parquet')==r['fit_sha256']
    assert digest(OUT/'plan.json')==r['plan_sha256']
    for name,h in r['code_sha256'].items():assert digest(ROOT/'ml40'/name)==h
    assert digest(ROOT/'ml40/hourly_evaluate.py')==r['evaluation_code_sha256']
    amendment=json.loads((OUT/'availability_amendment.json').read_text())
    assert amendment['created_utc']<r['evaluated_utc'] and not amendment['test_predictions_inspected']
    assert amendment['active_harc_test_dates']==active_harc_dates()
    guard=json.loads((OUT/'deployment_guard.json').read_text());assert guard['declared_utc']<r['evaluated_utc']
    assert not guard['fresh_scores_seen']
    strict=json.loads((OUT/'stress_freeze.json').read_text());assert strict['frozen_utc']<r['evaluated_utc']
    native=pd.read_parquet(BASE/'original_train_features.parquet')
    original=pd.read_parquet(DATA/'fit_features.parquet')
    np.testing.assert_array_equal(native.snr2500,original.loc[original.split.eq('train'),'snr2500'])
    assert digest(BASE/'original_train_features.parquet')==r['training_feature_sha256']
    correction=pd.read_parquet(BASE/'correction_train_features.parquet')
    assert set(correction.date_key)==set(CORRECTION_DATES)
    assert not set(correction.date_key)&set(original.date_key)
    assert digest(BASE/'correction_train_features.parquet')==r['correction_features_sha256']
    assert not set(TEST_DATES+active_harc_dates())&(set(original.date_key)|set(correction.date_key))
    for meta in (BASE/'raw').glob('*.meta.json'):
        raw=meta.with_name(meta.name.replace('.meta.json','.jsonl'));info=json.loads(meta.read_text())
        assert digest(raw)==info['sha256'] and info['rows']<50001
        assert 'cityHash64(id)%128=0' in info['query'] and 'band=7' in info['query']
    for date in active_harc_dates():
        audit=json.loads((OUT/'harc_audit'/f'harc-{date}-audit.json').read_text())
        assert digest(BASE/f'harc-{date}.parquet')==audit['sample_parquet_sha256']
        assert audit['timestamp_policy'].startswith('explicit UTC')
        sample=pd.read_parquet(BASE/f'harc-{date}.parquet')
        cw=sample[sample.source.eq('HARC_RBN')&sample['mode'].eq('CW')]
        if len(cw):
            # A few HARC SNR copies differ from the original daily RBN CSV.
            # Prepared CW deliberately uses the verified original values, not HARC sn.
            import zipfile
            from collect import RAW
            with zipfile.ZipFile(RAW/f'rbn-{date.replace("-","")}.zip') as archive:
                raw=pd.read_csv(archive.open(archive.namelist()[0]))
            raw=raw[raw.band.eq('40m')].copy()
            raw['time']=pd.to_datetime(raw['date'],utc=True)
            raw['frequency']=np.rint(raw.freq.to_numpy()*1000).astype('int64')
            raw=raw.rename(columns={'dx':'tx_sign','callsign':'rx_sign'})
            keys=['time','tx_sign','rx_sign','frequency']
            lookup=raw.groupby(keys).db.median()
            native_values=lookup.reindex(pd.MultiIndex.from_frame(cw[keys])).to_numpy()
            np.testing.assert_array_equal(native_values,cw.native_snr.to_numpy())
            np.testing.assert_allclose(cw.snr2500.to_numpy(),native_values-10*np.log10(2500/500),atol=1e-12)
    # Independently verify that validation selected the absolute-SNR winner.
    validation=pd.read_parquet(BASE/'validation_features.parquet')
    ref=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
    baseline=joblib.load(DATA/'models/standard-boosted_baseline.joblib')
    candidate_checks=0
    with threadpool_limits(limits=4):
        for mode,scores in r['validation'].items():
            d=validation[validation['mode'].eq(mode)]
            computed={}
            for name,score in scores.items():
                model=ref['model'] if name=='reference' else baseline['model'] if name=='existing_boosted_baseline' else joblib.load(BASE/f'{name}.joblib')
                p=model.predict(d,station_correction=False);check=independent_metric(d,p)
                for key,value in check.items():assert np.isclose(value,score[key],atol=1e-10),(mode,name,key)
                computed[name]=check;candidate_checks+=1
            reference=computed['reference']
            valid=[name for name,sc in computed.items() if sc['mae_db']<=reference['mae_db']*1.02 and sc['rmse_db']<=reference['rmse_db']*1.03]
            selected=min(valid,key=lambda name:computed[name]['hour_date_mae_db'])
            assert selected==r['selection'][mode]
    cal=pd.read_parquet(BASE/'calibration_features.parquet')
    assert not set(cal.date_key)&(set(native.date_key)|set(correction.date_key)|set(validation.date_key)|set(TEST_DATES))
    experimental=joblib.load(ROOT/'artifacts/gocap-40m-hourly-experimental.joblib')
    with threadpool_limits(limits=4):pc=predict_profiles(cal,experimental['models'],experimental['profile_selection'])
    residual=abs(pc-cal.snr2500.to_numpy());keys=(cal.source+'|'+cal['mode']).to_numpy();calibration_checks=0
    for key,info in experimental['intervals'].half_width.items():
        e=sorted(residual[keys==key]);k=min(len(e),math.ceil((len(e)+1)*.8))
        assert np.isclose(e[k-1],info['db']) and len(e)==info['rows'];calibration_checks+=1
    fresh=pd.read_parquet(BASE/'fresh_features.parquet');saved=pd.read_parquet(OUT/'fresh_predictions.parquet')
    assert digest(BASE/'fresh_features.parquet')==r['fresh_feature_sha256']
    np.testing.assert_array_equal(fresh.snr2500,saved.snr2500)
    assert fresh.source.isin(['WSPRNET','HARC_PSK','HARC_RBN']).all()
    assert fresh.snr2500.notna().all()
    recommended=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib')
    with threadpool_limits(limits=4):
        p0=ref['model'].predict(fresh,station_correction=False)
        p=predict_profiles(fresh,experimental['models'],experimental['profile_selection'])
        low,high=experimental['intervals'].predict(fresh,experimental['models'])
        deployed=predict_profiles(fresh,recommended['models'],recommended['profile_selection'])
    np.testing.assert_array_equal(p0,saved.reference_prediction)
    np.testing.assert_array_equal(p,saved.candidate_prediction)
    np.testing.assert_array_equal(low,saved.candidate_lower);np.testing.assert_array_equal(high,saved.candidate_upper)
    checked_groups=0
    for mode,score in r['fresh_results'].items():
        if 'candidate' not in score:continue
        use=fresh['mode'].eq(mode).to_numpy();sub=fresh.loc[use]
        for label,pred,lo,hi in [('reference',p0[use],saved.reference_lower.to_numpy()[use],saved.reference_upper.to_numpy()[use]),
                                  ('candidate',p[use],low[use],high[use])]:
            measured=independent_metric(sub,pred)
            for key,value in measured.items():assert np.isclose(value,score[label][key],atol=1e-10),(mode,label,key)
            y=sub.snr2500.to_numpy()
            coverage=sum(a<=b<=c for a,b,c in zip(lo,y,hi))/len(y)
            assert np.isclose(coverage,score[label]['coverage'])
            interval=[c-a+10*max(a-b,0)+10*max(b-c,0) for a,b,c in zip(lo,y,hi)]
            assert np.isclose(sum(interval)/len(interval),score[label]['mean_interval_score'])
            checked_groups+=1
        expected=p[use] if r['promoted_by_mode'][mode] else p0[use]
        np.testing.assert_array_equal(deployed[use],expected)
    # Every printed UTC-hour score reconciles to the same source observations.
    hourly=pd.read_csv(OUT/'hour_by_hour_errors.csv')
    for row in hourly.itertuples():
        use=fresh['mode'].eq(row.mode)&fresh.hour.dt.hour.eq(row.utc_hour)
        assert use.sum()==row.rows
        actual=fresh.loc[use,'snr2500'].to_numpy()
        assert np.isclose(abs(saved.loc[use,'reference_prediction'].to_numpy()-actual).mean(),row.reference_mae_db)
        assert np.isclose(abs(saved.loc[use,'candidate_prediction'].to_numpy()-actual).mean(),row.candidate_mae_db)
    stress_checks=0
    from .plan import station_held,geographic_held
    for name,h in strict['artifacts'].items():
        path=BASE/f'stress-{name}.joblib';assert digest(path)==h
        m=joblib.load(path);column=m['excluded_flag']
        if name=='station_holdout':assert not any(station_held(c) for c in m['training_stations'])
        else:assert not any(geographic_held(c+'00') for c in m['training_fields'])
        sub=fresh[fresh['mode'].eq('WSPR')&fresh[column]]
        with threadpool_limits(limits=4):
            for label,model in [('reference',m['reference']),('candidate',m['model'])]:
                check=independent_metric(sub,model.predict(sub,station_correction=False))
                for key,value in check.items():assert np.isclose(value,r['strict_snr_tests'][name][label][key],atol=1e-10)
                stress_checks+=1
    from .hourly_evaluate import bootstrap_cells
    for mode,s in r['fresh_results'].items():
        if 'candidate' not in s:continue
        use=fresh['mode'].eq(mode).to_numpy();b=bootstrap_cells(fresh.loc[use],p0[use],p[use])
        np.testing.assert_allclose(b['95_percent_date_bootstrap_interval'],s['difference']['95_percent_date_bootstrap_interval'])
    result={'status':'verified','objective':'absolute hourly SNR, no ranking selection',
        'validation_candidates_recomputed':candidate_checks,'calibration_quantiles_recomputed':calibration_checks,
        'fresh_rows':len(fresh),'source_mode_metric_groups':checked_groups,'separate_utc_hour_scores':len(hourly),
        'strict_metric_groups_recomputed':stress_checks,
        'checks':['frozen code/model/source checksums','correction dates excluded from reference fitting',
            'all candidate choices independently recomputed from absolute hourly errors',
            'new-date SNR predictions, MAE/RMSE/bias, interval coverage and interval score',
            'per-UTC-hour scores reconciled to observed data','strict station/field exclusion and SNR metrics',
            'calibration order statistics','date-cluster bootstrap reproducibility',
            'recommended per-mode predictions match accepted candidate or retained reference'],
        'output_sha256':{name:digest(OUT/name) for name in ['plan.json','run.json','availability_amendment.json','deployment_guard.json',
            'fresh_predictions.parquet','hour_by_hour_errors.csv']},
        'recommended_artifact_sha256':digest(ROOT/'artifacts/gocap-40m-hourly.joblib')}
    save('verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
