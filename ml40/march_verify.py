"""Independent March temporal-coverage, raw-truth, leakage and metric checks."""
import json
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .march_plan import ROOT,DATA,BASE,OUT,DATES,FRESH_TEST,ARTIFACT,KEYS,sha

def raw_hourly(date):
    raw=pd.read_json(BASE/'raw'/f'wspr-{date}.jsonl',lines=True)
    if raw.empty:return pd.DataFrame()
    raw['hour']=pd.to_datetime(raw.time,utc=True).dt.floor('h')
    mode_ok=raw.code.eq(1)|(raw.code.eq(0)&raw.hour.lt(pd.Timestamp('2020-04-01',tz='UTC')))
    raw=raw[mode_ok].copy()
    pattern=r'[A-R]{2}[0-9]{2}(?:[A-X]{2}(?:[0-9]{2})?)?'
    raw['tx_grid']=raw.tx_loc.str.upper().where(raw.tx_loc.str.upper().str.fullmatch(pattern,na=False)).str[:4]
    raw['rx_grid']=raw.rx_loc.str.upper().where(raw.rx_loc.str.upper().str.fullmatch(pattern,na=False)).str[:4]
    valid=raw.tx_grid.notna()&raw.rx_grid.notna()&raw.frequency.between(7000000,7300000)&raw.snr.between(-60,100)&raw.tx_sign.notna()&raw.rx_sign.notna()&raw.tx_sign.astype(str).str.len().ge(2)&raw.rx_sign.astype(str).str.len().ge(2)
    raw=raw[valid].drop_duplicates(['time','tx_sign','rx_sign','frequency']).copy()
    raw['source']='WSPRNET';raw['mode']='WSPR';raw['power_dbm']=raw.power.astype(float).where(raw.power.between(0,60))
    return raw.groupby(KEYS,dropna=False,observed=True).snr.agg(snr2500='median',reports='size').reset_index()

def verify():
    cfg=json.loads((OUT/'plan.json').read_text());prep=json.loads((OUT/'preparation.json').read_text());run=json.loads((OUT/'training_run.json').read_text());s=json.loads((OUT/'summary.json').read_text())
    assert sha(OUT/'legacy_mode_amendment.json')==prep['legacy_amendment_sha256']
    assert pd.Timestamp(json.loads((OUT/'legacy_amendment_created.json').read_text())['amended_before_fitting_utc'])<pd.Timestamp(run['frozen_utc'])
    assert cfg['years']==list(range(2010,2025)) and len(cfg['march_dates'])==465
    assert sha(OUT/'plan.json')==prep['plan_sha256']==run['plan_sha256']
    assert sha(ARTIFACT)==run['model_sha256']==s['new_model_sha256']
    assert sha(ROOT/'artifacts/gocap-40m-shared-points.joblib')==cfg['parent_sha256']
    assert sha(ROOT/'artifacts/gocap-40m-hourly.joblib')==cfg['baseline_sha256']
    assert sha(DATA/'fit_features.parquet')==cfg['original_fit_sha256'] and sha(ROOT/'data/raw/gfz.txt')==cfg['solar_input_sha256']
    for name,digest in run['code_sha256'].items():assert sha(ROOT/'ml40'/name)==digest
    for name,digest in cfg['original_input_sha256'].items():assert sha(DATA/'hourly'/name)==digest
    assert pd.Timestamp(run['frozen_utc'])<pd.Timestamp(s['evaluated_utc'])
    assert sha(BASE/'expanded_train_features.parquet')==prep['train_sha256']==run['training_data_sha256']
    training=pd.read_parquet(BASE/'expanded_train_features.parquet');assert training.hour.dt.year.between(2010,2024).all()
    assert set(training.hour.dt.year)==set(range(2010,2025));assert not training.duplicated(KEYS).any()
    reserved=set(pd.read_parquet(DATA/'hourly/validation_features.parquet',columns=['date_key']).date_key)|set(pd.read_parquet(DATA/'hourly/calibration_features.parquet',columns=['date_key']).date_key)
    assert not set(training.date_key.astype(str))&reserved
    original=pd.read_parquet(DATA/'hourly/original_train_features.parquet',columns=KEYS+['snr2500','reports'])
    matched=original.merge(training[KEYS+['snr2500','reports']],on=KEYS,validate='one_to_one',suffixes=('_original','_expanded'))
    assert len(matched)==len(original)==prep['original_training_rows']
    assert np.allclose(matched.snr2500_original,matched.snr2500_expanded) and np.allclose(matched.reports_original,matched.reports_expanded)
    coverage=pd.read_csv(OUT/'march_coverage.csv');assert len(coverage)==465 and coverage.date.nunique()==465
    wspr_metadata=training.loc[training['mode'].eq('WSPR'),KEYS+['snr2500','date_key']]
    training_groups={str(date):g for date,g in wspr_metadata.groupby('date_key',observed=True)}
    verified_rows=0;fresh_rows=0;trainable_march_rows=0
    for i,date in enumerate(DATES+FRESH_TEST,1):
        p=BASE/'processed'/f'wspr-{date}.parquet';raw=BASE/'raw'/f'wspr-{date}.jsonl'
        assert sha(raw)==json.loads(raw.with_suffix('.meta.json').read_text())['sha256']
        if date in DATES:assert sha(p)==prep['source_sha256'][date] and sha(raw)==prep['raw_sha256'][date]
        else:assert sha(p)==s['fresh_source_sha256'][date]
        actual=raw_hourly(date);cached=pd.read_parquet(p)
        assert len(actual)==len(cached)
        if len(actual):
            z=actual.merge(cached[KEYS+['snr2500','reports']],on=KEYS,validate='one_to_one',suffixes=('_raw','_cache'))
            assert len(z)==len(actual) and np.allclose(z.snr2500_raw,z.snr2500_cache) and np.allclose(z.reports_raw,z.reports_cache)
            if date in DATES:
                verified_rows+=len(z)
                if date not in reserved:
                    trainable_march_rows+=len(actual)
                    subset=training_groups[date]
                    compare=actual.merge(subset[KEYS+['snr2500']],on=KEYS,validate='one_to_one',suffixes=('_raw','_train'))
                    assert len(compare)==len(actual) and np.allclose(compare.snr2500_raw,compare.snr2500_train)
            else:fresh_rows+=len(z)
        if i%50==0:print('Independently verified daily raw aggregates',i,'/471',flush=True)
    old_march=original[original['mode'].eq('WSPR')&original.hour.dt.month.eq(3)]
    assert len(old_march)==prep['duplicate_original_rows_removed']
    assert trainable_march_rows-len(old_march)==prep['added_unique_march_rows']
    assert len(training)==len(original)+trainable_march_rows-len(old_march)
    model=joblib.load(ARTIFACT);assert list(model['models'])==['shared']
    fitted=model['models']['shared'];numeric=np.load(BASE/'matrix/numeric.npy',mmap_mode='r');encoded=np.load(BASE/'matrix/encoded.npy',mmap_mode='r')
    assert len(numeric)==len(encoded)==len(training)
    medians=np.array([np.nanmedian(numeric[:,i]) for i in range(numeric.shape[1])]);medians=np.nan_to_num(medians,nan=0)
    assert np.allclose(fitted.encoder.imputer.statistics_,medians,atol=1e-7)
    indices=np.unique(np.r_[0,len(training)-1,np.random.default_rng(40).integers(0,len(training),128)])
    direct=fitted.encoder.transform(fitted.frame(training.iloc[indices]));assert np.allclose(direct,encoded[indices],atol=1e-6)
    pred=pd.read_parquet(BASE/'fresh_march_predictions.parquet');assert len(pred)==fresh_rows==s['fresh_march_rows'] and pred.hour.dt.year.eq(2025).all()
    for name,expected in s['fresh_march_results'].items():
        e=pred[name+'_prediction']-pred.snr2500
        cells=pd.DataFrame({'date':pred.date_key,'hour':pred.hour.dt.hour,'absolute':abs(e)}).groupby(['date','hour']).absolute.mean()
        assert np.isclose(cells.mean(),expected['hour_date_mae_db']) and np.isclose(abs(e).mean(),expected['mae_db'])
        assert np.isclose(np.sqrt((e*e).mean()),expected['rmse_db']) and np.isclose(e.mean(),expected['bias_db'])
    for other,key in [('shared_parent','march_minus_shared_parent'),('current','march_minus_current')]:
        diff=abs(pred.march_prediction-pred.snr2500)-abs(pred[other+'_prediction']-pred.snr2500)
        cells=pd.DataFrame({'date':pred.date_key,'hour':pred.hour.dt.hour,'diff':diff}).groupby(['date','hour'])['diff'].mean().reset_index()
        g=cells.groupby('date')['diff'].agg(['sum','size']);ix=np.random.default_rng(40).integers(0,len(g),(10000,len(g)))
        means=g['sum'].to_numpy()[ix].sum(1)/g['size'].to_numpy()[ix].sum(1)
        assert np.isclose(cells['diff'].mean(),s[key]['difference_db']) and np.allclose(np.quantile(means,[.025,.975]),s[key]['date_bootstrap95_db'])
    with threadpool_limits(limits=4):direct=fitted.predict(pred)
    assert np.allclose(direct,pred.march_prediction)
    assert np.isclose(((pred.snr2500>=pred.march_lower80)&(pred.snr2500<=pred.march_upper80)).mean(),s['march_interval80_coverage'])
    assert np.isclose((pred.march_upper80-pred.march_lower80).mean(),s['march_interval80_mean_width_db'])
    known=pd.read_csv(OUT/'known500_hourly_comparison.csv')
    for name,col in [('march','march_snr_db'),('shared_parent','shared_snr_db'),('current','baseline_snr_db'),('voacap','voacap_snr_db')]:
        error=abs(known[col]-known.actual_snr_db)
        assert np.isclose(error.groupby(known.case_id).mean().mean(),s['known500_case_mae_db'][name])
    audit=pd.read_parquet(BASE/'known_mode_predictions.parquet');table=pd.read_csv(OUT/'known_mode_audit.csv')
    for row in table.itertuples():
        g=audit[audit['mode'].eq(row.mode)]
        col={'current':'baseline_prediction','shared_parent':'shared_prediction','march':'march_prediction'}[row.model]
        error=g[col]-g.snr2500
        cells=pd.DataFrame({'date':g.date_key,'hour':g.hour.dt.hour,'absolute':abs(error)}).groupby(['date','hour']).absolute.mean()
        assert len(g)==row.rows and np.isclose(cells.mean(),row.hour_date_mae_db) and np.isclose(abs(error).mean(),row.mae_db)
    result={'status':'passed','requested_march_days':465,'available_march_days':prep['available_march_days'],'expanded_training_rows':len(training),
        'all15_training_years_present':True,'no2025_fitting_or_reserved_date_leak':True,'original_rows_retained_once':True,
        'raw_march_hourly_rows_independently_rebuilt':verified_rows,'raw_fresh_test_rows_independently_rebuilt':fresh_rows,
        'bounded_encoder_global_medians_match':True,'bounded_encoder_sample_rows_match_original_transform':len(indices),
        'independent_metrics_and_bootstrap_match':True,'parent_and_default_models_unchanged':True}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
