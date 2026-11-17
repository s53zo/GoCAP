"""Evaluate frozen March augmentation on fresh2025March and known comparison sets."""
import json,datetime as dt
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .march_plan import ROOT,DATA,BASE,OUT,ARTIFACT,FRESH_TEST,plan,sha
from .hourly_models import predict_profiles,metric
from .shared_train import evaluate as per_mode

def hour_bootstrap(d,old,new):
    cells=pd.DataFrame({'date':d.date_key.astype(str).to_numpy(),'hour':d.hour.dt.hour.to_numpy(),
        'difference':abs(new-d.snr2500.to_numpy())-abs(old-d.snr2500.to_numpy())}).groupby(['date','hour']).difference.mean().reset_index()
    g=cells.groupby('date').difference.agg(['sum','size']);idx=np.random.default_rng(40).integers(0,len(g),(10000,len(g)))
    estimates=g['sum'].to_numpy()[idx].sum(1)/g['size'].to_numpy()[idx].sum(1)
    return {'difference_db':float(cells.difference.mean()),'date_bootstrap95_db':np.quantile(estimates,[.025,.975]).tolist(),'date_clusters':len(g)}

def run():
    cfg=plan();frozen=json.loads((OUT/'training_run.json').read_text());assert frozen['phase']=='frozen_before_fresh_test' and sha(ARTIFACT)==frozen['model_sha256']
    bundles={'march':joblib.load(ARTIFACT),'shared_parent':joblib.load(ROOT/'artifacts/gocap-40m-shared-points.joblib'),'current':joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib')}
    test=pd.concat([pd.read_parquet(BASE/'processed'/f'wspr-{date}.parquet') for date in FRESH_TEST],ignore_index=True)
    assert test.hour.dt.year.eq(2025).all() and set(test.date_key)==set(FRESH_TEST)
    results={};pred={}
    with threadpool_limits(limits=4):
        for name,bundle in bundles.items():
            p=predict_profiles(test,bundle['models'],bundle['profile_selection']);pred[name]=p;results[name]=metric(test,p)
            test[name+'_prediction']=p
        lo,hi=bundles['march']['intervals'].predict(test,bundles['march']['models'])
    test['march_lower80']=lo;test['march_upper80']=hi
    test.to_parquet(BASE/'fresh_march_predictions.parquet',index=False)
    cells=[]
    for (date,hour),g in test.groupby(['date_key','hour']):
        row={'date':str(date),'utc_hour':hour.hour,'observed_rows':len(g)}
        for name in bundles:row[name+'_mae_db']=float(abs(g[name+'_prediction']-g.snr2500).mean())
        cells.append(row)
    pd.DataFrame(cells).to_csv(OUT/'fresh_march_hourly_errors.csv',index=False)
    obs=pd.read_parquet(DATA/'shared40/case_features.parquet');old=pd.read_csv(ROOT/'results/ml40/shared40/hourly_comparison.csv')
    with threadpool_limits(limits=4):p=predict_profiles(obs,bundles['march']['models'],bundles['march']['profile_selection'])
    obs=obs.copy();obs['utc_hour']=obs.hour.dt.hour;obs['march_snr_db']=p
    joined=obs[['case_id','utc_hour','march_snr_db']].merge(old,on=['case_id','utc_hour'],validate='one_to_one')
    assert len(joined)==4820 and joined.actual_snr_db.notna().all()
    per=joined.groupby('case_id').apply(lambda g:pd.Series({name:float(abs(g[col]-g.actual_snr_db).mean()) for name,col in [('march','march_snr_db'),('shared_parent','shared_snr_db'),('current','baseline_snr_db'),('voacap','voacap_snr_db')]}),include_groups=False).reset_index()
    per.to_csv(OUT/'known500_case_errors.csv',index=False);joined.to_csv(OUT/'known500_hourly_comparison.csv',index=False)
    known500={name:float(per[name].mean()) for name in ['march','shared_parent','current','voacap']}
    audit=pd.read_parquet(DATA/'shared40/known_mode_audit_predictions.parquet')
    with threadpool_limits(limits=4):ap=predict_profiles(audit,bundles['march']['models'],bundles['march']['profile_selection'])
    audit['march_prediction']=ap;audit.to_parquet(BASE/'known_mode_predictions.parquet',index=False)
    mode=[]
    for name,col in [('current','baseline_prediction'),('shared_parent','shared_prediction'),('march','march_prediction')]:
        for m,score in per_mode(audit,audit[col].to_numpy()).items():mode.append({'model':name,'mode':m,**score})
    pd.DataFrame(mode).to_csv(OUT/'known_mode_audit.csv',index=False)
    summary={'evaluated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'fresh_march_rows':len(test),'fresh_march_dates':FRESH_TEST,'fresh_march_results':results,
        'march_minus_shared_parent':hour_bootstrap(test,pred['shared_parent'],pred['march']),'march_minus_current':hour_bootstrap(test,pred['current'],pred['march']),
        'march_interval80_coverage':float(((test.snr2500>=lo)&(test.snr2500<=hi)).mean()),'march_interval80_mean_width_db':float((hi-lo).mean()),
        'known500_case_mae_db':known500,'known500_scope':'Previously released benchmark; not a new blind test. VOACAP settings and observations unchanged.',
        'new_model_sha256':sha(ARTIFACT),'parent_sha256':cfg['parent_sha256'],'baseline_sha256':cfg['baseline_sha256'],
        'fresh_source_sha256':{date:sha(BASE/'processed'/f'wspr-{date}.parquet') for date in FRESH_TEST},'no2025_training':True,'no_test_based_retuning':True}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
