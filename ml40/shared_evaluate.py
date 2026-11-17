"""Score frozen model once on new500 WSPR cases and a known multi-mode audit."""
import json,datetime as dt
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from experiment import features
from .prepare import annotate
from .v3_features import enrich
from .shared_plan import ROOT,DATA,BASE,OUT,ARTIFACT,plan,sha
from .shared_train import evaluate as profile_scores
from .hourly_models import predict_profiles
from .voacap_run import scores
from .inference import THRESHOLDS

def bootstrap(c,col,seed=40):
    g=c.groupby('date_key')[col].agg(['sum','count']).sort_index();rng=np.random.default_rng(seed);ix=rng.integers(0,len(g),(10000,len(g)))
    b=g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
    return np.quantile(b,[.025,.975]).tolist()

def run():
    cfg=plan();frozen=json.loads((OUT/'case_freeze.json').read_text());train=json.loads((OUT/'training_run.json').read_text())
    assert train['phase']=='frozen_before_new_test' and sha(ARTIFACT)==train['model_sha256']
    assert sha(ROOT/'artifacts/gocap-40m-hourly.joblib')==cfg['baseline_sha256']
    cases=pd.read_csv(OUT/'cases_frozen.csv');obs=pd.read_parquet(BASE/'case_features.parquet')
    baseline=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib');candidate=joblib.load(ARTIFACT)
    selected=train['selected_candidate'];ablation_name=selected.replace('_points_','_aggregate_');ablation=joblib.load(BASE/(ablation_name+'.joblib'))
    rows=[]
    for c in cases.itertuples():
        for h in range(24):rows.append({**{k:getattr(c,k) for k in ['case_id','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']},
            'hour':pd.Timestamp(c.date_key,tz='UTC')+pd.Timedelta(hours=h),'source':'WSPRNET','mode':'WSPR','decoder_context':'ordinary_code1'})
    full=enrich(annotate(features(pd.DataFrame(rows),solar_start='2009')))
    with threadpool_limits(limits=4):
        full['baseline_snr_db']=predict_profiles(full,baseline['models'],baseline['profile_selection'])
        full['shared_snr_db']=predict_profiles(full,candidate['models'],candidate['profile_selection'])
        full['pooled_aggregate_snr_db']=ablation.predict(full)
        lo,hi=candidate['intervals'].predict(full,candidate['models']);full['shared_lower80_db']=lo;full['shared_upper80_db']=hi
    full=full.merge(obs[['case_id','hour','snr2500','reports']],on=['case_id','hour'],how='left',validate='one_to_one').rename(columns={'snr2500':'actual_snr_db','reports':'positive_reports'})
    native=pd.read_parquet(BASE/'voacap_all_scenarios.parquet');primary=native[native.scenario.eq(cfg['voacap_primary'])]
    full=full.merge(primary,on=['case_id','utc_hour'],validate='one_to_one')
    for name,threshold in THRESHOLDS.items():full[f'shared_{name}_theoretical_margin_db']=full.shared_snr_db-threshold
    keep=['case_id','hour','utc_hour','tx_grid','rx_grid','power_dbm','actual_snr_db','positive_reports','baseline_snr_db','shared_snr_db','pooled_aggregate_snr_db',
        'shared_lower80_db','shared_upper80_db','voacap_snr_db','voacap_left_db','voacap_right_db']+[c for c in full if c.endswith('_theoretical_margin_db')]
    full[keep].to_csv(OUT/'hourly_comparison.csv',index=False)
    actual=full[full.actual_snr_db.notna()].copy();paired=actual[actual.voacap_snr_db.notna()].copy();per=cases.copy()
    for col in ['baseline_snr_db','shared_snr_db','pooled_aggregate_snr_db','voacap_snr_db']:
        z=paired.assign(error=abs(paired[col]-paired.actual_snr_db)).groupby('case_id').error.mean()
        per[col.replace('snr','mae')]=per.case_id.map(z)
    per['shared_minus_baseline_mae_db']=per.shared_mae_db-per.baseline_mae_db
    per['shared_minus_voacap_mae_db']=per.shared_mae_db-per.voacap_mae_db
    per['shared_minus_aggregate_mae_db']=per.shared_mae_db-per.pooled_aggregate_mae_db
    per.to_csv(OUT/'case_results.csv',index=False)
    sensitivity=[]
    for scenario,g in native.groupby('scenario'):
        z=actual.drop(columns=[x for x in actual if x.startswith('voacap_')]).merge(g,on=['case_id','utc_hour'],validate='one_to_one')
        sensitivity.append({'scenario':scenario,**scores(z,'voacap_snr_db')})
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)
    # Existing released2025 archive is a known-benchmark audit, not new blind test data.
    audit=pd.read_parquet(DATA/'hourly/fresh_features.parquet');audit=audit[~audit['mode'].eq('WSPR')].reset_index(drop=True)
    with threadpool_limits(limits=4):
        bp=predict_profiles(audit,baseline['models'],baseline['profile_selection']);sp=predict_profiles(audit,candidate['models'],candidate['profile_selection'])
        al,ah=candidate['intervals'].predict(audit,candidate['models'])
    audit['baseline_prediction']=bp;audit['shared_prediction']=sp;audit['shared_lower80']=al;audit['shared_upper80']=ah
    audit.to_parquet(BASE/'known_mode_audit_predictions.parquet',index=False)
    audit_results=[]
    for model,p in [('baseline',bp),('shared',sp)]:
        for mode,stat in profile_scores(audit,p).items():audit_results.append({'model':model,'mode':mode,**stat})
    pd.DataFrame(audit_results).to_csv(OUT/'known_mode_audit.csv',index=False)
    summary={'phase':'evaluated','evaluated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'case_count':500,'observed_hours':len(actual),'full_prediction_hours':len(full),
        'positive_reports':int(actual.positive_reports.sum()),'primary_voacap_scenario':cfg['voacap_primary'],
        'baseline':scores(paired,'baseline_snr_db'),'shared':scores(paired,'shared_snr_db'),'pooled_aggregate_ablation':scores(paired,'pooled_aggregate_snr_db'),'voacap':scores(paired,'voacap_snr_db'),
        'shared_all_observed':scores(actual,'shared_snr_db'),'voacap_left':scores(actual,'voacap_left_db'),'voacap_right':scores(actual,'voacap_right_db'),
        'shared_minus_baseline_case_mae_db':float(per.shared_minus_baseline_mae_db.mean()),'shared_minus_baseline_95pct_date_bootstrap_db':bootstrap(per,'shared_minus_baseline_mae_db'),
        'shared_minus_voacap_case_mae_db':float(per.shared_minus_voacap_mae_db.mean()),'shared_minus_voacap_95pct_date_bootstrap_db':bootstrap(per,'shared_minus_voacap_mae_db'),
        'shared_minus_aggregate_case_mae_db':float(per.shared_minus_aggregate_mae_db.mean()),'shared_minus_aggregate_95pct_date_bootstrap_db':bootstrap(per,'shared_minus_aggregate_mae_db'),
        'shared_wins_baseline':int(per.shared_minus_baseline_mae_db.lt(0).sum()),'shared_wins_voacap':int(per.shared_minus_voacap_mae_db.lt(0).sum()),
        'shared_interval80_coverage':float(((actual.actual_snr_db>=actual.shared_lower80_db)&(actual.actual_snr_db<=actual.shared_upper80_db)).mean()),
        'shared_interval80_mean_width_db':float((actual.shared_upper80_db-actual.shared_lower80_db).mean()),
        'candidate_model_sha256':sha(ARTIFACT),'baseline_model_sha256':cfg['baseline_sha256'],'no_refitting_after_new_test':True,
        'mode_audit_scope':'Previously published2025 FT8/FT4/CW held-out archive; known-benchmark audit only, no VOACAP comparison with invented unknown power.',
        'mode_audit_input_sha256':sha(DATA/'hourly/fresh_features.parquet'),'case_freeze_sha256':sha(OUT/'case_freeze.json')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
