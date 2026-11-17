"""Run frozen 100-case comparison; never fit either predictor."""
import json,datetime as dt
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from experiment import features
from .prepare import annotate
from .v3_features import enrich
from .v3_scenario_check import steady
from .hourly_models import predict_profiles
from .inference import centre
from .voacap_protocol import BASE,OUT,ROOT,protocol,sha
from .voacap_engine import Engine,make_deck,parse_output,at_bin_centre

def scores(d,p):
    z=d[np.isfinite(d[p])].copy();z['error']=z[p]-z.actual_snr_db
    case=z.groupby('case_id').error.apply(lambda x:abs(x).mean())
    return {'cases':len(case),'observed_hours':len(d),'numeric_hours':len(z),'case_mae_db':float(case.mean()),
        'hour_mae_db':float(abs(z.error).mean()),'hour_rmse_db':float(np.sqrt((z.error**2).mean())), 'bias_db':float(z.error.mean())}

def run():
    cfg=protocol();frozen=json.loads((OUT/'case_freeze.json').read_text())
    assert sha(ROOT/'artifacts/gocap-40m-hourly.joblib')==frozen['model_sha256']
    cases=pd.read_csv(OUT/'cases_frozen.csv');obs=pd.read_parquet(BASE/'case_features.parquet')
    # Build all 24 hours independently of whether a reception was reported.
    rows=[]
    for c in cases.itertuples():
        for h in range(24):
            rows.append({**{k:getattr(c,k) for k in ['case_id','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']},
                'hour':pd.Timestamp(c.date_key,tz='UTC')+pd.Timedelta(hours=h),'source':'WSPRNET','mode':'WSPR','decoder_context':'ordinary_code1'})
    full=enrich(annotate(features(pd.DataFrame(rows),solar_start='2009')))
    bundle=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib')
    with threadpool_limits(limits=4):
        full['gocap_snr_db']=predict_profiles(full,bundle['models'],bundle['profile_selection'])
        full['gocap_steady_snr_db']=predict_profiles(steady(full),bundle['models'],bundle['profile_selection'])
    full=full.merge(obs[['case_id','hour','snr2500','reports']],on=['case_id','hour'],how='left',validate='one_to_one').rename(columns={'snr2500':'actual_snr_db','reports':'positive_reports'})
    frames=[];runroot=BASE/'runs';runroot.mkdir(exist_ok=True)
    with Engine() as engine:
        for number,c in enumerate(cases.itertuples(),1):
            lat,lon=centre(c.tx_grid);rlat,rlon=centre(c.rx_grid)
            for scenario in cfg['scenarios']:
                folder=runroot/c.case_id/scenario['id'];folder.mkdir(parents=True,exist_ok=True)
                deck=make_deck(lat,lon,rlat,rlon,c.year,c.month,c.ssn*scenario['ssn_scale'],c.reported_power_watts,scenario['antenna'],scenario['noise_parameter'])
                ip=folder/'input.dat';op=folder/'output.out'
                if ip.exists():assert ip.read_text()==deck
                ip.write_text(deck)
                if op.exists():parsed=parse_output(op.read_text())
                else:
                    text,parsed=engine.run(deck);op.write_text(text)
                for h in range(24):
                    mid,lo,hi=at_bin_centre(parsed,h)
                    frames.append({'case_id':c.case_id,'utc_hour':h,'scenario':scenario['id'],'voacap_snr_db':mid,
                        'voacap_lower_decile_db':lo,'voacap_upper_decile_db':hi,
                        'voacap_left_db':parsed[h]['snr2500_db'],'voacap_right_db':parsed[(h+1)%24]['snr2500_db']})
            if number%10==0:print(f'Completed {number}/{len(cases)} cases',flush=True)
    native=pd.DataFrame(frames);native.to_csv(OUT/'voacap_all_scenarios.csv',index=False)
    primary=native[native.scenario.eq(cfg['primary_scenario'])]
    result=full.merge(primary,on=['case_id','utc_hour'],validate='one_to_one')
    cols=['case_id','hour','utc_hour','tx_grid','rx_grid','power_dbm','actual_snr_db','positive_reports','gocap_snr_db','gocap_steady_snr_db','voacap_snr_db','voacap_left_db','voacap_right_db','voacap_lower_decile_db','voacap_upper_decile_db']
    result[cols].to_csv(OUT/'hourly_comparison.csv',index=False)
    actual=result[result.actual_snr_db.notna()].copy();paired=actual[actual.voacap_snr_db.notna()].copy()
    per=cases.copy()
    for p in ['gocap_snr_db','voacap_snr_db']:
        z=paired.assign(error=abs(paired[p]-paired.actual_snr_db)).groupby('case_id').error.mean()
        per[p.replace('snr','mae')]=per.case_id.map(z)
    per['gocap_minus_voacap_mae_db']=per.gocap_mae_db-per.voacap_mae_db
    per.to_csv(OUT/'case_results.csv',index=False)
    rng=np.random.default_rng(20261117)
    day=per.groupby('date_key').gocap_minus_voacap_mae_db.agg(['sum','count']).sort_index()
    sampled=rng.choice(len(day),(10000,len(day)),replace=True)
    boots=day['sum'].to_numpy()[sampled].sum(axis=1)/day['count'].to_numpy()[sampled].sum(axis=1)
    sensitivity=[]
    for name,g in native.groupby('scenario'):
        z=actual.drop(columns=[x for x in actual if x.startswith('voacap_')]).merge(g,on=['case_id','utc_hour'],validate='one_to_one')
        sensitivity.append({'scenario':name,**scores(z,'voacap_snr_db')})
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)
    summary={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cases':len(cases),'full_prediction_hours':len(result),'observed_hours':len(actual),
        'primary_scenario':cfg['primary_scenario'],'gocap':scores(paired,'gocap_snr_db'),'voacap':scores(paired,'voacap_snr_db'),
        'gocap_all_observed':scores(actual,'gocap_snr_db'),'gocap_steady':scores(paired,'gocap_steady_snr_db'),
        'voacap_left':scores(actual,'voacap_left_db'),'voacap_right':scores(actual,'voacap_right_db'),
        'gocap_minus_voacap_case_mae_db':float(per.gocap_minus_voacap_mae_db.mean()),
        'date_cluster_bootstrap_95pct_db':np.quantile(boots,[.025,.975]).tolist(),
        'case_wins_strict':int((per.gocap_minus_voacap_mae_db<0).sum()),'case_losses_strict':int((per.gocap_minus_voacap_mae_db>0).sum()),
        'case_ties_exact':int((per.gocap_minus_voacap_mae_db==0).sum()),'no_refitting':True}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
