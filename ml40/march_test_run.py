"""Frozen additional March2025 comparison:500paths, three GoCAP models,12VOACAP settings."""
import json,datetime as dt
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from experiment import features
from .prepare import annotate
from .v3_features import enrich
from .march_test_plan import ROOT,BASE,OUT,ARTIFACTS,plan,sha
from .voacap_engine import Engine,make_deck,parse_output,at_bin_centre
from .inference import centre
from .hourly_models import predict_profiles
from .voacap_run import scores

def run():
    cfg=plan();freeze=json.loads((OUT/'case_freeze.json').read_text());cases=pd.read_csv(OUT/'cases_frozen.csv');obs=pd.read_parquet(BASE/'case_features.parquet')
    for name,path in ARTIFACTS.items():assert sha(path)==cfg['models_sha256'][name]
    frames=[]
    for c in cases.itertuples():
        for h in range(24):frames.append({**{k:getattr(c,k) for k in ['case_id','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm','decoder_context']},
            'hour':pd.Timestamp(c.date_key,tz='UTC')+pd.Timedelta(hours=h),'source':'WSPRNET','mode':'WSPR'})
    full=enrich(annotate(features(pd.DataFrame(frames),solar_start='2009')))
    with threadpool_limits(limits=4):
        for name,path in ARTIFACTS.items():
            bundle=joblib.load(path);full[name+'_snr_db']=predict_profiles(full,bundle['models'],bundle['profile_selection'])
    full=full.merge(obs[['case_id','hour','snr2500','reports']],on=['case_id','hour'],how='left',validate='one_to_one').rename(columns={'snr2500':'actual_snr_db','reports':'positive_reports'})
    native=[];manifest=[]
    with Engine() as engine:
        for i,c in enumerate(cases.itertuples(),1):
            tlat,tlon=centre(c.tx_grid);rlat,rlon=centre(c.rx_grid)
            for sc in cfg['voacap_scenarios']:
                folder=BASE/'runs'/c.case_id/sc['id'];folder.mkdir(parents=True,exist_ok=True);ip=folder/'input.dat';op=folder/'output.out'
                deck=make_deck(tlat,tlon,rlat,rlon,c.year,3,c.ssn*sc['ssn_scale'],c.reported_power_watts,sc['antenna'],sc['noise_parameter'])
                if ip.exists():assert ip.read_text()==deck
                ip.write_text(deck)
                if op.exists():parsed=parse_output(op.read_text())
                else:text,parsed=engine.run(deck);op.write_text(text)
                manifest.append({'case_id':c.case_id,'scenario':sc['id'],'input_sha256':sha(ip),'output_sha256':sha(op)})
                for h in range(24):
                    mid,lo,hi=at_bin_centre(parsed,h);native.append({'case_id':c.case_id,'utc_hour':h,'scenario':sc['id'],'voacap_snr_db':mid,
                        'voacap_left_db':parsed[h]['snr2500_db'],'voacap_right_db':parsed[(h+1)%24]['snr2500_db']})
            if i%50==0:print('Compared',i,'/500 cases',flush=True)
    pd.DataFrame(manifest).to_csv(OUT/'run_manifest.csv',index=False);native=pd.DataFrame(native);native.to_parquet(BASE/'native_predictions.parquet',index=False)
    full=full.merge(native[native.scenario.eq(cfg['voacap_primary'])],on=['case_id','utc_hour'],validate='one_to_one')
    columns=['case_id','hour','utc_hour','date_key','distance_km','tx_grid','rx_grid','power_dbm','decoder_context','actual_snr_db','positive_reports','march_snr_db','shared_parent_snr_db','current_snr_db','voacap_snr_db','voacap_left_db','voacap_right_db']
    full[columns].to_csv(OUT/'hourly_comparison.csv',index=False);actual=full[full.actual_snr_db.notna()].copy();paired=actual[actual.voacap_snr_db.notna()].copy()
    per=cases.copy();models=['march','shared_parent','current','voacap']
    for name in models:
        e=abs(paired[name+'_snr_db']-paired.actual_snr_db).groupby(paired.case_id).mean();per[name+'_mae_db']=per.case_id.map(e)
    per['march_minus_voacap_mae_db']=per.march_mae_db-per.voacap_mae_db;per['march_minus_current_mae_db']=per.march_mae_db-per.current_mae_db
    per.to_csv(OUT/'case_results.csv',index=False)
    dates=per.groupby('date_key').agg(cases=('case_id','size'),march_mae_db=('march_mae_db','mean'),shared_parent_mae_db=('shared_parent_mae_db','mean'),current_mae_db=('current_mae_db','mean'),voacap_mae_db=('voacap_mae_db','mean'),
        march_wins_voacap=('march_minus_voacap_mae_db',lambda x:(x<0).sum()),median_distance_km=('distance_km','median'),smoothed_ssn=('ssn','first'))
    dates.to_csv(OUT/'per_date.csv')
    distance=per.groupby('distance_group').agg(cases=('case_id','size'),march_mae_db=('march_mae_db','mean'),current_mae_db=('current_mae_db','mean'),shared_parent_mae_db=('shared_parent_mae_db','mean'),voacap_mae_db=('voacap_mae_db','mean'))
    distance.to_csv(OUT/'per_distance.csv')
    ci={}
    for other in ['voacap','current','shared_parent']:
        per['delta']=per.march_mae_db-per[other+'_mae_db'];g=per.groupby('date_key').delta.agg(['sum','count']);ix=np.random.default_rng(40).integers(0,len(g),(10000,len(g)))
        estimates=g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
        ci[other]={'difference_db':float(per.delta.mean()),'date_cluster95_db':np.quantile(estimates,[.025,.975]).tolist()}
    sensitivity=[]
    for name,g in native.groupby('scenario'):
        z=actual.drop(columns=[c for c in actual if c.startswith('voacap_')]).merge(g,on=['case_id','utc_hour'],validate='one_to_one');sensitivity.append({'scenario':name,**scores(z,'voacap_snr_db')})
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)
    summary={'evaluated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'dates':cfg['dates'],'cases':500,'observed_hours':len(actual),'positive_reports':int(actual.positive_reports.sum()),'full_prediction_hours':len(full),
        'primary_voacap':cfg['voacap_primary'],'models':{name:scores(paired,name+'_snr_db') for name in models},'march_all_observed':scores(actual,'march_snr_db'),
        'paired_differences':ci,'march_wins_voacap':int(per.march_minus_voacap_mae_db.lt(0).sum()),'march_wins_current':int(per.march_minus_current_mae_db.lt(0).sum()),
        'voacap_left':scores(actual,'voacap_left_db'),'voacap_right':scores(actual,'voacap_right_db'),
        'no_test_based_fitting':True,'held_out_dates_disjoint_from_training':True,'frozen_case_sha256':sha(OUT/'case_freeze.json')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
