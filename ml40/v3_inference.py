"""Standalone v3 point SNR, conditional interval and relative-hour rank query."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .v3_features import scenario,PATH
from .v3_models import profile_predict
from .inference import THRESHOLDS


def predict(artifact,tx,rx,utc,sfi,kp,ap,mode='WSPR',power_watts=100,history=None,hours=1):
    if not 1<=hours<=24:raise ValueError('Choose 1 to 24 consecutive hours')
    if history is not None and hours!=1:raise ValueError('Timestamped history supports one query hour; omit history for a steady-scenario daily query')
    bundle=joblib.load(artifact)
    sidecar=Path(artifact).with_suffix('.json')
    evaluation=json.loads(sidecar.read_text()) if sidecar.exists() else {}
    start=pd.Timestamp(utc);start=start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    frames=[];assumptions=[]
    for h in range(hours):
        d,a=scenario(tx,rx,start+pd.Timedelta(hours=h),sfi,kp,ap,mode,power_watts,history)
        frames.append(d);assumptions.append(a)
    d=pd.concat(frames,ignore_index=True)
    p=profile_predict(d,bundle['models'],bundle['profile_selection'])
    lo,hi=bundle['intervals'].predict(d,bundle['models'])
    rank=bundle['ranker'].predict(d) if mode=='WSPR' else np.full(len(d),np.nan)
    output=[]
    for i,row in enumerate(d.itertuples()):
        domain=bundle['training_domain'][mode]
        outside=[k for k,(low,high) in domain.items() if pd.notna(d.iloc[i][k]) and not low<=float(d.iloc[i][k])<=high]
        output.append({'utc_hour':row.hour.isoformat(),'snr2500_db':float(p[i]),
            'nominal_80_percent_received_population_interval':[float(lo[i]),float(hi[i])],
            'relative_hour_rank_score':float(rank[i]) if mode=='WSPR' else None,
            'rank_comparison_group':tx.upper()+'>'+rx.upper()+'|'+row.hour.date().isoformat() if mode=='WSPR' else None,
            'theoretical_mode_margins_db':{k:float(p[i]-v) for k,v in THRESHOLDS.items()},
            'outside_training_ranges':outside,'antipodal_path_unsupported':bool(row.antipodal_path),
            'magnetic_epoch_extrapolation':bool(row.magnetic_epoch_outside)})
    return {'origin':tx.upper(),'destination':rx.upper(),'band_m':40,'observation_profile':mode,
        'point_model':bundle['profile_selection'][mode],'power_input_watts':power_watts,
        'power_used_by_profile':mode=='WSPR','solar_history_assumption':assumptions[0],
        'predictions':output,'rank_interpretation':'Higher is preferred within the same path/day; rank scores are not SNR or contact probabilities.',
        'uncertainty_interpretation':bundle.get('interval_method','Condition-dependent 10th/90th quantiles with held-out split calibration; no guarantee outside the observed reception population.'),
        'evaluation_recommendations':evaluation.get('recommend_components_over_reference',{}),
        'calibration_limit':bundle['calibration_limit'],
        'joint_scenario_support':'Marginal training ranges do not establish support for every combination.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact');p.add_argument('tx');p.add_argument('rx');p.add_argument('utc')
    p.add_argument('--sfi',type=float,required=True);p.add_argument('--kp',type=float,required=True);p.add_argument('--ap',type=float,required=True)
    p.add_argument('--mode',choices=['WSPR','FT8','FT4','CW'],default='WSPR');p.add_argument('--power-watts',type=float,default=100)
    p.add_argument('--history-file');p.add_argument('--hours',type=int,default=1);a=p.parse_args()
    history=json.loads(Path(a.history_file).read_text()) if a.history_file else None
    print(json.dumps(predict(a.artifact,a.tx,a.rx,a.utc,a.sfi,a.kp,a.ap,a.mode,a.power_watts,history,a.hours),indent=2,allow_nan=False))
