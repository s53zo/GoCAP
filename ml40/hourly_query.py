"""One SNR estimate per requested UTC hour. No rank or best-hour fields."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .v3_features import scenario
from .inference import THRESHOLDS
from .hourly_models import predict_profiles


def predict(artifact,tx,rx,utc,sfi,kp,ap,mode='WSPR',power_watts=100,hours=24,history=None):
    if not 1<=hours<=24:raise ValueError('Choose 1 to 24 consecutive UTC hours')
    if history is not None and hours!=1:raise ValueError('Timestamped history requires a single-hour query')
    if not np.isfinite([sfi,kp,ap,power_watts]).all():raise ValueError('Solar and power inputs must be finite')
    start=pd.Timestamp(utc);start=start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    rows=[]
    for hour in range(hours):
        d,assumption=scenario(tx,rx,start+pd.Timedelta(hours=hour),sfi,kp,ap,mode,power_watts,history)
        rows.append(d)
    d=pd.concat(rows,ignore_index=True);bundle=joblib.load(artifact)
    point=predict_profiles(d,bundle['models'],bundle['profile_selection']);lo,hi=bundle['intervals'].predict(d,bundle['models'])
    side=Path(artifact).with_suffix('.json');meta=json.loads(side.read_text()) if side.exists() else {}
    source=d.source.iloc[0];domain=bundle['training_domain'].get(source+'|'+mode,{})
    result=[]
    for i,row in enumerate(d.itertuples()):
        outside=[key for key,(low,high) in domain.items() if pd.notna(d.iloc[i][key]) and not low<=float(d.iloc[i][key])<=high]
        result.append({'utc_hour':row.hour.isoformat(),'estimated_snr_db_2500hz':float(point[i]),
            'received_population_interval_80pct_db':[float(lo[i]),float(hi[i])],
            'theoretical_mode_margins_db':{k:float(point[i]-v) for k,v in THRESHOLDS.items()},
            'outside_training_ranges':outside,'magnetic_epoch_extrapolation':bool(row.magnetic_epoch_outside),
            'antipodal_path_unsupported':bool(row.antipodal_path)})
    return {'origin':tx.upper(),'destination':rx.upper(),'band_m':40,'observation_profile':mode,
        'transmit_power_watts':power_watts,'power_used_by_profile':mode=='WSPR',
        'solar_scenario':{'sfi':sfi,'kp':kp,'previous_day_Ap':ap},'solar_history_assumption':assumption,
        'hourly_predictions':result,'selected_snr_model':bundle['profile_selection'][mode],
        'model_evaluation':meta.get('promoted_by_mode',{}),
        'interpretation':'Every requested hour has an SNR estimate. Unreported receptions remain unknown. Intervals describe reported receptions; mode margins are theoretical and are not contact probabilities.',
        'calibration_limit':bundle['calibration_limit'],
        'joint_scenario_support':'Marginal input ranges do not establish support for every combination of path, season and solar conditions.'}


def table(result):
    power=f"power scenario={result['transmit_power_watts']:g} W" if result['power_used_by_profile'] else 'transmit power unknown in this profile; power input unused'
    lines=[f"40m  {result['origin']} -> {result['destination']}  profile={result['observation_profile']}  {power}",
        'UTC hour                   SNR dB       80% interval      FT8 margin   CW DX margin   SSB DX margin']
    for row in result['hourly_predictions']:
        lo,hi=row['received_population_interval_80pct_db'];m=row['theoretical_mode_margins_db']
        lines.append(f"{row['utc_hour']:25s} {row['estimated_snr_db_2500hz']:7.1f}  [{lo:6.1f}, {hi:6.1f}]  {m['FT8']:10.1f}  {m['CW_DX']:12.1f}  {m['PHONE_SSB_DX']:13.1f}")
    lines+=['','SNR and margins use a 2500 Hz noise bandwidth. Margins use theoretical thresholds.',
        'History: '+json.dumps(result['solar_history_assumption']),result['calibration_limit']]
    outside=sorted({name for row in result['hourly_predictions'] for name in row['outside_training_ranges']})
    if outside:lines.append('Outside fitted input ranges: '+', '.join(outside))
    if any(row['magnetic_epoch_extrapolation'] for row in result['hourly_predictions']):lines.append('Requested year is outside the magnetic-coordinate coefficient range.')
    if any(row['antipodal_path_unsupported'] for row in result['hourly_predictions']):lines.append('The shortest path is ambiguous for these antipodal endpoints.')
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact');p.add_argument('tx');p.add_argument('rx');p.add_argument('utc')
    p.add_argument('--sfi',type=float,required=True);p.add_argument('--kp',type=float,required=True);p.add_argument('--ap',type=float,required=True)
    p.add_argument('--mode',choices=['WSPR','FT8','FT4','CW'],default='WSPR');p.add_argument('--power-watts',type=float,default=100)
    p.add_argument('--hours',type=int,default=24);p.add_argument('--history-file');p.add_argument('--format',choices=['json','table'],default='table')
    a=p.parse_args();history=json.loads(Path(a.history_file).read_text()) if a.history_file else None
    r=predict(a.artifact,a.tx,a.rx,a.utc,a.sfi,a.kp,a.ap,a.mode,a.power_watts,a.hours,history)
    print(json.dumps(r,indent=2,allow_nan=False) if a.format=='json' else table(r))
