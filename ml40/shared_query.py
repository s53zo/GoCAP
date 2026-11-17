"""One shared-model reference SNR and theoretical margins for all modes."""
import argparse,json
from pathlib import Path
ARTIFACT=Path(__file__).resolve().parents[1]/'artifacts/gocap-40m-shared-points.joblib'
from .hourly_query import predict,table

def query(tx,rx,utc,sfi,kp,ap,power_watts=5,hours=24,history=None,artifact=ARTIFACT):
    result=predict(artifact,tx,rx,utc,sfi,kp,ap,mode='WSPR',power_watts=power_watts,hours=hours,history=history)
    result['shared_model_scope']='One pooled WSPR/FT8/FT4/CW regressor, queried in the WSPR observation context at supplied power; all theoretical mode margins use this same reference SNR.'
    result['physical_mode_independence_verified']=False
    result['phone_measurement_validation']=False
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tx');p.add_argument('rx');p.add_argument('utc');p.add_argument('--sfi',type=float,required=True);p.add_argument('--kp',type=float,required=True);p.add_argument('--ap',type=float,required=True)
    p.add_argument('--power-watts',type=float,default=5);p.add_argument('--hours',type=int,default=24);p.add_argument('--history-file');p.add_argument('--format',choices=['table','json'],default='table')
    a=p.parse_args();history=json.load(open(a.history_file)) if a.history_file else None
    r=query(a.tx,a.rx,a.utc,a.sfi,a.kp,a.ap,a.power_watts,a.hours,history)
    print(json.dumps(r,indent=2,allow_nan=False) if a.format=='json' else table(r)+'\n'+r['shared_model_scope'])
