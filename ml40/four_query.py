"""Query the experimental WSPR model using four-character Maidenhead endpoints."""
import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from .four_data import ROOT,sha
from .four_drap import parse_map
from .four_models import component_predictions,SPECS
from .four_prepare import field,DEFAULT
from .four_run import ART
from .four_transfer import TRANSFER, PARENT, basis
from .hourly_models import predict_profiles
from .inference import centre
from .v3_features import scenario


def issue_map(text,hour):
    found=re.search(r'Product Valid At\s*:\s*(\d{4}-\d\d-\d\d \d\d:\d\d) UTC',text)
    if not found:
        raise ValueError('Missing NOAA product validity timestamp')
    age=(hour-pd.Timestamp(found[1],tz='UTC'))/pd.Timedelta(minutes=1)
    if not 5<=age<=15:
        raise ValueError('Supply a D-RAP map valid 5–15 minutes before the query hour')
    return parse_map(text)


def query(tx,rx,utc,sfi,kp,ap,power=5,hours=24,model='march_scenario',drap_file=None,noise_anomaly=None):
    if not 1<=hours<=24 or not np.isfinite([sfi,kp,ap,power]).all() or not .001<=power<=1000:
        raise ValueError('Choose 1–24 hours, finite solar inputs and 0.001–1000 watts')
    if (drap_file or noise_anomaly is not None) and hours!=1:
        raise ValueError('Measured inputs apply to a single forecast hour')
    if noise_anomaly is not None and not np.isfinite(noise_anomaly):
        raise ValueError('Receiver noise anomaly must be finite')
    start=pd.Timestamp(utc)
    start=(start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')).floor('h')
    march=model in ('march_scenario','march_nowcast')
    base_path=PARENT if march else DEFAULT
    base=joblib.load(base_path)
    pack=joblib.load(TRANSFER if march else ART)
    if sha(base_path)!=pack['parent_sha256' if march else 'base_sha256']:
        raise ValueError('Incompatible baseline artifact')
    name=model if march else pack['selected'].get(model,model)
    rows=[scenario(tx,rx,start+pd.Timedelta(hours=h),sfi,kp,ap,'WSPR',power)[0] for h in range(hours)]
    d=pd.concat(rows,ignore_index=True)
    for side,grid in [('tx',tx),('rx',rx)]:
        d[side+'_lat'],d[side+'_lon']=centre(grid.upper())
    d['base']=predict_profiles(d,base['models'],base['profile_selection'])
    d['base30']=predict_profiles(d.assign(power_dbm=30.),base['models'],base['profile_selection'])
    d['noise_now']=np.nan if noise_anomaly is None else np.clip(noise_anomaly,-25.,25.)
    d['abs_mean']=np.nan
    d['abs_max']=np.nan
    if drap_file:
        grids=np.full((24,90,90),np.nan)
        grids[start.hour]=issue_map(Path(drap_file).read_text(),start)
        d['abs_mean'],d['abs_max']=field(d,grids)
    if march:
        coef=pack[name.removeprefix('march_')]
        correction=basis(d,pack)[:,:len(coef)]@coef
    else:
        x=component_predictions(d,pack)
        correction=sum((v*x[k] for k,v in pack['weights'][name].items()),start=0)
    pred=d.base+np.where(d.distance_km<=15000,correction,0.)
    return pd.DataFrame(dict(utc=d.hour,model=name,snr_db=pred,baseline_db=d.base,
                             map_available=d.abs_mean.notna(),within_study_range=d.distance_km<=15000))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['tx','rx','utc']:
        p.add_argument(k)
    for k in ['sfi','kp','ap']:
        p.add_argument('--'+k,type=float,required=True)
    p.add_argument('--power',type=float,default=5.)
    p.add_argument('--hours',type=int,default=24)
    p.add_argument('--model',choices=['scenario','nowcast','march_scenario','march_nowcast',*SPECS],default='march_scenario')
    p.add_argument('--drap-file')
    p.add_argument('--noise-anomaly',type=float)
    print(query(**vars(p.parse_args())).to_string(index=False,float_format=lambda v:f'{v:.2f}'))
