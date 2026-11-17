"""Experimental 40 m WSPR scenario and prior-hour nowcast query."""
import argparse
from contextlib import nullcontext

import joblib
import numpy as np
import pandas as pd

from .five_data import ROOT, sha
from .five_features import climate, donors, native, soundings, xyz
from .five_models import SPECS
from .hourly_models import predict_profiles
from .inference import centre
from .v3_features import scenario
from .voacap_engine import Engine


def query(tx,rx,utc,sfi,kp,ap,power=5,hours=24,model='scenario',donor_file=None,
          tx_call=None,rx_call=None,sonde_file=None):
    if not 1 <= hours <= 24 or not np.isfinite([sfi,kp,ap,power]).all():
        raise ValueError('Choose finite inputs and 1 to 24 hours')
    if (donor_file or sonde_file) and hours!=1:
        raise ValueError('Measured inputs require a single query hour')
    if donor_file and not (tx_call and rx_call):
        raise ValueError('Supply both station calls to exclude the target from donor observations')
    start = pd.Timestamp(utc)
    start = (start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')).floor('h')
    pack = joblib.load(ROOT/'artifacts/gocap-40m-five.joblib')
    model = pack['selected'].get(model, model)
    rows = [scenario(tx,rx,start+pd.Timedelta(hours=i),sfi,kp,ap,'WSPR',power)[0] for i in range(hours)]
    d = pd.concat(rows,ignore_index=True)
    base = ROOT/'artifacts/gocap-40m-hourly.joblib'
    if sha(base)!=pack['base_sha256']:
        raise ValueError('Base artifact differs from the version used to fit the corrections')
    bundle = joblib.load(base)
    d['base'] = predict_profiles(d,bundle['models'],bundle['profile_selection'])
    d['utc_hour'] = d.hour.dt.hour
    txlat,txlon = centre(tx.upper())
    rxlat,rxlon = centre(rx.upper())
    mid = xyz([txlat],[txlon])[0]+xyz([rxlat],[rxlon])[0]
    d['mid_lon'] = np.degrees(np.arctan2(mid[1],mid[0]))
    d['tx_lon'],d['rx_lon'] = txlon,rxlon
    blocks = SPECS[model][2] if model!='default' else ()
    parts = []
    with (Engine() if 'voacap' in blocks else nullcontext()) as engine:
        for _,part in d.groupby(d.hour.dt.date):
            part = part.copy()
            if 'f2' in blocks:
                fo = climate(part)
                if 'sonde' in blocks:
                    readings = pd.read_csv(sonde_file) if sonde_file else pd.DataFrame()
                    soundings(part,fo,readings)
            if 'voacap' in blocks:
                part = native(part,engine,{})
            if 'donors' in blocks:
                part['live'],part['donor_count'] = 0.,0
                if donor_file:
                    pool = pd.read_csv(donor_file)
                    pool = pool[pd.to_datetime(pool.hour,utc=True).eq(start-pd.Timedelta(hours=1))].copy()
                    for role in ['tx','rx']:
                        pool[role+'_sign'] = pool[role+'_sign'].str.upper()
                        coords = [centre(g.upper()) for g in pool[role+'_grid']]
                        pool[role+'_lat'] = [v[0] for v in coords]
                        pool[role+'_lon'] = [v[1] for v in coords]
                        pool[role+'_grid'] = pool[role+'_grid'].str.upper()
                    if not np.isfinite(pool[['base','snr2500']]).all().all():
                        raise ValueError('Donor signal levels and base predictions must be finite')
                    part['tx_sign'],part['rx_sign'] = tx_call.upper(),rx_call.upper()
                    part['live'],part['donor_count'] = donors(part,pool)
            parts.append(part)
    d = pd.concat(parts,ignore_index=True)
    point = d.base if model=='default' else pack['models'][model].predict(d)
    half = pack['half_width'][model]
    result = pd.DataFrame(dict(utc=d.hour,model=model,snr_db=point,lower80=point-half,upper80=point+half))
    for key in ['donor_count','sonde_available']:
        if key in d:
            result[key] = d[key]
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['tx','rx','utc']:
        parser.add_argument(key)
    for key in ['sfi','kp','ap']:
        parser.add_argument('--'+key,type=float,required=True)
    parser.add_argument('--power',type=float,default=5)
    parser.add_argument('--hours',type=int,default=24)
    parser.add_argument('--model',choices=['scenario','nowcast','default',*SPECS],default='scenario')
    for key in ['donor-file','tx-call','rx-call','sonde-file']:
        parser.add_argument('--'+key)
    print(query(**vars(parser.parse_args())).to_string(index=False,float_format=lambda x:f'{x:.2f}'))
