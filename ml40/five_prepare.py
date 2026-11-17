import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

from experiment import features, grid4, hourly
from .prepare import annotate
from .v3_features import enrich
from .hourly_models import predict_profiles
from .five_features import climate, soundings, native, donors
from .five_data import ROOT, OUT, CACHE, DATES, sha
from .voacap_engine import Engine

CASE = ['tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']
DEFAULT = ROOT/'artifacts/gocap-40m-hourly.joblib'
DRAWS = [f'draw_{i}' for i in range(20)]


def measurements(path):
    raw = pd.read_json(path,lines=True)
    raw = raw[raw.code.eq(1) & raw.power.between(20,50)].copy()
    raw['time'] = pd.to_datetime(raw.time,utc=True)
    raw = raw.assign(source='WSPRNET',mode='WSPR',snr2500=raw.snr,power_dbm=raw.power)
    truth = hourly(raw)
    raw['tx_grid'],raw['rx_grid'] = raw.tx_loc.map(grid4),raw.rx_loc.map(grid4)
    raw = raw[raw.tx_grid.notna() & raw.rx_grid.notna() & raw.frequency.between(7e6,7.3e6)
              & raw.snr.between(-60,100) & raw.tx_sign.str.len().ge(2) & raw.rx_sign.str.len().ge(2)]
    raw = raw.drop_duplicates(['time','tx_sign','rx_sign','frequency']).copy()
    raw['hour'] = raw.time.dt.floor('h')
    g = raw.groupby(CASE+['hour']).snr
    grouped = g.agg(list).rename('signals').reset_index()
    merged = truth.merge(grouped,on=CASE+['hour'],validate='one_to_one')
    assert len(merged)==len(truth) and (merged.signals.str.len()==merged.reports).all()
    assert np.allclose(merged.signals.map(np.median),merged.snr2500)
    return merged


def prepare(date,bundle,engine,curves):
    d = measurements(CACHE/'raw'/f'{date}.jsonl')
    d['decoder_context'] = 'ordinary_code1'
    d = enrich(annotate(features(d,solar_start='2009')))
    d['base'] = predict_profiles(d,bundle['models'],bundle['profile_selection'])
    d = d[d.distance_km.between(100,15000)].copy()
    cases = d[CASE+['distance_km']].drop_duplicates().copy()
    cases['stratum'] = pd.cut(cases.distance_km,[0,1000,3000,6000,15000])
    # Seeded uniform station-pair sample within each distance class; no activity threshold.
    cases = cases.sample(frac=1,random_state=int(date.replace('-',''))).groupby('stratum',observed=True).head(60)
    cases['case'] = [f'{date}/{i}' for i in range(len(cases))]
    target = d.merge(cases[CASE+['case']],on=CASE,validate='many_to_one').reset_index(drop=True)
    for h,part in target.groupby('hour'):
        prev = d[d.hour.eq(h-pd.Timedelta(hours=1))]
        live,count = donors(part,prev)
        target.loc[part.index,'live'] = live
        target.loc[part.index,'donor_count'] = count
    rng = np.random.default_rng(int(date.replace('-','')))
    draws = np.stack([rng.choice(values,len(DRAWS)) for values in target.signals])
    target[DRAWS] = draws
    target['within_sd'] = target.signals.map(np.std)
    target = target.drop(columns='signals')
    fo = climate(target)
    soundings(target,fo,pd.read_csv(CACHE/'sonde'/f'{date}.csv'))
    return native(target,engine,curves)


def run():
    bundle = joblib.load(DEFAULT)
    (CACHE/'features').mkdir(exist_ok=True)
    manifest = json.loads((OUT/'sources.json').read_text())
    curves = {}
    with Engine() as engine, threadpool_limits(limits=3):
        for dates in DATES.values():
            for date in dates:
                path = CACHE/'features'/f'{date}.parquet'
                if path.exists():
                    continue
                assert sha(CACHE/'raw'/f'{date}.jsonl')==manifest[date]['sha256']
                frame = prepare(date,bundle,engine,curves)
                assert not frame.duplicated(['case','hour']).any()
                temp = path.with_suffix('.part')
                frame.to_parquet(temp,index=False)
                temp.replace(path)
                print(date,len(frame),flush=True)


if __name__=='__main__':
    run()
