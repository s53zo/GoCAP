"""Produce hourly targets, strictly lagged environmental inputs and matched observations."""
import json
import re
from itertools import combinations
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from experiment import features,grid4
from .five_features import xyz
from .five_prepare import DEFAULT,CASE
from .four_data import CACHE,ALL,OUT
from .hourly_models import predict_profiles
from .prepare import annotate
from .v3_features import enrich


def noise(date):
    cols=['rx_sign','rx_grid','hour','noise_now','typical','channels']
    p=CACHE/f'noise-{date}.jsonl'
    if p.stat().st_size==0:
        return pd.DataFrame(columns=cols)
    n=pd.read_json(p,lines=True).rename(columns={'h':'hour','site':'rx_sign','grid':'rx_grid'})
    n['hour']=pd.to_datetime(n.hour,utc=True)
    n=n[n.n.ge(3)].copy()
    keys=['rx_sign','rx_grid','receiver']
    start=pd.Timestamp(date,tz='UTC')
    history=n[n.hour<start-pd.Timedelta(hours=1)]
    med=history.groupby(keys).noise.agg(['median','count'])
    n=n.merge(med[med['count']>=24],on=keys)
    n['noise_now']=(n.noise-n['median']).clip(-25,25)
    n['utc']=n.hour.dt.hour
    typ=n[n.hour<start-pd.Timedelta(hours=1)].groupby(keys+['utc']).noise_now.median().rename('typical')
    n['hour']+=pd.Timedelta(hours=1)
    n=n[n.hour.dt.date==start.date()].copy()
    n['utc']=n.hour.dt.hour
    n=n.merge(typ,on=keys+['utc'],how='left')
    return n.groupby(['rx_sign','rx_grid','hour']).agg(noise_now=('noise_now','median'),typical=('typical','median'),channels=('receiver','size')).reset_index()


def field(d,grids):
    a=xyz(d.tx_lat,d.tx_lon)
    b=xyz(d.rx_lat,d.rx_lon)
    angle=np.arccos(np.clip((a*b).sum(1),-1,1))
    values=[]
    for u in [0,.25,.5,.75,1]:
        v=np.sin((1-u)*angle)[:,None]*a+np.sin(u*angle)[:,None]*b
        v[angle<1e-7]=a[angle<1e-7]
        lat=np.degrees(np.arctan2(v[:,2],np.hypot(v[:,0],v[:,1])))
        lon=np.degrees(np.arctan2(v[:,1],v[:,0]))
        row=np.clip(np.floor((90-lat)/2).astype(int),0,89)
        col=np.floor((lon+180)/4).astype(int)%90
        values.append(grids[d.hour.dt.hour,row,col])
    # NOAA's 1 dB highest affected frequency relative to 7.04 MHz.
    proxy=(np.maximum(np.stack(values,axis=1),0)/7.04)**1.5
    return proxy.mean(1),proxy.max(1)


def clean(date,prefix='spots'):
    r=pd.read_json(CACHE/f'{prefix}-{date}.jsonl',lines=True)
    r['time']=pd.to_datetime(r.time,utc=True)
    r['hour']=r.time.dt.floor('h')
    for side in ['tx','rx']:
        r[side+'_loc']=r[side+'_loc'].str.upper()
        r[side+'_grid']=r[side+'_loc'].map(grid4)
    r=r[r.code.eq(1)&r.frequency.between(7e6,7.3e6)&r.snr.between(-60,100)&r.power.between(0,60)
        &r.tx_sign.str.len().ge(2)&r.rx_sign.str.len().ge(2)].dropna(subset=['tx_grid','rx_grid'])
    return r.drop_duplicates(['time','tx_sign','rx_sign','frequency']).rename(columns={'power':'power_dbm'})


def power_pairs(r):
    keys=['tx_sign','rx_sign','tx_loc','rx_loc','hour']
    x=r.sort_values(keys+['time','id']).copy()
    # Ambiguous simultaneous messages at two powers cannot identify a step.
    same=x.groupby(keys+['time']).power_dbm.transform('nunique')
    x=x[same.eq(1)].drop_duplicates(keys+['time']).copy()
    g=x.groupby(keys)
    dt=g.time.diff().dt.total_seconds()
    x['step']=g.power_dbm.diff()
    x['delta']=g.snr.diff()
    x['previous_snr']=g.snr.shift()
    use=dt.between(120,720)&x.step.abs().between(3,20)&g.frequency.diff().abs().le(100)
    return x.loc[use,keys+['time','tx_grid','rx_grid','power_dbm','step','delta','snr','previous_snr']]


def matches(r):
    keys=['time','tx_sign','power_dbm','rx_grid']
    multi=r[r.duplicated(keys,keep=False)]
    rows=[]
    for _,group in multi.groupby(keys):
        for a,b in combinations(group.drop_duplicates('rx_sign').itertuples(),2):
            if abs(a.frequency-b.frequency)<=5:
                rows.append([a.hour,a.rx_grid,a.rx_sign,b.rx_sign,a.snr-b.snr,
                             a.rx_loc==b.rx_loc and bool(re.fullmatch('[A-R]{2}[0-9]{2}[A-X]{2}',a.rx_loc))])
    return pd.DataFrame(rows,columns=['hour','rx_grid','left','right','gap','same6'])


def prepare(date,bundle):
    raw=clean(date)
    p=power_pairs(raw)
    p.to_parquet(CACHE/f'steps-{date}.parquet',index=False)
    n=noise(date)
    m=matches(raw)
    if len(m):
        for side in ['left','right']:
            m=m.merge(n.rename(columns={'rx_sign':side,'noise_now':side+'_noise'})[[side,'hour','rx_grid',side+'_noise']],
                      on=[side,'hour','rx_grid'],how='left',validate='many_to_one')
    m.to_parquet(CACHE/f'matched-{date}.parquet',index=False)
    g=raw.groupby(CASE+['hour'])
    d=g.agg(snr2500=('snr','median'),reports=('snr','size'),tx_loc=('tx_loc','first'),rx_loc=('rx_loc','first'),
            nt=('tx_loc','nunique'),nr=('rx_loc','nunique')).reset_index()
    for side,count in [('rx','nr'),('tx','nt')]:
        d[side+'_loc']=d[side+'_loc'].where(d[count].eq(1),d[side+'_grid'])
    d=d.assign(source='WSPRNET',mode='WSPR',decoder_context='ordinary_code1')
    d=enrich(annotate(features(d,solar_start='2009')))
    d=d[d.distance_km<=15000].reset_index(drop=True)
    d['base']=predict_profiles(d,bundle['models'],bundle['profile_selection'])
    d['base30']=predict_profiles(d.assign(power_dbm=30.),bundle['models'],bundle['profile_selection'])
    d=d.merge(n,on=['hour','rx_grid','rx_sign'],how='left',validate='many_to_one')
    path=CACHE/'drap'/f'{date}.npz'
    d['abs_mean'],d['abs_max']=field(d,np.load(path)['grids'])
    # Post-hour messages label the evaluation subset only.
    msgs=json.loads(path.with_suffix('.json').read_text())['messages']
    def status(h):
        vals=[s.upper() for t,s in msgs.items() if int(t[:2])==h]
        if any(any(word in s for word in ('MINOR','MODERATE','STRONG','SEVERE','EXTREME')) for s in vals):return 'flare'
        if vals and all('NORMAL' in s for s in vals):return 'quiet'
        return 'other'
    d['status']=d.hour.dt.hour.map(status)
    cols=CASE+['hour','snr2500','reports','tx_loc','rx_loc','distance_km','tx_lat','rx_lat','tx_lon','rx_lon',
        'mid_sun','base','base30','noise_now','typical','channels','abs_mean','abs_max','status']
    d[cols].to_parquet(CACHE/f'hours-{date}.parquet',index=False)
    print(date,len(d),len(p),len(m),flush=True)


if __name__=='__main__':
    bundle=joblib.load(DEFAULT)
    with threadpool_limits(limits=3):
        for day in ALL:
            prepare(day,bundle)
