"""Vectorized path geometry, IGRF degree-one coordinates and completed solar history.

No reception or solar database is read by scenario inference.
"""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from .constants import BASE,SOLAR
from .inference import scenario_frame,KP_AP,centre

MAG=json.loads((Path(__file__).parent/'igrf_degree1.json').read_text())
PATH=['path_daylight_fraction','path_twilight_fraction','path_sun_mean','path_sun_min',
      'path_max_abs_lat','path_max_abs_magnetic_lat','tx_magnetic_lat','rx_magnetic_lat',
      'bearing_sin','bearing_cos','tx_hours_from_sunrise','rx_hours_from_sunrise',
      'tx_hours_from_sunset','rx_hours_from_sunset','tx_polar_day','rx_polar_day',
      'tx_polar_night','rx_polar_night','antipodal_path','magnetic_epoch_outside']
HISTORY=['ap_mean6','ap_mean72','ap_max24','ap_max72','ap_trend6_vs_previous18',
         'hours_since_ap48_capped72','sfi_mean27','sfi_difference27']
# Power is normalized in the target, never silently inferred for unknown-power modes.
BASE_NUMERIC=[c for c in BASE+SOLAR+['power_known','legacy_decoder'] if c!='power_dbm']
RICH_NUMERIC=BASE_NUMERIC+PATH+HISTORY


def geometry(d):
    d=d.copy();n=len(d)
    coords={g:centre(g) for g in set(d.tx_grid)|set(d.rx_grid)}
    ta=np.array([coords[g][0] for g in d.tx_grid]);tl=np.array([coords[g][1] for g in d.tx_grid])
    ra=np.array([coords[g][0] for g in d.rx_grid]);rl=np.array([coords[g][1] for g in d.rx_grid])
    a,b,la,lb=map(np.deg2rad,[ta,ra,tl,rl])
    va=np.column_stack([np.cos(a)*np.cos(la),np.cos(a)*np.sin(la),np.sin(a)])
    vb=np.column_stack([np.cos(b)*np.cos(lb),np.cos(b)*np.sin(lb),np.sin(b)])
    omega=np.arccos(np.clip((va*vb).sum(axis=1),-1,1));antipodal=omega>np.pi-1e-6
    denom=np.sin(omega);small=omega<1e-8;safe=np.where(abs(denom)<1e-8,1,denom)
    t=np.linspace(0,1,9)
    points=(np.sin((1-t[None,:])*omega[:,None])[:,:,None]*va[:,None,:]+
            np.sin(t[None,:]*omega[:,None])[:,:,None]*vb[:,None,:])/safe[:,None,None]
    points[small]=va[small,None,:]
    points/=np.maximum(np.linalg.norm(points,axis=2)[:,:,None],1e-12)
    lat=np.arctan2(points[:,:,2],np.hypot(points[:,:,0],points[:,:,1]));lon=np.arctan2(points[:,:,1],points[:,:,0])
    doy=d.hour.dt.day_of_year.to_numpy();hour=d.hour.dt.hour.to_numpy()+.5
    phase=2*np.pi*(doy-1)/365.2425
    dec=(.006918-.399912*np.cos(phase)+.070257*np.sin(phase)-.006758*np.cos(2*phase)+
         .000907*np.sin(2*phase)-.002697*np.cos(3*phase)+.00148*np.sin(3*phase))
    eq=229.18*(.000075+.001868*np.cos(phase)-.032077*np.sin(phase)-.014615*np.cos(2*phase)-.040849*np.sin(2*phase))
    ha=np.deg2rad(15*(hour[:,None]+np.rad2deg(lon)/15+eq[:,None]/60-12))
    sun=np.sin(lat)*np.sin(dec[:,None])+np.cos(lat)*np.cos(dec[:,None])*np.cos(ha)
    d['path_daylight_fraction']=(sun>0).mean(axis=1)
    d['path_twilight_fraction']=(abs(sun)<np.sin(np.deg2rad(6))).mean(axis=1)
    d['path_sun_mean']=sun.mean(axis=1);d['path_sun_min']=sun.min(axis=1)
    d['path_max_abs_lat']=abs(np.rad2deg(lat)).max(axis=1)
    year=d.hour.dt.year.to_numpy()+(doy-1)/365.2425
    g10=np.interp(year,MAG['years'],MAG['coefficients']['g10'])
    g11=np.interp(year,MAG['years'],MAG['coefficients']['g11'])
    h11=np.interp(year,MAG['years'],MAG['coefficients']['h11'])
    axis=-np.column_stack([g11,h11,g10]);axis/=np.linalg.norm(axis,axis=1)[:,None]
    magnetic=np.rad2deg(np.arcsin(np.clip((points*axis[:,None,:]).sum(axis=2),-1,1)))
    d['path_max_abs_magnetic_lat']=abs(magnetic).max(axis=1)
    d['tx_magnetic_lat']=magnetic[:,0];d['rx_magnetic_lat']=magnetic[:,-1]
    d['magnetic_epoch_outside']=((year<MAG['years'][0])|(year>MAG['years'][-1])).astype(float)
    bearing=np.arctan2(np.sin(lb-la)*np.cos(b),np.cos(a)*np.sin(b)-np.sin(a)*np.cos(b)*np.cos(lb-la))
    d['bearing_sin']=np.sin(bearing);d['bearing_cos']=np.cos(bearing)
    for role,latitude,longitude in [('tx',a,tl),('rx',b,rl)]:
        c=-np.tan(latitude)*np.tan(dec)
        polar_night=c>1;polar_day=c<-1
        sunrise=12-np.rad2deg(np.arccos(np.clip(c,-1,1)))/15
        sunset=24-sunrise;local=hour+longitude/15+eq/60
        d[role+'_hours_from_sunrise']=np.where(polar_night|polar_day,np.nan,(local-sunrise+12)%24-12)
        d[role+'_hours_from_sunset']=np.where(polar_night|polar_day,np.nan,(local-sunset+12)%24-12)
        d[role+'_polar_day']=polar_day.astype(float);d[role+'_polar_night']=polar_night.astype(float)
    d['antipodal_path']=antipodal.astype(float)
    for col in PATH:
        if col.startswith('path_'):d.loc[antipodal,col]=np.nan
    return d


@lru_cache(maxsize=1)
def solar_history():
    from experiment import solar_table
    blocks,daily=solar_table('2009');b=blocks.copy()
    ap=b.ap_previous
    b['ap_mean6']=ap.rolling(2,min_periods=2).mean()
    b['ap_mean72']=ap.rolling(24,min_periods=24).mean()
    b['ap_max24']=ap.rolling(8,min_periods=8).max();b['ap_max72']=ap.rolling(24,min_periods=24).max()
    b['ap_trend6_vs_previous18']=b.ap_mean6-ap.shift(2).rolling(6,min_periods=6).mean()
    position=pd.Series(np.arange(len(b)),index=b.index,dtype=float)
    previous=position.where(ap.ge(48)).ffill()
    b['hours_since_ap48_capped72']=((position-previous)*3).clip(upper=72).fillna(72)
    daily=daily.copy();daily['sfi_mean27']=daily.sfi_previous_day.rolling(27,min_periods=27).mean()
    return b[HISTORY[:6]],daily[['sfi_mean27']]


def enrich(d):
    d=geometry(d)
    blocks,daily=solar_history()
    boundary=pd.DatetimeIndex(d.hour).floor('3h')
    selected=blocks.reindex(boundary).reset_index(drop=True)
    for c in HISTORY[:6]:d[c]=selected[c].to_numpy()
    elapsed=(pd.DatetimeIndex(d.hour)-boundary).total_seconds()/3600
    d['hours_since_ap48_capped72']=np.minimum(72,d.hours_since_ap48_capped72.to_numpy()+elapsed)
    d['sfi_mean27']=daily.reindex(pd.DatetimeIndex(d.hour).floor('D')).sfi_mean27.to_numpy()
    d['sfi_difference27']=d.sfi_previous_day-d.sfi_mean27
    return d


def scenario(tx,rx,utc,sfi,kp,ap,mode='WSPR',power_watts=100,history=None):
    d=scenario_frame(tx,rx,utc,sfi,kp,ap,mode,power_watts);d=geometry(d)
    h=history or {};values=np.repeat(kp,24).astype(float)
    if 'kp_3h' in h:
        values=np.asarray(h['kp_3h'],dtype=float)
        if values.shape!=(24,) or not np.isfinite(values).all() or not ((values>=0)&(values<=9)).all():raise ValueError('kp_3h requires 24 completed 3-hour values in [0,9], oldest first')
        end=pd.Timestamp(h['ending_boundary_utc'])
        if end.tzinfo is None:raise ValueError('History boundary must include UTC timezone')
        if end.tz_convert('UTC')!=d.hour.iloc[0].floor('3h'):raise ValueError('History must end at the query hour\'s last completed 3-hour boundary')
        if not np.isclose(values[-1],kp):raise ValueError('Last historical Kp must match --kp')
    av=np.interp(values,np.arange(28)/3,KP_AP)
    sfi_values=np.asarray(h.get('sfi_previous27_days',[sfi]*27),dtype=float)
    if sfi_values.shape!=(27,) or not np.isfinite(sfi_values).all() or not (sfi_values>0).all():raise ValueError('SFI history requires 27 positive preceding completed-day values')
    if not np.isclose(sfi_values[-1],sfi):raise ValueError('Last SFI history value must match --sfi')
    d['ap_previous']=av[-1];d['ap_trailing24']=av[-8:].mean()
    d['ap_mean6']=av[-2:].mean();d['ap_mean72']=av.mean();d['ap_max24']=av[-8:].max();d['ap_max72']=av.max()
    d['ap_trend6_vs_previous18']=av[-2:].mean()-av[-8:-2].mean()
    elapsed=(d.hour.iloc[0]-d.hour.iloc[0].floor('3h')).total_seconds()/3600
    events=np.flatnonzero(av>=48)
    d['hours_since_ap48_capped72']=min(72,(23-events[-1])*3+elapsed) if len(events) else 72
    d['sfi_mean27']=sfi_values.mean();d['sfi_difference27']=sfi-sfi_values.mean()
    assumption={'kp_history':'supplied, timestamp checked' if 'kp_3h' in h else 'steady Kp over completed preceding72h',
        'sfi_history':'supplied preceding27 completed days' if 'sfi_previous27_days' in h else 'steady preceding27-day SFI',
        'Ap':'supplied previous-day context, independent of the Kp scenario'}
    return d,assumption
