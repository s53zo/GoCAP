"""Separable 40 m corrections with a grid-four-only deployment geometry."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

NODES=np.array([0.,120.,350.,700.,1200.,2200.,4000.,8000.])
WIDTH=np.array([100.,150.,220.,320.,500.,800.,1600.,3500.])


def held(call):
    return int.from_bytes(hashlib.sha256(call.encode()).digest()[:4],'big')%5==0


def offsets(s):
    return ((ord(s[5])-65+.5)/24-.5,(ord(s[4])-65+.5)/12-1)


def fine(s):
    return s.str.fullmatch('[A-R]{2}[0-9]{2}[A-X]{2}').fillna(False)


def locations(d):
    tables={}
    for side in ['tx','rx']:
        rows=d[[side+'_sign',side+'_loc',side+'_grid']].drop_duplicates()
        rows=rows[fine(rows[side+'_loc'])]
        for grid,g in rows.groupby(side+'_grid'):
            loc=g.sort_values(side+'_loc')[side+'_loc'].tolist()
            tables[side,grid]=[offsets(loc[int(len(loc)*q)]) for q in (.125,.375,.625,.875)]
    return tables


def positions(d,side,kind,tables,index=0):
    a=d[side+'_lat'].to_numpy()
    b=d[side+'_lon'].to_numpy()
    if kind=='centre':
        return a,b
    if kind=='fine':
        val=d[side+'_loc']
        lut={s:offsets(s) if len(s)==6 and 'A'<=s[4]<='X' and 'A'<=s[5]<='X' else (0.,0.) for s in val.unique()}
        off=np.array(val.map(lut).tolist())
    else:
        fallback=[(-.25,-.5),(.25,-.5),(-.25,.5),(.25,.5)]
        val=d[side+'_grid']
        lut={s:tables.get((side,s),fallback)[index] for s in val.unique()}
        off=np.array(val.map(lut).tolist())
    return a+off[:,0],b+off[:,1]


def design(tx,rx,sun):
    a,l=np.radians(tx)
    b,m=np.radians(rx)
    diff=m-l
    dist=6371.*np.arccos(np.clip(np.sin(a)*np.sin(b)+np.cos(a)*np.cos(b)*np.cos(diff),-1,1))
    bearing=np.arctan2(np.sin(diff)*np.cos(b),np.cos(a)*np.sin(b)-np.sin(a)*np.cos(b)*np.cos(diff))
    radial=np.exp(-.5*((dist[:,None]-NODES)/WIDTH)**2)
    return np.column_stack([radial,radial*sun[:,None],radial[:,:4]*np.sin(bearing)[:,None],
                            radial[:,:4]*np.cos(bearing)[:,None]])


def geo(d,kind,tables,reg=None):
    sun=d.mid_sun.to_numpy()
    if kind!='marginal':
        x=design(positions(d,'tx',kind,tables),positions(d,'rx',kind,tables),sun)
        return x if reg is None else reg.predict(x)
    tx=[positions(d,'tx',kind,tables,i) for i in range(4)]
    rx=[positions(d,'rx',kind,tables,i) for i in range(4)]
    return sum(reg.predict(design(a,b,sun)) for a in tx for b in rx)/16


def noise_terms(d,pack):
    clock=d.hour.dt.hour
    index=pd.MultiIndex.from_arrays([d.rx_grid,clock],names=['rx_grid','utc'])
    typ=pack['typical'].reindex(index).fillna(0).to_numpy()
    live=d.noise_now.to_numpy(dtype=float,na_value=np.nan)
    return typ,np.where(np.isfinite(live),live-typ,0.)


def component_predictions(d,pack):
    typ,live=noise_terms(d,pack)
    power=(pack['slope']*(d.power_dbm-30)+d.base30-d.base).to_numpy()
    absorption=np.column_stack([d.abs_mean,d.abs_max]).clip(0,30)
    return pd.DataFrame({
        'geometry_centre':geo(d,'centre',{},pack['centre']),
        'geometry':geo(d,'marginal',pack['locations'],pack['fine']),
        'typical_noise':pack['typ_coef']*typ,
        'live_noise':pack['live_coef']*live,
        'power':power,
        'absorption':np.nan_to_num(absorption)@pack['abs_coef'],
    },index=d.index)


def fit(d,steps):
    y=(d.snr2500-d.base).to_numpy()
    pack={'locations':locations(d)}
    both=fine(d.tx_loc)&fine(d.rx_loc)
    for kind in ['centre','fine']:
        x=geo(d[both],kind,{})
        pack[kind]=Ridge(alpha=300).fit(x,y[both])
    nd=d.dropna(subset=['typical']).assign(utc=lambda g:g.hour.dt.hour)
    # Equal station/day weight within square and UTC hour.
    station=nd.assign(date=nd.hour.dt.date).drop_duplicates(['rx_sign','rx_grid','hour'])
    pack['typical']=station.groupby(['rx_grid','utc']).typical.median()
    typ,live=noise_terms(d,pack)
    def slope(x,r):
        return np.clip(x@r/(x@x+100.),-1.,0.)
    pack['typ_coef']=slope(typ,y)
    use=d.noise_now.notna().to_numpy()
    pack['live_coef']=slope(live[use],(y-pack['typ_coef']*typ)[use])
    levels=(steps.delta/steps.step).groupby(steps.tx_sign).agg(['median','size'])
    pack['slope']=float(np.clip(levels[levels['size']>=5]['median'].median(),0,1.5))
    use=d.abs_mean.notna()&d.abs_max.notna()
    x=d.loc[use,['abs_mean','abs_max']].to_numpy().clip(0,30)
    # Absorption is a signal-loss term with zero contribution for missing maps.
    from scipy.optimize import lsq_linear
    pack['abs_coef']=lsq_linear(np.vstack([x,10*np.eye(2)]),np.r_[y[use],[0.,0.]],bounds=(-3.,0.)).x
    return pack


SPECS={'default':[], 'grid4_control':['geometry_centre'], 'grid6_marginal':['geometry'],
       'noise_scenario':['typical_noise'], 'noise_live':['typical_noise','live_noise'],
       'power_steps':['power'], 'flare':['absorption'],
       'scenario_combined':['geometry','typical_noise','power'],
       'live_combined':['geometry','typical_noise','live_noise','power','absorption']}
