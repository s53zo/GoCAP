"""Physics, native diurnal shape and prior-hour donor features for the pilot."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .inference import centre
from .voacap_engine import make_deck, at_bin_centre


def xyz(lat, lon):
    a, b = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)])


def climate(d):
    import PyIRI
    from PyIRI.main_library import IRI_density_1day
    time = d.hour.iloc[0]
    paths = d[['mid_lat','mid_lon']].drop_duplicates().reset_index(drop=True)
    loc = paths.reset_index(names='site')
    ids = d[['mid_lat','mid_lon']].merge(loc, sort=False, validate='many_to_one').site.to_numpy()
    f2, *_ = IRI_density_1day(time.year,time.month,time.day,np.arange(24)+.5,
                               paths.mid_lon.to_numpy(),paths.mid_lat.to_numpy(),
                               np.array([300.]),float(d.sfi_previous_day.iloc[0]),PyIRI.coeff_dir,1)
    h = d.hour.dt.hour.to_numpy()
    height, fo = f2['hm'][h, ids], f2['fo'][h, ids]
    # One-hop spherical-Earth secant approximation, restricted to <= 1000 km.
    angle = d.distance_km.to_numpy()/(2*6371.)
    vertical = height + 6371.*(1-np.cos(angle))
    required = 7.04 * vertical/np.hypot(vertical,6371.*np.sin(angle))
    d['f2_margin'] = np.where(d.distance_km <= 1000, fo-required, 0.)
    return fo


def soundings(d, fo, table):
    d['sonde_delta'] = 0.
    d['sonde_available'] = False
    if table.empty:
        return
    table = table[(table.confidence.between(60,100)) & table.fo.between(1,20)].copy()
    if table.empty:
        return
    table['time'] = pd.to_datetime(table.time, utc=True)
    match = pd.merge_asof(d[['hour']].reset_index(drop=True).reset_index(names='pos').sort_values('hour'),
                          table.sort_values('time'), left_on='hour', right_on='time',
                          direction='backward',allow_exact_matches=False,tolerance=pd.Timedelta('90min')).sort_values('pos')
    km = 6371.*np.arccos(np.clip(xyz(d.mid_lat,d.mid_lon) @ xyz([50.1],[4.6])[0],-1,1))
    use = (km <= 800) & d.distance_km.le(1000).to_numpy() & match.fo.notna().to_numpy()
    d['sonde_available'] = use
    d['sonde_delta'] = np.where(use, match.fo.to_numpy()-fo, 0.)


def native(d, engine, cache):
    date = d.hour.iloc[0]
    records = []
    for tx, rx in d[['tx_grid','rx_grid']].drop_duplicates().itertuples(index=False):
        key = (date.year,date.month,tx,rx)
        if key not in cache:
            deck = make_deck(*centre(tx),*centre(rx),date.year,date.month,100.,5.)
            _, curve = engine.run(deck)
            raw = np.array([at_bin_centre(curve,h)[0] for h in range(24)], dtype=float)
            cache[key] = (raw - np.nanmedian(raw), raw - np.roll(raw,1))
        shape, slope = cache[key]
        records.extend((tx,rx,h,shape[h],slope[h]) for h in range(24))
    curves = pd.DataFrame(records,columns=['tx_grid','rx_grid','utc_hour','voa_shape','voa_slope'])
    return d.merge(curves,on=['tx_grid','rx_grid','utc_hour'],validate='many_to_one')


def donors(target, previous):
    result, support = np.zeros(len(target)), np.zeros(len(target),dtype=int)
    if previous.empty:
        return result, support
    def coords(frame):
        return np.column_stack([xyz(frame.tx_lat,frame.tx_lon),xyz(frame.rx_lat,frame.rx_lon)])
    xy, pool = coords(target), coords(previous)
    radius = 2*np.sin(1200./(2*6371.))
    groups = cKDTree(pool).query_ball_point(xy,r=radius*np.sqrt(2))
    stations = previous[['tx_sign','rx_sign','tx_grid','rx_grid']].to_numpy()
    anomaly = (previous.snr2500-previous.base).to_numpy()
    for i, (c, group) in enumerate(zip(target.itertuples(),groups)):
        if not group:
            continue
        ix = np.asarray(group)
        dist = np.linalg.norm((pool[ix]-xy[i]).reshape(-1,2,3),axis=2)
        order = np.argsort(dist.sum(axis=1),kind='stable')
        used = {c.tx_sign,c.rx_sign}
        values, weights = [], []
        for j in order:
            tx,rx,tg,rg = stations[ix[j]]
            if max(dist[j]) > radius or tx in used or rx in used or (tg,rg)==(c.tx_grid,c.rx_grid):
                continue
            used.update((tx,rx))
            values.append(np.clip(anomaly[ix[j]],-20,20))
            weights.append(np.exp(-np.square(dist[j]/radius).sum()))
            if len(values)==12:
                break
        support[i] = len(values)
        if len(values)>=3:
            result[i] = np.average(values,weights=weights)
    return result,support
