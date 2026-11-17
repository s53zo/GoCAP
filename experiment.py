"""Reproducible positive-only, hourly 40m SNR pilot. No reception probabilities."""
from pathlib import Path
import hashlib
import json
import math
import platform
import sqlite3
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits

from collect import DATA, RAW

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

def grid4(value):
    import re
    s = str(value).strip().upper()
    return s[:4] if re.fullmatch(r'[A-R]{2}[0-9]{2}(?:[A-X]{2}(?:[0-9]{2})?)?', s) else None

def centre(grid):
    """Latitude, longitude of a 4-char square centre; never a country centroid."""
    if grid4(grid) is None:
        raise ValueError(grid)
    g = grid.upper()
    return -90 + (ord(g[1])-65)*10 + int(g[3]) + .5, -180 + (ord(g[0])-65)*20 + int(g[2])*2 + 1

def held_station(call):
    return int(hashlib.sha256(str(call).encode()).hexdigest()[:8], 16) % 5 == 0

def load_wspr():
    parts = [pd.read_json(p, lines=True) for p in sorted(RAW.glob('wspr-*.jsonl'))]
    d = pd.concat(parts, ignore_index=True)
    d['source'] = 'WSPRNET'
    d['mode'] = np.where(d.code.eq(1), 'WSPR', 'OTHER')
    d['snr2500'] = d.snr.astype(float)
    d['snr_native'] = d.snr.astype(float)
    d['power_dbm'] = d.power.astype(float)
    d['location_provenance'] = 'reported_locator'
    d['time'] = pd.to_datetime(d.time, utc=True)
    d['event_id'] = 'wspr:' + d.id.astype(str)
    return d

def load_pskr_current():
    p = RAW / 'pskr-current.xml'
    if not p.exists():
        return pd.DataFrame()
    rows = [r.attrib for r in ET.fromstring(p.read_text()).findall('receptionReport')]
    d = pd.DataFrame(rows).rename(columns={'receiverCallsign':'rx_sign','senderCallsign':'tx_sign',
        'receiverLocator':'rx_loc','senderLocator':'tx_loc','sNR':'snr2500'})
    d['time'] = pd.to_datetime(pd.to_numeric(d.flowStartSeconds), unit='s', utc=True)
    d['frequency'] = pd.to_numeric(d.frequency)
    d['snr2500'] = pd.to_numeric(d.get('snr2500'), errors='coerce')
    d['snr_native'] = d.snr2500
    # PSK Reporter accepts arbitrary decoders; CW bandwidth is not established here.
    d.loc[~d['mode'].isin(['FT8','FT4','WSPR']), 'snr2500'] = np.nan
    d['source'] = 'PSKR_LIVE'
    d['power_dbm'] = np.nan
    d['location_provenance'] = 'reported_locator'
    d['event_id'] = ['pskr-live:'+str(i) for i in range(len(d))]
    return d

def load_rbn_raw():
    parts=[]
    for path in sorted(RAW.glob('rbn-*.zip')):
        with zipfile.ZipFile(path) as z:
            d=pd.read_csv(z.open(z.namelist()[0]))
        d=d[d.band.eq('40m')].copy()
        d=d.rename(columns={'callsign':'rx_sign','dx':'tx_sign','date':'time','tx_mode':'mode'})
        # RBN's field named "mode" is CQ/DE; use tx_mode for the transmission mode.
        d=d.loc[:,~d.columns.duplicated(keep='last')]
        d['frequency']=pd.to_numeric(d.freq)*1000
        d['snr2500']=pd.to_numeric(d.db)-10*np.log10(2500/500)
        d['snr_native']=pd.to_numeric(d.db)
        d.loc[~d['mode'].eq('CW'),'snr2500']=np.nan
        d['time']=pd.to_datetime(d.time,utc=True)
        d['power_dbm']=np.nan
        d['tx_loc']=None;d['rx_loc']=None
        d['source']='RBN_RAW'
        d['location_provenance']='not present in daily raw archive'
        d['event_id']=[path.stem+':'+str(i) for i in d.index]
        parts.append(d[['source','mode','time','tx_sign','rx_sign','frequency','snr2500','snr_native',
            'power_dbm','tx_loc','rx_loc','location_provenance','event_id']])
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

@lru_cache(maxsize=4)
def solar_table(start='2022'):
    names = ['year','month','day','days','days_m','bsr','db'] + [f'kp{i}' for i in range(8)] + [f'ap{i}' for i in range(8)] + ['Ap','SN','F107obs','F107adj','definitive']
    d = pd.read_csv(RAW/'gfz.txt', sep=r'\s+', comment='#', names=names)
    d['date'] = pd.to_datetime(d[['year','month','day']], utc=True)
    d = d.set_index('date').sort_index()
    records = []
    # At hour H, use only completed 3h blocks and previous UTC-day daily indices.
    # These are final historical indices, not archived operational releases.
    for date, row in d.loc[start:].iterrows():
        for i in range(8):
            records.append({'available_boundary':date+pd.Timedelta(hours=3*(i+1)),
                'kp_previous': row[f'kp{i}'], 'ap_previous':row[f'ap{i}']})
    blocks = pd.DataFrame(records).set_index('available_boundary').sort_index()
    blocks[['kp_previous','ap_previous']] = blocks[['kp_previous','ap_previous']].replace(-1, np.nan)
    blocks['ap_trailing24'] = blocks.ap_previous.rolling(8, min_periods=8).mean()
    daily = d[['F107obs','Ap']].rename(columns={'F107obs':'sfi_previous_day','Ap':'Ap_previous_day'}).replace(-1, np.nan)
    daily.index = daily.index + pd.Timedelta(days=1)
    return blocks, daily

def audit(d):
    result = {}
    for source, g in d.groupby('source'):
        dupkey = ['source','time','tx_sign','rx_sign','frequency','mode']
        result[source] = {'raw_rows':len(g), 'days':g.time.dt.date.nunique(),
            'first_utc':str(g.time.min()), 'last_utc':str(g.time.max()),
            'modes':g['mode'].value_counts().to_dict(),
            'tx_stations':g.tx_sign.nunique(), 'rx_stations':g.rx_sign.nunique(),
            'invalid_tx_locator':int(g.tx_loc.map(grid4).isna().sum()),
            'invalid_rx_locator':int(g.rx_loc.map(grid4).isna().sum()),
            'missing_native_snr':int(g.snr_native.isna().sum()),
            'missing_or_unverified_snr2500':int(g.snr2500.isna().sum()),
            'missing_power':int(g.power_dbm.isna().sum()),
            'outside_broad_snr_screen':int((g.snr2500.notna() & ~g.snr2500.between(-60,100)).sum()),
            'duplicate_event_keys':int(g.duplicated(dupkey).sum()),
            'snr2500_quantiles':g.snr2500.quantile([0,.01,.1,.5,.9,.99,1]).to_dict() if g.snr2500.notna().any() else {},
            'power_dbm_counts':g.power_dbm.value_counts().to_dict(),
            'largest_rx_share':float(g.rx_sign.value_counts(normalize=True).iloc[0]),
            'largest_tx_share':float(g.tx_sign.value_counts(normalize=True).iloc[0])}
    return result

def hourly(d):
    d = d.copy()
    d['tx_grid'] = d.tx_loc.map(grid4)
    d['rx_grid'] = d.rx_loc.map(grid4)
    # Broad measurement validity screen, not a mode reception threshold.
    valid = (d.tx_grid.notna() & d.rx_grid.notna() & d.snr2500.between(-60,100)
             & d.frequency.between(7e6,7.3e6) & d['mode'].isin(['WSPR','FT8','FT4','CW'])
             & d.tx_sign.notna() & d.rx_sign.notna()
             & d.tx_sign.astype(str).str.len().ge(2) & d.rx_sign.astype(str).str.len().ge(2))
    d = d[valid].drop_duplicates(['source','time','tx_sign','rx_sign','frequency','mode']).copy()
    d['hour'] = d.time.dt.floor('h')
    d.loc[~d.power_dbm.between(0,60), 'power_dbm'] = np.nan
    keys = ['source','mode','hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']
    d = d.groupby(keys, dropna=False, observed=True).agg(snr2500=('snr2500','median'),
        reports=('snr2500','size')).reset_index()
    d['row_id'] = np.arange(len(d))
    return d

SOLAR = ['kp_previous','ap_previous','ap_trailing24','sfi_previous_day','Ap_previous_day']

def features(d,solar_start='2022'):
    d = d.copy()
    coords = {g: centre(g) for g in set(d.tx_grid) | set(d.rx_grid)}
    for role in ['tx','rx']:
        d[f'{role}_lat'] = d[f'{role}_grid'].map(lambda g: coords[g][0])
        d[f'{role}_lon'] = d[f'{role}_grid'].map(lambda g: coords[g][1])
    a,b = np.deg2rad(d.tx_lat.to_numpy()), np.deg2rad(d.rx_lat.to_numpy())
    la,lb = np.deg2rad(d.tx_lon.to_numpy()),np.deg2rad(d.rx_lon.to_numpy())
    dot = np.clip(np.sin(a)*np.sin(b)+np.cos(a)*np.cos(b)*np.cos(lb-la), -1, 1)
    omega = np.arccos(dot)
    d['distance_km'] = 6371.0088*omega
    # Unit vectors allow shortest-great-circle interpolation across the dateline.
    va = np.column_stack([np.cos(a)*np.cos(la),np.cos(a)*np.sin(la),np.sin(a)])
    vb = np.column_stack([np.cos(b)*np.cos(lb),np.cos(b)*np.sin(lb),np.sin(b)])
    mid = va+vb
    d['mid_lat'] = np.rad2deg(np.arctan2(mid[:,2],np.hypot(mid[:,0],mid[:,1])))
    d['mid_lon'] = np.rad2deg(np.arctan2(mid[:,1],mid[:,0]))
    doy = d.hour.dt.day_of_year.to_numpy()
    hour = d.hour.dt.hour.to_numpy()+.5
    phase = 2*np.pi*(doy-1)/365.2425
    for label, x in [('season',phase),('utc',2*np.pi*hour/24)]:
        d[label+'_sin'],d[label+'_cos'] = np.sin(x),np.cos(x)
    # Approximate solar declination / equation of time; features, not ray tracing.
    dec = (.006918-.399912*np.cos(phase)+.070257*np.sin(phase)
        -.006758*np.cos(2*phase)+.000907*np.sin(2*phase)
        -.002697*np.cos(3*phase)+.00148*np.sin(3*phase))
    eqtime = 229.18*(.000075+.001868*np.cos(phase)-.032077*np.sin(phase)
        -.014615*np.cos(2*phase)-.040849*np.sin(2*phase))
    for role in ['tx','rx','mid']:
        lat = np.deg2rad(d[role+'_lat'].to_numpy())
        ha = np.deg2rad(15*(hour+d[role+'_lon'].to_numpy()/15+eqtime/60-12))
        d[role+'_sun'] = np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(ha)
    for role in ['tx','rx']:
        lon=np.deg2rad(d[role+'_lon'].to_numpy())
        d[role+'_lon_sin'],d[role+'_lon_cos'] = np.sin(lon),np.cos(lon)
    blocks,daily = solar_table(solar_start)
    d = pd.merge_asof(d.sort_values('hour',kind='stable'),blocks.reset_index(),left_on='hour',right_on='available_boundary', direction='backward')
    assert (d.available_boundary <= d.hour).all()
    d['date'] = d.hour.dt.floor('D')
    d = d.join(daily,on='date',validate='many_to_one')
    d['distance_bin'] = (d.distance_km//1000).astype(int)
    d['utc_hour'] = d.hour.dt.hour
    d['mid_daylight'] = (d.mid_sun>0).astype(int)
    d['quarter'] = d.hour.dt.quarter
    return d.sort_values(['source','mode','hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm'],kind='stable').reset_index(drop=True)

BASE = ['tx_lat','rx_lat','tx_lon_sin','tx_lon_cos','rx_lon_sin','rx_lon_cos',
        'distance_km','mid_lat','season_sin','season_cos','utc_sin','utc_cos',
        'tx_sun','rx_sun','mid_sun','power_dbm']

class Historical:
    """Training-only, power-adjusted medians with fallbacks for unseen paths.

    Known reported power is removed before lookup and added back at prediction.
    This gives the lookup the same power information available to the ML model.
    Unknown powers contribute no correction; source/mode models remain separate.
    """
    def __init__(self,power_adjust=False):
        self.power_adjust=power_adjust

    def fit(self, d):
        d=d.copy()
        if self.power_adjust and 'power_dbm' in d:
            d['snr2500']=d.snr2500-d.power_dbm.fillna(0)
        self.global_ = d.snr2500.median()
        self.tables=[]
        for keys,n in [(['distance_bin','mid_daylight','quarter','utc_hour'],10),
                       (['tx_grid','rx_grid'],5),(['tx_grid','rx_grid','utc_hour'],3)]:
            stats=d.groupby(keys).snr2500.agg(['median','count'])
            self.tables.append((keys,stats.loc[stats['count']>=n,'median']))
        return self
    def predict(self,d):
        pred=pd.Series(self.global_,index=d.index)
        for keys,table in self.tables:
            idx=pd.MultiIndex.from_frame(d[keys])
            vals=pd.Series(table.reindex(idx).to_numpy(),index=d.index)
            pred=vals.fillna(pred)
        power=d.power_dbm.fillna(0).to_numpy() if self.power_adjust and 'power_dbm' in d else 0
        return pred.to_numpy()+power

def metric(y,p):
    return {'mae_db':float(mean_absolute_error(y,p)), 'rmse_db':float(math.sqrt(mean_squared_error(y,p)))}

def train_evaluate(d):
    results=[];predictions=[]
    # Date split is fixed before seeing performance; future means our sampled 2025 days.
    for (source,mode),g in d.groupby(['source','mode']):
        if source=='PSKR_LIVE':
            continue  # schema/quality probe only: recent capped API sample is not representative
        early=g[g.hour < pd.Timestamp('2025-01-01',tz='UTC')].copy()
        later=g[g.hour >= pd.Timestamp('2025-01-01',tz='UTC')].copy()
        if len(early)<1000 or len(later)<200:
            results.append({'source':source,'mode':mode,'status':'insufficient_rows',
                            'early_rows':len(early),'later_rows':len(later)})
            continue
        withheld={c for c in set(g.tx_sign)|set(g.rx_sign) if held_station(c)}
        strict=early[~early.tx_sign.isin(withheld) & ~early.rx_sign.isin(withheld)]
        strict_calls=set(strict.tx_sign)|set(strict.rx_sign)
        assert not strict_calls & withheld
        unseen_tx=~later.tx_sign.isin(strict_calls)
        unseen_rx=~later.rx_sign.isin(strict_calls)
        folds=[('later_period',early,later),
               ('later_unseen_endpoint',strict,later[unseen_tx|unseen_rx]),
               ('later_both_endpoints_unseen',strict,later[unseen_tx&unseen_rx])]
        fitted={}
        for fold,tr,te in folds:
            if len(tr)<1000 or len(te)<100:
                results.append({'source':source,'mode':mode,'fold':fold,'status':'insufficient_rows','train_rows':len(tr),'test_rows':len(te)})
                continue
            cachekey='all' if fold=='later_period' else 'strict'
            if cachekey not in fitted:
                models={'historical':Historical().fit(tr),
                        'historical_power':Historical(power_adjust=True).fit(tr)}
                for label,cols in [('geometry',BASE),('geometry_solar',BASE+SOLAR)]:
                    cols=[c for c in cols if tr[c].notna().any() and tr[c].nunique()>1]
                    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=150,
                        max_leaf_nodes=23,min_samples_leaf=50,l2_regularization=5,
                        learning_rate=.08,early_stopping=False,random_state=40)
                    model.fit(tr[cols],tr.snr2500)
                    models[label]=model
                fitted[cachekey]=models
            models=fitted[cachekey]
            pred=te[['row_id','source','mode','hour','tx_sign','rx_sign','snr2500']].copy()
            pred['fold']=fold
            for label,model in models.items():
                cols=list(model.feature_names_in_) if not label.startswith('historical') else None
                p=model.predict(te if label.startswith('historical') else te[cols])
                m=metric(te.snr2500,p)
                # Each receiver-day gets equal total weight for a second error view.
                group=te.rx_sign+'|'+te.hour.dt.strftime('%Y-%m-%d')
                weight=1/group.map(group.value_counts()).to_numpy()
                m['receiver_day_balanced_mae_db']=float(np.average(abs(te.snr2500.to_numpy()-p),weights=weight))
                results.append({'source':source,'mode':mode,'fold':fold,'model':label,
                    'status':'evaluated','train_rows':len(tr),'test_rows':len(te),
                    'test_days':te.hour.dt.date.nunique(),'train_stations':len(set(tr.tx_sign)|set(tr.rx_sign)),
                    'features':cols,
                    **m})
                pred[label]=p
            predictions.append(pred)
            if fold=='later_period':
                lohi=[]
                for q in [.1,.9]:
                    cols=[c for c in BASE+SOLAR if tr[c].notna().any() and tr[c].nunique()>1]
                    model=HistGradientBoostingRegressor(loss='quantile',quantile=q,
                        max_iter=150,max_leaf_nodes=23,min_samples_leaf=50,
                        l2_regularization=5,early_stopping=False,random_state=40)
                    model.fit(tr[cols],tr.snr2500)
                    lohi.append(model.predict(te[cols]))
                lo,hi=lohi
                results.append({'source':source,'mode':mode,'fold':fold,'model':'geometry_solar_80pct_interval',
                    'status':'evaluated','train_rows':len(tr),'test_rows':len(te),
                    'coverage':float(np.mean((te.snr2500>=lo)&(te.snr2500<=hi))),
                    'mean_width_db':float(np.mean(hi-lo)), 'crossed_intervals':int((hi<lo).sum())})
                pred['q10'],pred['q90']=lo,hi
                joblib.dump({'model':models['geometry_solar'],'features':list(models['geometry_solar'].feature_names_in_),
                    'target':'SNR2500 conditional on reported reception; source-specific station population',
                    'source':source,'mode':mode}, DATA/f'model-{source}-{mode}.joblib')
            print(source,mode,fold,len(tr),len(te),flush=True)
    if predictions:
        pd.concat(predictions).to_parquet(DATA/'predictions.parquet',index=False)
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2,default=int))
    return results

def main():
    (OUT/'run.json').write_text(json.dumps({'status':'running'}))
    parts=[load_wspr(),load_pskr_current(),load_rbn_raw()]
    # Optional CEDAR rows are prepared by collect_harc.py.
    for p in sorted(DATA.glob('harc-*.parquet')):
        h=pd.read_parquet(p)
        h['snr_native']=h.sn
        parts.append(h)
    d=pd.concat([p for p in parts if not p.empty],ignore_index=True)
    (OUT/'quality.json').write_text(json.dumps(audit(d),indent=2,default=int))
    # Only ordinary WSPR from WSPRNET is eligible; OTHER stays in the audit.
    d=features(hourly(d))
    d.to_parquet(DATA/'hourly.parquet',index=False)
    with sqlite3.connect(DATA/'gocap.sqlite') as con:
        out=d.copy()
        for col in ['hour','date','available_boundary']:
            out[col]=out[col].astype(str)
        out.to_sql('positive_hourly',con,if_exists='replace',index=False)
        con.execute('CREATE INDEX IF NOT EXISTS path_time ON positive_hourly(tx_grid,rx_grid,hour)')
    con.close()
    (OUT/'feature_audit.json').write_text(json.dumps({'hourly_rows':len(d),
        'rows_by_source_mode':d.groupby(['source','mode']).size().rename('rows').reset_index().to_dict('records'),
        'solar_missing':d[SOLAR].isna().sum().to_dict(),
        'time_split':'train before 2025-01-01 UTC; test sampled 2025 days',
        'solar_timing':'previous completed 3h block; trailing 8 blocks; previous UTC-day SFI/Ap; final indices, not operational vintages'},indent=2,default=int))
    with threadpool_limits(limits=4):
        metrics=train_evaluate(d)
    (OUT/'run.json').write_text(json.dumps({'status':'complete',
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'scikit_learn':__import__('sklearn').__version__,'random_state':40,
        'hourly_sha256':hashlib.sha256((DATA/'hourly.parquet').read_bytes()).hexdigest(),
        'predictions_sha256':hashlib.sha256((DATA/'predictions.parquet').read_bytes()).hexdigest(),
        'metrics_sha256':hashlib.sha256((OUT/'metrics.json').read_bytes()).hexdigest()},indent=2))
    print(json.dumps(metrics,indent=2,default=int),flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception:
        (OUT/'run.json').write_text(json.dumps({'status':'failed'}))
        raise
