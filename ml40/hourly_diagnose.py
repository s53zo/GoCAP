"""Validation-only diagnosis of absolute hourly SNR errors; no final-test access."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import DATA,ROOT,RESULTS
from .v3_features import enrich
from .test500 import digest

OUT=RESULTS/'hourly';OUT.mkdir(exist_ok=True)

def metric(g):
    e=g.prediction-g.snr2500
    return {'rows':len(g),'days':g.date_key.nunique(),'mae_db':float(abs(e).mean()),
        'rmse_db':float(np.sqrt(np.mean(e**2))),'bias_db':float(e.mean()),
        'median_error_db':float(e.median()),'p90_absolute_error_db':float(abs(e).quantile(.9))}

def diagnose():
    d=pd.read_parquet(DATA/'fit_features.parquet');train=d[d.split.eq('train')]
    val=enrich(d[d.split.eq('validation')].reset_index(drop=True))
    model=joblib.load(ROOT/'artifacts/gocap-40m.joblib')['model']
    with threadpool_limits(limits=4):val['prediction']=model.predict(val,station_correction=False)
    val['utc_hour']=val.hour.dt.hour
    for role in ['tx','rx']:
        val[role+'_local_hour']=np.floor((val.hour.dt.hour+.5+val[role+'_lon']/15)%24).astype(int)
    val['distance_group']=pd.cut(val.distance_km,[-1,1000,3000,6000,20001]).astype(str)
    val['season']=val.hour.dt.quarter.astype(str)
    val['kp_group']=pd.cut(val.kp_previous,[-.01,2,4,6,9]).astype(str)
    val['sfi_group']=pd.cut(val.sfi_previous_day,[0,80,120,160,200,1000]).astype(str)
    val['power_group']=pd.cut(val.power_dbm,[-1,20,30,37,50,60]).astype(str)
    known=set(train.tx_sign)|set(train.rx_sign)
    val['station_support']=np.where(val.tx_sign.isin(known)&val.rx_sign.isin(known),'both_known',
        np.where(~val.tx_sign.isin(known)&~val.rx_sign.isin(known),'both_unseen','one_unseen'))
    rows=[]
    for dimension in ['utc_hour','tx_local_hour','rx_local_hour','distance_group','season','kp_group','sfi_group','power_group','station_support']:
        for (mode,label),g in val.groupby(['mode',dimension],sort=True):
            rows.append({'mode':mode,'dimension':dimension,'segment':str(label)}|metric(g))
    pd.DataFrame(rows).to_csv(OUT/'validation_diagnostics.csv',index=False)
    (DATA/'hourly').mkdir(exist_ok=True);val.to_parquet(DATA/'hourly'/'diagnostic_validation.parquet',index=False)
    modes={mode:metric(g) for mode,g in val.groupby('mode')}
    (OUT/'diagnosis.json').write_text(json.dumps({'scope':'original validation dates only; no new test scores inspected',
        'reference_sha256':digest(ROOT/'artifacts/gocap-40m.joblib'),'by_mode':modes},indent=2))
    print(json.dumps(modes,indent=2))
    tab=pd.DataFrame(rows);print(tab[(tab['mode']=='WSPR')&tab.rows.ge(1000)].sort_values('bias_db').iloc[np.r_[0:6,-6:0]].to_string(index=False))
if __name__=='__main__':diagnose()
