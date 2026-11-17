"""Transfer the two promising corrections to the stronger frozen March regressor."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from experiment import features
from . import four_data,four_drap
from .four_prepare import CASE,DEFAULT,prepare
from .four_run import ROOT,CACHE,OUT,ART,load,held,geo,fine,sha,weights,optimise
from .hourly_models import predict_profiles
from .prepare import annotate
from .v3_features import enrich
from .four_models import component_predictions

PARENT=ROOT/'artifacts/gocap-40m-shared-march-tuned.joblib'
TRANSFER=ROOT/'artifacts/gocap-40m-four-march.joblib'
DATES=[f'2026-{m:02d}-13' for m in range(3,9)]


def parent(d,bundle):
    cols=CASE+['hour']
    parts=[]
    for _,g in d.groupby(d.hour.dt.date):
        frame=g[cols].assign(source='WSPRNET',mode='WSPR',decoder_context='ordinary_code1')
        f=enrich(annotate(features(frame,solar_start='2009')))
        f['march']=predict_profiles(f,bundle['models'],bundle['profile_selection'])
        parts.append(f[cols+['march']])
    return d[cols].merge(pd.concat(parts),on=cols,validate='one_to_one').march.to_numpy()


def basis(d,pack):
    a=d[['abs_mean','abs_max']].to_numpy(dtype=float)
    return np.column_stack([geo(d,'centre',{},pack['reg']),np.nan_to_num(a).clip(0,30)@pack['abs_coef']])


def train():
    d=load('hours','train')
    old=pd.read_parquet(ROOT/'data/ml40/march40/expanded_train_features.parquet',columns=['date_key'])
    d=d[~d.rx_sign.map(held)&~d.hour.dt.strftime('%Y-%m-%d').isin(old.date_key.unique())].reset_index(drop=True)
    v=load('hours','validation')
    v=v[~v.rx_sign.map(held)].reset_index(drop=True)
    b=joblib.load(PARENT)
    r=d.snr2500.to_numpy()-parent(d,b)
    use=(fine(d.tx_loc)&fine(d.rx_loc)).to_numpy()
    pack={'reg':Ridge(alpha=300).fit(geo(d[use],'centre',{}),r[use]),'parent_sha256':sha(PARENT)}
    good=d.abs_mean.notna()&d.abs_max.notna()
    x=d.loc[good,['abs_mean','abs_max']].to_numpy().clip(0,30)
    pack['abs_coef']=lsq_linear(np.vstack([x,10*np.eye(2)]),np.r_[r[good],[0.,0.]],bounds=(-3.,0.)).x
    x=basis(v,pack)
    y=v.snr2500.to_numpy()-parent(v,b)
    w=weights(v)
    pack['scenario']=optimise(x[:,:1],y,w)
    pack['nowcast']=optimise(x,y,w)
    joblib.dump(pack,TRANSFER)
    info=dict(model_sha256=sha(TRANSFER),parent_sha256=sha(PARENT),train_rows=len(d),dates=DATES,
              scenario=pack['scenario'].tolist(),nowcast=pack['nowcast'].tolist(),
              validation={k:float(w@abs(y-x[:,:len(pack[k])]@pack[k])) for k in ['scenario','nowcast']})
    (OUT/'transfer_selection.json').write_text(json.dumps(info,indent=2))
    print(info,flush=True)


def test():
    info=json.loads((OUT/'transfer_selection.json').read_text())
    assert sha(TRANSFER)==info['model_sha256'] and sha(PARENT)==info['parent_sha256']
    with ThreadPoolExecutor(max_workers=3) as pool:
        radio=pool.submit(four_data.collect,DATES,'transfer_sources.json')
        met=[pool.submit(four_drap.collect,date) for date in DATES]
        radio.result()
        for task in met:
            task.result()
    base=joblib.load(DEFAULT)
    b=joblib.load(PARENT)
    pack=joblib.load(TRANSFER)
    original=joblib.load(ART)
    rows=[]
    for date in DATES:
        prepare(date,base)
        d=pd.read_parquet(CACHE/f'hours-{date}.parquet')
        z=component_predictions(d,original)
        x=basis(d,pack)
        p=parent(d,b)
        predictions={'default':d.base,'march_tuned':p,
            'previous_combination':d.base+sum(z[k]*v for k,v in original['weights']['live_combined'].items()),
            'march_scenario':p+x[:,:1]@pack['scenario'],'march_nowcast':p+x@pack['nowcast']}
        for name,val in predictions.items():
            e=abs(val-d.snr2500)
            rows.append(dict(date=date,model=name,hour_mae=e.groupby(d.hour).mean().mean(),
                             rows=len(d),reports=int(d.reports.sum())))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'transfer_daily.csv',index=False)
    z=out.pivot(index='date',columns='model',values='hour_mae')
    dif=z.march_nowcast-z.march_tuned
    boot=np.random.default_rng(42).choice(dif,(10000,len(dif))).mean(1)
    result=dict(model_sha256=sha(TRANSFER),test_rows=int(out[out.model=='default'].rows.sum()),
                scores=z.mean().to_dict(),nowcast_minus_parent_ci=np.quantile(boot,[.025,.975]).tolist())
    (OUT/'transfer_summary.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['train','test'])
    with threadpool_limits(limits=3):
        {'train':train,'test':test}[p.parse_args().phase]()
