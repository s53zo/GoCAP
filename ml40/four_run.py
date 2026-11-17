"""Fit component models, freeze their validation weights and compare on later dates."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .four_data import CACHE,OUT,ROOT,DATES,sha
from .four_models import held, fine, geo, fit, component_predictions, SPECS
from .five_prepare import DEFAULT

ART=ROOT/'artifacts/gocap-40m-four.joblib'


def load(kind,split):
    return pd.concat([pd.read_parquet(CACHE/f'{kind}-{day}.parquet') for day in DATES[split]],ignore_index=True)


def weights(d):
    size=d.groupby('hour').hour.transform('size').to_numpy()
    return 1/(size*d.hour.nunique())


def optimise(matrix,error,w):
    coeff=np.zeros(matrix.shape[1])
    for _ in range(4):
        for j in range(len(coeff)):
            without=matrix@coeff-error-coeff[j]*matrix[:,j]
            losses=[np.dot(w,abs(without+v*matrix[:,j])) for v in [0,.25,.5,.75,1]]
            coeff[j]=np.argmin(losses)/4
    return coeff


def train():
    d=load('hours','train')
    d=d[~d.rx_sign.map(held)].reset_index(drop=True)
    steps=load('allsteps','train')
    steps=steps[~steps.rx_sign.map(held)]
    val=load('hours','validation')
    val=val[~val.rx_sign.map(held)].reset_index(drop=True)
    pack=fit(d,steps)
    x=component_predictions(val,pack)
    residual=(val.snr2500-val.base).to_numpy()
    w=weights(val)
    scores={}
    pack['weights']={}
    for name,cols in SPECS.items():
        matrix=x[cols].to_numpy()
        coef=optimise(matrix,residual,w)
        scores[name]=float(w@abs(matrix@coef-residual))
        pack['weights'][name]=dict(zip(cols,coef))
    pack['selected']={
        'scenario':min(['default','noise_scenario','power_steps','grid6_marginal','scenario_combined'],key=scores.get),
        'nowcast':min([k for k in SPECS if k!='grid4_control'],key=scores.get)}
    pack['base_sha256']=sha(DEFAULT)
    joblib.dump(pack,ART)
    info=dict(selected=pack['selected'],validation=scores,weights=pack['weights'],model_sha256=sha(ART),
              base_sha256=sha(DEFAULT),dates=DATES,train_rows=len(d),validation_rows=len(val),power_steps=len(steps),
              power_slope=pack['slope'],noise_slopes=[pack['typ_coef'],pack['live_coef']],
              absorption_coefficients=pack['abs_coef'].tolist(),training_receivers=d.rx_sign.nunique(),
              six_character_training=int((fine(d.tx_loc)&fine(d.rx_loc)).sum()))
    (OUT/'selection.json').write_text(json.dumps(info,indent=2))
    print('Frozen',pack['selected'],flush=True)


def cluster_ci(series):
    by=series.groupby(series.index.map(lambda t: {pd.Timestamp('2025-11-12').date():pd.Timestamp('2025-11-11').date(),
        pd.Timestamp('2026-02-02').date():pd.Timestamp('2026-02-01').date()}.get(t.date(),t.date()))).agg(['sum','size'])
    idx=np.random.default_rng(40).integers(0,len(by),size=(10000,len(by)))
    draws=by['sum'].to_numpy()[idx].sum(1)/by['size'].to_numpy()[idx].sum(1)
    return np.quantile(draws,[.025,.975]).tolist()


def evaluate():
    info=json.loads((OUT/'selection.json').read_text())
    assert info['model_sha256']==sha(ART)
    pack=joblib.load(ART)
    d=load('hours','test')
    x=component_predictions(d,pack)
    results={}
    for name,cols in SPECS.items():
        results[name]=d.base.to_numpy()+sum((x[c]*v for c,v in pack['weights'][name].items()),start=0)
        if cols:
            results[name+'_full']=d.base+x[cols].sum(axis=1)
    results['grid6_diagnostic']=d.base+geo(d,'fine',{},pack['fine'])
    vals=pd.DataFrame(results,index=d.index)
    err=vals.sub(d.snr2500,axis=0).abs()
    by=err.groupby(d.hour).mean()
    by.to_csv(OUT/'hourly_mae.csv')
    rows=[]
    for k in by:
        diff=by[k]-by.default
        lo,hi=cluster_ci(diff)
        rows.append(dict(model=k,hour_mae=by[k].mean(),pooled_mae=err[k].mean(),delta=diff.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    # Receiver novelty is assessed for both the new fit and the frozen WSPR base.
    prior=pd.read_parquet(ROOT/'data/ml40/hourly/original_train_features.parquet',columns=['rx_sign','mode'])
    base_rx=set(prior[prior['mode']=='WSPR'].rx_sign)
    trained=load('hours','train').rx_sign
    train_rx=set(trained[~trained.map(held)])
    unseen=~d.rx_sign.isin(base_rx|train_rx)
    flag=d[['hour','status']].drop_duplicates().set_index('hour').status.eq('flare')
    recovery=(d.hour-pd.Timedelta(hours=1)).map(flag).eq(True)|(d.hour-pd.Timedelta(hours=2)).map(flag).eq(True)
    both=fine(d.tx_loc)&fine(d.rx_loc)
    sets={'all':pd.Series(True,index=d.index),'receiver_reserved':d.rx_sign.map(held),'unseen_both':unseen,
          'unseen_both_with_noise':unseen&d.noise_now.notna(),'noise_available':d.noise_now.notna(),
          'short':d.distance_km<=1000,'short_both_six':(d.distance_km<=1000)&both,'both_six':both,
          'flare':d.status.eq('flare'),'quiet':d.status.eq('quiet'),'recovery':recovery&d.status.ne('flare')}
    sub=[]
    for label,mask in sets.items():
        scores=err[mask].groupby(d.hour[mask]).mean().mean()
        for k,v in scores.items():
            sub.append(dict(subset=label,model=k,rows=int(mask.sum()),stations=d[mask].rx_sign.nunique(),hour_mae=v))
    pd.DataFrame(sub).to_csv(OUT/'subgroups.csv',index=False)
    keep=['hour','rx_sign','tx_sign','rx_grid','tx_grid','power_dbm','snr2500','reports','status','distance_km','abs_mean','abs_max','noise_now']
    values=pd.concat([d[keep],vals],axis=1)
    values['fresh_rx']=unseen
    values.to_parquet(CACHE/'test_predictions.parquet',index=False)
    print(pd.DataFrame(rows)[['model','hour_mae','delta']].round(4).to_string(index=False))
    return pack


if __name__=='__main__':
    with threadpool_limits(limits=3):
        train()
        evaluate()
