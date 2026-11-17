"""Freeze validation selections, then score every candidate on the same 2026 hours."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .five_data import CACHE, DATES, OUT, ROOT, sha
from .five_prepare import DRAWS, DEFAULT
from .five_models import Correction, SPECS, STATIC, adjacent

ART = ROOT/'artifacts/gocap-40m-five.joblib'


def load(split):
    return pd.concat([pd.read_parquet(CACHE/'features'/f'{date}.parquet') for date in DATES[split]],
                     ignore_index=True)


def daily(d,p):
    a,b = adjacent(d)
    y = d.snr2500.to_numpy()
    err = pd.DataFrame({'date':d.date_key, 'hour':d.hour, 'case':d.case, 'ae':abs(p-y)})
    ae = err.groupby(['date','hour']).ae.mean().groupby('date').mean()
    path = err.groupby(['date','case']).ae.mean().groupby('date').mean()
    diff = abs((p[b]-p[a])-(y[b]-y[a]))
    delta = pd.Series(diff).groupby(d.date_key.to_numpy()[b]).mean()
    return pd.DataFrame({'hour_mae':ae,'path_mae':path,'change_mae':delta})


def train():
    d,v = load('train'),load('validation')
    models = {}
    scores = {}
    half = {}
    for name in ['default',*SPECS]:
        pred = v.base.to_numpy()
        if name in SPECS:
            target,weight,blocks = SPECS[name]
            models[name] = Correction(blocks).fit(d,target,weight)
            pred = models[name].predict(v)
        scores[name] = daily(v,pred).mean().to_dict()
        half[name] = float(np.quantile(abs(pred-v.snr2500),.8,method='higher'))
    # Choose separately for arbitrary scenarios and for situations with lagged inputs.
    best = {group:min(names,key=lambda n:scores[n]['hour_mae'])
            for group,names in [('scenario',STATIC),('nowcast',['default',*SPECS])]}
    bundle = dict(models=models,selected=best,half_width=half,base_sha256=sha(DEFAULT))
    joblib.dump(bundle,ART)
    info = dict(splits=DATES,selected=best,validation=scores,sha256=sha(ART),base_sha256=sha(DEFAULT),
                rows=len(d),validation_rows=len(v),feature_hashes={date:sha(CACHE/'features'/f'{date}.parquet')
                for dates in DATES.values() for date in dates})
    (OUT/'selection.json').write_text(json.dumps(info,indent=2))
    print('Frozen selections',best,flush=True)


def interval(vec):
    v = np.asarray(vec)
    boot = np.random.default_rng(405).choice(v,size=(10000,len(v))).mean(axis=1)
    return [float(v.mean()),*np.quantile(boot,[.025,.975]).tolist()]


def test():
    info = json.loads((OUT/'selection.json').read_text())
    assert sha(ART)==info['sha256']
    d = load('test')
    for date in DATES['test']:
        assert sha(CACHE/'features'/f'{date}.parquet')==info['feature_hashes'][date]
    bundle = joblib.load(ART)
    preds = {'default':d.base.to_numpy()}
    preds.update({n:m.predict(d) for n,m in bundle['models'].items()})
    from .hourly_models import predict_profiles
    for name,suffix in [('march','shared-march'),('march_tuned','shared-march-tuned')]:
        other = joblib.load(ROOT/f'artifacts/gocap-40m-{suffix}.joblib')
        preds[name] = predict_profiles(d,other['models'],other['profile_selection'])
    tables = {n:daily(d,p) for n,p in preds.items()}
    values = pd.concat(tables,names=['model','date'])
    values.to_csv(OUT/'daily_metrics.csv')
    summary = values.groupby('model').mean()
    summary['coverage80'] = pd.Series({n:(abs(preds[n]-d.snr2500)<=w).mean()
                                      for n,w in bundle['half_width'].items()})
    summary.to_csv(OUT/'metrics.csv')
    groups = {'all':np.ones(len(d),bool),'short':d.distance_km.le(1000),'sonde_available':d.sonde_available,
              'donors_available':d.donor_count.ge(3),'singleton':d.reports.eq(1),'multi_report':d.reports.ge(2)}
    sub = []
    for key,mask in groups.items():
        for n,p in preds.items():
            sub.append(dict(group=key,model=n,rows=int(mask.sum()),pooled_mae_db=abs(p[mask]-d.snr2500[mask]).mean()))
    pd.DataFrame(sub).to_csv(OUT/'subgroups.csv',index=False)
    pairs = [('hour_median','one_report'),('change','hour_median'),('f2','hour_median'),
             ('f2_sonde','f2'),('donors','hour_median'),('voacap_shape','hour_median'),
             ('combined_donors','static_combined'),('combined_live','combined_donors')]
    pairs += [(v,'default') for v in bundle['selected'].values()]
    ci = {a+' minus '+b:{k:interval(tables[a][k]-tables[b][k]) for k in tables[a]} for a,b in pairs}
    ref = d.base.to_numpy()
    draws = [abs(ref-d[col]).groupby(d.hour).mean().mean() for col in DRAWS]
    audit = dict(test_rows=len(d),cases=d.case.nunique(),positive_reports=int(d.reports.sum()),
                 date_hours=d.hour.nunique(),adjacent_pairs=len(adjacent(d)[0]),
                 multiple_report_hours=int(d.reports.gt(1).sum()),median_reports=float(d.reports.median()),
                 single_draw_mae_mean=float(np.mean(draws)),single_draw_mae_range=[min(draws),max(draws)],
                 ci=ci,sonde_supported=int(d.sonde_available.sum()),donor_supported=int(d.donor_count.ge(3).sum()))
    (OUT/'summary.json').write_text(json.dumps(audit,indent=2))
    keep = ['case','hour','tx_grid','rx_grid','power_dbm','distance_km','snr2500','reports','within_sd','donor_count','sonde_available']
    d[keep].assign(**preds).to_parquet(OUT/'test_predictions.parquet',index=False)
    print(summary.round(3))


if __name__=='__main__':
    with threadpool_limits(limits=3):
        train()
        test()
