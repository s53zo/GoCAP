import json
import numpy as np,pandas as pd
from ml40.voacap_protocol import BASE,OUT
from ml40.voacap_engine import Engine,make_deck
c=pd.read_csv(OUT/'cases_frozen.csv');expected=pd.read_parquet(BASE/'case_features.parquet')
parts=[]
for date,cases in c.groupby('date_key'):
    raw=pd.read_json(BASE/'raw'/f'wspr-{date}.jsonl',lines=True)
    raw=raw[raw.code.eq(1)].copy();raw['hour']=pd.to_datetime(raw.time,utc=True).dt.floor('h')
    raw['tx_grid']=raw.tx_loc.str.upper().str[:4];raw['rx_grid']=raw.rx_loc.str.upper().str[:4]
    raw=raw[raw.frequency.between(7e6,7.3e6)&raw.snr.between(-60,100)]
    raw=raw.drop_duplicates(['time','tx_sign','rx_sign','frequency'])
    raw=raw.rename(columns={'power':'power_dbm'})
    z=raw.merge(cases[['case_id','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']],on=['tx_sign','rx_sign','tx_grid','rx_grid','power_dbm'],validate='many_to_one')
    parts.append(z.groupby(['case_id','hour']).snr.agg(actual='median',count='size').reset_index())
a=pd.concat(parts);b=a.merge(expected,on=['case_id','hour'],validate='one_to_one');assert len(b)==len(expected)==len(a)
assert np.allclose(b.actual,b.snr2500) and np.allclose(b['count'],b.reports)
fit=pd.read_parquet('data/ml40/fit_features.parquet',columns=['hour']);assert fit.hour.max()<pd.Timestamp('2025-01-01',tz='UTC')
with Engine() as e:
    _,p=e.run(make_deck(46.5,15,41.5,-73,2025,4,100,1));_,q=e.run(make_deck(46.5,15,41.5,-73,2025,4,100,10))
assert all(q[h]['SNR']-p[h]['SNR']==10 for h in range(24))
s=json.loads((OUT/'summary.json').read_text());per=pd.read_csv(OUT/'case_results.csv');rng=np.random.default_rng(20261117)
g=per.groupby('date_key').gocap_minus_voacap_mae_db.agg(['sum','count']).sort_index();ix=rng.choice(len(g),(10000,len(g)),replace=True)
boot=g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
assert np.allclose(np.quantile(boot,[.025,.975]),s['date_cluster_bootstrap_95pct_db'])
v=json.loads((OUT/'verification.json').read_text());v.update({'raw_reports_independently_reaggregated':int(b['count'].sum()),'all_raw_hourly_medians_match':True,'fitting_features_latest_utc':fit.hour.max().isoformat(),'native_10db_power_increment_verified_hours':24,'bootstrap_independently_recomputed':True})
(OUT/'verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
