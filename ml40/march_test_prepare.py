"""Freeze500 March2025 paths by identifiers before prediction."""
import json,datetime as dt,hashlib
from collections import deque
import numpy as np,pandas as pd
from .march_test_plan import ROOT,DATA,BASE,OUT,DATES,GROUPS,KEYS,plan,sha

def select(d,cfg):
    c=d.groupby(KEYS,sort=True).agg(observed_hours=('hour','nunique'),distance_km=('distance_km','first')).reset_index()
    c=c[c.observed_hours.ge(6)&c.power_dbm.between(20,50)&c.distance_km.between(100,15000)].copy()
    c['distance_group']=pd.cut(c.distance_km,[99.999,1000,3000,6000,15000],labels=GROUPS).astype(str)
    c['case_id']=[hashlib.sha256((cfg['seed']+json.dumps(list(k),default=str)).encode()).hexdigest()[:20] for k in c[KEYS].itertuples(index=False,name=None)]
    rows=[];used=set()
    for group in GROUPS:
        part=c[c.distance_group.eq(group)].sort_values('case_id');queues=[deque(g.to_dict('records')) for _,g in part.groupby('date_key',sort=True)];count=0
        while count<125:
            progress=False
            for q in queues:
                while q:
                    row=q.popleft();path=(row['tx_grid'],row['rx_grid'])
                    if path in used:continue
                    used.add(path);rows.append(row);count+=1;progress=True;break
                if count==125:break
            if not progress:raise ValueError(f'Insufficient eligible paths in{group}:{count}/125')
    return pd.DataFrame(rows).sort_values('case_id').reset_index(drop=True)

def prepare():
    cfg=plan();training=pd.read_parquet(DATA/'march40/expanded_train_features.parquet',columns=['hour']);assert training.hour.dt.year.le(2024).all()
    from .march_plan import FRESH_TEST
    assert not set(DATES)&set(FRESH_TEST)
    d=pd.concat([pd.read_parquet(BASE/'processed'/f'wspr-{date}.parquet') for date in DATES],ignore_index=True)
    assert d['decoder_context'].eq('ordinary_code1').all() and d['mode'].eq('WSPR').all()
    c=select(d,cfg);c['year']=2025;c['month']=3
    solar=pd.read_csv(DATA/'voacap1000/silso_smoothed_v2.txt',sep=r'\s+',header=None,names=['year','month','decimal_year','ssn','std','count','provisional'])
    c=c.merge(solar[['year','month','ssn']],on=['year','month'],validate='many_to_one');c['reported_power_watts']=10**((c.power_dbm-30)/10)
    obs=d.merge(c[KEYS+['case_id']],on=KEYS,how='inner',validate='many_to_one').sort_values(['case_id','hour']).reset_index(drop=True)
    assert len(c)==500 and not obs.duplicated(['case_id','hour']).any()
    frozen={'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'case_ids':c.case_id.tolist(),'observed_hours':len(obs),'case_count':500,
        'plan_sha256':sha(OUT/'plan.json'),'source_sha256':{date:sha(BASE/'processed'/f'wspr-{date}.parquet') for date in DATES},
        'raw_sha256':{date:sha(BASE/'raw'/f'wspr-{date}.jsonl') for date in DATES},'selection_uses_snr_or_prediction':False}
    p=OUT/'case_freeze.json'
    if p.exists():
        old=json.loads(p.read_text());frozen['frozen_utc']=old['frozen_utc'];assert old==frozen
    p.write_text(json.dumps(frozen,indent=2));c.to_csv(OUT/'cases_frozen.csv',index=False);obs.to_parquet(BASE/'case_features.parquet',index=False)
    print('Frozen500 new March2025 cases;',len(obs),'observed hours.',flush=True)
if __name__=='__main__':prepare()
