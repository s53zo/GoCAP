"""Freeze500 new WSPR paths by identifiers, excluding all prior VOACAP paths."""
import datetime as dt,hashlib,json
from collections import deque
import numpy as np,pandas as pd
from .shared_plan import BASE,OUT,DATA,RESULTS,TEST_DATES,plan,sha
from .voacap_protocol import KEYS
GROUPS=['100to1000','1000to3000','3000to6000','6000to15000']

def select(d,prior,cfg):
    c=d.groupby(KEYS,sort=True).agg(observed_hours=('hour','nunique'),distance_km=('distance_km','first')).reset_index()
    c=c[c.observed_hours.ge(6)&c.power_dbm.between(20,50)&c.distance_km.between(100,15000)].copy()
    c['distance_group']=pd.cut(c.distance_km,[99.999,1000,3000,6000,15000],labels=GROUPS).astype(str)
    c['case_id']=[hashlib.sha256((cfg['seed']+json.dumps(list(k),default=str)).encode()).hexdigest()[:20] for k in c[KEYS].itertuples(index=False,name=None)]
    used=set(zip(prior.tx_grid,prior.rx_grid));rows=[]
    for distance in GROUPS:
        part=c[c.distance_group.eq(distance)].sort_values('case_id');queues=[deque(x.to_dict('records')) for _,x in part.groupby('date_key',sort=True)];count=0;target=cfg['distance_quotas'][distance]
        while count<target:
            progress=False
            for q in queues:
                while q:
                    row=q.popleft();key=(row['tx_grid'],row['rx_grid'])
                    if key in used:continue
                    used.add(key);rows.append(row);count+=1;progress=True;break
                if count==target:break
            if not progress:raise ValueError(f'Insufficient previously unused paths in {distance}:{count}/{target}')
    return pd.DataFrame(rows).sort_values('case_id').reset_index(drop=True)

def prepare():
    cfg=plan();d=pd.concat([pd.read_parquet(BASE/'processed'/f'wspr-{date}.parquet') for date in TEST_DATES],ignore_index=True)
    prior=pd.concat([pd.read_csv(RESULTS/f'voacap{n}/cases_frozen.csv') for n in [100,1000]],ignore_index=True)
    chosen=select(d,prior,cfg);assert len(chosen)==500
    solar=pd.read_csv(DATA/'voacap1000/silso_smoothed_v2.txt',sep=r'\s+',header=None,names=['year','month','decimal_year','ssn','std','count','provisional'])
    chosen['year']=pd.to_datetime(chosen.date_key).dt.year;chosen['month']=pd.to_datetime(chosen.date_key).dt.month
    chosen=chosen.merge(solar[['year','month','ssn']],on=['year','month'],validate='many_to_one')
    chosen['reported_power_watts']=10**((chosen.power_dbm-30)/10)
    observed=d.merge(chosen[KEYS+['case_id']],on=KEYS,how='inner',validate='many_to_one').sort_values(['case_id','hour']).reset_index(drop=True)
    assert not observed.duplicated(['case_id','hour']).any()
    frozen={'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'case_ids':chosen.case_id.tolist(),'cases':500,'observed_hours':len(observed),
        'plan_sha256':sha(OUT/'plan.json'),'source_sha256':{p.name:sha(p) for p in sorted((BASE/'processed').glob('*.parquet'))},
        'raw_source_sha256':{p.name:sha(p) for p in sorted((BASE/'raw').glob('*.jsonl'))},'selection_uses_signal_or_prediction':False}
    path=OUT/'case_freeze.json'
    if path.exists():
        old=json.loads(path.read_text());frozen['frozen_utc']=old['frozen_utc'];assert old==frozen
    path.write_text(json.dumps(frozen,indent=2));chosen.to_csv(OUT/'cases_frozen.csv',index=False);observed.to_parquet(BASE/'case_features.parquet',index=False)
    print('Frozen500 new paths with',len(observed),'observed hours:',chosen.groupby('distance_group').size().to_dict())
if __name__=='__main__':prepare()
