"""Year-balanced outcome-blind300case freeze with fitting-date exclusion proof."""
import datetime as dt,hashlib,json
from collections import deque
import numpy as np,pandas as pd
from .years_plan import ROOT,DATA,BASE,OUT,YEARS,DATES,KEYS,plan,sha

def select(d,cfg):
    c=d.groupby(KEYS,sort=True).agg(observed_hours=('hour','nunique'),distance_km=('distance_km','first')).reset_index()
    c=c[c.observed_hours.ge(6)&c.power_dbm.between(20,50)&c.distance_km.between(100,15000)].copy();c['year']=pd.to_datetime(c.date_key).dt.year
    c['case_id']=[hashlib.sha256((cfg['seed']+json.dumps(list(k),default=str)).encode()).hexdigest()[:20] for k in c[KEYS].itertuples(index=False,name=None)]
    rows=[]
    for year in YEARS:
        part=c[c.year.eq(year)].sort_values('case_id');queues=[deque(g.to_dict('records')) for _,g in part.groupby('date_key',sort=True)]
        used=set();count=0
        while count<20:
            progress=False
            for q in queues:
                while q:
                    row=q.popleft();path=(row['tx_grid'],row['rx_grid'])
                    if path in used:continue
                    used.add(path);rows.append(row);count+=1;progress=True;break
                if count==20:break
            if not progress:raise ValueError(f'Insufficient eligible paths in{year}:{count}/20; do not choose dates using scores')
    return pd.DataFrame(rows).sort_values('case_id').reset_index(drop=True)

def prepare():
    cfg=plan();dates=set(DATES)
    for p in [DATA/'march40/expanded_train_features.parquet',DATA/'hourly/original_train_features.parquet']:
        fitted=pd.read_parquet(p,columns=['hour']);assert not set(fitted.hour.dt.strftime('%Y-%m-%d'))&dates
    # Canonicalize processed caches with the final pre-score measurement processor.
    from .years_collect import process
    processor_sha=sha(ROOT/'ml40/years_collect.py')
    for date in DATES:
        raw=BASE/'raw'/f'wspr-{date}.jsonl';p=BASE/'processed'/f'wspr-{date}.parquet';meta=p.with_suffix('.meta.json')
        assert raw.exists() and meta.exists()
        old=json.loads(meta.read_text())
        if old.get('processor_sha256')!=processor_sha:
            d,audit=process(raw);d.to_parquet(p,index=False);meta.write_text(json.dumps({'raw_sha256':sha(raw),'prepared_sha256':sha(p),'audit':audit,'processor_sha256':processor_sha},indent=2))
    d=pd.concat([pd.read_parquet(BASE/'processed'/f'wspr-{date}.parquet') for date in DATES],ignore_index=True)
    chosen=select(d,cfg);assert len(chosen)==300 and chosen.case_id.nunique()==300
    solar=pd.read_csv(DATA/'voacap1000/silso_smoothed_v2.txt',sep=r'\s+',header=None,names=['year','month','decimal_year','ssn','std','count','provisional'])
    chosen['month']=4;chosen=chosen.merge(solar[['year','month','ssn']],on=['year','month'],validate='many_to_one')
    assert chosen.ssn.notna().all() and chosen.ssn.between(0,299).all();chosen['reported_power_watts']=10**((chosen.power_dbm-30)/10)
    obs=d.merge(chosen[KEYS+['case_id']],on=KEYS,how='inner',validate='many_to_one').sort_values(['case_id','hour']).reset_index(drop=True)
    assert not obs.duplicated(['case_id','hour']).any()
    freeze={'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'case_ids':chosen.case_id.tolist(),'case_count':300,'observed_hours':len(obs),'years':YEARS,
        'plan_sha256':sha(OUT/'plan.json'),'source_sha256':{date:sha(BASE/'processed'/f'wspr-{date}.parquet') for date in DATES},
        'raw_sha256':{date:sha(BASE/'raw'/f'wspr-{date}.jsonl') for date in DATES},'processor_sha256':processor_sha,'no_selected_day_in_fitting':True,'selection_uses_snr_or_prediction':False}
    fp=OUT/'case_freeze.json'
    if fp.exists():
        old=json.loads(fp.read_text());freeze['frozen_utc']=old['frozen_utc'];assert old==freeze
    fp.write_text(json.dumps(freeze,indent=2));chosen.to_csv(OUT/'cases_frozen.csv',index=False);obs.to_parquet(BASE/'case_features.parquet',index=False)
    print('Frozen300 cases;20/year;',len(obs),'observed hourly values.',flush=True)
if __name__=='__main__':prepare()
