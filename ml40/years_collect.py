"""Coherent fresh source data on60 genuinely held-out April dates."""
import datetime as dt,json,time,requests
import numpy as np,pandas as pd
from experiment import hourly,features,grid4
from .prepare import annotate
from .v3_features import enrich
from .years_plan import BASE,OUT,DATES,plan,sha

def process(path):
    raw=pd.read_json(path,lines=True)
    if raw.empty:return raw,{'raw_rows':0,'eligible_rows':0}
    raw['time']=pd.to_datetime(raw.time,utc=True)
    legacy=raw.code.eq(0)&raw.time.lt(pd.Timestamp('2020-05-01',tz='UTC'));eligible=legacy|raw.code.eq(1)
    audit={'raw_rows':len(raw),'eligible_rows':int(eligible.sum()),'code_counts':raw.code.value_counts().to_dict()}
    raw=raw[eligible].copy()
    if raw.empty:return raw,audit
    raw['source']='WSPRNET';raw['mode']='WSPR';raw['snr2500']=raw.snr.astype(float);raw['power_dbm']=raw.power.astype(float)
    raw['decoder_context']=np.where(raw.code.eq(0),'legacy_code0',np.where(raw.time.lt(pd.Timestamp('2023-01-16',tz='UTC')),'pre2023_code1','ordinary_code1'))
    h=hourly(raw)
    # Within an aggregate, any legacy0 report takes precedence over code1 context.
    raw['hour']=raw.time.dt.floor('h');raw['tx_grid']=raw.tx_loc.map(grid4);raw['rx_grid']=raw.rx_loc.map(grid4)
    raw=raw[raw.tx_grid.notna()&raw.rx_grid.notna()&raw.frequency.between(7000000,7300000)&raw.snr.between(-60,100)&raw.tx_sign.astype(str).str.len().ge(2)&raw.rx_sign.astype(str).str.len().ge(2)].drop_duplicates(['time','tx_sign','rx_sign','frequency']).copy()
    raw['power_dbm']=raw.power_dbm.where(raw.power_dbm.between(0,60))
    raw['context_rank']=raw.decoder_context.map({'legacy_code0':0,'pre2023_code1':1,'ordinary_code1':2})
    keys=['hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']
    ctx=raw.groupby(keys,dropna=False,observed=True).context_rank.min().rename('context_rank').reset_index()
    h=h.merge(ctx,on=keys,validate='one_to_one');h['decoder_context']=h.context_rank.map({0:'legacy_code0',1:'pre2023_code1',2:'ordinary_code1'})
    d=enrich(annotate(features(h,solar_start='2009')))
    return d,audit

def collect():
    plan();rawdir=BASE/'raw';processed=BASE/'processed';rawdir.mkdir(exist_ok=True);processed.mkdir(exist_ok=True)
    session=requests.Session()
    for date in DATES:
        path=rawdir/f'wspr-{date}.jsonl';meta=path.with_suffix('.meta.json');out=processed/path.with_suffix('.parquet').name
        if path.exists():assert sha(path)==json.loads(meta.read_text())['sha256']
        else:
            end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat();divisor=8 if int(date[:4])<=2015 else 128
            query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC')
AND cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%{divisor}=0
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
            for attempt in range(4):
                try:
                    r=session.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90));r.raise_for_status();lines=r.content.splitlines()
                    if len(lines)>=50001:raise ValueError('Query cap hit; no truncation permitted')
                    for row in lines:json.loads(row)
                    break
                except (requests.RequestException,json.JSONDecodeError):
                    if attempt==3:raise
                    time.sleep(10)
            path.write_bytes(r.content);meta.write_text(json.dumps({'query':query,'hash_divisor':divisor,'rows':len(lines),'sha256':sha(path),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2));time.sleep(3.2)
        if not out.exists():
            d,audit=process(path);d.to_parquet(out,index=False)
            out.with_suffix('.meta.json').write_text(json.dumps({'raw_sha256':sha(path),'prepared_sha256':sha(out),'audit':audit},indent=2))
        print(date,'reports',json.loads(meta.read_text())['rows'],'ready',flush=True)
    print('60held-out dates collected. No case scores generated.',flush=True)
if __name__=='__main__':collect()
