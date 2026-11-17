"""New outcome-blind coherent WSPR source sample for500 comparison paths."""
import datetime as dt,json,time,requests
from .shared_plan import BASE,OUT,TEST_DATES,plan,sha
from .prepare import process_wspr,annotate
from .v3_features import enrich

def collect():
    plan();raw=BASE/'raw';processed=BASE/'processed';raw.mkdir(exist_ok=True);processed.mkdir(exist_ok=True)
    session=requests.Session()
    for date in TEST_DATES:
        p=raw/f'wspr-{date}.jsonl';meta=p.with_suffix('.meta.json')
        if p.exists():assert sha(p)==json.loads(meta.read_text())['sha256']
        else:
            end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
            query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC')
AND (cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%128=0
OR (distance>=2700 AND cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%16=0))
ORDER BY time,id LIMIT 250001 FORMAT JSONEachRow"""
            for attempt in range(3):
                try:
                    r=session.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90));r.raise_for_status()
                    records=r.content.splitlines();assert len(records)<250001,'Query cap reached; no truncation permitted'
                    for line in records:json.loads(line)
                    break
                except (requests.RequestException,json.JSONDecodeError):
                    if attempt==2:raise
                    time.sleep(10)
            p.write_bytes(r.content);meta.write_text(json.dumps({'query':query,'rows':len(records),'sha256':sha(p),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2))
            print(date,len(records),'reports',flush=True);time.sleep(3.2)
        out=processed/f'wspr-{date}.parquet'
        if not out.exists():
            d,audit=process_wspr(p);d=enrich(annotate(d));d.to_parquet(out,index=False)
            out.with_suffix('.meta.json').write_text(json.dumps({'raw_sha256':sha(p),'audit':audit},indent=2))
    print('New source collection complete, no test scores generated.',flush=True)
if __name__=='__main__':collect()
