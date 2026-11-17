"""Bounded event-hash samples for independent residual training and final SNR testing."""
import datetime as dt
import json,time,requests
from .hourly_plan import BASE,CORRECTION_DATES,TEST_DATES,plan
from .prepare import process_wspr,annotate
from .v3_features import enrich
from .test500 import digest

def collect():
    plan();raw=BASE/'raw';processed=BASE/'processed';raw.mkdir(exist_ok=True);processed.mkdir(exist_ok=True)
    session=requests.Session()
    for kind,dates in [('correction',CORRECTION_DATES),('test',TEST_DATES)]:
        for date in dates:
            path=raw/f'{kind}-{date}.jsonl';meta=path.with_suffix('.meta.json')
            if path.exists() and meta.exists():assert digest(path)==json.loads(meta.read_text())['sha256']
            else:
                end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
                query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC') AND cityHash64(id)%128=0
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
                for attempt in range(3):
                    try:
                        response=session.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90));response.raise_for_status()
                        rows=response.content.splitlines()
                        if len(rows)>=50001:raise ValueError('No first-row truncated sample permitted')
                        for row in rows:json.loads(row)
                        break
                    except (requests.RequestException,json.JSONDecodeError):
                        if attempt==2:raise
                        time.sleep(10)
                path.write_bytes(response.content);meta.write_text(json.dumps({'query':query,'rows':len(rows),'sha256':digest(path)},indent=2))
                print('Collected',kind,date,len(rows),'positive reports',flush=True);time.sleep(3.2)
            output=processed/f'{kind}-{date}.parquet'
            if not output.exists():
                d,audit=process_wspr(path)
                if len(d):d=enrich(annotate(d))
                d.to_parquet(output,index=False)
                output.with_suffix('.meta.json').write_text(json.dumps({'raw_sha256':digest(path),'audit':audit},indent=2))
    print('Hourly collection complete; final test unscored.',flush=True)
if __name__=='__main__':collect()
