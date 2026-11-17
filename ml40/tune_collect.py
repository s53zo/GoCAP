"""Bounded untouched WSPR test collection for parameter tuning, never used to fit."""
import datetime as dt
import json
import time
import requests
from .plan import DATA
from .prepare import process_wspr,annotate
from .test500 import digest

BASE=DATA/'tuning'
DATES=[f'2025-{m:02d}-02' for m in range(1,13)]

def collect():
    raw=BASE/'fresh'/'raw';processed=BASE/'fresh'/'processed'
    raw.mkdir(parents=True,exist_ok=True);processed.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    for date in DATES:
        for kind in ['snr','ranking']:
            path=raw/f'{kind}-{date}.jsonl';meta=path.with_suffix('.meta.json')
            if path.exists() and meta.exists():
                assert digest(path)==json.loads(meta.read_text())['sha256']
            else:
                end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
                clause='cityHash64(id)%128=0' if kind=='snr' else "cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%512=0"
                query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC') AND {clause}
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
                for attempt in range(3):
                    try:
                        r=session.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90));r.raise_for_status()
                        records=r.content.splitlines()
                        if len(records)>=50001:raise ValueError('Sampling cap hit: no truncated test sample permitted')
                        for row in records:json.loads(row)
                        break
                    except (requests.RequestException,json.JSONDecodeError):
                        if attempt==2:raise
                        time.sleep(10)
                path.write_bytes(r.content)
                meta.write_text(json.dumps({'query':query,'rows':len(records),'sha256':digest(path),
                    'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2))
                print('Collected sealed fresh test',kind,date,len(records),flush=True)
                time.sleep(3.2)
            output=processed/f'{kind}-{date}.parquet'
            if not output.exists():
                d,audit=process_wspr(path)
                annotate(d).to_parquet(output,index=False)
    print('Fresh test collection complete; no predictions or scores generated.',flush=True)

if __name__=='__main__':collect()
