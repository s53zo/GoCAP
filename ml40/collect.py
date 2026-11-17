"""Sequential, bounded second-stage collection; resumable cached responses."""
import argparse
import datetime as dt
import hashlib
import json
import time
import requests
from .plan import DATA,RESULTS,manifest

RAW=DATA/'raw'
RAW.mkdir(exist_ok=True)

def wspr(limit=None):
    plan=manifest();selected=plan['wspr_dates'][:limit] if limit else plan['wspr_dates']
    session=requests.Session()
    for date in selected:
        path=RAW/f'wspr-{date}.jsonl';meta=path.with_suffix('.meta.json')
        if path.exists() and meta.exists():
            info=json.loads(meta.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Checksum mismatch')
            print(date,'cached',info['rows'],flush=True);continue
        end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
        query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC') AND cityHash64(id)%128=0
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
        for attempt in range(3):
            try:
                r=session.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90))
                r.raise_for_status()
                rows=r.content.splitlines()
                if len(rows)>=50001:raise ValueError('Sample cap reached; do not use a first-row truncated sample')
                for row in rows:json.loads(row)
                break
            except (requests.RequestException,json.JSONDecodeError):
                if attempt==2:raise
                time.sleep(10)
        part=path.with_suffix('.part');part.write_bytes(r.content);part.replace(path)
        info={'endpoint':'https://db1.wspr.live/','query':query,'rows':len(rows),
            'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),
            'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        meta.write_text(json.dumps(info,indent=2))
        print(date,len(rows),f'{len(r.content)/1e6:.2f} MB',flush=True)
        time.sleep(3.2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args();wspr(args.limit)
