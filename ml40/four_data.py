"""Matched positive-only 40 m experiment with frozen temporal and receiver splits."""
import json
import time
from pathlib import Path

import pandas as pd
import requests

from .voacap_protocol import sha

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/ml40/four'
OUT = ROOT/'results/ml40/four'
DATES = {
    'train': [f'2023-{m:02d}-{d:02d}' for m in range(1,8) for d in (10,20) if (m,d)!=(1,10)]
             + ['2023-02-17','2023-07-02','2025-01-14','2025-04-18'],
    'validation': ['2025-04-19','2025-04-21','2025-10-21','2025-11-08','2025-11-09'],
    'test': ['2025-11-11','2025-11-12','2025-11-18','2025-11-25','2025-12-05','2025-12-15',
             '2026-01-07','2026-01-18','2026-01-23','2026-02-01','2026-02-02','2026-02-09','2026-02-14'],
}
ALL = sum(DATES.values(),[])


def fetch(url, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url,timeout=(20,120),**kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt==3:
                raise
            time.sleep(4)


def collect(dates=ALL,manifest='sources.json'):
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    ledger = {}
    for date in dates:
        begin = pd.Timestamp(date)
        end = (begin+pd.Timedelta(days=1)).date()
        pre = (begin-pd.Timedelta(days=7)).date()
        mode_filter = "code=1"
        queries = {
            'spots':f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,code FROM wspr.rx
WHERE band=7 AND {mode_filter} AND time>='{date}' AND time<'{end}'
AND cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%48=0
ORDER BY time,id LIMIT 250001 FORMAT JSONEachRow""",
            'noise':f"""SELECT toStartOfHour(time) AS h, site,receiver,substring(upper(rx_loc),1,4) AS grid,
medianExact(rms_level) AS noise,count() AS n FROM wspr.wsprdaemon_noise
WHERE band='40' AND time>='{pre}' AND time<'{end}' AND rms_level BETWEEN -165 AND -70
AND (ov=0 OR ov IS NULL) GROUP BY h,site,receiver,grid ORDER BY h,site,receiver,grid
LIMIT 250001 FORMAT JSONEachRow""",
        }
        for key,q in queries.items():
            path = CACHE/f'{key}-{date}.jsonl'
            if not path.exists():
                r = fetch('https://db1.wspr.live/',params={'query':q})
                if len(r.content.splitlines())>=250001:
                    raise ValueError('Source cap reached')
                tmp = path.with_suffix('.part')
                tmp.write_bytes(r.content)
                tmp.replace(path)
                time.sleep(1.5)
            ledger[path.name] = {'query':q,'sha256':sha(path)}
        print(date,flush=True)
    (OUT/manifest).write_text(json.dumps(ledger,indent=2))


if __name__=='__main__':
    collect()
