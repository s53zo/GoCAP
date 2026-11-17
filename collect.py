"""Bounded, cached downloads for the 40m feasibility study (research only)."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import time
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW = DATA / 'raw'
RAW.mkdir(parents=True, exist_ok=True)

# Preselected independently of reception SNR; covers four seasons and the May 2024 storm.
DATES = ['2023-01-15', '2023-04-15', '2023-07-15', '2023-10-15',
         '2024-01-15', '2024-04-15', '2024-05-01', '2024-05-10',
         '2024-05-11', '2024-05-12', '2024-07-15', '2024-10-15',
         '2025-01-15', '2025-04-15', '2025-07-15', '2025-10-15']

def fetch(name, url, params=None, timeout=60):
    path = RAW / name
    meta = path.with_suffix(path.suffix + '.meta.json')
    if path.exists() and meta.exists():
        m = json.loads(meta.read_text())
        if m.get('params') != params:
            raise ValueError(f'Cached request parameters differ: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != m['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        return path
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    if 'db1.wspr.live' in url and response.content.lstrip().startswith(b'<'):
        raise ValueError('HTML returned by data API')
    payload = response.content
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_bytes(payload)
    tmp.replace(path)
    meta.write_text(json.dumps({'url': response.url, 'params': params,
        'retrieved_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}, indent=2))
    return path

def wspr(dates=DATES):
    for date in dates:
        end = (dt.date.fromisoformat(date) + dt.timedelta(days=1)).isoformat()
        # Hash selection avoids taking the first receiver/time block from sorted storage.
        # A 1/64 sample keeps queries and local storage bounded; no outcome filter.
        query = f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC') AND cityHash64(id)%64=0
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
        path = fetch(f'wspr-{date}.jsonl', 'https://db1.wspr.live/', {'query': query})
        n = sum(1 for _ in path.open())
        if n >= 50001:
            raise ValueError('Sample hit limit; increase hash divisor before proceeding')
        print(f'{date}: {n} records, {path.stat().st_size:,} bytes', flush=True)
        time.sleep(3.1)  # published limit: 20 requests/minute; one sequential connection

def rbn(dates):
    for date in dates:
        compact=date.replace('-','')
        p=fetch(f'rbn-{compact}.zip',f'https://data.reversebeacon.net/rbn_history/{compact}.zip')
        print(date,p.stat().st_size,flush=True)
        time.sleep(3.1)

def gfz():
    fetch('gfz.txt','https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt')

def pskr_probe():
    # One capped API request; schema/quality only, never historical training data.
    fetch('pskr-current.xml','https://retrieve.pskreporter.info/query',
        {'flowStartSeconds':-900,'frange':'7000000-7300000','rptlimit':1000,'rronly':1,'noactive':1})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dates', nargs='*', default=DATES)
    parser.add_argument('--source', choices=['wspr','rbn','gfz','pskr-probe'], default='wspr')
    args = parser.parse_args()
    if args.source=='wspr':wspr(args.dates)
    elif args.source=='rbn':rbn(args.dates)
    elif args.source=='gfz':gfz()
    else:pskr_probe()
