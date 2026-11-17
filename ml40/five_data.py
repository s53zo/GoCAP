"""Collect complete station-pair days for the five-idea WSPR experiment."""
import json
import time
from pathlib import Path

import requests

from .voacap_protocol import sha

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/ml40/five'
OUT = ROOT / 'results/ml40/five'
DATES = {
    'train': [f'2025-{month:02d}-{day:02d}' for month in range(1, 13) for day in (9, 17)],
    'validation': [f'2025-{month:02d}-23' for month in range(1, 13)],
    'test': [f'2026-{month:02d}-19' for month in range(1, 11)],
}


def collect():
    raw = CACHE / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split, dates in DATES.items():
        for date in dates:
            path = raw / f'{date}.jsonl'
            query = f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,code
FROM wspr.rx WHERE band=7 AND toDate(time,'UTC')=toDate('{date}')
AND cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%64=0
ORDER BY time,id LIMIT 200001 FORMAT JSONEachRow"""
            if not path.exists():
                for attempt in range(4):
                    try:
                        r = requests.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90))
                        r.raise_for_status()
                        lines = r.content.splitlines()
                        if not 0 < len(lines) < 200001:
                            raise ValueError('Empty or truncated WSPR window')
                        json.loads(lines[-1])
                        break
                    except requests.RequestException:
                        if attempt == 3:
                            raise
                        time.sleep(5)
                temp = path.with_suffix('.part')
                temp.write_bytes(r.content)
                temp.replace(path)
                time.sleep(3.2)
            manifest[date] = dict(split=split, sha256=sha(path), query=query)
            print(date, flush=True)
    (OUT/'sources.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    collect()
