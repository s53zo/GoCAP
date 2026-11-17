"""Increase repeatability of power-step measurements for train-identified transmitters."""
import json
import re
import time
import pandas as pd
from .four_data import CACHE,DATES,ALL,OUT,fetch,sha
from .four_prepare import clean,power_pairs


def collect():
    train=pd.concat([pd.read_parquet(CACHE/f'steps-{d}.parquet') for d in DATES['train']])
    counts=train.groupby('tx_sign').size().sort_values(ascending=False)
    calls=[x for x in counts.index if re.fullmatch(r'[A-Za-z0-9/-]+',x)][:8]
    names=','.join("'"+x+"'" for x in calls)
    manifest={}
    for date in ALL:
        p=CACHE/f'steptx-{date}.jsonl'
        end=(pd.Timestamp(date)+pd.Timedelta(days=1)).date()
        q=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,code FROM wspr.rx
WHERE band=7 AND code=1 AND time>='{date}' AND time<'{end}' AND tx_sign IN ({names})
ORDER BY time,id LIMIT 250001 FORMAT JSONEachRow"""
        if not p.exists():
            r=fetch('https://db1.wspr.live/',params={'query':q})
            assert len(r.content.splitlines())<250001
            p.write_bytes(r.content)
            time.sleep(1.5)
        ordinary=pd.read_parquet(CACHE/f'steps-{date}.parquet')
        supplement=power_pairs(clean(date,'steptx')) if p.stat().st_size else ordinary.iloc[:0]
        pd.concat([ordinary,supplement]).drop_duplicates(['time','rx_sign','tx_sign','tx_loc','rx_loc','step','power_dbm']).to_parquet(CACHE/f'allsteps-{date}.parquet',index=False)
        manifest[date]=dict(query=q,sha256=sha(p))
    (OUT/'power_sources.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    collect()
