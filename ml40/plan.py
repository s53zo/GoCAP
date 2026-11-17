"""Outcome-independent sampling and split plan, frozen before v2 modelling."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'ml40'
RESULTS=ROOT/'results'/'ml40'
DATA.mkdir(parents=True,exist_ok=True)
RESULTS.mkdir(parents=True,exist_ok=True)

def dates():
    return [f'{year}-{month:02d}-{day:02d}' for year in [2010,2013,2016,2019,2022,2024,2025]
        for month in range(1,13) for day in ([1,8,22] if year==2025 else [1,8,15,22])]

def split(date):
    d=dt.date.fromisoformat(str(date)[:10])
    if d.year==2025:return 'test'
    if d.year==2024 and d.day==8:return 'validation'
    if d.year==2024 and d.day==22:return 'calibration'
    return 'train'

HARC_DATES=['2024-01-15','2024-04-15','2024-07-15','2024-10-15',
    '2024-04-08','2024-10-08','2024-07-22','2024-10-22',
    '2025-04-22','2025-07-22','2025-10-22']

def station_held(call):
    return int(hashlib.sha256(str(call).encode()).hexdigest()[:8],16)%5==0

def geographic_held(grid):
    # Hold out 10% of 10x20-degree Maidenhead fields, not isolated neighbouring squares.
    return int(hashlib.sha256(('geo:'+str(grid)[:2]).encode()).hexdigest()[:8],16)%10==0

def manifest():
    value={'version':2,'band_m':40,'positive_only':True,'wspr_dates':dates(),
        'wspr_hash_divisor':128,'harc_dates':HARC_DATES,
        'splits':{d:split(d) for d in dates()},'seed':40,
        'station_holdout':'SHA256 callsign first 32 bits modulo 5 = 0; both roles excluded from fitting',
        'geographic_holdout':'SHA256 geo:field first 32 bits modulo 10 = 0; both roles excluded from fitting',
        'test_policy':'2025 days 1,8,22 are final test; 2025 day15 pilot dates are excluded',
        'selection_policy':'choose algorithms/epoch count on validation only; interval adjustment uses calibration only; test scores released after configuration freeze',
        'station_effects':'regularized additive TX/RX offsets from training residuals; mean/reference constraints; unseen offsets zero; antennas/noise not individually identified',
        'models':['boosted_baseline','boosted_station_adjusted','small_mlp_station_adjusted'],
        'primary_metric':'power-normalized hourly ranking among observed hours, with per-mode reporting',
        'secondary_metrics':['MAE','RMSE','interval coverage and width','unseen station performance','held geographic field performance'],
        'storage':'Parquet training samples and compact model artifacts; no new relational database'}
    path=RESULTS/'plan.json'
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Frozen plan differs; create an explicit revised plan rather than overwriting')
    path.write_text(json.dumps(value,indent=2))
    return value

if __name__=='__main__':
    m=manifest()
    from collections import Counter
    print(len(m['wspr_dates']),'WSPR days',dict(Counter(m['splits'].values())))
