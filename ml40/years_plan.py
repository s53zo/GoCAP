"""Fifteen-year held-out April comparison; no resubstitution on March fitting data."""
import json,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/ml40';OUT=ROOT/'results/ml40/years15';BASE=DATA/'years15'
OUT.mkdir(parents=True,exist_ok=True);BASE.mkdir(parents=True,exist_ok=True)
from .voacap_protocol import sha,ENGINE
YEARS=list(range(2010,2025));DATES=[f'{y}-04-{d:02d}' for y in YEARS for d in [6,13,20,27]]
ARTIFACTS={'march':ROOT/'artifacts/gocap-40m-shared-march.joblib','shared_parent':ROOT/'artifacts/gocap-40m-shared-points.joblib','current':ROOT/'artifacts/gocap-40m-hourly.joblib'}
KEYS=['date_key','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm','decoder_context']

def plan():
    prior=json.loads((ROOT/'results/ml40/voacap1000/protocol.json').read_text())
    value={'version':'years15-april-holdout-1','seed':'gocap15year-300-v1','years':YEARS,'dates':DATES,'month':4,'cases_per_year':20,'cases':300,
        'why_april':'March2010..2024 is in expanded fitting data. FourApril dates per year are absent from all point-fitting data. This is retrospective within-period held-out-day evaluation, not forward-year forecasting.',
        'source_sampling':'Coherent station-pair/day hash8 for2010..2015, hash128 for2016..2024. Time/band bounded sequential requests;50001row cap causes failure, never first-row truncation.',
        'eligibility':'Positive WSPR archive reports, fixed stations/four-char squares/power/context per day;>=6observed hours;20..50dBm;100..15000km. No SNR/error selection.',
        'legacy_policy':'code0 before2020-05-01 retained as legacy_code0 with uncertain decoder metadata. code1 before2023-01-16 is pre2023_code1; modern code1 ordinary. No hard common decoding threshold inferred for legacy contexts.',
        'selection':'Identifier hash, round-robin four dates within each year.20 unique directed square pairs per year; repeats across years allowed. No distance quota because early network coverage is sparse.',
        'scoring':'Mean per-case hourly absolute SNR error;20cases per year means years equally weighted. Also pooled MAE/RMSE/bias, per-year/distance/UTC, numeric coverage, and year-cluster bootstrap. Missing reports unknown.',
        'voacap_primary':prior['primary_scenario'],'voacap_scenarios':prior['scenarios'],'voacap_assumptions':prior['voacap'],
        'models_sha256':{n:sha(p) for n,p in ARTIFACTS.items()},'engine_sha256':sha(ENGINE),
        'solar_sha256':sha(DATA/'voacap1000/silso_smoothed_v2.txt'),'gfz_sha256':sha(ROOT/'data/raw/gfz.txt'),
        'fitting_inputs_sha256':{'march':sha(DATA/'march40/expanded_train_features.parquet'),'original':sha(DATA/'hourly/original_train_features.parquet')},
        'limits':'Final historical solar inputs. Unknown antennas/noise and detection selection. April-only across15years; path/station distributions differ by year, so year differences are not causal solar effects. Models trained on later years can predict earlier held-out days; this is not an operational vintage test.'}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==value,'Frozen15year plan changed'
    else:
        p.write_text(json.dumps(value,indent=2));(OUT/'plan_created.json').write_text(json.dumps({'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'sha256':sha(p)},indent=2))
    return value
if __name__=='__main__':plan()
