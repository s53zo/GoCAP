"""All-March2010..2024 expansion; frozen parent model and untouched2025 March audit."""
import json,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/ml40';OUT=ROOT/'results/ml40/march40';BASE=DATA/'march40'
OUT.mkdir(parents=True,exist_ok=True);BASE.mkdir(parents=True,exist_ok=True)
from .voacap_protocol import sha
YEARS=list(range(2010,2025));DATES=[f'{y}-03-{d:02d}' for y in YEARS for d in range(1,32)]
FRESH_TEST=[f'2025-03-{d:02d}' for d in [9,12,17,20,25,28]]
ARTIFACT=ROOT/'artifacts/gocap-40m-shared-march.joblib'
KEYS=['source','mode','hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']

def plan():
    value={'version':'march2010to2024-1','years':YEARS,'march_dates':DATES,'event_hash_divisor':128,'query_cap':50001,
        'user_scope':'Every year2010..2024 inclusive, every available March day. Same reproducible1/128 WSPR event sampling; not an exhaustive raw-report census.',
        'augmentation':'Add all available March WSPR aggregates to original pre2025 multi-mode training. Keep original FT8/FT4/CW; no invented PHONE data. Keep existing2024 validation/calibration out of fitting, including March08/22.',
        'duplicate_policy':'For existing aggregate keys, original samples win and must agree in target/report count. No reweighting through duplicate March rows.',
        'training':'Same one pooled SharedModel(points=True,weighting=observation),240 trees,31 leaves,minleaf100,L210,lr.05,absolute error. No hyperparameter search; isolate data expansion.',
        'features':'Same45 explicit0/25/50/75/100-percent features and original nine-point summaries/solar history. Keep decoder/source/mode and known-power status.',
        'fresh_march2025_test_dates':FRESH_TEST,'fresh_test_policy':'No2025 fitting. New day09/12/17/20/25/28 reports, never used for selecting this configuration; score only after artifact freeze.',
        'audit_policy':'Also evaluate known500-path shared40 suite and its known other-mode archive. No claim that previously reported suites are new blind tests; no new VOACAP runs required for training expansion.',
        'score':'Fresh March primary: equally weighted UTC hour/date MAE; pooled MAE/RMSE/bias,80% interval coverage. Known500: equally weighted path MAE against observed hours. Missing reception unknown.',
        'parent_sha256':sha(ROOT/'artifacts/gocap-40m-shared-points.joblib'),'baseline_sha256':sha(ROOT/'artifacts/gocap-40m-hourly.joblib'),
        'original_fit_sha256':sha(DATA/'fit_features.parquet'),'solar_input_sha256':sha(ROOT/'data/raw/gfz.txt'),
        'original_input_sha256':{n:sha(DATA/'hourly'/n) for n in ['original_train_features.parquet','validation_features.parquet','calibration_features.parquet']}}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==value,'Frozen March expansion differs'
    else:
        p.write_text(json.dumps(value,indent=2));(OUT/'plan_created.json').write_text(json.dumps({'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'plan_sha256':sha(p)},indent=2))
    return value
if __name__=='__main__':plan()
