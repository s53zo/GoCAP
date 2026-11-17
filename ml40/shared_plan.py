"""Frozen shared-mode / explicit path-point experiment, separate from deployment."""
import json,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ml40';RESULTS=ROOT/'results/ml40'
DATA.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
from .voacap_protocol import sha,ENGINE
BASE=DATA/'shared40';OUT=RESULTS/'shared40'
BASE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
TEST_DATES=[f'2025-{m:02d}-07' for m in range(1,13)]
ARTIFACT=ROOT/'artifacts/gocap-40m-shared-points.joblib'

def plan():
    old=json.loads((RESULTS/'voacap1000/protocol.json').read_text())
    value={'version':'shared-five-points-1','seed':'gocap-shared-points-500-v1','frozen_before_fitting':True,
        'objective':'One pooled multi-mode hourly native SNR regressor; explicit0/25/50/75/100-percent shortest-great-circle features. No strongest-hour objective.',
        'target':'Hourly median positive received SNR in2500Hz; no reception absence labels, no fabricated unknown transmit powers.',
        'training':'Original pre2025 fitting split only; all WSPR/FT8/FT4/CW. Existing2024day08 validation and day22 interval calibration are kept separate.',
        'mode_context':'One shared regressor retains source/mode/decoder context to represent detection selection and unknown hardware/power. Not a physically identified mode-independent channel-gain model.',
        'candidates':[{'id':f'pooled_{p}_{w}','points':p=='points','weighting':w} for p in ['aggregate','points'] for w in ['observation','mode_hour']],
        'tree':{'iterations':240,'learning_rate':.05,'leaves':31,'min_leaf':100,'l2':10,'loss':'absolute_error','seed':40},
        'selection':'Lowest equal-mode mean validation hour/date MAE among point-feature candidates passing WSPR<=baseline*1.05 and each other mode<=baseline*1.10. If none passes, lowest point-candidate macro error is exported experimental with failed guards; no baseline replacement. Aggregate candidates are feature ablations only.',
        'inference':'Default shared query uses the WSPR reference observation context at supplied power and applies existing theoretical thresholds to one SNR for WSPR/FT8/FT4/CW/PHONE. Hardware/PHONE physical calibration unverified. Other contexts are diagnostic only.',
        'test_dates':TEST_DATES,'primary_cases':500,'distance_quotas':{'100to1000':125,'1000to3000':125,'3000to6000':125,'6000to15000':125},
        'case_selection':'New day07 dates, fixed WSPR station pair/day/known reported power, >=6observed hours,20..50dBm,100..15000km. Identifier hash/date round-robin; exclude all directed square pairs in previous100/1000suites. No SNR/prediction/error selection.',
        'sampling':'Hash128 pair/day at shorter paths OR source distance>=2700km and hash16.250001cap; fail if capped rather than truncate.',
        'benchmark_scope':'WSPR primary because reported power is known for fair VOACAP input. Additional all-mode2025 hourly held-out archive audit uses previously published test dates; it is a known-benchmark audit, not a new blind multi-mode validation. No unknown-power VOACAP score.',
        'primary_scoring':'Case-average hourly absolute SNR error; also pooled MAE/RMSE/bias, per-distance/hour, date-cluster bootstrap and nominal80% interval coverage. All24 requested hours predicted; missing measurements stay unknown.',
        'voacap_scenarios':old['scenarios'],'voacap_primary':old['primary_scenario'],'voacap_assumptions':old['voacap'],
        'baseline_sha256':sha(ROOT/'artifacts/gocap-40m-hourly.joblib'),'fit_sha256':sha(DATA/'fit_features.parquet'),
        'engine_sha256':sha(ENGINE),'excluded_case_sha256':{size:sha(RESULTS/f'voacap{size}/cases_frozen.csv') for size in ['100','1000']},
        'solar_sha256':sha(DATA/'voacap1000/silso_smoothed_v2.txt')}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==value,'Frozen shared-model plan changed'
    else:p.write_text(json.dumps(value,indent=2))
    return value
if __name__=='__main__':plan()
