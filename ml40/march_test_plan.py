"""Frozen additional March2025 test; no fitting and no previously scored dates."""
import json,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/ml40';BASE=DATA/'march2025voa';OUT=ROOT/'results/ml40/march2025voa'
BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
from .voacap_protocol import sha,ENGINE
from .years_plan import ARTIFACTS,KEYS
DAYS=[10,11,13,14,16,18,19,21,23,24,26,27,29,30,31];DATES=[f'2025-03-{d:02d}' for d in DAYS]
GROUPS=['100to1000','1000to3000','3000to6000','6000to15000']

def plan():
    old=json.loads((ROOT/'results/ml40/years15/plan.json').read_text())
    value={'version':'march2025-unseen-500-1','seed':'gocap-march2025-unseen-500-v1','dates':DATES,'cases':500,'distance_quotas':dict.fromkeys(GROUPS,125),
        'scope':'Additional untouched March2025 dates; model artifacts are frozen. Prior March2025 dates1,2,3,4,6,7,8,9,12,15,17,20,22,25,28 are excluded; day5 also excluded conservatively.',
        'sampling':'Coherent pair/day hash128 for short paths OR source distance>=2700km and hash16;250001row cap fails rather than truncate. Source/mode validity uses modern code1.',
        'selection':'Identifier hash/date round-robin within four distance strata;500 unique directed square pairs, fixed stations/day/reported power/context,>=6observed hours,20..50dBm,100..15000km. No SNR/prediction/error choice.',
        'scoring':'Case-average hourly SNR MAE; also pooled MAE/RMSE/bias, per-date/hour/distance and date-cluster bootstrap. Missing reception unknown; no binary band-open labels.',
        'voacap_primary':old['voacap_primary'],'voacap_scenarios':old['voacap_scenarios'],'voacap_assumptions':old['voacap_assumptions'],
        'models_sha256':{n:sha(p) for n,p in ARTIFACTS.items()},'engine_sha256':sha(ENGINE),
        'solar_sha256':sha(DATA/'voacap1000/silso_smoothed_v2.txt'),'gfz_sha256':sha(ROOT/'data/raw/gfz.txt'),
        'fitting_inputs_sha256':old['fitting_inputs_sha256'],
        'limits':'Positive detection-selected WSPR observations, unknown antennas/noise; VOACAP monthly climatology versus particular dates. Final retrospective solar indices, not operational forecast-vintage validation. Route overlap with past suites is permitted; dates are new and paths unique within this suite.'}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==value,'Frozen March2025 plan changed'
    else:p.write_text(json.dumps(value,indent=2));(OUT/'plan_created.json').write_text(json.dumps({'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'sha256':sha(p)},indent=2))
    return value
if __name__=='__main__':plan()
