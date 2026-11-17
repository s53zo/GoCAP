"""Frozen hour-by-hour SNR experiment. Ranking never selects a model."""
import json,hashlib
from .plan import ROOT,DATA,RESULTS
def digest(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
BASE=DATA/'hourly';OUT=RESULTS/'hourly';BASE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
CORRECTION_DATES=[f'{y}-{m:02d}-05' for y in [2023,2024] for m in range(1,13)]
TEST_DATES=[f'2025-{m:02d}-04' for m in range(1,13)]
HARC_TEST_DATES=['2025-04-04','2025-07-04','2025-10-04']

def active_harc_dates():
    amendment=OUT/'availability_amendment.json'
    return json.loads(amendment.read_text())['active_harc_test_dates'] if amendment.exists() else HARC_TEST_DATES

def plan():
    value={'version':'hourly-snr-1','seed':40,'band_m':40,'positive_only':True,
        'primary_metric':'Mean absolute SNR error averaged equally over observed UTC hour/date cells; model selection separately per observation mode. No ranks or best-hour scores.',
        'secondary_metrics':['observation MAE','RMSE','bias','24 separate UTC-hour errors','80% interval coverage/width/score'],
        'correction_training_dates':CORRECTION_DATES,'new_wspr_test_dates':TEST_DATES,'new_harc_test_dates':HARC_TEST_DATES,
        'wspr_event_hash_divisor':128,'query_cap':50001,'harc_file_cap_gb':8,'harc_max_files':3,
        'candidate_settings':{'iterations':240,'learning_rate':.05,'max_leaf_nodes':31,'min_samples_leaf':100,'l2_regularization':10},
        'native_candidates':['native_base_day','native_rich_day','native_rich_hour','native_rich_observation','native_rich_small'],
        'native_policy':'Source-specific native SNR targets; reported WSPR power stays a predictor, never replaced by an assumed 1dB/dB correction. Unknown power stays unknown for other modes.',
        'correction_candidates':['independent_residual_half','independent_residual_full'],
        'correction_policy':'Reference is frozen. Residuals are learned only on 24 new 2023/2024 dates absent from its fitting data; small regularized trees; correction clipped to +/-3dB, tested at fixed strengths .5 and 1.',
        'selection':'Lowest validation hour/date-cell MAE, with observation MAE <= reference*1.02 and RMSE <= reference*1.03; select independently per mode. Existing 2024 validation used only for selection. Separate calibration dates adjust intervals.',
        'promotion':'Fresh-date paired bootstrap upper bound of hour/date-cell MAE difference <0 for WSPR; micro MAE <=reference, RMSE<=reference*1.02; each other fresh mode point MAE<=reference*1.05. Otherwise retain reference point model.',
        'uncertainty':'Separate per-mode finite-sample 80% absolute residual calibration; compare measured coverage/width/interval score on new data. This is reported-reception uncertainty, not closed/open-path evidence.',
        'strict_tests':'Refit selected mode models with designated stations or fields excluded in both roles; frozen strict reference artifacts supply comparators.',
        'calibration':'Archive-only; physical 100W/dipole calibration unverified by user choice.',
        'reference_sha256':digest(ROOT/'artifacts/gocap-40m.joblib'),'fit_sha256':digest(DATA/'fit_features.parquet'),
        'diagnosis_sha256':digest(OUT/'validation_diagnostics.csv')}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen hourly plan changed')
    path.write_text(json.dumps(value,indent=2));return value
if __name__=='__main__':plan()
