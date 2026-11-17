"""Frozen five-improvement experiment; legacy test scores never select candidates."""
import json
from .plan import ROOT,DATA,RESULTS
from .test500 import digest
BASE=DATA/'v3';OUT=RESULTS/'v3';BASE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
TRAIN_DATES=sorted([f'{y}-{m:02d}-15' for y in [2010,2013,2016,2019,2022,2024] for m in range(1,13)]+
                   [f'{y}-{m:02d}-01' for y in [2022,2024] for m in range(1,13)])
TEST_DATES=[f'2025-{m:02d}-03' for m in range(1,13)]

def plan():
    value={'version':3,'seed':40,'band_m':40,'positive_only':True,
        'coherent_train_dates':TRAIN_DATES,'fresh_test_dates':TEST_DATES,
        'pair_day_divisors':[128,256,512],'snr_event_divisor':128,'raw_query_cap':50001,
        'training_rank_group_min_observed_hours':4,'evaluation_rank_group_min_observed_hours':6,
        'regression_candidates':['scattered_wspr_base','coherent_wspr_base','coherent_wspr_rich','pooled_rich','pooled_source_heads'],
        'regressor_settings':{'max_iter':180,'learning_rate':.07,'max_leaf_nodes':23,'min_samples_leaf':60,'l2_regularization':10},
        'ranker_settings':{'n_estimators':[120,240,400],'max_depth':[3,5],'learning_rate':.04,
            'objective':'rank:pairwise','lambdarank_pair_method':'mean','lambdarank_num_pair_per_sample':8},
        'selection':'WSPR SNR: lowest validation pair/day/day-balanced MAE; pooled model: lowest equally source/mode-balanced validation MAE; ranker: highest validation median observed-hour Spearman; quantile intervals: fixed 10th/90th quantiles, then separate per-source/mode split conformal calibration.',
        'promotion':'Point-SNR improvement: paired date bootstrap upper bound of MAE difference <0 and each original mode regression MAE <= reference *1.05. Ranking improvement: paired date-bootstrap lower bound of median correlation difference >0, best-observed-hour loss <= reference *1.05. Interval improvement: fresh coverage between .75 and .85 and interval score lower than reference. Components assessed separately; failure retains reference recommendation.',
        'feature_ablation':'Same 96 train dates: scattered vs coherent base; coherent base vs rich; WSPR rich vs pooled rich; pooled vs source heads.',
        'solar_history':'Completed 3-hour ap history, preceding 6/24/72h summaries and capped storm recovery; previous completed-day SFI and preceding27-day SFI. Arbitrary scenarios default to explicitly steady history, with optional timestamp-checked history.',
        'station_calibration':'Implement documented positive-measurement intake/audit; no physical dipole calibration without independently supplied actual reference observations.',
        'reference_sha256':digest(ROOT/'artifacts/gocap-40m.joblib'),
        'fit_sha256':digest(DATA/'fit_features.parquet')}
    p=OUT/'plan.json'
    if p.exists() and json.loads(p.read_text())!=value:raise ValueError('Frozen v3 plan differs')
    p.write_text(json.dumps(value,indent=2));return value
if __name__=='__main__':plan()
