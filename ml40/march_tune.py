"""Fit and validate a higher-capacity March model, retaining the original as a candidate."""
import copy
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from .march_plan import ARTIFACT, BASE, DATA, ROOT, sha
from .hourly_models import HourlyIntervals
from .shared_train import evaluate

OUT = ROOT / 'results/ml40/march_tune'
TUNED = ROOT / 'artifacts/gocap-40m-shared-march-tuned.joblib'
DAYS = [2, 6, 10, 14, 18, 22, 26, 30]
STAGES = (120, 240, 400)


def run():
    OUT.mkdir(exist_ok=True)
    original = json.loads((ROOT / 'results/ml40/march40/training_run.json').read_text())
    train_path = BASE / 'expanded_train_features.parquet'
    assert sha(ARTIFACT) == original['model_sha256']
    assert sha(train_path) == original['training_data_sha256']
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(DATA / 'hourly/validation_features.parquet')
    cal = pd.read_parquet(DATA / 'hourly/calibration_features.parquet')
    assert train.hour.dt.year.max() == 2024
    assert not set(train.date_key) & (set(val.date_key) | set(cal.date_key))
    bundle = joblib.load(ARTIFACT)
    model = bundle['models']['shared']
    matrix = np.load(BASE / 'matrix/encoded.npy', mmap_mode='r')
    rows = np.random.default_rng(40).choice(len(train), 256, replace=False)
    assert len(matrix) == len(train)
    np.testing.assert_allclose(matrix[rows], model.encoder.transform(model.frame(train.iloc[rows])))
    target = train.snr2500.to_numpy(copy=True)
    del train
    xval = model.encoder.transform(model.frame(val))
    old = evaluate(val, model.model.predict(xval))
    wspr = 'WSPR'
    best = old[wspr]['hour_date_mae_db']
    selected = 'original'
    scores = {'original': old}
    higher = clone(model.model).set_params(max_iter=max(STAGES), max_leaf_nodes=63,
        learning_rate=.08, min_samples_leaf=80, warm_start=False)
    print('Fitting 400 rounds on', len(target), 'observations', flush=True)
    higher.fit(matrix, target, sample_weight=np.ones(len(target)))
    for step, pred in enumerate(higher.staged_predict(xval), 1):
        if step not in STAGES:
            continue
        score = evaluate(val, pred)
        scores[str(step)] = score
        error = score[wspr]['hour_date_mae_db']
        passes = all(score[m]['hour_date_mae_db'] <= 1.05 * old[m]['hour_date_mae_db']
                     for m in old if m != wspr)
        if passes and error < best:
            best, selected = error, str(step)
            chosen = copy.deepcopy(higher)
            chosen._predictors = chosen._predictors[:step]
            chosen.set_params(max_iter=step)
            np.testing.assert_allclose(chosen.predict(xval), pred)
    if selected != 'original':
        model.model = chosen
    bundle['version'] = 'shared-march-tuned-1'
    bundle['selection_objective'] = 'Minimum validation WSPR hour/date MAE with 5% per-mode degradation guard.'
    bundle['intervals'] = HourlyIntervals().fit(cal, bundle['models'], bundle['profile_selection'])
    joblib.dump(bundle, TUNED)
    metadata = dict(selected=selected, scores=scores, sha256=sha(TUNED),
                    training_sha256=sha(train_path), original_sha256=sha(ARTIFACT),
                    validation_sha256=sha(DATA/'hourly/validation_features.parquet'),
                    calibration_sha256=sha(DATA/'hourly/calibration_features.parquet'),
                    training_rows=len(target), validation_dates=sorted(set(val.date_key.astype(str))),
                    fresh_march_2026_days=DAYS)
    (OUT / 'selection.json').write_text(json.dumps(metadata, indent=2))
    print('Frozen', selected, best, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run()
