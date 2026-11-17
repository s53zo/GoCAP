"""Evaluate once on March 2026 WSPR and replay the fixed March 2025 path benchmark."""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
import requests
from threadpoolctl import threadpool_limits

from .march_tune import OUT, TUNED, DAYS
from .march_plan import ROOT, DATA, ARTIFACT, sha
from .march_evaluate import hour_bootstrap
from .shared_train import evaluate
from .hourly_models import metric, predict_profiles
from .years_collect import process

CACHE = DATA / 'march_tune'
DEFAULT = ROOT / 'artifacts/gocap-40m-hourly.joblib'


def collect():
    CACHE.mkdir(exist_ok=True)
    manifest = {}
    for day in DAYS:
        date = f'2026-03-{day:02d}'
        raw = CACHE / f'{date}.jsonl'
        query = f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND toDate(time,'UTC')=toDate('{date}') AND cityHash64(id)%128=0
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
        if not raw.exists():
            for attempt in range(3):
                try:
                    response = requests.get('https://db1.wspr.live/', params={'query':query}, timeout=(15,90))
                    response.raise_for_status()
                    lines = response.content.splitlines()
                    assert 0 < len(lines) < 50001
                    for line in lines:
                        json.loads(line)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(5)
            temp = raw.with_suffix('.part')
            temp.write_bytes(response.content)
            temp.replace(raw)
            time.sleep(3.2)
        frame, audit = process(raw)
        assert set(frame.date_key.astype(str)) == {date}
        path = raw.with_suffix('.parquet')
        frame.to_parquet(path, index=False)
        manifest[date] = dict(query=query, raw_sha256=sha(raw), prepared_sha256=sha(path), audit=audit)
        print(date, len(frame), flush=True)
    (OUT/'sources.json').write_text(json.dumps(manifest, indent=2))


def score():
    selection = json.loads((OUT/'selection.json').read_text())
    assert sha(TUNED) == selection['sha256']
    sources = json.loads((OUT/'sources.json').read_text())
    paths = [CACHE / f'2026-03-{day:02d}.parquet' for day in DAYS]
    for path in paths:
        assert sha(path) == sources[path.stem]['prepared_sha256']
    fresh = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    assert fresh.hour.dt.year.eq(2026).all() and fresh.snr2500.notna().all()
    pred, results = {}, {}
    for name, path in [('tuned', TUNED), ('march', ARTIFACT), ('default', DEFAULT)]:
        bundle = joblib.load(path)
        pred[name] = predict_profiles(fresh, bundle['models'], bundle['profile_selection'])
        assert np.isfinite(pred[name]).all()
        results[name] = metric(fresh, pred[name])
        half = bundle['intervals'].half_width['WSPRNET|WSPR']['db']
        results[name]['coverage80'] = float((abs(pred[name] - fresh.snr2500) <= half).mean())
        results[name]['interval_half_width_db'] = half
    cells = fresh[['date_key','hour']].copy()
    for name, values in pred.items():
        cells[name] = abs(values - fresh.snr2500)
    cell_errors = cells.groupby(['date_key','hour']).mean()
    assert len(cell_errors) == 24 * len(DAYS)
    cell_errors.to_csv(OUT/'fresh_hourly_errors.csv')
    for name in pred:
        np.testing.assert_allclose(cell_errors[name].mean(), results[name]['hour_date_mae_db'])
    deltas = {name: hour_bootstrap(fresh, pred[name], pred['tuned']) for name in ['march','default']}
    obs = pd.read_parquet(DATA/'march2025voa/case_features.parquet')
    old = pd.read_csv(ROOT/'results/ml40/march2025voa/hourly_comparison.csv').dropna(subset=['actual_snr_db'])
    tuned = joblib.load(TUNED)
    obs['tuned_snr_db'] = predict_profiles(obs, tuned['models'], tuned['profile_selection'])
    obs['utc_hour'] = obs.hour.dt.hour
    replay = old.merge(obs[['case_id','utc_hour','snr2500','tuned_snr_db']], on=['case_id','utc_hour'], validate='one_to_one')
    assert len(replay) == len(old) == 4867 and replay.case_id.nunique() == 500
    np.testing.assert_allclose(replay.actual_snr_db, replay.snr2500)
    replay_scores = {}
    for col in ['tuned','march','current','voacap']:
        error = abs(replay[col+'_snr_db'] - replay.actual_snr_db)
        replay_scores[col] = error.groupby(replay.case_id).mean().mean()
    audit = pd.read_parquet(DATA/'march40/known_mode_predictions.parquet')
    other_modes = {}
    for name, values in [('tuned', predict_profiles(audit, tuned['models'], tuned['profile_selection'])),
                         ('march', audit.march_prediction), ('default', audit.baseline_prediction)]:
        other_modes[name] = {m: s['hour_date_mae_db'] for m, s in evaluate(audit, values).items()}
    summary = dict(fresh=results, paired_differences=deltas, known500_case_mae=replay_scores,
                   known_other_modes_hour_date_mae=other_modes, positive_reports=int(fresh.reports.sum()),
                   distinct_directed_grid_pairs=len(fresh[['tx_grid','rx_grid']].drop_duplicates()),
                   tuned_sha256=sha(TUNED), default_sha256=sha(DEFAULT), march_sha256=sha(ARTIFACT))
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['collect','score'])
    with threadpool_limits(limits=4):
        {'collect':collect,'score':score}[parser.parse_args().phase]()
