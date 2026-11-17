"""Reconstruct the scored test targets and summary metrics independently."""
import json
import joblib
import numpy as np
import pandas as pd

from experiment import grid4
from .five_data import DATES, CACHE, OUT, ROOT, sha
from .five_prepare import CASE, DEFAULT
from .five_evaluate import ART


def verify():
    selection = json.loads((OUT/'selection.json').read_text())
    assert sha(ART)==selection['sha256'] and sha(DEFAULT)==selection['base_sha256']
    assert len(set(sum(DATES.values(),[])))==46
    assert max(DATES['train']+DATES['validation']) < min(DATES['test'])
    baseline_dates = pd.read_parquet(ROOT/'data/ml40/march40/expanded_train_features.parquet', columns=['date_key']).date_key
    assert set(baseline_dates).isdisjoint(sum(DATES.values(),[]))
    sources = json.loads((OUT/'sources.json').read_text())
    sonde = json.loads((OUT/'sonde_sources.json').read_text())
    for date in sources:
        assert sha(CACHE/'raw'/f'{date}.jsonl')==sources[date]['sha256']
        assert sha(CACHE/'sonde'/f'{date}.html')==sonde[date]['sha256']
        assert sha(CACHE/'features'/f'{date}.parquet')==selection['feature_hashes'][date]
    test = pd.read_parquet(OUT/'test_predictions.parquet')
    saved = pd.read_csv(OUT/'metrics.csv',index_col='model')
    pack = joblib.load(ART)
    before = pd.Timedelta(hours=1)
    for date in DATES['test']:
        frame = pd.read_parquet(CACHE/'features'/f'{date}.parquet')
        raw = pd.read_json(CACHE/'raw'/f'{date}.jsonl',lines=True)
        raw = raw[raw.code.eq(1) & raw.snr.between(-60,100) & raw.frequency.between(7e6,7.3e6)].copy()
        raw['tx_grid'],raw['rx_grid'] = raw.tx_loc.map(grid4),raw.rx_loc.map(grid4)
        raw = raw.dropna(subset=['tx_grid','rx_grid']).drop_duplicates(['time','tx_sign','rx_sign','frequency'])
        raw['hour'] = pd.to_datetime(raw.time,utc=True).dt.floor('h')
        raw['power_dbm'] = raw.power
        actual = raw.groupby(CASE+['hour']).snr.agg(['median','size']).reset_index()
        records = frame.merge(actual,on=CASE+['hour'],validate='one_to_one')
        assert len(records)==len(frame)
        np.testing.assert_array_equal(records.snr2500,records['median'])
        np.testing.assert_array_equal(records.reports,records['size'])
        stored = test[test.hour.dt.strftime('%Y-%m-%d').eq(date)]
        joined = frame[['case','hour','snr2500']].merge(stored,on=['case','hour'],validate='one_to_one')
        np.testing.assert_array_equal(joined.snr2500_x,joined.snr2500_y)
        assert len(joined)==len(frame)==len(stored)
        for name,model in pack['models'].items():
            np.testing.assert_allclose(model.predict(frame),joined[name])
    for name,metrics in saved.iterrows():
        err = abs(test[name]-test.snr2500)
        np.testing.assert_allclose(err.groupby(test.hour).mean().mean(),metrics.hour_mae)
        np.testing.assert_allclose(err.groupby(test.case).mean().mean(),metrics.path_mae)
        order = test.sort_values(['case','hour'])
        changes = order.groupby('case')[[name,'snr2500']].diff()
        use = order.groupby('case').hour.diff().eq(before)
        delta = abs(changes[name]-changes.snr2500)[use]
        np.testing.assert_allclose(delta.groupby(order.hour.dt.date[use]).mean().mean(),metrics.change_mae)
    print('Verified all 46 file sets, held-out targets and counts, artifact predictions and three metrics for all models.')


if __name__=='__main__':
    verify()
