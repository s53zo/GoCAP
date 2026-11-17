"""Independently reconcile saved predictions, holdouts, metrics and database."""
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd
from collect import DATA
from experiment import OUT,held_station

def verify():
    run=json.loads((OUT/'run.json').read_text())
    assert run['status']=='complete'
    for key,path in [('hourly_sha256',DATA/'hourly.parquet'),('predictions_sha256',DATA/'predictions.parquet'),('metrics_sha256',OUT/'metrics.json')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==run[key]
    h=pd.read_parquet(DATA/'hourly.parquet')
    p=pd.read_parquet(DATA/'predictions.parquet')
    m=pd.DataFrame(json.loads((OUT/'metrics.json').read_text()))
    actual=p.merge(h[['row_id','source','mode','hour','tx_sign','rx_sign','snr2500']],
        on=['row_id','source','mode','hour','tx_sign','rx_sign'],suffixes=('_prediction','_database'),validate='many_to_one')
    assert len(actual)==len(p)
    np.testing.assert_array_equal(actual.snr2500_prediction,actual.snr2500_database)
    checks=[]
    expected={('WSPRNET','WSPR'),('HARC_PSK','FT8'),('HARC_PSK','FT4'),('HARC_RBN','CW')}
    for (source,mode,fold),g in p.groupby(['source','mode','fold']):
        assert (source,mode) in expected
        assert (g.hour>=pd.Timestamp('2025-01-01',tz='UTC')).all()
        tr=h[(h.source==source)&(h['mode']==mode)&(h.hour<pd.Timestamp('2025-01-01',tz='UTC'))]
        if fold!='later_period':
            tr=tr[~tr.tx_sign.map(held_station)&~tr.rx_sign.map(held_station)]
            calls=set(tr.tx_sign)|set(tr.rx_sign)
            unseen_tx=~g.tx_sign.isin(calls);unseen_rx=~g.rx_sign.isin(calls)
            assert (unseen_tx|unseen_rx).all()
            if fold=='later_both_endpoints_unseen':assert (unseen_tx&unseen_rx).all()
        rows=m[(m.source==source)&(m['mode']==mode)&(m.fold==fold)]
        for model in ['historical','historical_power','geometry','geometry_solar']:
            row=rows[rows.model==model].iloc[0]
            assert int(row.train_rows)==len(tr) and int(row.test_rows)==len(g)
            error=g[model].to_numpy()-g.snr2500.to_numpy()
            assert np.isclose(abs(error).mean(),row.mae_db,atol=1e-12)
            assert np.isclose(np.sqrt(np.mean(error**2)),row.rmse_db,atol=1e-12)
            if isinstance(row.features,list):
                assert not {'tx_sign','rx_sign','snr2500','reports','row_id'}&set(row.features)
        if fold=='later_period':
            row=rows[rows.model=='geometry_solar_80pct_interval'].iloc[0]
            coverage=((g.snr2500>=g.q10)&(g.snr2500<=g.q90)).mean()
            assert np.isclose(coverage,row.coverage)
        checks.append({'source':source,'mode':mode,'fold':fold,'rows':len(g),'status':'reconciled'})
    assert len(checks)==12
    with sqlite3.connect(DATA/'gocap.sqlite') as con:
        sql=pd.read_sql_query('SELECT source,mode,COUNT(*) AS rows FROM positive_hourly GROUP BY source,mode',con)
    expected_counts=h.groupby(['source','mode']).size().rename('rows').reset_index()
    pd.testing.assert_frame_equal(sql.sort_values(['source','mode']).reset_index(drop=True),
        expected_counts.sort_values(['source','mode']).reset_index(drop=True),check_dtype=False)
    assert h.snr2500.notna().all()
    assert h.tx_grid.str.fullmatch('[A-R]{2}[0-9]{2}').all()
    assert h.rx_grid.str.fullmatch('[A-R]{2}[0-9]{2}').all()
    assert h.available_boundary.le(h.hour).all()
    result={'status':'verified','evaluation_groups':checks,'prediction_rows':len(p),
        'hourly_database_rows':len(h),'checks':['artifact hashes','source/target joins',
        'temporal split','unseen callsign splits','independently recomputed MAE/RMSE',
        'interval coverage','SQLite/Parquet reconciliation','locator validity','completed solar-period boundaries']}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(f'Verified {len(checks)} evaluation groups and {len(p):,} saved predictions.')

if __name__=='__main__':verify()
