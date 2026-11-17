import unittest
import numpy as np
import pandas as pd
from ml40.four_models import offsets,positions,noise_terms
from ml40.four_query import issue_map
from ml40.four_prepare import power_pairs


class FourTests(unittest.TestCase):
    def test_square_marginal_does_not_read_true_location(self):
        d=pd.DataFrame({'rx_grid':['JN76'],'rx_lat':[46.5],'rx_lon':[15.]})
        pts=[positions(d,'rx','marginal',{},k) for k in range(4)]
        np.testing.assert_allclose(np.mean(pts,axis=0).ravel(),[46.5,15.])
        self.assertAlmostEqual(offsets('JN76AA')[0],-0.5+1/48)

    def test_map_latency(self):
        for stamp in ['02:00','02:05','01:30']:
            with self.assertRaisesRegex(ValueError,'5–15'):
                issue_map(f'# Product Valid At : 2026-02-01 {stamp} UTC',pd.Timestamp('2026-02-01T02:00Z'))

    def test_previous_noise_fallback(self):
        d=pd.DataFrame({'hour':pd.to_datetime(['2026-01-01T04:00Z']),'rx_grid':['JN76'],'noise_now':[np.nan]})
        norm=pd.Series([2.],index=pd.MultiIndex.from_tuples([('JN76',4)],names=['rx_grid','utc']))
        typ,live=noise_terms(d,{'typical':norm})
        self.assertEqual((typ[0],live[0]),(2.,0.))

    def test_power_step_window(self):
        times=pd.to_datetime(['2026-01-01T01:00Z','2026-01-01T01:04Z','2026-01-01T01:30Z'])
        d=pd.DataFrame(dict(time=times,hour=times.floor('h'),tx_sign='A1',rx_sign='B2',tx_loc='JN76AA',
                       rx_loc='JN68CD',tx_grid='JN76',rx_grid='JN68',power_dbm=[20,30,20],
                       snr=[-15,-8,-15],id=[1,2,3],frequency=7040100))
        p=power_pairs(d)
        self.assertEqual(list(zip(p.step,p.delta)),[(10,7)])
