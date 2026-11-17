"""Absolute-hourly objective, split boundaries and prediction interfaces."""
import unittest
import numpy as np,pandas as pd
from ml40.hourly_models import metric,weights,augment
from ml40.hourly_plan import CORRECTION_DATES,TEST_DATES,active_harc_dates
from ml40.plan import dates
from ml40.tune_collect import DATES as TUNING_DATES
from ml40.v3_plan import TEST_DATES as V3_DATES
from ml40.v3_features import scenario

class HourlySNRChecks(unittest.TestCase):
    def test_metric_scores_snr_for_each_hour_not_ordering(self):
        d=pd.DataFrame({'date_key':['D']*3,'hour':pd.to_datetime(['2024-01-01T00:00Z','2024-01-01T00:00Z','2024-01-01T01:00Z']),
            'snr2500':[0.,0.,0.]})
        r=metric(d,np.array([0.,0.,10.]))
        self.assertAlmostEqual(r['hour_date_mae_db'],5.)
        self.assertAlmostEqual(r['mae_db'],10/3)
        d.snr2500=[-20.,-10.,0.]
        # Correct ordering with a 20dB offset still has a 20dB prediction error.
        self.assertAlmostEqual(metric(d,d.snr2500.to_numpy()+20)['hour_date_mae_db'],20)

    def test_dates_do_not_reuse_original_or_prior_final_test_dates(self):
        old=set(dates())|set(TUNING_DATES)|set(V3_DATES)
        self.assertFalse(set(CORRECTION_DATES)&old)
        self.assertFalse((set(TEST_DATES)|set(active_harc_dates()))&old)
        self.assertFalse(set(CORRECTION_DATES)&set(TEST_DATES))

    def test_hour_weights_balance_moderately_different_report_counts(self):
        d=pd.DataFrame({'date_key':['D']*25,'hour':[0]*5+[1]*20})
        w=weights(d,'hour')
        self.assertAlmostEqual(w[:5].sum(),w[5:].sum())
        self.assertTrue(np.isfinite(w).all())

    def test_local_time_features_are_cyclic_and_identical_for_training_query_frames(self):
        d,_=scenario('JN76','FN31','2024-06-01T00:00Z',150,3,10)
        a=augment(d)
        self.assertAlmostEqual(a.tx_local_sin.iloc[0]**2+a.tx_local_cos.iloc[0]**2,1.)
        self.assertAlmostEqual(a.rx_local_sin.iloc[0]**2+a.rx_local_cos.iloc[0]**2,1.)
        d['tx_lon']=0 # Cached legacy column must not silently override the grid coordinates.
        b=augment(d)
        self.assertAlmostEqual(a.tx_local_sin.iloc[0],b.tx_local_sin.iloc[0])

if __name__=='__main__':unittest.main()
