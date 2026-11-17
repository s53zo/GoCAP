"""Check tuning split, parameter coverage and observed-hour ranking invariants."""
import unittest
import numpy as np
import pandas as pd
from ml40.tune import parameter_trials,SPACE,REFERENCE,RankPanel,eligible
from ml40.train import ranking
from ml40.tune_collect import DATES
from ml40.plan import dates

class TuningChecks(unittest.TestCase):
    def test_all_six_parameters_covered_with_unique_trials(self):
        trials=parameter_trials()
        self.assertEqual(len(trials),37)
        self.assertEqual(trials[0],REFERENCE)
        self.assertEqual(len({tuple(x.items()) for x in trials}),37)
        for key,values in SPACE.items():self.assertEqual(set(x[key] for x in trials[1:]),set(values))
        self.assertEqual(len(DATES),12)
        self.assertFalse(set(DATES)&set(dates()))

    def test_vectorized_ranking_matches_existing_observed_hour_implementation(self):
        rows=[]
        for date in ['2024-01-08','2024-04-08']:
            for hour in range(8):
                for power in [30.,37.]:
                    rows.append({'source':'WSPRNET','mode':'WSPR','date_key':date,
                        'hour':pd.Timestamp(date,tz='UTC')+pd.Timedelta(hours=hour),
                        'tx_sign':'T','rx_sign':'R','tx_grid':'JN76','rx_grid':'FN31',
                        'power_dbm':power,'snr2500':float(hour%3)+power-30})
        d=pd.DataFrame(rows);p=d.snr2500.to_numpy()+np.arange(len(d))*.03
        got=RankPanel(d).score(p)
        expected=pd.DataFrame(ranking(d,p,'population_core'))
        expected=expected[expected.unit.eq('station_pair')]
        np.testing.assert_allclose(sorted(got.spearman),sorted(expected.spearman))
        np.testing.assert_allclose(sorted(got.best_hour_shortfall_db),sorted(expected.best_observed_hour_shortfall_db))
        d.loc[d.hour.dt.hour>=4,'tx_grid']='JN77'
        self.assertEqual(len(RankPanel(d).h),0)

    def test_mae_and_temporal_guards_reject_tradeoffs(self):
        reference={'balanced_mae_db':7.,'mode_mae_db':{'WSPR':6.,'FT8':7.},
            'quarter_rank_median':{'1':.4,'2':.4,'3':.4,'4':.4}}
        self.assertTrue(eligible(reference,reference))
        import copy
        bad=copy.deepcopy(reference);bad['balanced_mae_db']=7.2
        self.assertFalse(eligible(bad,reference))
        bad=copy.deepcopy(reference);bad['mode_mae_db']['FT8']=8
        self.assertFalse(eligible(bad,reference))
        bad=copy.deepcopy(reference);bad['quarter_rank_median'].update({'1':.2,'2':.2})
        self.assertFalse(eligible(bad,reference))

class FreshEvaluationChecks(unittest.TestCase):
    def test_date_bootstrap_preserves_paired_constant_difference(self):
        from ml40.tune_evaluate import bootstrap_rank
        a=pd.DataFrame({'case_index':range(6),'date':['D1']*2+['D2']*2+['D3']*2,
            'spearman':[.1,.2,.3,.4,.5,.6],'best_hour_shortfall_db':[1.]*6})
        b=a.copy();b['spearman']+=.1
        interval,paired=bootstrap_rank(a,b,replicates=100)
        self.assertEqual(len(paired),6)
        self.assertEqual(interval['independent_date_clusters'],3)
        np.testing.assert_allclose(interval['95_percent_date_bootstrap_interval'],[.1,.1],atol=1e-12)

if __name__=='__main__':unittest.main()
