"""Consequential invariants for sampling, splits and station adjustment."""
import unittest
import numpy as np
import pandas as pd
from ml40.plan import dates,split,station_held,geographic_held
from ml40.models import StationEffects,sample_weights

class ML40Checks(unittest.TestCase):
    def test_frozen_dates_do_not_reuse_pilot_final_test_days(self):
        all_dates=dates()
        self.assertEqual(len(all_dates),324)
        tests=[d for d in all_dates if split(d)=='test']
        self.assertEqual(len(tests),36)
        self.assertTrue(all(d[:4]=='2025' and d[-2:] in ['01','08','22'] for d in tests))
        self.assertTrue(all(d[:4]!='2025' for d in all_dates if split(d)=='train'))

    def test_two_way_station_effects_and_unknown_reference(self):
        rows=[];target=[]
        for day in range(4):
            for _ in range(10):
                for tx,tb in [('T1',5),('T2',-5)]:
                    for rx,rb in [('R1',3),('R2',-3)]:
                        rows.append({'tx_key':tx,'rx_key':rx,'date_key':str(day)})
                        target.append(tb+rb)
        d=pd.DataFrame(rows)
        model=StationEffects(penalty=1,rounds=12).fit(d,target,np.ones(len(d)))
        self.assertLess(np.mean(abs(model.predict(d)-target)),.2)
        unknown=pd.DataFrame({'tx_key':['NEW'],'rx_key':['OTHER']})
        self.assertEqual(model.predict(unknown)[0],0)
        rare=pd.DataFrame({'tx_key':['RARE'],'rx_key':['LONE'],'date_key':['day']})
        empty=StationEffects().fit(rare,[20],[1])
        self.assertEqual(empty.predict(rare)[0],0)

    def test_pair_day_weight_balancing_does_not_reward_chatty_pair(self):
        records=[]
        for tx,n in [('A',20),('B',2)]:
            for hour in range(n):records.append({'source':'S','mode':'M','date_key':'D',
                'tx_sign':tx,'rx_sign':'R','hour':hour})
        d=pd.DataFrame(records);w=sample_weights(d)
        self.assertTrue(np.isfinite(w).all())
        self.assertAlmostEqual(w[d.tx_sign=='A'].sum(),w[d.tx_sign=='B'].sum())

    def test_geographic_holdout_is_shared_by_squares_in_same_field(self):
        self.assertEqual(geographic_held('JN76'),geographic_held('JN11'))

    def test_standalone_inference_matches_training_geometry(self):
        from collect import RAW
        if not (RAW/"gfz.txt").exists():self.skipTest("Fetch GFZ indices with collect.py --source gfz")
        from ml40.inference import scenario_frame
        from experiment import features,BASE
        d=scenario_frame('JN76','FN31','2024-06-01T12:00Z',150,3,10)
        reference=features(d[['source','mode','hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']])
        np.testing.assert_allclose(d[BASE].to_numpy(float),reference[BASE].to_numpy(float),atol=1e-9)

    def test_strict_scenarios_exclude_keys_before_fitting(self):
        from ml40.train import scenario_rows
        d=pd.DataFrame({'split':['train','train','validation','validation','test'],
            'station_held':[True,False,True,False,True],
            'geographic_held':[False,True,False,True,True]})
        self.assertFalse(scenario_rows(d,'station_holdout','train').station_held.any())
        self.assertFalse(scenario_rows(d,'geographic_holdout','validation').geographic_held.any())
        self.assertTrue(scenario_rows(d,'station_holdout','test').station_held.all())

    def test_hour_ranking_accepts_neural_float32_predictions(self):
        from ml40.train import ranking
        d=pd.DataFrame({'source':['WSPRNET']*6,'mode':['WSPR']*6,'date_key':['2024-01-01']*6,
            'hour':pd.date_range('2024-01-01',periods=6,freq='h',tz='UTC'),
            'tx_sign':['T1']*6,'rx_sign':['R1']*6,'tx_grid':['JN76']*6,'rx_grid':['FN31']*6,
            'snr2500':[-20.,-15.,-10.,-5.,0.,5.],'power_dbm':[37.]*6})
        out=ranking(d,np.array([-20,-15,-10,-5,0,5],dtype=np.float32),'population_core')
        self.assertEqual(len(out),2)
        self.assertAlmostEqual(out[0]['spearman'],1)
        self.assertEqual(out[0]['best_observed_hour_shortfall_db'],0)
        # A moving transmitter is not six hours on one fixed geographic path.
        d.loc[d.index>=3,'tx_grid']='JN77'
        self.assertEqual(ranking(d,np.arange(6,dtype=np.float32),'population_core'),[])

if __name__=='__main__':unittest.main()
