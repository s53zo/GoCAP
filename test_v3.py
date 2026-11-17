"""Consequential checks for geometry, history boundaries, positive ranking and calibration."""
import unittest
import numpy as np
import pandas as pd
from ml40.v3_features import scenario,geometry,enrich,solar_history,HISTORY,PATH
from ml40.v3_models import rank_training,shift
from ml40.v3_calibration import validate
from ml40.v3_plan import TRAIN_DATES,TEST_DATES
from ml40.plan import dates
from ml40.tune_collect import DATES as TUNING_TEST

class V3Checks(unittest.TestCase):
    def test_reserved_dates_are_untouched_and_disjoint(self):
        self.assertEqual(len(TRAIN_DATES),96);self.assertEqual(len(TEST_DATES),12)
        self.assertFalse(set(TEST_DATES)&(set(dates())|set(TUNING_TEST)|set(TRAIN_DATES)))

    def test_geometry_reversal_and_coincident_path(self):
        a,_=scenario('JN76','FN31','2024-06-01T12:00Z',150,3,10)
        b,_=scenario('FN31','JN76','2024-06-01T12:00Z',150,3,10)
        for col in ['path_daylight_fraction','path_sun_mean','path_max_abs_magnetic_lat']:
            self.assertAlmostEqual(a[col].iloc[0],b[col].iloc[0])
        c,_=scenario('JN76','JN76','2024-06-01T12:00Z',150,3,10)
        self.assertTrue(np.isfinite(c[PATH].drop(columns=['tx_hours_from_sunrise','tx_hours_from_sunset','rx_hours_from_sunrise','rx_hours_from_sunset'])).all().all())
        self.assertEqual(c.antipodal_path.iloc[0],0)

    def test_explicit_history_matches_steady_scenario_and_rejects_future(self):
        a,_=scenario('JN76','FN31','2024-06-01T13:00Z',150,3,10)
        h={'ending_boundary_utc':'2024-06-01T12:00Z','kp_3h':[3]*24,'sfi_previous27_days':[150]*27}
        b,_=scenario('JN76','FN31','2024-06-01T13:00Z',150,3,10,history=h)
        np.testing.assert_allclose(a[HISTORY].to_numpy(float),b[HISTORY].to_numpy(float))
        h['ending_boundary_utc']='2024-06-01T15:00Z'
        with self.assertRaises(ValueError):scenario('JN76','FN31','2024-06-01T13:00Z',150,3,10,history=h)
        with self.assertRaises(ValueError):scenario('JN76','FN31','2024-06-01T13:00Z',150,3,10,history={'kp_3h':[3]*23})

    def test_historical_features_use_completed_blocks_and_previous_days(self):
        from collect import RAW
        if not (RAW/"gfz.txt").exists():self.skipTest("Fetch GFZ indices with collect.py --source gfz")
        from experiment import solar_table
        blocks,daily=solar_table('2009')
        timestamp=pd.Timestamp('2024-06-01T13:00Z');end=timestamp.floor('3h')
        kp=blocks.loc[end-pd.Timedelta(hours=69):end,'kp_previous'].tolist()
        sfi=daily.loc[timestamp.floor('D')-pd.Timedelta(days=26):timestamp.floor('D'),'sfi_previous_day'].tolist()
        frame,_=scenario('JN76','FN31',timestamp,sfi[-1],kp[-1],10,
            history={'ending_boundary_utc':end.isoformat(),'kp_3h':kp,'sfi_previous27_days':sfi})
        actual=enrich(frame)
        # GFZ's rounded Kp values imply tiny differences when converted back to ap.
        np.testing.assert_allclose(actual[HISTORY].to_numpy(float),frame[HISTORY].to_numpy(float),atol=.03)
        self.assertEqual(actual.hours_since_ap48_capped72.iloc[0],frame.hours_since_ap48_capped72.iloc[0])

    def test_pairwise_labels_only_compare_received_hours_at_known_power(self):
        rows=[]
        for hour,power,snr in [(1,30,-20),(3,40,-7),(7,30,-10),(9,30,-5)]:
            d,_=scenario('JN76','FN31',f'2024-06-01T{hour:02d}:00Z',150,3,10,power_watts=10**((power-30)/10))
            d['snr2500']=snr;d['date_key']='2024-06-01';d['reports']=1
            rows.append(d)
        out=rank_training(pd.concat(rows,ignore_index=True))
        self.assertEqual(len(out),4);self.assertEqual(out.hour.dt.hour.tolist(),[1,3,7,9])
        self.assertEqual(out.relevance.tolist(),[0,1,2,3])
        bad=pd.concat(rows,ignore_index=True);bad.loc[0,'power_dbm']=np.nan
        self.assertTrue(rank_training(bad).empty)

    def test_reference_intake_does_not_claim_calibration_from_incomplete_data(self):
        clean,errors=validate([{'id':'incomplete','received':True}])
        self.assertFalse(clean);self.assertEqual(len(errors),1)
        clean,errors=validate([]);self.assertFalse(clean);self.assertFalse(errors)

if __name__=='__main__':unittest.main()
