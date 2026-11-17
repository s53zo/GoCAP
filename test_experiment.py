"""Checks for the pilot's consequential geography, time and holdout invariants."""
import unittest
import numpy as np
import pandas as pd
from experiment import centre, grid4, solar_table, held_station, Historical

class ExperimentInvariants(unittest.TestCase):
    def test_maidenhead_centres_and_invalid_locations(self):
        self.assertEqual(centre('JN76'), (46.5,15))
        self.assertEqual(centre('AA00'), (-89.5,-179))
        self.assertEqual(grid4('jo21be70'), 'JO21')
        for bad in ['S5','JO2','ZZ99','JN76!','JN76YY','',None]:
            self.assertIsNone(grid4(bad))

    def test_bandwidth_conversion(self):
        self.assertAlmostEqual(10*np.log10(2500/500),6.98970004336)

    def test_sampling_preserves_large_integer_hash_bits(self):
        from collect_harc import hash_selection
        hashes=np.arange(2**63,2**63+8192,dtype=np.uint64)
        selected=hash_selection(hashes,np.full(len(hashes),1024,dtype=np.int64))
        self.assertEqual(int(selected.sum()),8)

    def test_solar_indices_use_completed_periods(self):
        from collect import RAW
        if not (RAW/"gfz.txt").exists():self.skipTest("Fetch GFZ indices with collect.py --source gfz")
        blocks,daily=solar_table()
        midnight=pd.Timestamp('2024-05-11',tz='UTC')
        prevday=pd.Timestamp('2024-05-10',tz='UTC')
        from collect import RAW
        raw=next(line.split() for line in (RAW/'gfz.txt').read_text().splitlines() if line.startswith('2024 05 10 '))
        self.assertEqual(blocks.loc[midnight].kp_previous,float(raw[14]))
        # At midnight, 24h average of completed ap blocks approximates prior Ap.
        self.assertAlmostEqual(blocks.loc[midnight].ap_trailing24,daily.loc[midnight].Ap_previous_day,delta=1)
        self.assertNotEqual(daily.loc[midnight].Ap_previous_day,daily.loc[prevday].Ap_previous_day)
        self.assertLess(blocks.index[blocks.index<=midnight+pd.Timedelta(hours=2)].max(),midnight+pd.Timedelta(hours=2))

    def test_station_holdout_excludes_both_roles(self):
        calls=[f'STATION{i}' for i in range(100)]
        tx=pd.Series(calls);rx=pd.Series(calls[1:]+calls[:1])
        held={c for c in calls if held_station(c)}
        mask=~tx.isin(held)&~rx.isin(held)
        self.assertGreater(len(held),0)
        self.assertGreater(mask.sum(),0)
        self.assertFalse((set(tx[mask])|set(rx[mask]))&held)

    def test_historical_fallback_and_no_test_target_access(self):
        train=pd.DataFrame({'distance_bin':[1]*10,'mid_daylight':[0]*10,'quarter':[1]*10,
            'utc_hour':[0]*10,'tx_grid':['JN76']*10,'rx_grid':['FN31']*10,'snr2500':range(10)})
        model=Historical().fit(train)
        test=train.iloc[:1].drop(columns='snr2500').copy()
        test['tx_grid']='AA00';test['rx_grid']='RR99'
        self.assertEqual(model.predict(test)[0],4.5)

    def test_historical_lookup_uses_reported_power(self):
        train=pd.DataFrame({'distance_bin':[1]*10,'mid_daylight':[0]*10,'quarter':[1]*10,
            'utc_hour':[0]*10,'tx_grid':['JN76']*10,'rx_grid':['FN31']*10,
            'power_dbm':[30]*10,'snr2500':[-20]*10})
        test=train.iloc[:1].drop(columns='snr2500').copy()
        test['power_dbm']=40
        self.assertEqual(Historical(power_adjust=True).fit(train).predict(test)[0],-10)

    def test_harc_timestamp_and_rbn_measurements(self):
        from collect import DATA
        for date,offset in [('2024-01-15',21600),('2025-01-15',0)]:
            path=DATA/f'harc-{date}.parquet'
            if not path.exists():
                self.skipTest('Historical HARC samples have not been collected')
            h=pd.read_parquet(path)
            delta=h.ut1_unix-h.time.map(lambda t:int(t.timestamp()))
            self.assertTrue(delta.eq(offset).all())
            cw=h.source.eq('HARC_RBN') & h['mode'].eq('CW')
            self.assertGreater(cw.sum(),100)
            np.testing.assert_allclose(h.loc[cw,'snr2500'],h.loc[cw,'native_snr']-10*np.log10(5))
            unknown=h['mode'].eq('UNVERIFIED_RBN_MODE')
            self.assertTrue(h.loc[unknown,'snr2500'].isna().all())

if __name__=='__main__':
    unittest.main()
