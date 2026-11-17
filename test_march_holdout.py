"""Additional March dates and outcome-blind distance-balanced case selection."""
import unittest
import pandas as pd
from ml40.march_test_plan import DAYS,DATES,GROUPS
from ml40.march_test_prepare import select

class MarchHoldoutTests(unittest.TestCase):
    def test_excludes_all_previously_used_march_days(self):
        self.assertEqual(len(DATES),15);self.assertTrue(set(DAYS).isdisjoint({1,2,3,4,5,6,7,8,9,12,15,17,20,22,25,28}))
        from ml40.march_plan import FRESH_TEST,DATES as FIT_MARCH
        self.assertFalse(set(DATES)&(set(FRESH_TEST)|set(FIT_MARCH)))
        self.assertTrue(all(d.startswith('2025-03-') for d in DATES))
    def test_blind_selection_has500_distinct_paths_and125_per_bin(self):
        rows=[]
        for group,distance in enumerate([500,2000,4000,8000]):
            for path in range(150):
                date=DATES[path%15]
                for hour in range(6):rows.append({'date_key':date,'hour':pd.Timestamp(date,tz='UTC')+pd.Timedelta(hours=hour),
                    'tx_sign':f'T{group}_{path}','rx_sign':f'R{path}','tx_grid':f'JN{group}{path//100}','rx_grid':f'FN{path%100:02d}',
                    'power_dbm':30.,'decoder_context':'ordinary_code1','distance_km':distance,'snr2500':float(hour)})
        d=pd.DataFrame(rows);first=select(d,{'seed':'unit-march500'});d.snr2500=1000.
        other=select(d.sample(frac=1,random_state=31),{'seed':'unit-march500'})
        self.assertEqual(first.case_id.tolist(),other.case_id.tolist());self.assertEqual(len(first),500)
        self.assertFalse(first.duplicated(['tx_grid','rx_grid']).any());self.assertEqual(first.groupby('distance_group').size().tolist(),[125]*4)
if __name__=='__main__':unittest.main()
