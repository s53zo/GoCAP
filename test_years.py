"""Year balance and outcome-blind selection for the15year comparison."""
import unittest
import numpy as np,pandas as pd
from ml40.years_plan import YEARS,DATES
from ml40.years_prepare import select

class YearsTests(unittest.TestCase):
    def test_dates_are_not_original_or_march_fitting_schedule(self):
        self.assertEqual(len(DATES),60);self.assertEqual(YEARS,list(range(2010,2025)))
        from ml40.plan import dates
        from ml40.march_plan import DATES as MARCH
        self.assertFalse(set(DATES)&(set(dates())|set(MARCH)))
    def test_year_balance_uniqueness_and_signal_blindness(self):
        rows=[]
        for year in YEARS:
            for day in [6,13,20,27]:
                for path in range(30):
                    for h in range(6):rows.append({'date_key':f'{year}-04-{day:02d}','hour':pd.Timestamp(year,4,day,h,tz='UTC'),
                        'tx_sign':'TX','rx_sign':f'R{path}','tx_grid':'JN76','rx_grid':f'FN{path:02d}','power_dbm':30.,
                        'decoder_context':'ordinary_code1','distance_km':1000.,'snr2500':float(h)})
        d=pd.DataFrame(rows);cfg={'seed':'test15years'};first=select(d,cfg);d.snr2500=-999.
        second=select(d.sample(frac=1,random_state=16),cfg)
        self.assertEqual(first.case_id.tolist(),second.case_id.tolist());self.assertEqual(len(first),300)
        self.assertTrue(first.groupby('year').size().eq(20).all())
        self.assertFalse(first.duplicated(['year','tx_grid','rx_grid']).any())
    def test_insufficient_data_fails_instead_of_repeating_paths(self):
        d=pd.DataFrame([{'date_key':'2010-04-06','hour':pd.Timestamp(2010,4,6,h,tz='UTC'),'tx_sign':'T1','rx_sign':'R1',
            'tx_grid':'JN76','rx_grid':'FN31','power_dbm':30.,'decoder_context':'legacy_code0','distance_km':1000.} for h in range(6)])
        with self.assertRaises(ValueError):select(d,{'seed':'small'})
if __name__=='__main__':unittest.main()
