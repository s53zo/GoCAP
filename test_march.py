"""March expansion coverage, row identity and bounded-encoder correctness."""
import unittest
import numpy as np,pandas as pd
from ml40.march_plan import YEARS,DATES,FRESH_TEST
from ml40.march_prepare import canonical_keys
from ml40.march_models import encoder_from_medians
from ml40.v3_models import Encoder

class MarchTests(unittest.TestCase):
    def test_every_year_and_day_with_no_future_fitting(self):
        self.assertEqual(YEARS,list(range(2010,2025)));self.assertEqual(len(DATES),465)
        for year in YEARS:self.assertEqual(sum(d.startswith(str(year)) for d in DATES),31)
        self.assertFalse(set(DATES)&set(FRESH_TEST));self.assertTrue(all(d.startswith('2025-03-') for d in FRESH_TEST))
        from ml40.plan import dates
        from ml40.shared_plan import TEST_DATES
        from ml40.hourly_plan import TEST_DATES as HOUR_DATES
        self.assertFalse(set(FRESH_TEST)&(set(dates())|set(TEST_DATES)|set(HOUR_DATES)))
    def test_row_identity_ignores_timestamp_storage_unit_but_retains_power(self):
        d=pd.DataFrame({'source':['WSPRNET'],'mode':['WSPR'],'hour':pd.to_datetime(['2024-03-01T12:00Z']),
            'tx_sign':['TX'],'rx_sign':['RX'],'tx_grid':['JN76'],'rx_grid':['FN31'],'power_dbm':[30.]})
        a=canonical_keys(d);other=d.copy();other['hour']=other.hour.astype('datetime64[ns, UTC]')
        self.assertEqual(a.tolist(),canonical_keys(other).tolist());other.power_dbm=37.
        self.assertNotEqual(a.tolist(),canonical_keys(other).tolist())
    def test_bounded_global_median_encoder_matches_original(self):
        d=pd.DataFrame({'a':[1.,np.nan,9.,2.],'b':[np.nan]*4,'c':[5.,6.,7.,np.nan],
            'source':['WSPRNET','HARC_PSK','WSPRNET','HARC_PSK'],'mode':['WSPR','FT8','WSPR','FT8'],'decoder_context':['ordinary_code1','archive_mode_known']*2})
        numeric=['a','b','c'];original=Encoder(True);original.numeric=numeric;original.fit(d)
        with np.errstate(all='ignore'):
            medians=np.array([np.nanmedian(d[c]) if d[c].notna().any() else np.nan for c in numeric])
        bounded=encoder_from_medians(medians,numeric,d)
        np.testing.assert_allclose(original.transform(d),bounded.transform(d),atol=1e-7)
if __name__=='__main__':unittest.main()
