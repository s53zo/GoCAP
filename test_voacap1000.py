"""Regression coverage for the expanded, separately frozen comparison."""
import unittest
import pandas as pd
from ml40.voacap_protocol import DATA,RESULTS
from ml40.voacap_prepare import select

class ExpandedSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=DATA/'voacap1000';cls.out=RESULTS/'voacap1000'
        if not (cls.out/'cases_frozen.csv').exists() or not (cls.base/'case_features.parquet').exists():raise unittest.SkipTest('Expanded benchmark not collected')
        cls.cases=pd.read_csv(cls.out/'cases_frozen.csv')
    def test_distinct_paths_and_balanced_distances(self):
        c=self.cases
        self.assertEqual(len(c),1000)
        self.assertEqual(len(set(zip(c.tx_grid,c.rx_grid))),1000)
        self.assertEqual(c.groupby('distance_group').size().tolist(),[250]*4)
        old=pd.read_csv(RESULTS/'voacap100/cases_frozen.csv')
        self.assertFalse(set(zip(c.tx_grid,c.rx_grid))&set(zip(old.tx_grid,old.rx_grid)))
    def test_identifiers_not_signal_strength_choose_cases(self):
        d=pd.concat([pd.read_parquet(p) for p in sorted((self.base/'processed').glob('*.parquet'))],ignore_index=True)
        d=d.sample(frac=1,random_state=64);d['snr2500']=-1000.
        selected=select(d,seed='gocap-voacap1000-2026-v1',count=1000)
        self.assertEqual(selected.case_id.tolist(),self.cases.case_id.tolist())
    def test_every_case_has_true_hourly_values(self):
        d=pd.read_parquet(self.base/'case_features.parquet')
        self.assertEqual(d.case_id.nunique(),1000)
        self.assertTrue(d.groupby('case_id').hour.nunique().ge(6).all())
        self.assertFalse(d.duplicated(['case_id','hour']).any())
        self.assertTrue(d.snr2500.notna().all())
if __name__=='__main__':unittest.main()
