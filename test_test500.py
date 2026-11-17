"""Checks of consequential 500-case selection and scoring behavior."""
import unittest
import numpy as np
import pandas as pd
from ml40.test500 import select_cases,rank_metrics,KEYS

class PropagationSuiteChecks(unittest.TestCase):
    def evidence(self):
        rows=[]
        for mode in ['WSPR','FT8']:
            for i in range(8):
                for hour in range(6 if mode=='WSPR' else 1):
                    rows.append(dict(zip(KEYS,['S',mode,f'2025-01-{1+i%2:02d}',
                        f'T{i}',f'R{i}',f'JN{i}0',f'FN{i}1']))|
                        {'hour':hour,'distance_km':500+1000*i,'snr2500':float(i-hour)})
        return pd.DataFrame(rows)

    def test_selection_is_independent_of_signal_and_input_order(self):
        d=self.evidence();q={'WSPR':4,'FT8':3}
        first=select_cases(d,q)
        changed=d.sample(frac=1,random_state=7).copy()
        changed['snr2500']=np.arange(len(changed))*1000
        second=select_cases(changed,q)
        self.assertEqual(first.case_id.tolist(),second.case_id.tolist())
        self.assertEqual(first.groupby('mode').size().to_dict(),q)
        self.assertFalse(first.duplicated(['mode','tx_grid','rx_grid']).any())

    def test_insufficient_hours_and_duplicate_square_paths_do_not_fill_quota(self):
        d=self.evidence();d=d[d['mode'].eq('WSPR')].copy()
        d.loc[d.tx_sign=='T0','hour']=0 # Six records are only one observed hour.
        selected=select_cases(d,{'WSPR':7})
        self.assertNotIn('T0',set(selected.tx_sign))
        with self.assertRaises(ValueError):select_cases(d,{'WSPR':8})
        d['tx_grid']='JN76';d['rx_grid']='FN31'
        with self.assertRaises(ValueError):select_cases(d,{'WSPR':2})

    def test_ranking_ties_missing_hours_and_constant_prediction(self):
        a=np.array([0.,0.,2.,3.,4.,5.])
        corr,loss=rank_metrics(a,a.copy())
        self.assertAlmostEqual(corr,1);self.assertEqual(loss,0)
        corr,loss=rank_metrics(a,-a)
        self.assertAlmostEqual(corr,-1);self.assertEqual(loss,5)
        self.assertEqual(rank_metrics(a[:5],a[:5]),(None,None))
        self.assertEqual(rank_metrics(a,np.ones(6)),(None,None))

if __name__=='__main__':unittest.main()
