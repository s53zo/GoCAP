"""Position, time, sample-selection and shared-mode invariants."""
import unittest
import numpy as np,pandas as pd
from ml40.shared_features import point_features,POINTS,PERCENT
from ml40.v3_features import scenario
from ml40.shared_models import training_weights

class SharedFeatureTests(unittest.TestCase):
    def test_positions_reversal_and_solar_alignment(self):
        a,_=scenario('JN76','FN31','2024-06-01T12:00Z',150,3,10);b,_=scenario('FN31','JN76','2024-06-01T12:00Z',150,3,10)
        x=point_features(a);y=point_features(b)
        self.assertEqual(len(POINTS),45);self.assertAlmostEqual(x.point0_latitude.iloc[0],46.5)
        for percent in PERCENT:
            for suffix in ['latitude','sun_sin_elevation','magnetic_latitude','local_sin','local_cos','lon_sin','lon_cos']:
                self.assertAlmostEqual(x[f'point{percent}_{suffix}'].iloc[0],y[f'point{100-percent}_{suffix}'].iloc[0],places=5)
        self.assertAlmostEqual(x.point0_sun_sin_elevation.iloc[0],a.tx_sun.iloc[0],places=6)
        self.assertAlmostEqual(x.point50_sun_sin_elevation.iloc[0],a.mid_sun.iloc[0],places=6)
        self.assertAlmostEqual(x.point100_sun_sin_elevation.iloc[0],a.rx_sun.iloc[0],places=6)
    def test_dateline_coincident_and_antipodal(self):
        d,_=scenario('RJ90','AJ00','2024-06-01T12:00Z',150,3,10);x=point_features(d)
        self.assertLess(x.point50_lon_cos.iloc[0],-.99)
        d,_=scenario('JN76','JN76','2024-06-01T12:00Z',150,3,10);x=point_features(d)
        for p in PERCENT:self.assertAlmostEqual(x[f'point{p}_latitude'].iloc[0],46.5)
        d,_=scenario('AA00','JR09','2024-06-01T12:00Z',150,3,10)
        x=point_features(d)
        self.assertTrue(d.antipodal_path.iloc[0]);self.assertTrue(x.point50_latitude.isna().all())
    def test_mode_balancing_does_not_invent_unknown_power(self):
        d=pd.DataFrame({'source':['WSPRNET']*20+['HARC_PSK']*2,'mode':['WSPR']*20+['FT8']*2,'date_key':['D']*22,'hour':[0]*22,'power_dbm':[30]*20+[np.nan]*2})
        w=training_weights(d,'mode_hour');self.assertAlmostEqual(w[:20].sum(),w[20:].sum());self.assertTrue(d.power_dbm.iloc[-2:].isna().all())
    def test_selection_is_outcome_blind_and_excludes_old_paths(self):
        from ml40.shared_plan import BASE,OUT,RESULTS
        from ml40.shared_prepare import select
        import json
        if not (OUT/'cases_frozen.csv').exists():self.skipTest('Shared benchmark not collected')
        paths=sorted((BASE/'processed').glob('*.parquet'))
        if len(paths)!=12:self.skipTest('Shared benchmark raw feature samples absent')
        d=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True);d['snr2500']=999.
        prior=pd.concat([pd.read_csv(RESULTS/f'voacap{n}/cases_frozen.csv') for n in [100,1000]],ignore_index=True)
        chosen=select(d.sample(frac=1,random_state=6),prior,json.loads((OUT/'plan.json').read_text()))
        self.assertEqual(chosen.case_id.tolist(),pd.read_csv(OUT/'cases_frozen.csv').case_id.tolist())
        self.assertFalse(set(zip(chosen.tx_grid,chosen.rx_grid))&set(zip(prior.tx_grid,prior.rx_grid)))
        self.assertEqual(chosen.groupby('distance_group').size().tolist(),[125]*4)
if __name__=='__main__':unittest.main()
