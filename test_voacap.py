import unittest
import pandas as pd
from ml40.voacap_engine import parse_output,at_bin_centre,BANDWIDTH_DB,make_deck
from ml40.voacap_prepare import select
from ml40.voacap_protocol import BASE,OUT,COUNT

class VoacapTests(unittest.TestCase):
    def test_requested_column_and_units(self):
        # MUF SNR deliberately differs from requested-channel SNR.
        lines=[]
        for h in range(1,25):
            for label,values in [('FREQ',[9.,7.]),('MODE',['2F2','1F2']),('SNR',[99.,20.]),('SNR LW',[1.,8.]),('SNR UP',[1.,12.])]:
                prefix=f'{h:6.1f}' if label=='FREQ' else ' '*6
                lines.append(prefix+''.join(f'{v:>5}' for v in values)+' '*50+label)
        p=parse_output('\n'.join(lines));self.assertAlmostEqual(p[0]['snr2500_db'],20-BANDWIDTH_DB)
        value,lo,hi=at_bin_centre(p,23)
        self.assertAlmostEqual(value,20-BANDWIDTH_DB);self.assertAlmostEqual(lo,12-BANDWIDTH_DB);self.assertAlmostEqual(hi,32-BANDWIDTH_DB)
        missing='\n'.join(lines).replace('  1F2','    -')
        self.assertIsNone(at_bin_centre(parse_output(missing),0)[0])
    def test_actual_watts_not_kw(self):
        deck=make_deck(46.5,15,41.5,-73,2025,4,100,.1)
        ant=[x for x in deck.splitlines() if x.startswith('ANTENNA')][0]
        self.assertTrue(ant.endswith(' 0.0001000'))
        self.assertIn('[default/gc10m.voa    ]',ant)
    def test_selection_is_snr_blind(self):
        paths=sorted((BASE/'processed').glob('*.parquet'))
        if not paths:self.skipTest('Collect the VOACAP benchmark reception samples first')
        pieces=[pd.read_parquet(p) for p in paths]
        d=pd.concat(pieces,ignore_index=True);a=select(d)
        d['snr2500']=999.;b=select(d.sample(frac=1,random_state=41))
        self.assertEqual(a.case_id.tolist(),b.case_id.tolist())
        self.assertEqual(a.case_id.tolist(),pd.read_csv(OUT/'cases_frozen.csv').case_id.tolist())
        self.assertEqual(a.groupby('distance_group').size().tolist(),[COUNT//4]*4)
        self.assertEqual(len(set(zip(a.tx_grid,a.rx_grid))),COUNT)
if __name__=='__main__':unittest.main()
