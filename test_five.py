import unittest

import numpy as np
import pandas as pd

from ml40.five_features import donors, soundings
from ml40.five_models import adjacent


class FiveTests(unittest.TestCase):
    def test_only_consecutive_same_case(self):
        times = pd.to_datetime(['2026-03-01 01:00','2026-03-01 03:00','2026-03-01 02:00','2026-03-01 04:00'],utc=True)
        a,b = adjacent(pd.DataFrame({'case':['a','a','b','a'],'hour':times}))
        self.assertEqual(list(zip(a,b)),[(1,3)])

    def test_independent_donors(self):
        row = dict(tx_sign='A',rx_sign='B',tx_grid='JN76',rx_grid='JN65',tx_lat=46.,tx_lon=15.,rx_lat=45.,rx_lon=13.)
        target = pd.DataFrame([row])
        pool = pd.DataFrame([{**row,'tx_sign':tx,'rx_sign':rx,'rx_grid':grid,'base':-15.,'snr2500':snr}
            for tx,rx,grid,snr in [('A','C','JN64',50.),('D','E','JN65',50.),('F','G','JN64',-12.),
                                    ('F','H','JN64',50.),('I','J','JN64',-12.),('K','L','JN64',-12.)]])
        signal,n = donors(target,pool)
        self.assertEqual(n[0],3)
        self.assertAlmostEqual(signal[0],3.)
        self.assertEqual(donors(target,pool.iloc[:3])[0][0],0.)

    def test_recent_sonde_only(self):
        d = pd.DataFrame(dict(hour=pd.to_datetime(['2026-01-19T02:00Z','2026-01-19T05:00Z']),mid_lat=[50.,50.],mid_lon=[5.,5.],distance_km=[300.,300.]))
        table = pd.DataFrame(dict(time=['2026-01-19T01:45Z','2026-01-19T02:00Z','2026-01-19T04:55Z'],confidence=[80,80,20],fo=[6.,12.,10.]))
        soundings(d,np.array([4.,4.]),table)
        self.assertEqual(list(d.sonde_available),[True,False])
        self.assertEqual(list(d.sonde_delta),[2.,0.])
