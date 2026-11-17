"""Established source/mode residual intervals retained in the recommended package."""
import numpy as np
from .v3_models import profile_predict

class ReferenceIntervals:
    def __init__(self,calibration,selection):self.calibration=calibration;self.selection=selection
    def predict(self,d,models):
        p=profile_predict(d,models,self.selection)
        half=np.array([self.calibration[r.source+'|'+r.mode]['half_width_db'] for r in d.itertuples()])
        return p-half,p+half
