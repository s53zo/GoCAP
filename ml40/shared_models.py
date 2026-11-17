"""One source/mode-aware pooled SNR regressor; no physical power imputation claim."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from .models import NUMERIC
from .v3_models import Encoder
from .v3_features import PATH,HISTORY
from .hourly_models import augment,LOCAL
from .shared_features import POINTS,point_features

def training_weights(d,kind):
    if kind=='observation':return np.ones(len(d))
    keys=['source','mode','date_key','hour'];n=d.groupby(keys).hour.transform('size').to_numpy()
    weights=1/n
    # Cap leverage inside a mode, then balance modes rather than allowing WSPR volume to dominate.
    for mode in d['mode'].unique():
        use=d['mode'].eq(mode).to_numpy();weights[use]=np.minimum(weights[use],np.median(weights[use])*20)
        weights[use]/=weights[use].sum()
    return weights/weights.mean()

class SharedModel:
    def __init__(self,points=True,weighting='mode_hour'):self.points=points;self.weighting=weighting
    def frame(self,d):
        d=augment(d)
        return point_features(d) if self.points else d
    def fit(self,d):
        x=self.frame(d);self.encoder=Encoder(True)
        self.encoder.numeric=NUMERIC+LOCAL+PATH+HISTORY+(POINTS if self.points else [])
        self.encoder.fit(x)
        self.model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=240,learning_rate=.05,
            max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=40)
        self.model.fit(self.encoder.transform(x),d.snr2500.to_numpy(),sample_weight=training_weights(d,self.weighting))
        return self
    def predict(self,d,station_correction=False):return self.model.predict(self.encoder.transform(self.frame(d)))
