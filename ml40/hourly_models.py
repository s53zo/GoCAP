"""Absolute hourly SNR learners and bounded independently trained residual corrections."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .models import NUMERIC
from .v3_models import Encoder
from .v3_features import PATH,HISTORY

LOCAL=['tx_local_sin','tx_local_cos','rx_local_sin','rx_local_cos']

def augment(d):
    d=d.copy();phase=2*np.pi*(d.hour.dt.day_of_year.to_numpy()-1)/365.2425
    eq=229.18*(.000075+.001868*np.cos(phase)-.032077*np.sin(phase)-.014615*np.cos(2*phase)-.040849*np.sin(2*phase))
    for role in ['tx','rx']:
        longitude=np.rad2deg(np.arctan2(d[role+'_lon_sin'].to_numpy(),d[role+'_lon_cos'].to_numpy()))
        local=d.hour.dt.hour.to_numpy()+.5+longitude/15+eq/60
        d[role+'_local_sin']=np.sin(2*np.pi*local/24);d[role+'_local_cos']=np.cos(2*np.pi*local/24)
    return d


def weights(d,kind='hour'):
    if kind=='observation':return np.ones(len(d))
    keys=['date_key']+(['hour'] if kind=='hour' else [])
    n=d.groupby(keys).hour.transform('size').to_numpy()
    w=1/n;w=np.minimum(w,np.median(w)*20)
    return w/w.mean()

class NativeModel:
    def __init__(self,rich=True,weighting='hour',small=False):
        self.rich=rich;self.weighting=weighting;self.small=small
    def fit(self,d):
        d=augment(d);self.encoder=Encoder(True)
        self.encoder.numeric=NUMERIC+LOCAL+(PATH+HISTORY if self.rich else [])
        self.encoder.fit(d)
        self.model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=240,
            learning_rate=.05,max_leaf_nodes=15 if self.small else 31,
            min_samples_leaf=200 if self.small else 100,l2_regularization=10,
            early_stopping=False,random_state=40)
        self.model.fit(self.encoder.transform(d),d.snr2500.to_numpy(),sample_weight=weights(d,self.weighting))
        return self
    def predict(self,d,station_correction=False):return self.model.predict(self.encoder.transform(augment(d)))

class ResidualModel:
    def __init__(self,reference,strength=1.):self.reference=reference;self.strength=strength
    def fit(self,d):
        d=augment(d);baseline=self.reference.predict(d,station_correction=False)
        d['reference_snr_prediction']=baseline
        self.encoder=Encoder(True);self.encoder.numeric=NUMERIC+LOCAL+PATH+HISTORY+['reference_snr_prediction']
        self.encoder.fit(d)
        self.model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,
            learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=300,l2_regularization=50,
            early_stopping=False,random_state=40)
        self.model.fit(self.encoder.transform(d),d.snr2500.to_numpy()-baseline,sample_weight=weights(d,'hour'))
        return self
    def predict(self,d,station_correction=False):
        d=augment(d);p=self.reference.predict(d,station_correction=False);d['reference_snr_prediction']=p
        correction=np.clip(self.model.predict(self.encoder.transform(d)),-3,3)
        return p+self.strength*correction


def metric(d,p):
    e=np.asarray(p)-d.snr2500.to_numpy()
    work=pd.DataFrame({'date':d.date_key.to_numpy(),'hour':d.hour.dt.hour.to_numpy(),'absolute_error':abs(e)})
    return {'hour_date_mae_db':float(work.groupby(['date','hour']).absolute_error.mean().mean()),
        'mae_db':float(abs(e).mean()),'rmse_db':float(np.sqrt(np.mean(e**2))),
        'bias_db':float(e.mean()),'rows':len(d),'days':int(d.date_key.nunique()),
        'observed_hour_date_cells':len(work.groupby(['date','hour']))}


def predict_profiles(d,models,selection):
    p=np.empty(len(d),dtype=float)
    for mode in sorted(set(d['mode'])):
        use=d['mode'].eq(mode).to_numpy();p[use]=models[selection[mode]].predict(d.loc[use],station_correction=False)
    return p

class HourlyIntervals:
    def fit(self,calibration,models,selection):
        from .train import quantile_finite
        self.selection=selection;self.half_width={}
        p=predict_profiles(calibration,models,selection);error=abs(calibration.snr2500.to_numpy()-p)
        groups=(calibration.source+'|'+calibration['mode']).to_numpy()
        for key in sorted(set(groups)):
            e=error[groups==key]
            if len(e)>=100:self.half_width[key]={'db':quantile_finite(e,.8),'rows':len(e)}
        return self
    def predict(self,d,models):
        p=predict_profiles(d,models,self.selection)
        half=np.array([self.half_width[r.source+'|'+r.mode]['db'] for r in d.itertuples()])
        return p-half,p+half
