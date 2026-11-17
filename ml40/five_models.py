"""Residual regressors for absolute hourly WSPR SNR and within-path changes."""
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

CORE = ['base','distance_km','power_dbm','tx_sun','rx_sun','mid_sun','tx_lat','rx_lat',
        'tx_lon_sin','tx_lon_cos','rx_lon_sin','rx_lon_cos','season_sin','season_cos',
        'kp_previous','Ap_previous_day','sfi_previous_day']
# (target, difference-loss weight, optional feature blocks)
SPECS = {
    'one_report': ('draw_0',0,()),
    'hour_median': ('snr2500',0,()),
    'change': ('snr2500',2,()),
    'f2': ('snr2500',0,('f2',)),
    'f2_sonde': ('snr2500',0,('f2','sonde')),
    'donors': ('snr2500',0,('donors',)),
    'voacap_shape': ('snr2500',0,('voacap',)),
    'static_combined': ('snr2500',2,('f2','voacap')),
    'combined_donors': ('snr2500',2,('f2','voacap','donors')),
    'combined_live': ('snr2500',2,('f2','voacap','donors','sonde')),
}
STATIC = ['default','one_report','hour_median','change','f2','voacap_shape','static_combined']


def adjacent(d):
    keys = d[['case','hour']].reset_index(drop=True).reset_index(names='i')
    later = keys.assign(hour=keys.hour+pd.Timedelta(hours=1)).rename(columns={'i':'prev'})
    pairs = keys.merge(later,on=['case','hour'],validate='one_to_one')
    return pairs.prev.to_numpy(),pairs.i.to_numpy()


class Correction:
    def __init__(self,blocks):
        self.blocks = blocks
        self.scale = StandardScaler()
        self.basis = RBFSampler(gamma=.035,n_components=192,random_state=40)

    def design(self,d):
        x = np.clip(self.scale.transform(d[CORE]),-6,6)
        parts = [np.ones((len(x),1)),x,self.basis.transform(x)*5.]
        for block in self.blocks:
            if block=='f2':
                short = d.distance_km.le(1000).to_numpy()
                value = d.f2_margin.fillna(0).clip(-10,15).to_numpy()/4
                parts.extend([short, value, np.maximum(value,0), value*d.mid_sun])
            elif block=='sonde':
                parts.extend([d.sonde_available, d.sonde_delta.fillna(0).clip(-8,8)/3])
            elif block=='donors':
                weight = d.donor_count.clip(0,12)/12
                parts.extend([d.live/5, d.live*weight/5])
            elif block=='voacap':
                shape = d.voa_shape.fillna(0).clip(-50,50)/15
                slope = d.voa_slope.fillna(0).clip(-40,40)/10
                parts.extend([shape,slope,shape*d.mid_sun,slope*d.mid_sun,
                              shape*(d.distance_km<=1000),slope*(d.distance_km>=6000)])
        return np.column_stack(parts)

    def fit(self,d,target,change_weight):
        self.scale.fit(d[CORE])
        self.basis.fit(self.scale.transform(d[CORE].iloc[:1]))
        x = self.design(d)
        y = (d[target]-d.base).to_numpy()
        if change_weight:
            a,b = adjacent(d)
            w = np.sqrt(change_weight*len(d)/len(a))
            x,y = np.vstack([x,w*(x[b]-x[a])]),np.r_[y,w*(y[b]-y[a])]
        # Fixed ridge penalty on the shared representation, intercept unpenalized.
        penalty = np.eye(x.shape[1])*30.
        penalty[0,0] = 0.
        self.coef = np.linalg.solve(x.T@x+penalty,x.T@y)
        return self

    def predict(self,d):
        return d.base.to_numpy()+self.design(d)@self.coef
