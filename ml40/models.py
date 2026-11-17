"""Compact learners with training-only, regularized station nuisance effects."""
import copy
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from .constants import BASE,SOLAR

NUMERIC=BASE+SOLAR+['power_known','legacy_decoder']
CATEGORICAL=['source','mode','decoder_context']

def sample_weights(d):
    """Balance pair-days, then source/mode days; cap tiny-cohort leverage."""
    pair=['source','mode','date_key','tx_sign','rx_sign']
    n=d.groupby(pair,dropna=False).hour.transform('size').to_numpy()
    w=pd.Series(1/n,index=d.index)
    day=['source','mode','date_key']
    denom=w.groupby([d[c] for c in day]).transform('sum')
    w=w/denom
    # Rare days must not give individual spots arbitrarily large influence.
    group=[d.source,d['mode']]
    med=w.groupby(group).transform('median')
    w=pd.Series(np.minimum(w.to_numpy(),20*med.to_numpy()),index=d.index)
    # Equal overall weight per source/mode despite different acquisition rates.
    totals=w.groupby(group).transform('sum')
    w=w/totals
    return (w/w.mean()).to_numpy(dtype=float)

class Encoder:
    def fit(self,d):
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        self.cat=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        self.scaler=StandardScaler()
        x=self.imputer.fit_transform(d[NUMERIC])
        self.scaler.fit(x)
        self.cat.fit(d[CATEGORICAL])
        return self
    def transform(self,d,scaled=False):
        x=self.imputer.transform(d[NUMERIC])
        if scaled:x=self.scaler.transform(x)
        return np.column_stack([x,self.cat.transform(d[CATEGORICAL])]).astype(np.float32)

def make_tree(iterations=180,loss='absolute_error',quantile=None):
    return HistGradientBoostingRegressor(loss=loss,quantile=quantile,max_iter=iterations,
        max_leaf_nodes=23,min_samples_leaf=60,l2_regularization=10,
        learning_rate=.07,early_stopping=False,random_state=40)

class StationEffects:
    """Two-way residual offsets; zero is the population reference, not a dipole."""
    def __init__(self,penalty=50,rounds=8):
        self.penalty=penalty;self.rounds=rounds
    def fit(self,d,residual,weights):
        residual=np.asarray(residual,dtype=float)
        w=np.asarray(weights,dtype=float)
        work=d[['tx_key','rx_key','date_key']].copy().reset_index(drop=True)
        self.tx={};self.rx={}
        for _ in range(self.rounds):
            for role,other in [('tx','rx'),('rx','tx')]:
                correction=work[other+'_key'].map(getattr(self,other)).fillna(0).to_numpy()
                # Winsorization protects nuisance terms from isolated SNR artifacts.
                r=np.clip(residual-correction,-30,30)
                tab=work[[role+'_key','date_key']].copy()
                tab['w']=w;tab['wr']=w*r
                agg=tab.groupby(role+'_key').agg(w=('w','sum'),wr=('wr','sum'),
                    observations=('w','size'),days=('date_key','nunique'))
                value=agg.wr/(agg.w+self.penalty)
                eligible=(agg.observations>=10)&(agg.days>=2)
                # Define a weighted-zero reference for each nuisance role.
                centre=float(np.average(value[eligible],weights=agg.w[eligible])) if eligible.any() else 0
                value=value[eligible]-centre
                setattr(self,role,value.to_dict())
        return self
    def predict(self,d):
        return d.tx_key.map(self.tx).fillna(0).to_numpy()+d.rx_key.map(self.rx).fillna(0).to_numpy()
    def summary(self):
        return {'tx_keys':len(self.tx),'rx_keys':len(self.rx),'penalty':self.penalty,
            'tx_rms_db':float(np.sqrt(np.mean(np.square(list(self.tx.values()))))) if self.tx else 0.,
            'rx_rms_db':float(np.sqrt(np.mean(np.square(list(self.rx.values()))))) if self.rx else 0.}

def crossfit_residuals(d,x,y,w):
    """Nuisance residuals from date-held-out base predictions, never validation/test."""
    fold=d.date_key.map(lambda s:int(hashlib.sha256(str(s).encode()).hexdigest()[:8],16)%3).to_numpy()
    pred=np.empty(len(d))
    for i in range(3):
        tr=fold!=i;te=fold==i
        if not te.any():continue
        if tr.sum()<100:raise ValueError('Insufficient independent training days for residual cross-fit')
        model=make_tree(iterations=100)
        model.fit(x[tr],y[tr],sample_weight=w[tr]);pred[te]=model.predict(x[te])
    return y-pred

class ModelBundle:
    def __init__(self,encoder,model,effects=None,scaled=False):
        self.encoder=encoder;self.model=model;self.effects=effects;self.scaled=scaled
    def predict(self,d,station_correction=True):
        p=self.model.predict(self.encoder.transform(d,scaled=self.scaled))
        if self.scaled:p=p*20
        if self.effects is not None and station_correction:p=p+self.effects.predict(d)
        return p

def train_candidates(train,validation,max_epochs=60):
    if set(train.split)!={'train'}:raise ValueError('Only training rows may fit model parameters')
    if set(validation.split)!={'validation'}:raise ValueError('Only validation rows may select epochs')
    encoder=Encoder().fit(train)
    x=encoder.transform(train);y=train.snr2500.to_numpy();w=sample_weights(train)
    baseline=make_tree().fit(x,y,sample_weight=w)
    residual=crossfit_residuals(train,x,y,w)
    effects=StationEffects().fit(train,residual,w)
    target=y-effects.predict(train)
    adjusted=make_tree().fit(x,target,sample_weight=w)
    models={'boosted_baseline':ModelBundle(encoder,baseline),
        'boosted_station':ModelBundle(encoder,adjusted,effects)}
    xs=encoder.transform(train,scaled=True)
    xv=encoder.transform(validation,scaled=True)
    yv=validation.snr2500.to_numpy();wv=sample_weights(validation)
    biasv=effects.predict(validation)
    nn=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',solver='adam',
        batch_size=min(1024,len(train)),learning_rate_init=.0007,alpha=.1,early_stopping=False,
        random_state=40,shuffle=True)
    history=[];best=None;best_score=float('inf');stale=0
    for epoch in range(1,max_epochs+1):
        nn.partial_fit(xs,target/20,sample_weight=w)
        p=nn.predict(xv)*20+biasv
        score=float(np.average(abs(yv-p),weights=wv))
        history.append({'epoch':epoch,'validation_balanced_mae_db':score})
        if score<best_score-.005:
            best=copy.deepcopy(nn);best_score=score;stale=0
        else:stale+=1
        print('neural epoch',epoch,'validation balanced MAE',round(score,4),flush=True)
        if stale>=8:break
    models['neural_station']=ModelBundle(encoder,best,effects,scaled=True)
    return models,{'epochs':history,'station_effects':effects.summary(),
        'weight_range':[float(w.min()),float(w.max())],
        'validation_selection':'best epoch by balanced MAE; patience 8; final test not consulted'}
