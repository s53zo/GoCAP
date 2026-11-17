"""Power-normalized SNR models, source residual heads, and positive-only rankers."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from .v3_features import BASE_NUMERIC,RICH_NUMERIC
from .models import sample_weights

CATEGORIES=['source','mode','decoder_context']
PAIR=['source','mode','date_key','tx_sign','rx_sign','tx_grid','rx_grid']

def shift(d):
    known=d['mode'].eq('WSPR')
    if d.loc[known,'power_dbm'].isna().any():raise ValueError('WSPR power must be known for normalization')
    return np.where(known,d.power_dbm.fillna(30)-30,0)

class Encoder:
    def __init__(self,rich=True):self.numeric=RICH_NUMERIC if rich else BASE_NUMERIC
    def fit(self,d):
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(d[self.numeric])
        self.cat=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(d[CATEGORIES]);return self
    def transform(self,d):
        return np.column_stack([self.imputer.transform(d[self.numeric]),self.cat.transform(d[CATEGORIES])]).astype(np.float32)

def tree(iterations=180,loss='absolute_error',quantile=None,leaves=23):
    return HistGradientBoostingRegressor(loss=loss,quantile=quantile,max_iter=iterations,max_leaf_nodes=leaves,
        min_samples_leaf=60,l2_regularization=10,learning_rate=.07,early_stopping=False,random_state=40)

class SNRModel:
    def __init__(self,rich=True,source_heads=False):self.rich=rich;self.source_heads=source_heads
    def fit(self,d):
        self.modes=set(d['mode']);self.encoder=Encoder(self.rich).fit(d)
        x=self.encoder.transform(d);y=d.snr2500.to_numpy()-shift(d);w=sample_weights(d)
        self.model=tree().fit(x,y,sample_weight=w);self.heads={}
        if self.source_heads:
            import hashlib
            fold=d.date_key.map(lambda s:int(hashlib.sha256(str(s).encode()).hexdigest()[:8],16)%3).to_numpy()
            oof=np.empty(len(d))
            for f in range(3):
                tr=fold!=f;te=fold==f
                base=tree(iterations=100).fit(x[tr],y[tr],sample_weight=w[tr]);oof[te]=base.predict(x[te])
            key=(d.source+'|'+d['mode']).to_numpy()
            for group in sorted(set(key)):
                use=key==group
                self.heads[group]=tree(iterations=80,leaves=7).fit(x[use],(y-oof)[use],sample_weight=w[use])
        return self
    def predict(self,d,station_correction=False):
        if not set(d['mode'])<=self.modes:raise ValueError('Requested mode was not trained in this model')
        x=self.encoder.transform(d);p=self.model.predict(x)
        key=(d.source+'|'+d['mode']).to_numpy()
        for group,model in self.heads.items():
            use=key==group
            if use.any():p[use]+=.5*model.predict(x[use])
        return p+shift(d)


def rank_training(d,min_hours=4):
    d=d[d['mode'].eq('WSPR')&d.power_dbm.notna()].copy()
    d['normalized_snr']=d.snr2500.to_numpy()-shift(d)
    keys=PAIR+['hour']
    # Every row is an actual reception. No resampling of absent hours is performed.
    first=d.groupby(keys,sort=True).first().reset_index()
    med=d.groupby(keys,sort=True).normalized_snr.median().reset_index(name='label_snr')
    out=first.merge(med,on=keys,validate='one_to_one')
    count=out.groupby(PAIR).hour.transform('nunique');distinct=out.groupby(PAIR).label_snr.transform('nunique')
    out=out[count.ge(min_hours)&distinct.ge(2)].copy().reset_index(drop=True)
    out['qid']=out.groupby(PAIR,sort=True).ngroup()
    out['relevance']=out.groupby('qid').label_snr.rank(method='dense').astype(int)-1
    # Values are ordinals of positive received strengths, never open/closed labels.
    return out.sort_values(['qid','hour']).reset_index(drop=True)

class HourRanker:
    def fit(self,d,rich=True,iterations=240,depth=3):
        from xgboost import XGBRanker
        self.encoder=Encoder(rich).fit(d);self.rich=rich
        groups=d.groupby('qid',sort=True).date_key.first()
        day_count=groups.map(groups.value_counts())
        weights=1/day_count.to_numpy();weights/=weights.mean()
        self.model=XGBRanker(n_estimators=iterations,max_depth=depth,learning_rate=.04,
            objective='rank:pairwise',tree_method='hist',n_jobs=4,reg_lambda=10,
            subsample=.85,colsample_bytree=.9,random_state=40,
            lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=8)
        self.model.fit(self.encoder.transform(d),d.relevance.to_numpy(),qid=d.qid.to_numpy(),sample_weight=weights)
        return self
    def predict(self,d):
        if not d['mode'].eq('WSPR').all():raise ValueError('Hourly ranker supports WSPR evidence only')
        return self.model.predict(self.encoder.transform(d))

class QuantileIntervals:
    def fit(self,train,calibration,point_models,profile_selection):
        from .train import quantile_finite
        self.encoder=Encoder(True).fit(train);x=self.encoder.transform(train)
        y=train.snr2500.to_numpy()-shift(train);w=sample_weights(train)
        self.lower=tree(loss='quantile',quantile=.1).fit(x,y,sample_weight=w)
        self.upper=tree(loss='quantile',quantile=.9).fit(x,y,sample_weight=w)
        self.corrections={};self.profile_selection=profile_selection
        lo,hi=self.raw(calibration,point_models)
        error=np.maximum(lo-calibration.snr2500.to_numpy(),calibration.snr2500.to_numpy()-hi)
        groups=(calibration.source+'|'+calibration['mode']).to_numpy()
        for group in sorted(set(groups)):
            e=error[groups==group]
            if len(e)>=100:self.corrections[group]={'adjustment_db':max(0,quantile_finite(e,.8)),'rows':len(e)}
        return self
    def raw(self,d,models):
        x=self.encoder.transform(d);offset=shift(d)
        a=self.lower.predict(x)+offset;b=self.upper.predict(x)+offset
        lo=np.minimum(a,b);hi=np.maximum(a,b)
        p=profile_predict(d,models,self.profile_selection)
        return np.minimum(lo,p),np.maximum(hi,p)
    def predict(self,d,models):
        lo,hi=self.raw(d,models)
        keys=d.source+'|'+d['mode']
        correction=keys.map(lambda k:self.corrections.get(k,{}).get('adjustment_db',np.nan)).to_numpy()
        return lo-correction,hi+correction


def profile_predict(d,models,selection):
    p=np.empty(len(d),dtype=float)
    for mode in sorted(set(d['mode'])):
        use=d['mode'].eq(mode).to_numpy();model=models[selection[mode]]
        p[use]=model.predict(d.loc[use])
    return p
