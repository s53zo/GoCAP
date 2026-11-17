"""Memory-bounded fit of the unchanged shared tree specification on larger data."""
import gc
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from .shared_models import SharedModel
from .shared_features import POINTS
from .hourly_models import LOCAL
from .models import NUMERIC
from .v3_features import PATH,HISTORY
from .v3_models import Encoder,CATEGORIES

def encoder_from_medians(medians,numeric,categories):
    encoder=Encoder(True);encoder.numeric=numeric
    medians=np.where(np.isfinite(medians),medians,0)
    encoder.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(pd.DataFrame([medians],columns=numeric))
    encoder.cat=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(categories[CATEGORIES])
    return encoder

class MarchModel(SharedModel):
    def __init__(self):super().__init__(points=True,weighting='observation')
    def fit_bounded(self,d,folder,batch=32768):
        numeric=NUMERIC+LOCAL+PATH+HISTORY+POINTS;n=len(d);folder.mkdir(exist_ok=True)
        # A column-contiguous map permits exact global median calculation one feature at a time.
        raw=np.lib.format.open_memmap(folder/'numeric.npy',mode='w+',dtype='float32',shape=(n,len(numeric)),fortran_order=True)
        for begin in range(0,n,batch):
            stop=min(begin+batch,n);frame=self.frame(d.iloc[begin:stop]);raw[begin:stop]=frame[numeric].to_numpy(dtype=np.float32);del frame
        raw.flush();medians=np.array([np.nanmedian(raw[:,j]) for j in range(len(numeric))])
        self.encoder=encoder_from_medians(medians,numeric,d)
        width=len(numeric)+sum(len(x) for x in self.encoder.cat.categories_)
        encoded=np.lib.format.open_memmap(folder/'encoded.npy',mode='w+',dtype='float32',shape=(n,width))
        for begin in range(0,n,batch):
            stop=min(begin+batch,n);values=np.array(raw[begin:stop],copy=True)
            invalid=np.isnan(values);values[invalid]=np.broadcast_to(self.encoder.imputer.statistics_,values.shape)[invalid]
            cat=self.encoder.cat.transform(d.iloc[begin:stop][CATEGORIES]).astype(np.float32)
            encoded[begin:stop,:len(numeric)]=values;encoded[begin:stop,len(numeric):]=cat
        encoded.flush();del raw;gc.collect()
        self.model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=240,learning_rate=.05,max_leaf_nodes=31,
            min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=40)
        self.model.fit(encoded,d.snr2500.to_numpy(),sample_weight=np.ones(n));del encoded;gc.collect()
        return self
