"""Recheck source hashes, raw hourly medians, saved scores, and the frozen March comparator."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import features,grid4
from .four_data import CACHE,DATES,OUT,ROOT,sha
from .four_confirmation import DATES as CONFIRM
from .four_run import ART
from .five_prepare import CASE,DEFAULT
from .prepare import annotate
from .v3_features import enrich
from .hourly_models import predict_profiles
from .four_models import component_predictions
from .four_transfer import TRANSFER,basis,DATES as FINAL

def run():
    meta=json.loads((OUT/'selection.json').read_text())
    assert sha(ART)==meta['model_sha256'] and sha(DEFAULT)==meta['base_sha256']
    pack=joblib.load(ART)
    for name in ['sources','confirmation_sources','transfer_sources']:
        for path,m in json.loads((OUT/f'{name}.json').read_text()).items():
            assert sha(CACHE/path)==m['sha256']
    for date,m in json.loads((OUT/'power_sources.json').read_text()).items():
        assert sha(CACHE/f'steptx-{date}.jsonl')==m['sha256']
    model=joblib.load(ROOT/'artifacts/gocap-40m-shared-march-tuned.joblib')
    records=[]
    ref=pd.read_csv(OUT/'hourly_mae.csv')
    ref['hour']=pd.to_datetime(ref.hour,utc=True)
    confirm=pd.read_csv(OUT/'confirmation_daily.csv')
    final_scores=pd.read_csv(OUT/'transfer_daily.csv')
    final=joblib.load(TRANSFER)
    assert sha(TRANSFER)==json.loads((OUT/'transfer_selection.json').read_text())['model_sha256']
    for date in DATES['test']+CONFIRM+FINAL:
        raw=pd.read_json(CACHE/f'spots-{date}.jsonl',lines=True)
        raw=raw[raw.code.eq(1)&raw.frequency.between(7e6,7.3e6)&raw.snr.between(-60,100)&
                raw.power.between(0,60)&raw.rx_sign.str.len().ge(2)&raw.tx_sign.str.len().ge(2)].copy()
        raw['hour']=pd.to_datetime(raw.time,utc=True).dt.floor('h')
        for side in ['tx','rx']:
            raw[side+'_grid']=raw[side+'_loc'].map(grid4)
        r=raw.dropna(subset=['tx_grid','rx_grid']).drop_duplicates(['time','tx_sign','rx_sign','frequency'])
        truth=r.rename(columns={'power':'power_dbm'}).groupby(CASE+['hour']).snr.agg(['median','size']).reset_index()
        src=pd.read_parquet(CACHE/f'hours-{date}.parquet')
        d=src.merge(truth,on=CASE+['hour'],validate='one_to_one')
        assert len(d)==len(src)
        np.testing.assert_array_equal(d.snr2500,d['median'])
        np.testing.assert_array_equal(d.reports,d['size'])
        cols=CASE+['hour','snr2500']
        f=enrich(annotate(features(d[cols].assign(source='WSPRNET',mode='WSPR',decoder_context='ordinary_code1'),solar_start='2009')))
        f['march_tuned']=predict_profiles(f,model['models'],model['profile_selection'])
        f=f.merge(d[cols+['base']],on=cols,validate='one_to_one')
        scores=abs(f.march_tuned-f.snr2500).groupby(f.hour).mean()
        x=component_predictions(d,pack)
        for name,terms in pack['weights'].items():
            predicted=d.base+sum((x[k]*w for k,w in terms.items()),start=0)
            actual=abs(predicted-d.snr2500).groupby(d.hour).mean()
            if date in FINAL:
                if name!='live_combined':
                    continue
                names={'default':d.base,'previous_combination':predicted,'march_tuned':d[cols].merge(f[cols+['march_tuned']],on=cols,validate='one_to_one').march_tuned.to_numpy()}
                basis_values=basis(d,final)
                for key in ['scenario','nowcast']:
                    coef=final[key]
                    names['march_'+key]=names['march_tuned']+basis_values[:,:len(coef)]@coef
                for key,level in names.items():
                    measured=abs(level-d.snr2500).groupby(d.hour).mean().mean()
                    saved=final_scores[(final_scores.date==date)&(final_scores.model==key)].hour_mae.iloc[0]
                    np.testing.assert_allclose(measured,saved)
            elif date in CONFIRM:
                np.testing.assert_allclose(actual.mean(),confirm[(confirm.date==date)&(confirm.model==name)].mae.iloc[0])
            else:
                np.testing.assert_allclose(actual,ref.set_index('hour').loc[actual.index,name])
        maps=np.load(CACHE/'drap'/f'{date}.npz')
        assert np.all(np.isin(maps['age'],[*range(5,16),999]))
        records.append(dict(date=date,split='transfer' if date in FINAL else 'confirmation' if date in CONFIRM else 'primary',
            rows=len(d),reports=int(d.reports.sum()),march_tuned_mae=scores.mean()))
    out=pd.DataFrame(records)
    out.to_csv(OUT/'verified_dates.csv',index=False)
    print(out.groupby('split').agg({'rows':'sum','reports':'sum','march_tuned_mae':'mean'}))

if __name__=='__main__':
    with threadpool_limits(limits=3):
        run()
