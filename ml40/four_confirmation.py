"""Six later dates after candidate freeze, without any fitting or selection."""
from concurrent.futures import ThreadPoolExecutor
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import four_data, four_drap
from .four_prepare import prepare,DEFAULT
from .four_run import ART,OUT,CACHE,sha
from .four_models import component_predictions

DATES=[f'2026-{m:02d}-27' for m in range(3,9)]


def run():
    checksum=json.loads((OUT/'selection.json').read_text())['model_sha256']
    assert sha(ART)==checksum
    pack=joblib.load(ART)
    with ThreadPoolExecutor(max_workers=3) as pool:
        radio=pool.submit(four_data.collect,DATES,'confirmation_sources.json')
        met=[pool.submit(four_drap.collect,date) for date in DATES]
        radio.result()
        for f in met:
            f.result()
    bundle=joblib.load(DEFAULT)
    daily=[]
    for date in DATES:
        prepare(date,bundle)
        d=pd.read_parquet(CACHE/f'hours-{date}.parquet')
        x=component_predictions(d,pack)
        for name,c in pack['weights'].items():
            yhat=d.base+sum((x[key]*w for key,w in c.items()),start=0)
            mae=abs(yhat-d.snr2500).groupby(d.hour).mean().mean()
            daily.append(dict(date=date,model=name,mae=mae,rows=len(d),reports=int(d.reports.sum()),maps=int(d[d.abs_mean.notna()].hour.nunique())))
    result=pd.DataFrame(daily)
    result.to_csv(OUT/'confirmation_daily.csv',index=False)
    z=result.pivot(index='date',columns='model',values='mae')
    print(z.mean(),flush=True)
    dif=(z.live_combined-z.default).to_numpy()
    boot=np.random.default_rng(41).choice(dif,(10000,len(dif))).mean(1)
    summary=dict(model_sha256=checksum,rows=int(result[result.model=='default'].rows.sum()),scores=z.mean().to_dict(),
                 live_minus_default_ci=np.quantile(boot,[.025,.975]).tolist())
    (OUT/'confirmation.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=3):
        run()
