"""Retain March2020 code0 archive reports in explicitly uncertain legacy context."""
import datetime as dt,json,time
import numpy as np,pandas as pd
from experiment import hourly,features
from .march_plan import BASE,OUT,sha
from .prepare import annotate
from .v3_features import enrich

def repair():
    amendment={'scope':'Only March2020 code0 records in this new experiment; existing artifacts/data unchanged.',
        'reason':'Original pre2020 legacy cutoff discarded all sampled March2020 reports, although archive records and reported WSPR receiver versions exist. Retain as legacy metadata, not verified modern code1.',
        'policy':'March2020 code0 is included with decoder_context=legacy_code0, preserving measurement uncertainty. No universal decoding threshold inferred; no absence labels.',
        'evidence':['https://wspr.live/ (code0 absent from current mode mapping; older archive example uses0)',
            'https://wsjt.sourceforge.io/FST4_Quick_Start.pdf (FST4/FST4W introduced withWSJT-X2.3)',
            'https://sourceforge.net/p/wsjt/mailman/message/37117193/ (2.3release candidate announcedSeptember2020, after this March)'],
        'resolved_before_model_fitting':True}
    ap=OUT/'legacy_mode_amendment.json'
    if ap.exists():assert json.loads(ap.read_text())==amendment
    else:ap.write_text(json.dumps(amendment,indent=2))
    for day in range(1,32):
        date=f'2020-03-{day:02d}';p=BASE/'raw'/f'wspr-{date}.jsonl';out=BASE/'processed'/p.with_suffix('.parquet').name
        meta=out.with_suffix('.meta.json')
        while not meta.exists():time.sleep(5)
        if json.loads(meta.read_text()).get('legacy_amendment_sha256')==sha(ap):continue
        raw=pd.read_json(p,lines=True);raw['time']=pd.to_datetime(raw.time,utc=True)
        legacy=raw.code.eq(0)&raw.time.lt(pd.Timestamp('2020-04-01',tz='UTC'));eligible=legacy|raw.code.eq(1)
        raw=raw[eligible].copy();raw['mode']='WSPR';raw['source']='WSPRNET';raw['snr2500']=raw.snr.astype(float);raw['power_dbm']=raw.power.astype(float)
        raw['decoder_context']=np.where(legacy[eligible],'legacy_code0','pre2023_code1')
        contexts=raw.groupby(raw.time.dt.floor('h')).decoder_context.first().to_dict()
        d=features(hourly(raw),solar_start='2009');d['decoder_context']=d.hour.map(contexts);d=enrich(annotate(d))
        part=out.with_suffix('.repair.parquet');d.to_parquet(part,index=False);part.replace(out)
        audit={'raw_rows':int(len(eligible)),'eligible_rows':int(eligible.sum()),'code_counts':pd.read_json(p,lines=True).code.value_counts().to_dict(),
            'legacy_code0_rows_included':int(legacy.sum()),'legacy_mode_uncertainty_retained':True}
        meta.write_text(json.dumps({'raw_sha256':sha(p),'prepared_sha256':sha(out),'audit':audit,'legacy_amendment_sha256':sha(ap)},indent=2))
        print(date,'legacy reports retained',int(legacy.sum()),'hourly aggregates',len(d),flush=True)
if __name__=='__main__':repair()
