"""Add465 March windows without duplicating originals or leaking reserved dates."""
import json
import numpy as np,pandas as pd
from .march_plan import DATA,BASE,OUT,DATES,FRESH_TEST,KEYS,plan,sha

def canonical_keys(d):
    k=d[KEYS].copy();k['hour']=pd.DatetimeIndex(k.hour).as_unit('s').asi8
    k['power_dbm']=k.power_dbm.astype(float)
    for c in ['source','mode','tx_sign','rx_sign','tx_grid','rx_grid']:k[c]=k[c].astype(str)
    return pd.util.hash_pandas_object(k,index=False)

def compact(d):
    d=d.copy()
    for c in d.select_dtypes(include='float').columns:
        if c!='snr2500':d[c]=d[c].astype(np.float32)
    for c in d.select_dtypes(include=['str','object']).columns:
        if c not in ['march_key','row_key']:d[c]=d[c].astype('category')
    return d

def prepare():
    cfg=plan()
    amendment=sha(OUT/'legacy_mode_amendment.json')
    for day in range(1,32):
        meta=json.loads((BASE/'processed'/f'wspr-2020-03-{day:02d}.meta.json').read_text())
        assert meta.get('legacy_amendment_sha256')==amendment,'March2020 legacy repair is required before fitting'
    original=pd.read_parquet(DATA/'hourly/original_train_features.parquet');original['march_key']=canonical_keys(original)
    assert not original.march_key.duplicated().any()
    validation=pd.read_parquet(DATA/'hourly/validation_features.parquet');calibration=pd.read_parquet(DATA/'hourly/calibration_features.parquet')
    reserved=set(validation.date_key.astype(str))|set(calibration.date_key.astype(str))
    parts=[];coverage=[]
    for date in DATES:
        path=BASE/'processed'/f'wspr-{date}.parquet'
        if not path.exists():raise ValueError('Incomplete March source collection:'+date)
        d=pd.read_parquet(path);rows=len(d)
        coverage.append({'date':date,'year':int(date[:4]),'raw_reports':json.loads((BASE/'raw'/f'wspr-{date}.meta.json').read_text())['rows'],
            'hourly_rows':rows,'role':'reserved_validation_or_calibration' if date in reserved else 'fitting_candidate','available':bool(rows)})
        if rows and date not in reserved:
            assert d['mode'].eq('WSPR').all() and d.hour.dt.year.le(2024).all();d['march_key']=canonical_keys(d);parts.append(d)
    added=pd.concat(parts,ignore_index=True)
    assert not added.march_key.duplicated().any()
    duplicates=added.merge(original[['march_key','snr2500','reports']],on='march_key',suffixes=('','_original'),validate='one_to_one')
    assert np.allclose(duplicates.snr2500,duplicates.snr2500_original) and np.allclose(duplicates.reports,duplicates.reports_original)
    added=added[~added.march_key.isin(original.march_key)].copy()
    expanded=pd.concat([original,added],ignore_index=True)
    # Some newly sampled2024 dates are designated calibration/validation by the old day-only splitter.
    # Only the actually reserved original dates are held out; new years/day08/22 remain fitting dates.
    expanded['split']='train'
    assert not set(expanded.date_key.astype(str))&reserved
    assert expanded.hour.dt.year.le(2024).all() and not expanded.march_key.duplicated().any()
    from .models import NUMERIC
    from .v3_features import PATH,HISTORY
    needed=list(dict.fromkeys(NUMERIC+PATH+HISTORY+['source','mode','decoder_context','hour','date_key','tx_grid','rx_grid','tx_sign','rx_sign','snr2500','march_key','reports','split']))
    full=compact(expanded[needed]);full.to_parquet(BASE/'expanded_train_features.parquet',index=False)
    for kind,dates in [('validation',validation),('calibration',calibration)]:
        assert not set(canonical_keys(dates))&set(expanded.march_key)
    train_dates=expanded.groupby(['mode','date_key']).size().rename('rows').reset_index();train_dates.to_csv(OUT/'training_dates.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'march_coverage.csv',index=False)
    byyear=pd.DataFrame(coverage).groupby('year').agg(requested_days=('date','size'),available_days=('available','sum'),raw_reports=('raw_reports','sum'),hourly_rows=('hourly_rows','sum')).reset_index()
    byyear.to_csv(OUT/'coverage_by_year.csv',index=False)
    summary={'status':'prepared','legacy_amendment_sha256':amendment,'original_training_rows':len(original),'added_unique_march_rows':len(added),'expanded_training_rows':len(expanded),
        'duplicate_original_rows_removed':len(duplicates),'expanded_training_days':int(expanded.date_key.nunique()),'available_march_days':int(sum(x['available'] for x in coverage)),
        'requested_march_days':465,'training_by_mode':expanded['mode'].value_counts().to_dict(),'training_by_year':expanded.hour.dt.year.value_counts().sort_index().to_dict(),
        'reserved_march_dates':[d for d in DATES if d in reserved],'compact_dataframe_bytes':int(full.memory_usage(deep=True).sum()),
        'train_sha256':sha(BASE/'expanded_train_features.parquet'),'source_sha256':{d:sha(BASE/'processed'/f'wspr-{d}.parquet') for d in DATES},
        'raw_sha256':{d:sha(BASE/'raw'/f'wspr-{d}.jsonl') for d in DATES},'plan_sha256':sha(OUT/'plan.json')}
    (OUT/'preparation.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['source_sha256','raw_sha256']},indent=2))
if __name__=='__main__':prepare()
