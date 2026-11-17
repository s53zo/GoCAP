"""Create versioned Parquet feature samples; do not create a relational database."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import hourly,features,BASE,SOLAR
from .plan import DATA,RESULTS,manifest,split,station_held,geographic_held

PROCESSED=DATA/'processed'
PROCESSED.mkdir(exist_ok=True)

def process_wspr(path):
    d=pd.read_json(path,lines=True)
    if d.empty:return d,{'raw_rows':0,'eligible_rows':0}
    d['time']=pd.to_datetime(d.time,utc=True)
    # Legacy code 0 predates modern multimode metadata. Preserve that uncertainty
    # as an observation-context feature rather than silently dropping old years.
    legacy=d.time.lt(pd.Timestamp('2020-01-01',tz='UTC')) & d.code.eq(0)
    eligible=d.code.eq(1)|legacy
    context=np.where(legacy,'legacy_code0',
        np.where(d.time.lt(pd.Timestamp('2023-01-16',tz='UTC')),'pre2023_code1','ordinary_code1'))
    check={'raw_rows':len(d),'eligible_rows':int(eligible.sum()),
        'code_counts':d.code.value_counts().to_dict(),'observation_context':pd.Series(context)[eligible].value_counts().to_dict()}
    d=d[eligible].copy()
    if d.empty:return d,check
    d['decoder_context']=pd.Series(context,index=eligible.index)[eligible]
    d['mode']='WSPR';d['source']='WSPRNET'
    d['snr2500']=d.snr.astype(float);d['power_dbm']=d.power.astype(float)
    context_map=d.groupby('hour' if 'hour' in d else d.time.dt.floor('h')).decoder_context.first().to_dict()
    out=features(hourly(d),solar_start='2009')
    out['decoder_context']=out.hour.map(context_map).fillna('unknown')
    return out,check

def process_harc(path):
    d=pd.read_parquet(path)
    check={'raw_rows':len(d),'sources':d.source.value_counts().to_dict(),
        'invalid_tx_locations':int(d.tx_loc.isna().sum()),'invalid_rx_locations':int(d.rx_loc.isna().sum())}
    if d.empty:return d,check
    out=features(hourly(d),solar_start='2009')
    out['decoder_context']='archive_mode_known'
    return out,check

def annotate(d):
    d=d.copy()
    d['split']=d.hour.map(lambda t:split(t.date().isoformat()))
    d['date_key']=d.hour.dt.strftime('%Y-%m-%d')
    d['tx_key']=d['mode']+'|'+d.tx_sign+'|'+d.tx_grid
    d['rx_key']=d.source+'|'+d['mode']+'|'+d.rx_sign+'|'+d.rx_grid
    d['station_held']=d.tx_sign.map(station_held)|d.rx_sign.map(station_held)
    d['geographic_held']=d.tx_grid.map(geographic_held)|d.rx_grid.map(geographic_held)
    d['power_known']=d.power_dbm.notna().astype(float)
    d['legacy_decoder']=(d.decoder_context!='ordinary_code1').astype(float)
    return d

def prepare(allow_partial=False,for_fit=False):
    plan=manifest();checks={};parts=[]
    missing=[]
    paths=[(DATA/'raw'/f'wspr-{date}.jsonl','wspr') for date in plan['wspr_dates']]
    paths += [(DATA/f'harc-{date}.parquet','harc') for date in plan['harc_dates']]
    for path,kind in paths:
        if not path.exists():missing.append(str(path.relative_to(DATA)));continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        outpath=PROCESSED/f'{path.stem}.parquet';metapath=outpath.with_suffix('.meta.json')
        if outpath.exists() and metapath.exists() and json.loads(metapath.read_text())['input_sha256']==digest:
            out=pd.read_parquet(outpath);check=json.loads(metapath.read_text())['check']
        else:
            out,check=process_wspr(path) if kind=='wspr' else process_harc(path)
            out.to_parquet(outpath,index=False)
            metapath.write_text(json.dumps({'input_sha256':digest,'check':check},indent=2,default=int))
        if len(out):parts.append(out)
        checks[path.name]=check
    missing_only_test=all(name.startswith('harc-') and split(name[5:15])=='test' for name in missing)
    if missing and not allow_partial and not (for_fit and missing_only_test):
        raise ValueError(f'Collection incomplete: {len(missing)} planned files missing. Use --partial for preparation checks only.')
    if not parts:raise ValueError('No observations yet')
    d=pd.concat(parts,ignore_index=True)
    d=annotate(d)
    keys=['source','mode','hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']
    d['row_key']=pd.util.hash_pandas_object(d[keys],index=False).map(lambda x:f'{x:016x}')
    if d.row_key.duplicated().any():raise ValueError('Duplicate aggregate keys across prepared sources')
    d=d.sort_values(['source','mode','hour','tx_sign','rx_sign','power_dbm'],kind='stable').reset_index(drop=True)
    d.to_parquet(DATA/'features.parquet',index=False)
    fitting=d[d.split!='test'].reset_index(drop=True)
    fitting.to_parquet(DATA/'fit_features.parquet',index=False)
    summary={'status':('ready_for_fitting' if for_fit and missing_only_test else 'partial') if missing else 'complete','missing_files':missing,
        'rows':len(d),'actual_dates':d.date_key.nunique(),
        'by_source_mode_split':d.groupby(['source','mode','split']).size().rename('rows').reset_index().to_dict('records'),
        'solar_missing':d[SOLAR].isna().sum().to_dict(),'input_checks':checks,
        'features_sha256':hashlib.sha256((DATA/'features.parquet').read_bytes()).hexdigest(),
        'fit_features_sha256':hashlib.sha256((DATA/'fit_features.parquet').read_bytes()).hexdigest()}
    (RESULTS/'data_audit.json').write_text(json.dumps(summary,indent=2,default=int))
    print(summary['status'],len(d),'aggregates',summary['actual_dates'],'days',len(missing),'missing files',flush=True)
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--for-fit',action='store_true')
    args=p.parse_args();prepare(args.partial,args.for_fit)
