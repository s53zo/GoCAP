"""Consent-based CEDAR download, then local bounded sampling of 40m records.

Run using .venv-hdf (Python 3.14 has a prebuilt h5py wheel).
User identity is read from an ignored local file and excluded from provenance.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
import requests

from collect import RAW, DATA, ROOT, fetch
AUDIT_DIR=ROOT/'results'

def hash_selection(hashes,divisors):
    import numpy as np
    return np.remainder(np.asarray(hashes,dtype=np.uint64),np.asarray(divisors,dtype=np.uint64))==0

def experiments():
    p=fetch('madrigal-meta.html','https://cedar.openmadrigal.org/getMetadata?fileType=0',timeout=120)
    with p.open() as f:
        return {r[4]:r for r in csv.reader(f) if len(r)>8 and r[8]=='8308'}

def get_filename(date):
    row=experiments()[date.replace('-','')]
    p=fetch(f'harc-files-{date}.txt','https://cedar.openmadrigal.org/getExperimentFilesService.py',{'id':int(row[0])},timeout=120)
    rows=list(csv.reader(p.read_text().splitlines()))
    default=[r for r in rows if len(r)>=6 and r[3]=='1' and r[5]=='0']
    if len(default)!=1:
        raise ValueError(f'Expected one default public HARC file for {date}')
    return default[0][0]

def download(date):
    path=RAW/f'harc-{date}.hdf5'
    meta=RAW/f'harc-{date}.meta.json'
    if path.exists() and meta.exists():
        return path
    filename=get_filename(date)
    identity=json.loads((ROOT/'.cedar-identity.json').read_text())
    params={**identity,'fileName':filename,'fileType':-2}
    try:
        response=requests.get('https://cedar.openmadrigal.org/getMadfile.cgi',params=params,stream=True,timeout=(20,120))
        response.raise_for_status()
        size=int(response.headers.get('content-length',0))
        if size>8_000_000_000:
            response.close()
            raise ValueError(f'File exceeds 8GB pilot cap: {size}')
        print(f'{date}: downloading {size/1e9:.2f} GB; server cannot filter files over 200MB',flush=True)
        digest=hashlib.sha256();total=0;last=0
        part=path.with_suffix('.part')
        with part.open('wb') as f:
            for chunk in response.iter_content(8*1024*1024):
                f.write(chunk);digest.update(chunk);total+=len(chunk)
                if total-last>=512*1024*1024:
                    print(f'{date}: {total/1e9:.2f}/{size/1e9:.2f} GB',flush=True);last=total
        if total!=size:
            raise ValueError('Incomplete download')
        with part.open('rb') as f:
            if f.read(8)!=b'\x89HDF\r\n\x1a\n':
                raise ValueError('Expected HDF5')
        part.replace(path)
        meta.write_text(json.dumps({'endpoint':'https://cedar.openmadrigal.org/getMadfile.cgi',
            'params':{'fileName':filename,'fileType':-2},'identity':'user supplied; consent recorded in conversation',
            'bytes':total,'sha256':digest.hexdigest(),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2))
        return path
    except requests.RequestException as e:
        # Requests exceptions contain URLs with identity; never print those URLs.
        raise RuntimeError(f'CEDAR download failed ({type(e).__name__}); private request details omitted') from None

def inspect(path):
    import h5py
    with h5py.File(path,'r') as f:
        def show(name,obj):
            if isinstance(obj,h5py.Dataset):
                print(name,obj.shape,obj.dtype,flush=True)
        f.visititems(show)
        if 'Data/Table Layout' in f:
            t=f['Data/Table Layout']
            print('FIRST ROWS:',t[:3],flush=True)
        for name in ['Metadata/Data Parameters','Metadata/Experiment Notes']:
            if name in f:
                print(name,f[name][:],flush=True)

def sample(date,path):
    import h5py
    import numpy as np
    import pandas as pd
    from collections import Counter
    checkpoint=DATA/'intermediate'/f'harc-selected-v2-{date}.parquet'
    if checkpoint.exists():
        state=json.loads(checkpoint.with_suffix('.json').read_text())
        return finalize_sample(date,pd.read_parquet(checkpoint),**state)
    parts=[];counts=Counter();offsets=Counter();modecounts=Counter()
    with h5py.File(path,'r') as f:
        table=f['Data/Table Layout']
        total_rows=len(table)
        for start in range(0,len(table),1_000_000):
            a=table[start:start+1_000_000]
            a=a[(a['tfreq']>=7e6)&(a['tfreq']<=7.3e6)]
            if not len(a):continue
            d=pd.DataFrame.from_records(a)
            for col in ['year','month','day','hour','min','sec','call_sign_tx','call_sign_rx','smode','ssrc']:
                d[col]=d[col].str.decode('ascii',errors='replace').str.strip()
            t=pd.to_datetime({'year':d.year.astype(int),'month':d.month.astype(int),'day':d.day.astype(int),
                'hour':d.hour.astype(int),'minute':d['min'].astype(int),'second':d.sec.astype(int)},utc=True)
            d['time']=t
            corrected_epoch=t.astype('int64')//(1_000_000 if t.dtype.unit=='us' else 1_000_000_000)
            # pandas datetime resolution varies by version; Timestamp.timestamp gives a check.
            assert int(corrected_epoch.iloc[0])==int(t.iloc[0].timestamp())
            offsets.update((d.ut1_unix-corrected_epoch).value_counts().to_dict())
            counts.update(d.ssrc.value_counts().to_dict())
            modecounts.update((d.ssrc+':'+d.smode).value_counts().to_dict())
            # Event-key hashing keeps all duplicate copies of selected reports for the audit.
            key=d[['time','call_sign_tx','call_sign_rx','tfreq','smode','ssrc']]
            hashes=pd.util.hash_pandas_object(key,index=False)
            # Keep both operands uint64. Mixing uint64 hashes and int64 divisors
            # can promote to float64 and destroy the low hash bits.
            divisor=np.where(d.ssrc.eq('RBN'),4,
                np.where(d.ssrc.eq('PSK') & d.smode.eq('FT4'),256,1024)).astype(np.uint64)
            selected=hash_selection(hashes.to_numpy(),divisor)
            psk=d.ssrc.eq('PSK').to_numpy()
            if psk.sum()>1000 and (selected & psk).sum()>.1*psk.sum():
                raise ValueError('Unexpected PSK sample fraction; refusing biased sample')
            d=d[selected].copy()
            d['event_hash']=hashes[selected].to_numpy()
            parts.append(d)
            if start%10_000_000==0:
                print(date,'scanned',start+len(a),'/',len(table),'sampled',sum(len(p) for p in parts),flush=True)
    d=pd.concat(parts,ignore_index=True)
    checkpoint.parent.mkdir(exist_ok=True)
    d.to_parquet(checkpoint,index=False)
    checkpoint.with_suffix('.json').write_text(json.dumps({'counts':dict(counts),
        'offsets':{str(k):int(v) for k,v in offsets.items()},'modecounts':dict(modecounts),
        'total_rows':total_rows},default=int))
    return finalize_sample(date,d,counts,offsets,modecounts,total_rows)

def finalize_sample(date,d,counts,offsets,modecounts,total_rows):
    import numpy as np
    import pandas as pd
    # Keep native source values and original coordinates for later provenance checks.
    d=d.rename(columns={'call_sign_tx':'tx_sign','call_sign_rx':'rx_sign','tfreq':'frequency','smode':'mode'})
    d['source']=d.ssrc.map({'RBN':'HARC_RBN','PSK':'HARC_PSK','WSP':'HARC_WSPR'}).fillna('HARC_UNKNOWN')
    d['snr2500']=d.sn
    d['power_dbm']=np.nan
    d['location_provenance']='HARC geolocated coordinates; per-record method unavailable'
    def grid(lat,lon):
        if not np.isfinite(lat) or not np.isfinite(lon) or not -90<=lat<90 or not -180<=lon<180:
            return None
        x=lon+180;y=lat+90
        return chr(65+int(x//20))+chr(65+int(y//10))+str(int((x%20)//2))+str(int(y%10))
    d['tx_loc']=[grid(a,b) for a,b in zip(d.txlat,d.txlon)]
    d['rx_loc']=[grid(a,b) for a,b in zip(d.rxlat,d.rxlon)]
    rbn=d.ssrc.eq('RBN')
    rawpath=RAW/f'rbn-{date.replace("-","")}.zip'
    import zipfile
    with zipfile.ZipFile(rawpath) as z:
        raw=pd.read_csv(z.open(z.namelist()[0]))
    raw=raw.rename(columns={'callsign':'rx_sign','dx':'tx_sign','date':'time'})
    raw=raw[raw.band.eq('40m') & pd.to_numeric(raw.freq,errors='coerce').notna()].copy()
    raw['time']=pd.to_datetime(raw.time,utc=True)
    raw['frequency']=(raw.freq*1000).round().astype('int64')
    keys=['time','tx_sign','rx_sign','frequency']
    native=raw.groupby(keys).agg(mode_count=('tx_mode','nunique'),verified_mode=('tx_mode','first'),native_snr=('db','median')).reset_index()
    d['frequency']=d.frequency.round().astype('int64')
    d=d.merge(native,on=keys,how='left',validate='many_to_one')
    rbn=d.ssrc.eq('RBN')
    matched=rbn & d.mode_count.eq(1)
    agreement=(d.loc[matched,'sn']-d.loc[matched,'native_snr']).abs()
    check={'date':date,'all_band_rows':total_rows,'rows_40m_by_native_source':dict(counts),
        'raw_source_mode_counts_40m':dict(modecounts),'unix_minus_utc_fields_seconds_40m':{str(k):int(v) for k,v in offsets.items()},
        'sampled_rows':len(d),'sampled_rbn':int(rbn.sum()),'sampled_rbn_matched_to_original_utc':int(matched.sum()),
        'rbn_snr_exact_agreement_fraction':float(agreement.eq(0).mean()),
        'sample_divisors':{'RBN':4,'PSK_FT4':256,'PSK_other':1024,'WSP':1024},
        'sample_expansion':'FT4 expanded from 1/1024 to 1/256 after strict training count was below the fixed 1000-row gate; other source/mode selections unchanged',
        'sampling':'pandas hash of time,tx,rx,frequency,native_mode,source; duplicate event copies retained',
        'timestamp_policy':'explicit UTC fields, cross-checked against original RBN event times; raw ut1_unix retained',
        'geolocation_policy':'coordinates mapped to 4-character squares; lookup vs reported provenance unavailable'}
    d.loc[matched,'mode']=d.loc[matched,'verified_mode']
    d.loc[rbn & ~matched,'mode']='UNVERIFIED_RBN_MODE'
    d.loc[rbn & d['mode'].eq('CW'),'snr2500']=d.loc[rbn & d['mode'].eq('CW'),'native_snr']-10*np.log10(5)
    # Only RBN CW and PSK FT8/FT4 have established measurement conventions here.
    known=(rbn & d['mode'].eq('CW'))|(d.ssrc.eq('PSK') & d['mode'].isin(['FT8','FT4']))
    d.loc[~known,'snr2500']=np.nan
    d['event_id']='harc:'+date+':'+d.recno.astype(str)
    d.to_parquet(DATA/f'harc-{date}.parquet',index=False)
    check['sample_parquet_sha256']=hashlib.sha256((DATA/f'harc-{date}.parquet').read_bytes()).hexdigest()
    AUDIT_DIR.mkdir(parents=True,exist_ok=True)
    (AUDIT_DIR/f'harc-{date}-audit.json').write_text(json.dumps(check,indent=2,default=int))
    print(json.dumps(check,indent=2,default=int),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dates',nargs='+')
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args()
    for date in args.dates:
        path=download(date)
        if args.inspect_only:inspect(path)
        else:sample(date,path)
