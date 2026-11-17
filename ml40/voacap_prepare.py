"""Choose exactly 100 diverse cases before either model is scored."""
import datetime as dt,hashlib,json
from collections import deque
import numpy as np,pandas as pd
from .voacap_protocol import BASE,OUT,DATES,KEYS,sha,protocol,ENGINE_ROOT,COUNT,SEED,RESULTS


def select(d,seed=SEED,count=COUNT):
    candidates=d.groupby(KEYS,sort=True).agg(observed_hours=('hour','nunique'),distance_km=('distance_km','first')).reset_index()
    candidates=candidates[candidates.observed_hours.ge(6)&candidates.power_dbm.between(20,50)&candidates.distance_km.between(100,15000)].copy()
    candidates['distance_group']=pd.cut(candidates.distance_km,[99.999,1000,3000,6000,15000],
        labels=['100to1000','1000to3000','3000to6000','6000to15000']).astype(str)
    candidates['case_id']=[hashlib.sha256((seed+json.dumps(list(k),default=str)).encode()).hexdigest()[:20]
        for k in candidates[KEYS].itertuples(index=False,name=None)]
    rows=[];used=set()
    if count==1000:
        prior=pd.read_csv(RESULTS/'voacap100/cases_frozen.csv');used.update(zip(prior.tx_grid,prior.rx_grid))
    for distance in ['100to1000','1000to3000','3000to6000','6000to15000']:
        c=candidates[candidates.distance_group.eq(distance)].sort_values('case_id')
        queues=[deque(g.to_dict('records')) for _,g in c.groupby('date_key',sort=True)];target=count//4;count_done=0
        while count_done<target:
            progress=False
            for q in queues:
                while q:
                    row=q.popleft();path=(row['tx_grid'],row['rx_grid'])
                    if path in used:continue
                    used.add(path);rows.append(row);count_done+=1;progress=True;break
                if count_done==target:break
            if not progress:raise ValueError(f'Insufficient eligible paths for {distance}: {count_done}/{target}')
    return pd.DataFrame(rows).sort_values('case_id').reset_index(drop=True)


def prepare():
    definition=protocol()
    assert all((BASE/'processed'/f'wspr-{date}.meta.json').exists() for date in DATES)
    d=pd.concat([pd.read_parquet(BASE/'processed'/f'wspr-{date}.parquet') for date in DATES],ignore_index=True)
    chosen=select(d);assert len(chosen)==COUNT and chosen.case_id.nunique()==COUNT
    solar=pd.read_csv(BASE/'silso_smoothed_v2.txt',sep=r'\s+',header=None,names=['year','month','decimal_year','ssn','std','count','provisional'])
    chosen['year']=pd.to_datetime(chosen.date_key).dt.year;chosen['month']=pd.to_datetime(chosen.date_key).dt.month
    chosen=chosen.merge(solar[['year','month','ssn']],on=['year','month'],validate='many_to_one')
    assert chosen.ssn.notna().all() and chosen.ssn.ge(0).all()
    chosen['reported_power_watts']=10**((chosen.power_dbm-30)/10)
    features=d.merge(chosen[KEYS+['case_id']],on=KEYS,validate='many_to_one',how='inner').sort_values(['case_id','hour']).reset_index(drop=True)
    assert not features.duplicated(['case_id','hour']).any()
    assert len(set(chosen.tx_grid+'>'+chosen.rx_grid))==COUNT
    frozen={'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'case_ids':chosen.case_id.tolist(),
        'protocol_sha256':sha(OUT/'protocol.json'),'model_sha256':definition['model_sha256'],
        'solar_sha256':sha(BASE/'silso_smoothed_v2.txt'),
        'source_sha256':{p.name:sha(p) for p in sorted((BASE/'processed').glob('*.parquet'))},
        'selection_uses_snr_or_prediction':False}
    path=OUT/'case_freeze.json'
    if path.exists():
        old=json.loads(path.read_text());frozen['frozen_utc']=old['frozen_utc'];assert old==frozen
    path.write_text(json.dumps(frozen,indent=2));chosen.to_csv(OUT/'cases_frozen.csv',index=False)
    features.to_parquet(BASE/'case_features.parquet',index=False)
    from .voacap_engine import Engine,parse_output
    smoke=json.loads((ENGINE_ROOT/'dxtron-smoke.json').read_text())
    import re
    deck=re.sub(r'\[default/dipole\.voa\s*\]','['+'samples/sample.23'.ljust(21)+']',smoke['input_deck'])
    with Engine() as engine:
        local,parsed=engine.run(deck)
    remote=parse_output(smoke['raw_output'])
    for hour in range(24):assert parsed[hour]['SNR']==remote[hour]['SNR']
    (OUT/'engine_validation.json').write_text(json.dumps({'status':'verified','native_hours_matching_dxtron_public_api':24,
        'engine_binary_sha256':definition['voacap_binary_sha256'],
        'engine_source_commit':'c12a98b348c39f36f2659b8b3aa97b86284c8a1d',
        'pilot':'JN76 to FN31, April2025, SSN100,1W,type23 quarter-wave-height dipoles,noise150; not used for case selection',
        'api_source':'https://voacap.dxtron.com/api/openapi.json'},indent=2))
    print(f'Frozen {COUNT} cases:',chosen.groupby('distance_group').size().to_dict(),len(features),'observed hours; engine matches API.')
if __name__=='__main__':prepare()
