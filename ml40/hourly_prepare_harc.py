"""Prepare downloaded fresh PSK/RBN samples in the main ML environment."""
import json,time
import pandas as pd
from .hourly_plan import BASE,OUT,HARC_TEST_DATES,digest,active_harc_dates
from .prepare import process_harc,annotate
from .v3_features import enrich


def prepare():
    processed=BASE/'processed';processed.mkdir(exist_ok=True);audit={}
    for date in active_harc_dates():
        source_audit=OUT/'harc_audit'/f'harc-{date}-audit.json'
        while not source_audit.exists():
            state=json.loads((OUT/'harc_collection.json').read_text()) if (OUT/'harc_collection.json').exists() else {}
            if state.get(date,{}).get('status')=='unavailable':break
            print('Waiting for fresh archive extraction',date,flush=True);time.sleep(20)
        if not source_audit.exists():audit[date]={'status':'unavailable'};continue
        output=processed/f'harc-test-{date}.parquet';source=BASE/f'harc-{date}.parquet'
        assert digest(source)==json.loads(source_audit.read_text())['sample_parquet_sha256']
        if not output.exists():
            d,quality=process_harc(source)
            d=enrich(annotate(d));d.to_parquet(output,index=False)
        else:d=pd.read_parquet(output)
        audit[date]={'status':'complete','source_sha256':digest(source),'prepared_sha256':digest(output),
            'source_mode_rows':d.groupby(['source','mode']).size().rename('rows').reset_index().to_dict('records')}
        print('Fresh hourly source sample ready',date,len(d),'positive aggregates',flush=True)
    (OUT/'fresh_harc_preparation.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':prepare()
