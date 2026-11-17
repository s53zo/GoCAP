"""Bounded seasonal HARC collection, isolated from the original pilot outputs."""
import hashlib
import json
import shutil
import time
from pathlib import Path
import requests
import collect_harc as harc
from collect import fetch,RAW as SHARED_RAW,DATA as PILOT_DATA
from .plan import DATA,RESULTS,HARC_DATES,manifest

def run():
    manifest()
    audits=RESULTS/'harc_audit';audits.mkdir(exist_ok=True)
    harc.DATA=DATA
    harc.AUDIT_DIR=audits
    state_path=RESULTS/'harc_collection.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    for date in HARC_DATES:
        out=DATA/f'harc-{date}.parquet'
        audit=audits/f'harc-{date}-audit.json'
        if out.exists() and audit.exists():
            assert hashlib.sha256(out.read_bytes()).hexdigest()==json.loads(audit.read_text())['sample_parquet_sha256']
            print(date,'cached sample',flush=True);continue
        pilot=PILOT_DATA/f'harc-{date}.parquet'
        pilot_audit=RESULTS.parent/f'harc-{date}-audit.json'
        if pilot.exists() and pilot_audit.exists():
            shutil.copy2(pilot,out);shutil.copy2(pilot_audit,audit)
            state[date]={'status':'complete','origin':'verified pilot sample'}
            state_path.write_text(json.dumps(state,indent=2));continue
        try:
            compact=date.replace('-','')
            fetch(f'rbn-{compact}.zip',f'https://data.reversebeacon.net/rbn_history/{compact}.zip',timeout=90)
            state[date]={'status':'downloading'};state_path.write_text(json.dumps(state,indent=2))
            path=harc.download(date)
            state[date]={'status':'extracting'};state_path.write_text(json.dumps(state,indent=2))
            harc.sample(date,path)
            state[date]={'status':'complete','origin':'CEDAR daily file + original RBN'}
        except (ValueError,KeyError,RuntimeError,requests.RequestException) as exc:
            # Private credentials are never included in state or exception URLs.
            state[date]={'status':'unavailable','reason_type':type(exc).__name__}
            print(date,'unavailable',type(exc).__name__,flush=True)
        state_path.write_text(json.dumps(state,indent=2))
        time.sleep(3.2)

if __name__=='__main__':run()
