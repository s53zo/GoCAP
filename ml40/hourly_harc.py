"""Three preselected untouched PSK/RBN daily archives; no model score inspected."""
import json,time
import requests
import collect_harc as harc
from collect import fetch
from .hourly_plan import BASE,OUT,HARC_TEST_DATES,plan,active_harc_dates
from .hourly_plan import digest

def collect():
    plan();harc.DATA=BASE;harc.AUDIT_DIR=OUT/'harc_audit';harc.AUDIT_DIR.mkdir(exist_ok=True)
    processed=BASE/'processed';processed.mkdir(exist_ok=True);state={}
    for date in active_harc_dates():
        output=BASE/f'harc-{date}.parquet'
        if output.exists():state[date]={'status':'complete','sha256':digest(output)};continue
        try:
            compact=date.replace('-','')
            fetch(f'rbn-{compact}.zip',f'https://data.reversebeacon.net/rbn_history/{compact}.zip',timeout=90)
            path=harc.download(date);harc.sample(date,path)
            state[date]={'status':'complete','sha256':digest(output)}
        except (ValueError,KeyError,RuntimeError,requests.RequestException) as exc:
            state[date]={'status':'unavailable','reason_type':type(exc).__name__}
            print('Fresh HARC unavailable',date,type(exc).__name__,flush=True)
        (OUT/'harc_collection.json').write_text(json.dumps(state,indent=2));time.sleep(3.2)
    (OUT/'harc_collection.json').write_text(json.dumps(state,indent=2))
if __name__=='__main__':collect()
