"""Independent, outcome-blind pair/day samples for coherent hourly ranking."""
import datetime as dt
import hashlib
import json
import time
import requests
from .plan import DATA,RESULTS,manifest,split
from .prepare import process_wspr

def run():
    plan=manifest()
    # Share the provider politely: the main collector also uses a 3.2s pause.
    # Wait for its bounded batch to finish rather than doubling its request rate.
    while not all((DATA/'raw'/f'wspr-{d}.meta.json').exists() for d in plan['wspr_dates']):
        print('Waiting for primary WSPR batch to finish before ranking queries',flush=True)
        time.sleep(30)
    selected=[d for d in plan['wspr_dates'] if split(d)!='train']
    cohort_plan={'dates':selected,'pair_day_hash_divisor':512,
        'fallback_divisor_if_50001_row_cap_reached':1024,
        'purpose':'retain all reported hours for outcome-blind sampled TX/RX/day keys; no absent-hour labels',
        'separation':'ranking evaluation only; not appended to training or used for final-test model selection'}
    p=RESULTS/'ranking_plan.json'
    if p.exists() and json.loads(p.read_text())!=cohort_plan:raise ValueError('Ranking plan changed')
    p.write_text(json.dumps(cohort_plan,indent=2))
    raw=DATA/'ranking'/'raw';raw.mkdir(parents=True,exist_ok=True)
    processed=DATA/'ranking'/'processed';processed.mkdir(exist_ok=True)
    session=requests.Session()
    for date in selected:
        path=raw/f'cohort-{date}.jsonl';meta=path.with_suffix('.meta.json')
        if not (path.exists() and meta.exists()):
            end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
            for divisor in [512,1024]:
                query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC')
AND cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%{divisor}=0
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
                r=session.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90));r.raise_for_status()
                rows=r.content.splitlines()
                for row in rows:json.loads(row)
                if len(rows)<50001:break
                time.sleep(3.2)
            else:raise ValueError('Ranking cohort exceeds both prespecified caps')
            path.write_bytes(r.content)
            meta.write_text(json.dumps({'query':query,'rows':len(rows),'divisor':divisor,
                'sha256':hashlib.sha256(r.content).hexdigest()},indent=2))
            print(date,'ranking rows',len(rows),flush=True)
            time.sleep(3.2)
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads(meta.read_text())['sha256']
        output=processed/f'cohort-{date}.parquet'
        if not output.exists():
            frame,audit=process_wspr(path)
            frame.to_parquet(output,index=False)

if __name__=='__main__':run()
