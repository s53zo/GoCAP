"""Sequential resumable465-date March sample, plus separate sealed2025 test dates."""
import datetime as dt,json,time,shutil,requests
import pyarrow.parquet as pq
from .march_plan import BASE,DATA,OUT,DATES,FRESH_TEST,plan,sha
from .prepare import process_wspr,annotate
from .v3_features import enrich

def collect():
    plan();raw=BASE/'raw';processed=BASE/'processed';raw.mkdir(exist_ok=True);processed.mkdir(exist_ok=True)
    session=requests.Session();state_path=OUT/'collection.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
    for date in DATES+FRESH_TEST:
        p=raw/f'wspr-{date}.jsonl';meta=p.with_suffix('.meta.json');out=processed/f'wspr-{date}.parquet'
        if p.exists():assert sha(p)==json.loads(meta.read_text())['sha256']
        else:
            old=DATA/'raw'/p.name;oldmeta=old.with_suffix('.meta.json')
            if old.exists() and oldmeta.exists():
                info=json.loads(oldmeta.read_text());assert sha(old)==info['sha256']
                assert 'cityHash64(id)%128=0' in info['query']
                shutil.copy2(old,p);shutil.copy2(oldmeta,meta)
            else:
                end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
                query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC') AND cityHash64(id)%128=0
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
                for attempt in range(4):
                    try:
                        r=session.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90));r.raise_for_status();lines=r.content.splitlines()
                        if len(lines)>=50001:raise ValueError('Query cap reached; no first-row truncation')
                        for line in lines:json.loads(line)
                        break
                    except (requests.RequestException,json.JSONDecodeError):
                        if attempt==3:raise
                        time.sleep(10)
                p.with_suffix('.part').write_bytes(r.content);p.with_suffix('.part').replace(p)
                meta.write_text(json.dumps({'query':query,'rows':len(lines),'sha256':sha(p),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2));time.sleep(3.2)
        if not out.exists():
            d,audit=process_wspr(p)
            if len(d):d=enrich(annotate(d))
            d.to_parquet(out,index=False);out.with_suffix('.meta.json').write_text(json.dumps({'raw_sha256':sha(p),'prepared_sha256':sha(out),'audit':audit},indent=2))
        info=json.loads(out.with_suffix('.meta.json').read_text());state[date]={'scope':'fresh_test' if date in FRESH_TEST else 'march_training_candidate','raw_reports':json.loads(meta.read_text())['rows'],'eligible_reports':int(info['audit'].get('eligible_rows',0)),'hourly_aggregates':pq.read_metadata(out).num_rows,'prepared_sha256':sha(out)}
        state_path.write_text(json.dumps(state,indent=2))
        print(date,'raw',state[date]['raw_reports'],'eligible reports',state[date]['eligible_reports'],'complete',len(state),'/',len(DATES+FRESH_TEST),flush=True)
    print('Full March temporal collection complete. Test predictions not generated.',flush=True)
if __name__=='__main__':collect()
