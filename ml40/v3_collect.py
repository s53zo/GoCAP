"""Outcome-blind fixed-pair coherent sequences; test samples remain unscored."""
import datetime as dt
import json,time,requests
from .v3_plan import BASE,TRAIN_DATES,TEST_DATES,plan
from .test500 import digest
from .prepare import process_wspr,annotate

def collect():
    plan();raw=BASE/'raw';processed=BASE/'processed'
    raw.mkdir(exist_ok=True);processed.mkdir(exist_ok=True);s=requests.Session()
    todo=[('train-coherent',d) for d in TRAIN_DATES]+[(kind,d) for d in TEST_DATES for kind in ['test-snr','test-coherent']]
    for kind,date in todo:
        path=raw/f'{kind}-{date}.jsonl';meta=path.with_suffix('.meta.json')
        if path.exists() and meta.exists():assert digest(path)==json.loads(meta.read_text())['sha256']
        else:
            end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
            for divisor in ([128] if kind=='test-snr' else [128,256,512]):
                clause=f'cityHash64(id)%{divisor}=0' if kind=='test-snr' else f"cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%{divisor}=0"
                query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC') AND {clause}
ORDER BY time,id LIMIT 50001 FORMAT JSONEachRow"""
                for attempt in range(3):
                    try:
                        r=s.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90));r.raise_for_status()
                        records=r.content.splitlines()
                        for row in records:json.loads(row)
                        break
                    except (requests.RequestException,json.JSONDecodeError):
                        if attempt==2:raise
                        time.sleep(10)
                time.sleep(3.2)
                if len(records)<50001:break
            else:raise ValueError('Every prespecified source sample reached query cap')
            if len(records)>=50001:raise ValueError('No truncated test sample permitted')
            path.write_bytes(r.content);meta.write_text(json.dumps({'query':query,'divisor':divisor,
                'rows':len(records),'sha256':digest(path),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2))
            print(kind,date,len(records),'positive reports',flush=True)
        output=processed/f'{kind}-{date}.parquet'
        if not output.exists():
            d,audit=process_wspr(path)
            if len(d):d=annotate(d)
            d.to_parquet(output,index=False)
            output.with_suffix('.meta.json').write_text(json.dumps({'input_sha256':digest(path),'audit':audit},indent=2))
    print('All v3 training and sealed test samples complete.',flush=True)
if __name__=='__main__':collect()
