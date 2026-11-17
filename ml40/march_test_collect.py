"""Additional March2025 coherent receptions, with original source context processor."""
import datetime as dt,json,time,requests
from .march_test_plan import BASE,OUT,DATES,plan,sha
from .years_collect import process

def collect():
    plan();raw=BASE/'raw';processed=BASE/'processed';raw.mkdir(exist_ok=True);processed.mkdir(exist_ok=True);session=requests.Session()
    for date in DATES:
        p=raw/f'wspr-{date}.jsonl';meta=p.with_suffix('.meta.json');out=processed/p.with_suffix('.parquet').name
        if p.exists():assert sha(p)==json.loads(meta.read_text())['sha256']
        else:
            end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
            q=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC')
AND (cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%128=0 OR
(distance>=2700 AND cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%16=0))
ORDER BY time,id LIMIT 250001 FORMAT JSONEachRow"""
            for attempt in range(4):
                try:
                    r=session.get('https://db1.wspr.live/',params={'query':q},timeout=(15,90));r.raise_for_status();lines=r.content.splitlines()
                    assert len(lines)<250001,'No source truncation allowed'
                    for line in lines:json.loads(line)
                    break
                except (requests.RequestException,json.JSONDecodeError):
                    if attempt==3:raise
                    time.sleep(10)
            p.write_bytes(r.content);meta.write_text(json.dumps({'query':q,'rows':len(lines),'sha256':sha(p),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2));time.sleep(3.2)
        if not out.exists():
            d,audit=process(p);d.to_parquet(out,index=False);out.with_suffix('.meta.json').write_text(json.dumps({'raw_sha256':sha(p),'prepared_sha256':sha(out),'audit':audit},indent=2))
        print(date,json.loads(meta.read_text())['rows'],'reports ready',flush=True)
    print('Additional March2025 source collection complete; no scores generated.',flush=True)
if __name__=='__main__':collect()
