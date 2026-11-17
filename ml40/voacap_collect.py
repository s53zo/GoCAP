"""Fresh coherent WSPR reports for outcome-blind comparison case selection."""
import datetime as dt,json,time,requests
from .voacap_protocol import BASE,OUT,DATES,sha,protocol,COUNT
from .prepare import process_wspr,annotate
from .v3_features import enrich


def collect():
    protocol();raw=BASE/'raw';raw.mkdir(exist_ok=True);processed=BASE/'processed';processed.mkdir(exist_ok=True)
    session=requests.Session()
    solar=BASE/'silso_smoothed_v2.txt';solar_meta=OUT/'solar_source.json'
    if not solar.exists():
        url='https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.txt'
        response=session.get(url,timeout=60);response.raise_for_status()
        if solar_meta.exists():
            import hashlib
            if hashlib.sha256(response.content).hexdigest()!=json.loads(solar_meta.read_text())['sha256']:
                raise ValueError('Live SILSO series changed; restore frozen source or start a new versioned comparison')
        solar.write_bytes(response.content)
        if not solar_meta.exists():
            solar_meta.write_text(json.dumps({'url':url,'sha256':sha(solar),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'attribution':'Source: WDC-SILSO, Royal Observatory of Belgium, Brussels, DOI:10.24414/qnza-ac80; CC BY-NC research use'},indent=2))
    assert sha(solar)==json.loads(solar_meta.read_text())['sha256']
    for date in DATES:
        p=raw/f'wspr-{date}.jsonl';meta=p.with_suffix('.meta.json')
        if p.exists() and meta.exists():assert sha(p)==json.loads(meta.read_text())['sha256']
        else:
            end=(dt.date.fromisoformat(date)+dt.timedelta(days=1)).isoformat()
            for divisor in ([128,256,512] if COUNT==100 else [16,32,64]):
                cap=50001 if COUNT==100 else 250001
                sampling=f"cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%{divisor}=0"
                if COUNT==1000:sampling=f"(cityHash64(tx_sign,rx_sign,toString(toDate(time,'UTC')))%128=0 OR (distance>=2700 AND {sampling}))"
                query=f"""SELECT id,time,rx_sign,rx_loc,tx_sign,tx_loc,frequency,power,snr,drift,version,code
FROM wspr.rx WHERE band=7 AND time >= toDateTime('{date} 00:00:00','UTC')
AND time < toDateTime('{end} 00:00:00','UTC')
AND {sampling}
ORDER BY time,id LIMIT {cap} FORMAT JSONEachRow"""
                for attempt in range(3):
                    try:
                        r=session.get('https://db1.wspr.live/',params={'query':query},timeout=(15,90));r.raise_for_status()
                        records=r.content.splitlines()
                        for row in records:json.loads(row)
                        break
                    except (requests.RequestException,json.JSONDecodeError):
                        if attempt==2:raise
                        time.sleep(10)
                time.sleep(3.2)
                if len(records)<cap:break
            else:raise ValueError('Every declared sample reaches cap')
            p.write_bytes(r.content);meta.write_text(json.dumps({'query':query,'divisor':divisor,'rows':len(records),
                'sha256':sha(p),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2))
            print(date,len(records),'reports',flush=True)
        out=processed/f'wspr-{date}.parquet'
        if not out.exists():
            d,audit=process_wspr(p)
            if len(d):d=enrich(annotate(d))
            d.to_parquet(out,index=False)
            out.with_suffix('.meta.json').write_text(json.dumps({'input_sha256':sha(p),'audit':audit},indent=2))
    print('All comparison data collected. No predictions or scores generated.',flush=True)
if __name__=='__main__':collect()
