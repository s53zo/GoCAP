"""Archive actual NOAA D-RAP maps dated at least five minutes before each issue."""
import hashlib
import io
import json
import re
import tarfile
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

from .four_data import CACHE, OUT, ALL, fetch

REMOTE = 'https://archive.data.noaa.gov/satellite-spaceweather/SWPC/Models/DRAP/SWX_DRAP20'


def parse_map(text):
    data = [np.fromstring(line.split('|')[1].replace('****','nan'),sep=' ') for line in text.splitlines() if '|' in line]
    grid = np.stack(data)
    lat = [int(line.split('|')[0]) for line in text.splitlines() if '|' in line]
    assert grid.shape==(90,90) and lat==list(range(89,-90,-2)) and np.isfinite(grid).any()
    return grid


def collect(date):
    folder = CACHE/'drap'
    folder.mkdir(exist_ok=True)
    out = folder/f'{date}.npz'
    log = out.with_suffix('.json')
    if out.exists() and log.exists():
        return json.loads(log.read_text())
    url = f'{REMOTE}/{date[:4]}/{date[5:7]}/SWX_DRAP20_C_SWPC_{date.replace("-","")}.tar.gz'
    pilot = folder/'pilot.tar.gz'
    if date=='2025-11-11' and pilot.exists():
        content = pilot.read_bytes()
    else:
        try:
            content = fetch(url).content
        except requests.HTTPError as exc:
            if exc.response.status_code!=404:
                raise
            content = b''
    grids = np.full((24,90,90),np.nan,dtype=np.float32)
    age = np.full(24,999)
    messages = {}
    if content:
        with tarfile.open(fileobj=io.BytesIO(content),mode='r|gz') as archive:
            for member in archive:
                stamp = re.search(r'(\d{8})(\d\d)(\d\d)00_GLOBAL\.txt$',member.name,re.I)
                if not stamp or stamp[1]!=date.replace('-',''):
                    continue
                h,m = int(stamp[2]),int(stamp[3])
                choose = h<23 and 5<=60-m<=15 and age[h+1]>60-m
                if not choose and m%5:
                    continue
                text = archive.extractfile(member).read().decode('ascii',errors='replace')
                if choose:
                    grids[h+1] = parse_map(text)
                    age[h+1] = 60-m
                if m%5==0:
                    line = re.search(r'X-RAY Message\s*:\s*([^\r\n]*)',text,re.I)
                    if line:
                        messages[f'{h:02}:{m:02}'] = line[1].strip()
    np.savez_compressed(out,grids=grids,age=age)
    log.write_text(json.dumps(dict(date=date,url=url,sha256=hashlib.sha256(content).hexdigest(),
                                   available=int((age<999).sum()),messages=messages),indent=2))
    print(date,int((age<999).sum()),flush=True)
    return json.loads(log.read_text())


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(collect,ALL))
    (OUT/'drap_sources.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='messages'} for r in result],indent=2))
