"""Archive Dourbes foF2 for a local, issue-time-lagged physics experiment."""
import json
import re
import time

import pandas as pd
import requests

from .five_data import CACHE, DATES, OUT, sha

SOURCE = 'https://giro.uml.edu/didbase/scaled.php'


def collect():
    folder = CACHE/'sonde'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for dates in DATES.values():
        for date in dates:
            path = folder/f'{date}.html'
            if not path.exists():
                start = pd.Timestamp(date) - pd.Timedelta(hours=2)
                end = pd.Timestamp(date) + pd.Timedelta(days=1)
                params = dict(location='DB049', date_start=start.strftime('%Y-%m-%d %H:%M'),
                              date_end=end.strftime('%Y-%m-%d %H:%M'), query_submit='Search',
                              DMUF='3000')
                params['chosenchars[]'] = '0'
                for attempt in range(3):
                    try:
                        r = requests.post(SOURCE, data=params, timeout=45)
                        r.raise_for_status()
                        break
                    except requests.RequestException:
                        if attempt == 2:
                            raise
                        time.sleep(5)
                path.write_text(r.text)
                time.sleep(1)
            rows = re.findall(r'(\d{4}-\d\d-\d\dT[\d:.]+Z)\s+(\d+)\s+([\d.]+)\s', path.read_text())
            table = pd.DataFrame(rows, columns=['time','confidence','fo'])
            table.to_csv(path.with_suffix('.csv'), index=False)
            manifest[date] = dict(samples=len(table), sha256=sha(path))
    (OUT/'sonde_sources.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    collect()
