"""Independent raw-feed reconciliation of the published corridor counts."""
import collections
import csv
import datetime as dt
import io
import hashlib
import json
import math
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
meta=json.loads((root/'dist/data/fi/metadata.json').read_text(encoding='utf8'))
features=json.loads((root/'dist/data/fi/segments.geojson').read_text(encoding='utf8'))['features']
feed=root/'data/raw/finland.zip'
assert hashlib.sha256(feed.read_bytes()).hexdigest()==meta['sha256'], 'Raw feed differs from the snapshot; restore the original or rebuild the output'
z=zipfile.ZipFile(feed)
def rows(name):
    return csv.DictReader(io.TextIOWrapper(z.open(name),encoding='utf-8-sig'))
cal={r['service_id']:r for r in rows('calendar.txt')}
ex={(r['service_id'],r['date']):r['exception_type'] for r in rows('calendar_dates.txt')}
trips={r['trip_id']:r['service_id'] for r in rows('trips.txt')}
stops={r['stop_id']:r['parent_station'] or r['stop_id'] for r in rows('stops.txt')}
sequences=collections.defaultdict(list)
for r in rows('stop_times.txt'):
    sequences[r['trip_id']].append((int(r['stop_sequence']),stops[r['stop_id']]))
edges={}
for tid,seq in sequences.items():
    points=[s for _,s in sorted(seq)]
    edges[tid]=[(min(a,b),max(a,b)) for a,b in zip(points,points[1:]) if a!=b]
for day,iso in enumerate(meta['dates']):
    date=dt.date.fromisoformat(iso);stamp=date.strftime('%Y%m%d');weekday=['monday','tuesday','wednesday','thursday','friday','saturday','sunday'][date.weekday()]
    expected=collections.Counter();n=0
    for tid,sid in trips.items():
        rule=cal.get(sid,{})
        runs=rule.get('start_date','99999999')<=stamp<=rule.get('end_date','00000000') and rule.get(weekday)=='1'
        override=ex.get((sid,stamp))
        if override:runs=override=='1'
        if runs:
            expected.update(edges[tid]);n+=1
    actual={(f['properties']['from'],f['properties']['to']):f['properties']['daily'][day] for f in features if f['properties']['daily'][day]}
    assert dict(expected)==actual,iso
    assert n==meta['daily_trains'][day],iso
for f in features:
    p=f['properties']
    assert p['average']==round(sum(p['daily'])/len(meta['dates']),3)
    assert [a+b for a,b in zip(p['forward'],p['reverse'])]==p['daily']
    assert sum(p['operators'].values())==sum(p['categories'].values())==sum(p['daily'])
    coords=f['geometry']['coordinates']
    assert len(coords)>=2
    assert all(math.isfinite(x) and math.isfinite(y) and -180<=x<=180 and -90<=y<=90 for x,y in coords)
print(f"PASS: all {len(features)} corridors reconcile exactly against raw GTFS on all {len(meta['dates'])} dates; directions, categories, operators and coordinates valid.")
