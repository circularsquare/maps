"""Independently reconcile Irish Rail counts to raw calendars and schedules.

Does not import the builder. Checks saved path membership/order and reconstructs
all daily/directional link counts. This validates aggregation, not source
completeness or independent surveyed railway geometry.
"""
import collections
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def validate(root=ROOT):
    directory = root/'dist/data/ie'
    meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    diagnostic = json.loads((directory/'diagnostics.json').read_text(encoding='utf-8'))
    raw = root/'data/raw/ireland.zip'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==meta['sha256'], 'Different raw snapshot'
    z = zipfile.ZipFile(raw)
    def rows(n):
        return list(csv.DictReader(io.TextIOWrapper(z.open(n),encoding='utf-8-sig'))) if n in z.namelist() else []
    calendar = {r['service_id']:r for r in rows('calendar.txt')}
    exceptions = {(r['service_id'],r['date']):r['exception_type'] for r in rows('calendar_dates.txt')}
    stops = {r['stop_id']:r for r in rows('stops.txt')}
    def parent(s):
        return stops[s].get('parent_station') or s
    times = collections.defaultdict(list)
    for r in rows('stop_times.txt'):
        times[r['trip_id']].append(r)
    for records in times.values():
        records.sort(key=lambda r:int(r['stop_sequence']))
    routes = {r['route_id']:r for r in rows('routes.txt')}
    trips = rows('trips.txt')
    counts, forward, reverse = collections.defaultdict(lambda:[0]*len(meta['dates'])),collections.defaultdict(lambda:[0]*len(meta['dates'])),collections.defaultdict(lambda:[0]*len(meta['dates']))
    totals = []
    weekdays = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    seen_trip_codes = collections.defaultdict(set)
    inferred = 0
    for day, text in enumerate(meta['dates']):
        date = dt.date.fromisoformat(text)
        compact = date.strftime('%Y%m%d')
        seen = set()
        for trip in trips:
            if routes[trip['route_id']]['route_type']!='2':
                continue
            service = trip['service_id']
            row = calendar.get(service)
            enabled = bool(row and row['start_date']<=compact<=row['end_date'] and row[weekdays[date.weekday()]]=='1')
            if (service,compact) in exceptions:
                enabled = exceptions[service,compact]=='1'
            if not enabled:
                continue
            ordered = times[trip['trip_id']]
            identity = (trip.get('trip_short_name',''),tuple((parent(t['stop_id']),t['arrival_time'],t['departure_time']) for t in ordered))
            if identity in seen:
                continue
            seen.add(identity)
            seen_trip_codes[day,trip.get('trip_short_name','')].add(identity)
            path = diagnostic['trip_paths'][trip['trip_id']]
            scheduled = []
            for t in ordered:
                s = parent(t['stop_id'])
                if not scheduled or s!=scheduled[-1]:
                    scheduled.append(s)
            assert path[0]==scheduled[0] and path[-1]==scheduled[-1]
            cursor = 0
            for station in scheduled:
                cursor = path.index(station,cursor)+1
            inferred += len(path)-len(scheduled)
            for a,b in zip(path,path[1:]):
                assert a in stops and b in stops and a!=b
                key=tuple(sorted((a,b)))
                counts[key][day]+=1
                (forward if a==key[0] else reverse)[key][day]+=1
        totals.append(len(seen))
    assert totals==meta['daily_trains']
    assert inferred==diagnostic['inferred_pass_station_trip_days']
    features=json.loads((directory/'segments.geojson').read_text(encoding='utf-8'))['features']
    assert len(features)==len(counts)==meta['segments']
    assert len({f['id'] for f in features})==len(features)
    for f in features:
        p=f['properties']
        key=(p['from'],p['to'])
        assert p['daily']==counts.pop(key)
        assert p['forward']==forward[key] and p['reverse']==reverse[key]
        assert abs(p['average']-sum(p['daily'])/len(totals))<.00051
        assert sum(p['operators'].values())==sum(p['categories'].values())==sum(p['daily'])
        assert set(p['categories'])<={'commuter','intercity','other'}
        assert p['geometry_quality']=='feed shape'
        coordinates=f['geometry']['coordinates']
        assert len(coordinates)>=2
        assert all(-11<x<-5 and 51<y<55.5 for x,y in coordinates)
        for s,point in [(key[0],coordinates[0]),(key[1],coordinates[-1])]:
            assert point==[float(stops[s]['stop_lon']),float(stops[s]['stop_lat'])]
    assert not counts
    reused = [(day,code,len(values)) for (day,code),values in seen_trip_codes.items() if len(values)>1]
    print('PASS: independently reconciled',len(features),'corridors on',len(totals),'days; daily trains',totals)
    print('Train codes with distinct schedules on same day:',reused)
    print('Geometry validation is limited to saved path consistency; source route correctness is separately documented.')


if __name__=='__main__':
    validate()
