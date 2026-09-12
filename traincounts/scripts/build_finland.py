"""Build corridor counts from Fintraffic's passenger GTFS including pass-by points.

Python 3.9+, standard library only. No routing API or credentials required.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://rata.digitraffic.fi/api/v1/trains/gtfs-passenger.zip'
WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']


def service_days(calendar, exceptions, dates):
    days = collections.defaultdict(set)
    for row in calendar:
        for i, date in enumerate(dates):
            if row['start_date'] <= date.strftime('%Y%m%d') <= row['end_date'] and row[WEEKDAYS[date.weekday()]] == '1':
                days[row['service_id']].add(i)
    lookup = {date.strftime('%Y%m%d'): i for i, date in enumerate(dates)}
    for row in exceptions:
        if row['date'] in lookup:
            index = lookup[row['date']]
            if row['exception_type'] == '1':
                days[row['service_id']].add(index)
            elif row['exception_type'] == '2':
                days[row['service_id']].discard(index)
            else:
                raise ValueError('Unknown calendar exception')
    return days


def distance(a, b):
    return math.hypot((a[0]-b[0])*math.cos(math.radians((a[1]+b[1])/2)), a[1]-b[1])*111195


def locate(points, coords):
    """Monotone nearest vertices. Preserve repeated visits on loop services."""
    result, cursor = [], 0
    for n, coord in enumerate(coords):
        if n == len(coords)-1:
            # GTFS shapes terminate at the trip destination, including loop trips.
            index = len(points)-1
        else:
            index = min(range(cursor, len(points)), key=lambda i: distance(points[i], coord))
        result.append(index)
        cursor = index
    return result


def build(feed, start, count, output):
    dates = [start + dt.timedelta(days=i) for i in range(count)]
    archive = zipfile.ZipFile(feed)
    def rows(name):
        if name not in archive.namelist():
            return []
        return csv.DictReader(io.TextIOWrapper(archive.open(name), encoding='utf-8-sig'))
    info = list(rows('feed_info.txt'))[0]
    if dates[0].strftime('%Y%m%d') < info['feed_start_date'] or dates[-1].strftime('%Y%m%d') > info['feed_end_date']:
        raise ValueError('Requested dates lie outside the published feed window')
    active = service_days(rows('calendar.txt'), rows('calendar_dates.txt'), dates)
    agencies = {r['agency_id']: r['agency_name'] for r in rows('agency.txt')}
    routes = {r['route_id']: r for r in rows('routes.txt')}
    trips = {r['trip_id']: r for r in rows('trips.txt') if active[r['service_id']]}
    if list(rows('frequencies.txt')):
        raise ValueError('Frequency-based service needs a separate adapter; refusing to undercount')
    stops = {r['stop_id']: r for r in rows('stops.txt')}
    def parent(s):
        return stops[s]['parent_station'] or s
    def coord(s):
        return [float(stops[s]['stop_lon']), float(stops[s]['stop_lat'])]
    times = collections.defaultdict(list)
    passenger_stops = set()
    for row in rows('stop_times.txt'):
        if row['trip_id'] in trips:
            times[row['trip_id']].append(row)
            if row['pickup_type'] != '1' or row['drop_off_type'] != '1':
                passenger_stops.add(parent(row['stop_id']))
    shape_ids = {t['shape_id'] for t in trips.values()}
    shapes = collections.defaultdict(list)
    for row in rows('shapes.txt'):
        if row['shape_id'] in shape_ids:
            shapes[row['shape_id']].append((int(row['shape_pt_sequence']), [float(row['shape_pt_lon']), float(row['shape_pt_lat'])]))
    for sid, values in shapes.items():
        points = []
        for _, point in sorted(values):
            if not points or point != points[-1]:
                points.append(point)
        shapes[sid] = points
    segments, patterns = {}, {}
    totals = [0]*count
    physical = set()
    for tid, trip in trips.items():
        day_indices = active[trip['service_id']]
        for day in day_indices:
            # Finland trip IDs begin with the physical train number.
            physical_key = (tid.split('_')[0], day)
            if physical_key in physical:
                raise ValueError('Overlapping physical train schedules: '+str(physical_key))
            physical.add(physical_key)
            totals[day] += 1
        ordered = sorted(times[tid], key=lambda r: int(r['stop_sequence']))
        sequence = []
        for row in ordered:
            stop = parent(row['stop_id'])
            if not sequence or stop != sequence[-1]:
                sequence.append(stop)
        if len(sequence) < 2:
            raise ValueError('Active trip missing its stop sequence: '+tid)
        pattern = (trip['shape_id'], tuple(sequence))
        if pattern not in patterns:
            points = shapes.get(trip['shape_id'], [])
            indices = locate(points, [coord(s) for s in sequence]) if points else []
            parts = []
            for j, (a, b) in enumerate(zip(sequence, sequence[1:])):
                geometry = points[indices[j]:indices[j+1]+1] if points else []
                valid = len(geometry) >= 2 and distance(geometry[0], coord(a)) < 1500 and distance(geometry[-1], coord(b)) < 1500
                if not valid:
                    geometry = [coord(a), coord(b)]
                # Common parent coordinates keep adjoining corridor segments connected.
                geometry = [coord(a)] + geometry[1:-1] + [coord(b)]
                if a > b:
                    geometry = geometry[::-1]
                parts.append((a, b, geometry, valid))
            patterns[pattern] = parts
        route = routes[trip['route_id']]
        category = 'commuter' if route['route_type'] == '109' else 'other' if route['agency_id'] not in ('10','9995') else 'intercity'
        for a, b, geometry, valid in patterns[pattern]:
            key = tuple(sorted((a,b)))
            if key not in segments:
                segments[key] = {'daily':[0]*count, 'forward':[0]*count, 'reverse':[0]*count, 'categories':collections.Counter(), 'operators':collections.Counter(), 'candidates':{}, 'services':set()}
            seg = segments[key]
            for day in day_indices:
                seg['daily'][day] += 1
                seg['forward' if a == key[0] else 'reverse'][day] += 1
            seg['categories'][category] += len(day_indices)
            seg['operators'][agencies[route['agency_id']]] += len(day_indices)
            seg['services'].add(route['route_short_name'])
            signature = json.dumps(geometry, separators=(',',':'))
            candidate = seg['candidates'].setdefault(signature, [0, geometry, valid])
            candidate[0] += len(day_indices)
    if not segments or min(totals) == 0:
        raise ValueError('Empty network or a day without scheduled service')
    features = []
    fallbacks = 0
    for i, ((a,b), seg) in enumerate(sorted(segments.items())):
        candidate = max(seg['candidates'].values(), key=lambda c: (c[2], c[0]))
        fallbacks += not candidate[2]
        props = {'id':i, 'from':a, 'to':b, 'from_name':stops[a]['stop_name'], 'to_name':stops[b]['stop_name'], 'daily':seg['daily'], 'forward':seg['forward'], 'reverse':seg['reverse'], 'average':round(sum(seg['daily'])/count,3), 'categories':dict(seg['categories']), 'operators':dict(seg['operators']), 'services':sorted(seg['services']), 'geometry_quality':'feed shape' if candidate[2] else 'straight-line fallback'}
        features.append({'type':'Feature', 'id':i, 'properties':props, 'geometry':{'type':'LineString','coordinates':candidate[1]}})
    stations = [{'type':'Feature','properties':{'name':stops[s]['stop_name'],'code':s},'geometry':{'type':'Point','coordinates':coord(s)}} for s in sorted(passenger_stops)]
    meta = {'country':'Finland','country_code':'FI','dates':[d.isoformat() for d in dates], 'daily_trains':totals,'average_trains':round(sum(totals)/count,1),'segments':len(features),'stations':len(stations),'geometry_fallbacks':fallbacks,'source_url':URL,'source_name':'Fintraffic / Digitraffic','license':'CC BY 4.0','feed_version':info['feed_version'],'feed_start':info['feed_start_date'],'feed_end':info['feed_end_date'],'sha256':hashlib.sha256(feed.read_bytes()).hexdigest(),'generated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'metric':'Scheduled train traversals per service day, both directions. Service dates use Europe/Helsinki.','scope':'Passenger rail in the Fintraffic feed; includes VR and HSL rail, plus any scheduled heritage trains. Excludes metro, trams, freight and replacement buses. Includes any cross-border portions published in this feed.','geometry_note':'Corridors between parent timetable points, including pass-by points. Parallel tracks are combined; the most frequent usable shape represents each corridor. Not a physical-track inventory. Shared corridors with different timetable-point sequences may remain separate; no automatic routing is inferred.'}
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('segments.geojson', {'type':'FeatureCollection','features':features}),('stations.geojson',{'type':'FeatureCollection','features':stations}),('metadata.json',meta)]:
        (output/name).write_text(json.dumps(data, ensure_ascii=False, separators=(',',':')), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feed', type=Path, default=ROOT/'data/raw/finland.zip')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--start', type=dt.date.fromisoformat, default=dt.date(2026,9,14))
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/data/fi')
    args = parser.parse_args()
    if not 1 <= args.days <= 366:
        parser.error('--days must be 1 to 366')
    if args.download:
        args.feed.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL, headers={'User-Agent':'traincounts/0.1'})
        with urllib.request.urlopen(request, timeout=180) as response:
            args.feed.write_bytes(response.read())
    build(args.feed, args.start, args.days, args.output)
