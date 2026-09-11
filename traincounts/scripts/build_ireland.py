"""Build Irish Rail corridors by projecting all stations onto official GTFS shapes.

Standard library only. Express trains inherit stations passed on their shape,
even when those stations do not appear in the train's boarding-stop sequence.
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

from build_finland import service_days, distance

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://www.transportforireland.ie/transitData/Data/GTFS_Irish_Rail.zip'
DOC = 'https://data.gov.ie/dataset/nta-gtfs'
TOLERANCE = 100  # metres; diagnostics check every scheduled stop against this
STATION_TOLERANCES = {'8220IR0007': 300}  # Connolly's terminal and through platforms
REVIEWED_SHA256 = '5cd1e163a5b0a0bc4d18982cd600be3e69572040edce87dfffec8cc666bf158d'
REVIEWED_SHAPE_CORRECTIONS = {'5882_930':'5882_186','5882_931':'5882_186','5882_932':'5882_186','5882_160':'5882_159'}


def projection(p, a, b):
    scale = math.cos(math.radians(p[1]))
    dx, dy = (b[0]-a[0])*scale, b[1]-a[1]
    px, py = (p[0]-a[0])*scale, p[1]-a[1]
    fraction = max(0, min(1, (px*dx+py*dy)/(dx*dx+dy*dy))) if dx or dy else 0
    return fraction, math.hypot(px-fraction*dx, py-fraction*dy)*111195


def project_stations(points, coords, tolerance=TOLERANCE, station_tolerances=None):
    """Return visits along a shape. A station can recur on an out-and-back shape.

    Candidate edges are spatially filtered; contiguous near-station edges form
    one visit, so a later return to the station remains a separate visit.
    """
    grid = collections.defaultdict(list)
    for sid, p in coords.items():
        grid[(math.floor(p[0]*100), math.floor(p[1]*100))].append(sid)
    near = collections.defaultdict(list)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        for x in range(math.floor(min(a[0],b[0])*100)-1, math.floor(max(a[0],b[0])*100)+2):
            for y in range(math.floor(min(a[1],b[1])*100)-1, math.floor(max(a[1],b[1])*100)+2):
                for sid in grid.get((x,y), []):
                    fraction, error = projection(coords[sid], a, b)
                    if error <= (station_tolerances or {}).get(sid,tolerance):
                        near[sid].append((index, index+fraction, error))
    visits = []
    for sid, matches in near.items():
        groups = []
        for match in matches:
            radius = (station_tolerances or {}).get(sid,tolerance)
            if not groups or match[0] > groups[-1][-1][0]+1 or distance(points[match[0]],coords[sid]) > radius:
                groups.append([])
            groups[-1].append(match)
        for group in groups:
            _, pos, error = min(group, key=lambda m:m[2])
            visits.append((pos,sid,error))
    # A winding approach can enter the station tolerance twice without reaching
    # another station. It is one corridor node; returns after other nodes remain.
    collapsed = []
    for visit in sorted(visits):
        if collapsed and collapsed[-1][1] == visit[1]:
            if visit[2] < collapsed[-1][2]:
                collapsed[-1] = visit
        else:
            collapsed.append(visit)
    return collapsed


def physical_signature(trip, times, parent):
    """Exact normalized schedule identity, not a portable train-number guess."""
    return (trip.get('trip_short_name',''), tuple((parent(t['stop_id']),t['arrival_time'],t['departure_time']) for t in times))


def route_category(route):
    if route['route_short_name'] == 'DART' or route['route_id'] in {
        'DUB-DRO/DUN-O', 'DUB-MAYNOOTH-O', 'DUB-PORTLAOIS-O', 'MAL-COBH-O'}:
        return 'commuter'
    return 'intercity'


def match_path(visits, sequence):
    cursor, indices = 0, []
    for sid in sequence:
        candidates = [i for i in range(cursor,len(visits)) if visits[i][1]==sid]
        if not candidates:
            return None
        index = candidates[0]
        indices.append(index)
        cursor=index+1
    return visits[indices[0]:indices[-1]+1]


def build(feed, start, count, output):
    archive = zipfile.ZipFile(feed)
    checksum = hashlib.sha256(feed.read_bytes()).hexdigest()
    provenance_file = feed.with_suffix('.http.json')
    provenance = json.loads(provenance_file.read_text(encoding='utf-8')) if provenance_file.exists() else {}
    if provenance.get('sha256') != checksum:
        provenance = {}
    def rows(name):
        return list(csv.DictReader(io.TextIOWrapper(archive.open(name),encoding='utf-8-sig'))) if name in archive.namelist() else []
    dates = [start+dt.timedelta(days=i) for i in range(count)]
    info = rows('feed_info.txt')[0]
    if dates[0].strftime('%Y%m%d') < info['feed_start_date'] or dates[-1].strftime('%Y%m%d') > info['feed_end_date']:
        raise ValueError('Requested window outside published feed range')
    if rows('frequencies.txt'):
        raise ValueError('Frequency expansion is required for this feed')
    active = service_days(rows('calendar.txt'),rows('calendar_dates.txt'),dates)
    agencies = {r['agency_id']:r for r in rows('agency.txt')}
    routes = {r['route_id']:r for r in rows('routes.txt') if r['route_type']=='2'}
    trips = {r['trip_id']:r for r in rows('trips.txt') if r['route_id'] in routes and active[r['service_id']]}
    if any(not t.get('trip_short_name') for t in trips.values()):
        raise ValueError('Missing Irish Rail train code; review physical-service identity before importing')
    stops = {r['stop_id']:r for r in rows('stops.txt')}
    def parent(s):
        return stops[s].get('parent_station') or s
    coords = {parent(s):[float(stops[parent(s)]['stop_lon']),float(stops[parent(s)]['stop_lat'])] for s in stops}
    times = collections.defaultdict(list)
    passenger = set()
    for r in rows('stop_times.txt'):
        if r['trip_id'] in trips:
            times[r['trip_id']].append(r)
            if r.get('pickup_type') != '1' or r.get('drop_off_type') != '1':
                passenger.add(parent(r['stop_id']))
    for tid in times:
        times[tid].sort(key=lambda r:int(r['stop_sequence']))
    shapes = collections.defaultdict(list)
    needed = {t['shape_id'] for t in trips.values()}
    for r in rows('shapes.txt'):
        if r['shape_id'] in needed:
            shapes[r['shape_id']].append((int(r['shape_pt_sequence']),[float(r['shape_pt_lon']),float(r['shape_pt_lat'])]))
    shapes = {s:[p for _,p in sorted(v)] for s,v in shapes.items()}
    projected = {s:project_stations(p,coords,station_tolerances=STATION_TOLERANCES) for s,p in shapes.items()}
    print('Projected',len(shapes),'active shapes',flush=True)
    segments = {}
    totals, raw_totals = [0]*count, [0]*count
    dedup = set()
    train_codes = set()
    diagnostic = {'projection_tolerance_m':TOLERANCE,'station_tolerance_overrides_m':STATION_TOLERANCES,'active_shapes':len(shapes),'unmatched_scheduled_stops':[], 'duplicate_trip_days_removed':0,'inferred_pass_station_trip_days':0,'max_scheduled_stop_error_m':0,'repeated_station_shapes':[], 'shape_substitutions':{},'trip_paths':{}}
    for sid, visits in projected.items():
        repeated = [s for s,n in collections.Counter(v[1] for v in visits).items() if n>1]
        if repeated:
            diagnostic['repeated_station_shapes'].append({'shape_id':sid,'stations':repeated})
    for tid, trip in trips.items():
        ordered = times[tid]
        signature = physical_signature(trip,ordered,parent)
        days = []
        for day in sorted(active[trip['service_id']]):
            raw_totals[day] += 1
            if (day,signature) in dedup:
                diagnostic['duplicate_trip_days_removed'] += 1
            else:
                if (day,trip['trip_short_name']) in train_codes:
                    raise ValueError('Distinct schedules reuse a train code on the same date; review identity: '+trip['trip_short_name'])
                train_codes.add((day,trip['trip_short_name']))
                dedup.add((day,signature))
                days.append(day)
                totals[day] += 1
        if not days:
            continue
        shape_id = trip['shape_id']
        visits = projected.get(shape_id,[])
        sequence = []
        for t in ordered:
            sid = parent(t['stop_id'])
            if not sequence or sid != sequence[-1]:
                sequence.append(sid)
        initial_path = match_path(visits,sequence)
        if initial_path and len(set(v[1] for v in initial_path)) < len(initial_path) and len(set(sequence)) == len(sequence):
            # Known source defects contain a large unscheduled out-and-back.
            # Substitute only another official shape with the identical ordered
            # scheduled stations and an unambiguous, non-repeating station path.
            if checksum!=REVIEWED_SHA256 or shape_id not in REVIEWED_SHAPE_CORRECTIONS:
                raise ValueError('Unreviewed repeated-station shape needs investigation: '+shape_id)
            alternative=REVIEWED_SHAPE_CORRECTIONS[shape_id]
            pp=match_path(projected[alternative],sequence)
            if not pp or len(set(v[1] for v in pp))!=len(pp):
                raise ValueError('Reviewed shape correction no longer fits scheduled stops')
            diagnostic['shape_substitutions'][tid] = {'original':shape_id,'replacement':alternative,'reason':'Snapshot-specific inferred correction: unscheduled repeated-station detour; alternate official shape follows all scheduled stops in order.'}
            shape_id = alternative
            visits = projected[shape_id]
        cursor, indices = 0, []
        for sid in sequence:
            candidates = [i for i in range(cursor,len(visits)) if visits[i][1]==sid]
            if not candidates:
                diagnostic['unmatched_scheduled_stops'].append({'trip_id':tid,'stop_id':sid,'shape_id':trip['shape_id']})
                break
            index = candidates[0]
            indices.append(index)
            cursor = index+1
            diagnostic['max_scheduled_stop_error_m'] = max(diagnostic['max_scheduled_stop_error_m'],visits[index][2])
        if len(indices) != len(sequence):
            continue
        path = visits[indices[0]:indices[-1]+1]
        if len(path)<2:
            raise ValueError('No route for active trip '+tid)
        diagnostic['trip_paths'][tid] = [s for _,s,_ in path]
        diagnostic['inferred_pass_station_trip_days'] += (len(path)-len(sequence))*len(days)
        route = routes[trip['route_id']]
        for (start_pos,a,_),(end_pos,b,_) in zip(path,path[1:]):
            if a==b:
                raise ValueError('Station loop without intermediate node '+a)
            key = tuple(sorted((a,b)))
            seg = segments.setdefault(key,{'daily':[0]*count,'forward':[0]*count,'reverse':[0]*count,'categories':collections.Counter(),'operators':collections.Counter(),'services':set(),'candidates':{}})
            for day in days:
                seg['daily'][day]+=1
                seg['forward' if a==key[0] else 'reverse'][day]+=1
            seg['categories'][route_category(route)] += len(days)
            seg['operators'][agencies[route['agency_id']]['agency_name']] += len(days)
            seg['services'].add(route['route_long_name'])
            points = shapes[shape_id]
            geometry = [coords[a]] + points[math.floor(start_pos)+1:math.ceil(end_pos)] + [coords[b]]
            if a!=key[0]:
                geometry.reverse()
            encoded = json.dumps(geometry,separators=(',',':'))
            candidate = seg['candidates'].setdefault(encoded,[0,geometry])
            candidate[0]+=len(days)
    output.mkdir(parents=True,exist_ok=True)
    (output/'diagnostics.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf-8')
    if diagnostic['unmatched_scheduled_stops']:
        raise ValueError('Unmatched scheduled stops; inspect diagnostics.json before publishing')
    if not totals or not min(totals):
        raise ValueError('Empty scheduled service day')
    features = []
    for i,((a,b),seg) in enumerate(sorted(segments.items())):
        geometry = max(seg['candidates'].values(),key=lambda x:x[0])[1]
        props = {'id':i,'from':a,'to':b,'from_name':stops[a]['stop_name'],'to_name':stops[b]['stop_name'],'daily':seg['daily'],'forward':seg['forward'],'reverse':seg['reverse'],'average':round(sum(seg['daily'])/count,3),'categories':dict(seg['categories']),'operators':dict(seg['operators']),'services':sorted(seg['services']),'geometry_quality':'feed shape'}
        features.append({'type':'Feature','id':i,'properties':props,'geometry':{'type':'LineString','coordinates':geometry}})
    stations = [{'type':'Feature','properties':{'name':stops[s]['stop_name'],'code':s},'geometry':{'type':'Point','coordinates':coords[s]}} for s in sorted(passenger)]
    meta = {'country':'Ireland · Irish Rail','country_code':'IE','dates':[d.isoformat() for d in dates],'daily_trains':totals,'raw_daily_trips':raw_totals,'average_trains':round(sum(totals)/count,1),'segments':len(features),'stations':len(stations),'geometry_fallbacks':0,'source_url':URL,'source_name':'National Transport Authority / Transport for Ireland','license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','documentation_url':DOC,'attribution':'Contains Irish Public Sector Data licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) licence.','feed_version':info['feed_version'],'feed_start':info['feed_start_date'],'feed_end':info['feed_end_date'],'feed_last_modified':provenance.get('last_modified'),'sha256':hashlib.sha256(feed.read_bytes()).hexdigest(),'generated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'timezone':next(iter(agencies.values()))['agency_timezone'],'bounds':[[-10.4,51.35],[-5.35,55.0]],'metric':'Scheduled train traversals per service day, both directions. Uses the feed timezone Europe/London.','scope':'Iarnród Éireann / Irish Rail services published by TFI, including DART, commuter and intercity rail plus Dublin–Belfast Enterprise. This is not all-island coverage: other Northern Ireland services, Luas, heritage rail, freight and replacement buses are excluded.','methodology_note':'Evaluate calendars and exceptions, normalize parent stations, and remove exact duplicate train-code/stop/time records on the same service date. Project all feed stations onto each detailed trip shape (100 m tolerance), including stations passed without a stop; trim the path to the scheduled origin and destination. Count every adjacent station traversal and direction across all selected service days, including zero days.','geometry_note':'Station-to-station corridors inferred along official shapes, including skipped express stops. All scheduled stops must match in order; otherwise the build fails. Parallel tracks share a corridor; the most frequent official shape supplies its display geometry. Station nodes do not split every junction, and nearby independent tracks may remain ambiguous; this is not a physical-track inventory.','diagnostics':{k:v for k,v in diagnostic.items() if k!='trip_paths'}}
    meta['methodology_note'] += ' Connolly alone uses a 300 m tolerance to cover separated terminal/through platforms. Route categories are coarse: DART and dedicated commuter routes are commuter; mixed intercity routes retain their intercity label.'
    meta['geometry_note'] += ' Eleven active trip records use snapshot-specific inferred corrections for four looping source shapes, substituting another official shape through the same ordered scheduled stations. These are inferred corrections, not independently confirmed diversions; see IRELAND.md and diagnostics.json. Unknown repeated-station patterns stop the build for review.'
    meta['downloaded_at'] = provenance.get('downloaded_at')
    for name,data in [('segments.geojson',{'type':'FeatureCollection','features':features}),('stations.geojson',{'type':'FeatureCollection','features':stations}),('metadata.json',meta)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('diagnostics',)},ensure_ascii=True,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feed',type=Path,default=ROOT/'data/raw/ireland.zip')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--start',type=dt.date.fromisoformat,default=dt.date(2026,9,14))
    parser.add_argument('--days',type=int,default=7)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/data/ie')
    args = parser.parse_args()
    if not 1<=args.days<=366:
        parser.error('--days must be 1..366')
    if args.download:
        args.feed.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(URL,timeout=180) as response:
            args.feed.write_bytes(response.read())
            args.feed.with_suffix('.http.json').write_text(json.dumps({'sha256':hashlib.sha256(args.feed.read_bytes()).hexdigest(),'last_modified':response.headers.get('Last-Modified'),'downloaded_at':dt.datetime.now(dt.timezone.utc).isoformat()}),encoding='utf-8')
    build(args.feed,args.start,args.days,args.output)
