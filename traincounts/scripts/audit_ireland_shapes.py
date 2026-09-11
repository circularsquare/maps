"""Reproduce evidence for the explicitly reviewed Irish Rail shape corrections."""
import collections
import csv
import io
import json
from pathlib import Path
import zipfile
from build_ireland import project_stations, match_path, distance, STATION_TOLERANCES, REVIEWED_SHAPE_CORRECTIONS

ROOT=Path(__file__).resolve().parents[1]


def audit():
    z=zipfile.ZipFile(ROOT/'data/raw/ireland.zip')
    def rows(n):
        return list(csv.DictReader(io.TextIOWrapper(z.open(n),encoding='utf-8-sig')))
    stops={r['stop_id']:r for r in rows('stops.txt')}
    coords={s:[float(r['stop_lon']),float(r['stop_lat'])] for s,r in stops.items()}
    shapes=collections.defaultdict(list)
    for r in rows('shapes.txt'):
        shapes[r['shape_id']].append((int(r['shape_pt_sequence']),[float(r['shape_pt_lon']),float(r['shape_pt_lat'])]))
    shapes={s:[p for _,p in sorted(v)] for s,v in shapes.items()}
    trips=rows('trips.txt')
    times=collections.defaultdict(list)
    for r in rows('stop_times.txt'):
        times[r['trip_id']].append(r)
    for v in times.values():
        v.sort(key=lambda r:int(r['stop_sequence']))
    diagnostics=json.loads((ROOT/'dist/data/ie/diagnostics.json').read_text(encoding='utf-8'))
    evidence=[]
    for original,replacement in REVIEWED_SHAPE_CORRECTIONS.items():
        trip=next(t for t in trips if t['shape_id']==original and t['trip_id'] in diagnostics['shape_substitutions'])
        scheduled=[r['stop_id'] for r in times[trip['trip_id']]]
        item={'original_shape':original,'replacement_shape':replacement,'example_trip':trip,'scheduled_stops':[(stops[r['stop_id']]['stop_name'],r['arrival_time'],r['departure_time']) for r in times[trip['trip_id']]],'replacement_route_ids':sorted({t['route_id'] for t in trips if t['shape_id']==replacement})}
        for key,sid in [('original',original),('replacement',replacement)]:
            path=match_path(project_stations(shapes[sid],coords,station_tolerances=STATION_TOLERANCES),scheduled)
            points=shapes[sid][int(path[0][0]):int(path[-1][0])+2]
            item[key+'_station_path']=[stops[s]['stop_name'] for _,s,_ in path]
            item[key+'_length_km']=round(sum(distance(a,b) for a,b in zip(points,points[1:]))/1000,3)
        evidence.append(item)
    (ROOT/'dist/data/ie/shape_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=True,indent=2))


if __name__=='__main__':
    audit()
