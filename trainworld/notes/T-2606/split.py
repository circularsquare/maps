"""T-2606: the market places west and east of a longitude (a place goes where most of its zones'
rail-reached stations are), and a small copy of the dump (zones, boxes, access, st_map) that
T-1650's subnet.py reads.

python split.py <dump.json> <cut lon> <small dump out> <west places out> <east places out>
"""
import json, sys, collections

d = json.load(open(sys.argv[1]))
cut = float(sys.argv[2])
small = {k: d[k] for k in ('zones', 'boxes', 'access', 'st_map', 'rn_line')}
json.dump(small, open(sys.argv[3], 'w'))
votes = collections.defaultdict(lambda: [0, 0])
for z, (place, *_rest) in enumerate(d['zones']):
    lon = (d['boxes'][5 * z + 1] + d['boxes'][5 * z + 3]) / 2
    if d['access'][z]:
        votes[place][0 if lon < cut else 1] += len(d['access'][z])
west = sorted(p for p, v in votes.items() if v[0] >= v[1])
east = sorted(p for p, v in votes.items() if v[0] < v[1])
open(sys.argv[4], 'w').write(','.join(map(str, west)))
open(sys.argv[5], 'w').write(','.join(map(str, east)))
print(len(west), 'west places', len(east), 'east places')
