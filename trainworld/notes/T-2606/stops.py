"""T-2606: one line's stops as lon, lat. python stops.py <net.json> <line>"""
import json, math, sys

net = json.load(open(sys.argv[1]))
l = int(sys.argv[2])
off = sum(net['lineN'][:l])
R = 6371008.8
xy = net['stXY']
for s in net['stops'][off:off + net['lineN'][l]]:
    print(round(math.degrees(xy[2 * s] / R), 2), round(math.degrees(math.atan(math.sinh(xy[2 * s + 1] / R))), 2))
