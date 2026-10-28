"""T-1650: a plain net with one line's trains an hour times `f` at every level.
python edit.py <plain net> <line> <f> <out.json>"""
import json, sys
n = json.load(open(sys.argv[1]))
l, f = int(sys.argv[2]), float(sys.argv[3])
for k in range(3):
    n['tph'][3 * l + k] *= f
json.dump(n, open(sys.argv[4], 'w'))
print('line', l, 'tph', n['tph'][3 * l:3 * l + 3], 'stops', n['lineN'][l])
