"""T-2621: how much of a strategy search is stops on one line alone (a chain a search could step
along without its queue): stations with one line, no complex and no other station within 300 m,
not a line's end; their route nodes and the station itself.

python chains.py <plain net.json>"""
import json, math, sys
from collections import defaultdict

n = json.load(open(sys.argv[1]))
xy = n['stXY']
S = len(xy) // 2
lines_at = defaultdict(set)
ends = set()
rn = 0
off = 0
for l, k in enumerate(n['lineN']):
    st = n['stops'][off:off + k]
    off += k
    rn += 2 * k
    for s in st:
        lines_at[s].add(l)
    if st:
        ends.add(st[0]); ends.add(st[-1])
cx = n['complex']
in_complex = defaultdict(int)
for c in cx:
    in_complex[c] += 1
# stations near another (a walk): a grid of 300 m cells
cell = defaultdict(list)
for s in range(S):
    cell[(int(xy[2 * s] // 300), int(xy[2 * s + 1] // 300))].append(s)
near = set()
for (i, j), ss in cell.items():
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for t in cell.get((i + di, j + dj), []):
                for s in ss:
                    if s != t and math.hypot(xy[2 * s] - xy[2 * t], xy[2 * s + 1] - xy[2 * t + 1]) < 300:
                        near.add(s)
served = [s for s in range(S) if lines_at[s]]
chain = [s for s in served if len(lines_at[s]) == 1 and s not in ends and s not in near and in_complex.get(cx[s], 1) <= 1]
print(f'{S} stations ({len(served)} on a line), {rn} route nodes; one-line stops off any walk or complex: {len(chain)} '
      f'({100*len(chain)/len(served):.0f}% of stations on a line), their route nodes {2*len(chain)} ({100*2*len(chain)/rn:.0f}%); '
      f'of the nodes a search pops (stations + route nodes): {100*3*len(chain)/(len(served)+rn):.0f}%')
