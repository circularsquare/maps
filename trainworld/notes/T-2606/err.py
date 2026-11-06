"""T-2606: what separate west and east clusters would lose on each line: the market's boardings
(free flow, every period) of the whole network against those of the two parts solved alone
(T-1650 subnet.py's clipped nets, whose kept lines are worked out the same way here).

python err.py <net.json> <small dump.json> <whole cuts.json> <part dump, part places file>... <out.json>
"""
import json, sys
import numpy as np

net = json.load(open(sys.argv[1]))
small = json.load(open(sys.argv[2]))
whole = json.load(open(sys.argv[3]))['line_board']
args = sys.argv[4:-1]
out = sys.argv[-1]
L = len(net['lineN'])
W = sum(np.array(whole[q]) for q in whole)


def kept_lines(places):
    merged = set()
    for z, acc in enumerate(small['access']):
        if small['zones'][z][0] in places:
            merged.update(acc)
    st_map = small['st_map']
    n_st = len(net['stXY']) // 2
    inreach = [st_map[i] in merged if st_map else i in merged for i in range(n_st)]
    keep, off = [], 0
    for l, k in enumerate(net['lineN']):
        stops = net['stops'][off:off + k]
        idx = [i for i, s in enumerate(stops) if inreach[s]]
        if idx and min(k - 1, idx[-1] + 1) > max(0, idx[0] - 1):
            keep.append(l)
        off += k
    return keep


parts = np.zeros(L)
for dump_path, places_path in zip(args[0::2], args[1::2]):
    places = set(open(places_path).read().split(','))
    keep = kept_lines(places)
    d = json.load(open(dump_path))
    rn = np.array(d['rn_line'])
    for per in d['periods']:
        b = np.bincount(rn, weights=np.array(per['board']), minlength=len(keep))
        for i, l in enumerate(keep):
            parts[l] += b[i]
lost = W - parts
order = np.argsort(-lost)
rows = []
print(f'market boardings over the day (free flow): whole {W.sum():.0f}, parts {parts.sum():.0f}, lost {lost.sum():.0f} ({100 * lost.sum() / W.sum():.1f}%)')
print('lines losing most: line, whole, parts, lost, lost share of its market boardings')
for l in order[:15]:
    rows.append([int(l), W[l], parts[l], lost[l]])
    print(f'  {l:4d} {W[l]:9.0f} {parts[l]:9.0f} {lost[l]:9.0f} {100 * lost[l] / max(W[l], 1):5.1f}%')
json.dump({'whole': W.tolist(), 'parts': parts.tolist(), 'top': rows}, open(out, 'w'))
