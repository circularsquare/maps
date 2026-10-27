import json, sys
SP = sys.argv[1]
for net in sys.argv[2:]:
    b = json.load(open(f'{SP}/{net}.before.json'))
    a = json.load(open(f'{SP}/{net}.after.json'))
    print(f"{net}: solve {b['solve_ms']:.0f} -> {a['solve_ms']:.0f} ms; stations {a['stations']}, pairs {a['pairs']}, commute pairs {a['cpairs']}, everyday {a['epairs']}")
    for k in ['rail', 'c_rail', 'e_rail', 'pax_min']:
        x, y = b[k], a[k]
        print(f'  {k:8} {x:16.4f} {y:16.4f}  rel {((y - x) / x if x else 0):+.2e}')
    for k in ['board', 'seg', 'entries', 'exits', 'c_seg', 'c_board', 'e_seg', 'r_seg', 'corridors', 'pair_rail', 'c_frac', 'st', 'xfer']:
        x, y = b.get(k), a.get(k)
        if x is None or y is None:
            print(f'  {k:9} missing before {x is None} after {y is None}')
            continue
        if len(x) != len(y):
            print(f'  {k:9} lengths differ {len(x)} {len(y)}')
            continue
        top = max((abs(v) for v in x), default=0)
        diffs = [abs(p - q) for p, q in zip(x, y)]
        same = sum(1 for d in diffs if d == 0)
        worst = max((d / max(abs(p), abs(q), 0.01 * top, 1e-12) for d, p, q in zip(diffs, x, y)), default=0)
        tot = sum(abs(v) for v in x)
        moved = sum(diffs) / tot if tot else 0
        print(f'  {k:9} n {len(x):9} identical {same / max(len(x), 1):6.1%}  moved {moved:.2e}  worst {worst:.2e} (of max(v, 1% of top {top:.1f}))')
