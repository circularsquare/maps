import json, sys, collections
city = sys.argv[1]
d = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{city}_dump.json', encoding='utf-8'))
for lv, caps in enumerate(d['capacity']):
    print('level', lv, 'markers', len(caps))
    by = collections.Counter()
    held = collections.Counter()
    for c in caps:
        by[c['kind']] += 1
        if c['held'] or c['rho'] >= 0.9999: held[c['kind']] += 1
    print(' by kind', dict(by), 'held/full', dict(held))
    if lv == 0:
        full = sorted([c for c in caps if c['held'] or c['rho'] >= 0.99], key=lambda c: -c['delayS'])
        for c in full[:40]:
            print('  ', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()})
        print(' fullTrack edges', len(d['fullTrack'][lv]))
