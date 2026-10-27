import json
A = 'C:/Users/anita/projects/maps/artifacts/trainworld/'
HOURS = [14, 4, 6]
PH = [4, 6, 4]
def rows(path):
    j = json.load(open(path))
    b0 = j['base']
    w0 = b0['split']['walk'] / 100 * b0['trips']; d0 = b0['split']['drive'] / 100 * b0['trips']; r0 = b0['rail']
    out = {}
    for s in j['scen']:
        sn = s['snap']; l = sn.get('line')
        if not l: continue
        road = s['add']['km']; tn = s['add'].get('trainsNeeded') or l['trainsNeeded']
        bh = sum(n * h for n, h in zip(tn, HOURS))
        peak = max(l['maxLoadPerPeriod'][p] / PH[p] * sn['peakHourFactor'][p] for p in range(3))
        cap = 60 / (s['hw'] / 60) * 80
        out[s['id']] = dict(road=road, hw=s['hw'] / 60, riders=l['riders'], perkm=l['riders'] / road, perbh=l['riders'] / bh, buses=tn, bh=bh,
                            load=peak / cap, dr=sn['rail'] - r0, dd=sn['split']['drive'] / 100 * sn['trips'] - d0, base=r0)
    return out
for w in ['real', 'new']:
    a = rows(A + 't1576/%s.json' % w); b = rows(A + 't1577/%s.json' % w)
    print(w, 'base rail 1576', next(iter(a.values()))['base'], '1577', next(iter(b.values()))['base'])
    for k in a:
        x = a[k]; y = b.get(k)
        if not y: print(k, 'missing'); continue
        print('| %s | %.1f | %g | %d | %d (%+.0f%%) | %d | %d | %d | %d | %s>%s | %.2f | %.2f | %+d | %+d |' % (k, y['road'], y['hw'] * 60 / 60, x['riders'], y['riders'], 100 * (y['riders'] / x['riders'] - 1), x['perkm'], y['perkm'], x['perbh'], y['perbh'], x['buses'], y['buses'], x['load'], y['load'], y['dr'], y['dd']))
