# T-782 scratch: compare two t782-measure JSONs: rail, lines' riders, stations' boardings
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
floor = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.
print('final s', a['finalSecs'], b['finalSecs'], 'rail', round(a['rail'], 1), round(b['rail'], 1), f"{100 * (b['rail'] - a['rail']) / a['rail']:+.3f}%")
print('market', a.get('market'), b.get('market'))
if 'lineFig' not in a or 'lineFig' not in b:
    sys.exit()
def rep(name, da, db, ix):
    rows = []
    for k in da:
        if k not in db: continue
        x, y = da[k][ix], db[k][ix]
        rows.append((k, x, y))
    tot_a = sum(r[1] for r in rows); moved = sum(abs(r[2] - r[1]) for r in rows)
    big = [r for r in rows if max(r[1], r[2]) >= floor]
    worst = sorted(big, key=lambda r: -abs(r[2] - r[1]) / max(r[1], r[2]))[:6]
    over2 = sum(1 for r in big if abs(r[2] - r[1]) / max(r[1], r[2]) > 0.02)
    print(f'{name}: n {len(rows)}, moved {100 * moved / tot_a:.2f}% of {tot_a:.0f}; over {floor:.0f}: {len(big)}, past 2%: {over2}')
    for k, x, y in worst:
        print(f'   {k}: {x:.0f} -> {y:.0f} ({100 * (y - x) / max(x, 1e-9):+.2f}%)')
rep('lines riders', a['lineFig'], b['lineFig'], 0)
rep('stations boardings', a['stationFig'], b['stationFig'], 0)
