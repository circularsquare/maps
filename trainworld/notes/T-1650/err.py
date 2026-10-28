"""T-1650: what freezing the market hides after an edit. Market boardings by line, before and after,
whole market and (optionally) one part's pairs alone, against the line's riders a day.

python err.py <browser measure> <base dump> <edit dump> [<part base dump> <part edit dump>] [period share]
"""
import json, sys, collections

def by_line(p):
    d = json.load(open(p))
    m = collections.Counter()
    for l, x in zip(d['rn_line'], d['board']):
        m[l] += x
    return m

b = json.load(open(sys.argv[1]))['lineFig']
riders = {int(k): v[0] for k, v in b.items()}
base, edit = by_line(sys.argv[2]), by_line(sys.argv[3])
part = len(sys.argv) > 5
if part:
    pb, pe = by_line(sys.argv[4]), by_line(sys.argv[5])
share = 0.35
def report(name, delta):
    big = [l for l in riders if riders[l] >= 10000]
    errs = sorted(((abs(delta[l]) / share / riders[l], l) for l in big), reverse=True)
    n = lambda t: sum(1 for e, _ in errs if e > t)
    tot = sum(abs(delta[l]) for l in riders) / share
    print(f'{name}: market boardings moved {tot:.0f} a day; of {len(big)} lines over 10,000: past 2% {n(0.02)}, past 5% {n(0.05)}, past 10% {n(0.10)}, past 20% {n(0.2)}; worst', ', '.join(f'line {l} {100 * e:.1f}% of {riders[l]:.0f}' for e, l in errs[:4]))
d_all = {l: edit[l] - base[l] for l in set(base) | set(edit)}
report('market frozen whole (all of its change hidden)', d_all)
if part:
    d_in = {l: pe[l] - pb[l] for l in set(pb) | set(pe)}
    report('part re-solved, crossing frozen (hidden: the change less the part\'s)', {l: d_all.get(l, 0) - d_in.get(l, 0) for l in d_all})
    print(f'part pairs carry {100 * sum(abs(v) for v in d_in.values()) / max(1, sum(abs(v) for v in d_all.values())):.0f}% of the change (sum of |delta| by line)')
