"""figs.py <dir A> <dir B>: per line and level, trains an hour, round trip, trains and pace from the
probe dumps of two runs; load dumps apart from edit dumps."""
import glob, os, re, sys

NUM = re.compile(r'-?\d+(?:\.\d+)?(?:e-?\d+)?')


def dump(path):
    out = {}
    for ln in open(path):
        parts = ln.rstrip('\n').split('\t')
        if len(parts) < 5:
            continue
        out[int(parts[0])] = [[float(x) for x in NUM.findall(p)] for p in parts[1:5]]
    return out


def main(a, b):
    groups = {}
    for pa in sorted(glob.glob(os.path.join(a, '*.txt'))):
        name = os.path.basename(pa)
        pb = os.path.join(b, name)
        if not os.path.exists(pb):
            continue
        kind = 'load' if name.endswith('.load.txt') else name[:-4].rsplit('.', 1)[1]
        g = groups.setdefault(kind, dict(files=0, n=0, same=0, rt1e4=0, rt1e3=0, rt1e2=0, tph1e3=0, tph1e2=0, trains=0, pace=0, worst_rt=(0, ''), worst_tph=(0, ''), worst_rt_abs=(0, '')))
        g['files'] += 1
        A, B = dump(pa), dump(pb)
        for l, (ta, ra, ca, pa_) in A.items():
            if l not in B:
                continue
            tb, rb, cb, pb_ = B[l]
            if ca != cb:
                g['trains'] += 1
            if pa_ != pb_:
                g['pace'] += 1
            for lev in range(min(len(ra), len(rb))):
                if ra[lev] <= 0 and rb[lev] <= 0:
                    continue
                g['n'] += 1
                d = abs(rb[lev] - ra[lev])
                rel = d / max(ra[lev], 1e-9)
                if d == 0 and ta[lev] == tb[lev]:
                    g['same'] += 1
                g['rt1e4'] += rel > 1e-4
                g['rt1e3'] += rel > 1e-3
                g['rt1e2'] += rel > 1e-2
                if rel > g['worst_rt'][0]:
                    g['worst_rt'] = (rel, f'{name} line {l} lev {lev}: {ra[lev]:.3f} vs {rb[lev]:.3f} s')
                if d > g['worst_rt_abs'][0]:
                    g['worst_rt_abs'] = (d, f'{name} line {l} lev {lev}: {ra[lev]:.3f} vs {rb[lev]:.3f} s')
                if lev < len(ta) and ta[lev] > 0:
                    rt = abs(tb[lev] - ta[lev]) / ta[lev]
                    g['tph1e3'] += rt > 1e-3
                    g['tph1e2'] += rt > 1e-2
                    if rt > g['worst_tph'][0]:
                        g['worst_tph'] = (rt, f'{name} line {l} lev {lev}: {ta[lev]:.4f} vs {tb[lev]:.4f}')
    for k, g in groups.items():
        print(f"{k}: {g['files']} dumps, {g['n']} line-levels, identical {g['same']}; round trip >0.01% {g['rt1e4']}, >0.1% {g['rt1e3']}, >1% {g['rt1e2']}; "
              f"tph >0.1% {g['tph1e3']}, >1% {g['tph1e2']}; trains differ {g['trains']} lines, pace differs {g['pace']} lines")
        print(f"   largest round trip move {g['worst_rt'][0]*100:.3f}% ({g['worst_rt'][1]}); largest in seconds {g['worst_rt_abs'][0]:.3f} ({g['worst_rt_abs'][1]})")
        print(f"   largest tph move {g['worst_tph'][0]*100:.3f}% ({g['worst_tph'][1]})")


main(sys.argv[1], sys.argv[2])
