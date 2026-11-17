"""T-3388: compare per-line settled figures of two probe runs (A: rule before, B: after).

    python cmp.py <dir A> <dir B>

For every <city>-<line>-<run>.<step>.txt in both: per line and level, trains (must match), trains
an hour and round trip. The envelope for a line-level is the range of trains an hour A's own pace
rounds visited in that step (A's answer is whichever phase its third round lands on), read from
A's <pfx>.ph.c<call>.<round>.txt; the step's call is found from the log (each settle prints
[settle0). A step with no pace rounds in A has no envelope: B must match A there.
"""
import glob
import os
import re
import sys

NUM = re.compile(r'-?\d+(?:\.\d+)?(?:e-?\d+)?')


def dump(path):
    out = {}
    for ln in open(path):
        parts = ln.rstrip('\n').split('\t')
        if len(parts) < 5:
            continue
        l = int(parts[0])
        tph, rt, tr = [[float(x) for x in NUM.findall(p)] for p in parts[1:4]]
        out[l] = (tph, rt, tr)
    return out


def ph(path):
    out = {}
    if not os.path.exists(path):
        return None
    for ln in open(path):
        parts = ln.rstrip('\n').split('\t')
        l = int(parts[0])
        tph = [float(x) for x in NUM.findall(parts[1])]
        rt = [float(x) for x in NUM.findall(parts[2])] if 'Some' in parts[2] else None
        out[l] = (tph, rt)
    return out


def step_calls(log):
    """step name -> index of the last settle call made in it"""
    calls, cur, res = -1, None, {}
    for ln in open(log):
        s = ln.strip()
        m = re.match(r'^(\w+):$', s)
        if m:
            cur = m.group(1)
            continue
        if s.startswith('[settle0'):
            calls += 1
            if cur:
                res[cur] = calls
    return res


def main(a, b):
    tot = dict(n=0, out=0, trains=0, d05=0, d1=0)
    worst = []
    for pa in sorted(glob.glob(os.path.join(a, '*.txt'))):
        name = os.path.basename(pa)
        if '.ph.' in name or name.endswith('.load.txt'):
            continue
        pb = os.path.join(b, name)
        if not os.path.exists(pb):
            continue
        pfx, step = name[:-4].rsplit('.', 1)
        calls = step_calls(os.path.join(a, pfx + '.log'))
        A, B = dump(pa), dump(pb)
        env = {}
        c = calls.get(step)
        if c is not None:
            # the range of trains an hour A's pace rounds visited in this step (A's answer is
            # whichever its last round lands on)
            rs = [r for r in (ph(os.path.join(a, f'{pfx}.ph.c{c}.{k}.txt')) for k in (1, 2, 3)) if r]
            if rs:
                for l in rs[0]:
                    vals = [r[l][0] for r in rs] + [A[l][0]] if l in A else [r[l][0] for r in rs]
                    env[l] = [(min(v[lev] for v in vals), max(v[lev] for v in vals)) for lev in range(3)]
        n = out = d05 = d1 = 0
        sa = [0.0] * 3
        sb = [0.0] * 3
        for l, (ta, ra, ca) in A.items():
            tb, rb, cb = B[l]
            if ca != cb:
                tot['trains'] += 1
            for lev in range(3):
                sa[lev] += ta[lev]
                sb[lev] += tb[lev]
                if ta[lev] <= 0 and tb[lev] <= 0:
                    continue
                n += 1
                d = abs(tb[lev] - ta[lev])
                rel = d / max(ta[lev], 1e-9)
                lo, hi = env.get(l, [(ta[lev], ta[lev])] * 3)[lev]
                tol = 1e-5 * max(1.0, ta[lev])
                ex = max(lo - tb[lev], tb[lev] - hi, 0.0) / max(ta[lev], 1e-9)
                if env:
                    tot.setdefault('ex', []).append((ex, name, l, lev))
                if tb[lev] < lo - tol or tb[lev] > hi + tol:
                    out += 1
                    worst.append((rel, name, l, lev, ta[lev], tb[lev], f'{lo:.4f}-{hi:.4f}'))
                if rel > 0.005:
                    d05 += 1
                if rel > 0.01:
                    d1 += 1
        tot['n'] += n
        tot['out'] += out
        tot['d05'] += d05
        tot['d1'] += d1
        print(f'{name:<34} line-levels {n:4d}  outside A envelope {out:3d}  >0.5% {d05:3d}  >1% {d1:3d}  '
              f'sum tph A {sa[0]:.2f}/{sa[1]:.2f}/{sa[2]:.2f}  B {sb[0]:.2f}/{sb[1]:.2f}/{sb[2]:.2f}  env {"yes" if env else "no"}')
    ex = sorted(tot.pop('ex', []), reverse=True)
    print('total', tot)
    if ex:
        print(f'steps with pace rounds in A: {len(ex)} line-levels; outside the range A visited by more than '
              f'0.1%: {sum(e[0] > 0.001 for e in ex)}, 0.5%: {sum(e[0] > 0.005 for e in ex)}, 1%: {sum(e[0] > 0.01 for e in ex)}; '
              f'largest {ex[0][0]*100:.2f}% ({ex[0][1]} line {ex[0][2]} lev {ex[0][3]})')
        for e in ex[:8]:
            print(f'    {e[0]*100:.2f}%  {e[1]} line {e[2]} lev {e[3]}')
    worst.sort(reverse=True)
    print('largest moves outside the envelope (rel, file, line, level, A, B, envelope):')
    for w in worst[:25]:
        print(f'  {w[0]*100:6.2f}%  {w[1]:<32} line {w[2]:3d} lev {w[3]}  A {w[4]:.4f}  B {w[5]:.4f}  A visited {w[6]}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
