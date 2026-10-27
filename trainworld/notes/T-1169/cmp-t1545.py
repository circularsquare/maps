import json, sys
S = r'C:\Users\anita\projects\maps\artifacts\trainworld\t1545' + '\\'
for k in sys.argv[1:]:
    b = json.load(open(S + f'before-{k}.json'))
    a = json.load(open(S + f'after-{k}.json'))
    print(f"{k}: final {b['finalSecs']} s -> {a['finalSecs']} s; rail a day {b['rail']:.1f} -> {a['rail']:.1f} ({(a['rail'] - b['rail']) / b['rail']:+.2e}); errors {len(b['errors'])} {len(a['errors'])}")
    for nm, d in (('before', b), ('after', a)):
        print('   ', nm, ' | '.join(f"w{m['worker']}{' aux' if m['aux'] else ''}: wasm {m['wasm']} held {m['heap']} peak {m['solvePeak']}" for m in d['memory']))
    nb, na = b['nums'], a['nums']
    if len(nb) != len(na):
        print('   numbers differ in count', len(nb), len(na))
        continue
    marks = b['sig']['marks']
    for i, (name, at) in enumerate(marks[:-1]):
        end = marks[i + 1][1]
        xs, ys = nb[at:end], na[at:end]
        same = sum(1 for x, y in zip(xs, ys) if x == y)
        top = max((abs(x) for x in xs if abs(x) < 1e299), default=0)
        worst, wi = 0, None
        for j, (x, y) in enumerate(zip(xs, ys)):
            if x != y:
                r = abs(x - y) / max(abs(x), abs(y), 0.01 * top, 1e-12)
                if r > worst:
                    worst, wi = r, j
        print(f'   {name:14} n {end - at:8} identical {same / max(end - at, 1):6.1%} worst {worst:.2e}' + (f' ({xs[wi]:.4f} -> {ys[wi]:.4f})' if wi is not None else ''))
    print('   modes same:', b.get('modes') == a.get('modes'))
