import json, glob, os
S = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/f9a7f88d-0531-4963-9df6-6771326e513f/scratchpad/t1750/out'
res = {}
for f in sorted(glob.glob(S + '/*.json')):
    name = os.path.basename(f)[:-5]
    d = json.load(open(f, encoding='utf-8'))
    rail = d['day']['commute'][1] + d['day']['everyday'][1]
    ops = sum(d['operators'].values())
    boards = sum(d['lines']['board'])
    res[name] = (rail, ops, boards)
base = {'tyo': res.get('tyo_G'), 'nyc': res.get('nyc_G')}
out = []
for name, (rail, ops, boards) in res.items():
    b = base[name[:3]]
    out.append(f"{name:16s} rail legs {rail/1e6:6.2f}M ({rail/b[0]:.3f} of G)  by-operator trips {ops/1e6:6.2f}M ({ops/b[1]:.3f})  line boardings {boards/1e6:6.2f}M ({boards/b[2]:.3f})")
print('\n'.join(out))
