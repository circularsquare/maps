import json, copy
S = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/f9a7f88d-0531-4963-9df6-6771326e513f/scratchpad/t1750/var'
for city, base in (('tyo', 'tyo_X'), ('nyc', 'nyc_G')):
    x = json.load(open(f'{S}/{base}.json'))
    for tag, tf, pf in (('t085', 0.85, 1.0), ('t115', 1.15, 1.0), ('p125', 1.0, 1.25)):
        w = copy.deepcopy(x)
        d = w['demand']
        d['times'] = [t * tf for t in d['times']]
        if pf != 1.0:
            # high level (index 0 of each line's 3) up by pf, at most 30 an hour
            for i in range(0, len(d['tph']), 3):
                d['tph'][i] = min(30.0, d['tph'][i] * pf) if d['tph'][i] < 30 else d['tph'][i]
        json.dump(w, open(f'{S}/{base}_{tag}.json', 'w'))
print('ok')
