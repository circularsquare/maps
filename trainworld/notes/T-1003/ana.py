"""T-1003: tables from the harness outputs.

    python ana.py base            the baseline table
    python ana.py levers s1,s2..  walk share under each setting
    python ana.py swap            distance-mix swap against New York and London
"""
import json, os, sys
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
OUT = f"{D}/out"
CN = "sha,bei,gua,shz,cho,chg,wuh,xia,tia,hag,zib,tay,gui,nag,tag,nat,jii,tik,huz,zhb,fug,hoe,sio,zii".split(",")
OTHER = "nyc,lon,par,tyo,mun,lyo,lee,mil,seo,mum,hon,tai".split(",")
NB, NG = 6, 6
reg = {x['id']: x for x in json.load(open('C:/Users/anita/projects/maps/trainworld/app/src/game/registry.json', encoding='utf-8'))['cities']}


def load(c, v):
    p = f"{OUT}/{c}_{v}.json"
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    a = np.array(j['acc']).reshape(2, NB, NG, -1)
    return j, a


def tot(a, layer=None, band=None, g=5):
    x = a if layer is None else a[layer:layer + 1]
    x = x.sum(0)  # bands, groups, vars
    x = x[:, g] if band is None else x[band:band + 1, g]
    return x.sum(0)


def base():
    print("| Zone | Walk, all | Commutes | Everyday | Everyday of trips | Under 1 / 2 / 3 km | Mean km | Walk under 2 km | Home park | Paid park | Drive min/km | A | B | T rail / walk |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for c in CN + OTHER:
        r = load(c, 'base')
        if r is None:
            continue
        j, a = r
        t = tot(a)
        tc, te = tot(a, 0), tot(a, 1)
        bands = a.sum(0)[:, 5, :]
        u = np.cumsum(bands[:, 0]) / t[0]
        w2 = bands[:2, 1].sum() / bands[:2, 0].sum()
        A = t[9] / (t[9] + t[11]); B = t[12] / (t[12] + t[14]); T = t[15] / (t[15] + t[17])
        print(f"| {reg[c]['name']} ({c}) | {100*t[1]/t[0]:.0f} | {100*tc[1]/tc[0]:.0f} | {100*te[1]/te[0]:.0f} | {100*te[0]/t[0]:.0f} | {100*u[0]:.0f} / {100*u[1]:.0f} / {100*u[2]:.0f} | {t[2]/t[0]:.1f} | {100*w2:.0f} | {t[8]/t[0]:.1f} | {t[7]/t[0]:.1f} | {t[4]/t[2]:.2f} | {100*A:.0f} | {100*B:.0f} | {100*T:.0f} / {100*t[16]/t[0]:.0f} |")


def levers(names):
    print("| Zone | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for c in CN + OTHER:
        cells = []
        for v in names:
            r = load(c, v)
            if r is None:
                cells.append("")
                continue
            t = tot(r[1])
            A = t[9] / (t[9] + t[11]); B = t[12] / (t[12] + t[14])
            cells.append(f"{100*t[1]/t[0]:.0f} ({100*A:.0f}/{100*B:.0f})")
        print(f"| {reg[c]['name']} | " + " | ".join(cells) + " |")


def swap():
    """walk share if the zone kept its own walking by band but had another city's mix of distances"""
    ref = {k: load(k, 'base')[1].sum(0)[:, 5, :] for k in ('nyc', 'lon')}
    print("| Zone | Own | NY mix | London mix | Walk by band <1/1-2/2-3/3-5 km |")
    print("|---|---|---|---|---|")
    for c in CN + OTHER:
        r = load(c, 'base')
        if r is None:
            continue
        b = r[1].sum(0)[:, 5, :]
        wb = b[:, 1] / np.maximum(b[:, 0], 1e-9)
        own = b[:, 1].sum() / b[:, 0].sum()
        s = {k: (wb * v[:, 0]).sum() / v[:, 0].sum() for k, v in ref.items()}
        print(f"| {reg[c]['name']} | {100*own:.0f} | {100*s['nyc']:.0f} | {100*s['lon']:.0f} | {' / '.join(f'{100*x:.0f}' for x in wb[:4])} |")


def groups():
    print("| Zone | <3k | 3-10k | 10-20k | 20-40k | >=40k (walk %, share of trips %) |")
    print("|---|---|---|---|---|---|")
    for c in CN + OTHER:
        r = load(c, 'base')
        if r is None:
            continue
        g = r[1].sum(0).sum(0)  # groups, vars
        cells = [f"{100*g[i,1]/max(g[i,0],1e-9):.0f} ({100*g[i,0]/g[5,0]:.0f})" for i in range(5)]
        print(f"| {reg[c]['name']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    w = sys.argv[1]
    if w == 'base':
        base()
    elif w == 'levers':
        levers(sys.argv[2].split(','))
    elif w == 'swap':
        swap()
    elif w == 'groups':
        groups()
