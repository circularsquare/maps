"""Checks on data/sub for one epoch, plus the dense-country look (results go in NOTES.md).

    python -I check_sub.py 2020
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "data", "sub")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stats(h, lo, step):
    """people, people-weighted median density, and occupied km2 (1 km grid: sum of people/density)."""
    v, o = h["v"], h["o"]
    tot = sum(v)
    acc, med = 0, None
    km2 = 0.0
    for j, x in enumerate(v):
        d = 10 ** (lo + (o + j + 0.5) * step)
        km2 += x / d
        acc += x
        if med is None and acc >= tot / 2:
            med = d
    return tot, med, km2


def share_above(h, lo, step, dmin):
    v, o = h["v"], h["o"]
    return sum(x for j, x in enumerate(v) if 10 ** (lo + (o + j) * step) >= dmin) / max(sum(v), 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    y = sys.argv[1]
    main_ = load(os.path.join(HERE, "data", "hist.json"))
    lo, step = main_["bins"]["lo"], main_["bins"]["step"]
    idx = load(os.path.join(SUB, "index.json"))["units"]
    allp = load(os.path.join(HERE, "logs", "sub_pops.json"))
    print(f"index: {len(idx)} units ({sum(u['kind'] == 'metro' for u in idx)} metros, "
          f"{sum(u['kind'] == 'state' for u in idx)} states), "
          f"{os.path.getsize(os.path.join(SUB, 'index.json')) / 1e6:.2f} MB")
    files = [f for f in os.listdir(SUB) if f != "index.json" and f.endswith(".json")]
    size = sum(os.path.getsize(os.path.join(SUB, f)) for f in files)
    big = sorted(files, key=lambda f: -os.path.getsize(os.path.join(SUB, f)))[:5]
    print(f"{len(files)} country files, {size / 1e6:.1f} MB in all; biggest "
          + ", ".join(f"{f} {os.path.getsize(os.path.join(SUB, f)) / 1e6:.1f} MB" for f in big))

    # states add up to their country exactly
    bad = []
    for k, c in main_["countries"].items():
        if y not in c["pop"]:
            continue
        s = sum(p.get(y, 0) for uid, p in allp.items() if uid.startswith("adm1:")
                and uid in units_of(k))
        if s != c["pop"][y]:
            bad.append((k, s, c["pop"][y]))
    print(f"countries whose states do not add up: {len(bad)} {bad[:10]}")

    byname = {}
    for u in idx:
        if u["kind"] == "metro":
            byname.setdefault((u["parent"], u["name"]), u)
    print("\nmetros:")
    for par, nm in [("USA", "New York"), ("USA", "Houston"), ("BGD", "Dhaka"), ("USA", "San Diego"),
                    ("MEX", "Tijuana"), ("USA", "El Paso"), ("MEX", "Ciudad Juárez"),
                    ("USA", "Detroit"), ("CAN", "Windsor"), ("SGP", "Singapore"),
                    ("MYS", "Johor Bahru"), ("COG", "Brazzaville"), ("COG", "Pointe-Noire"),
                    ("SOM", "Mogadishu"), ("SOM", "Hargeisa"), ("SOM", "Baidoa"), ("SOM", "Burao"),
                    ("SOM", "Kismayo"), ("SOM", "Bosaso"), ("DJI", "Djibouti"), ("YEM", "Sana'a"),
                    ("YEM", "Aden"), ("YEM", "Taiz")]:
        u = byname.get((par, nm))
        if not u:
            near = [v["name"] for v in idx if v["parent"] == par and v["kind"] == "metro"][:12]
            print(f"  {par} {nm}: not in index; metros there: {near}")
            continue
        h = load(os.path.join(SUB, f"{par}.json"))["hist"]["1km"][y][u["id"]]
        tot, med, km2 = stats(h, lo, step)
        print(f"  {par} {nm} ({u['id']}): {u['pop'][y]:,}  1km median {med:,.0f}/km2, "
              f"occupied {km2:,.0f} km2, mean {tot / km2:,.0f}/km2")

    print("\ndense countries (1 km):")
    for k in ("COG", "SOM", "DJI", "YEM", "COD", "NGA", "BGD", "USA"):
        h = main_["hist"]["1km"][y][k]
        tot, med, km2 = stats(h, lo, step)
        sub = load(os.path.join(SUB, f"{k}.json"))["hist"]["1km"][y]
        fpop = sum(sum(v["v"]) for uid, v in sub.items() if uid.startswith("fua:"))
        print(f"  {k}: {tot:,.0f} people, median {med:,.0f}/km2, occupied {km2:,.0f} km2, "
              f"share >=10k {share_above(h, lo, step, 1e4) * 100:.0f}%, "
              f">=20k {share_above(h, lo, step, 2e4) * 100:.0f}%, in metros {fpop / tot * 100:.0f}%")
        rest = None
        # outside metros: country minus metro histograms
        vals = {}
        for j, x in enumerate(h["v"]):
            vals[h["o"] + j] = x
        for uid, v in sub.items():
            if uid.startswith("fua:"):
                for j, x in enumerate(v["v"]):
                    vals[v["o"] + j] = vals.get(v["o"] + j, 0) - x
        keys_ = sorted(i for i in vals if vals[i] > 0)
        if keys_:
            rest = {"o": keys_[0], "v": [max(vals.get(i, 0), 0) for i in range(keys_[0], keys_[-1] + 1)]}
            t2, m2, a2 = stats(rest, lo, step)
            print(f"      outside metros: {t2:,.0f} people, median {m2:,.0f}/km2, "
                  f"occupied {a2:,.0f} km2, share >=10k {share_above(rest, lo, step, 1e4) * 100:.0f}%")


_UNITS = {}


def units_of(k):
    if k not in _UNITS:
        p = os.path.join(SUB, f"{k}.json")
        if os.path.exists(p):
            d = load(p)["hist"]["100m"]
            _UNITS[k] = {uid for yy in d.values() for uid in yy}
        else:
            _UNITS[k] = set()
    return _UNITS[k]


if __name__ == "__main__":
    main()
