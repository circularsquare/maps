"""T-700: walks along one bank of a zone astride water go straight.

    ..\\venv\\Scripts\\python.exe trainworld/checks/t700_walk_banks.py [city ...] [--packs DIR]
    (from the maps root; DIR: another folder of packs, such as a backup)

Reads each pack's `walk_zone` and `walk_part` (pipeline/routes.py's walk stage) the way the sim's
`routes::Routes` does, and checks:
- every pack: the people and jobs in a part of a zone whose walk factor to itself is over 5 (a
  blend of two banks; before T-700 Rotterdam had 4.2%, Shanghai 3.5%) are under 0.5%;
- Shanghai (`sha`): from Middle Henan Rd to the Bund's Puxi side the factor is under 1.2 (it was
  48), and from the Bund to Lujiazui, across the Huangpu, over 5.
With no cities, every pack with routes. One PASS or FAIL line per pack.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from read_pack import PACKS, load_pack  # noqa: E402

R = 6371008.8


class Walks:
    """A pack's walk factors by cell, decoded as `Routes::set_walk_zone`."""

    def __init__(self, h, a):
        self.zone = a["zone"].astype(np.int64)
        n = len(a["zone_gx"])
        self.gx, self.gy = a["zone_gx"].astype(int), a["zone_gy"].astype(int)
        reg = np.zeros(n, int)
        if "region" in a:
            reg[self.zone] = a["region"]
        nreg = reg.max() + 1
        rb = h["routes"]
        self.win = rb["walk_zone"]["window"]
        gw, gh = h["gravity"]["grid"]
        parts = rb["walk_zone"].get("parts") is True and "walk_part" in a
        self.part = a["walk_part"].astype(int) if parts else np.zeros(len(self.zone), int)
        np_ = np.ones(n, int)
        np.maximum.at(np_, self.zone, self.part + 1)
        at = {}
        for i in range(n):
            at[(self.gx[i], self.gy[i], reg[i])] = i
        b = a["walk_zone"]
        self.f = {}
        k = 0
        for i in range(n):
            for pi in range(np_[i]):
                for dx in range(-self.win, self.win + 1):
                    if not 0 <= self.gx[i] + dx < gw:
                        continue
                    for dy in range(-self.win, self.win + 1):
                        if not 0 <= self.gy[i] + dy < gh:
                            continue
                        for r in range(nreg):
                            j = at.get((self.gx[i] + dx, self.gy[i] + dy, r))
                            if j is None:
                                continue
                            for pj in range(np_[j]):
                                self.f[(i, pi, j, pj)] = math.exp(b[k] / 64.0)
                                k += 1
        assert k == len(b), f"walk_zone has {len(b)} bytes for {k} entries"

    def cells(self, c, d):
        i, j = self.zone[c], self.zone[d]
        if abs(self.gx[j] - self.gx[i]) > self.win or abs(self.gy[j] - self.gy[i]) > self.win:
            return 1.0
        return self.f[(i, self.part[c], j, self.part[d])]


def check(city, packs=PACKS):
    h, a = load_pack(city, packs)
    if "walk_zone" not in a or "routes" not in h:
        return None
    w = Walks(h, a)
    ends = a["pop"].astype(np.float64) + a["jobs"]
    bad = sum(ends[(w.zone == i) & (w.part == p)].sum() for (i, p, j, q), f in w.f.items() if i == j and p == q and f > 5)
    share = bad / ends.sum()
    fails = []
    if share > 0.005:
        fails.append(f"{share:.1%} of people and jobs in a part walking to itself at over x5")
    note = f"{share:.2%} in parts over x5"
    if city == "sha":
        lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
        k = math.cos(math.radians(lat0))

        def cell(lon, lat):
            # the nearest cell people live or work in (cells centred in the river hold almost
            # nobody, and either bank may claim them)
            x, y = R * math.radians(lon - lon0) * k, R * math.radians(lat - lat0)
            d2 = (a["x_m"] - x) ** 2 + (a["y_m"] - y) ** 2
            return int(np.argmin(np.where(ends > 500, d2, np.inf)))

        henan = cell(121.4824, 31.2363)
        bund = [cell(121.4857, 31.2360), cell(121.4878, 31.2345), cell(121.4900, 31.2350)]
        lujiazui = cell(121.5020, 31.2390)
        same = max(w.cells(henan, b) for b in bund)
        across = w.cells(bund[1], lujiazui)
        if same >= 1.2:
            fails.append(f"Middle Henan Rd to the Bund x{same:.2f} (one bank: under 1.2)")
        if across <= 5:
            fails.append(f"the Bund to Lujiazui x{across:.2f} (across the Huangpu: over 5)")
        note += f"; Middle Henan Rd to the Bund x{same:.2f}, the Bund to Lujiazui x{across:.2f}"
    return fails, note


def main(argv):
    packs = PACKS
    if "--packs" in argv:
        packs = argv[argv.index("--packs") + 1]
        argv = [v for v in argv if v not in ("--packs", packs)]
    cities = argv or sorted(f[:-5] for f in os.listdir(packs) if f.endswith(".json") and len(f) == 8)
    failed = 0
    for c in cities:
        r = check(c, packs)
        if r is None:
            continue
        fails, note = r
        failed += bool(fails)
        print(f"{'FAIL' if fails else 'PASS'} {c}: {'; '.join(fails) if fails else note}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
