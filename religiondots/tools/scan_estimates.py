"""Scout the national estimate layer, spec §15, before anything is drawn.

    python tools/scan_estimates.py                  one line per node, then its largest countries
    python tools/scan_estimates.py islam.shia -v    one node, every country, every flag
    python tools/scan_estimates.py --joins          the country-code joins and what failed
    python tools/scan_estimates.py --csv PATH       the whole table, for reading in a spreadsheet

Nothing here is drawn or shipped. It answers what §15 could not answer from the armchair: how
many countries would carry an outline for a religion, how many estimates §15.3 refuses because
the country's own source already measured that node, and what the world total comes to. Inputs
and their URLs are in `sources/estimates.md`.

THE CHAIN — Anita's call, 2026-09-14: Pew 2020 for totals, the World Religion Project only for
splits inside a religion.

  pew_2020  Pew, Religious Composition by Country 2010-2020. Seven families, by
            self-identification: "We rely on how people describe their own religious identity."
  wrp_2010  Correlates of War World Religion Project v1.1, national file. About thirty columns,
            latest year 2010, and NOT a self-identification source. Its own codebook says
            self-identification "does not apply to our project, since we had to rely mostly on
            secondary data", and its figures are reliability-weighted means of whatever existed.
            So it supplies only the SHARE of a Pew family that a sub-group holds, never a total.

A Shia figure is therefore `Pew 2020 Muslims x (WRP 2010 Shia / WRP 2010 Muslims)`: each link's
denominator is the node directly above it (§15.4).

PEW'S `Other_religions` HAS NO NODE, and Daoism, Sikhism, Shinto, Jainism and the rest sit inside
it. Two routes are computed for those, and they should agree:
  A  WRP column / WRP's other-type columns, times Pew's Other_religions    the chain
  B  WRP column / WRP population, times Pew population                    the check
`othrgen` is in A's denominator and is never quoted: the codebook says WRP used "other religion"
as the column that forced totals to match population.

**THEY DISAGREE BY MORE THAN A FACTOR OF TWO FOR MOST COUNTRIES** (first run, 2026-09-14: 23 of
35 Sikh estimates, 73 of 108 Baha'i, 74 of 104 indigenous), and each route fails in its own way.
A inflates where WRP's other-type columns are thin: 240k Sikhs in the Netherlands against B's
14k. B inflates wherever WRP counted practice or rolls rather than identity: 110M Shinto in Japan,
57M animists in China. So WRP is not a usable source for anything inside Pew's `Other_religions`,
and this file only shows how badly. Those religions come from hand rows (spec §15.4b).

WHAT THE WRP FILE GETS WRONG, found 2026-09-14 and checked in the raw rows:
  unsplit       every family column is 0 while the total is not, so no split was recorded. China
                is 33.6M Muslims with Sunni 0 and Shia 0. Such a zero is never a measurement
                (§12, Tajikistan), so no split row is emitted at all.
  single        one family column equals the total. Usually true (Algeria, Sudan). NOT always:
                **Comoros is 666,746 Shia and 0 Sunni** in both 2000 and 2010, and the Comoros
                are overwhelmingly Sunni. Turkey is 75.7M Sunni and 0 Shia; §15.3 refuses it.
  dual          WRP's own `dualrelig`: adherents exceed population. Japan is 211M on 127M.
  interpolated  `datatype` contains a 3.
  moved         the share moved more than 10 points between 2000 and 2010. It caught **Vietnam,
                75.5% Theravada in 2010** where 2000 had no split; Vietnam's Buddhists are
                overwhelmingly Mahayana.
  routes        route A and route B differ by more than a factor of two.
and three that no flag catches: **Lebanon 2010 is Sunni 1,101,870 and Shia 1,101,870**, a placeholder;
**Sri Lanka is 19.9% Mahayana**, where its Buddhists are Theravada; and **Ahmadiyya is recorded in
Indonesia alone** (500k), while eight built countries draw Ahmadis, so its world total means nothing.
WRP's Buddhist split exists for seven countries and two of them are wrong.

WHAT §15.3'S NODE TEST CANNOT SEE: a source category whose boundary is not a node boundary.
  * WRP's `chrstorth` holds Eastern and Oriental Orthodoxy together. Ethiopia's Tewahedo Church
    (50M) and Egypt's Copts came out as `christianity.orthodox`, which is Eastern, and passed the
    test because those countries draw `christianity.oriental.*`, a sibling. It has no node here.
  * Singapore's census `Taoism` "includes Chinese Traditional Beliefs" and is drawn as
    `chinesefolk`, so a WRP Daoism figure for Singapore restates that column under another name.
    `SAME_COLUMN` refuses it by hand, which is the case-by-case route §15.3 allows.
"""
import argparse
import csv
import io
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "estimates"
PEW_ZIP = RAW / "pew.zip"
WRP_CSV = RAW / "WRP_national.csv"
COW_CSV = RAW / "COW-country-codes.csv"
NE = ROOT / "data" / "geo" / "ne_10m_admin_0_countries.geojson"
COUNTS = ROOT / "data" / "processed" / "counts.json"
TREE = ROOT / "taxonomy" / "religions.json"

PEW_YEAR = "2020"
WRP_YEAR, WRP_BASE = "2010", "2000"

PEW_FAMILIES = {
    "Christians": "christianity", "Muslims": "islam", "Buddhists": "buddhism",
    "Hindus": "hinduism", "Jews": "judaism", "Religiously_unaffiliated": "unaffiliated",
    "Other_religions": None,
}

# Pew family node -> (WRP total column, {WRP family column: node, or None for no node}).
# Judaism's families are left out: the codebook says "we found no data" for them.
SPLITS = {
    "islam": ("islmgen", {"islmsun": "islam.sunni", "islmshi": "islam.shia",
                          "islmahm": "islam.ahmadiyya", "islmibd": None, "islmalw": None,
                          "islmnat": None}),
    "buddhism": ("budgen", {"budmah": "buddhism.mahayana", "budthr": "buddhism.theravada"}),
    "christianity": ("chrstgen", {"chrstcat": "christianity.catholic", "chrstorth": None,
                                  "chrstang": "christianity.anglican", "chrstprot": None}),
}
OTHER = {"zorogen": "zoroastrianism", "sikhgen": "sikhism", "shntgen": "shinto",
         "bahgen": "bahai", "taogen": "daoism", "jaingen": "jainism",
         "confgen": "confucianism", "anmgen": "indigenous", "syncgen": None}
OTHER_DENOM = list(OTHER) + ["othrgen"]
NO_NODE = {
    "islmibd": "Ibadi",
    "islmalw": "Alawite",
    "islmnat": "Nation of Islam",
    "chrstprot": "Protestant (the tree has no Protestant parent; christianity.protestant "
                 "is 'Protestant, unspecified')",
    "chrstorth": "Orthodox (Eastern and Oriental together; the tree keeps them as siblings)",
    "syncgen": "Syncretic",
}

# The refusal rule (with its SAME_COLUMN cases) and the hand rows are the shipping layer's own,
# imported rather than copied: a scout testing a different rule from the one that ships is
# measuring nothing, and a copied rule is a rule that will diverge (§7a-ii).
sys.path.insert(0, str(ROOT))
from estimates import refused  # noqa: E402
from estimates_hand import HAND  # noqa: E402

# Selections the default run reports, in the order the questions were asked.
NODES = ["islam.shia", "islam.sunni", "islam.ahmadiyya", "daoism", "confucianism", "shinto",
         "sikhism", "jainism", "zoroastrianism", "bahai", "alevism", "indigenous",
         "buddhism.theravada", "buddhism.mahayana", "christianity.catholic",
         "christianity.anglican",
         "christianity", "islam", "buddhism", "hinduism", "judaism", "unaffiliated"]
DETAIL = 16          # the first DETAIL nodes get a largest-countries list

# COW state name -> ISO 3166 alpha-2, where no Natural Earth name matches exactly.
COW_ALIAS = {
    "Yugoslavia": "RS", "Serbia": "RS", "Czech Republic": "CZ", "Macedonia": "MK",
    "East Timor": "TL", "Swaziland": "SZ", "Cape Verde": "CV",
    "Sao Tome and Principe": "ST", "Federated States of Micronesia": "FM",
    "St. Kitts and Nevis": "KN", "St. Lucia": "LC", "St. Vincent and the Grenadines": "VC",
    "Antigua & Barbuda": "AG", "Turkey": "TR", "Germany": "DE", "Congo": "CG",
    "Democratic Republic of the Congo": "CD", "Ivory Coast": "CI", "Myanmar": "MM",
    "Kosovo": "XK", "Bahamas": "BS", "Gambia": "GM",
}
# Pew's numeric code -> alpha-2 where Natural Earth carries none.
PEW_NUM_ALIAS = {"412": "XK"}


def norm(s):
    s = str(s).lower().replace("&", " and ").replace(".", " ").replace("'", "").replace("-", " ")
    return " ".join(s.split())


def num(v):
    if v is None:
        return 0.0
    v = str(v).replace(",", "").strip()
    try:
        return float(v)
    except ValueError:
        return 0.0


def cc_of(iso2):
    return "uk" if iso2 == "GB" else iso2.lower()


def approx(n):
    """Two significant figures, and nothing finer than Pew's own 10,000 (§15.4)."""
    if n is None:
        return "-"
    if n < 10_000:
        return "<10k"
    d = 10 ** (math.floor(math.log10(n)) - 1)
    n = round(n / d) * d
    if n >= 1e9:
        return f"{n / 1e9:g}bn"
    if n >= 1e6:
        return f"{n / 1e6:g}m"
    return f"{n / 1e3:g}k"


def pct(x):
    if x is None:
        return "-"
    return f"{x * 100:.2f}%" if x < 0.001 else f"{x * 100:.1f}%"


# ------------------------------------------------------------------------------------------
# loading and joining
# ------------------------------------------------------------------------------------------

def load_ne():
    feats = json.loads(NE.read_text(encoding="utf-8"))["features"]
    by_num, by_name, conflicts = {}, defaultdict(set), []
    names = ("ADMIN", "NAME", "NAME_LONG", "BRK_NAME", "NAME_SORT", "NAME_ALT", "NAME_CIAWF",
             "NAME_EN", "FORMAL_EN", "GEOUNIT", "SUBUNIT")
    for eh in (False, True):             # the plain codes first; the _EH ones only fill gaps
        for f in feats:
            p = f["properties"]
            k2, k3 = ("ISO_A2_EH", "ISO_N3_EH") if eh else ("ISO_A2", "ISO_N3")
            iso2 = str(p.get(k2) or "")
            if len(iso2) != 2:
                continue
            n3 = str(p.get(k3) or "")
            if n3.isdigit():
                n3 = n3.zfill(3)
                if n3 in by_num and by_num[n3] != iso2:
                    if not eh:
                        conflicts.append((n3, by_num[n3], iso2))
                else:
                    by_num[n3] = iso2
            for k in names:
                if p.get(k):
                    by_name[norm(p[k])].add(iso2)
    return by_num, by_name, conflicts


def load_pew():
    with zipfile.ZipFile(PEW_ZIP) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        rows = list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig"))))
    by_country, world = defaultdict(dict), {}
    for r in rows:
        rec = dict(code=r["Countrycode"].strip().zfill(3), pop=num(r["Population"]),
                   **{f: num(r[f]) for f in PEW_FAMILIES})
        if r["Level"] == "1":
            by_country[r["Country"]][r["Year"]] = rec
        elif r["Level"] == "3":
            world[r["Year"]] = rec
    return by_country, world


def load_wrp():
    with open(WRP_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_state = defaultdict(dict)
    for r in rows:
        by_state[r["state"].strip()][r["year"].strip()] = r
    with open(COW_CSV, encoding="utf-8-sig", newline="") as f:
        cow = {r["CCode"].strip(): r["StateNme"].strip() for r in csv.DictReader(f)}
    return by_state, cow


def join():
    by_num, by_name, conflicts = load_ne()
    pew, pew_world = load_pew()
    wrp, cow = load_wrp()
    report = defaultdict(list)
    report["ne_conflicts"] = conflicts

    pew_iso = {}
    for name, years in pew.items():
        code = years[PEW_YEAR]["code"]
        iso2 = PEW_NUM_ALIAS.get(code) or by_num.get(code)
        if iso2:
            pew_iso[iso2] = name
        else:
            report["pew_unjoined"].append((name, code, years[PEW_YEAR]["pop"]))

    wrp_iso = {}
    for state, years in wrp.items():
        if WRP_YEAR not in years:
            continue
        sname = cow.get(state, years[WRP_YEAR]["name"])
        iso2 = COW_ALIAS.get(sname)
        if not iso2:
            hits = by_name.get(norm(sname), set())
            if len(hits) == 1:
                iso2 = next(iter(hits))
            elif len(hits) > 1:
                report["wrp_ambiguous"].append((state, sname, sorted(hits)))
                continue
        if not iso2:
            report["wrp_unjoined"].append((state, sname, num(years[WRP_YEAR]["pop"])))
            continue
        if iso2 in wrp_iso:
            report["wrp_duplicate"].append((iso2, wrp_iso[iso2], state))
            continue
        wrp_iso[iso2] = state
        # a wrong twin disagrees about population; a genuine source disagreement also lands here
        if iso2 in pew_iso and "2010" in pew[pew_iso[iso2]]:
            a, b = num(years[WRP_YEAR]["pop"]), pew[pew_iso[iso2]]["2010"]["pop"]
            if b and not 0.75 <= a / b <= 1.33:
                report["pop_disagrees"].append((iso2, sname, pew_iso[iso2], a, b))
    for iso2 in wrp_iso:
        if iso2 not in pew_iso:
            report["wrp_not_in_pew"].append((iso2, cow.get(wrp_iso[iso2])))
    return pew, pew_world, pew_iso, wrp, wrp_iso, report


def load_covers():
    cj = json.loads(COUNTS.read_text(encoding="utf-8"))
    return {cc: set(c.get("covers") or []) for cc, c in cj["countries"].items()}


def load_tree():
    return {n["id"] for n in json.loads(TREE.read_text(encoding="utf-8"))["nodes"]}


# ------------------------------------------------------------------------------------------
# the estimate table
# ------------------------------------------------------------------------------------------

def build():
    pew, pew_world, pew_iso, wrp, wrp_iso, report = join()
    covers = load_covers()
    tree = load_tree()
    rows, unsplit, unmapped = [], [], []

    def emit(**kw):
        kw.setdefault("flags", [])
        kw.setdefault("low", None)
        kw.setdefault("high", None)
        kw.setdefault("check", None)
        rows.append(kw)

    for iso2, pname in sorted(pew_iso.items()):
        cc = cc_of(iso2)
        p = pew[pname][PEW_YEAR]
        for fam, node in PEW_FAMILIES.items():
            if node:
                emit(cc=cc, iso2=iso2, name=pname, node=node, of="country",
                     share=p[fam] / p["pop"] if p["pop"] else None, people=p[fam],
                     chain="pew_2020", year="2020", source="Pew 2020")

        state = wrp_iso.get(iso2)
        if not state:
            continue
        w = wrp[state][WRP_YEAR]
        w0 = wrp[state].get(WRP_BASE)
        base = []
        if "3" in w.get("datatype", ""):
            base.append("interpolated")
        if w.get("dualrelig", "").strip() == "1":
            base.append("dual")

        for fam, (tot, cols) in SPLITS.items():
            T = num(w[tot])
            if T <= 0:
                continue
            named = {c: num(w[c]) for c in cols}
            if sum(named.values()) <= 0:
                unsplit.append((cc, pname, fam, T))
                continue
            positive = [c for c, v in named.items() if v > 0]
            single = len(positive) == 1 and named[positive[0]] >= 0.999 * T
            famname = next(k for k, v in PEW_FAMILIES.items() if v == fam)
            for c in positive:
                share = named[c] / T
                if cols[c] is None:
                    unmapped.append((cc, pname, NO_NODE[c], share * p[famname]))
                    continue
                flags = list(base) + (["single"] if single else [])
                if w0 and num(w0[tot]) > 0 and abs(num(w0[c]) / num(w0[tot]) - share) > 0.10:
                    flags.append("moved")
                emit(cc=cc, iso2=iso2, name=pname, node=cols[c], of=fam, share=share,
                     people=share * p[famname], chain="pew_2020 x wrp_2010", year="2010 share",
                     source="Pew 2020 total, WRP 2010 share", flags=flags)

        D = sum(num(w[c]) for c in OTHER_DENOM)
        wpop = num(w["pop"])
        for c, node in OTHER.items():
            v = num(w[c])
            if v <= 0 or D <= 0:
                continue
            a = v / D * p["Other_religions"]
            b = v / wpop * p["pop"] if wpop else None
            if node is None:
                unmapped.append((cc, pname, NO_NODE[c], a))
                continue
            flags = list(base)
            if b and (a == 0 or not 0.5 <= a / b <= 2):
                flags.append("routes")
            emit(cc=cc, iso2=iso2, name=pname, node=node, of="pew other", share=v / D,
                 people=a, check=b, chain="pew_2020 x wrp_2010", year="2010 share",
                 source="Pew 2020 other religions, WRP 2010 share of them", flags=flags)

    for o in HAND:
        iso2 = "GB" if o["cc"] == "uk" else o["cc"].upper()
        pname = pew_iso[iso2]
        pop = pew[pname][PEW_YEAR]["pop"]
        for r in rows:
            if r["cc"] == o["cc"] and r["node"] == o["node"]:
                r["superseded"] = True
        single = o["low"] == o["high"]
        emit(cc=o["cc"], iso2=iso2, name=pname, node=o["node"], of="country",
             share=o["low"] if single else None,
             low=None if single else o["low"] * pop, high=None if single else o["high"] * pop,
             people=o["low"] * pop if single else None,
             chain="hand x pew_2020 population", year=o["year"], source=o["source"],
             flags=["hand"])

    for r in rows:
        assert r["node"] in tree, f"{r['node']} is not a node in religions.json"
        why = refused(covers, r["cc"], r["node"])
        r["status"] = (f"refused: {why}" if why
                       else "shown, not built" if r["cc"] not in covers else "shown, built")
    return dict(rows=rows, unsplit=unsplit, unmapped=unmapped, report=report, covers=covers,
                pew=pew, pew_world=pew_world, pew_iso=pew_iso, wrp_iso=wrp_iso)


# ------------------------------------------------------------------------------------------
# printing
# ------------------------------------------------------------------------------------------

def live(rows, node):
    return [r for r in rows if r["node"] == node and not r.get("superseded")
            and ((r["people"] or 0) > 0 or (r["high"] or 0) > 0)]


def mid(r):
    if r["people"] is not None:
        return r["people"]
    return (r["low"] + r["high"]) / 2


def summary(t, top):
    rows, covers = t["rows"], t["covers"]
    print(f"Pew countries joined {len(t['pew_iso'])} of {len(t['pew'])}; WRP joined "
          f"{len(t['wrp_iso'])}; built countries {len(covers)}; estimate-only countries "
          f"{sum(1 for i in t['pew_iso'] if cc_of(i) not in covers)}\n")
    head = (f"{'node':24s} {'drawn':>5s} {'est':>4s} {'shown':>5s} {'built':>5s} {'refused':>7s}"
            f"  {'world, all':>11s} {'world, shown':>12s}  flags")
    print(head)
    print("-" * len(head))
    for node in NODES:
        L = live(rows, node)
        drawn = sum(1 for cov in covers.values()
                    if any(c == node or c.startswith(node + ".") for c in cov))
        shown = [r for r in L if r["status"].startswith("shown")]
        built = [r for r in shown if r["status"] == "shown, built"]
        refused = [r for r in L if r["status"].startswith("refused")]
        flags = Counter(f for r in L for f in r["flags"])
        print(f"{node:24s} {drawn:5d} {len(L):4d} {len(shown):5d} {len(built):5d} "
              f"{len(refused):7d}  {approx(sum(mid(r) for r in L)):>11s} "
              f"{approx(sum(mid(r) for r in shown)):>12s}  "
              + " ".join(f"{k}:{v}" for k, v in sorted(flags.items())))

    w = t["pew_world"].get(PEW_YEAR)
    if w:
        print("\ncheck, Pew's own world row against the sum of its countries:")
        for fam, node in PEW_FAMILIES.items():
            if node:
                s = sum(r["people"] for r in live(rows, node) if r["chain"] == "pew_2020")
                print(f"  {node:14s} countries {approx(s):>7s}   Pew world {approx(w[fam]):>7s}")

    for node in NODES[:DETAIL]:
        L = sorted((r for r in live(rows, node) if r["status"].startswith("shown")),
                   key=mid, reverse=True)
        if not L:
            continue
        print(f"\n{node}: largest {min(top, len(L))} of {len(L)} outlined")
        for r in L[:top]:
            rng = f" ({approx(r['low'])}-{approx(r['high'])})" if r["low"] is not None else ""
            chk = f"  B {approx(r['check'])}" if r["check"] is not None else ""
            print(f"  {r['cc']:3s} {r['name'][:22]:22s} {pct(r['share']):>7s} of {r['of']:12s}"
                  f" {approx(r['people']):>7s}{rng}{chk}  {r['status']}"
                  + (f"  [{', '.join(r['flags'])}]" if r["flags"] else ""))

    print(f"\nunsplit (no family recorded, so no split rows): {len(t['unsplit'])}")
    for fam in SPLITS:
        L = sorted((u for u in t["unsplit"] if u[2] == fam), key=lambda u: -u[3])
        print(f"  {fam}: {len(L)}, largest " + ", ".join(f"{u[1]} {approx(u[3])}" for u in L[:8]))
    by = defaultdict(float)
    for _, _, what, n in t["unmapped"]:
        by[what] += n
    print("\nno node, so not drawable at all (world, via the chain):")
    for what, n in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {approx(n):>7s}  {what}")


def one(t, node):
    L = sorted((r for r in t["rows"] if r["node"] == node), key=lambda r: -(mid(r) or 0))
    for r in L:
        rng = f" range {approx(r['low'])}-{approx(r['high'])}" if r["low"] is not None else ""
        chk = f" check {approx(r['check'])}" if r["check"] is not None else ""
        sup = " SUPERSEDED" if r.get("superseded") else ""
        print(f"{r['cc']:3s} {r['name'][:24]:24s} {pct(r['share']):>7s} of {r['of']:12s} "
              f"{approx(r['people']):>7s}{rng}{chk}  {r['status']}{sup}  {r['chain']}"
              + (f"  [{', '.join(r['flags'])}]" if r["flags"] else ""))


def joins(t):
    rep = t["report"]
    for key in ("ne_conflicts", "pew_unjoined", "wrp_unjoined", "wrp_ambiguous",
                "wrp_duplicate", "pop_disagrees", "wrp_not_in_pew"):
        print(f"\n{key}: {len(rep[key])}")
        for item in rep[key]:
            print("  ", item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("node", nargs="?")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--joins", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    t = build()
    if a.joins:
        joins(t)
    elif a.node:
        one(t, a.node)
    else:
        summary(t, a.top)
    if a.csv:
        keys = ["cc", "iso2", "name", "node", "of", "share", "low", "high", "people", "check",
                "chain", "year", "status", "superseded", "flags", "source"]
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            for r in t["rows"]:
                wr.writerow({**r, "flags": " ".join(r["flags"])})
        print(f"\nwrote {len(t['rows'])} rows to {a.csv}")


if __name__ == "__main__":
    main()
