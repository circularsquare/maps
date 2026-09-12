"""Germany — splitting the register's 42.8M residual, from ESS `rlgdnade`.

Reads (or fetches) data/raw/de_ess/ and writes data/normalized/de_ess.csv.

WHAT THIS IS FOR.  Zensus 2022 reads religion off the Melderegister, so Germany is three
categories and 51.8% of the country lands in "Sonstige, keine, ohne Angabe" — Muslims,
Orthodox, free churches and everyone who belongs to nothing, in one cell, because the
register only knows what church tax requires (sources/de.md §2).  ZWST took the Jewish
communities out of it by counting them (de.md §7).  Nothing counts the rest, anywhere, so
this is a MODEL and it is the only route to them.

THE OPERATION is spec §3.4's permitted SPLIT and §14.10's permitted fractional-share model:
each Gemeinde keeps the register's own Catholic, Protestant and residual totals, and the
RESIDUAL is divided by its Bundesland's ESS composition.  The magnitude is always the
German state's own count at its own geography; the model only says what is inside it.

WHAT LICENSES IT IS THE CROSS-CHECK, and it is unusually good.  ESS and the register are
independent instruments that both see Catholics and Protestants, so they can be compared on
those two before the third is trusted to either.  Pooled over five rounds ESS gives 24.7%
Catholic and 23.9% EKD against the register's 25.1% and 23.1% — within half a point on one
and eight tenths on the other.  A second check arrived free: ESS puts Jews at 0.10% of
Germany and ZWST COUNTED 87,934, which is 0.106%.  Neither check was used to fit anything.

FIVE ROUNDS, NOT THREE.  `sources/de.md` §6 scouted this and reported rounds 8, 9 and 11 as
the usable ones.  Rounds 6 and 7 carry `rlgdnade` and `region` as well and are used here:
13,643 respondents rather than ~7,600, which is the difference between Bremen being usable
and not.  Rounds 5 and 10 raise `E201VariableNotFound` — round 5 has no `rlgdnade` and round
10 is the COVID round, whose German fieldwork carried a reduced variable set.

THE NON-RELIGIOUS ARE NOT DRAWN, AND THAT IS THE MAIN DESIGN DECISION IN THIS FILE.
ESS's largest German answer is "Not applicable" — 41.5%, everyone who said they belong to no
religion — and France maps exactly that answer to `unaffiliated` (fr2024.py).  Germany does
not, for two reasons that do not apply to France:

  * **`unrecorded` is a MEASURED cell here and France has no equivalent.**  The register
    genuinely counted 42.8M people into it.  Replacing a measured cell with a modelled
    `unaffiliated` trades something counted for something estimated, which is a downgrade
    dressed as detail.  Carving the religions out and leaving the rest keeps the measured
    cell — smaller, still measured, still honest about being a register artefact.
  * **spec §14.12: an ancestry-shaped cell models well and an attitude-shaped one badly.**
    Islam, Orthodoxy and the free churches in Germany are close to functions of descent and
    are the cells this method is best at.  Non-belief is a behaviour inside every group at
    once and is the cell it is worst at.  Drawing the religions and not the irreligion is
    that rule applied rather than restated.

So the split takes the residual's ~19% that ESS assigns to a named religion and leaves ~81%
where the register put it.  Germany stays about 90% `measured`, and `inferred dots: not
shown` returns it to very nearly the map it is today.

WHAT THE MODEL CANNOT SEE, and note_public has to say so:

  * **ESS UNDERCOUNTS MUSLIMS AND THE DIRECTION IS NOT IN DOUBT.**  4.13% pooled is about
    3.4M against BAMF's ~5.5M.  Four mechanisms push the same way: ESS samples residents
    15+ and Germany's Muslim population is markedly younger than average, so the all-ages
    share is higher than any 15+ survey can report; interviewing is in German, which sheds
    exactly the recent arrivals who are most likely to be Muslim; the household frame misses
    collective accommodation; and non-response correlates with migration background.  NONE
    OF IT IS CORRECTED HERE.  Rescaling to BAMF would fit the coefficient to the only
    independent check the model has, which §14.10's second condition forbids and §14.12
    calls spending the check — and it would have to take the difference out of the
    non-religious, for which there is no evidence at all.  It is stated instead.
  * **The composition is a Bundesland's and the geography is a Gemeinde's.**  Within a
    Land every municipality gets the same composition of its residual.  The 1km citizenship
    grid (de.md §6, Route B) is the fix for that and is not built.
  * **Pooling spans 2012 to 2023.**  The Muslim share rises across the window and the
    Catholic share falls, so a pooled figure understates the present on one and overstates
    it on the other.

Usage:
    python sources/de_ess.py --fetch    query the ESS API (5 rounds, ~10 calls)
    python sources/de_ess.py            pool and normalise from data/raw/de_ess/
"""

import csv
import json
import os
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "de_ess")
OUT = os.path.join(ROOT, "data", "normalized", "de_ess.csv")

SOURCE_ID = "de_ess_r6_11"
YEAR = 2023                    # last fieldwork year in the pool (round 11)
# A person's own answer to "which denomination do you belong to", not a register entry.
# The register rows this splits are `roll`; see the docstring and spec §3.1.
BASIS = "self_id"

UA = {"User-Agent": "religiondots/1.0 (+https://github.com/) research",
      "Content-Type": "application/json"}
ESS_API = "https://api.nsd.no/graphql"

# (datafile id, version), same ids sources/fr.py uses. Round 5 has no `rlgdnade`; round 10
# is the COVID round and raises E201VariableNotFound for it too.
ESS_ROUNDS = {
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    7: ("9c96a1b2-b027-43c1-8c74-e883f892d0bb", 91),
    8: ("ffc43f48-e15a-4a1c-8813-47eda377c355", 98),
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}

# `weightVariable` is the only difference between the two. pspwght is normalised to the
# sample size, so both totals are 2,420 in round 11 — but the per-cell WEIGHTED values are
# fractional and must not be used as respondent counts. N_FLOOR needs the unweighted pass.
_TAB = """query($id:ID!,$v:Int!,$bv:[String!]!){analysis{
 frequencyTabulationByVariables(input:{
   datafile:{id:$id,version:$v}, breakVariables:$bv, byVariables:["cntry"],
   instance:PUBLISHED, agencyId:INT_ESSERIC, includeMissing:true,%s
   metadataLanguage:"en"}){
 responses{by{value} response{
   variableValues{name values codeList{value label isMissing}} table{path count}}}}}}"""
ESS_TAB_W = _TAB % ' weightVariable:"pspwght",'
ESS_TAB_N = _TAB % ""

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category",
           "count", "respondents", "basis", "year", "source_id", "note"]

# ESS region label -> the two-digit Land prefix of the Amtlicher Gemeindeschlüssel.
LAND = {
    "Schleswig-Holstein": "01", "Hamburg": "02", "Niedersachsen": "03",
    "Bremen": "04", "Nordrhein-Westfalen": "05", "Hessen": "06",
    "Rheinland-Pfalz": "07", "Baden-Württemberg": "08", "Bayern": "09",
    "Saarland": "10", "Berlin": "11", "Brandenburg": "12",
    "Mecklenburg-Vorpommern": "13", "Sachsen": "14", "Sachsen-Anhalt": "15",
    "Thüringen": "16",
}

# The two answers the register already measures. They are the CROSS-CHECK and are never
# part of the split — the register's own Catholic and Protestant counts stand untouched.
REGISTER_ANSWERS = {
    "Römisch-Katholisch": "catholic",
    "Evangelisch/Protestantisch (EKD, ohne Freikirchen)": "protestant",
}
# Counted by ZWST at real community seats, so it is measured and must not be modelled on
# top (sources/de_zwst.py). Excluded from BOTH sides of the ratio, because the residual
# this file splits has already had them taken out of it.
COUNTED_ELSEWHERE = {"Jüdisch"}
# Not a religion answer: these stay in `unrecorded`, where the register put them.
NOT_DRAWN = {"Not applicable", "Refusal", "No answer", "Don't know"}

# Below this many pooled respondents a Bundesland takes the national composition instead of
# its own. sources/it.py's lesson, and the number is it.py's: a headline claim about a real
# place should not rest on a few dozen people. Bremen is the one that trips it.
N_FLOOR = 100

# The register's national shares, from sources/de.py's NATIONAL. Used only to PRINT the
# cross-check; nothing is fitted to them.
REGISTER_NATIONAL = {"catholic": 0.251, "protestant": 0.231}
ZWST_NATIONAL = 87_934


def _ess(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(ESS_API, data=body, headers=UA), timeout=900))
    if "errors" in r:
        e = r["errors"][0]
        raise SystemExit(f"!! ESS API: {e.get('message')} "
                         f"{e.get('extensions', {}).get('code', '')}")
    return r["data"]


def _pull(rnd, fid, ver, weighted):
    d = _ess(ESS_TAB_W if weighted else ESS_TAB_N,
             {"id": fid, "v": ver, "bv": ["region", "rlgdnade"]})
    resp = d["analysis"]["frequencyTabulationByVariables"]["responses"]
    de = [r for r in resp if r["by"][0]["value"] == "DE"]
    if not de:
        raise SystemExit(f"!! ESS round {rnd} has no DE response")
    return de[0]


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for rnd, (fid, ver) in ESS_ROUNDS.items():
        for weighted in (True, False):
            name = f"ess_de_r{rnd}{'_w' if weighted else '_n'}.json"
            dest = os.path.join(RAW, name)
            if os.path.exists(dest) and os.path.getsize(dest) > 500:
                print(f"  have {name}")
                continue
            print(f"  round {rnd} {'weighted' if weighted else 'unweighted'}…")
            d = _pull(rnd, fid, ver, weighted)
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False)


def _table(path):
    """-> [(region, denomination, count)] for one saved response."""
    d = json.load(open(path, encoding="utf-8"))
    r = d["response"]
    names = [v["name"] for v in r["variableValues"]]
    codes = [{c["value"]: c["label"] for c in v["codeList"]} for v in r["variableValues"]]
    vals = [v["values"] for v in r["variableValues"]]
    ri, di = names.index("region"), names.index("rlgdnade")
    out = []
    for cell in r["table"]:
        p = cell["path"]
        out.append((codes[ri].get(vals[ri][p[ri]], "?"),
                    codes[di].get(vals[di][p[di]], "?"),
                    float(cell["count"])))
    return out


def read():
    """Pooled (region, denomination) -> [weighted, respondents]."""
    pooled, rounds_used = {}, []
    for rnd in sorted(ESS_ROUNDS):
        w = os.path.join(RAW, f"ess_de_r{rnd}_w.json")
        n = os.path.join(RAW, f"ess_de_r{rnd}_n.json")
        if not (os.path.exists(w) and os.path.exists(n)):
            raise SystemExit(f"round {rnd} not downloaded — run with --fetch")
        rounds_used.append(rnd)
        for i, path in enumerate((w, n)):
            for reg, den, c in _table(path):
                pooled.setdefault((reg, den), [0.0, 0.0])[i] += c
    return pooled, rounds_used


def build(pooled, rounds_used):
    regions = sorted({r for r, _ in pooled})
    unknown = [r for r in regions if r not in LAND]
    if unknown:
        raise SystemExit(f"ESS region(s) not in LAND: {unknown}")

    dens = sorted({d for _, d in pooled})
    print(f"  {len(rounds_used)} rounds {rounds_used}, {len(regions)} regions, "
          f"{len(dens)} denominations")

    nat = {}
    for (_, den), (w, n) in pooled.items():
        e = nat.setdefault(den, [0.0, 0.0])
        e[0] += w
        e[1] += n
    nat_total = sum(v[0] for v in nat.values())

    print(f"\n  pooled {nat_total:,.0f} respondents")
    for den, (w, n) in sorted(nat.items(), key=lambda kv: -kv[1][0]):
        mark = ("  register" if den in REGISTER_ANSWERS else
                "  ZWST" if den in COUNTED_ELSEWHERE else
                "  not drawn" if den in NOT_DRAWN else "")
        print(f"    {den[:52]:<52} {100 * w / nat_total:6.2f}%  n={n:>6,.0f}{mark}")

    # ---- the cross-check that licenses the split (spec §14.10 condition 5)
    ok = True
    print("\n  cross-check against the register, on the two categories both instruments see:")
    for den, key in REGISTER_ANSWERS.items():
        got = nat.get(den, [0.0, 0.0])[0] / nat_total
        want = REGISTER_NATIONAL[key]
        good = abs(got - want) <= 0.02
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {key:<11} ESS {100 * got:5.2f}%  "
              f"register {100 * want:5.1f}%  diff {100 * (got - want):+5.2f}pp")
    jew = nat.get("Jüdisch", [0.0, 0.0])[0] / nat_total
    print(f"    -- judaism    ESS {100 * jew:5.3f}%  ZWST counted {ZWST_NATIONAL:,} "
          f"({100 * ZWST_NATIONAL / 82_719_540:.3f}%) — held out, not fitted")

    # ---- the split shares, per Bundesland
    def composition(cells):
        """denomination -> share OF THE RESIDUAL, for the religions only."""
        drawn = {d: w for d, (w, _) in cells.items()
                 if d not in REGISTER_ANSWERS and d not in COUNTED_ELSEWHERE
                 and d not in NOT_DRAWN}
        stays = sum(w for d, (w, _) in cells.items()
                    if d in NOT_DRAWN)
        base = sum(drawn.values()) + stays
        if base <= 0:
            return {}
        return {d: w / base for d, w in drawn.items()}

    nat_comp = composition(nat)
    print(f"\n  nationally {100 * sum(nat_comp.values()):.1f}% of the residual is a named "
          f"religion and the rest stays in `unrecorded`")

    rows, thin = [], []
    for reg in regions:
        cells = {d: pooled[(reg, d)] for _, d in pooled if (reg, d) in pooled}
        cells = {d: v for d, v in ((d, pooled.get((reg, d))) for d in dens) if v}
        n_reg = sum(v[1] for v in cells.values())
        comp = composition(cells)
        if n_reg < N_FLOOR or not comp:
            thin.append((reg, n_reg))
            comp = nat_comp
            note = (f"n={n_reg:.0f} < N_FLOOR {N_FLOOR}; national composition used "
                    f"instead of this Land's own")
        else:
            note = f"pooled ESS rounds {rounds_used}; n={n_reg:.0f}"
        for den, share in sorted(comp.items()):
            rows.append({
                "geo_id": LAND[reg], "geo_level": "bundesland", "geo_name": reg,
                "source_category": den, "count": round(share, 8),
                "respondents": int(round(cells.get(den, [0, 0])[1])),
                "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID, "note": note,
            })

    print(f"\n  {'OK ' if not thin else '-- '}{len(thin)} Land(s) below N_FLOOR "
          f"({N_FLOOR}) and given the national composition"
          + (": " + ", ".join(f"{r} n={n:.0f}" for r, n in thin) if thin else ""))

    # a share is a share
    bad = [r for r in rows if not (0.0 <= r["count"] <= 1.0)]
    if bad:
        raise SystemExit(f"{len(bad)} shares outside [0,1]")
    per_land = {}
    for r in rows:
        per_land[r["geo_id"]] = per_land.get(r["geo_id"], 0.0) + r["count"]
    worst = max(per_land.values())
    good = worst <= 1.0
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} every Land's religions sum to at most 1 of its "
          f"residual; largest is {100 * worst:.1f}%")

    print("\n  share of the residual drawn as a religion, by Land:")
    for gid, tot in sorted(per_land.items(), key=lambda kv: -kv[1]):
        name = next(r["geo_name"] for r in rows if r["geo_id"] == gid)
        isl = next((r["count"] for r in rows if r["geo_id"] == gid
                    and r["source_category"] == "Muslimisch/Islam"), 0.0)
        print(f"    {name:<24} {100 * tot:5.1f}%   of which Islam {100 * isl:5.1f}%")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows = build(*read())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
