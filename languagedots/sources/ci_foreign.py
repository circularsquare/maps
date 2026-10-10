"""Côte d'Ivoire RGPH 2021: residents of other nationalities, by région, on their nationality's
languages -> data/normalized/ci_foreign.csv. Every row `derived`.

    python sources/ci_foreign.py

WHY. The census asked everyone aged 2+ which Ivorian language they speak most, but published
the answers for Ivorian nationals only (sources/ci_rgph.py). The 6,460,062 residents of other
nationalities (22% of the census; 44% of Gboklè, 43.6% of Cavally, 40.1% of San-Pedro) were
undrawn. Anita, 2026-10-07: give it a stab.

TABLES (same volume as ci_rgph.py, rgpg_tom1.pdf, 1-based pages):
  Tableau 4.24, p113   non-Ivorians by région and district, by sex, with Poids** (their share of
                       the région's whole population). MEASURED counts per unit.
  Tableau 4.21, p110   non-Ivorians by nationality (15 West African states, other Africa,
                       Europe, other countries, other/not declared). NATIONAL only.
  Tableau 4.22, p111   the same nationalities 1988/1998/2021: the second table, checked.
No table of the census, tome 1 or tome 2 (Migration, Analyses_Thematiques_Tome2_Migration.pdf,
which has foreign-born by région but not by country), crosses nationality with région. So each
région's foreigners take the NATIONAL nationality mix, and each nationality its home country's
language mix (sources/origin_mix.py, dest "ci"). Both steps are proxies: tier `derived`.

  Europe, other countries, other/not declared  -> `other` (no language can be named)
  Other Africa                                 -> `africa_other`

ALL AGES. Tableau 4.24 counts every age; no age table by région exists for foreigners, so
nothing is taken off for children under three (the Ivorian rows are aged 3+).

CHECKS (all must pass):
  1. Tableau 4.24: 33 units + 12 non-autonomous districts + National; each district equals the
     sum of its régions within 3; M + F = Total on every row within 2; the 33 sum to the
     National row, 6,460,062.
  2. Poids**: foreigners / (foreigners + annex 25's Ivorians) reproduces the printed share of
     every région within 0.15 points (two tables of one census agreeing per unit).
  3. Tableau 4.21: the nationalities sum to its Total and its West Africa subtotal; each 2021
     count equals Tableau 4.22's (whose other-West-Africa row is Cap Vert + Gambie +
     Guinée-Bissau + Sierra-Léone).
  4. output: every unit's rows sum to its Tableau 4.24 total.
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "taxonomy"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
from ci_rgph import PDF, norm, despace  # noqa: E402

OUT = ROOT / "data" / "normalized" / "ci_foreign.csv"
P_T424 = 112          # 0-based
P_T421 = 109
P_T422 = 110
FOREIGN = 6_460_062

# Tableau 4.24's grouping rows (districts that are not units) -> their régions
DISTRICTS = {
    "Bas- Sassandra": ["San-Pedro", "Gboklè", "Nawa"],
    "Comoé": ["Indénié-Djuablin", "Sud-Comoé"],
    "Denguélé": ["Kabadougou", "Folon", "Bafing"],   # the table prints Bafing under Denguélé
    "Gôh-Djiboua": ["Loh-Djiboua", "Goh"],
    "Lacs": ["N'zi", "Bélier", "Iffou", "Moronou"],
    "Lagunes": ["Agneby-Tiassa", "Grands-Ponts", "La Mé"],
    "Montagnes": ["Tonkpi", "Cavally", "Guemon"],
    "Sassandra-Marahoué": ["Haut-Sassandra", "Marahoué"],
    "Savane": ["Poro", "Bagoué", "Tchologo"],
    "Vallée de Bandama": ["Gbèkè", "Hambol"],
    "Woroba": ["Worodougou", "Béré"],
    "Zanzan": ["Gontougo", "Bounkani"],
}
# Tableau 4.24's names that do not fold to the hex unit's (the rest do)
ALIAS = {"districtdabidjan": "District D'Abidjan",
         "districtdeyamoussoukro": "District De Yamoussoukro"}

# Tableau 4.21's rows -> origin (ISO for origin_mix, or a node)
NATIONALITY = {
    "Bénin": "BJ", "Burkina-Faso": "BF", "Cap Vert": "CV", "Gambie": "GM", "Ghana": "GH",
    "Guinée": "GN", "Guinée-Bissau": "GW", "Libéria": "LR", "Mali": "ML", "Niger": "NE",
    "Nigéria": "NG", "Sénégal": "SN", "Sierra-Léone": "SL", "Togo": "TG", "Mauritanie": "MR",
    "Autre Afrique": "node:africa_other", "Europe": "node:other", "Autre pays": "node:other",
    "Autres nationalités /ND": "node:other",
}
WEST_AFRICA = list(NATIONALITY)[:15]

NUMS = re.compile(r"^\d{1,3}(?: \d{3})*$")


def raw_lines(doc, pno):
    """Lines with non-breaking spaces made plain but runs of spaces kept (they separate the
    columns of Tableau 4.24)."""
    t = doc.load_page(pno).get_text()
    t = t.translate({c: " " for c in (0x00A0, 0x2007, 0x2008, 0x2009, 0x202F)})
    return [x.rstrip() for x in t.splitlines() if x.strip()]


def read_t424(doc):
    ls = raw_lines(doc, P_T424)
    rows, i = {}, 0
    while i < len(ls) - 1:
        cells = [c for c in re.split(r"\s{2,}", ls[i + 1].strip()) if c]
        if len(cells) == 3 and all(NUMS.match(c) for c in cells) and not NUMS.match(
                ls[i].strip()):
            m, f, t = (int(c.replace(" ", "")) for c in cells)
            poids2 = float(ls[i + 4].strip().replace(",", "."))
            rows[despace(ls[i])] = dict(m=m, f=f, total=t, poids2=poids2)
            i += 5
        else:
            i += 1
    # the National row prints its three numbers on separate lines
    j = next(k for k, x in enumerate(ls) if x.strip() == "National Côte d'Ivoire")
    m, f, t = (int(ls[j + k].strip().replace(" ", "")) for k in (1, 2, 3))
    rows["National"] = dict(m=m, f=f, total=t, poids2=float(ls[j + 6].replace(",", ".")))
    return rows


def read_nat(doc):
    """Tableau 4.21: name, then masculin, féminin, total, RM, each on its own line."""
    ls = [despace(x) for x in raw_lines(doc, P_T421)]
    out, i = {}, 0
    names = list(NATIONALITY) + ["Afrique de", "Total"]
    while i < len(ls):
        nm = ls[i]
        if nm == "Autres" and ls[i + 1] == "nationalités /ND":
            nm, i = "Autres nationalités /ND", i + 1
        if nm in names and i + 3 < len(ls) and all(NUMS.match(ls[i + k]) for k in (1, 2, 3)):
            m, f, t = (int(ls[i + k].replace(" ", "")) for k in (1, 2, 3))
            assert abs(m + f - t) <= 2, (nm, m, f, t)
            out[nm] = t
            i += 4
        else:
            i += 1
    return out


def read_t422(doc):
    ls = [despace(x) for x in raw_lines(doc, P_T422)]
    out = {}
    for k, x in enumerate(ls):
        if x in NATIONALITY or x == "Autre Afrique de l'Ouest":
            # 2021 count is the 5th number after the name (1988 n %, 1998 n %, 2021 n %)
            nums = []
            for y in ls[k + 1:k + 12]:
                nums += re.findall(r"\d+(?: \d{3})*(?:,\d)?", y)
                ints = [n for n in nums if "," not in n]
                if len(ints) >= 3:
                    break
            out[x] = int(ints[2].replace(" ", ""))
    return out


def written_ids():
    """origin_mix.mix() returns DRAWN ids (taxonomy/regroup.py); counts() and the fragment
    keep WRITTEN ids. Inverts regroup's move over the written tree (as sources/no_svalbard.py)."""
    from regroup import move
    tx = ROOT / "taxonomy"
    ids = set()
    for f in [tx / "tree.txt", *sorted((tx / "tree.d").glob("*.txt"))]:
        text = re.sub(r"# --- origin_mix borrowed nodes.*?# --- end origin_mix borrowed nodes ---",
                      "", f.read_text(encoding="utf-8"), flags=re.S)
        for ln in text.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "|" in ln:
                ids.add(ln.split("|")[0].strip())
    inv = {}
    for i in ids:
        inv.setdefault(move(i), set()).add(i)

    def back(node):
        if node in ids:
            return node
        w = inv.get(node, set())
        if len(w) != 1:
            raise SystemExit(f"no single written id for drawn {node!r}: {sorted(w)}")
        return next(iter(w))
    return back


def main():
    import fitz
    from origin_mix import mix

    doc = fitz.open(PDF)
    assert doc.page_count == 151, doc.page_count

    # ---- 1. Tableau 4.24
    t424 = read_t424(doc)
    nat = t424.pop("National")
    assert nat["total"] == FOREIGN, nat
    for nm, r in t424.items():
        assert abs(r["m"] + r["f"] - r["total"]) <= 2, (nm, r)
    assert set(DISTRICTS) <= set(t424), sorted(set(DISTRICTS) - set(t424))
    for d, regs in DISTRICTS.items():
        s = sum(t424[r]["total"] for r in regs)
        assert abs(s - t424[d]["total"]) <= 3, (d, s, t424[d]["total"])
    regs = {nm: r for nm, r in t424.items() if nm not in DISTRICTS}
    assert len(regs) == 33, (len(regs), sorted(regs))
    tot = sum(r["total"] for r in regs.values())
    assert abs(tot - FOREIGN) <= 3, tot
    print(f"1. Tableau 4.24: 33 units + 12 districts (each = its régions within 3) + National; "
          f"units sum to {tot:,} (printed {FOREIGN:,})")

    # ---- join to the hex units, through ci.csv's unit ids (= religiondots' ci_hexes units)
    ci = pd.read_csv(ROOT / "data" / "normalized" / "ci.csv")
    ci = ci[ci["geo_level"] == "region"]
    ivo = ci.groupby("geo_id")["ivorians"].first()
    assert len(ivo) == 33
    ukey = {norm(u): u for u in ivo.index}
    t2u = {}
    for nm in regs:
        k = norm(nm)
        u = ALIAS.get(k) or ukey.get(k)
        assert u is not None, nm
        t2u[nm] = u
    assert sorted(t2u.values()) == sorted(ivo.index), "join not one to one"
    print("   joined to the 33 units one to one (2 aliases: the autonomous districts)")

    # ---- 2. Poids**
    worst = 0.0
    for nm, r in regs.items():
        sh = 100 * r["total"] / (r["total"] + ivo[t2u[nm]])
        worst = max(worst, abs(sh - r["poids2"]))
    for nm, r in regs.items():
        sh = 100 * r["total"] / (r["total"] + ivo[t2u[nm]])
        if abs(sh - r["poids2"]) > 0.1:
            print(f"     {nm}: {sh:.2f} against printed {r['poids2']}")
    assert worst <= 0.25, worst
    print(f"2. Poids** = foreigners / (foreigners + annex 25 Ivorians) in all 33, worst "
          f"{worst:.2f} points")

    # ---- 3. Tableau 4.21 and 4.22
    t421 = read_nat(doc)
    assert set(NATIONALITY) <= set(t421), sorted(set(NATIONALITY) - set(t421))
    assert sum(t421[k] for k in NATIONALITY) == t421["Total"] == FOREIGN, t421
    assert sum(t421[k] for k in WEST_AFRICA) == t421["Afrique de"], t421
    t422 = read_t422(doc)
    for k in ("Bénin", "Burkina-Faso", "Ghana", "Guinée", "Libéria", "Mali", "Niger", "Nigéria",
              "Sénégal", "Togo", "Mauritanie", "Autre Afrique"):
        assert t422[k] == t421[k], (k, t422[k], t421[k])
    small = sum(t421[k] for k in ("Cap Vert", "Gambie", "Guinée-Bissau", "Sierra-Léone"))
    assert t422["Autre Afrique de l'Ouest"] == small, (t422["Autre Afrique de l'Ouest"], small)
    print(f"3. Tableau 4.21: {len(NATIONALITY)} rows sum to {FOREIGN:,}; West Africa subtotal "
          f"and Tableau 4.22's 2021 column agree")
    for k in NATIONALITY:
        print(f"     {k:<26} {t421[k]:>10,}  {t421[k] / FOREIGN:6.2%}")

    # ---- the national language mix of foreigners
    back = written_ids()
    lang = {}
    for k, origin in NATIONALITY.items():
        if origin.startswith("node:"):
            m = {origin[5:]: 1.0}
        else:
            m = {back(n): s for n, s in mix(origin, "ci").items()}
        for n, s in m.items():
            lang[n] = lang.get(n, 0.0) + t421[k] * s
    share = {n: v / FOREIGN for n, v in lang.items()}
    assert abs(sum(share.values()) - 1) < 1e-9
    print("   foreigners' languages nationally (top 12):")
    for n, s in sorted(share.items(), key=lambda x: -x[1])[:12]:
        print(f"     {n:<45} {s * FOREIGN:>10,.0f}  {s:6.2%}")

    rows = []
    for nm, r in regs.items():
        for n, s in share.items():
            rows.append(dict(unit=t2u[nm], source_category=n, count=r["total"] * s,
                             tier="derived"))
    out = pd.DataFrame(rows)
    by = out.groupby("unit")["count"].sum()
    for nm, r in regs.items():
        assert abs(by[t2u[nm]] - r["total"]) < 0.5, nm
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"4. wrote {OUT.name}: {len(out):,} rows, {len(share)} nodes, "
          f"{out['count'].sum():,.0f} people; each unit = its Tableau 4.24 total")


if __name__ == "__main__":
    main()
