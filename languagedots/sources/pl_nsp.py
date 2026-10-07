"""Poland, Narodowy Spis Powszechny 2021: language used at home by gmina -> data/normalized/pl.csv.

    python sources/pl_nsp.py [--fetch]

THE QUESTION. "Język używany w domu", the language(s) a person usually speaks at home. Polish
and/or up to two languages other than Polish, so one person may name one, two or three
languages. Self-enumeration online was the main mode (the alphabetical head of the list,
Abkhaz 446, Acholi 576, Afar 541, Adyghe 836, reads like the top of a drop-down; sources/pl.md).

SOURCE. GUS, "Język używany w domu - dane NSP 2021 dla kraju i jednostek podziału
terytorialnego" (one xlsx; TABL.1/2 national, 3 voivodeship, 4 powiat, 5 gmina), and the
summary annex "wyniki_ostateczne_nsp2021_narodowsc_jezyk_wyznanie_2023_11_29.xlsx", whose
Tab3_JDom gives the national cross-table by NUMBER of languages (one / more than one / of which
with Polish). stat.gov.pl serves an incomplete certificate chain (religiondots/sources/pl.md
§1), so --fetch turns verification off for that host and checks the files structurally.

WHAT EACH GMINA ROW HOLDS. T = Ogółem (persons), P = Polski (persons naming Polish), O = Inny
niż polski (persons naming at least one other language), N = Nieustalony (no language
recorded), then the other languages one by one as MENTIONS. A cell under 10 is not printed at
gmina level (under 3 at powiat and voivodeship level); O is "<10" in two gminas.

STEP 1, SUPPRESSED CELLS (derived). Per language, what a level lacks against its parent goes to
the parent's units that print no figure for that language, by their population T, capped at
the largest value a hidden cell can hold (2 at voivodeship and powiat, 9 at gmina): national
-> voivodeship -> powiat -> gmina. Every allocation fits under its caps (asserted). This puts
85,019 of 1,903,511 non-Polish mentions (4.5%) on gminas that did not print them, and is what
places the 246 languages no gmina prints near where they were counted.

STEP 2, SHARING EACH PERSON (spec §3.6). Persons, not mentions, are drawn. Per gmina:
  B  = P + O - (T - N)      persons naming Polish AND another language (inclusion-exclusion;
                            nationally 1,611,784, exactly Tab3's "więcej niż jednego, w tym
                            polskiego")
  B3 = RHO * B              of those, persons who named Polish and TWO others; RHO = 144,537 /
                            1,611,784 nationally (Tab3: 156,608 persons named two non-Polish
                            languages, 12,071 of them without Polish), no gmina figure exists
  Polish            = P - B/2 - B3/6        (a two-language person gives 1/2 to Polish, a
                                             three-language person 1/3)
  non-Polish total  = O - B/2 + B3/6        (everything else; sums to T - N with Polish)
  each language     = non-Polish total * its mentions / the gmina's non-Polish mentions
So Polish and the non-Polish total are the census's own combinations (exact up to B3), and only
the split between non-Polish languages inside a gmina is the mention scaling. Nationally this
gives Polish 37,038,636.5, which is Tab3's exact figure (36,256,834 + 1,467,247/2 +
144,537/3). Every row is tier `derived`.

CHECKS (all must pass, or nothing is written):
  1. each level sums to the national T, P, O, N (O allowing the two "<10" gminas)
  2. the four levels agree per language after step 1, and step 1 fits its caps
  3. the summary annex (a second release of the same census) agrees with TABL.2/3 per language
     per voivodeship, and its Tab3 B equals P + O - (T - N) nationally
  4. per gmina 0 <= B <= min(P, O); every gmina's persons sum to T - N; the country to
     38,003,737 (38,036,118 less 32,381 with no language recorded)
"""
import argparse
import math
import os
import ssl
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", module="openpyxl")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "pl"
OUT = HERE / "data" / "normalized" / "pl.csv"
MAIN = RAW / "jezyk_uzywany_w_domu_nsp2021.xlsx"
ANNEX = RAW / "wyniki_ostateczne_nar_jezyk_wyznanie_2023_11_29.xlsx"
BASE = "https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/6536/10/1/1/"
URLS = {
    MAIN: BASE + "jezyk_uzywany_w_domu_-_dane_nsp_2021_dla_kraju_i_jednostek_podzialu_terytorialnego.xlsx",
    ANNEX: BASE + "wyniki_ostateczne_nsp2021_narodowsc_jezyk_wyznanie_2023_11_29.xlsx",
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

TOTAL, POLISH, OTHER, NS = "Ogółem", "Polski", "Inny niż polski", "Nieustalony"
HEAD = (TOTAL, POLISH, OTHER, NS)
NATIONAL = {TOTAL: 38_036_118, POLISH: 37_868_618, OTHER: 1_746_903, NS: 32_381}
CAP = {"voivodeship": 2, "powiat": 2, "gmina": 9}    # largest value a hidden cell can hold


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False            # stat.gov.pl omits its intermediate certificate
    ctx.verify_mode = ssl.CERT_NONE
    for path, url in URLS.items():
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), context=ctx,
                                      timeout=300).read()
        if data[:2] != b"PK" or len(data) < 50_000:
            raise SystemExit(f"{path.name}: not an xlsx ({len(data):,} bytes)")
        path.write_bytes(data)
        print(f"wrote {path} ({len(data):,} bytes)")


def label(s):
    return str(s).replace("\n", " ").replace("w tym:", "").strip()


def read_level(wb, sheet):
    """Rows (code, name, label, n). The code is the last cell before the label; a unit's first
    row carries its names before the code, later rows only the code."""
    rows, name = [], None
    for i, row in enumerate(wb[sheet].iter_rows(values_only=True)):
        if i < 4:
            continue
        vals = [c for c in row if c is not None]
        if len(vals) < 4:
            continue
        if label(vals[-3]) == TOTAL:
            name = str(vals[-5]).strip()
        rows.append((str(vals[-4]).strip(), name, label(vals[-3]), vals[-2]))
    return pd.DataFrame(rows, columns=["code", "name", "label", "n"])


def read_national(wb):
    rows = []
    for i, row in enumerate(wb["TABL. 2"].iter_rows(values_only=True)):
        vals = [c for c in row if c is not None]
        if i >= 4 and len(vals) >= 3 and isinstance(vals[1], (int, float)):
            rows.append(("PL", "Polska", label(vals[0]), vals[1]))
    return pd.DataFrame(rows, columns=["code", "name", "label", "n"])


def allocate(residual, weights, cap):
    """Water-fill `residual` over units by `weights`, no unit above `cap`."""
    out = pd.Series(0.0, index=weights.index)
    left, free = float(residual), weights.copy()
    for _ in range(100):
        if left <= 1e-9 or free.empty:
            break
        w = free / free.sum() if free.sum() > 0 else pd.Series(1.0 / len(free), index=free.index)
        add = (w * left).clip(upper=cap - out[free.index])
        out[free.index] += add
        left -= add.sum()
        free = free[out[free.index] < cap - 1e-9]
    if left > 1e-6:
        raise SystemExit(f"allocation does not fit: {left:.1f} left over (cap {cap})")
    # Whole people: carry the fractions along the units in TERYT code order (neighbours in a
    # powiat sit near each other in it) and give a person where the running total crosses one,
    # never a share of a person spread thinly over every unit. A fraction never rounds past cap.
    out = out.sort_index()
    cum = out.cumsum().round(9)
    whole = (cum.apply(math.floor) - cum.shift(fill_value=0).apply(math.floor)).astype(float)
    assert whole.sum() == round(residual) and (whole <= cap).all()
    return whole


def fill(parent, child, child_parent, level):
    """Give each child level the mentions its parent has and it lacks, per language."""
    pop = child[child.label == TOTAL].set_index("code").n.astype(float)
    langs = child[~child.label.isin(HEAD)]
    have = langs.set_index(["code", "label"]).n.astype(float)
    psum = langs.assign(par=langs.code.map(child_parent)).groupby(["par", "label"]).n.sum()
    plangs = parent[~parent.label.isin(HEAD)].set_index(["code", "label"]).n.astype(float)
    kids = pd.Series(list(child_parent.keys()), index=list(child_parent.values()))
    rows, moved = [], 0.0
    for (pc, lab), pn in plangs.items():
        res = pn - psum.get((pc, lab), 0.0)
        if res < -1e-6:
            raise SystemExit(f"{level}: {lab} in {pc} sums above its parent by {-res}")
        if res <= 1e-9:
            continue
        units = kids.loc[[pc]] if pc in kids.index else pd.Series([], dtype=str)
        lacking = [u for u in units if (u, lab) not in have.index]
        got = allocate(res, pop.loc[lacking], CAP[level])
        rows += [(u, lab, v) for u, v in got.items() if v > 0]
        moved += res
    print(f"  {level}: {moved:,.0f} mentions allocated to unprinted cells")
    return pd.DataFrame(rows, columns=["code", "label", "n"])


def main(do_fetch):
    if do_fetch or not MAIN.exists():
        fetch()
    for p in (MAIN, ANNEX):
        if not zipfile.is_zipfile(p):
            raise SystemExit(f"{p} is not an xlsx")
    import openpyxl
    wb = openpyxl.load_workbook(MAIN, read_only=True)
    nat, voi, pw, gm = (read_national(wb), read_level(wb, "TABL. 3"), read_level(wb, "TABL. 4"),
                        read_level(wb, "TABL. 5"))
    assert len(nat) == 352, len(nat)

    # ---- check 1: every level sums to the national heads ----
    lt10 = gm.n.astype(str).str.strip() == "<10"
    print(f"gmina cells printed '<10': {int(lt10.sum())} ({sorted(gm.loc[lt10, 'label'].unique())})")
    assert set(gm.loc[lt10, "label"]) == {OTHER} and lt10.sum() == 2
    for name, d, k in (("country", nat, 1), ("voivodeship", voi, 16), ("powiat", pw, 380),
                       ("gmina", gm, 2477)):
        assert d.code.nunique() == k, (name, d.code.nunique())
        num = pd.to_numeric(d.n, errors="coerce").fillna(0)
        for h, v in NATIONAL.items():
            s = num[d.label == h].sum()
            ok = s == v or (name == "gmina" and h == OTHER and v - 18 <= s < v)
            assert ok, (name, h, s, v)
        print(f"  {name:12s} {k:5d} units  T {num[d.label == TOTAL].sum():,.0f}  "
              f"other-language mentions {num[~d.label.isin(HEAD)].sum():,.0f}")
    nat_m = nat[~nat.label.isin(HEAD)].n.sum()
    for d in (voi, pw, gm):
        d["n"] = pd.to_numeric(d.n, errors="coerce")
    o_lt10 = gm.loc[lt10, "code"].tolist()

    # ---- check 3: the summary annex, a second release of the same tables ----
    wa = openpyxl.load_workbook(ANNEX, read_only=True)
    t3 = [[c for c in r if c is not None] for r in wa["Tab3_JDom"].iter_rows(values_only=True)]
    tot = next(r for r in t3 if r and r[0] == TOTAL)
    one, multi, multi_pl, ns = tot[2], tot[3], tot[4], tot[5]
    assert ns == NATIONAL[NS] and one + multi + ns == NATIONAL[TOTAL]
    b_nat = NATIONAL[POLISH] + NATIONAL[OTHER] - (NATIONAL[TOTAL] - NATIONAL[NS])
    assert b_nat == multi_pl == 1_611_784, (b_nat, multi_pl)
    two_other = nat_m - NATIONAL[OTHER]                     # persons naming two non-Polish
    rho = (two_other - (multi - multi_pl)) / multi_pl
    print(f"annex Tab3: one language {one:,}, more than one {multi:,} ({multi_pl:,} with Polish); "
          f"two non-Polish {two_other:,}, of them with Polish {two_other - (multi - multi_pl):,}; "
          f"RHO {rho:.5f}")
    t4 = [[c for c in r if c is not None] for r in wa["Tab4_JDom_woj"].iter_rows(values_only=True)]
    hdr = next(i for i, r in enumerate(t4) if r and r[0] == TOTAL)
    vnames = [c for c in t4[hdr - 2] if c][:16]
    vcodes = voi[voi.label == TOTAL].sort_values("code")
    vmap = {}                                  # annex column order is TABL.3's code order
    for j, (code, nm) in enumerate(zip(vcodes.code, vcodes.name)):
        assert nm.lower() == vnames[j].lower(), (nm, vnames[j])
        vmap[j] = code
    vi = voi.set_index(["code", "label"]).n
    nat_i = nat.set_index("label").n
    bad = 0
    rows4 = []
    for r in t4[hdr:]:
        if r and isinstance(r[0], str) and r[0].strip().startswith("w procentach"):
            break                          # the same table again, in percent
        if r and r[0] is not None and isinstance(r[1], (int, float)):
            rows4.append(r)
    # The annex prints voivodeship cells of 1 and 2 that TABL.3 hides, so where TABL.3 has no
    # figure the annex's (<= 2) is taken as measured. Its "inne" row is its own remainder of
    # every language it does not list, not a TABL.2 label.
    unhidden, agree = [], 0
    for r in rows4:
        lab = label(r[0])
        if lab == "inne":
            continue
        if lab not in nat_i.index or nat_i[lab] != r[1]:
            bad += 1
            print(f"  !! annex national {lab}: {r[1]} vs TABL.2 {nat_i.get(lab)}")
        for j in range(16):
            v = r[2 + j]
            v = 0 if v in (None, "-", "–") else v
            mine = vi.get((vmap[j], lab), 0)
            mine = 0 if pd.isna(mine) else mine
            if not isinstance(v, (int, float)) or v == mine:
                agree += 1
            elif mine == 0 and 0 < v <= CAP["voivodeship"]:
                unhidden.append((vmap[j], lab, float(v)))
            else:
                bad += 1
                print(f"  !! annex {vmap[j]} {lab}: {v} vs TABL.3 {mine}")
    assert bad == 0
    print(f"annex Tab4 agrees with TABL.2/TABL.3 in all {agree} cells TABL.3 prints, for "
          f"{len(rows4) - 1} labels; it shows {len(unhidden)} cells TABL.3 hides "
          f"({sum(v for *_, v in unhidden):.0f} mentions), used as printed")
    voi = pd.concat([voi, pd.DataFrame(unhidden, columns=["code", "label", "n"]).assign(name=None)],
                    ignore_index=True)

    # ---- step 1: suppressed cells, national -> voivodeship -> powiat -> gmina ----
    print("step 1, unprinted cells")
    add_v = fill(nat, voi, {c: "PL" for c in voi.code.unique()}, "voivodeship")
    voi = pd.concat([voi, add_v.assign(name=None)], ignore_index=True)
    add_p = fill(voi, pw, {c: c[:2] for c in pw.code.unique()}, "powiat")
    pw = pd.concat([pw, add_p.assign(name=None)], ignore_index=True)
    add_g = fill(pw, gm, {c: c[:4] for c in gm.code.unique()}, "gmina")
    printed = gm.assign(src="printed")
    gm = pd.concat([printed, add_g.assign(name=None, src="allocated")], ignore_index=True)
    # check 2: every level now holds the national mentions per language
    for name, d in (("voivodeship", voi), ("powiat", pw), ("gmina", gm)):
        s = d[~d.label.isin(HEAD)].groupby("label").n.sum()
        diff = (s - nat[~nat.label.isin(HEAD)].set_index("label").n).abs().max()
        assert diff < 1e-6, (name, diff)
    print(f"  every level now sums to TABL.2 per language ({nat_m:,} mentions)")

    # ---- step 2: persons ----
    names = gm[gm.label == TOTAL].set_index("code").name
    w = gm.pivot_table(index="code", columns="label", values="n", aggfunc="sum")
    T, P, N = w[TOTAL], w[POLISH].fillna(0), w[NS].fillna(0)
    langs = w.drop(columns=list(HEAD)).fillna(0)
    M = langs.sum(axis=1)
    O = w[OTHER].copy()
    # The two "<10" cells hold what the national O lacks against the printed gminas (9 people):
    # each gets its floor (B >= 0, at least 1), the rest shared by its mentions.
    left = NATIONAL[OTHER] - O.drop(o_lt10).sum()
    floor = {c: max(T[c] - N[c] - P[c], 1.0) for c in o_lt10}
    spare = left - sum(floor.values())
    assert 0 <= spare and left <= 9 * len(o_lt10), (left, floor)
    for c in o_lt10:
        O[c] = floor[c] + spare * M[c] / sum(M[k] for k in o_lt10)
        print(f"  gmina {c} {names[c]}: O '<10' taken as {O[c]:.2f} of the {left:.0f} the two "
              f"hold (mentions {M[c]:.1f})")
    B = P + O - (T - N)
    assert (B >= -1e-9).all() and (B <= O + 1e-9).all() and (B <= P + 1e-9).all()
    B3 = rho * B
    pol = P - B / 2 - B3 / 6
    other = O - B / 2 + B3 / 6
    nolang = (M <= 0) & (other > 0)
    assert not nolang.any(), f"gminas with O > 0 and no mentions: {list(M[nolang].index)}"
    print(f"  mentions below O in {(M < O).sum()} gminas (unprinted cells under-allocated; "
          f"scaling absorbs it); median mentions/O {(M / O).median():.3f}")
    share = langs.mul((other / M.where(M > 0, 1)), axis=0)

    rows = []
    for c in w.index:
        rows.append((c, names[c], POLISH, pol[c], P[c]))
        for lab, v in share.loc[c][share.loc[c] > 0].items():
            rows.append((c, names[c], lab, v, langs.at[c, lab]))
        if N[c] > 0:
            rows.append((c, names[c], NS, N[c], N[c]))
    df = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count", "mentions"])
    df["geo_level"] = "gmina"
    df["tier"] = "derived"
    df.loc[df.source_category == NS, "tier"] = "measured"
    alloc = add_g.set_index(["code", "label"]).n
    df["mentions_allocated"] = [alloc.get((g, l), 0.0) for g, l in zip(df.geo_id, df.source_category)]

    # ---- check 4 ----
    per = df[df.source_category != NS].groupby("geo_id")["count"].sum()
    gap = (per - (T - N)).abs().max()
    assert gap < 1e-6, gap
    tot_drawn = per.sum()
    assert abs(tot_drawn - (NATIONAL[TOTAL] - NATIONAL[NS])) < 1e-3, tot_drawn
    exact_pl = one - (NATIONAL[OTHER] - multi) + (multi_pl - rho * multi_pl) / 2 + rho * multi_pl / 3
    print(f"persons drawn {tot_drawn:,.1f}; Polish {pol.sum():,.1f} (from Tab3's combinations "
          f"{exact_pl:,.1f})")
    assert abs(pol.sum() - exact_pl) < 1e-3
    top = df[df.source_category != NS].groupby("source_category")[["count", "mentions"]].sum()
    print(top.sort_values("count", ascending=False).head(15).round(0).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = df[["geo_id", "geo_level", "geo_name", "source_category", "count", "mentions",
             "mentions_allocated", "tier"]]
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT} ({len(df):,} rows, {df.geo_id.nunique():,} gminas, "
          f"{df.source_category.nunique()} categories)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    main(ap.parse_args().fetch)
