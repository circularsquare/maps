"""Côte d'Ivoire: share each région's "other national languages" out into the 82 labels the census
names nationally -> data/normalized/ci_model.csv (every row `modelled`).

    python sources/ci_model.py              (after sources/ci_rgph.py)
    python sources/ci_model.py --calibrate  (also re-runs the parameter search on the 13 named)

WHY (ask 011, Anita 2026-10-05: "share the 18% out by the Glottolog-seeded model"). Tableau 4.20
names 13 languages by région; the other 82 labels (3,850,778 Ivorians, 18%) are printed only
nationally (annex 20). Drawn as one unnamed Niger-Congo remainder they hide Mahou, Tagbana,
Koyaka, Abron, Dida and the rest, though most of them belong to one corner of the country. This
script keeps both things the census does publish exactly:
  * each région's remainder (sources/ci_rgph.py's "Ensemble des autres..." rows), and
  * each label's national count (annex 20),
and fits a labels x régions table between them by iterative proportional fitting (IPF). Only the
starting table (the seed) is borrowed, so the counts per région are modelled, not measured.

THE SEED, per label L and région r, is a mix of two things:
  * LOCAL: the population of r near where L is spoken: sum over r's Kontur hexes of
    pop x exp(-d / K), d the distance to L's nearest anchor. Anchors are Glottolog points
    (data/raw/glottolog/languages.csv, CC BY; every code below was looked up there by name, ISO
    code and country, none from memory) or, where Glottolog has no point, a région named by a
    source (ANCHOR_REGION). Normalised to sum to 1 over the 33 régions.
  * BACKGROUND: where L's speakers' ethnic group lives. Annex 27 gives each language's speakers
    by the census's five ethnic macro-groups (Akan, Krou, Mandé du Nord, Gur, Mandé du Sud, plus
    naturalised and "other"); annex 25 gives each région's Ivorians by the same groups. So
    background(L, r) = sum_g share of L's speakers in g x share of g's Ivorians living in r. This
    is what puts some of every language in Abidjan and in the cocoa south-west, where migrants
    from everywhere live. Normalised to 1.
  seed = (1 - B) x local + B x background. A label with no anchor (21 labels, 247,860 people,
  6% of the remainder: Bambara, Peul, Foula and 18 small or unlocated ones) takes the background.

K AND B were chosen by running the same model on the 12 languages Tableau 4.20 DOES name by
région (all but Dioula, a trade language with no home area), where the answer is known, and
taking the pair that misplaces the fewest speakers on average (--calibrate prints the grid).

WHAT IT CANNOT KNOW. Whether a language's area is really a disc around one point (Glottolog's
points are single; some are off, e.g. Baoulé's sits on the coast, which is why the calibration's
Baoulé error is large); which of a région's remainder languages its migrants speak beyond what
the ethnic groups say; anything inside a région (placement there is by population, as for every
row of Côte d'Ivoire).
"""
import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from ci_rgph import (PDF, P_A25, T420_COLUMNS, NAMED, OTHERS, A25_ALIAS,  # noqa: E402
                     lines_of, read_rows, norm)
from rdlink import RD_GEO  # noqa: E402

NORM = ROOT / "data" / "normalized"
OUT = NORM / "ci_model.csv"
GLOTTO = ROOT / "data" / "raw" / "glottolog" / "languages.csv"
P_A27 = (145, 146, 147)        # 0-based; page 147 repeats ten rows of 146 (asserted equal)
GROUPS = ["Akan", "Krou", "Mandé du Nord", "Gur", "Mandé du Sud", "Naturalisé", "Autre"]
REMAINDER_TOTAL = 3_850_778

K_KM = 10.0                    # chosen by --calibrate (see the docstring and sources/ci.md)
B = 0.5

# annex 20 label (normalised) -> Glottolog ids whose points anchor it. Where the census name is
# not Glottolog's: Ahizi = the three Aizi lects; Appolo ou N'zima = Nzima; Ehotilé = Beti (Côte
# d'Ivoire), ISO eot; Gagou and Gban are both Gban (gagu1242, Glottolog's alternative name Gagu);
# Gouin = Cerma; Komono = Khisa (ISO kqm); Lohron = Téén (ISO lor); Mona ou Mouan = Mwan; Ouan =
# Wan; Ngain = Ngen of Djonkro and Beng (the same people, Bonguéra in Iffou); Guébié = Gabogbo
# (ISO gie); Kla = Kla-Dan; Niédéboua = Nyedebwa (a Nyabwa dialect); Oubi = Glio-Oubi; Kroumen =
# the three Krumen lects; Dida = Yocoboué and Lakota Dida; Tchebara, Fodonon and Koufoulo =
# Glottolog's Tyebara, Fodonon and Kufuru dialects of Cebaara Senoufo (all on Cebaara's point);
# Baralaka and Finanga = Mahou dialects, Karandjan a Worodougou dialect, Nigbi a Koyaga dialect,
# Odienneka a Wojenaka dialect (each on its parent's point); Djamala is the Djimini people's
# Mandé-speaking neighbours at Dabakala (M'Brah and Koffi; academia.edu "Etudes historiques des
# peuplements Djimini et Djamala"), anchored on Djimini's point.
GLOTTOCODE = {
    "abidji": ["abid1235"], "aboure": ["abur1243"], "abron": ["abro1238"],
    "adjoukrou": ["adio1239"], "ahizi": ["apro1235", "mobu1235", "tiag1235"],
    "alladian": ["alla1248"], "appoloounzima": ["nzim1238"], "ebrie": ["ebri1238"], "avikamoubrignan": ["avik1243"],
    "ega": ["egaa1242"], "ehotile": ["beti1248"], "essouma": ["esum1241"],
    "krobou": ["krob1245"], "mbattoougoua": ["mbat1247"],
    "bakwe": ["bakw1243"], "dida": ["yoco1235", "lako1244"], "godie": ["godi1239"],
    "kodia": ["kodi1246"], "kouya": ["kouy1238"],
    "kroumen": ["pyek1235", "tepo1239", "plap1239"], "neyo": ["neyo1238"],
    "gnabouaouniaboua": ["nyab1255"], "niedeboua": ["nyed1238"], "oubi": ["glio1241"],
    "wane": ["wane1242"], "wobe": ["weno1238"], "guebie": ["gabo1234"],
    "mahoukaoumahou": ["maho1249"], "koyakaoukoyara": ["koya1253"],
    "worodougouka": ["woro1256"], "toura": ["tour1242"], "yohoureouyaoure": ["yaou1238"],
    "gagou": ["gagu1242"], "gban": ["gagu1242"], "monaoumouan": ["mwan1250"],
    "ouan": ["wann1242"], "ngain": ["ngen1256", "beng1286"], "gbin": ["gbin1239"],
    "kla": ["klad1234"], "koro": ["koro1306"], "djamala": ["djim1235"],
    "baralaka": ["bara1364"], "finanga": ["fina1241"], "karandjan": ["kara1485"],
    "nigbi": ["nigb1238"], "odienneka": ["odie1238"],
    "tagbana": ["tagw1240"], "djimini": ["djim1235"], "niarafolo": ["nyar1245"],
    "nafana": ["nafa1258"], "palaka": ["pala1342"], "tchebara": ["tyeb1238"],
    "fodonon": ["foda1238"], "koufoulo": ["kufu1238"], "lohron": ["teen1242"],
    "birifor": ["malb1235", "sout2790"], "degha": ["degg1238"], "komono": ["khis1238"],
    "gouinoukirma": ["cerm1238"], "siti": ["siti1241"],
}
# no Glottolog point: a région a source names. Andoh (Ano): "in the department of Prikro, Iffou"
# (fr.wikipedia "Ano (peuple)"; abidjan.net on Prikro's Anôh).
ANCHOR_REGION = {"andoh": ["Iffou"]}
# background only, said in sources/ci.md: migrant languages (Bambara, Peul, Foula: Glottolog's
# points are in Mali and Burkina Faso), labels no source located (Gandjé, Sokyia, Souaminlin,
# Gbonzron, Mangoro, Winnin, N'garadougouka, Komara, Ouadougou, Ouodougou, Kotrohou, Kouzié,
# Sia, Conja, Doma, Samogho), naturalised citizens and "à préciser".
BACKGROUND_ONLY = {
    "bambara", "peul", "foula", "gandje", "sokyia", "souaminlin", "gbonzron", "mangoro",
    "winnin", "ngaradougouka", "komaraoukamara", "ouadougou", "ouodougou", "kotrohou", "kouzie",
    "sia", "conja", "doma", "samogho", "naturalise", "autrelanguenationaleapreciser",
}
# the 12 named languages used to choose K and B (Dioula left out: a trade language)
CALIB = {
    "Baoulé": ["baou1238"], "Senoufo": ["ceba1235", "sena1262", "nyar1245"],
    "Malinké ou Malinka": ["fore1268", "kony1250", "woje1238"],
    "Agni": ["anyi1245", "anyi1244"], "Yacouba ou Dan": ["dann1241"],
    "Bété": ["gagn1235", "guib1246", "dalo1238"], "Akyé ou Attié": ["atti1239"],
    "Lobi": ["lobi1245"], "Gouro": ["guro1248"], "Abbey": ["abee1242"],
    "Koulango": ["bond1246", "boun1243"], "Guéré": ["weso1238", "wewe1238"],
}
CALIB_A20 = {"Baoulé": "baoule", "Senoufo": "senoufo", "Malinké ou Malinka": "malinkeoumaninka",
             "Agni": "agni", "Yacouba ou Dan": "yacoubaoudan", "Bété": "bete",
             "Akyé ou Attié": "akyeouattie", "Lobi": "lobi", "Gouro": "gouro",
             "Abbey": "abbey", "Koulango": "koulango", "Guéré": "guere"}
A27_ALIAS = {"autrelangueapreciser": "autrelanguenationaleapreciser"}


def ipf(seed, rows, cols, iters=3000, tol=1e-7):
    x = seed.astype(float).copy()
    for _ in range(iters):
        rs = x.sum(1)
        x *= np.where(rs > 0, rows / np.where(rs > 0, rs, 1), 0)[:, None]
        cs = x.sum(0)
        x *= np.where(cs > 0, cols / np.where(cs > 0, cs, 1), 0)[None, :]
        if np.abs(x.sum(1) - rows).max() < tol * rows.max():
            break
    return x


def parses(line):
    """Every way to read one PDF line as numbers: a head of digits (the PDF sometimes drops the space), then 3-digit groups."""
    toks = line.split()
    out = []

    def rec(i, acc):
        if i == len(toks):
            out.append(acc)
            return
        if not (1 <= len(toks[i]) <= 7 and toks[i].isdigit()):
            return
        j = i + 1
        while True:
            rec(j, acc + [int("".join(toks[i:j]))])
            if j < len(toks) and len(toks[j]) == 3 and toks[j].isdigit():
                j += 1
            else:
                break
    rec(0, [])
    return out


def read_a27(doc, nat):
    """Annex 27: language x ethnic macro-group. Returns {label: [7 group counts]}."""
    rows, label, nums = [], [], []
    for pno in P_A27:
        for ln in lines_of(doc, pno):
            s = ln.strip()
            if s and all(c.isdigit() or c == " " for c in s):
                nums.append(s)
                continue
            if nums:
                rows.append((" ".join(label), nums))
                label, nums = [], []
            label.append(s)
            if norm(" ".join(label)) not in nat and norm(s) in nat:
                label = [s]
        if nums:
            rows.append((" ".join(label), nums))
            label, nums = [], []
    out = {}
    for lab, lines in rows:
        k = norm(lab)
        k = A27_ALIAS.get(k, k)
        if k not in nat:
            # a label line glued to header text: take the longest suffix that is a label
            words = lab.split()
            hits = [norm(" ".join(words[i:])) for i in range(len(words))]
            hits = [A27_ALIAS.get(h, h) for h in hits if A27_ALIAS.get(h, h) in nat]
            if not hits:
                continue
            k = hits[0]
        found = set()
        for combo in itertools.product(*[parses(x) for x in lines]):
            v = [n for part in combo for n in part]
            if len(v) >= 7 and abs(sum(v[:7]) - nat[k]) <= 3:
                found.add(tuple(v[:7]))
        if len(found) != 1:
            raise SystemExit(f"annex 27 row {lab!r}: {len(found)} readings of {lines}")
        v = list(found.pop())
        if k in out:
            assert out[k] == v, ("annex 27 repeats a row differently", k, out[k], v)
        out[k] = v
    return out


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


class Geo:
    def __init__(self, units):
        import geopandas as gpd
        h = gpd.read_file(RD_GEO / "ci" / "ci_hexes.gpkg", columns=["unit", "pop"])
        h = h[h["pop"] > 0]
        c = h.geometry.representative_point()
        self.lon, self.lat = c.x.to_numpy(), c.y.to_numpy()
        self.pop = h["pop"].to_numpy(float)
        uidx = {u: i for i, u in enumerate(units)}
        self.u = h["unit"].map(uidx).to_numpy()
        assert not np.isnan(self.u.astype(float)).any(), "a hex unit outside the 33"
        self.u = self.u.astype(int)
        self.n = len(units)
        self.units = units

    def local(self, pts, k_km):
        d = np.full(len(self.pop), np.inf)
        for la, lo in pts:
            d = np.minimum(d, hav(la, lo, self.lat, self.lon))
        w = np.bincount(self.u, weights=self.pop * np.exp(-d / k_km), minlength=self.n)
        return w / w.sum()

    def region(self, names):
        w = np.array([1.0 if u in names else 0.0 for u in self.units])
        return w / w.sum()


def model(labels, rows, cols, local, back, k_unused=None, b=B):
    seed = np.zeros((len(labels), len(cols)))
    for i, lab in enumerate(labels):
        bg = back[lab]
        seed[i] = bg if local.get(lab) is None else (1 - b) * local[lab] + b * bg
    seed += 1e-9
    x = ipf(seed, rows, cols)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    import fitz

    df = pd.read_csv(NORM / "ci.csv")
    reg = df[df["geo_level"] == "region"]
    natdf = df[df["geo_level"] == "national"]
    nat = {norm(r.source_category): int(r["count"]) for _, r in natdf.iterrows()}
    label_of = {norm(r.source_category): r.source_category for _, r in natdf.iterrows()}
    named_keys = set(CALIB_A20.values()) | {"dioula", "aucunelanguenationaleparlee"}
    rest = [k for k in nat if k not in named_keys]
    assert len(rest) == 82 and sum(nat[k] for k in rest) == REMAINDER_TOTAL, \
        (len(rest), sum(nat[k] for k in rest))
    anchored = set(GLOTTOCODE) | set(ANCHOR_REGION)
    assert anchored.isdisjoint(BACKGROUND_ONLY)
    assert anchored | BACKGROUND_ONLY == set(rest), sorted(set(rest) ^ (anchored | BACKGROUND_ONLY))
    print(f"1. 82 remainder labels, {REMAINDER_TOTAL:,} people; {len(GLOTTOCODE)} on Glottolog "
          f"points, {len(ANCHOR_REGION)} on a région, {len(BACKGROUND_ONLY)} background only "
          f"({sum(nat[k] for k in BACKGROUND_ONLY):,} people)")

    units = sorted(reg["geo_id"].unique())
    assert len(units) == 33
    rem = reg[reg["source_category"] == OTHERS].set_index("geo_id")["count"].reindex(units)
    assert abs(rem.sum() - REMAINDER_TOTAL) <= 40, rem.sum()
    cols = rem.to_numpy(float) * REMAINDER_TOTAL / rem.sum()

    doc = fitz.open(PDF)
    a27 = read_a27(doc, nat)
    missing = sorted(set(nat) - set(a27))
    assert not missing, missing
    print(f"2. annex 27: all {len(a27)} labels read, each row's groups sum to its annex 20 count "
          f"within 3")

    # annex 25: région x group, joined as ci_rgph.py does
    ukey = {norm(u): u for u in units}
    a25 = {}
    for nm, v in read_rows(doc, P_A25, 8):
        k = norm(nm)
        hits = sorted({uk for uk in ukey if k.endswith(uk)}
                      | {u for ak, u in A25_ALIAS.items() if k.endswith(ak)})
        if len(hits) == 1:
            a25[ukey[hits[0]]] = v[:7]
    assert len(a25) == 33
    g = np.array([a25[u] for u in units], float)            # régions x groups
    gshare = g / g.sum(0)                                     # where each group lives
    print(f"3. annex 25: 33 régions x {len(GROUPS)} groups, {g.sum():,.0f} Ivorians")

    gl = pd.read_csv(GLOTTO, dtype=str, keep_default_na=False).set_index("ID")

    def pts(codes):
        return [(float(gl.at[c, "Latitude"]), float(gl.at[c, "Longitude"])) for c in codes]

    geo = Geo(units)

    def background(k):
        v = np.array(a27[k], float)
        return gshare @ (v / v.sum())

    # ---- calibration on the named languages, where the regional answer is known
    if a.calibrate:
        clabs = list(CALIB)
        truth = np.array([[reg[(reg.geo_id == u) & (reg.source_category == c)]["count"].sum()
                           for u in units] for c in clabs], float)
        crow, ccol = truth.sum(1), truth.sum(0)
        back = {c: background(CALIB_A20[c]) for c in clabs}
        print("\ncalibration: mean share of each named language's speakers put in the wrong "
              "région (unweighted over 12 languages; people-weighted in brackets)")
        best = None
        for k_km in (10, 20, 40, 80, 160):
            local = {c: geo.local(pts(CALIB[c]), k_km) for c in clabs}
            line = []
            for b in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                x = model(clabs, crow, ccol, local, back, b=b)
                err = 0.5 * np.abs(x - truth).sum(1) / crow
                wtd = 0.5 * np.abs(x - truth).sum() / crow.sum()
                line.append(f"B={b:.1f} {100 * err.mean():4.1f}% ({100 * wtd:4.1f}%)")
                if best is None or err.mean() < best[0]:
                    best = (err.mean(), k_km, b, err)
            print(f"  K={k_km:>3} km  " + "  ".join(line))
        print(f"  best: K={best[1]} km, B={best[2]}: "
              + ", ".join(f"{c.split()[0]} {100 * e:.0f}%" for c, e in zip(clabs, best[3])))
        print(f"  in use: K={K_KM:.0f} km, B={B}")
        local = {c: geo.local(pts(CALIB[c]), K_KM) for c in clabs}
        x = model(clabs, crow, ccol, local, back, b=B)
        ab = units.index("District D'Abidjan")
        print("  at the values in use, Abidjan (modelled / census) and the share of each "
              "language's error that is Abidjan:")
        for i, c in enumerate(clabs):
            err = np.abs(x[i] - truth[i])
            print(f"    {c:20s} {x[i, ab]:>9,.0f} / {truth[i, ab]:>9,.0f}   "
                  f"{100 * err[ab] / err.sum():.0f}% of the error")

    # ---- the model
    labels = rest
    rows = np.array([nat[k] for k in labels], float)
    local = {}
    for k in labels:
        if k in GLOTTOCODE:
            local[k] = geo.local(pts(GLOTTOCODE[k]), K_KM)
        elif k in ANCHOR_REGION:
            local[k] = geo.region(ANCHOR_REGION[k])
        else:
            local[k] = None
    back = {k: background(k) for k in labels}
    x = model(labels, rows, cols, local, back, b=B)
    assert np.allclose(x.sum(1), rows, rtol=1e-5), np.abs(x.sum(1) - rows).max()
    assert np.allclose(x.sum(0), cols, rtol=1e-5), np.abs(x.sum(0) - cols).max()
    print(f"4. fitted: every label's national count and every région's remainder met "
          f"(worst {np.abs(x.sum(1) - rows).max():.2f} and {np.abs(x.sum(0) - cols).max():.2f})")

    out = []
    for i, k in enumerate(labels):
        for j, u in enumerate(units):
            if x[i, j] >= 0.5:
                out.append((u, label_of[k], round(float(x[i, j]), 1), "modelled"))
    res = pd.DataFrame(out, columns=["unit", "source_category", "count", "tier"])
    assert abs(res["count"].sum() - REMAINDER_TOTAL) < 100, res["count"].sum()
    res.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {OUT}: {len(res):,} rows, {res['count'].sum():,.0f} people")

    # ---- report
    m = pd.DataFrame(x, index=[label_of[k] for k in labels], columns=units)
    print("\nlargest remainder languages by région (share of the région's remainder):")
    for u in sorted(units, key=lambda u: -cols[units.index(u)]):
        col = m[u].sort_values(ascending=False)
        tot = col.sum()
        print(f"  {u:26s} {tot:>9,.0f}  " + ", ".join(f"{n.title()} {100 * v / tot:.0f}%"
                                                  for n, v in col.head(4).items()))
    print("\nwhere the big remainder languages land (share of speakers):")
    for n in m.sum(1).sort_values(ascending=False).head(12).index:
        row = m.loc[n] / m.loc[n].sum()
        print(f"  {n.title():24s} " + ", ".join(f"{u} {100 * v:.0f}%" for u, v in
                                                row.sort_values(ascending=False).head(4).items()))


if __name__ == "__main__":
    main()
