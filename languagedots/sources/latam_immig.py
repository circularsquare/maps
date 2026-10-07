"""Immigrant languages for the Latin American maps whose census asked only about indigenous
languages (or only about peoples), with everyone else drawn on the main language (spec §3.5).
One helper since 2026-10-06 (session 5d7dac7e-br merged sources/latam_imm.py into it); the
records are sources/latam_immig.md (ar cl co ve br ec) and sources/mx.md, "Immigrant and settler
languages" §2 (mx cr pa hn ni).

Two ways in, one rule:

    from latam_immig import immigrant_languages, fold_into          # ar cl co ve br ec
    rows = immigrant_languages(df, host=SPANISH, dest="ar", drop_roots=())
        df: unit, iso (ISO 3166 alpha-2 of the birth country), count (foreign-born people)
        -> unit, node, count, kept: the origin's languages after retention; the part that
           speaks the host language at home is returned on `host`
    counts, report = fold_into(base, rows, host, dedupe=(), from_tier=None)

    from latam_immig import spread, unit_rows                        # mx cr pa hn ni
    spread({"US": 120, "CN": 4}, "cr")   # -> {node: people}, summing to 124
    unit_rows(org, "hn")                 # -> rows (unit, node, tier, count) incl. -Spanish

THE RULE. Each origin's languages are origin_mix.mix(iso, dest) (the shared table). The share
already on the host language stays there (Spanish-speaking origins in a Spanish-speaking
country, Portuguese ones in Brazil). The rest is RETAINED at France's TeO2 rate for the origin's
region (sources/fr_build.py TEO2: share with a foreign family language x share of those parents
who use it with their children; Americas 77.4%, Spain-Italy 62.4%, China 66.3%...): no Latin
American survey gives home-language retention by birth country (latam_immig.md §2). The rest goes
onto the host language. The origin's TeO2 region comes from fr_build.teo_region, with Natural
Earth's continent as the fallback block.

drop_roots: where the census asked every person about indigenous languages (mx cr pa hn ni ec),
immigrants who speak one are already drawn, so indigenous-American roots (DROP_ROOTS) are taken
out of every origin's mix and the rest rescaled. spread() and unit_rows() do that, and also keep
Spanish-speaking origins (HISPANIC) on Spanish whole, as the northern pass decided: their census
never asked them anything else, and the host language is theirs.

RECURSION GUARD. Argentina's immigrants take Chile's home mix and Chile's take Argentina's: a
home mix computed while another country's immigrant layer is being built is computed without the
origin's own immigrant layer (`active()`; each country's counts() asks it). The home mixes cut
languages under 1%, which immigrant languages almost never pass, so this changes nothing drawn.

MERGE NOTES (2026-10-06). latam_imm.py classed origins into TeO2 blocks with hand lists; this
module uses Natural Earth continents. They differ only for Armenia, Azerbaijan and Georgia (Asia
here, 70.1% kept, against Europe's 67.5%). France now counts as EU (66.75%) and the Maldives as
Asia here, as latam_imm.py had them.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "taxonomy"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

NE = ROOT.parent / "religiondots" / "data" / "geo" / "ne_10m_admin_0_countries.geojson"
_DEPTH = [0]


def active():
    """True while some country's immigrant layer is being computed: a nested counts() call (an
    origin's home mix) must then skip its own immigrant layer."""
    return _DEPTH[0] > 0


@lru_cache(None)
def _continents():
    import geopandas as gpd
    d = gpd.read_file(NE, ignore_geometry=True)
    out = {}
    for a, b, c in zip(d["ISO_A2"], d["ISO_A2_EH"], d["CONTINENT"]):
        for k in (a, b):
            if isinstance(k, str) and len(k) == 2 and k != "-9":
                out.setdefault(k, c)
    out.update({"MV": "Asia", "SU": "Europe", "YU": "Europe", "QT": "Europe", "HK": "Asia", "MO": "Asia",
                "GF": "South America", "GP": "North America", "MQ": "North America",
                "RE": "Africa", "YT": "Africa", "PS": "Asia", "EH": "Africa", "XK": "Europe",
                "BQ": "North America", "SS": "Africa", "AX": "Europe", "CW": "North America",
                **{k: "Antarctica" for k in ("BV", "GS", "IO", "CX", "CC", "TF", "HM", "UM", "AQ")}})
    return out


def keep_share(iso):
    """Share of an origin's non-host-language speakers who keep the language at home: TeO2's
    ref x kids for the origin's region (fr_build.TEO2, sources/fr.md §2b)."""
    from fr_build import TEO2, teo_region, _EU27
    import origin_mix
    iso = origin_mix.ALIAS.get(iso, iso)
    key = {"GR": "EL", "GB": "UK"}.get(iso, iso)
    cont = _continents().get(iso)
    if cont is None:       # a territory Natural Earth lacks (Svalbard...): a handful of people
        cont = "Europe"
    block = {"Europe": "EU" if key in _EU27 or key == "FR" else "EUR", "Africa": "AFR", "Asia": "ASI",
             "North America": "AME", "South America": "AME", "Oceania": "OCE",
             "Seven seas (open ocean)": "AFR", "Antarctica": "OCE"}[cont]
    ref, kids = TEO2[teo_region(key, block)]
    return ref / 100 * kids / 100


@lru_cache(None)
def _mix(iso, host, dest, drop_roots):
    import origin_mix
    try:
        m = origin_mix.mix(iso, dest)
    except KeyError:
        # origin_mix knows no language for it: Aland is Swedish; the rest are uninhabited
        # territories (Bouvet, South Georgia, Christmas Island...), coding slips in a census
        # list, left on the host language
        m = {"indoeuropean.germanic.north.swedish": 1.0} if iso == "AX" else {host: 1.0}
    if drop_roots:
        m = {n: s for n, s in m.items() if n.split(".")[0] not in drop_roots}
        t = sum(m.values())
        m = {n: s / t for n, s in m.items()} if t else {host: 1.0}
    return m


def origin_languages(iso, host, dest, drop_roots=()):
    """{node: share} before retention for one origin (see immigrant_languages). Computed with
    the recursion guard up, so an origin's home mix never includes its own immigrants."""
    _DEPTH[0] += 1
    try:
        return dict(_mix(iso, host, dest, tuple(sorted(drop_roots))))
    finally:
        _DEPTH[0] -= 1


def immigrant_languages(df, host, dest, drop_roots=()):
    """df (unit, iso, count) -> DataFrame (unit, node, count, kept); `kept` is False for the
    share retention moved onto the host language (and for host-language shares).
    drop_roots: tree roots taken out of every origin's mix (rescaled), for a census that asked
    every person about indigenous languages, so immigrants who speak one are already drawn."""
    import pandas as pd
    mixes = {iso: origin_languages(iso, host, dest, drop_roots) for iso in df["iso"].unique()}
    rows = []
    for r in df.itertuples(index=False):
        m = mixes[r.iso]
        k = keep_share(r.iso)
        for node, s in m.items():
            n = r.count * s
            if node == host:
                rows.append((r.unit, host, n, False))
            else:
                rows.append((r.unit, node, n * k, True))
                rows.append((r.unit, host, n * (1 - k), False))
    out = pd.DataFrame(rows, columns=["unit", "node", "count", "kept"])
    return out.groupby(["unit", "node", "kept"], as_index=False)["count"].sum()


def fold_into(base, imm, host, dedupe=(), from_tier=None):
    """Put the immigrant rows into a country's counts. `base` (unit, node, count, tier) has
    everyone not counted as an indigenous speaker on `host`; the immigrants' non-host languages
    come out of that host row, unit by unit. For a node in `dedupe` that the base already counts
    (Paraguayan Guarani in Argentina, measured among self-identified Guarani), only the part of
    the estimate above the measured count is added: the measured speakers are mostly the same
    immigrants. Never takes a host row below zero (prints how much was capped).
    -> (counts, report dict)"""
    import pandas as pd
    lang = imm[imm["kept"] & (imm["node"] != host)].groupby(["unit", "node"], as_index=False)[
        "count"].sum()
    measured = base.groupby(["unit", "node"])["count"].sum()
    rep = {"estimate": lang["count"].sum(), "deduped": 0.0, "capped": 0.0}
    if dedupe:
        have = [measured.get((u, n), 0.0) if n in dedupe else 0.0
                for u, n in zip(lang["unit"], lang["node"])]
        new = (lang["count"] - pd.Series(have, index=lang.index)).clip(lower=0)
        rep["deduped"] = float((lang["count"] - new).sum())
        lang["count"] = new
    hsel = (base["node"] == host) & ((base["tier"] == from_tier) if from_tier else True)
    host_rows = base[hsel].groupby("unit")["count"].sum()
    take = lang.groupby("unit")["count"].sum()
    over = (take - host_rows.reindex(take.index).fillna(0)).clip(lower=0)
    if over.sum() > 0:
        f = 1 - over / take
        lang["count"] *= lang["unit"].map(f).fillna(1)
        rep["capped"] = float(over.sum())
        take = lang.groupby("unit")["count"].sum()
    b = base.copy()
    b["count"] = b["count"].astype(float)
    hm = hsel.to_numpy()
    # spread a unit's removal over its host rows in proportion (normally one row per unit)
    tot = b[hm].groupby("unit")["count"].transform("sum")
    b.loc[hm, "count"] = b.loc[hm, "count"] - b.loc[hm, "unit"].map(take).fillna(0) * (
        b.loc[hm, "count"] / tot)
    lang["tier"] = "derived"
    out = pd.concat([b, lang[["unit", "node", "count", "tier"]]], ignore_index=True)
    out = out[out["count"] > 0]
    rep["added"] = float(lang["count"].sum())
    top = lang.groupby("node")["count"].sum().sort_values(ascending=False).head(6)
    print(f"  immigrant languages: estimate {rep['estimate']:,.0f}, already counted as "
          f"indigenous speakers {rep['deduped']:,.0f}, capped {rep['capped']:,.0f}, added "
          f"{rep['added']:,.0f} (" + ", ".join(f"{n.split('.')[-1]} {v:,.0f}"
                                              for n, v in top.items()) + ")")
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum(), rep


# ---------------------------------------------------------------------------------------------
# The northern pass's interface (mx cr pa hn ni; formerly sources/latam_imm.py)
# ---------------------------------------------------------------------------------------------
SPANISH = "indoeuropean.romance.spanish"
HISPANIC = {"ES", "MX", "GT", "SV", "HN", "NI", "CR", "PA", "CU", "DO", "PR", "VE", "CO", "EC",
            "PE", "BO", "CL", "AR", "UY", "PY", "GQ"}
# indigenous-American roots of the tree (taxonomy/tree.d/mx.txt, gt.txt, ni.txt, hn.txt, co.txt,
# pe.txt, bo.txt); Garifuna (arawakan.garifuna) is kept, since no census here asked it
DROP_ROOTS = frozenset({"mayan", "otomanguean", "utoaztecan", "mixezoque", "totonacan", "yuman",
                        "chibchan", "misumalpan", "jicaquean", "quechuan", "aymaran", "tupian",
                        "americas_other", "cariban", "tucanoan", "panoan", "araucanian"})


def origin_mix_for(iso, dest):
    """{node: share} before retention: Spanish for HISPANIC, else origin_mix less DROP_ROOTS."""
    if iso in HISPANIC:
        return {SPANISH: 1.0}
    return origin_languages(iso, SPANISH, dest, DROP_ROOTS)


def spread(by_origin, dest):
    """{iso: people} -> {node: people}, with TeO2 retention onto Spanish."""
    out = {}
    for iso, n in by_origin.items():
        if not n:
            continue
        m = origin_mix_for(iso, dest)
        k = keep_share(iso) if iso not in HISPANIC else 1.0
        for node, s in m.items():
            if node == SPANISH:
                out[SPANISH] = out.get(SPANISH, 0.0) + n * s
            else:
                out[node] = out.get(node, 0.0) + n * s * k
                out[SPANISH] = out.get(SPANISH, 0.0) + n * s * (1 - k)
    return out


def unit_rows(org, dest, spanish_tier="derived"):
    """org: DataFrame unit, origin, count (several rows per unit and origin allowed; origins
    "US_U18" and "XX" stay Spanish) -> rows (unit, node, tier, count): the immigrant languages,
    `derived`, and a matching negative Spanish row at `spanish_tier` to take them out."""
    org = org[~org["origin"].isin(["US_U18", "XX"])]
    rows = []
    for u, g in org.groupby("unit"):
        by = g.groupby("origin")["count"].sum().to_dict()
        for node, n in spread(by, dest).items():
            if node != SPANISH and n:
                rows.append((u, node, "derived", n))
                rows.append((u, SPANISH, spanish_tier, -n))
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) == 3 and sys.argv[2].islower():
        iso, dest = sys.argv[1], sys.argv[2]
        print(f"{iso} -> {dest}: keep {keep_share(iso):.1%}")
        for n, v in sorted(spread({iso: 1.0}, dest).items(), key=lambda kv: -kv[1]):
            print(f"  {v:6.1%}  {n}")
    else:
        for iso in sys.argv[1:] or ["PY", "BO", "IT", "CN", "HT", "DE", "SY"]:
            print(iso, f"{keep_share(iso):.3f}")
