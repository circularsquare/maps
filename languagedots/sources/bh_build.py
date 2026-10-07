"""Bahrain, 2020 census: Bahrainis and non-Bahrainis by governorate, read as languages.
-> data/normalized/bh.csv

    python sources/bh_build.py [--fetch]

Nobody in Bahrain is asked a language. Built by the Gulf home-mix method (sources/gulf_mix.py,
sources/gulf.md), every row `derived`:

  1. sources/bh_extract.py (its own process) copies religiondots' parsed census table: 4
     governorates x nationality group x sex (census 2020, data.gov.bh).
  2. Bahrainis: the Shia among them on Baharna Arabic (Glottolog baha1259), everyone else on Gulf
     Arabic. Bahrain's two Arabic dialects split along sect: the Baharna, Shia, speak Bahrani;
     Sunni Bahrainis speak the Gulf dialect (Holes, *Language Variation and Change in a
     Modernising Arab State*, 1987, and Glottolog's own pairing). No source counts either; the
     Shia counts per governorate are religiondots' Bahraini sect split (`../religiondots/sources/
     bh.md` §7: Arab Barometer I's level, 57.6% Shia of those naming a sect, placed by the two
     endowments' mosque registers), read from its normalized bh.csv, read-only. The Ajam,
     Shia of Persian descent, sit inside that Shia count and so on Baharna Arabic; no source
     counts them or how many still speak Persian.
  3. non-Bahrainis: each census group and sex at UN DESA 2020's named origins in that group (the
     group assignment is religiondots' bh.DESA_GROUP), each origin on its language or home mix;
     the census `Others` group (Oceania, Latin America and the rest, no DESA origin) on `other`.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gulf_mix as g  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "bh"
RD_BH_CSV = g.RD / "data" / "normalized" / "bh.csv"
OUT = ROOT / "data" / "normalized" / "bh.csv"
TOTAL = 1_501_635
GOVS = ["Capital", "Muharraq", "Northern", "Southern"]
SHIA = "Bahraini, Muslim, Shia"

GROUP_ISO = {   # religiondots sources/bh.py DESA_GROUP, by ISO
    "Gulf Co-operative Countries": {"KW", "QA", "SA", "AE", "OM"},
    "Other Arabs": {"EG", "MA", "SD", "TN", "SO", "JO", "LB", "PS", "SY", "YE", "IQ"},
    "Asian": {"AF", "BD", "IN", "NP", "PK", "LK", "ID", "PH", "TH", "TR", "IR"},
    "African": {"ER", "ET", "SS", "TD", "NG"},
    "European": {"GB", "FR", "NL"},
    "North American": {"US"},
}
NO_DESA = {"Others": g.OTHER}
SEX_COL = {"Male": 1, "Female": 2}


def run_extract():
    if "--fetch" in sys.argv or not (RAW / "bh_groups.csv").exists():
        if subprocess.run([sys.executable, str(HERE / "bh_extract.py")]).returncode:
            raise SystemExit("bh_extract.py failed")


def main():
    run_extract()
    grp = pd.read_csv(RAW / "bh_groups.csv")
    bah = grp[grp["group"] == "Bahraini"].groupby("governorate")["people"].sum()

    rd = pd.read_csv(RD_BH_CSV)
    rd_b = rd[rd["source_category"].str.startswith("Bahraini")]
    got = rd_b.groupby("geo_id")["count"].sum()
    if not (got.reindex(GOVS) == bah.reindex(GOVS)).all():
        raise SystemExit(f"religiondots' Bahraini rows {got.to_dict()} != census {bah.to_dict()}")
    shia = rd_b[rd_b["source_category"] == SHIA].set_index("geo_id")["count"].reindex(GOVS)
    print("  Bahrainis on Baharna Arabic (religiondots' Shia): " + ", ".join(
        f"{v} {shia[v]:,} of {bah[v]:,} ({shia[v] / bah[v]:.0%})" for v in GOVS))

    origins, _others, _ = g.desa("Bahrain", year=2020)
    stray = set(origins) - set().union(*GROUP_ISO.values())
    if stray:
        raise SystemExit(f"DESA origins for Bahrain in no group: {sorted(stray)}")
    fixed = {iso: g.origin_mix(iso, "BH", v[0]) for iso, v in origins.items()}
    nb = grp[grp["group"] != "Bahraini"].copy()
    mixes = {}
    for (grp_name, sex), _d in nb.groupby(["group", "sex"]):
        if grp_name in NO_DESA:
            mixes[(grp_name, sex)] = {NO_DESA[grp_name]: 1.0}
        else:
            w = {iso: origins[iso][SEX_COL[sex]] for iso in GROUP_ISO[grp_name] if iso in origins}
            mixes[(grp_name, sex)] = g.blend(w, "BH", fixed)
    nb["part"] = list(zip(nb["group"], nb["sex"]))
    wide = nb.pivot_table(index="governorate", columns="part", values="people", aggfunc="sum").fillna(0)
    fc = g.spread(wide, {p: mixes[p] for p in wide.columns})

    rows = []
    for v in GOVS:
        rows.append(dict(geo_id=v, geo_level="governorate", geo_name=v, origin="Bahraini, Shia",
                         source_category=g.BAHARNA_ARABIC, count=int(shia[v])))
        rows.append(dict(geo_id=v, geo_level="governorate", geo_name=v, origin="Bahraini, not Shia",
                         source_category=g.GULF_ARABIC, count=int(bah[v] - shia[v])))
    for gid, row in fc.iterrows():
        for n, c in row.items():
            if c > 0:
                rows.append(dict(geo_id=gid, geo_level="governorate", geo_name=gid,
                                 origin="non-Bahraini", source_category=n, count=int(c)))
    out = pd.DataFrame(rows)
    out["tier"] = "derived"
    out["year"] = 2020
    if int(out["count"].sum()) != TOTAL:
        raise SystemExit("output does not sum to the census")
    out.to_csv(OUT, index=False, encoding="utf-8")
    g.report(out, TOTAL, f"wrote {OUT}")


if __name__ == "__main__":
    main()
