"""United Arab Emirates, around 2024: Emiratis and non-Emiratis by emirate, read as languages.
-> data/normalized/ae.csv

    python sources/ae_build.py

Nobody in the UAE is asked a language (the 2005 federal census was the last, and the emirate
censuses since publish none found). Built by the Gulf home-mix method (sources/gulf_mix.py,
sources/gulf.md), every row `derived`:

  * the population by emirate is religiondots' (`../religiondots/sources/ae.py`, read-only, via
    its normalized CSVs): each emirate's newest total scaled to FCSC's 2024 national 11,294,243,
    and each emirate's newest count of Emiratis grown to 2024 (religiondots sources/ae.md §2);
  * Emiratis on Gulf Arabic;
  * non-Emiratis at one national mix of origins, UN DESA International Migrant Stock 2024 (the
    only origin table there is; nothing by emirate), each origin on its language or home mix;
    DESA's unnamed `Others` on `other`. One mix for both sexes, as religiondots found DESA's UAE
    men and women nearly alike.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gulf_mix as g  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = HERE.parent
RD_NORM = g.RD / "data" / "normalized"
OUT = ROOT / "data" / "normalized" / "ae.csv"
TOTAL_2024 = 11_294_243          # FCSC 2024, religiondots sources/ae.py NATIONAL_2024
EMIRATIS = 1_519_227             # religiondots sources/ae.py NOTE


def main():
    emi = pd.read_csv(RD_NORM / "ae.csv", dtype={"geo_id": str}).set_index("geo_id")
    fx = pd.read_csv(RD_NORM / "ae_foreign.csv", dtype={"geo_id": str})
    non = fx.groupby("geo_id")["count"].sum()
    units = pd.DataFrame({"name": emi["geo_name"], "citizens": emi["count"], "non": non})
    if units.isna().any().any() or len(units) != 7:
        raise SystemExit("religiondots' ae.csv and ae_foreign.csv do not cover the same 7 emirates")
    if int(units["citizens"].sum()) != EMIRATIS or \
            int(units["citizens"].sum() + units["non"].sum()) != TOTAL_2024:
        raise SystemExit("religiondots' UAE layer no longer sums to FCSC 2024 / its Emiratis")

    origins, others, _world = g.desa("United Arab Emirates")
    w = {k: v[0] for k, v in origins.items()}
    w["node:" + g.OTHER] = others[0]
    mix = g.blend(w, "AE")
    fc = g.spread(units[["non"]], {"non": mix})

    rows = [dict(geo_id=gid, geo_level="emirate", geo_name=r["name"], origin="Emirati",
                 source_category=g.GULF_ARABIC, count=int(r["citizens"]))
            for gid, r in units.iterrows()]
    for gid, row in fc.iterrows():
        for n, c in row.items():
            if c > 0:
                rows.append(dict(geo_id=gid, geo_level="emirate", geo_name=units.at[gid, "name"],
                                 origin="non-Emirati", source_category=n, count=int(c)))
    out = pd.DataFrame(rows)
    out["tier"] = "derived"
    out["year"] = 2024
    if int(out["count"].sum()) != TOTAL_2024:
        raise SystemExit("output does not sum to FCSC 2024")
    out.to_csv(OUT, index=False, encoding="utf-8")
    g.report(out, TOTAL_2024, f"wrote {OUT}")
    top = sorted(mix.items(), key=lambda kv: -kv[1])[:10]
    print("  non-Emirati mix: " + ", ".join(f"{n.split('.')[-1]} {s:.1%}" for n, s in top))


if __name__ == "__main__":
    main()
