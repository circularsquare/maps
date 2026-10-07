"""Kuwait, 2021 census: Kuwaitis and non-Kuwaitis by area, read as languages.
-> data/normalized/kw.csv

    python sources/kw_build.py [--fetch]

Nobody in Kuwait is asked a language. Built by the Gulf home-mix method (sources/gulf_mix.py,
sources/gulf.md), every row `derived`:

  1. sources/kw_extract.py (its own process) copies religiondots' parsed and checked census
     tables (--fetch, or when missing): 157 areas, Kuwaitis, and non-Kuwaitis by sex and
     nationality group, raked as religiondots does (the census prints groups by governorate,
     each area's 2014 group mix carries them to areas); and PACI's mid-2018 non-Kuwaitis by
     nationality and sex (GLMM).
  2. Kuwaitis on Gulf Arabic.
  3. Each nationality group and sex at its own mix of nationalities: PACI 2018's named ones
     (Egypt, Syria, Saudi Arabia; India, Bangladesh, the Philippines, Pakistan, Sri Lanka, Nepal)
     at their 2018 counts, the group's unnamed rest at UN DESA 2024's other origins in that group
     and sex. Groups PACI 2018 names nothing in (Africa, Europe, the Americas, Oceania) at DESA's
     origins alone; South Americans, whom DESA does not name, on Romance, and Australians on
     English.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gulf_mix as g  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "kw"
OUT = ROOT / "data" / "normalized" / "kw.csv"
KUWAITIS, NON_KUWAITIS = 1_488_435, 2_892_704   # census 2021, less Table 1's 4,578 not stated

# census groups (religiondots kw.GROUPS) -> PACI 2018 rows named inside them, and the DESA origins
# that belong to them
NAMED_2018 = {"ARAB": ("Arab countries", {"Egypt": "EG", "Syria": "SY", "Saudi Arabia": "SA"}),
              "ASIA": ("Asia", {"India": "IN", "Bangladesh": "BD", "Philippines": "PH",
                                "Pakistan": "PK", "Sri Lanka": "LK", "Nepal": "NP"})}
GROUP_ISO = {
    "ARAB": {"EG", "SY", "SA", "JO", "YE", "SD", "PS", "LB", "AE", "BH", "QA", "OM", "IQ", "TN",
             "MA", "SO", "LY", "DZ", "MR", "DJ", "KM"},
    "ASIA": {"IN", "BD", "PH", "PK", "LK", "NP", "ID", "TR", "TH", "AF", "IR", "CN", "MM", "VN"},
    "AFRICA": {"ER", "ET", "SS", "NG", "TD", "UG", "KE", "TZ", "GH"},
    "EUROPE": {"GB", "FR", "NL"},
    "NAM": {"US", "CA"},
}
NO_DESA = {"SAM": "indoeuropean.romance", "OCEANIA": "indoeuropean.germanic.english"}
SEX_COL = {"M": 1, "F": 2}


def run_extract():
    need = [RAW / "kw_area_groups.csv", RAW / "kw_nationality_2018.csv"]
    if "--fetch" in sys.argv or not all(p.exists() for p in need):
        if subprocess.run([sys.executable, str(HERE / "kw_extract.py")]).returncode:
            raise SystemExit("kw_extract.py failed")


def main():
    run_extract()
    ag = pd.read_csv(RAW / "kw_area_groups.csv", keep_default_na=False)
    n18 = pd.read_csv(RAW / "kw_nationality_2018.csv").set_index("label")
    origins, others, _ = g.desa("Kuwait")
    stray = set(origins) - set().union(*GROUP_ISO.values())
    if stray:
        raise SystemExit(f"DESA origins for Kuwait in no group: {sorted(stray)}")

    nat_n = {iso: v[0] for iso, v in origins.items()}
    for _grp, (_tot, named) in NAMED_2018.items():
        for lab, iso in named.items():
            nat_n[iso] = int(n18.at[lab, "total"])
    fixed = {iso: g.origin_mix(iso, "KW", n) for iso, n in nat_n.items()}

    mixes = {}
    for s in ("M", "F"):
        col = "men" if s == "M" else "women"
        for grp in ["ARAB", "ASIA", "AFRICA", "EUROPE", "NAM", "SAM", "OCEANIA"]:
            if grp in NO_DESA:
                mixes[(grp, s)] = {NO_DESA[grp]: 1.0}
                continue
            w = {}
            desa_rest = {iso: origins[iso][SEX_COL[s]] for iso in GROUP_ISO[grp] if iso in origins}
            if grp in NAMED_2018:
                tot_lab, named = NAMED_2018[grp]
                for lab, iso in named.items():
                    w[iso] = float(n18.at[lab, col])
                    desa_rest.pop(iso, None)
                rest = float(n18.at[tot_lab, col]) - sum(w.values())
                k = rest / sum(desa_rest.values())
                print(f"  {grp}/{s}: PACI 2018 names {sum(w.values()):,.0f}; the unnamed {rest:,.0f} "
                      f"at DESA's {len(desa_rest)} other origins ({sum(desa_rest.values()):,} in DESA)")
                w.update({iso: v * k for iso, v in desa_rest.items()})
            else:
                w = dict(desa_rest)
            mixes[(grp, s)] = g.blend(w, "KW", fixed)

    nk = ag[ag["group"] != "KUWAITI"].copy()
    nk["part"] = list(zip(nk["group"], nk["sex"]))
    wide = nk.pivot_table(index="geo_id", columns="part", values="people", aggfunc="sum").fillna(0)
    fc = g.spread(wide, {p: mixes[p] for p in wide.columns})

    kuw = ag[ag["group"] == "KUWAITI"].set_index("geo_id")
    gov = kuw["gov"].to_dict()
    rows = [dict(geo_id=gid, geo_level="area", geo_name=gid, governorate=gov[gid], origin="Kuwaiti",
                 source_category=g.GULF_ARABIC, count=int(r["people"])) for gid, r in kuw.iterrows()]
    for gid, row in fc.iterrows():
        for n, c in row.items():
            if c > 0:
                rows.append(dict(geo_id=gid, geo_level="area", geo_name=gid, governorate=gov[gid],
                                 origin="non-Kuwaiti", source_category=n, count=int(c)))
    out = pd.DataFrame(rows)
    out["tier"] = "derived"
    out["year"] = 2021
    if int(out.loc[out["origin"] == "Kuwaiti", "count"].sum()) != KUWAITIS or \
            int(out["count"].sum()) != KUWAITIS + NON_KUWAITIS:
        raise SystemExit("output does not sum to the census")
    out.to_csv(OUT, index=False, encoding="utf-8")
    g.report(out, KUWAITIS + NON_KUWAITIS, f"wrote {OUT}")
    for iso in ("IN", "PK"):
        top = sorted(fixed[iso].items(), key=lambda kv: -kv[1])[:6]
        print(f"  {iso}: " + ", ".join(f"{n.split('.')[-1]} {x:.1%}" for n, x in top))


if __name__ == "__main__":
    main()
