"""Kuwait: the 2021 census tables religiondots already parsed and checked, copied out as plain
CSVs for sources/kw_build.py. Run as its own process (kw_build.py does that): it loads
religiondots' sources/kw.py, which puts religiondots' taxonomy folder on sys.path.

READ-ONLY on religiondots: its parsers, checks and raking are called, nothing is written there.

Writes data/raw/kw/:
  * kw_area_groups.csv  157 census areas (Table 52): Kuwaitis, and non-Kuwaitis by sex and
                        nationality group (Arab incl. other GCC, Asian, African, European, North
                        American, South American, Australian), raked exactly as religiondots'
                        kw.py main() does: each area's December 2014 group mix (PACI via GLMM)
                        raked per governorate and sex to the census's Table 6 group totals
  * kw_nationality_2018.csv  PACI mid-2018 non-Kuwaitis by country or region and sex (GLMM)
"""
import importlib.util
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RD_KW = ROOT.parent / "religiondots" / "sources" / "kw.py"
OUT = ROOT / "data" / "raw" / "kw"


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    spec = importlib.util.spec_from_file_location("rd_kw", RD_KW)
    kw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kw)
    import pandas as pd

    areas, groups21 = kw.load_census()
    loc14 = kw.load_2014_localities()
    # religiondots kw.py main(): the crosswalk and the raking, same arithmetic
    prior14 = {}
    for n, d in loc14.items():
        tgt = kw.XWALK_2014.get(n)
        if tgt is None:
            continue
        for s in kw.SEXES:
            p = prior14.setdefault((tgt, s), dict.fromkeys(kw.GROUPS, 0))
            for g in kw.GROUPS:
                p[g] += d[s][g]
    recs = []
    for gov in kw.GOVS:
        sub = areas[areas["gov"] == gov]
        for s in kw.SEXES:
            col = pd.Series(groups21[(gov, s)])
            rows = sub.set_index("en")["nk_" + s.lower()].astype(float)
            pr = []
            for en in rows.index:
                p = prior14.get((en, s))
                if p and sum(p.values()) >= kw.PRIOR_MIN:
                    pr.append(pd.Series(p, dtype=float) / sum(p.values()))
                else:
                    pr.append(col / col.sum())
            prior = pd.DataFrame(pr, index=rows.index)[kw.GROUPS].mul(rows, axis=0)
            for g in kw.GROUPS:
                if col[g] > 0 and prior[g].sum() == 0:
                    prior[g] = rows * 1e-6
            m = kw.ipf(prior, rows, col[kw.GROUPS].astype(float), f"{gov}/{s}")
            for en, r in m.iterrows():
                for g in kw.GROUPS:
                    recs.append(dict(geo_id=en, gov=gov, sex=s, group=g, people=r[g]))
    grp = pd.DataFrame(recs)
    kwt = areas.set_index("en")
    grp["kuwaitis"] = 0
    kuw = pd.DataFrame(dict(geo_id=kwt.index, gov=kwt["gov"].values, sex="", group="KUWAITI",
                            people=(kwt["kw_m"] + kwt["kw_f"]).values, kuwaitis=1))
    out = pd.concat([kuw, grp], ignore_index=True).drop(columns="kuwaitis")
    nk = int(areas["nk_m"].sum() + areas["nk_f"].sum())
    if abs(grp["people"].sum() - nk) > 1:
        raise SystemExit("raked groups do not sum to the census's non-Kuwaitis")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "kw_area_groups.csv", index=False)

    t = kw.glmm_table("nationality_sex_2018")
    nat = []
    for i in range(len(t)):
        lab = re.sub(r"^of which ", "", str(t.iat[i, 0]).strip())
        try:
            v = tuple(int(str(t.iat[i, j]).replace(",", "")) for j in (1, 2, 3))
        except ValueError:
            continue
        nat.append(dict(label=lab, men=v[0], women=v[1], total=v[2]))
    pd.DataFrame(nat).to_csv(OUT / "kw_nationality_2018.csv", index=False)
    print(f"wrote {OUT / 'kw_area_groups.csv'} ({areas['en'].nunique()} areas, "
          f"{int(kuw['people'].sum()):,} Kuwaitis, {nk:,} non-Kuwaitis) and kw_nationality_2018.csv")


if __name__ == "__main__":
    main()
