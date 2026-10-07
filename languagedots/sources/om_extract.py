"""Oman: the register tables religiondots already parsed and checked, copied out as plain CSVs
for sources/om_build.py. Run as its own process (om_build.py does that): it loads religiondots'
sources/om.py, which puts religiondots' taxonomy folder on sys.path.

READ-ONLY on religiondots: its parsers and checks are called, nothing is written there. The
arithmetic below repeats religiondots' om.py main() up to its mixes (sources/om.md §4 there):

Writes data/raw/om/:
  * om_wilayat.csv  63 register wilayat (NCSI Statistical Year Book 2025, Table 7-2, end 2024):
                    governorate, Omanis, expatriates
  * om_gov.csv      11 governorates: expatriates, expatriate workers by sex (Table 8-4), and
                    dependants (expatriates less workers)
  * om_weights.csv  three national nationality mixes, as people: male workers and female workers
                    (Tables 17-4 + 18-4, 2024; "other nationalities" women at 2018's Uganda,
                    Indonesia, Ethiopia, Nepal, men at the named men's mix), and dependants
                    (GLMM's mid-2018 population less 2018 workers by nationality). Keys are ISO
                    codes, or ARAB for the government sector's "Other Arabs".
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RD_OM = ROOT.parent / "religiondots" / "sources" / "om.py"
OUT = ROOT / "data" / "raw" / "om"


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    spec = importlib.util.spec_from_file_location("rd_om", RD_OM)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    import pandas as pd

    page = om.load_yearbook()
    y2023 = om.yearbook_checks(page)
    pop18, w18, w18_total = om.glmm_checks(y2023)

    men, women = {}, {}
    for _label, key, _t, f_, m_ in om.GOV_SECTOR + om.PRIVATE_SECTOR:
        men[key] = men.get(key, 0) + m_
        women[key] = women.get(key, 0) + f_
    other_m, other_f = men.pop("OTHER"), women.pop("OTHER")
    women_2018 = {iso: pop18[iso][1] for iso in om.OTHER_WOMEN_FROM_2018}
    for iso, n in women_2018.items():
        women[iso] = women.get(iso, 0) + other_f * n / sum(women_2018.values())
    named_m = dict(men)
    for k, v in named_m.items():
        men[k] = v + other_m * v / sum(named_m.values())
    dep18 = {iso: max(0, pop18[iso][2] - w18[iso]) for iso in w18}

    rows = [dict(part=p, key=k, people=v) for p, d in (("men", men), ("women", women),
                                                       ("dependants", dep18)) for k, v in d.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "om_weights.csv", index=False)

    gov = pd.DataFrame([dict(gov=g, expatriates=om.GOVERNORATE_2024[g][0],
                             omanis=om.GOVERNORATE_2024[g][1],
                             workers_m=om.WORKERS_2024[g][3], workers_f=om.WORKERS_2024[g][2],
                             dependants=om.GOVERNORATE_2024[g][0] - om.WORKERS_2024[g][1])
                        for g in om.GOVERNORATE_2024])
    if (gov["dependants"] < 0).any():
        raise SystemExit("a governorate has more expatriate workers than expatriates")
    gov.to_csv(OUT / "om_gov.csv", index=False)
    wil = pd.DataFrame([dict(geo_id=f"{g}|{w}", gov=g, wilaya=w, expatriates=v[0], omanis=v[1])
                        for (g, w), v in om.WILAYAT_2024.items()])
    if int(wil["omanis"].sum()) != om.OMANIS or int(wil["expatriates"].sum()) != om.EXPATRIATES:
        raise SystemExit("wilayat do not sum to the register")
    wil.to_csv(OUT / "om_wilayat.csv", index=False)
    print(f"wrote {OUT}: {len(wil)} wilayat ({om.OMANIS:,} Omanis, {om.EXPATRIATES:,} expatriates), "
          f"{len(gov)} governorates, {len(rows)} mix weights")


if __name__ == "__main__":
    main()
