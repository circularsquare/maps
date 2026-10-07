"""Oman, end of 2024: Omanis and expatriates by wilaya, read as languages.
-> data/normalized/om.csv

    python sources/om_build.py [--fetch]

Nobody in Oman is asked a language; since 2020 the population is read off registers. Built by
the Gulf home-mix method (sources/gulf_mix.py, sources/gulf.md), every row `derived`:

  1. sources/om_extract.py (its own process) copies religiondots' parsed and checked register
     tables (NCSI Statistical Year Book 2025; GLMM's copies of NCSI tables), --fetch or when
     missing.
  2. Omanis on Omani Arabic, per wilaya (register, end 2024).
  3. Expatriates per governorate in three parts, each at its own national nationality mix, as
     religiondots does: male workers, female workers (2024 worker tables by nationality and sex),
     and dependants (expatriates less workers; nationality from mid-2018 population less 2018
     workers). Each wilaya's expatriates take their governorate's mix. Each nationality on its
     language or home mix; the government sector's "Other Arabs" on plain Arabic.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gulf_mix as g  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "om"
OUT = ROOT / "data" / "normalized" / "om.csv"
OMANIS, EXPATRIATES = 2_984_793, 2_283_279
OTHER_ARABS = "node:afroasiatic.arabic"


def run_extract():
    need = [RAW / f for f in ("om_wilayat.csv", "om_gov.csv", "om_weights.csv")]
    if "--fetch" in sys.argv or not all(p.exists() for p in need):
        if subprocess.run([sys.executable, str(HERE / "om_extract.py")]).returncode:
            raise SystemExit("om_extract.py failed")


def main():
    run_extract()
    wil = pd.read_csv(RAW / "om_wilayat.csv")
    gov = pd.read_csv(RAW / "om_gov.csv").set_index("gov")
    w = pd.read_csv(RAW / "om_weights.csv")
    if int(wil["omanis"].sum()) != OMANIS or int(wil["expatriates"].sum()) != EXPATRIATES:
        raise SystemExit("om_wilayat.csv does not sum to the register")
    parts = {p: dict(zip(d["key"], d["people"])) for p, d in w.groupby("part")}

    # nationals of each origin in the layer (for the 20,000 home-mix rule and India's Keralites)
    wm, wf, dep = gov["workers_m"].sum(), gov["workers_f"].sum(), gov["dependants"].sum()
    scale = {"men": wm / sum(parts["men"].values()), "women": wf / sum(parts["women"].values()),
             "dependants": dep / sum(parts["dependants"].values())}
    nat_n = {}
    for p, d in parts.items():
        for k, v in d.items():
            nat_n[k] = nat_n.get(k, 0) + v * scale[p]
    print("  expatriates by nationality in the layer: " + ", ".join(
        f"{k} {v:,.0f}" for k, v in sorted(nat_n.items(), key=lambda kv: -kv[1])))
    fixed = {k: g.origin_mix(k, "OM", n) for k, n in nat_n.items() if k != "ARAB"}
    mixes = {p: g.blend({(OTHER_ARABS if k == "ARAB" else k): v for k, v in d.items()}, "OM", fixed)
             for p, d in parts.items()}

    nodes = sorted({n for m in mixes.values() for n in m})
    gc = pd.DataFrame({gv: {n: r["workers_m"] * mixes["men"].get(n, 0)
                            + r["workers_f"] * mixes["women"].get(n, 0)
                            + r["dependants"] * mixes["dependants"].get(n, 0) for n in nodes}
                       for gv, r in gov.iterrows()}).T[nodes]
    gshare = gc.div(gc.sum(axis=1), axis=0)
    wil = wil.set_index("geo_id")
    fm = pd.DataFrame([gshare.loc[r["gov"]].to_numpy() * r["expatriates"] for _i, r in wil.iterrows()],
                      index=wil.index, columns=nodes)
    fc = g.sa.round_within_rows(fm)
    for gid, r in wil.iterrows():
        if int(fc.loc[gid].sum()) != int(r["expatriates"]):
            raise SystemExit(f"{gid}: rounded expatriates do not sum to the register")

    rows = [dict(geo_id=gid, geo_level="wilaya", geo_name=r["wilaya"], governorate=r["gov"],
                 origin="Omani", source_category=g.OMANI_ARABIC, count=int(r["omanis"]))
            for gid, r in wil.iterrows()]
    for gid, row in fc.iterrows():
        for n, c in row.items():
            if c > 0:
                rows.append(dict(geo_id=gid, geo_level="wilaya", geo_name=wil.at[gid, "wilaya"],
                                 governorate=wil.at[gid, "gov"], origin="expatriate",
                                 source_category=n, count=int(c)))
    out = pd.DataFrame(rows)
    out["tier"] = "derived"
    out["year"] = 2024
    if int(out["count"].sum()) != OMANIS + EXPATRIATES:
        raise SystemExit("output does not sum to the register")
    out.to_csv(OUT, index=False, encoding="utf-8")
    g.report(out, OMANIS + EXPATRIATES, f"wrote {OUT}")
    for p, m in mixes.items():
        top = sorted(m.items(), key=lambda kv: -kv[1])[:6]
        print(f"    {p}: " + ", ".join(f"{n.split('.')[-1]} {x:.1%}" for n, x in top))


if __name__ == "__main__":
    main()
