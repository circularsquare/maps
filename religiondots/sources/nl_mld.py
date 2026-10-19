"""Netherlands: each gemeente's Muslims divided by branch for its Turkish- and Moroccan-origin
Muslims, BORROWING Germany's MLD 2020 shares.

Writes data/normalized/nl_mld.csv: per gemeente, the share of its `Islam` (CBS EBB 2010-2014,
nl.csv) drawn as Sunni, Shi'a, Alevi and Ahmadi. countries/nl.py applies them; the rows stay
`modelled` like the survey rows they divide. Approved by Anita 2026-10-16
(unspecified_splits/islam.md §5): "split their Muslims by borrowing the German MLD 2020
origin-region sect shares applied to each country's Muslim origin mix as best the country
publishes it".

THE ORIGIN MIX. CBS StatLine 37713, population on 1 January 2014 by migratieachtergrond and
gemeente (both generations), for Turkije (H008766) and Marokko (H008673). Fetched from the open
OData API into data/raw/nl/cbs_37713_2014_tr_ma.json (`--fetch`).

HOW MANY OF THEM ARE MUSLIM. SCP, *De religieuze beleving van moslims in Nederland* (2018), on
2015 data: 86% of Turkish-Dutch and 94% of Moroccan-Dutch call themselves Muslim. A Dutch figure.

THE BORROWED SHARES. BAMF, *Muslimisches Leben in Deutschland 2020*, Table 32 (sources/de_mld.py):
Türkei-origin Muslims in Germany for the Turkish-Dutch, Nordafrika-origin for the Moroccan-Dutch.
A German survey's figure applied to the Netherlands; the note says so.

    Turkish-origin Muslims  = 0.86 x Turkish-background population      -> MLD Türkei column
    Moroccan-origin Muslims = 0.94 x Moroccan-background population     -> MLD Nordafrika column
    everyone else in the gemeente's `Islam`                             -> NOT SPLIT, `islam`

Where the two estimates exceed the gemeente's surveyed Muslims (the EBB is a sample, and its
share is of adults), both are scaled down to fit, and the count of such gemeenten is printed.
Other origins (Surinamese, Iraqi, Afghan, Somali, Iranian, converts) are not split: CBS publishes
them, but nothing Dutch says what share of each is Muslim.

Usage:
    python sources/nl_mld.py --fetch     download the CBS table, then build
    python sources/nl_mld.py             build data/normalized/nl_mld.csv
    python sources/nl_mld.py --dry-run   report, write nothing
"""

import json
import os
import sys
import urllib.parse
import urllib.request

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "sources"))
RAW = os.path.join(HERE, "data", "raw", "nl", "cbs_37713_2014_tr_ma.json")
NL = os.path.join(HERE, "data", "normalized", "nl.csv")
OUT = os.path.join(HERE, "data", "normalized", "nl_mld.csv")

ORIGIN = {"H008766": ("Turkije", 0.86, "Türkei"), "H008673": ("Marokko", 0.94, "Nordafrika")}
NAMED = {"Sunnitisch": "MLD 2020 (borrowed): Sunnitisch",
         "Schiitisch": "MLD 2020 (borrowed): Schiitisch",
         "Alevitisch": "MLD 2020 (borrowed): Alevitisch",
         "Ahmadi": "MLD 2020 (borrowed): Ahmadi"}


def fetch():
    rows = []
    for key in ORIGIN:
        flt = (f"Perioden eq '2014JJ00' and Leeftijd eq '10000' and "
               f"Migratieachtergrond eq '{key}' and substringof('GM',RegioS)")
        url = ("https://opendata.cbs.nl/ODataApi/odata/37713/TypedDataSet?$format=json&$filter="
               + urllib.parse.quote(flt))
        req = urllib.request.Request(url, headers={"User-Agent": "religiondots/nl_mld"})
        with urllib.request.urlopen(req, timeout=300) as r:
            got = json.loads(r.read())["value"]
        print(f"  CBS 37713 {ORIGIN[key][0]}: {len(got)} gemeenten")
        rows += got
    tmp = RAW + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, RAW)


def build():
    import de_mld
    t32 = de_mld.T32
    origin = pd.DataFrame(json.load(open(RAW, encoding="utf-8")))
    origin["geo_id"] = origin["RegioS"].str.strip()
    origin["key"] = origin["Migratieachtergrond"].str.strip()
    pop = origin.pivot_table(index="geo_id", columns="key", values="MannenEnVrouwen_1",
                             aggfunc="sum").fillna(0)

    nl = pd.read_csv(NL, dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    isl = nl[(nl["geo_level"] == "gemeente") & (nl["source_category"] == "Islam")]
    isl = isl.set_index("geo_id")["count"].astype(float)
    missing = sorted(set(isl.index) - set(pop.index))
    if missing:
        raise SystemExit(f"{len(missing)} EBB gemeenten have no CBS 37713 row, e.g. {missing[:5]}")

    rows, capped, tot = [], 0, {k: 0.0 for k in NAMED}
    est_all = 0.0
    for gid, m in isl.items():
        est = {key: pop.loc[gid, key] * share for key, (_, share, _) in ORIGIN.items()}
        s = sum(est.values())
        scale = 1.0
        if s > m and s > 0:
            scale, capped = m / s, capped + 1
        est = {k: v * scale for k, v in est.items()}
        est_all += sum(est.values())
        for k, cat in NAMED.items():
            n = sum(est[key] * t32[k][de_mld.REGIONS.index(reg)] / 100
                    for key, (_, _, reg) in ORIGIN.items())
            tot[k] += n
            share = n / m if m > 0 else 0.0
            rows.append(dict(geo_id=gid, geo_level="gemeente", source_category=cat,
                             share=round(share, 6), turkish_pop=pop.loc[gid, "H008766"],
                             moroccan_pop=pop.loc[gid, "H008673"], ebb_islam=round(m, 1),
                             source_id="cbs_37713_2014 x scp_2018 x bamf_mld_2020_t32"))
    out = pd.DataFrame(rows)
    T = isl.sum()
    print(f"  EBB Muslims {T:,.0f}; Turkish- and Moroccan-origin Muslims estimated {est_all:,.0f} "
          f"({100 * est_all / T:.1f}%); scaled down to fit in {capped} of {len(isl)} gemeenten")
    for k in NAMED:
        print(f"      {k:<12} {tot[k]:>10,.0f}  {100 * tot[k] / T:5.1f}% of Dutch Muslims")
    if (out.groupby("geo_id")["share"].sum() > 1 + 1e-9).any():
        raise SystemExit("a gemeente's named shares exceed its Muslims")
    return out


def main():
    if "--fetch" in sys.argv or not os.path.exists(RAW):
        fetch()
    out = build()
    if "--dry-run" in sys.argv:
        print("  --dry-run: nothing written")
        return
    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
