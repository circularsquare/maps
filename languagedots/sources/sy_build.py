"""Syria: cited minority estimates placed on their governorates, the rest of each governorate on
its Arabic -> data/normalized/sy.csv. Egypt's estimate route (sources/eg.md).

    python sources/sy_build.py

NOTHING ASKS. No census since 2004, and none since 1960 has tabulated language or ethnicity. No
World Values Survey wave includes Syria (the online tool's country lists for waves 4-7 checked
2026-10-05: Libya is there, Syria is not); no released Arab Barometer wave reached Syria
(religiondots' sources/sy.md: its late-2025 Syria fieldwork is not out). So every figure below
is a published estimate, every row `modelled`.

POPULATION BASE: religiondots' sy_lookup.csv (read-only), the Central Bureau of Statistics'
estimate of people living in each governorate on 31 December 2011 (Statistical Abstract 2012
table 3/2 via OCHA HDX), 21,377,000: the last figure before the war. Displacement since 2011
(millions abroad, millions moved inside) is not drawn; the UN's current governorate figures are
marked not for research. Every estimate below is likewise pre-war.

CARVE-OUTS (per governorate, from the CBS population):
  Kurdish (Kurmanji)
    Al-Hasakeh  55%: Balanche 2018 p51, Kurds "still constitute a slim majority (55%)" of the
                Jazira and Kobane cantons (the Jazira canton is most of Hasakah).
    Aleppo      Afrin district 100% ("almost 100% Kurdish", p51), Ayn al-Arab district 55%
                (p51, as above), Aleppo city 22.5% (p53, "Kurds made up 20-25% of Aleppo's
                population before the war"); the three as shares of the governorate's 2004
                census population (district and subdistrict figures from Wikipedia's district
                pages, 2004 census), applied to the 2011 governorate.
    Damascus    the rest of "the one million Kurds in Damascus and Aleppo" (p51) after Aleppo
                city's share above.
  Armenian      Aleppo 150,000 (p22: "Out of the 150,000 who lived in Aleppo before the war").
  Turkmen       Balanche's 1% of 2011 (p22, figure 16) nationally; Latakia takes the 2004 census
                population of Rabia and Qastal Ma'af subdistricts (8,214 + 16,784; p38: "mostly
                Sunni Turkmen"), grown at the national 2004-11 rate; the rest in Aleppo (Azaz,
                al-Rai and Jarabulus countryside, figure 29).
  Aramaic, variety not split (Turoyo and Assyrian Neo-Aramaic)
    Al-Hasakeh  12.5%, the governorate's Christians in 2011 (Wikipedia, Al-Hasakah
                Governorate, "according to census statistics"); "the majority of the Christians
                are ethnic Assyrians". An upper bound: it includes Armenians and Arabic-speaking
                Christians.
  Western Neo-Aramaic   Rural Damascus 30,000 (Wikipedia infobox, 2023): Maaloula, Jubb'adin.
  The rest: Mesopotamian Arabic in Al-Hasakeh, Ar-Raqqa and Deir-ez-Zor (Glottolog nort3142 and
  meso1252 both list Syria; the Euphrates qeltu and Jazira dialects), Levantine Arabic elsewhere.

CHECKS: units = religiondots' 14 governorates; total = 21,377,000; national Kurdish share against
the usual 10% (Wikipedia, Kurds in Syria) and Balanche's 14% (ethnic, includes Arabic-speaking
Kurds); every governorate's carve-outs under its population.
"""
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

OUT = HERE / "data" / "normalized" / "sy.csv"
LOOKUP = RD_GEO / "sy" / "sy_lookup.csv"
TOTAL = 21_377_000
SOURCE_ID = "estimates_balanche2018_wikipedia"

# Aleppo governorate, 2004 census (Wikipedia district pages, read 2026-10-05)
ALEPPO_2004 = 4_045_200          # governorate (as Wikipedia prints it, rounded to 100)
AFRIN_2004 = 172_095             # Afrin district
AYN_AL_ARAB_2004 = 192_513       # Ayn al-Arab (Kobani) district
ALEPPO_CITY_2004 = 2_181_061     # Mount Simeon subdistrict (Aleppo city)
NATIONAL_2004, NATIONAL_2011 = 17_921_000, TOTAL   # 2004 census; CBS end-2011
LATAKIA_TURKMEN_2004 = 8_214 + 16_784              # Rabia + Qastal Ma'af subdistricts

MESO = {"SY08", "SY09", "SY11"}  # Al-Hasakeh, Deir-ez-Zor, Ar-Raqqa


def main():
    pop, names = {}, {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["pop"])
            names[r["unit"]] = r["name"]
    assert len(pop) == 14 and sum(pop.values()) == TOTAL, (len(pop), sum(pop.values()))

    carve = {u: {} for u in pop}
    notes = {}

    def put(u, lab, n, note):
        carve[u][lab] = carve[u].get(lab, 0) + int(round(n))
        notes[(u, lab)] = note

    # Kurdish
    put("SY08", "Kurdish", 0.55 * pop["SY08"], "Balanche 2018: Jazira canton 55% Kurdish")
    aleppo_k_share = (AFRIN_2004 * 1.0 + AYN_AL_ARAB_2004 * 0.55
                      + ALEPPO_CITY_2004 * 0.225) / ALEPPO_2004
    city_k = ALEPPO_CITY_2004 * 0.225 / ALEPPO_2004 * pop["SY02"]
    put("SY02", "Kurdish", aleppo_k_share * pop["SY02"],
        f"Afrin 100%, Ayn al-Arab 55%, Aleppo city 22.5% (Balanche 2018) of 2004 census "
        f"shares: {aleppo_k_share:.1%} of the governorate")
    put("SY01", "Kurdish", 1_000_000 - city_k,
        f"Balanche 2018's one million Kurds in Damascus and Aleppo, less Aleppo city's "
        f"{city_k:,.0f}")
    # Armenian
    put("SY02", "Armenian", 150_000, "Balanche 2018: 150,000 Armenians in Aleppo before the war")
    # Turkmen
    tk = 0.01 * TOTAL
    lat = LATAKIA_TURKMEN_2004 * NATIONAL_2011 / NATIONAL_2004
    put("SY06", "Turkmen", lat, "Rabia and Qastal Ma'af subdistricts, 2004 census, grown to 2011")
    put("SY02", "Turkmen", tk - lat, "Balanche 2018's 1% Turkmen, less Latakia's")
    # Aramaic
    put("SY08", "Aramaic", 0.125 * pop["SY08"],
        "Christians 12.5% of the governorate, 2011 (Wikipedia); mostly Assyrian; upper bound")
    put("SY03", "Western Neo-Aramaic", 30_000, "Wikipedia (2023): Maaloula and Jubb'adin")

    out, natl = [], {}
    for u in sorted(pop):
        used = sum(carve[u].values())
        if used >= pop[u]:
            raise SystemExit(f"{names[u]}: carve-outs {used:,} exceed population {pop[u]:,}")
        rest = "Mesopotamian Arabic" if u in MESO else "Levantine Arabic"
        rows = dict(carve[u])
        rows[rest] = pop[u] - used
        notes[(u, rest)] = "the governorate's population less the carve-outs"
        assert sum(rows.values()) == pop[u]
        for lab, n in sorted(rows.items(), key=lambda kv: -kv[1]):
            natl[lab] = natl.get(lab, 0) + n
            out.append(dict(geo_id=u, geo_level="governorate", geo_name=names[u],
                            source_category=lab, count=n, tier="modelled",
                            source_id=SOURCE_ID, year=2011, note=notes[(u, lab)]))
        print(f"  {names[u]:<15} {pop[u]:>10,}  " + ", ".join(
            f"{lab} {n / pop[u]:.1%}" for lab, n in sorted(rows.items(), key=lambda kv: -kv[1])))

    k = natl["Kurdish"] / TOTAL
    print(f"  national Kurdish {k:.1%} (usual estimate ~10%, Balanche's ethnic 14%)")
    if not 0.07 <= k <= 0.15:
        raise SystemExit("national Kurdish share outside the published range")

    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                          "count", "tier", "source_id", "year", "note"])
        w.writeheader()
        w.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows, {sum(natl.values()):,} people")
    for lab, n in sorted(natl.items(), key=lambda kv: -kv[1]):
        print(f"      {lab:<22} {n:>11,}  {n / TOTAL * 100:5.2f}%")


if __name__ == "__main__":
    main()
