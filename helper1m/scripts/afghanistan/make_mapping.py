"""Draft nsia_to_cod.csv: which COD-AB v03 district (adm2 pcode) each of the
457 units in NSIA's 1404 district tables belongs to.

Run once; the CSV it writes is the record and fetch.py reads only that. The
automatic part matches the COD Dari name (adm2_name1) inside NSIA's Dari name
(NSIA often prints the Pashto and Dari spellings run together) and takes it
only when exactly one COD district in the province matches, falling back to
the English names letter for letter. Everything else,
and the two places where that rule picks wrongly, is in MANUAL below with the
reason.

A target is either one pcode, or a "pool" of donor pcodes with weights
("AF1208:30.9;AF1209:9.2") for a group of new districts NSIA carved out of
several old ones at once: the weights are how many people each donor lost in
the year the new units appeared, which is NSIA's own accounting of where they
came from (see README).
"""
import csv
import os
import re
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
import nsia

HERE = Path(__file__).parent
HELPER = HERE.parents[1]
REPO = HELPER.parent
RAW = HELPER / "data" / "afghanistan" / "raw"
COD_ZIP = REPO / "religiondots" / "data" / "raw" / "af" / "afg_admin_boundaries.geojson.zip"
OUT = HERE / "nsia_to_cod.csv"

# NSIA province name -> COD adm1_name
PROV = {"Maydanwardag": "Maidan Wardak", "Urozgan": "Uruzgan", "Helmand": "Hilmand",
        "Herat": "Hirat", "Sar-e- Pul": "Sar-e-Pul", "Paktia": "Paktya"}

# (NSIA province, NSIA 1404 English name) -> (target, note)
MANUAL = {
    ("Maydanwardag", "Markaz-e- Behsood"): ("AF0409", "COD Dari name misspelt (مدکز)"),
    ("Nangarhar", "Batikot"): ("AF0609", "substring rule also hits Kot"),
    ("Nangarhar", "Momand Darah"): ("AF0617", "Muhmand Dara"),
    ("Nangarhar", "Door Baba"): ("AF0622", "Dur Baba"),
    ("Nangarhar", "Spinghar"): ("AF0615", "temporary; Wikipedia Nangarhar Province: Achin 'includes the Spin Ghar District'"),
    ("Laghman", "Qarghayee"): ("AF0702", "Qarghayi"),
    ("Laghman", "Badpash"): ("AF0701", "temporary; Wikipedia Laghman Province: Mihtarlam 'includes the Badpash District'"),
    ("Laghman", "Frashgan"): ("AF0705", "new in 1404; Dawlat Shah fell 40,283 -> 22,719 that year"),
    ("Panjsher", "Hesa-e Awal"): ("AF0804", "Khenj (Hisa-e-Awal)"),
    ("Panjsher", "Shutol"): ("AF0806", "Shutul"),
    ("Panjsher", "Abshar"): ("AF0803", "temporary; parent not found in any source. Dara chosen: Wikipedia puts Abshar in the south-east, and Dara alone reads 0.39 of COD-PS against 0.74 for the province (0.70 with Abshar)"),
    ("Baghlan", "Provincial Capital ('Baghlan-e- Markazi)"): ("AF0905", "Baghlan-e-Jadid; in 1405 NSIA moves the 'provincial capital' label to Pul-e-Khumri"),
    ("Baghlan", "Jolga"): ("AF0909", "Khwaja Hejran (Jalga) in NSIA 1402"),
    ("Baghlan", "Borka"): ("AF0910", "Burka"),
    ("Baghlan", "Dand-e- Ghori"): ("AF0901", "new in 1403; Pul-e-Khumri fell 258,335 -> 225,324 that year"),
    ("Bamyan", "Yakawlang Dowm"): ("AF1005", "temporary (Yakawlang No. 2)"),
    ("Paktika", "Sar Rawza"): ("AF1205", "Sar Rawzah"),
    ("Paktika", "Neka"): ("AF1213", "Nika"),
    ("Paktika", "Khoshamand"): ("AF1216", "NSIA 1402 'Dila wa Khushamand' split into Delah + Khoshamand in 1403"),
    ("Paktika", "Chahar Buran"): ("AF1208:30.9;AF1209:9.2;AF1207:14.2", "pool: new in 1403 with Shkin, Bakkhil, Shakhil Abad, Nemat Abad; Gomal, Jani Khel and Zarghun Shahr lost these amounts (54.3k against the five's 54.4k)"),
    ("Paktika", "Shkin"): ("AF1208:30.9;AF1209:9.2;AF1207:14.2", "pool, as Chahar Buran (temporary)"),
    ("Paktika", "Bakkhil"): ("AF1208:30.9;AF1209:9.2;AF1207:14.2", "pool, as Chahar Buran (temporary)"),
    ("Paktika", "Shakhil Abad"): ("AF1208:30.9;AF1209:9.2;AF1207:14.2", "pool, as Chahar Buran (temporary)"),
    ("Paktika", "Nemat Abad"): ("AF1208:30.9;AF1209:9.2;AF1207:14.2", "pool, as Chahar Buran (temporary)"),
    ("Paktia", "Waza Zadran"): ("AF1305", "Zadran"),
    ("Paktia", "Zazi"): ("AF1307", "Jaji (Aryob Zazi)"),
    ("Paktia", "Laj-e-Ahmadkhel"): ("AF1308", "Lija Ahmad Khel"),
    ("Paktia", "Dand-e-Patan"): ("AF1311", "Dand Wa Patan"),
    ("Paktia", "Laja Mangal"): ("AF1308", "temporary; GeoNames 'Laja Mangal' falls in Lija Ahmad Khel"),
    ("Paktia", "Mirzaka"): ("AF1302", "temporary; Wikipedia Paktia Province: Ahmad Aba 'includes the unofficial district Mirzaka'"),
    ("Paktia", "Gerda Seari"): ("AF1305", "temporary; Wikipedia Paktia Province: Zadran 'sub-divided in 2005 to create Gerda Serai'"),
    ("Paktia", "Rohani Baba"): ("AF1303", "temporary; AAN election report: Rohani Baba was formerly part of Zurmat"),
    ("Khost", "Esmaeelkhel Aw Mandozai"): ("AF1402", "Mandozayi"),
    ("Khost", "Alisher Aw Terizi"): ("AF1408", "Terezayi"),
    ("Khost", "Sperah"): ("AF1411", "Spera"),
    ("Khost", "Zazi maydan"): ("AF1413", "Jaji Maydan"),
    ("Kunar", "Provincial Capital (Asadabad)"): ("AF1501", "Asad Abad"),
    ("Kunar", "Sawkai"): ("AF1509", "Chawkay"),
    ("Kunar", "Sheltan"): ("AF1506", "temporary; COD's district is 'Shigal wa Sheltan'"),
    ("Nuristan", "Provincial Capital (Paroon)"): ("AF1601", "Parun"),
    ("Nuristan", "Kantowa"): ("AF1601", "new in 1403; Paroon fell 16,091 -> 11,367 that year"),
    ("Nuristan", "Want"): ("AF1602", "new in 1403; Waygal fell 23,367 -> 19,748 that year (Wanat village is in Waygal)"),
    ("Badakhshan", "Arghanchkhwah"): ("AF1703", "Arghanj Khwah"),
    ("Badakhshan", "Yaftal"): ("AF1704", "Yaftal-e-Sufla"),
    ("Badakhshan", "Darwaz"): ("AF1722", "Darwaz-e-Payin (Mamay) in NSIA 1402"),
    ("Badakhshan", "Darwaz-e- Bala"): ("AF1727", "Darwaz-e-Balla"),
    ("Badakhshan", "Pamir"): ("AF1728", "new in 1404; Wakhan dipped that year"),
    ("Takhar", "Rustaq"): ("AF1810", "Rostaq"),
    ("Kunduz", "Kalbad"): ("AF1905", "temporary; Wikipedia Kunduz Province: Imam Sahib 'includes the Kalbaad District'"),
    ("Kunduz", "Gul Tepa"): ("AF1901", "temporary; Wikipedia: Kunduz District 'includes the Gul Tepah District'"),
    ("Kunduz", "Aqtash"): ("AF1904", "temporary; Wikipedia: Khan Abad 'includes the Aqtash District'"),
    ("Samangan", "Royi Doab"): ("AF2005", "Ruy-e-Duab"),
    ("Samangan", "Darah-e-sof-e-Payeen"): ("AF2006", ""),
    ("Samangan", "Darah-e-sof-e-Bala"): ("AF2007", ""),
    ("Balkh", "Marmal"): ("AF2105", "Marmul"),
    ("Balkh", "Chahi"): ("AF2109", "new in 1403; Dawlat Abad fell 125,672 -> 82,322 that year"),
    ("Balkh", "Kohi-e-Alborz"): ("AF2108", "new in 1403; Chimtal fell 109,138 -> 72,995 that year"),
    ("Balkh", "Kaldar"): ("split:AF2113,AF2116", "COD carves Sharak-e-Hayratan out of Kaldar; NSIA has no Hayratan unit, so Kaldar is split by Kontur"),
    ("Sar-e- Pul", "SayedAbad"): ("AF2203:47.5;AF2201:58.2", "pool: these four new 1403 units = Kohistanat's and Sar-e-Pul centre's losses that year"),
    ("Sar-e- Pul", "Albader"): ("AF2203:47.5;AF2201:58.2", "pool, as SayedAbad"),
    ("Sar-e- Pul", "Alfateh"): ("AF2203:47.5;AF2201:58.2", "pool, as SayedAbad"),
    ("Sar-e- Pul", "Aljihad"): ("AF2203:47.5;AF2201:58.2", "pool, as SayedAbad"),
    ("Ghor", "Chaharsada"): ("AF2304", "Charsadra"),
    ("Ghor", "Murghab"): ("AF2301", "new in 1402; Chighcheran (Feroz Koh) fell by the same amount"),
    ("Ghor", "Alfarooq"): ("AF2301:14.9;AF2304:3.5;AF2306:3.4", "pool: new in 1403 with Allah Yar; Feroz Koh, Chaharsada and Shahrak lost these amounts"),
    ("Ghor", "Allah yar"): ("AF2301:14.9;AF2304:3.5;AF2306:3.4", "pool, as Alfarooq"),
    ("Urozgan", "Provincial Capital (Terinkot)"): ("AF2501", "Tirinkot"),
    ("Urozgan", "Chahar Chino"): ("AF2504", "Shahid-e-Hassas (Shahidhassas in NSIA 1402)"),
    ("Zabul", "Tarnak-o-Jaldak"): ("AF2602", ""),
    ("Zabul", "Shinkai"): ("AF2603", ""),
    ("Zabul", "Shamolzai"): ("AF2610", ""),
    ("Zabul", "Khak-e-Afghan"): ("AF2611", "Kakar (Khak-e-Afghan) in NSIA 1402"),
    ("Zabul", "Seorai"): ("AF2603", "new in 1403; Shinkai fell 33,608 -> 20,151 that year"),
    ("Kandahar", "Panjwaee"): ("AF2704", ""),
    ("Kandahar", "Neash"): ("AF2712", "Nesh"),
    ("Kandahar", "Dand"): ("AF2701", "temporary; GeoNames 'Hukumati Dand' falls in COD's Kandahar district"),
    ("Kandahar", "Takhtapul"): ("AF2711", "temporary; GeoNames 'Takhtah Pul' falls in Spin Boldak"),
    ("Kandahar", "Kshata Shahwalikot"): ("AF2706", "Lower Shah Wali Kot, new in 1403; Shah Walikot fell by the same amount"),
    ("Jawzjan", "Kham AaSb"): ("AF2806", "Khamyab"),
    ("Faryab", "Khan Chaharbagh"): ("AF2914", ""),
    ("Faryab", "Bandar"): ("AF2910", "new in 1403; Kohistan fell by 30.9k against Bandar's 30.9k; GeoNames Bandar-e Mullaha is in Kohistan"),
    ("Faryab", "Chehel Gazi"): ("AF2902:48.9;AF2904:10.2;AF2907:40.2", "pool: new in 1403 with Khawaja Musa and Khaibar; Pashtun Kot, Almar and Qaisar lost these amounts"),
    ("Faryab", "Khawaja Musa"): ("AF2902:48.9;AF2904:10.2;AF2907:40.2", "pool, as Chehel Gazi"),
    ("Faryab", "Khaibar"): ("AF2902:48.9;AF2904:10.2;AF2907:40.2", "pool, as Chehel Gazi"),
    ("Faryab", "Ferdows"): ("AF2902", "new in 1404; Pashtun Kot fell 179,849 -> 162,232 that year"),
    ("Helmand", "Nawa"): ("AF3003", "Nawa-e-Barakzaiy"),
    ("Helmand", "Gereshk"): ("AF3004", "new in 1403 out of Nahr-e-Saraj (184,343 -> 24,587); Gereshk is Nahr-e-Saraj's town"),
    ("Helmand", "khanneshin"): ("AF3011", "Reg-i-Khan Nishin"),
    ("Helmand", "Marja"): ("AF3002", "temporary; Wikipedia: 'used to belong to Nad Ali District'; GeoNames Marjah in Nad-e-Ali"),
    ("Helmand", "Nawahmesh"): ("AF3012", "temporary; Etilaat-e Roz (etilaatroz.com/43917): Nawamish was part of Baghran (its administration later handed to Daykundi; NSIA keeps it in Helmand)"),
    ("Helmand", "Bahramcha"): ("AF3013:12.7;AF3006:6.1", "pool: new in 1403; Dishu and Garmser lost these amounts"),
    ("Helmand", "Bughni"): ("AF3012", "new in 1403; Baghran fell 116,566 -> 84,024 that year"),
    ("Helmand", "Babaji"): ("AF3001:22.8;AF3002:15.5;AF3004:19.1", "pool: new in 1403; Lashkargah, Nad-e-Ali and Nahr-e-Saraj lost these amounts"),
    ("Badghis", "Qades"): ("AF3104", "Qadis"),
    ("Badghis", "Sang-e-Atash"): ("AF3102", "new in 1403; Abkamari lost 27.1k against Sang-e-Atash's 27.3k; GeoNames Sang Atesh is in Ab Kamari"),
    ("Badghis", "Darah-e-Bom"): ("AF3104:15.3;AF3105:26.0;AF3103:2.0;AF3106:32.0", "pool: new in 1403 with Tagab Alam; Qadis, Bala Murghab, Muqur and Jawand lost these amounts"),
    ("Badghis", "Tagab Alam"): ("AF3104:15.3;AF3105:26.0;AF3103:2.0;AF3106:32.0", "pool, as Darah-e-Bom"),
    ("Herat", "Kushk-e-Kohna"): ("AF3210", ""),
    ("Herat", "Obey"): ("AF3212", "Obe"),
    ("Herat", "Kuhsan"): ("AF3213", "Kohsan"),
    ("Herat", "Zerekoh"): ("AF3214", "temporary; one of the five 2018 parts of Shindand"),
    ("Herat", "Pusht-e- koh"): ("AF3214", "temporary; part of Shindand"),
    ("Herat", "Kohezor"): ("AF3214", "temporary; part of Shindand"),
    ("Herat", "Zawal"): ("AF3214", "temporary; part of Shindand"),
    ("Farah", "Farahrod"): ("AF3306:36.2;AF3310:16.3", "pool: new in 1403; Bala Buluk and Gulistan lost these amounts (substring rule would wrongly pick Farah city)"),
    ("Nimroz", "Delaram"): ("AF3405", "temporary; GeoNames Dilaram is in Khashrod"),
}


def fa_norm(s):
    s = (s or "").replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")
    s = re.sub(r"[^؀-ۿ]", "", s)
    return s.replace("آ", "ا").replace("أ", "ا")


def main():
    units = nsia.read_xlsx(RAW / "nsia_1404.xlsx")
    g2 = gpd.read_file(f"zip://{COD_ZIP}!afg_admin2.geojson")
    rows, used = [], set()
    for r in units:
        cp = PROV.get(r["prov"], r["prov"])
        sub = g2[g2.adm1_name == cp]
        assert len(sub), r["prov"]
        key = (r["prov"], r["name"])
        if key in MANUAL:
            target, note = MANUAL[key]
            used.add(key)
        else:
            name = r["name"].lower()
            if "capital" in name or "center" in name or "centre" in name:
                hit = list(sub[sub.unittype.isin(["Provincial Centre", "Capital"])].adm2_pcode)
            else:
                nf = fa_norm(r["name_fa"])
                hit = [row.adm2_pcode for row in sub.itertuples()
                       if fa_norm(row.adm2_name1) and fa_norm(row.adm2_name1) in nf]
                if not hit:   # then the English names, letters only
                    la = re.sub(r"[^a-z]", "", name)
                    hit = [row.adm2_pcode for row in sub.itertuples()
                           if re.sub(r"[^a-z]", "", row.adm2_name.lower()) == la]
            assert len(hit) == 1, (key, hit)
            target, note = hit[0], ""
        rows.append({"province": r["prov"], "sno": r["sno"], "name_1404": r["name"],
                     "target": target, "note": note})
    unused = set(MANUAL) - used
    assert not unused, unused
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
