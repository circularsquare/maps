"""Bermuda: first language from the 2016 census's country of birth, national
-> data/normalized/bm.csv.

    python sources/bm_census.py

NO CENSUS LANGUAGE QUESTION (2016). Built as Barbados and Antigua (sources/bb.md, ag.md): the
Bermuda-born on English, the foreign-born on their birth country's languages through
sources/origin_mix.py (dest "bm"). Every row `derived`.

THE TABLE: Government of Bermuda, Department of Statistics, 2016 Population and Housing Census
Report (gov.bm/files/media-library/20260413/fdce7ebe-2016_census_report.pdf ->
data/raw/bm/bm_2016_census_report.pdf). Table 1 (p. 32 of the report, PDF p. 40): total 63,779,
Bermuda-born 44,411, not stated 36. Table 4.5 (PDF pp. 144-147): foreign-born population by
country of birth, 19,332, every country named.

CHECKS: Table 4.5's rows sum to its total 19,332; Bermuda-born + foreign-born + not stated =
63,779; Table 4.5's Azores + Portugal equal Table 1's Azores/Portugal (1,643), its UK, US and
Canada equal Table 1's.
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "bm" / "bm_2016_census_report.pdf"
OUT = HERE / "data" / "normalized" / "bm.csv"
TOTAL, BERMUDA, NOT_STATED, FOREIGN = 63_779, 44_411, 36, 19_332

ISO = {
    "United Kingdom": "GB", "United States": "US", "Canada": "CA", "Jamaica": "JM",
    "Philippines": "PH", "Azores": "PT-AZ", "India": "IN", "Portugal": "PT", "Barbados": "BB",
    "Ireland": "IE", "South Africa": "ZA", "Trinidad and Tobago": "TT", "Sri Lanka": "LK",
    "Germany": "DE", "Australia": "AU", "Dominican Republic": "DO", "New Zealand": "NZ",
    "Italy": "IT", "France": "FR", "Romania": "RO", "Saint Kitts and Nevis": "KN", "Kenya": "KE",
    "Guyana": "GY", "Saint Lucia": "LC", "Bangladesh": "BD", "Switzerland": "CH",
    "Saint Vincent and The Grenadines": "VC", "Brazil": "BR", "Ecuador": "EC", "Zimbabwe": "ZW",
    "China": "CN", "Indonesia": "ID", "Bahamas": "BS", "Grenada": "GD", "Japan": "JP",
    "Austria": "AT", "Mexico": "MX", "Thailand": "TH", "Hong Kong": "HK", "Malaysia": "MY",
    "Nepal": "NP", "Nigeria": "NG", "Netherlands": "NL", "Singapore": "SG", "Spain": "ES",
    "Colombia": "CO", "Poland": "PL", "Sweden": "SE", "Antigua and Barbuda": "AG",
    "Mauritius": "MU", "Russia": "RU", "Egypt": "EG", "Dominica": "DM", "Pakistan": "PK",
    "Morocco": "MA", "Peru": "PE", "Argentina": "AR", "Ghana": "GH", "Cuba": "CU",
    "Denmark": "DK", "Czech Republic": "CZ", "Panama": "PA", "Uganda": "UG", "Bulgaria": "BG",
    "Korea South": "KR", "Taiwan": "TW", "Hungary": "HU", "Belize": "BZ", "Ethiopia": "ET",
    "Venezuela": "VE", "Guatemala": "GT", "Israel": "IL", "Ukraine": "UA", "Chile": "CL",
    "Costa Rica": "CR", "El Salvador": "SV", "Haiti": "HT", "Honduras": "HN", "Norway": "NO",
    "Iran": "IR", "Puerto Rico": "PR", "Turkey": "TR", "American Samoa": "AS", "Belgium": "BE",
    "Cayman Islands": "KY", "Malta": "MT", "United Arab Emirates": "AE", "Algeria": "DZ",
    "Cameroon": "CM", "Fiji": "FJ", "Vietnam": "VN", "Cote D'Ivoire": "CI",
    "Croatia (Hrvatska)": "HR", "Estonia": "EE", "Finland": "FI", "Guernsey": "GG",
    "Lebanon": "LB", "Netherlands Antilles": "CW", "Senegal": "SN", "Sierra Leone": "SL",
    "Uruguay": "UY", "British Virgin Islands": "VG", "Isle Of Man": "IM", "Kazakhstan": "KZ",
    "Malawi": "MW", "Tanzania": "TZ", "Albania": "AL", "Bosnia and Herzegowina": "BA",
    "Burma": "MM", "Greece": "GR", "Guam": "GU", "Iceland": "IS", "Jersey": "JE",
    "Lithuania": "LT", "Montserrat": "MS", "Serbia and Montenegro": "RS", "Seychelles": "SC",
    "Slovakia": "SK", "Suriname": "SR", "Turkmenistan": "TM", "Zambia": "ZM", "Angola": "AO",
    "Cambodia": "KH", "Cyprus": "CY", "Iraq": "IQ", "Luxembourg": "LU", "Samoa": "WS",
    "Saudi Arabia": "SA", "Slovenia": "SI", "Virgin Islands": "VI",
    "British Indian Ocean Territory": "IO", "Central African Republic": "CF", "Georgia": "GE",
    "Gibraltar": "GI", "Guadeloupe": "GP", "Kuwait": "KW", "Libya": "LY", "Mali": "ML",
    "Martinique": "MQ", "Moldova": "MD", "Namibia": "NA", "New Caledonia": "NC",
    "Nicaragua": "NI", "Turks and Caicos Islands": "TC", "Azerbaijan": "AZ", "Bahrain": "BH",
    "Belarus": "BY", "Benin": "BJ", "Burkina Faso": "BF", "Congo (Democ Rep)": "CD",
    "Falkland Islands (Malvinas)": "FK", "Gambia": "GM", "Jordan": "JO", "Kiribati": "KI",
    "Kyrgyzstan": "KG", "Laos": "LA", "Liberia": "LR", "Oman": "OM", "Papua New Guinea": "PG",
    "Qatar": "QA", "Rwanda": "RW", "Sudan": "SD", "Swaziland": "SZ", "Syria": "SY",
    "Tonga": "TO", "Tunisia": "TN", "Uzbekistan": "UZ", "West Bank": "PS",
    "South Georgia & Sandwich Islands": "GB",   # 1 person; no population of its own
    "Europa Island": "FR",                       # 1 person; French Southern Territories
    "Paracel Islands": "CN",                     # 3 people; Chinese-administered
}


def table45():
    import fitz
    doc = fitz.open(RAW)
    rows = {}
    for p in range(143, 147):
        ls = [x.replace("\xa0", " ").strip() for x in doc[p].get_text().split("\n") if x.strip()]
        num = [bool(re.fullmatch(r"[\d,]+", x)) for x in ls]
        for i in range(len(ls) - 3):
            if not num[i] and num[i + 1] and num[i + 2] and num[i + 3]:
                n, m, f = (int(ls[i + k].replace(",", "")) for k in (1, 2, 3))
                assert m + f == n, ls[i:i + 4]
                assert ls[i] not in rows, ls[i]
                rows[ls[i]] = n
    tot = rows.pop("Total")
    assert tot == FOREIGN and sum(rows.values()) == FOREIGN, (tot, sum(rows.values()))
    return rows


def main():
    import fitz
    t1 = re.sub(r"\s+", " ", fitz.open(RAW)[39].get_text())
    for lab, n in (("Total", TOTAL), ("Bermuda", BERMUDA), ("Azores/ Portugal", 1_643),
                   ("United Kingdom", 4_088), ("Canada", 2_140), ("Not Stated", NOT_STATED)):
        assert f"{lab} {n:,}" in t1, lab
    assert BERMUDA + FOREIGN + NOT_STATED == TOTAL
    rows = table45()
    assert rows["Azores"] + rows["Portugal"] == 1_643
    missing = sorted(set(rows) - set(ISO))
    assert not missing, missing
    # every birthplace through origin_mix; rows carry node ids (taxonomy/bm2016.py passes them on)
    sys.path.insert(0, str(HERE / "sources"))
    from origin_mix import mix
    rows = {"Bermuda": BERMUDA, **rows}
    out = []
    for k, v in rows.items():
        iso = "PT" if k == "Azores" else ISO.get(k)
        # Bermuda, and the UK, US and Canada as in Barbados and Antigua (bb.md, ag.md): many of
        # those born there are Bermudians' children, and the home mixes' Spanish and French
        # (US 13.6%, Canada 20%) would put ~900 Spanish and French speakers into Bermuda
        if k in ("Bermuda", "United Kingdom", "United States", "Canada"):
            iso = "BM-EN"
        m = {"indoeuropean.germanic.english": 1.0} if iso == "BM-EN" else mix(iso, "bm")
        for node, s in m.items():
            out.append(dict(geo_id="BM", geo_name="Bermuda", birthplace=k,
                            source_category=node, count=v * s))
    df = pd.DataFrame(out)
    assert abs(df["count"].sum() - (TOTAL - NOT_STATED)) < 1e-6
    df["geo_level"] = "country"
    df["tier"] = "derived"
    df["year"] = 2016
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} birthplaces, {df['count'].sum():,} people "
          f"({NOT_STATED} not stated, not drawn)")


if __name__ == "__main__":
    main()
