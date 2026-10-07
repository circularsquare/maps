"""France: build data/normalized/fr.csv. Run through `python sources/fr_insee.py`.

Per département (101, the five overseas included):
  1. immigrants by country of birth (INSEE RP 2023): the 46 named countries where INSEE
     publishes them (52 départements of 500,000+), else its 10 groups; every remainder group
     split into countries by Eurostat's 2021 census citizenship counts for that NUTS 3.
  2. each country on its language (COUNTRY_LANG); a few split (Algeria, Morocco).
  3. regional and overseas languages from the surveys in sources/fr_regional.py.
  4. French: everyone else.
Every row is `derived`: nothing here is a count of a language. Checks print as they run and
stop the build where noted. The record is sources/fr.md.
"""
import json
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "taxonomy"))
import fr_regional as R  # noqa: E402
from fr_insee import RAW, OUT, EU_CTZ, DATASETS, YEAR  # noqa: E402

ROOT = os.path.dirname(HERE)
PLACE = os.path.join(ROOT, "data", "geo", "fr", "fr_place.gpkg")
POP_CACHE = os.path.join(RAW, "pop2023_dep_age.csv")
COM64_CACHE = os.path.join(RAW, "pop2023_com64_age.csv")
SOURCE_ID = "fr_rp2023_immi_x_surveys"

# ---------------------------------------------------------------------------------------------
# country of birth -> language. ISO 3166 alpha-2 as Eurostat writes it (EL Greece, UK). A dict
# splits the country's immigrants. sources/fr.md §4 gives the reasons for the arguable ones.
# ---------------------------------------------------------------------------------------------
ALGERIA_BERBER = 0.30      # Chaker (Inalco / CRB), the lower bound of his 30-40% estimate
# Morocco: the 2024 census's languages used, normalised (Darija 91.9, Tachelhit 14.2,
# Tamazight 7.4, Tarifit 3.2, Hassania 0.8; several allowed, so they sum to 117.5)
_MA = {"Moroccan Arabic": 91.9, "Tachelhit": 14.2, "Tamazight": 7.4, "Tarifit": 3.2,
       "Hassaniya": 0.8}
MOROCCO = {k: v / sum(_MA.values()) for k, v in _MA.items()}

COUNTRY_LANG = {
    # EU
    "BE": "French", "BG": "Bulgarian", "CZ": "Czech", "DK": "Danish", "DE": "German",
    "EE": "Estonian", "IE": "English", "EL": "Greek", "ES": "Spanish", "HR": "Croatian",
    "IT": "Italian", "CY": "Greek", "LV": "Latvian", "LT": "Lithuanian", "LU": "Luxembourgish",
    "HU": "Hungarian", "MT": "Maltese", "NL": "Dutch", "AT": "German", "PL": "Polish",
    "PT": "Portuguese", "RO": "Romanian", "SI": "Slovenian", "SK": "Slovak", "FI": "Finnish",
    "SE": "Swedish",
    # rest of Europe
    "IS": "Icelandic", "LI": "German", "NO": "Norwegian", "CH": "French", "UK": "English",
    "BA": "Bosnian", "ME": "Serbian", "MD": "Romanian", "MK": "Macedonian", "GE": "Georgian",
    "AL": "Albanian", "RS": "Serbian", "TR": "Turkish", "UA": "Ukrainian", "XK": "Albanian",
    "AD": "Catalan", "BY": "Russian", "VA": "Italian", "MC": "French", "RU": "Russian",
    "SM": "Italian",
    # Africa
    "AO": "Portuguese", "CM": "French", "CF": "Sango", "TD": "Arabic", "CG": "Lingala",
    "CD": "Lingala", "GQ": "Fang", "GA": "French", "ST": "Portuguese", "BI": "Kirundi",
    "KM": "Comorian", "DJ": "Somali", "ER": "Tigrinya", "ET": "Amharic", "KE": "Swahili",
    "MG": "Malagasy", "MW": "Nyanja", "MU": "Morisyen", "MZ": "Emakhuwa", "RW": "Kinyarwanda",
    "SC": "Seychellois Creole", "SO": "Somali", "UG": "Luganda", "TZ": "Swahili",
    "ZM": "Bemba", "ZW": "Shona",
    "DZ": {"Arabic": 1 - ALGERIA_BERBER, "Kabyle": ALGERIA_BERBER},
    "EG": "Arabic", "LY": "Arabic", "MA": MOROCCO, "SS": "Dinka", "SD": "Arabic",
    "TN": "Arabic", "EH": "Hassaniya", "BW": "Setswana", "LS": "Sesotho", "NA": "Oshiwambo",
    "ZA": "Zulu", "SZ": "Swati", "BJ": "Fon", "BF": "Moore", "CV": "Kabuverdianu",
    "CI": "French", "GM": "Mandinka", "GH": "Twi", "GN": "Fula", "GW": "Guinea-Bissau Kriol",
    "LR": "Liberian English", "ML": "Bambara", "MR": "Hassaniya", "NE": "Hausa",
    "NG": "Hausa", "SN": "Wolof", "SL": "Krio", "TG": "Ewe",
    # the Americas
    "CA": "French", "US": "English", "AG": "English", "AW": "Papiamento", "BS": "English",
    "BB": "English", "CU": "Spanish", "CW": "Papiamento", "DM": "Antillean Creole",
    "DO": "Spanish", "GD": "English", "HT": "Haitian Creole", "JM": "Jamaican Creole",
    "KN": "English", "LC": "Antillean Creole", "VC": "English", "SX": "English",
    "TT": "English", "BZ": "English", "CR": "Spanish", "SV": "Spanish", "GT": "Spanish",
    "HN": "Spanish", "MX": "Spanish", "NI": "Spanish", "PA": "Spanish", "AR": "Spanish",
    "BO": "Spanish", "BR": "Portuguese", "CL": "Spanish", "CO": "Spanish", "EC": "Spanish",
    "GY": "English", "PY": "Spanish", "PE": "Spanish", "SR": "Ndyuka", "UY": "Spanish",
    "VE": "Spanish",
    # Asia
    "KZ": "Kazakh", "KG": "Kyrgyz", "TJ": "Tajik", "TM": "Turkmen", "UZ": "Uzbek",
    "CN": "Chinese", "JP": "Japanese", "MN": "Mongolian", "KP": "Korean", "KR": "Korean",
    "TW": "Mandarin", "AF": "Dari", "BD": "Bengali", "BT": "Dzongkha", "IN": "Hindi",
    "IR": "Persian", "MV": "Dhivehi", "NP": "Nepali", "PK": "Punjabi", "LK": "Tamil",
    "BN": "Malay", "KH": "Khmer", "ID": "Indonesian", "LA": "Lao", "MY": "Malay",
    "MM": "Burmese", "PH": "Tagalog", "SG": "English", "TH": "Thai", "TL": "Tetun",
    "VN": "Vietnamese", "AM": "Armenian", "AZ": "Azerbaijani", "BH": "Arabic", "IQ": "Arabic",
    "IL": "Hebrew", "JO": "Arabic", "KW": "Arabic", "LB": "Arabic", "PS": "Arabic",
    "OM": "Arabic", "QA": "Arabic", "SA": "Arabic", "SY": "Arabic", "AE": "Arabic",
    "YE": "Arabic",
    # Oceania
    "AU": "English", "NZ": "English", "FJ": "Fijian", "PG": "Tok Pisin",
    "SB": "Solomon Islands Pijin", "VU": "Bislama", "KI": "Gilbertese", "MH": "Marshallese",
    "FM": "English", "NR": "Nauruan", "PW": "Palauan", "WS": "Samoan", "TO": "Tongan",
    "TV": "Tuvaluan",
}

# ---------------------------------------------------------------------------------------------
# Retention (Anita, 2026-10-05): the share of each origin's immigrants who speak only French at
# home goes onto French. TeO2 2019-20, INSEE "Immigrés et descendants d'immigrés" éd. 2023,
# fiche IMMFRA23-F18, figure 3 (metropolitan France, immigrants 18-59): share with a foreign
# family language (ref) x share of those, among parents, who speak it with their children
# (kids). French share = 1 - ref * kids. Countries take their TeO2 region; TeO2 publishes
# regions, so every country in one gets the region's share. Metropolitan départements only
# (TeO is metropolitan). Cross-check: TeO 2008, Condon & Régnard, "Trajectoires et origines"
# (Ined 2016) ch. 4 table 4, "français seulement" with children, sources/fr.md §2b.
# ---------------------------------------------------------------------------------------------
TEO2 = {  # region: (ref, kids), percent
    "Europe": (95, 71), "Spain, Italy": (99, 63), "Portugal": (99, 65), "other EU27": (89, 75),
    "Africa": (95, 55), "Algeria": (98, 63), "Morocco, Tunisia": (98, 59),
    "Sahel": (97, 54), "Guinean or Central Africa": (84, 33),
    "Asia": (96, 73), "Southeast Asia": (93, 61), "China": (99, 67),
    "Turkey, Middle East": (98, 76), "Americas, Oceania": (91, 85),
}
TEO_FRENCH = {k: 1 - r / 100 * c / 100 for k, (r, c) in TEO2.items()}
_EU27 = {"BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "HR", "CY", "LV", "LT", "LU", "HU",
         "MT", "NL", "AT", "PL", "RO", "SI", "SK", "FI", "SE"}
_REGION_OF = {"ES": "Spain, Italy", "IT": "Spain, Italy", "PT": "Portugal", "DZ": "Algeria",
              "MA": "Morocco, Tunisia", "TN": "Morocco, Tunisia", "CN": "China",
              "TW": "China"}
_REGION_OF.update({c: "other EU27" for c in _EU27})
_REGION_OF.update({c: "Sahel" for c in ("SN", "MR", "GM", "GW", "GN", "ML", "BF", "NE", "TD")})
_REGION_OF.update({c: "Guinean or Central Africa" for c in
                   ("CI", "GH", "TG", "BJ", "NG", "CM", "CF", "GA", "CG", "CD", "GQ")})
_REGION_OF.update({c: "Southeast Asia" for c in
                   ("KH", "LA", "VN", "TH", "MM", "MY", "SG", "ID", "PH", "BN", "TL")})
_REGION_OF.update({c: "Turkey, Middle East" for c in
                   ("TR", "IL", "LB", "SY", "IQ", "IR", "JO", "PS", "SA", "KW", "BH", "QA",
                    "AE", "OM", "YE")})


def teo_region(iso, block):
    """The TeO2 region whose share an origin takes; continent where TeO2 names none."""
    if iso in _REGION_OF:
        return _REGION_OF[iso]
    return {"EU": "other EU27", "EUR": "Europe", "AFR": "Africa", "ASI": "Asia",
            "AME": "Americas, Oceania", "OCE": "Americas, Oceania"}[block]


# INSEE's numeric country codes (D table) -> ISO2
INSEE_ISO = {"116": "KH", "12": "DZ", "120": "CM", "124": "CA", "144": "LK", "156": "CN",
             "170": "CO", "174": "KM", "178": "CG", "180": "CD", "276": "DE", "324": "GN",
             "356": "IN", "380": "IT", "384": "CI", "392": "JP", "422": "LB", "450": "MG",
             "466": "ML", "478": "MR", "480": "MU", "504": "MA", "528": "NL", "586": "PK",
             "616": "PL", "620": "PT", "642": "RO", "643": "RU", "686": "SN", "688": "RS",
             "704": "VN", "724": "ES", "756": "CH", "788": "TN", "792": "TR", "826": "UK",
             "840": "US",
             # these four share their codes with départements in INSEE's metadata file, whose
             # labels for them are département names; the codes are ISO 3166 numeric
             "24": "AO", "56": "BE", "76": "BR", "332": "HT"}


# ---------------------------------------------------------------------------------------------
def eurostat():
    """NUTS 3 x ISO2 foreign citizens, 2021, and each citizenship's continent block."""
    d = json.load(open(EU_CTZ, encoding="utf-8"))
    dims, sizes = d["id"], d["size"]
    cats = {x: list(d["dimension"][x]["category"]["index"]) for x in dims}
    order = cats["citizen"]
    blocks, cur = {}, None
    starts = {"BE": "EU", "IS": "EUR", "AO": "AFR", "CA": "AME", "KZ": "ASI", "AU": "OCE"}
    for c in order:
        cur = starts.get(c, cur)
        if len(c) == 2 and c not in ("FR",):
            blocks[c] = cur
    rows = []
    for k, v in d["value"].items():
        i, idx = int(k), []
        for s in reversed(sizes):
            idx.append(i % s)
            i //= s
        idx = list(reversed(idx))
        rec = {x: cats[x][idx[j]] for j, x in enumerate(dims)}
        if rec["citizen"] in blocks and len(rec["geo"]) == 5 and rec["geo"].startswith("FR"):
            rows.append((rec["geo"], rec["citizen"], v))
    df = pd.DataFrame(rows, columns=["nuts3", "iso", "n"])
    missing = sorted(set(blocks) - set(COUNTRY_LANG))
    if missing:
        sys.exit(f"!! Eurostat citizenships with no language: {missing}")
    return df, blocks


def groups(blocks):
    """INSEE group code -> the ISO2 countries it holds, for the D and R tables."""
    eu = {c for c, b in blocks.items() if b == "EU"}
    eur = {c for c, b in blocks.items() if b == "EUR"}
    afr = {c for c, b in blocks.items() if b == "AFR"}
    asi = {c for c, b in blocks.items() if b == "ASI"}
    ame = {c for c, b in blocks.items() if b == "AME"}
    oce = {c for c, b in blocks.items() if b == "OCE"}
    named_d = set(INSEE_ISO.values())
    d = {"UE27_OTH": eu - named_d, "EUR_OTH": eur - named_d, "AFR_OTH": afr - named_d,
         "ASIA_OTH": asi - named_d, "AME_OTH": ame - named_d, "36_9": oce}
    r = {"UE27_OTH": eu - {"PT", "IT", "ES"}, "EUR_OTH": eur - {"TR"},
         "AFR_OTH": afr - {"DZ", "MA", "TN"}, "ROW": asi | ame | oce}
    return d, r


def immigrants():
    """[dep, iso, count]: immigrants by country of birth, remainders split by Eurostat."""
    from fr_insee import read_melodi
    dd, _ = read_melodi("immi_d")
    rr, _ = read_melodi("immi_r")
    eu, blocks = eurostat()
    gd, gr = groups(blocks)
    import geopandas as gpd
    lay = gpd.read_file(PLACE, ignore_geometry=True)
    dep_nuts = dict(lay[["unit", "nuts3"]].drop_duplicates().values)

    # the D groups must partition the R groups (France level)
    fd = dd[dd["GEO_OBJECT"] == "FRANCE"].query("GEO == 'F'").set_index("AREA_COUNTRY")["count"]
    fr_ = rr[rr["GEO_OBJECT"] == "FRANCE"].query("GEO == 'F'").set_index("AREA_COUNTRY")["count"]
    iso_d = {k: INSEE_ISO.get(k, k) for k in fd.index}
    r_of = {}
    for code, members in gr.items():
        for c in members:
            r_of[c] = code
    for c in ("DZ", "MA", "TN", "PT", "IT", "ES", "TR"):
        r_of[c] = {"DZ": "12", "MA": "504", "TN": "788", "PT": "620", "IT": "380",
                   "ES": "724", "TR": "792"}[c]
    recon = {}
    for k, v in fd.items():
        if k == "_T":
            continue
        key = iso_d[k]
        rg = r_of[key] if key in r_of else r_of[next(iter(gd[key]))]
        recon[rg] = recon.get(rg, 0) + v
    print("France, D countries regrouped against the R groups:")
    worst = 0
    for k, v in fr_.items():
        if k == "_T":
            continue
        print(f"  {k:9} R {v:12,.0f}  D {recon.get(k, 0):12,.0f}")
        worst = max(worst, abs(v - recon.get(k, 0)) / v)
    if worst > 0.001:
        sys.exit(f"!! the D table does not regroup to the R table ({worst:.4%})")
    print(f"  immigrants: R {fr_['_T']:,.0f}, D {fd['_T']:,.0f}")

    eu_n = eu.groupby(["nuts3", "iso"])["n"].sum()
    eu_fr = eu.groupby("iso")["n"].sum()

    def split(dep, members, total):
        nuts = dep_nuts[dep]
        w = eu_n.loc[nuts] if nuts in eu_n.index.get_level_values(0) else pd.Series(dtype=float)
        w = w.reindex(sorted(members)).fillna(0)
        if w.sum() <= 0:
            w = eu_fr.reindex(sorted(members)).fillna(0)
        return (w / w.sum() * total)

    rows = []
    d_deps = set(dd.loc[dd["GEO_OBJECT"] == "DEP", "GEO"])
    for dep in sorted(set(rr.loc[rr["GEO_OBJECT"] == "DEP", "GEO"])):
        if dep in d_deps:
            t = dd[(dd["GEO_OBJECT"] == "DEP") & (dd["GEO"] == dep)]
            gmap = gd
        else:
            t = rr[(rr["GEO_OBJECT"] == "DEP") & (rr["GEO"] == dep)]
            gmap = gr
        tot = t.loc[t["AREA_COUNTRY"] == "_T", "count"].sum()
        got = 0.0
        for code, n in t[t["AREA_COUNTRY"] != "_T"][["AREA_COUNTRY", "count"]].values:
            if code in gmap:
                for iso, v in split(dep, gmap[code], n).items():
                    rows.append((dep, iso, v, "remainder"))
            else:
                rows.append((dep, INSEE_ISO[code], n, "named"))
            got += n
        if abs(got - tot) > 1:
            sys.exit(f"!! {dep}: countries sum to {got:,.0f} against {tot:,.0f}")
    df = pd.DataFrame(rows, columns=["dep", "iso", "count", "how"])
    print(f"immigrants placed: {df['count'].sum():,.0f} in {df['dep'].nunique()} départements; "
          f"{df.loc[df['how'] == 'remainder', 'count'].sum():,.0f} via a remainder split")
    return df, rr


def population():
    """Département x single year of age (RP 2023), and Pyrenees-Atlantiques' communes."""
    if not (os.path.exists(POP_CACHE) and os.path.exists(COM64_CACHE)):
        ds = DATASETS["pop"]
        z = zipfile.ZipFile(os.path.join(RAW, f"{ds}_2023.zip"))
        name = [n for n in z.namelist() if n.endswith("_data.csv")][0]
        deps, coms = [], []
        for ch in pd.read_csv(z.open(name), sep=";", dtype=str, chunksize=2_000_000,
                              usecols=["GEO", "GEO_OBJECT", "AGE", "SEX", "OBS_VALUE"]):
            ch = ch[ch["SEX"] == "_T"]
            deps.append(ch[ch["GEO_OBJECT"].isin(["DEP", "FRANCE"])])
            coms.append(ch[(ch["GEO_OBJECT"] == "COM") & ch["GEO"].str.startswith("64")])
        pd.concat(deps).drop(columns="SEX").to_csv(POP_CACHE, index=False)
        pd.concat(coms).drop(columns="SEX").to_csv(COM64_CACHE, index=False)
        print("  cached the population table's département and Pyrenees-Atlantiques rows")

    def tidy(path):
        p = pd.read_csv(path, dtype=str)
        p["n"] = p["OBS_VALUE"].astype(float)
        p = p[p["AGE"] != "_T"].copy()
        p["age"] = p["AGE"].str[1:].replace({"_GE100": "100"}).astype(int)
        return p[["GEO_OBJECT", "GEO", "age", "n"]]
    return tidy(POP_CACHE), tidy(COM64_CACHE)


def adults(pop, dep, age):
    p = pop[(pop["GEO_OBJECT"] == "DEP") & (pop["GEO"] == dep)]
    return p.loc[p["age"] >= age, "n"].sum(), p.loc[p["age"] < age, "n"].sum()


def regional(pop, com64):
    """[dep, label, count, note] from sources/fr_regional.py."""
    import geopandas as gpd
    lay = gpd.read_file(PLACE, ignore_geometry=True)
    zone = dict(lay.loc[lay["zone"] != "", ["lau", "zone"]].values)
    rows = []
    for s in R.SURVEYS:
        lang, age = s["lang"], s["age"]
        deps = s["deps"]
        if deps == "occitan":
            # 7% of the two régions' 15+, the pinned départements at their printed shares, the
            # rest shared evenly; then x the family share
            base = {d: adults(pop, d, age) for d in R.OCCITAN_DEPS}
            total_ad = sum(a for a, _ in base.values())
            pinned = sum(R.OCCITAN_PINNED[d] * base[d][0] for d in R.OCCITAN_PINNED)
            rest_ad = total_ad - sum(base[d][0] for d in R.OCCITAN_PINNED)
            rest_share = (R.OCCITAN_SPEAKERS - pinned) / rest_ad
            print(f"  Occitan: {R.OCCITAN_SPEAKERS:,} speakers over {total_ad:,.0f} 15+ "
                  f"({R.OCCITAN_SPEAKERS / total_ad:.2%}; the survey prints 7%); the "
                  f"unprinted départements at {rest_share:.2%}")
            deps = {d: R.OCCITAN_PINNED.get(d, rest_share) * R.OCCITAN_FAMILY
                    for d in R.OCCITAN_DEPS}
        for dep, share in deps.items():
            ad, ch = adults(pop, dep, age)
            if share == "basque_zones":
                c = com64.copy()
                c["zone"] = c["GEO"].map(zone)
                lost = sorted(set(zone) - set(c["GEO"]))
                if lost:
                    sys.exit(f"!! Basque communes missing from RP 2023: {lost[:5]}")
                c = c[c["zone"].notna()]
                za = c[c["age"] >= age].groupby("zone")["n"].sum()
                zc = c[c["age"] < age].groupby("zone")["n"].sum()
                print(f"  Basque 16+ by zone, RP 2023: {za.round().to_dict()} = {za.sum():,.0f} "
                      f"(the survey: {R.BASQUE_BASE_16:,})")
                n_ad = sum(za[z] * R.BASQUE_ZONES[z] for z in za.index)
                n_ch = sum(zc[z] * R.BASQUE_ZONES[z] for z in zc.index) * s["young_ratio"]
            elif isinstance(share, tuple):
                n_ad, n_ch = share[1], 0.0
            else:
                n_ad = ad * share
                if s.get("young_share") is not None:
                    n_ch = ch * s["young_share"]
                elif s.get("young_ratio"):
                    n_ch = ch * share * s["young_ratio"]
                else:
                    n_ch = 0.0
            rows.append((dep, lang, n_ad + n_ch, "regional"))
            print(f"  {lang:20} {dep:3} adults {n_ad:9,.0f}  children {n_ch:8,.0f}")
    return pd.DataFrame(rows, columns=["dep", "label", "count", "part"])


def overseas(pop, imm_lang, rr):
    """Creoles of 971, 972, 974 on the non-immigrant population; Guyane; Mayotte."""
    rows = []
    r = rr[(rr["GEO_OBJECT"] == "DEP")]
    for dep, (lang, sh_ad, sh_ch) in R.DOM_CREOLE.items():
        ad, ch = adults(pop, dep, R.DOM_AGE)
        # immigrants under 15 come from INSEE's age bands (Y_LT15)
        im_all = r[(r["GEO"] == dep) & (r["AREA_COUNTRY"] == "_T")]["count"].sum()
        im_ch = IMMI_CHILD.get(dep, 0.0)
        n = (ad - (im_all - im_ch)) * sh_ad + (ch - im_ch) * sh_ch
        rows.append((dep, lang, n, "overseas"))
        print(f"  {lang:18} {dep}: {n:9,.0f} of {ad + ch:9,.0f} ({n / (ad + ch):.1%})")
    ad, ch = adults(pop, "973", 0)
    tot = ad + ch
    nd = imm_lang[(imm_lang["dep"] == "973") & (imm_lang["label"].isin(["Ndyuka", "creole.english_based.ndyuka"]))]["count"].sum()
    rows.append(("973", "Guianese Creole", tot * R.GUYANE["Guianese Creole"], "overseas"))
    rows.append(("973", "Ndyuka", max(0.0, tot * R.GUYANE["Ndyuka"] - nd), "overseas"))
    print(f"  Guyane {tot:,.0f}: Guianese Creole {tot * R.GUYANE['Guianese Creole']:,.0f}; "
          f"Maroon {tot * R.GUYANE['Ndyuka']:,.0f} of whom {nd:,.0f} already born in Suriname")
    m = R.MAYOTTE
    abroad = m["pop"] * m["born_abroad"]
    native = m["pop"] * (1 - m["born_abroad"] - m["born_france"])
    k = m["shimaore"] + m["kibushi"]
    may = [("Comorian", abroad * m["comorian"]), ("Malagasy", abroad * m["malagasy"]),
           ("Shimaore", native * m["shimaore"] / k), ("Kibushi", native * m["kibushi"] / k)]
    for lab, n in may:
        rows.append(("976", lab, n, "mayotte"))
    french = m["pop"] - sum(n for _, n in may)
    rows.append(("976", "French", french, "mayotte"))
    print(f"  Mayotte {m['pop']:,}: " + ", ".join(f"{l} {n:,.0f}" for l, n in may)
          + f", French {french:,.0f}")
    return pd.DataFrame(rows, columns=["dep", "label", "count", "part"])


IMMI_CHILD = {}
FRENCH_NODE = "indoeuropean.romance.french"


def main():
    import fr2023
    imm, rr = immigrants()
    # immigrants under 15 in the overseas départements (for the creole base)
    ds = DATASETS["immi_r"]
    with zipfile.ZipFile(os.path.join(RAW, f"{ds}_2023.zip")) as z:
        n = [x for x in z.namelist() if x.endswith("_data.csv")][0]
        full = pd.read_csv(z.open(n), sep=";", dtype=str)
    full = full[(full["GEO_OBJECT"] == "DEP") & (full["SEX"] == "_T") &
                (full["AREA_COUNTRY"] == "_T") & (full["AGE"] == "Y_LT15")]
    IMMI_CHILD.update(dict(zip(full["GEO"], full["OBS_VALUE"].astype(float))))

    _, blocks = eurostat()
    rows = []
    moved = {}
    from origin_mix import mix
    for dep, iso, cnt, how in imm.itertuples(index=False):
        f = 0.0
        if not dep.startswith("97"):
            if iso not in blocks and iso not in _REGION_OF:
                sys.exit(f"!! {iso}: no continent for the TeO2 retention share")
            f = TEO_FRENCH[teo_region(iso, blocks.get(iso))]
        # the shared origin -> language table (sources/origin_mix.py); node ids, French as a label
        for lab, sh in mix(iso, "fr").items():
            if lab == FRENCH_NODE:
                lab = "French"
            if lab == "French":
                rows.append((dep, lab, cnt * sh, iso))
                continue
            rows.append((dep, lab, cnt * sh * (1 - f), iso))
            rows.append((dep, "French", cnt * sh * f, iso))
            moved[lab] = moved.get(lab, 0.0) + cnt * sh * f
    imm_lang = pd.DataFrame(rows, columns=["dep", "label", "count", "origin"])
    print("TeO2 retention, French share applied by region:")
    for k, v in TEO_FRENCH.items():
        print(f"  {k:28} {v:6.1%}")
    tot_moved = sum(moved.values())
    print(f"moved onto French: {tot_moved:,.0f} of "
          f"{imm.loc[~imm['dep'].str.startswith('97'), 'count'].sum():,.0f} metropolitan "
          f"immigrants; largest:")
    for lab, n in sorted(moved.items(), key=lambda x: -x[1])[:12]:
        print(f"  {lab:22} {n:10,.0f}")

    pop, com64 = population()
    print("regional languages:")
    reg = regional(pop, com64)
    print("overseas:")
    dom = overseas(pop, imm_lang, rr)

    tot = pop[pop["GEO_OBJECT"] == "DEP"].groupby("GEO")["n"].sum()
    fr_total = pop[(pop["GEO_OBJECT"] == "FRANCE") & (pop["GEO"] == "F")]["n"].sum()
    print(f"population RP 2023: {tot.sum():,.0f} in {len(tot)} départements "
          f"(France row {fr_total:,.0f})")
    if abs(tot.sum() - fr_total) > 10:
        sys.exit("!! départements do not sum to France")

    out = []
    for dep in sorted(set(tot.index) | {"976"}):
        parts = []
        if dep != "976":
            g = imm_lang[imm_lang["dep"] == dep].groupby("label")["count"].sum()
            for lab, n in g.items():
                parts.append((lab, n, "immigrants by country of birth"))
            for _, r_ in reg[reg["dep"] == dep].iterrows():
                parts.append((r_["label"], r_["count"], "regional survey"))
        for _, r_ in dom[dom["dep"] == dep].iterrows():
            parts.append((r_["label"], r_["count"], "overseas survey"))
        if dep == "976":
            out += [(dep, lab, n, part) for lab, n, part in parts]
            continue
        other = sum(n for lab, n, _ in parts if lab != "French")
        french_imm = sum(n for lab, n, _ in parts if lab == "French")
        rest = tot[dep] - other - french_imm
        if rest < 0:
            sys.exit(f"!! {dep}: the languages drawn exceed the population by {-rest:,.0f}")
        parts = [(lab, n, p) for lab, n, p in parts if lab != "French"]
        parts.append(("French", rest + french_imm, "everyone else"))
        out += [(dep, lab, n, part) for lab, n, part in parts]

    df = pd.DataFrame(out, columns=["geo_id", "source_category", "count", "part"])
    df = df.groupby(["geo_id", "source_category", "part"], as_index=False)["count"].sum()
    df = df[df["count"] > 0]
    df["geo_level"] = "dep"
    df["tier"] = "derived"
    df["year"] = YEAR
    df["source_id"] = SOURCE_ID
    for lab in df["source_category"].unique():
        fr2023.resolve(lab)
    df = df[["geo_id", "geo_level", "source_category", "count", "tier", "part", "year",
             "source_id"]]
    tmp = OUT + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT)

    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT}: {len(df):,} rows, {df['geo_id'].nunique()} départements, "
          f"{df['count'].sum():,.0f} people, {len(nat)} languages")
    print("national, top 30:")
    for lab, n in nat.head(30).items():
        print(f"  {lab:22} {n:12,.0f}  {n / nat.sum():6.2%}")
