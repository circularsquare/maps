"""One origin country -> language mix table, shared by every build that proxies immigrant
languages by country of birth or citizenship (fr es it be nl pt gr se; sources/origin_mix.md).

    from origin_mix import mix
    mix("MA", "be")    # -> {node: share}, summing to 1

    python sources/origin_mix.py MA be        print one mix and where it came from
    python sources/origin_mix.py --fragment <cc>
        rewrite the generated "borrowed nodes" block in taxonomy/tree.d/<cc>.txt: every node
        <cc>'s counts() draw that tree.txt lacks, with its ancestors (AGENT_BRIEF §3)

The order, for an origin `iso` and a destination `dest` (lower-case country code):
  1. OVERRIDES[(iso, dest)], then OVERRIDES[(iso, "*")]: a diaspora known to differ from its
     home population, each with a cited source (origin_mix.md §2).
  2. the HOME MIX, Saudi Arabia's method (sources/sa_census.py, sources/sa.md): the origin's own
     drawn counts on this map (countries/<cc>.py counts(), summed nationally), languages under
     1% left out and the rest scaled back to 100%. For an origin that is itself an immigration
     country, only its own languages (NATIVE below): the languages its immigrants brought are
     not what people born there speak.
  3. an origin not drawn on this map: its main language, France's table
     (fr_build.COUNTRY_LANG, labels -> nodes through taxonomy/fr2023.py), or TERRITORY.
The retention step (who speaks the host language at home instead) and the placement stay in each
country's build: this module says only what language an origin's speakers speak.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "taxonomy"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

MIN_SHARE = 0.01   # sa_census.MIN_SHARE

AR = "afroasiatic"
IE = "indoeuropean"
DARIJA = f"{AR}.darija"
TARIFIT = f"{AR}.berber.tarifit"
KABYLE = f"{AR}.berber.kabyle"
BERBER = f"{AR}.berber"
TURKISH = "turkic.turkish"
KURDISH = f"{IE}.iranian.kurdish"
IRAQI_ARABIC = f"{AR}.iraqi_arabic"
DARI = f"{IE}.iranian.dari"
PASHTO = f"{IE}.iranian.pashto"

# ---------------------------------------------------------------------------------------------
# Codes the builds use that are not ISO 3166 alpha-2 of a living state
# ---------------------------------------------------------------------------------------------
ALIAS = {"UK": "GB", "EL": "GR", "NC_": "NC", "CS": "RS", "AN": "CW"}
PSEUDO = {   # dissolved states: one node (born there before the split)
    "YU": f"{IE}.slavic.south.serbocroatian",   # Parkvall (se.md): BKS, not split
    "SU": f"{IE}.slavic.east.russian",
    "QT": f"{IE}.slavic.west.czech",            # SCB's Czechoslovakia
}
# small territories with no row in France's table (be_census.py, nl_build.py had these)
TERRITORY = {**{c: f"{IE}.germanic.english" for c in (
    "AI", "BM", "CK", "CQ", "FK", "GG", "GI", "IM", "JE", "KY", "MS", "PN", "SH", "TC", "VG",
    "VI", "AQ", "IO", "GU", "NU", "MP", "NF", "TK", "UM")},
    **{c: f"{IE}.romance.french" for c in ("BL", "MF", "PF", "PM", "TF", "WF", "FR", "MC")},
    "FO": f"{IE}.germanic.north.danish", "GL": f"{IE}.germanic.north.danish",
    "AS": "austronesian.oceanic.samoan", "GF": "creole.french_based.guianese",
    "GP": "creole.french_based.antillean", "MQ": "creole.french_based.antillean",
    "RE": "creole.french_based.reunionese", "YT": "nigercongo.bantu.shimaore",
    "PR": f"{IE}.romance.spanish", "HK": "sinotibetan.sinitic.cantonese",
    "MO": "sinotibetan.sinitic.cantonese", "SS": "nilosaharan.nilotic.dinka",
}

# ---------------------------------------------------------------------------------------------
# Origins that are themselves immigration countries: the home mix keeps only these languages
# (last part of the node id). Everything else on their map came with their own immigrants.
# Calls, origin_mix.md §1: the US keeps Spanish (a home language of 13.6% of its people, much
# of it US-born); Estonia and Latvia keep Russian; Cyprus keeps Turkish (the north).
# ---------------------------------------------------------------------------------------------
NATIVE = {
    "AE": ["gulf_arabic"], "BH": ["baharna_arabic", "gulf_arabic"], "KW": ["gulf_arabic"],
    "QA": ["gulf_arabic"], "OM": ["omani_arabic"], "SA": ["saudi_arabic"],
    "AT": ["german"], "DE": ["german"], "LI": ["german"],
    "AU": ["english"], "NZ": ["english", "maori"], "IE": ["english", "irish"],
    "GB": ["english", "welsh", "scottish_gaelic", "scots", "irish"],
    "US": ["english", "spanish"], "CA": ["english", "french"],
    "BE": ["dutch", "french", "german"], "LU": ["luxembourgish", "french", "german"],
    "CH": ["german", "french", "italian", "romansh"],
    "NL": ["dutch", "westphalian", "limburgish", "frisian", "gronings", "zeeuws"],
    "FR": ["french", "breton", "gallo", "basque", "alsatian", "lorraine_franconian", "occitan",
           "corsican", "catalan", "antillean", "guianese", "reunionese"],
    "ES": ["spanish", "catalan", "valencian", "galician", "basque", "asturian"],
    "PT": ["portuguese", "mirandese"],
    "IT": ["italian", "neapolitan", "sicilian", "venetian", "lombard", "piedmontese", "ligurian",
           "emilian", "romagnol", "friulian", "ladin", "sardinian", "gallurese", "sassarese",
           "francoprovencal", "german"],
    "GR": ["greek", "turkish", "pomak", "romani", "aromanian", "arvanitika", "macedonian",
           "bulgarian"],
    "SE": ["swedish", "meankieli", "saami_north", "saami_lule", "saami_south"],
    "FI": ["finnish", "swedish", "sami"],
    "CY": ["greek", "turkish"], "MT": ["maltese", "english"],
    "HK": ["cantonese", "mandarin", "english"], "MO": ["cantonese", "mandarin", "sinitic",
                                                       "portuguese"],
    "CW": ["papiamento", "dutch", "english"], "AW": ["papiamento", "dutch", "english"],
    "BQ": ["papiamento", "dutch", "english"],
    # 2026-10-05, session edd42a8c-fix (origin_mix.md §1): immigration countries whose home mix
    # carried their own immigrants' languages (Haitian 6% of Dominicans, Polish 4% of Icelanders,
    # Bengali 15% of Maldivians, Jamaican 25% of Caymanians)
    "DK": ["danish", "faroese", "greenlandic"], "NO": ["norwegian", "saami_north",
                                                       "saami_lule", "saami_south"],
    "IS": ["icelandic"], "MV": ["dhivehi"], "DO": ["spanish"], "MC": ["french", "monegasque"],
    "BS": ["bahamian", "english"], "KY": ["english"], "TC": ["turks_caicos", "english"],
    "AG": ["antiguan", "english"], "KN": ["antiguan", "english"], "MS": ["antiguan", "english"],
    "AI": ["antiguan", "creole", "english"], "VG": ["virgin_islands", "english"],
    "PW": ["palauan", "english"], "GU": ["english", "chamorro"],
}

# ---------------------------------------------------------------------------------------------
# Overrides: (origin, destination or "*") -> a function returning {node: share}. Each needs a
# source saying the diaspora differs from home and by how much (origin_mix.md §2).
# ---------------------------------------------------------------------------------------------


def _norm(d):
    t = sum(d.values())
    return {k: v / t for k, v in d.items()}


def _without(m, drop):
    """A mix with some nodes taken out, rescaled."""
    return _norm({k: v for k, v in m.items() if k not in drop})


def _put(share, node, rest):
    """`share` on `node`, the rest at the mix `rest`."""
    out = {k: v * (1 - share) for k, v in rest.items()}
    out[node] = out.get(node, 0.0) + share
    return out


def _nidi(d):
    # NIDI, Hogendoorn, "Taal en taligheid van mensen met een migratieachtergrond", Demos 39(4),
    # 2023, p. 6: first generation, the origin language understood best, normalised over the
    # languages named (nl_build.py's SPLITS before 2026-10-05)
    return lambda: _norm(d)


def _morocco_be():
    # Reniers, "On the History and Selectivity of Turkish and Moroccan Migration to Belgium",
    # International Migration 37(4), 1999: "More than 40 per cent of Moroccans living in Belgium
    # reported having passed their youth in one of the two provinces of the Rif. Almost all
    # immigrants from this region speak Tarifit." 40% Tarifit; the rest at Morocco's home mix
    # without Tarifit (most of them from Tanger, Tetouan and Oujda, mostly Arabic-speaking).
    return _put(0.40, TARIFIT, _without(home("MA"), {TARIFIT}))


def _morocco_es():
    # Idescat, EULP 2023, Catalonia, first language of people 15+: Tamazight 45,600 against
    # Arabic 179,600, so 20.2% of the two; variety not given, so the Berber group (es2021.py's
    # MOROCCO_BERBER before 2026-10-05)
    b = 45.6 / (45.6 + 179.6)
    return {DARIJA: 1 - b, BERBER: b}


def _algeria_fr():
    # Salem Chaker (Inalco / Centre de Recherche Berbere), Berber speakers 30-40% of Algerians
    # in France; the low end, as Kabyle (fr_build.ALGERIA_BERBER before 2026-10-05). The rest
    # at Algeria's home mix without its Berber nodes.
    m = home("DZ")
    return _put(0.30, KABYLE, _without(m, {n for n in m if n.startswith(BERBER)}))


def _turkey_cap():
    # Center for American Progress, "The Turkish Diaspora in Europe" (2020), DATA4U survey
    # Nov 2019 - Jan 2020, 2,357 people of Turkish origin in Germany, Austria, France and the
    # Netherlands: 6% self-identify primarily as Kurds. Identity, used as the Kurdish share;
    # the rest Turkish. Belgium also: Reniers 1999 finds Belgium's Turks from central Anatolia,
    # the Kurdish east under-represented.
    return {TURKISH: 0.94, KURDISH: 0.06}


def _india_it():
    # Barbara Bertolani (University of Trento), as reported by The Tribune (Chandigarh), 2019,
    # "70% Indian migrants in Italy from Punjab"; CeSPI Brief 2/2023 also gives 70% from
    # Punjab. 70% at Punjab's 2011 census mother-tongue mix (in.csv), the rest at India's.
    import sa_census as sa
    d, _ = sa.india_mix()
    pb = sa.mix_from(d[d["geo_name"] == "PUNJAB"])
    rest = home("IN")
    return {n: 0.70 * pb.get(n, 0.0) + 0.30 * rest.get(n, 0.0) for n in set(pb) | set(rest)}


# --- Uncited overrides (Anita, 2026-10-05: obvious ones allowed without a figure, marked
# "uncited" in origin_mix.md §2b). Shares here are judgement, not measurement.
FRENCH = f"{IE}.romance.french"
ITALIAN = f"{IE}.romance.italian"
ENGLISH = f"{IE}.germanic.english"
AFRIKAANS = f"{IE}.germanic.continental.afrikaans"
PORTUGUESE = f"{IE}.romance.portuguese"
CD_FRENCH = 0.40      # uncited: share of Congolese in Belgium and France speaking French at home


def _one(node):
    return lambda: {node: 1.0}


def _kinshasa():
    """Kinshasa's own drawn mix (countries/cd.py's CD1000: 58% Lingala, the rest by ethnic group),
    1% cut, rescaled."""
    if "CD1000" not in _cache:
        from countries import load_one
        df = load_one("cd")["counts"]()
        s = df[df["unit"] == "CD1000"].groupby("node")["count"].sum()
        s = s / s.sum()
        s = s[s >= MIN_SHARE]
        _cache["CD1000"] = (s / s.sum()).to_dict()
    return _cache["CD1000"]


def _congo_eu():
    # uncited: the Congolese diaspora in Belgium and France is Kinshasa-heavy and largely
    # French-speaking at home; French 40%, the rest at Kinshasa's drawn mix (Lingala ~58% of it)
    return _put(CD_FRENCH, FRENCH, _without(_kinshasa(), {FRENCH}))


def _south_africa_nl():
    # uncited: South Africans in the Netherlands are mostly white Afrikaans speakers; 75%
    # Afrikaans, 25% English
    return {AFRIKAANS: 0.75, ENGLISH: 0.25}


OVERRIDES = {
    # uncited (origin_mix.md §2b)
    ("BE", "fr"): _one(FRENCH),
    ("BE", "nl"): _one(f"{IE}.germanic.continental.dutch"),   # Flemish, next door
    ("CH", "fr"): _one(FRENCH),
    ("CH", "it"): _one(ITALIAN),
    ("CD", "be"): _congo_eu,
    ("CD", "fr"): _congo_eu,
    ("IN", "pt"): lambda: _india_it(),
    ("IN", "gr"): lambda: _india_it(),
    ("ZA", "nl"): _south_africa_nl,
    ("CA", "*"): _one(ENGLISH),
    ("CA", "fr"): _one(FRENCH),
    ("CA", "be"): _one(FRENCH),
    ("BR", "*"): _one(PORTUGUESE),
    # uncited, session edd42a8c-lats (origin_mix.md §2b): Venezuela's home mix has 1.1% Wayuu,
    # who live astride the Colombian border in Zulia; the emigrants to the rest of South
    # America are drawn as Spanish speakers
    **{("VE", d): _one(f"{IE}.romance.spanish") for d in ("ar", "cl", "co", "br", "pe", "ec")},
    # cited
    ("MA", "nl"): _nidi({DARIJA: 51, TARIFIT: 43}),
    ("SR", "nl"): _nidi({"creole.english_based.sranan": 64,
                         f"{IE}.indoaryan.bihari.sarnami": 29, "austronesian.javanese": 2}),
    ("IQ", "nl"): _nidi({IRAQI_ARABIC: 55, KURDISH: 35}),
    ("AF", "nl"): _nidi({DARI: 79, PASHTO: 16}),
    ("TR", "nl"): _nidi({TURKISH: 94, KURDISH: 4}),
    ("MA", "be"): _morocco_be,
    ("MA", "es"): _morocco_es,
    ("DZ", "fr"): _algeria_fr,
    ("TR", "fr"): _turkey_cap,
    ("TR", "be"): _turkey_cap,
    ("IN", "it"): _india_it,
    # INSEE, Pratiques culturelles en Guyane 2019-20: Maroon languages used daily by 8% of
    # Guyane (23,520 people), against 28,062 Suriname-born there; nearly all Suriname-born in
    # France live in Guyane (fr.md). Drawn on Ndyuka, the Maroon node France's map uses.
    ("SR", "fr"): lambda: {"creole.english_based.ndyuka": 1.0},
    # session edd42a8c-latn: Wikipedia, "Chinese people in Panama", Demographics: "Around 80%
    # of this population are of Hakka origin, with the rest being Cantonese and Mandarin
    # speakers" (origin, used as language; the rest split evenly). The cited reference was
    # not checked. China-born in Panama only; Taiwan and Hong Kong keep their home mixes.
    ("CN", "pa"): lambda: {"sinotibetan.sinitic.hakka": 0.80, "sinotibetan.sinitic.cantonese": 0.10,
                           "sinotibetan.sinitic.mandarin": 0.10},
}
# Sweden's Parkvall splits (Iraq, Syria, Turkey, Iran, Ethiopia; Finland-Swedes; Korean and
# Ethiopian adoptees) need SCB's 2006 stocks, so they stay in sources/se_build.py `splits` and
# `keep`; they are overrides all the same (origin_mix.md §2).

# ---------------------------------------------------------------------------------------------
_cache = {}
DRAWN = {p.stem.upper() for p in (ROOT / "countries").glob("*.py") if not p.stem.startswith("_")}


def home(iso):
    """The origin's own drawn mix, NATIVE-filtered, 1% cut, rescaled."""
    iso = ALIAS.get(iso, iso)
    if iso not in _cache:
        from countries import load_one
        df = load_one(iso.lower())["counts"]()
        s = df.groupby("node")["count"].sum()
        s = s[s > 0]
        if iso in NATIVE:
            keep = set(NATIVE[iso])
            s = s[[n.split(".")[-1] in keep for n in s.index]]
        s = s / s.sum()
        s = s[s >= MIN_SHARE]
        _cache[iso] = (s / s.sum()).to_dict()
    return _cache[iso]


GULF_ARAB = {"AE", "BH", "KW", "QA", "OM", "SA"}


def gulf_route(iso, dest):
    """For the Gulf and Jordan builds (gulf_mix.origin_mix, sa_census.nationality_mixes), which
    otherwise take France's single main language or an unfiltered home mix: an origin with an
    override here, or an immigration country (NATIVE, Gulf states aside), takes mix(). None
    otherwise. 2026-10-05: Canadians had been French, Swiss French, Americans the US's whole
    drawn mix (origin_mix.md §2b)."""
    iso = ALIAS.get(iso, iso)
    d = dest.lower()
    if (iso, d) in OVERRIDES or (iso, "*") in OVERRIDES or (
            iso in NATIVE and iso not in GULF_ARAB and iso in DRAWN):
        return mix(iso, d)
    return None


def main_language(iso):
    import fr2023
    from fr_build import COUNTRY_LANG
    key = {"GB": "UK", "GR": "EL"}.get(iso, iso)
    if key in COUNTRY_LANG:
        v = COUNTRY_LANG[key]
        items = [(v, 1.0)] if isinstance(v, str) else list(v.items())
        return {fr2023.NAMES[lab]: s for lab, s in items}
    if iso in TERRITORY:
        return {TERRITORY[iso]: 1.0}
    raise KeyError(f"origin_mix: no language for origin {iso!r}")


def explain(iso, dest="*"):
    iso = ALIAS.get(iso, iso)
    if (iso, dest) in OVERRIDES:
        return f"override for {dest}"
    if (iso, "*") in OVERRIDES:
        return "override"
    if iso in PSEUDO:
        return "dissolved state"
    if iso in DRAWN:
        return "home mix" + (" (own languages only)" if iso in NATIVE else "")
    return "main language"


def mix(iso, dest="*"):
    """{node: share} for people from `iso` living in `dest`."""
    key = (ALIAS.get(iso, iso), dest)
    if key in _cache:
        return _cache[key]
    i = key[0]
    if (i, dest) in OVERRIDES:
        m = OVERRIDES[(i, dest)]()
    elif (i, "*") in OVERRIDES:
        m = OVERRIDES[(i, "*")]()
    elif i in PSEUDO:
        m = {PSEUDO[i]: 1.0}
    elif i in DRAWN:
        m = home(i)
    else:
        m = main_language(i)
    if abs(sum(m.values()) - 1) > 1e-9 or min(m.values()) < 0:
        raise SystemExit(f"origin_mix: {i} -> {dest} sums to {sum(m.values())}")
    _cache[key] = m
    return m


# ---------------------------------------------------------------------------------------------
BLOCK_START = "# --- origin_mix borrowed nodes (python sources/origin_mix.py --fragment {cc}) ---"
BLOCK_END = "# --- end origin_mix borrowed nodes ---"


def fragment(cc):
    """Rewrite <cc>'s generated block of borrowed nodes: everything its counts() draw that
    tree.txt and the rest of its own fragment lack, with ancestors, labels from languages.json."""
    import json
    import re
    from countries import load_one
    tree = ROOT / "taxonomy" / "tree.txt"
    frag = ROOT / "taxonomy" / "tree.d" / f"{cc}.txt"
    start = BLOCK_START.format(cc=cc)

    def ids(text):
        out = set()
        for ln in text.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "|" in ln:
                out.add(ln.split("|")[0].strip())
        return out
    have = ids(tree.read_text(encoding="utf-8"))
    body = frag.read_text(encoding="utf-8") if frag.exists() else ""
    body = re.sub(re.escape(start) + r".*?" + re.escape(BLOCK_END) + r"\n?", "", body, flags=re.S)
    have |= ids(body)
    labels = {n["id"]: n["label"] for n in json.loads(
        (ROOT / "taxonomy" / "languages.json").read_text(encoding="utf-8"))["nodes"]}
    # languages.json carries drawn (regrouped) ids; written ids take their label from the
    # fragment that defines them
    for f in [tree, *sorted((ROOT / "taxonomy" / "tree.d").glob("*.txt"))]:
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "|" in ln:
                i, lab = [x.strip() for x in ln.split("|")[:2]]
                labels.setdefault(i, lab)
    # the WRITTEN ids (2026-10-06): load_one's counts() are regrouped to drawn ids, whose new
    # groups (bantu.zone_a, kwa.gbe...) regroup.txt itself defines, so build.py rejects them here
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"frag_{cc}", ROOT / "countries" / f"{cc}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    used = set(mod.ENTRY["counts"]()["node"].dropna())
    need = set()
    for n in used:
        parts = n.split(".")
        for k in range(1, len(parts) + 1):
            a = ".".join(parts[:k])
            if a not in have:
                need.add(a)
    missing = sorted(n for n in need if n not in labels)
    if missing:
        raise SystemExit(f"no label in languages.json for {missing} (run taxonomy/build.py)")
    lines = [start, "# Every node the origin mixes bring in that tree.txt lacks, repeated without "
             "colour."] + [f"{n} | {labels[n]}" for n in sorted(need)] + [BLOCK_END]
    if not body.endswith("\n") and body:
        body += "\n"
    frag.write_text(body + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"{frag}: {len(need)} borrowed nodes")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if a and a[0] == "--fragment":
        fragment(a[1])
    elif a:
        iso, dest = a[0], (a[1] if len(a) > 1 else "*")
        print(f"{iso} -> {dest}: {explain(iso, dest)}")
        for n, s in sorted(mix(iso, dest).items(), key=lambda kv: -kv[1]):
            print(f"  {s:6.1%}  {n}")
    else:
        raise SystemExit(__doc__)
