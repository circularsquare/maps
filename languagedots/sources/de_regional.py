"""Germany's regional and minority languages, which the Mikrozensus does not name.

    python sources/de_regional.py      -> data/normalized/de_regional.csv  (Land, node, count, area)
                                          data/normalized/de_regional_areas.csv (ars, area, factor)

All rows are `modelled`: a survey share or a published speaker estimate, placed by homeland
(AGENT_BRIEF §2, rich countries without a language question; ask 019 for the estimates).
countries/de.py takes each count out of the same Land's German and places it on the area's
Gemeinden (cells weighted by population x factor). sources/de.md has the reasoning.

Low German: Adler, Ehlers, Goltz, Kleene, Plewnia (2016), "Status und Gebrauch des
Niederdeutschen 2016", IDS/INS, Abb. 10: share of the German-speaking population aged 16+ who
speak Plattdeutsch "sehr gut", by Land (n=1,632, June 2016). The survey has no first-language or
home-language question; "sehr gut" is the nearest. The share is applied to the Land's German
speakers (Mikrozensus) in the surveyed area, times 0.85 for the 16-and-over share (under-20s:
0.8% speak it well, Abb. 11). Inside the area, Gemeinden under 20,000 people weigh double: the
survey finds competence higher there, and Platt heard in the neighbourhood 36.6% in places
under 2,000 against under 17% over 50,000 (p. 19).
"""
import os
import sys

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM = os.path.join(ROOT, "data", "normalized")
PLACE = os.path.join(ROOT, "data", "geo", "de", "de_grid_1km.gpkg")
RAW_NAMES = os.path.join(ROOT, "data", "raw", "de", "zensus2022_1000A-1023_gemeinden.zip")
OUT = os.path.join(NORM, "de_regional.csv")
OUT_AREAS = os.path.join(NORM, "de_regional_areas.csv")

GE = "indoeuropean.germanic"
NDS = f"{GE}.continental.lowgerman.plattdeutsch"
WEP = f"{GE}.continental.lowgerman.westphalian"   # nl's node: Glottolog Westphalic, west2356
HSB = "indoeuropean.slavic.west.upper_sorbian"
DSB = "indoeuropean.slavic.west.lower_sorbian"
FRR = f"{GE}.north_frisian"
STQ = f"{GE}.saterland_frisian"
DAN = f"{GE}.north.danish"

ADULT = 0.85      # aged 16 and over, the survey's universe (Destatis 2022: ~14.7% under 16)
SMALL = 20_000    # Gemeinden below this weigh RURAL x
RURAL = 2.0

# Abb. 10, "sehr gut" (and "gut" for the record): % of the surveyed population
LOW_GERMAN = {   # Land: (sehr gut, gut, node, area prefixes or None for the whole Land, minus)
    "01": (16.5, 8.0, NDS, None, ()),
    "02": (3.2, 6.3, NDS, None, ()),
    "03": (4.7, 12.7, NDS, None, ()),
    "04": (9.9, 7.7, NDS, None, ()),
    "13": (5.9, 14.8, NDS, None, ()),
    # NRW: the survey's "northern part"; taken as Westfalen-Lippe, the Regierungsbezirke
    # Muenster, Detmold and Arnsberg, less Siegen-Wittgenstein (Central German dialects)
    "05": (5.2, 6.6, WEP, ("055", "057", "059"), ("05970",)),
    # Sachsen-Anhalt's north: Altmark, Boerde, Magdeburg, Jerichower Land
    "15": (2.2, 9.6, NDS, ("15081", "15090", "15083", "15003", "15086"), ()),
    # Brandenburg's north: Prignitz, Ostprignitz-Ruppin, Uckermark, Oberhavel
    "12": (2.6, 0.2, NDS, ("12070", "12068", "12073", "12065"), ()),
}

# Sorbian settlement area (Saechsisches Sorbengesetz annex; Brandenburg's Sorben/Wenden-Gesetz),
# de.wikipedia "Sorbisches Siedlungsgebiet". Gemeinden in it with only a few named villages are
# left out; those in it "except" a few villages are in whole.
HSB_CORE = ["Crostwitz", "Panschwitz-Kuckau", "Ralbitz-Rosenthal", "Nebelschütz", "Räckelwitz"]
HSB_REST = ["Bautzen", "Doberschau-Gaußig", "Elsterheide", "Göda", "Großdubrau",
            "Großpostwitz/O.L.", "Hochkirch", "Hoyerswerda", "Königswartha", "Kubschütz", "Lohsa",
            "Malschwitz", "Neschwitz", "Obergurig", "Puschwitz", "Radibor", "Spreetal",
            "Weißenberg", "Wittichenau",
            "Bad Muskau", "Boxberg/O.L.", "Gablenz", "Groß Düben", "Hohendubrau", "Krauschwitz",
            "Kreba-Neudorf", "Mücka", "Rietschen", "Schleife", "Trebendorf", "Weißkeißel",
            "Weißwasser/O.L."]
DSB_AREA = ["Cottbus", "Burg (Spreewald)", "Briesen", "Dissen-Striesow", "Drachhausen", "Drebkau",
            "Drehnow", "Forst (Lausitz)", "Guhrow", "Heinersbrück", "Jänschwalde", "Kolkwitz",
            "Neuhausen/Spree", "Peitz", "Schmogrow-Fehrow", "Spremberg", "Tauer", "Teichland",
            "Turnow-Preilack", "Welzow", "Werben", "Wiesengrund",
            "Byhleguhre-Byhlen", "Lübben (Spreewald)", "Neu Zauche", "Schlepzig",
            "Spreewaldheide", "Straupitz",
            "Calau", "Lübbenau/Spreewald", "Neupetershain", "Neu-Seeland", "Senftenberg",
            "Vetschau/Spreewald"]

ESTIMATES = [
    # node, Land, count, area, how the area is weighted
    # Upper Sorbian: 20,000-25,000 speakers (Schoen, Scholze (eds.), Sorbisches Kulturlexikon,
    # Bautzen 2014, p. 291), low end; in the five Catholic core Gemeinden 60% of residents (the
    # low end of the 60-90% de.wikipedia gives for its villages), the rest over the settlement area
    (HSB, "14", 20_000, "hsb", "core 60%, rest rural-weighted"),
    # Lower Sorbian: about 5,000 speakers (Starosta, Bartels, "Niedersorbisch", Sorabicon)
    (DSB, "12", 5_000, "dsb", "rural-weighted"),
    # North Frisian: 8,000-10,000 (Land Schleswig-Holstein's official figure), low end; 3,500 on
    # Foehr and Amrum (de.wikipedia "Nordfriesische Sprache"), the rest over Nordfriesland and
    # Helgoland
    (FRR, "01", 8_000, "frr", "Foehr-Amrum 3,500, rest rural-weighted"),
    # Saterland Frisian: 1,500-2,500 (Fort 2001, Handbuch des Friesischen, p. 410; Stellmacher's
    # 1995 questionnaire 2,225); midpoint, on the Gemeinde Saterland
    (STQ, "03", 2_000, "stq", "population"),
    # Danish minority: 8,000-10,000 use Danish daily (Institut for Graenseregionsforskning /
    # Gesellschaft fuer bedrohte Voelker, via de.wikipedia "Daenische Minderheit"), low end;
    # Suedschleswig taken as Flensburg and the Kreise Schleswig-Flensburg, Nordfriesland,
    # Rendsburg-Eckernfoerde
    (DAN, "01", 8_000, "dan", "population"),
]
FRR_FOEHR = 3_500
HSB_CORE_SHARE = 0.60


def names():
    import io
    import zipfile
    with zipfile.ZipFile(RAW_NAMES) as z:
        n = [x for x in z.namelist() if x.endswith(".csv")][0]
        with z.open(n) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), sep=";", dtype=str,
                             usecols=["1_variable_code", "1_variable_attribute_code",
                                      "1_variable_attribute_label"])
    df = df[df["1_variable_code"] == "GEOGM4"].drop_duplicates("1_variable_attribute_code")
    return pd.Series(df["1_variable_attribute_label"].to_numpy(),
                     index=df["1_variable_attribute_code"].to_numpy())


def find(nm, land, wanted):
    """Gemeinde names -> ars, inside one Land; every name must match exactly one Gemeinde
    (the table adds ', Stadt' and the like after a comma)."""
    sub = nm[nm.index.str.startswith(land)]
    base = sub.str.split(",").str[0].str.strip()
    out = []
    for w in wanted:
        hit = base[base == w]
        if len(hit) != 1:
            near = sorted(base[base.str.startswith(w.split("/")[0].split(" ")[0])])[:6]
            raise SystemExit(f"{land} {w!r}: {len(hit)} matches; near: {near}")
        out.append(hit.index[0])
    return out


def main():
    place = gpd.read_file(PLACE, ignore_geometry=True, columns=["ars", "pop"])
    gpop = place.groupby("ars")["pop"].sum()
    land_pop = gpop.groupby(gpop.index.str[:2]).sum()
    nm = names()
    missing = set(gpop.index) - set(nm.index)
    assert not missing, f"layer Gemeinden without a name: {sorted(missing)[:5]}"

    de = pd.read_csv(os.path.join(NORM, "de.csv"), dtype={"geo_id": str})
    de = de[de["geo_level"] == "land"]
    german = de[de["source_category"] == "Deutsch"].groupby("geo_id")["count"].sum()

    rows, areas = [], []
    rural = lambda ars: [RURAL if gpop[a] < SMALL else 1.0 for a in ars]  # noqa: E731

    print("Low German, Adler et al. 2016 Abb. 10, 'sehr gut' x 16+ share x German speakers in the area")
    for land, (sg, g, node, pref, minus) in LOW_GERMAN.items():
        ars = [a for a in gpop.index if a.startswith(land)]
        if pref:
            ars = [a for a in ars if a[:len(pref[0])] in pref or a[:5] in pref]
            ars = [a for a in ars if a[:5] not in minus]
        area_pop = gpop[ars].sum()
        ger_area = german[land] * area_pop / land_pop[land]
        n = sg / 100 * ADULT * ger_area
        name = f"nds{land}"
        print(f"  {land}: {len(ars):>4} Gemeinden, {area_pop / 1e6:5.2f}M people "
              f"({100 * area_pop / land_pop[land]:3.0f}% of the Land); {sg}% -> {n:9,.0f}"
              f"   ((sehr) gut {sg + g:.1f}% would be {(sg + g) / 100 * ADULT * ger_area:9,.0f})")
        rows.append(dict(geo_id=land, node=node, count=round(n), area=name,
                         source="Adler et al. 2016, Plattdeutsch 'sehr gut'"))
        areas += [dict(ars=a, area=name, factor=f) for a, f in zip(ars, rural(ars))]

    for node, land, count, name, rule in ESTIMATES:
        if name == "hsb":
            core = find(nm, "14", HSB_CORE)
            rest = find(nm, "14", HSB_REST)
            n_core = round(HSB_CORE_SHARE * gpop[core].sum())
            assert n_core < count
            rows.append(dict(geo_id=land, node=node, count=n_core, area="hsb_core",
                             source="Sorbisches Kulturlexikon 2014"))
            rows.append(dict(geo_id=land, node=node, count=count - n_core, area="hsb",
                             source="Sorbisches Kulturlexikon 2014"))
            areas += [dict(ars=a, area="hsb_core", factor=1.0) for a in core]
            areas += [dict(ars=a, area="hsb", factor=f) for a, f in zip(rest, rural(rest))]
            print(f"  Upper Sorbian: core {len(core)} Gemeinden, {gpop[core].sum():,} people -> "
                  f"{n_core:,}; rest {len(rest)} Gemeinden, {gpop[rest].sum():,} people -> "
                  f"{count - n_core:,}")
            continue
        if name == "dsb":
            ars = find(nm, "12", DSB_AREA)
        elif name == "frr":
            foehr = [a for a in gpop.index if a.startswith("010545488")]
            assert len(foehr) == 15, len(foehr)   # 11 on Foehr, Wyk, 3 on Amrum
            ars = [a for a in gpop.index if a.startswith("01054") and a not in foehr]
            ars += ["010560025025"]   # Helgoland
            assert nm["010560025025"].startswith("Helgoland")
            rows.append(dict(geo_id=land, node=node, count=FRR_FOEHR, area="frr_foehr",
                             source="Land Schleswig-Holstein; Foehr-Amrum estimate"))
            areas += [dict(ars=a, area="frr_foehr", factor=1.0) for a in foehr]
            count = count - FRR_FOEHR
        elif name == "stq":
            ars = find(nm, "03", ["Saterland"])
        elif name == "dan":
            ars = [a for a in gpop.index if a[:5] in ("01001", "01059", "01054", "01058")]
        fac = rural(ars) if "rural" in rule else [1.0] * len(ars)
        areas += [dict(ars=a, area=name, factor=f) for a, f in zip(ars, fac)]
        rows.append(dict(geo_id=land, node=node, count=count, area=name, source=rule))
        print(f"  {name}: {len(ars)} Gemeinden, {gpop[ars].sum():,} people -> {count:,} ({rule})")

    out = pd.DataFrame(rows)
    ar = pd.DataFrame(areas)
    assert not ar.duplicated(["ars", "area"]).any()
    assert set(out["area"]) == set(ar["area"])
    # every count is taken out of its Land's German: it must leave plenty
    by_land = out.groupby("geo_id")["count"].sum()
    for land, n in by_land.items():
        assert n < 0.25 * german[land], (land, n, german[land])
        print(f"  Land {land}: {n:,.0f} out of German {german[land]:,.0f} ({100 * n / german[land]:.1f}%)")
    print(f"total {out['count'].sum():,.0f}; by node:")
    print(out.groupby("node")["count"].sum().round().astype(int).to_string())
    out.to_csv(OUT, index=False)
    ar.to_csv(OUT_AREAS, index=False)
    print(f"wrote {OUT} ({len(out)} rows) and {OUT_AREAS} ({len(ar)} rows)")


if __name__ == "__main__":
    sys.exit(main())
