"""
One coarse polygon per country on the map, for the coverage wash (spec §6.12).

NATURAL EARTH 10m, AND IT STARTED AT 110m, WHICH WAS WRONG. The first version reasoned that
the wash answers a continental-scale question, fades out by z7, and therefore needs to be
recognisable rather than accurate. The second half of that does not follow, and the
Philippines disproved it on sight: **110m gives this country SEVEN polygons and 110
vertices** for an archipelago of seven thousand islands, so at z6 the wash was straight
lines cutting across the Visayas with dots on both sides of them. A reader cannot tell a
deliberately coarse boundary from a broken one, and a layer whose job is to say "this
country is in scope" fails the moment it stops looking like the country.

10m gives the Philippines 97 parts and 7,406 vertices — 67x the detail — and costs 3.3 MB
across twenty countries against 59 KB. That is the real trade, and it is worth it: this is
one fetch, cached, gzipped on the wire, and smaller than the coarse dot buffers.

DISSOLVING THE PLACEMENT LAYERS would match the dots exactly and is still not worth it. It
was measured: the UK's 239,023 units dissolve to 4,565 parts and 2.2 MB, the Philippines'
barangays to 1,915 parts and 1.3 MB, at ~70 s per country — call it 20 MB and several
minutes of build. 10m sits close enough to the coastline that the remaining error is
smaller than the wash's own soft edge, so the exact version would buy very little for 6x
the weight.

SIMPLIFY IS WHAT MAKES 10m AFFORDABLE, and it is topology-preserving so islands stay
islands. At ~200 m it is well under a pixel everywhere this draws.

Usage:
    python country_shapes.py        # data/geo/ne_10m_... -> data/processed/country_shapes.geojson
"""
import json
import sys
from pathlib import Path

import shapely
import shapely.geometry

HERE = Path(__file__).parent
RES = "10m"
SRC = HERE / "data" / "geo" / f"ne_{RES}_admin_0_countries.geojson"
# The UK is drawn as its three CENSUSES rather than as one country (coverage.py UK_REGIONS,
# spec §6.12): England and Wales publish no Christian denomination, Scotland names two
# bodies, Northern Ireland names twenty-two, so one shape cannot be honestly lit or unlit.
# Natural Earth's map_units file is the same 10m geometry cut into constituent countries.
UNITS = HERE / "data" / "geo" / f"ne_{RES}_admin_0_map_units.geojson"
UNITS_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
             f"ne_{RES}_admin_0_map_units.geojson")
# region id -> the GU_A3 map units it is made of
SPLIT = {"uk": {"uk-ew": ("ENG", "WLS"), "uk-sc": ("SCT",), "uk-ni": ("NIR",)}}
OUT = HERE / "data" / "processed" / "country_shapes.geojson"
URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
       f"ne_{RES}_admin_0_countries.geojson")

# ~200 m. Well under a pixel at the zooms this draws, and it is what keeps an archipelago
# affordable: the cut is 4x smaller simplified than raw, with no visible difference.
SIMPLIFY = 0.002

# ---------------------------------------------------------------------------------------
# CLIP — territory inside a country's Natural Earth outline that the source does not cover
# ---------------------------------------------------------------------------------------
# The wash says "this country is in scope". Where part of it is NOT in scope, a wash that
# covers it says the opposite of the truth, and the dots inside it are missing rather than
# absent. Anita, 2026-09-06, on Georgia: the lit shape ran over Abkhazia and South Ossetia
# and made two unenumerated territories look like two empty ones.
#
# NEITHER `admin_0_countries` NOR `admin_0_map_units` CAN DO THIS — the first has Georgia as
# one de jure polygon and the second splits it into Georgia and Adjara and stops. The layer
# that can is **`ne_10m_admin_0_disputed_areas`**, which carries Abkhazia (8,652 km²) and
# South Ossetia (4,463 km²) as `TYPE == "Breakaway"` features with `SOV_A3 == "GEO"`, keyed
# by `BRK_NAME`.
#
# THE FIRST ATTEMPT USED THE COUNTRY'S OWN BOUNDARY BUILD AND WAS WRONG IN A WAY THAT LOOKED
# RIGHT. Subtracting geoBoundaries' Abkhazia ADM1 plus the Java and Akhalgori ADM2 units cut
# 14.8% and passed every spot check but one: **Tskhinvali stayed inside**, because
# geoBoundaries has no ADM2 for Tskhinvali or Znauri at all and those districts therefore sit
# in no polygon anywhere in that file. A cut assembled from named sub-units is only as
# complete as the file's own coverage, and a gap in it is invisible. Prefer a layer that
# names the territory you are actually removing.
#
# It is also the same publisher and vintage as the outline being cut, so the two agree along
# the coast and no slivers are produced at all — where geoBoundaries left nine.
#
# Entries are (relative path, property, names). A missing file raises rather than warns: a
# silently uncut wash is exactly the failure this exists to remove.
DISPUTED = HERE / "data" / "geo" / f"ne_{RES}_admin_0_disputed_areas.geojson"
DISPUTED_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
                f"geojson/ne_{RES}_admin_0_disputed_areas.geojson")
CLIP = {
    "ge": [(DISPUTED, "BRK_NAME", ("Abkhazia", "South Ossetia"))],
}

# Kept at zero cost in case a future cut does come from a different publisher: parts smaller
# than this are dropped as artefacts of two files disagreeing along a shared edge. With the
# disputed-areas layer there are none.
SLIVER_KM2 = 5.0

# Our country codes are ISO 3166-1 alpha-2 lowercased, and Natural Earth agrees for all but
# one: the United Kingdom is GB. Kept as an explicit table rather than an upper() call, so a
# future country whose code does not match is a KeyError here and not a silently missing wash.
ISO = {"uk": "GB"}

ROUND = 3          # ~110 m at the equator, well under this layer's own error


def _round(obj):
    if isinstance(obj, list):
        return [_round(o) for o in obj]
    if isinstance(obj, float):
        return round(obj, ROUND)
    return obj


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}\n  curl -sSL -o {SRC} {URL}")
    from countries import COUNTRIES

    ne = json.loads(SRC.read_text(encoding="utf-8"))
    want = {ISO.get(cc, cc.upper()): cc for cc in COUNTRIES}

    def _clip(cc, geom):
        """Subtract territory the source does not cover. See CLIP."""
        if cc not in CLIP:
            return geom
        cuts = []
        for path, prop, names in CLIP[cc]:
            if not path.exists():
                raise SystemExit(
                    f"{cc}: missing {path}, which CLIP needs to cut territory the source "
                    f"does not cover out of the wash.\n"
                    f"  curl -sSL -o {path} {DISPUTED_URL}")
            src = json.loads(path.read_text(encoding="utf-8"))
            found = {}
            for f in src["features"]:
                nm = str(f["properties"].get(prop) or "").strip()
                if nm in names:
                    found[nm] = shapely.geometry.shape(f["geometry"])
            missing_n = [n for n in names if n not in found]
            if missing_n:
                raise SystemExit(f"{cc}: {path.name} has no {prop} in {missing_n} — the "
                                 "boundary file has been revised; fix CLIP rather than "
                                 "letting the wash silently cover it again")
            cuts.extend(found.values())

        before = shapely.area(geom)
        out = shapely.difference(geom, shapely.union_all(cuts))
        if not shapely.is_valid(out):
            out = shapely.make_valid(out)
        # Drop the slivers the two files' disagreement leaves behind. Degrees squared is
        # fine for a threshold this coarse; ~1 deg^2 is ~12,300 km^2 at 42 N.
        deg2 = SLIVER_KM2 / 12_300.0
        parts = [p for p in shapely.get_parts(out) if shapely.area(p) >= deg2]
        dropped = len(shapely.get_parts(out)) - len(parts)
        out = shapely.union_all(parts) if parts else out
        print(f"  {cc}: clipped {100 * (1 - shapely.area(out) / before):.1f}% of the "
              f"Natural Earth outline ({sum(len(n) for _, _, n in CLIP[cc])} polygons"
              + (f", {dropped} slivers dropped" if dropped else "") + ")")
        return out

    def emit(feats, cc, rg, geom):
        geom = _clip(cc, geom)
        # Simplify with topology preserved, so islands stay islands and a coastline cannot
        # self-intersect into an invalid ring the fill would drop.
        g = shapely.simplify(geom, SIMPLIFY)
        if not shapely.is_valid(g):
            g = shapely.make_valid(g)
        feats.append({"type": "Feature", "properties": {"cc": cc, "rg": rg},
                      "geometry": _round(json.loads(shapely.to_geojson(g)))})

    feats, seen = [], set()
    for f in ne["features"]:
        p = f["properties"]
        # NE carries -99 in ISO_A2 for a handful of countries and puts the real code in
        # ISO_A2_EH; check both rather than trusting either.
        for key in ("ISO_A2", "ISO_A2_EH"):
            code = str(p.get(key) or "").strip()
            if code in want:
                cc = want[code]
                if cc in seen:
                    break
                seen.add(cc)
                if cc not in SPLIT:
                    emit(feats, cc, cc, shapely.geometry.shape(f["geometry"]))
                break

    # The split countries, from the map-units file, one feature per census rather than one
    # per country.
    if SPLIT:
        if not UNITS.exists():
            raise SystemExit(f"missing {UNITS}\n  curl -sSL -o {UNITS} {UNITS_URL}")
        mu = json.loads(UNITS.read_text(encoding="utf-8"))
        by_unit = {}
        for f in mu["features"]:
            gu = str(f["properties"].get("GU_A3") or "").strip()
            if gu:
                by_unit[gu] = shapely.geometry.shape(f["geometry"])
        for cc, parts in SPLIT.items():
            if cc not in want.values():
                continue
            for rg, units in parts.items():
                missing_u = [u for u in units if u not in by_unit]
                if missing_u:
                    raise SystemExit(f"no map unit {missing_u} for {rg} — Natural Earth's "
                                     "GU_A3 codes changed; see SPLIT in country_shapes.py")
                geoms = [by_unit[u] for u in units]
                emit(feats, cc, rg,
                     geoms[0] if len(geoms) == 1 else shapely.union_all(geoms))

    missing = sorted(set(COUNTRIES) - seen)
    if missing:
        raise SystemExit(f"no Natural Earth polygon for {missing} — check the ISO table "
                         "in country_shapes.py")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                              separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.name}  ({len(feats)} countries, {OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
