"""Count far-pulling POI kinds in the cached OSM area and point extracts (read-only, one core).
Reads the needed columns without geometry and filters in pandas."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import json
import pyogrio

OSM = r"C:\Users\anita\projects\maps\trainworld\data\raw\osm"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "trainworld", "T-263-poi_counts.json")


def tag(ot, key, val=None):
    s = ot.fillna("")
    pat = f'"{key}"=>' + (f'"{val}"' if val else "")
    return s.str.contains(pat, regex=False)


def area_counts(path):
    info = pyogrio.read_info(path)
    cols = [c for c in ["leisure", "tourism", "natural", "shop", "amenity", "landuse", "other_tags"] if c in info["fields"]]
    df = pyogrio.read_dataframe(path, columns=cols, read_geometry=False)
    ot = df["other_tags"].fillna("")
    wd = tag(ot, "wikidata")
    cap = tag(ot, "capacity")
    col = lambda c: df[c].fillna("") if c in df else ot.where(False, "")  # noqa: E731
    kinds = {
        "stadium": col("leisure") == "stadium",
        "theme_park": col("tourism") == "theme_park",
        "water_park": col("leisure") == "water_park",
        "zoo_aquarium": col("tourism").isin(["zoo", "aquarium"]),
        "museum": col("tourism") == "museum",
        "attraction": col("tourism") == "attraction",
        "beach": col("natural") == "beach",
        "mall": col("shop") == "mall",
        "ski": col("landuse") == "winter_sports",
        "expo": col("amenity").isin(["exhibition_centre", "conference_centre", "events_venue"]),
        "worship": col("amenity") == "place_of_worship",
    }
    r = {"rows": len(df), "fields": cols}
    for k, m in kinds.items():
        r[k] = [int(m.sum()), int((m & wd).sum()), int((m & cap).sum())]
    return r


def point_counts(path):
    df = pyogrio.read_dataframe(path, columns=["other_tags"], read_geometry=False)
    ot = df["other_tags"].fillna("")
    wd = tag(ot, "wikidata")
    kinds = {
        "stadium": tag(ot, "leisure", "stadium"),
        "theme_park": tag(ot, "tourism", "theme_park"),
        "zoo_aquarium": tag(ot, "tourism", "zoo") | tag(ot, "tourism", "aquarium"),
        "museum": tag(ot, "tourism", "museum"),
        "attraction": tag(ot, "tourism", "attraction"),
        "mall": tag(ot, "shop", "mall"),
    }
    r = {"rows": len(df)}
    for k, m in kinds.items():
        r[k] = [int(m.sum()), int((m & wd).sum())]
    return r


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for stem in sys.argv[1:]:
        r = {}
        ap = os.path.join(OSM, stem + ".areas.fgb")
        pp = os.path.join(OSM, stem + ".points.fgb")
        r["areas [n, with wikidata, with capacity]"] = area_counts(ap)
        if os.path.exists(pp):
            r["points [n, with wikidata]"] = point_counts(pp)
        res[stem] = r
        print(stem, json.dumps(r), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
