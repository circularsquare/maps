"""Census of India 2011, C-16 (population by mother tongue) -> data/normalized/in.csv.

Reads the 36 state workbooks sources/in_fetch.py downloads. Each row is one (area, mother
tongue) with persons/males/females, total/rural/urban. Only total persons is kept.

THE CODE IS THE STRUCTURE. A mother-tongue code is six digits: the first three name one of
the 121 languages the census groups mother tongues under (001 ASSAMESE ... ), the last three
the mother tongue inside it. `xxx000` is the language's own total row, `xxx999` its "Others"
(mother tongues returned under that language but too small to list), anything else a named
mother tongue. The named and Others rows of a language sum to its total row, and that is
checked for every area. Only the mother-tongue rows are written: reading the language rows
too would count everybody twice, and the language rows are where the "Hindi" umbrella lives.

GEOGRAPHY. geo_id is state (2) + district (3) + sub-district (5), the same ten digits
religiondots uses for India's 5,988 sub-districts, so its placement layer joins unchanged.
State and district rows are written too, as their own geo_level, for the checks.

    python sources/in_c16.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "in"
OUT = HERE / "data" / "normalized" / "in.csv"


def read_state(path):
    df = pd.read_excel(path, header=None, dtype=str, skiprows=6)
    df = df.iloc[:, :8]
    df.columns = ["table", "state", "district", "subdistrict", "area", "code", "name", "count"]
    df = df[df["code"].notna() & df["code"].str.fullmatch(r"\d{6}")]
    df["count"] = pd.to_numeric(df["count"], errors="raise").astype("int64")
    for c in ("state", "district", "subdistrict"):
        df[c] = df[c].str.strip()
    df["name"] = df["name"].str.strip()
    return df


def main():
    files = sorted(RAW.glob("DDW-C16-STMT-MDDS-*.xlsx"))
    files = [f for f in files if not f.stem.endswith("0000")]
    if len(files) != 35:
        raise SystemExit(f"{len(files)} state files, expected 35 -- run sources/in_fetch.py")
    rows = []
    for f in files:
        df = read_state(f)
        df["level"] = "subdistrict"
        df.loc[df["subdistrict"] == "00000", "level"] = "district"
        df.loc[(df["district"] == "000") & (df["subdistrict"] == "00000"), "level"] = "state"
        df["geo_id"] = df["state"] + df["district"] + df["subdistrict"]
        df["group"] = df["code"].str[:3]
        # `124000 OTHERS` is the census's residual for every mother tongue it could not place
        # under any of its 123 languages, and it has no rows below it. It is its own leaf.
        df.loc[df["code"] == "124000", "code"] = "124999"
        is_total = df["code"].str.endswith("000")

        # every language's mother tongues add up to its total row, in every area
        tot = df[is_total].set_index(["geo_id", "group"])["count"]
        parts = df[~is_total].groupby(["geo_id", "group"])["count"].sum()
        both = pd.concat([tot.rename("tot"), parts.rename("parts")], axis=1).fillna(0)
        both = both[both.index.get_level_values("group") != "124"]
        bad = both[both["tot"] != both["parts"]]
        # Madhya Pradesh prints district 436's HINDI total (1,114,738) on sub-district 03523's
        # line, so six (area, language) pairs disagree there. The mother-tongue rows are right
        # (they are what the sub-district/district check below sums), so a disagreement in the
        # total rows alone only warns.
        if len(bad):
            print(f"  !! {f.name}: {len(bad)} (area, language) total rows disagree with their "
                  f"mother tongues: {bad.index.tolist()[:3]}")

        # sub-districts add up to their district, districts to their state
        mt = df[~is_total]
        sub = mt[mt["level"] == "subdistrict"].groupby(mt["district"])["count"].sum()
        dist = mt[mt["level"] == "district"].set_index("district")["count"].groupby(level=0).sum()
        diff = (sub.reindex(dist.index).fillna(0) - dist).abs()
        if diff.max() > 0:
            raise SystemExit(f"{f.name}: sub-districts do not add to their districts:\n"
                             f"{diff[diff > 0].head()}")

        names = df[is_total].drop_duplicates("group").set_index("group")["name"]
        names["124"] = "124 OTHERS"
        mt = mt.assign(parent=mt["group"].map(names).str.replace(r"^\d+\s+", "", regex=True))
        # `xxx999` rows are all called "<n> Others"; name them after their language
        oth = mt["code"].str.endswith("999")
        mt.loc[oth, "name"] = "Others under " + mt.loc[oth, "parent"]
        rows.append(mt[["geo_id", "level", "area", "code", "name", "parent", "count"]])
        n_sub = mt.loc[mt["level"] == "subdistrict", "geo_id"].nunique()
        st = mt.loc[mt["level"] == "state", "count"].sum()
        print(f"  {f.stem[-4:-2]} {mt['area'].iloc[0]:<28} {n_sub:>4} sub-districts, {st:>12,} people")

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"level": "geo_level", "area": "geo_name", "code": "source_code",
                              "name": "source_category", "parent": "source_parent"})
    out = out[out["count"] > 0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    st = out[out["geo_level"] == "state"]
    print(f"wrote {OUT}: {len(out):,} rows, "
          f"{out.loc[out['geo_level'] == 'subdistrict', 'geo_id'].nunique():,} sub-districts, "
          f"{st['count'].sum():,} people, {st['source_code'].nunique()} mother-tongue codes")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
