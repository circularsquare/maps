"""Canada — Census of Population 2021 (Statistics Canada). Religion, subnational.

Reads the three raw downloads in data/raw/ca/ and writes data/normalized/ca.csv.

Three StatCan products are combined, because no single one has both the fine
geography and the deep classification:

  98-10-0345  Religion (168) x Canada / provinces / CMAs / CAs (+ their parts).
              The deepest religion classification StatCan disseminates.
  98-401-X2021005  Census Profile, Religion (25) x Canada / provinces / census
              divisions / census subdivisions.  CD + CSD rows are taken.
  98-401-X2021007  Census Profile, Religion (25) x CMAs / CAs / census tracts.
              CT rows only are taken.

Country and province rows come only from 98-10-0345, whose 168 categories are a
strict superset of the profile's 25, so nothing is duplicated at those levels.
See sources/ca.md for vintage, nesting, suppression and licence.

Run:  python sources/ca.py
"""

import csv
import io
import os
import re
import sys
import zipfile
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ca")
OUT = os.path.join(ROOT, "data", "normalized", "ca.csv")

Z_345 = os.path.join(RAW, "98100345-eng.zip")
Z_005 = os.path.join(RAW, "98-401-X2021005_English_CSV.zip")
Z_007 = os.path.join(RAW, "98-401-X2021007_English_CSV.zip")

SOURCE_ID = "ca_census_2021"
YEAR = "2021"
BASIS = "self_id"  # every row: the census asked the respondent (spec.md 3.1)

# Census Profile characteristic ids for the religion block (collapsed list of 25).
REL_ID_MIN, REL_ID_MAX = 1949, 1973

# GEO_LEVEL string in the profile -> our geo_level.  Provinces and territories
# are one level in the Standard Geographical Classification; StatCan labels them
# separately only for presentation.
PROFILE_LEVEL = {
    "Country": "country",
    "Province": "province",
    "Territory": "province",
    "Census division": "cd",
    "Census subdivision": "csd",
    "Census metropolitan area": "cma",
    "Census agglomeration": "ca",
    "Census tract": "ct",
}

# DGUID schema code -> our geo_level, for table 98-10-0345.
DGUID_LEVEL = {
    "A0000": "country",
    "A0002": "province",
    "S0503": "cma",
    "S0504": "ca",
    "S0505": "cma_part",
    "S0506": "ca_part",
}


def zip_member(z, suffix):
    for n in z.namelist():
        if n.lower().endswith(suffix):
            return n
    raise KeyError(suffix + " not in " + str(z.namelist()))


# ---------------------------------------------------------------- 98-10-0345

def read_345_religion_tree():
    """{member_id: (name, parent_id)} for the Religion (168) dimension."""
    with zipfile.ZipFile(Z_345) as z:
        txt = z.read(zip_member(z, "_metadata.csv")).decode("utf-8-sig", "replace")
    tree = OrderedDict()
    for r in csv.reader(io.StringIO(txt)):
        # member block rows: dimension id, name, code, member id, parent id, ...
        if len(r) >= 5 and r[0].strip() == "4" and r[3].strip().isdigit():
            tree[int(r[3])] = (r[1], int(r[4]) if r[4].strip() else None)
    if len(tree) != 168:
        raise SystemExit("expected 168 religion members, got %d" % len(tree))
    return tree


def read_345():
    """Yield normalized rows from 98-10-0345, and return them plus a lookup."""
    tree = read_345_religion_tree()
    parent_of_name = {}
    for mid, (name, pid) in tree.items():
        parent_of_name[name] = tree[pid][0] if pid else ""

    rows = []
    with zipfile.ZipFile(Z_345) as z:
        name = zip_member(z, ".csv")
        # the data file is the one that is not the metadata file
        name = [n for n in z.namelist() if not n.lower().endswith("_metadata.csv")][0]
        with z.open(name) as f:
            rd = csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
            hdr = next(rd)
            i_geo = hdr.index("GEO")
            i_dg = hdr.index("DGUID")
            i_age = hdr.index("Age (15C)")
            i_gen = hdr.index("Gender (3)")
            i_rel = hdr.index("Religion (168)")
            # first data column = "Total - Immigrant status and period of immigration"
            i_val = hdr.index([h for h in hdr if h.startswith(
                "Immigrant status and period of immigration (11):Total")][0])
            i_sym = i_val + 1
            for r in rd:
                if len(r) <= i_sym:
                    continue
                if r[i_age] != "Total - Age" or r[i_gen] != "Total - Gender":
                    continue
                dguid = r[i_dg]
                schema = dguid[4:9]
                level = DGUID_LEVEL.get(schema)
                if level is None:
                    raise SystemExit("unknown DGUID schema %r in 98-10-0345" % schema)
                cat = r[i_rel]
                note = "product=98-10-0345;parent=%s" % parent_of_name.get(cat, "")
                val, sym = r[i_val].strip(), r[i_sym].strip()
                if not val:
                    note += ";suppressed=%s" % (sym or "blank")
                rows.append([dguid, level, r[i_geo], cat, val, BASIS, YEAR, SOURCE_ID, note])
    return rows, tree


# ------------------------------------------------------------ Census Profile

def read_profile_tree(zpath):
    """Parent map for characteristics 1949-1973, from the zip's own meta file.

    The meta file indents characteristic names two spaces per level of nesting;
    that indentation is the only machine-readable statement of the hierarchy the
    profile makes, and the data CSV does not carry it.
    """
    with zipfile.ZipFile(zpath) as z:
        txt = z.read(zip_member(z, "_meta.txt")).decode("latin-1")
    names, indents = {}, {}
    for line in txt.splitlines():
        m = re.match(r"^(\d+)\t(.*)$", line)
        if not m:
            continue
        cid = int(m.group(1))
        if not (REL_ID_MIN <= cid <= REL_ID_MAX):
            continue
        raw = m.group(2).rstrip()
        # strip the trailing footnote marker, e.g. "... 25% sample data (161)"
        clean = re.sub(r"\s*\(\d+\)\s*$", "", raw).strip()
        names[cid] = clean
        indents[cid] = len(raw) - len(raw.lstrip(" "))
    if len(names) != 25:
        raise SystemExit("expected 25 profile religion characteristics, got %d" % len(names))

    parent = {}
    stack = []  # (indent, cid)
    for cid in sorted(names):
        ind = indents[cid]
        while stack and stack[-1][0] >= ind:
            stack.pop()
        parent[cid] = names[stack[-1][1]] if stack else ""
        stack.append((ind, cid))
    return names, parent


def read_profile(zpath, product, keep_levels):
    names, parent = read_profile_tree(zpath)
    rows = []
    with zipfile.ZipFile(zpath) as z:
        data = zip_member(z, "_data.csv")
        with z.open(data) as f:
            rd = csv.reader(io.TextIOWrapper(f, encoding="latin-1", newline=""))
            next(rd)
            for r in rd:
                if len(r) < 13:
                    continue
                cid = r[8]
                if not cid.isdigit():
                    continue
                cid = int(cid)
                if not (REL_ID_MIN <= cid <= REL_ID_MAX):
                    continue
                level = PROFILE_LEVEL.get(r[3])
                if level is None:
                    raise SystemExit("unknown profile GEO_LEVEL %r" % r[3])
                if level not in keep_levels:
                    continue
                cat = names[cid]
                # dq  = the 5-digit DATA_QUALITY_FLAG (see the zip's meta file:
                #       digit 1 incomplete enumeration, digit 4 long-form quality,
                #       9 = suppressed under the Statistics Act)
                # tnr = long-form total non-response rate, %, for the whole unit.
                #       StatCan stopped suppressing on quality in 2021, so a high
                #       TNR is published rather than withheld — spec.md 7 wants it.
                note = "product=%s;alt=%s;parent=%s;dq=%s;tnr_lf=%s" % (
                    product, r[2], parent[cid], r[7], r[6])
                val, sym = r[11].strip(), r[12].strip()
                if not val:
                    note += ";suppressed=%s" % (sym or "blank")
                rows.append([r[1], level, r[4], cat, val, BASIS, YEAR, SOURCE_ID, note])
    return rows


# --------------------------------------------------------------- validation

def reconcile(rows_345, tree, rows_005, rows_007):
    def canada_345():
        out = {}
        for r in rows_345:
            if r[1] == "country":
                out[r[3]] = int(r[4]) if r[4] else None
        return out

    ca = canada_345()
    total = ca["Total - Religion"]
    print()
    print("=" * 74)
    print("NATIONAL RECONCILIATION — Canada, 2021, population in private households")
    print("=" * 74)
    print("Total - Religion (98-10-0345, Canada) = %s" % f"{total:,}")

    # 1. does every parent equal the sum of its children, in the 168-tree?
    kids = defaultdict(list)
    for mid, (name, pid) in tree.items():
        if pid:
            kids[tree[pid][0]].append(name)
    bad = 0
    for parent_name, children in kids.items():
        p = ca[parent_name]
        s = sum(ca[c] for c in children)
        if p != s:
            bad += 1
            print("  RESIDUAL %-45s parent=%-12s children=%-12s diff=%+d"
                  % (parent_name, f"{p:,}", f"{s:,}", s - p))
    print("nesting check (168-category tree): %d of %d parent nodes differ from "
          "the sum of their children" % (bad, len(kids)))

    # 2. do the 13 province rows sum to the Canada row?
    prov = defaultdict(int)
    nprov = set()
    for r in rows_345:
        if r[1] == "province" and r[4]:
            prov[r[3]] += int(r[4])
            nprov.add(r[0])
    print("provinces+territories: %d; sum of their 'Total - Religion' = %s (Canada %s, diff %+d)"
          % (len(nprov), f"{prov['Total - Religion']:,}", f"{total:,}",
             prov["Total - Religion"] - total))

    # 3. do CSD rows sum to Canada?  (they will not, exactly: suppression)
    for label, rows, lvl in (("CSD", rows_005, "csd"), ("CD", rows_005, "cd"),
                             ("CT", rows_007, "ct")):
        agg = defaultdict(int)
        units, supp = set(), set()
        for r in rows:
            if r[1] != lvl:
                continue
            units.add(r[0])
            if r[4]:
                agg[r[3]] += int(r[4])
            else:
                supp.add(r[0])
        tname = [n for n in agg if n.startswith("Total - Religion")][0]
        tnr = {}
        for r in rows:
            if r[1] != lvl:
                continue
            for tok in r[8].split(";"):
                if tok.startswith("tnr_lf="):
                    try:
                        tnr[r[0]] = float(tok[7:])
                    except ValueError:
                        pass
        hi = sum(1 for v in tnr.values() if v >= 50)
        print("%-4s units=%-6d fully suppressed=%-5d sum of totals=%-12s "
              "(%.3f%% of Canada);  units with long-form TNR >= 50%%: %d"
              % (label, len(units), len(supp), f"{agg[tname]:,}",
                 100.0 * agg[tname] / total, hi))

    # 4. profile Canada row vs 98-10-0345 Canada row, category by category
    print()
    print("cross-product check — Census Profile 98-401-X2021005 Canada row vs 98-10-0345:")
    prof_ca = {}
    with zipfile.ZipFile(Z_005) as z:
        names, _ = read_profile_tree(Z_005)
    rows_prof_ca = read_profile(Z_005, "98-401-X2021005", {"country"})
    for r in rows_prof_ca:
        prof_ca[r[3]] = int(r[4]) if r[4] else None
    diffs = 0
    for cat, v in prof_ca.items():
        key = "Total - Religion" if cat.startswith("Total - Religion") else cat
        if key not in ca:
            print("   category not in 168-list: %r" % cat)
            continue
        if ca[key] != v:
            diffs += 1
            print("   %-50s profile=%-12s 0345=%-12s diff=%+d"
                  % (cat, f"{v:,}", f"{ca[key]:,}", v - ca[key]))
    print("   %d of %d shared categories differ" % (diffs, len(prof_ca)))

    print()
    print("national counts, 168-category list, largest 25:")
    leaves = sorted(((v, k) for k, v in ca.items() if k != "Total - Religion"),
                    reverse=True)
    for v, k in leaves[:25]:
        print("   %-55s %12s  %5.2f%%" % (k, f"{v:,}", 100.0 * v / total))


def main():
    for p in (Z_345, Z_005, Z_007):
        if not os.path.exists(p):
            raise SystemExit("missing raw file: %s  (see sources/ca.md)" % p)

    print("reading 98-10-0345 (Religion 168) ...")
    rows_345, tree = read_345()
    print("   %d rows" % len(rows_345))

    print("reading 98-401-X2021005 (CD + CSD) ... this reads a 2.6 GB CSV, ~2 min")
    rows_005 = read_profile(Z_005, "98-401-X2021005", {"cd", "csd"})
    print("   %d rows" % len(rows_005))

    print("reading 98-401-X2021007 (CT) ... this reads a 2.6 GB CSV, ~2 min")
    rows_007 = read_profile(Z_007, "98-401-X2021007", {"ct"})
    print("   %d rows" % len(rows_007))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category",
                    "count", "basis", "year", "source_id", "note"])
        for rows in (rows_345, rows_005, rows_007):
            w.writerows(rows)
    n = len(rows_345) + len(rows_005) + len(rows_007)
    print("wrote %s  (%d rows)" % (OUT, n))

    reconcile(rows_345, tree, rows_005, rows_007)


if __name__ == "__main__":
    main()
