"""
Validate the taxonomy and emit religions.json.

Checks, all of which have already caught something:
  1. every leaf's parent branch exists in branches.py      (no silent node creation, §2)
  2. no duplicate group code in a source map               (a dict literal hides these)
  3. no duplicate leaf id
  4. every group code in the source data is mapped or explicitly unmapped
  5. no mapped code that the source data does not contain  (typo detector)

Then writes:
  taxonomy/religions.json   the tree
  taxonomy/usrc_groups.csv  the same file with the `path` column filled

Run: python taxonomy/build_tree.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from branches import BRANCHES, LINEAGE             # noqa: E402
from usrc2020 import MAP, UNMAPPED, REVIEW         # noqa: E402
import cz2021                                      # noqa: E402
import br2010                                      # noqa: E402
import au2021, ie2022, mx2020, nz2023, uk2021      # noqa: E402

RAW = HERE.parent / "data" / "raw" / "2020_USRC_Group_Detail.xlsx"
CZ = HERE.parent / "data" / "normalized" / "cz.csv"
BR = HERE.parent / "data" / "normalized" / "br.csv"
# the three that arrive already allocated — the ALLOCATED file is the one whose
# categories must all be mapped, since that is what countries.py reads.
ALLOCATED = {
    "au2021": (au2021, HERE.parent / "data" / "normalized" / "au_sa2_allocated.csv"),
    "ie2022": (ie2022, HERE.parent / "data" / "normalized" / "ie_small_area_allocated.csv"),
    "mx2020": (mx2020, HERE.parent / "data" / "normalized" / "mx_municipio_allocated.csv"),
    "nz2023": (nz2023, HERE.parent / "data" / "normalized" / "nz_sa2_allocated.csv"),
}
UK = HERE.parent / "data" / "normalized" / "uk.csv"
UK_EW = HERE.parent / "data" / "normalized" / "uk_ew_allocated.csv"
UK_NI = HERE.parent / "data" / "normalized" / "uk_ni_allocated.csv"
CSV = HERE / "usrc_groups.csv"
OUT = HERE / "religions.json"

problems = []


def check_duplicate_keys(path: Path, dict_name: str):
    """A Python dict literal silently keeps the last of a duplicated key. Read the text."""
    src = path.read_text(encoding="utf-8")
    start = src.find(f"{dict_name} = {{")
    if start < 0:
        return
    body = src[start:]
    keys = re.findall(r'^\s{4}"([^"]+)":', body, re.M)
    seen, dupes = set(), []
    for k in keys:
        if k in seen:
            dupes.append(k)
        seen.add(k)
    for k in dupes:
        problems.append(f"duplicate group code {k!r} in {path.name} {dict_name}")


def check_lineage(nodes):
    """Validate LINEAGE against the tree that was just built, and flatten it for emit.

    The check that earns its place is the LAST one: every child of a parent that has a
    lineage must appear in it. Without that, adding a branch to BRANCHES silently drops it
    to the end of the panel and the end of the hue wheel — it would still draw, still be
    the right size, and just quietly not be where it descends. That is exactly the kind of
    failure this file exists to make loud (§2.4).
    """
    kids = {}
    for n in nodes:
        if "." in n["id"]:
            kids.setdefault(n["id"].rsplit(".", 1)[0], []).append(n["id"])
    known = {n["id"] for n in nodes}

    out = {}
    for parent, groups in LINEAGE.items():
        if parent not in known:
            problems.append(f"lineage: {parent!r} is not a node")
            continue
        listed, order = [], []
        for label, ids in groups:
            for nid in ids:
                if nid not in known:
                    problems.append(f"lineage {parent}: {nid!r} is not a node")
                elif nid.rsplit(".", 1)[0] != parent:
                    problems.append(f"lineage {parent}: {nid!r} is not a child of it")
                elif nid in listed:
                    problems.append(f"lineage {parent}: {nid!r} listed twice")
                listed.append(nid)
            order.append({"label": label, "ids": ids})
        for nid in sorted(set(kids.get(parent, [])) - set(listed)):
            problems.append(f"lineage {parent}: {nid!r} has no group — add it to "
                            "branches.py LINEAGE, or it sorts last with no reason")
        out[parent] = order
    return out


def main():
    branch_ids = {b[0] for b in BRANCHES}
    if len(branch_ids) != len(BRANCHES):
        problems.append("duplicate branch id in branches.py")

    # 1. every branch's own parent exists
    for bid, label, _ in BRANCHES:
        if "." in bid and bid.rsplit(".", 1)[0] not in branch_ids:
            problems.append(f"branch {bid} has no parent branch")

    check_duplicate_keys(HERE / "usrc2020.py", "MAP")

    # 2/3. leaves
    leaves = {}
    for code, path in MAP.items():
        parent = path.rsplit(".", 1)[0] if "." in path else None
        # a mapping may point AT a branch (islam, sikhism) where there is no depth below it
        if path in branch_ids:
            leaves.setdefault(path, []).append(code)
            continue
        if parent not in branch_ids:
            problems.append(f"{code}: parent branch {parent!r} not in branches.py "
                            f"(leaf {path})")
        leaves.setdefault(path, []).append(code)
    for path, codes in leaves.items():
        if len(codes) > 1:
            problems.append(f"leaf {path} claimed by several codes: {codes}")

    # 4/5. against the actual source data
    detail = pd.read_excel(RAW, sheet_name="2020 Group by Nation", dtype={"Group Code": str})
    detail = detail[detail["Group Code"].notna() & (detail["Group Code"] != "Totals")]
    detail["Group Code"] = detail["Group Code"].str.strip().str.zfill(3)
    data_codes = set(detail["Group Code"])

    mapped = set(MAP) | set(UNMAPPED)
    for code in sorted(data_codes - mapped):
        name = detail.loc[detail["Group Code"] == code, "Group Name"].iloc[0]
        problems.append(f"UNMAPPED group code {code} — {name}")
    for code in sorted(mapped - data_codes):
        problems.append(f"mapped code {code} is not in the source data (typo?)")

    # 6. Czechia. cz2021.py maps at BRANCH level and creates no leaves, so the only things
    #    to check are that every target exists and that every category in the data has a
    #    decision — mapped, or explicitly excluded. Skipped when cz.csv has not been built,
    #    so the taxonomy still builds on a fresh checkout.
    check_duplicate_keys(HERE / "cz2021.py", "MAP")
    for cat, target in cz2021.MAP.items():
        if target not in branch_ids:
            problems.append(f"cz2021: {cat!r} -> {target!r}, which is not in branches.py")
    if CZ.exists():
        cz_cats = set(pd.read_csv(CZ, usecols=["source_category"])["source_category"])
        decided = set(cz2021.MAP) | set(cz2021.EXCLUDED)
        for cat in sorted(cz_cats - decided):
            problems.append(f"cz2021: UNMAPPED category {cat!r}")
        for cat in sorted(decided - cz_cats):
            problems.append(f"cz2021: mapped category {cat!r} is not in cz.csv (typo?)")
    else:
        print(f"note: {CZ.name} not built, skipping the Czech category check")

    # 7. Brazil. Same shape as Czechia, except that only the LEAVES of IBGE's own nested
    #    tree may be mapped — an internal node is an aggregate of its children and mapping
    #    it would double count (br2010.py, "THE CUT"). So the check is three-way: every
    #    target exists, every leaf has a decision, and no INTERNAL node is mapped.
    check_duplicate_keys(HERE / "br2010.py", "MAP")
    for cat, target in br2010.MAP.items():
        if target not in branch_ids:
            problems.append(f"br2010: {cat!r} -> {target!r}, which is not in branches.py")
    if BR.exists():
        d = pd.read_csv(BR, usecols=["source_category", "year", "note"],
                        low_memory=False)
        d = d[d["year"] == 2010]
        code = d["note"].str.extract(r"code=(\d+)")[0]
        parent = d["note"].str.extract(r"parent=(\d*)")[0]
        internal = set(parent.dropna()) - {""}
        is_leaf = code.notna() & ~code.isin(internal)
        br_leaves = set(d.loc[is_leaf, "source_category"])
        br_internal = set(d.loc[code.notna() & code.isin(internal), "source_category"])
        decided = set(br2010.MAP) | set(br2010.EXCLUDED)
        for cat in sorted(br_leaves - decided):
            problems.append(f"br2010: UNMAPPED leaf category {cat!r}")
        for cat in sorted(set(br2010.MAP) & br_internal):
            problems.append(f"br2010: {cat!r} is an INTERNAL node of IBGE's tree and is "
                            "mapped — that double counts its children")
        for cat in sorted(decided - br_leaves - br_internal - {"Total"}):
            problems.append(f"br2010: mapped category {cat!r} is not in br.csv (typo?)")
    else:
        print(f"note: {BR.name} not built, skipping the Brazilian category check")

    # 8. Australia, Ireland and Mexico. All three are flat lists over an ALLOCATED file, so
    #    the check is the same shape as Czechia's: every target must exist, and every
    #    category in the file must be mapped or explicitly excluded.
    for name, (mod, path) in ALLOCATED.items():
        check_duplicate_keys(HERE / f"{name}.py", "MAP")
        for cat, target in mod.MAP.items():
            if target not in branch_ids:
                problems.append(
                    f"{name}: {cat!r} -> {target!r}, which is not in branches.py")
        if not path.exists():
            print(f"note: {path.name} not built, skipping the {name} category check")
            continue
        cats = set(pd.read_csv(path, usecols=["source_category"],
                               low_memory=False)["source_category"])
        decided = set(mod.MAP) | set(mod.EXCLUDED)
        for cat in sorted(cats - decided):
            problems.append(f"{name}: UNMAPPED category {cat!r}")
        for cat in sorted(decided - cats):
            problems.append(
                f"{name}: mapped category {cat!r} is not in {path.name} (typo?)")

    # 9. The United Kingdom, which is three classifications across three files: the two
    #    allocated ones plus Scotland, which needs no allocation and so is read straight
    #    from uk.csv. One mapping covers all three (uk2021.py).
    check_duplicate_keys(HERE / "uk2021.py", "MAP")
    for cat, target in uk2021.MAP.items():
        if target not in branch_ids:
            problems.append(f"uk2021: {cat!r} -> {target!r}, which is not in branches.py")
    uk_cats, missing_file = set(), False
    for path, where in [(UK_EW, None), (UK_NI, None), (UK, "uk_sc_census_2022")]:
        if not path.exists():
            missing_file = True
            continue
        if where is None:
            uk_cats |= set(pd.read_csv(path, usecols=["source_category"],
                                       low_memory=False)["source_category"])
        else:
            u = pd.read_csv(path, usecols=["source_category", "source_id", "geo_level"],
                            low_memory=False)
            uk_cats |= set(u.loc[(u["source_id"] == where)
                                 & (u["geo_level"] == "output_area"),
                                 "source_category"])
    if missing_file:
        print("note: a uk allocated file is missing, skipping the UK category check")
    else:
        decided = set(uk2021.MAP) | set(uk2021.EXCLUDED)
        for cat in sorted(uk_cats - decided):
            problems.append(f"uk2021: UNMAPPED category {cat!r}")
        for cat in sorted(decided - uk_cats):
            problems.append(f"uk2021: mapped category {cat!r} is in none of the three UK "
                            "tables (typo?)")

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1

    # ---- emit the tree
    labels = {b[0]: b[1] for b in BRANCHES}
    notes = {b[0]: b[2] for b in BRANCHES}
    names = dict(zip(detail["Group Code"], detail["Group Name"]))

    nodes = []
    for bid, label, note in BRANCHES:
        node = {"id": bid, "label": label, "kind": "branch"}
        if note:
            node["note"] = note
        # A source whose finest granularity IS this branch — ASARB has one "Muslim
        # Estimate" row and no Sunni/Shia below it. Record the mapping on the branch;
        # it is not a leaf and inventing one under it would assert detail nobody has.
        if bid in leaves:
            code = leaves[bid][0]
            node["sources"] = {"usrc2020": code}
            if code in REVIEW:
                node["review"] = REVIEW[code]
        nodes.append(node)
    for path, codes in sorted(leaves.items()):
        if path in labels:
            continue
        code = codes[0]
        nodes.append({"id": path, "label": names[code], "kind": "leaf",
                      "sources": {"usrc2020": code},
                      **({"review": REVIEW[code]} if code in REVIEW else {})})

    lineage = check_lineage(nodes)
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1

    OUT.write_text(
        json.dumps({"nodes": nodes, "lineage": lineage}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # ---- fill the path column
    df = pd.read_csv(CSV, dtype={"Group Code": str})
    df["path"] = df["Group Code"].map(
        lambda c: MAP.get(c, "UNMAPPED" if c in UNMAPPED else ""))
    df.to_csv(CSV, index=False)

    depths = [n["id"].count(".") + 1 for n in nodes]
    print(f"ok — {len(nodes)} nodes: {sum(1 for n in nodes if n['kind']=='branch')} branches, "
          f"{sum(1 for n in nodes if n['kind']=='leaf')} leaves")
    print(f"     depth 1-{max(depths)}, {len(REVIEW)} flagged for review, "
          f"{len(UNMAPPED)} held off the tree")
    print(f"     wrote {OUT.name} and filled `path` in {CSV.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
