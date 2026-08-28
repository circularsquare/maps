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

from branches import BRANCHES                      # noqa: E402
from usrc2020 import MAP, UNMAPPED, REVIEW         # noqa: E402

RAW = HERE.parent / "data" / "raw" / "2020_USRC_Group_Detail.xlsx"
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

    OUT.write_text(json.dumps({"nodes": nodes}, indent=2, ensure_ascii=False),
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
