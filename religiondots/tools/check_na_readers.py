"""Every reader of a normalised file keeps the categories pandas would read as NaN.

    python tools/check_na_readers.py       BAD for an unsafe reader, exit 1 if there is one
    python tools/check_na_readers.py -v    also every safe reader, and readers it cannot place

THE TRAP (playbooks/census_table.md, "pandas deletes a category named `None`"; spec §12
"Parsing the table"). `pd.read_csv` turns the strings in pandas' default NA list (`None`, `NA`,
`null`, `N/A` and more) into NaN. A source whose no-religion answer is printed `None` then
loses that row: `resolve(nan)` finds nothing and `countries.py` drops the row with no error.
The Philippines, Guyana and Zimbabwe (1.26M people) were each found by hand, and each fix was
`keep_default_na=False, na_values=[""]` on one reader. Nothing checked the other readers.

WHAT IT DOES
  1. Scans data/normalized/*.csv with the csv module, which keeps every cell a string, for a
     text cell equal to one of pandas' own default NA strings. Those files are at risk.
  2. Reads each at-risk column again with pandas' defaults and counts the NaNs, so the result
     is about the pandas installed here and not a copied list.
  3. Parses every .py in the tree, finds each `read_csv` call and works out which files it
     reads: a literal, `os.path.join`, a `/` path, a name assigned in the same function or at
     module level, a loop over literals, and an f-string filled from the literal arguments its
     function is called with in the same file. What cannot be worked out becomes `*`.
  4. A reader of an at-risk file is safe with `keep_default_na=False` or `na_filter=False`, or
     with a literal `usecols` that leaves out every at-risk column. Otherwise it is BAD, unless
     registered below.

The registers are re-verified on every run rather than trusted:
  NOT_READ  at-risk rows under a source_id nothing reads. Asserted: every at-risk cell in the
            file is under a listed source_id, the source_id is written in no .py but its
            writer, and no *_allocated.csv carries it.
  WIDE      a reader that takes its file from the command line. Asserted: what it produced
            on disk carries no source_id that has at-risk rows.
  KNOWN     a reader known to lose rows. Printed as WARN with the people lost on every run,
            until someone allowed to edit the reader fixes it. Its file list is pinned, so a
            new file reaching the same reader is BAD.

It reads every normalised file once, so it takes a minute or two on the full tree.
"""

import argparse
import ast
import csv
import fnmatch
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NORM = os.path.join(ROOT, "data", "normalized")
SELF = "tools/check_na_readers.py"      # its own default read in confirm() is the measurement

TEXT_COLUMNS = ("source_category", "geo_id", "geo_name", "geo_level", "source_id")
PRUNE = {"data", ".git", "__pycache__", "node_modules"}
OTHER_DIRS = ("/geo", "/raw", "/processed", "hierarchy")
WILD = "*"
MAX_CANDIDATES = 256

# file -> {source_id: (the .py that writes it, why nothing reads it)}
NOT_READ = {
    "uk.csv": {
        "uk_ni_census_2021_brought_up_in": (
            "sources/uk.py",
            "NISRA MS-B23 and MS-B24, the religion a person was brought up in, where `None` "
            "is an answer. Kept in uk.csv as a witness and drawn nowhere: every uk.csv reader "
            "selects another source_id (Scotland in countries.py::_uk_counts and "
            "taxonomy/build_tree.py, the three drawn censuses in coverage.py::regions, "
            "England and Wales in uk_split.py, England and Wales and Northern Ireland in "
            "taxonomy/hierarchy/make_uk.py), and allocate.py is run on uk_ew_census_2021 "
            "and uk_ni_census_2021."),
    },
}

# (reader file, function) -> why its file comes from outside the code
WIDE = {
    ("allocate.py", "main"):
        "reads data/normalized/<--source>.csv with pandas' defaults. Safe while nothing it has "
        "allocated carries a source_id with at-risk rows, which is asserted; allocating a "
        "source whose categories include `None` would lose them. Not this lint's to edit.",
}

# (reader file, function) -> (a line its safety rests on, why). A reader that reads an at-risk
# file for its units or its `Total` rows, where a lost `None` row changes nothing. Asserted:
# the pinned line is still in the function, so an edit that changes what it reads fails here.
ROWS_NOT_USED = {
    ("sources/bz_geo.py", "main"): (
        '.drop_duplicates("geo_id")[["geo_id", "geo_name"]]',
        "the district ids and names only; every district has other rows"),
    ("sources/bz_grid.py", "main"): (
        '(cen["source_category"] == "Total")',
        "the district `Total` rows only, for a printed Kontur ratio"),
    ("sources/ge_geo.py", "build_units"): (
        '.drop_duplicates("geo_id").copy()',
        "the region ids and names only; every region has other rows"),
    ("sources/ge_geo.py", "build_grid"): (
        '(df["source_category"] == "Total")',
        "the region `Total` rows only, for the Kontur check"),
    ("sources/ug_grid.py", "main"): (
        'norm[norm["source_category"] == "Total"]',
        "the district `Total` rows only, for a printed Kontur ratio"),
}

# (reader file, function) -> {"files": pinned at-risk files it reads, "why": ...}
KNOWN = {
    # countries.py::_micro_counts was the first entry (Bermuda, Niue and Tuvalu lost their
    # `None` row); fixed 2026-09-14, the day this lint found it.
}


# ---------------------------------------------------------------- 1-2. the files at risk

def na_tokens():
    from pandas._libs.parsers import STR_NA_VALUES
    return set(STR_NA_VALUES) - {""}


def scan(tokens):
    """({file: info} for files holding an NA token in a text column, {file: source_ids})."""
    csv.field_size_limit(2 ** 31 - 1)
    at_risk, sids_of = {}, {}
    for fn in sorted(os.listdir(NORM)):
        if not fn.endswith(".csv"):
            continue
        cells, empties, sids, people, seen = {}, {}, {}, {}, set()
        with open(os.path.join(NORM, fn), encoding="utf-8", newline="") as fh:
            rd = csv.reader(fh)
            hdr = next(rd, None)
            if not hdr:
                continue
            cols = [(j, c) for j, c in enumerate(hdr) if c in TEXT_COLUMNS]
            j_sid = hdr.index("source_id") if "source_id" in hdr else None
            j_lvl = hdr.index("geo_level") if "geo_level" in hdr else None
            j_cnt = hdr.index("count") if "count" in hdr else None
            for row in rd:
                sid = row[j_sid] if j_sid is not None and j_sid < len(row) else ""
                seen.add(sid)
                for j, c in cols:
                    v = row[j] if j < len(row) else ""
                    if v == "":
                        empties[c] = empties.get(c, 0) + 1
                    elif v in tokens:
                        cells[(c, v)] = cells.get((c, v), 0) + 1
                        sids[sid] = sids.get(sid, 0) + 1
                        if c == "source_category" and j_cnt is not None:
                            try:
                                n = float(row[j_cnt])
                            except (ValueError, IndexError):
                                n = 0.0
                            lvl = row[j_lvl] if j_lvl is not None else ""
                            people[lvl] = people.get(lvl, 0.0) + n
        sids_of[fn] = seen
        if cells:
            at_risk[fn] = {"cells": cells, "empties": empties, "sids": sids, "people": people}
    return at_risk, sids_of


def confirm(at_risk):
    """Add `lost`: cells pandas' default read turns into NaN beyond the empty ones."""
    import pandas as pd

    for fn, info in at_risk.items():
        cols = sorted({c for c, _ in info["cells"]})
        df = pd.read_csv(os.path.join(NORM, fn), usecols=cols, dtype=str, low_memory=False)
        info["lost"] = {c: int(df[c].isna().sum()) - info["empties"].get(c, 0) for c in cols}
        info["cols"] = {c for c, n in info["lost"].items() if n > 0}


def people_line(info):
    return ", ".join(f"{n:,.0f} at {lvl or '?'}" for lvl, n in sorted(info["people"].items()))


# ---------------------------------------------------------------- 3. the readers

class Module:
    def __init__(self, path):
        self.rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8") as fh:
            self.text = fh.read()
        self.tree = ast.parse(self.text)
        self.parent = {}
        self.calls = {}
        for node in ast.walk(self.tree):
            for ch in ast.iter_child_nodes(node):
                self.parent[ch] = node
            if isinstance(node, ast.Call):
                name = _call_name(node)
                if name:
                    self.calls.setdefault(name, []).append(node)

    def around(self, node):
        """Enclosing functions, innermost first."""
        out = []
        while node in self.parent:
            node = self.parent[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                out.append(node)
        return out


def _call_name(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None


def _own_nodes(scope):
    """Every node of `scope` that is not inside a nested function or class."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))


def _combine(parts):
    out = [""]
    for p in parts:
        out = list(dict.fromkeys(a + b for a in out for b in p))[:MAX_CANDIDATES]
    return out


def strings(mod, expr, around, depth=0):
    """Candidate path strings for `expr`, with `*` for whatever cannot be known."""
    if depth > 8:
        return [WILD]
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value.replace("\\", "/")]
    if isinstance(expr, ast.JoinedStr):
        parts = []
        for v in expr.values:
            if isinstance(v, ast.Constant):
                parts.append([str(v.value)])
            elif isinstance(v, ast.FormattedValue):
                parts.append(strings(mod, v.value, around, depth + 1))
            else:
                parts.append([WILD])
        return _combine(parts)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Div, ast.Add)):
        sep = ["/"] if isinstance(expr.op, ast.Div) else [""]
        return _combine([strings(mod, expr.left, around, depth + 1), sep,
                         strings(mod, expr.right, around, depth + 1)])
    if isinstance(expr, ast.Call):
        name = _call_name(expr)
        if name == "join" and expr.args:
            parts = []
            for i, a in enumerate(expr.args):
                if i:
                    parts.append(["/"])
                parts.append(strings(mod, a, around, depth + 1))
            return _combine(parts)
        if name in ("Path", "str", "fspath", "abspath", "normpath") and expr.args:
            return strings(mod, expr.args[0], around, depth + 1)
        return [WILD]
    if isinstance(expr, ast.Name):
        return _name_values(mod, expr.id, around, depth + 1)
    return [WILD]


def _name_values(mod, name, around, depth):
    for fn in around:
        vals = []
        if not isinstance(fn, ast.Lambda):
            for n in _own_nodes(fn):
                if isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == name for t in n.targets):
                    vals += strings(mod, n.value, mod.around(n), depth + 1)
                elif isinstance(n, ast.For) and isinstance(n.target, ast.Name) \
                        and n.target.id == name:
                    it = n.iter
                    if isinstance(it, (ast.Tuple, ast.List)) and all(
                            isinstance(e, ast.Constant) for e in it.elts):
                        vals += [str(e.value) for e in it.elts]
                    else:
                        vals.append(WILD)
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        if not vals and name in params:
            vals = _call_site_values(mod, fn, name)
        if vals:
            return list(dict.fromkeys(vals))
    vals = []
    for n in mod.tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            vals += strings(mod, n.value, [], depth + 1)
    return list(dict.fromkeys(vals)) or [WILD]


def _call_site_values(mod, fn, name):
    """The literal values `fn` is called with for parameter `name`, anywhere in the module."""
    if isinstance(fn, ast.Lambda):
        return [WILD]
    params = [a.arg for a in fn.args.args]
    i = params.index(name) if name in params else None
    vals = []
    for call in mod.calls.get(fn.name, []):
        arg = next((k.value for k in call.keywords if k.arg == name), None)
        if arg is None and i is not None and i < len(call.args) \
                and not any(isinstance(a, ast.Starred) for a in call.args[:i + 1]):
            arg = call.args[i]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            vals.append(arg.value)
        else:
            vals.append(WILD)
    return vals or [WILD]


def readers(mod):
    """(line, function, candidate paths, call) for every read_csv in the module."""
    for node in ast.walk(mod.tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "read_csv"):
            continue
        target = node.args[0] if node.args else next(
            (k.value for k in node.keywords if k.arg == "filepath_or_buffer"), None)
        if target is None:
            continue
        around = mod.around(node)
        func = next((f.name for f in around if not isinstance(f, ast.Lambda)), "<module>")
        yield node.lineno, func, strings(mod, target, around), node


def targets(cands, files):
    """(at-risk files the candidates can name, whether any candidate was placed at all)."""
    hit, placed = set(), False
    for c in cands:
        base = c.rsplit("/", 1)[-1]
        d = c[:len(c) - len(base)]
        if "normalized" in d:
            placed = True
            hit |= {f for f in files if fnmatch.fnmatchcase(f, base)}
        elif WILD not in base and base.endswith(".csv"):
            placed = True
            if not any(o in d for o in OTHER_DIRS):
                hit |= {f for f in files if f == base}
    return hit, placed


def safe(call, cols):
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    for key in ("keep_default_na", "na_filter"):
        v = kw.get(key)
        if isinstance(v, ast.Constant) and v.value is False:
            return f"{key}=False"
    uc = kw.get("usecols")
    if isinstance(uc, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) for e in uc.elts):
        used = {e.value for e in uc.elts}
        if not used & cols:
            return f"usecols leaves out {', '.join(sorted(cols))}"
    return None


def python_files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in PRUNE]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(base, f)


# ---------------------------------------------------------------- 4. the verdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true", help="also list safe and unplaced readers")
    args = ap.parse_args()

    tokens = na_tokens()
    at_risk, sids_of = scan(tokens)
    confirm(at_risk)
    bad = 0

    def say(kind, msg):
        nonlocal bad
        bad += kind == "BAD"
        print(f"  {kind:<4} {msg}")

    print(f"normalised files holding one of pandas' {len(tokens)} default NA strings:")
    for fn, info in at_risk.items():
        cells = ", ".join(f"{c} {v!r} x{n:,}" for (c, v), n in sorted(info["cells"].items()))
        lost = sum(info["lost"].values())
        print(f"    {fn:<24} {cells}; a default pandas read loses {lost:,}"
              + (f"; people: {people_line(info)}" if info["people"] else ""))
    at_risk = {fn: info for fn, info in at_risk.items() if info["cols"]}

    modules = []
    for path in python_files():
        try:
            modules.append(Module(path))
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  --   {os.path.relpath(path, ROOT)} not parsed: {e}")

    print("\nregisters:")
    live = dict(at_risk)
    for fn, sids in NOT_READ.items():
        info = at_risk.get(fn)
        if info is None:
            say("BAD", f"NOT_READ {fn}: holds no at-risk rows any more; remove the entry")
            continue
        stray = {s: n for s, n in info["sids"].items() if s not in sids}
        ok = not stray
        for sid, (writer, _why) in sids.items():
            users = [m.rel for m in modules if sid in m.text and m.rel not in (writer, SELF)]
            allocated = [f for f, ss in sids_of.items() if f.endswith("_allocated.csv") and sid in ss]
            ok &= not users and not allocated
            if users or allocated:
                say("BAD", f"NOT_READ {fn} {sid}: named in {users or 'no other .py'}, carried by "
                           f"{allocated or 'no allocated file'}; it is read after all")
        if stray:
            say("BAD", f"NOT_READ {fn}: at-risk rows under unregistered source_ids {stray}")
        if ok:
            del live[fn]
            say("OK", f"NOT_READ {fn}: all {sum(info['sids'].values()):,} at-risk cells are "
                      f"under {', '.join(sids)}, which only {', '.join(w for w, _ in sids.values())} "
                      "names and no allocated file carries")

    risky_sids = {s for info in live.values() for s in info["sids"]}
    for (rel, func), why in WIDE.items():
        produced = {f: ss & risky_sids for f, ss in sids_of.items()
                    if f.endswith("_allocated.csv") and ss & risky_sids}
        say("BAD" if produced else "OK",
            f"WIDE {rel}::{func}: " + (f"produced {produced}" if produced else
                                       "nothing it produced carries an at-risk source_id"))

    print("\nreaders of at-risk files:")
    seen_known = set()
    unplaced = []
    for mod in modules:
        if mod.rel == SELF:
            continue
        for line, func, cands, call in readers(mod):
            hits, placed = targets(cands, live)
            if not placed:
                unplaced.append(f"{mod.rel}:{line} {func}")
            if not hits:
                continue
            where = f"{mod.rel}:{line} {func}"
            cols = set().union(*(live[f]["cols"] for f in hits))
            why_safe = safe(call, cols)
            if why_safe:
                if args.v:
                    say("OK", f"{where}: {', '.join(sorted(hits))} ({why_safe})")
                continue
            if (mod.rel, func) in WIDE:
                if args.v:
                    say("OK", f"{where}: wide reader, registered and verified above")
                continue
            if (mod.rel, func) in ROWS_NOT_USED:
                pin, why = ROWS_NOT_USED[(mod.rel, func)]
                body = next((ast.get_source_segment(mod.text, n) or "" for n in ast.walk(mod.tree)
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and n.name == func), "")
                if pin in body:
                    if args.v:
                        say("OK", f"{where}: {', '.join(sorted(hits))}, rows not used ({why})")
                else:
                    say("BAD", f"{where}: registered in ROWS_NOT_USED but no longer contains "
                               f"{pin!r}; re-read what it takes from {', '.join(sorted(hits))}")
                continue
            known = KNOWN.get((mod.rel, func))
            if known:
                seen_known.add((mod.rel, func))
                for f in sorted(hits & known["files"]):
                    say("WARN", f"{where}: {f} loses {sum(live[f]['lost'].values()):,} row(s), "
                                f"{people_line(live[f])} people. {known['why']}")
                hits = hits - known["files"]
                if not hits:
                    continue
            lost = "; ".join(f"{f} {people_line(live[f]) or 'no counts'}" for f in sorted(hits))
            say("BAD", f"{where}: reads {', '.join(sorted(hits))} without keep_default_na=False "
                       f"({lost}). Add `keep_default_na=False, na_values=[\"\"]`")
    for key in set(KNOWN) - seen_known:
        say("BAD", f"KNOWN {key[0]}::{key[1]} no longer reads an at-risk file; remove the entry")

    if args.v and unplaced:
        print(f"\n  --   {len(unplaced)} read_csv call(s) whose file could not be placed: "
              + "; ".join(unplaced))
    print(f"\n{'OK' if not bad else 'FAILED'}: {len(at_risk)} at-risk file(s), "
          f"{bad} unsafe reader(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
