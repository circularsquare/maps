"""T-3873: split trainworld/sim/src/track/net.rs (by method group) and wasm/tests.rs (by area), moves only.

    python split_track.py [--ref REF]     (run from anywhere; works on the tree this script is pointed at by --tree)

Reads the UNSPLIT files from git (`git show REF:path`, default REF = master), writes the split files into the
working tree. Re-run after master's net.rs / wasm/tests.rs change: rebase, then run with --ref master (or the
commit your tree is based on). Methods and tests are assigned by name: a group starts at its first-named method
and takes everything up to the next cut point, so a method added upstream lands in the group of the method
before it. EXTRA moves single methods out of their positional group.
"""
import argparse, os, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="master")
ap.add_argument("--tree", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
args = ap.parse_args()
TRACK = os.path.join(args.tree, "trainworld", "sim", "src", "track")
REL = "trainworld/sim/src/track/"


def git_show(path):
    r = subprocess.run(["git", "-C", args.tree, "show", f"{args.ref}:{REL}{path}"], capture_output=True)
    if r.returncode:
        sys.exit(r.stderr.decode())
    return r.stdout.decode("utf-8").replace("\r\n", "\n")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{text.count(chr(10)):6d}  {os.path.relpath(path, args.tree)}")


# ---------------------------------------------------------------- chunking

def chunk_starts(lines, indent, item_re):
    """Start index of each item, taking the doc comments / attributes above it. Returns list of (start, name)."""
    out = []
    ind = " " * indent
    for i, l in enumerate(lines):
        m = item_re.match(l)
        if not m:
            continue
        s = i
        while s > 0 and (lines[s - 1].startswith(ind + "//") or lines[s - 1].startswith(ind + "#[")) and not lines[s - 1].startswith(ind + "  "):
            s -= 1
        out.append((s, m.group("name")))
    return out


# ---------------------------------------------------------------- net.rs

# (file, first method). A method belongs to the group of the latest cut point at or before it.
NET_CUTS = [
    ("service", "frame"),
    ("store", "fit"),
    ("stations", "node_tracks"),
    ("edit", "node_hsr"),
    ("derive", "take_touch"),
    ("validate", "validate"),
    ("costs", "blueprint_cost_items"),
    ("paths", "line_constructed"),
]
# single methods that sit in another group's range but belong to the stations file (the part the rework replaces)
NET_EXTRA = {
    "station_data": "stations",
    "station_upgrades_price": "stations",
    "set_station": "stations",
    "check_station": "stations",
    "platform_reach": "stations",
}
NET_DOC = {
    "service": "Service rows, train slots and the clock-shaped reads (hours, mix, running trains).",
    "store": "Storage: the node, edge and line arrays, ids, bus and ferry reads, edge accessors.",
    "stations": "Stations and one-track queries, with set_station, check_station and platform_reach.",
    "edit": "Setters, the ground (water, terrain, roads) and what a setter reprices.",
    "derive": "Derived data and crossings.",
    "validate": "Validation of edges, nodes, stations and routes, and turns.",
    "costs": "Blueprint and built costs.",
    "paths": "Line paths, stretches, routes and path search.",
}
NET_VIS = re.compile(r"^(    )(const fn |fn )")


def split_net():
    src = git_show("net.rs")
    if "\nmod service;" in src:
        sys.exit("net.rs at that ref is already split")
    lines = src.split("\n")
    head_end = next(i for i, l in enumerate(lines) if l == "impl Network {")
    tail = max(i for i, l in enumerate(lines) if l == "}")  # closing brace of impl Network
    assert lines[tail + 1:] in ([], [""]), "unexpected items after impl Network"
    body = lines[head_end + 1:tail]
    # nothing but methods at 4-space indent inside the impl
    item_re = re.compile(r"^    (?:pub(?:\([a-z]+\))? )?(?:const )?fn (?P<name>\w+)")
    starts = chunk_starts(body, 4, item_re)
    for i, l in enumerate(body):
        if re.match(r"^    [^ /#}]", l) and not item_re.match(l):
            sys.exit(f"unexpected impl item: {l}")
    names = [n for _, n in starts]
    assert len(set(names)) == len(names), "duplicate method names"
    cut_names = [n for _, n in NET_CUTS]
    for n in cut_names + list(NET_EXTRA):
        if n not in names:
            sys.exit(f"method {n} not found")
    assert body[:starts[0][0]] == [l for l in body[:starts[0][0]] if l.strip() == ""], "text before first method"
    group = {}
    cur = None
    for n in names:
        for g, c in NET_CUTS:
            if c == n:
                cur = g
        group[n] = NET_EXTRA.get(n, cur)
    chunks = []
    for k, (s, n) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else len(body)
        chunks.append((n, body[s:e]))
    files = {g: [] for g, _ in NET_CUTS}
    for n, ch in chunks:
        files[group[n]].append(ch)
    for g, parts in files.items():
        out = [f"//! {NET_DOC[g]}", "", "use super::*;", "", "impl Network {"]
        for ch in parts:
            for l in ch:
                m = NET_VIS.match(l)
                out.append(l.replace(m.group(0), "    pub(super) " + m.group(2), 1) if m else l)
        while out[-1] == "":
            out.pop()
        out.append("}")
        out.append("")
        text = "\n".join(out)
        assert text.count("\n") < 1500
        write(os.path.join(TRACK, "net", g + ".rs"), text)
    # net.rs: everything before the impl, then the mod declarations
    top = lines[:head_end]
    while top and top[-1] == "":
        top.pop()
    nl = chr(10)
    allbody = nl.join(nl.join(ch) for _, ch in chunks)
    used = sorted(set(re.findall(r"super::(\w+)::", allbody)))
    bound = set(re.findall(r"^use super::(\w+)::\{self", nl.join(top), re.M))
    need = [m for m in used if m not in bound]
    imp = ["", "// the track-level modules the methods name as `super::x`, now one level down", f"use super::{{{', '.join(need)}}};"] if need else []
    top = top + imp
    decl = [""] + [f"mod {g};" for g, _ in NET_CUTS] + [""]
    write(os.path.join(TRACK, "net.rs"), "\n".join(top + decl))


# ---------------------------------------------------------------- wasm/tests.rs

TEST_CUTS = [
    ("ground_tests", "terrain_block"),
    ("line_tests", "an_express_added_over_constructed_track_runs"),
    ("fleet_tests", "auto_trains_is_the_fewest_that_hold_the_headway"),
    ("blueprint_tests", "blueprint_construct_run"),
    ("drag_tests", "a_click_keeps_the_track_drawn_before"),
    ("route_tests", "degenerate_routes_do_not_panic"),
    ("stroke_tests", "CLICKS"),
    ("platform_tests", "free_at"),
    ("station_tests", "preview_station_agrees_with_add_station"),
    ("quad_tests", "quad_track_and_twin_stations"),
    ("group_tests", "line_groups_join_copy_leave_and_save"),
    ("reach_tests", "platform_room_refusals_name_what_is_reached"),
]


def split_tests():
    src = git_show("wasm/tests.rs")
    if "\nmod ground_tests;" in src:
        sys.exit("tests.rs at that ref is already split")
    lines = src.split("\n")
    item_re = re.compile(r"^(?:pub(?:\([a-z]+\))? )?(?:const |static |fn |struct )(?P<name>\w+)")
    starts = chunk_starts(lines, 0, item_re)
    # column-0 items other than fn/const would be missed: refuse
    for l in lines:
        if re.match(r"^[a-z#]", l) and not item_re.match(l) and not l.startswith(("use ", "#[test]", "#[cfg")):
            sys.exit(f"unexpected top-level line: {l}")
    # #[test] / #[cfg] attrs are taken into the chunk by chunk_starts (they start with "#[")
    pre = lines[:starts[0][0]]
    chunks = []
    for k, (s, n) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        chunks.append([n, lines[s:e], None])
    names = [c[0] for c in chunks]
    assert len(set(names)) == len(names)
    cut = dict((c, a) for a, c in TEST_CUTS)
    for c in cut:
        if c not in names:
            sys.exit(f"{c} not found")
    area, cur = {}, None
    for c in chunks:
        cur = cut.get(c[0], cur)
        c[2] = cur  # None before the first cut: parent
    is_test = lambda c: any(l.startswith("#[test]") for l in c[1])
    # helpers used from another area go to the parent
    changed = True
    while changed:
        changed = False
        for c in chunks:
            if is_test(c) or c[2] is None:
                continue
            pat = re.compile(r"\b" + re.escape(c[0]) + r"\b")
            for o in chunks:
                if o is not c and o[2] != c[2] and pat.search("\n".join(o[1])):
                    c[2] = None
                    changed = True
                    break
    tail_trim = lambda ls: ls
    areas = [a for a, _ in TEST_CUTS]
    for a in areas:
        out = ["use super::*;", ""]
        for c in chunks:
            if c[2] == a:
                out += [l.replace("super::super::", "super::super::super::") for l in c[1]]
        while out[-1] == "":
            out.pop()
        out.append("")
        text = "\n".join(out)
        assert text.count("\n") < 1500, a
        write(os.path.join(TRACK, "wasm", "tests", a + ".rs"), text)
    out = list(pre)
    for c in chunks:
        if c[2] is None:
            out += c[1]
    while out[-1] == "":
        out.pop()
    out += [""] + [f"mod {a};" for a in areas] + [""]
    write(os.path.join(TRACK, "wasm", "tests.rs"), "\n".join(out))
    return {c[0]: c[2] for c in chunks if not is_test(c)}


if __name__ == "__main__":
    split_net()
    h = split_tests()
    print("helpers:", h)
