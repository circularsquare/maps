"""The changelog for the next published update (T-3424).

One JSON file per release under trainworld/changelog/. The unpublished one is next.json; publish renames it to
<version>.json. An item: id (c001...), text (what the agent wrote), edited (Anita's wording, or null), hidden,
task, by, at, rev (counts changes, so a page saving over a newer change gets refused), area (its icon, T-3466: one of
AREAS), fix (a bug fix: lighter text, after its area's other items), parent (the id of the group it sits under) and
group (true for a group item, like "track bug fixes", that others sit under) and highlight (true on a top-level item or
group that the game lists first, T-3591: for an update with more than about 20 shown items, 1-8 of the biggest
changes for a player). The page shows edited if set, else
text. The order of items in the file is her order (the board's drag; new items go first); publish keeps the area
grouping the game shows and uses her order within each area. A hidden item stays in the file and out of the game; a hidden group hides its items too.

    python trainworld/tools/changelog.py add "fixed trains drawn through each other on single track" --area track --fix --task T-3379 --by bugs1
    python trainworld/tools/changelog.py group "track bug fixes" --area track --fix
    python trainworld/tools/changelog.py set c002 [--area track] [--fix | --no-fix] [--under c055 | --under none]
    python trainworld/tools/changelog.py reword c002 "<new line>"      (sets edited, as the board's edit does)
    python trainworld/tools/changelog.py hide c002 [--show]
    python trainworld/tools/changelog.py highlight c002 [--off]     (a top-level item or a group)
    python trainworld/tools/changelog.py move c002 [--before c005] [--under c010|none]   (no --before: last; --under
                                                          puts it in a group, none takes it out; default: as it is)
    python trainworld/tools/changelog.py list
    python trainworld/tools/changelog.py line "update v0.3.6: <the start menu's line, 80 characters at most>"
    python trainworld/tools/changelog.py publish --version v0.3.6 [--at 2026-11-17T15:00:00Z] [--dry-run]

Player text rules: her "User-facing text" section (plain words, lowercase here, no dashes between phrases). Never
write a placeholder with braces; a straight apostrophe becomes the curly one at publish (the strings are ICU messages).
Stdlib only; serve.py imports this module for the board's saves.
"""
import argparse
import datetime
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # trainworld/
DIR = os.path.join(ROOT, "changelog")
NEXT = os.path.join(DIR, "next.json")
UPDATES_TS = os.path.join(ROOT, "app", "src", "ui", "updates.ts")
MESSAGES = os.path.join(ROOT, "app", "src", "i18n", "messages", "en.json")
CONTEXT = os.path.join(ROOT, "app", "src", "i18n", "context.json")
HIGHLIGHT_MAX = 8
LINE_MAX = 80  # checks/i18n.mjs UPDATE_LINE_MAX
CURLY = chr(0x2019)
# app/src/ui/updates.ts AREAS, in the order the game sorts by. "other" fits none of Anita's areas (a plain dot); "bug" is
# a fix that fits no area (an area's own fixes take its icon)
AREAS = ("track", "demand", "scheduling", "fares", "schematic", "community", "performance", "other", "bug")


class Locked:
    """A lock file shared by the CLI and serve.py, so an agent's add and her save never overwrite each other."""

    def __init__(self, path):
        self.lock = path + ".lock"

    def __enter__(self):
        os.makedirs(os.path.dirname(self.lock), exist_ok=True)
        end = time.time() + 10
        while True:
            try:
                self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.lock) > 30:   # left by a crash
                        os.remove(self.lock)
                        continue
                except OSError:
                    pass
                if time.time() > end:
                    raise RuntimeError("changelog is locked")
                time.sleep(0.05)

    def __exit__(self, *a):
        os.close(self.fd)
        try:
            os.remove(self.lock)
        except OSError:
            pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read(path=NEXT):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"release": "next", "line": "", "items": []}


def write_file(data, path=NEXT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def state(path=NEXT):
    """What the page reads: the file plus its version (mtime in microseconds, 0 with no file)."""
    d = read(path)
    try:
        d["version"] = os.stat(path).st_mtime_ns // 1000
    except OSError:
        d["version"] = 0
    return d


def check_area(area):
    if area not in AREAS:
        raise ValueError("area is one of: " + ", ".join(AREAS))


def check_parent(d, parent, item_id=None):
    if parent is None:
        return
    g = next((i for i in d["items"] if i["id"] == parent), None)
    if g is None or not g.get("group") or parent == item_id:
        raise ValueError("%s is not a group" % parent)


def add_item(text, task=None, by="agent", path=NEXT, area=None, fix=False, parent=None, group=False):
    text = " ".join(text.split())
    if not text:
        raise ValueError("empty text")
    check_area(area)
    with Locked(path):
        d = read(path)
        check_parent(d, parent)
        n = max([int(i["id"][1:]) for i in d["items"]] + [0]) + 1
        item = {"id": "c%03d" % n, "text": text, "edited": None, "hidden": False,
                "task": task, "by": by, "at": now(), "rev": 1, "area": area, "fix": bool(fix)}
        if parent:
            item["parent"] = parent
        if group:
            item["group"] = True
        d["items"].insert(0, item)       # her order is the file's: a new item goes first, where she sees it
        write_file(d, path)
    return item


KEEP = object()      # move_item's `under` when the parent is not part of the move


def move_item(item_id, before, path=NEXT, under=KEEP):
    """Her drag on the board: put item_id just before the item `before` (an id), or last when before is None, under the
    group `under` (an id; None for the top level; KEEP leaves the parent as it is). `before` must have the item's new
    parent. A group cannot go under a group. An item that changes parent loses its highlight, and is hidden when
    the group is. Returns the status: 200, 404, 400."""
    with Locked(path):
        d = read(path)
        items = d["items"]
        item = next((i for i in items if i["id"] == item_id), None)
        if item is None:
            return 404
        parent = item.get("parent") if under is KEEP else (under or None)
        changed = parent != item.get("parent")
        if parent:
            g = next((i for i in items if i["id"] == parent), None)
            if g is None:
                return 404
            if not g.get("group") or parent == item_id or item.get("group"):
                return 400
        if before is not None:
            tgt = next((i for i in items if i["id"] == before), None)
            if tgt is None:
                return 404
            if (tgt.get("parent") or None) != parent or tgt is item:
                return 400
        if changed:
            if parent:
                item["parent"] = parent
                item.pop("highlight", None)
                if g.get("hidden"):
                    item["hidden"] = True
            else:
                item.pop("parent", None)
            item["rev"] += 1
            item["changed"] = now()
        items.remove(item)
        items.insert(len(items) if before is None else next(n for n, i in enumerate(items) if i["id"] == before), item)
        write_file(d, path)
    return 200


def delete_group(item_id, rev, path=NEXT):
    """Her board's delete, for a group item with nothing under it (agents' items stay as the record). (status, item)."""
    with Locked(path):
        d = read(path)
        item = next((i for i in d["items"] if i["id"] == item_id), None)
        if item is None:
            return 404, None
        if item["rev"] != rev:
            return 409, item
        if not item.get("group") or any(i.get("parent") == item_id for i in d["items"]):
            return 400, item
        d["items"].remove(item)
        write_file(d, path)
    return 200, item


def update_item(item_id, fields, rev, path=NEXT):
    """Her save: edited (string, or null to go back to the agent's text), hidden, highlight, area, fix, parent (a group's id
    or null). Returns (status, item): 409 with the current item when rev is not the item's; ValueError on a bad
    area or parent."""
    with Locked(path):
        d = read(path)
        item = next((i for i in d["items"] if i["id"] == item_id), None)
        if item is None:
            return 404, None
        if item["rev"] != rev:
            return 409, item
        if "edited" in fields:
            e = fields["edited"]
            if e is not None:
                e = " ".join(str(e).split()) or None
            item["edited"] = e if e != item["text"] else None
        if "hidden" in fields:
            item["hidden"] = bool(fields["hidden"])
        if "area" in fields:
            check_area(fields["area"])
            item["area"] = fields["area"]
        if "fix" in fields:
            item["fix"] = bool(fields["fix"])
        if "highlight" in fields:
            if fields["highlight"] and (item.get("parent") or fields.get("parent")):
                raise ValueError("only a top-level item or a group can be a highlight")
            if fields["highlight"]:
                item["highlight"] = True
            else:
                item.pop("highlight", None)
        if "parent" in fields:
            p = fields["parent"] or None
            if item.get("group") and p:
                raise ValueError("a group cannot sit under another")
            check_parent(d, p, item_id)
            if p:
                item["parent"] = p
                item.pop("highlight", None)      # an item under a group is no highlight of its own
            else:
                item.pop("parent", None)
        item["rev"] += 1
        item["changed"] = now()
        write_file(d, path)
    return 200, item


def shown(item):
    return item["edited"] or item["text"]


def rank(i):
    a = i.get("area")
    return (AREAS.index(a) if a in AREAS else len(AREAS)) * 2 + (1 if i.get("fix") else 0)


def visible(d):
    """What reaches the game, in the game's order: [(item, [its group's items])]. Sorted by area, an area's fixes
    after its other items, else her (file) order; a group with nothing visible under it is left out."""
    items = d["items"]
    groups = {i["id"]: i for i in items if i.get("group")}
    out = []
    for i in items:
        if i["hidden"] or i.get("parent") in groups:
            continue
        kids = [c for c in items if c.get("parent") == i["id"] and not c["hidden"]] if i.get("group") else []
        if i.get("group") and not kids:
            continue
        out.append((i, kids))
    return [x for _, x in sorted(enumerate(out), key=lambda p: (rank(p[1][0]), p[0]))]


KEEP_STARS = " Words between asterisks are bold: keep the asterisks around the translated words, as in the source."


def esc(s):
    return json.dumps(s, ensure_ascii=False)


def publish(version, at=None, dry=False, path=NEXT):
    m = re.fullmatch(r"v(\d+(?:\.\d+)+)", version)
    if not m:
        raise SystemExit("version looks like v0.3.6")
    prefix = "updates.v" + m.group(1).replace(".", "")
    d = read(path)
    vis = visible(d)
    line = (d.get("line") or "").strip()
    if not vis:
        raise SystemExit("no visible items: nothing to publish")
    bare = [i["id"] for i, _ in vis if i.get("area") not in AREAS]
    if bare:
        raise SystemExit("no area: %s (python trainworld/tools/changelog.py set <id> --area ...)" % ", ".join(bare))
    if not line:
        raise SystemExit("no start-menu line: python trainworld/tools/changelog.py line \"update %s: ...\"" % version)
    if len(line) > LINE_MAX:
        raise SystemExit("the line is %d characters, the limit is %d" % (len(line), LINE_MAX))
    hl_ids = {i["id"] for i, _ in vis if i.get("highlight")}
    if len(hl_ids) > HIGHLIGHT_MAX:
        print("warning: %d highlights, the notes are meant to have 1-%d" % (len(hl_ids), HIGHLIGHT_MAX))
    strings = {prefix + "Line": line}
    keys = []      # (key, item) of the list's items
    hl_keys = {}   # a highlight's id: its key
    subs = {}      # a group's key: its items' keys
    n = 0
    for i, kids in vis:
        n += 1
        k = "%sChange%d" % (prefix, n)
        strings[k] = shown(i)
        keys.append((k, i))
        if i["id"] in hl_ids:
            hl_keys[i["id"]] = k
        for c in kids:
            n += 1
            ck = "%sChange%d" % (prefix, n)
            strings[ck] = shown(c)
            subs.setdefault(k, []).append(ck)
    for k, v in list(strings.items()):
        v2 = v.replace("'", CURLY)
        if re.search(r"[{}]", v2):
            raise SystemExit("%s: braces would be read as a placeholder: %s" % (k, v))
        strings[k] = v2
    ts = open(UPDATES_TS, encoding="utf-8", newline="").read()
    eol = "\r\n" if "\r\n" in ts else "\n"
    if 'version: "%s"' % version in ts:
        raise SystemExit("%s is already in updates.ts" % version)
    at = at or now()
    changes = "".join('      ["%s", "%s"%s],\n' % (k, i["area"], ', "fix"' if i.get("fix") else "") for k, i in keys)
    sub = "".join('      "%s": [%s],\n' % (k, ", ".join('"%s"' % c for c in cs)) for k, cs in subs.items())
    hl = [hl_keys[i["id"]] for i in d["items"] if i["id"] in hl_keys]     # her (file) order
    hl_ts = '    highlights: [%s],\n' % ", ".join('"%s"' % k for k in hl) if hl else ""
    entry = ('  {\n    version: "%s",\n    at: "%s",\n    line: "%sLine",\n    changes: [\n%s    ],\n%s%s  },\n'
             % (version, at, prefix, changes, hl_ts, "    sub: {\n%s    },\n" % sub if sub else ""))
    entry = entry.replace(chr(10), eol)
    anchor = "export const UPDATES: Update[] = [" + eol
    if ts.count(anchor) != 1:
        raise SystemExit("cannot find the UPDATES list in updates.ts")
    if dry:
        print(entry)
        for k, v in strings.items():
            print("  %s: %s" % (k, v))
        print("dry run: nothing written (%d hidden items stay out)" % (len(d["items"]) - sum(1 + len(k) for _, k in vis)))
        return
    new_ts = ts.replace(anchor, anchor + entry, 1)
    en = open(MESSAGES, encoding="utf-8", newline="").read()
    ctx = open(CONTEXT, encoding="utf-8", newline="").read()
    # newest first: before the first existing updates.v...Line key
    pat = re.compile(r'^  "updates\.v(\d+)Line":', re.M)
    for name, text in (("en.json", en), ("context.json", ctx)):
        if not pat.search(text):
            raise SystemExit("no updates.v...Line key in " + name)

    def put(text, lines):
        mm = max(pat.finditer(text), key=lambda x: int(x.group(1)))   # before the newest update's keys
        e = "\r\n" if "\r\n" in text else "\n"
        return text[:mm.start()] + "".join("  %s: %s,%s" % (esc(k), esc(v), e) for k, v in lines) + text[mm.start():]

    new_en = put(en, list(strings.items()))
    ctxl = [(prefix + "Line", "Start-menu announcement for %s. Casual, lowercase, at most %d characters." % (version, LINE_MAX))]
    ctxl += [(k, "Changelog line for %s, written by an agent, wording checked before release.%s" % (version, KEEP_STARS if "*" in v else ""))
             for k, v in strings.items() if k != prefix + "Line"]
    new_ctx = put(ctx, ctxl)
    json.loads(new_en), json.loads(new_ctx)
    for p, s in ((UPDATES_TS, new_ts), (MESSAGES, new_en), (CONTEXT, new_ctx)):
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(s)
    d["release"] = version
    d["published"] = at
    write_file(d, os.path.join(DIR, version + ".json"))
    if os.path.exists(path):
        os.remove(path)
    print("wrote updates.ts, en.json and context.json; the other languages need translating (the usual i18n step)")
    print("moved the items to changelog/%s.json" % version)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("text")
    a.add_argument("--area", required=True, choices=AREAS)
    a.add_argument("--fix", action="store_true", help="a bug fix (lighter text); give it its area, bug only when none fits")
    a.add_argument("--under", help="the id of a group to sit under")
    a.add_argument("--task")
    a.add_argument("--by", default="agent")
    g = sub.add_parser("group")
    g.add_argument("text")
    g.add_argument("--area", required=True, choices=AREAS)
    g.add_argument("--fix", action="store_true")
    g.add_argument("--by", default="agent")
    s = sub.add_parser("set")
    s.add_argument("id")
    s.add_argument("--area", choices=AREAS)
    s.add_argument("--fix", action="store_true", default=None)
    s.add_argument("--no-fix", dest="fix", action="store_false")
    s.add_argument("--under", help="a group's id, or none")
    w = sub.add_parser("reword")
    w.add_argument("id")
    w.add_argument("text")
    h = sub.add_parser("hide")
    h.add_argument("id")
    h.add_argument("--show", action="store_true", help="unhide")
    hl = sub.add_parser("highlight")
    hl.add_argument("id")
    hl.add_argument("--off", action="store_true", help="remove the highlight")
    mv = sub.add_parser("move")
    mv.add_argument("id")
    mv.add_argument("--before")
    mv.add_argument("--under", help="a group's id, or none for the top level")
    sub.add_parser("list")
    l = sub.add_parser("line")
    l.add_argument("text")
    p = sub.add_parser("publish")
    p.add_argument("--version", required=True)
    p.add_argument("--at")
    p.add_argument("--dry-run", action="store_true")
    r = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # item text can be CJK; the Windows console's code page is not
    if r.cmd == "add":
        it = add_item(r.text, r.task, r.by, area=r.area, fix=r.fix, parent=r.under)
        print("added %s" % it["id"])
    elif r.cmd == "group":
        it = add_item(r.text, None, r.by, area=r.area, fix=r.fix, group=True)
        print("added group %s" % it["id"])
    elif r.cmd == "set":
        fields = {}
        if r.area:
            fields["area"] = r.area
        if r.fix is not None:
            fields["fix"] = r.fix
        if r.under:
            fields["parent"] = None if r.under == "none" else r.under
        it = next((i for i in read()["items"] if i["id"] == r.id), None)
        if it is None:
            raise SystemExit("no item " + r.id)
        code, _ = update_item(r.id, fields, it["rev"])
        if code != 200:
            raise SystemExit("not saved (%d): try again" % code)
    elif r.cmd in ("reword", "hide"):
        # same save as the board's edit: sets edited (the shown wording) or hidden, bumps rev, stamps changed
        it = next((i for i in read()["items"] if i["id"] == r.id), None)
        if it is None:
            raise SystemExit("no item " + r.id)
        fields = {"edited": r.text} if r.cmd == "reword" else {"hidden": not r.show}
        code, _ = update_item(r.id, fields, it["rev"])
        if code != 200:
            raise SystemExit("not saved (%d): try again" % code)
        print("%s %s" % (r.cmd, r.id))
    elif r.cmd == "highlight":
        it = next((i for i in read()["items"] if i["id"] == r.id), None)
        if it is None:
            raise SystemExit("no item " + r.id)
        try:
            code, _ = update_item(r.id, {"highlight": not r.off}, it["rev"])
        except ValueError as e:
            raise SystemExit(str(e))
        if code != 200:
            raise SystemExit("not saved (%d): try again" % code)
        print("%s %s" % ("unhighlighted" if r.off else "highlighted", r.id))
    elif r.cmd == "move":
        under = KEEP if r.under is None else (None if r.under == "none" else r.under)
        code = move_item(r.id, r.before, under=under)
        if code != 200:
            raise SystemExit("not moved (%d): the ids need to exist, --under must be a group (an item, not a group, "
                             "goes in one), and --before must sit in the item's new place" % code)
    elif r.cmd == "list":
        d = read()
        print("line: %s" % (d.get("line") or "(none)"))
        for i in d["items"]:
            print("%s%s%s %s %s%s %s%s%s" % (
                i["id"], " [hidden]" if i["hidden"] else "", " [highlight]" if i.get("highlight") else "", i.get("area") or "(no area)", "fix " if i.get("fix") else "",
                i.get("task") or "-", "  " if i.get("parent") else "", "[group] " if i.get("group") else "", shown(i))
                + ("  (edited)" if i["edited"] else ""))
    elif r.cmd == "line":
        with Locked(NEXT):
            d = read()
            d["line"] = " ".join(r.text.split())
            write_file(d)
    elif r.cmd == "publish":
        publish(r.version, r.at, r.dry_run)


if __name__ == "__main__":
    main()
