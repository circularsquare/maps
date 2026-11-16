"""The changelog for the next published update (T-3424).

One JSON file per release under trainworld/changelog/. The unpublished one is next.json; publish renames it to
<version>.json. An item: id (c001...), text (what the agent wrote), edited (Anita's wording, or null), hidden,
task, by, at, rev (counts changes, so a page saving over a newer change gets refused). The page shows
edited if set, else text. A hidden item stays in the file and out of the game.

    python trainworld/tools/changelog.py add "fixed trains drawn through each other on single track" --task T-3379 --by bugs1
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
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # trainworld/
DIR = os.path.join(ROOT, "changelog")
NEXT = os.path.join(DIR, "next.json")
UPDATES_TS = os.path.join(ROOT, "app", "src", "ui", "updates.ts")
MESSAGES = os.path.join(ROOT, "app", "src", "i18n", "messages", "en.json")
CONTEXT = os.path.join(ROOT, "app", "src", "i18n", "context.json")
LINE_MAX = 80  # checks/i18n.mjs UPDATE_LINE_MAX
CURLY = chr(0x2019)


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


def add_item(text, task=None, by="agent", path=NEXT):
    text = " ".join(text.split())
    if not text:
        raise ValueError("empty text")
    with Locked(path):
        d = read(path)
        n = max([int(i["id"][1:]) for i in d["items"]] + [0]) + 1
        item = {"id": "c%03d" % n, "text": text, "edited": None, "hidden": False,
                "task": task, "by": by, "at": now(), "rev": 1}
        d["items"].append(item)
        write_file(d, path)
    return item


def update_item(item_id, fields, rev, path=NEXT):
    """Her save: edited (string, or null to go back to the agent's text) and/or hidden. Returns (status, item):
    409 with the current item when rev is not the item's."""
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
        item["rev"] += 1
        item["changed"] = now()
        write_file(d, path)
    return 200, item


def shown(item):
    return item["edited"] or item["text"]


def esc(s):
    return json.dumps(s, ensure_ascii=False)


def publish(version, at=None, dry=False, path=NEXT):
    m = re.fullmatch(r"v(\d+(?:\.\d+)+)", version)
    if not m:
        raise SystemExit("version looks like v0.3.6")
    prefix = "updates.v" + m.group(1).replace(".", "")
    d = read(path)
    vis = [i for i in d["items"] if not i["hidden"]]
    line = (d.get("line") or "").strip()
    if not vis:
        raise SystemExit("no visible items: nothing to publish")
    if not line:
        raise SystemExit("no start-menu line: python trainworld/tools/changelog.py line \"update %s: ...\"" % version)
    if len(line) > LINE_MAX:
        raise SystemExit("the line is %d characters, the limit is %d" % (len(line), LINE_MAX))
    strings = {prefix + "Line": line}
    keys = []
    for n, i in enumerate(vis, 1):
        k = "%sChange%d" % (prefix, n)
        strings[k] = shown(i)
        keys.append(k)
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
    entry = ('  {\n    version: "%s",\n    at: "%s",\n    line: "%sLine",\n    changes: [%s],\n  },\n'
             % (version, at, prefix, ", ".join('"%s"' % k for k in keys)))
    entry = entry.replace("\n", eol)
    anchor = "export const UPDATES: Update[] = [" + eol
    if ts.count(anchor) != 1:
        raise SystemExit("cannot find the UPDATES list in updates.ts")
    if dry:
        print(entry)
        for k, v in strings.items():
            print("  %s: %s" % (k, v))
        print("dry run: nothing written (%d hidden items stay out)" % (len(d["items"]) - len(vis)))
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
    ctxl += [(k, "Changelog line for %s, written by an agent, wording checked before release." % version) for k in keys]
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
    a.add_argument("--task")
    a.add_argument("--by", default="agent")
    sub.add_parser("list")
    l = sub.add_parser("line")
    l.add_argument("text")
    p = sub.add_parser("publish")
    p.add_argument("--version", required=True)
    p.add_argument("--at")
    p.add_argument("--dry-run", action="store_true")
    r = ap.parse_args()
    if r.cmd == "add":
        it = add_item(r.text, r.task, r.by)
        print("added %s" % it["id"])
    elif r.cmd == "list":
        d = read()
        print("line: %s" % (d.get("line") or "(none)"))
        for i in d["items"]:
            print("%s%s %s %s%s" % (i["id"], " [hidden]" if i["hidden"] else "", i.get("task") or "-", shown(i),
                                    "  (edited)" if i["edited"] else ""))
    elif r.cmd == "line":
        with Locked(NEXT):
            d = read()
            d["line"] = " ".join(r.text.split())
            write_file(d)
    elif r.cmd == "publish":
        publish(r.version, r.at, r.dry_run)


if __name__ == "__main__":
    main()
