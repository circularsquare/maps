"""Regrouping: new middle levels in the tree without renaming any node id where it is written.

A node id is its path (`nigercongo.bantu.swahili`), so moving a node under a new group changes its
id, and ids are written in ~200 fragments, mappings and normalized CSVs. Instead of rewriting those,
`taxonomy/regroup.txt` lists the moves once, and the three places that turn written ids into drawn
ones apply them:

  * taxonomy/build.py, to the tree read from tree.txt and tree.d/ (after colouring it, so every
    node keeps the colour it had where it is written) and to the mapping values it checks;
  * countries.py, to every country's counts() and `parts` nodes, so the dots carry the new ids;
  * tools/audit_groups.py.

So ids are WRITTEN as before (`{BANTU}.swahili`, `afroasiatic.arabic`) and DRAWN at their new place
(`nigercongo.bantu.zone_g.swahili.swahili`, `afroasiatic.arabic.arabic`). languages.json, the dots,
the tiles and the viewer only ever see the new ids. GROUPING.md says what was grouped and why.

regroup.txt, line by line (`#` starts a comment):

  + NEW.ID | Label [| L C h]     a new group. Without a colour it is generated near its parent's
                                (build.py's slot grid). Indented lines below it are WRITTEN ids
                                (several per line allowed) that move into it with everything under
                                them, keeping their last segment.
  = ID | Label [| Leaf label]    the language ID becomes a group of itself and its dialects: the
                                group takes ID's place (ID's final place if another rule moves it)
                                and its colour; ID's own people go to a new leaf ID.<last>, labelled
                                Leaf label (default: ID's label). ID's existing children stay under
                                the group. Indented lines are written ids (dialects) moving in.
  > ID                          indented written ids move into the existing node ID (at its final
                                place).

A written id is moved by the most specific rule that covers it: an `=` rule on exactly that id,
else the longest member prefix.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE = HERE / "regroup.txt"


class Rules:
    def __init__(self, text):
        self.groups = []      # (new id, label, lch or None, vacated written id or None, leaf label)
        self.prefix = {}      # written id -> new id (node and everything under it)
        self.exact = {}       # written id -> (new id, label or None)  (that node only)
        cur = None
        pending = []          # (kind, written id, group spec) resolved after all prefixes are known
        for ln, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line[0] == ">":
                cur = {"kind": ">", "id": line[1:].strip()}
                self.groups.append(cur)
                continue
            if line[0] in "+=":
                parts = [s.strip() for s in line[1:].split("|")]
                nid, label = parts[0], parts[1]
                extra = parts[2] if len(parts) > 2 and parts[2] else None
                if line[0] == "+":
                    lch = tuple(float(x) for x in extra.split()) if extra else None
                    if lch is not None and len(lch) != 3:
                        raise SystemExit(f"regroup.txt:{ln}: colour must be 'L C h'")
                    cur = {"kind": "+", "id": nid, "label": label, "lch": lch}
                else:
                    cur = {"kind": "=", "id": nid, "label": label, "leaf": extra}
                self.groups.append(cur)
                continue
            if not raw[0].isspace() or cur is None:
                raise SystemExit(f"regroup.txt:{ln}: member line without a group above it")
            for wid in line.split():
                pending.append((wid, cur, ln))
        seen = {}
        for wid, g, ln in pending:
            if wid in seen:
                raise SystemExit(f"regroup.txt:{ln}: {wid} is already moved on line {seen[wid]}")
            seen[wid] = ln
        # `+` members first: their targets do not depend on other rules
        for wid, g, ln in pending:
            if g["kind"] == "+":
                self.prefix[wid] = g["id"] + "." + wid.rsplit(".", 1)[-1]
        # `=` groups sit where their language ends up; in order, so a later one may sit inside an
        # earlier one's move
        for g in self.groups:
            if g["kind"] in "=>":
                g["final"] = self.move(g["id"])
                if g["kind"] == "=":
                    last = g["id"].rsplit(".", 1)[-1]
                    self.exact[g["id"]] = (g["final"] + "." + last, g["leaf"])
                for wid, gg, ln in pending:
                    if gg is g:
                        self.prefix[wid] = g["final"] + "." + wid.rsplit(".", 1)[-1]
                self.__dict__.pop("_memo", None)      # rules changed: forget answers so far

    def move(self, nid):
        """The drawn id of a written id."""
        if not isinstance(nid, str):
            return nid
        memo = self.__dict__.setdefault("_memo", {})
        if nid not in memo:
            memo[nid] = self._move(nid)
        return memo[nid]

    def _move(self, nid):
        if nid in self.exact:
            return self.exact[nid][0]
        best = None
        for old in self.prefix:
            if (nid == old or nid.startswith(old + ".")) and (best is None or len(old) > len(best)):
                best = old
        return nid if best is None else self.prefix[best] + nid[len(best):]


_RULES = None


def rules():
    global _RULES
    if _RULES is None:
        _RULES = Rules(FILE.read_text(encoding="utf-8") if FILE.exists() else "")
    return _RULES


def move(nid):
    return rules().move(nid)


def apply(nodes):
    """Regroup a coloured tree in place order: nodes are dicts with id, label, parent, lch (the
    as-written tree, coloured). Returns the new node list (ids moved, groups added, parents
    recomputed); a new `+` group without a colour gets lch None, for build.py to generate."""
    r = rules()
    out, where = [], {}
    for n in nodes:
        m = dict(n)
        m["id"] = r.move(n["id"])
        if n["id"] in r.exact and r.exact[n["id"]][1]:
            m["label"] = r.exact[n["id"]][1]
        if m["id"] in where:
            # the same node written both ways (old and new id): keep the first
            continue
        where[m["id"]] = len(out)
        out.append(m)
    lch_of = {n["id"]: n.get("lch") for n in nodes}
    label_of = {n["id"]: n["label"] for n in nodes}
    added = []
    for g in r.groups:
        if g["kind"] == ">":
            continue
        gid = g["id"] if g["kind"] == "+" else g["final"]
        if gid in where:
            raise SystemExit(f"regroup.txt: group {gid} is already a node")
        if g["kind"] == "=":
            # its language may be missing from a partial (--only) tree: then a generated colour
            node = {"id": gid, "label": g["label"], "lch": lch_of.get(g["id"])}
        else:
            node = {"id": gid, "label": g["label"], "lch": g["lch"]}
        where[gid] = None
        added.append(node)
    # a group goes just before its first member, so parents still come before children
    first = {}
    for i, n in enumerate(out):
        for g in added:
            if n["id"].startswith(g["id"] + ".") and g["id"] not in first:
                first[g["id"]] = i
    res, by_pos = [], {}
    for g in added:
        if g["id"] in first:          # a group none of whose members is in this tree is dropped
            by_pos.setdefault(first[g["id"]], []).append(g)
    for i, n in enumerate(out):
        # outer groups before inner ones
        for g in sorted(by_pos.get(i, []), key=lambda g: g["id"].count(".")):
            res.append(g)
        res.append(n)
    ids = {n["id"] for n in res}
    for n in res:
        p = n["id"].rsplit(".", 1)[0] if "." in n["id"] else None
        if p and p not in ids:
            raise SystemExit(f"regroup: {n['id']}: parent {p} is not a node")
        n["parent"] = p
    del label_of
    return res
