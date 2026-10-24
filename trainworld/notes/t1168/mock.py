"""T-1168 mockups: the line panel's stops diagram with related lines, drawn from a real save.

python mock.py <save> <line index> <out.html> [first row name] [last row name] [--hover k:text ...]

Classification is relate.py's. Layout: rows are the stations anyone drawn stops at, in order along
the selected line (side lines' own stations projected onto it); side lines get columns right then
left of the spine, inner first, packed by span; a side line that goes on past its last meeting turns
out to its side's gutter half a row beyond, ending in its bullet; crossings are horizontal bars
behind the row. Writes one HTML page holding the light and the dark panel side by side.
"""
import html, math, re, sys
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import relate

ROW = 25
COL = 10
BAR = 6
GUT = 28
SLOTS = [("R", 1), ("L", 1), ("R", 2), ("L", 2), ("R", 3), ("L", 3)]
NAME_W = 340  # the dock's inner width


def symbol(name):
    t = name.split()
    rest = t[1:] if len(t) > 1 else t
    if len(rest) == 1 and len(rest[0]) <= 3:
        return rest[0]
    if len(rest) >= 2 and len(rest[-1]) <= 3 and rest[-1].isupper():
        return rest[-1]
    return "".join(w[0] for w in rest if w[0].isalnum())[:2].upper()


def label(name):
    t = name.split()
    return " ".join(t[1:]).replace("subway", "").strip().title() if len(t) > 1 else name


def short(nm):
    return nm.split(" / ")[0]


def build(save, li):
    d, cx, routes = relate.load(save)
    L = routes[li]
    rel = relate.classify(d, cx, routes, li)
    d["_routes"], d["_rel"] = routes, rel
    nodes, lines = d["nodes"], d["lines"]
    rows = {}

    def row(key, sL, name, side):
        if key not in rows:
            rows[key] = {"key": key, "sL": sL, "name": short(name), "Lstop": False, "side": side, "stops": set(), "cross": []}
        return rows[key]
    for n, s in L.nodes:
        if n in cx:
            r = row(cx[n], s, nodes[n]["name"], False)
            if n in L.stop_set:
                r["Lstop"] = True
    items = []
    for r in rel:
        O = routes[r["o"]]
        if r["kind"] != "alongside":
            continue
        keys = set()
        for n, s in O.nodes:
            if n not in cx or s < r["sO0"] - 1 or s > r["sO1"] + 1:
                continue
            k = cx[n]
            if k not in rows:
                if n not in O.stop_set:
                    continue
                _, sl = L.nearest((nodes[n]["x"], nodes[n]["y"]), r["sL0"], r["sL1"])
                row(k, sl, nodes[n]["name"], True)
            if n in O.stop_set:
                rows[k]["stops"].add(r["o"])
            keys.add(k)
        ms = sorted(r["meet"], key=lambda m: m["sL"])
        top, bot = ms[0], ms[-1]
        dirn = 1 if (bot["sO"] >= top["sO"]) else -1
        def beyond(m, up):
            # the part of O past this meeting, away from the span
            toward_start = (dirn > 0) == up
            so = m["sO"] if "sO2" not in m else (min(m["sO"], m["sO2"]) if toward_start else max(m["sO"], m["sO2"]))
            if toward_start:
                past = [k for k, s in enumerate(O.stop_s) if s is not None and s < so - 1]
                return (past, lines[r["o"]]["stops"][0]) if so > 1 else None
            past = [k for k, s in enumerate(O.stop_s) if s is not None and s > so + 1]
            return (past, lines[r["o"]]["stops"][-1]) if so < O.length - 1 else None
        items.append({"o": r["o"], "keys": keys, "sL0": r["sL0"], "sL1": r["sL1"], "track": r["track"], "rel": r,
                      "up": beyond(top, True), "down": beyond(bot, False), "stations": r["stations"]})
    for r in rel:
        if r["kind"] == "crossing":
            for m in r["meet"]:
                if m["kind"] == "station" and m["cx"] in rows and r["o"] not in rows[m["cx"]]["cross"]:
                    rows[m["cx"]]["cross"].append(r["o"])
    order = sorted(rows.values(), key=lambda x: x["sL"])
    # rows anyone drawn stops at; the crossing lines of a dropped row are lost with it
    keep = []
    for x in order:
        if x["Lstop"] or x["stops"]:
            keep.append(x)
    idx = {x["key"]: i for i, x in enumerate(keep)}
    for it in items:
        rs = [idx[k] for k in it["keys"] if k in idx]
        # the rows inside the span count too: the column passes them
        inside = [i for i, x in enumerate(keep) if it["sL0"] - 1 <= x["sL"] <= it["sL1"] + 1]
        rs = rs + inside
        it["r0"], it["r1"] = (min(rs), max(rs)) if rs else (None, None)
        it["shared"] = sum(1 for k in it["keys"] if k in idx and keep[idx[k]]["Lstop"] and it["o"] in keep[idx[k]]["stops"])
    # a side line beside the spine over fewer than three rows is drawn as crossings at its rows
    short_ = [it for it in items if it["r0"] is not None and it["r1"] - it["r0"] < 2]
    for it in short_:
        for k in it["keys"]:
            if k in idx and it["o"] not in keep[idx[k]]["cross"]:
                keep[idx[k]]["cross"].append(it["o"])
    items = [it for it in items if it["r0"] is not None and it["r1"] - it["r0"] >= 2]
    items.sort(key=lambda it: (-it["shared"], -(it["r1"] - it["r0"])))
    # a side line stopping more often than the spine over its span goes right, as often or less
    # left; most stops shared nearest; the other side when its own is full
    for it in items:
        span = keep[it["r0"]:it["r1"] + 1]
        extra = sum(1 for x in span if it["o"] in x["stops"] and not x["Lstop"])
        skipped = sum(1 for x in span if x["Lstop"] and it["o"] not in x["stops"])
        it["pref"] = "R" if extra > skipped else "L"
    taken = {s: [] for s in SLOTS}
    over = []
    for it in items:
        a = it["r0"] - (0.6 if it["up"] else 0.1)
        b = it["r1"] + (0.6 if it["down"] else 0.1)
        slots = SLOTS if not it["pref"] else [s for s in SLOTS if s[0] == it["pref"]] + [s for s in SLOTS if s[0] != it["pref"]]
        for s in slots:
            if all(b < x0 or a > x1 for x0, x1 in taken[s]):
                taken[s].append((a, b)); it["slot"] = s; break
        else:
            over.append(it)
    items = [it for it in items if "slot" in it]
    return d, L, keep, items, over, rel


def chip_svg(x, y, text, colour, anchor="middle"):
    w = max(16, 7 * len(text) + 8)
    if anchor == "start": x0 = x
    elif anchor == "end": x0 = x - w
    else: x0 = x - w / 2
    return (f'<rect x="{x0:.1f}" y="{y-8:.1f}" width="{w}" height="16" rx="8" fill="{colour}"/>'
            f'<text x="{x0 + w/2:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">{html.escape(text)}</text>'), w


def panel(d, L, rows, items, over, theme, hovers):
    lines, nodes = d["lines"], d["nodes"]
    Lcol = lines[L.li]["colour"]
    nL = max([s[1] for s in (it["slot"] for it in items) if s[0] == "L"] or [0])
    nR = max([s[1] for s in (it["slot"] for it in items) if s[0] == "R"] or [0])
    gl = GUT if nL else 6
    X0 = gl + nL * COL + 6
    gr = GUT if nR else 6
    GW = X0 + nR * COL + 6 + gr
    def xs(slot):
        side, k = slot
        return X0 + (k * COL if side == "R" else -k * COL)
    H = len(rows) * ROW + 4
    yc = lambda i: i * ROW + ROW / 2 + 2
    out = []
    # crossings behind everything: up to two a row, ordered as the transfer bullets are
    for i, r in enumerate(rows):
        cs = r["cross"][:2]
        for k, o in enumerate(cs):
            dy = 0 if len(cs) == 1 else (-3 if k == 0 else 3)
            out.append(f'<line x1="2" x2="{GW-2}" y1="{yc(i)+dy}" y2="{yc(i)+dy}" stroke="{lines[o]["colour"]}" stroke-width="3" data-k="cross"/>')
    # turns out to the gutter, grouped by side and the gap between two rows: lines leaving
    # downward first, outermost first, then lines coming in, innermost first; 7 px apart, so a
    # bundle leaving together fans out without crossing itself. Their bullets stack in the gutter.
    turns = {}
    for it in items:
        if it["up"]:
            turns.setdefault((it["slot"][0], it["r0"] - 0.5), []).append((1, it["slot"][1], it))
        if it["down"]:
            turns.setdefault((it["slot"][0], it["r1"] + 0.5), []).append((0, -it["slot"][1], it))
    turn_y = {}
    chips_at = []
    for (sd, g), ev in turns.items():
        ev.sort(key=lambda t: (t[0], t[1]))
        mid = g * ROW + ROW / 2 + 2
        n = len(ev)
        for j, (kind, _, it) in enumerate(ev):
            y = mid + (j - (n - 1) / 2) * 7
            turn_y[(id(it), "up" if kind == 1 else "down")] = y
            chips_at.append((sd, mid + (j - (n - 1) / 2) * 17, it))
    # side columns, outer first so an inner line's turn passes over them
    for it in sorted(items, key=lambda it: -it["slot"][1]):
        x = xs(it["slot"]); col = lines[it["o"]]["colour"]
        y0, y1 = yc(it["r0"]), yc(it["r1"])
        side = -1 if it["slot"][0] == "L" else 1
        gx = 14 if side < 0 else GW - 14
        segs = []
        yend = turn_y[(id(it), "down")] - 5 if it["down"] else y1
        if it["up"]:
            yt = turn_y[(id(it), "up")]
            segs.append(f'M{gx},{yt} H{x + side*5} Q{x},{yt} {x},{yt+5} V{yend}')
        else:
            segs.append(f'M{x},{y0} V{yend}')
        if it["down"]:
            yb = turn_y[(id(it), "down")]
            segs.append(f'M{x},{yb-5} Q{x},{yb} {x + side*5},{yb} H{gx}')
        p = " ".join(segs)
        out.append(f'<path d="{p}" fill="none" stroke="var(--bg)" stroke-width="{BAR+3}"/>')
        out.append(f'<path d="{p}" fill="none" stroke="{col}" stroke-width="{BAR}" data-k="side"/>')
    # the spine
    li = [i for i, r in enumerate(rows) if r["Lstop"]]
    out.append(f'<line x1="{X0}" x2="{X0}" y1="{yc(li[0])}" y2="{yc(li[-1])}" stroke="{Lcol}" stroke-width="{BAR}"/>')
    # bullets: at a turn's end in the gutter, else level with the column's first row
    for it in items:
        if not it["up"] and not it["down"]:
            chips_at.append((it["slot"][0], yc(it["r0"]), it))
    # no two bullets in a gutter overlap: each sits at least 17 px below the one above
    last = {"L": -99.0, "R": -99.0}
    for sd, y, it in sorted(chips_at, key=lambda c: c[1]):
        y = max(y, last[sd] + 17)
        last[sd] = y
        gx = 14 if sd == "L" else GW - 14
        out.append(chip_svg(gx, y, symbol(lines[it["o"]]["name"]), lines[it["o"]]["colour"])[0])
    # dots: one pill over each run of neighbouring columns that stop
    order_x = sorted([(X0, "spine")] + [(xs(it["slot"]), it) for it in items], key=lambda t: t[0])
    for i, r in enumerate(rows):
        active = []
        for x, it in order_x:
            if it == "spine":
                if li[0] <= i <= li[-1]:
                    active.append((x, r["Lstop"]))
            elif it["r0"] <= i <= it["r1"]:
                active.append((x, it["o"] in r["stops"]))
        run = []
        for x, st in active + [(None, False)]:
            if st:
                run.append(x)
            elif run:
                a, b = run[0], run[-1]
                big = 10 if any(rr for rr in [r["cross"]]) else 8
                out.append(f'<rect x="{a - big/2 - 1}" y="{yc(i) - big/2 - 1}" width="{b - a + big + 2}" height="{big + 2}" rx="{big/2 + 1}" fill="#fff" stroke="#2b2b2b" stroke-width="2"/>')
                run = []
    svg = f'<svg width="{GW}" height="{H}" viewBox="0 0 {GW} {H}" style="position:absolute;left:0;top:0;overflow:visible">{"".join(out)}</svg>'
    # names, and the crossing bullets as the transfer bullets are
    texts = []
    for i, r in enumerate(rows):
        chips = ""
        for o in r["cross"][:2]:
            c, w = chip_svg(0, 8, symbol(lines[o]["name"]), lines[o]["colour"], "start")
            chips += f'<svg width="{w}" height="16" style="flex:none">{c}</svg>'
        if len(r["cross"]) > 2:
            chips += '<span class="more">+</span>'
        cls = "nm" + ("" if r["Lstop"] else " muted")
        texts.append(f'<div class="row" style="top:{i*ROW+2}px;left:{GW+4}px"><span class="{cls}">{html.escape(r["name"])}</span>{chips}</div>')
    tips = "".join(f'<div class="mark" style="left:{x}px;top:{y}px">{chr(97 + k)}</div>' for k, (x, y, t) in enumerate(hovers))
    head = ('<div class="h">Stops<span class="tog on" title="">' + RELATED_ICON + '</span>'
            '<span class="right muted small">boardings in past 24 h</span></div>')
    sym = symbol(lines[L.li]["name"])
    bw = max(30, 10 * len(sym) + 14)
    big = (f'<svg width="{bw}" height="30"><rect width="{bw}" height="30" rx="15" fill="{Lcol}"/>'
           f'<text x="{bw/2}" y="21" text-anchor="middle" font-size="17" font-weight="700" fill="#fff">{html.escape(sym)}</text></svg>')
    return (f'<div class="panel {theme}"><div class="title">{big}<span class="big">{html.escape(label(lines[L.li]["name"]))}</span></div>'
            + head + f'<div class="strip" style="height:{H}px">{svg}{"".join(texts)}{tips}</div></div>')


RELATED_ICON = ('<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">'
                '<path d="M8 1.5V14.5"/><path d="M8 4.5C8 7 4 7 4 9.5V14.5"/><path d="M8 4.5C8 7 12 7 12 9.5V14.5"/></svg>')

CSS = """
body { margin: 0; display: flex; gap: 0; font-family: Nunito, 'Zen Maru Gothic', system-ui, sans-serif; font-size: 14px; line-height: 1.45; }
.panel { --muted: color-mix(in srgb, var(--text) 58%, var(--bg)); --rule: color-mix(in srgb, var(--text) 22%, var(--bg));
  background: var(--bg); color: var(--text); width: 340px; padding: 12px 14px 20px; box-sizing: content-box; }
.light { --bg: #fbf7f0; --text: #4b403a; --accent: #d9788e; --map-land: #f2f2ef; --map-ink: #2b2b2b; }
.dark { --bg: #262120; --text: #ebe3de; --accent: #e08aa0; --map-land: #2a2c30; --map-ink: #e8e8e8; }
.title { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.big { font-size: 18px; font-weight: 700; }
.h { display: flex; align-items: center; font-weight: 700; margin: 10px 0 6px; border-top: 1px solid var(--rule); padding-top: 10px; gap: 8px; }
.h .right { margin-left: auto; font-weight: 400; }
.muted { color: var(--muted); } .small { font-size: 12px; }
.tog { display: inline-grid; place-items: center; width: 22px; height: 22px; border: 1px solid var(--rule); color: var(--text); }
.tog.on { background: color-mix(in srgb, var(--accent) 18%, var(--bg)); border-color: var(--accent); }
.strip { position: relative; }
.row { position: absolute; right: 0; height: 25px; display: flex; align-items: center; gap: 4px; white-space: nowrap; }
.row .nm { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.row .more { width: 10px; font-size: 12px; color: var(--muted); text-align: center; }
.tag { font-size: 12px; color: var(--map-ink); background: var(--map-land); padding: 0 4px; border: 1px solid var(--rule); max-width: 300px; }
.mark { position: absolute; z-index: 5; width: 16px; height: 16px; margin: -8px 0 0 -8px; border-radius: 50%; background: #222; color: #fff;
  font-size: 11px; font-weight: 700; display: grid; place-items: center; }
.notes { width: 260px; }
.note { display: flex; gap: 8px; align-items: flex-start; margin: 0 0 10px; }
.note b { flex: none; width: 16px; height: 16px; border-radius: 50%; background: #222; color: #fff; font-size: 11px; display: grid; place-items: center; }
"""


def routes_of(d):
    return d["_routes"]


def rel_of(d):
    return d["_rel"]


def and_list(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def km(m):
    return f"{m/1000:.1f} km"


def hover_text(d, L, rows, items, spec):
    """The hover a drawn thing would have, and where to show it (x, y in the strip)."""
    lines, nodes = d["lines"], d["nodes"]
    me = label(lines[L.li]["name"])
    kind, _, arg = spec.partition(":")
    yc = lambda i: i * ROW + ROW / 2 + 2
    if kind in ("col", "up", "down"):
        it = items[int(arg)]
        o = label(lines[it["o"]]["name"])
        if kind == "col":
            own = sum(1 for x in rows[it["r0"]:it["r1"] + 1] if it["o"] in x["stops"] and not x["Lstop"])
            t = f"{o} runs beside {me} from {rows[it['r0']]['name']} to {rows[it['r1']]['name']}\n{it['shared']} stops in common\n{own} {'stop' if own == 1 else 'stops'} of its own"
            if it["track"]:
                t += "\nOn the same track"
            for g in it["rel"]["gaps"]:
                st, run, lstep = g
                if st > relate.STRAY_FLOOR_M:
                    t += f"\n{km(run)} on its own track beside {km(lstep)} of {me}\nUp to {st:.0f} m from {me}'s track"
            return 0, (it["r0"] + it["r1"]) / 2 * ROW, t
        past, term = it[kind]
        r = it["r0"] - 0.5 if kind == "up" else it["r1"] + 0.5
        # back beside the spine further along: the stops between, not the far terminus
        back = [j for j in items if j is not it and j["o"] == it["o"] and ((j["r0"] > it["r1"]) == (kind == "down"))]
        if back:
            j = min(back, key=lambda j: abs(j["r0"] - r))
            a, b = sorted([it["rel"]["sO0"], it["rel"]["sO1"], j["rel"]["sO0"], j["rel"]["sO1"]])[1:3]
            n = sum(1 for s in routes_of(d)[it["o"]].stop_s if s is not None and a + 1 < s < b - 1)
            where = rows[j["r0"]]["name"] if kind == "down" else rows[j["r1"]]["name"]
            t = (f"{o} leaves here\nBack beside {me} at {where} after {n} stops of its own" if kind == "down"
                 else f"{o} joins here\nIt left {me} at {where} {n} stops back")
            return 0, yc(r) + 10, t
        t = f"{o} goes on to {short(nodes[term]['name'])}\n{len(past)} more {'stop' if len(past) == 1 else 'stops'}"
        return 0, yc(r) + 10, t
    i = next(k for k, x in enumerate(rows) if x["name"].startswith(arg))
    x = rows[i]
    if kind == "dot":
        at = [lines[o]["name"] for o in []]
        stop = ([me] if x["Lstop"] else []) + [label(lines[it["o"]]["name"]) for it in items if it["r0"] <= i <= it["r1"] and it["o"] in x["stops"]]
        passing = ([me] if not x["Lstop"] else []) + [label(lines[it["o"]]["name"]) for it in items if it["r0"] <= i <= it["r1"] and it["o"] not in x["stops"]]
        t = f"{and_list(stop)} {'stops' if len(stop) == 1 else 'stop'} at {x['name']}"
        if passing:
            t += f"\n{and_list(passing)} {'passes' if len(passing) == 1 else 'pass'}"
        return 0, yc(i) + 10, t
    o = x["cross"][0]
    t = f"{label(lines[o]['name'])} crosses at {x['name']}"
    # met before or after: the gap that kept the meetings apart
    groups = sorted([g for g in rel_of(d) if g["o"] == o], key=lambda g: g["sO0"])
    for k, g in enumerate(groups):
        if g["kind"] == "crossing" and k > 0 and any(m.get("cx") == x["key"] for m in g["meet"]):
            for m in g["meet"]:
                if "gap" in m:
                    st, run, lstep = m["gap"]
                    prev = [p for p in groups[k - 1]["meet"] if p["kind"] == "station"]
                    at = short(d["nodes"][prev[-1]["Onode"]]["name"]) if prev else "the last meeting"
                    t += f"\nIt left {me} at {at} {km(run)} back\nIts track strays up to {km(st)} from {me}'s in between"
    return 0, yc(i) + 10, t


def page(save, li, out, first=None, last=None, hovers=()):
    d, L, rows, items, over, rel = build(save, li)
    if first:
        a = next(i for i, r in enumerate(rows) if r["name"].startswith(first))
        b = next(i for i, r in enumerate(rows) if r["name"].startswith(last)) if last else len(rows) - 1
        for it in items:
            it["r0"] -= a; it["r1"] -= a
            if it["r0"] < 0: it["r0"] = 0; it["up"] = it["up"] if it["r0"] >= 0 else None
            if it["r1"] > b - a: it["r1"] = b - a
        rows = rows[a:b + 1]
        items = [it for it in items if it["r1"] > it["r0"] and it["r0"] <= b - a]
    hv = []
    nL = max([it["slot"][1] for it in items if it["slot"][0] == "L"] or [0])
    nR = max([it["slot"][1] for it in items if it["slot"][0] == "R"] or [0])
    X0 = (GUT if nL else 6) + nL * COL + 6
    GW = X0 + nR * COL + 6 + (GUT if nR else 6)
    for h in hovers:
        spec, _, at = h.partition("@")
        _, y, t = hover_text(d, L, rows, items, spec)
        kind, _, arg = spec.partition(":")
        if kind in ("col", "up", "down"):
            it = items[int(arg)]
            side = it["slot"][0]
            if kind == "col":
                x = X0 + (1 if side == "R" else -1) * it["slot"][1] * COL
                y = (it["r0"] + it["r1"]) / 2 * ROW + ROW / 2 + 2
                x += 9 if side == "R" else -9
            else:
                x = (14 + 16) if side == "L" else GW - 14 - 16
                y -= 10
        elif kind == "dot":
            x, y = GW - 4, y - 10
        else:
            x, y = 6, y - 10
        hv.append((x, int(y), t))
        print("hover", spec, "->", t.replace("\n", " | "))
    notes = "".join(f'<div class="note"><b>{chr(97 + k)}</b><div class="tag">{html.escape(t).replace(chr(10), "<br>")}</div></div>' for k, (_, _, t) in enumerate(hv))
    body = (panel(d, L, rows, items, over, "light", hv) + panel(d, L, rows, items, over, "dark", [])
            + (f'<div class="panel light notes"><div class="h">Hovers</div>{notes}</div>' if hv else ""))
    open(out, "w", encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><link rel=stylesheet href='https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap'><style>{CSS}</style>{body}")
    # report for the notes
    lines = d["lines"]
    print("rows", len(rows), "columns", [(lines[it["o"]]["name"], it["slot"], it["r0"], it["r1"], "up" if it["up"] else "", "down" if it["down"] else "", "shared", it["shared"]) for it in items])
    print("overflow", [lines[it["o"]]["name"] for it in over])
    print("crossings", sum(len(r["cross"]) for r in rows))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--hover=")]
    hov = [a[len("--hover="):] for a in sys.argv[1:] if a.startswith("--hover=")]
    page(args[0], int(args[1]), args[2], args[3] if len(args) > 3 else None, args[4] if len(args) > 4 else None, hov)
