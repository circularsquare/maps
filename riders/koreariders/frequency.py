# -*- coding: utf-8 -*-
"""선구별 열차종별 운행횟수 — how many trains a day run over each section.

`6. 운전` sheets 2(5)–2(7) break the network into 117 sections and give
trains/day on each, by train type. That is the one published thing that speaks
directly to what happens at a junction, because the count *changes* there and
the change is the service stepping on or off:

    경부선  서울-금천구청   새마을 26 + 무궁화 40 = 66
            금천구청-의왕   26 + 40 = 66
            의왕-천안      26 + 40 = 66
            천안-조치원     20 + 31 = 51      <- 15 trains leave at 천안
    장항선  천안-신창       6 + 9   = 15      <- and here they are

Flat from 서울 to 천안 means nothing joins or leaves in between, which is how
we know a junction step at 용산 is wrong however well it fits the 승하차.

    python frequency.py            # print every line's section profile
"""

import re
import sys

import openpyxl

import lines as LN

MEMBER = "1.지역간철도/6. 운전(1~6)_완.xlsx"
SHEETS = ("2(5)", "2(6)", "2(7)")

# The sheets name their lines a little differently from the passenger tables,
# and a few of ours are too new to appear at all (강릉선, 중부내륙선) -- those
# simply get no constraint.
ALIAS = {
    "수서평택선": "수서고속선",
    "경의1선": "경의선",
}

# Sheet column heading -> the yearbook passenger types it covers. 전동차 is
# 광역전철 and has no 승하차 row of its own, so it is left out; ITX-청춘 is a
# different service from ITX-새마을 and only runs on 경춘선.
#
# A column can cover more than one passenger type, and that is why counts are
# kept per *column* rather than exploded to types. The sheet has one 새마을
# column carrying both 새마을 and ITX-새마을 services, so a line whose types are
# CONVENTIONAL used to add those trains twice -- 경부선 서울-금천구청 came out
# at 92 trains against the 새마을 26 + 무궁화 40 = 66 that actually run, a 40 %
# inflation, and every conventional line was affected. It never showed, because
# the only things read off these counts were whether a total was zero and which
# way it stepped, and a consistent double-count changes neither. It matters the
# moment a count is used as a denominator.
COLUMNS = {
    "KTX": ["KTX"],
    "SRT": ["SRT"],
    "고속열차": ["KTX", "SRT"],
    "새마을": ["새마을", "ITX-새마을"],
    "ITX-청춘": ["ITX-새마을"],
    "무궁화": ["무궁화"],
    "통근": ["통근"],
    # 전동차 is 광역전철 and has no 승하차 row, so it never joins an intercity
    # total -- "전동차" is not in any line's TYPES, so `total()` cannot pick it
    # up. It is read anyway because it is the one thing that tells a section
    # with no intercity service apart from a section with no service at all,
    # and the map has to say which. 경원선 청량리-광운대 runs 184 commuter trains
    # a day and no intercity ones; 소요산-신탄리 runs nothing whatever.
    "전동차": ["전동차"],
}

COMMUTER = "전동차"


def total(runs, kinds):
    """Trains a day over the columns serving any of `kinds`, each counted once."""
    want = set(kinds)
    return sum(n for col, n in runs.items()
               if want.intersection(COLUMNS.get(col, (col,))))


def _flat(v):
    return re.sub(r"\s+", "", str(v)) if v is not None else ""


def sections():
    """[(line, from, to, {train type: trains per day}), ...] in sheet order."""
    wb = openpyxl.load_workbook(LN._open(MEMBER), data_only=True)
    out = []
    for name in SHEETS:
        ws = wb[name]
        # 2(5) puts 선로용량 one column left of the other two, so find the
        # headings rather than assuming where they sit.
        head = {}
        for c in range(1, 20):
            for r in (4, 5):
                h = _flat(ws.cell(row=r, column=c).value)
                if h in COLUMNS:
                    head[c] = h
        line = ""
        for r in range(6, ws.max_row + 1):
            a = _flat(ws.cell(row=r, column=1).value)
            if a:
                line = ALIAS.get(a, a)
            sec = _flat(ws.cell(row=r, column=3).value)
            if not sec or "-" not in sec:
                continue
            # Keyed by the sheet's own column, so a column covering two
            # passenger types is still one set of trains. See COLUMNS.
            runs = {}
            for c, h in head.items():
                v = ws.cell(row=r, column=c).value
                if isinstance(v, (int, float)) and v:
                    runs[h] = runs.get(h, 0) + int(v)
            frm, to = sec.split("-", 1)
            out.append((line, frm, to, runs))
    return out


_BY_LINE = None


def by_line():
    """{line: [(from, to, runs), ...]}, sections kept in their published order.

    Cached -- opening the workbook takes long enough that doing it once per
    line turns a report into a coffee break.
    """
    global _BY_LINE
    if _BY_LINE is None:
        out = {}
        for line, frm, to, runs in sections():
            out.setdefault(line, []).append((frm, to, runs))
        _BY_LINE = out
    return _BY_LINE


def changes(canon, kinds):
    """{station: (before, after)} trains a day either side of a boundary.

    Only interior boundaries — where one section ends and the next begins — say
    anything about a junction. A station that is not a boundary has no step at
    all, which is the useful half of this: it forbids one.
    """
    secs = by_line().get(canon)
    if not secs:
        return None
    tot = [total(runs, kinds) for _, _, runs in secs]
    out = {}
    for i in range(len(secs) - 1):
        if secs[i][1] == secs[i + 1][0]:        # contiguous, so a real boundary
            out[secs[i][1]] = (tot[i], tot[i + 1])
    return out


def runs_along(canon, kinds, stops):
    """Trains a day on the section each segment falls in.

    `stops` must be in the line's own 기점 -> 종점 order, the order the sheet
    lists its sections in. Returns one count per segment, so one fewer than
    there are stops, or None where the line has no section data.

    The zero counts are the point. 경원선 north of 소요산 has been shut for the
    전철 works and the sheets show nothing running on either of its last two
    sections -- no trains means no passengers, and that is a far harder thing to
    say than any prior on a junction step.
    """
    secs = by_line().get(canon)
    if not secs:
        return None
    tot = [total(runs, kinds) for _, _, runs in secs]
    starts = {}
    for i, (frm, _, _) in enumerate(secs):
        starts.setdefault(frm, i)
    out, cur = [], 0
    for nm in stops[:-1]:
        hit = match(nm, starts)
        if hit is not None:
            cur = hit
        out.append(tot[cur])
    return out


def service_along(canon, kinds, stops):
    """Per segment, (intercity trains, commuter trains) a day each way.

    Same walk as `runs_along`, but it also carries 전동차 so a caller can tell
    "no intercity service here, but the 광역전철 runs" from "nothing runs here
    at all". The map draws both cases the same way and needs to label them
    differently -- 경원선 north of 청량리 is Seoul's line 1, busy and fully
    drawn by the metro layer, while 소요산-백마고지 was shut.

    Counts are one way; the sheet's own header says 작성기준: 편도.
    """
    secs = by_line().get(canon)
    if not secs:
        return None
    inter = [total(runs, kinds) for _, _, runs in secs]
    comm = [runs.get(COMMUTER, 0) for _, _, runs in secs]
    starts = {}
    for i, (frm, _, _) in enumerate(secs):
        starts.setdefault(frm, i)
    out, cur = [], 0
    for nm in stops[:-1]:
        hit = match(nm, starts)
        if hit is not None:
            cur = hit
        out.append((inter[cur], comm[cur]))
    return out


def match(nm, ch):
    """Look a chain station up in a `changes()` table.

    The sheets abbreviate — 대전조 for 대전조차장 — so fall back to the longest
    key that prefixes the name. Longest matters: both 대전 and 대전조 prefix
    대전조차장 and only one of them is the junction.
    """
    if nm in ch:
        return ch[nm]
    best = None
    for k in ch:
        if nm.startswith(k) and (best is None or len(k) > len(best)):
            best = k
    return ch[best] if best else None


def main():
    table, _ = LN.resolve()
    known = by_line()
    print("%-11s %-9s %8s %8s %8s   %s"
          % ("line", "at", "before", "after", "step", "as a share"))
    print("-" * 66)
    for canon in sorted(table):
        spec = table[canon]
        if "error" in spec or canon not in known:
            continue
        ch = changes(canon, spec["types"])
        for nm, (a, b) in sorted(ch.items()):
            mark = "" if a != b else "   (no step -- forbids one)"
            print("%-11s %-9s %8d %8d %+8d   %s%s"
                  % (canon, nm, a, b, b - a,
                     "%.0f%%" % (100.0 * (b - a) / a) if a else "-", mark))
    missing = [c for c in sorted(table)
               if "error" not in table[c] and c not in known]
    print("\nno section data: %s" % ", ".join(missing))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
