"""Fills china_holidays.json with the usual pattern for each game year's real year that has no State
Council notice yet (T-1157). Years marked "notice" in the file are kept as they are; a year marked
"pattern" is rebuilt here. When a notice comes out, replace that year's entry by hand with the notice's
days and "source": "notice". The rules are in DECISIONS.md (2026-10-24, T-1157). Run from the maps root:
    venv/Scripts/python.exe trainworld/pipeline/china_holidays_pattern.py
"""

import datetime as dt
import json
import pathlib

import holidays

FILE = pathlib.Path(__file__).resolve().parent / "china_holidays.json"
YEARS = (2026, 2027, 2029, 2030, 2031, 2033, 2034)  # the game's real years (holidays_json.YEARS)
ONE = dt.timedelta(days=1)


def weekend(d):
    return d.weekday() >= 5


def lunar_dates(y):
    """The year's moveable festivals from the package: Spring Festival's eve, Qingming, Dragon Boat, Mid-Autumn."""
    cal = holidays.country_holidays("CN", years=y, observed=False, language="en_US")
    found = {}
    for d, n in cal.items():
        for key, text in (("eve", "New Year's Eve"), ("qingming", "Tomb-Sweeping"), ("dragon", "Dragon Boat"),
                          ("autumn", "Mid-Autumn")):
            if text in n and key not in found:
                found[key] = d
    assert len(found) == 4, (y, found)
    return found


def single(d):
    """A one-day festival: a Tuesday takes the Monday before (the Saturday before is worked), a Thursday
    the Friday after (the Sunday after is worked); a weekend one takes the weekend and the Monday."""
    wd = d.weekday()
    if wd == 0:
        return [d - 2 * ONE, d], []
    if wd == 1:
        return [d - 2 * ONE, d], [d - 3 * ONE]
    if wd == 2:
        return [d, d], []
    if wd == 3:
        return [d, d + 2 * ONE], [d + 3 * ONE]
    if wd == 4:
        return [d, d + 2 * ONE], []
    if wd == 5:
        return [d, d + 2 * ONE], []
    return [d - ONE, d + ONE], []


def multi(first, days, cap=None, want=None):
    """A festival of several statutory days: weekday substitutes for statutory days on a weekend, then up
    to two more weekdays swapped for working weekend days, and the weekend that follows. `cap` is the
    longest break (the most swaps that fit); `want` is a length to reach with the fewest swaps. Returns
    the first and last day off and the swapped days' count."""
    stat = [first + i * ONE for i in range(days)]
    subs = sum(weekend(d) for d in stat)

    def end_for(k):
        n, d = subs + k, stat[-1]
        while n > 0:
            d += ONE
            if not weekend(d):
                n -= 1
        while weekend(d + ONE):
            d += ONE
        return d

    lengths = [(end_for(k) - first).days + 1 for k in range(3)]
    if want:
        k = next((k for k in range(3) if lengths[k] >= want), 2)
    else:
        k = max(k for k in range(3) if lengths[k] <= cap or k == 0)
    return [first, end_for(k)], k


def pattern_year(y):
    f = lunar_dates(y)
    breaks = []  # (name, first, last, [swapped weekday count], makeup pending)
    out = {}

    def add(name, span, swaps=0, fixed=()):
        out[name] = {"off": span, "swaps": swaps, "work": list(fixed)}

    add("New Year's Day", *single(dt.date(y, 1, 1))[:1], fixed=single(dt.date(y, 1, 1))[1])
    span, k = multi(f["eve"], 4, cap=9)
    add("Spring Festival", span, k)
    add("Qingming Festival", single(f["qingming"])[0], fixed=single(f["qingming"])[1])
    span, k = multi(dt.date(y, 5, 1), 2, want=5)
    add("Labor Day", span, k)
    add("Dragon Boat Festival", single(f["dragon"])[0], fixed=single(f["dragon"])[1])
    add("Mid-Autumn Festival", single(f["autumn"])[0], fixed=single(f["autumn"])[1])
    span, k = multi(dt.date(y, 10, 1), 3, cap=8)
    add("National Day", span, k)
    if f["autumn"] == dt.date(y, 10, 1):  # the two share the break
        out["Mid-Autumn Festival"]["off"] = [f["autumn"], f["autumn"]]

    def off(d):
        return any(h["off"][0] <= d <= h["off"][1] for h in out.values())

    for h in out.values():
        for i in range(h["swaps"]):  # the first swap takes the nearest worked weekend day before the break, the second after
            if i == 0:
                d = h["off"][0] - ONE
                while not weekend(d) or off(d) or d in h["work"]:
                    d -= ONE
            else:
                d = h["off"][1] + ONE
                while not weekend(d) or off(d) or d in h["work"]:
                    d += ONE
            h["work"].append(d)
    # whole-year check: days off outside the year are dropped by holidays_json; make-up days must be weekend days
    for name, h in out.items():
        for d in h["work"]:
            assert weekend(d) and not off(d), (y, name, d)
    rows = []
    for name, h in out.items():
        lo, hi = h["off"]
        if hi < dt.date(y, 1, 1) or lo > dt.date(y, 12, 31):
            continue
        work = sorted(d for d in h["work"] if d.year == y)
        rows.append({"name": name, "off": [max(lo, dt.date(y, 1, 1)).isoformat(), min(hi, dt.date(y, 12, 31)).isoformat()],
                     "work": [d.isoformat() for d in work]})
    rows.sort(key=lambda r: r["off"][0])
    return {"source": "pattern", "holidays": rows}


def main():
    data = json.loads(FILE.read_text(encoding="utf-8"))
    for y in YEARS:
        cur = data["years"].get(str(y))
        if cur and cur["source"] == "notice":
            continue
        data["years"][str(y)] = pattern_year(y)
    data["years"] = dict(sorted(data["years"].items()))
    lines = ['{', '  "years": {']
    ys = list(data["years"].items())
    for i, (y, e) in enumerate(ys):
        head = {k: v for k, v in e.items() if k != "holidays"}
        lines.append(f'    "{y}": {{')
        for k, v in head.items():
            lines.append(f'      {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},')
        lines.append('      "holidays": [')
        hs = e["holidays"]
        for j, h in enumerate(hs):
            lines.append(f'        {json.dumps(h, ensure_ascii=False)}{"," if j < len(hs) - 1 else ""}')
        lines.append('      ]')
        lines.append(f'    }}{"," if i < len(ys) - 1 else ""}')
    lines += ['  }', '}', '']
    FILE.write_text("\n".join(lines), encoding="utf-8")
    for y, e in data["years"].items():
        print(y, e["source"])
        for h in e["holidays"]:
            print("  ", h["name"], h["off"], "work", h["work"])


if __name__ == "__main__":
    main()
