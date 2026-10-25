"""National public holidays and weekends per country for the game's calendar (T-862, T-262,
docs/world.md "Calendar").

Writes app/src/game/holidays.json from the `holidays` package (MIT; `pip install holidays`).
The game's year is 365 days with no leap day and its weekdays run on unbroken, so each game year
starts one weekday later than the last. A game year takes the holidays of the real non-leap year
whose January 1 is the same weekday, so holidays tied to a weekday (Thanksgiving, Easter Monday)
land on it, and moveable feasts move between game years as they do between real ones; the cycle
repeats every seven game years.

Output: {"years": {"0": 2029, ...} (January 1's weekday, 0 Monday, to the real year),
         "names": [...], "countries": {"US": {"0": [[day of year, name index], ...], ...}, ...},
         "weeks": {"SA": [0, 0, 0, 0, 2, 1, 0], ...} (each weekday's kind, Monday first: 0 a
            weekday, 1 a Saturday, 2 a Sunday; only where not Saturday and Sunday),
         "workdays": {"HU": {"3": [day of year, ...]}, ...} (weekend days worked in their place),
         "working": {"US": [name index, ...], ...} (holidays most people work through: they keep
            their weekday)}.
A country's days off: its public holidays with the substitute days its law gives (the package's
"observed" days), plus `EXTRA` categories where those are days most people have off. A weekend's
day of rest runs as a Sunday, the other day as a Saturday, a one-day weekend as a Sunday (T-262).
China's days off and make-up working days (T-1157) come from china_holidays.json instead of the
package: the State Council's notice where there is one, else the usual pattern
(china_holidays_pattern.py). English names where the package has them. Run from the maps root:
    venv/Scripts/python.exe trainworld/pipeline/holidays_json.py
"""

import datetime as dt
import json
import pathlib
import re

import holidays

OUT = pathlib.Path(__file__).resolve().parents[1] / "app" / "src" / "game" / "holidays.json"
CHINA = pathlib.Path(__file__).resolve().parent / "china_holidays.json"

# non-leap years by January 1's weekday (0 Monday)
YEARS = {0: 2029, 1: 2030, 2: 2031, 3: 2026, 4: 2027, 5: 2033, 6: 2034}
# categories beyond public that are days off for most people (T-262), and the name shown for them;
# other countries' bank categories are bank closures only
EXTRA = {"JP": ("bank", "New Year holidays"), "KR": ("bank", None)}
# a two-day weekend's day of rest: Friday where the weekend holds it (the Muslim day of prayer),
# else Sunday; Israel's is Saturday (Shabbat)
REST = {"IL": 5}
# US federal holidays most private employers work through (T-262), observed days too; in the US and
# its territories only (Colombia's Columbus Day is a day off)
WORKING_IN = {"US", "AS", "GU", "MP", "PR", "UM", "VI"}
WORKING = re.compile(r"^(Columbus Day|Veterans Day|Martin Luther King,? Jr\.? Day|Washington's Birthday|"
                     r"Juneteenth National Independence Day)( \(observed\))?$")


def week_kinds(code, weekend):
    """Each weekday's kind, Monday first: 0 a weekday, 1 a Saturday, 2 a Sunday."""
    rest = REST.get(code, 4 if 4 in weekend else 6)
    if rest not in weekend:
        rest = weekend[-1]
    return [0 if d not in weekend else 2 if (d == rest or len(weekend) == 1) else 1 for d in range(7)]


def china_table():
    """China's days off and worked weekend days per real year from china_holidays.json (T-1157): the State
    Council's notice where there is one, the usual pattern otherwise. The package has neither the make-up
    working days nor the longer breaks."""
    data = json.loads(CHINA.read_text(encoding="utf-8"))["years"]
    out = {}
    for y in YEARS.values():
        entries, work = {}, []
        for h in data[str(y)]["holidays"]:
            lo, hi = (dt.date.fromisoformat(s) for s in h["off"])
            for i in range((hi - lo).days + 1):
                d = lo + dt.timedelta(days=i)
                if d.year == y:
                    entries[d] = f"{entries[d]}; {h['name']}" if d in entries and h["name"] not in entries[d] else h["name"]
            for s in h["work"]:
                d = dt.date.fromisoformat(s)
                if d.year == y:
                    work.append(d)
        assert all(d.weekday() >= 5 and d not in entries for d in work), y
        out[y] = (entries, sorted(work))
    return out


def english(cls):
    langs = getattr(cls, "supported_languages", ()) or ()
    for lang in ("en_US", "en_GB", "en_IN", "en_CA", "en_NZ", "en_AU"):
        if lang in langs:
            return lang
    return None


def main():
    for wd, y in YEARS.items():
        assert dt.date(y, 1, 1).weekday() == wd and y % 4 != 0, y
    names, index = [], {}
    countries, weeks, workdays = {}, {}, {}
    china = china_table()
    for code in sorted(holidays.list_supported_countries()):
        if len(code) != 2:
            continue
        try:
            cls = getattr(holidays, code)
        except AttributeError:
            continue
        lang = english(cls)
        by_year, worked, added = {}, {}, 0
        for wd, y in YEARS.items():
            kw = {"years": y, "observed": True}
            if lang:
                kw["language"] = lang
            cal = holidays.country_holidays(code, **kw)
            entries = {d: n for d, n in cal.items() if d.year == y}
            plain = holidays.country_holidays(code, **{**kw, "observed": False})
            added += sum(1 for d in entries if d not in plain)
            if code == "CN":
                entries = china[y][0]
            if code in EXTRA:
                cat, label = EXTRA[code]
                for d, n in holidays.country_holidays(code, **{**kw, "categories": (cat,)}).items():
                    if d.year == y and d not in entries:
                        entries[d] = label or n
                        added += 1
            days = {}
            for d, name in sorted(entries.items()):
                doy = (d - dt.date(y, 1, 1)).days
                # one entry a day; two holidays on one date are joined
                days[doy] = f"{days[doy]}; {name}" if doy in days else name
            rows = []
            for doy, name in sorted(days.items()):
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
                rows.append([doy, index[name]])
            by_year[str(wd)] = rows
            ww = sorted((d - dt.date(y, 1, 1)).days for d in getattr(cal, "weekend_workdays", ())
                        if d.year == y and d.weekday() in cal.weekend and d not in entries)
            if code == "CN":
                ww = [(d - dt.date(y, 1, 1)).days for d in china[y][1]]
            if ww:
                worked[str(wd)] = ww
        countries[code] = by_year
        weekend = sorted(holidays.country_holidays(code, years=YEARS[0]).weekend)
        if weekend != [5, 6] or code in REST:
            weeks[code] = week_kinds(code, weekend)
        if worked:
            workdays[code] = worked
        if added or worked or code in weeks:
            print(f"{code}: {added} days off added over the 7 years, {sum(map(len, worked.values()))} weekend workdays, "
                  f"weekend {weekend} {weeks.get(code, '')}")
    work_names = [i for i, n in enumerate(names) if WORKING.match(n)]
    working = {cc: [i for i in work_names if any(r[1] == i for rows in countries[cc].values() for r in rows)]
               for cc in sorted(WORKING_IN & countries.keys())}
    print("working holidays:", {cc: [names[i] for i in v] for cc, v in working.items()})
    OUT.write_text(json.dumps({"years": {str(k): v for k, v in YEARS.items()}, "names": names, "countries": countries,
                               "weeks": weeks, "workdays": workdays, "working": working},
                              ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(countries)} countries, {len(names)} names, {len(weeks)} other weeks, "
          f"{len(workdays)} with weekend workdays, {OUT.stat().st_size} bytes -> {OUT}")


if __name__ == "__main__":
    main()
