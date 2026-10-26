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
            their weekday),
         "supers": {"CN": {"0": [[first going-out day of year, days going out, days away, days coming
            back, name index], ...], ...}, ...} (superholiday runs, T-264: the first day may fall in the
            year before and the last in the year after)}.
A country's days off: its public holidays with the substitute days its law gives (the package's
"observed" days), plus `EXTRA` categories where those are days most people have off. A weekend's
day of rest runs as a Sunday, the other day as a Saturday, a one-day weekend as a Sunday (T-262).
China's days off and make-up working days (T-1157) come from china_holidays.json instead of the
package: the State Council's notice where there is one, else the usual pattern
(china_holidays_pattern.py). Japan's Obon (August 13-16) is added as days off. English names where
the package has them. Run from the maps root:
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
# T-264: days off the package lacks, (month, day) ranges with their name
ADDED = {"JP": [((8, 13), (8, 16), "Obon")]}
# T-264: holidays that make the run of days off around them a superholiday, and the run's name: a
# pattern on one holiday's name, the countries it holds in (None: everywhere) and those it does not
SUPER = [
    (r"^spring festival", {"CN"}, None, "Spring Festival"),
    (r"^spring festival", {"AZ", "TM"}, None, "Nowruz"),
    (r"chinese new year|chinese spring festival|lunar new year", {"VN"}, None, "Tet"),
    (r"chinese new year|chinese spring festival|lunar new year", None, {"VN"}, "Lunar New Year"),
    (r"korean new year", None, None, "Seollal"),
    (r"chuseok", None, None, "Chuseok"),
    (r"^thanksgiving( day)?\b", None, None, "Thanksgiving"),
    (r"^((catholic|orthodox) )?christmas( day)?\b", None, None, "Christmas"),
    (r"\b(eid|id)[ -](al|ul)[ -]fitr", None, None, "Eid al-Fitr"),
    (r"\beid[ -](al|ul)[ -]adha", None, None, "Eid al-Adha"),
    (r"diwali|deepavali", None, None, "Diwali"),
    (r"nowruz|novruz|navruz", None, None, "Nowruz"),
    (r"songkran", None, None, "Songkran"),
    (r"^national day\b", {"CN"}, None, "Golden Week"),
    (r"^constitution (memorial )?day\b", {"JP"}, None, "Golden Week"),
    (r"^obon$", {"JP"}, None, "Obon"),
]
SUPER_RE = [(re.compile(p, re.I), only, unless, name) for p, only, unless, name in SUPER]
# a superholiday's run is at least this many days off
MIN_RUN = 3


def super_name(code, name):
    """The superholiday a holiday's name (several joined by "; ") makes its run, or None."""
    for part in name.split("; "):
        for rx, only, unless, label in SUPER_RE:
            if (only is None or code in only) and not (unless and code in unless) and rx.search(part):
                return label
    return None


def runs_of(code, days_off, worked, kinds, y, super_of):
    """Superholiday runs meeting real year `y` (T-264): days off (holidays, weekend days not worked)
    around a holiday `super_of` names, an ordinary working day between two days off counted in, at
    least MIN_RUN days. `days_off`: date -> holiday name over y-1 to y+1; `worked`: weekend days
    worked; `kinds`: each weekday's kind. Each run as (first going-out day, days going out, days
    away, days coming back, name): five days or more go out over the two days before and come back
    over the last two, shorter runs over one."""
    lo, hi = dt.date(y - 1, 11, 1), dt.date(y + 1, 2, 28)
    n = (hi - lo).days + 1
    dates = [lo + dt.timedelta(days=i) for i in range(n)]
    off = [d in days_off or (kinds[d.weekday()] != 0 and d not in worked) for d in dates]
    bridged = [off[i] or (0 < i < n - 1 and off[i - 1] and off[i + 1] and dates[i] not in worked) for i in range(n)]
    out, i = [], 0
    while i < n:
        if not bridged[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and bridged[j + 1]:
            j += 1
        names = [super_of(days_off[d]) for d in dates[i:j + 1] if d in days_off]
        label = next((x for x in names if x), None)
        length = j - i + 1
        if label and length >= MIN_RUN:
            edge = 2 if length >= 5 else 1
            first = dates[i] - dt.timedelta(days=edge)
            start = (first - dt.date(y, 1, 1)).days
            if start + edge + length > 0 and start < 365:
                out.append((start, edge, length - edge, edge, label))
        i = j + 1
    return out


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


def calendar(code, y, lang, china):
    """A country's days off in real year `y` (date -> name), the count of days the package's observed
    days and `EXTRA` add, and its weekend days worked. China's from its table where it has the year."""
    kw = {"years": y, "observed": True}
    if lang:
        kw["language"] = lang
    cal = holidays.country_holidays(code, **kw)
    entries = {d: n for d, n in cal.items() if d.year == y}
    plain = holidays.country_holidays(code, **{**kw, "observed": False})
    added = sum(1 for d in entries if d not in plain)
    ww = {d for d in getattr(cal, "weekend_workdays", ()) if d.year == y and d.weekday() in cal.weekend and d not in entries}
    if code == "CN" and y in china:
        entries, ww = dict(china[y][0]), set(china[y][1])
    if code in EXTRA:
        cat, label = EXTRA[code]
        for d, n in holidays.country_holidays(code, **{**kw, "categories": (cat,)}).items():
            if d.year == y and d not in entries:
                entries[d] = label or n
                added += 1
    for (m0, d0), (m1, d1), name in ADDED.get(code, ()):
        d, end = dt.date(y, m0, d0), dt.date(y, m1, d1)
        while d <= end:
            entries[d] = f"{entries[d]}; {name}" if d in entries else name
            d += dt.timedelta(days=1)
    return entries, added, ww


def main():
    for wd, y in YEARS.items():
        assert dt.date(y, 1, 1).weekday() == wd and y % 4 != 0, y
    names, index = [], {}
    countries, weeks, workdays, supers = {}, {}, {}, {}

    def name_index(name):
        if name not in index:
            index[name] = len(names)
            names.append(name)
        return index[name]

    china = china_table()
    for code in sorted(holidays.list_supported_countries()):
        if len(code) != 2:
            continue
        try:
            cls = getattr(holidays, code)
        except AttributeError:
            continue
        lang = english(cls)
        weekend = sorted(holidays.country_holidays(code, years=YEARS[0]).weekend)
        kinds = week_kinds(code, weekend)
        cals = {}

        def cal_of(y):
            if y not in cals:
                cals[y] = calendar(code, y, lang, china)
            return cals[y]

        by_year, worked, added, runs = {}, {}, 0, {}
        for wd, y in YEARS.items():
            entries, n_added, ww = cal_of(y)
            added += n_added
            # T-264: superholiday runs, the years either side for runs over New Year
            around, work = {}, set()
            for yy in (y - 1, y, y + 1):
                e, _, w = cal_of(yy)
                around.update({d: n for d, n in e.items() if not (code in WORKING_IN and all(WORKING.match(p) for p in n.split("; ")))})
                work |= w
            found = runs_of(code, around, work, kinds, y, lambda n: super_name(code, n))
            if found:
                runs[str(wd)] = [[s, a, b, c, name_index(label)] for s, a, b, c, label in found]
            days = {}
            for d, name in sorted(entries.items()):
                doy = (d - dt.date(y, 1, 1)).days
                # one entry a day; two holidays on one date are joined
                days[doy] = f"{days[doy]}; {name}" if doy in days else name
            by_year[str(wd)] = [[doy, name_index(name)] for doy, name in sorted(days.items())]
            ww = sorted((d - dt.date(y, 1, 1)).days for d in ww)
            if ww:
                worked[str(wd)] = ww
        countries[code] = by_year
        if weekend != [5, 6] or code in REST:
            weeks[code] = kinds
        if worked:
            workdays[code] = worked
        if runs:
            supers[code] = runs
        if added or worked or code in weeks:
            print(f"{code}: {added} days off added over the 7 years, {sum(map(len, worked.values()))} weekend workdays, "
                  f"weekend {weekend} {weeks.get(code, '')}")
    work_names = [i for i, n in enumerate(names) if WORKING.match(n)]
    working = {cc: [i for i in work_names if any(r[1] == i for rows in countries[cc].values() for r in rows)]
               for cc in sorted(WORKING_IN & countries.keys())}
    print("working holidays:", {cc: [names[i] for i in v] for cc, v in working.items()})
    OUT.write_text(json.dumps({"years": {str(k): v for k, v in YEARS.items()}, "names": names, "countries": countries,
                               "weeks": weeks, "workdays": workdays, "working": working, "supers": supers},
                              ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(countries)} countries, {len(names)} names, {len(weeks)} other weeks, "
          f"{len(workdays)} with weekend workdays, {len(supers)} with superholidays, {OUT.stat().st_size} bytes -> {OUT}")
    for code in ("CN", "JP", "KR", "US", "VN", "IN", "TR"):
        for wd, rows in sorted(supers.get(code, {}).items())[:1]:
            y = YEARS[int(wd)]
            print(code, y, [(str(dt.date(y, 1, 1) + dt.timedelta(days=s)), a, b, c, names[i]) for s, a, b, c, i in rows])


if __name__ == "__main__":
    main()
