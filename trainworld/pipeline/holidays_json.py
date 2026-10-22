"""National public holidays per country for the game's calendar (T-862, docs/world.md "Calendar").

Writes app/src/game/holidays.json from the `holidays` package (MIT; `pip install holidays`).
The game's year is 365 days with no leap day and its weekdays run on unbroken, so each game year
starts one weekday later than the last. A game year takes the holidays of the real non-leap year
whose January 1 is the same weekday, so holidays tied to a weekday (Thanksgiving, Easter Monday)
land on it, and moveable feasts move between game years as they do between real ones; the cycle
repeats every seven game years.

Output: {"years": {"0": 2029, ...} (January 1's weekday, 0 Monday, to the real year),
         "names": [...], "countries": {"US": {"0": [[day of year, name index], ...], ...}, ...}}.
Public holidays only (the package's default category), no "observed" substitute days, English
names where the package has them. Run from the maps root:
    venv/Scripts/python.exe trainworld/pipeline/holidays_json.py
"""

import datetime as dt
import json
import pathlib

import holidays

OUT = pathlib.Path(__file__).resolve().parents[1] / "app" / "src" / "game" / "holidays.json"

# non-leap years by January 1's weekday (0 Monday)
YEARS = {0: 2029, 1: 2030, 2: 2031, 3: 2026, 4: 2027, 5: 2033, 6: 2034}


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
    countries = {}
    for code in sorted(holidays.list_supported_countries()):
        if len(code) != 2:
            continue
        try:
            cls = getattr(holidays, code)
        except AttributeError:
            continue
        lang = english(cls)
        by_year = {}
        for wd, y in YEARS.items():
            kw = {"years": y, "observed": False}
            if lang:
                kw["language"] = lang
            days = {}
            for d, name in sorted(holidays.country_holidays(code, **kw).items()):
                if d.year != y:
                    continue
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
        countries[code] = by_year
    OUT.write_text(json.dumps({"years": {str(k): v for k, v in YEARS.items()}, "names": names, "countries": countries},
                              ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(countries)} countries, {len(names)} names, {OUT.stat().st_size} bytes -> {OUT}")


if __name__ == "__main__":
    main()
