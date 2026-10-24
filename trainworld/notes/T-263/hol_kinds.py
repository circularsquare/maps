"""Try the proposed holiday kinds on real calendars (read-only): days off = the country's weekend days
and public holidays (less a few working holidays); a single working day between two days off is a
bridge; a run of 3+ days off holding a holiday, or holding a homecoming holiday at all, is a long
holiday, with a going-out day before it and a coming-back day at its end."""
import datetime as dt
import re
import holidays

HOMECOMING = re.compile(r"lunar new year|spring festival|chinese new year|korean new year|chuseok|tet\b|"
                        r"eid al-fitr|eid al-adha|ramadan feast|sacrifice feast|diwali|deepavali|"
                        r"^thanksgiving|christmas day|^christmas$|nowruz|songkran|hari raya", re.I)
WORKING = re.compile(r"^columbus day|^veterans day|^martin luther king|^washington's birthday|^juneteenth", re.I)
EN = ("en_US", "en_GB", "en_IN")


def cal(code, year):
    cls = getattr(holidays, code)
    lang = next((l for l in EN if l in (getattr(cls, "supported_languages", ()) or ())), None)
    kw = {"years": [year - 1, year, year + 1], "observed": False}
    if lang:
        kw["language"] = lang
    return holidays.country_holidays(code, **kw), cls.weekend


def runs(code, year):
    h, weekend = cal(code, year)
    d0, d1 = dt.date(year, 1, 1), dt.date(year, 12, 31)
    days = [d0 + dt.timedelta(i) for i in range(-10, 376)]

    def off(d):
        n = h.get(d)
        return d.weekday() in weekend or (n is not None and not WORKING.search(n))

    flags = [off(d) for d in days]
    bridge = [False] * len(days)
    for i in range(1, len(days) - 1):
        if not flags[i] and flags[i - 1] and flags[i + 1] and (h.get(days[i - 1]) or h.get(days[i + 1])):
            bridge[i] = True
    out, i = [], 0
    while i < len(days):
        if not (flags[i] or bridge[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(days) and (flags[j + 1] or bridge[j + 1]):
            j += 1
        names = sorted({h.get(days[k]) for k in range(i, j + 1) if h.get(days[k])})
        n = j - i + 1
        home = any(HOMECOMING.search(x) for x in names)
        if names and (n >= 3 or home) and d0 <= days[i] <= d1:
            out.append((days[i], n, home, "; ".join(names)[:70]))
        i = j + 1
    return out


for code in ["CN", "JP", "KR", "US", "GB", "DE", "FR", "IN", "ID", "SA", "TR", "BR", "VN", "NG"]:
    r = runs(code, 2027)
    print(f"{code}: {len(r)} long holidays, {sum(1 for x in r if x[2])} homecoming")
    for d, n, home, names in r:
        print(f"   {d:%b %d} {n}d {'home ' if home else '     '}{names}")
