# -*- coding: utf-8 -*-
"""Fetch a 철도통계 table from www.kric.go.kr and write it as CSV.

A second, separate site from the data.kric.go.kr catalogue `kric_index.py`
scrapes.  `www.kric.go.kr/jsp/industry/rss/<page>.jsp` serves live statistical
tables with no login and no key -- per line, per station, each crossed with
train type.  Four things about it that look like failure and are not:

  * a bare fetch gets **403**; send a browser User-Agent.
  * without `q_fdate` the page renders "총 0건" and only the field-description
    table, and the monthly pages want `q_month` as well.  Either missing prints
    "검색된 자료가 없습니다", which reads like an empty dataset and is not one.
  * the table is **paged at 15 rows** with no visible pager.  The parameter is
    `pageNo`, named only in /ext/script/JControl.js.  A national station table
    is 243 rows, so page 1 stops in the middle of 경부선.
  * the Excel button is `mode=excel` on the same URL and it redirects to
    /ErrorJsp.jsp.  Page through instead.

    python kric_stats.py --list                     # every rss page linked
    python kric_stats.py --year raillinepassmonList 2023
    python kric_stats.py --year railstapassmonList 2023
    python kric_stats.py raillinepassdivList 2023    # one un-monthly table

`--year` walks all twelve months and every page and writes
data/kric/<page>_<year>.csv with a `month` column.  The console dies on Korean
under Windows cp1252, so everything is written to a file.
"""

import csv
import io
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "kric")

BASE = "http://www.kric.go.kr/jsp/industry/rss/%s.jsp"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

PAGE = re.compile(r"rss/([A-Za-z_]+)\.jsp")
COUNT = re.compile(u"총\s*([\d,]+)\s*건")
TAG = re.compile(r"<[^>]+>")
SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
NUM = re.compile(r"^-?[\d,]+$")


def fetch(page, year=None, month=None, page_no=None, **extra):
    params = {}
    if year:
        params["q_fdate"] = str(year)
    if month:
        params["q_month"] = str(month)
    if page_no and int(page_no) > 1:
        params["pageNo"] = str(page_no)
    params.update(extra)
    r = requests.get(BASE % page, params=params,
                     headers={"User-Agent": UA}, timeout=90)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def rows_of(html):
    """Every table row on the page, as lists of cell text."""
    out = []
    for tr in re.findall(r"<tr\b.*?</tr>", SCRIPT.sub("", html), re.S | re.I):
        cells = [" ".join(TAG.sub("", c).replace("&nbsp;", " ")
                          .replace("&amp;", "&").split())
                 for c in re.findall(r"<t[hd]\b.*?</t[hd]>", tr, re.S | re.I)]
        if any(cells):
            out.append(cells)
    return out


def parse(html):
    """(column names, data rows).

    These tables carry a field-description table first, then the real one:
    a row of type names, sometimes a row of 승차수/하차수 sub-headers, then
    rows of `name` followed by numbers.  Two traps in the header:

      * its first cell is the *총계* column's label ("수송인원(명)",
        "역명/열차종별"), not a train type, so the types are `header[1:]` and
        the first number in a data row is the total.  Dropping the wrong end
        shifts every type by one and the result still looks plausible.
      * the station pages give each type a 승차수 and a 하차수, so a
        13-label header describes 26 numbers.

    Returns names for every numeric column, total first.
    """
    all_rows = rows_of(html)
    header, sub = None, None
    for i, cells in enumerate(all_rows):
        if any(c.startswith("KTX") for c in cells):
            header = cells
            nxt = all_rows[i + 1] if i + 1 < len(all_rows) else []
            if nxt and all(u"차수" in c for c in nxt):
                sub = nxt
            break
    data = [c for c in all_rows if len(c) > 1 and NUM.match(c[1])]
    if not header or not data:
        return [], data

    # Reconcile against the real row width rather than trusting the labels:
    # the line table's first label *is* the total column's header, the station
    # table's names 합계 outright, and only the station table doubles every
    # type into 승차수/하차수.  Guessing wrong shifts every column by one and
    # the numbers still look plausible.
    labels = header[1:]
    n = len(data[0]) - 1
    pair = [s for s in (sub or [])[:2]] or [u"승차수", u"하차수"]

    if n == len(labels):
        cols = labels
    elif n == len(labels) + 1:
        cols = [u"합계"] + labels
    elif n == 2 * len(labels):
        cols = ["%s %s" % (t, s) for t in labels for s in pair]
    elif n == 2 * (len(labels) + 1):
        cols = ["%s %s" % (t, s) for t in [u"합계"] + labels for s in pair]
    else:
        raise SystemExit("%d numbers under %d labels (%s) -- unknown shape"
                         % (n, len(labels), ", ".join(labels)))
    return cols, data


def n_records(html):
    m = COUNT.search(html)
    return int(m.group(1).replace(",", "")) if m else None


def walk(page, year, month=None, limit=40):
    """Every row of one (year, month), paging until the count is covered."""
    seen, out, total = set(), [], None
    header = None
    for p in range(1, limit + 1):
        html = fetch(page, year, month, p)
        if total is None:
            total = n_records(html)
        head, data = parse(html)
        header = header or head
        fresh = [d for d in data if tuple(d) not in seen]
        if not fresh:
            break
        for d in fresh:
            seen.add(tuple(d))
        out.extend(fresh)
        if total is not None and len(out) >= total:
            break
        time.sleep(0.3)
    return header, out, total


def year_csv(page, year):
    if not os.path.isdir(D):
        os.makedirs(D)
    dest = os.path.join(D, "%s_%s.csv" % (page, year))
    header, wrote = None, 0
    with io.open(dest, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for month in range(1, 13):
            head, rows, total = walk(page, year, month)
            if not rows:
                print("   %s-%02d  no rows" % (year, month))
                continue
            if header is None:
                header = head or []
                w.writerow(["month", "name"] + header)
            for r in rows:
                if len(r) - 1 != len(header):
                    raise SystemExit(
                        "%s %s-%02d: %d numbers under %d column names -- the "
                        "header changed shape, do not trust the CSV"
                        % (page, year, month, len(r) - 1, len(header)))
                w.writerow([month] + r)
                wrote += 1
            print("   %s-%02d  %d rows (of %s)" % (year, month, len(rows), total))
    print("%s   %d rows" % (dest, wrote))
    return dest


def main(argv):
    if not argv or argv[0] == "--list":
        html = fetch("raillinepassdivList", 2023)
        pages = sorted(set(PAGE.findall(html)))
        if not os.path.isdir(D):
            os.makedirs(D)
        io.open(os.path.join(D, "pages.txt"), "w",
                encoding="utf-8").write("\n".join(pages))
        print("%d rss pages" % len(pages))
        for p in pages:
            print("   " + p)
        return

    if argv[0] == "--year":
        year_csv(argv[1], argv[2])
        return

    page = argv[0]
    year = argv[1] if len(argv) > 1 else None
    extra = dict(a.split("=", 1) for a in argv[2:])
    html = fetch(page, year, **extra)
    if not os.path.isdir(D):
        os.makedirs(D)
    stem = os.path.join(D, "%s_%s" % (page, year or "nodate"))
    io.open(stem + ".html", "w", encoding="utf-8").write(html)
    head, data = parse(html)
    with io.open(stem + ".csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if head:
            w.writerow(["name"] + head)
        for r in data:
            w.writerow(r)
    # ASCII only: the console is cp1252 under Windows and dies on Korean.
    print("%s.csv   %d rows, %s records" % (stem, len(data), n_records(html)))


if __name__ == "__main__":
    main(sys.argv[1:])
