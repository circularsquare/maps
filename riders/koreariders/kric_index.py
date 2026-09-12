# -*- coding: utf-8 -*-
"""Scrape the 레일포털 (data.kric.go.kr) open-data catalogue into a local index.

The portal has ~1,200 datasets behind a paged, server-rendered list with no
search API worth the name, so the cheapest way to find out what it holds is to
pull the whole catalogue once and grep it locally.

    python kric_index.py            # write data/kric_index.csv
    python kric_index.py 거리 역간   # grep the saved index for these words
"""

import csv
import io
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = os.path.join(D, "kric_index.csv")

LIST = "https://data.kric.go.kr/rips/M_01_01/intro.do?page=%d&lcd=A"
UA = "Mozilla/5.0 (koreariders catalogue scrape)"

ROW = re.compile(
    r'<td>(?P<cat>[^<]*)</td>\s*<td class="tl">.*?'
    r'detail\.do[^"]*?\bid=(?P<id>\d+)[^"]*">\s*'
    r'<strong class="list_title">(?P<title>.*?)</strong>.*?'
    r'<div class="list_desc">(?P<desc>.*?)</div>\s*'
    r'제공기관\s*:\s*(?P<org>[^/]*?)\s*/\s*수정일\s*:\s*(?P<mod>[\d.]*)',
    re.S)
FMT = re.compile(r'alt="([A-Z]+)"')
TAG = re.compile(r"<[^>]+>")


def clean(s):
    return TAG.sub("", s).replace("&amp;", "&").strip()


def scrape():
    rows, page, seen = [], 1, set()
    while True:
        r = requests.get(LIST % page, headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        found = 0
        for m in ROW.finditer(r.text):
            key = m.group("id")
            if key in seen:
                continue
            seen.add(key)
            block = r.text[m.start():m.end() + 400]
            fmts = FMT.findall(block)
            rows.append({
                "id": key,
                "category": clean(m.group("cat")),
                "title": clean(m.group("title")),
                "org": clean(m.group("org")),
                "modified": m.group("mod"),
                "format": "/".join(sorted(set(fmts))),
                "description": clean(m.group("desc")),
            })
            found += 1
        print("   page %-4d %3d new (%d total)" % (page, found, len(rows)))
        if not found:
            break
        page += 1
        time.sleep(0.3)

    with io.open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n   wrote %s (%d datasets)" % (OUT, len(rows)))


def grep(words):
    with io.open(OUT, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    hits = [r for r in rows
            if all(w in (r["title"] + r["description"] + r["category"])
                   for w in words)]
    print("%d of %d datasets match %s\n" % (len(hits), len(rows), words))
    for r in hits:
        print("id=%-5s %-14s %-8s %s" % (r["id"], r["category"][:14],
                                         r["format"], r["title"]))
        print("        %s" % r["description"][:150])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1:
        grep(sys.argv[1:])
    else:
        scrape()
