# -*- coding: utf-8 -*-
"""Pull hourly boardings/alightings per station from the Seoul Open Data API.

Dataset: OA-12252 서울시 지하철 호선별 역별 시간대별 승하차 인원 정보
Service: CardSubwayTime  (monthly totals, split by hour, boarding and alighting)

Needs a free key from https://data.seoul.go.kr/together/mypage/actkeyMain.do
put in secrets.env as:

    SEOUL_API_KEY=your_key_here

Without a key it falls back to the public `sample` key, which returns only the
first 5 rows -- enough to check the plumbing works, useless for real output.

Writes data/hourly.csv in long form:

    month,line,station,hour,boardings,alightings
"""

import csv
import io
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "hourly.csv")

# Months to pull. The API holds a long back-run; a full year lets us pick a
# reference month later and check it against its neighbours for anomalies.
MONTHS = ["%d%02d" % (y, m) for y in (2024,) for m in range(1, 13)]

PAGE = 1000  # API max rows per request


def load_key():
    path = os.path.join(HERE, "secrets.env")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("SEOUL_API_KEY="):
                    return line.strip().split("=", 1)[1].strip()
    sys.stderr.write(
        "no SEOUL_API_KEY in secrets.env -- falling back to the `sample` key,\n"
        "which returns 5 rows per month. Plumbing check only.\n\n"
    )
    return "sample"


def fetch(key, month, start, end):
    url = "http://openapi.seoul.go.kr:8088/%s/json/CardSubwayTime/%d/%d/%s" % (
        key,
        start,
        end,
        month,
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        body = json.loads(r.read().decode("utf-8"))
    if "CardSubwayTime" not in body:
        raise RuntimeError("%s: %s" % (month, json.dumps(body, ensure_ascii=False)))
    block = body["CardSubwayTime"]
    code = block["RESULT"]["CODE"]
    if code != "INFO-000":
        raise RuntimeError("%s: %s %s" % (month, code, block["RESULT"]["MESSAGE"]))
    return block["list_total_count"], block["row"]


# The API orders its hour columns 4..23 then 0..3, matching the service day.
HOURS = list(range(4, 24)) + list(range(0, 4))


def main():
    key = load_key()
    limit = 5 if key == "sample" else PAGE

    rows_out = []
    for month in MONTHS:
        got = []
        start = 1
        while True:
            end = start + limit - 1
            total, rows = fetch(key, month, start, end)
            got.extend(rows)
            if key == "sample" or len(got) >= total or not rows:
                break
            start = end + 1
            time.sleep(0.2)

        for r in got:
            for h in HOURS:
                rows_out.append(
                    (
                        r["USE_MM"],
                        r["SBWY_ROUT_LN_NM"],
                        r["STTN"],
                        h,
                        int(r.get("HR_%d_GET_ON_NOPE" % h) or 0),
                        int(r.get("HR_%d_GET_OFF_NOPE" % h) or 0),
                    )
                )
        print("%s  %4d stations  %6d rows" % (month, len(got), len(got) * 24))
        sys.stdout.flush()

    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "line", "station", "hour", "boardings", "alightings"])
        w.writerows(rows_out)
    print("\nwrote %s  (%d rows)" % (OUT, len(rows_out)))


if __name__ == "__main__":
    main()
