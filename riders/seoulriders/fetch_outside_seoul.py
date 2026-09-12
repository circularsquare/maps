# -*- coding: utf-8 -*-
"""Ridership for the operators Seoul's card file does not settle. All anonymous.

  data/kric_station_monthly.csv   KRIC 철도통계, 13 operators, station x month
  data/incheon_hourly.csv         인천교통공사, station x hour x month
  data/incheon_daily.csv          data.go.kr 15004329, station x day

`card_daily_*.csv` covers 27 lines but only the ones whose fares Seoul settles,
so 인천1/2, 7호선's Incheon end, 신분당, 김포골드, 의정부, 에버라인, 우이신설,
신림 and 진접 have no daily total and no hourly profile -- 103 complexes that
float on whatever their Seoul-bound trips imply. These three files close that.

**KRIC 철도통계 is `www.kric.go.kr`, not the `data.kric.go.kr` レール portal that
`fetch_schedules.py` uses.** Different site, different data: 철도운영현황 >
도시철도여객수송 > 역별 승강차실적(월), per station, 승차 and 하차 separately,
2022 onwards, for every 도시철도 operator in the country.

It agrees with the card file. Joining KRIC's 서울교통공사 November 2023 against
`card_daily_202311.csv` gives a median ratio of 1.0042 over 187 stations, 147 of
them inside 2%. It is the same measurement, so it drops into the same
re-levelling with no scaling between the two.

**Two traps in the KRIC pull.** The HTML view pages at 15 rows and has no page
parameter worth finding -- always take the Excel export, which is the same form
POST plus `mode=excel` and returns the whole table. And KRIC splits a transfer
complex per line with a bracketed suffix (고속터미널, 고속터미널(7),
고속터미널(9)); sum them or the station comes out short by however many lines
you missed.

**The Incheon hourly file is on the 사전정보 공표목록 board, not the open-data
portal.** 운수기획팀 has posted 「역 시간대별 통행량」 every month since 2015-10:
station x hour x 승차/하차, one .xlsx per month, three sheets for 인천1/2/7.
Its November 2023 total is 12,663,474 boardings, which is KRIC's figure for that
month to the person. `msg_seq` is not derivable, so the board is scanned for it.

It is a *monthly* aggregate, weekdays and weekends mixed, where 서울교통공사's
OA-12921 is per day. `build_od.py` uses only the shape of an hourly profile, but
the shape of a month is still flatter than the shape of a weekday -- see
"Incheon's hours are a month, not a day" in README.md for the correction.

**The daily file is only ever the current 12 months.** data.go.kr rotates it
annually with no archive, so it cannot reach 2023 and is here for day-type
factors -- Incheon's own weekday/Saturday/Sunday split -- rather than for levels.

    python fetch_outside_seoul.py            # everything, skipping what exists
    python fetch_outside_seoul.py --years 2023 2024
    python fetch_outside_seoul.py --force
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36")

KRIC_OUT = os.path.join(D, "kric_station_monthly.csv")
ICT_HOURLY_OUT = os.path.join(D, "incheon_hourly.csv")
ICT_DAILY_OUT = os.path.join(D, "incheon_daily.csv")

RAW = os.path.join(D, "raw_outside")       # the untouched downloads

# --------------------------------------------------------------------------
# KRIC 철도통계
# --------------------------------------------------------------------------

KRIC_URL = "https://www.kric.go.kr/jsp/industry/rss/citystapassList.jsp"

# Every operator on the 수도권 network. 서울교통공사 is pulled too -- not because
# the card file misses it, but because it is the only overlap the two sources
# have and so the only way to check that they agree.
KRIC_OPS = [
    ("A010010011", u"서울교통공사"),
    ("A010010027", u"인천교통공사"),
    ("A010010029", u"공항철도"),
    ("A010010030", u"서울메트로9"),
    ("A010010042", u"용인경량전철"),
    ("A010010043", u"의정부경전철"),
    ("A010010045", u"네오트랜스(주)"),
    ("A010010046", u"경기철도"),
    ("A010010047", u"김포골드라인"),
    ("A010010048", u"남서울경전철"),
    ("A010010050", u"우이신설도시철도"),
    ("A010010053", u"새서울철도"),
    ("A010010054", u"남양주도시공사"),
]


def kric_raw(year, code):
    return os.path.join(RAW, "kric_%s_%s.xls" % (year, code))


def fetch_kric(s, years, force):
    """One POST per operator per year -> the same .xls the site's button gives."""
    for year in years:
        for code, name in KRIC_OPS:
            path = kric_raw(year, code)
            if os.path.exists(path) and not force:
                continue
            r = s.post(KRIC_URL,
                       data={"q_menuId": "", "fdate": "", "q_org_cd": code,
                             "q_fdate": str(year), "mode": "excel"},
                       headers={"Referer": KRIC_URL}, timeout=120)
            r.raise_for_status()
            # A year an operator did not run returns the ErrorJsp redirect
            # stub rather than a file; that is not an error, just an absence.
            if not r.content.startswith(b"\xd0\xcf"):
                print("   %s %s: no file (%d bytes)" % (year, name, len(r.content)))
                continue
            with open(path, "wb") as f:
                f.write(r.content)
            print("   %s %-10s %6d bytes" % (year, name, len(r.content)))
            time.sleep(0.4)


def parse_kric(years):
    """Tidy every downloaded .xls into one long CSV.

    Sheet layout: two header rows, a 합 계 row, then one row per station with
    columns [연도, 기관, 역명, 호선, 합계승차, 합계하차, 1월승차, 1월하차, ...].
    """
    rows = []
    for year in years:
        for code, name in KRIC_OPS:
            path = kric_raw(year, code)
            if not os.path.exists(path):
                continue
            df = pd.read_excel(path, header=None)
            df = df[df[1] == name]
            for _, r in df.iterrows():
                station = str(r[2]).strip()
                line = str(r[3]).strip()
                if not station or station == "nan":
                    continue
                for m in range(1, 13):
                    b, a = r[4 + 2 * m], r[5 + 2 * m]
                    if pd.isna(b) and pd.isna(a):
                        continue
                    rows.append((name, line, station, int(year), m,
                                 int(b or 0), int(a or 0)))
    with io.open(KRIC_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operator", "line", "station", "year", "month",
                    "boardings", "alightings"])
        w.writerows(rows)
    print("   wrote %s  %s rows"
          % (os.path.basename(KRIC_OUT), format(len(rows), ",")))


# --------------------------------------------------------------------------
# 인천교통공사 역 시간대별 통행량
# --------------------------------------------------------------------------

ICT_LIST = ("https://www.ictr.or.kr/main/bbs/bbsMsgList.do"
            "?cate1=918&bcd=opendata&pgno=%d")
ICT_DOWN = ("https://www.ictr.or.kr/main/bbs/bbsMsgFileDown.do"
            "?bcd=opendata&msg_seq=%s&fileno=1")

# "2023년 11월 역 시간대별 통행량", with 역별 in some older months.
ICT_TITLE = re.compile(u"(20\\d\\d)\\s*년\\s*(\\d{1,2})\\s*월\\s*역별?\\s*시간대별\\s*통행량")


def ict_index(s, pages=15):
    """month (YYYYMM) -> msg_seq, by scanning the pre-disclosure board."""
    found = {}
    for pg in range(1, pages + 1):
        r = s.get(ICT_LIST % pg, timeout=60)
        r.raise_for_status()
        hits = re.findall(r'msg_seq=(\d+)[^>]*>\s*([^<]+)', r.text)
        if not hits:
            break
        for seq, title in hits:
            m = ICT_TITLE.search(re.sub(r"\s+", " ", title))
            if m:
                found.setdefault("%04d%02d" % (int(m.group(1)), int(m.group(2))),
                                 seq)
        time.sleep(0.25)
    return found


def ict_raw(month):
    return os.path.join(RAW, "incheon_hourly_%s.xlsx" % month)


def fetch_incheon_hourly(s, years, force):
    print("   scanning the 사전정보 공표목록 for msg_seq ...")
    index = ict_index(s)
    print("   %d monthly postings found (%s .. %s)"
          % (len(index), min(index), max(index)))
    want = [m for m in sorted(index) if int(m[:4]) in years]
    for month in want:
        path = ict_raw(month)
        if os.path.exists(path) and not force:
            continue
        r = s.get(ICT_DOWN % index[month],
                  headers={"Referer": ICT_LIST % 1}, timeout=120)
        r.raise_for_status()
        if not r.content.startswith(b"PK"):
            print("   %s: not an xlsx (%d bytes)" % (month, len(r.content)))
            continue
        with open(path, "wb") as f:
            f.write(r.content)
        print("   %s  %6d bytes" % (month, len(r.content)))
        time.sleep(0.3)
    return want


def parse_incheon_hourly(months):
    """Tidy the monthly workbooks into one long CSV.

    Each sheet is one line. Row 3/4 carry the header, the station name sits on
    the 승차 row only with the 하차 row blank beneath it, and the last rows are
    line totals -- named `1호선 계` on one sheet and `공사계` on another, so
    anything ending in 계 goes. Hour labels are 05시 .. 23시 then 24시이후,
    which is exactly build_od.py's HOURS = range(5, 25).
    """
    rows = []
    for month in months:
        path = ict_raw(month)
        if not os.path.exists(path):
            continue
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            line = sheet.strip()                # '1호선 ' has a trailing space
            df = xl.parse(sheet, header=None)
            hdr = df.iloc[4].tolist()
            hours = {}
            for col, label in enumerate(hdr):
                label = str(label).strip()
                m = re.match(u"^(\\d{1,2})시$", label)
                if m:
                    hours[col] = int(m.group(1))
                elif label == u"24시이후":
                    hours[col] = 24
            if not hours:
                continue
            station = None
            for _, r in df.iloc[5:].iterrows():
                name = str(r[0]).strip()
                if name and name != "nan":
                    station = name
                kind = str(r[1]).strip()
                if station is None or kind not in (u"승차", u"하차"):
                    continue
                if station.endswith(u"계"):      # '1호선 계', '공사계'
                    continue
                for col, hour in hours.items():
                    v = r[col]
                    if pd.isna(v):
                        continue
                    rows.append((line, station, month, kind, hour, int(v)))
    with io.open(ICT_HOURLY_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "station", "month", "kind", "hour", "count"])
        w.writerows(rows)
    print("   wrote %s  %s rows"
          % (os.path.basename(ICT_HOURLY_OUT), format(len(rows), ",")))


# --------------------------------------------------------------------------
# 인천교통공사 역별일별 이용인원현황 (data.go.kr 15004329)
# --------------------------------------------------------------------------

ICT_DAILY_PK = 15004329


def fetch_incheon_daily(s, force):
    if os.path.exists(ICT_DAILY_OUT) and not force:
        print("   already have %s, skipping" % os.path.basename(ICT_DAILY_OUT))
        return
    page_url = "https://www.data.go.kr/data/%d/fileData.do" % ICT_DAILY_PK
    page = s.get(page_url, timeout=60)
    page.raise_for_status()
    m = re.search(
        r"fn_fileDataDown\('(\d+)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'",
        page.text)
    if not m:
        raise SystemExit(
            "incheon daily: no download call on %s -- the portal's markup has "
            "changed. Download it by hand into %s." % (page_url, ICT_DAILY_OUT))
    pk_s, detail_pk, atch, sn = m.groups()
    r = s.post("https://www.data.go.kr/tcs/dss/selectFileDataDownload.do",
               data={"publicDataPk": pk_s, "publicDataDetailPk": detail_pk,
                     "atchFileId": atch, "fileDetailSn": sn,
                     "publicDataTyCode": "PR0051"},
               headers={"Referer": page_url,
                        "X-Requested-With": "XMLHttpRequest"},
               timeout=60)
    try:
        j = json.loads(r.text)
    except ValueError:
        raise SystemExit("incheon daily: non-JSON reply: %s" % r.text[:200])
    if not j.get("status"):
        # The portal shows a captcha once a download quota is hit.
        raise SystemExit(
            "incheon daily: refused (%s).\nOpen %s and download it by hand "
            "into %s." % (str(j)[:160], page_url, ICT_DAILY_OUT))
    d = s.get("https://www.data.go.kr/cmm/cmm/fileDownload.do",
              params={"atchFileId": j["atchFileId"],
                      "fileDetailSn": j["fileDetailSn"]},
              headers={"Referer": page_url}, timeout=300)
    d.raise_for_status()
    with open(ICT_DAILY_OUT, "wb") as f:
        f.write(d.content)
    df = pd.read_csv(ICT_DAILY_OUT, encoding="cp949")
    print("   wrote %s  %s rows, %s .. %s"
          % (os.path.basename(ICT_DAILY_OUT), format(len(df), ","),
             df[u"통행일자"].min(), df[u"통행일자"].max()))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA

    print("KRIC 역별 승강차실적(월):")
    fetch_kric(s, args.years, args.force)
    parse_kric(args.years)

    print("\n인천교통공사 역 시간대별 통행량:")
    months = fetch_incheon_hourly(s, set(args.years), args.force)
    parse_incheon_hourly(months)

    print("\n인천교통공사 역별일별 이용인원현황:")
    fetch_incheon_daily(s, args.force)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
