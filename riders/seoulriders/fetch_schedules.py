# -*- coding: utf-8 -*-
"""Download the timetables for the lines beyond 1-9. Both sources are anonymous.

  data/kric_urbanrail_timetable.xlsx   레일포털 (data.kric.go.kr), 국가철도공단
  data/incheon2_*.csv                  data.go.kr, 인천교통공사

레일포털 is the one worth remembering. It is the upstream source for
data.go.kr's 전국도시철도운행정보표준데이터 -- the portal listing only points back
here -- and unlike Seoul's own portal it needs no account and no key. One file
carries complete timetables for 39 lines nationwide.

인천2호선 is the exception. It is in the KRIC file, but with a headway where the
times should be ("운행시격 3분20초~10분"), so its four real timetables come from
인천교통공사 on data.go.kr instead. Those need a two-step download: POST for a
file id, then GET the file.

    python fetch_schedules.py
"""

import io
import json
import os
import re
import sys
import urllib.parse

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36")

KRIC_PAGE = "https://data.kric.go.kr/rips/M_01_01/detail.do?id=900"
KRIC_FILE = ("https://data.kric.go.kr/rips/dataset/download.file"
             "?type=filedata&id=900&operation=1")
KRIC_OUT = os.path.join(D, "kric_urbanrail_timetable.xlsx")

# 인천교통공사 열차운행시각표, one dataset per direction per day type.
# 2023-12-31 was a Sunday, so only the 휴일 pair is used downstream; the
# 평일 pair is here so a weekday build needs no second trip.
INCHEON2 = {
    15051209: "incheon2_holiday_up.csv",     # 운연 -> 검단오류
    15051207: "incheon2_holiday_down.csv",   # 검단오류 -> 운연
    15051210: "incheon2_weekday_up.csv",
    15051208: "incheon2_weekday_down.csv",
}


def fetch_kric(session):
    if os.path.exists(KRIC_OUT):
        print("kric: already have %s (%.1f MB), skipping"
              % (os.path.basename(KRIC_OUT), os.path.getsize(KRIC_OUT) / 1e6))
        return
    print("kric: downloading %s ..." % KRIC_FILE)
    r = session.get(KRIC_FILE, headers={"Referer": KRIC_PAGE}, timeout=300)
    r.raise_for_status()
    if len(r.content) < 1_000_000 or not r.content.startswith(b"PK"):
        raise SystemExit(
            "kric: expected a large xlsx, got %d bytes starting %r.\n"
            "   Open %s in a browser and save the file to %s by hand."
            % (len(r.content), r.content[:40], KRIC_PAGE, KRIC_OUT))
    with open(KRIC_OUT, "wb") as f:
        f.write(r.content)
    print("   wrote %s (%.1f MB)" % (KRIC_OUT, len(r.content) / 1e6))


def fetch_incheon2(session):
    """data.go.kr file downloads: POST for a file handle, then GET the file."""
    for pk, name in INCHEON2.items():
        out = os.path.join(D, name)
        if os.path.exists(out):
            print("incheon2: already have %s, skipping" % name)
            continue
        page_url = "https://www.data.go.kr/data/%d/fileData.do" % pk
        page = session.get(page_url, timeout=60)
        page.raise_for_status()
        m = re.search(
            r"fn_fileDataDown\('(\d+)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'",
            page.text)
        if not m:
            raise SystemExit(
                "incheon2: no download call on %s -- the portal's markup has "
                "changed. Download it by hand into %s." % (page_url, out))
        pk_s, detail_pk, atch, sn = m.groups()

        r = session.post(
            "https://www.data.go.kr/tcs/dss/selectFileDataDownload.do",
            data={"publicDataPk": pk_s, "publicDataDetailPk": detail_pk,
                  "atchFileId": atch, "fileDetailSn": sn,
                  "publicDataTyCode": "PR0051"},
            headers={"Referer": page_url, "X-Requested-With": "XMLHttpRequest"},
            timeout=60)
        try:
            j = json.loads(r.text)
        except ValueError:
            raise SystemExit("incheon2: %d gave non-JSON: %s"
                             % (pk, r.text[:200]))
        if not j.get("status"):
            # The portal shows a captcha once a download quota is hit.
            raise SystemExit(
                "incheon2: %d refused (%s).\nOpen %s and download it by hand "
                "into %s." % (pk, str(j)[:160], page_url, out))

        d = session.get("https://www.data.go.kr/cmm/cmm/fileDownload.do",
                        params={"atchFileId": j["atchFileId"],
                                "fileDetailSn": j["fileDetailSn"]},
                        headers={"Referer": page_url}, timeout=180)
        d.raise_for_status()
        cd = d.headers.get("Content-Disposition", "")
        orig = urllib.parse.unquote(
            re.search(r"filename=\"?([^\";]+)", cd).group(1)) if "filename=" in cd else "?"
        with open(out, "wb") as f:
            f.write(d.content)
        print("   wrote %s  %s bytes   (%s)"
              % (name, format(len(d.content), ","), orig.strip('"')))


def main():
    if not os.path.isdir(D):
        os.makedirs(D)
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    fetch_kric(s)
    fetch_incheon2(s)
    print("\ndone. Next: python kric.py")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
