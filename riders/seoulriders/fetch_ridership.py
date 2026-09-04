# -*- coding: utf-8 -*-
"""Download the two files a weekday build needs. Both are anonymous.

  data/congestion_raw.csv     서울교통공사 지하철혼잡도정보, data.go.kr 15071311
  data/congestion_line9.xlsx  서울시 9호선 혼잡도, 서울 열린데이터광장 OA-22197
  data/card_daily_<YYYYMM>.csv  서울시 지하철 역별 승하차인원, 서울 열린데이터광장
                                OA-12914, one file per month

**혼잡도 is the thing this project never had: a measured answer to the question
the map draws.** It gives, for a typical 평일/토요일/일요일, the average load of
the trains passing each station in each direction in each half hour, as a
percentage of 정원. Lines 1-8, 서울교통공사 territory only, 282 stations. Our
build produces the same quantity by routing riders; `validate.py --congestion`
compares the two, which is the first independent check the pipeline has had on
its *output* rather than its inputs.

**card_daily is what makes a weekday possible at all.** The hourly file we
already hold (`hourly_2023_raw.csv`) covers only 서울교통공사, 282 of 626
stations. This one has no hours, but it has every day and **all 27 lines** --
Korail through-running, 공항철도, 분당, 신림, 우이신설 and the rest. So a weekday's
*level* is measured everywhere even where its *shape* is not, which is exactly
the split `build_od.py` already knows how to exploit.

Neither portal needs an account. Note the two different download dances:

  data.go.kr        scrape fn_fileDataDown off the page, POST for a file
                    handle, GET the file (same as fetch_schedules.py)
  data.seoul.go.kr  POST fileView.do for the archive listing, then POST
                    nio_download.do with infId + seq. **infSeq matters** --
                    it is 3 for OA-12914, and with the wrong value the server
                    returns an HTML alert saying 잘못된 접근입니다 rather than
                    an error status.

    python fetch_ridership.py                  # defaults below
    python fetch_ridership.py 202311 202411    # extra months
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

CONGESTION_PK = 15071311
CONGESTION_OUT = os.path.join(D, "congestion_raw.csv")

# 9호선 is not in the 서울교통공사 file -- it is not their line. 서울시메트로9호선
# publish their own, and in a different shape: an xlsx with one sheet per
# 상하선 x 평일/휴일 x **일반/급행**. That express split is the point. 9호선's
# 급행 is the most crowded service in Seoul and a blended average describes
# neither half of it.
LINE9_INF = "OA-22197"
LINE9_INFSEQ = "1"
LINE9_SEQ = "2"          # "2023년 9호선 역별 시간별 혼잡도 자료.xlsx"
LINE9_OUT = os.path.join(D, "congestion_line9.xlsx")

CARD_INF = "OA-12914"
CARD_INFSEQ = "3"
# 202312 holds the OD date itself and is not optional: the re-levelling is a
# per-station ratio between the reference day and 2023-12-31, so both ends of
# that ratio have to come from the same source. 202311 is the ordinary month
# nearest the OD date, so the weekday level and the measured pair structure
# come from the same season; it has no public holidays. 202411 is a second
# opinion, equally holiday-free.
CARD_MONTHS = ["202312", "202311", "202411"]


def card_out(month):
    return os.path.join(D, "card_daily_%s.csv" % month)


# --------------------------------------------------------------------------

def fetch_congestion(s):
    if os.path.exists(CONGESTION_OUT):
        print("congestion: already have %s, skipping"
              % os.path.basename(CONGESTION_OUT))
        return
    page_url = "https://www.data.go.kr/data/%d/fileData.do" % CONGESTION_PK
    page = s.get(page_url, timeout=60)
    page.raise_for_status()
    m = re.search(
        r"fn_fileDataDown\('(\d+)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'",
        page.text)
    if not m:
        raise SystemExit(
            "congestion: no download call on %s -- the portal's markup has "
            "changed. Download it by hand into %s." % (page_url, CONGESTION_OUT))
    pk_s, detail_pk, atch, sn = m.groups()

    r = s.post("https://www.data.go.kr/tcs/dss/selectFileDataDownload.do",
               data={"publicDataPk": pk_s, "publicDataDetailPk": detail_pk,
                     "atchFileId": atch, "fileDetailSn": sn,
                     "publicDataTyCode": "PR0051"},
               headers={"Referer": page_url, "X-Requested-With": "XMLHttpRequest"},
               timeout=60)
    try:
        j = json.loads(r.text)
    except ValueError:
        raise SystemExit("congestion: non-JSON reply: %s" % r.text[:200])
    if not j.get("status"):
        # The portal shows a captcha once a download quota is hit.
        raise SystemExit(
            "congestion: refused (%s).\nOpen %s and download it by hand into "
            "%s." % (str(j)[:160], page_url, CONGESTION_OUT))

    d = s.get("https://www.data.go.kr/cmm/cmm/fileDownload.do",
              params={"atchFileId": j["atchFileId"],
                      "fileDetailSn": j["fileDetailSn"]},
              headers={"Referer": page_url}, timeout=300)
    d.raise_for_status()
    cd = d.headers.get("Content-Disposition", "")
    m = re.search(r"filename=\"?([^\";]+)", cd)
    orig = urllib.parse.unquote(m.group(1)).strip('"') if m else "?"
    with open(CONGESTION_OUT, "wb") as f:
        f.write(d.content)
    print("congestion: wrote %s  %s bytes   (%s)"
          % (os.path.basename(CONGESTION_OUT), format(len(d.content), ","), orig))


def fetch_line9(s):
    if os.path.exists(LINE9_OUT):
        print("line9: already have %s, skipping" % os.path.basename(LINE9_OUT))
        return
    ref = ("https://data.seoul.go.kr/dataList/%s/F/1/datasetView.do" % LINE9_INF)
    s.get(ref, timeout=60)
    d = s.post("https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do",
               params={"useCache": "false"},
               data={"infId": LINE9_INF, "seq": LINE9_SEQ,
                     "infSeq": LINE9_INFSEQ},
               headers={"Referer": ref}, timeout=600)
    d.raise_for_status()
    if not d.content.startswith(b"PK"):
        raise SystemExit(
            "line9: expected an xlsx, got %d bytes starting %r.\nOpen %s and "
            "save the 역별 시간별 혼잡도 xlsx to %s by hand."
            % (len(d.content), d.content[:40], ref, LINE9_OUT))
    with open(LINE9_OUT, "wb") as f:
        f.write(d.content)
    print("line9: wrote %s  %s bytes"
          % (os.path.basename(LINE9_OUT), format(len(d.content), ",")))


def card_archive(s):
    """name -> seq for every file in the OA-12914 archive."""
    ref = "https://data.seoul.go.kr/dataList/%s/F/1/datasetView.do" % CARD_INF
    r = s.post("https://data.seoul.go.kr/dataList/fileView.do",
               data={"infId": CARD_INF, "srvType": "F", "serviceKind": "1",
                     "currentPageNo": "1"},
               headers={"Referer": ref, "X-Requested-With": "XMLHttpRequest"},
               timeout=60)
    r.raise_for_status()
    out = {}
    for row in re.split(r"</tr>", r.text):
        seq = re.search(r"downloadFile\('(\d+)'\)", row)
        name = re.search(r"([\w.]+\.csv)", row)
        if seq and name:
            out[name.group(1)] = seq.group(1)
    return out


def fetch_card(s, months):
    want = [m for m in months if not os.path.exists(card_out(m))]
    for m in months:
        if m not in want:
            print("card: already have %s, skipping"
                  % os.path.basename(card_out(m)))
    if not want:
        return

    archive = card_archive(s)
    if not archive:
        raise SystemExit(
            "card: the OA-12914 file list came back empty -- the portal's "
            "markup has changed. Download the CARD_SUBWAY_MONTH files by hand.")
    print("card: %d files in the archive, %s .. %s"
          % (len(archive), min(archive), max(archive)))

    ref = "https://data.seoul.go.kr/dataList/%s/F/1/datasetView.do" % CARD_INF
    s.get(ref, timeout=60)
    for month in want:
        name = "CARD_SUBWAY_MONTH_%s.csv" % month
        seq = archive.get(name)
        if seq is None:
            print("card: %s is not in the archive -- skipped" % name)
            continue
        d = s.post("https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do",
                   params={"useCache": "false"},
                   data={"infId": CARD_INF, "seq": seq, "infSeq": CARD_INFSEQ},
                   headers={"Referer": ref}, timeout=600)
        d.raise_for_status()
        if b"<html" in d.content[:200].lower():
            raise SystemExit(
                "card: %s came back as HTML, not a file:\n   %s\nThe portal "
                "rejects nio_download.do when infSeq is wrong for the dataset."
                % (name, d.content[:200].decode("utf-8", "replace")))
        with open(card_out(month), "wb") as f:
            f.write(d.content)
        print("card: wrote %s  %s bytes"
              % (os.path.basename(card_out(month)), format(len(d.content), ",")))


def main():
    if not os.path.isdir(D):
        os.makedirs(D)
    months = sys.argv[1:] or CARD_MONTHS
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    fetch_congestion(s)
    fetch_line9(s)
    fetch_card(s, months)
    print("\ndone. Next: python build_od.py --day weekday")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
