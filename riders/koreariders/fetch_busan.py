"""Download public city-rail CSVs using data.go.kr's file flow.

No login or key. Stop on refusal; use the printed public page to download by hand.
"""
import json
import re
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent / "data"
DATASETS = {
    "busan": {3057229: "counts.csv", 3033564: "distances.csv", 15043686: "stations.csv"},
    "daejeon": {15060591: "counts.csv"},
    "gwangju": {15060048: "counts.csv"},
    "daegu": {15002503: "counts.csv"},
    "busan_gimhae": {15105181: "counts.csv"},
    "donghae": {15100373: "counts.csv"},
}


def main():
    session = requests.Session()
    for city, datasets in DATASETS.items():
        folder = ROOT / city
        folder.mkdir(parents=True, exist_ok=True)
        for pk, name in datasets.items():
            path = folder / name
            if path.exists():
                print(city, name, "already exists")
                continue
            page_url = f"https://www.data.go.kr/data/{pk}/fileData.do"
            print(page_url, flush=True)
            page = session.get(page_url, timeout=45)
            page.raise_for_status()
            m = re.search(r"fn_fileDataDown\('(\d+)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'", page.text)
            if not m:
                raise RuntimeError(f"No file download found; download {page_url} manually into {path}")
            ident, detail, attachment, serial = m.groups()
            r = session.post("https://www.data.go.kr/tcs/dss/selectFileDataDownload.do",
                             data={"publicDataPk": ident, "publicDataDetailPk": detail,
                                   "atchFileId": attachment, "fileDetailSn": serial, "publicDataTyCode": "PR0051"},
                             headers={"Referer": page_url, "X-Requested-With": "XMLHttpRequest"}, timeout=45)
            r.raise_for_status()
            j = r.json()
            if not j.get("status"):
                raise RuntimeError(f"Download refused; download {page_url} manually into {path}")
            r = session.get("https://www.data.go.kr/cmm/cmm/fileDownload.do",
                            params={"atchFileId": j["atchFileId"], "fileDetailSn": j["fileDetailSn"]},
                            headers={"Referer": page_url}, timeout=60)
            r.raise_for_status()
            if b"<html" in r.content[:500].lower():
                raise RuntimeError(f"Not a CSV: {page_url}")
            path.write_bytes(r.content)
            print(city, name, len(r.content), "bytes", flush=True)
            (folder / (name + ".source.json")).write_text(json.dumps({"url": page_url,
                "content_disposition": r.headers.get("Content-Disposition", "")}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
