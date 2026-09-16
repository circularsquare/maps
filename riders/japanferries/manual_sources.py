"""Route figures found outside the regional agents' sweeps -> sources/manual.csv.

- Nagasaki city's monthly 乗降船客数 by route at Nagasaki port
  (raw/dl/nagasaki_city_port_monthly.xlsx, sheets R3-R7 = 2021-2025), summed to
  calendar years, and the older 長崎市統計年鑑 table 38 (2013-2017). Boardings +
  landings at Nagasaki, so passengers between intermediate ports (奈良尾-福江) are
  not in it. Children count 0.5.
- Akashi-Iwaya, FY2012-FY2021, transcribed from the Kobe transport bureau report
  (raw/dl/kobe_akashi_iwaya_report_r5.pdf, p.30, figure 2-23: 旅客総数, operator data).
"""
import csv
import os
import sys

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "raw", "dl")
OUT = os.path.join(HERE, "sources", "manual.csv")
FIELDS = ["prefecture", "operator", "route", "ports", "vessel", "fiscal_year", "year_basis",
          "annual_passengers", "passenger_km", "route_km", "needs_download", "source_title",
          "source_url", "source_page", "notes"]

NAGASAKI_PORTS = {
    "長崎～五島": "長崎 - 奈良尾 - 福江",
    "長崎～天草": "茂木 - 富岡",
    "長崎～伊王島～高島": "長崎 - 伊王島 - 高島",
    "長崎～上五島": "長崎 - 上五島",
    "長崎～香焼": "長崎 - 香焼",
}
NAGASAKI_NOTE = ("Boardings + landings at Nagasaki port only (legs between other ports excluded); "
                 "children counted as 0.5. Operators named for the whole table: 九州商船, 苓北観光汽船, "
                 "野母商船, 五島産業汽船 (and 長崎汽船 in the yearbook) — not split by route.")

AKASHI = {2015: 734299, 2016: 712471, 2017: 736754, 2018: 744411, 2019: 736766, 2020: 546874, 2021: 611871}


def num(v):
    return float(v) if isinstance(v, (int, float)) else 0.0


def nagasaki_monthly():
    wb = openpyxl.load_workbook(os.path.join(DL, "nagasaki_city_port_monthly.xlsx"), read_only=True, data_only=True)
    rows = []
    for sheet, year in (("R3", 2021), ("R4", 2022), ("R5", 2023), ("R6", 2024), ("R7", 2025)):
        table = [list(r) for r in wb[sheet].iter_rows(values_only=True)]
        header = table[2]
        routes = {i: str(v).strip() for i, v in enumerate(header) if isinstance(v, str) and "～" in v}
        months = [r for r in table[4:] if isinstance(r[1], (int, float)) and 1 <= r[1] <= 12]
        assert len(months) == 12, (sheet, len(months))
        total_check = sum(num(r[2]) + num(r[3]) for r in months)
        route_sum = 0.0
        for i, name in routes.items():
            pax = sum(num(r[i]) + num(r[i + 1]) for r in months)
            route_sum += pax
            rows.append((name, year, round(pax)))
        print(f"Nagasaki {year}: routes sum {route_sum:,.0f} vs 総数 {total_check:,.0f}")
    return rows, "長崎市 運輸通信統計「乗降船客数」(月次)", "https://www.city.nagasaki.lg.jp/uploaded/attachment/66737.xlsx", "sheets R3-R7, 12 months summed"


def nagasaki_yearbook():
    wb = openpyxl.load_workbook(os.path.join(DL, "nagasaki_city_yearbook_h30_transport.xlsx"), read_only=True, data_only=True)
    table = [list(r) for r in wb["乗降船客数"].iter_rows(values_only=True)]
    header = next(r for r in table if any(isinstance(v, str) and "～" in v for v in r))
    routes = {i: str(v).strip() for i, v in enumerate(header) if isinstance(v, str) and "～" in v}
    rows = []
    for r in table:
        label = str(r[0] or "").strip()
        if not label.endswith("年"):
            continue
        n = int(label.replace("平成", "").replace("年", "").translate(str.maketrans("０１２３４５６７８９", "0123456789")))
        year = 1988 + n
        check = num(r[1]) + num(r[2])
        route_sum = 0.0
        for i, name in routes.items():
            pax = num(r[i]) + num(r[i + 1])
            route_sum += pax
            rows.append((name, year, round(pax)))
        print(f"Nagasaki yearbook {year}: routes sum {route_sum:,.0f} vs 総数 {check:,.0f}")
    return rows, "平成30年版長崎市統計年鑑 38 乗降船客数", "https://www.city.nagasaki.lg.jp/uploaded/attachment/14666.xlsx", "sheet 乗降船客数"


def tokyo_bay_ferry():
    """Kanaya (Chiba port yearbook 2024, table 5) and Kurihama (Yokosuka city yearbook table 79)."""
    rows = []
    wb = openpyxl.load_workbook(os.path.join(DL, "chiba_port_yearbook_r6_t4-5.xlsx"), read_only=True, data_only=True)
    table = [list(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    start = next(i for i, r in enumerate(table) if any(isinstance(v, str) and "月別乗降人員" in v for v in r))
    total = next(r for r in table[start:]
                 if any(isinstance(v, str) and v.replace("　", "").replace(" ", "") == "合計" for v in r))
    kanaya = next(v for v in total if isinstance(v, (int, float)))
    months = sum(next(v for v in r if isinstance(v, (int, float)))
                 for r in table[start:] if any(isinstance(v, str) and v.strip().endswith("月") for v in r))
    print(f"Kanaya 2024: 合計 {kanaya:,} vs months {months:,}")
    rows.append({
        "prefecture": "千葉県", "operator": "東京湾フェリー", "route": "久里浜～金谷", "ports": "久里浜 - 金谷",
        "vessel": "ferry", "fiscal_year": 2024, "year_basis": "CY", "annual_passengers": int(kanaya),
        "source_title": "千葉県港湾統計年報 令和6年 地方港湾 5 フェリー月別乗降人員数（浜金谷－久里浜）",
        "source_url": "https://www.pref.chiba.lg.jp/kouwan/toukeidata/nenpou/r6/documents/r6nenpo-t4-5.xlsx",
        "source_page": "Sheet1 table 5 合計",
        "notes": "Boardings + landings at 浜金谷, the ferry's only port on the Chiba side; equals 港湾統計 2024 浜金谷.",
    })

    wb = openpyxl.load_workbook(os.path.join(DL, "yokosuka_city_yearbook_r5_ch10.xlsx"), read_only=True, data_only=True)
    ws = next(w for w in wb.worksheets if "定期航路" in w.title)
    for r in ws.iter_rows(values_only=True):
        label = next((v for v in r if isinstance(v, str) and "年）" in v), None)
        nums = [v for v in r if isinstance(v, (int, float))]
        if not label or len(nums) != 4:
            continue
        year = int(label.split("（")[1][:4])
        rows.append({
            "prefecture": "神奈川県", "operator": "", "route": "久里浜港 定期航路（計）", "ports": "久里浜 - 金谷",
            "vessel": "ferry", "fiscal_year": year, "year_basis": "CY", "annual_passengers": int(nums[0] + nums[2]),
            "source_title": "横須賀市統計書 令和5年度版 79 定期航路乗降人員・車輌台数",
            "source_url": "https://www.city.yokosuka.kanagawa.jp/0830/data/t-k-syo/documents/n05010.xlsx",
            "source_page": "sheet 79",
            "notes": "乗込 + 上陸 人員 at 久里浜港 for all scheduled routes, not split; mostly 東京湾フェリー, "
                     "but any 東海汽船 sailings from 久里浜 would be inside it. Year basis assumed calendar (年次).",
        })
    return rows


def main():
    out = []
    for r in tokyo_bay_ferry():
        out.append({k: r.get(k, "") for k in FIELDS})
    for rows, title, url, page in (nagasaki_yearbook(), nagasaki_monthly()):
        for name, year, pax in rows:
            out.append({
                "prefecture": "長崎県", "operator": "", "route": name, "ports": NAGASAKI_PORTS.get(name, ""),
                "vessel": "mixed", "fiscal_year": year, "year_basis": "CY", "annual_passengers": pax,
                "passenger_km": "", "route_km": "", "needs_download": "", "source_title": title,
                "source_url": url, "source_page": page, "notes": NAGASAKI_NOTE,
            })
    for year, pax in AKASHI.items():
        out.append({
            "prefecture": "兵庫県", "operator": "淡路ジェノバライン", "route": "明石～岩屋航路", "ports": "明石 - 岩屋",
            "vessel": "passenger", "fiscal_year": year, "year_basis": "FY", "annual_passengers": pax,
            "passenger_km": "", "route_km": "", "needs_download": "",
            "source_title": "アフターコロナを見据えた明石～岩屋航路の新たな活性化策を探る調査業務 報告書（令和5年3月 国土交通省神戸運輸監理部）",
            "source_url": "https://wwwtb.mlit.go.jp/kobe/content/000292201.pdf", "source_page": "p.30 図2-23",
            "notes": "旅客総数 from 淡路ジェノバライン data; bicycles and motorbikes counted separately and not included.",
        })
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT}: {len(out)} rows")


if __name__ == "__main__":
    main()
