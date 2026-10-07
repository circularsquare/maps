"""TCDD's station list from its network statement (Şebeke Bildirimi 2026, Ek-3.3.1.3 İstasyon
Özellikleri, data/raw/tr/sb2026_ek3313_istasyon.pdf) as a CSV, data/raw/tr/tcdd_istasyonlar.csv.

    python tr_stations_pdf.py

Each row is one station, siding or halt of TCDD's network with its region, status (Gar
Müdürlüğü, İstasyon Şefliği, Durak, Sayding...), whether it is open to traffic, its
chainage (Mihver Km), whether it is open to passenger operation ("Yolcu İşletme", + or -)
and its daily boardings plus alightings. tr_register reads the passenger column. The table
is a drawn grid of single cells, so it is read by position: a row starts at the region
number in the first column and runs to the next; each column is found by its rotated header.
"""
import csv
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "data" / "raw" / "tr" / "sb2026_ek3313_istasyon.pdf"
OUT = ROOT / "data" / "raw" / "tr" / "tcdd_istasyonlar.csv"

# header text (start) -> output column
HEADERS = {
    "Bölgesi": "region",
    "İstasyon / Sayding / Durak Adı": "name",
    "Statü": "status",
    "Trafik Hizmetlerine Açık": "traffic",
    "Mihver Km": "km",
    "Yolcu İşletme": "passenger",
    "Günlük Giden + Gelen Yolcu Sayısı": "daily_pax",
}


def main():
    import fitz
    doc = fitz.open(PDF)
    rows = []
    cols = {}                    # the first page's header serves the pages after it
    for page in doc:
        d = page.get_text("dict")
        lines = []
        head_y = 0.0
        for b in d["blocks"]:
            for l in b.get("lines", []):
                txt = " ".join(s["text"] for s in l["spans"]).strip()
                if not txt:
                    continue
                x0, y0, x1, y1 = l["bbox"]
                if abs(l["dir"][0]) < 0.1 and l["dir"][1] < -0.9:
                    head_y = max(head_y, y1)
                    for h, c in HEADERS.items():
                        if txt.startswith(h):
                            cols[c] = (x0 + x1) / 2
                else:
                    lines.append((x0, y0, x1, y1, txt))
        if "region" not in cols or "name" not in cols:
            print(f"  page {page.number}: no header ({sorted(cols)})")
            continue
        starts = sorted(y0 for x0, y0, x1, y1, t in lines
                        if abs((x0 + x1) / 2 - cols["region"]) < 4 and re.fullmatch(r"\d", t)
                        and y0 > head_y)
        for i, ys in enumerate(starts):
            ye = starts[i + 1] if i + 1 < len(starts) else 1e9
            cell = {}
            for c, cx in cols.items():
                parts = [t for x0, y0, x1, y1, t in sorted(lines, key=lambda r: (r[1], r[0]))
                         if ys - 1 <= y0 < ye - 1 and x0 - 2 <= cx <= x1 + 2]
                cell[c] = " ".join(parts).strip()
            if cell.get("name"):
                rows.append(cell)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(HEADERS.values()))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADERS.values()})
    yes = sum(1 for r in rows if r.get("passenger") == "(+)")
    print(f"{len(rows)} rows, {yes} open to passengers -> {OUT}")


if __name__ == "__main__":
    main()
