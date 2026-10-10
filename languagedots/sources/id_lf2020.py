"""Indonesia: the 2020 census long form's home-language table by regency, for placement only.

    python sources/id_lf2020.py [--fetch]

SOURCE. Sensus Penduduk 2020 Long Form (fieldwork 2022; BPS files it as "sp2022"), table 201,
"Jumlah Penduduk Berumur 5 Tahun ke Atas Menurut Wilayah, Jenis Kelamin, dan Penggunaan Bahasa
Daerah untuk Berkomunikasi Sehari-hari dalam Keluarga": persons 5+, Ya / Tidak (uses a regional
language in the family or not; Tidak = Indonesian or a foreign language, BPS's Profil Suku dan
Keragaman Bahasa Daerah, 2024, p.12), all 514 kabupaten/kota. Open, no key:
    https://sensus.bps.go.id/topik/tabular/sp2022/201/<area>/3
area 1 = Indonesia (province rows), 2-35 = the 34 provinces of 2020 in code order (regency rows);
format 3 = JSON. Sample-weighted estimates. No regional language is named below the nation.

USE. countries/id.py seeds Indonesian inside each 2010 province by the regency's Tidak share, so
the 2010 counts stay as they are and only where Indonesian sits inside a province changes (the
Jakarta border edge: DKI 95-97% Tidak, Kota Bekasi 95%, Depok 92%, against 1-4% in Garut,
Tasikmalaya and Cianjur; West Java's 2010 Indonesian had been spread evenly). sources/id.md §11.

CHECKS: every regency's Ya + Tidak = Total; each province's regencies sum to its row in the
national table (within 3: rounding of weighted estimates); 514 regencies; national total
253,679,348.
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "id" / "lf2020"
OUT = HERE / "data" / "normalized" / "id_lf2020_regency.csv"
URL = "https://sensus.bps.go.id/topik/tabular/sp2022/201/{area}/3"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
AREAS = range(1, 36)
NATIONAL = 253_679_348
N_REGENCIES = 514


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for a in AREAS:
        path = RAW / f"lf_201_w{a:02d}.json"
        if path.exists() and path.stat().st_size > 100:
            continue
        req = urllib.request.Request(URL.format(area=a), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        json.loads(body.decode("utf-8"))           # refuse to save anything but JSON
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        print(f"  area {a}: {len(body):,} bytes")
        time.sleep(1.2)


def load(a):
    path = RAW / f"lf_201_w{a:02d}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run with --fetch")
    return json.loads(path.read_text(encoding="utf-8"))["data"]


def totals(rows, level):
    """{code: {"name", "Ya", "Tidak", "Total"}} for both sexes at one level."""
    out = {}
    for r in rows:
        # level 1 province, 2 regency; the national row has none
        if r["level_wilayah"] != level or r["nama_item__kategori_1"] != "Total":
            continue
        d = out.setdefault(r["kode_wilayah"], {"name": r["nama_wilayah"]})
        d[r["nama_item__kategori_2"]] = int(r["nilai"] or 0)
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    prov = totals(load(1), 1)
    reg = {}
    for a in AREAS[1:]:
        got = totals(load(a), 2)
        if not got:
            raise SystemExit(f"area {a}: no regency rows")
        p = {k[:2] for k in got}
        if len(p) != 1:
            raise SystemExit(f"area {a}: regencies of several provinces {sorted(p)}")
        p = p.pop()
        prow = next((v for k, v in prov.items() if k[:2] == p), None)
        if prow is None:
            raise SystemExit(f"area {a}: province {p} not in the national table")
        for c in ("Ya", "Tidak", "Total"):
            s = sum(v[c] for v in got.values())
            if abs(s - prow[c]) > 3:
                raise SystemExit(f"province {p} {c}: regencies sum {s:,}, national table "
                                 f"{prow[c]:,}")
        reg.update(got)
    for k, v in reg.items():
        if v["Ya"] + v["Tidak"] != v["Total"]:
            raise SystemExit(f"{k} {v['name']}: Ya + Tidak != Total")
    if len(reg) != N_REGENCIES:
        raise SystemExit(f"{len(reg)} regencies, expected {N_REGENCIES}")
    nat = sum(v["Total"] for v in prov.values())
    if nat != NATIONAL:
        raise SystemExit(f"national total {nat:,}, expected {NATIONAL:,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["regency", "name", "ya", "tidak", "total", "tidak_share"])
        for k in sorted(reg):
            v = reg[k]
            w.writerow([k, v["name"], v["Ya"], v["Tidak"], v["Total"],
                        f"{v['Tidak'] / v['Total']:.5f}" if v["Total"] else ""])
    tmp.replace(OUT)
    tid = sum(v["Tidak"] for v in reg.values())
    print(f"  {len(reg)} regencies, {nat:,} people 5+, Tidak {tid:,} ({tid / nat * 100:.1f}%)")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
