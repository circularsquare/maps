"""Download the raw files for Tajikistan into helper1m/data/tajikistan/raw/.

stat.tj is WordPress; every file below was found through its media API
(https://stat.tj/wp-json/wp/v2/media?per_page=100&page=N, 3,569 files on
2026-10-02). Skips files already on disk. Roughly 30 MB in all.
"""
import gzip
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "tajikistan" / "raw"
UA = "helper1m-research/1.0"
FILES = {
    # annual bulletin "Шумораи аҳолии Ҷумҳурии Тоҷикистон то 1 январи соли 20XX"
    "bulletin_2025.pdf": "https://www.stat.tj/wp-content/uploads/2025/12/machmuai-shumorai-aholi-to-1.01.2025.pdf",
    "bulletin_2024.pdf": "https://www.stat.tj/wp-content/uploads/2024/09/machmuai-shumorai-aholi-to-1.01.2024.pdf",
    "bulletin_2022_corrected.pdf": "https://www.stat.tj/wp-content/uploads/2024/08/machmuai-shumorai-aholi-to-1.01.2022-ispravlenij.pdf",
    # census 2020, volume I, table 1 (Russian)
    "census2020_vol1_table1_ru.pdf": "https://www.stat.tj/wp-content/uploads/2024/05/tablicza-1.-chislennost-postoyannogo-naseleniya-po-oblastyam-rajonam-gorodskim-poseleniyam-rajonnym-czentram-i-selskim-naselennym-punktam-s-chis.pdf",
    # OSM extract (boundaries), Geofabrik snapshot of 2026-10-01
    "tajikistan-261001.osm.pbf": "https://download.geofabrik.de/asia/tajikistan-261001.osm.pbf",
    # Kontur population 400 m hexagons, used only as a check (check_kontur.py)
    "kontur_population_TJ_20231101.gpkg.gz": "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/kontur_population_TJ_20231101.gpkg.gz",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        out = RAW / name
        if out.exists():
            print("have", name)
            continue
        req = urllib.request.Request(urllib.parse.quote(url, safe=":/?=&%"),
                                     headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f)
        print(name, out.stat().st_size)
    gz = RAW / "kontur_population_TJ_20231101.gpkg.gz"
    gpkg = gz.with_suffix("")
    if gz.exists() and not gpkg.exists():
        with gzip.open(gz) as src, open(gpkg, "wb") as dst:
            shutil.copyfileobj(src, dst)


if __name__ == "__main__":
    main()
