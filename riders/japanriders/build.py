"""
Build a clean per-segment GeoJSON for the national Japan rail throughput map.

Input:  data/tsukajinin_20260117.geojson
        gtfs-gis.jp consolidated 輸送密度 dataset (Akira Nishizawa / U-Tokyo CSIS).
        One feature per railway segment: geometry + passenger density (人/日)
        for fiscal years 2018-2024.

Output: data/segments.geojson
        Same geometry, slimmed properties, plus a chosen display density:
        the latest available fiscal year per segment (FY2024, falling back to
        FY2023 and earlier). All 7 years are kept for a possible year slider.

Sentinels dropped: -9999 (not surveyed) and -9000 (service suspended — e.g.
flood-damaged sections of 肥薩線, 日田彦山線).

Run directly: python build.py   (fast, ~1s)
"""
import io
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / 'data' / 'tsukajinin_20260117.geojson'
OUT = ROOT / 'data' / 'segments.geojson'

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# The raw file uses uppercase geometry types ("LINESTRING"), which violate the
# GeoJSON spec and are rejected by strict parsers (MapLibre). Normalize to PascalCase.
GEOM_TYPE = {'LINESTRING': 'LineString', 'MULTILINESTRING': 'MultiLineString'}

# Hand-curated map from segment from/to to the canonical name used in
# stations.geojson (国土数値情報 S12). Sourced from audit_fromto.py — every
# entry was confirmed to (a) not exist as a station under its original spelling
# and (b) exist under the rewritten spelling. Covers:
#   * kana variants NFKC doesn't fold: ヶ↔ケ (compound-name particle), ッ↔ツ
#   * historical katakana spelling: フィ↔フイ (Fujifilm)
#   * station renames where one side hasn't caught up
# Without this, the runtime's geomFromTo would still pick the right station,
# but each tooltip would carry a "*" marking the rescue. Folding the rename
# upstream removes the noise.
NAME_FIXES = {
    # ケ → ヶ (segments use big katakana KE; stations use small ヶ)
    '苧ケ瀬':     '苧ヶ瀬',
    '三好ケ丘':   '三好ヶ丘',
    '須ケ口':     '須ヶ口',
    '巽ケ丘':     '巽ヶ丘',
    '霞ケ関':     '霞ヶ関',
    '西ケ原':     '西ヶ原',
    '市ケ谷':     '市ヶ谷',
    '南阿佐ケ谷': '南阿佐ヶ谷',
    '宝ケ池':     '宝ヶ池',
    '泉ケ丘':     '泉ヶ丘',
    '尼ケ坂':     '尼ヶ坂',
    # ヶ → ケ — only one station the other way (JR 高山線 各務ケ原)
    '各務ヶ原':   '各務ケ原',
    # ッ → ツ
    '三ッ沢上町': '三ツ沢上町',
    '三ッ沢下町': '三ツ沢下町',
    '四ッ谷':     '四ツ谷',
    # フィ → フイ (the company spells itself フイルム, even though フィルム reads same)
    '富士フィルム前': '富士フイルム前',
    # Renames where stations.geojson has the modern name (S12-22 was after the rename)
    '花月園前':       '花月総持寺',           # 2020
    '仲木戸':         '京急東神奈川',          # 2020
    '新逗子':         '逗子・葉山',           # 2020
    '南町田':         '南町田グランベリーパーク',  # 2019
    '西武遊園地':     '多摩湖',              # 2021
    '産業道路':       '大師橋',              # 2020
    '松原団地':       '獨協大学前',           # 2017
    '国際展示場正門': '東京ビッグサイト',        # 2019
    '船の科学館':     '東京国際クルーズターミナル',  # 2019
    # Source data quirks
    '?利伽羅':         '倶利伽羅',             # mojibake in tsukajinin (倶 → 0x3F)
    '広電西広島(己斐)': '広電西広島（己斐）',     # halfwidth → fullwidth parens
    '西鉄福岡(天神)':   '西鉄福岡（天神）',
    '弘前中央':         '中央弘前',             # segment data has it backwards
}
# Scoped overrides for ambiguous names: the bare name exists as a real station
# elsewhere so a blanket fold would corrupt it. Keyed by operator.
PER_OP_FIXES = {
    # JR 江北 (renamed 2022, S12-22 still has 肥前山口) collides with the
    # 日暮里・舎人ライナー 江北 in Tokyo, which is correctly named 江北.
    ('九州旅客鉄道', '江北'): '肥前山口',
    # 仙台 南北線's southern terminus is 泉中央; the gtfs-gis.jp source wrote
    # 和泉中央, which is the (entirely different) Osaka Senboku terminus.
    ('仙台市', '和泉中央'): '泉中央',
}


def normalize_name(op, s):
    """Canonicalize a segment from/to to match stations.geojson naming."""
    if not s:
        return s
    s = unicodedata.normalize('NFKC', s)   # folds fullwidth digits / latin (ＪＲ → JR, ２ → 2)
    return PER_OP_FIXES.get((op, s)) or NAME_FIXES.get(s, s)


def clean(v):
    """Year value -> float, or None for missing / sentinel / negative."""
    if v is None or v < 0:          # -9999 not surveyed, -9000 service suspended
        return None
    return float(v)


def build_base(keep_empty=False):
    """Parse the raw gtfs-gis.jp file into clean Feature dicts.

    Returns (features, n_dropped). Shared with build_census.py (the v2 stitch).
    With keep_empty=True, features that have no 輸送密度 in any year are also
    returned (density=None) — build_census.py uses those for their geometry,
    to stitch in census detail for lines gtfs-gis.jp never measured.
    """
    src = json.loads(SRC.read_text(encoding='utf-8'))

    out = []
    no_data = 0
    for f in src['features']:
        p = f['properties']
        years = {y: clean(p.get(f'{y}年度')) for y in YEARS}

        # display density: latest available fiscal year, newest first
        density = year = None
        for y in reversed(YEARS):
            if years[y] is not None:
                density, year = years[y], y
                break
        if density is None:
            no_data += 1
            if not keep_empty:
                continue

        op = p['事業者名']
        props = {
            'op': op,
            'line': p['路線名'],
            'from': normalize_name(op, p['始点駅']),
            'to':   normalize_name(op, p['終点駅']),
            'kind': p['種類'],
            'km': p['営業キロ'],
            'remark': p['備考'],
            'density': density,
            'year': year,
        }
        for y in YEARS:
            props[f'y{y}'] = years[y]

        geom = f['geometry']
        geom['type'] = GEOM_TYPE.get(geom['type'], geom['type'])

        out.append({
            'type': 'Feature',
            'geometry': geom,
            'properties': props,
        })

    return out, no_data


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    out, no_data = build_base()

    fc = {'type': 'FeatureCollection', 'features': out}
    OUT.write_text(json.dumps(fc, ensure_ascii=False), encoding='utf-8')

    out.sort(key=lambda f: -f['properties']['density'])
    print(f'wrote {OUT.name}  segments={len(out)}  dropped(no data any year)={no_data}')
    print('\ntop 10 by display density:')
    for f in out[:10]:
        p = f['properties']
        print(f"  {p['density']:>9,.0f}  FY{p['year']}  {p['op']} {p['line']}  {p['from']}-{p['to']}")


if __name__ == '__main__':
    main()
