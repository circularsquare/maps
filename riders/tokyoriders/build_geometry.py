"""
Join segments_kanto.json (from build_segments.py) with N02 railway geometry to produce
a GeoJSON FeatureCollection of per-segment ridership lines.

v1 strategy: represent each segment as a straight line between its two stations' coordinates.
Station coords come from N02-24_Station.geojson (LineString midpoint used as the anchor).

Name normalization handles:
  - operator aliases (東京都交通局 <-> 東京都, etc.)
  - "本線" vs "線" endings
  - parenthesized disambiguation (京葉線（１） etc.)
  - "N号線XXX線" prefixes used by N02 for subway lines
  - small/large kana variants (四ッ谷 vs 四ツ谷)

Unmatched stations / segments are reported and skipped.

Output:
  data/processed/segments_kanto.geojson
"""

import json
import re
import sys
import io
import collections
import unicodedata
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
OUT = DATA / 'processed'

N02_STATION = DATA / 'N02-24' / 'UTF-8' / 'N02-24_Station.geojson'
SEGMENTS_JSON = OUT / 'segments_kanto.json'
OUT_GEOJSON = OUT / 'segments_kanto.geojson'

# Census operator -> list of N02 operator aliases (first match wins)
OPERATOR_ALIASES = {
    '東京都交通局': ['東京都'],
    '横浜市交通局': ['横浜市'],
    '東京急行電鉄': ['東急電鉄'],       # renamed 2019
    '小湊鉄道': ['小湊鐵道'],            # 鉄 vs 鐵
    '箱根登山鉄道': ['小田急箱根'],      # merged 2024
}

# Census station name (pre-rename) -> N02 current name. Many Kanto stations were renamed 2017–2021.
STATION_ALIASES = {
    '松原団地': '獨協大学前',                  # 東武伊勢崎線 2017
    '仲木戸': '京急東神奈川',                  # 京急本線 2020
    '花月園前': '花月総持寺',                  # 京急本線 2020
    '新逗子': '逗子・葉山',                    # 京急逗子線 2020
    '産業道路': '大師橋',                      # 京急大師線 2020
    '南町田': '南町田グランベリーパーク',      # 東急田園都市線 2019
    '西武遊園地': '多摩湖',                    # 西武多摩湖線/山口線 2021
    '遊園地西': '多摩湖',
    '船の科学館': '東京国際クルーズターミナル',  # ゆりかもめ 2019
    '国際展示場正門': '東京ビッグサイト',        # ゆりかもめ 2019
    '羽田空港国際線ターミナル': '羽田空港第3ターミナル',
    '羽田空港国際線ビル': '羽田空港第3ターミナル',
    '羽田空港国内線ターミナル': '羽田空港第1・第2ターミナル',
    '羽田空港第1ビル': '羽田空港第1ターミナル',
    '羽田空港第2ビル': '羽田空港第2ターミナル',
    '佐貫': '龍ケ崎市',                        # JR常磐線 2020 (関東鉄道 still has 佐貫)
}

SMALL_TO_LARGE_KANA = str.maketrans({
    'ッ': 'ツ', 'ャ': 'ヤ', 'ュ': 'ユ', 'ョ': 'ヨ',
    'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ',
    'ヵ': 'カ', 'ヶ': 'ケ',
})


def norm_str(s):
    if s is None:
        return ''
    s = unicodedata.normalize('NFKC', s)
    return s.strip()


def norm_line(n):
    n = norm_str(n)
    n = re.sub(r'[（(].*?[)）]', '', n)          # drop parens
    n = re.sub(r'^\d+号線', '', n)               # drop N号線 prefix (subways)
    if n.endswith('本線'):
        n = n[:-2] + '線'
    return n.strip()


def norm_station(n):
    return norm_str(n).translate(SMALL_TO_LARGE_KANA)


# Kanto bbox — census only covers 首都圏; loose fallback must stay inside this box
# or it picks same-named stations in Tohoku/Kansai (e.g. 大久保 in Akita vs Tokyo).
KANTO_BBOX = (138.5, 34.8, 141.2, 37.3)  # lon_min, lat_min, lon_max, lat_max


def in_kanto(lonlat):
    x, y = lonlat
    return KANTO_BBOX[0] <= x <= KANTO_BBOX[2] and KANTO_BBOX[1] <= y <= KANTO_BBOX[3]


def load_n02_stations():
    """Return two dicts:
       exact: (op, norm_line, norm_stn) -> (lon, lat)
       loose: (op, norm_stn)            -> (lon, lat)   (Kanto-bbox-restricted, first-seen wins)
    """
    exact = {}
    loose = {}
    with open(N02_STATION, encoding='utf-8') as f:
        d = json.load(f)
    for feat in d['features']:
        p = feat['properties']
        op = norm_str(p['N02_004'])
        ln = norm_line(p['N02_003'])
        st = norm_station(p['N02_005'])
        coords = feat['geometry']['coordinates']
        mid = coords[len(coords) // 2]
        lonlat = (mid[0], mid[1])
        exact.setdefault((op, ln, st), lonlat)
        if in_kanto(lonlat):
            loose.setdefault((op, st), lonlat)
    return exact, loose


def station_coord(op, line, stn, exact, loose):
    """Find station coord. Returns (lon,lat) or None."""
    op_n = norm_str(op)
    ln_n = norm_line(line)
    st_n = norm_station(stn)
    # try both original and alias station name (station renames 2017–2021)
    st_candidates = [st_n]
    if stn in STATION_ALIASES:
        st_candidates.append(norm_station(STATION_ALIASES[stn]))

    op_candidates = [op_n] + OPERATOR_ALIASES.get(op_n, [])
    for c_op in op_candidates:
        for c_st in st_candidates:
            if (c_op, ln_n, c_st) in exact:
                return exact[(c_op, ln_n, c_st)]
    for c_op in op_candidates:
        for c_st in st_candidates:
            if (c_op, c_st) in loose:
                return loose[(c_op, c_st)]
    return None


def main():
    exact, loose = load_n02_stations()
    print(f'N02 station lookups: exact={len(exact)}  loose={len(loose)}')

    with open(SEGMENTS_JSON, encoding='utf-8') as f:
        data = json.load(f)
    segments = data['segments']
    stations = data['stations']
    lines = data['lines']

    # Resolve coords for every station code
    coord_by_code = {}
    missing_ops = collections.Counter()
    for code, meta in stations.items():
        c = station_coord(meta['op_name'], meta['line_name'], meta['name'], exact, loose)
        if c is not None:
            coord_by_code[code] = c
        else:
            missing_ops[(meta['op_name'], meta['line_name'])] += 1
    print(f'resolved {len(coord_by_code)}/{len(stations)} station codes')
    if missing_ops:
        print('top unresolved (op, line):')
        for k, v in missing_ops.most_common(10):
            print(f'  {v:>4}  {k}')

    feats = []
    dropped = 0
    for s in segments:
        a = coord_by_code.get(s['from_code'])
        b = coord_by_code.get(s['to_code'])
        if a is None or b is None:
            dropped += 1
            continue
        feats.append({
            'type': 'Feature',
            'geometry': {'type': 'LineString', 'coordinates': [list(a), list(b)]},
            'properties': {
                'line_code': s['line_code'],
                'line_name': s['line_name'],
                'op_name': s['op_name'],
                'from': s['from_name'],
                'to': s['to_name'],
                'riders': s['riders'],
            },
        })

    fc = {'type': 'FeatureCollection', 'features': feats}
    OUT_GEOJSON.write_text(json.dumps(fc, ensure_ascii=False), encoding='utf-8')
    print(f'\nwrote {OUT_GEOJSON}  kept={len(feats)}  dropped={dropped}')


if __name__ == '__main__':
    main()
