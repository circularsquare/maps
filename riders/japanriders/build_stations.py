"""
build_stations.py — station ridership bubbles for the japan rail flow map.

Input:  data/S12-22/UTF-8/S12-22_NumberOfPassengers.geojson
        国土数値情報 S12 駅別乗降客数データ (MLIT), FY2011-FY2021. One LineString
        feature per station-line. The properties carry 11 fiscal-year blocks of
        4 fields each (S12_006..S12_049): [status, 重複コード, remark, 乗降客数];
        the 乗降客数 (daily boardings + alightings) is the 4th field of a block.

Output: data/stations.geojson
        One Point feature per physical station complex — the bubble layer drawn
        on top of the line-throughput map.

Method:
  - All operators (JR, metro, private, third-sector, tram, monorail).
  - Group records by S12 group code (駅グループコード — stations of the same
    name within 300 m, which S12 assigns ACROSS operators). `S12_001g` is None
    for stations MLIT did not assign — each of those is its own bubble (NOT
    collapsed under a shared "None" key).
  - Per group, pick the latest fiscal year (2021->2011) any record reports
    a positive 乗降客数, and use ONLY that year's figures. This is critical:
    S12's reporting convention changed around FY2019. Before then, every
    operator at a co-managed station independently reported the SAME joint
    figure (寄居 FY2018: Chichibu + JR East + Tobu each 7,476 — triple count).
    From FY2019 MLIT switched to "one operator carries the joint with a
    'X を含む' remark, others report 0". Picking each operator's own latest-
    positive year mixes both regimes and double-counts. Forcing one year per
    group keeps the data internally consistent.
  - Sum across records within the chosen year → one bubble per complex
    carrying the total daily 乗降客数. The bubble sits on the busiest record.

Note: summing 乗降客数 across operators double-counts transfer passengers — the
conventional caveat behind the well-known "新宿 = world's busiest" figure.

Coverage: S12's newest edition is FY2021, so most bubbles predate the
FY2024/FY2023 line data. JR East's S12 feed is partial, and some private
operators withhold unmanned-station counts; every station with visible
ridership is in.

Run directly: python build_stations.py   (fast, ~1s)
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / 'data' / 'S12-22' / 'UTF-8' / 'S12-22_NumberOfPassengers.geojson'
OUT = ROOT / 'data' / 'stations.geojson'

# S12 carries 11 fiscal years, FY2011-FY2021, as 4-field blocks starting at
# S12_006. The 乗降客数 is the 4th field of each block.
YEARS = list(range(2011, 2022))


def per_year_count(p):
    """{year: 乗降客数} for every fiscal year the record reports > 0."""
    out = {}
    for i, y in enumerate(YEARS):
        c = p.get(f'S12_{6 + i * 4 + 3:03d}')
        if c and c > 0:
            out[y] = int(c)
    return out


def midpoint(geom):
    """A station segment is a few metres long, so the mean of its vertices is
    its midpoint for practical purposes."""
    if geom['type'] == 'LineString':
        pts = geom['coordinates']
    elif geom['type'] == 'MultiLineString':
        pts = [pt for part in geom['coordinates'] for pt in part]
    else:                                            # already a Point
        return geom['coordinates']
    n = len(pts)
    return [round(sum(axis) / n, 6) for axis in zip(*pts)]


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    src = json.loads(SRC.read_text(encoding='utf-8'))

    # Group every disclosed record by S12 group code. Records without one
    # (S12_001g is None) become solo groups — collapsing them under a single
    # "None" key fused 259 unrelated stations across Japan into one mega-bubble
    # at 黒部ダム (the busiest single-record member of that None bucket).
    groups = {}    # gkey -> list of (feature, per_year_dict)
    for f in src['features']:
        p = f['properties']
        per_y = per_year_count(p)
        if not per_y:
            continue
        gkey = p.get('S12_001g') or (
            '_solo', p['S12_001'], p['S12_002'], p['S12_003'],
            tuple(midpoint(f['geometry'])))
        groups.setdefault(gkey, []).append((f, per_y))

    feats = []
    for items in groups.values():
        # latest year ANY record in this group reports > 0 — see module docstring
        # for why per-group (not per-record) latest matters
        years_present = set()
        for _, per_y in items:
            years_present.update(per_y)
        year = max(years_present)
        # records with a positive count in the chosen year — these contribute
        records = []
        for f, per_y in items:
            c = per_y.get(year)
            if not c:
                continue
            p = f['properties']
            records.append((c, p['S12_001'], p['S12_002'], p['S12_003'],
                            midpoint(f['geometry'])))
        if not records:
            continue
        records.sort(reverse=True)                                  # busiest first
        riders0, station, op0, line0, coord = records[0]
        total = sum(r[0] for r in records)
        # every operator in the group, busiest-in-chosen-year first; operators
        # silenced by the post-FY2019 "X を含む" convention drop to the tail
        op_val = {}
        for f, _ in items:
            op_ = f['properties']['S12_002']
            op_val.setdefault(op_, 0)
        for r, _, op_, _, _ in records:
            op_val[op_] = max(op_val[op_], r)
        ops_list = sorted(op_val, key=lambda o: -op_val[o])
        feats.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': coord},
            'properties': {
                'station': station,
                'op': op0,                             # busiest operator — drives tooltip colour
                'line': line0,                         # busiest operator's S12 line
                'ops': ops_list,                       # all operators in the group, busiest first
                'riders': total,                       # summed daily 乗降客数 (one year only)
                'year': year,
            },
        })
    feats.sort(key=lambda f: -f['properties']['riders'])
    OUT.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats},
                              ensure_ascii=False), encoding='utf-8')

    # ── report ──
    multi = sum(1 for f in feats if len(f['properties']['ops']) > 1)
    by_yr = Counter(f['properties']['year'] for f in feats)
    print(f'wrote {OUT.name}  stations={len(feats)}  ({multi} multi-operator)')
    print('\nby latest fiscal year:')
    for y in sorted(by_yr):
        print(f'  FY{y}: {by_yr[y]:>5}')
    print('\ntop 12 by summed 乗降客数:')
    for f in feats[:12]:
        p = f['properties']
        tail = ' …' if len(p['ops']) > 3 else ''
        print(f"  {p['riders']:>10,}  FY{p['year']}  {p['station']:<8} "
              f"({len(p['ops'])} op: {' / '.join(p['ops'][:3])}{tail})")


if __name__ == '__main__':
    main()
