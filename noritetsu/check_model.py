"""Check built lines against published lengths and station counts.

    python check_model.py --region jp
    python check_model.py --region jp --find 山手

A line model can be self-consistent and still wrong: sections can be sliced out of the wrong
part of a route, a straight-line fallback can bridge two ends of a prefecture, and the total
still adds up to something.  The only check that catches that is an outside number.  These are
operating lengths (営業キロ) as published by the operators, which is also what the Japanese
line-completion hobby counts, so a ratio far from 1.00 is a real defect and not a definition.
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

# (name fragment, operator fragment, published km, published stations or None, note)
#
# The operator fragment is not optional decoration: 東西線 is three different lines in Tokyo,
# Kyoto and Sapporo, and 中央 matches the Mizuho Shinkansen service via 鹿児島中央.
#
# OSM DOES NOT SPLIT LINES THE WAY THE OPERATORS DO, and the Chuo Main Line is the example:
# officially one 424.6 km line Tokyo-Nagoya, in OSM two route_masters split at Shiojiri where
# JR East hands over to JR Central. Both are checked separately here rather than pretending
# either should total 424.6. The remaining 27.7 km is the old route via Tatsuno.
# Register lines, as 国土数値情報 N02 names them and as the operators publish their 営業キロ.
# N02 writes 山陰線 where the timetable writes 山陰本線, and the register is what it means.
REGISTER = {
    "jp": [
        ("山陰線", "西日本", 673.8, "Kyoto-Hatabu, JR West"),
        ("東海道線", "東海", 341.3, "Atami-Maibara, JR Central"),
        ("東海道線", "西日本", 143.6, "Maibara-Kobe, JR West"),
        ("山陽線", "西日本", 534.4, "Kobe-Moji"),
        ("奥羽線", "東日本", 484.5, "Fukushima-Aomori"),
        ("日豊線", "九州", 462.6, "Kokura-Kagoshima"),
        ("根室線", "北海道", 362.1, "after the 2024 Furano-Shintoku closure"),
        ("山手線", "東日本", 20.6, "Shinagawa-Shinjuku-Tabata, the register line"),
        ("御堂筋線", "", 24.5, "Osaka Metro"),
        ("丸ノ内線", "", 24.2, "Tokyo Metro; Honancho branch is its own register line"),
        ("大江戸線", "", 40.7, "Toei"),
    ],
    # BAV Schienennetz km-lines, by their register name, against the length Wikipedia gives
    # for the line. The chainage check below covers every line; these are the outside numbers.
    "ch": [
        ("St. Moritz - Tirano", "RhB", 60.7, "Bernina line"),
        ("Brig - Visp - Zermatt", "", 44.0, "MGB, ex-BVZ"),
        ("Montreux - Zweisimmen", "", 62.4, "MOB; Zweisimmen-Lenk is its own km-line"),
        ("Zermatt - Gornergrat", "", 9.3, "Gornergratbahn, rack"),
        ("Brig - Andermatt - Disentis", "", 96.9, "MGB, ex-FO, via the Furka base tunnel"),
    ],
}

KNOWN = {
    "jp": [
        ("山手線", "東日本", 34.5, 30, "JR East, loop"),
        ("東海道本線", "", 589.5, None, "Tokyo-Kobe"),
        ("中央線", "東日本", 222.1, None, "Tokyo-Shiojiri, JR East half"),
        ("中央線", "東海", 174.8, None, "Shiojiri-Nagoya, JR Central half"),
        ("銀座線", "", 14.3, 19, "Tokyo Metro"),
        ("丸ノ内線", "", 27.4, 28, "Tokyo Metro, 24.2 main + 3.2 Honancho branch"),
        ("大江戸線", "", 40.7, 38, "Toei"),
        ("御堂筋線", "", 24.5, 20, "Osaka Metro"),
        # Matched on 東京 alone, not the company name: Tokyo Metro is tagged 東京地下鉄 on
        # the Tozai line and 東京メトロ on the Namboku line. Operator strings are free text
        # in OSM and are not consistent even within one company.
        ("東西線", "東京", 30.8, 23, "Tokyo Metro"),
        ("南北線", "東京", 21.3, 19, "Tokyo Metro"),
        ("京浜東北線", "", 81.2, None, "JR East, operating pattern"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--find", default=None, help="print every line matching this fragment")
    ap.add_argument("--shared", default=None,
                    help="for the biggest line matching this, which lines share its track")
    ap.add_argument("--coverage", action="store_true",
                    help="how much of the network only a named train reaches")
    ap.add_argument("--station", default=None,
                    help="every station whose name contains this, and how far apart they are")
    args = ap.parse_args()

    d = ROOT / "dist" / "data" / args.region
    lines = json.loads((d / "lines.json").read_text(encoding="utf-8"))["lines"]
    stations = json.loads((d / "stations.json").read_text(encoding="utf-8"))["stations"]
    byid = {l["id"]: l for l in lines}

    if args.find:
        hits = [l for l in lines if args.find in (l["name"] or "")
                or args.find in (l["ref"] or "")]
        print(f"{len(hits)} lines matching {args.find!r}\n")
        for l in sorted(hits, key=lambda l: -l["km"]):
            print(f"  {l['id']:<10} {l['km']:>8.1f} km  {len(l['sections']):>4} sections  "
                  f"{len(l['display']):>4} display stops  {l['variants']} variants  "
                  f"{l['straight_sections']} straight")
            print(f"      {l['ref']:<6} {l['name']}  [{l['operator']}]  {l['colour']}")
        return

    if args.coverage:
        """How much of the network only a NAMED TRAIN reaches.

        Completion counts lines, not trains. But OSM does not map a line relation everywhere:
        rural Japan often has only the limited expresses. The San'in Main Line, 673 km of JR
        West main line, has no route relation at all and its track is not named either, so
        the only thing calling at Matsue is the Super Oki. Anything counted here is track a
        rider can ride and the totals cannot see.
        """
        svc_only_st, no_line_st = [], []
        for sid, s in stations.items():
            ls = [l for l in (s.get("l") or []) if l in {x["id"] for x in lines}] \
                if False else (s.get("l") or [])
            kinds = [byid[l] for l in ls if l in byid]
            if not kinds:
                no_line_st.append(sid)
            elif all(l["service"] for l in kinds):
                svc_only_st.append(sid)
        gid_service = {}
        for l in lines:
            for a, b, km, gid in l["sections"]:
                if gid not in gid_service:
                    gid_service[gid] = (l["service"], km)
        # A piece of track counts as service-only when no non-service line has a section on
        # it. Sections are per line, so group by the credit ranges instead: approximate by
        # asking whether any non-service line has a section between the same two stations.
        pair_real = set()
        for l in lines:
            if l["service"]:
                continue
            for a, b, km, gid in l["sections"]:
                pair_real.add((a, b) if a <= b else (b, a))
        svc_km = 0.0
        for l in lines:
            if not l["service"]:
                continue
            for a, b, km, gid in l["sections"]:
                if ((a, b) if a <= b else (b, a)) not in pair_real:
                    svc_km += km
        print(f"{len(stations)} stations")
        print(f"  {len(svc_only_st):>5} are served only by a named train, never by a line")
        print(f"  {len(no_line_st):>5} have no line at all")
        print(f"\n{svc_km:,.0f} km of track is reached only by a named train and so counts "
              f"towards nothing")
        print("\nexamples of stations only a named train reaches")
        for sid in svc_only_st[:15]:
            s = stations[sid]
            names = [byid[l]["name"] for l in (s.get("l") or []) if l in byid][:3]
            print(f"  {(s['e'] or s['n']):<26} {', '.join(names)}")
        return

    if args.station:
        import math
        hits = [(i, s) for i, s in stations.items()
                if args.station in (s["n"] or "") or args.station.lower() in (s["e"] or "").lower()]
        print(f"{len(hits)} stations match {args.station!r}\n")
        for i, s in sorted(hits, key=lambda kv: -len(kv[1].get("l", []))):
            print(f"  {i:<14} {s['n']:<12} {s['e']:<22} {len(s.get('l', []))} lines"
                  f"  ({s['y']:.5f}, {s['x']:.5f})")
        # How far apart, so the merge radius can be argued with rather than guessed at again.
        for a in range(len(hits)):
            for b in range(a + 1, len(hits)):
                sa, sb = hits[a][1], hits[b][1]
                dx = (sb["x"] - sa["x"]) * math.cos(math.radians(sa["y"])) * 111320
                dy = (sb["y"] - sa["y"]) * 110570
                d = math.hypot(dx, dy)
                if d < 2000:
                    print(f"    {hits[a][0]} to {hits[b][0]}: {d:.0f} m"
                          f"   {sa['n']!r} / {sb['n']!r}")
        return

    if args.shared:
        hits = [l for l in lines if args.shared in (l["name"] or "")]
        if not hits:
            print(f"nothing matches {args.shared!r}")
            return
        line = max(hits, key=lambda l: l["km"])
        cr = json.loads((d / "credits.json").read_text(encoding="utf-8"))
        # The build writes it keyed by the COVERING section; this wants the other direction.
        covered_by = {}
        for g_s, ranges in cr["covers"].items():
            for g, lo, hi in ranges:
                covered_by.setdefault(str(g), []).append([int(g_s), lo, hi])
        owner, sec_km = {}, {}
        for l in lines:
            for a, b, km, gid in l["sections"]:
                owner[gid], sec_km[gid] = l, km
        mine = {gid for a, b, km, gid in line["sections"]}

        print(f"{line['name']} [{line['operator']}]  {line['km']:.1f} km, "
              f"{len(line['sections'])} sections")
        print(f"corridor buffer {cr['buffer_m']} m\n")
        print("If you rode all of it, what else would that complete:\n")

        tally = {}
        for gid_s, ranges in covered_by.items():
            gid = int(gid_s)
            if gid in mine:
                continue
            spans = sorted((lo, hi) for g, lo, hi in ranges if g in mine)
            if not spans:
                continue
            frac, end = 0.0, -1.0
            for lo, hi in spans:                 # union of the covered ranges
                lo = max(lo, end)
                if hi > lo:
                    frac += hi - lo
                    end = hi
            if frac <= 0:
                continue
            o = owner[gid]
            name = o["name"] or o["id"]
            t = tally.setdefault(name, [0.0, 0.0])
            t[0] += frac * sec_km[gid]
            t[1] += sec_km[gid]
        print(f"{'km credited':>12}  {'of line':>9}  line")
        for name, (km, total) in sorted(tally.items(), key=lambda kv: -kv[1][0])[:15]:
            print(f"{km:>12.1f}  {total:>9.1f}  {name}")
        print(f"\n{len(tally)} other lines get some credit from riding this one")
        return

    print(f"{len(lines)} lines, {len(stations)} stations, "
          f"{sum(l['km'] for l in lines):,.0f} route-km\n")

    reg = [l for l in lines if l.get("src", "osm") != "osm"]
    if reg:
        print(f"{len(reg)} register lines, {sum(l['km'] for l in reg):,.0f} km"
              + (" (Japan's passenger network is about 27,300 km)" if args.region == "jp"
                 else "") + "\n")
        # A register that publishes its own chainage (km_official) can check every line, not
        # just the handful in REGISTER. It is the register's measure of the same track the
        # section walk used, so a ratio off 1.00 is the walk going wrong -- the double-track
        # doubling n02.py describes would show up here as 2.00.
        chk = [l for l in reg if l.get("km_official") and l["km"] >= 2]
        if chk:
            rs = sorted(l["km"] / l["km_official"] for l in chk)
            off = sorted((l for l in chk if abs(l["km"] / l["km_official"] - 1) > 0.05),
                         key=lambda l: l["km"] / l["km_official"])
            print(f"built against the register's own chainage, {len(chk)} lines of 2 km or "
                  f"more: median {rs[len(rs)//2]:.3f}, {len(off)} off by more than 5%")
            for l in off:
                print(f"  {l['km'] / l['km_official']:>5.2f}  {l['km']:>7.1f} of "
                      f"{l['km_official']:>7.1f}  {l['ref']:>8}  {l['name']}")
            print()
        print(f"{'register line':<16} {'built':>8} {'published':>10} {'ratio':>7}  note")
        worst_r = 0.0
        for frag, op, km, note in REGISTER.get(args.region, []):
            hits = [l for l in reg if frag == (l["name"] or "")
                    and (not op or op in (l["operator"] or ""))]
            if not hits:
                print(f"{frag:<16} {'NOT FOUND':>8}")
                worst_r = max(worst_r, 9.99)
                continue
            # With no operator given, SUM them: the register splits a line where it changes
            # hands, so 東海道線 is three companies and no one of them is the line.
            built = max(l["km"] for l in hits) if op else sum(l["km"] for l in hits)
            l = {"km": built}
            ratio = built / km
            worst_r = max(worst_r, abs(ratio - 1))
            flag = " <--" if abs(ratio - 1) > 0.05 else ""
            print(f"{frag:<16} {built:>8.1f}  {km:>9.1f}  {ratio:>6.2f}{flag}"
                  f"  {note}{'' if op else f'  [{len(hits)} operators]'}")
        print(f"\nworst register deviation {worst_r:.2f}\n")

    # OSM-derived objects only: with a register loaded, these names also match register
    # lines, and the two measure different things (the Yamanote LOOP against the Yamanote
    # register line) so comparing them to one published figure is meaningless.
    osm_lines = [l for l in lines if l.get("src", "osm") == "osm"] if reg else lines
    print(f"{'OSM line':<20} {'built':>8} {'published':>10} {'ratio':>7}  {'stops':>9}  note")
    worst = 0.0
    for frag, op, km, stops, note in KNOWN.get(args.region, []):
        hits = [l for l in osm_lines if frag in (l["name"] or "")
                and (not op or op in (l["operator"] or ""))]
        if not hits and reg:
            # An OSM line that was the register line twice over is dropped in the merge
            # (build_model.is_twin), so the register line is what there is to check.
            hits = [l for l in reg if frag in (l["name"] or "")
                    and (not op or op in (l["operator"] or ""))]
            note += " [register; OSM twin dropped]"
        if not hits:
            print(f"{frag:<20} {'NOT FOUND':>8}")
            worst = max(worst, 9.99)
            continue
        l = max(hits, key=lambda l: l["km"])
        ratio = l["km"] / km
        worst = max(worst, abs(ratio - 1))
        # Counted from the SECTIONS, not the display order: the display order is one variant
        # and so misses a branch's stations, which is how Marunouchi read 25 against 28.
        built = len({s for sec in l["sections"] for s in sec[:2]})
        s = f"{built}" + (f"/{stops}" if stops else "")
        flag = " <--" if abs(ratio - 1) > 0.05 else ""
        print(f"{frag:<20} {l['km']:>8.1f}  {km:>9.1f}  {ratio:>6.2f}  {s:>9}{flag}  {note}")
    print(f"\nworst deviation {worst:.2f}")


if __name__ == "__main__":
    main()
