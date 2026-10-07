"""ISO 3166 numeric -> alpha-2, from Natural Earth's admin 0 countries (religiondots' copy,
read-only). For census servers that code country of birth by ISO numeric (Costa Rica 2011).

    from iso_numeric import alpha2
    alpha2(188)   # 'CR'
"""
import json
from pathlib import Path

NE = (Path(__file__).resolve().parent.parent.parent / "religiondots" / "data" / "geo"
      / "ne_10m_admin_0_countries.geojson")
# codes Natural Earth lacks or gives -99 (dissolved or small)
EXTRA = {250: "FR", 578: "NO", 530: "CW", 891: "YU", 810: "SU", 200: "QT", 336: "VA",
         312: "GP", 474: "MQ", 254: "GF", 638: "RE", 175: "YT", 744: "NO", 162: "AU",
         166: "AU", 334: "AU", 581: "US", 732: "EH", 830: "GB", 833: "IM", 831: "GG", 832: "JE",
         535: "BQ", 534: "SX", 663: "MF", 652: "BL", 158: "TW", 344: "HK", 446: "MO",
         275: "PS", 383: "XK", 728: "SS", 729: "SD"}
_map = None


def table():
    global _map
    if _map is None:
        d = json.loads(NE.read_text(encoding="utf-8"))
        _map = {}
        for f in d["features"]:
            p = f["properties"]
            try:
                n = int(p.get("ISO_N3") or p.get("ISO_N3_EH") or -99)
            except ValueError:
                continue
            a2 = p.get("ISO_A2") if p.get("ISO_A2") not in (None, "-99") else p.get("ISO_A2_EH")
            if n > 0 and a2 and a2 != "-99":
                _map[n] = a2
        _map.update(EXTRA)
    return _map


def alpha2(n):
    return table().get(int(n))
