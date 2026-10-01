"""Belgium: Infrabel's RINF id is four digits. NNN0 is line NNN (0360 is 36, 1620 is 162). The
other endings are Infrabel's own variant codes and do NOT map onto the public letters
(0503 is 50A, 0366 is 36N), so those are named from OSM's relations instead. Two groups are
known and fixed here: line 0, the Brussels North-South junction, is six ids, one per track;
the four high-speed lines are 9010-9040."""
import re

from rinf_countries import osm_ref_default

BE_FIXED = {**{f"00{k}0": "0" for k in range(1, 7)},
            "9010": "1", "9020": "2", "9030": "3", "9040": "4"}


def be_ref(line_id):
    m = re.fullmatch(r"(\d{3})0", line_id or "")
    return str(int(m.group(1))) if m else None


def be_osm_ref(ref):
    """OSM maps line 0's three track pairs as L01, L02 and L03."""
    r = osm_ref_default(ref)
    return "0" if r in ("01", "02", "03") else r


COUNTRY = {
    "iso3": "BEL", "wikidata": "Q31", "langs": ["nl", "fr", "en", "de"],
    "fixed": BE_FIXED, "ref": be_ref, "osm_ref": be_osm_ref,
    # Infrabel's network statement writes "L.36", "L.3" (hogesnelheidslijnen L.3 en L.4).
    "name": "L.{ref}", "name_en": "Line {ref}",
    "im": {"0088_IM": "Infrabel"},
}
