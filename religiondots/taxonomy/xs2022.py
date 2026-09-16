"""Israeli settlements in the West Bank: CBS 2022 census -> religiondots taxonomy.

Israel's mapping (`il2022.py`), restricted to the two groups this entry draws: Jews, with the
observance rows, and the register's Others. Nothing is re-decided here; every node, and every
REVIEW reason for those groups, is Israel's.

**Muslims, Christians and Druze are EXCLUDED, and not because they are unmapped.** CBS counts
373,257 Muslims and Christians in the same units beyond the Green Line, nearly all in East
Jerusalem, and Palestine's 2017 census counts East Jerusalem's Palestinians (`sources/ps.md` §2).
Palestine's entry draws them. Drawing them here too would draw the same people twice. Keeping them
out of MAP also keeps `islam` and `christianity` out of this entry's coverage, so selecting Islam
leaves the entry unlit ("not asked here") rather than lit and empty.

`sources/xs.py` already writes only the Jews and Others rows to `data/normalized/xs.csv`; the
EXCLUDED entries are there so a future xs.csv that carries the other groups fails nothing silently
and draws none of them.
"""

import os
import sys

# THE ONLY MAPPING THAT IMPORTS ANOTHER COUNTRY'S. countries.py, check_mapping.py and coverage.py put
# taxonomy/ on sys.path and import this as `xs2022`, and the bare import below works there;
# tools/review_dump.py puts only the repo root there and imports `taxonomy.xs2022`, where it failed
# with "No module named 'il2022'". So add this directory the way the sources/ scripts add theirs
# before importing a sibling. Appended, not inserted, so it never changes which module wins a name
# in a process that already has taxonomy/ on the path (sources/geo_checks.py, module shadowing).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from il2022 import EXCLUDED as _IL_EXCLUDED  # noqa: E402
from il2022 import MAP as _IL_MAP  # noqa: E402
from il2022 import REVIEW as _IL_REVIEW  # noqa: E402
from il2022 import _key  # noqa: E402

GROUPS = ("Jews", "Others")


def _group(cat):
    return _key(cat).split(" [")[0]


_ELSEWHERE = ("drawn on Palestine's entry, not here: in CBS's units beyond the Green Line these "
              "are nearly all East Jerusalem's Palestinians, whom PCBS's 2017 census counts "
              "(sources/xs.py).")

EXCLUDED = dict(_IL_EXCLUDED)
EXCLUDED.update({"Muslims": _ELSEWHERE, "Christians": _ELSEWHERE, "Druze": _ELSEWHERE})

MAP = {k: v for k, v in _IL_MAP.items() if _group(k) in GROUPS}

REVIEW = {k: v for k, v in _IL_REVIEW.items() if _group(k) in GROUPS}
REVIEW["Christians"] = (
    "EXCLUDED here and drawn on Palestine's entry. The register does not say which Christians "
    "are Arab, so the non-Arab Christians living in these units (ex-Soviet immigrants, as in "
    "Israel's entry) are on neither. Most of the 13,712 are in East Jerusalem; outside it the "
    "settlements hold 61.")

EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
