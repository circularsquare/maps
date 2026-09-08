"""Fetch the CGSS waves that are open, into data/raw/cn/cgss/.

Three waves that CNSDA puts behind a data application are sitting unrestricted in a
replication package on Harvard Dataverse: **CGSS 2010, 2011 and 2013**, in
`doi:10.7910/DVN/R1S5RP`, "Meritocracy as Authoritarian Co-Optation" (which also carries a
second copy of 2012). Each is a near-complete wave, not a variable subset: 963, 592 and 650
variables, and all three keep BOTH things this project needs, which a replication subset very
easily loses:

    s41    采访地点-省/自治区/直辖市     the province of interview
    a5     您的宗教信仰是                2010, single choice
    a501 + a511..a521                    2011 and 2013, the multi-select block

That is the same pair of shapes `cn_cgss.py` already handles for 2012/2017 and 2021.

**`?format=original` IS NOT OPTIONAL.** Dataverse ingests .dta into a .tab and the ingested
copy loses every value label, including the province names in s41 — which would leave the
province column as bare integers and silently break the join. sources/cn_cgss.md records this
trap for the 2012/2017 fetch and it applies identically here.

**Licence.** The dataset's terms are *"not to be distributed/posted outside of the Harvard
Dataverse. All downloads should take place directly on Harvard Dataverse."* That is a
no-redistribution term, not a no-use one, and this script does exactly what it asks by pulling
from Harvard directly. Per §6b the depositor is a third party (the paper's authors), so cite
**Renmin University's CNSDA / CGSS** as the origin, never the replication package.

**Still walled after this: 2015, 2018 and 2023.** Searched and not found open, see
sources/cn_clds.md and spec §14.20.

    python sources/cn_cgss_fetch.py
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "cn", "cgss")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0 Safari/537.36"

# Harvard Dataverse file ids in doi:10.7910/DVN/R1S5RP, with the size the API reports for
# the INGESTED .tab; the original .dta comes back a different size, which is expected.
WAVES = {
    "cgss2010.dta": 7424597,
    "cgss2011.dta": 7424594,
    "cgss2013.dta": 7424577,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    for name, fid in WAVES.items():
        dest = os.path.join(OUT, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"  {name}: already here, {os.path.getsize(dest)/1e6:.1f} MB")
            continue
        url = (f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
               f"?format=original")
        print(f"  {name}: fetching file {fid} ...", flush=True)
        tmp = dest + ".part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r, \
                    open(tmp, "wb") as fh:
                disp = r.headers.get("Content-Disposition", "")
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(tmp, dest)
            print(f"      {os.path.getsize(dest)/1e6:.1f} MB   {disp}")
        except Exception as e:
            print(f"      FAILED: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
