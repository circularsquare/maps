# -*- coding: utf-8 -*-
"""Copy the map into the Jekyll site as website/korearail/.

Explicit only -- nothing here runs on its own and nothing else calls it. The
site is a separate repo (`projects/website`), Anita reviews and commits it
herself, and japanriders got out of sync with its published copy precisely
because the mirroring was manual and ad hoc.

    python publish.py            # copy, reporting what changed
    python publish.py --check    # say what would change, write nothing

What it does beyond copying:

  * prepends Jekyll's empty front matter, without which the Liquid tag below
    is served as literal text;
  * inserts `{%- include analytics.html -%}` after the charset meta, which is
    where every other page on the site carries it;
  * adds og:image, the two twitter card tags and the favicon link. The page
    already carries og:title, og:description and og:type, so only the image
    side is missing.

`index.png` and `favicon.png` are **not** generated here -- they are a
screenshot and an icon, and a stale preview image is worse than none. The
script says whether they exist and leaves them alone.
"""

import argparse
import io
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
SITE = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "website", "korearail"))

# Everything index.html fetches through dataURL(). Keep in step with it: a
# missing file is a layer that silently does not draw, since every fetch is
# wrapped in a try or an `r.ok` test.
DATA = [
    "segments.geojson",
    "stations.geojson",
    "metro_segments.geojson",
    "metro_stations.geojson",
    "busan_segments.geojson",
    "daegu_busan_segments.geojson",
    "small_city_segments.geojson",
    "donghae_segments.geojson",
    "gyeongchun_segments.geojson",
]

HEAD_EXTRA = (
    '<meta property="og:image" content="index.png" />\n'
    '<meta name="twitter:card" content="summary_large_image" />\n'
    '<meta name="twitter:image" content="index.png" />\n'
    '<link rel="icon" type="image/png" href="favicon.png" />\n'
)

# Unlisted is the default and has to be, because the site leaks a new page
# three ways and only one of them is obvious:
#
#   * `jekyll-sitemap` puts every built page in sitemap.xml automatically.
#     `sitemap: false` in the front matter is the documented opt-out.
#   * a search engine that finds the URL some other way (a shared link, a
#     referrer header) will index it regardless of the sitemap, so the robots
#     meta is the actual instruction and the sitemap flag is only tidiness.
#   * pages/projects/projects.md is the site's own list -- but it is hand
#     written, so leaving it alone is enough. Nothing auto-adds.
#
# The feed does not leak: scripts/build_feed_data.py walks `pages/` only, and
# this lands at the repo root.
NOINDEX = '<meta name="robots" content="noindex, nofollow" />\n'


def transform(src, public=False):
    """The source page as the site should serve it."""
    if src.startswith("---"):
        raise SystemExit("index.html already has front matter -- refusing")

    out = src.replace(
        '<meta charset="utf-8">',
        '<meta charset="utf-8">\n{%- include analytics.html -%}', 1)

    # Put the image tags with the og ones rather than at the top of <head>,
    # so the block reads as a unit to anyone editing it later.
    m = re.search(r'<meta property="og:type"[^>]*/>\n', out)
    if not m:
        raise SystemExit("no og:type meta to anchor the image tags to")
    extra = HEAD_EXTRA if public else HEAD_EXTRA + NOINDEX
    out = out[:m.end()] + extra + out[m.end():]

    front = "---\n---\n" if public else "---\nsitemap: false\n---\n"
    return front + out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change, write nothing")
    ap.add_argument("--public", action="store_true",
                    help="drop the noindex and the sitemap opt-out "
                         "(default is unlisted: live, but not indexed)")
    args = ap.parse_args()

    src = io.open(os.path.join(HERE, "index.html"), encoding="utf-8").read()
    page = transform(src, public=args.public)

    print("site: %s" % SITE)
    print("   %s" % ("PUBLIC -- indexable and in the sitemap" if args.public
                     else "unlisted -- noindex + sitemap:false; the URL works "
                          "for anyone given it"))
    if not args.check:
        for d in (SITE, os.path.join(SITE, "data")):
            if not os.path.isdir(d):
                os.makedirs(d)

    dest = os.path.join(SITE, "index.html")
    old = (io.open(dest, encoding="utf-8").read()
           if os.path.exists(dest) else None)
    if old == page:
        print("   index.html          unchanged")
    else:
        print("   index.html          %s (%d bytes)"
              % ("would write" if args.check else "written", len(page)))
        if not args.check:
            io.open(dest, "w", encoding="utf-8", newline="\n").write(page)

    for name in DATA:
        s = os.path.join(D, name)
        if not os.path.exists(s):
            print("   %-22s MISSING from data/ -- layer will not draw" % name)
            continue
        t = os.path.join(SITE, "data", name)
        same = (os.path.exists(t)
                and os.path.getsize(t) == os.path.getsize(s)
                and os.path.getmtime(t) >= os.path.getmtime(s))
        if same:
            print("   %-22s unchanged" % name)
            continue
        print("   %-22s %s (%.1f MB)"
              % (name, "would copy" if args.check else "copied",
                 os.path.getsize(s) / 1e6))
        if not args.check:
            shutil.copy2(s, t)

    for name in ("index.png", "favicon.png"):
        if not os.path.exists(os.path.join(SITE, name)):
            print("   %-22s absent -- needed for the social preview and tab "
                  "icon; not generated here" % name)

    ver = re.search(r"const DATA_VERSION = '([^']+)'", src)
    print("   DATA_VERSION %s -- bump it in index.html on any data change, or "
          "browsers serve the old geojson" % (ver.group(1) if ver else "?"))
    print("\nnothing is committed; the site is a separate repo to review.")


if __name__ == "__main__":
    main()
