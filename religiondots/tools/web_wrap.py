"""Copy a map's index.html into the website repo the way the site expects it.

    python tools/web_wrap.py <source index.html> <website dir> [--sitemap false]

A PLAIN COPY LOSES TWO THINGS AND NEITHER OF THEM IS VISIBLE.  Every page in the website repo
carries Jekyll front matter and an `{%- include analytics.html -%}` line in its head; the
working copy in maps/ carries neither, because it has to run from a bare `npx serve` where
Liquid does not exist.  Overwriting the deployed file with the working one therefore drops the
site's analytics silently -- religiondots shipped that way until 2026-09-09, and nothing about
the page looked wrong.

Front matter is PRESERVED from whatever is already at the destination rather than regenerated,
because it carries per-page settings that live nowhere else: korearail's is `sitemap: false`,
which is how an unpublicised map stays out of the sitemap.
"""

import argparse
import io
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANALYTICS = "{%- include analytics.html -%}"
CHARSET = re.compile(r'^([ \t]*)<meta\s+charset=[^>]*>\s*$', re.I | re.M)


def existing_front_matter(path):
    """The destination's own front matter, or None.  `---` on line 1, up to the next `---`."""
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as f:
        head = f.read(4096)
    if not head.startswith("---"):
        return None
    end = head.find("\n---", 3)
    if end < 0:
        return None
    return head[:end + 4].rstrip("\n") + "\n"


def wrap(src_path, dst_path, sitemap):
    src = io.open(src_path, encoding="utf-8").read()
    if src.startswith("---"):
        raise SystemExit(f"{src_path} already has front matter; this is the working copy's job "
                         f"to not have.  Refusing to double-wrap.")

    fm = existing_front_matter(dst_path)
    if fm is None:
        fm = "---\nsitemap: false\n---\n" if sitemap == "false" else "---\n---\n"
        note = f"new front matter ({fm.strip().splitlines()[1:] or 'empty'})"
    else:
        note = "front matter preserved"

    if ANALYTICS in src:
        out = fm + src
        note += ", analytics already present"
    else:
        m = CHARSET.search(src)
        if not m:
            raise SystemExit(f"{src_path}: no <meta charset> to anchor the analytics include on")
        at = m.end()
        out = fm + src[:at] + "\n" + m.group(1) + ANALYTICS + src[at:]
        note += ", analytics inserted"

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    io.open(dst_path, "w", encoding="utf-8", newline="").write(out)
    print(f"  {dst_path}\n     {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst_dir")
    ap.add_argument("--sitemap", default="false",
                    help="'false' writes `sitemap: false` into NEW front matter; ignored when "
                         "the destination already has some")
    a = ap.parse_args()
    wrap(a.src, os.path.join(a.dst_dir, "index.html"), a.sitemap)


if __name__ == "__main__":
    main()
