#!/usr/bin/env python3
"""Build the player wiki as static pages for the website (T-3447).

    python trainworld/tools/wiki_site.py                  live version's wiki into the website clone, with analytics
    python trainworld/tools/wiki_site.py --preview DIR    same pages without analytics, for serve.py
    python trainworld/tools/wiki_site.py --rev <commit>   another version's wiki;  --head  the working tree's (unpublished)

The wiki describes the code as it is now, so the default is the wiki as committed at the publish of the latest
release: the last commit that touched trainworld/changelog/<newest>.json. Pages for features that are not
published yet stay out. Run it after every game publish (the publish skill). Never commits or pushes.

Reads wiki/read/*.md (figures filled), drops the "Exact rule:" and "In the code:" lines, renders with
wiki_build.render. Output is plain HTML with relative links; the website copy starts with an empty front
matter and includes the site's analytics.html, the one definition of its Google tag.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiki_build  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent          # trainworld/
REPO = ROOT.parent
SITE_SUBDIR = Path("projects") / "trainsworld" / "wiki"
MDLINK = re.compile(r"\[([^\]]+)\]\(([a-z0-9-]+)\.md\)")

CSS = """
body { margin: 0; background: #e0fadd; color: #24160a; font: 16px/1.55 Georgia, serif; }
.bar { border-top: 5px solid #204939; border-bottom: 1px solid #204939; padding: 10px 16px; font-size: 15px; }
.bar a { color: #24160a; }
main { max-width: 40em; margin: 0 auto; padding: 8px 16px 48px; }
h1 { font-size: 1.5em; font-weight: normal; margin: 1em 0 .4em; }
h2 { font-size: 1.15em; font-weight: normal; margin: 1.6em 0 .4em; padding-bottom: 2px; border-bottom: 1px solid #204939; }
a { color: #7a9221; text-decoration: none; }
a:hover { color: #4c5711; text-decoration: underline; }
li { margin: .35em 0; }
table { border-collapse: collapse; margin: 1em 0; display: block; overflow-x: auto; }
th, td { padding: 3px 14px 3px 0; text-align: left; font-weight: normal; vertical-align: top; }
th { border-bottom: 1px solid #204939; }
code { font: .85em Consolas, monospace; }
.foot { margin-top: 3em; font-size: 13px; opacity: .75; }
"""

TEMPLATE = """{front}<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="icon" type="image/png" sizes="32x32" href="/favicon.png">
<style>{css}</style>
{analytics}</head>
<body>
<div class="bar"><a href="/">anita.garden</a> / <a href="/projects/trainsworld/">trains world</a> / <a href="index.html">wiki</a></div>
<main>
{body}
<p class="foot">written by ai, for the live version of the game ({version}). <a href="/projects/trainsworld/">back to trains world</a></p>
</main>
</body>
</html>
"""


def git(*args):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        sys.exit(f"wiki_site: git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def live_release():
    """(version, commit) of the newest published release: its changelog file's last commit."""
    files = [p.stem for p in (ROOT / "changelog").glob("v*.json")]
    newest = max(files, key=lambda v: [int(x) for x in v[1:].split(".")])
    commit = git("log", "-1", "--format=%H", "--", f"trainworld/changelog/{newest}.json").strip()
    return newest, commit


def pages_at(rev):
    """{name: markdown} of wiki/read at a commit, or from the working tree when rev is None."""
    if rev is None:
        return {p.stem: p.read_text(encoding="utf-8") for p in (ROOT / "wiki" / "read").glob("*.md")}
    names = git("ls-tree", "--name-only", rev, "trainworld/wiki/read/").split()
    return {Path(n).stem: git("show", f"{rev}:{n}") for n in names if n.endswith(".md")}


def clean(name, md):
    out = []
    for line in md.replace("\r\n", "\n").split("\n"):
        if line.startswith((wiki_build.CODE_LINE, wiki_build.EXACT_LINE)):
            continue
        if name == "index" and line.startswith("How trips and track money work"):
            line = "How trips and track money work, in plain words."
        out.append(MDLINK.sub(lambda m: f"[[{m.group(2)}|{m.group(1)}]]", line))
    return "\n".join(out).rstrip() + "\n"


def default_website():
    p = REPO
    if ".tw-trees" in p.parts:
        p = Path(*p.parts[:p.parts.index(".tw-trees")])
    return p.parent / "website"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", metavar="DIR", help="write here without analytics instead of into the website clone")
    ap.add_argument("--website", metavar="DIR", help="the website clone (default: ../website of the maps root)")
    ap.add_argument("--rev", help="wiki as of this commit")
    ap.add_argument("--head", action="store_true", help="the working tree's wiki, unpublished features included")
    a = ap.parse_args()

    version, rev = live_release()
    if a.head:
        rev, version = None, "unpublished"
    elif a.rev:
        rev, version = a.rev, a.rev[:8]
    pages = pages_at(rev)
    if "index" not in pages or len(pages) < 5:
        sys.exit(f"wiki_site: no wiki/read pages at {rev or 'the working tree'}")

    if a.preview:
        out, analytics, front = Path(a.preview), "", ""
    else:
        site = Path(a.website) if a.website else default_website()
        if not (site / "_includes" / "analytics.html").is_file():
            sys.exit(f"wiki_site: {site} is not the website clone (no _includes/analytics.html)")
        out, analytics, front = site / SITE_SUBDIR, "{% include analytics.html %}\n", "---\n---\n"
    tmp = out.with_name(out.name + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, md in pages.items():
        md = clean(name, md)
        title = re.match(r"# (.*)", md).group(1)
        full = title if name == "index" else f"{title} (trains world wiki)"
        html = TEMPLATE.format(front=front, title=wiki_build.esc(full if name != "index" else "trains world wiki"),
                               css=CSS, analytics=analytics, body=wiki_build.render(md), version=version)
        if not a.preview and ("{{" in html.split("</style>", 1)[1] or "{%" in html.replace("{% include analytics.html %}", "")):
            sys.exit(f"wiki_site: {name}: Liquid-looking braces in a page; Jekyll would eat them")
        with open(tmp / f"{name}.html", "w", encoding="utf-8", newline="\n") as f:
            f.write(html)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"wiki_site: {len(pages)} pages of {version} ({(rev or 'working tree')[:8]}) written to {out}")


if __name__ == "__main__":
    main()
