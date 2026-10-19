"""Cheap checks for naming agents, instead of a web search.

    python asia1m/naming/lookup.py hits <term> [<context>]   zh-wiki pages containing the term
                                                             (and context, e.g. the province):
                                                             total and the top titles
    python asia1m/naming/lookup.py en <name>                 English Wikipedia: is there an
                                                             article (or redirect) by this name,
                                                             and what does a search turn up
    python asia1m/naming/lookup.py page <title>              lead and history/name/geography
                                                             sections of a zh-wiki article

Recognisability rule of thumb from `hits "灵州" 宁夏`: a name whose top titles are about this place
passes; one whose hits are about somewhere else, or that has a handful of hits, is weak.
"""
import sys, json, re, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8")
UA = {"User-Agent": "asia1m-naming/0.1 (personal map project)"}
import os, time, msvcrt, contextlib
LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lookup.lock")
@contextlib.contextmanager
def queued():
    """Several agents run lookups at once; Wikipedia answers that with 429. Queue every request
    across processes through one lock file and space them out."""
    f = open(LOCK, "a+")
    while True:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1); break
        except OSError:
            time.sleep(0.2)
    try:
        yield
        time.sleep(1.2)   # 0.6 s still drew 429s with four agents running
    finally:
        f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1); f.close()

def api(**p):
    import urllib.error
    u = "https://zh.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**p, "format": "json"})
    for wait in [3, 8, 20, 40, None]:
        try:
            with queued():
                return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30))
        except urllib.error.HTTPError as e:
            if e.code != 429 or wait is None: raise
            time.sleep(wait)
cmd = sys.argv[1]
if cmd == "hits":
    term, ctx = sys.argv[2], " ".join(sys.argv[3:])
    q = f'"{term}" {ctx}'.strip()
    r = api(action="query", list="search", srsearch=q, srlimit=8, srinfo="totalhits", srprop="snippet", variant="zh-cn")
    print(f'{q}: {r["query"]["searchinfo"]["totalhits"]} pages')
    for s in r["query"]["search"]:
        snip = re.sub(r"<.*?>", "", s["snippet"])[:90]
        print(f'- {s["title"]}: {snip}')
    exact = api(action="query", titles=term, redirects=1, prop="info", variant="zh-cn")
    p = next(iter(exact["query"]["pages"].values()))
    print("own article:", "no" if "missing" in p else p["title"])
elif cmd == "en":
    # can an English reader look the name up? exact English Wikipedia title (with redirects), then search
    import urllib.request as ur
    def en(**p):
        u = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**p, "format": "json"})
        return json.load(ur.urlopen(ur.Request(u, headers=UA), timeout=30))
    term = " ".join(sys.argv[2:])
    r = en(action="query", titles=term, redirects=1, prop="extracts", exintro=1, explaintext=1, exsentences=2)
    p = next(iter(r["query"]["pages"].values()))
    print("en article:", "none" if "missing" in p else f'{p["title"]}: {p.get("extract", "")[:200]}')
    s = en(action="query", list="search", srsearch=term, srlimit=5)
    for x in s["query"]["search"]:
        print(f'- {x["title"]}: {re.sub(r"<.*?>", "", x["snippet"])[:90]}')
elif cmd == "page":
    r = api(action="query", titles=sys.argv[2], redirects=1, prop="extracts", explaintext=1, variant="zh-cn")
    p = next(iter(r["query"]["pages"].values()))
    t = p.get("extract", "")
    keep = re.compile(r"历史|沿革|名称|得名|由来|地理|河流|水文|山")
    parts = re.split(r"\n(==+ .*? ==+)\n", t)
    print(p.get("title"), "\n", parts[0][:800])
    for i in range(1, len(parts) - 1, 2):
        if keep.search(parts[i]): print(parts[i], parts[i + 1].strip()[:1200])
else:
    sys.exit(__doc__)
