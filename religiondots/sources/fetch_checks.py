"""Shared download, PDF and second-copy checks for census-table loaders.

`playbooks/census_table.md` lists the traps these catch. Import them; do not copy them. A
loader run as `python sources/<cc>.py` has `sources/` on its path, so `from fetch_checks import
check_body` works as it stands.

WHY ONE MODULE. Each census loader carried its own copy of the same checks (`sources/gw.py`,
`gn.py`, `cg.py`, `td.py`), and the copies had drifted into two rules that contradict each other
on a real file:

  * `gw.py`, `gn.py` and `td.py` refused any PDF without `%%EOF` in its last 2 KB;
  * `cg.py` pinned the exact size instead, because Congo's complete Word 2007 export keeps
    93 KB of free-object xref after its last `%%EOF`, and the trailer rule rejects it.

Both were right about their own file. The rule that holds for both (spec §12 "A TRUNCATED
WAYBACK CAPTURE OPENS AS THE WHOLE DOCUMENT"): **a pinned digest or size is the verdict**, and
without a pin a missing trailer is a reason to LOOK, not a verdict. The download is still
refused until someone has looked, because a 1 MiB fragment of Guinea-Bissau's volume opened in
MuPDF as all 92 pages: opening is never the test.

What is here:

    digest(body)                    SHA-1 in base32, the form the Wayback CDX `digest` field uses
    kind_of(body), MAGIC            what the first bytes say the file is
    check_body(body, kind, ...)     a 200 is not the file: size, magic bytes, pins, trailer
    trailer_report(body)            where the `%%EOF` markers are and what follows the last
    check_pdf_doc(doc, pages)       page count and a text layer on every page, as say() pairs
    image_pages(doc, pnos)          pages whose table is a picture under a text caption
    cdx_url, parse_cdx,             the Wayback CDX: query, answer, one capture per digest,
      one_per_digest, wayback_raw     and the untouched bytes of a capture
    wp_json(body, headers)          one page of a WordPress REST listing, checked
    pinned_differences(...)         two copies of one table that differ by a recorded amount

Nothing here fetches. Every check either returns a value or raises `FetchCheckError` (a
SystemExit) with a message saying what to look at; none changes the bytes it is given.
"""

import base64
import hashlib
import json
import re

TRAILER_WINDOW = 2048      # where `%%EOF` sits in an ordinary PDF: its last 2 KB
BOT_WALL_BYTES = 1024      # a 200 under a kilobyte is a bot wall or a parking page

# The first bytes of each kind of file a statistics office serves. `xml` is SpreadsheetML (an
# Excel 2003 XML workbook, often saved as .xls): read it with xml.etree and honour `ss:Index`.
MAGIC = {
    "pdf": b"%PDF-",
    "zip": b"PK\x03\x04",          # xlsx, docx and plain zip
    "xls": b"\xd0\xcf\x11\xe0",    # OLE2: xls and doc
    "xml": b"<?xm",
}
KIND_ALIASES = {"xlsx": "zip", "docx": "zip", "doc": "xls"}


class FetchCheckError(SystemExit):
    """A check failed. A SystemExit, so a loader that does not catch it stops with the message."""


def digest(body):
    """SHA-1 of `body` in base32, the form the Wayback CDX writes in its `digest` field."""
    return base64.b32encode(hashlib.sha1(body).digest()).decode()


def kind_of(body):
    """'pdf', 'zip', 'xls' or 'xml' from the first bytes, else None."""
    for kind, head in MAGIC.items():
        if body[:len(head)] == head:
            return kind
    return None


def has_trailer(body, window=TRAILER_WINDOW):
    return b"%%EOF" in body[-window:]


def trailer_report(body):
    """Where the `%%EOF` markers are and what follows the last one, in one line."""
    marks = [m.start() for m in re.finditer(rb"%%EOF", body)]
    if not marks:
        return f"no %%EOF anywhere in its {len(body):,} bytes"
    after = body[marks[-1] + len(b"%%EOF"):]
    shown = after.strip()[:32]
    return (f"{len(marks)} %%EOF marker(s), the last at byte {marks[-1]:,} and followed by "
            f"{len(after):,} bytes" + (f" starting {shown!r}" if shown else ""))


def check_body(body, kind="pdf", *, where="", pin_size=None, pin_digest=None, forbid=(),
               min_bytes=BOT_WALL_BYTES):
    """Raise FetchCheckError unless `body` is the file; return a one-line description if it is.

    In order: a body under `min_bytes` (a bot wall or parking page answers 200 with a few
    hundred bytes); the magic bytes for `kind`; each `forbid` substring, case-insensitively
    (Congo's squatted domain serves the real brochure with `http` spam links injected); then
    the pins.

    A pinned DIGEST or SIZE is the verdict, and the trailer is not consulted when there is one:
    a complete Word export can keep bytes after `%%EOF` (Congo), and a truncated capture cannot
    match a pinned SHA-1. With NO pin, a PDF without `%%EOF` in its last 2 KB is refused with
    `trailer_report()`, because unverified bytes must not be written: open it, compare the page
    count with the document's own table list, check text on every page, then pin its size or
    digest and fetch again.
    """
    what = f"{where}: " if where else ""
    if len(body) < min_bytes:
        raise FetchCheckError(
            f"{what}only {len(body):,} bytes, starting {body[:40]!r}. A bot wall or a parking "
            "page answers 200 with this little; try a second client once, then give Anita the "
            "URL rather than iterating on headers")
    head = MAGIC[KIND_ALIASES.get(kind, kind)]
    if body[:len(head)] != head:
        raise FetchCheckError(
            f"{what}not a {kind} file: {len(body):,} bytes starting {body[:16]!r}. A 200 is not "
            "the file; an HTML start is a redirect, error or challenge page saved under the "
            "file's name")
    low = body.lower()
    for bad in forbid:
        if bad.lower() in low:
            raise FetchCheckError(
                f"{what}contains {bad!r}, which the real file does not: a squatted or rewritten "
                "copy (cnsee.org serves Congo's brochure with spam links injected)")
    if pin_digest is not None and digest(body) != pin_digest:
        raise FetchCheckError(
            f"{what}digest {digest(body)} is not the pinned {pin_digest} ({len(body):,} bytes; "
            f"{trailer_report(body)}). A truncated capture (the Wayback Machine has served the "
            "first 1 MiB of a file under a digest of its own) or a re-issued file: check every "
            "transcription against it before re-pinning")
    if pin_size is not None and len(body) != pin_size:
        raise FetchCheckError(
            f"{what}{len(body):,} bytes, not the pinned {pin_size:,} ({trailer_report(body)}). "
            "Truncated or re-issued: open it and check the pages and every transcription "
            "before re-pinning")
    pins = " and ".join(p for p, v in (("digest", pin_digest), ("size", pin_size))
                        if v is not None)
    if kind == "pdf" and not has_trailer(body):
        if not pins:
            raise FetchCheckError(
                f"{what}no %%EOF in the last {TRAILER_WINDOW // 1024} KB: {trailer_report(body)}. "
                "Not a verdict on its own: a truncated capture opens as the whole document, and "
                "a complete Word export can keep bytes after %%EOF (Congo, 93 KB). Look: open "
                "it, compare the page count with the document's own table list, check text on "
                "every page, then pin its size or digest")
        return (f"{len(body):,} bytes, matches its pinned {pins}; no %%EOF in the last 2 KB "
                f"({trailer_report(body)}), accepted on the pin")
    return f"{len(body):,} bytes, " + (f"matches its pinned {pins}" if pins else "not pinned")


def check_pdf_doc(doc, pages=None):
    """[(ok, message)] for the page count and a text layer on every page; feed each to say().

    Opening is not a test: a 1 MiB fragment of Guinea-Bissau's volume opened as all 92 pages.
    An empty page, or a page count other than the one the document's own table list implies,
    is. Only for a PDF with a text layer; a scanned volume fails the second on purpose.
    """
    out = []
    if pages is not None:
        out.append((doc.page_count == pages,
                    f"the document is {doc.page_count} pages (expected {pages})"))
    empty = [i for i in range(doc.page_count) if not doc.load_page(i).get_text().strip()]
    out.append((not empty, f"every page has a text layer (empty page indices: {empty})"))
    return out


NUMBER = re.compile(r"\d+(?:[ .,  ]\d{3})*(?:[.,]\d+)?")


def image_pages(doc, pnos=None, min_numbers=20):
    """Pages holding pictures and too few numbers in their text to be a table read as text.

    spec §12 "A PDF CAN HAVE A TEXT LAYER FOR ITS PROSE AND PICTURES FOR ITS TABLES": the
    caption is searchable while the numbers are an image, so a text search says the volume has
    no table. Returns [(page index, images on the page, numbers in its text)]. Call it on the
    pages a table should be on before concluding it is absent; render those pages, transcribe,
    and let the table's own arithmetic check the transcription.
    """
    out = []
    for i in (range(doc.page_count) if pnos is None else pnos):
        page = doc.load_page(i)
        images = len(page.get_images(full=True))
        numbers = len(NUMBER.findall(page.get_text()))
        if images and numbers < min_numbers:
            out.append((i, images, numbers))
    return out


# ---------------------------------------------------------------- the Wayback Machine

CDX = "https://web.archive.org/cdx/search/cdx"
CDX_FIELDS = ("original", "timestamp", "statuscode", "digest", "length")


def cdx_url(url, match_type="prefix", fields=CDX_FIELDS):
    """The CDX query for every capture under `url` (spec §12 "A PORTAL MIGRATION HIDES FILES").

    Ask for the OLD path prefix, not the new portal, or `match_type="domain"` for a whole
    government domain. https only. No `filter=`: a bad filter answers 500 with nothing in it,
    so fetch the prefix whole and grep it locally with parse_cdx().
    """
    from urllib.parse import urlencode

    if match_type not in ("exact", "prefix", "host", "domain"):
        raise ValueError(f"match_type {match_type!r}")
    return CDX + "?" + urlencode({"url": url, "matchType": match_type, "fl": ",".join(fields)})


def parse_cdx(text, fields=CDX_FIELDS):
    """A CDX answer as a list of dicts, one per capture; [] for an empty answer.

    Raises on anything that is not CDX lines of `fields`: an HTML page is an error or a
    throttled answer (the CDX answers 503 in bursts, so retry before calling a prefix empty).
    """
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(" ")
        if line.lstrip().startswith("<") or len(parts) != len(fields):
            raise FetchCheckError(
                f"CDX line {n} is not the {len(fields)} fields {', '.join(fields)}: "
                f"{line[:80]!r}. An HTML page is an error or a throttled answer; retry")
        rows.append(dict(zip(fields, parts)))
    return rows


def one_per_digest(rows):
    """{digest: the latest status-200 capture with that digest}.

    One URL with two digests has one bad copy (Guinea-Bissau: the second digest was the first
    1 MiB of the file), so fetch one capture per digest and let check_body() decide.
    """
    out = {}
    for r in rows:
        if r.get("statuscode", "200") != "200":
            continue
        d = r["digest"]
        if d not in out or r["timestamp"] > out[d]["timestamp"]:
            out[d] = r
    return out


def wayback_raw(timestamp, url):
    """The archived bytes, untouched: `id_` stops the archive rewriting links or adding a frame."""
    return f"https://web.archive.org/web/{timestamp}id_/{url}"


# ---------------------------------------------------------------- WordPress

def wp_json(body, headers=None, *, where=""):
    """One page of a WordPress REST listing (`wp/v2/media`, `wp/v2/search`): (items, total_pages).

    spec §12 "A WP FILE DOWNLOAD INSTALL IS INVISIBLE IN `wp-json/`". Raises on an HTML page (a
    bot wall, or a theme answering 200 for a route this site does not serve), on WordPress's
    error object, and on anything but a list. `total_pages` is `X-WP-TotalPages`, None when the
    header is absent: page to it in full. The media index lists only what registered with it,
    never all of `wp-content/uploads` (Pakistan), so an empty search is not a negative.
    """
    what = f"{where}: " if where else ""
    text = body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else body
    if text.lstrip()[:1] == "<":
        raise FetchCheckError(
            f"{what}an HTML page, not JSON ({text.strip()[:60]!r}). A bot wall, or no REST route "
            "at this path: try `index.php?rest_route=/wp/v2/...` instead of `/wp-json/`")
    try:
        data = json.loads(text)
    except ValueError as e:
        raise FetchCheckError(f"{what}not JSON ({e}): {text[:80]!r}") from None
    if isinstance(data, dict) and "code" in data:
        raise FetchCheckError(
            f"{what}WordPress error {data.get('code')!r}: {data.get('message', '')}. "
            "`rest_no_route`: try `index.php?rest_route=`. A closed listing often sits beside "
            "an open `/download/<int>` route naming files in Content-Disposition")
    if not isinstance(data, list):
        raise FetchCheckError(f"{what}a JSON {type(data).__name__}, not a list of items")
    total = None
    if headers is not None:
        low = {str(k).lower(): v for k, v in headers.items()}
        if "x-wp-totalpages" in low:
            total = int(low["x-wp-totalpages"])
    return data, total


# ---------------------------------------------------------------- a second copy of the table

def pinned_differences(ours, theirs, expected, *, what):
    """Assert two copies of one table differ by exactly the recorded amounts; return one line.

    For a disagreement that is known and explained, such as Mozambique's national Quadro 11
    against UNSD table 28, a different edit of the same census (sources/mz.py). Printing the
    difference is not a check: a re-issued file or an edited second copy moves it and nothing
    fails. Raises naming every category that moved.
    """
    keys = set(expected)
    if set(ours) != keys or set(theirs) != keys:
        raise FetchCheckError(
            f"{what}: categories {sorted(set(ours) | set(theirs))} are not the pinned "
            f"{sorted(keys)}")
    got = {k: ours[k] - theirs[k] for k in expected}
    moved = [f"{k} {got[k]:+,} (pinned {v:+,})" for k, v in expected.items() if got[k] != v]
    if moved:
        raise FetchCheckError(
            f"{what}: the recorded difference moved: " + "; ".join(moved) + ". A re-issued "
            "file or an edited second copy; find which before re-pinning, and say why")
    return f"{what} (pinned): " + ", ".join(f"{k} {got[k]:+,}" for k in expected)
