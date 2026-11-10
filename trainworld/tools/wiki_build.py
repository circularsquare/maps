#!/usr/bin/env python3
"""Build trainworld/wiki/*.md into trainworld/wiki/html/, filling figures from the source.

    python trainworld/tools/wiki_build.py           write read/ (tracked, figures filled) and html/
    python trainworld/tools/wiki_build.py --check   write nothing, exit 1 on any problem or a stale read/
    python trainworld/tools/wiki_build.py --html    write html/ only (snapbuild; leaves tracked files alone)

Nothing is written if anything is wrong.

A marker is {{path:EXPR}} or {{path:EXPR|decimals}}, path relative to trainworld/. EXPR is arithmetic
over the file's names: CONST, CONST[2], Struct.field (the field's value in `impl Default for Struct`)
or Obj.key (a key of a TS `const Obj = {...}`), with + - * / ** ( ) numbers, ln(), sqrt(), max(), min().
Rules for pages: wiki/RULES.md.
"""
import ast
import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "wiki"
OUT = WIKI / "html"
SKIP = {"RULES.md"}

MARK = re.compile(r"\{\{([^{}:]+):([^{}|]+?)(?:\|(\d+))?\}\}")
LINK = re.compile(r"\[\[([a-z0-9-]+)(?:\|([^\]]+))?\]\]")
CODE_LINE = "In the code:"
EXACT_LINE = "Exact rule:"
READ = WIKI / "read"


class Bad(Exception):
    pass


_texts = {}
_refs = {}


def text_of(path):
    if path not in _texts:
        f = ROOT / path
        if not f.is_file():
            raise Bad(f"no such file: {path}")
        _texts[path] = re.sub(r"/\*.*?\*/", "", f.read_text(encoding="utf-8"), flags=re.S)
    return _texts[path]


def strip_comment(s):
    return re.sub(r"//[^\n]*", "", s)


def until_semicolon(text, start):
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return text[start:i]
    raise Bad("constant has no closing semicolon")


def const_value(path, name):
    """The right-hand side of `const NAME`, in Rust or TS. One definition only."""
    head = re.compile(r"^[ \t]*(?:export\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+" + re.escape(name) + r"\b[^=\n]*=", re.M)
    hits = list(head.finditer(text_of(path)))
    if not hits:
        return None
    if len(hits) > 1:
        raise Bad(f"{name} is defined {len(hits)} times in {path}")
    return strip_comment(until_semicolon(text_of(path), hits[0].end())).strip()


def balanced(text, start):
    """The text inside the brace opening at or after `start`."""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    raise Bad("unbalanced braces")


def field_value(path, owner, field):
    """`field` in `impl Default for owner`, or a key of a TS object `const owner = {...}`."""
    text = text_of(path)
    m = re.search(r"impl\s+Default\s+for\s+" + re.escape(owner) + r"\b", text)
    if m:
        body = balanced(text, m.end())
    else:
        v = const_value(path, owner)
        if v is None or not v.lstrip().startswith("{"):
            raise Bad(f"no `impl Default for {owner}` or object `const {owner}` in {path}")
        body = v
    hits = re.findall(r"^[ \t]*\"?" + re.escape(field) + r"\"?\s*:\s*(.+?),?[ \t]*(?://[^\n]*)?$", body, re.M)
    hits = [h for h in hits if not re.fullmatch(r"\s*(?:f32|f64|i32|u32|u8|usize|bool|Option<.*)\s*", h)]
    if not hits:
        raise Bad(f"{owner}.{field} has no default in {path}")
    if len(hits) > 1:
        raise Bad(f"{owner}.{field} is set {len(hits)} times in {path}")
    return strip_comment(hits[0]).strip()


def module_file(path, ref):
    """The file a Rust path like super::api or crate::track::params names, seen from `path`; None if unknown."""
    if not path.endswith(".rs") or not path.startswith("sim/src/"):
        return None
    mod = path[len("sim/src/"):-3].split("/")
    if mod[-1] in ("mod", "lib"):
        mod.pop()
    for seg in ref.split("::"):
        if seg == "crate":
            mod = []
        elif seg == "super":
            mod = mod[:-1]
        elif seg != "self":
            mod = mod + [seg]
    for cand in ("/".join(mod) + ".rs", "/".join(mod) + "/mod.rs"):
        if (ROOT / "sim/src" / cand).is_file():
            return "sim/src/" + cand
    return None


def global_const(name, path):
    """A const not in this file: the one definition under sim/src (Rust) or app/src (TS)."""
    found = []
    rust = path.endswith(".rs")
    base, glob = ("sim/src", "**/*.rs") if rust else ("app/src", "**/*.ts")
    for f in (ROOT / base).glob(glob):
        rel = f.relative_to(ROOT).as_posix()
        v = const_value(rel, name)
        if v is not None:
            found.append((rel, v))
    if len(found) != 1:
        where = ", ".join(r for r, _ in found) or "nowhere"
        raise Bad(f"{name}: expected one definition, found {where}")
    return found[0]


def to_python(src):
    """A Rust or TS value expression as Python source."""
    s = src.strip()
    if "::<" in s or re.search(r"\.\s*[a-z_]+\(", s) or s.startswith("{") or '"' in s:
        raise Bad(f"cannot evaluate `{s}`")
    s = re.sub(r"\bas\s+(?:f32|f64|i32|i64|u32|u64|u8|usize)\b", "", s)
    s = re.sub(r"(\d)_(?=\d)", r"\1", s)
    s = re.sub(r"(\d)_?(?:f32|f64|i32|i64|u32|u64|u8|usize)\b", r"\1", s)

    def path_ref(m):
        token = f"__ref{len(_refs)}"
        _refs[token] = (m.group(1), m.group(2))
        return token
    return re.sub(r"\b((?:(?:crate|super|self|[a-z_][a-z0-9_]*)::)*(?:crate|super|self|[a-z_][a-z0-9_]*))::([A-Za-z_][A-Za-z0-9_]*)", path_ref, s)


_json = {}


def json_value(path, dotted):
    """A number or list of numbers in a JSON file, by dotted key path (list indexes are [i] in the marker)."""
    if path not in _json:
        f = ROOT / path
        if not f.is_file():
            raise Bad(f"no such file: {path}")
        _json[path] = json.loads(f.read_text(encoding="utf-8"))
    v = _json[path]
    for k in dotted.split("."):
        if not isinstance(v, dict) or k not in v:
            raise Bad(f"no key {dotted} in {path}")
        v = v[k]
    if isinstance(v, bool) or not (isinstance(v, (int, float)) or (isinstance(v, list) and all(isinstance(x, (int, float)) for x in v))):
        raise Bad(f"{dotted} in {path} is not a number or a list of numbers")
    return v


def evaluate(expr, path, stack=()):
    if path.endswith(".json"):
        def key(m):
            if m.group(0) in ("ln", "sqrt", "max", "min") and expr[m.end():m.end() + 1] == "(":
                return m.group(0)
            token = f"__ref{len(_refs)}"
            _refs[token] = ("json", m.group(0))
            return token
        expr = re.sub(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", key, expr)
    try:
        tree = ast.parse(to_python(expr), mode="eval").body
    except SyntaxError:
        raise Bad(f"cannot parse `{expr}`")
    return walk(tree, path, stack)


def name_value(name, path, stack):
    target = path
    if name in _refs:
        ref, name = _refs[name]
        if ref == "json":
            return json_value(path, name)
        target = module_file(path, ref)
        if target is None:
            target, v = global_const(name, path)
            return evaluate(v, target, stack + ((target, name),))
    if (target, name) in stack:
        raise Bad(f"{name} refers to itself")
    v = const_value(target, name)
    if v is None:
        target, v = global_const(name, path)
    return evaluate(v, target, stack + ((target, name),))


def walk(n, path, stack):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.Name):
        return name_value(n.id, path, stack)
    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
        v = field_value(path, n.value.id, n.attr)
        return evaluate(v, path, stack + ((path, f"{n.value.id}.{n.attr}"),))
    if isinstance(n, (ast.List, ast.Tuple)):
        return [walk(e, path, stack) for e in n.elts]
    if isinstance(n, ast.Subscript):
        i = walk(n.slice, path, stack)
        seq = walk(n.value, path, stack)
        if not isinstance(seq, list) or not isinstance(i, int) or not 0 <= i < len(seq):
            raise Bad("bad index")
        return seq[i]
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -walk(n.operand, path, stack)
    if isinstance(n, ast.BinOp):
        a, b = walk(n.left, path, stack), walk(n.right, path, stack)
        if isinstance(a, list) or isinstance(b, list):
            raise Bad("arithmetic on an array")
        if isinstance(n.op, ast.Add):
            return a + b
        if isinstance(n.op, ast.Sub):
            return a - b
        if isinstance(n.op, ast.Mult):
            return a * b
        if isinstance(n.op, ast.Div):
            return a / b
        if isinstance(n.op, ast.Pow):
            return a ** b
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("ln", "sqrt", "max", "min") and len(n.args) == 1:
        v = walk(n.args[0], path, stack)
        if n.func.id in ("max", "min"):
            return max(v) if n.func.id == "max" else min(v)
        return math.log(v) if n.func.id == "ln" else math.sqrt(v)
    raise Bad(f"cannot evaluate `{ast.dump(n)[:60]}`")


def fmt(v, decimals):
    if isinstance(v, list) or isinstance(v, bool) or not isinstance(v, (int, float)):
        raise Bad("a marker must come to one number")
    if decimals is not None:
        s = f"{v:,.{decimals}f}"
    else:
        s = f"{v:,.2f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    return s


def expand(body, page, problems):
    def one(m):
        path, expr, dec = m.group(1).strip(), m.group(2).strip(), m.group(3)
        try:
            return fmt(evaluate(expr, path), int(dec) if dec else None)
        except Bad as e:
            problems.append(f"{page}: {m.group(0)}: {e}")
        except (ArithmeticError, TypeError, ValueError) as e:
            problems.append(f"{page}: {m.group(0)}: {type(e).__name__}: {e}")
        return "?"
    return MARK.sub(one, body)


def check_refs(line, page, problems):
    """A code span with a / is a path; one without must appear in the path named before it."""
    current = None
    for span in re.findall(r"`([^`]+)`", line):
        if "/" in span:
            if not (ROOT / span).exists():
                problems.append(f"{page}: {span} does not exist")
                current = None
            else:
                current = span
        elif current:
            ident = span.split("::")[-1]
            if not re.search(r"\b" + re.escape(ident) + r"\b", (ROOT / current).read_text(encoding="utf-8")):
                problems.append(f"{page}: `{span}` is not in {current}")
        else:
            problems.append(f"{page}: `{span}` comes before any file")


def check_headings(line, page, problems):
    """Each "quoted heading" after a doc path must be a heading of that doc."""
    for path, rest in re.findall(r"`(docs/[^`]+)`((?:\s*\"[^\"]+\")+)", line):
        f = ROOT / path
        if not f.is_file():
            continue
        heads = [l for l in f.read_text(encoding="utf-8").split("\n") if l.startswith("#")]
        for head in re.findall(r"\"([^\"]+)\"", rest):
            if not any(head in h for h in heads):
                problems.append(f"{page}: {path} has no heading \"{head}\"")


def lint(body, page, pages, problems):
    lines = body.split("\n")
    if page != "index.md":
        for prefix in (EXACT_LINE, CODE_LINE):
            found = [l for l in lines if l.startswith(prefix)]
            if len(found) != 1:
                problems.append(f"{page}: needs exactly one '{prefix}' line, has {len(found)}")
            for l in found:
                check_refs(l, page, problems)
                check_headings(l, page, problems)
    if not lines[0].startswith("# "):
        problems.append(f"{page}: first line must be a # title")
    for target, _ in LINK.findall(body):
        if target not in pages:
            problems.append(f"{page}: link to missing page [[{target}]]")
    for i, l in enumerate(lines, 1):
        if l.startswith(CODE_LINE):
            continue
        rest = MARK.sub("", l)
        rest = re.sub(r"`[^`]*`", "", rest)
        if l.startswith(EXACT_LINE):
            rest = re.sub(r"\"[^\"]*\"", "", rest)
        rest = LINK.sub("", rest)
        for d in re.findall(r"\S*\d\S*", rest):
            problems.append(f"{page}:{i}: hand-written number '{d}': use a marker or say it in words")
        if re.search(r"—|–| - [a-z]", rest):
            problems.append(f"{page}:{i}: dash between phrases")
        for tail in re.findall(r", (?:making|ensuring|highlighting|underscoring|allowing|helping|letting|giving)\b", rest):
            problems.append(f"{page}:{i}: tail '{tail}'")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)

    def link(m):
        name = m.group(1)
        return f'<a href="{name}.html">{m.group(2) or name.replace("-", " ")}</a>'
    return LINK.sub(link, s)


def render(md):
    out, para, items, table = [], [], [], []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
        if table:
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in table if not re.fullmatch(r"[\s|:-]+", r)]
            head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            out.append(f"<table><tr>{head}</tr>{body}</table>")
            table.clear()

    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            flush()
            n = len(s) - len(s.lstrip("#"))
            out.append(f"<h{n}>{inline(s[n:].strip())}</h{n}>")
        elif s.startswith("- "):
            if para or table:
                flush()
            items.append(s[2:])
        elif s.startswith("|"):
            if para or items:
                flush()
            table.append(s)
        elif not s:
            flush()
        elif s.startswith((CODE_LINE, EXACT_LINE)):
            flush()
            out.append(f'<p class="code">{inline(s)}</p>')
        elif items:
            items[-1] += " " + s
        else:
            para.append(s)
    flush()
    return "\n".join(out)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ background: #fbfaf7; color: #222; font: 17px/1.55 Georgia, serif; margin: 0 auto; max-width: 40em; padding: 1.5em 16px 4em; }}
h1 {{ font-size: 1.6em; margin-bottom: .3em; }}
h2 {{ font-size: 1.15em; margin-top: 1.6em; }}
a {{ color: #15609e; }}
nav {{ font-size: .85em; margin-bottom: 1.5em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ padding: .2em 1em .2em 0; text-align: left; font-weight: normal; }}
th {{ border-bottom: 1px solid #ccc; }}
code {{ font: .85em Consolas, monospace; }}
.code {{ color: #777; font-size: .85em; margin-top: 2.5em; border-top: 1px solid #e5e2da; padding-top: .8em; }}
</style>
</head>
<body>
{nav}
{body}
</body>
</html>
"""


def read_markdown(md):
    """A page as agents read it: figures already filled, links as ordinary markdown."""
    def link(m):
        name = m.group(1)
        return f"[{m.group(2) or name.replace('-', ' ')}]({name}.md)"
    return LINK.sub(link, md)


def stale_read(read):
    """Files in read/ that differ from what a build would write, or should not be there."""
    have = {p.name for p in READ.glob("*.md")} if READ.is_dir() else set()
    stale = sorted(have ^ set(read))
    for name in sorted(have & set(read)):
        if (READ / name).read_text(encoding="utf-8").replace("\r\n", "\n") != read[name]:
            stale.append(name)
    return stale


def write_read(read):
    READ.mkdir(exist_ok=True)
    for p in READ.glob("*.md"):
        if p.name not in read:
            p.unlink()
    for name in stale_read(read):
        if name in read:
            with open(READ / name, "w", encoding="utf-8", newline="\n") as f:
                f.write(read[name])


def main():
    check_only = "--check" in sys.argv[1:]
    files = sorted(p for p in WIKI.glob("*.md") if p.name not in SKIP)
    pages = {p.stem for p in files}
    if "index" not in pages:
        sys.exit("wiki_build: wiki/index.md is missing")
    problems, built, linked_from = [], {}, {}
    for p in files:
        src = p.read_text(encoding="utf-8")
        for target, _ in LINK.findall(src):
            linked_from.setdefault(target, set()).add(p.stem)
        lint(src, p.name, pages, problems)
        built[p.stem] = expand(src, p.name, problems)
    for name in sorted(pages - {"index"}):
        if "index" not in linked_from.get(name, set()):
            problems.append(f"{name}.md: not linked from index.md")
    markers = sum(len(MARK.findall(p.read_text(encoding="utf-8"))) for p in files)
    if problems:
        print(f"wiki_build: FAILED, {len(problems)} problem(s), nothing written:", file=sys.stderr)
        for pr in problems:
            print("  " + pr, file=sys.stderr)
        sys.exit(1)
    read = {f"{name}.md": read_markdown(md) for name, md in built.items()}
    if check_only:
        stale = stale_read(read)
        if stale:
            print("wiki_build: FAILED, wiki/read/ is out of date (run python trainworld/tools/wiki_build.py): " + ", ".join(stale), file=sys.stderr)
            sys.exit(1)
        print(f"wiki_build: OK, {len(files)} pages, {markers} figures")
        return
    if "--html" not in sys.argv[1:]:
        write_read(read)
    tmp = WIKI / "html.new"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for name, md in built.items():
        title = re.match(r"# (.*)", md).group(1)
        nav = "" if name == "index" else '<nav><a href="index.html">All pages</a></nav>'
        (tmp / f"{name}.html").write_text(TEMPLATE.format(title=esc(title), nav=nav, body=render(md)), encoding="utf-8")
    if OUT.exists():
        shutil.rmtree(OUT)
    tmp.rename(OUT)
    print(f"wiki_build: wrote {len(built)} pages to {OUT}, {markers} figures")


if __name__ == "__main__":
    main()
