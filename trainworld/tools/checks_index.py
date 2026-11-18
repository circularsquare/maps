"""Writes trainworld/checks/INDEX.md: one line per file in checks/ (and its lib/, fixtures/, ceiling/), sorted by name.
Columns: name | kind | smoke | runs since SINCE | last commit | what it proves (header comment's first sentence).
Runs come from a scan of the Claude transcripts for `node|python ... checks/<file>` commands (about 20 s).
    python trainworld/tools/checks_index.py [since YYYY-MM-DD]      (default 2026-11-01)
Known-broken checks are marked in BROKEN below; edit it when one is fixed or newly found."""
import os, re, sys, json, glob, subprocess, collections
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
CH = os.path.join(TW, 'checks')
REPO = os.path.dirname(TW)
SINCE = sys.argv[1] if len(sys.argv) > 1 else '2026-11-01'

# file -> why it fails or cannot finish (do not list a check that merely needs a game running)
BROKEN = {
    't2626-profit-estimate.mjs': 'too slow: did not finish in 280 s on 3 cores (2026-11-18)',
    't1201-fare-rows.mjs': 'Money tab selector fixed 2026-11-18, still fails 8 of 11 lines: the fare rows moved (now Base fare, two stretches, Transfer fare, country scale) and its row indices and buttons are stale',
}
# helpers a check imports, or data a check reads: not checks themselves
HELPERS = {'t2407-draw.py', 't1177-data.mjs', 't1966-layout.mjs', 'lib.mjs', 'static-serve.mjs', 'token_usage.py'}
FIRST_LINE = {}  # file -> text, for files with no header comment

def kind_of(rel):
    b = os.path.basename(rel)
    if rel.startswith('lib/'): return 'lib'
    if rel.startswith(('fixtures/', 'ceiling/')) or not b.endswith(('.mjs', '.py', '.cmd')): return 'fixture'
    if b in HELPERS: return 'helper'
    if re.match(r't\d', b): return 'check'
    return 'tool'

def header(path):
    try: text = open(path, encoding='utf-8', errors='replace').read(6000)
    except Exception: return ''
    if not path.endswith(('.mjs', '.py', '.cmd', '.rs')): return ''
    lines = text.splitlines()
    out = []
    if path.endswith('.py'):
        m = re.match(r'\s*(?:#[^\n]*\n)*\s*[rbu]?("""|\'\'\')(.*?)\1', text, re.S)
        if m: return ' '.join(m.group(2).split())
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith('#!') or (not s and not out): continue
        if s.startswith('//') or s.startswith('#') or s.upper().startswith('REM '):
            out.append(re.sub(r'^(//+|#+|REM)\s?', '', s, flags=re.I).strip())
        elif s.startswith('/*') or s.startswith('*'):
            out.append(s.strip('/* '))
        elif not out and i < 12 and re.match(r'(import|const|let|var|from)(?![a-z])', s): continue
        else: break
    return ' '.join(x for x in out if x)

def summary(txt):
    txt = re.sub(r'^(T-\d+(?:[,/ ]+(?:and )?T-\d+)*|t\d+)\s*[:,-]?\s*', '', txt.strip())
    txt = re.sub(r'^\(T-\d+[^)]*\)\s*:?\s*', '', txt)
    m = re.search(r'\.(\s|$)', txt)
    s = txt[:m.start()] if m else txt
    s = s.replace('|', '/')
    if len(s) > 100: s = s[:97].rsplit(' ', 1)[0] + '...'
    return s or '(no header comment)'

# files
files = []
for root, dirs, fs in os.walk(CH):
    rel_root = os.path.relpath(root, CH).replace('\\', '/')
    if rel_root == 'old' or rel_root.startswith(('old/', '__pycache__')):
        dirs[:] = []; continue
    dirs[:] = [d for d in dirs if d not in ('old', '__pycache__')]
    for f in fs:
        if f == 'INDEX.md' or f.endswith('.pyc'): continue
        files.append(f if rel_root == '.' else rel_root + '/' + f)
files.sort(key=str.lower)
n_old = len(os.listdir(os.path.join(CH, 'old'))) if os.path.isdir(os.path.join(CH, 'old')) else 0

smoke_src = open(os.path.join(CH, 'smoke.mjs'), encoding='utf-8').read() if os.path.exists(os.path.join(CH, 'smoke.mjs')) else ''
in_smoke = {f for f in files if re.search(r'[\'"/\\]' + re.escape(os.path.basename(f)) + r'[\'"]', smoke_src)}

# last commit date per file, from one git log
log = subprocess.run(['git', '-C', REPO, 'log', '--format=@%cs', '--name-only', '--', 'trainworld/checks'],
                     capture_output=True, text=True, encoding='utf-8').stdout
last = {}; cur = None
for ln in log.splitlines():
    if ln.startswith('@'): cur = ln[1:]
    elif ln.startswith('trainworld/checks/'): last.setdefault(ln[len('trainworld/checks/'):], cur)

# runs since SINCE, from transcripts (the main tree's project folder holds every session's)
RUN = re.compile(r'(?:node|python(?:\.exe)?|\.cmd)[^\n"]{0,80}?checks[/\\]{1,2}([\w.-]+\.(?:mjs|py|cmd))')
cut = datetime.fromisoformat(SINCE).timestamp()
runs = collections.Counter()
root = os.path.expanduser(r'~\.claude\projects')
for p in glob.glob(os.path.join(root, '*maps*', '**', '*.jsonl'), recursive=True):
    if os.path.getmtime(p) < cut: continue
    with open(p, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if 'tool_use' not in line or 'checks' not in line: continue
            try: o = json.loads(line)
            except Exception: continue
            if o.get('type') != 'assistant': continue
            for c in (o.get('message') or {}).get('content') or []:
                if isinstance(c, dict) and c.get('type') == 'tool_use' and c.get('name') in ('Bash', 'PowerShell'):
                    for f in set(RUN.findall((c.get('input') or {}).get('command', '') or '')): runs[f] += 1

rows = []
for f in files:
    b = os.path.basename(f)
    k = kind_of(f)
    what = FIRST_LINE.get(f) or (summary(header(os.path.join(CH, f))) if k != 'fixture' else '')
    if k == 'fixture': what = what or 'data file read by a check'
    if f in BROKEN: what = 'BROKEN (' + BROKEN[f] + '). ' + what
    rows.append(f"{f} | {k} | {'smoke' if f in in_smoke else '-'} | {runs.get(b, 0)} | {last.get(f, '')} | {what}")

head = f"""# checks/ index

Generated by `python trainworld/tools/checks_index.py` (about 20 s); do not edit by hand. Grep this before writing a scratch script.
Columns: name | kind | smoke | runs since {SINCE} (transcript scan) | last commit | what it proves (header comment's first sentence).
Kinds: check (proves a task's fix), tool (standing script), helper (imported by checks), fixture (data), lib (shared code).
BROKEN means it fails or cannot finish as it stands. `old/` holds {n_old} retired checks and is not listed.
Generated {datetime.now():%Y-%m-%d}; {len(rows)} files; {sum(1 for f in files if f in in_smoke)} in smoke.

"""
open(os.path.join(CH, 'INDEX.md'), 'w', encoding='utf-8', newline='\n').write(head + '\n'.join(rows) + '\n')
print(len(rows), 'files,', sum(runs.values()), 'runs counted ->', os.path.join(CH, 'INDEX.md'))
