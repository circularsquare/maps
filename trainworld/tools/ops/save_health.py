"""Read the save health summary from the account server, read-only (T-3775). One GET, nothing is changed.

    venv\Scripts\python.exe trainworld/tools/ops/save_health.py            scan reports and the quarantine summary
    venv\Scripts\python.exe trainworld/tools/ops/save_health.py --json     the raw answer
    venv\Scripts\python.exe trainworld/tools/ops/save_health.py --base http://127.0.0.1:8083/trainworld/inbox/

Ids and counts only; the server sends no game names or logins. The key is SHONEI_FEEDBACK_ADMIN_KEY, kept at
~/shonei-feedback-admin-key.txt and never printed. Answers 404 until the server has the savehealth route deployed.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://market.anita.garden/trainworld/inbox/"
KEY_FILE = Path.home() / "shonei-feedback-admin-key.txt"


def get(base):
    req = urllib.request.Request(base + "savehealth", headers={
        "Authorization": "Bearer " + KEY_FILE.read_text(encoding="utf-8").strip(),
        "User-Agent": "trainworld-ops/1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sys.exit("404: the server has no savehealth route yet (T-3775 not deployed), or the inbox key is unset there.")
        if e.code == 401:
            sys.exit("401: the server refused the key in " + str(KEY_FILE))
        sys.exit(f"the server answered {e.code}")


def when(t):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)) + " UTC" if t else "?"


def counts(d):
    return ", ".join(f"{k}: {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "none"


def main(argv):
    base = argv[argv.index("--base") + 1] if "--base" in argv else BASE
    out = get(base)
    if "--json" in argv:
        print(json.dumps(out, indent=2))
        return
    print("== scan reports (newest last)")
    for r in out.get("reports") or []:
        run = r.get("run") or {}
        print(f"{r.get('date')}  pass {r.get('pass')}  check v{r.get('checkV')}  new damage {r.get('newDamage')}  "
              f"stopped {r.get('stopped') or 'no'}  pass done {r.get('passDone')}")
        print(f"   this run: {counts({k: v for k, v in run.items() if v})}")
        for k in ("corrupt", "missing", "unreadableIndexes", "noIntact", "unresolved", "races"):
            for it in (r.get(k) or [])[:20]:
                print(f"   {k}: account {it.get('account')} game {it.get('game', '-')} rev {it.get('rev', '-')}"
                      f"{'  ' + it['note'] if it.get('note') else ''}")
            if len(r.get(k) or []) > 20:
                print(f"   {k}: ... {len(r[k]) - 20} more (use --json)")
    if not out.get("reports"):
        print("no reports yet")
    q = out.get("quarantine") or {}
    print(f"\n== quarantine: {q.get('total', 0)} records, {q.get('accounts', 0)} accounts, {q.get('games', 0)} games")
    for label, key in (("by status", "byStatus"), ("by found by", "byFoundBy"), ("by level", "byLevel"),
                       ("by reason", "byReason"), ("by day", "byDay")):
        print(f"{label}: {counts(q.get(key) or {})}")
    print("\nnewest records")
    for r in q.get("records") or []:
        print(f"{r['at']}  acct {r['account']}  game {r['game']}  {r['foundBy']}  rev {r.get('rev', '-')}  "
              f"{r['bytes']} B  {r['status']}  level {r.get('level', '-')}  track {r.get('track', '-')}  {r.get('reason', '')}")


if __name__ == "__main__":
    main(sys.argv[1:])
