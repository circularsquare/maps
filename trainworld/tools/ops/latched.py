"""Games latched as a sandbox on the leaderboards, and taking a wrong latch off (T-3687, shonei-server
trainworld_latch.go). Needs the server deploy of shonei-server c960adc.

    venv\\Scripts\\python.exe trainworld/tools/ops/latched.py                    list every latch (GET only)
    venv\\Scripts\\python.exe trainworld/tools/ops/latched.py --wrong            only those whose save says normal
    venv\\Scripts\\python.exe trainworld/tools/ops/latched.py --unlatch <account> <game>   take one off (Anita's call)

A page up to v0.4.x could latch the game a player was leaving when they opened a sandbox file, a copy of a Community
share or a modded save. "wrong" means the game's current save on the account is a normal game, not a copy and not
modded, so the latch was a mistake. A game not on the account shows mode "-": nothing says either way. Taking a latch
off lists the game again from its next send and drops its share's Sandbox tag. Agents run only the listing.

The key is the server's SHONEI_FEEDBACK_ADMIN_KEY, kept at ~/shonei-feedback-admin-key.txt; it is never printed.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# TW_INBOX_URL and TW_INBOX_KEY point it at a local server instead
URL = os.environ.get("TW_INBOX_URL", "https://market.anita.garden/trainworld/inbox/latched")
KEY_FILE = Path.home() / "shonei-feedback-admin-key.txt"


def call(method="GET", body=None):
    key = os.environ.get("TW_INBOX_KEY") or KEY_FILE.read_text(encoding="utf-8").strip()
    req = urllib.request.Request(URL, method=method, data=None if body is None else json.dumps(body).encode(), headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "User-Agent": "trainworld-ops/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    if "--unlatch" in argv:
        i = argv.index("--unlatch")
        acct, game = argv[i + 1], argv[i + 2]
        print(call("POST", {"account": acct, "game": game}))
        return
    rows = call()["games"]
    if "--wrong" in argv:
        rows = [r for r in rows if r["wrong"]]
    for r in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["at"]))
        flags = " ".join(k for k in ("copy", "modded") if r.get(k))
        print(f"{when}  {r['account']}  {r['game']}  {r['mode'] or '-':<8} {'WRONG' if r['wrong'] else '':<6}"
              f"{flags:<12} share {r.get('share') or '-':<9} {r.get('name', '')}")
    print(f"{len(rows)} latched, {sum(r['wrong'] for r in rows)} of them normal games")


if __name__ == "__main__":
    main(sys.argv[1:])
