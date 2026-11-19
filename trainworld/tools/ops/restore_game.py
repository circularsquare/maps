"""Put a deleted or lost game back into a player's account under a new game id (T-3789; shonei-server
trainworld_restore.go). Create-only: nothing on the account is written over or deleted, the old id stays
deleted, and the player's browsers show the game back as a card of its own at their next start menu.

    venv\\Scripts\\python.exe trainworld/tools/ops/restore_game.py list-deleted <account>
    venv\\Scripts\\python.exe trainworld/tools/ops/restore_game.py undelete <account> <old game id> [--rev N] [--yes]
    venv\\Scripts\\python.exe trainworld/tools/ops/restore_game.py put <account> <save file> [--name NAME]
        [--stations N] [--lines N] [--yes]

Every write is a dry run unless --yes is given: the server runs the same checks (the save check, the account's
50 games and 64 MB) and says what it would do, writing nothing. Each run with --yes needs Anita's yes. A game that
does not fit is refused "full"; a restore never erases a held deleted game to make room.

list-deleted   the account's deleted games still held (7 days after the delete): old id, when deleted and
               erased, day, stations, lines, and each stored rev (the current one, its backups, its pinned
               copy) with the server's check.
undelete       a held game as a new game: its current rev, or --rev N (one listed by list-deleted), with its
               name, card facts and picture.
put            a .save file as a new game. The card's day, cash and mode are read from the save; stations
               and lines are not, so pass them (the backup's index has them) or the card shows 0 until the
               player opens the game and it saves. --name defaults to the file name.

The account is the noritetsu id (u + 16 hex): `save_health.py` and the feedback inbox show it.

The key is the server's SHONEI_GAMES_ADMIN_KEY (not the inbox key), kept at ~/shonei-games-admin-key.txt and
never printed. TW_GAMES_URL and TW_GAMES_ADMIN_KEY point it at a local server instead.

A game deleted more than 7 days ago, or from before the server held deleted games, is only in the off-site
backup (7 nightly copies in R2). Getting it out is manual, needs Anita's yes each time (her private key opens
every player's data), and reads one account folder only, streamed so nothing else lands on disk (Git Bash,
from an empty folder of its own; notes/T-3786.md did this for Rus's game):

    rclone lsf r2:shonei-backups                                             # shonei-YYYYMMDDTHHMMSSZ.tar.gz.age
    rclone cat r2:shonei-backups/<name> | age -d -i <private key file> \\
        | tar -tzf - --wildcards './trainworld/games/<account>/*'             # list first
    rclone cat r2:shonei-backups/<name> | age -d -i <private key file> \\
        | tar -xzf - --wildcards './trainworld/games/<account>/*'             # then extract

(The bucket is shonei-server's BACKUP_DEST, deploy/shonei-backup.env.example; `r2:` is the remote on Anita's PC
that T-3786 used; the key file is the one in shonei-server's deploy/README.md "Restore".)

The folder's index.json names each game's revs (`<game>.r<rev>.save`). Pick the rev, run `put` with that file,
then delete everything extracted.
"""
import gzip
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

URL = os.environ.get("TW_GAMES_URL", "https://market.anita.garden").rstrip("/") + "/trainworld/admin/games/"
KEY_FILE = Path.home() / "shonei-games-admin-key.txt"


def key():
    k = os.environ.get("TW_GAMES_ADMIN_KEY")
    if k:
        return k
    if not KEY_FILE.exists():
        sys.exit(f"no key: put SHONEI_GAMES_ADMIN_KEY's value in {KEY_FILE}")
    return KEY_FILE.read_text(encoding="utf-8").strip()


def call(method, path, body=None, ctype=None):
    headers = {"Authorization": "Bearer " + key(), "User-Agent": "trainworld-ops/1"}
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(URL + path, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            out = json.loads(e.read())
        except ValueError:
            out = {}
        if e.code == 404 and not out.get("code") and out.get("error") == "not found":
            sys.exit("404: the server has no restore route (T-3789 not deployed), or SHONEI_GAMES_ADMIN_KEY is unset there.")
        if e.code == 401:
            sys.exit("401: the server refused the key")
        sys.exit(f"the server answered {e.code}: {json.dumps(out)}")


def when(t):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)) + " UTC" if t else "-"


def header_facts(data):
    """The card facts a save's own header gives: day, cash, mode, city. The rest (stations, lines, cities, the
    picture) the game works out from the network when it saves."""
    raw = gzip.decompress(data)
    if raw[:4] != b"TWG1":
        sys.exit("not a trainworld save (no TWG1)")
    n = struct.unpack("<I", raw[4:8])[0]
    h = json.loads(raw[8:8 + n])
    facts = {"at": int(time.time() * 1000), "day": int(h.get("clock", 0) // 86400)}
    for k in ("cash", "mode", "city"):
        if k in h:
            facts[k] = h[k]
    return facts


def multipart(fields):
    boundary = uuid.uuid4().hex
    out = b""
    for name, (filename, ctype, value) in fields.items():
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{boundary}\r\nContent-Disposition: {disp}\r\nContent-Type: {ctype}\r\n\r\n".encode() + value + b"\r\n"
    return out + f"--{boundary}--\r\n".encode(), "multipart/form-data; boundary=" + boundary


def flag(argv, name, cast=str):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return cast(v)
    return None


def said(out, yes):
    if out.get("dryRun"):
        print(f"dry run: would add a new game (rev {out['rev']}, check {out['check']}); the account would have "
              f"{out['games']} games, {out['used'] / 1e6:.1f} MB")
        print("nothing written; run again with --yes to write it")
    else:
        print(f"added game {out['id']} (rev {out['rev']}, check {out['check']}); the account uses {out['used'] / 1e6:.1f} MB")
        print("the player sees it at their next start menu (a reload)")


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    argv = list(argv)
    yes = "--yes" in argv
    if yes:
        argv.remove("--yes")
    rev = flag(argv, "--rev", int)
    name = flag(argv, "--name")
    stations = flag(argv, "--stations", int)
    lines = flag(argv, "--lines", int)
    if len(argv) < 2 or argv[0] not in ("list-deleted", "undelete", "put"):
        sys.exit(__doc__)
    cmd, acct = argv[0], argv[1]
    if cmd == "list-deleted":
        out = call("GET", f"{acct}/deleted")
        rows = out["deleted"]
        for g in rows:
            print(f"{g['id']}  deleted {when(g['deletedAt'])}  erased by {when(g['erasedBy'])}  day {g.get('day', '-')}  "
                  f"stations {g.get('stations', '-')}  lines {g.get('lines', '-')}  cities {g.get('cities', '-')}  "
                  f"origin {(g.get('origin') or '-')[:4]}  picture {'yes' if g['picture'] else 'no'}")
            for r in g["revs"]:
                print(f"    rev {r['rev']:<5} {r['kind']:<8} saved {when(r['savedAt'])}  {r['bytes']:>9} B  {r.get('check') or 'unchecked'}")
        print(f"{len(rows)} deleted games held (kept {out['keepDays']} days)")
    elif cmd == "undelete":
        if len(argv) != 3:
            sys.exit(__doc__)
        body = {"game": argv[2], "dryRun": not yes}
        if rev:
            body["rev"] = rev
        out = call("POST", f"{acct}/undelete", json.dumps(body).encode(), "application/json")
        print(f"from deleted game {out['from']} rev {out['fromRev']}")
        said(out, yes)
    else:
        if len(argv) != 3:
            sys.exit(__doc__)
        path = Path(argv[2])
        data = path.read_bytes()
        facts = header_facts(data)
        if stations is not None:
            facts["stations"] = stations
        if lines is not None:
            facts["lines"] = lines
        meta = {"name": name or path.stem, "facts": facts}
        body, ctype = multipart({"meta": (None, "application/json", json.dumps(meta).encode()),
                                 "save": ("game.save", "application/octet-stream", data)})
        print(f"{path.name}: {len(data)} bytes, day {facts['day']}, stations {facts.get('stations', '-')}, "
              f"lines {facts.get('lines', '-')}")
        out = call("POST", f"{acct}/put" + ("" if yes else "?dryRun=1"), body, ctype)
        said(out, yes)


if __name__ == "__main__":
    main(sys.argv[1:])
