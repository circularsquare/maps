"""T-4364 job: download (or find on disk) every Geofabrik extract the listed plan rows build from, before the
pack writes, so Geofabrik's daily switch-over (a "latest" link that 404s for a while, T-4027, notes/T-3688.md) costs
a wait instead of a parked row. Retries each failed extract every RETRY_S, for up to TRIES tries.

    prefetch.py <id,id,...> [--extra <extract,extract>] [--list <file>] [--skip-missing] [--dry]

--extra adds extracts no plan row names (a hand pack's); --list appends each extract downloaded now (not already on
disk) to <file>, for the job to delete at the end. Exit 0 when all are on disk, else 1.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import routes  # noqa: E402

RETRY_S = 900
TRIES = 8


def main():
    ids = [i for i in sys.argv[1].split(",") if i]
    argv = sys.argv[2:]
    extra = argv[argv.index("--extra") + 1].split(",") if "--extra" in argv else []
    lst = argv[argv.index("--list") + 1] if "--list" in argv else None
    rows = {r["id"]: r for r in json.load(open(os.path.join(TW, "pipeline", "world_plan.json"), encoding="utf-8"))["metros"]}
    want = []
    for i in ids:
        if i not in rows:
            if "--skip-missing" in argv:  # a spec's new row, before its edit: its template's extracts are listed
                print(f"  {i}: no plan row yet, skipped")
                continue
            raise SystemExit(f"{i}: no plan row")
        want += rows[i].get("extracts") or []
    want = list(dict.fromkeys(want + extra))
    print("extracts:", ", ".join(want))
    if "--dry" in argv:
        for e in want:
            print(f"  {e}: {'on disk' if os.path.exists(routes.pbf_path(e)) else 'to download'}")
        return
    bad = []
    for e in want:
        if os.path.exists(routes.pbf_path(e)):
            print(f"  {e}: on disk")
            continue
        for k in range(TRIES):
            try:
                routes.download(e)
                if lst:
                    with open(lst, "a", encoding="utf-8") as f:
                        f.write(routes.pbf_path(e) + "\n")
                break
            except Exception as x:  # noqa: BLE001 (a 404 at the switch-over, a reset, a timeout)
                print(f"  {e}: try {k + 1} of {TRIES} failed: {x}", flush=True)
                if k + 1 < TRIES:
                    time.sleep(RETRY_S)
        else:
            bad.append(e)
    if bad:
        print("FAILED:", ", ".join(bad))
        sys.exit(1)
    print("all on disk")


if __name__ == "__main__":
    main()
