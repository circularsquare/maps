"""Inspect cloud saves; optionally archive affected games outside live backup rotation.

Run on the server with --store /opt/shonei/trainworld/games and --archive a NEW,
private directory outside that store. Live files are only read. The archive contains
private player data; never publish it. No accounts, credentials or sessions are read.
Validation covers complete gzip/checksum, bounded output, JSON header and track magic;
it is not a full simulation load. The archive holds original bytes, never repairs.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import zlib

SAVE = re.compile(r"([a-z0-9]{1,40})\.r([0-9]+)\.save\Z")
MAX_RAW = 64 << 20


def invalid_constant(value):
    raise ValueError("non-finite JSON number")


def validate(data):
    try:
        stream = zlib.decompressobj(31)
        raw = stream.decompress(data, MAX_RAW + 1)
        if len(raw) > MAX_RAW or stream.unconsumed_tail:
            return {"status": "unverified", "reason": "unpacked size exceeds 64 MiB"}
        if not stream.eof:
            return {"status": "corrupt", "reason": "incomplete gzip stream"}
        if stream.unused_data.strip(b"\0"):
            return {"status": "corrupt", "reason": "unexpected data after gzip member"}
        if len(raw) < 8 or raw[:4] != b"TWG1":
            return {"status": "corrupt", "reason": "missing TWG1 header"}
        n = int.from_bytes(raw[4:8], "little")
        if n > len(raw) - 28:
            return {"status": "corrupt", "reason": "truncated header or track frame"}
        h = json.loads(raw[8:8+n].decode("utf-8"), parse_constant=invalid_constant)
        if not isinstance(h, dict) or h.get("game") != "anitabuilder":
            return {"status": "corrupt", "reason": "invalid game header"}
        if not isinstance(h.get("format"), int) or isinstance(h["format"], bool):
            return {"status": "corrupt", "reason": "invalid format number"}
        clock = h.get("clock")
        if not isinstance(clock, (int, float)) or isinstance(clock, bool) or not math.isfinite(clock) or clock < 0:
            return {"status": "corrupt", "reason": "invalid clock"}
        track = raw[8+n:]
        if h["format"] != 1 or track[:4] not in (b"TWT4", b"TWT5", b"TWT6"):
            return {"status": "unverified", "reason": "unsupported save or track format"}
        return {"status": "intact", "day": int(clock // 86400), "track_bytes": len(track), "raw_bytes": len(raw)}
    except (ValueError, UnicodeError, zlib.error) as error:
        return {"status": "corrupt", "reason": str(error)}


def write_private(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as f:
        os.chmod(path, 0o600)
        f.write(data)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
        raise RuntimeError("archive verification failed: " + str(path))


def scan(store, archive=None):
    store = store.resolve(strict=True)
    if archive:
        archive = archive.resolve()
        if archive == store or store in archive.parents:
            raise ValueError("archive must be outside the live save store")
        archive.mkdir(parents=True, mode=0o700, exist_ok=False)
    counts = dict(accounts=0, games=0, files=0, bytes=0, intact=0, corrupt=0,
                  unverified=0, affected_games=0, affected_accounts=0,
                  corrupt_current_games=0, games_with_pre_corruption_copy=0,
                  games_with_any_intact_copy=0, archived_files=0, archived_bytes=0,
                  rotated_during_scan=0, index_errors=0)
    manifest = {"schema": 1, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "store": str(store), "archive": str(archive) if archive else None,
                "validation": "strict gzip/checksum + header + track magic, not a simulation load",
                "affected": [], "issues": []}
    started = time.monotonic()
    for account in sorted(store.iterdir()):
        if not account.is_dir() or not re.fullmatch(r"[a-z0-9]{1,40}", account.name):
            continue
        counts["accounts"] += 1
        try:
            index_bytes = (account / "index.json").read_bytes()
            index = json.loads(index_bytes)
        except (OSError, ValueError) as error:
            counts["index_errors"] += 1
            manifest["issues"].append({"account": account.name, "error": str(error)})
            continue
        groups = {}
        games = index.get("games", {})
        for gid, game in games.items():
            groups[gid] = {"meta": game, "files": {}, "missing": []}
            for entry in [game] + game.get("backups", []):
                groups[gid]["files"][f"{gid}.r{entry['rev']}.save"] = entry
        # Include orphaned revisions still present; they can be the only intact older copy.
        for file in account.glob("*.save"):
            match = SAVE.fullmatch(file.name)
            if match:
                gid = match[1]
                groups.setdefault(gid, {"meta": {}, "files": {}, "missing": []})["files"].setdefault(file.name, {})
        account_affected = False
        for gid, group in groups.items():
            counts["games"] += 1
            records, payloads = [], {}
            for filename, entry in sorted(group["files"].items()):
                source = account / filename
                try:
                    data = source.read_bytes()
                except FileNotFoundError:
                    # Rotation can remove an indexed backup while this read-only scan runs.
                    current_index = json.loads((account / "index.json").read_bytes())
                    current_game = current_index.get("games", {}).get(gid, {})
                    refs = [current_game] + current_game.get("backups", [])
                    rev = int(SAVE.fullmatch(filename)[2])
                    if any(r.get("rev") == rev for r in refs):
                        group["missing"].append(filename)
                    else:
                        counts["rotated_during_scan"] += 1
                    continue
                verdict = validate(data)
                counts["files"] += 1
                counts["bytes"] += len(data)
                counts[verdict["status"]] += 1
                rev = int(SAVE.fullmatch(filename)[2])
                record = {"file": filename, "rev": rev, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "current_at_scan": rev == group["meta"].get("rev"),
                          "savedAt": entry.get("savedAt"), "tier": entry.get("tier"), **verdict}
                records.append(record)
                payloads[filename] = data
            bad = [r for r in records if r["status"] != "intact"]
            if not bad and not group["missing"]:
                continue
            counts["affected_games"] += 1
            account_affected = True
            intact = [r for r in records if r["status"] == "intact"]
            corrupt = [r for r in records if r["status"] == "corrupt"]
            before = [r for r in intact if corrupt and r["rev"] < min(b["rev"] for b in corrupt)]
            newest_before = max(before, key=lambda r: r["rev"], default=None)
            newest_intact = max(intact, key=lambda r: r["rev"], default=None)
            counts["corrupt_current_games"] += any(r["current_at_scan"] for r in corrupt)
            counts["games_with_pre_corruption_copy"] += newest_before is not None
            counts["games_with_any_intact_copy"] += newest_intact is not None
            item = {"account": account.name, "game": gid, "name": group["meta"].get("name"),
                    "metadata_at_scan": group["meta"], "revisions": records, "missing": group["missing"],
                    "latest_intact_before_first_known_corruption": newest_before["file"] if newest_before else None,
                    "latest_intact": newest_intact["file"] if newest_intact else None}
            manifest["affected"].append(item)
            if archive:
                target = archive / "accounts" / account.name
                if not (target / "index.json").exists():
                    write_private(target / "index.json", index_bytes)
                for filename, data in payloads.items():
                    write_private(target / filename, data)
                    counts["archived_files"] += 1
                    counts["archived_bytes"] += len(data)
                write_private(target / (gid + ".recovery.json"), (json.dumps(item, indent=2) + "\n").encode())
                print(json.dumps({"progress": "affected game archived", "affected_games": counts["affected_games"],
                                  "archived_files": counts["archived_files"]}), flush=True)
        counts["affected_accounts"] += account_affected
        if counts["accounts"] % 500 == 0:
            print(json.dumps({"progress": "scan", "accounts": counts["accounts"], "files": counts["files"],
                              "corrupt": counts["corrupt"], "seconds": round(time.monotonic() - started)}), flush=True)
    manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["counts"] = counts
    if archive:
        write_private(archive / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        write_private(archive / "README.txt", b"Private original player-save recovery archive. Do not publish.\nSee manifest.json for account/game/revision mapping, validation and SHA-256 hashes.\nNo live save was modified. Intact means file integrity, not full simulation validation.\nRestore as a separate copy only after checking with the affected player.\n")
    print(json.dumps({"complete": True, "archive": str(archive) if archive else None,
                      "seconds": round(time.monotonic() - started), **counts}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if hasattr(os, "nice"):
        os.nice(10)
    scan(args.store, args.archive)
