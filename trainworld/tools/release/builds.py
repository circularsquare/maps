"""Which release a build id belongs to (T-3770). feedback-inbox/builds.json maps each published build id (the base-36 stamp in
build.json, `version:` in a feedback report) to its release; deploy.py adds the new build when it publishes, so
nobody edits the file.

    label("mwlo7xyt")  -> "v0.4.1 (mwlo7xyt)"
    label("mxxxxxx")   -> "after v0.4.2 (mxxxxxx)"   newer than every listed build is after the newest release
    label("dev")       -> "dev"
"""
import json
import os
import re

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feedback-inbox", "builds.json")  # served beside the inbox page
ID = re.compile(r"[a-z0-9]{6,10}")


def load(path=FILE):
    try:
        with open(path, encoding="utf-8") as f:
            b = json.load(f).get("builds")
        return b if isinstance(b, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def label(build, builds=None):
    builds = load() if builds is None else builds
    if not isinstance(build, str) or not ID.fullmatch(build):
        return build
    if build in builds:
        return "%s (%s)" % (builds[build], build)
    n = int(build, 36)
    earlier = [b for b in builds if int(b, 36) < n]
    if not earlier:
        return build
    return "after %s (%s)" % (builds[max(earlier, key=lambda b: int(b, 36))], build)


def record(build, release, path=FILE):
    """Add `build` as part of `release` (newest first). A build already listed keeps its release."""
    builds = load(path)
    if not ID.fullmatch(build or "") or build in builds:
        return False
    rows = {build: release, **builds}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write('{\n  "builds": {\n%s\n  }\n}\n' % ",\n".join('    "%s": "%s"' % kv for kv in rows.items()))
    return True
