"""Token use of Claude Code agents, from their transcripts (JSONL), deduplicated by message id.

Totals for one or more transcripts:
    python trainworld/checks/token_usage.py <transcript.jsonl> [...]
An agent's transcript is <config folder>/projects/<project>/<session>/subagents/agent-<id>.jsonl;
the session's own is <config folder>/projects/<project>/<session>.jsonl. The config folders are ~/.claude and
~/.claude-2 (maps/claude2.py); a path that is not a file is looked up as a session id or agent id in all of them.

Options (before the paths):
    --since 2026-10-14T21:24:45Z   count only calls at or after this time (UTC)
    --until ...                    and before this time
    --timeline                     one line per call: time, model, tokens, cumulative, and what the
                                   call did (its tool calls' descriptions), for splitting a task into parts

Weighted tokens: cache reads count 0.1, cache writes 1.25, output 5 (relative to fresh input), the
ratios of Anthropic's API prices, so different kinds of token can be added into one figure.
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import claude2  # noqa: E402 (maps root; the list of Claude config folders)

KINDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
WEIGHT = {"input_tokens": 1.0, "cache_creation_input_tokens": 1.25,
          "cache_read_input_tokens": 0.1, "output_tokens": 5.0}


def calls(path, since="", until=""):
    """The transcript's model calls in order: (time, model, usage, what it did)."""
    seen = {}
    order = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") if isinstance(rec, dict) else None
            if not isinstance(msg, dict) or not isinstance(msg.get("usage"), dict):
                continue
            ts = rec.get("timestamp", "")
            if (since and ts < since) or (until and ts >= until):
                continue
            key = msg.get("id") or rec.get("uuid")
            what = []
            for block in msg.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    inp = block.get("input") or {}
                    what.append(inp.get("description") or block.get("name", ""))
            if key not in seen:
                order.append(key)
                seen[key] = [ts, msg.get("model", "?"), msg["usage"], what]
            else:
                seen[key][2] = msg["usage"]
                seen[key][3] = seen[key][3] + what
    return [seen[k] for k in order]


def find(arg):
    """arg itself if it is a file, else the transcript of that session id or agent id in any config folder."""
    if os.path.isfile(arg):
        return arg
    name = os.path.basename(arg)
    name = name[:-6] if name.endswith(".jsonl") else name
    name = name[len("agent-"):] if name.startswith("agent-") else name
    for d in claude2.project_dirs():
        hits = (glob.glob(os.path.join(d, "*", name + ".jsonl"))
                + glob.glob(os.path.join(d, "*", "*", "subagents", "agent-" + name + ".jsonl")))
        if hits:
            return hits[0]
    sys.exit(f"no transcript for {arg} in {', '.join(claude2.config_dirs())}")


def weighted(u):
    return sum(WEIGHT[k] * (u.get(k) or 0) for k in KINDS)


def main():
    args = sys.argv[1:]
    since = until = ""
    timeline = False
    while args and args[0].startswith("--"):
        a = args.pop(0)
        if a == "--since":
            since = args.pop(0)
        elif a == "--until":
            until = args.pop(0)
        elif a == "--timeline":
            timeline = True
    for path in map(find, args):
        cs = calls(path, since, until)
        if timeline:
            cum = 0.0
            for ts, model, u, what in cs:
                w = weighted(u)
                cum += w
                print(f"{ts[:19]}\t{model}\t{w / 1e3:8.1f}k\t{cum / 1e6:7.2f}M\t{'; '.join(what)[:120]}")
        tot = defaultdict(lambda: defaultdict(float))
        for ts, model, u, what in cs:
            t = tot[model]
            t["calls"] += 1
            for k in KINDS:
                t[k] += u.get(k) or 0
            t["weighted"] += weighted(u)
        for model, t in tot.items():
            print(f"{path}\n  {model}: {int(t['calls'])} calls, input {int(t['input_tokens']):,}, "
                  f"cache writes {int(t['cache_creation_input_tokens']):,}, "
                  f"cache reads {int(t['cache_read_input_tokens']):,}, output {int(t['output_tokens']):,}; "
                  f"weighted {t['weighted'] / 1e6:.2f}M")


if __name__ == "__main__":
    main()
