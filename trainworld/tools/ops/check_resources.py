"""Every traffic and storage limit the live game depends on, measured now (T-3243). Read-only everywhere.

    python trainworld/tools/ops/check_resources.py            (from the maps root; venv\\Scripts\\python.exe works too)
    python trainworld/tools/ops/check_resources.py --no-r2    (skip the R2 bucket listing, about a minute)
    python trainworld/tools/ops/check_resources.py --json out.json

One line per resource: use, limit, %, growth a day, days until 80%. A line is flagged when it is over 50%, or
when its growth reaches 80% within 30 days, or when something that should exist does not. What each line means
and what to do about a flag: trainworld/docs/ops-checks.md.

Where things come from:
- the box: one ssh call that runs a small Python probe on stdin (nothing is written there), reading the stores'
  files, the journal, sysstat and /proc. The same `ssh root@5.161.112.237` the server's own tools use.
- limits set in shonei-server's code are read from the local clone (../shonei-server), so a raised cap shows up
  here once it is in that repo; the box runs whatever was last deployed, which the doc says how to check.
- R2: `rclone lsjson` on the `r2:` remote that deploy.py uses (rclone's own config holds the key).
  Operation counts and the data host's cache come from Cloudflare's GraphQL analytics with her ops token,
  ~/cloudflare-ops-token.txt (or CF_ANALYTICS_TOKEN); never printed. Without it they are estimated from plays.
- GitHub: `gh api` for the website repo's size; `curl`-style HEAD requests for cache status.
Each run appends its numbers to ../artifacts/trainworld/ops/history.jsonl; growth uses that history when it
reaches back a day or more, else timestamps inside the stores.
"""
import argparse
import datetime as dt
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import time
import urllib.request

MAPS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SHONEI = os.path.join(MAPS, "..", "shonei-server")
WEBSITE = os.path.join(MAPS, "..", "website")
HISTORY = os.path.join(MAPS, "artifacts", "trainworld", "ops", "history.jsonl")
SERVER = "root@5.161.112.237"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", SERVER]
UA = {"User-Agent": "ops-check"}

GB = 1 << 30
MB = 1 << 20
# Outside limits nobody's code sets. Where each figure comes from is in docs/ops-checks.md.
HETZNER_TRAFFIC = 1 << 40          # CPX11 in Ashburn: 1 TB outgoing a month included (EU locations get 20 TB)
RESEND_DAY, RESEND_MONTH = 100, 3000   # Resend free plan; a quota error in the journal overrides the daily one
R2_FREE_STORAGE = 10 * GB
R2_FREE_A, R2_FREE_B = 1_000_000, 10_000_000
PAGES_SITE = 1 * GB                # GitHub Pages: published site
PAGES_REPO = 1 * GB                # GitHub: recommended repo size (5 GB "strongly recommended")
PAGES_BANDWIDTH = 100 * GB         # GitHub Pages: soft bandwidth a month
# Measured by T-3243 (2026-11-12) on one live first load of /trainsworld/?new=1: requests to R2 and bytes that
# came from GitHub because Cloudflare passed them through (cf-cache-status DYNAMIC). Re-measure with
# first_load_requests.mjs beside this file after a release that changes what a first load fetches.
R2_READS_PER_PLAY = 33
PAGES_BYTES_PER_NEW_BROWSER = 2.74 * MB

# ── limits from shonei-server's code ────────────────────────────────────────────────────────────────────────
CODE_LIMITS = {  # name: (file, default if the clone is missing)
    "MaxAccounts": ("auth.go", 1000),
    "MaxNoriAccounts": ("noritetsu.go", 10000),
    "MaxNoriPasswordAccounts": ("noritetsu.go", 5000),
    "MaxNoriStoreBytes": ("noritetsu.go", 512 * MB),
    "MaxTwGamesBytes": ("trainworld_games.go", 8 * GB),
    "twGamesWarnAt": ("trainworld_games.go", 4 * GB),
    "MaxTwAccountBytes": ("trainworld_games.go", 64 * MB),
    "MaxTwAccountGames": ("trainworld_games.go", 50),
    "MaxTwStoreBytes": ("trainworld.go", 128 * MB),
    "MaxTwCommunityBytes": ("trainworld_community.go", 2 * GB),
    "MaxTwShares": ("trainworld_community.go", 5000),
    "MaxFeedbackStoreBytes": ("feedback.go", 512 * MB),
    "MaxAgentChatStoreBytes": ("agentchat.go", 16 * MB),
    "ptMaxBrowsers": ("trainworld_playtime.go", 200000),
    "ptMaxPlaysDay": ("trainworld_playtime.go", 100000),
    "achMaxBrowsers": ("trainworld_achievements.go", 200000),
    "LBMaxLog": ("leaderboard.go", 200000),
}


def code_limits():
    out, src = {}, {}
    for name, (fname, default) in CODE_LIMITS.items():
        out[name], src[name] = default, f"default (no {fname})"
        try:
            text = open(os.path.join(SHONEI, fname), encoding="utf-8").read()
        except OSError:
            continue
        m = re.search(rf"\b{name}\s*(?:int64|int)?\s*=\s*([0-9_]+)(?:\s*<<\s*([0-9]+))?", text)
        if m:
            out[name] = int(m.group(1).replace("_", "")) << int(m.group(2) or 0)
            src[name] = f"{fname} {name}"
    try:
        text = open(os.path.join(SHONEI, "noritetsu.go"), encoding="utf-8").read()
        m = re.search(r"newAccounts:\s*rate\.NewLimiter\(perHour\((\d+)\)", text)
        out["noriSignupsPerHour"], src["noriSignupsPerHour"] = (int(m.group(1)), "noritetsu.go newAccounts") if m else (300, "default")
    except OSError:
        out["noriSignupsPerHour"], src["noriSignupsPerHour"] = 300, "default"
    return out, src


# ── the box ─────────────────────────────────────────────────────────────────────────────────────────────────
# Runs on the server from stdin. Reads only; prints one JSON object. Never prints an address, name or hash.
PROBE = r'''
import json, os, re, subprocess, time, collections, datetime as dt
B = "/opt/shonei/"
now = time.time()
today = dt.datetime.utcfromtimestamp(now).date()
out = {"now": now}
def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""
def tree(p):
    tot = n = 0
    for root, dirs, files in os.walk(p):
        for f in files:
            try:
                tot += os.stat(os.path.join(root, f)).st_size; n += 1
            except OSError:
                pass
    return tot, n
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
def day(t):
    return dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")

st = os.statvfs("/")
out["disk"] = {"used": (st.f_blocks - st.f_bfree) * st.f_frsize, "size": st.f_blocks * st.f_frsize,
               "avail": st.f_bavail * st.f_frsize, "inodes": st.f_files - st.f_ffree, "inodes_max": st.f_files}
mi = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo")}
out["mem"] = {"total": mi["MemTotal"], "avail": mi["MemAvailable"], "swap": mi.get("SwapTotal", 0)}
out["load"] = [float(x) for x in open("/proc/loadavg").read().split()[:3]]
out["nproc"] = os.cpu_count()
out["uptime_days"] = float(open("/proc/uptime").read().split()[0]) / 86400
pid = run("systemctl show -p MainPID --value shonei").strip()
svc = {"active": run("systemctl is-active shonei").strip()}
if pid and pid != "0":
    s = open("/proc/%s/status" % pid).read()
    svc["rss"] = int(re.search(r"VmRSS:\s+(\d+)", s).group(1)) * 1024
    svc["fds"] = len(os.listdir("/proc/%s/fd" % pid))
    m = re.search(r"Max open files\s+(\d+)", open("/proc/%s/limits" % pid).read())
    svc["fd_max"] = int(m.group(1)) if m else None
    svc["started"] = os.stat("/proc/%s" % pid).st_mtime
out["service"] = svc
out["journal_bytes"] = tree("/var/log/journal")[0]
m = re.search(r"^SystemMaxUse=(\d+)([KMG]?)", run("cat /etc/systemd/journald.conf /etc/systemd/journald.conf.d/*.conf 2>/dev/null"), re.M)
out["journal_max"] = int(m.group(1)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2)] if m else None

# Outgoing traffic this calendar month, from sysstat's daily averages (txkB/s of eth0).
tx = {}
for d in range(1, today.day + 1):
    f = "/var/log/sysstat/sa%02d" % d
    txt = ""
    if os.path.exists(f) and os.path.getmtime(f) > now - 32 * 86400:
        txt = run("LC_ALL=C sar -n DEV -f %s 2>/dev/null" % f)
    g = "/var/log/sysstat/sar%02d" % d
    if not txt and os.path.exists(g) and os.path.getmtime(g) > now - 32 * 86400:
        txt = open(g, errors="ignore").read()
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 10 and p[0] == "Average:" and p[1] == "eth0":
            tx[d] = float(p[5]) * 1024 * 86400
            break
out["tx_days"] = tx

# Stores
st = {}
for name in ["trainworld/games", "trainworld/scores", "trainworld/community", "noritetsu/saves", "feedback",
             "agentchat", "saves", "kitchen", "shop"]:
    st[name] = tree(B + name)
st["opt_shonei"] = tree(B)
out["stores"] = st
games_accounts = [d for d in os.listdir(B + "trainworld/games") if os.path.isdir(B + "trainworld/games/" + d)]
out["games_accounts"] = len(games_accounts)
per = sorted(tree(B + "trainworld/games/" + d)[0] for d in games_accounts)
out["games_per_account"] = {"median": per[len(per) // 2] if per else 0, "p99": per[int(len(per) * .99)] if per else 0,
                            "max": per[-1] if per else 0}

na = (load(B + "noritetsu/accounts.json") or {}).get("accounts") or []
kinds = collections.Counter((a.get("logins") or ["?:"])[0].split(":")[0] for a in na)
created = [a.get("created", 0) for a in na]
per_day = collections.Counter(day(c) for c in created if c > now - 8 * 86400)
per_hour = collections.Counter(int(c // 3600) for c in created if c > now - 7 * 86400)
out["nori"] = {"accounts": len(na), "kinds": dict(kinds), "per_day": dict(per_day),
               "peak_hour": max(per_hour.values()) if per_hour else 0}
sa = load(B + "accounts.json") or {}
out["shonei_accounts"] = len(sa)

pt = load(B + "trainworld/playtime.json") or {}
days = pt.get("days") or {}
out["play"] = {"browsers": len(pt.get("browsers") or {}),
               "days": {k: {"plays": v.get("plays", 0), "new": v.get("newBrowsers", 0), "browsers": v.get("browsers", 0),
                            "accounts": v.get("accounts", 0), "ids": len(v.get("ids") or {})}
                        for k, v in sorted(days.items())[-8:]}}
ach = load(B + "trainworld/achievements.json") or {}
out["ach_browsers"] = len(ach.get("browsers") or {})
cm = load(B + "trainworld/community/index.json") or {}
shares = cm.get("shares") or []
out["community"] = {"shares": len(shares), "per_day": dict(collections.Counter(day(s.get("at", 0)) for s in shares))}
out["players"] = len(((load(B + "trainworld/players.json") or {}).get("players")) or {})
lb = {}
for g in ["circles", "guesscity"]:
    d = load(B + g + ".json") or {}
    lb[g] = {"scores": len(d.get("scores") or d.get("log") or []), "plays": len(d.get("plays") or [])}
out["lb"] = lb

# Journal, last 8 days: mail per UTC day, quota errors, refusals, the games size the server logs at each start.
mail = collections.Counter(); quota = {}; events = collections.Counter(); games_log = []
pat = {"signup_refused": "sign-up refused", "budget": "budget reached", "over_budget": "over budget",
       "full": " is full", "alert": "[alert]", "panic": "panic:", "broken": "broken=true"}
j = subprocess.Popen("journalctl -u shonei --since -8d -o short-iso --no-pager", shell=True, stdout=subprocess.PIPE, text=True, errors="ignore")
for line in j.stdout:
    d = line[:10]
    if "[mail] sent " in line:
        mail[d] += 1
    elif "quota_exceeded" in line or "mail quota used up" in line:
        quota.setdefault(d, mail[d])
    for k, s in pat.items():
        if s in line and (k != "alert" or "store at" not in line):
            if line[:19] >= dt.datetime.utcfromtimestamp(now - 86400).strftime("%Y-%m-%dT%H:%M:%S"):
                events[k] += 1
    m = re.search(r"(\d+) KB of games", line)
    if m:
        games_log.append((line[:19], int(m.group(1)) * 1024))
j.wait()
out["mail"] = {"per_day": dict(mail), "quota_hit_at": quota}
out["mail_month"] = 0
mm = run("journalctl -u shonei --since %s-01 -o cat --no-pager | grep -c '\\[mail\\] sent '" % today.strftime("%Y-%m"))
out["mail_month"] = int(mm.strip() or 0)
out["events_24h"] = dict(events)
out["games_log"] = games_log
since = "--since @%d" % int(svc.get("started", now - 86400) - 60)
out["google_on"] = "Google sign-in on" in run("journalctl -u shonei %s -o cat --no-pager | grep -m1 'Google sign-in'" % since)

bk = {"offsite_timer": run("systemctl is-active shonei-backup.timer").strip(),
      "offsite_env": os.path.exists("/etc/shonei-backup.env"),
      "offsite_last": run("journalctl -u shonei-backup -n 50 -o short-iso --no-pager | grep -E 'shonei-backup (ok|FAILED)' | tail -1").strip()[:160]}
out["backup"] = bk
print(json.dumps(out))
'''


def probe_box():
    r = subprocess.run(SSH + ["nice -n 19 python3 -I -"], input=PROBE, capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-400:])
    return json.loads(r.stdout)


# ── R2 ──────────────────────────────────────────────────────────────────────────────────────────────────────
DATED = re.compile(r"^\d{4}-\d{2}-\d{2}[a-z]?$")


def r2_listing():
    r = subprocess.run(["rclone", "lsjson", "-R", "--files-only", "--fast-list", "--no-mimetype", "--no-modtime",
                        "--s3-no-check-bucket", "r2:anitamaps"], capture_output=True, text=True, timeout=900)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-300:])
    top, versions = {}, {}
    for o in json.loads(r.stdout):
        parts = o["Path"].split("/")
        t = top.setdefault(parts[0], [0, 0]); t[0] += o["Size"]; t[1] += 1
        if parts[0] == "trainworld" and len(parts) > 2:
            v = versions.setdefault(parts[1], [0, 0]); v[0] += o["Size"]; v[1] += 1
    return top, versions


TOKEN_FILES = [os.path.expanduser("~/cloudflare-ops-token.txt"), os.path.expanduser("~/.cloudflare-analytics-token")]
CF_API = "https://api.cloudflare.com/client/v4"


def cf_token():
    """The Cloudflare ops token (read zone settings and cache rules, list R2 buckets, account analytics), or None.
    Never printed."""
    if os.environ.get("CF_ANALYTICS_TOKEN"):
        return os.environ["CF_ANALYTICS_TOKEN"]
    for p in TOKEN_FILES:
        if os.path.exists(p):
            return open(p).read().strip()
    return None


def cf_get(token, path):
    req = urllib.request.Request(CF_API + path, headers={"Authorization": "Bearer " + token, **UA})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def cf_gql(token, query, variables):
    req = urllib.request.Request(CF_API + "/graphql", data=json.dumps({"query": query, "variables": variables}).encode(),
                                 headers={"Authorization": "Bearer " + token, "Content-Type": "application/json", **UA})
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    if d.get("errors"):
        raise RuntimeError(str(d["errors"][0].get("message", d["errors"]))[:160])
    return d["data"]


CLASS_A = {"ListBuckets", "PutBucket", "ListObjects", "PutObject", "CopyObject", "CompleteMultipartUpload",
           "CreateMultipartUpload", "UploadPart", "UploadPartCopy", "ListMultipartUploads", "ListParts",
           "PutBucketEncryption", "PutBucketCors", "PutBucketLifecycleConfiguration", "LifecycleStorageTierTransition"}
FREE_OPS = {"DeleteObject", "DeleteBucket", "AbortMultipartUpload"}
R2_OPS_Q = """query($a: String!, $s: Time!, $e: Time!) { viewer { accounts(filter: {accountTag: $a}) {
  r2OperationsAdaptiveGroups(limit: 10000, filter: {datetime_geq: $s, datetime_leq: $e}) {
    sum { requests } dimensions { actionType bucketName datetimeHour } } } } }"""


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def cloudflare():
    """R2 operations by bucket (month to date, last 24 h, GetObject per hour for 8 h) and data.anita.garden's
    cache statuses for 24 h. None without a token; parts the token may not read come back as an error string."""
    token = cf_token()
    if not token:
        return None
    out = {}
    acct = cf_get(token, "/accounts")["result"][0]["id"]
    now = dt.datetime.utcnow().replace(microsecond=0)
    month0 = now.replace(day=1, hour=0, minute=0, second=0)
    day0 = now - dt.timedelta(hours=24)
    mtd, last24, hourly = {}, {}, {}
    # The adaptive groups answer at most a few days per query, so the month goes in 3-day steps.
    s = month0
    while s < now:
        e = min(now, s + dt.timedelta(days=3))
        for g in cf_gql(token, R2_OPS_Q, {"a": acct, "s": iso(s), "e": iso(e)})["viewer"]["accounts"][0]["r2OperationsAdaptiveGroups"]:
            k, n = g["dimensions"], g["sum"]["requests"]
            cls = "A" if k["actionType"] in CLASS_A else None if k["actionType"] in FREE_OPS else "B"
            if not cls:
                continue
            b = k["bucketName"] or "(account)"
            mtd.setdefault(b, {"A": 0, "B": 0})[cls] += n
            h = dt.datetime.strptime(k["datetimeHour"], "%Y-%m-%dT%H:%M:%SZ")
            if h >= day0.replace(minute=0, second=0):
                last24.setdefault(b, {"A": 0, "B": 0})[cls] += n
            if b == "anitamaps" and k["actionType"] == "GetObject" and h >= now - dt.timedelta(hours=9):
                hourly[k["datetimeHour"]] = hourly.get(k["datetimeHour"], 0) + n
        s = e
    out["mtd"], out["last24"], out["hourly"] = mtd, last24, dict(sorted(hourly.items()))
    try:
        zone = cf_get(token, "/zones?name=anita.garden")["result"][0]["id"]
        q = """query($z: String!, $s: Time!, $e: Time!) { viewer { zones(filter: {zoneTag: $z}) {
          httpRequestsAdaptiveGroups(limit: 100, filter: {datetime_geq: $s, datetime_leq: $e, clientRequestHTTPHost: "data.anita.garden"}) {
            count dimensions { cacheStatus } } } } }"""
        st = {}
        for g in cf_gql(token, q, {"z": zone, "s": iso(day0), "e": iso(now)})["viewer"]["zones"][0]["httpRequestsAdaptiveGroups"]:
            st[g["dimensions"]["cacheStatus"]] = st.get(g["dimensions"]["cacheStatus"], 0) + g["count"]
        out["cache"] = st
    except Exception as e:
        out["cache"] = f"not readable ({e})"
    return out


# ── the web side ────────────────────────────────────────────────────────────────────────────────────────────
def head(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return {k.lower(): v for k, v in r.headers.items()}


def cert_days(host):
    ctx = ssl.create_default_context()
    with socket.create_connection((host, 443), timeout=20) as s, ctx.wrap_socket(s, server_hostname=host) as t:
        end = ssl.cert_time_to_seconds(t.getpeercert()["notAfter"])
    return (end - time.time()) / 86400


def domain_days(name):
    req = urllib.request.Request(f"https://rdap.org/domain/{name}", headers=UA)
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    for e in d.get("events", []):
        if e.get("eventAction") == "expiration":
            return (dt.datetime.fromisoformat(e["eventDate"].replace("Z", "+00:00")).timestamp() - time.time()) / 86400
    return None


def website_repo():
    url = subprocess.run(["git", "-C", WEBSITE, "remote", "get-url", "origin"], capture_output=True, text=True).stdout
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?\s*$", url)
    return m.group(1) if m else None


def site_bytes():
    r = subprocess.run(["git", "-C", WEBSITE, "ls-tree", "-r", "-l", "HEAD"], capture_output=True, text=True)
    tot = 0
    for line in r.stdout.splitlines():
        p = line.split(None, 4)
        if len(p) == 5 and p[3].isdigit() and not p[4].split("/")[0].startswith("_"):
            tot += int(p[3])
    return tot


def newest_wasm():
    d = os.path.join(WEBSITE, "trainsworld", "assets")
    try:
        files = [f for f in os.listdir(d) if f.endswith(".wasm")]
    except OSError:
        return None
    return max(files, key=lambda f: os.path.getmtime(os.path.join(d, f))) if files else None


# ── rows ────────────────────────────────────────────────────────────────────────────────────────────────────
class Row:
    def __init__(self, group, name, used, limit, unit="", growth=None, note="", key=None, missing=False):
        self.group, self.name, self.used, self.limit, self.unit = group, name, used, limit, unit
        self.growth, self.note, self.key, self.missing = growth, note, key or f"{group}/{name}", missing

    def pct(self):
        return None if not self.limit or self.used is None else 100.0 * self.used / self.limit

    def days80(self):
        if not self.growth or self.growth <= 0 or not self.limit or self.used is None:
            return None
        return max(0.0, (0.8 * self.limit - self.used) / self.growth)

    def flag(self):
        p, d = self.pct(), self.days80()
        if self.missing:
            return "MISSING"
        if p is not None and p >= 100:
            return "OVER"
        if p is not None and p >= 50:
            return "HIGH"
        if d is not None and d <= 30:
            return "SOON"
        return ""


def fmt(v, unit):
    if v is None:
        return "?"
    if unit == "B":
        for u, s in (("TB", 1 << 40), ("GB", GB), ("MB", MB), ("KB", 1 << 10)):
            if abs(v) >= s:
                return f"{v / s:.1f} {u}"
        return f"{v:.0f} B"
    if unit == "load":
        return f"{v:.2f}"
    return f"{v:,.0f}"


def growth_from_history(key, used, now, history):
    """Per day, from the oldest run between 1 and 7 days ago."""
    best = None
    for h in history:
        age = (now - h["t"]) / 86400
        if 1 <= age <= 7 and key in h["used"] and (best is None or h["t"] < best["t"]):
            best = h
    if best is None or used is None:
        return None
    return (used - best["used"][key]) / ((now - best["t"]) / 86400)


def last_full_days(per_day, n=2):
    """Mean of the last n complete UTC days in a {YYYY-MM-DD: count} map (today left out)."""
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    days = sorted(k for k in per_day if k < today)[-n:]
    return sum(per_day[k] for k in days) / len(days) if days else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-r2", action="store_true", help="skip the R2 bucket listing")
    ap.add_argument("--json", help="also write the rows here")
    ap.add_argument("--no-history", action="store_true", help="do not append this run to the history file")
    a = ap.parse_args()
    now = time.time()
    lim, lim_src = code_limits()
    rows, errors = [], []

    try:
        box = probe_box()
    except Exception as e:
        box = None
        errors.append(f"box probe failed: {e}")

    def plays_day():
        if not box:
            return None, None
        days = box["play"]["days"]
        return (last_full_days({k: v["plays"] for k, v in days.items()}),
                last_full_days({k: v["new"] for k, v in days.items()}))
    plays, new_browsers = plays_day()

    if box:
        d, m = box["disk"], box["mem"]
        games_growth = None
        log = [(dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc).timestamp(), b) for t, b in box["games_log"]]
        old = [x for x in log if now - x[0] >= 86400]
        games_used = box["stores"]["trainworld/games"][0]
        if old:
            t0, b0 = max(old)
            games_growth = (games_used - b0) / ((now - t0) / 86400)
        rows.append(Row("box", "disk", d["used"], d["size"], "B", games_growth, f"{fmt(d['avail'], 'B')} free; grows with the games store"))
        rows.append(Row("box", "inodes", d["inodes"], d["inodes_max"]))
        rows.append(Row("box", "memory in use", m["total"] - m["avail"], m["total"], "B", None,
                        f"shonei process {fmt(box['service'].get('rss'), 'B')}, swap {fmt(m['swap'], 'B')}"))
        rows.append(Row("box", "load (15 min)", box["load"][2], box["nproc"], "load", None, f"{box['nproc']} vCPU"))
        rows.append(Row("box", "open files (shonei)", box["service"].get("fds"), box["service"].get("fd_max")))
        rows.append(Row("box", "journal", box["journal_bytes"], None, "B", None,
                        f"journald trims it at {fmt(box['journal_max'], 'B')} (counted in disk), never fills"))
        tx = box["tx_days"]
        today = dt.datetime.utcnow()
        dim = ((today.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)).day
        if tx:
            mtd = sum(tx.values()) * 1.0
            full = [v for k, v in tx.items() if int(k) < today.day] or list(tx.values())
            daily = sum(full[-3:]) / len(full[-3:])
            proj = mtd - tx.get(str(today.day), tx.get(today.day, 0)) + daily * (dim - today.day + 1)
            rows.append(Row("box", "traffic out, month (projected)", proj, HETZNER_TRAFFIC, "B", None,
                            f"{fmt(daily, 'B')}/day lately; Hetzner bills only outgoing"))
        else:
            rows.append(Row("box", "traffic out, month", None, HETZNER_TRAFFIC, "B", None, "no sysstat data"))
        rows.append(Row("box", "service", None, None, "", None,
                        f"{box['service'].get('active')}, up {((now - box['service'].get('started', now)) / 3600):.1f} h, box up {box['uptime_days']:.0f} days",
                        missing=box["service"].get("active") != "active"))

        s = box["stores"]
        gpa = box["games_per_account"]
        rows.append(Row("server", "trainworld games, all", games_used, lim["MaxTwGamesBytes"], "B", games_growth,
                        f"{box['games_accounts']:,} accounts; alert at {fmt(lim['twGamesWarnAt'], 'B')}; per account median "
                        f"{fmt(gpa['median'], 'B')}, max {fmt(gpa['max'], 'B')} of {fmt(lim['MaxTwAccountBytes'], 'B')}"))
        rows.append(Row("server", "trainworld games, biggest account", gpa["max"], lim["MaxTwAccountBytes"], "B"))
        nori = box["nori"]
        nori_growth = last_full_days(nori["per_day"])
        k = nori["kinds"]
        rows.append(Row("server", "accounts (noritetsu + game)", nori["accounts"], lim["MaxNoriAccounts"], "", nori_growth,
                        f"{k.get('google', 0):,} Google, {k.get('pw', 0):,} password"))
        pw_share = k.get("pw", 0) / max(1, nori["accounts"])
        rows.append(Row("server", "password accounts", k.get("pw", 0), lim["MaxNoriPasswordAccounts"], "",
                        nori_growth * pw_share if nori_growth else None))
        rows.append(Row("server", "sign-ups, busiest hour (7 days)", nori["peak_hour"], lim["noriSignupsPerHour"], "", None,
                        "global bucket; past it sign-ups get 'busy'"))
        rows.append(Row("server", "noritetsu ride saves", s["noritetsu/saves"][0], lim["MaxNoriStoreBytes"], "B"))
        rows.append(Row("server", "leaderboard scores", s["trainworld/scores"][0], lim["MaxTwStoreBytes"], "B"))
        cm = box["community"]
        rows.append(Row("server", "community shares", cm["shares"], lim["MaxTwShares"], "", last_full_days(cm["per_day"])))
        rows.append(Row("server", "community bytes", s["trainworld/community"][0], lim["MaxTwCommunityBytes"], "B"))
        pd_ = box["play"]["days"]
        rows.append(Row("server", "browsers tracked (play stats)", box["play"]["browsers"], lim["ptMaxBrowsers"], "", new_browsers,
                        "past it new browsers go uncounted in stats; plays still count"))
        rows.append(Row("server", "plays in a day (busiest of 7)", max((v["plays"] for v in pd_.values()), default=0),
                        lim["ptMaxPlaysDay"], "", None, f"{plays or 0:,.0f}/day lately"))
        rows.append(Row("server", "browsers tracked (achievements)", box["ach_browsers"], lim["achMaxBrowsers"]))
        rows.append(Row("server", "feedback store", s["feedback"][0], lim["MaxFeedbackStoreBytes"], "B"))
        rows.append(Row("server", "agent guestbook", s["agentchat"][0], lim["MaxAgentChatStoreBytes"], "B"))
        rows.append(Row("server", "shonei accounts", box["shonei_accounts"], lim["MaxAccounts"]))
        for g, v in box["lb"].items():
            rows.append(Row("server", f"{g} play log", max(v["plays"], v["scores"]), lim["LBMaxLog"], "", None, "oldest trimmed past it"))
        ev = box["events_24h"]
        rows.append(Row("server", "refusals in the journal (24 h)", None, None, "", None,
                        ", ".join(f"{k} {v}" for k, v in sorted(ev.items())) or "none",
                        missing=any(ev.get(k) for k in ("signup_refused", "budget", "full", "panic", "broken"))))

        ml = box["mail"]
        day_cap = max(ml["quota_hit_at"].values()) if ml["quota_hit_at"] else RESEND_DAY
        busiest = max(ml["per_day"].values(), default=0)
        mail_daily = last_full_days(ml["per_day"])
        rows.append(Row("mail", "Resend, busiest day (8 days)", busiest, day_cap, "", None,
                        ("quota hit on " + ", ".join(f"{k} after {v}" for k, v in sorted(ml["quota_hit_at"].items())))
                        if ml["quota_hit_at"] else f"free plan figure {RESEND_DAY}/day"))
        left = dim - today.day + 1
        mproj = box["mail_month"] + (mail_daily or 0) * left
        rows.append(Row("mail", "Resend, month (projected)", mproj, RESEND_MONTH, "", None,
                        f"{box['mail_month']:,} sent since the 1st, {mail_daily or 0:,.0f}/day lately"))
        bk = box["backup"]
        rows.append(Row("backup", "off-site copy to R2 (T-2918)", None, None, "", None,
                        f"timer {bk['offsite_timer']}, env file {'there' if bk['offsite_env'] else 'missing'}; {bk['offsite_last'] or 'never ran'}",
                        missing=bk["offsite_timer"] != "active"))
        rows.append(Row("backup", "Hetzner daily images", None, None, "", None, "not readable from here: Cloud Console, server, Backups"))
        rows.append(Row("google", "Google sign-in", None, None, "", None,
                        ("on at the last start" if box["google_on"] else "NOT on at the last start") + "; no user cap once In production",
                        missing=not box["google_on"]))

    # R2
    if not a.no_r2:
        try:
            top, versions = r2_listing()
            total = sum(v[0] for v in top.values())
            rows.append(Row("r2", "anitamaps storage", total, R2_FREE_STORAGE, "B", None,
                            "past the free 10 GB it bills $0.015/GB-month: " + ", ".join(f"{k} {fmt(v[0], 'B')}" for k, v in sorted(top.items(), key=lambda x: -x[1][0])[:4])))
            dated = sorted(v for v in versions if DATED.match(v))
            vbytes = sum(versions[v][0] for v in dated)
            rows.append(Row("r2", "trainworld data versions", len(dated), None, "", None,
                            f"{fmt(vbytes, 'B')}; oldest {dated[0] if dated else '-'}; r2_purge.py keeps 3"))
            month = dt.datetime.utcnow().strftime("%Y-%m")
            uploads = sum(versions[v][1] for v in dated if v.startswith(month))
        except Exception as e:
            errors.append(f"R2 listing failed: {e}")

    # R2 operations and the data host's cache, from Cloudflare's analytics (the ops token)
    cf = None
    try:
        cf = cloudflare()
    except Exception as e:
        errors.append(f"Cloudflare analytics: {e}")
    today = dt.datetime.utcnow()
    dim = ((today.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)).day
    left = dim - today.day + (24 - today.hour) / 24.0  # days left in the month, today's rest included
    if cf:
        tot = {c: sum(v[c] for v in cf["mtd"].values()) for c in "AB"}
        day = {c: sum(v[c] for v in cf["last24"].values()) for c in "AB"}
        by = lambda c: ", ".join(f"{b} {v[c]:,}" for b, v in sorted(cf["last24"].items(), key=lambda x: -x[1][c]) if v[c])
        for c, free, price in (("A", R2_FREE_A, 4.50), ("B", R2_FREE_B, 0.36)):
            proj = tot[c] + day[c] * left
            cost = max(0.0, proj - free) / 1e6 * price
            rows.append(Row("r2", f"Class {c}, month (projected)", proj, free, "", None,
                            f"{tot[c]:,} so far; last 24 h {day[c]:,} ({by(c) or 'none'}); about ${cost:,.2f} past the free tier"))
        hours = list(cf["hourly"].items())[:-1]  # the current hour is partial
        if hours:
            rows.append(Row("r2", "anitamaps GetObject by hour", None, None, "", None,
                            "  ".join(f"{h[11:13]}h {n / 1e6:.2f}M" for h, n in hours[-8:])))
        st = cf["cache"]
        if isinstance(st, dict) and st:
            n = sum(st.values())
            hit = sum(v for k, v in st.items() if k in ("hit", "stale", "revalidated", "updating"))
            rows.append(Row("cache", "data.anita.garden hits, 24 h", hit, n, "", None,
                            ", ".join(f"{k} {v:,}" for k, v in sorted(st.items(), key=lambda x: -x[1]))))
        else:
            rows.append(Row("cache", "data.anita.garden hits", None, None, "", None, str(st)))
    elif not a.no_r2 and "uploads" in locals():
        rows.append(Row("r2", "Class A this month (estimate)", uploads, R2_FREE_A, "", None,
                        "no Cloudflare token: objects in this month's data versions still on R2"))
        if plays:
            rows.append(Row("r2", "Class B a month (estimate)", plays * R2_READS_PER_PLAY * dim, R2_FREE_B, "", None,
                            f"no Cloudflare token: {plays:,.0f} plays/day x {R2_READS_PER_PLAY} reads on a first load, a lower bound"))
    # A browser that may not store the game's data asks the edge again on every visit.
    try:
        # A GET for one byte: a HEAD skips the cache and the header the cache adds.
        req = urllib.request.Request("https://data.anita.garden/trainworld/terrain-world/v1/index.bin", headers={"Range": "bytes=0-0", **UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            cc, cs = r.headers.get("cache-control", ""), r.headers.get("cf-cache-status", "?")
        rows.append(Row("cache", "data.anita.garden browser caching", None, None, "", None, f"Cache-Control: {cc} ({cs})",
                        missing="no-store" in cc or "no-cache" in cc))
    except Exception as e:
        errors.append(f"data cache headers: {e}")

    # GitHub Pages
    try:
        repo = website_repo()
        if repo:
            kb = int(subprocess.run(["gh", "api", f"repos/{repo}", "--jq", ".size"], capture_output=True, text=True, timeout=60).stdout.strip())
            rows.append(Row("pages", "website repo on GitHub", kb * 1024, PAGES_REPO, "B", None, "5 GB is GitHub's firm line"))
    except Exception as e:
        errors.append(f"gh api: {e}")
    try:
        rows.append(Row("pages", "published site (tracked files)", site_bytes(), PAGES_SITE, "B"))
    except Exception as e:
        errors.append(f"site size: {e}")
    try:
        wasm = newest_wasm()
        status = head(f"https://anita.garden/trainsworld/assets/{wasm}").get("cf-cache-status", "?") if wasm else "?"
        if new_browsers is not None:
            per = PAGES_BYTES_PER_NEW_BROWSER if status in ("DYNAMIC", "BYPASS", "MISS", "?") else 0.1 * MB
            rows.append(Row("pages", "bandwidth a month (estimate)", new_browsers * per * dim, PAGES_BANDWIDTH, "B", None,
                            f"game wasm cf-cache-status {status}; {new_browsers:,.0f} new browsers/day; soft limit"))
    except Exception as e:
        errors.append(f"pages cache check: {e}")

    # certificates and the domain
    for host in ("market.anita.garden", "anita.garden", "data.anita.garden"):
        try:
            dl = cert_days(host)
            rows.append(Row("certs", host, None, None, "", None, f"{dl:.0f} days left", missing=dl < 14))
        except Exception as e:
            errors.append(f"cert {host}: {e}")
    try:
        dd = domain_days("anita.garden")
        rows.append(Row("domain", "anita.garden renewal", None, None, "", None, f"{dd:.0f} days left", missing=dd is not None and dd < 60))
    except Exception as e:
        errors.append(f"domain: {e}")

    # history-based growth for the rows that have none yet
    history = []
    try:
        history = [json.loads(l) for l in open(HISTORY, encoding="utf-8") if l.strip()]
    except OSError:
        pass
    for r in rows:
        g = growth_from_history(r.key, r.used, now, history)
        if g is not None:
            r.growth = g

    # print
    w = max(len(f"{r.group} / {r.name}") for r in rows) if rows else 30
    print(f"resources at {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC  (flags: OVER >=100%, HIGH >=50%, SOON = 80% within 30 days, MISSING)")
    for r in rows:
        p, d80 = r.pct(), r.days80()
        use = f"{fmt(r.used, r.unit)} / {fmt(r.limit, r.unit)}" if r.limit else (fmt(r.used, r.unit) if r.used is not None else "")
        pct = f"{p:5.1f}%" if p is not None else "      "
        gr = f"+{fmt(r.growth, r.unit)}/day" if r.growth else ""
        eta = f"80% in {d80:.0f} d" if d80 is not None and d80 < 365 else ""
        print(f"{r.flag():7} {r.group + ' / ' + r.name:<{w}}  {use:>22} {pct}  {gr:>14} {eta:>12}  {r.note}")
    for e in errors:
        print("ERROR  ", e)
    flagged = [r for r in rows if r.flag()]
    print(f"\n{len(flagged)} flagged of {len(rows)}; limits from shonei-server: "
          + ", ".join(sorted({v.split()[0] for v in lim_src.values()})))

    rec = {"t": now, "used": {r.key: r.used for r in rows if r.used is not None}}
    if not a.no_history:
        os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
        with open(HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump([{**r.__dict__, "pct": r.pct(), "days80": r.days80(), "flag": r.flag()} for r in rows], f, indent=1)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
