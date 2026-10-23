"""T-1064: the one way this project talks to AWS (docs/aws.md). The account is a friend's, shared
free, and the IAM user `anita` has AdministratorAccess on all of it, so nothing in AWS stops a call
from touching her things: every guard is here. Every call goes to REGION with PROFILE, every
resource made carries TAGS, and every change names ids that were first checked to carry TAGS.

Run the scripts beside this file with the maps venv (botocore is there, for presigned URLs):
    venv\\Scripts\\python.exe trainworld\\tools\\aws\\status.py
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REGION = "us-east-2"            # Ohio. Never another: status --all-regions finds strays
PROFILE = "default"
ACCOUNT = "429583250612"
USER_ARN = f"arn:aws:iam::{ACCOUNT}:user/anita"
TAGS = {"owner": "anita", "project": "anitabuilder"}
BUCKET = "anitabuilder-builds-use2-k7q4"   # ours alone; the friend's buckets are never named here
SG_NAME = "anitabuilder-runner"            # no inbound rules: the machine talks out only
MAX_HOURS = 12                  # the hard timer launch.py accepts at most
MAX_PRICE_CAP = 2.0             # $/h: launch.py refuses a spot bid above this
BURSTABLE = re.compile(r"^t\d")  # t2/t3/t4g "unlimited" credits bill extra under load: refused
RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")

HERE = Path(__file__).resolve().parent
TW = HERE.parents[1]
MAPS = TW.parent
STATE = MAPS / "artifacts" / "trainworld" / "aws"   # run records, redacted user data, fetched results
INBOX = STATE / "inbox"                               # results land here, never in data/packs

# Credentials come from the CLI profile only: an R2 key or another account's in the environment
# would otherwise win (botocore and the CLI read AWS_* first).
for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)


class Refused(SystemExit):
    """A guard said no. The message says why and what to do instead."""
    def __init__(self, msg):
        super().__init__("REFUSED: " + msg)


def aws_exe():
    exe = shutil.which("aws")
    if exe:
        return exe
    for p in (r"C:\Program Files\Amazon\AWSCLIV2\aws.exe", "/usr/local/bin/aws", "/usr/bin/aws"):
        if os.path.exists(p):
            return p
    raise SystemExit("aws CLI not found (docs/aws.md: reload PATH from the registry)")


def aws(*args, text=False, check=True):
    """One CLI call in REGION with PROFILE; JSON back (or text). Never prints credentials."""
    cmd = [aws_exe(), *[str(a) for a in args], "--region", REGION, "--profile", PROFILE, "--no-cli-pager"]
    if not text:
        cmd += ["--output", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        if check:
            raise SystemExit(f"aws {' '.join(str(a) for a in args[:2])} failed: {p.stderr.strip()[-800:]}")
        return None
    if text:
        return p.stdout
    return json.loads(p.stdout) if p.stdout.strip() else {}


_checked = False


def preflight():
    """The credentials are anita's in the friend's account, or nothing happens."""
    global _checked
    if _checked:
        return
    me = aws("sts", "get-caller-identity")
    if me.get("Arn") != USER_ARN or me.get("Account") != ACCOUNT:
        raise Refused(f"credentials are {me.get('Arn')}, not {USER_ARN}")
    _checked = True


def tag_list(extra=None):
    t = dict(TAGS, **(extra or {}))
    return [{"Key": k, "Value": v} for k, v in t.items()]


def tag_filters():
    return [f"Name=tag:{k},Values={v}" for k, v in TAGS.items()]


def is_ours(tags):
    d = {t["Key"]: t["Value"] for t in (tags or [])}
    return all(d.get(k) == v for k, v in TAGS.items())


def tagdict(tags):
    return {t["Key"]: t["Value"] for t in (tags or [])}


# ---------------------------------------------------------------- EC2 lookups, ours only

LIVE = ["pending", "running", "stopping", "stopped", "shutting-down"]


def instances(states=None, region=None):
    args = ["ec2", "describe-instances", "--filters", *tag_filters()]
    if states:
        args.append("Name=instance-state-name,Values=" + ",".join(states))
    if region and region != REGION:
        return _other_region(region, args)
    out = aws(*args)
    return [i for r in out.get("Reservations", []) for i in r["Instances"] if is_ours(i.get("Tags"))]


def _other_region(region, args):
    """A read-only lookup in another region (status --all-regions); never a change."""
    cmd = [aws_exe(), *args, "--region", region, "--profile", PROFILE, "--output", "json", "--no-cli-pager"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        return [{"error": p.stderr.strip()[-200:]}]
    out = json.loads(p.stdout or "{}")
    return [i for r in out.get("Reservations", []) for i in r["Instances"] if is_ours(i.get("Tags"))]


def require_ours(kind, rid):
    """Look the id up again and refuse unless it carries TAGS: every change goes through this."""
    if kind == "instance":
        out = aws("ec2", "describe-instances", "--instance-ids", rid)
        res = [i for r in out.get("Reservations", []) for i in r["Instances"]]
    elif kind == "volume":
        res = aws("ec2", "describe-volumes", "--volume-ids", rid).get("Volumes", [])
    elif kind == "snapshot":
        res = aws("ec2", "describe-snapshots", "--owner-ids", "self", "--snapshot-ids", rid).get("Snapshots", [])
    elif kind == "image":
        res = aws("ec2", "describe-images", "--owners", "self", "--image-ids", rid).get("Images", [])
    elif kind == "address":
        res = aws("ec2", "describe-addresses", "--allocation-ids", rid).get("Addresses", [])
    elif kind == "sg":
        res = aws("ec2", "describe-security-groups", "--group-ids", rid).get("SecurityGroups", [])
    elif kind == "spot":
        res = aws("ec2", "describe-spot-instance-requests", "--spot-instance-request-ids", rid).get("SpotInstanceRequests", [])
    else:
        raise ValueError(kind)
    if len(res) != 1 or not is_ours(res[0].get("Tags")):
        raise Refused(f"{kind} {rid} does not carry owner=anita, project=anitabuilder; it is not ours, left alone")
    return res[0]


def live_spot_requests():
    """Our open or active spot requests. A filtered listing lags a cancel by a minute or so, while a
    lookup by id does not, so each one the listing shows is looked up again by id."""
    listed = [s["SpotInstanceRequestId"] for s in
              aws("ec2", "describe-spot-instance-requests", "--filters", *tag_filters()).get("SpotInstanceRequests", [])
              if s["State"] in ("open", "active")]
    if not listed:
        return []
    out = aws("ec2", "describe-spot-instance-requests", "--spot-instance-request-ids", *listed)
    return [s for s in out.get("SpotInstanceRequests", []) if s["State"] in ("open", "active") and is_ours(s.get("Tags"))]


def leftovers():
    """Everything tagged ours in REGION that costs or could cost: the teardown's proof."""
    f = tag_filters()
    out = {
        "instances": [i for i in instances() if i["State"]["Name"] != "terminated"],
        "spot_requests": live_spot_requests(),
        "volumes": aws("ec2", "describe-volumes", "--filters", *f).get("Volumes", []),
        "snapshots": aws("ec2", "describe-snapshots", "--owner-ids", "self", "--filters", *f).get("Snapshots", []),
        "images": aws("ec2", "describe-images", "--owners", "self", "--filters", *f).get("Images", []),
        "addresses": aws("ec2", "describe-addresses", "--filters", *f).get("Addresses", []),
        "security_groups": aws("ec2", "describe-security-groups", "--filters", *f).get("SecurityGroups", []),
    }
    return out


# ---------------------------------------------------------------- the bucket

def bucket_is_ours():
    """True if BUCKET exists and carries TAGS; False if it does not exist; Refused if it is not ours."""
    head = aws("s3api", "head-bucket", "--bucket", BUCKET, check=False)
    if head is None:
        return False
    tags = aws("s3api", "get-bucket-tagging", "--bucket", BUCKET, check=False)
    if not tags or not is_ours(tags.get("TagSet")):
        raise Refused(f"bucket {BUCKET} exists without our tags: not ours, left alone")
    return True


def check_key(key):
    """Keys are <run>/<name>: nothing outside a run's own prefix is written or removed."""
    run = key.split("/", 1)[0]
    if "/" not in key or not RUN_NAME.match(run) or ".." in key or key.endswith("/"):
        raise Refused(f"key {key!r}: keys are <run>/<file>, run matching {RUN_NAME.pattern}")
    return key


def bucket_objects(prefix=""):
    if not bucket_is_ours():
        return []
    out = aws("s3api", "list-objects-v2", "--bucket", BUCKET, *(["--prefix", prefix] if prefix else []))
    return out.get("Contents", []) or []


def presign(method, key, hours):
    """A presigned GET or PUT URL for one key of BUCKET, so the machine needs no credentials of its
    own (no IAM role made in the friend's account). The URL carries the access key id (not the
    secret) and a signature good for this key only, until it expires."""
    check_key(key)
    import botocore.session
    from botocore.config import Config
    s = botocore.session.Session(profile=PROFILE)
    c = s.create_client("s3", region_name=REGION,
                        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
    op = {"GET": "get_object", "PUT": "put_object"}[method]
    return c.generate_presigned_url(op, Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=int(hours * 3600))


# ---------------------------------------------------------------- run records

def now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def at(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def parse(s):
    return datetime.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)


def record_path(run):
    return STATE / f"{run}.json"


def load_record(run):
    p = record_path(run)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_record(run, rec):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = record_path(run).with_suffix(".part")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, record_path(run))


def say(*a):
    print(*a, flush=True)
