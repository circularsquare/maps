"""T-4166: timing runs on Anita's own AWS account, for when this PC's cores are busy. docs/aws.md "Her own account".

    bench.py setup                 once: the bucket (private, tagged, objects expire) and a security group with no inbound rules
    bench.py run [--commit REV] [--base REV] [--perf SAVES] [--edit SAVES] [--smoke] [--runs N] [--hours H] [--dry]
    bench.py status                anything up or stored in the account
    bench.py teardown <run>        terminate the run's machine if it is still up and delete the run's objects

run: one 4-vCPU machine (spot, else on-demand, never over PRICE_CAP an hour) gets the commit's code (git archive, a few
MB) and the data its saves read (a fixed list, benchdata/<save>.txt beside this file, cached in the bucket by content),
builds the game (Rust and Vite), runs tools/perf_bench.mjs (compare when --base is given), tools/edit_bench.mjs and/or
checks/smoke.mjs, sends the results back, and shuts itself down. The results land in artifacts/trainworld/awsbench/<run>/;
then this terminates the machine if it is still there, deletes the run's objects and prints the run's cost.

Separate from the friend's account on purpose (no longer used, T-4177): its own profile, account number, tag and
bucket; nothing from awslib.py; it refuses to run if the profile turns out to be the friend's account.
"""
import datetime
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)  # the profile alone decides the account (an R2 key in the environment would win otherwise)

PROFILE = "anita-own"
ACCOUNT = "537239093801"
FRIEND_ACCOUNT = "429583250612"           # never this one (docs/aws.md)
USER_ARN = f"arn:aws:iam::{ACCOUNT}:user/bench"
REGION = "us-east-2"
TAG = {"project": "twbench"}               # every resource made here carries it; nothing names a person
BUCKET = "twbench-use2-r7k2m9"             # ours alone, in her account
SG_NAME = "twbench-runner"                 # no inbound rules: the machine talks out only
TYPE = "m8a.xlarge"                        # 4 vCPUs (one physical core each), 16 GB
# on-demand $/h, Ohio, 2026-11-23 (the public price list); the .large ones (2 vCPUs) only while the account's quota is
# under 4 vCPUs: their numbers are a check that the flow works, not a bench (perf_bench wants 3 cores)
TYPES = {"m8a.xlarge": 0.24344, "m7a.xlarge": 0.23184, "c8a.xlarge": 0.21554, "m8a.large": 0.12172, "c8a.large": 0.10777, "m7a.large": 0.11592}
PRICE_CAP = 0.25                           # $/h: no machine above it, spot or on-demand
MAX_HOURS = 3                              # the hard timer at most
DISK_GB = 30                               # gp3, deleted with the machine
EBS_GB_MONTH = 0.08                        # gp3 $/GB-month
IPV4_HOUR = 0.005                          # the public IPv4 address the machine needs to reach the internet without a NAT
UBUNTU = ("099720109477", "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*")  # Canonical's account, name
EXPIRE_DAYS = {"runs/": 3, "data/": 45}    # the bucket's lifecycle: a forgotten object goes by itself

HERE = Path(__file__).resolve().parent
TW = HERE.parents[1]
MAPS = TW.parent
STATE = MAPS / "artifacts" / "trainworld" / "awsbench"
LIVE = ["pending", "running", "stopping", "stopped", "shutting-down"]
GAME_PATHS = ["trainworld/app", "trainworld/sim"]   # from the commit
TOOL_PATHS = ["cores.py", "trainworld/tools", "trainworld/checks", "trainworld/T-007-real-nyc.save"]  # from the main tree
SAVES = ("nyc",)                           # saves with a data list (benchdata/<save>.txt)


class Refused(SystemExit):
    def __init__(self, msg):
        super().__init__("REFUSED: " + msg)


def say(*a):
    print(*a, flush=True)


def now():
    return datetime.datetime.now(datetime.timezone.utc)


# ---------------------------------------------------------------------------------------------- the account

_clients = {}


def client(name):
    if name not in _clients:
        import botocore.session
        from botocore.config import Config
        s = botocore.session.Session(profile=PROFILE)
        cfg = Config(signature_version="s3v4", s3={"addressing_style": "virtual"}, retries={"max_attempts": 5}) if name == "s3" else None
        _clients[name] = s.create_client(name, region_name=REGION, config=cfg)
    return _clients[name]


def preflight():
    """The profile is the bench user of her own account, or nothing happens."""
    me = client("sts").get_caller_identity()
    if me["Account"] == FRIEND_ACCOUNT:
        raise Refused(f"profile {PROFILE} is the friend's account ({FRIEND_ACCOUNT}); this script is for her own only")
    if me["Account"] != ACCOUNT or me["Arn"] != USER_ARN:
        raise Refused(f"profile {PROFILE} is {me['Arn']}, not {USER_ARN}")


def tags(extra=None):
    return [{"Key": k, "Value": v} for k, v in dict(TAG, **(extra or {})).items()]


def is_ours(t):
    d = {x["Key"]: x["Value"] for x in (t or [])}
    return all(d.get(k) == v for k, v in TAG.items())


def tagdict(t):
    return {x["Key"]: x["Value"] for x in (t or [])}


def instances(states=LIVE):
    out = client("ec2").describe_instances(Filters=[{"Name": "instance-state-name", "Values": states}])
    return [i for r in out["Reservations"] for i in r["Instances"]]


def our_instance(iid):
    """Look the id up again; refuse unless it carries our tag."""
    res = [i for r in client("ec2").describe_instances(InstanceIds=[iid])["Reservations"] for i in r["Instances"]]
    if len(res) != 1 or not is_ours(res[0].get("Tags")):
        raise Refused(f"instance {iid} does not carry project={TAG['project']}: left alone")
    return res[0]


# ---------------------------------------------------------------------------------------------- the bucket

def bucket_exists():
    from botocore.exceptions import ClientError
    try:
        client("s3").head_bucket(Bucket=BUCKET, ExpectedBucketOwner=ACCOUNT)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            return False
        raise Refused(f"bucket {BUCKET}: {e} (a name another account owns? pick a new BUCKET)")
    t = client("s3").get_bucket_tagging(Bucket=BUCKET, ExpectedBucketOwner=ACCOUNT)["TagSet"]
    if not is_ours(t):
        raise Refused(f"bucket {BUCKET} has no project={TAG['project']} tag: not ours")
    return True


def check_key(key):
    if not (key.startswith("runs/") or key.startswith("data/")) or ".." in key or key.endswith("/"):
        raise Refused(f"key {key!r}: only runs/<run>/<file> and data/<file>")
    return key


def s3_put(path, key):
    check_key(key)
    with open(path, "rb") as f:
        client("s3").put_object(Bucket=BUCKET, Key=key, Body=f, ExpectedBucketOwner=ACCOUNT)


def s3_has(key):
    from botocore.exceptions import ClientError
    try:
        return client("s3").head_object(Bucket=BUCKET, Key=check_key(key), ExpectedBucketOwner=ACCOUNT)
    except ClientError:
        return None


def s3_get(key):
    return client("s3").get_object(Bucket=BUCKET, Key=check_key(key), ExpectedBucketOwner=ACCOUNT)["Body"].read()


def s3_list(prefix):
    out, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix, "ExpectedBucketOwner": ACCOUNT, **({"ContinuationToken": token} if token else {})}
        r = client("s3").list_objects_v2(**kw)
        out += r.get("Contents", [])
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def s3_rm_run(run):
    """Deletes runs/<run>/ only: never another prefix."""
    keys = [o["Key"] for o in s3_list(f"runs/{run}/")]
    for k in keys:
        check_key(k)
        client("s3").delete_object(Bucket=BUCKET, Key=k, ExpectedBucketOwner=ACCOUNT)
    return len(keys)


def presign(method, key, hours):
    op = {"GET": "get_object", "PUT": "put_object"}[method]
    return client("s3").generate_presigned_url(op, Params={"Bucket": BUCKET, "Key": check_key(key)}, ExpiresIn=int(hours * 3600))


# ---------------------------------------------------------------------------------------------- setup

def setup():
    preflight()
    s3 = client("s3")
    if not bucket_exists():
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
        s3.put_bucket_tagging(Bucket=BUCKET, Tagging={"TagSet": tags()})
        say(f"made bucket {BUCKET}")
    s3.put_public_access_block(Bucket=BUCKET, ExpectedBucketOwner=ACCOUNT, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_lifecycle_configuration(Bucket=BUCKET, ExpectedBucketOwner=ACCOUNT, LifecycleConfiguration={"Rules": [
        {"ID": f"expire-{p.strip('/')}", "Filter": {"Prefix": p}, "Status": "Enabled", "Expiration": {"Days": d},
         "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}} for p, d in EXPIRE_DAYS.items()]})
    say(f"bucket {BUCKET}: private, objects expire ({EXPIRE_DAYS})")
    say(f"security group {sg()} ({SG_NAME}, no inbound rules)")


def sg():
    ec2 = client("ec2")
    got = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "tag:project", "Values": [TAG["project"]]}])["SecurityGroups"]
    if got:
        return got[0]["GroupId"]
    vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
    if len(vpc) != 1:
        raise SystemExit("no default VPC in us-east-2: stop and ask (never make a VPC, NAT or endpoint)")
    g = ec2.create_security_group(GroupName=SG_NAME, VpcId=vpc[0]["VpcId"], Description="twbench runner: no inbound, outbound only",
                                  TagSpecifications=[{"ResourceType": "security-group", "Tags": tags({"Name": SG_NAME})}])
    return g["GroupId"]


# ---------------------------------------------------------------------------------------------- inputs

def git(*a, binary=False):
    p = subprocess.run(["git", "-C", str(MAPS), *a], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if p.returncode:
        raise SystemExit(f"git {' '.join(a[:3])} failed: {p.stderr.decode(errors='replace')[-400:]}")
    return p.stdout if binary else p.stdout.decode().rstrip()


def code_bundle(rev, out):
    """One tree for the machine: the commit's game (app, sim), the main tree's uncommitted app/src/game/*.json as
    checks/agent-tree.mjs copies them (the packs were built for them), and the bench tools as they are in the main tree
    (TOOL_PATHS' tracked files, edits included), as perf_bench build does at home. Returns (sha, json overlaid, tools edited)."""
    sha = git("rev-parse", "--verify", rev + "^{commit}")
    raw = git("archive", "--format=tar", sha, "--", *GAME_PATHS, binary=True)
    dirty = [l[3:] for l in git("status", "--porcelain", "--", "trainworld/app/src/game").splitlines()
             if l[3:].endswith(".json") and l[:2].strip() in ("M", "A", "??")]
    tools = [f for f in git("ls-files", "--", *TOOL_PATHS).splitlines() if (MAPS / f).is_file()]
    edited = [l[3:] for l in git("status", "--porcelain", "--", *TOOL_PATHS).splitlines() if l[:2].strip() == "M"]
    with tarfile.open(out, "w:gz", compresslevel=6) as t:
        with tarfile.open(fileobj=io.BytesIO(raw)) as src:
            for m in src.getmembers():
                if m.name not in dirty:
                    t.addfile(m, src.extractfile(m) if m.isfile() else None)
        for f in dirty + tools:
            t.add(str(MAPS / f), arcname=f)
    return sha, dirty, edited


# what the game asks for (packs/...) -> where it lives under trainworld/data (app/vite.config.ts), longest first
DATA_ROOTS = [("packs/terrain-world/", "terrain-world-v2/"), ("packs/world.water", "terrain/world.water"), ("packs/", "packs/")]


def data_files(saves):
    """(path under trainworld/data, local file) for every line of the saves' data lists, each line a path the game
    asked for in a bench at home (perf_bench's PERF_REQLOG)."""
    want = set()
    for s in saves:
        lst = HERE / "benchdata" / f"{s}.txt"
        if not lst.exists():
            raise SystemExit(f"no data list for {s} ({lst}): run perf_bench at home with PERF_REQLOG to make one (notes/T-4166.md)")
        want |= {l.strip() for l in lst.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
    out = []
    for w in sorted(want):
        rel = next((to + w[len(fr):] for fr, to in DATA_ROOTS if w.startswith(fr)), None)
        if rel is None:
            raise SystemExit(f"data list line {w!r} is not under packs/")
        local = TW / "data" / rel
        if local.is_file():
            out.append((rel, local))
        else:
            say(f"  data list: {w} is not on this PC (left out)")
    return out


def data_bundle(saves, tmp):
    """The data tar, cached in the bucket under its content's name: built and uploaded only when a file changed."""
    files = data_files(saves)
    h = hashlib.sha256()
    for rel, local in files:
        st = local.stat()
        h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    key = f"data/{h.hexdigest()[:20]}.tgz"
    head = s3_has(key)
    if head:
        say(f"data: {len(files)} files, cached as {key} ({head['ContentLength'] / 1e6:.0f} MB)")
        return key, head["ContentLength"]
    out = tmp / "data.tgz"
    with tarfile.open(out, "w:gz", compresslevel=3) as t:
        for rel, local in files:
            t.add(str(local), arcname=rel)
    size = out.stat().st_size
    t0 = time.time()
    s3_put(out, key)
    say(f"data: {len(files)} files, {size / 1e6:.0f} MB uploaded as {key} in {time.time() - t0:.0f} s")
    out.unlink()
    return key, size


# ---------------------------------------------------------------------------------------------- launch

def ubuntu_ami():
    imgs = client("ec2").describe_images(Owners=[UBUNTU[0]], Filters=[{"Name": "name", "Values": [UBUNTU[1]]}, {"Name": "state", "Values": ["available"]}])["Images"]
    if not imgs:
        raise SystemExit("no Ubuntu 24.04 image found")
    img = max(imgs, key=lambda i: i["CreationDate"])
    return img["ImageId"], img["RootDeviceName"], img["Name"]


def spot_prices(itype):
    out = client("ec2").describe_spot_price_history(InstanceTypes=[itype], ProductDescriptions=["Linux/UNIX"], StartTime=now())
    by = {}
    for p in out["SpotPriceHistory"]:
        by.setdefault(p["AvailabilityZone"], float(p["SpotPrice"]))
    return by


def render(run, urls, minutes, deadline):
    tpl = (HERE / "userdata" / "bench.sh").read_text(encoding="utf-8")
    fill = {"{{RUN}}": run, "{{MINUTES}}": str(minutes), "{{DEADLINE_EPOCH}}": str(int(deadline))}
    text, red = tpl, tpl
    for k, v in fill.items():
        text, red = text.replace(k, v), red.replace(k, v)
    text = text.replace("{{URLS}}", "\n".join(f"{k}='{v}'" for k, v in urls.items()))
    red = red.replace("{{URLS}}", "\n".join(f"{k}='<presigned>'" if k.startswith(("GET_", "PUT_")) else f"{k}='{v}'" for k, v in urls.items()))
    if "{{" in text:
        raise SystemExit("a user data placeholder was left unfilled")
    if len(text.encode()) > 16000:
        raise SystemExit("user data over 16 KB")
    return text, red


def launch(run, itype, userdata):
    """Spot in the cheapest zone, then the other zones, then on-demand; never above PRICE_CAP. (instance, market, price)."""
    from botocore.exceptions import ClientError
    ec2 = client("ec2")
    ami, root, ami_name = ubuntu_ami()
    od = TYPES[itype]
    common = dict(ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1, UserData=userdata,
                  InstanceInitiatedShutdownBehavior="terminate", SecurityGroupIds=[sg()],
                  MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
                  BlockDeviceMappings=[{"DeviceName": root, "Ebs": {"VolumeSize": DISK_GB, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                  TagSpecifications=[{"ResourceType": r, "Tags": tags({"Name": f"twbench-{run}", "run": run})} for r in ("instance", "volume", "network-interface")])
    subnets = {s["AvailabilityZone"]: s["SubnetId"] for s in ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"]}
    tries = sorted(((p, az) for az, p in spot_prices(itype).items() if az in subnets), key=lambda x: x[0])
    say(f"{itype}: spot " + ", ".join(f"{az} ${p:.4f}" for p, az in tries) + f"; on-demand ${od}/h; cap ${PRICE_CAP}/h; image {ami_name}")
    errors = []
    for price, az in tries:
        if price > PRICE_CAP:
            continue
        try:
            r = ec2.run_instances(**common, SubnetId=subnets[az], InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {
                "MaxPrice": str(min(PRICE_CAP, od)), "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}})
            return r["Instances"][0], "spot", price
        except ClientError as e:
            blocked(e)
            errors.append(f"spot {az}: {e.response['Error']['Code']} ({e.response['Error'].get('Message', '')[:160]})")
    if od > PRICE_CAP:
        raise SystemExit(f"no spot capacity ({errors}) and on-demand ${od}/h is over the cap")
    for az in sorted(subnets):
        try:
            r = ec2.run_instances(**common, SubnetId=subnets[az])
            say(f"no spot ({'; '.join(errors)}): on-demand")
            return r["Instances"][0], "on-demand", od
        except ClientError as e:
            blocked(e)
            errors.append(f"on-demand {az}: {e.response['Error']['Code']} ({e.response['Error'].get('Message', '')[:160]})")
    raise SystemExit(f"no machine: {'; '.join(errors)}")


def blocked(e):
    code, msg = e.response["Error"]["Code"], e.response["Error"].get("Message", "")
    if code in ("UnauthorizedOperation", "AccessDenied") or "explicit deny" in msg:
        raise SystemExit("AWS refused to start a machine: the $20 monthly cutoff (notes/aws-budget-action.md) is probably "
                         "on. Nothing was started. It lifts when she takes the policy off the bench user.")
    if code in ("VcpuLimitExceeded", "MaxSpotInstanceCountExceeded") and instances():
        raise SystemExit("the account's 5 vCPUs are taken by another machine: one at a time")


# ---------------------------------------------------------------------------------------------- run

def opts(argv):
    o = {"commit": "HEAD", "base": None, "perf": None, "edit": None, "smoke": False, "runs": 1, "hours": 2.0, "dry": False, "type": TYPE}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--smoke", "--dry"):
            o[a[2:]] = True
        elif a in ("--commit", "--base", "--perf", "--edit", "--runs", "--hours", "--type"):
            o[a[2:]] = argv[i + 1]
            i += 1
        else:
            raise SystemExit(f"unknown argument {a}\n{__doc__}")
        i += 1
    if not (o["perf"] or o["edit"] or o["smoke"]):
        o["perf"] = "nyc"
    for k in ("perf", "edit"):
        o[k] = [s for s in (o[k] or "").split(",") if s]
        bad = [s for s in o[k] if s not in SAVES]
        if bad:
            raise SystemExit(f"--{k} {','.join(bad)}: saves with a data list are {', '.join(SAVES)}")
    o["runs"], o["hours"] = int(o["runs"]), float(o["hours"])
    if not 0 < o["hours"] <= MAX_HOURS:
        raise Refused(f"--hours {o['hours']}: above 0, at most {MAX_HOURS}")
    if o["type"] not in TYPES:
        raise Refused(f"--type {o['type']}: one of {', '.join(TYPES)} (4 vCPUs, the account's limit is 5)")
    return o


def run(argv):
    o = opts(argv)
    preflight()
    if not bucket_exists():
        raise Refused("no bucket yet: bench.py setup first")
    live = instances()
    if live:
        raise Refused("a machine is up in the account: " + ", ".join(f"{i['InstanceId']} ({tagdict(i.get('Tags')).get('run', 'untagged')}, {i['State']['Name']})" for i in live)
                      + ". One at a time (the account has 5 vCPUs); bench.py teardown <run> if its run is over")
    name = "b" + now().strftime("%Y%m%d-%H%M%S")
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / (name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    saves = sorted(set(o["perf"]) | set(o["edit"]) | ({"nyc"} if o["smoke"] else set()))
    head_sha, head_json, edited = code_bundle(o["commit"], tmp / "head.tgz")
    base_sha = code_bundle(o["base"], tmp / "base.tgz")[0] if o["base"] else None
    say(f"run {name}: head {head_sha[:10]}" + (f" (with the main tree's {', '.join(head_json)})" if head_json else "")
        + (f"; tools with the main tree's edits to {', '.join(edited)}" if edited else "")
        + (f", base {base_sha[:10]}" if base_sha else "") + f"; perf {o['perf'] or '-'}, edit {o['edit'] or '-'}, smoke {'yes' if o['smoke'] else 'no'}")
    jobs = " ".join([*(f"perf:{s}" for s in o["perf"]), *(f"edit:{s}" for s in o["edit"]), *(["smoke"] if o["smoke"] else [])])
    minutes = int(o["hours"] * 60)
    if o["dry"]:
        files = data_files(saves)
        say(f"dry run: jobs {jobs}; timer {minutes} min; data {len(files)} files, {sum(f.stat().st_size for _, f in files) / 1e6:.0f} MB "
            f"before compression; code {sum(f.stat().st_size for f in tmp.iterdir()) / 1e6:.1f} MB; image {ubuntu_ami()[2]}; spot {spot_prices(o['type'])}")
        clear(tmp)
        return
    data_key, data_size = data_bundle(saves, tmp)
    for n in ("head", "base"):
        if (tmp / f"{n}.tgz").exists():
            s3_put(tmp / f"{n}.tgz", f"runs/{name}/{n}.tgz")
    url_h = o["hours"] + 1
    urls = {"GET_head": presign("GET", f"runs/{name}/head.tgz", url_h), "GET_data": presign("GET", data_key, url_h)}
    if base_sha:
        urls["GET_base"] = presign("GET", f"runs/{name}/base.tgz", url_h)
    for n in ("log", "results", "done"):
        urls[f"PUT_{n}"] = presign("PUT", f"runs/{name}/{n}", url_h)
    urls["JOBS"] = jobs
    urls["RUNS"] = str(o["runs"])
    deadline = time.time() + minutes * 60
    text, red = render(name, urls, minutes, deadline)
    try:
        inst, market, price = launch(name, o["type"], text)
    except BaseException:
        s3_rm_run(name)  # nothing started: the run's code goes too
        clear(tmp)
        raise
    (STATE / f"{name}.userdata.sh").write_text(red, encoding="utf-8")
    iid = inst["InstanceId"]
    rec = {"run": name, "instance": iid, "type": o["type"], "market": market, "price_per_hour": price, "az": inst["Placement"]["AvailabilityZone"],
           "launched": now().isoformat(timespec="seconds"), "timer_min": minutes, "head": head_sha, "base": base_sha, "main_json": head_json, "tools_edited": edited,
           "jobs": jobs, "runs": o["runs"], "data": data_key, "data_mb": round(data_size / 1e6)}
    save_rec(name, rec)
    say(f"launched {iid} ({market}, ${price:.4f}/h, {rec['az']}); it shuts itself down within {minutes} min whatever happens")
    try:
        rec["machine_status"] = wait(name, iid, deadline)
    finally:
        fetch(name, STATE / name)
        end(name, rec, iid)
        clear(tmp)


def clear(tmp):
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def save_rec(name, rec):
    (STATE / f"{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


def wait(name, iid, deadline):
    """Follows the machine's log until it says done, the machine is gone, or the timer is past (plus 5 minutes)."""
    shown = 0
    while time.time() < deadline + 300:
        if s3_has(f"runs/{name}/done"):
            log_tail(name, shown)
            return s3_get(f"runs/{name}/done").decode().strip()
        shown = log_tail(name, shown)
        st = our_instance(iid)["State"]["Name"]
        if st in ("shutting-down", "terminated"):
            time.sleep(20)
            log_tail(name, shown)
            return s3_get(f"runs/{name}/done").decode().strip() if s3_has(f"runs/{name}/done") else f"machine {st} without a done"
        time.sleep(30)
    return "timer passed without a done"


def log_tail(name, shown):
    """Prints the log's step lines (== ...), PASS/FAIL lines and the bench's result lines after line number shown."""
    if not s3_has(f"runs/{name}/log"):
        return shown
    lines = s3_get(f"runs/{name}/log").decode(errors="replace").splitlines()
    for l in lines[shown:]:
        if l.startswith(("== ", "!! ", "PASS", "FAIL", "nyc ", "base ", "head ", "bench ")):
            say("  machine: " + l[:260])
    return len(lines)


def fetch(name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ("log", "done"):
        if s3_has(f"runs/{name}/{k}"):
            (out_dir / f"{k}.txt").write_bytes(s3_get(f"runs/{name}/{k}"))
    if s3_has(f"runs/{name}/results"):
        with tarfile.open(fileobj=io.BytesIO(s3_get(f"runs/{name}/results"))) as t:
            for m in t.getmembers():
                if m.name.startswith("/") or ".." in m.name or not (m.isfile() or m.isdir()):
                    continue
                t.extract(m, out_dir)
        say(f"results in {out_dir}")
    else:
        say(f"no results came back; the machine's log is in {out_dir / 'log.txt'}")


def end(name, rec, iid):
    """Terminate (if still up), wait for it to be gone, delete the run's objects, cost, and what is left."""
    ec2 = client("ec2")
    if our_instance(iid)["State"]["Name"] not in ("shutting-down", "terminated"):
        ec2.terminate_instances(InstanceIds=[iid])
        say(f"terminated {iid}")
    ec2.get_waiter("instance_terminated").wait(InstanceIds=[iid], WaiterConfig={"Delay": 10, "MaxAttempts": 60})
    i = our_instance(iid)
    m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", i.get("StateTransitionReason", ""))
    gone = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc) if m else now()
    hours = (gone - i["LaunchTime"]).total_seconds() / 3600
    cost = hours * (rec["price_per_hour"] + IPV4_HOUR + DISK_GB * EBS_GB_MONTH / 730)
    rec.update({"ended": gone.isoformat(timespec="seconds"), "minutes": round(hours * 60, 1), "cost_usd": round(cost, 4)})
    n = s3_rm_run(name)
    save_rec(name, rec)
    say(f"run {name}: {rec['minutes']} min on {rec['type']} ({rec['market']} ${rec['price_per_hour']:.4f}/h) = about ${cost:.3f} "
        f"(machine, disk and IPv4); {n} run objects deleted")
    left = leftovers()
    busy = {k: v for k, v in left.items() if v}
    say("left: " + (json.dumps(busy) if busy else "NOTHING IS RUNNING OR BILLING but the data cache (bench.py status)"))
    return left


def leftovers():
    ec2 = client("ec2")
    return {
        "instances": [i["InstanceId"] for i in instances()],
        "volumes": [v["VolumeId"] for v in ec2.describe_volumes()["Volumes"]],
        "spot_requests": [s["SpotInstanceRequestId"] for s in ec2.describe_spot_instance_requests(Filters=[{"Name": "state", "Values": ["open", "active"]}])["SpotInstanceRequests"]],
        "snapshots": [s["SnapshotId"] for s in ec2.describe_snapshots(OwnerIds=["self"])["Snapshots"]],
        "images": [m["ImageId"] for m in ec2.describe_images(Owners=["self"])["Images"]],
        "addresses": [a.get("AllocationId") for a in ec2.describe_addresses()["Addresses"]],
        "nat_gateways": [n["NatGatewayId"] for n in ec2.describe_nat_gateways(Filter=[{"Name": "state", "Values": ["pending", "available"]}])["NatGateways"]],
        "run_objects": sorted({o["Key"].split("/")[1] for o in s3_list("runs/")}) if bucket_exists() else [],
    }


def status():
    preflight()
    left = leftovers()
    data = s3_list("data/") if bucket_exists() else []
    gb = sum(o["Size"] for o in data) / 1e9
    say(json.dumps(left, indent=1))
    say(f"data cache: {len(data)} objects, {gb:.2f} GB, about ${gb * 0.023:.3f} a month")


def teardown(name):
    preflight()
    p = STATE / f"{name}.json"
    if not p.exists():
        if any(tagdict(i.get("Tags")).get("run") == name for i in instances()):
            raise SystemExit(f"no record {p}, yet a machine of run {name} is up: look at bench.py status")
        say(f"run {name}: no record and no machine; {s3_rm_run(name)} run objects deleted")
        return
    rec = json.loads(p.read_text(encoding="utf-8"))
    if not (STATE / name).exists():
        fetch(name, STATE / name)
    end(name, rec, rec["instance"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["setup"]:
        setup()
    elif a[:1] == ["run"]:
        run(a[1:])
    elif a[:1] == ["status"]:
        status()
    elif a[:1] == ["teardown"] and len(a) == 2:
        teardown(a[1])
    else:
        raise SystemExit(__doc__)
