"""T-1064: launch one spot machine for a run. docs/aws.md. The only way an instance is made.

    launch.py <run> --kind proof|wave --type m7a.8xlarge --hours 4 [--disk 200] [--max-price 0.6]
              [--file name=localpath ...] [--put name ...] [--dry]

- <run>: a new name (letters, digits, dashes); its keys in the bucket are <run>/<name>.
- --kind: the user data template, userdata/<kind>.sh.
- --hours: the hard timer, at most awslib.MAX_HOURS. The machine shuts itself down then, and a
  shutdown terminates it (instance-initiated shutdown behaviour = terminate), disk included.
- --file name=path: uploaded to <run>/<name>; the machine gets a presigned GET URL as $GET_<name>.
- --put name: the machine gets a presigned PUT URL for <run>/<name> as $PUT_<name> (log and
  heartbeat always). URLs last the timer plus two hours; the machine holds no AWS credentials.

Refuses: a run name used before, any instance of ours still live (one machine at a time),
burstable types (t2/t3/t4g bill extra credits), more than MAX_HOURS, a bid over MAX_PRICE_CAP.
Always spot, one-time (a persistent request would relaunch after the timer kills the machine).
"""
import base64
import json
import os
import re
import sys
import time

import awslib as A

AMI_PARAM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"


def args(argv):
    o = {"files": {}, "puts": ["log", "heartbeat"], "disk": 30, "max_price": None, "dry": False}
    if not argv or argv[0].startswith("-"):
        raise SystemExit(__doc__)
    o["run"] = argv[0]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--dry":
            o["dry"] = True
        elif a in ("--kind", "--type", "--hours", "--disk", "--max-price", "--file", "--put"):
            v = argv[i + 1]
            i += 1
            if a == "--file":
                n, p = v.split("=", 1)
                o["files"][n] = p
            elif a == "--put":
                o["puts"].append(v)
            else:
                o[a[2:].replace("-", "_")] = v
        else:
            raise SystemExit(f"unknown argument {a}\n{__doc__}")
        i += 1
    return o


def checks(o):
    run = o["run"]
    if not A.RUN_NAME.match(run):
        raise A.Refused(f"run name {run!r} must match {A.RUN_NAME.pattern}")
    for n in list(o["files"]) + o["puts"]:
        if not re.match(r"^[A-Za-z0-9_]+$", n):
            raise A.Refused(f"file name {n!r}: letters, digits and _ only (it becomes a shell variable)")
    for n, p in o["files"].items():
        if not os.path.isfile(p):
            raise SystemExit(f"--file {n}: {p} not found")
    if not (A.HERE / "userdata" / f"{o.get('kind')}.sh").exists():
        raise A.Refused(f"--kind {o.get('kind')}: no userdata/{o.get('kind')}.sh")
    t = o.get("type") or ""
    if not re.match(r"^[a-z][a-z0-9-]*\.[a-z0-9]+$", t):
        raise A.Refused(f"--type {t!r} is not an instance type")
    if A.BURSTABLE.match(t):
        raise A.Refused(f"{t} is burstable: unlimited CPU credits bill extra under a build's load; take c7a/m7a")
    h = float(o.get("hours") or 0)
    if not 0 < h <= A.MAX_HOURS:
        raise A.Refused(f"--hours {h}: give a timer above 0 and at most {A.MAX_HOURS}")
    o["hours"] = h
    o["disk"] = int(o["disk"])
    if not 8 <= o["disk"] <= 500:
        raise A.Refused(f"--disk {o['disk']} GB: 8-500")
    if A.load_record(run) is not None:
        raise A.Refused(f"run {run} was launched before ({A.record_path(run)}); take a new name")


def cheapest_az(itype):
    out = A.aws("ec2", "describe-spot-price-history", "--instance-types", itype,
                "--product-descriptions", "Linux/UNIX", "--start-time", A.iso(A.now()))
    by = {}
    for p in out.get("SpotPriceHistory", []):
        by.setdefault(p["AvailabilityZone"], float(p["SpotPrice"]))
    if not by:
        raise SystemExit(f"no spot price for {itype} in {A.REGION}")
    az = min(by, key=by.get)
    return az, by[az], by


def our_sg():
    out = A.aws("ec2", "describe-security-groups", "--filters", *A.tag_filters(), f"Name=group-name,Values={A.SG_NAME}")
    sgs = out.get("SecurityGroups", [])
    if sgs:
        return sgs[0]["GroupId"]
    vpc = A.aws("ec2", "describe-vpcs", "--filters", "Name=is-default,Values=true").get("Vpcs", [])
    if len(vpc) != 1:
        raise SystemExit("no default VPC in the region: stop and ask (never make a VPC, NAT or endpoint)")
    spec = json.dumps([{"ResourceType": "security-group", "Tags": A.tag_list({"Name": A.SG_NAME})}])
    g = A.aws("ec2", "create-security-group", "--group-name", A.SG_NAME, "--vpc-id", vpc[0]["VpcId"],
              "--description", "anitabuilder spot runner: no inbound, outbound only", "--tag-specifications", spec)
    A.say(f"made security group {g['GroupId']} ({A.SG_NAME}, no inbound rules)")
    return g["GroupId"]


def render(o, urls, deadline):
    tpl = (A.HERE / "userdata" / f"{o['kind']}.sh").read_text(encoding="utf-8")
    lines = [f"{v}='{u}'" for v, u in urls.items()]
    red = [f"{v}='<presigned {v.split('_', 1)[0]} {o['run']}/{v.split('_', 1)[1]}>'" for v in urls]
    minutes = max(1, int(o["hours"] * 60))
    fill = {"{{RUN}}": o["run"], "{{DEADLINE_EPOCH}}": str(int(deadline)), "{{MINUTES}}": str(minutes)}
    text, redacted = tpl, tpl
    for k, v in fill.items():
        text, redacted = text.replace(k, v), redacted.replace(k, v)
    text = text.replace("{{URLS}}", "\n".join(lines))
    redacted = redacted.replace("{{URLS}}", "\n".join(red))
    if "{{" in text:
        raise SystemExit("a template placeholder was left unfilled")
    if len(base64.b64encode(text.encode())) > 16384:
        raise SystemExit("user data over 16 KB: fewer files, or bundle them")
    return text, redacted


def main(argv):
    o = args(argv)
    checks(o)
    A.preflight()
    A.STATE.mkdir(parents=True, exist_ok=True)
    lock = A.STATE / "launch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise A.Refused(f"another launch holds {lock} (if no launch is under way and it is old, delete it)")
    try:
        os.write(fd, f"{os.getpid()} {o['run']} {A.iso(A.now())}".encode())
        os.close(fd)
        return launch(o)
    finally:
        os.remove(lock)


def launch(o):
    live = A.instances(A.LIVE)
    if live:
        ids = [f"{i['InstanceId']} ({A.tagdict(i.get('Tags')).get('run')}, {i['State']['Name']})" for i in live]
        raise A.Refused(f"a machine of ours is still live: {', '.join(ids)}. One at a time: wait for it, or "
                        "teardown.py it if its run is over")
    if not A.bucket_is_ours():
        raise A.Refused(f"bucket {A.BUCKET} missing: bucket.py ensure first")
    az, price, by = cheapest_az(o["type"])
    bid = float(o["max_price"]) if o["max_price"] else round(max(price * 2, price + 0.05), 4)
    if bid > A.MAX_PRICE_CAP:
        raise A.Refused(f"bid ${bid}/h over the cap ${A.MAX_PRICE_CAP}/h")
    if bid < price:
        raise A.Refused(f"bid ${bid}/h is under today's spot price ${price}/h in {az}")
    ami = A.aws("ssm", "get-parameter", "--name", AMI_PARAM)["Parameter"]["Value"]
    img = A.aws("ec2", "describe-images", "--image-ids", ami)["Images"][0]
    subnets = A.aws("ec2", "describe-subnets", "--filters", "Name=default-for-az,Values=true",
                    f"Name=availability-zone,Values={az}").get("Subnets", [])
    if not subnets:
        raise SystemExit(f"no default subnet in {az}")
    start = time.time()
    deadline = start + o["hours"] * 3600
    url_h = o["hours"] + 2
    A.say(f"run {o['run']}: {o['type']} spot in {az} at ${price}/h (bid cap ${bid}/h; {by}); timer {o['hours']} h; "
          f"disk {o['disk']} GB; AMI {ami} ({img.get('Name')})")
    if o["dry"]:
        A.say("dry run: nothing made")
        return
    urls = {}
    for n, p in o["files"].items():
        key = f"{o['run']}/{n}"
        import bucket
        bucket.put(p, key)
        urls[f"GET_{n}"] = A.presign("GET", key, url_h)
    for n in o["puts"]:
        urls[f"PUT_{n}"] = A.presign("PUT", f"{o['run']}/{n}", url_h)
    text, redacted = render(o, urls, deadline)
    (A.STATE / f"{o['run']}.userdata.sh").write_text(redacted, encoding="utf-8")
    ud = A.STATE / f"{o['run']}.userdata.tmp"
    extra = {"Name": f"anitabuilder-{o['run']}", "run": o["run"],
             "deadline": A.iso(A.at(deadline))}
    specs = [{"ResourceType": r, "Tags": A.tag_list(extra)}
             for r in ("instance", "volume", "network-interface", "spot-instances-request")]
    market = {"MarketType": "spot", "SpotOptions": {"MaxPrice": str(bid), "SpotInstanceType": "one-time",
                                                    "InstanceInterruptionBehavior": "terminate"}}
    bdm = [{"DeviceName": img["RootDeviceName"],
            "Ebs": {"VolumeSize": o["disk"], "VolumeType": "gp3", "DeleteOnTermination": True}}]
    try:
        with open(ud, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        out = A.aws("ec2", "run-instances", "--image-id", ami, "--instance-type", o["type"], "--count", "1",
                    "--subnet-id", subnets[0]["SubnetId"], "--security-group-ids", our_sg(),
                    "--instance-initiated-shutdown-behavior", "terminate",
                    "--instance-market-options", json.dumps(market),
                    "--block-device-mappings", json.dumps(bdm),
                    "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled",
                    "--tag-specifications", json.dumps(specs),
                    "--user-data", f"file://{ud}")
    finally:
        if ud.exists():
            ud.unlink()  # the presigned URLs stay out of files; the redacted copy is beside it
    inst = out["Instances"][0]
    rec = {"run": o["run"], "kind": o["kind"], "instance_id": inst["InstanceId"], "type": o["type"], "az": az,
           "spot_price_at_launch": price, "bid": bid, "ami": ami, "disk_gb": o["disk"], "hours": o["hours"],
           "launched": A.iso(A.now()), "deadline": extra["deadline"],
           "results": [f"{o['run']}/{n}" for n in o["puts"] if n not in ("log", "heartbeat")],
           "files": sorted(o["files"]),
           "spot_request": inst.get("SpotInstanceRequestId")}
    A.save_record(o["run"], rec)
    A.say(f"launched {inst['InstanceId']} (spot request {inst.get('SpotInstanceRequestId')}); record {A.record_path(o['run'])}")
    A.say(f"watch: status.py ; its log: bucket.py get {o['run']}/log ; end: teardown.py {o['run']}")


if __name__ == "__main__":
    main(sys.argv[1:])
