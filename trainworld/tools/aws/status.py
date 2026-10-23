"""T-1064: what of ours exists, and what a run cost. Read-only. docs/aws.md.

    status.py                  everything tagged ours in us-east-2 (instances, spot requests, volumes,
                               snapshots, images, Elastic IPs, security groups) and the bucket's size;
                               machines past their timer are flagged OVERDUE
    status.py --all-regions    also instances and volumes tagged ours in every other region
    status.py cost <run>       the run's machine hours, spot price and dollars (saved in its record)

Run it first, before any other AWS step, and last, after a teardown.
"""
import re
import sys

import awslib as A

EBS_GB_MONTH = 0.08     # gp3, us-east-2
IPV4_HOUR = 0.005       # a public IPv4 address while the machine lives
EGRESS_GB = 0.09        # S3 to the internet (bucket.py get)


def show(all_regions=False):
    A.preflight()
    left = A.leftovers()
    A.say(f"account {A.ACCOUNT}, region {A.REGION}, tag owner=anita project=anitabuilder")
    n = 0
    for kind, items in left.items():
        for x in items:
            n += kind != "security_groups"
            A.say(f"  {kind}: {describe(kind, x)}")
    objs = A.bucket_objects() if A.bucket_is_ours() else None
    if objs is None:
        A.say(f"  bucket {A.BUCKET}: none")
    else:
        runs = sorted({o['Key'].split('/', 1)[0] for o in objs})
        A.say(f"  bucket {A.BUCKET}: {len(objs)} objects, {sum(o['Size'] for o in objs) / 1e6:.1f} MB"
              + (f" (runs {', '.join(runs)})" if runs else ""))
    if all_regions:
        regions = [r["RegionName"] for r in A.aws("ec2", "describe-regions")["Regions"]]
        for r in regions:
            if r == A.REGION:
                continue
            found = A.instances(region=r)
            found = [i for i in found if "error" in i or i["State"]["Name"] != "terminated"]
            for i in found:
                n += 1
                A.say(f"  OTHER REGION {r}: {i}")
        A.say(f"  other regions checked: {len(regions) - 1}")
    if n == 0:
        A.say("NOTHING OF OURS IS RUNNING OR BILLING (security groups and an empty bucket cost nothing)")
    else:
        A.say(f"{n} things of ours exist: if no run is under way, teardown.py")
    return n


def describe(kind, x):
    t = A.tagdict(x.get("Tags"))
    if kind == "instances":
        over = ""
        if t.get("deadline") and A.now() > A.parse(t["deadline"]):
            over = "  OVERDUE: past its timer, teardown.py it"
        return (f"{x['InstanceId']} {x['InstanceType']} {x['State']['Name']} run={t.get('run')} "
                f"launched {x['LaunchTime'][:19]} deadline {t.get('deadline')}{over}")
    if kind == "spot_requests":
        return f"{x['SpotInstanceRequestId']} {x['State']} {x.get('InstanceId')}"
    if kind == "volumes":
        return f"{x['VolumeId']} {x['Size']} GB {x['State']}"
    if kind == "snapshots":
        return f"{x['SnapshotId']} {x.get('VolumeSize')} GB"
    if kind == "images":
        return f"{x['ImageId']} {x.get('Name')}"
    if kind == "addresses":
        return f"{x.get('AllocationId')} {x.get('PublicIp')} associated={x.get('AssociationId')}"
    if kind == "security_groups":
        return f"{x['GroupId']} {x['GroupName']} (free)"
    return str(x)


def last_put(run):
    """The machine's last upload (log, heartbeat) plus 2 minutes for its shutdown, or None."""
    import datetime
    objs = [o for o in A.bucket_objects(run + "/") if o["Key"].split("/", 1)[1] in ("log", "heartbeat")]
    if not objs:
        return None
    return max(A.parse(o["LastModified"]) for o in objs) + datetime.timedelta(minutes=2)


def ended_at(inst, rec):
    """When the machine stopped billing: the terminate time AWS states, else what a teardown saw."""
    if inst and inst["State"]["Name"] in ("terminated", "shutting-down"):
        m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", inst.get("StateTransitionReason") or "")
        if m:
            return A.parse(m.group(1).replace(" ", "T")), "AWS's terminate time"
        last = last_put(rec["run"])
        if last and (not rec.get("ended") or last < A.parse(rec["ended"])):
            return last, "its last upload to the bucket plus 2 minutes (it shut itself down)"
        if rec.get("ended"):
            return A.parse(rec["ended"]), "the teardown's record"
        return A.now(), "now (an upper bound)"
    if inst is None and rec.get("ended"):
        return A.parse(rec["ended"]), "the teardown's record"
    return A.now(), "now (still live)" if inst else "now (an upper bound)"


def cost(run):
    A.preflight()
    rec = A.load_record(run)
    if rec is None:
        raise SystemExit(f"no record for run {run} in {A.STATE}")
    out = A.aws("ec2", "describe-instances", "--instance-ids", rec["instance_id"], check=False)
    inst = None
    if out:
        res = [i for r in out.get("Reservations", []) for i in r["Instances"]]
        inst = res[0] if res and A.is_ours(res[0].get("Tags")) else None
    start = A.parse(inst["LaunchTime"]) if inst else A.parse(rec["launched"])
    end, why = ended_at(inst, rec)
    hours = max(60, (end - start).total_seconds()) / 3600  # spot bills by the second, a minute at least
    hist = A.aws("ec2", "describe-spot-price-history", "--instance-types", rec["type"],
                 "--availability-zone", rec["az"], "--product-descriptions", "Linux/UNIX",
                 "--start-time", A.iso(start), "--end-time", A.iso(end)).get("SpotPriceHistory", [])
    prices = [float(p["SpotPrice"]) for p in hist] or [rec["spot_price_at_launch"]]
    price = max(prices)
    machine = hours * price
    disk = rec["disk_gb"] * hours * EBS_GB_MONTH / 730
    ipv4 = hours * IPV4_HOUR
    egress = sum(rec.get("fetched", {}).values()) / 1e9 * EGRESS_GB
    total = machine + disk + ipv4 + egress
    c = {"start": A.iso(start), "end": A.iso(end), "end_from": why, "hours": round(hours, 3),
         "spot_price_max": price, "spot_prices_seen": sorted(set(prices)), "machine_usd": round(machine, 4),
         "disk_usd": round(disk, 4), "ipv4_usd": round(ipv4, 4), "egress_usd": round(egress, 4),
         "total_usd": round(total, 4)}
    rec["cost"] = c
    A.save_record(run, rec)
    A.say(f"run {run}: {rec['type']} in {rec['az']}, {start:%H:%M}-{end:%H:%M} UTC ({why}), {hours:.2f} h "
          f"at up to ${price}/h spot")
    A.say(f"  machine ${machine:.4f} + disk {rec['disk_gb']} GB ${disk:.4f} + public IPv4 ${ipv4:.4f} "
          f"+ egress ${egress:.4f} = ${total:.4f} (S3 requests and storage: fractions of a cent)")
    return c


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["cost"] and len(a) == 2:
        cost(a[1])
    elif a in ([], ["--all-regions"]):
        show(bool(a))
    else:
        raise SystemExit(__doc__)
