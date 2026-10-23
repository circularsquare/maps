"""T-1064: end a run, then prove nothing of ours is left. docs/aws.md. The only way things are destroyed.

    teardown.py <run>        terminate the run's machine (if still up), then sweep our leftovers
    teardown.py --all-ours   terminate every machine tagged ours (a run nobody remembers), then sweep

The sweep removes, among things tagged owner=anita project=anitabuilder only: open spot requests,
unattached volumes, snapshots, images, Elastic IPs, and our security group once no machine of ours
is left. Each id is looked up again and checked for our tags right before it is touched. The bucket's
objects stay (fetch the results, then bucket.py rm <run>). Ends with status.py --all-regions;
exits 1 if anything of ours still exists.
"""
import sys
import time

import awslib as A
import status


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    A.preflight()
    run = None if argv[0] == "--all-ours" else argv[0]
    if run is not None and not A.RUN_NAME.match(run):
        raise SystemExit(__doc__)
    live = [i for i in A.instances() if i["State"]["Name"] != "terminated"]
    pick = [i for i in live if run is None or A.tagdict(i.get("Tags")).get("run") == run]
    ids = []
    for i in pick:
        A.require_ours("instance", i["InstanceId"])
        ids.append(i["InstanceId"])
    if ids:
        A.say(f"terminating {' '.join(ids)}")
        A.aws("ec2", "terminate-instances", "--instance-ids", *ids)
        t0 = time.time()
        A.aws("ec2", "wait", "instance-terminated", "--instance-ids", *ids, text=True, check=False)
        A.say(f"  terminated ({time.time() - t0:.0f} s)")
    else:
        A.say("no machine of the run is live" if run else "no machine of ours is live")
    if run is not None:
        rec = A.load_record(run)
        if rec is not None and not rec.get("ended"):
            rec["ended"] = A.iso(A.now())  # the latest it can have stopped billing; status cost prefers AWS's time
            A.save_record(run, rec)
    sweep()
    if run is not None and A.load_record(run):
        status.cost(run)
    n = status.show(all_regions=True)
    sys.exit(1 if n else 0)


def sweep():
    f = A.tag_filters()
    cancelled = []
    for s in A.aws("ec2", "describe-spot-instance-requests", "--filters", *f).get("SpotInstanceRequests", []):
        if s["State"] in ("open", "active"):
            A.require_ours("spot", s["SpotInstanceRequestId"])
            A.aws("ec2", "cancel-spot-instance-requests", "--spot-instance-request-ids", s["SpotInstanceRequestId"])
            A.say(f"cancelled spot request {s['SpotInstanceRequestId']}")
            cancelled.append(s["SpotInstanceRequestId"])
    for _ in range(12):  # the cancel shows in a listing some seconds later
        if not cancelled:
            break
        left = A.aws("ec2", "describe-spot-instance-requests", "--spot-instance-request-ids", *cancelled)
        cancelled = [s["SpotInstanceRequestId"] for s in left.get("SpotInstanceRequests", []) if s["State"] in ("open", "active")]
        if cancelled:
            time.sleep(5)
    for v in A.aws("ec2", "describe-volumes", "--filters", *f, "Name=status,Values=available").get("Volumes", []):
        A.require_ours("volume", v["VolumeId"])
        A.aws("ec2", "delete-volume", "--volume-id", v["VolumeId"])
        A.say(f"deleted volume {v['VolumeId']} ({v['Size']} GB)")
    for im in A.aws("ec2", "describe-images", "--owners", "self", "--filters", *f).get("Images", []):
        A.require_ours("image", im["ImageId"])
        A.aws("ec2", "deregister-image", "--image-id", im["ImageId"])
        A.say(f"deregistered image {im['ImageId']}")
    for s in A.aws("ec2", "describe-snapshots", "--owner-ids", "self", "--filters", *f).get("Snapshots", []):
        A.require_ours("snapshot", s["SnapshotId"])
        A.aws("ec2", "delete-snapshot", "--snapshot-id", s["SnapshotId"])
        A.say(f"deleted snapshot {s['SnapshotId']}")
    for a in A.aws("ec2", "describe-addresses", "--filters", *f).get("Addresses", []):
        A.require_ours("address", a["AllocationId"])
        if a.get("AssociationId"):
            A.aws("ec2", "disassociate-address", "--association-id", a["AssociationId"])
        A.aws("ec2", "release-address", "--allocation-id", a["AllocationId"])
        A.say(f"released Elastic IP {a.get('PublicIp')}")
    if not [i for i in A.instances() if i["State"]["Name"] != "terminated"]:
        for g in A.aws("ec2", "describe-security-groups", "--filters", *f).get("SecurityGroups", []):
            A.require_ours("sg", g["GroupId"])
            for attempt in range(6):  # a just-terminated machine's network interface lingers a little
                if A.aws("ec2", "delete-security-group", "--group-id", g["GroupId"], check=False) is not None:
                    A.say(f"deleted security group {g['GroupId']}")
                    break
                time.sleep(20)
            else:
                A.say(f"security group {g['GroupId']} still in use; it costs nothing, the next teardown takes it")


if __name__ == "__main__":
    main(sys.argv[1:])
