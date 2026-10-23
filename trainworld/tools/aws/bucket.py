"""T-1064: our S3 bucket (awslib.BUCKET) and nothing else. docs/aws.md.

    bucket.py ensure                    make the bucket if missing: tagged, private, objects expire in 14 days
    bucket.py ls [run]                  objects and sizes, by run
    bucket.py put <file> <run>/<name>   upload one file
    bucket.py get <run>/<name> [--big]  download one object into artifacts/trainworld/aws/inbox/<run>/
                                        (refused over 5 GB without --big: egress is $0.09/GB)
    bucket.py rm <run> [--unfetched]    delete one run's objects (refused while its results are
                                        unfetched, unless --unfetched)
    bucket.py delete                    delete the bucket itself, when empty

Every write checks first that the bucket carries our tags; keys are always <run>/<name>.
"""
import json
import os
import sys

import awslib as A

BIG = 5e9
LIFECYCLE = {"Rules": [
    {"ID": "expire-14d", "Status": "Enabled", "Filter": {"Prefix": ""}, "Expiration": {"Days": 14},
     "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}]}


def ensure():
    A.preflight()
    if A.bucket_is_ours():
        A.say(f"{A.BUCKET}: exists, ours")
        return
    A.aws("s3api", "create-bucket", "--bucket", A.BUCKET,
          "--create-bucket-configuration", f"LocationConstraint={A.REGION}")
    A.aws("s3api", "put-bucket-tagging", "--bucket", A.BUCKET,
          "--tagging", json.dumps({"TagSet": A.tag_list()}))
    A.aws("s3api", "put-public-access-block", "--bucket", A.BUCKET, "--public-access-block-configuration",
          "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true")
    A.aws("s3api", "put-bucket-lifecycle-configuration", "--bucket", A.BUCKET,
          "--lifecycle-configuration", json.dumps(LIFECYCLE))
    if not A.bucket_is_ours():
        raise SystemExit("bucket made but its tags did not read back")
    A.say(f"{A.BUCKET}: made, tagged {A.TAGS}, private, objects expire after 14 days")


def ls(run=None):
    A.preflight()
    objs = A.bucket_objects((run + "/") if run else "")
    by = {}
    for o in objs:
        r = o["Key"].split("/", 1)[0]
        by.setdefault(r, []).append(o)
    for r, os_ in sorted(by.items()):
        A.say(f"{r}: {len(os_)} objects, {sum(o['Size'] for o in os_) / 1e6:.1f} MB")
        for o in os_:
            A.say(f"   {o['Size'] / 1e6:10.2f} MB  {o['LastModified']}  {o['Key']}")
    A.say(f"total {sum(o['Size'] for o in objs) / 1e6:.1f} MB in {len(objs)} objects")
    return objs


def put(path, key):
    A.preflight()
    A.check_key(key)
    if not A.bucket_is_ours():
        raise A.Refused(f"{A.BUCKET} does not exist: bucket.py ensure first")
    A.aws("s3", "cp", path, f"s3://{A.BUCKET}/{key}", "--only-show-errors", text=True)
    A.say(f"put {key} ({os.path.getsize(path) / 1e6:.1f} MB)")


def get(key, big=False):
    A.preflight()
    A.check_key(key)
    objs = [o for o in A.bucket_objects(key) if o["Key"] == key]
    if not objs:
        raise SystemExit(f"no object {key}")
    size = objs[0]["Size"]
    if size > BIG and not big:
        raise A.Refused(f"{key} is {size / 1e9:.1f} GB (${size / 1e9 * 0.09:.2f} of egress); pass --big if it is meant")
    run, name = key.split("/", 1)
    dest = A.INBOX / run / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    A.aws("s3", "cp", f"s3://{A.BUCKET}/{key}", str(dest), "--only-show-errors", text=True)
    rec = A.load_record(run)
    if rec is not None:
        rec.setdefault("fetched", {})[key] = size
        A.save_record(run, rec)
    A.say(f"got {key} -> {dest} ({size / 1e6:.1f} MB)")
    return dest


def rm(run, unfetched=False):
    A.preflight()
    if not A.RUN_NAME.match(run):
        raise A.Refused(f"run {run!r} does not match {A.RUN_NAME.pattern}")
    objs = A.bucket_objects(run + "/")
    rec = A.load_record(run) or {}
    want = set(rec.get("results", []))
    missing = sorted(k for k in want if k not in rec.get("fetched", {}) and any(o["Key"] == k for o in objs))
    if missing and not unfetched:
        raise A.Refused(f"run {run}'s results are not fetched yet: {missing}; bucket.py get them, or pass --unfetched")
    for o in objs:
        A.aws("s3api", "delete-object", "--bucket", A.BUCKET, "--key", A.check_key(o["Key"]))
    A.say(f"removed {len(objs)} objects under {run}/")


def delete():
    A.preflight()
    if not A.bucket_is_ours():
        A.say("no bucket")
        return
    objs = A.bucket_objects()
    if objs:
        raise A.Refused(f"{len(objs)} objects still in {A.BUCKET}: bucket.py rm <run> first")
    A.aws("s3api", "delete-bucket", "--bucket", A.BUCKET)
    A.say(f"deleted {A.BUCKET}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    cmd, rest = a[0], [x for x in a[1:] if not x.startswith("--")]
    flags = {x for x in a[1:] if x.startswith("--")}
    if cmd == "ensure":
        ensure()
    elif cmd == "ls":
        ls(*rest[:1])
    elif cmd == "put" and len(rest) == 2:
        put(rest[0], rest[1])
    elif cmd == "get" and len(rest) == 1:
        get(rest[0], "--big" in flags)
    elif cmd == "rm" and len(rest) == 1:
        rm(rest[0], "--unfetched" in flags)
    elif cmd == "delete":
        delete()
    else:
        raise SystemExit(__doc__)
