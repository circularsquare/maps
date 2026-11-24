"""T-1064: stop a raw AWS call that creates, changes or destroys something, before it runs.
docs/aws.md. Meant as a Claude Code PreToolUse hook on Bash and PowerShell:

    guard.py --hook          read the hook's JSON on stdin; exit 2 (blocked, reason on stderr) or 0
    guard.py "<command>"     say ALLOW or REFUSE for one command line, and why
    guard.py --test          the built-in table of lines it must allow and refuse

AWS means Anita's own account only (profile anita-own). The friend's account is retired (Anita,
2026-11-24, T-4177): anything aimed at it is refused outright, whatever it does: a raw aws call without
`--profile anita-own`, `--profile default` or AWS_PROFILE=default, its account number next to an aws or
boto call, inline boto3/botocore that doesn't name anita-own, and the friend-account scripts in tools/aws/
(awslib refuses too). In her own account, read-only calls pass (describe, list, get, head, wait, `s3 ls`,
`sts get-caller-identity`), and so does tools/aws/bench.py, which checks the account itself. Everything else
that names the aws CLI, or boto3/botocore with a changing method, is refused with the script to use instead.
"""
import json
import re
import sys

OURS = "twbench-"  # bucket names of ours start so (bench.py BUCKET, her own account)
OWN_PROFILE = "anita-own"
FRIEND_ACCOUNT = "429583250612"  # retired, T-4177: never reached again
FRIEND_SCRIPTS = re.compile(r"""(?<![\w.-])py(?:thon\d?)?(?:\.exe)?['"]?\s+(?:-\S+\s+)*['"]?[^\s'"]*aws[\\/]+"""
                            r"""(?:awslib|status|bucket|launch|teardown|bundle|seed|wavepack|land)\.py\b|\b(?:import|from)\s+awslib\b""", re.I)
FRIEND_PROFILE = re.compile(r"""--profile[\s=]+['"]?default\b|AWS_PROFILE['"]?\s*[:=]\s*['"]?default\b|profile(?:_name)?\s*=\s*['"]default['"]""", re.I)
BOTO_USE = re.compile(r"\b(?:boto3|botocore)(?:\.session)?\s*\.\s*(?:client|resource|session|Session)\b|\bimport\s+(?:boto3|botocore)|\bfrom\s+(?:boto3|botocore)\b")
READ_PREFIXES = ("describe-", "list-", "get-", "head-", "lookup-", "filter-log-events", "search-",
                 "batch-get-", "estimate-")
SECRET_READS = ("get-secret-value", "get-password-data", "export-credentials", "get-session-token",
                "get-federation-token", "get-object", "get-object-acl", "download-")
VALUED = {"--region", "--profile", "--output", "--query", "--endpoint-url", "--cli-read-timeout",
          "--cli-connect-timeout", "--color", "--ca-bundle", "--cli-binary-format"}
# Case-sensitive: the CLI is typed `aws`, while "AWS" in prose (a commit message) is not a call.
AWS_AT = re.compile(r"""(?:^|[\s;&|(`'"=,\[{])(?:[^\s;&|'"`]*[\\/])?aws(?:\.exe)?(?=[\s'",\])}]|$)""")
BOTO = re.compile(r"\b(boto3|botocore|aiobotocore|s3fs)\b")
BOTO_CHANGE = re.compile(r"""\.(terminate|delete|run|request|create|put|modify|cancel|stop|start|reboot|attach|detach|"""
                         r"""associate|disassociate|allocate|release|deregister|register|authorize|revoke|update|"""
                         r"""copy|upload|restore|import|replace|enable|disable|purchase|rm|mkdir|move|mv)\w*\s*\(""", re.I)
SEARCHERS = re.compile(r"^\s*(grep|rg|findstr|Select-String|git\s+(-C\s+\S+\s+)?(log|grep|diff|show|status|commit)|type|cat|less|head|tail|Get-Content)\b", re.I)
EXEC_PIPE = re.compile(r"\|\s*(sh|bash|pwsh|powershell|iex|Invoke-Expression|cmd)\b|\bInvoke-Expression\b|\biex\b", re.I)

ADVICE = ("AWS means Anita's own account only: use trainworld/tools/aws/bench.py, or read-only calls with "
          "--profile anita-own; the friend's account and its scripts are retired (T-4177); "
          "read trainworld/docs/aws.md first")


def words(rest):
    """The aws call's words: quotes, commas and brackets stripped (a Python list form counts too),
    stopping at the end of the shell command."""
    rest = re.split(r"[;|&\n`]", rest, maxsplit=1)[0]
    return [w.strip("'\"[](){}") for w in re.split(r"[\s,]+", rest) if w.strip("'\"[](){}")]


def judge_aws(ws):
    """ALLOW or REFUSE for one aws invocation's words (after `aws`)."""
    i, svc, op, profile = 0, None, None, None
    rest = []
    while i < len(ws):
        w = ws[i]
        if w.startswith("--"):
            if w.startswith("--profile="):
                profile = w.split("=", 1)[1]
            elif w == "--profile" and i + 1 < len(ws):
                profile = ws[i + 1]
            if w.split("=")[0] in VALUED and "=" not in w:
                i += 1
        elif svc is None:
            svc = w
        elif op is None:
            op = w
        else:
            rest.append(w)
        i += 1
    flat = " ".join(ws)
    if svc is None or svc in ("help", "--version") or op in (None, "help"):
        return True, "no operation"
    if svc == "configure" and op in ("list", "list-profiles"):
        return True, "reads the CLI's settings"
    if profile != OWN_PROFILE:
        return False, (f"`aws {svc} {op}` without --profile {OWN_PROFILE} would use the friend's account (retired, T-4177)"
                       if profile in (None, "default") else f"profile {profile} is not {OWN_PROFILE}, the only AWS account in use")
    if "--with-decryption" in flat:
        return False, f"`{svc} {op} --with-decryption` reads a secret value"
    if svc == "configure":
        if op in ("list", "list-profiles") or (op == "get" and rest[:1] and rest[0] in ("region", "output")):
            return True, "reads the CLI's settings"
        return False, f"`configure {op}` can print or change credentials; the secret key never goes to a log or chat"
    if svc == "s3":
        if op == "ls":
            target = next((w for w in rest if w.startswith("s3://")), None)
            if target and not target[5:].startswith(OURS):
                return False, f"`s3 ls {target}` lists a bucket that is not ours (the friend's buckets are never read)"
            return True, "lists"
        if op == "presign":
            return True, "signs a URL, changes nothing"
        return False, f"`s3 {op}` copies, moves or deletes objects; bucket.py does it in our bucket only, under one run's prefix"
    if op in SECRET_READS or op.startswith("download-"):
        return False, f"`{svc} {op}` reads secret or object contents"
    if svc == "s3api" and op.startswith(("list-object", "head-object", "get-bucket", "head-bucket")):
        b = None
        for j, w in enumerate(rest):
            if w == "--bucket" and j + 1 < len(rest):
                b = rest[j + 1]
        if b is None:
            m = re.search(r"--bucket[= ](\S+)", flat)
            b = m.group(1) if m else None
        if b and not b.startswith(OURS):
            return False, f"`s3api {op}` on bucket {b}, which is not ours (the friend's buckets are never read)"
        return True, "reads our bucket"
    if op == "wait" or op.startswith(READ_PREFIXES):
        return True, "read-only"
    return False, f"`aws {svc} {op}` creates, changes or destroys something"


def judge(cmd):
    """(allowed, reason) for a whole command line."""
    if SEARCHERS.match(cmd) and not EXEC_PIPE.search(cmd):
        first = cmd.strip().split()[0]
        if not re.search(r"[;&]|\$\(", cmd):
            return True, f"{first} only reads text"
    if FRIEND_SCRIPTS.search(cmd):
        return False, "the friend-account scripts in tools/aws/ (awslib and the wave flow) are retired with that account (T-4177)"
    if FRIEND_PROFILE.search(cmd):
        return False, "the default profile was the friend's account, retired (T-4177)"
    if FRIEND_ACCOUNT in cmd and (AWS_AT.search(cmd) or BOTO.search(cmd)):
        return False, f"the friend's account {FRIEND_ACCOUNT} is retired (T-4177)"
    if BOTO_USE.search(cmd) and OWN_PROFILE not in cmd:
        return False, f"inline boto3/botocore without profile {OWN_PROFILE} would use the friend's account's old default (retired, T-4177)"
    for m in AWS_AT.finditer(cmd):
        ok, why = judge_aws(words(cmd[m.end():]))
        if not ok:
            return False, why
    if BOTO.search(cmd):
        m = BOTO_CHANGE.search(cmd)
        if m:
            return False, f"inline {BOTO.search(cmd).group(1)} with a changing call `{m.group(0).strip()}`"
    return True, "no changing AWS call"


TESTS = [
    # (command line, allowed?)
    ("aws sts get-caller-identity --profile anita-own", True),
    ("aws --version", True),
    ("aws ec2 describe-instances --profile anita-own --filters Name=tag:project,Values=twbench", True),
    ("aws ec2 describe-spot-price-history --region us-east-2 --profile anita-own --instance-types m7a.8xlarge", True),
    ("aws ec2 wait instance-terminated --profile=anita-own --instance-ids i-0abc", True),
    ("aws ec2 get-console-output --profile anita-own --instance-id i-0abc --output text", True),
    ("aws ssm get-parameter --profile anita-own --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id", True),
    ("aws s3 ls --profile anita-own", True),
    ("aws s3 ls s3://twbench-use2-r7k2m9/runs/ --profile anita-own", True),
    ("aws s3api list-objects-v2 --bucket twbench-use2-r7k2m9 --profile anita-own", True),
    ("aws s3 ls s3://anitabuilder-builds-use2-k7q4/proof1/ --profile anita-own", False),
    ("aws configure list", True),
    ("aws configure get region --profile anita-own", True),
    ('$env:PATH = "x"; aws --version', True),
    ('& "C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe" ec2 describe-volumes --profile anita-own --filters Name=tag:project,Values=twbench', True),
    # the friend's account, retired 2026-11-24 (T-4177): refused whatever the call does
    ("venv\\Scripts\\python.exe trainworld\\tools\\aws\\teardown.py proof1", False),
    ("venv/Scripts/python.exe trainworld/tools/aws/launch.py w1 --kind wave --type m7a.8xlarge --hours 4", False),
    ("python trainworld/tools/aws/bucket.py rm proof1", False),
    ("venv\\Scripts\\python.exe -I trainworld\\tools\\aws\\status.py --all-regions", False),
    ('python -c "import awslib; awslib.preflight()"', False),
    ("aws sts get-caller-identity", False),
    ("aws ec2 describe-instances --profile default", False),
    ("aws s3 ls --profile=default", False),
    ("aws ec2 describe-instances --profile someoneelse", False),
    ('$env:AWS_PROFILE = "default"; aws sts get-caller-identity --profile anita-own', False),
    ("aws organizations describe-account --account-id 429583250612 --profile anita-own", False),
    ('python -c "import boto3; print(boto3.client(\'sts\').get_caller_identity())"', False),
    ('python -c "import boto3; s = boto3.Session(profile_name=\'anita-own\'); print(s.client(\'sts\').get_caller_identity())"', True),
    ("git add trainworld/tools/aws/awslib.py trainworld/tools/aws/status.py", True),
    ("python trainworld/tools/status.py", True),
    ("pip show boto3", True),
    ("aws configure list-profiles", True),
    ('grep -rn "429583250612" trainworld/docs', True),
    ('grep -rn "aws ec2 terminate-instances" trainworld/docs', True),
    ("git log --oneline -5", True),
    ('git -C C:/x commit -q -m "anitabuilder: AWS build tooling (T-1064)." -m "trailer"', True),
    ('git commit -m "aws ec2 terminate-instances is refused"; aws ec2 terminate-instances --instance-ids i-1', False),
    ("npm run check --prefix app", True),
    ("aws ec2 terminate-instances --instance-ids i-0123456789abcdef0", False),
    ("aws ec2 terminate-instances --filters Name=instance-state-name,Values=running", False),
    ("aws ec2 run-instances --image-id ami-1 --instance-type m7a.8xlarge", False),
    ("aws ec2 request-spot-instances --type persistent --launch-specification file://x.json", False),
    ("aws ec2 create-nat-gateway --subnet-id subnet-1", False),
    ("aws ec2 create-vpc-endpoint --vpc-id vpc-1 --service-name com.amazonaws.us-east-2.s3", False),
    ("aws ec2 allocate-address", False),
    ("aws ec2 create-snapshot --volume-id vol-1", False),
    ("aws ec2 create-image --instance-id i-1 --name x", False),
    ("aws ec2 delete-volume --volume-id vol-1", False),
    ("aws ec2 stop-instances --instance-ids i-1", False),
    ("aws ec2 create-tags --resources i-1 --tags Key=owner,Value=anita", False),
    ("aws s3 rm s3://ancientgreek --recursive", False),
    ("aws s3 rm s3://anitabuilder-builds-use2-k7q4/w1/ --recursive", False),
    ("aws s3 sync ./out s3://anitabuilder-builds-use2-k7q4/w1 --delete", False),
    ("aws s3 cp s3://anitabuilder-builds-use2-k7q4/w1/results.tgz .", False),
    ("aws s3 mb s3://my-new-bucket", False),
    ("aws s3 rb s3://filmbro-db-staging --force", False),
    ("aws s3 ls s3://hypertext-db/", False),
    ("aws s3api list-objects-v2 --bucket jeanland-metaphors", False),
    ("aws s3api put-bucket-policy --bucket anitabuilder-builds-use2-k7q4 --policy file://p.json", False),
    ("aws s3api get-object --bucket ckopx-research --key a b", False),
    ("aws iam create-role --role-name x --assume-role-policy-document file://t.json", False),
    ("aws iam attach-role-policy --role-name x --policy-arn arn:aws:iam::aws:policy/AdministratorAccess", False),
    ("aws configure get aws_secret_access_key", False),
    ("aws configure export-credentials", False),
    ("aws secretsmanager get-secret-value --secret-id x", False),
    ("aws ssm get-parameter --name /x --with-decryption", False),
    ("aws ec2 describe-instances && aws ec2 terminate-instances --instance-ids i-1", False),
    ('$env:PATH = "x"; aws ec2 terminate-instances --instance-ids i-1', False),
    ('& "C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe" ec2 terminate-instances --instance-ids i-1', False),
    ("aws --region us-west-2 ec2 run-instances --image-id ami-1", False),
    ("aws --profile default --output json ec2 delete-security-group --group-id sg-1", False),
    ('python -c "import subprocess; subprocess.run([\'aws\', \'ec2\', \'terminate-instances\', \'--instance-ids\', \'i-1\'])"', False),
    ('python -c "import boto3; boto3.client(\'ec2\').terminate_instances(InstanceIds=[\'i-1\'])"', False),
    ('python -c "import boto3; print(boto3.client(\'ec2\').describe_instances())"', False),
    ('echo "aws ec2 terminate-instances --instance-ids i-1" | bash', False),
    ("aws cloudformation create-stack --stack-name x --template-body file://t.yaml", False),
    ("aws elbv2 create-load-balancer --name x", False),
    # her own account (profile anita-own, T-4166): raw changing calls are refused there too; bench.py does them
    ("aws sts get-caller-identity --profile anita-own", True),
    ("aws ec2 describe-spot-price-history --profile anita-own --instance-types m8a.xlarge", True),
    ("aws --profile anita-own ec2 run-instances --image-id ami-1 --instance-type m8a.xlarge", False),
    ("aws ec2 terminate-instances --profile anita-own --instance-ids i-1", False),
    ("aws iam attach-user-policy --profile anita-own --user-name bench --policy-arn arn:aws:iam::aws:policy/AdministratorAccess", False),
    ("venv\\Scripts\\python.exe trainworld\\tools\\aws\\bench.py run --perf nyc --edit nyc", True),
]


def test():
    bad = 0
    for cmd, want in TESTS:
        ok, why = judge(cmd)
        mark = "ok  " if ok == want else "FAIL"
        bad += ok != want
        print(f"{mark} {'ALLOW ' if ok else 'REFUSE'} {cmd[:110]}  ({why})")
    print(f"{len(TESTS) - bad}/{len(TESTS)} as expected")
    return bad


def hook():
    raw = sys.stdin.buffer.read().decode("utf-8", "replace")
    try:
        cmd = (json.loads(raw).get("tool_input") or {}).get("command") or ""
    except ValueError:
        cmd = raw
    try:
        ok, why = judge(cmd)
    except Exception as x:  # a guard that crashes must not let an aws call through
        ok, why = (not re.search(r"\baws\b|boto", cmd)), f"guard error {x!r}"
    if ok:
        return 0
    sys.stderr.write(f"Blocked by trainworld/tools/aws/guard.py: {why}. Raw AWS calls that change anything, or reach "
                     f"the friend's retired account, are refused in this project: {ADVICE}.\n")
    return 2


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--hook"]:
        sys.exit(hook())
    if a == ["--test"]:
        sys.exit(1 if test() else 0)
    if len(a) == 1:
        ok, why = judge(a[0])
        print(("ALLOW" if ok else "REFUSE") + f": {why}")
        sys.exit(0 if ok else 2)
    raise SystemExit(__doc__)
