# AWS: rules and flow (T-1064)

Read this whole page before any AWS step. It is short on purpose.

Anita's IAM user `anita` lives in a **friend's** account, shared for free. The user has
**AdministratorAccess on the whole account**: nothing in AWS stops a command from stopping the
friend's machine or emptying her buckets. The guards are the scripts in `tools/aws/` and a hook
(`tools/aws/guard.py`) that blocks raw `aws` calls which create, change or destroy anything.
Design, sizes and costs: notes/T-1064.md.

## Rules

1. **Only long data builds go on AWS** (Anita, 2026-10-23). No servers, no hosting, no publishing.
2. **Only `tools/aws/*.py` create or destroy.** Raw `aws ... create-*/run-instances/terminate-*/
   delete-*/s3 rm/s3 sync` are refused by the hook; never work round it (boto3 one-liners, a
   script of your own, another shell). If the scripts cannot do something, add it to them, with the
   same tag checks, and say so in your report.
3. **Ours means tagged `owner=anita` and `project=anitabuilder`.** Every resource is tagged at
   creation; every change looks the id up again and refuses without both tags. Never act on a
   filter that could match an untagged thing, never "all".
4. **The friend's things are never read, changed or used**: her EC2 instance, her buckets
   (ancientgreek, cdk-..., ckopx-research, filmbro-db-staging, hypertext-db, jeanland-metaphors,
   linktree-crawler-data, and any other not named `anitabuilder-builds-*`), her security groups.
   Our one bucket is `anitabuilder-builds-use2-k7q4`; no person uploads to it by hand.
5. **One region, `us-east-2` (Ohio).** The scripts force it. A listing in the wrong region looks
   empty while a machine runs elsewhere: `status.py --all-regions` checks every region.
6. **Spot, one-time, with a hard timer, one machine at a time.** `launch.py` sets a shutdown at the
   timer (at most 12 h) from user data, a cron check of the deadline that survives a reboot, and
   shutdown behaviour terminate, so a forgotten machine dies with its disk. It refuses while any
   machine of ours is live.
7. **Nothing that bills while idle**: no NAT gateway, VPC endpoint, load balancer, Elastic IP,
   snapshot, AMI, extra volume, IAM role or CloudFormation stack. The default VPC, a security group
   with no inbound rules, and presigned URLs for the bucket are all a run needs.
8. **No secrets anywhere.** The secret key stays in `~/.aws/credentials`; never in the repo, a log,
   user data, a note or chat. The machine gets presigned URLs (one key each, expiring), never keys.
9. **Results land in the inbox, not the tree.** `bucket.py get` writes to
   `artifacts/trainworld/aws/inbox/<run>/`; a landing step (T-1065) moves rows into `data/` with
   the plan lock, never mid-batch, never into `data/packs` by hand. Nothing goes to R2 or the site
   from this flow: packs are published only on Anita's word.
10. **Every run ends with `teardown.py <run>`** and its listing in your report: hours, spot price,
    dollars, and "NOTHING OF OURS IS RUNNING OR BILLING". If a session may die mid-run, say in the
    task's notes which run is up and when its timer ends.

## The flow

All commands from the maps root with `venv\Scripts\python.exe` (botocore is there). If `aws` is
not found, reload PATH: `$env:PATH = [Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH','User')`.

1. `trainworld\tools\aws\status.py` first. Anything of ours already up? Find out whose run it is
   (its `run` tag, `artifacts/trainworld/aws/<run>.json`, the task notes) before anything else.
2. `trainworld\tools\aws\bucket.py ensure` (once; it exists since 2026-10-23).
3. Build the inputs at home: `bundle.py <code.tgz> <plan.txt>` packs the working tree's pipeline,
   sim, checks, tools/aws and app/src/game JSON (plan and registry included) with the wave plan
   (`#states <fips> ...`, then `<name> <extracts> <ids> <jobs> <workers>` a runner region a line, the
   first the check row). Everything else the machine downloads from the original sources.
4. `trainworld\tools\aws\launch.py <run> --kind wave --type m7a.8xlarge --hours 3 --disk 100
   --file code=<code.tgz> --file reqs=trainworld\tools\aws\requirements.txt --get go
   --put results_<name> ...` (one put a plan line) with a new run name; `--dry` first prints the zone,
   price, bid cap and AMI. `userdata/wave.sh` sets up, builds the check row while it makes the state
   caches and reads the extracts (four reads at most at once), sends `results_<check>`, then waits up
   to 90 minutes for `<run>/go`: compare the check row with a Windows build, then
   `bucket.py put <a file holding "go"> <run>/go` (anything else stops it). Each region is built with
   `world_run.py run --build-only` and sent as it ends; a spot notice sends what is built. For a
   machine with no check to make, put the go right after the launch.
5. Watch without waiting in the foreground: `bucket.py get <run>/log` (the machine puts its log
   every minute) or `status.py`. A run that has gone quiet past its expected time is torn down, not
   waited on.
6. When the log says done (or the machine is gone): `bucket.py get <run>/<results>` for each result.
7. `trainworld\tools\aws\teardown.py <run>`: terminates the machine if it is still up, sweeps our
   leftovers, prints the cost and the listing, exits 1 if anything of ours is left.
8. Land and gate at home: back up (`notes/t1018/us/backup.py`), `land.py <run>` (rows the machine
   left built go into data/ and turn "built" under the plan lock; parked rows are listed with their
   causes), then `world_run.py run --rows ... --no-join` gates them (a built row with its packs skips
   the download and build), then the join; then `bucket.py rm <run>` once the results are checked.
   The bucket's objects expire after 14 days anyway.

## Mistakes we foresee

| Mistake | What it would cost or break | What prevents it |
|---|---|---|
| A raw `aws ec2 terminate-instances` or `stop-instances`, or a filter that also matches the friend's machine | Her work lost; her trust | guard.py blocks raw changing calls; teardown.py terminates only ids it re-read with both tags |
| `aws s3 rm --recursive`, `s3 sync --delete`, `s3 rb` on the wrong bucket or prefix | Her data gone, no undo | guard.py blocks every raw `s3` write; bucket.py has our bucket's name built in, checks its tags, and deletes only `<run>/` keys |
| Listing or reading the friend's buckets "to look" | A breach of what was lent | Rule 4; guard.py refuses `s3 ls`/`s3api` on a bucket not named `anitabuilder-builds-*` |
| Launching on-demand, or a type with burstable credits (t3, t4g) | 3-10 times the price; surprise credit charges | launch.py is spot only and refuses `t*` types and bids over $2/h |
| A persistent spot request | It relaunches after the timer kills the machine, for ever | launch.py asks for one-time only; guard.py blocks raw `request-spot-instances`; teardown.py cancels any open request of ours |
| No timer, or a session dying mid-run | A 32-core machine billing for days unseen | The timer is in user data before anything else, plus a cron deadline check and shutdown = terminate; status.py flags OVERDUE |
| EBS volumes, snapshots, AMIs, Elastic IPs left behind | A few dollars a month each, unseen | Root disk deletes on termination; nothing makes snapshots, AMIs or EIPs; teardown.py sweeps any tagged ones and lists |
| A NAT gateway, VPC endpoint, load balancer or IAM role from a tutorial | NAT about $33 a month idle; roles widen access | Rule 7; guard.py blocks the raw creates; the machine needs only outbound internet and presigned URLs |
| Pulling GBs out of AWS again and again | $0.09 a GB | bucket.py get refuses over 5 GB without `--big`; results are fetched once into the inbox |
| The wrong region | Listings look empty while a machine runs | awslib forces us-east-2; teardown ends with `status.py --all-regions` |
| Secrets in logs or user data | The account open to anyone who sees them | Presigned URLs only; the AWS_* environment is cleared; guard.py refuses `configure get` of keys and `export-credentials`; launch.py keeps only a redacted copy of user data |
| A second agent launching while a run is live | Two machines, clashing results | launch.py refuses while any machine of ours is live, and holds a local launch lock |
| Reusing a run name | Results overwritten, cost reports mixed | launch.py refuses a name with a record in `artifacts/trainworld/aws/` |
| Copying packs from the inbox straight into `data/packs` mid-batch | A live run's gate or join sees half a batch; the plan disagrees with disk | Rule 9; landing goes through T-1065's step with the plan lock and no live run |
| Deleting a run's objects before the results are home | The run paid for, nothing to show | bucket.py rm refuses while a run's result keys are unfetched |
| Publishing from AWS, or pushing packs to R2 | Unreviewed data on the site | Rule 9; nothing in tools/aws/ touches R2 or the site |
| A machine too small, or too many extract reads at once | A read asks GDAL for a 6 GB node index a reader (26 GB for Tennessee): on 4 GB the extract does not open, and six reads at once overrun 128 GB | Builds need 1.1 GB at most, reads 26 GB: no t-class or micro, at most 4 reads at once on an m7a.8xlarge (notes/T-1064.md) |
| A result tar written by two senders at once (the spot watcher and the region's end met in us-w1b) | A corrupt tar: the region's rows rebuilt | wave.sh lets one sender a region at a time; wavepack.py writes under a temporary name of its own; land.py checks every file against the manifest's SHA-256 |
| Trusting a run that went quiet | Hours billed for nothing | Rule 10 and step 5: tear it down, read its log, relaunch smaller |
