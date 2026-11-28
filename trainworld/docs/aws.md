# AWS: rules and flow

Read this whole page before any AWS step. It is short on purpose.

**AWS means Anita's own account only** (account `537239093801`, CLI profile `anita-own`). Every raw `aws` call names
`--profile anita-own`; there is no default profile on this PC. The guard hook (`tools/aws/guard.py`, a PreToolUse hook
on Bash and PowerShell) refuses raw calls that create, change or destroy anything, and anything aimed at the friend's
account.

**No secrets anywhere.** The secret key stays in `~/.aws/credentials`; never in the repo, a log, user data, a note or
chat. Anita runs `aws configure` herself; agents never do.

## The friend's account: retired

The builds account a friend lent (account `429583250612`, IAM user `anita`) is no longer used (Anita, 2026-11-24,
T-4177). Its profile is gone from `~/.aws`; `guard.py` refuses its account number, `--profile default`, inline boto3
without `anita-own` and the scripts that used it; `awslib.py` refuses every call. Those scripts (`awslib.py`,
`status.py`, `bucket.py`, `launch.py`, `teardown.py`, `bundle.py`, `seed.py`, `wavepack.py`, `land.py`,
`userdata/wave.sh`, `proof.sh`, `terrain.sh`) stay as a record of the wave flow (notes/archive/t1000-1499/T-1064.md);
none of them may be pointed at another account without Anita's word. Nothing of ours is left there (T-4177).
No AWS builds until she sets up an account for them.

## Her account (`anita-own`, T-4166)

Anita's own account `537239093801` (paid plan), made 2026-11-23. CLI profile `anita-own`, IAM
user `bench` (EC2 and S3 full access). Setup she followed: notes/aws-own-account-setup.md.

1. **Timing runs**: perf_bench, edit_bench, smoke. This PC first; a machine there only when the local cores are
   busy (her 11-24).
2. **Only `tools/aws/bench.py` makes or removes anything there.** guard.py refuses raw changing calls whatever the
   profile. bench.py refuses unless the profile is `bench` in `537239093801`, and refuses the friend's account by
   number.
3. **$20 a month is a hard stop**: a Budgets action denies `bench` new machines at $20 of actual cost
   (notes/aws-budget-action.md); a machine already up ends by its timer. Billing data lags up to a day.
4. **One 4-vCPU machine at a time**, at most $0.25 an hour (spot first, then on-demand), at most 3 hours, with a hard
   timer: a shutdown set first thing at launch, a cron check of the deadline, and shutdown behaviour terminate. The quota raises were refused;
   on 2026-11-24 AWS enforced **1 vCPU on-demand and no spot**. Her 11-27: the raise to about 5 vCPU was refused too; AWS wants a usage history first, so use the account now and then for runs that fit (country adds, say) to build one.
5. **Nothing billing while idle**: no NAT gateway, VPC endpoint, load balancer, Elastic IP, snapshot, AMI, extra
   volume, IAM role or stack. What stays is the data cache in bucket `twbench-use2-r7k2m9`
   (private, tagged `project=twbench`; `runs/` objects expire in 3 days, `data/` in 45): about 60 MB, under a
   cent a month. Tags never name a person.
6. **Its numbers stay apart from home's.** Linux, a different CPU and WebGL on SwiftShader: results land in
   `artifacts/trainworld/awsbench/<run>/`, never in notes/perf-ledger.jsonl or edit-ledger.jsonl. Compare two
   builds within one run (`--base`), never against a home line.

**Timing runs go here by default** (her 11-28): perf_bench and edit_bench run on AWS, not after a quiet moment at home, and the account is used on purpose to build the usage history AWS asked for. Every machine gets one line in `notes/aws-usage.md` (date, purpose, type, vCPU, minutes, cost); tell her when the history looks long enough to ask for the raise again. As of 11-28 the account still allows 1 vCPU on-demand and no spot, so a 4-vCPU launch fails with VcpuLimitExceeded or MaxSpotInstanceCountExceeded.

The flow, from the maps root with `venv\Scripts\python.exe`:

- `trainworld\tools\aws\bench.py run --commit <rev> [--base <rev>] --perf nyc --edit nyc [--smoke] [--dry]` does all of
  it and waits: the commit's app and sim (`git archive`, the main tree's uncommitted app/src/game JSON as
  agent-tree.mjs adds it) with the main tree's bench tools; the data its saves read
  (`tools/aws/benchdata/<save>.txt`, uploaded once per change of content); launch; Node, Chrome, Rust and wasm-pack
  installed on the machine; each tree built with its own perf_bench (own Cargo targets, T-4167); the benches; the
  results fetched; the machine terminated; the run's objects deleted; cost and leftovers printed. About 35-40 minutes
  for perf and edit on nyc.
- A run whose results list paths in `missing-packs.txt` needs them added to the save's data list. A new save needs
  a data list: run perf_bench at home with `PERF_REQLOG=<file>` and keep its `/packs/` lines.
- `bench.py status`: anything up or stored. `bench.py teardown <run>`: a run whose session died. `bench.py setup`:
  the bucket and security group (done 2026-11-24).
