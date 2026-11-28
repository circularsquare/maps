# AWS usage record

Every machine launched in Anita's own account (anita-own, 537239093801) by tools/aws/bench.py, one line each. bench.py prints
the minutes and cost when it ends a run but keeps no log of its own, so the agent who runs it adds the line here.
The account needs a usage history before AWS will raise the vCPU quota (her 11-27, 11-28).

| date | purpose | instance type | vCPU | minutes | rough cost |
|---|---|---|---|---|---|
| 2026-11-28 | perf_bench nyc + edit_bench nyc at 17de3100ed (run b20261128-132917): no machine started | m8a.xlarge (wanted) | 4 | 0 | $0 |
