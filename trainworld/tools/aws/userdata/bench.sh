#!/bin/bash
# T-4166: the bench machine (tools/aws/bench.py; docs/aws.md "Her own account"). The timer comes first: whatever
# happens below, the machine shuts down at it, and a shutdown terminates it, disk included.
shutdown -h +{{MINUTES}} "twbench timer"
echo "* * * * * root [ \$(date +\%s) -ge {{DEADLINE_EPOCH}} ] && /sbin/shutdown -h now" > /etc/cron.d/twbench-deadline
{{URLS}}
RUN={{RUN}}
LOG=/var/log/twbench.log
OUT=/w/out
mkdir -p $OUT
exec >>$LOG 2>&1
put() { curl -s -f -m 300 --retry 3 -X PUT --upload-file "$2" "$1" >/dev/null || echo "!! upload of $2 failed"; }
( while sleep 60; do put "$PUT_log" $LOG; done ) &
step() { echo "== $(date -u +%H:%M:%S) $*"; }
NODE=v22.23.2
WASM_PACK=v0.15.0
RUST=1.99.0
PW=1.61.1

main() {
  set -euo pipefail
  export HOME=/root DEBIAN_FRONTEND=noninteractive
  step "setup on $(nproc) vCPUs, $(free -g | awk '/Mem/ {print $2}') GB"
  mkdir -p /w/head /w/data
  curl -sSf -m 600 "$GET_data" | tar xz -C /w/data
  for t in head base; do
    v=GET_$t; [ -n "${!v:-}" ] || continue
    mkdir -p /w/$t; curl -sSf -m 300 "${!v}" | tar xz -C /w/$t
    ln -sfn /w/data /w/$t/trainworld/data
  done
  curl -sSf -m 300 https://nodejs.org/dist/$NODE/node-$NODE-linux-x64.tar.xz | tar xJ -C /opt
  export PATH=/opt/node-$NODE-linux-x64/bin:$PATH
  ln -sf /usr/bin/python3 /usr/local/bin/python
  step chrome
  curl -sSf -m 300 -o /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  apt-get update -q >/dev/null
  apt-get install -y -q /tmp/chrome.deb >/dev/null
  google-chrome --version
  step rust
  curl -sSf -m 300 https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain $RUST --target wasm32-unknown-unknown
  ln -sf /root/.cargo/bin/* /usr/local/bin/
  for u in https://github.com/drager/wasm-pack/releases/download/$WASM_PACK/wasm-pack-$WASM_PACK-x86_64-unknown-linux-musl.tar.gz \
           https://github.com/rustwasm/wasm-pack/releases/download/$WASM_PACK/wasm-pack-$WASM_PACK-x86_64-unknown-linux-musl.tar.gz; do
    curl -sSfL -m 120 "$u" | tar xz -C /tmp && break
  done
  cp /tmp/wasm-pack-$WASM_PACK-x86_64-unknown-linux-musl/wasm-pack /usr/local/bin/ || cargo install -q wasm-pack --version ${WASM_PACK#v} --locked
  rustc --version; wasm-pack --version
  step npm
  (cd /w/head/trainworld/app && npm ci --no-audit --no-fund --loglevel=error)
  if [ -d /w/base ]; then
    if cmp -s /w/head/trainworld/app/package-lock.json /w/base/trainworld/app/package-lock.json; then ln -s /w/head/trainworld/app/node_modules /w/base/trainworld/app/node_modules
    else (cd /w/base/trainworld/app && npm ci --no-audit --no-fund --loglevel=error); fi
  fi
  npm i --prefix /opt/pw --no-audit --no-fund --loglevel=error playwright-core@$PW
  export PLAYWRIGHT_MODULE=file:///opt/pw/node_modules/playwright-core/index.mjs
  N=$(nproc); MIN=$(( N < 3 ? N : 3 ))
  for t in head base; do
    [ -d /w/$t ] || continue
    python /w/$t/cores.py cap $N >/dev/null
    step "build $t"
    # each tree builds with its own perf_bench, so each has its own Cargo target dirs (T-4167)
    (cd /w/$t && node trainworld/tools/perf_bench.mjs build /w/$t --label $t 2>&1 | grep -v -e "pack missing" -e "place list missing" | tail -40)
  done
  HD=/w/head/artifacts/trainworld/perfbench/dist/head
  BD=/w/base/artifacts/trainworld/perfbench/dist/base
  cd /w/head
  for j in $JOBS; do
    s=${j#*:}
    case $j in
      perf:*)
        step "perf $s"
        if [ -d /w/base ]; then PERF_REQLOG=$OUT/req-perf-$s.txt node trainworld/tools/perf_bench.mjs compare $BD $HD --saves $s --runs $RUNS --min-cores $MIN 2>&1 | tee $OUT/perf-$s.txt || echo "!! perf $s failed"
        else PERF_REQLOG=$OUT/req-perf-$s.txt node trainworld/tools/perf_bench.mjs run $HD --label head --saves $s --runs $RUNS --no-ledger --min-cores $MIN 2>&1 | tee $OUT/perf-$s.txt || echo "!! perf $s failed"; fi ;;
      edit:*)
        step "edit $s"
        for t in base head; do
          [ -d /w/$t ] || continue
          d=/w/$t/artifacts/trainworld/perfbench/dist/$t
          REQLOG=$OUT/req-edit-$s-$t.txt node trainworld/checks/static-serve.mjs $d 6158 & srv=$!
          sleep 2
          cmp=""; [ $t = head ] && [ -d /w/base ] && cmp="--compare base"
          TW_CORE_MASK=$(( (1 << N) - 1 )) node trainworld/tools/edit_bench.mjs --url http://127.0.0.1:6158/ --save $s --build $t --label $t --cores $N $cmp 2>&1 | tee $OUT/edit-$s-$t.txt || echo "!! edit $s $t failed"
          kill $srv || true
        done ;;
      smoke)
        step smoke
        TW_CORES=$N node trainworld/checks/smoke.mjs 2>&1 | tee $OUT/smoke.txt || echo "!! smoke failed" ;;
    esac
  done
  grep -h "^404 /packs/" $OUT/req-*.txt 2>/dev/null | sort | uniq -c > $OUT/missing-packs.txt || true
  mkdir -p $OUT/perfbench $OUT/editbench
  cp -r /w/head/artifacts/trainworld/perfbench/runs $OUT/perfbench/ 2>/dev/null || true
  cp /w/head/artifacts/trainworld/perfbench/compare-*.json $OUT/perfbench/ 2>/dev/null || true
  cp /w/head/trainworld/notes/edit-ledger.jsonl $OUT/editbench/ 2>/dev/null || true
  cp /w/*/artifacts/trainworld/perfbench/dist/*/perf-build.json $OUT/ 2>/dev/null || true
  google-chrome --version > $OUT/machine.txt; uname -a >> $OUT/machine.txt; lscpu | grep -e "Model name" -e "^CPU(s)" >> $OUT/machine.txt
}

( main ); rc=$?
step "done $rc"
tar czf /tmp/results.tgz -C $OUT .
put "$PUT_results" /tmp/results.tgz
echo "$rc" > /tmp/done
put "$PUT_log" $LOG
put "$PUT_done" /tmp/done
shutdown -h now
