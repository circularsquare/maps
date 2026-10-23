#!/bin/bash
# T-1065: one wave's builds, rendered by launch.py (--kind wave). Runs once as root at first boot.
# Files: GET_code (bundle.py: trainworld's code, plan, registry and wave/plan.txt), GET_reqs, GET_seed
# (optional: a tar of caches unpacked under trainworld/ first).
# GET_go: polled once the check row is sent; put from home (bucket.py put): "go", or anything else to stop.
# Puts: PUT_log, PUT_heartbeat, PUT_results_<name> for each line of wave/plan.txt.
# plan.txt: "#states <fips> ..." then lines "<name> <extracts,..> <ids,..> <jobs> <workers>"; the
# first line is the check row (rebuilt from scratch and compared at home before the go).
set -u
export HOME=/root
# ---- the hard timer, before anything else: shutdown terminates this machine and its disk
shutdown -h +{{MINUTES}}
echo "*/2 * * * * root [ \$(date +\%s) -gt {{DEADLINE_EPOCH}} ] && /sbin/shutdown -h now" > /etc/cron.d/tw-deadline
LOG=/var/log/tw-run.log
exec > >(tee -a "$LOG") 2>&1
{{URLS}}
W=/opt/tw; TW=$W/trainworld; L=$W/logs; S=$W/state; PLAN=$W/wave/plan.txt
mkdir -p $W $L $S
UA="trainworld-pipeline/0.1 (rail game data build; python-requests)"
put() { curl -sS --retry 3 -X PUT -T "$1" "$2" -o /dev/null -w "  put $(basename "$1"): %{http_code}\n" || true; }
beat() { date -u +%FT%TZ > /tmp/hb; put /tmp/hb "$PUT_heartbeat"
  { cat "$LOG"; for f in $L/*.log; do [ -e "$f" ] && { echo "---- tail $f"; tail -n 12 "$f"; }; done
    echo "---- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg); memory used $(free -g | awk '/Mem:/ {print $3}') GB; disk $(df -h / | awk 'NR==2 {print $5}')"
  } > /tmp/all.log; put /tmp/all.log "$PUT_log"; }
step() { echo; echo "== $(date -u +%FT%TZ) $*"; }
( while sleep 60; do beat >/dev/null 2>&1; done ) &
trap 'beat; shutdown -h now' EXIT
lines() { grep -v '^#' $PLAN; }

send() {  # one region's results, whatever is built so far
  local n=$1 v=PUT_results_$1 ids; ids=$(lines | awk -v n="$1" '$1 == n {print $3}')
  [ -n "${!v:-}" ] || { echo "no PUT url for $n"; return; }
  mkdir $S/sending_$n 2>/dev/null || { echo "$n is being sent already"; return; }  # one sender a region (the spot watcher and the region's end met in us-w1b)
  (cd $W && $PY -I $TW/tools/aws/wavepack.py "$n" "$ids" /tmp/results_$n.tgz $S/t0) && put /tmp/results_$n.tgz "${!v}" && touch $S/sent_$n
  rmdir $S/sending_$n
}
spotwatch() { while sleep 5; do
  t=$(curl -sS -m 2 -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" http://169.254.169.254/latest/api/token)
  if curl -fs -m 2 -H "X-aws-ec2-metadata-token: $t" http://169.254.169.254/latest/meta-data/spot/instance-action; then
    step "SPOT INTERRUPTION NOTICE: sending what is built"; touch $S/interrupted
    for n in $(lines | awk '{print $1}'); do [ -e $S/started_$n ] && [ ! -e $S/sent_$n ] && send $n; done
    beat; return; fi; done; }
runreg() {  # name extracts ids jobs workers
  touch $S/started_$1; step "region $1 starts: $3"
  (cd $TW && $PY -I pipeline/world_run.py run --rows "$3" --build-only --coarse 1000 --jobs $4 --workers $5 --keep-downloads > $L/run_$1.log 2>&1)
  step "region $1 ended ($?): $(grep -c ': built in' $L/run_$1.log) built, $(grep -c 'PARKED' $L/run_$1.log) parked"; send $1; }
readx() {  # one extract: download, then the region's OSM caches
  local e=$1 b=${1##*/} f=data/raw/osm/${1//\//_}.osm.pbf; echo "== $(date -u +%FT%TZ) read $e starts"
  { curl -fsSL --retry 4 -A "$UA" -o $f.part "https://download.geofabrik.de/$e-latest.osm.pbf" && mv $f.part $f &&
    $PY -I pipeline/world_run.py step @read_extract $e > $L/read_$b.log 2>&1 && touch $S/read_$b.ok; } || touch $S/read_$b.fail
  echo "== $(date -u +%FT%TZ) read $e: $(du -m $f 2>/dev/null | cut -f1) MB, $(cd $S; ls read_$b.*)"; }

step "run {{RUN}}: boot"
nproc; free -g | head -2; df -h / | tail -1
cd $W && curl -fsS -o code.tgz "$GET_code" && tar xzf code.tgz && curl -fsS -o reqs.txt "$GET_reqs" || exit 1
if [ -n "${GET_seed:-}" ]; then  # optional: caches from home under trainworld/ (the check row's OSM, so it reads what home read)
  curl -fsS -o seed.tgz "$GET_seed" && tar xzf seed.tgz -C $TW && rm seed.tgz || exit 1; fi
touch $S/t0; cat $PLAN
# T-1066: US builds read OSM from the extract only; Overpass is cut off so a query would fail at once, in the log
echo "127.0.0.1 overpass-api.de overpass.private.coffee overpass.kumi.systems maps.mail.ru" >> /etc/hosts

step "toolchains: apt, Python 3.9 and the pinned packages (uv), Rust and the pipeline's bins"
export DEBIAN_FRONTEND=noninteractive
apt-get -qq update && apt-get -qq install -y build-essential pkg-config >/dev/null || exit 1
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh
export PATH=/opt/uv:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy
uv python install 3.9 && uv venv -q -p 3.9 /opt/venv && uv pip install -q -p /opt/venv/bin/python -r reqs.txt || exit 1
PY=/opt/venv/bin/python
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal && . "$HOME/.cargo/env" || exit 1
cd $TW && mkdir -p data/raw/osm data/work data/packs
CARGO_TARGET_DIR=$TW/data/work/target cargo build -q --release --features demand --bin pack_gravity --bin pack_walk --manifest-path sim/Cargo.toml || exit 1
export PY UA L S
export -f readx
spotwatch &
( while sleep 5; do free -m | awk '/Mem:/ {print $3}'; done > $L/mem.txt ) &

step "the check row, and at once the wave's state caches and extracts (3 reads at a time beside the check row's)"
read -r CN CEX CIDS CJ CW < <(lines | head -1)
$PY -I - "$CIDS" <<'EOF'
import json, sys
p = "pipeline/world_plan.json"
d = json.load(open(p, encoding="utf-8"))
for r in d["metros"]:
    if r.get("units") and r["id"] in sys.argv[1].split(","):
        r["status"] = "planned"
        r.pop("run", None)
json.dump(d, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
EOF
runreg $CN $CEX $CIDS $CJ $CW &
CHECK=$!
STATES=$(awk '$1 == "#states" {$1 = ""; print}' $PLAN)
( $PY -I pipeline/world_run.py step usa state $STATES > $L/states.log 2>&1; echo "== $(date -u +%FT%TZ) state caches: $?" ) &
STPID=$!
lines | tail -n +2 | awk '{print $2}' | tr ',' '\n' | awk '!s[$0]++' | xargs -P 3 -I{} bash -c 'readx {}' &
wait $CHECK

step "waiting for go from home (the check row against its Windows build), at most 90 min"
GO=""
for i in $(seq 180); do curl -fsS -o /tmp/go "$GET_go" 2>/dev/null && { GO=$(tr -dc a-z < /tmp/go); break; }; sleep 30; done
echo "go: '${GO:-none}'"
[ "$GO" = go ] || { step "no go: stopping"; exit 0; }
wait $STPID

step "the wave's regions, each once its extracts are read"
TODO=$(lines | tail -n +2 | awk '{print $1}'); PIDS=""
while [ -n "$TODO" ]; do LEFT=""
  for n in $TODO; do ok=1
    for e in $(lines | awk -v n=$n '$1 == n {print $2}' | tr ',' ' '); do
      [ -e $S/read_${e##*/}.ok ] || [ -e $S/read_${e##*/}.fail ] || ok=0; done
    if [ $ok = 1 ]; then runreg $(lines | awk -v n=$n '$1 == n') & PIDS="$PIDS $!"; else LEFT="$LEFT $n"; fi
  done; TODO=$LEFT; [ -n "$TODO" ] && sleep 10; done
wait $PIDS
step "done: $(cd $S; ls | tr '\n' ' '); peak memory $(sort -n $L/mem.txt | tail -1) MB"
