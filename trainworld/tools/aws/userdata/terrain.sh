#!/bin/bash
# T-1854: the fine terrain intermediate, rendered by launch.py (--kind terrain). Runs once as root at first boot.
# Files: GET_code (a tgz with trainworld/pipeline/terrain_fine.py and blocks.json), GET_reqs.
# Puts: PUT_log, PUT_heartbeat, PUT_manifest, PUT_part0 .. PUT_part<n> (one per part; parts are cut at
# --part-gb so they fit the puts given). Nothing else: no go to wait for, no check row.
set -u
export HOME=/root
# ---- the hard timer, before anything else: shutdown terminates this machine and its disk
shutdown -h +{{MINUTES}}
echo "*/2 * * * * root [ \$(date +\%s) -gt {{DEADLINE_EPOCH}} ] && /sbin/shutdown -h now" > /etc/cron.d/tw-deadline
LOG=/var/log/tw-run.log
exec > >(tee -a "$LOG") 2>&1
{{URLS}}
W=/opt/tw; mkdir -p $W
put() { curl -sS --retry 5 -X PUT -T "$1" "$2" -o /dev/null -w "  put $(basename "$1"): %{http_code}\n"; }
beat() { date -u +%FT%TZ > /tmp/hb; put /tmp/hb "$PUT_heartbeat"
  { cat "$LOG"; [ -e $W/run.log ] && { echo "---- tail run.log"; tail -n 15 $W/run.log; }
    echo "---- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg); memory used $(free -g | awk '/Mem:/ {print $3}') GB; disk $(df -h / | awk 'NR==2 {print $5}')"
  } > /tmp/all.log; put /tmp/all.log "$PUT_log"; }
step() { echo; echo "== $(date -u +%FT%TZ) $*"; }
( while sleep 60; do beat >/dev/null 2>&1; done ) &
trap 'beat; shutdown -h now' EXIT

step "run {{RUN}}: boot"
nproc; free -g | head -2; df -h / | tail -1
cd $W && curl -fsS -o code.tgz "$GET_code" && tar xzf code.tgz && curl -fsS -o reqs.txt "$GET_reqs" || exit 1

step "Python 3.9 and the pinned packages (uv)"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh
export PATH=/opt/uv:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy
uv python install 3.9 && uv venv -q -p 3.9 /opt/venv && uv pip install -q -p /opt/venv/bin/python -r reqs.txt || exit 1
PY=/opt/venv/bin/python

# the sender terrain_fine.py calls with (part file, part number): the part, then the manifest so far
cat > $W/send.sh <<'EOF'
#!/bin/bash
v=PUT_part$2
[ -n "${!v:-}" ] || { echo "no PUT url for part $2"; exit 1; }
curl -sS --retry 5 -X PUT -T "$1" "${!v}" -o /dev/null -w "  put $(basename "$1"): %{http_code}\n" -f || exit 1
curl -sS --retry 5 -X PUT -T "$(dirname "$1")/manifest.json" "$PUT_manifest" -o /dev/null -w "  put manifest: %{http_code}\n"
EOF
chmod +x $W/send.sh
for v in $(compgen -v PUT_); do export "$v"; done

step "terrain_fine.py run: $(nproc) workers, 24 fetchers"
cd $W/trainworld && $PY -I pipeline/terrain_fine.py run $W/blocks.json $W/out --jobs $(nproc) --fetch 24 --part-gb 2.0 --send $W/send.sh > $W/run.log 2>&1
step "run ended ($?)"; tail -n 5 $W/run.log
put $W/out/manifest.json "$PUT_manifest"
step "done; shutting down (terminates)"
