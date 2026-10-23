#!/bin/bash
# T-1064's proof, rendered by launch.py (--kind proof). Runs once as root at first boot.
# Files: GET_code (git archive of pipeline, sim, checks), GET_reqs (requirements.txt), GET_hello.
# Puts: PUT_log, PUT_heartbeat, PUT_hello_back, PUT_results.
set -u
export HOME=/root
# ---- the hard timer, before anything else: shutdown terminates this machine and its disk
shutdown -h +{{MINUTES}}
echo "*/2 * * * * root [ \$(date +\%s) -gt {{DEADLINE_EPOCH}} ] && /sbin/shutdown -h now" > /etc/cron.d/tw-deadline
LOG=/var/log/tw-run.log
exec > >(tee -a "$LOG") 2>&1
{{URLS}}
put() { curl -sS --retry 3 -X PUT -T "$1" "$2" -o /dev/null -w "  put $(basename "$1"): %{http_code}\n" || true; }
beat() { date -u +%FT%TZ > /tmp/hb; put /tmp/hb "$PUT_heartbeat"; put "$LOG" "$PUT_log"; }
step() { echo; echo "== $(date -u +%FT%TZ) $*"; }
( while sleep 60; do beat >/dev/null 2>&1; done ) &
trap 'beat; shutdown -h now' EXIT
UA="trainworld-pipeline/0.1 (rail game data build; python-requests)"

step "run {{RUN}}: boot"
uname -a; nproc; free -g | head -2; df -h / | tail -1
W=/opt/tw; mkdir -p $W; cd $W

step "code and a file from our bucket (presigned GET)"
curl -fsS -o code.tar.gz "$GET_code" && tar xzf code.tar.gz && ls trainworld
curl -fsS -o reqs.txt "$GET_reqs"
curl -fsS -o hello.txt "$GET_hello" && sha256sum hello.txt | tee hello_back.txt
uname -a >> hello_back.txt
put hello_back.txt "$PUT_hello_back"

step "sources reachable from here"
for u in https://download.geofabrik.de/north-america/us/rhode-island-latest.osm.pbf \
         https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_44_tabblock20.zip \
         https://lehd.ces.census.gov/data/lodes/LODES8/ri/wac/ri_wac_S000_JT00_2023.csv.gz \
         https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R6_C11.zip \
         https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip \
         https://overpass-api.de/api/status \
         https://query.wikidata.org/ \
         https://api.worldbank.org/v2/country/US?format=json \
         https://davidmegginson.github.io/ourairports-data/airports.csv; do
  curl -sS -o /dev/null -I -L -A "$UA" --max-time 30 -w "%{http_code} %{time_total}s $u\n" "$u" || echo "FAIL $u"
done
curl -sS -o /dev/null -L -A "$UA" --max-time 60 -w "GET %{http_code} %{time_total}s overpass status\n" https://overpass-api.de/api/status || echo "FAIL overpass GET"

step "C toolchain (Rust's linker and build scripts need cc)"
export DEBIAN_FRONTEND=noninteractive
time apt-get -qq update && time apt-get -qq install -y build-essential pkg-config time bzip2 >/dev/null && cc --version | head -1

step "Python 3.9 and the pinned packages (uv)"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh
export PATH=/opt/uv:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy
uv python install 3.9 && uv venv -q -p 3.9 /opt/venv && uv pip install -q -p /opt/venv/bin/python -r reqs.txt
PY=/opt/venv/bin/python
$PY -V

step "imports"
cd $W/trainworld
$PY -I -c "import sys; sys.path[:0] = ['pipeline', 'checks']
import build_city, usa, tier0, routes, merge_cells, nocar, everyday_rates, coarse_zones, places, osm_features
import world_run, shared_lock, register, pack_gate
print('imports OK')"

mkdir -p data/raw/osm data/work data/packs   # git archive carries no empty folders; the pipeline assumes them

step "GDAL in the pyogrio wheel: the OSM driver (the wheel's GDAL 3.10.3 has it, as at home)"
$PY -I -c "import pyogrio; d = pyogrio.list_drivers(); print(pyogrio.__version__, pyogrio.__gdal_version_string__, 'OSM:', d.get('OSM'), len(d), 'drivers')"
# memory used by the whole machine every 2 s, for the read's peak (osm_features asks GDAL for a
# 6,000 MB in-memory node index a reader; a 4 GB machine cannot open the extract at all)
( while sleep 2; do free -m | awk '/Mem:/ {print $3}'; done > /tmp/mem_used.txt ) &
MEMPID=$!

step "pipeline step: usa.py state 44 (Rhode Island's blocks, PL 94-171 and LODES, from census.gov)"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2
/usr/bin/time -v $PY -I pipeline/world_run.py step usa state 44 2>&1 | grep -v "^\s*\(Average\|Exit\|Page\|Socket\|Signals\|Swaps\|File system\|Voluntary\|Involuntary\|Minor\|Major\|Percent\|Command\)"
ls -la data/work/usa data/raw | head -20

step "pipeline step: Geofabrik download and read_extract (Tennessee, a wave 1 state)"
time curl -fsSL -A "$UA" -o data/raw/osm/north-america_us_tennessee.osm.pbf https://download.geofabrik.de/north-america/us/tennessee-latest.osm.pbf
ls -la data/raw/osm
free -m | awk '/Mem:/ {print "memory used before the read: " $3 " MB of " $2}'
time $PY -I pipeline/world_run.py step @read_extract north-america/us/tennessee
sort -n /tmp/mem_used.txt | tail -1 | awk '{print "peak memory used by the machine during the steps: " $1 " MB"}'
ls -la data/raw/osm

step "GHSL tile download (30 MB)"
time curl -fsSL -A "$UA" -o /tmp/ghsl.zip https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R6_C11.zip
ls -la /tmp/ghsl.zip

step "Rust: rustup, then pack_gravity and pack_walk"
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
. "$HOME/.cargo/env"
time cargo build -q --release --features demand --bin pack_gravity --bin pack_walk --manifest-path sim/Cargo.toml
ls -la sim/target/release/pack_gravity sim/target/release/pack_walk

step "results back to the bucket (presigned PUT)"
kill $MEMPID 2>/dev/null
tar czf /tmp/results.tgz data/work/usa data/raw/osm/*.stations.fgb /tmp/mem_used.txt 2>/dev/null
ls -la /tmp/results.tgz
put /tmp/results.tgz "$PUT_results"

step "done; shutting down (terminates)"
