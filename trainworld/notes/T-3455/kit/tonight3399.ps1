# T-3399 overnight: Central America and Venezuela (stage 1), then T-617's gla, lis, pot (stage 2).
# Start: powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 <kit>\tonight.cmd
# Skip stage 2: create <kit>\NO-STAGE2 before stage 1 ends. Stop: trainworld\data\work\world_run\STOP (world_run steps
# stop at a clean point), then this script's PID (first line of <kit>\tonight.state).
param([int]$Want = 10, [int]$Min = 6)
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Maps = "C:\Users\anita\projects\maps"
$Py = "$Maps\venv\Scripts\python.exe"
$Art = "$Maps\artifacts\trainworld"
$Log = "$Art\t3399-tonight.log"
$State = "$Kit\tonight.state"
Set-Location $Maps
$env:PYTHONIOENCODING = "utf-8"
$env:TW_SIM_DIR = "$Kit\simroot\sim"
$env:TW_CARGO_TARGET_DIR = "$Kit\simroot\sim\target"
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$CA = "GT,BZ,SV,HN,NI,CR,PA,VE"

function Say($m) {
  $l = "$(Get-Date -Format s) $m"
  Add-Content -Path $Log -Value $l -Encoding utf8
  Set-Content -Path $State -Value @("pid $PID", $l) -Encoding utf8
}
function Run($name, $argline, $out) {
  Say "start $name : $argline > $out"
  $p = Start-Process -FilePath $Py -ArgumentList $argline -WorkingDirectory $Maps -RedirectStandardOutput $out -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
  Say "end $name : exit $($p.ExitCode)"
  return $p.ExitCode
}
function Claim($n) {
  $o = & $Py -I "$Maps\cores.py" claim $n countries-T3399 --pid $PID "T-3399 overnight world_run and join" 2>&1
  return [int](($o | Select-Object -Last 1).ToString().Trim())
}
function WaitFree() {
  # no other run or join may hold locks (shenzhen's shz rebuild, a light pass); wait up to 4 hours
  for ($i = 0; $i -lt 48; $i++) {
    $s = & $Py -I trainworld\pipeline\world_run.py status 2>&1 | Out-String
    $locks = ($s -split "\r?\n") | Where-Object { $_ -match "^lock " }
    if (-not $locks) { return $true }
    Say "waiting for locks: $($locks -join ' | ')"
    Start-Sleep -Seconds 300
  }
  return $false
}
function Jobs() {
  for ($i = 0; $i -lt 6; $i++) {
    $g = Claim $Want
    if ($g -ge $Min) { Say "cores granted $g"; return $g }
    Say "cores granted $g (want $Min or more), retrying in 10 min"
    Start-Sleep -Seconds 600
  }
  $g = Claim $Want
  Say "cores granted $g after an hour of retries; going on"
  return [Math]::Max($g, 2)
}
function After($tag) {
  Run "terrain blocks $tag" "trainworld\pipeline\terrain_fine.py blocks $Kit\blocks-$tag.json" "$Art\t3399-terrain-blocks-$tag.log" | Out-Null
  Run "terrain fetch $tag" "trainworld\pipeline\terrain_fine.py run $Kit\blocks-$tag.json trainworld\data\work\terrain-fine --jobs 1 --fetch 8" "$Art\t3399-terrain-fine-$tag.log" | Out-Null
  Run "zone_terrain $tag" "trainworld\pipeline\zone_terrain.py build --jobs $J" "$Art\t3399-zone-terrain-$tag.log" | Out-Null
  Run "market_drives $tag" "trainworld\pipeline\market_drives.py build --jobs $J" "$Art\t3399-market-drives-$tag.log" | Out-Null
  Run "build_inputs $tag" "trainworld\checks\build_inputs.py" "$Art\t3399-build-inputs-$tag.log" | Out-Null
}

Say "tonight.ps1 started, pid $PID"
if (-not (WaitFree)) { Say "STOPPED: locks still held after 4 hours"; exit 3 }
Run "sim refresh" "-I trainworld\tools\country\setup_sim.py $Kit refresh" "$Art\t3399-sim.log" | Out-Null
if ((Run "sim check" "-I trainworld\tools\country\setup_sim.py $Kit check" "$Art\t3399-sim-check.log") -ne 0) { Say "STOPPED: sim copy check failed"; exit 3 }
$J = Jobs

# ---- stage 1: the eight countries
Run "backup pre" "-I trainworld\tools\country\backup.py T3399 pre MX,CO" "$Art\t3399-backup-pre.log" | Out-Null
$rc = Run "run" "-I trainworld\pipeline\world_run.py run --countries $CA --limit 300 --jobs $J --port 6371 --no-join" "$Art\t3399-run.log"
if ($rc -ne 0) { Say "STOPPED: run exit $rc (2 is a run fault: the batch report says which); no join"; exit 4 }
if (-not (WaitFree)) { Say "STOPPED: locks held before the join"; exit 3 }
Run "backup prejoin" "-I trainworld\tools\country\backup.py T3399 prejoin $CA,MX,CO" "$Art\t3399-backup-prejoin.log" | Out-Null
$rc = Run "join" "-I trainworld\pipeline\world_run.py join --countries $CA --jobs $J" "$Art\t3399-join.log"
Run "status 1" "-I trainworld\pipeline\world_run.py status" "$Art\t3399-status-1.log" | Out-Null
if ($rc -ne 0) { Say "STOPPED: join exit $rc; undo with tools\country\restore.py T3399 prejoin --go"; exit 5 }
After "s1"
Say "stage 1 done"

# ---- stage 2: T-617, Glasgow, Lisbon and Porto rebuilt on today's unit matching
if (Test-Path "$Kit\NO-STAGE2") { Say "stage 2 skipped (NO-STAGE2)"; & $Py -I "$Maps\cores.py" release countries-T3399 | Out-Null; exit 0 }
if (-not (WaitFree)) { Say "STOPPED before stage 2: locks held"; exit 3 }
Run "backup pre2" "-I trainworld\tools\country\backup.py T3399 pre2 GB,PT,ES,IE" "$Art\t3399-backup-pre2.log" | Out-Null
Run "t617 planned" "-I $Kit\t617_planned.py gla,lis,pot --go" "$Art\t3399-t617-planned.log" | Out-Null
$rc = Run "run t617" "-I trainworld\pipeline\world_run.py run --rows gla,lis,pot --jobs $J --port 6371 --no-join" "$Art\t3399-run-t617.log"
if ($rc -ne 0) { Say "STOPPED: t617 run exit $rc; no join"; exit 6 }
Run "backup prejoin2" "-I trainworld\tools\country\backup.py T3399 prejoin2 GB,PT,ES,IE" "$Art\t3399-backup-prejoin2.log" | Out-Null
$rc = Run "join t617" "-I trainworld\pipeline\world_run.py join --countries GB,PT --jobs $J" "$Art\t3399-join-t617.log"
Run "status 2" "-I trainworld\pipeline\world_run.py status" "$Art\t3399-status-2.log" | Out-Null
if ($rc -ne 0) { Say "STOPPED: t617 join exit $rc; undo with tools\country\restore.py T3399 prejoin2 --go"; exit 7 }
After "s2"
& $Py -I "$Maps\cores.py" release countries-T3399 | Out-Null
Say "all done"
