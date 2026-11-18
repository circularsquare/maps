# Night of 2026-11-17/18, batch 2 (T-3455) finished by nightfix3 after night1117b.ps1 stopped at wave 1's gate
# (8 rows parked: seven on a GEOS TopologyException in osm_units.reclaimed, mop over the 90-minute step limit; both
# fixed in osm_units.reclaimed). Reruns only those 8 rows, then batch 2's join with night1117b's own lists, then the
# night's after-steps and start tiers. Backup pre and setplanned already ran: never rerun them (backup pre would
# overwrite the undo point, setplanned would reset the 20 gated rows).
# Start (no window):
#   powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1118fix.cmd
# Stop: create trainworld\data\work\world_run\STOP, then kill this script's PID only (first line of night1118fix.state).
param([int]$Want = 9, [int]$Min = 6, [string]$Until = "11:00", [switch]$Rehearse, [switch]$JoinOnly,
      [string]$SimDir = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\76388cba-0327-41aa-a4e4-4e7d7376bf41\scratchpad\countries")
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Maps = "C:\Users\anita\projects\maps"
$TW = "$Maps\trainworld"
$Py = "$Maps\venv\Scripts\python.exe"
$Art = "$Maps\artifacts\trainworld"
$Pre = if ($Rehearse) { "night1118fix-rehearse" } else { "night1118fix" }
$Log = "$Art\$Pre.log"
$State = if ($Rehearse) { "$Art\night1118fix-rehearse.state" } else { "$Kit\night1118fix.state" }
$StopFile = "$TW\data\work\world_run\STOP"
$Port = 6372
$Name = "nightfix3"
Set-Location $Maps
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$Started = Get-Date
$UntilT = [datetime]::Today.Add([timespan]::Parse($Until))
if ($UntilT -lt $Started) { $UntilT = $UntilT.AddDays(1) }
$script:Stage = "start"
$J = $Want

function Say($m) {
  $l = "$(Get-Date -Format s) [$script:Stage] $m"
  Add-Content -Path $Log -Value $l -Encoding utf8
  Set-Content -Path $State -Value @("pid $PID", $l) -Encoding utf8
}
function Run($name, $argline, $out) {
  if ($Rehearse) { Say "(rehearse) $name : $argline > $out"; return 0 }
  Say "start $name : $argline > $out"
  $p = Start-Process -FilePath $Py -ArgumentList $argline -WorkingDirectory $Maps -RedirectStandardOutput $out -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
  Say "end $name : exit $($p.ExitCode)"
  return $p.ExitCode
}
function Claim($n, $what) {
  if ($Rehearse) { return $n }
  $o = & $Py -I "$Maps\cores.py" claim $n $Name --pid $PID $what 2>&1
  return [int](($o | Select-Object -Last 1).ToString().Trim())
}
function Release() { if ($Rehearse) { return }; & $Py -I "$Maps\cores.py" release $Name 2>&1 | Out-Null }
function Stopped() {
  if (Test-Path $StopFile) { Say "STOPPED: $StopFile exists (delete it and start again to go on)"; return $true }
  return $false
}
function Fits($min) {
  $end = (Get-Date).AddMinutes($min)
  if ($end -le $UntilT) { return $true }
  Say "skipped: about $min min would end $(Get-Date $end -Format HH:mm), after $(Get-Date $UntilT -Format HH:mm)"
  return $false
}
function WaitFree() {
  if ($Rehearse) { return $true }
  for ($i = 0; $i -lt 6; $i++) {
    $s = & $Py -I "$TW\pipeline\world_run.py" status 2>&1 | Out-String
    $locks = ($s -split "\r?\n") | Where-Object { $_ -match "^lock " }
    if (-not $locks) { return $true }
    Say "waiting for locks: $($locks -join ' | ')"
    Start-Sleep -Seconds 300
  }
  return $false
}
function Quit($code, $m) {
  Say $m
  Release
  exit $code
}

Say "night1118fix.ps1 started, pid $PID; until $(Get-Date $UntilT -Format 'yyyy-MM-dd HH:mm')$(if ($Rehearse) { ' (REHEARSAL)' })"
if (Stopped) { Quit 2 "STOPPED at start" }
if (-not (WaitFree)) { Quit 3 "STOPPED: locks still held after 30 minutes" }

# ---- the sim copy: the one given, else this night's own (as night1117b.ps1)
$sim = $SimDir
if ((Run "sim check" "-I $TW\tools\country\setup_sim.py $SimDir check" "$Art\$Pre-sim-check.log") -ne 0) {
  $sim = "$Art\night1117-sim"
  if (Test-Path "$sim\simroot\sim") { Run "sim refresh (own)" "-I $TW\tools\country\setup_sim.py $sim refresh" "$Art\$Pre-sim.log" | Out-Null }
  else { New-Item -ItemType Directory -Force $sim | Out-Null; Run "sim make (own)" "-I $TW\tools\country\setup_sim.py $sim make" "$Art\$Pre-sim.log" | Out-Null }
  if ((Run "sim check (own)" "-I $TW\tools\country\setup_sim.py $sim check" "$Art\$Pre-sim-check.log") -ne 0) { Quit 3 "STOPPED: no sim copy checks" }
} else {
  Run "sim refresh" "-I $TW\tools\country\setup_sim.py $SimDir refresh" "$Art\$Pre-sim.log" | Out-Null
}
$env:TW_SIM_DIR = "$sim\simroot\sim"
$env:TW_CARGO_TARGET_DIR = "$sim\simroot\sim\target"
$g = Claim $Want "T-3455 batch 2 finish"
if ($g -lt $Min) { Quit 3 "STOPPED: cores granted $g, want $Min or more" }
$J = [Math]::Min($g, $Want)
Say "cores granted $g, jobs $J"

# ---- batch 2: the 8 parked rows again, then the join
$script:Stage = "batch 2 T-3455"
$bc = "AE,BH,CN,EG,ID,IL,IN,JP,KP,KR,KW,MY,PS,QA,SA,SG,TH"
$all = "mop,bus,iks,too,asn,cuc,sne,msr,ale,shl,azh,isy,hud,dut,chn,ngt,mdt,thi,klt,srk,maa,joo,pnt,kut,kog,kuw,brm,doh"
$rows = "mop,shl,chn,ngt,mdt,klt,joo,kut"
if (-not $JoinOnly) {
  if (Stopped) { Quit 2 "STOPPED before the rerun; nothing joined" }
  $rc = Run "run T3455 parked rows" "-I $TW\pipeline\world_run.py run --rows $rows --jobs $J --port $Port --no-join" "$Art\$Pre-T3455-run.log"
  $gr = Run "gated T3455 all rows" "-I $Kit\gated.py $all" "$Art\$Pre-T3455-gated.log"
  if ($rc -ne 0 -or $gr -ne 0) {
    if (-not $Rehearse) { Say "rows: $((Get-Content "$Art\$Pre-T3455-gated.log") -join '; ')" }
    Quit 4 "STOPPED: rerun exit $rc, rows not all gated; no join (nightfix3 decides: relax, park with a note, or join the rest with -JoinOnly)"
  }
}
if (-not (Fits 150)) { Quit 2 "STOPPED: rows gated, the join would end after $Until; not joined" }
if (-not (WaitFree)) { Quit 3 "STOPPED: locks held before the T3455 join" }
if (Stopped) { Quit 2 "STOPPED before the T3455 join (rows gated, waiting)" }
if ((Run "backup prejoin T3455" "-I $TW\tools\country\backup.py T3455 prejoin $bc" "$Art\$Pre-T3455-backup-prejoin.log") -ne 0) { Quit 6 "STOPPED: prejoin backup failed; no join" }
$rc = Run "join T3455" "-I $TW\pipeline\world_run.py join --countries KR,EG,IN,ID,QA,KW,MY --jobs $J --port $Port" "$Art\$Pre-T3455-join.log"
Run "status T3455" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-T3455-status.log" | Out-Null
if ($rc -ne 0) { Quit 5 "STOPPED: T3455 join exit $rc; undo the join with tools\country\restore.py T3455 prejoin --go (the whole batch: T3455 pre --go)" }
if ((Run "stale T3455" "-I $TW\tools\country\stale.py" "$Art\$Pre-T3455-stale.log") -ne 0) {
  Quit 7 "STOPPED after the T3455 join: market packs name freed places ($Pre-T3455-stale.log); tools\country\stale.py --cmd writes the fix, run it before any other run"
}
Say "batch 2 joined"

# ---- after both batches, as night1117b.ps1 (it stopped before them), then start tiers (world-run.md "Start tiers")
$script:Stage = "after"
$blocks = "$Art\night1118fix-blocks.json"
Run "terrain blocks" "trainworld\pipeline\terrain_fine.py blocks $blocks" "$Art\$Pre-terrain-blocks.log" | Out-Null
Run "terrain fetch" "trainworld\pipeline\terrain_fine.py run $blocks trainworld\data\work\terrain-fine --jobs 1 --fetch 8" "$Art\$Pre-terrain-fine.log" | Out-Null
Run "zone_terrain" "trainworld\pipeline\zone_terrain.py build --jobs $J" "$Art\$Pre-zone-terrain.log" | Out-Null
Run "market_drives" "trainworld\pipeline\market_drives.py build --jobs $J" "$Art\$Pre-market-drives.log" | Out-Null
Run "build_inputs" "trainworld\checks\build_inputs.py" "$Art\$Pre-build-inputs.log" | Out-Null
Run "start tiers" "-I $TW\pipeline\start_difficulty.py --fresh --jobs $J" "$Art\$Pre-start-difficulty.log" | Out-Null
Run "status end" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status-end.log" | Out-Null
$script:Stage = "end"
Release
Say "all done"
