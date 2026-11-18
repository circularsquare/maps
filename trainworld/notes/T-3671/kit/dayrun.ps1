# T-3671, a daytime rebuild batch on at most 6 cores (agent dayrun, 2026-11-18): cai takes Qasr Al-Nile, elm takes slw
# whole (Mansoura in one piece), clt rebuilt with anu retired into it. One join, EG and GT. notes/T-3671.md.
# Start (no window):
#   powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3671\kit\dayrun.cmd
# Stop: create trainworld\data\work\world_run\STOP, then kill this script's PID only (first line of dayrun.state).
# -From skips the steps before it: edits (backup pre and plan edits done), run, join (rows gated), after.
param([int]$Want = 6, [int]$Min = 4, [string]$From = "start", [int]$FloorGB = 5,
      [string]$SimDir = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\59c85afe-f6fc-46fa-b1d6-fdaa946e7f95\scratchpad\dayrun")
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Maps = "C:\Users\anita\projects\maps"
$TW = "$Maps\trainworld"
$Py = "$Maps\venv\Scripts\python.exe"
$Art = "$Maps\artifacts\trainworld"
$Pre = "dayrun"
$Log = "$Art\$Pre.log"
$State = "$Kit\dayrun.state"
$StopFile = "$TW\data\work\world_run\STOP"
$Port = 6374
$Name = "dayrun"
$Rows = "cai,elm,clt"
$Bk = "EG,SA,IL,PS,GT,MX,SV,HN,BZ"
Set-Location $Maps
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$script:Stage = "start"
$Order = @("start", "edits", "run", "join", "after")
function Reached($s) { return $Order.IndexOf($From) -le $Order.IndexOf($s) }

function Say($m) {
  $l = "$(Get-Date -Format s) [$script:Stage] $m"
  Add-Content -Path $Log -Value $l -Encoding utf8
  Set-Content -Path $State -Value @("pid $PID", $l) -Encoding utf8
}
function Run($name, $argline, $out) {
  Say "start $name : $argline > $out"
  $p = Start-Process -FilePath $Py -ArgumentList $argline -WorkingDirectory $Maps -RedirectStandardOutput $out -RedirectStandardError "$out.err" -NoNewWindow -PassThru
  $null = $p.Handle  # without the handle held, ExitCode reads empty after WaitForExit
  $p.WaitForExit()
  Say "end $name : exit $($p.ExitCode)"
  return $p.ExitCode
}
function Release() { & $Py -I "$Maps\cores.py" release $Name 2>&1 | Out-Null }
function Quit($code, $m) { Say $m; Release; exit $code }
function FreeGB() { return [math]::Round((Get-PSDrive C).Free / 1GB, 1) }
function Floor($what) { $f = FreeGB; if ($f -lt $FloorGB) { Quit 8 "STOPPED before $what : $f GB free, under $FloorGB" }; Say "disk $f GB free before $what" }

Say "dayrun.ps1 started, pid $PID, from $From"
if (Test-Path $StopFile) { Quit 2 "STOPPED: $StopFile exists" }
$s = & $Py -I "$TW\pipeline\world_run.py" status 2>&1 | Out-String
$locks = ($s -split "\r?\n") | Where-Object { $_ -match "^lock " }
if ($locks) { Quit 3 "STOPPED: locks held: $($locks -join ' | ')" }
if ((Run "sim check" "-I $TW\tools\country\setup_sim.py $SimDir check" "$Art\$Pre-sim-check.log") -ne 0) { Quit 3 "STOPPED: sim copy does not check" }
$env:TW_SIM_DIR = "$SimDir\simroot\sim"
$env:TW_CARGO_TARGET_DIR = "$SimDir\simroot\sim\target"
$o = & $Py -I "$Maps\cores.py" claim $Want $Name --pid $PID "T-3671 daytime rebuild batch" 2>&1
$g = [int](($o | Select-Object -Last 1).ToString().Trim())
if ($g -lt $Min) { Quit 3 "STOPPED: cores granted $g, want $Min or more" }
$J = [Math]::Min($g, $Want)
$env:NQ_JOBS = "$J"
Say "cores granted $g, jobs $J"

if (Reached "edits") {
  $script:Stage = "edits"
  Floor "backup pre"
  if ((Run "backup pre" "-I $TW\tools\country\backup.py T3671 pre $Bk" "$Art\$Pre-backup-pre.log") -ne 0) { Quit 6 "STOPPED: backup pre failed" }
  Say "PACK WRITES FROM HERE (retires, run, join)"
  foreach ($sp in @("egy", "gtm")) {
    if ((Run "edit $sp" "-I $Kit\plan_edit.py $Kit\t3671-$sp.json --go" "$Art\$Pre-edit-$sp.log") -ne 0) { Quit 4 "STOPPED: plan edit $sp failed; restore.py T3671 pre --go" }
  }
  foreach ($sp in @("egy", "gtm")) {
    if ((Run "retire $sp" "-I $Kit\plan_edit.py $Kit\t3671-$sp.json retire --go" "$Art\$Pre-retire-$sp.log") -ne 0) { Quit 4 "STOPPED: retire $sp failed; restore.py T3671 pre --go" }
  }
}

if (Reached "run") {
  $script:Stage = "run"
  $rc = Run "run rows" "-I $TW\pipeline\world_run.py run --rows $Rows --jobs $J --port $Port --no-join" "$Art\$Pre-run.log"
  $gr = Run "gated" "-I $Kit\gated.py $Rows" "$Art\$Pre-gated.log"
  if ($rc -ne 0) { Quit 4 "STOPPED: run exit $rc ($Pre-run.log); no join" }
  if ($gr -ne 0) {
    # cai and elm are shipped metros: a park leaves a hole, so no join and an undo; a clt park alone costs nothing
    $cg = Run "gated cai,elm" "-I $Kit\gated.py cai,elm" "$Art\$Pre-gated-metros.log"
    if ($cg -ne 0) { Quit 4 "STOPPED: cai or elm not gated ($Pre-gated.log); no join. Undo: restore.py T3671 pre --go" }
    Say "clt not gated (it stays parked, dead space as before); joining cai and elm"
  }
}

if (Reached "join") {
  $script:Stage = "join"
  Floor "backup prejoin"
  if ((Run "backup prejoin" "-I $TW\tools\country\backup.py T3671 prejoin $Bk" "$Art\$Pre-backup-prejoin.log") -ne 0) { Quit 6 "STOPPED: prejoin backup failed; no join" }
  Floor "join"
  $rc = Run "join" "-I $TW\pipeline\world_run.py join --countries EG,GT --jobs $J --port $Port" "$Art\$Pre-join.log"
  Run "status" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status.log" | Out-Null
  if ($rc -ne 0) { Quit 5 "STOPPED: join exit $rc; undo with tools\country\restore.py T3671 prejoin --go" }
  if ((Run "stale" "-I $TW\tools\country\stale.py" "$Art\$Pre-stale.log") -ne 0) { Quit 7 "STOPPED after the join: market packs name freed places ($Pre-stale.log)" }
  Run "successors dry" "-I $TW\pipeline\zone_successors.py --dry" "$Art\$Pre-successors-dry.log" | Out-Null
  Run "successors" "-I $TW\pipeline\zone_successors.py" "$Art\$Pre-successors.log" | Out-Null
  Say "joined"
}

if (Reached "after") {
  $script:Stage = "after"
  Run "terrain blocks" "trainworld\pipeline\terrain_fine.py blocks $Kit\blocks.json" "$Art\$Pre-terrain-blocks.log" | Out-Null
  Run "terrain fine" "trainworld\pipeline\terrain_fine.py run $Kit\blocks.json trainworld\data\work\terrain-fine --jobs 1 --fetch 8" "$Art\$Pre-terrain-fine.log" | Out-Null
  Run "zone_terrain" "trainworld\pipeline\zone_terrain.py build --jobs $J" "$Art\$Pre-zone-terrain.log" | Out-Null
  Run "market_drives" "trainworld\pipeline\market_drives.py build --jobs $J" "$Art\$Pre-market-drives.log" | Out-Null
  Say "PACKS FREE: no more pack or terrain writes from this run"
  Run "status end" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status-end.log" | Out-Null
  Run "build_inputs" "trainworld\checks\build_inputs.py" "$Art\$Pre-build-inputs.log" | Out-Null
  Run "start tiers" "$TW\pipeline\start_difficulty.py --fresh --jobs $J" "$Art\$Pre-start-difficulty.log" | Out-Null
}
$script:Stage = "end"
Release
Say "all done"
