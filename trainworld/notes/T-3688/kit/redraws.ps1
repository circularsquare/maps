# T-3688 overnight redraws (agent redraws, 2026-11-21): Muzaffarpur's zone (haj) split by district into haj, hjp Hajipur
# and sxp Samastipur (uji retired into sxp). One IN join. Manila waits for Anita's look (t3688-phl.json, unapplied);
# Hokkaido is not built here (notes/T-3688.md).
# Start (no window):
#   powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3688\kit\redraws.cmd
# Stop: create trainworld\data\work\world_run\STOP, then kill this script's PID only (first line of redraws.state).
# -From skips the steps before it: edits (backup pre and plan edits done), wave1, wave2, join, after.
param([int]$WaitPid = 0, [int]$Want = 6, [int]$Min = 4, [string]$From = "start", [int]$FloorGB = 5, [string]$Bk = "IN,NP,BD",
      [string]$SimDir = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\427323a8-53ce-483c-9c28-8f1452e36f9f\scratchpad\redraws\sim")
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Maps = "C:\Users\anita\projects\maps"
$TW = "$Maps\trainworld"
$Py = "$Maps\venv\Scripts\python.exe"
$Art = "$Maps\artifacts\trainworld"
$Pre = "redraws"
$Log = "$Art\$Pre-run.log"
$State = "$Kit\redraws.state"
$StopFile = "$TW\data\work\world_run\STOP"
$Port = 6377
$Name = "redraws"
$Wave1 = "haj"
$Wave2 = "hjp,sxp"
$All = "$Wave1,$Wave2"
Set-Location $Maps
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$script:Stage = "start"
$Order = @("start", "edits", "wave1", "wave2", "join", "after")
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
function Quit($code, $m) { Say "FAILED: $m"; Release; exit $code }
function FreeGB() { return [math]::Round((Get-PSDrive C).Free / 1GB, 1) }
function Floor($what) { $f = FreeGB; if ($f -lt $FloorGB) { Quit 8 "stopped before $what : $f GB free, under $FloorGB" }; Say "disk $f GB free before $what" }

Say "redraws.ps1 started, pid $PID, from $From"
if ($WaitPid) {
  # the first start's retire of uji, left running when that start was stopped for Manila
  Say "waiting for pid $WaitPid (retire ind)"
  Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
  $reg = Get-Content "$TW\app\src\game\registry.json" -Raw
  if ($reg -match '"id": "uji"' -or -not (Test-Path "$TW\data\work\pre-T3688\retired\uji.bin")) { Quit 4 "retire ind did not finish (uji still registered or not moved; redraws-retire-ind.log)" }
  Say "retire ind finished: uji unregistered and moved"
}
if (Test-Path $StopFile) { Quit 2 "$StopFile exists" }
$s = & $Py -I "$TW\pipeline\world_run.py" status 2>&1 | Out-String
$locks = ($s -split "\r?\n") | Where-Object { $_ -match "^lock " }
if ($locks) { Quit 3 "locks held: $($locks -join ' | ')" }
if ((Run "sim check" "-I $TW\tools\country\setup_sim.py $SimDir check" "$Art\$Pre-sim-check.log") -ne 0) { Quit 3 "sim copy does not check" }
$env:TW_SIM_DIR = "$SimDir\simroot\sim"
$env:TW_CARGO_TARGET_DIR = "$SimDir\simroot\sim\target"
$o = & $Py -I "$Maps\cores.py" claim $Want $Name --pid $PID "T-3688 rebuild" 2>&1
$g = [int](($o | Select-Object -Last 1).ToString().Trim())
if ($g -lt $Min) { Quit 3 "cores granted $g, want $Min or more" }
$J = [Math]::Min($g, $Want)
$env:NQ_JOBS = "$J"
Say "cores granted $g, jobs $J"

if (Reached "edits") {
  $script:Stage = "edits"
  Floor "backup pre"
  if ((Run "backup pre" "-I $TW\tools\country\backup.py T3688 pre $Bk" "$Art\$Pre-backup-pre.log") -ne 0) { Quit 6 "backup pre failed" }
  Say "PACK WRITES FROM HERE (retires, waves, join)"
  foreach ($sp in @("ind")) {
    if ((Run "edit $sp" "-I $Kit\plan_edit.py $Kit\t3688-$sp.json --go" "$Art\$Pre-edit-$sp.log") -ne 0) { Quit 4 "plan edit $sp failed; undo: restore.py T3688 pre --go" }
  }
  foreach ($sp in @("ind")) {
    if ((Run "retire $sp" "-I $Kit\plan_edit.py $Kit\t3688-$sp.json retire --go" "$Art\$Pre-retire-$sp.log") -ne 0) { Quit 4 "retire $sp failed; undo: restore.py T3688 pre --go" }
  }
}

# givers first: a build cuts every outline on disk out of its own, so a taker built beside a giver's old outline is a sliver
foreach ($w in @(@("wave1", $Wave1, "--keep-downloads"), @("wave2", $Wave2, ""))) {
  if (Reached $w[0]) {
    $script:Stage = $w[0]
    $rc = Run "run $($w[0])" "-I $TW\pipeline\world_run.py run --rows $($w[1]) --jobs $J --port $Port --no-join $($w[2])" "$Art\$Pre-$($w[0]).log"
    $gr = Run "gated $($w[0])" "-I $Kit\gated.py $($w[1])" "$Art\$Pre-gated-$($w[0]).log"
    if ($rc -ne 0) { Quit 4 "$($w[0]) run exit $rc ($Pre-$($w[0]).log); no join. Undo: restore.py T3688 pre --go" }
    if ($gr -ne 0) { Quit 4 "$($w[0]): a row is not gated ($Pre-gated-$($w[0]).log); no join. Undo: restore.py T3688 pre --go" }
  }
}

if (Reached "join") {
  $script:Stage = "join"
  Floor "backup prejoin"
  if ((Run "backup prejoin" "-I $TW\tools\country\backup.py T3688 prejoin $Bk" "$Art\$Pre-backup-prejoin.log") -ne 0) { Quit 6 "prejoin backup failed; no join" }
  Floor "join"
  $rc = Run "join" "-I $TW\pipeline\world_run.py join --countries IN --jobs $J --port $Port" "$Art\$Pre-join.log"
  Run "status" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status.log" | Out-Null
  if ($rc -ne 0) { Quit 5 "join exit $rc; undo: restore.py T3688 prejoin --go (then pre)" }
  if ((Run "stale" "-I $TW\tools\country\stale.py" "$Art\$Pre-stale.log") -ne 0) { Quit 7 "after the join: market packs name freed places ($Pre-stale.log)" }
  Run "successors dry" "-I $TW\pipeline\zone_successors.py --dry" "$Art\$Pre-successors-dry.log" | Out-Null
  if ((Run "successors" "-I $TW\pipeline\zone_successors.py" "$Art\$Pre-successors.log") -ne 0) { Quit 7 "zone_successors failed" }
  Say "joined"
}

if (Reached "after") {
  $script:Stage = "after"
  if ((Run "zone_terrain" "trainworld\pipeline\zone_terrain.py build --jobs $J" "$Art\$Pre-zone-terrain.log") -ne 0) { Quit 7 "zone_terrain failed" }
  Say "PACKS FREE: no more pack or terrain writes from this run"
  Run "status end" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status-end.log" | Out-Null
  if ((Run "start tiers" "$TW\pipeline\start_difficulty.py --fresh --jobs $J" "$Art\$Pre-start-difficulty.log") -ne 0) { Quit 7 "start tiers failed" }
  if ((Run "build_inputs" "trainworld\checks\build_inputs.py" "$Art\$Pre-build-inputs.log") -ne 0) { Quit 7 "build_inputs flags ($Pre-build-inputs.log); the packs are shipped" }
  if ((Run "split_zones" "trainworld\checks\split_zones.py --ids $All" "$Art\$Pre-split-zones.log") -ne 0) { Quit 7 "split_zones finds a split zone ($Pre-split-zones.log); the packs are shipped" }
}
$script:Stage = "end"
Release
Say "DONE: all steps passed"
