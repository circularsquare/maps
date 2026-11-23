# T-3688 Manila redraw (agent manilaspec, 11-22, after Anita's "keep malolos with manila"): mai Manila gives its fringes
# to Balanga (Mariveles retired into it), San Miguel, Tanay, Lemery, Nasugbu and Angeles, and to two new zones, San Pablo
# City and Lipa City (t3688-phl.json). One PH join. A copy of redraws.ps1 (Muzaffarpur), notes/T-3688.md "Manila".
# Start (no window), only once redraws-run.log ends "DONE:" (the script refuses otherwise):
#   powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3688\kit\redraws-ph.cmd
# Stop: create trainworld\data\work\world_run\STOP, then kill this script's PID only (first line of redraws-ph.state).
# -From skips the steps before it: edits (backup pre, plan edit and retire of mri), wave1, wave2, join, after.
param([int]$Want = 6, [int]$Min = 4, [string]$From = "start", [int]$FloorGB = 5, [string]$Bk = "PH", [switch]$Anyway,
      [string]$SimDir = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\427323a8-53ce-483c-9c28-8f1452e36f9f\scratchpad\redraws\sim")
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Maps = "C:\Users\anita\projects\maps"
$TW = "$Maps\trainworld"
$Py = "$Maps\venv\Scripts\python.exe"
$Art = "$Maps\artifacts\trainworld"
$Pre = "redraws-ph"
$Log = "$Art\$Pre-run.log"
$State = "$Kit\redraws-ph.state"
$StopFile = "$TW\data\work\world_run\STOP"
$Port = 6377
$Name = "redraws-ph"
$Tag = "T3688ph"
$Wave1 = "mai"
$Wave2 = "ang,ctg,ler,lpa,nab,smg,spb,tna"
$All = "$Wave1,$Wave2"
$Before = "$Art\redraws-run.log"  # the Muzaffarpur job, which must have finished
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

Say "redraws-ph.ps1 started, pid $PID, from $From"
# the India join and its after-steps write packs between the locks the status check sees; -Anyway once that job is
# known to be over some other way (failed and restored)
$last = (Get-Content $Before -Tail 1 -ErrorAction SilentlyContinue | Out-String).Trim()
if (-not $Anyway -and $last -notmatch "DONE:") { Quit 2 "the Muzaffarpur job has not ended DONE (last line of redraws-run.log: $last)" }
if (Test-Path $StopFile) { Quit 2 "$StopFile exists" }
$s = & $Py -I "$TW\pipeline\world_run.py" status 2>&1 | Out-String
$locks = ($s -split "\r?\n") | Where-Object { $_ -match "^lock " }
if ($locks) { Quit 3 "locks held: $($locks -join ' | ')" }
if ((Run "sim check" "-I $TW\tools\country\setup_sim.py $SimDir check" "$Art\$Pre-sim-check.log") -ne 0) { Quit 3 "sim copy does not check (make one: setup_sim.py <folder> make, then -SimDir <folder>)" }
$env:TW_SIM_DIR = "$SimDir\simroot\sim"
$env:TW_CARGO_TARGET_DIR = "$SimDir\simroot\sim\target"
$o = & $Py -I "$Maps\cores.py" claim $Want $Name --pid $PID "T-3688 Manila rebuild" 2>&1
$g = [int](($o | Select-Object -Last 1).ToString().Trim())
if ($g -lt $Min) { Quit 3 "cores granted $g, want $Min or more" }
$J = [Math]::Min($g, $Want)
$env:NQ_JOBS = "$J"
Say "cores granted $g, jobs $J"

if (Reached "edits") {
  $script:Stage = "edits"
  Floor "backup pre"
  if ((Run "backup pre" "-I $TW\tools\country\backup.py $Tag pre $Bk" "$Art\$Pre-backup-pre.log") -ne 0) { Quit 6 "backup pre failed" }
  Say "PACK WRITES FROM HERE (retire, waves, join)"
  foreach ($sp in @("phl")) {
    if ((Run "edit $sp" "-I $Kit\plan_edit.py $Kit\t3688-$sp.json --go" "$Art\$Pre-edit-$sp.log") -ne 0) { Quit 4 "plan edit $sp failed; undo: revert_spec.py t3688-phl.json data\work\pre-$Tag\pre --go" }
  }
  foreach ($sp in @("phl")) {
    if ((Run "retire $sp" "-I $Kit\plan_edit.py $Kit\t3688-$sp.json retire --go" "$Art\$Pre-retire-$sp.log") -ne 0) { Quit 4 "retire $sp failed; undo: restore.py $Tag pre --go" }
  }
}

# givers first: a build cuts every outline on disk out of its own, so a taker built beside a giver's old outline is a sliver.
# Both waves keep the extract: Geofabrik's PH download 404s (11-22) and a rebuild after a park would need it again.
foreach ($w in @(@("wave1", $Wave1, "--keep-downloads"), @("wave2", $Wave2, "--keep-downloads"))) {
  if (Reached $w[0]) {
    $script:Stage = $w[0]
    $rc = Run "run $($w[0])" "-I $TW\pipeline\world_run.py run --rows $($w[1]) --jobs $J --port $Port --no-join $($w[2])" "$Art\$Pre-$($w[0]).log"
    $gr = Run "gated $($w[0])" "-I $Kit\gated.py $($w[1])" "$Art\$Pre-gated-$($w[0]).log"
    if ($rc -ne 0) { Quit 4 "$($w[0]) run exit $rc ($Pre-$($w[0]).log); no join. Undo: restore.py $Tag pre --go" }
    if ($gr -ne 0) { Quit 4 "$($w[0]): a row is not gated ($Pre-gated-$($w[0]).log); no join. Undo: restore.py $Tag pre --go" }
  }
}

if (Reached "join") {
  $script:Stage = "join"
  Floor "backup prejoin"
  if ((Run "backup prejoin" "-I $TW\tools\country\backup.py $Tag prejoin $Bk" "$Art\$Pre-backup-prejoin.log") -ne 0) { Quit 6 "prejoin backup failed; no join" }
  Floor "join"
  $rc = Run "join" "-I $TW\pipeline\world_run.py join --countries PH --jobs $J --port $Port" "$Art\$Pre-join.log"
  Run "status" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status.log" | Out-Null
  if ($rc -ne 0) { Quit 5 "join exit $rc; undo: restore.py $Tag prejoin --go (then pre)" }
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
