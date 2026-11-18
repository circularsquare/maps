# Night queue after T-3399's tonight.ps1 (nightprep, 2026-11-17). Stages, highest value first, each backed up, logged
# and checked; the first failure stops the queue with the restore command in the log (a later stage would change
# the shared files a restore puts back, so nothing runs after a failure):
#   scan  T-3455 coastal gaps scan, read-only, beside T-3399 once T-3399 holds its cores (3 cores)
#   A     T-3488 Arnavutkoy into Istanbul's zone, Karamursel to Kocaeli, Altinova to Yalova; Turkey's join
#   B     T-3499 Bucaramanga's zone takes the rest of Soto province (no longer an enclave) and T-2725 Colombia's
#         28 rows moving over 10% on DANE totals; Colombia's join
#   C     T-3455 the flagged rows' solve units, most people first, at most -MaxScanGroups of them
# A stage starts only when its estimate ends before -Until (local time, default 10:00).
# Start: powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 <kit>\nightqueue.cmd
# Stop: trainworld\data\work\world_run\STOP (world_run and the scan stop at a clean point; this script stops at its
# next step), then this script's PID (first line of <kit>\nightqueue.state). -NoWait: do not wait for T-3399.
param([int]$Want = 10, [int]$Min = 6, [string]$Until = "10:00", [switch]$NoWait, [int]$ScanCores = 3,
      [int]$MaxScanGroups = 2, [switch]$NoScan, [switch]$Rehearse, [string]$ScanDir = "")
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Maps = "C:\Users\anita\projects\maps"
$TW = "$Maps\trainworld"
$Py = "$Maps\venv\Scripts\python.exe"
$Art = "$Maps\artifacts\trainworld"
$Log = if ($Rehearse) { "$Kit\rehearse.log" } else { "$Art\night-queue.log" }
$State = if ($Rehearse) { "$Kit\rehearse.state" } else { "$Kit\nightqueue.state" }
$StopFile = "$TW\data\work\world_run\STOP"
$K3399 = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\76388cba-0327-41aa-a4e4-4e7d7376bf41\scratchpad\countries"
$S3399 = "$K3399\tonight.state"
$L3399 = "$Art\t3399-tonight.log"
if (-not $ScanDir) { $ScanDir = "$Kit\scan" }
$Port = 6372
Set-Location $Maps
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$Started = Get-Date
$UntilT = [datetime]::Today.Add([timespan]::Parse($Until))
if ($UntilT -lt $Started) { $UntilT = $UntilT.AddDays(1) }
$script:Scan = $null
$script:Stage = "start"

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
function Claim($n, $name, $what) {
  if ($Rehearse) { return $n }
  $o = & $Py -I "$Maps\cores.py" claim $n $name --pid $PID $what 2>&1
  return [int](($o | Select-Object -Last 1).ToString().Trim())
}
function Release($name) { if ($Rehearse) { return }; & $Py -I "$Maps\cores.py" release $name 2>&1 | Out-Null }
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
  for ($i = 0; $i -lt 48; $i++) {
    $s = & $Py -I "$TW\pipeline\world_run.py" status 2>&1 | Out-String
    $locks = ($s -split "\r?\n") | Where-Object { $_ -match "^lock " }
    if (-not $locks) { return $true }
    Say "waiting for locks: $($locks -join ' | ')"
    Scan-Tick
    Start-Sleep -Seconds 300
  }
  return $false
}
function Jobs() {
  for ($i = 0; $i -lt 6; $i++) {
    $g = Claim $Want "nightqueue" "night queue: T-3488, T-3499/T-2725, T-3455 rebuilds and joins"
    if ($g -ge $Min) { Say "cores granted $g"; return $g }
    Say "cores granted $g (want $Min or more), retrying in 10 min"
    Scan-Tick
    Start-Sleep -Seconds 600
  }
  $g = Claim $Want "nightqueue" "night queue rebuilds and joins"
  Say "cores granted $g after an hour of retries; going on"
  return [Math]::Max($g, 2)
}
function ScanRunning() { return ($script:Scan -is [System.Diagnostics.Process]) -and -not $script:Scan.HasExited }
function Quit($code, $m) {
  Say $m
  Release "nightqueue"
  if (ScanRunning) {
    Say "waiting for the scan (pid $($script:Scan.Id), read-only) to end, so its cores stay on the ledger"
    while (ScanRunning) { Start-Sleep -Seconds 120 }
    Scan-Tick
  }
  exit $code
}

# ---- the scan, beside T-3399 once T-3399 holds its cores
function Scan-Tick() {
  if ($NoScan -or ($Rehearse -and -not (Test-Path "$ScanDir\flagged.json"))) { return }
  if ($script:Scan -is [System.Diagnostics.Process]) {
    if ($script:Scan.HasExited -and -not $script:ScanEnded) {
      $script:ScanEnded = $true
      Say "scan ended: exit $($script:Scan.ExitCode) (log $Art\t3455-scan.log)"
      Release "nightscan"
      if (Test-Path "$ScanDir\flagged.json") { Copy-Item "$ScanDir\flagged.json" "$Art\t3455-flagged.json" -Force }
    }
    return
  }
  if ($script:Scan -eq "done") { return }
  if (Test-Path "$ScanDir\flagged.json") { $script:ScanEnded = $true; $script:Scan = "done"; return }
  # T-3399 first: the scan starts once T-3399 holds its cores (its log says so) or has ended
  if ((T3399-Alive) -and -not (Select-String -Path $L3399 -Pattern 'cores granted \d+$|after an hour of retries' -Quiet -ErrorAction SilentlyContinue)) { return }
  $g = Claim $ScanCores "nightscan" "T-3455 coastal gaps scan (read-only)"
  if ($g -lt 2) { Release "nightscan"; Say "scan waits: $g cores free"; return }
  $g = [Math]::Min($g, $ScanCores)
  $a = "-I $Kit\scan_coast.py run $ScanDir --workers $g --limit-s 900"
  Say "scan starts on $g cores: $a"
  $script:Scan = Start-Process -FilePath $Py -ArgumentList $a -WorkingDirectory $Maps -RedirectStandardOutput "$Art\t3455-scan.log" -RedirectStandardError "$Art\t3455-scan.log.err" -NoNewWindow -PassThru
}
function T3399-Alive() {
  if (-not (Test-Path $S3399)) { return $false }
  $first = (Get-Content $S3399 -TotalCount 1)
  if ($first -match "pid (\d+)") { return [bool](Get-Process -Id ([int]$Matches[1]) -ErrorAction SilentlyContinue) }
  return $false
}

# ---- one stage's runs: waves of rows built in turn, then the join
function Waves($task, $waves, $port, $back) {
  for ($k = 0; $k -lt $waves.Count; $k++) {
    if (Stopped) { return 1 }
    $rows = ($waves[$k] -join ",")
    $keep = if ($k -lt $waves.Count - 1) { " --keep-downloads" } else { "" }
    $rc = Run "run $task wave $($k + 1)" "-I $TW\pipeline\world_run.py run --rows $rows --jobs $J --port $port --no-join$keep" "$Art\nq-$task-run$($k + 1).log"
    $g = Run "gated $task wave $($k + 1)" "-I $Kit\gated.py $rows" "$Art\nq-$task-gated$($k + 1).log"
    if ($rc -ne 0 -or $g -ne 0) {
      Say "rows: $((Get-Content "$Art\nq-$task-gated$($k + 1).log") -join '; ')"
      Quit 4 "STOPPED: $task wave $($k + 1) run exit $rc, rows not all gated; no join. Undo the stage with tools\country\restore.py $back pre --go, or relax the parked rows (adding-a-country.md 6), run --rows again and join"
    }
  }
  return 0
}
function Join($task, $countries, $back, $bc) {
  if (-not (WaitFree)) { Quit 3 "STOPPED: locks held before the $task join" }
  if (Stopped) { Quit 2 "STOPPED before the $task join (rows gated, waiting)" }
  Run "backup prejoin $task" "-I $TW\tools\country\backup.py $back prejoin $bc" "$Art\nq-$task-backup-prejoin.log" | Out-Null
  $rc = Run "join $task" "-I $TW\pipeline\world_run.py join --countries $countries --jobs $J --port $Port" "$Art\nq-$task-join.log"
  Run "status $task" "-I $TW\pipeline\world_run.py status" "$Art\nq-$task-status.log" | Out-Null
  if ($rc -ne 0) { Quit 5 "STOPPED: $task join exit $rc; undo the join with tools\country\restore.py $back prejoin --go (the whole stage: $back pre --go)" }
  if ((Run "stale $task" "-I $TW\tools\country\stale.py" "$Art\nq-$task-stale.log") -ne 0) {
    Quit 7 "STOPPED after the $task join: market packs name freed places (nq-$task-stale.log); tools\country\stale.py --cmd writes the fix, run it before any other run"
  }
}

Say "nightqueue.ps1 started, pid $PID; until $(Get-Date $UntilT -Format 'yyyy-MM-dd HH:mm')"
if (Test-Path "$ScanDir\flagged.json") { Say "scan already done ($ScanDir\flagged.json)" }

# ---- wait for T-3399's tonight.ps1
if (-not $NoWait) {
  $t = 0
  while (-not ((Test-Path $S3399) -and ((Get-Item $S3399).LastWriteTime -gt $Started.AddHours(-14)))) {
    if ($t -ge 6) { Quit 3 "STOPPED: T-3399's tonight.state did not appear in 30 min; start with -NoWait to run without it" }
    Say "waiting for T-3399 to start ($S3399)"
    Start-Sleep -Seconds 300; $t++
  }
  while (T3399-Alive) { Scan-Tick; Start-Sleep -Seconds 300 }
  $last = (Get-Content $L3399 -Tail 1)
  Say "T-3399 ended: $last"
  if ($last -notmatch "all done|stage 2 skipped|STOPPED: (t617 )?run exit|STOPPED: locks|STOPPED: sim copy|STOPPED before stage 2") {
    Quit 9 "STOPPED: T-3399 did not end cleanly (a failed join wants restore.py T3399 prejoin, which would undo anything run after it); nothing run"
  }
}
Scan-Tick
if (Stopped) { Quit 2 "STOPPED before stage A" }
if (-not (WaitFree)) { Quit 3 "STOPPED: locks still held after 4 hours" }

# the sim copy: T-3399's if it checks, else this kit's own
$sim = $K3399
if ((Run "sim check" "-I $TW\tools\country\setup_sim.py $K3399 check" "$Art\nq-sim-check.log") -ne 0) {
  $sim = $Kit
  if (Test-Path "$Kit\simroot\sim") { Run "sim refresh (own)" "-I $TW\tools\country\setup_sim.py $Kit refresh" "$Art\nq-sim.log" | Out-Null }
  else { Run "sim make (own)" "-I $TW\tools\country\setup_sim.py $Kit make" "$Art\nq-sim.log" | Out-Null }
  if ((Run "sim check (own)" "-I $TW\tools\country\setup_sim.py $Kit check" "$Art\nq-sim-check.log") -ne 0) { Quit 3 "STOPPED: no sim copy checks" }
} else {
  Run "sim refresh" "-I $TW\tools\country\setup_sim.py $K3399 refresh" "$Art\nq-sim.log" | Out-Null
}
$env:TW_SIM_DIR = "$sim\simroot\sim"
$env:TW_CARGO_TARGET_DIR = "$sim\simroot\sim\target"
$J = Jobs
$env:NQ_JOBS = "$J"

# ---- A: T-3488 Istanbul
$script:Stage = "A T-3488"
if (Fits 130) {
  if (-not (WaitFree)) { Quit 3 "STOPPED: locks held" }
  $bc = "TR,GR,BG,GE,AM,AZ,IR,IQ,SY"
  if ((Run "backup pre" "-I $TW\tools\country\backup.py T3488 pre $bc" "$Art\nq-T3488-backup-pre.log") -ne 0) { Quit 6 "STOPPED: backup failed (see the log); nothing changed" }
  if ((Run "plan edit" "-I $Kit\plan_edit.py $Kit\t3488.json --go" "$Art\nq-T3488-plan.log") -ne 0) { Quit 6 "STOPPED: plan edit refused (see the log); nothing written, or undo with tools\country\restore.py T3488 pre --go" }
  if ((Run "retire arv" "-I $Kit\plan_edit.py $Kit\t3488.json retire --go" "$Art\nq-T3488-retire.log") -ne 0) { Quit 6 "STOPPED: retiring arv failed; undo with tools\country\restore.py T3488 pre --go" }
  $w = (Get-Content "$Kit\t3488.waves.json" -Raw | ConvertFrom-Json).waves
  Waves "T3488" $w $Port "T3488" | Out-Null
  Join "T3488" "TR" "T3488" $bc
  Run "zone successors dry" "-I $TW\pipeline\zone_successors.py --dry" "$Art\nq-T3488-successors-dry.log" | Out-Null
  Run "zone successors" "-I $TW\pipeline\zone_successors.py" "$Art\nq-T3488-successors.log" | Out-Null
  Say "stage A done"
}

# ---- B: T-3499 Bucaramanga and T-2725 Colombia on census totals
$script:Stage = "B T-3499"
if (-not (Stopped) -and (Fits 150)) {
  if (-not (WaitFree)) { Quit 3 "STOPPED: locks held" }
  $bc = "CO,VE,EC,PA,PE,BR"
  if ((Run "backup pre" "-I $TW\tools\country\backup.py T3499 pre $bc" "$Art\nq-T3499-backup-pre.log") -ne 0) { Quit 6 "STOPPED: backup failed (see the log); nothing changed" }
  if ((Run "plan edit" "-I $Kit\plan_edit.py $Kit\t3499.json --go" "$Art\nq-T3499-plan.log") -ne 0) { Quit 6 "STOPPED: plan edit refused (see the log); undo with tools\country\restore.py T3499 pre --go" }
  $w = (Get-Content "$Kit\t3499.waves.json" -Raw | ConvertFrom-Json).waves
  Waves "T3499" $w $Port "T3499" | Out-Null
  Join "T3499" "CO" "T3499" $bc
  Say "stage B done"
}

# ---- C: T-3455 the flagged rows
$script:Stage = "C T-3455"
if (-not $NoScan -and -not (Stopped)) {
  while (ScanRunning) {
    if (-not (Fits 60)) { Quit 0 "scan still running; no time left for its rebuilds (flagged.json will be in $ScanDir)" }
    Start-Sleep -Seconds 300
  }
  Scan-Tick
  if (Test-Path "$ScanDir\flagged.json") {
    $f = Get-Content "$ScanDir\flagged.json" -Raw -Encoding utf8 | ConvertFrom-Json
    $n = 0
    foreach ($g in $f.groups) {
      if ($n -ge $MaxScanGroups) { break }
      if (Stopped) { break }
      if (-not (Fits ($g.est_join_min + 40))) { continue }
      $fresh = @($g.rows | Where-Object { (Test-Path "$TW\data\packs\$_.boundary.geojson") -and (Get-Item "$TW\data\packs\$_.boundary.geojson").LastWriteTime -gt $Started })
      $todo = @($g.rows | Where-Object { $fresh -notcontains $_ })
      if ($fresh.Count) { Say "group $($g.key): $($fresh -join ',') rebuilt tonight already" }
      if (-not $todo.Count) { continue }
      $n++
      $rows = ($todo -join ",")
      $back = "T3455g$n"
      $bc = ($g.backup -join ",")
      Say "group $n $($g.key): $rows ($($g.people) people, $($g.built_m2) m2 built in the gaps); backup $bc"
      if (-not (WaitFree)) { Quit 3 "STOPPED: locks held" }
      if ((Run "backup pre g$n" "-I $TW\tools\country\backup.py $back pre $bc" "$Art\nq-$back-backup-pre.log") -ne 0) { Quit 6 "STOPPED: backup failed; nothing changed" }
      if ((Run "planned g$n" "-I $Kit\setplanned.py $rows T-3455 --go" "$Art\nq-$back-planned.log") -ne 0) { Quit 6 "STOPPED: setplanned failed; undo with tools\country\restore.py $back pre --go" }
      Waves $back @(, $todo) $Port $back | Out-Null
      Join $back (($g.countries) -join ",") $back $bc
    }
    if ($n -eq 0) { Say "no flagged group fits or none flagged" }
  } else { Say "no scan result; stage C skipped" }
}

$script:Stage = "end"
Run "build_inputs" "$TW\checks\build_inputs.py" "$Art\nq-build-inputs.log" | Out-Null
Run "status end" "-I $TW\pipeline\world_run.py status" "$Art\nq-status-end.log" | Out-Null
Release "nightqueue"
while (ScanRunning) { Start-Sleep -Seconds 300 }
Scan-Tick
Say "all done"
