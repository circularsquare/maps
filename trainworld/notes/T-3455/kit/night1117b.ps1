# Night of 2026-11-17, resumed by nightfix after night1117.ps1 stopped at batch 1 wave 1's gate (22:35).
# The same queue from that point: batch 1 (T-3511) reruns its three parked wave 1 rows (sve and roc with origins moved
# into their own units, jau with pop_vs_fua relaxed), then wave 2, the join and zone successors; then batch 2 (T-3455)
# whole; batch 3 only without -NoCif; the after-steps. Steps that already ran (backup pre, plan edits, retires,
# batch 1's setplanned, wave 1) are left out: never rerun them (backup pre would overwrite the undo point).
# Start (no window):
#   powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117b.cmd
# Rehearse: powershell -File <this file> -Rehearse
# Stop: create trainworld\data\work\world_run\STOP, then kill this script's PID only (first line of night1117b.state).
param([int]$Want = 9, [int]$Min = 6, [string]$Until = "10:00", [switch]$Rehearse, [switch]$NoCif,
      [string]$SimDir = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\76388cba-0327-41aa-a4e4-4e7d7376bf41\scratchpad\countries")
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Maps = "C:\Users\anita\projects\maps"
$TW = "$Maps\trainworld"
$Py = "$Maps\venv\Scripts\python.exe"
$Art = "$Maps\artifacts\trainworld"
$Pre = if ($Rehearse) { "night1117b-rehearse" } else { "night1117b" }
$Log = "$Art\$Pre.log"
$State = if ($Rehearse) { "$Art\night1117b-rehearse.state" } else { "$Kit\night1117b.state" }
$StopFile = "$TW\data\work\world_run\STOP"
$Port = 6372
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
    Start-Sleep -Seconds 300
  }
  return $false
}
function Jobs() {
  for ($i = 0; $i -lt 6; $i++) {
    $g = Claim $Want "nightqueue" "night 2026-11-17: T-3511 and T-3455 rebuilds and joins (nightfix)"
    if ($g -ge $Min) { Say "cores granted $g"; return [Math]::Min($g, $Want) }
    Say "cores granted $g (want $Min or more), retrying in 10 min"
    Start-Sleep -Seconds 600
  }
  $g = Claim $Want "nightqueue" "night 2026-11-17: T-3511 and T-3455 rebuilds and joins (nightfix)"
  Say "cores granted $g after an hour of retries; going on"
  return [Math]::Max([Math]::Min($g, $Want), 2)
}
function Quit($code, $m) {
  Say $m
  Release "nightqueue"
  exit $code
}

# ---- one batch's runs: waves of rows built in turn (each wave gated before the next), then the join
function Waves($task, $waves, $back) {
  for ($k = 0; $k -lt $waves.Count; $k++) {
    if (Stopped) { Quit 2 "STOPPED before $task wave $($k + 1); nothing joined. Undo with tools\country\restore.py $back pre --go" }
    $rows = ($waves[$k] -join ",")
    $keep = if ($k -lt $waves.Count - 1) { " --keep-downloads" } else { "" }
    $rc = Run "run $task wave $($k + 1)" "-I $TW\pipeline\world_run.py run --rows $rows --jobs $J --port $Port --no-join$keep" "$Art\$Pre-$task-run$($k + 1).log"
    $g = Run "gated $task wave $($k + 1)" "-I $Kit\gated.py $rows" "$Art\$Pre-$task-gated$($k + 1).log"
    if ($rc -ne 0 -or $g -ne 0) {
      if (-not $Rehearse) { Say "rows: $((Get-Content "$Art\$Pre-$task-gated$($k + 1).log") -join '; ')" }
      Quit 4 "STOPPED: $task wave $($k + 1) run exit $rc, rows not all gated; no join. Undo the batch with tools\country\restore.py $back pre --go, or relax the parked rows (adding-a-country.md 6), run --rows again and join"
    }
  }
}
function Join($task, $countries, $back, $bc, $post) {
  if (-not (WaitFree)) { Quit 3 "STOPPED: locks held before the $task join" }
  if (Stopped) { Quit 2 "STOPPED before the $task join (rows gated, waiting)" }
  if ((Run "backup prejoin $task" "-I $TW\tools\country\backup.py $back prejoin $bc" "$Art\$Pre-$task-backup-prejoin.log") -ne 0) { Quit 6 "STOPPED: prejoin backup failed; no join. Undo the batch with tools\country\restore.py $back pre --go" }
  $rc = Run "join $task" "-I $TW\pipeline\world_run.py join --countries $countries --jobs $J --port $Port" "$Art\$Pre-$task-join.log"
  Run "status $task" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-$task-status.log" | Out-Null
  if ($rc -ne 0) { Quit 5 "STOPPED: $task join exit $rc; undo the join with tools\country\restore.py $back prejoin --go (the whole batch: $back pre --go)" }
  if ($post) { & $post }
  if ((Run "stale $task" "-I $TW\tools\country\stale.py" "$Art\$Pre-$task-stale.log") -ne 0) {
    Quit 7 "STOPPED after the $task join: market packs name freed places ($Pre-$task-stale.log); tools\country\stale.py --cmd writes the fix, run it before any other run"
  }
}
function Successors() {
  Run "zone successors dry" "-I $TW\pipeline\zone_successors.py --dry" "$Art\$Pre-successors-dry.log" | Out-Null
  Run "zone successors" "-I $TW\pipeline\zone_successors.py" "$Art\$Pre-successors.log" | Out-Null
}
function Step($name, $argline, $out, $back, $what) {
  if ((Run $name $argline $out) -ne 0) { Quit 6 "STOPPED: $name failed ($out). $what" }
}

Say "night1117b.ps1 started, pid $PID; until $(Get-Date $UntilT -Format 'yyyy-MM-dd HH:mm')$(if ($Rehearse) { ' (REHEARSAL)' })"
if (Stopped) { Quit 2 "STOPPED at start" }
if (-not (WaitFree)) { Quit 3 "STOPPED: locks still held after 4 hours" }

# ---- the sim copy: the one given, else this night's own
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
$J = Jobs
$ran = $true

# ---- batch 1: T-3511, from wave 1's rerun of its parked rows (backup pre, plan edits, retires, setplanned and
# wave 1 ran in night1117.ps1)
$script:Stage = "batch 1 T-3511"
$bc = "AR,AU,BG,BR,CL,CN,GR,IN,RU,TR,UY"
$w = @(@("sve", "roc", "jau"), @("dir", "slr", "per", "lal"))
Waves "T3511" $w "T3511"
Join "T3511" "TR,AU,AR,IN,CN" "T3511" $bc { Successors }
Say "batch 1 done"

# ---- batch 2: T-3455 coastal rows (nic and sjc stay out: Monaco and Sinuiju, T-3529)
$script:Stage = "batch 2 T-3455"
if (-not (Stopped) -and (Fits 270)) {
  $bc = "AE,BH,CN,EG,ID,IL,IN,JP,KP,KR,KW,MY,PS,QA,SA,SG,TH"
  $rows = "mop,bus,iks,too,asn,cuc,sne,msr,ale,shl,azh,isy,hud,dut,chn,ngt,mdt,thi,klt,srk,maa,joo,pnt,kut,kog,kuw,brm,doh"
  if (-not (WaitFree)) { Quit 3 "STOPPED: locks held" }
  Step "backup pre" "-I $TW\tools\country\backup.py T3455 pre $bc" "$Art\$Pre-T3455-backup-pre.log" "T3455" "Nothing changed."
  Step "setplanned" "-I $Kit\setplanned.py $rows T-3455 --go" "$Art\$Pre-T3455-planned.log" "T3455" "Undo with tools\country\restore.py T3455 pre --go"
  Waves "T3455" @(, ($rows -split ",")) "T3455"
  Join "T3455" "KR,EG,IN,ID,QA,KW,MY" "T3455" $bc $null
  Say "batch 2 done"
}

# ---- batch 3 (optional): Philippines, cif-ph
$script:Stage = "batch 3 cif-ph"
if (-not $NoCif -and -not (Stopped) -and (Fits 90)) {
  $rows = "ceb,aro,ilg,snr,tgb,cyo"
  if (-not (WaitFree)) { Quit 3 "STOPPED: locks held" }
  Step "backup pre" "-I $TW\tools\country\backup.py T3455ph pre PH" "$Art\$Pre-T3455ph-backup-pre.log" "T3455ph" "Nothing changed."
  Step "setplanned" "-I $Kit\setplanned.py $rows T-3455 --go" "$Art\$Pre-T3455ph-planned.log" "T3455ph" "Undo with tools\country\restore.py T3455ph pre --go"
  Waves "T3455ph" @(, ($rows -split ",")) "T3455ph"
  Join "T3455ph" "PH" "T3455ph" "PH" $null
  Say "batch 3 done"
}

# ---- after both, as T-3399's tonight.ps1
$script:Stage = "after"
if ($ran) {
  $blocks = "$Art\night1117-blocks.json"
  Run "terrain blocks" "trainworld\pipeline\terrain_fine.py blocks $blocks" "$Art\$Pre-terrain-blocks.log" | Out-Null
  Run "terrain fetch" "trainworld\pipeline\terrain_fine.py run $blocks trainworld\data\work\terrain-fine --jobs 1 --fetch 8" "$Art\$Pre-terrain-fine.log" | Out-Null
  Run "zone_terrain" "trainworld\pipeline\zone_terrain.py build --jobs $J" "$Art\$Pre-zone-terrain.log" | Out-Null
  Run "market_drives" "trainworld\pipeline\market_drives.py build --jobs $J" "$Art\$Pre-market-drives.log" | Out-Null
  Run "build_inputs" "trainworld\checks\build_inputs.py" "$Art\$Pre-build-inputs.log" | Out-Null
}
Run "status end" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status-end.log" | Out-Null
$script:Stage = "end"
Release "nightqueue"
Say "all done"
