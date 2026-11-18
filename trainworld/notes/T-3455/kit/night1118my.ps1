# T-3455 batch 2, Malaysia again (nightfix3): night1118fix's join parked kog, pnt, joo and kut because an older member
# of buk-my, kur (Kuala Terengganu), failed water_points_in_water 0.92 against OSM water added after its 11-02 build.
# kur's check is relaxed (setbuilt.py, gate_relax only; kur stays shipped) and the four rows set built before this
# starts. Then: run --rows (gate), gated.py, a prejoin backup under its own tag (T3455my), join --countries MY,
# stale.py, then the after-steps again for the four (zone_terrain, market_drives, build_inputs) and start tiers.
# Start (no window):
#   powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1118my.cmd
# Stop: create trainworld\data\work\world_run\STOP, then kill this script's PID only (first line of night1118my.state).
param([int]$Want = 9, [int]$Min = 6, [string]$Until = "11:00",
      [string]$SimDir = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\76388cba-0327-41aa-a4e4-4e7d7376bf41\scratchpad\countries")
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Maps = "C:\Users\anita\projects\maps"
$TW = "$Maps\trainworld"
$Py = "$Maps\venv\Scripts\python.exe"
$Art = "$Maps\artifacts\trainworld"
$Pre = "night1118my"
$Log = "$Art\$Pre.log"
$State = "$Kit\night1118my.state"
$StopFile = "$TW\data\work\world_run\STOP"
$Port = 6372
$Name = "nightfix3"
Set-Location $Maps
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$UntilT = [datetime]::Today.Add([timespan]::Parse($Until))
$script:Stage = "start"

function Say($m) {
  $l = "$(Get-Date -Format s) [$script:Stage] $m"
  Add-Content -Path $Log -Value $l -Encoding utf8
  Set-Content -Path $State -Value @("pid $PID", $l) -Encoding utf8
}
function Run($name, $argline, $out) {
  Say "start $name : $argline > $out"
  $p = Start-Process -FilePath $Py -ArgumentList $argline -WorkingDirectory $Maps -RedirectStandardOutput $out -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
  Say "end $name : exit $($p.ExitCode)"
  return $p.ExitCode
}
function Release() { & $Py -I "$Maps\cores.py" release $Name 2>&1 | Out-Null }
function Quit($code, $m) { Say $m; Release; exit $code }
function Fits($min) { return (Get-Date).AddMinutes($min) -le $UntilT }

Say "night1118my.ps1 started, pid $PID; until $(Get-Date $UntilT -Format 'yyyy-MM-dd HH:mm')"
if (Test-Path $StopFile) { Quit 2 "STOPPED: $StopFile exists" }
$s = & $Py -I "$TW\pipeline\world_run.py" status 2>&1 | Out-String
$locks = ($s -split "\r?\n") | Where-Object { $_ -match "^lock " }
if ($locks) { Quit 3 "STOPPED: locks held: $($locks -join ' | ')" }
if ((Run "sim check" "-I $TW\tools\country\setup_sim.py $SimDir check" "$Art\$Pre-sim-check.log") -ne 0) { Quit 3 "STOPPED: sim copy does not check" }
$env:TW_SIM_DIR = "$SimDir\simroot\sim"
$env:TW_CARGO_TARGET_DIR = "$SimDir\simroot\sim\target"
$o = & $Py -I "$Maps\cores.py" claim $Want $Name --pid $PID "T-3455 Malaysia again" 2>&1
$g = [int](($o | Select-Object -Last 1).ToString().Trim())
if ($g -lt $Min) { Quit 3 "STOPPED: cores granted $g, want $Min or more" }
$J = [Math]::Min($g, $Want)
Say "cores granted $g, jobs $J"

$script:Stage = "MY"
$rows = "kog,pnt,joo,kut"
$rc = Run "run MY rows" "-I $TW\pipeline\world_run.py run --rows $rows --jobs $J --port $Port --no-join" "$Art\$Pre-run.log"
$gr = Run "gated MY rows" "-I $Kit\gated.py $rows" "$Art\$Pre-gated.log"
if ($rc -ne 0 -or $gr -ne 0) { Quit 4 "STOPPED: run exit $rc, rows not all gated; no join ($Pre-gated.log)" }
if (-not (Fits 75)) { Quit 2 "STOPPED: rows gated, the join would end after $Until; not joined" }
if ((Run "backup prejoin" "-I $TW\tools\country\backup.py T3455my prejoin MY,SG,BN,ID,TH" "$Art\$Pre-backup-prejoin.log") -ne 0) { Quit 6 "STOPPED: prejoin backup failed; no join" }
$rc = Run "join MY" "-I $TW\pipeline\world_run.py join --countries MY --jobs $J --port $Port" "$Art\$Pre-join.log"
Run "status" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status.log" | Out-Null
if ($rc -ne 0) { Quit 5 "STOPPED: MY join exit $rc; undo with tools\country\restore.py T3455my prejoin --go" }
if ((Run "stale" "-I $TW\tools\country\stale.py" "$Art\$Pre-stale.log") -ne 0) { Quit 7 "STOPPED after the MY join: market packs name freed places ($Pre-stale.log)" }
Say "MY joined"

$script:Stage = "after"
Run "zone_terrain" "trainworld\pipeline\zone_terrain.py build --jobs $J" "$Art\$Pre-zone-terrain.log" | Out-Null
Run "market_drives" "trainworld\pipeline\market_drives.py build --jobs $J" "$Art\$Pre-market-drives.log" | Out-Null
Run "build_inputs" "trainworld\checks\build_inputs.py" "$Art\$Pre-build-inputs.log" | Out-Null
Run "start tiers" "$TW\pipeline\start_difficulty.py --fresh --jobs $J" "$Art\$Pre-start-difficulty.log" | Out-Null
Run "status end" "-I $TW\pipeline\world_run.py status" "$Art\$Pre-status-end.log" | Out-Null
$script:Stage = "end"
Release
Say "all done"
