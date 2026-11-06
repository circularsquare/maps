# T-2642 scratch: launch.ps1 <net> <modes> <reps> <crowd> <log name> [<affinity hex>] [<q>]: run.cmd pinned, the core claimed
param([string]$net, [string]$modes, [string]$reps, [string]$crowd, [string]$log, [string]$aff = "4000", [string]$q = "0", [string]$test = "t2642_ab")
$sc = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21\scratchpad\searchcost"
$env:TW_T780_NET = $net
$env:TW_MODES = $modes
$env:TW_REPS = $reps
$env:TW_CROWD = $crowd
$env:TW_Q = $q
$env:TW_TEST = $test
$p = Start-Process -FilePath cmd.exe -ArgumentList "/c start /b /wait /affinity $aff $sc\run.cmd $sc\$log" -PassThru -WindowStyle Hidden
Set-Location C:\Users\anita\projects\maps
python cores.py claim 1 searchcost --pid $p.Id "T-2642 A/B $log, pinned $aff"
