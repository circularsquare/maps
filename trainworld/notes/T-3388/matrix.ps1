param([string]$Exe, [string]$Tag, [string]$Lines = "lon:27,lon:2,lon:20,nyc:-,nyc:0", [string]$Steps = "count,count2,spur,station,lone", [switch]$Old, [switch]$Seq)
# Each step (and its undo) from a fresh load, on each city:line; or with -Seq every step in one
# session. Logs, per-line dumps and per-pace-round dumps into $S\$Tag. -Old: the rule before T-3388.
$S = 'C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\d516cc87-bbe1-4627-9f97-62b5c973620b\scratchpad\londonedit'
New-Item -ItemType Directory -Force "$S\$Tag" | Out-Null
$env:T3388 = '1'
if ($Old) { $env:T3388_OLD = '1' } else { Remove-Item Env:T3388_OLD -ErrorAction SilentlyContinue }
foreach ($cl in $Lines.Split(',')) {
  $city, $line = $cl.Split(':')
  $runs = New-Object System.Collections.ArrayList
  if ($Seq) { [void]$runs.Add(@('seq', 'all')) } else { foreach ($st in $Steps.Split(',')) { [void]$runs.Add(@($st, "$st,${st}_undo")) } }
  foreach ($r in $runs) {
    $pfx = "$S\$Tag\$city-$line-$($r[0])"
    $env:T3388_PH = $pfx + ".ph"
    $a = @("$S\$city.net", $pfx, $r[1])
    if ($line -ne '-') { $a += $line }
    $o = & $Exe @a 2>&1 | ForEach-Object { "$_" }
    $o | Set-Content "$pfx.log"
    $o | Select-String -Pattern " ms  r=" | ForEach-Object { "$city $line | $_" }
  }
}
Remove-Item Env:T3388_OLD -ErrorAction SilentlyContinue
