# crowdfast: one hang.mjs run (claims 3 cores itself, 2 demand workers). -Name -Base -Qs -OpsFile -Repeat -WaitMax
param([string]$Want = "3", [string]$Name, [string]$Base = "http://localhost:6160/", [string]$Qs = "&lod=fine", [string]$OpsFile, [string]$Repeat = "1", [string]$WaitMax = "900")
$S = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\ca3a5e8b-220b-4b3d-a5e9-87f500e687dc\scratchpad\crowdfast"
Set-Location C:\Users\anita\projects\maps\.tw-trees\crowdfast
$env:TW_CORES_WANT = $Want
$env:NAME = $Name
$env:QS = $Qs
$env:OPS = Get-Content "$S\$OpsFile" -Raw
$env:REPEAT = $Repeat
$env:WAITMAX = $WaitMax
$env:BASE = $Base
if ($Name -like "*prof*") { $env:PROF = "1" } else { Remove-Item Env:PROF -ErrorAction SilentlyContinue }
node "$S\hang.mjs" *> "$S\out\$Name.log"
