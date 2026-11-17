# settle2: one hang.mjs run. -Name -Base -Qs -Ops (json) -Repeat -WaitMax
param([string]$Name, [string]$Base = "http://localhost:6154/", [string]$Qs = "&lod=fine", [string]$OpsFile, [string]$Repeat = "1", [string]$WaitMax = "900")
$S = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\f3eca8a5-6550-4b65-9245-def721281ba7\scratchpad\settle2"
Set-Location C:\Users\anita\projects\maps
$env:TW_CORES = "3"
$env:NAME = $Name
$env:QS = $Qs
$env:OPS = Get-Content $OpsFile -Raw
$env:REPEAT = $Repeat
$env:WAITMAX = $WaitMax
$env:BASE = $Base
node "$S\hang.mjs"
