# settle2: hang.mjs runs one after another on the cores settle2 holds. -PlanFile: lines name|QS|OPS json|REPEAT
param([string]$PlanFile)
$S = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\f3eca8a5-6550-4b65-9245-def721281ba7\scratchpad\settle2"
Set-Location C:\Users\anita\projects\maps
foreach ($item in Get-Content $PlanFile) {
  if (-not $item.Trim() -or $item.StartsWith("#")) { continue }
  $p = $item.Split("|")
  $env:TW_CORES = "3"
  $env:NAME = $p[0]
  $env:QS = $p[1]
  $env:OPS = $p[2]
  $env:REPEAT = $(if ($p.Length -gt 3) { $p[3] } else { "1" })
  $env:WAITMAX = $(if ($p.Length -gt 4) { $p[4] } else { "900" })
  $env:BASE = "http://localhost:6154/"
  node "$S\hang.mjs"
}
"series done"
