# crowdfast: runs one after another. -Plan "name:want:qs:ops;..."
param([string]$Plan)
$S = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\ca3a5e8b-220b-4b3d-a5e9-87f500e687dc\scratchpad\crowdfast"
foreach ($item in $Plan.Split(";")) {
  $p = $item.Split(":")
  & "$S\one.ps1" -Want $p[1] -Name $p[0] -Qs $p[2] -OpsFile $p[3]
}
"series done" | Out-File -Append "$S\out\runs.log"
