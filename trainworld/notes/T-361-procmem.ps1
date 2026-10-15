param([string]$Profile)
# Chrome processes of one profile: the browser whose command line names the profile, and all its descendants.
$all = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'"
$root = $all | Where-Object { $_.CommandLine -like "*$Profile*" -and $_.CommandLine -notlike "*--type=*" }
$ids = @{}
foreach ($r in $root) { $ids[[int]$r.ProcessId] = $true }
$changed = $true
while ($changed) {
  $changed = $false
  foreach ($p in $all) {
    if (-not $ids.ContainsKey([int]$p.ProcessId) -and $ids.ContainsKey([int]$p.ParentProcessId)) { $ids[[int]$p.ProcessId] = $true; $changed = $true }
  }
}
$out = @()
foreach ($p in $all) {
  if ($ids.ContainsKey([int]$p.ProcessId)) {
    $t = 'browser'
    if ($p.CommandLine -match '--type=([a-z-]+)') { $t = $Matches[1] }
    if ($p.CommandLine -match '--utility-sub-type=([A-Za-z.]+)') { $t = $t + ':' + $Matches[1] }
    $out += [pscustomobject]@{ pid = [int]$p.ProcessId; type = $t; privateMB = [math]::Round($p.PrivatePageCount / 1MB); wsMB = [math]::Round($p.WorkingSetSize / 1MB) }
  }
}
$out | ConvertTo-Json -Compress
