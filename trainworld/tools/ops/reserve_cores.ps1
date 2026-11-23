# Keeps the lowest cores free for Anita's own work: everything a Claude session starts (agents' builds, browsers,
# Vite, checks) is pinned to the other cores at below-normal priority. The session's own claude.exe stays at normal
# priority on every core, since it draws her terminal (pinning it made typing lag 4 s). New children start unpinned,
# so the orchestrator runs it with -Loop in the background at takeover.
#   powershell -File trainworld/tools/ops/reserve_cores.ps1 [-Loop 30]        the claude.exe this shell runs under
#   powershell -File trainworld/tools/ops/reserve_cores.ps1 -Root 1234        another session's claude.exe
# Never touched, and never walked below: terminal hosts (a tab opened from this session makes WindowsTerminal.exe a
# descendant, and it hosts all her tabs), explorer, and any claude.exe (another session's loop decides for its own children).
#   -WhatIf lists what would be pinned and changes nothing.
param([int]$Root = 0, [int]$Reserved = 2, [int]$Loop = 0, [switch]$WhatIf)
$Skip = @('windowsterminal.exe','openconsole.exe','conhost.exe','wt.exe','explorer.exe','claude.exe')

function Get-Tree {
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name
  $byId = @{}; $kids = @{}
  foreach ($p in $all) {
    $byId[[int]$p.ProcessId] = $p
    $pp = [int]$p.ParentProcessId
    if (-not $kids.ContainsKey($pp)) { $kids[$pp] = New-Object System.Collections.ArrayList }
    [void]$kids[$pp].Add([int]$p.ProcessId)
  }
  return @{ byId = $byId; kids = $kids }
}

$t = Get-Tree
if ($Root -eq 0) {
  $id = $PID
  while ($id -and $t.byId.ContainsKey($id) -and $t.byId[$id].Name -ne 'claude.exe') { $id = [int]$t.byId[$id].ParentProcessId }
  if (-not $id -or -not $t.byId.ContainsKey($id)) { Write-Output 'no claude.exe above this shell'; exit 1 }
  $Root = $id
}
$n = [Environment]::ProcessorCount
$all = ([int64]1 -shl $n) - 1
$mask = $all - (([int64]1 -shl $Reserved) - 1)

while ($true) {
  try { $r = Get-Process -Id $Root -ErrorAction Stop } catch { Write-Output "root $Root gone"; exit 0 }
  if (-not $WhatIf) { try { $r.ProcessorAffinity = [IntPtr]$all; $r.PriorityClass = 'Normal' } catch {} }
  $done = 0; $failed = 0
  $queue = New-Object System.Collections.Queue
  if ($t.kids.ContainsKey($Root)) { foreach ($k in $t.kids[$Root]) { $queue.Enqueue($k) } }
  while ($queue.Count) {
    $id = $queue.Dequeue()
    $nm = if ($t.byId.ContainsKey($id)) { [string]$t.byId[$id].Name } else { '' }
    if ($Skip -contains $nm.ToLower()) { continue }
    if ($WhatIf) { Write-Output ("would pin {0} {1}" -f $id, $nm) }
    try {
      $pr = Get-Process -Id $id -ErrorAction Stop
      if ($WhatIf) { $done++; continue }
      if ([int64]$pr.ProcessorAffinity -band 3) { $pr.ProcessorAffinity = [IntPtr]$mask }
      if ($pr.PriorityClass -ne 'BelowNormal' -and $pr.PriorityClass -ne 'Idle') { $pr.PriorityClass = 'BelowNormal' }
      $done++
    } catch { $failed++ }
    if ($t.kids.ContainsKey($id)) { foreach ($k in $t.kids[$id]) { if ($k -ne $id) { $queue.Enqueue($k) } } }
  }
  if ($Loop -le 0) {
    Write-Output ("root {0} kept normal on all cores; {1} processes under it on cores {2}-{3} at below normal ({4} gone or refused)" -f $Root, $done, $Reserved, ($n - 1), $failed)
    break
  }
  Start-Sleep -Seconds $Loop
  $t = Get-Tree
}
