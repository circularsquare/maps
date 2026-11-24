# Keeps the lowest cores free for Anita's own work: everything a Claude session starts (agents' builds, browsers,
# Vite, checks) is pinned to the other cores at below-normal priority. The session's own claude.exe, the OpenConsole.exe
# above it and WindowsTerminal.exe run at AboveNormal on every core, since they draw her terminal (pinning it made typing lag 4 s). New children start unpinned,
# so the orchestrator runs ONE loop for every session at takeover: -All -Loop 30 (each pass walks every claude.exe, so
# sessions started later are covered too). A second -All -Loop exits at once while one runs (named mutex), so running it twice is harmless.
#   powershell -NoProfile -ExecutionPolicy Bypass -File trainworld/tools/ops/reserve_cores.ps1 -All -Loop 30
#   powershell -File trainworld/tools/ops/reserve_cores.ps1 [-Loop 30]        only the claude.exe this shell runs under
#   powershell -File trainworld/tools/ops/reserve_cores.ps1 -Root 1234        only another session's claude.exe
# Never touched, and never walked below: terminal hosts (a tab opened from this session makes WindowsTerminal.exe a
# descendant, and it hosts all her tabs), explorer, and any claude.exe (with -All each is a root of its own). Processes
# outside every claude.exe tree (her Chrome, apps) are never touched.
#   -WhatIf lists what would be pinned and changes nothing.
param([int]$Root = 0, [int]$Reserved = 2, [int]$Loop = 0, [switch]$WhatIf, [switch]$All)
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

$n = [Environment]::ProcessorCount
$allMask = ([int64]1 -shl $n) - 1
$mask = $allMask - (([int64]1 -shl $Reserved) - 1)
$script:mutex = $null
if ($All -and $Loop -gt 0) {
  $script:mutex = New-Object System.Threading.Mutex($false, 'Local\trainworld_reserve_cores_all')
  $got = $false
  try { $got = $script:mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $got = $true }
  if (-not $got) { Write-Output 'another -All loop is already running; exiting'; exit 0 }
}

$t = Get-Tree
if (-not $All -and $Root -eq 0) {
  $id = $PID
  while ($id -and $t.byId.ContainsKey($id) -and $t.byId[$id].Name -ne 'claude.exe') { $id = [int]$t.byId[$id].ParentProcessId }
  if (-not $id -or -not $t.byId.ContainsKey($id)) { Write-Output 'no claude.exe above this shell'; exit 1 }
  $Root = $id
}

# One session: its claude.exe and the OpenConsole.exe / WindowsTerminal.exe above it run at AboveNormal (her typing
# path); everything below it is pinned. Adds to $script:pinned / $script:refused.
function Invoke-Session($t, [int]$Root, $wtAll) {
  $boost = New-Object System.Collections.ArrayList
  $a = $Root
  while ($a -and $t.byId.ContainsKey($a)) {
    if (@('openconsole.exe','windowsterminal.exe') -contains ([string]$t.byId[$a].Name).ToLower()) { [void]$boost.Add($a) }
    $a = [int]$t.byId[$a].ParentProcessId
  }
  if (-not ($boost | ? { $t.byId[$_].Name -eq 'WindowsTerminal.exe' })) { foreach ($w in $wtAll) { [void]$boost.Add($w) } }
  if (-not $WhatIf) { try { $r = Get-Process -Id $Root -ErrorAction Stop; $r.ProcessorAffinity = [IntPtr]$allMask; $r.PriorityClass = 'AboveNormal' } catch {} }
  foreach ($b in $boost) {
    if ($WhatIf) { Write-Output ('would raise {0} {1} to AboveNormal' -f $b, $t.byId[$b].Name); continue }
    try { $bp = Get-Process -Id $b -ErrorAction Stop; $bp.ProcessorAffinity = [IntPtr]$allMask; $bp.PriorityClass = 'AboveNormal' } catch {}
  }
  $done = 0; $failed = 0
  $queue = New-Object System.Collections.Queue
  if ($t.kids.ContainsKey($Root)) { foreach ($k in $t.kids[$Root]) { $queue.Enqueue($k) } }
  while ($queue.Count) {
    $id = $queue.Dequeue()
    $nm = if ($t.byId.ContainsKey($id)) { [string]$t.byId[$id].Name } else { '' }
    if ($Skip -contains $nm.ToLower()) { continue }
    if ($WhatIf) { Write-Output ("would pin {0} {1} (session {2})" -f $id, $nm, $Root) }
    try {
      $pr = Get-Process -Id $id -ErrorAction Stop
      if ($WhatIf) { $done++; continue }
      if ([int64]$pr.ProcessorAffinity -band (([int64]1 -shl $Reserved) - 1)) { $pr.ProcessorAffinity = [IntPtr]$mask }
      if ($pr.PriorityClass -ne 'BelowNormal' -and $pr.PriorityClass -ne 'Idle') { $pr.PriorityClass = 'BelowNormal' }
      $done++
    } catch { $failed++ }
    if ($t.kids.ContainsKey($id)) { foreach ($k in $t.kids[$id]) { if ($k -ne $id) { $queue.Enqueue($k) } } }
  }
  $script:pinned += $done; $script:refused += $failed
}

try {
  while ($true) {
    $wt = @(Get-Process -Name WindowsTerminal -ErrorAction SilentlyContinue | % { [int]$_.Id })
    if ($All) { $roots = @($t.byId.Keys | ? { $t.byId[$_].Name -eq 'claude.exe' } | sort) }
    else {
      if (-not $t.byId.ContainsKey($Root) -or $t.byId[$Root].Name -ne 'claude.exe') { Write-Output "root $Root gone"; exit 0 }
      $roots = @($Root)
    }
    $script:pinned = 0; $script:refused = 0
    foreach ($r in $roots) { Invoke-Session $t $r $wt }
    $done = $script:pinned; $failed = $script:refused
    if ($Loop -le 0) {
      Write-Output ("{0} claude.exe above normal on all cores ({1}); {2} processes under them on cores {3}-{4} at below normal ({5} gone or refused)" -f $roots.Count, ($roots -join ','), $done, $Reserved, ($n - 1), $failed)
      break
    }
    Start-Sleep -Seconds $Loop
    $t = Get-Tree
  }
} finally { if ($script:mutex) { try { $script:mutex.ReleaseMutex() } catch {} } }
