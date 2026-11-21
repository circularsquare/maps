# Keeps the lowest cores free for Anita's own work: pins a Claude session's process tree to the other cores and
# drops it to below-normal priority. New children inherit both, so run it once per orchestrator session (takeover).
#   powershell -File trainworld/tools/ops/reserve_cores.ps1            the claude.exe this shell runs under
#   powershell -File trainworld/tools/ops/reserve_cores.ps1 -Root 1234 another session's claude.exe
param([int]$Root = 0, [int]$Reserved = 2)

$all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name
$byId = @{}; $kids = @{}
foreach ($p in $all) {
  $byId[[int]$p.ProcessId] = $p
  $pp = [int]$p.ParentProcessId
  if (-not $kids.ContainsKey($pp)) { $kids[$pp] = New-Object System.Collections.ArrayList }
  [void]$kids[$pp].Add([int]$p.ProcessId)
}
if ($Root -eq 0) {
  $id = $PID
  while ($id -and $byId.ContainsKey($id) -and $byId[$id].Name -ne 'claude.exe') { $id = [int]$byId[$id].ParentProcessId }
  if (-not $id -or -not $byId.ContainsKey($id)) { Write-Output 'no claude.exe above this shell'; exit 1 }
  $Root = $id
}
$n = [Environment]::ProcessorCount
$mask = ([int64]1 -shl $n) - 1 - (([int64]1 -shl $Reserved) - 1)
$queue = New-Object System.Collections.Queue; $queue.Enqueue($Root); $done = 0; $failed = 0
while ($queue.Count) {
  $id = $queue.Dequeue()
  try {
    $pr = Get-Process -Id $id -ErrorAction Stop
    $pr.ProcessorAffinity = [IntPtr]$mask
    $pr.PriorityClass = 'BelowNormal'
    $done++
  } catch { $failed++ }
  if ($kids.ContainsKey($id)) { foreach ($k in $kids[$id]) { if ($k -ne $id) { $queue.Enqueue($k) } } }
}
Write-Output ("root {0}: {1} processes on cores {2}-{3} at below normal ({4} gone or refused); mask 0x{5:X}" -f $Root, $done, $Reserved, ($n - 1), $failed, $mask)
