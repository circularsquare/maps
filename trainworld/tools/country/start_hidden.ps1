# Start a .cmd detached from this session, with no window, so it outlives the session
# (docs/adding-a-country.md 5: a run started from a session dies with it).
#
#   powershell -File start_hidden.ps1 <path.cmd> [-Cwd <folder>] [-NoChildren]
#   powershell -File start_hidden.ps1 --help
#
# Starts `cmd.exe /c "<path.cmd>"` through WMI in <folder> (default: the maps root, found from this
# file's place) and prints "pid <cmd pid> rc <0 = started>". Three seconds later it lists the cmd's
# child processes (the python of the run is one of them); -NoChildren skips the wait and the list.
# Write the cmd's PID and the python PID into your task notes before waiting
# (`waitany.ps1 <pid>` waits for either), and `python cores.py claim ... --pid <python pid>`.
param([string]$Cmd, [string]$Cwd, [switch]$NoChildren)
if (-not $Cmd -or $Cmd -in @('--help', '-h', '/?')) {
    foreach ($l in (Get-Content $MyInvocation.MyCommand.Path)) { if ($l -notmatch '^#') { break }; $l -replace '^# ?', '' }
    return
}
$Cmd = (Resolve-Path -LiteralPath $Cmd).Path
if (-not $Cwd) { $Cwd = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path }
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow = [uint16]0}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd.exe /c `"$Cmd`""; CurrentDirectory = $Cwd; ProcessStartupInformation = $si }
"pid $($r.ProcessId) rc $($r.ReturnValue)"
if (-not $NoChildren -and $r.ReturnValue -eq 0) {
    Start-Sleep 3
    $all = Get-CimInstance Win32_Process
    $queue = @($r.ProcessId)
    while ($queue.Count) {
        $head, $queue = $queue[0], @($queue | Select-Object -Skip 1)
        foreach ($p in ($all | Where-Object { $_.ParentProcessId -eq $head })) {
            "child $($p.ProcessId) $($p.Name) (of $head)"
            $queue += $p.ProcessId
        }
    }
}
