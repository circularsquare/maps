# Wait until any of the given PIDs has exited, then print which and when
# (docs/adding-a-country.md 5: the one background wait per stage; never poll).
#
#   powershell -File waitany.ps1 <pid,pid,...> [-Every <seconds, default 30>] [-Timeout <seconds, default none>]
#   powershell -File waitany.ps1 --help
#
# Run it as a Bash call with run_in_background, then end your turn. Prints "exited: <pids> at
# <HH:mm:ss>" and exits 0; with -Timeout, "timeout, still running: <pids>" and exit 1. A PID that is
# already gone counts as exited at once, so give it the python PID (or the cmd PID) of a live run.
param([string]$Pids, [int]$Every = 30, [int]$Timeout = 0)
if (-not $Pids -or $Pids -in @('--help', '-h', '/?')) {
    foreach ($l in (Get-Content $MyInvocation.MyCommand.Path)) { if ($l -notmatch '^#') { break }; $l -replace '^# ?', '' }
    return
}
$ids = $Pids.Split(',') | ForEach-Object { [int]$_ }
$t0 = Get-Date
while ($true) {
    $gone = $ids | Where-Object { -not (Get-Process -Id $_ -ErrorAction SilentlyContinue) }
    if ($gone) { "exited: $($gone -join ',') at $(Get-Date -Format 'HH:mm:ss')"; exit 0 }
    if ($Timeout -gt 0 -and ((Get-Date) - $t0).TotalSeconds -ge $Timeout) {
        "timeout, still running: $($ids -join ',')"; exit 1
    }
    Start-Sleep $Every
}
