# Wait until any of the given PIDs has exited: waitany.ps1 <pid,pid,...>
param([string]$Pids)
$ids = $Pids.Split(',') | ForEach-Object { [int]$_ }
while ($true) {
    $gone = $ids | Where-Object { -not (Get-Process -Id $_ -ErrorAction SilentlyContinue) }
    if ($gone) { "exited: $($gone -join ',') at $(Get-Date -Format 'HH:mm:ss')"; break }
    Start-Sleep 30
}
