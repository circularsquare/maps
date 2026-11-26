# Rerun the T-4364 zone ranking: start it hidden at 4 workers, claim its cores under its own pid, release when done.
# From the maps root:  powershell -File trainworld\notes\T-4364\rank_zones.ps1   (about 10 minutes)
$root = "C:\Users\anita\projects\maps"
$log = "$root\artifacts\trainworld\zone-ranking.log"
$p = Start-Process -FilePath "$root\venv\Scripts\python.exe" -ArgumentList "-I", "$root\trainworld\notes\T-4364\rank_zones.py", "--jobs", "4" `
    -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
python "$root\cores.py" claim 4 rank_zones --pid $p.Id "T-4364 rank_zones.py" --timeout 0
$p.WaitForExit()
python "$root\cores.py" release rank_zones
"exit $($p.ExitCode)" | Out-File -Append -Encoding utf8 $log
