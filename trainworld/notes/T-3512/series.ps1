# T-3512 before/after series: each run waits for 3 free cores, then traces one save.
# Names: nyc-* or cn-* (the save); *-new-* the settle tree as built, *-old-* the tree with ?perfOff=localKeep,localSkip,
# *-main-* the main tree's Vite on 6147 (master as it was). Every run pins zone levels with ?lod=fine.
param([string]$Plan = "nyc-new-1,nyc-old-1", [string]$Edits = "low,undo,all,undo,high,undo")
$S = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\f3eca8a5-6550-4b65-9245-def721281ba7\scratchpad\settle"
$PY = "C:\Users\anita\projects\maps\venv\Scripts\python.exe"
Set-Location C:\Users\anita\projects\maps
& $PY C:\Users\anita\projects\maps\cores.py claim 3 settle --pid $PID "T-3512 settle traces" | Out-Null
foreach ($name in $Plan.Split(",")) {
  & $PY "$S\waitcores.py" 3 | Out-Null
  $env:TW_CORES = "3"
  $env:NAME = $name
  if ($name -like "*-main-*") {
    $env:BASE = "http://localhost:6147/"
    $env:LIB = "file:///C:/Users/anita/projects/maps/trainworld/checks/"
  } else {
    $env:BASE = "http://localhost:6148/"
    $env:LIB = "file:///C:/Users/anita/projects/maps/.tw-trees/settle/trainworld/checks/"
  }
  $env:EDITS = $Edits
  # *-ref-*: every edit's cluster solved whole from nothing, no period kept and no local solve (the figures' reference)
  $env:QS = "&lod=fine" + $(if ($name -like "*-old-*") { "&perfOff=localKeep,localSkip" } elseif ($name -like "*-ref-*") { "&perfOff=periods,localKeep,localSkip" } else { "" })
  if ($name -like "cn-*") { $env:SAVE = "C:/Users/anita/projects/maps/t1156-all-china.save" } else { Remove-Item Env:SAVE -ErrorAction SilentlyContinue }
  node "$S\trace.mjs"
}
& $PY C:\Users\anita\projects\maps\cores.py release settle | Out-Null
"series done"
