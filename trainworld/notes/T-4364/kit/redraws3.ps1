# T-4364 region redraws (agent redraws3, 11-26): every proposal Anita said yes to on 11-26, one country at a time:
# prefetch its extracts, back up, apply its plan edits and retires, build its rows in waves (givers first), back up
# again, seams, its one join; then the next. Egypt and Taiwan go last, behind a wait for her answer on the two
# 11-26 proposals (Mansoura; Chiayi and Tainan). notes/T-4364.md "The job".
# Start (no window):
#   powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-4364\kit\redraws3.cmd
# Dry run (read-only, in the foreground): powershell -File <this file> -Dry
# Stop: create trainworld\data\work\world_run\STOP, then kill this script's PID only (first line of redraws3.state).
# -From <step> skips the steps before it; steps are listed by -List (gb, il.edits, il.w1, il.join, ..., wait, eg.edits, ..., after).
param([int]$Want = 8, [int]$Min = 4, [string]$From = "start", [int]$FloorGB = 10, [switch]$Dry, [switch]$List,
      [string]$SimDir = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\554327b9-cb26-4c02-abcf-41d74d42cf79\scratchpad\redraws3\simcopy")
$ErrorActionPreference = "Continue"
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Notes = Split-Path -Parent $Kit
$Maps = "C:\Users\anita\projects\maps"
$TW = "$Maps\trainworld"
$Py = "$Maps\venv\Scripts\python.exe"
$WR = "$TW\pipeline\world_run.py"
$Edit = "$TW\notes\T-3688\kit\plan_edit.py"   # the kit that applies (T-4364's own copy refuses --go)
$Gated = "$TW\notes\T-3688\kit\gated.py"
$Art = "$Maps\artifacts\trainworld"
$Pre = "redraws3"
$Log = "$Art\$Pre-run.log"
$State = "$Kit\redraws3.state"
$StopFile = "$TW\data\work\world_run\STOP"
$Answer = "$Notes\answer-1126.txt"
$Downloads = "$Art\$Pre-downloads.txt"
$Port = 6379
$HandPort = 6378
$Name = "redraws3"
Set-Location $Maps
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# her answer on the two 11-26 proposals: answer-1126.txt, lines "mansoura=yes|no" and "chiayi=yes|no"
function Answer($key) {
  if (-not (Test-Path $Answer)) { return $null }
  foreach ($l in (Get-Content $Answer)) { if ($l -match "^\s*$key\s*=\s*(yes|no)\s*$") { return $Matches[1] } }
  return $null
}
function Countries() {
  $eg = if ((Answer "mansoura") -eq "yes") { @{Specs = @("egy-mansoura"); Waves = @("elm,elr", "brt,dut,mif,skm")} } else { @{Specs = @("egy-delta"); Waves = @("elr")} }
  $tw = if ((Answer "chiayi") -eq "yes") { @{Specs = @("twn-all"); Waves = @("eai,tig", "huw,kao,taa,tac")} } else { @{Specs = @("twn-taitung"); Waves = @("tig", "huw,kao,tac")} }
  return @(
    @{K = "il"; Tag = "T4364il"; Bk = "IL,PS,JO,LB,SY"; Specs = @("isr"); Waves = @("hif,tel"); Join = "IL"},
    @{K = "de"; Tag = "T4364de"; Bk = "DE,NL,BE,LU,FR,CH,AT,CZ,PL,DK"; Specs = @("deu-mettmann"); Waves = @("dus"); Join = "DE"},
    @{K = "in"; Tag = "T4364in"; Bk = "IN,NP,BD"; Specs = @("ind-patna"); Waves = @("bes,hjp,pat"); Join = "IN"},
    @{K = "mx"; Tag = "T4364mx"; Bk = "MX,US,GT,BZ"; Specs = @("mex"); Waves = @("mia", "fel", "vdd"); Join = "MX"},
    @{K = "no"; Tag = "T4364no"; Bk = "NO,SE,FI,RU"; Specs = @("nor-stavanger", "nor-islands"); Waves = @("vea,osl,tlk", "svg,stf,vef"); Join = "NO"},
    @{K = "cn"; Tag = "T4364cn"; Bk = "CN,MO,HK,VN,KP"; Specs = @("cn-guangdong", "cn-huiyang", "cn-guilin", "cn-nanning", "cn-wuhan", "cn-liaoyang", "mac");
      Waves = @("sih,tis,lgx,liu,shz,scn,nag,axx,wuh", "qnc,zhu,pns", "ans,mcu"); Join = "CN,MO"; Check = "macau"},
    @{K = "us"; Tag = "T4364us"; Bk = "US,CA"; Specs = @(); Boston = $true; Waves = @("hand:bos", "his"); Join = "US"},
    @{K = "eg"; Tag = "T4364eg"; Bk = "EG,IL,PS,SA,SD,LY"; Specs = $eg.Specs; Waves = $eg.Waves; Join = "EG"; Late = $true},
    @{K = "tw"; Tag = "T4364tw"; Bk = "TW"; Specs = $tw.Specs; Waves = $tw.Waves; Join = "TW"; Late = $true}
  )
}
function Steps() {
  $o = @("start", "gb")
  foreach ($c in (Countries)) {
    if ($c.Late -and -not ($o -contains "wait")) { $o += "wait" }
    $o += "$($c.K).edits"
    for ($i = 1; $i -le $c.Waves.Count; $i++) { $o += "$($c.K).w$i" }
    $o += "$($c.K).join"
  }
  return $o + @("after")
}
$Order = Steps
if ($List) { $Order; return }
if (-not ($Order -contains $From)) { "unknown -From $From; steps: $($Order -join ' ')"; exit 2 }
function Reached($s) { return $Order.IndexOf($From) -le $Order.IndexOf($s) }
$script:Stage = "start"

function Say($m) {
  $l = "$(Get-Date -Format s) [$script:Stage] $m"
  if ($Dry) { $l; return }
  Add-Content -Path $Log -Value $l -Encoding utf8
  Set-Content -Path $State -Value @("pid $PID", $l) -Encoding utf8
}
function Run($name, $argline, $out) {
  Say "start $name : $argline > $out"
  $p = Start-Process -FilePath $Py -ArgumentList $argline -WorkingDirectory $Maps -RedirectStandardOutput $out -RedirectStandardError "$out.err" -NoNewWindow -PassThru
  $null = $p.Handle  # without the handle held, ExitCode reads empty after WaitForExit
  $p.WaitForExit()
  Say "end $name : exit $($p.ExitCode)"
  return $p.ExitCode
}
function Release() { if (-not $Dry) { & $Py -I "$Maps\cores.py" release $Name 2>&1 | Out-Null } }
function Quit($code, $m) { Say "FAILED: $m"; Release; exit $code }
function FreeGB() { return [math]::Round((Get-PSDrive C).Free / 1GB, 1) }
function Floor($what) { $f = FreeGB; if ($f -lt $FloorGB) { Quit 8 "stopped before $what : $f GB free, under $FloorGB" }; Say "disk $f GB free before $what" }
function Stopped() { if (Test-Path $StopFile) { Quit 9 "STOP file ($StopFile)" } }
function Claim() {
  $o = & $Py -I "$Maps\cores.py" claim $Want $Name --pid $PID "T-4364 region redraws" 2>&1
  $g = [int](($o | Select-Object -Last 1).ToString().Trim())
  if ($g -lt $Min) { Quit 3 "cores granted $g, want $Min or more" }
  $script:J = [Math]::Min($g, $Want)
  $env:NQ_JOBS = "$script:J"
  Say "cores granted $g, jobs $script:J"
}
# specs are applied from copies in the run's folder, so plan_edit's <spec>.waves.json lands there, beside a record of
# exactly what was applied
$Specs = "$Art\$Pre-specs"
function Spec($s) {
  if (-not $Dry) {
    New-Item -ItemType Directory -Force $Specs | Out-Null
    if (-not (Test-Path "$Specs\t4364-$s.json")) { Copy-Item "$Notes\t4364-$s.json" "$Specs\t4364-$s.json" }
    return "$Specs\t4364-$s.json"
  }
  return "$Notes\t4364-$s.json"
}

# ---------------------------------------------------------------- dry run: everything read-only, then stop
if ($Dry) {
  $tmp = "$env:TEMP\redraws3-dry"
  New-Item -ItemType Directory -Force $tmp | Out-Null
  "steps: $($Order -join ' ')"
  "answers now: mansoura=$(Answer 'mansoura') chiayi=$(Answer 'chiayi') (no file or no line = no)"
  & $Py -I $WR status 2>&1 | Where-Object { $_ -match "^lock |^STOP" }
  "disk $(FreeGB) GB free"
  $all = @()
  foreach ($c in (Countries)) { foreach ($w in $c.Waves) { $all += ($w -replace "^hand:", "") -split "," } }
  & $Py -I "$Kit\prefetch.py" (($all | Where-Object { $_ -ne "bos" }) -join ",") --extra north-america/us/massachusetts --skip-missing --dry
  Copy-Item (Spec "gbr-brighton-names") "$tmp\t4364-gbr-brighton-names.json" -Force
  & $Py -I $Edit "$tmp\t4364-gbr-brighton-names.json" 2>&1 | Select-Object -Last 3
  $variants = @(Countries)
  foreach ($c in $variants) {
    "=== $($c.K): backup $($c.Tag) $($c.Bk); specs $($c.Specs -join ' '); waves $($c.Waves -join ' | '); join $($c.Join)"
    & $Py -I "$TW\tools\country\backup.py" $c.Tag pre $c.Bk --dry 2>&1 | Select-Object -Last 1
    foreach ($s in $c.Specs) {
      Copy-Item (Spec $s) "$tmp\t4364-$s.json" -Force
      $o = & $Py -I $Edit "$tmp\t4364-$s.json" 2>&1
      ($o | Where-Object { $_ -match "^(waves|retired|names|\S+ .*: people)|Error|Traceback|not in|no such" }) -join "`n"
      & $Py -I $Edit "$tmp\t4364-$s.json" retire 2>&1 | Select-Object -Last 8
    }
    if ($c.Boston) { & $Py -I "$Kit\boston_edit.py" 2>&1 }
  }
  # the other answer's Egypt and Taiwan specs too
  foreach ($s in @("egy-mansoura", "egy-delta", "twn-all", "twn-taitung")) {
    Copy-Item (Spec $s) "$tmp\t4364-$s.json" -Force
    "--- $s"; (& $Py -I $Edit "$tmp\t4364-$s.json" 2>&1 | Where-Object { $_ -match "^(waves|retired)|Error|not in|no such" }) -join "`n"
  }
  foreach ($p in @($Port, $HandPort)) { if (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) { "port $p IN USE" } else { "port $p free" } }
  & $Py -I "$TW\tools\country\setup_sim.py" $SimDir check 2>&1 | Select-Object -Last 2
  "DRY DONE"
  return
}

# ---------------------------------------------------------------- start
Say "redraws3.ps1 started, pid $PID, from $From; steps $($Order -join ' ')"
Stopped
$s = & $Py -I $WR status 2>&1 | Out-String
$locks = ($s -split "\r?\n") | Where-Object { $_ -match "^lock " }
if ($locks) { Quit 3 "locks held: $($locks -join ' | ')" }
if ((Run "sim check" "-I $TW\tools\country\setup_sim.py $SimDir check" "$Art\$Pre-sim-check.log") -ne 0) { Quit 3 "sim copy does not check (setup_sim.py <folder> make, then -SimDir <folder>)" }
$env:TW_SIM_DIR = "$SimDir\simroot\sim"
$env:TW_CARGO_TARGET_DIR = "$SimDir\simroot\sim\target"
Claim

if (Reached "gb") {
  $script:Stage = "gb"
  # Brighton's rename (zone-names.json only), before the first backup so every later backup holds it
  if ((Run "names gb" "-I $Edit $(Spec 'gbr-brighton-names') --go" "$Art\$Pre-gb-names.log") -ne 0) { Quit 4 "Brighton rename failed (zone-names.json as it was)" }
}

$rebuilt = @()
foreach ($c in (Countries)) {
  $k = $c.K
  if ($c.Late -and (Reached "wait") -and -not $script:waited) {
    $script:waited = $true
    $script:Stage = "wait"
    if ((Answer "mansoura") -and (Answer "chiayi")) { Say "answers: mansoura=$(Answer 'mansoura') chiayi=$(Answer 'chiayi')" }
    else {
      Release
      Say "waiting for $Answer (lines mansoura=yes|no and chiayi=yes|no); cores released; checked every 5 minutes"
      while (-not ((Answer "mansoura") -and (Answer "chiayi"))) { Stopped; Start-Sleep 300 }
      Say "answers: mansoura=$(Answer 'mansoura') chiayi=$(Answer 'chiayi')"
      Claim
    }
    # the specs and waves follow the answers: Countries() and the step list again
    $c = (Countries) | Where-Object { $_.K -eq $k }
    $Order = Steps
    if (-not ($Order -contains $From)) { Quit 2 "-From $From is not a step with these answers: $($Order -join ' ')" }
  }
  foreach ($w in $c.Waves) { $rebuilt += ($w -replace "^hand:", "") -split "," }
  $undo = "undo: venv\Scripts\python.exe -I trainworld\tools\country\restore.py $($c.Tag) pre --go (notes/T-4364.md 'The job')"
  $ids = (($c.Waves | ForEach-Object { $_ -replace "^hand:", "" }) -join ",")
  if (Reached "$k.edits") {
    $script:Stage = "$k.edits"
    Stopped
    $plan = (($ids -split ",") | Where-Object { $_ -ne "bos" }) -join ","
    $extra = if ($c.Boston) { "--extra north-america/us/massachusetts" } else { "" }
    # specs' new rows (vdd, svg, skm) are not plan rows yet: their extracts are their templates', fetched after the edits
    if ((Run "prefetch $k" "-I $Kit\prefetch.py $plan $extra --list $Downloads --skip-missing" "$Art\$Pre-$k-prefetch.log") -ne 0) { Quit 6 "$k : an extract would not download ($Pre-$k-prefetch.log); nothing of $k written" }
    Floor "backup $k"
    if ((Run "backup $k pre" "-I $TW\tools\country\backup.py $($c.Tag) pre $($c.Bk)" "$Art\$Pre-$k-backup-pre.log") -ne 0) { Quit 6 "$k : backup pre failed; nothing of $k written" }
    Say "PACK WRITES FROM HERE ($k)"
    foreach ($s in $c.Specs) {
      if ((Run "edit $s" "-I $Edit $(Spec $s) --go" "$Art\$Pre-edit-$s.log") -ne 0) { Quit 4 "plan edit $s failed; $undo" }
    }
    if ($c.Boston) {
      if ((Run "boston edit" "-I $Kit\boston_edit.py --go" "$Art\$Pre-edit-boston.log") -ne 0) { Quit 4 "boston_edit failed; $undo, then boston_edit.py --undo" }
    }
    foreach ($s in $c.Specs) {
      if ((Run "retire $s" "-I $Edit $(Spec $s) retire --go" "$Art\$Pre-retire-$s.log") -ne 0) { Quit 4 "retire $s failed; $undo" }
    }
    if ((Run "prefetch $k after edits" "-I $Kit\prefetch.py $plan $extra --list $Downloads" "$Art\$Pre-$k-prefetch2.log") -ne 0) { Quit 6 "$k : an extract would not download ($Pre-$k-prefetch2.log); $undo" }
  }
  for ($i = 1; $i -le $c.Waves.Count; $i++) {
    $w = $c.Waves[$i - 1]
    if (-not (Reached "$k.w$i")) { continue }
    $script:Stage = "$k.w$i"
    Stopped
    if ($w -like "hand:*") {
      # a hand pack (Boston): build_city and its later steps through world_run's step runner, then world_run's own gate
      $h = $w -replace "^hand:", ""
      $ex = "north-america/us/massachusetts"
      # the extract's OSM caches first, as world_run reads a region's before its builds: the stations and routes
      # steps and the gate read them (T-4364: bos's stations step found none). Caches made here go with the downloads
      $stem = "$TW\data\raw\osm\" + ($ex -replace "/", "_")
      $caches = @(".points.fgb", ".stations.fgb", ".areas.fgb", ".water.fgb", ".ways2.npz") | ForEach-Object { "$stem$_" }
      $had = @($caches | Where-Object { Test-Path -LiteralPath $_ })
      # and, as world_run takes a registered row out before rebuilding it, the neighbours' market packs healed of their
      # commutes into the hand pack (a smaller rebuild leaves them out of range at the gate's register), its entry kept
      if ((Run "$h heal" "-I -u $Kit\hand_run.py heal $h --jobs $script:J" "$Art\$Pre-$h-heal.log") -ne 0) { Quit 4 "$h heal failed ($Pre-$h-heal.log); no join. $undo, then boston_edit.py --undo" }
      if ((Run "$h read" "-I -u $WR step @read_extract $ex" "$Art\$Pre-$h-read.log") -ne 0) { Quit 4 "$h read of $ex failed ($Pre-$h-read.log); no join. $undo, then boston_edit.py --undo" }
      $caches | Where-Object { (Test-Path -LiteralPath $_) -and ($had -notcontains $_) } | Add-Content -Path $Downloads -Encoding utf8
      $env:OMP_NUM_THREADS = "$script:J"; $env:CARGO_BUILD_JOBS = "$script:J"
      foreach ($st in @(@("build", "step build_city $h"), @("stations", "step pack_gate stations $h --extracts $ex"), @("places", "step places $h"),
                        @("routes", "step routes $h --extracts $ex --workers $script:J --keep-pbf"), @("merge", "step merge_cells $h"),
                        @("everyday", "step everyday_rates $h"))) {
        if ((Run "$h $($st[0])" "-I -u $WR $($st[1])" "$Art\$Pre-$h-$($st[0]).log") -ne 0) { Remove-Item Env:OMP_NUM_THREADS, Env:CARGO_BUILD_JOBS; Quit 4 "$h $($st[0]) failed ($Pre-$h-$($st[0]).log); no join. $undo, then boston_edit.py --undo" }
      }
      Remove-Item Env:OMP_NUM_THREADS, Env:CARGO_BUILD_JOBS
      Run "$h water" "-I -u $WR step look_water $h $Art\$Pre-$h-water.png" "$Art\$Pre-$h-water.log" | Out-Null
      # the gate reads the build's figures from <logs>/<id>.build.log, as world_run's gate does
      New-Item -ItemType Directory -Force "$Art\$Pre-hand" | Out-Null
      Copy-Item "$Art\$Pre-$h-build.log" "$Art\$Pre-hand\$h.build.log" -Force
      if ((Run "$h gate" "-I -u $Kit\hand_run.py gate $h --jobs $script:J --port $HandPort --logs-from $Art\$Pre-hand" "$Art\$Pre-$h-gate.log") -ne 0) { Quit 4 "$h gate failed ($Pre-$h-gate.log); no join. $undo, then boston_edit.py --undo" }
      continue
    }
    $rc = Run "run $k w$i" "-I $WR run --rows $w --jobs $script:J --port $Port --no-join --keep-downloads" "$Art\$Pre-$k-w$i.log"
    $gr = Run "gated $k w$i" "-I $Gated $w" "$Art\$Pre-$k-gated-w$i.log"
    if ($rc -ne 0) { Quit 4 "$k w$i run exit $rc ($Pre-$k-w$i.log); no join. $undo" }
    if ($gr -ne 0) { Quit 4 "$k w$i : a row is not gated ($Pre-$k-gated-w$i.log); no join. $undo" }
  }
  if ($c.Check -eq "macau" -and (Reached "$k.join")) {
    if ((Run "macau check" "-I $Kit\macau_check.py" "$Art\$Pre-macau-check.log") -ne 0) { Quit 4 "Zhuhai's new outline holds Macau's land ($Pre-macau-check.log); no join. $undo" }
  }
  if (Reached "$k.join") {
    $script:Stage = "$k.join"
    Stopped
    Floor "backup $k prejoin"
    if ((Run "backup $k prejoin" "-I $TW\tools\country\backup.py $($c.Tag) prejoin $($c.Bk)" "$Art\$Pre-$k-backup-prejoin.log") -ne 0) { Quit 6 "$k : prejoin backup failed; no join" }
    if ((Run "seams $k" "-I $TW\pipeline\seams.py" "$Art\$Pre-$k-seams.log") -ne 0) { Say "warning: seams.py failed ($Pre-$k-seams.log); borders.json as it was" }
    Floor "join $k"
    $rc = Run "join $k" "-I $WR join --countries $($c.Join) --jobs $script:J --port $Port" "$Art\$Pre-$k-join.log"
    Run "status $k" "-I $WR status" "$Art\$Pre-$k-status.log" | Out-Null
    if ($rc -ne 0) { Quit 5 "$k join exit $rc ($Pre-$k-join.log); undo: restore.py $($c.Tag) prejoin --go, then $($c.Tag) pre" }
    $gr = Run "shipped $k" "-I $Kit\shipped.py $ids" "$Art\$Pre-$k-shipped.log"
    if ($gr -ne 0) { Quit 5 "$k : a row is not shipped after its join ($Pre-$k-shipped.log); undo: restore.py $($c.Tag) prejoin --go, then $($c.Tag) pre" }
    # the join is clean: its prejoin copy goes (pre stays as the undo)
    $pj = "$TW\data\work\pre-$($c.Tag)\prejoin"
    if (Test-Path $pj) { Remove-Item -LiteralPath $pj -Recurse -Force; Say "prejoin backup $pj deleted (joined clean)" }
    Say "joined $k"
  }
}

if (Reached "after") {
  $script:Stage = "after"
  Stopped
  if ((Run "stale" "-I $TW\tools\country\stale.py" "$Art\$Pre-stale.log") -ne 0) { Quit 7 "market packs name freed places ($Pre-stale.log); the packs are shipped" }
  $succ = @("has=tel", "rml=tel", "asq=tel", "akk=hif", "yiz=hif", "kax=mia", "met=dus", "mkm=bes", "bbm=pat", "lbe=lgx", "hnp=axx", "hya=shz", "ygi=nag", "lbk=ans", "smd=elr")
  if ((Answer "mansoura") -eq "yes") { $succ += "aad=elm" }
  if ((Answer "chiayi") -eq "yes") { $succ += "mig=eai" }
  $sets = ($succ | ForEach-Object { "--set $_" }) -join " "
  Run "successors dry" "-I $TW\pipeline\zone_successors.py --dry" "$Art\$Pre-successors-dry.log" | Out-Null
  if ((Run "successors" "-I $TW\pipeline\zone_successors.py $sets" "$Art\$Pre-successors.log") -ne 0) { Quit 7 "zone_successors failed ($Pre-successors.log)" }
  Run "zone_terrain" "trainworld\pipeline\zone_terrain.py build --jobs $script:J" "$Art\$Pre-zone-terrain.log" | Out-Null
  Say "PACKS FREE: no more pack or terrain writes from this run"
  Run "status end" "-I $WR status" "$Art\$Pre-status-end.log" | Out-Null
  if ((Run "start tiers" "$TW\pipeline\start_difficulty.py --fresh --jobs $script:J" "$Art\$Pre-start-difficulty.log") -ne 0) { Quit 7 "start tiers failed; the packs are shipped" }
  if ((Run "build_inputs" "trainworld\checks\build_inputs.py" "$Art\$Pre-build-inputs.log") -ne 0) { Quit 7 "build_inputs flags ($Pre-build-inputs.log); the packs are shipped" }
  $ids = ($rebuilt | Select-Object -Unique) -join ","
  if ((Run "split_zones" "trainworld\checks\split_zones.py --ids $ids" "$Art\$Pre-split-zones.log") -ne 0) { Quit 7 "split_zones finds a split zone ($Pre-split-zones.log); the packs are shipped" }
  if (Test-Path $Downloads) {
    foreach ($f in (Get-Content $Downloads | Select-Object -Unique)) { if ($f -and (Test-Path -LiteralPath $f)) { Remove-Item -LiteralPath $f -Force; Say "deleted $f" } }
  }
}
$script:Stage = "end"
Release
Say "DONE: all steps passed"
