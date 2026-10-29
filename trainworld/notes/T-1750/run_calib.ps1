$s = 'C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f9a7f88d-0531-4963-9df6-6771326e513f\scratchpad\t1750'
$bin = 'C:\Users\anita\projects\maps\trainworld\data\work\target\release\calib.exe'
$root = 'C:\Users\anita\projects\maps\trainworld'
New-Item -ItemType Directory -Force "$s\out" | Out-Null
$nocrowd = @('--no-cap', '--set', 'crowd_alpha=0,left_behind=0')
$runs = @(
  @('tyo_G', 'tyo', 'tyo_G', @()),
  @('tyo_B', 'tyo', 'tyo_B', @()),
  @('tyo_X', 'tyo', 'tyo_X', @()),
  @('tyo_G_nocap', 'tyo', 'tyo_G', @('--no-cap')),
  @('tyo_G_nocrowd', 'tyo', 'tyo_G', $nocrowd),
  @('tyo_X_nocrowd', 'tyo', 'tyo_X', $nocrowd),
  @('tyo_G_c15', 'tyo', 'tyo_G_c15', @()),
  @('tyo_X_c15', 'tyo', 'tyo_X_c15', @()),
  @('nyc_G', 'nyc', 'nyc_G', @()),
  @('nyc_R', 'nyc', 'nyc_R', @()),
  @('nyc_B', 'nyc', 'nyc_B', @()),
  @('nyc_G_nocap', 'nyc', 'nyc_G', @('--no-cap')),
  @('nyc_G_nocrowd', 'nyc', 'nyc_G', $nocrowd),
  @('nyc_G_c15', 'nyc', 'nyc_G_c15', @()),
  @('nyc_R_c15', 'nyc', 'nyc_R_c15', @())
)
foreach ($r in $runs) {
  $name = $r[0]; $pack = $r[1]; $net = $r[2]; $extra = $r[3]
  $t = Get-Date
  & $bin --net "$s\var\$net.json" --pack "$root\data\packs\$pack.json" --out "$s\out\$name.json" --rounds 3 @extra > "$s\out\$name.log" 2>&1
  "$name done in $([int]((Get-Date) - $t).TotalSeconds) s, exit $LASTEXITCODE" | Add-Content "$s\out\progress.log"
}
"ALL DONE" | Add-Content "$s\out\progress.log"
