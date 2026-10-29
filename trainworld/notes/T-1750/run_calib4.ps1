$s = 'C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f9a7f88d-0531-4963-9df6-6771326e513f\scratchpad\t1750'
$bin = 'C:\Users\anita\projects\maps\trainworld\data\work\target\release\calib.exe'
$root = 'C:\Users\anita\projects\maps\trainworld'
$runs = @(
  @('nyc_G_xfer0', 'nyc', 'nyc_G', 'transfer_min=0'),
  @('nyc_G_park14', 'nyc', 'nyc_G', 'park_min=14'),
  @('nyc_G_wdrive16', 'nyc', 'nyc_G', 'w_drive=1.6'),
  @('tyo_G_xfer0', 'tyo', 'tyo_G', 'transfer_min=0'),
  @('tyo_G_park14', 'tyo', 'tyo_G', 'park_min=14'),
  @('tyo_G_wdrive16', 'tyo', 'tyo_G', 'w_drive=1.6')
)
foreach ($r in $runs) {
  $name = $r[0]; $pack = $r[1]; $net = $r[2]; $set = $r[3]
  $t = Get-Date
  & $bin --net "$s\var\$net.json" --pack "$root\data\packs\$pack.json" --out "$s\out\$name.json" --rounds 3 --set $set > "$s\out\$name.log" 2>&1
  "$name done in $([int]((Get-Date) - $t).TotalSeconds) s, exit $LASTEXITCODE" | Add-Content "$s\out\progress4.log"
}
"ALL DONE" | Add-Content "$s\out\progress4.log"
