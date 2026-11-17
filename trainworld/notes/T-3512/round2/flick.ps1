# settle2: the bubble flicker runs, one after another
$S = "C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\f3eca8a5-6550-4b65-9245-def721281ba7\scratchpad\settle2"
Set-Location C:\Users\anita\projects\maps
$env:TW_CORES = "3"
$log = "$S\flick.log"
"== met-ref (perfOff=periods,localKeep,localSkip)" | Out-File $log -Encoding utf8
node "$S\flicker.mjs" http://localhost:6154/ met-ref met "&perfOff=periods,localKeep,localSkip" 2>&1 | Select-Object -Last 6 | Out-File $log -Append -Encoding utf8
"== met-noskip (perfOff=localSkip)" | Out-File $log -Append -Encoding utf8
node "$S\flicker.mjs" http://localhost:6154/ met-noskip met "&perfOff=localSkip" 2>&1 | Select-Object -Last 6 | Out-File $log -Append -Encoding utf8
"== nyc (default)" | Out-File $log -Append -Encoding utf8
node "$S\flicker.mjs" http://localhost:6154/ nyc-1 nyc "&lod=fine" 2>&1 | Select-Object -Last 6 | Out-File $log -Append -Encoding utf8
"== done" | Out-File $log -Append -Encoding utf8
