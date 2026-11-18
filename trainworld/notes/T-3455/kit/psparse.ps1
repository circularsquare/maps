param([string]$f)
$t=$null;$e=$null
[System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e) | Out-Null
if ($e.Count) { $e | ForEach-Object { "$($_.Extent.StartLineNumber): $($_.Message)" }; exit 1 } else { "parse ok" }
