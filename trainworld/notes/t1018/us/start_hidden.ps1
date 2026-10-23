# Start a .cmd detached from this session (WMI) with no visible window: start_hidden.ps1 <path.cmd>
param([string]$Cmd)
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow = [uint16]0}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd.exe /c `"$Cmd`""; CurrentDirectory = 'C:\Users\anita\projects\maps'; ProcessStartupInformation = $si }
"pid $($r.ProcessId) rc $($r.ReturnValue)"
