# T-4025: CPU seconds of each thread of one process, with the thread's name (Chrome names its threads:
# CrRendererMain, DedicatedWorker thread, Compositor, ...). One line per thread: "<seconds> <name>".
#   powershell -NoProfile -File trainworld/tools/perf_threads.ps1 <pid>
param([int]$procId)
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class TwThreads {
  [DllImport("kernel32.dll", SetLastError = true)] public static extern IntPtr OpenThread(int access, bool inherit, int id);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
  [DllImport("kernel32.dll")] public static extern int GetThreadDescription(IntPtr h, out IntPtr name);
  [DllImport("kernel32.dll")] public static extern IntPtr LocalFree(IntPtr p);
  public static string Name(int id) {
    IntPtr h = OpenThread(0x1000, false, id); if (h == IntPtr.Zero) return "";
    try { IntPtr p; if (GetThreadDescription(h, out p) < 0) return ""; string s = Marshal.PtrToStringUni(p); LocalFree(p); return s ?? ""; } finally { CloseHandle(h); }
  }
}
"@
foreach ($t in (Get-Process -Id $procId).Threads) {
  try { '{0:N2} {1} {2}' -f $t.TotalProcessorTime.TotalSeconds, $t.Id, [TwThreads]::Name($t.Id) } catch { }
}
