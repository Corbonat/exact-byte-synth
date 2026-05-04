# Keep the system awake while a long-running benchmark is active.
# Uses SetThreadExecutionState with ES_SYSTEM_REQUIRED + ES_CONTINUOUS
# to suppress idle standby without requiring admin rights.
# Stops automatically when the script exits or is killed.

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PowerUtil {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS       = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED  = 0x00000001;
    public const uint ES_AWAYMODE_REQUIRED= 0x00000040;
}
"@

[void][PowerUtil]::SetThreadExecutionState(
    [PowerUtil]::ES_CONTINUOUS -bor
    [PowerUtil]::ES_SYSTEM_REQUIRED -bor
    [PowerUtil]::ES_AWAYMODE_REQUIRED
)

Write-Host "keep-awake armed (pid=$PID). Ctrl+C to release."
try {
    while ($true) { Start-Sleep -Seconds 30 }
} finally {
    [void][PowerUtil]::SetThreadExecutionState([PowerUtil]::ES_CONTINUOUS)
    Write-Host "keep-awake released."
}
