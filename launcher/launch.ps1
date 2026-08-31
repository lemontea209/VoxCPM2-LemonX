# ============================================================
# yzylauncher-win
# Start if not running; bring window to front if already running
# ============================================================

# --- Config ---
$ExeRelativePath = "win-unpacked\yzylauncher-win.exe"
$ProcessName     = "yzylauncher-win"

# --- Switch to script directory ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$ExeFullPath = Join-Path $ScriptDir $ExeRelativePath

if (-not (Test-Path $ExeFullPath)) {
    Write-Host "[ERROR] Executable not found:" -ForegroundColor Red
    Write-Host "        $ExeFullPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Win32 API for reliable window activation ---
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")]
    public static extern bool IsIconic(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool BringWindowToTop(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
    [DllImport("kernel32.dll")]
    public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")]
    public static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);
}
"@

$SW_RESTORE = 9
$SW_SHOW    = 5

function Bring-WindowToFront([IntPtr]$hwnd) {
    if ($hwnd -eq [IntPtr]::Zero) { return $false }

    if ([Win32]::IsIconic($hwnd)) {
        [void][Win32]::ShowWindowAsync($hwnd, $SW_RESTORE)
    } else {
        [void][Win32]::ShowWindowAsync($hwnd, $SW_SHOW)
    }

    $foreHwnd = [Win32]::GetForegroundWindow()
    $dummy = [uint32]0
    $foreThread = [Win32]::GetWindowThreadProcessId($foreHwnd, [ref]$dummy)
    $currentThread = [Win32]::GetCurrentThreadId()

    $attached = $false
    if ($foreThread -ne $currentThread) {
        $attached = [Win32]::AttachThreadInput($currentThread, $foreThread, $true)
    }

    [void][Win32]::BringWindowToTop($hwnd)
    $ok = [Win32]::SetForegroundWindow($hwnd)

    if ($attached) {
        [void][Win32]::AttachThreadInput($currentThread, $foreThread, $false)
    }

    return $ok
}

# --- Main ---
$proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } |
        Select-Object -First 1

if ($null -eq $proc) {
    $any = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $any) {
        Write-Host "[INFO] Process exists but window not ready, waiting..." -ForegroundColor Yellow
        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Milliseconds 500
            $any.Refresh()
            if ($any.MainWindowHandle -ne 0) {
                $proc = $any
                break
            }
        }
    }
}

if ($null -ne $proc -and $proc.MainWindowHandle -ne 0) {
    Write-Host "[INFO] Already running (PID=$($proc.Id)), activating window..." -ForegroundColor Cyan
    $ok = Bring-WindowToFront $proc.MainWindowHandle
    if ($ok) {
        Write-Host "[OK] Window activated" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Activation call failed, but window may already be in front" -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] Not running, starting..." -ForegroundColor Cyan
    try {
        Start-Process -FilePath $ExeFullPath -WorkingDirectory (Split-Path $ExeFullPath -Parent)
        Write-Host "[OK] Started" -ForegroundColor Green
    } catch {
        $errMsg = $_.Exception.Message
        Write-Host "[ERROR] Failed to start: $errMsg" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}