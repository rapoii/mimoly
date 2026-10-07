<#
.SYNOPSIS
    Run `hermes verify` for mimoly and work around the Windows teardown hang.

.DESCRIPTION
    On Windows the Hermes verify runner terminates only the `cmd.exe` shell it
    spawned, not the child `python.exe` (there is no working process-group kill
    on Windows). The orphaned server keeps the stdout pipe open, so the runner's
    final `proc.stdout.read()` blocks forever: `hermes verify` never exits, never
    prints its JSON, and the orphan then squats on the port so the NEXT verify
    cannot boot — a snowball.

    This wrapper reproduces what a human does by hand:
      1. ensure the port is free (verify must bind it),
      2. launch `hermes verify --json` in the background,
      3. wait for readiness, then kill the server verify spawned,
      4. wait for verify to exit and print its result.

    It never kills a server that was already running before it started.

.PARAMETER Port
    Readiness port from the recipe. Default 8080.

.PARAMETER Force
    If the port is already in use, kill the occupant first instead of aborting.

.PARAMETER Workspace
    Hermes workspace root (where `hermes verify` is run). Defaults to this repo's
    grandparent when run from projects/mimoly, else the current directory.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File verify.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File verify.ps1 -Force
#>
[CmdletBinding()]
param(
    [int]$Port = 8080,
    [int]$ReadyTimeoutSec = 120,
    [int]$VerifyTimeoutSec = 420,
    [switch]$Force,
    [string]$Workspace = ""
)

$ErrorActionPreference = "Stop"

function Get-ListenerPid {
    param([int]$P)
    try {
        $c = Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction SilentlyContinue |
             Select-Object -First 1
        if ($c) { return [int]$c.OwningProcess }
    } catch { }
    return $null
}

function Stop-Tree {
    param([int]$ProcessId)
    # /T kills the whole tree: shell AND its python child (the part the runner misses).
    & taskkill /F /T /PID $ProcessId 2>$null | Out-Null
}

# --- resolve workspace root -------------------------------------------------
if (-not $Workspace) {
    $here = (Get-Location).Path
    # projects/mimoly -> workspace root is two levels up
    $cand = (Resolve-Path (Join-Path $here "..\..") -ErrorAction SilentlyContinue)
    if ($cand -and (Test-Path (Join-Path $cand ".hermes"))) { $Workspace = $cand.Path }
    elseif ($cand -and (Test-Path (Join-Path $cand "projects\mimoly"))) { $Workspace = $cand.Path }
    else { $Workspace = $here }
}
Write-Host "[verify] workspace: $Workspace"

# --- pre-flight: port must be free -----------------------------------------
$occupied = Get-ListenerPid $Port
if ($occupied) {
    if ($Force) {
        Write-Host "[verify] port $Port busy (PID $occupied) -> killing (-Force)"
        Stop-Tree $occupied
        Start-Sleep -Seconds 2
    } else {
        Write-Warning "Port $Port is already in use by PID $occupied. Stop it first, or re-run with -Force."
        exit 2
    }
}

# --- snapshot servers that existed BEFORE us (never kill these) -------------
$preExisting = @()
try {
    $preExisting = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'mimoly\.py\s+serve' } |
        Select-Object -ExpandProperty ProcessId)
} catch { }

# --- launch hermes verify in the background --------------------------------
$outFile = Join-Path $env:TEMP ("mimoly_verify_{0}.json" -f (Get-Random))
$errFile = "$outFile.err"
$hermes = (Get-Command hermes -ErrorAction SilentlyContinue)
if (-not $hermes) {
    Write-Error "`hermes` not found on PATH."
    exit 3
}
Write-Host "[verify] running: hermes verify --json  (-> $outFile)"
$psi = Start-Process -FilePath $hermes.Source `
    -ArgumentList "verify", "--json" `
    -WorkingDirectory $Workspace -NoNewWindow -PassThru `
    -RedirectStandardOutput $outFile -RedirectStandardError $errFile

# --- wait for readiness, then free the pipe by killing the spawned server ---
$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
$killed = $false
while ((Get-Date) -lt $deadline) {
    if ($psi.HasExited) { break }
    $lp = Get-ListenerPid $Port
    if ($lp -and ($preExisting -notcontains $lp)) {
        Start-Sleep -Milliseconds 700   # let the readiness probe get its 200 first
        Write-Host "[verify] server up (PID $lp); releasing stdout pipe"
        Stop-Tree $lp
        $killed = $true
        break
    }
    Start-Sleep -Milliseconds 500
}

# --- wait for verify itself to finish --------------------------------------
if (-not $psi.WaitForExit($VerifyTimeoutSec * 1000)) {
    Write-Warning "hermes verify did not exit within ${VerifyTimeoutSec}s; killing it."
    Stop-Tree $psi.Id
    Start-Sleep -Seconds 1
}

# --- report ----------------------------------------------------------------
$raw = if (Test-Path $outFile) { Get-Content -Raw $outFile } else { "" }
$json = $null
try { $json = $raw | ConvertFrom-Json } catch { }

Write-Host ""
if ($json) {
    Write-Host ("recipe : {0}" -f $json.recipe)
    Write-Host ("ok     : {0}" -f $json.ok)
    foreach ($p in $json.phases) {
        Write-Host ("  {0,-10} ok={1} exit={2} ({3}s)" -f $p.phase, $p.ok, $p.exitCode, $p.duration)
    }
    if ($json.readiness) {
        Write-Host ("  readiness  ready={0} status={1}" -f $json.readiness.ready, $json.readiness.statusCode)
    }
} else {
    Write-Host "Could not parse verify JSON. Raw tail:"
    Write-Host ($raw.Substring([Math]::Max(0, $raw.Length - 800)))
}

Remove-Item $outFile -ErrorAction SilentlyContinue
Remove-Item "$outFile.err" -ErrorAction SilentlyContinue

if ($json -and $json.ok) { exit 0 }
elseif (-not $killed) { Write-Warning "Server was never observed on port $Port; readiness may have failed."; exit 1 }
else { exit 1 }
