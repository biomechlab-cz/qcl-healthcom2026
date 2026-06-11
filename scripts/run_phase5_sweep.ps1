# Phase 5.4 + 5.4b - Sequential hardware sweep on ibm_boston.
# 5 folds * {L1 T-REx+DD, L2 +ZNE} = 10 submissions.
# Pre-flight check inside run_hardware_pilot.py aborts if any single
# submission would push cumulative QPU past 150 min cap.

$ErrorActionPreference = "Continue"
$PythonExe = ".venv\Scripts\python.exe"
$Script = "scripts\run_hardware_pilot.py"

Write-Host "=== Phase 5.4 - L1 (T-REx + DD) sweep across 5 folds ==="
foreach ($fold in 0..4) {
    Write-Host ""
    Write-Host "--- fold $fold L1 ---"
    & $PythonExe $Script --dataset WESAD --fold-idx $fold `
        --backend ibm_boston --hardware `
        --shots 8192 --resilience-level 1 `
        --planned-qpu-min 3
    if (-not $?) {
        Write-Host "Fold $fold L1 failed (exit code $LASTEXITCODE); continuing."
    }
}

Write-Host ""
Write-Host "=== Phase 5.4b - L2 (+ ZNE) sweep across 5 folds ==="
foreach ($fold in 0..4) {
    Write-Host ""
    Write-Host "--- fold $fold L2 ---"
    & $PythonExe $Script --dataset WESAD --fold-idx $fold `
        --backend ibm_boston --hardware `
        --shots 8192 --resilience-level 2 `
        --planned-qpu-min 8
    if (-not $?) {
        Write-Host "Fold $fold L2 failed (exit code $LASTEXITCODE); continuing."
    }
}

Write-Host ""
Write-Host "=== Sweep complete ==="
& $PythonExe scripts\ibm_usage.py --path results\ibm_usage.csv --summary
