$ErrorActionPreference = "Stop"

$python = $env:PYTHON_BIN
if (-not $python) { $python = "python" }

if (-not (Test-Path ".venv")) {
  Write-Host "[run.ps1] Creating venv..."
  & $python -m venv .venv
}

Write-Host "[run.ps1] Activating venv..."
& .\.venv\Scripts\Activate.ps1

Write-Host "[run.ps1] Upgrading pip..."
python -m pip install -U pip

Write-Host "[run.ps1] Installing requirements..."
pip install -r requirements.txt

Write-Host "[run.ps1] Running a short training (autoreg)..."
python -m qsimlm.train_2q_special --model autoreg --n_train 20000 --n_test 2000 --epochs 8

Write-Host "[run.ps1] Done."
