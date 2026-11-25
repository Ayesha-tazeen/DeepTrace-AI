# Run from the DeepTrace-AI folder in VS Code's PowerShell terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_windows.ps1
$ErrorActionPreference = "Stop"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
