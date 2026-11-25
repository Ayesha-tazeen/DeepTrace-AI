# DeepTrace AI — Quick Launch: FastAPI Backend
# Run from the project root: .\scripts\start_api.ps1

$python = "C:\Users\ayesh\AppData\Local\Python\bin\python.exe"
$root = Split-Path $MyInvocation.MyCommand.Path -Parent | Split-Path -Parent

Set-Location $root
Write-Host "🛡️  Starting DeepTrace AI FastAPI Backend..."
Write-Host "   API URL:   http://127.0.0.1:8000"
Write-Host "   Docs URL:  http://127.0.0.1:8000/docs"
Write-Host ""
& $python run_api.py
