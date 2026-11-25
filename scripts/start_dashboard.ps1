# DeepTrace AI — Quick Launch: Streamlit Dashboard
# Run from the project root: .\scripts\start_dashboard.ps1

$python = "C:\Users\ayesh\AppData\Local\Python\bin\python.exe"
$root = Split-Path $MyInvocation.MyCommand.Path -Parent | Split-Path -Parent

Set-Location $root
Write-Host "🛡️  Starting DeepTrace AI Streamlit Dashboard..."
Write-Host "   Dashboard URL: http://localhost:8501"
Write-Host ""
& $python -m streamlit run app_streamlit.py
