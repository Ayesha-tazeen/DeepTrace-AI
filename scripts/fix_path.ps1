# DeepTrace AI - Fix PATH for streamlit/uvicorn commands
# Run this once in PowerShell to make bare 'streamlit' and 'uvicorn' work

$scriptsPath = "C:\Users\ayesh\AppData\Local\Python\pythoncore-3.15-64\Scripts"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($currentPath -notlike "*$scriptsPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$scriptsPath", "User")
    Write-Host "✅ Added Python Scripts to user PATH: $scriptsPath"
    Write-Host "   Please restart PowerShell for the change to take effect."
} else {
    Write-Host "✅ Python Scripts already in PATH — no change needed."
}

Write-Host ""
Write-Host "After restarting PowerShell, you can run:"
Write-Host "  streamlit run app_streamlit.py"
Write-Host "  python run_api.py"
Write-Host ""
Write-Host "Or use these commands immediately without restarting:"
Write-Host "  `$env:PATH += ';$scriptsPath'"
Write-Host "  streamlit run app_streamlit.py"
