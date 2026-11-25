@echo off
title DeepTrace AI - Streamlit Dashboard
echo.
echo  =========================================
echo   DeepTrace AI - Cybersecurity Dashboard
echo  =========================================
echo.
echo  Starting Streamlit at http://localhost:8501
echo.

cd /d "c:\Users\ayesh\OneDrive\Desktop\DeepTrace-AI"

REM Try python -m streamlit (always works regardless of PATH)
"C:\Users\ayesh\AppData\Local\Python\bin\python.exe" -m streamlit run app_streamlit.py

pause
