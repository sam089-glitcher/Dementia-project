@echo off
echo Starting Streamlit App...
cd /d "%~dp0"
streamlit run src/api/app.py --server.port 8501 --server.address localhost
pause

