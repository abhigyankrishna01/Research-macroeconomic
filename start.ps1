$ErrorActionPreference = "Stop"

Write-Host "== Financial Fraud Detection: START ==" -ForegroundColor Cyan

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (!(Test-Path "$root\venv\Scripts\python.exe")) {
    Write-Host "Virtual environment not found. Creating venv and installing dependencies..." -ForegroundColor Yellow
    python -m venv venv
    & "$root\venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt"
}

if (!(Test-Path "$root\frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location "$root\frontend"
    npm install
    Pop-Location
}

Write-Host "Starting Docker infrastructure..." -ForegroundColor Green
docker compose up -d

Write-Host "Applying DB migration (is_fraud column)..." -ForegroundColor Green
Start-Sleep -Seconds 5
docker exec -i fraud_db psql -U admin -d fraud_db -c "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS is_fraud BOOLEAN DEFAULT NULL;" | Out-Null

Write-Host "Launching MLflow..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; .\venv\Scripts\Activate.ps1; python -m mlflow server --host 0.0.0.0 --port 5000"

Write-Host "Launching FastAPI consumer..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\consumer'; ..\venv\Scripts\Activate.ps1; python -m uvicorn main:app --host 0.0.0.0 --port 8001"

Write-Host "Launching frontend..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev -- --port 3002"

Write-Host "Launching producer..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; .\venv\Scripts\Activate.ps1; python producer.py"

Write-Host ""
Write-Host "Stack started. Open:" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3002"
Write-Host "API:      http://127.0.0.1:8001"
Write-Host "MLflow:   http://localhost:5000"
