$ErrorActionPreference = "SilentlyContinue"

Write-Host "== Financial Fraud Detection: STOP ==" -ForegroundColor Cyan

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Stopping app processes (producer, uvicorn, mlflow, next)..." -ForegroundColor Green

$patterns = @(
    "producer.py",
    "uvicorn main:app",
    "mlflow server",
    "next dev"
)

foreach ($pattern in $patterns) {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*$pattern*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

Write-Host "Stopping Docker compose services..." -ForegroundColor Green
docker compose down

Write-Host "Stopping host Redis container..." -ForegroundColor Green
docker rm -f fraud_redis_host | Out-Null

Write-Host "All services stopped." -ForegroundColor Cyan
