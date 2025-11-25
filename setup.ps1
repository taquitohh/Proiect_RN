# Script de setup pentru Windows PowerShell
# ==========================================

Write-Host "=== Setup Proiect Retele Neuronale ===" -ForegroundColor Cyan
Write-Host ""

# Verificare Python
Write-Host "Verificare Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Python instalat: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "Python NU este instalat!" -ForegroundColor Red
    Write-Host "Descarcati Python de la: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Important: Bifati 'Add Python to PATH' la instalare!" -ForegroundColor Yellow
    exit 1
}

# Verificare Node.js
Write-Host ""
Write-Host "Verificare Node.js..." -ForegroundColor Yellow
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Node.js instalat: $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "Node.js NU este instalat!" -ForegroundColor Red
    Write-Host "Descarcati Node.js de la: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Creare mediu virtual Python
Write-Host ""
Write-Host "Creare mediu virtual Python..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "Mediu virtual creat!" -ForegroundColor Green
}

# Activare si instalare dependente Python
Write-Host ""
Write-Host "Activare mediu virtual si instalare dependente Python..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Write-Host "Dependente Python instalate!" -ForegroundColor Green

# Instalare dependente Frontend
Write-Host ""
Write-Host "Instalare dependente Frontend..." -ForegroundColor Yellow
Set-Location frontend
npm install
Set-Location ..
Write-Host "Dependente Frontend instalate!" -ForegroundColor Green

Write-Host ""
Write-Host "=== Setup Complet! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Pentru a rula proiectul:" -ForegroundColor Cyan
Write-Host "1. Backend: cd src; python -m uvicorn api:app --reload --port 8000" -ForegroundColor White
Write-Host "2. Frontend: cd frontend; npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "Apoi deschideti: http://localhost:3000" -ForegroundColor Yellow
