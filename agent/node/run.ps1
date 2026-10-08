# Start the support node (Windows).
#
#   .\run.ps1                 first run: creates .venv, installs requirements
#   .\run.ps1 -SkipInstall    later runs: straight to the server
#   .\run.ps1 -Config other.yaml
#
# Host/port come from config.yaml. Once it is up, check it from the main
# machine's browser: http://<this-machine-ip>:8010/health

param(
    [switch]$SkipInstall,
    [string]$Config = "config.yaml"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating .venv..." -ForegroundColor Cyan
    python -m venv .venv
    $SkipInstall = $false
}

if (-not $SkipInstall) {
    Write-Host "Installing requirements (this takes a while the first time)..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip
    # Install torch separately first if you need a specific CUDA build:
    #   & $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu126
    & $venvPython -m pip install -r requirements.txt
}

$env:NODE_CONFIG = $Config

$settings = & $venvPython -c @"
import os, yaml
config = yaml.safe_load(open(os.environ['NODE_CONFIG'], encoding='utf-8'))
print(config.get('host', '0.0.0.0'))
print(config.get('port', 8010))
"@
$nodeHost, $nodePort = $settings -split "`n" | ForEach-Object { $_.Trim() }

Write-Host ""
Write-Host "Support node starting on http://${nodeHost}:${nodePort}" -ForegroundColor Green
foreach ($address in (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" })) {
    Write-Host ("  reachable at http://{0}:{1}  <- use this in the backend's remote_services" -f $address.IPAddress, $nodePort) -ForegroundColor Yellow
}
Write-Host ""

& $venvPython -m uvicorn server:app --host $nodeHost --port $nodePort
