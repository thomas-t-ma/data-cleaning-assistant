$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required but was not found."
}

if (-not (Test-Path ".git")) {
    throw "Run this from a Git clone of the project."
}

Write-Host "==> Syncing source from GitHub"
git fetch origin main
if ($LASTEXITCODE -ne 0) { throw "git fetch failed." }
git rebase --autostash origin/main
if ($LASTEXITCODE -ne 0) { throw "git rebase failed. Resolve the Git conflict, then run sync.ps1 again." }

$PythonExe = $null
$PythonPrefix = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
    $PythonPrefix = @("-3")
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
}
elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonExe = "python3"
}
else {
    throw "Python 3.11+ was not found. Install Python, then run this script again."
}

$Version = & $PythonExe @PythonPrefix -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
& $PythonExe @PythonPrefix -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'
if ($LASTEXITCODE -ne 0) {
    throw "Python $Version found; Python 3.11+ is required."
}

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "==> Creating .venv with Python $Version"
    & $PythonExe @PythonPrefix -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv." }
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
}

$DepsHash = (Get-FileHash "pyproject.toml" -Algorithm SHA256).Hash.ToLower()
$StampFile = Join-Path $PSScriptRoot ".venv\.data-cleaning-assistant-pyproject.sha256"
$OldHash = if (Test-Path $StampFile) { (Get-Content $StampFile -Raw).Trim() } else { "" }

if ($DepsHash -ne $OldHash) {
    Write-Host "==> Installing/updating project dependencies"
    & $VenvPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    Set-Content -Path $StampFile -Value $DepsHash
}
else {
    Write-Host "==> Dependencies unchanged"
}

if ($args -notcontains "--no-test") {
    Write-Host "==> Running tests"
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
}

Write-Host ""
Write-Host "Ready. Python: $VenvPython"
Write-Host "CLI: $PSScriptRoot\.venv\Scripts\data-cleaner.exe"
