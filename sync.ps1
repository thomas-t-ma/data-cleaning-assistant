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

function Rebuild-Venv {
    Write-Host "==> Rebuilding .venv with Python $Version"
    if (Test-Path ".venv") {
        Remove-Item -Recurse -Force ".venv"
    }

    & $PythonExe @PythonPrefix -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Python could not create a virtual environment. Install a full Python 3.11+ distribution and retry."
    }

    & $VenvPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> pip is missing from the new environment; bootstrapping it"
        & $VenvPython -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "This Python installation does not provide pip/ensurepip. Install a full Python distribution and retry."
        }
    }

    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
}

if (-not (Test-Path $VenvPython)) {
    Rebuild-Venv
}
else {
    & $VenvPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> Existing .venv has Python but no working pip"
        Rebuild-Venv
    }
}

$DepsHash = (Get-FileHash "pyproject.toml" -Algorithm SHA256).Hash.ToLower()
$StampFile = Join-Path $PSScriptRoot ".venv\.data-cleaning-assistant-pyproject.sha256"
$OldHash = if (Test-Path $StampFile) { (Get-Content $StampFile -Raw).Trim() } else { "" }

$EnvOk = $true
& $VenvPython -c 'import data_cleaning_assistant, openpyxl, pandas, pyarrow, pydantic, typer, rich, dateutil, yaml' 2>$null
if ($LASTEXITCODE -ne 0) { $EnvOk = $false }
& $VenvPython -m pip check *> $null
if ($LASTEXITCODE -ne 0) { $EnvOk = $false }

if (($DepsHash -ne $OldHash) -or (-not $EnvOk)) {
    if ((-not $EnvOk) -and ($DepsHash -eq $OldHash)) {
        Write-Host "==> Environment is incomplete; repairing dependencies"
    }
    else {
        Write-Host "==> Installing/updating project dependencies"
    }
    & $VenvPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    & $VenvPython -c 'import data_cleaning_assistant, openpyxl, pandas, pyarrow, pydantic, typer, rich, dateutil, yaml'
    if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed." }
    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Installed dependency set is inconsistent." }
    Set-Content -Path $StampFile -Value $DepsHash
}
else {
    Write-Host "==> Dependencies verified"
}

if ($args -notcontains "--no-test") {
    Write-Host "==> Running tests"
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
}

Write-Host ""
Write-Host "Ready. Python: $VenvPython"
Write-Host "CLI: $PSScriptRoot\.venv\Scripts\data-cleaner.exe"
