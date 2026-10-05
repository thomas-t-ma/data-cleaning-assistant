param(
    [switch]$BuildInstaller
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ($env:OS -ne "Windows_NT") {
    throw "The Windows desktop build must run on Windows."
}

Push-Location $Root
try {
    $Dist = Join-Path $Root "dist"
    $Build = Join-Path $Root "build\windows-desktop"
    $Spec = Join-Path $Build "spec"

    Remove-Item $Build -Recurse -Force -ErrorAction SilentlyContinue
    New-Item $Build -ItemType Directory -Force | Out-Null
    New-Item $Spec -ItemType Directory -Force | Out-Null
    New-Item $Dist -ItemType Directory -Force | Out-Null

    $PyInstallerArgs = @(
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name", "Data Cleaning Assistant",
        "--collect-all", "webview",
        "--hidden-import", "data_cleaning_assistant.web_app",
        "--distpath", $Dist,
        "--workpath", (Join-Path $Build "pyinstaller"),
        "--specpath", $Spec,
        (Join-Path $Root "packaging\windows\desktop_entry.py")
    )
    python -m PyInstaller @PyInstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed."
    }

    $Exe = Join-Path $Dist "Data Cleaning Assistant\Data Cleaning Assistant.exe"
    if (-not (Test-Path $Exe)) {
        throw "Packaged executable was not created at $Exe"
    }

    & $Exe --self-test
    if ($LASTEXITCODE -ne 0) {
        throw "Packaged desktop self-test failed with exit code $LASTEXITCODE."
    }

    $Version = (
        python -c "import data_cleaning_assistant; print(data_cleaning_assistant.__version__)"
    ).Trim()
    if (-not $Version) {
        throw "Could not determine package version."
    }

    $PortableZip = Join-Path $Dist "DataCleaningAssistant-$Version-Windows-x64-portable.zip"
    Remove-Item $PortableZip -Force -ErrorAction SilentlyContinue
    Compress-Archive -Path (Join-Path $Dist "Data Cleaning Assistant\*") -DestinationPath $PortableZip -CompressionLevel Optimal

    $Artifacts = @($PortableZip)

    if ($BuildInstaller) {
        $Candidates = @(
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        )
        $ISCC = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $ISCC) {
            $Command = Get-Command iscc.exe -ErrorAction SilentlyContinue
            if ($Command) {
                $ISCC = $Command.Source
            }
        }
        if (-not $ISCC) {
            throw "Inno Setup 6 (ISCC.exe) is required when -BuildInstaller is used."
        }

        & $ISCC "/DMyAppVersion=$Version" (Join-Path $Root "packaging\windows\DataCleaner.iss")
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed."
        }

        $Installer = Join-Path $Dist "installer\DataCleaningAssistant-$Version-Windows-x64-Setup.exe"
        if (-not (Test-Path $Installer)) {
            throw "Installer was not created at $Installer"
        }
        $Artifacts += $Installer
    }

    $ChecksumPath = Join-Path $Dist "SHA256SUMS.txt"
    $Lines = foreach ($Artifact in $Artifacts) {
        $Hash = Get-FileHash $Artifact -Algorithm SHA256
        "$($Hash.Hash.ToLowerInvariant())  $([IO.Path]::GetFileName($Artifact))"
    }
    Set-Content -Path $ChecksumPath -Value $Lines -Encoding ascii

    Write-Host "Windows desktop build complete."
    Write-Host "Packaged executable: $Exe"
    Write-Host "Portable archive: $PortableZip"
    if ($BuildInstaller) {
        Write-Host "Installer: $Installer"
    }
    Write-Host "Checksums: $ChecksumPath"
}
finally {
    Pop-Location
}
