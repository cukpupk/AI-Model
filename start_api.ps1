$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot"

$pythonExe = $null
$pythonPrefixArgs = @()
$candidates = @(
    "python",
    "py",
    "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe",
    "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
)

foreach ($c in $candidates) {
    try {
        if ($c -eq "py") {
            & py -V | Out-Null
            $pythonExe = "py"
            $pythonPrefixArgs = @("-3")
            break
        }

        if ($c -eq "python") {
            & python --version | Out-Null
            $pythonExe = "python"
            $pythonPrefixArgs = @()
            break
        }

        if (Test-Path $c) {
            & $c --version | Out-Null
            $pythonExe = $c
            $pythonPrefixArgs = @()
            break
        }
    }
    catch {
        continue
    }
}

if (-not $pythonExe) {
    Write-Host "Python not found. Please install Python 3.11+ and add to PATH." -ForegroundColor Red
    exit 1
}

Write-Host "Starting API..." -ForegroundColor Green
Write-Host "Swagger UI: http://127.0.0.1:8000/swagger" -ForegroundColor Cyan

$env:HF_HOME = Join-Path $PSScriptRoot "models\hf_cache"
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

$apiArgs = @("-m", "uvicorn", "rag_service.api:app", "--host", "127.0.0.1", "--port", "8000")
& $pythonExe @pythonPrefixArgs @apiArgs
