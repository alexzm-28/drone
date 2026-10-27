$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Run setup_sim.ps1 first.'
}
& .\.venv\Scripts\python.exe -m drone_sim @args
exit $LASTEXITCODE
