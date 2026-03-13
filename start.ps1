param(
    [int]$Port = 8501,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StreamlitArgs
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$app = Join-Path $repoRoot "app.py"

if (-not (Test-Path $python)) {
    Write-Error "Missing virtual environment interpreter at $python. Create .venv and install requirements first."
    exit 1
}

if (-not (Test-Path $app)) {
    Write-Error "Could not find app.py at $app."
    exit 1
}

$arguments = @(
    "-m",
    "streamlit",
    "run",
    $app,
    "--server.port",
    $Port
) + $StreamlitArgs

Push-Location $repoRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
