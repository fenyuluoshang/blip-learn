param(
    [string]$OutputDir = 'runs/blip-tuning-20261022'
)

$ErrorActionPreference = 'Stop'
$tuningRoot = $PSScriptRoot
$tuningPython = Join-Path $tuningRoot '.venv-blip\Scripts\python.exe'
$tuningScript = Join-Path $tuningRoot 'run_blip_tuning.py'
$tuningOutput = [System.IO.Path]::GetFullPath((Join-Path $tuningRoot $OutputDir))
$tuningStatusPath = Join-Path $tuningOutput 'status.json'
if (Test-Path -LiteralPath $tuningStatusPath) {
    $tuningStatus = Get-Content -LiteralPath $tuningStatusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $activeTuning = Get-CimInstance Win32_Process -Filter "ProcessId = $($tuningStatus.pid)" -ErrorAction SilentlyContinue
    if ($activeTuning -and $activeTuning.CommandLine -like '*run_blip_tuning.py*') {
        throw "Tuning is already running with PID $($tuningStatus.pid)."
    }
}
$tuningLogStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$tuningStdout = "$tuningOutput.$tuningLogStamp.log"
$tuningStderr = "$tuningOutput.$tuningLogStamp.stderr.log"
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$tuningArguments = @('-u', ('"' + $tuningScript + '"'), '--output', ('"' + $tuningOutput + '"'))
$tuningProcess = Start-Process -FilePath $tuningPython -ArgumentList $tuningArguments -WorkingDirectory $tuningRoot `
    -WindowStyle Hidden -RedirectStandardOutput $tuningStdout -RedirectStandardError $tuningStderr -PassThru
[PSCustomObject]@{ProcessId=$tuningProcess.Id; Output=$tuningOutput; Log=$tuningStdout; ErrorLog=$tuningStderr} | ConvertTo-Json
