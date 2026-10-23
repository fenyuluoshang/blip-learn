param(
    [string]$OutputDir = 'runs/blip-full-random80-20261022',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$trainingRoot = $PSScriptRoot
$trainingPython = Join-Path $trainingRoot '.venv-blip\Scripts\python.exe'
$trainingScript = Join-Path $trainingRoot 'train_blip.py'
$trainingOutput = [System.IO.Path]::GetFullPath((Join-Path $trainingRoot $OutputDir))
if ((Test-Path -LiteralPath $trainingOutput) -and -not $Resume) {
    throw "Experiment directory already exists. Use -Resume or choose a new -OutputDir: $trainingOutput"
}
if ($Resume -and -not (Test-Path -LiteralPath (Join-Path $trainingOutput 'last\training_state.pt'))) {
    throw 'No resumable last checkpoint exists.'
}
$trainingStatusPath = Join-Path $trainingOutput 'status.json'
if (Test-Path -LiteralPath $trainingStatusPath) {
    $trainingStatus = Get-Content -LiteralPath $trainingStatusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $activeTraining = Get-CimInstance Win32_Process -Filter "ProcessId = $($trainingStatus.pid)" -ErrorAction SilentlyContinue
    if ($activeTraining -and $activeTraining.CommandLine -like '*train_blip.py*') {
        throw "Training is already running with PID $($trainingStatus.pid)."
    }
}
$trainingLogFolder = Split-Path -Parent $trainingOutput
New-Item -ItemType Directory -Path $trainingLogFolder -Force | Out-Null
$trainingLogStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$trainingStdout = "$trainingOutput.$trainingLogStamp.log"
$trainingStderr = "$trainingOutput.$trainingLogStamp.stderr.log"
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$trainingArguments = @('-u', ('"' + $trainingScript + '"'), '--output', ('"' + $trainingOutput + '"'),
    '--batch-size', '2', '--gradient-accumulation', '8', '--eval-batch-size', '4',
    '--cache-batch-size', '4', '--workers', '2', '--epochs', '10', '--patience', '3',
    '--seed', '111', '--validation-fraction', '0.2', '--checkpoint-every', '250', '--log-every', '25')
if ($Resume) { $trainingArguments += '--resume' }
$trainingProcess = Start-Process -FilePath $trainingPython -ArgumentList $trainingArguments -WorkingDirectory $trainingRoot `
    -WindowStyle Hidden -RedirectStandardOutput $trainingStdout -RedirectStandardError $trainingStderr -PassThru
[PSCustomObject]@{
    ProcessId = $trainingProcess.Id
    Output = $trainingOutput
    Log = $trainingStdout
    ErrorLog = $trainingStderr
    Resume = $Resume.IsPresent
} | ConvertTo-Json
