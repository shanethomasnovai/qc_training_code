# Run from any directory; inputs are resolved relative to this script.
[CmdletBinding()]
param(
    [ValidateRange(1, 100000)]
    [int]$Epochs = 50,
    [ValidateRange(1, 100000)]
    [int]$BatchSize = 16,
    [ValidateRange(0.000000001, 1.0)]
    [double]$LearningRate = 0.001,
    [ValidateRange(1, 100000)]
    [int]$Patience = 10,
    [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$trainingPath = Join-Path $PSScriptRoot 'training.py'
$manifestPath = Join-Path $PSScriptRoot 'prepared_visus_nhp\manifest.csv'
$runsPath = Join-Path $PSScriptRoot 'runs'

foreach ($requiredPath in @($pythonPath, $trainingPath, $manifestPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file missing: $requiredPath"
    }
}

# Unique names preserve earlier runs, including one-epoch checks.
$runName = 'shane_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$trainingArguments = @(
    $trainingPath,
    '--dataset_csv_path', $manifestPath,
    '--grader', 'shane',
    '--output_dir', $runsPath,
    '--model_name', $runName,
    '--epochs', $Epochs.ToString(),
    '--batch_size', $BatchSize.ToString(),
    '--learning_rate', $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--optimizer', 'AdamW',
    '--patience', $Patience.ToString(),
    '--seed', $Seed.ToString(),
    '--reduce_lr'
)

Write-Host "Training Shane labels for up to $Epochs epochs (early stopping patience: $Patience)."
Write-Host "Shared manifest: $manifestPath"
Write-Host "Output: $(Join-Path $runsPath $runName)"
Write-Host 'VISUS-NHP remains held out; no test evaluation is performed.'

& $pythonPath @trainingArguments
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE."
}

Write-Host "Finished. Best checkpoint: $(Join-Path $runsPath "$runName\model.pth")"
