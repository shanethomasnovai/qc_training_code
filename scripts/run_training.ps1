# Run from any directory; inputs are resolved relative to this script.
[CmdletBinding()]
param(
    [ValidateSet('soft')]
    [string]$Grader = 'soft',
    [string]$RatingsCsvPath,
    [string[]]$GraderColumns,
    [string]$RatingsSheet = 'Grades',
    [switch]$ExcludeKnownUnmatched,
    [switch]$CheckOnly,
    [ValidateSet('cross_entropy', 'continuous_cdw_ce')]
    [string]$Loss = 'cross_entropy',
    [double]$Alpha = 1.0,
    [string]$ClassWeightsFrom,
    [ValidateSet('previousStudent', 'shane')]
    [string]$WeightReferenceGrader,
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
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$trainingPath = Join-Path $projectRoot 'autoqc\training.py'
$manifestPath = Join-Path $projectRoot 'prepared_visus_nhp\manifest.csv'
$runsPath = Join-Path $projectRoot 'runs'

foreach ($requiredPath in @($pythonPath, $trainingPath, $manifestPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file missing: $requiredPath"
    }
}

# Unique names preserve earlier runs, including one-epoch checks.
$runPrefix = if ($Loss -eq 'continuous_cdw_ce') { 'continuous_cdw_ce' } else { $Grader }
$runName = $runPrefix + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$trainingArguments = @(
    '-m', 'autoqc.training',
    '--dataset_csv_path', $manifestPath,
    '--grader', $Grader,
    '--loss', $Loss,
    '--alpha', $Alpha.ToString([System.Globalization.CultureInfo]::InvariantCulture),
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
if (-not $RatingsCsvPath) { throw 'Combined-grader training requires -RatingsCsvPath' }
$RatingsCsvPath = (Resolve-Path -LiteralPath $RatingsCsvPath -ErrorAction Stop).Path
$trainingArguments += @('--ratings_csv_path', $RatingsCsvPath)
$trainingArguments += @('--ratings_sheet', $RatingsSheet)
if ($ExcludeKnownUnmatched) { $trainingArguments += '--exclude_known_unmatched' }
if ($GraderColumns) { $trainingArguments += @('--grader_columns') + $GraderColumns }
if ($ClassWeightsFrom) {
    $ClassWeightsFrom = (Resolve-Path -LiteralPath $ClassWeightsFrom -ErrorAction Stop).Path
    $trainingArguments += @('--class_weights_from', $ClassWeightsFrom)
}
if ($WeightReferenceGrader) { $trainingArguments += @('--weight_reference_grader', $WeightReferenceGrader) }
if ($CheckOnly) { $trainingArguments += '--check_only' }

Write-Host "Training combined grader ratings with $Loss for up to $Epochs epochs (early stopping patience: $Patience)."
Write-Host "Shared manifest: $manifestPath"
Write-Host "Output: $(Join-Path $runsPath $runName)"
Write-Host 'VISUS-NHP remains held out; no test evaluation is performed.'

Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath @trainingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

if ($CheckOnly) { Write-Host 'Input checks complete; no run created.' }
else { Write-Host "Finished. Best checkpoint: $(Join-Path $runsPath "$runName\model.pth")" }
