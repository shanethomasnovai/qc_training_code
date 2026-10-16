# Run from any directory; inputs are resolved relative to this script.
[CmdletBinding()]
param(
    [ValidateSet('previousStudent', 'shane', 'soft')]
    [string]$Grader = 'shane',
    [string]$RatingsCsvPath,
    [string[]]$GraderColumns,
    [string]$RatingsSheet = 'Grades',
    [switch]$ExcludeKnownUnmatched,
    [switch]$CheckOnly,
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
$runName = $Grader + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$trainingArguments = @(
    $trainingPath,
    '--dataset_csv_path', $manifestPath,
    '--grader', $Grader,
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
if ($Grader -eq 'soft') {
    if (-not $RatingsCsvPath) { throw '-Grader soft requires -RatingsCsvPath' }
    $trainingArguments += @('--ratings_csv_path', $RatingsCsvPath)
    $trainingArguments += @('--ratings_sheet', $RatingsSheet)
    if ($ExcludeKnownUnmatched) { $trainingArguments += '--exclude_known_unmatched' }
    if ($GraderColumns) { $trainingArguments += @('--grader_columns') + $GraderColumns }
    if ($ClassWeightsFrom) { $trainingArguments += @('--class_weights_from', $ClassWeightsFrom) }
    if ($WeightReferenceGrader) { $trainingArguments += @('--weight_reference_grader', $WeightReferenceGrader) }
} elseif ($RatingsCsvPath -or $GraderColumns -or $ClassWeightsFrom -or $WeightReferenceGrader) {
    throw 'Ratings options require -Grader soft'
}
if ($CheckOnly) { $trainingArguments += '--check_only' }

Write-Host "Training $Grader labels for up to $Epochs epochs (early stopping patience: $Patience)."
Write-Host "Shared manifest: $manifestPath"
Write-Host "Output: $(Join-Path $runsPath $runName)"
Write-Host 'VISUS-NHP remains held out; no test evaluation is performed.'

& $pythonPath @trainingArguments
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE."
}

if ($CheckOnly) { Write-Host 'Input checks complete; no run created.' }
else { Write-Host "Finished. Best checkpoint: $(Join-Path $runsPath "$runName\model.pth")" }
