# Runs train.py in the "carla" conda env and appends its output to training_log.txt
# (which progress.py reads).
#   .\run_training.ps1 -Generations 50 -Npcs 0
#   .\run_training.ps1 -Resume checkpoints\neat-checkpoint-52 -Generations 28 -Npcs 30
#   .\run_training.ps1 -Viz -VizSave viz      # live network window + a picture per generation
param(
    [int]$Version = 2,
    [string]$Resume = "",
    [int]$Generations = 50,
    [int]$Npcs = 30,
    [switch]$Viz,
    [string]$VizSave = "",
    [switch]$NoRender
)

$python = "$env:USERPROFILE\anaconda3\envs\carla\python.exe"
$trainArgs = @("-u", "train.py", "--version", $Version, "--generations", $Generations, "--npcs", $Npcs)
if ($NoRender) { $trainArgs += "--no-render" }
if ($Viz) { $trainArgs += "--viz" }
if ($VizSave) { $trainArgs += @("--viz-save", $VizSave) }
$header = "`n===== STARTED v$Version $(Get-Date -f 'yyyy-MM-dd HH:mm') with $Npcs traffic cars"
if ($Resume) {
    $trainArgs += @("--resume", $Resume)
    $header = "`n===== RESUMED v$Version $(Get-Date -f 'yyyy-MM-dd HH:mm') from $(Split-Path $Resume -Leaf) with $Npcs traffic cars"
}

Set-Location $PSScriptRoot
"$header =====" | Out-File -Append -Encoding utf8 training_log.txt
& $python @trainArgs 2>&1 | Where-Object { $_ -notmatch '^INFO' } | Tee-Object -FilePath training_log.txt -Append
