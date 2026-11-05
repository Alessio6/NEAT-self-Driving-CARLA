# Trains with the "carla" conda env's Python, appends the output to
# logs\training_log.txt (which progress.py reads) and, if training crashes
# (e.g. a CARLA or GPU driver hiccup), waits for CARLA and resumes from the
# latest checkpoint.
#
#   .\run_training.ps1 -Generations 100 -Npcs 0 -NoRender
#   .\run_training.ps1 -Resume checkpoints\neat-checkpoint-40 -Generations 60 -Npcs 30 -NoRender
#   .\run_training.ps1 -Viz -VizSave viz      # live network window + a picture per generation
#
# Run it with:  powershell -ExecutionPolicy Bypass -File .\run_training.ps1 ...
param(
    [string]$Resume = "",
    [int]$Generations = 50,
    [int]$Npcs = 30,
    [switch]$NoRender,
    [switch]$Viz,
    [string]$VizSave = "",
    [int]$MaxRestarts = 20
)

$python = "$env:USERPROFILE\anaconda3\envs\carla\python.exe"
$checkpointDir = "checkpoints"
$log = "logs\training_log.txt"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Get-CheckpointNumber([string]$path) {
    if ($path -match 'neat-checkpoint-(\d+)$') { return [int]$Matches[1] }
    return 0
}

function Get-LatestCheckpoint {
    Get-ChildItem $checkpointDir -Filter "neat-checkpoint-*" -ErrorAction SilentlyContinue |
        Sort-Object { Get-CheckpointNumber $_.Name } | Select-Object -Last 1
}

function Wait-ForCarla([int]$timeoutSeconds = 600) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $client = New-Object Net.Sockets.TcpClient
        try { $client.Connect("localhost", 2000); $client.Close(); return $true } catch { }
        $client.Close()
        Start-Sleep 10
    }
    return $false
}

# Training stops at this generation number, however many restarts it takes.
$targetGeneration = (Get-CheckpointNumber $Resume) + $Generations
$restarts = 0

while ($true) {
    $trainArgs = @("-u", "train.py", "--npcs", $Npcs,
                   "--generations", ($targetGeneration - (Get-CheckpointNumber $Resume)))
    if ($NoRender) { $trainArgs += "--no-render" }
    if ($Viz) { $trainArgs += "--viz" }
    if ($VizSave) { $trainArgs += @("--viz-save", $VizSave) }
    $when = Get-Date -f 'yyyy-MM-dd HH:mm'
    if ($Resume) {
        $trainArgs += @("--resume", $Resume)
        $header = "RESUMED $when from $(Split-Path $Resume -Leaf) with $Npcs traffic cars"
    } else {
        $header = "STARTED $when with $Npcs traffic cars"
    }
    "`n===== $header =====" | Out-File -Append -Encoding utf8 $log

    & $python @trainArgs 2>&1 | Where-Object { $_ -notmatch '^INFO' } | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) { break }

    $restarts++
    $latest = Get-LatestCheckpoint
    if ($latest -and (Get-CheckpointNumber $latest.Name) -ge $targetGeneration) { break }
    if ($restarts -gt $MaxRestarts) {
        Write-Host "Training crashed $restarts times - giving up. Resume manually from $($latest.Name)."
        exit 1
    }
    Write-Host "`nTraining crashed (exit code $exitCode). Restart $restarts of $MaxRestarts - waiting for CARLA..."
    if (-not (Wait-ForCarla)) {
        Write-Host "CARLA isn't responding on port 2000 - restart it, then resume from $($latest.Name)."
        exit 1
    }
    Start-Sleep 15  # give CARLA a moment to settle
    if ($latest) { $Resume = Join-Path $checkpointDir $latest.Name }
}
