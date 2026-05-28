param(
    [Parameter(Mandatory=$true)]
    [string]$Plan,

    [string]$FalcorRoot = "T:\Falcor",
    [string]$SourceRoot = "T:\Falcor\neuralpvs_export_test\datasets",
    [string]$OutputRoot = "T:\Falcor\neuralpvs_export_test\datasets",
    [int]$ChunkSize = 10,
    [string]$ChunkDir = "",
    [string]$ExporterExe = ""
)

$ErrorActionPreference = "Stop"

if (-not $ExporterExe) {
    $ExporterExe = Join-Path $FalcorRoot "build\windows-vs2022\bin\Release\NeuralPVSExporter.exe"
}

if (-not (Test-Path $ExporterExe)) {
    throw "NeuralPVSExporter.exe not found: $ExporterExe"
}

if (-not (Test-Path $Plan)) {
    throw "Batch plan not found: $Plan"
}

if (-not $ChunkDir) {
    $planDir = Split-Path -Parent $Plan
    $planName = [System.IO.Path]::GetFileNameWithoutExtension($Plan)
    $ChunkDir = Join-Path $planDir "${planName}_chunks_auto"
}

New-Item -ItemType Directory -Force $ChunkDir | Out-Null

Write-Host "Plan       : $Plan"
Write-Host "SourceRoot : $SourceRoot"
Write-Host "OutputRoot : $OutputRoot"
Write-Host "ChunkSize  : $ChunkSize"
Write-Host "ChunkDir   : $ChunkDir"
Write-Host "Exporter   : $ExporterExe"
Write-Host ""

$rows = Import-Csv $Plan

$status = foreach ($r in $rows) {
    $ds = Join-Path $SourceRoot $r.dataset_name
    $gv = (Get-ChildItem "$ds\gv\*_gv.bin.gz" -ErrorAction SilentlyContinue).Count
    $pvv = (Get-ChildItem "$ds\pvv\*_pvv.bin.gz" -ErrorAction SilentlyContinue).Count

    [PSCustomObject]@{
        part = [int]$r.part
        dataset_name = $r.dataset_name
        seed = $r.seed
        samples = [int]$r.samples
        scene_path = $r.scene_path
        camera_path_csv = $r.camera_path_csv
        manifest = $r.manifest
        command = $r.command
        gv = $gv
        pvv = $pvv
        complete = ($gv -eq [int]$r.samples -and $pvv -eq [int]$r.samples)
    }
}

$remaining = @($status | Where-Object { -not $_.complete } | Sort-Object part)

if ($remaining.Count -eq 0) {
    Write-Host "All batch parts are already complete."
    exit 0
}

Write-Host "Remaining parts: $($remaining.Count)"
Write-Host "First remaining: part $($remaining[0].part) / $($remaining[0].dataset_name)"
Write-Host ""

$chunkFiles = @()
for ($i = 0; $i -lt $remaining.Count; $i += $ChunkSize) {
    $end = [Math]::Min($i + $ChunkSize - 1, $remaining.Count - 1)
    $chunkRows = @($remaining[$i..$end])
    $chunkPath = Join-Path $ChunkDir ("chunk_{0:D3}.csv" -f [int]($i / $ChunkSize))

    $lines = @("part,dataset_name,seed,samples,scene_path,camera_path_csv,manifest,command")
    foreach ($r in $chunkRows) {
        $lines += "$($r.part),$($r.dataset_name),$($r.seed),$($r.samples),$($r.scene_path),$($r.camera_path_csv),$($r.manifest),$($r.command)"
    }
    $lines | Set-Content $chunkPath -Encoding ASCII
    $chunkFiles += $chunkPath
}

Write-Host "Chunk plans:"
$chunkFiles | ForEach-Object { Write-Host "  $_" }
Write-Host ""

foreach ($chunk in $chunkFiles) {
    Write-Host "============================================================"
    Write-Host "Running chunk: $chunk"
    Write-Host "============================================================"

    & $ExporterExe `
        --batch-plan $chunk `
        --output-root $OutputRoot `
        --auto-start-batch `
        --exit-when-done

    if ($LASTEXITCODE -ne 0) {
        throw "NeuralPVSExporter failed for chunk '$chunk' with exit code $LASTEXITCODE"
    }

    Write-Host "Completed chunk: $chunk"
    Write-Host ""
}

Write-Host "All requested chunks completed. Re-run validation on the original plan."
