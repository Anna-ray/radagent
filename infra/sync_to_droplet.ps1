<#
.SYNOPSIS
    Sync RadAgent local artifacts to an MI300X droplet.

.DESCRIPTION
    Pushes specialist checkpoint, RAG index, and sample test images.
    Uses scp under the hood (works on Windows 10+ with built-in OpenSSH).

.PARAMETER DropletIp
    The droplet's public IP, e.g. 165.232.10.20.

.PARAMETER SshKey
    Path to your private SSH key. Default: ~\.ssh\amd_radagent

.PARAMETER User
    SSH user. Default: root.

.EXAMPLE
    .\infra\sync_to_droplet.ps1 -DropletIp 165.232.10.20
#>
param(
    [Parameter(Mandatory=$true)] [string] $DropletIp,
    [string] $SshKey = "$HOME\.ssh\amd_radagent",
    [string] $User = "root",
    [string] $RemoteRoot = "/shared-docker/radagent",
    [int] $NImages = 10
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SshKey)) {
    Write-Error "SSH key not found at $SshKey. Pass -SshKey to override."
}

# Items to sync. Path on local -> path on droplet.
$artifacts = @(
    @{ Local = "runs\nih14_convnextv2_base_384\best.pt";
       Remote = "$RemoteRoot/runs/nih14_convnextv2_base_384/best.pt" },
    @{ Local = "runs\nih14_convnextv2_base_384\calibration.json";
       Remote = "$RemoteRoot/runs/nih14_convnextv2_base_384/calibration.json" },
    @{ Local = "runs\nih14_convnextv2_base_384\calibration_bands.json";
       Remote = "$RemoteRoot/runs/nih14_convnextv2_base_384/calibration_bands.json" },
    @{ Local = "configs\nih14_convnextv2_base.yaml";
       Remote = "$RemoteRoot/configs/nih14_convnextv2_base.yaml" },
    @{ Local = "data\rag\index.faiss";
       Remote = "$RemoteRoot/data/rag/index.faiss" },
    @{ Local = "data\rag\chunks.jsonl";
       Remote = "$RemoteRoot/data/rag/chunks.jsonl" },
    @{ Local = "data\rag\manifest.json";
       Remote = "$RemoteRoot/data/rag/manifest.json" }
)

# Code: tar a focused subset (avoids node_modules / __pycache__ / runs)
$codeArchive = "$env:TEMP\radagent_code.tar.gz"
if (Test-Path $codeArchive) { Remove-Item $codeArchive }

Write-Host "[sync] packaging code into $codeArchive ..."
$tarPaths = @("radagent", "scripts", "infra", "configs")
$existing = $tarPaths | Where-Object { Test-Path $_ }
tar -czf $codeArchive --exclude="__pycache__" --exclude="*.pyc" $existing
if ($LASTEXITCODE -ne 0) { Write-Error "tar failed" }
Write-Host "  -> $((Get-Item $codeArchive).Length / 1MB) MB"

# Sample test images
$sampleDir = "$env:TEMP\radagent_samples"
if (Test-Path $sampleDir) { Remove-Item $sampleDir -Recurse -Force }
New-Item -ItemType Directory -Path $sampleDir | Out-Null

$nihRoot = "C:\Users\pc\Desktop\radagent\data\nih"
if (Test-Path $nihRoot) {
    Write-Host "[sync] picking $NImages sample CXRs from $nihRoot ..."
    $sampleImages = Get-ChildItem -Path $nihRoot -Recurse -Filter *.png |
                    Get-Random -Count $NImages
    $sampleImages | ForEach-Object {
        Copy-Item $_.FullName -Destination $sampleDir
    }
    $sampleArchive = "$env:TEMP\radagent_samples.tar.gz"
    if (Test-Path $sampleArchive) { Remove-Item $sampleArchive }
    tar -czf $sampleArchive -C "$env:TEMP" radagent_samples
    Write-Host "  -> $((Get-Item $sampleArchive).Length / 1MB) MB"
} else {
    Write-Warning "NIH dataset not found at $nihRoot, skipping sample images"
    $sampleArchive = $null
}

# Make sure remote dirs exist
Write-Host "[sync] preparing remote dirs ..."
$mkdirCmd = @(
    "mkdir -p $RemoteRoot",
    "mkdir -p $RemoteRoot/runs/nih14_convnextv2_base_384",
    "mkdir -p $RemoteRoot/configs",
    "mkdir -p $RemoteRoot/data/rag",
    "mkdir -p $RemoteRoot/data/samples"
) -join " && "
ssh -i $SshKey "$User@$DropletIp" $mkdirCmd

# Push each artifact
foreach ($a in $artifacts) {
    if (-not (Test-Path $a.Local)) {
        Write-Warning "Missing locally: $($a.Local) -- skipping"
        continue
    }
    Write-Host "[sync] -> $($a.Remote)"
    scp -i $SshKey $a.Local "${User}@${DropletIp}:$($a.Remote)"
    if ($LASTEXITCODE -ne 0) { Write-Error "scp failed for $($a.Local)" }
}

# Push code archive
Write-Host "[sync] -> $RemoteRoot/code.tar.gz"
scp -i $SshKey $codeArchive "${User}@${DropletIp}:$RemoteRoot/code.tar.gz"
ssh -i $SshKey "$User@$DropletIp" "cd $RemoteRoot && tar -xzf code.tar.gz && rm code.tar.gz"

# Push samples
if ($sampleArchive) {
    Write-Host "[sync] -> $RemoteRoot/data/samples.tar.gz"
    scp -i $SshKey $sampleArchive "${User}@${DropletIp}:$RemoteRoot/data/samples.tar.gz"
    ssh -i $SshKey "$User@$DropletIp" "cd $RemoteRoot/data && tar -xzf samples.tar.gz && rm samples.tar.gz"
}

Write-Host ""
Write-Host "[sync] done."
Write-Host "On the droplet, you should now have:"
Write-Host "  $RemoteRoot/runs/nih14_convnextv2_base_384/best.pt"
Write-Host "  $RemoteRoot/data/rag/index.faiss"
Write-Host "  $RemoteRoot/data/samples/<10 images>"
Write-Host "  $RemoteRoot/{radagent,scripts,infra,configs}/..."


