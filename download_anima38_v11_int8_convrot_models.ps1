param(
    [switch]$KeepSource
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
chcp 65001 | Out-Null

Set-Location -LiteralPath $PSScriptRoot

$Repository = "lylogummy/Anima-3.8B"
$Revision = "3ef641256377dc4e7efbf35d426ca31c1fe5180b"
$SourceDirectory = Join-Path $PSScriptRoot "tmp\anima38-v11-conversion"
$SourcePath = Join-Path $SourceDirectory "Anima-3.8B-v1.1.safetensors"
$OutputPath = Join-Path $PSScriptRoot "models\Stable-diffusion\Anima-3.8B-v1.1-int8-convrot.safetensors"
$PythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$ConverterPath = Join-Path $PSScriptRoot "tools\convert_anima38_int8_convrot.py"
$ConverterModule = "tools.convert_anima38_int8_convrot"
$ConvertedBytes = [Int64]5543364574

$Downloads = @(
    @{
        Label = "Anima 3.8B Qwen3.5 4B mixed FP8 text encoder"
        Url = "https://huggingface.co/$Repository/resolve/$Revision/text_encoders/qwen35_4b.safetensors"
        RelativePath = "models\text_encoder\qwen35_4b.safetensors"
        Bytes = [Int64]4779016600
        Sha256 = "ea289be7c916726d09953c7db9971c82b280e694b5d7c47f8ad9ffad6acb54ba"
        Markers = @("embed_tokens.weight", "layers.31.input_layernorm.weight")
        Source = "https://huggingface.co/$Repository/tree/$Revision/text_encoders"
        License = "https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE"
    }
)

$SourceDownload = @{
    Label = "Anima 3.8B v1.1 BF16 bundle conversion source"
    Url = "https://huggingface.co/$Repository/resolve/$Revision/difussion_models/Anima-3.8B-v1.1.safetensors"
    Bytes = [Int64]8809227318
    Sha256 = "4a458d26b21efa350073422f756d521b4397d9ca5964da4dc6bd9ae258a29629"
    Markers = @(
        "anima_3_8b_semantic_connector_v2_bundle",
        "net.anima_v2_connector.semantic_resampler.query_tokens",
        "net.blocks.51.mlp.layer2.weight"
    )
    Source = "https://huggingface.co/$Repository/tree/$Revision/difussion_models"
    License = "https://huggingface.co/$Repository#licenses"
}

function Format-Bytes {
    param([Int64]$Bytes)

    if ($Bytes -ge 1GB) {
        return ("{0:N2} GiB" -f ($Bytes / 1GB))
    }
    if ($Bytes -ge 1MB) {
        return ("{0:N2} MiB" -f ($Bytes / 1MB))
    }
    return ("{0:N0} bytes" -f $Bytes)
}

function New-RequiredDirectory {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path
        if (-not $item.PSIsContainer) {
            throw "A file exists where a directory is required: $Path"
        }
        return
    }
    New-Item -ItemType Directory -Path $Path | Out-Null
}

function Get-SafetensorsHeaderText {
    param([string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        if ($stream.Length -lt 16) {
            throw "File is too small to be safetensors: $Path"
        }
        $lengthBytes = New-Object byte[] 8
        if ($stream.Read($lengthBytes, 0, 8) -ne 8) {
            throw "Cannot read safetensors header length: $Path"
        }
        $headerLength = [BitConverter]::ToUInt64($lengthBytes, 0)
        if ($headerLength -le 2 -or $headerLength -gt 256MB) {
            throw "Invalid safetensors header length: $Path ($headerLength bytes)"
        }
        $headerBytes = New-Object byte[] ([Int32]$headerLength)
        $offset = 0
        while ($offset -lt $headerBytes.Length) {
            $read = $stream.Read($headerBytes, $offset, $headerBytes.Length - $offset)
            if ($read -le 0) {
                throw "Cannot read the full safetensors header: $Path"
            }
            $offset += $read
        }
    }
    finally {
        $stream.Dispose()
    }

    $headerText = [System.Text.Encoding]::UTF8.GetString($headerBytes)
    $null = $headerText | ConvertFrom-Json
    return $headerText
}

function Assert-SafetensorsFile {
    param(
        [string]$Path,
        [Int64]$ExpectedBytes,
        [string]$ExpectedSha256,
        [string[]]$Markers
    )

    $actualBytes = (Get-Item -LiteralPath $Path).Length
    if ($actualBytes -ne $ExpectedBytes) {
        throw "File size mismatch: $Path. Expected $ExpectedBytes bytes, got $actualBytes bytes."
    }
    $headerText = Get-SafetensorsHeaderText -Path $Path
    foreach ($marker in $Markers) {
        if (-not $headerText.Contains($marker)) {
            throw "Required safetensors marker was not found: $marker. Target: $Path"
        }
    }
    $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $ExpectedSha256) {
        throw "SHA256 mismatch: $Path. Expected $ExpectedSha256, got $actualSha256."
    }
}

function Download-ModelFile {
    param(
        [string]$CurlPath,
        [string]$Label,
        [string]$Url,
        [string]$TargetPath,
        [Int64]$ExpectedBytes,
        [string]$ExpectedSha256,
        [string[]]$Markers,
        [string]$Source,
        [string]$License
    )

    New-RequiredDirectory (Split-Path -Parent $TargetPath)
    Write-Host ""
    Write-Host "== $Label =="
    Write-Host "Target: $TargetPath"
    Write-Host ("Size: {0}" -f (Format-Bytes $ExpectedBytes))
    Write-Host "Source: $Source"
    Write-Host "License terms: $License"

    if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
        Assert-SafetensorsFile $TargetPath $ExpectedBytes $ExpectedSha256 $Markers
        Write-Host "Existing file is valid. Skipping download."
        return
    }

    $partialPath = "$TargetPath.part"
    if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
        $partialBytes = (Get-Item -LiteralPath $partialPath).Length
        if ($partialBytes -gt $ExpectedBytes) {
            throw "Partial file is larger than expected: $partialPath"
        }
        Write-Host ("Resuming from {0}." -f (Format-Bytes $partialBytes))
    }

    & $CurlPath --location --fail --continue-at - --retry 5 --retry-delay 5 --retry-all-errors --output $partialPath $Url
    if ($LASTEXITCODE -ne 0) {
        throw "curl download failed: $Label"
    }
    Assert-SafetensorsFile $partialPath $ExpectedBytes $ExpectedSha256 $Markers
    Move-Item -LiteralPath $partialPath -Destination $TargetPath
    Write-Host "Done."
}

function Assert-NativeAnimaDependencies {
    $nativeFiles = @(
        @{
            Path = Join-Path $PSScriptRoot "models\text_encoder\qwen_3_06b_base.safetensors"
            Marker = "model.embed_tokens.weight"
        },
        @{
            Path = Join-Path $PSScriptRoot "models\VAE\qwen_image_vae.safetensors"
            Marker = "decoder.conv1.weight"
        }
    )
    foreach ($native in $nativeFiles) {
        if (-not (Test-Path -LiteralPath $native.Path -PathType Leaf)) {
            throw "Native Anima dependency is missing: $($native.Path)"
        }
        $headerText = Get-SafetensorsHeaderText -Path $native.Path
        if (-not $headerText.Contains($native.Marker)) {
            throw "Native Anima dependency has an unexpected layout: $($native.Path)"
        }
    }
}

function Assert-AnimaConvRotCheckpoint {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Converted Anima checkpoint is missing: $Path"
    }
    $actualBytes = (Get-Item -LiteralPath $Path).Length
    if ($actualBytes -ne $ConvertedBytes) {
        throw "Converted Anima checkpoint size mismatch. Expected $ConvertedBytes bytes, got $actualBytes bytes."
    }
    $headerText = Get-SafetensorsHeaderText -Path $Path
    $markers = @(
        "anima38_v11_main_attention_mlp_v1",
        "net.anima_v2_connector.semantic_resampler.query_tokens",
        "net.blocks.0.self_attn.q_proj.weight_scale",
        "net.blocks.51.mlp.layer2.comfy_quant"
    )
    foreach ($marker in $markers) {
        if (-not $headerText.Contains($marker)) {
            throw "Converted Anima checkpoint is missing marker: $marker"
        }
    }
    $layerCount = [regex]::Matches($headerText, '\.comfy_quant"').Count
    if ($layerCount -ne 520) {
        throw "Expected 520 INT8 ConvRot layers, found $layerCount in $Path"
    }
    $sidecarPath = "$Path.sha256"
    if (-not (Test-Path -LiteralPath $sidecarPath -PathType Leaf)) {
        throw "Converted Anima checksum sidecar is missing: $sidecarPath"
    }
    $record = (Get-Content -Raw -LiteralPath $sidecarPath).Trim()
    if ($record -notmatch '^([0-9a-f]{64})  Anima-3\.8B-v1\.1-int8-convrot\.safetensors$') {
        throw "Converted Anima checksum sidecar is invalid: $sidecarPath"
    }
    $expectedSha256 = $Matches[1]
    $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw "Converted Anima checkpoint SHA256 mismatch. Expected $expectedSha256, got $actualSha256."
    }
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Forge Python was not found: $PythonPath"
}
if (-not (Test-Path -LiteralPath $ConverterPath -PathType Leaf)) {
    throw "Anima converter was not found: $ConverterPath"
}

$curlPath = (Get-Command curl.exe -ErrorAction Stop).Source
Assert-NativeAnimaDependencies

Write-Host "This setup installs Anima 3.8B v1.1 with an INT8 ConvRot DiT, the bundled BF16 Semantic Connector v2, and the released mixed-FP8 Qwen3.5 4B encoder."
Write-Host "The BF16 bundle is staged only for conversion and removed after success unless -KeepSource is used."

foreach ($download in $Downloads) {
    $targetPath = Join-Path $PSScriptRoot $download.RelativePath
    Download-ModelFile `
        -CurlPath $curlPath `
        -Label $download.Label `
        -Url $download.Url `
        -TargetPath $targetPath `
        -ExpectedBytes $download.Bytes `
        -ExpectedSha256 $download.Sha256 `
        -Markers $download.Markers `
        -Source $download.Source `
        -License $download.License
}

if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
    Assert-AnimaConvRotCheckpoint -Path $OutputPath
    Write-Host "Existing Anima 3.8B v1.1 INT8 ConvRot checkpoint is valid. Skipping conversion."
}
else {
    Download-ModelFile `
        -CurlPath $curlPath `
        -Label $SourceDownload.Label `
        -Url $SourceDownload.Url `
        -TargetPath $SourcePath `
        -ExpectedBytes $SourceDownload.Bytes `
        -ExpectedSha256 $SourceDownload.Sha256 `
        -Markers $SourceDownload.Markers `
        -Source $SourceDownload.Source `
        -License $SourceDownload.License

    & $PythonPath -B -m $ConverterModule $SourcePath $OutputPath --device cuda:0 --group-size 256
    if ($LASTEXITCODE -ne 0) {
        throw "Anima 3.8B v1.1 INT8 ConvRot conversion failed. The BF16 source was kept for retry."
    }
    Assert-AnimaConvRotCheckpoint -Path $OutputPath
}

if ((-not $KeepSource) -and (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
    $resolvedSource = [System.IO.Path]::GetFullPath($SourcePath)
    $resolvedSourceDirectory = [System.IO.Path]::GetFullPath($SourceDirectory)
    if (-not $resolvedSource.StartsWith($resolvedSourceDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a source outside the conversion directory: $resolvedSource"
    }
    Remove-Item -LiteralPath $resolvedSource
    if ((Get-ChildItem -LiteralPath $resolvedSourceDirectory -Force | Measure-Object).Count -eq 0) {
        Remove-Item -LiteralPath $resolvedSourceDirectory
    }
    Write-Host "Removed the temporary BF16 conversion source."
}

Write-Host ""
Write-Host "Anima 3.8B v1.1 files are ready."
Write-Host "Preset: anima"
Write-Host "Checkpoint: Anima-3.8B-v1.1-int8-convrot.safetensors"
Write-Host "VAE / Text Encoder: qwen_image_vae.safetensors, qwen_3_06b_base.safetensors"
Write-Host "Semantic Connector v2: automatic, bundled, fixed strength 1.0"
Write-Host "Diffusion in Low Bits: Automatic"
