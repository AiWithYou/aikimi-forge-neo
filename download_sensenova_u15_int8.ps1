param(
    [switch]$RuntimeOnly,
    [switch]$ModelOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

if ($RuntimeOnly -and $ModelOnly) {
    throw "-RuntimeOnly and -ModelOnly cannot be used together."
}

$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
chcp 65001 | Out-Null

Set-Location -LiteralPath $PSScriptRoot

$SourceRepository = "starsFriday/ComfyUI-SenseNova"
$SourceRevision = "e6dfd45762eb46f805067fe079c14bcb643ccccd"
$SourceTreeUrl = "https://api.github.com/repos/$SourceRepository/git/trees/$($SourceRevision)?recursive=1"
$SourceRawRoot = "https://raw.githubusercontent.com/$SourceRepository/$SourceRevision"
$RuntimeRoot = Join-Path $PSScriptRoot "models\SenseNova-U1\runtime-final"
$RuntimeRevisionPath = Join-Path $RuntimeRoot ".sensenova_runtime_revision"

$ModelRepository = "joyfox/SenseNova-U1.5-8B-MoT-FP8"
$ModelRevision = "57de22ad4e2fc24c77f56dfe45dbb87a60dfebee"
$ModelFileName = "SenseNova-U1.5-8B-MoT-pruned-int8_convrot.safetensors"
$ModelUrl = "https://huggingface.co/$ModelRepository/resolve/$ModelRevision/$ModelFileName"
$ModelPath = Join-Path $PSScriptRoot "models\SenseNova-U1\$ModelFileName"
$ModelBytes = [Int64]17734813848
$ModelSha256 = "cf6ed9ee3be516612b7fe083edfc7c9dd5d059cc759e300d2cf1f2726c0d250e"
$LoraRepository = "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"
$LoraRevision = "e909f4636d119d65fe4cba8770c19daff2ac102e"
$LoraFileName = "SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"
$LoraUrl = "https://huggingface.co/$LoraRepository/resolve/$LoraRevision/$LoraFileName"
$LoraPath = Join-Path $PSScriptRoot "models\SenseNova-U1\$LoraFileName"
$LoraBytes = [Int64]814867236
$LoraSha256 = "3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708"
$LoraTargetCount = 294
$ParallelDownloads = 16

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

function Get-GitBlobSha1 {
    param([string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $prefix = [System.Text.Encoding]::UTF8.GetBytes("blob $($bytes.LongLength)`0")
    $payload = New-Object byte[] ($prefix.Length + $bytes.Length)
    [Array]::Copy($prefix, 0, $payload, 0, $prefix.Length)
    [Array]::Copy($bytes, 0, $payload, $prefix.Length, $bytes.Length)
    $sha1 = [System.Security.Cryptography.SHA1]::Create()
    try {
        return ([BitConverter]::ToString($sha1.ComputeHash($payload))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha1.Dispose()
    }
}

function Install-SenseNovaRuntimeSource {
    $headers = @{ "User-Agent" = "Forge-Neo-SenseNova-Installer" }
    Write-Host "Downloading pinned final SenseNova ConvRot runtime: $SourceRevision"
    $tree = Invoke-RestMethod -UseBasicParsing -Headers $headers -Uri $SourceTreeUrl
    if ($tree.truncated) {
        throw "Pinned SenseNova source tree response was truncated. No runtime files were changed."
    }
    $entries = @(
        $tree.tree | Where-Object {
            $_.type -eq "blob" -and (
                $_.path -like "SenseNova/*" -or
                $_.path -like "SenseNova-U1.5-8B-MoT/*" -or
                $_.path -eq "LICENSE"
            )
        }
    )
    if ($entries.Count -lt 25) {
        throw "Pinned SenseNova source tree is incomplete: only $($entries.Count) files were listed."
    }

    $runtimeRootFull = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $runtimePrefix = "$runtimeRootFull$([System.IO.Path]::DirectorySeparatorChar)"
    foreach ($entry in $entries) {
        $relativePath = [string]$entry.path
        $pathParts = @($relativePath -split "[/\\]")
        if (
            [System.IO.Path]::IsPathRooted($relativePath) -or
            $pathParts.Count -eq 0 -or
            $pathParts -contains ".." -or
            $pathParts -contains "." -or
            $pathParts -contains ""
        ) {
            throw "Pinned SenseNova source tree contained an unsafe path."
        }
        $targetPath = [System.IO.Path]::GetFullPath((Join-Path $runtimeRootFull $relativePath))
        if (-not $targetPath.StartsWith($runtimePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Pinned SenseNova source tree path escaped the managed runtime directory."
        }
        $targetDirectory = Split-Path -Parent $targetPath
        New-RequiredDirectory $targetDirectory

        if (Test-Path -LiteralPath $targetPath) {
            $existingSha1 = Get-GitBlobSha1 -Path $targetPath
            if ($existingSha1 -eq $entry.sha) {
                continue
            }
        }

        $partialPath = "$targetPath.part"
        Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri "$SourceRawRoot/$($entry.path)" -OutFile $partialPath
        $downloadedSha1 = Get-GitBlobSha1 -Path $partialPath
        if ($downloadedSha1 -ne $entry.sha) {
            throw "Git blob SHA-1 mismatch for $($entry.path)."
        }
        Move-Item -Force -LiteralPath $partialPath -Destination $targetPath
    }

    [System.IO.File]::WriteAllText($RuntimeRevisionPath, "$SourceRevision`n", $Utf8NoBom)
    Write-Host "SenseNova runtime source is ready: $RuntimeRoot"
}

function Assert-SenseNovaDependencies {
    $forgePython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    $workerEnvironment = Join-Path $PSScriptRoot "models\SenseNova-U1\worker-env"
    $pythonPath = Join-Path $workerEnvironment "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $forgePython -PathType Leaf)) {
        throw "Forge Python was not found: $forgePython"
    }
    $uvCommand = Get-Command uv -CommandType Application -ErrorAction Stop
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        & $uvCommand.Source venv $workerEnvironment --python $forgePython
        if ($LASTEXITCODE -ne 0) { throw "SenseNova worker environment creation failed." }
    }
    & $uvCommand.Source pip install --python $pythonPath --link-mode copy `
        --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match `
        -r (Join-Path $PSScriptRoot "tools\requirements-sensenova.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "SenseNova worker dependency installation failed."
    }

    & $pythonPath -c "import accelerate, comfy_kitchen, safetensors, tokenizers, torch, torchvision, tqdm, transformers; from modules_forge.sensenova_u15_environment import validate_running_versions; validate_running_versions()"
    if ($LASTEXITCODE -ne 0) {
        throw "SenseNova worker dependencies are incomplete. Retry -RuntimeOnly."
    }
    Write-Host "SenseNova Python dependencies are ready."
}

function Assert-ConvRotSafetensorsHeader {
    param([string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        if ($stream.Length -lt 16) {
            throw "File is too small to be safetensors: $Path"
        }
        $header = New-Object byte[] 8
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length) {
            throw "Cannot read safetensors header length: $Path"
        }
        $headerLength = [BitConverter]::ToUInt64($header, 0)
        if ($headerLength -lt 3 -or $headerLength -gt 256MB) {
            throw "Invalid safetensors header length: $headerLength"
        }
        $metadata = New-Object byte[] ([Int32]$headerLength)
        $offset = 0
        while ($offset -lt $metadata.Length) {
            $read = $stream.Read($metadata, $offset, $metadata.Length - $offset)
            if ($read -le 0) {
                throw "Cannot read complete safetensors header: $Path"
            }
            $offset += $read
        }
    }
    finally {
        $stream.Dispose()
    }

    $metadataText = [System.Text.Encoding]::UTF8.GetString($metadata)
    if (-not $metadataText.Contains(".comfy_quant") -or -not $metadataText.Contains("fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight")) {
        throw "Checkpoint does not contain the SenseNova INT8 ConvRot signature: $Path"
    }
    $convRotLayerCount = [regex]::Matches($metadataText, '\.comfy_quant"').Count
    if ($convRotLayerCount -ne 588) {
        throw "Expected 588 INT8 ConvRot layers, found $convRotLayerCount in $Path"
    }
}

function Assert-ModelFile {
    param([string]$Path)

    $actualBytes = (Get-Item -LiteralPath $Path).Length
    if ($actualBytes -ne $ModelBytes) {
        throw "File size mismatch for $Path. Expected $ModelBytes bytes, got $actualBytes bytes."
    }
    Assert-ConvRotSafetensorsHeader -Path $Path

    Write-Host "Verifying SHA-256 (this reads the complete 16.52 GiB file once)..."
    $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $ModelSha256) {
        throw "SHA-256 mismatch for $Path. Expected $ModelSha256, got $actualSha256."
    }
}

function Assert-Official8StepLoraHeader {
    param([string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 8
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length) {
            throw "Cannot read the official 8-step LoRA header length: $Path"
        }
        $headerLength = [BitConverter]::ToUInt64($header, 0)
        if ($headerLength -lt 3 -or $headerLength -gt 256MB) {
            throw "Invalid official 8-step LoRA header length: $headerLength"
        }
        $metadata = New-Object byte[] ([Int32]$headerLength)
        $offset = 0
        while ($offset -lt $metadata.Length) {
            $read = $stream.Read($metadata, $offset, $metadata.Length - $offset)
            if ($read -le 0) {
                throw "Cannot read the complete official 8-step LoRA header: $Path"
            }
            $offset += $read
        }
    }
    finally {
        $stream.Dispose()
    }

    $metadataText = [System.Text.Encoding]::UTF8.GetString($metadata)
    if (-not $metadataText.Contains('"tensor_kind":"neo_hf_lora"')) {
        throw "The official 8-step LoRA tensor_kind marker is missing: $Path"
    }
    $downCount = [regex]::Matches($metadataText, '\.lora_down\.weight"').Count
    $upCount = [regex]::Matches($metadataText, '\.lora_up\.weight"').Count
    $alphaCount = [regex]::Matches($metadataText, '\.alpha"').Count
    if ($downCount -ne $LoraTargetCount -or $upCount -ne $LoraTargetCount -or $alphaCount -ne $LoraTargetCount) {
        throw "Official 8-step LoRA target coverage mismatch: down=$downCount up=$upCount alpha=$alphaCount"
    }
}

function Assert-Official8StepLora {
    param([string]$Path)

    $actualBytes = (Get-Item -LiteralPath $Path).Length
    if ($actualBytes -ne $LoraBytes) {
        throw "Official 8-step LoRA size mismatch. Expected $LoraBytes bytes, got $actualBytes bytes."
    }
    Assert-Official8StepLoraHeader -Path $Path
    Write-Host "Verifying official 8-step LoRA SHA-256..."
    $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $LoraSha256) {
        throw "Official 8-step LoRA SHA-256 mismatch. Expected $LoraSha256, got $actualSha256."
    }
}

function Install-Official8StepLora {
    $targetDirectory = Split-Path -Parent $LoraPath
    New-RequiredDirectory $targetDirectory

    if (Test-Path -LiteralPath $LoraPath -PathType Leaf) {
        Assert-Official8StepLora -Path $LoraPath
        [System.IO.File]::WriteAllText("$LoraPath.sha256", "$LoraSha256  $LoraFileName`n", $Utf8NoBom)
        Write-Host "Existing official 8-step T2I LoRA is valid. Skipping download."
        return
    }

    $curlPath = (Get-Command curl.exe -ErrorAction Stop).Source
    $partialPath = "$LoraPath.part"
    Write-Host "Downloading the official SenseNova U1.5 8-step T2I LoRA."
    $partialBytes = if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
        (Get-Item -LiteralPath $partialPath).Length
    }
    else {
        [Int64]0
    }
    if ($partialBytes -gt $LoraBytes) {
        throw "Official 8-step LoRA partial file is larger than expected: $partialPath"
    }
    if ($partialBytes -lt $LoraBytes) {
        & $curlPath --location --fail --continue-at - --retry 5 --retry-delay 5 --retry-all-errors --output $partialPath $LoraUrl
        if ($LASTEXITCODE -ne 0) {
            throw "Official 8-step LoRA download failed. Run this script again to resume."
        }
    }
    Assert-Official8StepLora -Path $partialPath
    Move-Item -LiteralPath $partialPath -Destination $LoraPath
    [System.IO.File]::WriteAllText("$LoraPath.sha256", "$LoraSha256  $LoraFileName`n", $Utf8NoBom)
    Write-Host "Official 8-step T2I LoRA is ready."
}

function Download-ConvRotChunks {
    param(
        [string]$CurlPath,
        [string]$Url,
        [string]$TargetPath,
        [Int64]$ExpectedBytes
    )

    $partialPath = "$TargetPath.part"
    $chunkDirectory = "$partialPath.chunks"
    New-RequiredDirectory $chunkDirectory
    $chunkRoot = (Resolve-Path -LiteralPath $chunkDirectory).Path
    $targetDirectory = (Resolve-Path -LiteralPath (Split-Path -Parent $TargetPath)).Path
    if (-not $chunkRoot.StartsWith($targetDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Chunk directory escaped the SenseNova model directory: $chunkRoot"
    }

    function Merge-ChunkResume {
        param(
            [string]$ChunkPath,
            [string]$ResumePath,
            [Int64]$ExpectedChunkBytes
        )

        if (-not (Test-Path -LiteralPath $ResumePath -PathType Leaf)) {
            return
        }
        $chunkLength = if (Test-Path -LiteralPath $ChunkPath -PathType Leaf) {
            (Get-Item -LiteralPath $ChunkPath).Length
        }
        else {
            [Int64]0
        }
        $resumeLength = (Get-Item -LiteralPath $ResumePath).Length
        if ($chunkLength + $resumeLength -gt $ExpectedChunkBytes) {
            $originalLength = $ExpectedChunkBytes - $resumeLength
            if ($originalLength -lt 0 -or $originalLength -gt $chunkLength) {
                throw "Resumed chunk would exceed its expected size: $ChunkPath"
            }
            $repair = [System.IO.File]::Open(
                $ChunkPath,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Write
            )
            try {
                $repair.SetLength($originalLength)
            }
            finally {
                $repair.Dispose()
            }
            $chunkLength = $originalLength
        }
        if ($chunkLength -eq 0) {
            Move-Item -Force -LiteralPath $ResumePath -Destination $ChunkPath
            return
        }

        $output = [System.IO.File]::Open(
            $ChunkPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::Write
        )
        try {
            [void]$output.Seek(0, [System.IO.SeekOrigin]::End)
            $input = [System.IO.File]::OpenRead($ResumePath)
            try {
                $input.CopyTo($output)
            }
            finally {
                $input.Dispose()
            }
        }
        finally {
            $output.Dispose()
        }
        Remove-Item -LiteralPath $ResumePath
    }

    $chunkBytes = [Int64](32MB)
    $chunks = New-Object System.Collections.Generic.List[object]
    for ($start = [Int64]0; $start -lt $ExpectedBytes; $start += $chunkBytes) {
        $end = [Math]::Min($ExpectedBytes - 1, $start + $chunkBytes - 1)
        $index = $chunks.Count + 1
        $path = Join-Path $chunkRoot ("chunk-{0:D3}.bin" -f $index)
        $chunks.Add([pscustomobject]@{
            Index = $index
            Start = $start
            End = $end
            Bytes = $end - $start + 1
            Path = $path
        })
    }

    $pending = New-Object System.Collections.Generic.List[object]
    foreach ($chunk in $chunks) {
        $resumePath = "$($chunk.Path).resume"
        Merge-ChunkResume -ChunkPath $chunk.Path -ResumePath $resumePath -ExpectedChunkBytes $chunk.Bytes
        $existingBytes = if (Test-Path -LiteralPath $chunk.Path -PathType Leaf) {
            (Get-Item -LiteralPath $chunk.Path).Length
        }
        else {
            [Int64]0
        }
        if ($existingBytes -eq $chunk.Bytes) {
            continue
        }
        if ($existingBytes -gt $chunk.Bytes) {
            throw "Partial chunk is larger than expected: $($chunk.Path)"
        }
        $pending.Add([pscustomobject]@{
            Index = $chunk.Index
            Start = $chunk.Start + $existingBytes
            End = $chunk.End
            RemainingBytes = $chunk.Bytes - $existingBytes
            Path = $chunk.Path
            ResumePath = $resumePath
        })
    }

    if ($pending.Count -gt 0) {
        $configPath = Join-Path $chunkRoot "download.curl"
        $configLines = New-Object System.Collections.Generic.List[string]
        for ($i = 0; $i -lt $pending.Count; $i++) {
            $chunk = $pending[$i]
            $outputPath = ([string]$chunk.ResumePath).Replace("\", "/")
            $configLines.Add("location")
            $configLines.Add("fail")
            $configLines.Add("retry = 5")
            $configLines.Add("retry-delay = 3")
            $configLines.Add("retry-all-errors")
            $configLines.Add("url = `"$Url`"")
            $configLines.Add("range = `"$($chunk.Start)-$($chunk.End)`"")
            $configLines.Add("output = `"$outputPath`"")
            if ($i -lt $pending.Count - 1) {
                $configLines.Add("next")
            }
        }
        [System.IO.File]::WriteAllLines($configPath, $configLines, $Utf8NoBom)

        Write-Host ("Downloading {0} remaining 32 MiB chunk(s), up to {1} in parallel." -f $pending.Count, $ParallelDownloads)
        & $CurlPath --parallel --parallel-immediate --parallel-max $ParallelDownloads --progress-bar --show-error --config $configPath
        if ($LASTEXITCODE -ne 0) {
            throw "Parallel INT8 ConvRot download failed. Run this script again to reuse completed chunks."
        }
        foreach ($chunk in $pending) {
            $expectedChunkBytes = ($chunks[$chunk.Index - 1]).Bytes
            Merge-ChunkResume -ChunkPath $chunk.Path -ResumePath $chunk.ResumePath -ExpectedChunkBytes $expectedChunkBytes
        }
    }

    foreach ($chunk in $chunks) {
        if (-not (Test-Path -LiteralPath $chunk.Path -PathType Leaf)) {
            throw "Downloaded chunk is missing: $($chunk.Path)"
        }
        $actualBytes = (Get-Item -LiteralPath $chunk.Path).Length
        if ($actualBytes -ne $chunk.Bytes) {
            throw "Chunk size mismatch: $($chunk.Path). Expected $($chunk.Bytes), got $actualBytes."
        }
    }

    $assemblingPath = "$partialPath.assembling"
    $output = [System.IO.File]::Open($assemblingPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        foreach ($chunk in $chunks) {
            $input = [System.IO.File]::OpenRead($chunk.Path)
            try {
                $input.CopyTo($output)
            }
            finally {
                $input.Dispose()
            }
        }
    }
    finally {
        $output.Dispose()
    }

    if ((Get-Item -LiteralPath $assemblingPath).Length -ne $ExpectedBytes) {
        throw "Assembled INT8 ConvRot file has an unexpected size: $assemblingPath"
    }
    Move-Item -Force -LiteralPath $assemblingPath -Destination $partialPath

}

function Install-ConvRotModel {
    $targetDirectory = Split-Path -Parent $ModelPath
    New-RequiredDirectory $targetDirectory

    Write-Host "Downloading final SenseNova U1.5 INT8 ConvRot checkpoint."
    Write-Host "Target: $ModelPath"
    Write-Host ("Size: {0}" -f (Format-Bytes $ModelBytes))
    Write-Host "Source: https://huggingface.co/$ModelRepository"
    Write-Host "Note: the base is the formal SenseNova U1.5 release; this INT8 ConvRot conversion and loader are community-maintained."

    if (Test-Path -LiteralPath $ModelPath -PathType Leaf) {
        Assert-ModelFile -Path $ModelPath
        [System.IO.File]::WriteAllText("$ModelPath.sha256", "$ModelSha256  $ModelFileName`n", $Utf8NoBom)
        Write-Host "Existing INT8 ConvRot model is valid. Skipping download."
        return
    }

    $curlPath = (Get-Command curl.exe -ErrorAction Stop).Source
    $partialPath = "$ModelPath.part"
    if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
        $partialBytes = (Get-Item -LiteralPath $partialPath).Length
        if ($partialBytes -gt $ModelBytes) {
            throw "Partial file is larger than the pinned model: $partialPath"
        }
        if ($partialBytes -eq $ModelBytes) {
            Write-Host "Completed partial file is ready for integrity verification."
        }
        else {
            Write-Host ("Resuming legacy single-file download from {0}." -f (Format-Bytes $partialBytes))
            & $curlPath --location --fail --continue-at - --retry 5 --retry-delay 5 --retry-all-errors --output $partialPath $ModelUrl
            if ($LASTEXITCODE -ne 0) {
                throw "INT8 ConvRot download failed. Run this script again to resume the .part file."
            }
        }
    }
    else {
        Download-ConvRotChunks -CurlPath $curlPath -Url $ModelUrl -TargetPath $ModelPath -ExpectedBytes $ModelBytes
    }

    Assert-ModelFile -Path $partialPath
    Move-Item -LiteralPath $partialPath -Destination $ModelPath
    [System.IO.File]::WriteAllText("$ModelPath.sha256", "$ModelSha256  $ModelFileName`n", $Utf8NoBom)
    $chunkDirectory = "$partialPath.chunks"
    if (Test-Path -LiteralPath $chunkDirectory -PathType Container) {
        $chunkRoot = (Resolve-Path -LiteralPath $chunkDirectory).Path
        $targetDirectory = (Resolve-Path -LiteralPath (Split-Path -Parent $ModelPath)).Path
        if (-not $chunkRoot.StartsWith($targetDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Chunk cleanup target escaped the SenseNova model directory: $chunkRoot"
        }
        foreach ($file in Get-ChildItem -LiteralPath $chunkRoot -File) {
            if (-not $file.FullName.StartsWith($chunkRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Chunk cleanup target escaped the chunk directory: $($file.FullName)"
            }
            Remove-Item -LiteralPath $file.FullName
        }
        Remove-Item -LiteralPath $chunkRoot
    }
    Write-Host "SenseNova final INT8 ConvRot model is ready."
}

Write-Host "Forge Neo SenseNova U1.5 setup"
Write-Host "Pinned runtime source: $SourceRevision"
Write-Host "Pinned INT8 ConvRot model revision: $ModelRevision"

if (-not $ModelOnly) {
    Install-SenseNovaRuntimeSource
    Assert-SenseNovaDependencies
}
if (-not $RuntimeOnly) {
    Install-ConvRotModel
    Install-Official8StepLora
}

Write-Host ""
Write-Host "Setup completed."
Write-Host "Open the SenseNova U1.5 tab in Forge Neo."
