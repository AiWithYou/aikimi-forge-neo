param(
    [ValidateSet(
        "LocalSafe",
        "LocalAPI",
        "LANAuthenticated",
        "Development",
        "LowVRAM",
        "RTX3090Recommended"
    )]
    [string]$Profile = "LocalSafe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$RepositoryRoot = $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot

$Common = @(
    "--uv",
    "--port", "7861",
    "--theme", "dark"
)

$Arguments = switch ($Profile) {
    "LocalSafe" {
        @($Common + @(
            "--api",
            "--server-name", "127.0.0.1",
            "--bnb",
            "--tiled-conv2d", "128",
            "--cuda-malloc"
        ))
    }
    "LocalAPI" {
        @($Common + @(
            "--nowebui",
            "--server-name", "127.0.0.1"
        ))
    }
    "Development" {
        @($Common + @(
            "--api",
            "--server-name", "127.0.0.1",
            "--ui-debug-mode"
        ))
    }
    "LowVRAM" {
        @($Common + @(
            "--api",
            "--server-name", "127.0.0.1",
            "--lowvram",
            "--tiled-conv2d", "64"
        ))
    }
    "RTX3090Recommended" {
        @($Common + @(
            "--api",
            "--server-name", "127.0.0.1",
            "--bnb",
            "--tiled-conv2d", "128",
            "--cuda-malloc"
        ))
    }
    "LANAuthenticated" {
        $GradioAuth = Join-Path $RepositoryRoot "secrets\gradio-auth.txt"
        $ApiAuth = Join-Path $RepositoryRoot "secrets\api-auth.txt"
        if (-not (Test-Path -LiteralPath $GradioAuth -PathType Leaf)) {
            Write-Host "[Aikimi Studio Neo] LANAuthenticated requires secrets\gradio-auth.txt. See docs/security-model.md." -ForegroundColor Red
            exit 2
        }
        if (-not (Test-Path -LiteralPath $ApiAuth -PathType Leaf)) {
            Write-Host "[Aikimi Studio Neo] LANAuthenticated requires secrets\api-auth.txt. See docs/security-model.md." -ForegroundColor Red
            exit 2
        }
        @($Common + @(
            "--aikimi-remote",
            "--listen",
            "--api",
            "--gradio-auth-path", $GradioAuth,
            "--api-auth-path", $ApiAuth
        ))
    }
}

$ModelPathConfig = Join-Path $RepositoryRoot "forge_neo_model_paths.yaml"
if (Test-Path -LiteralPath $ModelPathConfig -PathType Leaf) {
    $Arguments += @("--forge-ref-comfy-yaml", $ModelPathConfig)
}

Write-Host "[Aikimi Studio Neo] Launch profile: $Profile"
& (Join-Path $RepositoryRoot "webui.bat") @Arguments
exit $LASTEXITCODE
