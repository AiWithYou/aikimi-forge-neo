param(
    [ValidateSet("off", "dense", "fixed", "rules", "jev")]
    [string]$Mode = "dense",
    [ValidateRange(1,100)][double]$Keep = 75,
    [ValidateSet(64,128,256,512)][int]$BlockSize = 256,
    [ValidateRange(64,1048576)][int]$MinTokens = 1024,
    [ValidateRange(1,100)][int]$Warmup = 1,
    [ValidateRange(1,100)][int]$Interval = 4,
    [ValidateRange(0,8)][int]$MaxCalls = 1,
    [ValidateRange(0.5,20)][double]$Timeout = 3
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0
Set-Location -LiteralPath $PSScriptRoot
$Names = @("AIKIMI_QWEN21_SPARSE_OPTIONS", "AIKIMI_JEV_ALLOW_CLOUD", "TYPESAFE_API_KEY")
$Previous = @{}
foreach ($Name in $Names) { $Previous[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process") }
$Ptr = [IntPtr]::Zero
try {
    $env:AIKIMI_QWEN21_SPARSE_OPTIONS = @{
        mode=$Mode; keep_percent=$Keep; block_size=$BlockSize; min_tokens=$MinTokens
        warmup_evaluations=$Warmup; update_interval=$Interval; max_calls=$MaxCalls; timeout=$Timeout
    } | ConvertTo-Json -Compress
    $env:AIKIMI_JEV_ALLOW_CLOUD = $null
    $env:TYPESAFE_API_KEY = $null
    Write-Host "[Qwen 2.1 Sparse] $Mode / target keep $Keep%. Existing Qwen Image 2.1 tab; restart Neo to switch modes."
    if ($Mode -eq "jev" -and $MaxCalls -gt 0) {
        Write-Host "Qwen layer statistics will be sent to https://api.typesafe.ai. No prompts, images or weights are sent. API charges may apply."
        . (Join-Path $PSScriptRoot 'tools/jev_credentials.ps1')
        Enable-AikimiJevCredential
    }
    & (Join-Path $PSScriptRoot "aikimi-launch.ps1")
} finally {
    if ($Ptr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Ptr) }
    foreach ($Name in $Names) { [Environment]::SetEnvironmentVariable($Name, $Previous[$Name], "Process") }
}
