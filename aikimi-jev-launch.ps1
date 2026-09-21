$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host 'Jevモードではモデルの層別集約統計をTypeSafe APIへ送信します。'
Write-Host '生画像・音声・重みはAPIへ送信せず、APIキーは生成履歴に保存しません。API利用料金が発生する場合があります。'
$oldKey = $env:TYPESAFE_API_KEY
$oldAllow = $env:AIKIMI_JEV_ALLOW_CLOUD
$ptr = [IntPtr]::Zero
try {
    . (Join-Path $PSScriptRoot 'tools/jev_credentials.ps1')
    Enable-AikimiJevCredential
    & .\aikimi-launch.bat
} finally {
    if ($ptr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    $env:TYPESAFE_API_KEY = $oldKey
    $env:AIKIMI_JEV_ALLOW_CLOUD = $oldAllow
}
