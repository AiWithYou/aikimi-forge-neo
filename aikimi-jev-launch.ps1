$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host 'JevモードではH3のプロンプトと集約統計、Animaの集約統計をTypeSafe APIへ送信します。'
Write-Host '生画像・音声・重みはAPIへ送信せず、APIキーは生成履歴に保存しません。API利用料金が発生する場合があります。'
$oldKey = $env:TYPESAFE_API_KEY
$oldAllow = $env:AIKIMI_JEV_ALLOW_CLOUD
$ptr = [IntPtr]::Zero
try {
    $env:TYPESAFE_API_KEY = $null
    $env:AIKIMI_JEV_ALLOW_CLOUD = $null
    $answer = Read-Host 'Jevモードの外部送信を許可しますか [y/N]'
    if ($answer -eq 'y') {
        $secure = Read-Host 'TYPESAFE_API_KEY（この起動セッションだけで使用）' -AsSecureString
        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        $env:TYPESAFE_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
        $env:AIKIMI_JEV_ALLOW_CLOUD = '1'
    }
    & .\aikimi-launch.bat
} finally {
    if ($ptr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    $env:TYPESAFE_API_KEY = $oldKey
    $env:AIKIMI_JEV_ALLOW_CLOUD = $oldAllow
}
