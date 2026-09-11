#requires -Version 7.0
param(
    [ValidateSet('krea2', 'anima38', 'sensenova', 'h3')]
    [string]$Model,
    [switch]$DryRun,
    [switch]$KeepSource,
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-SetupCommand {
    param([string]$Executable, [string[]]$CommandArguments)
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "処理に失敗しました（終了コード $LASTEXITCODE）。上のエラーを確認してください。"
    }
}

function Assert-SetupIdle {
    param([string]$RepositoryRoot)
    $prefixes = @(
        (Join-Path $RepositoryRoot 'venv\'),
        (Join-Path $RepositoryRoot 'models\SenseNova-U1\worker-env\'),
        (Join-Path $RepositoryRoot 'repositories\minimax-h3\')
    )
    foreach ($process in Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'") {
        if (-not $process.ExecutablePath) { continue }
        foreach ($prefix in $prefixes) {
            if ($process.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Neoまたはモデル処理が起動中です。終了してからセットアップを実行してください。'
            }
        }
    }
}

function Invoke-AikimiModelSetup {
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][ValidateSet('krea2', 'anima38', 'sensenova', 'h3')][string]$SelectedModel,
        [switch]$PlanOnly,
        [switch]$PreserveSource
    )
    $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
    $SelectedModel = $SelectedModel.ToLowerInvariant()
    if ($PreserveSource -and $SelectedModel -ne 'anima38') {
        throw '-KeepSourceはAnimaの変換元を残す場合に指定してください。'
    }
    $python = Join-Path $RepositoryRoot 'venv\Scripts\python.exe'
    $environment = Join-Path $RepositoryRoot 'venv'
    $requirements = Join-Path $RepositoryRoot 'requirements.txt'
    foreach ($relative in @('launch.py', 'requirements.txt', 'tools\aikimi_setup.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $RepositoryRoot $relative) -PathType Leaf)) {
            throw "Neoのファイルがありません: $relative"
        }
    }
    if ((Test-Path -LiteralPath $environment) -and ((Get-Item -LiteralPath $environment -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Neoのvenvが外部フォルダーへのリンクになっています。通常のNeo内の環境を使用してください。'
    }
    # 起動時の認証設定検証に必要。版は本体のrequirementsを正本にする。
    $bootstrap = @(Get-Content -LiteralPath $requirements | Where-Object { $_ -match '^starlette==[^\s]+$' })
    if ($bootstrap.Count -ne 1) { throw 'requirements.txtのStarlette固定版を確認できません。' }

    $modelScript = if ($SelectedModel -eq 'h3') { 'tools\setup_minimax_h3.py' } else { 'tools\aikimi_setup.py' }
    if (-not (Test-Path -LiteralPath (Join-Path $RepositoryRoot $modelScript))) { throw "導入処理がありません: $modelScript" }
    if ($SelectedModel -eq 'sensenova' -and -not (Test-Path -LiteralPath (Join-Path $RepositoryRoot 'download_sensenova_u15_int8.ps1'))) {
        throw 'SenseNovaの環境準備処理がありません。'
    }
    $modelArguments = if ($SelectedModel -eq 'h3') {
        @('-B', (Join-Path $RepositoryRoot $modelScript))
    } else {
        @('-B', (Join-Path $RepositoryRoot $modelScript), 'install', $SelectedModel)
    }
    if ($PreserveSource) { $modelArguments += '--keep-source' }

    Write-Host "Aikimi Studio Neo / $SelectedModel"
    if ($PlanOnly) {
        Write-Host '実行予定（ダウンロード・変更なし）'
        Write-Host '1. Neo内のPython 3.13環境を準備'
        Write-Host '2. 通常の起動処理でGPUライブラリと本体の依存関係を準備（画面は起動しません）'
        Write-Host "3. $python $($modelArguments -join ' ')"
        if ($SelectedModel -eq 'anima38') { Write-Host '   BF16版からINT8 ConvRotへ自動変換し、検証します。' }
        if ($SelectedModel -eq 'sensenova') { Write-Host '4. SenseNova専用Pythonと固定版ライブラリを準備' }
        if ($SelectedModel -eq 'h3') { Write-Host '   ComfyUI・専用Python・標準INT8モデルをNeo内に準備します。' }
        return
    }

    Assert-SetupIdle $RepositoryRoot
    $uv = (Get-Command uv -CommandType Application -ErrorAction Stop).Source
    Get-Command git -CommandType Application -ErrorAction Stop | Out-Null
    Push-Location -LiteralPath $RepositoryRoot
    $savedArguments = [Environment]::GetEnvironmentVariable('COMMANDLINE_ARGS', 'Process')
    try {
        # 利用者の通常起動オプションは、この準備専用プロセスへ持ち込まない。
        [Environment]::SetEnvironmentVariable('COMMANDLINE_ARGS', '', 'Process')
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
            Invoke-SetupCommand $uv @('venv', '--python', '3.13', '--no-python-downloads', $environment)
        }
        Invoke-SetupCommand $python @('-I', '-c', 'import sys; assert sys.version_info[:2] == (3, 13), "Neo requires Python 3.13"')
        Write-Host 'Neoの実行環境を準備しています。'
        Invoke-SetupCommand $uv (@('pip', 'install', '--python', $python, '--index-url', 'https://pypi.org/simple') + $bootstrap)
        Invoke-SetupCommand $python @('-B', (Join-Path $RepositoryRoot 'launch.py'), '--exit', '--uv', '--bnb')

        Write-Host 'モデルを準備しています。既存ファイルは検証して再利用します。'
        Invoke-SetupCommand $python $modelArguments
        if ($SelectedModel -eq 'sensenova') {
            Write-Host 'SenseNova専用環境を準備しています。'
            Invoke-SetupCommand (Get-Command pwsh -CommandType Application -ErrorAction Stop).Source @(
                '-NoProfile', '-File', (Join-Path $RepositoryRoot 'download_sensenova_u15_int8.ps1'), '-RuntimeOnly'
            )
        }
        Write-Host '準備ができました。aikimi-launch.batからNeoを起動できます。'
    }
    finally {
        [Environment]::SetEnvironmentVariable('COMMANDLINE_ARGS', $savedArguments, 'Process')
        Pop-Location
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $exitCode = 0
    try {
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        if (-not $Model) {
            if ($NoPause) { throw '-Modelにkrea2 / anima38 / sensenova / h3を指定してください。' }
            Write-Host 'Aikimi Studio Neo モデルセットアップ'
            Write-Host '1  Krea2                    INT8配布版を取得'
            Write-Host '2  Anima 3.8B v1.1          BF16取得・INT8自動変換'
            Write-Host '3  SenseNova U1.5           INT8モデル・専用環境'
            Write-Host '4  MiniMax H3              INT8モデル・専用ComfyUI'
            $selection = Read-Host '番号を選択（Enterで終了）'
            if (-not $selection) { exit 0 }
            $choices = @{ '1'='krea2'; '2'='anima38'; '3'='sensenova'; '4'='h3' }
            if (-not $choices.ContainsKey($selection)) { throw '1〜4の番号を選択してください。' }
            $Model = $choices[$selection]
        }
        Invoke-AikimiModelSetup -RepositoryRoot $PSScriptRoot -SelectedModel $Model -PlanOnly:$DryRun -PreserveSource:$KeepSource
    }
    catch {
        Write-Host ("準備を完了できませんでした: " + $_.Exception.Message) -ForegroundColor Red
        $exitCode = 1
    }
    if (-not $NoPause) { Read-Host 'Enterで閉じる' | Out-Null }
    exit $exitCode
}
