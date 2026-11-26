#requires -Version 7.0
param(
    [ValidateSet('krea2', 'anima38', 'h3', 'h3-image', 'p-h3', 'qwen21', 'flux3', 'moss', 'irodori', 'yue2', 'clef')]
    [string]$Model,
    [switch]$DryRun,
    [switch]$Status,
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
        [Parameter(Mandatory)][ValidateSet('krea2', 'anima38', 'h3', 'h3-image', 'p-h3', 'qwen21', 'flux3', 'moss', 'irodori', 'yue2', 'clef')][string]$SelectedModel,
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

    $scripts = @{
        'h3' = 'tools\setup_minimax_h3.py'
        'h3-image' = 'tools\setup_minimax_h3_image.py'
        'p-h3' = 'tools\setup_p_h3_image.py'
        'qwen21' = 'tools\setup_qwen_image21.py'
        'flux3' = 'tools\setup_flux3_klein.py'
        'moss' = 'tools\setup_moss_image.py'
        'irodori' = 'tools\setup_irodori.py'
        'yue2' = 'modules_forge\yue2_studio\setup.py'
        'clef' = 'tools\setup_clef.py'
    }
    $modelScript = if ($scripts.ContainsKey($SelectedModel)) { $scripts[$SelectedModel] } else { 'tools\aikimi_setup.py' }
    if (-not (Test-Path -LiteralPath (Join-Path $RepositoryRoot $modelScript))) { throw "導入処理がありません: $modelScript" }
    $modelArguments = @('-B', (Join-Path $RepositoryRoot $modelScript))
    $modelPython = $python
    if (-not $scripts.ContainsKey($SelectedModel)) { $modelArguments += @('install', $SelectedModel) }
    if ($SelectedModel -eq 'yue2') {
        $modelPython = 'py'
        $modelArguments = @('-3.12', '-X', 'utf8') + $modelArguments
    }
    if ($PreserveSource) { $modelArguments += '--keep-source' }

    Write-Host "Aikimi Forge Neo / $SelectedModel"
    if ($PlanOnly) {
        Write-Host '実行予定（ダウンロード・変更なし）'
        Write-Host '1. Neo内のPython 3.13環境を準備'
        Write-Host '2. 通常の起動処理でGPUライブラリと本体の依存関係を準備（画面は起動しません）'
        Write-Host "3. $modelPython $($modelArguments -join ' ')"
        if ($SelectedModel -eq 'anima38') { Write-Host '   BF16版からINT8 ConvRotへ自動変換し、検証します。' }
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
        Invoke-SetupCommand $modelPython $modelArguments
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
        if ($Status) {
            $python = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
            $command = @('-B', '-X', 'utf8', (Join-Path $PSScriptRoot 'tools\aikimi_setup.py'), 'status')
            if (Test-Path -LiteralPath $python) { Invoke-SetupCommand $python $command }
            else { Invoke-SetupCommand 'py' (@('-3.13') + $command) }
            exit 0
        }
        if (-not $Model) {
            if ($NoPause) { throw '-Model または -Status を指定してください。' }
            $labels = [ordered]@{
                'krea2' = 'Krea2 Turbo INT8'
                'anima38' = 'Anima 3.8B v1.1 · BF16からINT8へ変換'
                'h3' = 'MiniMax H3 · 動画'
                'h3-image' = 'MiniMax-H3-Image · 画像'
                'p-h3' = 'P-H3 Image · 画像'
                'qwen21' = 'Qwen Image 2.1'
                'flux3' = 'FLUX 3 Image Klein'
                'moss' = 'MOSS Image'
                'irodori' = 'Irodori TTS · 読み上げ'
                'yue2' = 'YuE2 Music · 作曲'
                'clef' = 'Clef · 画像・文章の評価'
            }
            $choices = @($labels.Keys)
            Write-Host 'Aikimi Forge Neo モデルセットアップ'
            for ($i = 0; $i -lt $choices.Count; $i++) { Write-Host "$($i + 1)  $($labels[$choices[$i]])" }
            $selection = Read-Host '番号を選択（Enterで終了）'
            if (-not $selection) { exit 0 }
            $index = 0
            if (-not [int]::TryParse($selection, [ref]$index) -or $index -lt 1 -or $index -gt $choices.Count) {
                throw "1〜$($choices.Count)の番号を選択してください。"
            }
            $Model = $choices[$index - 1]
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
