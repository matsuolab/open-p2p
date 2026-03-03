# Windows環境での依存関係インストール手順（Open-P2P 推論サーバ + Recap 統合）

## 概要

Windows環境で以下を「ネイティブ」で動かすためのセットアップ手順です。

- Open-P2P（`open-p2p`）: 推論サーバ（Python + PyTorch）
- Recap（`recap`）: 画面キャプチャ + 入力トレース（Rust + GStreamer）

この手順に従うことで、クリーンなWindows環境から再現可能にセットアップできます。

> NOTE:
> - Recap は別リポジトリですが、Windows で詰まりやすい依存関係（MSYS2/GStreamer・protoc・just）は共通して記録します。
> - 本ドキュメントは「現状動かすためのPoC手順」と「既知の制限」を含みます（長時間安定稼働は未解決）。

## 前提条件

- Windows 11（推奨）または Windows 10
- 管理者権限（一部のインストールで必要）
- インターネット接続

### リポジトリ配置（想定）

- `C:\Users\azureuser\open-p2p`
- `C:\Users\azureuser\recap`

---

## セットアップ手順（順番通りに実行）

### ステップ1: uvパッケージマネージャーのインストール

#### 1.1 インストール

```powershell
# PowerShellで実行（管理者権限不要）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 1.2 PATH環境変数の設定（永続化）

インストール後、`uv`コマンドが使えるようにPATHを設定します：

```powershell
# 現在のユーザーのPATHに追加（永続化）
$uvPath = "$env:USERPROFILE\.local\bin"
if ($env:Path -notlike "*$uvPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$env:Path;$uvPath", "User")
    $env:Path = "$env:Path;$uvPath"
}

# 確認
uv --version
```

**確認方法:**
```powershell
# 新しいPowerShellウィンドウを開いて確認
uv --version
```

---

### ステップ2: pyproject.tomlの確認・修正

#### 2.1 現在の設定確認

`open-p2p/pyproject.toml`の`[tool.uv]`セクションを確認：

```toml
[tool.uv]
environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'win32' and platform_machine == 'x86_64'",
    "sys_platform == 'win32' and platform_machine == 'AMD64'",  # ← これが必要
    "sys_platform == 'darwin'",
]
```

**重要:** Windows環境では`platform_machine`が`AMD64`と報告される場合があるため、`AMD64`の行が**必須**です。

#### 2.2 修正が必要な場合

もし`AMD64`の行が無い場合は追加してください：

```powershell
# pyproject.tomlを編集
notepad open-p2p\pyproject.toml
```

`[tool.uv]`セクションに以下を追加：
```toml
"sys_platform == 'win32' and platform_machine == 'AMD64'",
```

---

### ステップ3: Rust/Cargoのインストール

#### 3.1 インストール

```powershell
# wingetを使用（推奨）
winget install Rustlang.Rustup

# または、公式インストーラーを使用
# https://rustup.rs/ から rustup-init.exe をダウンロードして実行
```

#### 3.2 PATH環境変数の確認

インストール後、新しいPowerShellウィンドウで確認：

```powershell
# Rust/Cargoが使えるか確認
rustc --version
cargo --version
```

**もしコマンドが見つからない場合:**

```powershell
# Cargoのパスを確認（通常は以下）
$cargoPath = "$env:USERPROFILE\.cargo\bin"
if ($env:Path -notlike "*$cargoPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$env:Path;$cargoPath", "User")
    $env:Path = "$env:Path;$cargoPath"
}
```

---

### ステップ4: Visual Studio Build Toolsのインストール

#### 4.1 インストール

**必須:** 「C++によるデスクトップ開発」ワークロードをインストールします。

1. [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)をダウンロード
2. インストーラーを実行
3. **「C++によるデスクトップ開発」**ワークロードを選択してインストール

または、PowerShellで自動検出・インストール：

```powershell
# Visual Studio Installerのパスを確認
$vsInstaller = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vs_installer.exe"
if (Test-Path $vsInstaller) {
    Write-Host "Visual Studio Installer found: $vsInstaller"
} else {
    Write-Host "Visual Studio Installer not found. Please install Visual Studio Build Tools manually."
}
```

#### 4.2 MSVCリンカーのPATH設定

ビルドツールのインストール後、MSVCリンカー（`link.exe`）のパスを設定します：

```powershell
# MSVCリンカーのパスを自動検出
$vcToolsVersions = @("2022", "2019", "2017")
$found = $false

foreach ($version in $vcToolsVersions) {
    $possiblePaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\$version\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "${env:ProgramFiles}\Microsoft Visual Studio\$version\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\$version\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\$version\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64"
    )
    
    foreach ($pattern in $possiblePaths) {
        $paths = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Sort-Object -Descending
        if ($paths) {
            $linkerPath = $paths[0].FullName
            if (Test-Path "$linkerPath\link.exe") {
                Write-Host "Found MSVC linker at: $linkerPath"
                
                # PATHに追加（永続化）
                if ($env:Path -notlike "*$linkerPath*") {
                    [Environment]::SetEnvironmentVariable("Path", "$env:Path;$linkerPath", "User")
                    $env:Path = "$env:Path;$linkerPath"
                }
                
                $found = $true
                break
            }
        }
    }
    if ($found) { break }
}

if (-not $found) {
    Write-Host "MSVC linker not found. Please install 'Desktop development with C++' workload."
} else {
    # 確認
    where.exe link.exe
}
```

**確認方法:**
```powershell
# 新しいPowerShellウィンドウで確認
where.exe link.exe
# 出力例: C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64\link.exe
```

---

### ステップ5: vcpkgのインストールと設定

#### 5.1 vcpkgのインストール

```powershell
# vcpkgをクローン（Cドライブのルートに推奨）
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# ブートストラップ（初回ビルド）
.\bootstrap-vcpkg.bat
```

#### 5.2 vcpkgの環境変数設定（永続化）

```powershell
# VCPKG_ROOT環境変数を設定
$vcpkgRoot = "C:\vcpkg"
[Environment]::SetEnvironmentVariable("VCPKG_ROOT", $vcpkgRoot, "User")
$env:VCPKG_ROOT = $vcpkgRoot

# 確認
echo $env:VCPKG_ROOT
```

---

### ステップ6: FFmpeg 7.1.2のインストール（vcpkg経由）

#### 6.1 FFmpeg 7.1.2のインストール

`elefant-rust`が依存する`ffmpeg-sys-next 7.1.3`と互換性のあるFFmpeg 7.1.2をインストールします。  

```powershell
cd C:\vcpkg

# FFmpeg 7.1.2のコミットにチェックアウト
git checkout 34823ada10080ddca99b60e85f80f55e18a44eea

# FFmpegをインストール（時間がかかります：30分〜1時間）
.\vcpkg install ffmpeg:x64-windows

# インストール確認
.\vcpkg list ffmpeg
```

**重要:** このステップは時間がかかります（30分〜1時間）。FFmpegのビルドが完了するまで待ってください。

#### 6.2 FFmpegヘッダーの確認

```powershell
# avfft.hが存在するか確認（必須）
Test-Path "C:\vcpkg\installed\x64-windows\include\libavcodec\avfft.h"

# ライブラリファイルの確認
Test-Path "C:\vcpkg\installed\x64-windows\lib\avcodec.lib"
```

**期待される出力:** 両方とも`True`である必要があります。

---

### ステップ7: 環境変数の最終設定

#### 7.1 ビルド環境変数の設定

`elefant-rust`のビルド時にFFmpegを見つけられるように環境変数を設定します：

```powershell
# vcpkgのインストールパス
$vcpkgInstalled = "C:\vcpkg\installed\x64-windows"

# INCLUDE環境変数（ヘッダーファイルの検索パス）
$includePath = "$vcpkgInstalled\include"
if ($env:INCLUDE) {
    [Environment]::SetEnvironmentVariable("INCLUDE", "$env:INCLUDE;$includePath", "User")
    $env:INCLUDE = "$env:INCLUDE;$includePath"
} else {
    [Environment]::SetEnvironmentVariable("INCLUDE", $includePath, "User")
    $env:INCLUDE = $includePath
}

# LIB環境変数（ライブラリファイルの検索パス）
$libPath = "$vcpkgInstalled\lib"
if ($env:LIB) {
    [Environment]::SetEnvironmentVariable("LIB", "$env:LIB;$libPath", "User")
    $env:LIB = "$env:LIB;$libPath"
} else {
    [Environment]::SetEnvironmentVariable("LIB", $libPath, "User")
    $env:LIB = $libPath
}

# 確認
echo "INCLUDE: $env:INCLUDE"
echo "LIB: $env:LIB"
```

#### 7.2 競合する環境変数の削除

他のFFmpegインストール（winget等）の環境変数が競合する可能性があるため、削除します：

```powershell
# このセッションでのみ削除（永続的な設定は手動で削除）
Remove-Item Env:FFMPEG_DIR -ErrorAction SilentlyContinue
Remove-Item Env:PKG_CONFIG -ErrorAction SilentlyContinue
Remove-Item Env:PKG_CONFIG_PATH -ErrorAction SilentlyContinue
Remove-Item Env:PKG_CONFIG_LIBDIR -ErrorAction SilentlyContinue

# 永続的な設定を削除する場合（必要に応じて）
# [Environment]::SetEnvironmentVariable("FFMPEG_DIR", $null, "User")
```

---

### ステップ8: Python依存関係のインストール

#### 8.1 リポジトリに移動

```powershell
cd C:\Users\azureuser\open-p2p
```

#### 8.2 ロックファイルの生成

```powershell
# ロックファイルを生成（依存関係のバージョンを固定）
uv lock
```

**`uv lock`とは:**
- `pyproject.toml`を読み取り、依存関係のバージョンを解決
- `uv.lock`ファイルを生成（再現可能なビルドのため）

#### 8.3 依存関係のインストール

```powershell
# 依存関係をインストール（elefant-rustもビルドされる）
uv sync
```

**`uv sync`とは:**
- `uv.lock`を読み取り、指定されたバージョンのパッケージをインストール
- ローカルパッケージ（`elefant-rust`）もビルド
- 仮想環境（`.venv`）を自動管理

**このステップで行われること:**
1. Pythonパッケージのダウンロード・インストール
2. `elefant-rust`のRustビルド（FFmpegヘッダーが必要）
3. 仮想環境へのインストール

**エラーが発生した場合:**

```powershell
# .venvディレクトリを削除して再試行
Remove-Item -Recurse -Force .venv
uv sync
```

#### 8.4 インストール確認

```powershell
# PyTorchの確認
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# elefant-rustの確認
uv run python -c "import elefant_rust; print('elefant-rust imported successfully')"

# その他の主要パッケージ
uv run python -c "import numpy; import protobuf; print('Dependencies OK')"
```

---

## トラブルシューティング

### エラー: `uv lock`でプラットフォーム不一致

**症状:**
```
error: The current Python platform is not compatible with the lockfile's supported environments
```

**解決策:**
`pyproject.toml`の`[tool.uv]`セクションに`AMD64`の行があるか確認：

```toml
[tool.uv]
environments = [
    "sys_platform == 'win32' and platform_machine == 'AMD64'",  # ← これが必要
]
```

確認後、再度：
```powershell
uv lock
uv sync
```

---

### エラー: `linker 'link.exe' not found`

**症状:**
```
error: linker 'link.exe' not found
```

**解決策:**
1. Visual Studio Build Toolsがインストールされているか確認
2. 「C++によるデスクトップ開発」ワークロードがインストールされているか確認
3. ステップ4.2のPATH設定スクリプトを実行
4. 新しいPowerShellウィンドウで`where.exe link.exe`を実行して確認

---

### エラー: `avfft.h file not found`

**症状:**
```
fatal error: '/usr/include/libavcodec/avfft.h' file not found
```

**原因:**
- FFmpeg 8.0.1がインストールされている（`avfft.h`が削除された）
- `ffmpeg-sys-next 7.1.3`はFFmpeg 7.1.xを期待している

**解決策:**
1. vcpkgでFFmpeg 7.1.2をインストール（ステップ6を参照）
2. 環境変数`FFMPEG_DIR`が設定されていないか確認・削除
3. `INCLUDE`と`LIB`環境変数が正しく設定されているか確認

```powershell
# FFmpeg 7.1.2の確認
Test-Path "C:\vcpkg\installed\x64-windows\include\libavcodec\avfft.h"
# Trueである必要がある
```

---

### エラー: `.venv`のアクセス拒否

**症状:**
```
error: failed to remove file ...\lib64: アクセスが拒否されました
```

**解決策:**
```powershell
# .venvディレクトリを削除
Remove-Item -Recurse -Force .venv

# 再インストール
uv sync
```

---

### エラー: `uv sync`が非常に遅い

**原因:**
- `elefant-rust`のビルドに時間がかかる（初回）
- FFmpegのビルドに時間がかかる（vcpkg経由）

**対処:**
- 初回ビルドは30分〜1時間かかる場合があります
- 進行状況を確認しながら待機してください

---

## 環境変数の確認コマンド

セットアップ完了後、以下のコマンドで環境変数を確認できます：

```powershell
# すべての関連環境変数を確認
Write-Host "=== Environment Variables ==="
Write-Host "VCPKG_ROOT: $env:VCPKG_ROOT"
Write-Host "INCLUDE: $env:INCLUDE"
Write-Host "LIB: $env:LIB"
Write-Host "PATH (first 500 chars): $($env:Path.Substring(0, [Math]::Min(500, $env:Path.Length)))"

# ツールの確認
Write-Host "`n=== Tools ==="
Write-Host "uv: $(uv --version)"
Write-Host "rustc: $(rustc --version)"
Write-Host "cargo: $(cargo --version)"
where.exe link.exe | Select-Object -First 1

# FFmpegの確認
Write-Host "`n=== FFmpeg ==="
Test-Path "C:\vcpkg\installed\x64-windows\include\libavcodec\avfft.h"
Test-Path "C:\vcpkg\installed\x64-windows\lib\avcodec.lib"
```

---

## 次のステップ

依存関係のインストールが完了したら：

1. **推論サーバーの起動テスト**
   ```powershell
   cd C:\Users\azureuser\open-p2p
   uv run elefant/policy_model/inference.py `
     --config config/policy_model/150M.yaml `
     --use_random_weights
   ```

2. **テストクライアントでの接続確認**
   ```powershell
   # 別のPowerShellウィンドウで
   uv run python elefant/inference/script/inference_load_test.py
   ```

3. **Recapとの統合テスト**
   - 下記「Recap（Rust）セットアップ」を参照（または `WINDOWS_SETUP.md` / `WINDOWS_NATIVE_SUPPORT.md`）

詳細は `WINDOWS_SETUP.md` と `WINDOWS_NATIVE_SUPPORT.md` を参照してください。

---

## Recap（Rust）セットアップ（Windows）

> ここから先は `C:\Users\azureuser\recap` リポジトリ向けです。

### ステップR1: MSYS2（UCRT64）インストール

```powershell
winget install MSYS2.MSYS2
```

スタートメニューから **「MSYS2 UCRT64」** を起動して、以下を実行します。

```bash
# システム更新
pacman -Syu

# GStreamer + 基本ライブラリ
pacman -S \
  mingw-w64-ucrt-x86_64-pkg-config \
  mingw-w64-ucrt-x86_64-glib2 \
  mingw-w64-ucrt-x86_64-gstreamer \
  mingw-w64-ucrt-x86_64-gst-plugins-base

# Recap 実行に必要なプラグイン（要素不足で落ちるのを避ける）
pacman -S \
  mingw-w64-ucrt-x86_64-gst-plugins-good \
  mingw-w64-ucrt-x86_64-gst-plugins-bad \
  mingw-w64-ucrt-x86_64-gst-plugins-ugly \
  mingw-w64-ucrt-x86_64-gst-libav
```

### ステップR2: PowerShell 側の PATH / env（Recap ビルド＆実行用）

```powershell
# GStreamer / pkg-config（UCRT64）
$env:Path = "C:\msys64\ucrt64\bin;$env:Path"
$env:PKG_CONFIG_PATH = "C:\msys64\ucrt64\lib\pkgconfig"

# just が内部で使う sh.exe
$env:Path = "C:\msys64\usr\bin;$env:Path"
sh --version
```

### ステップR3: protoc（Protocol Buffers）

推奨: winget で入れる

```powershell
winget install Google.Protobuf
protoc --version
```

もし `protoc` が見つからない場合は、uv キャッシュ内の `protoc.exe` を PATH に追加して暫定対応できます。

```powershell
$protocDir = "C:\Users\azureuser\AppData\Local\uv\cache\archive-v0\-I76pR_xgdmQRd5psOPS7\torch\bin"
$env:Path = "$protocDir;$env:Path"
$env:PROTOC = "$protocDir\protoc.exe"
protoc --version
```

### ステップR4: Recap ビルド & 実行

```powershell
cd C:\Users\azureuser\recap

# 初回のみ
cargo build --release
cargo install just
```

Recap → 推論サーバ（TCP）接続用環境変数：

```powershell
$env:USE_TCP = "1"
$env:INFERENCE_HOST = "127.0.0.1"
$env:INFERENCE_PORT = "9999"
```

実行：

```powershell
cd C:\Users\azureuser\recap
just trace
```

---

## 推論サーバ実行（Windows / TCP）

推論サーバは別 PowerShell で起動し、Recap 実行中も起動し続けます。

```powershell
cd C:\Users\azureuser\open-p2p
$env:TORCHDYNAMO_DISABLE = "1"

uv run elefant/policy_model/inference.py `
  --config checkpoints/150M/model_config.yaml `
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

起動後に `Server started on TCP 127.0.0.1:9999` が出れば待機状態です。

---

## 変更点（今回の Windows 対応でコード側に入れたもの）

### Open-P2P（`elefant/policy_model/inference.py`）

- **Windows の stdin リスナーを抑止**（asyncio Proactor で `connect_read_pipe(sys.stdin)` が不安定なため）
- **起動時の FPS テスト（ダミー推論）を短縮**（待機状態に早く入るように）
- **TCP 待ち受け（127.0.0.1:9999）で Recap から接続**

### Recap

- **Windows の GStreamer で未提供な `d3d12screencapturesrc` プロパティをデフォルトで使わない**
  - デフォルトは「全画面キャプチャ」
  - 任意で試す場合のみ `RECAP_USE_WINDOW_CAPTURE_PROPS=1`
- **録画後チェック（GStreamer 再デコード）を Windows ではスキップ**
  - MSYS2/GStreamer のネイティブクラッシュ回避の応急処置

---

## 既知の問題 / 注意点（TODO）

- **推論が遅い（~1–2 FPS）**
  - `torch.compile` が Windows では安定せず、`TORCHDYNAMO_DISABLE=1` で回避しているため。
  - TODO: torch/関連最適化スタックのバージョン固定・再導入の検討。

- **長時間実行で Recap がクラッシュする場合がある**
  - MSYS2 UCRT64 の GStreamer（`mp4mux`/タイムスタンプ）周りでネイティブアサート → `STATUS_STACK_BUFFER_OVERRUN` を踏むことがある。
  - TODO: クリップ分割・別GStreamerビルド・Linux環境での再検証。

- **MP4 が標準プレイヤーで再生できない場合がある**
  - H.264 が `High 4:4:4` など互換性低いプロファイルになることがある。
  - 回避: ffmpeg で “コンテナだけ” 作り直す（再エンコードなし）

```powershell
ffmpeg -i "...\video.mp4" -c copy fixed_output.mp4
```

