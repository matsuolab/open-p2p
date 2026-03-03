# Windows環境でのセットアップガイド

## 概要

Windows環境で Open-P2P の推論サーバー（TCP）を実行し、Recap（別リポジトリ）から接続して動作確認するためのセットアップ手順です。
WSL（Windows Subsystem for Linux）は不要で、Windowsネイティブで動作します（ただしRecap側のキャプチャはMSYS2/GStreamer依存があります）。

## 前提条件

- Windows 11（推奨）または Windows 10
- NVIDIA GPU（推論実行用）
- Python 3.13.2
- Rust 1.88.0以上（Recap側のみ）

## セットアップ手順

### 1. Python環境のセットアップ

#### uvのインストール

```powershell
# PowerShellで実行
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

または、公式サイトからインストーラーをダウンロード：
https://github.com/astral-sh/uv

#### リポジトリのクローン

```powershell
git clone https://github.com/elefant-ai/open-p2p.git
cd open-p2p
```

### 2. 依存関係のインストール（Windows向けに lock 生成 → sync）

#### P2P側（Open-P2P）の依存関係

```powershell
# 重要: Windows では platform_machine が AMD64 と報告される場合があるため
# pyproject.toml の [tool.uv] environments に AMD64 が含まれていることを確認してから lock を作る
uv lock
uv sync
```

**注意:** 
- `pyproject.toml` の `tool.uv.environments` に Windows/AMD64 が入っていないと `uv lock` が失敗することがあります（詳細は `DEPENDENCIES_WINDOWS.md`）。
- 推論を「まず動かす」目的では、`torch.compile` 周りの問題回避として `TORCHDYNAMO_DISABLE=1` を使用します。

#### 依存関係の確認

```powershell
# インストールされたパッケージを確認
uv pip list

# PyTorchがWindows版でインストールされているか確認
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Hugging Face認証

```powershell
# Gemmaトークナイザーの認証（推論実行に必要）
uv run huggingface-cli login
```

### 4. モデルチェックポイントのダウンロード

```powershell
# 150Mモデルをダウンロード（例）
uv run python scripts/download_checkpoints.py 150M
```

### 5. Recap側のセットアップ（オプション）

Recapを使用する場合のみ必要です。

Recap は **別リポジトリ `C:\Users\azureuser\recap`** を使用します。
MSYS2(UCRT64)/GStreamer/protoc/just などの依存関係が必要で、手順は `DEPENDENCIES_WINDOWS.md` にまとめています。

## 実行方法

### 推論サーバーの起動

#### Windows環境（TCPモード、自動）

```powershell
# torch.compile 周りの回避（まず動かす目的）
$env:TORCHDYNAMO_DISABLE = "1"

# 環境変数の設定（オプション、デフォルト値が使用される）
$env:INFERENCE_HOST = "127.0.0.1"
$env:INFERENCE_PORT = "9999"

# 推論サーバーを起動
uv run elefant/policy_model/inference.py `
  --config checkpoints/150M/model_config.yaml `
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

#### Linux環境（Unix Domain Socket、デフォルト）

```bash
# 従来通りUnix Domain Socketを使用
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

#### Linux環境でTCPを使用（明示的）

```bash
# 環境変数でTCPを指定
USE_TCP=1 INFERENCE_PORT=9999 uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

### テストクライアントの実行

```powershell
# 推論サーバーを起動した状態で、別のターミナルで実行
uv run elefant/inference/script/inference_load_test.py
```

### Recapとの統合

```powershell
# Recap側で環境変数を設定
$env:USE_TCP = "1"
$env:INFERENCE_HOST = "127.0.0.1"
$env:INFERENCE_PORT = "9999"

# Recapを起動
cd C:\Users\azureuser\recap
just trace
```

## 環境変数

| 環境変数 | デフォルト値 | 説明 |
|---------|------------|------|
| `USE_TCP` | Windows: 自動、Linux: `0` | TCP接続を使用する場合は`1`に設定 |
| `INFERENCE_HOST` | `127.0.0.1` | TCP接続時のホスト名 |
| `INFERENCE_PORT` | `9999` | TCP接続時のポート番号 |

## トラブルシューティング

### 依存関係のインストールエラー

```powershell
# キャッシュをクリアして再インストール
uv cache clean
uv sync
```

### ポートが既に使用されている

```powershell
# 別のポートを指定
$env:INFERENCE_PORT = "9998"
```

### Windowsファイアウォールの設定

```powershell
# 管理者権限でPowerShellを実行
New-NetFirewallRule -DisplayName "Open-P2P Inference" -Direction Inbound -LocalPort 9999 -Protocol TCP -Action Allow
```

### PyTorchのCUDAサポート確認

```powershell
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

### 接続エラーの確認

1. 推論サーバーが起動しているか確認
2. ポート番号が一致しているか確認（デフォルト: 9999）
3. ファイアウォール設定を確認
4. ログを確認（`INFO`レベルで接続情報が表示されます）

**よくあるハマりどころ:**
- Recap 側ログに `Connecting to Unix domain socket at /tmp/uds.recap` が出る場合、`USE_TCP=1` が効いていません。
  - `just trace` を起動する PowerShell セッションで `USE_TCP/INFERENCE_HOST/INFERENCE_PORT` が設定されているか確認してください。

## コマンド一覧

### 基本的なコマンド

```powershell
# 依存関係のインストール
uv sync

# 推論サーバーの起動（Windows、TCP自動）
uv run elefant/policy_model/inference.py --config <config_path> --checkpoint_path <checkpoint_path>

# テストクライアントの実行
uv run elefant/inference/script/inference_load_test.py

# ランダムウェイトで動作確認（チェックポイント不要）
uv run elefant/policy_model/inference.py --config config/policy_model/150M.yaml --use_random_weights
```

### 環境変数の設定（PowerShell）

```powershell
# TCP接続を明示的に指定
$env:USE_TCP = "1"

# ホストとポートを指定
$env:INFERENCE_HOST = "127.0.0.1"
$env:INFERENCE_PORT = "9999"

# 環境変数の確認
echo $env:USE_TCP
echo $env:INFERENCE_HOST
echo $env:INFERENCE_PORT
```

### 環境変数の設定（CMD）

```cmd
set USE_TCP=1
set INFERENCE_HOST=127.0.0.1
set INFERENCE_PORT=9999
```

## パフォーマンス

- localhostでのTCP通信は、Unix Domain Socketと比較して数ミリ秒程度のレイテンシ増加が見込まれます
- 実用上は問題ない範囲ですが、実際の環境での検証を推奨します
- 既存の < 50ms エンドツーエンドレイテンシ要件を満たせる見込みです

## 次のステップ

1. 推論サーバーを起動して動作確認
2. Recapと統合してエンドツーエンドテスト
3. レイテンシの測定と検証

詳細は `WINDOWS_NATIVE_SUPPORT.md` を参照してください。

