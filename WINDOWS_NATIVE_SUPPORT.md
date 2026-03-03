# Windows Native Support (Windows完結化)

## 概要

Open-P2P の推論サーバーを Windows 環境で **WSLなし**で動作させる（TCP化する）ための変更内容です。

また、統合対象である Recap（別リポジトリ）側の「Windowsで詰まりやすいポイント」（GStreamer/キャプチャ）についても、
Windows完結化の観点で最低限の注意点を追記します。

## 変更の背景

### 問題点
- 元の実装はUnix Domain Socket (UDS) を使用しており、Windowsでは直接使用できない
- Windows環境ではWSL経由でUDSに接続する必要があった
- Windows完結化のため、TCPソケットへの対応が必要

### 解決策
- Windows環境では自動的にTCPソケット（localhost）を使用
- Linux環境では従来通りUnix Domain Socketを使用（後方互換性維持）
- 環境変数で明示的にTCP/UDSを切り替え可能

## 変更内容

### P2P側（Open-P2P）の変更

#### 1. `elefant/inference/unix_socket_server.py`

**変更内容:**
- プラットフォーム判定の追加（`sys.platform == "win32"`）
- TCPソケット対応の追加（`asyncio.start_server()`）
- シグナルハンドリングのWindows対応（`SIGBREAK`の使用）
- 環境変数による接続タイプの切り替え（`USE_TCP=1`）

#### 追加（運用上の注意）

- Windows の asyncio(Proactor) 環境では、stdin を `connect_read_pipe()` で読む実装が不安定な場合があるため、
  **Windows ではターミナル入力リスナーを起動しない**方針にしています（推論/通信自体には影響しない）。

**技術的詳細:**
- Windows環境では自動的に`asyncio.start_server()`を使用
- Linux環境では従来通り`asyncio.start_unix_server()`を使用
- シグナルハンドリングはWindowsでは`signal.signal()`を使用（`loop.add_signal_handler()`は制限あり）

**なぜこれで動作するか:**
- Pythonの`asyncio`はWindowsでTCPソケットを完全サポート
- 通信プロトコル（バイナリプロトコル）は変更不要
- `StreamReader`/`StreamWriter`のインターフェースはUDS/TCPで共通

#### 2. `elefant/inference/script/inference_load_test.py`

**変更内容:**
- クライアント側の接続方法をTCP対応に変更
- プラットフォーム判定と環境変数サポートを追加

**技術的詳細:**
- `asyncio.open_unix_connection()` → `asyncio.open_connection()`（Windows時）
- 環境変数`USE_TCP`、`INFERENCE_HOST`、`INFERENCE_PORT`をサポート

### Recap側の変更

#### 1. `crates/wsl-tools/src/lib.rs`

**変更内容:**
- TCP接続メソッド`connect_tcp()`を追加
- `tokio::net::TcpStream`を使用したTCP接続を実装

**技術的詳細:**
```rust
pub async fn connect_tcp(host: &str, port: u16) -> Result<TcpStream, io::Error>
```

**なぜこれで動作するか:**
- `tokio::net::TcpStream`はWindowsで完全サポート
- `AsyncRead`/`AsyncWrite`トレイトを実装しており、既存コードと互換性がある

#### 2. `src/handler/capture/mod.rs`

**変更内容:**
- `send_inference_frames()`と`receive_inference_actions()`をジェネリクス化
- 環境変数による接続タイプの切り替えロジックを追加

**技術的詳細:**
- 関数をジェネリクスにして、`AsyncRead`/`AsyncWrite`を実装する任意の型を受け取れるように変更
- 環境変数`USE_TCP`、`INFERENCE_HOST`、`INFERENCE_PORT`をサポート

**なぜこれで動作するか:**
- Rustのジェネリクスにより、型安全性を保ちながら複数の型に対応可能
- `SocatStream`と`TcpStream`は両方とも`AsyncRead`/`AsyncWrite`を実装
- 既存の通信プロトコル（バイナリプロトコル）は変更不要

## 使用方法

### Windows環境での推論サーバー起動

```bash
# 環境変数を設定（オプション、デフォルト値が使用される）
$env:INFERENCE_HOST = "127.0.0.1"
$env:INFERENCE_PORT = "9999"

# 推論サーバーを起動（自動的にTCPモード）
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

> NOTE: まず動かす目的では `TORCHDYNAMO_DISABLE=1` を推奨（torch.compile回りの不安定さ回避）。

### Linux環境での推論サーバー起動（従来通り）

```bash
# Unix Domain Socketを使用（デフォルト）
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt

# または、明示的にTCPを使用
USE_TCP=1 INFERENCE_PORT=9999 uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

### Recap側の設定

```bash
# Windows環境でTCP接続を使用
$env:USE_TCP = "1"
$env:INFERENCE_HOST = "127.0.0.1"
$env:INFERENCE_PORT = "9999"

# Recapを起動
just trace
```

> NOTE: Recap の Windows 実行には MSYS2(UCRT64) の GStreamer が必要です。
> 依存関係と PATH の完全版は `DEPENDENCIES_WINDOWS.md` に集約しています。

## 環境変数

| 環境変数 | デフォルト値 | 説明 |
|---------|------------|------|
| `USE_TCP` | Windows: 自動、Linux: `0` | TCP接続を使用する場合は`1`に設定 |
| `INFERENCE_HOST` | `127.0.0.1` | TCP接続時のホスト名 |
| `INFERENCE_PORT` | `9999` | TCP接続時のポート番号 |

## 後方互換性

- Linux環境では従来通りUnix Domain Socketを使用（デフォルト動作）
- 既存のコードは変更不要（自動的に適切な接続方式を選択）
- 環境変数で明示的に切り替え可能

## パフォーマンスへの影響

- localhostでのTCP通信は、Unix Domain Socketと比較して数ミリ秒程度のレイテンシ増加が見込まれます
- 実用上は問題ない範囲ですが、実際の環境での検証を推奨します
- 既存の < 50ms エンドツーエンドレイテンシ要件を満たせる見込みです

## トラブルシューティング

### Windowsファイアウォールの設定

TCP接続を使用する場合、Windowsファイアウォールでlocalhostのポートを許可する必要があります：

```powershell
# ポート9999を許可（管理者権限が必要）
New-NetFirewallRule -DisplayName "Open-P2P Inference" -Direction Inbound -LocalPort 9999 -Protocol TCP -Action Allow
```

### ポートが既に使用されている場合

別のポート番号を指定：

```bash
$env:INFERENCE_PORT = "9998"
```

### 接続エラーの確認

- 推論サーバーが起動しているか確認
- ポート番号が一致しているか確認
- ファイアウォール設定を確認

#### Recap が UDS に繋いでしまう

Recap 側ログで `Connecting to Unix domain socket at /tmp/uds.recap` が出る場合、TCP化できていません。
`just trace` を起動する PowerShell セッションで以下が設定されているか確認してください。

```powershell
$env:USE_TCP = "1"
$env:INFERENCE_HOST = "127.0.0.1"
$env:INFERENCE_PORT = "9999"
```

---

## 既知の制限（Windows / PoC段階）

- **Recap のウィンドウ単位キャプチャは Windows+MSYS2 GStreamer では未対応な場合がある**
  - `d3d12screencapturesrc` の `window-handle` 等が MSYS2 ビルドに存在しないケースがあるため、
    デフォルトは「全画面キャプチャ」へフォールバックする実装を採用。
- **長時間実行で GStreamer ネイティブクラッシュが発生することがある**
  - `mp4mux`/タイムスタンプ周りで C++ 側が落ちることがあり、現状は根本解決は未対応。
  - まずは短いセッションでの検証を推奨。

## テスト

### P2P側のテスト

```bash
# 構文チェック
python -m py_compile elefant/inference/unix_socket_server.py
python -m py_compile elefant/inference/script/inference_load_test.py

# 実際のテスト（推論サーバーを起動してから）
uv run elefant/inference/script/inference_load_test.py
```

### Recap側のテスト

```bash
# ビルドチェック
cargo check --package wsl-tools
cargo check

# 実際のテスト（推論サーバーを起動してから）
just trace
```

## 依存関係

### P2P側
- Python 3.13.2
- 既存の依存関係（PyTorch、NumPy等）はWindows対応済み
- 追加の依存関係は不要

### Recap側
- Rust 1.88.0以上
- `tokio`（既存の依存関係）
- 追加の依存関係は不要

## 実装の詳細

### 型の統一（Recap側）

Recap側では、`SocatStream`（Unix Domain Socket用）と`TcpStream`（TCP用）の型を統一する必要がありました。

**解決方法:**
- 関数をジェネリクスにして、`AsyncRead`/`AsyncWrite`トレイトを実装する任意の型を受け取れるように変更
- これにより、型安全性を保ちながら、複数の接続方式に対応可能

**例:**
```rust
async fn send_inference_frames<W>(
    recv: Recv<Frame>,
    mut writer: W,
    ...
) -> Result<(), anyhow::Error>
where
    W: tokio::io::AsyncWrite + Unpin + Send,
{
    // 既存のロジックはそのまま
}
```

## 参考資料

- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [Tokio documentation](https://tokio.rs/)
- [Windows Socket Programming](https://learn.microsoft.com/en-us/windows/win32/winsock/windows-sockets-start-page-2)

