# Windows完結化 - 変更内容サマリー

## 目的

Windows環境で WSL（Windows Subsystem for Linux）なしに、推論サーバーとクライアント（Recap）を接続して動かせるようにする。

## スコープ

- Open-P2P（サーバ）: UDS 前提から **TCP 対応**へ
- Recap（クライアント）: UDS/TCP 切り替え対応（環境変数）
- Windows 実行の現実的な制限（GStreamer 等）を把握し、PoC として動く状態まで持っていく

## 変更されたファイル

### P2P側（Open-P2P）

1. **`elefant/inference/unix_socket_server.py`**
   - TCPソケット対応を追加
   - プラットフォーム判定（Windows自動TCP化）
   - シグナルハンドリングのWindows対応

2. **`elefant/inference/script/inference_load_test.py`**
   - クライアント側のTCP接続対応
   - 環境変数サポート

3. **`README.md`**
   - Windows Native Supportセクションを追加

4. **`elefant/policy_model/inference.py`**
   - 起動時のダミー推論（FPS test）の調整（起動を短縮）
   - Windows の stdin リスナー起動回避（asyncio Proactor で不安定なため）

### Recap側

1. **`crates/wsl-tools/src/lib.rs`**
   - `connect_tcp()`メソッドを追加

2. **`src/handler/capture/mod.rs`**
   - `send_inference_frames()`と`receive_inference_actions()`をジェネリクス化
   - 環境変数による接続タイプの切り替えロジック

3. **`crates/recap_gst/src/srcs/windows.rs`**
   - Windows/MSYS2 の GStreamer で未提供な `d3d12screencapturesrc` プロパティに対するフォールバック（全画面キャプチャ）

4. **`crates/recap_gst/src/record_window.rs`**
   - GStreamer プロパティ不足時のフォールバック（最小構成での再試行）

5. **`src/handler/capture/on_finish_check.rs`**
   - Windows では録画後チェック（GStreamer 再デコード）をスキップ（ネイティブクラッシュ回避の応急処置）

## 技術的な変更点

### 接続方式の変更

**変更前:**
- Unix Domain Socket (`/tmp/uds.recap`) のみ
- WindowsではWSL経由で接続が必要

**変更後:**
- Windows: TCPソケット（`127.0.0.1:9999`）を自動使用
- Linux: Unix Domain Socket（デフォルト）またはTCP（環境変数で指定）

### 後方互換性

- Linux環境では既存の動作を維持（デフォルトでUnix Domain Socket）
- 既存のコードは変更不要（自動的に適切な接続方式を選択）
- 環境変数で明示的に切り替え可能


## ドキュメント

以下のドキュメントを作成しました：

- **`WINDOWS_NATIVE_SUPPORT.md`**: 技術的な変更内容の詳細
- **`WINDOWS_SETUP.md`**: Windows環境でのセットアップガイド
- **`DEPENDENCIES_WINDOWS.md`**: 依存関係とコマンドの説明

## 次のステップ

1. 実際のWindows環境で動作確認
2. レイテンシの測定と検証
3. 必要に応じてパフォーマンス最適化

## 既知の課題（TODO）

- **推論速度**
  - `torch.compile` 周りが Windows では不安定なため、`TORCHDYNAMO_DISABLE=1` で回避している（速度は低下）。
  - TODO: torch/最適化スタックのバージョン固定・段階的な再有効化。
- **Recap（Windows + MSYS2 GStreamer）の長時間安定性**
  - `mp4mux`/タイムスタンプ周りでネイティブクラッシュ（`STATUS_STACK_BUFFER_OVERRUN`）が発生する場合がある。
  - TODO: クリップ分割・別GStreamerビルド・Linux環境での再検証。

