# Windows 環境向け elefant / open-p2p 修正まとめ

このドキュメントは、elefant (open-p2p) を Windows 上で推論実行できるよう行った修正をチームメンバーと共有するためのものです。

---

## ベースとなるリポジトリとバージョン

- **リポジトリ**: https://github.com/elefant-ai/open-p2p
- **推奨ベースコミット**: 修正適用前に次のコマンドでコミットハッシュを記録してください

```powershell
# 修正適用前のベースコミットを記録
git rev-parse HEAD
# 例: a329d98cbe62119679a254d71bea6446773541bc
```

- **2026年1月時点の最新**: `a329d98cbe62119679a254d71bea6446773541bc`（"update validation to solve issue 13"）
- **環境再現用クローン**:

```powershell
git clone https://github.com/elefant-ai/open-p2p.git
cd open-p2p
git checkout <ベースコミットハッシュ>  # 例: a329d98cbe62119679a254d71bea6446773541bc
```

---

## 修正の目的

元リポジトリは Linux を前提としており、以下の理由で Windows ではそのまま動作しません。

1. **elefant_rust**: Rust 製拡張で FFmpeg に依存。Windows では vcpkg や bindgen との互換性問題がある
2. **torchcodec**: Linux 専用の依存で、Windows ではインストール不可
3. **インポート連鎖**: config 読み込み時に `video_proto_dataset` → `zmq_queue` → `elefant_rust` がロードされ、推論にも関わらず失敗する

これらの問題を避けるため、Windows では elefant_rust を外し、依存のない経路で推論できるようにしています。

---

## 修正一覧

### 1. `pyproject.toml`

**変更内容**: `elefant_rust` を Windows ではビルドしないようにする

```diff
-    "elefant_rust",
+    "elefant_rust; sys_platform != 'win32'",
```

**補足**: `[tool.uv]` の `environments` で Windows AMD64 を有効にしている場合の例（既存プロジェクトに含まれている可能性あり）:

```toml
[tool.uv]
environments = [
  "sys_platform == 'win32' and (platform_machine == 'AMD64' or platform_machine == 'x86_64')",
  "sys_platform == 'linux' and platform_machine == 'x86_64'",
  "sys_platform == 'darwin'",
]
```

---

### 2. `elefant/data/dataset_config.py`（新規作成）

**目的**: `RandAugmentationConfig` と `VideoProtoDatasetConfig` を、`elefant_rust` / `zmq_queue` に依存しないモジュールとして分離する。

- `video_proto_dataset` は `zmq_queue` と `elefant_rust` に依存している
- `config.py` が `RandAugmentationConfig` を `elefant.data` からインポートすると、その経路で `video_proto_dataset` がロードされてしまう
- そこで config だけを別モジュールに切り出し、推論パスで `video_proto_dataset` を読まないようにする

**内容**: `video_proto_dataset.py` にあった `RandAugmentationConfig` と `VideoProtoDatasetConfig` をそのまま移す。

---

### 3. `elefant/data/video_proto_dataset.py`

**変更内容**: 上記 config を `dataset_config` からインポートするように変更

```diff
- class RandAugmentationConfig(ConfigBase):
-     ...
- class VideoProtoDatasetConfig(ConfigBase):
-     ...
+ from elefant.data.dataset_config import RandAugmentationConfig, VideoProtoDatasetConfig
```

---

### 4. `elefant/policy_model/config.py`

**変更内容**: `RandAugmentationConfig` などを `dataset_config` と `action_mapping` から直接インポート

```diff
- from elefant.data import (
-     RandAugmentationConfig,
-     UniversalAutoregressiveActionMappingConfig,
- )
+ from elefant.data.dataset_config import RandAugmentationConfig
+ from elefant.data.action_mapping import UniversalAutoregressiveActionMappingConfig
```

---

### 5. `elefant/data/__init__.py`

**変更内容**: 遅延インポートの参照先を `dataset_config` に変更

```diff
  "RandAugmentationConfig": (
-     "elefant.data.video_proto_dataset",
+     "elefant.data.dataset_config",
      "RandAugmentationConfig",
  ),
  "VideoProtoDatasetConfig": (
-     "elefant.data.video_proto_dataset",
+     "elefant.data.dataset_config",
      "VideoProtoDatasetConfig",
  ),
```

---

### 6. `elefant/data/rescale/resize.py`

**変更内容**: `elefant_rust` が無い環境では `torchvision` でリサイズするフォールバックを追加

- `_get_rust_module()` で `elefant_rust` のインポートを試行
- 失敗時は `F.resize()`（torchvision）でリサイズ

---

### 7. `elefant/data/rescale/rescale.py`

**変更内容**: `torchcodec` 未導入時もインポートが通るようにする

```diff
  try:
      from torchcodec.decoders import VideoDecoder
- except RuntimeError:
-     pass
+ except (RuntimeError, ImportError):
+     VideoDecoder = None
```

`ModuleNotFoundError` は `ImportError` のサブクラスなので、`ImportError` で捕捉。
`rescale_local_video` は推論では使わないため、`VideoDecoder = None` のままでも問題なし。

---

### 8. `elefant/policy_model/inference.py`

**変更内容**: `resize_image_for_model` のインポート元を、`video_proto_dataset` を経由しないモジュールに変更

```diff
- from elefant.data.video_proto_dataset import resize_image_for_model
+ from elefant.data.rescale.resize import resize_image_for_model
```

---

### 9. `elefant/policy_model/stage3_finetune.py`

**変更内容**: データセットクラスのインポートを関数内に移動して遅延ロード

- `ActionLabelVideoProtoDataset` / `ActionLabelVideoProtoDatasetConfig`: `_init_train_dataset()` 内
- `DummyDataset` / `DummyDatasetConfig`: `_init_dummy_dataset()` 内
- バリデーション用データセット: 対応するループ内

これにより、推論時（データセットを使わないとき）に `action_label_video_proto_dataset` や `dummy_dataset` がロードされず、結果として `video_proto_dataset` → `zmq_queue` → `elefant_rust` の経路も通らない。

---

### 10. `scripts/download_checkpoints.py`

**変更内容**: プロジェクト依存を外したスタンドアロンスクリプトに変更

- `huggingface_hub` の代わりに `requests` で直接ダウンロード
- `uv run --script` で実行可能（`uv run python scripts/download_checkpoints.py` でも可）
- `elefant_rust` などのビルドを避けられる

**実行例**:

```powershell
uv run --script scripts/download_checkpoints.py 150M
# または
uv run python scripts/download_checkpoints.py 150M
```

---

## 環境再現手順（Windows）

```powershell
# 1. クローン（ベースコミットを指定）
git clone https://github.com/elefant-ai/open-p2p.git
cd open-p2p
git checkout <ベースコミット>   # 上記で記録したハッシュ

# 2. 本ドキュメントに従い修正を適用、または修正済みブランチをチェックアウト

# 3. uv で依存インストール
uv sync

# 4. チェックポイント取得
uv run --script scripts/download_checkpoints.py 150M

# 5. 推論実行（google/embeddinggemma-300M の利用には Hugging Face ログインが必要）
$env:TORCHDYNAMO_DISABLE = "1"
uv run python elefant/policy_model/inference.py `
  --config checkpoints/150M/model_config.yaml `
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

---

## 注意事項

- **学習**: `elefant_rust` と `video_proto_dataset` は学習パイプラインで使われます。Windows では学習は想定していません（Linux 環境を推奨）。
- **テキストトークナイザー**: `google/embeddinggemma-300M` は gated モデルです。利用には Hugging Face で利用申請とログイン（`hf auth login` または `HF_TOKEN`）が必要です。
- **torchcodec**: 学習時のビデオリサイズ用で、推論には不要です。Windows では `pyproject.toml` で `torchcodec` が Linux 専用になっているため、インストールされません。

---

## 修正ファイル一覧（まとめ）

| ファイル | 種別 |
|---------|------|
| `pyproject.toml` | 修正 |
| `elefant/data/dataset_config.py` | 新規 |
| `elefant/data/video_proto_dataset.py` | 修正 |
| `elefant/data/__init__.py` | 修正 |
| `elefant/data/rescale/resize.py` | 修正 |
| `elefant/data/rescale/rescale.py` | 修正 |
| `elefant/policy_model/config.py` | 修正 |
| `elefant/policy_model/inference.py` | 修正 |
| `elefant/policy_model/stage3_finetune.py` | 修正 |
| `scripts/download_checkpoints.py` | 修正 |
