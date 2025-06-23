# SIGNATE NEDO Challenge Motion Decoding Using Biosignals テーマ2 7th Solution

## プロジェクト構成

```bash
.
├── src/                          # メインソースコード
│   ├── data/                     # データ処理モジュール
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # データ前処理
│   │   └── dataset.py           # PyTorchデータセット
│   ├── models/                   # モデル定義
│   │   ├── __init__.py
│   │   ├── cwt.py               # Continuous Wavelet Transform
│   │   └── nedo_model.py        # メインモデル（ResNet + GRU）
│   ├── training/                 # 学習関連
│   │   ├── __init__.py
│   │   ├── trainer.py           # 学習ループ
│   │   └── loss.py              # 損失関数
│   ├── utils/                    # ユーティリティ
│   │   ├── __init__.py
│   │   ├── config.py            # 設定クラス
│   │   └── utils.py             # 汎用関数
│   └── main.py                   # メインエントリーポイント
├── scripts/                      # 実行スクリプト
│   ├── preprocess_data.py        # データ前処理スクリプト
│   ├── train_model.py            # 学習スクリプト
│   ├── evaluate_model.py         # 評価スクリプト
│   └── inference.py              # 推論スクリプト
├── input/                        # データディレクトリ
│   ├── .gitkeep                  # Git追跡用
│   ├── nedo-challenge-cwt-for-train/
│   ├── nedo-challenge-cwt-for-test/
│   └── nedo-challenge-cwt-for-ref/
├── pyproject.toml                # 依存関係管理
├── poetry.lock
├── run_pipeline.bat              # パイプライン実行スクリプト (Windows)
├── run_pipeline.sh               # パイプライン実行スクリプト (Linux/Mac)
└── README.md
```

## 使用方法

### 1. 環境セットアップ

```bash
# Poetry を使用して依存関係をインストール
poetry install

# データ配置
# input/ ディレクトリに以下のファイルを配置してください：
# - train.mat, test.mat, reference.mat
# - 前処理済みCWTファイル（train_cwt.pt, test_cwt.pt, ref_cwt.pt等）
```

### 2. データ前処理

```bash
# 学習データの前処理
python src/main.py preprocess --data_type train --data_dir ./input

# テストデータの前処理
python src/main.py preprocess --data_type test --data_dir ./input

# 参照データの前処理
python src/main.py preprocess --data_type reference --data_dir ./input

# テストデータのみの前処理（専用スクリプト）
python scripts/preprocess_test_data.py --input_dir ./input --output_dir ./input/nedo-challenge-cwt-for-test
```

### 3. モデル学習

```bash
# フルモデル学習（exp084相当）
python src/main.py train --mode full --data_dir ./input --max_epoch 20

# プレイヤーごとのモデル学習（exp088相当）
python src/main.py train --mode player_specific --data_dir ./input --pretrained_dir ./models --max_epoch 5
```

### 4. モデル評価

```bash
# 学習済みモデルの評価
python src/main.py evaluate --model_dir ./models --data_dir ./input
```

### 5. 推論・提出ファイル生成

```bash
# 参照データでの検証
python src/main.py inference --mode reference --model_dir ./models --data_dir ./input

# テストデータでの推論（提出ファイル生成）
python src/main.py inference --mode test --model_dir ./models --data_dir ./input --output_path submission.json

# 両方実行
python src/main.py inference --mode both --model_dir ./models --data_dir ./input --output_path submission.json
```

## アーキテクチャ

### モデル概要

- **バックボーン**: ResNet34d (timm)
- **時系列モデリング**: GRU
- **前処理**: Continuous Wavelet Transform (CWT)
- **データ拡張**: 左右反転 + Test Time Augmentation
- **アンサンブル**: 5 seeds × 4 players

### 学習戦略

1. **exp084**: 全プレイヤーデータでフルモデル学習（20エポック）
2. **exp088**: プレイヤーごとの追加学習（5エポック）

### 特徴

- **CWT変換**: 筋電位信号を時間-周波数スペクトログラムに変換
- **左右反転拡張**: ボードスタンス（regular/goofy）に応じた左右チャネル入れ替え
- **補助損失**: データ拡張識別タスクによる正則化
- **カスタム損失**: RMSE loss for 時系列予測

## データ配置

プロジェクトを使用する前に、以下のファイルを適切なディレクトリに配置してください：

### 基本データファイル

`input/` ディレクトリに以下のファイルを配置：

- `train.mat` - 学習データ
- `test.mat` - テストデータ  
- `reference.mat` - 参照データ

### 前処理済みCWTファイル

- `input/nedo-challenge-cwt-for-train/` に学習用CWTファイル
- `input/nedo-challenge-cwt-for-test/` にテスト用CWTファイル
- `input/nedo-challenge-cwt-for-ref/` に参照用CWTファイル

### ディレクトリ構造例

```bash
input/
├── train.mat
├── test.mat
├── reference.mat
├── nedo-challenge-cwt-for-train/
│   └── train_cwt.pt
├── nedo-challenge-cwt-for-test/
│   └── test_cwt.pt
└── nedo-challenge-cwt-for-ref/
    └── ref_cwt.pt
```
