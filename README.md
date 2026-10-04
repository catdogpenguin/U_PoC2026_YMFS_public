# イベント会場ブース配置最適化シミュレーション

イベント会場(調布市マルシェ)のブース・テーブル配置を、来場者エージェントの
人流シミュレーションと多目的最適化(NSGA-II)で探索する Python 実装。
「移動負担 `f_move`」と「混雑度 `f_congestion`」の 2 目的を同時に最小化する。

- リポジトリ: https://github.com/catdogpenguin/U_PoC2026_YMFS
- 想定 Python: 3.10 以上(開発・検証は 3.12)

このリポジトリには 2 つのモデルが入っている。

| フォルダ | 位置づけ | 会場 | 配置対象 | 最適化 |
| --- | --- | --- | --- | --- |
| `real_world_problem_0823revised/` | **現行版** | 実会場の多角形(57×98 セル、1 セル = 1m) | 46 オブジェクト(通常ブース 28・テーブル 18) | 直接配置 NSGA-II(120 次元) |
| `test_problem/` | 旧版(検証用) | 20×16 の矩形会場 | 6 ブース | pymoo 標準 NSGA-II |

新しく開発・実行するときは `real_world_problem_0823revised/` を使う。

---

## 目次

1. [ディレクトリ構成](#ディレクトリ構成)
2. [セットアップ](#セットアップ)
3. [使い方(現行版)](#使い方現行版)
4. [旧版 test_problem の使い方](#旧版-test_problem-の使い方)
5. [GitHub での開発フロー](#github-での開発フロー)
6. [仕様書からの主要な実装対応](#仕様書からの主要な実装対応)
7. [実装上の設計判断](#実装上の設計判断仕様書が明示しない部分)

---

## ディレクトリ構成

```text
U_PoC2026_YMFS/                          # リポジトリルート
├── README.md                            # 本ファイル
├── requirements.txt                     # 依存ライブラリ
├── .gitignore                           # 生成物・仮想環境・キャッシュの除外設定
│
├── real_world_problem_0823revised/      # 【現行版】実会場モデル
│   ├── README.md                        #   最適化 UI 設計書(UI の構成・検証手順の詳細)
│   ├── config/                          #   設定の一元管理(YAML)
│   │   ├── default_config.yaml          #     会場形状(広場・道路・障害物)・時間・容量・最適化設定
│   │   ├── booths.yaml                  #     46 オブジェクトの仕様と実会場での配置(fixed_position)
│   │   ├── visitor_profiles.yaml        #     来場者プロファイル(速度・混雑回避・待ち耐性・興味傾向)
│   │   ├── BOOTH_PARAMS_20260817.md     #     ブース入力値の決め方の根拠
│   │   └── CHANGES_20260817.md          #     2026-08-17 の仕様変更点(正面空間・待機密度など)
│   ├── fig/
│   │   └── table.png                    #   可視化で使うテーブルのアイコン
│   ├── src/                             #   ライブラリ本体
│   │   ├── models.py                    #     定数・dataclass・YAML ローダ
│   │   ├── venue.py                     #     会場グリッド構築(多角形形状)・配置適用
│   │   ├── rng.py                       #     乱数 3 系統分離と softmax 抽選
│   │   ├── routing.py                   #     BFS 距離場・混雑考慮 A*
│   │   ├── constraints.py               #     配置制約と違反量 g
│   │   ├── visitors.py                  #     来場者テーブルの決定的生成
│   │   ├── simulator.py                 #     人流シミュレーション本体
│   │   ├── metrics.py                   #     目的関数 f_move / f_congestion と KPI
│   │   ├── parallel.py                  #     1 配置評価の入口・プロセス並列
│   │   ├── direct_optimization.py       #     直接配置 NSGA-II(現行の最適化本体)
│   │   ├── optimization.py              #     pymoo Problem・代表解抽出・結果保存
│   │   └── visualization.py             #     レイアウト図・ヒートマップ・パレート図・GIF
│   ├── scripts/                         #   実行用 CLI
│   │   ├── run_optimization.py          #     直接配置 NSGA-II の実行と成果物出力
│   │   ├── run_fixed_layout.py          #     実会場の配置で 1 回だけシミュレーション(比較基準)
│   │   ├── visualize_solution.py        #     配置 JSON の可視化
│   │   ├── benchmark.py                 #     1 評価あたりの計算時間の測定
│   │   └── benchmark_parallel.py        #     並列数ごとの速度向上の測定
│   ├── tests/                           #   テスト
│   │   ├── config_20260629/             #     テスト専用の 20×16 矩形会場の設定
│   │   ├── conftest.py
│   │   ├── test_constraints.py          #     配置制約
│   │   ├── test_fixed_layout.py         #     実会場配置(fixed_position)の妥当性
│   │   ├── test_movement.py             #     セル容量・競合解決・目的関数の直交性
│   │   ├── test_queue.py                #     ブース体験・待ち行列・諦め
│   │   ├── test_reproducibility.py      #     再現性
│   │   ├── test_venue_shape.py          #     多角形会場の構築
│   │   └── preview_venue_shape.py       #     会場形状と固定配置を図で確認する補助スクリプト
│   ├── ui/                              #   Web UI(Streamlit)
│   │   ├── app.py                       #     画面本体
│   │   ├── optimize_worker.py           #     最適化を別プロセスで回すワーカー
│   │   ├── run_store.py                 #     runs/ 以下の読み書き
│   │   ├── render.py                    #     図の生成とキャッシュ
│   │   ├── smoke_test.py                #     画面を開かずに通しで動作確認
│   │   └── README.md
│   ├── results/                         #   CLI の出力先(生成物。.gitignore 対象)
│   └── runs/                            #   UI の実行ディレクトリ(生成物。.gitignore 対象)
│
└── test_problem/                        # 【旧版】20×16 矩形会場・6 ブースの検証用モデル
    ├── config/                          #   default_config.yaml / booths.yaml / visitor_profiles.yaml
    ├── src/                             #   現行版と同じ構成のライブラリ(UI・直接配置方式なし)
    ├── scripts/                         #   run_optimization.py / visualize_solution.py / benchmark.py
    └── tests/
```

補足:

- `config/default_config.yaml` の会場形状は、実会場の手描き概略図
  (`data/会場概略図/venue_abst.jpg`)をもとにしている。この画像はリポジトリには含めていない。
- `results/` と `runs/` は実行すると自動で作られる。コミットしない。

---

## セットアップ

### 1. クローン

```bash
git clone https://github.com/catdogpenguin/U_PoC2026_YMFS.git
cd U_PoC2026_YMFS
```

### 2. 仮想環境(`.venv`)の作成と有効化

プロジェクト直下に `.venv` を作る(`.gitignore` で除外済み)。

**Windows(PowerShell)**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

有効化されるとプロンプト先頭に `(.venv)` が表示される。抜けるときは `deactivate`。

### 3. 依存ライブラリのインストール

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit plotly        # Web UI を使う場合のみ(requirements.txt には未記載)
```

### 4. 動作確認

```bash
cd real_world_problem_0823revised
python -m pytest tests/ -q          # 全テストが通れば環境 OK
python scripts/benchmark.py         # 1 評価あたりの計算時間を測定
```

---

## 使い方(現行版)

以下はすべて `real_world_problem_0823revised/` に移動してから実行する。

```bash
cd real_world_problem_0823revised
```

### Web UI で実行する(おすすめ)

```bash
streamlit run ui/app.py
```

ブラウザが開いたら、サイドバーで会場・ブース・来場者数・世代数などを設定して
「最適化を開始」を押す。最適化は別プロセスで動くので、ブラウザを閉じても計算は続く。
実行結果は `runs/<YYYYMMDD_HHMMSS>/` に保存され、あとから開き直せる。

UI が動かないときは、まず画面なしで通し確認をする。

```bash
python -m ui.smoke_test                 # 矩形会場・縮小設定で一通り
python -m ui.smoke_test --skip-anim     # GIF 生成を飛ばして速く確認
```

画面構成・実行ディレクトリの中身・検証手順の詳細は
[`real_world_problem_0823revised/README.md`](real_world_problem_0823revised/README.md) を参照。

### コマンドラインで実行する

```bash
# 実会場で使われた配置(booths.yaml の fixed_position)を 1 回シミュレーション
python scripts/run_fixed_layout.py
python scripts/run_fixed_layout.py --animate --frame-interval 2   # GIF も出力

# 直接配置 NSGA-II(既定: 個体数 100・世代数 100)
python scripts/run_optimization.py

# 縮小設定での動作確認(個体数・世代数・並列数を指定)
python scripts/run_optimization.py --pop 8 --gen 2 --workers 4

# 配置 JSON の可視化(ヒートマップ・アニメーション GIF)
python scripts/visualize_solution.py results/knee_layout.json --animate

# 計算時間の測定
python scripts/benchmark.py
python scripts/benchmark_parallel.py --workers 1 2 4

# 会場形状と固定配置の確認図
python tests/preview_venue_shape.py
```

`run_optimization.py` の探索設定(突然変異率・整然さの上限など)は
`python scripts/run_optimization.py --help` で一覧できる。

成果物は `results/` に出力される。主なもの:
`pareto_solutions.csv`、`best_movement_layout.json`、`best_congestion_layout.json`、
`knee_layout.json`、`direct_*_designs.csv`、`direct_*_method.json`、
`mutation_history.csv`、`layout_diversity_history.csv` など。
共有したい結果は Issue や PR に個別に添付するか、`git add -f` で明示的に追加する。

実会場モデルは 1 評価が重い。いきなり既定設定で回さず、
まず `--pop 8 --gen 2` で 1 評価あたりの時間を測ってから世代数を決める。

パラメータ(会場形状・来場者数・ブース仕様・最適化設定など)はすべて `config/` の YAML で変更する。
コードに直書きしないこと。

---

## 旧版 test_problem の使い方

20×16 の矩形会場に 6 ブースを置く、初期の検証用モデル。現行版との結果比較や、
アルゴリズムの挙動を軽い設定で確かめたいときに使う。

```bash
cd test_problem
python -m pytest tests/ -q
python scripts/run_optimization.py --pop 16 --gen 8 --workers 4
```

> 2 つのフォルダはどちらも `src` という名前のパッケージを持つ。
> テストはリポジトリ直下からまとめて実行せず、必ずそれぞれのフォルダに移動して実行する。

---

## GitHub での開発フロー

標準的な **GitHub Flow**(`main` を常に動く状態に保ち、作業ごとにブランチを切って PR でマージ)を採用する。

### 0. 初回のみ: 認証と署名設定

HTTPS + [Personal Access Token](https://github.com/settings/tokens) または SSH 鍵を設定しておく。
コミットに使う名前とメールが未設定なら:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 1. main を最新化して作業ブランチを切る

ブランチ名は `種別/内容` の形式を推奨。

```bash
git switch main
git pull origin main
git switch -c feature/your-topic
```

| 接頭辞 | 用途 | 例 |
| --- | --- | --- |
| `feature/` | 新機能の追加 | `feature/queue-refactor` |
| `fix/` | バグ修正 | `fix/astar-deadlock` |
| `docs/` | 文書の修正 | `docs/update-readme` |
| `refactor/` | 挙動を変えないコード整理 | `refactor/simulation-class` |
| `test/` | テスト追加・修正 | `test/add-unit-tests` |
| `experiment/` | 実験的な変更 | `experiment/parallel-processing` |
| `chore/` | 設定・依存関係など | `chore/update-dependencies` |

### 2. 変更してコミット

```bash
git add <変更したファイル>          # 例: git add real_world_problem_0823revised/src/simulator.py
git status                          # results/・runs/・.bak などが混ざっていないか確認
git commit -m "待ち行列の前詰め処理を修正"
```

コミットメッセージは「何をしたか」が一目で分かるように簡潔に。1 コミット 1 論点を心がける。

> **フォルダごとコピーして版を分けない。** 以前の版は git の履歴に残るので、
> `xxx_0901revised/` のような複製フォルダや `*.bak` ファイルは作らず、同じフォルダを修正してコミットする。
> 節目の版には `git tag v0823` のようにタグを付けておくと、あとから戻りやすい。

### 3. リモートへ push

初回はブランチの upstream を設定する(`-u`):

```bash
git push -u origin feature/your-topic
```

2 回目以降は `git push` だけでよい。

### 4. プルリクエスト(PR)を GitHub 上で作成

1. ブラウザで https://github.com/catdogpenguin/U_PoC2026_YMFS を開く。
2. push 直後なら上部に表示される **「Compare & pull request」** をクリック。
   表示されない場合は **「Pull requests」タブ → 「New pull request」** で、
   base = `main`、compare = 自分のブランチ を選ぶ。
3. 本文に概要・変更内容・テスト結果を書く。
4. 必要ならレビュアーやラベルを指定して **「Create pull request」** をクリック。
5. レビュー指摘があれば同じブランチに追加コミットして `git push`。PR は自動で更新される。
6. 承認後、GitHub 上で **「Merge pull request」** を実行(スカッシュマージ推奨)。

### 5. マージ後の後片付け

```bash
git switch main
git pull origin main               # マージ結果を取り込む
git branch -d feature/your-topic   # ローカルの作業ブランチを削除
```

リモート側のブランチは、PR マージ時に GitHub の「Delete branch」ボタンで削除できる。

### コンフリクトが起きたら

作業中に `main` が進んで衝突した場合:

```bash
git switch feature/your-topic
git fetch origin
git merge origin/main              # 衝突ファイルを編集して解決
git add <解決したファイル>
git commit                         # マージコミットを作成
git push
```

### Windows での注意

WSL と Windows の間でファイルをコピーすると、`*:Zone.Identifier` という不要なファイルができることがある。
ファイル名に `:` を含むため、これがコミットされると Windows で `git switch` などが失敗する。
`git status` で見つけたらコミットに含めないこと。

---

## 仕様書からの主要な実装対応

- **乱数 3 系統分離**: `src/rng.py`。系統1=来場者生成 `[seed,0]`、
  系統2=来場者別意思決定 `[seed,1,vid]`、系統3=競合タイブレーク `[seed,2]`、
  初期集団サンプリング `[seed,3]`。softmax 抽選は累積和方式で 1 選択 =
  乱数 1 個消費に固定(`Generator.choice` の内部消費数に依存しない)。
  配置に依存しないため、ワーカーでの評価と UI での再評価は完全に一致する。
- **f_move の直交性**: `moved_cells`(競合解決で実際に移動したセル数)
  のみから自由歩行換算時間を算出。ブロック待機・行列待ち・入場遅延は含まない
  (`tests/test_movement.py::test_f_move_not_polluted_by_blocking` で検証)。
- **一括競合解決**: サブステップ開始時点の占有スナップショットで空きを
  判定するため、同一サブステップ内の退出は次サブステップから有効。優先順位は
  ブロック回数 → 残り滞在時間 → 系統3乱数。
- **待ち行列の実セル占有**: 待機者は先頭詰めで行列セルを占有し、
  先頭入場・離脱時に前詰めされる。
- **直接配置 NSGA-II**(現行版): 通常ブースは `(x, y, θ)`、方向を持たないテーブルは `(x, y)` を
  そのまま染色体にする(28×3 + 18×2 = 120 次元)。初期集団は固定配置を使わず、
  物理制約だけを満たすランダム配置から作る。詳細は
  [`real_world_problem_0823revised/README.md`](real_world_problem_0823revised/README.md) の「使用する最適化方式」を参照。

---

## 実装上の設計判断(仕様書が明示しない部分)

1. **行列参加のテレポート簡略化**: 受付セルに到達した来場者が行列に入るとき、
   受付セルから該当スロットセルへ直接移動する(この移動は `moved_cells` に
   数えない)。行列末尾まで歩く挙動は再現しない。
2. **待機列のセル容量**: 待機列セルは通路として使わず、1 セル(1m²)あたり 2 人まで
   割り当てる(2026-08-17 変更。`config/CHANGES_20260817.md`)。受付セルは到着・入場処理専用。
3. **ブース正面空間**: ブース正面の幅全体 × 奥行き 2 セルを物理制約として確保する。
   正面空間同士は共有できるが、他ブースの本体・受付・退出口・待機列は入れない。
4. **通路幅チェック**: 「各通行可能セルが少なくとも 1 つの w×w 全通行可能ブロックに属する」
   ことで判定し、違反セル数を違反量 g に加算する。w は `default_config.yaml` で指定する。
5. **move_credit の累積上限**: ブロック中の残高蓄積による後追い高速移動を
   防ぐため、上限を 1 ステップの最大移動量とした。
6. **行列参加時の諦め判定**: 満杯、または残り時間 <(推定待ち時間 +
   平均体験時間)のときに諦める。待機中は許容時間の厳密超過(>)で諦める。
7. **タイムアウト・無効配置のペナルティ**: 目的関数値 1.0e6 の定数。
   制約付き NSGA-II では違反量 g で劣後させるため、実質的な影響はない。
8. **体験成功率**: 総体験回数 / (総体験回数 + 総諦め回数) と定義。
9. **入退場**: 実会場では入口・出口を指定せず、グリッド境界に達する道路セル
   (領域外への開口部)から来場者が自由に到来・退出する。
   テスト用の矩形会場では、下辺中央の入口セルから出現する。
