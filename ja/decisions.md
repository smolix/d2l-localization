# Decisions log: 日本語 (ja)

## index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Introduction" (chapter title) → 「イントロダクション」 (distinguishes Chapter 1 from front-matter "Preface")
- "Preface" → 「まえがき」, "Acknowledgments" → 「謝辞」 (standard Japanese monograph front-matter terms)
- "First edition" / "Second edition" → 「初版」 / 「第 2 版」
- "post-training" → 「事後学習」 (paired with glossary `pretraining` = 「事前学習」 in LLM lifecycle contexts)
- "calculus" → 「微分積分」, "dynamics" (in `Mathematics for Deep Learning`) → 「ダイナミクス」
- "semi-supervised learning" → 「半教師あり学習」, "interpretability" → 「解釈可能性」

## chapter_preface/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Preface" → 「まえがき」 ("Preface to the Second Edition" → 「第 2 版へのまえがき」, "Preface to the First Edition" → 「初版へのまえがき」)
- "state space model" → 「状態空間モデル」
- "semi-supervised learning" → 「半教師あり学習」

## chapter_preface/index.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "from scratch" → 「ゼロから」 (e.g., ゼロから構築する実装)
- "generalization bound" → 「汎化限界」
- "serving" (model deployment) → 「サービング」

## chapter_preface/index.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Acknowledgments" → 「謝辞」

## chapter_installation/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "companion library" / "companion package" → 「本書付属ライブラリ」 / 「本書付属パッケージ」
- "hosted notebook" → 「ホスト型ノートブック」
- "lock file" → 「ロックファイル」
- "preamble" → 「プリアンブル」
- "virtual environment" → 「仮想環境」

## chapter_notation/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dynamics" (chapter topic, `chap_mdl-dynamics`) → 「ダイナミクス」
- "generative adversarial network" → 「敵対的生成ネットワーク」
- "Numerical Objects" (section heading) → 「数値オブジェクト」
- "indicator function" → 「指示関数」
- "Moore–Penrose pseudoinverse" → 「ムーア・ペンローズ擬似逆行列」
- "Hadamard (elementwise) product" → 「アダマール積（要素ごとの積）」
- "spectral (operator) norm" → 「スペクトルノルム（作用素ノルム）」
- "nuclear (trace) norm" → 「核ノルム（トレースノルム）」
- "divergence" (of a vector field, $\nabla \cdot \mathbf{f}$) → 「発散」 (distinguished from statistical 「ダイバージェンス」)
- "Laplacian" → 「ラプラシアン」
- "Pearson correlation coefficient" → 「ピアソン相関係数」
- "differential entropy" → 「微分エントロピー」
- "Kullback–Leibler (KL) divergence" → 「カルバック・ライブラー（KL）ダイバージェンス」

## chapter_notation/index.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "design matrix" → 「計画行列」
- "empirical loss" → 「経験損失」
- "attention scoring function" → 「アテンションスコア関数」
- "feedthrough (skip) term" → 「直達項（スキップ項）」
- "delta update rule" → 「デルタ更新則」
- "state-value function" → 「状態価値関数」
- "pairing (relativistic) objective" → 「ペアリング（相対論的）目的関数」
- "sampler" → 「サンプラ」 (follows `ja/style.md` §3.3 `-er` rule)
- "velocity field" → 「速度場」

## chapter_introduction/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Introduction" (chapter heading) → 「はじめに」
- "model family" → 「モデル族」
- "surrogate loss" → 「代理損失」
- "wake word" → 「ウェイクワード」

## chapter_introduction/index.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "sequence-to-sequence" (modifier) → 「系列対系列」（e.g. *sequence-to-sequence* problems → 「*系列対系列*の問題」, *sequence-to-sequence learning* → 「*系列対系列学習*」）
- "base model" vs "foundation model" → 「ベースモデル」 vs 「基盤モデル」
- "post-training" → 「事後学習」
- "credit assignment" → 「信頼割り当て」
- "partial observability" → 「部分観測性」
- "exploitation" (paired with exploration in RL) → 「活用」

## chapter_introduction/index.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Q-learning" → 「Q 学習」（`ja/style.md` §2.2 の半角英字と日本語の間のスペース規則に従う）
- "bitter lesson" → 「苦い教訓」
- "end-to-end" → 「エンドツーエンド」

## chapter_convolutional-neural-networks/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "locality" → 「局所性」（standard mathematical/CNN property paired with 平行移動同変性）
- "filter bank" → 「フィルタバンク」（follows `ja/style.md` §3.3 dropping final ー on フィルタ）
- "convnet" → 「畳み込みネットワーク」（concise counterpart to CNN / convolutional neural network）
- "vision transformer" → 「ビジョン Transformer」（follows `ja/style.md` §2.2 spacing rule between katakana and Latin script）
- "Resources and Further Reading" → 「参考資料と関連文献」（standard Japanese technical textbook heading）

## chapter_convolutional-neural-networks/why-conv.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "flatten" (a tensor/image into a vector) → 「平坦化する」 (standard Japanese deep learning term, matching Flatten layer = 平坦化層)
- "locality" → 「局所性」
- "equivariance" / "invariance" → 「同変性」 / 「不変性」, "translation invariance" / "translation invariant" → 「平行移動不変性」 / 「平行移動不変」 (paired with glossary `translation equivariance` = 「平行移動同変性」)
- "dense layer" → 「密結合層」 (distinguished from `fully connected layer` = 「全結合層」 where the source contrasts dense vs. sparse local connectivity)
- "grid" (pixel/spatial/hidden/infinite/periodic grid) → 「グリッド」
- "Where's Waldo" / "Waldo" → 「ウォーリーをさがせ！」 / 「ウォーリー」 (established Japanese title and character name for the franchise)
- "commuting diagram" → 「可換図式」
- "Toeplitz matrix" → 「テプリッツ行列」 (follows `ja/style.md` §5 katakana rule for mathematical terms named after people)
- "third-order tensor" / "fourth-order tensor" → 「3 階テンソル」 / 「4 階テンソル」
- "time-delay neural network" → 「時間遅れニューラルネットワーク」
- "sample efficiency" → 「サンプル効率」

## chapter_convolutional-neural-networks/why-conv.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "circulant matrix" → 「巡回行列」
- "rotation equivariance" → 「回転同変性」
- "spectrogram" → 「スペクトログラム」, "mel" → 「メル」

## chapter_convolutional-neural-networks/conv-layer.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "edge map" → 「エッジマップ」, "edge detection" / "edge detector" → 「エッジ検出」 / 「エッジ検出器」
- "locality" → 「局所性」
- "dense layer" → 「全結合層」 (matches `fully connected layer` in `sec_why-conv`)
- "filter bank" → 「フィルタバンク」 (follows `ja/style.md` §3.3 `-er` rule)

## chapter_convolutional-neural-networks/conv-layer.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "flatten" / "flattened" → 「平坦化する」 / 「平坦化した」
- "patch matrix" → 「パッチ行列」
- "banded matrix" → 「帯行列」, "Toeplitz" → 「テプリッツ」
- "theoretical receptive field" / "effective receptive field" → 「理論的受容野」 / 「有効受容野」

## chapter_convolutional-neural-networks/conv-layer.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "blur kernel" / "sharpening kernel" → 「ぼかしカーネル」 / 「先鋭化カーネル」
- "locally connected layer" → 「局所結合層」
- "centered finite-difference" → 「中心有限差分」

## chapter_convolutional-neural-networks/padding-and-strides.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dense prediction" → 「密な予測」
- "reflect padding" / "replicate padding" → 「反射パディング」 / 「複製パディング」
- "channels-last" → 「チャネルラスト形式」 (paired with 「チャネルラスト」/「チャネルファースト」 in `channels.md`)
- "tap" (of a convolutional kernel) → 「タップ」
- "multiply-add" → 「積和演算」 (consistent with `channels.md`)

## chapter_convolutional-neural-networks/padding-and-strides.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "patchify" → 「パッチ化」, "stem" (network stem) → 「ステム」
- "vision transformer" → 「ビジョン Transformer」 (with half-width space per `ja/style.md` §2.2)
- "effective kernel size" → 「実効カーネルサイズ」
- "gridding" (dilation artifact) → 「グリッディング」

## chapter_convolutional-neural-networks/padding-and-strides.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "mirror padding" → 「ミラーパディング」

## chapter_convolutional-neural-networks/channels.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "multi-channel" / "multi-input" / "multi-output" → 「多チャネル」 / 「多入力」 / 「多出力」
- "multiply-add" → 「積和演算」 (MAC in slides kept as `MAC` or `積和演算（MAC）`)
- "channels-first" / "channels-last" → 「チャネルファースト」 / 「チャネルラスト」
- "depthwise-separable convolution" / "depthwise convolution" → 「深さ方向分離可能畳み込み」 / 「深さ方向畳み込み」
- "luminance" / "isoluminant" / "color opponency" → 「輝度」 / 「等輝度」 / 「反対色」

## chapter_convolutional-neural-networks/channels.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "patch matrix" → 「パッチ行列」 (`im2col` kept in English)

## chapter_convolutional-neural-networks/pooling.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "shift tolerance" / "tolerance (to small shifts)" → 「シフト耐性」 / 「（小さなシフトに対する）耐性」 (distinguished from `robustness` = 「頑健性」 and `invariance` = 「不変性」)
- "shift-equivariant" / "shift equivariance" → 「シフト同変」 / 「シフト同変性」
- "downsampler" / "subsampler" → 「ダウンサンプラ」 / 「サブサンプラ」 (follows `ja/style.md` §3.3 `-er` rule and `decisions.md` `"sampler" → 「サンプラ」`)
- "deterministic reduction" → 「決定論的な縮約」

## chapter_convolutional-neural-networks/pooling.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "neutral element" → 「単位元」 (algebraic identity of a reduction operator such as $-\infty$ for $\max$)
- "channels-last layout" → 「チャネルラスト形式」
- "minimum pooling" → 「最小プーリング」, "softmax pooling" → 「ソフトマックスプーリング」
- "blur-pool" → 「ブラープール」, "two-tap box filter" → 「2 タップのボックスフィルタ」

## chapter_convolutional-neural-networks/lenet.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "convolutional encoder" → 「畳み込みエンコーダ」
- "dense classification head" / "dense head" / "dense block" (FC sense) → 「全結合分類ヘッド」 / 「全結合ヘッド」 / 「全結合ブロック」
- "flatten" / "flatten layer" → 「平坦化する」 / 「平坦化層」
- "channels-first" / "channels-last" → 「チャネルファースト」 / 「チャネルラスト」
- "multiply-adds" → 「積和演算」 (and count 「積和演算数」)
- "radial-basis-function (RBF)" → 「動径基底関数（RBF）」
- "support vector machine" → 「サポートベクターマシン」

## chapter_convolutional-neural-networks/lenet.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "plateau" → 「プラトー」
- "at chance" / "chance-level" → 「ランダム予測の水準」 / 「ランダム予測水準」
- "translation-equivariant" / "translation-invariant" → 「平行移動同変」 / 「平行移動不変」
- "shift tolerance" / "shift invariance" → 「シフト耐性」 / 「シフト不変性」
- "spatial locality" / "spatial adjacency" → 「空間的局所性」 / 「空間的隣接性」

## chapter_convolutional-modern/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Modern Convolutional Neural Networks" (chapter title) → 「現代の畳み込みニューラルネットワーク」
- "Vision Transformer" → 「Vision Transformer」 (proper architecture name kept in Latin script per `ja/style.md` §3.2, matching Transformer/ResNet/ConvNeXt; proposed glossary entry `ビジョンTransformer` would also conflict with §2.2 spacing)
- "convnet" / "efficient convnets" → 「畳み込みネットワーク」 / 「効率的な畳み込みネットワーク」
- "dense head" → 「全結合ヘッド」; "stem–body–head" → 「ステム・ボディ・ヘッド」
- "depthwise convolution" / "depthwise-separable convolution" → 「深さ方向畳み込み」 / 「深さ方向分離可能畳み込み」 (standard Japanese DL terminology)
- "inverted bottleneck" → 「逆ボトルネック」, "compound scaling" → 「複合スケーリング」, "squeeze-and-excitation gate" → 「Squeeze-and-Excitation ゲート」, "structural re-parameterization" → 「構造的再パラメータ化」
- "wall-clock time" → 「実経過時間」

## chapter_convolutional-modern/alexnet.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "top-5 error" → 「top-5 誤差」 (`ja/style.md` §2.2 の半角英数字と日本語の間のスペース規則に従う)
- "bags of visual words" → 「Bag-of-Visual-Words」
- "feature engineering" → 「特徴量エンジニアリング」
- "color-opponent blob" → 「反対色ブロブ」
- "multiply-add" → 「積和演算」 (スライド中の略記 `MAC` は `MAC` を維持)
- "local response normalization" → 「局所応答正規化」
- "He initialization" → 「He 初期化」 (`ja/style.md` §2.2 の半角英字と日本語の間のスペース規則に従う)
- "dense layer" / "dense head" → 「密結合層」 / 「密結合ヘッド」 (`fully connected layer` = 「全結合層」と訳し分け)
- "flatten" → 「平坦化する」 / 「平坦化」

## chapter_convolutional-modern/alexnet.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "half-cosine decay" → 「半コサイン減衰」 (glossary `cosine decay` = 「コサイン減衰」に対応)
- "validation accuracy" → 「検証正解率」
- "memory-bandwidth bound" → 「メモリ帯域幅律速」 (glossary `compute-bound` = 「計算律速」と対)

## chapter_convolutional-modern/blocks.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "branch" / "multi-branch" → 「ブランチ」 / 「マルチブランチ」
- "multiply-adds" → 「積和演算」, "MACs" → 「MAC」
- "dense head" → 「密なヘッド」, "dense layer" → 「全結合層」, "dense matrix" → 「密行列」
- "flatten" / "flattening" → 「平坦化」 (distinguished from smoothing = 「平滑化」)
- "stem / body / head" → 「ステム / ボディ / ヘッド」, "stem–body–head" → 「ステム・ボディ・ヘッド」, "trunk" → 「トランク」, "backbone" → 「バックボーン」
- "quarter-width VGG" → 「1/4 幅の VGG」（表では「VGG（1/4 幅）」）
- "peak learning rate" → 「ピーク学習率」

## chapter_convolutional-modern/blocks.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "translation-equivariant" → 「平行移動同変」
- "ablation" → 「アブレーション」

## chapter_convolutional-modern/blocks.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "auxiliary head" → 「補助ヘッド」

## chapter_convolutional-modern/batch-norm.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "group normalization" → 「グループ正規化」
- "scale invariance" → 「スケール不変性」
- "dense layer" → 「全結合層」 (synonymous with "fully connected layer" in Japanese DL texts)
- "pre-activation" → 「活性化前の値」 / 「活性化前の〜」
- "training mode" / "prediction mode" → 「訓練モード」 / 「予測モード」
- "running statistics" / "running estimates" / "running averages" → 「移動統計量」 / 「移動推定値」 / 「移動平均」
- "peak learning rate" → 「ピーク学習率」

## chapter_convolutional-modern/batch-norm.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "internal covariate shift" → 「内部共変量シフト」
- "warmup" (in learning-rate schedules) → 「ウォームアップ」
- "non-trainable buffer" → 「訓練対象外のバッファ」

## chapter_convolutional-modern/batch-norm.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "instance normalization" → 「インスタンス正規化」
- "normalizer-free network (NFNet)" → 「ノーマライザフリーネットワーク」（follows `ja/style.md` §3.3 `-er` rule）
- "adaptive gradient clipping" → 「適応的勾配クリッピング」

## chapter_convolutional-modern/resnet.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "degradation problem" / "degradation" → 「劣化問題」 / 「劣化」（ResNet の最適化失敗を指す定訳）
- "plain network" / "plain block" → 「プレーンネットワーク」 / 「プレーンブロック」（ショートカットなしの積層を `regular block`「通常のブロック」と区別して指す標準訳）
- "identity mapping" → 「恒等写像」
- "reparameterization" → 「再パラメータ化」
- "function class" → 「関数クラス」, "nested" (function classes) → 「入れ子」
- "shortcut connection" → 「ショートカット接続」, "projection shortcut" → 「射影ショートカット」

## chapter_convolutional-modern/resnet.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "He initialization" → 「He 初期化」（`ja/style.md` §2.2 の半角英字と日本語の間のスペース規則に従う）
- "stem" / "body" / "head" / "stage" → 「ステム」 / 「ボディ」 / 「ヘッド」 / 「ステージ」
- "multiply-adds" → 「積和演算」（回数としては「積和演算回数」）
- "cardinality" (ResNeXt) → 「カーディナリティ」

## chapter_convolutional-modern/resnet.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dense block" → 「密ブロック」, "growth rate" → 「成長率」, "transition layer" → 「遷移層」
- "pre-activation" / "post-activation" → 「プレアクティベーション」 / 「ポストアクティベーション」

## chapter_convolutional-modern/efficient-convnets.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Efficient Convnets" / "convnet" → 「効率的な畳み込みネットワーク」 / 「畳み込みネットワーク」
- "depthwise-separable convolution" → 「デプスワイズ分離可能畳み込み」 (paired with glossary `depthwise convolution` = 「デプスワイズ畳み込み」 and `pointwise convolution` = 「ポイントワイズ畳み込み」)
- "dense convolution" → 「密な畳み込み」 (contrasted with grouped/depthwise-separable convolutions)
- "multiply-add(s)" → 「積和演算」 (keeping `MAC` where slides use the abbreviation `MACs`)
- "inverted bottleneck" → 「逆ボトルネック」
- "compound scaling" → 「複合スケーリング」
- "structural re-parameterization" → 「構造的再パラメータ化」
- "memory-bandwidth bound" → 「メモリ帯域幅律速」
- "He initialization" → 「He 初期化」 (following `ja/style.md` §2.2 spacing)

## chapter_convolutional-modern/efficient-convnets.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "neural architecture search (NAS)" → 「ニューラルアーキテクチャ探索（neural architecture search、NAS）」
- "Squeeze-and-Excitation (SE) gate" → 「Squeeze-and-Excitation（SE）ゲート」 ("squeeze" → 「圧縮」, "excitation" → 「励起」)
- "BN folding" / "fold (into)" → 「BN の折り込み」 / 「折り込む」 (distinguished from `convolution` = 「畳み込み」 and `fusion` = 「融合」)
- "running statistics / running mean / running variance" → 「移動統計量 / 移動平均 / 移動分散」

## chapter_convolutional-modern/efficient-convnets.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "ShuffleNet" → 「ShuffleNet」, "width multiplier" → 「幅乗数」

## chapter_convolutional-modern/training-recipes.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "convnet" → 「畳み込みネットワーク」 (standard short form matching glossary `convolutional neural network` = 「畳み込みニューラルネットワーク」)
- "horizontal flip" → 「水平反転」, "random erasing" → 「ランダム消去」 (standard Japanese computer-vision data-augmentation terms)
- "step decay" → 「ステップ減衰」 (paired with glossary `cosine decay` = 「コサイン減衰」)
- "stochastic weight averaging (SWA)" → 「確率的重み平均化」 (paired with glossary `exponential moving average` = 「指数移動平均」)
- "shadow copy" / "shadow weights" → 「シャドウコピー」 / 「シャドウ重み」 (standard katakana in EMA literature)
- "running statistics" (in batch normalization) → 「移動統計量」
- "residual branch" → 「残差分岐」

## chapter_convolutional-modern/training-recipes.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "He initialization" → 「He 初期化」 (inserts half-width space between Latin name `He` and kanji per `ja/style.md` §2.2 and §3.2)
- "coupled weight decay" → 「結合型重み減衰」 (paired with glossary `decoupled weight decay` = 「分離型重み減衰」)
- "ablation" → 「アブレーション」 (standard katakana in ML literature)
- "backbone" → 「バックボーン」 (standard katakana in computer vision)

## chapter_convolutional-modern/convnext.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "convnet / convolutional network" → 「畳み込みネットワーク」
- "modernize / modernization" (of an architecture) → 「現代化する / 現代化」
- "vision transformer" → 「ビジョン Transformer」（`ja/style.md` §2.2 の半角英字と日本語の間のスペース規則に従う）
- "stem / body / head" (network topology) → 「ステム / ボディ / ヘッド」
- "patchify stem" → 「パッチ化ステム」（`*patchify* stem` → `*パッチ化*（patchify）ステム`）
- "inverted bottleneck" → 「逆ボトルネック」
- "multiply-add(s)" → 「積和演算」
- "effective receptive field" → 「有効受容野」
- "layer scale" → 「層スケール」（`ja/style.md` §7.1 の `layer` = 「層」に従う）
- "residual branch" → 「残差ブランチ」
- "patch-merging layer" → 「パッチ統合層」

## chapter_convolutional-modern/convnext.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "channels-first / channels-last" → 「チャネルファースト / チャネルラスト」
- "drop-path" → 「ドロップパス」
- "mixin" → 「ミックスイン」
- "submanifold sparse convolution" → 「サブマニホールドスパース畳み込み」

## chapter_convolutional-modern/convnext.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "masked autoencoder" → 「マスク付きオートエンコーダ」
- "sparse convolution" → 「スパース畳み込み」
- "deformable convolution" → 「変形可能畳み込み」
- "wall-clock time" → 「実時間」

## chapter_convolutional-modern/cnn-design.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "design space" → 「設計空間」
- "neural architecture search (NAS)" → 「ニューラルアーキテクチャ探索」
- "stem / body / head" (convnet macro-architecture) → 「ステム / ボディ / ヘッド」（複合語では「ステム・ボディ・ヘッド」）
- "stage" → 「ステージ」
- "bottleneck ratio" → 「ボトルネック比」, "group width" → 「グループ幅」, "group count" → 「グループ数」
- "inverted bottleneck" → 「逆ボトルネック」
- "coupled weight decay" → 「結合型重み減衰」（glossary の `decoupled weight decay` = 「分離型重み減衰」と対）
- "He initialization" → 「He 初期化」（`ja/style.md` §2.2 の半角英字と日本語の間のスペース規則に従う）
- "multiply-add" → 「積和演算」
- "quantized linear widths" → 「量子化線形幅」
- "stochastic dominance" → 「確率優位」

## chapter_convolutional-modern/cnn-design.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "empirical cumulative distribution function (CDF)" → 「経験累積分布関数」
- "linear ramp" → 「線形ランプ」
- "quantized linear staircase" → 「量子化線形階段」
- "cardinality" (ResNeXt parallel group count) → 「カーディナリティ」

## chapter_convolutional-modern/cnn-design.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Squeeze-and-Excitation gate" → 「Squeeze-and-Excitation ゲート」（固有名は英語維持、`ja/style.md` §3.2 に従う）
- "depthwise separability" → 「デプスワイズ分離構造」（glossary の `depthwise convolution` = 「デプスワイズ畳み込み」と整合）
- "expansion ratio" / "reduction ratio" → 「拡大比」 / 「縮小比」

## chapter_recurrent-neural-networks/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "*The Time Machine*" (dataset / novella title) → 「『The Time Machine』」 (follows `ja/style.md` §2.1 book-title rule; kept in Latin script as it serves as the English text dataset throughout Chapter 8)
- "bits per byte" / "bits-per-byte" → 「バイトあたりビット数」 (unit symbol `bpb` kept in Latin script)
- "pre-tokenization" / "pre-tokenized" → 「事前トークン化」 / 「事前トークン化を伴う」
- "truncated BPTT" (truncated backpropagation through time) → 「打ち切り通時的誤差逆伝播法」（「打ち切り BPTT」）
- "truncated sampling" / "truncated sampler" / "tail truncation" → 「打ち切りサンプリング」 / 「打ち切りサンプラ」 / 「裾の打ち切り」
- "rollout" → 「ロールアウト」

## chapter_recurrent-neural-networks/sequence.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "time series" → 「時系列」
- "prefix" (of a sequence) → 「プレフィックス」
- "fixed window" / "window length" → 「固定ウィンドウ」 / 「ウィンドウ長」
- "latent summary" / "latent autoregressive model" → 「潜在要約」 / 「潜在自己回帰モデル」
- "horizon" (memory or forecast horizon) → 「ホライズン」
- "Markov model" / "Markov condition" / "$m$-th order Markov model" → 「マルコフモデル」 / 「マルコフ条件」 / 「$m$ 次マルコフモデル」
- "stationarity" / "stationary" / "time-homogeneous" → 「定常性」 / 「定常」 / 「時間斉次」
- "multistep" / "rollout" → 「複数ステップ」 / 「ロールアウト」

## chapter_recurrent-neural-networks/sequence.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "lag" / "lag feature" → 「ラグ」 / 「ラグ特徴量」
- "moving average" → 「移動平均」
- "phase lag" / "phase delay" → 「位相遅れ」

## chapter_recurrent-neural-networks/sequence.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dilated lag" → 「膨張ラグ」 (parallel to "dilated convolution" = 「膨張畳み込み」)
- "regime switching" → 「レジームスイッチング」

## chapter_recurrent-neural-networks/text-sequence.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "*The Time Machine*" (novella title) → 「『The Time Machine』」 (follows `ja/style.md` §2.1 book-title convention)
- "pre-tokenization" → 「事前トークン化」 (paired with glossary `tokenization` = 「トークン化」)
- "production tokenizer" / "production vocabulary" → 「実用トークナイザ」 / 「実用語彙」
- "merge tree" → 「マージ木」

## chapter_recurrent-neural-networks/text-sequence.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "fertility" (tokenizer tokens-per-word metric) → 「肥沃度」 (glossed on first occurrence as `*肥沃度*（fertility）`)
- "glitch token" → 「グリッチトークン」
- "prompt injection" → 「プロンプトインジェクション」
- "special token" → 「特殊トークン」

## chapter_recurrent-neural-networks/language-model.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "$n$-gram" / "n-gram" → 「$n$ グラム」 / 「n グラム」（`ja/style.md` §2.2 の半角英数・数式と日本語の間の半角スペース規則に従う）
- "unigram" / "bigram" / "trigram" → 「ユニグラム」 / 「バイグラム」 / 「トライグラム」
- "Zipf's law" → 「ジップの法則」（人名単体の Zipf は `ja/style.md` §5 に従いラテン文字維持）
- "smoothing" → 「平滑化」、"Laplace smoothing" → 「ラプラス平滑化」、"Kneser–Ney smoothing" → 「Kneser–Ney 平滑化」
- "bits per byte" → 「バイトあたりビット数」
- "effective branching factor" → 「実効分岐数」
- "held-out" → 「ホールドアウト」
- "singleton" → 「シングルトン」

## chapter_recurrent-neural-networks/language-model.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "absolute discounting" → 「絶対ディスカウンティング」
- "interpolation" (in smoothing) → 「補間」
- "continuation count" → 「継続カウント」
- "backoff" / "back off" → 「バックオフ」 / 「バックオフする」
- "arithmetic coder" → 「算術符号器」

## chapter_recurrent-neural-networks/language-model.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "data leakage" / "leakage" → 「データリーク」 / 「リーク」

## chapter_recurrent-neural-networks/rnn.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "recurrent cell" → 「リカレントセル」, "recurrent layer" → 「リカレント層」, "recurrent weight (matrix)" → 「リカレント重み（行列）」, "recurrent connection" → 「リカレント結合」 (matches `recurrent neural network` = 「リカレントニューラルネットワーク」; `recurrent state` uses glossary 「再帰状態」)
- "unroll (across time)" → 「（時間方向に）展開する」
- "lossy (bottleneck / compression / summary)" → 「非可逆な（ボトルネック / 圧縮 / 要約）」
- "multiply-add (MAC)" → 「積和演算（MAC）」
- "embedding lookup" → 「埋め込み参照」
- "trigram / bigram" → 「トライグラム / バイグラム」 (spelled-out English names use standard Japanese NLP katakana; numeric forms `3-gram`, `4-gram`, `$n$-gram` stay numeric)

## chapter_recurrent-neural-networks/rnn.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "bits per byte" → 「バイトあたりビット数」（数値と伴う場合は「バイトあたり 2.1 ビット」、スライドの略記 `bpb` はそのまま保持）
- "Kneser–Ney smoothing" → 「Kneser–Ney 平滑化」（`ja/style.md` §5 に従い人名はラテン文字のまま保持）
- "effective context horizon" → 「実効コンテキスト範囲」
- "plain model / plain recurrent cell" → 「単純なモデル / 単純なリカレントセル」
- "initializer" → 「初期化子」

## chapter_recurrent-neural-networks/rnn.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "warm up" → 「ウォームアップする」 ("prefix warm-up loop" → 「プレフィックスのウォームアップループ」)
- "vanilla RNN" → 「バニラ RNN」
- "fused kernel" → 「融合カーネル」
- "threshold activation" → 「しきい値活性化関数」

## chapter_recurrent-neural-networks/bptt.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "state Jacobian" → 「状態ヤコビ行列」 (paired with binding glossary `Jacobian` = 「ヤコビ行列」)
- "unroll" (a recurrence / RNN over time) → 「展開する」（時間方向に展開する）
- "detach" / "detachment" (from computational graph) → 「切り離す」 / 「切り離し」
- "vanilla RNN" → 「素の RNN」
- "global gradient clipping" → 「大域的勾配クリッピング」
- "structured linear recurrence" → 「構造化線形漸化式」 (extends glossary `linear recurrence` = 「線形漸化式」)
- "horizon" (gradient / memory / learning / credit-assignment horizon) → 「ホライズン」
- "adjoint" (in reverse-mode automatic differentiation) → 「随伴変数」

## chapter_recurrent-neural-networks/bptt.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "detach" / "detachment" (from computational graph) → 「切り離す」 / 「切り離し」
- "vanilla RNN" → 「素の RNN」
- "horizon" (learning / memory / gradient horizon) → 「ホライズン」

## chapter_recurrent-neural-networks/decoding.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "decoding" (noun) → 「デコーディング」 (aligned with glossary `greedy decoding` = 「貪欲デコーディング」 and `autoregressive decoding` = 「自己回帰デコーディング」; verb "decode" → 「デコードする」)
- "open-ended generation" / "open-ended text" → 「オープンエンドな生成」 / 「オープンエンドなテキスト」
- "temperature scaling" → 「温度スケーリング」
- "truncation" / "tail truncation" → 「打ち切り」 / 「裾の打ち切り」 ("unreliable tail" → 「信頼できない裾」)
- "nucleus sampling" → 「ニュークリアスサンプリング」
- "neural text degeneration" → 「ニューラルテキスト退化」
- "myopia" (of greedy decoding) → 「近視眼性」 ("myopic" → 「近視眼的」)
- "exhaustive search" → 「全探索」

## chapter_recurrent-neural-networks/decoding.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "cutoff" → 「カットオフ」
- "support" (of a probability distribution) → 「台」

## chapter_recurrent-neural-networks/decoding.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "teacher forcing" / "teacher-forced training" → 「教師強制」 / 「教師強制による訓練」
- "bits-per-byte" → 「バイトあたりビット数」
- "stateless" / "stateful" → 「ステートレス」 / 「ステートフル」
