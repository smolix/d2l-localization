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

- "Introduction" (chapter heading) → 「イントロダクション」
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

## chapter_basics/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Introduction" (Chapter 1) → 「イントロダクション」 (distinguishes Chapter 1 from the Preface)
- "from scratch" / "concisely" (the recurring two-stage implementation pair) → 「ゼロから」 / 「簡潔に」
- "*The Time Machine*" → 『The Time Machine』 (book title enclosed in 『』 per ja/style.md §2.1 while keeping the English title of the text dataset)

## chapter_preliminaries/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Preliminaries" (chapter title) → 「準備」 (standard chapter title in Japanese mathematics and ML textbooks)
- "calculus" → 「微分積分」; "differential calculus" → 「微分法」 (standard Japanese university curriculum terminology)
- "data preprocessing" → 「データ前処理」 (standard Japanese ML terminology, matching `pandas.md`)
- "autograd" (shorthand for automatic differentiation) → 「自動微分」 (aligned with glossary entry for automatic differentiation)
- "foundation model" → 「基盤モデル」 (established Japanese ML term)
- "Vector–Jacobian Product" → 「ベクトル・ヤコビ行列積」 (aligned with glossary entries ベクトル and ヤコビ行列)

## chapter_preliminaries/ndarray.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Data Manipulation" → 「データ操作」 (section title and recurring concept)
- "axis" / "order" (of a tensor) → 「軸」 / 「階数」 ($k$ 階テンソル)
- "reshaping" / "flatten" / "singleton axis" → 「形状変換」 / 「平坦化」 / 「単一軸」
- "indexing and slicing" / "gather" → 「インデックス参照とスライス」 / 「ギャザー」
- "row-major order" → 「行優先順」
- "mutable" / "immutable" / "in-place" / "view" → 「ミュータブル」 / 「イミュータブル」 / 「インプレース」 / 「ビュー」

## chapter_preliminaries/ndarray.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "elementwise" → 「要素ごとの」 / 「要素ごとに」
- "type promotion" → 「型の昇格」
- "reduction" (along tensor axes) → 「縮約」
- "rebinding" → 「再束縛」

## chapter_preliminaries/pandas.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "preprocessing" → 「前処理」
- "target" (supervised learning target column/tensor) → 「ターゲット」
- "missing value" → 「欠損値」
- "imputation" → 「補完」
- "missingness indicator" → 「欠損性インジケータ」 (dropping final ー per style.md §3.3)
- "categorical feature" → 「カテゴリ特徴量」 (dropping final ー on カテゴリ per style.md §3.3)
- "data leakage" → 「データリーケージ」

## chapter_preliminaries/pandas.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "standardization" → 「標準化」, "z-score normalization" → 「z スコア正規化」
- "design matrix" → 「計画行列」

## chapter_preliminaries/linear-algebra.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "order" (of a tensor, `ndim`) → 「階数」 (`$k$-th order` → 「$k$ 階」, "higher-order tensor" → 「高階テンソル」)
- "dimensionality" (of a vector) → 「次元数」 (distinguished from informal "dimension" 「次元」 and tensor "order" 「階数」)
- "reduction" / "axis reduction" → 「リダクション」 / 「軸方向のリダクション」 (distinguished from "aggregate" 「集約する」 and tensor contraction 「縮約」)
- "elementwise" → 「要素ごとの」
- "flatten" → 「平坦化する」
- "matrix--vector product" / "matrix--matrix product" → 「行列・ベクトル積」 / 「行列・行列積」
- "design matrix" → 「計画行列」
- "shear" (linear map) → 「せん断」

## chapter_preliminaries/linear-algebra.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Cauchy--Schwarz inequality" → 「コーシー・シュワルツの不等式」
- "Manhattan distance" → 「マンハッタン距離」
- "spectral theorem" → 「スペクトル定理」
- "dominant eigenvector" → 「主固有ベクトル」
- "orthonormal eigenbasis" → 「正規直交固有基底」

## chapter_preliminaries/linear-algebra.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "gradient clipping" → 「勾配クリッピング」

## chapter_preliminaries/calculus.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Calculus" (section heading) → 「微分積分」; "integral calculus" / "differential calculus" → 「積分法」 / 「微分法」; "matrix calculus" → 「行列微分」
- "difference quotient" → 「差分商」
- "secant line" / "tangent line" → 「割線」 / 「接線」
- "surrogate loss" → 「代理損失」
- "0–1 error" / "0–1 accuracy" → 「0-1 誤差」 / 「0-1 正解率」
- "first-order" → 「一次」（一次線形近似、一次テイラー展開、一次最適化；二次関数・二次形式と統一）
- "catastrophic cancellation" → 「桁落ち」
- "machine epsilon" → 「計算機イプシロン」
- "gradient checking" → 「勾配チェック」
- "steepest ascent" / "steepest descent" → 「最急上昇」 / 「最急降下」
- "level set" / "level curve" → 「等位集合」 / 「等高線」
- "Cauchy–Schwarz" → 「コーシー・シュワルツの不等式」
- "adjoint" (reverse-mode AD) → 「随伴変数」

## chapter_preliminaries/calculus.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "forward difference" / "central difference" → 「前進差分」 / 「中心差分」
- "Vector–Jacobian Product (VJP)" → 「ベクトル・ヤコビ行列積（VJP）」

## chapter_preliminaries/autograd.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "detach / detaching" (computational graph) → 「切り離す / 切り離し」 (established Japanese term for detaching nodes/subgraphs from a computational graph)
- "gradient tracking" → 「勾配追跡」
- "sweep" (forward/backward/reverse sweep) → 「スイープ」（順方向スイープ / 逆方向スイープ）
- "vector–Jacobian product (VJP)" / "Jacobian–vector product (JVP)" → 「ベクトル・ヤコビ行列積」 / 「ヤコビ行列・ベクトル積」 (aligns with binding glossary entry Jacobian = ヤコビ行列)
- "micro-batch" → 「マイクロバッチ」
- "bootstrap target" (RL TD target) → 「ブートストラップターゲット」 (distinct from statistical resampling ブートストラップ法)
- "leaf tensor" → 「リーフテンソル」

## chapter_preliminaries/autograd.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "control flow" → 「制御フロー」
- "piecewise linear" → 「区分線形」
- "Hessian–vector product" → 「ヘッセ行列・ベクトル積」 (aligns with binding glossary entry Hessian = ヘッセ行列)
- "random vector" (in code, a vector populated with random numbers) → 「ランダムなベクトル」 (distinct from probability theory's 確率ベクトル)

## chapter_preliminaries/probability.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "sample space" → 「標本空間」
- "event" → 「事象」
- "running frequency" → 「逐次頻度」
- "Bessel's correction" → 「ベッセル補正」
- "error bars" → 「エラーバー」
- "biased" (of a coin) → 「偏りがある」 (distinct from additive model parameter 「バイアス」 and estimator bias 「統計的バイアス」)

## chapter_preliminaries/probability.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Chebyshev's inequality" → 「チェビシェフの不等式」
- "explaining away" → 「説明し尽くし」（動詞は「説明し尽くす」）
- "sensitivity" / "specificity" / "prevalence" → 「感度」 / 「特異度」 / 「有病率」
- "base-rate neglect" → 「基準率の無視」
- "paired comparison" / "paired evaluation" → 「対応のある比較」 / 「対応のある評価」

## chapter_linear-regression/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "proving ground" → 「検証の場」 (captures testing/verification role in both body prose and slide kicker)
- "training harness" → 「訓練ハーネス」 (consistent with 訓練 for training and standard software-engineering usage)
- "polynomial curve fitting" → 「多項式曲線フィッティング」 (standard term in Japanese ML textbooks)
- "held-out risk" → 「ホールドアウトリスク」 (keeps risk = リスク per ja/style.md §7.1)

## chapter_linear-regression/linear-regression.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "design matrix" → 「計画行列」 (standard Japanese statistics/ML term)
- "covariate" → 「共変量」 (standard Japanese statistics/ML term)
- "target" (regression output variable) → 「目的変数」 (standard Japanese ML/statistics term)
- "model class" → 「モデルクラス」 (katakana compound)
- "affine" / "affine combination" / "affine map" → 「アフィン」 / 「アフィン結合」 / 「アフィン写像」
- "collinearity" → 「多重共線性」

## chapter_linear-regression/linear-regression.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "full-batch gradient descent" / "Full-batch GD" → 「フルバッチ勾配降下法」 / 「フルバッチ GD」
- "vectorize" / "vectorization" → 「ベクトル化する」 / 「ベクトル化」
- "warm-up" → 「ウォームアップ」
- "homoscedastic" → 「等分散」
- "Laplace distribution" / "Laplace noise" → 「ラプラス分布」 / 「ラプラスノイズ」
- "dense layer" → 「密結合層」

## chapter_linear-regression/linear-regression.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "quantile" / "$q$-quantile" → 「分位点」 / 「$q$ 分位点」
- "pinball loss" → 「ピンボール損失」
- "absolute error" → 「絶対誤差」
- "input layer" / "output layer" → 「入力層」 / 「出力層」

## chapter_linear-regression/linear-regression.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "centering" → 「中心化」
- "Black-Scholes model" → 「ブラック・ショールズモデル」
- "Moore-Penrose pseudoinverse" → 「ムーア・ペンローズ擬似逆行列」

## chapter_linear-regression/synthetic-regression-data.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "design matrix" → 「計画行列」 (standard statistical/ML term in Japanese; "augmented design matrix" → 「拡大計画行列」)
- "noise floor" / "loss floor" → 「ノイズフロア」 / 「損失の下限」 ("noise floor" is an established katakana term; "loss floor" and standalone "the floor" refer to the lower bound of expected loss, rendered as 「損失の下限」 / 「下限」)
- "statistical estimation error" → 「統計的推定誤差」
- "Cramér–Rao bound" → 「クラメール・ラオの下界」 (katakana for person-named mathematical term per ja/style.md §5)
- "sample size" → 「標本サイズ」 (matches glossary's 「標本分散」 for sample variance and ja/style.md §7.1 「標本」 for statistical sample)
- "collinear" / "collinearity" → 「多重共線性（を持つ）」 (standard regression terminology in Japanese)

## chapter_linear-regression/linear-regression-scratch.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "implementation from scratch" / "from scratch" → 「スクラッチ実装」 / 「スクラッチで」「スクラッチからの」 (consistent with `LinearRegressionScratch` and `\eta_{\textrm{scratch}}`)
- "design matrix" → 「計画行列」 (standard statistics/ML term)
- "noise floor" / "irreducible noise floor" → 「ノイズフロア」 / 「削減不能なノイズフロア」 (established signal-processing and ML term)
- "excess loss" → 「過剰損失」 (standard learning-theory term)
- "generator" (Python iterator) → 「ジェネレータ」 (ja/style.md §3.3; distinct from generative-model `生成器`)

## chapter_linear-regression/linear-regression-scratch.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "robust loss" → 「ロバストな損失」 (standard ML term)

## chapter_linear-regression/oo-design.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "object-oriented design" → 「オブジェクト指向設計」 ("object-oriented" → 「オブジェクト指向」)
- "training harness" / "harness" → 「訓練ハーネス」 / 「ハーネス」
- "trainer" (lowercase common noun) → 「トレーナ」 (follows ja/style.md §3.3 dropping final ー on -er loanwords)
- "metric" → 「評価指標」
- "utility" / "utilities" → 「ユーティリティ」
- "boilerplate" → 「ボイラープレート」
- "split" (dataset partition) → 「分割」 ("training split" → 「訓練分割」, "validation split" → 「検証分割」)
- "generator" (Python iterator) → 「ジェネレータ」 (ja/style.md §3.3; distinguished from generative-model glossary entry 「生成器」)

## chapter_linear-regression/oo-design.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "progress board" → 「プログレスボード」
- "noise floor" → 「ノイズフロア」
- "on-the-fly generation" → 「オンザフライ生成」
- "Planck's law" → 「プランクの法則」, "Planck's constant" → 「プランク定数」, "Boltzmann constant" → 「ボルツマン定数」 (terms named after people in katakana per ja/style.md §5)

## chapter_linear-regression/linear-regression-concise.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "from scratch" / "scratch (implementation)" → 「スクラッチ（スクラッチ実装）」 (established usage in Japanese ML texts)
- "dense layer" → 「全結合層」 (aligned with "fully connected layer" across frameworks)
- "lazy shape inference" / "infer (input dimension/shape)" → 「形状の遅延決定」 / 「（自動的に・遅延）決定する」 (per style.md §7.1: avoid 推論 for shape inference)
- "noise floor" → 「ノイズフロア」 (established technical term)

## chapter_linear-regression/generalization.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "polynomial curve fitting" → 「多項式曲線フィッティング」 (standard Japanese ML terminology, e.g. Bishop PRML)
- "polynomial features" → 「多項式特徴量」 (following `feature` = 「特徴量」 for input variables in `ja/style.md` §7.1)
- "design matrix" → 「計画行列」 (standard statistics / regression term)
- "Vandermonde" (matrix) → 「Vandermonde」（Vandermonde 計画行列、Vandermonde 行列） (person name kept in Latin script per `ja/style.md` §5 & §3.2)
- "true risk" → 「真のリスク」 (alongside glossary's `population risk` = 「母集団リスク」)
- "noise floor" → 「ノイズフロア」
- "irreducible noise" → 「削減不能ノイズ」
- "held-out" → 「ホールドアウト」（ホールドアウトの / ホールドアウトされた）
- "$K$-fold cross-validation" / "fold" → 「$K$ 分割交差検証」 / 「フォールド」
- "leave-one-out (cross-validation)" → 「1 つ抜き（交差検証）」 / 「1 つ抜き法」
- "sample complexity" → 「サンプル複雑度」

## chapter_linear-regression/generalization.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "repeated cross-validation" → 「反復交差検証」
- "feature selection" → 「特徴選択」 (standard ML term)

## chapter_linear-regression/generalization.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Chebyshev polynomials / basis" → 「Chebyshev 多項式 / Chebyshev 基底」 (person name kept in Latin script per `ja/style.md` §5 & §3.2)
- "Sherman--Morrison identity" → 「Sherman–Morrison の恒等式」 (person names kept in Latin script per `ja/style.md` §5)

## chapter_linear-regression/weight-decay.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Tikhonov regularization" → 「Tikhonov 正則化」 (person name kept in Latin script per style.md §5; mixed term glossed on first use)
- "regularization constant" → 「正則化定数」 (standard ML term)
- "well-determined / poorly determined / undetermined directions" → 「よく定まる方向 / 定まりの悪い方向 / 定まらない方向」 (consistent across u01–u03)
- "shrinkage factor" → 「縮小因子」 (uses glossary 「縮小」 while distinguishing from regression coefficients 「係数」)
- "Lipschitz" → 「Lipschitz」 (person name kept in Latin script per style.md §5)

## chapter_linear-regression/weight-decay.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "effective number of parameters" → 「有効パラメータ数」 (standard statistical learning term)
- "oracle" (oracle parameters/predictor) → 「オラクル」 (standard ML benchmark term)

## chapter_linear-regression/weight-decay.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "reproducing kernel Hilbert space (RKHS)" → 「再生核ヒルベルト空間」 (standard Japanese ML/math term)
- "proximal step" → 「近接ステップ」 (standard convex optimization term)
- "design matrix" → 「計画行列」 (standard statistics/ML term)

## chapter_linear-classification/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "linear classification" → 「線形分類」（no glossary entry; standard ML term）
- "one-hot" → 「ワンホット」（ワンホットベクトル、ワンホットラベル、ワンホットターゲット；katakana compound per ja/style.md §3.2）
- "deployment" → 「デプロイ」（デプロイ時のデータ、デプロイ時のシフト、デプロイ時の頑健性；standard ML engineering usage）
- "importance weighting" → 「重点重み付け」（standard statistical ML term）
- "error bar" → 「エラーバー」（standard katakana term）

## chapter_linear-classification/softmax-regression.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "hard assignment" / "soft assignment" → 「ハード割り当て」 / 「ソフト割り当て」 (established Japanese ML terminology, e.g., PRML Japanese edition)
- "multinomial logistic regression" → 「多項ロジスティック回帰」 (standard statistical/ML term)
- "shift invariance" → 「シフト不変性」 (standard ML term)
- "decision region" → 「決定領域」 (standard ML term pairing with 決定境界)
- "ordinal regression" → 「順序回帰」 (standard statistical/ML term)
- Fashion-MNIST 10 categories → 「T シャツ、ズボン、プルオーバー、ドレス、コート、サンダル、シャツ、スニーカー、バッグ、アンクルブーツ」

## chapter_linear-classification/softmax-regression.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Gibbs' inequality" → 「ギブスの不等式」 (standard Japanese information-theory term, following style.md §5 exception for theorems named after people)
- "Kullback–Leibler divergence" → 「カルバック・ライブラー・ダイバージェンス」 / 「KL ダイバージェンス」 (standard Japanese ML terminology)
- "error rate" → 「誤り率」 (standard classification metric term pairing with 正解率)
- "zero-one error" → 「0–1 誤差」 (standard statistical learning term)
- "surrogate (loss)" → 「代理損失」 (standard ML term for convex surrogate losses)
- "linearly separable" → 「線形分離可能」 (standard ML term)

## chapter_linear-classification/softmax-regression.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Gumbel-max trick" → 「Gumbel-max トリック」; "Gumbel distribution" → 「ガンベル分布」 (standard Japanese ML terminology)
- "temperature scaling" → 「温度スケーリング」 (standard calibration term)
- "proper scoring rule" → 「固有スコアリングルール」 (standard statistical decision theory term)

## chapter_linear-classification/image-classification-dataset.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "channel-first" / "channel-last" → 「チャネルファースト」 / 「チャネルラスト」 (standard katakana in Japanese deep learning frameworks and texts)
- Fashion-MNIST class names ("T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot") → 「T シャツ、ズボン、プルオーバー、ドレス、コート、サンダル、シャツ、スニーカー、バッグ、アンクルブーツ」
- "flatten" (tensor reshaping) → 「平坦化」 (standard Japanese DL terminology, distinct from 平滑化 "smoothing")

## chapter_linear-classification/softmax-regression-scratch.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "from scratch" / "from-scratch" / "scratch" → 「スクラッチで」/「スクラッチ実装」/「スクラッチ版の〜」 (consistent across prose, headings, slides, and tables)
- "concise implementation" → 「簡潔な実装」
- "template" → 「テンプレート」 (katakana is standard in Japanese CV/ML texts)
- "one-hot" → 「ワンホット」 (follows ja-corrections.jsonl `one-hot-encoding` → `ワンホットエンコーディング`)
- "clamp" → 「クランプ」
- Fashion-MNIST class names: T-shirt → 「T シャツ」, trousers → 「ズボン」, pullover → 「プルオーバー」, dress → 「ドレス」, coat → 「コート」, sandal → 「サンダル」, shirt → 「シャツ」, sneaker → 「スニーカー」, bag → 「バッグ」, ankle boot → 「アンクルブーツ」

## chapter_linear-classification/softmax-regression-scratch.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "LogSumExp trick" → 「LogSumExp トリック」
- "smooth maximum" → 「滑らかな最大値」 (in Exercise 5)
- "exponent bits / mantissa bits" → 「指数部…ビット / 仮数部…ビット」

## chapter_linear-classification/classification.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Measuring Classifiers" → 「分類器の評価」 (section heading; concise noun phrase matching Japanese ML textbook usage)
- "reliability diagram" → 「信頼性図」 (no glossary entry; standard term in Japanese calibration literature)
- "temperature scaling" → 「温度スケーリング」 (no glossary entry; standard term in Japanese ML literature)
- "confidence" / "overconfidence" / "underconfidence" → 「確信度」 / 「過信」 / 「過小確信」 (separates prediction confidence from reliability 「信頼性」)
- "empirical accuracy" → 「経験正解率」 (aligns with statistical learning theory terminology such as 「経験誤差」)
- "cost-sensitive" → 「コスト考慮型」 (standard term in Japanese ML literature)
- "prevalence" → 「有病率」（疾患スクリーニングの文脈）/ 「陽性率」（一般のクラス比率の文脈）
- "false alarm" / "false-alarm rate" → 「誤報」 / 「誤報率」
- "F1 score" → 「F1 スコア」 (half-width space between Latin/digits and katakana per ja/style.md §2.2)
- "precision-recall curve" → 「適合率・再現率曲線」 (coordinate compound joined by ・ per ja/style.md §2.1)
- "reject option" / "abstain" → 「棄却オプション」 / 「棄権」 (standard decision-theory terminology in Japanese ML texts)

## chapter_linear-classification/classification.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "rejection" / "abstain" → 「棄却」 / 「棄権」 (consistent with u01)
- "precision-recall curve" → 「適合率・再現率曲線」 (consistent with u01)

## chapter_linear-classification/generalization-classification.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "validation split" / "evaluation split" → 「検証分割」 / 「評価分割」
- "memorization" / "memorizer" → 「暗記」 / 「暗記器」 (contrasts cleanly with 「学習」 and 「分類器」)
- "empirical error" / "population error" → 「経験誤差」 / 「母集団誤差」; "population distribution" → 「母集団分布」
- "Hoeffding's inequality", "Radon's theorem", "Sauer's lemma", "Boole" → Latin script for person names per ja/style.md §5 (「Hoeffding の不等式」「Radon の定理」「Sauer の補題」)
- "coverage" / "empirical coverage" / "undercover" → 「被覆率」 / 「経験被覆率」 / 「被覆率が低下する」
- "plug-in" (normal interval / radius / variance) → 「プラグイン」（プラグイン正規区間・プラグイン正規半径・プラグイン分散）
- "shatter" / "shattering" → 「粉砕する」 / 「粉砕」 (standard Japanese statistical learning theory terminology)
- "vacuous" (bound) → 「自明」（「自明化」）
- "rule of three" → 「3 の法則」
- "VC dimension" → 「VC 次元」 (with half-width space per ja/style.md §2.2 and §3.2, overriding glossary's unspaced 「VC次元」)

## chapter_linear-classification/generalization-classification.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "VC bound" → 「VC 上界」
- "one-vs-rest" → 「1 対他」
- "convex hull" → 「凸包」
- "monomial" → 「単項式」
- "VC dimension" → 「VC 次元」 (with half-width space per ja/style.md §2.2 and §3.2, overriding glossary's unspaced 「VC次元」)

## chapter_linear-classification/environment-and-distribution-shift.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "concept shift" → 「概念シフト」 (paired with 共変量シフト and ラベルシフト; established term in Japanese ML)
- "source distribution" / "target distribution" → 「ソース分布」 / 「ターゲット分布」
- "domain classifier" → 「ドメイン分類器」
- "classifier two-sample test" → 「分類器二標本検定」
- "support" (of a distribution) → 「サポート」, "support overlap" → 「サポートの重なり」
- "misspecified (model)" → 「誤特定された（モデル）」 (standard term in Japanese statistics, e.g. Shimodaira)
- "shift regime" → 「シフトの形態」

## chapter_linear-classification/environment-and-distribution-shift.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "performative prediction" → 「遂行的予測」 (established translation in Japanese ML)
- "invertible" / "singular" (matrix) → 「正則」 / 「特異」

## chapter_multilayer-perceptrons/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Xavier initialization" / "He initialization" → 「Xavier 初期化」「He 初期化」 (ja/style.md §2.2, §3.2: Latin person name + half-width space + 初期化)
- "exclusive-OR (XOR)" → 「排他的論理和（XOR）」
- "linear collapse" → 「線形崩壊」
- "machine precision" → 「機械精度」 (floating-point precision, distinct from classification precision 適合率)

## chapter_multilayer-perceptrons/mlp.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "exclusive-OR (XOR)" → 「排他的論理和（exclusive-OR、XOR）」 (thereafter XOR)
- "design matrix" → 「計画行列」
- "pre-activation" → 「活性化前入力」
- "rectified linear unit" (spelled-out full form) → 「整流線形ユニット」; "rectifying activation" / "rectifier" → 「整流型活性化関数」
- "squashing activation" / "squashing function" → 「圧縮型活性化関数」 / 「圧縮関数」
- "dead ReLU" → 「死んだ ReLU」
- "space-folding" → 「空間の折りたたみ」 / 「空間を折りたたむ」
- "tent function" / "tent map" → 「テント関数」 / 「テント写像」
- "linear piece" / "linear segment" (1D) → 「線形区間」; "linear region" → 「線形領域」
- "knot" (of a polyline/spline) → 「節点」

## chapter_multilayer-perceptrons/mlp.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "universal approximation theorem" → 「万能近似定理」
- "Gaussian error linear unit" → 「ガウス誤差線形ユニット」
- "sawtooth wave" → 「のこぎり波」
- "ridge function" → 「リッジ関数」
- "joint" (of a ReLU polyline) → 「折れ点」

## chapter_multilayer-perceptrons/mlp-implementation.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "He initialization" → 「He 初期化」 (with half-width space between Latin name and kanji per ja/style.md §2.2 & §3.2)
- "multiply-add" → 「積和演算」（演算回数を指す場合は「積和演算数」）
- "wall-clock time" → 「実時間」
- "pre-activation" → 「活性化前の値」
- "sanity check" → 「妥当性確認」

## chapter_multilayer-perceptrons/mlp-implementation.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "multiply-add" → 「積和演算」（演算回数を指す場合は「積和演算数」）
- "wall-clock time" → 「実時間」

## chapter_multilayer-perceptrons/backprop.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dead ReLU unit" → 「死んだ ReLU ユニット」 (standard Japanese ML term for dead/dying ReLU)
- "reverse topological order" → 「逆トポロジカル順序」 (standard graph/autodiff term)
- "sweep" → 「スイープ」 (consistent with chapter_preliminaries/autograd.md)

## chapter_multilayer-perceptrons/backprop.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "multiply-add" → 「積和演算」 (standard numerical computing term)
- "central difference" → 「中心差分」 (standard numerical analysis term)

## chapter_multilayer-perceptrons/numerical-stability-and-init.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Xavier initialization" / "Glorot initialization" → 「Xavier 初期化」 / 「Glorot 初期化」 (half-width space between Latin name and kanji per ja/style.md §2.2 & §3.2)
- "He initialization" / "Kaiming initialization" → 「He 初期化」 / 「Kaiming 初期化」 (half-width space per ja/style.md §2.2 & §3.2)
- "fan-in" / "fan-out" → 「ファンイン」 / 「ファンアウト」 (established katakana in Japanese ML literature)
- "pre-activation" → 「活性化前の値」 (pairs cleanly with glossary 「活性化値」 for activation)
- "mean square" → 「二乗平均」 (standard JIS Z 8101-1 order for E[X^2], avoiding ambiguity with (E[X])^2)
- "permutation symmetry" → 「置換対称性」

## chapter_multilayer-perceptrons/numerical-stability-and-init.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "submultiplicative" / "submultiplicativity" → 「劣乗法的」 / 「劣乗法性」 (established Japanese linear algebra term)
- "permutation invariance" → 「置換不変性」

## chapter_multilayer-perceptrons/generalization-deep.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "overparameterization" / "overparameterized" → 「過剰パラメータ化」 / 「過剰パラメータ化された」; "underparameterized" → 「過少パラメータ化された」
- "interpolation" / "interpolator" / "minimum-norm interpolator" → 「補間」 / 「補間解」 / 「最小ノルム補間解」
- "capacity" / "capacity bound" / "generalization bound" / "VC bound" → 「容量」 / 「容量上界」 / 「汎化上界」 / 「VC 上界」
- "vacuous (bound)" → 「空虚な（上界）」; "population error" → 「母集団誤差」
- "conditioning (numerical)" → 「行列の条件」
- "maximum-margin separator" → 「最大マージン分離器」
- "clean (labels/examples)" → 「クリーンな」; "corrupted labels" → 「破損したラベル」 / 「破損ラベル」
- "grokking" → 「グロッキング」

## chapter_multilayer-perceptrons/generalization-deep.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "patience (early stopping)" → 「ペイシェンス」
- "grokking" → 「グロッキング」

## chapter_multilayer-perceptrons/dropout.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "co-adaptation" → 「共適応」 (standard Japanese ML term for hidden-unit co-adaptation in dropout)
- "memorization" / "memorize" (of training quirks or noisy labels) → 「暗記」 / 「暗記する」 (distinguishes rote memorization from memory storage 「メモリ」)
- "pre-activation" → 「活性化前の値」
- "He initialization" → 「He 初期化」 (with half-width space per `ja/style.md` §2.2 and §3.2)
- "trainer" → 「トレーナ」 (drops final ー per `ja/style.md` §3.3 `-er` rule)
- "warm-up" → 「ウォームアップ」

## chapter_multilayer-perceptrons/dropout.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Monte Carlo dropout" → 「モンテカルロドロップアウト」 (solid katakana compound per `ja/style.md` §3.2)

## chapter_multilayer-perceptrons/kaggle-house-price.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "tabular data" / "tabular" → 「表形式データ」 / 「表形式」 (standard Japanese ML terminology)
- "data leakage" / "target leakage" → 「データリーク」 / 「ターゲットリーク」 (standard Japanese ML and Kaggle terminology)
- "$K$-fold cross-validation" / "fold" → 「$K$ 分割交差検証」 / 「フォールド」 (standard Japanese ML terminology)
- "imputation" / "mean imputation" / "median imputation" / "zero imputation" → 「補完」 / 「平均値補完」 / 「中央値補完」 / 「ゼロ補完」
- "design matrix" → 「計画行列」 (standard statistics/ML term)
- "He initialization" → 「He 初期化」 (following ja/style.md §3.2 Latin proper name + Japanese noun)
- "indicator" → 「インジケータ」 (dropping final ー per ja/style.md §3.3 *-or* rule)

## chapter_multilayer-perceptrons/kaggle-house-price.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "target encoding" → 「ターゲットエンコーディング」 (standard Japanese ML/Kaggle terminology)
- "leaderboard" → 「リーダーボード」 (standard Kaggle terminology)

## chapter_builders-guide/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Computation" (Chapter title, `chap_computation`) → 「計算」 (concise noun-phrase title matching the shortened 1-word English chapter title, while "Deep Learning Computation" on the cover slide is 「深層学習の計算」)
- "tree of modules" / "module tree" / "state tree" → 「モジュールの木」 / 「モジュール木」 / 「状態木」 (consistent with earlier chapters' `モジュール木` and `マージ木`)
- "lazy shape inference" → 「形状の遅延決定」 (avoids `推論` for shape inference per `ja/style.md` §7.1)
- "master weights" → 「マスター重み」 (standard Japanese ML/NVIDIA mixed-precision terminology)
- "straight-through estimator" → 「ストレートスルー推定器」
- "container" / "builder" / "caching allocator" → 「コンテナ」 / 「ビルダ」 / 「キャッシングアロケータ」 (drops final `ー` on *-er*/*-or* loanwords per `ja/style.md` §3.3)

## chapter_builders-guide/model-construction.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "module" → 「モジュール」 (recursive framework abstraction for layers, blocks, and models)
- "sequential container" → 「逐次コンテナ」 (no glossary entry; glossed on first body occurrence)
- "registry" / "register" (child modules) → 「レジストリ」 / 「登録（する）」; "unregistered" → 「未登録（の）」
- "plain list" / "plain Python list" → 「通常のリスト」 / 「通常の Python リスト」
- "forward method" → 「順伝播メソッド」 (distinguishes the concept from framework method names `forward`, `call`, `__call__`)
- "lazy initialization" / "lazy layer" → 「遅延初期化」 / 「遅延層」
- "configuration" / "config" → 「設定」 (glossed on first occurrence)

## chapter_builders-guide/model-construction.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "compiled execution" → 「コンパイル実行」 (paired with glossary's "eager execution" → 「即時実行」)
- "buffer" (non-trainable tensor state) → 「バッファ」 (drops final ー per `ja/style.md` §3.3)
- "tracer" (abstract value in graph tracing) → 「トレーサ」 (drops final ー per `ja/style.md` §3.3)
- "initializer" → 「イニシャライザ」 (drops final ー per `ja/style.md` §3.3)
- "dry run" → 「ドライラン」
- "builder" → 「ビルダ」 (drops final ー per `ja/style.md` §3.3)
- "foundation model" → 「基盤モデル」 (standard Japanese ML term; glossed on first occurrence)

## chapter_builders-guide/parameters-state-memory.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "state tree" → 「状態木」; "module tree" → 「モジュール木」
- "registration" → 「登録」
- "non-trainable" → 「訓練対象外の」
- "running average / running mean / running variance / running statistics" → 「移動平均 / 移動平均 / 移動分散 / 移動統計量」
- "parameter freezing / freezing" → 「パラメータ凍結 / 凍結」

## chapter_builders-guide/parameters-state-memory.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "meta device / meta tensor" → 「メタデバイス / メタテンソル」
- "deduplication" → 「重複除外」

## chapter_builders-guide/init.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "initializer" → 「イニシャライザ」 (loanword from -er drops final ー per ja/style.md §3.3; applies uniformly to initializer classes, objects, and functions across frameworks)
- "gain" (in initialization) → 「ゲイン」 (standard katakana term in Japanese DL/PyTorch documentation)
- "mean square" → 「二乗平均」 (aligns with glossary `root mean square` = 「二乗平均平方根」)
- "truncated normal" → 「切断正規分布」 (standard mathematical statistics term in Japanese)
- "module tree" / "block tree" → 「モジュールツリー」 / 「ブロックツリー」
- "body" (of a residual block) → 「本体」
- "residual branch" → 「残差ブランチ」
- "in place" / "in-place" → 「インプレースで」 / 「インプレースの」
- "positively homogeneous" → 「正斉次」

## chapter_builders-guide/custom-layers.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "custom layer" / "custom gradient" → 「カスタム層」 / 「カスタム勾配」
- "gain" (multiplicative scale parameter in normalization/initialization) → 「ゲイン」
- "straight-through estimator (STE)" → 「ストレートスルー推定器（straight-through estimator、STE）」
- "surrogate gradient" / "surrogate derivative" → 「代理勾配」 / 「代理導関数」（`ja/decisions.md` の `"surrogate loss" → 「代理損失」` に準拠）
- "random features" / "random Fourier features" → 「ランダム特徴量」 / 「ランダムフーリエ特徴量」
- "sequential container" → 「シーケンシャルコンテナ」
- "fused kernel" → 「融合カーネル」

## chapter_builders-guide/custom-layers.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "running statistics" / "running mean and variance" → 「移動統計量」 / 「移動平均や移動分散」
- "gradient check" → 「勾配チェック」
- "clamp" (clipping operation) → 「クランプ」

## chapter_builders-guide/numerics.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "precision" (numerical representation precision) → 「精度」 (glossary only lists classification precision 「適合率」)
- "mixed-precision training" → 「混合精度訓練」
- "master weights" → 「マスター重み」
- "normal number" / "subnormal number" → 「正規化数」 / 「非正規化数」 (standard JIS/IEEE 754 terms)
- "round-to-nearest" → 「最近傍丸め」
- "sign" / "exponent" / "mantissa" (floating-point fields) → 「符号」 / 「指数部」 / 「仮数部」

## chapter_builders-guide/numerics.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dtype promotion" → 「dtype の昇格」
- "dynamic loss scaling" → 「動的損失スケーリング」
- "loss scaler" → 「損失スケーラ」 (following ja/style.md §3.3 *-er* rule)
- "reduction" → 「リダクション」

## chapter_builders-guide/numerics.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "failure modes" → 「失敗モード」
- "accumulator" / "accumulation" → 「アキュムレータ」 / 「累積」 (following ja/style.md §3.3 *-or* rule)
- "round-to-nearest-even" → 「最近傍偶数丸め」

## chapter_builders-guide/saving-loading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "state dictionary" / "state dict" → 「状態辞書」
- "state tree" → 「状態ツリー」
- "module tree" → 「モジュールツリー」
- "serialize" / "serialization" → 「シリアライズする」 / 「シリアライズ」
- "pretrained weights" → 「事前学習済みの重み」
- "missing keys" / "unexpected keys" → 「欠落キー」 / 「予期しないキー」
- "crash-safe" → 「クラッシュセーフ」
- "trust boundary" → 「信頼境界」
- "allowlist" → 「許可リスト」

## chapter_builders-guide/saving-loading.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "stateful optimization" → 「状態を持つ最適化」
- "running average" → 「移動平均」

## chapter_builders-guide/saving-loading.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "backbone" → 「バックボーン」
- "classifier head" / "classification head" → 「分類器ヘッド」 / 「分類ヘッド」
- "freezing" / "unfreezing" → 「凍結」 / 「凍結解除」

## chapter_builders-guide/gpus-devices-memory.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "device placement" → 「デバイス配置」
- "soft placement" → 「ソフトプレースメント」
- "synchronization point" → 「同期ポイント」
- "pinned (page-locked) host memory" → 「ピン留め（ページロック）ホストメモリ」
- "memory budget" → 「メモリバジェット」
- "device-aware `Trainer`" → 「デバイス対応の `Trainer`」

## chapter_builders-guide/gpus-devices-memory.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "caching allocator" → 「キャッシングアロケータ」
- "memory view" → 「メモリビュー」
- "out-of-memory (OOM)" → 「メモリ不足（out-of-memory、OOM）」
- "gradient checkpointing" → 「勾配チェックポイント」
- "rematerialization" → 「再実体化」
- "step memory" → 「ステップメモリ」

## chapter_builders-guide/reproducibility-inspection.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "inspection" / "model inspection" → 「検査」 / 「モデルの検査」 (no glossary entry; standard Japanese software/ML term for inspecting internal model state)
- "pseudorandom number generator (PRNG)" → 「擬似乱数生成器（PRNG）」 (extends glossary `乱数生成器` for pseudorandom)
- "generator object" → 「ジェネレータオブジェクト」 (follows `ja/style.md` §3.3 `ジェネレータ` for PRNG `Generator` objects, distinguishing from GAN `生成器`)
- "worker" → 「ワーカー」 (retains final ー like `ユーザー`, `サーバー` in `ja/style.md` §3.3; standard in Japanese OS/PyTorch usage)
- "non-finite value" → 「非有限値」 (concise, exact counterpart to `isfinite`)

## chapter_builders-guide/reproducibility-inspection.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "atomic addition" / "atomic reduction" → 「アトミック加算」 / 「アトミックリダクション」 (standard GPU/parallel computing terminology)
- "scatter-add" → 「スキャッタ加算」 (parallel reduction into indexed bins)
- "leaf layer" → 「リーフ層」 (submodule with no children in module tree)
- "hook" / "pre-hook" → 「フック」 / 「事前フック」 (`register_forward_hook` / `register_forward_pre_hook`)
