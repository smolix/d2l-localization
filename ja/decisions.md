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

## chapter_advanced/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "critical batch size" → 「臨界バッチサイズ」, "maximal update parametrization ($\mu$P)" → 「最大更新パラメータ化（$\mu$P）」
- "additive attention" → 「加法アテンション」
- "content-addressable lookup" → 「内容参照型検索」
- "associative scan" → 「結合的なスキャン」
- "selective state space model" → 「選択的状態空間モデル」
- "sharding" → 「シャーディング」
- "value iteration" → 「価値反復法」 (paired with glossary `policy iteration` = 「方策反復法」)
- "imitation learning" → 「模倣学習」, "model predictive control (MPC)" → 「モデル予測制御」, "deadly triad" → 「死の三要素」
- "vocoder" → 「ボコーダ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)
- "probability path" → 「確率経路」

## chapter_optimization/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Optimization Algorithms" (chapter heading, `chap_optimization`) → 「最適化アルゴリズム」
- "Landscapes and Gradient Descent" (section heading, `sec_gd`) → 「ランドスケープと勾配降下法」
- "Scaling Up" (section heading, `sec_scaling`) → 「スケールアップ」
- "edge of stability" → 「安定性の縁」
- "critical damping" → 「臨界減衰」
- "excess loss" → 「過剰損失」
- "critical batch size" → 「臨界バッチサイズ」
- "Maximal Update Parametrization" ($\mu\text{P}$) → 「最大更新パラメータ化」 (follows `ja/style.md` §3.3 dropping final ー on パラメータ)
- "warmup–stable–decay (WSD)" → 「ウォームアップ・安定・減衰（WSD）」
- "Newton–Schulz orthogonalization" / "Newton–Schulz polynomial" → 「Newton–Schulz 直交化」 / 「Newton–Schulz 多項式」 (keeps person names in Latin script per `ja/style.md` §5)
- "iterate(s)" (optimization sequence $\mathbf{x}_t$) → 「反復点」
- "loss surface" → 「損失曲面」 (distinguished from "loss landscape" = 「損失ランドスケープ」 where geometric surface curvature is described)
- "lever" (tunable design control of an optimizer or training recipe) → 「制御軸」
- "playbook" → 「プレイブック」

## chapter_optimization/gd.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "edge of stability" → 「安定性の縁」
- "progressive sharpening" → 「漸進的鋭化」
- "strict saddle point" → 「狭義鞍点」
- "degenerate critical point" → 「退化臨界点」
- "local minimum" / "local maximum" (point in parameter space) → 「極小点」 / 「極大点」 (distinguished from scalar function values 「極小値」 / 「極大値」 when classifying critical points alongside 「臨界点」 and 「鞍点」)
- "iterate" (noun, $\mathbf{x}_t$ in optimization) → 「反復点」
- "level set" → 「等位集合」
- "stable manifold" → 「安定多様体」
- "$L$-smooth" / "$L$-smoothness" → 「$L$ 滑らか」 / 「$L$ 滑らかさ」

## chapter_optimization/gd.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "preconditioned sharpness" → 「前処理後の鋭さ」
- "Cauchy–Schwarz inequality" → 「コーシー・シュワルツの不等式」
- "Lipschitz continuous" → 「リプシッツ連続」

## chapter_optimization/gd.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "preconditioned gradient descent" → 「前処理付き勾配降下法」
- "diagonal preconditioning" / "diagonal preconditioner" → 「対角前処理」 / 「対角前処理行列」
- "quadratic convergence" → 「二次収束」
- "Hessian–vector product" → 「ヘッセ行列・ベクトル積」
- "Wigner's semicircle law" → 「ウィグナーの半円則」

## chapter_optimization/sgd.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "full-batch gradient descent" / "full-batch" → 「フルバッチ勾配降下法」 / 「フルバッチ」（ミニバッチ勾配降下法・ミニバッチと対）
- "excess loss" → 「過剰損失」（経験リスク最小化・確率的最適化における定訳）
- "noise floor" → 「ノイズフロア」
- "interpolating regime" / "noise-limited regime" / "underparameterized" / "overparameterized" → 「補間レジーム」 / 「ノイズ制限レジーム」 / 「過少パラメータ」 / 「過剰パラメータ」
- "sampling with replacement" / "sampling without replacement" → 「復元サンプリング」 / 「非復元サンプリング」（glossary `sampling` = 「サンプリング」と整合）
- "random reshuffling" → 「ランダムな再シャッフル」
- "finite-population correction factor" → 「有限母集団修正係数」
- "hardware amortization" → 「ハードウェアによる償却」
- "critical batch size" → 「臨界バッチサイズ」
- "NASA airfoil self-noise dataset" / "airfoil dataset" → 「NASA 翼型自己騒音データセット」 / 「翼型データセット」
- "Robbins–Monro conditions" → 「Robbins–Monro 条件」（`ja/style.md` §5 の人名ラテン文字維持規則および §2.2 スペース規則に従う）
- "warmup–stable–decay schedule" → 「ウォームアップ・安定・減衰スケジュール」

## chapter_optimization/sgd.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "ansatz" / "asymptotic ansatz" → 「仮定形」 / 「漸近的な仮定形」（微分方程式・漸近解析の標準訳）
- "square-root scaling" → 「平方根スケーリング」（線形スケーリングと対）

## chapter_optimization/test-models.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "nuisance hyperparameter" / "nuisance parameter" → 「局外ハイパーパラメータ」 / 「局外パラメータ」（統計学における定訳「局外母数 / 局外パラメータ」に整合）
- "usable learning-rate range" → 「使用可能な学習率範囲」（"effective learning rate"「実効学習率」との混同を避ける明確な訳語）
- "significance threshold" → 「有意性のしきい値」（スライド・表では「有意性しきい値」）
- "plain SGD" → 「素の SGD」（モメンタムや重み減衰を伴わない標準的な SGD、`decisions.md` の `"vanilla RNN" → 「素の RNN」` と整合）
- "byte-level" → 「バイトレベル」, "shard" → 「シャード」, "lookup table" → 「ルックアップテーブル」
- "decoder-only Transformer" → 「デコーダのみの Transformer」
- "learning-rate sweep" / "sweep" → 「学習率スイープ」 / 「スイープする」
- "single-pass streaming regime" → 「シングルパスのストリーミング設定」
- "landmark" (loss landmark) → 「ランドマーク」

## chapter_optimization/test-models.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dead ReLU" → 「死んだ ReLU」（"dead ReLU activations" → 「死んだ ReLU の活性化値」）
- "add-one smoothed" → 「1 加算平滑化を用いた」
- "usable learning-rate range" → 「使用可能な学習率範囲」
- "significance threshold" → 「有意性のしきい値」
- "plain SGD" → 「素の SGD」

## chapter_optimization/momentum.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "velocity" (in momentum optimization) → 「速度」, "velocity buffer" → 「速度バッファ」
- "effective step size" → 「実効ステップサイズ」
- "plain gradient descent" / "plain GD" / "plain SGD" → 「通常の勾配降下法」 / 「通常の GD」 / 「通常の SGD」
- "overdamped" / "critical damping (critically damped)" / "underdamped" → 「過減衰」 / 「臨界減衰」 / 「不足減衰」
- "contraction factor" → 「縮小率」
- "memory window" / "memory horizon" / "memory timescale" → 「記憶ウィンドウ」 / 「記憶ホライズン」 / 「記憶の時間スケール」 (distinguishes temporal gradient memory from hardware RAM 「メモリ」)
- "noise-limited" / "curvature-limited" → 「ノイズ律速」 / 「曲率律速」 (aligned with `compute-bound` = 「計算律速」 and `memory-bandwidth bound` = 「メモリ帯域幅律速」)
- "Nesterov Accelerated Gradient" / "Nesterov look-ahead (momentum)" → 「ネステロフの加速勾配法」 / 「ネステロフの先読み（モメンタム）」 (aligned with glossary `Nesterov momentum` = 「ネステロフのモメンタム」)
- "Vieta's formulas" → 「解と係数の関係」 (standard Japanese mathematical term)
- "overshoot" / "ringing" → 「オーバーシュート」 / 「リンギング」
- "stability ceiling" → 「安定性の上限」

## chapter_optimization/momentum.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "look-ahead point" → 「先読み点」, "anticipatory damping" → 「先回り減衰」
- "curvature-limited regime" / "noise-limited regime" → 「曲率律速レジーム」 / 「ノイズ律速レジーム」, "noise floor" → 「ノイズフロア」
- "strongly convex function" → 「強凸関数」
- "noise-to-signal ratio" → 「ノイズ対信号比」

## chapter_optimization/adam.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "per-coordinate" → 「座標ごとの」 (consistent across adaptive learning-rate sections)
- "nominal learning rate" → 「公称学習率」
- "step length" → 「ステップ長」 (distinguished from `step size` = 「ステップサイズ」)
- "uncentered second moment" → 「非中心二次モーメント」, "unbiased moment estimator" → 「不偏モーメント推定量」
- "diagonal scale invariance" → 「対角スケール不変性」 (extends `scale invariance` = 「スケール不変性」)
- "sign descent" → 「符号降下法」
- "decay timescale" → 「減衰タイムスケール」, "equilibrium norm" → 「平衡ノルム」

## chapter_optimization/adam.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "coupled $\ell_2$ regularization" / "coupled $\ell_2$ penalty" → 「結合型 $\ell_2$ 正則化」 / 「結合型 $\ell_2$ ペナルティ」 (paired with `decoupled weight decay` = 「分離型重み減衰」)
- "decay timescale" → 「減衰タイムスケール」, "equilibrium norm" → 「平衡ノルム」
- "rotational equilibrium" → 「回転平衡」, "relative angular update rate" → 「相対角度更新率」
- "limit cycle" → 「リミットサイクル」

## chapter_optimization/muon.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "steepest descent" → 「最急降下」 ("steepest descent direction" → 「最急降下方向」)
- "sign descent" → 「符号降下法」
- "orthogonalized gradient" → 「直交化勾配」
- "Newton–Schulz iteration" → 「ニュートン・シュルツ反復」, "Newton–Schulz polynomial" → 「ニュートン・シュルツ多項式」 (follows `ja/style.md` §5 katakana rule for mathematical methods named after people)
- "norm ball" → 「ノルム球」, "coordinate box" → 「座標ボックス」
- "odd matrix polynomial" → 「奇の行列多項式」
- "readout head" / "output head" → 「読み出しヘッド」 / 「出力ヘッド」

## chapter_optimization/muon.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "RMS alignment" → 「RMS アライメント」, "entrywise RMS" → 「要素ごとの RMS」
- "hybrid optimizer" → 「ハイブリッドオプティマイザ」
- "QK-Clip" → 「QK-Clip」 ("bilinear QK-Clip" → 「双線形 QK-Clip」)

## chapter_optimization/lr-scheduler.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "cumulative learning-rate budget" / "remaining budget" / "travel budget" → 「累積学習率予算」 / 「残余予算」 / 「移動予算」
- "terminal learning rate" → 「終端学習率」
- "piecewise-constant (step) decay" / "polynomial decay" → 「区分定数減衰（ステップ減衰）」 / 「多項式減衰」
- "frozen noise" / "frozen variance" → 「凍結ノイズ」 / 「凍結分散」
- "noise floor" → 「ノイズフロア」
- "warmup–stable–decay (WSD)" → 「ウォームアップ・安定・減衰（WSD）」
- "stable–then–decay (late linear decay)" / "late decay" → 「安定後減衰（終盤線形減衰）」 / 「終盤減衰」
- "cooldown" → 「クールダウン」
- "iterate" → 「反復点」
- "river valley" → 「河谷」
- "nat / nats" (natural-logarithm loss unit) → 「nat」
- "constructor" / "wrapper" → 「コンストラクタ」 / 「ラッパ関数」（`ja/style.md` §3.3 の `-er` / `-or` 長音省略規則に従う）

## chapter_optimization/lr-scheduler.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "checkpoint branching" / "cooldown branch" → 「チェックポイント分岐」 / 「クールダウンブランチ」
- "river valley" / "river bed" / "valley floor" → 「河谷」 / 「河床」 / 「谷底」

## chapter_optimization/batch-size.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "critical batch size" → 「臨界バッチサイズ」
- "gradient-noise scale" / "simple gradient-noise scale" → 「勾配ノイズスケール」 / 「単純勾配ノイズスケール」
- "signal power" / "noise power" → 「信号パワー」 / 「ノイズパワー」
- "small-batch (noise-dominated) regime" / "large-batch (curvature-dominated) regime" → 「小バッチ（ノイズ支配）領域」 / 「大バッチ（曲率支配）領域」
- "two-batch estimator" → 「2 バッチ推定量」
- "Pareto frontier" / "Pareto curve" / "knee" → 「パレートフロンティア」 / 「パレート曲線」 / 「屈曲点」
- "linear scaling (rule)" / "square-root scaling (rule)" → 「線形スケーリング（則）」 / 「平方根スケーリング（則）」
- "noise floor" → 「ノイズフロア」, "excess loss" → 「過剰損失」
- "wall-clock bound" → 「実経過時間律速」 (paired with glossary `compute-bound` = 「計算律速」)

## chapter_optimization/batch-size.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "time-optimal batch size" / "compute-optimal batch size" → 「時間最適バッチサイズ」 / 「計算量最適バッチサイズ」
- "data overhead" → 「データオーバーヘッド」

## chapter_optimization/scaling.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Scaling Up" / "scale-up" → 「スケールアップ」
- "proxy model" / "proxy" → 「プロキシモデル」 / 「プロキシ」
- "standard parametrization (SP)" → 「標準パラメータ化（standard parametrization、SP）」
- "Maximal Update Parametrization ($\mu\text{P}$)" → 「最大更新パラメータ化（maximal update parametrization、$\mu\text{P}$）」
- "coordinate check" → 「座標チェック」
- "base width" → 「基準幅」
- "fan-in" / "fan-out" → 「ファンイン」 / 「ファンアウト」
- "output readout" / "readout" → 「出力読み出し」 / 「読み出し」
- "log–log slope" → 「両対数傾き」
- "coherent" / "incoherent" (sum of updates) → 「コヒーレント」 / 「インコヒーレント」
- "spectral-norm optimizer" / "spectral optimizer" / "spectral scaling" → 「スペクトルノルムオプティマイザ」 / 「スペクトルオプティマイザ」 / 「スペクトルスケーリング」

## chapter_optimization/scaling.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "decay timescale" → 「減衰の時間スケール」
- "equilibrium norm" → 「平衡ノルム」

## chapter_optimization/practice.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Practice" (section heading of `chapter_optimization/practice.md`) → 「実践」
- "global-norm gradient clipping" / "global-norm clipping" → 「大域ノルムによる勾配クリッピング」 / 「大域ノルム勾配クリッピング」 / 「大域ノルムクリッピング」
- "guard" / "brake" / "constant brake" (clipping regimes) → 「ガード」 / 「ブレーキ」 / 「常時ブレーキ」
- "gradient spike" / "loss spike" / "spike recovery" → 「勾配スパイク」 / 「損失スパイク」 / 「スパイク回復」（「スパイクからの回復」）
- "weight averaging" / "weight EMA" → 「重み平均化」 / 「重み EMA」
- "live weights" / "live parameters" / "live model" (contrasted with EMA) → 「ライブ重み」 / 「ライブパラメータ」 / 「ライブモデル」
- "cooldown" / "cooldown branch" → 「クールダウン」 / 「クールダウン分岐」

## chapter_optimization/practice.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "gradient-noise scale" / "noise scale" → 「勾配ノイズスケール」 / 「ノイズスケール」
- "critical batch size" → 「臨界バッチサイズ」
- "z-loss" → 「$z$ 損失」 (or 「z 損失」 in plain-text slide titles)
- "uniform-shift invariance" → 「一様シフト不変性」
- "training harness" → 「訓練ハーネス」

## chapter_optimization/further-reading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Further Reading" (page/section heading) → 「関連文献」 (consistent with `decisions.md` `"Resources and Further Reading" → 「参考資料と関連文献」`)
- "scientific / nuisance / fixed hyperparameters" → 「科学的 / 局外 / 固定ハイパーパラメータ」 (standard statistical terminology for *nuisance parameter* = 局外パラメータ, glossed on first occurrence)
- "loss surface" → 「損失曲面」 (distinguished from `loss landscape` = 「損失ランドスケープ」 where the source discusses geometric cross-sections and surfaces)
- "mode connectivity" → 「モード連結性」
- "Sharpness-Aware Minimization (SAM)" → 「Sharpness-Aware Minimization (SAM)」 (kept in English as a proper optimizer name per `ja/style.md` §3.2)
- "Hessian-free optimization" / "Saddle-Free Newton" → 「ヘッセ行列フリー最適化」 / 「鞍点フリーニュートン法」
- "Gauss–Newton matrix" / "Krylov subspace" / "Newton–Schulz" → 「ガウス・ニュートン行列」 / 「クリロフ部分空間」 / 「ニュートン・シュルツ」 (follows `ja/style.md` §5 katakana rule for classical mathematical concepts named after people)
- "cosine annealing" / "warm restarts" / "snapshot ensembles" / "model soups" → 「コサインアニーリング」 / 「ウォームリスタート」 / 「スナップショットアンサンブル」 / 「モデルスープ」
- "critical batch size" → 「臨界バッチサイズ」, "hyperparameter transfer" → 「ハイパーパラメータ転移」

## chapter_attention/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "lookup" / "soft lookup" / "direct lookup" / "all-pairs lookup" / "key–value lookup" → 「ルックアップ」 / 「ソフトルックアップ」 / 「直接ルックアップ」 / 「全ペア間のルックアップ」（スライドでは「全ペアルックアップ」） / 「キー・値ルックアップ」 (distinguished from `参照` used for "consult / reference")
- "content-based retrieval" → 「コンテンツベース検索」 (pairs established IR katakana compound with glossary `retrieval` = 「検索」)
- "scaled dot-product" → 「スケール付きドット積」 (paired with glossary `dot product` = 「ドット積」)
- "boolean mask" / "boolean masking" → 「ブールマスク」 / 「ブールマスキング」
- "order blindness" → 「順序への不感性」
- "quadratic scaling" / "quadratic cost" / "sub-quadratic" → 「二次スケーリング」 / 「二次コスト」 / 「準二次」
- "mechanistic interpretation" / "mechanistic opacity" → 「機構的解釈」 / 「機構的不透明性」 (paired with `decisions.md` `interpretability` = 「解釈可能性」)
- "online softmax" → 「オンラインソフトマックス」
- "sliding-window attention" / "sliding window" → 「スライディングウィンドウアテンション」 / 「スライディングウィンドウ」
- "attention circuit" / "QK/OV circuit" → 「アテンション回路」 / 「QK/OV 回路」 (applies `ja/style.md` §2.2 spacing to glossary `OV circuit` = 「OV回路」)

## chapter_attention/queries-keys-values.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "lookup" → 「参照」 ("hard lookup" → 「ハード参照」, "soft lookup" → 「ソフト参照」, "exact lookup" → 「完全一致参照」, "key–value lookup" → 「キー・値参照」; matches `decisions.md` `"embedding lookup" → 「埋め込み参照」` and distinguishes `lookup` from `retrieval` = 「検索」)
- "Nadaraya–Watson kernel regression" / "Nadaraya–Watson estimator" → 「Nadaraya–Watson カーネル回帰」 / 「Nadaraya–Watson 推定量」 (person names kept in Latin script per `ja/style.md` §5, matching `Kneser–Ney 平滑化`)
- "bandwidth" (of a smoothing kernel) → 「バンド幅」 (standard Japanese statistical/ML terminology for kernel bandwidth, distinguished from hardware `memory bandwidth` = 「メモリ帯域幅」)
- "compatibility score" → 「適合度スコア」
- "leave-one-out cross-validation" / "leave-one-out error" → 「1 つ抜き交差検証」 / 「1 つ抜き誤差」
- "inverse temperature" → 「逆温度」
- "convex hull" → 「凸包」, "simplex" → 「単体」 ("simplex constraint" → 「単体制約」)
- "one-hot" → 「ワンホット」 (matches glossary `one-hot encoding` = 「ワンホットエンコーディング」)

## chapter_attention/queries-keys-values.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "heatmap" → 「ヒートマップ」
- "interventional experiments" → 「介入実験」
- "orthonormal" → 「正規直交」 (paired with glossary `orthogonal` = 「直交」)
- "isotropic" (kernel / bandwidth) → 「等方的」
- "row-stochastic matrix" → 「行確率行列」

## chapter_attention/attention-scoring.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dot-product score" → 「ドット積スコア」 (paired with glossary `dot-product attention` = 「ドット積アテンション」)
- "saturate / saturation" (of softmax) → 「飽和する / 飽和」 (standard Japanese ML terminology)
- "entropy collapse" → 「エントロピー崩壊」
- "maximal update parametrization" → 「最大更新パラメータ化」 (follows `ja/style.md` §3.3 `パラメータ`)
- "QK-norm" → 「QK-norm」 (architectural technique name kept in Latin script per `ja/style.md` §3.2)
- "masked softmax" → 「マスク付きソフトマックス」 (consistent with `decisions.md` `"masked autoencoder" → 「マスク付きオートエンコーダ」`)

## chapter_attention/attention-scoring.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "masked softmax" → 「マスク付きソフトマックス」
- "sliding window" → 「スライディングウィンドウ」 (standard katakana in ML)
- "packed sequence(s)" → 「パック化系列」, "document packing" → 「ドキュメントパッキング」
- "permutation matrix" → 「置換行列」

## chapter_attention/attention-scoring.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "causal sliding window" → 「因果スライディングウィンドウ」
- "relative standard deviation" → 「相対標準偏差」

## chapter_attention/multihead-attention.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "decoder-only language model" → 「デコーダのみの言語モデル」 (follows `ja/style.md` §3.3 dropping final ー on デコーダ)
- "copy-both task" → 「両方コピータスク」 (diagnostic task requiring one query to recover both source value vectors)
- "linear readout" / "readout" → 「線形読み出し」 / 「読み出し」
- "recurrence" (as a sequence-layer mechanism contrasted with convolution and attention) → 「再帰」

## chapter_attention/multihead-attention.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "key–value head" → 「キー・値ヘッド」 (uses coordinate compound dot per `ja/style.md` §2.1)
- "row-major" → 「行優先」

## chapter_attention/multihead-attention.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "permutation matrix" → 「置換行列」

## chapter_attention/positional-information.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "permutation equivariance" / "permutation equivariant" / "permutation symmetry" → 「置換同変性」 / 「置換同変」 / 「置換対称性」 (parallels glossary `translation equivariance` = 「平行移動同変性」)
- "sinusoidal (position) encoding" → 「正弦波（位置）エンコーディング」, "learned (position embeddings)" (scheme label) → 「学習型（位置埋め込み）」
- "NoPE (*No Positional Encoding*)" → 「NoPE」 (proper method acronym kept in Latin script, like RoPE and ALiBi)
- "extrapolation" / "length extrapolation" → 「外挿」 / 「長さ外挿」
- "content–position cross terms" / "position–position term" → 「内容・位置の交差項」 / 「位置・位置の項」
- "fast pairs" / "slow pairs" (in RoPE frequency decomposition) → 「高速ペア」 / 「低速ペア」
- "NTK-aware scaling" / "YaRN" → 「NTK 考慮スケーリング」 / 「YaRN」
- "recency bias" / "recency horizon" → 「直近バイアス」 / 「直近ホライズン」
- "open-weights (language models)" → 「オープンウェイトの（言語モデル）」
- "tied embedding weights" / "tied output head" → 「共有された埋め込み重み」 / 「重み共有出力ヘッド」

## chapter_attention/attention-at-scale.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "online softmax" → 「オンラインソフトマックス」
- "materialize / materialization" (of a score/weight matrix in memory) → 「実体化する / 実体化」
- "running statistics / running maximum / running sum" (in online/streaming algorithms) → 「逐次統計量 / 逐次最大値 / 逐次和」 (distinguished from exponential moving averages = 「移動統計量」 in batch normalization)
- "accumulator" → 「アキュムレータ」 (follows `ja/style.md` §3.3 `-or` rule)
- "score stripe" → 「スコアストライプ」
- "high-bandwidth memory (HBM)" → 「広帯域メモリ（high-bandwidth memory、HBM）」 (thereafter `HBM`)
- "on-chip SRAM / off-chip (device) memory" → 「オンチップ SRAM / オフチップ（デバイス）メモリ」
- "log-normalizer" → 「対数正規化項」, "normalizer" (denominator $s$) → 「正規化項」
- "ridge point" (in roofline model) → 「リッジポイント」

## chapter_attention/attention-at-scale.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "partial state" → 「部分状態」
- "tiled schedule" → 「タイル化されたスケジュール」

## chapter_attention/windowed-linear-attention.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "sliding-window attention" / "windowed attention" → 「スライディングウィンドウアテンション」 / 「ウィンドウアテンション」 ("windowed layer" → 「ウィンドウ層」)
- "global layer" / "global attention" → 「大域層」 / 「大域アテンション」 (paired symmetrically with `local` = 「局所」)
- "band mask" / "diagonal band" → 「帯マスク」 / 「対角帯」 (consistent with `banded matrix` = 「帯行列」)
- "tile sparsity" / "tiled kernel" → 「タイルスパース性」 / 「タイル化カーネル」
- "fast weight(s)" / "fast weight programmer" → 「高速重み」 / 「高速重みプログラマ」 (follows `ja/style.md` §3.2 `重み` and §3.3 `-er` rule)
- "associative memory" / "associative capacity" / "associative recall" → 「連想メモリ」 / 「連想容量」 / 「連想想起」
- "outer-product state" → 「外積状態」
- "crosstalk" / "cross-talk" → 「クロストーク」
- "intra-chunk" / "inter-chunk" → 「チャンク内」 / 「チャンク間」

## chapter_attention/windowed-linear-attention.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "data-dependent decay gate" / "decay gate" → 「データ依存の減衰ゲート」 / 「減衰ゲート」
- "associative recall" → 「連想想起」 (consistent with `u01`)
- "filler" (tokens) → 「フィラー（埋め草）」 / 「フィラー」

## chapter_attention/what-attention-computes.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "attention circuit" / "mechanistic circuit" → 「アテンション回路」 / 「機構的回路」
- "QK circuit" / "OV circuit" → 「QK 回路」 / 「OV 回路」 (half-width space per `ja/style.md` §2.2 & §3.2)
- "Q-composition" / "K-composition" / "V-composition" → 「Q 合成」 / 「K 合成」 / 「V 合成」
- "previous-token head" / "match-and-copy head" → 「直前トークンヘッド」 / 「照合コピーヘッド」
- "skip-trigram" → 「スキップトライグラム」 (paired with `decisions.md` `trigram` = 「トライグラム」)
- "unembedding projection" → 「逆埋め込み射影」, "tied" (weights/embeddings/output head) → 「重み共有された」
- "gauge freedom" → 「ゲージ自由度」
- "phase transition" → 「相転移」
- "first copy" / "second copy" / "later copies" → 「初回コピー」 / 「2 回目のコピー」 / 「後続のコピー（後続コピー）」
- "in-context lookup" → 「コンテキスト内参照」 (distinguished from `retrieval` = 「検索」)

## chapter_attention/what-attention-computes.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "copy matrix" → 「コピー行列」
- "head ablation" / "zero-ablation" / "mean ablation" → 「ヘッドアブレーション」 / 「ゼロアブレーション」 / 「平均アブレーション」
- "Cauchy–Schwarz" (inequality) → 「コーシー・シュワルツの不等式」 (`ja/style.md` §5 の人名由来用語の規則に従う)

## chapter_attention/what-attention-computes.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "top successor" → 「最上位後続トークン」
- "in-context gain" → 「コンテキスト内利得」

## chapter_attention/further-reading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Further Reading" (page/section heading) → 「関連文献」, "Recommended Resources" → 「推奨される参考資料」 (paired with `decisions.md` `"Resources and Further Reading" → 「参考資料と関連文献」`)
- "lookup" (key-value attention lookup) → 「ルックアップ」 (distinguished from bibliographic reference 「参照」 and `retrieval` = 「検索」)
- "associative memory" / "modern Hopfield network" / "associative recall" → 「連想記憶」 / 「現代的ホップフィールドネットワーク」 / 「連想想起」
- "open-weights model" → 「オープンウェイトモデル」
- "mechanistic circuit" / "circuit" (in Transformer interpretability) → 「機構的回路」 / 「回路」
- "activation patching" / "attribution patching" → 「活性化パッチング」 / 「帰属パッチング」
- "superposition" (in mechanistic interpretability) → 「重ね合わせ」
- "polysemantic" / "monosemantic" → 「多義的」 / 「単一意味」

## chapter_transformers/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Transformers" (chapter title) → 「Transformer」 (follows `ja/style.md` §3.2 keeping architecture proper names in Latin script and singular in Japanese headings)
- "pre-norm" / "post-norm" → 「Pre-Norm」 / 「Post-Norm」 (standard Latin notation in Japanese LLM literature, avoiding confusion between `norm` = ノルム and `normalization` = 正規化)
- "query–key normalization" → 「クエリ・キー正規化」 (inline code `` `QK-Norm` `` kept as-is)
- "key–value cache" / "KV cache" → 「キー・値キャッシュ」 / 「KV キャッシュ」 (follows `ja/style.md` §2.2 spacing and §7.1 `value` = 値)
- "weight-tied" / "weight tying" → 「重み共有」
- "open-weights LLM" → 「オープンウェイト LLM」
- "token collapse" → 「トークン崩壊」
- "conditional computation" → 「条件付き計算」
- "latent attention" → 「潜在アテンション」
- "speculative decoding" → 「投機的デコーディング」

## chapter_transformers/transformer-block.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Transformer block" → 「Transformer ブロック」 (`ja/style.md` §2.2 および §3.2 の英語＋日本語間スペース規則に従う)
- "pre-norm" / "post-norm" → 「Pre-Norm」 / 「Post-Norm」 (日本の LLM 文献で定着しているラテン文字表記、`LayerNorm`・`RMSNorm`・`QK-Norm` と整合)
- "sublayer" → 「サブ層」 (`ja/style.md` §7.1 の `layer` = 「層」と整合)
- "position-wise" → 「位置ごとの」
- "token mixing" / "channel mixing" → 「トークン混合」 / 「チャネル混合」
- "token collapse" / "token spread" → 「トークン崩壊」 / 「トークンの広がり」 ("logit spread" → 「ロジットの広がり」)
- "query–key normalization (QK-Norm)" → 「クエリ・キー正規化（QK-Norm）」
- "gated feed-forward network" → 「ゲート付きフィードフォワードネットワーク」
- "Xavier initialization" → 「Xavier 初期化」 (`ja/style.md` §2.2・§3.2、および `decisions.md` の「He 初期化」と整合)

## chapter_transformers/transformer-block.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "gated linear unit" → 「ゲート付き線形ユニット」
- "gate projection" / "up-projection" → 「ゲート射影」 / 「アップ射影」

## chapter_transformers/kv-cache.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "KV cache" / "key–value (KV) cache" → 「KV キャッシュ」 / 「キー・値（KV）キャッシュ」 (`ja/style.md` §2.2, §3.2, §7.1 に従う)
- "post-RoPE" → 「RoPE 適用後」
- "ridge point" / "compute roof" → 「リッジポイント」 / 「計算ルーフ」 (ルーフラインモデルの標準用語)
- "batched decoding" / "free batching" / "half-saturation batch size" → 「バッチデコード」 / 「無償のバッチ化」 / 「半飽和バッチサイズ」
- "uptraining" / "uptrain" → 「アップトレーニング」 / 「アップトレーニングする」
- "sliding window" → 「スライディングウィンドウ」

## chapter_transformers/kv-cache.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Multi-Head Latent Attention (MLA)" → 「マルチヘッド潜在アテンション（MLA）」
- "mean-pooling" → 「平均プーリング」
- "uptraining" → 「アップトレーニング」
- "sink token" → 「シンクトークン」

## chapter_transformers/kv-cache.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "virtual sink logit" → 「仮想シンクロジット」
- "ring-buffer cache" / "circular buffer" → 「リングバッファキャッシュ」 / 「循環バッファ」

## chapter_transformers/encoders-decoders.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "wiring" (of Transformer blocks) → 「配線」
- "encoder-only" / "decoder-only" → 「エンコーダのみ」 / 「デコーダのみ」
- "causal decoder" → 「因果デコーダ」
- "cloze task" → 「穴埋め課題」
- "bathtub (U-shaped) curve" → 「バスタブ型（U 字型）曲線」
- "anti-diagonal" → 「反対角線」
- "attribution" / "attribution map" → 「帰属」 / 「帰属マップ」
- "latent bottleneck" → 「潜在ボトルネック」
- "reranker" / "resampler" → 「リランカ」 / 「リサンプラ」（`ja/style.md` §3.3 の `-er` 語末規則に従う）

## chapter_transformers/encoders-decoders.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "wiring" (of Transformer blocks) → 「配線」
- "latent bottleneck" → 「潜在ボトルネック」
- "resampler" → 「リサンプラ」, "reranking" → 「リランキング」
- "ragged (batch)" → 「不揃いな（バッチ）」
- "content-addressed" / "position-addressed" → 「内容ベース」 / 「位置ベース」

## chapter_reinforcement-learning/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Linear-Quadratic Regulator (LQR)" / "iterative LQR (iLQR)" → 「線形二次レギュレータ（LQR）」 / 「反復 LQR（iLQR）」 (follows `ja/style.md` §3.3 dropping final ー on `-or`)
- "receding-horizon Model Predictive Control (MPC)" → 「リシーディングホライズン型のモデル予測制御（MPC）」（スライド等の略記では「リシーディングホライズン MPC」）
- "Differential Dynamic Programming" → 「微分動的計画法」, "Riccati" (in named terms) → 「リカッチ」
- "Bellman's principle" / "Bellman contraction" → 「ベルマンの原理」 / 「ベルマン縮小」（`ja/style.md` §5 に従い、人名単体の Bellman はラテン文字維持、用語名は glossary の「ベルマン最適方程式」「ベルマン作用素」に合わせてカタカナ）
- "Thompson sampling" → 「トンプソンサンプリング」
- "temporal differences" → 「時間差分」（glossary の `temporal-difference error` = 「時間差分誤差」と整合）
- "verifier" / "checker" → 「検証器」 / 「判定器」
- "demonstration" → 「デモンストレーション」, "imitation learning" → 「模倣学習」, "cloning" (behavior cloning) → 「行動クローニング」
- "group-relative weights" → 「グループ相対重み」
- "underactuated robotics" → 「劣駆動ロボティクス」

## chapter_reinforcement-learning/mdp.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "cell" (of a gridworld / FrozenLake) → 「マス」
- "command" (action command on FrozenLake) → 「コマンド」; directional commands `*left*`, `*down*`, `*right*`, `*up*` → 「*左*」「*下*」「*右*」「*上*」
- "start-state distribution" → 「初期状態分布」
- "verifier" / "verifier task" → 「検証器」 / 「検証器タスク」; "checker" → 「判定器」
- "simulator" → 「シミュレータ」 (follows `ja/style.md` §3.3 `-or` rule)
- "shaping bonus" / "potential-based shaping" → 「シェイピングボーナス」 / 「ポテンシャルベースのシェイピング」
- "state augmentation" → 「状態拡張」
- "reference policy" → 「参照方策」; "importance ratio" → 「重要度比」

## chapter_reinforcement-learning/mdp.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "absorbing state" → 「吸収状態」
- "continuing task" → 「継続タスク」
- "stationary policy" → 「定常方策」
- "belief state" → 「信念状態」
- "frame stacking" → 「フレーム積層」
- "reward hacking" → 「報酬ハッキング」
- "telescoping sum" → 「望遠鏡和」

## chapter_reinforcement-learning/mdp.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "belief" (in POMDP) → 「信念」
- "approach bonus" → 「接近ボーナス」
- "sparse" (rewards) → 「疎」

## chapter_reinforcement-learning/mdp.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "model-based" / "model-free" → 「モデルベース」 / 「モデルフリー」
- "hedge" (evasive catch-all response in VerifierTask) → 「両賭け」
- "planning discount" → 「計画割引率」; "planner" → 「プランナ」 (follows `ja/style.md` §3.3 `-er` rule)

## chapter_reinforcement-learning/value-iter.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "calm ice" / "slippery ice" → 「滑らない氷」 / 「滑る氷」（`is_slippery=False` / `True` の対比；"calm lake" / "slippery lake" → 「滑らない湖」 / 「滑る湖」）
- "cell" (gridworld square on FrozenLake) → 「マス」（ノートブックのコードセル「コードセル」と区別）
- "backup" / "backup diagram" / "sampled backup" → 「バックアップ」 / 「バックアップ線図」 / 「標本バックアップ」
- "Bellman expectation equation" → 「ベルマン期待方程式」（glossary の `Bellman optimality equation` = 「ベルマン最適方程式」と対）
- "look-ahead" / "one-step look-ahead" → 「先読み」 / 「1 ステップ先読み」

## chapter_reinforcement-learning/value-iter.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Bellman optimality operator" → 「ベルマン最適作用素」（glossary `Bellman operator` = 「ベルマン作用素」に対応）
- "sweep" (DP table pass) → 「スイープ」
- "finite-horizon" → 「有限ホライズン」（glossary `effective horizon` = 「実効ホライズン」に対応）
- "sup norm" → 「上限ノルム」
- "contraction" / "$\gamma$-contraction" → 「縮小写像」 / 「$\gamma$ 縮小写像」（性質を指す場合は「縮小性」）
- "stochastic shortest-path problem" → 「確率的最短経路問題」
- "certificate" / "stopping certificate" → 「精度保証」 / 「停止の精度保証」

## chapter_reinforcement-learning/value-iter.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "generalized policy iteration (GPI)" → 「一般化方策反復（GPI）」
- "transition kernel" → 「遷移カーネル」

## chapter_reinforcement-learning/value-iter.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Monte Carlo evaluation" → 「モンテカルロ評価」（glossary `Monte Carlo` = 「モンテカルロ」, `policy evaluation` = 「方策評価」に対応）
- "simulator" → 「シミュレータ」（`ja/style.md` §3.3 の `-or` 長音省略規則に従う）

## chapter_reinforcement-learning/value-iter.md#u05 (gemini gemini-4-carbon (medium), 2026-11-27)

- "reward shaping" / "potential-based shaping" → 「報酬シェイピング」 / 「ポテンシャルに基づくシェイピング」（表などでは「ポテンシャル型シェイピング」）
- "approach bonus" → 「接近ボーナス」
- "absorbing state" → 「吸収状態」
- "Manhattan distance" → 「マンハッタン距離」
- "telescoping sum" → 「望遠鏡和」

## chapter_reinforcement-learning/value-iter.md#u06 (gemini gemini-4-carbon (medium), 2026-11-27)

- "discounted state-visitation distribution" → 「割引状態訪問分布」
- "in-place sweep" → 「インプレースなスイープ」
- "performance-difference lemma" → 「性能差補題」
- "modified policy iteration" → 「修正方策反復法」

## chapter_reinforcement-learning/qlearning.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Temporal-Difference Learning" / "temporal-difference (TD)" → 「時間差分学習」 / 「時間差分」（略記 `TD` は「TD 誤差」「TD 更新」のように保持）
- "action value" / "state value" → 「行動価値」 / 「状態価値」（`ja/decisions.md` の `"state-value function" → 「状態価値関数」` と整合）
- "tabular" (e.g. tabular Q-learning, tabular control) → 「表形式の」（例：「表形式の Q 学習」「表形式制御」）、データ構造としての "table" → 「テーブル」
- "semi-gradient" → 「半勾配」, "residual-gradient" → 「残差勾配」, "double-sampling problem" → 「二重サンプリング問題」
- "Robbins–Monro conditions" → 「Robbins–Monro 条件」（`ja/style.md` §5 および §2.2 に従う）
- "slippery lake / slippery ice" vs "calm lake / calm ice" → 「滑る湖 / 滑る氷」 vs 「滑らない湖 / 滑らない氷」
- "damped step / damped backup" → 「緩和ステップ / 緩和バックアップ」（学習率やステップサイズの「減衰（decay）」と区別）

## chapter_reinforcement-learning/qlearning.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "$\epsilon$-greedy" → 「$\epsilon$ 貪欲」（スライド見出し等のプレーン表記は「イプシロン貪欲」）
- "tie-breaking / break ties" → 「同値解消 / 同値を解消する」
- "sweep" (of dynamic programming / value iteration) → 「スイープ」
- "Double Q-learning" → 「ダブル Q 学習」

## chapter_reinforcement-learning/qlearning.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "maximization bias" → 「最大化バイアス」
- "certainty equivalence / certainty-equivalent planning" → 「確実性等価 / 確実性等価計画」
- "sup-norm" → 「上限ノルム」
- "wrapper" / "planner" → 「ラッパ」 / 「プランナ」（`ja/style.md` §3.3 の `-er` 長音省略規則に従う）

## chapter_reinforcement-learning/qlearning.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "cliff walking" (`CliffWalking-v1`) → 「崖歩き」

## chapter_reinforcement-learning/exploration.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "arm" (of a bandit) → 「腕」 ("two-armed bandit" → 「2 腕バンディット」, "multi-armed bandit" → 「多腕バンディット問題」)
- "empirical mean" → 「経験平均」
- "bandit rule" → 「バンディット規則」
- "epsilon-greedy" / "$\epsilon$-greedy" → 「イプシロン貪欲」 / 「$\epsilon$ 貪欲」（`ja/style.md` §2.2 の数式と日本語の間の半角スペース規則に従う）
- "annealed (rule)" / "harmonic annealing" → 「アニーリング（規則）」 / 「調和アニーリング」
- "Boltzmann exploration" → 「ボルツマン探索」
- "Thompson sampling" → 「トンプソンサンプリング」
- "Hoeffding's inequality" → 「ヘフディングの不等式」
- "optimism" (under uncertainty) → 「楽観主義」
- "verifier" / "verifier task" → 「検証器」 / 「検証器タスク」

## chapter_reinforcement-learning/exploration.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "posterior sampling" → 「事後サンプリング」
- "index" (of UCB) → 「インデックス」

## chapter_reinforcement-learning/exploration.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "chain (of states)" → 「（状態の）チェーン」
- "absorbing (state)" → 「吸収状態」
- "optimistic initialization" → 「楽観的初期化」
- "count-based bonus" → 「カウントに基づくボーナス」
- "coverage" → 「カバレッジ」
- "calm lake" / "slippery lake" ("slippery ice") → 「穏やかな湖」 / 「滑りやすい湖」（「滑りやすい氷」）

## chapter_reinforcement-learning/exploration.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "reference policy" → 「参照方策」
- "preference" (policy logit before softmax) → 「選好度」
- "pass@K" → 「pass@K」（ラテン文字表記を維持）
- "peeling argument" → 「ピーリング法による議論」

## chapter_reinforcement-learning/imitation.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "demonstration" → 「デモンストレーション」 ("Learning from Demonstrations" → 「デモンストレーションからの学習」)
- "expert" / "learner" → 「エキスパート」 / 「学習者」
- "calm ice" / "calm lake" → 「滑らない氷」 / 「滑らない湖」, "slippery ice" / "slippery lake" → 「滑りやすい氷」 / 「滑りやすい湖」 (`FrozenLake-v1` の `is_slippery=False` / `True` に対応)
- "preference" (action preference $\theta_{s,a}$ in a softmax policy) → 「選好」
- "tabular" (policy / MDP / model) → 「表形式」
- "compounding error" → 「複合誤差」
- "curated responses" → 「厳選された応答」
- "policy container" → 「方策コンテナ」, "constructor" → 「コンストラクタ」, "accelerator" → 「アクセラレータ」 (`ja/style.md` §3.3 の `-er` / `-or` 長音省略規則に従う)

## chapter_reinforcement-learning/imitation.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "deployment" / "deploy" → 「運用」 / 「運用する」（`environment-and-distribution-shift.md` と統一）
- "missing coverage" → 「カバレッジの欠落」 / 「カバレッジ欠落」
- "state occupancy" / "state visitation" → 「状態占有度」 / 「状態訪問」
- "absorbing (state)" → 「吸収状態」
- "telescoping (argument / sum)" → 「望遠鏡和（の議論）」
- "chain" (chain MDP) → 「チェーン」
- "cloned imitator" / "recovering imitator" → 「クローン型の模倣者」 / 「回復型の模倣者」

## chapter_reinforcement-learning/imitation.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dataset aggregation (DAgger)" → 「データセット集約（DAgger）」
- "no-regret learner" → 「ノーリグレット学習者」
- "supervised fine-tuning (SFT)" → 「教師ありファインチューニング（SFT）」
- "on-policy distillation" → 「オンポリシー蒸留」
- "imitation learning" → 「模倣学習」

## chapter_reinforcement-learning/imitation.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "inverse reinforcement learning" → 「逆強化学習」, "maximum-entropy inverse reinforcement learning" → 「最大エントロピー逆強化学習」
- "potential-based shaping" / "reward shaping" → 「ポテンシャルに基づく報酬シェイピング」 / 「報酬シェイピング」

## chapter_reinforcement-learning/optimal-control.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "optimal control" → 「最適制御」, "control theory" → 「制御理論」
- "Linear-Quadratic Regulator (LQR)" / "Linear-Quadratic Regulation" → 「線形二次レギュレータ」 / 「線形二次制御」 (follows `ja/style.md` §3.3 dropping final ー on レギュレータ)
- "iterative LQR (iLQR)" → 「反復 LQR」
- "Model Predictive Control (MPC)" → 「モデル予測制御」, "receding-horizon" → 「移動ホライズン」
- "Riccati recursion" / "Discrete Algebraic Riccati Equation (DARE)" → 「リッカチ漸化式」 / 「離散代数リッカチ方程式」
- "cost-to-go" → 「残りコスト」, "stage cost" → 「ステージコスト」
- "certainty equivalence" → 「確実性等価」
- "controllability" / "observability" → 「可制御性」 / 「可観測性」
- "double integrator" → 「二重積分器」, "proportional-derivative (PD) controller" → 「比例・微分（PD）制御器」
- "energy pumping" → 「エネルギー注入」, "underactuated" / "underactuation" → 「劣駆動」 / 「劣駆動性」

## chapter_reinforcement-learning/optimal-control.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "inverted pendulum" → 「倒立振子」, "swing-up" → 「振り上げ」
- "semi-implicit Euler" → 「半陰的オイラー法」
- "torque saturation" → 「トルク飽和」, "basin of attraction" → 「吸引領域」, "catch basin" → 「捕捉領域」
- "Differential Dynamic Programming (DDP)" → 「微分動的計画法」, "nominal trajectory" → 「ノミナル軌跡」
- "Levenberg–Marquardt damping" → 「レーベンバーグ・マーカート減衰」
- "back-swing" → 「バックスイング」, "whip and catch" → 「振り抜きと捕捉」

## chapter_reinforcement-learning/optimal-control.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Time-varying LQR (TV-LQR)" → 「時変 LQR」
- "feedback tube" / "perturbation tube" → 「フィードバックチューブ」 / 「摂動チューブ」
- "warm start" / "warm-started" → 「ウォームスタート」 / 「ウォームスタートされた」
- "disturbance rejection" → 「外乱抑制」
- "system identification" → 「システム同定」, "adaptive control" → 「適応制御」
- "random shooting" → 「ランダムシューティング」

## chapter_reinforcement-learning/optimal-control.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "completing the square" → 「平方完成」, "Joseph stabilized form" → 「Joseph の安定化形式」
- "controllability matrix" → 「可制御性行列」
- "energy shaping" → 「エネルギー整形」
- "costate" → 「共状態」 (paired with `decisions.md` `"adjoint" → 「随伴変数」`)

## chapter_reinforcement-learning/baselines.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "baseline" (in policy-gradient RL) → 「ベースライン」 (standard katakana in Japanese RL literature)
- "actor-critic" → 「アクター・クリティック」 (coordinate compound with ・ per `ja/style.md` §2.1, pairing with 「クリティック不要」)
- "preference" / "preference table" (softmax action preferences $\theta_{s,a}$) → 「選好」 / 「選好テーブル」 ("value table" → 「価値テーブル」, distinguishing lookup tables from printed results tables 「表」)
- "calm lake" / "slippery ice" (FrozenLake settings) → 「穏やかな湖」 / 「滑りやすい氷」
- "zero-mean identity" / "zero-mean lemma" / "conditional identity" → 「ゼロ平均恒等式」 / 「ゼロ平均補題」 / 「条件付き恒等式」

## chapter_reinforcement-learning/baselines.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "centering" / "batch centering" / "group centering" → 「中心化」 / 「バッチ中心化」 / 「グループ中心化」
- "leave-one-out" / "leave-one-out baseline" → 「1 つ抜き」 / 「1 つ抜きベースライン」 (standard Japanese ML/statistics term matching 「1 つ抜き交差検証」; acronym `RLOO` kept in Latin script)
- "verifier task" / "verifier" → 「検証器タスク」 / 「検証器」
- "loss divisor" → 「損失の除数」

## chapter_reinforcement-learning/baselines.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "arm" (experimental variant in controlled comparisons) → 「アーム」 (preserves experimental-design terminology without colliding with 「条件」 for "condition")
- "plain SGD" → 「素の SGD」 (consistent with `bptt.md` `"vanilla RNN" → 「素の RNN」`)
- "exact baseline" / "value baseline" → 「厳密ベースライン」 / 「価値ベースライン」
- "probe" (frozen probe policy) → 「プローブ」
- "nats" (unit of entropy) → 「nat」 (kept as Latin unit symbol per `ja/style.md` §4)

## chapter_deep-reinforcement-learning/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "actor-critic" → 「アクタークリティック」 (matches glossary `actor` = 「アクター」, `critic` = 「クリティック」, `soft actor-critic` = 「ソフトアクタークリティック」)
- "deep Q-network" → 「深層 Q ネットワーク」 (follows `ja/style.md` §2.2 spacing around half-width `Q`, parallel to `Q-learning` = 「Q 学習」)
- "deadly triad" → 「死の三要素」 (standard Japanese RL term from Sutton & Barto)
- "replay buffer" → 「リプレイバッファ」 (follows `ja/style.md` §3.3 dropping final `ー` on `バッファ`)
- "world model" → 「世界モデル」 (established Japanese RL/DL term)
- "planning" / "plan" (in RL) → 「プランニング」 / 「プランニングする」 (standard Japanese RL term)
- "swing up" / "swing-up" (pendulum control) → 「振り上げる」 / 「振り上げ」
- "probability-ratio clipping" / "ratio clipping" → 「確率比クリッピング」 / 「比率クリッピング」
- "twin critics" → 「ツインクリティック」
- "soft backup" → 「ソフトバックアップ」
- "gridworld" → 「グリッドワールド」

## chapter_deep-reinforcement-learning/actor-critic.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "actor-critic" / "actor-critic method" → 「アクター・クリティック」 / 「アクター・クリティック法」 (standard Japanese RL terminology; coordinate compound per `ja/style.md` §2.1)
- "multi-step return" / "$n$-step return" → 「複数ステップ収益」 / 「$n$ ステップ収益」 (paired with `decisions.md` `"multistep" → 「複数ステップ」` and glossary `return` = 「収益」)
- "temporal-difference (TD) error" → 「時間差分誤差」（以降「TD 誤差」）
- "semi-gradient" → 「半勾配」 (standard Japanese RL term in Sutton & Barto)
- "Bellman equation" / "Bellman expectation backup" / "backup diagram" → 「ベルマン方程式」 / 「ベルマン期待値バックアップ」 / 「バックアップ線図」
- "generalized policy iteration" → 「一般化方策反復法」 (paired with glossary `policy iteration` = 「方策反復法」)

## chapter_deep-reinforcement-learning/actor-critic.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "fitted TD" → 「適合 TD」
- "deadly triad" → 「死の三要素」 (standard Japanese RL term in Sutton & Barto)
- "telescoping cancellation" / "telescoping identity" → 「逐次相殺」 / 「逐次相殺の恒等式」

## chapter_deep-reinforcement-learning/actor-critic.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "$\lambda$-return" → 「$\lambda$ 収益」
- "eligibility trace" → 「適格度トレース」 (standard Japanese RL term in Sutton & Barto)
- "backward scan" → 「逆方向スキャン」
- "group baseline" → 「グループベースライン」

## chapter_deep-reinforcement-learning/ppo.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Trust Region Policy Optimization (TRPO)" → 「信頼領域方策最適化（TRPO）」 (paired with glossary `trust region` = 「信頼領域」, `policy` = 「方策」, `optimization` = 「最適化」)
- "change of measure" → 「測度変換」 (standard probability/measure-theory term)
- "surrogate objective" / "per-step surrogate" / "population surrogate" → 「代理目的関数」 / 「ステップごとの代理目的関数」 / 「母集団代理目的関数」 (paired with `decisions.md` `"surrogate loss" → 「代理損失」`)
- "performance difference lemma" → 「性能差補題」
- "clipped objective" / "clipped surrogate" → 「クリップ付き目的関数」 / 「クリップ付き代理目的関数」
- "entropy bonus" → 「エントロピーボーナス」

## chapter_deep-reinforcement-learning/ppo.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "bootstrap" (RL temporal-difference sense) → 「ブートストラップ」 / 「ブートストラップする」 (glossary's proposed `ブートストラップ法` refers to statistical resampling; RL literature uses 「ブートストラップ」)
- "discounted state-occupancy distribution" / "occupancy" → 「割引状態占有分布」 / 「状態占有分布」
- "monotonic-improvement bound" / "monotonic improvement" → 「単調改善下界」 / 「単調改善」

## chapter_deep-reinforcement-learning/ppo.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "approximate KL" → 「近似 KL」
- "clip fraction" → 「クリップ率」
- "domain randomization" → 「ドメインランダマイゼーション」 (standard katakana in Japanese robotics and RL literature)

## chapter_deep-reinforcement-learning/ppo.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "coupling" (probabilistic coupling argument) → 「カップリング」
- "Pinsker's inequality" → 「ピンスカーの不等式」 (follows `ja/style.md` §5 katakana rule for mathematical terms named after people)
- "Fisher-vector product" → 「フィッシャー・ベクトル積」

## chapter_deep-reinforcement-learning/regularized.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "gridworld" → 「グリッドワールド」
- "hazard cell" / "hazard lane" → 「危険セル」 / 「危険レーン」
- "reference policy" → 「参照方策」
- "preference" / "preference learning" → 「選好」 / 「選好学習」
- "Bradley-Terry model" → 「ブラッドリー・テリーモデル」（`ja/style.md` §5 の人名由来用語のカタカナ規則に従う）
- "potential-based shaping" / "shaping" → 「ポテンシャルに基づくシェイピング」 / 「シェイピング」
- "rms error" / "RMS error" → 「二乗平均平方根（RMS）誤差」（以降およびスライド・演習では「RMS 誤差」）
- "gradient ascent" → 「勾配上昇法」

## chapter_deep-reinforcement-learning/regularized.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Goodhart's law" → 「グッドハートの法則」
- "Gibbs' inequality" → 「ギブスの不等式」
- "nat" / "nats" (unit of information) → 「nat」（`ja/style.md` §4 の単位記号の原則に従いラテン文字維持）
- "exponential tilt" / "tilt" → 「指数ティルト」 / 「ティルト」
- "log-partition function" → 「対数分配関数」
- "point mass" → 「点質量」
- "exponentiated gradient ascent" → 「指数勾配上昇法」
- "simplex" → 「単体」
- "Boltzmann exploration" → 「ボルツマン探索」
- "control-as-inference" → 「推論としての制御」
- "Direct Preference Optimization" → 「直接選好最適化」
- "reward-divergence frontier" → 「報酬・ダイバージェンスフロンティア」
- "Best-of-$N$ sampling" → 「Best-of-$N$ サンプリング」
- "reverse divergence" / "forward divergence" → 「逆方向のダイバージェンス」 / 「順方向のダイバージェンス」

## chapter_deep-reinforcement-learning/regularized.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "soft backup" / "soft Bellman backup" → 「ソフトバックアップ」 / 「ソフトベルマンバックアップ」
- "soft value" / "soft action value" → 「ソフト価値」 / 「ソフト行動価値」
- "soft value iteration" → 「ソフト価値反復法」
- "hard maximum" → 「ハード最大値」（glossary の `soft maximum` = 「ソフト最大値」と対）
- "contraction" / "$\gamma$-contraction" → 「縮小写像」 / 「$\gamma$ 縮小写像」
- "KL budget" → 「KL 予算」
- "overoptimization" → 「過最適化」
- "verifier" → 「検証器」
- "hedged answer" → 「保険をかけた回答」

## chapter_deep-reinforcement-learning/regularized.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "annotator" → 「アノテータ」（`ja/style.md` §3.3 の `-or` 長音省略規則に従う）
- "bisection" → 「二分法」
- "one-hot code" → 「ワンホット符号」

## chapter_deep-reinforcement-learning/sac.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "pathwise gradient" / "pathwise estimator" → 「パスワイズ勾配」 / 「パスワイズ推定量」
- "replay" / "replay buffer" → 「リプレイ」 / 「リプレイバッファ」
- "maximization bias" → 「最大化バイアス」
- "soft value" / "soft policy evaluation" / "soft policy improvement" → 「ソフト価値」 / 「ソフト方策評価」 / 「ソフト方策改善」
- "tilted policy" / "tilted optimum" / "tilted distribution" → 「傾斜方策」 / 「傾斜最適方策」 / 「傾斜分布」
- "reverse-KL projection" → 「逆 KL 射影」
- "twin critics" → 「ツインクリティック」
- "automatic temperature" → 「自動温度調整」

## chapter_deep-reinforcement-learning/sac.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "squash" / "squashed Gaussian policy" → 「スクワッシュ」 / 「スクワッシュされたガウス方策」
- "log-determinant" → 「対数行列式」
- "trapezoid rule" / "quadrature" → 「台形公式」 / 「数値積分」
- "catastrophic cancellation" → 「桁落ち」
- "Polyak averaging" → 「Polyak 平均」 (keeps person name in Latin script per `ja/style.md` §5, matching `He 初期化`)
- "dual gradient descent" → 「双対勾配降下法」

## chapter_deep-reinforcement-learning/sac.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Polyak targets" / "Polyak update" → 「Polyak ターゲット」 / 「Polyak 更新」
- "update-to-data ratio" / "update ratio" → 「更新対データ比」 / 「更新比」
- "container" → 「コンテナ」 (drops final ー per `ja/style.md` §3.3)

## chapter_deep-reinforcement-learning/sac.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "primacy bias" → 「初期バイアス」 (glossed on first occurrence as `*初期バイアス*（primacy bias）`)
- "damping controller" / "spinning controller" → 「減衰制御器」 / 「回転制御器」
- "realized discounted soft return" → 「実現割引ソフト収益」

## chapter_deep-reinforcement-learning/model-based.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "planning" / "plan" → 「計画」 ("replanning" → 「再計画」; matches glossary `planning horizon` = 「計画ホライズン」)
- "planner" → 「プランナ」, "controller" → 「コントローラ」 (follows `ja/style.md` §3.3 `-er` rule)
- "model predictive control (MPC)" → 「モデル予測制御（model predictive control、MPC）」
- "receding-horizon" → 「移動ホライズン」 (standard Japanese control-engineering term)
- "world model" → 「世界モデル」, "dynamics model" → 「ダイナミクスモデル」
- "swing-up" / "swing up" → 「振り上げ」 / 「振り上げる」
- "compounding error" / "compound" (errors) → 「累積誤差」 / 「累積する」
- "open-loop" → 「開ループ」
- "random shooting" → 「ランダムシューティング」, "warm start" → 「ウォームスタート」
- "replay buffer" → 「リプレイバッファ」 (follows `ja/style.md` §3.3 dropping final ー on バッファ)
- "imagined step(s) / transition(s) / rollout(s)" → 「想像上のステップ / 遷移 / ロールアウト」
- "real step(s) / environment / episode" → 「実ステップ / 実環境 / 実エピソード」

## chapter_deep-reinforcement-learning/model-based.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "coverage" (data/state-space coverage) → 「カバレッジ」
- "out-of-support extrapolation" → 「サポート外への外挿」
- "amortize" → 「償却する」

## chapter_deep-reinforcement-learning/model-based.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "policy distillation" / "distillation" / "distill" → 「方策蒸留」 / 「蒸留」 / 「蒸留する」
- "model-based policy optimization (MBPO)" → 「モデルベース方策最適化（model-based policy optimization、MBPO）」
- "branched rollout(s)" → 「分岐ロールアウト」
- "update ratio" → 「更新比率」 (`UTD` kept in Latin script)
- "tree search" → 「木探索」, "lookahead" / "look ahead" → 「先読み」 / 「先読みする」

## chapter_deep-reinforcement-learning/model-based.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "cross-entropy method" → 「交差エントロピー法」
- "sampling distribution" (proposal distribution in CEM) → 「サンプリング分布」
- "disagreement" (of an ensemble) → 「不一致」
- "control arm" (experimental baseline) → 「対照群」

## chapter_deep-reinforcement-learning/offline-rl.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "pessimism" / "pessimistic" → 「悲観主義」 / 「悲観的」, "pessimistic value iteration" → 「悲観的価値反復法」 (standard RL counterpart to optimism = 「楽観主義」)
- "coverage condition" / "coverage" → 「被覆条件」 / 「被覆」
- "empirical MDP" → 「経験 MDP」, "certainty-equivalent" / "certainty equivalence" → 「確実性等価」 / 「確実性等価性」
- "stitching" / "trajectory stitching" → 「スティッチング」 / 「軌跡のスティッチング」
- "support" (empirical sample backing in offline RL) → 「サポート」 ("thin support" → 「サポートが薄い」)
- "naive (table / offline Q-learning)" → 「素朴な（テーブル / オフライン Q 学習）」
- "lower confidence bound (LCB)" → 「信頼下限（LCB）」 (paired with glossary `upper confidence bound` = 「信頼上限」)
- "two-armed bandit" → 「2 本腕バンディット」 ("arm" → 「腕」)

## chapter_deep-reinforcement-learning/offline-rl.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Hoeffding's inequality" → 「ヘフディングの不等式」 (follows `ja/style.md` §5 katakana rule for theorems named after people)
- "Conservative Q-learning (CQL)" → 「保守的 Q 学習（CQL）」
- "return-to-go" → 「残余収益」
- "return conditioning" / "return-conditioned" → 「収益条件付け」 / 「収益条件付き」
- "deadly triad" → 「死の三要素」 (established Japanese RL term from Sutton & Barto)

## chapter_deep-reinforcement-learning/offline-rl.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "direct preference optimization (DPO)" → 「直接選好最適化（DPO）」
- "ordinary importance sampling" / "weighted importance sampling" → 「通常の重要度サンプリング」 / 「重み付き重要度サンプリング」

## chapter_deep-reinforcement-learning/rl-sequences.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "verifier" / "checker" / "grader" → 「検証器」 / 「判定器」 / 「採点器」 ("lenient checker" / "lenient score" → 「寛容な判定器」 / 「寛容スコア」, "exact checker" / "exact score" → 「厳密な判定器」 / 「厳密スコア」)
- "reinforcement learning from verifiable rewards (RLVR)" → 「検証可能な報酬からの強化学習（RLVR）」
- "response weight" → 「応答重み」
- "loophole" → 「抜け穴」, "hedge" (hedged candidate response) → 「ヘッジ」
- "tilted optimum" → 「傾斜最適解」
- "deadly triad" → 「死の三要素」
- "Group Relative Policy Optimization (GRPO)" → 「グループ相対方策最適化（GRPO）」
- "chain-of-thought" → 「思考の連鎖」

## chapter_deep-reinforcement-learning/rl-sequences.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Group Sequence Policy Optimization (GSPO)" → 「グループ系列方策最適化（GSPO）」
- "Direct Preference Optimization (DPO)" → 「直接選好最適化（DPO）」
- "Bradley-Terry model" → 「Bradley-Terry モデル」（人名は ja/style.md §5 に従いラテン文字維持）
- "clipping band" / "band" → 「クリッピング許容帯」 / 「許容帯」
- "forward divergence" / "reverse divergence" → 「順方向ダイバージェンス」 / 「逆方向ダイバージェンス」

## chapter_deep-reinforcement-learning/further-reading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Further Reading" (section heading) → 「関連文献」（`decisions.md` の `"Resources and Further Reading" → 「参考資料と関連文献」` と整合）
- "domain randomization" → 「ドメインランダム化」
- "prioritized replay" → 「優先度付き経験再生」（glossary の `experience replay` = 「経験再生」と対応）
- "dueling network / dueling head" → 「デュエリングネットワーク / デュエリングヘッド」
- "distributional methods" → 「分布型手法」
- "deadly triad" → 「死の三要素」
- "replay buffer" → 「リプレイバッファ」（`ja/style.md` §3.3 の長音省略規則に従う）
- "twin critics" → 「ツインクリティック」
- "update ratio" → 「更新比」
- "primacy bias" → 「初頭バイアス」

## chapter_deep-reinforcement-learning/further-reading.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "planner" → 「プランナ」（`ja/style.md` §3.3 の `-er` 長音省略規則に従う）
- "compounding error" → 「累積誤差」
- "cross-entropy method" → 「交差エントロピー法」
- "pessimism" (in offline RL) → 「悲観主義」
- "expectile" → 「エクスペクタイル」
- "advantage-weighted regression" → 「アドバンテージ重み付き回帰」
- "return-conditioned sequence model" → 「収益条件付き系列モデル」
- "diffusion policy" → 「拡散方策」
- "process reward" → 「プロセス報酬」
- "self-play" → 「自己対戦」
- "meta-learning" / "hierarchical learning" → 「メタ学習」 / 「階層型学習」
- "pseudo-count" → 「擬似カウント」

## chapter_deep-reinforcement-learning/further-reading.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "eligibility trace" → 「適格度トレース」
- "stratified bootstrap confidence interval" → 「層化ブートストラップ信頼区間」
- "performance profile" → 「性能プロファイル」
- "preference" (tabular policy logit) → 「選好度」
- "clip fraction" → 「クリップ率」
- "leave-one-out weight" → 「1 つ抜き重み」
- "tilted optimum" → 「傾斜最適解」

## chapter_generative-adversarial-networks/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "implicit distribution" / "implicit generator" / "implicit density estimation" → 「陰的な分布」 / 「陰的生成器」 / 「陰的密度推定」
- "two-sample (comparison / game / classification / critic)" → 「2 サンプル（比較 / ゲーム / 分類 / クリティック）」 (aligned with glossary `sample` = 「サンプル」 across GAN formulations)
- "Jensen–Shannon divergence" → 「イェンセン・シャノン・ダイバージェンス」 (paired with glossary `Jensen's inequality` = 「イェンセンの不等式」 and `Kullback–Leibler divergence` = 「カルバック・ライブラー・ダイバージェンス」)
- "non-saturating generator loss" / "saturating" / "non-saturating" → 「非飽和生成器損失」 / 「飽和型」 / 「非飽和型」
- "integral probability metric (IPM)" → 「積分確率指標（integral probability metric、IPM）」
- "Wasserstein-1 distance" → 「1 次ワッサースタイン距離」 (paired with glossary `Wasserstein distance` = 「ワッサースタイン距離」)
- "Bradley–Terry ranking" → 「ブラッドリー・テリー（Bradley–Terry）ランキング」
- "zero-centered gradient penalty" → 「ゼロ中心勾配ペナルティ」
- "vocoder" → 「ボコーダ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)
- "witness (function)" (in IPMs) → 「証人関数」

## chapter_mathematics-for-deep-learning/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Gershgorin circles" → 「ゲルシュゴリンの円」, "Marchenko–Pastur law" → 「マルチェンコ・パスツール則」, "Eckart–Young low-rank approximation" → 「エッカート・ヤングの低ランク近似」
- "parameter-efficient fine-tuning" → 「パラメータ効率的ファインチューニング」
- "Karush–Kuhn–Tucker (KKT) optimality conditions" → 「カルーシュ・キューン・タッカー（KKT）最適性条件」, "Lagrangian duality" → 「ラグランジュ双対性」
- "concentration inequality" → 「集中不等式」, "out-of-sample generalization" → 「標本外汎化」, "Rademacher complexity" → 「ラデマッハ複雑度」, "bias–variance trade-off" → 「バイアス・分散のトレードオフ」
- "Wasserstein optimal transport" → 「ワッサーシュタイン最適輸送」, "contrastive representation learning" → 「対照表現学習」
- "phase portrait" → 「相図」, "continuous adjoint sensitivity method" → 「連続随伴感度法」
- "Itô stochastic differential equation (SDE)" → 「伊藤確率微分方程式（SDE）」, "Itô's lemma" → 「伊藤の補題」 (Japanese mathematician 伊藤清)
- "Ornstein–Uhlenbeck process" → 「オルンシュタイン・ウーレンベック過程」, "Fokker–Planck equation" → 「フォッカー・プランク方程式」

## chapter_mdl-linear-algebra/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Vectors, Angles, and Projections" (`sec_mdl-geometry-linear-algebraic-ops`) → 「ベクトル、角度、射影」
- "low-rank adapter" → 「低ランクアダプタ」 (drops final `ー` per `ja/style.md` §3.3)
- "Gershgorin discs" → 「Gershgorin の円盤」, "Eckart–Young–Mirsky (theorem)" → 「Eckart–Young–Mirsky の定理」 (person names in Latin script per `ja/style.md` §5)
- "pseudospectrum / pseudospectra" → 「擬スペクトル」 (standard Japanese numerical linear algebra term)
- "eigensolver / linear solver" → 「固有値ソルバ / 線形ソルバ」 (drops final `ー` per `ja/style.md` §3.3)
- "defective (matrix / shear)" → 「欠陥（行列 / を持つせん断）」
- "rank–nullity" → 「ランク・退化次数の定理」 (retains binding glossary term `ランク` for `rank`)
- "transient growth" → 「過渡的増大」

## chapter_mdl-calculus/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Calculus and Automatic Differentiation" (chapter title) → 「微分積分と自動微分」
- "Single-Variable Calculus" / "Multivariable Calculus" → 「1 変数の微分積分」 / 「多変数の微分積分」
- "loss surface" → 「損失曲面」 (geometric surface sense alongside glossary `loss landscape` = 「損失ランドスケープ」)
- "vector calculus" → 「ベクトル解析」, "Stokes' theorem" → 「ストークスの定理」, "measure theory" → 「測度論」
- "Taylor approximation / Taylor polynomial" → 「テイラー近似 / テイラー多項式」, "Lagrange remainder" → 「ラグランジュの剰余項」
- "differential method" (matrix calculus via $df = \langle \nabla f, d\mathbf{X}\rangle$) → 「全微分法」 (distinguished from `differential calculus` = 「微分法」)
- "dual numbers" → 「二重数」, "Wengert tape" → 「Wengert テープ」, "implicit differentiation" → 「陰関数微分」, "forward-over-reverse" → 「フォワード・オーバー・リバース」
- "Fundamental Theorem of Calculus (FTC)" → 「微分積分学の基本定理」, "Fubini's theorem" → 「フビニの定理」, "Leibniz rule" → 「ライプニッツ則」, "Gaussian integral" → 「ガウス積分」, "antiderivative" → 「原始関数」
- "pathwise (gradient / estimator / reparameterization)" → 「パスワイズ（パスワイズ勾配・パスワイズ推定量・パスワイズ再パラメータ化）」

## chapter_mdl-optimization/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "conditioning" (numerical sensitivity of a problem, paired with condition number) → 「条件性」 (distinguished from probabilistic 「条件付け」 and `preconditioning` = 「前処理」, aligning with `ill-conditioning` = 「悪条件性」)
- "unit roundoff" → 「丸めの単位」 (standard Japanese numerical analysis term paired with `machine epsilon` = 「計算機イプシロン」)
- "interior-point method" / "operator-splitting method" → 「内点法」 / 「作用素分割法」
- "primal problem" / "primal optimum" → 「主問題」 / 「主最適値」 (paired with glossary `dual problem` = 「双対問題」)

## chapter_mdl-information-theory/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Information Theory and Divergences" (chapter title) → 「情報理論とダイバージェンス」
- "nat(s)" → 「nat」 (kept in Latin script per `ja/style.md` §4, avoiding ambiguity with mechanical ナット)
- "log-loss" → 「対数損失」
- "f-divergence" / "$f$-divergence" → 「f ダイバージェンス」 / 「$f$ ダイバージェンス」 (half-width space between Latin/math `f` and katakana per `ja/style.md` §2.2, overriding glossary's unspaced `fダイバージェンス`)
- "forward KL" / "reverse KL" → 「順方向 KL」 / 「逆方向 KL」, "mode-covering" / "mode-seeking" → 「モード被覆型」 / 「モード探索型」
- "Kraft's inequality", "Pinsker's inequality", "Fano's inequality", "Stein's identity" → 「クラフトの不等式」, 「ピンスカーの不等式」, 「ファノの不等式」, 「スタインの恒等式」 (established Japanese information-theory/statistics terms named after people per `ja/style.md` §5)
- "Fisher divergence", "Wasserstein $W_1$", "Sinkhorn OT" → 「フィッシャーダイバージェンス」, 「ワッサーシュタイン $W_1$」, 「シンクホーン OT」
- "dyadic source" → 「2 進情報源」
- "error-correcting code", "Kolmogorov complexity", "information geometry" → 「誤り訂正符号」, 「コルモゴロフ複雑性」, 「情報幾何学」

## chapter_tools-for-deep-learning/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Tools for Deep Learning" (part title) → 「深層学習のためのツール」
- "accelerator" → 「アクセラレータ」 (drops final `ー` per `ja/style.md` §3.3 `-or` rule)
- "distributed training" → 「分散訓練」 (paired with glossary `training` = 「訓練」)
- "workload" → 「ワークロード」, "interconnect" → 「インターコネクト」

## chapter_appendix-tools-for-deep-learning/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Notebooks" (section title, `sec_interactive_development`) → 「ノートブック」
- "cluster" / "GPU cluster" → 「クラスタ」 / 「GPU クラスタ」, "accelerator" → 「アクセラレータ」, "simulator" → 「シミュレータ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` / `-or` loanwords)
- "artifact" / "external artifact" → 「アーティファクト」 / 「外部アーティファクト」
- "open-weight" → 「オープンウェイト」
- "pinning" (revision / SHA pinning) → 「ピン留め」
- "unified memory" → 「ユニファイドメモリ」 (follows `ja/style.md` §3.3 dropping final `ー` on `メモリ`)
- "speculative decoding" → 「投機的デコーディング」 (paired with `ja/decisions.md` `"decoding" → 「デコーディング」`)
- "continuous batching" → 「連続バッチング」
- "hosted runtime" / "hosted VM" → 「ホスト型ランタイム」 / 「ホスト型 VM」 (paired with `ja/decisions.md` `"hosted notebook" → 「ホスト型ノートブック」`)
- "system sizing" / "sizing" → 「システムサイジング」 / 「サイジング」

## chapter_appendix-tools-for-deep-learning/d2l.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "API Reference" → 「API リファレンス」

## chapter_appendix-tools-for-deep-learning/hyperparameters-lib.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "base class" → 「基底クラス」 (standard object-oriented terminology in Japanese, consistent with `chapter_linear-regression/oo-design.md`)

## chapter_appendix-tools-for-deep-learning/interactive-development.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "execution counter" → 「実行カウンタ」, "debugger" → 「デバッガ」, "interpreter" → 「インタプリタ」, "variable inspector" → 「変数インスペクタ」, "timer" → 「タイマ」, "worker (process)" → 「ワーカ（プロセス）」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` / `-or` loanwords)
- "Restart Kernel and Run All Cells" / "restart and run all" → 「カーネルを再起動してすべてのセルを実行」 / 「カーネルの再起動と全セル実行」
- "orphaned kernel" → 「孤立カーネル」
- "post-mortem debugging" → 「事後デバッグ」

## chapter_appendix-tools-for-deep-learning/interactive-development.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "loopback (address)" → 「ループバック（アドレス）」, "port forwarding" → 「ポートフォワーディング」
- "preempted" / "preemption" → 「プリエンプション（強制終了）」
- "use-before-define" → 「定義前参照」
- "worker process" → 「ワーカプロセス」 (consistent with `u01` and `ja/style.md` §3.3)

## chapter_appendix-tools-for-deep-learning/hosted-notebooks.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "durable storage" → 「永続ストレージ」 ("durable" → 「永続的な」 / 「永続」)
- "scratch space" / "scratch disk" → 「スクラッチ領域」 / 「スクラッチディスク」
- "setup cell" → 「セットアップセル」, "idempotent" → 「べき等」
- "compute unit(s)" (Google Colab) → 「コンピューティングユニット」 (matches Google Colab's official Japanese UI term, solid katakana per `ja/style.md` §3.2)
- "runtime version" / "local runtime" → 「ランタイムバージョン」 / 「ローカルランタイム」
- "free tier" / "paid tier(s)" → 「無料枠」 / 「有料プラン」
- "provider" / "container" / "pointer" / "importer" → 「プロバイダ」 / 「コンテナ」 / 「ポインタ」 / 「インポータ」 (drops final `ー` per `ja/style.md` §3.3 `-er` rule)
- "Young's checkpoint law" → 「Young のチェックポイント則」 (person name in Latin script per `ja/style.md` §5)

## chapter_appendix-tools-for-deep-learning/hosted-notebooks.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "portable" / "portability" → 「ポータブル（な）」 / 「ポータビリティ」
- "secret manager" / "secret store" → 「シークレットマネージャ」 / 「シークレットストア」 (drops final `ー` on `マネージャ` per `ja/style.md` §3.3 `-er` rule)
- "Poisson process" → 「ポアソン過程」 (standard Japanese probability terminology per `ja/style.md` §5)

## chapter_appendix-tools-for-deep-learning/cloud-instances.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Cloud Computing" (section title, `sec_cloud_instances`) → 「クラウドコンピューティング」
- "hyperscaler" / "GPU specialist" / "marketplace" → 「ハイパースケーラ」 / 「GPU スペシャリスト」 / 「マーケットプレイス」 (`ja/style.md` §3.3 の `-er` 長音省略規則に従いハイパースケーラ・プロバイダ・コンシューマ・ハイパーバイザ・アロケータ・ドライバと表記)
- "cost per completed result" → 「完了した結果あたりのコスト」（スライド・表では「完了結果あたりのコスト」）
- "spot capacity" / "interruptible capacity" → 「スポット容量」 / 「中断可能な容量」
- "egress" / "egress fees" → 「エグレス」 / 「エグレス料金」
- "trust boundary" → 「信頼境界」
- "smoke test" → 「スモークテスト」

## chapter_appendix-tools-for-deep-learning/cloud-instances.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "data iterator" / "iterator" → 「データイテレータ」 / 「イテレータ」（`ja/style.md` §3.3 の `-or` 長音省略規則に従う）
- "Poisson process" → 「ポアソン過程」（`ja/style.md` §5 の人名由来の数学用語カタカナ表記規則に従う）

## chapter_appendix-tools-for-deep-learning/hardware.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "memory capacity" → 「メモリ容量」, "compute throughput" → 「演算スループット」 (paired with glossary `memory bandwidth` = 「メモリ帯域幅」 and `throughput` = 「スループット」)
- "discrete GPU" → 「ディスクリート GPU」 (standard hardware counterpart to `unified memory` = 「ユニファイドメモリ」)
- "decode bound" / "decode bandwidth bound" → 「デコード上限」 / 「デコード帯域幅上限」
- "tensor cores" / "Tensor Cores" → 「テンソルコア」 (aligns with binding glossary `tensor` = 「テンソル」)
- "street price" → 「実売価格」, "break-even" → 「損益分岐点」, "total cost of ownership (TCO)" → 「総所有コスト（TCO）」

## chapter_appendix-tools-for-deep-learning/hardware.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "pipeline-parallel" / "pipeline parallelism" → 「パイプライン並列」 (paired with glossary `data parallelism` = 「データ並列」 and `tensor parallelism` = 「テンソル並列」)
- "productive hour(s)" → 「実稼働時間」, "resale value" → 「再販価値」, "wall power" → 「コンセント消費電力」

## chapter_appendix-tools-for-deep-learning/software-ecosystem.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Ecosystem" (section title, `sec_software_ecosystem`) → 「エコシステム」
- "contamination" / "benchmark contamination" → 「データ汚染」 / 「ベンチマーク汚染」 ("contamination-resistant" → 「データ汚染に強い」)
- "model card" / "dataset card" → 「モデルカード」 / 「データセットカード」
- "revealed preference" → 「顕示選好」
- "immutable revision" → 「不変のリビジョン」 (concise slide form: 「不変リビジョン」)

## chapter_appendix-tools-for-deep-learning/software-ecosystem.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "deserialization" → 「デシリアライズ」 (paired with `ja/decisions.md` `"serialization" → 「シリアライズ」`)
- "derived artifact" → 「派生アーティファクト」
- "header" / "loader" → 「ヘッダ」 / 「ローダ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)

## chapter_appendix-tools-for-deep-learning/training-systems.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Distributed Model Training" (section heading, `sec_training_systems`) → 「分散モデル訓練」
- "scaling ladder" / "rung" → 「スケーリングラダー」 / 「段」（「第 2 段」「第 3 段」）
- "fully sharded data parallelism (FSDP)" → 「完全シャーディングデータ並列」 (paired with glossary `data parallelism` = 「データ並列」 and `decisions.md` `sharding` = 「シャーディング」)
- "context parallelism" → 「コンテキスト並列」 (paired with glossary `tensor parallelism` = 「テンソル並列」, `pipeline parallelism` = 「パイプライン並列」, `expert parallelism` = 「エキスパート並列」)
- "collective(s)" (collective communication operations) → 「集団通信」
- "launcher" → 「ランチャ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)
- "global batch" / "global batch size" → 「グローバルバッチ」 / 「グローバルバッチサイズ」
- "bubble" / "bubble fraction" (pipeline parallelism) → 「バブル」 / 「バブル率」
- "fault tolerance" → 「耐障害性」
- "elastic resumption" → 「エラスティックな再開」

## chapter_appendix-tools-for-deep-learning/training-systems.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "data-bound" / "I/O-bound" → 「データ律速」 / 「I/O 律速」 (paired with glossary `compute-bound` = 「計算律速」 and `decisions.md` `memory-bandwidth bound` = 「メモリ帯域幅律速」)
- "spot-priced instance(s)" → 「スポット価格のインスタンス」

## chapter_appendix-tools-for-deep-learning/training-systems.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "interleaved assignment" (pipeline parallelism) → 「インターリーブ割り当て」
- "delayed parameter update" → 「遅延パラメータ更新」

## chapter_appendix-tools-for-deep-learning/model-serving.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Model Serving" / "Local Serving" → 「モデルサービング」 / 「ローカルサービング」 (consistent with `decisions.md` `"serving" → 「サービング」`)
- "time per output token (TPOT)" → 「出力トークンあたりの時間（TPOT）」 (paired with glossary `time to first token` = 「最初のトークンまでの時間」)
- "service level objective (SLO)" → 「サービスレベル目標（SLO）」, "goodput" → 「グッドプット」
- "paged KV cache" → 「ページド KV キャッシュ」, "prefix caching / prefix cache" → 「プレフィックスキャッシュ」
- "radix tree" → 「基数木」
- "conformance test" → 「適合性テスト」
- "draft model" / "target model" → 「ドラフトモデル」 / 「ターゲットモデル」
- "chunked prefill" → 「チャンク化プレフィル」, "admission control" → 「受付制御」
- "wrapper" / "provider" / "datacenter" → 「ラッパ」 / 「プロバイダ」 / 「データセンタ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)

## chapter_appendix-tools-for-deep-learning/model-serving.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Operations" (section heading) → 「運用」
- "tail-latency monitoring" → 「テールレイテンシの監視」
- "client cancellation propagation" → 「クライアントのキャンセル伝播」
- "co-pinned" → 「一括ピン留め」 (extends `decisions.md` `"pinning" → 「ピン留め」`)
- "first come, first served" → 「先着順」, "preempt" → 「プリエンプトする」
- "speculative sampling" → 「投機的サンプリング」 (paired with `decisions.md` `"speculative decoding" → 「投機的デコーディング」`)

## chapter_appendix-tools-for-deep-learning/developers-guide.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Contributor Guide" → 「貢献者ガイド」, "contributor" → 「貢献者」 (matches `index.md` and `chapter_appendix-tools-for-deep-learning/index.md`)
- "source of truth" / "single source of truth" → 「唯一の正本」 ("single-source" → 「シングルソース」)
- "coding agent" → 「コーディングエージェント」 (solid katakana compound per `ja/style.md` §3.2)
- "stale" / "staleness" (of cached notebook outputs) → 「陳腐化した（陳腐化）」 (consistent noun/verb/adjective across prose, slides, and exercises; glossed in `u02` body text)
- "maintainer" / "counter" / "extractor" / "helper" → 「メンテナ」 / 「カウンタ」 / 「抽出器」 / 「ヘルパ」 (drops final `ー` on `-er` loanwords per `ja/style.md` §3.3)

## chapter_appendix-tools-for-deep-learning/developers-guide.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "stale" / "staleness" → 「陳腐化した（陳腐化）」 (glossed on first body-text occurrence as `陳腐化（stale）`)
- "probabilistic bisection" → 「確率的二分法」
- "binary entropy" → 「二値エントロピー」
- "linter" / "tracer" / "reviewer" → 「リンタ」 / 「トレーサ」 / 「レビュア」 (drops final `ー` on `-er` loanwords per `ja/style.md` §3.3)

## chapter_appendix-tools-for-deep-learning/utils.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "helper" / "helper function" → 「ヘルパ」 / 「ヘルパ関数」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)
- "marker" (plot marker / completion marker) → 「マーカ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)
- "introspection" → 「イントロスペクション」 (standard OOP/Python term in Japanese)
- "bucketing" → 「バケッティング」 (batch bucketing for JIT compilation)

## chapter_transformers/gpt.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "weight tying" / "tying" (output projection to input embeddings) → 「重み共有」 (consistent with `chapter_transformers/index.md#u01` decision over proposed glossary `重み結合`)
- "selective weight decay" → 「選択的重み減衰」
- "byte-level" → 「バイトレベル」
- "run-to-run spread" → 「実行間のばらつき」

## chapter_transformers/gpt.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "weight tying" → 「重み共有」 (consistent with `u01` and `chapter_transformers/index.md#u01`)
- "top-$k$" / "Top-$p$ (nucleus) sampling" → 「top-$k$」 / 「Top-$p$（ニュークリアス）サンプリング」
- "low-rank latent compression (MLA)" → 「低ランク潜在圧縮（low-rank latent compression、MLA）」
- "decoupled RoPE" → 「分離型 RoPE」

## chapter_transformers/vision-transformer.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Vision Transformer (ViT)" → 「Vision Transformer（ViT）」 (architecture proper name kept in Latin script per `ja/style.md` §3.2 and `ja/decisions.md`)
- "grid topology" → 「グリッドのトポロジー」
- "bag of patches" → 「パッチの集合（bag of patches）」 (body) / 「Bag-of-Patches」 (slide)
- "sinusoidal code" → 「正弦波符号」
- "masked patch reconstruction" → 「マスク付きパッチ再構成」

## chapter_transformers/vision-transformer.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "permutation test" → 「置換検定」
- "mean attention distance" → 「平均アテンション距離」

## chapter_transformers/scaling-laws.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "neural scaling laws" → 「ニューラルスケーリング則」 (extends glossary `scaling law` = 「スケーリング則」)
- "non-embedding parameters" → 「非埋め込みパラメータ」 (pairs `embedding` = 「埋め込み」 with `parameter` = 「パラメータ」)
- "power law" / "power-law" → 「べき乗則」 (standard mathematical and ML terminology)
- "data budget" / "token budget" / "step budget" / "FLOP budget" → 「データ予算」 / 「トークン予算」 / 「ステップ予算」 / 「FLOP 予算」 (consistent with glossary `compute budget` = 「計算予算」)
- "over-training" / "over-trained" (training past the Chinchilla optimum) → 「過剰訓練」 / 「過剰訓練された」 (distinguished from `overfitting` = 「過学習」)
- "lifetime compute" / "lifetime inference compute" → 「生涯計算量」 / 「生涯推論計算量」 (paired with glossary `training compute` = 「訓練計算量」)
- "IsoFLOP" (contours / budget lines / trade-off) → 「IsoFLOP」（「IsoFLOP 等高線」「IsoFLOP 予算線」「IsoFLOP トレードオフ」）
- "counter" (hardware / operator counter) → 「カウンタ」 (drops final `ー` per `ja/style.md` §3.3 `-er` rule)

## chapter_transformers/scaling-laws.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "nonlinear least squares" → 「非線形最小二乗法」 (extends glossary `least squares` = 「最小二乗法」)
- "lifetime inference compute" → 「生涯推論計算量」 (consistent with `u01` `"lifetime compute" → 「生涯計算量」`)

## chapter_transformers/moe.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Sparse Mixture-of-Experts (MoE)" → 「スパース混合エキスパート（MoE）」（glossary `mixture of experts` = 「混合エキスパート」および `chapter_transformers/index.md` と統一）
- "stored parameters" / "active parameters" → 「保持パラメータ（保持パラメータ数）」 / 「アクティブパラメータ（アクティブパラメータ数）」
- "Top-$k$ routing" / "Top-k routing" → 「Top-$k$ ルーティング」 / 「Top-k ルーティング」（`ja/style.md` §2.2 の半角英数・数式と日本語の間の半角スペース規則に従う）
- "dispatch" / "scatter" / "gather" → 「ディスパッチ（する）」 / 「スキャッタ（する）」 / 「ギャザー（する）」
- "token dropping" / "drop" → 「トークンドロップ」 / 「ドロップ（する）」
- "routing collapse" → 「ルーティング崩壊」（`token collapse` = 「トークン崩壊」と対）
- "Switch auxiliary (load-balancing) loss" → 「Switch 補助（負荷分散）損失」
- "loss-free dynamic router bias control" / "loss-free bias controller" → 「損失不要の動的ルータバイアス制御」 / 「損失不要（の）バイアスコントローラ」（`ja/style.md` §3.3 の `-er` 長音省略規則に従う）
- "frozen router" → 「凍結ルータ」

## chapter_transformers/moe.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "fine-grained expert segmentation" / "fine-grained" / "coarse" → 「細粒度エキスパート分割」 / 「細粒度」 / 「粗粒度」
- "IsoFLOP" → 「IsoFLOP」（スケーリング則の専門用語としてラテン文字表記を維持）

## chapter_transformers/further-reading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Further Reading" (page/section heading) → 「関連文献」 (consistent with `decisions.md`)
- "post-sublayer normalization" → 「サブ層後正規化」, "Sandwich-LN / Peri-LN" → 「Sandwich-LN / Peri-LN」
- "soft-capping" / "soft-capped logits" → 「ソフトキャッピング」 / 「ロジットのソフトキャッピング」
- "parallel Transformer block" → 「並列 Transformer ブロック」
- "weight absorption" (in MLA decoding) → 「重み吸収」
- "self-distillation" → 「自己蒸留」
- "over-train" (training past the Chinchilla compute-optimal ratio) → 「過剰に訓練する」 (distinguished from `overfit` = 「過学習」)
- "sparse upcycling" → 「スパースアップサイクリング」

## chapter_recurrent-modern/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Gated Recurrent Networks" (section heading, `sec_lstm`) → 「ゲート付きリカレントネットワーク」
- "linear dynamical system" / "linear system (in continuous time)" → 「線形力学系」 / 「（連続時間の）線形システム」 (control-theory dynamical system sense, distinct from glossary's proposed algebraic `連立一次方程式`)
- "mixer" / "sequence mixer" → 「ミキサ」 / 「系列ミキサ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)
- "delta rule" → 「デルタ則」
- "exact recall" / "Delayed Recall" / "Associative Recall" → 「正確な想起」 / 「遅延想起」 / 「連想想起」 (memory retrieval sense of *recall*, distinct from classification metric `再現率`)
- "Selective Copying" / "String Reversal" / "Key Rebinding" → 「選択的コピー」 / 「文字列反転」 / 「キー再束縛」
- "zero-order-hold discretization" → 「ゼロ次ホールド離散化」
- "block-semiseparable" → 「ブロック準分離」
- "LeCun initialization" → 「LeCun 初期化」 (follows `ja/style.md` §5 and §2.2)

## chapter_recurrent-modern/lstm.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "plain recurrent network" / "plain RNN" → 「素のリカレントネットワーク」 / 「素の RNN」 (aligns with `bptt.md` `"vanilla RNN" → 「素の RNN」`)
- "memory cell" → 「記憶セル」, "additive memory" → 「加法的な記憶」 (aligns with glossary `long short-term memory` = 「長・短期記憶」 and temporal memory = 「記憶」)
- "input node" → 「入力ノード」
- "delayed recall" → 「遅延想起」 (memory recall sense, paired with `decisions.md` `"associative recall" → 「連想想起」`, distinguished from classification recall = 「再現率」)
- "half-life" → 「半減期」
- "sequence mixer" / "mixer" → 「系列ミキサ」 / 「ミキサ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)

## chapter_recurrent-modern/lstm.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "candidate state" → 「候補状態」
- "chrono initialization" → 「クロノ初期化」
- "keep-gate" → 「保持ゲート」 (e.g. `Keep-gate LSTM and GRU` → 「保持ゲート付き LSTM と GRU」)
- "LeCun normal initialization" / "LeCun normal" → 「LeCun 正規初期化」 (`ja/style.md` §5・§2.2 に従い人名ラテン文字＋半角スペースを維持、`He 初期化`・`Xavier 初期化` と統一)
- "nats/byte" → 「nat/byte」 (`ja/decisions.md` の `"nat / nats" → 「nat」` および `ja/style.md` §4 の単位記号維持規則に従う)

## chapter_recurrent-modern/lstm.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "bidirectional (network / mixer / predictor)" → 「双方向（ネットワーク / ミキサ / 予測器）」

## chapter_recurrent-modern/seq2seq.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Sequence-to-Sequence Learning" (page/section heading) → 「系列対系列学習」 (matches `ja/decisions.md` `"sequence-to-sequence" (modifier) → 「系列対系列」`)
- "source (sequence / sentence / token / string)" / "target (sequence / token)" → 「ソース（系列 / 文 / トークン / 文字列）」 / 「ターゲット（系列 / トークン）」
- "reference (translation)" / "candidate (translation)" → 「参照訳」 / 「候補訳」
- "near miss" / "paraphrase" → 「ニアミス」 / 「言い換え」
- "string reversal" → 「文字列反転」
- "LLM judge(s)" → 「LLM ジャッジ」
- "free-running decoding" → 「フリーランニングのデコーディング」

## chapter_recurrent-modern/seq2seq.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "near miss" / "paraphrase" → 「ニアミス」 / 「言い換え」 (consistent with `u01`)
- "Fano's inequality" → 「ファノの不等式」 (follows `ja/style.md` §5 and `ja/decisions.md`)
- "binary entropy" → 「二値エントロピー」

## chapter_recurrent-modern/linear-recurrence.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "affine recurrence" → 「アフィン漸化式」 (paired with glossary `linear recurrence` = 「線形漸化式」)
- "minimal GRU (`minGRU` / `MinGRU`)" → 「最小 GRU」（略称 `minGRU` およびクラス名 `` `MinGRU` `` はラテン文字維持）
- "write vector / write(s)" → 「書き込みベクトル / 書き込み」
- "doubling schedule / doubling scan" (Hillis–Steele) → 「倍増スケジュール / 倍増スキャン」
- "work-efficient (scan / tree schedule)" → 「演算量効率的（なスキャン / 木スケジュール）」; "work" (in work–depth model) → 「演算量」
- "up-sweep/down-sweep tree" → 「アップスイープ・ダウンスイープ木」
- "combine" (binary associative scan operator $\bullet$) → 「結合演算」（動詞は「結合する」）
- "sequence mixer / mixer" → 「系列ミキサ / ミキサ」（`ja/style.md` §3.3 の `-er` 長音省略規則に従う）

## chapter_recurrent-modern/linear-recurrence.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "log–log axes" → 「両対数軸」 (consistent with `decisions.md` `"log–log slope" → 「両対数傾き」`)
- "continuous-time state space model" → 「連続時間状態空間モデル」 (combines glossary `continuous-time` = 「連続時間」, `state space` = 「状態空間」, `model` = 「モデル」)

## chapter_recurrent-modern/ssm.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Structured State Space Models" → 「構造化状態空間モデル」 (extends `decisions.md` `"state space model" → 「状態空間モデル」`)
- "zero-order hold (ZOH)" → 「ゼロ次ホールド（ZOH）」 (standard control-engineering term)
- "linear time-invariant (LTI)" / "time-invariant" → 「線形時不変（LTI）」 / 「時不変」 (standard signal-processing and control term)
- "causal convolution" → 「因果畳み込み」 ("circular convolution" → 「巡回畳み込み」)
- "timescale" / "multi-timescale" → 「時間スケール」 / 「多重時間スケール」
- "half-life" → 「半減期」
- "rate" (SSM continuous-time eigenvalue $\lambda_n$) → 「レート」 ("real rate" → 「実レート」, "complex rate" → 「複素レート」, "decay rate" → 「減衰率」)
- "leaky integrator" → 「漏れ積分器」
- "impulse response" → 「インパルス応答」
- "delayed recall" / "long-range recall" → 「遅延想起」 / 「長距離想起」 (sequence-memory recall, aligned with `decisions.md` `"associative recall" → 「連想想起」`)

## chapter_recurrent-modern/ssm.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "diagonal state space layer" → 「対角状態空間層」
- "forward Euler rule" / "backward Euler rule" / "bilinear rule" → 「前進オイラー法」 / 「後退オイラー法」 / 「双一次法」 (standard Japanese numerical ODE discretization terms)
- "content blindness" → 「内容への不感性」 (paired with `decisions.md` `"order blindness" → 「順序への不感性」`)

## chapter_recurrent-modern/mamba.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "selective copying" → 「選択的コピー」
- "content-based selection" / "content selection" / "content-selective" → 「内容に基づく選択」 / 「内容選択」 / 「内容選択的」
- "marked symbol" → 「マーク付き記号」
- "time-invariant" → 「時不変」
- "zero-order hold (ZOH)" → 「ゼロ次ホールド（ZOH）」
- "half-life" → 「半減期」, "multi-timescale" → 「多重時間スケール」
- "write vector" / "read vector" → 「書き込みベクトル」 / 「読み出しベクトル」
- "sequence mixer" → 「系列ミキサ」（`ja/style.md` §3.3 の `-er` 長音省略規則に従う）
- "parallel associative scan" → 「並列結合スキャン」, "adjoint scan" → 「随伴スキャン」, "doubling scan / doubling round" → 「倍増スキャン / 倍増ラウンド」

## chapter_recurrent-modern/mamba.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "depthwise causal convolution" → 「デプスワイズ因果畳み込み」
- "spike ratio" → 「スパイク比」
- "reverse-time" → 「逆時間」

## chapter_recurrent-modern/matrix-state.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Matrix-State Recurrences" / "matrix recurrence" → 「行列状態漸化式」 / 「行列漸化式」 (paired with glossary `linear recurrence` = 「線形漸化式」)
- "state space duality (SSD)" → 「状態空間双対性（SSD）」 (paired with glossary `state space` = 「状態空間」 and `duality` = 「双対性」)
- "gated linear attention (GLA)" → 「ゲート付き線形アテンション（GLA）」
- "block-semiseparable" / "semiseparable" → 「ブロック半分離可能」 / 「半分離可能」
- "codebook" → 「コードブック」
- "nearest-neighbor recall" / "recall" (memory recall metric) → 「最近傍想起率」 / 「想起率」 (distinguished from binary classification `recall` = 「再現率」, aligning with `decisions.md` `"associative recall" → 「連想想起」`)

## chapter_recurrent-modern/matrix-state.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dual form" / "primal" → 「双対形式」 / 「主（形式・漸化式）」
- "sequence mixer" → 「系列ミキサ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)
- "multi-query associative recall" → 「マルチクエリ連想想起」 (extends `decisions.md` `"associative recall" → 「連想想起」`)
- "rebound key" → 「再束縛キー」 (extends `decisions.md` `"rebinding" → 「再束縛」`)

## chapter_recurrent-modern/deltanet.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "delta rule" → 「デルタ則」 (established Japanese ML term for Widrow–Hoff delta rule, matching `decisions.md` `"delta update rule" → 「デルタ更新則」`)
- "additive matrix memory" / "additive writes" → 「加算型行列メモリ」 / 「加算的書き込み」
- "bind" / "rebind" / "rebinding" → 「束縛する」 / 「再束縛する」 / 「再束縛」 (matches `decisions.md` `"rebinding" → 「再束縛」`)
- "read before writing" / "read-before-write" → 「書き込み前に読み出す」 / 「書き込み前読み出し」
- "write strength" → 「書き込み強度」
- "Householder reflection" → 「ハウスホルダー鏡映」 (standard Japanese numerical linear algebra term, following `ja/style.md` §5 katakana rule for mathematical concepts named after people)
- "signal erosion" → 「信号浸食」 (distinguished from `decay` = 「減衰」 and `interference` = 「干渉」)
- "WY representation" / "WY form" → 「WY 表現」 / 「WY 形式」
- "solver" → 「ソルバ」 (follows `ja/style.md` §3.3 `-er` rule and `decisions.md` `"linear solver" → 「線形ソルバ」`)

## chapter_recurrent-modern/deltanet.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "effective corrected value" → 「実効補正値」
- "strictly lower-triangular" / "unit-lower-triangular" → 「狭義下三角」 / 「単位下三角」 (standard Japanese linear algebra terminology)
- "state tracking" → 「状態追跡」
- "mixer" / "sequence mixer" → 「ミキサ」 / 「系列ミキサ」 (follows `ja/style.md` §3.3 `-er` rule)
- "gated linear attention" → 「ゲート付き線形アテンション」 (aligned with glossary `linear attention` = 「線形アテンション」)

## chapter_recurrent-modern/deltanet.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "cross-moment" → 「交差モーメント」, "Gram inverse" → 「グラム逆行列」
- "proximal implicit step" / "implicit step" → 「近接陰的ステップ」 / 「陰的ステップ」
- "recursive least squares" → 「逐次最小二乗法」 (established Japanese signal-processing/control/ML term for RLS)
- "fast weights" / "slow weights" → 「高速重み」 / 「低速重み」 (paired with `decisions.md` `"fast weight(s)" → 「高速重み」`)
- "bit capacity" → 「ビット容量」

## chapter_recurrent-modern/hybrids.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "hybrid architecture" / "hybrid" → 「ハイブリッドアーキテクチャ」 / 「ハイブリッド」
- "in-context retrieval" → 「コンテキスト内検索」 (paired with glossary `retrieval` = 「検索」 and `context length` = 「コンテキスト長」)
- "multi-query associative recall" → 「マルチクエリ連想想起」; "recall" (memory recall task/metric) → 「想起」 / 「想起率」 (distinguished from binary classification `recall` = 「再現率」)
- "bit-counting bound" / "counting bound" → 「ビット数え上げ限界」 / 「数え上げ限界」
- "Welch's bound" / "Welch floor" → 「Welch の下界」 / 「Welch の下限」 (person name kept in Latin script per `ja/style.md` §5)
- "mixer" / "sequence mixer" → 「ミキサ」 / 「系列ミキサ」 (drops final `ー` per `ja/style.md` §3.3 `-er` rule)
- "pair load" / "load" → 「ペア負荷」 / 「負荷」

## chapter_recurrent-modern/hybrids.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "gated linear attention" → 「ゲート付き線形アテンション」
- "bands of positions" / "position band" → 「位置の帯域」
- "lagging run" → 「遅れをとった実行」

## chapter_recurrent-modern/hybrids.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "selective copying" → 「選択的コピー」
- "triangular solve" → 「三角求解」

## chapter_recurrent-modern/further-reading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "long convolution(s)" → 「長い畳み込み」 (paired with glossary `convolution` = 「畳み込み」)
- "log-linear attention" → 「対数線形アテンション」 (paired with glossary `linear attention` = 「線形アテンション」)
- "constant-depth threshold circuit class" → 「定数深さしきい値回路クラス」, "logarithmic-depth circuits" → 「対数深さ回路」
- "reflection" (orthogonal reflection matrix $\mathbf{I} - 2\mathbf{k}\mathbf{k}^\top$) → 「鏡映」 (consistent with `deltanet.md`)
- "trapezoidal discretization" → 「台形離散化」 (paired with glossary `discretization` = 「離散化」)
- "reverse-complement-equivariant" → 「逆相補鎖同変」 (genomics reverse complement 「逆相補鎖」 + `decisions.md` `equivariance` = 「同変性」)

## chapter_computational-performance/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Computational Performance" (chapter title, `chap_performance`) → 「計算性能」 (consistent with `ja/style.md` §7.1 and `chapter_advanced/index.md`)
- "The Performance Model" (section title, `sec_perf_model`) → 「性能モデル」
- "bandwidth roof" / "compute roof" → 「帯域幅ルーフ」 / 「計算ルーフ」 (paired with `decisions.md` `"compute roof" → 「計算ルーフ」` and glossary `roofline` = 「ルーフライン」)
- "host dispatch" / "dispatch" → 「ホストディスパッチ」 / 「ディスパッチ」
- "operator fusion" / "operation fusion" → 「演算融合」
- "CUDA graph(s)" → 「CUDA グラフ」
- "collective(s)" (distributed communication primitives) → 「集合通信」
- "ring allreduce" → 「リング allreduce」 (`allreduce` kept in Latin script)

## chapter_computational-performance/performance-model.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "performance model" → 「性能モデル」 (standard Japanese HPC/systems term, avoiding katakana パフォーマンス per `ja/style.md` §7.1)
- "bandwidth-bound" / "overhead-bound" → 「帯域幅律速」 / 「オーバーヘッド律速」 (paired symmetrically with glossary `compute-bound` = 「計算律速」)
- "bandwidth roof" / "compute roof" → 「帯域幅ルーフ」 / 「計算ルーフ」 (consistent with `ja/decisions.md` `kv-cache.md#u01`)
- "tensor core(s)" / "Tensor Cores" → 「テンソルコア」
- "host dispatch" / "enqueue" → 「ホストディスパッチ」 / 「エンキュー」
- "device barrier" → 「デバイスバリア」

## chapter_computational-performance/performance-model.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "kernel fusion" / "operator fusion" → 「カーネル融合」 / 「演算子融合」 (consistent with `ja/decisions.md` `"fused kernel" → 「融合カーネル」`)
- "profiler" / "profiling" → 「プロファイラ」 / 「プロファイリング」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)

## chapter_computational-performance/performance-model.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "streaming multiprocessor (SM)" → 「ストリーミングマルチプロセッサ（SM）」 (follows `ja/style.md` §3.3 dropping final `ー` on `プロセッサ`)
- "wave" (GPU thread-block wave) → 「ウェーブ」

## chapter_computational-performance/hardware.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "arithmetic intensity" / "intensity" ($I$) → 「演算密度」 (consistent with `attention-at-scale`, `kv-cache`, and `appendix-tools-for-deep-learning/hardware`)
- "bandwidth-bound" / "latency-bound" → 「帯域幅律速」 / 「レイテンシ律速」 (paired with glossary `compute-bound` = 「計算律速」 and `memory-bound` = 「メモリ律速」)
- "sector" (GPU memory transaction unit) → 「セクタ」 (drops final `ー` per `ja/style.md` §3.3), "DRAM row" → 「DRAM 行」（初出「*行*（row）」）
- "shoreline" (die perimeter for I/O) → 「ショアライン」
- "silicon interposer" → 「シリコンインターポーザ」 (drops final `ー` per `ja/style.md` §3.3)
- "reticle limit" → 「レチクル限界」
- "kernel fusion" / "operator fusion" → 「カーネル融合」 / 「演算子融合」 (paired with `decisions.md` `"fused kernel" → 「融合カーネル」`)
- "activation recomputation" → 「活性化値の再計算」
- "tensor core(s)" / "Tensor Core(s)" → 「テンソルコア」

## chapter_computational-performance/hardware.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "streaming multiprocessor (SM)" → 「ストリーミングマルチプロセッサ（SM）」 (drops final `ー` on `プロセッサ` per `ja/style.md` §3.3)
- "warp" → 「ワープ」
- "Little's law" → 「リトルの法則」 (person-named law in katakana per `ja/style.md` §5)
- "in flight" (bytes/requests in flight) → 「インフライト」
- "systolic array" → 「シストリックアレイ」
- "fused multiply-add" → 「融合積和演算」 (paired with `decisions.md` `"multiply-add" → 「積和演算」`)
- "microscaling" → 「マイクロスケーリング」
- "pageable (memory)" → 「ページャブル（メモリ）」 (paired with `decisions.md` `"pinned (page-locked) host memory" → 「ピン留め（ページロック）ホストメモリ」`)
- "copy engine" → 「コピーエンジン」, "staging buffer" → 「ステージングバッファ」
- "peer-to-peer (P2P)" → 「ピアツーピア（P2P）」
- "memory sweep time" ($C/\beta$) → 「メモリスイープ時間」
- "rules of thumb" → 「経験則」

## chapter_computational-performance/hardware.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "allreduce" → 「オールリデュース（allreduce）」
- "pointer chasing" → 「ポインタチェイシング（ポインタ追跡）」 (drops final `ー` on `ポインタ` per `ja/style.md` §3.3)
- "cache line" → 「キャッシュライン」
- "latency hiding" → 「レイテンシ隠蔽」

## chapter_computational-performance/compilation.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "computation graph" → 「計算グラフ」
- "graph capture" → 「グラフキャプチャ」, "graph break" → 「グラフブレーク」 (standard Japanese PyTorch / Dynamo terminology)
- "operation fusion" / "operator fusion" → 「演算融合」 / 「演算子融合」 (paired with `decisions.md` `"fused kernel" → 「融合カーネル」`)
- "kernel replay" → 「カーネルリプレイ」 (paired with `decisions.md` `"replay" → 「リプレイ」`)
- "bandwidth-bound" / "overhead-bound" → 「帯域幅律速」 / 「オーバーヘッド律速」 (paired with glossary `compute-bound` = 「計算律速」)
- "placeholder" / "compiler" / "register" → 「プレースホルダ」 / 「コンパイラ」 / 「レジスタ」 (drops final `ー` per `ja/style.md` §3.3)
- "lowering" / "lower" (a compiler IR to kernels) → 「ローワリング」 / 「ローワリングする」
- "bucketing" / "bucket" → 「バケット化」 / 「バケット化する」

## chapter_computational-performance/compilation.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "CUDA graph" → 「CUDA グラフ」, "XLA command buffer" → 「XLA コマンドバッファ」
- "break-even" → 「損益分岐点」 / 「損益分岐」 ("break-even step $n^*$" → 「損益分岐ステップ $n^*$」)
- "wrapper" / "pointer" / "counter" / "scheduler" / "driver" → 「ラッパ」 / 「ポインタ」 / 「カウンタ」 / 「スケジューラ」 / 「ドライバ」 (drops final `ー` per `ja/style.md` §3.3)

## chapter_computational-performance/memory-precision.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "global batch" → 「グローバルバッチ」 (paired with `ja/decisions.md` `"micro-batch" → 「マイクロバッチ」` and `"full-batch" → 「フルバッチ」`)
- "measured peak memory" → 「実測ピークメモリ」, "planned peak memory" → 「計画ピークメモリ」
- "Tensor Core" / "tensor cores" → 「Tensor Core」 (hardware unit name kept in Latin script per `ja/style.md` §5)
- "allocator" → 「アロケータ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-or` loanwords, consistent with `キャッシングアロケータ`)

## chapter_computational-performance/memory-precision.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "selective recomputation" → 「選択的再計算」 (glossed on first occurrence)
- "chunked cross-entropy loss" → 「チャンク化交差エントロピー損失」
- "roofline floor" → 「ルーフライン下限」 (paired with glossary `roofline` = 「ルーフライン」 and `ja/decisions.md` `"loss floor" → 「損失の下限」`)
- "significand" → 「仮数部」 (IEEE 754 standard Japanese term, matching `ja/decisions.md` `"mantissa" → 「仮数部」`)

## chapter_computational-performance/memory-precision.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "buffer donation" → 「バッファ譲渡」 (JAX `donate_argnums` terminology)
- "unscale" → 「アンスケール」 (loss scaling inverse step in mixed-precision training)

## chapter_computational-performance/multiple-gpus.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "multi-GPU" → 「マルチ GPU」 (`ja/style.md` §2.2・§3.2 のカタカナ＋ラテン文字間スペース規則に従う)
- "collective communication" / "collective primitives" → 「集団通信」 / 「集団通信プリミティブ」
- "allreduce" / "ring allreduce" / "star allreduce" → 「allreduce」 / 「リング allreduce」 / 「スター allreduce」 (glossary の `reduce-scatter` = 「reduce-scatter」・`all-gather` = 「all-gather」と統一)
- "broadcast" → 「ブロードキャスト」 (日本語の並列計算・深層学習で定着したカタカナ表記)
- "host dispatch" → 「ホストディスパッチ」
- "bus bandwidth" → 「バス帯域幅」
- "start-up latency" → 「起動レイテンシ」

## chapter_computational-performance/multiple-gpus.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "latency–bandwidth model" → 「レイテンシ・帯域幅モデル」
- "gradient fusion" → 「勾配融合」 (`decisions.md` の `"fused kernel" → 「融合カーネル」` と整合)
- "overhead-bound" / "host-bound" → 「オーバーヘッド律速」 / 「ホスト律速」 (`compute-bound` = 「計算律速」・`memory-bandwidth bound` = 「メモリ帯域幅律速」と対)
- "launcher" / "host launcher" / "single-launcher" → 「ランチャ」 / 「ホストランチャ」 / 「単一ランチャ」 (`ja/style.md` §3.3 の `-er` 長音省略規則に従う)
- "strong scaling" / "weak scaling" → 「強スケーリング」 / 「弱スケーリング」
- "Amdahl's law" / "Gustafson's law" → 「アムダールの法則」 / 「グスタフソンの法則」 (`ja/style.md` §5 の人名由来法則のカタカナ規則に従う)
- "speedup" → 「高速化率」（定量的な倍率 $R(k)$）/ 「高速化」（定性的な速度向上）

## chapter_computational-performance/multi-gpu-practice.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "allreduce" / "ring allreduce" → 「オールリデュース」 / 「リングオールリデュース」 (standard katakana in Japanese distributed DL/HPC literature, paired with 「ブロードキャスト」「ギャザー」「スキャッタ」「リダクション」)
- "collective (operation / library)" → 「集団通信（集団通信演算 / 集団通信ライブラリ）」 (established MPI/NCCL/HPC term in Japanese)
- "bucket" / "bucketing" / "bucket model" → 「バケット」 / 「バケット化」 / 「バケットモデル」
- "overlap" (of communication and computation) → 「オーバーラップ」 / 「オーバーラップさせる」
- "exposed communication (time)" / "exposed overhead" → 「露出通信（時間）」 / 「露出オーバーヘッド」
- "weak scaling" / "strong scaling" / "scaling efficiency" → 「弱スケーリング」 / 「強スケーリング」 / 「スケーリング効率」
- "single program, multiple data (SPMD)" → 「単一プログラム複数データ（single program, multiple data、SPMD）」
- "device mesh" / "partitioner" → 「デバイスメッシュ」 / 「パーティショナ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)
- "streaming multiprocessor (SM)" / "SM contention" → 「ストリーミングマルチプロセッサ（SM）」 / 「SM 競合」 (follows `ja/style.md` §3.3 `プロセッサ`)
- "bus bandwidth" → 「バス帯域幅」, "copy engine" → 「コピーエンジン」, "peer-to-peer (P2P)" → 「ピアツーピア（P2P）」
- "one-step check" → 「1 ステップチェック」 (consistent with `decisions.md` 「勾配チェック」「座標チェック」)

## chapter_computational-performance/multi-gpu-practice.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "synchronized batch normalization" → 「同期バッチ正規化」
- "checksum" → 「チェックサム」
- "calibrate" → 「較正する」

## chapter_computational-performance/sharding.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "training state" → 「訓練状態」 (glossed on first occurrence)
- "Fully Sharded Data Parallelism (FSDP)" → 「完全シャーディングデータ並列（Fully Sharded Data Parallelism、FSDP）」 (slides/tables thereafter use `FSDP` or `完全シャーディングデータ並列`)
- "collective" / "collective operation(s)" / "collective phase(s)" → 「集団通信」 / 「集団通信演算」 / 「集団通信フェーズ」; individual primitives `allreduce`, `ring allreduce`, `reduce-scatter`, `all-gather`, `all-to-all` → 「allreduce」, 「リング allreduce」, 「reduce-scatter」, 「all-gather」, 「all-to-all」 (aligned with glossary `all-gather` and `reduce-scatter`)
- "bus bandwidth" → 「バス帯域幅」 (paired with `memory bandwidth` = 「メモリ帯域幅」)
- "mixture(s) of experts" → 「混合エキスパート」
- "paper prediction" → 「机上予測」

## chapter_computational-performance/sharding.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "buffer donation" / "donate" → 「バッファ譲渡」 / 「譲渡する（バッファ譲渡する）」
- "prefetching" → 「プリフェッチ」
- "column-wise" / "row-wise" → 「列方向」 / 「行方向」
- "link-bound" → 「リンク律速」 (paired with `compute-bound` = 「計算律速」 and `memory-bandwidth bound` = 「メモリ帯域幅律速」)

## chapter_computational-performance/fast-transformer.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Amdahl's law" → 「アムダールの法則」 (follows `ja/style.md` §5 katakana rule for laws named after people)
- "kernel fusion" → 「カーネル融合」 (paired with `decisions.md` `"fused kernel" → 「融合カーネル」`)
- "bandwidth-bound" → 「帯域幅律速」 (paired with glossary `compute-bound` = 「計算律速」 and `decisions.md` `"memory-bandwidth bound" → 「メモリ帯域幅律速」`)
- "tensor core(s)" / "Tensor Core(s)" → 「テンソルコア」
- "allreduce" → 「オールリデュース」 (consistent with katakana collective terms `シャーディング`, `ギャザー`, `スキャッタ`)
- "reduced precision" → 「低精度化」
- "rung" (optimization stage in prose/slides) → 「段階」 (avoids literal metaphor translationese and distinguishes stages from training `ステップ`)

## chapter_computational-performance/fast-transformer.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "model FLOPs utilization (MFU)" → 「モデル FLOPs 利用率（MFU）」
- "weak scaling" / "strong scaling" → 「弱スケーリング」 / 「強スケーリング」
- "peer-to-peer access" → 「ピアツーピアアクセス」
- "log-perplexity" → 「対数パープレキシティ」

## chapter_computational-performance/fast-transformer.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "causality gate" (autoregressive causal-mask check) → 「因果性ゲート」 (distinguished from statistical causality = 「因果関係」)

## chapter_computational-performance/further-reading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "collective algorithm" / "collective library" → 「集合通信アルゴリズム」 / 「集合通信ライブラリ」
- "cache-aware roofline" → 「キャッシュ考慮型ルーフライン」
- "Hardware FLOPs utilization (HFU)" → 「ハードウェア FLOPs 利用率（HFU）」
- "block-scaled format" / "microscaling" → 「ブロックスケール形式」 / 「マイクロスケーリング」
- "regional compilation" → 「領域コンパイル」
- "selective recomputation" → 「選択的再計算」
- "recursive halving and doubling" → 「再帰的半減・倍増法」
- "parameter server" → 「パラメータサーバー」（`ja/style.md` §3.3 に従い `パラメータ` は長音省略、`サーバー` は長音維持）

## chapter_computational-performance/legacy-multigpu-lib.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "multi-GPU" → 「マルチ GPU」 (follows `ja/style.md` §2.2 spacing between katakana and `GPU`)
- "helper" → 「ヘルパ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)

## chapter_optimization/legacy-timer-lib.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "(build-only, unlisted)" → 「（ビルド専用、非掲載）」 (consistent with `chapter_appendix-tools-for-deep-learning/hyperparameters-lib.md#u01`)

## chapter_recurrent-modern/legacy-tokenizer-lib.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "shim" → 「シム」 (standard software engineering katakana term for compatibility shims)
- "(build-only, unlisted)" → 「（ビルド専用、非掲載）」 (consistent with `chapter_appendix-tools-for-deep-learning/hyperparameters-lib.md`)

## chapter_reinforcement-learning/policy-gradient.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "log-derivative trick" → 「対数微分トリック」 (standard Japanese RL/ML term)
- "stochastic gradient ascent" → 「確率的勾配上昇法」 (paired with glossary `stochastic gradient descent` = 「確率的勾配降下法」 and `decisions.md` `"gradient ascent" → 「勾配上昇法」`)
- "REINFORCE estimator" → 「REINFORCE 推定量」 (consistent with `baselines.md`)

## chapter_reinforcement-learning/policy-gradient.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "running example(s)" → 「共通例題」 (consistent with `mdp.md`)
- "exact-match verifier" → 「完全一致の検証器」 (consistent with `mdp.md`)

## chapter_reinforcement-learning/policy-gradient.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "discounted state-occupancy distribution" → 「割引状態占有分布」 (consistent with `ppo.md`)

## chapter_reinforcement-learning/deep-rl.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "tangent kernel" / "neural tangent kernel" → 「接カーネル」 / 「ニューラル接カーネル」 (standard Japanese DL terminology, matching `decisions.md` `"tangent line" → 「接線」` and glossary `kernel` = 「カーネル」)
- "truncation bootstrap" → 「打ち切りブートストラップ」 (pairs `mdp.md` `"truncation" → 「打ち切り」` with `ppo.md` `"bootstrap" → 「ブートストラップ」`)
- "interference" (cross-state interference under function approximation) → 「干渉」 (standard Japanese RL/ML term contrasted with generalization = 「汎化」)
- "explained variance" → 「説明分散」 (standard statistical/RL metric term)

## chapter_reinforcement-learning/deep-rl.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Gaussian policy" → 「ガウス方策」 (consistent with `decisions.md` `"squashed Gaussian policy" → 「スクワッシュされたガウス方策」` and `ja/style.md` §5)
- "deterministic policy gradient" → 「決定論的方策勾配」 (standard Japanese RL terminology)

## chapter_reinforcement-learning/deep-rl.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "state coupling" / "coupling" (across states via shared network parameters) → 「状態間の結合」 / 「結合」

## chapter_deep-reinforcement-learning/dqn.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Baird's counterexample" → 「Baird の反例」 (person name kept in Latin script per `ja/style.md` §5)
- "frozen copy" → 「凍結コピー」, "frozen target network" → 「凍結ターゲットネットワーク」
- "coupling" (source of instability in data/targets) → 「結合」
- "ring buffer" → 「リングバッファ」 (follows `ja/style.md` §3.3 dropping final ー on バッファ)

## chapter_deep-reinforcement-learning/dqn.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "experience replay buffer" → 「経験再生バッファ」 (pairs glossary `experience replay` = 「経験再生」 with `ja/style.md` §3.3 `バッファ`; standalone "replay buffer" uses `decisions.md` 「リプレイバッファ」)
- "online network" / "online weights" → 「オンラインネットワーク」 / 「オンライン重み」
- "single estimator" / "double estimator" → 「単一推定量」 / 「二重推定量」

## chapter_deep-reinforcement-learning/dqn.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Double DQN" → 「Double DQN」 (proper method name kept in Latin script per `ja/style.md` §3.2 & §5)
- "overestimation" / "overrate" → 「過大評価」 / 「過大評価する」
- "catastrophic forgetting" → 「破滅的忘却」 (standard Japanese DL/RL term)

## chapter_generative-adversarial-networks/gan.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "two-sample test" → 「2 サンプル検定」 (paired with `index.md#u01` `"two-sample comparison" → 「2 サンプル比較」`)
- "source label" → 「出所ラベル」, "realness logit" → 「真実性ロジット」 (consistent with notation table in `chapter_generative-adversarial-networks/index.md#u01`)
- "disjoint supports" → 「互いに素な台」 (consistent with `decisions.md` `"support" → 「台」` for probability distributions)
- "instance noise" → 「インスタンスノイズ」 (standard katakana in GAN literature)

## chapter_generative-adversarial-networks/gan.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "envelope theorem" → 「包絡線定理」 (standard mathematical optimization / economics term)
- "softened reverse KL divergence" → 「軟化された逆方向 KL ダイバージェンス」（スライド略記では「軟化逆方向 KL」）
- "explained variance" → 「説明分散」 (standard statistical term pairing with glossary `variance` = 「分散」)

## chapter_generative-adversarial-networks/gan.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Kantorovich–Rubinstein duality" → 「カントロヴィッチ・ルビンシュタイン双対性」 (follows `ja/style.md` §5 katakana rule for theorems named after people)
- "loader" (minibatch loader) → 「ローダ」 (follows `ja/style.md` §3.3 `-er` rule)

## chapter_generative-adversarial-networks/objectives.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "payoff" / "payoff function" / "expected payoff" → 「利得」 / 「利得関数」 / 「期待利得」
- "conditional Bayes risk" / "Bayes-risk gap" → 「条件付きベイズリスク」 / 「ベイズリスクギャップ」
- "link function" / "critic link" → 「リンク関数」 / 「クリティックのリンク」
- "Fenchel–Young variational bound" / "Fenchel–Young inequality" / "Fenchel conjugate" → 「フェンシェル・ヤング変分下界」 / 「フェンシェル・ヤングの不等式」 / 「フェンシェル共役」
- "Vincze–Le Cam divergence" → 「Vincze–Le Cam ダイバージェンス」 (keeps less-known statisticians' names in Latin script per `ja/style.md` §5)
- "squared hinge (loss)" → 「二乗ヒンジ（損失）」

## chapter_generative-adversarial-networks/objectives.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "1-Lipschitz" / "$\kappa$-Lipschitz" / "$\kappa$-Lipschitzness" → 「1 リプシッツ」 / 「$\kappa$ リプシッツ」 / 「$\kappa$ リプシッツ性」 (aligns with glossary `Lipschitz constant` = 「リプシッツ定数」 and `ja/style.md` §2.2 spacing)
- "Kantorovich–Rubinstein duality" / "Kantorovich potential" → 「カントロヴィチ・ルビンシュタイン双対性」 / 「カントロヴィチポテンシャル」
- "Earth Mover's distance" → 「アースムーバー距離」
- "weight clipping" → 「重みクリッピング」
- "two-sided / one-sided (gradient) penalty" → 「両側 / 片側（勾配）ペナルティ」
- "median heuristic" / "median-bandwidth MMD" / "multi-bandwidth MMD" → 「中央値ヒューリスティック」 / 「中央値バンド幅 MMD」 / 「マルチバンド幅 MMD」

## chapter_generative-adversarial-networks/relativistic.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "relativistic pairing objective" / "pairing divergence" → 「相対論的ペアリング目的関数」 / 「ペアリングダイバージェンス」 (consistent with `chapter_generative-adversarial-networks/index.md#u01`)
- "relativistic rank weight" → 「相対論的ランク重み」 (uses binding glossary `rank` = 「ランク」 and `weight` = 「重み」)
- "product space" / "product measure" / "product distribution" → 「積空間」 / 「積測度」 / 「積分布」 ("lifting" → 「持ち上げ」)
- "tied pair" / "mixed pair" (in pairwise ranking) → 「同点ペア」 / 「混合ペア」
- "mode coverage" → 「モード被覆」 (paired with glossary `mode collapse` = 「モード崩壊」 and `mode dropping` = 「モード欠落」)
- "ranker" → 「ランカ」 (follows `ja/style.md` §3.3 dropping final `ー` on `-er` loanwords)

## chapter_generative-adversarial-networks/relativistic.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "fixed-pairing (objective)" / "collision" (in discrete GAN landscapes) → 「固定ペアリング（目的関数）」 / 「衝突」
- "best-response (critic / objective)" → 「最適応答（クリティック / 目的関数）」
- "cycle rule" → 「閉路則」

## chapter_generative-adversarial-networks/convergence.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Dirac-GAN" → 「Dirac-GAN」 (proper model name kept in Latin script per `ja/style.md` §3.2 & §5)
- "locally asymptotically stable" / "locally exponentially stable" → 「局所漸近安定」 / 「局所指数安定」
- "linear center" (in ODE phase portraits) → 「線形中心点」
- "damped harmonic oscillator" → 「減衰調和振動子」
- "one-centered penalty" → 「1 中心ペナルティ」 (paired with `decisions.md` `"zero-centered gradient penalty" → 「ゼロ中心勾配ペナルティ」`)

## chapter_generative-adversarial-networks/convergence.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "dual Sobolev norm" → 「双対ソボレフノルム」 (follows `ja/style.md` §5 katakana rule for mathematical concepts named after people)
- "low-pass filter" / "low-pass filtering" → 「ローパスフィルタ」 / 「ローパスフィルタリング」 (drops final `ー` on `フィルタ` per `ja/style.md` §3.3)
- "mode coverage" / "covered modes" / "on-mode fraction" → 「モード被覆」 / 「被覆モード数」 / 「モード上サンプル比率」
- "critic steepness" → 「クリティックの急峻さ」
- "factorial experiment" / "factorial design" → 「要因実験」 / 「要因計画」

## chapter_generative-adversarial-networks/convergence.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "best response" (game theory) → 「最適応答」
- "one-sided limit" → 「片側極限」

## chapter_generative-adversarial-networks/dcgan.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Deep Convolutional GAN (DCGAN)" → 「Deep Convolutional GAN（DCGAN）」 (kept in Latin script as a proper architecture name per `ja/style.md` §3.2)
- "normalization-free backbone" → 「正規化なしバックボーン」 (aligned with `chapter_generative-adversarial-networks/index.md`)
- "fractional-strided convolution" → 「分数ストライド畳み込み」
- "half-life" (of EMA) → 「半減期」
- "held-out" (images / sprites) → 「ホールドアウト」
- "checkerboard artifact(s)" → 「チェッカーボードアーティファクト」
- "bilinear resampling / upsampling / downsampling" → 「双線形リサンプリング / 双線形アップサンプリング / 双線形ダウンサンプリング」 (paired with glossary `bilinear interpolation` = 「双線形補間」)
- "He fan-in initialization" → 「He のファンイン初期化」 (person name in Latin script per `ja/style.md` §5, matching `He 初期化`)
- "Fréchet Distance (FID)" / "Kernel Inception Distance (KID)" → 「フレシェ距離（Fréchet Distance、FID）」 / 「Kernel Inception Distance（KID）」

## chapter_generative-adversarial-networks/dcgan.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "experimental arm" / "arm" → 「実験群」 (paired with `ja/decisions.md` `"control arm" → 「対照群」`)
- "relativistic pairing loss" → 「相対論的ペアリング損失」 (paired with `ja/decisions.md` `"relativistic pairing objective" → 「相対論的ペアリング目的関数」`)
- "relativistic rank weight" → 「相対論的ランク重み」
- "train–holdout gap" → 「訓練・ホールドアウトギャップ」

## chapter_generative-adversarial-networks/dcgan.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Wasserstein-2 distance" → 「2 次ワッサースタイン距離」 (paired with `ja/decisions.md` `"Wasserstein-1 distance" → 「1 次ワッサースタイン距離」`)
- "non-leaking (augmentation)" / "leak" → 「リークしない（データ拡張）」 / 「リークする」
- "cross-fade" → 「クロスフェード」
- "Adaptive Discriminator Augmentation (ADA)" / "DiffAugment" → 「Adaptive Discriminator Augmentation（ADA）」 / 「DiffAugment」 (kept in Latin script per `ja/style.md` §3.2)

## chapter_generative-adversarial-networks/dcgan.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "skewness" → 「歪度」

## chapter_generative-adversarial-networks/conditional.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "unconditional" / "conditional GAN (cGAN)" → 「無条件」 / 「条件付き GAN（cGAN）」
- "feature modulation" → 「特徴変調」, "conditional batch normalization" → 「条件付きバッチ正規化」
- "projection discriminator" / "projection discriminator head" / "projection critic" → 「射影識別器」 / 「射影識別器ヘッド」 / 「射影クリティック」
- "auxiliary classifier GAN (AC-GAN)" → 「補助分類器 GAN（AC-GAN）」
- "condition alignment" / "class alignment" → 「条件アライメント」 / 「クラスアライメント」
- "sample fidelity" / "unconditional (sample) realism" → 「サンプル忠実度」 / 「無条件（のサンプル）リアリズム」
- "within-class diversity" / "intra-class diversity" → 「クラス内多様性」
- "referee classifier" / "referee" → 「レフェリー分類器」 / 「レフェリー」

## chapter_generative-adversarial-networks/conditional.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "referee classifier" / "referee" → 「レフェリー分類器」 / 「レフェリー」
- "real floor" / "real floor set" → 「実データ下限」 / 「実データ下限セット」
- "centroid-prototype baseline" / "prototype baseline" / "centroid" → 「重心プロトタイプベースライン」 / 「プロトタイプベースライン」 / 「重心」
- "truncation trick" / "truncated sampling" → 「切断トリック」 / 「切断サンプリング」

## chapter_generative-adversarial-networks/adversarial-losses.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "perception–distortion trade-off" → 「知覚・歪みトレードオフ」 (coordinate compound with `・` per `ja/style.md` §2.1)
- "conditional mean" / "conditional median" → 「条件付き平均」 / 「条件付き中央値」
- "unimodal" / "bimodal" / "multimodal" (of distributions) → 「単峰性」 / 「二峰性」 / 「多峰性」
- "Adversarial Diffusion Distillation (ADD)" / "Distribution Matching Distillation (DMD)" → 「敵対的拡散蒸留（ADD）」 / 「分布マッチング蒸留（DMD）」
- "perceptual loss" → 「知覚損失」
- "feature-matching loss" → 「特徴マッチング損失」 (follows `ja/style.md` §7.1 using `特徴` in compounds and for learned features)
- "support" (of a distribution) → 「台」 ("disjoint supports" → 「台が重ならない場合」)

## chapter_generative-adversarial-networks/adversarial-losses.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Kantorovich potential" → 「カントロビッチポテンシャル」 (follows `ja/style.md` §5 katakana rule for mathematical terms named after people)
- "Implicit Maximum Likelihood Estimation (IMLE)" → 「陰的最尤推定（IMLE）」
- "Direct Discriminative Optimization (DDO)" → 「直接識別最適化（DDO）」
- "underdetermined inverse problem" → 「劣決定な逆問題」

## chapter_generative-adversarial-networks/further-reading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "one-centered penalty" → 「1 中心ペナルティ」 (paired with `decisions.md` `"zero-centered gradient penalty" → 「ゼロ中心勾配ペナルティ」`)
- "extragradient method" → 「外挿勾配法」, "optimistic gradient descent" → 「楽観的勾配降下法」, "negative momentum" → 「負のモメンタム」
- "lazy regularization" → 「遅延正則化」
- "non-leaking (data augmentation / transformation)" → 「非リーク型データ拡張 / 非リークな変換」
- "feature modulation layer" → 「特徴変調層」

## chapter_diffusion-models/index.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "denoising diffusion probabilistic model (DDPM)" → 「ノイズ除去拡散確率モデル（DDPM）」 (combines glossary `denoising` = 「ノイズ除去」, `diffusion` = 「拡散」, `probabilistic model` = 「確率モデル」)
- "score-based generative modeling" → 「スコアベース生成モデリング」, "score" ($\nabla_{\mathbf{x}} \log p$) → 「スコア関数」 (follows `ja/style.md` §7.1)
- "stochastic interpolant" → 「確率的補間」
- "probability-flow ODE" → 「確率フロー ODE」 (inserts half-width space before `ODE` per `ja/style.md` §2.2)
- "linear path" → 「線形経路」, "variance-preserving (VP) path" → 「分散保持（VP）経路」
- "Tweedie's formula" / "Tweedie's identity" → 「トゥイーディーの公式」 / 「トゥイーディーの恒等式」 (follows `ja/style.md` §5 katakana rule for classical statistical results named after people)
- "Fréchet distance (FD)" → 「フレシェ距離（FD）」
- "variational bound" / "likelihood bound" → 「変分境界」 / 「尤度境界」
- "few-step distillation" / "few-step" → 「少数ステップ蒸留」 / 「少数ステップ」
- "token masking" / "absorbing masked diffusion" → 「トークンマスキング」 / 「吸収型マスク付き拡散」

## chapter_diffusion-models/denoising.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "running example" → 「通し例」 (standard Japanese textbook term for an example carried throughout a chapter)
- "Gaussian mixture" / "mixture of Gaussians" (distribution sense) → 「混合ガウス分布」 (glossary's proposed `混合ガウスモデル` refers to the statistical model; when referring directly to a probability distribution on $\mathbb{R}^d$, 「混合ガウス分布」 is standard in Japanese ML texts such as PRML)
- "responsibility" (posterior mixture weight $r_k(\mathbf{x})$) → 「負担率」 (established Japanese ML term from Bishop『パターン認識と機械学習』)
- "mode weights" / "strays" / "spread" (2D Gaussian-mixture benchmark metrics) → 「モード重み」 / 「外れ点」 / 「広がり」
- "Tweedie's formula" / "Vincent's theorem" / "Hyvärinen's identity" → 「Tweedie の公式」 / 「Vincent の定理」 / 「Hyvärinen の恒等式」 (retains Latin script for person names per `ja/style.md` §5 and §3.2)
- "noise predictor" → 「ノイズ予測器」 (paired with glossary `denoiser` = 「ノイズ除去器」 and `noise prediction` = 「ノイズ予測」)
- "normalizer" (of an energy-based model, $Z_{\boldsymbol\theta}$) → 「正規化定数」 (aligned with glossary `normalizing constant` = 「正規化定数」)

## chapter_diffusion-models/denoising.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "blindness (to mode weights)" → 「（モード重みに対する）不感性」 (paired with `decisions.md` `"order blindness" → 「順序への不感性」`)
- "Kramers escape time" / "Kramers' formula" → 「Kramers の脱出時間」 / 「Kramers の公式」 (retains Latin script for person name per `ja/style.md` §5 and §3.2)

## chapter_diffusion-models/denoising.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Wiener filter" → 「ウィーナーフィルタ」 (established Japanese signal-processing term, dropping final `ー` on `フィルタ` per `ja/style.md` §3.3)
- "explicit score matching" → 「明示的スコアマッチング」 (standard Japanese ML counterpart to `ノイズ除去スコアマッチング`)
- "nonlinear manifold" → 「非線形多様体」 (standard mathematical/ML terminology)
- "moment generating function" → 「積率母関数」 (standard Japanese mathematical statistics term)

## chapter_diffusion-models/gaussian-paths.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "clean point" / "clean data" → 「クリーンな点」 / 「クリーンなデータ」 (standard Japanese diffusion terminology contrasting uncorrupted $\mathbf{x}_0$ with noisy $\mathbf{x}_t$)
- "rectified flow" → 「レクティファイドフロー」 (katakana matching glossary `flow matching` = 「フローマッチング」)
- "cosine path" / "cosine schedule" → 「コサイン経路」 / 「コサインスケジュール」 (paired with glossary `Gaussian path` = 「ガウス経路」 and `cosine decay` = 「コサイン減衰」)
- "ladder" (of discrete noise levels) → 「ラダー」 (standard MCMC/diffusion katakana avoiding collision with algorithmic 「ステップ」)
- "strays" (samples far from all modes) → 「はぐれ点」 (distinguished from dataset outliers 「外れ値」)
- "clock" (the shared log-SNR time parameterization) → 「時計」 (retains the operational-clock metaphor across prose and slides without colliding with hardware clock frequency 「クロック」)
- "angular velocity" → 「角速度」, "$\mathbf{v}$-prediction" → 「$\mathbf{v}$ 予測」
- "target dictionary" / "target conversion dictionary" → 「目標の辞書」 / 「目標の変換辞書」 (paired with glossary `regression target` = 「回帰目標」)

## chapter_diffusion-models/gaussian-paths.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "score–velocity identity" → 「スコア・速度恒等式」
- "loss floor" → 「損失の下限」 (consistent with `synthetic-regression-data.md#u01`; target-specific floors rendered as 「ノイズの下限」「速度の下限」)
- "excess (loss)" (above the loss floor) → 「過剰損失」 (consistent with `linear-regression-scratch.md#u01`)
- "Tweedie's formula" → 「Tweedie の公式」 (retains person name in Latin script per `ja/style.md` §5)

## chapter_diffusion-models/integration.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "probability-flow ODE" → 「確率フロー ODE」, "reverse-time SDE" / "reverse SDE" → 「逆時間 SDE」 / 「逆 SDE」 (inserts half-width space between Japanese and Latin acronym per `ja/style.md` §2.2, overriding unspaced proposed glossary entries)
- "linear path" / "VP path" / "VE path" / "cosine path" → 「線形経路」 / 「VP 経路」 / 「VE 経路」 / 「コサイン経路」
- "path-agnostic" → 「経路に依存しない」（スライド等の簡潔な表現では「経路非依存の」）
- "ancestral chain" / "ancestral step" → 「祖先連鎖」 / 「祖先ステップ」 (paired with glossary `ancestral sampling` = 「祖先サンプリング」)
- "Langevin chain" / "Langevin step" / "annealed ladder" → 「ランジュバン連鎖」 / 「ランジュバンステップ」 / 「アニーリングラダー」
- "Picard–Lindelöf uniqueness" → 「ピカール・リンデレーフの一意性」 (follows `ja/style.md` §5 katakana rule for theorems named after people)

## chapter_diffusion-models/integration.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Euler's method" / "Euler step" → 「オイラー法」 / 「オイラーステップ」
- "stray(s)" / "stray fraction" → 「逸脱点」 / 「逸脱点の割合」（スライドでは「逸脱率」）
- "within-mode spread" → 「モード内の広がり」

## chapter_diffusion-models/integration.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "denoising score model" / "noise-conditional score network" → 「ノイズ除去スコアモデル」 / 「ノイズ条件付きスコアネットワーク」
- "rectified flow" → 「Rectified Flow」 (proper method name kept in Latin script per `ja/style.md` §3.2)
- "$\mathbf{v}$-prediction" → 「$\mathbf{v}$ 予測」 (with half-width space per `ja/style.md` §2.2)
- "latent interpolation" → 「潜在空間での補間」（スライドでは「潜在補間」）

## chapter_diffusion-models/integration.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "commitment time" / "speciation time" → 「コミットメント時刻」 / 「種分化時刻」 (dynamical regimes in diffusion models per Biroli et al., 2024)
- "bimodal" → 「二峰性」

## chapter_diffusion-models/image-diffusion.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "ancestral chain" / "ancestral sampler" → 「祖先連鎖」 / 「祖先サンプラ」 (matches `マルコフ連鎖` in `ja/style.md` §5 and `sampler` = 「サンプラ」 in `ja/decisions.md`)
- "multi-resolution" → 「多重解像度」
- "label dropout" → 「ラベルドロップアウト」, "null label" → 「ヌルラベル」
- "typicality" → 「典型性」 (paired with `diversity` = 「多様性」 in CFG trade-off discussions)
- "reference floor" / "reference value" (of FD) → 「基準下限」 / 「基準値」

## chapter_diffusion-models/image-diffusion.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "half-life" (of EMA) → 「半減期」, "decay" (EMA factor) → 「減衰率」
- "Euler step(s)" → 「オイラーステップ」

## chapter_diffusion-models/image-diffusion.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "class agreement" → 「クラス一致率」
- "dictionary" (of equivalent diffusion targets) → 「変換辞書」

## chapter_diffusion-models/image-diffusion.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "guidance interval" → 「ガイダンス区間」, "autoguidance" → 「オートガイダンス」
- "tilted density / tilted Gaussian" → 「傾斜密度 / 傾斜ガウス分布」 (aligned with `ja/decisions.md` `"tilted distribution" → 「傾斜分布」`)

## chapter_diffusion-models/image-diffusion.md#u05 (gemini gemini-4-carbon (medium), 2026-11-27)

- "static thresholding" / "dynamic thresholding" → 「静的しきい値処理」 / 「動的しきい値処理」
- "truncation trick" (BigGAN) → 「打ち切りトリック」 (aligned with `ja/decisions.md` `"truncated sampling" → 「打ち切りサンプリング」`)

## chapter_diffusion-models/samplers.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "denoising diffusion implicit model (DDIM)" → 「ノイズ除去拡散陰的モデル（DDIM）」 (aligns with glossary `denoiser` = 「ノイズ除去器」 and `decisions.md` `implicit` = 「陰的」; subsequent uses `DDIM`)
- "ancestral chain" / "ancestral step" → 「祖先連鎖」 / 「祖先ステップ」 (paired with glossary `ancestral sampling` = 「祖先サンプリング」)
- "ODE solver" / "solver" → 「ODE ソルバ」 / 「ソルバ」 (follows `ja/style.md` §2.2 spacing and §3.3 dropping final `ー` on `-er` loanwords, consistent with `decisions.md` `固有値ソルバ / 線形ソルバ`)
- "multistep (method / solver / sampler)" → 「多段（法 / ソルバ / サンプラ）」 (standard Japanese numerical ODE terminology, distinguished from order 「1 次 / 2 次」)
- "Heun's method" / "Adams–Bashforth" → 「ホイン法」 / 「アダムス・バッシュフォース法」 (standard Japanese numerical analysis terms per `ja/style.md` §5)
- "inversion" (image-to-noise ODE inversion) → 「逆変換」 (unambiguously distinguishes generative mapping inversion from spatial/color flipping 「反転」)
- "partial noising" / "inpainting" → 「部分ノイズ付加」 / 「インペインティング」
- "fresh noise" / "kept noise" → 「新たなノイズ」 / 「保持ノイズ」

## chapter_diffusion-models/samplers.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "exponential integrator" → 「指数積分器」 (matches `decisions.md` `"double integrator" → 「二重積分器」`)
- "spread" (mean squared distance to nearest mode center) → 「広がり」 (consistent with `decisions.md` `"token spread" → 「トークンの広がり」`)
- "predictor–corrector" → 「予測子・修正子」 (standard Japanese numerical ODE term)
- "network function evaluations (NFE)" → 「ネットワーク関数評価回数（NFE）」 (subsequent uses `NFE` or `ネットワーク評価回数`)

## chapter_diffusion-models/samplers.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "null-text inversion" → 「ヌルテキスト逆変換」 (paired with `null label` = 「ヌルラベル」 and `inversion` = 「逆変換」)
- "spherical (linear) interpolation (Slerp)" → 「球面（線形）補間（Slerp）」 (distinguishes mathematical interpolation 「補間」 from image completion 「補完」)
- "replacement inpainting" → 「置換インペインティング」

## chapter_diffusion-models/samplers.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "harmonization" (in RePaint inpainting) → 「調和化」
- "commitment time" → 「コミットメント時刻」

## chapter_diffusion-models/design-space.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "likelihood bound" / "variational bound" → 「尤度境界」 / 「変分境界」 (consistent with `chapter_diffusion-models/index.md`)
- "forward chain" / "reverse chain" / "ancestral chain" → 「順方向連鎖」 / 「逆方向連鎖」 / 「祖先連鎖」
- "stray" / "stray samples" (samples far from all mixture modes) → 「逸脱点」 / 「逸脱サンプル」
- "Runge–Kutta method" / "Heun's method" → 「ルンゲ・クッタ法」 / 「ホイン法」

## chapter_diffusion-models/design-space.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "cosine path" → 「コサイン経路」
- "$\mathbf{v}$-prediction" / "$\mathbf{v}$-target" / "$\mathbf{v}$-loss" → 「$\mathbf{v}$ 予測」 / 「$\mathbf{v}$ ターゲット」 / 「$\mathbf{v}$ 損失」
- "logit-normal" → 「ロジット正規分布」 / 「ロジット正規」

## chapter_diffusion-models/design-space.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "preconditioned clean image" → 「前処理付きクリーン画像」
- "ODE solver" → 「ODE ソルバ」 (follows `ja/style.md` §2.2 spacing and §3.3 `-er` rule)
- "low-pass filter" → 「ローパスフィルタ」 (follows `ja/style.md` §3.3 dropping final `ー` on `フィルタ`)
- "Hutchinson trace estimator" → 「Hutchinson トレース推定量」 (person name kept in Latin script per `ja/style.md` §5)

## chapter_diffusion-models/design-space.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Rademacher vector" → 「ラデマッハベクトル」 (paired with `decisions.md` `"Rademacher complexity" → 「ラデマッハ複雑度」`)
- "bits per dimension" → 「次元あたりビット数」 (paired with `decisions.md` `"bits per byte" → 「バイトあたりビット数」`)

## chapter_diffusion-models/latent-diffusion.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "latent" / "latents" (noun) → 「潜在表現」, "latent number" / "latent value" → 「潜在値」 (pairs with glossary `latent space` = 「潜在空間」, `latent diffusion` = 「潜在拡散」, `latent code` = 「潜在コード」)
- "diffusion transformer (DiT)" → 「拡散 Transformer（DiT）」 (adds half-width space to proposed glossary `拡散Transformer` per `ja/style.md` §2.2 & §3.2)
- "rate–distortion theory" / "rate–distortion function" → 「レート・歪み理論」 / 「レート・歪み関数」; "rate", "distortion", "modelability" → 「レート」「歪み」「モデル化のしやすさ」
- "Fréchet distance (FD)" → 「フレシェ距離（Fréchet distance、FD）」 (follows `ja/style.md` §5 katakana rule for mathematical distances named after people, parallel to 「ワッサースタイン距離」)
- "excess kurtosis" → 「超過尖度」
- "masked diffusion" → 「マスク付き拡散」
- "floor" (baseline FD between two finite real-image splits) → 「フロア」

## chapter_diffusion-models/latent-diffusion.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "decoder tolerance" / "perturbation tolerance" / "intolerance" → 「デコーダの耐性」 / 「摂動耐性」 / 「耐性の低さ（耐性不足）」 (consistent with `decisions.md` `"shift tolerance" → 「シフト耐性」`)
- "representation autoencoder" → 「表現オートエンコーダ」
- "unpatchify" → 「アンパッチ化」 (paired with `decisions.md` `"patchify" → 「パッチ化」`)

## chapter_diffusion-models/latent-diffusion.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "multimodal DiT (MM-DiT)" → 「マルチモーダル DiT（MM-DiT）」
- "null label" (in classifier-free guidance) → 「ヌルラベル」
- "probability-flow ODE" → 「確率フロー ODE」 (adds half-width space to proposed glossary `確率フローODE` per `ja/style.md` §2.2)

## chapter_diffusion-models/latent-diffusion.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "hourglass transformer" → 「砂時計型 Transformer」
- "patchification" → 「パッチ化」 (consistent with `decisions.md` `"patchify" → 「パッチ化」`)

## chapter_diffusion-models/few-step.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "few-step generation" / "few-step" → 「少数ステップ生成」 / 「少数ステップ」
- "probability-flow ODE" → 「確率フロー ODE」（`ja/style.md` §2.2 の半角スペース規則を適用）
- "teacher" / "student" (in distillation) → 「教師」 / 「生徒」
- "Euler step" / "Euler's method" → 「オイラーステップ」 / 「オイラー法」（`ja/style.md` §5 の人名由来用語のカタカナ規則に従う）
- "compositional / Lagrangian / Eulerian identity" → 「合成 / ラグランジュ型 / オイラー型恒等式」（流体力学のラグランジュ・オイラー記述に対応し、最適化の「ラグランジアン」や代数学の恒等式と区別）
- "flow-map distillation" → 「フロー写像蒸留」（glossary `flow map` = 「フロー写像」および `consistency distillation` = 「一貫性蒸留」に対応）
- "MeanFlow" / "MeanFlow identity" → 「MeanFlow」 / 「MeanFlow 恒等式」（固有名は `ja/style.md` §3.2 に従いラテン文字維持）
- "re-noise" / "re-noising" → 「再ノイズ化する」 / 「再ノイズ化」
- "stray(s)" (samples >3 SD from all mode centers) → 「逸脱サンプル」（割合は「逸脱率」、動詞は「逸脱する」）
- "spread" (within-mode mean squared distance) → 「広がり」（「モード内の広がり」）

## chapter_diffusion-models/few-step.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "Fréchet distance (FD)" → 「フレシェ距離（Fréchet distance、FD）」（以降は「FD」；`ja/style.md` §3.1・§5 に従う）
- "ODE solver" → 「ODE ソルバ」（`ja/style.md` §2.2 の半角スペース規則および §3.3 の `-er` 長音省略規則に従う）
- "distribution matching distillation (DMD)" → 「分布マッチング蒸留（DMD）」
- "adversarial distillation (ADD)" → 「敵対的蒸留（ADD）」

## chapter_diffusion-models/few-step.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "distribution matching" → 「分布マッチング」（`flow matching` = 「フローマッチング」と対応）
- "distribution matching distillation (DMD)" → 「分布マッチング蒸留（DMD）」
- "adversarial distillation" → 「敵対的蒸留」

## chapter_diffusion-models/discrete-diffusion.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "discrete diffusion" / "discrete diffusion model" → 「離散拡散」 / 「離散拡散モデル」
- "masked diffusion" / "masked diffusion model" / "diffusion language model" → 「マスク拡散」 / 「マスク拡散モデル」 / 「拡散言語モデル」 (matches glossary `masked language model` = 「マスク言語モデル」)
- "absorbing (masking) kernel" / "absorbing chain" → 「吸収（マスキング）カーネル」 / 「吸収連鎖」 ("uniform kernel" → 「一様カーネル」)
- "transition matrix" → 「遷移行列」
- "unmask" / "unmasking" / "reveal" (of masked tokens) → 「マスク解除する（マスクを解く）」 / 「マスク解除」
- "independence error" → 「独立性誤差」
- "variational bound" / "masked (diffusion) bound" (on negative log-likelihood $L_{\textrm{VLB}}$) → 「変分上界」 / 「マスク拡散の変分上界」 (since $-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0) \leq L_{\textrm{VLB}}$ is an upper bound on NLL, paired with "floor" = 「下限」)
- "all-mask sequence / state / image" → 「全マスク系列 / 全マスク状態 / 全マスク画像」
- "time-agnostic" → 「時刻非依存（時刻に依存しない）」
- "continuous-time Markov chain (CTMC)" → 「連続時間マルコフ連鎖（CTMC）」

## chapter_diffusion-models/discrete-diffusion.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27)

- "confidence remasking" / "confidence-ordered" → 「確信度再マスキング」 / 「確信度順の」 (aligns with `decisions.md` `"confidence" → 「確信度」`)
- "phantom blob" / "phantom mode" → 「偽のブロブ」 / 「偽のモード」

## chapter_diffusion-models/discrete-diffusion.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27)

- "speckle(s)" (isolated interior pixel differing from all four neighbors) → 「スペックル」
- "raster order" → 「ラスタ順」 (follows `ja/style.md` §3.3 `-er` rule)
- "salt-and-pepper noise" → 「ごま塩ノイズ」 (standard Japanese image-processing term)

## chapter_diffusion-models/discrete-diffusion.md#u04 (gemini gemini-4-carbon (medium), 2026-11-27)

- "order-agnostic autoregressive model" → 「順序非依存な自己回帰モデル」 ("order-agnostic" → 「順序非依存」)
- "parallel decoding" → 「並列デコーディング」 (paired with glossary `autoregressive decoding` = 「自己回帰デコーディング」)
- "block diffusion" → 「ブロック拡散」
- "infilling" → 「穴埋め生成」

## chapter_diffusion-models/discrete-diffusion.md#u05 (gemini gemini-4-carbon (medium), 2026-11-27)

- "fused attention" → 「融合アテンション」 (paired with `decisions.md` `"fused kernel" → 「融合カーネル」`)

## chapter_diffusion-models/further-reading.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "ODE solver" → 「ODE ソルバ」 (inserts half-width space per `ja/style.md` §2.2 and drops final `ー` per §3.3, overriding proposed `ODEソルバー`)
- "diffusion transformer" → 「拡散 Transformer」 (inserts half-width space per `ja/style.md` §2.2, overriding proposed `拡散Transformer`)
- "rectified flow" → 「レクティファイドフロー」
- "persistent contrastive divergence" → 「持続的コントラスティブダイバージェンス」, "noise-contrastive estimation" → 「ノイズ対照推定」
- "empirical Bayes" → 「経験ベイズ」
- "DDIM inversion" / "null-text inversion" → 「DDIM 反転」 / 「ヌルテキスト反転」
- "progressive distillation" → 「漸進的蒸留」
- "remasking" / "unmasking" → 「再マスキング」 / 「アンマスキング」

## chapter_generative-adversarial-networks/convergence-lib.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27)

- "launch plumbing" → 「起動処理の内部機構」 (consistent with `chapter_appendix-tools-for-deep-learning/utils.md#u01` `"Internal Plumbing" → 「内部機構」`)
- "(build-only, unlisted)" → 「（ビルド専用、非掲載）」 (consistent with `chapter_appendix-tools-for-deep-learning/hyperparameters-lib.md#u01`)
