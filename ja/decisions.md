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
