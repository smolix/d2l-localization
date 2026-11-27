<!--en:42325983-->
# 確率と統計学習

<!--en:cb182177-->
本編全体を通して、回帰モデルは二乗誤差（squared error）で、分類器は交差エントロピー（cross-entropy）で訓練し（:numref:`sec_linear_regression`、:numref:`sec_softmax`）、有限のテストセット上で正解率を報告し、検証スコアを比較してモデルを選択してきた（:numref:`sec_generalization_basics`）。これらのステップはいずれも、有限の標本（sample）から推論（inference）を引き出している。コイン投げで表が $9$ 回（$13$ 回中）出た場合を考えてみよう。経験的な表の割合は $9/13 \approx 0.69$ であるが、公正なコインであっても 8 回に 1 回を超える頻度で少なくとも 9 回は表が出る。テスト正解率とは、まさにその同じコインをテスト例ごとに 1 回投げたものにほかならない。本章では、有限の標本がいつ信頼できる結論を支え、いつ判断を誤らせるのかを説明する数学を展開する。

<!--en:529dfcd4-->
標本に伴う不確実性を整理するには、4 つの異なる概念が土台となる。結果がどのようにばらつくかを記述する*分布*、標本を数値へ写す*推定量*（estimator）、仮想的な標本間にわたる推定量の広がりとは異なる、手元の単一の標本で条件付けた*事後分布*（posterior）、そして有限の標本における小さな誤差が新たなデータにおける小さな誤差をいつ保証するかを裏づける*汎化限界*（generalization bound）である。

<!--en:1003eeff-->
本章では、これら 4 つの概念を順に取り上げる。まず、連続密度（:numref:`sec_mdl-random_variables`）、同時分布とヤコビ行列（Jacobian）による変換（:numref:`sec_mdl-joint-distributions`）、代表的な名前付き分布（:numref:`sec_mdl-distributions`）、そしてそれらを統一する指数分布族（exponential family、:numref:`sec_mdl-exponential-family`）を確立する。これにより、線形回帰、ロジスティック回帰、ソフトマックス回帰がいずれも「予測マイナス観測」という同じ形の勾配を共有する理由が明らかになる。次に、:numref:`sec_mdl-statistics` ではエラーバー、仮説検定、信頼区間、ブートストラップ法を展開し、:numref:`sec_mdl-maximum_likelihood` では最尤法（maximum likelihood）から訓練損失を導出してパラメータの分散をフィッシャー曲率へと結びつける。続いて、隠れ変数を動機としてエビデンス下界（evidence lower bound）と期待値最大化法（expectation–maximization）を導入し（:numref:`sec_mdl-latent-em-elbo`）、事後分布が閉じた形を持たない場合に備えてラプラス近似、モンテカルロ法、マルコフ連鎖モンテカルロ法、変分近似を扱う（:numref:`sec_mdl-bayesian-computation`）。最後に、:numref:`sec_mdl-concentration-generalization` で有限標本における裾の上界と高次元における球殻への集中を証明し、:numref:`sec_mdl-generalization-bounds` でそれらを選択後の一様収束（uniform convergence）へと拡張し（:numref:`chap_classification_generalization`）、:numref:`sec_mdl-naive_bayes` で分布、平滑化、較正（calibration）を画像分類へと応用する。

<!--en:97b2c763-->
:numref:`sec_prob` の内容に馴染みのある読者は冒頭のモーメントの代数計算を流し読みしてよく、また :numref:`sec_mdl-bayesian-computation` と :numref:`sec_mdl-naive_bayes` は最初の通読時には読み飛ばしても後続の節の理解には支障がない。エントロピーとダイバージェンスの本格的な理論は :numref:`chap_mdl-information-theory` に引き継がれ、測度論（measure theory）、時系列（time series）、およびノンパラメトリックな拡張については :numref:`sec_mdl-probability-statistics-further-reading` で概観する。

<!--en:cc62f985-->
確率と統計学習：不確実性下の推論の基礎<br>**密度とヤコビ行列・指数分布族・推定と検定・MLE と ELBO・ベイズ計算・集中と汎化**

<!--en:08c18ba1-->
::: {.slide title="なぜ深層学習に統計的基礎が必要か"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:8d7c7f89-->
- **通し例のコイン（表 $9$ 回・全 $13$ 回）：**$\hat{\theta} = 9/13 \approx 0.692$ だが、公正なコイン（$\theta = 0.5$）でも表が $\ge 9$ 回出る確率は $13.3\%$ あり、有限のホールドアウトセット上のテスト正解率もこれと同じ構造を持つ

<!--en:94bc2259-->
- **順方向と逆方向の推論：**既知のモデルから結果の頻度を導くのが*確率*（§{sec_mdl-random_variables}〜§{num:sec_mdl-exponential-family}）であり、有限のデータから推定値、エラーバー、事後分布を逆向きに導くのが*統計*（§{sec_mdl-statistics}〜§{num:sec_mdl-bayesian-computation}）

<!--en:f4b9d48f-->
- **選択後の保証（§{sec_mdl-concentration-generalization}〜§{num:sec_mdl-generalization-bounds}）：**複数の検証実験から最良のモデルを選ぶ操作は形を変えた多重検定であり、一様収束による上界が必要になる

<!--en:2246f51c-->
::: {.slide title="確率的学習の 4 つの柱"}

<!--en:00be04d1-->
[章の構成]{.kicker}

<!--en:499c9244-->
- **1. 分布と統一形式（§{sec_mdl-random_variables}〜§{num:sec_mdl-exponential-family}）：**密度、ヤコビ行列、代表的な分布、指数分布族 $h(\mathbf{x})e^{\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})}$

<!--en:ab9f460e-->
- **2. 推定、検定、尤度（§{sec_mdl-statistics}〜§{num:sec_mdl-maximum_likelihood}）：**バイアス・分散のトレードオフ、$p$ 値、信頼区間、KL 射影としての MLE、フィッシャー情報量

<!--en:f71e27f4-->
- **3. 潜在モデルと事後分布（§{sec_mdl-latent-em-elbo}〜§{num:sec_mdl-bayesian-computation}）：**周辺尤度 $\log p(\mathbf{x})$、イェンセンの不等式、ELBO、EM、ラプラス近似、VI、MCMC

<!--en:374da574-->
- **4. 集中と汎化（§{sec_mdl-concentration-generalization}〜§{num:sec_mdl-naive_bayes}）：**チェルノフ・ヘフディングの上界、高次元 $d$ における球殻の幾何構造、ラデマッハ複雑度、ナイーブベイズ

<!--en:4311b0a9-->
::: {.slide title="ロードマップ：分布から事後分布まで（§{sec_mdl-random_variables}〜§{num:sec_mdl-bayesian-computation}）"}

<!--en:d623abd2-->
[各節ガイド（パート I）]{.kicker}

<!--en:e65eaf49-->
- **§{sec_mdl-random_variables}〜§{num:sec_mdl-joint-distributions}（連続変数の微分積分）：**$F'(x) = p(x)$、逆変換サンプリング $X = F^{-1}(U)$、イブの法則、ヤコビ行列による体積の伸縮 $|\det J_g|^{-1}$

<!--en:3b5ddc86-->
- **§{sec_mdl-distributions}〜§{num:sec_mdl-exponential-family}（分布一覧と指数分布族）：**二項分布からポアソン分布への極限と CLT 極限、シューア補行列によるガウス条件付け、キュムラント導関数 $\nabla A = \mathbb{E}[T]$、共役更新

<!--en:baaec0ef-->
- **§{sec_mdl-statistics}〜§{num:sec_mdl-bayesian-computation}（推論と潜在変数）：**$\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$、対応のある検定、ブートストラップ法、フィッシャー曲率 $nI(\theta)$、ELBO による EM、MCMC・VI による事後分布

<!--en:a3fcca6b-->
::: {.slide title="ロードマップ：理論保証と統合（§{sec_mdl-concentration-generalization}〜§{num:sec_mdl-naive_bayes}）"}

<!--en:1bfcc6b1-->
[各節ガイド（パート II）]{.kicker}

<!--en:f75f0a61-->
- **§{sec_mdl-concentration-generalization} 集中不等式：**チェルノフ・ヘフディングによりチェビシェフの $1/(n\varepsilon^2)$ 上界を指数減衰 $\exp(-2n\varepsilon^2)$ へと強化し、高次元 $d$ のガウス分布が半径 $\sqrt{d}$ の薄い球殻に集中する理由を証明する

<!--en:0a4c09c1-->
- **§{sec_mdl-generalization-bounds} 汎化限界：**ユニオンバウンド、ラデマッハ複雑度、良性過学習を通じ、モデル探索全体にわたる最悪ケースの汎化誤差 $\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)|$ を上から抑える

<!--en:72e02ec4-->
- **§{sec_mdl-naive_bayes} ナイーブベイズ分類器：**カテゴリ・ベルヌーイ尤度とラプラスの 1 加算平滑化を組み合わせ、エンドツーエンドの生成型画像分類器を構築する

<!--en:c8eafc5f-->
::: {.slide title="第 §{num:chap_mdl-probability-statistics} 章の読み進め方"}

<!--en:123f19cf-->
[学習ガイド]{.kicker}

<!--en:988773de-->
- **中核となる理論的骨格：**連続密度と指数分布族が本編のすべての教師あり損失関数と統計的エラーバーをいかに統一するかを理解するため、**§{sec_mdl-random_variables}〜§{num:sec_mdl-maximum_likelihood}** を順に読む

<!--en:6ec8a3b2-->
- **速習ルート：**§{sec_prob}（準備章の離散確率）にすでに慣れている場合は、§{sec_mdl-random_variables}〜§{num:sec_mdl-joint-distributions} の基本的なモーメントの代数計算は流し読みし、逆変換サンプリング、イブの法則、ヤコビ行列式、ガウス分布の条件付けに集中する

<!--en:9024af89-->
- **モジュール化された発展トピック：**最初の通読時には、自己完結している **§{sec_mdl-bayesian-computation}（ベイズ計算）**と **§{sec_mdl-naive_bayes}（ナイーブベイズ）**を読み飛ばしても後続の節の理解には支障がない

<!--en:d64cd66b-->
- **他の数学章とのつながり：**線形代数のスペクトルと正定値性は**第 §{num:chap_mdl-linear-algebra} 章**、多変数のヤコビ行列と凸性は**第 §{num:chap_mdl-calculus}〜§{num:chap_mdl-optimization} 章**に基づき、エントロピー・KL の本格的な理論は**第 §{num:chap_mdl-information-theory} 章**へと続く

<!--en:c9e005f9-->
**次の節（§{sec_mdl-random_variables} 確率変数）：**なぜ連続的な結果では単一点の確率が 0 になるのか、密度と CDF が $F'(x) = p(x)$ を通じてどう結びつくのか、そして重い裾がなぜ標本平均を破綻させるのか
