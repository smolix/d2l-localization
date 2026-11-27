<!--en:e436e849-->
# 情報理論とダイバージェンス

<!--en:6ef8edf1-->
本書のほとんどのモデルは、交差エントロピー損失（cross-entropy loss）を最小化することで訓練される。これは、観測された結果に高い確率を割り当てるという直観的な目標を、勾配降下法のための滑らかな目的関数へと変換するものである。しかし、その損失の数値は実際には何を測っているのだろうか。言語モデルが文字あたり 1.2 nat を示したとき、その値は原理的にどこまで下がり得るのか。また、交差エントロピーを変分オートエンコーダ（variational autoencoder）の KL ペナルティや敵対的生成ネットワーク（generative adversarial network）のクリティックに置き換えるとき、何を比較しているのだろうか。

<!--en:666cb776-->
Claude Shannon は 1948 年、ノイズのある回線を通してどれだけ効率よくメッセージを送れるかという問いに取り組むなかで、これらの疑問に答えた。予測不可能性の尺度である*エントロピー*は、いかなるモデルの交差エントロピーも下回ることのできない厳密な下限を定める（:numref:`subsec_info_theory_basics`）。ここから導かれる 3 つの帰結が、深層学習全体を貫いている。第 1 に、交差エントロピーはデータのエントロピーを、データからモデルへの *KL ダイバージェンス*の分だけ上回り、この差は両者が一致するときに限り 0 になる。第 2 に、ビット単位で表したモデルの総損失は、高々 2 ビットのオーバーヘッドを除けば文字どおり圧縮ファイルサイズそのものであり、これによって言語モデルのパープレキシティ（:numref:`subsec_perplexity`）に直接的な操作的意味が与えられる。第 3 に、学習されたクリティックはダイバージェンスや共有情報量の下界しか与えない。対照学習（contrastive learning）において $N$ 個の候補から対応する相手 1 つを識別できても保証されるのは高々 $\log N$ nat であり、弱い敵対的クリティック（:numref:`sec_gan_objectives`）のもとでは遠く離れた分布どうしが近く見えてしまうことがある。

<!--en:75196a36-->
本章は 3 部構成をとる。第 1 部では、エントロピーと交差エントロピーを展開し（:numref:`sec_mdl-information_theory`）、対数損失（log-loss）を可逆圧縮（lossless compression）およびパープレキシティへと直接結びつける（:numref:`sec_mdl-coding-compression`）。第 2 部では、密度比ダイバージェンス（density-ratio divergence）の決定的な破綻を取り上げる（:numref:`sec_mdl-divergences-distances`）。モデルのサンプルがデータと重ならないときには必ず KL が発散し、有界な $f$ ダイバージェンスは飽和して勾配が 0 になってしまう。滑らかなテスト関数の期待値や最適輸送（optimal transport）のコストを比較すれば（:numref:`sec_mdl-test-functions-transport`）有用な勾配を回復でき、一方で対数密度（log density）の傾きを一致させれば（:numref:`sec_mdl-scores`）、計算困難な分配関数（partition function）を扱うことなく非正規化モデルを比較できる。第 3 部では、相互情報量（mutual information）、データ処理、ファノの不等式（Fano's inequality）、情報ボトルネック（information bottleneck）を通じて確率変数の対を調べ（:numref:`sec_mdl-mutual-information`）、最終的にサンプルに基づく推定と InfoNCE へと至る（:numref:`sec_mdl-mi-estimation`）。なお、最後の 2 節は :numref:`sec_mdl-information_theory` の後だけで自己完結しているため、輸送とスコアの話題を :numref:`chap_gans` および :numref:`chap_diffusion` まで取っておきたい場合は、そのまま先に読み進めてもよい。

<!--en:6168ab04-->
本章では、イェンセンの不等式（:numref:`chap_mdl-optimization`）、条件付き分布、および最尤法（:numref:`chap_mdl-probability-statistics`）を既知とする。誤り訂正符号（error-correcting code）、コルモゴロフ複雑性（Kolmogorov complexity）、情報幾何学（information geometry）については、:numref:`sec_mdl-information-theory-further-reading` で概観する。

<!--en:bb63df72-->
情報理論とダイバージェンス<br>**エントロピー・交差エントロピー・情報源符号化・f ダイバージェンス・最適輸送・スコア・相互情報量**

<!--en:a1b635eb-->
::: {.slide title="なぜ交差エントロピーでモデルを訓練するのか"}

<!--en:d19f72f6-->
[第 §{num:chap_mdl-information-theory} 章の中心的な問い]{.kicker}

<!--en:4220ee9e-->
- **損失の数値は何を測るか：**あらゆる交差エントロピー損失は、削減不能なデータの下限と非負のダイバージェンスのギャップに分解される（$\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P \| Q) \ge H(P)$）

<!--en:f8366967-->
- **なぜ対数損失が文字どおりファイルサイズなのか：**算術符号化は $n$ 個の記号を $q(x_i \mid x_{<i})$ のもとで $n\,\textrm{CE}_2(P, Q) + 2$ ビットに圧縮する

<!--en:0ba6ec15-->
- **KL が発散するときはどうするか：**最適輸送（$W_1$）、カーネル MMD、スコアマッチングは、サンプルまたは傾きだけから分布を比較する

<!--en:7eb76565-->
::: {.slide title="すべての実務家が押さえるべき 3 つの基本定理"}

<!--en:787799c2-->
[中核となる数学的支柱]{.kicker}

<!--en:5e149f89-->
- **ギブスの不等式（§{sec_mdl-information_theory}）：**$D_{\textrm{KL}}(P \| Q) \ge 0$（$= 0 \iff P = Q$）であるため、いかなる分類器の損失も条件付きラベルエントロピー $H(Y \mid X)$ を下回れない

<!--en:5c3a99c4-->
- **情報源符号化（§{sec_mdl-coding-compression}）：**クラフトの不等式（$\sum_i 2^{-\ell_i} \le 1$）と算術符号化により、任意の因果的 LLM が可逆圧縮器になる

<!--en:efa94483-->
- **変分クリティック（§{sec_mdl-divergences-distances}〜§{num:sec_mdl-mi-estimation}）：**$f$-GAN と InfoNCE はサンプルからダイバージェンスと相互情報量の下界を与え、InfoNCE の下界は $\log N$ nat（バッチサイズ $N$）で頭打ちになる

<!--en:6045bd7e-->
::: {.slide title="第 §{num:chap_mdl-information-theory} 章のロードマップ：連携する 7 つの節"}

<!--en:00be04d1-->
[章の構成]{.kicker}

<!--en:466fc526-->
- **トラック 1：対数損失・符号化・パープレキシティ（§{sec_mdl-information_theory}〜§{num:sec_mdl-coding-compression}）：**サプライザル、シャノンエントロピー、ギブスの下限、ラベル平滑化、蒸留、クラフトの不等式、算術符号化、MDL、AEP、パープレキシティ

<!--en:c2ba23fe-->
- **トラック 2：ダイバージェンス・輸送・スコア（§{sec_mdl-divergences-distances}〜§{num:sec_mdl-scores}）：**$f$ ダイバージェンス、ピンスカーの不等式、$f$-GAN、IPM、カーネル MMD、ワッサーシュタイン $W_1$、シンクホーン OT、スタインの恒等式、スコアマッチング、KSD

<!--en:69a456db-->
- **トラック 3：従属性と対照下界（§{sec_mdl-mutual-information}〜§{num:sec_mdl-mi-estimation}）：**相互情報量、DPI、ファノの不等式、情報ボトルネック、KSG 近傍法、McAllester–Stratos の上限、InfoNCE

<!--en:c98d44c1-->
::: {.slide title="利用可能な情報に応じた分布比較手法の選択"}

<!--en:34675d55-->
[実践的な選択ガイド]{.kicker}

<!--en:03a980b7-->
- **正規化済み密度 $q_{\boldsymbol{\theta}}(\mathbf{x}) > 0$：**最尤法により**順方向 KL** $D_{\textrm{KL}}(P \| Q_{\boldsymbol{\theta}})$ を最小化する（モード被覆型）

<!--en:98fc89e4-->
- **非正規化ターゲット $\tilde{p}(\mathbf{z}, \mathbf{x})$ と計算可能な $q_{\boldsymbol{\theta}}$：**変分推論・ELBO により**逆方向 KL** $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \| P)$ を最小化する（モード探索型）

<!--en:c8340f17-->
- **暗黙的サンプル生成器または EBM：**サンプルのみの生成器には**カーネル MMD**・**ワッサーシュタイン $W_1$** を用い、スコア $-\nabla_{\mathbf{x}}E_{\boldsymbol{\theta}}$ のみが計算可能な場合は**フィッシャーダイバージェンス**・**KSD** を用いる

<!--en:0abd062d-->
::: {.slide title="通しベンチマークと学習パス"}

<!--en:2aa2f8db-->
[本章の読み進め方]{.kicker}

<!--en:8e50e6b0-->
本章のすべての理論的限界は、具体的かつ再現可能なベンチマークで検証する。

<!--en:7076359f-->
- **離散ベンチマーク：**2 進情報源 $(\tfrac12, \tfrac14, \tfrac18, \tfrac18)$、非対称ペア $P = (0.6, 0.3, 0.1)$ と $Q = (0.2, 0.5, 0.3)$、ビット反転（$\delta = 0.1$）、および『The Time Machine』の文字レベル圧縮

<!--en:3fb67706-->
- **連続ベンチマーク：**平均シフトおよび分散スケーリングを施したガウス分布 $\mathcal{N}(0, 1)$ 対 $\mathcal{N}(d, \sigma^2)$、二峰性混合ガウス分布（$70/30$ の比率）、および相関のある 2 変量ガウス分布の組（$\rho = 0.9$ と $\rho = 0.99$）

<!--en:832e47e3-->
- **モジュール化された 3 つのトラック：**
  1. **対数損失と言語モデル（§{sec_mdl-information_theory}〜§{num:sec_mdl-coding-compression}）：**教師あり学習とパープレキシティの基礎として最初から順に通読する
  2. **生成モデリングにおける比較（§{sec_mdl-divergences-distances}〜§{num:sec_mdl-scores}）：**GAN、フローマッチング、拡散モデルに不可欠な基礎知識
  3. **従属性と対照学習（§{sec_mdl-mutual-information}〜§{num:sec_mdl-mi-estimation}）：**§{sec_mdl-information_theory} の後だけで自己完結しており、VAE、SimCLR、CLIP に向けて直接読み進められる
