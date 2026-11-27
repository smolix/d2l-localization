<!--en:9cff79a7-->
# 微分積分と自動微分

<!--en:dd6404e4-->
本書全体を通して、ニューラルネットワークの訓練は同じリズムをたどる。損失を評価し、フレームワークにその勾配を求めさせ、すべての重みを谷の方向へ小さなステップだけ動かすという手順である。これまではその仕組みに関する 2 つの自然な問いに立ち止まることなく、このループに頼ってきた。第 1 に、学習率を 2 倍にした途端、訓練が突然発散したとする。勾配そのものは依然として正しいが、局所的な傾きで損失を確実に予測できる小さな近傍を、そのステップが飛び越えてしまったのである。この線形近似（linear approximation）はどこまで信頼できるのだろうか。第 2 に、勾配がいかに驚くほど低コストであるかに注目しよう。1 億個の重みを 1 つずつ摂動させてすべての傾きを推定するには 1 億回の順伝播（forward pass）を要するが、1 回の逆伝播の呼び出しは、およそ数回の順伝播に相当するコストですべての傾きを返す。なぜそのようなことが可能なのだろうか。

<!--en:5fec5b3b-->
いずれの問いも微分積分に属する。微分積分とは、連続的な変化をどう測るか、そして無限に多くの微小な断片をどう足し合わせるかを Newton と Leibniz が問うた 17 世紀に生まれた分野である。その 2 つの道具である微分と積分は、互いに逆の関係にあることがわかる。深層学習において、導関数は現在の重みの近傍における損失曲面（loss surface）の最良の線形モデルを与え、損失の値そのものを除けば、オプティマイザが目にする唯一の情報である。一方、データや重みが確率的になると直ちに積分が登場する。確率や期待損失は曲線の下の面積であり、方策勾配（policy gradient、:numref:`sec_policygradient`）のような手法では、サンプリング元の分布そのものに関してその積分を微分しなければならない。

<!--en:5fa04170-->
本書の序盤では、微分の基本規則を述べ（:numref:`sec_calculus`）、自動微分を既製のツールとして扱った（:numref:`sec_autograd`）。本編の各章ではそれ以上の知識を必要としないため、この付録の章は、その仕組みがなぜ機能し、その保証がどこから得られるのかを知りたい読者のための任意の読み物である。議論の根幹を担うのは 3 つの結果である。すなわち、曲率（curvature）の上界を :numref:`sec_gd` で引用した安全なステップサイズの規則へと変換する降下補題（descent lemma）、損失側から逆向きに掛けることではるかに低コストとなり誤差逆伝播法（backpropagation、:numref:`sec_backprop`）を導くヤコビ行列（Jacobian）の積としての連鎖律（chain rule）、そして汎用性と分散の間でトレードオフをなす、期待値の勾配に対する相補的な 2 つの推定量である。

<!--en:5d924947-->
本章は 3 つのパートで展開する。まず 1 変数（:numref:`sec_mdl-single_variable_calculus`）、続いて多変数（:numref:`sec_mdl-multivariable_calculus`）において関数の線形近似と二次近似（quadratic approximation）を構築し、安全なステップサイズに上界を与え、極小点と鞍点（saddle point）を見分けられるようにする。次に、計算グラフ（computation graph）上の経路にわたる和（:numref:`sec_mdl-chain-rule-graphs`）から各層のヤコビ行列の積（:numref:`sec_mdl-matrix-calculus`）、さらにはそれらを自動的に乗算するアルゴリズム（:numref:`sec_mdl-matrix-calculus-autodiff`）へと連鎖律をたどる。最後のパートでは積分（:numref:`sec_mdl-integral_calculus`）を導入し、サンプリングと微分を組み合わせて期待値の勾配を推定する（:numref:`sec_mdl-expectation-gradients`）。

<!--en:7d2d7fbb-->
初等的な微分積分にすでに慣れている場合は、いくつかの節の冒頭にあるウォームアップのページを流し読みしてもよい。本章では、勾配を超えるベクトル解析（発散やストークスの定理など）および測度論（measure theory）は割愛する。微分方程式については :numref:`chap_mdl-dynamics` を参照されたい。

<!--en:fe9487c3-->
微分積分と自動微分：局所モデルから誤差逆伝播法まで<br>**テイラー近似・降下補題・勾配とヘッセ行列・計算グラフ・行列微分・自動微分・期待値の勾配**

<!--en:e98fefaa-->
::: {.slide title="すべての訓練ステップの背後にある 2 つの根本的な問い"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:138c191a-->
- **1. 局所的な傾きはどこまで信頼できるか：**勾配 $\nabla\ell(\mathbf{w})$ は一次線形モデルであり、曲率がどれだけ急激に損失を逸らすかによって安全な学習率 $\eta$ が決まる

<!--en:5a2f8c69-->
- **2. なぜ $10^8$ 個の重みの勾配はこれほど低コストなのか：**$10^8$ 個の重みを 1 つずつ摂動させると約 12 日を要するが、1 回の逆伝播は $10^8$ 個すべての厳密な偏微分を約 30 ms で返す

<!--en:8b9709e9-->
::: {.slide title="深層学習の微分積分を支える 3 つの柱"}

<!--en:445a79e0-->
[統一原理]{.kicker}

<!--en:0896dab4-->
- **柱 1・降下補題（§{sec_mdl-single_variable_calculus}〜§{num:sec_mdl-multivariable_calculus}）：**テイラー展開のラグランジュの剰余項により、曲率の上界 $|f''| \le L$ を安全なステップサイズ $0 < \eta < 2/L$（$\eta^\star = 1/L$）へ変換する

<!--en:809fe1e1-->
- **柱 2・リバースモードのヤコビ行列積（§{sec_mdl-chain-rule-graphs}〜§{num:sec_mdl-matrix-calculus-autodiff}）：**$((\mathbf{u}^\top\mathbf{J}_K)\mathbf{J}_{K-1})\cdots\mathbf{J}_1$ を出力側から逆向きに結合して計算することで、すべての勾配を $O(1)$ 回のスイープで求める

<!--en:3893726c-->
- **柱 3・期待値の微分（§{sec_mdl-integral_calculus}〜§{num:sec_mdl-expectation-gradients}）：**スコア関数（REINFORCE）とパスワイズ再パラメータ化は、汎用性とモンテカルロ分散の間でトレードオフをなす

<!--en:45784cea-->
::: {.slide title="第 §{num:chap_mdl-calculus} 章のロードマップ：1 次元スライスからサンプリングまで"}

<!--en:b514d0e1-->
[体系的な展開]{.kicker}

<!--en:e8b513fb-->
- **パート I・局所モデル（§{sec_mdl-single_variable_calculus}〜§{num:sec_mdl-multivariable_calculus}）：**テイラー多項式、ラグランジュの剰余項、降下補題、勾配 $\nabla f \perp$ 等位集合、対称なヘッセ行列 $\mathbf{H}_f$

<!--en:acfcff02-->
- **パート II・連鎖律と自動微分（§{sec_mdl-chain-rule-graphs}〜§{num:sec_mdl-matrix-calculus-autodiff}）：**DAG 上の経路和、全微分法 $df = \langle \nabla f, d\mathbf{X}\rangle$、二重数、Wengert テープ、$2\sqrt{T}$ チェックポイント保存、HVP、陰関数微分

<!--en:dafbee57-->
- **パート III・積分と期待値（§{sec_mdl-integral_calculus}〜§{num:sec_mdl-expectation-gradients}）：**リーマン和、微分積分学の基本定理、フビニの定理、フローにおける $|\det\mathbf{J}_{\boldsymbol{\phi}}|$、ライプニッツ則、REINFORCE、パスワイズ勾配

<!--en:742191d0-->
::: {.slide title="各節の軸となる通しベンチマーク"}

<!--en:a077b1cf-->
[各節をつなぐ具体例]{.kicker}

<!--en:7f1efce0-->
- **局所モデルと行列モデル（§{sec_mdl-single_variable_calculus}〜§{num:sec_mdl-matrix-calculus}）：**ロジスティック損失 $\ell(w) = \log(1+e^{-w})$（$L = 1/4$）、$(0, \log 2)$ における log-sum-exp（$\nabla f = (\tfrac{1}{3}, \tfrac{2}{3})^\top$）、4 入力の整数 DAG（$\bar{w} = 73{,}472$）、2-2-1 ReLU MLP

<!--en:d70237ab-->
- **自動微分と確率的微分積分（§{sec_mdl-matrix-calculus-autodiff}〜§{num:sec_mdl-expectation-gradients}）：**スクラッチ実装の `Dual`/`Var` エンジン（$802$ 対 $80{,}200$ 演算）、極座標によるガウス積分 $\sqrt{2\pi}$、$\mathbb{E}[X^2]$ の分散（$\sigma = 2.0$ 対 $7.18$）

<!--en:c74ef8f3-->
::: {.slide title="第 §{num:chap_mdl-calculus} 章の読み進め方"}

<!--en:123f19cf-->
[学習ガイド]{.kicker}

<!--en:8c69f975-->
- **初等的な微分積分に慣れている場合：**§{sec_mdl-single_variable_calculus} の導関数の表と §{sec_mdl-integral_calculus} のリーマン和・原始関数の復習は流し読みし、**平均値の定理、ラグランジュの剰余項、降下補題**（§{sec_mdl-single_variable_calculus}）および **$\mathbb{R}^n$ における変数変換**（§{sec_mdl-integral_calculus}）へ直接進む

<!--en:deb5217b-->
- **深層学習の理論と導出に不可欠：**§{sec_mdl-matrix-calculus} の**全微分法** $df = \langle \nabla f, d\mathbf{X}\rangle$ とトレースの移動則を習得すれば、100 項目もの行列微分の公式表が不要になり、正方行列で表面化しない転置バグを防げる

<!--en:8f5ffabd-->
- **システムと現代の機械学習に不可欠：**リバースモードが約 3 回の順伝播のコストで済む理由、活性化チェックポイントがメモリを $O(T)$ から $2\sqrt{T}$ へ削減する仕組み、フォワード・オーバー・リバースが 4 TB のヘッセ行列を作らずに $\mathbf{H}\mathbf{v}$ を計算する方法については **§{sec_mdl-matrix-calculus-autodiff}（自動微分）**を、モンテカルロ勾配の分散については **§{sec_mdl-expectation-gradients}** を学習すること

<!--en:e680ad79-->
**次の節（§{sec_mdl-single_variable_calculus} 1 変数の微分積分）：**一階導関数と二階導関数によって局所的な線形モデルと二次モデルを構築し、近似誤差に上界を与え、勾配降下法の安全なステップサイズを保証する仕組み
