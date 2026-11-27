<!--en:168ab129-->
# 記法

<!--en:dbaf2a9a-->
線形代数、微積分、確率、最適化、そして現代のニューラルアーキテクチャにまたがる本書において、明確で一貫した記法は、数学的な概念とコードとを結びつける大きな助けとなる。本書全体を通して、統一された組版上の階層に従う。すなわち、小文字の斜体（$x$）はスカラー、小文字の太字（$\mathbf{x}$）はベクトル、大文字の太字（$\mathbf{X}$）は行列、サンセリフ体の大文字（$\mathsf{X}$）は高階のテンソル、カリグラフィー体（$\mathcal{X}$）は集合、大文字の斜体（$X$）は確率変数を表す。

<!--en:3a7674bd-->
以下の一覧には、*一般的な組版上の慣例*を示す項目（たとえば「$x$：スカラー」は、小文字の斜体が一般にスカラーを表すことを意味する）と、「$\mathbb{R}$：実数全体の集合」のように*特定の固定された意味*を持つ記号とが含まれる。

<!--en:38fc2576-->
使える文字の数には限りがあるため、いくつかの標準的な文字は分野をまたいで複数の意味を持つ。いずれの場合も、記号の構文上の形式と文脈から意味は一意に定まる。たとえば、シグモイド関数 $\sigma(\cdot)$ は括弧付きの引数をとり、標準偏差 $\sigma_X$ は確率変数を下付き添字に持ち、特異値 $\sigma_i(\mathbf{X})$ は添字付きで行列に適用され、拡散モデル（diffusion model）のノイズスケジュール $\sigma_t$ は時刻 $t$ を添字にとる。混同しやすい記号については、以下で明示的に注意を付す。また、分野固有の語彙が多く導入される章では、最適化（:numref:`tab_mdl-opt-notation`）、情報理論（:numref:`tab_mdl-it-notation`）、ダイナミクス（:numref:`tab_mdl-dyn-notation`）、強化学習（:numref:`tab_rl_notation`）、敵対的生成ネットワーク（:numref:`tab_gan_notation`）、拡散モデル（:numref:`tab_diffusion-notation`）をはじめ、章の冒頭に専用の記法一覧表を設けている。本章の後半の各節では、いくつかの発展的なトピックについて主要な記法をまとめる。

<!--en:b543a219-->
## 数値オブジェクト

<!--en:1553b069-->
* $x$：スカラー

<!--en:6f7adca9-->
* $\mathbf{x}$：ベクトル（デフォルトでは列ベクトル）

<!--en:9039f3bc-->
* $(x_1, \ldots, x_n)$：成分 $x_1, \ldots, x_n$ を持つ列ベクトルのインライン表記。地の文では「行 $(0, 1, 2)$ と $(3, 4, 5)$」のように、行列の行も同じ形式で表記する

<!--en:c4155746-->
* $\mathbf{X}$：行列

<!--en:5382b6c0-->
* $\mathsf{X}$：2 つより多くの軸を持つテンソル

<!--en:a6b889b3-->
* $\mathbf{I}$：文脈から定まる次元の単位行列（主対角成分が $1$ で、それ以外の成分がすべて $0$ の正方行列）

<!--en:8a2976ff-->
* $\mathbf{0}$、$\mathbf{1}$：文脈から定まる次元の、すべての成分が 0 のベクトルおよびすべての成分が 1 のベクトル

<!--en:e6f5ea3a-->
* $x_i$：ベクトル $\mathbf{x}$ の $i^\textrm{th}$ 要素

<!--en:fba12516-->
* $x_{ij}$、$[\mathbf{X}]_{i,j}$：行列 $\mathbf{X}$ の第 $i$ 行第 $j$ 列の要素

<!--en:fcd1b3c7-->
## 集合論

<!--en:66d52d87-->
* $\mathcal{X}$：集合

<!--en:53f0b6d2-->
* $\mathbb{R}$：実数全体の集合

<!--en:0bb19b3c-->
* $\mathbb{R}^n$：$n$ 次元実ベクトル全体の集合

<!--en:a77cb2f0-->
* $\mathbb{R}^{a\times b}$：$a$ 行 $b$ 列の実数行列全体の集合

<!--en:2af64644-->
* $|\mathcal{X}|$：集合 $\mathcal{X}$ の濃度（要素数）

<!--en:add2aa69-->
* $\mathcal{A}\cup\mathcal{B}$：集合 $\mathcal{A}$ と $\mathcal{B}$ の和集合

<!--en:288bba92-->
* $\mathcal{A}\cap\mathcal{B}$：集合 $\mathcal{A}$ と $\mathcal{B}$ の共通部分

<!--en:b3f689c6-->
* $\mathcal{A}\setminus\mathcal{B}$：差集合（$\mathcal{A}$ の要素のうち $\mathcal{B}$ に属さないものからなる集合）

<!--en:c7827ae0-->
## 関数と演算子

<!--en:fb5d3c14-->
* $f(\cdot)$：関数

<!--en:e82af5d0-->
* $\log(\cdot)$：自然対数（底は $e$）

<!--en:d1fa7655-->
* $\log_2(\cdot)$：底を $2$ とする対数

<!--en:102e2f62-->
* $\exp(\cdot)$：指数関数

<!--en:75fcc17d-->
* $\sigma(\cdot)$：ロジスティックシグモイド関数 $\sigma(x) = 1/(1 + e^{-x})$

<!--en:3108b61c-->
* $\mathrm{softmax}(\mathbf{o})$：各成分が $\exp(o_i) / \sum_j \exp(o_j)$ であるソフトマックスベクトル

<!--en:5be09752-->
* $\mathbf{1}(\cdot)$：指示関数（indicator function）。真偽値の引数が真のとき $1$、それ以外のとき $0$ をとる（引数をとることで、すべての成分が 1 のベクトル $\mathbf{1}$ と区別される）

<!--en:734d3617-->
* $\operatorname{argmax}_x f(x)$、$\operatorname{argmin}_x f(x)$：$x$ の値のうち、$f(x)$ を最大化または最小化するもの

<!--en:83e60b16-->
* $\sum$：要素の集まりにわたる総和

<!--en:06207a7e-->
* $\prod$：要素の集まりにわたる総乗

<!--en:dcdf5e98-->
* $O(\cdot)$：$O(n^2)$ 回の演算コストのように、定数倍を除いた漸近的な上界

<!--en:6fe9b800-->
* $\propto$：比例関係（着目している変数に依存しない定数倍だけ両辺が異なることを表す）

<!--en:1a4dc64e-->
* $\stackrel{\textrm{def}}{=}$：左辺の記号を定義する等号

<!--en:24e95e8f-->
## 線形代数

<!--en:2b69e940-->
* $(\cdot)^\top$：ベクトルまたは行列の転置

<!--en:8e257c98-->
* $\mathbf{X}^{-1}$：正方行列 $\mathbf{X}$ の逆行列

<!--en:f758a663-->
* $\mathbf{X}^{+}$：行列 $\mathbf{X}$ のムーア・ペンローズ擬似逆行列（pseudoinverse）

<!--en:8069dc7a-->
* $\langle \mathbf{x}, \mathbf{y} \rangle$：ベクトル $\mathbf{x}$ と $\mathbf{y}$ の内積（ドット積）。$\mathbf{x}^\top \mathbf{y}$ とも表記する

<!--en:cc0f7397-->
* $\odot$：アダマール積（要素ごとの積）

<!--en:2e0e7e5c-->
* $[\cdot, \cdot]$：ベクトルまたは行列の連結（concatenation）

<!--en:f5af0971-->
* $\|\cdot\|_p$：ベクトルの $\ell_p$ ノルム

<!--en:b1629f49-->
* $\|\cdot\|$：ユークリッドノルム（$\ell_2$ ノルム）

<!--en:b6102a93-->
* $\|\mathbf{X}\|_2$：行列 $\mathbf{X}$ のスペクトルノルム（作用素ノルム）。最大の特異値に等しい

<!--en:59374292-->
* $\|\mathbf{X}\|_F$：行列 $\mathbf{X}$ のフロベニウスノルム。全成分の $\ell_2$ ノルムに等しい

<!--en:c08cca10-->
* $\|\mathbf{X}\|_*$：行列 $\mathbf{X}$ の核ノルム（トレースノルム）。特異値の総和に等しい

<!--en:83a161a4-->
* $\operatorname{diag}(\mathbf{x})$：ベクトル $\mathbf{x}$ の成分を主対角線上に並べた対角行列

<!--en:67f2c341-->
* $\operatorname{rank}\mathbf{X}$、$\operatorname{tr}\mathbf{X}$、$\det\mathbf{X}$：行列 $\mathbf{X}$ のランク、トレース、行列式

<!--en:02170e28-->
* $\lambda_i$、$\lambda_{\max}(\mathbf{X})$：正方行列の固有値、および対称行列 $\mathbf{X}$ の最大固有値

<!--en:e198b385-->
* $\sigma_i(\mathbf{X})$：行列 $\mathbf{X}$ の大きい方から $i^\textrm{th}$ の特異値（シグモイド関数 $\sigma(\cdot)$ と混同しないこと）

<!--en:26003fb6-->
* $\kappa(\mathbf{X})$：行列 $\mathbf{X}$ の条件数（condition number）。最大の特異値と最小の非ゼロ特異値との比に等しい

<!--en:6381befc-->
* $\rho(\mathbf{X})$：正方行列 $\mathbf{X}$ のスペクトル半径（spectral radius）。固有値の絶対値（複素絶対値）の最大値に等しい（後述の相関係数 $\rho(X, Y)$ と混同しないこと）

<!--en:c322f183-->
* $\mathbf{X}\succeq 0$、$\mathbf{X}\succ 0$：対称行列 $\mathbf{X}$ が半正定値（positive semidefinite、$\mathbf{X}\succeq 0$）または正定値（positive definite、$\mathbf{X}\succ 0$）であること

<!--en:60cfb36d-->
## 微積分

<!--en:bdb91049-->
* $\frac{dy}{dx}$、$f'(x)$：$y$ の $x$ に関する導関数、および関数 $f$ の導関数の点 $x$ における値

<!--en:f952525c-->
* $\frac{\partial y}{\partial x}$：$y$ の $x$ に関する偏微分

<!--en:dcee0dfb-->
* $\nabla_{\mathbf{x}} y$：スカラー $y$ のベクトル $\mathbf{x}$ に関する勾配

<!--en:33e03308-->
* $\mathbf{J}$、$\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$：ベクトル値関数 $\mathbf{f}$ の $\mathbf{x}$ に関するヤコビ行列

<!--en:fb49d508-->
* $\nabla^2 f$：$f$ の 2 階偏微分を並べたヘッセ行列

<!--en:36e0899e-->
* $\nabla \cdot \mathbf{f}$、$\Delta f$：ベクトル場 $\mathbf{f}$ の発散（divergence）、およびスカラー関数 $f$ のラプラシアン

<!--en:8785498d-->
* $\int_a^b f(x) \,dx$：関数 $f$ の $a$ から $b$ までの $x$ に関する定積分

<!--en:7704e7ef-->
* $\int f(x) \,dx$：関数 $f$ の $x$ に関する不定積分（原始関数）

<!--en:41832806-->
## 確率と情報理論

<!--en:f4e1706c-->
* $X$：確率変数

<!--en:ff5c8c1d-->
* $P$：確率分布

<!--en:7c064ec6-->
* $X \sim P$：確率変数 $X$ が分布 $P$ に従うこと

<!--en:23431f0e-->
* $P(X=x)$：確率変数 $X$ が値 $x$ をとる事象に割り当てられた確率

<!--en:3e1deb21-->
* $P(X \mid Y)$：$Y$ が与えられたもとでの $X$ の条件付き確率分布

<!--en:e56da4d5-->
* $p(\cdot)$：分布 $P$ に対応する確率密度関数（probability density function、PDF）または確率質量関数（probability mass function、PMF）

<!--en:c24c492c-->
* $\mathbb{E}[X]$：確率変数 $X$ の期待値

<!--en:ea850fe5-->
* $\mathbb{E}_{x \sim P}[f(x)]$：$x$ が分布 $P$ からサンプリングされるときの $f(x)$ の期待値

<!--en:cf0c9ffa-->
* $X \perp Y$：確率変数 $X$ と $Y$ が統計的に独立であること

<!--en:a55dcb48-->
* $X \perp Y \mid Z$：確率変数 $X$ と $Y$ が、$Z$ が与えられたもとで条件付き独立であること

<!--en:3fcde23b-->
* $\sigma$、$\sigma_X$：一般の標準偏差、および特に確率変数 $X$ の標準偏差

<!--en:ffbcd637-->
* $\textrm{Var}(X)$：確率変数 $X$ の分散。$\sigma^2_X$ に等しい

<!--en:b0dc3197-->
* $\textrm{Cov}(X, Y)$：確率変数 $X$ と $Y$ の共分散

<!--en:c0c96095-->
* $\boldsymbol{\Sigma}$：共分散行列

<!--en:5956b357-->
* $\rho(X, Y)$：$X$ と $Y$ のピアソン相関係数。$\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$ に等しい

<!--en:c7936b1c-->
* $\mathcal{N}(\mu, \sigma^2)$：平均 $\mu$、分散 $\sigma^2$ の単変量ガウス分布（正規分布）。$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ はその多変量版を表す

<!--en:72798c23-->
* $H(X)$、$h(X)$：離散確率変数 $X$ のシャノンエントロピー、および連続確率変数 $X$ の微分エントロピー（differential entropy）

<!--en:b99715fd-->
* $I(X; Y)$：確率変数 $X$ と $Y$ の相互情報量（mutual information）

<!--en:0509fe36-->
* $D_{\textrm{KL}}(P\|Q)$：分布 $P$ から分布 $Q$ へのカルバック・ライブラー（Kullback–Leibler、KL）ダイバージェンス（相対エントロピー）

<!--en:1927e1b8-->
## データ、モデル、訓練

<!--en:a07a11bd-->
* $n$：データセットに含まれるデータ例の数

<!--en:03fa9b0e-->
* $d$：入力特徴量（feature）の数（1 つのデータ例の次元数）

<!--en:16ce3abd-->
* $\mathbf{x}^{(i)}$、$y^{(i)}$：$i^\textrm{th}$ のデータ例の特徴ベクトルと目的ラベル

<!--en:0107d01d-->
* $\mathcal{D}$：$\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ などのデータセット

<!--en:75138db0-->
* $\mathbf{X} \in \mathbb{R}^{n \times d}$：1 行に 1 つのデータ例を並べた計画行列（design matrix）またはミニバッチ行列

<!--en:7a656ac6-->
* $\mathcal{B}$：サイズ $|\mathcal{B}|$ のデータ例のミニバッチ

<!--en:c096f8d5-->
* $\hat{y}$：目的値 $y$ に対するモデルの予測

<!--en:63bf8d44-->
* $\mathbf{w}$、$b$：線形モデルの重みベクトルとスカラーバイアス

<!--en:288190d2-->
* $\mathbf{W}^{(\ell)}$、$\mathbf{b}^{(\ell)}$：第 $\ell$ 層の重み行列とバイアスベクトル（例：$\mathbf{W}^{(1)}$）

<!--en:18bbe889-->
* $\theta$、$\boldsymbol{\theta}$：モデルのパラメータ（1 つのベクトルにまとめる場合は太字で $\boldsymbol{\theta}$ と表記する）

<!--en:e3d2f287-->
* $l$、$\ell$：データ例ごとの損失関数（loss function）

<!--en:c50e10e7-->
* $L$：データセットまたはミニバッチ全体で平均した経験損失（empirical loss）

<!--en:bcd6dd8e-->
* $\eta$：勾配ベースのオプティマイザにおける学習率（learning rate、ステップサイズ）

<!--en:08a8c7c8-->
* $\lambda$：$\ell_2$ 重み減衰（weight decay）ペナルティ $\frac{\lambda}{2}\|\mathbf{w}\|^2$ の正則化強度

<!--en:77407414-->
* $t$、$T$：離散または連続のタイムステップ $t$、および系列（sequence）の全長 $T$

<!--en:6bdec9b0-->
* $\mathbf{x}_t$、$\mathbf{h}_t$：タイムステップ $t$ における入力と隠れ状態（hidden state）

<!--en:8eac3302-->
## アテンションと Transformer

<!--en:f1492a82-->
これらの記号は :numref:`chap_attention` で導入され、:numref:`chap_transformers` およびそれに基づく後続の章全体で用いられる。

<!--en:e1817408-->
* $\mathbf{q}$、$\mathbf{k}_i$、$\mathbf{v}_i$：クエリベクトル、および項目 $i$ のキーベクトルと値ベクトル（:numref:`sec_queries-keys-values`）

<!--en:0dc2f7fe-->
* $a(\mathbf{q}, \mathbf{k})$：アテンションスコア関数（スケール化ドット積アテンションにおける $\mathbf{q}^\top \mathbf{k} / \sqrt{d}$ など）

<!--en:80c27aca-->
* $\alpha(\mathbf{q}, \mathbf{k}_i)$：すべてのキーにわたるスコアにソフトマックス関数を適用して得られる、キー $i$ に割り当てられたアテンション重み

<!--en:53bf7480-->
* $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$：$n$ 個のクエリ、$m$ 個のキー、$m$ 個の値を 1 行に 1 つずつ積み重ねた行列。スケール化ドット積アテンションはこれらを $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})\,\mathbf{V}$ に写す（:numref:`sec_attention-scoring-functions`）

<!--en:f54ed6a4-->
* $d$、$d_v$：クエリとキーの次元（幅、$d$）、および値の次元（$d_v$）

<!--en:3c50e9f1-->
* $H$、$d_h$：アテンションヘッドの数、およびヘッドごとの次元 $d_h = d/H$（:numref:`sec_multihead-attention`）

<!--en:c58b3d4a-->
## 状態空間モデル

<!--en:a2a7e4cc-->
これらの記号は :numref:`chap_modern_rnn` で導入される。

<!--en:5c0964d0-->
* $\mathbf{A}$、$\mathbf{B}$、$\mathbf{C}$、$D$：状態空間モデル（:numref:`subsec_state-space`）における状態遷移行列、入力射影、出力読み出し（readout）、および直達項（feedthrough、スキップ項）

<!--en:b0759425-->
* $N$：チャネルごとの潜在状態の次元

<!--en:403dc13d-->
* $\lambda_n$：対角状態行列の $n^\textrm{th}$ の対角成分（レート）。対応する状態成分はステップごとに $e^{\Delta \lambda_n}$ 倍に減衰する

<!--en:e3625723-->
* $\Delta$、$\Delta_t$：離散化（discretization）のステップサイズ（:numref:`subsec_zoh`）、および Mamba における入力依存（選択的）なステップサイズ（:numref:`sec_mamba`）

<!--en:c55e8f68-->
* $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$：外積 $\mathbf{v}_t \mathbf{k}_t^\top$ によって更新され、クエリ $\mathbf{q}_t$ で読み出される行列値の再帰状態（:numref:`sec_matrix-state`）。ここでは、キーの次元を $d_k$ と表記し、値の次元 $d_v$ と並べて用いる

<!--en:34b5c20e-->
* $\beta_t$：デルタ更新則（:numref:`sec_deltanet`）における書き込み強度（学習率）

<!--en:cbe6efec-->
## 強化学習

<!--en:c96e7fff-->
これらの記号は :numref:`chap_reinforcement_learning` および :numref:`chap_deep_rl` 全体で用いられる。各記号が定義される節を含む完全な一覧は :numref:`tab_rl_notation` に示す。

<!--en:599d1165-->
* $\mathcal{S}$、$\mathcal{A}$：状態空間と行動空間

<!--en:001349d1-->
* $s_t$、$a_t$、$r_t$：タイムステップ $t$ における状態、行動、およびスカラー報酬（reward）

<!--en:44054002-->
* $P(s' \mid s, a)$：状態 $s$ で行動 $a$ をとったときに状態 $s'$ へ遷移する確率を与えるマルコフ遷移カーネル

<!--en:06cd841a-->
* $r(s, a)$：状態 $s$ で行動 $a$ をとったときの期待即時報酬

<!--en:279ca80d-->
* $\gamma$：割引率（discount factor）、$\gamma \in [0, 1)$

<!--en:21544d8c-->
* $\tau$、$R(\tau)$：軌跡（trajectory）$(s_0, a_0, r_0, s_1, \ldots)$、およびその割引収益（return）$R(\tau) = \sum_t \gamma^t r_t$

<!--en:ac408b2c-->
* $\pi(a \mid s)$：状態 $s$ で行動 $a$ を選択する確率を与える確率的方策（policy）。決定論的方策では、$\pi(s)$ が選択される行動を表す

<!--en:5f6b338a-->
* $\pi_\theta$、$J(\theta)$：パラメータ $\theta$ によってパラメータ化された方策、およびその期待割引収益

<!--en:2e31653a-->
* $V^\pi(s)$、$Q^\pi(s, a)$：方策 $\pi$ のもとでの状態価値関数と行動価値関数（状態 $s$ から開始したときの期待収益、および状態 $s$ で行動 $a$ をとったときの期待収益）

<!--en:a7e33bab-->
* $V^*$、$Q^*$：最適状態価値関数と最適行動価値関数

<!--en:32e96a75-->
* $A^\pi(s, a)$：アドバンテージ関数 $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$

<!--en:dc80fa03-->
* $\delta_t$：ステップ $t$ における時間差分誤差（temporal-difference error、TD 誤差）

<!--en:c3a2bb6c-->
* $\epsilon$：$\epsilon$ 貪欲な行動選択における探索（exploration）の確率

<!--en:f351aaeb-->
## 敵対的生成ネットワーク

<!--en:f1166b6a-->
これらの記号は :numref:`chap_gans` で用いられるものである。:numref:`tab_gan_notation` には、すべての記号が定義箇所の節および文献でよく使われる別記法とともにまとめられている。

<!--en:4bbfd87f-->
* $p$、$q$、$m$：真のデータ密度、生成器（generator）のモデル密度、およびそれらの等重み混合密度 $m = \tfrac12(p + q)$

<!--en:823ac510-->
* $x \sim p$、$x' \sim q$：データ分布から得られた実サンプルと、生成器から得られた合成サンプル

<!--en:4d71ce16-->
* $y$：サンプルの出所を表す二値ラベル（実データでは $y = 1$、生成サンプルでは $y = 0$）

<!--en:e87417c0-->
* $z$、$G$：潜在ノイズ変数と、$x' = G(z)$ を出力する生成器ネットワーク

<!--en:36103bee-->
* $D$：クリティック（または識別器）が出力する真実性のロジット。これにより $\sigma(D(x))$ はサンプル $x$ が実データである推定確率を表す

<!--en:08d42f7f-->
* $V$：対数損失ミニマックスゲームの価値関数

<!--en:e8ec8891-->
* $\rho = p/q$、$\lambda = \log \rho$：尤度比（likelihood ratio、密度比）と対数密度比

<!--en:20f2df45-->
* $\eta = p/(p + q)$：サンプルが実データであるベイズ最適クラス事後確率（なお :numref:`chap_gans` の中では、オプティマイザのステップサイズには $h$ を用い、$\eta$ は使わない）

<!--en:180c1813-->
* $f$、$D_f$：ダイバージェンス関数とその $f$ ダイバージェンス

<!--en:2d61659e-->
* $\Phi$、$d_{\mathrm{Rp}}$：ペアリング（相対論的、relativistic）目的関数と、最適クリティックにおけるその値

<!--en:f0ad3ecb-->
* $\gamma$、$R_1$、$R_2$：勾配ペナルティの重み、ならびに実データ上（$R_1$）および生成サンプル上（$R_2$）で評価されるゼロ中心の勾配ペナルティ

<!--en:29b79ea9-->
* $c$：クラスラベルやテキストプロンプトなどの条件付け変数

<!--en:f23c6307-->
## 拡散モデル

<!--en:03ec2647-->
これらの記号は :numref:`chap_diffusion` で用いられるもので、:numref:`chap_mdl-dynamics`（:numref:`tab_mdl-dyn-notation`）の慣例と一致している。本書全体を通して、時間の向きは単一のものに統一する。すなわち、クリーンデータは $t = 0$ に、純粋なノイズは $t = 1$ にあり、生成サンプラは $t = 1$ から $t = 0$ へと逆向きに積分する（フローマッチングの論文の中にはこの向きを逆にするものもある。:numref:`sec_diffusion-gaussian-paths` を参照）。この章で使われる記号は、論文で用いられる表記とともに :numref:`tab_diffusion-notation` にまとめられている。

<!--en:4467f835-->
* $\mathbf{x}_0$、$\boldsymbol{\epsilon}$：クリーンなデータ点と、独立な標準ガウスノイズのベクトル $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

<!--en:e2a8f576-->
* $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$：ガウス確率経路に沿った時刻 $t \in [0, 1]$ におけるノイズ付き状態

<!--en:cafaa6cb-->
* $\alpha_t$、$\sigma_t$：経路に沿った信号スケジュールとノイズスケジュールの係数（学習率やデータの標準偏差とは異なる）

<!--en:65ffe1b2-->
* $\bar\alpha_t = \alpha_t^2$、$\beta(t)$：分散保持（variance-preserving）経路の二乗信号係数とそのレート（:numref:`sec_diffusion-gaussian-paths`）

<!--en:fe3627af-->
* $k$、$T$、$\beta_k$：DDPM 連鎖のステップ、総ステップ数、およびステップ $k$ で加えられる分散（:numref:`sec_diffusion-gaussian-paths`）

<!--en:1806198a-->
* $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$：対数信号対雑音比（log signal-to-noise ratio、log-SNR）。$+\infty$（クリーンデータ、$t=0$）から $-\infty$（純粋なノイズ、$t=1$）まで単調に減少する

<!--en:83b101f7-->
* $p_t$、$\nabla \log p_t$：$\mathbf{x}_t$ の時刻 $t$ における周辺密度、およびそのスコア関数（score function）

<!--en:871a4b2a-->
* $\hat{\mathbf{x}}_0$、$\hat{\boldsymbol{\epsilon}}$：ノイズ除去ネットワークが予測する事後期待値 $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$ と $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$

<!--en:3d2c85b6-->
* $\mathbf{u}_t$、$\mathbf{u}_{\boldsymbol{\theta}}$、$\hat{\mathbf{u}}$：経路に沿ってサンプルを輸送する確率フロー（probability flow）の速度場、それを近似するよう訓練されたパラメータ $\boldsymbol{\theta}$ を持つニューラルネットワーク、およびそのネットワークによる予測速度

<!--en:1a3753de-->
* $\bar{\mathbf{x}} = \mathbf{x}_t / \alpha_t$、$\bar\sigma = \sigma_t / \alpha_t$：すべてのガウス経路が $\bar{\mathbf{x}} = \mathbf{x}_0 + \bar\sigma \boldsymbol{\epsilon}$ の形になり、サンプラがステップを進める座標系（:numref:`sec_diffusion-integration`）。ここでの上線は $\alpha_t$ による除算を意味し、$\bar\alpha_t = \alpha_t^2$ における二乗とは異なる

<!--en:90803d42-->
* $\mathbf{v} = \alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_0$：$\mathbf{v}$ 予測（:numref:`sec_diffusion-gaussian-paths`）における目標ベクトル

<!--en:ee4944bd-->
* $\tilde{\mathbf{u}}$：分類器なしガイダンス（classifier-free guidance）を適用した速度場（:numref:`subsec_diffusion-images-guided`）

<!--en:853446e5-->
* $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$、$\mathbf{u}_{\textrm{ref}}$：時刻 $t$ と $s$ の間の平均速度（average velocity）、および少数ステップ蒸留の損失（:numref:`sec_diffusion-few-step`）で用いられる参照速度（教師モデルの予測 $\hat{\mathbf{u}}$ または条件付き速度 $\boldsymbol{\epsilon} - \mathbf{x}_0$）

<!--en:0c6982e3-->
* $c$、$\gamma$：条件付け信号およびガイダンス強度（:numref:`subsec_diffusion-images-guided`）

<!--en:75498650-->
[ディスカッション](https://d2l.discourse.group/t/25)
