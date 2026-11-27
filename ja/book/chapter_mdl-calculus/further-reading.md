<!--en:7902ddfd-->
# 関連文献

<!--en:d2ffd1d7-->
本章全体を通して、深層学習の実践では日常的に踏み越えられる見通しのよい仮定のもとで主要な結果を証明してきた。実際には、ReLU ネットワークには鋭い折れ点（kink）が至るところにあり、リバースモードのテープは容易に GPU メモリを超過し、1,000 次元の積分はグリッド上に配置できない。以下の短い解説では、こうした仮定が崩れたときに何が起こるかをまとめ、続いて本章の主要内容を扱う教科書、講義、チュートリアルを厳選して紹介する。なかでも日常的な訓練に直結するため最初に読んでおきたいトピックが 2 つある。1 つは、折れ点を経由する誤差逆伝播法（backpropagation）がどのような意味で依然として妥当であるかを説明する保存場（conservative field）であり、もう 1 つは、長い計算においてメモリを扱いやすい範囲に抑える再帰的チェックポイント保存（recursive checkpointing）である。本書の他の箇所で本節の内容を前提とするものはない。

<!--en:6b973d86-->
## 折れ点における微分

<!--en:aff6a25c-->
### 保存場

<!--en:358b97f6-->
:numref:`sec_mdl-single_variable_calculus` で例示したように、標準的な規約 $\mathrm{ReLU}'(0)=0$ を採用すると奇妙な現象が生じる。恒等関数を $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$ と表すと、誤差逆伝播法は傾き $0$ を $x=0$ において返すが、この値は $x$ の真の導関数でもなければ有効な劣勾配（subgradient）ですらない。古典的な劣微分（subdifferential）ではこの挙動を説明できない。構成要素の関数が凸でなくなった途端、和の法則や連鎖律（chain rule）が集合の包含関係としてしか成り立たなくなるからである。この隔たりを埋めるため、:citet:`Bolte.Pauwels.2021` は*保存場*を導入した。これは構成上そのまま連鎖律を満たし、測度ゼロの集合を除くいたるところで通常の勾配と一致する、導関数の集合値への一般化である。各折れ点における傾きの選択をどのように固定しても、誤差逆伝播法は常にそのような場の要素を返す。では、なぜ折れ点からなる例外集合は問題にならないほど小さいのだろうか。ラデマッハの定理（Rademacher's theorem）:cite:`Rademacher.1919` により、$\mathbb R^n$ 上のすべてのリプシッツ関数はほとんどいたるところで微分可能であるため、ReLU ネットワークの損失が持つ折れ点は測度ゼロの集合にとどまる。

<!--en:303b85df-->
同論文は、誤差逆伝播法によるこれらの値で駆動される確率的勾配降下法（stochastic gradient descent）が依然として収束するかという実務上の問いにも決着をつけている。標準的なニューラルネットワークの層から構成された損失に対し、反復点の有界性と、ステップサイズおよび勾配ノイズ（gradient noise）に関する通常の条件を仮定すれば、反復点は保存場がゼロを含む臨界点（critical point）の集合へと収束する。この保証は厳密演算のもとで成り立つものであるが、有限精度の浮動小数点演算では、反復点が実際に折れ点の上に着地することがある。:numref:`sec_mdl-single_variable_calculus` では、:citet:`Bertoin.Bolte.Gerchinovitz.ea.2021` が計測し 2023 年の正誤表で修正した 3 つの数値精度にわたる発生頻度を紹介するとともに、$\mathrm{ReLU}'(0)$ の選び方が結果的に訓練へほとんど影響しないことを示している。

<!--en:6e481674-->
### 劣勾配法

<!--en:862dfc2d-->
滑らかな最適化は、勾配と逆向きの小さなステップが目的関数を下げることを保証する降下補題（descent lemma）に依拠している。滑らかでない関数ではこの保証が崩れる。$|x|$ の最小点において、有効な劣勾配 $\tfrac12$ は実際には上り方向を指すことが :numref:`sec_mdl-single_variable_calculus` に示されている。それでもなお驚くべきことに、劣勾配法（subgradient method）は収束する。その解析が目的関数値（objective value）そのものではなく、最小点（minimizer）までのユークリッド距離を追跡するからである。$\sum_t\eta_t=\infty$ かつ $\sum_t\eta_t^2<\infty$ を満たす減衰ステップサイズを用いれば、それまでに得られた最良の目的関数値は最小値へと収束する。最適な収束率（convergence rate）を得るには別のスケジュールが必要となる。劣勾配のノルムが高々 $G$ である凸関数に対し、最小点から距離 $R$ の点を出発点として定数ステップサイズ $R/(G\sqrt T)$ を用いると、$T$ 個の反復点のうち最良のものは最小値から $RG/\sqrt T$ 以内に到達する。次元が $T$ を超える場合、劣勾配のみに依拠するいかなる手法も、この収束率を定数倍を超えて改善することはできない :cite:`Nesterov.2018`。和の法則が等号で成り立つための厳密な条件を含め、劣微分法の体系的な展開については :citet:`Rockafellar.1970` を参照されたい。

<!--en:44bd0467-->
## ベクトル写像と行列写像の微分

<!--en:f29c7da0-->
### ベクトル値写像に対する平均値の定理

<!--en:4a2725c0-->
1 次元では、:numref:`sec_mdl-single_variable_calculus` の平均値の定理（mean value theorem）により、ある中間点 $\xi$ が存在して $f(b)-f(a)=f'(\xi)(b-a)$ が成り立つことが保証される。ベクトル値曲線 $\boldsymbol\gamma:\mathbb R\to\mathbb R^n$ に対しても同じ等式が成り立つと考えたくなるが、実際にはまったく成り立たない。単位円 $\boldsymbol\gamma(t)=(\cos t,\sin t)$ を考えてみよう。$t=2\pi$ で 1 周すると曲線は出発点に戻るため $\boldsymbol\gamma(2\pi)-\boldsymbol\gamma(0)=\mathbf 0$ となるが、速度 $\boldsymbol\gamma'(\xi)$ は単位長さを持ち、$\mathbf 0$ となる $\xi$ は一切存在しない。高次元で生き残るのは、*平均値不等式*（mean value inequality）$\|\boldsymbol\gamma(b)-\boldsymbol\gamma(a)\|\le\sup_\xi\|\boldsymbol\gamma'(\xi)\|\,|b-a|$ のみである（定理 5.19、:citet:`Rudin.1976`）。幸い、:numref:`sec_mdl-multivariable_calculus` におけるテイラー展開の上界では、この定理のベクトル版は一切必要ない。スカラー値の損失を線分上に制限することで、通常の 1 変数の定理がそのまま適用できるからである。

<!--en:9d2f8415-->
### 行列値写像のヤコビ行列

<!--en:792c9e0f-->
:numref:`sec_mdl-matrix-calculus` ではスカラー損失を微分するため、すべての勾配は都合よく対応する行列変数と同じ形状を持つ。しかし、行列値関数を行列入力に関して微分すると、完全な導関数は 4 つの添字を持つ配列になる。代数計算を線形で馴染みのある形に保つ標準的な方法は、列を縦に並べる作用素 $\operatorname{vec}$ で両方の行列を平坦化し、4 つの添字を持つテンソルを通常のヤコビ行列（Jacobian matrix）へと変換することである。この平坦化のもとでは、行列の全微分 $d\mathbf F=\mathbf A\,d\mathbf X\,\mathbf B$ は $d\operatorname{vec}\mathbf F=(\mathbf B^\top\otimes\mathbf A)\,d\operatorname{vec}\mathbf X$ となり、連鎖律は単にクロネッカー積（Kronecker product）どうしの乗算になる。:citet:`Giles.2008b` は、行列積、逆行列、行列式、および標準的な行列分解に対するフォワード・リバース両方向の微分規則を使いやすい一覧にまとめている。

<!--en:0a27cc09-->
## 自動微分

<!--en:e876fc98-->
### 最適なヤコビ行列の累積

<!--en:e9f3b586-->
:numref:`sec_mdl-matrix-calculus` で示したように、連鎖するヤコビ行列を掛ける順序を変えるだけで、計算コストはおよそ層の幅に相当する倍率だけ変化し得る。一直線に並んだ $K$ 個の因子の積であれば、最も低コストな括弧の付け方を求める問題は古典的な連鎖行列積問題であり、動的計画法（dynamic programming）によって $O(K^3)$ 時間で解ける。しかし、分岐やスキップ接続を持つ一般の計算グラフ（computation graph）では、消去順序の探索空間がはるかに広がる。各中間ノードは、そこへ入る辺とそこから出る辺の上の偏微分を任意の順序で掛け合わせることで消去できるからである。乗算回数が最小となる消去順序を求める問題は、実は NP 完全であることが知られている :cite:`Naumann.2008`。そのため、深層学習フレームワークは消去順序の大域的な最適化を試みず、フォワードモード、リバースモード、およびそれらの合成を提供して、モードの選択をユーザーに委ねている。:citet:`Griewank.Walther.2008` は、頂点・辺・面の消去を完全な一般性のもとで展開し、微分コンパイラで用いられるヒューリスティクスを概説している。

<!--en:a06624e4-->
### 再帰的チェックポイント保存

<!--en:094675ad-->
:numref:`sec_mdl-matrix-calculus-autodiff` で扱った 1 段階のチェックポイント保存方式は、ピーク時の活性化メモリ（activation memory）を保存状態数 $T/s+s$ へと削減し、順伝播（forward pass）を 1 回追加するコストと引き換えに $2\sqrt T$（最適なブロックサイズ $s$ のとき）まで抑える :cite:`Chen.Xu.Zhang.ea.2016`。しかし、非常に長い系列や展開された反復ソルバでは、$O(\sqrt T)$ のメモリでさえ大きすぎることがある。同じ考え方を再計算される各区間の内部に再帰的に適用すれば、メモリは一気に $O(\log T)$ まで下がり、必要な再計算は $O(T\log T)$ となる :cite:`Griewank.1992`。各ステップのコストが等しい場合、任意の固定されたメモリバジェットに対して Griewank の二項チェックポイント保存スケジュールが最適であることが証明されている :cite:`Griewank.Walther.2008`。実際には、チェックポイント保存のユーティリティは、指定されたブロックごとに 1 段階のチェックポイント保存を実装している。

<!--en:17f669c1-->
PyTorch は `torch.utils.checkpoint` を提供している。

<!--en:27148441-->
TensorFlow は `tf.recompute_grad` を提供している。

<!--en:9ff8db7f-->
JAX は `jax.checkpoint` を提供しており、これらのブロックを入れ子にすることで再帰的なスケジュールを手動で構築できる。

<!--en:ddd1af4b-->
### テイラーモード

<!--en:73310c4e-->
$k$ 階の方向微分（directional derivative）を求めるためにフォワードモードの自動微分（automatic differentiation）を $k$ 回入れ子にするのは無駄が多い。入れ子を 1 段増やすごとに状態が倍増し、1 つの値につき $2^k$ 個の成分を保持することになるからである。*テイラーモード*は、二重数（dual number）を打ち切りべき級数 $a_0+a_1\varepsilon+\cdots+a_k\varepsilon^k$（ただし $\varepsilon^{k+1}=0$）に置き換えることで、この指数的な肥大化を回避する。各基本演算は 1 回のパスで $k+1$ 個すべての係数を伝播する。具体的には、乗算は 2 つの係数系列を $O(k^2)$ 回の演算で畳み込み、$\exp$、$\log$、$\sin$ などの超越関数も同じ $O(k^2)$ のコストの漸化式から計算できる :cite:`Griewank.Walther.2008`。:numref:`sec_mdl-matrix-calculus-autodiff` にある、1 回の順伝播で二階導関数を計算する演習問題は $k=2$ の場合にあたる。

<!--en:1ffe76e6-->
`jax.experimental.jet` は任意の次数 $k$ に対するテイラーモードを実装している。

<!--en:d48bdc26-->
### フォワード勾配

<!--en:40b1e987-->
リバースモードの微分では中間値のテープを保存する必要がある一方、フォワードモードではテープが不要になる代わりに、入力パラメータごとに 1 回のパスが必要となる。:citet:`Baydin.Pearlmutter.Syme.ea.2022` は、ガウス分布に従うランダムな接ベクトル $\mathbf v\sim\mathcal N(\mathbf 0,\mathbf I)$ に沿って単一の順伝播を実行し、$(\nabla\ell^\top\mathbf v)\,\mathbf v$ を確率的勾配の推定値とすることで、この両方のコストを回避した。$\mathbb{E}[\mathbf v\mathbf v^\top]=\mathbf I$ であるためこの推定量は不偏であり、パラメータ数が $n$ のとき平均二乗誤差（mean squared error）は $(n+1)\|\nabla\ell\|^2$ となる。メモリ上のテープは完全に不要になるが、そのトレードオフは厳しい。典型的な推定誤差は真の勾配のノルムのおよそ $\sqrt n$ 倍に達し、小規模なモデルでは許容できても、現代のニューラルネットワークのスケールでは現実的でない大きさになる。

<!--en:3484711e-->
## 積分

<!--en:76ccb530-->
### ルベーグ積分

<!--en:3afd9d00-->
リーマン積分は定義域を区間に分割するが、この方針は激しく不連続な関数に対して破綻する。:numref:`sec_mdl-integral_calculus` で示したように、有理数の指示関数では代表点付きの和が決して一定の値に落ち着かず、さらにリーマン可積分な関数列の各点極限でさえリーマン可積分にとどまるとは限らない。Lebesgue の着眼点は、定義域ではなく値域をスライスすることにある。すなわち、各高さの水準を $f$ が上回る集合の大きさを測り、それらの測度を足し合わせるのである。有理数全体や極座標における半直線 $\theta=0$ のような測度ゼロの集合は積分値に一切寄与しないため、有理数の指示関数はすっきりと $0$ に積分される。:numref:`sec_mdl-integral_calculus` と :numref:`sec_mdl-expectation-gradients` のいずれにおいても、この理論の主力となる 3 つの定理、すなわちフビニ・トネリの定理、$n$ 次元の変数変換（change of variables）、および無限領域上の積分記号下の微分を証明なしで用いた。これら 3 つの定理はすべて、極限と積分をいつ交換できるかを定める単調収束定理（monotone convergence theorem）と優収束定理（dominated convergence theorem）に基づいている。$f_k\to f$ が各点収束であり、かつ $|f_k|\le g$ が単一の可積分関数 $g$ に対して成り立つならば、$\int f_k\to\int f$ となる :cite:`Folland.1999`。

<!--en:aa9afb07-->
### 準モンテカルロ法

<!--en:380436fc-->
:numref:`subsec_mdl-monte-carlo` では、誤差が $N^{-2/d}$ で減衰する二次のグリッドと、次元によらず誤差が $N^{-1/2}$ で減衰するモンテカルロ積分を比較した。しかし、純粋な無作為標本では偶発的な密集や空白の偏りが避けられない。*低食い違い量*（low-discrepancy）点集合（ソボル列、ハルトン列、格子則など）は、単位超立方体をはるかに均一に埋め尽くすように構成されており、有界変動（Hardy と Krause の意味で）の被積分関数に対して $O(N^{-1}(\log N)^d)$ の誤差を達成する。ただし、純粋に最悪ケースの保証として見ると、この上界は厳しい現実を突きつける。定数因子を無視しても、この上界が $N^{-1/2}$ を下回るのは、数千点を超えてから（$d=2$ のとき）、さらには $10^{15}$ 点を超えてから（$d=5$ のとき）に限られる。それでも実際には、準モンテカルロ法（quasi-Monte Carlo）は高次元においてさえ通常のモンテカルロ法をしばしば上回る。現実の被積分関数は、多くの場合、一度に少数の有効な座標にしか強く依存しないからである。決定論的な点集合からはそのままではエラーバーが得られないため、*ランダム化準モンテカルロ法*（randomized quasi-Monte Carlo）では点集合にランダムなシフトやスクランブルを施し、独立な反復試行から不偏な分散推定値を得られるようにする。:citet:`Owen.2013` の第 15〜17 章では、決定論的手法とランダム化手法の両方が解説されている。

<!--en:26a4906a-->
### スコア関数とパスワイズを超える勾配推定量

<!--en:aca1b809-->
:numref:`sec_mdl-expectation-gradients` では、スコア関数推定量とパスワイズ推定量を導出し、定数ベースラインを用いてスコア関数推定量の分散を低減した。より洗練された制御変量（control variate）は、モデル自体の構造を活用する。REBAR :cite:`Tucker.Mnih.Maddison.ea.2017` は、連続緩和（continuous relaxation）のパスワイズ勾配をスコア関数推定量から引き、その解析的な期待値を足し戻すことで、離散確率変数に対して推定量を不偏に保ったまま分散を下げる。Gumbel-softmax :cite:`Jang.Gu.Poole.2017,Maddison.Mnih.Teh.2017` やストレートスルー推定器（straight-through estimator）:cite:`Bengio.Leonard.Courville.2013` のような連続緩和そのものは、元の離散的な目的関数に対する統計的バイアスが生じる代わりに低い分散を実現する。第 3 のアプローチである*測度値微分*（measure-valued derivative）は、確率密度の導関数を 2 つの密度のスケール付きの差として表し、両方からサンプリングすることで勾配を推定する。:citet:`Mohamed.Rosca.Figurnov.ea.2020` はこれら 3 つの系統を単一の枠組みに統合し、それぞれが適用できる条件を明示している。

<!--en:5fd245f8-->
## 書籍、講義、チュートリアル

<!--en:f3e47eb0-->
**書籍**

<!--en:68dc8bfa-->
- Deisenroth、Faisal、Ong による『[Mathematics for Machine Learning](https://mml-book.github.io/)』は、第 5 章（「Vector Calculus」）で機械学習に向けた勾配、ヤコビ行列、および連鎖律を扱っている（:numref:`sec_mdl-single_variable_calculus`、:numref:`sec_mdl-multivariable_calculus`）。

<!--en:22f58cb0-->
- Gilbert Strang による『[Calculus](https://ocw.mit.edu/courses/res-18-001-calculus-fall-2023/)』（MIT OpenCourseWare）は無料で公開されている教科書であり、積分、微分積分学の基本定理、および重積分に関する各章が :numref:`sec_mdl-integral_calculus` の副読本となる。

<!--en:04840500-->
- Petersen と Pedersen による『[The Matrix Cookbook](https://archive.org/details/imm3274)』は行列微分の恒等式をまとめた公式集であり、全微分法（:numref:`sec_mdl-matrix-calculus`）で導出した結果を検算するのに役立つ。

<!--en:b4291163-->
- Griewank と Walther による『[Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717761)』および Naumann による『[The Art of Differentiating Computer Programs](https://epubs.siam.org/doi/book/10.1137/1.9781611972078)』は、フォワードモードとリバースモード、VJP と JVP、チェックポイント保存、および微分コンパイラに関する標準的な専門書である（:numref:`sec_mdl-matrix-calculus-autodiff`）。

<!--en:6eeb657c-->
**講義とチュートリアル**

<!--en:0fa7dd2a-->
- 3Blue1Brown による「[Essence of Calculus](https://www.3blue1brown.com/topics/calculus)」、および MIT OpenCourseWare の「[Single Variable Calculus (18.01SC)](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)」と「[Multivariable Calculus (18.02SC)](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/)」。

<!--en:fc48b701-->
- Edelman と Johnson による「[Matrix Calculus for Machine Learning and Beyond (18.S096)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/)」（MIT OCW）、Parr と Howard による「[The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)」、および Laue、Mitterreiter、Giesen による [matrixcalculus.org](https://www.matrixcalculus.org/)。

<!--en:8c4697b5-->
- Christopher Olah による「[Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)」、Baydin らによる「[Automatic Differentiation in Machine Learning: a Survey](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)」、JAX ドキュメント内の「[The Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)」と「[Autodidax](https://docs.jax.dev/en/latest/autodidax.html)」（:numref:`sec_mdl-matrix-calculus-autodiff`）、ならびにスコア関数（REINFORCE）推定量、パスワイズ再パラメータ化、制御変量を用いた分散低減（variance reduction）を扱う Mohamed らの「[Monte Carlo Gradient Estimation in Machine Learning](https://jmlr.org/papers/v21/19-346.html)」（:numref:`sec_mdl-expectation-gradients`）。

<!--en:9b6632a7-->
[ディスカッション](https://d2l.discourse.group/t/36858)
