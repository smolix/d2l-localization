<!--en:3f1a15d6-->
# 期待値とその勾配

<!--en:d285a975-->
これまでに最小化してきた損失のほとんどすべては、ミニバッチ内のデータ例、ランダムなドロップアウトマスク、あるいは入力に注入されたノイズに関する平均である。ランダム性の源がモデルパラメータに依存しない限り、そのような平均の微分は単純明快である。平均の導関数は導関数の平均に等しいため、ミニバッチ勾配降下法（minibatch gradient descent）は何ら特別な仕組みを必要とせず直接これを計算できる。難しさが生じるのは、パラメータが確率分布そのものを制御する場合である。変分オートエンコーダ（variational autoencoder、VAE）は、エンコーダが出力する平均と分散をもつ分布から潜在コード（latent code）を引き、方策勾配（policy gradient）エージェント（:numref:`sec_policygradient`）は、改善しようとしているまさにその方策（policy）から行動をサンプリングする。いずれの場合も、目的関数はパラメータとともに分布が変化するサンプルに関する期待値となるが、いったん引かれたサンプルは単なる数値にすぎない。そこには、自身を生成したパラメータの記憶は一切残っていないのである。

<!--en:0e647f2b-->
この種の最も単純な問題を考えてみよう。$X\sim\mathcal N(\theta,1)$ を平均 $\theta$、単位分散をもつ正規確率変数とし、次の期待二乗損失を考える。

<!--en:70bed0aa-->
この閉形式は、$X=\theta+Z$（ただし $Z$ は標準正規確率変数）とおけば直ちに導かれる。すなわち $\mathbb{E}[(\theta+Z)^2]=\theta^2+2\theta\,\mathbb{E}[Z]+\mathbb{E}[Z^2]$ であり、ここで対称性から $\mathbb{E}[Z]=0$、また $\mathbb{E}[Z^2]=1$ である（:numref:`subsec_mdl-integration-by-parts`）。$\theta=1.5$ における厳密な導関数の値は $\ell'(1.5)=3$ となる。しかし実際には、モデルは $X$ のサンプルを引くことはできても、$\ell(\theta)$ を閉形式で評価することはできない。では、サンプルのみからどのようにして $\ell'(\theta)$ を推定できるだろうか。

<!--en:b1d3ee51-->
微分積分は、根本的に異なる 2 つの答えを与えてくれる。どちらも期待値を積分として表し（:numref:`sec_mdl-integral_calculus`）、微分をその積分の内側へと通す。そこから先は道が分かれる。*スコア関数*（score-function）推定量はサンプルには手をつけず確率密度を微分するのに対し、*パスワイズ*（pathwise）推定量はノイズの密度には手をつけずサンプルを微分する。どちらの手法があらゆる設定で優位に立つわけでもなく、両者を比較すること（:numref:`subsec_mdl-stochastic-gradient-estimators`）が本節の中心的な目標である。それに先立ち、まずは期待値がどのように積分として書かれるかを簡単に振り返り、微分を積分の内側へ通す操作がいつ正当化されるのかを確かめておこう。このステップこそ、巧妙な誤りが紛れ込みやすい箇所だからである。

<!--en:9338789b-->
## 積分としての期待値

<!--en:3b603a20-->
何かを微分する前に、密度と期待値の記法を定めておこう。確率の章ではこれら両方の概念を詳しく展開するが（:numref:`sec_mdl-random_variables`）、ここでは前節の積分の言葉を用いて改めて述べる。領域 $\mathcal X$ 上の確率**密度**（probability density）とは、全体の積分が $1$ に等しい非負関数のことである。

<!--en:373ce6f4-->
確率変数 $X$（密度 $p$ をもつ）が区間 $[a, b]$ に入る確率は、その区間上における $p$ の下の面積 $P(a\le X\le b)=\int_a^b p(x)\,dx$ で与えられる（:numref:`sec_mdl-random_variables`）。密度は確率そのものではなく単位長さあたりの確率を測るため、その値は $1$ を超えうる。たとえば $[0,\tfrac12]$ 上の一様密度は、その台（support）上のいたるところで $2$ に等しい。本節の主な通し例は**標準正規**分布 $\mathcal N(0,1)$ であり、その密度 $p(z)=e^{-z^2/2}/\sqrt{2\pi}$ は :numref:`sec_mdl-integral_calculus` のガウス積分によって正規化されている。

<!--en:d0ccdf79-->
**期待値**（expectation）は、各結果にその確率密度で重み付けしたうえで、$X$ あるいは関数 $g(X)$ の重み付き平均を計算するものである。

<!--en:8826562a-->
期待値 $\mathbb{E}[g(X)]$ は、積分が絶対収束するとき、すなわち $\int_{\mathcal X}|g(x)|\,p(x)\,dx<\infty$ が成り立つときに確定した値をもつ。裾の重い分布はこの条件に反することがあり、たとえば演習問題 :exref:`ex_mdl-integral-calculus-heavy-tails` のコーシー密度は有限の平均をもたない。

<!--en:8228c5e8-->
深層学習では、期待値が閉形式の式をもつことはめったになく、しかも数千次元もの空間で定義されることが多いため、グリッド上の数値積分（numerical quadrature）は到底現実的ではない。そこで代わりに、独立なサンプル $x_1,\ldots,x_n$ を $p$ から引き、標本平均 $\frac1n\sum_{i=1}^n g(x_i)$ を作る。これは*モンテカルロ*推定値として知られる。1 サンプル $g(X)$ の標準偏差が $\sigma$ であれば、モンテカルロ平均の二乗平均平方根誤差（root-mean-square error、RMSE）は次元にかかわらず $\sigma/\sqrt n$ となる（:numref:`subsec_mdl-monte-carlo`）。本節で扱う 2 つの勾配推定量はいずれもこの形のモンテカルロ平均である。どちらも同じ真の勾配に対する不偏推定量であるため、違いは 1 サンプルあたりのばらつきにしか現れない。したがって、1 サンプルの標準偏差 $\sigma$ が両者を比較する自然な尺度となる。

<!--en:e20f990a-->
## 積分記号下の微分

<!--en:1477f52a-->
### 順序交換

<!--en:b5602bf7-->
例題の損失 $\ell(\theta)=\mathbb{E}[X^2]$ は、パラメータ $\theta$ に応じて変化する積分である。一般に、パラメータが積分に作用する箇所は 2 つある。微分積分学の基本定理（:numref:`sec_mdl-integral_calculus`）が扱う積分区間の端点を通じて作用する場合と、被積分関数そのものを通じて作用する場合である。

<!--en:56d252b2-->
積分区間の端点が固定されており、被積分関数のみが $\theta$ に依存するときは、微分を積分の内側へ通すことで微分できる。

<!--en:fdc5a885-->
幾何学的には、全体の面積の変化率は、各点における高さの変化率を積分したものに等しい。以下では、この微分と積分の入れ替えを*順序交換*（swap）と呼ぶ。2 つの勾配推定量はいずれもこれに依拠している。導関数は差分商の極限であり、積分は和の極限であり、有限個の和と差分は常に可換であるから、一見するとこの恒等式には何の問題もないように思える。しかし、2 つの極限の交換は決して自動的に成り立つものではなく、正則性条件（regularity condition）を必要とする。$x$ の有界区間上では、$\partial_\theta f$ が両変数に関して連続であれば十分である。非有界な領域上では、それに加えて $x$ の可積分関数であって、$|\partial_\theta f(x,\theta')|$ をすべての $\theta'$（$\theta$ の近傍）に対して上から抑えるものが存在することも必要となる。これは優収束定理（dominated convergence theorem）からの直接の帰結である（証明は省き :cite:`Folland.1999` による）。正規密度を用いた通し例 $f(x,\theta)=x^2p_\theta(x)$ では、ガウス分布の裾がいかなる多項式の増大よりも速く減衰するため、このような優関数による上界が常に存在する。

<!--en:a40c20bc-->
### ライプニッツ則

<!--en:e93e67ef-->
積分区間の端点も $\theta$ とともに動く場合、単純な順序交換だけでは不完全であり、誤った答えを与えてしまう。ライプニッツ則（Leibniz rule）は不足している境界項を補うものであり、内部の変化と移動する境界からの寄与を単純に足し合わせればよい（:numref:`fig_mdl-leibniz-boundary`）。

<!--en:fe9a187b-->
**命題（ライプニッツ則）：** *$f$ および $\partial_\theta f$ が、$\{(x,\theta):x\in[a(\theta),b(\theta)]\}$（ただし $\theta$ はある開区間を動く）を含む領域上で連続であり、かつ $a$ と $b$ が微分可能であるとする。このとき、次式が成り立つ。*

<!--en:169bec48-->
**証明：**$H(\alpha,\beta,\theta)=\int_\alpha^\beta f(x,\theta)\,dx$ と定義する。微分積分学の基本定理により $\partial_\beta H=f(\beta,\theta)$ および $\partial_\alpha H=-f(\alpha,\theta)$ となり、一方で有界区間 $[\alpha,\beta]$ 上での順序交換により $\partial_\theta H=\int_\alpha^\beta\partial_\theta f\,dx$ が得られる。これら 3 つの偏微分はすべて連続であるため、多変数の連鎖律（:numref:`sec_mdl-chain-rule-graphs`）が合成写像 $\theta\mapsto H(a(\theta),b(\theta),\theta)$ に適用でき、3 つの項がそれぞれ $a'(\theta)$、$b'(\theta)$、$1$ で重み付けされて結合される。$\blacksquare$

<!--en:c19a50dc-->
![一様分布の例である $f(x,\theta)=x/\theta$（区間 $[0,\theta]$ 上）に対するライプニッツ則。$\theta$ が $d\theta$ だけ増加すると、被積分関数の曲線は $f(\cdot,\theta+d\theta)$（破線。ここでは下がるため、斜線領域は失われる面積を表す）へと移動する一方、右側の境界は外側へ動いて高さ $f(b,\theta)$、幅 $b'\,d\theta$ の細い帯を掃き込む。](../img/mdl-cal-leibniz-boundary.svg)

<!--en:247352ce-->
境界項を無視することは些細な誤差にとどまらず、導関数の符号を反転させてしまうことさえある。$X$ が $[0,\theta]$ 上の一様分布に従い、その区間上で密度 $1/\theta$ をもつとしよう。その平均は $\mathbb{E}[X]=\int_0^\theta\frac x\theta\,dx=\frac\theta2$ であり、導関数は $+\tfrac12$ である。被積分関数だけを微分すると $\int_0^\theta-\frac{x}{\theta^2}\,dx=-\tfrac12$ となり、符号が逆になってしまう。ここに上端の境界からの寄与 $f(\theta,\theta)\,b'(\theta)=\frac\theta\theta\cdot1=1$ を加えることで、$-\tfrac12+1=+\tfrac12$ と真の導関数が復元される。分布の台がパラメータに依存する場合には、単純な順序交換だけでは正当化されない（演習問題 :exref:`ex_mdl-integral-calculus-moving-support` では $\mathbb{E}[X^2]$ を計算する）。

<!--en:32f61b8d-->
### Log-Integral-Exp

<!--en:0c269f13-->
順序交換からは、log-sum-exp 関数の連続版も導かれる。連鎖律と :eqref:`eq_mdl-diff-under-integral` を $\log\int e^{\theta x}p(x)\,dx$ に適用すると、次式が得られる。

<!--en:e600e67b-->
右辺は、$x$ の平均を $e^{\theta x}p(x)$ に比例する指数的に*傾斜させた*（tilted）密度のもとでとったものである。標準正規密度の場合、$\theta z-\tfrac12z^2=\tfrac12\theta^2-\tfrac12(z-\theta)^2$ と平方完成することで、傾斜密度が $\mathcal N(\theta,1)$ であり、かつ $\log\mathbb{E}[e^{\theta Z}]=\theta^2/2$ となることがわかる。実際、その導関数 $\theta$ は傾斜平均に一致している。log-sum-exp の勾配がソフトマックスベクトルであるのとちょうど同じように、log-integral-exp の勾配は期待値となる。この恒等式には、対数分配関数（log-partition function）の導関数 $A'(\eta)$ として :numref:`sec_mdl-exponential-family` で再び出会うことになる。

<!--en:1b3e03ee-->
## スコア関数勾配とパスワイズ勾配

<!--en:94ff0dca-->
期待損失に順序交換を適用すると、積分 $\int g(x)\,\partial_\theta p_\theta(x)\,dx$ が得られる。残念ながら、この積分をモンテカルロサンプリングで直接推定することはできない。$\partial_\theta p_\theta$ は確率密度ではないため（正と負の両方の値をとり、積分するとゼロになる）、そこからサンプルを引くことができないからである。しかし 2 つの単純な恒等式を使えば、この積分をサンプルから推定可能な期待値の形へと書き戻せる。

<!--en:19b0f589-->
### スコア関数推定量

<!--en:1dc30004-->
$p_{\boldsymbol\theta}$ を $\boldsymbol\theta$ によってパラメータ化された密度とする。連鎖律を $\log p_{\boldsymbol\theta}(x)$ に適用すると、次の**対数微分の恒等式**（log-derivative identity）が得られる。

<!--en:56acc4b3-->
言葉で言えば、密度の勾配は、密度そのものにその対数の勾配（**スコア**と呼ばれるベクトル $\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}$）を掛けたものに等しい。順序交換が正当化され、かつ $p_{\boldsymbol\theta}$ の台が $\boldsymbol\theta$ とともに動かないときには、スコアの期待値はゼロになる。すなわち $\mathbb{E}[\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]=\int\nabla_{\boldsymbol\theta}p_{\boldsymbol\theta}\,dx=\nabla_{\boldsymbol\theta}\int p_{\boldsymbol\theta}\,dx=\nabla_{\boldsymbol\theta}1=\mathbf 0$ である。$[0,\theta]$ 上の一様密度を見れば、なぜ台が固定されていることが不可欠なのかがわかる。そのスコアは $\partial_\theta\log(1/\theta)=-1/\theta$ であり、期待値は $-1/\theta\ne0$ となる。というのも、$\int_0^\theta p_\theta(x)\,dx=1$ を微分すると境界項 $p_\theta(\theta)=1/\theta$ も現れるからである。なお、スコアの平均がゼロになる性質は、フィッシャー情報量（Fisher information、:numref:`sec_mdl-maximum_likelihood`）の出発点でもある。

<!--en:07bc2d7f-->
:eqref:`eq_mdl-log-derivative` に関数 $g(x)$（$\boldsymbol\theta$ に依存しない）を掛けて積分すると、次の**スコア関数恒等式**（score-function identity）が得られる。

<!--en:4a3fd967-->
この恒等式により、期待損失の勾配は再び $p_{\boldsymbol\theta}$ のもとでの期待値へと戻る。各サンプルの損失 $g(x)$ が、$\boldsymbol\theta$ がそのサンプルの対数確率（log-probability）をどれだけ強く押し上げるかによって重み付けされるのである。これをモンテカルロ法で推定するには、サンプル $x_i\sim p_{\boldsymbol\theta}$ を引き、$g(x_i)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(x_i)$ の平均をとればよい。強化学習（reinforcement learning、RL）において、これは方策勾配法（:numref:`sec_policygradient`）の REINFORCE 推定量である :cite:`Williams.1992`。（もし $g$ も $\boldsymbol\theta$ に直接依存する場合には、積の微分法則により単に $\mathbb{E}[\nabla_{\boldsymbol\theta}g(X,\boldsymbol\theta)]$ の項が加わる。）

<!--en:20003492-->
スコアの平均がゼロであるため、定数の**ベースライン** $b$（$x$ に依存しない）を任意に選んで $g(X)$ から引いても、推定量に統計的バイアスを生じさせることはない（:numref:`sec_baselines`）。すなわち $\mathbb{E}[(g(X)-b)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]=\mathbb{E}[g(X)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]$ が成り立つ。$b$ を適切に選べば、推定値の分散を大幅に減らすことができる。

<!--en:973453ba-->
### パスワイズ推定量

<!--en:bccdef04-->
2 つ目の恒等式は、サンプルに $\boldsymbol\theta$ の記憶を取り戻させるものである。多くの場合、$p_{\boldsymbol\theta}$ からのサンプルは、決定論的な変換 $x=T_{\boldsymbol\theta}(\mathbf z)$ を補助的なノイズ変数 $\mathbf z\sim q$（その分布は $\boldsymbol\theta$ に依存しない）へ施すことで生成できる。たとえば正規確率変数 $\mathcal N(\mu,\sigma^2)$ は、単に $\mu+\sigma z$（ただし $z\sim\mathcal N(0,1)$）と表せる。期待値を $\mathbf z$ に関する積分として書き直せば、密度から $\boldsymbol\theta$ が完全に取り除かれるため、順序交換によって次の**パスワイズ恒等式**（pathwise identity）が得られる。

<!--en:aeefe2e5-->
この推定量は密度を微分する代わりに、サンプルそのものを通して微分する。自動微分が $\boldsymbol\theta$ から $T_{\boldsymbol\theta}$ を経て $g$ に至る経路をたどるのである。これは*再パラメータ化トリック*（reparameterization trick）として知られ、変分オートエンコーダがガウス潜在変数（latent variable）を通して逆伝播する仕組みである :cite:`Kingma.Welling.2014,rezende2014stochastic`。推定量自体の歴史はさらに古く、シミュレーションの分野ではパスワイズ微分や微小摂動微分（infinitesimal-perturbation derivative）の名で長く使われてきた :cite:`Mohamed.Rosca.Figurnov.ea.2020`。正当に適用するには、$T_{\boldsymbol\theta}$ が $\boldsymbol\theta$ に関して微分可能であること、$g$ が経路に沿って微分可能であること（Lipschitz 連続な $g$ であれば、$X$ が連続密度をもつ限り十分である）、そして順序交換のための可積分な優関数による上界が存在することが必要となる。なお、たとえ $X$ の台が $\boldsymbol\theta$ とともに動いたとしても、積分は $\mathbf z$（その密度 $q$ は固定されたままである）に関してとられるため、境界項は一切現れないことに注目しよう。

<!--en:a7c17d2f-->
$g$ に対する微分可能性の要件は、無視してよい単なる技術的細目ではない。ステップ関数 $g(x)=\mathbf{1}(x>0)$（すなわち $1$ を正の $x$ に対してとり、それ以外では $0$ となる関数）を $X=\theta+z$ のもとで考えてみよう。真の期待値 $\mathbb{E}[g(X)]=P(z>-\theta)$ の導関数は $\varphi(\theta)=e^{-\theta^2/2}/\sqrt{2\pi}$ であり、その値は約 $0.13$（$\theta=1.5$ のとき）である。ところがパスワイズ推定量のサンプル $\partial_\theta\mathbf{1}(\theta+z>0)$ は $0$ を値としてとり、それがすべての $z$（確率ゼロの事象である唯一の点 $z=-\theta$ を除く）で成り立つため、パスワイズ推定量は $0$ を確率 $1$ で返してしまう。対照的に、スコア関数推定量のサンプル $\mathbf{1}(x>0)\,z$ は期待値 $\int_{-\theta}^\infty z\,\varphi(z)\,dz=\bigl[-\varphi(z)\bigr]_{-\theta}^\infty=\varphi(\theta)$ をもち、厳密な導関数を復元する。スコア関数推定量は $g$ やサンプリングのステップを通して微分することが一切ないため、不連続な損失にも、ベルヌーイ分布やカテゴリ選択のような離散分布（そこでは $\boldsymbol\theta$ からサンプルへ至る微分可能な経路が存在しない）にもまったく同様に適用できる。

<!--en:083e36f9-->
### 2 つの推定量の比較

<!--en:1ad6927b-->
これで同じ導関数に対する 2 つの不偏推定量（unbiased estimator）が出そろった。通し例である $\ell(\theta)=\mathbb{E}[X^2]$（ただし $X=\theta+z$）は十分に単純なため、両者の分散を手計算で求めることができる。パスワイズ推定量では、1 サンプルあたりの寄与は $\partial_\theta(\theta+z)^2 = 2X$ であり、その分散は $\textrm{Var}(2X)=4$ となる。スコア関数推定量では、正規位置族（normal location family）のスコアが $\partial_\theta\log p_\theta(X)=X-\theta=z$ であるから、1 サンプルあたりの寄与は $X^2z$、あるいは $(X^2-b)z$（ベースライン $b$ を用いる場合）となる。$X^2z$ の分散を求めるには、標準正規分布のモーメント $\mathbb{E}[z^2]=1$、$\mathbb{E}[z^4]=3$、$\mathbb{E}[z^6]=15$ を用いる。これらはいずれもスタインの恒等式 $\mathbb{E}[g'(Z)]=\mathbb{E}[Z\,g(Z)]$ から導かれる（:numref:`subsec_mdl-integration-by-parts`）。対称性により、$z$ の奇数乗の項は二次モーメント（second moment）においてすべて消える。

<!--en:e07e4c80-->
ここから平均の二乗 $(2\theta)^2$ を引くと、分散 $\theta^4+14\theta^2+15$ が残る。分散を最小化する定数ベースラインは $b^\star=\theta^2+3$ となることがわかり、これにより分散は $8\theta^2+6$ まで下がる（演習問題 :exref:`ex_mdl-integral-calculus-quartic-estimators`）。$\theta=1.5$ における 3 つの推定量の 1 サンプル標準偏差は、パスワイズ推定量が $2$、素のスコア関数推定量が $7.18$、最適ベースライン付きスコア関数推定量が $4.90$ となる。ベースラインは目に見えて有効ではあるものの、この損失に関しては差を埋めきることはできない。$8\theta^2+6>4$ がすべての $\theta$ に対して成り立つからである。

<!--en:ebeef0a1-->
これら 3 つの閉形式による予測を、1 回のモンテカルロシミュレーションで確かめてみよう。パスワイズ微分は単に $2x_i$ であるため、ここでは自動微分すら必要ない。ここまでの式変形が正しければ、3 つの標本標準偏差はそれぞれ $2$、$7.18$、$4.90$ の近くに収まり、各標本平均は標準誤差（standard error）$\mathrm{sd}/\sqrt n$ の数倍以内で真の導関数の値 $3$ に一致するはずである。

<!--en:e2d6216f-->
典型的な実行結果では、3 つの平均はいずれも標準誤差の 2〜3 倍以内で $3$ の近くに収まり、経験標準偏差も閉形式の予測と数パーセント以内で一致する。一定のサンプル数の予算のもとでは、パスワイズ推定値は素のスコア関数推定値より約 $3.6$ 倍精度が高い。言い換えれば、素のスコア関数推定量が同じ精度に達するには約 13 倍のサンプル数を要する。

<!--en:725435cb-->
しかし、パスワイズ推定量のほうが常に分散が小さいと結論づけるのは誤りである。振動する損失 $g(x)=\sin(\omega x)$ を考えてみよう。このとき期待値の真の導関数は $\omega\cos(\omega\theta)e^{-\omega^2/2}$ となる。パスワイズ推定量のサンプル $\omega\cos(\omega X)$ は周波数 $\omega$ に比例して大きくなり、$\omega/\sqrt2$ に近い標準偏差をもつのに対し、スコア関数推定量のサンプル $\sin(\omega X)\,z$ の標準偏差は高々 $\sqrt{\mathbb{E}[z^2]}=1$ である（$|\sin|\le1$ だからである）。被積分関数が激しく振動する場合、関数値そのものは有界にとどまっていても、その導関数は巨大になりうる。すでに引いたサンプルを再利用し、$\omega=1$ と $\omega=3$ において 2 つの推定量を比較してみよう。

<!--en:c418b70d-->
$\omega=1$ の時点ですでにスコア関数推定量のほうがばらつきが小さく（約 $0.5$ 対 $0.7$）、$\omega=3$ では 3 倍の差をつけて優位に立つ（約 $0.7$ 対 $2.1$）。どちらの標本平均も、標準誤差の数倍以内で厳密な導関数の値と一致している。

<!--en:c9dee437-->
どちらの推定量も、あらゆる問題で優位に立つわけではない。パスワイズ推定量は $g$ を微分するため、$g'$ が大きいときに分散が増大する。一方、スコア関数推定量は $g$ にスコアを掛けるため、$g$ そのものが大きいときに分散が増大する。いずれの場合も、所定の精度を得るためにどれだけのサンプル数が必要かは、1 サンプルの標準偏差 $\sigma$ によって決まる。この $\sigma$ はサンプルを引く元の分布にも依存するため、$\sigma$ を小さくするような提案分布（proposal distribution）を選ぶという発想から、直ちに重要度サンプリング（importance sampling）が導かれる。これについては :numref:`sec_mdl-bayesian-computation` で取り上げる。

<!--en:30ac03ff-->
## まとめ

<!--en:d111d520-->
モデルのパラメータがサンプルの分布を支配するとき、期待損失の勾配をモンテカルロサンプリングで推定するには、まずその勾配を期待値の形に書き直さなければならない。そのための 2 つの方法はいずれも微分を積分の内側へと通すが、このステップには可積分な優関数による上界が必要であり、さらに積分区間の端点がパラメータとともに動く場合にはライプニッツ則の境界項も必要となる。スコア関数恒等式は対数密度を微分するものであり、任意の損失 $g$ や離散確率変数に対して機能するが、サンプルの値が $g$ の大きさに比例するため、適切なベースラインがなければノイズが大きくなりやすい。一方、パスワイズ恒等式は再パラメータ化されたサンプルを通して微分するものであり、微分可能な $g$ を必要とするが、常にではないものの多くの場合ずっと低い分散を達成する。どちらの推定量も二乗平均平方根誤差 $\sigma/\sqrt n$ をもつため（:numref:`subsec_mdl-monte-carlo`）、与えられた問題に対してどちらが望ましいかは、1 サンプルの標準偏差 $\sigma$ によって決まる。

<!--en:2294243a-->
## 演習問題

<!--en:dff04ff2-->
1. **動く台：**$X$ が区間 $[0,\theta]$（ただし $\theta>0$）上の一様分布に従うとし、$\ell(\theta)=\mathbb{E}[X^2]=\int_0^\theta \tfrac{x^2}{\theta}\,dx=\tfrac{\theta^2}{3}$ かつ $\ell'(\theta)=\tfrac{2\theta}{3}$ とする。
    1. $\ell$ をライプニッツ則 :eqref:`eq_mdl-leibniz` を用いて微分せよ。その結果のどの部分が単純な順序交換 :eqref:`eq_mdl-diff-under-integral` から得られ、どの部分が欠落するか。
    1. 台の上では $\log p_\theta(x)=-\log\theta$ である。スコア関数推定量 $X^2\,\partial_\theta\log p_\theta(X)$（:eqref:`eq_mdl-score-function-gradient`）の期待値を計算せよ。続いて $g$ を $[0,\infty)$ 上の連続関数とする。$g(X)\,\partial_\theta\log p_\theta(X)$ の平均が、単純な順序交換 :eqref:`eq_mdl-diff-under-integral` を $\mathbb{E}[g(X)]=\int_0^\theta g(x)\,p_\theta(x)\,dx$ に適用したものと等しいことを示せ。これが $\frac{d}{d\theta}\mathbb{E}[g(X)]$ の推定値としてもつ統計的バイアスは、$g(\theta)$ と $\theta$ を用いてどのように表されるか。
    1. $X=\theta U$（ただし $U$ は $[0,1]$ 上の一様分布に従う）と表し、:eqref:`eq_mdl-pathwise-gradient` のパスワイズ推定量の期待値を計算せよ。$X$ の台が $\theta$ とともに動くにもかかわらず、境界項が不要であるのはなぜか。
   :exlabel:`ex_mdl-integral-calculus-moving-support`

<!--en:65cfcb7c-->
1. [code] **四次関数に対する推定量の分散：** ● $X\sim\mathcal N(\theta,1)$ とし、$X=\theta+z$ と表し、$s(X)=\partial_\theta\log p_\theta(X)=z$ をスコアとする。
    1. $g(X)=X^4$ とおくと、$\frac{d}{d\theta}\mathbb{E}[X^4]=4\theta^3+12\theta$ となる。1 サンプルのパスワイズ推定量を :eqref:`eq_mdl-pathwise-gradient` から、1 サンプルのスコア関数推定量を :eqref:`eq_mdl-score-function-gradient` からそれぞれ導出せよ。演習問題 :exref:`ex_mdl-integral-calculus-gaussian-normalizers` (b) の恒等式を用いて、両者の平均がいずれも $4\theta^3+12\theta$ となることを示せ。
    1. 定数ベースライン $b$ に対し、$(g(X)-b)\,s(X)$ の分散が $b^\star=\mathbb{E}[g(X)s(X)^2]/\mathbb{E}[s(X)^2]$ によって最小化されることを示せ。$g(X)=X^2$ の場合に $b^\star=\theta^2+3$ となることを示し、さらに $\mathbb{E}[z^2]=1$、$\mathbb{E}[z^4]=3$、$\mathbb{E}[z^6]=15$ を用いて、本文で挙げた 3 つの分散 $4$、$\theta^4+14\theta^2+15$、$8\theta^2+6$ を導出せよ。
    1. $b^\star$ を $g(X)=X^4$ に対して求めよ。$10^6$ 個のサンプルを用い、$\theta\in\{0,1,2\}$ においてパスワイズ推定量、スコア関数推定量、ベースライン付きスコア関数推定量の標準偏差を推定せよ。どの $\theta$ においてベースラインによる改善効果が最も大きいか。また、パスワイズ推定量との差を埋めきることはあるか。$g'$ と $g$ の大きさから、これらの順序関係を説明せよ。
    1. $g(x)=\mathbf{1}(x>0)$ を $\theta=1.5$ において考える。本文では、パスワイズ推定量がほとんどすべてのサンプルで $0$ となる一方、スコア関数推定量 $\mathbf{1}(x>0)\,z$ は正しい平均 $\varphi(\theta)=0.1295$ をもつことを示した。シミュレーションによって両者を確かめよ。
   :exlabel:`ex_mdl-integral-calculus-quartic-estimators`

<!--en:912fec6b-->
1. [extended] **コイン投げを通じた勾配：** ●● $x$ が離散的な選択であるとき、$\theta$ からサンプルへ至る微分可能な経路は存在しないため、パスワイズ推定量は存在しない。Gumbel-softmax :cite:`Jang.Gu.Poole.2017,Maddison.Mnih.Teh.2017` のような連続緩和は微分可能な代理変数を構成し、ストレートスルー推定量 :cite:`Bengio.Leonard.Courville.2013` は順伝播でハードサンプルを用い、逆伝播でその代理変数を用いるが、いずれも元の目的関数に対して統計的バイアスをもつ。本問ではそのバイアスを測定する。:citet:`Tucker.Mnih.Maddison.ea.2017` はこれを簡単な例題で示している。1 つのベルヌーイ変数 $b\in\{0,1\}$ が $P(b=1)=\sigma(\theta)$（ただし $\theta$ はロジット）を満たし、目的関数が $\ell(\theta)=\mathbb{E}[(b-t)^2]$（ターゲット $t=0.45$）であるとする。$U$ を $(0,1)$ 上の一様確率変数とし、$\zeta=\log U-\log(1-U)$ とおき、温度 $\tau>0$ に対して $z=\sigma\bigl((\theta+\zeta)/\tau\bigr)$（Gumbel-softmax 緩和の 2 クラスの場合）とおく。
    1. $\ell(\theta)$ と $\ell'(\theta)$ を閉形式で求めよ。どの確率 $\sigma(\theta)$ が $\ell$ を最小化するか。また、有限の $\theta$ でそれは達成されるか。
    1. $b=\mathbf{1}(z>\tfrac12)$ が $P(b=1)=\sigma(\theta)$ をすべての $\tau$ に対して満たすことを示せ。
    1. $\ell'(\theta)$ に対する 1 サンプル推定量を次の 3 通り実装せよ。すなわち、REINFORCE :eqref:`eq_mdl-score-function-gradient`（ベースラインは $\ell(\theta)$）、緩和推定量 $\frac{d}{d\theta}(z-t)^2$、そして損失にはハードサンプルを用い導関数には緩和サンプルを用いるストレートスルー推定量 $2(b-t)\frac{dz}{d\theta}$ である。$\ell'(\theta)$ と比較して各推定量の平均と標準偏差を、$\theta\in\{-2,0,2\}$ および $\tau\in\{1,0.5,0.1\}$ に対して表にまとめよ。各推定量のバイアスは $\theta$ と $\tau$ にどのように依存するか。
    1. $\theta$ に対する確率的勾配降下法を、初期値 $\theta=0$、ステップサイズ $1$ で $5000$ ステップにわたり、REINFORCE、緩和推定量、ストレートスルー推定量のそれぞれ（各 $\tau$ について）で実行せよ。ただし REINFORCE のベースラインは過去の損失の指数移動平均に置き換えること。各実行に沿って $\ell(\theta)$ をプロットせよ。さらに緩和目的関数 $R_\tau(\theta)=\mathbb{E}[(z-t)^2]$ を $U$ 上の数値積分によって計算し、その最小点を緩和推定量の実行が落ち着く $\theta$ の値と比較せよ。$\tau$ を小さくすると、緩和推定量のバイアスと分散はどのようなトレードオフを示すか。
   :exlabel:`ex_mdl-integral-calculus-coin-flip`

<!--en:bc62d015-->
[ディスカッション](https://d2l.discourse.group/t/36878)

<!--en:bc41bda6-->
期待値とその勾配<br>**積分記号下の微分・ライプニッツ則・スコア関数（REINFORCE）・パスワイズ再パラメータ化・分散のトレードオフ**

<!--en:7bf6c776-->
::: {.slide title="パラメータがサンプリング分布を支配するとき"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:ddd904f1-->
- **パラメータ化されたサンプリングが現れる場面：**VAE は潜在コード $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}, \operatorname{diag}(\boldsymbol{\sigma}_{\boldsymbol{\theta}}^2))$ を引き、強化学習の方策は行動 $a \sim \pi_{\boldsymbol{\theta}}(a \mid s)$ をサンプリングする

<!--en:d5c84a15-->
- **計算グラフの分断という障害：**生のサンプル $x_i \sim p_{\boldsymbol{\theta}}$ は単なる数値であり、$\boldsymbol{\theta}$ へつながる計算グラフをもたない

<!--en:c2bd7d0a-->
- **通しベンチマーク：**$X \sim \mathcal{N}(\theta, 1)$ かつ $\ell(\theta) = \mathbb{E}[X^2] = \theta^2 + 1$ に対し、$\theta = 1.5$ での真の傾きは $\ell'(1.5) = 2\theta = 3$

<!--en:9d932056-->
::: {.slide title="積分としての期待値とモンテカルロ誤差"}

<!--en:c645f0a9-->
[基礎]{.kicker}

<!--en:57ff7198-->
- **積分としての期待値：**$\mathbb{E}_{X \sim p}[g(X)] = \int_{\mathcal{X}} g(x)p(x)\,dx$ を $\hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n g(x_i)$ で近似

<!--en:e33e26cd-->
- **次元に依存しない RMSE（$\sigma/\sqrt{n}$）：**二乗平均平方根誤差は $\sigma/\sqrt{n}$ のオーダーで次元 $d$ にかかわらず減少する

<!--en:b0ca989b-->
- **比較の尺度：**あらゆる不偏勾配推定量のサンプル効率は、1 サンプルの標準偏差 $\sigma$ によって決まる

<!--en:7ecd8cca-->
::: {.slide title="積分記号下の微分：順序交換"}

<!--en:cd018cc7-->
[極限の交換]{.kicker}

<!--en:cf13888c-->
- **内部の順序交換：**固定された台 $\mathcal{X}$ 上で $\frac{d}{d\theta}\int_{\mathcal{X}} f(x, \theta)\,dx = \int_{\mathcal{X}} \frac{\partial f}{\partial\theta}(x, \theta)\,dx$

<!--en:2a64fffd-->
- **正則性条件：**$\partial_\theta f$ の有界区間 $[a, b]$ 上での同時連続性に加え、可積分な優関数による上界 $|\partial_\theta f(x, \theta')| \le M(x)$ が非有界な $\mathbb{R}$ 上で存在すること（ガウス分布の裾はこれを満たす）

<!--en:a816b28d-->
::: {.slide title="Log-Integral-Exp と指数型分布族の傾斜"}

<!--en:995312ec-->
[連続版ソフトマックス]{.kicker}

<!--en:30a12eb6-->
- **対数分配関数の微分：**$\log\int e^{\theta x}p(x)\,dx$ を順序交換で微分すると、傾斜期待値 $\mathbb{E}_{X \sim \tilde{p}_\theta}[X]$（ただし $\tilde{p}_\theta(x) \propto e^{\theta x}p(x)$）が得られる

<!--en:e84b7451-->
- **標準正規分布での確認：**$Z \sim \mathcal{N}(0, 1)$ に対し、$\log\mathbb{E}[e^{\theta Z}] = \tfrac{1}{2}\theta^2$ の導関数は $\theta$ となり、$\tilde{p}_\theta = \mathcal{N}(\theta, 1)$ の平均と一致する

<!--en:d1c13cdb-->
::: {.slide title="ライプニッツ則：動く積分境界"}

<!--en:9f17d24d-->
[動く台]{.kicker}

<!--en:8e640428-->
- **内部の移動＋掃き動く境界：**$a(\theta)$ と $b(\theta)$ が $\theta$ とともに動くとき、被積分関数の曲線が縦方向に動く寄与（$\int_a^b \partial_\theta f\,dx$）と、端点が横方向に掃く寄与（$f(b,\theta)b' - f(a,\theta)a'$）の両方で全体の面積が変化する

<!--en:1ebebf63-->
- **多変数の連鎖律（§{sec_mdl-chain-rule-graphs}）による証明：**$H(\alpha, \beta, \theta) = \int_\alpha^\beta f(x, \theta)\,dx$ とおく。微分積分学の基本定理より $\partial_\beta H = f(\beta, \theta)$ かつ $\partial_\alpha H = -f(\alpha, \theta)$、有界区間での順序交換より $\partial_\theta H = \int_\alpha^\beta \partial_\theta f\,dx$ となり、$\frac{d}{d\theta}H(a(\theta), b(\theta), \theta)$ を評価すると 3 つの経路の積の和になる

<!--en:dc8fa72c-->
::: {.slide title="境界項を無視すると符号が反転する理由"}

<!--en:66bdb56c-->
[動く台の落とし穴]{.kicker}

<!--en:d003adf3-->
- **Uniform$(0, \theta)$ の平均：**$\mathbb{E}[X] = \int_0^\theta \frac{x}{\theta}\,dx = \frac{\theta}{2}$ の真の導関数は $+\frac{1}{2}$

<!--en:be336091-->
- **素朴な順序交換による符号の誤り：**内部だけを微分すると $\int_0^\theta -\frac{x}{\theta^2}\,dx = -\frac{1}{2}$ となる

<!--en:31bec4da-->
- **ライプニッツ境界項：**上端の移動による寄与 $f(\theta, \theta)b'(\theta) = 1 \cdot 1 = +1$ を加えることで $-\frac{1}{2} + 1 = +\frac{1}{2}$ を復元

<!--en:fd48abe7-->
::: {.slide title="密度の導関数から直接サンプリングできない理由"}

<!--en:c0dd27ee-->
[モンテカルロ法の障害]{.kicker}

<!--en:ba7c2eb4-->
- **$\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}(x)$ は密度ではない：**$\int g(x)\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}(x)\,dx$ において、$\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}$ は負の値もとり、積分すると $\mathbf{0}$ になる

<!--en:053f5925-->
- **期待値へ戻す 2 つの回避策：**
  1. *スコア関数：*$p_{\boldsymbol{\theta}}(x)$ を掛けて割る（$\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}$）
  2. *パスワイズ：*$X = T_{\boldsymbol{\theta}}(\mathbf{z})$（ただし $\mathbf{z} \sim q(\mathbf{z})$）へ変数変換する

<!--en:7e2b12f1-->
::: {.slide title="対数微分トリックとスコア関数"}

<!--en:72fb7940-->
[スコア関数恒等式]{.kicker}

<!--en:7a4b0e9f-->
- **REINFORCE 恒等式：**$\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}} = p_{\boldsymbol{\theta}}\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}$ より $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)]$

<!--en:9189fde2-->
- **ゼロ平均のスコア：**固定された台の上では $\mathbb{E}[\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)] = \mathbf{0}$（$\mathcal{N}(\theta, 1)$ では $s_\theta(X) = X - \theta = Z$）

<!--en:45f7cbbd-->
- **汎用性：**$g(x)$ を微分しないため、離散的な行動やブラックボックス報酬にも適用可能

<!--en:b6c73342-->
::: {.slide title="制御変量ベースラインによる分散低減"}

<!--en:f3d2f84c-->
[ベースライン減算]{.kicker}

<!--en:18cf286f-->
- **不偏なベースライン減算：**$\mathbb{E}[s_\theta(X)] = 0$ より、$g(X)$ を $g(X) - b$ に置き換えても任意の定数 $b$ に対して平均は変わらない

<!--en:25e62144-->
- **最適な定数ベースライン：**$b^\star = \frac{\mathbb{E}[g(X)s_\theta(X)^2]}{\mathbb{E}[s_\theta(X)^2]}$（$= \theta^2 + 3$、ただし $g(X) = X^2$ のとき）により、分散は $O(\theta^4)$ から $8\theta^2 + 6 = O(\theta^2)$ へ減少

<!--en:137c921f-->
::: {.slide title="パスワイズ（再パラメータ化）推定量"}

<!--en:73f03067-->
[サンプルを通じた微分]{.kicker}

<!--en:a38b4a2f-->
- **再パラメータ化（$X = T_{\boldsymbol{\theta}}(\mathbf{z})$、$\mathbf{z} \sim q$）：**パラメータに依存しないノイズ $\mathbf{z}$ へ変数変換すると $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{\mathbf{z} \sim q}[\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))]$

<!--en:cb4c414d-->
- **$g$ を通じた直接の連鎖律：**$\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\mathbf{z}$ に対し、逆伝播は $g(\boldsymbol{\mu} + \boldsymbol{\sigma}\odot\mathbf{z})$ を低分散で直接微分する

<!--en:43e28602-->
::: {.slide title="パスワイズ推定量が破綻するとき：ステップ損失と離散選択"}

<!--en:ff8d5f64-->
[微分可能性の境界]{.kicker}

<!--en:acea94ab-->
- **ステップ損失 $g(x) = \mathbf{1}(x > 0)$（$\theta = 1.5$）：**$\Phi(\theta)$ の真の導関数は $\varphi(1.5) \approx 0.1295$

<!--en:be87e9a8-->
- **パスワイズ推定量はほとんど確実に $0$ を返す：**ステップ関数 $\theta \mapsto \mathbf{1}(\theta + z > 0)$ は、確率ゼロの不連続点 $z = -\theta$ を除くいたるところで平坦

<!--en:9012c708-->
- **スコア関数は成功する：**$\mathbb{E}[\mathbf{1}(\theta + Z > 0)Z] = \varphi(\theta) \approx 0.1295$

<!--en:e220ec5d-->
::: {.slide title="E[X²] における分散の解析的比較"}

<!--en:e8a496ed-->
[厳密なモーメント計算]{.kicker}

<!--en:4f532f01-->
- **$\ell(\theta) = \mathbb{E}[X^2]$（$\theta = 1.5$）における厳密な分散：**
  - *パスワイズ（$2X$）：*$\operatorname{Var} = 4 \implies \sigma = 2.000$
  - *スコア＋ベースライン（$b^\star = \theta^2+3$）：*$\operatorname{Var} = 8\theta^2+6 \implies \sigma = 4.899$
  - *素の REINFORCE（$X^2 Z$）：*$\operatorname{Var} = \theta^4+14\theta^2+15 \implies \sigma = 7.181$（分散が $12.9\times$ 大きい）

<!--en:86fef3fb-->
::: {.slide title="モンテカルロ検証：パスワイズ対 REINFORCE"}

<!--en:a55ebaca-->
[実験ベンチマーク]{.kicker}

<!--en:026d42ef-->
$n = 200{,}000$ 個のサンプルを $\theta = 1.5$ で引くと、3 つの推定量がすべて $\ell'(1.5) = 3.0000$ に対して不偏であることが確認でき、理論上の厳密な標準偏差 $2.000$、$7.181$、$4.899$ が再現される。

<!--en:038e4738-->
::: {.slide title="スコア関数がパスワイズを上回るとき：激しく振動する損失"}

<!--en:a0923fa5-->
[高周波領域]{.kicker}

<!--en:bdacecc5-->
パスワイズ推定量は常にスコア関数推定量より分散が小さいか。**いいえ。**振動する損失 $g(x) = \sin(\omega x)$（ただし $\frac{d}{d\theta}\mathbb{E}[\sin(\omega X)] = \omega\cos(\omega\theta)e^{-\omega^2/2}$）を考える。

<!--en:164bbae1-->
- **パスワイズの分散は $|g'|^2 = O(\omega^2)$ で増大：**1 サンプルは $\omega\cos(\omega X)$ であり、標準偏差は周波数に比例して $\approx \omega/\sqrt{2}$ のように増大する

<!--en:6e3c9384-->
- **スコア関数の分散は $|g|^2 \le 1$ で有界：**1 サンプルは $\sin(\omega X)\,Z$ であり、$|\sin(\omega X)| \le 1$ より、その二次モーメントは高々 $\mathbb{E}[Z^2] = 1$ で $\omega$ がどれほど大きくなっても抑えられる

<!--en:2376a307-->
::: {.slide title="判断ガイド：勾配推定量の選び方"}

<!--en:5d233c39-->
[設計上のトレードオフ]{.kicker}

<!--en:0498280d-->
| 性質・要件 | スコア関数（REINFORCE） | パスワイズ（再パラメータ化） | 緩和・ストレートスルー（例：Gumbel-Softmax） |

<!--en:e17bf90c-->
| **基本式** | $g(X)\,\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)$ | $\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))$ $= \mathbf{J}_{T}^\top \nabla_x g$ | 滑らかな代理変数 $z_\tau = \sigma((\theta+\zeta)/\tau)$ を微分 |

<!--en:f363d4a4-->
| **微分可能な $g$ が必要か** | **不要**（ステップ関数やブラックボックス報酬にも有効） | **必要**（$g$ は微分可能 / Lipschitz 連続） | **必要**（代理損失が微分可能であること） |

<!--en:7d373e79-->
| **離散的な $X$ を扱えるか** | **可**（ベルヌーイ、カテゴリ行動） | **不可**（連続な経路 $T_{\boldsymbol{\theta}}(\mathbf{z})$ がない） | **可**（温度 $\tau$ で離散選択を近似） |

<!--en:3c989dd4-->
| **統計的バイアス** | **不偏**（固定された台の上で） | **不偏**（滑らかな $g$ に対して） | **バイアスあり**（$\tau \to 0$ でバイアスは縮小、分散は発散） |

<!--en:30aea8c3-->
| **1 サンプルの分散** | $\|g(X)\|^2$ に応じて増大（ベースライン $b$ が必要） | $\|\nabla_x g(X)\|^2$ に応じて増大（滑らかな $g$ では小さい） | 温度 $\tau$ で制御（バイアス・分散のトレードオフ） |

<!--en:62658bfe-->
::: {.slide title="理解度チェック：期待値の微分"}

<!--en:b6d9a92f-->
[復習と演習]{.kicker}

<!--en:02d1b095-->
ライプニッツ則、スコア関数推定量、パスワイズ推定量に関する次の 3 問に取り組んでみよう。

<!--en:14ca7ad6-->
1. **動く台によるバイアス：**$X \sim \mathrm{Uniform}(0, \theta)$（密度は $p_\theta(x) = 1/\theta$、台は $[0, \theta]$）では、スコアは $\partial_\theta \log(1/\theta) = -1/\theta$ となる。スコア関数推定量 $X^2 \partial_\theta \log p_\theta(X)$ の期待値は $\ell(\theta) = \mathbb{E}[X^2] = \theta^2/3$ に対してどうなるか。また、なぜ $\ell'(\theta) = 2\theta/3$ からずれるのか。

<!--en:fae67e1c-->
2. **Uniform$(0, \theta)$ の再パラメータ化：**$X \sim \mathrm{Uniform}(0, \theta)$ を $U \sim \mathrm{Uniform}(0, 1)$ を用いてどのように再パラメータ化すれば、パスワイズ推定量が境界項なしに厳密な不偏導関数 $\ell'(\theta) = 2\theta/3$ を与えるか。

<!--en:5999e77b-->
3. **高次元における分散の比較：**$g(\mathbf{x}) = \|\mathbf{x}\|_2^2$ かつ $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\theta}, \mathbf{I}_d)$ のとき、$g(\mathbf{X})$（REINFORCE に入る量）と $\nabla_{\mathbf{x}} g(\mathbf{X}) = 2\mathbf{X}$（パスワイズに入る量）の大きさはそれぞれ次元 $d$ に応じてどのようにスケールするか。

<!--en:b908ffa6-->
::: {.slide title="理解度チェック：期待値の微分（続き）"}

<!--en:b6d9a92f-2-->
[復習と演習]{.kicker}

<!--en:557b98b6-->
**解答：**

<!--en:9f92ab87-->
1. $\mathbb{E}[X^2(-1/\theta)] = -\frac{1}{\theta}\frac{\theta^2}{3} = -\frac{\theta}{3}$ となる。これは素朴な内部の順序交換 $\int_0^\theta x^2 \partial_\theta(1/\theta)\,dx$ に等しく、ライプニッツ則の上端境界項 $f(\theta, \theta)b'(\theta) = \theta^2(1/\theta)(1) = +\theta$ が欠落している（$-\frac{\theta}{3} + \theta = +\frac{2\theta}{3}$）

<!--en:06a74c08-->
2. 固定された台の上で $X = T_\theta(U) = \theta U$（$U \in [0, 1]$）とおけば、パスワイズ推定量は $\partial_\theta(\theta U)^2 = 2\theta U^2$ となり、その期待値は $2\theta\mathbb{E}[U^2] = 2\theta(\tfrac{1}{3}) = \frac{2\theta}{3}$ となる

<!--en:0bc27b7f-->
3. スカラー損失 $g(\mathbf{X}) = \sum_{j=1}^d X_j^2 = O(d)$ がスコアベクトルの全 $d$ 成分に掛かるため、REINFORCE の分散は $O(d^2)$（ベースライン付きでは $O(d)$）で増大するのに対し、座標 $k$ に関するパスワイズ微分は単に $2X_k$ であり、分散は $O(1)$ で次元 $d$ に依存しない

<!--en:f477f248-->
::: {.slide title="まとめ：期待値の勾配のツールキット"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:b61503d9-->
| 公式・推定量 | 数学的恒等式 | 条件と分散の振る舞い |

<!--en:ed8ed366-->
| **内部の順序交換** | $\frac{d}{d\theta}\int f(x,\theta)\,dx = \int \frac{\partial f}{\partial \theta}(x,\theta)\,dx$ | 固定された積分区間、連続な $\partial_\theta f$＋可積分な優関数による上界 |

<!--en:03bb6f7f-->
| **ライプニッツ境界則** | $\frac{d}{d\theta}\int_{a(\theta)}^{b(\theta)} f\,dx = \int_a^b \partial_\theta f\,dx + f(b,\theta)b' - f(a,\theta)a'$ | 積分区間や台が $\theta$ とともに動くときに必須 |

<!--en:165e5227-->
| **Log-Integral-Exp** | $\frac{d}{d\theta}\log\int e^{\theta x}p(x)\,dx = \mathbb{E}_{\tilde{p}_\theta}[X]$（$\tilde{p}_\theta \propto e^{\theta x}p$） | $\nabla\operatorname{LSE}(\mathbf{z}) = \operatorname{softmax}(\mathbf{z})$ の連続版 |

<!--en:e8962950-->
| **スコア関数（REINFORCE）** | $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{p_{\boldsymbol{\theta}}}\bigl[(g(X)-b)\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)\bigr]$ | 不連続な $g$ や離散的な $X$ に有効、分散は $\|g-b\|^2$ に比例 |

<!--en:0fd2db34-->
| **最適ベースライン $b^\star$** | $b^\star = \mathbb{E}[g(X)s_\theta(X)^2] / \mathbb{E}[s_\theta(X)^2]$ | 固定された台の上で $\mathbb{E}[\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)] = \mathbf{0}$ となるため不偏 |

<!--en:763f7804-->
| **パスワイズ（再パラメータ化）** | $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{\mathbf{z}\sim q}\bigl[\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))\bigr]$ | 滑らかな $g$ と連続な $X = T_{\boldsymbol{\theta}}(\mathbf{z})$ が必要、分散は $\|\nabla g\|^2$ に比例 |
