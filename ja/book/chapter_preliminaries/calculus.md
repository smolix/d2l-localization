<!--en:5ed86b2f-->
# 微分積分

<!--en:4220625a-->
古代ギリシャにおいて、Archimedes は辺の数を $n \to \infty$ と増やした正多角形を円に内接させることで円の面積を計算した（:numref:`fig_circle_area`）。三角形が細くなるにつれて、その高さは $r$ に近づき、底辺の合計は円周 $2\pi r$ に近づくため、面積は $\frac{1}{2} r (2\pi r) = \pi r^2$ となる。微小なステップの極限において曲がった量を単純な線形の断片で近似するという考え方は、積分法と*微分法*（differential calculus）の両方の根底にある。

<!--en:6972b253-->
![円に内接する辺の数 4、5、6、8 の正多角形。辺の数が増えるにつれて、多角形の面積は円の面積に近づく。](../img/polygon-circle.svg)

<!--en:4737c3fd-->
深層学習において、微分法は*最適化*（optimization）の原動力である。モデルの訓練とは、モデルが訓練例にどれだけ当てはまっていないかを測るスカラー値の損失関数を最小化するように、パラメータを調整することを意味する。数百万から数十億個のパラメータがある場合、試行錯誤で各パラメータを探ろうとすれば、更新ごとに数百万回の順伝播が必要になるだろう。その代わりに、*導関数*（derivative）は各パラメータに関する損失の瞬間的な変化率を与える。すべての偏微分を*勾配*（gradient）ベクトルにまとめれば最急上昇の方向がわかり、さらに*連鎖律*（chain rule）を用いれば入れ子になった層全体の勾配を 1 回の逆伝播で組み立てられる。

<!--en:5fdcc53c-->
本節では、すべての数値を手計算で確かめられるほど小さな線形回帰（linear regression）の例を一貫して用いながら、導関数、勾配、連鎖律を展開する。:numref:`sec_autograd` では、深層学習フレームワークが任意のプログラムに対してこれらの計算をどのように自動化するかを学ぶ。

<!--en:da5733be-->
## 導関数

<!--en:410f022f-->
### 極限としての導関数

<!--en:c07f7586-->
単一パラメータのモデルがすべてのデータ例に対して定数値 $w$ を予測するとし、ラベルが $1$、$1$、$0$ である 3 つの訓練例でこのモデルを評価する。二乗誤差（squared error）の和をとると、次のようになる。

<!--en:1f3f0fd4-->
$w = 1$ における損失は $L(1) = 1$ である。

<!--en:49143084-->
$L(w)$ が微小な摂動 $h \neq 0$ に対してどのように変化するかを見るため、差分商（difference quotient）$\frac{L(1+h) - L(1)}{h}$ を、$h$ が $10^{-1}$ から $10^{-5}$ まで小さくなるときについて計算してみよう。

<!--en:4dbac01b-->
$L(1+h) = 1 + 2h + 3h^2$ を展開すると、$\frac{L(1+h) - L(1)}{h} = 2 + 3h \to 2$（$h \to 0$ のとき）となることがわかる。厳密には、$f: \mathbb{R} \rightarrow \mathbb{R}$ に対し、関数 $f$ の点 $x$ における*導関数*は次の極限で定義される。

<!--en:8965ea4d-->
幾何学的には、差分商は点 $P = (x, f(x))$ と $Q = (x+h, f(x+h))$ を通る*割線*（secant line）の傾きである。$h \to 0$ につれて、この割線は回転し、点 $P$ における傾き $f'(x)$ の*接線*（tangent line）へと近づく（:numref:`fig_calc_secant_tangent`）。

<!--en:fc6c518d-->
![差分商 $(f(x+h)-f(x))/h$ は $P$ と $Q$ を通る割線の傾きである。$h \to 0$ につれて $Q$ は $P$ に向かって移動し、割線は $P$ における接線へと変化して、その傾きは $f'(x)$ となる。](../img/calculus-secant-tangent.svg)

<!--en:440f161e-->
すべての関数がいたるところで微分可能とは限らない。たとえば $|x|$ は $x = 0$ に角を持つ（左からの傾きは $-1$、右からの傾きは $+1$）。さらに悪いことに、離散的な分類の正解率はほとんどいたるところで平坦（$f'(x) = 0$）であり、予測が反転する箇所で不連続に跳び上がるため、勾配信号をまったく与えない。分類器を訓練する際に、交差エントロピー（cross-entropy）のような滑らかな代理損失（surrogate loss）を最小化するのはこのためである（:numref:`fig_calc_surrogate`）。

<!--en:0c1ae89c-->
![0-1 誤差（誤予測で $1$、正解で $0$）と滑らかな代理損失を、分類器が正しいときに正となるスコア $z$ に対してプロットしたもの。0-1 誤差は不連続点の両側で平坦であるため、その傾きからは何もわからない。一方、代理損失はいたるところで右下がりの傾きを持ち、パラメータをどちらへ動かすべきかを教えてくれる。](../img/calculus-surrogate-loss.svg)

<!--en:f6f622c0-->
### 微分の公式

<!--en:7edc5a78-->
$y = f(x)$ が与えられたとき、導関数を $f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x)$ と書く。よく使われる初等関数の導関数は次の通りである。

<!--en:d202ca95-->
また、微分可能な関数 $f, g$ の組み合わせは次の法則に従う。

<!--en:91fc6fb9-->
:numref:`fig_calc_product_rule` は、積の微分公式を $f(x) \times g(x)$ の長方形の面積によって幾何学的に示したものである（完全な証明は :numref:`sec_mdl-single_variable_calculus` を参照）。これらの公式を $L(w) = 3w^2 - 4w + 2$ に適用すると直ちに $L'(w) = 6w - 4$ が得られ、したがって $L'(1) = 2$ となる。

<!--en:05e15158-->
![長方形の面積として見た積の微分公式。$x$ が $h$ だけ増えると、辺 $f(x)$ と $g(x)$ はそれぞれ約 $f'(x) h$ と $g'(x) h$ だけ伸び、面積は 2 つの細い帯状の部分と面積 $f'(x) g'(x) h^2$ の角の部分だけ増えるが、この角は小さい $h$ では無視できる。](../img/calculus-product-rule.svg)

<!--en:aa98099f-->
### 一次線形近似と最小値

<!--en:ecac162d-->
$\frac{f(x+h) - f(x)}{h} \approx f'(x)$ の両辺に $h$ を掛けると、次の*一次線形近似*（first-order linear approximation）が得られる。

<!--en:973eeb45-->
$L'(1) = 2 > 0$ であるため、$w$ を減らせば損失は小さくなる。導関数と逆向きに $w \leftarrow w - \eta L'(w)$（ステップサイズ $\eta = 0.1$）と進むと、$w$ は $1$ から $1 - 0.1(2) = 0.8$ へ移動し、$L$ は $1$ から $L(0.8) = 0.72$ へと下がる。これは接線による予測値 $1 + 2(-0.2) = 0.60$ に近い（$0.12$ の差は無視した二次の曲率 $3h^2$ によるものである）。滑らかな極小値（local minimum）では必ず接線が水平（$f'(x) = 0$）になる。そこで $L'(w) = 6w - 4 = 0$ と置くと、唯一の大域的最小点（global minimizer）として $w^* = 2/3$、すなわち 3 つのラベル $1, 1, 0$ の標本平均（sample mean）が得られる。

<!--en:4658ee63-->
## 偏微分と勾配

<!--en:7133f178-->
### 偏微分

<!--en:74b944d0-->
次に、先ほどの 3 つのデータ例の問題にスカラー入力 $0$、$1$、$2$（対応するラベルはそれぞれ $1$、$1$、$0$）を与え、アフィン直線 $w_1 + w_2 \cdot \textrm{input}$ を当てはめてみよう。

<!--en:7dbf47ff-->
多変数関数 $y = f(x_1, \ldots, x_n)$ において、$x_i$ に関する*偏微分*（partial derivative）は、他のすべての変数を固定したまま座標 $i$ のみを微小変化させる。

<!--en:279daf91-->
$(w_1, w_2) = (1, 0)$ において、$w_2 = 0$ に固定すると $L(w_1, 0) = 3w_1^2 - 4w_1 + 2$ となるため $\frac{\partial L}{\partial w_1}(1, 0) = 2$ である。一方、$w_1 = 1$ に固定すると $L(1, w_2) = 5w_2^2 + 4w_2 + 1$ となり、その導関数は $10w_2 + 4$ であるため $\frac{\partial L}{\partial w_2}(1, 0) = 4$ となる（:numref:`fig_calc_partial_slices`）。

<!--en:991c3fdd-->
![直線モデルの損失 $L(w_1, w_2)$ を $\mathbf{w} = (1, 0)$（黒点）の周辺で示したもの。$w_2 = 0$ に固定すると定数モデルの損失であるオレンジ色の曲線が切り出され、$w_1 = 1$ に固定すると緑色の曲線が切り出される。黒点におけるそれぞれの接線（破線）の傾きは $\partial L / \partial w_1 = 2$ と $\partial L / \partial w_2 = 4$ である。](../img/calculus-partial-slices.svg)

<!--en:4716713d-->
行列・ベクトル形式（:numref:`sec_linear-algebra`）では、各データ例を $(1, \textrm{input}_i)$ という $\mathbf{X}$ の行として書くと、次のようになる。

<!--en:29578a16-->
これらの偏微分を数値的に確かめるには、`line_loss(w)` と有限差分（finite difference）の補助関数 `numerical_gradient(f, w, h=1e-6)` を定義すればよい。この補助関数は、`w` の座標を 1 つずつ $h = 10^{-6}$ だけ `float64` で摂動させる。

<!--en:c3351956-->
`numerical_gradient(line_loss, w)` を $\mathbf{w} = (1, 0)$ で評価すると、解析的に求めた偏微分 $(2, 4)$ が $O(h) \approx 10^{-6}$ の打ち切り誤差（truncation error）の範囲内で再現される。

<!--en:f69ccbce-->
なお、`numerical_gradient` は $n$ 回余分に $f$ を評価する必要があり（パラメータ数が $n$ の場合）、$n = 10^8$ ともなると非現実的になる。

<!--en:72af8e7f-->
### 有限差分における数値精度の限界

<!--en:94c23282-->
また、有限差分では $h$ が小さすぎるときにも浮動小数点の桁落ちという問題が生じる。有限精度のもとでは、計算機イプシロンが $\epsilon \approx 2.2 \times 10^{-16}$（`float64`）や $\epsilon \approx 1.2 \times 10^{-7}$（`float32`）であるため、ほぼ等しい数値の差 $f(x+h) - f(x)$ をとると上位の有効桁が失われる*桁落ち*（catastrophic cancellation）が起き、$\sim \epsilon$ 程度の丸め残差が $h$ で割られることになる。したがって、*打ち切り誤差*（$\propto h$）と*丸め誤差*（rounding error、$\propto \epsilon / h$）が競合する。

<!--en:950c97dc-->
64 ビットの NumPy 演算を、深層学習でデフォルトとして使われる 32 ビット浮動小数点形式（floating-point format）と比較するため、ステップのグリッド `h` を `float32` のフレームワークテンソル `h32` にキャストする。

<!--en:17a8b425-->
ここで、`quotient_error(loss, 1, 2, ...)` を $w = 1$（厳密な導関数の値は $L'(1) = 2$）において全 $61$ 個のステップサイズにわたり `float64`（`h`）と `float32`（`h32`）の両方で評価し、打ち切り誤差の直線 $3h$ と `float32` の丸め誤差の漸近線 $\epsilon / h$ を重ねて描画する。

<!--en:0937cbaa-->
$3h \approx \epsilon / h$ を釣り合わせると最適なステップサイズは $h^* \approx \sqrt{\epsilon / 3}$ となり、その形式が持つ有効桁数の半分が失われる（最小誤差は $\sim 10^{-8}$ に達するのが $h \approx 10^{-8}$ の `float64` であり、$\sim 10^{-4}$ に達するのが $h \approx 10^{-4}$ の `float32` である）。連鎖律を用いた解析的な微分は、$O(n)$ の計算コストと $\sqrt{\epsilon}$ への精度低下の両方を回避できるため、`float64` による有限差分は単体テストでの勾配チェックの用途に残されている。

<!--en:4f6e3e89-->
### 勾配と最急降下

<!--en:14b6cb3a-->
全 $n$ 個の偏微分（関数 $f: \mathbb{R}^n \to \mathbb{R}$ の各変数に関するもの）を連結すると、*勾配*ベクトル $\nabla_{\mathbf{x}} f(\mathbf{x}) = (\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n})^\top$（$\mathbf{x}$ と同じ形状）が得られる。これにより、:eqref:`eq_calculus-linear-1d` は同時摂動 $\mathbf{h} \in \mathbb{R}^n$ へと一般化される。

<!--en:5c5f70c8-->
ステップ $\mathbf{h} = t\mathbf{d}$（長さ $t > 0$、単位ベクトル $\mathbf{d}$ は $\|\mathbf{d}\|_2 = 1$）に対して、:eqref:`eq_dot_cos` より $f(\mathbf{x} + t\mathbf{d}) - f(\mathbf{x}) \approx t\,\|\nabla f(\mathbf{x})\|\cos\theta$ となる。したがって、$\nabla f(\mathbf{x})$ は*最急上昇*（steepest ascent）の方向（$\theta = 0^\circ$）を指し、$-\nabla f(\mathbf{x})$ は*最急降下*（steepest descent）の方向（$\theta = 180^\circ$）を指す。また、$\nabla f(\mathbf{x})$ は点 $\mathbf{x}$ を通る等位集合（level set）と直交する（$\theta = 90^\circ$、:numref:`fig_calc_gradient_field`）。

<!--en:b4eb400a-->
![直線モデルの損失 $L(w_1, w_2)$ の等位集合。勾配（赤）はそれぞれの始点を通る等位集合に直交し、上り方向を指す。$\mathbf{w} = (1, 0)$（黒点）から出発すると、$\eta = 0.1$ の勾配降下法（オレンジ）は最小点を直接指していない $-\nabla L$ に沿って進み始め、ジグザグに最小点へと向かう。](../img/calculus-gradient-field.svg)

<!--en:5b0e695d-->
これが*勾配降下法*（gradient descent）$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$ の動機となる。$\mathbf{w} = (1, 0)$ から学習率（learning rate）$\eta = 0.1$ で 1 ステップ進めると、$(1, 0) - 0.1(2, 4) = (0.8, -0.4)$ へ移動し、損失は $1.0$ から $0.4$ に減少する。なぜ :numref:`fig_calc_gradient_field` の最急降下は谷をジグザグに進み、大域的最小値 $\mathbf{w}^* = (7/6, -1/2)$ をまっすぐ目指さないのだろうか。それは、「最急」がユークリッド距離でのステップ長 $\|\mathbf{h}\|_2$ の単位あたりで測られ、各座標の単位に依存するからである。1 つの特徴量の軸を引き伸ばすと等高線が潰れ、$-\nabla L$ が最小点から逸れる方向に傾く。実務において特徴量の標準化（:numref:`sec_pandas`）や曲率の前処理（preconditioning、:numref:`chap_optimization`）が極めて重要になるのはこのためである。

<!--en:dba3e234-->
### 行列微分の恒等式

<!--en:48b96484-->
ベクトル勾配に関する 3 つの恒等式（:numref:`sec_mdl-matrix-calculus`）を用いると、座標を書き下すことなくベクトル表現を微分できる。

<!--en:43c872a0-->
$L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = \mathbf{w}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{w} - 2(\mathbf{X}^\top\mathbf{y})^\top\mathbf{w} + \mathbf{y}^\top\mathbf{y}$ を展開してこれらの公式を適用すると、次の式が得られる。

<!--en:67a5b0f2-->
言葉で言い換えると、勾配の各成分は、$\mathbf{X}$ の対応する特徴量の列と残差（residual）ベクトル $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ とのドット積（dot product）の 2 倍である。最適解 $\nabla L(\mathbf{w}^*) = \mathbf{0}$ においては、予測誤差はすべての特徴量と直交する。

<!--en:ce6ecec8-->
## 連鎖律

<!--en:6f8e0b0d-->
深層ネットワークは、入れ子になった多数の層 $y = f_L(f_{L-1}(\cdots f_1(\mathbf{x})\cdots))$ を合成したものである。合成関数を展開せずに微分するには*連鎖律*を用いる。

<!--en:ec5a11eb-->
スカラー関数 $z = f(u)$ と $u = g(x)$ では、変化率は $\frac{dz}{dx} = \frac{dz}{du}\frac{du}{dx}$ のように掛け合わされる。より一般に、$z = f(u_1, \ldots, u_m)$ であり各変数が $u_j = g_j(x_1, \ldots, x_n)$ で与えられるとき、変化率は各経路 $x_i \to u_j \to z$ に沿って掛け合わされ、すべての並列経路にわたって足し合わされる。

<!--en:04b6fd58-->
ここで、$\mathbf{J} \in \mathbb{R}^{m \times n}$ は $\mathbf{u}$ の $\mathbf{x}$ に関する*ヤコビ行列*（Jacobian matrix）であり、その成分は $J_{ji} = \frac{\partial u_j}{\partial x_i}$ である。アフィン写像 $\mathbf{u} = \mathbf{A}\mathbf{x} - \mathbf{y}$ の場合、ヤコビ行列は単に $\mathbf{J} = \mathbf{A}$ となる。

<!--en:c0949c8c-->
小さな例を手計算で追ってみよう。$z = u^2 + v$（ただし $u = x_1 x_2$、$v = x_2^2$）とし、$(x_1, x_2) = (1, 2)$ で評価する。入力 $x_2$ から $z$ へは、$u$ を通る経路と $v$ を通る経路の 2 つの経路がある（:numref:`fig_calc_chain_paths`）。それぞれの経路は局所的な導関数の積をもたらし、2 つの寄与を足し合わせることで $\partial z / \partial x_2 = 4 \cdot 1 + 1 \cdot 4 = 8$ が得られる。

<!--en:02554915-->
![$z = u^2 + v$（$u = x_1 x_2$、$v = x_2^2$）の $(x_1, x_2) = (1, 2)$ における連鎖律。各ボックスは変数の値（黒）と $z$ のその変数に関する導関数（赤）を示し、各赤い矢印は局所的な導関数を表す。導関数は経路に沿って掛け合わされ、$x_2$ のように 2 つの経路が合流する箇所で足し合わされる。](../img/calculus-parallel-dag.svg)

<!--en:f63a3677-->
直線モデルの損失 $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ を 2 つの段階、すなわち（1）内側の残差 $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$（ヤコビ行列は $\mathbf{J} = \mathbf{X}$）と（2）外側の二乗ノルム $L = \|\mathbf{r}\|^2$（勾配は $\nabla_{\mathbf{r}} L = 2\mathbf{r}$）に分解すると、多変数の連鎖律によって $\nabla_{\mathbf{w}} L = \mathbf{X}^\top(2\mathbf{r})$ が 2 回のパスで復元される。

<!--en:6936402f-->
*順伝播*（forward pass）では $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ と $L = \mathbf{r}^\top\mathbf{r}$ を計算する。一方、*逆伝播*（backward pass）、すなわち*誤差逆伝播法*（backpropagation）では $2\mathbf{r}$ に $\mathbf{X}^\top$ を掛け、おおよそ 1 回の順伝播評価と同じコストですべてのパラメータに関する厳密な勾配を計算する。:numref:`sec_autograd` では、この機械的な逆伝播をフレームワークの自動微分（automatic differentiation）エンジンに委ねる。

<!--en:30ac03ff-->
## まとめ

<!--en:825f904e-->
導関数 $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ は接線の傾きと局所線形近似 $f(x+h) \approx f(x) + f'(x)h$ を与える。$f: \mathbb{R}^n \to \mathbb{R}$ に対しては、勾配 $\nabla_{\mathbf{x}} f(\mathbf{x})$ がすべての偏微分をまとめたものとなり、$f(\mathbf{x}+\mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{h}$ を満たし、等位集合に直交して最急上昇の方向を指すことから、勾配降下法 $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$ の根拠となる。$O(n)$ 回の評価コストがかかり桁落ちによって精度が失われる（$\sim \sqrt{\epsilon}$）有限差分とは異なり、多変数の連鎖律 $\nabla_{\mathbf{x}} z = \mathbf{J}^\top \nabla_{\mathbf{u}} z$ は、1 回の逆伝播（*誤差逆伝播法*）によって損失からすべてのパラメータへ厳密な勾配を逆方向に伝播させる。

<!--en:2294243a-->
## 演習問題

<!--en:11ffcb1a-->
1. **底と指数の両方が変化するときの $x^x$ の微分。**$x > 0$ に対し、関数 $f(x) = x^x$（$x$ が底と指数の両方に現れる）を考える。
    1. べき乗の微分公式のみを適用すること（指数を定数とみなして $x \cdot x^{x-1} = x^x$ を得ること）と、指数関数の微分公式のみを適用すること（底を定数とみなして $x^x \log x$ を得ること）がいずれも誤りである理由を説明せよ。
    1. $x^x = e^{x \log x}$ と書き換え、連鎖律と積の微分公式を適用して $f'(x)$ を求めよ。真の導関数が小問 1 の 2 つの単純な答えの*和*になっていることに注目し、なぜそうなるのかを、$g(u, v) = u^v$ に対して $u = x, v = x$ に沿って 2 変数の連鎖律を適用することで説明せよ。

<!--en:e70f34e9-->
1. **なぜ二乗誤差からは平均が得られ、絶対誤差からは中央値が得られるのか。**単一パラメータのモデルが定数値 $w \in \mathbb{R}$ を、$n$ 個の訓練例（スカラーラベルは $y_1, \ldots, y_n$）に対して予測するとする。
    1. 二乗誤差の和 $L_2(w) = \sum_{i=1}^n (w - y_i)^2$ を $w$ に関して微分し、導関数をゼロと置いて、唯一の最小点が標本*平均* $w^* = \frac{1}{n}\sum_{i=1}^n y_i$ であることを証明せよ。
    1. 次に、絶対誤差の和 $L_1(w) = \sum_{i=1}^n |w - y_i|$ を考える。任意の点 $w$（どのラベル $y_i$ とも等しくない点）において、導関数 $L_1'(w)$ は $w$ より真に小さいラベルの個数から $w$ より真に大きいラベルの個数を引いた値に等しいことを示せ。$w$ がどの値をとるとき $L_1'(w)$ は存在しないか。以上から、$L_1(w)$ は $\{y_1, \ldots, y_n\}$ の任意の標本*中央値*によって最小化されることを導け。
    1. 4 つのラベル $\{1, 1, 0, 10\}$ に対し、$L_2$ の最小点（平均）と $L_1$ の最小点（中央値）を計算せよ。外れ値のラベル $10$ が破損して $100$ になった場合、それぞれの最小点はどうなるか。

<!--en:8d227e94-->
1. **ユークリッドノルムの勾配。**勾配 $\nabla_{\mathbf{x}} f(\mathbf{x})$ を、二乗していないユークリッドノルム $f(\mathbf{x}) = \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$ に対して任意の非ゼロベクトル $\mathbf{x} \neq \mathbf{0}$ で計算し、勾配が存在するすべての点で $\|\nabla_{\mathbf{x}} f(\mathbf{x})\|_2 = 1$ となることを確かめよ。原点 $\mathbf{x} = \mathbf{0}$ では勾配はどうなるか。また、それはなぜか。

<!--en:5253e0cf-->
1. **勾配降下法のステップサイズ安定性。**等方的な二次関数 $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ を、勾配降下法の更新式 $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)$（学習率 $\eta > 0$）によって最小化することを考える。
    1. $\mathbf{x}_0 = (1, 1)^\top$ から出発し、1 ステップ後の $\mathbf{x}_1$ と新しい目的関数値 $f(\mathbf{x}_1)$ を、3 通りの学習率 $\eta = 0.1$、$\eta = 1.0$、$\eta = 2.0$ について計算せよ。
    1. $\mathbf{x}_{k+1}$ を $\mathbf{x}_k$ と $\eta$ を用いた閉じた式で表せ。学習率 $\eta > 0$ がどの範囲にあるとき、勾配降下法の 1 ステップによって $f(\mathbf{x})$ は任意の非ゼロの初期点 $\mathbf{x}_k \neq \mathbf{0}$ から真に減少するか。また、$\eta$ がどの特別な値のとき 1 ステップで大域的最小値 $\mathbf{x}^* = \mathbf{0}$ へ直接到達するか。さらに、$\eta > 1$ のときは何が起きるか。
    1. 勾配降下法のステップでパラメータベクトルが変化しない停留点 $\mathbf{x}$（$\mathbf{x}_{k+1} = \mathbf{x}_k$ を満たす点）を、$f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ の場合と一般の微分可能な目的関数 $f$ の場合の両方について特徴づけよ。

<!--en:2aa487b4-->
1. [code] **中心差分と前進差分。** ● 本節で解析した片側の*前進差分* $\frac{f(x+h) - f(x)}{h}$ は $O(h)$ の打ち切り誤差を持つ。対称な*中心差分*

   $$\frac{f(x+h) - f(x-h)}{2h}$$

   は関数 $f$ を $x+h$ と $x-h$ で評価することで、$O(h)$ となる二次の項が相殺されるようにしている。中心差分版の `quotient_error` を実装し、対数グリッド上の $h$ の値にわたって `float64` で評価せよ。
    1. 二次の損失 $L(w) = 3w^2 - 4w + 2$ の $w = 1$ において、前進差分と中心差分の誤差曲線を同じ両対数軸上にプロットせよ。二次関数では、なぜ中心差分の誤差が計算機イプシロン（$\sim 10^{-16}$）まで、大きな $h = 10^{-1}$ であっても一気に下がるのか。
    1. 次に、二次関数ではない $f(x) = e^x$ の $x = 0$（$f'(0) = 1$）において同じ比較を繰り返せ。大きい $h$ における中心差分曲線の打ち切り誤差領域の傾きを、傾き $1$ の前進差分曲線と比較し、各手法における最適なステップサイズ $h^*$ と達成可能な最小誤差を求め、テイラー展開 $e^{\pm h} = 1 \pm h + \frac{h^2}{2} \pm \frac{h^3}{6} + \cdots$ を用いてその両方を説明せよ。

<!--en:d457ddb3-->
1. [code] **`float64` と `float32` における勾配チェック。** ● 有限差分は訓練には遅すぎるものの、中心有限差分は手計算で導出した勾配やカスタムカーネルの勾配を単体テストする際の定番手法である。
    1. 関数 `central_gradient(f, x, h=1e-5)` を実装し、中心差分によって勾配 $\nabla f(\mathbf{x})$（$f: \mathbb{R}^n \to \mathbb{R}$）を推定せよ。これを `float64` で用い、二次形式の勾配の恒等式 $\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ をランダムな*非対称* $5 \times 5$ 行列 $\mathbf{A}$ とランダムなベクトル $\mathbf{x} \in \mathbb{R}^5$ に対してテストし、5 つの座標にわたる最大相対誤差を求めよ。
    1. 次に、*バグのある*式 $2\mathbf{A}\mathbf{x}$（$\mathbf{A}$ が対称でないことを忘れたときによくある誤り）に対して同じチェックを実行せよ。このとき相対誤差はどのくらい大きくなるか。
    1. 最後に、正しい式 $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ のチェックを、32 ビット `float32` のフレームワークテンソルを用いて $h \in \{10^{-1}, 10^{-2}, \ldots, 10^{-6}\}$ にわたって繰り返せ。どの $h$ が `float32` で最小の相対誤差を達成するか。また、それが前進差分の最適値 $h^* \approx 10^{-4}$ と異なるのはなぜか。さらに、`float32` において正しい勾配実装とバグのある実装を明確に切り分けられる相対誤差のしきい値はどの程度か。

<!--en:f162ccbe-->
[ディスカッション](https://d2l.discourse.group/t/32)

<!--en:c02ea5db-->
[ディスカッション](https://d2l.discourse.group/t/33)

<!--en:6e9d2b15-->
[ディスカッション](https://d2l.discourse.group/t/197)

<!--en:1575f8ea-->
[ディスカッション](https://d2l.discourse.group/t/17969)

<!--en:0cc0fd39-->
微分法：導関数、勾配、ヤコビ行列、連鎖律<br>**局所線形近似・接線の傾き・最急降下・計算 DAG・有限差分の限界**

<!--en:9b0361c4-->
::: {.slide title="なぜ微分積分が学習を駆動するのか：最適化の羅針盤"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:b0b2ea51-->
- **平坦な 0-1 正解率を滑らかな代理損失で置換：**階段関数状の誤差はほとんどいたるところで傾きゼロであるのに対し、滑らかな損失（交差エントロピー、MSE）は局所的な接線 $f(x+h) \approx f(x) + f'(x)h$ を与える

<!--en:7233b752-->
- **1 つの勾配と $d+1$ 回の手探り評価：**$d = 10^9$ 個の重みを順に摂動させると $10^9$ 回の順伝播を要するが、$\nabla_{\mathbf{w}}\mathcal{L}$ は全 $d$ 座標での上り方向を 1 回の逆伝播で示す

<!--en:83777c05-->
::: {.slide title="割線の傾きから導関数へ"}

<!--en:bd012d73-->
[1 変数微積分]{.kicker}

<!--en:fbf2101b-->
- **導関数** $f'(x)$ は瞬間的な変化率であり、$(x, f(x))$ と $(x+h, f(x+h))$ を結ぶ**割線の傾き**の $h \to 0$ における極限
  $$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

<!--en:fd6e97a0-->
- $L(w) = 3w^2 - 4w + 2$ の $w = 1$ において、$L(1+h) = 1 + 2h + 3h^2$ を展開すると $\frac{L(1+h)-L(1)}{h} = 2 + 3h \to 2$

<!--en:2de1039e-->
::: {.slide title="局所線形近似としての接線の可視化"}

<!--en:2e4fe053-->
[局所線形性]{.kicker}

<!--en:9b1d2771-->
- **$a$ の周りでの一次テイラー展開：**
  $$f(x) \approx f(a) + f'(a)(x - a)$$
  $w = 1$（$L(1) = 1$、$L'(1) = 2$）における接線は $T(w) = 1 + 2(w - 1) = 2w - 1$

<!--en:53adeff8-->
- **局所線形性が勾配降下法を正当化する理由：**微小ステップ $\Delta w = -\eta L'(w)$（学習率 $\eta > 0$）に対し、
  $$L\big(w - \eta L'(w)\big) \approx L(w) - \eta \big[L'(w)\big]^2 < L(w)$$

<!--en:c87398d1-->
- $w = 1$ から $\eta = 0.1$ で更新すると $w = 1 - 0.1(2) = 0.8$ となり、$L$ は $1.0$ から $0.72$ に減少する（接線による予測：$0.60$）。$L'(w^*) = 6w^* - 4 = 0$ と置けば大域的最小値 $w^* = 2/3$ が求まる。

<!--en:d422adfc-->
::: {.slide title="基本的な微分公式と積の微分公式"}

<!--en:8761149c-->
[解析的微分]{.kicker}

<!--en:65cfcb23-->
| 関数 | 導関数 $\frac{d}{dx}$ | 法則 | 公式 |

<!--en:3ae81821-->
| 定数 $C$ | $0$ | **線形性** | $(\alpha u + \beta v)' = \alpha u' + \beta v'$ |

<!--en:48112488-->
| べき乗 $x^n$ | $n x^{n-1}$ | **積の微分** | $(u\,v)' = u'v + u\,v'$ |

<!--en:3def97e7-->
| 指数 $e^x$ | $e^x$ | **商の微分** | $(u/v)' = (u'v - u\,v')/v^2$ |

<!--en:df4c84c5-->
| 対数 $\ln x$ | $1/x$ | **$3w^2 - 4w + 2$** | $3(2w) - 4(1) = 6w - 4$ |

<!--en:f94a93c3-->
- **積の微分公式の幾何学：**長方形 $f(x)g(x)$ を $h$ だけ広げると、2 つの帯 $g(x)f'(x)h + f(x)g'(x)h$ と、$O(h^2)$ の角（$h \to 0$ で消失）が加わる

<!--en:ef667715-->
::: {.slide title="偏微分：多変数曲面のスライス"}

<!--en:a36b8745-->
[多変数微積分]{.kicker}

<!--en:f0c122a0-->
- $f: \mathbb{R}^n \to \mathbb{R}$ に対し、**偏微分** $\frac{\partial f}{\partial x_i}$ は $x_i$ 以外の全座標を固定して 1 次元スライスの傾きを測る
  $$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

<!--en:9747374f-->
- 直線モデルの損失 $L(w_1, w_2) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ を $\mathbf{w} = (1, 0)$ でスライスすると $L(w_1, 0) = 3w_1^2 - 4w_1 + 2$ および $L(1, w_2) = 5w_2^2 + 4w_2 + 1$ となり、$\frac{\partial L}{\partial w_1} = 2$、$\frac{\partial L}{\partial w_2} = 4$

<!--en:4ea1b61f-->
::: {.slide title="勾配ベクトル：最急上昇・最急降下・等高線"}

<!--en:3353e0e3-->
[勾配の幾何学]{.kicker}

<!--en:74437d8b-->
- 全 $n$ 個の偏微分を並べると**勾配** $\nabla_{\mathbf{x}} f(\mathbf{x}) = \big[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\big]^\top \in \mathbb{R}^n$ になる

<!--en:0f24fd93-->
- コーシー・シュワルツの不等式より、単位ベクトル $\mathbf{u}$（角度 $\theta$ を $\nabla f(\mathbf{x})$ となす）に沿った方向変化率は $D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x})^\top \mathbf{u} = \|\nabla f(\mathbf{x})\|_2 \cos\theta$
  - **最急上昇（$\theta = 0^\circ$）：**$\nabla f(\mathbf{x})$ は真っすぐ上りを指す
  - **最急降下（$\theta = 180^\circ$）：**$-\nabla f(\mathbf{x})$ は真っすぐ下りを指す
  - **等高線に直交（$\theta = 90^\circ$）：**$D_{\mathbf{u}} f = 0$ が等高線 $f(\mathbf{x}) = c$ に沿って成り立つ

<!--en:17539517-->
::: {.slide title="多変数線形近似と勾配降下法のステップ"}

<!--en:106df48d-->
[一次最適化]{.kicker}

<!--en:8cf03165-->
全 $n$ 座標を同時に $\boldsymbol{\epsilon} \in \mathbb{R}^n$ だけ摂動させると、各軸の線形変化が足し合わされる。

<!--en:8a82567b-->
- **局所的な減少の保証：**勾配降下ステップ $\boldsymbol{\epsilon} = -\eta \nabla_{\mathbf{x}} f(\mathbf{x})$ を小さな学習率 $\eta > 0$ で選ぶと、
  $$f\big(\mathbf{x} - \eta \nabla_{\mathbf{x}} f(\mathbf{x})\big) \approx f(\mathbf{x}) - \eta \big\|\nabla_{\mathbf{x}} f(\mathbf{x})\big\|_2^2 \le f(\mathbf{x})$$

<!--en:2421b398-->
- $\nabla_{\mathbf{x}} f(\mathbf{x}) \neq \mathbf{0}$ であり、線形近似が成り立つほど $\eta$ が小さければ、各ステップで $f$ は厳密に減少する

<!--en:623ff18a-->
- **勾配降下法がジグザグに進む理由：**直線モデルの $\mathbf{w}_0 = [1, 0]^\top$ では $\nabla L(\mathbf{w}_0) = [2, 4]^\top$ であり、$\eta = 0.1$ で 1 ステップ更新すると、
  $$\mathbf{w}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} - 0.1 \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 0.8 \\ -0.4 \end{bmatrix}, \quad L(\mathbf{w}_1) = 0.40 < 1.0$$

<!--en:37669c17-->
- $-\nabla L$ は中心 $\mathbf{w}^* = [7/6, -1/2]^\top$ を指すのではなく*楕円状*の等高線に直交するため、軌跡は谷をジグザグに下る

<!--en:a3bdabb4-->
::: {.slide title="多変数の連鎖律と計算 DAG"}

<!--en:f5f26c40-->
[合成関数と DAG]{.kicker}

<!--en:073f9d6b-->
- **単一経路の連鎖律：**$z = f(u)$ かつ $u = g(x)$ のとき、局所的な感度は辺の系列 $x \to u \to z$ に沿って掛け合わされる
  $$\frac{dz}{dx} = \frac{dz}{du}\frac{du}{dx}$$

<!--en:c1ea5a0b-->
- **DAG 上の多変数連鎖律：**$y = f(\mathbf{u})$ が中間ノード $u_1, \ldots, u_m$ に依存し、各 $u_i = g_i(\mathbf{x})$ が入力 $x_1, \ldots, x_n$ に依存するとき、
  $$\frac{\partial y}{\partial x_j} = \sum_{i=1}^m \frac{\partial y}{\partial u_i} \frac{\partial u_i}{\partial x_j} \quad \Longleftrightarrow \quad \nabla_{\mathbf{x}} y = \mathbf{J}^\top \nabla_{\mathbf{u}} y$$

<!--en:3f877869-->
- **グラフの規則：**$x_j$ から $y$ への各有向経路に沿って辺の重み（局所的な導関数）を掛け合わせ、**すべての並列経路にわたって和をとる**

<!--en:ef967a43-->
::: {.slide title="計算例：計算 DAG 上の並列経路の和"}

<!--en:554e0ed7-->
[多変数連鎖律の具体例]{.kicker}

<!--en:f5ceab83-->
$z = u^2 + v$（$u = x_1 x_2$、$v = x_2^2$）を $(x_1, x_2) = (1, 2)$ において考える。

<!--en:701e3104-->
- **順伝播（青）：**$u = 2,\; v = 4 \implies z = 2^2 + 4 = 8$

<!--en:0f568de6-->
- **逆伝播（赤）：**上流の随伴変数は $\frac{\partial z}{\partial u} = 2u = 4$ と $\frac{\partial z}{\partial v} = 1$ であり、
  $$\frac{\partial z}{\partial x_1} = 4 \cdot 2 = 8, \quad \frac{\partial z}{\partial x_2} = 4(1) + 1(4) = 8$$

<!--en:6098f9b6-->
- $\frac{\partial z}{\partial u} = 4$ を再利用することで、$z(x_1, x_2)$ の数式展開を避けられる

<!--en:710b680e-->
::: {.slide title="行列微分の恒等式とベクトル値関数のヤコビ行列"}

<!--en:08e5232d-->
[ベクトル・行列微分]{.kicker}

<!--en:ed87852b-->
| 表現 | 関数 $f(\mathbf{x})$ | 勾配・ヤコビ行列 | 深層学習での応用 |

<!--en:f7baacb3-->
| **線形形式** | $\mathbf{a}^\top \mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{a}^\top \mathbf{x}) = \mathbf{a}$ | 線形ニューロン $\mathbf{w}^\top \mathbf{x} + b$ |

<!--en:0569b947-->
| **二次形式** | $\mathbf{x}^\top \mathbf{A} \mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ | ヘッセ行列の項（$2\mathbf{A}\mathbf{x}$、ただし $\mathbf{A} = \mathbf{A}^\top$ のとき） |

<!--en:1cd38ae2-->
| **二乗 $L_2$ ノルム** | $\|\mathbf{x}\|_2^2 = \mathbf{x}^\top \mathbf{x}$ | $\nabla_{\mathbf{x}}\|\mathbf{x}\|_2^2 = 2\mathbf{x}$ | 重み減衰と MSE の残差 |

<!--en:ad10c17c-->
| **最小二乗法** | $\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$ | $\nabla_{\mathbf{w}} L = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ | 線形回帰のバッチ勾配 |

<!--en:441ef7f1-->
| **ベクトル写像** | $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ | $\mathbf{J}_{\mathbf{f}}(\mathbf{x}) \in \mathbb{R}^{m \times n},\; (\mathbf{J}_{\mathbf{f}})_{ij} = \frac{\partial f_i}{\partial x_j}$ | 層の連結：$\mathbf{J}_{\mathbf{f} \circ \mathbf{g}} = \mathbf{J}_{\mathbf{f}}\mathbf{J}_{\mathbf{g}}$ |

<!--en:4c9b0701-->
任意のアフィン写像 $\mathbf{u} = \mathbf{A}\mathbf{x} + \mathbf{b}$ に対して $\mathbf{J}_{\mathbf{u}}(\mathbf{x}) = \mathbf{A}$ であるため、$\nabla_{\mathbf{u}} y$ を逆伝播するときは $\mathbf{A}^\top$ を掛けて $\nabla_{\mathbf{x}} y = \mathbf{A}^\top \nabla_{\mathbf{u}} y$ とする。

<!--en:dc5877d2-->
::: {.slide title="連鎖律から誤差逆伝播法へ：ベクトル・ヤコビ行列積"}

<!--en:37c3ee3d-->
[順伝播と逆伝播]{.kicker}

<!--en:abcf1fe8-->
- **2 段階のグラフ（$\mathbf{w} \to \mathbf{r} \to L$）：**順伝播で $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ と $L = \|\mathbf{r}\|_2^2$ を計算

<!--en:2c7c60c4-->
- **逆伝播 VJP：**$\nabla_{\mathbf{r}} L = 2\mathbf{r}$ を $\nabla_{\mathbf{w}} L = \mathbf{X}^\top (2\mathbf{r})$ へと $O(mn)$ FLOPs で伝播

<!--en:67634904-->
::: {.slide title="なぜ数値的な有限差分をそのまま使わないのか？"}

<!--en:01e101ba-->
[数値微分と解析的微分]{.kicker}

<!--en:2a67ea66-->
近似 $f'(x) \approx \frac{f(x+h) - f(x)}{h}$ には、深層学習において **2 つの致命的な欠陥**がある。

<!--en:45c0f3f8-->
1. **数値誤差の V 字曲線：**
   - **大きすぎる $h$：**テイラー展開の打ち切り誤差が $O(h)$（中心差分では $O(h^2)$）で増大
   - **小さすぎる $h$：**ほぼ等しい浮動小数点数 $f(x+h) - f(x)$ の減算で仮数部のビットが消失（**桁落ち**、誤差 $\sim \epsilon / h$ により `float32` は有効桁数 $\sim 4$ 桁が上限）

<!--en:b6cf0bff-->
2. **計算量の $O(d)$ 爆発：**$d = 10^9$ 個の重みを摂動させると 1 ステップあたり **$10^9 + 1$ 回の順伝播**が必要になるが、リバースモード自動微分（§{sec_autograd}）なら **1 回の逆伝播**で済む

<!--en:8e717126-->
::: {.slide title="理解度チェック：勾配・等高線・連鎖律"}

<!--en:929b7671-->
[理解度の確認]{.kicker}

<!--en:fc0740c9-->
1. **勾配の幾何学：**$\mathbf{x} = (1, 2)$ において $\nabla f(\mathbf{x}) = [3, -4]^\top$ である。**最急降下**の単位ベクトル $\mathbf{u}^*$、変化率 $D_{\mathbf{u}^*} f(\mathbf{x})$、および等高線に接するベクトルは何か。

<!--en:db4a526f-->
2. **行列微分の落とし穴：**非対称な $\mathbf{A} = [[1, 2], [0, 1]]$ と $\mathbf{x} = [1, 1]^\top$ に対し、$\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x})$ は何か。なぜ $2\mathbf{A}\mathbf{x}$ は誤りなのか。

<!--en:f068524e-->
3. **深層ネットワークへのスケーリング：**パラメータ数 $d = 10^8$ のとき、1 ステップあたりに必要な順伝播の回数は、中心差分と誤差逆伝播法でそれぞれ何回か。

<!--en:557b98b6-->
**解答：**

<!--en:13e7aad7-->
1. $\|\nabla f\|_2 = 5 \implies \mathbf{u}^* = \mathbf{[-3/5,\, 4/5]^\top}$ であり、変化率は $-\|\nabla f\|_2 = \mathbf{-5}$。接ベクトル：$\mathbf{[4, 3]^\top}$（$3(4) - 4(3) = 0$）

<!--en:c9c83500-->
2. $\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x} = \mathbf{[4, 4]^\top}$（$\mathbf{A} \neq \mathbf{A}^\top$ であるため、$2\mathbf{A}\mathbf{x} = [6, 2]^\top$ では積の微分公式の転置項が欠落する）

<!--en:06c13fb1-->
3. 中心差分は **$2d = 2 \times 10^8$ 回の順伝播**を要するのに対し、誤差逆伝播法は **1 回の順伝播 + 1 回の逆伝播**（順伝播 1 回の $\approx 2\text{--}3\times$）で済む

<!--en:0ce49791-->
::: {.slide title="まとめ：微分の諸概念・形状・幾何学的意味・最適化での役割"}

<!--en:c5a222f8-->
[重要ポイント]{.kicker}

<!--en:4db68606-->
| 対象 | 公式 | 形状 | 幾何学的意味 | 深層学習での役割 |

<!--en:3d998d7e-->
| **導関数** $f'(x)$ | $\lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$ | `()` | $x$ における接線の傾き | 1 次元線形近似 $f(x+h) \approx f(x) + f'(x)h$ |

<!--en:66674809-->
| **偏微分** $\frac{\partial f}{\partial x_i}$ | $x_{j \neq i}$ を固定した導関数 | `()` | 軸 $i$ スライスの傾き | 単一の重み $w_i$ に対する損失の感度 |

<!--en:4d8c76bd-->
| **勾配** $\nabla_{\mathbf{x}} f(\mathbf{x})$ | $[\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^\top$ | **`(n,)`** | 上り方向を指し等高線と $\perp$ | 降下ステップ $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}\mathcal{L}$ |

<!--en:2d9a0cbf-->
| **ヤコビ行列** $\mathbf{J}_{\mathbf{f}}(\mathbf{x})$ | $(\mathbf{J}_{\mathbf{f}})_{ij} = \partial f_i / \partial x_j$ | **`(m, n)`** | 局所線形変換 $\mathbf{J}\mathbf{h}$ | 層間の感度行列 |

<!--en:8b4f33b9-->
| **連鎖律（VJP）** | $\nabla_{\mathbf{x}} y = \mathbf{J}_{\mathbf{u}}(\mathbf{x})^\top \nabla_{\mathbf{u}} y$ | `(n,)` | 全 DAG 経路にわたる和 | リバースモードの誤差逆伝播法（§{sec_autograd}） |

<!--en:bde749dc-->
| **有限差分** | $\frac{f(x+h)-f(x-h)}{2h}$ | $2d$ 回の評価 | $[x-h, x+h]$ 上の割線の傾き | `float64` による単体テストの勾配チェック |
