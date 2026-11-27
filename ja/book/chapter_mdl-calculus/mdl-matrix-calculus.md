<!--en:f4c72ea5-->
# 行列微分

<!--en:73790c9f-->
直前の 2 節では、損失をスカラー変数の関数として扱った。:numref:`sec_mdl-multivariable_calculus` では偏微分を並べた列として勾配を構成し、:numref:`sec_mdl-chain-rule-graphs` ではノードごとに 1 つの数値を持つグラフ上の逆伝播によってその列を計算した。しかし、ニューラルネットワークはスカラーでは記述されない。各層はベクトルをベクトルへ写し、その重みは行列をなす。ベクトルや行列に関する損失の導関数とは何であり、どうすれば層全体に対して一度に計算できるのだろうか。

<!--en:a9950bb3-->
手計算できるほど小さなネットワークを見れば、何が必要かがわかる。2 つの入力、2 つの ReLU 隠れユニット、1 つの出力を持つ次のネットワークを考えよう。

<!--en:3d87584d-->
ここで、$\boldsymbol\varphi$ は各成分に $\max(0,\cdot)$ を適用する写像である。$\mathbf x=(1,1)$ において活性化前の値は $\mathbf W_1\mathbf x+\mathbf b=(3,-2)$ となるため、2 番目の隠れユニットは非活性となり、隠れベクトルは $(3,0)$、出力は $g=6$ となる。訓練ステップでは、この出力に関して 3 つの問いが生じる。第 1 に、入力が微小な $\boldsymbol\delta$ だけ動くと、出力は $2\delta_1+4\delta_2$ だけ変化する。この係数は 2 つの数値からなる行であり、列である勾配の転置になっている。第 2 に、重み行列が $\Delta\mathbf W_1$ だけ動くと、出力は $2\,\Delta W_{11}+2\,\Delta W_{12}$ だけ変化し、第 2 行からはまったく影響を受けない。したがって、行列に関する導関数はそれ自体が行列 $\left(\begin{smallmatrix}2&2\\0&0\end{smallmatrix}\right)$ となる。第 3 に、隠れ層 $\mathbf x\mapsto\boldsymbol\varphi(\mathbf W_1\mathbf x+\mathbf b)$ はベクトルをベクトルへ写す。その導関数はスカラーの傾きでも勾配ベクトルでも表せないが、出力はまさにこの層を介してのみ $\mathbf W_1$ に依存している。

<!--en:9896c009-->
このような導関数は、手計算では驚くほど間違えやすい。正方の重み行列で転置を落としても、次元は一致したままでエラーも発生せず、ただ訓練が狂ってしまう。行列微分の公式集には何百もの項目が並んでいるが、公式を引くだけではこの種のうっかりミスはほとんど防げない。

<!--en:1fb10f31-->
本節では、これら 3 つの問いすべてに、単一の数学的対象と単一の原理で答える。ベクトル値写像の導関数は、その写像の最良線形近似として定義される行列、すなわち*ヤコビ行列*（Jacobian）である。写像の合成はヤコビ行列の積に対応するため、ネットワークの導関数は層ごとに 1 つずつの行列の積となる。さらに、層が必要とするあらゆる勾配（線形形式や二次形式、最小二乗法、行列積、ソフトマックス）は、1 つの手法から導かれる。関数の一次の変化を書き下し、その係数として勾配を読み取るのである。同じ手法から、:numref:`sec_backprop` で用いた各層の逆伝播だけでなく、$\mathbf p-\mathbf y$ という :numref:`subsec_softmax_and_derivatives` の分類器の勾配も得られる。

<!--en:da468fe6-->
特に覚えておくべき結果は 3 つある。ヤコビ行列の積 :eqref:`eq_mdl-jacobian-product`、勾配は常にその変数と同じ形状を持つという規則、そして行列積の逆伝播則 :eqref:`eq_mdl-matmul-vjp` である。初読時には連鎖律の証明を読み飛ばしても構わないが、全微分法（differential method、:numref:`subsec_mdl-differential-method`）は読み飛ばせない。以降の議論はすべてこれに依拠するからである。

<!--en:857f9054-->
本節全体を通して、以下のインポートを使用する。

<!--en:85828374-->
## ヤコビ行列

<!--en:034d0cbc-->
### 最良線形近似

<!--en:e8b70472-->
$\mathbf x=(1,1)$ の近傍では、冒頭のネットワークの隠れ層はアフィンであるため、その線形近似は厳密に一致し、調べるべき誤差が残らない。そこでまずは、本節と次節で共通の例として再利用する非線形な写像から始め、定義が整ったところで再びネットワークに戻ることにしよう。

<!--en:57c9a288-->
$\mathbf x_0$ において、この写像の値は $\mathbf f(\mathbf x_0)=(0.9794,\ 0.6250)$ である。2 つの出力はそれぞれ 2 つの偏微分を持ち、$\partial f_1/\partial x=1$、$\partial f_1/\partial y=\cos y=0.8776$、$\partial f_2/\partial x=x=0.5$、$\partial f_2/\partial y=1$ となる。これら 4 つの数値を、出力ごとに 1 行、入力ごとに 1 列を持つ行列に並べる。

<!--en:fec0a449-->
第 $j$ 列は、入力 $j$ が動いたときに両方の出力がどう応答するかを記録している。この行列は $\mathbf f$ の $\mathbf x_0$ における導関数である。すなわち、$\mathbf f(\mathbf x_0+\boldsymbol\delta)\approx\mathbf f(\mathbf x_0)+\mathbf J\boldsymbol\delta$ が微小な $\boldsymbol\delta$ に対して成り立ち、他のどの行列もこれほど精密には $\mathbf f$ を近似しない。

<!--en:5b6eb18a-->
1 変数の場合、導関数は最良の直線近似 $f(x+\delta)\approx f(x)+f'(x)\,\delta$ の傾きを与える。ベクトル値への一般化も同じ局所近似に基づく。写像 $\mathbf f:\mathbb R^n\to\mathbb R^m$ が点 $\mathbf x$ で*微分可能*（differentiable）であるとは、ある行列 $\mathbf J\in\mathbb R^{m\times n}$ が存在して次式を満たすことをいう。

<!--en:d2d1660b-->
これは、線形モデル $\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta$ の誤差を $\|\boldsymbol\delta\|$ で割ったものが、$\boldsymbol\delta\to\mathbf 0$ のとき 0 に収束することを意味する。このような行列が存在するとき、それは一意に定まる。これを*ヤコビ行列* $\mathbf J_{\mathbf f}(\mathbf x)$ と呼び、その各成分は単なる偏微分となる。

<!--en:03a5d172-->
その理由を見るには、摂動 $\boldsymbol\delta=t\,\mathbf e_j$（座標軸 $j$ に沿った微小ステップ）を :eqref:`eq_mdl-jacobian-linearization` に代入し、第 $i$ 成分に注目すればよい。$t$ で割って $t\to0$ とすると、$\partial f_i/\partial x_j$ が $(i,j)$ 成分として残り、これは出発点とした行列 $\mathbf J$ の選び方によらない。したがって、偏微分は $\mathbf J$ の各成分を与えるが、それを定義するのは線形近似である。

<!--en:5c101401-->
偏微分の存在だけでは、微分可能性は保証されない。$f(x,y)=2xy/(x^2+y^2)$（ただし $f(0,0)=0$）という :numref:`subsec_mdl-gradient` の関数を考えよう。原点では両方の偏微分が $0$ であるにもかかわらず、$f$ は値 $1$ を対角線 $x=y$ 上のいたるところでとる。ヤコビ行列の唯一の候補はゼロ行ベクトルであるが、対角線上の近似誤差は $1$ のままであり、$\boldsymbol\delta$ をどれほど小さくしても変わらないため、$f$ は原点でヤコビ行列を持たない。このような振る舞いを排除するには、偏微分の連続性が十分条件となる。すなわち、すべての偏微分 $\partial f_i/\partial x_j$ が $\mathbf x$ の近傍で存在し、かつ $\mathbf x$ で連続であれば、$\mathbf f$ は $\mathbf x$ で微分可能である :cite:`Rudin.1976`。

<!--en:d6346454-->
:numref:`fig_mdl-cal-jacobian-ellipse` は、この定義を幾何学的に示している。$\mathbf x_0$ の近傍では、写像はそのヤコビ行列のように振る舞う。半径 $r$ で $\mathbf x_0$ を中心とする円は、楕円 $\mathbf f(\mathbf x_0)+\mathbf J\boldsymbol\delta$（$\|\boldsymbol\delta\|=r$）へとほぼ写され、その半軸の長さは $r$ と $\mathbf J$ の特異値の積になる（:numref:`sec_mdl-svd-low-rank` で示したように、任意の行列は単位円を特異値を半軸とする楕円へ写す）。ここでは特異値が $1.706$ と $0.329$ であるため、半径 $0.35$ の円は半軸 $0.60$ と $0.12$ の楕円になる。真の像と楕円との最大のずれは $0.061$（$r=0.35$ のとき）および $0.015$（$r=0.175$ のとき）であり、半径を半分にするとずれは 4 分の 1 に縮まる。

<!--en:6e36b754-->
![微分可能な写像は局所的にそのヤコビ行列と一致する。半径 $0.35$ と $0.175$ で $\mathbf x_0=(0.5,0.5)$ を中心とする 2 つの円（左）と、$\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ によるそれらの像（右、青）および楕円 $\mathbf f(\mathbf x_0)+\mathbf J(\mathbf x_0)\boldsymbol\delta$（オレンジ、破線）の比較。各楕円の半軸は、半径に特異値 $1.71$、$0.33$（行列 $\mathbf J$ の特異値）を掛けた値である。半径を半分にすると、像と楕円の最大のずれは $0.061$ から $0.015$ へと縮まり、剰余項 $o(\|\boldsymbol\delta\|)$ の振る舞いを示す。](../img/mdl-cal-jacobian-ellipse.svg)

<!--en:52fa1559-->
定義は線形モデルの誤差を $\|\boldsymbol\delta\|$ で割った比が 0 に収束することを保証するが、その速さまでは述べていない。摂動を 10 分の 1 ずつ縮めながら、その比を調べてみよう。

<!--en:c916787b-->
比は $0.09$ から $0.009$、$0.0009$、$0.00009$ へと減少する。0 に収束しており、定義の剰余項 $o(\|\boldsymbol\delta\|)$ が確認できる。さらに、この比は 10 分の 1 に $\|\boldsymbol\delta\|$ を縮めるごとに 10 分の 1 に減少していることにも注目したい。つまり誤差そのものは 100 分の 1 に減少し、$O(\|\boldsymbol\delta\|^2)$ となっている。定義が保証するのは最初の性質だけであり、より速い二次の減衰が成り立つのは $\mathbf f$ が連続な二階導関数を持つからである。

<!--en:aaee95f0-->
### 勾配、ヘッセ行列、導関数の形状

<!--en:99eab6cd-->
:numref:`sec_mdl-multivariable_calculus` で扱った勾配とヘッセ行列（Hessian）は、どちらもヤコビ行列の特別な場合である。この観点に立つと、手計算のたびに生じる形状の疑問、すなわち導関数は行なのか列なのか、また行列に関する導関数はどう並べるべきなのかという問題に決着がつく。

<!--en:3c0fda6b-->
スカラー値関数 $f:\mathbb R^n\to\mathbb R$（$m=1$）に対して、ヤコビ行列は 1 行のベクトル $\partial f/\partial\mathbf x=[\partial f/\partial x_1,\ldots,\partial f/\partial x_n]$ となり、:eqref:`eq_mdl-jacobian-linearization` は一次展開 $f(\mathbf x+\boldsymbol\delta)\approx f(\mathbf x)+(\partial f/\partial\mathbf x)\,\boldsymbol\delta$ となる。:numref:`sec_mdl-multivariable_calculus` では勾配を列ベクトルとして書いた。両者は単に互いの転置である。

<!--en:a2b3c086-->
本書では、それぞれの目的に応じて両方を使い分ける。ベクトル値写像のヤコビ行列は出力ごとに 1 行を持つ $m\times n$ 行列とし、後述する連鎖律が転置なしですっきりと合成できるようにする。一方、スカラー関数の勾配 $\nabla_{\mathbf x}f$ は $\mathbf x$ の形状に合わせた列ベクトルとし、オプティマイザがそのまま減算できるようにする。冒頭のネットワークでは、$\partial g/\partial\mathbf x=(2,\ 4)$ が $1\times2$ のヤコビ行列であり、$\nabla_{\mathbf x}g=(2,4)^\top$ が列ベクトルとしての勾配である。

<!--en:09191df4-->
行列 $\mathbf W\in\mathbb R^{p\times q}$ のスカラー関数には、もう 1 つ規約が必要になる。その勾配 $\nabla_{\mathbf W}f$ を、$p\times q$ 行列で各成分が $\partial f/\partial W_{ij}$ であるものとして定義し、$\mathbf W$ の形状に一致させる。すると一次展開は次のように書ける。

<!--en:6ed0c8ac-->
ここで、内積 $\langle\cdot,\cdot\rangle$ は同じ形状の 2 つの行列を成分ごとに掛けて総和をとる演算であり、行列をベクトルに平坦化したときの通常のドット積と同じである。言葉で言えば、$f$ の一次の変化はすべての成分にわたる「感度 × 微小変化」の和である。冒頭の例では、$\nabla_{\mathbf W_1}g$ は $2\times2$ 行列 $\left(\begin{smallmatrix}2&2\\0&0\end{smallmatrix}\right)$ であり、$\langle\nabla_{\mathbf W_1}g,\Delta\mathbf W_1\rangle=2\,\Delta W_{11}+2\,\Delta W_{12}$ から先ほど求めた変化量が復元される。ここから、本節のあらゆる公式に対する 1 つの妥当性確認が得られる。すなわち、勾配は常にその変数と同じ形状を持つ。導出した勾配の形状が合わなければ、転置が抜けているのである。

<!--en:485f2187-->
もう一度微分すると、ヘッセ行列が現れる。勾配場 $\nabla f:\mathbb R^n\to\mathbb R^n$ はそれ自体がベクトル値写像であり、そのヤコビ行列は二階偏微分の行列、すなわちヘッセ行列 $\mathbf H_f$ であって、その成分は $[\mathbf H_f]_{ij}=\partial^2 f/\partial x_i\partial x_j$ で与えられる。厳密に言えば、$[\mathbf J_{\nabla f}]_{ij}=\partial^2 f/\partial x_j\partial x_i$ は 2 回の微分の順序が逆になるが、交差偏微分が連続である限り両者は一致し、$\mathbf H_f$ は対称になる（:numref:`sec_mdl-multivariable_calculus` のクレロー・シュワルツの定理）。

<!--en:c2908534-->
## ヤコビ行列の連鎖律

<!--en:97899d81-->
ニューラルネットワークは層の合成である。スカラーのグラフ上では、連鎖律は入力から出力へ至るすべての経路にわたって導関数の積を足し合わせた（:numref:`subsec_mdl-chain-rule-paths`）。各層がベクトル値写像になると、その経路上の和は行列積になる。

<!--en:98da1538-->
### 行列積としての合成

<!--en:58801f19-->
**命題（連鎖律）** *$\mathbf f:\mathbb R^n\to\mathbb R^p$ が $\mathbf x$ で微分可能であり、$\mathbf g:\mathbb R^p\to\mathbb R^m$ が $\mathbf f(\mathbf x)$ で微分可能であれば、合成写像 $\mathbf g\circ\mathbf f$ は $\mathbf x$ で微分可能であり、次式が成り立つ。*

<!--en:70d285b2-->
**証明** $\mathbf y=\mathbf f(\mathbf x)$、$\mathbf A=\mathbf J_{\mathbf f}(\mathbf x)$、$\mathbf B=\mathbf J_{\mathbf g}(\mathbf y)$ とおく。基本的な考え方は、単に $\mathbf f$ の線形モデルを $\mathbf g$ の線形モデルに代入することである。:eqref:`eq_mdl-jacobian-linearization` より、$\mathbf f(\mathbf x+\boldsymbol\delta)=\mathbf y+\mathbf A\boldsymbol\delta +o(\|\boldsymbol\delta\|)$ が成り立つ。この変位を $\boldsymbol\eta=\mathbf A\boldsymbol\delta+o(\|\boldsymbol\delta\|)$ とおき、$\mathbf g$ に渡すと次式を得る。

<!--en:f9e388d4-->
2 つ目の等号は 2 つの事実に基づいている。第 1 に、$o(\|\boldsymbol\delta\|)$ の剰余項に定数行列 $\mathbf B$ を掛けても $o(\|\boldsymbol\delta\|)$ のままである。第 2 に、$\|\boldsymbol\eta\|\le\|\mathbf A\|\,\|\boldsymbol\delta\|+o(\|\boldsymbol\delta\|)$ は $\|\boldsymbol\delta\|$ の定数倍で上から抑えられるため、剰余項 $o(\|\boldsymbol\eta\|)$（$\mathbf g$ の展開に現れる項）もまた $o(\|\boldsymbol\delta\|)$ となる。純粋に代数的ではないステップはここだけである。したがって $\boldsymbol\delta$ の係数は :eqref:`eq_mdl-jacobian-linearization` を満たす行列となり、一意性により合成写像のヤコビ行列 $\mathbf J_{\mathbf g\circ\mathbf f}=\mathbf B\mathbf A$ に一致する。$\blacksquare$

<!--en:45fafdb2-->
この積の 1 つの成分に注目すれば、スカラーの場合の経路上の和が復元される。すなわち $[\mathbf B\mathbf A]_{ik}=\sum_j (\partial g_i/\partial y_j)(\partial f_j/\partial x_k)$ は、すべての中間変数 $y_j$ にわたって、$x_k$ から $g_i$ への各経路に沿った導関数の積を足し合わせている。

<!--en:8b504874-->
### ネットワークのヤコビ行列

<!--en:68938a3f-->
冒頭のネットワークに戻ると、これは 3 つの写像の合成とみなせる。アフィン変換 $\mathbf a=\mathbf W_1\mathbf x+\mathbf b$、ReLU による活性化値 $\mathbf h=\boldsymbol\varphi(\mathbf a)$、そして線形出力 $g=\mathbf w_2^\top\mathbf h$ である。第 1 の写像はもともとアフィンであるため、そのヤコビ行列は $\mathbf W_1$ である。バイアスは傾きを変えずに値を平行移動させるにすぎない。ReLU は各成分に独立に作用し、傾きは $1$（$a_i>0$ のとき）および $0$（$a_i<0$ のとき）となるため、そのヤコビ行列は対角行列 $\operatorname{diag}(\mathbf m)$（これらの傾きからなる*マスク* $\mathbf m$ を対角に並べた行列）となる。最後に、出力写像のヤコビ行列は $\mathbf w_2^\top$ である。$\mathbf x=(1,1)$ においては $\mathbf a=(3,-2)$、$\mathbf m=(1,0)$ であり、連鎖律により 3 つの因子の積をとると次のようになる。

<!--en:1ead6a0f-->
マスクによって $\mathbf w_2^\top$ の第 2 成分が 0 になるため、$\mathbf W_1$ の第 2 行は出力に一切届かない。すなわち、非活性なユニットの重みは導関数にまったく寄与しない（:numref:`fig_mdl-cal-relu-net-jacobian`）。得られた行 $(2,4)$ は、冒頭で見た変化量 $2\delta_1+4\delta_2$ と一致している。入力が変わればマスクも変わり、ヤコビ行列も変化する。たとえば $\mathbf x=(2,-3)$ では活性化前の値が $(-4,5)$ となり、マスクは $(0,1)$ に反転して $\mathbf J_g=(0,\ -1)\,\mathbf W_1=(-3,\ 1)$ となる。ReLU は区分線形であるため、マスクが変わらないほど小さなステップをとれば、各軸に沿った有限差分によってこれらの成分が厳密に再現される（演習問題 :exref:`ex_mdl-matrix-calculus-relu-jacobian`）。

<!--en:e51639d5-->
![2-2-1 ReLU ネットワークの $\mathbf x=(1,1)$ におけるヤコビ行列は 3 つの行列の積である。第 2 ユニットの活性化前の値は $-2$ であるためマスク成分が $0$ となり、$\mathbf W_1$ の第 2 行は出力に届かない。結果として $\mathbf J_g=\mathbf w_2^\top\operatorname{diag}(1,0)\,\mathbf W_1=(2,\ 4)$ となる。](../img/mdl-cal-relu-net-jacobian.svg)

<!--en:20bd6e44-->
あらゆる多層パーセプトロン（multilayer perceptron、MLP）は、同じ 2 種類の因子から構成される。全結合層 $\mathbf x\mapsto\mathbf W\mathbf x$ のヤコビ行列は $\mathbf W$ である。要素ごとの活性化関数 $\boldsymbol\varphi(\mathbf x)=(\varphi(x_1),\ldots,\varphi(x_n))$ は異なる座標どうしを結合しないため、そのヤコビ行列は対角行列 $\mathbf J_{\boldsymbol\varphi}(\mathbf x)=\operatorname{diag}\bigl(\varphi'(x_1),\ldots,\varphi'(x_n)\bigr)$ となる。ここでは $\varphi$ がすべての $x_i$ で微分可能であると仮定している。ReLU は $0$ で導関数を持たないが、フレームワークでは慣例として固定値を割り当てる（:numref:`sec_mdl-tangent-fails`）。深さ $K$ のネットワークは合成 $\mathbf f=\mathbf f_K\circ\cdots\circ\mathbf f_1$ であり、:eqref:`eq_mdl-jacobian-chain` を繰り返し適用すると、その導関数は 1 つの積になる。

<!--en:407c25d6-->
ここで各因子は、それぞれの層への入力において評価される。全結合層と活性化関数が交互に並ぶ場合、これは $\mathbf J=\operatorname{diag}(\boldsymbol\varphi_K')\,\mathbf W_K\cdots \operatorname{diag}(\boldsymbol\varphi_1')\,\mathbf W_1$ となり、重み行列と対角行列が交互に現れる。ベクトルに対角行列を掛ける演算は各成分を定数倍するだけなので、各活性化因子の計算コストは $O(n)$ 回の演算で済み、$O(n^2)$ にはならない。

<!--en:d9a0f71e-->
ここで、本節の後半でも再利用する 2-2-1 ネットワークをコードで定義し、積 $\mathbf w_2^\top\operatorname{diag}(\mathbf m)\mathbf W_1$ を自動微分（automatic differentiation）が返す勾配と比較してみよう。

<!--en:d7a68f3f-->
予想どおり、どちらの行も $(2,4)$ を出力する。

<!--en:2baa70c8-->
積 :eqref:`eq_mdl-jacobian-product` からは、深いネットワークを通過するうちに導関数がどれほど大きく、あるいは小さくなりうるかもわかる。すべての因子が同じ行列 $\mathbf W$ である場合（活性化関数を無視して、毎ステップ同一の重み行列を適用するリカレントネットワークのような場合）、積は $\mathbf W^K$ に簡約される。対角化可能な行列 $\mathbf W=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$ に対しては $\mathbf W^K=\mathbf V\boldsymbol\Lambda^K\mathbf V^{-1}$ が成り立つため、大きな $K$ に対してそのノルムは $|\lambda_{\max}|^K$ のように増大または縮小する。ここで $\lambda_{\max}$ は絶対値が最大の固有値であり、$0.9^{50}\approx0.005$ であるのに対し、$1.1^{50}\approx117$ となる。この指数関数的なスケーリングこそが、深層ネットワークやリカレントネットワークにおける勾配消失・勾配爆発問題である（:numref:`sec_numerical_stability`、:numref:`sec_bptt`）。ただし、$|\lambda_{\max}|^K$ が表すのは長期的な変化率にすぎない。$\mathbf V$ の固有ベクトルが直交から程遠い場合、積は最終的に減衰する前に何ステップにもわたって増大しうるし（:numref:`subsec_mdl-transient-growth`）、異なる因子の積を支配するのは固有値ではなく特異値である（:numref:`subsec_mdl-jacobian-products`）。

<!--en:9b2e77ec-->
しかし、訓練という観点では :eqref:`eq_mdl-jacobian-product` に重要な限界がある。この式が与えるのは*入力*に関する導関数なのである。重み行列 $\mathbf W_k$ は変数ではなく固定された因子として現れているため、重みが変化したときに出力がどう変化するかはまだわからない。重み勾配を計算するには層ごとにあと 1 つの規則が必要であり、:numref:`subsec_mdl-layer-vjp` でそれを導出して同じネットワークに適用する。

<!--en:79443745-->
### 乗算の順序

<!--en:a3fdaf4e-->
行列積は結合法則を満たすため、積 :eqref:`eq_mdl-jacobian-product` はどのような順序で括弧を付けても厳密に同じ行列になる。しかし、括弧の付け方によって計算コストにはネットワークの幅に比例して拡大するほどの莫大な差が生じる。訓練で求めたいのはスカラー損失 $\ell$ の勾配であるため、連鎖の左端は行ベクトル、すなわち $1\times h$ のヤコビ行列 $\mathbf u^\top=\partial\ell/\partial\mathbf h_K$（最終層の出力 $\mathbf h_K$ に関する損失のヤコビ行列）から始まる。左から順に掛けていけば、中間結果は常に行ベクトルのまま保たれ、すべてのステップが行列・ベクトル積になる。反対に右から掛けていくと、最後にようやく行ベクトルが掛けられるまで、高コストな行列・行列積をずっと作り続けることになる。低コストな順序とは、すべてのステップを行列・ベクトル積に保つ順序である。

<!--en:4106138b-->
演算回数を簡単に数えれば、この差は具体的になる。$\mathbf u^\top\mathbf J_3\mathbf J_2\mathbf J_1$ の各因子の形状を $1\times h$、$h\times h$、$h\times h$、$h\times n$ とし、$h=n=1000$ とおく。なお、$p\times q$ 行列と $q\times r$ 行列の積には $pqr$ 回の乗算を要する。左から掛ける場合、3 回の行列・ベクトル積を行うため、乗算回数は合計 $h^2+h^2+hn=3\times10^6$ 回である。右から掛ける場合、$\mathbf J_2\mathbf J_1$ と続く $\mathbf J_3(\mathbf J_2\mathbf J_1)$ の計算にそれぞれ $h^2n=10^9$ 回の乗算を要し、最後の $\mathbf u^\top$ との積で $hn=10^6$ 回が加わって合計 $2.001\times10^9$ 回となる。300 万回対 20 億回であり、約 $670$ 倍、すなわち $2h/3$ 倍の差がある（:numref:`fig_mdl-cal-jacobian-chain-order`）。

<!--en:c504c6ba-->
![$\mathbf u^\top\mathbf J_3\mathbf J_2\mathbf J_1$（$h=n=1000$）に対する 2 通りの計算順序。左から掛けると中間結果はすべて $1\times h$ の行となり、各ステップが行列・ベクトル積となって乗算は $3\times10^6$ 回で済む。右から掛けると最初の 2 ステップがそれぞれ $10^9$ 回の乗算を要する行列・行列積になる。コストを決めるのは深さではなく形状である。](../img/mdl-cal-jacobian-chain-order.svg)

<!--en:c4f1a8a7-->
これら 2 つの順序は、ネットワークをたどる 2 つの向きに対応している。左からの乗算は、層あたり 1 回の行列・ベクトル積のコストで、出力の感度を層に沿って逆方向へ伝播させる。一方、右からの乗算が低コストになるのは、連鎖の右端に列ベクトル $\mathbf v$ がある場合であり、単一の入力方向を順方向へ伝播させて $\mathbf J\mathbf v$ を求める。これら 2 つの順序を自動化するアルゴリズムが、それぞれリバースモードおよびフォワードモードの自動微分と呼ばれる（:numref:`sec_mdl-matrix-calculus-autodiff`）。

<!--en:38e2c0b9-->
上の演算回数では、すべての層の幅が等しいと仮定した。各因子の形状が異なるとき、左端または右端から一方向に掛ける順序が最適とは限らない。連鎖状の積であれば、最小コストの括弧の付け方は動的計画法（dynamic programming）で求められるが（演習問題 :exref:`ex_mdl-matrix-calculus-chain-order`）、一般の計算グラフではこの問題は NP 完全である :cite:`Naumann.2008`。

<!--en:0029658b-->
## ベクトルと行列に関する導関数

<!--en:3de79ba0-->
層を訓練するには、入力、重み、バイアスといったベクトルと行列の双方に関してスカラー損失の勾配を求める必要がある。本項では、必要となるすべての勾配を単一の手法で導出する。この手法は、それぞれの結果を自動的に変数の形状で返す。

<!--en:43032a0d-->
### 全微分法

<!--en:0fef6d9f-->
変数の微小な摂動を $d\mathbf x$ と書き、$df$ をその結果生じる $f$ の変化のうち $d\mathbf x$ に関して線形な部分を表すものとする。これは単に、項 $\mathbf J\boldsymbol\delta$（:eqref:`eq_mdl-jacobian-linearization` において $\boldsymbol\delta$ を $d\mathbf x$ に書き換えたもの）にほかならない。:eqref:`eq_mdl-jacobian-linearization` と :eqref:`eq_mdl-matrix-gradient` より、この線形部分は勾配との内積となる。

<!--en:039cd052-->
これはベクトル変数（$\langle\mathbf a,\mathbf b\rangle=\mathbf a^\top\mathbf b$）と行列変数のどちらに対しても成り立つ。:eqref:`eq_mdl-differential-rule` は 2 通りに読むことができる。勾配が既知なら、微小変化に対して $f$ がどう応答するかを教えてくれる。一方、$f$ が与えられたなら、勾配を直接求める手順になる。すなわち、通常の微分規則を使って $df$ を計算し、$\langle\mathbf M,\,d\mathbf x\rangle$ の形に整理して、$\nabla f=\mathbf M$ を読み取るのである。この読み取りに曖昧さはない。なぜなら、$d\mathbf x$ の線形関数を $\langle\mathbf M,d\mathbf x\rangle$ と表す行列 $\mathbf M$ はただ 1 つしかないからである。実際、$d\mathbf x$ を、ただ 1 つの $1$ を $(i,j)$ 成分に持ち他がすべて 0 である行列にとれば、$M_{ij}$ が取り出せる。

<!--en:d7cf8408-->
$df$ の計算には、おなじみの 4 つの規則があれば十分である。第 1 に、微分は線形である。第 2 に、積の微分法則 $d(\mathbf A\mathbf B)=d\mathbf A\,\mathbf B+\mathbf A\,d\mathbf B$ に従う。なぜなら、$(\mathbf A+d\mathbf A)(\mathbf B+d\mathbf B)$ を展開した式と $\mathbf A\mathbf B+d\mathbf A\,\mathbf B+\mathbf A\,d\mathbf B$ との差は、二次の項 $d\mathbf A\,d\mathbf B$ だけだからである。第 3 に、転置と可換であり、$d(\mathbf A^\top)=(d\mathbf A)^\top$ が成り立つ。第 4 に、成分ごとに適用されるスカラー関数は各成分でスカラーの連鎖律に従い、$d\varphi(\mathbf X)=\varphi'(\mathbf X)\odot d\mathbf X$ となる。ここで $\odot$ は要素ごとの乗算を表す。

<!--en:4c3fabeb-->
$df$ を $\langle\mathbf M,d\mathbf x\rangle$ の形に整理するには、もう 1 つだけ道具があればよい。*移動則*（move rule）は、$d\mathbf x$ の隣に掛かっている因子を内積の反対側へと移す規則である。

<!--en:61e38222-->
これはトレースの巡回性 $\operatorname{tr}(\mathbf X\mathbf Y)=\operatorname{tr}(\mathbf Y\mathbf X)$ から直接導かれる。実際、$\operatorname{tr}(\mathbf A^\top\mathbf B\mathbf C)=\operatorname{tr}\bigl((\mathbf B^\top\mathbf A)^\top\mathbf C\bigr) =\operatorname{tr}(\mathbf C\mathbf A^\top\mathbf B)=\operatorname{tr}\bigl((\mathbf A\mathbf C^\top)^\top\mathbf B\bigr)$ が成り立つ。言葉で言えば、変数の左にある因子は転置をとって相手側の左へ移り、右にある因子は転置をとって相手側の右へ移る。要素ごとの積の因子はそのまま移り、$\langle\mathbf A,\mathbf B\odot\mathbf C\rangle=\langle\mathbf A\odot\mathbf B,\mathbf C\rangle$ となる。両辺とも $\sum_{ij}A_{ij}B_{ij}C_{ij}$ に等しいからである。

<!--en:e93aad74-->
最も単純な 2 つの例は、それぞれ 1 行で済む。線形形式では $d(\mathbf a^\top\mathbf x)=\mathbf a^\top d\mathbf x=\langle\mathbf a,d\mathbf x\rangle$ より $\nabla_{\mathbf x}\,\mathbf a^\top\mathbf x=\mathbf a$ である。二次形式では、積の微分法則が 2 つの $\mathbf x$ の両方に作用する。

<!--en:5ebf0cf9-->
2 つ目の等号では $1\times1$ 行列の転置をとった。$d\mathbf x^\top\mathbf A\mathbf x$ はスカラーなので、自身の転置 $\mathbf x^\top\mathbf A^\top d\mathbf x$ に等しい。係数を読み取れば $\nabla_{\mathbf x}\,\mathbf x^\top\mathbf A\mathbf x=(\mathbf A+\mathbf A^\top)\mathbf x$ が得られ、これは $2\mathbf A\mathbf x$ に簡約される（$\mathbf A$ が対称な場合）。

<!--en:259581d6-->
添字を書き下すことでも同じ結果に至る。成分ごとに計算すると、同じ導関数は $\partial\bigl(\sum_{ij}x_iA_{ij}x_j\bigr)/\partial x_k=\sum_jA_{kj}x_j+\sum_ix_iA_{ik}$ となり、第 $k$ 成分として $(\mathbf A+\mathbf A^\top)\mathbf x$ の対応する成分が復元される。添字による導出は長くなるが、転置に関して疑問の余地が残らない。:numref:`sec_backprop` では添字を使って行列積の規則を導出している。

<!--en:a9834374-->
任意の行列恒等式に対する手早い妥当性確認は、すべての行列を $1\times1$ に置き換えることである。その場合、式はおなじみのスカラーの公式に帰着しなければならない。先ほどの二次形式の勾配は $\frac{d}{dx}(ax^2)=2ax=(a+a)x$ となるため、この確認を通過する。この *$1\times1$ チェック*は係数の誤りや項の抜けを確実に検出するが、転置や因子の順序には無力である。$1\times1$ 行列は可換であり、自身の転置に等しいからである。形状規則はたいていの転置ミスを検出してくれるが、残念ながら因子が正方行列のときには機能しない。これこそ本節の冒頭で警告した落とし穴である。正方の因子に対しては、数値チェックでしか誤りを見つけられない。$\mathbf A=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$、$\mathbf x=(1,2)$ としよう。このとき $\mathbf x^\top\mathbf A\mathbf x=x_1x_2$ であり、真の勾配 $(x_2,x_1)=(2,1)$ は $(\mathbf A+\mathbf A^\top)\mathbf x$ に一致する。一方、素朴な推測 $2\mathbf A\mathbf x=(4,0)$（スカラーの公式 $2ax$ から連想される式）は、$1\times1$ チェックを通過し、形状も正しいにもかかわらず、やはり誤りである。

<!--en:67c6a4bd-->
全微分法は、行列変数も追加の手間なく扱える。最小二乗法の損失 $\ell=\|\mathbf W\mathbf x-\mathbf y\|^2$ において残差を $\mathbf r=\mathbf W\mathbf x-\mathbf y$ とおくと、$\mathbf W$ を変化させたとき $d\mathbf r=d\mathbf W\,\mathbf x$ となり、移動則によって $\mathbf x$ が反対側へ移る。すなわち $d\ell=2\mathbf r^\top d\mathbf r=\langle 2\mathbf r,\ d\mathbf W\,\mathbf x\rangle =\langle2\mathbf r\mathbf x^\top,\ d\mathbf W\rangle$ であり、したがって次式を得る。

<!--en:714d5ea5-->
これは残差と入力のランク 1 の外積であり、自動的に $\mathbf W$ の形状になっている。代わりに $\mathbf x$ を変化させると、$d\mathbf r=\mathbf W\,d\mathbf x$ より $\nabla_{\mathbf x}\ell=2\mathbf W^\top\mathbf r$ となる。どちらの場合も、勾配は残差の 2 倍に、変数に掛かっている因子の転置を掛けたものになっている。

<!--en:8706aec4-->
### 層の逆伝播則

<!--en:fa1d68a4-->
ネットワークの内部では、層 $\mathbf Y=\mathbf F(\mathbf X)$ はスカラー損失 $\ell$ に至る長い計算の 1 ステップである。その完全なヤコビ行列を作るのは現実的ではない。$1000\times1000$ の重み行列を $1000$ 個の例からなるミニバッチに適用すると、$10^6$ 個の出力が $10^6$ 個の重みから生成されるため、重みに関する出力のヤコビ行列は $10^{12}$ 個もの成分を持つことになる。幸い、誤差逆伝播法が必要とする量ははるかに少ない。層がすべきことは、自身の出力に関する損失の勾配を、入力と重みに関する勾配へと変換することだけである。:numref:`subsec_mdl-backpropagation` に従い、$\bar{\mathbf Y}=\nabla_{\mathbf Y}\ell$ と書いて $\mathbf Y$ の随伴変数（adjoint）を表す。これは $\mathbf Y$ と同じ形状の行列であり、$\mathbf Y$ の各成分に関する損失の全微分を保持する。

<!--en:04dfef0b-->
**命題（層の逆伝播則）** *$\mathbf Y=\mathbf F(\mathbf X)$ が $\mathbf X$ で微分可能であり、$\ell$ が $\mathbf Y$ の微分可能なスカラー関数であるとする。このとき、$\bar{\mathbf X}$（合成関数 $\ell(\mathbf F(\mathbf X))$ の随伴変数）は、次式を満たす一意な行列である。*

<!--en:601c1b10-->
*ここで、$d\mathbf Y$ は $\mathbf F$ において $d\mathbf X$ が引き起こす一次の変化である。*

<!--en:4c2f99ea-->
**証明** 連鎖律 :eqref:`eq_mdl-jacobian-chain` により（行列を平坦化したベクトルとみなせば）、合成関数の線形部分は、$\ell$ の線形部分を $\mathbf F$ の線形部分に適用したものとなり、$d\ell=\langle\bar{\mathbf Y},d\mathbf Y\rangle$ が成り立つ。$d\mathbf Y$ は $d\mathbf X$ に関して線形であるため、$d\ell$ もそうであり、$\langle\mathbf M,d\mathbf X\rangle$ の形で一意な行列 $\mathbf M$ を用いて表せる。:eqref:`eq_mdl-differential-rule` より、$\mathbf M=\bar{\mathbf X}$ である。$\blacksquare$

<!--en:cafb61c0-->
実際には、この命題の適用は移動則の適用に帰着する。$d\mathbf y=\mathbf J\,d\mathbf x$ を満たすベクトル層に対しては、$\langle\bar{\mathbf y},\mathbf J\,d\mathbf x\rangle=\langle\mathbf J^\top\bar{\mathbf y},d\mathbf x\rangle$ となるため、$\bar{\mathbf x}=\mathbf J^\top\bar{\mathbf y}$ を得る。すなわち、層の逆伝播則はその線形化の転置にほかならない。線形変換の転置は随伴（adjoint）とも呼ばれ、随伴変数という名称はこれに由来する。$\bar{\mathbf x}^\top=\bar{\mathbf y}^\top\mathbf J$ と書き直せば、これは :eqref:`eq_mdl-jacobian-product` を左から右へ掛ける順序の 1 ステップそのものである。行ベクトルとヤコビ行列の積は*ベクトル・ヤコビ行列積*（vector–Jacobian product）と呼ばれる。この命題により、行列変数に対してもヤコビ行列を一度も作ることなくこれを計算できる。

<!--en:0eaf380a-->
深層学習で最も頻繁に現れる導関数は、行列積の導関数である。ミニバッチを処理するか単一のベクトルを処理するかによって重みはデータの右側または左側に置かれるため、両方の配置をまとめて導出しよう。1 行に 1 つの例を並べたミニバッチ $\mathbf X\in\mathbb R^{n\times d}$（本文全体の規約）に重み行列 $\mathbf W\in\mathbb R^{d\times q}$ を掛けると、$\mathbf Y=\mathbf X\mathbf W$ が得られる。積の微分法則より $d\mathbf Y=d\mathbf X\,\mathbf W+\mathbf X\,d\mathbf W$ となり、移動則によって内積は変数ごとの項に分解される。すなわち $\langle\bar{\mathbf Y},d\mathbf X\,\mathbf W\rangle+\langle\bar{\mathbf Y},\mathbf X\,d\mathbf W\rangle =\langle\bar{\mathbf Y}\mathbf W^\top,d\mathbf X\rangle+\langle\mathbf X^\top\bar{\mathbf Y},d\mathbf W\rangle$ であり、両方の係数を読み取ると次式を得る。

<!--en:7b52796b-->
単一の例を列ベクトルとして $\mathbf y=\mathbf W\mathbf x$ と書いた場合も、まったく同じ計算から次式が得られる。

<!--en:77b78cc0-->
これは :eqref:`eq_mdl-lstsq-grad` の形と一致しており、重み勾配は出力の随伴変数と入力の外積になる。2 つの配置は単に互いの転置である。$\mathbf x_i^\top$ と $\bar{\mathbf y}_i^\top$ をそれぞれ $\mathbf X$ と $\bar{\mathbf Y}$ の行とすれば、ミニバッチの公式 $\mathbf X^\top\bar{\mathbf Y}=\sum_i\mathbf x_i\bar{\mathbf y}_i^\top$ は例ごとにこの外積を足し合わせたものになっている。

<!--en:7309c379-->
したがって、線形層の逆伝播は順伝播の積と同じサイズの 2 つの行列積からなり、順伝播は $ndq$ 回の乗算、逆伝播は $2ndq$ 回の乗算を要する。形状を確認すれば両方の転置も確かめられる。$\bar{\mathbf Y}\mathbf W^\top$ は $n\times d$ で $\mathbf X$ と一致し、$\mathbf X^\top\bar{\mathbf Y}$ は $d\times q$ で $\mathbf W$ と一致する。より複雑な行列演算（逆行列、行列式、各種の行列分解）の逆伝播則も、まったく同じ手法から導かれる :cite:`Giles.2008b`。

<!--en:b65c8f43-->
あと 2 種類の層が揃えば、多層パーセプトロンが完成する。要素ごとの活性化値 $\mathbf Y=\boldsymbol\varphi(\mathbf X)$ は $d\mathbf Y=\varphi'(\mathbf X)\odot d\mathbf X$ を満たすため、$\varphi'(\mathbf X)$ を内積の反対側へ移すと $\bar{\mathbf X}=\varphi'(\mathbf X)\odot\bar{\mathbf Y}$ が得られる。これは先ほどの対角ヤコビ行列をそのままマスクとして適用したものである。バイアスの加算 $\mathbf y=\mathbf z+\mathbf b$ では $d\mathbf y=d\mathbf b$ より $\bar{\mathbf b}=\bar{\mathbf y}$ となる。ミニバッチに対しては、バイアスは $\mathbf Z=\mathbf X\mathbf W+\mathbf 1\mathbf b^\top$ によって各行にブロードキャストされるが、これも単なる行列積であり、すべての例にわたって随伴変数の和をとる $\bar{\mathbf b}=\bar{\mathbf Z}^\top\mathbf 1$ を与える。これら 3 つの局所規則が基本要素となり、:numref:`sec_backprop` では多層パーセプトロン全体の逆伝播が組み立てられる。

<!--en:416a7444-->
これらの規則を使えば、冒頭の重み勾配行列も導出できる。:numref:`fig_mdl-cal-relu-net-jacobian` のネットワークを、$\bar g=1$ から 1 層ずつ逆向きに $\mathbf x=(1,1)$ においてたどると次のようになる。

<!--en:039dbbd0-->
1. 出力 $g=\mathbf w_2^\top\mathbf h$ から、$\bar{\mathbf h}=\mathbf w_2=(2,-1)$ および $\bar{\mathbf w}_2=\mathbf h=(3,0)$ が得られる。

<!--en:f0ee916a-->
2. ReLU から $\bar{\mathbf a}=\mathbf m\odot\bar{\mathbf h}=(2,0)$ が得られ、バイアスから $\bar{\mathbf b}=\bar{\mathbf a}=(2,0)$ が得られる。

<!--en:032d0326-->
3. アフィン写像 $\mathbf a=\mathbf W_1\mathbf x+\mathbf b$ から、$\bar{\mathbf W}_1=\bar{\mathbf a}\,\mathbf x^\top=\left(\begin{smallmatrix}2&2\\0&0\end{smallmatrix}\right)$ および $\bar{\mathbf x}=\mathbf W_1^\top\bar{\mathbf a}=(2,4)$ が得られる。

<!--en:f6f4346c-->
$\bar{\mathbf x}$ は先ほど求めたヤコビ行列の行 $(2,4)$ の転置になっており、$\bar{\mathbf W}_1$ は冒頭で重みを微小変化させて得た行列そのものである。コード上で `g` を $\mathbf W_1$、$\mathbf b$、$\mathbf w_2$ に関して微分すると、3 つの配列すべてが一致することが確認できる。

<!--en:c07086a7-->
どこにゼロが現れるかにも注目しておきたい。非活性な隠れユニットは 3 か所にゼロを残す。$\bar{\mathbf W}_1$ のゼロ行、$\bar{\mathbf b}$ のゼロ成分、そして $\bar{\mathbf w}_2$ のゼロ成分である（その出力 $h_2=0$ が $w_{2,2}$ に掛かるためである）。この入力に対する訓練ステップでは、そのユニットに属する 4 つのパラメータはすべて更新されないまま残る。

<!--en:ab481cff-->
正方の因子に対する転置の誤りを検出する数値テストがまだ残っていた。失敗しえないテストは何も証明しないため、ここでは誤りが露呈するようにテストを設計する。以下の各因子はすべてランダムな正方行列（したがって非対称行列）とし、正しい規則の隣に同じ形状のもっともらしい誤答候補を並べて出力する。二次形式と行列積の両方をテストしよう。行列積については、スカラー $\langle\bar{\mathbf Y},\mathbf X\mathbf W\rangle$ をランダムな $\bar{\mathbf Y}$ に対して微分する。逆伝播則の命題により、$\mathbf X$ と $\mathbf W$ に関するその勾配はまさに $\bar{\mathbf X}$ と $\bar{\mathbf W}$ になる。

<!--en:74ac018c-->
導出した規則は `True` を出力し、誤答候補は `False` を出力する。ここではテストの結果以上に、テストの設計が重要である。長方行列の因子、たとえば $\mathbf X$ を形状 $3\times4$、$\mathbf W$ を形状 $4\times2$ とした場合、積に対する 2 つの誤答候補は実行すらできない。転置の位置を誤ると数値を比較する前に形状エラーが発生するが、これは形状規則が役割を果たした結果である。同様に、対称な $\mathbf A$ を使うと、誤答候補 $2\mathbf A\mathbf x$ は真の勾配と一致してテストをすり抜けてしまう。厄介なのは正方かつ非対称な因子であり、数値チェックではまさにそれを生成しなければならない。

<!--en:d0ceddc4-->
行列積の規則を使えば、積の両方の因子が未知である損失、たとえば行列因子分解（matrix factorization）の損失 $\|\mathbf X-\mathbf U\mathbf V\|_F^2$（演習問題 :exref:`ex_mdl-matrix-calculus-matmul-rules`）も同じように容易に扱える。いずれの場合も、逆伝播則は順伝播でその変数に掛かっていた因子の転置を受け取った随伴変数に掛ける。また、各規則が順伝播のどの値を必要とするかにも注目したい。行列積では $\mathbf X$ と $\mathbf W$、活性化では $\varphi'(\mathbf X)$ が必要であり、バイアスでは何も必要としない。逆伝播ではこれらの中間値を順伝播から保持しておく必要があり、そのメモリコストは :numref:`sec_mdl-matrix-calculus-autodiff` で直接計測する。

<!--en:5acc5c1a-->
### ソフトマックスと交差エントロピー

<!--en:d092ce06-->
分類器が最も頻繁に評価する導関数は、ソフトマックスと交差エントロピー損失を合成した関数の導関数である。ソフトマックスはスコアのベクトル $\mathbf z$、すなわち*ロジット*を確率ベクトルへ変換する。$S=\sum_k e^{z_k}$ とおけば $p_i=e^{z_i}/S$ である。2 つのロジットに対して :numref:`sec_mdl-multivariable_calculus` で log-sum-exp のヘッセ行列を導いたのと同じ商の微分法則から、任意のクラス数に対するソフトマックスのヤコビ行列が得られる。$\delta_{ij}=1$（$i=j$ のとき）、それ以外では $0$ と書けば、次式が得られる。

<!--en:3d2707f7-->
言葉で言えば、ロジット $i$ を増やすと $p_i$ は変化率 $p_i(1-p_i)$ で増加し、他のすべての確率 $p_j$ は変化率 $p_ip_j$ で減少する。簡単な検算として、確率の和は 1 であるため $\mathbf J\mathbf 1=\mathbf p-\mathbf p(\mathbf p^\top\mathbf 1)=\mathbf 0$ となる。すべてのロジットに同じ定数を加えても $\mathbf p$ は変化しないため、これは当然成り立たなければならない。結果として、ソフトマックスのヤコビ行列は常に特異である。

<!--en:9bcc8f9a-->
また、$\mathbf J_{\operatorname{softmax}}$ が対称であることにも注目したい。一般のヤコビ行列は対称とは限らないが、これが対称になるのは実際にはヘッセ行列だからである。すなわち、ソフトマックスは本章で繰り返し用いている例である log-sum-exp 関数 $\operatorname{LSE}(\mathbf z)=\log\sum_k e^{z_k}$ の勾配であり（$\partial\operatorname{LSE}/\partial z_i=e^{z_i}/S=p_i$）、勾配のヤコビ行列はその関数のヘッセ行列に一致する。2 ロジットの公式に :eqref:`eq_mdl-softmax-jacobian` が付け加えるのは、この行列の明快な統計的解釈である。これは確率 $\mathbf p$ に従って抽出されるワンホットベクトルの共分散行列になっており、したがって任意の $\mathbf z$ と任意のベクトル $\mathbf v$ に対して、二次形式 $\mathbf v^\top\mathbf J\mathbf v=\sum_ip_iv_i^2-\bigl(\sum_ip_iv_i\bigr)^2$ は $\mathbf v$ の各成分の、分布 $\mathbf p$ の下での分散になる。分散は決して負にならないため、このヘッセ行列はいたるところで半正定値（positive semidefinite）となり、log-sum-exp が凸性を持つことが証明される。定数ベクトル $\mathbf v$ の分散は 0 であり、ここからも零ベクトルを与える方向 $\mathbf 1$ が再び得られる。

<!--en:a37d3975-->
ここで、ソフトマックスと交差エントロピー損失 $\ell=-\sum_i y_i\log p_i$（ターゲット分布 $\mathbf y$ に対する損失）を合成しよう。連鎖律により行ベクトル $\partial\ell/\partial\mathbf p=(-y_i/p_i)_i$ に :eqref:`eq_mdl-softmax-jacobian` を掛けると、式は劇的に簡約される。

<!--en:d8179c49-->
ロジットに関する勾配は、単に予測確率からターゲットを引いたものになり、これは $\hat{\mathbf y}-\mathbf y$ として :numref:`subsec_softmax_and_derivatives` で書いた公式そのものである。最後の変形では $\sum_iy_i=1$ しか使っていないため、この結果はワンホットラベルだけでなく任意の確率ベクトル $\mathbf y$ に対して成り立つ。したがって、ラベル平滑化（label smoothing）や蒸留のターゲットもまったく同じ公式に従う。和が 1 にならないターゲットに対しては、勾配は代わりに $(\sum_iy_i)\,\mathbf p-\mathbf y$ となる。たとえば $\mathbf z=(1,-0.5,2)$ では $\mathbf p=(0.254,\ 0.057,\ 0.690)$ であり、第 2 クラスをターゲットとするとき、勾配は $(0.254,\ -0.943,\ 0.690)$ となる。

<!--en:14f97036-->
実際のライブラリでは、数値安定性のためにソフトマックスと交差エントロピーを単一のプリミティブに融合している（:numref:`subsec_softmax-implementation-revisited`、:numref:`subsec_mdl-stable-softmax`）。以下では、自動微分によって $\mathbf z$ におけるソフトマックスのヤコビ行列を計算し、その固有値を出力する。分散による議論から予測されるとおり、固有値は非負で 1 つが 0 になるはずである。続いて、融合された損失を、先ほどのワンホットターゲットと、成分の和が 1 にならないツーホットターゲット $\mathbf y=(1,1,0)$ の両方に対して微分する。

<!--en:45ba0539-->
固有値は $0$、$0.079$、$0.378$ となり、方向 $\mathbf 1$ に対応する 1 つのゼロと 2 つの正の値が得られた。ワンホットターゲットに対しては、融合された損失の勾配は $\mathbf p-\mathbf y=(0.2537,-0.9434,0.6897)$ となり、手計算と一致する。しかしツーホットターゲットに対しては $(-0.4926,-0.8868,1.3793)$ を返し、これは $2\mathbf p-\mathbf y$ であって、隣に出力された $\mathbf p-\mathbf y$ ではない。「予測からターゲットを引いた形」になるのは和が 1 になるターゲットの性質であり、損失単独の性質ではない。

<!--en:2673f521-->
本節では、単純な 3 層の連鎖に対して $\mathbf J_K\cdots\mathbf J_1$ を手計算で掛け合わせた。しかし実際には、プログラムによって分岐や合流を含む任意の計算グラフを扱い、各逆伝播則が必要とする順伝播の値を保持し、パラメータがどれほど多くてもネットワークの評価の小さな定数倍にとどまるコストですべての勾配を計算しなければならない。そのプログラムこそが自動微分であり、:numref:`sec_mdl-matrix-calculus-autodiff` の主題である。

<!--en:30ac03ff-->
## まとめ

<!--en:e9db935f-->
写像 $\mathbf f:\mathbb R^n\to\mathbb R^m$ のヤコビ行列とは、その最良線形近似を表す $m\times n$ 行列であり、各成分は $[\mathbf J_{\mathbf f}]_{ij}=\partial f_i/\partial x_j$ で与えられる。勾配（1 行のベクトルを列として書いたもの）もヘッセ行列（勾配のヤコビ行列）も、その特別な場合である。入力に関するネットワークの導関数は各層のヤコビ行列の積 $\mathbf J_K\cdots\mathbf J_1$ となり、乗算の順序がコストを左右する。幅 $1000$ の 3 層では、300 万回対 20 億回の乗算という差が生じる。全微分則 $df=\langle\nabla f,d\mathbf x\rangle$ を用いれば、行列積の逆伝播則 $\bar{\mathbf X}=\bar{\mathbf Y}\mathbf W^\top$、$\bar{\mathbf W}=\mathbf X^\top\bar{\mathbf Y}$ や、ソフトマックスと交差エントロピーの合成が $\mathbf p-\mathbf y$ に簡約される結果を含め、層が必要とするあらゆる勾配が自動的に変数の形状で導出される。手計算でのうっかりミスの大半は形状規則と $1\times1$ チェックで検出できるが、正方の因子における転置の誤りは両方をすり抜けるため、非対称行列を用いた数値テストが必要になる。

<!--en:2294243a-->
## 演習問題

<!--en:ccf39f10-->
1. **ReLU ネットワークのヤコビ行列。**$\mathbf g(\mathbf x)=\mathbf W_2\,\boldsymbol\varphi(\mathbf W_1\mathbf x+\mathbf b)$ とし、$\mathbf W_1\in\mathbb R^{h\times n}$、$\mathbf b\in\mathbb R^h$、$\mathbf W_2\in\mathbb R^{m\times h}$、さらに要素ごとの ReLU を $\boldsymbol\varphi$ とする。活性化前の値 $\mathbf W_1\mathbf x+\mathbf b$ のどの成分も 0 でないと仮定する。:eqref:`eq_mdl-jacobian-chain` より $\mathbf J_{\mathbf g}(\mathbf x)=\mathbf W_2\mathbf M(\mathbf x)\mathbf W_1$ であり、ここで対角行列 $\mathbf M(\mathbf x)$ は活性な各隠れユニットに対して成分 $1$ を、非活性な各隠れユニットに対して成分 $0$ を持つ。
    1. $\mathbf g(\mathbf x+\boldsymbol\delta)=\mathbf g(\mathbf x)+\mathbf J_{\mathbf g}(\mathbf x)\,\boldsymbol\delta$ が十分小さな任意の $\boldsymbol\delta$ に対して厳密に成り立ち、したがって :eqref:`eq_mdl-jacobian-linearization` の剰余項が恒等的に消えることを示せ。本節の 2-2-1 ネットワークの $\mathbf x=(1,1)$ において、最大の $r$ として、すべての $\boldsymbol\delta$（$\|\boldsymbol\delta\|_\infty<r$）に対してこの等式が成り立つ値を求めよ。$n=m=2$ のとき、小さな円の $\mathbf g$ による像は :numref:`fig_mdl-cal-jacobian-ellipse` の像とどう異なるか。
    1. $k$ 個の隠れユニットが $\mathbf x$ において活性であるとする。$\operatorname{rank}\mathbf J_{\mathbf g}(\mathbf x)\le k$ を示せ。$n=m=2$ かつ $k=1$ のとき、$\mathbf x$ の周りの小さな円の像はどうなるか。
    1. いずれかの活性化前の値が 0 になる入力の集合を記述せよ。$\mathbf J_{\mathbf g}$ は、$\mathbf x$ がこの集合を横切るときどう変化するか。
   :exlabel:`ex_mdl-matrix-calculus-relu-jacobian`

<!--en:2d87440b-->
1. [code] **最小コストの連鎖順序。**● :eqref:`eq_mdl-jacobian-product` より、$K$ 個の写像の連鎖のヤコビ行列は $\mathbf J_K\cdots\mathbf J_1$（ただし $\mathbf J_k\in\mathbb R^{d_k\times d_{k-1}}$）である。$p\times q$ 行列と $q\times r$ 行列の乗算には $pqr$ 回の乗算を要する。
    1. 右から掛ける順方向順序のコストと、左から掛ける逆方向順序のコストを、$d_0,\ldots,d_K$ の関数として表せ。広い層と狭い層が交互に並ぶ写像 $(d_0,\ldots,d_4)=(1000,10,1000,10,1000)$ に対して両者を評価し、本節で求めた $(1\times h)(h\times h)(h\times h)(h\times n)$ の演算回数と比較せよ。
    1. 部分連鎖 $\mathbf J_j\cdots\mathbf J_i$ 上の動的計画法によって最小コストの括弧の付け方を求めるプログラムを書き、(a) の次元に対して実行せよ。どの順序が選ばれ、(a) の 2 つの順序のうち優れた方と比べてどれだけ低コストになるか。
    1. スカラー出力 $d_K=1$ の場合について、数千個のランダムな次元ベクトル（$K$ は $2$ から $6$ の範囲とし、$d_0,\ldots,d_{K-1}$ の各成分を $\{1,\ldots,10\}$ から一様に抽出する）上で最適順序と逆方向順序を比較せよ。逆方向以外の順序の方が低コストになる次元ベクトルを特徴づけよ。
   :exlabel:`ex_mdl-matrix-calculus-chain-order`

<!--en:9ab95190-->
3. [code] **行列積の逆伝播則。**
    1. $\mathbf Y=\mathbf X\mathbf W$ に対して :eqref:`eq_mdl-matmul-vjp` を全微分法で導出し、さらに添字を用いて再度導出して転置を確認せよ。
    1. ベクトル・ヤコビ行列積 `backward(head_grad)` を用い、$\mathbf X\in\mathbb R^{3\times4}$、$\mathbf W\in\mathbb R^{4\times2}$ およびランダムな随伴変数 $\bar{\mathbf Y}$ に対して両方の規則を検証せよ。順伝播の積と逆伝播の 2 つの積それぞれの乗算回数を数えよ。
    1. 行列因子分解の損失 $\|\mathbf X-\mathbf U\mathbf V\|_F^2$（$\mathbf R=\mathbf X-\mathbf U\mathbf V$）について、$\nabla_{\mathbf U}=-2\mathbf R\mathbf V^\top$ および $\nabla_{\mathbf V}=-2\mathbf U^\top\mathbf R$ を数値的に確認せよ。次に、一方の因子を固定して他方の勾配を 0 とおく交互最小二乗法の更新ステップを書き下し、各ステップが 1 回の線形方程式の求解になることを示せ。
   :exlabel:`ex_mdl-matrix-calculus-matmul-rules`

<!--en:099dd688-->
3. [code] **行列積の逆伝播則。**
    1. $\mathbf Y=\mathbf X\mathbf W$ に対して :eqref:`eq_mdl-matmul-vjp` を全微分法で導出し、さらに添字を用いて再度導出して転置を確認せよ。
    1. ベクトル・ヤコビ行列積 `torch.func.vjp` を用い、$\mathbf X\in\mathbb R^{3\times4}$、$\mathbf W\in\mathbb R^{4\times2}$ およびランダムな随伴変数 $\bar{\mathbf Y}$ に対して両方の規則を検証せよ。順伝播の積と逆伝播の 2 つの積それぞれの乗算回数を数えよ。
    1. 行列因子分解の損失 $\|\mathbf X-\mathbf U\mathbf V\|_F^2$（$\mathbf R=\mathbf X-\mathbf U\mathbf V$）について、$\nabla_{\mathbf U}=-2\mathbf R\mathbf V^\top$ および $\nabla_{\mathbf V}=-2\mathbf U^\top\mathbf R$ を数値的に確認せよ。次に、一方の因子を固定して他方の勾配を 0 とおく交互最小二乗法の更新ステップを書き下し、各ステップが 1 回の線形方程式の求解になることを示せ。
   :exlabel:`ex_mdl-matrix-calculus-matmul-rules`

<!--en:c364a51f-->
3. [code] **行列積の逆伝播則。**
    1. $\mathbf Y=\mathbf X\mathbf W$ に対して :eqref:`eq_mdl-matmul-vjp` を全微分法で導出し、さらに添字を用いて再度導出して転置を確認せよ。
    1. ベクトル・ヤコビ行列積 `tape.gradient(..., output_gradients=...)` を用い、$\mathbf X\in\mathbb R^{3\times4}$、$\mathbf W\in\mathbb R^{4\times2}$ およびランダムな随伴変数 $\bar{\mathbf Y}$ に対して両方の規則を検証せよ。順伝播の積と逆伝播の 2 つの積それぞれの乗算回数を数えよ。
    1. 行列因子分解の損失 $\|\mathbf X-\mathbf U\mathbf V\|_F^2$（$\mathbf R=\mathbf X-\mathbf U\mathbf V$）について、$\nabla_{\mathbf U}=-2\mathbf R\mathbf V^\top$ および $\nabla_{\mathbf V}=-2\mathbf U^\top\mathbf R$ を数値的に確認せよ。次に、一方の因子を固定して他方の勾配を 0 とおく交互最小二乗法の更新ステップを書き下し、各ステップが 1 回の線形方程式の求解になることを示せ。
   :exlabel:`ex_mdl-matrix-calculus-matmul-rules`

<!--en:c0ec462d-->
3. [code] **行列積の逆伝播則。**
    1. $\mathbf Y=\mathbf X\mathbf W$ に対して :eqref:`eq_mdl-matmul-vjp` を全微分法で導出し、さらに添字を用いて再度導出して転置を確認せよ。
    1. ベクトル・ヤコビ行列積 `jax.vjp` を用い、$\mathbf X\in\mathbb R^{3\times4}$、$\mathbf W\in\mathbb R^{4\times2}$ およびランダムな随伴変数 $\bar{\mathbf Y}$ に対して両方の規則を検証せよ。順伝播の積と逆伝播の 2 つの積それぞれの乗算回数を数えよ。
    1. 行列因子分解の損失 $\|\mathbf X-\mathbf U\mathbf V\|_F^2$（$\mathbf R=\mathbf X-\mathbf U\mathbf V$）について、$\nabla_{\mathbf U}=-2\mathbf R\mathbf V^\top$ および $\nabla_{\mathbf V}=-2\mathbf U^\top\mathbf R$ を数値的に確認せよ。次に、一方の因子を固定して他方の勾配を 0 とおく交互最小二乗法の更新ステップを書き下し、各ステップが 1 回の線形方程式の求解になることを示せ。
   :exlabel:`ex_mdl-matrix-calculus-matmul-rules`

<!--en:871cc9b3-->
4. **ソフトマックスのベクトル・ヤコビ行列積。**逆伝播ではソフトマックスのヤコビ行列 :eqref:`eq_mdl-softmax-jacobian` を陽に作ることは決してなく、確率の随伴変数 $\mathbf u=\bar{\mathbf p}$ をロジットの随伴変数 $\bar{\mathbf z}=\mathbf J_{\operatorname{softmax}}^\top\mathbf u$ へ直接写す。
    1. $\mathbf J_{\operatorname{softmax}}^\top\mathbf u=\mathbf p\odot\bigl(\mathbf u-(\mathbf p^\top\mathbf u)\,\mathbf 1\bigr)$ を示せ（$\odot$ は要素ごとの積）。$n$ クラスの場合について、この式の演算回数と、$\mathbf J_{\operatorname{softmax}}$ を作って $\mathbf u$ を掛ける場合の演算回数をそれぞれ数えよ。
    1. 自己アテンションは $n\times n$ のスコア行列 $\mathbf S$ の各行にソフトマックスを適用して重み $\mathbf P$ を生成する。$\mathbf P$ の（$\mathbf S$ に関する）ヤコビ行列は何個の成分を持ち、そのうち非ゼロになりうるのは何個か。float32 で $n=4096$ トークンの場合について、完全なヤコビ行列、その非ゼロ成分、および (a) の式を行ごとに適用して返す $\mathbf S$ の随伴変数が要するメモリを比較せよ。

<!--en:cefb1804-->
1. [code] **LayerNorm のヤコビ行列。**層正規化 :cite:`Ba.Kiros.Hinton.2016` は、$\mathbf x\in\mathbb R^n$ を $\hat{\mathbf x}=(\mathbf x-\mu\mathbf 1)/\sigma$ へ写す。ここで $\mu=\frac1n\sum_ix_i$、$\sigma^2=\frac1n\sum_i(x_i-\mu)^2>0$ である（学習されるスケールも $\epsilon$ もないものとする）。
    1. 全微分法により $\mathbf J=\frac1\sigma\bigl(\mathbf I-\frac1n\mathbf 1\mathbf 1^\top-\frac1n\hat{\mathbf x}\hat{\mathbf x}^\top\bigr)$ を示せ。*ヒント：*$d\mu=\frac1n\mathbf 1^\top d\mathbf x$ および $d\sigma=\frac1n\hat{\mathbf x}^\top d\mathbf x$ である。
    1. $\mathbf 1$ と $\hat{\mathbf x}$ が直交し、かつ $\|\hat{\mathbf x}\|^2=n$ であることを示せ。これから、$\sigma\mathbf J$ が $\operatorname{span}\{\mathbf 1,\hat{\mathbf x}\}$ の直交補空間への直交射影であり、$\operatorname{rank}\mathbf J=n-2$（$n\ge2$ のとき）となることを導け。入力のどのような 2 種類の変化を LayerNorm は感知できないか。また、それはなぜか。
    1. 自動微分で計算したヤコビ行列（ランダムな $\mathbf x\in\mathbb R^5$ に対するもの）と照合して (a) と 2 つの零ベクトルを確認し、$\mathbf J$ の固有値を出力せよ。

<!--en:d89dd310-->
1. **対数行列式の勾配。**● 正規化フロー :cite:`rezende2015variational` は、確率ベクトル $\mathbf X$ を可逆な層に順に通すモデルである。1 つの層 $\mathbf Y=\mathbf g(\mathbf X)$ に対して、$\mathbf Y$ の点 $\mathbf g(\mathbf x)$ における対数密度は、密度の変数変換の公式より $\log p_{\mathbf X}(\mathbf x)-\log|\det\mathbf J_{\mathbf g}(\mathbf x)|$ となる。線形層 $\mathbf x\mapsto\mathbf A\mathbf x$ では補正項が $-\log|\det\mathbf A|$ となるため、最尤法による訓練では $\log|\det\mathbf A|$ を $\mathbf A$ に関して微分する。
    1. $\det(\mathbf I+t\mathbf M)$ を $t$ の一次まで展開し、$\det(\mathbf A+d\mathbf A)=\det\mathbf A\,\det(\mathbf I+\mathbf A^{-1}d\mathbf A)$ を用いて、ヤコビの公式 $d\det\mathbf A=\det\mathbf A\,\operatorname{tr}(\mathbf A^{-1}d\mathbf A)$ を可逆行列 $\mathbf A$ に対して導け。さらに :eqref:`eq_mdl-differential-rule` を用いて $\nabla_{\mathbf A}\log|\det\mathbf A|=\mathbf A^{-\top}$ を導け。なぜ $\det\mathbf A$ の符号は無関係なのか。
    1. ランダムな $3\times3$ の可逆行列上で、`np.linalg.slogdet` の中心差分によってこの恒等式を確認せよ。
    1. $\mathbf A$ が特異行列に近づくとき、勾配のノルムはどう振る舞うか。これは、重み行列がほぼ特異になったフロー層に対して何を意味するか。

<!--en:7e8e86cf-->
1. **スペクトルノルムの勾配。**● スペクトル正規化 :cite:`Miyato.Kataoka.Koyama.ea.2018` は、ネットワークの各重み行列 $\mathbf W$ をその最大特異値 $\sigma_1(\mathbf W)$ で割る手法であり、訓練では $\sigma_1$ を微分することになる。
    1. $\mathbf S$ を単純固有値 $\lambda$ と単位固有ベクトル $\mathbf q$ を持つ実対称行列とし、これを $\mathbf S+t\mathbf E$（$\mathbf E$ は対称行列）へ摂動させる。$\lambda(t)$ と $\mathbf q(t)$（$(\mathbf S+t\mathbf E)\mathbf q=\lambda\mathbf q$ と $\mathbf q^\top\mathbf q=1$ で定まる関数）は $t=0$ の近傍で微分可能であるとしてよい（$\lambda$ が単純固有値であるため陰関数定理から従う）。両式を $t=0$ で微分してヘルマン・ファインマンの定理 $\lambda'(0)=\mathbf q^\top\mathbf E\,\mathbf q$ を示し、$\nabla_{\mathbf S}\lambda$ を内積 $\operatorname{tr}(\mathbf A^\top\mathbf B)$ の下で求めよ。
    1. (a) を $\sigma_1(\mathbf W)^2=\lambda_{\max}(\mathbf W^\top\mathbf W)$ に適用し、$\nabla_{\mathbf W}\sigma_1=\mathbf u_1\mathbf v_1^\top$ が（$\sigma_1>0$ が単純であるとき）成り立つことを示せ。ここで $\mathbf u_1$ と $\mathbf v_1$ は上位の左特異ベクトルと右特異ベクトルである。なぜ $\sigma_1>0$ が必要なのか。$\mathbf W=\mathbf w\in\mathbb R^{n\times1}$ を $\mathbf w=\mathbf 0$ の近傍で考察せよ。NumPy でランダムな $5\times3$ 行列上の中心差分によってこの公式を確認せよ。
    1. $\sigma_1(\mathbf I+t\operatorname{diag}(1,-1))$ を正負両符号の小さな $t$ に対して計算せよ。$\sigma_1$ は $2\times2$ 単位行列において微分可能か。また、そこでは (a) のどの仮定が破れているか。

    *Alan Edelman および Steven G. Johnson による [Matrix Calculus for Machine Learning and Beyond, problem set 2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_pset2.pdf)（MIT 18.S096、IAP 2023、問題 3）を改変。*

<!--en:37006531-->
[ディスカッション](https://d2l.discourse.group/t/36857)

<!--en:e1f80a3e-->
行列微分<br>**ヤコビ行列・連鎖律と乗算順序・全微分法・層の逆伝播則（VJP）・ソフトマックスと交差エントロピー**

<!--en:d0f87d05-->
::: {.slide title="スカラーノードからベクトル・行列の層へ"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:96efe379-->
- **$\mathbf{x} = (1, 1)^\top$ における 2-2-1 ReLU ネットワーク：**$g(\mathbf{x}) = \mathbf{w}_2^\top\boldsymbol{\varphi}(\mathbf{W}_1\mathbf{x} + \mathbf{b})$ より $\mathbf{a} = (3, -2)^\top$、$\mathbf{h} = (3, 0)^\top$、$g = 6$

<!--en:64181478-->
- **3 つの異なる導関数の形状：**入力のヤコビ行列 $\partial g/\partial\mathbf{x} \in \mathbb{R}^{1 \times 2}$、重み勾配 $\nabla_{\mathbf{W}_1}g \in \mathbb{R}^{2 \times 2}$、隠れ層のヤコビ行列 $\partial\mathbf{h}/\partial\mathbf{x} \in \mathbb{R}^{2 \times 2}$

<!--en:48cd7aad-->
- **転置バグの危険性：**正方の重み行列で転置が抜けても形状エラーは出ないが、全微分法ならこれを防げる

<!--en:13c16d9b-->
::: {.slide title="ヤコビ行列：ベクトル関数の最良線形写像"}

<!--en:d4577a42-->
[定義と成分]{.kicker}

<!--en:ad547e4e-->
- **ベクトル関数の一次線形化：**$\mathbf{f}(\mathbf{x} + \boldsymbol{\delta}) = \mathbf{f}(\mathbf{x}) + \mathbf{J}_{\mathbf{f}}(\mathbf{x})\boldsymbol{\delta} + o(\|\boldsymbol{\delta}\|)$（ただし $[\mathbf{J}_{\mathbf{f}}(\mathbf{x})]_{ij} = \frac{\partial f_i}{\partial x_j}(\mathbf{x}) \in \mathbb{R}^{m \times n}$）

<!--en:132eb3c0-->
- **行と列の見方：**第 $i$ 行は $\nabla f_i(\mathbf{x})^\top$、第 $j$ 列は $\partial\mathbf{f}/\partial x_j$

<!--en:d8265172-->
- **非線形写像 $(x+\sin y, y+x^2/2)^\top$ の $(0.5, 0.5)^\top$ での値：**$\mathbf{J}_{\mathbf{f}} = \left(\begin{smallmatrix}1 & \cos 0.5 \\ 0.5 & 1\end{smallmatrix}\right)$

<!--en:4642dbb5-->
::: {.slide title="ヤコビ行列の幾何学：円から楕円への写像"}

<!--en:7dff711d-->
[特異値による変形]{.kicker}

<!--en:1b9d1321-->
- **局所的なアフィン変形：**$\mathbf{x}_0 = (0.5, 0.5)^\top$ の近傍で $\mathbf{f}(\mathbf{x}_0 + \boldsymbol{\delta})$ は線形変換 $\mathbf{f}(\mathbf{x}_0) + \mathbf{J}(\mathbf{x}_0)\boldsymbol{\delta}$ のように振る舞う

<!--en:d5894f06-->
- **特異値が半軸を決める：**任意の行列は半径 $r$ の円を半軸 $r\sigma_1$、$r\sigma_2$ の楕円へ写す（$\sigma_1 \approx 1.706$、$\sigma_2 \approx 0.329$ は $\mathbf{J}(\mathbf{x}_0)$ の特異値）

<!--en:fb9bbdf0-->
- **ずれの二次収束：**半径 $r = 0.35$ では真の像（青）とヤコビ行列の楕円（オレンジ破線）のずれは高々 $0.061$ であり、半径を $r = 0.175$ に半減するとずれは 4 分の 1 の $0.015$ に縮まる

<!--en:03a44dab-->
::: {.slide title="ヤコビ行列における剰余項の二次減衰の検証"}

<!--en:7484a6ca-->
[数値的確認]{.kicker}

<!--en:3f8fcdd9-->
ステップ $\boldsymbol{\delta}$ を 10 分の 1 ずつ縮めると、相対線形近似誤差 $\|\mathbf{f}(\mathbf{x}_0 + \boldsymbol{\delta}) - \mathbf{f}(\mathbf{x}_0) - \mathbf{J}(\mathbf{x}_0)\boldsymbol{\delta}\| / \|\boldsymbol{\delta}\|$ が $\|\boldsymbol{\delta}\|$ に比例して 0 に近づくことが確認できる。

<!--en:5c404f30-->
**ヤコビ行列の存在十分条件：**§{sec_mdl-multivariable_calculus} と同様、偏微分の存在だけでは不十分である（$2xy/(x^2+y^2)$ は $\mathbf{0}$ でヤコビ行列を持たない）。すべての偏微分 $\partial f_i / \partial x_j$ が $\mathbf{x}$ の近傍で**連続**ならば $\mathbf{f}$ は $\mathbf{x}$ で微分可能であり、滑らかな（$C^2$ 級の）写像は $O(\|\boldsymbol{\delta}\|^2)$ の誤差を持つ。

<!--en:7ef510b6-->
::: {.slide title="分子レイアウト・分母レイアウトと形状規則"}

<!--en:f2a77ed4-->
[形状の規約]{.kicker}

<!--en:3ec2624f-->
- **ヤコビ行列 $\mathbf{J}_{\mathbf{f}} \in \mathbb{R}^{m \times n}$（分子レイアウト）：**転置なしで $\mathbf{J}_{\mathbf{g}}\mathbf{J}_{\mathbf{f}}$ により直接合成できる

<!--en:c94acb1a-->
- **勾配 $\nabla_{\mathbf{X}} f$（形状の不変則）：**常に $\mathbf{X}$ の形状（列ベクトル $\mathbb{R}^{n \times 1}$ や行列 $\mathbb{R}^{p \times q}$）と一致し、$df = \langle \nabla_{\mathbf{X}} f, d\mathbf{X}\rangle$ で定まる

<!--en:0f5e69ed-->
- **ヘッセ行列 $\mathbf{H}_f = \mathbf{J}_{\nabla f} \in \mathbb{R}^{n \times n}$：**勾配写像のヤコビ行列

<!--en:9297b85f-->
::: {.slide title="ヤコビ行列の連鎖律：行列の乗算"}

<!--en:4bd86744-->
[写像の合成]{.kicker}

<!--en:992e4c41-->
- **行列の連鎖律：**$\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^p$ と $\mathbf{g} : \mathbb{R}^p \to \mathbb{R}^m$ が微分可能ならば $\mathbf{J}_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}) = \mathbf{J}_{\mathbf{g}}(\mathbf{f}(\mathbf{x}))\,\mathbf{J}_{\mathbf{f}}(\mathbf{x})$

<!--en:7f494752-->
- **3 行の証明：**$\boldsymbol{\eta} = \mathbf{A}\boldsymbol{\delta} + o(\|\boldsymbol{\delta}\|) \implies \mathbf{g}(\mathbf{y}+\boldsymbol{\eta}) = \mathbf{g}(\mathbf{y}) + \mathbf{B}\mathbf{A}\boldsymbol{\delta} + o(\|\boldsymbol{\delta}\|)$

<!--en:b168b3d1-->
- **経路和との対応：**$[\mathbf{B}\mathbf{A}]_{ik} = \sum_{j=1}^p \frac{\partial g_i}{\partial y_j}\frac{\partial f_j}{\partial x_k}$

<!--en:12f6e8d1-->
::: {.slide title="多層パーセプトロンのヤコビ行列"}

<!--en:1ab75ad2-->
[層ごとの因子]{.kicker}

<!--en:f631a960-->
- **2-2-1 $\mathrm{ReLU}$ ネットワークを構成する 3 つの層：**
  1. アフィン層 $\mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}$ のヤコビ行列は $\mathbf{J}_1 = \mathbf{W}_1$
  2. 要素ごとの $\mathrm{ReLU}$ $\mathbf{h} = \boldsymbol{\varphi}(\mathbf{a})$ は座標間を結合しないため、ヤコビ行列は対角行列 $\mathbf{J}_2 = \operatorname{diag}(\mathbf{m})$（指示マスク $m_i = \mathbf{1}(a_i > 0)$）
  3. 出力層 $g = \mathbf{w}_2^\top \mathbf{h}$ は $1 \times 2$ の行ヤコビ行列 $\mathbf{J}_3 = \mathbf{w}_2^\top$ を持つ

<!--en:25c61bfd-->
- **$\mathbf{x} = (1, 1)^\top$（$\mathbf{a} = (3, -2)^\top$、$\mathbf{m} = (1, 0)^\top$）のとき：**積は $(2, 4)$ となる。
  $$\mathbf{J}_g(\mathbf{x}) = \mathbf{w}_2^\top \operatorname{diag}(\mathbf{m})\,\mathbf{W}_1$$

<!--en:0fbfc3f8-->
::: {.slide title="コードによる 2-2-1 ネットワークのヤコビ行列の検証"}

<!--en:0b41a912-->
[自動微分による検証]{.kicker}

<!--en:05c5b73e-->
$K$ 層のネットワーク $\mathbf{f} = \mathbf{f}_K \circ \cdots \circ \mathbf{f}_1$ に対して連鎖律を反復適用すると $\mathbf{J} = \mathbf{J}_K \mathbf{J}_{K-1} \cdots \mathbf{J}_1$ が得られ、重み行列 $\mathbf{W}_k$ と計算量 $O(n)$ の対角活性化マスク $\operatorname{diag}(\boldsymbol{\varphi}_k')$ が交互に並ぶ。

<!--en:f64209b0-->
::: {.slide title="深いヤコビ行列の積：ノルムの消失と爆発"}

<!--en:5695923a-->
[スペクトルスケーリング]{.kicker}

<!--en:f992edef-->
- **深さに対する指数スケーリング（$\mathbf{J} = \mathbf{J}_K \cdots \mathbf{J}_1$）：**同一の $\mathbf{W} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ を繰り返すと、$\mathbf{W}^K = \mathbf{V}\boldsymbol{\Lambda}^K\mathbf{V}^{-1}$ は $|\lambda_{\max}|^K$ のオーダーで変化する

<!--en:439bb8d8-->
- **深さ $K = 50$ のとき：**$|\lambda_{\max}| = 0.9 \implies 0.9^{50} \approx 0.005$（勾配消失）、$|\lambda_{\max}| = 1.1 \implies 1.1^{50} \approx 117$（勾配爆発）

<!--en:89772c56-->
- **入力勾配と重み勾配：**$\mathbf{J}_K \cdots \mathbf{J}_1$ が与えるのは $\partial g/\partial\mathbf{x}$ であり、訓練に必要なのは $\nabla_{\mathbf{W}_k}\ell$

<!--en:72f7e9ca-->
::: {.slide title="結合法則と 670× のコスト差"}

<!--en:a688e324-->
[乗算の順序]{.kicker}

<!--en:db9d17e9-->
- **同じ積、異なる括弧の付け方：**スカラー損失 $\ell$ を微分するため、$1 \times h$ の行ベクトル $\mathbf{u}^\top = \partial\ell/\partial\mathbf{h}_3$ を $\mathbf{J}_3\mathbf{J}_2\mathbf{J}_1$ に掛ける（$h = n = 1000$）

<!--en:d649d55e-->
- **左から右（リバースモード / VJP）：**$(((\mathbf{u}^\top \mathbf{J}_3)\mathbf{J}_2)\mathbf{J}_1)$ は中間結果をすべて $1 \times h$ の行ベクトルに保ち、3 回の行列・ベクトル積で乗算は $h^2 + h^2 + hn = 3 \times 10^6$ 回

<!--en:a6cb1ef8-->
- **右から左（行列・行列積の罠）：**先に $\mathbf{J}_2\mathbf{J}_1$ と $\mathbf{J}_3(\mathbf{J}_2\mathbf{J}_1)$ を作ると $2h^2n + hn = 2.001 \times 10^9$ 回の乗算を要し、**$667\times$（$2h/3$）の速度低下**となる

<!--en:764cda10-->
::: {.slide title="全微分法：公式集を置き換える 1 つの規則"}

<!--en:13e64b06-->
[一次の微分]{.kicker}

<!--en:324d75e8-->
- **ステップ 1：$df$ の展開：**線形性、行列積の微分法則 $d(\mathbf{A}\mathbf{B}) = (d\mathbf{A})\mathbf{B} + \mathbf{A}(d\mathbf{B})$、転置 $d(\mathbf{A}^\top) = (d\mathbf{A})^\top$、要素ごとの微分 $d\varphi(\mathbf{X}) = \varphi'(\mathbf{X})\odot d\mathbf{X}$ を適用する

<!--en:4a59db87-->
- **ステップ 2：標準内積との照合：**$df = \langle \mathbf{M}, d\mathbf{X}\rangle = \operatorname{tr}(\mathbf{M}^\top d\mathbf{X})$ の形に整理し、$\nabla_{\mathbf{X}} f = \mathbf{M}$ を読み取る

<!--en:47121926-->
::: {.slide title="dX を分離するトレースの移動則"}

<!--en:9b4b7f38-->
[随伴の代数]{.kicker}

<!--en:4aef26e1-->
- **トレースの巡回恒等式（$\operatorname{tr}(\mathbf{U}\mathbf{V}) = \operatorname{tr}(\mathbf{V}\mathbf{U})$）：**
  - 左からの因子：$\langle \mathbf{A}, \mathbf{B}\mathbf{C}\rangle = \langle \mathbf{B}^\top\mathbf{A}, \mathbf{C}\rangle$
  - 右からの因子：$\langle \mathbf{A}, \mathbf{B}\mathbf{C}\rangle = \langle \mathbf{A}\mathbf{C}^\top, \mathbf{B}\rangle$
  - アダマールマスク：$\langle \mathbf{A}, \mathbf{B} \odot \mathbf{C}\rangle = \langle \mathbf{A} \odot \mathbf{B}, \mathbf{C}\rangle$

<!--en:f450828a-->
::: {.slide title="線形形式・二次形式・最小二乗法の勾配導出"}

<!--en:e0bca90d-->
[具体的な導出]{.kicker}

<!--en:8808a9f1-->
- **1. 線形形式 $\mathbf{a}^\top\mathbf{x}$：**$df = \langle \mathbf{a}, d\mathbf{x}\rangle \implies \nabla_{\mathbf{x}}(\mathbf{a}^\top\mathbf{x}) = \mathbf{a}$

<!--en:a20088bb-->
- **2. 二次形式 $\mathbf{x}^\top\mathbf{A}\mathbf{x}$：**$df = \langle (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}, d\mathbf{x}\rangle \implies \nabla_{\mathbf{x}}(\mathbf{x}^\top\mathbf{A}\mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$（$= 2\mathbf{A}\mathbf{x}$ となるのは $\mathbf{A} = \mathbf{A}^\top$ の場合に限る）

<!--en:7c4f3b2d-->
- **3. 最小二乗法 $\|\mathbf{W}\mathbf{x} - \mathbf{y}\|_2^2$：**$d\ell = \langle 2\mathbf{r}\mathbf{x}^\top, d\mathbf{W}\rangle \implies \nabla_{\mathbf{W}}\ell = 2(\mathbf{W}\mathbf{x} - \mathbf{y})\mathbf{x}^\top$

<!--en:15c5760f-->
::: {.slide title="妥当性確認：形状・1×1 スカラー・非対称テスト"}

<!--en:2bbfdc94-->
[行列勾配のデバッグ]{.kicker}

<!--en:58ec089c-->
- **1. 形状チェック：**長方行列（$p \neq q$）での転置漏れは検出できるが、正方行列（$p = q$）は見逃す

<!--en:4d887d7a-->
- **2. $1 \times 1$ スカラーチェック：**係数 $2$ の抜けや符号ミスは検出できるが、転置はすべて見逃す

<!--en:c7c6cd46-->
- **3. 非対称行列による数値テスト：**ランダムな非対称正方行列上で自動微分と比較すれば*あらゆる*バグを検出できる

<!--en:4c25d777-->
::: {.slide title="層の一般の逆伝播則（VJP）"}

<!--en:03a307b1-->
[層の随伴変数]{.kicker}

<!--en:9e452ada-->
- **$10^{12}$ 要素のテンソルヤコビ行列は不要：**層 $\mathbf{Y} = \mathbf{F}(\mathbf{X})$ は上流の随伴変数 $\bar{\mathbf{Y}} = \nabla_{\mathbf{Y}}\ell$ を直接 $\bar{\mathbf{X}} = \nabla_{\mathbf{X}}\ell$ へ写す

<!--en:7e16b360-->
- **随伴の双対性による恒等式：**$\bar{\mathbf{X}}$ は $\langle \bar{\mathbf{X}}, d\mathbf{X}\rangle = \langle \bar{\mathbf{Y}}, d\mathbf{Y}\rangle$ をすべての $d\mathbf{X}$ に対して満たす一意な行列

<!--en:a3d8b867-->
- **ベクトルの VJP：**$d\mathbf{y} = \mathbf{J}\,d\mathbf{x}$ のとき、$\bar{\mathbf{x}} = \mathbf{J}^\top\bar{\mathbf{y}}$

<!--en:61c352a7-->
::: {.slide title="バッチ行列積の逆伝播則"}

<!--en:7d71c639-->
[MLP の基本プリミティブ]{.kicker}

<!--en:4748213d-->
- **バッチ行列積（$\mathbf{Y} = \mathbf{X}\mathbf{W}$）：**$\langle \bar{\mathbf{Y}}, (d\mathbf{X})\mathbf{W} + \mathbf{X}(d\mathbf{W})\rangle$ から直ちに $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}^\top$ と $\bar{\mathbf{W}} = \mathbf{X}^\top\bar{\mathbf{Y}}$ が得られる

<!--en:0cd524c4-->
- **要素ごとの演算とバイアスの規則：**$\mathbf{Y} = \boldsymbol{\varphi}(\mathbf{X}) \implies \bar{\mathbf{X}} = \varphi'(\mathbf{X}) \odot \bar{\mathbf{Y}}$、$\mathbf{Z} = \mathbf{Y} + \mathbf{1}\mathbf{b}^\top \implies \bar{\mathbf{b}} = \bar{\mathbf{Z}}^\top\mathbf{1}$

<!--en:4ed495b8-->
::: {.slide title="コードによる正方行列の転置バグの検出"}

<!--en:1b718c47-->
[数値的検証]{.kicker}

<!--en:53ac6452-->
スカラーのプローブ関数 $s = \langle \bar{\mathbf{Y}}, \mathbf{X}\mathbf{W}\rangle$ をランダムな**正方非対称**の $3 \times 3$ 行列上で微分し、同じ形状のもっともらしい転置ミスと並べることで、行列積の逆伝播則を直接テストできる。

<!--en:16ad2ee2-->
導出した規則はすべて `True` を出力し、もっともらしい誤答候補（`2 A x`、`Ybar W`、`(X^T Ybar)^T`）はすべて `False` を出力する。独自の逆伝播則は必ず非対称行列でテストすべきである。

<!--en:9df08b40-->
::: {.slide title="2-2-1 ネットワークの全重み勾配の手計算トレース"}

<!--en:94e42837-->
[全層のスイープ]{.kicker}

<!--en:fab30781-->
3 つの局所 VJP 則を $\bar{g} = 1$ から逆向きに $\mathbf{x} = (1, 1)^\top$（$\mathbf{a} = (3, -2)^\top$、マスク $\mathbf{m} = (1, 0)^\top$、$\mathbf{h} = (3, 0)^\top$）において適用すると、2-2-1 ネットワークのすべてのパラメータ勾配が得られる。

<!--en:e6d335c1-->
1. **出力層 $g = \mathbf{w}_2^\top\mathbf{h}$：**$\bar{\mathbf{h}} = \bar{g}\mathbf{w}_2 = (2, -1)^\top$、$\bar{\mathbf{w}}_2 = \bar{g}\mathbf{h} = (3, 0)^\top$

<!--en:aee07695-->
2. **$\mathrm{ReLU}$ とバイアス $\mathbf{h} = \boldsymbol{\varphi}(\mathbf{a})$、$\mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}$：**$\bar{\mathbf{a}} = \mathbf{m} \odot \bar{\mathbf{h}} = (2, 0)^\top$、$\bar{\mathbf{b}} = \bar{\mathbf{a}} = (2, 0)^\top$

<!--en:4bc45c8f-->
3. **第 1 線形層：**$\bar{\mathbf{W}}_1 = \bar{\mathbf{a}}\mathbf{x}^\top$ $= \begin{pmatrix} 2 & 2 \\ 0 & 0 \end{pmatrix}$、$\bar{\mathbf{x}} = \mathbf{W}_1^\top\bar{\mathbf{a}} = (2, 4)^\top$

<!--en:dfa56fb2-->
::: {.slide title="ソフトマックスのヤコビ行列と Log-Sum-Exp の凸性"}

<!--en:c389af16-->
[分類器の導関数]{.kicker}

<!--en:235473c4-->
- **ソフトマックスのヤコビ行列：**$p_i = e^{z_i}/\sum_k e^{z_k}$ を微分すると $\mathbf{J}_{\operatorname{softmax}}(\mathbf{z}) = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ となり、シフト不変性から零ベクトル $\mathbf{1}$ を持つ

<!--en:d23277f6-->
- **LSE の凸性の統計的証明：**$\mathbf{p} = \nabla\operatorname{LSE}(\mathbf{z})$ より、いたるところで $\mathbf{v}^\top\mathbf{H}_{\operatorname{LSE}}\mathbf{v} = \operatorname{Var}_{I \sim \mathbf{p}}(v_I) \ge 0$

<!--en:a74927d0-->
::: {.slide title="ソフトマックスと交差エントロピーの p − y への簡約"}

<!--en:09e7377e-->
[ロジットの勾配]{.kicker}

<!--en:4d9220bf-->
- **$1/p_i$ のきれいな相殺：**$\frac{\partial\ell}{\partial p_i} = -\frac{y_i}{p_i}$ を $\mathbf{J}_{\operatorname{softmax}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ に掛けると分母の $p_i$ が相殺される

<!--en:6eaf1fbd-->
- **ロジット勾配の公式：**$\nabla_{\mathbf{z}}\ell = \bigl(\sum_{i=1}^C y_i\bigr)\mathbf{p} - \mathbf{y}$ であり、$\mathbf{p} - \mathbf{y}$ に簡約される（$\sum_{i=1}^C y_i = 1$ のとき）

<!--en:80eb705c-->
::: {.slide title="ソフトマックスのスペクトルとツーホットターゲットの落とし穴の検証"}

<!--en:f989df99-->
[コードによる検証]{.kicker}

<!--en:513c7091-->
ロジット $\mathbf{z} = (1, -0.5, 2)^\top$ において、ソフトマックスのヤコビ行列は 1 つのゼロ固有値（方向 $\mathbf{1}$）と 2 つの正の固有値を持ち、正規化されていないツーホットターゲット $\mathbf{y} = (1, 1, 0)^\top$（$\sum_i y_i = 2$）の勾配は $2\mathbf{p} - \mathbf{y}$ となって $\mathbf{p} - \mathbf{y}$ にはならない。

<!--en:43215298-->
::: {.slide title="理解度チェック：行列微分"}

<!--en:b6d9a92f-->
[復習と演習]{.kicker}

<!--en:712482ed-->
全微分法、トレースの移動則、ヤコビ行列の連鎖律を用いて、次の 3 つの問いに答えよ。

<!--en:dbe7b769-->
1. **行列因子分解の勾配：**損失 $\ell(\mathbf{U}, \mathbf{V}) = \tfrac{1}{2}\|\mathbf{U}\mathbf{V} - \mathbf{X}\|_F^2$（残差 $\mathbf{R} = \mathbf{U}\mathbf{V} - \mathbf{X} \in \mathbb{R}^{m \times n}$、$\mathbf{U} \in \mathbb{R}^{m \times r}$、$\mathbf{V} \in \mathbb{R}^{r \times n}$）に対し、$d\ell = \langle \mathbf{R}, (d\mathbf{U})\mathbf{V} + \mathbf{U}(d\mathbf{V})\rangle$ を用いて $\nabla_{\mathbf{U}}\ell$ と $\nabla_{\mathbf{V}}\ell$ を求めよ。

<!--en:1e08faa9-->
2. **ソフトマックス VJP の計算量：**$C$ 個のクラスまたはトークン上の自己アテンションにおいて、$\mathbf{J}_{\operatorname{softmax}} \in \mathbb{R}^{C \times C}$ を作って $\mathbf{u}$ を掛けると $O(C^2)$ 回の演算を要する。$\mathbf{J}_{\operatorname{softmax}}^\top\mathbf{u} = (\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)\mathbf{u}$ は、$\mathbf{J}$ を作らずにどれほど高速に評価できるか。

<!--en:043b5c49-->
3. **死んだ ReLU のパラメータ：**2-2-1 $\mathrm{ReLU}$ ネットワークの $\mathbf{x} = (1, 1)^\top$ において、隠れユニット 2 の活性化前の値は $a_2 = -2 < 0$ であった。この例において、ユニット 2 のどのパラメータが非ゼロの勾配を受け取ったか。

<!--en:557b98b6-->
**解答：**

<!--en:bef67de6-->
1. トレースの移動則を適用すると $d\ell = \langle \mathbf{R}\mathbf{V}^\top, d\mathbf{U}\rangle + \langle \mathbf{U}^\top\mathbf{R}, d\mathbf{V}\rangle$ となるため、$\nabla_{\mathbf{U}}\ell = \mathbf{R}\mathbf{V}^\top \in \mathbb{R}^{m \times r}$ および $\nabla_{\mathbf{V}}\ell = \mathbf{U}^\top\mathbf{R} \in \mathbb{R}^{r \times n}$ を得る（どちらも変数の形状と一致する）。

<!--en:16ff99d8-->
2. $(\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)\mathbf{u}$ $= \mathbf{p} \odot \bigl(\mathbf{u} - (\mathbf{p}^\top\mathbf{u})\mathbf{1}\bigr)$ と書き換えれば、必要なのは 1 回のドット積 $\mathbf{p}^\top\mathbf{u}$、1 回のブロードキャスト減算、1 回の要素ごとの乗算だけであり、**$O(C)$ の計算時間とメモリ**で済む（$O(C^2)$ ではなくなる）。

<!--en:208a945b-->
3. **1 つもない：**$\bar{\mathbf{W}}_1$ の第 2 行は $(0, 0)$、$\bar{b}_2 = 0$（$m_2 = 0$ のため）、さらに $\bar{w}_{2,2} = h_2 = 0$ である。ある $\mathrm{ReLU}$ ユニットが訓練セット全体で負の値をとる場合、勾配降下法はその重みを一切更新しない（「死んだ $\mathrm{ReLU}$」）。

<!--en:1f9deb9b-->
::: {.slide title="まとめ：行列微分と層の逆伝播則"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:58d4cc85-->
| 演算 / 概念 | 順伝播の式 | ヤコビ行列 / 逆伝播 VJP 則 |

<!--en:0ba85676-->
| **ベクトル写像の線形化** | $\mathbf{y} = \mathbf{f}(\mathbf{x}) : \mathbb{R}^n \to \mathbb{R}^m$ | $\mathbf{f}(\mathbf{x}+\boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}_{\mathbf{f}}\boldsymbol{\delta}$、$[\mathbf{J}_{\mathbf{f}}]_{ij} = \frac{\partial f_i}{\partial x_j}$ |

<!--en:f36a3421-->
| **連鎖律と VJP の順序** | $\mathbf{f} = \mathbf{f}_K \circ \cdots \circ \mathbf{f}_1$ | $\mathbf{u}^\top\mathbf{J} = ((\mathbf{u}^\top\mathbf{J}_K)\cdots)\mathbf{J}_1$（$O(Kh^2)$ 対 右から左の $O(Kh^3)$） |

<!--en:9008e8e7-->
| **全微分と移動則** | $df = \langle \nabla_{\mathbf{X}}f, d\mathbf{X}\rangle = \operatorname{tr}((\nabla_{\mathbf{X}}f)^\top d\mathbf{X})$ | $\langle \mathbf{A}, \mathbf{B}\mathbf{C}\rangle = \langle \mathbf{B}^\top\mathbf{A}, \mathbf{C}\rangle$ $= \langle \mathbf{A}\mathbf{C}^\top, \mathbf{B}\rangle$ |

<!--en:3c8a69e8-->
| **二次形式** | $f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}\mathbf{x}$ | $\nabla_{\mathbf{x}} f = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$（$= 2\mathbf{A}\mathbf{x}$ は $\mathbf{A} = \mathbf{A}^\top$ と同値） |

<!--en:3b908d6a-->
| **バッチ線形層** | $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^\top$ | $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}^\top$、$\bar{\mathbf{W}} = \mathbf{X}^\top\bar{\mathbf{Y}}$、$\bar{\mathbf{b}} = \bar{\mathbf{Y}}^\top\mathbf{1}$ |

<!--en:a29b74ac-->
| **要素ごとの活性化** | $\mathbf{Y} = \boldsymbol{\varphi}(\mathbf{X})$ | $\bar{\mathbf{X}} = \varphi'(\mathbf{X}) \odot \bar{\mathbf{Y}}$（$O(n)$ 演算の対角マスク） |

<!--en:f93de165-->
| **ソフトマックスと交差エントロピー** | $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$、$\ell = -\mathbf{y}^\top\log\mathbf{p}$ | $\mathbf{J}_{\operatorname{softmax}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$、$\nabla_{\mathbf{z}}\ell = \mathbf{p} - \mathbf{y}$（$\sum_i y_i = 1$ のとき） |
