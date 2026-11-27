<!--en:dfe88535-->
# 誤差逆伝播法

<!--en:93f840b7-->
:numref:`sec_mlp-implementation` では、導関数を 1 つも手書きすることなく Fashion-MNIST の MLP がもつ $203{,}530$ 個すべてのパラメータを訓練した。自動微分（automatic differentiation）があらゆる勾配を計算してくれたからである。:numref:`sec_autograd` ではその一般原理を見た。すなわち、リバースモード自動微分は順伝播（forward pass）を計算グラフ（computational graph）として記録し、スカラー損失から逆向きに連鎖律（chain rule）をたどるのである。

<!--en:3c8bfbcf-->
本節では、多層パーセプトロン（multilayer perceptron、MLP）についてそのブラックボックスの中身を開いてみよう。MLP 全体に必要なのは、行列積、ブロードキャストされたバイアスの加算、要素ごとの活性化という **3 つの局所的な行列微分の規則**だけである。これらを順にたどることで、工学上重要な 3 つの問いに答えることができる。

<!--en:6130e6f7-->
1. どのようにして 1 回の逆伝播（backward pass）は、1 回の順伝播の高々約 $2\times$ の演算コストで数十万個もの偏微分（partial derivative）を計算するのか。

<!--en:68bc102c-->
2. なぜ訓練は推論（inference）よりもはるかに多くの GPU メモリを消費し、そのメモリ量はミニバッチサイズ $n$ と深さ $K$ に対して線形に増加するのか。

<!--en:6ee47c78-->
3. 死んだ ReLU ユニット（dead ReLU unit）や勾配消失・勾配爆発（vanishing/exploding gradients）は、逆伝播の漸化式から機械的にどのように生じるのか。

<!--en:7c4f2d09-->
## 計算グラフ

<!--en:f5edf441-->
### 小さなグラフ

<!--en:113b7acd-->
リバースモード微分を最も単純な形で復習するため、3 つのスカラーからなる式 $e = (a + b)\,c$ を $a = 2, b = 1, c = -3$ において考えよう（:numref:`fig_mdl-mlp-backprop-small`）。*順伝播*（左から右）では $u = a + b = 3$ と $e = u\,c = -9$ を計算する。*逆伝播*（右から左）では $\partial e / \partial e = 1$ をシードとして与え、ノードごとに 1 つの局所導関数を適用する。

<!--en:02aae982-->
- 乗算ノード $e = u\,c$ では、$\partial e / \partial u = c = -3$ および $\partial e / \partial c = u = 3$ となる。

<!--en:34bcecc4-->
- 加算ノード $u = a + b$ では、$\partial u / \partial a = \partial u / \partial b = 1$ であるため、$\partial e / \partial a = \partial e / \partial b = -3$ となる。

<!--en:d9319349-->
![$e = (a + b)\,c$ の $a = 2, b = 1, c = -3$ における計算グラフ。順伝播の値（黒）とリバースモードの導関数 $\partial e / \partial (\cdot)$（赤）を示す。](../img/mdl-mlp-backprop-small.svg)

<!--en:dc037565-->
ここにすでに表れている 2 つの普遍的な規則に注目しよう。(i) 加算ノードは入力された勾配をそのまま各被加数へコピーするのに対し、乗算ノードは入力された勾配に*他方*の因子を掛ける（そのため順伝播で $u$ と $c$ をキャッシュしておく必要がある）。(ii) 逆伝播はグラフを*逆トポロジカル順序*（reverse topological order）で走査する。

<!--en:2abc79ed-->
### MLP のグラフ

<!--en:f1ab404a-->
次に、隠れ層（hidden layer）を 1 つもつ MLP（:numref:`fig_forward`）をミニバッチ $\mathbf{X} \in \mathbb{R}^{n \times d}$ 上で考えよう。パラメータを $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$、$\mathbf{b}^{(1)} \in \mathbb{R}^h$、$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$、$\mathbf{b}^{(2)} \in \mathbb{R}^q$ とし、$\ell_2$ 重み減衰（weight decay）ペナルティを $s = \frac{\lambda}{2}(\|\mathbf{W}^{(1)}\|_{\textrm{F}}^2 + \|\mathbf{W}^{(2)}\|_{\textrm{F}}^2)$ とすると、次のようになる。

<!--en:0d0c3a5d-->
ここで、$L$ はミニバッチ上の平均交差エントロピー損失（cross-entropy loss）であり、$\|\mathbf{W}\|_{\textrm{F}}^2 = \sum_{r,j} w_{rj}^2$ は二乗フロベニウスノルムである。

<!--en:d0836c3f-->
![重み減衰を伴う 1 隠れ層 MLP（$d = 784, h = 256, q = 10$）における順伝播の計算グラフ。パラメータノードは網掛けされており、各重み行列はそのアフィン層と正則化ペナルティ $s$ の両方に入力される。](../img/mdl-mlp-forward-graph.svg)

<!--en:17b008a9-->
## MLP における誤差逆伝播法

<!--en:798c3056-->
### 行列に対する局所規則

<!--en:0f7bdb09-->
任意の行列 $\mathbf{A}$ に対し、$\partial J / \partial \mathbf{A}$ は偏微分の行列 $(\partial J / \partial a_{ij})$ を表し、$\mathbf{A}$ と*まったく同じ形状*をもつものとする。3 つの局所的なベクトル・ヤコビ行列規則（:numref:`subsec_mdl-layer-vjp`）によって、MLP のすべての層を扱うことができる。

<!--en:7fd937bb-->
1. **行列積（$\mathbf{A} = \mathbf{X}\mathbf{W}$、ただし $\mathbf{X} \in \mathbb{R}^{n \times d}, \mathbf{W} \in \mathbb{R}^{d \times h}$）：**上流の勾配 $\mathbf{G} = \partial J / \partial \mathbf{A} \in \mathbb{R}^{n \times h}$ が与えられたとき、$a_{ij} = \sum_r x_{ir} w_{rj}$ であるから、連鎖律より $\partial J / \partial w_{rj} = \sum_i x_{ir} g_{ij}$ および $\partial J / \partial x_{ir} = \sum_j g_{ij} w_{rj}$ が得られる。行列形式で書けば次のようになる。
   $$
   \frac{\partial J}{\partial \mathbf{W}} = \mathbf{X}^\top \mathbf{G}, \qquad
   \frac{\partial J}{\partial \mathbf{X}} = \mathbf{G}\,\mathbf{W}^\top.
   $$
   :eqlabel:`eq_backprop-matmul`
   形状がぴったりかみ合うことに注目しよう。$\mathbf{X}^\top\mathbf{G}$ は $(d \times n)(n \times h) = d \times h$（$\mathbf{W}$ と一致）となり、$\mathbf{G}\mathbf{W}^\top$ は $(n \times h)(h \times d) = n \times d$（$\mathbf{X}$ と一致）となる。スカラーの乗算とまったく同様に、$\partial J / \partial \mathbf{W}$ の計算にはキャッシュされた入力 $\mathbf{X}$ が必要であり、一方で $\partial J / \partial \mathbf{X}$ の計算は $\mathbf{W}^\top$ を通じて誤差信号を逆方向へ伝播させる。

<!--en:786b77d2-->
2. **ブロードキャストされたバイアスの加算（$\mathbf{Z} = \mathbf{A} + \mathbf{b}$、ただし $\mathbf{b} \in \mathbb{R}^h$）：**$b_j$ はすべての行 $i$ にわたって列 $j$ に加算されるため、$\partial J / \partial \mathbf{A} = \mathbf{G}$ は変化せずそのまま通過し、一方で $\partial J / \partial \mathbf{b} = \mathbf{G}^\top\mathbf{1}_n$ は $\mathbf{G}$ を $n$ 個のデータ例（行）にわたって合計する。

<!--en:32d98056-->
3. **要素ごとの活性化（$\mathbf{H} = \phi(\mathbf{Z})$）：**$h_{ij} = \phi(z_{ij})$ は成分 $(i, j)$ のみに依存するため、上流の勾配は局所的な活性化の傾きによって成分ごとにゲートされる（アダマール積 $\odot$）。
   $$\frac{\partial J}{\partial \mathbf{Z}} = \frac{\partial J}{\partial \mathbf{H}} \odot \phi'(\mathbf{Z}).$$

<!--en:e2e02f72-->
### 逆伝播

<!--en:a55b8c8b-->
$\partial J / \partial \mathbf{O} = \frac{1}{n}(\hat{\mathbf{Y}} - \mathbf{Y})$（:numref:`subsec_softmax_and_derivatives` 参照。ここで $\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{O})$ であり、$\mathbf{Y} \in \{0, 1\}^{n \times q}$ はワンホットラベル行列である）および $\partial s / \partial \mathbf{W}^{(k)} = \lambda \mathbf{W}^{(k)}$ から出発し、3 つの規則を出力から入力へ向かって逆トポロジカル順序で適用すると、MLP の完全な逆伝播が得られる。

<!--en:aa31ea54-->
一般の $K$ 層 MLP（$\mathbf{Z}^{(k)} = \mathbf{H}^{(k-1)}\mathbf{W}^{(k)} + \mathbf{b}^{(k)}, \mathbf{H}^{(k)} = \phi(\mathbf{Z}^{(k)})$）では、各層 $k = K-1, \ldots, 1$ がまったく同じ 2 ステップのモジュールを実行する。

<!--en:7bf4dd3f-->
### 計算例

<!--en:b248a355-->
:eqref:`eq_backprop-mlp` を、ごく小さなネットワーク（$n = 1, d = h = 2, q = 1$、ReLU 活性化、バイアスは 0、$\lambda = 0$、二乗損失（squared loss）$L = \frac{1}{2}(o - y)^2$）上で手計算により追ってみよう。

<!--en:46a61340-->
順伝播（:numref:`fig_mdl-mlp-backprop-graph`）では、$\mathbf{Z} = \mathbf{X}\mathbf{W}^{(1)} = \begin{bmatrix} -1 & 2 \end{bmatrix}$、$\mathbf{H} = \mathrm{ReLU}(\mathbf{Z}) = \begin{bmatrix} 0 & 2 \end{bmatrix}$（ユニット 1 は非活性）、$o = \mathbf{H}\mathbf{W}^{(2)} = -2$、$L = \frac{1}{2}(-2)^2 = 2$ となる。逆伝播では、$\partial L / \partial o = o - y = -2$ であり、各勾配は次のようになる。

<!--en:e83fafed-->
$\partial L / \partial \mathbf{W}^{(1)}$ の 1 列目がすべて 0 になっていることに注目しよう。$z_1 = -1 < 0$ であるため、$\phi'(z_1) = 0$ によって上流の勾配 $\partial L / \partial h_1 = -4$ が遮断され、ユニット 1 へ入力されるどの重みにも届かないからである。

<!--en:2fd42c2f-->
![計算例（:eqref:`eq_backprop-example`）における順伝播の値（黒）と逆伝播の勾配（赤）。$z_1 = -1 < 0$ であるため、ReLU ゲートによって $\partial L / \partial z_1$ および $\partial L / \partial \mathbf{W}^{(1)}$ の 1 列目が 0 になる。](../img/mdl-mlp-backprop-graph.svg)

<!--en:4f5d3d74-->
### ミニバッチ上での逆伝播の検証

<!--en:70adbda6-->
:eqref:`eq_backprop-mlp` を `mlp_backward` として実装し、64 ビット浮動小数点（floating point）を用いてランダムな $256 \times 784$ のミニバッチ（$h = 256, q = 10, \lambda = 10^{-3}$）上で各フレームワークの自動微分と比較して検証しよう。

<!--en:118d7b60-->
256 個のデータ例からなるランダムなミニバッチを、:numref:`sec_mlp-implementation` とまったく同じ次元および初期化（initialization）に $\lambda = 10^{-3}$ を加えた設定で生成し、`mlp_backward` を各フレームワーク組み込みの自動微分と比較しよう。勾配チェックの定石に従い、64 ビット浮動小数点（丸め誤差の下限が約 $10^{-16}$ と小さいため、わずかな代数的バグも確実に検出できる）で実行し、パラメータテンソルごとに最大の相対差を出力する。

<!--en:b1140cd7-->
すべてのパラメータ勾配は自動微分の結果と相対誤差 $10^{-15}$ 以内で一致しており、これは倍精度（double precision）のマシンイプシロン $2.2 \times 10^{-16}$ の数倍程度にすぎない。

<!--en:08580c55-->
## メモリと計算量

<!--en:7fa4ab64-->
### 保存される活性化値とパラメータメモリ

<!--en:9269bfbf-->
:eqref:`eq_backprop-mlp` を見ると、逆伝播において $\partial J / \partial \mathbf{W}^{(2)}$ と $\partial J / \partial \mathbf{W}^{(1)}$ を計算するには、順伝播で $\mathbf{X}$、$\mathbf{H}$（ReLU の場合はマスク $\mathbf{Z} > 0$ の情報も兼ねる）、および $\hat{\mathbf{Y}}$ を保持しておく必要がある。Fashion-MNIST の MLP（$d = 784, h = 256, q = 10$）においてミニバッチサイズを $n = 256$ とすると、キャッシュされたこれらのテンソルが保持する要素数は次のようになる。

<!--en:266b7ecd-->
これはすでに $203{,}530$ 個のパラメータ（$\approx 0.8\textrm{ MB}$）を上回っている。$K$ 層のネットワークでは、活性化メモリ（activation memory）はバッチサイズ $n$ と深さ $K$ の*両方*に対して線形に増大するのに対し、推論では次の層を計算した直後に各層の活性化値を破棄できる。

<!--en:bb419b69-->
### 演算コスト

<!--en:45c6d9f1-->
順伝播では、各データ例あたり $d h + h q = 784 \times 256 + 256 \times 10 = 203{,}264$ 回の積和演算（multiply-add）を実行する。逆伝播では、内部層の各行列積 $\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}$ から同じサイズの逆方向の行列積が *2 つ*（重みに対する $\mathbf{H}^{(k-1)\top}\mathbf{G}$ と下層の活性化値に対する $\mathbf{G}\mathbf{W}^{(k)\top}$）生じる一方、最初の層で必要なのは $\mathbf{X}^\top\mathbf{G}_Z$ のみである（$\partial J / \partial \mathbf{X}$ は不要なため）。したがって、この 2 層 MLP の逆伝播は $d h + 2 h q = 205{,}824$ 回の積和演算で済み、順伝播とほぼ同じである。計算量の大部分を最初の層が占めており、そこでは逆方向の積が 1 回しか必要ないからである。深層ネットワークではほぼすべての層が内部層となるため、逆伝播には順伝播の $\sim 2\times$ のコストがかかり、訓練 1 ステップ全体では順伝播による推論のおよそ $3\times$ となる。いずれにせよ、$203{,}530$ 個すべての導関数が追加の順伝播 1〜2 回分のコストで求まるのに対し、片側有限差分（finite difference）では*たった 1 つ*のスカラーパラメータにつき 1 回分の追加の順伝播を費やすことになる。

<!--en:30ac03ff-->
## まとめ

<!--en:c3f7335d-->
多層パーセプトロンにおける誤差逆伝播法（backpropagation）は、逆トポロジカル順序で適用される 3 つの局所的な行列規則に帰着する。すなわち、$\partial J / \partial \mathbf{W} = \mathbf{X}^\top\mathbf{G}$ と $\partial J / \partial \mathbf{X} = \mathbf{G}\mathbf{W}^\top$（行列積 $\mathbf{A} = \mathbf{X}\mathbf{W}$ に対して）、行方向の和 $\mathbf{G}^\top\mathbf{1}$（ブロードキャストされたバイアスに対して）、および要素ごとのゲート処理 $\mathbf{G} \odot \phi'(\mathbf{Z})$（活性化に対して）であり、パラメータがデータ損失と重み減衰の両方に入力される箇所では流入する勾配を合算する。$\mathbf{X}^\top\mathbf{G}$ は各層への順伝播の入力に依存するため、訓練時には順方向スイープ全体にわたって中間の活性化値をキャッシュする必要があり、バッチサイズと深さに対して線形に増大する $O(nKh)$ のメモリを要する。その見返りとして、1 回の逆方向スイープだけで、1 回の順伝播の高々約 2 倍の FLOPs ですべてのパラメータ勾配を計算できる。

<!--en:2294243a-->
## 演習問題

<!--en:449eeca1-->
1. **層のヤコビ行列。**スカラー $J$ の $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ に関する勾配は、$\mathbf{W}^{(2)}$ と同じ形状をもつ。一方、行列 $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} \in \mathbb{R}^{n \times q}$ の $\mathbf{W}^{(2)}$ に関する導関数はより大きな対象であり、4 つの添字のあらゆる組み合わせに対して 1 つずつ成分 $\partial o_{ij} / \partial w_{rt}$ をもつ。
    1. $\partial \mathbf{O} / \partial \mathbf{W}^{(2)}$ の成分数はいくつか。また、そのうち非ゼロになり得るものはいくつあるか。
    1. 連鎖律 $\partial J / \partial w_{rt} = \sum_{i, j} (\partial J / \partial o_{ij})(\partial o_{ij} / \partial w_{rt})$ を計算し、$\mathbf{H}^\top \partial J / \partial \mathbf{O}$ を導出せよ。
    1. $\partial \mathbf{O} / \partial \mathbf{W}^{(2)}$ は、:numref:`sec_mlp-implementation` の MLP で $n = 256$ としたとき、32 ビット浮動小数点でどれだけのメモリを消費するか。また、誤差逆伝播法が決してこれを構成しない理由を説明せよ。

<!--en:808a22a0-->
1. **より深いネットワークのメモリ。**$d$ 個の入力、$q$ 個の出力、および $K$ 個の層（そのうち $K - 1$ 個の隠れ層は幅 $h$ の ReLU 層）をもち、$n$ 個のデータ例からなるミニバッチで訓練される MLP を考える。
    1. 本文中の $K$ 層に対するステップを用いて、逆伝播が読み取る値を列挙し、その個数を求めよ。また、パラメータの個数を求めよ。
    1. 予測時には、次の層が使い終わり次第、各値を破棄できる。予測が任意の時点で保持する値の最大個数を求めよ。
    1. $d = 784$、$h = 256$、$q = 10$ のとき、保存される値の個数がパラメータ数を上回る最小のミニバッチサイズを、$K = 2$ および $K = 11$ のそれぞれについて求めよ。

<!--en:1db58f6f-->
1. [code] **勾配チェック。**自動微分が登場する以前、手書きの逆伝播は有限差分と比較して検証されていた。
    1. 1 つの成分 $w$ を $\mathbf{W}^{(1)}$ から選び、`mlp_backward` による勾配を、中心差分 $\bigl(J(w + \epsilon) - J(w - \epsilon)\bigr) / (2\epsilon)$（$\epsilon = 10^{-1}, 10^{-2}, \ldots, 10^{-7}$）と 32 ビット浮動小数点および 64 ビット浮動小数点のそれぞれで比較せよ。相対誤差を $\epsilon$ に対して両対数軸でプロットせよ。
    1. 曲線の形状を説明せよ。$\epsilon$ を小さくするとき減少するのはどの誤差で、増加するのはどの誤差か。また、それぞれの精度において両者が釣り合うのはどこか。
    1. 活性化前の値が 0 から $\epsilon$ 以内にあるとき、勾配が正しくてもこのチェックが失敗することがある。その理由と回避方法を説明せよ。

<!--en:5fc8b664-->
1. **フォワードモード。**誤差逆伝播法は出力に $\partial J / \partial J = 1$ をシードとして与え、グラフを逆方向にスイープする。これに対し、フォワードモード微分は選んだ 1 つのスカラーパラメータに $\partial w / \partial w = 1$（対象パラメータ $w$ に対する微分）をシードとして与え、順方向にスイープする。各ノードは、入力の導関数から自身の値の $w$ に関する導関数を計算する。
    1. フォワードモードを用いて、$\partial L / \partial \mathbf{W}^{(1)}$ を :eqref:`eq_backprop-example` のネットワークについて再計算せよ（$\mathbf{W}^{(1)}$ の成分ごとに 1 回のスイープを行うこと）。
    1. このネットワークの 6 個すべての重み勾配を得るには、各モードで何回ずつスイープが必要か。リバースモードでは 1 回しか計算しない積のうち、フォワードモードが毎回のスイープで繰り返し計算するのはどの積か。

    *Simon Prince『[Understanding Deep Learning](https://udlbook.github.io/udlbook/)』問題 7.12 および 7.13 を改変。*

<!--en:346234ed-->
1. [extended] **自動微分エンジン。** ● 小型の自動微分エンジンを構築し、それを用いて :eqref:`eq_backprop-example` のネットワークの勾配を再計算せよ。
    1. 各計算結果について、その入力と生成元の演算（計算グラフ）を記録するスカラーの `Value` クラスを作成し、`+`、`*`、`relu` をサポートせよ（*ヒント：*`__add__` と `__mul__` を実装し、それぞれが親ノードへの参照と勾配を 1 ステップ伝播させる小さな関数を保持する新しい `Value` を返すようにする）。
    1. `backward()` を実装せよ。グラフをトポロジカルソートし、出力の勾配に $1$ をシードとして与え、逆順にノードをたどって各ノードから親ノードへ勾配を渡すようにせよ。1 つの値が複数回使われる場合に勾配が*累積*される（`+=` であり、`=` ではない）ようにすること。
    1. 作成したエンジンを用いて :numref:`fig_mdl-mlp-backprop-small` の計算グラフと :eqref:`eq_backprop-example` の計算例を再計算し（行列積はスカラー演算に展開せよ）、得られた勾配を本節で導出した勾配と比較せよ。
    1. スカラー入力 $x$ から $v_i = f_i(x)$（$i = 1, 2, 3$）、中間値 $m = v_1 + v_2 + v_3$、値 $p_j = g_j(m)$（$j = 1, 2, 3$）、および損失 $L = p_1 + p_2 + p_3$ が順に生成される連鎖を考える。$\alpha_i = f_i'(x)$、$\beta_j = g_j'(m)$ とおく。経路上の和としての連鎖律は $3 \times 3 = 9$ 本の経路（$x$ から $L$ への経路）を列挙し、各経路が積 $\alpha_i \beta_j$ を寄与する。これに対し、作成したエンジンは $(\alpha_1 + \alpha_2 + \alpha_3)(\beta_1 + \beta_2 + \beta_3)$ を計算すること（$m$ で勾配を 1 回だけ累積することによる）を示し、この共有によって深層ネットワークでも誤差逆伝播法が現実的な計算量に収まる理由を説明せよ。

    *Andrej Karpathy の [micrograd](https://github.com/karpathy/micrograd)（`micrograd/engine.py`）を改変。*
   :exlabel:`ex_backprop-autograd-engine`

<!--en:d24f2249-->
[ディスカッション](https://d2l.discourse.group/t/102)

<!--en:cefa09b6-->
MLP における誤差逆伝播法と計算グラフ<br>**リバースモード DAG・局所的な行列規則・5 行で書ける MLP の逆伝播・活性化メモリと FLOP コスト**

<!--en:ffe91e05-->
::: {.slide title="1 回の逆伝播ですべての 203,530 個の勾配が求まる理由"}

<!--en:00b9344a-->
[ブラックボックスを開く]{.kicker}

<!--en:150ba285-->
- **有限差分とリバースモードの比較：**各スカラー重み $w \pm \epsilon$ を摂動させるとステップごとに $203{,}530$ 回の順伝播が必要になるが、リバースモード微分は*すべて*のパラメータ勾配を **1 回の順伝播の高々 $\sim 2\times$ の FLOPs** で計算する

<!--en:7dad2e69-->
- **中間層の誤差の共有：**誤差逆伝播法は各層の上流感度 $\partial J / \partial \mathbf{Z}^{(k)}$ を出力側から入力側へ *1 回だけ*計算し、1 回の行列積で $\partial J / \partial \mathbf{W}^{(k)}$ を求める

<!--en:dae56e1b-->
- **訓練のメモリコスト：**$\mathbf{H}^{(k-1)\top}\mathbf{G}_{\mathbf{Z}^{(k)}}$ を計算するには、すべての隠れ層における順伝播の活性化値を $O(nKh)$ の GPU RAM にキャッシュする必要がある

<!--en:80281870-->
::: {.slide title="3 ノードの計算グラフ：順伝播と逆伝播"}

<!--en:c166b87e-->
[計算グラフ・スカラーでの構造]{.kicker}

<!--en:f8e88bd0-->
リバースモード微分を最も単純な形で見るため、$e = (a + b)\,c$ を $a = 2, b = 1, c = -3$ で追ってみよう。

<!--en:1c1dfb36-->
- **順伝播（左から右）：**中間値を評価してキャッシュする（$u = a + b = 3$、次いで $e = u\,c = -9$）

<!--en:7d09f9a6-->
- **逆伝播（右から左）：**シード $\partial e / \partial e = 1$ を与え、逆トポロジカル順序でノードごとに 1 つの局所導関数を適用する
  - **乗算ノード（$e = u\,c$）：**キャッシュされた*他方*の入力と入れ替えて掛ける（$\partial e / \partial u = c = -3$、$\partial e / \partial c = u = 3$）
  - **加算ノード（$u = a + b$）：**入力された勾配をそのまま両方の枝へ渡す（$\partial e / \partial a = \partial e / \partial b = -3$）

<!--en:31d7c373-->
::: {.slide title="正則化された MLP の計算グラフ"}

<!--en:c632b454-->
[計算グラフ・行列 DAG]{.kicker}

<!--en:eadc9beb-->
ミニバッチ $\mathbf{X} \in \mathbb{R}^{n \times d}$ に対し、$\ell_2$ 重み減衰 $s = \frac{\lambda}{2}(\|\mathbf{W}^{(1)}\|_{\textrm{F}}^2 + \|\mathbf{W}^{(2)}\|_{\textrm{F}}^2)$ を伴う 1 隠れ層 MLP は有向非巡回グラフを形成する。

<!--en:5a589966-->
- **順伝播の連鎖：**$\mathbf{Z} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$ $\to$ $\mathbf{H} = \phi(\mathbf{Z})$ $\to$ $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ から次を得る
  $$J = L(\mathbf{O}, \mathbf{Y}) + s$$

<!--en:885d08d9-->
- **分岐するパラメータノード（網掛け）：**各重み行列 $\mathbf{W}^{(k)}$ は、そのアフィン層とフロベニウスペナルティ $s$ という *2 つ*の子ノードに入力される

<!--en:a7ab9a20-->
- **分岐における多変数の連鎖律：**ノードが複数の下流経路に入力される場合、その逆伝播の勾配はすべての出力辺から戻ってくる勾配の**和**（`+=`）になる

<!--en:fb6750af-->
::: {.slide title="局所規則 1：行列積 A = XW"}

<!--en:304ee12b-->
[局所的な行列微分]{.kicker}

<!--en:0c4a8f41-->
- **成分展開：**成分 $w_{rj}$ はすべてのデータ例 $i$ にわたって列 $j$ に影響を与え（$\sum_i x_{ir} g_{ij}$）、成分 $x_{ir}$ はすべての出力ユニット $j$ にわたって行 $i$ に影響を与える（$\sum_j g_{ij} w_{rj}$）

<!--en:8f36b97e-->
- **形状の一致による確認：**$\mathbf{X}^\top\mathbf{G}$ は $(n \times d)$ と $(n \times h)$ を掛けて $(d \times h)$ にする*唯一*の方法であり、キャッシュされた順伝播の入力 $\mathbf{X}$ を必要とする

<!--en:00b8514b-->
::: {.slide title="局所規則 2・3：ブロードキャストされたバイアスと活性化"}

<!--en:304ee12b-2-->
[局所的な行列微分]{.kicker}

<!--en:e634be2f-->
- **ブロードキャストされたバイアス（$\mathbf{Z} = \mathbf{A} + \mathbf{b}$）：**$\mathbf{G}$ をそのまま $\mathbf{A}$ にコピーし、バッチ軸に沿って合計する（`G.sum(axis=0)`）

<!--en:93d29a19-->
- **要素ごとの活性化（$\mathbf{H} = \phi(\mathbf{Z})$）：**上流の勾配を成分ごとに（アダマール積 $\odot$ で）局所的な傾き $\phi'(z_{ij})$ によってゲートする

<!--en:b68ad3e4-->
::: {.slide title="MLP における完全な 5 行の逆伝播"}

<!--en:f7d3012f-->
[逆方向スイープの組み立て]{.kicker}

<!--en:f545a8c8-->
- **トップダウンの貢献度割り当て：**ソフトマックスのロジットにシード $\mathbf{G}_{\mathbf{O}} = \frac{1}{n}(\hat{\mathbf{Y}} - \mathbf{Y})$ を与え、$\nabla\mathbf{W}^{(2)} = \mathbf{H}^\top\mathbf{G}_{\mathbf{O}} + \lambda\mathbf{W}^{(2)}$ を求める

<!--en:08e384b5-->
- **隠れ層への誤差伝播：**$\mathbf{G}_{\mathbf{O}}\mathbf{W}^{(2)\top}$ を隠れユニットへ逆射影し、$\odot\,\phi'(\mathbf{Z})$ でゲートして $\nabla\mathbf{W}^{(1)} = \mathbf{X}^\top\mathbf{G}_{\mathbf{Z}} + \lambda\mathbf{W}^{(1)}$ を構成する

<!--en:21e3f219-->
::: {.slide title="K 層への一般化：逆伝播の漸化式"}

<!--en:66910ece-->
[深層 MLP の漸化式]{.kicker}

<!--en:ae9382f3-->
- **2 ステップの層モジュール：**各隠れ層 $k = K - 1, \ldots, 1$ は上流の誤差を $\mathbf{W}^{(k+1)\top}$ で射影し、$\odot\,\phi'(\mathbf{Z}^{(k)})$ でゲートして $\mathbf{H}^{(k-1)\top}\mathbf{G}_{\mathbf{Z}^{(k)}}$ を求める

<!--en:2140e1df-->
- **深さにわたる積の連鎖：**$\mathbf{W}^{(k)\top}$ と $\phi'(\mathbf{Z}^{(k)})$ を繰り返し掛ける構造から、深層ネットワークで勾配が消失または爆発する理由が説明できる

<!--en:0fe6ad9f-->
::: {.slide title="数値計算例：順伝播と逆伝播の手計算"}

<!--en:0a27393b-->
[計算例・数値]{.kicker}

<!--en:f5c368f1-->
- **設定（$2 \to 2 \to 1$ ReLU）：**$\mathbf{X} = [1, 2]$、$\mathbf{W}^{(1)} = \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix}$、$\mathbf{W}^{(2)} = [2, -1]^\top$、$y = 0$、$L = \frac{1}{2}(o - y)^2$

<!--en:de1aec6c-->
- **順伝播：**$\mathbf{Z} = [-1, 2] \to \mathbf{H} = [0, 2] \to o = -2 \to L = 2$

<!--en:ad645a55-->
- **逆伝播：**$\partial L/\partial o = -2 \implies \nabla\mathbf{H} = [-4, 2] \implies \nabla\mathbf{Z} = [-4, 2] \odot [0, 1] = [0, 2]$ となり、$\nabla\mathbf{W}^{(1)}$ の 1 列目が 0 になる

<!--en:a3be2b57-->
::: {.slide title="逆伝播グラフにおける死んだ ReLU ゲートの可視化"}

<!--en:b9fee103-->
[計算例・グラフ]{.kicker}

<!--en:d8c8e724-->
計算グラフ上の順伝播の値（黒）とリバースモードの導関数（赤）を調べることで、**死んだ ReLU ユニット**がどのようにして学習を停止させるかが機械的にわかる。

<!--en:b0688692-->
- **0 になる入力列：**ユニット 1 の上流感度 $\partial L / \partial h_1 = -4$ が大きくても、活性化前の値が負（$z_1 = -1 < 0$）であるため ReLU の傾きは $\phi'(z_1) = 0$ になる

<!--en:63815ea4-->
- **データセット全体で非活性なら永久に停滞：**ゲートによって $\partial L / \partial z_1 = 0$ になると $\partial L / \partial \mathbf{W}^{(1)}$ の 1 列目全体が 0（$\begin{bmatrix} 0 & 0 \end{bmatrix}^\top$）になり、$\mathbf{W}^{(1)}$ の 1 列目は決して更新されない

<!--en:550d1a10-->
::: {.slide title="5 行の逆伝播のコード実装"}

<!--en:871c4e3e-->
[自動微分に対する逆伝播の検証]{.kicker}

<!--en:7cd802aa-->
5 つの行列方程式をそのまま `mlp_backward(X, Y, W1, b1, W2, b2, lambd)` として実装する。

<!--en:ade79fab-->
なお ReLU の場合、$\mathbf{H}$（または $\mathbf{Z}$）を保存しておけば、二値マスク `(Z > 0)` を `G_Z = (G_O @ W2.T) * (Z > 0)` の行で評価するのに十分である。

<!--en:5d96c8cb-->
::: {.slide title="Float64 による勾配チェック：自動微分と 10⁻¹⁵ まで一致"}

<!--en:871c4e3e-2-->
[自動微分に対する逆伝播の検証]{.kicker}

<!--en:25b33e0d-->
`mlp_backward` とフレームワークの `autograd` の両方を 64 ビット浮動小数点（`float64`、マシンイプシロン $\approx 2.2 \times 10^{-16}$）でランダムな $256 \times 784$ のミニバッチに対して実行すると、導出結果が相対誤差 $10^{-15}$ 以内で検証できる。

<!--en:14347382-->
::: {.slide title="訓練が推論より多くのメモリを消費する理由"}

<!--en:24f0683f-->
[システムコスト・活性化メモリ]{.kicker}

<!--en:cb40bdc7-->
- **推論（$O(nh)$ RAM）：**次の層の計算が終わり次第、各層の活性化バッファを上書きできる

<!--en:b197aa82-->
- **訓練（$O(nKh)$ RAM）：**$\mathbf{X}$（$0.80\textrm{ MB}$）、$\mathbf{H}$（$0.26\textrm{ MB}$）、$\hat{\mathbf{Y}}$（$0.01\textrm{ MB}$）を、逆伝播が $\mathbf{H}^{(k-1)\top}\mathbf{G}_{\mathbf{Z}^{(k)}}$ を計算するまでキャッシュしなければならない

<!--en:27d97f97-->
- **バッチサイズに対するスケーリング：**1 隠れ層 MLP でさえ $n = 256$ ではキャッシュされた活性化値（$1.08\textrm{ MB}$）がモデルパラメータ（$0.81\textrm{ MB}$）を上回る

<!--en:c5d73f20-->
::: {.slide title="演算コスト：深層ネットワークで逆伝播が順伝播の ~2× かかる理由"}

<!--en:1848d3ca-->
[システムコスト・FLOPs]{.kicker}

<!--en:58a4b519-->
- **順伝播（画像あたり $203{,}264$ MACs）：**層ごとに 1 回の行列積（$\mathbf{X}\mathbf{W}^{(1)}$ と $\mathbf{H}\mathbf{W}^{(2)}$）を計算する

<!--en:92762c7a-->
- **逆伝播（画像あたり $205{,}824$ MACs）：**内部層の各順伝播の行列積 $\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}$ から、同じ FLOP 数の逆伝播の行列積が **2 つ**生じる（$\mathbf{H}^{(k-1)\top}\mathbf{G}$ と $\mathbf{G}\mathbf{W}^{(k)\top}$）

<!--en:799c3afa-->
- **3× の経験則：**この 2 層 MLP では逆伝播 ≈ 順伝播（最初の層は $\partial J/\partial \mathbf{X}$ を省略）だが、深層ネットワークではほぼすべての層が内部層となるため、逆伝播は順伝播の $\approx 2\times$ の FLOPs、訓練 1 ステップ全体では推論の $\approx 3\times$ の FLOPs を要する

<!--en:043907fd-->
::: {.slide title="リバースモードとフォワードモード・有限差分の比較"}

<!--en:9420b48e-->
[システムコスト・アルゴリズムの比較]{.kicker}

<!--en:54396114-->
スカラー損失 $J: \mathbb{R}^P \to \mathbb{R}$（$P = 203{,}530$）に対し、深層学習がフォワードモード微分や数値的な有限差分ではなくリバースモードの誤差逆伝播法を用いるのはなぜか。

<!--en:01373feb-->
| 微分手法 | 1 回のスイープで計算するもの | $\nabla J \in \mathbb{R}^P$ に必要なスイープ回数 | 活性化メモリ | ステップあたりの総 FLOPs（$P = 203{,}530$） |

<!--en:e3dcea2a-->
| **有限差分** $\frac{J(w+\epsilon)-J(w-\epsilon)}{2\epsilon}$ | $1$ 個のスカラー方向微分 | $P$ 回（中心差分なら $2P$ 回） | $O(nh)$（キャッシュ不要） | 順伝播の $\sim 203{,}530 \times$（$\sim 4 \times 10^{10}$） |

<!--en:c19f1d9c-->
| **フォワードモード**（シード $\dot{w}_i = 1$） | 接線方向微分 $\partial (\cdot) / \partial w_i$（重み 1 個分） | 順方向スイープ $P$ 回 | $O(nh)$（キャッシュ不要） | 順伝播の $\sim 203{,}530 \times$ |

<!--en:ef868ba6-->
| **リバースモード（逆伝播）**（シード $\bar{J} = 1$） | **すべて**の重みに対する随伴変数 $\partial J / \partial (\cdot)$ | **逆方向スイープ $1$ 回** | **$O(nKh)$ をキャッシュ** | **順伝播の $\sim 3 \times$（$\sim 6 \times 10^5$）** |

<!--en:f75f89af-->
リバースモードは多入力のスカラー出力に対して $O(1)$ 回のスイープで済む（$P \gg 1$）のに対し、フォワードモードが有利になるのは出力数が入力数を上回る場合（$q \gg P$）に限られる。

<!--en:b73cda75-->
::: {.slide title="理解度チェック：形状・メモリスケーリング・ヤコビ行列"}

<!--en:ffcb46a8-->
[確認]{.kicker}

<!--en:4281f3c3-->
誤差逆伝播法の仕組みとシステム上のスケーリングについて理解を確認しよう。

<!--en:3b454001-->
1. **外積の和：**バッチサイズ $n = 256$ の Fashion-MNIST MLP において、$\mathbf{X}^\top\mathbf{G}_{\mathbf{Z}}$ の形状は $(784, 256) \times (256, 256) = (784, 256)$ である。データ例 $i = 1, \ldots, 256$ にわたる和として書いたとき、各データ例はどのようなランク 1 行列を寄与するか。

<!--en:16d46932-->
2. **10 層の活性化メモリ：**この MLP を $10$ 個の隠れ層（各層の幅 $256$、$n = 256$）に深くした場合、訓練時にミニバッチあたり何 MB の `float32` 活性化値をキャッシュする必要があるか。またパラメータメモリと比べるとどうか。

<!--en:e6a84cac-->
3. **なぜ完全なヤコビ行列を作らないのか：**完全な 4D テンソル $\partial \mathbf{O} / \partial \mathbf{W}^{(2)}$（形状 $(n, q, h, q) = (256, 10, 256, 10)$）を明示的に実体化すると、`float32` のメモリをどれだけ占有するか。

<!--en:557b98b6-->
**解答：**

<!--en:53e1aa1d-->
1. **外積の和：**$\mathbf{X}^\top\mathbf{G}_{\mathbf{Z}} = \sum_{i=1}^{256} \mathbf{x}_i \mathbf{g}_{\mathbf{z}, i}^\top$（ここで $\mathbf{x}_i \in \mathbb{R}^{784}$ はデータ例 $i$ の入力、$\mathbf{g}_{\mathbf{z}, i} \in \mathbb{R}^{256}$ はその活性化前の勾配）

<!--en:f3de1c92-->
2. **深さ 10 でのメモリ：**$\mathbf{X}$（$200{,}704$）+ $10 \times \mathbf{H}^{(k)}$（$10 \times 65{,}536$）+ $\hat{\mathbf{Y}}$（$2{,}560$）$= 858{,}624$ 個の浮動小数点数 $\approx 3.43\textrm{ MB}$（$795{,}658$ 個のパラメータ $\approx 3.18\textrm{ MB}$ を上回る）

<!--en:d32644ad-->
3. **完全なヤコビ行列の肥大化：**小さな出力層だけでも $256 \times 10 \times 256 \times 10 \times 4\textrm{ B} \approx 26.2\textrm{ MB}$ になるが、ベクトル・ヤコビ行列積は $\mathbf{H}^\top\mathbf{G}_{\mathbf{O}}$ を直接 $256 \times 10$（$10\textrm{ KB}$）へ縮約する

<!--en:d95790c1-->
::: {.slide title="まとめ：多層パーセプトロンにおける誤差逆伝播法"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:5df37c66-->
| 順伝播の演算 | キャッシュするテンソル | 逆伝播のパラメータ勾配 | 逆伝播の入力勾配 | FLOP・メモリスケーリング |

<!--en:80ccfd86-->
| **線形層 $\mathbf{A} = \mathbf{X}\mathbf{W}$** | $\mathbf{X} \in \mathbb{R}^{n \times d}$、$\mathbf{W} \in \mathbb{R}^{d \times h}$ | $\frac{\partial J}{\partial \mathbf{W}} = \mathbf{X}^\top\mathbf{G}$ | $\frac{\partial J}{\partial \mathbf{X}} = \mathbf{G}\mathbf{W}^\top$ | 逆伝播で $2 \times n d h$ 回の積和演算 |

<!--en:04c254f1-->
| **ブロードキャストバイアス $\mathbf{Z} = \mathbf{A} + \mathbf{b}$** | なし | $\frac{\partial J}{\partial \mathbf{b}} = \mathbf{G}^\top\mathbf{1}_n$（行方向の和） | $\frac{\partial J}{\partial \mathbf{A}} = \mathbf{G}$ | $O(nh)$ の和、追加キャッシュ不要 |

<!--en:6abceaea-->
| **活性化 $\mathbf{H} = \phi(\mathbf{Z})$** | $\mathbf{Z}$（ReLU なら $\mathbf{H}$ でも可） | なし | $\frac{\partial J}{\partial \mathbf{Z}} = \mathbf{G} \odot \phi'(\mathbf{Z})$ | $O(nh)$ の要素ごとの演算、$\phi' = 0$ で不活性化 |

<!--en:05c831e6-->
| **重み減衰 $s = \frac{\lambda}{2}\|\mathbf{W}\|_{\textrm{F}}^2$** | $\mathbf{W}$ | $+\lambda\mathbf{W}$ を $\frac{\partial J}{\partial \mathbf{W}}$ に加算 | — | DAG の分岐で入力勾配を合算 |

<!--en:382af61b-->
| **$K$ 層 MLP 全体のスイープ** | $\mathbf{X}, \mathbf{H}^{(1 \dots K-1)}, \hat{\mathbf{Y}}$ | 1 回の逆方向パスですべての $\mathbf{W}^{(k)}, \mathbf{b}^{(k)}$ | $\mathbf{G}\mathbf{W}^{(k)\top} \odot \phi'$ を伝播 | 順伝播の $\sim 2\times$ FLOPs、$O(nKh)$ RAM |
