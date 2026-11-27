<!--en:5a7d784a-->
# 線形代数

<!--en:a2ee5897-->
画像のミニバッチを $255$ で割る操作は各ピクセルを個別にスケーリングするだけだが、予測モデルでは数百から数千もの入力特徴量を組み合わせて判断を下さなければならない。*線形代数*（linear algebra）はそのような重み付き結合を記述する数学の言語であり、ミニバッチ全体に対する 1 つの層の順伝播（forward pass）を簡潔な 1 本の式で表し、出力の形状や浮動小数点演算のコストを実行前に読み取ることを可能にする。

<!--en:461edbe0-->
$28 \times 28$ のグレースケール衣類画像（:numref:`chap_classification`）を $10$ 個のカテゴリに分類する線形画像分類器（image classifier）を考えよう。各写真を平坦化すると $28 \times 28 = 784$ ピクセルになる。平坦化した $256$ 枚の画像のミニバッチをデータ行列 $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ の行として積み重ね、$784$ 個ずつの重みを $10$ 個のカテゴリそれぞれについて $\mathbf{W} \in \mathbb{R}^{784 \times 10}$ の列として並べ、$10$ 個のカテゴリごとのオフセットを $\mathbf{b} \in \mathbb{R}^{10}$ にまとめると（パラメータ数は合計 $7{,}850$ 個）、ミニバッチ全体の $256 \times 10 = 2{,}560$ 個のカテゴリスコアは次の式で与えられる。

<!--en:81a6ff34-->
ここで、$\mathbf{b}$ はブロードキャストによってすべての行に加算される（:numref:`fig_la_matmul`）。

<!--en:937a7676-->
![256 枚の画像と 10 個のカテゴリからなるミニバッチに対する線形分類器のスコア。$\mathbf{X}$ の第 $i$ 行は画像 $i$ を保持し、$\mathbf{W}$ の第 $j$ 列はカテゴリ $j$ の重みを保持し、この 2 つからスコア $o_{ij}$ が定まる。](../img/linear-algebra-matmul.svg)

<!--en:d4361331-->
本節では、スカラー、ベクトル、行列、テンソルから出発し、リダクション、ドット積、行列積を経て $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ へと積み上げ、最後に*ノルム*と*固有値*（eigenvalue）を扱う（記法の規約は :numref:`chap_notation` にまとめられている）。

<!--en:2a9f53fc-->
## スカラー、ベクトル、行列、テンソル

<!--en:c7b8c99c-->
$0^\textrm{th}$、$1^\textrm{st}$、$2^\textrm{nd}$、$4^\textrm{th}$ 階の配列の作成と形状変更はすでに :numref:`sec_ndarray` で扱った。ここでは、各階数（コードでは `ndim`）を標準的な数学上の名称および記法と対応づける。

<!--en:3899e623-->
### スカラー

<!--en:3968eed2-->
単一の数値は*スカラー*と呼ばれ、通常の小文字イタリック体（$x, y, z$）で表される。$x \in \mathbb{R}$ と書けば、$x$ が実数であることを表す。コードでは、スカラーは $0^\textrm{th}$ 階のテンソルであり、空の形状 `()` を持つ。

<!--en:c5566523-->
### ベクトル

<!--en:051856e6-->
*ベクトル*は固定長のスカラーの 1 次元配列（スカラーはその*要素*または*成分*）であり、太字の小文字（$\mathbf{x}, \mathbf{y}, \mathbf{w}$）で表される。$\mathbf{x} \in \mathbb{R}^n$ は $n$ 次元の実ベクトルを表し、特に断らない限り*列ベクトル*（column vector）として扱う。

<!--en:51083aca-->
文中にインラインで書く場合は $\mathbf{x} = (x_1, \ldots, x_n)$ とも表記する。コードでは、ベクトルは $1^\textrm{st}$ 階のテンソルである。

<!--en:ccce12fe-->
数学の記法では 1 始まりのインデックス（$x_1, \ldots, x_n$）を使うのに対し、Python では 0 始まりのインデックスを使うため、3 番目の成分 $x_3$ には `x[2]` でアクセスする。

<!--en:4a1cb6f9-->
ベクトルの*次元数*（dimensionality）$n$（成分の個数）は、Python の組み込み関数 `len(x)` によって、あるいは要素数 1 のタプルである形状 `x.shape` の唯一の要素として得られる。

<!--en:4d0ee512-->
「次元」という言葉が曖昧に使われることによる混乱を防ぐため、本書ではテンソルの軸の数（`ndim`）には*階数*（order）、ベクトルの成分の数には*次元数*という用語を充てる。

<!--en:ffb9bb37-->
### 行列

<!--en:12b176ce-->
*行列*は $2^\textrm{nd}$ 階のテンソルであり、太字の大文字（$\mathbf{A}, \mathbf{X}, \mathbf{W}$）で表される。$\mathbf{A} \in \mathbb{R}^{m \times n}$ と書けば $m$ 行 $n$ 列であることを示し（$m = n$ のとき $\mathbf{A}$ は*正方行列*（square matrix）である）、成分 $a_{ij}$ が第 $i$ 行・第 $j$ 列に並ぶ。

<!--en:5c7f3f90-->
表形式データや平坦化したミニバッチの行列 $\mathbf{X} \in \mathbb{R}^{n \times d}$ では、1 行につき 1 つのデータ例、1 列につき 1 つの特徴量を格納する。以下では、$2 \times 3$ 行列 `A` を構築する。

<!--en:fd02dfe6-->
行列の行と列を入れ替えると、その*転置*（transpose）$\mathbf{A}^\top \in \mathbb{R}^{n \times m}$（成分は $[\mathbf{A}^\top]_{ij} = a_{ji}$）が得られ、コードでは `.T` でアクセスする。

<!--en:ca183af4-->
TensorFlow では `tf.transpose(A)` を使用する。

<!--en:1907be66-->
正方行列は、$\mathbf{S} = \mathbf{S}^\top$（$s_{ij} = s_{ji}$ がすべての $i, j$ で成り立つ）のとき*対称行列*（symmetric matrix）である。

<!--en:643e0fff-->
### 高階テンソル

<!--en:0a1e06b5-->
3 つ以上の軸を持つテンソルは、サンセリフ体の大文字（$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$）で表される。以下では、合成した 4 次元の画像ミニバッチ `images`（形状は $(256, 1, 28, 28)$、要素は $[0, 1]$ の範囲）を生成し、各画像を平坦化して 2 次元行列 $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ を作る。

<!--en:0311aff7-->
## 要素ごとの演算とリダクション

<!--en:0c83abb3-->
### 要素ごとの演算とアダマール積

<!--en:cff3b51a-->
同じ形状を持つ 2 つの行列 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$ の要素ごとの積は*アダマール積*（Hadamard product）$\mathbf{A} \odot \mathbf{B}$ と呼ばれ、その成分は $[\mathbf{A} \odot \mathbf{B}]_{ij} = a_{ij} b_{ij}$ であり、コードでは `*` で計算する。

<!--en:73c269a9-->
### 軸に沿ったリダクション

<!--en:6efc58b8-->
`axis=0` を `.sum()` に渡すと行方向にわたって和をとり列和のベクトル（$\sum_{i=1}^m a_{ij}$）を生成するのに対し、`axis=1` を渡すと列方向にわたって和をとり行和のベクトル（$\sum_{j=1}^n a_{ij}$）を生成する。`axis` を省略すると全要素の和をとる（:numref:`fig_la_reduce`）。

<!--en:acbfb0a9-->
![$2 \times 3$ 行列のリダクション。軸 0 にわたって和をとると列ごとに 1 つの和が残り、軸 1 にわたって和をとると行ごとに 1 つの和が残る（リダクションされた軸は形状から消える）。`keepdims=True` を指定すると行和は形状 $(2, 1)$ を保つため、ブロードキャストによって 3 つの列へと引き伸ばされ、除算 `A / sum_A` によって各行の和が 1 になるようスケーリングされる。](../img/linear-algebra-reduce-axes.svg)

<!--en:474bb3c2-->
### ブロードキャストのためのリダクション軸の保持

<!--en:0e6ab631-->
`A`（形状 `(2, 3)`）の各行をその行和で正規化しようとしても、`A.sum(axis=1)`（形状 `(2,)`）は `(2, 3)` に対してブロードキャストできない。末尾の次元 $2$ と $3$ が衝突するためである。`keepdims=True` を渡すとリダクションされた軸が長さ $1$ のまま保持され、`sum_A` の形状は `(2, 1)` になる。

<!--en:8140f768-->
PyTorch では `dim`・`keepdim` と、NumPy 互換の別名である `axis`・`keepdims` の両方を受け付ける。

<!--en:7ffcb41b-->
TensorFlow のテンソルには `.sum` メソッドや `.mean` メソッドがないため、代わりに `tf.reduce_sum(A, axis=1)` と `tf.reduce_mean(X, axis=1)` を使用する。

<!--en:5519a55b-->
`sum_A` の形状が `(2, 1)` であるため、`A / sum_A` で割ると各行の和が 3 つの列すべてにブロードキャストされ、得られる $2 \times 3$ 行列のどの行も和が $1$ になる。

<!--en:0e6774b4-->
$256 \times 784$ のミニバッチ $\mathbf{X}$ において、ピクセルごとの列平均（`axis=0`、形状 `(784,)`）を引く操作はそのままブロードキャストされるが、画像ごとの行平均（`axis=1`）を引くには `keepdims=True`（形状 `(256, 1)`）が必要である。

<!--en:f4df3712-->
注意すべき点として、もし $\mathbf{X}$ がたまたま正方行列（$784 \times 784$）であった場合、`keepdims=True` を `X - X.mean(axis=1)` で省略してもエラーは発生せず、画像 $j$ の平均がすべての画像の第 $j$ 列から暗黙のうちに引かれてしまう（演習問題 :exref:`ex_linear-algebra-shape-prediction`）。

<!--en:a7d9b6e0-->
## 積

<!--en:e3d2789e-->
### ドット積

<!--en:7ffcd610-->
2 つのベクトル $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ が与えられたとき、それらの*ドット積*、すなわち*内積*（inner product）$\mathbf{x}^\top \mathbf{y}$（$\langle \mathbf{x}, \mathbf{y} \rangle$ とも書く）は、対応する成分同士の積の総和である。

<!--en:d52c7195-->
コードでは、2 つの 1 次元ベクトル `x` と `y` のドット積をフレームワークの `dot` 関数で直接計算する。

<!--en:c060483d-->
TensorFlow では `tf.tensordot(x, y, axes=1)` で計算する。

<!--en:0c4b0fe5-->
同じ計算は、`x * y` で要素ごとに掛けてから得られた積を `sum` で足し合わせることでも実行できる（$\sum_i x_i = 0 + 1 + 2 = 3$ が $\mathbf{y} = \mathbf{1}$ のときに得られる）。

<!--en:ca72f865-->
### 行列・ベクトル積

<!--en:d2aaa11d-->
$\mathbf{A} \in \mathbb{R}^{m \times n}$ を $m$ 個の行ベクトル $\mathbf{a}_1^\top, \ldots, \mathbf{a}_m^\top \in \mathbb{R}^n$ に分割すると、*行列・ベクトル積*（matrix–vector product）$\mathbf{A}\mathbf{x}$（ただし $\mathbf{x} \in \mathbb{R}^n$）は、各行とのドット積を並べた $m$ 次元ベクトルになる（:numref:`fig_la_matvec`）。

<!--en:b9b0d86b-->
![行列・ベクトル積 $\mathbf{A}\mathbf{x}$ は $\mathbf{A}$ の行ごとに 1 つのドット積をとるため、$2\times3$ 行列は長さ 3 のベクトルを長さ 2 のベクトルに写す。1 行目からは $0 \cdot 0 + 1 \cdot 1 + 2 \cdot 2 = 5$ が得られる。](../img/linear-algebra-matvec.svg)

<!--en:6932d4c5-->
Python では、中置演算子 `@` によって行列・ベクトル積と行列・行列積の両方を計算できる。

<!--en:eae97a09-->
TensorFlow では `@` に 2 次元行列が必要であるため、行列・ベクトル積には `tf.linalg.matvec(A, x)` を使用する。

<!--en:4376ef49-->
列ごとに見れば、$\mathbf{A}\mathbf{x} = \sum_{j=1}^n x_j (\mathbf{A}\mathbf{e}_j)$ は $\mathbf{A}$ の列ベクトルの線形結合であり、ここで $j^\textrm{th}$ 列 $\mathbf{A}\mathbf{e}_j$ は $j^\textrm{th}$ 標準基底ベクトル（basis vector）$\mathbf{e}_j$ の像である。$\mathbb{R}^2$ において、$2 \times 2$ 行列は $\mathbf{e}_1, \mathbf{e}_2$ が張る単位正方形を、その列ベクトルが張る平行四辺形へと写す（:numref:`fig_la_maps`）。

<!--en:329f7bbe-->
![$2 \times 2$ 行列は単位正方形（破線）を、その列ベクトル $\mathbf{A}\mathbf{e}_1$ と $\mathbf{A}\mathbf{e}_2$ が張る平行四辺形に写す。左から順に、$90^\circ$ の回転、2 倍（$\mathbf{e}_1$ 方向）と $1/2$ 倍（$\mathbf{e}_2$ 方向）のスケーリング、せん断。](../img/mdl-prelim-linear-maps.svg)

<!--en:0ecd0b86-->
### 行列・行列積

<!--en:9c0abb6f-->
$\mathbf{A} \in \mathbb{R}^{m \times k}$ と $\mathbf{B} \in \mathbb{R}^{k \times n}$ が与えられたとき、それらの行列積 $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$ の $(i, j)$ 成分は、$\mathbf{A}$ の第 $i$ 行と $\mathbf{B}$ の第 $j$ 列のドット積に等しい。

<!--en:32fdda9b-->
これで、10 クラスの線形分類器 $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ を $256 \times 784$ のミニバッチ `X` に対して評価し、`argmax` を軸 $1$ に沿って適用することで各画像のスコアが最も高いカテゴリを選べる。

<!--en:38effaa7-->
$m \times k$ 行列と $k \times n$ 行列の積では、$mn$ 個の長さ $k$ のドット積を計算するため、コストは $\approx 2mkn$ FLOPs となる（$2 \times 256 \times 784 \times 10 \approx 4 \times 10^6$ FLOPs が `X @ W` にかかり、また $2n^3$ FLOPs がメモリ上のわずか $3n^2$ 個の要素に対して 2 つの $n \times n$ 行列の積で実行される。:numref:`chap_performance` を参照）。線形代数では 2 つの特別な正方行列が繰り返し登場する。1 つは*単位行列*（identity matrix）$\mathbf{I} \in \mathbb{R}^{n \times n}$（主対角成分が $1$ でそれ以外が $0$）であり、任意のベクトルを変化させない（$\mathbf{I}\mathbf{x} = \mathbf{x}$）。もう 1 つは*逆行列*（inverse）$\mathbf{A}^{-1}$ であり、可逆行列（invertible matrix）$\mathbf{A}$ に対して $\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ によって定義される（:numref:`subsec_mdl-inverses`）。

<!--en:7e144df5-->
後の章で扱うモデルでは、ミニバッチ内のデータ例ごとに 1 つずつ用意された行列のスタック全体を一度に掛け合わせる。`@` 演算子はこれをループなしで実行する。すなわち、3 つ以上の軸を持つテンソルに対しては最後の 2 つの軸で行列積を計算し、先頭側の軸にわたってブロードキャストする（:numref:`subsec_broadcasting`）。`@` では対応できない場合でも、`einsum` を使えば :eqref:`eq_matmul` の和を添字ごとに書き下せる。文字列 `'bil,lj->bij'` は各因子と結果の軸に名前を付けたものであり、結果に含まれない添字（ここでは `l`）について和がとられる。`A` の定数倍を 3 つ積み重ね、それらすべてに `B` を掛けてみよう。

<!--en:de5eb4a1-->
結果にはスタック内の行列ごとに 1 つずつ、計 3 個の $2 \times 4$ の積が格納されており、`einsum` の結果はすべての成分で `@` と一致する。

<!--en:5c071cc4-->
## ノルム

<!--en:9cdebe33-->
*ノルム*はベクトルや行列の大きさを測る量であり、機械学習では予測誤差 $\hat{\mathbf{y}} - \mathbf{y}$ を測る*損失関数*（loss function）としても、重みの大きさを制約する*正則化ペナルティ*（regularization penalty）としても用いられる。

<!--en:5343bea0-->
### ユークリッド長と角度

<!--en:b17eba08-->
*$\ell_2$ ノルム*（または*ユークリッドノルム*）は、$\mathbf{x} \in \mathbb{R}^n$ に対して次の式で定義される。

<!--en:fc956649-->
これはしばしば $\|\mathbf{x}\|$ と略記される。平面上の 2 つのベクトル $\mathbf{x}, \mathbf{y}$ が角度 $\theta$ をなすとき、余弦定理 $\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ と双線形展開 $(\mathbf{x} - \mathbf{y})^\top(\mathbf{x} - \mathbf{y}) = \|\mathbf{x}\|^2 - 2\mathbf{x}^\top\mathbf{y} + \|\mathbf{y}\|^2$ を等置すると、次式が得られる（:numref:`fig_la_dot`）。

<!--en:9894b2b0-->
任意の次元 $d$ において、:eqref:`eq_dot_cos` を逆に読むことで、非ゼロベクトル間の*コサイン類似度*（cosine similarity）を次のように定義する。

<!--en:a0f8c0a5-->
この量は向きのみに依存し、$+1$（同じ向き）から $0$（直交）を経て $-1$（反対向き）までの値をとる（:numref:`fig_la_cosine`）。

<!--en:86550ed3-->
![ドット積 $\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ は、$\mathbf{x}$ の長さと符号付き長さ $\|\mathbf{y}\|\cos\theta$（$\mathbf{y}$ を $\mathbf{x}$ を通る直線へ射影した長さ）の積である。](../img/linear-algebra-dot.svg)

<!--en:c8ee7197-->
![角度 $\theta$ をなす 2 つの単位ベクトル。それらのドット積は $\cos\theta$ に等しく、$1$（同じ向き）から $0$（垂直）を経て $-1$（反対向き）までの値をとる。](../img/mdl-prelim-cosine.svg)

<!--en:d016a4b3-->
*コーシー・シュワルツの不等式*（Cauchy–Schwarz inequality）により $|\mathbf{x}^\top\mathbf{y}| \leq \|\mathbf{x}\|\|\mathbf{y}\|$ が保証される（したがって $\cos\theta \in [-1, 1]$）。実際、単位ベクトル $\mathbf{u} = \mathbf{x}/\|\mathbf{x}\|$ と $\mathbf{v} = \mathbf{y}/\|\mathbf{y}\|$ に対して $0 \leq \|\mathbf{u} \mp \mathbf{v}\|^2 = 2 \mp 2\mathbf{u}^\top\mathbf{v}$ を展開すれば $|\mathbf{u}^\top\mathbf{v}| \leq 1$ が得られる。

<!--en:b97a9d32-->
### $\ell_1$ ノルム、$\ell_p$ ノルム、フロベニウスノルム

<!--en:9ffe2932-->
*$\ell_1$ ノルム*（または*マンハッタン距離*、Manhattan distance）は、各成分の絶対値の総和をとる（:numref:`fig_la_norms`）。

<!--en:1f37d52b-->
![$\mathbf{x}$ の $\ell_2$ ノルムは $\mathbf{x}$ への直線経路の長さであり、$\ell_1$ ノルムは格子に沿った経路の長さである。](../img/linear-algebra-norms.svg)

<!--en:626727cd-->
$\ell_2$ は成分を二乗するのに対し $\ell_1$ は二乗しないため、$\ell_1$ 損失は大きな外れ値（outlier）の誤差に対してより頑健であり、$\ell_1$ の重みペナルティは重みを厳密な 0 へと導く（スパース性、sparsity）。どちらも *$\ell_p$ ノルム*の族 $\|\mathbf{x}\|_p = (\sum_{i=1}^n |x_i|^p)^{1/p}$（$p \geq 1$）に属し、$\|\mathbf{x}\|_\infty = \max_i |x_i|$ が $p \to \infty$ の極限に対応する。あらゆる正当なノルムは、（i）絶対斉次性 $\|\alpha\mathbf{x}\| = |\alpha|\|\mathbf{x}\|$、（ii）三角不等式（triangle inequality）$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$、（iii）正定値性（$\|\mathbf{x}\| > 0$ が $\mathbf{x} \neq \mathbf{0}$ に対して成り立つ）を満たす。

<!--en:2913f1cb-->
$m \times n$ 行列 $\mathbf{A}$ に対して、*フロベニウスノルム*は $\ell_2$ ノルムをその $mn$ 個の成分全体について計算したものである。

<!--en:3f7ac19a-->
PyTorch では `torch.linalg.vector_norm` と `torch.linalg.matrix_norm` が分かれており、後者は 3 次元テンソルを 2 次元行列のバッチとして扱う。

<!--en:6149495a-->
## 固有ベクトルと固有値

<!--en:d5e0b636-->
50 層のネットワークが入力ベクトル $\mathbf{x}$ に行列を繰り返し掛けると何が起きるだろうか。非ゼロベクトル $\mathbf{v} \neq \mathbf{0}$ が $\mathbf{A} \in \mathbb{R}^{n \times n}$ のスカラー*固有値* $\lambda$ に対応する*固有ベクトル*（eigenvector）であるとは、$\mathbf{A}$ を掛けても $\mathbf{v}$ が回転せず単に定数倍されることをいう。

<!--en:e072f8c3-->
先ほどの対称行列 $\mathbf{S} = \left[\begin{smallmatrix} 2 & 1 \\ 1 & 2 \end{smallmatrix}\right]$ では、$(1, 1)$ は $\lambda = 3$ に対応する固有ベクトルであり、$(-1, 1)$ は $\lambda = 1$ に対応する固有ベクトルである（:numref:`fig_la_eigen`）。

<!--en:248b1828-->
![左：$\mathbf{S}$ は単位円（破線）を、固有ベクトルを主軸とし $\lambda = 3$ 倍および $\lambda = 1$ 倍に引き伸ばされた楕円へと写し、$\mathbf{e}_1$ は $\mathbf{S}\mathbf{e}_1 = (2, 1)$ へと向きを変える。右：$\mathbf{S}^k\mathbf{e}_1$（$k = 0, \ldots, 4$）の向きは、最大固有値の固有ベクトル $(1, 1)$ に近づいていく。](../img/mdl-prelim-eigen.svg)

<!--en:d62cfc4f-->
実対称行列の固有値と単位固有ベクトルは `linalg.eigh` を使って計算する。

<!--en:ef21e9f5-->
*スペクトル定理*（spectral theorem、:numref:`sec_mdl-eigendecompositions`）により、すべての $n \times n$ 実対称行列 $\mathbf{A}$ は $n$ 個の実固有値と、固有ベクトルからなる完全な正規直交基底（orthonormal basis）$\mathbf{v}_1, \ldots, \mathbf{v}_n$ を持つ。任意のベクトルを $\mathbf{x} = \sum_{i=1}^n c_i \mathbf{v}_i$（ただし $c_i = \mathbf{v}_i^\top \mathbf{x}$）と展開すると、次式が得られる。

<!--en:9921cce4-->
$k$ が大きくなるにつれて $|\lambda_i|$ が最大の項が支配的になり、$\mathbf{A}^k\mathbf{x}$ は主固有ベクトルの向きに揃いながら 1 ステップあたり $\max_i |\lambda_i|$ 倍で成長する（*べき乗法*、power iteration）。対称行列では、この成長率は任意のベクトルをその行列が引き伸ばせる最大の倍率、すなわち*スペクトルノルム* $\|\mathbf{A}\|_2 = \max_{\mathbf{x}\neq\mathbf{0}} \|\mathbf{A}\mathbf{x}\|/\|\mathbf{x}\|$ にも一致する。

<!--en:3abbe95e-->
50 層の思考実験に戻ろう。$\mathbf{M} = \left[\begin{smallmatrix} 1 & 0.1 \\ 0.1 & 1 \end{smallmatrix}\right]$ の固有値 $\lambda_1 = 1.1$ と $\lambda_2 = 0.9$ は $1$ からわずか $10\%$ しか離れていないが、その 2 つの単位固有ベクトルに $\mathbf{M}$ を $50$ 回掛けると、1 つ目は $1.1^{50} \approx 117.4$ 倍に増幅され、2 つ目は $0.9^{50} \approx 0.0052$ 倍に縮小する。

<!--en:dd011050-->
ネットワークの重み行列は対称行列ではなく、その大半は正方行列ですらない。そのような行列がベクトルをどれだけ引き伸ばすかも、固有値からわかるのだろうか。答えは否である。$\mathbf{A} = \left[\begin{smallmatrix} 0 & 2 \\ 0 & 0 \end{smallmatrix}\right]$ の固有値はどちらも $0$ であり $\mathbf{A}^2 = \mathbf{0}$ となるが、それでも $\mathbf{A}$ は $\mathbf{e}_2$ を $2\mathbf{e}_1$ に写すため長さを 2 倍にする。$\|\mathbf{A}\mathbf{x}\|_2^2 = \mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}$ であるから、ベクトルの伸縮を支配するのは対称行列 $\mathbf{A}^\top\mathbf{A}$ である。その固有値の平方根 $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$ が $\mathbf{A}$ の*特異値*（singular value）であり、正方行列か否かにかかわらずすべての行列に対して存在する。$\mathbf{S}$ と $\mathbf{A}$ について計算してみよう。

<!--en:01e9fe22-->
対称行列 $\mathbf{S}$ の特異値は（丸め誤差を除けば）$3$ と $1$ であり、固有値の絶対値に等しい。すなわち、対称行列ではこれら 2 つの概念は一致する。$\mathbf{A}$ の特異値は $2$ と $0$ であり、$\mathbf{A}^\top\mathbf{A}$ の固有値からも同じ 2 つの数値が昇順で得られる。任意の行列において、最大の特異値はそのスペクトルノルムに等しい。また、非ゼロの特異値の個数は行列の*ランク*、すなわち積によって潰れずに残る独立な方向の数であり、$\mathbf{A}$ では 1 である。これら 2 つの事実はいずれも特異値分解（singular value decomposition、:numref:`sec_mdl-svd-low-rank`）から導かれる。

<!--en:4f7d783d-->
各層の重み行列がベクトルの長さを平均して $1$ の近くに保たない限り、活性化値や勾配は深さとともに指数関数的に*爆発*（explode）または*消失*（vanish）する（:numref:`sec_numerical_stability`）。

<!--en:30ac03ff-->
## まとめ

<!--en:2339a129-->
線形代数はミニバッチ全体を扱うモデルを簡潔に表現する。平坦化した $256$ 枚の画像を $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ として積み重ねれば、全 $2{,}560$ 個のカテゴリスコアを $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ によって一括計算できる。リダクションは指定した軸に沿って値を集約する（`keepdims=True` はブロードキャストのために軸を長さ $1$ として保持する）。ドット積 $\mathbf{x}^\top\mathbf{y} = \sum_i x_i y_i$ は行列・ベクトル積 $\mathbf{A}\mathbf{x}$ と行列・行列積 $\mathbf{A}\mathbf{B}$（計算量は $\approx 2mkn$ FLOPs）を構成する。ノルム（$\ell_2$、$\ell_1$、$\ell_p$、フロベニウス、スペクトル）はベクトルや行列の大きさを測り、正規化したドット積はコサイン類似度 $\cos\theta = \mathbf{x}^\top\mathbf{y}/(\|\mathbf{x}\|\|\mathbf{y}\|)$ を与え、固有ベクトル（$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$）は行列積の反復によって各成分が $\lambda^k$ 倍にスケーリングされる理由を説明する。対称行列でない行列に対しては特異値がベクトルの伸縮を測り、その最大値がスペクトルノルムとなる。

<!--en:2294243a-->
## 演習問題

<!--en:e403b4d6-->
1. **転置と行列積。**次の代数恒等式を証明せよ。
    1. 行列積の転置が因子の順序を逆転させること、すなわち $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$ が（$\mathbf{A}\mathbf{B}$ が定義される限り常に）成り立つことを示せ。（両辺の $(i, j)$ 成分を :eqref:`eq_matmul` を用いて書き出せ。）
    1. *任意の*長方行列 $\mathbf{X} \in \mathbb{R}^{m \times n}$ に対して、積 $\mathbf{X}\mathbf{X}^\top$ は常に正方行列かつ対称行列になることを示せ（$\mathbf{X}^\top\mathbf{X}$ についても同様）。本節の $256 \times 784$ の画像ミニバッチ $\mathbf{X}$ に対して、$\mathbf{X}\mathbf{X}^\top$ の形状はどうなるか。また、その $(i, j)$ 成分は何を測っているか。

<!--en:6a7bcb2e-->
2. [code] **リダクションとブロードキャストの形状予測。**コードを一切実行せずに、`images`（形状 $(256, 1, 28, 28)$）と `X`（形状 $(256, 784)$）に対する以下の各式の結果の形状（またはエラー）を予測し、その後コードで答えを確認せよ。
    1. `len(images)` は何を返すか。`images.sum(axis=(2, 3))` の形状は `keepdims=False` の場合と `keepdims=True` の場合でそれぞれどうなるか。ブロードキャストによって `images` をそのまま割れるのはどちらか。また、`images / images.sum(axis=(2, 3), keepdims=True)` は何を計算しているか。
    1. `X - X.mean(axis=1)` を $256 \times 784$ 行列 `X` に対して評価すると何が起きるか。また、`keepdims=True` によってそれがどのように修正されるか。
    1. ここで、ミニバッチがたまたま $n = 784$ 枚の画像を含んでおり、`X` が $784 \times 784$ の正方行列であるとする。`X - X.mean(axis=1)` が*エラーを一切出さずに*実行されることを示せ。`X - X.mean(axis=1)` の $(i, j)$ 成分が実際に何を計算しているかを `X` の成分を用いて書き出し、正方行列における暗黙のブロードキャストバグが形状エラーよりもはるかに危険である理由を説明せよ。
   :exlabel:`ex_linear-algebra-shape-prediction`

<!--en:16b2fba5-->
2. [code] **リダクションとブロードキャストの形状予測。**コードを一切実行せずに、`images`（形状 $(256, 1, 28, 28)$、`float32` にキャスト済み）と `X`（形状 $(256, 784)$）に対する以下の各式の結果の形状（またはエラー）を予測し、その後コードで答えを確認せよ。
    1. `len(images)` は何を返すか。`tf.reduce_sum(images, axis=(2, 3))` の形状は `keepdims=False` の場合と `keepdims=True` の場合でそれぞれどうなるか。ブロードキャストによって `images` をそのまま割れるのはどちらか。また、`images / tf.reduce_sum(images, axis=(2, 3), keepdims=True)` は何を計算しているか。
    1. `X - tf.reduce_mean(X, axis=1)` を $256 \times 784$ 行列 `X` に対して評価すると何が起きるか。また、`keepdims=True` によってそれがどのように修正されるか。
    1. ここで、ミニバッチがたまたま $n = 784$ 枚の画像を含んでおり、`X` が $784 \times 784$ の正方行列であるとする。`X - tf.reduce_mean(X, axis=1)` が*エラーを一切出さずに*実行されることを示せ。`X - tf.reduce_mean(X, axis=1)` の $(i, j)$ 成分が実際に何を計算しているかを `X` の成分を用いて書き出し、正方行列における暗黙のブロードキャストバグが形状エラーよりもはるかに危険である理由を説明せよ。
   :exlabel:`ex_linear-algebra-shape-prediction`

<!--en:ea2255e8-->
3. **ベクトルの集合は互いにどこまで鈍角になれるか。**以下の設問では、$k$ 個のベクトル間のすべての対のドット積が同時にどこまで負になれるかを探る。
    1. 平面上の 3 つのベクトル $\mathbf{u}, \mathbf{v}, \mathbf{w} \in \mathbb{R}^2$ は、$\mathbf{u}^\top\mathbf{v} < 0$、$\mathbf{v}^\top\mathbf{w} < 0$、$\mathbf{u}^\top\mathbf{w} < 0$ を同時に満たせるか。具体的な数値例を挙げるか、存在しないことを証明せよ。
    1. $k \geq 2$ 個の単位ベクトル $\mathbf{u}_1, \ldots, \mathbf{u}_k \in \mathbb{R}^d$（$d \geq k - 1$）が対称に配置され、異なるすべての対がまったく同じドット積 $\mathbf{u}_i^\top\mathbf{u}_j = c$（すべての $i \neq j$）を持つとする。$0 \leq \|\mathbf{u}_1 + \cdots + \mathbf{u}_k\|_2^2$ を展開することで、$c \geq -\frac{1}{k - 1}$ を証明せよ。10 クラス分類器における単位長のクラステンプレート $10$ 個がすべて等しい角度で離れているとき、この下界は $10$ 個のテンプレート間の対ごとのコサイン類似度について何を意味するか。

    *MIT 18.06、[問題セット 1](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/22297c2a6dcf06d82e93ee4af115e91a_MIT18_06S10_pset1_s10_soln.pdf)、1.2 節、問題 28 より改題。*

<!--en:24c49717-->
1. [code] **複数のパラメータテンソルにわたるノルム。**本節で導入したベクトルノルムのルーチンは、高階テンソルにも適用できる。
    1. 形状 $(2, 3, 4)$ の 3 次元テンソルに対してデフォルト引数でベクトルノルムを評価し、どのような数式を計算しているかを言葉で述べ、各要素を手動で二乗・総和・平方根をとることで答えを確認せよ。
    1. 深層ニューラルネットワークは、学習可能なパラメータをさまざまな階数と形状を持つ重みテンソルおよびバイアステンソルのリストに分けて保持する。訓練の監視や勾配クリッピング（:numref:`sec_rnn-scratch`）の際には、全パラメータを 1 つの巨大なベクトルに連結したときの単一の全体 $\ell_2$ ノルムがしばしば必要になる。テンソルを平坦化したり連結したりせずに、この全体 $\ell_2$ ノルムを個々のテンソルノルムを用いて表し、形状 $(3, 4)$、$(4,)$、$(2, 3, 5)$ の 3 つのテンソルに対してコードで式を検証せよ。

<!--en:d06354fd-->
1. **行列積の結合法則と計算コスト。**行列積は結合法則を満たし、数学的には $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$ が成り立つが、連鎖する行列積に括弧を付ける順序によって計算コストは何桁も変わり得る。$\mathbf{A} \in \mathbb{R}^{1000 \times 50000}$、$\mathbf{B} \in \mathbb{R}^{50000 \times 20}$、$\mathbf{C} \in \mathbb{R}^{20 \times 10000}$ が与えられたとき、$(\mathbf{A}\mathbf{B})\mathbf{C}$ と $\mathbf{A}(\mathbf{B}\mathbf{C})$ の両方について、総 FLOP 数と中間の積行列の要素数を求めよ。2 つの評価順序で、演算量と中間メモリ量はそれぞれ何倍異なるか。

<!--en:561876fb-->
1. [code] **ループなしの全対ユークリッド距離行列。**● $50$ 個ずつの $\mathbb{R}^8$ のベクトルからなる 2 つの集合が $\mathbf{X} \in \mathbb{R}^{50 \times 8}$ と $\mathbf{Y} \in \mathbb{R}^{50 \times 8}$ の行として積み重ねられているとき、Python のループを使わずに二乗ユークリッド距離の $50 \times 50$ 行列 $\mathbf{D}$（成分 $D_{ij} = \|\mathbf{x}_i - \mathbf{y}_j\|_2^2$）を計算したい。
    1. まず 3 次元ブロードキャストによって $\mathbf{D}$ を計算せよ。すなわち、$\mathbf{X}$ を $(50, 1, 8)$ に、$\mathbf{Y}$ を $(1, 50, 8)$ に形状変更し、差をとって二乗し、最後の軸に沿って和をとれ。次に、$\mathbf{D}$ を二次式の展開 $\|\mathbf{x}_i - \mathbf{y}_j\|_2^2 = \|\mathbf{x}_i\|_2^2 + \|\mathbf{y}_j\|_2^2 - 2\mathbf{x}_i^\top\mathbf{y}_j$ によって計算せよ。これには行ノルムのリダクション、2 次元ブロードキャスト、および 1 回の行列積 $\mathbf{X}\mathbf{Y}^\top$ しか必要ない。両者の結果が浮動小数点精度の範囲で一致することを確認せよ。
    1. $\mathbf{X}$ と $\mathbf{Y}$ がそれぞれ $10{,}000$ 個の $\mathbb{R}^{128}$ のベクトルを含むとき、3 次元ブロードキャスト法が確保しなければならない中間の浮動小数点数の個数は、行列積を用いる方法と比べていくつになるか。大規模データではどちらの方法を使うべきか。
    1. $\|\mathbf{x}_i\|_2^2 + \|\mathbf{y}_j\|_2^2 - 2\mathbf{x}_i^\top\mathbf{y}_j$ における浮動小数点の丸め誤差によって、ごく小さな負の値（$-10^{-7}$ など）が $\mathbf{x}_i = \mathbf{y}_j$ のときに生じ得る理由を説明し、平方根をとる前に出力をクランプする方法を示せ。

    *Stanford CS231n、[課題 1](https://cs231n.github.io/assignments2026/assignment1/)（$k$ 近傍距離）、および fast.ai、[Deep Learning from the Foundations](https://github.com/fastai/course-v3/blob/master/files/dl-2019/notes/notes-2-8.md)、Lesson 8 より改題。*

<!--en:ffed3a72-->
1. [code] **べき乗法と収束率。**● $\mathbf{v}_{k+1} = \mathbf{A}\mathbf{v}_k / \|\mathbf{A}\mathbf{v}_k\|_2$ をランダムな単位ベクトル $\mathbf{v}_0$ から繰り返し更新する正規化べき乗法を、次の $3 \times 3$ 対称行列に対して実装せよ。

   $$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 0 & 4 \\ 3 & 4 & 5 \end{bmatrix}.$$

    1. 最初の反復回数 $k$（連続する反復ベクトル間のコサイン類似度の絶対値 $|\mathbf{v}_k^\top \mathbf{v}_{k-1}|$ が $0.999$ を初めて超える時点）を記録せよ。収束したベクトル $\mathbf{v}_k$ を `eigh(A)` が返す主固有ベクトルと比較し、2 つのベクトルが全体の符号 $\pm 1$ だけ異なり得る理由を説明せよ。
    1. 各ステップにおいて、$\mathbf{v}_k$ と真の主固有ベクトル $\mathbf{u}_{\max}$ がなす角の正弦 $\sin\theta_k = \sqrt{1 - (\mathbf{v}_k^\top \mathbf{u}_{\max})^2}$ を測定し、1 ステップあたりの縮小比 $\sin\theta_{k+1} / \sin\theta_k$ を計算せよ。この漸近的な比を $\mathbf{A}$ の固有値を用いて :eqref:`eq_matrix_power` から表し、べき乗法がごく少数のステップで収束した理由を説明せよ。
    1. 次に、シフトした行列 $\mathbf{B} = \mathbf{A} - 9\mathbf{I}$ に対してべき乗法を実行せよ。べき乗法は $\mathbf{A}$ のどの固有値および固有ベクトルに収束するか。また、連続する反復ベクトル $\mathbf{v}_k$ と $\mathbf{v}_{k+1}$ の符号はどうなるか、その理由とともに答えよ。

    *MIT 18.06、[問題セット 1](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/22297c2a6dcf06d82e93ee4af115e91a_MIT18_06S10_pset1_s10_soln.pdf)、2.1 節、問題 29〜30 より改題。*

<!--en:1743a7e9-->
[ディスカッション](https://d2l.discourse.group/t/30)

<!--en:d3a8635e-->
[ディスカッション](https://d2l.discourse.group/t/31)

<!--en:a8812ecb-->
[ディスカッション](https://d2l.discourse.group/t/196)

<!--en:cf6c98e6-->
[ディスカッション](https://d2l.discourse.group/t/17968)

<!--en:375e7c9a-->
線形代数：深層学習の計算エンジン<br>**テンソルと対称性・軸方向のリダクション・射影と行列積・ノルム・スペクトルダイナミクス**

<!--en:e3ca9289-->
::: {.slide title="なぜ線形代数が深層学習を支えるのか"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:17e9bb10-->
- **線形写像と非線形写像の交互適用：**すべての全結合層・畳み込み層・アテンション層は、**線形変換**（$\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$）と**非線形な活性化**を交互に繰り返す

<!--en:4d810f12-->
- **単一ベクトルからハードウェアバッチへ：**平坦化した $256$ 枚の Fashion-MNIST 画像を $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ に、$10$ 個のクラステンプレートを $\mathbf{W} \in \mathbb{R}^{784 \times 10}$ にまとめると、$256$ 件の評価を GPU の Tensor Cores 上で **1 回の行列積**として実行できる

<!--en:3eae8dd3-->
- **予測可能な形状とコスト：**出力形状 $(256, 10)$ と演算量（$2 \times 256 \times 784 \times 10 \approx 4.0 \times 10^6\text{ FLOPs}$）は実行前に確定する

<!--en:5ee600cb-->
::: {.slide title="スカラー・ベクトルと次元数対テンソルの階数"}

<!--en:b47479c8-->
[基本要素]{.kicker}

<!--en:3cca4604-->
- **スカラー（$x \in \mathbb{R}$）：**空の形状 `()` を持つ $0^\textrm{th}$ 階のテンソル（例：損失 $\mathcal{L}$、学習率 $\eta$）

<!--en:a2ecb13e-->
- **ベクトル（$\mathbf{x} \in \mathbb{R}^d$）：**形状 `(d,)` を持つ $1^\textrm{st}$ 階のテンソルであり、数学的には列ベクトル $\mathbf{x} = [x_1, \ldots, x_d]^\top$ として扱う

<!--en:e6868c4a-->
- **用語の落とし穴：**ベクトルの**次元数** $d$ は軸 0 に沿った長さ（`len(x)`）であるのに対し、テンソルの**階数（ランク）**は軸の数（`ndim`）を指す。$784$ 次元のベクトルは $1^\textrm{st}$ 階のテンソル

<!--en:b2da0c9f-->
::: {.slide title="行列、転置、対称性"}

<!--en:a1481f02-->
[2 次元表現]{.kicker}

<!--en:1815c456-->
- **計画行列（$\mathbf{X} \in \mathbb{R}^{n \times d}$）：**各行 $i$ に 1 つのデータ例、各列 $j$ に 1 つの特徴量を格納

<!--en:889ecfb2-->
- **転置（$[\mathbf{A}^\top]_{ij} = a_{ji}$）：**主対角線を軸に成分を反転させ、形状を $(m, n) \to (n, m)$ に変換

<!--en:daf1a779-->
- **対称行列（$\mathbf{S} = \mathbf{S}^\top$）：**共分散 $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$ やヘッセ行列 $\nabla^2 \mathcal{L}$

<!--en:36dfe6ca-->
::: {.slide title="高階テンソルと要素ごとの（アダマール）演算"}

<!--en:8b9aed76-->
[テンソルとアダマール積]{.kicker}

<!--en:52cb22fd-->
- **高階テンソル：**3 次元のトークン系列 `(B, T, d)` や 4 次元の画像 `(256, 1, 28, 28)` は `reshape(256, -1)` で 2 次元行列に展開

<!--en:d579cab2-->
- **アダマール積（$\mathbf{A} \odot \mathbf{B}$、`A * B`）：**対応する成分同士を掛ける演算 $[\mathbf{A} \odot \mathbf{B}]_{ij} = a_{ij} b_{ij}$ であり、ゲーティング（LSTM、SwiGLU）に用いられ、行列積 `A @ B` とは異なる

<!--en:57b5e3d0-->
::: {.slide title="軸方向のリダクション：次元の集約"}

<!--en:412006fb-->
[リダクション]{.kicker}

<!--en:5350aca3-->
- リダクション（`.sum()`、`.mean()`）は選択した軸に沿って要素を集約し、**その軸を形状から取り除く**。
  - **全軸（`A.sum()`）：**`(2, 3) -> ()` へと集約され、スカラーになる
  - **行方向（`axis=0`）：**$c_j = \sum_{i=1}^m a_{ij}$ により形状 **`(3,)`** の列和が残る
  - **列方向（`axis=1`）：**$r_i = \sum_{j=1}^n a_{ij}$ により形状 **`(2,)`** の行和が残る

<!--en:4d4b8410-->
::: {.slide title="軸を保持する和（keepdims=True）とブロードキャスト正規化"}

<!--en:de5a2274-->
[リダクションとブロードキャスト]{.kicker}

<!--en:a3d926ef-->
- **なぜ `A / A.sum(axis=1)` が失敗するか：**`(2, 3)` を `(2,)` で割ると右詰めされて `(2, 3)` 対 `(1, 2)` となり、末尾の次元 $3 \neq 2$ が衝突する

<!--en:89e78bf5-->
- **`keepdims=True` で長さ 1 を保持（`(2, 1)`）：**`A / sum_A` が列方向にブロードキャストされ、各行の和が $1$ になる（**ソフトマックス**、**アテンション**）

<!--en:92140e49-->
- **$\mathbf{X} \in \mathbb{R}^{256 \times 784}$ の中心化：**ピクセルごとの平均（`axis=0`、`(784,)`）はそのままブロードキャストされるが、画像ごとの平均（`axis=1`、`(256, 1)`）には `keepdims=True` が必要（**LayerNorm**）

<!--en:4a076fd7-->
::: {.slide title="ドット積：重み付き和と幾何学的射影"}

<!--en:fc1c769f-->
[内積]{.kicker}

<!--en:7f04b0d4-->
- $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ に対し、それらの**ドット積**（内積）は対応する座標の積の総和をとる。
  $$\mathbf{x}^\top \mathbf{y} = \langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d} x_i y_i = \|\mathbf{x}\|_2 \|\mathbf{y}\|_2 \cos\theta$$

<!--en:57eda7b1-->
- **深層学習における役割：**単一ニューロンの前活性化値 $\mathbf{w}^\top \mathbf{x} + b$ はオフセット付きのドット積であり、$w_i \geq 0$ かつ $\sum_i w_i = 1$ のとき $\mathbf{w}^\top \mathbf{x}$ は凸結合（重み付き平均）を計算する

<!--en:a866b490-->
::: {.slide title="コサイン類似度とコーシー・シュワルツの不等式"}

<!--en:3dda3eba-->
[高次元における角度]{.kicker}

<!--en:5e1a83b4-->
- ユークリッド長で割ることで、任意の次元 $d$ における**コサイン類似度**を定義する。
  $$\cos\theta = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2} \in [-1, 1]$$

<!--en:3a97012c-->
- **スケール不変性：**明るさや埋め込みの大きさ（`2 * x`）を定数倍しても $\cos\theta$ は不変であり、トークン・画像・検索の埋め込みにおける定番の類似度スコアとなる

<!--en:2e7af27c-->
- **コーシー・シュワルツの不等式（$|\mathbf{x}^\top\mathbf{y}| \leq \|\mathbf{x}\|_2\|\mathbf{y}\|_2$）：**$0 \leq \|\mathbf{u} \mp \mathbf{v}\|_2^2 = 2 \mp 2\mathbf{u}^\top\mathbf{v}$ を単位ベクトル $\mathbf{u}, \mathbf{v}$ に適用して直接導かれる

<!--en:5361ae52-->
::: {.slide title="行列・ベクトル積：特徴空間上の線形写像"}

<!--en:43ed7a0c-->
[行列・ベクトル積]{.kicker}

<!--en:d0b78abd-->
- 行列 $\mathbf{A} \in \mathbb{R}^{m \times n}$ は線形変換 $\mathbb{R}^n \to \mathbb{R}^m$ を定め、相補的な 2 つの見方がある。
  1. **行ごとの見方（$m$ 個の並列ドット積）：**$[\mathbf{A}\mathbf{x}]_i = \mathbf{a}_i^\top \mathbf{x}$ は入力 $\mathbf{x}$ を $m$ 個の行テンプレート（$\mathbf{A}$ の各行）へ射影する
  2. **列ごとの見方（基底の結合）：**$\mathbf{A}\mathbf{x} = \sum_{j=1}^n x_j \mathbf{a}_{:,j}$ は $\mathbf{A}$ の列ベクトルを $\mathbf{x}$ の座標で重み付けした線形結合を作る

<!--en:c18351a7-->
::: {.slide title="幾何学的視点：行列は空間をどう変換するか"}

<!--en:34234143-->
[線形幾何]{.kicker}

<!--en:f40b1d19-->
- $\mathbf{A}\mathbf{e}_j$ は $j^\textrm{th}$ 列を $\mathbf{A}$ から取り出すため、各列は標準基底ベクトル $\mathbf{e}_1, \mathbf{e}_2$ の行き先を表す。
  - **回転・軸方向のスケーリング・せん断**は、破線の単位正方形を $\mathbf{A}\mathbf{e}_1$ と $\mathbf{A}\mathbf{e}_2$ が張る平行四辺形へと変形する
  - ニューラルネットワークの各層にわたって、重み行列 $\mathbf{W}$ は特徴空間を回転・伸縮・射影し、絡み合ったクラスを線形分離可能にする

<!--en:5a5f4987-->
::: {.slide title="行列・行列積：線形層のバッチ化"}

<!--en:ec5e4916-->
[行列・行列積]{.kicker}

<!--en:44e5aac2-->
- $\mathbf{A} \in \mathbb{R}^{m \times k}$ と $\mathbf{B} \in \mathbb{R}^{k \times n}$ に対し、$\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$（`A @ B`）は $mn$ 個の長さ $k$ のドット積を計算する。
  $$c_{ij} = \mathbf{a}_i^\top \mathbf{b}_{:,j} = \sum_{l=1}^{k} a_{il}\, b_{lj}$$

<!--en:14b7fe90-->
- **形状の規則：**$(m, k) \times (k, n) \to (m, n)$ であり、内側の次元 $k$ は一致して縮約され消える

<!--en:a67cd923-->
- **10 クラス分類器のバッチ化（$\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$）：**`(256, 784) @ (784, 10) + (10,) -> (256, 10)` により全 $2{,}560$ 個のロジットを一括評価し、軸 1 に沿った `argmax` で各画像のクラスを予測する

<!--en:0bdf452c-->
::: {.slide title="行列積の計算量と演算密度"}

<!--en:e7fe691c-->
[システムと FLOPs]{.kicker}

<!--en:311c2dee-->
- **FLOP 数：**$(m, k) \times (k, n)$ の積は $mn$ 個の長さ $k$ のドット積を計算し、コストは **$\approx 2mkn\text{ FLOPs}$**

<!--en:8d62184e-->
- **$O(n^2)$ バイト対 $O(n^3)$ FLOPs：**要素ごとの演算（$0.33\text{ FLOP/el}$）や単一ベクトルの行列・ベクトル積（$2\text{ FLOP/el}$）は HBM 帯域幅がボトルネックになる

<!--en:652c5047-->
- **バッチ化する理由：**複数のベクトルを行列積 $\mathbf{X}\mathbf{W}$ にまとめると演算密度が $O(n)$ に高まり（$2{,}730\text{ FLOP/el}$、$n=4096$ の場合）、GPU の Tensor Cores を使い切れる

<!--en:8631da31-->
::: {.slide title="理解度チェック：リダクション・ブロードキャスト・行列積の形状予測"}

<!--en:929b7671-->
[理解度の確認]{.kicker}

<!--en:2d84dbac-->
平坦化したミニバッチ `X`（形状 `(256, 784)`）と、行ごとに格納された重み行列 `W_T`（形状 `(10, 784)`）が与えられたとする。

<!--en:5fc52908-->
1. なぜ `X @ W_T` は失敗するのか。また、`(256, 10)` のロジット行列を計算する式は何か。

<!--en:0788f94f-->
2. `X.sum(axis=0)` と `X.sum(axis=1, keepdims=True)` の形状はそれぞれ何か。

<!--en:304dc4e8-->
3. ミニバッチがたまたま $n = 784$ 枚の画像を含み、`X` が正方行列 `(784, 784)` になったとする。`X - X.mean(axis=1)` を `keepdims=True` *なし*で評価すると何が起きるか。

<!--en:557b98b6-->
**解答：**

<!--en:1039b8e9-->
1. 内側の次元 $784 \neq 10$ が不一致であるため失敗する。**`X @ W_T.T`** で転置すれば `(256, 784) @ (784, 10) -> (256, 10)` と揃う

<!--en:e627917e-->
2. `X.sum(axis=0)` の形状は **`(784,)`**（ピクセルごとの合計）、`X.sum(axis=1, keepdims=True)` の形状は **`(256, 1)`**（画像ごとの合計）

<!--en:4c5a1955-->
3. **暗黙のブロードキャストバグ：**`X.mean(axis=1)` の形状は `(784,)` であり、右詰めされて `(1, 784)` となり、エラーを出さずに*画像 $j$ の平均をすべての画像のピクセル $j$ から*引いてしまう

<!--en:7d06e264-->
::: {.slide title="ベクトルと行列のノルム：大きさの尺度"}

<!--en:b39d105e-->
[ノルム]{.kicker}

<!--en:a19b37a7-->
- すべてのノルム $\|\cdot\|$ は $\|\alpha\mathbf{x}\| = |\alpha|\|\mathbf{x}\|$、$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$、および $\|\mathbf{x}\| > 0$（$\mathbf{x} \neq \mathbf{0}$ のとき）を満たす。
  - **ユークリッドノルム（$\ell_2$）：**$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$（$(3, -4)$ では $\|\mathbf{x}\|_2 = 5$）
  - **マンハッタン距離（$\ell_1$）：**$\|\mathbf{x}\|_1 = \sum_i |x_i|$（$(3, -4)$ では $\|\mathbf{x}\|_1 = 7$）
  - **$\ell_p$ ノルムとフロベニウスノルム：**$\|\mathbf{x}\|_p = (\sum_i |x_i|^p)^{1/p}$、$\|\mathbf{A}\|_\textrm{F} = \sqrt{\sum_{i,j} a_{ij}^2}$

<!--en:e2f67264-->
::: {.slide title="なぜノルムの選択が重要か：外れ値、重み減衰、スパース性"}

<!--en:e6689021-->
[機械学習におけるノルム]{.kicker}

<!--en:25657f5a-->
- **外れ値（$\ell_2^2$ 対 $\ell_1$ 損失）：**$\mathbf{r} = (1, -1, 1, -1, 10)$ では、単一の外れ値 $10$ が $\|\mathbf{r}\|_2^2$（傾き $2r$）の **$96\%$** を占めるのに対し、$\|\mathbf{r}\|_1$（有界な傾き $\pm 1$）では **$71\%$** にとどまる

<!--en:f2189057-->
- **重み減衰（$\frac{\lambda}{2}\|\mathbf{W}\|_\textrm{F}^2$）：**丸みを帯びた $\ell_2$ 球が重みを 0 に向けて滑らかに縮小する

<!--en:130d50e6-->
- **スパース性（$\lambda\|\mathbf{w}\|_1$）：**$\ell_1$ ひし形の軸上の尖った角が不要な重みを**厳密な 0** へと導く

<!--en:a04776c7-->
::: {.slide title="固有値分解：不変な方向とスペクトル定理"}

<!--en:54cce930-->
[固有値分解]{.kicker}

<!--en:1465fe79-->
- 非ゼロの $\mathbf{v}$ が $\mathbf{A} \in \mathbb{R}^{n \times n}$ の**固有値** $\lambda$ に対応する**固有ベクトル**であるとは、$\mathbf{A}\mathbf{v} = \lambda \mathbf{v} \Longrightarrow \mathbf{A}^k \mathbf{v} = \lambda^k \mathbf{v}$ を満たすこと

<!--en:d2bdf815-->
- **スペクトル定理：**すべての実対称行列 $\mathbf{S} = \mathbf{S}^\top$ は $n$ 個の**実固有値**と**正規直交固有基底**を持つ（`linalg.eigh`）

<!--en:8efc277b-->
::: {.slide title="べき乗法とスペクトルノルム"}

<!--en:bfa75699-->
[行列積の反復]{.kicker}

<!--en:5ef7357f-->
- 任意の入力 $\mathbf{x} = \sum_{i=1}^n c_i \mathbf{v}_i$ を正規直交固有基底で展開すると、行列積の反復による変化がわかる。
  $$\mathbf{A}^k \mathbf{x} = \sum_{i=1}^n c_i \lambda_i^k \mathbf{v}_i$$

<!--en:4bc6f80d-->
- **べき乗法：**$k$ が大きくなると $|\lambda_i|$ が最大の項が支配的になり、$\mathbf{A}^k\mathbf{x}$ は主固有ベクトルの向きに揃い、そのノルムは 1 ステップあたり $\max_i |\lambda_i|$ 倍に伸びる

<!--en:345f04ce-->
- 対称行列 $\mathbf{A}$ では、$\max_i |\lambda_i|$ は**スペクトルノルム** $\|\mathbf{A}\|_2 = \max_{\mathbf{x}\neq\mathbf{0}} \|\mathbf{A}\mathbf{x}\|_2 / \|\mathbf{x}\|_2$（最大伸縮率）に等しい

<!--en:ca91fb4d-->
::: {.slide title="深層ネットワークのスペクトルダイナミクス：信号の爆発と消失"}

<!--en:e4de5c27-->
[なぜ初期化と正規化が重要か]{.kicker}

<!--en:e5f6e025-->
- $L = 50$ 層にわたると、わずか $10\%$ の固有値のずれ（$\lambda_1 = 1.1$、$\lambda_2 = 0.9$）でも $1.1^{50} \approx \mathbf{117.4\times}$ へと爆発し、あるいは $0.9^{50} \approx \mathbf{0.0052\times}$ へと消失する

<!--en:4e5674a7-->
- 各層の伸縮率（特異値）を $1$ の近くに保つことが、**Xavier/He 初期化**、**残差接続**、**LayerNorm** の動機となる

<!--en:dc9c827f-->
::: {.slide title="まとめ：線形代数の演算・形状・役割"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:78c9afef-->
| 演算 | 形式 | 形状の規則 | FLOPs | 深層学習における役割 |

<!--en:34b7c196-->
| **アダマール積** | $\mathbf{A} \odot \mathbf{B}$（`A * B`） | $(m, n) \to (m, n)$ | $mn$ | LSTM・SwiGLU のゲーティング、マスク |

<!--en:df28d43e-->
| **軸方向のリダクション** | `sum(axis=1, keepdims=True)` | $(m, n) \to (m, 1)$ | $mn$ | ソフトマックス正規化、LayerNorm |

<!--en:27eb2724-->
| **ドット積とコサイン** | $\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|_2\|\mathbf{y}\|_2\cos\theta$ | $(d,), (d,) \to ()$ | $2d$ | ニューロンのロジット、クエリ・キー類似度 |

<!--en:b49749e0-->
| **行列・ベクトル積** | $\mathbf{A}\mathbf{x}$（`A @ x`） | $(m, n) \times (n,) \to (m,)$ | $2mn$ | 単一データ例の線形変換 |

<!--en:20ce1c9b-->
| **行列・行列積** | $\mathbf{C} = \mathbf{X}\mathbf{W}$（`X @ W`） | $(n, k) \times (k, m) \to (n, m)$ | $2nkm$ | バッチ化された全結合層・アテンション層 |

<!--en:2b77318a-->
| **ノルム（$\ell_2, \ell_1, \textrm{F}$）** | $\|\mathbf{x}\|_2, \|\mathbf{x}\|_1, \|\mathbf{W}\|_\textrm{F}$ | 任意のテンソル $\to ()$ | $2N$ | MSE・MAE 損失、重み減衰、クリッピング |

<!--en:555bc743-->
| **固有値分解** | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v} \Rightarrow \mathbf{A}^k\mathbf{v} = \lambda^k\mathbf{v}$ | $(n, n) \to (\lambda_i, \mathbf{v}_i)$ | $O(n^3)$ | 信号の爆発・消失、曲率 |
