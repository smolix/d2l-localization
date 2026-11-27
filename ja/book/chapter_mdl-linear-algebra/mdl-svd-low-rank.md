<!--en:7def598f-->
# 特異値分解と低ランク近似

<!--en:215fd002-->
:numref:`sec_mdl-eigendecompositions` では、固有値分解（eigendecomposition）が正方行列を「単に伸縮するだけの方向」によって記述し、行列のべき乗を単なる数値のべき乗へと帰着させることを見た。しかし、深層学習に現れる大半の行列はそのような性質を持たない。重み行列はある幅の入力を別の幅の出力へ写し、データ行列はデータ例ごとに 1 行、特徴量ごとに 1 列を持つ。どちらも自身の出力へ繰り返し適用されるわけではなく、正方行列である必要もない。さらに行列が*正方行列であっても*、固有値だけでは不十分なことがある。せん断行列 $\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ は*欠陥*（defective）を持ち、平面を張るには 2 方向が必要なところ固有ベクトルの方向を 1 つしか持たない。また :numref:`sec_mdl-spectra-in-practice` では、2 つの固有値がともに 1 より小さいにもかかわらず、6 乗するとベクトルを 10 倍に引き伸ばす行列に出会った。ここで必要なのは、*あらゆる*行列に対して存在し、伸縮を直接測ることができ、大きな行列を精度よく近似するのにどれほど少ない数値で足りるかを教えてくれる分解である。

<!--en:20b036c5-->
最後の要件のありがたみは、具体的な写真を考えると実感しやすい。$400\times500$ ピクセルの猫のグレースケール画像は、単なる 200,000 個の数値からなる行列である。他のあらゆる行列と同様、この画像も 1 本の列と 1 本の行の外積（outer product）で作られるランク 1 の成分の和として書ける。驚くべきことに、うまく選んだわずか 20 個の成分（18,020 個の数値、すなわち元データの 9%）だけで、相対誤差わずか 8% で写真を再現できる（:numref:`fig_mdl-la-eckart-young`）。どの 20 個の成分がこれを達成するのか、また別の 20 個を選べばさらに良くできるのだろうか。深層学習の至るところでこれと同じ問いが現れる。大規模モデルのファインチューニング用アダプタにどれだけのパラメータが必要か、そしてデータセットのどの方向が本当に残す価値があるのかを、この問いが教えてくれるのである。

<!--en:ed208837-->
鍵となる考え方は、入力と出力の両方を単一の基底で済ませようとするのをやめることである。入力空間に 1 つの正規直交基底（orthonormal basis）を、出力空間に別の正規直交基底を用意すれば、*あらゆる*行列は独立な非負の伸縮率の列 $\sigma_1\ge\sigma_2\ge\cdots\ge0$ へと帰着する。このたった 1 つのリストから、行列のランク、最良の低ランク近似（low-rank approximation）、主成分分析（principal component analysis、PCA）、最小二乗法（least squares）の解、さらには連立一次方程式（linear system）が測定誤差に対してどれほど敏感かまで読み取れる。:numref:`sec_linear-algebra` では、行列の大きさを測る 2 つの方法として、フロベニウスノルム $\|\mathbf{A}\|_F$（全成分の二乗和の平方根）とスペクトルノルム $\|\mathbf{A}\|_2$（$\mathbf{A}$ が単位ベクトルを引き伸ばせる最大の倍率）を導入した。本節では、そのどちらも特異値（singular value）$\sigma_i$ の単純な関数として表される。

<!--en:e200037c-->
以下の議論の柱となるのは 2 つの主要な結果である。1 つは任意の行列に対する特異値分解（singular value decomposition、SVD）の存在であり、もう 1 つは分解を途中で打ち切ることであらゆるランクにおける最良近似が得られると保証する Eckart–Young の定理である。そこから、擬似逆行列（pseudoinverse）と条件数（condition number）がわずか数行で導かれる。最後の小節ではこれらの道具を現代の深層学習と結びつけるが、初読時には飛ばしてもよい。以下の数値実験はすべて NumPy を用い、特に断らない限り、本節を通じて $\mathbf{T}$ はせん断行列、$\mathbf{A}$ は一般の行列を表す。

<!--en:606d4afb-->
## 特異値分解

<!--en:77b308ce-->
### 分解とその幾何学

<!--en:cbeef8ae-->
:numref:`sec_linear-algebra` では、対称行列 $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ が単位円を楕円へ写し、その半軸が固有ベクトル $(1,1)$ と $(-1,1)$ の方向に沿って長さ 3 と 1 を持つことを見た。驚くべきことに、まもなく証明するように、*非対称*行列もまた単位円を楕円へと写す（行列が特異行列であれば線分に潰す）。違いは、この楕円の軸の上に着地する直交した入力方向が、もはやその軸自身の方向とは一致しない点にある。代わりに、入力空間用の正規直交フレームと出力空間用の正規直交フレームという 2 つの正規直交フレームが存在し、特異値分解はその両方に名前を与える。

<!--en:58ba3010-->
**定理（特異値分解）。** *任意の実 $m\times n$ 行列 $\mathbf{A}$ は次のように書ける。*

<!--en:065d86f6-->
*ここで、$\mathbf{U}\in\mathbb{R}^{m\times m}$ と $\mathbf{V}\in\mathbb{R}^{n\times n}$ は直交行列（$\mathbf{U}^\top\mathbf{U}=\mathbf{I}$、$\mathbf{V}^\top\mathbf{V}=\mathbf{I}$）であり、$\boldsymbol{\Sigma}\in\mathbb{R}^{m\times n}$ は主対角線上の非負成分 $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_{\min(m,n)}\ge0$ を除いてすべて 0 である。*列ベクトル $\mathbf{v}_i$（$\mathbf{V}$ の各列）を*右特異ベクトル*（right singular vector）、列ベクトル $\mathbf{u}_i$（$\mathbf{U}$ の各列）を*左特異ベクトル*（left singular vector）、対角成分 $\sigma_i$ を*特異値*と呼ぶ。

<!--en:5901abc9-->
幾何学的に言えば、形状にかかわらず*あらゆる*行列は、直交変換、対角方向の伸縮、そして 2 つ目の直交変換を順に施したものにほかならない。$\mathbf{A}\mathbf{V}=\mathbf{U}\boldsymbol{\Sigma}$ と書き直して第 $i$ 列を読み取れば、1 方向ずつの作用が明示的にわかる。

<!--en:c94a78ec-->
各入力方向 $\mathbf{v}_i$ は対応する出力方向 $\mathbf{u}_i$ へと写され、$\sigma_i$ 倍に伸縮される。$n>m$ のときは、余分な入力方向 $\mathbf{v}_{m+1},\ldots,\mathbf{v}_n$ はゼロに潰される。:eqref:`eq_mdl-svd` を右から左へ読むと、$\mathbf{A}$ をベクトル $\mathbf{x}$ に作用させる操作は次の 3 段階で進む。

<!--en:63fa7c68-->
1. $\mathbf{V}^\top$ は入力を回転させ、各右特異ベクトル $\mathbf{v}_i$ を座標軸 $\mathbf{e}_i$ に揃える。

<!--en:5db33b8b-->
2. $\boldsymbol{\Sigma}$ は軸 $i$ を $\sigma_i$ 倍に伸縮し、$m\ne n$ のときは出力次元に合わせてゼロ座標を追加するか、余分な座標を切り捨てる。

<!--en:82aeabb5-->
3. $\mathbf{U}$ は伸縮された各軸を回転させ、軸 $i$ が $\mathbf{u}_i$ の方向を向くようにする。

<!--en:5d3fac5f-->
ここで「回転」とは直交変換の便宜的な略称であり、鏡映を含む場合もある。

<!--en:ab918194-->
:numref:`fig_mdl-la-svd-action` に、せん断 $\mathbf{T}$ におけるこれら 3 段階を示す。特異値は $\sigma_1=1.618$ と $\sigma_2=0.618$ であり、第 1 右特異ベクトル $\mathbf{v}_1$ は $58.3^\circ$ の角度で $x$ 軸の上方を向く一方、その像の方向 $\mathbf{u}_1$ は $31.7^\circ$ を向く。入力フレームと出力フレームが異なるため、$\mathbf{U}\ne\mathbf{V}$ となる。これらの正確な数値は :numref:`subsec_mdl-svd-via-ata` で導出する。

<!--en:a0bb0486-->
![単位円に作用するせん断 $\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$（左から右へ読む）。$\mathbf{V}^\top$ が $-58.3^\circ$ 回転させ、$\boldsymbol{\Sigma}$ が $\varphi=1.618$ 倍に $x$ 方向へ、$1/\varphi=0.618$ 倍に $y$ 方向へ伸縮し、$\mathbf{U}$ が $31.7^\circ$ 回転させる。入力フレーム $\{\mathbf{v}_i\}$ と出力フレーム $\{\mathbf{u}_i\}$ は異なる。](../img/mdl-la-svd-action.svg)

<!--en:23e082e8-->
$n$ 次元でも同じ 3 段階によって、単位球面はベクトル $\sigma_i\mathbf{u}_i$ を半軸とする楕円体へと変形される。入力方向のうち正の特異値を持つものが $r<n$ 個だけであれば、残りの $n-r$ 方向はゼロに潰れ、像は平坦な $r$ 次元楕円体となって $\mathbf{u}_1,\ldots,\mathbf{u}_r$ が張る空間の内部に収まる。

<!--en:7671f562-->
**コンパクト形式と薄い形式。**正の特異値の個数を $r$ とする（:numref:`subsec_mdl-svd-subspaces` で $r$ がまさに $\mathbf{A}$ のランクに一致することを確かめる）。$\boldsymbol{\Sigma}$ の最初の $r$ 個を超える行と列はすべて 0 であるため、それらと対になる特異ベクトルは積に一切寄与しない。これらのゼロ成分を取り除くと、*コンパクト SVD*（compact SVD）$\mathbf{A}=\mathbf{U}_r\boldsymbol{\Sigma}_r\mathbf{V}_r^\top$ が得られる。ここで $\mathbf{U}_r\in\mathbb{R}^{m\times r}$ と $\mathbf{V}_r\in\mathbb{R}^{n\times r}$ は最初の $r$ 本の特異ベクトルを並べた行列であり、$\boldsymbol{\Sigma}_r=\operatorname{diag}(\sigma_1,\ldots,\sigma_r)$ である。完全形式とコンパクト形式の中間に位置するのが*薄い SVD*（thin SVD）である。これは数値計算ライブラリで縮小形式を指定した際に返される既定の形式であり、$\min(m,n)$ 本の列を $\mathbf{U}$ と $\mathbf{V}$ に残しつつ、$r<\min(m,n)$ のときにはゼロ特異値もそのまま記録する。ランク 2 の $3\times2$ 行列では、完全形式の因子の形状は $3\times3$、$3\times2$、$2\times2$ であるのに対し、薄い形式の因子の形状は $3\times2$、$2\times2$、$2\times2$ となる（この行列は列フルランクであるため、ここではコンパクト SVD と一致する）。完全形式と薄い形式は行列が正方でないときに異なり、薄い形式とコンパクト形式は行列がランク落ちしているときに異なる。:numref:`fig_mdl-la-svd-forms` に、3 つの形式がすべて異なるランク 3 の $6\times4$ 行列を示す。

<!--en:3e459e71-->
![$m\times n$ 行列（ランク $r$）における SVD 因子の形状（$m=6$、$n=4$、$r=3$ の縮尺で描画）。(a) 完全形式、(b) 薄い形式、(c) コンパクト形式、(d) $r$ 個のランク 1 項からなるダイアド和。$\boldsymbol{\Sigma}$ の対角線上の塗りつぶされた正方形は正の特異値であり、破線の正方形はゼロ特異値である。$\mathbf{U}$ と $\mathbf{V}^\top$ の破線部分は $\boldsymbol{\Sigma}$ のゼロとしか掛からないため、より小さい各形式では積を変えずにそれらの一部を省ける。](../img/mdl-la-svd-forms.svg)

<!--en:d0680536-->
**ダイアド和。**コンパクト SVD を列と行の積として展開すると、$\mathbf{A}$ はしばしば*ダイアド*と呼ばれるランク 1 行列の重み付き和（ダイアド和、dyadic sum）として表される。

<!--en:2ecf1d3f-->
各項 $\mathbf{u}_i\mathbf{v}_i^\top$ は出力方向と入力方向の外積に特異値の重みを掛けたものであり、重い項から軽い項へとあらかじめ並んでいる。せん断行列の場合は $\mathbf{T}=\varphi\,\mathbf{u}_1\mathbf{v}_1^\top+\varphi^{-1}\,\mathbf{u}_2\mathbf{v}_2^\top$ となり、重み $1.618$ の成分と重み $0.618$ の 2 つ目の成分に分解される。このうち重い成分だけを残すというのが、:numref:`subsec_mdl-eckart-young` で扱う低ランク近似の基本的な考え方である。

<!--en:6c4b9b93-->
### 存在性

<!--en:4670fa6f-->
$\sigma_i$、$\mathbf{v}_i$、$\mathbf{u}_i$ はどこから現れ、なぜ出力方向まで自動的に正規直交になるのだろうか。:numref:`subsec_mdl-spectral-theorem` のスペクトル定理により、あらゆる対称行列は実固有値を持つ固有ベクトルの正規直交基底を備えている。これを*グラム行列*（Gram matrix）$\mathbf{A}^\top\mathbf{A}$、すなわちサイズ $n\times n$ の $\mathbf{A}$ の列同士のドット積からなる行列に適用できる。この行列は対称行列であり、さらに半正定値（positive semidefinite、:numref:`subsec_mdl-psd`）でもある。なぜなら、$\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}=\|\mathbf{A}\mathbf{x}\|^2\ge0$ がすべてのベクトル $\mathbf{x}$ に対して成り立つからである。

<!--en:7a0e69fe-->
**:eqref:`eq_mdl-svd` の証明（存在性）。**スペクトル定理より、$\mathbf{A}^\top\mathbf{A}$ は固有ベクトルの正規直交基底 $\mathbf{v}_1,\ldots,\mathbf{v}_n$ と、対応する固有値 $\lambda_1\ge\cdots\ge\lambda_n$ を持つ。$\lambda_i=\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_i=\|\mathbf{A}\mathbf{v}_i\|^2$ であるから、すべての固有値は非負である。そこで次のように定義する。

<!--en:41a4e115-->
すなわち、特異値はグラム行列の固有値の平方根であり、$r$ は正の特異値の個数を表す。各 $i\le r$ に対して $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$ とおく。ここで最も重要な確認事項は、こうして写されたベクトルが自動的に正規直交になることであり、これは $\mathbf{v}_j$ が $\mathbf{A}^\top\mathbf{A}$ の固有ベクトルであることと、$\mathbf{v}_i$ が正規直交であることから直ちに従う。

<!--en:0ec35a98-->
したがって、入力フレームの正規直交性は $\mathbf{A}^\top\mathbf{A}$ を介してそのまま出力フレームへと受け継がれる。次に次元を確認しよう。$r$ 本の正規直交ベクトル $\mathbf{u}_1,\ldots,\mathbf{u}_r$ は $\mathbb{R}^m$ に属するため $r\le m$ であり、固有値は全部で $n$ 個あるため $r\le n$ である。ゆえに $r\le\min(m,n)$ となり、$\lambda_i=0$ がすべての $i>r$ で成り立つ。$\mathbf{u}_1,\ldots,\mathbf{u}_r$ を延長して正規直交基底 $\mathbf{u}_1,\ldots,\mathbf{u}_m$（$\mathbb{R}^m$ の基底）を作る（たとえば任意の基底延長に Gram–Schmidt 法を適用すればよい）。$r<i\le n$ に対しては $\|\mathbf{A}\mathbf{v}_i\|^2=\lambda_i=0$ であるから、$\mathbf{A}\mathbf{v}_i=\mathbf{0}$ となる。ここで $\mathbf{A}\mathbf{V}$ と $\mathbf{U}\boldsymbol{\Sigma}$ を 1 列ずつ比較すると、第 $i$ 列（$\mathbf{U}\boldsymbol{\Sigma}$ の列）は、$\sigma_i\mathbf{u}_i$（$i\le\min(m,n)$ のとき。$i>r$ ではゼロになる）または $\mathbf{0}$（$i>m$ のとき。この範囲には $\boldsymbol{\Sigma}$ の対角成分がない）である。いずれの場合も、第 $i$ 列（$\mathbf{A}\mathbf{V}$）は第 $i$ 列（$\mathbf{U}\boldsymbol{\Sigma}$）と一致する。よって $\mathbf{A}\mathbf{V}=\mathbf{U}\boldsymbol{\Sigma}$ が成り立ち、右から $\mathbf{V}^\top=\mathbf{V}^{-1}$ を掛ければ $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ が得られる。$\blacksquare$

<!--en:9d92fdeb-->
この証明はそのまま計算手順にもなっている。対称行列 $\mathbf{A}^\top\mathbf{A}$ を対角化し、固有値の平方根をとって特異値を求め、固有ベクトルを $\mathbf{A}$ で写して左特異ベクトルを得ればよい。

<!--en:44f04639-->
各ステップの働きを具体的に確かめるため、せん断行列を手計算で追ってみよう。$\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ に対し、グラム行列は $\mathbf{T}^\top\mathbf{T}=\left(\begin{smallmatrix}1&1\\1&2\end{smallmatrix}\right)$ である。トレースは 3、行列式は 1 であるから、特性多項式は $\lambda^2-3\lambda+1$ となり、固有値 $(3\pm\sqrt5)/2$ が得られる。$\varphi^2=\varphi+1$ を満たす黄金比 $\varphi=(1+\sqrt5)/2$ を導入すると、これらの固有値は $\varphi^2$ と $\varphi^{-2}$ へと綺麗に簡約される。したがって特異値は $\sigma_1=\varphi\approx1.618$ および $\sigma_2=1/\varphi\approx0.618$ となり、その積は $1=|\det\mathbf{T}|$ である。

<!--en:d8f7618b-->
続いて特異ベクトルを求めよう。第 1 右特異ベクトル $(x,y)$ について、$(\mathbf{T}^\top\mathbf{T}-\varphi^2\mathbf{I})(x,y)=\mathbf{0}$ の第 1 行は $(1-\varphi^2)x+y=0$ となり、$\varphi^2-1=\varphi$ を用いると $y=\varphi x$、すなわち $\mathbf{v}_1\propto(1,\varphi)$ が得られる。これに $\mathbf{T}$ を作用させると $\mathbf{T}(1,\varphi)=(1+\varphi,\varphi)=\varphi\,(\varphi,1)$ となるため、$\mathbf{u}_1\propto(\varphi,1)$ である。両ベクトルとも長さは $\sqrt{1+\varphi^2}=\sqrt{\varphi+2}$ であり、正規化すると $\mathbf{v}_1\approx(0.526,0.851)$（角度 $\arctan\varphi=58.3^\circ$）および $\mathbf{u}_1\approx(0.851,0.526)$（角度 $31.7^\circ$）となる。第 2 の組は直交性から $\mathbf{v}_2=(-\varphi,1)/\sqrt{\varphi+2}$、$\mathbf{u}_2=(-1,\varphi)/\sqrt{\varphi+2}$ と求まり、実際 $\mathbf{T}\mathbf{v}_2=(1-\varphi,1)/\sqrt{\varphi+2}=\varphi^{-1}\mathbf{u}_2$ が $(1-\varphi)\varphi=-1$ より成り立つ。$\mathbf{V}$ と $\mathbf{U}$ はどちらも回転行列 $\mathbf{R}_\theta=\left(\begin{smallmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{smallmatrix}\right)$ であるため、せん断行列は次のように分解される。

<!--en:47393a35-->
幾何学的には、このせん断は平面を $-58.3^\circ$ 回転させ、$1.618$ 倍に $x$ 軸方向へ、$0.618$ 倍に $y$ 軸方向へ伸縮し、最後に $31.7^\circ$ だけ回転させて戻す。2 つの角度の和が $90^\circ$ になるのは、$\mathbf{u}_1$ が単に $\mathbf{v}_1$ の座標を入れ替えたものだからであり、これは対称性 $\mathbf{T}^\top=\mathbf{J}\mathbf{T}\mathbf{J}$（入れ替え行列 $\mathbf{J}=\left(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right)$ に関する関係）を反映している。

<!--en:36678fdf-->
NumPy の `np.linalg.svd` はこの手計算と一致するだろうか。特異ベクトルは符号を除いてしか定まらないため（:numref:`subsec_mdl-svd-uniqueness`）、以下のセルではせん断行列を `A` として格納し、まず $\mathbf{u}_1$ が第 1 象限に入り $\mathbf{U}$ が純粋な回転になるよう符号を揃えている。

<!--en:1e2c6b65-->
結果は一致する。ライブラリは $\varphi$ と $1/\varphi$ に加え、:eqref:`eq_mdl-shear-svd` の 2 つの回転を返す。

<!--en:88c7593a-->
同じ構成法は長方形行列に対してもそのまま機能する。$3\times2$ 行列として、各行が $(3,1)$、$(1,3)$、$(0,2)$ であるものを考えよう。そのグラム行列は $\left(\begin{smallmatrix}10&6\\6&14\end{smallmatrix}\right)$ であり、固有値は $12\pm\sqrt{40}$ である。次のセルでは、証明の 3 ステップ（$\mathbf{A}^\top\mathbf{A}$ の固有ベクトル、平方根、および $\mathbf{u}_i=\mathbf{A}\mathbf{v}_i/\sigma_i$）をそのまま実行し、結果を `np.linalg.svd` と比較する。

<!--en:7c62d6a9-->
どちらの方法でも $\sigma_1=\sqrt{12+\sqrt{40}}\approx4.281$ と $\sigma_2=\sqrt{12-\sqrt{40}}\approx2.382$ が得られる。証明で保証されたとおり、写されたベクトル $\mathbf{u}_i$ は正規直交となり、$\mathbf{U}$ と $\mathbf{V}$ はともに符号を除いてライブラリの出力と一致する。

<!--en:a244dc02-->
### 一意性と最大伸縮率

<!--en:a52a07e5-->
行列の SVD は一意だろうか。特異*値*は常に一意である。$\mathbf{A}^\top\mathbf{A}$ の固有値の平方根であり、$\mathbf{A}$ によって完全に定まるからである。これに対し、特異*ベクトル*には次の 3 種類の自由度がある。

<!--en:ceb306e0-->
* 単根（重複しない）の $\sigma_i>0$ に対しては、自由度は同時符号反転 $(\mathbf{u}_i,\mathbf{v}_i)\to(-\mathbf{u}_i,-\mathbf{v}_i)$ のみである。片方だけの符号を反転させると $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ が崩れるため、両方のベクトルを同時に反転させなければならない。

<!--en:9390dce3-->
* 重複特異値 $\sigma>0$（添字集合を $\mathcal{I}$ とする）に対しては、列集合 $\mathbf{V}_{\mathcal I}$ に同じサイズの任意の直交行列 $\mathbf{Q}$ を掛けて $\mathbf{V}_{\mathcal I}\mathbf{Q}$ と回転させてよい。ただし、$\mathbf{U}_{\mathcal I}$ にも*同じ* $\mathbf{Q}$ を掛けて $\mathbf{U}_{\mathcal I}=\mathbf{A}\mathbf{V}_{\mathcal I}/\sigma$ を保つ必要がある。

<!--en:e4ef4b12-->
* $\sigma=0$ に対しては、零空間のベクトル $\mathbf{v}_{r+1},\ldots,\mathbf{v}_n$ と左零空間のベクトル $\mathbf{u}_{r+1},\ldots,\mathbf{u}_m$ はダイアド和 :eqref:`eq_mdl-svd-dyadic` に現れないため、互いに完全に独立に回転させることができる。

<!--en:c52651fe-->
2 つの数値計算ライブラリ（あるいは同じライブラリでも異なる 2 台の計算機）が特異ベクトルに対して逆の符号を返すことがあるのも、また本節のコードセルでベクトルを表示する前に符号の規約を固定しているのも、この自由度のためである。

<!--en:96cb4a17-->
$\mathbf{A}$ を分解するだけでなく、最大特異値には直接的な幾何学的意味がある。それは $\mathbf{A}$ が任意の単位ベクトルを引き伸ばせる最大の倍率である。

<!--en:544416cb-->
これは :numref:`subsec_mdl-rayleigh` の Rayleigh の命題から直ちに従う。すなわち、単位ベクトル $\mathbf{x}$ 全体にわたって二次形式（quadratic form）$\mathbf{x}^\top\mathbf{S}\mathbf{x}$（対称行列 $\mathbf{S}$ に対するもの）を最大化すると、最大固有ベクトルにおいて最大固有値に等しい最大値をとる。$\mathbf{S}=\mathbf{A}^\top\mathbf{A}$ とおけば、$\|\mathbf{A}\mathbf{x}\|^2=\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}$ は $\mathbf{x}=\mathbf{v}_1$ で最大値 $\lambda_1=\sigma_1^2$ をとることがわかる。したがって、$\sigma_1$ はまさにスペクトルノルム $\|\mathbf{A}\|_2$（:numref:`sec_linear-algebra`）に一致する。せん断行列では、いかなる単位ベクトルも $\varphi=1.618$ 倍を超えて引き伸ばされることはなく、$\mathbf{v}_1$ がその最大伸縮率を達成する。残りの特異値も、先行する右特異ベクトルと直交する空間に制限して同じ最大化問題を解くことで順に得られる。この手続きを*デフレーション*と呼び、$\sigma_k=\max\{\|\mathbf{A}\mathbf{x}\| : \|\mathbf{x}\|=1,\ \mathbf{x}\perp\mathbf{v}_1,\ldots,\mathbf{v}_{k-1}\}$ と書ける。実際、演習問題 :exref:`ex_mdl-svd-low-rank-norms-singular-values`a ではこの論理を逆転させ、スペクトル定理を使わずに最大伸縮率だけから SVD 全体を構成する。

<!--en:f6f86406-->
### 特異値と固有値

<!--en:4d90fcfb-->
特異値と固有値はどのような関係にあるのだろうか。:eqref:`eq_mdl-svd` を 2 つのグラム行列に代入し、$\mathbf{U}^\top\mathbf{U}=\mathbf{I}$ と $\mathbf{V}^\top\mathbf{V}=\mathbf{I}$ を用いると次式が得られる。

<!--en:d41030c0-->
これら両式はいずれも対称行列の固有値分解になっている。すなわち、右特異ベクトルは $\mathbf{A}^\top\mathbf{A}$ の固有ベクトル、左特異ベクトルは $\mathbf{A}\mathbf{A}^\top$ の固有ベクトルであり、二乗 $\sigma_i^2$ は 2 つのグラム行列が共有する非ゼロ固有値である。$\mathbf{A}$ が対称半正定値行列のときは、SVD と固有値分解は $\mathbf{U}=\mathbf{V}$ として一致する。

<!--en:e2f37a27-->
しかし、それ以外のあらゆる行列では 2 つの分解は異なり、特異値が固有値の絶対値に等しいと思い込むのはよくある誤りである。本章で繰り返し用いている行列 $\mathbf{A}=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ を考えよう。その固有値は $2\pm i$ であり、絶対値はともに $\sqrt5\approx2.236$ である（:numref:`sec_mdl-eigendecompositions`）。一方、グラム行列 $\mathbf{A}^\top\mathbf{A}=\left(\begin{smallmatrix}2&-1\\-1&13\end{smallmatrix}\right)$ はトレース 15、行列式 25 を持つため、$\sigma_{1,2}^2=(15\pm5\sqrt5)/2$ となり、したがって $\sigma_{1,2}=(5\pm\sqrt5)/2\approx3.618,\,1.382$ となって $2.236$ とは大きく異なる。それでもなお、あらゆる正方行列において 2 つのリストを結ぶ明快な恒等式が 1 つ成り立つ。

<!--en:ed9397d1-->
第 1 の等号は直交行列の行列式が $\pm1$ であることから従い、第 2 の等号は行列式が全固有値の積に等しいことから従う。$2\times2$ 行列が共通の絶対値 $|\lambda|$ を持つ共役複素固有値の対を備えるとき、ここから $|\lambda|=\sqrt{\sigma_1\sigma_2}$ が強制される。実際、この例でも $\sqrt{3.618\cdot1.382}=\sqrt5$ が成り立っている。またこの恒等式は、:numref:`sec_mdl-linear-maps` で見た行列式の体積としての解釈も復元する。回転は面積を保つため、面積変化のすべてを対角伸縮 $\boldsymbol{\Sigma}$ が担い、上の例題行列は単位正方形を面積 $3.618\cdot1.382=5$ の平行四辺形へと写す。

<!--en:94ab5a4e-->
**実用上の SVD 計算。**本節の存在証明では $\mathbf{A}^\top\mathbf{A}$ から SVD を構成したが、数値計算ライブラリがその積を直接作ることは決してない。$\mathbf{A}^\top\mathbf{A}$ を作ると条件数が二乗され（:numref:`subsec_mdl-condition-number`）、浮動小数点演算において小さな特異値が失われてしまうからである。その代わりにライブラリは、$\mathbf{A}$ に直接直交変換を施して二重対角形へと簡約し（Golub–Kahan 二重対角化）、その二重対角行列を対角化する :cite:`Golub.Van-Loan.1996`。

<!--en:a148c0ea-->
### 4 つの基本部分空間

<!--en:739aeb42-->
$\mathbf{A}$ はどの入力をゼロに潰し、どの出力に到達できるのだろうか。:numref:`sec_mdl-linear-maps` では、これらの集合をそれぞれ零空間（null space）と列空間（column space）と名付け、線形独立な列の最大本数としてランクを定義した。$\mathbf{U}$ と $\mathbf{V}$ は可逆であるため、これらを掛けても次元は一切変わらない。したがって $\operatorname{rank}\mathbf{A}=\operatorname{rank}\boldsymbol{\Sigma}$ となり、ランクは単なる非ゼロ対角成分の個数に帰着する。

<!--en:37dfcf75-->
言い換えれば、ランクとは潰れずに生き残る方向の数を数えたものである。後述する丸め許容誤差と組み合わせれば、特異値を数える方法はランクを判定する上で数値的に最も信頼性が高い。ガウスの消去法より計算コストは高いものの、はるかに安定している。浮動小数点の丸め誤差（rounding error）のような摂動 $\mathbf{E}$ が加わっても、いかなる特異値も $\|\mathbf{E}\|_2$ を超えてずれることはないからである（:numref:`subsec_mdl-eckart-young` の Weyl の不等式）。したがって、大きな特異値と微小な特異値の間に明確なギャップがあれば、その結果を信頼できる。

<!--en:8c328207-->
さらに都合のよいことに、特異ベクトルを添字 $r$ で分割すれば、$\mathbf{A}$ の 4 つの基本部分空間（fundamental subspace）すべての明示的な正規直交基底がダイアド和 :eqref:`eq_mdl-svd-dyadic` から直接得られる。

<!--en:b076611e-->
* $\mathbf{A}$ の**列空間**（値域）は $\operatorname{span}\{\mathbf{u}_1,\ldots,\mathbf{u}_r\}$ である。あらゆる出力 $\mathbf{A}\mathbf{x}=\sum_{i\le r}\sigma_i(\mathbf{v}_i^\top\mathbf{x})\mathbf{u}_i$ はこれらのベクトルの線形結合であり、かつ各基底ベクトルが $\mathbf{u}_i=\mathbf{A}(\mathbf{v}_i/\sigma_i)$ によって実際に達成されるからである。

<!--en:0e5550ad-->
* $\mathbf{A}$ の**零空間**は $\operatorname{span}\{\mathbf{v}_{r+1},\ldots,\mathbf{v}_n\}$ である。同じ式より、$\mathbf{A}\mathbf{x}=\mathbf{0}$ が成り立つのは $\mathbf{v}_i^\top\mathbf{x}=0$ がすべての $i\le r$ で成り立つとき、かつそのときに限る（$\mathbf{u}_i$ は線形独立であり、$\sigma_i$ は正だからである）。これは $\mathbf{x}$ が $\mathbf{v}_1,\ldots,\mathbf{v}_r$ に直交することを意味する。

<!--en:fa36ab35-->
* **行空間**（row space、$\mathbf{A}^\top$ の列空間）は $\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_r\}$ であり、**左零空間**（left null space、$\mathbf{A}^\top$ の零空間）は $\operatorname{span}\{\mathbf{u}_{r+1},\ldots,\mathbf{u}_m\}$ である。これは同じ 2 つの議論を $\mathbf{A}^\top=\mathbf{V}\boldsymbol{\Sigma}^\top\mathbf{U}^\top$（単に $\mathbf{u}_i$ と $\mathbf{v}_i$ の役割を入れ替えたもの）に適用すれば導かれる。

<!--en:d299833e-->
これにより、あらゆる線形変換の見通しが驚くほど明快になる。入力空間は $r$ 次元の行空間と $(n-r)$ 次元の零空間という直交する 2 つの部分空間に分解される。同様に、出力空間も $r$ 次元の列空間と $(m-r)$ 次元の左零空間に分解される。行空間の上では、$\mathbf{A}$ は各 $\mathbf{v}_i$ を $\sigma_i\mathbf{u}_i$ へと写し、列空間への 1 対 1 の全単射を与える。一方、零空間の元はすべてゼロへと送られる。

<!--en:cfd22fc9-->
最も単純な非自明の例でこれを見るため、:numref:`sec_mdl-linear-maps` の特異行列 $\mathbf{B}=\left(\begin{smallmatrix}2&1\\4&2\end{smallmatrix}\right)$ を考えよう。グラム行列 $\left(\begin{smallmatrix}20&10\\10&5\end{smallmatrix}\right)$ の固有値は 25 と 0 であるため、$\sigma_1=5$、$\sigma_2=0$、$r=1$ となる。4 つの部分空間は次の各ベクトルが張る直線である。

<!--en:32931c56-->
ここで列空間は直線 $y=2x$（$\mathbf{u}_1$ の方向）であり、左零空間はそれに直交する $\mathbf{u}_2$ 方向の直線である。入力平面においては、零空間は $(1,-2)$ 方向の直線であり、行空間を張る行ベクトル $(2,1)$ に直交している。:numref:`fig_mdl-la-svd-subspaces` に一般の幾何学的関係を示す。:numref:`subsec_mdl-pseudoinverse` で見るように、擬似逆行列はこの行空間から列空間への全単射をちょうど逆向きにたどる写像にほかならない。

<!--en:5f330662-->
![4 つの基本部分空間。$\mathbf{A}$ は行空間 $\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_r\}$ を列空間 $\operatorname{span}\{\mathbf{u}_1,\dots,\mathbf{u}_r\}$ の上へ $\mathbf{v}_i\mapsto\sigma_i\mathbf{u}_i$ によって 1 対 1 に写し、零空間をゼロへと写す。擬似逆行列 $\mathbf{A}^{+}$（破線）はこの写像を逆向きに $\sigma_i\mathbf{u}_i\mapsto\mathbf{v}_i$ とたどり、左零空間をゼロへと送る。](../img/mdl-la-svd-subspaces.svg)

<!--en:4cc5fccf-->
**数値ランク。**浮動小数点演算においては、数学的にランク落ちしている行列であっても、特異値が文字どおり厳密にゼロになるとは限らない。丸め誤差によって $\epsilon_{\text{mach}}\sigma_1$ 程度の微小な値が残るのが普通だからである。そのため実用上は、閾値を用いて数値ランク（numerical rank）を判定する。NumPy の `np.linalg.matrix_rank` は $\sigma_1\max(m,n)\,\epsilon_{\text{mach}}$ を超える特異値の個数を数えており、この許容誤差は最大特異値と行列の次元の双方に応じて自然にスケールする。ランク 2 の $4\times4$ 行列において、「ゼロ」であるはずの特異値が実際にどのような値になるか見てみよう。

<!--en:f4dd954c-->
2 つの特異値はオーダー 1 であるのに対し、残りの 2 つは純粋な丸めノイズであり、$10^{-15}$ 以下と許容誤差を大きく下回っている。丸めの結果たまたま片方が厳密なゼロになることはあっても、両方がゼロになることはめったにない。したがって厳密なゼロを数えるとランクを誤判定してしまうが、閾値を用いれば真のランク 2 が正しく復元される。

<!--en:0074b762-->
## 低ランク近似

<!--en:a314113a-->
### 最良低ランク近似

<!--en:d86b51f8-->
ここで本節冒頭の猫の写真に戻ろう。どの 20 個のランク 1 成分を残すべきだろうか。ダイアド和 :eqref:`eq_mdl-svd-dyadic` はすでに $\mathbf{A}$ の成分を重いものから軽いものへと並べているため、最初の $k$ 項を残すのが自然な候補となる。こうして得られるのが*打ち切り SVD*（truncated SVD）である。

<!--en:aa89522a-->
行列 $\mathbf{A}_k$ のランクは高々 $k$ であり、これを保存するには $k$ 個の特異値、$k$ 本の長さ $m$ の列、$k$ 本の長さ $n$ の列だけで足りるため、合計 $k(m+n+1)$ 個の数値で済む（元の $mn$ 個の代わり）。$400\times500$ の写真で $k=20$ とすれば、200,000 個のうちわずか 18,020 個の数値である。すると直ちに 2 つの疑問が生じる。近似誤差はどれほど大きいか、そしてもっと巧妙なランク $k$ 行列を作ればさらに良くできるのか、という問いである。

<!--en:93f91be8-->
**特異値で表される行列ノルム。**近似誤差を測るには行列ノルムが必要である。:numref:`sec_linear-algebra` で導入したフロベニウスノルム $\|\mathbf{A}\|_F^2=\sum_{ij}a_{ij}^2$ は、$\mathbf{A}$ の各列の長さの二乗和であり、それらは $\mathbf{A}^\top\mathbf{A}$ の対角成分に等しい。したがって $\|\mathbf{A}\|_F^2=\operatorname{tr}(\mathbf{A}^\top\mathbf{A})$ となる。ここに SVD を代入し、トレースの巡回性（:numref:`sec_mdl-linear-maps`）を使って $\mathbf{V}$ を反対側へ移すと次式が得られる。

<!--en:66067072-->
ゆえにフロベニウスノルムは特異値の二乗和の平方根であり、ここから左右どちらから直交行列を掛けてもフロベニウスノルムは決して変わらないことが直ちにわかる。せん断行列では $1+1+0+1=3=\varphi^2+\varphi^{-2}$ となる。一方、スペクトルノルムは単なる最大特異値 $\|\mathbf{A}\|_2=\sigma_1$ である（:eqref:`eq_mdl-sigma1-variational`）。

<!--en:c18b13f4-->
両方のノルムが特異値で表されたところで、線形代数における中心的な近似定理を述べよう。

<!--en:93d32511-->
**定理（Eckart–Young–Mirsky）。** *$\sigma_i=0$（$i>r$）とおく。すべての $k\ge0$ に対し、ランクが高々 $k$ の行列で達成可能な最小のスペクトルノルム誤差は次式で与えられる。*

<!--en:2208ecb1-->
*また、最小の二乗フロベニウス誤差は $\min_{\operatorname{rank}\mathbf{B}\le k}\|\mathbf{A}-\mathbf{B}\|_F^2=\sum_{i>k}\sigma_i^2$ である。いずれの最小値も、打ち切り SVD である $\mathbf{B}=\mathbf{A}_k$ において達成される。*

<!--en:d4bf4831-->
要するに、どちらのノルムにおいても、いかなるランク $k$ 行列も打ち切り SVD を上回ることはできない。最良のランク $k$ 近似の誤差は、切り捨てた最初の特異値（スペクトルノルムの場合）、あるいは切り捨てたすべての特異値のエネルギーの総和（フロベニウスノルムの場合）に一致する。Eckart と Young がフロベニウスノルムの場合を証明し :cite:`Eckart.Young.1936`、後に Mirsky が左右両側からの直交行列の積に対して不変なすべてのノルムへと拡張した :cite:`Mirsky.1960`。まずスペクトルノルムの評価を証明しよう。フロベニウスノルムの評価は、同じ議論からそのまま導かれるもう 1 つの道具である Weyl の不等式を使えば明快に従う。

<!--en:fcfbd92f-->
一般の証明を書く前に、$n=3$、$k=1$ の場合を思い描いてみよう。ランク 1 の任意の行列 $\mathbf{B}$ は入力のある平面全体をゼロに潰す一方、$\mathbf{A}$ の上位 2 本の右特異ベクトルは別の平面を張る。$\mathbb{R}^3$ において原点を通る 2 つの平面は、少なくとも 1 本の直線で交わらなければならない。その直線上では、$\mathbf{B}$ の出力はゼロであるのに対し、$\mathbf{A}$ はすべてのベクトルを少なくとも $\sigma_2$ 倍に引き伸ばす。したがって $\mathbf{A}-\mathbf{B}$ は、ある単位ベクトルを少なくとも $\sigma_2$ 倍に引き伸ばさざるを得ない（:numref:`fig_mdl-la-svd-dimension-count`）。以下の証明は、この次元の勘定を $n$ 次元へそのまま持ち上げたものである。

<!--en:c04b9e2f-->
![$n=3$、$k=1$ における Eckart–Young の背後にある次元の勘定。平面 $\ker\mathbf{B}$ と上位 2 本の右特異ベクトルが張る平面は直線で交わらなければならない。その直線上では $\mathbf{B}$ はゼロになる一方、$\mathbf{A}$ は少なくとも $\sigma_2$ 倍に引き伸ばす。](../img/mdl-la-svd-dimension-count.svg)

<!--en:900e34e7-->
**証明（スペクトルノルム）。** *第 1 に、打ち切り SVD は誤差 $\sigma_{k+1}$ を達成する。*残差 $\mathbf{A}-\mathbf{A}_k=\sum_{i>k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ はすでに最大特異値 $\sigma_{k+1}$ を持つコンパクト SVD の形をしており、$\|\mathbf{A}-\mathbf{A}_k\|_2=\sigma_{k+1}$ が :eqref:`eq_mdl-sigma1-variational` より従う。同様に、:eqref:`eq_mdl-frobenius-singular` から $\|\mathbf{A}-\mathbf{A}_k\|_F^2=\sum_{i>k}\sigma_i^2$ も得られる。

<!--en:7357527d-->
*第 2 に、ランクが高々 $k$ のいかなる行列もこれを下回ることはできない。*$k\ge r$ ならば誤差はゼロで証明すべきことはないため、$k<r$ と仮定し、$\operatorname{rank}\mathbf{B}\le k$ とする。ランク・退化次数の定理（:numref:`sec_mdl-linear-maps`）より、零空間 $\mathcal{K}=\ker\mathbf{B}$ の次元は少なくとも $n-k$ である。一方、張る空間 $\mathcal{V}=\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_{k+1}\}$（上位 $k+1$ 本の右特異ベクトルによるもの）の次元は $k+1$ である。これら 2 つの次元の和は $n$ を超えるため、2 つの部分空間は非ゼロベクトルを共有しなければならない。

<!--en:adafd794-->
ここで等号は 2 つの部分空間の次元公式であり、不等号は $\dim(\mathcal{K}+\mathcal{V})\le n$ を用いている（次元公式を確かめるには、$\mathcal{K}\cap\mathcal{V}$ の基底をそれぞれ $\mathcal{K}$ の基底と $\mathcal{V}$ の基底へ別々に延長すればよい。合わせたベクトル群は線形独立で $\mathcal{K}+\mathcal{V}$ を張り、その本数は $\dim\mathcal{K}+\dim\mathcal{V}-\dim(\mathcal{K}\cap\mathcal{V})$ となる）。そこで単位ベクトル $\mathbf{x}\in\mathcal{K}\cap\mathcal{V}$ を 1 つ選び、$\mathbf{x}=\sum_{i\le k+1}c_i\mathbf{v}_i$（$\sum_i c_i^2=1$）と展開する。すると $\mathbf{B}\mathbf{x}=\mathbf{0}$ であり、$\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ と $\mathbf{u}_i$ の正規直交性を用いれば次式を得る。

<!--en:45c3494d-->
ゆえに $\|\mathbf{A}-\mathbf{B}\|_2\ge\|(\mathbf{A}-\mathbf{B})\mathbf{x}\|=\|\mathbf{A}\mathbf{x}\|\ge\sigma_{k+1}$ が成り立つ。$\blacksquare$

<!--en:f49f1112-->
この幾何学的意味は一文で繰り返しておく価値がある。任意のランク $k$ の競合行列 $\mathbf{B}$ は、上位 $k+1$ 本の右特異ベクトルが張る空間の中の少なくとも 1 方向をゼロに潰さなければならず、そして $\mathbf{A}$ はその空間内のあらゆる方向を少なくとも $\sigma_{k+1}$ 倍に引き伸ばすのである。せん断行列 $\mathbf{T}$ で $k=1$ とすると、打ち切り $\mathbf{T}_1=\varphi\,\mathbf{u}_1\mathbf{v}_1^\top=\tfrac{1}{\sqrt5}\left(\begin{smallmatrix}\varphi&\varphi^2\\1&\varphi\end{smallmatrix}\right)\approx\left(\begin{smallmatrix}0.724&1.171\\0.447&0.724\end{smallmatrix}\right)$ の誤差は両ノルムで $1/\varphi=0.618$ となり（残差がランク 1 であるためスペクトルノルムとフロベニウスノルムが一致する）、いかなるランク 1 行列もこれより近づくことはできない。

<!--en:6c64e862-->
フロベニウスノルムについて定理の証明を完結させるため、今証明したスペクトルノルムの評価から一般的な摂動不等式を先に導いておこう。

<!--en:0f64085d-->
**補題（特異値に対する Weyl の不等式）。** *同じ形状を持つ任意の 2 つの行列 $\mathbf{X},\mathbf{Y}$ と、添字 $i,j\ge1$（$i+j-1\le\min(m,n)$）に対し、次式が成り立つ。*

<!--en:f2c571f5-->
**証明。**$\mathbf{X}_{i-1}$ と $\mathbf{Y}_{j-1}$ を打ち切り SVD :eqref:`eq_mdl-truncated-svd` とする（$\mathbf{X}_0=\mathbf{0}$ とおく）。その和 $\mathbf{X}_{i-1}+\mathbf{Y}_{j-1}$ のランクは高々 $i+j-2$ であるから、スペクトルノルムの下界を $\mathbf{X}+\mathbf{Y}$ に $k=i+j-2$ として適用すれば $\sigma_{i+j-1}(\mathbf{X}+\mathbf{Y})\le\|\mathbf{X}+\mathbf{Y}-\mathbf{X}_{i-1}-\mathbf{Y}_{j-1}\|_2$ が得られる。三角不等式（triangle inequality）と Eckart–Young の達成可能性の議論より、右辺は高々 $\|\mathbf{X}-\mathbf{X}_{i-1}\|_2+\|\mathbf{Y}-\mathbf{Y}_{j-1}\|_2=\sigma_i(\mathbf{X})+\sigma_j(\mathbf{Y})$ である。$\blacksquare$

<!--en:b4337401-->
言葉で言えば、$\mathbf{Y}$ を $\mathbf{X}$ に加えたとき、第 $(i+j-1)$ 特異値が第 $i$ 特異値（$\mathbf{X}$ のもの）を上回る量は高々 $\sigma_j(\mathbf{Y})$ に抑えられる。$j=1$ として不等式を両方向に適用すれば、行列に $\mathbf{Y}$ の摂動を加えてもいかなる特異値も $\|\mathbf{Y}\|_2$ を超えて動かないことがわかる :cite:`Horn.Johnson.2012`。詳細は演習問題 :exref:`ex_mdl-svd-low-rank-weyl-perturbation` で扱う。

<!--en:d75344c7-->
**証明（フロベニウスノルム）。**Weyl の補題を $\mathbf{X}=\mathbf{A}-\mathbf{B}$、$\mathbf{Y}=\mathbf{B}$、$j=k+1$ として適用する。$\operatorname{rank}\mathbf{B}\le k$ より $\sigma_{k+1}(\mathbf{B})=0$ であるから、$\sigma_{k+i}(\mathbf{A})\le\sigma_i(\mathbf{A}-\mathbf{B})$ がすべての $i\ge1$ に対して成り立つ。両辺を二乗して $i$ について和をとり、:eqref:`eq_mdl-frobenius-singular` を適用すると次式を得る。

<!--en:5abcff95-->
そしてこの下界を打ち切り $\mathbf{A}_k$ が達成する。$\blacksquare$

<!--en:1cadbd57-->
**最小点の一意性。**2 つのノルムの間には 1 つ微妙な違いがある。スペクトルノルムでは、最良ランク $k$ 近似は $k\ge1$ かつ $\sigma_{k+1}>0$ である限り一意になることは決してない。たとえば $\mathbf{A}=\operatorname{diag}(3,2,1)$、$k=1$ のとき、行列 $\mathbf{B}=\operatorname{diag}(3+t,0,0)$（$|t|\le1$）はすべて $\|\mathbf{A}-\mathbf{B}\|_2=\max(|t|,2,1)=2=\sigma_2$ を与える。対照的にフロベニウスノルムでは、$\mathbf{A}_k$ は $\sigma_k>\sigma_{k+1}$ または $\sigma_{k+1}=0$（この場合は $\mathbf{A}_k=\mathbf{A}$）である限り*一意*な最小点（minimizer）となる :cite:`Golub.Van-Loan.1996`。次に見る主成分分析がフロベニウスノルムで定式化される理由の 1 つが、この一意性にある。

<!--en:c4480424-->
それでは、$400\times500$ の猫の写真でこの定理を検証してみよう。Eckart–Young は各打ち切りのスペクトル誤差を $\sigma_{k+1}$ に定めるが、実際の画像の特異値がどれほど速く減衰するかや、素朴なランク $k$ の競合手法がどれほど劣るかまでは教えてくれない。以下のセルでは `img/cat1.jpg` を読み込み、赤・緑・青に対する標準的な輝度重みを使ってグレースケールへ変換し、`np.einsum` を用いて打ち切り近似を構築する（この関数は入力を掛け合わせ、矢印の後に省かれたすべての添字について和をとる。第 $i$ 行（`Vt` の行）は $\mathbf{v}_i^\top$ であるため、`'i,mi,in->mn'` でダイアド和が計算される）。

<!--en:ae5b5bb0-->
保証どおり、すべてのランクでスペクトル誤差は $\sigma_{k+1}$ と一致し、ランク 20 ではわずか 9% の数値を保持するだけで相対誤差 8% に収まっている。:numref:`fig_mdl-la-eckart-young` を見ると、猫はランク 1 では明るいぼやけた塊にすぎないが、ランク 5 では 4 本の足を持つシルエットとして識別でき、ランク 20 になると仰向けに寝転ぶ猫だと紛れもなくわかる。左パネルを見れば、なぜこれほど少数の成分で足りるのかが明らかになる。最初の 100 個の添字にわたって、特異値はおよそべき乗則 $\sigma_i\propto i^{-1.1}$ に従って減衰しているのである。

<!--en:af0703e5-->
![$400\times500$ の写真における Eckart–Young。左：特異値はおよそべき乗則に従って減衰する（両対数軸。網掛け帯はランク 20 のパネルで残した 20 個を示す）。右：各打ち切り $\mathbf{A}_k$ と、その相対フロベニウス誤差および元の数値数に対する保持割合 $k(m+n+1)/mn$。](../img/mdl-la-eckart-young.svg)

<!--en:b59687a6-->
**エネルギー比。**フロベニウス誤差の公式は、$k$ を選ぶための実用的な指針を与える。二乗「エネルギー」の総和 $\sum_i\sigma_i^2$ のうち、ランク $k$ の打ち切りによって保持される割合は次式で表される。

<!--en:89a75956-->
すなわち、相対フロベニウス誤差は 1 からエネルギー比を引いた値の平方根になる。したがってエネルギーの 95% を保持することは相対誤差 $\sqrt{0.05}\approx22\%$ に対応し、99% を保持すれば誤差は 10% まで下がる。ただし、この比率の解釈には注意が必要である。上の写真では、定数画像がランク 1 であり（すべての行が同一）、かつ写真の平均輝度が大きいため、最初のダイアドが主にその平均を捉えることになり、$\sigma_1^2$ だけでエネルギーの 89% を占めてしまう。そのままではランク 4 ですでにエネルギーの 95% を保持する（相対誤差 21%）が、ピクセル値の平均をあらかじめ差し引くと、95% に達するにはランク 8 が必要になる（4 ではなくなることは :numref:`subsec_mdl-spectrum-control` で確認する）。常に $\sigma_1$ が非ゼロの平均に支配されていないか確認するか、主成分分析と同様にまず平均を差し引くべきである。

<!--en:36b18338-->
素朴なランク 20 の手法と比べて Eckart–Young がどれほど優れているかを見るため、セルの末尾では写真からノルムが最大の実列を 20 本選び、すべての列をそれらが張る空間へ射影している（行列自身の行と列の部分集合で近似する CUR 分解の背後にある考え方である）。その相対誤差は 0.31 であり、打ち切り SVD の 0.08 と比べて 4 倍近く悪い。

<!--en:d2716bb4-->
**ノイズ除去としての打ち切り。**SVD を打ち切る理由は圧縮だけではなく、ノイズ除去（denoising）も重要な用途である。観測された $m\times n$ 行列（$m\le n$）が低ランクの信号にランダムノイズが加わった $\mathbf{A}=\mathbf{B}+\mathbf{N}$ であり、$\mathbf{N}$ の各成分が平均 0、標準偏差 $\sigma_{\text{noise}}$ で独立だとする。:numref:`subsec_mdl-sample-covariance-mp` のマルチェンコ・パスツール則を $\mathbf{N}\mathbf{N}^\top/n$ に適用すると、大きなノイズ行列の特異値は区間 $[(\sqrt n-\sqrt m)\sigma_{\text{noise}},(\sqrt n+\sqrt m)\sigma_{\text{noise}}]$ の内部に集中する。特異値がこの上端を大きく超える信号成分はノイズの塊（バルク）からくっきりと浮かび上がる一方、弱い成分はその中に埋もれて消える。ノイズレベル（noise level）が既知の正方行列に対し、Gavish と Donoho は漸近的に最適なハード閾値が $\tfrac{4}{\sqrt3}\sqrt n\,\sigma_{\text{noise}}\approx2.309\sqrt n\,\sigma_{\text{noise}}$ であり、ノイズの上端 $2\sqrt n\,\sigma_{\text{noise}}$ よりわずかに上に位置することを証明した :cite:`Gavish.Donoho.2014`。演習問題 :exref:`ex_mdl-svd-low-rank-optimal-hard-threshold` で定数 $4/\sqrt3$ を導出する。

<!--en:d5af3c15-->
### 主成分分析

<!--en:713127de-->
各データ例を行として並べたデータセット $\mathbf{X}\in\mathbb{R}^{n\times d}$（データ点数 $n$、特徴量数 $d$）が与えられたとしよう。データの変動の大半がごく少数の方向に集中していることは多い。ここで 2 つの自然な問いが生じる。データの最大の分散を捉える単位方向はどれか、そして二乗誤差を最小にする $k$ 次元アフィン部分空間はどれか、という問いである。驚くべきことに、両者の答えは完全に一致し、中心化したデータの打ち切り SVD そのものになる。

<!--en:b9629589-->
**中心化。**まず任意の $k$ 次元部分空間とその直交射影行列 $\mathbf{P}$ を固定し、どのオフセット $\mathbf{c}$ が総二乗再構成誤差（reconstruction error）$\sum_i\|(\mathbf{I}-\mathbf{P})(\mathbf{x}_i-\mathbf{c})\|^2$ を最小化するかを考えよう。この目的関数は $\mathbf{c}$ に関する凸二次関数であり、その勾配 $-2(\mathbf{I}-\mathbf{P})\sum_i(\mathbf{x}_i-\mathbf{c})$ は標本平均 $\mathbf{c}=\bar{\mathbf{x}}$ において $\mathbf{P}$ によらずゼロになる。したがって最良のアフィン部分空間は常に平均を通るため、まずデータの*中心化*（centering）から始める。すなわち $\tilde{\mathbf{X}}=\mathbf{X}-\mathbf{1}\bar{\mathbf{x}}^\top$ とおく。その経験共分散は対称半正定値行列 $\mathbf{C}=\tfrac1n\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ である（不偏分散の係数 $\tfrac1{n-1}$ を使ってもすべての固有値が同じ倍率でスケールされるだけであり、主方向もそれらが説明する分散の割合も変わらない）。

<!--en:8c725c8f-->
**命題（Rayleigh による PCA）。** *すべての単位ベクトル $\mathbf{w}$ にわたって、射影後の分散 $\tfrac1n\|\tilde{\mathbf{X}}\mathbf{w}\|^2=\mathbf{w}^\top\mathbf{C}\mathbf{w}$ を最大化するのは $\mathbf{C}$ の最大固有ベクトル、すなわち最大右特異ベクトル $\mathbf{v}_1$（$\tilde{\mathbf{X}}$ のもの）であり、最大分散は $\sigma_1^2/n$ である。*

<!--en:bb618499-->
**証明。**中心化した点を $\mathbf{w}$ の上へ射影するとスカラー座標 $\tilde{\mathbf{x}}^\top\mathbf{w}$ が得られ、その経験分散は $\tfrac1n\sum_i(\tilde{\mathbf{x}}_i^\top\mathbf{w})^2=\tfrac1n\|\tilde{\mathbf{X}}\mathbf{w}\|^2=\mathbf{w}^\top\mathbf{C}\mathbf{w}$ となる。:numref:`subsec_mdl-rayleigh` の Rayleigh の命題より、この二次形式は $\mathbf{C}$ の最大固有ベクトルで最大値 $\lambda_1(\mathbf{C})$ をとる。:eqref:`eq_mdl-svd-gram` より $\mathbf{C}=\tfrac1n\mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top$ であるから、$\mathbf{C}$ の固有ベクトルは右特異ベクトル $\mathbf{v}_i$（$\tilde{\mathbf{X}}$ のもの）であり、固有値は $\sigma_i^2/n$ である。$\blacksquare$

<!--en:f94c8024-->
$\mathbf{v}_1$ に直交する単位ベクトルの中では分散は $\mathbf{v}_2$ で最大化され、帰納的に続ければ上位 $k$ 個の*主方向*（principal direction）$\mathbf{v}_1,\ldots,\mathbf{v}_k$ が得られ、第 $i$ 成分が分散 $\sigma_i^2/n$ を説明する。ここで再構成誤差を確認しよう。中心化した各点を $\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$ へ射影する操作は、$\tilde{\mathbf{X}}$ に右から $\mathbf{V}_k\mathbf{V}_k^\top$ を掛けることに対応する。$\mathbf{V}^\top\mathbf{V}_k$ は単位行列の最初の $k$ 列を取り出すため、次式が成り立つ。

<!--en:e55917e2-->
したがって、上位 $k$ 個の主方向への射影は、中心化データのランク $k$ 打ち切り SVD をとることと完全に一致する。フロベニウスノルムにおける Eckart–Young の定理より、他のいかなる射影も、さらには他のいかなるランク $k$ 行列も、より小さい二乗誤差で中心化データ点を再構成することはできない。切り捨てられた分散 $\sum_{i>k}\sigma_i^2/n$ は平均二乗再構成誤差に等しく、説明分散比（explained-variance ratio）はエネルギー比 :eqref:`eq_mdl-energy-ratio`（$\tilde{\mathbf{X}}$ に対するもの）そのものである。

<!--en:ace4706c-->
:numref:`fig_mdl-la-pca` に、300 個の点からなる点群（標準偏差 3 と 0.8 を持ち、$\pi/5=36^\circ$ 回転した軸に沿う分布）と、その 2 本の主軸を示す。母集団においては、第 1 方向が分散の $9/9.64\approx93.4\%$ を占め、長軸に一致する。以下のセルでは、300 点の有限標本からこれらの値がどれほど正確に復元されるかを確認する。

<!--en:45926ed7-->
![300 個の点からなる点群（中心 $(1,0.6)$、標準偏差 3 と 0.8、回転角 $36^\circ$ の軸に沿う分布）と、平均から描いた主軸（各軸の長さは標準偏差の 2 倍である $2\sigma_i/\sqrt{n}$）。第 1 主軸は最大分散の方向を指す。](../img/mdl-la-pca.svg)

<!--en:eb56a887-->
乱数抽出を繰り返すと、第 1 主方向は分散のおよそ 91% から 95% を説明し、真の長軸から数度以内に収まる。しかしデータの中心化を忘れると、平均のオフセットが点群の広がりを超えた途端に、第 1 特異ベクトルは点群の平均の方向へと振れてしまい、点群が*どのように*変動しているかではなく*どこに*位置しているかを表すようになってしまう。演習問題 :exref:`ex_mdl-svd-low-rank-centering-pca` でこの影響を定量化する。

<!--en:cb16e640-->
## SVD による連立一次方程式の解法

<!--en:a8b1ed70-->
### 擬似逆行列と最小二乗法

<!--en:4e9c3e4a-->
未知数より方程式が多い連立一次方程式 $\mathbf{A}\mathbf{x}=\mathbf{b}$ は一般に厳密解を持たず、逆に未知数より方程式が少ない連立一次方程式は無数の解を持つ。どちらの状況でも、:numref:`sec_mdl-geometry-linear-algebraic-ops` で導入した最小二乗法の解を求める（ここでは目的ベクトルを $\mathbf{b}$ と書き、$\mathbf{y}$ の代わりに用いる）。

<!--en:95b8d1f1-->
さらに、複数の $\mathbf{x}$ が同じ最小残差を達成するときは、その中でノルムが最小のものを選ぶ。

<!--en:5f298f1e-->
まず最も単純な劣決定（underdetermined）の例として、2 つの未知数に対する 1 本の方程式 $x_1+x_2=2$ を考えよう。解は直線全体をなし、その上のどこでも残差は 0 である。直線上のどの点を選ぶべきだろうか。ここでは $\mathbf{A}=(1\;\;1)$ であり、非ゼロ特異値は $\sigma_1=\sqrt2$ の 1 つだけで、右特異ベクトルは $\mathbf{v}_1=(1,1)/\sqrt2$、左特異ベクトルは $u_1=1$ となる。この方程式は $\mathbf{x}$ の $\mathbf{v}_1$ 方向の座標を $\sqrt2$ に固定する一方、零空間方向 $\mathbf{v}_2=(1,-1)/\sqrt2$ に沿う座標は残差に一切影響しない。その自由な座標を 0 に設定すると、原点から直線へ下ろした垂線の足 $(1,1)$（ノルム $\sqrt2$）が得られ、他のあらゆる解 $(1+t,1-t)$ は真に大きい二乗ノルム $2+2t^2$ を持つ（:numref:`fig_mdl-la-min-norm`）。

<!--en:b8a0c51e-->
![方程式 1 本・未知数 2 個の最小二乗法。直線上のすべての点が $x_1+x_2=2$ を満たし、擬似逆行列は原点に最も近い点、すなわち垂線の足を返す。](../img/mdl-la-min-norm.svg)

<!--en:1f796974-->
一般の規則は、単にすべての非ゼロ特異値の逆数をとり、直交因子を転置するだけである。$\mathbf{A}$ の*ムーア・ペンローズ擬似逆行列*（Moore–Penrose pseudoinverse）は次式で定義される。

<!--en:63f78361-->
ここで $\boldsymbol{\Sigma}^{+}$ の形状は $n\times m$ であり、$\boldsymbol{\Sigma}$ を転置した形状になっている。幾何学的には、$\mathbf{A}^{+}$ は元に戻せるすべての伸縮を逆転させ、左零空間をゼロへと送る。せん断行列に対しては $\mathbf{T}^{+}=\mathbf{T}^{-1}=\left(\begin{smallmatrix}1&-1\\0&1\end{smallmatrix}\right)$ となり、$\mathbf{A}=(1\;\;1)$ に対しては $\mathbf{A}^{+}=\mathbf{v}_1\sigma_1^{-1}u_1=(\tfrac12,\tfrac12)^\top$ となるため、$\mathbf{A}^{+}\cdot2=(1,1)$ が得られる。

<!--en:81e3b766-->
**命題（最小ノルム最小二乗法）。** *$\|\mathbf{A}\mathbf{x}-\mathbf{b}\|$ のすべての最小点の中で、ノルムが最小となる一意の解は $\hat{\mathbf{x}}=\mathbf{A}^{+}\mathbf{b}$ である。*

<!--en:e0de77af-->
**証明。**$\mathbf{y}=\mathbf{V}^\top\mathbf{x}$、$\mathbf{c}=\mathbf{U}^\top\mathbf{b}$ とおいて特異基底へ変数変換する。$\mathbf{U}$ は直交行列でありユークリッドノルムを保つため、次式が成り立つ。

<!--en:0e7f86a1-->
特異基底へ回転することで、連立方程式は互いに独立なスカラー方程式へと分離される。第 1 の和は、$y_i=c_i/\sigma_i$（$i\le r$）とおくことで項ごとに最小化される。第 2 の和は $\mathbf{b}$ の列空間外の成分を表す固定定数であり、$\mathbf{x}$ をどう選んでも減らすことはできない。一方、座標 $y_i$（$i>r$）は残差に一切現れず、$\|\mathbf{x}\|=\|\mathbf{y}\|$ であるから、それらを 0 に設定することで最小ノルムの一意な最小点が得られる。この選択は $\mathbf{y}=\boldsymbol{\Sigma}^{+}\mathbf{c}$ であり、元の変数へ戻せば $\hat{\mathbf{x}}=\mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top\mathbf{b}=\mathbf{A}^{+}\mathbf{b}$ となる。$\blacksquare$

<!--en:35587565-->
特異基底で見ると、残差の最小化とノルムの最小化は完全に互いに素な座標群（$i\le r$ と $i>r$）を支配しており、両方の目的が同時に達成される。$\mathbf{A}$ が正方かつ可逆のときは、すべての $\sigma_i$ が正となり、$\mathbf{A}^{+}=\mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top=\mathbf{A}^{-1}$ となる。しかし同じ式は、最小二乗法の主な危険性もあらわにする。微小な $\sigma_i$ で割ることにより、$\mathbf{b}$ に含まれる $\mathbf{u}_i$ 方向のノイズが $1/\sigma_i$ 倍に増幅されてしまうのである。

<!--en:a18ef6a0-->
この不安定な方向を抑え込む標準的な処方箋が 2 つある。*リッジ回帰*（ridge regression）は $\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2+\lambda\|\mathbf{x}\|^2$（$\lambda>0$）を最小化し、$\mathbf{x}_\lambda=(\mathbf{A}^\top\mathbf{A}+\lambda\mathbf{I})^{-1}\mathbf{A}^\top\mathbf{b}$ を与える。一方、*打ち切り擬似逆行列*（truncated pseudoinverse）による解 $\mathbf{x}_k$ は、最初の $k$ 個の特異値項だけを $\mathbf{A}^{+}\mathbf{b}$ から残す。SVD を代入し、$\mathbf{A}^\top\mathbf{A}+\lambda\mathbf{I}=\mathbf{V}(\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}+\lambda\mathbf{I})\mathbf{V}^\top$ に注意すると、両方の解は見通しのよいフィルタ形式をとる。

<!--en:3b731726-->
どちらも正則化なしの擬似逆行列の項 $\mathbf{u}_i^\top\mathbf{b}/\sigma_i$ から出発する。リッジ回帰は第 $i$ 成分に滑らかなフィルタ係数 $\sigma_i^2/(\sigma_i^2+\lambda)$ を掛け、この係数は $\sigma_i^2\gg\lambda$ のとき 1 付近にとどまり、$\sigma_i^2\ll\lambda$ のとき 0 へと縮む。これに対し、打ち切りは 1 か 0 かの急峻な階段関数を適用する。たとえば $\lambda=1$ のとき、特異値 $3$、$1$、$0.3$ に対するフィルタ係数はそれぞれ $0.9$、$0.5$、$0.083$ となる。二乗損失で訓練される線形モデルにおいて、重み減衰（weight decay、:numref:`sec_weight_decay`）はまさにリッジ回帰そのものであり、これらのフィルタ係数はどの方向が減衰されるかを正確に示している。

<!--en:52285a07-->
**正規方程式。**古典的な教科書における最小二乗法の導出では、正規方程式（normal equations）$\mathbf{A}^\top\mathbf{A}\mathbf{x}=\mathbf{A}^\top\mathbf{b}$ を解く。任意の $\mathbf{A}$ に対してその解は最小二乗法の最小点となり（$\mathbf{A}^\top(\mathbf{A}\mathbf{x}-\mathbf{b})=\mathbf{0}$ は残差が列空間に直交することを表すため）、$\mathbf{A}$ が列フルランクであれば解は一意に定まる。しかし数値計算上、$\mathbf{A}^\top\mathbf{A}$ を直接作るのは得策ではない。条件数が二乗されてしまうからである（:numref:`subsec_mdl-condition-number`）。そのため現代のライブラリは、代わりに SVD や QR 分解を用いて最小二乗問題を解く。

<!--en:e3b138d3-->
過剰決定（overdetermined）系と劣決定系を 1 つずつ使って擬似逆行列を試してみよう。1 つ目は直線 $b\approx x_1+x_2 t$ を 4 つの観測値 $\mathbf{b}=(6,5,7,10)$（$t=1,2,3,4$）に当てはめる問題であり、2 つ目は先ほどの 1 本の方程式 $x_1+x_2=2$ である。

<!--en:b54fea9d-->
擬似逆行列と `np.linalg.lstsq` は当てはめ直線 $3.5+1.4t$（残差 $\sqrt{4.2}\approx2.049$）で一致し、1 本の方程式に対しては擬似逆行列は垂線の足 $(1,1)$ を返す。

<!--en:57bea96c-->
### 条件数

<!--en:c66e5bf6-->
実際の計算では、連立一次方程式の各成分はノイズを含む測定値や有限精度の丸めから生じる。計算された解を信頼する前に、データのわずかな変化が答えをどれほど変え得るかを知っておく必要がある。単純な $2\times2$ の系を見るだけで、その増幅がいかに劇的になり得るかがわかる。$\mathbf{K}\mathbf{x}=\mathbf{b}$ において $\mathbf{K}=\left(\begin{smallmatrix}1&1\\1&1.0001\end{smallmatrix}\right)$、$\mathbf{b}=(2,2.0001)$ としよう。厳密解は $\mathbf{x}=(1,1)$ である。ここで $b_2$ をほんの $10^{-4}$ だけ動かして $2.0002$ にすると、解は一気に $(0,2)$ まで飛び跳ねる。データにおける $10^5$ 分の 3 程度の摂動が、答えをまったく別物に変えてしまったのである。その理由は $\mathbf{K}$ の特異値（およそ $2.00005$ と $5\times10^{-5}$）を見ればわかる。

<!--en:086baca6-->
行列 $\mathbf{A}$（ランク $r$）の*条件数*とは、最大の特異値と最小の非ゼロ特異値の比である。

<!--en:7179f882-->
幾何学的には、$\kappa(\mathbf{A})$ は $\mathbf{A}$ が異なる方向をどれほど不均一に伸縮するかを測っている。$\mathbf{K}$ では約 $4\times10^4$ に達するのに対し、せん断行列ではわずか $\varphi^2\approx2.618$ である。$\mathbf{A}$ が対称正定値（positive definite）行列のときは特異値が固有値と一致するため $\kappa=\lambda_{\max}/\lambda_{\min}$ となり、:numref:`subsec_mdl-rayleigh` で見た二次関数のボウルに対する条件数が復元される。次の命題は、連立一次方程式を解く際の相対誤差に対する厳密な最悪増幅率が $\kappa(\mathbf{A})$ であることを示している。

<!--en:eccb035e-->
**命題（誤差増幅）。** *$\mathbf{A}$ を正方かつ可逆とし、$\mathbf{A}\mathbf{x}=\mathbf{b}$（$\mathbf{b}\ne\mathbf{0}$）とする。摂動 $\delta\mathbf{b}$ によって解が $\mathbf{x}+\delta\mathbf{x}$ に変化したとき、次式が成り立つ。*

<!--en:1622e42d-->
*さらに、任意の $\mathbf{A}$ に対して等号を達成する $\mathbf{b}$ と $\delta\mathbf{b}$ が存在する。*

<!--en:798f848f-->
**証明。**$\mathbf{A}\,\delta\mathbf{x}=\delta\mathbf{b}$ より $\delta\mathbf{x}=\mathbf{A}^{-1}\delta\mathbf{b}$ である。$\mathbf{A}^{-1}=\mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top$ の最大特異値は $1/\sigma_n$ であるから、:eqref:`eq_mdl-sigma1-variational` より $\|\delta\mathbf{x}\|\le\sigma_n^{-1}\|\delta\mathbf{b}\|$ が成り立つ。同様に、$\mathbf{b}=\mathbf{A}\mathbf{x}$ より $\|\mathbf{b}\|\le\sigma_1\|\mathbf{x}\|$ が得られる。これら 2 つの不等式を組み合わせると次式を得る。

<!--en:37472fe0-->
両方の評価で同時に等号が成り立つのは、$\mathbf{b}=\mathbf{u}_1$（単位長さの右辺に対して $\mathbf{x}=\sigma_1^{-1}\mathbf{v}_1$ を最小の長さにするとき）かつ $\delta\mathbf{b}=\varepsilon\mathbf{u}_n$（$\delta\mathbf{x}=\varepsilon\sigma_n^{-1}\mathbf{v}_n$ を大きさ $\varepsilon$ の摂動に対して最大の長さにするとき）である。$\blacksquare$

<!--en:d0aa2913-->
したがって、データの相対誤差は高々 $\kappa$ 倍に増幅される。$\kappa=10^4$ の場合、$\mathbf{b}$ の相対誤差 $10^{-8}$ が $\mathbf{x}$ では $10^{-4}$ まで膨らみ得るため、有効精度が 4 桁失われる。先ほどの $2\times2$ の例では、$\mathbf{b}$ の相対変化量は $10^{-4}/\|\mathbf{b}\|\approx3.5\times10^{-5}$ であるため、上界は $\mathbf{x}$ の相対変化として $1.41$ まで許容し、実際の変化量は $1.0$ となっている。これは摂動 $(0,10^{-4})$ が長さの $1/\sqrt2$ を最も増幅される方向 $\mathbf{u}_2\approx(1,-1)/\sqrt2$ に持つ一方、$\mathbf{b}$ がほぼ真っ直ぐ $\mathbf{u}_1$ の方向を向いているためである。

<!--en:e21deea8-->
理想的な極限にある直交行列では、すべての特異値が 1 に等しいため $\kappa=1$ となり、回転と鏡映は誤差を決して増幅しない。直交因子 $\mathbf{U}$ と $\mathbf{V}$ を用いた計算が数値的に安全である理由も、直交初期化された重み行列があらゆる入力ベクトルのノルムを保つ理由もここにある。その対極として、グラム行列を作ると条件数は二乗される。

<!--en:37296b6e-->
これは :eqref:`eq_mdl-svd-gram` がそのまま $\mathbf{A}^\top\mathbf{A}$ の SVD になっており、その特異値が $\sigma_i^2$ となるからである（任意の形状・任意のランクの行列 $\mathbf{A}$ で成り立つ）。せん断行列では、$\kappa$ は $\varphi^2\approx2.62$ から $\varphi^4\approx6.85$ へと跳ね上がる。以下のセルでは、$2\times2$ の例を検証するとともに、先ほどの直線当てはめ行列について $\kappa(\mathbf{A})$ と $\kappa(\mathbf{A}^\top\mathbf{A})$ を比較する。

<!--en:7cd5297c-->
出力から手計算の結果が確認できる。解は $(1,1)$ から $(0,2)$ へ飛び、$\kappa(\mathbf{K})\approx40{,}002$ である。控えめなサイズの $4\times2$ 直線当てはめ行列でさえ $\kappa=7.47$ であり、正規方程式を作ると条件数はすでに $\kappa^2=55.8$ まで膨れ上がってしまう。

<!--en:482971da-->
**条件性とボウルの形状。**大きな条件数は、最適化においても細長く伸びた等位集合として直接現れる。列フルランクの最小二乗目的関数 $\tfrac12\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2$ に対し、ヘッセ行列（Hessian）は $\mathbf{H}=\mathbf{A}^\top\mathbf{A}$ であり、その等位集合は半軸の長さが $1/\sigma_i$ に比例する $\mathbf{v}_i$ 方向の楕円体となる。したがって等高線の軸比は $\kappa(\mathbf{A})$ となる一方、ヘッセ行列自身の条件数は $\kappa(\mathbf{A})^2$ となる。このような細長い谷の上では、勾配降下法（gradient descent）は急峻な壁の間をジグザグに往復して谷底に沿う進行が遅くなり（:numref:`fig_mdl-la-condition`）、その収束率（convergence rate）は :numref:`sec_mdl-gradient-based-optimization` で導出するとおり $\kappa(\mathbf{H})$ によって決まる。

<!--en:4ca3db05-->
![良条件な二次関数のボウル（$\kappa(\mathbf{H})=1.3$、ほぼ円形の等高線、ほぼ直線的な経路）と悪条件なボウル（$\kappa(\mathbf{H})=18$、細長い谷を横切るジグザグの軌跡）における勾配降下法。パネルのタイトルはヘッセ行列の条件数 $\kappa(\mathbf{H})$ を示し、等高線はその平方根の倍率（第 2 パネルでは約 $4.2$ 倍）だけ細長く伸びている。](../img/mdl-la-condition.svg)

<!--en:9fe4db5b-->
## 深層学習における SVD

<!--en:cd0c9c2a-->
現代の大規模モデルの訓練で繰り返し現れる 3 つの問いは、重み行列の特異値へと直接帰着する。すなわち、ファインチューニング用アダプタにどれだけのパラメータが必要か、重み空間のあらゆる方向を対等に扱う更新をどう構成するか、そして単一の層が入力を最大どれだけ引き伸ばし得るか、という 3 つである。

<!--en:ee942c44-->
### 低ランク更新

<!--en:c3626c51-->
大規模な事前学習済みモデルの全パラメータをファインチューニング中に更新するのは、メモリと計算量の両面でコストが大きい。LoRA :cite:`Hu.Shen.Wallis.ea.2021` はこれを回避するため、事前学習済みの各重み行列 $\mathbf{W}\in\mathbb{R}^{m\times n}$ を固定し、ランク $k$ の補正項 $\Delta\mathbf{W}=\mathbf{B}\mathbf{A}$ のみを学習する。ここで $\mathbf{B}\in\mathbb{R}^{m\times k}$、$\mathbf{A}\in\mathbb{R}^{k\times n}$ である（ここでの $\mathbf{A}$ は原論文の記法に従い $k\times n$ の低次元射影を表しており、前節までの一般行列ではない）。このアダプタで訓練可能なパラメータ数は $k(m+n)$ 個となり、元の $mn$ 個から大幅に減る。$4096\times4096$ の層で $k=8$ とすれば、パラメータ数は $16{,}777{,}216$ 個から $8\cdot8192=65{,}536$ 個、すなわちわずか 0.39% へと減少する。:numref:`fig_mdl-la-lora` に示すように、固定された重みと低ランクバイパスは同じ入力に対して並列に作用し、それぞれの出力を足し合わせる。

<!--en:f61d605c-->
![1 層分の LoRA アーキテクチャ。事前学習済みの $\mathbf{W}$（全 $mn = 16.8$M 個のパラメータ、$m=n=4096$ の場合）は固定したままにし、並列バイパスがまず入力を $k=8$ 次元へ $\mathbf{A}$ で圧縮してから $\mathbf{B}$ で元の次元へ戻すため、訓練されるのは $k(m+n) \approx 65.5$K 個のパラメータ（約 $0.39\%$）のみである。2 つの経路は $\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$ と加算される。訓練後は $\mathbf{B}\mathbf{A}$ を $\mathbf{W}$ に統合できるため、デプロイ時の追加コストは一切かからない。](../img/mdl-la-lora.svg)

<!--en:dcbb3191-->
ここで Eckart–Young の定理は何を教えてくれ、何を教えてくれ*ない*のだろうか。もしフルファインチューニングによって特異値 $\sigma_1\ge\sigma_2\ge\cdots$ を持つ更新が得られるとすれば、いかなるランク $k$ アダプタもその特定の更新に対してスペクトルノルムで $\sigma_{k+1}$ 未満、フロベニウスノルムで $(\sum_{i>k}\sigma_i^2)^{1/2}$ 未満には近づけないことが Eckart–Young から保証される。しかし、勾配降下法が実際にどのアダプタを学習するかまでは規定し*ない*。LoRA が成功するのは、たとえフルファインチューニングの更新に対する最良ランク $k$ 近似でなくとも、*何らかの*低ランク更新が低い損失を達成するときである。多くの下流タスク（downstream task）においてそのような低ランク解が存在するという事実こそ、LoRA の背後にある経験的発見である。演習問題 :exref:`ex_mdl-svd-low-rank-weyl-perturbation` では、ランク $k$ アダプタが $\mathbf{W}$ の特異値を最大どれだけ動かし得るかを評価する。

<!--en:5b3d41ca-->
### 直交化された更新

<!--en:035c65b5-->
Muon オプティマイザ :cite:`Jordan.Jin.Boza.ea.2024` は、生のモメンタム行列 $\mathbf{M}$ の方向にそのまま重み行列を更新するのではなく、$\mathbf{M}$ の*極因子*（polar factor）、すなわち $\mathbf{M}$ の特異ベクトルを保ったまま非ゼロ特異値をすべて 1 に置き換えた行列の方向に更新する。オプティマイザ全体の動機と詳細は :numref:`sec_muon` で扱うこととし、ここではそれが依拠する 2 つの線形代数の道具、すなわち極因子とは何か、そして行列積だけを使ってそれをどう計算するかを与える。

<!--en:72c06407-->
**極分解。**正方行列 $\mathbf{A}$ に対し、SVD の中に $\mathbf{V}^\top\mathbf{V}=\mathbf{I}$ を挿入すると、直交因子と対称な伸縮の積へと分解（極分解、polar decomposition）される。

<!--en:2fea36e8-->
ここで $\mathbf{Q}=\mathbf{U}\mathbf{V}^\top$ は（直交行列の積として）直交行列であり、$\mathbf{P}=\mathbf{V}\boldsymbol{\Sigma}\mathbf{V}^\top$ は固有値 $\sigma_i$ を持つ対称半正定値行列である。このように、あらゆる正方行列は対称な伸縮に直交変換を続けたものへと分解される。これは複素数を極形式 $z=e^{i\theta}|z|$ と書くことの行列版にあたる。なお、$\mathbf{Q}$ は純粋な回転とは限らない。$\mathbf{A}$ が可逆のとき $\det\mathbf{Q}=\operatorname{sign}\det\mathbf{A}$ となるため、$\mathbf{Q}$ は $\det\mathbf{A}<0$ のとき常に鏡映を含む。せん断行列 $\mathbf{T}$ に対しては、:eqref:`eq_mdl-shear-svd` より $\mathbf{Q}=\mathbf{R}_{31.7^\circ}\mathbf{R}_{58.3^\circ}^\top=\mathbf{R}_{-26.6^\circ}=\tfrac{1}{\sqrt5}\left(\begin{smallmatrix}2&1\\-1&2\end{smallmatrix}\right)$（$\tan26.6^\circ=\tfrac12$ より）および $\mathbf{P}=\tfrac{1}{\sqrt5}\left(\begin{smallmatrix}2&1\\1&3\end{smallmatrix}\right)$ が得られる。積を計算すると、確かに $\mathbf{Q}\mathbf{P}=\tfrac15\left(\begin{smallmatrix}5&5\\0&5\end{smallmatrix}\right)=\mathbf{T}$ および $\mathbf{P}^2=\left(\begin{smallmatrix}1&1\\1&2\end{smallmatrix}\right)=\mathbf{T}^\top\mathbf{T}$ が成り立っている。

<!--en:fadf1e59-->
長方形行列 $\mathbf{M}$ に対しても、その薄い SVD を $\mathbf{M}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ とすれば、極因子はやはり $\mathbf{U}\mathbf{V}^\top$ である。これは $\mathbf{M}$ と同じ特異ベクトルを持ち、すべての特異値が 1 に等しいため、その列（$m\ge n$ のとき）または行（$m\le n$ のとき）が正規直交になる。$\mathbf{M}$ がランク落ちしている場合（$r<\min(m,n)$）は極因子は一意ではないが、コンパクト SVD から得られる $\mathbf{U}_r\mathbf{V}_r^\top$ が自然な標準的選択となる。演習問題 :exref:`ex_mdl-svd-low-rank-polar-procrustes` では、正方行列において $\mathbf{U}\mathbf{V}^\top$ がフロベニウスノルムの意味で $\mathbf{A}$ に最も近い直交行列でもあることを示す。

<!--en:5b1191e1-->
**行列の奇多項式。**訓練ステップごとにすべての重み行列の完全 SVD を計算するのは、遅すぎて現実的ではない。幸いなことに、行列積（matrix multiplication）だけを使えば、特異ベクトルには一切触れずに特異値だけを変換できる。

<!--en:1356a816-->
**命題（奇多項式は特異値に作用する）。** *$\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ を薄い SVD とし、$p(\sigma)=\sum_j c_j\sigma^{2j+1}$ を奇多項式とする。このとき次式が成り立つ。*

<!--en:7310fd38-->
**証明。**$\mathbf{X}^\top\mathbf{X}=\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top$ であるから、べき乗をとると $(\mathbf{X}^\top\mathbf{X})^j=\mathbf{V}\boldsymbol{\Sigma}^{2j}\mathbf{V}^\top$ となり、左から $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ を掛けると $\mathbf{X}(\mathbf{X}^\top\mathbf{X})^j=\mathbf{U}\boldsymbol{\Sigma}^{2j+1}\mathbf{V}^\top$ が得られる。これを $j$ について足し合わせれば証明が完了する。$\blacksquare$

<!--en:9301d807-->
したがって、任意の行列奇多項式は $\mathbf{U}$ と $\mathbf{V}$ をそのまま保ちながら、各特異値に対して独立にスカラー関数 $p$ を適用する。極因子 $\mathbf{U}\mathbf{V}^\top$ に到達するには、繰り返し適用することですべての正の特異値を 1 へと向かわせる奇多項式 $p$ を選べばよい。

<!--en:daa8383c-->
**Newton–Schulz。**古典的な *Newton–Schulz 反復* $\mathbf{X}\leftarrow\tfrac12(3\mathbf{X}-\mathbf{X}\mathbf{X}^\top\mathbf{X})$ は、すべての特異値に 3 次多項式 $p(\sigma)=\tfrac12(3\sigma-\sigma^3)$ を適用する。その不動点（fixed point）は $0$ と $\pm1$ である。区間 $(0,1]$ 上では、$p$ は 1 を行き過ぎることなくすべての値を上向きに押し上げる。なぜなら $p(\sigma)-\sigma=\tfrac12\sigma(1-\sigma^2)\ge0$ であり、かつ $p$ は最大値 $p(1)=1$ をとるからである。実際、$p$ は区間 $(0,\sqrt3)$ 全体を $(0,1]$ の中へ写すため、1 のこの*直接吸引領域*（immediate basin）に含まれるすべての初期値は 1 へと収束する一方、それより大きい初期値では不規則な挙動を示し得る（演習問題 :exref:`ex_mdl-svd-low-rank-newton-schulz-basin`）。初期行列を $\mathbf{X}_0=\mathbf{M}/\|\mathbf{M}\|_F$ とスケーリングすれば、すべての特異値が確実に $(0,1]$ の内部から出発する。$\|\mathbf{M}\|_F\ge\sigma_1(\mathbf{M})$ だからである。またゼロ特異値はゼロのままとどまるため（$p(0)=0$ より）、ランク落ちした $\mathbf{M}$ も綺麗に $\mathbf{U}_r\mathbf{V}_r^\top$ へと収束する。

<!--en:9fbfab92-->
これには何回の反復が必要だろうか。0 の近傍では $p(\sigma)\approx\tfrac32\sigma$ であるから、小さな特異値 $s$ は 1 ステップごとに $3/2$ 倍に成長し、オーダー 1 に達するまでにおよそ $\log_{3/2}(1/s)$ ステップを要する。1 の近傍では展開すると $p(1-e)=1-\tfrac32e^2+\tfrac12e^3$ となり、誤差が毎ステップおよそ二乗されるため、オーダー 1 から $10^{-6}$ まで縮むのに追加で約 4 ステップしかかからない。この 2 つの段階をつなぎ合わせると、およそ $\log_{3/2}(1/s_{\min})+4$ ステップと予測され、これは最小の正規化特異値 $s_{\min}=\sigma_{\min}(\mathbf{M})/\|\mathbf{M}\|_F$ によって決まる。モメンタムに見立てたランダムな $6\times4$ 行列でこの大まかな見積もりを検証してみよう。

<!--en:099bde0f-->
実際のステップ数は予測と 1 反復以内の差で一致し、反復結果は完全 SVD から求めた厳密な極因子から $10^{-5}$ 以内に収まっている。$s_{\min}$ は乱数抽出ごとに変動するためステップ数も変動し、ほぼランク落ちに近い $\mathbf{M}$ ではより多くの反復を要する。実用上の Muon では、3 次多項式の代わりに調整された 5 次多項式を用い、1 への厳密な収束と引き換えに高速化を図っている（:numref:`subsec_muon-newton-schulz`）。

<!--en:e28da1fa-->
### スペクトルの計測と制御

<!--en:09391158-->
3 つ目の問いは線形層が入力を最大どれだけ引き伸ばし得るかというものであり、最大特異値がその答えを直接与える。より広く見れば、特異値スペクトル全体から行列が本当に低ランクに近いかどうかがわかる。そのため現代の訓練レシピ（training recipe）では、完全 SVD のコストを払うことなく特異値を抑え、調べ、近似する。

<!--en:87012e6f-->
**スペクトル正規化。**:eqref:`eq_mdl-sigma1-variational` より、線形層 $\mathbf{x}\mapsto\mathbf{W}\mathbf{x}$ は $\|\mathbf{W}\mathbf{x}-\mathbf{W}\mathbf{x}'\|\le\sigma_1\|\mathbf{x}-\mathbf{x}'\|$ を満たし、$\mathbf{x}-\mathbf{x}'$ が $\mathbf{v}_1$ と同じ方向を向くときに等号が成り立つ。写像が入力の差を最大どれだけ増幅し得るかを抑える最小の定数を*リプシッツ定数*（Lipschitz constant）と呼ぶため、ここでのリプシッツ定数は単なる $\sigma_1=\|\mathbf{W}\|_2$ である。$\mathbf{W}$ を $\sigma_1$ で割って層の伸縮率の上限を 1 に抑える手法がスペクトル正規化（spectral normalization）であり、GAN の識別器（discriminator）の訓練を安定化させる :cite:`Miyato.Kataoka.Koyama.ea.2018`。その際、完全 SVD を計算する代わりに、$\sigma_1$ を $\mathbf{W}^\top\mathbf{W}$ に対するべき乗法（power iteration、:numref:`subsec_mdl-power-iteration`）で推定する。具体的には $\mathbf{u}\leftarrow\mathbf{W}\mathbf{v}$ と $\mathbf{v}\leftarrow\mathbf{W}^\top\mathbf{u}$ を半ステップごとに正規化しながら交互に繰り返す。この 1 往復で $\mathbf{W}^\top\mathbf{W}$ を 1 回掛けたことになり、$\|\mathbf{W}\mathbf{v}\|$ が $\sigma_1$ の推定値を与える。1 往復ごとに $\mathbf{v}$ は $\mathbf{v}_1$ へ引き寄せられ、誤差はおよそ $(\sigma_2/\sigma_1)^2$ 倍に縮小する。:citet:`Miyato.Kataoka.Koyama.ea.2018` は、前ステップのベクトルを初期値とするウォームスタートを行えば、訓練ステップあたり 1 往復で十分であることを実験的に見出した（なお畳み込み層では、カーネルを 2D 行列に変形することで畳み込みのスペクトルノルムの厳密値ではないものの近似値が得られる）。

<!--en:c7346e11-->
**スペクトルの計測。**先ほどの LoRA における 0.39% という数値は、ランク 8 で十分だと仮定していた。ある行列が本当に低ランクに近いかどうかは特異値がどれほど速く減衰するかに完全に依存するため、打ち切る前に計測すべきである。以下のセルでは、2 つの $400\times500$ 行列、すなわち構造をまったく持たないランダムなガウス行列と、ピクセル値の平均を差し引いた猫の写真を比較する。

<!--en:894bccc4-->
対比は鮮明である。ガウス行列の特異値はマルチェンコ・パスツール区間 $[\sqrt n-\sqrt m,\sqrt n+\sqrt m]\approx[2.4,42.4]$（:numref:`subsec_mdl-eckart-young`）全体に広がって急激な減衰を見せず、エネルギーの 95% を捉えるには 400 のうちおよそランク 270 が必要となるため、低ランクアダプタの方が元の行列よりかえって*多くの*パラメータを使ってしまう。対照的に、中心化した写真はランク 8 でエネルギーの 95% に達し、パラメータ数 4% 未満のアダプタに相当する。どちらも訓練されたニューラルネットワークの重みではないため、この比較はあくまで両極端を示す例にすぎない。LoRA において問題となるのはタスク固有の更新行列であり、そのスペクトルは仮定するのではなく実際に計測しなければならない。

<!--en:7bff6add-->
**ランダム化 SVD。**行列が大きすぎて直接分解できない場合、*ランダム化 SVD*（randomized SVD）:cite:`Halko.Martinsson.Tropp.2011` は高精度なランク $k$ 近似を、ごく少数の行列・ベクトル積（$\mathbf{A}$ との積）だけで計算する。その中核となる要素が*値域探索*（range finder）である。すなわち、$\mathbf{A}$ にランダムなガウス行列 $\boldsymbol{\Omega}$（$k$ より少し多い列を持つ）を掛け、$\mathbf{A}\boldsymbol{\Omega}$ の列を正規直交化して $\mathbf{Q}$ とし、はるかに小さい行列 $\mathbf{Q}^\top\mathbf{A}$ の SVD を計算する。特異値が減衰していれば、$\mathbf{A}$ の列のランダムな線形結合は常に上位の特異方向に支配されるため、$\mathbf{Q}$ は高い確率でそれらを捉える。演習問題 :exref:`ex_mdl-svd-low-rank-randomized-range-finder` で、減衰が遅いスペクトルに対するべき乗法による改良も含めてこのアルゴリズムを実装し、:numref:`sec_mdl-linear-algebra-further-reading` でランダム化スケッチングというより広い文脈の中に位置づける。

<!--en:023afb51-->
最後に、SVD の 3 つの限界を心に留めておこう。第 1 に、完全 SVD には $O(mn\min(m,n))$ 回の演算コストがかかる（$4096^3\approx7\times10^{10}$ 回にものぼり、単一の $4096\times4096$ 重み行列でも膨大である）。Newton–Schulz 反復やランダム化値域探索のような実用的アルゴリズムが行列積のみに頼るのはこのためである。第 2 に、低ランク性はデータが持つ性質であって無条件に保証されるものではない。このことは直前のガウス行列が示したとおりである。第 3 に、SVD が捉えるのは*線形*な構造のみである。平面上の円周上に並ぶ点群は、円が本質的に 1 次元であるにもかかわらずランク 2 のデータ行列をなす。曲がった関数や非線形な構造を理解するには導関数（derivative）が必要であり、次章ではそのテーマへと進む。

<!--en:30ac03ff-->
## まとめ

<!--en:4eca3bac-->
形状にかかわらず任意の実行列 $\mathbf{A}$ には、入力の正規直交基底 $\{\mathbf{v}_i\}$ と出力の正規直交基底 $\{\mathbf{u}_i\}$ が存在し、その間でこの行列は非負の伸縮率の単純な列 $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$（$\sigma_1\ge\sigma_2\ge\cdots\ge0$）として作用する。この分解は、グラム行列 $\mathbf{A}^\top\mathbf{A}$ にスペクトル定理を適用することで $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$ として得られる。

<!--en:4d7b1626-->
順序付けられたこのたった 1 つの特異値のリストから、5 つの中心的な量が一度に導かれる。すなわち、ランク（正の $\sigma_i$ の個数）、スペクトルノルムとフロベニウスノルム（$\|\mathbf{A}\|_2=\sigma_1$ と $\|\mathbf{A}\|_F^2=\sum_i\sigma_i^2$）、打ち切り SVD $\mathbf{A}_k$ が与える最良ランク $k$ 近似（Eckart–Young の定理であり、中心化データに対しては主成分分析となる）、最小ノルム最小二乗解を返すムーア・ペンローズ擬似逆行列 $\mathbf{A}^{+}$、そして相対誤差の増幅率を抑える条件数 $\kappa=\sigma_1/\sigma_r$ である。深層学習においても、同じ特異値スペクトルがファインチューニング用の低ランクアダプタのサイズを決め、Muon が用いる極因子を定義し、スペクトル正規化が制御するリプシッツ定数を与える。

<!--en:2294243a-->
## 演習問題

<!--en:632be335-->
1. **ノルムと最大伸縮率。**本文ではスペクトル定理から :eqref:`eq_mdl-sigma1-variational` を導いたが、最大伸縮率だけから逆に SVD を構成することもできる。
    1. $\sigma_1=\max_{\|\mathbf{x}\|=1}\|\mathbf{A}\mathbf{x}\|$ とおき、$\sigma_1>0$ と仮定して、スペクトル定理を使わずに SVD を構成せよ。(i) ある単位ベクトル $\mathbf{v}_1$ で最大値が達成されることを示し、$\mathbf{u}_1=\mathbf{A}\mathbf{v}_1/\sigma_1$ とおけ。(ii) 単位ベクトル $\mathbf{w}\perp\mathbf{v}_1$ に対し、関数 $t\mapsto\|\mathbf{A}(\mathbf{v}_1+t\mathbf{w})\|^2/(1+t^2)$ は $t=0$ で最大値をとるため、そこでの導関数は 0 になる。ここから $\mathbf{u}_1^\top\mathbf{A}\mathbf{w}=0$、すなわち $\mathbf{A}$ が $\mathbf{v}_1^\perp$ を $\mathbf{u}_1^\perp$ の中へ写すことを導け。(iii) $\mathbf{A}$ の $\mathbf{v}_1^\perp$ への制限に対して帰納的に繰り返せ。単位球面のコンパクト性はどこで使われたか。
    1. $\|\mathbf{A}\|_2 \le \|\mathbf{A}\|_F \le \sqrt{r}\,\|\mathbf{A}\|_2$（ただし $r = \operatorname{rank}\mathbf{A}$）を示せ。左右それぞれの不等号で等号を達成するのはどのような行列か。
    1. 比 $\|\mathbf{A}\|_F^2/\|\mathbf{A}\|_2^2$ は $\mathbf{A}$ の*安定ランク*と呼ばれ、(b) より 1 と $\operatorname{rank}\mathbf{A}$ の間にある。減衰スペクトル $\sigma_i=e^{-i/12}$（$i=0,\ldots,255$、幾何級数）に対して安定ランクを閉じた式で求め、このスペクトルのエネルギーの 95% を保持するにはランク 18 が必要であることを示せ。次に、$400\times500$ のガウス行列（:numref:`subsec_mdl-spectrum-control`）の安定ランクを $\|\mathbf{G}\|_F^2\approx mn$ と $\sigma_1\approx\sqrt m+\sqrt n$ から見積もり、エネルギーの 95% を保持するランク（約 270）と比較せよ。なぜ安定ランクは通常のランクよりはるかに小さくなり得るのか。
   :exlabel:`ex_mdl-svd-low-rank-norms-singular-values`

<!--en:acf48602-->
1. **Weyl 摂動限界。**更新 $\boldsymbol{\Delta}$、たとえばランク $k$ の LoRA 更新 $\mathbf{B}\mathbf{A}$（原論文の記号）は、重み行列 $\mathbf{W}$ の特異値を変化させる。Weyl の不等式 :eqref:`eq_mdl-weyl-singular`（:numref:`subsec_mdl-eckart-young` で証明）は、その変化幅の上限を与える。
    1. $j = 1$ を :eqref:`eq_mdl-weyl-singular` に代入し、不等式を両方向に適用することで、$|\sigma_i(\mathbf{W}+\boldsymbol{\Delta})-\sigma_i(\mathbf{W})|\le\sigma_1(\boldsymbol{\Delta})$ がすべての $i$ で成り立つことを導け。
    1. $\mathbf{W} = \operatorname{diag}(1, 2)$、$\boldsymbol{\Delta} = 0.1\,\mathbf{1}\mathbf{1}^\top$ とする。$\mathbf{W}+\boldsymbol{\Delta}$ の特異値を計算し、それぞれの変化量を $\sigma_1(\boldsymbol{\Delta})$ と比較せよ。
    1. 次に $\operatorname{rank}\boldsymbol{\Delta}\le k$ とする。:numref:`subsec_mdl-eckart-young` のフロベニウスノルムの証明では、:eqref:`eq_mdl-weyl-singular` を $j = k+1$ としてランクが高々 $k$ の行列に適用した。同じ手順を $\boldsymbol{\Delta}$ に適用して、交錯不等式 $\sigma_{i+k}(\mathbf{W})\le\sigma_i(\mathbf{W}+\boldsymbol{\Delta})$ および $\sigma_{i+k}(\mathbf{W}+\boldsymbol{\Delta})\le\sigma_i(\mathbf{W})$ を導け。これらを用いて、$\mathbf{W}+\boldsymbol{\Delta}$ と $\mathbf{W}$ のランクの差が高々 $k$ であること、さらに高々 $k$ 個の特異値（$\mathbf{W}+\boldsymbol{\Delta}$ のもの）しか $\sigma_1(\mathbf{W})$ を超えないことを、$\sigma_1(\boldsymbol{\Delta})$ がどれほど大きくても成り立つ形で示せ。なぜ (a) の評価からはこれらの結論が得られないのか。
   :exlabel:`ex_mdl-svd-low-rank-weyl-perturbation`

<!--en:e054d0e4-->
1. [code] **PCA における中心化と Fashion-MNIST。**$\mathbf{X}\in\mathbb{R}^{n\times d}$ の各行をデータ点とし、平均行ベクトルを $\bar{\mathbf{x}}$、共分散を $\mathbf{C}$ とする（:numref:`subsec_mdl-pca` の主成分分析と同じ設定）。
    1. $\tfrac1n\mathbf{X}^\top\mathbf{X} = \mathbf{C}+\bar{\mathbf{x}}\bar{\mathbf{x}}^\top$ を示せ。
    1. :numref:`subsec_mdl-pca` と同様の点群（300 点、標準偏差 3 と 0.8、回転角 $\pi/5$ の軸に沿う分布）を生成し、すべての点を $c\,(1,-1)/\sqrt2$（$c\in\{0, 1, 3, 10, 30\}$）だけ平行移動させよ。各 $c$ について、*中心化していない*データの最大右特異ベクトル、主軸 $(\cos\frac\pi5, \sin\frac\pi5)$ とのなす角、および比 $\sigma_1^2/\sum_i\sigma_i^2$ を計算せよ。
    1. (a) を用いて、最大特異ベクトルが主軸から平均の方向へと向きを変えるオフセットの大きさを $\mathbf{C}$ の固有値から予測せよ。
    1. 再び (a) を用いて比 $\sigma_1^2/\sum_i\sigma_i^2$ を大きな $c$ に対して近似し、$c\to\infty$ における極限を求めよ。$c = 10$ では、この比は中心化後の説明分散比に近い値となる。この一致はデータの性質を反映したものか、それとも選んだオフセットによるものか。
    1. `d2l.FashionMNIST` を用いて（:numref:`sec_mdl-geometry-linear-algebraic-ops` と同様に）Fashion-MNIST の T シャツとズボンのテスト画像を読み込み、長さ 784 のベクトルへ平坦化し、中心化した上で上位 2 本の右特異ベクトルを計算せよ。最初の 2 つの主成分が張る平面上に 2 クラスをプロットせよ。これら 2 方向は分散のどれだけの割合を占めるか。さらに $\mathbf{v}_1$ とクラス平均の差 $\bar{\mathbf{x}}_{\text{trouser}}-\bar{\mathbf{x}}_{\text{t-shirt}}$ のなす角を計算し、2 つの方向が近いものの厳密には一致しない理由を説明せよ。最後に、1 枚の T シャツ画像を最初の 5 個、20 個、50 個の主成分から再構成し、それぞれの相対誤差を報告せよ。
   :exlabel:`ex_mdl-svd-low-rank-centering-pca`

<!--en:81a78a47-->
1. [code] **リッジなし極限と勾配降下法。**過剰パラメータ化された線形モデルは方程式より多くのパラメータを持つため、最小二乗法は無数の最小点を持つ。小問 (a) は任意の形状・ランクで成り立ち、(b) と (c) では $\mathbf{A}\in\mathbb{R}^{m\times n}$（$m < n$）を用いる。
    1. フィルタ形式 :eqref:`eq_mdl-ridge-filter` を用いて、$(\mathbf{A}^\top\mathbf{A}+\lambda\mathbf{I})^{-1}\mathbf{A}^\top\to\mathbf{A}^{+}$（$\lambda\to0^{+}$）が任意の形状・ランクの行列 $\mathbf{A}$ に対して成り立つことを示せ。なぜ $(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ は $m<n$ のとき存在しないのか。
    1. $\tfrac12\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2$ に対するステップサイズ $0 < \eta < 2/\sigma_1^2$ の勾配降下法は、$\mathbf{x}_{t+1} = \mathbf{x}_t-\eta\,\mathbf{A}^\top(\mathbf{A}\mathbf{x}_t-\mathbf{b})$ と更新する。$\mathbf{x}_t-\mathbf{x}_0$ が常に $\mathbf{A}$ の行空間内にとどまること、および $\mathbf{x}_t$ が $\mathbf{A}^{+}\mathbf{b}+(\mathbf{I}-\mathbf{A}^{+}\mathbf{A})\mathbf{x}_0$ へ収束することを示せ。初期化 $\mathbf{x}_0 = \mathbf{0}$ はどの解を選び出すか。
    1. $20\times50$ のガウス行列 $\mathbf{A}$ とガウスベクトル $\mathbf{b}$ に対して (a) と (b) を検証せよ。$\mathbf{x}_0 = \mathbf{0}$ とランダムな $\mathbf{x}_0$ の両方から勾配降下法を実行し、それぞれの収束先とそのノルムを $\mathbf{A}^{+}\mathbf{b}$ と比較せよ。
   :exlabel:`ex_mdl-svd-low-rank-ridgeless-gradient-descent`

<!--en:e3255a09-->
1. [code] **Newton–Schulz の吸引領域。**Newton–Schulz 反復 $\mathbf{X}\leftarrow\tfrac12(3\mathbf{X}-\mathbf{X}\mathbf{X}^\top\mathbf{X})$（:numref:`subsec_mdl-orthogonalized-updates`）は $\mathbf{X} = \mathbf{M}/\|\mathbf{M}\|_F$ から出発する。
    1. $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ を薄い SVD とする。:eqref:`eq_mdl-odd-polynomial` を使わずに直接計算することで、1 ステップ後の結果が $\mathbf{U}\,p(\boldsymbol{\Sigma})\,\mathbf{V}^\top$（各対角成分に $p(\sigma) = \tfrac12(3\sigma-\sigma^3)$ を適用したもの）になることを示せ。
    1. $(0,\sqrt3)$ が 1 の直接吸引領域であることを示せ。すなわち、$p$ がそれを $(0, 1]$ の中へ写し、反復列 $p(p(\cdots p(\sigma_0)))$ がすべての $\sigma_0\in(0,\sqrt3)$ に対して 1 へ収束することを示せ。また、$\sqrt3$ を超える初期値の中にも 1 に到達するもの（$2.15$ を試せ）と到達しないもの（2 を試せ）があることを示せ。さらに $p(1-e)$ を $e$ のべき乗で展開せよ。1 までの距離が小さくなった後は、どれほどの速さで縮小するか。
    1. $6\times4$ のガウス行列 $\mathbf{M}$ に対し、$\|\mathbf{X}-\mathbf{U}\mathbf{V}^\top\|_F < 10^{-6}$ になるまでのステップ数を、初期値 $\mathbf{M}/c$ で $c = \|\mathbf{M}\|_F$ とした場合と $c = \sigma_1(\mathbf{M})$ とした場合のそれぞれで数えよ。両方のステップ数を $\log_{3/2}(1/s_{\min})+4$（$s_{\min}$ はスケーリング後の最小特異値）と比較せよ。1 ステップ節約できるにもかかわらず、Muon が $\|\mathbf{M}\|_F$ で割るのはなぜか（$\sigma_1(\mathbf{M})$ ではなく）。
    1. $\mathbf{M}/c$ を初期値として、$c = \sigma_1(\mathbf{M})/2$ の場合と $c = \sigma_1(\mathbf{M})/2.5$ の場合で反復を開始せよ。それぞれ何に収束するか。(b) の多項式 $p$ を用いて両方の結果を説明せよ。
   :exlabel:`ex_mdl-svd-low-rank-newton-schulz-basin`

<!--en:62944835-->
1. [code] **極因子と Procrustes 問題。** ● 極分解 :eqref:`eq_mdl-polar` は正方行列を $\mathbf{A} = \mathbf{Q}\mathbf{P}$（$\mathbf{Q}$ は直交行列、$\mathbf{P}$ は対称半正定値行列）と表す。
    1. $\mathbf{P} = (\mathbf{A}^\top\mathbf{A})^{1/2}$ が $\mathbf{A}$ によって一意に定まること、さらに $\mathbf{Q}$ も $\mathbf{A}$ によって（$\mathbf{A}$ が可逆のとき）一意に定まることを示せ。また $\operatorname{diag}(1, 0)$ の異なる 2 つの極分解を挙げよ。
    1. フロベニウスノルムにおいて $\mathbf{Q} = \mathbf{U}\mathbf{V}^\top$ が $\mathbf{A}$ に最も近い直交行列であることを示せ。その際、問題を $\operatorname{tr}(\mathbf{R}^\top\mathbf{A})$ の直交行列 $\mathbf{R}$ に関する最大化へと帰着させよ。
    1. 回転だけ異なる 2 つの埋め込み空間は、対応点の組 $\mathbf{X},\mathbf{Y}\in\mathbb{R}^{d\times N}$ から $\|\mathbf{R}\mathbf{X}-\mathbf{Y}\|_F$ を直交行列 $\mathbf{R}$ 上で最小化することで位置合わせできる。(b) を用いて、最小点が $\mathbf{Y}\mathbf{X}^\top$ の極因子であることを示せ。$d = 50$、$N = 1000$ とし、ランダムな直交行列 $\mathbf{R}$（ガウス行列の QR 分解の $\mathbf{Q}$ 因子）、ガウス行列 $\mathbf{X}$、および $\mathbf{Y} = \mathbf{R}\mathbf{X}+\mathbf{E}$（$\mathbf{E}$ の成分は独立同分布な $\mathcal{N}(0, s^2)$、$s\in\{0, 0.1, 1, 3\}$）でこれを検証せよ。$\|\hat{\mathbf{R}}-\mathbf{R}\|_F/\sqrt d$ を、この推定量と制約なし最小二乗推定量 $\mathbf{Y}\mathbf{X}^{+}$ について比較せよ。
   :exlabel:`ex_mdl-svd-low-rank-polar-procrustes`

<!--en:e4bbc2a2-->
1. [code] **ランダム化値域探索。** ● :numref:`subsec_mdl-spectrum-control` のランダム化 SVD は、$\mathbf{A}$ にガウス行列 $\boldsymbol\Omega\in\mathbb{R}^{n\times(k+p)}$ を掛け、$\mathbf{Y} = \mathbf{A}\boldsymbol\Omega$ を正規直交化して $\mathbf{Q}$ とし、$\mathbf{Q}^\top\mathbf{A}$ の SVD をランク $k$ で打ち切る。
    1. これを実装せよ。$\mathbf{A}\in\mathbb{R}^{1000\times500}$ を $\mathbf{U}\operatorname{diag}(\sigma_1,\ldots,\sigma_{500})\mathbf{V}^\top$（$\mathbf{U}$ と $\mathbf{V}$ はランダムな列正規直交行列）として構成し、速い減衰 $\sigma_i = e^{-i/5}$ の場合と遅い減衰 $\sigma_i = 1/i$ の場合の 2 通りを用意せよ。$k = 20$、$p\in\{0, 5, 10\}$ に対し、ランク $k$ の近似結果のスペクトルノルム誤差を $\sigma_{k+1}$（:eqref:`eq_mdl-eckart-young` の最小誤差）で割り、10 回の $\boldsymbol\Omega$ の抽出にわたって平均せよ。なぜ片方のスペクトルではオーバーサンプリングだけで十分なのに、もう片方では不十分なのか。
    1. $\mathbf{Y}$ を $(\mathbf{A}\mathbf{A}^\top)^q\mathbf{A}\boldsymbol\Omega$（$q\in\{1, 2\}$）に置き換えよ。各スペクトルで誤差比はどう変化するか。$q$ の効果を $(\mathbf{A}\mathbf{A}^\top)^q\mathbf{A}$ の特異値を用いて説明せよ。また、各変種には $\mathbf{A}$ または $\mathbf{A}^\top$ との積が何回必要か（計算コストを答えよ）。
   :exlabel:`ex_mdl-svd-low-rank-randomized-range-finder`

<!--en:0cd07d81-->
1. [code] **最適ハード閾値。** ●● 閾値 $\tfrac{4}{\sqrt3}\sqrt n\,\sigma_{\text{noise}}$（:citet:`Gavish.Donoho.2014`）は、単一の信号成分の振る舞いから導かれる。$\mathbf{A} = \theta\,\mathbf{u}\mathbf{v}^\top+\mathbf{N}$ を $n\times n$ 行列とし、$\mathbf{u},\mathbf{v}$ を単位ベクトル、$\theta > 0$ とし、$\mathbf{N}$ の各成分は独立同分布なガウス分布 $\mathcal{N}(0, \sigma_{\text{noise}}^2)$ に従うとする。$x = \theta/(\sqrt n\,\sigma_{\text{noise}})$ とおき、$(\sigma_1, \hat{\mathbf{u}}, \hat{\mathbf{v}})$ を $\mathbf{A}$ の最大特異三つ組とする（ただし対の符号は $\mathbf{u}^\top\hat{\mathbf{u}}\ge0$ となるように選ぶ）。$n\to\infty$ のとき、:citet:`Benaych-Georges.Nadakuditi.2012` によれば、$\sigma_1$ はバルク上端 $2\sqrt n\,\sigma_{\text{noise}}$ へ収束して特異ベクトルは $\mathbf{u}$、$\mathbf{v}$ と直交するようになり（$x\le1$ のとき）、また $x > 1$ のときは次式が成り立つ。

   $$\frac{\sigma_1}{\sqrt n\,\sigma_{\text{noise}}}\to x+\frac1x,\qquad
   (\mathbf{u}^\top\hat{\mathbf{u}})^2\to1-\frac1{x^2},\qquad
   (\mathbf{v}^\top\hat{\mathbf{v}})^2\to1-\frac1{x^2}.$$

   $x = 1$ におけるこの相転移は、演習問題 :exref:`ex_mdl-eigendecomposition-spiked-covariance`（:numref:`sec_mdl-eigendecompositions`）における $\beta = \sqrt c$ での固有値の相転移の特異値版にあたる。

    1. 信号 $\theta\,\mathbf{u}\mathbf{v}^\top$ に対する 2 つの推定量として、ランク 1 打ち切り $\sigma_1\hat{\mathbf{u}}\hat{\mathbf{v}}^\top$ とゼロ行列を考える。それぞれの二乗フロベニウス誤差を $\theta$、$\sigma_1$、$\mathbf{u}^\top\hat{\mathbf{u}}$、$\mathbf{v}^\top\hat{\mathbf{v}}$ を用いて表せ。
    1. 上の極限を用いて、打ち切りの方が漸近誤差が小さくなるのはちょうど $x > \sqrt3$ のときであり、$x = \sqrt3$ における特異値が $\tfrac{4}{\sqrt3}\sqrt n\,\sigma_{\text{noise}}$ であることを示せ。ノイズの上端を超える特異値を生じるにもかかわらず切り捨てるべき信号強度はどの範囲か。
    1. $n = 500$、$\sigma_{\text{noise}} = 1/\sqrt n$ とし、$x\in\{0.5, 1.2, 1.5, \sqrt3, 2, 3\}$ のそれぞれで 10 回ずつ乱数抽出して (a) と (b) を検証せよ。$\sigma_1$、余弦の二乗、および両方の誤差をそれぞれの極限値と比較せよ。
   :exlabel:`ex_mdl-svd-low-rank-optimal-hard-threshold`

<!--en:a4371b20-->
[ディスカッション](https://d2l.discourse.group/t/36826)

<!--en:534fc39e-->
特異値分解と低ランク近似<br>**2 つの正規直交基底・4 つの部分空間・Eckart–Young と PCA・擬似逆行列と条件数・LoRA と Muon**

<!--en:6349d81d-->
::: {.slide title="なぜ深層学習では単一の基底では不十分か"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:c1455a7b-->
- **長方形の層とデータ表：**重み行列 $\mathbf{W} \in \mathbb{R}^{m \times n}$ やデータ表 $\mathbf{X} \in \mathbb{R}^{n \times d}$ は正方行列ではないため、固有値が存在しない

<!--en:6cda47d3-->
- **2 つの正規直交基底（$\mathbf{V}$ と $\mathbf{U}$）：**入力基底 $\mathbf{V}$ と出力基底 $\mathbf{U}$ を対にすることで、*あらゆる*行列が非負の伸縮 $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$ に帰着する

<!--en:e9c0303f-->
- **低ランク圧縮：**$k = 20$ 個のランク 1 成分を残すだけで、$400 \times 500$ の猫の写真を $9\%$ の記憶容量と $8\%$ の相対誤差で表現できる

<!--en:5b7ae00f-->
::: {.slide title="SVD：回転・軸方向の伸縮・回転"}

<!--en:c638361f-->
[行列分解]{.kicker}

<!--en:5c02b670-->
- **定理（$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$）：**任意の実 $m \times n$ 行列は、直交行列 $\mathbf{U} \in \mathbb{R}^{m \times m}$（左特異ベクトル $\mathbf{u}_i$）、対角行列 $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$（$\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$）、直交行列 $\mathbf{V} \in \mathbb{R}^{n \times n}$（右特異ベクトル $\mathbf{v}_i$）に分解される

<!--en:ef55f139-->
- **方向ごとの作用（$\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$）：**$\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\mathbf{x}$ を右から左へ読むと、入力フレーム $\{\mathbf{v}_i\}$ を座標軸へ回転させ（$\mathbf{V}^\top$）、軸 $i$ を $\sigma_i$ 倍に伸縮し（$\boldsymbol{\Sigma}$）、出力フレーム $\{\mathbf{u}_i\}$ へ回転させる（$\mathbf{U}$）

<!--en:2721299c-->
::: {.slide title="SVD の完全・薄い・コンパクト・ダイアド形式"}

<!--en:c5c8a7c2-->
[SVD の表現形式]{.kicker}

<!--en:0279e1f4-->
- **完全 SVD と薄い SVD：**完全 SVD は正方直交行列 $\mathbf{U}_{m\times m}, \mathbf{V}_{n\times n}$ を保持し、薄い SVD は $p = \min(m, n)$ 列を保持する（`full_matrices=False`）

<!--en:a8c3cac7-->
- **コンパクト SVD（$r = \operatorname{rank}\mathbf{A}$）：**正の伸縮率 $\sigma_1 \ge \cdots \ge \sigma_r > 0$ のみを保持

<!--en:fb35f186-->
- **ダイアド和：**$\mathbf{A} = \sum_{i=1}^r \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ はランク 1 の外積を重い順に並べる

<!--en:33106fe2-->
::: {.slide title="存在証明：AᵀA に対するスペクトル定理"}

<!--en:04d85825-->
[なぜ SVD は常に存在するか]{.kicker}

<!--en:1fbc43da-->
- **1. 半正定値グラム行列 $\mathbf{A}^\top\mathbf{A}$ の対角化：**正規直交固有ベクトル $\mathbf{v}_1, \ldots, \mathbf{v}_n$ は固有値 $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$ を持つ

<!--en:0a5b5b14-->
- **2. 出力空間への写像（$\sigma_i = \sqrt{\lambda_i}$）：**$i \le r$ では $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$ とおき、$i > r$ では $\|\mathbf{A}\mathbf{v}_i\|^2 = 0 \implies \mathbf{A}\mathbf{v}_i = \mathbf{0}$ となる

<!--en:3ebde1a3-->
- **3. 自動的な正規直交性：**$\mathbf{u}_i^\top\mathbf{u}_j = \frac{\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j}{\sigma_i\sigma_j} = \delta_{ij}$

<!--en:ac5ce1c0-->
::: {.slide title="計算例：欠陥を持つせん断行列 T の SVD"}

<!--en:3edeee5b-->
[手計算と数値計算による SVD]{.kicker}

<!--en:1d57f2ef-->
- **黄金比の伸縮：**$\mathbf{T} = \left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ に対し、$\mathbf{T}^\top\mathbf{T} = \left(\begin{smallmatrix}1&1\\1&2\end{smallmatrix}\right)$ は $\operatorname{tr} = 3, \det = 1$、固有値 $\lambda = (3 \pm \sqrt{5})/2 = \varphi^2, \varphi^{-2}$（$\varphi = (1+\sqrt{5})/2 \approx 1.618$）を持ち、$\sigma_1 = \varphi \approx 1.618$ と $\sigma_2 = 1/\varphi \approx 0.618$ を与える

<!--en:ae7a0261-->
- **入力角と出力角：**$(\mathbf{T}^\top\mathbf{T} - \varphi^2\mathbf{I})\mathbf{v}_1 = \mathbf{0}$ を解くと $\mathbf{v}_1 \propto (1, \varphi)$（$58.3^\circ$）および $\mathbf{u}_1 = \mathbf{T}\mathbf{v}_1/\varphi \propto (\varphi, 1)$（$31.7^\circ$）が得られ、$\mathbf{T} = \mathbf{R}_{31.7^\circ}\operatorname{diag}(\varphi, 1/\varphi)\mathbf{R}_{58.3^\circ}^\top$ となる

<!--en:23abfc76-->
::: {.slide title="計算例：欠陥を持つせん断行列 T の SVD（続き）"}

<!--en:3edeee5b-2-->
[手計算と数値計算による SVD]{.kicker}

<!--en:5f55c37e-->
::: {.slide title="コードによる AᵀA からの長方形行列の SVD 構成"}

<!--en:3edeee5b-3-->
[手計算と数値計算による SVD]{.kicker}

<!--en:d138db82-->
$3 \times 2$ 行列（$\mathbf{A}^\top\mathbf{A} = \left(\begin{smallmatrix}10&6\\6&14\end{smallmatrix}\right)$、$\lambda = 12 \pm \sqrt{40}$）で証明の手順を実行すると、同時符号反転 $(\mathbf{u}_i, \mathbf{v}_i) \mapsto (-\mathbf{u}_i, -\mathbf{v}_i)$ を除いて `np.linalg.svd` が再現される

<!--en:43def4c0-->
**ライブラリが Golub–Kahan 二重対角化を使う理由：**$\mathbf{A}^\top\mathbf{A}$ は存在証明を与えるが、浮動小数点で $\mathbf{A}^\top\mathbf{A}$ を作ると条件数が二乗されて $\kappa(\mathbf{A})^2$ となり、小さな特異値が失われる。実用的な SVD ソルバは $\mathbf{A}$ に直接直交変換を適用する。

<!--en:6786d638-->
::: {.slide title="最大伸縮率・ノルム・固有値との関係"}

<!--en:1d0451bf-->
[特異値と固有値]{.kicker}

<!--en:895624a0-->
- **特異値で表されるノルム：**$\|\mathbf{A}\|_2 = \max_{\|\mathbf{x}\|=1}\|\mathbf{A}\mathbf{x}\| = \sigma_1$（$\mathbf{v}_1$ で達成）および $\|\mathbf{A}\|_F^2 = \sum_{i=1}^r \sigma_i^2$

<!--en:87b68b3b-->
- **特異値 $\neq$ 固有値の絶対値：**例題行列 $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ では $|\lambda_{1,2}| = \sqrt{5} \approx 2.236$ であるのに対し、$\sigma_1 \approx 3.618, \sigma_2 \approx 1.382$

<!--en:8800d6ef-->
- **体積の恒等式：**$|\det\mathbf{A}| = \prod_i \sigma_i = \prod_i |\lambda_i| = 5$

<!--en:e5633e34-->
::: {.slide title="SVD から得られる 4 つの基本部分空間"}

<!--en:8d124865-->
[部分空間の幾何学]{.kicker}

<!--en:e24b019b-->
- **厳密なランク（$r = \#\{\sigma_i > 0\}$）：**直交行列は次元を保つため、$\operatorname{rank}\mathbf{A} = \operatorname{rank}\boldsymbol{\Sigma} = r$

<!--en:65b49f33-->
- **4 つの部分空間すべての正規直交基底：**
  - *行空間* $\operatorname{col}(\mathbf{A}^\top) = \operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_r\} \subseteq \mathbb{R}^n$
  - *零空間* $\ker(\mathbf{A}) = \operatorname{span}\{\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n\} \subseteq \mathbb{R}^n$
  - *列空間* $\operatorname{col}(\mathbf{A}) = \operatorname{span}\{\mathbf{u}_1, \ldots, \mathbf{u}_r\} \subseteq \mathbb{R}^m$
  - *左零空間* $\ker(\mathbf{A}^\top) = \operatorname{span}\{\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m\} \subseteq \mathbb{R}^m$

<!--en:bcdaf05a-->
- **行空間上の全単射：**$\mathbf{A}$ は $\mathbf{v}_i \mapsto \sigma_i\mathbf{u}_i$（$i \le r$）と 1 対 1 に写し、$\ker\mathbf{A}$ をゼロに潰す

<!--en:ab12972a-->
::: {.slide title="数値ランクと浮動小数点の閾値処理"}

<!--en:af221f3a-->
[実用上の部分空間の幾何学]{.kicker}

<!--en:32f43893-->
- **厳密なゼロは当てにならない：**丸め誤差により特異値は最大 $\|\mathbf{E}\|_2 \sim \epsilon_{\text{mach}}\sigma_1$ ほど摂動を受けるため（Weyl の不等式）、数学的なゼロは通常 $10^{-15}\sigma_1$ 程度の微小値となり、厳密な $0$ になるのは一部に限られる

<!--en:bafa107f-->
- **標準的なランク許容誤差：**`np.linalg.matrix_rank` は $\text{tol} = \sigma_1 \max(m, n)\,\epsilon_{\text{mach}}$ を超える特異値を数える

<!--en:0de1e335-->
::: {.slide title="打ち切り SVD と Eckart–Young の定理"}

<!--en:b543f5a2-->
[最良低ランク近似]{.kicker}

<!--en:ffd7cb7c-->
- **ランク $k$ 打ち切り SVD：**$\mathbf{A}_k = \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ は $k(m + n + 1)$ 個の数値を保持する（元の $mn$ 個の代わり）

<!--en:2abd922f-->
- **Eckart–Young–Mirsky の定理：**$\mathbf{A}_k$ は $\|\mathbf{A} - \mathbf{B}\|$ を $\operatorname{rank}\mathbf{B} \le k$ の全行列にわたってスペクトルノルム（$\sigma_{k+1}$）とフロベニウスノルム（$\sqrt{\sum_{i>k}\sigma_i^2}$）の双方で最小化する

<!--en:ff4cebcb-->
- **一意性：**$\sigma_k > \sigma_{k+1}$ である限りフロベニウスノルムにおいて一意

<!--en:08e958b3-->
::: {.slide title="部分空間の交叉による Eckart–Young の証明"}

<!--en:446a2f57-->
[Eckart–Young の証明]{.kicker}

<!--en:1c08f8b4-->
- **次元の勘定（$\dim\mathcal{K} + \dim\mathcal{V} > n$）：**任意の競合行列 $\mathbf{B}$（$\operatorname{rank}\mathbf{B} \le k$）の零空間 $\mathcal{K} = \ker\mathbf{B}$ は次元が少なくとも $n - k$ であり、一方 $\mathcal{V} = \operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{k+1}\}$ の次元は $k + 1$

<!--en:8aa8500f-->
- **非自明な交叉：**$(n - k) + (k + 1) = n + 1 > n$ より、$\mathcal{K} \cap \mathcal{V}$ は単位ベクトル $\mathbf{x} = \sum_{i=1}^{k+1} c_i\mathbf{v}_i$（$\sum c_i^2 = 1$）で $\mathbf{B}\mathbf{x} = \mathbf{0}$ を満たすものを含む

<!--en:af7cdcdc-->
- **下界：**この $\mathbf{x}$ 上で、$\mathbf{A}$ は少なくとも $\sigma_{k+1}$ 倍に引き伸ばす
  $$\|\mathbf{A} - \mathbf{B}\|_2^2 \ge \|\mathbf{A}\mathbf{x}\|^2 = \sum_{i=1}^{k+1} c_i^2\sigma_i^2 \ge \sigma_{k+1}^2$$

<!--en:d2fe36ab-->
::: {.slide title="Weyl の不等式とフロベニウスノルムでの証明"}

<!--en:bbe6e269-->
[摂動とフロベニウス限界]{.kicker}

<!--en:c675117a-->
- **Weyl の補題：**$\sigma_{i+j-1}(\mathbf{X} + \mathbf{Y}) \le \sigma_i(\mathbf{X}) + \sigma_j(\mathbf{Y})$ は、$\mathbf{X} + \mathbf{Y}$ をランク $(i+j-2)$ の打ち切り $\mathbf{X}_{i-1} + \mathbf{Y}_{j-1}$ と比較して得られる

<!--en:301f86d1-->
- **1-リプシッツ安定性（$j = 1$）：**$|\sigma_i(\mathbf{A} + \mathbf{E}) - \sigma_i(\mathbf{A})| \le \|\mathbf{E}\|_2$

<!--en:b02dc226-->
- **フロベニウスノルムの証明（$j = k + 1$）：**$\sigma_i(\mathbf{A} - \mathbf{B}) \ge \sigma_{k+i}(\mathbf{A}) \implies \|\mathbf{A} - \mathbf{B}\|_F^2 \ge \sum_{i>k}\sigma_i^2$

<!--en:8325b921-->
::: {.slide title="400×500 の写真における Eckart–Young"}

<!--en:f4d7a406-->
[低ランク圧縮]{.kicker}

<!--en:0401e6b4-->
- **べき乗則減衰（$\sigma_i \propto i^{-1.1}$）：**自然画像の特異値は急速に減衰するため、$k = 20$ 個のダイアドがフロベニウスエネルギーの $99.4\%$（相対誤差 $8\%$）をわずか $9\%$ のパラメータで捉え、ノルム上位 20 本の画像列への射影（誤差 $31\%$）にほぼ $4\times$ の差をつける

<!--en:ee83e44a-->
::: {.slide title="エネルギー比・中心化・SVD ノイズ除去"}

<!--en:ac2363f5-->
[ランク k の選択]{.kicker}

<!--en:fc7bc863-->
- **フロベニウスエネルギー比：**ランク $k$ で保持される総二乗エネルギーの割合が相対フロベニウス誤差を直接決定する
  $$\text{Energy}(k) = \frac{\sum_{i \le k} \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} = 1 - \frac{\|\mathbf{A} - \mathbf{A}_k\|_F^2}{\|\mathbf{A}\|_F^2}$$

<!--en:6bc9b36e-->
- **平均の支配と Gavish–Donoho ノイズ除去：**中心化していない写真では平均輝度だけで $\sigma_1^2$ がエネルギーの $89\%$ を占める。加法ノイズ $\mathbf{A} = \mathbf{B} + \mathbf{N}$（$\sigma_{\text{noise}}$）の下では、最適ハード閾値処理は $\tfrac{4}{\sqrt{3}}\sqrt{n}\,\sigma_{\text{noise}} \approx 2.309\sqrt{n}\,\sigma_{\text{noise}}$ 未満の特異値を切り捨てる

<!--en:e8285937-->
::: {.slide title="エネルギー比・中心化・SVD ノイズ除去（続き）"}

<!--en:ac2363f5-2-->
[ランク k の選択]{.kicker}

<!--en:12883fff-->
::: {.slide title="中心化 SVD による主成分分析"}

<!--en:01a7963e-->
[次元削減]{.kicker}

<!--en:71672a77-->
- **まず中心化（$\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^\top$）：**最適な $k$ 次元アフィン平面は常に標本平均 $\bar{\mathbf{x}}$ を通り、共分散は $\mathbf{C} = \tfrac{1}{n}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} = \tfrac{1}{n}\mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top$

<!--en:870e8729-->
- **最大分散 = 最小誤差：**Rayleigh と Eckart–Young より、上位右特異ベクトル $\mathbf{v}_1, \ldots, \mathbf{v}_k$（$\tilde{\mathbf{X}}$ のもの）は射影後の分散（$\sigma_i^2/n$）の最大化と再構成誤差の最小化を同時に達成する
  $$\tilde{\mathbf{X}}\mathbf{V}_k\mathbf{V}_k^\top = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^\top = \tilde{\mathbf{X}}_k$$

<!--en:1ea42704-->
::: {.slide title="コードで見る PCA の分散と再構成誤差"}

<!--en:653250e1-->
[実践における PCA]{.kicker}

<!--en:e8327347-->
300 点の点群（$36^\circ$ 回転、母標準偏差 $3.0$ と $0.8$）に対し、中心化データの SVD は主軸を復元し、ランク 1 の平均二乗誤差は $\sigma_2^2/n$ に一致する

<!--en:e463bb22-->
**中心化を忘れるとどうなるか：**`X.mean(0)` を引かないと、$\mathbf{v}_1$ は点群が最も広がる軸ではなく、原点から重心 $\bar{\mathbf{x}}$ へ向かう方向を指してしまう。

<!--en:aa02b9ce-->
::: {.slide title="ムーア・ペンローズ擬似逆行列 A⁺"}

<!--en:c3ab6d89-->
[連立一次方程式の解法]{.kicker}

<!--en:6941a537-->
- **非ゼロ伸縮の反転：**任意の $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \in \mathbb{R}^{m \times n}$ に対し、*ムーア・ペンローズ擬似逆行列* $\mathbf{A}^+ \in \mathbb{R}^{n \times m}$ は次式で定まる
  $$\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$$
  ただし $\boldsymbol{\Sigma}^+_{ii} = 1/\sigma_i$（$\sigma_i > 0$ のとき）、それ以外は $0$

<!--en:584dadfc-->
- **最小ノルム最小二乗解（$\hat{\mathbf{x}} = \mathbf{A}^+\mathbf{b}$）：**特異基底の座標 $\mathbf{y} = \mathbf{V}^\top\mathbf{x}$ と $\mathbf{c} = \mathbf{U}^\top\mathbf{b}$ では $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 = \sum_{i \le r}(\sigma_i y_i - c_i)^2 + \sum_{i > r} c_i^2$

<!--en:5a3a695e-->
- **独立な役割分担：**$y_i = c_i/\sigma_i$（$i \le r$）が残差を最小化し、零空間座標 $y_i = 0$（$i > r$）が $\|\mathbf{x}\| = \|\mathbf{y}\|$ を最小化する

<!--en:b89786e5-->
::: {.slide title="スペクトルフィルタとしてのリッジ回帰"}

<!--en:e5951b8f-->
[正則化最小二乗法]{.kicker}

<!--en:88887837-->
- **小さな特異値が危険な理由：**$\mathbf{A}^+\mathbf{b} = \sum_{i \le r} \frac{\mathbf{u}_i^\top\mathbf{b}}{\sigma_i}\mathbf{v}_i$ において微小な $\sigma_i$ で割ると、$\mathbf{u}_i$ 方向の測定ノイズが $1/\sigma_i$ 倍に増幅される

<!--en:dce07e2d-->
- **リッジ回帰と打ち切り SVD のフィルタ係数：**リッジ回帰 $\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|^2$（重み減衰）は、$\sigma_i^2 \ll \lambda$ の方向を階段状に切り捨てるのではなく滑らかに減衰させる
  $$\mathbf{x}_\lambda = (\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})^{-1}\mathbf{A}^\top\mathbf{b} = \sum_{i=1}^r \underbrace{\frac{\sigma_i^2}{\sigma_i^2 + \lambda}}_{\text{ridge filter}} \frac{\mathbf{u}_i^\top\mathbf{b}}{\sigma_i}\,\mathbf{v}_i$$

<!--en:476fa6b8-->
::: {.slide title="スペクトルフィルタとしてのリッジ回帰（続き）"}

<!--en:e5951b8f-2-->
[正則化最小二乗法]{.kicker}

<!--en:dd6280ee-->
::: {.slide title="条件数 κ(A) = σ₁ / σᵣ"}

<!--en:7fee687a-->
[感度と条件性]{.kicker}

<!--en:951dbff6-->
- **両端の伸縮率の比：**$\kappa(\mathbf{A}) = \sigma_1/\sigma_r \ge 1$（$\kappa = 1$ は直交行列 $\mathbf{Q}$、$\kappa = \infty$ は特異行列）

<!--en:8952f9ff-->
- **相対誤差の増幅：**$\mathbf{A}\mathbf{x} = \mathbf{b}$ に摂動 $\delta\mathbf{b}$ が加わると、解の相対誤差は高々 $\kappa(\mathbf{A})$ 倍に増幅される：$\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(\mathbf{A})\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}$

<!--en:334a9eb7-->
- **グラム行列による二乗：**$\kappa(\mathbf{A}^\top\mathbf{A}) = (\sigma_1/\sigma_r)^2 = \kappa(\mathbf{A})^2$

<!--en:532e26c1-->
::: {.slide title="条件数による誤差増幅と細長い谷"}

<!--en:8ac50f08-->
[コードと最適化における条件性]{.kicker}

<!--en:1a3f826c-->
- **勾配降下法のジグザグ挙動：**$\tfrac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ では等位集合の軸比が $\kappa(\mathbf{A})$、ヘッセ行列 $\mathbf{H} = \mathbf{A}^\top\mathbf{A}$ の条件数が $\kappa(\mathbf{A})^2$ となり、勾配降下法は急峻な谷壁の間で跳ね返る

<!--en:1a4f4500-->
::: {.slide title="LoRA：事前学習済み重みの低ランク適応"}

<!--en:c282af95-->
[現代の深層学習における SVD]{.kicker}

<!--en:66c5e62b-->
- **並列ランク $k$ バイパス：**LoRA は事前学習済みの重み $\mathbf{W} \in \mathbb{R}^{m \times n}$ を固定し、ランク $k$ の更新 $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$（$\mathbf{B} \in \mathbb{R}^{m \times k}$、$\mathbf{A} \in \mathbb{R}^{k \times n}$）を学習する
  $$\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$$

<!--en:d88c4699-->
- **$256\times$ のパラメータ圧縮：**$4096 \times 4096$ の Transformer 層（重み数 $16.8\text{M}$）では、ランク $k = 8$ により $k(m+n) = 65{,}536$ 個（$0.39\%$）のパラメータのみを訓練し、$\mathbf{W}$ に統合すれば推論時の追加コストはゼロになる

<!--en:e7632779-->
- **Eckart–Young が語ること：**任意のフル更新 $\Delta\mathbf{W}_{\text{full}}$ は誤差 $\sigma_{k+1}(\Delta\mathbf{W}_{\text{full}})$ 以内でランク $k$ アダプタにより近似できるが、LoRA は*何らかの*低ランク更新が低い損失に達するときに成功する（Eckart–Young の帰結ではなく経験的発見）

<!--en:bb7116ae-->
::: {.slide title="極分解と Muon の Newton–Schulz 反復"}

<!--en:3404bb3f-->
[直交化された行列更新]{.kicker}

<!--en:848c8ee2-->
- **極因子（$\mathbf{Q} = \mathbf{U}\mathbf{V}^\top$）：**最も近い直交行列（$\sigma_i \to 1$）であり、Muon がすべての有効方向を単位スケールで更新するために用いる

<!--en:16ebe4d9-->
- **行列積のみの Newton–Schulz 反復：**$\mathbf{X}_{t+1} = \tfrac{1}{2}(3\mathbf{X}_t - \mathbf{X}_t\mathbf{X}_t^\top\mathbf{X}_t)$ を $\mathbf{M}/\|\mathbf{M}\|_F$ から反復すると $p(\sigma) = \tfrac{1}{2}(3\sigma - \sigma^3) \to 1$ が適用される

<!--en:01874494-->
::: {.slide title="極分解と Muon の Newton–Schulz 反復（続き）"}

<!--en:3404bb3f-2-->
[直交化された行列更新]{.kicker}

<!--en:caa05cc5-->
::: {.slide title="スペクトル正規化と減衰の計測"}

<!--en:12a018c4-->
[スペクトルの制御と診断]{.kicker}

<!--en:4324e5b6-->
- **$\sigma_1 = \|\mathbf{W}\|_2$ によるリプシッツ制御：**$\|\mathbf{W}\mathbf{x} - \mathbf{W}\mathbf{x}'\| \le \sigma_1\|\mathbf{x} - \mathbf{x}'\|$ であるため、スペクトル正規化は $\mathbf{W}$ を $\sigma_1$ の推定値（更新ごとに 1 ステップの交互べき乗法 $\mathbf{u} \leftarrow \mathbf{W}\mathbf{v}$、$\mathbf{v} \leftarrow \mathbf{W}^\top\mathbf{u}$ で追跡）で割る

<!--en:0cf125e8-->
- **低ランク性は仮定せず計測する：**$400 \times 500$ のガウスノイズ行列と中心化した猫の写真を比較すると、低ランク圧縮にスペクトルの減衰が不可欠な理由がわかる

<!--en:77589ef0-->
::: {.slide title="概念チェック：SVD・Eckart–Young・条件数"}

<!--en:26a317d8-->
[概念チェック]{.kicker}

<!--en:f2fddd7e-->
1. **特異値と固有値：**$\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$（固有値 $2 \pm i$、$|\lambda| = \sqrt{5}$）の特異値は $\sqrt{5}$ に等しいか。また $\sigma_1, \sigma_2$ は $\det\mathbf{A} = 5$ とどう関係するか。

<!--en:e4ee50ac-->
2. **せん断行列の条件数：**せん断行列 $\mathbf{T} = \left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ の特異値は $\sigma_1 = \varphi$ と $\sigma_2 = 1/\varphi$（$\varphi \approx 1.618$）である。$\kappa(\mathbf{T})$ と $\kappa(\mathbf{T}^\top\mathbf{T})$ はいくらか。

<!--en:56eaab30-->
3. **劣決定系の擬似逆行列：**1 本の方程式 $x_1 + x_2 = 2$ に対して擬似逆行列はどの解 $\hat{\mathbf{x}} = \mathbf{A}^+\mathbf{b}$ を返すか。またその理由は何か。

<!--en:51647851-->
**解答：**
**1.** 等しくない。$\mathbf{A}^\top\mathbf{A} = \left(\begin{smallmatrix}2&-1\\-1&13\end{smallmatrix}\right)$ の固有値は $\sigma_{1,2}^2 = (15 \pm 5\sqrt{5})/2$ であり、$\sigma_1 \approx 3.618$、$\sigma_2 \approx 1.382$ となる。その積は $\sigma_1\sigma_2 = 5 = |\det\mathbf{A}| = |\lambda_1\lambda_2|$ である。
**2.** $\kappa(\mathbf{T}) = \sigma_1/\sigma_2 = \varphi / (1/\varphi) = \varphi^2 \approx 2.618$ であり、グラム行列を作ると二乗されて $\kappa(\mathbf{T}^\top\mathbf{T}) = \varphi^4 \approx 6.854$ となる。
**3.** $\hat{\mathbf{x}} = (1, 1)$。直線 $x_1 + x_2 = 2$ 上で原点に最も近い一意の点である（零空間ベクトル $(1, -1)$ 方向の成分がゼロ）。

<!--en:bdfd89ce-->
::: {.slide title="まとめ：特異値分解"}

<!--en:c5a222f8-->
[重要ポイント]{.kicker}

<!--en:90899ac6-->
| 概念・手法 | 数式表現 | 主要な性質と深層学習での役割 |

<!--en:7e10e6bb-->
| **SVD とダイアド和** | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_{i=1}^r \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ | 任意の $m \times n$ 行列に存在し、$\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$、$\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$ |

<!--en:a31b6905-->
| **4 つの部分空間とノルム** | $r = \#\{\sigma_i > 0\}$、$\|\mathbf{A}\|_2 = \sigma_1$、$\|\mathbf{A}\|_F^2 = \sum \sigma_i^2$ | $\mathbf{v}_{1..r}$ は行空間、$\mathbf{v}_{r+1..n}$ は零空間、$\mathbf{u}_{1..r}$ は列空間、$\mathbf{u}_{r+1..m}$ は左零空間を張る |

<!--en:6dcfc3e0-->
| **Eckart–Young と PCA** | $\|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}$、$\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i>k} \sigma_i^2$ | 打ち切り SVD $\mathbf{A}_k$ が最良ランク $k$ 近似を与え、中心化データでは $\tilde{\mathbf{X}}_k$ が PCA となる |

<!--en:46013541-->
| **擬似逆行列とリッジ回帰** | $\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$、フィルタ $\frac{\sigma_i^2}{\sigma_i^2 + \lambda}$ | $\hat{\mathbf{x}} = \mathbf{A}^+\mathbf{b}$ が最小ノルム最小二乗解を与え、リッジ回帰が微小な $\sigma_i$ のノイズを減衰 |

<!--en:507f86d3-->
| **条件数 $\kappa(\mathbf{A})$** | $\kappa(\mathbf{A}) = \sigma_1/\sigma_r$、$\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}$ | 誤差増幅率と谷の細長さを規定し、$\kappa(\mathbf{A}^\top\mathbf{A}) = \kappa(\mathbf{A})^2$ |

<!--en:a15e9770-->
| **LoRA・Muon・リプシッツ制御** | $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$、$\mathbf{Q} = \mathbf{U}\mathbf{V}^\top$、$\mathbf{W}/\sigma_1$ | LoRA はアダプタを圧縮し、Newton–Schulz は Muon 用の $\mathbf{U}\mathbf{V}^\top$ を計算し、$\sigma_1$ は層の伸縮率を抑える |
