<!--en:57266ae0-->
# 固有値分解

<!--en:f32cc06a-->
:numref:`sec_mdl-linear-maps` では、行列を 1 回適用したときに空間に何が起きるかを問うた。しかし機械学習における多くの計算では、同じ行列を何度も繰り返し適用する。同一の層を持つ深層線形ネットワークは 50 層を通過すると $\mathbf A^{50}\mathbf x$ を計算し、二次目的関数に対する勾配降下法はステップごとに誤差へ同じ固定された行列を掛ける。いずれの場合も知りたいのは、信号が増大するか、減衰して消えるか、あるいは一定の値に落ち着くか、そしてそれがどれほど速く進むかである。行列のべき乗を具体的に掛け算すれば数値は得られるが、なぜそうなるのかの説明にはならない。

<!--en:0639cfe4-->
2 つのウェブページ間のランダムウォークは、この問いの最小の例である。ページ 1 にいるサーファーがそれぞれ確率 $1/2$ でページ 1 に留まるかページ 2 へクリックして移動し、ページ 2 からは確率 $3/4$ でそのまま留まり、確率 $1/4$ でページ 1 に戻るとしよう。$\mathbf p_k$ を $k$ 回のクリック後に各ページにいる確率を並べたベクトルとする。1 回クリックするごとに、このベクトルには次の固定された行列が掛けられる。

<!--en:2c3b1c80-->
この行列の各列はそれぞれのページからのクリック確率を記録しているため、列の和は 1 になる。ページ 1 から出発して $\mathbf p_0=(1,0)$ とすると、最初の数回の分布は $(0.5,0.5)$、$(0.375,0.625)$、$(0.344,0.656)$、$(0.336,0.664)$ と推移する。クリックを重ねるたびに変化量は小さくなり、ランダムウォークがある一定の分布へ落ち着いていくように見える。50 回クリックした後にサーファーはどこにおり、分布はどれほどの速さで収束するのだろうか。PageRank は、まさにこれと同じ問いをウェブ全体に対して投げかけている。

<!--en:7d044185-->
鍵となるのは、行列が単なる定数倍として作用する特別な方向、すなわち固有ベクトル（eigenvector）を座標軸として初期ベクトルを書き直すことである。その座標系では、各成分がスカラーのべき乗 $\lambda^k$ として独立に時間発展する。

<!--en:b70522ee-->
:numref:`sec_linear-algebra` では、行列が伸縮のみを施す方向としてすでに固有ベクトルに出会った。対称行列 $\mathbf S=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ は $(1,1)$ を 3 倍に、$(1,-1)$ を 1 倍に引き伸ばし、任意の対称行列に対して $\mathbf A^k\mathbf x=\sum_i c_i\lambda_i^k\mathbf v_i$ が成り立った。本節では対称性の仮定を外す。任意の正方行列の固有値（eigenvalue）を求める方法を学び、空間全体を張る固有ベクトルの組がいつ存在し、存在しないときには何がそれに代わるのかを確認したうえで、なぜ対称性があるとこれほど豊かな性質が保証されるのかを証明する。

<!--en:7eca0eaa-->
なかでも特に重要な結果は 3 つある。固有ベクトル展開 :eqref:`eq_mdl-eig-power-expansion`（$\mathbf A^k\mathbf x$ の展開）、スペクトル定理（spectral theorem）、そして二次形式（quadratic form）を重み付き二乗和として表す :eqref:`eq_mdl-quadform` である。先を急ぐ読者は、:numref:`subsec_mdl-diagonalization` から直接 :numref:`subsec_mdl-spectral-theorem` へ進んでよい。行列が大きいときに固有値の位置をどう見積もって計算するか、また固有値だけでは判断を誤る状況とはどのようなものかという 2 つの実践的な問いは、:numref:`sec_mdl-spectra-in-practice` に譲る。本節全体を通して計算には NumPy を用いる。NumPy の `eig` 関数は、実行列が複素固有値を持つ場合には複素数をそのまま返す。

<!--en:9e830aa9-->
## 固有値と対角化

<!--en:bbd9d45a-->
### 固有ベクトルと固有値

<!--en:3fb5db39-->
特定の入力ベクトルに対しては、行列 $\mathbf P$ はあたかも 1 つの数であるかのようにスケール倍するだけに働く。$\mathbf P$ を $(1,2)$ に適用すると $\mathbf P(1,2)=(\tfrac12+\tfrac12,\ \tfrac12+\tfrac32)=(1,2)$ となってまったく変化しない一方、$\mathbf P$ を $(1,-1)$ に適用すると $(\tfrac12-\tfrac14,\ \tfrac12-\tfrac34)=\tfrac14(1,-1)$ となる。第 1 の方向に沿っては $\mathbf P$ は $1$ 倍として作用し、第 2 の方向に沿っては $1/4$ 倍として作用する。

<!--en:7547fdec-->
一般に、ゼロでないベクトル $\mathbf v$ は、正方行列 $\mathbf A$ と対応する*固有値* $\lambda$ に対して次の関係を満たすとき、その行列の*固有ベクトル*と呼ばれる。

<!--en:8f490631-->
幾何学的には、$\mathbf A$ は固有ベクトルを倍率 $\lambda$ で伸縮させるだけであり、$\lambda<0$ のときはその向きを反転させる。サーファーの行列 $\mathbf P$ に対する固有対（eigenpair）$(\lambda,\mathbf v)$ は、$(1,(1,2))$ と $(\tfrac14,(1,-1))$ である。零ベクトルはすべての $\lambda$ に対して :eqref:`eq_mdl-eigpair` を自明に満たしてしまうため、固有ベクトルからは除外する。しかし、$\mathbf A\mathbf v=\lambda\mathbf v$ を満たすすべての解（$\lambda$ は固定）は、$\mathbf 0$ を含めると線形部分空間をなし、これを $\lambda$ の*固有空間*（eigenspace）と呼ぶ。

<!--en:1ad31c5d-->
一方、固有値 $\lambda=0$ は完全に正当な固有値である。これは $\mathbf A\mathbf v=\mathbf 0$ がゼロでないある $\mathbf v$ に対して成り立つことを意味するから、正方行列が特異（singular）であることと、$0$ がその固有値の 1 つであることは同値になる。また、実行列が実固有ベクトルを持つとは限らないことにも注意しよう。$90^\circ$ の回転は、平面上のゼロでないあらゆる方向を回転させてしまう。その固有値は複素数になるが、この場合は :numref:`subsec_mdl-complex-rotation` で詳しく扱う。

<!--en:21f79e0e-->
可逆な $2\times2$ 行列が単位円に作用してできる楕円の主軸と、固有ベクトルとを混同しがちであるが、:numref:`fig_mdl-la-eig-ellipse` に示すとおり両者は別の概念である。対称行列 $\mathbf S$ では、確かに固有ベクトルは楕円の軸上に並ぶ。しかし上三角行列 $\left(\begin{smallmatrix}1&1\\0&2\end{smallmatrix}\right)$ ではそうなっていない。その固有ベクトルは $(1,0)$（$\lambda=1$）と $(1,1)$（$\lambda=2$）であるのに対し、楕円の軸は $58^\circ$ と $-32^\circ$ の方向を向いている（その求め方は :numref:`sec_mdl-svd-low-rank` で示す）。楕円の軸が最大および最小の伸縮方向を表すのに対し、固有ベクトルはまったく回転しない方向を表す。対称行列ではこれら 2 組の方向が常に一致することが :numref:`subsec_mdl-spectral-theorem` のスペクトル定理で証明されるが、一般の行列では一致するとは限らない。

<!--en:a18eae54-->
![単位円（破線）とその像（青）。左：対称行列 $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ では、単位固有ベクトル（緑）が楕円の軸上に乗り、それらの像（赤）はそれぞれ 3 倍と 1 倍にスケールされる（$\lambda=1$ では 2 本の矢印が重なる）。右：上三角行列 $\left(\begin{smallmatrix}1&1\\0&2\end{smallmatrix}\right)$ では、固有ベクトル $(1,0)$ と $(1,1)$ は楕円の軸（灰色）と一致しない。](../img/mdl-la-eig-ellipse.svg)

<!--en:be5929af-->
### 特性多項式

<!--en:a3f12851-->
$\mathbf P$ の固有ベクトルは観察によって見つけたが、より大きな行列ではその方法は通用しない。任意の正方行列の固有値を体系的に求めるには、$\lambda\mathbf v$ を :eqref:`eq_mdl-eigpair` の左辺へ移項し、$\mathbf v$ をくくり出す。

<!--en:187a344a-->
ゼロでない解 $\mathbf v$ が存在するのは、$\mathbf A-\lambda\mathbf I$ が非自明な零空間（null space）を持つとき、すなわち特異であって $\det(\mathbf A-\lambda\mathbf I)=0$ を満たすとき、かつそのときに限る。この行列式を展開すると $n$ 次の $\lambda$ の多項式が得られ、これを*特性多項式*（characteristic polynomial）$p(\lambda)=\det(\mathbf A-\lambda\mathbf I)$ と呼ぶ。その根こそが $\mathbf A$ の固有値にほかならない。根 $\lambda$ が求まれば、斉次連立一次方程式 :eqref:`eq_mdl-eigvalue_der` を解くことで対応する固有空間が得られる。

<!--en:7ae7175b-->
この手順をサーファーの行列に適用してみよう。

<!--en:4c5c9a7c-->
根は $1$ と $1/4$ であり、先ほど見つけた 2 つの倍率と一致する。各根において $\mathbf P-\lambda\mathbf I$ は特異であるため、1 つの行を見るだけで固有ベクトルが求まる。$\lambda=1$ のときは、$\mathbf P-\mathbf I$ の第 1 行から $-\tfrac12v_1+\tfrac14v_2=0$、すなわち $v_2=2v_1$ となるので、$\mathbf v=(1,2)$ と選べる。$\lambda=\tfrac14$ のときは、$\mathbf P-\tfrac14\mathbf I$ の第 1 行から $\tfrac14v_1+\tfrac14v_2=0$ となり、$\mathbf v=(1,-1)$ が得られる。なお、すべての固有ベクトルはゼロでないスカラー倍の自由度を除いてのみ定まることに留意しておこう。$\mathbf v$ が固有ベクトルであれば、$c\mathbf v$ も任意の $c\neq0$ に対して固有ベクトルである。:eqref:`eq_mdl-eigpair` の両辺がともに $c$ 倍されるからである。

<!--en:73bc5548-->
実際には、この代数計算は数値計算ライブラリに任せる。NumPy の `eig` は、固有値の配列と、対応する固有ベクトルを列に並べた行列を返す。以下のコードの最終行では、すべての固有対に対して :eqref:`eq_mdl-eigpair` が成り立つことを一括して確かめている。

<!--en:df74775f-->
順序、スケール、符号はいずれもライブラリの実装次第である。ここでは `eig` は固有値を $1/4$ と $1$ の順で返し、各列は $(1,-1)$ および $(1,2)$ に平行な単位長のベクトルとなっている。

<!--en:af6b5808-->
数値計算ライブラリは、実際に特性多項式を作って解いているわけではない。$n\ge5$ では多項式の根をべき根で表す一般の閉じた公式が存在せず、それほど大きくない $n$ でさえ、多項式の係数に生じる微小な丸め誤差（rounding error）に対して根が極端に敏感だからである。その代わりに `eig` は、固有値を変えることなく直交基底変換の列を順次適用し、行列を段階的に三角行列へと近づけていく。三角行列になれば、固有値は対角線上にそのまま現れる。$\lambda\mathbf I$ を引いても三角行列のままであり、三角行列の行列式は単に対角成分の積になるからである（:numref:`subsec_mdl-determinant-general`）。分解すること自体が不可能なほど巨大な行列に対しては、べき乗法（power iteration）（:numref:`subsec_mdl-power-iteration`）が行列・ベクトル積のみを用いて絶対値の大きい固有値を抽出する。

<!--en:29539ed5-->
### 対角化

<!--en:194a2070-->
これで冒頭の $\mathbf P^{50}\mathbf p_0$ に関する問いに答える準備が整った。$\mathbf P$ の 2 つの固有ベクトルを行列 $\mathbf V$ の列として並べ、対応する固有値を $\boldsymbol\Lambda$ の対角線上に配置する。

<!--en:5af99181-->
2 つの固有ベクトル方程式 $\mathbf P\mathbf v_i=\lambda_i\mathbf v_i$ を列として横に並べると、簡潔に $\mathbf P\mathbf V=\mathbf V\boldsymbol\Lambda$ と書ける。$\det\mathbf V=-1-2=-3\neq0$ より行列 $\mathbf V$ は可逆であり、$\mathbf V^{-1}=\tfrac13\left(\begin{smallmatrix}1&1\\2&-1\end{smallmatrix}\right)$ となる。右から $\mathbf V^{-1}$ を掛けると次式を得る。

<!--en:c013e877-->
この分解を右から左へ 3 つのステップとして読もう。まず固有ベクトル座標へ変換し（$\mathbf V^{-1}$）、次に各座標をそれぞれの固有値で独立にスケール倍し（$\boldsymbol\Lambda$）、最後に元の座標へ戻す（$\mathbf V$）。正方行列 $\mathbf A$ が $\mathbf A=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$（ただし $\mathbf V$ は可逆行列、$\boldsymbol\Lambda$ は対角行列）と書けるとき、その行列は*対角化可能*（diagonalizable）であるといい、この分解そのものを*固有値分解*（eigendecomposition）と呼ぶ。このような分解が存在するのは、$\mathbf A$ が $n$ 個の線形独立な固有ベクトルを持ち、それらで $\mathbf V$ の列を埋められるとき、かつそのときに限る。

<!--en:9c027785-->
対角化可能な行列のべき乗は、これで実に簡単に計算できる。$k$ 個の $\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$ の積では、隣り合うすべての $\mathbf V^{-1}\mathbf V$ の組が打ち消し合う。

<!--en:f1a41cf9-->
したがって、対角化可能な行列のべき乗はスカラーのべき乗に帰着する。$\mathbf A^k$ をベクトル $\mathbf x$ に適用し、$\mathbf c=\mathbf V^{-1}\mathbf x$ を固有基底（eigenbasis）における $\mathbf x$ の座標とすれば（すなわち $\mathbf x=\sum_i c_i\mathbf v_i$）、次式が得られる。

<!--en:062341cf-->
各固有座標は、等比数列 $c_i\lambda_i^k$ として独立に推移する。ページ 1 から出発するサーファーの場合、$\mathbf c=\mathbf V^{-1}\mathbf p_0=\tfrac13(1,2)=(\tfrac13,\tfrac23)$ であるから、任意のクリック回数 $k$ の後の分布が閉じた形で次のように求まる。

<!--en:15192317-->
これで冒頭の問いは完全に解決した。$k$ が大きくなるにつれて第 2 項は消え、分布は $\boldsymbol\pi=(1/3,2/3)$ に近づく。つまり長期的には、サーファーはページ 1 の 2 倍の時間をページ 2 で過ごすことになる。残る誤差は各成分で $\tfrac23\,4^{-k}$ であり、常に固有ベクトル $(1,-1)$ の方向を向いている。1 回クリックするごとに誤差は $4$ で割られるため、10 回のクリック後には $6\times10^{-7}$ まで下がり、50 回のクリック後には $10^{-30}$ 未満となる。検算として $k=3$ を代入すると $(\tfrac13+\tfrac1{96},\ \tfrac23-\tfrac1{96})=(0.34375,0.65625)$ となり、冒頭で計算した 3 回目の反復値と一致する。:numref:`fig_mdl-la-markov-iterates` にその幾何学的な様子を示す。すべての $\mathbf p_k$ の成分の和は 1 であるため、反復列は直線 $x+y=1$ 上を滑りながら、$\lambda=1$ の固有ベクトルがこの直線と交わる点へ向かって進んでいく。

<!--en:2bf48901-->
![サーファーの分布 $\mathbf p_k$ は直線 $x+y=1$ に沿って $\boldsymbol\pi=(1/3,2/3)$（固有ベクトル $(1,2)$（$\lambda=1$）がこの直線と交わる点）へと移動していく。誤差は固有ベクトル $(1,-1)$（$\lambda=1/4$）の方向を向き、クリックごとに 4 分の 1 に縮小する。](../img/mdl-la-markov-iterates.svg)

<!--en:85118567-->
50 回の行列積と 2 項からなる固有展開が同じ $\mathbf p_{50}$ を与えることを検証し、連続する誤差の比も確認してみよう。

<!--en:2c5b619d-->
2 つの計算結果は一致し、隣り合う誤差の比はすべて $0.25$ となっている。固有値 $1$ が $\mathbf p_0$ のうち生き残る成分を支配し、固有値 $1/4$ が減衰して消える成分を支配している。

<!--en:27981af0-->
同じ手法は、対角化可能な行列に対する他の関数にもそのまま適用できる。どの固有値も 0 でなければ、$\mathbf A^{-1}=\mathbf V\boldsymbol\Lambda^{-1}\mathbf V^{-1}$ となり、対角線上の各固有値の逆数をとるだけで逆行列が求まる。同様に、行列指数関数（matrix exponential）$e^{\mathbf A t}=\mathbf V e^{\boldsymbol\Lambda t}\mathbf V^{-1}$（線形微分方程式 $\dot{\mathbf x}=\mathbf A\mathbf x$ の解を与える）も、各固有値の指数関数をとることで構成できる。これについては :numref:`sec_mdl-odes-solvers` で詳しく展開する。

<!--en:c69ce60d-->
**相似。** 固有値分解は、$\mathbf P$ と $\boldsymbol\Lambda$ が 2 つの異なる基底で表されたまったく同じ線形変換であることを示している。一般に、$\mathbf A'=\mathbf B\mathbf A\mathbf B^{-1}$ という関係（ただし $\mathbf B$ はある可逆行列）で結ばれる 2 つの行列は*相似*（similar）であるという。次の命題は、基底を取り替えても行列の固有値は変わらないことを保証する。証明には :numref:`sec_mdl-linear-maps` で学んだ行列式の乗法性を用いる。

<!--en:cb744fdd-->
特にこの性質から、$\det(\mathbf B^{-1})=1/\det\mathbf B$ が $\mathbf B\mathbf B^{-1}=\mathbf I$ より従う。

<!--en:1b5c7c3c-->
**命題（相似な行列は特性多項式を共有する）。** *任意の正方行列 $\mathbf A$ と任意の可逆行列 $\mathbf B$ に対して、次が成り立つ。*

<!--en:8d49e44c-->
**証明。** $\lambda\mathbf I=\mathbf B(\lambda\mathbf I)\mathbf B^{-1}$ と書き換えて $\mathbf B$ と $\mathbf B^{-1}$ をくくり出すと、$\det(\mathbf B\mathbf A\mathbf B^{-1}-\lambda\mathbf I) =\det\bigl(\mathbf B(\mathbf A-\lambda\mathbf I)\mathbf B^{-1}\bigr) =\det\mathbf B\,\det(\mathbf A-\lambda\mathbf I)\,\det\mathbf B^{-1} =\det(\mathbf A-\lambda\mathbf I)$ となる。$\blacksquare$

<!--en:e4ad40da-->
したがって、相似な行列は根としての重複度も含めて同じ固有値を持ち、特性多項式によって決まる量はすべて*相似不変量*（similarity invariant）となる。最も有用な 2 つの不変量は行列式とトレースである。任意の対角化可能な行列に対して、次が成り立つ。

<!--en:a2aa7212-->
すなわち、行列式は固有値の積に等しい。$\mathbf V$ がどのような体積の歪みをもたらしても $\mathbf V^{-1}$ がそれを元に戻すため、$\boldsymbol\Lambda$ による各軸のスケーリングだけが残る。トレースについては、その巡回性（:numref:`subsec_mdl-einsum`）を用いる。

<!--en:fd940d24-->
これは積が正方行列になる任意の $\mathbf X$ と $\mathbf Y$ に対して成り立つ。$\mathbf X=\mathbf V$、$\mathbf Y=\boldsymbol\Lambda\mathbf V^{-1}$ と置けば、次が得られる。

<!--en:743470f9-->
したがって、トレースは固有値の和に等しい。サーファーの行列では、$\operatorname{tr}\mathbf P=\tfrac12+\tfrac34=\tfrac54=1+\tfrac14$ および $\det\mathbf P=\tfrac38-\tfrac18=\tfrac14=1\cdot\tfrac14$ と、両方の恒等式を一目で確かめられる。任意の $2\times2$ 行列では、これら 2 つの数から $\lambda^2-(\operatorname{tr})\lambda+\det$ の根として 2 つの固有値が直接定まる。

<!--en:7ffdcb0d-->
実は、固有値を $\mathbb C$ 上で根としての重複度込みで数えれば、対角化可能かどうかにかかわらず*すべての*正方行列で両方の恒等式が成り立つ。*代数学の基本定理*（fundamental theorem of algebra）（ここでは既知とする）により、すべての $n$ 次多項式は $\mathbb C$ 上で完全に因数分解できる。

<!--en:d37bb527-->
$\lambda=0$ を代入すれば、ただちに $\det\mathbf A=\prod_i\lambda_i$ が得られる。トレースについては、両辺の $\lambda^{n-1}$ の係数を比較する。右辺の係数は $(-1)^{n-1}\sum_i\lambda_i$ である。一方、左辺で $\mathbf A-\lambda\mathbf I$ の展開項のうち次数 $n-1$ に達するのは対角成分同士の積だけであり、その係数は $(-1)^{n-1}\sum_i a_{ii}$ となる。ゆえに $\operatorname{tr}\mathbf A=\sum_i\lambda_i$ が成り立つ。実行列の実数でない固有値は必ず複素共役の対として現れるため、その和も積も実数のまま保たれる。

<!--en:a27d3c47-->
最後に、対角化可能な行列のランクは、重複度込みで数えた非ゼロ固有値の個数に等しい。可逆行列 $\mathbf V$ と $\mathbf V^{-1}$ を掛けても次元は保たれるため、$\operatorname{rank}\mathbf A=\operatorname{rank}\boldsymbol\Lambda$ が成り立つからである。行列が対角化可能でない場合にどうなるかは、演習問題 :exref:`ex_mdl-eigendecomposition-rank-zero-eigenvalues` で調べる。

<!--en:446c9fb2-->
### 代数的重複度と幾何学的重複度

<!--en:6bea9831-->
サーファーの行列は 2 つの相異なる固有値を持ち、そこから 2 つの線形独立な固有ベクトルが得られた。これは偶然ではない。相異なる固有値からは、常に線形独立な固有ベクトルが生じる。

<!--en:eee4e989-->
**命題（相異なる固有値に属する固有ベクトルの線形独立性）。** *相異なる固有値に属する固有ベクトルは線形独立である。特に、任意の $n\times n$ 行列が $n$ 個の相異なる固有値を持てば、その行列は対角化可能である。*

<!--en:9d7aed42-->
**証明。** 矛盾を導くため、固有ベクトル $\mathbf{v}_1,\ldots,\mathbf{v}_m$（対応する固有値 $\lambda_1,\ldots,\lambda_m$ は互いに異なる）が線形従属であると仮定しよう。非自明な線形従属関係のうち、非ゼロの係数が最も少ないもの $\sum_{i=1}^{m} c_i\mathbf{v}_i = \mathbf 0$ を 1 つ選び、必要なら添字を付け替えて $c_m\neq0$ とする。もし非ゼロの項が $c_m\mathbf{v}_m=\mathbf 0$ だけなら $\mathbf{v}_m=\mathbf 0$ となってしまうため、この関係式には少なくとも 2 つの非ゼロ係数が含まれる。ここでこの関係式に $\mathbf{A}$ を適用し、元の関係式の $\lambda_m$ 倍を引く。

<!--en:19e8c82c-->
すると $\mathbf{v}_m$ の項が消え、より短い関係式が残る。しかもこれは依然として非自明である。他の任意の非ゼロ係数 $c_i$（$i<m$）は、固有値が互いに異なるため $c_i(\lambda_i-\lambda_m)\neq0$ として生き残るからである。これは選んだ関係式の最小性に矛盾する。$\mathbf A$ が $n$ 個の相異なる固有値を持つとき、その $n$ 個の線形独立な固有ベクトルは $\mathbb C^n$（すべての固有値が実数なら $\mathbb R^n$）の基底をなすため、$\mathbf{V}$ は可逆となる。$\blacksquare$

<!--en:678684c9-->
固有値に重複がある場合は、2 種類の重複度を区別して追う必要がある。$\lambda$ が特性多項式の根として現れる回数を*代数的重複度*（algebraic multiplicity）$a(\lambda)$ と呼ぶ。一方、その固有空間の次元、すなわちその固有値がもたらす線形独立な固有ベクトルの個数を*幾何学的重複度*（geometric multiplicity）$g(\lambda)$ と呼ぶ。

<!--en:d92953d6-->
**命題。** *正方行列のすべての固有値 $\lambda$ に対して、$1\le g(\lambda)\le a(\lambda)$ が成り立つ。*

<!--en:39d7ac7d-->
**証明。** $\mathbf A-\lambda\mathbf I$ は特異であるから、その零空間は非自明であり、$g\ge1$ が従う。上界を示すため、固有空間の基底 $\mathbf v_1,\ldots,\mathbf v_g$ をとり、それを拡張して $\mathbb C^n$ の基底とし、これらの基底ベクトルを列として並べた行列を $\mathbf B$ とする。このとき、最初の $g$ 列について $\mathbf A\mathbf B$ の列は単に $\lambda\mathbf v_1,\ldots,\lambda\mathbf v_g$ となるから、この新しい基底のもとで行列は次のブロック上三角行列の形をとる。

<!--en:e8b175ab-->
:eqref:`eq_mdl-similar-charpoly` より、この行列は特性多項式 $p$ を $\mathbf A$ と共有する。いま固定した固有値を表すのに用いている記号と区別するため、多項式の変数を $t$ と書こう。$\lambda$ に対して $t\mathbf I$ を引いてもブロック上三角構造は保たれ、対角ブロックは $(\lambda-t)\mathbf I_g$ と $\mathbf Y-t\mathbf I$ になる。ブロック上三角行列の行列式は対角ブロックの行列式の積に等しいから（:numref:`subsec_mdl-determinant-general`）、$p(t)=(\lambda-t)^g\det(\mathbf Y-t\mathbf I)$ を得る。したがって $(\lambda-t)^g$ は特性多項式を割り切り、$a\ge g$ が成り立つ。$\blacksquare$

<!--en:3832abe5-->
1 つの固有値から得られる線形独立な固有ベクトルの個数が、根としての重複度を超えることは決してないが、それより少なくなることはあり得る。いま各固有空間から基底を 1 組ずつ選び、選んだベクトルをすべて集めよう。これらは全体としても線形独立である。実際、それらの線形結合が零ベクトルに等しいとして固有値ごとに項をまとめれば、各グループは零ベクトルか固有ベクトルのいずれかになり、相異なる固有値に関する命題からすべてのグループが零ベクトルでなければならないからである。ゆえに行列が持つ線形独立な固有ベクトルは高々 $\sum_\lambda g(\lambda)$ 個である。$\mathbb C$ 上では代数的重複度の和は $n$ に等しく、かつすべての固有値に対して $g(\lambda)\le a(\lambda)$（各 $\lambda$ ごと）が成り立つため、明快な判定条件が得られる。すなわち、*$\mathbf A$ の固有ベクトルが張る空間が $\mathbb C^n$ 全体に一致するのは、$g(\lambda)=a(\lambda)$ がすべての複素固有値 $\lambda$ に対して成り立つとき、かつそのときに限る*。

<!--en:510fa46a-->
実行列が*実*固有ベクトルからなる基底を持つためには、さらにその固有値がすべて実数でなければならない。$90^\circ$ の回転行列 $\mathbf R_{90}=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$ は、なぜこの追加条件が不可欠なのかを示している。その固有値 $\pm i$ はそれぞれ $g=a=1$ を満たすため完全な複素固有基底を持つが、実固有ベクトルは 1 つも存在しない。

<!--en:8eec5fab-->
固有値が重複する場合には、対角化可能性そのものが成り立たなくなることがある。最も単純な反例は、:numref:`subsec_mdl-composition` で登場した*せん断*（shear）である。

<!--en:414e3129-->
この行列の特性多項式は $(1-\lambda)^2$ であるから、$\lambda=1$ の代数的重複度は $2$ である。ところが $(\mathbf{T}-\mathbf{I})\mathbf{v}=\mathbf 0$ は $\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)\mathbf v=\mathbf 0$ となり、これから $v_2=0$ が強制される。固有空間は $(1,0)$ が張る直線だけであり、幾何学的重複度は $1$ にとどまる。したがって固有基底は存在せず、このせん断は対角化可能でない。ある固有値に対して $g(\lambda)<a(\lambda)$ となる行列は*欠陥行列*（defective）と呼ばれる。:numref:`fig_mdl-la-defective-shear` を見れば、固有ベクトルが足りない理由が視覚的にわかる。せん断はすべての水平線をその高さに比例して横へ滑らせるため、原点を通る直線で向きが保たれるのは水平軸ただ 1 本だけなのである。（対照的に、重複固有値を持つ対角化可能な $2\times2$ 行列は、その固有値を $\lambda$ とすれば $\lambda\mathbf I$ にならざるを得ず、あらゆる方向を保つ。）

<!--en:abf8defe-->
![平面に作用する欠陥を持つせん断。すべての水平線がその高さに比例した量だけ右へ滑ってグリッドを傾ける一方、$x$ 軸（緑）は自分自身に写る。すなわち $\mathbf{T}\mathbf{e}_1=\mathbf{e}_1$ が $\lambda=1$ に対する唯一の固有方向である。もう一方の基底ベクトルは水平成分を得て $\mathbf{T}\mathbf{e}_2=(1,1)$ となるため、向きが保たれる第 2 の方向は存在せず、代数的重複度 2 に対して幾何学的重複度は 1 となる。](../img/mdl-la-defective-shear.svg)

<!--en:70272054-->
せん断を反復適用すると、欠陥性が行列のべき乗の振る舞いをどう変えるかがわかる。$\mathbf T^k=\left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$ となるため、$\mathbf T^k\mathbf e_2=(k,1)$ は唯一の固有値が $1$ であるにもかかわらず際限なく増大する。純粋な固有展開からは、このような線形の因子 $k$ は決して生じない。次項では、この因子がどこから現れるのかを説明する。

<!--en:1d90e9b8-->
### ジョルダン標準形

<!--en:9a1af67d-->
幾何学的重複度が代数的重複度に満たないとき、固有基底は存在しない。では、代わりにどのような構造が現れるのだろうか。本書の以降の議論において最も重要な帰結は 1 つ、すなわち欠陥固有値は $\lambda^k$ に $k$ の多項式を掛けるということである。複素数体上では、すべての正方行列は次の*ジョルダンブロック*（Jordan block）から構成されるブロック対角行列と相似になる。

<!--en:3066d4b8-->
これは $m\times m$ のブロックであり、対角線上に単一の固有値 $\lambda$ が並び、そのすぐ上の優対角線（superdiagonal）に $1$ が並ぶ。この分解が*ジョルダン標準形*（Jordan normal form）であり :cite:`Horn.Johnson.2012`、本書では証明なしに用いる。:eqref:`eq_mdl-similar-charpoly` により、その固有値は元の行列と一致する。

<!--en:a1e2deb3-->
各ジョルダンブロックからは、その最初の基底ベクトルという 1 つの固有ベクトルだけが得られる。したがって $\lambda$ の幾何学的重複度は $\lambda$ を持つブロックの個数に等しく、代数的重複度はそれらのブロックのサイズの和に等しい。特に、行列が対角化可能であることと、すべてのジョルダンブロックのサイズが $1\times1$ であることは同値である。せん断 $\mathbf T$（:eqref:`eq_mdl-defective-shear`）は、単一のブロック $\mathbf{J}_2(1)$ そのものである。

<!--en:0f68a56c-->
反復適用のもとで、ジョルダンブロックは単なるスケーリング以上の働きをする。$\mathbf{J}_m(\lambda)=\lambda\mathbf{I}+\mathbf{N}$ と書こう。ここで $\mathbf{N}$ は対角線のすぐ上に $1$ が並び、それ以外の成分がすべて 0 の行列である。幾何学的には、$\mathbf N$ は各基底ベクトルを 1 つ前の基底ベクトルへ送るシフトとして作用するため、$\mathbf N^m=\mathbf 0$ となる（べき乗が零行列になる行列を*べき零*（nilpotent）と呼ぶ）。$\mathbf{I}$ と $\mathbf{N}$ は可換であるため、二項定理によってべき乗を展開できる。$m=2$ のときは最初の 2 項より後のすべての項が $\mathbf{N}^2=\mathbf 0$ を含むため、次の項だけが残る。

<!--en:3222bd81-->
長期的にはやはり固有値が増大と減衰を分けるものの、そこに $k$ の多項式（次数は $m-1$、すなわち $m\times m$ ブロックのサイズより 1 小さい）が掛かるようになる。せん断のように $|\lambda|=1$ のときは、反復列は線形にドリフトする。$|\lambda|$ が $1$ よりわずかに小さいときには、さらに驚くべき現象が起きる。右上の成分 $k\lambda^{k-1}$ は最初増大し、$k\approx-1/\log|\lambda|$ 付近でピークに達してから、ようやく 0 に向かって減衰するのである。$\lambda=0.99$ の場合、ピークは $k\approx100$ 付近に現れ、この成分は $100\cdot0.99^{99}\approx37$ に達する。この予測を数値的に確かめてみよう。

<!--en:3cd4939b-->
$\mathbf{J}$ の固有値はすべて $0.99$ であるにもかかわらず、そのべき乗のノルムは約 $37$（$k=100$）まで上昇し、$27$（$k=200$）に留まったのち、ようやく指数因子 $0.99^k$ が支配的になって減衰へ転じる。固有値だけを見ていても、この過渡的増大（transient growth）を予見することはできない。この現象は :numref:`subsec_mdl-transient-growth` で改めて取り上げる。

<!--en:fc74a323-->
ジョルダンブロックは、実行列が実固有基底を持てない 1 つ目の要因、すなわち固有ベクトルの個数そのものが足りない場合を表している。2 つ目の要因はまったく性質が異なる。実行列がある平面を*回転*させるために、その平面内に実固有ベクトルを 1 つも持たない場合である。

<!--en:d56e2042-->
### 複素固有値

<!--en:61406e46-->
角度 $\theta$ の平面回転を考えよう（:numref:`subsec_mdl-orthogonal-matrices`）。

<!--en:2218bb3a-->
$\theta$ が $\pi$ の倍数でない限り、$\mathbf{R}_\theta$ はゼロでないすべての実ベクトルの向きを変えてしまうため、実固有ベクトルは存在しない。その特性多項式 $\lambda^2-2\cos\theta\,\lambda+1$ は、複素共役な根 $\lambda=\cos\theta\pm i\sin\theta=e^{\pm i\theta}$ を持つ。ここで絶対値 $|\lambda|=1$ は $\mathbf{R}_\theta$ が長さを保つことを表し、偏角 $\pm\theta$ は回転角を記録している。

<!--en:12b2c821-->
次に、成分を見ただけでは回転が隠れているとはわかりにくい、本章で通しで用いる $2\times2$ 行列を見てみよう。

<!--en:4c1e32b4-->
この行列の特性多項式は $\lambda^2-4\lambda+5$ であり、その根は $\lambda=2\pm i=\sqrt5\,e^{\pm i\theta}$（ただし $\theta=\arctan\tfrac12\approx26.6^\circ$）となる。$2+i$ に対する固有ベクトルを求めるため、$\mathbf A-(2+i)\mathbf I$ の第 2 行を読み取ると $-w_1+(1-i)w_2=0$ が得られる。$w_2=1$ と置けば次式を得る。

<!--en:4709f305-->
回転はどこに隠れているのだろうか。複素固有対を持つ任意の実行列 $\mathbf{A}$ をとり、$\mathbf{A}\mathbf{w}=\lambda\mathbf{w}$（ただし $\lambda=a+ib$、$b\neq0$）として、固有ベクトルを実部と虚部に分け $\mathbf{w}=\mathbf{u}+i\mathbf{v}$ と書く。$\mathbf{A}(\mathbf{u}+i\mathbf{v})=(a+ib)(\mathbf{u}+i\mathbf{v})$ を展開して実部と虚部をそれぞれ比較すると、次の 2 式が得られる。

<!--en:28826672-->
どちらの像も $\operatorname{span}\{\mathbf{u},\mathbf{v}\}$ の内部に留まるため、この部分空間は $\mathbf{A}$ のもとで不変である。しかもこれは正真正銘の 2 次元平面である。もし $\mathbf u$ と $\mathbf v$ が線形従属であれば、それらが張る空間は自分自身に写される 1 本の直線（$\mathbf w\neq\mathbf 0$ より零空間ではない）となり、その直線上で $\mathbf A$ はある実スカラー $\mu$ 倍として作用することになる。すると $\mathbf A\mathbf w=\mu\mathbf w$ となって $\lambda=\mu$ が実数となり、$b\neq0$ に矛盾する。そこで $\mathbf{v}$ と $\mathbf{u}$ を列に並べて $\mathbf{B}=[\mathbf{v}\;\;\mathbf{u}]$ と置くと、上の 2 つの等式は $\mathbf{A}\mathbf{B}=\mathbf{B}\left(\begin{smallmatrix}a & -b\\ b & \phantom{-}a\end{smallmatrix}\right)$、すなわち同値な形として次式になる。

<!--en:d6afe2c1-->
不変平面の基底 $(\mathbf v,\mathbf u)$ のもとでは、$\mathbf{A}$ は回転・スケーリングブロック $r\mathbf R_\theta$ として作用する。すなわち絶対値 $r$ 倍にスケールし、偏角 $\theta$ だけ回転させる。この導出では $\mathbf A\mathbf w=\lambda\mathbf w$ しか使っていないため、得られるブロックは $\mathbf w$ のスケールや位相の回転の選び方に依存しない。この対応は逆向きにも成り立つ。$(x,y)\in\mathbb{R}^2$ を複素数 $x+iy$ と同一視すれば、このブロックは $(x,y)$ を $(ax-by,\,bx+ay)$ へ写すが、これは複素数 $a+ib$ の乗算にほかならない。

<!--en:292387ad-->
通しで用いる行列 $\mathbf A$ については、$\mathbf B=\left(\begin{smallmatrix}-1&1\\0&1\end{smallmatrix}\right)$ かつ $\mathbf B^{-1}\mathbf A\mathbf B=\left(\begin{smallmatrix}2&-1\\1&2\end{smallmatrix}\right)$ となり、$\sqrt5$ 倍のスケーリングと $26.6^\circ$ の回転を表す。すると、1 ステップごとにベクトルの長さが一様に $\sqrt5\approx2.24$ 倍になると思うかもしれない。`eig` が返す固有ベクトルからこのブロックを計算し、実際の反復列 $\mathbf A^k(1,0)$ の長さを追ってみよう。

<!--en:4d4942de-->
`eig` がどの固有ベクトルを選んでも、得られるブロックは $\left(\begin{smallmatrix}2&-1\\1&2\end{smallmatrix}\right)$ となる。ところがベクトルの長さは、ステップごとに一定の倍率 $2.24$ で増大するわけでは*ない*。軌跡 $(1,0)\to(1,-1)\to(-1,-4)\to(-9,-11)\to(-31,-24)$ に沿って見ると、1 ステップあたりのノルムの比は約 $1.4$ から $3.5$ の間で大きく揺れ動く。理由は、スケーリングが一様になるのが正規直交ではない座標系 $(\mathbf v,\mathbf u)$ の内部に限られるからである。元の標準座標に戻すと、反復列は引き伸ばされた楕円に沿って外向きに渦を巻き、約 $2\pi/\theta\approx13.6$ ステップで 1 周する。長期的な幾何平均 $\|\mathbf A^k\mathbf x\|^{1/k}$ だけが $\sqrt5$ に落ち着く（実際、$2.25$ に $k=20$ までに到達する）。標準座標において剛体回転と一様なスケーリングになるのは、行列が*正規*（normal）であるときに限られる。この性質は :numref:`subsec_mdl-normal-matrices` で定義する。

<!--en:43b636e4-->
実基底変換を除けば、すべての実正方行列はここまでに見いだした 3 つの基本動作、すなわち直線に沿った伸縮、ジョルダンブロック内のせん断、そして平面内の回転を伴うスケーリングを単独または組み合わせて構成される。この分解は*実ジョルダン標準形*（real Jordan form）として知られている :cite:`Horn.Johnson.2012`。

<!--en:0c27975a-->
## 対称行列と二次形式

<!--en:e72e701b-->
### スペクトル定理

<!--en:17238fe8-->
実ベクトルからなる正規直交固有基底（orthonormal eigenbasis）の存在が保証されるのは、どのような行列だろうか。対称性は、前節で出会った 2 つの障害をどちらも取り除く。複素固有値も、固有ベクトルの不足も生じなくなるのである。:numref:`sec_linear-algebra` の対称行列 $\mathbf S$ がその好例である。その単位固有ベクトル $(1,1)/\sqrt2$ と $(1,-1)/\sqrt2$ は互いに直交しており、次の分解を与える。

<!--en:f42cda1b-->
固有ベクトル行列 $\mathbf V$ は直交行列（$\mathbf V^\top\mathbf V=\mathbf I$）であるため、その逆行列は単なる転置になる。スペクトル定理は、これが*すべての*実対称行列に対して成り立つことを保証する。これは本章の中心定理であり、正定値性、レイリー商（Rayleigh quotient）、そして :numref:`sec_mdl-svd-low-rank` の内容すべてがこの定理の上に築かれる。

<!--en:87c242cf-->
**定理（スペクトル定理）。** *すべての実対称行列 $\mathbf{A}$ は、実固有値と実固有ベクトルからなる正規直交基底を持つ。同値な表現として、*

<!--en:aa852b09-->
*と書ける。ここで $\boldsymbol{\Lambda}=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$ は実行列であり、$\mathbf{V}$ は直交行列である。*

<!--en:29312ef2-->
これを 3 段階で証明する。第 1 にすべての固有値が実数であること、第 2 に相異なる固有値に属する固有ベクトルが直交すること、第 3 に空間全体を張る正規直交固有基底が常に存在することを示す。複素ベクトルに対してはエルミート内積（Hermitian inner product）$\langle\mathbf x,\mathbf y\rangle=\mathbf x^*\mathbf y$ を用いる。ここで $\mathbf x^*$ は共役転置（conjugate transpose）であり、実ベクトル上では通常のドット積に帰着する。外部から借りる事実は、特性多項式が少なくとも 1 つの複素根を持つこと（代数学の基本定理）、すなわち $\mathbb{C}$ 上に少なくとも 1 組の固有対が存在することだけである。(i) と (ii) はそれぞれわずか 2 行で済む。本質的な議論は (iii) にある。

<!--en:2231843f-->
**証明 (i)：固有値は実数である。** いったん複素成分を許して、$\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ かつ $\mathbf{v}\neq\mathbf 0$ とする。$\mathbf{A}$ は実対称行列であるから $\mathbf A^*=\mathbf A^\top=\mathbf A$ が成り立ち、したがって次を得る。

<!--en:55fb82c3-->
$\mathbf{v}^*\mathbf{v}=\|\mathbf v\|^2>0$ であるから、両辺をこれで割れば $\lambda=\bar\lambda$ となり、固有値は実数でなければならない。するとその固有ベクトルは実係数の斉次連立一次方程式 $(\mathbf A-\lambda\mathbf I)\mathbf v=\mathbf 0$ の解となるため、やはり実ベクトルとして選べる。$\blacksquare$

<!--en:7a35d955-->
**証明 (ii)：相異なる固有値の固有ベクトルは直交する。** $\mathbf{A}\mathbf{u}=\lambda\mathbf{u}$ および $\mathbf{A}\mathbf{v}=\mu\mathbf{v}$ とし、$\lambda\neq\mu$ とする。(i) により両固有値および両固有ベクトルは実数としてよい。$\mathbf{A}=\mathbf{A}^\top$ を用いて内積の中で $\mathbf{A}$ をもう一方へ移すと、次式が成り立つ。

<!--en:ef744ba9-->
よって $(\lambda-\mu)\langle\mathbf{u},\mathbf{v}\rangle=0$ となる。$\lambda\neq\mu$ であるから、$\langle\mathbf{u},\mathbf{v}\rangle=0$ でなければならない。$\blacksquare$

<!--en:4ca5174c-->
**証明 (iii)：正規直交固有基底の存在。** $n$ に関する帰納法で示す。$n=1$ のとき主張は明らかである。$n>1$ とすると、(i) より実固有対 $(\lambda_1,\mathbf{v}_1)$ で $\|\mathbf{v}_1\|=1$ を満たすものが存在する。ここで $(n-1)$ 次元の直交補空間（orthogonal complement）$U=\mathbf{v}_1^{\perp}$ を考えよう。対称性により、$\mathbf{A}$ は $U$ を再びそれ自身の中へ写す。実際、任意の $\mathbf{x}\in U$ に対して次が成り立つ。

<!--en:d5425b69-->
$U$ の正規直交基底を 1 組選び、そのベクトルを列として並べた行列を $\mathbf Q\in\mathbb R^{n\times(n-1)}$ とする。$\mathbf A\mathbf Q$ の各列はすべて $U$ に属するため、$\mathbf A\mathbf Q=\mathbf Q\mathbf C$ と書ける。ここで $\mathbf C=\mathbf Q^\top\mathbf A\mathbf Q$ は $\mathbf A$ を $U$ に制限した変換を表す。$\mathbf C$ が対称行列であり（$\mathbf C^\top=\mathbf Q^\top\mathbf A^\top\mathbf Q=\mathbf C$）、サイズが $(n-1)\times(n-1)$ であることに注目しよう。帰納法の仮定より、$\mathbf C=\mathbf V'\boldsymbol\Lambda'\mathbf V'^\top$ が、ある直交行列 $\mathbf V'$ に対して成り立つ。そこで $\mathbf Q\mathbf V'$ の各列を調べると、$(\mathbf Q\mathbf V')^\top\mathbf Q\mathbf V'=\mathbf V'^\top\mathbf V'=\mathbf I$ よりこれらは正規直交であり、$U$ に属するから $\mathbf v_1$ と直交し、さらに $\mathbf A$ の固有ベクトルにもなっている。なぜなら $\mathbf A\mathbf Q\mathbf V'=\mathbf Q\mathbf C\mathbf V'=\mathbf Q\mathbf V'\boldsymbol\Lambda'$ が成り立つからである。これらを $\mathbf v_1$ の後ろに並べれば、直交行列 $\mathbf V=[\mathbf v_1\;\mathbf Q\mathbf V']$ が得られ、$\mathbf A\mathbf V=\mathbf V\boldsymbol\Lambda$ を満たす。$\blacksquare$

<!--en:6d6fb659-->
幾何学的には、:eqref:`eq_mdl-spectral` は任意の対称な線形変換を 3 つの明快な操作へと分解する。まず $\mathbf V^\top$ が正規直交固有ベクトルを座標軸に揃える。これは純粋な回転か、それに鏡映を組み合わせた変換である（上の $\mathbf S$ では $\det\mathbf V=-1$ であった）。次に $\boldsymbol\Lambda$ が各座標軸を対応する固有値倍に伸縮させる。最後に $\mathbf V$ が軸を元の向きへ戻す。これにより、:numref:`fig_mdl-la-eig-ellipse` の左パネルが $\mathbf S$ について図示した性質が、すべての対称行列に対して証明されたことになる。

<!--en:ae3d4ae4-->
長方形か正方かを問わず、あらゆる行列 $\mathbf A$ から対称行列 $\mathbf A^\top\mathbf A$ を作ることができる。その二次形式 $\mathbf x^\top\mathbf A^\top\mathbf A\mathbf x=\|\mathbf A\mathbf x\|^2$ は決して負にならない。次項ではこの性質に名前を付け、判定法を導き、活用していく。

<!--en:8a59a4a6-->
### 正定値性

<!--en:f45197f6-->
二次形式 $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ が負の値をとることはあるだろうか。損失曲面の臨界点が最小点であるか、行列が正当な共分散行列であるか、そして最小二乗法の問題が一意解を持つかは、この 1 つの問いにかかっている。対称行列 $\mathbf{A}$ が次を満たすとき、*半正定値*（positive semidefinite、PSD）であるといい、$\mathbf{A}\succeq0$ と書く。

<!--en:3cbccb37-->
さらに、ゼロでないすべてのベクトルに対して不等号が狭義に成り立つとき、*正定値*（positive definite、PD）であるといい、$\mathbf{A}\succ0$（すべての非ゼロ $\mathbf{x}$ に対して厳密に正）と書く。たとえば $\mathbf x^\top\mathbf S\mathbf x=2x_1^2+2x_1x_2+2x_2^2=x_1^2+x_2^2+(x_1+x_2)^2$ は $\mathbf x\neq\mathbf 0$ のとき常に正であるから $\mathbf S\succ0$ であるが、$\left(\begin{smallmatrix}1&0\\0&-1\end{smallmatrix}\right)$ は $\mathbf e_2$ に対して $-1$ をとるため PD でも PSD でもない。手計算による平方完成は高次元になるとすぐに現実的でなくなるが、固有値を見れば任意の対称行列に対してこの問いに明快な決着がつく。

<!--en:159f020a-->
**命題（固有値による定値性の判定）。** *対称行列 $\mathbf{A}$ に対して、次が成り立つ。*

<!--en:0dbf8bc1-->
**証明。** スペクトル定理を適用して $\mathbf{A}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ と書き、長さを保つ直交基底変換によって固有ベクトル座標 $\mathbf{y}=\mathbf{V}^\top\mathbf{x}$ へ移ると、次式を得る。

<!--en:fa60b274-->
固有ベクトル座標では、二次形式は固有値を重みとする単純な重み付き二乗和になる。すべての $\lambda_i\ge0$ であれば、この和が非負であることは明らかである。逆に $\mathbf{x}=\mathbf{v}_j$ を代入すると $\mathbf y=\mathbf e_j$ かつ $\mathbf{v}_j^\top\mathbf{A}\mathbf{v}_j=\lambda_j$ となるため、$\lambda_j\ge0$ でなければならない。正定値の場合も、$\mathbf y\neq\mathbf 0$ が $\mathbf x\neq\mathbf 0$ のとき常に成り立つことから、等号を除いたまったく同じ議論により従う。$\blacksquare$

<!--en:a8965fb8-->
:eqref:`eq_mdl-quadform` を $\mathbf S$ と $\mathbf x=\mathbf e_1$ で確かめてみよう。固有座標は $\mathbf y=\mathbf V^\top\mathbf e_1=(1/\sqrt2,1/\sqrt2)$ であり、重み付き和は $3\cdot\tfrac12+1\cdot\tfrac12=2$ となって $\mathbf e_1^\top\mathbf S\mathbf e_1=s_{11}$ と一致する。深層学習で正定値性が登場する場面の大半は、次の 3 つの帰結で説明できる。

<!--en:bfdc9812-->
* **PD ならば可逆である。** $\mathbf{A}\succ0$ かつ $\mathbf{A}\mathbf{x}=\mathbf 0$ とすると $\mathbf{x}^\top\mathbf{A}\mathbf{x}=0$ となり、これから $\mathbf{x}=\mathbf 0$ が導かれる。

<!--en:68167778-->
* **グラム行列と共分散行列は PSD である。** 任意の行列 $\mathbf{X}$ に対して $\mathbf{x}^\top\mathbf{X}^\top\mathbf{X}\mathbf{x}=\|\mathbf{X}\mathbf{x}\|^2\ge0$ であるから、$\mathbf X^\top\mathbf X\succeq0$ が成り立つ。これが狭義に PD となるのは、$\mathbf{X}$ が列フルランク（full column rank）を持つとき（すなわち $\mathbf{X}\mathbf{x}=\mathbf 0$ から $\mathbf{x}=\mathbf 0$ が従うとき）、かつそのときに限る。中心化したデータの標本共分散は、グラム行列をサンプル数で割ったものであるため、常に PSD である。

<!--en:0919117f-->
* **PD のヘッセ行列は狭義の極小値を保証する。** $f$ が 2 回連続微分可能で $\nabla f(\mathbf x^*)=\mathbf 0$ を満たし、そのヘッセ行列 $\nabla^2 f(\mathbf x^*)$（2 階偏微分の対称行列）が PD であるとする。微小なステップ $\boldsymbol\delta$ に対し、二次テイラー展開（:numref:`subsec_mdl-second-order-taylor`）は $f(\mathbf x^*+\boldsymbol\delta)=f(\mathbf x^*)+\tfrac12\boldsymbol\delta^\top\nabla^2 f(\mathbf x^*)\boldsymbol\delta+o(\|\boldsymbol\delta\|^2)$ と書ける。:eqref:`eq_mdl-quadform` より二次項は少なくとも $\tfrac12\lambda_{\min}\|\boldsymbol\delta\|^2>0$ となるため、この関数は $\mathbf x^*$ で狭義の極小値をとる。なお、この条件は十分条件であって必要条件ではない。実際、$f(x)=x^4$ は $0$ において狭義の最小値をとるが、そこでの二階導関数は $0$ である。

<!--en:c565d01d-->
正定値性には、*コレスキー分解*（Cholesky factorization）による高速なアルゴリズム的判定法もある。すなわち、$\mathbf{A}\succ0$ が成り立つのは、$\mathbf{A}=\mathbf{L}\mathbf{L}^\top$ を満たす対角成分が正の下三角行列 $\mathbf{L}$ が存在するとき、かつそのときに限る。一方の向きはただちに従う。そのような $\mathbf{L}$ は可逆であるから、$\mathbf{x}^\top\mathbf{A}\mathbf{x}=\|\mathbf{L}^\top\mathbf{x}\|^2>0$ がゼロでないすべての $\mathbf{x}$ に対して成り立つ。コレスキー分解を試みることは正定値性を判定する標準的な数値計算手法であり、すべての固有値を計算するよりはるかに低コストで済む。さらに副産物として、同じコレスキー因子 $\mathbf{L}$ を用いれば、独立な標準ガウス分布からのサンプル $\mathbf z$ を、サンプル $\mathbf L\mathbf z$（共分散は $\mathbf L\mathbf L^\top=\mathbf A$）へと変換できる。

<!--en:11c70960-->
幾何学的には、固有値の符号が曲面 $z=\mathbf{x}^\top\mathbf{A}\mathbf{x}$ の形状を決定する（:numref:`fig_mdl-la-psd`）。すべての固有値が正ならお椀型になり、固有値 0 の方向があればその方向に沿って平坦な谷樋型になり、符号が混在していれば鞍型になる。

<!--en:0c0093ad-->
![3 つの対角行列に対する二次形式の曲面 $z=\mathbf{x}^\top\mathbf{A}\mathbf{x}$ は、固有値の符号に応じた形状をとる。すべての $\lambda_i>0$ ならお椀型（正定値）、1 つの $\lambda_i=0$ なら谷樋型（半正定値）、符号が混在していれば鞍型（不定値）になる。](../img/mdl-la-psd.svg)

<!--en:25f08722-->
グラム行列のランク条件は、その固有値にそのまま現れる。以下では、列が線形独立であるため $\mathbf{X}$ に対して $\mathbf X^\top\mathbf X$ が PD になるはずの縦長行列と、第 2 列が第 1 列の 2 倍であるため固有値 0 を持つ PSD 行列になるはずの行列を比較する。

<!--en:c1ef47ea-->
フルランクの場合は両方の固有値が正になっている。一方、列が線形従属な場合は片方の固有値が 0 に落ちる。二次形式は $(2,-1)$、すなわち $\mathbf X$ が零ベクトルへ写すまさにその方向に沿って 0 になる。

<!--en:9b214841-->
### レイリー商

<!--en:9880eb1d-->
恒等式 :eqref:`eq_mdl-quadform` の役割は符号の判定にとどまらない。この式によって、最大固有値と最小固有値が最適化問題の解として特徴づけられる。対称行列 $\mathbf{A}$ に対し、*レイリー商*を次のように定義する。

<!--en:e0393cc2-->
これは単位長さの二乗あたりの二次形式の値を測る量であり、$\mathbf{x}$ が固有ベクトルのときには $R(\mathbf{x})$ はその固有値に一致する。本項全体を通して、固有値は絶対値ではなく符号付きの値によって $\lambda_1\ge\cdots\ge\lambda_n$ の順に並べる。

<!--en:b3fc5281-->
**命題（Rayleigh）。** *固有値（$\mathbf{A}$ は対称行列）を $\lambda_1\ge\cdots\ge\lambda_n$ とするとき、*

<!--en:5bc1af9b-->
*が成り立ち、最大値と最小値はそれぞれ対応する固有ベクトルにおいて達成される。*

<!--en:0b420877-->
**証明。** $\mathbf{x}$ を定数倍しても $R(\mathbf{x})$ は変わらないため、単位ベクトルに限って考えればよい。$\mathbf{y}=\mathbf{V}^\top\mathbf{x}$ と置くとノルムが保たれ（$\|\mathbf y\|=\|\mathbf x\|=1$）、:eqref:`eq_mdl-quadform` は $R(\mathbf x)=\sum_i\lambda_i y_i^2$ となる。これは非負の重み $y_i^2$（総和は $1$）による固有値の重み付き平均である。いかなる平均も最小値と最大値の間に収まるため、$\lambda_n\le R(\mathbf x)\le\lambda_1$ が成り立つ。$\mathbf x=\mathbf v_1$ または $\mathbf x=\mathbf v_n$ としてすべての重みを単一の座標に集中させれば、上下の限界がそれぞれ達成される。$\blacksquare$

<!--en:34f6061a-->
通しで用いている対称行列 $\mathbf S$ に当てはめると、$R(\mathbf x)$ は $[1,3]$ の範囲を動き、$45^\circ$ の固有ベクトル方向で最大となり、$135^\circ$ の固有ベクトル方向で最小となる。その中間のあらゆる方向では、固有値 $1$ と $3$ が固有座標の二乗に比例して混ぜ合わされる。

<!--en:9fcb735f-->
部分空間全体にわたって最大化すれば、中間のすべての固有値も復元できる。*クーラン・フィッシャーの定理*（Courant–Fischer theorem）によれば、$\lambda_j=\max_{\dim U=j}\ \min_{\mathbf 0\neq\mathbf x\in U}R(\mathbf x)$ は、すべての $j$ 次元部分空間 $U$ にわたって保証されるレイリー商の最大値を与える（演習問題 :exref:`ex_mdl-eigendecomposition-computed-eigenvalue-accuracy`d で証明の手順を追う）。

<!--en:77548198-->
最大・最小固有値の 2 つの応用は、本書全体を通して繰り返し登場する。第 1 に、任意の行列 $\mathbf A$ に対し、$\mathbf A^\top\mathbf A$ のレイリー商の最大値は $\max_{\mathbf x\neq\mathbf 0}\|\mathbf A\mathbf x\|^2/\|\mathbf x\|^2$ であり、これは $\mathbf A$ がベクトルを引き伸ばし得る最大倍率の二乗に等しい。第 2 に、凸二次関数 $f(\mathbf x)=\tfrac12\mathbf x^\top\mathbf A\mathbf x$（$\mathbf A\succ0$）に対する勾配降下法を考えよう。その更新式は $\mathbf x\leftarrow(\mathbf I-\eta\mathbf A)\mathbf x$ である。:eqref:`eq_mdl-eig-power-expansion` により、各固有座標にはステップごとに $1-\eta\lambda_i$ が掛かるため、任意の初期点から反復が収束するための必要十分条件は、$|1-\eta\lambda_i|<1$ がすべての $i$ で成り立つこと、すなわち $0<\eta<2/\lambda_{\max}$ となる。このように最大固有値はステップサイズの上限を定め（これは滑らかさの定数 $L$ として :numref:`sec_mdl-gradient-based-optimization` に登場する）、ステップサイズが $\eta\le1/\lambda_{\max}$ の範囲にあるとき、最小固有値が最も遅い縮小率 $1-\eta\lambda_{\min}$ を決定する。両者の比 $\lambda_{\max}/\lambda_{\min}$ は $\mathbf A$ の条件数（condition number）であり、:numref:`subsec_mdl-condition-number` で任意の行列へと一般化する。

<!--en:5f778ebe-->
### 標本共分散とマルチェンコ・パスツール則

<!--en:1a59b3bd-->
高次元データの中に構造を探すときには、標本共分散行列の大きな固有値を調べる。しかし、どの固有値が本当に大きいのかを判断するには、まずデータにまったく構造がないときに固有値がどう見えるかという基準が必要である。いま $n$ 個の純粋なノイズのサンプルを $d$ 次元空間から抽出し、各座標は独立で平均 $0$、分散 $1$（したがって真の共分散は $\mathbf I$）であるとして、それらを行として並べた行列を $\mathbf X\in\mathbb R^{n\times d}$ としよう。標本共分散 $\mathbf C=\tfrac1n\mathbf X^\top\mathbf X$ の固有値は、$1$ の近くに集まるだろうか。任意の単位ベクトル $\mathbf w$ に対して、レイリー商 $\mathbf w^\top\mathbf C\mathbf w$ はデータを $\mathbf w$ 上へ射影したときの標本分散になる。したがって $\mathbf C$ の最大固有値と最小固有値は、分散が最も大きい方向と最も小さい方向に沿った標本分散そのものであり、主成分分析（principal component analysis）（:numref:`subsec_mdl-pca`）がデータの構造として解釈するまさにその量である。

<!--en:7d4bb3e7-->
驚くべきことに、固有値は $1$ の近くに集まら*ない*。$d,n\to\infty$ の極限を、*アスペクト比* $c=d/n\le1$ を固定したまま考えよう。すると $\mathbf C$ の固有値のヒストグラムは、次の区間を台とする決定論的な密度関数へ収束する。

<!--en:76069afe-->
この区間は*マルチェンコ・パスツールのバルク*（Marchenko–Pastur bulk）として知られており :cite:`Marchenko.Pastur.1967`、ノイズが有限の 4 次モーメントを持つ限り、最大・最小固有値はちょうどその両端へ収束する。$c>1$ のときは行列 $\mathbf C$ のランクが高々 $n<d$ となるため、このバルクに加えて少なくとも $d-n$ 個の固有値が恒等的に 0 になる。1 次元あたり 4 サンプル（$c=1/4$）ある場合でさえ、純粋なノイズの固有値は $[0.25,2.25]$ 全体に大きく広がる。ランダムな揺らぎが最も大きく出た方向の標本分散は真の分散の 2 倍を超え、最も小さい方向では 4 分の 1 にしかならない。$n=4000$、$d=1000$ の規模になれば、この漸近極限はすでに驚くほど正確である。実際に確かめてみよう。

<!--en:96dd23cd-->
何度実行しても、最小固有値と最大固有値は予測された両端 $0.25$ と $2.25$ から 100 分の数以内の範囲に収まる。このバルクの内部に落ちる固有値はどれも純粋なノイズと区別できず、上端を明確に超える固有値だけが真の構造の証拠となる。固有値がバルクを抜け出すにはどれほど強い信号が必要かについては、演習問題 :exref:`ex_mdl-eigendecomposition-spiked-covariance` で調べる。

<!--en:293d28b6-->
本節全体を通して、2 つのことを当然の前提としてきた。行列の固有値を実際に計算できること、そして固有値を見れば、多数回のステップ後だけでなく数ステップ後のべき乗の振る舞いもわかるということである。大規模な行列や、対称からほど遠い行列に対しては、どちらの前提も安全ではない。:numref:`sec_mdl-spectra-in-practice` ではこの両方を取り上げる。

<!--en:30ac03ff-->
## まとめ

<!--en:fbf997b0-->
固有ベクトル座標のもとでは、対角化可能な行列はスカラーの並びとして作用し、$\mathbf A^k\mathbf x=\sum_i c_i\lambda_i^k\mathbf v_i$ によってマルコフ連鎖、深層線形ネットワーク、および二次関数上の勾配降下法における反復適用を明快に説明する。$\mathbb C$ 上で空間全体を張る固有基底が存在するための必要十分条件は、すべての固有値について幾何学的重複度が代数的重複度と一致することである。一致しない場合にはジョルダンブロックが $k$ の多項式因子をもたらして過渡的増大を引き起こし得るほか、複素共役対は不変平面内での回転とスケーリングに対応する。実対称行列はこの両方の落とし穴を免れており、常に実固有値と正規直交固有基底を持つ。その二次形式は重み付き二乗和となるため、固有値の符号が正定値性を決定し、最大・最小固有値はレイリー商の最大値・最小値と一致する。最後に、純粋なノイズから計算した標本共分散の固有値はマルチェンコ・パスツール区間全体にわたって広がるため、バルクの上端を超える固有値だけが真の構造を示す信号となる。

<!--en:2294243a-->
## 演習問題

<!--en:bd59a684-->
1. **トレース、行列式、定値性。** 恒等式 :eqref:`eq_mdl-det-eig` と :eqref:`eq_mdl-tr-eig` は、和と積を通じて $2\times2$ 行列の 2 つの固有値を定める。
    1. 特性多項式を作らずにトレースと行列式だけから $\left(\begin{smallmatrix}4&1\\1&4\end{smallmatrix}\right)$ の固有値を求め、続いて固有ベクトルを求めよ。サーファーの行列 $\mathbf P$（トレースは $5/4$、行列式は $1/4$）についても同様に行え。
    1. $2\times2$ の対称行列が正定値であるための必要十分条件は、そのトレースと行列式がともに正であることを示せ。半正定値性を特徴づけるのは、トレースと行列式に対するどのような符号条件か。
    1. 行列 $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$、$\left(\begin{smallmatrix}1&2\\2&1\end{smallmatrix}\right)$、$\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$ はそれぞれ、二次関数 $f(\mathbf{x})=\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ の臨界点 $\mathbf{0}$ におけるヘッセ行列である。(b) を用いて、各 $f$ について原点が狭義の最小点、狭義でない最小点、鞍点のいずれであるかを判定せよ。鞍点となるものについては、$f$ が減少する方向を 1 つ挙げよ。
    1. コレスキー因子 $\mathbf{L}$（対象となる行列は $\left(\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\right)$）の対角成分が $\sqrt a$ と $\sqrt{(ac-b^2)/a}$ になることを示せ。正の対角成分を持つ因子が存在する条件が (b) の判定条件と一致するのはなぜか。
   :exlabel:`ex_mdl-eigendecomposition-trace-determinant`

<!--en:3db8ce94-->
1. **ランクと固有値 0。** 本節では、対角化可能な行列のランクが重複度込みの非ゼロ固有値の個数に等しいことを示した。
    1. $\mathbf{N}=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$ のランクと固有値を求めよ。対角化可能な行列に対する議論のどの段階が $\mathbf{N}$ では破綻するか。
    1. $g_0$ を固有値 $0$（対象は $n\times n$ 行列 $\mathbf{A}$）の幾何学的重複度とし、ただし $g_0=0$（$0$ が固有値でない場合）とする。任意の正方行列に対して $\operatorname{rank}\mathbf{A}=n-g_0$ が成り立つことを示せ。さらにこれから、ランクは代数的重複度込みで数えた非ゼロ固有値の個数以上であり、等号が成り立つための必要十分条件は固有値 $0$ が欠陥固有値でないことであると結論せよ。
    1. 行列の*特異値*は $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$、すなわち PSD 行列 $\mathbf A^\top\mathbf A$ の固有値の平方根で定義される。$\mathbf{N}$、および :numref:`subsec_mdl-jordan` のジョルダンブロック $\mathbf{J}_m(0)$ の特異値を求めよ。非ゼロ特異値の個数を、ランクおよび非ゼロ固有値の個数と比較せよ。
   :exlabel:`ex_mdl-eigendecomposition-rank-zero-eigenvalues`

<!--en:5aa1e4c1-->
1. **中心化行列。** $\mathbf{1}\in\mathbb{R}^n$ をすべての成分が 1 のベクトルとし、$\mathbf{H}=\mathbf{I}-\tfrac1n\mathbf{1}\mathbf{1}^\top$ と置く。
    1. $n\mathbf{I}-\mathbf{1}\mathbf{1}^\top$ に対し、$\mathbf{x}^\top(n\mathbf{I}-\mathbf{1}\mathbf{1}^\top)\mathbf{x}$ を差 $x_i-x_j$ の二乗和として表すことで、この行列が半正定値であることを示せ。続いて $n\mathbf{I}-\mathbf{1}\mathbf{1}^\top$ のすべての固有値と対応する固有ベクトルを求めよ。
    1. $\mathbf{H}$ が対称行列であり $\mathbf{H}^2=\mathbf{H}$ を満たすことを示し、(a) からその固有値とランクを導け。
    1. $\mathbf{X}\in\mathbb{R}^{n\times d}$ の各行を $n$ 個のサンプルとする。すべての行から標本平均を引くと $\mathbf{H}\mathbf{X}$ となるため、標本共分散は $\mathbf{C}=\tfrac1n\mathbf{X}^\top\mathbf{H}\mathbf{X}$ と書ける。$\mathbf{C}\succeq0$ かつ $\operatorname{rank}\mathbf{C}\le n-1$ を示せ。$512$ 次元の埋め込みについて、$256$ 個の例からなるミニバッチから計算した標本共分散は、少なくとも何個の固有値 0 を持つか。これは :numref:`subsec_mdl-sample-covariance-mp` における $c=d/n>1$ の場合にあたる。この性質により、$\mathbf{C}$ に対するどのような演算が不可能になるか。

   *Gilbert Strang の [MIT 18.065 問題集](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/cd4c57b7e61b8ca9fdd3511a30aca052_MIT18_065S18PSets.pdf)（第 13 回講義の問題 4）を改変。*
   :exlabel:`ex_mdl-eigendecomposition-centering-matrix`

<!--en:be95b9c1-->
1. [code] **計算された固有値の精度。** ● `np.linalg.eigvals` の背後にあるアルゴリズムは後退安定である。すなわち、ある摂動行列を伴う $\mathbf{A}+\mathbf{E}$（ただし $\|\mathbf{E}\|_2$ は $\epsilon_{\textrm{mach}}\|\mathbf{A}\|_2$ と同程度の大きさであり、倍精度では $\epsilon_{\textrm{mach}}\approx2.2\times10^{-16}$）の厳密な固有値を返す :cite:`Golub.Van-Loan.1996`。微小な摂動によって、欠陥固有値は $\varepsilon$ の分数べき乗だけ動き得る。
    1. $\mathbf{e}_1,\ldots,\mathbf{e}_m$ を標準基底ベクトルとする。$\mathbf{J}_m(\lambda)+\varepsilon\,\mathbf{e}_m\mathbf{e}_1^\top$（ジョルダンブロックの左下の隅に $\varepsilon$ を置いた行列）の固有値が $\lambda+\varepsilon^{1/m}\omega$（ただし $\omega$ は 1 の $m$ 乗根全体を動く）となることを示せ。
    1. $m=1,\ldots,4$ に対し、$\mathbf{Q}\mathbf{J}_m(2)\mathbf{Q}^\top$ を構成せよ（ただし直交行列 $\mathbf{Q}$ はガウスランダム行列の QR 分解からとる）。これを $20$ 回試行し、$\max_i|\hat\lambda_i-2|$（$\hat\lambda_i$ は数値計算された固有値）の中央値を記録して $\epsilon_{\textrm{mach}}^{1/m}$ と比較せよ。対称行列 $\mathbf{Q}(2\mathbf{I})\mathbf{Q}^\top$ についても同じ実験を行え。
    1. $\mathbf{A}$ と $\mathbf{E}$ を対称行列とする。:numref:`subsec_mdl-rayleigh` の Rayleigh の命題を用いて、$\lambda_{\max}(\mathbf{A}+\mathbf{E})\le\lambda_{\max}(\mathbf{A})+\|\mathbf{E}\|_2$ および $\lambda_{\min}(\mathbf{A}+\mathbf{E})\ge\lambda_{\min}(\mathbf{A})-\|\mathbf{E}\|_2$ を示せ。これによって (b) の対称行列の結果はどのように説明されるか。
    1. クーラン・フィッシャーの定理 $\lambda_j=\max_{\dim U=j}\ \min_{\mathbf 0\neq\mathbf x\in U}R(\mathbf x)$ を、対称行列 $\mathbf A$（固有値は $\lambda_1\ge\cdots\ge\lambda_n$）に対して証明せよ。上界については $U$ と $\operatorname{span}(\mathbf v_j,\ldots,\mathbf v_n)$（次元は $n-j+1$ であるため、共通部分は少なくとも 1 本の直線を含む）の共通部分をとれ。下界については $U=\operatorname{span}(\mathbf v_1,\ldots,\mathbf v_j)$ とせよ。さらにこれから、(c) をすべての固有値へ拡張するワイルの不等式 $|\lambda_j(\mathbf A+\mathbf E)-\lambda_j(\mathbf A)|\le\|\mathbf E\|_2$（任意の $j$）を導け。
   :exlabel:`ex_mdl-eigendecomposition-computed-eigenvalue-accuracy`

<!--en:cb98fd12-->
1. [code] **バルクを超えるスパイク。** ● :numref:`subsec_mdl-sample-covariance-mp` では、マルチェンコ・パスツールのバルクを純粋なノイズの固有値が広がる範囲として扱った。いまデータに真の方向が 1 つ含まれているとしよう。すなわち $n$ 個のサンプルは独立なガウス座標を持ち、第 1 座標を除くすべての座標の分散が $1$ で、第 1 座標の分散だけが $1+\beta$ であるとする。$n\to\infty$ とし $c=d/n$ を固定した極限において、:citet:`Baik.Silverstein.2006` は $\mathbf{C}=\tfrac1n\mathbf{X}^\top\mathbf{X}$ の最大固有値が次へ収束することを示した。

   $$
   \lambda_{\max}\to\begin{cases}(1+\sqrt c)^2, & \beta\le\sqrt c,\\
   (1+\beta)\left(1+c/\beta\right), & \beta>\sqrt c.\end{cases}
   $$

    1. $n=2000$、$d=500$ とし、$\mathbf{C}$ の最大固有値を $\beta\in\{0,0.3,0.5,0.7,1,2\}$ のそれぞれについて 5 回の試行の平均として求め、理論予測と比較せよ。
    1. $d=500$、$\beta=0.4$ に固定する。上の公式を用いて、最大固有値がバルクを抜け出すサンプル数 $n$ の閾値を求めよ。次にサンプル数を $n\in\{500,1000,2000,4000,8000\}$ と増やしながら、最大固有値をバルクの上端、理論予測、および $\beta=0$（同じ $n$）に対する最大固有値と比較せよ。$n=4000$ では分離が見えにくいのはなぜか。
    1. 実務者が、固有値がバルクの上端を超える主成分だけを残すとする。$c=1/4$ においてある方向が残されるには、その方向の分散はどれほど大きくなければならないか。また、他の方向より分散が $10\%$ 大きい方向を検出するには、1 次元あたり何サンプル必要か。
   :exlabel:`ex_mdl-eigendecomposition-spiked-covariance`

<!--en:74216e99-->
[ディスカッション](https://d2l.discourse.group/t/411)

<!--en:91071da4-->
[ディスカッション](https://d2l.discourse.group/t/1086)

<!--en:d25e7816-->
[ディスカッション](https://d2l.discourse.group/t/1087)

<!--en:d25e7816-2-->
[ディスカッション](https://d2l.discourse.group/t/1087)

<!--en:0ad021f7-->
固有値分解<br>**固有ベクトル・対角化と行列のべき乗・欠陥スペクトルと複素スペクトル・スペクトル定理・二次形式**

<!--en:07e092c7-->
::: {.slide title="行列の反復適用：P⁵⁰ はどこへ行き着くか？"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:51515f10-->
- **機械学習における線形変換の反復：**深層線形ネットワークは $\mathbf{W}^{50}\mathbf{x}$ を適用し、二次関数上の勾配降下法は誤差を $(\mathbf{I} - \eta\mathbf{H})^k$ で更新し、マルコフ連鎖は遷移行列 $\mathbf{p}_{k+1} = \mathbf{P}\mathbf{p}_k$ を反復する

<!--en:b931c8b8-->
- **2 ページのランダムサーファー：**ページ 1 からは確率 $1/2$ ずつでページ 1 または 2 へ進み、ページ 2 からは確率 $1/4$ で 1 に戻り、確率 $3/4$ で留まる
  $$\mathbf{P} = \begin{bmatrix}1/2 & 1/4 \\ 1/2 & 3/4\end{bmatrix}$$

<!--en:a5bc4ceb-->
- **急速な収束：**$\mathbf{p}_0 = (1, 0)$ から始めると、反復列は $(0.5, 0.5)$ $\to$ $(0.375, 0.625)$ $\to$ $(0.344, 0.656)$ と推移する。固有ベクトルは極限と正確な収束率の両方を説明する

<!--en:d1b9821f-->
::: {.slide title="固有ベクトル：行列が伸縮のみを施す方向"}

<!--en:55623f5d-->
[基本定義]{.kicker}

<!--en:16d1c86a-->
- **固有対の方程式（$\mathbf{v} \neq \mathbf{0}$）：**ゼロでないベクトル $\mathbf{v}$ が $\mathbf{A}$ の*固有ベクトル*（*固有値*は $\lambda$）であるとは、$\mathbf{A}$ が $\mathbf{v}$ に純粋なスカラー倍として作用することをいう
  $$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

<!--en:5700a82a-->
- **サーファー行列の固有対：**$\mathbf{P}(1, 2) = 1\cdot(1, 2)$（定常方向、$\lambda_1 = 1$）と $\mathbf{P}(1, -1) = \tfrac{1}{4}(1, -1)$（減衰する誤差方向、$\lambda_2 = 1/4$）

<!--en:66b1f7ef-->
- **回転しない軸と最大伸縮軸：**固有ベクトルは*回転しない*方向を表す。変換後の単位円が描く楕円の軸とは対称行列では一致するが、一般の行列では一致しない

<!--en:fc67d781-->
::: {.slide title="det(A − λI) = 0 による固有値の求め方"}

<!--en:a7d96a6e-->
[特性多項式]{.kicker}

<!--en:9220199f-->
- **特異性の条件：**$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ を $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$ と書き換えると、ゼロでない固有ベクトル $\mathbf{v}$ が存在することと $\mathbf{A} - \lambda\mathbf{I}$ が特異であることは同値になる
  $$p(\lambda) = \det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

<!--en:9ed3f91b-->
- **$2 \times 2$ サーファー行列の計算例：**$\det(\mathbf{P} - \lambda\mathbf{I}) = (\tfrac{1}{2}-\lambda)(\tfrac{3}{4}-\lambda) - \tfrac{1}{8} = \lambda^2 - \tfrac{5}{4}\lambda + \tfrac{1}{4} = (\lambda - 1)(\lambda - \tfrac{1}{4})$ より、根は $\lambda_1 = 1$ と $\lambda_2 = 1/4$

<!--en:7a873051-->
- **固有空間の算出：**$\lambda_1 = 1$ を $(\mathbf{P} - \mathbf{I})\mathbf{v} = \mathbf{0}$ に代入して $-\tfrac{1}{2}v_1 + \tfrac{1}{4}v_2 = 0 \implies \mathbf{v}_1 = (1, 2)$、$\lambda_2 = 1/4$ を代入して $\tfrac{1}{4}v_1 + \tfrac{1}{4}v_2 = 0 \implies \mathbf{v}_2 = (1, -1)$ を得る

<!--en:5c8470bc-->
::: {.slide title="ライブラリが実際に固有値を計算する方法"}

<!--en:6b28b2b8-->
[数値的な固有値ソルバ]{.kicker}

<!--en:dc9d14ac-->
- **$p(\lambda) = 0$ を直接解かない：**$n \ge 5$ ではべき根による閉じた公式が存在せず、多項式の根は係数の丸め誤差に対して破滅的なほど敏感になる

<!--en:827cdce7-->
- **反復的な直交相似変換（QR・シューア）：**ソルバは $\mathbf{A}_{k+1} = \mathbf{Q}_k^\top\mathbf{A}_k\mathbf{Q}_k$ を反復適用し、固有値を保ったまま非対角成分を 0 へ追い込む

<!--en:b023b24c-->
- **三角行列からの読み取り：**上三角行列になれば、$\det(\mathbf{U} - \lambda\mathbf{I}) = \prod_i (u_{ii} - \lambda)$ よりすべての固有値が対角線上に並ぶ

<!--en:91f67ff8-->
::: {.slide title="対角化は行列のべき乗をスカラーに帰着させる"}

<!--en:68039e91-->
[対角化]{.kicker}

<!--en:75bae96a-->
- **固有基底による分解（$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$）：**$n$ 個の線形独立な固有ベクトルを $\mathbf{V}$ に並べると $\mathbf{A}\mathbf{V} = \mathbf{V}\boldsymbol{\Lambda}$ となる

<!--en:b4aabba2-->
- **べき乗の打ち消し合い：**内側の $\mathbf{V}^{-1}\mathbf{V} = \mathbf{I}$ が消えて $\mathbf{A}^k = \mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1}$ となり、$\mathbf{A}^k\mathbf{x} = \sum_{i=1}^n c_i\lambda_i^k\mathbf{v}_i$（$\mathbf{c} = \mathbf{V}^{-1}\mathbf{x}$）を得る

<!--en:c67927b8-->
- **サーファーの閉じた解：**$\mathbf{p}_k = \tfrac{1}{3}(1, 2) + \tfrac{2}{3}(\tfrac{1}{4})^k(1, -1) \to (1/3, 2/3)$ へ収束し、クリックごとに誤差が $4$ で割られる

<!--en:dd43fd2c-->
::: {.slide title="閉じた形の固有展開の検証"}

<!--en:5efedb90-->
[コードで見る対角化]{.kicker}

<!--en:4b09123a-->
50 回の行列積 $\mathbf{P}^{50}\mathbf{p}_0$ とスカラーのべき乗公式 $\sum_i c_i \lambda_i^{50}\mathbf{v}_i$ は計算機精度まで一致し、ステップごとに誤差が $4$ 分の 1 になる。

<!--en:e940118e-->
**一般の行列関数：**まったく同じ固有基底のレシピによって、逆行列 $\mathbf{A}^{-1} = \mathbf{V}\boldsymbol{\Lambda}^{-1}\mathbf{V}^{-1}$（すべての $\lambda_i \neq 0$ のとき）や、連続時間の微分方程式を解く行列指数関数 $e^{\mathbf{A}t} = \mathbf{V}e^{\boldsymbol{\Lambda}t}\mathbf{V}^{-1}$（$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$ に対する解）が定まる

<!--en:e9430aeb-->
::: {.slide title="相似不変量：トレースと行列式"}

<!--en:4aaadf4b-->
[相似と不変量]{.kicker}

<!--en:df7b6c5b-->
- **基底からの独立性：**相似な行列 $\mathbf{A}' = \mathbf{B}\mathbf{A}\mathbf{B}^{-1}$ は、まったく同じ特性多項式 $\det(\mathbf{A} - \lambda\mathbf{I}) = \prod_{i=1}^n (\lambda_i - \lambda)$ を共有する

<!--en:a9366444-->
- **トレースと行列式の恒等式：**すべての正方行列に対して $\operatorname{tr}\mathbf{A} = \sum_{i=1}^n \lambda_i$ と $\det\mathbf{A} = \prod_{i=1}^n \lambda_i$ が成り立つ

<!--en:4ec40e25-->
- **$2 \times 2$ 行列の近道：**$p(\lambda) = \lambda^2 - (\operatorname{tr}\mathbf{A})\lambda + \det\mathbf{A} = 0$

<!--en:57c0eb7c-->
::: {.slide title="相異なる固有値は独立性を保証する"}

<!--en:9852fda4-->
[重複度と対角化可能性]{.kicker}

<!--en:7aac4cfe-->
- **相異なる固有値 $\implies$ 線形独立な固有ベクトル：**最小の従属関係に $\mathbf{A} - \lambda_m\mathbf{I}$ を適用する証明から、$n$ 個の相異なる固有値があれば常に $\mathbf{A}$ を対角化できる

<!--en:76573358-->
- **重複度の不等式（$1 \le g(\lambda) \le a(\lambda)$）：**幾何学的重複度 $g(\lambda) = \dim\ker(\mathbf{A} - \lambda\mathbf{I})$ が根の重複度 $a(\lambda)$ を超えることはない

<!--en:989bd8ce-->
- **対角化可能性の判定条件：**$\mathbf{A}$ が $\mathbb{C}$ 上で対角化可能である必要十分条件は、$g(\lambda) = a(\lambda)$ がすべての固有値 $\lambda$ で成り立つこと

<!--en:6da75ed7-->
::: {.slide title="欠陥行列：せん断は固有基底を持たない"}

<!--en:230650df-->
[対角化が破綻するとき]{.kicker}

<!--en:4f08bac6-->
- **重根（2 重の $\lambda = 1$）：**水平せん断 $\mathbf{T} = \left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ では $\det(\mathbf{T} - \lambda\mathbf{I}) = (1 - \lambda)^2 = 0$ となり、代数的重複度は $a(1) = 2$

<!--en:78bf7c71-->
- **単一の固有方向（$g(1) = 1$）：**$(\mathbf{T} - \mathbf{I})\mathbf{v} = \left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)\mathbf{v} = \mathbf{0}$ を解くと $v_2 = 0$ となり、水平軸 $\operatorname{span}\{(1, 0)\}$ だけが残る

<!--en:d8a2e493-->
- **べき乗のもとでの線形ドリフト：**$g(1) < a(1)$ のため $\mathbf{T}$ は*欠陥行列*であり、べき乗 $\mathbf{T}^k = \left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$ は $k$ に関して線形に増大する（$\lambda = 1$ にもかかわらず）

<!--en:227feb56-->
::: {.slide title="ジョルダンブロックは λᵏ に k の多項式を掛ける"}

<!--en:0f2d91f9-->
[ジョルダン標準形]{.kicker}

<!--en:abd93422-->
- **ジョルダンブロック（$\mathbf{J}_m(\lambda) = \lambda\mathbf{I} + \mathbf{N}$）：**べき零シフト $\mathbf{N}^m = \mathbf{0}$ により $\mathbf{J}_2(\lambda)^k = \lambda^k\mathbf{I} + k\lambda^{k-1}\mathbf{N}$ となる

<!--en:cd5fe572-->
- **過渡的増幅：**$\lambda = 0.99$ では、$k(0.99)^{k-1}$ が $37.0$（$k = 100$）まで上昇してから減衰に転じる

<!--en:123cc1fe-->
::: {.slide title="ジョルダンブロックは λᵏ に k の多項式を掛ける（続き）"}

<!--en:0f2d91f9-2-->
[ジョルダン標準形]{.kicker}

<!--en:b8d52b9d-->
::: {.slide title="複素固有値は回転とスケーリングを表す"}

<!--en:7a8e3bc2-->
[実行列における複素スペクトル]{.kicker}

<!--en:98e9804b-->
- **実不変平面：**$\mathbf{A}(\mathbf{u} + i\mathbf{v}) = (a + ib)(\mathbf{u} + i\mathbf{v})$ を実部と虚部に分けると、$\mathbf{B}^{-1}\mathbf{A}\mathbf{B} = r\mathbf{R}_\theta$（基底 $\mathbf{B} = [\mathbf{v}\;\;\mathbf{u}]$ のもと）を得る

<!--en:7ad9bf2b-->
- **絶対値と偏角：**$r = \sqrt{a^2 + b^2}$ 倍にスケールし、$\theta = \arg(a + ib)$ だけ回転する

<!--en:c1f34002-->
- **通しで用いる行列 $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$：**$\lambda = 2 \pm i = \sqrt{5}\,e^{\pm i\,26.6^\circ}$

<!--en:1da9503e-->
::: {.slide title="通しで用いる行列 A が描く楕円スパイラル"}

<!--en:241031c3-->
[コードで見る複素スペクトル]{.kicker}

<!--en:001bb5f3-->
複素固有ベクトルの実部と虚部を取り出すと $\mathbf{A}$ は厳密なブロック $\left(\begin{smallmatrix}2&-1\\1&2\end{smallmatrix}\right)$ に変換されるが、標準座標では 1 ステップごとのノルム比は一様にならない。

<!--en:a23ab865-->
::: {.slide title="通しで用いる行列 A が描く楕円スパイラル（続き）"}

<!--en:241031c3-2-->
[コードで見る複素スペクトル]{.kicker}

<!--en:23094045-->
**1 ステップごとの比が $1.4$〜$3.5$ で揺れる理由：**回転・スケーリングが一様になるのは直交しない基底 $(\mathbf{v}, \mathbf{u})$ のもとだけであり、標準座標では反復列が傾いた楕円に沿って渦を巻き、幾何平均の増大率が $\sqrt{5} \approx 2.236$ に落ち着く

<!--en:ae095b18-->
::: {.slide title="実正方行列の 3 つの基本動作"}

<!--en:52fed4ce-->
[実ジョルダン標準形のまとめ]{.kicker}

<!--en:a5f7d86b-->
- **1. 相異なる実固有値：**実軸に沿った純粋な指数的スケーリング $\lambda_i^k$（サーファー行列 $\mathbf{P}$、対称行列 $\mathbf{S}$）

<!--en:8d9b3577-->
- **2. 欠陥ジョルダンブロック：**多項式×指数 $k\lambda^{k-1}$ によるドリフトや過渡的増大（せん断 $\mathbf{T}$、$\mathbf{J}_2(0.99)$）

<!--en:366a9ee5-->
- **3. 複素共役対：**不変 2 次元平面内での回転・スケーリング $r^k\mathbf{R}_{k\theta}$（通しで用いる行列 $\mathbf{A}$）

<!--en:31c29a8e-->
::: {.slide title="実対称行列に対するスペクトル定理"}

<!--en:c572779f-->
[対称行列]{.kicker}

<!--en:934f22cc-->
- **スペクトル定理（$\mathbf{A} = \mathbf{A}^\top \implies \mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$）：**実固有値 $\boldsymbol{\Lambda}$ と正規直交固有ベクトル $\mathbf{V}^\top\mathbf{V} = \mathbf{I}$ を持つ

<!--en:2831ea0a-->
- **証明 (i) 実固有値 $\lambda$：**$\lambda\|\mathbf{v}\|^2 = \mathbf{v}^*\mathbf{A}\mathbf{v} = \bar{\lambda}\|\mathbf{v}\|^2 \implies \lambda \in \mathbb{R}$

<!--en:c65b9c45-->
- **証明 (ii) 直交する $\mathbf{v}_i$：**$(\lambda - \mu)\mathbf{u}^\top\mathbf{v} = 0 \implies \mathbf{u} \perp \mathbf{v}$（$\lambda \neq \mu$ のとき）

<!--en:15632d1d-->
::: {.slide title="スペクトル定理の帰納法による証明と幾何"}

<!--en:c572779f-2-->
[対称行列]{.kicker}

<!--en:061efea5-->
- **証明 (iii) — 直交補空間に関する帰納法：**単位固有ベクトル $\mathbf{v}_1$（$\mathbf{A}$ の固有ベクトル）に対し、対称性より $(n-1)$ 次元部分空間 $U = \mathbf{v}_1^\perp$ は不変に保たれる（$\mathbf{x} \perp \mathbf{v}_1 \implies (\mathbf{A}\mathbf{x})^\top\mathbf{v}_1 = \mathbf{x}^\top(\mathbf{A}\mathbf{v}_1) = \lambda_1(\mathbf{x}^\top\mathbf{v}_1) = 0$）

<!--en:dc98e144-->
- **$U$ への制限：**正規直交基底 $\mathbf{Q} \in \mathbb{R}^{n \times (n-1)}$（$U$ の基底）をとると、より小さい $(n-1) \times (n-1)$ 対称行列 $\mathbf{C} = \mathbf{Q}^\top\mathbf{A}\mathbf{Q}$ が得られ、その正規直交固有基底から $\mathbf{V} = [\mathbf{v}_1\;\;\mathbf{Q}\mathbf{V}']$ が完成する

<!--en:3a880094-->
- **回転・伸縮・逆回転の幾何：**$\mathbf{V}^{-1} = \mathbf{V}^\top$ であるため、対称行列 $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ の適用は、正規直交固有軸を標準軸へ回転させ（$\mathbf{V}^\top$）、各軸を $\lambda_i$ 倍に伸縮し（$\boldsymbol{\Lambda}$）、再び回転して戻す（$\mathbf{V}$）操作になる

<!--en:de6b2b71-->
- **対称行列 $\mathbf{S} = \left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ の具体例：**単位固有ベクトル $(1, 1)/\sqrt{2}$（$\lambda_1 = 3$）と $(1, -1)/\sqrt{2}$（$\lambda_2 = 1$）は互いに直交し、出力楕円の主軸とそのまま一致する

<!--en:328c3b8e-->
::: {.slide title="二次形式と正定値性"}

<!--en:12555027-->
[二次形式]{.kicker}

<!--en:38094e0a-->
- **固有座標における二乗和：**$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ と $\mathbf{y} = \mathbf{V}^\top\mathbf{x}$ を代入すると、任意の対称二次形式は独立な二乗和に分離される
  $$\mathbf{x}^\top\mathbf{A}\mathbf{x} = \mathbf{y}^\top\boldsymbol{\Lambda}\mathbf{y} = \sum_{i=1}^n \lambda_i\,y_i^2 = \sum_{i=1}^n \lambda_i (\mathbf{v}_i^\top\mathbf{x})^2$$

<!--en:472b6319-->
- **固有値の符号による定値性の判定：**
  - *正定値*（$\mathbf{A} \succ 0$、お椀型）：$\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0$、ただし任意の $\mathbf{x} \neq \mathbf{0} \iff$ すべての $\lambda_i > 0$
  - *半正定値*（$\mathbf{A} \succeq 0$、谷樋型）：$\mathbf{x}^\top\mathbf{A}\mathbf{x} \ge 0$、任意の $\mathbf{x} \iff$ すべての $\lambda_i \ge 0$
  - *不定値*（鞍型）：固有値 $\lambda_i$ の符号が混在

<!--en:bb2fce15-->
::: {.slide title="グラム行列、ヘッセ行列、コレスキー判定"}

<!--en:1e9896d3-->
[機械学習における正定値性]{.kicker}

<!--en:1337be84-->
- **深層学習で PD・PSD 行列が現れる場面：**
  - *グラム行列と共分散行列：*任意の $\mathbf{X}$ に対して $\mathbf{x}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{x} = \|\mathbf{X}\mathbf{x}\|^2 \ge 0$ より $\mathbf{X}^\top\mathbf{X} \succeq 0$ であり、$\mathbf{X}^\top\mathbf{X} \succ 0 \iff \mathbf{X}$ が列フルランク
  - *狭義の極小値：*臨界点 $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ において、PD のヘッセ行列 $\nabla^2 f(\mathbf{x}^\star) \succ 0$ は二次テイラー展開の項を下から $\tfrac{1}{2}\lambda_{\min}\|\boldsymbol{\delta}\|^2 > 0$ で抑える
  - *コレスキー分解（$\mathbf{A} = \mathbf{L}\mathbf{L}^\top$）：*対角成分が正の下三角コレスキー因子 $\mathbf{L}$ の存在判定は $\mathbf{A} \succ 0$ の最速な数値チェックであり、相関を持つガウスサンプル $\mathbf{L}\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{A})$ も生成する

<!--en:ba99387b-->
::: {.slide title="レイリー商は二次形式の伸縮範囲を定める"}

<!--en:d2aaf5d0-->
[変分的特徴づけ]{.kicker}

<!--en:89916bcc-->
- **固有値の凸平均：**単位ベクトル $\mathbf{x}$（$\mathbf{y} = \mathbf{V}^\top\mathbf{x}$）に対し、$R(\mathbf{x}) = \frac{\mathbf{x}^\top\mathbf{A}\mathbf{x}}{\mathbf{x}^\top\mathbf{x}} = \sum_{i=1}^n \lambda_i y_i^2 \in [\lambda_{\min}, \lambda_{\max}]$

<!--en:16caa032-->
- **最大伸縮率：**$\mathbf{A}^\top\mathbf{A}$ に対し、$\max_{\|\mathbf{x}\|=1}\|\mathbf{A}\mathbf{x}\|^2 = \lambda_{\max}(\mathbf{A}^\top\mathbf{A}) = \|\mathbf{A}\|_2^2$

<!--en:5a86c1fe-->
- **勾配降下法の収束限界：**$\tfrac{1}{2}\mathbf{x}^\top\mathbf{A}\mathbf{x}$ 上では、$|1 - \eta\lambda_i| < 1$ のために $0 < \eta < 2/\lambda_{\max}$ が必要

<!--en:8604b65e-->
::: {.slide title="標本共分散のノイズ：マルチェンコ・パスツール則"}

<!--en:05948379-->
[高次元統計]{.kicker}

<!--en:23b0ae88-->
- **マルチェンコ・パスツールのバルク：**$n$ 個の純粋なノイズサンプル（$\mathbb{R}^d$ 上、$c = d/n$）では、$\mathbf{C} = \tfrac{1}{n}\mathbf{X}^\top\mathbf{X}$ の固有値が $[(1 - \sqrt{c})^2, (1 + \sqrt{c})^2]$ に広がる（$[0.25, 2.25]$、$c = 1/4$ のとき）

<!--en:96172acd-->
- **信号の検出：**$(1 + \sqrt{c})^2$ を超えて抜け出す固有値だけが真のデータ構造を反映する

<!--en:ccc614e8-->
::: {.slide title="概念チェック：固有値、べき乗、定値性"}

<!--en:26a317d8-->
[概念チェック]{.kicker}

<!--en:097cb772-->
1. **トレースと行列式からの固有値：**ある $2 \times 2$ 行列が $\operatorname{tr}\mathbf{A} = 8$ かつ $\det\mathbf{A} = 15$ を満たす。その固有値は何か。

<!--en:2ec0b02b-->
2. **正定値性を判定する 2 つの方法：**$p(\lambda)$ の根を計算せずに、$\mathbf{S} = \left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ が狭義に正定値であることの独立な証明を 2 つ挙げよ。

<!--en:0e8baef2-->
3. **$d > n$ におけるランク落ち：**$512 \times 512$ の標本共分散行列 $\mathbf{C}$ を、中心化された埋め込みからなるサイズ $n = 256$ のミニバッチから計算するとき、$\mathbf{C}$ は少なくとも何個の固有値 0 を持つか。

<!--en:349c6fac-->
**解答：**
**1.** $\lambda^2 - 8\lambda + 15 = (\lambda - 3)(\lambda - 5) = 0$ の根より、$\lambda_1 = 5$ と $\lambda_2 = 3$。
**2.** (i) $\operatorname{tr}\mathbf{S} = 4 > 0$ かつ $\det\mathbf{S} = 3 > 0$ より、2 つの実固有値はともに正である。(ii) 平方完成すると、$\mathbf{x}^\top\mathbf{S}\mathbf{x} = x_1^2 + x_2^2 + (x_1 + x_2)^2 > 0$ がすべての $\mathbf{x} \neq \mathbf{0}$ に対して成り立つ。
**3.** $256$ 個の行を中心化するとランクは高々 $n - 1 = 255$ となるため、$\mathbf{C}$ は少なくとも $512 - 255 = 257$ 個の固有値 0 を持ち、正則化なしには逆行列を求められない。

<!--en:b878817b-->
::: {.slide title="まとめ：固有値分解"}

<!--en:c5a222f8-->
[重要ポイント]{.kicker}

<!--en:4617f689-->
| 概念 | 数式表現 | 主な性質・深層学習での役割 |

<!--en:b33c887e-->
| **固有対と特性多項式** | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$、$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ | $\lambda$ 倍にスケールされる不変方向。$\operatorname{tr}\mathbf{A} = \sum_i \lambda_i$ かつ $\det\mathbf{A} = \prod_i \lambda_i$ |

<!--en:bbd9fc94-->
| **対角化とべき乗** | $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$、$\mathbf{A}^k\mathbf{x} = \sum_i c_i \lambda_i^k \mathbf{v}_i$ | 存在条件は $g(\lambda) = a(\lambda)$ がすべての $\lambda$ で成り立つこと。$k$ ステップのダイナミクスをスカラーのべき乗へ分離 |

<!--en:69f87fcb-->
| **欠陥行列と複素固有値** | $\mathbf{J}_2(\lambda)^k = \lambda^k\mathbf{I} + k\lambda^{k-1}\mathbf{N}$、$\lambda = r e^{\pm i\theta}$ | 固有ベクトルの不足は $k\lambda^{k-1}$ の過渡的増大を起こし、複素共役対は 2 次元平面を回転・スケールする |

<!--en:ce6541e9-->
| **スペクトル定理** | $\mathbf{A} = \mathbf{A}^\top \implies \mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ | すべての実対称行列は実固有値と正規直交固有基底を持つ（$\mathbf{V}^{-1} = \mathbf{V}^\top$） |

<!--en:4ce070cf-->
| **二次形式とレイリー商** | $\mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_i \lambda_i y_i^2$、$R(\mathbf{x}) \in [\lambda_{\min}, \lambda_{\max}]$ | $\mathbf{A} \succ 0 \iff$ すべての $\lambda_i > 0$。勾配降下法のステップサイズ上限 $\eta < 2/\lambda_{\max}$ を規定 |

<!--en:f8d35536-->
| **マルチェンコ・パスツールのバルク** | $\lambda(\tfrac{1}{n}\mathbf{X}^\top\mathbf{X}) \in [(1-\sqrt{c})^2, (1+\sqrt{c})^2]$ | 純粋なノイズの標本固有値は $c = d/n$ に応じて広がり、$(1+\sqrt{c})^2$ を超えるスパイクのみが信号 |
