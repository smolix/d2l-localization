<!--en:7902ddfd-->
# 関連文献

<!--en:45df84d7-->
本節では、本章で展開した線形代数を数値解析、ランダム行列理論、および深層学習の応用へと結びつけ、続いて本章の中核的な内容を網羅する厳選された教科書と講義の一覧を示す。主な関心が深層学習にある場合は、まずランダム射影（random projection）、反復ソルバ、ランダム行列と訓練可能な深さ、およびスケッチングから読み始め、その後に低ランクアテンションとスペクトルノルムオプティマイザを扱う最後の 2 項目へ進むとよい。本書の他の箇所は本節の内容に一切依存しない。

<!--en:699aefbb-->
## 幾何学と線形変換

<!--en:73d04265-->
### 他の内積：カーネル法

<!--en:82d3bfd8-->
:numref:`subsec_mdl-nearest-mean` の最近傍平均規則では、テスト画像を 1 回のドット積によって 2 つのクラス平均と比較して分類するため、決定境界は必然的に平坦な超平面になる。しかし、T シャツとシャツを見分ける場合のように、クラスが湾曲していたり複数のクラスタに分かれていたりすると、この単純な仕組みは破綻する。カーネル法（kernel method）は、まったく同じ幾何学的規則を保ったまま内積だけを変更する。すなわち、$\mathbf{x}\cdot\mathbf{x}_i$ を、別の（場合によっては無限次元の）特徴空間における真の内積に対応する類似度関数 $k(\mathbf{x},\mathbf{x}_i)$ に置き換えるのである。するとクラス平均はその隠れた空間での平均となり、各訓練例が重み $k(\mathbf{x},\mathbf{x}_i)$ で投票するようになり、結果として得られる境界は元の入力空間内で自在に曲がることができる。どの訓練点が実際に非ゼロの重みを必要とするかを問うと、そのままサポートベクターマシンへと至る。:citet:`Scholkopf.Smola.2002` はまさにこの分類器から筆を起こし、そこからカーネル法の理論全体を築き上げている。

<!--en:d94e6320-->
### ランダム射影

<!--en:ce2bf907-->
:numref:`sec_mdl-geometry-linear-algebraic-ops` では、高次元空間 $\mathbb{R}^d$ におけるランダムな方向どうしが互いにほぼ直交することを見た。驚くべきことに、この幾何学的な性質を利用すると、データを一切見ることなくデータセットの次元を縮小できる。Johnson–Lindenstrauss の補題 :cite:`Johnson.Lindenstrauss.1984` によれば、任意の $n$ 個の点からなる集合に対し、適切にスケーリングしたランダムガウス行列を掛けて $k$ 次元（オーダーは $\varepsilon^{-2}\log n$）へ写すと、$\binom{n}{2}$ 個すべてのペア間距離が高い確率で $1\pm\varepsilon$ 倍の範囲内に保たれる。主成分分析（:numref:`subsec_mdl-pca`）とは異なり、ランダム射影はデータにまったく依存せず、目標次元 $k$ は外囲次元（ambient dimension）$d$ にすら依存しない。したがって、この手法が最も威力を発揮するのは、$d$ が $\varepsilon^{-2}\log n$ よりはるかに大きいときである。:citet:`Dasgupta.Gupta.2003` は明示的な定数を伴う驚くほど短い証明を与えており、一方で :citet:`Vershynin.2018` は確率ベクトル（random vector）の集中に関するより広い理論の中にこの補題を位置づけている。

<!--en:ae60100d-->
### 大規模な連立一次方程式の解法

<!--en:2b99d2ca-->
:numref:`subsec_mdl-numerical-issues` では、LU 分解によって連立一次方程式を解き、大規模な疎行列（sparse matrix）の逆行列は一般に密になることを確認した。病的な最悪ケースの行列が存在するにもかかわらず、部分ピボット選択付きガウスの消去法（Gaussian elimination with partial pivoting）が浮動小数点演算でこれほど高い信頼性をもって機能するのはなぜだろうか。:citet:`Trefethen.Bau.1997` は最悪ケースと実用上の安定性の間にあるこの乖離を明快に説明しており、:citet:`Higham.2002` は丸め誤差（rounding error）に関する決定版の解析を与えている。

<!--en:aaacf8de-->
大規模な問題では、疎行列ソルバは決して $\mathbf{A}^{-1}$ を作らず、密な LU 因子の構築すら避けることが多い。疎行列の直接法（direct method）は、消去の過程で新たに生じる非ゼロ要素ができるだけ少なくなるよう、未知数の順序を注意深く並べ替える。反復法（iterative method）はさらに一歩進め、べき乗法（:numref:`subsec_mdl-power-iteration`）とまったく同様に、行列・ベクトル積 $\mathbf{A}\mathbf{x}$ のみを通じて行列にアクセスする。$\mathbf{A}$ が対称正定値であるとき、:citet:`Hestenes.Stiefel.1952` の共役勾配法（conjugate gradient method）は、$k$ ステップ後の誤差を高々 $2\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^k$ 倍（$\mathbf{A}$ が誘導するノルムのもとで）にまで縮小する。ここで、$\kappa$ は :numref:`subsec_mdl-condition-number` で導入した条件数である。:citet:`Golub.Van-Loan.1996` は共役勾配法とその関連手法を詳細に扱っている。

<!--en:e7fb8fcf-->
### 縮約順序とテンソル分解

<!--en:f7838f1a-->
`einsum` 記法（:numref:`subsec_mdl-einsum`）は複数のテンソルを縮約した数学的結果を指定するが、ペアごとの縮約をどの順序で実行するかは定めない。この選択によって計算量は劇的に変わり得る。たとえば $n\times n$ 行列に対して、$(\mathbf{A}\mathbf{B})\mathbf{x}$ の計算には約 $n^3$ 回の乗算を要するが、$\mathbf{A}(\mathbf{B}\mathbf{x})$ ならばわずか $2n^2$ 回で済む。多くの因子が関わる場合、最も計算コストの低い順序を選ぶ問題は括弧の付け方の探索になり、`np.einsum(..., optimize=True)` や `opt_einsum` パッケージ :cite:`Smith.Gray.2018` がこれを自動化する。

<!--en:efaab0e0-->
低ランク近似（low-rank approximation）の考え方は行列を超えて高階テンソルへも広がるが、そこには厳しい落とし穴がある。3 元テンソルは、ランク 1 の外積 $\mathbf{a}_i\otimes\mathbf{b}_i\otimes\mathbf{c}_i$ の和（CP 分解）、小さなコアテンソルに各モードに沿って因子行列を掛けた形（Tucker 分解）、あるいは 3 元コアの連鎖（:citet:`Oseledets.2011` のテンソルトレイン）として表せる。しかし残念ながら、テンソル一般に対しては Eckart–Young の定理ほど明快な結果は成り立たない。CP ランクの決定は NP 困難であり、与えられた CP ランクでの最良近似が存在しないことさえある。CP 分解と Tucker 分解およびそれらのアルゴリズムについては、:citet:`Kolda.Bader.2009` のサーベイが標準的な案内書である。

<!--en:058d0ee7-->
## 固有値とスペクトル

<!--en:e07275d5-->
### 実用上の固有値計算

<!--en:192071c2-->
5 次以上の一般の多項式にはべき根による解の公式が存在しないため、実用的な固有値ソルバはすべて反復計算に頼らざるを得ない。さらに悪いことに、多項式の根はその係数に含まれる微小な丸め誤差に対して極端に敏感になり得るため、実用ライブラリが :numref:`subsec_mdl-finding-eigenvalues` の特性多項式（characteristic polynomial）を陽に構成することもない。その代わり、`eig` はまず直交相似変換を適用し、$O(n^3)$ 回の演算で行列を Hessenberg 形（第 1 劣対角線より下がすべて 0 の形）へと簡約する。続いてシフト付き QR 法（shifted QR algorithm）を実行する。すなわち、$\mathbf{H}-\mu\mathbf{I}=\mathbf{Q}\mathbf{R}$ と分解し、$\mathbf{R}\mathbf{Q}+\mu\mathbf{I}$ を作り、これを繰り返す。各ステップは相似変換であるため固有値は一切変化しない一方、劣対角成分は 0 に向かって縮小し、対角線上にスペクトルが現れる（実行列の複素共役ペアは $2\times 2$ の対角ブロックに落ち着く）。対称行列では Hessenberg 形が三重対角になるため、`eigh` はこの構造を利用して大幅な高速化を実現している。:citet:`Trefethen.Bau.1997` は部分空間上のべき乗法から QR 法を明快に導出しており、:citet:`Golub.Van-Loan.1996` は実用実装における工学的な詳細を網羅している。

<!--en:60662639-->
### 擬スペクトルと Kreiss の行列定理

<!--en:4f407776-->
:numref:`subsec_mdl-pseudospectra` では、非正規行列においては固有値が単位円盤の内部にあっても $\|\mathbf{A}^k\|_2$ の大きな過渡的増大（transient growth）は防げないことを見た。Kreiss の行列定理は、擬スペクトル（pseudospectrum）から得られる視覚的な直感を定量的な評価式へと変える。Kreiss 定数を $K(\mathbf{A})=\sup_{|z|>1}(|z|-1)\|(z\mathbf{I}-\mathbf{A})^{-1}\|_2$ と定義すると、任意の $n\times n$ 行列に対して累乗ノルムの最悪ケースは次の不等式で上下から挟まれる。すなわち、$K(\mathbf{A})\le\sup_k\|\mathbf{A}^k\|_2\le e\,n\,K(\mathbf{A})$ である。下界は :numref:`sec_mdl-spectra-in-practice` の演習問題で証明できる。一方、上界からは、擬スペクトルが単位円盤に密着している行列では非有界な過渡的増大が起こり得ないことが保証される。:citet:`Trefethen.Embree.2005` はこの理論を網羅的に展開しており、線形化した固有値がすべて厳密に安定であるにもかかわらず乱流へ遷移する流体流れへの古典的な応用も詳しく扱っている。

<!--en:9172f9f3-->
### 混合時間とスペクトルギャップ

<!--en:5fbf65b9-->
:numref:`sec_mdl-eigendecompositions` の 2 ページからなるマルコフ連鎖では、第 2 固有値が $1/4$ であるため、定常分布までの距離がクリックごとに 4 分の 1 に縮小した。では、一般のマルコフ連鎖が混合するまでには何ステップ必要だろうか。可逆な連鎖に対し、$\lambda_\star$ を $1$ 以外のすべての固有値における絶対値の最大値とする。定常分布との全変動距離（total variation distance）が $\varepsilon$ 以内に入るまでに必要なステップ数は、上下ともに $1/(1-\lambda_\star)$ の定数倍で抑えられる（定数は $\varepsilon$ と最小の定常確率に依存する）。少なくとも確率 $1/2$ で現在の状態にとどまる怠惰な連鎖（lazy chain）では、すべての固有値が非負になるため、$\lambda_\star$ は単に第 2 固有値 $\lambda_2$ となる。さらに Cheeger の不等式は、スペクトルギャップ $1-\lambda_2$ を連鎖の*コンダクタンス*（部分集合の定常質量に対する、そこから流出する確率フローの最小比率）によって上下両側から評価する。言い換えれば、グラフのボトルネックから収束率を直接読み取ることができる。:citet:`Levin.Peres.2017` は、:numref:`subsec_mdl-pagerank` が :citet:`Horn.Johnson.2012` から引用している収束の結果とあわせて、この理論をきわめて明快に解説している。

<!--en:5b7bc8fd-->
### ランダム行列と訓練可能な深さ

<!--en:8132756a-->
:numref:`subsec_mdl-jacobian-products` では、個々の層のヤコビ行列のスペクトル半径だけでは、ヤコビ行列の長い積の振る舞いを制御できないことを確認した。では、非常に深いネットワークを初期化からそのまま訓練できるかどうかは何によって決まるのだろうか。:citet:`Saxe.McClelland.Ganguli.2014` は深層線形ネットワークの学習ダイナミクス（learning dynamics）を厳密に解き、直交する初期重みを用いれば訓練時間が深さに依存しなくなる一方、スケーリングしたガウス重みでは深さが増すにつれて学習が遅くなることを示した。この解析を非線形ネットワークへ拡張した :citet:`Pennington.Schoenholz.Ganguli.2017` は、自由確率論（free probability）を用いて入出力ヤコビ行列の特異値分布を完全に求めている。その結果、すべての特異値が $1$ の近傍に密に集まる*動的等長性*（dynamical isometry）と呼ばれるレジームは、直交重みと $\tanh$ 活性化関数の組み合わせでは達成できるものの ReLU では達成できず、このレジームで初期化したネットワークは層が深くてもはるかに高速に訓練できることが示された。

<!--en:71964f0e-->
### 訓練済み重みのスペクトル

<!--en:529080ae-->
:numref:`subsec_mdl-sample-covariance-mp` のマルチェンコ・パスツール則（Marchenko–Pastur law）は、純粋な独立ノイズからなる行列の特異値スペクトルを記述するものであり、初期化したばかりのネットワークの重み行列はまさにこのような姿をしている。:citet:`Martin.Mahoney.2021` は、訓練済みニューラルネットワークの重みのスペクトルをこのランダム行列のベースラインと比較し、訓練によって鋭い外れ値と重いべき乗則の裾の両方が生じることを観察した。同研究はこの裾指数（tail exponent）を暗黙的な自己正則化（implicit self-regularization）の兆候と解釈している。マルチェンコ・パスツール則からのこうした乖離は多くのアーキテクチャで一貫して見られるものの、テストデータで評価せずに裾指数を用いてモデルの品質を順位付けできるかについては、証明された定理ではなく経験的な相関にとどまっている。

<!--en:ca1c595c-->
## 打ち切り SVD を超える低ランク構造

<!--en:74a64460-->
### 行列補完と核ノルム

<!--en:89c3f7a7-->
映画の評価値行列では全成分のうち数パーセントしか観測されていないことがある一方、Eckart–Young の定理（:numref:`subsec_mdl-eckart-young`）を用いるには最初から行列全体がわかっている必要がある。行列補完（matrix completion）はこの隔たりを埋めるため、観測されたすべての成分に一致するという制約のもとで*核ノルム*（nuclear norm）$\|\mathbf{Z}\|_*=\sum_i\sigma_i$ を最小化する。これは、ベクトルのスパース性に対して $\ell_1$ ノルムが凸な代理となるのと同様に、行列のランクに対する凸な代理として働く。:citet:`Candes.Recht.2009` は、背後にある低ランク行列の特異ベクトルが少数のスパイクに集中せず座標全体に広がっている限り、この凸最適化問題によって $n^2$ 個よりはるかに少ないランダムな観測成分から元の行列が高い確率で厳密に復元できることを証明した。Soft-Impute :cite:`Mazumder.Hastie.Tibshirani.2010` は、欠損部分の補完ステップと特異値のソフト閾値処理（soft-thresholding）を交互に繰り返すことで、この問題のペナルティ付き版を解く。

<!--en:b8f801e1-->
### CUR 分解とスケッチング

<!--en:a57c7e9f-->
打ち切り SVD は近似誤差の面では最適であるものの、その特異ベクトルはすべての行と列を混ぜ合わせてしまうため解釈しにくく、さらに $m\times n$ 行列の完全な SVD を計算するには $O(mn\min(m,n))$ 回の演算コストがかかる。これに対し CUR 分解は、データ行列の実際の列 $\mathbf{C}$ と実際の行 $\mathbf{R}$ を取り出し、それらを小さなコア行列 $\mathbf{U}$ で結ぶことで近似を構成する。:numref:`subsec_mdl-eckart-young` では、ノルムが最大の列を単純に選ぶ素朴な戦略が打ち切り SVD に大きく劣ることを見た。:citet:`Mahoney.Drineas.2009` はその改善策を示しており、上位の特異ベクトルへの射影の長さの二乗である*レバレッジスコア*に応じて列と行をサンプリングすれば、高い確率で Eckart–Young の誤差の小さな定数倍以内に収まる。

<!--en:b110febd-->
より広く見れば、:numref:`subsec_mdl-spectrum-control` のランダム化値域探索（randomized range finder）:cite:`Halko.Martinsson.Tropp.2011` は*スケッチング*の 1 つの例である。すなわち、大きな行列に細長いランダム行列を掛け、はるかに小さなスケッチ上で問題を解く手法である。:citet:`Woodruff.2014` は、これと同じ発想によって最小二乗法や回帰がどのように高速化され、元のデータセットよりはるかに少ない行数しか持たないスケッチから最適解の $1+\varepsilon$ 倍以内の解を返せるかを概説している。

<!--en:660c6dd9-->
### アダプタの派生手法

<!--en:7c0a009a-->
LoRA（:numref:`subsec_mdl-low-rank-updates`）は、低ランク更新 $\mathbf{B}\mathbf{A}$ を $\mathbf{B}=\mathbf{0}$ およびランダムな $\mathbf{A}$ で初期化するため、最初の勾配ステップでは事前学習済みの重みとは無関係なランダムな $k$ 次元部分空間を通してしか入力を捉えられない。いくつかの派生手法は、この初期化やパラメータ化を単純な方法で変更している。PiSSA :cite:`Meng.Wang.Zhang.2024` は $\mathbf{B}\mathbf{A}$ を事前学習済みの重みのランク $k$ 打ち切り SVD で初期化して残差を凍結するため、ファインチューニングの開始直後から元の行列の主方向を適応させることができる。DoRA :cite:`Liu.Wang.Yin.ea.2024` は重みの各列をスカラーの大きさと単位方向に分解し、大きさは直接訓練しつつ、低ランク更新を方向のみに適用する。

<!--en:c20051e6-->
### アテンションにおける低ランク構造

<!--en:b677cf9a-->
$n$ トークンの系列に対する自己アテンションは $n\times n$ のスコア行列を作るため、アテンションの各章で扱う効率的なアーキテクチャのいくつかは、この行列またはそのキー・値キャッシュを低ランク因子分解（low-rank factorization）に置き換えている。線形アテンション（:numref:`sec_windowed-linear-attention`）はアテンションスコアを特徴量のドット積 $\phi(\mathbf{q})\cdot\phi(\mathbf{k})$ として表し、スコア行列のランクを特徴量の次元以下に抑える。マルチヘッド潜在アテンション（multi-head latent attention、:numref:`subsec_kv-cache-low-rank`）:cite:`DeepSeek-AI.2024` は、ヘッドごとに別々のキーと値を保持する代わりにトークンあたり 1 つの低次元潜在ベクトルを保存することでキャッシュを圧縮する。Linformer :cite:`Wang.Li.Khabsa.ea.2020` は、訓練済みアテンション行列の特異値が急速に減衰するという経験的な観測から出発し、キーと値を系列次元に沿って固定長へと射影する。

<!--en:89b36b62-->
### 他のノルムのもとでの最急降下法

<!--en:48a7c0f1-->
:numref:`subsec_mdl-orthogonalized-updates` では、勾配行列の極因子（polar factor）$\mathbf{U}\mathbf{V}^\top$ を計算した。幾何学的には、この行列はステップサイズをユークリッドノルムではなくスペクトルノルムで測ったときの最急降下方向である。:citet:`Bernstein.Newhouse.2024` はこの知見を深層学習のオプティマイザに対する統一的な視点へと発展させた。移動平均を無効にすると、Adam は符号降下法（sign descent、すなわち $\ell_\infty$ ノルムのもとでの最急降下法）に帰着し、Shampoo はスペクトルノルムのもとでの最急降下法に帰着する。この視点に立てば、各層に適切なノルムを選ぶことこそが中核となる設計上の判断であり、更新則はそこから自動的に導かれる。この考え方に基づいて構築されたスペクトルノルムオプティマイザは :numref:`sec_muon` で展開する。

<!--en:4bf0c8ae-->
## 書籍、講義、講義ノート

<!--en:f3e47eb0-->
**書籍**

<!--en:da56012a-->
- 『[Mathematics for Machine Learning](https://mml-book.github.io/)』（Deisenroth、Faisal、Ong 著）：第 2 章（線形代数）、第 3 章（解析幾何）、第 4 章（行列分解）で本章の内容を機械学習の視点から網羅している無料の教科書である。

<!--en:b9fd7f74-->
- 『[Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/)』（Boyd、Vandenberghe 著）：最小二乗法とデータを軸に構成されたスライド・コード付きの無料の書籍であり、:numref:`sec_mdl-geometry-linear-algebraic-ops` および :numref:`sec_mdl-svd-low-rank` とよく対応している。

<!--en:8a9663f0-->
- 『[Introduction to Linear Algebra, 6th ed.](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html)』および『[Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/)』（Gilbert Strang 著）：列による描像（column picture）と 4 つの基本部分空間を軸に据えた古典的な入門書と、データ解析および最適化に向けて SVD を中心に据えたその姉妹編である。

<!--en:1bdf237e-->
- 『[Linear Algebra Done Right, 4th ed.](https://linear.axler.net/)』（Sheldon Axler 著）：行列式を使わずに固有値、作用素、およびスペクトル定理（:numref:`sec_mdl-eigendecompositions`）を展開する無料の教科書である。

<!--en:0fc9888a-->
- 『[Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html)』（Trefethen、Bau 著）：条件性（conditioning）、安定性、QR 法、ならびに `solve`、`eig`、`svd` の背後にある浮動小数点アルゴリズムをわかりやすく解説した書籍である。

<!--en:6eeb657c-->
**講義とチュートリアル**

<!--en:ab5ecc55-->
- 「[MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)」および「[MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)」（Gilbert Strang 担当）：講義ノートと演習問題を備えた完結した講義コースである。

<!--en:72749a6f-->
- 3Blue1Brown による「[Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)」および Steve Brunton による「[Singular Value Decomposition](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv)」：線形変換、行列式、固有ベクトル、および SVD を視覚的に解説する講義シリーズである。

<!--en:f2a84ae3-->
- Kolter、Do による「[Linear Algebra Review and Reference](https://cs229.stanford.edu/section/cs229-linalg.pdf)」（Stanford CS229）および Ström、Åström、Akenine-Möller による『[Immersive Linear Algebra](http://immersivemath.com/ila/index.html)』。

<!--en:161de105-->
[ディスカッション](https://d2l.discourse.group/t/36856)
