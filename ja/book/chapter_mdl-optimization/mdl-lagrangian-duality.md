<!--en:572c847e-->
# 双対性

<!--en:e7119089-->
反復ソルバの進行が止まり収束が報告されたとき、返された点は本当に最適なのだろうかという自然な疑問が残る。制約のない凸関数であれば、勾配がゼロであることによって直ちに決着がつく（:numref:`sec_mdl-convexity`）。しかし制約がある場合、最適解は通常、勾配がゼロでない境界上に位置するため（:numref:`sec_mdl-constrained-optimization-duality`）、この単純な判定法は使えなくなる。したがって、数値ソルバにも数学的な証明にも、最適値に対する信頼できる下界という別の道具が必要になる。本節では、そのような下界を体系的に構成する方法を展開し、KKT 条件におけるラグランジュ乗数が実際に何を測っているのかを明らかにする。

<!--en:52215460-->
この難しさを具体的に見るため、:numref:`subsec_mdl-kkt-conditions` のクリッピング問題に立ち返ろう。そこでは、二乗距離 $\tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$（目標点 $\mathbf{x}_0 = (3, 4)$ までの距離）を単位円板 $\|\mathbf{x}\| \le 1$ 上で最小化し、最適解 $\mathbf{x}^\star = (0.6, 0.8)$ と最適値 $8$ を求めた（ここでは目標点を $\mathbf{x}_0$ と書き、$\mathbf{y}$ は使わない。$\mathbf{y}$ は後でラベルおよび凸共役（convex conjugate）の引数として用いるためである）。いま、反復ソルバが $\mathbf{x} = (0.55, 0.835)$ で停止したとしよう。この点は $\|\mathbf{x}\|^2 = 0.9997 \le 1$ を満たすため実行可能点（feasible point）であり、目的関数値 $8.01$ を達成する。この $8.01$ がほぼ最適なのか、それとも円板内の別の点で $7$ に到達するのかを、どうすれば判定できるだろうか。真の最適解における勾配は $(-2.4, -3.2)$ とゼロにはほど遠く、円板内には無限に多くの点が含まれるため、1 つずつ調べることは問題外である。

<!--en:777003f0-->
中心となる発想は、制約を目的関数に折り込むことである。非負の係数 $\lambda \ge 0$ を制約関数 $\tfrac12(\|\mathbf{x}\|^2 - 1)$ に掛けて目的関数に加える。すべての実行可能点においてこの追加項は非正となるため、修正した目的関数を $\mathbb{R}^2$ *全体*にわたって最小化すれば、その値は必ず制約付き最適値以下になる。このように $\lambda \ge 0$ を選ぶごとに制約なし最小化を通じて下界が得られ、とくに $\lambda = 4$ とすると下界 $8$ が得られるため、ソルバが返した $8.01$ は最適値から $0.01$ 以内にあることが直ちに証明される。このような下界のうち最も厳しいものを探すことで、対になる最適化問題である*双対問題*（dual problem）が定義される。注目すべきことに、元の問題が非凸であっても、双対問題は常に凸問題となる。狭義実行可能点（strictly feasible point）を持つ凸問題では、双対問題の最適値は制約付き最適値と一致し、最適乗数は各制約を緩和したときに最適値がどれだけ速く改善するかを表す。

<!--en:5ccb2b35-->
以下ではまず、この下界の考え方を用いてクリッピング問題の最適性を証明し、あらゆる最適化問題に対して成り立つ 2 行の議論である*弱双対性*（weak duality）を述べる。次に、本節で唯一繊細な幾何学的ステップを含む証明によって凸問題に対する*強双対性*（strong duality）を確立し、最適乗数を*シャドープライス*として解釈することで、重み減衰（weight decay）をノルム制約に直接結びつける。続いて、線形制約付き問題の双対を計算する便利な道具として*凸共役*を導入した後（初読時には流し読みしてもよい）、サポートベクターマシン（support vector machine、SVM）、注水法（water-filling）による電力配分、および最良の双対下界が最適値を真に下回る非凸問題という 3 つの示唆に富む例を順に見ていく。計算はすべて NumPy で扱えるほど小規模であり、プロットには `d2l` を用いる。

<!--en:40c031e5-->
## 双対問題

<!--en:e47938fc-->
最初の目標は、クリッピング問題のいかなる実行可能点も $8$ 未満の値をとらないことを、実行可能点を 1 つずつ調べることなく証明することである。本節のすべての下界は、任意の実行可能点においてすべての不等式制約関数は非正である、というただ 1 つの基本的な事実に基づいている。

<!--en:67511456-->
### 1 つの乗数から得られる下界

<!--en:57d90043-->
クリッピング問題を振り返ろう。目的関数 $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$（ただし $\mathbf{x}_0 = (3, 4)$）を、制約 $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$ のもとで最小化する問題である。係数 $\tfrac12$ を $g$ に含めているのは、単にその勾配を $\mathbf{x}$ に一致させるためである。KKT 条件から、解はすでにわかっている。最適解は $\mathbf{x}^\star = \mathbf{x}_0 / \|\mathbf{x}_0\| = (0.6, 0.8)$、最適値は $p^\star = \tfrac12(5 - 1)^2 = 8$ であり、最適乗数は $\lambda^\star = 4$ となる。この乗数は停留条件 $\nabla f(\mathbf{x}^\star) + \lambda^\star \mathbf{x}^\star = (-2.4, -3.2) + 4\,(0.6, 0.8) = \mathbf{0}$ から定まる。

<!--en:0bb19240-->
ここで、任意の乗数 $\lambda \ge 0$ と任意の実行可能点 $\mathbf{x}$ をとる。$g(\mathbf{x}) \le 0$ かつ $\lambda \ge 0$ であるから、その積 $\lambda g(\mathbf{x})$ は非正であり、したがって $f(\mathbf{x}) + \lambda g(\mathbf{x}) \le f(\mathbf{x})$ が成り立つ。左辺を $\mathbb{R}^2$ 全体にわたって最小化すれば値はさらに小さくなるか変わらないため、その制約なし最小値は高々 $f(\mathbf{x})$ である。これはすべての実行可能な $\mathbf{x}$ に対して成り立つため、最良の実行可能な $\mathbf{x}$ に対しても成り立つ。

<!--en:12379109-->
左辺を具体的に計算してみよう。最小化する関数はラグランジアン $\mathcal{L}(\mathbf{x}, \lambda) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2 + \tfrac\lambda2(\|\mathbf{x}\|^2 - 1)$ である。任意の $\lambda \ge 0$ に対し、これは $\mathbf{x}$ に関する凸二次関数でヘッセ行列は $(1 + \lambda)\mathbf{I}$ となるため、一意な最小点（minimizer）は勾配 $(\mathbf{x} - \mathbf{x}_0) + \lambda \mathbf{x}$ がゼロになる点、すなわち $\mathbf{x}(\lambda) = \mathbf{x}_0 / (1 + \lambda)$ で達成される。$\lambda$ を大きくするほど、$\mathbf{x}_0$ は原点に向かってより強く縮小される。この $\mathbf{x}(\lambda)$ をラグランジアンに代入すると（$\mathbf{x}(\lambda) - \mathbf{x}_0 = -\lambda\mathbf{x}_0 / (1 + \lambda)$ および $\|\mathbf{x}_0\|^2 = 25$ を用いる）、$\tfrac12\|\mathbf{x}(\lambda) - \mathbf{x}_0\|^2 = \tfrac{25}{2}\lambda^2/(1+\lambda)^2$ および $\tfrac\lambda2\|\mathbf{x}(\lambda)\|^2 = \tfrac{25}{2}\lambda/(1+\lambda)^2$ となる。これら 2 つの分数を合わせると因子 $(1 + \lambda)$ が約分され、最小値は $\lambda$ の単純な関数として残る。

<!--en:8ea99047-->
これら 2 つの項は互いに逆方向へ引っ張り合う。$\lambda$ が大きくなるにつれて、第 1 項は $12.5$ に向かって増加する一方、第 2 項（定数 $-\tfrac12$ が $g$ の中に含まれることに由来する項）は際限なく線形に減少する。最も厳しい下界は、これら 2 つの効果が釣り合う点にある。

<!--en:bf076653-->
いくつか数値を代入してみると、この釣り合いが具体的によくわかる。$q(0) = 0$、$q(1) = 5.75$、$q(2) = 7.33$、$q(3) = 7.875$、$q(4) = 8$、$q(5) = 7.92$、$q(8) = 7.11$ である。これらの値はどれ 1 つをとっても $p^\star$ に対する正当な下界となっている。微分すると $q'(\lambda) = \tfrac{25}{2(1 + \lambda)^2} - \tfrac12$ となり、これは $(1 + \lambda)^2 = 25$、すなわち $\lambda = 4$ のときにゼロになる。さらに $q''(\lambda) = -25/(1 + \lambda)^3 < 0$ であるから、$\lambda = 4$ が一意な最大点であり、$q(4) = 10 - 2 = 8 = p^\star$ を達成する。最良の下界は真の最適値に一致し、それを達成する乗数は KKT 乗数にほかならない。これにより、ソルバが得た値 $8.01$ は最適値から $0.01$ 以内にあることが保証される。最適化していない $\lambda = 3$ のような適当な値でさえ、すでに最適値から $0.135$ 以内にあることを保証してくれる。

<!--en:f09aa88d-->
なぜ $\lambda = 4$ で最適値に到達するのだろうか。ラグランジアンの最小点における制約値 $g(\mathbf{x}(\lambda)) = \tfrac12\bigl(25/(1 + \lambda)^2 - 1\bigr)$ に注目しよう。$\lambda < 4$ のときは乗数が弱すぎて最小点を単位円板の内側へ引き込めず（たとえば $\mathbf{x}(1) = (1.5, 2)$ のノルムは $2.5$ である）、$g(\mathbf{x}(\lambda)) > 0$ となって下界は実行不可能な点を反映したものになる。逆に $\lambda > 4$ のときは乗数が最小点を内側へ引き込みすぎてしまい（たとえば $\mathbf{x}(8)$ のノルムは $\tfrac59$ である）、$\lambda g(\mathbf{x}(\lambda)) < 0$ が下界を引き下げてしまう。ちょうど $\lambda = 4$ のときに最小点 $\mathbf{x}(4) = (0.6, 0.8)$ が境界円上に乗り、ペナルティ項 $\lambda g(\mathbf{x}(\lambda))$ がゼロとなって、下界が実行可能点の目的関数値と一致するのである。

<!--en:ff322069-->
$q(\lambda)$ と $p^\star$ を $\lambda$ の値のグリッド上で並べてプロットし、これらの下界がどのように振る舞うかを確認しよう。

<!--en:820973a4-->
得られた曲線は凹関数であり、至るところで水平線 $p^\star = 8$ の下側に留まりつつ、$\lambda = 4$ で 1 度だけ接する。驚くべきことに、ここでの凹性は円板の幾何形状とは何の関係もない。以下で示すように、双対関数（dual function）は*あらゆる*最適化問題に対して凹関数となる。

<!--en:0f078303-->
### ラグランジュ双対関数

<!--en:dc07b7a5-->
上の下界の議論には、クリッピング問題に固有の性質は何ひとつ使われていなかった。同じ構成法を :numref:`subsec_mdl-kkt-conditions` の一般の制約付き問題に適用しよう。

<!--en:8fc1451c-->
この問題の最適値を $p^\star$ とし、すべての実行可能点にわたる $f$ の下限として定義する（実行可能点が存在しない場合は $p^\star = +\infty$ とする）。ここで $\mathcal{D}$ は $f$ およびすべての制約関数が定義される定義域である。これは通常単なる $\mathbb{R}^n$ であるが、本節末尾の双対ギャップ（duality gap）の例で $[0, 1]$ に対して行うように、ある集合をハード制約として残す場合はその集合となる。ラグランジアンは各制約に対して 1 つずつ乗数を導入し、次式で定義される。

<!--en:593d2a74-->
ここで、不等式制約には $\lambda_i \ge 0$ を対応させ、等式制約には符号を制限しない $\nu_j \in \mathbb{R}$ を対応させる。また、$\boldsymbol{\lambda} \succeq 0$ と書いて、$\lambda_i \ge 0$ がすべての座標 $i$ について成り立つことを表す。

<!--en:d1c47070-->
乗数を固定してラグランジアンを $\mathbf{x}$ について最小化すると、下界の構成は $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ の関数になる。**ラグランジュ双対関数**（Lagrange dual function）は次式で定義される。

<!--en:49672262-->
言葉で言えば、$q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ とは、$f$ がどこまで小さくなりうるかを、ハード制約を取り除いて $\boldsymbol{\lambda}$ と $\boldsymbol{\nu}$ で重み付けした線形ペナルティに置き換えたもとで問うものである。クリッピング問題では、これにより :eqref:`eq_mdl-opt-ball-dual` が得られた。一般に、この下限は達成されるとは限らず、$-\infty$ になることさえある。たとえば、$x$ を制約 $-x \le 0$ のもとで最小化する場合を考えよう。ラグランジアン $(1 - \lambda)x$ は $\lambda = 1$ でない限り下に有界でないため、$q(1) = 0$ であり、$q(\lambda) = -\infty$ がそれ以外のすべての $\lambda$ に対して成り立つ。$-\infty$ という下界は自明に正しいが何の役にも立たない。$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) > -\infty$ となる乗数の集合を $q$ の*実効定義域*（effective domain）と呼ぶ。後ほどサポートベクターマシンの双対問題を導出する際、この暗黙の定義域制限は明示的な等式制約として現れる。

<!--en:1eca180d-->
$q$ の 2 つの基本的な性質は、$f$、$g_i$、$h_j$ に凸性や微分可能性すら含むいかなる仮定も置かずに成り立つ。

<!--en:f61675fa-->
**命題（双対関数は凹関数である）。** *任意の主問題（primal problem）に対し、$q(\boldsymbol{\lambda}, \boldsymbol{\nu})$（:eqref:`eq_mdl-opt-dual-function`）は凹関数である。*

<!--en:a6b2ddff-->
**証明。** 任意の $\mathbf{x} \in \mathcal{D}$ を固定する。乗数は $\mathcal{L}$ に線形に入るため、写像 $(\boldsymbol{\lambda}, \boldsymbol{\nu}) \mapsto \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ はアフィンである。双対関数 $q$ は、このアフィン関数族の各点ごとの下限である。したがって、その**ハイポグラフ**（グラフ上またはその下側にある点の集合）は、これらのアフィン関数の下側にある半空間の共通部分となる。凸集合の任意の共通部分は凸集合であるため、$q$ は凹関数である。 $\blacksquare$

<!--en:b449d6fc-->
:numref:`fig_mdl-opt-dual-envelope` に、クリッピング問題に対するこの構成を示す。各破線は固定された 1 つの $\mathbf{x}$ に対するラグランジアンを $\lambda$ の関数として描いたものであり、$q$ はそれらすべての直線の下側に留まる曲線である。クリッピング問題の目的関数はたまたま凸関数であるが、上の議論ではその性質を一切使っていない。直線群の下側包絡線（lower envelope）は、$f$ と $g$ がどのような形状であっても凹関数になる。

<!--en:8dd7c903-->
![下側包絡線としてのクリッピング問題の双対関数。固定された各 $\mathbf{x}$ は 1 本の直線 $\lambda \mapsto \mathcal{L}(\mathbf{x}, \lambda)$（破線）を与え、$q(\lambda)$ はそれらの直線が $\lambda$ でとる最小値となる。描かれている直線は最小点 $\mathbf{x}(\lambda) = \mathbf{x}_0/(1+\lambda)$（いくつかの $\lambda$ の値に対するもの）に対応し、それぞれが $q$ に自身の $\lambda$ で接している（黒丸）。$\mathbf{x}^\star = \mathbf{x}(4)$ の直線は $g(\mathbf{x}^\star) = 0$ のため水平であり、$q$ に最大値 $d^\star = q(4) = p^\star = 8$ で接する。](../img/mdl-opt-dual-envelope.svg)

<!--en:4cc68c5a-->
**命題（弱双対性）。** *$\mathbf{x}$ が実行可能であり $\boldsymbol{\lambda} \succeq 0$ ならば、$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le f(\mathbf{x})$ が成り立つ。したがって、次式を得る。*

<!--en:480fe1c0-->
**証明。** 任意の実行可能な $\mathbf{x}$ と任意の $\boldsymbol{\lambda} \succeq 0$ に対し、$\lambda_i g_i(\mathbf{x}) \le 0$（$\lambda_i \ge 0$ かつ $g_i(\mathbf{x}) \le 0$ であるため）および $\nu_j h_j(\mathbf{x}) = 0$ が成り立つ。したがって、次式が成り立つ。

<!--en:1bcd02d2-->
左辺で $(\boldsymbol{\lambda}, \boldsymbol{\nu})$（ただし $\boldsymbol{\lambda} \succeq 0$）にわたる上限をとり、右辺で実行可能な $\mathbf{x}$ にわたる下限をとれば、:eqref:`eq_mdl-opt-weak-duality` が得られる。 $\blacksquare$

<!--en:21660043-->
弱双対性は、クリッピング問題におけるすべての $q(\lambda)$ を保証付きの下界に変えた 2 行の不等式にほかならない。またこれは、:numref:`subsec_mdl-kkt-conditions` における KKT 条件の十分性を支える仕組みそのものでもある。そこでは同じ不等式の連鎖を KKT 乗数において評価することで、相補性条件（complementary slackness）により両方の不等号が等号で成立する。

<!--en:569722bc-->
$q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ を $\boldsymbol{\lambda} \succeq 0$ および $\boldsymbol{\nu}$ について最大化することを**双対問題**と呼ぶ。これは凸集合上で凹関数を最大化する問題であるため、たとえ主問題がどれほど強い非凸性を持っていても、双対問題は*常に*凸最適化問題となる。その最適値 $d^\star$ はこの構成法が与えうる最も厳しい下界であり、双対実行可能なすべての $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ が正当な下界を与える。非負の差 $p^\star - d^\star \ge 0$ を**双対ギャップ**と呼ぶ。クリッピング問題ではこのギャップはゼロであったが、本節最後の具体例では真に正のギャップが生じる。となると次に生じる自然な問いは、ギャップがゼロになることを事前に保証できるのはいつか、ということである。

<!--en:8d7df980-->
## 強双対性

<!--en:3c57ca05-->
クリッピング問題では、最良の双対下界が主最適値と厳密に一致した。しかし非凸問題では、双対下界が主最適値に真に届かず、解の一部しか保証できないことがある。幸い、凸性とただ 1 つの狭義実行可能点という 2 つの穏やかな条件さえあれば、このギャップを埋めるには十分である。達成可能な制約値と目的関数値を単純な幾何学的視点から眺めると、なぜこの両方が必要なのかがよくわかる。

<!--en:9ad8f1db-->
### スレーター条件

<!--en:ca1f9704-->
**命題（スレーター条件は強双対性を導く）。** *問題が凸であり（$f$ とすべての $g_i$ が凸、すべての $h_j$ がアフィン）、スレーター条件（:numref:`subsec_mdl-constraint-qualifications`）を満たすとする。すなわち、実行可能点 $\bar{\mathbf{x}}$ が $\mathcal{D}$ の相対的内部（relative interior）に存在し、$g_i(\bar{\mathbf{x}}) < 0$ がアフィンでないすべての不等式制約 $g_i$ に対して成り立つとする。このとき $d^\star = p^\star$ が成り立ち、さらに $p^\star$ が有限であれば双対最適解が達成される。*

<!--en:82193ac2-->
これは :numref:`subsec_mdl-kkt-conditions` で凸問題に対して導入した制約想定（constraint qualification）と同じものである。すなわち、湾曲したすべての不等式制約の真に内側に少なくとも 1 つの点が存在しなければならない（多次元における完全な証明については :citet:`Boyd.Vandenberghe.2004` の 5.3.2 節を参照）。クリッピング問題はスレーター条件を容易に満たす。$f$ と $g$ はともに凸であり、原点 $\mathbf{0}$ は $g(\mathbf{0}) = -\tfrac12 < 0$ を満たすからである。したがって、導関数を 1 度も計算する前から $q(4) = p^\star$ は保証されていたのである。

<!--en:5b53975c-->
凸性と狭義実行可能性が*なぜ*ギャップを埋めるのかを理解するため、単一の不等式制約 $g(\mathbf{x}) \le 0$ を持つ問題を考えよう。すべての点 $\mathbf{x} \in \mathcal{D}$ をその制約値と目的関数値の組 $(g(\mathbf{x}), f(\mathbf{x}))$ に写し、これらの組の上側および右側にあるすべての点を集めた集合を作る。

<!--en:13a2b897-->
この $(u, t)$ 平面において実行可能点は $u \le 0$ に対応するため、主最適値 $p^\star$ とは単に高さ $t$ のうち $\mathcal{A}$ が縦軸 $u = 0$ 上で到達する最小値にほかならない。いま $\lambda \ge 0$ を固定する。すべての $(u, t)$ のうち与えられた $\mathbf{x}$ から生じるものの上では、線形関数 $t + \lambda u$ は $(g(\mathbf{x}), f(\mathbf{x}))$ で最小となるため、$q(\lambda) = \inf_{(u, t) \in \mathcal{A}} (t + \lambda u)$ となる。幾何学的には、直線 $t + \lambda u = c$（傾き $-\lambda$）を下から押し上げ、ちょうど $\mathcal{A}$ を下から支えるところで止めたときの縦軸 $u = 0$ 上の切片が $q(\lambda)$ である。集合 $\mathcal{A}$ 全体がこの支持直線（supporting line）の上または上側に位置するため、切片 $q(\lambda)$ が $p^\star$ を超えることは決してない。これが弱双対性を視覚化した姿である（:numref:`fig_mdl-opt-primal-dual-gap`）。

<!--en:63c7230c-->
![双対性の支持直線による幾何構造。各 $\mathbf{x}$ は $(g(\mathbf{x}),f(\mathbf{x}))$ に写され、網掛け領域 $\mathcal{A}$ はこれらの組と、その上側および右側にあるすべての点を含む。双対関数を $\lambda$ で評価することは、傾き $-\lambda$ の直線を $\mathcal{A}$ を下から支えるまで押し上げることに相当し、$u=0$ におけるその切片が $q(\lambda)$ となる。(a) クリッピング問題：$\mathcal{A}$ は凸集合で $\mathbf{0}$ は狭義実行可能点であるため、最適な支持直線は $(0,p^\star)$ に傾き $-4$ で接し、$d^\star=p^\star=8$ を与える。(b) $-x^2$ を $[0,1]$ 上で制約 $x\le\tfrac12$ のもとで最小化する問題：$\mathcal{A}$ が非凸であるため、最も高い支持直線でも $(0,p^\star)$ の真に下側を通り、$d^\star=-\tfrac12<-\tfrac14=p^\star$ となる。](../img/mdl-opt-primal-dual-gap.svg)

<!--en:fd143e5f-->
この幾何学的描像があれば、単一の不等式制約に対する強双対性の証明は見通しのよいものになる。

<!--en:84ceb212-->
**証明（$g(\bar{\mathbf{x}}) < 0$ かつ $p^\star$ が有限である単一不等式制約の場合）。** 第 1 に、$\mathcal{A}$ は凸集合である。任意の 2 点 $(u_1, t_1)$ と $(u_2, t_2)$ を $\mathcal{A}$ からとり（それぞれ $\mathbf{x}_1$ と $\mathbf{x}_2$ に対応するとする）、任意の $\theta \in [0, 1]$ をとる。$g$ の凸性より、内分点 $\mathbf{x}_\theta = \theta\mathbf{x}_1 + (1 - \theta)\mathbf{x}_2$ は $g(\mathbf{x}_\theta) \le \theta g(\mathbf{x}_1) + (1 - \theta) g(\mathbf{x}_2) \le \theta u_1 + (1 - \theta) u_2$ を満たし、同じ不等式が $f$ に対しても成り立つ。ゆえに $(u_1, t_1)$ と $(u_2, t_2)$ の凸結合（convex combination）は $\mathcal{A}$ に属する。

<!--en:1b9651b0-->
第 2 に、$(0, p^\star)$ は $\mathcal{A}$ の境界上にある。すなわち、この点は $\mathcal{A}$ 内の点列の極限である一方、$(0, p^\star - \varepsilon) \notin \mathcal{A}$ がすべての $\varepsilon > 0$ に対して成り立つ。凸集合の任意の境界点には支持直線が存在する。すなわち、ゼロでないベクトル $(\lambda, \gamma) \neq (0, 0)$ が存在して、$\gamma t + \lambda u \ge \gamma p^\star$ がすべての $(u, t) \in \mathcal{A}$ に対して成り立つ（:citet:`Boyd.Vandenberghe.2004` の 2.5.2 節を参照。これは :numref:`sec_mdl-convexity` で劣勾配（subgradient）を導いた幾何学的事実と同じものである）。$\mathcal{A}$ は上方向および右方向に限りなく広がっているため、$\gamma \ge 0$ かつ $\lambda \ge 0$ でなければならない。

<!--en:c053643c-->
では、$\gamma = 0$ となることはありうるだろうか。もしそうだとすると、支持直線は垂直で $\lambda > 0$ となり、$\lambda u \ge 0$（したがって $u \ge 0$）が $\mathcal{A}$ 全体で成り立つことになる。しかしこれは、狭義実行可能点 $\bar{\mathbf{x}}$ が $u = g(\bar{\mathbf{x}}) < 0$ を与えることと矛盾する。したがって $\gamma > 0$ である。両辺を $\gamma$ で割ると、$t + (\lambda/\gamma)\,u \ge p^\star$ がすべての $(u, t) \in \mathcal{A}$ に対して得られ、これは $q(\lambda/\gamma) \ge p^\star$ を意味する。弱双対性と合わせれば $q(\lambda/\gamma) = p^\star$ となり、双対最適解が達成されてその値は $p^\star$ に一致する。 $\blacksquare$

<!--en:50f6400b-->
この証明において、2 つの仮定がいかにきれいに役割分担しているかに注目しよう。凸性は $\mathcal{A}$ を凸集合にし、$(0, p^\star)$ における支持直線の存在を保証するために 1 度だけ使われる。スレーター条件は、その支持直線が（無限大の乗数に対応する）垂直線になるのを防ぐために 1 度だけ使われる。:numref:`fig_mdl-opt-primal-dual-gap` のパネル (a) のクリッピング問題では、ノルム $\|\mathbf{x}\| = r$ の点は最小目的関数値 $\tfrac12(5 - r)^2$ と制約値 $u = \tfrac12(r^2 - 1)$ を持つため、$\mathcal{A}$ の下側境界は曲線 $t = \tfrac12\bigl(5 - \sqrt{1 + 2u}\bigr)^2$ となる。その $u = 0$ における傾きは $-4$ であり、傾き $-\lambda^\star = -4$ の最適な支持直線と一致する。パネル (b) は本節末尾で扱う非凸の例を先取りしたものである。そこでは $\mathcal{A}$ が逆向きに湾曲しているため、$\mathcal{A}$ の下側にあるどの直線も $(0, p^\star)$ の真に下を通過してしまう。

<!--en:fc55be48-->
### 鞍点

<!--en:5d43e006-->
強双対性は、双対関数に言及することすらなく、ラグランジアンのみを用いて言い換えることもできる。実際、主問題は min–max 問題として書き直せる。任意の $\mathbf{x} \in \mathcal{D}$ に対して、次式が成り立つ。

<!--en:66d24b5c-->
もし $\mathbf{x}$ がいずれかの制約に違反していれば、対応する乗数を大きくすることで $\mathcal{L}$ を $+\infty$ へ発散させられる。一方、$\mathbf{x}$ が実行可能であれば、乗数側にできる最善の選択は余裕のあるすべての制約に対して $\lambda_i = 0$ と置くことであり、結果として $f(\mathbf{x})$ が残る。したがって $p^\star = \inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \mathcal{L}$ となる一方、定義より $d^\star = \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \inf_{\mathbf{x}} \mathcal{L}$ である。弱双対性とは、任意の 2 変数関数に対して成り立つ普遍的な不等式 $\sup \inf \le \inf \sup$ にほかならない。強双対性は、最小化と最大化の順序を入れ替えても値が変わらないことを主張する。両方の最適解が達成されるとき、この順序交換可能性はラグランジアンに鞍点が存在することと同値である。

<!--en:165d5173-->
**命題（強双対性と鞍点）。** *強双対性が成り立ち、主最適解 $\mathbf{x}^\star$ と双対最適解 $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ の両方が達成されるとする。このとき $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ は、ラグランジアンの* **鞍点**（saddle point）*である。すなわち、すべての $\mathbf{x} \in \mathcal{D}$、すべての $\boldsymbol{\lambda} \succeq 0$、およびすべての $\boldsymbol{\nu}$ に対して、次式が成り立つ。*

<!--en:37ec584e-->
*逆に、$\mathcal{L}$ の $\mathbf{x} \in \mathcal{D}$ および $(\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu})$ に関する任意の鞍点は双対ギャップをゼロにし、その鞍点値は $p^\star = d^\star$ となる。*

<!--en:b00567e3-->
**証明。** $q$ の定義と弱双対性より、次式が成り立つ。

<!--en:c316cfea-->
ここで最後の不等号は、$\lambda_i^\star \ge 0$ かつ $g_i(\mathbf{x}^\star) \le 0$ によりすべてのペナルティ項が非正となることから従う。強双対性（$d^\star = p^\star$）により、この不等式の連鎖はすべて等号で成り立たなければならない。左側の等号は、$\mathbf{x}^\star$ が $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ を最小化することを意味し、:eqref:`eq_mdl-opt-saddle-point` の右側の不等式を与える。右側の等号からは $\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$ が従い、これは相補性条件にほかならない。ゆえに、任意の $\boldsymbol{\lambda} \succeq 0$ と $\boldsymbol{\nu}$ に対して $\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x}^\star) + \sum_i \lambda_i g_i(\mathbf{x}^\star)
\le f(\mathbf{x}^\star)
= \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ が成り立ち、:eqref:`eq_mdl-opt-saddle-point` の左側の不等式が得られる。

<!--en:22beb46f-->
逆に、$(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ が鞍点であれば、次式が成り立つ。

<!--en:8b6455d8-->
これより $p^\star \le d^\star$ がわかる。逆向きの不等式は弱双対性から得られるため、$p^\star = d^\star$ が成り立つ。 $\blacksquare$

<!--en:6b69060b-->
クリッピングの例では、鞍点は $(\mathbf{x}^\star, \lambda^\star) = ((0.6, 0.8), 4)$ である。$\mathbf{x}$ を $\mathbf{x}^\star$ からずらすと $\mathcal{L}(\cdot, 4)$ は増加し、一方で $\lambda$ を動かしても $\mathcal{L}(\mathbf{x}^\star, \lambda) = 8$ は $g(\mathbf{x}^\star) = 0$ のため一定のまま保たれる。

<!--en:ec0ee7dc-->
強双対性は実用上 3 つの利点をもたらす。

<!--en:a38b5087-->
- *双対問題は主問題より単純または小規模になりうる。*サポートベクターマシンの双対問題はデータ点ごとに 1 つの変数を持ち、単純な符号制約と高々 1 つの等式制約しか持たない。また、演習問題 :exref:`ex_mdl-lagrangian-duality-ridge-dual` のリッジ回帰の双対問題は、特徴量の次元ではなくデータ例の数に等しいサイズの連立一次方程式（linear system）に帰着する。

<!--en:ea7c6fd1-->
- *任意の双対実行可能点が停止基準を与える。*任意の主実行可能な $\mathbf{x}$ と双対実行可能な $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ に対し、差 $f(\mathbf{x}) - q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ は $f(\mathbf{x})$ が $p^\star$ からどれだけ離れうるかの上界を与える。

<!--en:a84c9e43-->
- *双対解から主問題の解を復元できる。*:eqref:`eq_mdl-opt-saddle-point` より、$\mathbf{x}^\star$ は $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ の最小点集合の*中に*含まれる。その最小点が一意であるとき（たとえば $f$ が狭義凸のとき）、直ちに $\mathbf{x}^\star$ が得られる（クリッピング問題で $\mathbf{x}(4) = \mathbf{x}_0 / 5 = \mathbf{x}^\star$ となったのと同様である）。最小点が一意でない場合は、実行可能性と相補性条件によって最小点集合の中から $\mathbf{x}^\star$ が特定される。線形計画問題では $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star)$ は $\mathbf{x}$ に関して平坦になり、後述するオフセットにペナルティを課さない SVM では、ラグランジアンの最小化だけではオフセットが定まらず、相補性条件がそれを決定する。

<!--en:a4c21721-->
鞍点の命題はまた、:numref:`subsec_mdl-kkt-conditions` で凸問題に対する十分条件として示した KKT 条件の必要性にも決着をつける。微分可能な $f$、$g_i$、$h_j$ を $\mathcal{D} = \mathbb{R}^n$ 上に持つ凸問題がスレーター条件を満たし、最適解を $\mathbf{x}^\star$ で達成するとしよう。このとき強双対性が成り立ち、達成された双対最適解 $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ が存在する。$\mathbf{x}^\star$ は $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ を $\mathbb{R}^n$ 全体にわたって最小化するため、その点において勾配 $\nabla_{\mathbf{x}} \mathcal{L}$ はゼロになり、さらに今見たとおり相補性条件も成り立つ。主実行可能性および双対実行可能性と合わせれば、これらはまさに 4 つの KKT 条件そのものである。すなわちスレーター条件のもとで、KKT 条件は必要かつ十分である。

<!--en:d4685887-->
最後に、この min–max 構造は明示的な制約を持たない深層学習の問題にも現れる。敵対的生成ネットワーク（:numref:`sec_basic_gan`）や敵対的訓練（adversarial training）は、直接 $\min_{\boldsymbol{\theta}} \max_{\boldsymbol{\phi}}$ ゲームとして定式化される :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014,Madry.Makelov.Schmidt.ea.2018`。鞍点が存在しなければ、$\min\max$ 値と $\max\min$ 値の間にギャップが生じうる。さらに鞍点が存在する場合であっても、最小化変数に対する勾配降下ステップと最大化変数に対する勾配上昇ステップを同時に行うと収束しないことがある。実際、原点を鞍点とする単純な双線形目的関数 $xy$ に対しては、同時降下上昇法は任意の正のステップサイズで外側へ渦巻状に発散する（演習問題 :exref:`ex_mdl-lagrangian-duality-descent-ascent`）。

<!--en:d3317854-->
## 価格としての乗数

<!--en:235d3159-->
制約にはどれほどの価値があるだろうか。クリッピング問題において、単位円板をわずかに広げたとき、$\mathbf{x}_0$ にどれだけ近づけるだろうか。最適なラグランジュ乗数は、この問いに直接答えてくれる。さらに都合のよいことに、同じ感度解析の議論から、損失が凸である場合には重み減衰と重みに対するノルム制約がまったく同じ解の族を描くことも明らかになる。

<!--en:1ef82dda-->
### シャドープライス

<!--en:97163e32-->
各制約の価値を測るため、右辺に摂動を加えて最適目的関数値がどのように変化するかを追跡しよう。摂動ベクトル $\mathbf{u} \in \mathbb{R}^m$ と $\mathbf{v} \in \mathbb{R}^p$ に対し、摂動後の最適値を次式で定義する。

<!--en:bf4ceac2-->
$(\mathbf{0}, \mathbf{0})$ では元の最適値 $p^\star(\mathbf{0}, \mathbf{0})$ が復元され、一方で $u_i > 0$ とすることは第 $i$ 不等式制約を緩和することに相当する。実はこの関数はすでに幾何学的に目にしている。単一の不等式制約の場合、$p^\star(u)$ は高さ $t$ のうち :numref:`fig_mdl-opt-primal-dual-gap` の集合 $\mathcal{A}$ が横座標 $u$ で到達する最小値にほかならない。$\mathcal{A}$ の下側境界が $p^\star(u)$ のグラフであり、傾き $-\lambda^\star$ の支持直線は至るところでその下側にある。次の命題は、この幾何学的不等式を任意の個数の制約へと拡張するものである。

<!--en:c38c6829-->
**命題（乗数はシャドープライスである）。** *強双対性が成り立ち、双対最適解 $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ が存在するとする。このとき、すべての $\mathbf{u}$ と $\mathbf{v}$ に対して次式が成り立つ。*

<!--en:84e66b9f-->
*さらに $p^\star$ が $(\mathbf{0}, \mathbf{0})$ で微分可能であれば、その点において $\nabla_{\mathbf{u}} p^\star = -\boldsymbol{\lambda}^\star$ および $\nabla_{\mathbf{v}} p^\star = -\boldsymbol{\nu}^\star$ が成り立つ。*

<!--en:278a6978-->
**証明。** $\mathbf{x}$ を摂動後の問題に対する任意の実行可能点とすると、$g_i(\mathbf{x}) \le u_i$ および $h_j(\mathbf{x}) = v_j$ が成り立つ。強双対性、下限としての $q$ の定義、$\lambda_i^\star \ge 0$、および摂動後の制約を用いると、次式が得られる。

<!--en:c4a5b3e5-->
摂動後の問題で実行可能なすべての $\mathbf{x}$ にわたる下限をとれば、:eqref:`eq_mdl-opt-shadow-price` が得られる。さらに $p^\star$ が原点で微分可能であれば、非負関数 $p^\star(\mathbf{u}, \mathbf{v}) - p^\star(\mathbf{0}, \mathbf{0}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v}$ は大域的最小値 $0$ を $(\mathbf{0}, \mathbf{0})$ で達成するため、その点において勾配はゼロでなければならない。 $\blacksquare$

<!--en:1ef9ea18-->
最適乗数 $\lambda_i^\star$ は制約 $i$ の**シャドープライス**と呼ばれ、その制約を緩和したときに最適値が改善する変化率を測る。なお、:eqref:`eq_mdl-opt-shadow-price` は大域的な不等式であり、*いかなる*大きさの摂動に対しても成り立つ。制約 $i$ を $u_i > 0$ だけ緩和しても最適値は高々 $\lambda_i^\star u_i$ しか下がらない一方、同じ量だけ厳しくすると最適値は少なくとも $\lambda_i^\star u_i$ だけ上がる。一次近似では、$p^\star$ が微分可能なとき変化量は $-\lambda_i^\star u_i$ となる。余裕のある制約では相補性条件により $\lambda_i^\star = 0$ となるが、これも直感と一致する。有効になっていない制約を緩めても最適値は改善しないからである。

<!--en:92c79868-->
クリッピング問題でこれを確認してみよう。$g(\mathbf{x}) \le 0$ を $g(\mathbf{x}) \le u$ に緩和すると円板は半径 $r = \sqrt{1 + 2u}$ へ拡大し、$p^\star(u) = \tfrac12\bigl(5 - \sqrt{1 + 2u}\bigr)^2$（先ほど求めた $\mathcal{A}$ の下側境界）が得られる。その $u = 0$ における導関数は確かに $-4 = -\lambda^\star$ となっている。制約を $u = \tfrac12$ だけ緩和すると（半径 $\sqrt2$ の円板）、新しい最適値は $p^\star(\tfrac12) = \tfrac12(5 - \sqrt2)^2 \approx 6.43$ となり、:eqref:`eq_mdl-opt-shadow-price` が保証する大域的下界 $8 - 4 \cdot \tfrac12 = 6$ をしっかり上回っている。

<!--en:b2f67a97-->
1 つ留意しておきたい繊細な点がある。乗数は制約のスケーリングの仕方に依存するということである。ここでの $\lambda^\star = 4$ が価格付けしているのは $u$、すなわち二乗半径の*半分*であり、半径 $r$ そのものではない。半径 1 単位あたりの変化率を連鎖律で求めると $\mathrm{d}p^\star / \mathrm{d}r = -(5 - r)$ となり、これは $r$ に $u$ に関する導関数を掛けたものに等しく、$-\lambda^\star$ と一致するのはたまたま $r = 1$ だからにすぎない。制約を定数倍すれば乗数もそれに応じてスケーリングされる。たとえば $\|\mathbf{x}\|^2 - 1 \le 0$ のように係数 $\tfrac12$ を付けずに書けば、乗数は半分の $2$ になる。

<!--en:3a522141-->
:numref:`subsec_mdl-closest-point` の最近接点問題は等式制約の場合の例となっている。そこでは、$\tfrac12\|\mathbf{x}\|^2$ の超平面 $\mathbf{a}^\top\mathbf{x} = b$ 上での最小値は $p^\star(b) = b^2 / (2\|\mathbf{a}\|^2)$ であり、$\partial p^\star / \partial b = -\nu^\star$ は :eqref:`eq_mdl-opt-shadow-price` を摂動 $h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = v$（超平面を $b + v$ へ平行移動させるもの）に適用した結果そのものである。これにより、同節で数値的に確認した $0.02 = 0.2 \cdot 0.1$ という増加量が説明される。

<!--en:cb335928-->
### 重み減衰とノルム制約

<!--en:7c9c65b7-->
深層学習では、モデルの複雑さを制御するために、重み減衰（:numref:`sec_weight_decay`）を訓練損失 $\ell(\mathbf{w})$ に加えるか、あるいは重みにハードなノルム制約を課すことがよく行われる。本書の記法に従えば、これら 2 つの問題は次のように書ける。

<!--en:4688b0a2-->
ペナルティ付き問題の停留点（stationary point）では $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ が成り立つ。一方、制約付き問題では、乗数 $\lambda \ge 0$ を $g(\mathbf{w}) = \tfrac12\|\mathbf{w}\|^2 - \tfrac12 r^2$ に対して導入すると、KKT 停留条件 $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ が得られる。重み減衰係数がラグランジュ乗数の役割を果たす形で、まったく同じ方程式が現れるのである。

<!--en:450b5476-->
損失 $\ell$ が*凸*であるとき、この同値性は双方向に成り立つ。まずペナルティから制約へ向かう方向として、$\mathbf{w}_\lambda$ をペナルティ付き目的関数の最小点とし、ノルムバジェットを $r = \|\mathbf{w}_\lambda\|$ と選ぶ。すると $\mathbf{w}_\lambda$ は実行可能であり、制約は有効になっており（相補性条件を満たす）、KKT 停留条件も自動的に成り立つ。凸問題では KKT 条件が十分条件であるため（:numref:`subsec_mdl-kkt-conditions`）、$\mathbf{w}_\lambda$ は制約付き問題の解となる。

<!--en:bce1c70c-->
次に制約からペナルティへ向かう方向として、任意のバジェット $r > 0$ をとる。原点 $\mathbf{w} = \mathbf{0}$ が狭義実行可能点であるため、スレーター条件が成り立つ。制約付き最小値が $\mathbf{w}^\star$ で達成されるなら、KKT 必要条件から乗数 $\lambda^\star(r) \ge 0$ が存在し、$\mathbf{w}^\star$ は凸のペナルティ付き目的関数 $\ell(\mathbf{w}) + \tfrac{\lambda^\star}{2}\|\mathbf{w}\|^2$ を最小化する。$\lambda^\star > 0$ である限り、二次項によってこの目的関数は狭義凸となるため、$\mathbf{w}^\star$ がその一意な最小点となる。一方、バジェットに余裕があるときは $\lambda^\star = 0$ となり、ペナルティ付き問題は正則化なしの最小化に帰着する。シャドープライスの命題より $\lambda^\star(r) = -\partial p^\star / \partial(\tfrac12 r^2)$ が成り立つ。すなわち重み減衰係数とは、二乗ノルムの半分 1 単位あたりの訓練損失の限界削減量にほかならない。

<!--en:325d14fc-->
実際、どちらの方向の議論も $\ell_2$ ノルムに固有のものではない。$\tfrac12\|\mathbf{w}\|^2$ を任意の凸ペナルティ $\Omega(\mathbf{w})$、たとえば Lasso（:numref:`subsec_mdl-proximal-operators`）で用いられる $\ell_1$ ノルム $\|\mathbf{w}\|_1$ に置き換えてみよう。任意の最小点 $\mathbf{w}_\lambda$（$\ell + \lambda\Omega$ の最小点）は、バジェット $\Omega(\mathbf{w}) \le \Omega(\mathbf{w}_\lambda)$ の制約付き問題を解く。実際、そのバジェット内のすべての $\mathbf{w}$ に対して $\ell(\mathbf{w}) \ge \ell(\mathbf{w}_\lambda) + \lambda\,(\Omega(\mathbf{w}_\lambda) - \Omega(\mathbf{w})) \ge \ell(\mathbf{w}_\lambda)$ が成り立つからである。逆に、凸な $\ell$ と $\inf \Omega$ を真に上回る任意のバジェットに対してはスレーター条件が成り立ち、鞍点の命題によって制約付き問題の任意の最小点が $\ell + \lambda^\star\Omega$ も最小化することが保証される。

<!--en:b0e393c0-->
ただし、完全な同値性が成り立つためには $\ell$ の凸性が不可欠である。非凸損失の場合、ペナルティ付き問題の停留点は依然として対応する制約付き問題の KKT 点になるものの、大域的最適解の集合は食い違うことがあり、*いかなる*ペナルティ重み $\lambda$ を用いても到達できないノルムバジェットが生じうる（演習問題 :exref:`ex_mdl-lagrangian-duality-penalty-nonconvex`）。

<!--en:96d1aa2c-->
同じ二次項に対する第 3 の解釈にも触れておこう。ノイズ分散（noise variance）$\sigma^2$ を持つ二乗損失（squared loss）と、重みに対する等方的なガウス事前分布（Gaussian prior）$\mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ を仮定すると、ペナルティ $\tfrac\lambda2\|\mathbf{w}\|^2$（ただし $\lambda = \sigma^2/\tau^2$）は定数項と全体のスケールを除いて負の対数事前分布と一致する（:numref:`sec_mdl-maximum_likelihood`）。このように、単一の数値 $\lambda$ は、重み減衰係数、ノルムバジェットに対するラグランジュ乗数、およびノイズ分散と事前分布の分散の比という 3 通りの読み方ができる。

<!--en:16d30b1e-->
この同値性を数値的に検証してみよう。ランダムなデータ上の凸最小二乗損失 $\ell(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2$ に対し、まず $(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^\top\mathbf{b}$ によってペナルティ付き問題を閉形式で解き、$r = \|\mathbf{w}_\lambda\|$ とおく。次に、ステップサイズ $1/\lambda_{\max}(\mathbf{A}^\top\mathbf{A})$ の射影勾配降下法（projected gradient descent）を用い、クリッピングによって半径 $r$ の球へ射影しながら制約付き問題を解く。制約付きソルバは同じ重みを復元できるだろうか。また、停留条件から計算した乗数（$\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ と $\mathbf{w}$ の内積をとって得られる $\lambda = -\mathbf{w}^\top\nabla\ell(\mathbf{w}) / r^2$）は $\lambda$ と一致するだろうか。

<!--en:ba5f24c4-->
どの行においても、制約付き問題の解はペナルティ付き問題の解と機械精度で一致し、停留条件から復元した乗数も表示された小数点以下 6 桁すべてで $\lambda$ と一致している（具体的なバジェット $r$ の値は乱数の抽出ごとに変わる）。ペナルティと制約は同じ解の経路に対する 2 通りのパラメータ化にすぎず、ラグランジュ乗数が両者の間を変換しているのである。

<!--en:1a6399a9-->
## 凸共役

<!--en:483b3ac5-->
双対関数 $q$ を評価するには、通常、乗数の組を選ぶたびに新たな制約なし最小化問題を解き直す必要がある。しかし制約が線形である場合には、共通する計算をくくり出して、その最小化を最初に 1 度だけ済ませておくことができる。制約 $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$ に対してラグランジアンは $f(\mathbf{x}) + \boldsymbol{\lambda}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$ であるから、次式が成り立つ。

<!--en:a119b51f-->
この下限は $f$ と単一のベクトル $\mathbf{A}^\top\boldsymbol{\lambda}$ のみに依存し、$\mathbf{A}$ や $\mathbf{b}$ に個別には依存しないことに注目しよう。よく使われる各目的関数 $f$ に対して $\inf_{\mathbf{x}} [f(\mathbf{x}) - \mathbf{y}^\top\mathbf{x}]$ の値を $\mathbf{y}$ の関数として 1 度だけ計算しておけば、目的関数 $f$ を持つ*いかなる*線形制約付き問題の双対問題も、対応表からそのまま読み取れるようになる。その対応表こそが凸共役である。

<!--en:9e0ca218-->
### 支持直線

<!--en:7ffc00d6-->
まず 1 次元で $f(x) = \tfrac12 x^2$ を考え、傾き $y = 2$ をとろう。直線 $t = 2x + c$（傾き $2$）のうち、至るところで放物線以下にあるものはどれだろうか。$2x + c \le \tfrac12 x^2$ がすべての $x$ で成り立つ必要があり、これは $c \le \min_x (\tfrac12 x^2 - 2x) = -2$（$x = 2$ で達成）を意味する。そのような直線のうち最も高い $t = 2x - 2$ は点 $(2, 2)$ で放物線に接し、そこでは $f'(2) = 2$ である。傾き $2$ の直線をグラフに接するまで下げていくと縦軸切片は $-2$ に落ち着き、数値 $\max_x (2x - \tfrac12 x^2) = 2$ はその切片の符号を反転させたものになる。この様子を :numref:`fig_mdl-opt-conjugate-intercept`(a) に示す。

<!--en:f906f08d-->
![支持直線としての凸共役。(a) $f(x)=\tfrac12x^2$ と傾き $2$ の場合、傾き $2$ の破線をグラフに接するまで下げると $x=2$ で接して切片は $-2$ となり、$f^*(2)=2$ を与える。(b) $f(x)=|x|$ の場合、傾き $\tfrac12$ の直線は原点でグラフを下から支えて切片 $0$ となる一方、傾き $2$ の直線はどのような切片をとっても $|x|$ より速く上昇するため、$f^*(\tfrac12)=0$ および $f^*(2)=+\infty$ となる。このように、凸共役が有限になるのは、グラフの下側に支持直線が存在する傾きに限られる。](../img/mdl-opt-conjugate-intercept.svg)

<!--en:4d2a6fe7-->
関数 $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ の**凸共役**（フェンシェル・ルジャンドル変換とも呼ばれる）は、すべての傾きベクトル $\mathbf{y}$ に対してこの符号反転した切片を記録したものである（本書では共役にアスタリスク $*$ を、最適解・最適値に星印 $\star$ を用いる）。

<!--en:ff5caa31-->
幾何学的には、$f^*(\mathbf{y})$ は傾き $\mathbf{y}$ を持ち至るところで $f$ のグラフ以下にあるアフィン関数のうち最も高いものの縦軸切片にマイナスを付けた値である。凸共役は、$f$ を各点での値によってではなく、$f$ の支持直線の族によって記述する。$f(x) = \tfrac12 x^2$ と任意の傾き $y$ に対して上限は $x = y$ で達成され、$f^*(y) = \tfrac12 y^2$ となる。すなわち、この二次関数は自分自身の共役（自己共役）である。

<!--en:a533e28a-->
$f$ が値 $+\infty$ をとることを許すのは、制限された定義域を表現する便利な方法である。$f(\mathbf{x}) = +\infty$ となる点は上限に一切寄与せず、このような拡大実数値関数の凸性は単にそのエピグラフの凸性を意味する。

<!--en:2b703676-->
上限そのものも $+\infty$ になりうる。$f(x) = |x|$ と傾き $y = 2$ を考えよう。$x > 0$ では $2x - |x| = x$ が際限なく大きくなる。つまり、切片をどれほど低くしても、傾き $2$ の直線は V 字型のグラフの下側に留まることができない。一方、$|y| \le 1$ のときは常に $yx - |x| \le 0$ が成り立ち、$x = 0$ で等号となるため、原点を通る傾き $y$ の直線がグラフを下から支える。したがって、$f^*(y) = 0$（$|y| \le 1$ のとき）であり、それ以外では $f^*(y) = +\infty$ となる（:numref:`fig_mdl-opt-conjugate-intercept`(b)）。言い換えれば、絶対値関数の共役は区間 $[-1, 1]$ の*指示関数*（indicator function）、すなわち集合の内部で $0$、外部で $+\infty$ をとる関数である。

<!--en:a5ea470b-->
:eqref:`eq_mdl-opt-conjugate` から直ちに 2 つの性質が従う。第 1 に、$f^*$ は $f$ が凸であるかどうかにかかわらず*常に*凸関数である。固定された各 $\mathbf{x}$ に対して式 $\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ は $\mathbf{y}$ に関してアフィンであり、アフィン関数族の各点ごとの上限はそのエピグラフがそれらの上側半空間の共通部分となるため凸関数になる（これは $q$ が凹関数になることを示した議論のちょうど鏡像である）。

<!--en:f61859de-->
第 2 に、$f^*(\mathbf{y})$ はすべての $\mathbf{x}$ にわたる上限であるから、$f^*(\mathbf{y}) \ge \mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ がすべての $\mathbf{x}$ と $\mathbf{y}$ に対して成り立つ。項を整理すると**フェンシェル・ヤングの不等式**が得られる。

<!--en:42eae714-->
等号が成り立つのは、$\mathbf{x}$ が :eqref:`eq_mdl-opt-conjugate` の上限を達成するとき、かつそのときに限る。$f$ が凸かつ微分可能であれば、これは $\mathbf{y} = \nabla f(\mathbf{x})$ のとき、かつそのときに限って生じる。というのも、このとき $\mathbf{x} \mapsto \mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ は凹関数となり、勾配がゼロになる点で最大化されるからである。ここでは凸性が重要である。非凸の二重井戸関数（double well）$\phi_0(x) = (x^2 - 1)^2$ では $\phi_0'(0) = 0$ であるにもかかわらず、$\phi_0^*(0) = -\min \phi_0 = 0$ かつ $\phi_0(0) + \phi_0^*(0) = 1 > 0$ となる。これは $x = 0$ がグラフを下から水平な直線で支える点ではなく極大点だからである。自己共役対 $\tfrac12\|\mathbf{x}\|^2$ に対しては、フェンシェル・ヤングの不等式はおなじみの不等式 $\mathbf{y}^\top\mathbf{x} \le \tfrac12\|\mathbf{x}\|^2 + \tfrac12\|\mathbf{y}\|^2$ に帰着する。

<!--en:acbc5454-->
共役へ移ることで何か情報が失われるだろうか。任意の凸関数 $f$ が*固有*（proper）（決して $-\infty$ をとらず、恒等的に $+\infty$ でもない）かつ*閉*（closed）（エピグラフが閉集合である）ならば、支持直線族はグラフを完全に決定し、共役をもう 1 度とると元の $f$ が復元される。すなわち $f^{**} = f$ が成り立つ :cite:`Rockafellar.1970`。一方、$f$ が非凸である場合（かつ少なくとも 1 つのアフィン関数で下から抑えられる場合）、$f^{**}$ は下側から抑える最も大きい閉凸関数、すなわち $f$ の凸包絡関数（convex envelope）となる（演習問題 :exref:`ex_mdl-convexity-conjugate-pairs`d）。

<!--en:95b3ffb0-->
本節および演習問題で用いる 4 つの共役対を以下にまとめる。

<!--en:b2bb5740-->
- $\tfrac12\|\mathbf{x}\|^2$ と $\tfrac12\|\mathbf{y}\|^2$（自己共役対）。

<!--en:81105514-->
- $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ と $\tfrac12\mathbf{y}^\top\mathbf{A}^{-1}\mathbf{y}$（ただし $\mathbf{A} \succ 0$ であり、上限は $\mathbf{x} = \mathbf{A}^{-1}\mathbf{y}$ で達成される）。

<!--en:157c6b6b-->
- $e^x$ と、$y \log y - y$（$y > 0$ のとき。さらに値 $0$ を $y = 0$ で、$+\infty$ を $y < 0$ でとる。上限は $x = \log y$ で達成され、たとえば $f^*(1) = -1$、$f^*(e) = 0$ となる）。

<!--en:9df6acd0-->
- $\lambda\|\mathbf{x}\|_1$ とボックス $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le \lambda\}$ の指示関数（上の絶対値関数のベクトル版）。

<!--en:59df5aa5-->
### 線形制約付き問題の双対

<!--en:5a8221ce-->
ここで、凸共役を導入する動機となった線形制約付き問題、すなわち $f(\mathbf{x})$ を制約 $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$ のもとで最小化する問題に戻ろう。下限からマイナス符号をくくり出すと、そのまま $f^*$ を定義する上限へと変形できる。

<!--en:4184aaa8-->
等式制約 $\mathbf{A}\mathbf{x} = \mathbf{b}$ の場合も、まったく同じ 3 行の変形で $q(\boldsymbol{\nu}) = -f^*(-\mathbf{A}^\top\boldsymbol{\nu}) - \mathbf{b}^\top\boldsymbol{\nu}$（ただし $\boldsymbol{\nu}$ の符号は制約なし）が得られる。なお、これは $q$ が凹関数である理由に対する第 2 の説明にもなっている。$f^*$ は凸関数でその引数は $\boldsymbol{\lambda}$ に関して線形であり、マイナス符号が凸性を凹性へと反転させるからである。

<!--en:b2742340-->
:eqref:`eq_mdl-opt-dual-conjugate` を考えられる最小の例で試してみよう。半平面内で原点に最も近い点を求める問題、すなわち $\min_{\mathbf{x}} \tfrac12\|\mathbf{x}\|^2$（制約は $\mathbf{a}^\top\mathbf{x} \le b$、ただし $\mathbf{a} = (1, 2)$、$b = -1$）である。原点は制約に違反するため（$\mathbf{a}^\top\mathbf{0} = 0 > -1$）、最適解において制約は有効になっていなければならない。$f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|^2$ と $\|\mathbf{a}\|^2 = 5$ を :eqref:`eq_mdl-opt-dual-conjugate` に代入すると、1 ステップで双対関数が得られる。

<!--en:53adc8f5-->
$q'(\lambda) = 1 - 5\lambda = 0$ とおくと $\lambda^\star = \tfrac15$ および $d^\star = \tfrac15 - \tfrac1{10} = \tfrac1{10}$ が得られる。$\lambda^\star$ におけるラグランジアンの最小点は共役の上限を達成する点であり、$\mathbf{x} = -\lambda^\star\mathbf{a} = (-0.2, -0.4)$ となる。直接確かめると $\mathbf{a}^\top\mathbf{x} = -1 = b$ かつ $f(\mathbf{x}) = \tfrac12(0.04 + 0.16) = \tfrac1{10} = d^\star$ であり、弱双対性によって $\mathbf{x}$ が最適解であることが保証される。これは :numref:`subsec_mdl-closest-point` の射影公式 $(b / \|\mathbf{a}\|^2)\,\mathbf{a}$ を別の超平面に対して再現したものであり、主問題が 2 変数であったのに対し、双対問題は 1 変数となってわずか 1 行で解けたことになる。

<!--en:28adf05c-->
### log-sum-exp とエントロピー

<!--en:f28e32a3-->
深層学習のあらゆる場面で、ソフトマックスは生のスコアを指数変換して正規化することで確率へと変換する。なぜ他の正値関数ではなく指数関数なのだろうか。log-sum-exp 関数 $\mathrm{lse}(\mathbf{x}) = \log \sum_i e^{x_i}$ の凸共役がその明快な答えを与えてくれる。確率単体（probability simplex）を $\Delta = \{\mathbf{p} : \mathbf{p} \succeq 0,\ \mathbf{1}^\top\mathbf{p} = 1\}$、シャノンエントロピーを $H(\mathbf{p}) = -\sum_i p_i \log p_i$ と表そう。

<!--en:9985e750-->
$\mathrm{lse}^*(\mathbf{p}) = \sup_{\mathbf{x}} \bigl(\mathbf{p}^\top\mathbf{x} - \mathrm{lse}(\mathbf{x})\bigr)$ を計算するため、まず上限がいつ有限になりうるかを調べる。もし負の座標 $p_i$ が 1 つでもあれば、$x_i \to -\infty$ とすることで $\mathbf{p}^\top\mathbf{x}$ は $+\infty$ へ発散する一方、$\mathrm{lse}(\mathbf{x})$ は有界に留まるため、上限は $+\infty$ となる。また $\mathbf{1}^\top\mathbf{p} \neq 1$ であれば、すべての座標を定数だけシフトさせて $\mathbf{x} = c\mathbf{1}$ とおくと $c(\mathbf{1}^\top\mathbf{p} - 1) - \log n$（$\mathrm{lse}(c\mathbf{1}) = c + \log n$ であるため）となり、やはり $c \to \pm\infty$ としたときに発散する。ゆえに $\mathrm{lse}^*(\mathbf{p})$ が有限となるのは確率単体 $\Delta$ 上に限られる。

<!--en:4eea0f23-->
$\mathbf{p}$ が $\Delta$ の内部にあるとき（すべての $p_i > 0$）、$\mathbf{x}$ に関する勾配をゼロとおくと $\mathrm{softmax}(\mathbf{x}) = \mathbf{p}$ が得られ、その解は $\mathbf{x} = \log\mathbf{p}$（加法定数の自由度を除いて一意）となって値 $\sum_i p_i \log p_i$ を与える。一方、$\mathbf{p}$ が $\Delta$ の境界上にあり一部の $p_i = 0$ となっている場合、ソフトマックスの出力は常に真に正であるため有限な最大点は存在しない。しかし、該当する座標で $x_i \to -\infty$ とすれば同じ値に収束するため、この式は $\Delta$ 全体へ連続的に拡張される（ただし標準的な規約 $0 \log 0 = 0$ を用いる）。$\mathrm{lse}$ は凸かつ至るところ有限であるため $\mathrm{lse}^{**} = \mathrm{lse}$ が成り立ち、次の共役対が得られる。

<!--en:5a8c8d02-->
:eqref:`eq_mdl-opt-lse-entropy` の 2 行目を言葉で読んでみよう。log-sum-exp とは、期待スコア $\mathbf{p}^\top\mathbf{x}$ にエントロピーボーナス $H(\mathbf{p})$ を加えた量を、$n$ 個の結果上のすべての確率分布にわたって最大化したものである。どの分布がこの最大値を達成するだろうか。フェンシェル・ヤングの不等式の等号成立条件より、それは $\mathbf{p} = \nabla\mathrm{lse}(\mathbf{x})$ であり、その第 $i$ 成分は $\partial\,\mathrm{lse}/\partial x_i = e^{x_i} / \sum_j e^{x_j}$、すなわちソフトマックスそのものである（そのヘッセ行列は :numref:`sec_mdl-convexity` で計算した）。言い換えれば、同じ期待スコア $\mathbf{p}^\top\mathbf{x}$ を達成するすべての確率分布の中で、ソフトマックスは最大エントロピーを持つ分布であり、:numref:`subsec_mdl-max-entropy-derivation` における最大エントロピー法からの導出と一致する（演習問題 :exref:`ex_mdl-lagrangian-duality-max-entropy` ではその制約側からソフトマックスを導出する）。エントロピー項がなければ、$\mathbf{p}^\top\mathbf{x}$ を $\Delta$ 上で最大化するとすべての確率質量が $\arg\max_i x_i$ に集中するが、エントロピーを加えることでそのハードな選択が滑らかな確率分布へと軟化される。これとまったく同じ共役性の構成を $f$ ダイバージェンスの凸生成関数に適用すると、$f$-GAN の訓練に用いられる変分下界が得られる（:numref:`sec_mdl-f-gan-dual`）。

<!--en:20a08b01-->
## 具体例

<!--en:320f6403-->
ここからは、双対性が実際の問題で何を明らかにするかを浮き彫りにする 3 つの具体例を順に見ていこう。サポートベクターマシンでは、相補性条件によって解がごく少数のデータ点に集中し、さらに双対問題ではデータが内積のみを通じて現れる。注水法では、ラグランジュ乗数が電力 1 単位あたりの通信レートという具体的な工学的単位を持つ。最後に、1 次元の非凸問題を通じて、凸性が防いでいた真に正の双対ギャップを具体的に確認する。

<!--en:f15cda19-->
### サポートベクターマシンの双対問題

<!--en:c216616e-->
線形分離可能な訓練データ $(\mathbf{x}_i, y_i)$（ラベルは $y_i \in \{\pm 1\}$）が与えられたとき、サポートベクターマシン :cite:`Boser.Guyon.Vapnik.1992` は可能な限り最大のマージン（margin）を持つ分離超平面 $\mathbf{w}^\top\mathbf{x} + b = 0$ を求める。その主問題は次式で表される。

<!--en:d5ee7aa7-->
点 $\mathbf{x}$ から超平面までの距離は $|\mathbf{w}^\top\mathbf{x} + b| / \|\mathbf{w}\|$ であるから（:numref:`sec_mdl-geometry-linear-algebraic-ops`）、2 つのマージン超平面 $\mathbf{w}^\top\mathbf{x} + b = \pm 1$ は決定境界（decision boundary）の両側にそれぞれ距離 $1/\|\mathbf{w}\|$ の位置にあり、互いの間隔は $2/\|\mathbf{w}\|$ となる。したがって $\tfrac12\|\mathbf{w}\|^2$ を最小化することはマージンを最大化することに相当する。これは訓練点ごとに 1 つのアフィン不等式制約を持つ凸二次計画問題である。制約がアフィンであるため、スレーター条件には実行可能点が 1 つ存在すれば十分であり、データを分離する任意の $(\mathbf{w}, b)$ を定数倍してすべての点が少なくとも $1$ のマージンを持つようにスケーリングすればよい。ゆえに強双対性が保証される。

<!--en:e4ee6636-->
各制約に非負の乗数 $\alpha_i \ge 0$ を 1 つずつ導入し、ラグランジアンを次式で作る。

<!--en:df442b4c-->
双対関数 $q(\boldsymbol{\alpha}) = \inf_{\mathbf{w}, b} \mathcal{L}$ を計算するため、まず $\mathcal{L}$ がペナルティの課されていないオフセット $b$ に関してアフィンであり、その係数が $-\sum_i \alpha_i y_i$ であることに注目しよう。この係数がゼロでない限り、$b \to \pm\infty$ とすることで $\mathcal{L}$ は $-\infty$ へ発散するため、$q(\boldsymbol{\alpha}) = -\infty$ が $\sum_i \alpha_i y_i \neq 0$ のときは常に成り立つ。ここに :eqref:`eq_mdl-opt-dual-function` の実効定義域が実際に機能している姿が見て取れる。有限性の条件 $\sum_i \alpha_i y_i = 0$ が、双対問題における明示的な等式制約になるのである。この定義域上では $b$ の項が消え、$\mathbf{w}$ に関する狭義凸な二次関数が残るため、その一意な最小値は $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0}$ となる点で達成される。

<!--en:1afe8d33-->
したがって、最適な重みベクトルは訓練入力をそれぞれの乗数で重み付けした線形結合となる。この式を $\mathcal{L}$ に代入すると、**SVM の双対問題**が得られる。

<!--en:8653be82-->
ソルバを選ぶ前から、2 つの際立った帰結が導かれる。第 1 に、相補性条件 $\alpha_i (1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)) = 0$ より、$\alpha_i > 0$ となる任意の点は $y_i(\mathbf{w}^\top\mathbf{x}_i + b) = 1$ を満たし、いずれかのマージン超平面上にぴったり乗っていなければならない。これらの点を**サポートベクター**と呼ぶ。:numref:`fig_mdl-opt-svm-margin` では、丸で囲まれた 4 点の中にしかサポートベクターは存在しえない。それらの点では制約が有効になっており、:numref:`fig_mdl-opt-kkt-active-set` と同様にその法線ベクトルが $\nabla f$ と釣り合う一方、$\alpha_i = 0$ である点はすべて $\mathbf{w}$ から完全に脱落する。第 2 に、訓練入力は :eqref:`eq_mdl-opt-svm-dual` の中に内積 $\mathbf{x}_i^\top\mathbf{x}_j$ のみを通じて現れる。これらの内積をカーネル関数（kernel function）$k(\mathbf{x}_i, \mathbf{x}_j)$ に置き換えれば直ちに非線形分類器が得られるが、これは主問題の定式化では隠れていた構造である。

<!--en:191df6e3-->
![下のコードセルで用いる 8 点の最大マージン分離。実線は決定境界 $\mathbf{w}^{\star\top}\mathbf{x} + b^\star = 0$（$\mathbf{w}^\star = (\tfrac47, \tfrac47)$、$b^\star = -\tfrac37$）であり、破線はマージン超平面 $\mathbf{w}^{\star\top}\mathbf{x} + b^\star = \pm 1$（互いの間隔は $2/\|\mathbf{w}^\star\|$）である。マージン超平面上にある丸で囲まれた 4 点のみが $\alpha_i > 0$ を持ちうる。](../img/mdl-opt-svm-margin.svg)

<!--en:8796a539-->
等式制約 $\sum_i \alpha_i y_i = 0$ が生じたのは、ひとえにペナルティなしのオフセット $b$ によってラグランジアンがある方向にアフィンとなったためである。双対問題の実行可能領域（feasible region）を非負象限（負の成分をゼロにクリップするだけで射影できる単純な集合）に保つため、コードではオフセットを重みに折り込んだ密接に関連する変種を解くことにしよう。すなわち、各入力に定数特徴量を加えて $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$ とし、$\tilde{\mathbf{w}} = (\mathbf{w}, b)$ とおいて、$\tfrac12\|\tilde{\mathbf{w}}\|^2$ を制約 $y_i \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}_i \ge 1$ のもとで最小化する。この変種は $b$ にもペナルティを課すため決定境界は :eqref:`eq_mdl-opt-svm-primal` とわずかに異なりうるが、引き換えにラグランジアンがすべての主変数に関して狭義凸となり、等式制約が不要になる。$\mathbf{Q}_{ij} = y_i y_j \tilde{\mathbf{x}}_i^\top \tilde{\mathbf{x}}_j$ と定義すると、双対問題は次式となる。

<!--en:b25ab82b-->
これは非負象限上の凹二次関数の最大化問題である。射影勾配上昇法（:numref:`subsec_mdl-projected-gd` の更新則のステップ符号を反転させたもの）を適用すると、$\boldsymbol{\alpha} \leftarrow \max(\mathbf{0},\, \boldsymbol{\alpha} + \eta\,(\mathbf{1} - \mathbf{Q}\boldsymbol{\alpha}))$ を安全なステップサイズ $\eta = 1/\lambda_{\max}(\mathbf{Q})$ で反復することになる。平面上の 8 点に対してこの上昇法を実行し、復元された超平面がデータを分離するか、正の乗数がすべてマージン $1$ の点に乗っているか、そして双対値が主問題の値と一致するかという 3 つの予測を検証しよう。

<!--en:9455b74d-->
3 つの予測はすべて成り立っている。4 つの乗数が正であり、まさにその 4 点がマージン $1$ に位置している一方、残りの 4 点はマージンが $1.29$ から $2.43$ の間にあり $\alpha_i = 0$ となっている。出力された数値は厳密解 $\tilde{\mathbf{w}}^\star = (\tfrac47, \tfrac47, -\tfrac37)$（すなわち $\mathbf{w}^\star = (\tfrac47, \tfrac47)$、$b^\star = -\tfrac37$）と一致し、$p^\star = d^\star = \tfrac{41}{98} \approx 0.418367$、双対ギャップは $10^{-16}$ 付近となっている（主問題の値と双対値が等しいことは、最適解における恒等式 $\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ からも直接従う。演習問題 :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`a を参照）。

<!--en:130d0530-->
この出力結果は、双対解の解釈に関する繊細な点も浮き彫りにしている。主問題の目的関数 $\tfrac12\|\tilde{\mathbf{w}}\|^2$ は狭義凸であるため、主最適解 $\tilde{\mathbf{w}}^\star$ は一意である。しかし双対最適解 $\boldsymbol{\alpha}^\star$ は一意では*ない*。ここでは $\mathbf{Q} = \mathbf{G}\mathbf{G}^\top$ であり、$\mathbf{G}$ は $8 \times 3$ 行列（各行が $y_i\tilde{\mathbf{x}}_i^\top$）であるため、$\mathbf{Q}$ のランクは高々 $3$ である。4 つのサポートベクターは $\mathbb{R}^3$ 内の 4 本のベクトルに対応するため線形従属であり、それら 4 つの添字上にのみ非ゼロ成分を持つ非ゼロベクトル $\mathbf{d}$ で $\mathbf{G}^\top\mathbf{d} = \mathbf{0}$ を満たすものが存在する。$\boldsymbol{\alpha}^\star$ を $\mathbf{d}$ の方向に動かしても二次項は変化せず（$\mathbf{Q}\mathbf{d} = \mathbf{0}$ のため）、またすべてのサポートベクターがマージン $1$ を持つことから $\mathbf{1}^\top\mathbf{d} = \sum_i d_i\, y_i\tilde{\mathbf{w}}^{\star\top}\tilde{\mathbf{x}}_i = \tilde{\mathbf{w}}^{\star\top}\mathbf{G}^\top\mathbf{d} = 0$ となるため一次項も変化しない。したがって乗数ベクトルからなる線分全体が最適解となり、いずれも同じ $\tilde{\mathbf{w}}^\star$ と同じ双対値を与える。$\mathbf{0}$ から出発する射影勾配上昇法は、単にその線分上の 1 点を返したにすぎない。この線分の各端点では、マージン $1$ を持つ 4 点のうち 1 点で実際に $\alpha_i = 0$ となる（演習問題 :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`c）。相補性条件は一方向の含意であることに注意しよう。すなわち、$\alpha_i > 0$ ならばマージンが $1$ であることは保証されるが、マージンが $1$ だからといって $\alpha_i > 0$ が強制されるわけではない。

<!--en:c48a714f-->
### 注水法

<!--en:5b6754a6-->
2 つ目の具体例は、閉形式解（closed-form solution）と忘れがたい物理的描像の両方を備えている。送信機が総電力バジェット $P$ を $n$ 個の独立な通信チャネルに配分しなければならないとしよう。チャネル $i$ は背景雑音電力 $N_i > 0$ を持ち、そこに電力 $P_i \ge 0$ を割り当てると 1 回の使用あたり $\log(1 + P_i / N_i)$ nat の伝送レートが得られる :cite:`Cover.Thomas.1999`（Cover と Thomas は実ガウスチャネルに対して係数 $\tfrac12$ を含めているが、これは以下のラグランジュ乗数を定数倍するだけで最適配分は変えない）。総伝送レートを最大化するため、次の問題を解く。

<!--en:70620c6d-->
凹関数 $U$ の最大化は、アフィン制約のもとでの凸関数 $-U$ の最小化と同値である。すべての成分が真に正である実行可能な配分が存在するためスレーター条件が成り立ち、KKT 条件は必要十分条件となる。バジェットの等式制約に用いる乗数 $\nu$（$\sum_i P_i - P = 0$ に対応）と、非負制約に用いる乗数 $\lambda_i \ge 0$（$-P_i \le 0$ に対応）を導入すると、$-U$ のラグランジアンの停留条件は次式となる。

<!--en:4e29a72a-->
ここから 2 つの場合を読み取ろう。チャネル $i$ に正の電力が割り当てられるとき（$P_i > 0$）、相補性条件により $\lambda_i = 0$ が強制されるため $N_i + P_i = 1/\nu$ となり、電力を受けるすべての稼働チャネルはまったく同じ合計レベル $1/\nu$ に揃う。一方、チャネル $i$ に電力が割り当てられないとき（$P_i = 0$）、双対実行可能性 $\lambda_i = \nu - 1/N_i \ge 0$ から $N_i \ge 1/\nu$ が要求される。この共通のしきい値を $\theta = 1/\nu$ と書けば、最適配分は次の簡潔な形で表される。

<!--en:b5214aa9-->
これが古典的な**注水法**の規則である。各チャネルを底の高さが $N_i$ にある器と見立て、器の並び全体へ体積 $P$ の水を注ぎ込む場面を想像してほしい。:numref:`fig_mdl-opt-water-filling` に示すように、水面は一様な水位 $\theta$ に落ち着き、最も雑音の小さい静かなチャネルに最も多くの電力が注がれる一方、底の高さが $\theta$ を超える器は完全に乾いたまま残る。$\sum_i \max(\theta - N_i, 0)$ は連続な区分線形関数であり、$\theta \le \min_i N_i$ でゼロ、$\theta > \min_i N_i$ で狭義単調増加であるため、任意のバジェット $P > 0$ に対して一意な水位 $\theta$ が定まり、二分法によって容易に求められる（:numref:`subsec_mdl-projected-gd` の単体への射影は、各座標から共通のレベルを引いてゼロでクリップするという、このしきい値構造をちょうど反転させた形になっている）。

<!--en:b05e09bd-->
![下のコードセルの雑音フロアとバジェットに対する注水法。電力 $P=3$ を底の高さ $N_i$ の器に注ぐと、雑音が小さい上位 3 つのチャネルが共通の水位 $\theta=1/\nu^\star\approx1.43$ まで満たされてそれぞれ $P_i^\star=\theta-N_i$ が割り当てられる一方、底が $\theta$ より高い 2 つの器には相補性条件の要請どおり何も割り当てられない。](../img/mdl-opt-water-filling.svg)

<!--en:f6a6baeb-->
シャドープライスの命題から、$\nu^\star$ には直ちに工学的な意味が与えられる。$U$ の最大化は $-U$ の最小化であるから、バジェットを $P$ から $P + v$ へ緩和すると $-U^\star$ は変化率 $-\nu^\star$ で変化し、したがって $\mathrm{d}U^\star / \mathrm{d}P = \nu^\star = 1/\theta$ となる。すなわち最適乗数は、追加電力 1 単位あたりに得られる追加の通信レートを表す。5 つのチャネルに対する注水法の配分を計算し、このシャドープライスによる予測を $P \pm 10^{-4}$ における有限差分（finite difference）と比較してみよう。

<!--en:41b0307d-->
電力バジェットは機械精度で満たされており、稼働している 3 つのチャネルはすべて共通の水位 $\theta = (3 + 0.1 + 0.4 + 0.8)/3 \approx 1.4333$ に揃い、雑音の大きい $1.6$ と $2.5$ のチャネルは双対実行可能性の要請どおり乾いたまま留まっている。さらに、乗数 $\nu^\star = 1/\theta = 0.697674$ は有限差分による感度 $\mathrm{d}U^\star / \mathrm{d}P$ と表示された 6 桁すべてで一致している。

<!--en:9e5deb0e-->
### 明示的な双対ギャップ

<!--en:9c3ae3c3-->
弱双対性はあらゆる問題に対して成り立つが、強双対性はそうではない。1 次元の非凸の例を見ると、そのギャップが具体的によくわかる。定義域 $\mathcal{D} = [0, 1]$ 上で次の問題を考えよう。

<!--en:5b958b5c-->
$f(x) = -x^2$ は凸ではなく凹であるため、この問題は非凸問題である。実行可能領域は $[0, \tfrac12]$ であり、最小値は右端点で達成されて $p^\star = f(\tfrac12) = -\tfrac14$ となる。ここでは制約 $g(x) \le 0$ のみを双対化し、$\mathcal{D} = [0, 1]$ はハードな定義域制約として残すことで、:eqref:`eq_mdl-opt-dual-function` の下限を $[0, 1]$ 上でとることにする（弱双対性はそのまま変わらず成り立つ）。任意の $\lambda \ge 0$ に対し、ラグランジアン $\mathcal{L}(x, \lambda) = -x^2 + \lambda(x - \tfrac12)$ は $x$ に関して凹関数であるため、$[0, 1]$ 上での最小値は必ず両端点のいずれか、すなわち $\mathcal{L}(0, \lambda) = -\lambda/2$ または $\mathcal{L}(1, \lambda) = \lambda/2 - 1$ で達成される。したがって、次式が得られる。

<!--en:8f787b2a-->
保証されているとおり、これは区分線形な凹関数である。2 本の直線は $\lambda^\star = 1$ で交わり、そこでは $d^\star = -\tfrac12 < -\tfrac14 = p^\star$ となって、大きさ $\tfrac14$ の真に正の双対ギャップが生じる。$q(\lambda)$ を $p^\star$ と並べてプロットし、この非凸問題の双対曲線を本節冒頭のクリッピング問題のプロットと比較してみよう。

<!--en:4f77df99-->
滑らかに上昇して $p^\star$ に接したクリッピング問題の曲線とは異なり、このテント型の曲線は主最適値より丸々 4 分の 1 だけ低い $-\tfrac12$ でピークを迎える。狭義実行可能点が確実に存在することにも注目しよう。実際、$x = \tfrac14$ は $\mathcal{D}$ の内部にあり $g(\tfrac14) = -\tfrac14 < 0$ を満たすため、このギャップはスレーター条件の不成立ではなく、もっぱら凸性の欠如によって引き起こされている。:numref:`fig_mdl-opt-primal-dual-gap` のパネル (b) に立ち返ると、像の曲線 $\{(u, -(u + \tfrac12)^2) : u \in [-\tfrac12, \tfrac12]\}$ が下向きに湾曲しているため、$\mathcal{A}$ の下側にあるいかなる支持直線も $(0, p^\star)$ に届くことができず、最も高い支持直線（傾き $-1$）でも $-\tfrac12$ までしか保証できないのである。

<!--en:95b911ce-->
未解決の課題が 2 つ残っている。第 1 に、非凸問題では双対問題は最適値の下界を与えるものの、最適値に到達するとは限らない。第 2 に、たとえ鞍点が存在する場合でも、$\mathbf{x}$ と $\boldsymbol{\lambda}$ を同時に更新すると収束しないことがあり、これがミニマックス訓練における中心的な難しさとなっている。深層ネットワーク全体の訓練は非凸であるが、その構成要素の多く（射影、クリップ付き更新、最終層の当てはめ、凸信頼領域部分問題など）は凸であり、本節の内容はそれらにそのまま適用できる。もっとも、ここまでの本章の導出はすべて厳密な実数演算を仮定してきた。次節では、最適化アルゴリズムそのものではなく有限の浮動小数点精度（floating-point precision）が結果を左右するときに何が起きるかを見ていく。

<!--en:30ac03ff-->
## まとめ

<!--en:859aafdf-->
任意の制約付き最適化問題に対し、ラグランジュ双対関数 $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ は凹関数であり、弱双対性 $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le p^\star$ を満たすため、$\boldsymbol{\lambda} \succeq 0$ を選ぶたびに主最適値の下界が保証される。主問題が凸でありスレーター条件（アフィンでないすべての不等式制約に対する狭義実行可能点の存在）を満たすとき、強双対性によってギャップはゼロになり（$d^\star = p^\star$）、達成された主・双対最適解の組はラグランジアンの鞍点をなす。線形制約の場合、双対関数は凸共役を用いて $q(\boldsymbol{\lambda}) = -f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}$ に帰着し、log-sum-exp と負のエントロピーのような共役対から直ちに双対定式化が得られる。強双対性のもとでは、最適乗数はシャドープライスとして働き（$\lambda_i^\star = -\partial p^\star / \partial u_i$ および $\nu_j^\star = -\partial p^\star / \partial v_j$ が $p^\star$ の微分可能な点で成り立つ）、凸損失において重み減衰がノルム制約と一致する理由、SVM の乗数がサポートベクター以外でゼロになる理由、そして注水法の乗数 $1/\theta$ が電力 1 単位あたりの通信レートを測る理由を説明する。凸性がなければ、$-x^2$ を $[0, 1]$ 上で制約 $x \le \tfrac12$ のもとで最小化する例が示したように、真に正の双対ギャップが残りうる。

<!--en:2294243a-->
## 演習問題

<!--en:3cb5eba8-->
1. **共役対。**:eqref:`eq_mdl-opt-conjugate` から以下の各凸共役を計算せよ。
    1. $f(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$（ただし $\lambda > 0$）。
    1. $f(\mathbf{x}) = \tfrac12 \mathbf{x}^\top \mathbf{A} \mathbf{x}$（ただし $\mathbf{A} \succ 0$）。フェンシェル・ヤングの不等式 :eqref:`eq_mdl-opt-fenchel-young` はこの対に対して何を主張するか。
    1. ソフトプラス $f(t) = \log(1 + e^t)$（定義域 $\mathbb{R}$）。ロジット $t$ のソフトラベル $p \in [0, 1]$ に対する二値交差エントロピーは $\log(1 + e^t) - pt$ である。フェンシェル・ヤングの不等式を用いて、これが少なくとも二値エントロピー $H(p) = -p\log p - (1-p)\log(1-p)$ 以上であり、等号が成り立つのは $p = 1/(1 + e^{-t})$（$t$ のシグモイド）のとき、かつそのときに限ることを示せ。さらに、両者の差を KL ダイバージェンスとして表せ。
    1. 二重井戸関数 $\phi_0(x) = (x^2 - 1)^2$（:numref:`subsec_mdl-recognizing-convexity` の関数 $\phi$ から傾き項 $x/2$ を除いたもの）は凸ではない。その凸包絡関数 $\phi_0^{**}$ と、$\phi_0^{**} < \phi_0$ となる集合を求めよ。

   *Boyd および Vandenberghe の『[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)』例 3.22、3.26 を改変。*
   :exlabel:`ex_mdl-convexity-conjugate-pairs`

<!--en:9843fc77-->
1. **最大エントロピーと温度。**確率ベクトル $\mathbf{p} \in \mathbb{R}^n$ のエントロピーは $H(\mathbf{p}) = -\sum_i p_i \log p_i$ である。本問では制約側からソフトマックスを導出する。
    1. $H$ を制約 $\sum_i p_i = 1$ のもとで 1 つのラグランジュ乗数を用いて最大化せよ（符号制約 $p_i \ge 0$ は無視する）。唯一の停留点が一様分布であることを示せ。符号制約を無視しても答えが変わらないのはなぜか。
    1. すべてが等しいわけではないスコア $s_1, \ldots, s_n$ と、目標値 $\bar s$（$\min_i s_i$ と $\max_i s_i$ の真に間にある値）を固定し、制約 $\sum_i p_i s_i = \bar s$ を追加せよ。最大点が $\mathbf{p} = \mathrm{softmax}(\beta\mathbf{s})$（スカラー $\beta$ はスコア制約の乗数から定まる）となることを示せ。また、$\mathrm{softmax}(\beta\mathbf{s})$ のもとでの平均スコアが $\beta$ に関して狭義単調増加であり、目標値を満たす $\beta$ がただ 1 つ存在することを示せ。
    1. $H^\star(\bar s)$ を最大エントロピーとする。$\mathrm{d}H^\star / \mathrm{d}\bar s = -\beta$ であり、:numref:`subsec_mdl-closest-point` の乗数がオフセット $b$ を価格付けするのと同様に、この乗数がスコア制約を価格付けすることを示せ。どのような目標値 $\bar s$ に対して $\beta$ は負になるか。また、ソフトマックス温度 $1/\beta$ は $\bar s$ が $\max_i s_i$ に近づくときどうなるか。
   :exlabel:`ex_mdl-lagrangian-duality-max-entropy`

<!--en:86524813-->
1. **$\ell_1$ 最小化に対する双対保証。**基底追跡法

   $$\min_{\mathbf{x}}\; \|\mathbf{x}\|_1 \quad \textrm{subject to} \quad \mathbf{A}\mathbf{x} = \mathbf{b},$$

   （ただし $\mathbf{A} \in \mathbb{R}^{m \times n}$、$m < n$）は、劣決定な連立一次方程式からスパースな解を選び出す。制約はアフィンであるため、スレーター条件には実行可能点が 1 つあれば十分であり、方程式が解を持つ限り $d^\star = p^\star$ が成り立つ。また $\|\cdot\|_1$ の凸共役はボックス $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le 1\}$ の指示関数である。
    1. :eqref:`eq_mdl-opt-dual-conjugate` の等式制約版を用い、$\boldsymbol{\nu}$ を $-\boldsymbol{\nu}$ に置き換えると、双対問題が $\max_{\boldsymbol{\nu}} \mathbf{b}^\top \boldsymbol{\nu}$（制約 $\|\mathbf{A}^\top \boldsymbol{\nu}\|_\infty \le 1$）になることを示せ。
    1. $\mathbf{A} = (1\;\; 2)$ および $b = 2$ に対して両方の問題を手計算で解き、$d^\star = p^\star$ を確認せよ。
    1. $\mathbf{x}^\star$ と $\boldsymbol{\nu}^\star$ を最適解とする。$x_i^\star = 0$ が $|(\mathbf{A}^\top \boldsymbol{\nu}^\star)_i| < 1$ となるすべての成分で成り立ち、$(\mathbf{A}^\top \boldsymbol{\nu}^\star)_i = \mathrm{sign}(x_i^\star)$ が $x_i^\star \neq 0$ となるすべての成分で成り立つことを示せ。候補解 $\mathbf{x}$（$\mathbf{A}\mathbf{x} = \mathbf{b}$ を満たす）が与えられたとする。問題を解くことなくベクトル $\boldsymbol{\nu}$ によって $\mathbf{x}$ が最適解であることを保証するには、どのような条件を満たせばよいか。

<!--en:0d13d622-->
1. **サポートベクターと双対値。**オフセットを折り込んだ SVM 双対問題 :eqref:`eq_mdl-opt-svm-dual-folded` において、$\boldsymbol{\alpha}^\star$ を最適解とし、$\tilde{\mathbf{w}}^\star = \sum_i \alpha_i^\star y_i \tilde{\mathbf{x}}_i$ とおく。
    1. 非負象限上の双対問題の KKT 条件を書き下すことで、$\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ を示せ。さらに、双対最適値が主問題の値 $\tfrac12 \|\tilde{\mathbf{w}}^\star\|^2$ に等しいことを導け。
    1. $\alpha_j^\star = 0$ である訓練点を削除しても、残りの乗数は縮小された問題に対して最適なままであり、$\tilde{\mathbf{w}}^\star$ も変化しないことを示せ。1 つ抜き検証では、$n$ 個の各点を残りの $n - 1$ 点で訓練した SVM によって分類する。1 つ抜き誤り率は高々 $\alpha_i^\star > 0$ となる点の割合以下であることが、すべての最適解 $\boldsymbol{\alpha}^\star$ に対して成り立つことを導け。どの最適解 $\boldsymbol{\alpha}^\star$ が最良の上界を与えるか。
    1. [code] SVM のコードセルの行列 $\mathbf{Q}$ のランクは $3$ である。$S$ を $\alpha_i^\star > 0$ となる 4 つの添字の集合とし、$\mathbf{G}_S$ を `G` の対応する行からなる部分行列とする。非ゼロの $\mathbf{d} \in \mathbb{R}^4$ で $\mathbf{G}_S^\top\mathbf{d} = \mathbf{0}$ を満たすものを求め（`np.linalg.svd` を $\mathbf{G}_S^\top$ に適用せよ）、$\boldsymbol{\alpha}^\star_S + t\mathbf{d}$ をいずれかの成分がゼロに達するまで両方向へ動かせ。両方の端点が双対実行可能であり、双対値 $\tfrac{41}{98}$ を持ち、同じ $\tilde{\mathbf{w}}^\star$ を与えることを検証せよ。各端点において、マージン $1$ のどの点が $\alpha_i = 0$ になるか。このことは、乗数から「サポートベクター」を読み取ることについて何を意味するか（端点は $(0, 0.204, 0, 0, 0, 0.192, 0, 0.441)$ および $(0.204, 0, 0, 0, 0, 0.273, 0, 0.359)$ となるはずである）。
   :exlabel:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`

<!--en:f06790ef-->
1. [code] **勾配降下上昇法。** ● 主双対法や GAN の訓練では、最小化変数を勾配降下法で、最大化変数を勾配上昇法で同時に更新する。
    1. 原点を鞍点とする双線形ゲーム $\min_x \max_y\, xy$ に対し、同時降下上昇法 $x \leftarrow x - \eta y$ および $y \leftarrow y + \eta x$ を線形変換として表せ。任意の $\eta > 0$ に対し、ゼロでないあらゆる初期点から鞍点を離れて渦巻状に発散することを示せ。また、$y$ の更新に新しい $x$ を用いる交互更新版が、行列式 $1$ の線形変換であることを示せ。$\eta < 2$ のとき、その反復点はどのように振る舞うか。
    1. :numref:`subsec_mdl-closest-point` の最近接点問題のラグランジアン $\mathcal{L}(\mathbf{x}, \nu) = \tfrac12\mathbf{x}^\top \mathbf{x} + \nu(\mathbf{a}^\top \mathbf{x} - b)$ に対し、ここでは $\mathbf{a} = (1, 2)$、$b = 1$、初期値 $\mathbf{x} = \mathbf{0}$、$\nu = 0$ として同時降下上昇法を実行せよ。反復点が $(\mathbf{x}^\star, \nu^\star)$ に収束するステップサイズの範囲を求めよ。さらにその範囲を、組 $(\mathbf{a}^\top \mathbf{x} / \|\mathbf{a}\|, \nu)$ と、$\mathbf{x}$ のうち $\mathbf{a}$ に直交する成分のそれぞれに作用する更新の固有値から導出せよ。
    1. 双線形ゲームの目的関数に $\tfrac{\epsilon}{2} x^2$（ただし $0 < \epsilon \le 2$）を加えよ。このとき同時降下上昇法はどのステップサイズに対して収束するか。また、前問のラグランジアンのどの項が $\epsilon$ の役割を果たしているか。
   :exlabel:`ex_mdl-lagrangian-duality-descent-ascent`

<!--en:1194be41-->
1. [code] **凸性がない場合のペナルティと制約の比較。** ● 凸損失に対しては、:numref:`subsec_mdl-weight-decay-duality` で重み減衰とノルム制約が同じ解を描くことを示した。代わりに非凸の 1 次元損失 $\ell(w) = (w - 1)^2 (w - 3)^2 - w$（極小点は $w = 1.16$ と $w = 3.11$ の付近にある）を考えよ。
    1. $w$ の細かいグリッド上で、対数グリッド上の各ペナルティ付き最小点 $w_\lambda = \mathrm{argmin}_w\, \ell(w) + \tfrac\lambda2 w^2$（各 $\lambda$ に対するもの）と、制約付き最適値 $p^\star(r) = \min \{\ell(w) : w^2 \le r^2\}$（$r \in [0, 3.2]$）を計算せよ。点 $(w_\lambda^2, \ell(w_\lambda))$ を曲線 $(r^2, p^\star(r))$ の上に重ねてプロットせよ。バジェット $r$ のうち、いかなるペナルティ付き最小点のノルム $|w_\lambda|$ にも決して一致しないものはどの範囲か。
    1. $w_\lambda$ が $\ell(w) + \tfrac\lambda2 w^2$ を最小化することは、傾き $-\tfrac\lambda2$ で $(w_\lambda^2, \ell(w_\lambda))$ を通る直線が集合 $\{(w^2, \ell(w)) : w \in \mathbb{R}\}$ のすべての点の下側にあることと同値であることを示せ。この事実と :numref:`fig_mdl-opt-primal-dual-gap` の支持直線の描像を用いて、欠落するバジェットが生じる理由を説明せよ。
    1. 欠落する範囲から制約が有効となるバジェット $r$ を 1 つ選べ。本文と同様に制約を $\tfrac12 w^2 \le \tfrac12 r^2$ と書いたときの制約付き解の KKT 乗数 $\lambda$ を計算し、この $\lambda$ を用いたペナルティ付き目的関数がより小さい値をとる点を求めよ。さらに、この制約付き問題の双対ギャップ $p^\star(r) - d^\star(r)$ を計算せよ。
   :exlabel:`ex_mdl-lagrangian-duality-penalty-nonconvex`

<!--en:c7b7aec4-->
1. [extended] **エントロピー正則化付き最適輸送。** ●● 確率ベクトル $\mathbf{p} \in \mathbb{R}^n$ と $\mathbf{r} \in \mathbb{R}^m$、およびコスト行列 $\mathbf{C} \in \mathbb{R}^{n \times m}$ が与えられたとき、最適輸送は最も低コストな輸送計画 $\boldsymbol{\Gamma}$ で質量 $\mathbf{p}$ を質量 $\mathbf{r}$ へ移すものを求める。

   $$\mathrm{OT} = \min_{\boldsymbol{\Gamma} \ge 0}\; \sum_{i,j} C_{ij} \Gamma_{ij} \quad \textrm{subject to} \quad \boldsymbol{\Gamma} \mathbf{1} = \mathbf{p}, \;\; \boldsymbol{\Gamma}^\top \mathbf{1} = \mathbf{r},$$

   これは $nm$ 変数の線形計画問題である。エントロピー正則化 :cite:`Cuturi.2013` では、目的関数に $\varepsilon \sum_{i,j} \Gamma_{ij} (\log \Gamma_{ij} - 1)$（ただし $\varepsilon > 0$）を加える。正則化後の最適解を $\boldsymbol{\Gamma}_\varepsilon$ とする。シンクホーン反復によるその解法は、分布間の比較に広く用いられる :cite:`Peyre.Cuturi.2019`。
    1. 周辺分布の制約に対する項 $\boldsymbol{\phi}^\top (\mathbf{p} - \boldsymbol{\Gamma} \mathbf{1}) + \boldsymbol{\psi}^\top (\mathbf{r} - \boldsymbol{\Gamma}^\top \mathbf{1})$ を用いてラグランジアンを作れ。$\boldsymbol{\Gamma}_\varepsilon = \mathrm{diag}(\mathbf{u})\, \mathbf{K}\, \mathrm{diag}(\mathbf{v})$（ただし $K_{ij} = e^{-C_{ij} / \varepsilon}$、$u_i = e^{\phi_i / \varepsilon}$、$v_j = e^{\psi_j / \varepsilon}$）となることを示せ。符号制約 $\Gamma_{ij} \ge 0$ をラグランジアンから省いてよいのはなぜか。
    1. 双対関数が

       $$q(\boldsymbol{\phi}, \boldsymbol{\psi}) = \mathbf{p}^\top \boldsymbol{\phi} + \mathbf{r}^\top \boldsymbol{\psi} - \varepsilon \sum_{i,j} e^{(\phi_i + \psi_j - C_{ij}) / \varepsilon},$$

       となること、およびこれを $\boldsymbol{\phi}$ について（$\boldsymbol{\psi}$ を固定して）厳密に最大化すると要素ごとのシンクホーン更新 $\mathbf{u} = \mathbf{p} / (\mathbf{K} \mathbf{v})$ が得られることを示せ。これより、シンクホーンのアルゴリズムは凹関数上のブロック座標上昇法であり、双対値が決して減少しないことを導け。すべての双対値が $\mathrm{OT}$ の下界にもなるのはなぜか。
    1. $H(\boldsymbol{\Gamma}) = -\sum_{i,j} \Gamma_{ij} \log \Gamma_{ij}$ とおき、すべての実行可能な輸送計画が $\max(H(\mathbf{p}), H(\mathbf{r})) \le H(\boldsymbol{\Gamma}) \le H(\mathbf{p}) + H(\mathbf{r})$ を満たすことを認めるものとする。このとき $0 \le \sum_{i,j} C_{ij} (\boldsymbol{\Gamma}_\varepsilon)_{ij} - \mathrm{OT} \le \varepsilon \min(H(\mathbf{p}), H(\mathbf{r}))$ を示せ。
    1. コストが $[0, 1]$ の範囲にあるとする。float64 では、$e^{-x}$ は $x$ が約 $745$ を超えるとゼロに丸められる。どのような $\varepsilon$ に対して $\mathbf{K}$ の成分がゼロになりうるか、またそのとき乗法的な更新 $\mathbf{u} = \mathbf{p} / (\mathbf{K}\mathbf{v})$ に何が起きるか。$\boldsymbol{\phi}$ と $\boldsymbol{\psi}$ を直接更新し、$\mathbf{K}$ との積の代わりに log-sum-exp を用いればこの問題を回避できる理由を説明せよ。
   :exlabel:`ex_mdl-lagrangian-duality-entropic-ot`

<!--en:35adf513-->
1. [code] **リッジ回帰の双対問題。**$\min_{\mathbf{w}}\tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2 + \tfrac\lambda2\|\mathbf{w}\|^2$（ただし $\mathbf{A} \in \mathbb{R}^{n \times d}$）を、$\min_{\mathbf{w}, \mathbf{z}}\tfrac12\|\mathbf{z}\|^2 + \tfrac\lambda2\|\mathbf{w}\|^2$（制約 $\mathbf{A}\mathbf{w} - \mathbf{b} - \mathbf{z} = \mathbf{0}$、乗数 $\boldsymbol{\alpha} \in \mathbb{R}^n$）と書き換えよ。
    1. ラグランジアンを $\mathbf{z}$ と $\mathbf{w}$ について最小化し、$\mathbf{z} = \boldsymbol{\alpha}$、$\mathbf{w} = -\mathbf{A}^\top\boldsymbol{\alpha}/\lambda$、および双対関数 $q(\boldsymbol{\alpha}) = -\tfrac12\|\boldsymbol{\alpha}\|^2 - \tfrac1{2\lambda}\boldsymbol{\alpha}^\top\mathbf{A}\mathbf{A}^\top\boldsymbol{\alpha} - \mathbf{b}^\top\boldsymbol{\alpha}$ を導け。なぜ強双対性が成り立つのか。
    1. $q$ を最大化して $\boldsymbol{\alpha}^\star = -\lambda(\mathbf{A}\mathbf{A}^\top + \lambda\mathbf{I})^{-1}\mathbf{b}$ および $\mathbf{w}^\star = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top + \lambda\mathbf{I})^{-1}\mathbf{b}$ を求めよ。SVM の重みベクトルがデータ点の線形結合になるのと同様に、なぜ解が $\mathbf{A}$ の行ベクトルの線形結合になるのか、またなぜデータが $\mathbf{A}\mathbf{A}^\top$ のみを通じて現れるのかを説明せよ。
    1. データ例数 $n = 20$、特徴量数 $d = 50$ に対し、主問題の解法 $(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^\top\mathbf{b}$ と双対公式が約 $10^{-14}$ の精度で一致すること、および主問題の値と双対値が一致することを数値的に検証せよ。$n \ll d$ のとき計算コストが低いのはどちらの解法か。また、$\mathbf{A}\mathbf{A}^\top$ をカーネル行列に置き換えると何が得られるか。
   :exlabel:`ex_mdl-lagrangian-duality-ridge-dual`

<!--en:b81ac181-->
[ディスカッション](https://d2l.discourse.group/t/36860)

<!--en:5c6046df-->
双対性<br>**ラグランジュ双対関数・弱双対性と強双対性・スレーター条件・シャドープライス・フェンシェル共役・SVM と注水法**

<!--en:540d09bf-->
::: {.slide title="勾配が非ゼロのときの最適性の保証"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:0bb3660b-->
$\min_{\|\mathbf{x}\|_2 \le 1}\tfrac12\|\mathbf{x} - \mathbf{x}_0\|_2^2$（$\mathbf{x}_0 = (3, 4)$）に対する反復ソルバが、実行可能点 $\mathbf{x} = (0.55, 0.835)$（$f(\mathbf{x}) = 8.01 \ge p^\star$）で停止したとする。

<!--en:f701a7a6-->
- **制約なし判定が機能しない理由：**制約付き最適解 $\mathbf{x}^\star = (0.6, 0.8)$ では $\nabla f(\mathbf{x}^\star) = (-2.4, -3.2) \neq \mathbf{0}$ となる

<!--en:3e61db22-->
- **双対下界による保証：**任意の $\lambda \ge 0$ に対し円板上で $\lambda g(\mathbf{x}) \le 0$ となるため、$f(\mathbf{x}) + \lambda g(\mathbf{x})$ を $\mathbb{R}^2$ 全体で最小化すれば確実な下界 $q(\lambda) \le p^\star$ が保証される

<!--en:f595dfb0-->
::: {.slide title="制約の折り込みによる制約なし下界の構成"}

<!--en:1ca48427-->
[双対問題]{.kicker}

<!--en:09442e20-->
クリッピング問題（$f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$、$g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$、$\|\mathbf{x}_0\|^2 = 25$）において、任意の $\lambda \ge 0$ を固定する。

<!--en:b0f4da25-->
- **閉形式の制約なし最小化：**$\nabla_{\mathbf{x}}\mathcal{L}(\mathbf{x}, \lambda) = (\mathbf{x} - \mathbf{x}_0) + \lambda\mathbf{x} = \mathbf{0}$ とおくと、一意な最小点 $\mathbf{x}(\lambda) = \frac{\mathbf{x}_0}{1 + \lambda}$ が得られる

<!--en:a6f7d250-->
- **下界 $q(\lambda) = \mathcal{L}(\mathbf{x}(\lambda), \lambda)$ の評価：**
  $$q(\lambda) = \frac{25}{2}\frac{\lambda^2}{(1+\lambda)^2} + \frac{\lambda}{2}\left(\frac{25}{(1+\lambda)^2} - 1\right) = \frac{25}{2}\frac{\lambda}{1 + \lambda} - \frac{\lambda}{2}$$

<!--en:396871de-->
- **最も厳しい下界：**$q'(\lambda) = \frac{25}{2(1+\lambda)^2} - \frac12 = 0$ から $\lambda^\star = 4$ が選ばれ、$q(4) = 10 - 2 = \mathbf{8 = p^\star}$ となって $8.01$ が最適値から $0.01$ 以内だと直ちに証明される

<!--en:d8f6f97d-->
::: {.slide title="単位円板上の双対関数の可視化"}

<!--en:1ca48427-2-->
[双対問題]{.kicker}

<!--en:ef3dec2e-->
すべての $\lambda \ge 0$ が正当な下界 $q(\lambda) \le p^\star = 8$ を与える。

<!--en:b9addcaf-->
- $\lambda < 4$ のとき、$\mathbf{x}(\lambda)$ は円板の外側にある（$g > 0$）

<!--en:4a46c5ed-->
- $\lambda > 4$ のとき、$\mathbf{x}(\lambda)$ は真に内側へ引き込まれる（$g < 0$）

<!--en:37b63ff7-->
- $\lambda^\star = 4$ のとき、$\mathbf{x}(4) = (0.6, 0.8)$ は境界円上に乗る（$g = 0$）

<!--en:f521a4e5-->
::: {.slide title="ラグランジュ双対関数は常に凹関数"}

<!--en:1ca48427-3-->
[双対問題]{.kicker}

<!--en:0187b497-->
**ラグランジュ双対関数**は、$\mathcal{L}$ の $\mathbf{x} \in \mathcal{D}$ にわたる制約なし下限である。

<!--en:a91621ee-->
- **普遍的な凹性：**固定した各 $\mathbf{x}$ に対し、$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ は $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ に関してアフィンである。アフィン関数族の各点ごとの下限は常に凹であるため、**主問題が非凸でも $q$ は凹関数になる**

<!--en:488fad6a-->
::: {.slide title="弱双対性と双対ギャップ"}

<!--en:1ca48427-4-->
[双対問題]{.kicker}

<!--en:e15cea5a-->
凹である双対関数を双対実行可能な乗数（$\boldsymbol{\lambda} \succeq \mathbf{0}$）上で最大化することで、凸な**双対問題**が定義される。

<!--en:875372b1-->
- **弱双対性（$d^\star \le p^\star$）の 2 行の証明：**任意の主実行可能な $\mathbf{x}$（$g_i(\mathbf{x}) \le 0, h_j(\mathbf{x}) = 0$）と双対実行可能な $(\boldsymbol{\lambda} \succeq \mathbf{0}, \boldsymbol{\nu})$ に対し、各ペナルティ項は $\lambda_i g_i(\mathbf{x}) \le 0$ および $\nu_j h_j(\mathbf{x}) = 0$ を満たす
  $$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{z} \in \mathcal{D}}\mathcal{L}(\mathbf{z}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x}) \le f(\mathbf{x})$$

<!--en:78108cda-->
- **双対ギャップと停止保証：**非負の差 $p^\star - d^\star \ge 0$ を**双対ギャップ**と呼ぶ。任意の双対実行可能な組 $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ は、任意の主実行可能な候補点 $\mathbf{x}$ の準最適性を $f(\mathbf{x}) - p^\star \le f(\mathbf{x}) - q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ により直ちに上から抑える

<!--en:f79c54ae-->
::: {.slide title="スレーター条件と強双対性"}

<!--en:ba03859a-->
[強双対性]{.kicker}

<!--en:e4ac947e-->
- **スレーター条件（$\Rightarrow d^\star = p^\star$）：**主問題が凸（$f, g_i$ が凸、$h_j$ がアフィン）であり、ある点 $\bar{\mathbf{x}} \in \mathrm{relint}(\mathcal{D})$ で $g_i(\bar{\mathbf{x}}) < 0$（非アフィンな $g_i$）、$g_i(\bar{\mathbf{x}}) \le 0$（アフィンな $g_i$）、$h_j(\bar{\mathbf{x}}) = 0$ を満たすものが存在すれば、**強双対性が成り立ち（$d^\star = p^\star$）**、双対最適乗数 $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ が存在する

<!--en:7a1b5090-->
- **アフィン制約の簡略化：**すべての不等式が線形・アフィンである場合（LP や SVM）、単なる実行可能性だけで十分となる

<!--en:142df958-->
::: {.slide title="強双対性の支持直線による幾何構造"}

<!--en:ba03859a-2-->
[強双対性]{.kicker}

<!--en:861c196b-->
単一の不等式 $g(\mathbf{x}) \le 0$ に対し、達成可能な制約値と目的関数値のエピグラフ集合を作る。

<!--en:6a1a37d9-->
- **$p^\star$ と $q(\lambda)$ の可視化：**$p^\star$ は最小の高さ $t$ として $\mathcal{A}$ が縦軸 $u = 0$ 上で到達する値である。$q(\lambda) = \inf_{(u,t)\in\mathcal{A}}(t + \lambda u)$ の評価は傾き $-\lambda$ の直線を $\mathcal{A}$ を支えるまで下から押し上げる操作であり、$u = 0$ での切片が $q(\lambda)$ となる

<!--en:40b4247a-->
- **凸性とスレーター条件が効く理由：**凸性により $\mathcal{A}$ が凸集合となって $(0, p^\star)$ に支持直線が存在し、スレーター点 $u = g(\bar{\mathbf{x}}) < 0$ がその直線の垂直化を防ぐ

<!--en:0bc188e7-->
::: {.slide title="鞍点と min–max の順序交換可能性"}

<!--en:ba03859a-3-->
[強双対性]{.kicker}

<!--en:ef6d22c8-->
主最適値と双対最適値は、$\mathcal{L}$ 上の min–max ゲームにおける 2 通りの順序に対応する。

<!--en:11695b42-->
- **鞍点の同値性：**両最適解が達成され $d^\star = p^\star$ となることは、$(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ が鞍点（$\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) \le \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$）であることと同値である

<!--en:369a91ec-->
- **KKT 必要条件：**$\mathbf{x}$ に関する最小化から $\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0}$ が得られ、$d^\star = p^\star$ から $\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$ が強制される

<!--en:ed1deffa-->
::: {.slide title="深層学習における min–max ゲーム：GAN と頑健性"}

<!--en:ba03859a-4-->
[強双対性]{.kicker}

<!--en:5ccc136e-->
- **主解の復元と次元削減：**$f$ が狭義凸のとき、$\mathbf{x}^\star = \mathrm{argmin}_{\mathbf{x}}\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ により $n$ 変数の双対解から主解が復元される（例：$n \ll d$ のカーネル SVM）

<!--en:d5426d44-->
- **GAN と敵対的訓練の回転不安定性：**双線形ゲーム $\min_x\max_y xy$ では、同時勾配降下上昇法は毎ステップ半径を $\sqrt{1 + \eta^2} > 1$ 倍にし、外挿勾配法や減衰で安定化しない限り外側へ渦巻状に発散する

<!--en:55b6a930-->
::: {.slide title="シャドープライスとしてのラグランジュ乗数"}

<!--en:cdaeec6a-->
[感度と価格]{.kicker}

<!--en:50ad3a7f-->
制約を $g_i(\mathbf{x}) \le u_i$ および $h_j(\mathbf{x}) = v_j$ に緩和すると、最適値はどれだけ改善するか。摂動後の最適値を $p^\star(\mathbf{u}, \mathbf{v})$ とおく。

<!--en:028e0d65-->
- **大域的な感度下界：**強双対性と双対最適解 $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ のもとで、摂動問題で実行可能な任意の $\mathbf{x}$ は（$(\mathbf{u}, \mathbf{v})$ に対し）次式を満たす
  $$p^\star(\mathbf{0}, \mathbf{0}) = q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) \le f(\mathbf{x}) + \sum_{i=1}^m \lambda_i^\star g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j^\star h_j(\mathbf{x}) \le f(\mathbf{x}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v}$$
  摂動問題で実行可能な $\mathbf{x}$ にわたる下限をとると次式を得る
  $$p^\star(\mathbf{u}, \mathbf{v}) \ge p^\star(\mathbf{0}, \mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u} - \boldsymbol{\nu}^{\star\top}\mathbf{v}$$

<!--en:2ee71d49-->
- **微分によるシャドープライス：**$p^\star(\mathbf{u}, \mathbf{v})$ が $(\mathbf{0}, \mathbf{0})$ で微分可能なとき、次式が成り立つ
  $$\lambda_i^\star = -\frac{\partial p^\star}{\partial u_i}(\mathbf{0}, \mathbf{0}), \qquad \nu_j^\star = -\frac{\partial p^\star}{\partial v_j}(\mathbf{0}, \mathbf{0})$$

<!--en:da50e5fc-->
::: {.slide title="重み減衰とノルムバジェットの同値性"}

<!--en:cdaeec6a-2-->
[感度と価格]{.kicker}

<!--en:75e10d57-->
凸な $\ell(\mathbf{w})$ に対し、$\ell_2$ ペナルティを加えることは重みをノルム球に制約することと解の経路として同値である。

<!--en:17c7c1c0-->
- **同一の KKT 停留条件：**どちらも $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ を共有する

<!--en:99d55019-->
- **全単射の対応：**ペナルティ問題の最小点 $\mathbf{w}_\lambda$ は半径 $r = \|\mathbf{w}_\lambda\|_2$ の球制約問題を解き、球制約の乗数は $\lambda^\star = -\partial p^\star/\partial(\tfrac12 r^2)$ を満たす

<!--en:18ad2d61-->
::: {.slide title="数値的検証：ペナルティと球への射影"}

<!--en:cdaeec6a-3-->
[感度と価格]{.kicker}

<!--en:754fc022-->
凸最小二乗損失 $\ell(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|_2^2$（$30 \times 5$）上で、$\lambda \in \{0.1, 1.0, 10.0\}$ に対しペナルティ付きリッジ問題を閉形式で解き、$r = \|\mathbf{w}_\lambda\|_2$ とおいて半径 $r$ の球上で PGD を実行する。

<!--en:2c96b715-->
- ノルム球上の PGD はペナルティ問題の解 $\mathbf{w}_\lambda$ を機械精度（$10^{-16}$）で復元し、$-\mathbf{w}^\top\nabla\ell(\mathbf{w})/r^2$ から復元した乗数も小数点以下 6 桁すべてで $\lambda$ と一致する

<!--en:96041f54-->
::: {.slide title="フェンシェル凸共役：支持直線"}

<!--en:d2ab5ffe-->
[凸共役]{.kicker}

<!--en:65a75c17-->
**凸共役**（またはフェンシェル共役）$f^*(\mathbf{y})$ は、関数 $f(\mathbf{x})$ を傾き $\mathbf{y}$ の支持超平面の切片によって符号化する。

<!--en:28214bb8-->
- **常時凸性と二重共役性：**$f^*$ は $\mathbf{y}$ のアフィン関数族の各点ごとの上限であるため常に凸になる。$f$ が固有・閉・凸ならば $f^{**} = f$ が成り立つ

<!--en:dd0b2ccc-->
- **フェンシェル・ヤングの不等式（$\forall\,\mathbf{x}, \mathbf{y}$）：**
  $$\mathbf{y}^\top\mathbf{x} \le f(\mathbf{x}) + f^*(\mathbf{y})$$
  凸かつ微分可能な $f$ に対し、等号成立は $\mathbf{y} = \nabla f(\mathbf{x})$ と同値である

<!--en:6e5df341-->
::: {.slide title="4 つの基本的なフェンシェル共役対"}

<!--en:d2ab5ffe-2-->
[凸共役]{.kicker}

<!--en:1847b3ff-->
- **二次関数：**$\tfrac12\|\mathbf{x}\|_2^2$ は自己共役（$\tfrac12\|\mathbf{y}\|_2^2$）であり、$\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$（$\mathbf{A} \succ \mathbf{0}$）の共役は $\tfrac12\mathbf{y}^\top\mathbf{A}^{-1}\mathbf{y}$ となる

<!--en:7af9ec43-->
- **指数関数 $\leftrightarrow$ エントロピー：**$f(x) = e^x$ の共役は $f^*(y) = y\log y - y$（$y > 0$ のとき、$x^\star = \log y$ で達成）となる

<!--en:149fc90c-->
- **$\ell_1$ ペナルティ $\leftrightarrow$ $\ell_\infty$ ボックス：**$f(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$ の共役は $I_{\{\|\mathbf{y}\|_\infty \le \lambda\}}(\mathbf{y})$ となる

<!--en:775bb9c9-->
::: {.slide title="線形制約付き問題の 1 行での双対導出"}

<!--en:d2ab5ffe-3-->
[凸共役]{.kicker}

<!--en:61bf9ddb-->
任意の線形制約付き問題 $\min_{\mathbf{x}} f(\mathbf{x})$（制約 $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$ または $\mathbf{A}\mathbf{x} = \mathbf{b}$）に対し、双対関数 $q(\boldsymbol{\lambda})$ はフェンシェル共役 $f^*$ に直接帰着する。

<!--en:8760bba9-->
- **具体例（半平面内の最近接点）：**$f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|_2^2$ を制約 $\mathbf{a}^\top\mathbf{x} \le b$（$\mathbf{a} = (1, 2)$、$\|\mathbf{a}\|^2 = 5$、$b = -1$）のもとで最小化する

<!--en:8f97ead0-->
- **$f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|_2^2$ を用いた 1 行での双対導出：**
  $$q(\lambda) = -f^*(-\lambda\mathbf{a}) - b\lambda = -\tfrac12\lambda^2\|\mathbf{a}\|_2^2 - b\lambda = \lambda - \tfrac52\lambda^2 \quad (\lambda \ge 0)$$

<!--en:742ac6eb-->
- $q'(\lambda) = 1 - 5\lambda = 0$ から $\lambda^\star = \tfrac15$、$d^\star = p^\star = \tfrac{1}{10}$、および主最適解 $\mathbf{x}^\star = -\lambda^\star\mathbf{a} = (-0.2, -0.4)$ が得られる

<!--en:f59a4c1d-->
::: {.slide title="log-sum-exp と負のシャノンエントロピー"}

<!--en:d2ab5ffe-4-->
[凸共役]{.kicker}

<!--en:c9be4cad-->
$\mathrm{lse}(\mathbf{x}) = \log\sum_{i=1}^n e^{x_i}$ のフェンシェル共役は、確率単体 $\Delta$ 上の負のシャノンエントロピーである。

<!--en:fd0dbf9d-->
- **二重共役の変分法則（$\mathrm{lse}^{**} = \mathrm{lse}$）：**$\mathrm{lse}(\mathbf{x}) = \max_{\mathbf{p} \in \Delta}(\mathbf{p}^\top\mathbf{x} + H(\mathbf{p}))$ であり、$\mathbf{p}^\star = \mathrm{softmax}(\mathbf{x})$ で一意に達成される

<!--en:8ef88646-->
::: {.slide title="サポートベクターマシン（SVM）の双対問題の導出"}

<!--en:158c57bf-->
[双対の例：SVM]{.kicker}

<!--en:f2f8fe4e-->
線形分離可能なデータ $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$（$y_i \in \{\pm 1\}$）に対し、ハードマージン SVM は次の凸二次計画問題を解くことで幾何マージン $2/\|\mathbf{w}\|_2$ を最大化する。

<!--en:57672f7c-->
- **ラグランジアンの構成と最小化（$\alpha_i \ge 0$）：**
  $$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \tfrac12\|\mathbf{w}\|_2^2 + \sum_{i=1}^n \alpha_i\big(1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\big)$$
  1. **$b$ に関する最小化：**$\partial_b\mathcal{L} = -\sum_{i=1}^n \alpha_i y_i = 0$（そうでなければ $\inf_b \mathcal{L} = -\infty$）
  2. **$\mathbf{w}$ に関する最小化：**$\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = \mathbf{0} \implies \mathbf{w}^\star = \sum_{i=1}^n \alpha_i^\star y_i \mathbf{x}_i$

<!--en:c1d32d3d-->
::: {.slide title="サポートベクターとカーネルトリック"}

<!--en:158c57bf-2-->
[双対の例：SVM]{.kicker}

<!--en:3b063b7b-->
$\mathbf{w}^\star = \sum_i \alpha_i^\star y_i \mathbf{x}_i$ と $\sum_i \alpha_i y_i = 0$ を $\mathcal{L}$ に代入すると、**SVM の双対問題**が得られる。

<!--en:3d50d0bf-->
- **サポートベクターのスパース性：**相補性条件により、$\alpha_i^\star > 0$ となるのはマージン上にある有効な点 $y_i(\mathbf{w}^{\star\top}\mathbf{x}_i + b^\star) = 1$ のみに限られる

<!--en:f780dc02-->
- **カーネルトリック：**訓練と予測は $\mathbf{x}_i$ に内積 $\mathbf{x}_i^\top\mathbf{x}_j \mapsto k(\mathbf{x}_i, \mathbf{x}_j)$ のみを通じて依存する

<!--en:3207b5fb-->
::: {.slide title="射影勾配上昇法による SVM 双対問題の解法"}

<!--en:158c57bf-3-->
[双対の例：SVM]{.kicker}

<!--en:0ed65246-->
オフセット $b$ を $\tilde{\mathbf{w}} = (\mathbf{w}, b)$ に折り込み $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$ とすると、双対問題は非負象限上の $\max_{\boldsymbol{\alpha} \succeq \mathbf{0}}\big(\mathbf{1}^\top\boldsymbol{\alpha} - \tfrac12\boldsymbol{\alpha}^\top\mathbf{Q}\boldsymbol{\alpha}\big)$ となり、射影勾配上昇法で解ける。

<!--en:b00c4e70-->
- 正の乗数 $4$ 個はすべてマージン $1.0000$ に位置し、$p^\star = d^\star = 41/98 \approx 0.418367$ でギャップは $10^{-16}$ となる

<!--en:c10b61ee-->
::: {.slide title="チャネル間の注水法による電力配分"}

<!--en:bc8a5ac0-->
[双対の例：注水法]{.kicker}

<!--en:c11bcb6d-->
総電力 $P > 0$ を $n$ 個のチャネル（雑音フロア $N_i > 0$）に配分し、合計シャノン容量を最大化する。

<!--en:c63d1530-->
- **KKT 停留条件：**乗数 $\nu$（$\sum_i P_i = P$ に対応）と $\lambda_i \ge 0$（$-P_i \le 0$ に対応、$\lambda_i P_i = 0$）のもと、停留条件は $\frac{1}{N_i + P_i} = \nu - \lambda_i$ となる

<!--en:d9cce500-->
- **一様な水位 $\theta = 1/\nu^\star$：**稼働チャネルは $\theta$ まで満たされ（$P_i^\star = \max(\theta - N_i,\, 0)$）、$\theta$ は次式を満たす
  $$\sum_{i=1}^n \max(\theta - N_i,\, 0) = P$$

<!--en:142662f4-->
::: {.slide title="注水法とそのシャドープライスの検証"}

<!--en:bc8a5ac0-2-->
[双対の例：注水法]{.kicker}

<!--en:325c41bb-->
雑音フロア $\mathbf{N} = (0.1, 0.4, 0.8, 1.6, 2.5)$ と総バジェット $P = 3.0$ に対し、二分法で水位 $\theta$ を求め、ラグランジュ乗数 $\nu^\star = 1/\theta$ が限界容量 $\mathrm{d}U^\star / \mathrm{d}P$ に等しいことを検証する。

<!--en:3292990f-->
- 雑音が小さい上位 3 チャネルは $N_i + P_i^\star = \theta = 1.433333$ まで満たされ、雑音の大きい 2 チャネル（$N_i > \theta$）は乾いたまま（$P_4^\star = P_5^\star = 0$）となり、$\nu^\star = 1/\theta = 0.697674$ は $\mathrm{d}U^\star/\mathrm{d}P$ と 6 桁で一致する

<!--en:23951907-->
::: {.slide title="強双対性が破れるとき：明示的な双対ギャップ"}

<!--en:87f9b99b-->
[非凸問題の双対性]{.kicker}

<!--en:70c8d4fc-->
強双対性に凸性が必要な理由を見るため、*凹*の目的関数 $f(x) = -x^2$ を $\mathcal{D} = [0, 1]$ 上で制約 $g(x) = x - \tfrac12 \le 0$ のもとで最小化する（$p^\star = f(\tfrac12) = -\tfrac14$）。

<!--en:e61145f4-->
- $x = 1/4$ が狭義実行可能（$g(1/4) = -1/4 < 0$）であるにもかかわらず、非凸性により真に正の双対ギャップ $p^\star - d^\star = 1/4$ が残る

<!--en:72a18dda-->
::: {.slide title="理解度チェック：双対下界・価格・水位"}

<!--en:26a317d8-->
[理解度チェック]{.kicker}

<!--en:aad51b96-->
双対による保証、シャドープライス、注水法の理解を確かめよう。

<!--en:24bd50c0-->
1. 円板クリッピング問題（$p^\star = 8$）で、ソルバが $f(\mathbf{x}) = 8.01$ の実行可能点で停止した。$\lambda^\star = 4$ を求めずに、未最適化の乗数 $\lambda = 3$（$q(3) = 7.875$）からどのような準最適性の上界が保証されるか。

<!--en:fa0e6d78-->
2. $(3, 4)$ の単位円板（$\tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$）へのクリッピングでは $p^\star(0) = 8$、最適乗数 $\lambda^\star = 4$ である。制約を $u = \tfrac12$ だけ緩和（半径 $r = \sqrt{2}$ に拡大）したとき、シャドープライスは $p^\star(\tfrac12)$ にどのような下界を保証し、真の $p^\star(\tfrac12)$ とどう比較されるか。

<!--en:4a34f4e3-->
3. 雑音フロア $(0.1, 0.4, 0.8, 1.6, 2.5)$ の注水法で、厳密な水位 $\theta = 1.433333$（$P = 3$ の場合）をどのように手計算で求めたか。

<!--en:e41de942-->
**解答：****(1)** 弱双対性より $p^\star \ge q(3) = 7.875$ が保証され、直ちに $f(\mathbf{x}) - p^\star \le 8.01 - 7.875 = \mathbf{0.135}$ が証明される。**(2)** 大域的感度下界より $p^\star(\tfrac12) \ge p^\star(0) - \lambda^\star u = 8 - 4(\tfrac12) = \mathbf{6.00}$ が保証される。真の新しい最適値は $\tfrac12(5 - \sqrt{2})^2 \approx \mathbf{6.43} > 6.00$ である。**(3)** 雑音が小さい上位 $3$ チャネルが稼働すると推測すると $\sum_{i=1}^3(\theta - N_i) = 3\theta - (0.1 + 0.4 + 0.8) = 3 \implies 3\theta = 4.3 \implies \theta = \mathbf{4.3/3 \approx 1.4333}$ となり、$0.8 < 1.4333 < 1.6$ のため整合する。

<!--en:2c677b90-->
::: {.slide title="まとめ：ラグランジュ双対性と共役"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:ac4e390e-->
| 概念・定理 | 数学的記述 | 必要な仮定 | 実用上の意義 |

<!--en:3f9e48a0-->
| **ラグランジュ双対 $q(\boldsymbol{\lambda}, \boldsymbol{\nu})$** | $\inf_{\mathbf{x} \in \mathcal{D}}\big(f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})\big)$ | なし | 主問題が非凸でも $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ に関して常に凹関数 |

<!--en:8503bd9b-->
| **弱双対性** | $d^\star = \sup_{\boldsymbol{\lambda} \succeq \mathbf{0}, \boldsymbol{\nu}} q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le p^\star$ | なし | 双対実行可能な任意の $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ が $p^\star$ の下界を保証 |

<!--en:32eadba3-->
| **強双対性（スレーター）** | $d^\star = p^\star$ および $\mathcal{L}$ の鞍点 | 主問題の凸性＋狭義内点 $\bar{\mathbf{x}}$ | 双対ギャップがゼロ、KKT 条件が必要十分 |

<!--en:f2c8c8ba-->
| **シャドープライス** | $p^\star(\mathbf{u}, \mathbf{v}) \ge p^\star(\mathbf{0}, \mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u} - \boldsymbol{\nu}^{\star\top}\mathbf{v}$ | 強双対性（$d^\star = p^\star$） | $\lambda_i^\star = -\partial p^\star/\partial u_i$、重み減衰とノルムバジェットを対応づける |

<!--en:25c79312-->
| **フェンシェル共役 $f^*(\mathbf{y})$** | $\sup_{\mathbf{x}}(\mathbf{y}^\top\mathbf{x} - f(\mathbf{x}))$、$\mathbf{y}^\top\mathbf{x} \le f(\mathbf{x}) + f^*(\mathbf{y})$ | 固有閉凸なら $f^{**} = f$ | 線形制約双対は $-f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}$、$\mathrm{lse}^* = -H$ |
