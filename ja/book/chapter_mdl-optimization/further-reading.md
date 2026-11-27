<!--en:7902ddfd-->
# 関連文献

<!--en:02430e52-->
本節では本章の範囲を超えた話題を紹介するが、そのすべてが同じ緊急度を持つわけではない。そのうち 5 つはニューラルネットワークの訓練に直接関わるため、それぞれの見出しの先頭に配置している。すなわち、安定性の縁（edge of stability）、反復点の平均化（iterate averaging）、補間（interpolation）、フィッシャー行列、および鞍点法（saddle-point method）である。残りの話題は古典的な最適化理論および数値解析からのものであり、最後に一般的な教科書、講義、サーベイをまとめる。なお、訓練の実践的な側面については :numref:`sec_optimization-further-reading` で概観している。本書の他の箇所は本節の内容に一切依存しないため、関心のある項目を選んで自由に拾い読みするとよい。

<!--en:7edc1d9b-->
## 勾配法

<!--en:192ccab0-->
### 安定性の縁における自己安定化

<!--en:15ea91fa-->
ニューラルネットワークをフルバッチ勾配降下法で訓練すると、驚くべき現象が起こる。損失の鋭さ（sharpness、ヘッセ行列の固有値のうち最大のもの）が安定限界 $2/\eta$ まで上昇したのち、発散するのではなくちょうどその付近にとどまり続けるのである :cite:`Cohen.Kaur.Li.ea.2021`。:numref:`subsec_mdl-edge-of-stability` の二次モデルは、鋭さが $2/\eta$ を超えた瞬間に最大固有モードが増大する理由を示すことでこの振る舞いの前半を説明するが、純粋な二次モデルでは際限のない発散が予測されるため、何が鋭さを再び引き下げるのかを説明できない。:citet:`Damian.Nichani.Lee.2023` は、テイラー展開にもう 1 つだけ項、すなわち最大固有ベクトル方向の 3 次の項を残すことでこの謎を解明した。その方向に沿って振動が大きくなると、この 3 次の結合が反復点を鋭さそのものの勾配と逆向きに押し出す。それによって鋭さが再び $2/\eta$ 未満へと下がり、次のサイクルが始まるまで振動が減衰する。この 3 次の項がなければ、振動は歯止めなく増大し続けるだろう。

<!--en:949ee687-->
### 最適多項式と共役勾配法

<!--en:17086891-->
二次関数 $f = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$（固有値が $[\mu, L]$ に含まれ、最小点が $\mathbf{0}$ であるもの）の最小化を考えよう。$k$ ステップ目の更新がそれまでに得られた勾配の張る空間（span）にとどまる任意の一次手法は、$\mathbf{x}_k = p_k(\mathbf{A})\mathbf{x}_0$ という形の反復点を生成する。ここで、$p_k$ は次数が高々 $k$ で $p_k(0) = 1$ を満たす多項式である。このように見れば、二次関数に対する高速な勾配法を設計することは、区間 $[\mu, L]$ 全体にわたって可能な限り小さくなる多項式を選ぶことと等価である。区間全体での最大値を最小にする多項式はスケーリングされたチェビシェフ多項式であり、その最大値は $2\rho^k/(1 + \rho^{2k}) \approx 2\rho^k$ となる。ここで縮小率 $\rho = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ は、ヘビーボール法で現れたものと完全に一致する（:numref:`subsec_mdl-momentum-acceleration`）。次元 $d$ が $k$ に比べて大きいとき、固有値は区間全体に広がり得るため、この族に属するいかなる手法もその最悪ケースの収束率（convergence rate）を上回ることはできない。共役勾配法（conjugate gradient method）は実用上さらに優れた振る舞いを示す。各ステップにおいて、$f$ を*実際の*行列 $\mathbf{A}$ と初期点 $\mathbf{x}_0$ に対して最小化する多項式を、$\mu$ や $L$ を一切知ることなく自動的に見つけ出し、厳密演算のもとでは高々 $d$ ステップで終了するのである。:citet:`Nocedal.Wright.2006`（第 5 章）は、標準的な共役勾配法とその前処理付き版の両方を導出している。

<!--en:35b51398-->
### 半正定値計画による収束率の保証

<!--en:085b6ec1-->
タイトな収束率を手計算で証明するのは、非常に繊細な作業として知られている。:numref:`subsec_mdl-convergence-rates` における手計算の証明ではステップごとの縮小率として $1 - \mu/L$ が得られたが、同項で引用した厳密な縮小率は $(1 - \mu/L)^2$ であり、必要なステップ数は半分になる。このよりタイトな評価は、*性能推定*（performance estimation）と呼ばれる技法から得られる。:citet:`Drori.Teboulle.2014` は、巧みな代数的変形を手作業で探す代わりに、すべての $L$ 滑らかな凸関数にわたるアルゴリズムの最悪ケース性能を、反復点と勾配がなすグラム行列（すべてのペアの内積を並べた表）上の最適化問題として定式化した。この問題を上から評価すると小さな半正定値計画問題（semidefinite program、SDP）に帰着し、その双対解から人間が検証可能な証明が文字通り得られる。:citet:`Taylor.Hendrickx.Glineur.2018` はまさにこの方法で二乗の縮小率を導出した。また制御理論の視点に立つ :citet:`Lessard.Recht.Packard.2016` は、勾配を $\mu$ と $L$ のみで制約された線形力学系内部の未知の非線形性とみなし、ステップ数によらずサイズが一定の半正定値計画問題によって収束率を保証している。注目すべきことに、同じ枠組みは反例の発見にもつながり、ヘビーボール法が収束しない関数（:numref:`subsec_mdl-momentum-acceleration`）もこうして発見された。

<!--en:eaaf3ff2-->
## 確率的手法と適応的手法

<!--en:3b625f59-->
### 定数ステップによる平均化

<!--en:34a54772-->
:numref:`subsec_mdl-stochastic-gradients` で見たように、定数ステップの SGD はステップサイズ $\eta$ に比例するノイズフロアに落ち着く。時間とともに $\eta$ を縮小すればこのノイズフロアは解消されるが、終盤の進行が遅くなるうえ、スケジュールが調整に対して敏感になってしまう。反復点を平均化すれば、より明快な解決策が得られる。個々の反復点は最小点の周囲で跳ね回り続けるものの、それらの移動平均をとることで平均ゼロのノイズが相殺されるからである。:citet:`Polyak.Juditsky.1992` は、$1/k$ より緩やかに減衰するステップサイズで得られた反復点を平均化すると、漸近的に最適な共分散が達成されることを示した。最小二乗法に対しては、:citet:`Bach.Moulines.2013` がさらに踏み込み、*定数*のステップサイズですでに十分であることを示している。すなわち、平均化された反復点は $1/n$ のオーダーの誤差に $n$ 個のデータ例で到達し、これには強凸性すら必要ない。

<!--en:d7549745-->
### 補間

<!--en:1535d3a4-->
古典的な確率的最適化では、最適点においても個々の訓練例がパラメータを互いに相反する方向へ引っ張り合うため、避けられないノイズフロアが生じると仮定する。これに対し、現代の過剰パラメータ化されたニューラルネットワークは、しばしば大きく異なるレジームで動作する。モデルがすべての訓練例に厳密に適合するとき、最小点では各訓練例の勾配がすべてゼロになる。するとノイズフロアは消失し、強凸な損失のもとでは、定数ステップの SGD がいかなる分散低減（variance reduction）も用いずに線形収束する（:numref:`sec_mdl-variance-reduction`）。:citet:`Ma.Bassily.Belkin.2018` は最小二乗法についてこの補間レジームを解析し、それ以上ミニバッチを大きくしても訓練が高速化しなくなる臨界バッチサイズ（critical batch size）を特定した。興味深いことに、このしきい値を支配しているのは勾配ノイズではなく、訓練例ごとの曲率とフルバッチの曲率の比である。また、勾配の分散が全勾配（full gradient）のノルムの二乗に比例して縮小するという関連条件である*強増大条件*（strong growth condition）のもとでは、SGD を加速してモメンタムと同等の収束率を達成することさえ可能になる :cite:`Vaswani.Bach.Schmidt.2019`。

<!--en:dda69add-->
### フィッシャー行列とガウス・ニュートン行列

<!--en:a1e581a9-->
K-FAC（:numref:`subsec_mdl-preconditioning-ladder`）などの二次近似は、フィッシャー情報行列 $\mathbf{F} = \mathbb{E}[\nabla\log p\,\nabla\log p^\top]$ の近似を目的とする。ここで期待値は、モデル自身の予測分布（predictive distribution）からラベルを抽出してとる :cite:`Amari.1998,Martens.Grosse.2015`。ソフトマックス交差エントロピーや二乗誤差のような標準的な損失に対しては、$\mathbf{F}$ はガウス・ニュートン行列 $\mathbf{J}^\top\mathbf{H}_{\mathrm{out}}\mathbf{J}$ と一致する（$\mathbf{H}_{\mathrm{out}}$ はネットワークの出力に関する損失のヘッセ行列である）。より一般に、この恒等式はネットワークの出力が指数分布族（exponential family）の尤度の自然パラメータ（natural parameter）を表すときには常に成り立つ（:numref:`sec_mdl-exponential-family`）。実務上よくある落とし穴は、モデルからのサンプリングの代わりにデータセットの正解ラベルにわたって同じ外積を平均した*経験*フィッシャー行列（empirical Fisher）で代用してしまうことである。これは計算が安価であり、コードベース上でも単に「Fisher」と表記されることが多いが、一般には真のフィッシャー情報行列にもガウス・ニュートン行列にも一致しない :cite:`Martens.2020`。

<!--en:afea960d-->
### ミラー降下法

<!--en:e4c6580f-->
AdaGrad を計量の観点から見ると（:numref:`subsec_mdl-per-coordinate`）、ステップ長は二次のユークリッドノルムで測られる。この幾何構造は、あらゆる方向が同等に見える $\mathbb{R}^d$ のような制約なし空間には適しているが、各座標が非負かつ総和が 1 でなければならず乗法的な更新の方がはるかに自然である確率単体（probability simplex）の上では扱いにくい。ミラー降下法（mirror descent）は、勾配ステップの近接形式における二乗距離 $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ をブレグマンダイバージェンス（Bregman divergence）$D_\phi(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x}) - \phi(\mathbf{y}) - \nabla\phi(\mathbf{y})^\top(\mathbf{x} - \mathbf{y})$ に置き換えることで、勾配降下法の幾何構造を適応させる :cite:`Nemirovski.Yudin.1983,Beck.Teboulle.2003`。$\phi$ として負のエントロピーを選ぶと、更新は指数勾配則（exponentiated gradient rule）$x_i \leftarrow x_i e^{-\eta g_i} / \sum_j x_j e^{-\eta g_j}$ となる。勾配が座標ごとに有界であるとき、この選択によって :numref:`subsec_mdl-per-coordinate` で定義したリグレットは、$d$ 次元単体上において射影勾配降下法（projected gradient descent）の $\sqrt{d}$ スケーリングからはるかに緩やかな $\sqrt{\log d}$ へと改善される。

<!--en:6cf784ca-->
## 制約と双対性

<!--en:ee84a291-->
### 鞍点法

<!--en:4952e533-->
ラグランジュ双対性（:numref:`subsec_mdl-strong-duality`）で登場した鞍点は、GAN の訓練（:numref:`sec_basic_gan`）をはじめとする 2 人ゲームの均衡としても直接現れる。しかし残念ながら、一方のプレイヤーには勾配降下法を、もう一方のプレイヤーには勾配上昇法を同時に適用する素朴な戦略はうまく機能しない。単純な双線形ゲーム $xy$ においてさえ、任意の正のステップサイズに対して鞍点から外側へと渦を巻いて発散してしまうのである（演習問題 :exref:`ex_mdl-lagrangian-duality-descent-ascent`）。外挿勾配法（extragradient method）は、一歩先を覗き見ることでこの渦を断ち切る。すなわち、試行ステップを進め、その先読み点で勾配を評価し、得られた勾配を元の反復点から適用する。リプシッツ勾配を持つ凸凹（convex–concave）問題では、その平均化された反復点は $O(1/k)$ の速さで収束する :cite:`Nemirovski.2004`。また楽観的手法（optimistic method）は、直前のステップの勾配を試行勾配の安価な代用として再利用することで、半分の勾配コストで同様の安定化効果を達成しており、:citet:`Daskalakis.Ilyas.Syrgkanis.ea.2018` はこの発想を GAN の訓練の安定化に応用して成果を上げている。

<!--en:abb5d03e-->
### 問題クラスと内点法

<!--en:f31a59a4-->
:numref:`subsec_mdl-worked-duals` で手計算により解いた双対問題は二次計画問題（quadratic program）であり、これは線形計画問題、二次計画問題、二次錐計画問題（second-order cone program）、半正定値計画問題という凸最適化クラスの標準的な入れ子階層の中に位置づけられる。現代の内点法（interior-point method）は、これらすべてのクラスを多項式時間で高い精度まで解くことができる。内点法は不等式 $g_i(\mathbf{x}) \le 0$ を直接扱う代わりに、滑らかな対数バリア（logarithmic barrier）$-\tfrac1s\sum_i \log(-g_i(\mathbf{x}))$ を目的関数に加え、$s$ を大きくしながら制約なし最小点を追跡することで、境界へと向かう*中心パス*（central path）を描き出す。このパス上の各点において、量 $-1/(s\,g_i(\mathbf{x}))$ は自動的に双対実行可能な乗数をなし、双対ギャップ（duality gap）$m/s$（$m$ は制約の個数）を保証する。これにより、:numref:`sec_mdl-lagrangian-duality` で手作業により構成した最適性の証明書が自動的に得られる :cite:`Boyd.Vandenberghe.2004,Nesterov.2018`。

<!--en:26a2b22f-->
### 作用素分割と ADMM

<!--en:e35817f6-->
:numref:`subsec_mdl-proximal-operators` の近接勾配ステップは、滑らかな損失と単一の単純な非滑らかペナルティの和を扱う。これに対し、構造を持つ多くの問題は合成形式 $\min_{\mathbf{x}} f(\mathbf{x}) + h(\mathbf{A}\mathbf{x})$ をとる。ここで両方の項は凸であり、ともに非滑らかであってもよく、そのフェンシェル双対（Fenchel dual）は $\max_{\mathbf{y}} -f^*(-\mathbf{A}^\top\mathbf{y}) - h^*(\mathbf{y})$ となる :cite:`Rockafellar.1970`。2 つの項を切り離して個別に扱えるようにするため、補助変数 $\mathbf{z} = \mathbf{A}\mathbf{x}$ を導入し、二次ペナルティ $\tfrac{\tau}{2}\|\mathbf{A}\mathbf{x} - \mathbf{z}\|^2$ をラグランジアンに加える。まず $\mathbf{x}$ について最小化し、次に $\mathbf{z}$ について最小化し（これは $h$ に対する近接ステップに帰着する）、最後に $\mathbf{y}$ について双対上昇ステップをとる操作を交互に繰り返すと、交互方向乗数法（alternating direction method of multipliers、ADMM）が得られ、これは任意の $\tau > 0$ に対して収束する。:citet:`Boyd.Parikh.Chu.ea.2011` はこの分割によって大規模な最適化問題を複数の計算機へ自然に分散できることを示しており、:citet:`Parikh.Boyd.2014` は代表的な正則化項や制約に対する閉形式の近接作用素をまとめた便利なカタログを提供している。

<!--en:a5d96aa1-->
### フランク・ウルフ法

<!--en:023bffd4-->
射影勾配降下法では、反復ごとに制約集合への射影が必要になる。これはボックス制約や $\ell_2$ 球に対しては容易であるが、核ノルム球（特異値の総和が与えられた上界以下である行列の集合）の上では、1 回の射影に完全な SVD のコストがかかる。フランク・ウルフ法（Frank–Wolfe method）:cite:`Frank.Wolfe.1956` は、制約集合上で目的関数の線形近似を代わりに最小化することで射影を完全に回避する。具体的には、$\mathbf{s}_k = \operatorname{argmin}_{\mathbf{s} \in C} \nabla f(\mathbf{x}_k)^\top\mathbf{s}$ を求め、凸結合（convex combination）$\mathbf{x}_{k+1} = (1 - \gamma_k)\mathbf{x}_k + \gamma_k\mathbf{s}_k$ をステップサイズ $\gamma_k = 2/(k + 2)$ でとる。核ノルム球上ではこの線形部分問題に必要なのは勾配の最大特異ベクトル対だけであり、確率単体上では最小の座標を 1 つ見つける操作に帰着する。この手法は $O(LD^2/k)$ の速さで収束し（$D$ は制約集合の直径）、さらに利点として、量 $\nabla f(\mathbf{x}_k)^\top(\mathbf{x}_k - \mathbf{s}_k)$ が追加の計算なしに毎ステップ $f(\mathbf{x}_k) - f^\star$ の上界を与える :cite:`Jaggi.2013`。

<!--en:a567f2b2-->
## 浮動小数点

<!--en:ae19a077-->
### 確率的丸め誤差解析

<!--en:0c9c491c-->
:numref:`subsec_mdl-catastrophic-cancellation` における総和の最悪ケースの上界は $nu$ として線形に増大する（$n$ 個の項を順次加算する場合）が、実際に計測される丸め誤差は通常それよりはるかに小さい。なぜなら最悪ケース解析では、個々の丸め誤差がすべて同じ最悪の方向へ積み重なると仮定しているからである。これに対し :citet:`Higham.Mary.2019` は、各ステップの丸め誤差を独立で平均ゼロの確率変数としてモデル化し、高い確率で成り立つオーダー $\sqrt{n}\,u$ の上界を証明した。これは実験で典型的に見られるランダムウォーク的な相殺と一致する。同様に重要な点として、この解析は楽観的な $\sqrt{n}\,u$ の振る舞いがいつ破綻するかも明らかにしている。すなわち、$10^7$ 個の $0.1$ を足し合わせる同項の例のように、長い区間にわたって加算がすべて同じ方向へ丸められ、丸め誤差どうしが強く相関する場合である。

<!--en:8f03209a-->
### 増大係数と反復改良法

<!--en:9bbef36c-->
:numref:`subsec_mdl-conditioning-revisited` では、部分ピボット選択付き LU 分解が実用上は後退安定であることを述べた。厳密に言えば、その後退誤差は $n$ の低次多項式と $\rho_n u$ の積で上から抑えられる。ここで*増大係数*（growth factor）$\rho_n$ は、消去の過程で生じる最大の中間成分と $\mathbf{A}$ の最大成分との比を測る量である。最悪ケースでは、Wilkinson の有名な病的行列によって $\rho_n$ は $2^{n-1}$ にまで達する。しかし実際の計算で現れるほぼすべての行列では $\rho_n$ は穏やかな値にとどまり、万全の保証が必要な場合でも、QR 分解を用いれば約 2 倍の演算コストで指数関数的な増大を完全に排除できる :cite:`Higham.2002,Trefethen.Bau.1997`。すべての計算に倍精度のコストを払うことなく高い精度を回復するため、混合精度の反復改良法（iterative refinement）では、$\mathbf{A}$ を fp32 で安価に分解し、残差を fp64 で評価して、その低精度の因子を用いて補正量を求める。また :numref:`sec_mdl-numerical-stability-conditioning` の最後の演習問題では、この基本的なループが停滞するまでに $\mathbf{A}$ がどこまで悪条件になり得るかを調べる。この低精度の因子を、一般の連立一次方程式に対する反復ソルバである GMRES の内部で前処理行列として代わりに用いれば、混合精度の反復改良法をはるかに悪条件な行列へも適用できるようになる :cite:`Carson.Higham.2018`。

<!--en:06ac78b6-->
## 最適化全般の参考資料

<!--en:f3e47eb0-->
**書籍**

<!--en:416818bc-->
- 『[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)』（Boyd、Vandenberghe 著）：凸集合と凸関数、双対性、および内点法を扱っており、無料の [PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) が公開されている。:numref:`sec_mdl-convexity`、:numref:`sec_mdl-constrained-optimization-duality`、:numref:`sec_mdl-lagrangian-duality` の基本文献である。

<!--en:3bf8eea1-->
- 『[Numerical Optimization](https://doi.org/10.1007/978-0-387-40065-5)』（Nocedal、Wright 著、第 2 版、Springer）：直線探索、信頼領域法、準ニュートン法、共役勾配法、および制約付き最適化を、:numref:`sec_mdl-gradient-based-optimization`、:numref:`sec_mdl-curvature-methods`、:numref:`sec_mdl-constrained-optimization-duality` よりも踏み込んで解説している。

<!--en:520d06d2-->
- 『[Lectures on Convex Optimization](https://doi.org/10.1007/978-3-319-91578-4)』（Nesterov 著、第 2 版、Springer）：計算量限界と加速一次手法を扱う凸最適化の講義録であり、:numref:`sec_mdl-gradient-based-optimization` と :numref:`sec_mdl-convexity` が引用する一般的な証明を収録している。

<!--en:27dd5000-->
- 『[Optimization for Data Analysis](https://www.cambridge.org/core/books/optimization-for-data-analysis/C02C3708905D236AA354D1CE1739A6A2)』（Wright、Recht 著、Cambridge、2022 年）：データ解析向けの勾配法、加速法、確率的手法、および座標降下法による最適化を扱っており、:numref:`sec_mdl-gradient-based-optimization` から :numref:`sec_mdl-adaptive-stochastic-methods` までの副読本となる。

<!--en:59b32a2b-->
- 『[Accuracy and Stability of Numerical Algorithms](https://doi.org/10.1137/1.9780898718027)』（Higham 著、第 2 版、SIAM）：丸め誤差解析、総和計算、ならびに連立一次方程式と最小二乗法の条件性を扱っている（:numref:`sec_mdl-numerical-stability-conditioning`）。

<!--en:83560036-->
- 『[Mathematics for Machine Learning](https://mml-book.github.io/)』（Deisenroth、Faisal、Ong 著）：勾配降下法、制約付き最適化、および凸性に関する自己完結した無料の章を収録している。

<!--en:b3edb95b-->
**講義と講義動画**

<!--en:f47090d9-->
- 「[Stanford EE364a: Convex Optimization I](https://web.stanford.edu/class/ee364a/)」（Boyd 担当）：Boyd、Vandenberghe の凸最適化の教科書に対応するスライド、課題、ソフトウェアであり、[2023 年の講義動画](https://www.youtube.com/playlist?list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h)も公開されている（:numref:`sec_mdl-convexity`、:numref:`sec_mdl-lagrangian-duality`）。

<!--en:f78d1b4d-->
- 「[CMU 10-725: Convex Optimization](https://www.stat.cmu.edu/~ryantibs/convexopt/)」（Tibshirani 担当）：劣勾配、近接手法、確率的手法、双対性、および ADMM に関する機械学習向けの凸最適化スライドと講義録であり、[2016 年秋の講義動画](https://www.youtube.com/playlist?list=PLjbUi5mgii6AVdvImLB9-Hako68p9MpIC)も利用できる（:numref:`sec_mdl-stochastic-gradient-methods`、:numref:`sec_mdl-constrained-optimization-duality`、:numref:`sec_mdl-lagrangian-duality`）。

<!--en:ec1d8b8a-->
**チュートリアル、講義ノート、サーベイ**

<!--en:acdfd421-->
- 「[Why Momentum Really Works](https://distill.pub/2017/momentum/)」（Goh 著、Distill）：凸二次関数におけるモメンタムを対話的な図で解説している（:numref:`subsec_mdl-momentum-acceleration`）。

<!--en:698eab79-->
- 「[Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)」（Bottou、Curtis、Nocedal 著）：大規模学習のための確率的勾配法による最適化理論をまとめている（:numref:`sec_mdl-stochastic-gradient-methods`）。

<!--en:444c6f03-->
- 「[An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/)」（Ruder 著）：モメンタム、ネステロフの加速法、AdaGrad、RMSProp、Adam といった勾配降下法の最適化アルゴリズムを並べて比較している（:numref:`sec_mdl-adaptive-stochastic-methods`）。

<!--en:8f9c5162-->
- 「[What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://doi.org/10.1145/103162.103163)」（Goldberg 著）：浮動小数点演算における IEEE 形式、丸め、および桁落ちを解説している（:numref:`sec_mdl-numerical-stability-conditioning`）。

<!--en:1370a5e9-->
- 「[Optimization (Chapter 5)](https://mlstory.org/optimization.html)」（Hardt、Recht 著『Patterns, Predictions, and Actions』）：教師あり学習における最適化を簡潔に解説している（:numref:`sec_mdl-gradient-based-optimization`、:numref:`sec_mdl-stochastic-gradient-methods`）。

<!--en:cce82a9e-->
[ディスカッション](https://d2l.discourse.group/t/36861)
