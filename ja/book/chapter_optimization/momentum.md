<!--en:69ea020b-->
# モメンタム

<!--en:8604647f-->
:numref:`sec_gd` では、通常の勾配降下法が悪条件（ill-conditioned）の谷で苦戦する理由を見た。最も急峻な曲率 $\lambda_{\max}$ によって安定性の上限 $\eta < 2/\lambda_{\max}$ が課されるため、最も平坦な方向 $\lambda_{\min}$ はステップごとに高々 $O(1/\kappa)$ しか縮小しない（ここで $\kappa = \lambda_{\max}/\lambda_{\min}$ は条件数である）。通常の勾配降下法が遅いのは、各ステップがその時点の勾配 $\mathbf{g}_t$ のみに依存しているからである。しかし、直近の勾配の履歴には明確な幾何学的情報が含まれている。谷底に沿っては連続する勾配が一貫して同じ方向を指すのに対し、急峻な壁を横切る方向では反復点が左右にオーバーシュートするため勾配の符号が交互に反転する。

<!--en:06950f0c-->
:citet:`Polyak.1964` は、過去の勾配の指数重み付き和である**速度**（velocity）に沿ってステップを進める手法を提案した。これにより、平坦な方向に沿う持続的な成分は蓄積され、急峻な方向を横切る振動成分は打ち消し合う。摩擦を受けながら溝を転がり落ちる重い球になぞらえられる**ヘビーボールモメンタム**は、パラメータごとに単一の速度バッファを保持するコストと引き換えに、二次ボウル上での最悪ケースの反復回数を $O(\kappa)$ から $O(\sqrt{\kappa})$ へと削減する。

<!--en:36e5beac-->
本節では、速度の漸化式、その実効ステップサイズ（effective step size）と記憶の時間スケール、および臨界減衰（critical damping）による $\sqrt{\kappa}$ 加速則を解析する。続いて、フルバッチのソフトマックス回帰と `FashionCNN` のミニバッチ訓練でモメンタムを検証し、ネステロフの先読みモメンタムを導入したうえで、$\beta$ と $\eta$ を調整するための実践的な指針を確立する。

<!--en:dfc60fe6-->
## モメンタム法

<!--en:46752dd1-->
### 速度と実効ステップサイズ

<!--en:5b5be529-->
勾配 $\mathbf{g}_t$（ステップ $t$ における厳密勾配またはミニバッチ勾配）が与えられたとき、ヘビーボールモメンタムは速度ベクトル $\mathbf{v}_t$（初期値 $\mathbf{v}_0 = \mathbf{0}$）を保持し、次のように更新する。

<!--en:e4811ab0-->
ここで、$\beta \in [0, 1)$ はモメンタム係数である（$\beta = 0$ とすると勾配降下法に戻る）。この漸化式を展開すると、速度は過去のすべての勾配の指数重み付き和として表される。

<!--en:7f63dedf-->
この重み付き和が、谷 $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$（$\lambda_1 = 0.2, \lambda_2 = 4$）の 2 つの座標に対して $\eta = 0.6$ でどのように働くか考えてみよう。この学習率は通常の勾配降下法の安定限界 $2/\lambda_2 = 0.5$ を超えており、通常の勾配降下法では $x_2$ 方向に沿ってステップごとに $1 - 4(0.6) = -1.4$ 倍となって発散する。ここにモメンタム $\beta = 0.5$ を加えると軌跡が安定化し、両座標とも原点の $0.01$ 以内に $20$ ステップで到達する。

<!--en:ae08f7d7-->
モメンタムが平坦な座標を加速しつつ同時に急峻な座標を安定化できる理由は、次の 2 つの極限的なケースから説明できる。

<!--en:6c72fa63-->
* **持続する勾配（平坦な谷底）：**連続する勾配がほぼ一定（$\mathbf{g}_{t-k} \approx \mathbf{g}$）にとどまるとき、速度は $\mathbf{v}_t \approx (1 + \beta + \beta^2 + \cdots)\mathbf{g} = \frac{1}{1-\beta}\mathbf{g}$ まで蓄積し、次の**実効ステップサイズ**をもたらす。
  $$\eta_\textrm{eff} = \frac{\eta}{1-\beta}.$$

<!--en:cb56143c-->
* **符号が反転する勾配（急峻な壁）：**連続する勾配の符号が交互に反転する（$\mathbf{g}_{t-k} \approx (-1)^k \mathbf{g}$）とき、幾何級数の和は $1 - \beta + \beta^2 - \cdots = \frac{1}{1+\beta}$ となり、実効ステップサイズは $\frac{\eta}{1+\beta}$ へと減衰する。

<!--en:3436c174-->
:numref:`fig_opt_momentum_gain` に、単位サイズの勾配に対する両ケースを示す。

<!--en:4da29d1d-->
![速度 $v_t = \beta v_{t-1} + g_t$（青）の推移（単位勾配 $g_t$ は灰色の棒、$\beta = 0.5$）。左：同符号の勾配が足し合わさり、$v_t$ は $1/(1-\beta) = 2$ に近づく。右：符号が交互に反転する勾配は大半が打ち消し合い、$v_t$ は振幅 $1/(1+\beta) = 2/3$ の振動に落ち着く。](../img/mdl-opt-momentum-gain.svg)

<!--en:e013621c-->
$\eta = 0.6$、$\beta = 0.5$ のとき、モメンタムは谷底に沿った前進を $\eta_{\text{eff}} = 0.6 / 0.5 = 1.2$ へと増幅する一方、急峻な壁を横切る振動を $0.6 / 1.5 = 0.4 < 0.5$ へと減衰させる。その結果、曲率 $\lambda$ に対する安定性の上限は $\eta\lambda < 2$ から $\eta\lambda < 2(1 + \beta)$ へと引き上げられる（演習問題 :exref:`ex_momentum-stability`）。

<!--en:4f710858-->
### $\beta$ の時間スケールとミニバッチノイズ

<!--en:97ad41ef-->
重み $1, \beta, \beta^2, \ldots$ の和は $1/(1-\beta)$ となるため、速度は $\tau = 1/(1-\beta)$ ステップの実効記憶ウィンドウと $\log(1/2)/\log\beta$ ステップの半減期を持つ。

<!--en:cec865dd-->
$\beta$ を $0.9$ から $0.99$ に増やすと、記憶ホライズンと実効ステップサイズ $\eta_{\text{eff}} = \eta/(1-\beta)$ の両方が 10 倍（$10$ ステップから $100$ ステップへ）に拡大する。

<!--en:756f357f-->
確率的勾配 $\mathbf{g}_t = \nabla f(\mathbf{x}_t) + \boldsymbol{\xi}_t$ のもとで、$1/(1-\beta)$ ステップにわたる平均化は :numref:`sec_sgd` の定常ノイズ球（noise ball）を縮小させるだろうか。答えは否である。各ノイズ摂動 $\boldsymbol{\xi}_t$ は $\mathbf{v}_t$ に重み $1$ で加わり、その後のステップで $\beta^k$ として減衰するため、累積変位として $\eta(1 + \beta + \beta^2 + \cdots)\boldsymbol{\xi}_t = \eta_{\text{eff}}\boldsymbol{\xi}_t$ だけ寄与する。したがって、実効ステップサイズ $\eta_{\text{eff}} = \eta/(1-\beta)$ を揃えた場合、モメンタムありの SGD とモメンタムなしの SGD はほぼ同一の分散を持つ定常ノイズ球に落ち着く（演習問題 :exref:`ex_momentum-momentum-under-minibatch-noise`）。ゆえに、ノイズ律速（noise-limited）の訓練で $\beta$ を大きくする際には、$\eta$ を $1 - \beta$ に比例してスケーリングし、$\eta_{\text{eff}}$ を一定に保つべきである。

<!--en:3834eb8f-->
## 加速と臨界減衰

<!--en:7ccb4310-->
組 $(\eta, \beta)$ としてどのような選択が、固有値 $\lambda_{\min} \le \cdots \le \lambda_{\max}$ と条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ を持つ二次形式において最適となるだろうか。:citet:`Polyak.1964` は、次のように設定すればよいことを証明した。

<!--en:e5dcb3f6-->
このとき、最悪ケースにおけるステップごとの縮小率（contraction factor）は次の値に達する。

<!--en:fa1b1dbe-->
これにより、相対精度 $\epsilon$ に到達するために必要なステップ数は $\frac{\kappa}{2}\log(1/\epsilon)$ から $\frac{\sqrt{\kappa}}{2}\log(1/\epsilon)$ へと削減される（:numref:`subsec_mdl-momentum-acceleration`）。

<!--en:997bd433-->
縮小率が $\sqrt{\beta^\star}$ に等しくなる理由を見るため、$\eta v_t = x_t - x_{t+1}$ を曲率 $\lambda$ の固有ベクトルに沿った 1 次元の速度更新式に代入すると、次の 2 階線形漸化式が得られる。

<!--en:c97db829-->
この漸化式の特性多項式は $z^2 - (1 + \beta - \eta\lambda)z + \beta = 0$ である。解と係数の関係により、2 つの根の積は $z_1 z_2 = \beta$ となる。判別式 $\Delta = (1 + \beta - \eta\lambda)^2 - 4\beta$ が非正であるとき（これはすべての $\beta \ge \beta_{\text{c}}(\lambda) = (1 - \sqrt{\eta\lambda})^2$ で生じる）、根は同じ絶対値 $|z_1| = |z_2| = \sqrt{\beta}$ を持つ複素共役対をなし、その大きさは曲率 $\lambda$ に依存しない（:numref:`fig_opt_critical_damping`）。

<!--en:4b4a7fe1-->
![曲率 $\lambda$ に沿ったヘビーボールモメンタムのステップごとの縮小率（$\beta$ の関数）。収束は臨界減衰のしきい値 $\beta_\textrm{c} = (1 - \sqrt{\eta\lambda})^2$ で最速となり、$\beta_\textrm{c}$ 未満では過減衰、$\beta_\textrm{c}$ を超えると絶対値 $\sqrt{\beta}$ で振動する。](../img/mdl-opt-critical-damping.svg)

<!--en:952b3f5a-->
:eqref:`eq_momentum-optimal` の最適パラメータは、最も平坦な固有値と最も急峻な固有値の臨界減衰しきい値を $(1 - \sqrt{\eta^\star\lambda_{\min}})^2 = (\sqrt{\eta^\star\lambda_{\max}} - 1)^2 = \beta^\star$ として等しくする。これにより両端のモードを臨界減衰させるとともに、すべての中間固有値 $\lambda \in [\lambda_{\min}, \lambda_{\max}]$ が不足減衰（underdamped）の複素根領域に入り、縮小率 $\sqrt{\beta^\star}$ で縮小することを保証する。

<!--en:204174d9-->
先ほどの谷（$\kappa = 20$）では、:eqref:`eq_momentum-optimal` から $\eta^\star \approx 0.668, \beta^\star \approx 0.403$ が得られる。これにより漸近縮小率は $0.905$ から $\sqrt{\beta^\star} \approx 0.635$ に低下し、相対誤差 $10^{-6}$ に達するまでのステップ数は $139$ から $40$ へと減少する。

<!--en:9245e22c-->
4 行目（$\eta = 0.3, \beta = 0.8$）に注目しよう。$\beta$ を臨界しきい値 $\beta_{\text{c}}$ より大幅に大きく選ぶと、両座標とも絶対値 $\sqrt{0.8} \approx 0.894$ の不足減衰となり、軌跡はオーバーシュートして原点周りで減衰振動（リンギング）を起こす。

<!--en:0981e9c1-->
## データ上でのモメンタム

<!--en:2c4af4d7-->
これまでの解析から、データ上で訓練する際には次の 2 つの異なるレジームが予測される。

<!--en:72ed1371-->
1. **曲率律速レジーム（高精度な勾配、大きな $\kappa$）：**モメンタムによってより大きな実効ステップサイズ $\eta_{\text{eff}}$ が可能になり、必要な反復回数が大幅に削減される。

<!--en:06afa5d5-->
2. **ノイズ律速レジーム（小さなミニバッチ）：**定常ノイズフロアは $\eta_{\text{eff}} = \eta/(1-\beta)$ に支配されるため、$\beta = 0.9$ を加えると最適学習率は $1 - \beta = 0.1$ 倍に下がる一方、収束速度は同程度にとどまる。

<!--en:d73ceef7-->
Fashion-MNIST 上で $\beta = 0$ と $\beta = 0.9$ を比較し、両レジームを検証しよう。対象は、（1）フルバッチのソフトマックス回帰（$b = 60{,}000$、厳密な勾配、非負のピクセル強度により悪条件）と、（2）`FashionCNN`（:numref:`subsec_fashioncnn`）の小ミニバッチ訓練（$b = 32$）である。まず、正規化した Fashion-MNIST テンソルをメモリ内ローダ `Batches` としてアクセラレータ上に事前読み込みする。

<!--en:62d3802f-->
次に、ゼロ初期化した線形モデル `softmax_regression` と、モメンタムを引数にとる SGD コンストラクタ `make_sgd(beta)` を定義する。

<!--en:e63032a0-->
ここで `d2l.lr_sweep` を用いて `grids` にわたる学習率スイープを両モデルと $\beta \in \{0.0, 0.9\}$ に対して実行し、各設定における最良の学習率、目標損失への到達ステップ数、および最終損失を記録する。

<!--en:6f291a4e-->
いずれの予測も定量的に成り立つことがわかる。

<!--en:a42eefb1-->
* フルバッチの**ソフトマックス回帰**では、モメンタム $\beta = 0.9$ を加えることで最適学習率が $\eta = 0.125$ から $\eta = 0.4$ に上昇し（$\eta_{\text{eff}} = 4.0$ と実効ステップサイズが $32\times$ に拡大）、損失 $0.5$ に到達するまでのステップ数は $6\times$ 以上削減され（$625 \to 100$ ステップ）、最終損失も $0.472$ から $0.365$ へと低下する。

<!--en:25d3708d-->
* 小バッチの **`FashionCNN`**（$b = 32$）では、訓練はノイズ律速となる。最適学習率は $\eta \approx 0.1$〜$0.2$（$\beta = 0$ 時）から $\eta \approx 0.02$〜$0.04$（$\beta = 0.9$ 時）へと低下し（比率にして $5$〜$10$ 倍の変化であり、実効ステップサイズ $\eta_{\text{eff}}$ は $\beta = 0$ での値から公比 2 のグリッド上で 1 段階以内に保たれる）、いずれの設定でも損失 $0.3$ に到達し、最終損失も $0.26$〜$0.30$ と :numref:`sec_test_models` の単一シード変動の範囲内に収まる。

<!--en:b493bcd5-->
## ネステロフのモメンタム

<!--en:768cf354-->
不足減衰の実行例（$\eta = 0.3, \beta = 0.8$）においてヘビーボールモメンタムが最小点をオーバーシュートしたのは、速度によってまさに到達しようとしている地点ではなく、現在位置 $\mathbf{x}_t$ で勾配を評価していたためである。:citet:`Nesterov.1983` は、**先読み点**（look-ahead point）$\mathbf{x}_t - \eta\beta\mathbf{v}_{t-1}$ で勾配を評価する手法を提案した（:numref:`fig_opt_nesterov_lookahead`）。

<!--en:65de7c87-->
![同じ反復点と速度からのヘビーボールモメンタムとネステロフのモメンタムの 1 ステップの比較。ヘビーボールモメンタムは $\mathbf{x}_t$ で勾配を評価してオーバーシュートするが、ネステロフのモメンタムは先読み点 $\mathbf{x}_t - \eta\beta\mathbf{v}_{t-1}$ で勾配を評価し、即座に引き戻す補正を加える。](../img/mdl-opt-nesterov-lookahead.svg)

<!--en:814d3202-->
先読み点で勾配を評価すると、急峻な方向の振動を抑える先回り減衰（anticipatory damping）が導入される。さらに、理論的保証もヘビーボールモメンタムより広い。定数 $\beta = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ を用いれば、ネステロフのモメンタムは二次関数だけでなく、あらゆる滑らかな強凸関数で $\sqrt{\kappa}$ のレートを達成する。また、増加スケジュール $\beta_t = (t-1)/(t+2)$ を用いれば、一般の滑らかな凸目的関数に対して最適な $O(1/t^2)$ の収束率を達成する :cite:`Nesterov.2018`（:numref:`subsec_mdl-momentum-acceleration`）。ここでは定数の $\beta$ を用いる。

<!--en:db0ef59f-->
実務上は、ほとんどのアーキテクチャにおいて $\beta = 0.9$ が頑健なデフォルト値となり、学習率は実効ステップサイズ $\eta_{\text{eff}} = \eta/(1-\beta)$ を通じて調整される。しかし、モメンタムは依然としてすべての座標に単一のスカラーのステップサイズを適用する。Transformer のように層によって勾配のスケールが何桁も異なる場合には、座標ごとの適応的なスケーリングが必要となる（:numref:`sec_adam`）。

<!--en:30ac03ff-->
## まとめ

<!--en:c05f895f-->
ヘビーボールモメンタムは、指数減衰する速度 $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \mathbf{g}_t$（記憶ホライズン $1/(1-\beta)$）に沿ってパラメータを更新し、持続的な勾配方向を実効ステップサイズ $\eta_{\text{eff}} = \eta/(1-\beta)$ へと増幅する一方、符号が交互に反転する方向を $\eta/(1+\beta)$ へと減衰させ、安定性の上限を $\eta\lambda < 2(1+\beta)$ に引き上げる。条件数 $\kappa$ の二次ボウル上では、$(\eta^\star, \beta^\star)$ によって両端の固有値を臨界減衰させることで、反復計算量は $O(\kappa)$ から $O(\sqrt{\kappa})$ へと低下する。ネステロフのモメンタムは先読み点 $\mathbf{x}_t - \eta\beta\mathbf{v}_{t-1}$ で勾配を評価することで先回り減衰をもたらし、不足減衰によるリンギングを抑えるとともに、加速効果を一般の滑らかな凸関数へと拡張する。ミニバッチノイズのもとでは、実効ステップサイズ $\eta_{\text{eff}} = \eta/(1-\beta)$ を揃えれば定常ノイズ球が保たれるため、モメンタムは曲率律速（curvature-limited）レジームで大幅な高速化をもたらす一方、ノイズ律速レジームでは $\eta \propto 1 - \beta$ のスケーリングを必要とする。

<!--en:2294243a-->
## 演習問題

<!--en:7c83754d-->
1. [code] **モメンタムと学習率のグリッド。**`momentum_2d` を `f_valley` 上で $\beta \in \{0, 0.25, 0.5, 0.8\}$ および $\eta \in \{0.1, 0.3, 0.6, 1.0\}$ にわたり最大 $1{,}000$ ステップ実行せよ。$\|\mathbf{x}_t\| \le 10^{-3}$ に到達するまでのステップ数（または発散）を表にまとめ、各 $\eta$ における経験的な最適値を $\beta^\star$（:eqref:`eq_momentum-optimal`）および $\beta_{\text{c}}(\lambda) = (1 - \sqrt{\eta\lambda})^2$ と比較せよ。
   :exlabel:`ex_momentum-momentum-learning-rate-together`

<!--en:41e3027e-->
1. **速度のノイズ対信号比。**$\mathbf{g}_t = \mathbf{g} + \boldsymbol{\xi}_t$ において、座標ごとに独立同分布で平均ゼロ・分散 $\sigma^2$ のノイズが加わるとき、$\mathbf{v}_t$（:eqref:`eq_momentum`）の定常平均が $\mathbf{g}/(1-\beta)$、定常分散が $\sigma^2/(1-\beta^2)$ となり、速度のノイズ対信号比が $\sqrt{(1-\beta)/(1+\beta)}$ 倍に縮小することを示せ。
   :exlabel:`ex_momentum-velocity-noise`

<!--en:1e1a713e-->
1. **ネステロフのモメンタムの同値な形式と安定性。**
    1. ライブラリで用いられる形式 $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \nabla f(\mathbf{y}_t)$、$\mathbf{y}_{t+1} = \mathbf{y}_t - \eta(\nabla f(\mathbf{y}_t) + \beta\mathbf{v}_t)$ が、:eqref:`eq_nesterov` と $\mathbf{y}_t = \mathbf{x}_t - \eta\beta\mathbf{v}_{t-1}$ のもとでまったく同じ速度を生成することを数学的帰納法により示せ。
       :exlabel:`ex_momentum-two-forms-nesterov-momentum`
    1. :eqref:`eq_momentum-recurrence` から、$z^2 - (1 + \beta - \eta\lambda)z + \beta = 0$ の両根が $0 \le \beta < 1$ において単位円の真に内部に存在するのは、$0 < \eta\lambda < 2(1+\beta)$ のとき、かつそのときに限ることを証明せよ。
       :exlabel:`ex_momentum-stability`

<!--en:9dc3878e-->
1. **指数平均形式のモメンタム。**:eqref:`eq_momentum` を、正規化された EMA 形式 $\mathbf{m}_t = \beta\mathbf{m}_{t-1} + (1-\beta)\mathbf{g}_t$、$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha\mathbf{m}_t$（Adam で使用、:numref:`sec_adam`）と比較せよ。$\mathbf{m}_t = (1-\beta)\mathbf{v}_t$ となり、したがって $\alpha = \eta/(1-\beta) = \eta_{\text{eff}}$ のとき両者の軌跡が一致することを示せ。
   :exlabel:`ex_momentum-averaged-form`

<!--en:7487ec2a-->
1. [code] **バッチサイズにわたるミニバッチノイズ下のモメンタム。** ●
    1. $f(x) = \frac{\lambda}{2}x^2$ 上でノイズ付き勾配 $g_t = \lambda x_t + \xi_t$ を用いるとき、ステップサイズ $\alpha$ の通常の SGD と、モメンタム（$\beta = 0.9$）で $\eta = (1-\beta)\alpha$ とした場合の反復点の定常分散を $\alpha\lambda \in \{0.05, 0.2\}$ について比較せよ。
    1. `d2l.FashionCNN` を $1{,}000$ ステップにわたり、バッチサイズ $b \in \{32, 500, 6000\}$、$\beta \in \{0, 0.9\}$、および揃えた実効ステップサイズ $\eta/(1-\beta) = 0.2$ で訓練せよ。続いて $\eta$ を $b = 6000$ においてスイープし、バッチサイズが曲率律速レジームに入るほど大きくなるとモメンタムが訓練を加速することを検証せよ。
   :exlabel:`ex_momentum-momentum-under-minibatch-noise`

<!--en:e6b11b68-->
[ディスカッション](https://d2l.discourse.group/t/1070)

<!--en:3183d2d1-->
[ディスカッション](https://d2l.discourse.group/t/1071)

<!--en:93649e9b-->
モメンタムとネステロフの加速勾配法<br>**速度の漸化式・実効ステップサイズ・記憶の時間スケール・臨界減衰と √κ 加速・ネステロフの先読み**

<!--en:67a6068e-->
::: {.slide title="記憶を持たない勾配降下法が谷で苦戦する理由"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:f8aa5735-->
- **記憶なしによる $O(\kappa)$ のボトルネック：**$f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$（$\kappa = 20$）では、通常の GD は時間方向の勾配アライメントを無視するため、$47$ ステップかけて誤差 $1\%$ に達する（$139$ ステップで $10^{-6}$）
  - **平坦な谷底に沿う方向（$x_1$）：**勾配が**持続的に同じ方向**を指す
  - **急峻な壁を横切る方向（$x_2$）：**勾配がステップごとに**符号を交互に反転**させる

<!--en:f6ba3323-->
- **Polyak のヘビーボールの着想（1964）：**指数減衰する速度の和を蓄積し、谷底の勾配を強め合わせ、壁方向の振動を打ち消す

<!--en:cbe9da60-->
::: {.slide title="ヘビーボールモメンタム：速度の漸化式"}

<!--en:90578f94-->
[更新式]{.kicker}

<!--en:6eb13c6c-->
- **速度の更新（$\beta \in [0, 1)$、$\mathbf{v}_0 = \mathbf{0}$）：**
  $$\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + \mathbf{g}_t, \qquad \mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\mathbf{v}_t$$

<!--en:e57229d9-->
- **展開した幾何級数の履歴：**
  $$\mathbf{v}_t = \sum_{k=0}^{t-1} \beta^k \mathbf{g}_{t-k}$$

<!--en:6b6f9187-->
- **物理的なアナロジー：**$f(\mathbf{x})$ を慣性 $\beta$ と粘性摩擦 $1 - \beta$ を伴って転がり落ちる重い球（記憶ウィンドウ $\tau = \frac{1}{1-\beta}$）

<!--en:c0c59793-->
::: {.slide title="発散する学習率（η = 0.6）のモメンタムによる安定化"}

<!--en:06892055-->
[谷における軌跡]{.kicker}

<!--en:2751cbb9-->
- $\eta = 0.6 > 2/\lambda_2 = 0.5$ では、通常の勾配降下法はステップごとに $x_2$ が $1 - 4(0.6) = -1.4$ 倍となるため、急峻な壁を横切って激しく発散する（$1.4^{20} \approx 837$）

<!--en:c4df96af-->
- **$\beta = 0.5$** のヘビーボールモメンタムを*まったく同じ* $\eta = 0.6$ で加えると、壁を横切る振動が減衰し、両座標が原点の $1\%$ 以内に **$20$ ステップ**で到達する

<!--en:28f1efb2-->
- **一見した逆説：**過去の勾配の和をとると谷底方向の総ステップ長は増大するのに、なぜ同時に急峻な壁を*安定化*できるのか。

<!--en:7f9bd9ad-->
::: {.slide title="持続方向の増幅と交代方向の相殺"}

<!--en:40756f41-->
[実効ステップサイズ]{.kicker}

<!--en:3e80f0a4-->
- **持続する谷底モード（$\mathbf{g}_{t-k} \approx \mathbf{g}$）：**幾何級数の和 $\sum_{k \ge 0}\beta^k = \frac{1}{1-\beta}$ により、実効ステップサイズは **$\eta_{\text{eff}} = \frac{\eta}{1-\beta}$** となる（$1.2$、$\eta=0.6, \beta=0.5$ の場合）

<!--en:69850254-->
- **交代する壁モード（$\mathbf{g}_{t-k} \approx (-1)^k\mathbf{g}$）：**交代級数の和 $\sum_{k \ge 0}(-\beta)^k = \frac{1}{1+\beta}$ により、ステップは **$\eta_{\text{alt}} = \frac{\eta}{1+\beta}$** に減衰する（$0.4 < 0.5$ で安定）

<!--en:fe9c9156-->
- **拡大された安定性の上限：**$\eta\lambda_{\max} < 2(1+\beta)$ となり、通常の GD より $\frac{1+\beta}{1-\beta}$ 倍大きい谷底ステップが可能になる

<!--en:5edae217-->
::: {.slide title="モメンタムによる安定性の上限の拡大"}

<!--en:d67b5088-->
[安定性と最大高速化率]{.kicker}

<!--en:b5beaac6-->
符号が反転するモード $x_t \propto (-1)^t$ は減衰したステップサイズ $\frac{\eta}{1+\beta}$ を受けるため、安定性のしきい値 $\frac{\eta}{1+\beta}\lambda < 2$ は $1 + \beta$ 倍に引き上げられる。

<!--en:e9fabbcd-->
| オプティマイザ | 符号反転時のステップサイズ | 曲率 $\lambda$ の安定性の上限 | 谷での上限（$\lambda_{\max} = 4$） | 最大安定谷底ステップ $\eta_{\text{eff}}$ |

<!--en:5e33461b-->
| **通常の勾配降下法（$\beta = 0$）** | $\eta$ | $\eta\lambda < 2$ | $\eta < 0.50$ | $0.50$ |

<!--en:a3595806-->
| **ヘビーボールモメンタム（$\beta$）** | $\frac{\eta}{1 + \beta}$ | **$\eta\lambda < 2(1 + \beta)$** | $\eta < 0.75$（$\beta = 0.5$ 時） | **$\frac{2(1+\beta)}{(1-\beta)\lambda_{\max}} = 1.50$**（$\beta = 0.5$ 時） |

<!--en:38ec5240-->
- $\beta = 0.5$ のとき、モメンタムは谷底に沿って通常の勾配降下法の最大安定ステップより最大 $\frac{1+\beta}{1-\beta} = 3\times$ 大きい実効ステップをとれる

<!--en:eb90c4db-->
::: {.slide title="β の記憶時間スケールと半減期"}

<!--en:69562191-->
[指数ホライズン]{.kicker}

<!--en:63e6d8af-->
- 幾何級数の重み $1, \beta, \beta^2, \ldots$ の和は $\frac{1}{1-\beta}$ となるため、速度は特徴的な**記憶ウィンドウ**にわたって勾配を平均化する
  $$\tau = \frac{1}{1 - \beta} \text{ steps}$$

<!--en:a0ba2b60-->
- **半減期：**勾配の重み $\beta^k$ は $0.5$ まで $k_{1/2} = \frac{\ln 0.5}{\ln\beta} \approx 0.693\,\tau$ ステップで減衰し、最初の $\tau$ ステップが常に重み全体の $1 - \beta^{1/(1-\beta)} \approx 1 - e^{-1} \approx 63\text{--}75\%$ を占める

<!--en:1aa8dffe-->
- $\beta$ を $0.9 \to 0.99$ に変えると、$\tau$ と $\eta_{\text{eff}}$ の両方が **10 倍**（$10 \to 100$ ステップ）に拡大する

<!--en:02b8cece-->
::: {.slide title="ミニバッチノイズ下のモメンタム：変わらないノイズフロア"}

<!--en:9ba355d1-->
[確率的ノイズとの相互作用]{.kicker}

<!--en:600f704d-->
- **単一のノイズ摂動 $\boldsymbol{\xi}_t$ の積分インパルス：**$\mathbf{v}_{t+k}$ 内に重み $\beta^k$ で残り、合計で次の累積変位をもたらす
  $$-\eta\,(1 + \beta + \beta^2 + \cdots)\,\boldsymbol{\xi}_t = -\frac{\eta}{1 - \beta}\,\boldsymbol{\xi}_t = -\eta_{\text{eff}}\,\boldsymbol{\xi}_t$$

<!--en:bd79a7bc-->
- **同じ $\eta_{\text{eff}}$ ではノイズ球は縮小しない：**モメンタムは各ノイズ摂動を $\tau = \frac{1}{1-\beta}$ ステップに分散させるだけで、ステップサイズ $\eta_{\text{eff}}$ の通常の SGD と同じ定常分散に落ち着く

<!--en:4b82e2b8-->
- **調整則：**ノイズ律速の訓練で $\beta = 0.9$ を有効にする際は、$\eta$ を $1 - \beta = 0.1$ 倍に縮小する

<!--en:f346caa3-->
::: {.slide title="二次形式の最適加速：O(κ) から O(√κ) へ"}

<!--en:b8b673ae-->
[Polyak の加速則]{.kicker}

<!--en:e23f921f-->
- **Polyak の最適調整（1964）：**
  $$\eta^\star = \frac{4}{\left(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}}\right)^2}, \qquad \beta^\star = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2$$

<!--en:f4d030ad-->
- **$O(\kappa)$ から $O(\sqrt{\kappa})$ ステップへ：**縮小率が $\frac{\kappa-1}{\kappa+1} \approx 1 - \frac{2}{\kappa}$（通常の GD）から **$\sqrt{\beta^\star} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \approx 1 - \frac{2}{\sqrt{\kappa}}$** に改善（ステップ数が $37\times$ 削減、$\kappa = 1{,}400$ の場合）

<!--en:6a7365e2-->
- **なぜ $\sqrt{\beta}$ なのか：**$\beta \ge \beta_{\text{c}}(\lambda) = (1-\sqrt{\eta\lambda})^2$ では、特性根 $z_1, z_2$ は複素共役となり、$z_1 z_2 = \beta \implies |z| = \sqrt{\beta}$ が成り立つ

<!--en:235f7ca8-->
::: {.slide title="2 階漸化式と √β 複素根の法則"}

<!--en:8a09bacf-->
[縮小率が √β になる理由]{.kicker}

<!--en:b5780800-->
なぜ*すべての*中間固有モードは、まったく同じ縮小率 $\sqrt{\beta^\star}$ で曲率 $\lambda$ に依らず縮小するのか。

<!--en:72a44218-->
- 曲率 $\lambda$ の固有ベクトルに沿う方向（$g_t = \lambda x_t$）で、$\eta v_{t-1} = x_{t-1} - x_t$ を $x_{t+1} = x_t - \eta(\beta v_{t-1} + \lambda x_t)$ に代入すると、次の 2 階線形漸化式が得られる
  $$x_{t+1} - (1 + \beta - \eta\lambda)\,x_t + \beta\,x_{t-1} = 0$$

<!--en:ca33077d-->
- **特性多項式：**$x_t = z^t$ とおくと $z^2 - (1 + \beta - \eta\lambda)z + \beta = 0$ が得られ、2 つの根は解と係数の関係 **$z_1 z_2 = \beta$** を満たす

<!--en:28a82da9-->
- **複素共役領域：**判別式 $\Delta = (1 + \beta - \eta\lambda)^2 - 4\beta \le 0$（すべての $\beta \ge \beta_{\text{c}}(\lambda) = (1 - \sqrt{\eta\lambda})^2$ で成立）のとき、根 $z_1, z_2 = z_1^*$ は同じ絶対値を持つ複素共役対となる
  $$|z_1| = |z_2| = \sqrt{z_1 z_2} = \sqrt{\beta}$$

<!--en:e212d813-->
::: {.slide title="過減衰・臨界減衰・不足減衰モード"}

<!--en:688a5219-->
[臨界減衰の幾何学]{.kicker}

<!--en:17ef1ef4-->
- **曲率 $\lambda$ のモードにおける 3 つの減衰レジーム：**
  - **過減衰（$\beta < \beta_{\text{c}}$）：**異なる 2 つの実根を持ち、大きい方の根が振動せずゆっくりゼロへ近づく
  - **臨界減衰（$\beta = \beta_{\text{c}} = (1 - \sqrt{\eta\lambda})^2$）：**根が $z_1 = z_2 = \sqrt{\beta_{\text{c}}}$ で重なり、そのモードで最速の縮小を達成する
  - **不足減衰（$\beta > \beta_{\text{c}}$）：**複素根が振動しながら包絡線の縮小率 $\sqrt{\beta}$ で減衰する

<!--en:b6822252-->
- **Polyak の最適調整 $(\eta^\star, \beta^\star)$** は $\lambda_{\min}$ と $\lambda_{\max}$ の臨界しきい値を一致させる：$(1 - \sqrt{\eta^\star\lambda_{\min}})^2 = (\sqrt{\eta^\star\lambda_{\max}} - 1)^2 = \beta^\star$

<!--en:6cf99f82-->
::: {.slide title="κ = 20 の谷における定量的検証"}

<!--en:adf60e20-->
[40 ステップ対 139 ステップ]{.kicker}

<!--en:64d83061-->
- 先ほどの谷（$\lambda_1 = 0.2, \lambda_2 = 4.0, \kappa = 20$）の場合：
  - 最適調整により $\eta^\star \approx 0.668$、$\beta^\star \approx 0.403$ となる
  - ステップごとの縮小率は $0.905$（通常の GD）から $\sqrt{\beta^\star} \approx 0.635$ に低下する
  - 相対誤差 $10^{-6}$ までのステップ数が **$3.5\times$** 減少し、**$139$ ステップから $40$ ステップ**に短縮される

<!--en:72f9cc3a-->
::: {.slide title="β が臨界減衰を超えたときの不足減衰リンギング"}

<!--en:6c368af7-->
[オーバーシュートの危険性]{.kicker}

<!--en:287e88b7-->
- $\beta = 0.8$、$\eta = 0.3$ と設定し、両座標の臨界減衰しきい値（$\beta_{\text{c}}(\lambda_1) = 0.57, \beta_{\text{c}}(\lambda_2) = 0.01$）を大幅に超えるとどうなるか。

<!--en:aec81681-->
- 両座標とも絶対値 $\sqrt{0.8} \approx 0.894$ の深い不足減衰複素根領域に入る（通常の GD の $0.905$ よりわずかに速い程度）

<!--en:09ac7d43-->
- 速度が下り方向への推進力を $\tau = \frac{1}{1-0.8} = 5$ ステップ分記憶するため、反復点は原点をオーバーシュートし、両軸に沿って往復振動（リンギング）を起こす

<!--en:8ff6b868-->
::: {.slide title="実データにおける 2 つのレジーム：曲率とノイズ"}

<!--en:eb9c79b3-->
[Fashion-MNIST での実証実験]{.kicker}

<!--en:5c602bb3-->
理論からは、$\beta = 0$ と $\beta = 0.9$ を Fashion-MNIST 上で比較した際に次の 2 つの対照的な結果が予測される。

<!--en:7e278928-->
- **1. 曲率律速（フルバッチのソフトマックス回帰、$b = 60{,}000$）：**厳密な勾配と悪条件のピクセル共分散。$\beta = 0.9$ を加えると最適な実効ステップサイズが $32\times$ に拡大し（$\eta_{\text{eff}}: 0.125 \to 4.0$）、損失 $0.5$ までのステップ数が **$6.25\times$** 削減される（$625 \to 100$ ステップ）

<!--en:19986031-->
- **2. ノイズ律速（`FashionCNN`、小バッチ $b = 32$）：**最適な $\eta$ は $\sim 10\times$ 低下して（$0.2 \to 0.02$）$\eta_{\text{eff}} = \frac{\eta}{1-\beta}$ を保ち、同程度のステップ数と損失に落ち着く

<!--en:ead647db-->
::: {.slide title="ネステロフの加速勾配法：先読み補正"}

<!--en:f3f0260d-->
[先回り減衰]{.kicker}

<!--en:8081036d-->
- なぜ不足減衰のヘビーボールモメンタムはオーバーシュートしたのか。$\nabla f(\mathbf{x}_t)$ を、モメンタムステップ $-\eta\beta\mathbf{v}_{t-1}$ が運ぼうとしている先ではなく現在位置で評価していたためである

<!--en:8ada0986-->
- **ネステロフのモメンタム（1983）**は**先読み点** $\tilde{\mathbf{x}}_t = \mathbf{x}_t - \eta\beta\,\mathbf{v}_{t-1}$ で勾配を評価する
  $$\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + \nabla f(\mathbf{x}_t - \eta\beta\,\mathbf{v}_{t-1})$$

<!--en:3036a06c-->
- 先読みによる曲率評価はオーバーシュートが起きる*前に*速度にブレーキをかけ、スケジュール $\beta_t = (t-1)/(t+2)$ のもとで一般の滑らかな凸関数に対して最適な $O(1/t^2)$ の収束率を達成する

<!--en:3af6bc90-->
::: {.slide title="ネステロフの先読みによる不足減衰リンギングの抑制"}

<!--en:3f8bb317-->
[ヘビーボール対ネステロフ]{.kicker}

<!--en:d454abae-->
- 谷においてまったく同じ不足減衰のハイパーパラメータ（$\eta = 0.3, \beta = 0.8$）でネステロフのモメンタムを実行すると、即座に先回り減衰が働く：
  - 急峻な壁方向の $x_2$ の振動は $|x_2| < 10^{-4}$ 未満へ $20$ ステップ以内に減衰（ヘビーボールでは $-0.25$）
  - 谷底に沿う $x_1$ 方向のオーバーシュートは約半分に縮小

<!--en:48251e3d-->
- 実務上は $\beta = 0.9$（ヘビーボールまたはネステロフの先読み）が標準のデフォルトであり、$\eta_{\text{eff}} = \frac{\eta}{1-\beta}$ を通じて調整する

<!--en:60ba1718-->
::: {.slide title="概念チェック：実効ステップ・時間スケール・√κ"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:5cce3512-->
各モメンタムのシナリオについて定量的な答えを求めよ。

<!--en:44281f24-->
1. **$\eta$ の調整（$\beta$ 変更時）：**ミニバッチ SGD が $\eta = 0.02$、$\beta = 0.9$ で良好に訓練できているとする。モメンタムを $\beta = 0.98$ に増やした場合、どの学習率 $\eta$ が実効ステップサイズ $\eta_{\text{eff}}$ と定常ノイズ球を保つか。また、記憶ウィンドウ $\tau$ はどう変化するか。

<!--en:0e4a1be0-->
2. **$\sqrt{\kappa}$ 加速：**条件数 $\kappa = 10{,}000$ の二次形式において最適なモメンタム $\beta^\star$ はいくらか。また、最適なモメンタムは誤差を $10^{-2}$ 倍に縮小するまでのステップ数を、最適な通常の勾配降下法と比べて何倍削減するか。

<!--en:9b831269-->
3. **バッチサイズによる遷移：**$\beta = 0.9$ を加えたとき、なぜ $6\times$ ものステップ数削減がフルバッチのソフトマックス回帰（$b = 60{,}000$）では生じ、`FashionCNN`（$b = 32$）では生じなかったのか。

<!--en:a5a47016-->
::: {.slide title="概念チェック：実効ステップ・時間スケール・√κ（解答）"}

<!--en:929b7671-2-->
[理解度チェック]{.kicker}

<!--en:557b98b6-->
**解答：**

<!--en:50b29342-->
1. $\beta = 0.9$ のとき、$\eta_{\text{eff}} = \frac{0.02}{1 - 0.9} = 0.2$、$\tau = 10$ ステップである。$\beta = 0.98$（$1 - \beta = 0.02$）で $\eta_{\text{eff}} = 0.2$ を保つには **$\eta = 0.2 \times 0.02 = 0.004$**（$5\times$ 縮小）が必要となり、記憶ウィンドウは **$5\times$（$10 \to 50$ ステップ）に拡大**する。

<!--en:b4b6c3b9-->
2. $\sqrt{\kappa} = 100$ より **$\beta^\star = (99/101)^2 \approx 0.9608$** となり、必要ステップ数は **$\frac{\kappa}{\sqrt{\kappa}} = \sqrt{\kappa} = 100\times$** 削減される（$\approx 230$ ステップ対 $23{,}000$ ステップ）。

<!--en:0dedd414-->
3. $b = 32$ では訓練が**ノイズ律速**（$b \ll b_{\text{crit}}$）であるため、$\eta_{\text{eff}}$ はノイズ球によって頭打ちになる。一方、フルバッチ（$b = 60{,}000$）では勾配ノイズがゼロで訓練が**曲率律速**となるため、モメンタムにより $32\times$ 大きい $\eta_{\text{eff}}$ が可能になる。

<!--en:5126529b-->
::: {.slide title="まとめ：ヘビーボールとネステロフのモメンタム"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:0382e49d-->
| 概念・規則 | 数式 | 主な性質 | 実践的な指針 |

<!--en:90440225-->
| **ヘビーボールの速度** | $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \mathbf{g}_t,\; \mathbf{x}_{t+1} = \mathbf{x}_t - \eta\mathbf{v}_t$ | $\sum_{k=0}^{t-1} \beta^k \mathbf{g}_{t-k}$ に展開される | デフォルトは $\beta = 0.9$、パラメータごとに状態バッファ 1 個を追加 |

<!--en:71cf2ec9-->
| **実効ステップとウィンドウ** | $\eta_{\text{eff}} = \frac{\eta}{1-\beta},\quad \tau = \frac{1}{1-\beta}$ | 符号反転ステップを $\frac{\eta}{1+\beta}$ に減衰、上限 $\eta\lambda < 2(1+\beta)$ | ノイズ下では $\eta \propto (1-\beta)$ にスケーリング（$\beta$ 変更時） |

<!--en:ce449060-->
| **臨界減衰と $\sqrt{\kappa}$** | $\beta_{\text{c}}(\lambda) = (1 - \sqrt{\eta\lambda})^2,\; \rho^\star = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ | $\beta \ge \beta_{\text{c}}$ の複素根は絶対値 $|z| = \sqrt{\beta}$ を持つ | 曲率律速でのステップ計算量を $O(\kappa)$ から $O(\sqrt{\kappa})$ に削減 |

<!--en:fc134325-->
| **ネステロフの先読み** | $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \nabla f(\mathbf{x}_t - \eta\beta\mathbf{v}_{t-1})$ | 予測位置 $\tilde{\mathbf{x}}_t$ で曲率を評価 | 不足減衰リンギングを抑制、Muon（§{sec_muon}）の基礎となる |
