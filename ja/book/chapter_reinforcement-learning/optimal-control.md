<!--en:e68f4615-->
# 最適制御、LQR、モデル予測制御

<!--en:25f0563b-->
動的計画法（:numref:`sec_valueiter`）は、16 個の状態価値からなる表を反復走査し、すべてのマスでベルマンの最適性方程式が成り立つまで更新することで滑りやすい湖の問題を解いた。それ以降に扱った Q 学習から行動クローニングに至るすべての手法も、状態ごとに表の 1 行を保持してきた。では、:numref:`sec_mdp` の車両の例のように状態が位置・角度・速度からなる連続ベクトル $\mathbf{x}_t \in \mathbb{R}^d$ であり、行動が連続的な制御入力 $\mathbf{u}_t \in \mathbb{R}^m$ として既知のダイナミクス $\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$ を通じて作用する場合はどうなるだろうか。グリッドはスケールしない。$d$ 個の座標それぞれを $100$ 個のビンに分割すると $100^d$ 個のセルが生じるが、これこそ Bellman の*次元の呪い*（curse of dimensionality）である :cite:`BellmanDPBook`。

<!--en:2d9364b2-->
制御理論（control theory）は、深層強化学習が登場するはるか以前に、:citet:`Kalman.1960` による代数的観察を出発点として連続状態・連続行動空間におけるこの問題を解決していた。遷移ダイナミクスが状態と制御入力に関して線形であり、1 ステップのコストがその両方に関して二次であるとき、ベルマンの後ろ向き動的計画法はすべてのステップで価値関数の二次形式を保つ。$100^d$ 個の要素をもつ表の代わりに、厳密な価値関数 $V_t(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}_t \mathbf{x}$ は、$d \times d$ の対称行列 $\mathbf{P}_t$ によって $d(d+1)/2$ 個の独立な数値だけでパラメータ化される。ベルマンバックアップは*リッカチ漸化式*（Riccati recursion）に帰着し、最適方策は線形状態フィードバック則 $\mathbf{u}_t^* = -\mathbf{K}_t \mathbf{x}_t$ となる。これが*線形二次レギュレータ*（linear-quadratic regulator、LQR）である :cite:`Kalman.1960,Anderson.Moore.2007`。

<!--en:347d1d6a-->
現実の物理システムの多くは非線形であり、アクチュエータには飽和がある。2 つのアイデアが LQR を非線形連続制御へと拡張し、現代の軌跡最適化およびロボティクスの骨格をなしている :cite:`Tedrake.2023,Bertsekas.2025`。第 1 に、*反復 LQR*（iterative LQR、iLQR）:cite:`Li.Todorov.2004` は、候補となる軌跡の周りで非線形ダイナミクスの線形化とコストの二次近似を繰り返し、後ろ向きパスで時変 LQR 問題を解いて前向きパスで軌跡を改善する。第 2 に、*モデル予測制御*（model predictive control、MPC）:cite:`Rawlings.Mayne.Diehl.2017` は、有限ホライズンの軌跡最適化を閉ループ方策へと変換する。すなわち、各ステップ $t$ において現在の観測状態 $\mathbf{x}_t$ から移動する先読みホライズン（receding lookahead horizon）にわたって制御問題を解き直し、最初の制御入力 $\mathbf{u}_{t|t}^*$ のみを適用して、ステップ $t+1$ で再計画する。本節では LQR、iLQR、および移動ホライズン MPC を展開し、局所線形フィードバックで十分な領域と非線形なエネルギー注入（energy pumping）が必要となる領域を明らかにしたうえで、MPC が :numref:`sec_model_based` における学習済みニューラルダイナミクスを用いた計画の土台をいかに築くかを示す。

<!--en:ad8ea4d7-->
## 線形二次制御

<!--en:f01a288f-->
### 状態空間ダイナミクスと二次コスト

<!--en:fc623c45-->
制御理論の記法では、連続状態を $\mathbf{x}_t \in \mathbb{R}^d$、行動（または制御入力）を $\mathbf{u}_t \in \mathbb{R}^m$ と書く。強化学習が報酬 $r(\mathbf{x}_t, \mathbf{u}_t)$ を最大化するのに対し、最適制御（optimal control）はステージコスト $c(\mathbf{x}_t, \mathbf{u}_t) = -r(\mathbf{x}_t, \mathbf{u}_t)$ を最小化する。離散時間線形システムは次のように時間発展する。

<!--en:7c80c399-->
ここで、$\mathbf{A} \in \mathbb{R}^{d \times d}$ は状態遷移行列、$\mathbf{B} \in \mathbb{R}^{d \times m}$ は制御入力行列である（:numref:`sec_mdl-odes-solvers` の線形常微分方程式の離散時間版に相当する）。目標状態は原点 $\mathbf{x} = \mathbf{0}$ である。$H$ ステップの有限ホライズンにわたり、次の総二次コストを最小化する。

<!--en:8e23e83b-->
ここで、$\mathbf{Q} \succeq \mathbf{0}$ と $\mathbf{Q}_H \succeq \mathbf{0}$ は $d \times d$ の対称半正定値な状態コスト行列であり、$\mathbf{R} \succ \mathbf{0}$ は $m \times m$ の対称正定値な制御コスト行列である。$\mathbf{R} \succ \mathbf{0}$ を課すことで、大きな制御入力には常に正のペナルティが伴うため、$\mathbf{u}_t$ に関する 1 ステップ最小化は一意な有限解をもつ。

<!--en:6f1c41ad-->
### 厳密な連続動的計画法とリッカチ漸化式

<!--en:be89e91c-->
$V_t(\mathbf{x})$ を状態 $\mathbf{x}$、ステップ $t \in \{0, \ldots, H\}$ からの最適残りコスト（cost-to-go）とする。終端ステップ $t = H$ では残りの制御入力はなく、$V_H(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}_H \mathbf{x}$（ただし $\mathbf{P}_H = \mathbf{Q}_H$）となる。$t+1$ から $t$ へと後ろ向きに進むと、有限ホライズン版のベルマンの最適性方程式（:numref:`sec_valueiter`）は次の式を与える。

<!--en:1458829c-->
後ろ向き帰納法により、ある対称半正定値行列について $V_{t+1}(\mathbf{x}') = {\mathbf{x}'}^\top \mathbf{P}_{t+1} \mathbf{x}'$（$\mathbf{P}_{t+1} \succeq \mathbf{0}$）が成り立つと仮定しよう。$\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$ を代入すると、括弧内の行動価値関数 $Q_t(\mathbf{x}, \mathbf{u})$ は $\mathbf{u}$ に関する狭義凸二次関数に展開される。

<!--en:b406a119-->
$\mathbf{u}$ に関する勾配をゼロとおくと、次のようになる。

<!--en:36c6c7ef-->
これより、一意な最適制御が閉形式で得られる。

<!--en:92801560-->
$\mathbf{u} = -\mathbf{K}_t \mathbf{x}$ を $Q_t(\mathbf{x}, \mathbf{u})$ に戻して代入すると、$V_t(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}_t \mathbf{x}$ が純粋な二次形式のままであることが確認でき、$\mathbf{P}_t$ は後ろ向きの*離散時間リッカチ漸化式*（discrete-time Riccati recursion）に従う。

<!--en:bad37c53-->
この漸化式は $\mathbf{P}_H = \mathbf{Q}_H$ から後ろ向きに 1 回計算するだけでよく、その際に状態を知る必要はない。したがってシステムの実行コストは低い。各ステップで制御器は $\mathbf{x}_t$ を計測し、保存しておいたゲインを掛けるだけでよい（:numref:`fig_rl_lqr_loop`）。

<!--en:5e9dfeb3-->
![閉ループでの LQR。制御器は計測された状態 $\mathbf{x}_t$ にゲイン $-\mathbf{K}_t$ を掛け、システム（制御用語でいう*プラント*）は次の状態で応答し、その状態が再び計測される。すべてのゲインはループ開始前に後ろ向きリッカチ漸化式によって計算される。](../img/mdl-rl-lqr-loop.svg)

<!--en:f92e6997-->
:eqref:`eq_lqr_gain` と :eqref:`eq_riccati_recursion` には、特筆すべき 3 つの性質がある。

<!--en:b3867507-->
1. **次元の呪いがないこと：**各ベルマンバックアップは、無限の行動空間 $\mathbb{R}^m$ とすべての状態 $\mathbb{R}^d$ にわたる最小化を、$O(d^3 + m^3)$ 回の $\mathbf{P}_{t+1}$ に対する行列演算に置き換える。

<!--en:984fbcb8-->
2. **線形フィードバックゲイン：**最適方策 $\pi_t^*(\mathbf{x}) = -\mathbf{K}_t \mathbf{x}$ は状態に関して線形であり、ゲイン行列 $\mathbf{K}_t \in \mathbb{R}^{m \times d}$ は $(\mathbf{A}, \mathbf{B}, \mathbf{Q}, \mathbf{R})$ のみに依存し、初期状態 $\mathbf{x}_0$ には依存しない。

<!--en:f5827be8-->
3. **確実性等価（certainty equivalence）：**平均ゼロの独立ノイズ $\mathbf{w}_t$（共分散 $\boldsymbol{\Sigma}$）が遷移に加わっても（$\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t + \mathbf{w}_t$）、交差項 $\mathbb{E}[\mathbf{w}_t^\top \mathbf{P}_{t+1}(\mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t)]$ は消え、価値関数に定数 $\textrm{tr}(\mathbf{P}_{t+1}\boldsymbol{\Sigma})$ が加わるだけで最適フィードバック行列 $\mathbf{K}_t$ は変化しない（演習問題 :exref:`ex_optimal-control-certainty-equivalence`）。

<!--en:02354bfa-->
ホライズン $H \to \infty$ の極限では（標準的な可制御性（controllability）および可観測性（observability）の条件のもとで）、$\mathbf{P}_t$ は*離散代数リッカチ方程式*（discrete algebraic Riccati equation、DARE）の定常解 $\mathbf{P}_\infty$ に収束し、時不変制御器 $\mathbf{u}_t = -\mathbf{K}_\infty \mathbf{x}_t$ は閉ループダイナミクス $\mathbf{x}_{t+1} = (\mathbf{A} - \mathbf{B}\mathbf{K}_\infty)\mathbf{x}_t$ を安定化して、$\mathbf{A} - \mathbf{B}\mathbf{K}_\infty$ のすべての固有値を単位円の厳密な内側、すなわち $\rho(\mathbf{A} - \mathbf{B}\mathbf{K}_\infty) < 1$ に配置する。

<!--en:b52bbe6e-->
### 二重積分器

<!--en:48209fb7-->
`riccati_lqr` を :numref:`sec_mdp` の状態拡張の例で扱った車両でテストしよう。これは直線上を動く単位質量の質点であり、状態は $\mathbf{x}_t = (p_t, v_t)^\top$（位置と速度）、制御入力は $u_t$（加速度）、タイムステップは $\Delta t = 0.1$ である。Newton の運動法則より、$p_{t+1} = p_t + v_t \Delta t + \frac{1}{2} u_t \Delta t^2$ および $v_{t+1} = v_t + u_t \Delta t$ が成り立つ。位置と速度のずれに $\mathbf{Q} = \textrm{diag}(1.0, 0.1)$ でペナルティを課し、制御入力の重み $r_u \in \{0.01, 0.10, 1.00\}$ を $\mathbf{R} = [r_u]$ において変化させる。

<!--en:593574f7-->
また、閉ループの位置軌跡 $p_t$（初期状態 $\mathbf{x}_0 = (1, 0)^\top$）を 3 つの制御入力ペナルティのもとでプロットできる。

<!--en:deb10f65-->
定常フィードバック則 $u_t^* = -k_p p_t - k_v v_t$ の 2 つの成分に注目しよう。動的計画法によって、位置誤差 $p_t$ に対してゲイン $k_p$ で押し戻し、速度 $v_t$ をゲイン $k_v$ で減衰させる*比例・微分*（proportional-derivative、PD）制御器が自動的に導出されている。制御入力が安価なとき（$R = 0.01$）、LQR は積極的なゲイン $\mathbf{K}_\infty = [7.613, 4.585]$ を選び、状態ノルムを 1 ステップあたり $\rho(\mathbf{A} - \mathbf{B}\mathbf{K}_\infty) = 0.761$ の比率で縮小させる。一方、制御入力が高価なとき（$R = 1.00$）は、穏やかな加速度 $\mathbf{K}_\infty = [0.930, 1.395]$ を用い、縮小率は $\rho = 0.930$、累積コスト $V_0(\mathbf{x}_0) = \mathbf{x}_0^\top \mathbf{P}_\infty \mathbf{x}_0$ はより大きくなる（$P_{00} = 15.00$ 対 $6.02$）。

<!--en:38ae19ce-->
## 平衡点周りでの非線形システムの線形化

<!--en:d141faa3-->
### `Pendulum-v1` の解析的線形化

<!--en:5ba48bcf-->
ここで、本章の残りの部分および :numref:`chap_deep_rl` を通じて用いる物理ベンチマークである倒立振子（inverted pendulum、`Pendulum-v1`）を考えよう。質量 $m = 1$、長さ $l = 1$ の剛体棒の一端が重力 $g = 10$ のもとで軸支されており、有界なモータトルク $u_t \in [-2, 2]$ によって駆動される。連続状態は $\mathbf{x}_t = (\vartheta_t, \dot\vartheta_t)^\top$ であり、$\vartheta_t$ は鉛直上向きから測った角度（$\vartheta = 0$ が直立、$\vartheta = \pi$ が真下への懸垂）、$\dot\vartheta_t \in [-8, 8]$ は角速度である。水平位置（$\vartheta = \pi/2$）における最大重力トルクは $mgl/2 = 5 > 2$ であるため、このモータは*劣駆動*（underactuated）であり、棒を最下点から頂上まで一息に持ち上げることはできない。シミュレータの 1 ステップ $\Delta t = 0.05\text{ s}$ にわたる半陰的オイラー法（semi-implicit Euler）の遷移 $\mathbf{x}_{t+1} = f(\mathbf{x}_t, u_t)$ は次の式で与えられる。

<!--en:4ad09bd2-->
ただし、角速度は $[-8, 8]$ にクリッピングされる。各ステップでは二次ステージコスト $c(\vartheta_t, \dot\vartheta_t, u_t) = \vartheta_t^2 + 0.1\,\dot\vartheta_t^2 + 0.001\,u_t^2$ が発生し（ここで $\vartheta_t$ は $[-\pi, \pi]$ に折り返され、$\mathbf{Q} = \textrm{diag}(1.0, 0.1)$ および $\mathbf{R} = [[0.001]]$ に対応する）、等価な表現として報酬 $r_t = -c(\vartheta_t, \dot\vartheta_t, u_t)$ が得られる。標準的なベンチマークのエピソードは $200$ ステップ（$10\text{ s}$）実行される。無制御の振子のスコアが $-1{,}200$ から $-1{,}500$ 程度であるのに対し、棒を振り上げ（swing-up）て直立状態に保つ制御器は、ベンチマークのランダムな初期状態にわたる平均で約 $-200$ 以上のスコアを達成する。本節では既知の運動方程式 :eqref:`eq_pendulum_discrete` を用いて直接計画を立てるが、:numref:`sec_deeprl`、:numref:`sec_sac`、および :numref:`sec_model_based` では、物理法則を知らないエージェントが行動を選択したりダイナミクスモデルを学習したりする場面でこの同じ振子ベンチマークに立ち返る。

<!--en:8552cd51-->
直立姿勢 $\mathbf{x}^* = (0, 0)^\top$ とゼロトルク $u^* = 0$ の組は不動点 $f(\mathbf{x}^*, u^*) = \mathbf{x}^*$ である。任意の基準対 $(\bar{\mathbf{x}}, \bar{u})$ の近傍では、1 次のテイラー展開によって非線形遷移を偏差 $\delta\mathbf{x}_t = \mathbf{x}_t - \bar{\mathbf{x}}$ と $\delta u_t = u_t - \bar{u}$ に関する局所線形モデルで近似できる。

<!--en:87d91525-->
:eqref:`eq_pendulum_discrete` を $(\vartheta_t, \dot\vartheta_t)$ および $u_t$ で微分すると、$\partial\dot\vartheta_{t+1}/\partial\vartheta_t = \frac{3g}{2l}\cos(\vartheta_t)\Delta t$ および $\partial\dot\vartheta_{t+1}/\partial u_t = \frac{3}{ml^2}\Delta t$ が得られる。

<!--en:6ff9e26c-->
直立不動点において、開ループのヤコビ行列 $\mathbf{A}$ は不安定な固有値 $\lambda_1 = 1.2133 > 1$ をもつ。そのため、わずかな傾きであっても 1 ステップあたり $21\%$ 超の割合で指数関数的に増大する（:numref:`sec_mdl-odes-solvers` の相図と同様である）。`Pendulum-v1` のコスト行列 $\mathbf{Q} = \textrm{diag}(1, 0.1)$ と $\mathbf{R} = [[0.001]]$ を用いてリッカチ漸化式を解くと、フィードバックゲイン $\mathbf{K}_\infty = [19.693, 5.263]$ が得られ、閉ループの固有値は両方とも単位円の内側（$\lambda_1 = 0.8537, \lambda_2 = 0.2467$）に引き込まれる。

<!--en:929b3736-->
### トルク飽和と吸引領域

<!--en:bf42e44a-->
モータトルクが $[-2, 2]$ にクリッピングされているとき、この線形制御器は頂上からどこまで離れた振子を捕捉できるだろうか。振子を引き下ろす重力トルクは $\frac{1}{2}mgl\sin\vartheta = 5\sin\vartheta$ であるのに対し、モータが供給できるトルクは高々 $|u| = 2$ である。$|\vartheta| > \arcsin(2/5) \approx 0.412\text{ rad}$（$23.6^\circ$）を超えると、重力がモータの最大保持トルクを上回る。クリッピング付き LQR 方策 $u_t = \textrm{clip}(-\mathbf{K}_\infty \mathbf{x}_t, -2, 2)$ を、完全な非線形振子に対して静止状態の 5 つの初期角度 $\vartheta_0$ からテストしてみよう。

<!--en:e0b29754-->
静的保持限界の内側（$\vartheta_0 = 0.20$ および $0.40\text{ rad}$）では、線形 LQR はほぼゼロのコストで振子をそのまま直立状態（$\vartheta_{200} = 0.000$）へと導く（収益は $-0.32$ と $-2.97$）。このしきい値をわずかに超えると（$\vartheta_0 = 0.50\text{ rad}$）、最大の負トルク $u = -2$ でも落下を食い止められず、200 ステップ後も棒は最下点を通過して揺れ続けている（収益 $-770.94$）。しきい値の先では、結果はもはや $\vartheta_0$ に対して単調ではない。$0.60\text{ rad}$ から始めた場合も棒は同様に落下するが、さらに数回揺れたあと、たまたまクリッピング付き制御器が捕捉できるほど十分に遅い速度で上がってくる（収益 $-546.37$）。これは制御ではなく偶然である。最下点（$\vartheta_0 = \pi$）からは、線形制御器の収益は $-1471.86$ となり、一度も頂上に到達しない。モータが劣駆動（$u_{\max} = 2 < 5$）であるため、$\vartheta_0 = \pi$ からの振り上げには*エネルギー注入*が必要になる。すなわち、一方向に押し、振子が最下点を通って振り戻る際にトルクを反転させて運動エネルギーと位置エネルギーを蓄え、棒が頂上付近の捕捉領域まで登り切って初めて直立安定化へ切り替える必要がある。

<!--en:6782f321-->
## 反復 LQR による非線形軌跡最適化

<!--en:eaf66320-->
### 軌跡に沿った線形化

<!--en:7af0af04-->
$(\mathbf{0}, 0)$ での単一の線形化は、非線形な sin$(\vartheta)$ の曲率（$\vartheta_0 = \pi$ からのスイングに沿ったもの）について何の情報ももたない。そこで、*反復線形二次制御*（iterative linear-quadratic regulation、iLQR）:cite:`Li.Todorov.2004` は、微分動的計画法（differential dynamic programming）:cite:`Mayne.1966` のガウス・ニュートン型変種であり、ノミナル軌跡全体の周りで線形化する。

<!--en:38d13582-->
ハードクリッピングによる折れ目を避けてトルク制約 $u_t \in [-2, 2]$ を滑らかに満たすため、$u_t = 2\tanh(v_t)$ とおき、無制約の制御変数 $v_t \in \mathbb{R}$ でパラメータ化する。任意のノミナル制御系列 $\bar{v}_{0:H-1}$ と、その前向きロールアウト $\bar{\mathbf{x}}_{0:H}$（真の非線形ダイナミクス $\mathbf{x}_{t+1} = f(\mathbf{x}_t, v_t)$ に従うもの。なお以下の振り上げでは速度上限に達しないため速度クリッピングは省く）が与えられたとき、iLQR の各反復は次の 2 つのパスを実行する。

<!--en:afcc99c6-->
1. **$(\bar{\mathbf{x}}_{0:H}, \bar{v}_{0:H-1})$ に沿った後ろ向きリッカチパス：**各ステップ $t = H-1, \ldots, 0$ において局所ヤコビ行列 $\mathbf{A}_t = \frac{\partial f}{\partial \mathbf{x}}(\bar{\mathbf{x}}_t, \bar{v}_t)$ と $\mathbf{B}_t = \frac{\partial f}{\partial v}(\bar{\mathbf{x}}_t, \bar{v}_t)$ を計算し、行動価値の摂動 $\delta Q_t(\delta\mathbf{x}_t, \delta v_t)$ の 2 次テイラー近似を、ステージコストの勾配 $(\boldsymbol{\ell}_{\mathbf{x},t}, \ell_{v,t})$、ヘッセ行列 $(\mathbf{L}_{\mathbf{x}\mathbf{x},t}, \ell_{vv,t})$、および次ステップの価値勾配 $\mathbf{v}_{\mathbf{x},t+1}$ とヘッセ行列 $\mathbf{V}_{\mathbf{x}\mathbf{x},t+1}$ から構成する。

   $$
   \begin{aligned}
   \mathbf{q}_{\mathbf{x},t} &= \boldsymbol{\ell}_{\mathbf{x},t} + \mathbf{A}_t^\top \mathbf{v}_{\mathbf{x},t+1}, \qquad q_{v,t} = \ell_{v,t} + \mathbf{B}_t^\top \mathbf{v}_{\mathbf{x},t+1}, \\
   \mathbf{Q}_{\mathbf{x}\mathbf{x},t} &= \mathbf{L}_{\mathbf{x}\mathbf{x},t} + \mathbf{A}_t^\top \mathbf{V}_{\mathbf{x}\mathbf{x},t+1}\mathbf{A}_t, \qquad \mathbf{q}_{v\mathbf{x},t} = \mathbf{B}_t^\top \mathbf{V}_{\mathbf{x}\mathbf{x},t+1}\mathbf{A}_t, \\
   q_{vv,t} &= \ell_{vv,t} + \mu + \mathbf{B}_t^\top \mathbf{V}_{\mathbf{x}\mathbf{x},t+1}\mathbf{B}_t,
   \end{aligned}
   $$

   ここで、$\mu > 0$ はレーベンバーグ・マーカート減衰項である。$\delta v_t$ に関して最小化すると、開ループのフィードフォワード補正量 $k_t = -q_{v,t}/q_{vv,t}$ と局所線形フィードバックゲイン $\mathbf{K}_t = -\mathbf{q}_{v\mathbf{x},t}/q_{vv,t}$ が得られる。

<!--en:361fe6f1-->
2. **バックトラッキング直線探索を伴う前向きロールアウト：**線形モデルは $(\bar{\mathbf{x}}_t, \bar{v}_t)$ の近傍でのみ正確であるため、ステップサイズ $\alpha \in \{1, 1/2, 1/4, \ldots\}$ を用いて*真の非線形ダイナミクス*をロールアウトする。

   $$
   \begin{aligned}
   v_t^{\textrm{new}} &= \bar{v}_t + \alpha\, k_t + \mathbf{K}_t(\mathbf{x}_t^{\textrm{new}} - \bar{\mathbf{x}}_t), \\
   \mathbf{x}_{t+1}^{\textrm{new}} &= f(\mathbf{x}_t^{\textrm{new}}, v_t^{\textrm{new}}),
   \end{aligned}
   $$
   :eqlabel:`eq_ilqr_forward`

   そして、真の非線形軌跡コストを減少させる最初の $\alpha$ を採用する。

<!--en:d6d8b592-->
軌跡コストの設計における 1 つの工夫は示唆に富む。$\vartheta_0 = \pi$ からエネルギーを注入するには、振子はまず反対側の $\vartheta > \pi$ へと振れてから、勢いよく $\vartheta = 0$ へ向かって振り戻る必要がある。もし重い二次ペナルティ $\vartheta_t^2$ を序盤のすべてのステップ $t$ に課すと、局所勾配法は $0$ から離れる方向へ振ることを拒んでしまう。バックスイング中に $\vartheta_t^2$ が増加するからである。そこで、スイング段階では状態コストの重みを軽くし（$q_{\textrm{run}} = 0.02$、$t < H - 20$）、終盤の捕捉区間では重くする（$q_T = 100$、$t \geq H - 20$）ことで、iLQR は序盤のステップを使ってエネルギーを注入し、終盤のステップで棒を $(\vartheta, \dot\vartheta) = (0, 0)$ に固定できる。

<!--en:6c2fa944-->
iLQR が 35 回の反復によって最下点 $(\pi, 0)$ から発見した軌跡を見てみよう。

<!--en:eae7aee9-->
- **ステップ 0〜10（初期のバックスイング）：**左方向へ押し（$u_0 = -1.782$）、$\vartheta_{10} = 2.652\text{ rad}$ まで振る。

<!--en:cfc48410-->
- **ステップ 10〜30（エネルギー注入）：**トルクを反転させ（$u_{10} = +0.965 \to u_{20} = +1.681$）、最下点を通過して反対側の $\vartheta_{30} = 4.038\text{ rad}$ まで振子を駆動し、重力位置エネルギーを蓄える。

<!--en:dc509d95-->
- **ステップ 30〜60（振り抜きと捕捉）：**再びトルクを反転させ（$u_{30} = -1.807 \to u_{40} = -1.832$）、最大 $5.7\text{ rad/s}$ で最下点を通過し、$\dot\vartheta_{40} = -4.440\text{ rad/s}$ の速度を保ったまま水平位置（$\vartheta_{40} = 1.567\text{ rad}$）を越えて直立の捕捉領域（$\vartheta_{50} = 0.375\text{ rad}$）へと駆け上がり、滑らかに制動して頂上で静止する（$\vartheta_{60} = 0.0009\text{ rad}$、$\vartheta_{80} = 0.0001\text{ rad}$、$u_{80} = 0.000$）。

<!--en:7e6e2474-->
この計画は局所最適解であり、最良の振り上げだと誤解してはならない。iLQR の各反復は与えられた軌跡を改善するものであるため、結果は初期制御系列に依存する。`ilqr_swingup` は最初の 15 ステップにわたる小さな押し（$\bar{v}_t = 0.2$）とそれ以降のゼロ入力から出発し、固定された減衰係数 $\mu$ を用いている。この初期の押しは重要であり、$\bar{v} = 0$ から始めると、同じ 35 回の反復では棒を頂上で静止させることができない。初期推定値へのこの依存性は以下で再び現れ、制御器が衝撃から回復できるかどうかを左右することになる。

<!--en:64648dcc-->
## 移動ホライズンモデル予測制御

<!--en:0701b5a3-->
### 開ループ計画、局所フィードバックチューブ、オンライン再計画

<!--en:a4d41115-->
最適軌跡 $(\bar{\mathbf{x}}_{0:H}, \bar{u}_{0:H-1})$ は、それだけではまだ頑健な制御器ではない。この計画を実行するには次の 3 通りの方法がある。

<!--en:474eb915-->
1. **開ループ実行：**事前計算された系列 $\bar{u}_t$ を、実際の状態 $\mathbf{x}_t$ を見ずにそのまま適用する。直立姿勢は不安定な鞍点であるため、わずかな外乱や数値ドリフトでも軌跡はコースから外れてしまう。

<!--en:2f4c7d72-->
2. **時変 LQR（time-varying LQR、TV-LQR）追従：**iLQR の最後の後ろ向きパスで計算された局所フィードバック行列 $\mathbf{K}_t$ を用い、$v_t = \bar{v}_t + \mathbf{K}_t(\mathbf{x}_t - \bar{\mathbf{x}}_t)$ を適用する。これによりオンライン計画コストなしでノミナル軌跡の周りに局所線形フィードバックチューブが形成されるが、強い衝撃によって $\mathbf{x}_t$ がチューブの外へ押し出され、新たな振り上げのためのエネルギー注入が必要な領域に入ると破綻する。

<!--en:2f5b9dfe-->
3. **移動ホライズンモデル予測制御（MPC）：**各タイムステップ $t$ において真の状態 $\mathbf{x}_t$ を観測し、$\mathbf{x}_t$ を始点とする先読みウィンドウ上で有限ホライズン制御問題を解き直し（前ステップの計画を 1 ステップずらしてウォームスタートする）、*最初の制御入力のみ* $u_{t|t}^*$ を適用して、これを $t+1$ で繰り返す :cite:`Rawlings.Mayne.Diehl.2017`。

<!--en:1b060e36-->
ここで実装する `mpc_action` では、3 番目の方式における 2 つの細部を定めている。第 1 はウィンドウの長さである。シフトされた計画は直前の計画より 1 ステップ短くなるため、ウィンドウはノミナル計画の 80 ステップから始まり、それとともに短縮していく。残り `H_min` $= 25$ ステップになった時点で計画の末尾をゼロで埋め、以降はウィンドウをその長さに保ったまま移動させる。第 2 はウォームスタートが失敗したときの対処である。iLQR は局所手法であるため、ウォームスタートされた計画は、元の軌跡がもはや頂上へ到達しなくなった場合でも古い軌跡の近くにとどまってしまう。再最適化した計画の終端が直立位置から $0.2\text{ rad}$ を超えて離れている場合、`mpc_action` はそれを破棄し、`H_mpc` $= 60$ ステップ（$3\text{ s}$）にわたる新たな振り上げを、`ilqr_swingup` がデフォルトで用いる初期の押しから計画し直す。

<!--en:64104767-->
3 つの制御器すべてを、$T = 120$ ステップにわたり初期状態 $\mathbf{x}_0 = (\pi, 0)$ から比較しよう。振子がちょうど水平位置を越えて直立の捕捉領域へ登っていくステップにおいて、外部から角速度インパルス $\Delta\dot\vartheta \in \{0.0, +0.5, +2.2\}\text{ rad/s}$ がステップ $t = 45$ で加わるものとする。

<!--en:a744764b-->
:numref:`fig_rl_optimal_control` に本節の 3 つの段階を並べて示す。直立平衡点周りの局所 LQR の価値等高線とトルク飽和境界、$(\vartheta, \dot\vartheta)$ 位相平面における iLQR のエネルギー注入軌跡、そして強いインパルス $\Delta\dot\vartheta = +2.2\text{ rad/s}$（$t = 45$）のもとでの外乱抑制（disturbance rejection）軌跡である。

<!--en:f1f9ce22-->
![局所 LQR から移動ホライズン MPC までの最適制御。(a) 直立 LQR の残りコスト等高線 $V_\infty(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_\infty\mathbf{x}$ と静的保持境界 $|\vartheta| \leq \arcsin(2/5) \approx 0.412\text{ rad}$（点線）。$\vartheta_0 = 0.40\text{ rad}$ からの出発では原点に収束する（青）のに対し、$\vartheta_0 = 0.50\text{ rad}$ ではモータトルク $|u| \leq 2$ が飽和して落下する（オレンジ破線）。(b) 80 ステップの iLQR 振り上げ軌跡（$(\vartheta, \dot\vartheta)$ 位相平面）。$(\pi, 0)$ から $\vartheta_{30} = 4.04\text{ rad}$ までエネルギーを注入したのち、最下点を振り抜けて直立平衡点 $(0, 0)$ に至る。(c) 折り返し角度 $\vartheta_t$ の時間変化（インパルス $\Delta\dot\vartheta = +2.2\text{ rad/s}$ が $t = 45$ すなわち $2.25\text{ s}$ の縦点線で加わった場合）。開ループと TV-LQR は失敗するが、移動ホライズン MPC は新たな振り上げを計画して $\vartheta = 0$ に戻る。](../img/mdl-rl-optimal-control.svg)

<!--en:c494df1b-->
この振子ベンチマークにおける 3 段階の外乱レベルは、3 つの制御器の特性を鮮明に描き分ける。

<!--en:827bdd9c-->
- **外乱なし（$\Delta\dot\vartheta = 0.0$）：**3 つとも振子を振り上げ、120 ステップにわたって $-332$ から $-334$ の収益を得る。これがベンチマークの目安として挙げた $-200$ を下回るのは 2 つの理由による。第 1 に、ベンチマークがランダムな初期状態にわたって平均するのに対し、ここでは最悪の状態である最下点から出発していること。第 2 に、iLQR は収益を採点する振子の報酬そのものではなく、スイング中の状態重みを軽くした独自の軌跡コストを最小化していることである。$t=80$ 以降の最後の 40 ステップでは、開ループは不安定な鞍点で $\vartheta_{120} = +0.075\text{ rad}$ へとドリフトし始めるが、TV-LQR と MPC は $\vartheta_{120} = 0.000\text{ rad}$ を維持する。

<!--en:bf667b07-->
- **中程度の外乱（$\Delta\dot\vartheta = +0.5\text{ rad/s}$）：**開ループ実行は事前計算されたスケジュールから弾き出されて再び落下し、最終的に $\vartheta_{120} = -2.939\text{ rad}$（収益 $-502.3$）に終わる。一方、TV-LQR（$\vartheta_{120} = -0.001\text{ rad}$、収益 $-333.9$）と MPC（$\vartheta_{120} = +0.000\text{ rad}$、収益 $-331.7$）はどちらも衝撃を吸収し、棒を直立状態に固定する。

<!--en:9ba62eeb-->
- **強い外乱（$\Delta\dot\vartheta = +2.2\text{ rad/s}$）：**衝撃によって上昇が食い止められ、振子は最下点に向かって落下し始める。古いノミナル軌跡周りの TV-LQR の線形フィードバックはトルクが飽和して破綻する（$\vartheta_{120} = +1.959\text{ rad}$、収益 $-797.9$）。回復には古い軌跡の線形近傍には存在しない新たな振り上げのエネルギー注入が必要だからである。MPC は振子を直立平衡点へと復帰させる（$\vartheta_{120} = 0.000\text{ rad}$、収益 $-558.2$）が、これは再シード（re-seed）規則のおかげである。衝撃が加わったステップでウォームスタート計画が頂上を外すため、`mpc_action` は外乱後の状態から新たな振り上げを計画し、以降はそれに従う。ウォームスタート計画は古い軌跡の近くにとどまるため、ウォームスタートからの再計画だけでは回復できない。

<!--en:d06401f6-->
### 古典的制御から強化学習へ

<!--en:ef0d2db7-->
:numref:`tab_control_vs_rl` は古典的最適制御と強化学習の対応関係をまとめた対訳表であり、本章の以降の節および :numref:`chap_deep_rl` が LQR と iLQR の各仮定をどのように緩和していくかを示している。

<!--en:de521a8c-->
:古典的最適制御と強化学習の架け橋。

<!--en:b095358b-->
| 概念 | 古典的最適制御 | 強化学習 |

<!--en:feae8bf9-->
| 変数 | 状態 $\mathbf{x}_t \in \mathbb{R}^d$、制御入力 $\mathbf{u}_t \in \mathbb{R}^m$ | 状態 $s_t \in \mathcal{S}$、行動 $a_t \in \mathcal{A}$ |

<!--en:a470ed08-->
| 目的関数 | 累積コスト $\sum_t c(\mathbf{x}_t, \mathbf{u}_t)$ の最小化 | 期待割引収益 $\mathbb{E}[\sum_t \gamma^t r(s_t, a_t)]$ の最大化 |

<!--en:97fdce49-->
| 価値関数 | 残りコスト $V_t(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}_t \mathbf{x}$（LQR） | 状態価値 $V^\pi(s)$ または行動価値 $Q^\pi(s, a)$ |

<!--en:23e91ca9-->
| ベルマンバックアップ | リッカチ漸化式 :eqref:`eq_riccati_recursion` | ベルマン期待値・最適作用素（:numref:`sec_valueiter`） |

<!--en:34993305-->
| 既知の非線形物理系 $f$ | 勾配ベースの軌跡最適化（iLQR・DDP）＋ MPC | 木探索またはサンプリングベースの MPC（:numref:`sec_model_based`） |

<!--en:555994e6-->
| 未知ダイナミクス、学習モデル $\hat{f}_\phi$ | システム同定＋確実性等価 MPC | ランダムシューティング・CEM によるモデルベース強化学習（:numref:`sec_model_based`） |

<!--en:a912dbc4-->
| 未知ダイナミクス、モデルなし | モデルフリー適応制御 | 方策勾配（:numref:`sec_policygradient`、:numref:`sec_deeprl`）、PPO（:numref:`sec_ppo`）、DQN（:numref:`sec_dqn`）、SAC（:numref:`sec_sac`） |

<!--en:a9ae8950-->
ダイナミクス $f$ が未知であり、しかも遷移モデルを当てはめずに行動しなければならない場合、後ろ向きリッカチパスに必要なヤコビ行列 $\mathbf{A}_t = \partial f/\partial \mathbf{x}$ と $\mathbf{B}_t = \partial f/\partial \mathbf{u}$ は計算できない。:numref:`sec_policygradient` では、確率的方策 $\pi_\theta(a \mid s)$ が自身のサンプリングした行動の対数確率を代わりに微分することで、これらの未知の導関数をいかに迂回するかを示す。

<!--en:30ac03ff-->
## まとめ

<!--en:74c7b621-->
線形ダイナミクス $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t$ と二次コスト $\mathbf{x}^\top\mathbf{Q}\mathbf{x} + \mathbf{u}^\top\mathbf{R}\mathbf{u}$ に対して、ベルマンの後ろ向き動的計画法は連続空間における厳密な閉形式解をもつ。すなわち、残りコストは二次形式 $V_t(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x}$ となり、$\mathbf{P}_t$ は後ろ向きリッカチ漸化式に従い、最適方策は線形フィードバック制御器 $\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}_t$（LQR）で与えられる。非線形システムを不安定不動点の周りで解析的ヤコビ行列によって線形化すると、アクチュエータ制限下の捕捉領域内にある状態は安定化できるが、その外側で劣駆動の振り上げによるエネルギー注入が必要になると破綻する。反復 LQR（iLQR）はノミナル軌跡の周りでダイナミクスの線形化とコストの二次近似を繰り返し、後ろ向きリッカチパスと前向き非線形ロールアウトによって多段階のエネルギー注入軌跡を発見する。開ループ計画の実行は外乱のもとで破綻し、固定軌跡周りの時変 LQR フィードバックは小さな摂動しか扱えないが、移動ホライズンモデル予測制御（MPC）は各ステップで現在の状態から有限ホライズン問題を解き直すため、大きな外乱によって全く新しい軌跡が必要になった場合でも回復できる。

<!--en:2294243a-->
## 演習問題

<!--en:dc95d141-->
1. **リッカチバックアップにおける平方完成。**$\mathbf{S}_{t+1} = \mathbf{R} + \mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{B}$ および $\mathbf{K}_t = \mathbf{S}_{t+1}^{-1}\mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{A}$ とおく。
    1. 積を展開することで、任意の $\mathbf{x} \in \mathbb{R}^d$ と $\mathbf{u} \in \mathbb{R}^m$ に対して

       $$Q_t(\mathbf{x}, \mathbf{u}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x} + (\mathbf{u} + \mathbf{K}_t\mathbf{x})^\top \mathbf{S}_{t+1}(\mathbf{u} + \mathbf{K}_t\mathbf{x}),$$

       が成り立つことを示せ（ここで $\mathbf{P}_t$ は :eqref:`eq_riccati_recursion` で与えられる）。
    1. $\mathbf{R} \succ \mathbf{0}$ と $\mathbf{P}_{t+1} \succeq \mathbf{0}$ を用いて、$\mathbf{S}_{t+1} \succ \mathbf{0}$ であること、$\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}$ が $Q_t(\mathbf{x}, \mathbf{u})$ の一意な大域的最小点であること、および $\mathbf{P}_t \succeq \mathbf{0}$ であることを導け。
    1. Joseph の安定化形式

       $$\mathbf{P}_t = \mathbf{Q} + \mathbf{K}_t^\top\mathbf{R}\mathbf{K}_t + (\mathbf{A} - \mathbf{B}\mathbf{K}_t)^\top\mathbf{P}_{t+1}(\mathbf{A} - \mathbf{B}\mathbf{K}_t),$$

       が成り立つことを確かめ、$\mathbf{K}_t$ を任意のフィードバック行列 $\mathbf{K}$ に置き換えても右辺のすべての項が明らかに半正定値となる理由を説明せよ。

<!--en:feed7733-->
1. **加法的ガウスノイズのもとでの確実性等価。**線形ダイナミクス :eqref:`eq_lqr_dynamics` が、平均ゼロで独立なノイズベクトル $\mathbf{w}_0, \ldots, \mathbf{w}_{H-1}$（共分散 $\mathbb{E}[\mathbf{w}_t\mathbf{w}_t^\top] = \boldsymbol{\Sigma}$）によって摂動され、$\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t + \mathbf{w}_t$ となると仮定する。
    1. 後ろ向き帰納法により、最適残りコストが $V_t(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x} + c_t$ となることを証明せよ。ここで、$\mathbf{P}_t$ は決定論的な場合と全く同じリッカチ漸化式 :eqref:`eq_riccati_recursion` に従い、$c_t = \sum_{k=t+1}^{H} \textrm{tr}(\mathbf{P}_k \boldsymbol{\Sigma})$ である。
    1. 最適フィードバック方策 $\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}_t$ がノイズの共分散 $\boldsymbol{\Sigma}$ に依存しないことを導け。もしノイズが乗法的であり、$\mathbf{x}_{t+1} = (\mathbf{A} + \tilde{\mathbf{A}}_t)\mathbf{x}_t + (\mathbf{B} + \tilde{\mathbf{B}}_t)\mathbf{u}_t$ である場合、この確実性等価の性質はどこで破綻するか。
   :exlabel:`ex_optimal-control-certainty-equivalence`

<!--en:36b5f72e-->
1. [code] **可制御性と無限ホライズン極限。**線形システム $(\mathbf{A}, \mathbf{B})$（状態数 $d$）は、可制御性行列 $\mathcal{C} = [\mathbf{B}, \mathbf{A}\mathbf{B}, \ldots, \mathbf{A}^{d-1}\mathbf{B}]$ が行フルランク $d$ をもつとき*可制御*であるという。
    1. $\mathbf{x}_0 = \mathbf{0}$ から出発したとき、$d$ ステップ後に到達する状態が $\mathbf{x}_d = \mathcal{C}\,[\mathbf{u}_{d-1}^\top, \mathbf{u}_{d-2}^\top, \ldots, \mathbf{u}_0^\top]^\top$ に等しくなることを示し、$\mathcal{C}$ が行フルランクをもつことが $\mathbb{R}^d$ 内の任意の目標状態に $d$ ステップで到達するための必要十分条件であることを示せ。
    1. $\textrm{rank}(\mathcal{C})$ と $\det(\mathcal{C})$ を、二重積分器 $(\mathbf{A}_{\textrm{di}}, \mathbf{B}_{\textrm{di}})$ および線形化した直立振子 $(\mathbf{A}_p, \mathbf{B}_p)$ に対して計算せよ。
    1. 第 1 座標に制御が作用しない（$\mathbf{B}_{\textrm{un}} = [0, 0]^\top$ に相当する）非結合な不安定システム $\mathbf{A} = \textrm{diag}(1.2, 0.8), \mathbf{B} = [0, 1]^\top$ を設定し、`riccati_lqr` を $H \in \{10, 50, 100\}$ に対して実行せよ。$\mathbf{P}_0$ のどの成分が $H \to \infty$ で発散するか。また、それはなぜか。

<!--en:62e380fc-->
1. [code] **エネルギー整形と軌跡最適化の比較。**古典的な非線形制御では、振子の全力学的エネルギー $E(\vartheta, \dot\vartheta) = \frac{1}{6}ml^2\dot\vartheta^2 + \frac{1}{2}mgl\cos\vartheta$ を制御して振り上げることが多い。この値は $E^* = \frac{1}{2}mgl = 5$（直立静止平衡点 $(\vartheta=0, \dot\vartheta=0)$）および $-5$（最下点 $(\vartheta=\pi, \dot\vartheta=0)$）に等しい :cite:`Tedrake.2023`。
    1. 連続時間の摩擦のない軌跡（$\ddot\vartheta = \frac{3g}{2l}\sin\vartheta + \frac{3}{ml^2}u$）に沿ったエネルギーの時間変化率が $\dot E = u\dot\vartheta$ となることを示し、トルク $u = -k_E (E - E^*)\dot\vartheta$（$k_E > 0$）を加えると $(E - E^*)^2$ が $\dot\vartheta \neq 0$ である限り単調に減少することを示せ。
    1. `pendulum_step` 上でハイブリッド制御器を実装せよ。すなわち、クリッピング付きエネルギー注入 $u = \textrm{clip}(-k_E (E - E^*)\dot\vartheta, -2, 2)$（微小な初期キック $u_0 = 2$ を伴う）を $|\textrm{wrap}(\vartheta)| > 0.35$ のときに適用し、直立 LQR 制御器 $u = \textrm{clip}(-\mathbf{K}_\infty \mathbf{x}, -2, 2)$ へ $|\textrm{wrap}(\vartheta)| \leq 0.35$ となった時点で切り替えるものとする。200 ステップの $(\pi, 0)$ からの収益を iLQR の収益と比較せよ。

<!--en:3ae20b74-->
1. [code] **MPC におけるホライズン長と終端コスト。**`mpc_action` はシフトされた計画を少なくとも `H_min` $= 25$ ステップにわたって再最適化し、その計画が頂上を外したときには `H_mpc` $= 60$ ステップ（$3\text{ s}$）にわたる新たな振り上げを計画する。
    1. 本節の $\Delta\dot\vartheta = +2.2$ の実験を `H_mpc` $\in \{15, 25, 40, 60\}$ で繰り返せ。どのホライズンで MPC は振子を頂上へ戻せるか。それらをノミナルな振り上げが頂上に到達するのに要するステップ数と比較し、短いホライズンではプランナが最下点付近に閉じ込められる理由を説明せよ。
    1. 終端コスト行列 $q_T \mathbf{Q}_{\textrm{base}}$（`ilqr_swingup` 内）を、無限ホライズンの DARE 行列 $\mathbf{P}_\infty$（直立平衡点で `riccati_lqr` により計算したもの）に置き換えよ。すでに直立の捕捉領域内（$|\vartheta_0| \leq 0.35$）にある初期状態に対して、終端コスト $\mathbf{x}_H^\top \mathbf{P}_\infty \mathbf{x}_H$ をもつ移動ホライズン LQR 制御器は、*厳密な*無限ホライズン最適制御 $\mathbf{u}_t^* = -\mathbf{K}_\infty \mathbf{x}_t$ を、先読みがわずか 1 ステップ（$H = 1$）のときでさえ生成することを示せ。

<!--en:b2493278-->
1. **Pontryagin の共状態と価値勾配。**$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$ を滑らかな決定論的システムとし、$J = \sum_{t=0}^{H-1} c(\mathbf{x}_t, \mathbf{u}_t) + c_H(\mathbf{x}_H)$ とおく。ラグランジュ乗数（*共状態*または*随伴変数*とよばれる）$\boldsymbol{\lambda}_1, \ldots, \boldsymbol{\lambda}_H \in \mathbb{R}^d$ を、制約 $\mathbf{x}_{t+1} - f(\mathbf{x}_t, \mathbf{u}_t) = \mathbf{0}$ に対して導入する。
    1. ラグランジアン $\mathcal{L} = J + \sum_{t=0}^{H-1} \boldsymbol{\lambda}_{t+1}^\top (f(\mathbf{x}_t, \mathbf{u}_t) - \mathbf{x}_{t+1})$ を $\mathbf{x}_t$ と $\mathbf{u}_t$ で微分し、後ろ向き随伴方程式

       $$\boldsymbol{\lambda}_H = \nabla_{\mathbf{x}} c_H(\mathbf{x}_H), \qquad \boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{A}_t^\top \boldsymbol{\lambda}_{t+1},$$

       および勾配 $\nabla_{\mathbf{u}_t} J = \nabla_{\mathbf{u}} c(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{B}_t^\top \boldsymbol{\lambda}_{t+1}$ を導出せよ :cite:`Pontryagin.Boltyanskii.Gamkrelidze.ea.1962`。
    1. この後ろ向き随伴漸化式を、1 次のベクトル $\mathbf{v}_{\mathbf{x},t}$ および $q_{v,t}$（`ilqr_swingup` の後ろ向きパスにおけるもの）、ならびに :numref:`sec_bptt` の通時的誤差逆伝播法と比較せよ。iLQR の 2 次のヘッセ行列の漸化式 $\mathbf{V}_{\mathbf{x}\mathbf{x},t}$ は、1 次の通時的誤差逆伝播法に対して何を付け加えているか。

<!--en:5c578ac8-->
[ディスカッション](https://d2l.discourse.group/t/36887)

<!--en:964cfee6-->
最適制御、LQR、モデル予測制御<br>**線形二次制御・後ろ向きリッカチ漸化式・局所線形化・反復 LQR・移動ホライズン MPC**

<!--en:0fe41d6b-->
::: {.slide title="連続状態空間と Bellman の呪い"}

<!--en:b0445376-->
[動機とメンタルモデル]{.kicker}

<!--en:519a32a5-->
- **次元の呪い：**$d$ 個の状態座標をそれぞれ $100$ ビンに離散化すると $100^d$ 個のグリッドセルが必要（$10^{28}$ 個：7 関節ロボットアームで $d = 14$ の場合）

<!--en:5f7aa81b-->
- **Kalman の代数的洞察（1960）：**線形ダイナミクスと二次ステージコストは厳密な二次価値関数 $V_t(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x}$ を保ち、独立な行列要素はわずか $d(d+1)/2$ 個

<!--en:becc2d74-->
::: {.slide title="線形二次レギュレータ（LQR）問題"}

<!--en:e82de56e-->
[線形ダイナミクスと二次コスト]{.kicker}

<!--en:b1606d16-->
- **ダイナミクスとペナルティ：**$\mathbf{A}, \mathbf{B}$ が状態の時間発展を支配し、$\mathbf{Q} \succeq \mathbf{0}$ が状態誤差に、$\mathbf{R} \succ \mathbf{0}$ が制御入力に正のペナルティを課す

<!--en:7be432e0-->
::: {.slide title="厳密な連続動的計画法"}

<!--en:3b1a2852-->
[ベルマンバックアップの展開]{.kicker}

<!--en:f853f082-->
$t = H$（$\mathbf{P}_H = \mathbf{Q}_H$）からの後ろ向き帰納法によりステップ $t+1$ の残りコストが二次形式 $V_{t+1}(\mathbf{x}') = {\mathbf{x}'}^\top\mathbf{P}_{t+1}\mathbf{x}'$ であると仮定すると、ベルマンの最適性方程式は次のようになる。

<!--en:100e9be2-->
- **$\mathbf{u}$ に関する狭義凸二次関数：**行動価値の括弧内 $Q_t(\mathbf{x}, \mathbf{u})$ を展開すると、$\mathbf{x}$ の二次項、交差項 $2\mathbf{u}^\top\mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{A}\mathbf{x}$、制御の二次項 $\mathbf{u}^\top(\mathbf{R} + \mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{B})\mathbf{u}$ に整理される

<!--en:7d5edaaf-->
- **一階最適性：**$\nabla_{\mathbf{u}} Q_t(\mathbf{x}, \mathbf{u}) = 2(\mathbf{R} + \mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{B})\mathbf{u} + 2\mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{A}\mathbf{x} = \mathbf{0}$ とおくことで、バックアップが閉形式で解析的に解ける

<!--en:4f00e5de-->
::: {.slide title="後ろ向きリッカチ漸化式とゲイン行列"}

<!--en:bcdef964-->
[閉形式の線形状態フィードバック]{.kicker}

<!--en:757c5394-->
- **3 つの構造的性質：**（1）1 ステップあたり $O(d^3 + m^3)$、（2）$\mathbf{K}_t$ は初期状態 $\mathbf{x}_0$ に非依存、（3）加法的ガウスノイズ $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ のもとでも $\mathbf{K}_t$ は不変（*確実性等価*）

<!--en:0e33df82-->
::: {.slide title="二重積分器：LQR による PD 制御器の導出" layout="tight"}

<!--en:5bb47a6b-->
[無限ホライズンの定常フィードバック]{.kicker}

<!--en:01b93a7d-->
::: {.slide title="二重積分器：LQR による PD 制御器の導出（続き）" layout="tight"}

<!--en:5bb47a6b-2-->
[無限ホライズンの定常フィードバック]{.kicker}

<!--en:1453b7fd-->
`riccati_lqr` を 1 次元の車両 $\mathbf{x}_t = (p_t, v_t)^\top$（加速度 $u_t$ で駆動）に適用すると、比例・微分（PD）制御器 $u_t^* = -k_p p_t - k_v v_t$ が自動的に導出される。

<!--en:20e15f8f-->
- **漸近安定性：**$H \to \infty$ で $\mathbf{P}_t \to \mathbf{P}_\infty$（離散代数リッカチ方程式）となり、$\mathbf{A} - \mathbf{B}\mathbf{K}_\infty$ の全固有値が単位円の厳密な内側に配置される（$\rho < 1$）

<!--en:c047ff2f-->
::: {.slide title="制御入力ペナルティと減衰速度"}

<!--en:4db807b3-->
[二重積分器の閉ループ軌跡]{.kicker}

<!--en:338b2db6-->
制御ペナルティ $R \in \{0.01, 0.10, 1.00\}$ を初期位置 $\mathbf{x}_0 = (1, 0)^\top$ から変化させると、$\mathbf{R}$ がアクチュエータの制御量と収束速度をどうトレードオフするかがわかる。

<!--en:b0cb2959-->
- **安価な制御（$R = 0.01$）：**高いゲイン $\mathbf{K}_\infty = [7.613, 4.585]$ が状態誤差を急速に縮小させる（$\rho = 0.761$、価値 $P_{00} = 6.02$）

<!--en:071055c2-->
- **高価な制御（$R = 1.00$）：**穏やかなゲイン $\mathbf{K}_\infty = [0.930, 1.395]$ が緩やかに制動する（$\rho = 0.930$、価値 $P_{00} = 15.00$）

<!--en:33c15fab-->
::: {.slide title="Pendulum-v1 連続制御ベンチマーク"}

<!--en:0a6dd942-->
[非線形制御ベンチマーク]{.kicker}

<!--en:5981ea43-->
- **強い劣駆動性：**モータトルクは $u_t \in [-2, 2]$ に制限される一方、水平位置（$\vartheta = \pi/2$）での最大重力トルクは $\frac{1}{2}mgl = 5 > 2$

<!--en:347d6790-->
- **ステージコストと達成基準：**$c_t = \vartheta_t^2 + 0.1\,\dot{\vartheta}_t^2 + 0.001\,u_t^2$。ランダム行動のスコアは $\approx -1{,}200$ であるのに対し、振り上げと直立維持はランダムな初期状態での平均で**約 $-200$ 以上**を達成

<!--en:58f3bf72-->
::: {.slide title="平衡点における非線形物理系の線形化"}

<!--en:97b94b3f-->
[1 次テイラー近似]{.kicker}

<!--en:87cbcc76-->
滑らかな非線形システム $\mathbf{x}_{t+1} = f(\mathbf{x}_t, u_t)$ の不動点 $f(\mathbf{x}^*, u^*) = \mathbf{x}^*$ の近傍では、1 次のテイラー展開により $(\delta\mathbf{x}_t, \delta u_t)$ に関する局所線形モデルが得られる。

<!--en:550f9f3d-->
- **`Pendulum-v1` の直立平衡点：**$(\vartheta^*, \dot{\vartheta}^*) = (0, 0)$、$u^* = 0$ において、$\sin\vartheta_t \approx \vartheta_t$ より不安定な頂上の鞍点周りの厳密な局所線形モデル $(\mathbf{A}_p, \mathbf{B}_p)$ が得られる

<!--en:f8ac0d72-->
::: {.slide title="局所 LQR による直立振子の安定化"}

<!--en:d8688f4d-->
[不安定固有値を単位円の内側へ移す]{.kicker}

<!--en:c7255ef2-->
厳密なヤコビ行列 $(\mathbf{A}_p, \mathbf{B}_p)$ を `Pendulum-v1` の直立平衡点 $\vartheta = 0$ で計算し、リッカチ漸化式を実行すると鞍点が安定化される。

<!--en:8e266f8d-->
- **開ループの不安定性と閉ループの安定性：**開ループのヤコビ行列は不安定固有値 $\lambda_1 = 1.2133 > 1$ をもつ（傾きが毎ステップ $21\%$ 増大）が、LQR ゲイン $\mathbf{K}_\infty = [19.693, 5.263]$ が両固有値を単位円内（$0.8537$ と $0.2467$）に引き込む

<!--en:36d0f5ab-->
::: {.slide title="局所 LQR が破綻する領域：トルク飽和"}

<!--en:8da34ba4-->
[劣駆動性と捕捉領域]{.kicker}

<!--en:94fe11da-->
重力はトルク $\frac{1}{2}mgl\sin\vartheta = 5\sin\vartheta$ で振子を引き下ろす一方、モータは $|u_t| \leq 2$ に制限されるため、静的保持境界は $|\vartheta| = \arcsin(2/5) \approx 0.412\text{ rad}$（$23.6^\circ$）となる。

<!--en:0c650c16-->
- **明確な捕捉領域の境界：**$\vartheta_0 = 0.20$ または $0.40\text{ rad}$ から始めると、クリッピング付き線形 LQR が棒を捕捉し $\vartheta_{200} = 0.000\text{ rad}$ で直立維持する

<!--en:ce823331-->
- **捕捉領域外での破綻：**$\vartheta_0 = 0.50\text{ rad}$（$5\sin 0.50 = 2.40 > 2$）や最下点（$\vartheta_0 = \pi$）からは最大トルク $u = -2$ でも重力に抗えず、振り上げには複数回スイングする**エネルギー注入**が必要になる。$0.60\text{ rad}$ からも同様に落下し、数回のスイング後に偶然捕捉されるにすぎない。

<!--en:b2e8a7f7-->
::: {.slide title="反復 LQR（iLQR）：軌跡に沿った線形化"}

<!--en:99799131-->
[ガウス・ニュートン軌跡最適化]{.kicker}

<!--en:fce7b53d-->
- **滑らかなトルク圧縮：**$u_t = 2\tanh(v_t)$ により微分可能な形で $u_t \in (-2, 2)$ を課す

<!--en:ef69c0d5-->
- **後ろ向きリッカチ＋前向き直線探索：**時変ヤコビ行列 $(\mathbf{A}_t, \mathbf{B}_t)$ を計算し、後ろ向きに $(k_t, \mathbf{K}_t)$ を求め、バックトラッキング係数 $\alpha$ で真の非線形物理系をロールアウトする

<!--en:fe085d34-->
::: {.slide title="iLQR による多段階エネルギー注入の発見" layout="tight"}

<!--en:87b66eb8-->
[最下点からの非線形振り上げ]{.kicker}

<!--en:ae54042f-->
35 回の反復により、下向き平衡点 $(\pi, 0)$ から $H = 80$ ステップ（$4.0\text{ s}$）にわたる 3 段階の振り上げ・捕捉軌跡を iLQR が発見する。

<!--en:f71d0087-->
- **ステップ 0〜10（バックスイング）：**左へ押し（$u_0 = -1.78$）、$\vartheta_{10} = 2.65\text{ rad}$ まで振る

<!--en:70870944-->
- **ステップ 10〜30（エネルギー注入）：**トルクを反転させ（$u_{20} = +1.68$）、最下点を通過して $\vartheta_{30} = 4.04\text{ rad}$ まで振り上げる

<!--en:40ab622b-->
- **ステップ 30〜80（振り抜きと捕捉）：**再び反転させ（$u_{40} = -1.83$）、上昇中も $\dot{\vartheta}_{40} = -4.44\text{ rad/s}$ で動きつつ制動し、$\vartheta_{80} = 0.0001\text{ rad}$ で静止する

<!--en:db3d3971-->
::: {.slide title="iLQR による多段階エネルギー注入の発見（続き）"}

<!--en:87b66eb8-2-->
[最下点からの非線形振り上げ]{.kicker}

<!--en:53bae74d-->
懸垂状態の開始から直立捕捉状態までの 10 ステップごとの振り上げ軌跡を示す。

<!--en:f5f1f813-->
::: {.slide title="開ループ・TV-LQR・移動ホライズン MPC の比較"}

<!--en:845a98a8-->
[軌跡計画から閉ループ制御へ]{.kicker}

<!--en:f72870f8-->
1. **開ループ（$u_t = \bar{u}_t$）：**状態 $\mathbf{x}_t$ を見ずに事前計算したトルクを適用するため、わずかな衝撃でも不安定な頂上の鞍点から落下する

<!--en:90d2aeaa-->
2. **TV-LQR（$v_t = \bar{v}_t + \mathbf{K}_t(\mathbf{x}_t - \bar{\mathbf{x}}_t)$）：**オンライン計算コストなしで軌跡を局所フィードバックチューブで囲むが、チューブ外へ押し出されると破綻する

<!--en:11787734-->
3. **移動ホライズン MPC：**毎ステップ計測した $\mathbf{x}_t$ からシフト済み計画をウォームスタートとして iLQR を解き直し、$u_{t|t}^*$ を適用して $t+1$ で再計画する。ウォームスタート計画が頂上を外した場合は 60 ステップの振り上げを新たに計画する。

<!--en:14f9d41a-->
::: {.slide title="t = 45 でのインパルスに対する外乱抑制" layout="tight"}

<!--en:16d9137a-->
[開ループ・TV-LQR・MPC の検証]{.kicker}

<!--en:44b82a94-->
外部からの角速度インパルス $\Delta\dot{\vartheta} \in \{0.0, +0.5, +2.2\}\text{ rad/s}$ をステップ $t = 45$（振子がちょうど捕捉領域へ登る瞬間）で加えると、3 つの制御器の差が明確に現れる。

<!--en:6990b962-->
::: {.slide title="t = 45 でのインパルスに対する外乱抑制（続き）" layout="tall-output"}

<!--en:16d9137a-2-->
[開ループ・TV-LQR・MPC の検証]{.kicker}

<!--en:8097d6eb-->
各衝撃に対する各制御器の最終角度と収益を示す。

<!--en:38c7aac4-->
::: {.slide title="相図と強い衝撃からの回復"}

<!--en:caa4028d-->
[なぜオンライン再計画が不可欠か]{.kicker}

<!--en:1c64d539-->
下の 3 つのパネルを比較すると、局所 LQR、軌跡最適化、移動ホライズン再計画がどのように噛み合うかがわかる。

<!--en:b7118125-->
- **中程度の衝撃（$\Delta\dot{\vartheta} = +0.5$）：**開ループは落下する（$\vartheta_{120} = -2.94$）が、TV-LQR と MPC は線形チューブ内で摂動を吸収する（$\vartheta_{120} = 0.000$）

<!--en:4cb7cdc2-->
- **強い衝撃（$\Delta\dot{\vartheta} = +2.2$）：**上昇が阻まれ、TV-LQR は旧計画周りの線形フィードバックが飽和して失敗する（$\vartheta_{120} = +1.96$）一方、**MPC はウォームスタート計画が頂上を外したことを検知し、新たな振り上げを再シードして $\vartheta_{120} = 0.000$ に復帰させる**

<!--en:6f19a7bd-->
::: {.slide title="古典的最適制御と強化学習"}

<!--en:92be91c3-->
[両分野をつなぐ統一対訳表]{.kicker}

<!--en:933f09ed-->
- **変数と目的関数：**最適制御はステージコスト $\sum_t c(\mathbf{x}_t, \mathbf{u}_t)$ を $(\mathbf{x}_t, \mathbf{u}_t) \in \mathbb{R}^d \times \mathbb{R}^m$ 上で最小化し、RL は割引収益 $\mathbb{E}[\sum_t \gamma^t r(s_t, a_t)]$ を最大化する

<!--en:726d12c5-->
- **バックアップ：**$\mathbf{x}^\top\mathbf{P}_t\mathbf{x}$ に対するリッカチ漸化式は、ベルマン作用素 $\mathcal{T}$ による $V(s), Q(s, a)$ の更新に対応する

<!--en:49549736-->
- **未知の物理系：**システム同定＋ MPC は、モデルベース RL（§{sec_model_based}）やモデルフリー方策勾配（§{sec_policygradient}）へと直結する

<!--en:46bca58d-->
::: {.slide title="概念チェック：LQR、iLQR、MPC"}

<!--en:929b7671-->
[理解度の確認]{.kicker}

<!--en:b28f9f39-->
リッカチバックアップ、トルク飽和、移動ホライズン制御についての直感を確かめよう。

<!--en:799e1572-->
1. **リッカチ行列の次元：**状態次元 $d = 36$ のヒューマノイドロボットにおいて、各軸 $10$ ビンの粗いグリッドと対称リッカチ行列 $\mathbf{P}_t$ では、それぞれ何個の要素が必要になるか。

<!--en:8b11b9a3-->
2. **確実性等価：**平均ゼロのガウス分布に従う突風 $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ が線形の二重積分器 $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t + \mathbf{w}_t$ を摂動するとき、最適フィードバックゲイン $\mathbf{K}_t$ はどう変化するか。

<!--en:7699d875-->
3. **TV-LQR 対 MPC：**なぜ TV-LQR は $+0.5\text{ rad/s}$ のインパルス（$t = 45$）からは回復できたのに、$+2.2\text{ rad/s}$ のインパルスでは完全に失敗したのか。

<!--en:557b98b6-->
**解答：**

<!--en:be245716-->
1. 10 ビンのグリッドには $10^{36}$ 個のセルが必要だが、$\mathbf{P}_t \in \mathbb{R}^{36 \times 36}$ の独立な数値はわずか $d(d+1)/2 = 36 \times 37 / 2 = 666$ 個である。

<!--en:3b675c10-->
2. $\mathbf{K}_t$ は一切変化しない。$\mathbb{E}[(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{w})^\top\mathbf{P}_{t+1}(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{w})]$ において $\mathbf{w}$ の線形の交差項は消え、$\mathbb{E}[\mathbf{w}^\top\mathbf{P}_{t+1}\mathbf{w}] = \textrm{tr}(\mathbf{P}_{t+1}\boldsymbol{\Sigma})$ は $\mathbf{u}$ に無関係な定数を加えるだけである。

<!--en:3e196cab-->
3. $+0.5\text{ rad/s}$ の衝撃では棒がノミナル軌跡周りの局所線形チューブ内にとどまったが、$+2.2\text{ rad/s}$ ではトルクが飽和する領域まで叩き落とされ、MPC によって再計画される全く新しい複数回スイングのエネルギー注入が必要になるためである。

<!--en:bf971734-->
::: {.slide title="まとめ：古典的最適制御手法"}

<!--en:c5a222f8-->
[要点のまとめ]{.kicker}

<!--en:a026a255-->
| 手法 | ダイナミクスとコストモデル | 方策の形式と能力 |

<!--en:44776645-->
| **有限・無限ホライズン LQR** | 線形 $\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$、二次 $\mathbf{x}^\top\mathbf{Q}\mathbf{x} + \mathbf{u}^\top\mathbf{R}\mathbf{u}$ | 後ろ向きリッカチ漸化式による厳密な線形フィードバック $\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}_t$ |

<!--en:2f6b126b-->
| **局所平衡点 LQR** | ヤコビ行列による線形化 $(\mathbf{A}, \mathbf{B})$（不動点 $(\mathbf{x}^*, u^*)$） | アクチュエータ飽和限界の内側で不安定平衡点を局所安定化 |

<!--en:7c20eacc-->
| **反復 LQR（iLQR）** | ノミナルロールアウトに沿った時変ヤコビ行列 $(\mathbf{A}_t, \mathbf{B}_t)$ | 後ろ向きリッカチ＋直線探索により非線形な複数回スイング軌跡を発見 |

<!--en:405cc2b2-->
| **TV-LQR 追従** | ノミナル $(\bar{\mathbf{x}}_t, \bar{v}_t)$ 周りの線形摂動チューブ | 小さな外乱に対する閉ループフィードバック $v_t = \bar{v}_t + \mathbf{K}_t(\mathbf{x}_t - \bar{\mathbf{x}}_t)$ |

<!--en:43d673c8-->
| **移動ホライズン MPC** | $\mathbf{x}_t$ から有限ホライズン最適化をオンラインで再計算 | 大域的な閉ループ回復を実現し、大きな衝撃後も新たな振り上げを計画 |
