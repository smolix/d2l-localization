<!--en:8fd1c5f0-->
# 確率的勾配降下法とミニバッチ勾配降下法

<!--en:f655d9b5-->
:numref:`sec_gd` では、すべてのパラメータ更新に目的関数の厳密な勾配を用いたため、ステップごとに訓練セット全体にわたって損失を評価する必要があった。深層学習において、フルバッチ勾配降下法は計算コストの面から現実的ではない。ImageNet の $1.28$ 百万枚の画像で ResNet-50 を $90$ エポックにわたり $256$ 枚のミニバッチで訓練すると $450{,}000$ 回の更新を要するが :cite:`Goyal.Dollar.Girshick.ea.2017`、もしフルバッチで $450{,}000$ ステップ実行すればデータセット全体を $450{,}000$ 周走査することになり、$5{,}000$ 倍もの計算量が必要になる。

<!--en:446ef162-->
その代わりに、**確率的勾配降下法**（stochastic gradient descent、SGD）はランダムにサンプリングした 1 つの例、あるいは $b$ 個の例からなる小さな**ミニバッチ**から勾配を推定する（:numref:`subsec_linear-regression-minibatch-sgd`）。この推定量は不偏であり、その分散は $1/b$ に比例して減少するが、サンプリングによるノイズは最適化の終盤の挙動を根本的に変えてしまう。すなわち、一定の学習率 $\eta$ のもとでは、反復点はある 1 点の最小値へ収束せず、定常的な**ノイズ球**（noise ball）の内部で揺らぎ続ける。

<!--en:a96a3133-->
本節では、勾配ノイズ（gradient noise）と不均一な曲率を組み合わせて考察する。:numref:`sec_optimization-intro` の 2 次元の谷を用いて、一定ステップサイズの SGD における各曲率方向の定常分散を導出し、そこから得られる過剰損失（excess loss）の予測を実際の回帰データセットで検証する。さらに、非復元サンプリング（sampling without replacement）とハードウェアの並列性がミニバッチの選択をどのように左右するかを調べ、学習率減衰（:numref:`sec_scheduler`）およびモメンタム（:numref:`sec_momentum`）を導入する動機を明らかにする。

<!--en:726e1d90-->
## 確率的勾配降下法

<!--en:09ceefe5-->
### サンプリングされた勾配と谷

<!--en:70986b6a-->
経験リスク（empirical risk）$f(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n f_i(\mathbf{x})$ が $n$ 個の訓練例に対して定義されるとき、厳密な勾配は例ごとの勾配の平均となる。

<!--en:8670e650-->
1 例ずつ用いる SGD は、各ステップでインデックス $i \in \{1, \ldots, n\}$ を一様ランダムに引き、次のように更新する。

<!--en:6acd284e-->
$\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$ であるため、確率的勾配（stochastic gradient）は $\nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}) + \boldsymbol{\xi}$ と分解できる。ここで、勾配ノイズ $\boldsymbol{\xi}$ の平均は 0 である。

<!--en:baf9cd19-->
$\boldsymbol{\xi}$ が曲率とどのように相互作用するかを確かめるため、まず勾配の両座標に平均 0・分散 1 の独立なガウスノイズを加える `sgd_2d` を定義する。

<!--en:48ddbe2b-->
続いて、$300$ ステップの `sgd_2d` を谷の関数 $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$（:numref:`sec_optimization-intro`）上で、学習率 $\eta = 0.1$、初期点 $(-5, -2)$ から実行する。

<!--en:445a78a4-->
最初の $\sim 100$ ステップでは、決定論的な勾配が支配的となり、反復点は原点へと近づく。原点付近に達すると、ノイズの振幅が一定のままであるのに対し $\|\nabla f(\mathbf{x})\|$ は小さくなるため、反復点は平坦な軸 $x_1$ に沿って引き伸ばされた細長い雲の中に閉じ込められる。

<!--en:1f96e2a1-->
### ノイズ球

<!--en:53bc1bf4-->
この雲の大きさは解析的に計算できる。曲率 $\lambda > 0$ をもつ固有ベクトルの方向に沿って、平均 0 の独立な勾配ノイズ $\xi_t$（分散 $\sigma^2$）が加わるとき、SGD の 1 ステップは次のように時間発展する。

<!--en:270b04d1-->
両辺を二乗して期待値をとると（$\mathbb{E}[x_t \xi_t] = 0$ を用いる）、二次モーメント（second moment）の漸化式が得られる。

<!--en:1bbcbfde-->
任意の安定なステップサイズ $0 < \eta < 2/\lambda$ に対して、平均 $\mathbb{E}[x_t]$ は 0 へ減衰し、二次モーメントは決定論的な縮小とノイズの注入が釣り合う定常分散へと収束する。すなわち $\mathbb{E}[x_\infty^2] = (1 - \eta\lambda)^2 \mathbb{E}[x_\infty^2] + \eta^2 \sigma^2$ より、次式が成り立つ。

<!--en:c8f86ee2-->
:eqref:`eq_sgd-noise-ball` は**ノイズ球**の性質を特徴づけている（有限時間における完全な軌跡の導出は :numref:`subsec_mdl-stochastic-gradients` を参照）。

<!--en:d5b94bb6-->
1. **安定性の上限の維持：**$\eta \to 2/\lambda$ に近づくにつれて、分母は $(2 - \eta\lambda) \to 0$ となり、分散は発散する。

<!--en:130b7049-->
2. **曲率に反比例するスケーリング：**定常分散は曲率 $\lambda$ に反比例する。谷の関数において $\eta = 0.1$、$\sigma^2 = 1$ とすると、定常分散は $0.1 / (0.2 \times 1.98) \approx 0.2525$（平坦な座標 $x_1$ 方向、$\text{std} \approx 0.50$）および $0.1 / (4 \times 1.6) \approx 0.0156$（急峻な座標 $x_2$ 方向、$\text{std} \approx 0.125$）となる。平坦な方向は二重の不利を被る。過渡期には収束が遅く、定常状態では大きく揺らぐからである。

<!--en:fc91a634-->
3. **ステップサイズに対する線形スケーリング：**$\eta\lambda \ll 1$ のとき、学習率 $\eta$ を半分にすると、あらゆる方向で定常分散も半分になる。これは $10{,}000$ 本の独立な軌跡をシミュレーションすることで確認できる。

<!--en:0affc5da-->
### 学習率の減衰

<!--en:0ee29bdd-->
一定ステップサイズの SGD は根本的なトレードオフに直面する。学習率 $\eta$ を小さくすればノイズ球は $\propto \eta$ で縮小するが、1 ステップあたりの縮小率 $\eta\lambda$ が下がるため初期の収束が遅くなる。ノイズフロアを取り除くには、時間とともに学習率 $\eta_t$ を減衰させる必要がある。:citet:`Robbins.Monro.1951` は、勾配の分散が有界である滑らかな強凸目的関数（strongly convex objective）に対し、ステップサイズが次の条件を満たせば SGD が厳密な最小点へ収束することを示した。

<!--en:8f847d8f-->
第 1 の条件は累積ステップ総量が無限大であることを保証し、任意の初期化から最適点へ到達できるようにする。一方、第 2 の条件は注入される累積ノイズ分散 $\propto \sum_t \eta_t^2$ が有限にとどまることを保証する。$\eta_t = c / (t + t_0)$（ただし $c > 1/(2\lambda_{\min})$）のような多項式スケジュールは両方の条件を満たし、期待二乗誤差を $\mathbb{E}[\|\mathbf{x}_t - \mathbf{x}^\star\|^2] = O(1/t)$ のレートで減少させる（:numref:`subsec_mdl-step-size-decay`、演習問題 :exref:`ex_sgd-one-over-t`）。ステップ数 $T$ の固定された訓練予算のもとでは、実用的な深層学習は漸近的な $1/t$ スケジュールに代えて、コサイン、線形、あるいはウォームアップ・安定・減衰（warmup–stable–decay）スケジュールを用いる（:numref:`sec_scheduler`）。

<!--en:a300530f-->
## ミニバッチ

<!--en:1232eb14-->
### 勾配の分散とバッチサイズ

<!--en:d3b8bee2-->
1 つの例を用いる代わりに、ミニバッチ SGD はランダムな部分集合 $\mathcal{B}_t$（$b$ 個の例からなる）にわたって勾配を平均する。

<!--en:179c1b29-->
例ごとの勾配の共分散行列を $\boldsymbol{\Sigma}(\mathbf{x}) = \mathbb{E}_i\big[(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))^\top\big]$ とおく。そのトレース $\operatorname{tr}\boldsymbol{\Sigma} = \mathbb{E}_i\|\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2$ は、例ごとの勾配の分散の総和を表す。独立に復元サンプリングされた $b$ 個の例に対して平均をとると、共分散は $b$ で割られる。

<!--en:cddb18dc-->
$\sigma^2 \leftarrow \sigma^2 / b$ を :eqref:`eq_sgd-noise-ball` に代入すると、曲率 $\lambda$ の方向に沿ったミニバッチ SGD の定常分散が得られる。

<!--en:7feea5a7-->
ステップサイズが小さい領域 $\eta\lambda \ll 1$ では、学習率とバッチサイズはそれらの比 **$\eta / b$** のみを通じて定常的なノイズ球を支配する。すなわち、バッチサイズ $b$ を 2 倍にすることは、ノイズフロアに対して学習率 $\eta$ を半分にするのと同じ効果をもつ。

<!--en:480d07a3-->
### データにおける過剰損失

<!--en:b3dd1a58-->
ノイズ球内部での揺らぎは、訓練損失をその最小値 $f(\mathbf{x}^\star)$ からどれだけ押し上げるだろうか。$k$ 番目の固有ベクトル（ヘッセ行列 $\mathbf{H}$ の固有値 $\lambda_k$、例ごとの勾配の分散 $\sigma_k^2$ をもつ方向）に沿った分散 $v_k \approx \frac{\eta \sigma_k^2}{2 \lambda_k b}$ の定常的な変位は、二次損失の期待値に $\frac{1}{2}\lambda_k v_k$ だけ寄与する。

<!--en:6c3053f2-->
ここで曲率 $\lambda_k$ が約分されて消えることに注目しよう。平坦な方向ほど広い半径にわたって揺らぐが（$\propto 1/\lambda_k$）、平坦な方向への単位変位が損失を押し上げる量は小さい（$\propto \lambda_k$）ためである。直交する $d$ 個すべての固有ベクトルにわたって和をとると（$\sum_{k=1}^d \sigma_k^2 = \operatorname{tr}\boldsymbol{\Sigma}$）、定常状態の**過剰損失**が得られる。

<!--en:9671d8be-->
:eqref:`eq_sgd-excess-loss` を NASA 翼型自己騒音（airfoil self-noise）回帰データセット（$n = 1{,}500$ 件の風洞測定値からなり、標準化された $5$ 個の特徴量と標準化された $1$ 個の音圧ターゲットをもつ）で検証する。

<!--en:ff97bc25-->
二乗損失を用いた線形回帰に対し（バイアスは定数特徴量に対する $6$ 番目の重みとして表す）、厳密なヘッセ行列 $\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$、閉形式の最小点 $\mathbf{w}^\star$、最大曲率 $L = \lambda_{\max}(\mathbf{H}) \approx 2.10$、および例ごとの勾配の分散 $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) \approx 3.24$ を計算する。

<!--en:db43ec2e-->
この予測に対して経験的な過剰損失を測定するため、`sgd_runs` は復元サンプリングによるベクトル化されたミニバッチ SGD の軌跡を `num_runs=400` 本シミュレーションし、`excess_loss` は $\frac{1}{2}(\mathbf{w} - \mathbf{w}^\star)^\top \mathbf{H} (\mathbf{w} - \mathbf{w}^\star)$ を評価する。

<!--en:e0c003da-->
初期点を $\mathbf{w}^\star$、バッチサイズを $b = 10$ として、3 エポック目（$150$ ステップ分）にわたる `excess_loss` の平均を $\eta \in \{0.05, 0.025\}$ のそれぞれで求め、$\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$ と比較する。

<!--en:836db8ca-->
$b = 10$ において、実測された過剰損失は、約 $0.004$（$\eta = 0.05$ のとき）および $0.002$（$\eta = 0.025$ のとき）となり、予測値 $\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$ とよく一致している。ここで、モデルがすべての訓練例に厳密に適合できるだけの表現力をもち、$\nabla f_i(\mathbf{x}^\star) = \mathbf{0}$ がすべての $i$ について成り立つ**補間レジーム**（interpolating regime）では何が起きるかに注目しよう。この極限では $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) = 0$ となり、反復点が $\mathbf{x}^\star$ に近づくにつれてノイズ球は消失し、一定ステップサイズの SGD は厳密な最小点へと幾何級数的に収束する。多数のエポックにわたって訓練される過剰パラメータ化された分類器はこの補間レジームに近づく一方、巨大なコーパスを 1 パスだけ走査して訓練される大規模言語モデルは、$\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) > 0$ となるノイズ制限レジーム（noise-limited regime）にとどまる。

<!--en:197bb3a8-->
### 非復元サンプリング

<!--en:48966da6-->
実際には、データローダはミニバッチを独立に復元サンプリングするのではなく、各エポックの開始時にデータセットをランダムに並べ替え、互いに素なサイズ $b$ のミニバッチへと分割する。$b$ 個の異なる例をサイズ $n$ の有限母集団から非復元サンプリングすると、例ごとの勾配誤差の間に負の相関が生じる（$\sum_{i=1}^n (\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})) = \mathbf{0}$ であるため）。その結果、ミニバッチ勾配の分散 :eqref:`eq_sgd-variance` には有限母集団修正係数（finite-population correction factor）が掛かる（演習問題 :exref:`ex_sgd-finite-population`）。

<!--en:d40c0ffd-->
$b \ll n$ のとき（$10$ 個を $1{,}500$ 個から選ぶ場合）、この係数はほぼ 1 に等しく（$0.994$）、$b = n$ のときはフルバッチ勾配降下法で要請されるとおり分散が 0 になる。さらに 1 エポック全体を通してみると、ランダムな再シャッフル（random reshuffling）によってエポック内の $n/b$ ステップにわたるミニバッチ勾配誤差が 1 次のオーダーで打ち消し合うため、学習率が小さいときには復元サンプリングよりも定常誤差が小さくなる :cite:`Mishchenko.Khaled.Richtarik.2020`。

<!--en:beedd71c-->
### バッチサイズとハードウェアのスループット

<!--en:9ac2de5a-->
$\eta\lambda \ll 1$ であり、かつ $\eta / b$ を一定に保つとき、サイズ $b$ のミニバッチに対して学習率 $\eta$ で 1 ステップ更新すると、期待誤差は $(1 - \eta\lambda)$ 倍に縮小し、1 例ずつ $b$ ステップを学習率 $\eta / b$ で進めた場合と同じノイズ球に到達する。これは $(1 - \eta\lambda / b)^b \approx 1 - \eta\lambda$ が成り立つためである。例ごとの勾配評価の回数だけで数えるなら、この線形レジームの範囲内で $b$ を大きくしても、学習は速くも遅くもならない。

<!--en:d3283d1c-->
ミニバッチを不可欠なものにしているのは、**ハードウェアによる償却**（hardware amortization）である。各パラメータ更新には $b$ に依存しない一定の Python ディスパッチおよびカーネル起動のオーバーヘッドが生じる上、ベクトルプロセッサ（CPU や GPU）はモデルの重みをメモリから 1 度読み出すだけで $b$ 個の例を並列に評価できる（:numref:`chap_performance`）。翼型データセットにおいて、1 エポックあたりの実経過時間と処理した例数に対する損失の両方をベンチマークするため、`train_sgd` は $\mathbf{w} = \mathbf{0}$ から線形回帰に対してミニバッチ SGD を実行し、更新ごとのデータセット全体の損失と 1 エポックあたりの秒数を返す。

<!--en:6ab9e743-->
それでは、`train_sgd` を $b \in \{1, 10, 100\}$ に対して一定の比 $\eta / b = 0.005$ で実行し、あわせてフルバッチ勾配降下法（$b = 1{,}500$、$\eta = 0.9 < 2/L \approx 0.95$）も実行しよう。

<!--en:70bea9ba-->
処理した総例数（$b \times \text{updates}$）に対して訓練損失をプロットすると、4 つのバッチサイズすべてのサンプル効率を重ねて比較できる。

<!--en:3d53bde8-->
$b = 100$ 個の例に対する 1 回の更新は、$b = 1$ での更新とほぼ同じ実経過時間で完了するため、$b = 100$ の 1 エポックは約 $50\text{--}100\times$ 高速になりながら、処理した例数あたりではまったく同じ損失曲線をたどる。同時に、線形スケーリング $\eta \propto b$ がどこで破綻するかにも注目しよう。$\eta / b = 0.005$ を保ったまま $b = 100$ にすると学習率は $\eta = 0.5$ まで上がり、$\eta L = 0.5 \times 2.10 = 1.05$ となって $2$ と比べて十分に小さいとは言えなくなる（その結果、:eqref:`eq_sgd-noise-ball` の分母の逆数 $(2 - \eta\lambda)^{-1}$ が増大する）。さらに $b \approx 190$ を超えるバッチサイズでは、安定性の上限 $\eta < 2/L \approx 0.95$ に違反してしまう。この臨界バッチサイズ（critical batch size）の遷移については、:numref:`sec_batch_size` で体系的に解析する。

<!--en:30ac03ff-->
## まとめ

<!--en:73a77c60-->
ミニバッチ SGD は $b$ 個の例から勾配を推定し、サンプリングによる分散を $\operatorname{tr}(\boldsymbol{\Sigma})/b$（非復元サンプリングではこれに $(n-b)/(n-1)$ を掛けた値）まで低減しつつ、ステップごとのディスパッチおよびメモリ帯域幅のコストをミニバッチ全体で償却する。一定の学習率 $\eta$ のもとでは、反復点は定常的な**ノイズ球**へと収束し、曲率 $\lambda$ の方向に沿った座標分散は $\frac{\eta\sigma^2}{b\lambda(2 - \eta\lambda)} \approx \frac{\eta\sigma^2}{2\lambda b}$ となって最も平坦な方向で最も広くなる。全モードにわたって二次損失への寄与を足し合わせると $\lambda$ が相殺され、定常状態の期待**過剰損失** $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ が残る。この過剰損失が 0 になるのは、最適点においてモデルがすべての訓練例を補間する場合に限られる。$\eta L \ll 1$ が成り立つ限り、1 例あたりの過渡的な進捗と定常ノイズフロアはいずれも $\eta$ と $b$ にその比 $\eta/b$ のみを通じて依存するが、$b$ を $\eta \propto b$ に従って増やし続けると、やがて決定論的な曲率の上限 $\eta < 2/L$ に近づきこの性質は成り立たなくなる。

<!--en:2294243a-->
## 演習問題

<!--en:62bf73a2-->
1. **ランダムなデータセットとしてのノイズ。**分散 1 のガウス勾配ノイズを伴う `sgd_2d` を $f(\mathbf{x}) = 0.1 x_1^2 + 2x_2^2$ 上で実行することは、$f(\mathbf{x}, \mathbf{w}) = 0.1 (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$（ただし $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{C})$）上で 1 例ずつの SGD を実行することと等価であることを示し、共分散 $\mathbf{C}$ を求めよ。
   :exlabel:`ex_sgd-noise-dataset`

<!--en:7a1d1555-->
1. [code] **$1/t$ スケジュールの定数。** ● 曲率 $\lambda$ の 1 次元二次関数に対し、分散 $\sigma^2$ の勾配ノイズと学習率 $\eta_t = c / (t + t_0)$ のもとで以下を行え。
   :exlabel:`ex_sgd-one-over-t`
    1. $e_t = \mathbb{E}[x_t^2]$ に対する厳密な漸化式を書き下し、漸近的な仮定形 $e_t \approx A/t$ を代入することで、$A = c^2 \sigma^2 / (2 c \lambda - 1)$ が（$c > 1/(2\lambda)$ のとき常に）成り立つことを示せ。どの $c$ の値が $A$ を最小化するか。
    1. $e_t$ の決定論的な漸化式を $\lambda = 0.2, \sigma = 1$、初期値 $e_0 = 25$、$t_0 = 20$、$c \in \{1, 2.5, 5, 10\}$ として $t = 10^6$ まで反復計算し、両対数軸上で $t\,e_t$ を $t$ に対してプロットせよ。さらに、単一の $c$ では $\lambda_1 = 0.2$ と $\lambda_2 = 4$ の両方に対して同時に最適になり得ない理由を説明せよ。

<!--en:f290484c-->
1. [code] **有限母集団係数と再シャッフル。**
    1. $\mathbf{e}_i = \nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})$ とおく。$\sum_{i=1}^n \mathbf{e}_i = \mathbf{0}$ を用いて $\mathbb{E}_{i \neq j}[\mathbf{e}_i^\top \mathbf{e}_j] = -\operatorname{tr}\boldsymbol{\Sigma}/(n - 1)$ を示し、:eqref:`eq_sgd-finite-population` を導出せよ。さらに翼型データセットの例ごとの勾配 `G` を用い、$b \in \{10, 100, 500, 1000\}$ を復元サンプリングおよび非復元サンプリングした場合について、この公式を実験的に検証せよ。
       :exlabel:`ex_sgd-finite-population`
    1. $n = 100$ 個のベクトル $\mathbf{w}_i$ を演習問題 :exref:`ex_sgd-noise-dataset` の分布から生成し、復元サンプリングとエポックごとの再シャッフルによる 20 エポックの SGD の軌跡を $\eta \in \{0.1, 0.01\}$ で比較せよ。
       :exlabel:`ex_sgd-reshuffling`

<!--en:2ed4333a-->
1. [code] **訓練中の勾配の分散。** ●
   :exlabel:`ex_sgd-variance-training`
    1. 翼型の線形モデルにおいて、$\operatorname{tr}\boldsymbol{\Sigma}$ の値を $\mathbf{w} = \mathbf{0}$ と $\mathbf{w} = \mathbf{w}^\star$ で比較せよ。
    1. 2 層の ReLU MLP（隠れユニット数 $64$）を翼型データで $10$ エポック（$b = 10, \eta = 0.05$）訓練し、$\operatorname{tr}\boldsymbol{\Sigma}$（`W1`、`b1`、`W2`、`b2` ごとの内訳）を初期化時点と訓練後とで比較せよ。

<!--en:9477e62a-->
1. [code] **バッチサイズと学習率のスケーリング。** ● 翼型の線形回帰モデルにおいて $\eta = 0.05$（$b = 10$ のとき）を基準に固定し、$b \in \{1, 10, 100, 1500\}$ を線形スケーリング（$\eta \propto b$）と平方根スケーリング（$\eta \propto \sqrt{b}$）の両方で試せ。$L = 2.10$ を用いてどの設定が発散するかを説明し、さらに `H` の全 $6$ 個の固有ベクトルにわたる厳密な定常過剰損失を :eqref:`eq_sgd-noise-ball` から計算して、$b = 100, \eta = 0.5$ のフロアが $b = 10, \eta = 0.05$ よりもわずかに高くなる理由を説明せよ。
   :exlabel:`ex_sgd-batch-lr`

<!--en:4f71e3ad-->
[ディスカッション](https://d2l.discourse.group/t/497)

<!--en:57b439c6-->
[ディスカッション](https://d2l.discourse.group/t/1067)

<!--en:b0d9b1a2-->
確率的勾配降下法とミニバッチ勾配降下法<br>**サンプリングされた勾配・定常ノイズ球・Robbins–Monro 減衰・ミニバッチ分散・過剰損失・ハードウェアのスループット**

<!--en:f7d80d4e-->
::: {.slide title="なぜフルバッチ勾配降下法は非現実的か"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:ecfa25ed-->
- **フルバッチのコスト爆発：**$\nabla f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x})$ を ImageNet（$n = 1.28\text{M}$）上で $450{,}000$ 回の更新にわたり評価するには、データセット全体の走査が $450{,}000$ 回必要になる（$5{,}000\times$ もの計算量、対 $90$ エポック、$b = 256$）

<!--en:62baf0b6-->
- **ミニバッチの効率（$b = 256$）：**1 ステップあたり $1/5{,}000$ のコストで降下方向を推定

<!--en:bb8d9199-->
- **根本的なトレードオフ：**$b \ll n$ 個の例をサンプリングすると**勾配ノイズ**が注入され、一定学習率の反復点は定常的なノイズ球に閉じ込められる

<!--en:4d9a9224-->
::: {.slide title="不偏なサンプリング勾配とノイズ分解"}

<!--en:46ec1b9c-->
[確率的勾配]{.kicker}

<!--en:b20edfdf-->
- **不偏推定量：**$i \sim \text{Uniform}\{1, \ldots, n\}$ を引くと $\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$ が成り立つ

<!--en:4961d2a9-->
- **信号とノイズへの分解：**
  $$\nabla f_i(\mathbf{x}_t) = \nabla f(\mathbf{x}_t) + \boldsymbol{\xi}_t, \qquad \mathbb{E}[\boldsymbol{\xi}_t \mid \mathbf{x}_t] = \mathbf{0}$$

<!--en:36725e0a-->
- **$\mathbf{x}^\star$ でも消えないノイズ：**$\nabla f(\mathbf{x}^\star) = \mathbf{0}$ の点であっても、例どうしで最適点が食い違えば個々の勾配は $\nabla f_i(\mathbf{x}^\star) \neq \mathbf{0}$ となり、一定学習率の反復点を絶えず $\mathbf{x}^\star$ から弾き出す

<!--en:9e8359a3-->
::: {.slide title="谷における SGD：過渡的な降下からノイズの雲へ"}

<!--en:b6f5ea38-->
[谷のシミュレーション]{.kicker}

<!--en:d2043201-->
- 分散 1 の独立なガウスノイズ $\xi_1, \xi_2 \sim \mathcal{N}(0, 1)$ を谷の関数 $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$（$\lambda_1 = 0.2, \lambda_2 = 4.0$）に $\eta = 0.1$ で $300$ ステップ加えると、2 つの段階が現れる
  1. **過渡段階（ステップ 1〜100）：**決定論的な勾配 $\|\nabla f(\mathbf{x})\|$ が $\boldsymbol{\xi}$ を支配し、反復点を $(-5, -2)$ から $(0, 0)$ へと駆動する
  2. **定常段階（ステップ 100 以降）：**原点付近で $\|\nabla f(\mathbf{x})\|$ がノイズレベルまで縮小し、反復点は平坦な軸 $x_1$ に沿って伸びた細長い雲に閉じ込められる

<!--en:267e1984-->
::: {.slide title="ノイズ球の定常分散の導出"}

<!--en:d187213b-->
[二次モーメントの漸化式]{.kicker}

<!--en:8dfc6501-->
- **1 次元のノイズ付き時間発展（$x_{t+1} = (1 - \eta\lambda)x_t - \eta\xi_t$）：**二乗して期待値をとると（$\mathbb{E}[x_t\xi_t] = 0$）、決定論的な縮小とノイズ注入が分離される
  $$\mathbb{E}\big[x_{t+1}^2\big] = (1 - \eta\lambda)^2\,\mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2$$

<!--en:d0335cd1-->
- **定常状態の釣り合い（$0 < \eta < 2/\lambda$）：**$\mathbb{E}[x_{t+1}^2] = \mathbb{E}[x_t^2] = \mathbb{E}[x_\infty^2]$ とおくと**ノイズ球**が得られる
  $$\mathbb{E}\big[x_\infty^2\big] = \frac{\eta\,\sigma^2}{\lambda\,(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{2\lambda}$$

<!--en:0f26254b-->
::: {.slide title="定常ノイズ球を支配する 3 つの法則"}

<!--en:f3086122-->
[幾何学的解釈]{.kicker}

<!--en:ebec19d4-->
- **1. 安定性の上限の維持：**$(2 - \eta\lambda) \to 0$（$\eta \to 2/\lambda$ のとき）となるため、決定論的な限界で分散が発散する（谷の関数では $\eta < 0.5$）

<!--en:df1a209b-->
- **2. 曲率に反比例するスケーリング（$\propto 1/\lambda$）：**平坦な方向は過渡的な減衰が遅く（$\eta\lambda$）、定常状態の揺らぎも大きい（分散は $16\times$：$\lambda_1 = 0.2$ 方向と $\lambda_2 = 4.0$ 方向の比）

<!--en:48aa7909-->
- **3. $\eta/b$ に対する線形スケーリング：**$\eta\lambda \ll 1$ のとき定常分散は $\frac{\eta\sigma^2}{2\lambda b}$ に比例し、$\eta$ を半減するか $b$ を倍増するとノイズ球は半分になる

<!--en:bfa50525-->
::: {.slide title="10,000 回のシミュレーションによるノイズ球の公式の確認"}

<!--en:63718f6d-->
[実験的検証]{.kicker}

<!--en:2c6eef42-->
- $10{,}000$ 本の独立な SGD の軌跡を谷の関数（$\lambda_1 = 0.2, \lambda_2 = 4.0, \sigma^2 = 1$）上で $2{,}000$ ステップにわたり、$\eta = 0.1$ と $\eta = 0.05$ でシミュレーションする
  - $\eta = 0.1$ のとき：実測された分散は $\frac{\eta}{\lambda(2-\eta\lambda)} = (0.2525, 0.0156)$ と $1\%$ 以内で一致
  - $\eta = 0.05$ のとき：平坦な方向の分散は約半分の $0.1256$ になる一方、急峻な方向の分散は半分強まで低下する（$0.0156 \to 0.0069$。$2 - \eta\lambda_2$ が $1.6 \to 1.8$ に増加するため）

<!--en:b85b14de-->
::: {.slide title="ノイズフロアの除去：Robbins–Monro 条件"}

<!--en:1ce1ae4f-->
[減衰するステップサイズ]{.kicker}

<!--en:ca14110b-->
- **Robbins–Monro 条件（1951）：**ノイズ下での厳密な収束 $\mathbf{x}_t \to \mathbf{x}^\star$ のための条件
  $$\sum_{t=1}^{\infty} \eta_t = \infty \qquad \text{and} \qquad \sum_{t=1}^{\infty} \eta_t^2 < \infty$$

<!--en:95f182c6-->
- **無限の移動総量（$\sum \eta_t = \infty$）：**任意の初期化から $\mathbf{x}^\star$ に到達可能（幾何減衰 $\eta_0\gamma^t$ は早期に停止し、この条件に違反）

<!--en:fc1a40a4-->
- **有限の累積ノイズ（$\sum \eta_t^2 < \infty$）：**終盤の分散を 0 に縮小（$\eta_t = \frac{c}{t+t_0}$ により平均二乗誤差が $O(1/t)$ で減衰し達成）

<!--en:f20670c4-->
::: {.slide title="ミニバッチ平均と 1/b 分散低減則"}

<!--en:9c96161c-->
[ミニバッチ SGD]{.kicker}

<!--en:198e994a-->
ミニバッチ SGD は、$b$ 個の例からなる集合 $\mathcal{B}_t$（ステップ $t$ で抽出）にわたって勾配を平均する。

<!--en:147ba75f-->
- **例ごとの勾配の共分散：**$\boldsymbol{\Sigma}(\mathbf{x}) = \mathbb{E}_i\big[(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))^\top\big]$ とおき、全分散を $\operatorname{tr}\boldsymbol{\Sigma} = \mathbb{E}_i\|\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2$ とする

<!--en:9688de18-->
- **$1/b$ でスケーリングする分散：**独立に復元サンプリングした $b$ 個の例では、共分散が $b$ で割られる
  $$\mathbb{E}\big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}_t)}{b}$$

<!--en:d8e155e8-->
- **$\eta / b$ に対するノイズ球の不変性：**ノイズ球の式に $\sigma^2 \leftarrow \sigma^2 / b$ を代入すると $\mathbb{E}[x_\infty^2] \approx \frac{\eta}{b}\cdot\frac{\sigma^2}{2\lambda}$（$\eta\lambda \ll 1$ のとき）となり、**$b$ の倍増と $\eta$ の半減はまったく同じ定常ノイズ球をもたらす**

<!--en:87dbd50d-->
::: {.slide title="定常過剰損失：なぜ曲率が相殺されるか"}

<!--en:423e5cf1-->
[訓練が停滞する水準]{.kicker}

<!--en:7f25f273-->
- **座標の分散と損失の曲率：**固有ベクトル $k$（曲率 $\lambda_k$、ノイズ分散 $\sigma_k^2$）に沿ったパラメータの定常分散は $v_k \approx \frac{\eta\sigma_k^2}{2\lambda_k b}$

<!--en:50224b03-->
- **$\lambda_k$ の厳密な相殺：**二次損失で $\frac{1}{2}\lambda_k$ を掛けるとすべてのモードで $\lambda_k$ が約分され、$\frac{1}{2}\lambda_k v_k = \frac{\eta\sigma_k^2}{4b}$ となる

<!--en:015bf6a1-->
- **曲率に依存しない総過剰損失：**全 $d$ モードにわたって和をとると（$\sum_k \sigma_k^2 = \operatorname{tr}\boldsymbol{\Sigma}$）、次式を得る
  $$\mathbb{E}\big[f(\mathbf{x}_\infty)\big] - f(\mathbf{x}^\star) \approx \frac{\eta\,\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$$

<!--en:ffc1f205-->
::: {.slide title="NASA 翼型データセットにおける過剰損失の検証"}

<!--en:c4757c7b-->
[実データでの厳密な計測]{.kicker}

<!--en:c6d27dc6-->
- 標準化された NASA 翼型自己騒音データセット（$n = 1{,}500$ 件の風洞実験例、特徴量 $5$ 個 $+ 1$ バイアス $= 6$ 個の重み）では、二乗損失の線形回帰に対して厳密な閉形式のヘッセ行列と最小点 $\mathbf{w}^\star$ が求まる
  $$\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$$

<!--en:df5be1be-->
- 最大固有値は $L = \lambda_{\max}(\mathbf{H}) \approx 2.10$ であり、全 $1{,}500$ 件の例ごとの勾配 $\mathbf{g}_i = (\mathbf{x}_i^\top\mathbf{w}^\star - y_i)\mathbf{x}_i$（$\mathbf{w}^\star$ における値）を評価すると $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) \approx 3.24$ となる

<!--en:4d263c37-->
::: {.slide title="実データでの過剰損失の実測値と η tr Σ / (4b) の一致"}

<!--en:1915288f-->
[理論と実験]{.kicker}

<!--en:42cb59b7-->
- バッチサイズ $b = 10$、$\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) = 3.24$ のとき、公式による予測は次のとおり
  - $\eta = 0.05$ のとき：$\frac{0.05 \times 3.24}{4 \times 10} \approx 0.0041$ だけ $f(\mathbf{w}^\star) \approx 0.242$ を上回る
  - $\eta = 0.025$ のとき：$\frac{0.025 \times 3.24}{4 \times 10} \approx 0.0020$

<!--en:6ae7b80d-->
- 独立な $400$ 回のミニバッチ SGD 実行について 3 エポック目にわたり平均をとると、$2 \times 10^{-4}$ 以内で一致することを確認

<!--en:c57986b6-->
::: {.slide title="補間レジームとノイズ制限レジーム"}

<!--en:e1123082-->
[ノイズ球はいつ消失するか]{.kicker}

<!--en:85d7b249-->
過剰損失の公式 $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ から、各訓練例が $\mathbf{x}^\star$ で一致するかどうかに応じて、根本的に異なる 2 つの訓練レジームが浮かび上がる。

<!--en:bf8da6ba-->
| 訓練レジーム | $\mathbf{x}^\star$ における例ごとの勾配 | ノイズのトレース $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)$ | 一定学習率 SGD の挙動 | 深層学習の例 |

<!--en:6dc0a6f1-->
| **過少パラメータ・ノイズラベルレジーム** | 例ごとに不一致（$\nabla f_i(\mathbf{x}^\star) \neq \mathbf{0}$、平均のみ $\frac{1}{n}\sum_i \nabla f_i(\mathbf{x}^\star) = \mathbf{0}$） | $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) > 0$ | 過剰損失 $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ で停滞、**学習率減衰 $\eta_t \to 0$ が必要** | 翼型回帰、巨大コーパス上での 1 パス LLM 事前学習 |

<!--en:e7b9fa1b-->
| **補間レジーム** | モデルが全訓練例に厳密に適合（$\nabla f_i(\mathbf{x}^\star) = \mathbf{0}$ が全 $i$ で成立） | $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) = 0$ | $\mathbf{x}_t \to \mathbf{x}^\star$ に伴いノイズ球が自動的に縮小、**幾何級数的に収束** | 損失 0 まで多エポック訓練された過剰パラメータ分類器 |

<!--en:a7f174fd-->
::: {.slide title="非復元サンプリングとランダムな再シャッフル"}

<!--en:dfb30e84-->
[実用的なデータローダ]{.kicker}

<!--en:ac1fa555-->
実際のデータローダはミニバッチを復元サンプリングするのではなく、毎エポック $n$ 個の例をランダムに並べ替え、互いに素なサイズ $b$ のミニバッチにスライスする。

<!--en:e094aaa5-->
- **有限母集団の分散修正：**$n$ 個の例ごとの勾配誤差の和は 0 であるため（$\sum_{i=1}^n (\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})) = \mathbf{0}$）、ミニバッチ内の異なる例どうしはわずかに負の相関をもち（$\mathbb{E}_{i \neq j}[\mathbf{e}_i^\top \mathbf{e}_j] = -\frac{\operatorname{tr}\boldsymbol{\Sigma}}{n-1}$）、分散は次のようにスケーリングされる
  $$\mathbb{E}\big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}\cdot\frac{n - b}{n - 1}$$

<!--en:db5bdb52-->
- **極限の確認：**$b = 10, n = 1{,}500$ では修正係数は $0.994 \approx 1$ となり、$b = n$ ではフルバッチ勾配降下法で要請されるとおり分散が消失（$0$）

<!--en:107558b7-->
- **エポック単位の相殺：**$n/b$ ステップからなる 1 パス全体では、ランダムな再シャッフルにより 1 次の勾配誤差のエポック合計が 0 となり、小さい $\eta$ において定常ノイズ球がさらに縮小する（Mishchenko et al., 2020）

<!--en:d3382b4a-->
::: {.slide title="ハードウェアによる償却：100 例でも 1 例と同じコスト"}

<!--en:c3911eda-->
[スループットのベンチマーク]{.kicker}

<!--en:90a213a1-->
- オプティマイザの各更新では、Python ディスパッチ、自動微分グラフ、カーネル起動の固定オーバーヘッドに加え、重みのメモリ読み出しが $b$ に無関係に発生する

<!--en:af3a9895-->
- $1{,}500$ 例の翼型データセットで $\eta / b = 0.005$ を固定した場合：
  - $b = 1$（$1{,}500$ 回更新/エポック）と $b = 100$（$15$ 回更新/エポック）はいずれも*更新あたり*約 1 ミリ秒であり、$b = 100$ は**エポックあたり $50\text{--}100\times$ 高速**になる

<!--en:7298dff2-->
::: {.slide title="サンプル効率と線形スケーリング η ∝ b の破綻点"}

<!--en:f1f4a44f-->
[1 例あたりの進捗]{.kicker}

<!--en:ac28137c-->
- **小バッチの曲線が重なる理由：**$\eta / b = 0.005$ を固定し $\eta\lambda \ll 1$ とするとき、$b$ ステップ（学習率 $\eta/b$）での縮小率は $(1 - \eta\lambda/b)^b \approx 1 - \eta\lambda$ となり、処理例数あたりではステップサイズ $\eta$ の 1 ステップと一致する

<!--en:bc88d1be-->
- **曲率の上限 $2/L$ 付近での破綻：**
  - $b = 100$ では $\eta = 0.5$ から $\eta L = 0.5 \times 2.10 = 1.05$ となり、ノイズ球の分母の逆数 $(2 - \eta L)^{-1}$ が増大して最終エポックの損失がわずかに上昇する（$0.248$ 対 $0.246$）
  - $b_{\max} = \frac{2/L}{0.005} \approx 190$ を超えると、線形スケーリングは $\eta < 2/L \approx 0.95$ に違反して発散する

<!--en:de4be7ab-->
::: {.slide title="概念チェック：ノイズ球・過剰損失・バッチサイズ"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:fdbfa154-->
各ミニバッチ SGD のシナリオにおける結果を計算せよ。

<!--en:31577dd3-->
1. **バッチサイズの 4 倍化：**小ステップ領域（$\eta\lambda \ll 1$）において、バッチサイズを $b = 16$ から $b = 64$ に増やし $\eta$ を固定したままにすると、定常過剰損失は何倍に変化するか。同時に $\eta$ も $4\times$ にスケーリングした場合はどうなるか。

<!--en:f0898de7-->
2. **異方的なノイズ球と等方的な過剰損失：**谷の関数（$\lambda_1 = 0.2, \lambda_2 = 4.0$）で勾配ノイズの分散が等しい（$\sigma_1^2 = \sigma_2^2 = 1$）とき、定常的な座標分散の比 $\mathbb{E}[x_{\infty,1}^2] / \mathbb{E}[x_{\infty,2}^2]$（小さい $\eta$ における値）はいくらか。また、それぞれの過剰損失への寄与の比はいくらか。

<!--en:9fba46c1-->
3. **線形スケーリングの上限：**損失ランドスケープの最大曲率が $L = 5.0$ であり、$\eta = 0.02$ を $b = 8$ で調整した（$\eta/b = 0.0025$）とき、バッチサイズ $b$ がいくらになると線形スケーリング $\eta = 0.0025\,b$ は決定論的な安定性限界 $2/L$ に達するか。

<!--en:7715016c-->
::: {.slide title="概念チェック：ノイズ球・過剰損失・バッチサイズ（解答）"}

<!--en:929b7671-2-->
[理解度チェック]{.kicker}

<!--en:557b98b6-->
**解答：**

<!--en:3122394b-->
1. $\eta$ を固定すると過剰損失 $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}}{4b}$ は **$4\times$** の比率で減少し、$\eta$ も $4\times$ にスケーリングすると $\eta / b$ が一定に保たれるため過剰損失は**変化しない**（ただしフロア到達までの更新回数は $4\times$ 削減される）

<!--en:e619d524-->
2. 座標分散は $1/\lambda$ に比例するため、$\mathbb{E}[x_{\infty,1}^2]/\mathbb{E}[x_{\infty,2}^2] \approx 4.0 / 0.2 =$ **$20\times$ 広く**（$x_1$ 方向）、過剰損失 $\frac{1}{2}\lambda_k \mathbb{E}[x_{\infty,k}^2] \approx \frac{\eta\sigma_k^2}{4b}$ では $\lambda_k$ が相殺されて両軸が**等しく（$1:1$）**寄与する

<!--en:4cecfc85-->
3. 安定性限界は $2/L = 2/5.0 = 0.4$ であり、線形スケーリングは **$b = 0.4 / 0.0025 = 160$ 例**でこれに到達する

<!--en:faf2dff9-->
::: {.slide title="まとめ：確率的勾配降下法とミニバッチ勾配降下法"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:65666ad7-->
| 量・法則 | 数式 | スケーリング依存性 | 実務上の意味 |

<!--en:c779ed4a-->
| **ミニバッチ勾配の分散** | $\mathbb{E}\|\mathbf{g}_t - \nabla f\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}\frac{n-b}{n-1}$ | 分散 $\propto 1/b$、標準偏差 $\propto 1/\sqrt{b}$ | $100\times$ 大きいバッチで勾配誤差の標準偏差を $10\times$ 削減 |

<!--en:190a72fa-->
| **定常ノイズ球** | $\mathbb{E}[x_\infty^2] = \frac{\eta\sigma^2}{b\lambda(2 - \eta\lambda)} \approx \frac{\eta\sigma^2}{2\lambda b}$ | 比 $\eta/b$ に比例し、曲率 $\lambda$ に反比例 | 平坦な方向ほど最も広いパラメータ半径にわたり揺らぐ |

<!--en:4f616e47-->
| **定常過剰損失** | $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ | 曲率 $\lambda_k$ が相殺、$\eta/b$ と $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)$ に比例 | 補間モデル（$\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)=0$）では消失、それ以外は学習率減衰が必要 |

<!--en:fb727913-->
| **Robbins–Monro 条件** | $\sum_t \eta_t = \infty,\; \sum_t \eta_t^2 < \infty$ | $\eta_t = c/(t+t_0)$ で $\mathbb{E}\|\mathbf{x}_t - \mathbf{x}^\star\|^2 = O(1/t)$ | 無限の移動総量と有限の累積ノイズ分散 |

<!--en:1bb7e23e-->
| **線形スケーリング領域** | $\eta(b) \propto b$（$\eta L \ll 2$） | 一定の $\eta/b$ で 1 例あたり同じ進捗・同じノイズフロア | $b_{\text{crit}}$ まで並列ハードウェアでステップごとのオーバーヘッドを償却（§{sec_batch_size}） |
