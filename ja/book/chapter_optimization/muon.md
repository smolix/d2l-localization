<!--en:ec4f9b5c-->
# Muon

<!--en:2de61e91-->
これまでに見てきたすべてのオプティマイザ（SGD、モメンタム、AdamW）は、ニューラルネットワークのパラメータを構造のない平坦なベクトルとして扱っている。AdamW でさえ、2 つのスカラー平均 $(m_{t,ij}, v_{t,ij})$ を各要素 $W_{ij}$（重み行列 $\mathbf{W}$ の成分）に対して独立に保持しているにすぎない。仮に $\mathbf{W}$ の要素を任意に並べ替えたとしても、AdamW はまったく同じ更新ステップを踏むことになる。

<!--en:5e701ddc-->
しかし、Transformer のパラメータの $80\%$ 以上は 2 次元の隠れ射影行列 $\mathbf{W} \in \mathbb{R}^{m \times n}$ にあり、これらの行列は層の活性化値 $\mathbf{x} \in \mathbb{R}^n$ を $\mathbf{y} = \mathbf{W}\mathbf{x} \in \mathbb{R}^m$ へ写す線形作用素として働く。重み行列を*線形作用素とみなしたとき*、その最急降下（steepest descent）の方向はどうなるだろうか。

<!--en:fa3eee24-->
:citet:`Bernstein.Newhouse.2024` が強調するように、最急降下は更新ステップを抑えるために選ぶノルムに対して相対的に定義される。ステップをユークリッド（$\ell_2$）ノルムで測れば SGD の生の勾配方向が得られ、座標ごとの $\ell_\infty$ ボックスノルムで測れば Adam の記憶なし極限である符号降下法（sign descent）が得られる。その代わりに、更新 $\Delta\mathbf{W}$ が層の出力にもたらし得る最大変化量 $\|\Delta\mathbf{W}\mathbf{x}\|_2 / \|\mathbf{x}\|_2$、すなわち**スペクトルノルム** $\|\Delta\mathbf{W}\|_2$ を抑えることにすると、最急降下方向は勾配行列のすべての非ゼロ特異値を $1$ に置き換えたものとなり、**直交化勾配**（orthogonalized gradient）$\mathbf{U}\mathbf{V}^\top$ が得られる。

<!--en:7de100ae-->
各ステップで厳密な特異値分解（singular value decomposition、SVD）を計算するのは GPU にとって遅すぎるが、5 ステップの奇の行列多項式（**ニュートン・シュルツ反復**、Newton–Schulz iteration）を用いれば、行列積のみを用いて $\mathbf{U}\mathbf{V}^\top$ を近似できる。これを隠れ層の 2 次元重み行列に対するネステロフのモメンタムバッファに適用し（埋め込み、1 次元パラメータ、出力ヘッドは AdamW に任せる）、得られるのが **Muon**（*Momentum Orthogonalized by Newton–Schulz*、:citet:`Jordan.Jin.Boza.ea.2024`）である。本節では、ユークリッドノルム球、$\ell_\infty$ ノルム球、スペクトルノルム球における最急降下を導出し、調整済みの 5 次ニュートン・シュルツ反復を解析したうえで、ハイブリッドな Muon+AdamW オプティマイザをゼロから実装し、`TinyLM` 上で AdamW とベンチマーク比較する。

<!--en:d6c2575c-->
## ノルムと最急降下

<!--en:0db74f83-->
### ノルム球における最急降下

<!--en:abd395cc-->
1 次テイラー展開 $f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) + \langle \mathbf{g}, \mathbf{d} \rangle$（ただし勾配は $\mathbf{g} = \nabla f(\mathbf{x})$）が与えられたとき、半径 $\eta > 0$、ノルム $\|\cdot\|$ のもとでの最急降下ステップは次の最適化問題を解くことで得られる。

<!--en:a2e52008-->
ノルムの幾何構造が異なれば、まったく異なる更新方向が導かれる :cite:`Bernstein.Newhouse.2024`（:numref:`fig_opt_norm_balls` および :numref:`tab_muon_norms`）。

<!--en:ec32b180-->
1. **ユークリッド球（$\|\mathbf{d}\|_2 \le \eta$）：**コーシー・シュワルツの不等式により、最小点は負の勾配と同じ方向を向き、$\mathbf{d}^\star = -\eta\,\mathbf{g}/\|\mathbf{g}\|_2$ となって線形化された目的関数を $\eta\|\mathbf{g}\|_2$ だけ減少させる（SGD の方向）。

<!--en:f5ff7920-->
2. **座標ボックス（$\|\mathbf{d}\|_\infty = \max_i |d_i| \le \eta$）：**$\sum_i g_i d_i$ を制約 $|d_i| \le \eta$ のもとで最小化すると、すべての座標がその境界まで押し出されて $\mathbf{d}^\star = -\eta\,\operatorname{sign}(\mathbf{g})$ となり、線形化された目的関数は双対な $\ell_1$ ノルム $\eta\|\mathbf{g}\|_1$ だけ減少する。これは Adam の記憶なし（$\beta_1 = \beta_2 = 0$）極限である符号降下法にほかならない（:numref:`sec_adam`）。

<!--en:857bbe50-->
### 行列とスペクトルノルム

<!--en:979843e1-->
隠れ重み行列 $\mathbf{W} \in \mathbb{R}^{m \times n}$（写像 $\mathbf{x} \mapsto \mathbf{W}\mathbf{x}$）に対し、摂動 $\Delta\mathbf{W}$ を加えると出力の変化は $\|\Delta\mathbf{W}\mathbf{x}\|_2 \le \|\Delta\mathbf{W}\|_2 \|\mathbf{x}\|_2$ で抑えられる。ここで、$\|\Delta\mathbf{W}\|_2 = \sigma_1(\Delta\mathbf{W})$ は**スペクトルノルム**（作用素ノルム、:numref:`subsec_lin-alg-eigenvalues`）である。勾配行列 $\mathbf{G} \in \mathbb{R}^{m \times n}$ の簡約特異値分解（:numref:`sec_mdl-svd-low-rank`）を次式で表す。

<!--en:f0bd4b85-->
ここで、$\mathbf{U} \in \mathbb{R}^{m \times r}$ と $\mathbf{V} \in \mathbb{R}^{n \times r}$ は正規直交列をもち、$\sigma_1 \ge \cdots \ge \sigma_r > 0$ である。なお、フロベニウスノルム $\|\mathbf{G}\|_F = \sqrt{\sum_i \sigma_i^2}$（平坦化した行列のユークリッドノルム）は、特異値の大きさが同程度であるとき、スペクトルノルム $\sigma_1$ を最大で $\sqrt{\min(m, n)}$ 倍上回り得る。

<!--en:976b8dc3-->
スペクトルノルム制約 $\|\Delta\mathbf{W}\|_2 \le \eta$ のもとで :eqref:`eq_muon-ball` を解くと、次の**直交化勾配**が得られる。

<!--en:67bdc3a1-->
これは $\mathbf{G}$ のすべての非ゼロ特異値を $1$ に設定する操作である。証明は直ちに得られる。任意の行列 $\mathbf{A}$（ただし $\|\mathbf{A}\|_2 \le 1$）に対し、トレースの巡回不変性より次式が成り立つ。

<!--en:a8bc9ce4-->
これは、$\|(\mathbf{U}^\top \mathbf{A} \mathbf{V})\|_2 \le \|\mathbf{A}\|_2 \le 1$ から $|(\mathbf{U}^\top \mathbf{A} \mathbf{V})_{ii}| \le 1$ が導かれるためである（演習問題 :exref:`ex_muon-diagonal-entry`）。ここで $\mathbf{A} = -\mathbf{U}\mathbf{V}^\top$ とおけば $\mathbf{U}^\top\mathbf{A}\mathbf{V} = -\mathbf{I}_r$ となり、最小値 $-\eta\sum_i \sigma_i = -\eta\|\mathbf{G}\|_*$（双対ノルムである**核ノルム**、nuclear norm）が達成される。

<!--en:85910aca-->
![3 つのノルム球のもとでの最急降下方向。ユークリッド $\ell_2$ 球（勾配方向）、座標ごとの $\ell_\infty$ ボックス（符号降下法）、およびスペクトルノルム $\|\cdot\|_2$（直交化勾配 $\mathbf{U}\mathbf{V}^{\top}$ であり、有効なすべての特異値が $1$ に等しい）。](../img/mdl-opt-norm-balls.svg)

<!--en:25b7a7e1-->
Adam は*行列の個々の要素*にわたって更新の大きさを揃える（それでもなおランク $1$ の方向へ潰れることがあり、たとえば $\operatorname{sign}\!\begin{bmatrix}3 & 1\\ 1 & 1\end{bmatrix} = \begin{bmatrix}1 & 1\\ 1 & 1\end{bmatrix}$ の特異値は $2$ と $0$ になる）。これに対し、スペクトル最急降下 $-\eta\mathbf{U}\mathbf{V}^\top$ は、等しいステップサイズ $\eta$ で勾配のすべての有効な特異方向 $(\mathbf{u}_i, \mathbf{v}_i)$ に沿って進む。:numref:`fig_opt_muon_spectrum` はこれがスペクトルに与える作用を示している。少数の方向に支配された勾配が、すべての有効な方向を等しく重み付けする更新へと変換されるのである。

<!--en:289952b7-->
![勾配行列 $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ の最大特異値に対する相対特異値 $\sigma_i/\sigma_1$（左）と、その直交化 $\mathbf{U}\mathbf{V}^\top$ の特異値（右）。直交化は特異ベクトルを保ったまますべての非ゼロ特異値を $1$ にするため、更新は各特異方向に沿って $\eta$ ずつ進む。](../img/mdl-opt-muon-spectrum.svg)

<!--en:16789a4c-->
:3 つのノルム幾何のもとでの最急降下。

<!--en:6558932e-->
| 更新に対するノルム球 | 最急降下方向 | 線形化減少量（双対ノルム） | 適したパラメータ群 |

<!--en:db71bc3b-->
| ユークリッド球（$\|\mathbf{d}\|_2 \leq \eta$） | $-\eta\, \mathbf{g} / \|\mathbf{g}\|_2$ | $\eta\, \|\mathbf{g}\|_2$ | 非構造化ベクトル（SGD） |

<!--en:82087f1e-->
| 座標ボックス（$\|\mathbf{d}\|_\infty \leq \eta$） | $-\eta\, \mathrm{sign}(\mathbf{g})$ | $\eta\, \|\mathbf{g}\|_1$ | 非一様な座標・埋め込み（AdamW） |

<!--en:312654d2-->
| スペクトルノルム（$\|\Delta \mathbf{W}\|_2 \leq \eta$） | $-\eta\, \mathbf{U}\mathbf{V}^\top$ | $\eta \|\mathbf{G}\|_* = \eta \sum_i \sigma_i$ | 密な 2 次元隠れ重み行列（Muon） |

<!--en:8733ed8a-->
### 行列型の前処理行列（Shampoo）との関連とパラメータの分割

<!--en:1a572e43-->
直交化勾配 $\mathbf{U}\mathbf{V}^\top$ は、クロネッカー分解型の行列前処理における記憶なし極限としても現れる。**Shampoo** :cite:`Gupta.Koren.Singer.2018` は左右のグラム行列 $\mathbf{L}_t = \sum_{s \le t} \mathbf{G}_s \mathbf{G}_s^\top$ と $\mathbf{R}_t = \sum_{s \le t} \mathbf{G}_s^\top \mathbf{G}_s$ を蓄積し、$\mathbf{L}_t^{-1/4}\mathbf{G}_t\mathbf{R}_t^{-1/4}$ によって勾配に前処理を施す。これを過去の履歴を蓄積せず単一の勾配 $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ に対して評価すると、特異値は恒等的に相殺する。

<!--en:e600834e-->
$\mathbf{L}_t$ と $\mathbf{R}_t$ を保持して逆行列べき乗を計算する代わりに、標準的な一次モーメントのモメンタムバッファに直交化を適用することで、Muon は行列あたりわずか 1 つのモメンタムバッファで全行列前処理の効果を取り込む（:numref:`subsec_mdl-preconditioning-ladder`）。

<!--en:7e131a93-->
スペクトルノルムの幾何構造は活性化値空間のあいだの密な線形変換に固有のものであるため、Muon は**隠れ層の 2 次元重み行列にのみ**（アテンションの `qkv` や `proj`、および MLP の重み）適用される。ワンホットな行参照を行う埋め込みテーブル、1 次元のバイアスや LayerNorm のスケール、および最終層の語彙読み出しヘッドは AdamW で最適化される :cite:`Jordan.Jin.Boza.ea.2024`。

<!--en:adae62e2-->
## ニュートン・シュルツ反復による直交化

<!--en:4fd247fa-->
訓練中に明示的な SVD によって $\mathbf{U}\mathbf{V}^\top$ を計算するのは時間がかかりすぎて実用的ではない。そこで、任意の**奇の行列多項式**（odd matrix polynomial）が $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ の特異ベクトルを保ったまま各特異値に独立に作用することに着目する。

<!--en:2c498b6f-->
これは $\mathbf{X}\mathbf{X}^\top = \mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$ が成り立つためである。まず $\mathbf{X} \leftarrow \mathbf{X} / \|\mathbf{X}\|_F$ と正規化すれば、すべての特異値が $[0, 1]$ に収まる。したがって、奇のスカラー多項式 $p(x)$ として $(0, 1]$ を $1$ に近づけるものを反復適用すれば（**ニュートン・シュルツ反復**、:numref:`subsec_mdl-orthogonalized-updates`）、高速な行列積のみを用いて $\mathbf{X}$ を直交化できる。

<!--en:2a0774f7-->
古典的な 3 次多項式 $p(x) = \frac{3}{2}x - \frac{1}{2}x^3$（原点での傾きが $p'(0) = 1.5$ と小さいため、$5$ ステップでは小さな特異値の持ち上げが遅すぎる）の代わりに、:citet:`Jordan.Jin.Boza.ea.2024` は急な初期の傾き $p'(0) = 3.4445$ をもつ調整済みの 5 次多項式を用いている。

<!--en:328dcff1-->
:numref:`fig_opt_ns_maps` に示すように、:eqref:`eq_muon-quintic` を 5 回合成すると、$[0.01, 1]$ 内のすべての特異値が狭い帯域 $[0.68, 1.13]$ に持ち上げられる（さらに $0.0015$ より大きいすべての特異値も少なくとも $0.68$ に達する）。これは $1$ への厳密な漸近収束と引き換えに、小さな特異方向を $5$ ステップで急速に増幅させる設計である。

<!--en:1afde417-->
![特異値に作用するニュートン・シュルツ多項式。左：古典的な 3 次多項式と調整済み 5 次多項式（:eqref:`eq_muon-quintic`）の 1 回の適用。右：$[10^{-3}, 1]$ 上での 5 回の合成反復（多段階多項式を用いる Polar Express 法 :cite:`Amsel.Persson.Musco.ea.2026` も併記）。](../img/mdl-opt-ns-maps.svg)

<!--en:252a0f4f-->
$m \times n$ 行列（$m \le n$）とトークンバッチサイズ $B$ に対し、5 回のニュートン・シュルツ反復にかかる計算コストは $5(4m^2n + 2m^3)$ FLOPs であるのに対し、順伝播と逆伝播は $6mnB$ FLOPs である。すなわちオーバーヘッドは $\sim 5m/B$ であり、$<1\%$（実用規模のバッチサイズ $B \sim 5 \times 10^5$ のとき）、`TinyLM` 上でも $\sim 12\%$（$B = 4{,}096$ のとき）にとどまる。`newton_schulz` を $96 \times 64$ のガウス行列に適用すると、反復によって特異値スペクトルが $[0.68, 1.20]$（$k = 3$ の後）、さらに $[0.68, 1.13]$（$k = 5$ の後）へと急速に圧縮される様子が確認できる。

<!--en:10a6c2b9-->
実際のネットワークの勾配では、2 つの点が Muon に有利に働く。第 1 に、Muon はノイズの多い単一バッチの勾配ではなく、ネステロフのモメンタムバッファ $\mathbf{G}_t + \beta\mathbf{M}_t$（$\sim 1/(1-\beta) = 20$ ステップにわたる平均、ただし $\beta = 0.95$）を直交化する。第 2 に、フロベニウスノルムの $0.15\%$ 未満の特異値（$3.4445^5 \approx 485$ 倍の増幅では $0.68$ に届かない領域）は $1$ まで押し上げられるのではなく線形に増幅されるにとどまるため、スペクトル末尾の純粋なノイズ方向が自然に減衰される。

<!--en:c8b784e4-->
## ゼロからの Muon の実装

<!--en:64a6cdef-->
### 更新式と RMS アライメント

<!--en:59724c9e-->
$m \times n$ の隠れ重み行列 $\mathbf{W}$ とその勾配 $\mathbf{G}_t$ に対し、Muon はモメンタムバッファ $\mathbf{M}_t$（$\beta = 0.95$）を更新し、5 回のニュートン・シュルツ反復 $\mathrm{NS}_5$ をネステロフの先読み勾配 $\mathbf{G}_t + \beta\mathbf{M}_t$ に適用するとともに、分離型重み減衰（decoupled weight decay）を組み合わせる :cite:`Liu.Su.Yao.ea.2025`。

<!--en:f833cb5b-->
なぜ $0.2\sqrt{\max(m, n)}$ を掛けるのだろうか。$m \times n$ の直交行列 $\mathbf{U}\mathbf{V}^\top$ は $\min(m, n)$ 個の単位特異値をもち、フロベニウスノルムは $\sqrt{\min(m, n)}$、要素ごとの RMS は $\sqrt{\min(m, n)/(mn)} = 1/\sqrt{\max(m, n)}$ となる。したがって $0.2\sqrt{\max(m, n)}$ を掛けることで、行列の次元にかかわらず更新の要素ごとの RMS は $0.2\,\eta$ になる。これは :numref:`sec_adam` で計測した AdamW の経験的な要素ごとの RMS ステップ幅 $\sim 0.2\,\eta$ と一致するため、両方のオプティマイザで同じ学習率 $\eta$ と重み減衰 $\lambda$ を共有できる :cite:`Liu.Su.Yao.ea.2025`。

<!--en:ce156b91-->
各処理を確認するために Muon を自ら実装したが、フレームワークにも標準実装が用意されている。その設定は :eqref:`eq_muon-update` の係数 $0.2\sqrt{\max(m, n)}$ とどのように対応しているのだろうか。これを確かめるため、ゼロ初期化した重みと $96 \times 64$ のガウス勾配に対して自前の実装とライブラリの実装でそれぞれ 1 ステップを実行し、2 つの更新を比較してみよう。

<!--en:4e9314eb-->
2 つの更新の差は約 1% 以内に収まっており、要素ごとの RMS はいずれも約 $0.18\,\eta$ である。これが公称値の $0.2\,\eta$ をわずかに下回るのは、5 回のニュートン・シュルツ反復後の特異値が $0.68$ から $1.13$ のあいだに広がっており、厳密な $1$ にはならないためである。以下の実験では、本節で作成した自前の実装を引き続き使用する。

<!--en:2aed377b-->
`torch.optim.Muon` は 2 次元パラメータのみを受け付けるため、バイアス、正規化パラメータ、埋め込みには依然として別のオプティマイザが必要である。設定 `adjust_lr_fn='match_rms_adamw'` が係数 $0.2\sqrt{\max(m, n)}$ に対応し、デフォルトの `'original'` では代わりに $\sqrt{\max(1, m/n)}$ を掛ける。また、デフォルトの重み減衰は $0.1$ である。ライブラリ実装はニュートン・シュルツ反復を bfloat16 で実行するため、2 つの更新のあいだに約 1% の差が生じる。

<!--en:b096c4d0-->
`optax.contrib.muon` は係数の定数部分を `consistent_rms` として受け取る。値を $0.2$ に指定すると $0.2\sqrt{\max(m, n)}$ となり、指定しない場合は代わりに $\sqrt{\max(1, \text{fan-out}/\text{fan-in})}$ が掛けられる。この実装は 2 次元パラメータに Muon を、それ以外のすべてのパラメータに Adam を適用するため、モデル全体をそのまま渡すことができる。

<!--en:4daae54f-->
### パラメータ群とハイブリッドオプティマイザ

<!--en:97795973-->
`TinyLM` において、内部ブロックの $8$ 個の行列（`qkv`、`proj`、`mlp1`、`mlp2` が $2$ ブロック分）を Muon に振り分け、残る $22$ 個のテンソル（埋め込み、LayerNorm ベクトル、バイアス、出力ヘッド）を AdamW に振り分けると、$393{,}216$ 個のパラメータ（全 $470{,}784$ 個のうち $84\%$）が Muon に割り当てられる。Muon はパラメータあたり AdamW の 2 つのモーメントではなく 1 つのモメンタムバッファしか保持しないため、このハイブリッドオプティマイザはオプティマイザ状態の総メモリ使用量を $42\%$ 削減する。

<!--en:058ac43a-->
そこで、隠れ層の 2 次元行列に分離型重み減衰を適用する `Muon` と、残りのパラメータに重み減衰を適用しない `AdamW` を、`muon_adamw` によって単一のハイブリッドオプティマイザにまとめる。

<!--en:c87b66e2-->
### AdamW との比較

<!--en:192729aa-->
共通の初期重みから出発し、$\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}, 3 \times 10^{-2}, 10^{-1}\}$ の範囲および隠れ行列に対する $\lambda = 0.1$ のもとで $2{,}000$ ステップ訓練を行い、$250$ ステップごとに検証損失を評価して Muon+AdamW ハイブリッドを標準の AdamW とベンチマーク比較する。まず、ベースラインとなる `adamw` のスイープを `run_lm` で実行する。

<!--en:fda4616d-->
次に、`muon_adamw` ハイブリッドについて、まったく同じ学習率グリッド `lrs` と初期シード `seed` でスイープを実行する。

<!--en:bd9a3371-->
各オプティマイザを検証損失が最良となった学習率（`best_adamw` と `best_muon`）で比較し、平滑化した訓練曲線と検証曲線をプロットするとともに、検証損失が $1.6$ nat を下回るステップ数および実用可能な学習率の範囲を表にまとめる。

<!--en:08ca519c-->
`TinyLM` 上では、Muon+AdamW は検証損失で $>0.1$ nat 以上のリードをステップ $500$ の時点で築き、$1.6$ nat のしきい値を AdamW より少なくとも 1 チェックポイント（$250$ ステップ）早く通過する。さらに、実用可能な学習率の上限を $\eta = 3 \times 10^{-2}$ まで広げる（この学習率では AdamW は $2$ nat を大きく上回って終了するが、これは要素ごとの正規化のもとで初期のランク $1$ 的な勾配が、直交化された更新より最大 $\sqrt{\min(m, n)}$ 倍大きいスペクトルノルムをもち得るためである）。

<!--en:425d6803-->
より大規模な領域では、この優位性は小さくなり、検証事例もまだ限られている。完全な学習率減衰を用いた実験において、:citet:`Wen.Hall.Ma.ea.2025` は入念に調整した AdamW に対するステップ数の削減率を計測しており、モデルが大きくなるにつれて約 $1.4\times$（パラメータ数 $0.1\text{B}$）から約 $1.1\times$（$1.2\text{B}$）へと縮小することを報告している。一方、:citet:`Liu.Su.Yao.ea.2025` は独自のスケーリング実験において AdamW の約 2 倍の計算効率を報告している。1 兆パラメータの Kimi K2 は Muon とアテンションロジットのクリッピングを用いて訓練されたが :cite:`Kimi.Team.2025`、その規模で比較対象となる AdamW の訓練実験は存在しない。ただし、オプティマイザ状態をほぼ半減できる利点は規模によらず成り立つ。

<!--en:30ac03ff-->
## まとめ

<!--en:79bec545-->
最急降下は、更新を抑えるノルム球に対して相対的に定義される。ユークリッド $\ell_2$ 球からは SGD が、座標ごとの $\ell_\infty$ ボックスからは符号降下法（平滑化すると Adam）が、そしてスペクトルノルム球 $\|\Delta\mathbf{W}\|_2 \le \eta$ からは行列作用素のすべての有効な特異方向に沿って等しく進む直交化勾配 $-\eta\mathbf{U}\mathbf{V}^\top$ が得られる。**Muon** は、各隠れ 2 次元重み行列のネステロフのモメンタムバッファ上で、調整済み 5 次ニュートン・シュルツ多項式（:eqref:`eq_muon-quintic`）の 5 回の反復によって $\mathbf{U}\mathbf{V}^\top$ を近似し、更新に $0.2\sqrt{\max(m, n)}$ を掛けて AdamW の要素ごとの RMS ステップサイズに合わせるとともに分離型重み減衰を適用し、埋め込み、1 次元ベクトル、読み出しヘッドは AdamW に任せる。`TinyLM` において、この Muon+AdamW ハイブリッドは初期の収束を加速し、AdamW より $3\times$ 高い学習率でも安定性を保ち、オプティマイザ状態のメモリを $42\%$ 削減する。

<!--en:2294243a-->
## 演習問題

<!--en:9a685631-->
1. **対角成分の上界とスペクトルステップの一意性。**$\mathbf{U} \in \mathbb{R}^{m \times r}$ と $\mathbf{V} \in \mathbb{R}^{n \times r}$ が正規直交列をもつとする。$\|\mathbf{U}^\top\mathbf{A}\mathbf{V}\|_2 \le \|\mathbf{A}\|_2$ および $|(\mathbf{U}^\top\mathbf{A}\mathbf{V})_{ii}| \le \|\mathbf{A}\|_2$ が成り立つことを示し、さらに :eqref:`eq_muon-spectral-step` における最小点 $\Delta\mathbf{W}^\star = -\eta\mathbf{U}\mathbf{V}^\top$ が一意となる条件を求めよ。
   :exlabel:`ex_muon-diagonal-entry`

<!--en:c75f5092-->
1. [code] **経験的な更新 RMS と特異値スペクトル。**`Muon` に計測処理を追加して `TinyLM` 上で実行し、$\Delta\mathbf{W} / \eta$ の要素ごとの RMS を $2{,}000$ ステップにわたって追跡するとともに、単一のミニバッチ勾配 $\mathbf{G}_t$ とモメンタムバッファ $\mathbf{M}_t$ の特異値スペクトルをステップ $0$ およびステップ $500$ で比較せよ。
   :exlabel:`ex_muon-step-size`

<!--en:6560717a-->
1. [code] **ニュートン・シュルツの反復回数と多項式の次数。**調整済み 5 次多項式（:eqref:`eq_muon-quintic`）で `num_iters` $\in \{1, 3, 5, 10\}$ とした場合と、$5$ 回の反復による古典的な 3 次多項式 $p(x) = 1.5x - 0.5x^3$ を `TinyLM` 上で比較し、検証損失と 1 ステップあたりの実経過時間を報告せよ。
   :exlabel:`ex_muon-iterations`

<!--en:b648b5d9-->
1. [code] **埋め込みを AdamW に残す理由。**
   :exlabel:`ex_muon-embeddings`
    1. $m \times n$ の勾配行列 $\mathbf{M}$ が $r \le n$ 個の非ゼロかつ線形独立な行のみをもつ（ミニバッチ内の $r$ 個の異なるトークンに対応する）とする。このとき、$\mathbf{U}\mathbf{V}^\top$ のすべての有効な行が単位 $\ell_2$ ノルムをもつことが、該当トークンの出現回数が 1 回であるか $500$ 回であるかによらず成り立つことを示せ。
    1. `token_emb` を `TinyLM` の Muon パラメータ群に移し、`token_emb` を AdamW 群に残した場合と比べて検証パープレキシティが悪化することを実験的に確かめよ。

<!--en:cd9d3531-->
[ディスカッション](https://d2l.discourse.group/t/36764)

<!--en:c6c95b7f-->
行列前処理と Muon<br>**ノルム球での最急降下・スペクトルノルムと UVᵀ・ニュートン・シュルツ 5 次多項式・RMS アライメント・Muon+AdamW ハイブリッド**

<!--en:d7fc58d1-->
::: {.slide title="重み行列は平坦なベクトルではなく線形作用素"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:b53781d0-->
- **要素ごとの手法は置換に無感覚：**SGD、モメンタム、AdamW は $\mathbf{W} \in \mathbb{R}^{m \times n}$ を $mn$ 個の独立な座標として扱うため、$\mathbf{W}$ の要素を並べ替えても AdamW のステップは変わらない

<!--en:f52b4369-->
- **作用素の幾何構造（`TinyLM` の $83.5\%$）：**内部の 2 次元重み行列は層の活性化値を $\mathbf{x} \in \mathbb{R}^n \mapsto \mathbf{y} = \mathbf{W}\mathbf{x} \in \mathbb{R}^m$ と写す

<!--en:29776c00-->
- **中心的な問い：**活性化値の変化 $\Delta\mathbf{y} = \Delta\mathbf{W}\,\mathbf{x}$ を抑えたとき、最急降下方向はどうなるか

<!--en:861bf260-->
::: {.slide title="ノルム球に依存する最急降下方向"}

<!--en:847a5857-->
[最急降下の幾何構造]{.kicker}

<!--en:06b9879f-->
1 次テイラー展開 $f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) + \langle\mathbf{g}, \mathbf{d}\rangle$ のもとで、半径 $\eta > 0$、ノルム $\|\cdot\|$ における最急降下ステップは次の問題を解く（Bernstein & Newhouse, 2024）。

<!--en:d499e476-->
- **1. ユークリッド球（$\|\mathbf{d}\|_2 \le \eta$）：**負の勾配 $\mathbf{d}^\star = -\eta\,\frac{\mathbf{g}}{\|\mathbf{g}\|_2}$ の方向で最小化される（**SGD** の方向）

<!--en:0f68f2a3-->
- **2. 座標ボックス（$\|\mathbf{d}\|_\infty = \max_i |d_i| \le \eta$）：**各座標を頂点へ押し出し $\mathbf{d}^\star = -\eta\,\operatorname{sign}(\mathbf{g})$ となる（**符号降下法**、平滑化すると **Adam**）

<!--en:8fd1ec29-->
::: {.slide title="スペクトル作用素ノルムによる行列ステップの計測"}

<!--en:1197854b-->
[行列作用素の幾何構造]{.kicker}

<!--en:c145343f-->
- **スペクトルノルムは活性化値の変化を抑える：**
  $$\|\Delta\mathbf{W}\,\mathbf{x}\|_2 \le \|\Delta\mathbf{W}\|_2\,\|\mathbf{x}\|_2, \quad \|\Delta\mathbf{W}\|_2 = \sigma_1(\Delta\mathbf{W})$$

<!--en:e77f880f-->
- **SVD とフロベニウスノルムの比較：**$\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_{i=1}^r \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ に対し、$\|\mathbf{G}\|_2 = \sigma_1$ は最大の方向別拡大率を測る一方、$\|\mathbf{G}\|_F = \sqrt{\sum_i \sigma_i^2}$ は直交する $r=128$ の更新を $\sqrt{128} \approx 11.3\times$ だけ過大評価する

<!--en:51faf65e-->
::: {.slide title="スペクトルノルムのもとでの最急降下は UVᵀ"}

<!--en:f1b01b59-->
[直交化勾配]{.kicker}

<!--en:44525c27-->
- **スペクトル最急降下は有効なすべての特異値を $1$ に置き換える：**
  $$\Delta\mathbf{W}^\star = \operatorname*{argmin}_{\|\Delta\mathbf{W}\|_2 \le \eta}\; \langle\mathbf{G}, \Delta\mathbf{W}\rangle = -\eta\,\mathbf{U}\mathbf{V}^\top$$

<!--en:669a550c-->
- **トレースの双対性による証明：**$\|\mathbf{A}\|_2 \le 1$ に対し $\langle\mathbf{G}, \mathbf{A}\rangle = \sum_{i=1}^r \sigma_i (\mathbf{U}^\top\mathbf{A}\mathbf{V})_{ii} \ge -\sum_{i=1}^r \sigma_i = -\|\mathbf{G}\|_*$ となり、下限は $\mathbf{A} = -\mathbf{U}\mathbf{V}^\top$ で達成される

<!--en:88cb1f6d-->
::: {.slide title="要素ごとの符号降下法がランク 1 に潰れ得る理由"}

<!--en:3135b2f3-->
[要素ごとの均一化とスペクトル均一化]{.kicker}

<!--en:3c1b954f-->
- $\mathbf{G} = \begin{bmatrix}3 & 1\\ 1 & 1\end{bmatrix}$（$\sigma_1 \approx 3.41, \sigma_2 \approx 0.59$）を考える
  - **ユークリッド球（$-\mathbf{G}/\|\mathbf{G}\|_F$）：**特異値は $(0.99, 0.17)$ となり、第 2 の方向がほとんど更新されない
  - **座標ボックス（$-\operatorname{sign}(\mathbf{G}) = -\mathbf{1}\mathbf{1}^\top$）：**特異値は $(2.00, 0.00)$ となり、**ランク 1 に潰れる**
  - **スペクトルノルム（$-\mathbf{U}\mathbf{V}^\top = -\mathbf{I}_2$）：**特異値は **$(1.00, 1.00)$** となり、直交するすべての特徴量モードに沿って十分なステップを踏む

<!--en:a79a30f8-->
::: {.slide title="3 つのノルム幾何と Shampoo との関連"}

<!--en:18db82cd-->
[前処理行列の統一的視点]{.kicker}

<!--en:22e5bdca-->
| 更新に対するノルム制約 | 最急降下方向 | 線形化減少量 | 適したパラメータ群 |

<!--en:4e1b7052-->
| **ユークリッド球（$\|\mathbf{d}\|_2 \le \eta$）** | $-\eta\,\mathbf{g}/\|\mathbf{g}\|_2$ | $\eta\,\|\mathbf{g}\|_2$ | 非構造化ベクトル（SGD） |

<!--en:ce778d11-->
| **座標ボックス（$\|\mathbf{d}\|_\infty \le \eta$）** | $-\eta\,\operatorname{sign}(\mathbf{g})$ | $\eta\,\|\mathbf{g}\|_1$ | 埋め込み、1 次元正規化・バイアス、読み出しヘッド（AdamW） |

<!--en:5f8f20a8-->
| **スペクトルノルム（$\|\Delta\mathbf{W}\|_2 \le \eta$）** | $-\eta\,\mathbf{U}\mathbf{V}^\top$ | $\eta\,\|\mathbf{G}\|_* = \eta\sum_i \sigma_i$ | 密な 2 次元隠れ射影行列（Muon） |

<!--en:dab4bb0b-->
- **Muon は記憶なしの Shampoo**：**Shampoo**（Gupta et al., 2018）は左右のグラム行列 $\mathbf{L}_t = \sum_{s \le t}\mathbf{G}_s\mathbf{G}_s^\top$ と $\mathbf{R}_t = \sum_{s \le t}\mathbf{G}_s^\top\mathbf{G}_s$ を蓄積し、$\mathbf{L}_t^{-1/4}\mathbf{G}_t\mathbf{R}_t^{-1/4}$ により前処理する。単一の行列 $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ で評価すると次式が成り立つ。
  $$(\mathbf{G}\mathbf{G}^\top)^{-1/4}\,\mathbf{G}\,(\mathbf{G}^\top\mathbf{G})^{-1/4} = \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\mathbf{U}^\top \cdot \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \cdot \mathbf{V}\boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top = \mathbf{U}\mathbf{V}^\top$$

<!--en:0dd6c57f-->
::: {.slide title="SVD なしの直交化：奇の行列多項式"}

<!--en:90ef9d24-->
[ニュートン・シュルツ反復]{.kicker}

<!--en:d65255f4-->
- **奇の行列多項式は特異ベクトルを保つ：**$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$、$\mathbf{A} = \mathbf{X}\mathbf{X}^\top = \mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$ に対して次式が成り立つ
  $$p(\mathbf{X}) = (a\mathbf{I} + b\mathbf{A} + c\mathbf{A}^2)\mathbf{X} = \mathbf{U}\,p(\boldsymbol{\Sigma})\,\mathbf{V}^\top$$

<!--en:3cc3ded6-->
- **GPU 向けの 2 ステップ手順：**
  1. $\mathbf{X}_0 = \mathbf{G} / (\|\mathbf{G}\|_F + \epsilon)$ と正規化し、$\sigma_i \in [0, 1]$ に収める
  2. 高速な行列積のみで $\mathbf{X}_{k+1} = p(\mathbf{X}_k)$ を $5$ ステップ反復し、$p^{(5)}(\sigma_i) \approx 1$ とする

<!--en:4e979f50-->
::: {.slide title="Jordan らによる調整済み 5 次ニュートン・シュルツ多項式"}

<!--en:e4128d7a-->
[厳密な漸近収束より 5 ステップの高速性]{.kicker}

<!--en:0a704b16-->
- 古典的な 3 次多項式 $p(x) = 1.5x - 0.5x^3$ は漸近的に $1$ へ収束するが、原点での傾きが小さく（$p'(0) = 1.5$、すなわち $1.5^5 \approx 7.6$）、$5$ 回の反復では小さな特異値の持ち上げが遅すぎる

<!--en:1aa93dc7-->
- **Muon の調整済み 5 次多項式**（Jordan et al., 2024）は原点付近の傾き $p'(0) = 3.4445$ を最大化する（$3.4445^5 \approx 485\times$ の増幅）
  $$p(x) = 3.4445\,x - 4.7750\,x^3 + 2.0315\,x^5$$

<!--en:8b9c617b-->
- $5$ 回の反復後、$[0.0015, 1]$ 内のすべての特異値が帯域 **$[0.68, 1.13]$** に収まる

<!--en:5438a9aa-->
::: {.slide title="ニュートン・シュルツ反復の実装と FLOP コスト"}

<!--en:ea34d214-->
[GPU での高速な直交化]{.kicker}

<!--en:bf01a058-->
- **縦長行列（$m > n$）の転置：**$m > n$ のときは $\mathbf{X} \leftarrow \mathbf{M}^\top$ と転置し、グラム行列 $\mathbf{A} = \mathbf{X}\mathbf{X}^\top$ のサイズを小さい方の $\min(m,n) \times \min(m,n)$ に保つ

<!--en:1b2e0861-->
- **大規模訓練では無視できる FLOP オーバーヘッド：**
  - $m \times n$ 行列（$m \le n$）とトークンバッチサイズ $B$ に対し、$5$ ステップのニュートン・シュルツ反復のコストは $5(4m^2 n + 2m^3)$ FLOPs であるのに対し、順伝播＋逆伝播は $6mnB$ FLOPs
  - オーバーヘッドは $\sim \frac{5m}{B}$ であり、**$<1\%$**（実用バッチサイズ $B \sim 5 \times 10^5$ トークン）、`TinyLM` では $\sim 12\%$（$m = 128, B = 4{,}096$）

<!--en:40019407-->
::: {.slide title="ニュートン・シュルツ反復による特異値スペクトルの圧縮"}

<!--en:3aff906a-->
[スペクトルの平坦化]{.kicker}

<!--en:8abffa1c-->
- `newton_schulz` を $96 \times 64$ のガウス行列 $\mathbf{G}$（正規化後の特異値は $[0.02, 0.22]$ に分布）に適用すると、スペクトルが急速に収束する
  - **$k = 1$：**最大 $3.44\times$ 持ち上がる
  - **$k = 3$：**スペクトル全体が $[0.68, 1.20]$ に圧縮される
  - **$k = 5$：**$[0.68, 1.13]$ に平坦化される

<!--en:549e8cc9-->
- **組み込みのノイズ減衰：**相対値 $0.15\%$（基準 $\|\mathbf{G}\|_F$、すなわち $0.68 / 3.4445^5 \approx 0.0014$）を下回る末尾の特異値は、$1$ まで増幅されず小さいままにとどまる

<!--en:5b4296f8-->
::: {.slide title="Muon の完全な更新式と要素ごとの RMS アライメント"}

<!--en:e14d8b60-->
[AdamW の学習率との整合]{.kicker}

<!--en:e129c287-->
- **Muon の更新式（$\beta = 0.95$、Liu et al., 2025）：**
  $$\mathbf{W}_{t+1} = (1-\eta\lambda)\mathbf{W}_t - 0.2\eta\sqrt{\max(m,n)}\;\mathrm{NS}_5(\mathbf{G}_t + \beta\mathbf{M}_t)$$

<!--en:408df713-->
- **なぜ $\mathrm{NS}_5$ の前に平均化するのか：**非線形な直交化を適用する前に、$20$ ステップにわたって持続する特異方向を抽出するため

<!--en:8b31d973-->
- **なぜ $0.2\sqrt{\max(m, n)}$ を掛けるのか：**$\text{RMS}(\mathbf{U}\mathbf{V}^\top) = 1/\sqrt{\max(m,n)}$ であるため、この係数によって要素ごとのステップ RMS が **$0.2\,\eta$** となり、AdamW と一致して単一の $\eta$ と $\lambda$ を共有できる

<!--en:757b8914-->
::: {.slide title="ゼロからの Muon の実装"}

<!--en:a4fbb6b9-->
[コード実装]{.kicker}

<!--en:57287f91-->
`Muon` が保持するのは**パラメータあたり 1 つのモメンタムバッファ（`buf`）**のみであり、オプティマイザ状態は AdamW の半分になる。

<!--en:184a948b-->
::: {.slide title="パラメータの分割：Muon + AdamW ハイブリッド"}

<!--en:493c4f4c-->
[ハイブリッド構成]{.kicker}

<!--en:8fcdee72-->
- スペクトルノルムの幾何構造は、連続的な活性化値ベクトル間の密な線形変換に固有のもの
  - **Muon へ振り分け（$8$ テンソル、$393{,}216$ パラメータ、$83.5\%$）：**内部のすべての 2 次元ブロック行列（両方の Transformer ブロックにわたる `qkv`、`proj`、`mlp1`、`mlp2`）、分離型重み減衰 $\lambda = 0.1$ を適用
  - **AdamW へ振り分け（$22$ テンソル、$77{,}568$ パラメータ、$16.5\%$）：**トークン埋め込み・位置埋め込みの参照テーブル、1 次元のバイアスと LayerNorm スケール、および語彙読み出し `head`（$\lambda = 0$）

<!--en:2183e6e8-->
- **オプティマイザ状態のメモリを 42% 削減：**パラメータの $83.5\%$ に 1 個のバッファ、$16.5\%$ に 2 個のバッファを保持すると、パラメータあたりのバッファ数は $0.835(1) + 0.165(2) = 1.165$ 個となり、従来の $2.0$ 個から減少する（AdamW の状態メモリの $58\%$）

<!--en:b3f612d1-->
::: {.slide title="TinyLM における Muon + AdamW と AdamW の比較"}

<!--en:6a8b4076-->
[訓練と検証の推移]{.kicker}

<!--en:6b52c6b0-->
- `AdamW` と `Muon + AdamW` ハイブリッドの両方を $\eta \in \{10^{-3}, 3{\times}10^{-3}, 10^{-2}, 3{\times}10^{-2}, 10^{-1}\}$ でスイープ（隠れ行列に $\lambda = 0.1$、同一の初期重みから $2{,}000$ ステップ）
  - **初期の速い収束：**ステップ $500$ の時点で、Muon+AdamW は検証損失で **$>0.10$ nat** リードする（約 $1.7$〜$1.75$ 対 $1.88$ nat）
  - **しきい値の早期達成：**検証損失 $1.60$ nat を AdamW より少なくとも **$250$ ステップ早く**下回る

<!--en:dc723930-->
::: {.slide title="高学習率での安定性：3× 高い学習率で訓練可能な Muon+AdamW"}

<!--en:ad21732d-->
[実用可能な学習率の範囲]{.kicker}

<!--en:9d986920-->
- **AdamW の学習率スイープ：**
  - $\eta = 0.03$（上限より $3\times$ 高い値、実用範囲の上限は $\eta = 0.01$）では、AdamW は $2.4$ nat を上回って停滞する。初期のランク 1 に近い符号更新は、同じ要素 RMS をもつ直交化更新よりスペクトルノルムが $\sqrt{\min(m,n)} \approx 11\times$ 大きくなるためである

<!--en:921a2dde-->
- **Muon + AdamW ハイブリッドのスイープ：**
  - すべての隠れ行列の更新のスペクトルノルムを抑えることで、ハイブリッドは $\eta = 0.03$ でも訓練可能であり続け（検証損失は約 $1.67$）、実用可能な学習率の上限を $3\times$ 押し上げる

<!--en:8b35c6c5-->
::: {.slide title="長時間の事前学習への Muon のスケーリングと QK-Clip"}

<!--en:e52359b6-->
[長期事前学習における注意点]{.kicker}

<!--en:5cd76a59-->
- **ステップ削減効果は規模とともに縮小：0.1B で ≈1.4× から 1.2B パラメータで ≈1.1× へ（Wen et al., 2025）。**Kimi K2（1T、2025）は Muon で訓練されたが、比較対象の AdamW 実験はない。長期訓練には 2 つの安全策が必要

<!--en:43e4deb9-->
- **1. 分離型重み減衰（$\lambda > 0$）が必須：**$\mathrm{NS}_5$ が毎ステップ特異値を $\sim 1$ に正規化するため、重みノルム $\|\mathbf{W}\|_F$ は $(1-\eta\lambda)$ による縮小がなければ際限なく増大する

<!--en:8e631e30-->
- **2. 双線形 QK-Clip：**アテンションロジット $\mathbf{q}^\top\mathbf{k} = \mathbf{x}^\top\mathbf{W}_Q^\top\mathbf{W}_K\mathbf{x}$ は 2 つの行列の積であるため、$\mathbf{W}_Q, \mathbf{W}_K$ を最大ロジットが $100$ を超えたときに再スケーリングしてソフトマックスの飽和を防ぐ

<!--en:3ef29bd6-->
::: {.slide title="理解度チェック：スペクトルノルム・5 次多項式・RMS スケール"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:8e74b271-->
Muon の設計に関する各問いについて、正確な数値を求めよ。

<!--en:3b5d9f15-->
1. **要素ごとの RMS アライメント：**`TinyLM` の第 1 の MLP 行列 $\mathbf{W} \in \mathbb{R}^{512 \times 128}$ に対し、厳密な直交行列 $\mathbf{U}\mathbf{V}^\top \in \mathbb{R}^{512 \times 128}$ のフロベニウスノルムと要素ごとの RMS はいくらか。また、係数 $0.2\sqrt{\max(512, 128)}$ を掛けると要素ごとの RMS はいくらになるか。

<!--en:aab193ca-->
2. **埋め込みを AdamW に任せる理由：**あるミニバッチにトークン $A$ が $500$ 回、トークン $B$ が 1 回含まれており、埋め込みの勾配行列 $\mathbf{G}$ が 2 つの非ゼロ行 $\mathbf{g}_A \perp \mathbf{g}_B$（$\|\mathbf{g}_A\|_2 = 500\,\|\mathbf{g}_B\|_2$）をもつとする。厳密な直交化 $\mathbf{U}\mathbf{V}^\top$ はこの 2 行をどう変換し、AdamW は代わりにどう処理するか。

<!--en:80584480-->
3. **3 ステップ対 5 ステップのニュートン・シュルツ到達範囲：**5 次多項式の初期の傾き $p'(0) = 3.4445$ のもとで、正規化特異値 $\sigma / \|\mathbf{G}\|_F$ が $0.68$ に到達するには、反復 $3$ 回と $5$ 回のそれぞれで少なくともどの程度の大きさが必要か。

<!--en:de3e4419-->
::: {.slide title="理解度チェック：スペクトルノルム・5 次多項式・RMS スケール（解答）"}

<!--en:929b7671-2-->
[理解度チェック]{.kicker}

<!--en:557b98b6-->
**解答：**

<!--en:d475a1b7-->
1. $\mathbf{U}\mathbf{V}^\top$ は $\min(512, 128) = 128$ 個の単位特異値をもつため、$\|\mathbf{U}\mathbf{V}^\top\|_F = \sqrt{128}$ であり、全 $512 \times 128 = 65{,}536$ 要素にわたる要素ごとの RMS は $\sqrt{\frac{128}{512 \times 128}} = \frac{1}{\sqrt{512}} \approx 0.0442$ となる。これに $0.2\sqrt{512} \approx 4.525$ を掛けると、要素ごとの RMS はちょうど **$0.2$** になる。

<!--en:c5cbd933-->
2. $\mathbf{U}\mathbf{V}^\top$ の有効な 2 つの行はいずれも単位 $\ell_2$ ノルムをもつ（語彙全体にわたって低頻度トークンと高頻度トークンの方向が結合される）。一方、AdamW は各行の各要素をそれぞれの過去の EMA $\sqrt{\hat{v}_{t,ij}}$ によって独立に正規化する。

<!--en:b66c8e38-->
3. 3 ステップでは小さな値を高々 $3.4445^3 \approx 40.9$ 倍しか増幅できないため、$\sigma/\|\mathbf{G}\|_F \ge 0.68/40.9 \approx$ **$1.66\%$** が必要となる。一方、5 ステップでは $3.4445^5 \approx 485$ 倍増幅されるため、**$0.14\%$** まで到達できる。

<!--en:cdd0014a-->
::: {.slide title="まとめ：ノルム球・ニュートン・シュルツ反復・Muon"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:6e132752-->
| 概念・構成要素 | 数式 | 計算量・メモリ特性 | 深層学習における役割 |

<!--en:cfebe371-->
| **スペクトル最急降下** | $\operatorname*{argmin}_{\|\Delta\mathbf{W}\|_2 \le \eta}\langle\mathbf{G}, \Delta\mathbf{W}\rangle = -\eta\,\mathbf{U}\mathbf{V}^\top$ | $\mathbf{G}$ の有効な特異値をすべて $1$ に設定、減少量は $\eta\|\mathbf{G}\|_*$ | 最悪ケースの活性化値変化 $\|\Delta\mathbf{W}\mathbf{x}\|_2 / \|\mathbf{x}\|_2 \le \eta$ を抑える |

<!--en:d6207780-->
| **ニュートン・シュルツ 5 次多項式** | $p(\mathbf{X}) = 3.4445\mathbf{X} - 4.7750\mathbf{A}\mathbf{X} + 2.0315\mathbf{A}^2\mathbf{X}$ | $5$ 回の行列積反復で $\sigma/\|\mathbf{G}\|_F \ge 0.15\%$ を $[0.68, 1.13]$ に持ち上げる | GPU 上で $\mathbf{U}\mathbf{V}^\top$ を $\sim 5m/B < 1\%$ の FLOP オーバーヘッドで近似 |

<!--en:6394cbe8-->
| **Muon の更新式とスケール** | $\mathbf{W}_{t+1} = (1-\eta\lambda)\mathbf{W}_t - 0.2\eta\sqrt{\max(m,n)}\,\mathrm{NS}_5(\mathbf{G}_t+\beta\mathbf{M}_t)$ | 行列あたり $1$ 個のモメンタムバッファ（$\beta = 0.95$）、要素 RMS $\approx 0.2\eta$ | AdamW と $\eta$ および $\lambda$ をそのまま共有し、状態メモリを $42\%$ 削減 |

<!--en:9496eb5e-->
| **Muon + AdamW ハイブリッド** | 隠れ 2 次元行列に Muon（$84\%$）、埋め込み・ヘッド・1 次元に AdamW（$16\%$） | `TinyLM` で安定な学習率上限が $3\times$ 拡大、0.1B で $\approx 1.4\times$、1.2B で $\approx 1.1\times$ のステップ削減 | 分離型重み減衰 $\lambda > 0$ と併用（超大規模では QK-Clip も併用） |
