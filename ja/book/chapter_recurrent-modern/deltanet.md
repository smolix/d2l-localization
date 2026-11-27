<!--en:a7514c01-->
# デルタ則

<!--en:8c450eec-->
:numref:`sec_matrix-state` の加算型行列メモリは、新しいペアごとに $\mathbf{v}_t \mathbf{k}_t^\top$ を既存の $\mathbf{S}_{t-1}$ に足し合わせることで記憶する。この仕組みは、すべてのキーが 1 度しか束縛されない限り機能する。しかし実際のテキストやコードでは、キーの*再束縛*（rebind）が頻繁に起こる。たとえば、プログラムが `rate = 0.05` を代入した後に `rate = 0.07` へ更新したり、物語の中で登場人物がパリからローマへ移動したりする場合である。加算的書き込みのもとでは、2 度書き込まれたキーをクエリすると古い値と新しい値の和が返され、更新を繰り返すほど最新の値は古い値の下に埋もれてしまう。固定サイズの行列メモリにおいて、あるキーに格納された値を上書きするにはどうすればよいだろうか。

<!--en:dbcf1af4-->
その答えは、*書き込み前に読み出す*ことである。すなわち、$\mathbf{k}_t$ に対する現在のメモリの予測（$\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$）を計算し、予測誤差 $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$ のみを書き込む。この*デルタ則*（delta rule）は :cite:`Widrow.Hoff.1960,Schlag.Irie.Schmidhuber.2021`、順伝播中に二乗想起損失 $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ に対する勾配降下法のオンラインステップを 1 回実行することに相当する。以下では、デルタ則の射影幾何学を導出し、ランダムなキーと相関のあるキーにおける再束縛の利点と信号浸食（signal erosion）のコストの両方を測定する。さらに、スカラー減衰と組み合わせた `GatedDeltaNet` を導入し :cite:`Yang.Kautz.Hatamizadeh.2024`、チャンク単位の三角求解によって評価したうえで :cite:`Yang.Wang.Zhang.ea.2024`、最後に系列メモリをオンライン回帰ソルバとして捉える統一的な視点で締めくくる。

<!--en:addf4128-->
## 加算的書き込み

<!--en:eddc9cf6-->
### 2 度書き込まれたキー

<!--en:a29e2b5a-->
まず、キーが再束縛されたときに加算的書き込みがどのように破綻するかを定量化しよう。単位キー $\mathbf{k}$ が $R$ 回書き込まれ、それぞれの値が $\mathbf{v}^{(1)}, \ldots, \mathbf{v}^{(R)}$ であるとき、加算型メモリはそれらの重みなし和を返す。

<!--en:cff7d725-->
さらに、キー間干渉 $\sum_i (\mathbf{k}_i^\top \mathbf{k})\mathbf{v}_i$ がこれに加わる。スカラー減衰 $a_t \in (0, 1)$ を導入すれば新しい値に比べて古い値を減衰させられるが、同じ経過時間の*すべて*のキーを無差別に減衰させてしまうため、1 度だけ書き込まれた事実も、再束縛されたキーの古い値と同じ速さで薄れてしまう（演習問題 :exref:`ex_dn-decay`）。

<!--en:f3e9a211-->
キー間干渉から再束縛を切り離すため、`overwrite_recall` は $\mathbb{R}^{64}$ 内の互いに正規直交する 8 個のキーをテストする。具体的には、32 個の単位ベクトルからなるコードブックから値を取り出して各キーを $R \in \{1, 2, 4, 8\}$ 回書き込み、$\mathbf{S}\mathbf{k}$ に対するコードブック内の最近傍が最新の値 $\mathbf{v}^{(R)}$ と一致する頻度を測定する。

<!--en:b915f5a8-->
$\mathbb{R}^{64}$ 内の正規直交する 8 個のキー間で干渉がゼロであっても、加算型メモリの想起率は $1.00$（$R = 1$）から $0.51$（$R = 2$、足し合わされた 2 つの値の間でのコイントス）、さらに $0.21$（$R = 8$）へと低下する（演習問題 :exref:`ex_dn-order`）。

<!--en:8acc7bc5-->
## 勾配ステップとしての書き込み

<!--en:bb340002-->
### デルタ則

<!--en:f57461fa-->
他の方向をできるだけ乱さずに $\mathbf{k}_t$ に束縛された値を置き換えるため、新しく入力されたペア $(\mathbf{k}_t, \mathbf{v}_t)$ に対するメモリの二乗予測誤差を測る。

<!--en:1e62642d-->
そして、勾配降下法を 1 ステップ進める更新 $\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}} \ell_t(\mathbf{S}_{t-1})$ を、ステップサイズすなわち*書き込み強度*（write strength）を $\beta_t$ として適用すると次のようになる。

<!--en:d9cace69-->
これが*デルタ則*である :cite:`Widrow.Hoff.1960,Schlag.Irie.Schmidhuber.2021`。現在の予測 $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$ を読み出し、ランク 1 の補正 $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$ を加える。なお、この勾配ステップは訓練時と推論時の両方において、各トークンの*順伝播の内部で*実行される。単位キー $\|\mathbf{k}_t\| = 1$ に対して :eqref:`eq_dn-delta` の直後に $\mathbf{k}_t$ をクエリすると、$\mathbf{S}_t \mathbf{k}_t = (1 - \beta_t)\hat{\mathbf{v}}_t + \beta_t \mathbf{v}_t$ が返される。これは $\beta_t = 1$ のとき厳密な置換となり、$\beta_t \in (0, 1)$ のとき凸結合となる（対照的に、加算的書き込み $\mathbf{v}_t \mathbf{k}_t^\top$ は線形アライメント $\mathbf{v}_t^\top \mathbf{S}\mathbf{k}_t$ に対する勾配上昇法のステップであり、飽和点を持たない）。

<!--en:76991b4b-->
### 遷移

<!--en:1fe3a9c3-->
$\mathbf{S}_{t-1}$ を :eqref:`eq_dn-delta` からくくり出すと、既存のメモリに作用する行列遷移が明らかになる。

<!--en:32eb5d1a-->
単位キー $\mathbf{k}_t$ に対して、対称行列 $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ は $\mathbf{k}_t$ に直交するすべての部分空間を不変に保ち（固有値 $1$）、$\mathbf{k}_t$ の方向を固有値 $1 - \beta_t$ でスケーリングする（:numref:`fig_dn-geometry`）。

<!--en:a57359f2-->
- $\beta_t = 0$：恒等変換（書き込みなし）。

<!--en:1f04d7b4-->
- $\beta_t \in (0, 1)$：$\mathbf{k}_t$ 方向の部分縮小。

<!--en:fe753b0e-->
- $\beta_t = 1$：直交射影 $\mathbf{I} - \mathbf{k}_t \mathbf{k}_t^\top$。$\mathbf{k}_t$ 成分を完全に消去してから $\mathbf{v}_t \mathbf{k}_t^\top$ を書き込む。

<!--en:e62cd025-->
- $\beta_t = 2$：ハウスホルダー鏡映（Householder reflection）$\mathbf{I} - 2\mathbf{k}_t \mathbf{k}_t^\top$（固有値 $-1$）。$\mathbf{k}_t$ 方向の符号を反転させる。

<!--en:c79aec36-->
![遷移 $\mathbf{I} - \beta\mathbf{k}\mathbf{k}^\top$ が、ベクトル $\mathbf{s}$ に対して単位キー $\mathbf{k}$ とそれに直交する方向が張る平面内で作用する様子。$\mathbf{k}$ 方向の成分は $1 - \beta$ 倍され、$\beta = 0$ ではそのまま保たれ、$\beta = 0.5$ では半分になり、$\beta = 1$ では除去され、$\beta = 2$ では反転する。直交成分は変化しない。](../img/mdl-modernrnn-delta-geometry.svg)

<!--en:aec4f32b-->
固有値 $1 - \beta_t\|\mathbf{k}_t\|^2$ から、2 つの帰結が導かれる。

<!--en:bcc7b4af-->
1. **単位ノルムキーが不安定性を防ぐ：**$\|\mathbf{k}_t\| = 1$ かつ $\beta_t \in [0, 2]$ のとき、遷移のすべての固有値は $[-1, 1]$ に収まるため、斉次なメモリ状態が幾何級数的に発散することはない。

<!--en:cbc20745-->
2. **負の固有値が法 2 のカウント（パリティ）を可能にする：**$\beta_t \in [0, 1]$ のとき、遷移の固有値はすべて非負となり、有限精度では任意の系列長にわたってパリティを追跡できない :cite:`Grazzi.Siems.Franke.ea.2025`。$\beta_t$ の範囲を $2$ まで広げると、カウント対象のトークンごとに記憶状態を反転させる鏡映（固有値 $-1$）が可能になり、任意の長さで厳密なパリティを計算できる。:numref:`subsec_dn-state-tracking` では、訓練された層がこの解を見つけられるかを検証する。

<!--en:c0e1690e-->
### 補正のコスト

<!--en:1d613e91-->
正規直交するキーに対しては、$\beta_t = 1$ とすることで他のキーに一切触れずに再束縛されたキーの古い値を消去できる。では、$n > d_k$ の場合や、キーがランダムな単位ベクトルであって異なるキー $\mathbf{k}_i$ と $\mathbf{k}_t$ が非ゼロの重なり $\mathbf{k}_t^\top \mathbf{k}_i$ を持つ場合はどうなるだろうか。後続の各デルタステップは $\mathbf{k}_t$ に沿って進み、$\mathbf{k}_t$ 方向の成分を $\mathbf{v}_i \mathbf{k}_i^\top$ から射影除去するため、キー $i$ に記憶された信号は 1 ステップあたり $1 - (\mathbf{k}_t^\top \mathbf{k}_i)^2 \approx 1 - 1/d_k$ 倍に縮小する。その結果、その後に $m$ 回の書き込みが行われると、古いペアの信号は $(1 - 1/d_k)^m \approx e^{-m/d_k}$ しか残らない。

<!--en:7e6fe857-->
対象を絞った消去がいつ有効でいつ有害になるかを確認するため、`delta_rule(beta)` でオンラインのデルタ書き込みを実装し、`correction_cost` で複数の書き込みラウンドとキー間の相関にわたり幅 $d = 64$ での二乗読み出し誤差と最近傍想起率の両方を測定する。

<!--en:1441e30b-->
ここでは 4 つのメモリ規則、すなわち `additive`、`delta, β = 1`、`delta, β = 0.5`、およびオフラインの `least squares`（`np.linalg.lstsq` により全ペアで同時に $\sum_{s} \|\mathbf{S}\mathbf{k}_s - \mathbf{v}_s\|^2$ を最小化する最小二乗法）を、次の 3 つの設定で比較する。（1）各 4 回書き込まれる 8 個の正規直交キー、（2）1 回書き込まれる $n \in \{32, 64, 128, 256\}$ 個の独立なランダム単位キー、（3）1 回書き込まれる相関のある 64 個のキー（$\rho = 0.5$）である。

<!--en:d27535cd-->
この表から、オンライン射影のトレードオフが浮き彫りになる。

<!--en:9033da41-->
- **再束縛（`8 orthonormal keys, 4 writes`）：**`delta` は想起率 $1.00$ を $\beta = 1$ と $\beta = 0.5$ の両方で達成する（二乗誤差はそれぞれ $0.00$ と $0.33$）のに対し、`additive` と重みなしの `least squares` は各キーに束縛された 4 つの値すべてを平均してしまい、スコアはわずか $0.28$ にとどまる。

<!--en:359c6209-->
- **相関のあるキー（`64 keys, ρ = 0.5`）：**すべてのキーが共通の方向を持つため、`additive` はあらゆる値をその共有成分の上に積み重ねてしまう（二乗誤差 $5.82$、想起率 $0.54$）。一方、`delta` は書き込み前に共有方向の予測を差し引くため、想起率は $0.77$（$\beta = 1$）および $0.89$（$\beta = 0.5$）に達する。

<!--en:ed75b2fa-->
- **過密な独立キー（`128 keys`、`256 keys`）：**ここでは、`delta, β = 1` は `additive` よりも二乗誤差が小さい（$1.06$ 対 $1.99$、$n = 128$ のとき）にもかかわらず、最近傍想起率は*低い*（$0.53$ 対 $0.96$）。なぜだろうか。古い読み出し値を $\mathbf{0}$ に向かって縮小させると、二乗誤差は $\|\mathbf{0} - \mathbf{v}_j\|^2 = 1$ に向かって下がる一方で、初期のペアの信号が後続のペアに比べて浸食されるからである。より穏やかなステップ $\beta = 0.5$ を用いると、より低い二乗誤差（$0.67$）と大幅に高い想起率（$0.82$）の両方が回復する。このことから、$\beta_t$ を入力依存のゲートとし、再束縛されたキーを上書きするときは $\beta_t \approx 1$ を使い、事実を蓄積するときはより小さな $\beta_t$ を使えるようにする動機が得られる。

<!--en:e86329ae-->
## Gated DeltaNet

<!--en:8fbf6ef5-->
### 減衰と補正

<!--en:dfe58852-->
デルタ則は $\mathbf{S}_{t-1}$ を現在のキー $\mathbf{k}_t$ の方向にしか変更しないため、直近のキーに直交する方向は単独では決して減衰しない。*Gated DeltaNet* は :cite:`Yang.Kautz.Hatamizadeh.2024`、大域的な忘却を担う入力依存のスカラー減衰 $a_t \in (0, 1)$ と、対象キーの上書きを担うデルタ則のランク 1 方向別更新を組み合わせる。

<!--en:5e0c0ee7-->
2 つのゲートは異なる役割を担い、両者の組み合わせによってこの層は新しいペアを 3 通りの方法で扱うことができる（:numref:`fig_dn-gate-modes`）。$a_t$ が 0 に近いとき、層はそれまでに記憶した内容をすべて一掃して新しいペアだけを残す。これは話題の切り替わりに対する適切な応答である。$a_t \approx 1$ かつ $\beta_t = 1$ のときは、先ほどの再束縛テストと同様に、現在のキーに対応する値を置き換えつつ直交するキーには手を触れない。$a_t \approx 1$ で $\beta_t$ が小さいときは、読み出し値を新しい値へ向かって途中までしか動かさないため、類似したキーに対する複数回の書き込みでは最後の 1 つが独り勝ちするのではなく平均化される。

<!--en:ea903058-->
![ゲートの 3 つの設定のもとでの、$2 \times 2$ メモリに対する Gated DeltaNet の 1 ステップ。矢印は現在のキー $\mathbf{k}_t$ 方向（青）と直交するキー方向（紫）の読み出し値を表し、ステップ前が破線、ステップ後が実線である。減衰 $a_t$ が 0 に近いとメモリが一掃され、新しいペアだけが残る。$a_t \approx 1$ のとき、書き込み強度 $\beta_t = 1$ では $\mathbf{k}_t$ に対応する値が $\mathbf{v}_t$ に置き換えられて他方のキーはそのまま保たれる一方、小さな $\beta_t$ では読み出し値が $\mathbf{v}_t$ に向かって途中までしか移動しない。](../img/mdl-modernrnn-gated-delta-modes.svg)

<!--en:5dbd7fcc-->
### チャンク単位の計算

<!--en:1e26dab2-->
遷移 $a_t(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ はスカラーではなく $d_k \times d_k$ 行列であるため、素朴なトークン単位のスキャンでは $d_k \times d_k$ 行列の乗算が必要となり、$O(T d_k^3)$ のコストがかかる。:citet:`Yang.Wang.Zhang.ea.2024` は、*WY 表現*（WY representation）を用いて :eqref:`eq_dn-gated` をサイズ $m$ のチャンク単位で評価する方法を示した。まず :eqref:`eq_dn-gated` を、減衰付きの加算的更新 $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{u}_t \mathbf{k}_t^\top$ として書き換える。ここで、*実効補正値*（effective corrected value）は $\mathbf{u}_t = \beta_t(\mathbf{v}_t - a_t \mathbf{S}_{t-1}\mathbf{k}_t)$ である。長さ $m$ のチャンクの内部で初期状態を $\mathbf{S}_0$ とし、$a_t \mathbf{S}_{t-1}\mathbf{k}_t = \gamma_t \mathbf{S}_0 \mathbf{k}_t + \sum_{s < t} L_{ts}(\mathbf{k}_s^\top \mathbf{k}_t)\mathbf{u}_s$（ただし $L_{ts} = \prod_{r=s+1}^t a_r$、$\gamma_t = \prod_{r=1}^t a_r$）を展開すると、狭義下三角な $m \times m$ の連立一次方程式がチャンク内の補正値 $\mathbf{u}_1, \ldots, \mathbf{u}_m$ について得られる。

<!--en:03c87c24-->
右辺は $\mathbf{S}_0$ に関してアフィンであるため、`delta_rule_writes` は単位下三角な $m \times m$ の系 $\mathbf{M} = \mathbf{I} + \textrm{diag}(\boldsymbol{\beta})\,\textrm{tril}_{-1}(\mathbf{L} \odot \mathbf{K}\mathbf{K}^\top)$ を、2 つの右辺に対してすべてのチャンクとヘッドにわたり並列に解き、$\mathbf{W} = \mathbf{M}^{-1}(\boldsymbol{\beta} \odot \mathbf{V})$ と $\mathbf{Z} = \mathbf{M}^{-1}((\boldsymbol{\beta} \odot \boldsymbol{\gamma}) \odot \mathbf{K})$ を求める。これにより、$\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ は $\mathbf{S}_0$ が確定した時点で直ちに得られる。

<!--en:38c5a861-->
$\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ をチャンク末尾の状態 $\mathbf{S}_m = \gamma_m \mathbf{S}_0 + \mathbf{U}^\top \mathbf{D}\mathbf{K}$（ただし $\mathbf{D}$ はチャンク末尾までの減衰を並べた対角行列）に代入すると、チャンク遷移は $T / m$ 個のチャンクにわたるアフィンな行列漸化式として表される。

<!--en:82ede4ce-->
$(\mathbf{A}, \mathbf{W}^\top \mathbf{D}\mathbf{K})$ を $T / m$ 個のチャンクにわたってスキャンするコストはわずか $(T/m) d_k^3$ 回の演算で済み（チャンク内の $O(T d_k^2)$ の行列積と $m \approx d_k$ のときに一致する）、その後は $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ およびすべての出力 $\mathbf{Y}$ が 1 回のバッチ行列積で求まる。

<!--en:90e5770f-->
`chunked_delta_rule` を検証するため、まずトークンごとに逐次計算する参照実装 `delta_rule_recurrent` を用意する。

<!--en:1e07fc2a-->
続いて、チャンクサイズ $m \in \{16, 64\}$ の `chunked_delta_rule` を、単位正規化されたキーを持つ長さ $T = 256$ のランダム系列上で `delta_rule_recurrent` と比較する。

<!--en:1a831ded-->
チャンク単位のアルゴリズムは、トークン単位の漸化式と相対誤差 $\sim 10^{-7}$ の精度で一致する。

<!--en:2cdc020d-->
### 層

<!--en:f9495a07-->
`GatedDeltaNet` は `d2l.GatedLinearAttention`（:numref:`sec_matrix-state`）のサブクラスであり、書き込み強度の射影 `W_beta`（ヘッドごとに $\beta_{t,h} = \sigma(\mathbf{w}_{\beta,h}^\top \mathbf{x}_t + b_{\beta,h}) \in (0, 1)$）だけを追加し、`memory` をオーバーライドして `chunked_delta_rule` を呼び出す。それ以外のすべての構成要素（幅 4 の因果畳み込み、$\ell_2$ 正規化されたクエリとキー、ヘッドごとのスカラー減衰、ヘッドごとの RMS 正規化、および SiLU 出力ゲート）は同一である。

<!--en:08491a83-->
### 再束縛

<!--en:b55ef495-->
訓練された `GatedDeltaNet` 層は、`GatedLinearAttention` が苦戦する再束縛されたキーの上書きを実行できるだろうか。1 ブロックの `SeqLM`（$d = 64$、次元 16 のヘッド 4 個、初期減衰時間スケール $[256, 2048]$）を `d2l.associative_recall` でテストする。ここでは 128 種類から選んだ 16 個のキーと 16 種類から選んだ値を用い、16 個のキーのうち 8 個を最初の束縛からわずか 2 回の書き込み後に新しい値へ再束縛する（`num_rebound=8, gap=2`）。この間隔を短く保つことで 2 つの束縛の経過時間がほぼ同じになるため、経過時間に基づく大域的減衰では、1 度しか書き込まれていないキーまで忘れることなしに新しい値と古い値を容易には切り分けられない（2 ブロックの加算型モデルは深さ方向を使って 1 回のデルタステップを近似できるため、ここでは単一ブロックを用いる。演習問題 :exref:`ex_dn-rebinding` を参照）。

<!--en:5a3d9dae-->
`recall_by_kind` を用いて再束縛されたキー（`rebound`）と 1 度だけ書き込まれたキー（`~rebound`）の正解率を分けて集計し、`GatedLinearAttention` と `GatedDeltaNet` をそれぞれ 2 つの乱数シードで 16 ペアの再束縛タスク上で訓練する。

<!--en:f35b0ad3-->
260 パラメータの `W_beta` ヘッドを追加するだけで、再束縛されたキーに対する想起率は約 $0.9$（`gated linear attention`）から $0.96$ 以上（`Gated DeltaNet`）へと上昇し、1 度だけ書き込まれたキーとの差は約 $2$ パーセントポイント以内に縮まる。

<!--en:7df486a6-->
### 状態追跡

<!--en:74e16226-->
これまでの実験はすべて、何かをメモリに記憶させて後から返すことを求めていた。これとは異なる要求として、トークンごとに変化しうる状態を*追跡*するという課題がある。たとえば、一連の入れ替えの後にどのカップにボールが隠れているか、引用符が開いたままか、あるいは最も単純な例として、これまでに読み込んだ 1 の個数が奇数か偶数かといった問題である。パリティに必要な状態はわずか 1 ビットであり、1 が来れば反転し、0 が来ればそのまま保たれる。何かを長期間保持する必要はないが、1 つでも見落とすとそれ以降の答えがすべて誤りになるため、何ひとつ飛ばすことは許されない。

<!--en:5c22d934-->
GRU はプログラムと同じやり方でこの課題を解く。そのゲートは直前の状態 $\mathbf{h}_{t-1}$ を参照できるため、1 ステップで「入力が 1 なら反転する」という計算を実行できる。本章の線形漸化式は、並列訓練と引き換えにこの仕組みを手放した。すなわち、減衰と書き込みが入力のみに依存する（:numref:`sec_linear-recurrence`）。記憶されたビットを反転させるには、状態のある座標に負の数を掛けなければならないが、minGRU の減衰 $\mathbf{z}_t \in (0, 1)$ は座標を縮小できても符号を変えることは決してできない。デルタ則も、書き込み強度がシグモイドである限り同じ制約を受ける。遷移 :eqref:`eq_dn-transition` のキー方向の固有値が $1 - \beta_t \in (0, 1)$ となるからである。しかし $\beta_t \in (0, 2)$ とすれば、固有値は $-1$ に達しうる。入力が 1 のたびに値 $\mathbf{v}$ を $\beta_t = 2$ で 1 つの固定キー $\mathbf{k}$ に書き込み、0 のときには何も書き込まない層は、1 が来るたびに読み出し値を $\mathbf{S}_t\mathbf{k} = 2\mathbf{v} - \mathbf{S}_{t-1}\mathbf{k}$ へと変化させる。空のメモリから始めれば、読み出し値は $2\mathbf{v}$ と $\mathbf{0}$ の間を交互に行き来し、これがそのままパリティとなる。

<!--en:1841b126-->
では、このような構成法が存在するとして、訓練によってそれを見つけられるだろうか。2 ブロック・幅 64 の `d2l.SeqLM`（この課題で :citet:`Grazzi.Siems.Franke.ea.2025` が用いた深さ）を 64 ビットのランダムな列で訓練し、各ステップでそこまでのビットのパリティを出力させる。続いて 256 ビットの列でテストすることで、規則そのものを学習したモデルと、訓練時の長さに適合しただけのモデルとを切り分ける。比較するミキサは、GRU、minGRU、`GatedDeltaNet`、および書き込み強度のシグモイドを 2 倍にした点以外は同じ層である `GatedDeltaNetReflect` の 4 つである。上の議論が正しければ、GRU と $\beta_t \in (0, 2)$ を持つ層は訓練長を超える位置を含めすべての位置で正解する一方、minGRU とシグモイドのデルタ則は失敗し、訓練時より長い列になるとランダム予測の水準である $0.5$ まで低下するはずである。

<!--en:504dc123-->
各ミキサについて 2 回ずつ訓練を実行し、訓練中に観測した長さである位置 1〜64 と、位置 65〜256 における正解率を報告する。

<!--en:89905f1c-->
GRU は両方の実行において、訓練長でもそれを超える領域でもすべての位置で正解している。一方、minGRU とシグモイドのデルタ則はそうならない。最初の 64 位置ではランダム予測の水準を上回り、minGRU は $0.5$ をわずかに超え、Gated DeltaNet は約 $0.75$ から $0.8$ に達するものの、どちらも規則を学習してはいない。位置 64 を超えると、両者ともコイントスと同じ $0.50$ の正解率に落ち込む。書き込み強度を 2 倍にすると、この結果が一変する。`GatedDeltaNetReflect` は訓練長で $1.00$ に達し、その正解率を 256 ビットまで維持する。これは偶然うまくいった 2 回の実行ではなく典型的な結果である。さらに約 50 回実行したところ、1 回を除くすべてがこのように汎化し、その例外も位置 64 以降で約 $0.9$ のスコアを記録した。つまり、線形性そのものが障害なのではない。線形漸化式が手放したのは状態に依存する更新であり、それにもかかわらず行列メモリが状態を追跡できるかどうかは、遷移の固有値がとりうる範囲によって決まる。minGRU は負の固有値も、自身の状態を参照するゲートも持たない。2 ブロック目を重ねても欠けている符号は補えず、モデルは失敗する。

<!--en:c4603589-->
もっとも、この鏡映は繊細である。2 倍にしたシグモイドは $\beta_t = 2$ に漸近するだけで到達はしないため、反転のたびに記憶されたビットも少しずつ縮小し、書き込み強度が 2 の手前で止まった実行では長い列で正解率が低下する（演習問題 :exref:`ex_dn-parity`）。複数のカップの入れ替えを追うような、より難しい状態追跡には、1 トークンあたり複数の鏡映が必要となる。

<!--en:31afd808-->
## 回帰としてのメモリ

<!--en:e4d09b7e-->
一歩引いて眺めると、本章のすべての行列メモリ、さらにはソフトマックスアテンションも、系列を読み進めながらキー $\mathbf{K}_t \in \mathbb{R}^{t \times d_k}$ から値 $\mathbf{V}_t \in \mathbb{R}^{t \times d_v}$ への回帰予測器をその場で当てはめる操作として理解できる :cite:`Wang.Shi.Fox.2025`。$\sum_{s \le t} \|\mathbf{S}\mathbf{k}_s - \mathbf{v}_s\|^2$ の厳密な線形最小二乗法による最小点は次の式で与えられる。

<!--en:4251c516-->
ここで、$(\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ は $t < d_k$ のとき擬似逆行列に置き換えられる。:eqref:`eq_dn-least-squares` の 2 つの因子に注目しよう（:numref:`tab_dn-regression`）。

<!--en:7943d2dd-->
- **線形アテンション**は最初の因子である非正規化交差モーメント $\mathbf{V}_t^\top \mathbf{K}_t = \sum_{s \le t} \mathbf{v}_s \mathbf{k}_s^\top$ のみを保持し、重なり合うキーを無相関化するグラム逆行列 $(\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ を省略する。

<!--en:af53bcef-->
- **デルタ則**はトークンごとに :eqref:`eq_dn-least-squares` に向かってオンラインの確率的勾配ステップを 1 回進めるのに対し、**Gated DeltaNet** は各ステップの前に重み減衰（weight decay、$a_t < 1$）を加える。陽的な勾配ステップを近接陰的ステップ（proximal implicit step）$\arg\min_{\mathbf{S}} \frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ に置き換えると、正規化されたステップサイズ $\beta_t / (1 + \beta_t\|\mathbf{k}_t\|^2)$ を持つ Longhorn が得られ :cite:`Liu.Wang.Wu.ea.2024`（演習問題 :exref:`ex_dn-implicit`）、一方で $(\mathbf{K}_t^\top \mathbf{K}_t + \lambda\mathbf{I})^{-1}$ を陽に保持すれば逐次最小二乗法（recursive least squares）が得られる（演習問題 :exref:`ex_dn-rls`）。

<!--en:83f55fe6-->
- **ソフトマックスアテンション**は、保存されたすべてのペアに対してクエリ時に*ノンパラメトリック*な Nadaraya–Watson カーネル回帰（:numref:`sec_attention-pooling`）を実行し、$O(t)$ の KV キャッシュというコストと引き換えにパラメトリックな圧縮を回避する。

<!--en:ab92c337-->
厳密な当てはめも、1 ペアずつ最新の状態に保ち続けることができる。各ペアは $\mathbf{K}_t^\top \mathbf{K}_t$ をランク 1 の項 $\mathbf{k}_t\mathbf{k}_t^\top$ だけ変化させるため、Sherman–Morrison の公式（:numref:`subsec_mdl-block-matrices`）によって新たな逆行列計算を行うことなく逆行列を更新できる。これが逐次最小二乗法である。この手法は、メモリの隣にサイズ $d_k \times d_k$ の第 2 の状態を保持する（演習問題 :exref:`ex_dn-rls`）。

<!--en:c87f7ae0-->
:これまでに読み込んだペアに対する回帰としてのメモリ。

<!--en:99d76f8d-->
| メモリ | 当てはめる対象 | 方法 | 状態 |

<!--en:e898c6e8-->
| ソフトマックスアテンション | すべての値のカーネル重み付き平均 | クエリ時 | 全ペア |

<!--en:496a3722-->
| 線形アテンション | $\mathbf{V}^\top\mathbf{K}$（キー補正なしの最小二乗法） | 1 ペアにつき 1 回の加算 | $d_v \times d_k$ |

<!--en:bb77d212-->
| ゲート付き線形アテンション | 同上（直近のペアに重み付け） | 1 ペアにつき 1 回の加算 | $d_v \times d_k$ |

<!--en:1e980258-->
| デルタ則 | 最小二乗法の近似 | 1 ペアにつき 1 回の勾配ステップ | $d_v \times d_k$ |

<!--en:c5cb210f-->
| Gated DeltaNet | 重み減衰付き最小二乗法の近似 | 1 ペアにつき 1 回の勾配ステップ | $d_v \times d_k$ |

<!--en:de31b8ab-->
これらすべてのアーキテクチャにおいて、学習は 2 つの時間スケールで進行する。状態 $\mathbf{S}_t$ の*高速重み*（fast weights）が順伝播中に現在のコンテキストにおけるキー・値の対応関係を当てはめる一方、ネットワークの*低速重み*（slow weights、$\mathbf{W}_{qkv}, \mathbf{W}_a, \mathbf{W}_\beta, \ldots$）は、そのオンライン当てはめが有効に働くようなキー、値、クエリ、ステップサイズを生成するよう、系列をまたいで誤差逆伝播法により訓練される。しかし、どれほど最適な書き込み規則を用いても、$d_v \times d_k$ 行列のランクとビット容量は有限である。:numref:`sec_hybrids` ではその厳然たる限界を測定し、ごく一部のフルアテンション層を挟み込むことでその限界をどのように引き上げられるかを示す。

<!--en:30ac03ff-->
## まとめ

<!--en:644547ce-->
加算型行列メモリは同じキーへのすべての書き込みを重ね合わせるため、再束縛されたキーをクエリすると古い値と新しい値の和が返される。デルタ則は書き込み前に現在の予測 $\mathbf{S}_{t-1}\mathbf{k}_t$ を読み出し、誤差補正 $\beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$ のみを加える。これは $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ に対する勾配降下法の 1 回のオンラインステップであり、その遷移 $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ はメモリを $\mathbf{k}_t$ 方向に選択的に縮小する。対象を絞った消去はキーが再束縛される場合や相関を持つ場合に加算的書き込みを上回る一方、多数のランダムなキーが重なり合う状況で強い消去（$\beta_t = 1$）を行うと初期の信号が浸食されうる。`GatedDeltaNet` は入力依存の書き込み強度 $\beta_t$ とヘッドごとのスカラー減衰 $a_t$ を組み合わせ、下三角求解（`delta_rule_writes`）とチャンク単位のスキャン（`chunked_delta_rule`）によって漸化式を並列チャンクで評価し、連想想起において再束縛されたキーを確実に復元する。

<!--en:2294243a-->
## 演習問題

<!--en:abdd1c3b-->
1. **順序は結果に影響しない。** 一連の書き込みを行った後の加算型メモリが、書き込みの順序に依存しないことを示せ。ここから、いかに巧妙な読み出し規則であっても、加算型メモリだけからは 1 つのキーに書き込まれた $R$ 個の値のうち最新のものを復元できないことを結論づけ、さらに $R$ 個の値がすべて異なるときに最良の規則が達成できる想起率を求めよ。コードブックのサイズを 32 個から 1,024 個の値へ増やしたとき、8 回の書き込み後に測定される想起率はどのように変化するか。また、それはなぜか。
   :exlabel:`ex_dn-order`

<!--en:00fc44cb-->
1. [code] **補正の代わりに減衰を用いる。** `overwrite_recall` の加算的書き込みに定数減衰を与え、$\mathbf{S} \leftarrow a\mathbf{S} + \mathbf{v}\mathbf{k}^\top$ とせよ。
    1. $R = 2, 4, 8$ に対して、最新の値が少なくとも 95% の割合で想起される最大の減衰 $a$ を求めよ。なぜその答えは同じキーに対する 2 回の書き込みの間に何回の書き込みが挟まるかに依存するのか。また、このテストにおいてより小さな減衰が決して悪影響を与えないのはなぜか。
    1. 次に、正規直交キーを幅 64 における 16 個のランダムな単位キーに置き換えよ。そのうち 8 個をこれまで通り $R$ 回書き直し、残りの 8 個は最初に 1 度だけ書き込んで、最後に両方の種類のキーに対する想起率を測定せよ。どちらの種類のキーにも対応できる書き込み規則はどれか。
   :exlabel:`ex_dn-decay`

<!--en:e3c84f91-->
1. **陰的ステップ。** $\tfrac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 + \tfrac{\beta}{2}\|\mathbf{S}\mathbf{k} - \mathbf{v}\|^2$ の最小点が $\mathbf{S}_{t-1} - \frac{\beta}{1 + \beta\|\mathbf{k}\|^2} (\mathbf{S}_{t-1}\mathbf{k} - \mathbf{v})\mathbf{k}^\top$ であることを示せ（ヒント：勾配を 0 と置き、その結果に $\mathbf{k}$ を掛けてまず $\mathbf{S}\mathbf{k}$ を求めるとよい）。さらにその遷移の固有値を求め、このステップが新しいペアを厳密に想起するのは極限 $\beta \to \infty$ の場合に限られることを示せ。
   :exlabel:`ex_dn-implicit`

<!--en:8eea3d19-->
1. [code] **逐次最小二乗法。** :eqref:`eq_dn-least-squares` に $\lambda \to 0$ で漸近する正則化最小二乗法の当てはめは、メモリが $\mathbf{P}_t = (\mathbf{K}_t^\top \mathbf{K}_t + \lambda\mathbf{I})^{-1}$ も併せて保持すれば 1 ペアずつ更新できる。
    1. :numref:`subsec_mdl-block-matrices` の Sherman–Morrison の公式を用いて、新しいペアが 1 つ与えられた後の $\mathbf{P}_t$ と $\mathbf{S}_t$ の更新式を導出し、$\mathbf{S}_t$ の更新が $\mathbf{P}_t\mathbf{k}_t$ 方向に（$\mathbf{k}_t$ ではなく）書き込むデルタ則であることを示せ。デルタ則の $d_v d_k$ 個に対し、この状態はいくつの数値を保持するか。
    1. これを実装し、上書きテストにおいてデルタ則と比較せよ。すべてのペアに対する最良の当てはめである逐次最小二乗法が、最新の値をうまく想起できないのはなぜか。また、どうすればこれを解決できるか。
   :exlabel:`ex_dn-rls`

<!--en:78302446-->
1. [code] **パリティ。** ● `GatedDeltaNet` の書き込み強度はシグモイドであるため、キー方向の遷移の固有値 $1 - \beta_t$ は $(0, 1)$ に収まる。$2\sigma(\cdot)$ を $\sigma(\cdot)$ の代わりに用いると、固有値は $(-1, 1)$ に収まるようになる。
    1. それでもなお、訓練された層は $\beta_t = 2$ に漸近することしかできない。:numref:`subsec_dn-state-tracking` のパリティの構成法において、カウント対象のすべてのトークンで $\beta = 2 - \epsilon$ と置き、カウント対象のトークンが $n$ 個続いた後の読み出し値が $\big(1 - (\epsilon - 1)^n\big)\mathbf{v}$ となることを示せ。奇数回カウントした後と偶数回カウントした後の読み出し値の差は、どれほどの速さで縮小するか。$\epsilon = 0.01$ のとき、この差が $\|\mathbf{v}\|$ の 1% 未満になるのはカウント対象のトークンが何個続いた後か。また、それは有限精度での読み出しにとって何を意味するか。
    1. :numref:`subsec_dn-state-tracking` では訓練長の 4 倍でモデルをテストした。`Gated DeltaNet, 2σ` を改めて訓練し、1,024 ビットおよび 4,096 ビットの列でテストせよ。正解率が低下し始めるのはどの長さか。また、その長さは最初の小問の答えと整合しているか。
   :exlabel:`ex_dn-parity`

<!--en:23506d47-->
1. [code] **再束縛への別の経路。** 再束縛の実験では、書き込み規則だけが値を補正できるように 3 つの設定を固定していた。これらを 1 つずつ緩和し、それぞれの結果を説明せよ。
    1. ゲート付き線形アテンションのメソッド `decays`（$\log a_t$ を返す）をオーバーライドして同じ形状のゼロを返すようにし、すべての $a_t = 1$ として減衰を無効化したうえで再度訓練せよ。書き直されたキーと 1 度だけ書き込まれたキーに対する想起率はそれぞれどのように変化するか。
    1. 減衰を元に戻し、各書き直しが後方の任意の位置で起こるようにせよ（`gap=None` を `task` に指定する）。2 つの層の間の差は縮まるか。2 つの値の経過時間を使って説明せよ。
    1. 両方の層を 1 ブロックではなく 2 ブロックで訓練せよ。2 つの加算層を組み合わせることで、どのようにして値を補正できるか。
   :exlabel:`ex_dn-rebinding`

<!--en:e74aae10-->
[ディスカッション](https://d2l.discourse.group/t/36784)

<!--en:6f6e76f7-->
デルタ則と Gated DeltaNet<br>**書き込み前読み出しによる誤差補正・ハウスホルダー射影の幾何学・チャンク単位の WY 三角求解・オンライン回帰としてのメモリ**

<!--en:149202ac-->
::: {.slide title="加算型行列メモリが再束縛されたキーで破綻する理由"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:6ba14efa-->
プログラムや物語では状態が絶えず更新される（`rate = 0.05` $\to$ `rate = 0.07`、あるいはパリからローマへ移動する登場人物）。単位キー $\mathbf{k}$ が $R$ 回にわたって値 $\mathbf{v}^{(1)}, \ldots, \mathbf{v}^{(R)}$ で書き込まれると、加算型メモリはそれらの重みなし重ね合わせを返す。

<!--en:cb551021-->
互いに正規直交する $8$ 個のキーを $\mathbb{R}^{64}$ 内で（キー間干渉ゼロ）各 $R \in \{1, 2, 4, 8\}$ 回書き込んでテストする。

<!--en:aa850fde-->
キー数の $8\times$ の次元があっても、最新の値に対する加算型メモリの想起率はコイントス水準（$0.51$、$R = 2$）さらには $0.21$（$R = 8$）へと崩壊する。大域的減衰 $a_t < 1$ は、*すべて*のキーを無差別に古びさせることでしか古い値を薄められない。

<!--en:046c7199-->
::: {.slide title="デルタ則：想起損失に対する 1 回のオンライン勾配ステップ"}

<!--en:f42cfda6-->
[書き込み前の読み出し]{.kicker}

<!--en:6c14dae3-->
- **書き込み前の読み出し：**まず予測 $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$ を読み出し、スケール付き誤差残差 $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$ を書き込む

<!--en:6002e646-->
- **$\beta_t = 1$ での厳密な置換：**$\|\mathbf{k}_t\| = 1$ のとき、1 ステップで $\mathbf{S}_t \mathbf{k}_t = \mathbf{v}_t$ となる

<!--en:e1d289da-->
::: {.slide title="デルタ遷移行列の射影幾何学"}

<!--en:7e4ea30e-->
[方向別消去]{.kicker}

<!--en:30cb410d-->
デルタ更新から $\mathbf{S}_{t-1}$ をくくり出すと、右から掛かる状態遷移行列が現れる。

<!--en:12dd69bc-->
単位キー $\|\mathbf{k}_t\| = 1$ に対し、$\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ は固有値 **$1$** を $(d_k - 1)$ 次元部分空間 $\mathbf{k}_t^\perp$ 上に持ち、固有値 **$1 - \beta_t$** を $\mathbf{k}_t$ 方向に持つ。

<!--en:1e237e32-->
- **$\beta_t = 0$：**単位行列 $\mathbf{I}$（メモリは不変）

<!--en:c73bdf92-->
- **$\beta_t \in (0, 1)$：**$\mathbf{k}_t$ 方向の部分縮小

<!--en:c91fd284-->
- **$\beta_t = 1$：**直交射影 $\mathbf{I} - \mathbf{k}_t\mathbf{k}_t^\top$（$\mathbf{k}_t$ を完全に消去してから $\mathbf{v}_t \mathbf{k}_t^\top$ を書き込む）

<!--en:37a8eb53-->
- **$\beta_t = 2$：**ハウスホルダー鏡映 $\mathbf{I} - 2\mathbf{k}_t\mathbf{k}_t^\top$（固有値 $-1$、$\mathbf{k}_t$ 方向の符号を反転）

<!--en:c8f2b5b0-->
::: {.slide title="単位キーによる安定性とハウスホルダーパリティ鏡映"}

<!--en:add1fec0-->
[スペクトル特性]{.kicker}

<!--en:560229c2-->
- **必須のキー $\ell_2$ 正規化：**$\mathbf{k}_t$ 方向の固有値 $\lambda_{\mathbf{k}} = 1 - \beta_t\|\mathbf{k}_t\|^2$ が $[-1, 1]$ に収まる（$\beta_t \in [0, 2]$ 全体で）のは、$\|\mathbf{k}_t\| = 1$ の場合に限られる

<!--en:c175820e-->
- **負の固有値がパリティを可能にする：**正の減衰 $a_t > 0$ では法 $2$ のカウントは不可能だが、$\beta_t = 2$ ではハウスホルダー鏡映（$\lambda_{\mathbf{k}} = -1$）が読み出しを $\mathbf{0} \leftrightarrow 2\mathbf{v}$ の間で切り替える

<!--en:37abad06-->
::: {.slide title="補正の隠れたコスト：信号浸食"}

<!--en:c035ecef-->
[過密なランダムキー]{.kicker}

<!--en:047642b0-->
- **副次的な縮小：**後続の各ランダム単位キー $\mathbf{k}_s$ は期待二乗重なり $1/d_k$ を $\mathbf{k}_i$ との間に持ち、1 ステップごとに $\mathbf{v}_i$ を $1 - 1/d_k$ 倍に縮小させる

<!--en:3068966a-->
- **過密による減衰：**$m = 2d_k$ 回のランダム書き込み（$\beta = 1$）の後、初期の信号は $e^{-2} \approx 13.5\%$ まで浸食される

<!--en:d926a87d-->
::: {.slide title="3 つの設定における 4 つの書き込み規則のベンチマーク" layout="code"}

<!--en:b8ef1134-->
[加算型・デルタ則・最小二乗法の比較]{.kicker}

<!--en:c4fb8137-->
`additive`、`delta, β = 1`、`delta, β = 0.5`、およびオフラインの `least squares`（$\mathbf{S}^\star = \mathbf{V}^\top \mathbf{K}(\mathbf{K}^\top \mathbf{K})^+$、全ペアを同時に最小二乗法で当てはめ）を幅 $d = 64$ で比較する。

<!--en:f14e0e5a-->
::: {.slide title="対象を絞った消去が有効なとき、そしてなぜ βt を学習すべきか"}

<!--en:2cc16424-->
[書き込み規則の比較表の解釈]{.kicker}

<!--en:ab32d595-->
ベンチマークから明らかになる 3 つの領域が、入力依存ゲート $\beta_t \in (0, 1)$ を学習する動機となる。

<!--en:ff529376-->
1. **キーの再束縛（`8 orthonormal keys, 4 writes`）：**`delta`（$\beta = 1$ または $0.5$）は最新の値に対し **想起率 $1.00$** を達成するが、`additive` と重みなし `least squares` は各キーに束縛された $4$ 個の値すべてを平均してしまう（$0.28$）

<!--en:de46967a-->
2. **相関のあるキー（`64 keys, ρ = 0.5`）：**`additive` はすべての値を共有キー方向に積み重ねる（誤差 $5.82$、想起率 $0.54$）が、`delta` は書き込み前に共有方向の予測を差し引くため想起率 **$0.89$**（$\beta = 0.5$）を得る

<!--en:ba406d64-->
3. **過密な独立キー（`128 keys`、`256 keys`）：**$n = 128$ では、`delta, β = 1` は `additive` より二乗誤差が*小さい*（$1.06$ 対 $1.99$）のに想起率は*悪い*（$0.53$ 対 $0.96$）。古い読み出し値を $\mathbf{0}$ へ縮小すると二乗誤差は $\|\mathbf{0} - \mathbf{v}_j\|^2 = 1$ 付近に抑えられる一方、初期の信号が浸食されるためである。ステップを半分の $\beta = 0.5$ にすると **想起率 $0.82$** と **二乗誤差 $0.67$** を回復する。

<!--en:2141b3e2-->
::: {.slide title="Gated DeltaNet：大域的減衰と対象を絞った上書き"}

<!--en:3792beea-->
[Gated DeltaNet の漸化式]{.kicker}

<!--en:954507c4-->
- **3 つの動作モード（Yang et al., 2024）：**
  1. **大域的消去（$a_t \to 0$）：**話題の境界で行列全体を一掃する
  2. **対象キーの更新（$a_t \approx 1, \beta_t \to 1$）：**$\mathbf{k}_t$ を上書きしつつ $\mathbf{k}_t^\perp$ を保持する
  3. **穏やかな蓄積（$a_t \approx 1, \beta_t \approx 0.2$）：**浸食の少ない連想書き込み

<!--en:fdc094b9-->
::: {.slide title="下三角求解によるチャンク単位の評価（WY 形式）"}

<!--en:39be9e78-->
[チャンク内の三角連立系]{.kicker}

<!--en:5ced0d7e-->
- **WY 表現：**デルタ漸化式を $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{u}_t \mathbf{k}_t^\top$（補正値 $\mathbf{u}_t$）と書き換える

<!--en:55dac04e-->
- **並列 $m \times m$ 三角求解：**全チャンク並列に $\mathbf{M}\mathbf{W} = \boldsymbol{\beta} \odot \mathbf{V}$ と $\mathbf{M}\mathbf{Z} = (\boldsymbol{\beta} \odot \boldsymbol{\gamma}) \odot \mathbf{K}$ を解き、$\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ を得る

<!--en:fbc50197-->
::: {.slide title="チャンク間の行列スキャンと数値的検証"}

<!--en:eceb3ce9-->
[チャンクアルゴリズムの検証]{.kicker}

<!--en:b8891f38-->
$\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ をチャンク末尾の状態に代入すると、$T/m$ 個のチャンクにわたるアフィン行列漸化式が得られる。

<!--en:7941f9e5-->
- **均衡のとれた計算量：**$\mathbf{A}^{(c)}$ を $T/m$ 個のチャンクにわたり合成するコストは $(T/m)d_k^3$ FLOPs であり、チャンク内 GEMM の $O(T d_k^2)$ とチャンクサイズ $m \approx d_k$ のときに一致する

<!--en:a0132ae6-->
::: {.slide title="ドロップイン置換可能な GLA サブクラスとしての GatedDeltaNet 実装"}

<!--en:597bc086-->
[簡潔なオブジェクト指向設計]{.kicker}

<!--en:9c8d1db4-->
`GatedLinearAttention`（§{sec_matrix-state}）は状態漸化式を `memory(X, Q, K, V, log_a)` の内部に分離しているため、`GatedDeltaNet` は線形射影 `W_beta`（$d \to h$）を 1 つ追加して `memory` をオーバーライドするだけでよい。

<!--en:c98a55d1-->
それ以外のすべての構成要素（幅 4 の因果畳み込み、$\ell_2$ 正規化されたクエリとキー、対数一様なヘッド減衰の初期化、ヘッドごとの RMSNorm、および SiLU 出力ゲート）はそのまま継承される。

<!--en:383f1923-->
::: {.slide title="キー再束縛実験：短い間隔での上書き"}

<!--en:edf40355-->
[再束縛ベンチマーク結果]{.kicker}

<!--en:c17b2812-->
1 ブロックの `GatedLinearAttention` モデルと 1 ブロックの `GatedDeltaNet` モデル（$d = 64$、$4$ 個の $16 \times 16$ ヘッド）を $16$ ペアの連想想起で比較する。ここでは $8$ 個のキー（全 $16$ 個中）が、**最初の束縛からわずか `gap=2` 回の書き込み後**に新しい値へ再束縛される。

<!--en:81718780-->
- **なぜ `gap=2` がデルタ則を切り分けるか：**最初の束縛からわずか $2$ ペア後に上書きされると両方の束縛の経過時間がほぼ等しくなるため、大域的減衰 $a_t$ では、それ以前に 1 度だけ書き込まれたキーまで消去することなしに古い値を薄められない

<!--en:f18a6bdc-->
- **確実な上書き：**$260$ パラメータの `W_beta` 射影を追加すると、再束縛キーの正解率は複数回の実行で **$0.96$ 以上**に向上し、`GatedLinearAttention`（約 $0.9$）を上回って 1 回書き込みの正解率との差も約 $2$ ポイント以内に収まる

<!--en:667d2e70-->
::: {.slide title="なぜ 1 ブロックで検証したか：深さがデルタステップを近似する"}

<!--en:ed9152e6-->
[複数層の相互作用]{.kicker}

<!--en:20632a06-->
- **2 つの加算層が 1 回のデルタステップを模倣：**ブロック 1（`GLA`）が $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}^{(1)}\mathbf{k}_t$ をクエリして残差ストリームに出力し、ブロック 2（`GLA`）が $(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$ を書き込める

<!--en:4cb86c2b-->
- **層内デルタ則が優れる理由：**`GatedDeltaNet` は*各層の内部で*任意の再束縛回数 $R$ に対して書き込み前読み出しを実行し、より深い層を推論のために解放する

<!--en:ac5f68bf-->
::: {.slide title="陰的近接デルタステップ（Longhorn）"}

<!--en:63b3626e-->
[キー正規化なしの無条件安定性]{.kicker}

<!--en:a7ae83c3-->
- **陰的近接目的関数（Liu et al., 2024）：**$\frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|_F^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ を*新しい*状態 $\mathbf{S}_t$ において最小化する

<!--en:6d44d6ad-->
- **無条件安定性：**キー方向の固有値は $\frac{1}{1 + \beta_t\|\mathbf{k}_t\|^2} \in (0, 1)$ となり、任意の $\|\mathbf{k}_t\|$ と $\beta_t > 0$ に対して成り立つ

<!--en:56b68b56-->
::: {.slide title="オンライン回帰ソルバとしての系列ミキサの統一"}

<!--en:f8ac5a25-->
[テスト時回帰としての視点]{.kicker}

<!--en:a7c172a4-->
- **高速重み（$\mathbf{S}_t$）：**順伝播中に文脈内の線形回帰 $\mathbf{S}_t \mathbf{k}_s \approx \mathbf{v}_s$ をトークン $1\ldots t$ にわたり解く

<!--en:b0062406-->
- **低速重み（$\theta$）：**外側ループの AdamW で訓練され、$(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, a_t, \beta_t)$ を形成する

<!--en:337a5401-->
::: {.slide title="回帰アルゴリズムとしての系列メモリの分類"}

<!--en:de5cfb1c-->
[回帰としての比較表]{.kicker}

<!--en:c760839b-->
| 系列メモリ | 文脈内回帰の目的関数・推定量 | トークンペア $(\mathbf{k}_t, \mathbf{v}_t)$ ごとのオンライン更新 | 保持する状態 |

<!--en:f2a5d482-->
| **線形アテンション** | 非正規化交差モーメント $\mathbf{V}_t^\top \mathbf{K}_t$（$(\mathbf{K}_t^\top\mathbf{K}_t)^{-1}$ を省略） | $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | $d_v \times d_k$ |

<!--en:b7ec41c0-->
| **ゲート付き線形アテンション** | 指数重み付き交差モーメント | $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | $d_v \times d_k$ |

<!--en:0881a85e-->
| **DeltaNet** | $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ に対するオンライン SGD | $\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$ | $d_v \times d_k$ |

<!--en:df73f627-->
| **Gated DeltaNet** | 重み減衰 $a_t \in (0, 1)$ 付きオンライン SGD | $\mathbf{S}_t = a_t\mathbf{S}_{t-1}(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top) + \beta_t\mathbf{v}_t\mathbf{k}_t^\top$ | $d_v \times d_k$ |

<!--en:392b9550-->
| **Longhorn（陰的ステップ）** | 近接ステップ $\min_{\mathbf{S}} \frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ | 正規化ステップ $\frac{\beta_t}{1 + \beta_t\|\mathbf{k}_t\|^2}$ を用いるデルタ則 | $d_v \times d_k$ |

<!--en:5b253289-->
| **逐次最小二乗法** | 厳密なリッジ回帰 $\mathbf{V}_t^\top\mathbf{K}_t(\mathbf{K}_t^\top\mathbf{K}_t + \lambda\mathbf{I})^{-1}$ | $\mathbf{P}_t\mathbf{k}_t$ に沿った Sherman–Morrison 更新 | $d_v d_k + d_k^2$ |

<!--en:e3054842-->
| **ソフトマックスアテンション** | ノンパラメトリックな Nadaraya–Watson カーネル回帰 | $(\mathbf{k}_t, \mathbf{v}_t)$ をキャッシュに追加し、クエリ時に評価 | $t \times (d_k + d_v)$ |

<!--en:d9a85aa2-->
::: {.slide title="理解度チェック：デルタ遷移とチャンク求解"}

<!--en:929b7671-->
[理解度の確認]{.kicker}

<!--en:b1fef131-->
以下の 3 つの問いでデルタ則の理解度を確認せよ。

<!--en:88f4dcb5-->
1. $\mathbf{S}_{t-1}\mathbf{k}_t = \hat{\mathbf{v}}_t$ とする。単位キー $\|\mathbf{k}_t\| = 1$ と書き込み強度 $\beta_t = 0.5$ でデルタ則を適用したとき、直後の読み出し $\mathbf{S}_t \mathbf{k}_t$ は何を返し、また $\mathbf{S}_t \mathbf{q}$ は任意のクエリ $\mathbf{q} \perp \mathbf{k}_t$ に対してどうなるか。

<!--en:a8e0db03-->
2. なぜ陰的近接更新（Longhorn）では $\beta_t$ が $\tilde{\beta}_t = \beta_t / (1 + \beta_t\|\mathbf{k}_t\|^2)$ に置き換わるのか。また、なぜ陰的ステップでは厳密なパリティを計算できないのか。

<!--en:7e6824e2-->
3. `chunked_delta_rule` において、チャンク内の係数行列 $\mathbf{M} = \mathbf{I} + \textrm{diag}(\boldsymbol{\beta})\,\textrm{tril}_{-1}(\mathbf{L} \odot \mathbf{K}\mathbf{K}^\top)$ が単位下三角行列になるのはなぜか。

<!--en:557b98b6-->
**解答：**

<!--en:cf367c97-->
1. $\mathbf{S}_t \mathbf{k}_t = (1 - 0.5)\hat{\mathbf{v}}_t + 0.5\mathbf{v}_t = \mathbf{\tfrac{1}{2}(\hat{\mathbf{v}}_t + \mathbf{v}_t)}$（中間への混合）。$\mathbf{q} \perp \mathbf{k}_t$ に対しては $\mathbf{k}_t^\top \mathbf{q} = 0$ となるため、$\mathbf{S}_t \mathbf{q} = \mathbf{\mathbf{S}_{t-1}\mathbf{q}}$（**完全に不変**）

<!--en:4117543a-->
2. $\mathbf{S} - \mathbf{S}_{t-1} + \beta_t(\mathbf{S}\mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^\top = \mathbf{0}$ を解く際、*更新後*の予測 $\mathbf{S}\mathbf{k}_t$ を用いるため、キー方向の固有値は $1 - \tilde{\beta}_t\|\mathbf{k}_t\|^2 = \frac{1}{1 + \beta_t\|\mathbf{k}_t\|^2} \in (0, 1)$（任意の $\beta_t > 0$）となる。これによりキーの正規化なしでオーバーシュートを防げる一方、パリティの鏡映に必要な負の固有値（$\lambda = -1$）は除外される

<!--en:a5193fad-->
3. 因果性により、補正値 $\mathbf{u}_t$（ステップ $t$）はチャンク内のより前の書き込み $s < t$ のみに依存するため（$\textrm{tril}_{-1}$ は対角線上およびそれより上が 0 であり、$\mathbf{M}$ の対角成分はすべて $1$ となる）

<!--en:8c964813-->
::: {.slide title="まとめ：デルタ則と Gated DeltaNet"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:ed4ad812-->
| 概念・構成要素 | 数式表現 | 主な機能とトレードオフ |

<!--en:dc754dcf-->
| **デルタ則** | $\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$ | $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ に対するオンライン勾配ステップ。書き込み前に読み出す |

<!--en:41170557-->
| **遷移の幾何学** | $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$（$\|\mathbf{k}_t\| = 1$） | 固有値 $1$（$\mathbf{k}_t^\perp$ 上）、固有値 $1 - \beta_t$（$\mathbf{k}_t$ 方向）：$\beta=1$ で射影、$\beta=2$ で鏡映 |

<!--en:7ec0b203-->
| **信号浸食のコスト** | $(1 - 1/d_k)^m \approx e^{-m/d_k}$（$m$ 回のランダム書き込み後） | 強い $\beta_t = 1$ は重なりのある初期キーを浸食。学習された $\beta_t \in (0, 1)$ が両者を均衡させる |

<!--en:7b244eb5-->
| **Gated DeltaNet** | $\mathbf{S}_t = a_t\mathbf{S}_{t-1}(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top) + \beta_t\mathbf{v}_t\mathbf{k}_t^\top$ | 大域的なヘッド減衰 $a_t$ と対象キーの上書き $\beta_t$ を統合（再束縛で $0.96+$） |

<!--en:7bb2223d-->
| **チャンク単位の WY アルゴリズム** | 三角求解 $\mathbf{M}\mathbf{W}, \mathbf{M}\mathbf{Z}$ ＋チャンクスキャン | 並列 Tensor Core GEMM により $d_k \times d_k$ 行列遷移を $O(T d_k^2)$ の計算量で評価 |
