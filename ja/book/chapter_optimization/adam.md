<!--en:a8637903-->
# Adam と AdamW

<!--en:ef5fe0e6-->
:numref:`sec_test_models` では、単純なミニバッチ SGD は `TinyLM` において学習率 $\eta = 0.5$ を用いてようやく約 $2.1$ nats/byte の損失に到達した。これに対し、100 分の 1 未満の学習率である $\eta = 3 \times 10^{-3}$ を用いた Adam は、以下で測定するように $1.59$ nats/byte という損失に同じ $2{,}000$ ステップで到達する。公称学習率が 100 分の 1 ほど小さいオプティマイザがなぜより速く訓練を進められるのか、そしてなぜ Transformer における普遍的なデフォルトとなっているのだろうか。

<!--en:894e3ede-->
モメンタム（:numref:`sec_momentum`）は勾配を時間方向に平均化するが、それでもすべてのパラメータに単一のスカラーのステップサイズを適用する。しかし深層ネットワーク、とりわけ Transformer では、埋め込み、アテンション射影、MLP 層、正規化パラメータの間で勾配の大きさが何桁も異なる。**Adam**（*Adaptive Moment Estimation*、:citet:`Kingma.Ba.2014`）は、モメンタムと、**AdaGrad** :cite:`Duchi.Hazan.Singer.2011` および **RMSProp** :cite:`Tieleman.Hinton.2012` を経て発展した座標ごとの二乗平均平方根（root-mean-square、RMS）勾配の前処理行列（preconditioner）を組み合わせる。各座標の更新量をその座標自身の過去の勾配スケールで割ることにより、学習率 $\eta$ は生の勾配に対する乗数ではなく、パラメータ空間における直接的なステップ長となる。

<!--en:8315c05e-->
しかし、まさにこの座標ごとの除算が標準的な $\ell_2$ 正則化（regularization）を破綻させる。損失に $\frac{\lambda}{2}\|\mathbf{x}\|^2$ を加えると、各重みの減衰率がその勾配スケールの逆数で再スケーリングされてしまうからである。:citet:`Loshchilov.Hutter.2019` は重み減衰（weight decay）を適応的な前処理行列から分離することでこの問題を解決し、**AdamW** を生み出した。本節では、AdaGrad と RMSProp から Adam への発展をたどり、`TinyLM` 上で座標ごとの勾配スケールを測定してモメンタム付き SGD と Adam を比較し、さらに AdamW とその重み減衰のタイムスケール $\tau = 1/(\eta\lambda)$ および平衡ノルムを導出したうえで、どのパラメータグループを重み減衰の対象外とすべきかを詳しく述べる。

<!--en:8056206f-->
## 座標ごとの適応的学習率

<!--en:a42d48be-->
### AdaGrad と RMSProp

<!--en:1ba631ca-->
疎または不均一な特徴量を持つモデル（たとえば、頻出トークンは毎回のミニバッチに現れる一方、希少なトークンはごくまれにしか現れないトークン埋め込みなど）では、単一の大域的な学習率ですべての座標に適切に対応することはできない。**AdaGrad** :cite:`Duchi.Hazan.Singer.2011` は二乗勾配の累積和 $\mathbf{s}_t$ を要素ごとに蓄え、各座標のステップを $\sqrt{\mathbf{s}_t + \epsilon}$ に反比例させてスケーリングする。

<!--en:f0633bad-->
ここで、$\mathbf{s}_0 = \mathbf{0}$ であり、$\epsilon > 0$ はゼロ除算を防ぐための定数である。座標軸に沿った谷においては、各座標をそれ自身の勾配の大きさでスケーリングすることで方向ごとの曲率（curvature）の差が自動的に打ち消され、:numref:`subsec_gd-preconditioning` の対角前処理行列に対する勾配のみを用いた近似として機能する（:numref:`subsec_mdl-per-coordinate`）。

<!--en:0a169e47-->
$\mathbf{s}_t$ は二乗勾配を上界なく単調に蓄積するため、持続的な確率的勾配のもとでは AdaGrad の実効ステップサイズが $O(t^{-1/2})$ で減衰し、非凸な深層ネットワークではやがて学習の進行が止まってしまう。**RMSProp** :cite:`Tieleman.Hinton.2012` はこの累積和を、二乗勾配の指数移動平均（exponential moving average、EMA）$\mathbf{v}_t$（ウィンドウ幅は $\sim 1/(1-\beta_2)$ ステップ）に置き換える。

<!--en:7ba73e1a-->
*（補足：標準的な最適化の記法に従い、ここでの $\mathbf{v}_t$ は :numref:`sec_momentum` の速度ベクトルではなく、二次モーメントである二乗勾配の平均を表す。）*古い勾配を忘却することで、RMSProp は中程度の学習率 $\eta = 0.4$ からでも消失しないステップサイズを維持する。

<!--en:7913f864-->
:eqref:`eq_rmsprop` の 2 つの性質に注目しよう。

<!--en:246c077e-->
1. **バイアス補正なしでの初期ステップの増幅：**$\mathbf{v}_0 = \mathbf{0}$ であるため、最初のステップでは $\mathbf{v}_1 = (1 - \beta_2)\mathbf{g}_1^2$ となり、初回のジャンプが大きさ $\eta/\sqrt{1-\beta_2} \approx 3.16\,\eta$（$\beta_2 = 0.9$ のとき）、さらには $31.6\,\eta$（$\beta_2 = 0.999$ のとき）と過大になる。

<!--en:a4103cec-->
2. **正規化されたステップ長：**ある座標の勾配の大きさがほぼ定常であるときは常に $\sqrt{v_t} \approx |g_t|$ となり、更新量の大きさは $\approx \eta$ となって $|g_t|$ がどれほど小さくなっても変わらない。したがって極小値の近傍では、定学習率の適応的手法は $O(\eta)$ の近傍内で振動し続け、スケジュールによって $\eta$ を減衰させるまで収まらない（:numref:`sec_scheduler`）。

<!--en:7d762b3b-->
### Adam とバイアス補正

<!--en:d91eeacc-->
**Adam** :cite:`Kingma.Ba.2014` は、一次モーメント（first moment）の EMA（モメンタム、$\mathbf{m}_t$）と非中心二次モーメント（second moment）の EMA（$\mathbf{v}_t$）を組み合わせ、両方の推定量に厳密な**バイアス補正**（bias correction）を適用する。$\mathbf{v}_t = (1-\beta_2)\sum_{s=1}^t \beta_2^{t-s}\mathbf{g}_s^2$ を初期値 $\mathbf{v}_0 = \mathbf{0}$ から展開すると、重みの総和は $1 - \beta_2^t$ となり、$1$ にはならないことがわかる。そこで $1 - \beta_1^t$ と $1 - \beta_2^t$ で割ることにより、不偏モーメント推定量 $\hat{\mathbf{m}}_t$ と $\hat{\mathbf{v}}_t$ が得られる。

<!--en:7a1252af-->
パラメータ更新では、バイアス補正された一次モーメントを座標ごとの RMS スケール $\sqrt{\hat{\mathbf{v}}_t}$ で割る。

<!--en:5c2b5f66-->
標準的なデフォルト値は $\beta_1 = 0.9$、$\beta_2 = 0.999$（大規模言語モデルの訓練では $\beta_2 = 0.95$ も用いられる :cite:`brown2020language`）、$\epsilon = 10^{-8}$ である。$t = 1$ では $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ かつ $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$ となるため、初回のステップは厳密に $-\eta\,\operatorname{sign}(\mathbf{g}_1)$ となる。

<!--en:e2a17231-->
より一般に、目的関数の任意の座標を正の定数 $c > 0$ で再スケーリングすると、$\hat{m}_{t,i}$ と $\sqrt{\hat{v}_{t,i}}$ の両方が $c$ 倍されるため、それらの比は（$\epsilon \to 0$ において）不変に保たれる。したがって、**Adam のもとでは、$\eta$ はパラメータ空間における座標ごとの直接的なステップ長を指定する**ことになる。すなわち、勾配の符号が一貫している座標はステップごとに $\approx \eta$ だけ移動する一方、平均ゼロのノイズが支配的な座標は $\approx \eta\sqrt{(1-\beta_1)/(1+\beta_1)} \approx 0.23\,\eta$ だけ移動する。

<!--en:bf6d195e-->
Adam をゼロから実装するには、各パラメータテンソルに対してステップカウンタ $t$ と 2 つのバッファ $(\mathbf{m}, \mathbf{v})$ を保持する必要がある。

<!--en:bf97666a-->
## 言語モデルにおける Adam

<!--en:0b5e2a05-->
### `TinyLM` における勾配スケールと SGD との比較

<!--en:55d22636-->
なぜ Transformer にとって座標ごとのスケーリングがこれほど不可欠なのだろうか。`TinyLM`（:numref:`subsec_tinylm`）全体の勾配スケールを測定するため、まず `WikiText103` データセット（$b = 64$、ウィンドウ長 $64$）を読み込む。

<!--en:cf723051-->
次に、各パラメータテンソルの二乗平均平方根（RMS）勾配を $10$ ミニバッチにわたって計算する `gradients(model, data)`、パラメータ値を複製する `snapshot(model)`、および初期状態の `TinyLM` インスタンスと Adam（$\eta = 3 \times 10^{-3}$）を定義する。

<!--en:a9ff1c09-->
テンソルを $8$ 種類の機能別パラメータ群に `rms_by_kind` で分類し、ステップ $0$ とステップ $500$ における RMS 勾配の大きさを、正規化された Adam のステップ長 $\|\Delta\mathbf{x}\|_{\text{RMS}} / \eta$ と並べて比較する。

<!--en:4bbed372-->
生の勾配の大きさは、初期化時点で **$40\times$ の範囲**（トークン埋め込みの $\sim 2 \times 10^{-4}$ からバイアスの $\sim 7 \times 10^{-3}$ まで）に及び、**$15\times$ の範囲**がステップ $500$ の時点でも残る。SGD のもとでは、単一のスカラー学習率によって勾配が最大のグループ（バイアス、LayerNorm、出力ヘッド）が埋め込みよりも $15\text{--}40\times$ 速く動かされる。一方 Adam のもとでは、8 つすべてのパラメータグループにわたる正規化されたステップサイズは狭い $2\times$ の範囲内（$0.1\eta\text{--}0.3\eta$ であり、純粋なノイズに対する比 $\sqrt{(1-\beta_1)/(1+\beta_1)} \approx 0.23$ に近い）に収まる。

<!--en:9fc21fb7-->
`TinyLM` の勾配スケールは $10^{-4}\text{--}10^{-2}$ 付近にあり、かつ Adam の EMA には $(1-\beta_1) = 0.1$ が含まれるため、$\eta \sim 3 \times 10^{-3}$ の Adam は、モメンタム $0.9$ かつ $\eta \sim 0.3$ の SGD と同程度の大きさのパラメータステップを生み出す。共通の $3$ つのシードにわたって両オプティマイザを比較するため、まずモデルとオプティマイザを生成するクロージャ `make_lm`、`make_sgd`、`make_adam` を用意する。

<!--en:b105798f-->
続いて、シードごとに同一の初期重みから出発し、モメンタム付き SGD を $\eta \in \{0.1, 0.3, 0.5, 1.0\}$ にわたって、Adam を $\eta \in \{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}, 10^{-2}, 3 \times 10^{-2}\}$ にわたってスイープする。

<!--en:60a250e9-->
各オプティマイザを最良の学習率（`best_rate`）で集計し、実用可能な学習率の範囲（最良値から $0.15$ nats 以内）、$1.8$ nats に到達するまでのステップ数、1 ステップでの最大の損失スパイク、および平滑化された損失の軌跡を比較する。

<!--en:7102bfae-->
`TinyLM` において、Adam は目標損失 $1.8$ nats にモメンタム付き SGD のほぼ半分のステップ数（$650\text{--}800$ 対 $1{,}100\text{--}1{,}500$）で到達し、より低い最終損失で訓練を終える。さらに $10\times$ の実用可能な学習率ウィンドウ（$10^{-3}$〜$10^{-2}$）で動作し、$\eta = 3 \times 10^{-2}$ においてさえ正規化された更新長が $\eta$ で上から抑えられるため有限に留まる。一方、SGD の最良学習率 $\eta = 0.3$ は発散のわずか 1 グリッド手前に位置しており、1 ステップでの最悪の損失スパイクは Adam の数倍に達し、$\eta = 0.5$ では大半の実行が `inf` へと発散する。サブワード語彙を持つより大規模な Transformer では、ブロック単位のヘッセ行列の不均一性 :cite:`Zhang.Chen.Ding.ea.2024` と裾の重いトークン頻度 :cite:`Kunstner.Chen.Lavington.ea.2023,Kunstner.Yadav.Milligan.ea.2024` によって、この差はさらに拡大する。

<!--en:333a8557-->
この実行では、1 回のスパイクによって SGD のあるシードが大きく後退し、最後まで $1.8$ に到達しない。

<!--en:f11b62d5-->
### $\epsilon$ の役割と二次モーメント推定の限界

<!--en:6dbe2a5b-->
:eqref:`eq_adam-update` において、$\epsilon$ は座標ごとの最大実効学習率を $\eta / \epsilon$ で頭打ちにする。$\epsilon \gg \sqrt{\hat{v}_{t,i}}$ の極限では、Adam はステップサイズ $(1-\beta_1)\eta/\epsilon$ のモメンタム付き SGD に帰着する。逆に $\beta_2$ が小さすぎると、記憶の短い二次モーメント推定（second moment estimate）が、まれに生じる逆向きの大きな勾配よりも、頻繁に現れる小さな勾配に大きな重みを与えてしまうことがある。:citet:`Reddi.Kale.Kumar.2019` はこの現象を利用して人工的な周期凸反例（$+4, -1, -1$）を構成し、小さな $\beta_2$ を持つ Adam が収束しないことを示した（:numref:`subsec_mdl-adam-counterexample`、演習問題 :exref:`ex_adam-two-repairs-variance-estimate`）。実用上は $\beta_2 \in [0.95, 0.999]$ に保つことで、単調な累積最大値（AMSGrad）を導入しなくてもこの病理を防ぐことができる。

<!--en:faf73fbb-->
## 分離型重み減衰（AdamW）

<!--en:7b574ad8-->
### 結合型 $\ell_2$ 正則化と分離型重み減衰

<!--en:6529a6bf-->
単純な SGD（:numref:`sec_weight_decay`）のもとでは、目的関数に $\ell_2$ ペナルティ $\frac{\lambda}{2}\|\mathbf{x}\|^2$ を加えることは勾配に $\lambda\mathbf{x}_t$ を加えることになり、これは各ステップの前に重みを $(1 - \eta\lambda)$ 倍に縮小（shrinkage）することと数学的に同一である。

<!--en:52fe8f24-->
しかし、ペナルティ付きの勾配 $\mathbf{g}_t + \lambda\mathbf{x}_t$ を Adam に入力すると、ペナルティ項まで座標ごとの前処理行列 $\sqrt{\hat{v}_{t,i}} + \epsilon$ で割られてしまう。

<!--en:35729dec-->
結合型 $\ell_2$ 正則化のもとでは、損失勾配の小さい重みは勾配の大きい重みよりも最大 $10^3\times$ も強く減衰させられる。あるいは、$\lambda|x_{t,i}|$ が損失勾配に対して支配的になると、Adam はペナルティをそれ自身の大きさで正規化し、重みをゼロへ向けてステップあたり速度 $\eta$ で押し縮め、$\lambda|x| \sim |g_{\text{loss}}|$ になるまでそれが続く。

<!--en:0d7dd9bb-->
:citet:`Loshchilov.Hutter.2019` は、重み減衰を適応的な勾配ステップから分離した **AdamW** を導入した。

<!--en:3744beaf-->
ここで、$\hat{\mathbf{m}}_t$ と $\hat{\mathbf{v}}_t$ はデータ損失の勾配 $\mathbf{g}_t$ のみから計算される。

<!--en:1d103323-->
減衰なし、結合型の `Adam + l2`、および分離型の `AdamW` を `TinyLM` 上で比較するため、まず埋め込み以外の 2D 重み行列の要素ごとの RMS を測る `matrix_rms` と、ステップごとの損失および行列の RMS を $250$ ステップごとに記録する `train_tracked` を定義する。

<!--en:7b802673-->
共通のシードから 3 つの設定すべてを $1{,}000$ ステップ、$\eta = 3 \times 10^{-3}$ および $\lambda = 0.1$ で実行すると、結合型 $\ell_2$ ペナルティと分離型重み減衰（decoupled weight decay）の劇的な違いが現れる。

<!--en:8f23c06c-->
結合型の `Adam + l2`（$\lambda = 0.1$）は隠れ層の重み行列を初期 RMS の $0.045$ から $< 0.003$ まで崩壊させ、ネットワークの表現を破壊してユニグラムのバイト頻度エントロピー（$\sim 3.2$ nats/byte）で停滞してしまう。一方、分離型の `AdamW` は正則化なしのベースラインと並んで順調に訓練が進みつつ、重みの増大を抑え込んでいる。

<!--en:c41f0208-->
### 減衰のタイムスケールと平衡ノルム

<!--en:3d808424-->
Adam の正規化された更新方向を $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ と書き、:eqref:`eq_adamw` を $t$ ステップにわたって展開すると次式が得られる。

<!--en:b2eb197f-->
したがって、AdamW はパラメータ $\mathbf{x}_t$ を、減衰係数 $1 - \eta\lambda \approx e^{-\eta\lambda}$ を持つ過去の更新の指数加重移動平均として表しており、その特性**減衰タイムスケール**（decay timescale）は次式で与えられる。

<!--en:b9b66ece-->
単位はステップ数である :cite:`Wang.Aitchison.2025`。$\eta = 3 \times 10^{-3}$、$\lambda = 0.1$ のとき、減衰タイムスケールは $\tau = 1 / 0.0003 \approx 3{,}333$ ステップとなる。$T$ ステップの訓練全体では、重み減衰は $T / \tau = T\eta\lambda$ タイムスケールにわたって作用する。

<!--en:6c1dcbd9-->
$\tau = 1/(\eta\lambda)$ より長い訓練では、初期重みは減衰して消え去り、各座標 $x_{t,i}$ は確率的な**平衡ノルム**（equilibrium norm）に到達する。そこではステップごとの縮小量 $\eta\lambda x_{t,i}$ と正規化された更新 $-\eta n_{t,i}$（ただし $n_{t,i} = g_{t,i}/\sqrt{\hat{v}_{t,i}}$ は二乗平均が単位大きさ $\sigma^2 \approx 1$）の注入とが釣り合う。SGD のノイズ球（noise ball、:numref:`sec_sgd`）において曲率を $\lambda_{\text{decay}}$ に置き換えたものとまったく同じ分散の漸化式により、次式が成り立つ。

<!--en:fc419ecd-->
重みの二乗平均平方根の大きさが $\text{RMS}(\mathbf{x}_\infty) \propto \sqrt{\eta / (2\lambda)}$ に安定することで、正規化された重み行列の**相対角度更新率**（relative angular update rate）も $\eta / \text{RMS}(\mathbf{x}_\infty) \propto \sqrt{2\eta\lambda}$ に安定する（*回転平衡*、:citet:`Kosson.Messmer.Jaggi.2024`）。これにより、正則化なしの重みが制御不能な $\sqrt{t}$ のランダムウォーク成長を起こし、長期の事前学習において実効的な角度ステップサイズが $1/\sqrt{t}$ で減衰してしまう事態が防がれる :cite:`DAngelo.Andriushchenko.Varre.ea.2024`。$\eta$ をスイープする際には、ステップごとの縮小率 $s = \eta\lambda$（したがってタイムスケール $\tau = 1/s$）を固定しておけば、学習率の探索を減衰タイムスケールから切り離すことができる :cite:`Loshchilov.Hutter.2019,Wang.Aitchison.2025,Bergsma.Dey.Gosal.ea.2025b`。

<!--en:04ee3719-->
### 重み減衰の対象外となるパラメータとオプティマイザ状態のメモリ

<!--en:6842756a-->
標準的な Transformer の訓練レシピでは、重み行列に分離型重み減衰を適用し、バイアスと正規化パラメータは対象外とする。OLMo 2 はさらに埋め込みテーブルも対象外としており、本書でもその選択を採用する。その結果、重み減衰は **2D の内部射影行列のみ**（`TinyLM` の $90\%$）に適用され、次の 2 つのパラメータグループは対象外（$\lambda = 0$）となる。

<!--en:834db49d-->
1. **1D のバイアスおよび LayerNorm・RMSNorm のゲイン・シフトベクトル：**正規化のスケールは $1$ 付近から始まり、特徴量の分散を直接制御する。これらをゼロへ向けて減衰させると、角度更新速度を調整するどころか層の活性化値を歪めてしまう。

<!--en:9fba5b40-->
2. **トークン埋め込みテーブルと位置埋め込みテーブル（OLMo 2 での除外）：**トークン埋め込みの各行は、そのトークンがミニバッチに現れたときにしか勾配を受け取らないのに対し、分離型重み減衰は毎ステップすべての行を縮小するため、希少なトークンの表現を崩壊させ、下流の LayerNorm の勾配を膨張させてしまう :cite:`OLMo.2025`。

<!--en:ef05da27-->
### オプティマイザ状態のメモリ

<!--en:22924577-->
最後に、パラメータごとに 2 つのバッファ $(\mathbf{m}_t, \mathbf{v}_t)$ を保持することで、静的なパラメータのメモリ使用量は 2 倍になる。32 ビット浮動小数点（スカラーあたり $4$ バイト）では、訓練可能な各パラメータに **$16$ バイト**のデバイスメモリが必要になる。内訳は、$4$ バイトがパラメータ $\mathbf{x}$、$4$ バイトが勾配 $\mathbf{g}$、そして $8$ バイトが AdamW の状態 $(\mathbf{m}, \mathbf{v})$ である（:numref:`sec_memory_precision`）。パラメータ数が $10^9$ のモデルでは、活性化値を一切保存する前の段階で、オプティマイザ状態だけで $8\text{ GB}$ を消費する。パラメータを凍結する（オプティマイザから除外する）か、$(\mathbf{m}, \mathbf{v})$ を量子化または因子分解することで、このパラメータあたり $8$ バイトのオーバーヘッドを直接解消または圧縮できる。

<!--en:d7363093-->
AdamW は重みの各要素を独立に正規化するが、2D 重み行列は活性化値のベクトルに作用する線形作用素として働く。:numref:`sec_muon` では、各 2D 重み行列の更新を要素ごとではなく作用素全体として正規化することで、さらに優れた結果が得られるかを問う。

<!--en:30ac03ff-->
## まとめ

<!--en:45a7a112-->
**AdaGrad** は各座標を累積二乗勾配の平方根に反比例させてスケーリングし、**RMSProp** はその上界のない和を忘却を伴う指数移動平均 $\mathbf{v}_t$ に置き換え、**Adam** はさらに一次モーメントのモメンタム $\mathbf{m}_t$ とバイアス補正 $1 - \beta_1^t$、$1 - \beta_2^t$ を加える。$\hat{\mathbf{m}}_t$ を $\sqrt{\hat{\mathbf{v}}_t}$ で割ることで、Adam は勾配の対角再スケーリングに対して不変となり、$\eta$ をパラメータ空間における座標ごとの直接的なステップ長へと変え、生の勾配が $15\text{--}40\times$ も異なる Transformer の各パラメータグループ間で更新量の大きさを均一化する。**AdamW** は重み減衰 $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$ を適応的な前処理行列から分離し、一様な減衰タイムスケール $\tau = 1/(\eta\lambda)$、$\sqrt{\eta/(2\lambda)}$ に比例する平衡時の重み RMS、および $\sqrt{2\eta\lambda}$ に比例する定常的な相対角度ステップを確立する。標準的なレシピでは 2D 重み行列のみを減衰させて 1D のバイアスと正規化パラメータ（さらに OLMo 2 に従えば埋め込みテーブル）を対象外とし、$(\mathbf{m}_t, \mathbf{v}_t)$ は `float32` における訓練可能なパラメータあたり $8$ バイト（全 $16$ バイト中）を占める。

<!--en:2294243a-->
## 演習問題

<!--en:8a807910-->
1. [code] **座標回転と対角スケール不変性。**
    1. `adam_2d` を `d2l.f_valley` 上で実行し、目的関数を $c > 0$（ただし $c \gg \epsilon$）で再スケーリングしても、また $\lambda_2$ を $4$ から $400$ に変更しても、Adam が不変であることを検証せよ。
    1. 谷を $45^\circ$ 回転させて $f(\mathbf{x}) = 0.05(x_1 + x_2)^2 + (x_1 - x_2)^2$ とせよ。`adagrad_2d`、`rmsprop_2d`、`adam_2d` を $\eta = 0.4$ で $2{,}000$ ステップにわたり回転前と回転後の谷（および $\lambda_2 = 400$ の場合）で実行し、固有基底を座標軸から回転させると対角前処理行列に対して悪条件性が復活する理由を説明せよ。
   :exlabel:`ex_adam-rotation`

<!--en:5dab189c-->
1. [code] **定学習率のリミットサイクルと AMSGrad。**●
    1. Adam のもとで 1D 座標が $x_t = (-1)^t a$ のように $f(x) = \frac{\lambda}{2}x^2$ 上で交互に振動するとき、その振幅が $a = \frac{1-\beta_1}{1+\beta_1}\frac{\eta}{2}$ となり $\lambda$ に依存しないことを示せ。
    1. `adam_2d` を `f_valley` 上で $20{,}000$ ステップ実行し、反復点が原点の $10^{-40}$ 以内まで近づく（$\hat{v}_t$ が初期の勾配を記憶している間）にもかかわらず、$\hat{v}_t$ がモメンタムの安定性しきい値を下回って減衰すると弾き出される理由を説明せよ。また、$\hat{v}_t$ を $\max_{s \le t} \hat{v}_s$（AMSGrad）に置き換えるとこの弾き出しが防げることを検証せよ。
   :exlabel:`ex_adam-constant-rate`

<!--en:661a48c7-->
1. [extended] **頻出トークン・希少トークンとグループ別学習率。**
    1. :citet:`Kunstner.Yadav.Milligan.ea.2024` に従い、$256$ 個のバイト値を `WikiText103` における出現頻度に基づいて頻出トークン（$\ge 0.1\%$）と希少トークン（$< 0.1\%$）に分割し、チューニングされたモメンタム付き SGD、Adam、およびモメンタム付き符号降下法（$\eta\,\operatorname{sign}(\mathbf{m}_t)$）のもとで各グループの検証損失を比較せよ。
       :exlabel:`ex_adam-frequent-and-rare-tokens`
    1. モメンタム付き SGD のもとで `TinyLM` の 3 つのパラメータグループ（埋め込み、2D 行列、1D ベクトル）に個別の学習率を割り当て、SGD と Adam の差のうちどれだけがグループ単位のスケーリングと座標単位のスケーリングのそれぞれによって縮まるかを測定せよ。
       :exlabel:`ex_adam-per-group-learning-rates`

<!--en:0765ea34-->
1. [code] **敵対的な周期勾配と二次モーメントの修正手法。**周期 3 の系列として、$f_t(x) = 4x$（$t \equiv 1 \pmod 3$ のとき）およびそれ以外では $f_t(x) = -x$ と定義された $x \in [-1, 1]$ 上の関数列において :cite:`Reddi.Kale.Kumar.2019`、$\beta_1 = 0, \beta_2 = 1/17, \eta_t = 0.5/\sqrt{t}$ で Adam を実行し、AMSGrad および Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` と比較せよ。
   :exlabel:`ex_adam-two-repairs-variance-estimate`

<!--en:dc804bc8-->
1. **AdamW の 2 つの極限と回転平衡。**
   :exlabel:`ex_adamw-two-limits`
    1. $x_{t+1} = (1 - \eta\lambda)x_t - \eta n_t$ について、決定論的な平衡 $|x_\infty| = 1/\lambda$（$n_t = -1$ が定数のとき）、平均ゼロのノイズ平衡 $\mathbb{E}[x_\infty^2] \approx \eta/(2\lambda)$（$n_t$ が独立同分布で $\mathbb{E}[n_t]=0, \mathbb{E}[n_t^2]=1$ のとき）、および混合平衡 $\mathbb{E}[x_\infty^2] \approx \bar{n}^2/\lambda^2 + \eta/(2\lambda)$（$n_t = \bar{n} + \xi_t$ のとき）を導出せよ。
    1. $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\mathbf{u}_t$ を $\mathbb{R}^{1000}$ 上で $\mathbf{u}_t \perp \mathbf{x}_t$ の条件のもとでシミュレーションし :cite:`Kosson.Messmer.Jaggi.2024`、平衡時の RMS が $\sqrt{\eta/(2\lambda)}$ に一致し、ステップごとの回転角が $\sqrt{2\eta\lambda}$ に一致することを検証せよ。

<!--en:6ca9574e-->
1. [code] **$(\eta, \lambda)$ の同時探索とパラメータグループの除外。**●
    1. `TinyLM` を $8{,}000$ 個の重複ウィンドウ（`d2l.TimeMachine`）上で $800$ ステップにわたり、$\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$ と $\lambda$ のグリッド全体で訓練し、$\lambda$ によるパラメータ化とステップごとの縮小率 $s = \eta\lambda$ によるパラメータ化を比較せよ。
       :exlabel:`ex_adamw-joint-search`
    1. `WikiText103` 上の検証パープレキシティを $\lambda \in \{0.5, 1.0\}$ において、すべてのパラメータを減衰させる場合と 2D 重み行列のみを減衰させる場合（:numref:`subsec_adamw-what-not-to-decay`）とで比較し、改善の大部分をもたらしているのが埋め込みの除外なのか 1D ベクトルの除外なのかを特定せよ。
       :exlabel:`ex_adamw-exemptions`

<!--en:0b71c54f-->
[ディスカッション](https://d2l.discourse.group/t/1078)

<!--en:d5ba59f0-->
[ディスカッション](https://d2l.discourse.group/t/1079)

<!--en:2ecbc1b4-->
適応的学習率手法：AdaGrad、RMSProp、Adam、AdamW<br>**座標ごとの RMS スケーリング・バイアス補正・ステップ長の不変性・分離型重み減衰・タイムスケールと平衡ノルム**

<!--en:97a0d94f-->
::: {.slide title="なぜ Transformer には座標ごとの学習率が必要か"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:08656d77-->
- **層間の 40× の勾配格差：**Transformer では、トークン埋め込み、アテンション射影、MLP 層、LayerNorm パラメータの間で生の勾配の大きさが $15\text{--}40\times$ 異なり、スカラーの SGD ステップサイズでは埋め込みの学習が進まない

<!--en:93b5e1ac-->
- **適応的 RMS 前処理：**各座標の更新量を過去の RMS 勾配スケールで割ることで、$\eta$ は**パラメータ空間における座標ごとの直接的なステップ長**となり（$\eta^\star \approx 3 \times 10^{-3}$、SGD では $0.5$）、**重み減衰の分離**（AdamW）が必要になる

<!--en:3b357688-->
::: {.slide title="AdaGrad：累積二乗勾配による前処理"}

<!--en:a292733c-->
[座標ごとのスケーリング]{.kicker}

<!--en:b6dc4e9f-->
**AdaGrad**（Duchi et al., 2011）は二乗勾配の要素ごとの累積和 $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ を $\mathbf{s}_0 = \mathbf{0}$ から蓄え、各座標を $\sqrt{\mathbf{s}_t + \epsilon}$ に反比例させてスケーリングする

<!--en:765af9e7-->
- **希少な特徴量と頻出の特徴量：**更新頻度の低い座標（希少なトークン埋め込みなど）は分母 $\sqrt{s_{t,i}}$ が小さく保たれ、その特徴量が現れたときに十分なステップを踏む

<!--en:78de294a-->
- **座標軸に沿った自動前処理：**$|g_i| = \lambda_i |x_i|$ は曲率に比例するため、$\sqrt{s_{t,i}}$ で割ることで谷における $\kappa = 20$ という $x_1$ と $x_2$ の間の格差が打ち消される

<!--en:876d2768-->
::: {.slide title="RMSProp：忘却を伴う二乗勾配の移動平均"}

<!--en:dd0f2e3d-->
[古い勾配の忘却]{.kicker}

<!--en:0e4e33d1-->
- **深層ネットワークで AdaGrad が停滞する理由：**$\mathbf{s}_t = \sum_{s=1}^t \mathbf{g}_s^2$ が上界なく単調増加する（$\mathbf{s}_t \propto t$）ため、AdaGrad の実効ステップサイズは $O(t^{-1/2})$ で減衰し、学習が早期に止まる

<!--en:147142ef-->
- **RMSProp**（Tieleman & Hinton, 2012）は累積和を、記憶ウィンドウ $\tau_2 = \frac{1}{1-\beta_2}$ ステップにわたる指数移動平均（EMA）に置き換える
  $$\mathbf{v}_t = \beta_2\,\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2$$

<!--en:b6acf52a-->
- 古い勾配を忘却することで（$\beta_2 = 0.9 \implies 10$ ステップのウィンドウ）、RMSProp は消失しないステップサイズを維持し、$\eta = 0.4$ で谷の上を急速に収束する

<!--en:06425692-->
::: {.slide title="補正なし RMS 正規化がもたらす 2 つの帰結"}

<!--en:e6a92478-->
[初期のジャンプと終盤の振動]{.kicker}

<!--en:88ac056d-->
- **1. コールドスタート時のステップ増幅：**$\mathbf{v}_0 = \mathbf{0}$ から始まるステップ 1 では $\mathbf{v}_1 = (1-\beta_2)\mathbf{g}_1^2$ となり、過大なジャンプ $|\Delta x_{1,i}| = \frac{\eta}{\sqrt{1-\beta_2}}$（$3.16\,\eta$、$\beta_2=0.9$ の場合。**$31.6\,\eta$**、$\beta_2=0.999$ の場合）が生じる

<!--en:cca5b714-->
- **2. 終盤でも消失しないステップ：**$|g_t|$ が局所的にほぼ一定となる極小値の近傍では $\sqrt{v_t} \approx |g_t|$ となり、更新は $-\eta\,\operatorname{sign}(g_t)$（固定長 $\eta$）となって、$\pm\eta/2$ の範囲内で飛び跳ね続け、$\eta_t \to 0$ になるまで収まらない（§{sec_scheduler}）

<!--en:24d3e5ec-->
::: {.slide title="Adam：バイアス補正を伴う一次・二次モーメント"}

<!--en:f96f640d-->
[適応的モーメント推定]{.kicker}

<!--en:c1f6dae8-->
- **2 つの移動平均（$\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$）：**
  $$\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$

<!--en:4943e33f-->
- **バイアス補正付きの更新：**
  $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}, \quad \mathbf{x}_{t+1} = \mathbf{x}_t - \eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$$

<!--en:229f5793-->
- $1 - \beta^t$ で割ることで、すべてのステップ $t \ge 1$ において幾何重みを正規化する（デフォルト値：$\beta_1 = 0.9, \beta_2 \in \{0.95, 0.999\}, \epsilon = 10^{-8}$）

<!--en:d42b4cf0-->
::: {.slide title="スケール不変性：η は座標ごとの直接的なステップ長"}

<!--en:ec0cc5fe-->
[Adam の幾何学]{.kicker}

<!--en:5d1ad436-->
- **厳密なステップ 1 の大きさ：**$t = 1$ では $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ かつ $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$ となるため、$\sqrt{\hat{v}_{1,i}} \gg \epsilon$ に対して
  $$\Delta x_{1,i} = -\eta\,\frac{g_{1,i}}{|g_{1,i}|} = -\eta\,\operatorname{sign}(g_{1,i})$$
  すべての座標が **$\pm\eta$** だけステップ 1 で移動する（谷では $45^\circ$ の対角方向）

<!--en:e58d319f-->
- **対角スケール不変性：**損失に定数を掛けるか座標 $i$ を $c > 0$ で再スケーリングすると、$g_{t,i} \to c\,g_{t,i}$、$\hat{m}_{t,i} \to c\,\hat{m}_{t,i}$、$\sqrt{\hat{v}_{t,i}} \to c\sqrt{\hat{v}_{t,i}}$ と変化し、比から $c$ が相殺される

<!--en:1dadfca4-->
- **ステップ長の範囲：**持続的な勾配では $\approx \eta$、平均ゼロのノイズでは $\approx \eta\sqrt{\frac{1-\beta_1}{1+\beta_1}} \approx 0.23\,\eta$ だけ進む

<!--en:f8055ce6-->
::: {.slide title="ゼロからの Adam の実装"}

<!--en:a4fbb6b9-->
[コードによる実装]{.kicker}

<!--en:ff77a51e-->
訓練可能な各パラメータテンソルは、ステップカウンタ $t$ と同じ形状の 2 つのバッファ $(\mathbf{m}, \mathbf{v})$ を保持する

<!--en:645d6589-->
::: {.slide title="40× の勾配格差と 2× の Adam ステップ差"}

<!--en:b4ea92bb-->
[TinyLM における勾配スケール]{.kicker}

<!--en:d0ff0f1d-->
- $10$ ミニバッチにわたる RMS 勾配の大きさを、全 $8$ 種類の機能別パラメータグループについて `TinyLM` 上で測定すると、スカラー学習率の SGD が苦戦する理由がわかる
  - **ステップ 0：**生の勾配 RMS は **$40\times$ の範囲**に及ぶ（$2 \times 10^{-4}$ の `token_emb` から $7 \times 10^{-3}$ の `biases` まで）
  - **ステップ 500：**勾配 RMS は依然として **$15\times$ の範囲**に広がる

<!--en:7d8bb88a-->
- **Adam による均一化：**$\sqrt{\hat{\mathbf{v}}_t}$ で割ることで、実際の RMS パラメータ更新 $\|\Delta\mathbf{x}\|_{\text{RMS}}/\eta$ は全 $8$ グループにわたって狭い **$2\times$ の範囲（$0.12\text{--}0.28\,\eta$）**に圧縮される

<!--en:500c43fe-->
::: {.slide title="学習率スイープ：Adam とモメンタム付き SGD" layout="tall-output"}

<!--en:529d752a-->
[3 シードのグリッド比較]{.kicker}

<!--en:2cf95604-->
- モメンタム付き SGD（$\beta = 0.9$）を $\eta \in \{0.1, 0.3, 0.5, 1.0\}$、Adam を $\eta \in \{3{\times}10^{-4}, 10^{-3}, 3{\times}10^{-3}, 10^{-2}, 3{\times}10^{-2}\}$ にわたり $3$ シードでスイープする（$2{,}000$ ステップ、`TinyLM`）
  - **100× の公称スケール差：**SGD の最良学習率は $\eta = 0.3$ である一方、Adam は $\eta = 3 \times 10^{-3}$ である（$\approx 0.1 \times \frac{0.3}{\text{scale}}$）
  - **高学習率側での緩やかな性能低下：**SGD は $\eta = 0.5$（最適値の $2\times$ 未満）で大半が `inf` に発散するが、Adam は $\eta = 3 \times 10^{-2}$（最適値の $10\times$）でもステップ長が $\eta$ で抑えられるため有限に留まる

<!--en:4c7a976b-->
::: {.slide title="Adam は SGD の半分のステップ数で目標損失に到達"}

<!--en:511a36d4-->
[収束速度・範囲・安定性]{.kicker}

<!--en:9a0443fd-->
- 各オプティマイザの最良学習率における $3$ 回の実行を比較すると、`TinyLM` 上で Adam には 3 つの決定的な利点が見られる
  1. **収束速度：**$1.8$ nats に **$650\text{--}775$ ステップ**で到達（モメンタム付き SGD は $1{,}125\text{--}1{,}550+$ ステップ）
  2. **実用可能な学習率ウィンドウ（$+0.15$ nats 以内）：**Adam は **$10\times$ の範囲**（$10^{-3}$〜$10^{-2}$）であるのに対し、SGD は 1〜2 グリッド点（$0.1$〜$0.3$）
  3. **スパイクへの耐性：**1 ステップでの最悪の損失上昇は Adam では全実行で約 $0.2$ nats に留まるのに対し、SGD では最大 $1\text{--}1.7$ nats

<!--en:9003682e-->
::: {.slide title="符号降下法・ε の役割・小さい β₂ の病理"}

<!--en:6b3e6a18-->
[動作原理と境界ケース]{.kicker}

<!--en:2bef4448-->
- **平滑化された符号降下法としての Adam：**$\beta_1 = \beta_2 = 0, \epsilon = 0$ とおくと $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\operatorname{sign}(\mathbf{g}_t)$ となり、これにモメンタムを加えると Transformer における SGD に対する Adam の優位性が再現される（Kunstner et al., 2023）

<!--en:2a04100f-->
- **最大学習率の上限としての $\epsilon$：**$\frac{\eta}{\sqrt{\hat{v}_{t,i}} + \epsilon} \le \frac{\eta}{\epsilon}$ であり、$\epsilon \gg \sqrt{\hat{v}_{t,i}}$ では Adam は学習率 $(1-\beta_1)\eta/\epsilon$ のモメンタム付き SGD に帰着する

<!--en:99be6189-->
- **$\beta_2$ を小さくしすぎてはならない理由（Reddi et al., 2018）：**記憶の短い $\beta_2$ では、まれな大勾配が $\sqrt{v_t}$ を膨らませる一方で逆向きの小勾配が大きく進んでしまうが、$\beta_2 \in [0.95, 0.999]$ に保てば防げる

<!--en:7907a72e-->
::: {.slide title="なぜ標準の ℓ₂ 正則化は Adam で破綻するか"}

<!--en:03107fc6-->
[結合型減衰と分離型減衰]{.kicker}

<!--en:c893ee44-->
- **Adam 内部での結合型 $\ell_2$ ペナルティ（$\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda\mathbf{x}_t$）：**減衰ペナルティが各座標の前処理行列で割られる
  $$\Delta x_{t,i}^{\text{decay}} = -\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; x_{t,i}$$

<!--en:fa580a0d-->
- **病理：**損失勾配の小さい重みは活発な重みより最大 $10^3\times$ も強く減衰されるか、ペナルティがそれ自身の大きさで正規化されて $x_{t,i} \to 0$ へとステップあたり速度 $\eta$ で崩壊する

<!--en:6271520a-->
::: {.slide title="AdamW：前処理行列からの重み減衰の分離"}

<!--en:f660ab7a-->
[Loshchilov & Hutter (2019)]{.kicker}

<!--en:9cc0fb60-->
**AdamW** は $\hat{\mathbf{m}}_t$ と $\hat{\mathbf{v}}_t$ をデータ損失の勾配 $\mathbf{g}_t$ のみから計算し、一様な乗法的縮小 $(1 - \eta\lambda)$ を重みに直接適用する

<!--en:cc222010-->
::: {.slide title="結合型 ℓ₂ は重み行列を崩壊させ、AdamW は訓練を進める"}

<!--en:009ce442-->
[λ = 0.1 における 1,000 ステップ比較]{.kicker}

<!--en:7dc766a7-->
- `TinyLM` を $1{,}000$ ステップ、$\eta = 3 \times 10^{-3}$ および $\lambda = 0.1$ で同一の初期重み（初期の行列 RMS $= 0.0447$）から訓練
  - **減衰なし（$\lambda = 0$）：**最終損失 **$\approx 1.74$ nats**、行列 RMS は $0.108\text{--}0.110$ まで増大
  - **分離型 `AdamW`（$\lambda = 0.1$）：**最終損失 **$1.72\text{--}1.74$ nats**、行列 RMS は $0.097\text{--}0.100$ に抑制
  - **結合型 `Adam + ℓ2`（$\lambda = 0.1$）：**行列 RMS が $16\times$ 縮小して **$0.0027\text{--}0.0028$** まで崩壊し、内部表現が失われてモデルは **$3.15\text{--}3.16$ nats** で停滞（ユニグラムのバイト頻度エントロピーは $3.18$）

<!--en:0c147f33-->
::: {.slide title="AdamW の展開：減衰タイムスケール τ = 1/(ηλ)"}

<!--en:fee52234-->
[更新の EMA としての重み]{.kicker}

<!--en:868cda0b-->
- **正規化された更新 $\mathbf{u}_s = \frac{\hat{\mathbf{m}}_s}{\sqrt{\hat{\mathbf{v}}_s}+\epsilon}$ の展開された EMA（Wang & Aitchison, 2025）：**
  $$\mathbf{x}_t = (1 - \eta\lambda)^t\,\mathbf{x}_0 - \eta\sum_{s=0}^{t-1} (1 - \eta\lambda)^{t-1-s}\,\mathbf{u}_s$$

<!--en:6e56b88b-->
- **重み減衰のタイムスケール：**$\tau = \frac{1}{\eta\lambda}$ ステップ（$3{,}333$ ステップ：$\eta = 3 \times 10^{-3}, \lambda = 0.1$ のとき）

<!--en:57162f92-->
- **ハイパーパラメータの分離：**$\eta$ をスイープする際は、ステップごとの縮小率 $s = \eta\lambda = 1/\tau$ を固定する（$\lambda = s/\eta$）

<!--en:97489e78-->
::: {.slide title="平衡時の重みノルムと回転平衡"}

<!--en:0c89e46d-->
[定常状態の重みダイナミクス]{.kicker}

<!--en:05498d7d-->
- **確率的なノルムの平衡（$t \gtrsim 2\tau$）：**減衰 $(1-\eta\lambda)x_t$ と単位 RMS のステップ $-\eta n_t$ の釣り合いから
  $$\text{RMS}(\mathbf{x}_\infty) \approx \sqrt{\frac{\eta}{2\lambda}}$$

<!--en:6f0e8879-->
- **$\lambda = 0$ が角度方向の学習を遅らせる理由：**減衰がないと $\text{RMS}(\mathbf{x}_t) \propto \eta\sqrt{t}$ がランダムウォークで増大し、スケール不変な LayerNorm サブ層における相対角度ステップが $1/\sqrt{t}$ で縮小する

<!--en:1bac190d-->
- **回転平衡（Kosson et al., 2024）：**AdamW のもとでは、$\frac{\eta}{\text{RMS}(\mathbf{x}_\infty)} \approx \sqrt{2\eta\lambda}$ が層によらず一定に保たれる

<!--en:b124f4ce-->
::: {.slide title="重み減衰の対象外となるパラメータと状態メモリ"}

<!--en:e0039170-->
[パラメータグループと 16 バイトの使用量]{.kicker}

<!--en:a1eb5ac6-->
- **2D の内部射影行列のみを減衰（`TinyLM` の $90\%$）：**1 つ目の除外は標準的であり、2 つ目は OLMo 2 に従う
  - **1D のバイアスおよび LayerNorm のスケール・シフトを除外（$\lambda = 0$）：**LayerNorm のゲインは特徴量の分散を $1.0$ 付近で直接制御するため、減衰させると活性化値が歪む
  - **トークン埋め込みと位置埋め込みを除外（$\lambda = 0$）：**希少なトークンの行はサンプリングされたときにしか勾配を受け取らないため、毎ステップの減衰は希少な埋め込みを崩壊させ、下流の LayerNorm の勾配を膨張させる（OLMo 2, 2025）

<!--en:9a600dad-->
- **`float32` ではパラメータあたり 16 バイト：**重み（$4\text{ B}$）$+$ 勾配（$4\text{ B}$）$+$ AdamW の状態 $\mathbf{m}, \mathbf{v}$（$8\text{ B}$）$= 16\text{ B}$（$8\text{ GB}$ をオプティマイザ状態だけで占める［$1\text{B}$ パラメータモデルの場合］）

<!--en:c011b1f1-->
::: {.slide title="概念チェック：バイアス補正・タイムスケール・VRAM"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:4ef5ca81-->
3 つの具体的な計算で理解度を確かめる

<!--en:bf140907-->
1. **バイアス補正の欠落：**$\beta_1 = 0.9, \beta_2 = 0.999, \eta = 10^{-3}$ の Adam をバイアス補正*なし*で実行した場合、パラメータ更新の大きさ $|\Delta x_{1,i}|$ はステップ $t = 1$ において補正ありの Adam と比べてどうなるか。

<!--en:090372a6-->
2. **タイムスケールと相対ステップの不変性：**ある実行では $\eta = 3 \times 10^{-3}$、$\lambda = 0.5$ で AdamW を用いる。学習率を半分の $\eta = 1.5 \times 10^{-3}$ にし、重み減衰を 2 倍の $\lambda = 1.0$ にしたとき、減衰タイムスケール $\tau$、平衡時の重み RMS、および相対角度ステップ $\eta / \text{RMS}$ はどう変化するか。

<!--en:14df45e0-->
3. **VRAM の計算：**70 億（$7$ billion）パラメータの Transformer を、マスター重み、勾配、$(\mathbf{m}, \mathbf{v})$ をすべて `float32` に保った AdamW で訓練する。活性化値を除くこれらの静的バッファは何 GB を占めるか。

<!--en:557b98b6-->
**解答：**

<!--en:977d5435-->
1. バイアス補正なしでは、$|\Delta x_{1,i}| = \eta\frac{1 - \beta_1}{\sqrt{1 - \beta_2}} = \eta\frac{0.1}{\sqrt{0.001}} \approx$ **$3.16\,\eta = 3.16 \times 10^{-3}$** となる（$3.16\times$ の大きさであり、補正ありの Adam の厳密なステップ長は $\eta = 10^{-3}$）。

<!--en:692bef24-->
2. $\eta\lambda = 1.5 \times 10^{-3}$ が不変であるため、**タイムスケール $\tau = \frac{1}{\eta\lambda} \approx 667$ ステップ**と**相対角度ステップ $\sqrt{2\eta\lambda} \approx 0.055$** はともに**変化しない**一方、平衡ノルム $\sqrt{\eta/(2\lambda)}$ は**半分になる**（スケール不変な LayerNorm サブ層はこの変化の影響を受けない）。

<!--en:a16c6e5f-->
3. $16\text{ bytes/parameter}$ より、静的メモリは $7 \times 10^9 \times 16\text{ B} =$ **$112\text{ GB}$** であり、そのうち **$56\text{ GB}$（$50\%$）**を AdamW の状態 $(\mathbf{m}, \mathbf{v})$ が占める。

<!--en:08a2fded-->
::: {.slide title="まとめ：座標ごとの適応的手法と AdamW"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:4d246ed8-->
| オプティマイザ・概念 | 更新式・不変量 | メモリ・タイムスケール | 深層学習における役割 |

<!--en:4cda9446-->
| **AdaGrad** | $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,\; \Delta\mathbf{x}_t = -\frac{\eta}{\sqrt{\mathbf{s}_t+\epsilon}}\odot\mathbf{g}_t$ | バッファ $1$ 個（パラメータあたり $4\text{ B}$）、ステップは $O(t^{-1/2})$ で減衰 | 疎な凸特徴量向け。深い非凸ネットワークでは停滞 |

<!--en:94c79ca8-->
| **RMSProp** | $\mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$ | バッファ $1$ 個、スケールウィンドウ $\tau_2 = \frac{1}{1-\beta_2}$ | 消失しない適応的ステップ。モメンタムとバイアス補正を欠く |

<!--en:4d3c4bd8-->
| **Adam** | $\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},\; \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t},\; \Delta\mathbf{x}_t = -\eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$ | バッファ $2$ 個（パラメータあたり $8\text{ B}$）、$\tau_1 = 10, \tau_2 = 1{,}000$ | 対角再スケーリングに不変。層間の $40\times$ の勾配格差を均一化 |

<!--en:38b6f7c4-->
| **AdamW** | $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$ | 減衰タイムスケール $\tau = \frac{1}{\eta\lambda}$ | $\sqrt{\hat{v}_t}$ によるスケーリングを回避。2D 重み行列のみを減衰 |

<!--en:b1d54e68-->
| **平衡ノルム** | $\text{RMS}(\mathbf{x}_\infty) \approx \sqrt{\frac{\eta}{2\lambda}},\quad \frac{\eta}{\text{RMS}(\mathbf{x}_\infty)} \approx \sqrt{2\eta\lambda}$ | $\sim 2\tau = \frac{2}{\eta\lambda}$ ステップ後に収束 | $\sqrt{t}$ の重み増大と $1/\sqrt{t}$ の実効学習率減衰を防止 |
