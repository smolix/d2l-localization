<!--en:0962c676-->
# 積分によるサンプリング

<!--en:86042f8f-->
ガウス経路（Gaussian path）に沿って訓練されたネットワークは、すべてのノイズレベル $t \in [0, 1]$ において周辺速度（marginal velocity）$\mathbf{u}_t(\mathbf{x})$（あるいは同値なスコア $\nabla \log p_t(\mathbf{x})$）を予測する。この決定論的なベクトル場を $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（$t = 1$）から $t = 0$ まで逆向きに積分すると、なぜ $p_0$ に従うサンプルが、固定レベルのランジュバン連鎖が見逃した正確なモード重み $0.5, 0.3, 0.2$ を伴って生成されるのだろうか。

<!--en:a7f344f8-->
その保証は、確率輸送の*連続の式*（continuity equation）からもたらされる。各条件付きガウス分布の雲 $p_t(\mathbf{x} \mid \mathbf{x}_0)$ はそれぞれの条件付き速度（conditional velocity）のもとで質量保存則を満たすため、$p(\mathbf{x}_0)$ について平均をとれば、周辺密度 $p_t$ が事後平均速度 $\mathbf{u}_t(\mathbf{x})$ によって輸送されることがわかる。:citet:`song2021score` はこの*確率フロー ODE*（probability-flow ODE）による決定論的サンプリングを定式化し、一方で :citet:`ho2020denoising` は確率的な逆マルコフ連鎖によってサンプリングした。後に示すように、DDPM の各祖先ステップは、確率フロー ODE のオイラーステップと現在のノイズレベルにおけるランジュバンステップの和へと一次近似で分解される。本節では、これら両方のサンプラを導出し、経路に依存しない $(\bar{\mathbf{x}}, \bar\sigma)$ 座標系で ODE を表したうえで、3 モード混合分布において訓練済みネットワークを厳密なオラクルと比較する。

<!--en:ee3537de-->
各節は独立したノートブックとして動作するため、下のセルでは :numref:`sec_diffusion-gaussian-paths` から必要な要素、すなわち 3 モード混合分布とその 2 つの経路である分散保持（variance-preserving、VP）経路および線形経路を再構築する。また、前節と同様に、任意の点とノイズレベルにおける混合分布の厳密な事後平均 $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ を `oracle` という関数名で定義する。さらに、本節で扱う 2 つのサンプラに対応して 2 つのネットワークを再訓練する。1 つはフロー用の線形経路上の速度モデルであり、もう 1 つは DDPM の連鎖用の VP 経路上のノイズモデルである。訓練レシピは前節のステップ数を半分にしたもので、2 万ステップで十分である。これを確認するため、直近 1,000 ステップの平均訓練損失を表示し、前節で求めた下限（速度では約 $4.9$、ノイズでは $0.54$）と比較する。2 つの訓練を合わせても、1 基の GPU で約 1 分で完了する。

<!--en:3322648f-->
どちらの損失も下限に達しており、いずれのネットワークにも回帰からこれ以上学習すべきことはほとんど残っていない。肝心のサンプルにおいてこれらのネットワークがどれだけオラクルに迫れるかは、本節の後半で検証する。

<!--en:f4c9350a-->
## 確率質量の輸送

<!--en:c111e835-->
まず、速度場が密度に何をもたらすかから始めよう。$p_t$ に従って分布する点の雲を考え、各点をその位置における速度 $\mathbf{u}_t(\mathbf{x})$ で微小時間だけ動かしてみる。点は生成も消滅もしないため、$\mathbf{x}$ における密度の変化は、流入する質量から流出する質量を引いた差によってのみ生じる。単位時間あたりに小さな窓を通過する質量は、密度と速度の積 $p_t \mathbf{u}_t$ である。この流れが $\mathbf{x}$ の周囲の微小な箱から単位体積あたりに正味で運び出す量は、その発散 $\nabla \cdot (p_t \mathbf{u}_t) = \sum_i \partial (p_t u_{t,i}) / \partial x_i$ であり、その場所の密度はまさにその割合で減少する。これが流体力学における*連続の式*である。

<!--en:43d1a915-->
密度と速度場がこの式を満たすとき、場に従って動く点は時間を順方向に進めても逆方向に進めても、すべての時刻で経路の密度を保つ。では、:numref:`sec_diffusion-gaussian-paths` の周辺速度は $p_t$ とともにこの式を満たすだろうか。

<!--en:6d8a6d00-->
まず 1 つのクリーンな点 $\mathbf{x}_0$ を考えよう。これにノイズを加えた点はガウス分布の雲 $\mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$ をなし、雲の各点は条件付き速度 $\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon$ で動く。ここで $\boldsymbol\epsilon = (\mathbf{x} - \alpha_t \mathbf{x}_0)/\sigma_t$ は、その点を位置 $\mathbf{x}$ に置いたノイズである。この場を $\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_0)$ と呼ぶことにすると、これはこの 1 つの雲を正しく動かす。すなわち、同じノイズ $\boldsymbol\epsilon$ を保ったまま動く点は各時刻で $\alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ に位置し、常にその雲からのサンプルとなっているため、この雲とその速度場は連続の式を満たす。周辺密度 $p_t$ はデータ全体にわたる雲の平均であり、連続の式は密度に関して線形であるため、各雲の式を平均すると、右辺に平均質量流を持つ $p_t$ の式が得られる。ベイズの定理 $p(\mathbf{x}_0)\, p_t(\mathbf{x} \mid \mathbf{x}_0) = p_t(\mathbf{x})\, p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})$ により、この流れは次のようになる。

<!--en:bc73525b-->
途中に現れた事後分布による平均は、周辺速度の定義そのものである。したがって、すべての雲が合わせて $\mathbf{x}$ を横切って運ぶ質量は、その位置の密度と周辺速度の積に等しく、$p_t$ と $\mathbf{u}_t$ は連続の式を満たす。次節以降のサンプラはこの結果に基づいている。必要な条件を含めた厳密な証明は :numref:`sec_mdl-flow-matching-transport` で与える。

<!--en:3115aed0-->
速度を平均しても軌跡の平均にはならないではないか、と疑問に思う読者もいるだろう。その指摘は正しい。条件付き経路（線形経路では直線）はあらゆる時刻であらゆる点を通過し、いたるところで互いに交差する。これに対し、1 つのベクトル場の 2 本の軌跡が同じ時刻に交わることは決してない。ある時刻のある点における速度はただ 1 つしか存在しないからである。$\mathbf{u}_t$ に従うサンプルは、いかなる条件付き経路もたどらない。:numref:`fig_diffusion-crossing-paths` に、1 次元の 2 成分混合ガウス分布に対する両方の曲線を示す。オレンジ色で描かれたノイズは右側のモードにあるクリーンな点から生じたものだが、その軌跡は左側のモードに行き着く。上の議論は個々の軌跡については何も主張していない。主張しているのは、この速度場があらゆる時刻で*密度*を再現するということであり、サンプラに必要なのはその密度だけである。

<!--en:90fccf45-->
![1 次元の 2 成分混合ガウス分布における、線形経路上の条件付き経路と軌跡。時間は $t = 0$ のデータ（下）から $t = 1$ のノイズ（上）へと進み、陰影は密度 $p_t$ を表す。(a) クリーンな点 $\mathbf{x}_0$（バツ印）とノイズ $\boldsymbol\epsilon$（点）の 9 組のペアが作る条件付き経路は直線であり、互いに交差する。(b) 同じ 9 個のノイズから出発する周辺速度場 $\mathbf{u}_t$ の軌跡は決して交わらない。オレンジ色のノイズは、条件付き経路に沿えば右側のモードに行き着くが、軌跡に沿えば左側のモードに行き着く。](../img/mdl-diffusion-crossing-paths.svg)

<!--en:d2dd39ce-->
これで冒頭の問いに答えが出た。厳密に解けば、フローはあらゆる瞬間で質量を保存するため、モード重みを誤ることはあり得ない。ノイズの半分は必ず重み 1/2 のモードに行き着く。ランジュバン連鎖は、一度データ分布に到達すればそれを定常に保てるものの、谷を越えてそこへ運ぶ仕組みを持たなかった。アニーリングラダーは、ノイズが谷を埋める粗いレベルで谷を越えたが、必要なレベル数やステップ数は手探りで決めるしかなかった。これに対し、フローは既知の密度である標準ガウス分布から出発し、理論的な保証をもってデータへと輸送する。それでも問題が生じ得る箇所は 2 つ残っている。離散ステップは解を近似するにすぎず、ネットワークも速度場を近似するにすぎない点である。本節の残りでは、この両方の誤差を測定する。

<!--en:86fa4127-->
以上より、このサンプラは初期点 $\mathbf{x}$ を $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ から抽出し、次の常微分方程式（ordinary differential equation、ODE）を解く。

<!--en:62728969-->
すべての時刻においてその解は $p_t$ からのサンプルであり、$t = 0$ ではデータからのサンプルとなる。:citet:`song2021score` はこれを*確率フロー ODE*（:numref:`sec_mdl-probability-flow-ode`）と呼んだ。出発後にランダムな要素は一切入らず、サンプルのランダム性はすべてその初期点に集約されている。

<!--en:8090c07d-->
## 確率フロー ODE に対するオイラー法

<!--en:5d08d962-->
ODE は、小さな直線ステップを数多く積み重ねることで数値的に解かれる。*オイラー法*（Euler's method、:numref:`sec_mdl-euler-runge-kutta`）は、時刻 $1$ から $0$ までを $K$ 個の等間隔なステップに分割し、各ステップで点を速度と逆向きに $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_t(\mathbf{x})/K$（$t = 1, 1 - 1/K, \ldots, 1/K$）と動かす。では、フローには何ステップ必要なのだろうか。一般のデータについて直接考えるのは難しいため、:citet:`Nakkiran.Bradley.Zhou.ea.2025` にならい、答えが自明なデータから始めて段階的に議論を組み立てていこう。

<!--en:95c35aa1-->
### 1 点のデータからデータ平均へ

<!--en:4e3536d0-->
データが 1 つの点 $\mathbf{x}_0$ のみからなる場合を考える。線形経路（$\alpha_t = 1 - t$、$\sigma_t = t$）上では、ある点 $\mathbf{x}$ は時刻 $t$ においてノイズ $\boldsymbol\epsilon = (\mathbf{x} - (1 - t)\mathbf{x}_0)/t$ を持ち、速度 $\boldsymbol\epsilon - \mathbf{x}_0 = (\mathbf{x} - \mathbf{x}_0)/t$ で動く。すべての軌跡は一定速度で $\mathbf{x}_0$ へと向かう直線となり、長さ 1 のオイラーステップ 1 回、すなわち $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_1(\mathbf{x}) = \mathbf{x}_0$ で厳密な解が得られる。データ点が 2 つになると、速度場はそのような 2 つの直線的な速度場を事後確率で重み付け平均したものとなり、事後分布が一方の点から他方の点へと切り替わる場所で軌跡が曲がる（:numref:`subsec_mdl-fm-two-point`）。

<!--en:9efebc2a-->
任意の分布は点質量の混合分布とみなせるため、事後分布がいずれの点とも決まらない場所では必ず軌跡が曲がる。$t = 1$ においては、事後分布は完全に未決定である。純粋なノイズの点からは元のクリーンな点について何の手がかりも得られないため、事後分布は事前分布に一致し、次の関係が成り立つ。

<!--en:bfa66923-->
したがって、長さ 1 のオイラーステップを 1 回行うと、どの初期点もデータの平均へと送られる。本節の混合分布については、コードを実行する前から結果を予測できる。すなわち $0.5\,(-2.5, -1.5) + 0.3\,(2.5, -1.5) + 0.2\,(0, 2.5) = (-0.5, -0.7)$ であり、これは各モードの間の、最も近いモードからも標準偏差の 4 倍以上離れた点である。ステップ数を増やすとどうなるかを見るため、線形経路上の混合分布の厳密な速度場 `mix.velocity` に対し、原点および原点から距離 $1.5$ にある 4 点の計 5 個の固定初期点から、1、10、100 ステップでオイラー法を実行してみよう。

<!--en:29bfa2e1-->
1 ステップの列は予測どおりであり、5 つの初期点すべてがデータの存在しない $(-0.50, -0.70)$ に着地している。10 ステップになると、すべての初期点がすでにいずれかのモードの内部に入り、100 ステップの終点から約 $0.2$ 以内の位置に達する。ただし、いずれもデータ平均に近い手前側にとどまっており、粗いステップには 1 ステップの誤差がわずかに残っている。次に、各初期点が*どの*モードに到達するかに注目しよう。原点、その下の初期点、および左の初期点はすべて最も重みの大きいモードに行き着き、右の初期点は右側のモード、上の初期点は上側のモードに行き着く。ここに質量保存則が働いている。標準ガウス分布の半分は重み 1/2 のモードに行き着かなければならないため、そのモードが最も広い初期点の領域（ここではガウス分布の中心を含む）を集めるのである。また、終点がモードの中心に集中するわけでもない。原点は $(-1.98, -1.30)$ に行き着くが、これはモードの中心からモードの標準偏差で約 1 つ分離れた位置である。フローはノイズからデータへの写像であり、それぞれの初期点が固有のサンプルに対応している。

<!--en:b4103b81-->
### すべての経路に共通する座標系

<!--en:8f2bb8b1-->
時刻 $t$ に関するオイラーステップは経路に依存する。VP 経路上では、ノイズレベルは $\sigma_t = 0.96$（$t = 0.5$）にとどまり、そこから $0.32$（$t = 0.1$）まで低下する（:numref:`sec_diffusion-gaussian-paths`）。つまり、前半の時間ではノイズがほとんど除去されず、その 3 分の 2 が次の 5 分の 2 の時間で除去される。$t$ に関するステップは始点における変化率のまま進むため、このように曲がった経路上では大きなステップをとると大きく外れてしまう（演習問題 :exref:`ex_integration-coordinates`）。そこで本書のライブラリのサンプラでは、あらゆるガウス経路が同じ形に見える座標系でステップを進める。経路を $\alpha_t$ で割ってみよう。

<!--en:d911a327-->
この座標系ではデータのスケールは一定に保たれ、ノイズだけがデータの $\bar\sigma = 0$ から純粋なノイズの $\bar\sigma = \infty$ まで増大する。あらゆる経路が :numref:`sec_diffusion-gaussian-paths` の分散爆発（variance-exploding、VE）経路（アニーリングラダーが下っていった経路）になり、$\bar\sigma$ がそのノイズレベルとなる。:citet:`Karras.Aittala.Aila.ea.2022` はすべての経路をこの方法でサンプリングしている。ここでの上線（バー）は $\alpha_t$ による除算を表しており、DDPM の $\bar\alpha_t = \alpha_t^2$ における二乗の意味ではない（:numref:`tab_diffusion-notation`）。1 つの条件付き経路に沿うと、$\bar{\mathbf{x}}$ は $\bar\sigma$ に対して傾き $\boldsymbol\epsilon$ の直線上を動く。周辺速度場はこの傾きの事後平均、すなわちノイズ予測（noise prediction）である。

<!--en:c97c14fa-->
最初の式の事後平均をとると $\bar{\mathbf{x}}_t = \hat{\mathbf{x}}_0 + \bar\sigma_t \hat{\boldsymbol\epsilon}$ となるため、$\bar\sigma_t$ から $\bar\sigma_s$ へのオイラーステップは $\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\hat{\boldsymbol\epsilon} = \hat{\mathbf{x}}_0 + \bar\sigma_s \hat{\boldsymbol\epsilon}$ となり、これに $\alpha_s$ を掛けると次式を得る。

<!--en:15fd5d90-->
このように、1 回のステップはクリーンな点とノイズを予測し、それらを次のノイズレベルに配置する。この更新式には :numref:`sec_diffusion-samplers` で DDIM という名前のもとで再び出会うことになる。線形経路上ではこれは $t$ に関するオイラーステップそのものであり、それは $\mathbf{x}_t = (1 - t)\hat{\mathbf{x}}_0 + t\hat{\boldsymbol\epsilon}$ を $\mathbf{x}_t - (t - s)(\hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0)$ に代入すれば確かめられる。他の経路上では異なる手法となり、1 点のみからなるデータに対しては任意のステップ幅で厳密になるのに対し、上で見た VP 経路のストライドにおける $t$ のオイラーステップは大きく外れる（演習問題 :exref:`ex_integration-coordinates`）。

<!--en:df5a835b-->
`ode_sample` は、レベル $\bar\sigma$ のリスト（`sigma_bars` が経路の $t$ グリッドから計算する）に沿ってこのステップを進める。純粋なノイズでは $\bar\sigma$ は無限大となり $\bar{\mathbf{x}}$ は定義されないが、ステップに必要なのは $\hat{\mathbf{x}}_0$ と $\hat{\boldsymbol\epsilon}$ だけであり、予測器はこれらをノイズそのものから計算できるため、最初のステップは $\mathbf{x}$ から開始する。

<!--en:e100c2ed-->
サンプラは $\hat{\mathbf{x}}_0$ と $\hat{\boldsymbol\epsilon}$ を任意の $(\alpha, \sigma)$ で必要とし、`make_pred` はどの種類の訓練済みネットワークであってもそのような予測器に変換する。本節では各ネットワークをそれ自身の経路上で動かすため、`make_pred` は予測値をこのペアに変換するだけである。後の節では別の経路のグリッド上でネットワークを動かすが、その場合 `make_pred` はまずネットワーク自身の経路上で同じ対数信号対雑音比（log-SNR）を持つ時刻を（`time_at` 内の二分法によって）見つけ、点をその経路に合わせて再スケーリングする。なお、ノイズモデルには 1 つ死角がある。純粋なノイズ $\alpha = 0$ ではクリーンな点 $\hat{\mathbf{x}}_0 = (\mathbf{x} - \sigma\hat{\boldsymbol\epsilon})/\alpha$ が定義されないため、ノイズモデルは線形経路上でサンプラを開始できない。一方、VP 経路上では $\alpha$ は常に正に保たれ、約 $0.007$（$t = 1$）となるため、ノイズモデルはそちらで用いる。

<!--en:fcd347d3-->
### 学習された速度場

<!--en:7d579c3a-->
訓練した速度ネットワークは厳密な速度場と同じように振る舞い、1 ステップ後にデータ平均に着地し、10 ステップ後に各モードへ収まるだろうか。標準ガウス分布から抽出した $2{,}000$ 個の初期点から、`ode_sample` を 1、10、100 ステップ実行して確かめよう。数値の意味を読み取りやすくするため、さらに 2 行追加する。データそのものから抽出した $2{,}000$ 点（プロットの灰色の点）と、同じ初期点からオラクルで同じ 10 ステップを実行した結果である。最後に、オラクルを用いた `ode_sample` と、`mix.velocity` を用いた単純なオイラー法のループを比較する。線形経路上では両者は同一のステップを踏むため、丸め誤差の範囲で一致するはずである。

<!--en:19ba1928-->
1 ステップでは、すべての初期点がデータ平均 $\approx (-0.5, -0.7)$ へと崩壊する（$t = 1$ における小数第 2 位程度の小さなネットワーク近似誤差を除く）。10 回のオイラーステップを実行すると、3 つのモードが $0.5, 0.3, 0.2$ に近い重みで分離されるものの、$\approx 3\%$ の逸脱点が生じ（真のデータでは $1.1\%$〜$1.5\%$）、モード内の広がりもわずかに狭くなる（$0.45$ 対 $0.48$〜$0.49$）。厳密な `oracle` を用いて 10 回のオイラーステップを実行すると、同じモード重み（差は $0.003$ 以内）と $0.46$ の広がりが得られる。厳密な速度場でもネットワークとほぼ同じだけモードが狭まることから、この縮小はソルバに起因している。一方、逸脱点には 2 つの要因がある。オラクルでも $2\%$〜$3\%$ の逸脱点が残り、データよりは多いがネットワークよりは少ないため、ソルバとネットワークの双方が寄与している。100 回のオイラーステップを実行すれば、学習されたネットワークは真のデータの逸脱点の割合（$\approx 1.3\%$）、広がり、およびモード重みにモンテカルロ誤差の範囲内で一致し、厳密なオラクルに対する `ode_sample` も陽的オイラー法のループと単精度浮動小数点の精度（$10^{-4}$）まで一致する。

<!--en:0dd21ccd-->
## 祖先連鎖と逆 SDE

<!--en:134342a1-->
DDPM のサンプリング方法はこれとは異なる。:numref:`fig_diffusion-forward-reverse` の逆連鎖を、ノイズを伴うステップを 1 回ずつ踏みながら $t = 1$ から $t = 0$ まで実行するのである。そもそもノイズを含むステップの連鎖でなぜうまくサンプリングできるのか、そしてそれはフローとどのような関係にあるのだろうか。

<!--en:1292cc6b-->
### 祖先ステップ

<!--en:6cde4c64-->
小さな順方向ステップの逆過程はほぼガウス分布に従い :cite:`sohl2015deep`、クリーンな点が与えられれば厳密にガウス分布となる。$s < t$ に対し、ベイズの定理により $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ は $\mathbf{x}_s$ に関する 2 つのガウス分布、すなわち順方向ステップ $q(\mathbf{x}_t \mid \mathbf{x}_s)$ とノイズ付加 $q(\mathbf{x}_s \mid \mathbf{x}_0)$ の積に比例するため、$\mathbf{x}_t$ と $\mathbf{x}_0$ の線形結合を平均に持つガウス分布になる。Ho らのステップは、その平均において $\hat{\mathbf{x}}_0$ を $\mathbf{x}_0$ の代わりに用い、分散 $\beta = 1 - \alpha_t^2/\alpha_s^2$（$s$ から $t$ への順方向ステップが加える分散）の新たなノイズを加える。これが*祖先*（ancestral）ステップと呼ばれるのは、グラフィカルモデルの祖先サンプリング（ancestral sampling）と同様に、各 $\mathbf{x}_s$ がその親 $\mathbf{x}_t$ のもとで抽出されるからである。VP 経路上において、$\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon})/\alpha_t$ とスコア $\nabla \log p_t = -\hat{\boldsymbol\epsilon}/\sigma_t$ を平均の式に代入して整理すると、ステップは次のように書ける。

<!--en:24bff9d3-->
これは新しいサンプラなのだろうか、それとも先ほどのサンプラにノイズを加えたものなのだろうか。これを見るには、$1/\sqrt{1 - \beta} \approx 1 + \beta/2$ と展開し、$\beta$ に関する一次の項とノイズ項 $\sqrt\beta\, \mathbf{z}$ を残すとわかりやすい。するとステップは次の 2 つに分解される。

<!--en:5444b49e-->
角括弧の部分は、ランジュバンステップ $\mathbf{x} \leftarrow \mathbf{x} + \frac{h}{2} \nabla \log p + \sqrt{h}\, \boldsymbol\xi$（:numref:`sec_diffusion-denoising`）をステップサイズ $h = \beta$ で密度 $p_t$ に適用したものであり、$p_t$ を定常に保ちつつ逸脱した点をそこへ引き戻す。第 2 項は、時刻 $t$ から $s$ へ戻る確率フロー ODE の 1 オイラーステップである。VP 経路上での周辺速度は $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$ となるが、これは :numref:`sec_diffusion-gaussian-paths` のスコア・速度の恒等式において $\dot\alpha_t/\alpha_t = -\beta(t)/2$ および $\alpha_t^2 + \sigma_t^2 = 1$ としたものにほかならない。ここで $\beta(t) = 0.1 + 19.9\, t$ は VP スケジュールのレートであり、微小ステップでは $\beta \approx \beta(t)\, (t - s)$ が成り立つ。したがって、$t$ から $s$ へ戻るオイラーステップは点を $-(t - s)\, \mathbf{u}_t = \tfrac12 \beta(t)\, (t - s)\, (\mathbf{x}_t + \nabla \log p_t) \approx \tfrac{\beta}{2} (\mathbf{x}_t + \nabla \log p_t)$ だけ動かし、これが第 2 項に一致する。すなわち、祖先ステップとは、フローのステップに現在のノイズレベルにおけるランジュバン補正を加えたものである。この連鎖は ODE と同じ輸送を行いつつ、ランジュバン項の働きによって点を $p_t$ へと引き戻し、それ以前のステップで生じた誤差の一部を打ち消すことができる。実際にその効果が現れるかどうかは :numref:`sec_diffusion-samplers` で測定し、そこでは :numref:`fig_diffusion-two-samplers` において同じ初期点から出発する決定論的軌跡とノイズを含む軌跡を描画する。

<!--en:12dbc865-->
### 逆 SDE

<!--en:46c430fc-->
連続時間では、この時間反転は厳密になる。本節のコードではこの極限を必要としないが、論文ではサンプラがこの記法で書かれるため、記法を 1 つだけ導入しておく価値がある。*確率微分方程式*（stochastic differential equation、SDE）$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\, dt + g(t)\, d\mathbf{W}$ とは、$\Delta t \to 0$ の極限における次のステップの略記である。

<!--en:65baf2b6-->
すなわち、決定論的な*ドリフト* $\mathbf{f}$ に、タイムステップに比例して分散が増大するガウスノイズを加えたものである。DDPM の順方向連鎖は $\Delta t = 1/T$、$\mathbf{f} = -\tfrac12 \beta(t)\, \mathbf{x}$、$g^2 = \beta(t)$ としたこの形のステップになっている（$\sqrt{1 - \beta_k} \approx 1 - \beta_k/2$ かつ $\beta_k \approx \beta(t)\, \Delta t$ であるため）。:citet:`Anderson.1982` は、このような過程を $t = 1$ から $t = 0$ へと逆向きにたどる過程が、ステップ $\mathbf{x} \leftarrow \mathbf{x} - [\mathbf{f} - g^2 \nabla \log p_t(\mathbf{x})]\, \Delta t + g \sqrt{\Delta t}\, \mathbf{z}$ の $\Delta t \to 0$ における極限によって与えられることを示した。この極限において、逆過程は順過程と同じ周辺分布 $p_t$ を持つ。論文では、この反転を時間が逆向きに進む次の SDE として記述する。

<!--en:82da04d9-->
ここで、$d\bar{\mathbf{W}}$ は各逆方向ステップで加わる新たなノイズを表す。なぜこれら 2 つの式は同じ周辺分布を共有するのだろうか。両者の逆方向ステップの差は $\tfrac12 g^2 \nabla \log p_t\, \Delta t + g \sqrt{\Delta t}\, \mathbf{z}$ であり、これはステップサイズ $h = g^2 \Delta t$ による $p_t$ 上のランジュバンステップにほかならない。ランジュバンステップは $p_t$ を不変に保つため、先ほどの祖先ステップの分解が連続時間でそのまま成り立っているのである。VP 経路の場合、右側の ODE は $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$ となり、その経路の確率フロー ODE :eqref:`eq_diffusion-flow-ode` と一致する。一方、逆 SDE のステップは $\mathbf{x} + \tfrac12 \beta(t) \Delta t\, \mathbf{x} + \beta(t) \Delta t\, \nabla \log p_t + \sqrt{\beta(t) \Delta t}\, \mathbf{z}$ となり、$\beta = \beta(t)\, \Delta t$ とおいた祖先ステップの一次展開に一致する。すなわち、祖先連鎖は一次近似において、逆 SDE の最も単純な離散化であるオイラー・丸山法（Euler–Maruyama method、:numref:`sec_mdl-euler-maruyama`）に対応している。ノイズを一切加えない ODE とこの SDE の間には、同じ周辺分布を持つ一連のサンプラ族が存在する。サンプラが注入するノイズが大きいほど、余分な広がりを引き戻すドリフト内のスコア項も強くなる（:numref:`sec_mdl-time-reversal`、:numref:`fig_mdl-dyn-lambda-family`）。

<!--en:7cf3032f-->
`ancestral_sample` は任意の経路と任意の予測器 `pred(x, alpha, sigma) -> (x0, eps)` に対し、一次近似ではなく厳密なステップを実行する。具体的には、$\mathbf{x}_s$ の条件付き平均（$\mathbf{x}_t$ と $\hat{\mathbf{x}}_0$ が与えられたもとでの平均）へと移動し、分散 $\sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$（$s$ から $t$ への順方向ステップの分散であり、VP 経路上では $\beta$ に等しい）のノイズを加える。

<!--en:77c5d3dd-->
フローと同じ $2{,}000$ 個の初期点から、ノイズモデルを用いてこの連鎖を 200 ステップ実行する。Ho らは画像に対して 1,000 ステップを用いたが、2 次元平面上では 200 ステップで十分である。

<!--en:a12f50ed-->
この連鎖は、$2{,}000$ 点の標本誤差の範囲内で 3 つのモードを正しい重みで復元し、逸脱点の割合も 100 ステップのフローと同様に約 1% 以下に収まる。

<!--en:05668a05-->
## 両サンプラからのサンプル

<!--en:33d87545-->
ここまでは各サンプラを $2{,}000$ 点に対して 1 回ずつ実行しただけであり、ネットワークの誤差とサンプリングの偶然によるばらつきを見分けるには点数が少なすぎた。そこで、同一の条件のもとでネットワークとオラクルを比較する。$20{,}000$ 個の初期点を用意し、ノイズモデルによる 200 回の祖先ステップと速度モデルによる 100 回のオイラーステップを実行したうえで、それぞれネットワークを `oracle` に置き換え、同じ初期点（連鎖については注入するノイズも同じ）から同じ計算を繰り返す。これにより、ネットワークの行と対応するオラクルの行との差はすべてネットワークに起因するものとなる。このサンプルサイズでは、モード重みの標準誤差は約 $0.0035$ である。さらに下のセルでは、ネットワークとオラクルが異なるモードへと送った初期点の個数も数える。

<!--en:294cc471-->
どちらのサンプラでも、訓練済みネットワークと厳密なオラクルが異なるモードへ送る初期点は 20,000 個のうちわずか $0.2\%$〜$0.5\%$ にすぎず、ネットワーク実行時の各モード重みも対応するオラクル実行時の値と $0.004$ 以内で一致する（二項分布の標準誤差 $\approx 0.0035$ と同程度である）。また、両サンプラとも真のデータの広がり（誤差 $0.01$ 以内）および逸脱点の割合（$\approx 1.1\%$、すなわち $3s$ を超えるガウス分布の裾）を再現している。

<!--en:04b63e24-->
この混合分布において 100〜200 ステップを実行すれば、決定論的な ODE と確率的な祖先連鎖は実質的に同等の分布の精度を達成する。両者の違いはその構造的性質にある。確率フロー ODE はノイズ $\mathbf{x}_1$ とデータ $\mathbf{x}_0$ の間に決定論的で可逆な全単射を定めるため、潜在空間での補間や厳密な尤度評価が可能になる。一方、祖先連鎖はランジュバンノイズを注入することで、軌跡に沿った離散化誤差やスコア誤差を減衰させることができる（:numref:`sec_diffusion-samplers`）。:numref:`tab_diffusion_models` に、本章で扱う主要な生成モデルが経路・ターゲット・サンプラをどのように組み合わせているかをまとめる。

<!--en:7f7788bc-->
:本章のモデル：ネットワークの訓練に用いるターゲットと経路、およびデータを生成するサンプラ。

<!--en:49e62904-->
| モデル | ターゲットと経路 | サンプラ | 該当箇所 |

<!--en:ba306d9b-->
| ノイズ除去スコアモデル | 単一レベルでのスコア $-\boldsymbol\epsilon/\sigma$ | ランジュバン動力学 | :numref:`sec_diffusion-denoising` |

<!--en:10c9ea6d-->
| ノイズ条件付きスコアネットワーク | VE 経路上のラダーの各レベルにおけるスコア（重み $\tilde\sigma^2$） | アニーリング付きランジュバン動力学 | :numref:`sec_diffusion-gaussian-paths` |

<!--en:722fa829-->
| DDPM | VP 経路上の $\boldsymbol\epsilon$ | 祖先連鎖または確率フロー ODE | :numref:`sec_diffusion-gaussian-paths`、本節 |

<!--en:b4f11a93-->
| フローマッチング、Rectified Flow | 線形経路上の $\boldsymbol\epsilon - \mathbf{x}_0$ | 確率フロー ODE 上のオイラーステップ | :numref:`sec_diffusion-gaussian-paths`、本節 |

<!--en:23f12ad9-->
| $\mathbf{v}$ 予測 | コサイン経路上の $\mathbf{v}$ | いずれも可 | :numref:`sec_diffusion-gaussian-paths`、:numref:`sec_diffusion-design-space` |

<!--en:7a36f1cc-->
| EDM | VE 経路上のクリーンな点（ノイズレベルごとに再スケーリング） | 確率フロー ODE の 2 次ソルバ | :numref:`sec_diffusion-samplers`、:numref:`sec_diffusion-design-space` |

<!--en:c622172e-->
:numref:`sec_diffusion-images` では 2 次元から $28 \times 28$ の画像へと進む。そこでは厳密なオラクルは利用できず、関数評価のたびに畳み込み U-Net の順伝播が必要になる。

<!--en:30ac03ff-->
## まとめ

<!--en:5d8ea303-->
周辺速度 $\mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon \mid \mathbf{x}_t = \mathbf{x}]$ は連続の式 $\partial_t p_t + \nabla \cdot (p_t \mathbf{u}_t) = 0$ を周辺密度 $p_t$ とともに満たすため、確率フロー ODE $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$ を $t = 1$ から $t = 0$ まで積分すれば、モード重みを保ったまま $\mathcal{N}(\mathbf{0}, \mathbf{I})$ が $p_0$ へと輸送される。$t = 1$ から $t = 0$ への 1 回のオイラーステップはすべての初期点をデータ平均 $\mathbb{E}[\mathbf{x}_0]$ へと崩壊させてしまうが、再スケーリングされた座標 $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ と $\bar\sigma = \sigma_t/\alpha_t$ を用いれば、あらゆるガウス経路が共通の常微分方程式 $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$ に帰着し、そのオイラーステップは $\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sigma_s \hat{\boldsymbol\epsilon}$ と書ける。ステップ分散 $\beta$ の一次近似において、DDPM の確率的な祖先ステップはこの確率フローのオイラーステップと $p_t$ 上のランジュバンステップの和へと分解され、これは Anderson の逆時間 SDE に対するオイラー・丸山法による離散化に対応している。

<!--en:2294243a-->
## 演習問題

<!--en:0c014eac-->
1. **2 つの座標系における 1 ステップ。**
    1. 線形経路上において、ステップ $\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sigma_s \hat{\boldsymbol\epsilon}$ がオイラーステップ $\mathbf{x}_t - (t - s)\, \mathbf{u}_t(\mathbf{x}_t)$（$t$ に関するステップ）に等しいことを示せ。
    1. 任意のガウス経路上において、データが 1 点 $\mathbf{x}_0 = \mathbf{0}$ のみであるとする。$\hat{\boldsymbol\epsilon} = \mathbf{x}/\sigma_t$ となること、$\mathbf{x}_t$ を通る厳密な軌跡が $\mathbf{x}_s = (\sigma_s/\sigma_t)\, \mathbf{x}_t$ であること、そして最初の小問のステップがすべての $s < t$ に対してこの厳密な軌跡を正確にたどることを示せ。
    1. VP 経路上では、このデータの周辺速度は $\mathbf{u}_t(\mathbf{x}) = (\dot\sigma_t/\sigma_t)\, \mathbf{x}$ であり、$d\sigma_t^2/dt = \beta(t)\, \alpha_t^2$ かつ $\beta(t) = 0.1 + 19.9\, t$ が成り立つ。$t = 0.5$（$\alpha_t \approx 0.28$、$\sigma_t \approx 0.96$）から $s = 0.1$（$\sigma_s \approx 0.32$）まで 1 ステップ進めよ。$t$ に関するオイラーステップは $\mathbf{x}$ を何倍にするか。また本来は何倍にすべきか。この結果は、少数ステップで動作するサンプラが用いるべき座標系について何を示唆しているか。
   :exlabel:`ex_integration-coordinates`

<!--en:ef6c7b02-->
1. **ガウスデータに対する祖先ステップの一次近似。** ● データを $\mathcal{N}(\mathbf{0}, v_0 \mathbf{I})$ とすると、VP 経路上では $p_t = \mathcal{N}(\mathbf{0}, v_t \mathbf{I})$（ただし $v_t = \alpha_t^2 v_0 + \sigma_t^2$）となる。
    1. :eqref:`eq_diffusion-ancestral-score` の祖先ステップを $t$ から $s$ まで、厳密なスコアと $\beta = 1 - \alpha_t^2/\alpha_s^2$ を用いて 1 回行うと、$p_t$ が $\mathcal{N}\big(\mathbf{0}, (v_s + \delta)\mathbf{I}\big)$（ただし $\delta = \beta^2 (1/v_t - 1)/(1 - \beta)$）に写されることを示せ。
    1. このステップが厳密になるのはどのようなデータのときか。また、混合分布の 1 成分（$v_0 = 0.25$）のように標準ガウス分布より幅の狭いデータに対しては、得られる分散は大きすぎるか、それとも小さすぎるか。
    1. グリッドを細分化してステップ数を 2 倍にし、各ステップの $\beta$ を約半分にしたとき、連鎖全体で蓄積される誤差はどのように変化するか。
   :exlabel:`ex_gaussian-paths-ancestral-first-order`

<!--en:62048611-->
1. **ステップサイズとガウス反転。** ● 祖先連鎖はガウス遷移を用いるが、これはステップが小さい場合に正当化される。本問ではどれほど小さければよいかを調べる。$\sigma_{t|s}^2 = \sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ を、$s$ から $t > s$ への順方向ステップの分散とする。
    1. 厳密な逆遷移が混合分布 $q(\mathbf{x}_s \mid \mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \mid \mathbf{x}_t}\big[q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)\big]$ であり、その共分散が $\tau^2 \mathbf{I} + c^2\, \textrm{Cov}[\mathbf{x}_0 \mid \mathbf{x}_t]$ となることを示せ。ここで、$\tau^2 = \sigma_{t|s}^2 \sigma_s^2/\sigma_t^2$ は $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ の分散であり、$c = \alpha_s \sigma_{t|s}^2/\sigma_t^2$ はその平均における $\mathbf{x}_0$ の係数である。
    1. 1 次元においてデータが等確率で $\pm 1$ をとるものとし、$x_t = 0$ とする。逆遷移が $\mathcal{N}(\pm c, \tau^2)$ の等重み混合分布となり、それが二峰性になるのは $\zeta^2 = c^2/\tau^2$ が 1 を超えるとき、かつそのときに限ること、そして $\zeta^2 = \alpha_s^2 \sigma_{t|s}^2 / (\sigma_t^2 \sigma_s^2)$ となることを示せ。また、$\zeta$ はステップ幅 $t - s$ を細かくするときどのように縮小するか。
    1. VP 経路上において、$\zeta$ を $t = 0.5$ から $s = 0.495$、$0.45$、$0.3$ への各ステップ、および $t = 0.3$ から $s = 0$ への 1 ステップについて計算せよ。二峰性になるのはどの逆遷移か。最後のケースにおいて、最適な平均を持つガウス遷移は $x_t = 0$ で何を生成し、本来は何を生成すべきか。
   :exlabel:`ex_gaussian-paths-gaussian-reversal`

<!--en:b637d11e-->
1. [code] **コミットメント時刻。** ● 経路上のどの時点でサンプルは自身の属するモードを決定するのだろうか。:citet:`Biroli.Bonnaire.deBortoli.ea.2024` はこれをクローン作成によって測定している。すなわち、$p_t$ の 1 点から出発し、独立なノイズを用いて逆連鎖のコピーを 2 本走らせ、両者が同じモードに行き着くかどうかを記録する。
    1. $2{,}000$ 個のクリーンな点を抽出し、VP 経路上で $\mathbf{x}_t$（$t = 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1$）へとノイズを加え、各点から `ancestral_sample(pred_eps, x_t, vp, steps, t_start=t)` のコピーを 2 本ずつ約 $200\, t$ ステップで実行せよ。終点の最近傍平均が一致するペアの割合 $\varphi(t)$ をプロットせよ。$\varphi$ は $t \to 1$ のときどの値に近づくか、またそれはなぜか。
    1. $\Lambda$ をデータの共分散行列の最大固有値とする。Biroli らは、第 1 主成分方向の信号がノイズを上回らなくなる時点 $\alpha_t^2 \Lambda = 1$ によって種分化時刻を定義している。この混合分布に対してその時刻を計算し、得られた曲線およびモード重みを決定づけた :numref:`sec_diffusion-gaussian-paths` のアニーリングラダーの粗いレベルと比較せよ。演習問題 :exref:`ex_mdl-flow-matching-transport-late-decisions`（:numref:`sec_mdl-flow-matching-transport`）および演習問題 :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging`（:numref:`sec_mdl-fokker-planck-probability-flow`）では、1 次元の 2 モードに対して同じ問題を扱っている。
   :exlabel:`ex_gaussian-paths-commitment-time`

<!--en:f35e681c-->
1. **同じ比率、同じフロー。** ● 2 つのガウス経路の係数が $\alpha^B_t = c_t\, \alpha^A_t$ および $\sigma^B_t = c_t\, \sigma^A_t$（$c_t$ は滑らかな正値関数）という関係にあり、すべての時刻で同じ対数信号対雑音比を持つとする。
    1. $p^B_t(\mathbf{y}) = c_t^{-d}\, p^A_t(\mathbf{y}/c_t)$ を示せ。さらに、:numref:`sec_diffusion-gaussian-paths` のスコア・速度の恒等式 $\mathbf{u}_t(\mathbf{x}) = (\dot\alpha_t/\alpha_t)\, \mathbf{x} - \big(\sigma_t \dot\sigma_t - \sigma_t^2 \dot\alpha_t/\alpha_t\big) \nabla \log p_t(\mathbf{x})$ を用いて、$\mathbf{u}^B_t(\mathbf{y}) = (\dot c_t / c_t)\, \mathbf{y} + c_t\, \mathbf{u}^A_t(\mathbf{y}/c_t)$ となることを示せ。ここから、確率フロー ODE の軌跡が $\mathbf{y}_t = c_t\, \mathbf{x}_t$ で関係づけられることを結論づけよ。
    1. $\tau(t)$ を単調増加関数とする。経路 $p_{\tau(t)}$ が速度場 $\dot\tau(t)\, \mathbf{u}_{\tau(t)}$ によって生成され、$\mathbf{u}$ と同じ軌跡を持つことを示せ。
    1. 時間変換 $\tau(t)$ として、コサイン経路 $(\cos\frac{\pi\tau}{2}, \sin\frac{\pi\tau}{2})$ の時刻 $\tau(t)$ における対数信号対雑音比が線形経路の時刻 $t$ における値と一致するものを求め、$c_t = 1/\sqrt{(1 - t)^2 + t^2}$ となることを示せ。経路上のどこで再スケーリングが最大になるか。
    1. コサイン経路の確率フロー ODE は $\mathcal{N}(\mathbf{0}, \mathbf{I})$ に従うデータに対して何をするか、またそれはなぜか。同じデータに対して線形経路の確率フロー ODE は何をするか。
   :exlabel:`ex_gaussian-paths-same-ratio-same-flow`

<!--en:709ed053-->
[ディスカッション](https://d2l.discourse.group/t/36889)

<!--en:b17891ec-->
逆時間積分：ODE と祖先連鎖<br>**連続の式・確率フロー ODE・オイラーステップとデータ平均・経路非依存の座標系・祖先連鎖と逆 SDE**

<!--en:acdd0d69-->
::: {.slide title="速度場に従うとなぜモード重みが復元されるか"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:28a851ed-->
- **固定レベルランジュバン法の限界：**局所スコア $\nabla\log p(\mathbf{x})$ は到達後の $p$ を定常に保つが、モード間を隔てる $10\text{--}12$ nat の谷を越える大域的なフラックスを生まない

<!--en:e9eb191e-->
- **既知の事前分布からの連続輸送：**逆時間積分は既知の事前分布 $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ を $p_t$ に沿って $p_0$ まで連続的に輸送する

<!--en:d51d7799-->
- **確率質量の保存：**軌跡は生成も消滅もしないため、モード $k$ に到達するガウス初期点の割合は $\pi_k$ に一致する

<!--en:98b92a54-->
::: {.slide title="訓練した 2 つの予測器と厳密なオラクル"}

<!--en:5e7332f1-->
[実験設定]{.kicker}

<!--en:29220287-->
離散化誤差とニューラルネットワークの近似誤差を切り分けるため、訓練した 2 つの `TimeMLP(256, 3)` モデルを混合分布の厳密な `oracle` と比較する。

<!--en:78aeab7a-->
- **線形経路の速度ネットワーク（`ema_v`）：**$\boldsymbol\epsilon - \mathbf{x}_0$ をターゲットとして $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$ 上で訓練され、理論的な速度損失の下限（$\approx 4.91$）に到達

<!--en:e7103092-->
- **VP 経路のノイズネットワーク（`ema_eps`）：**ノイズ $\boldsymbol\epsilon$ をターゲットとして分散保持スケジュール上で訓練され、理論的なノイズ損失の下限（$\approx 0.54$）に到達

<!--en:578bde99-->
- **厳密な事後分布オラクル（`oracle`）：**混合分布の閉形式の事後平均 $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ を任意の $(\mathbf{x}, \alpha_t, \sigma_t)$ で計算

<!--en:81bee7cd-->
::: {.slide title="質量保存と連続の式"}

<!--en:5ebecffe-->
[確率輸送]{.kicker}

<!--en:18c20a3a-->
- **質量流束ベクトル $\mathbf{J}_t = p_t(\mathbf{x})\mathbf{u}_t(\mathbf{x})$：**単位時間あたりに単位面積を横切る確率質量を表し、その負の発散が密度の増加率となる

<!--en:8cd80059-->
- **厳密な周辺輸送定理：**$(p_t, \mathbf{u}_t)$ が連続の式を満たすとき、$\mathbf{x}_1 \sim p_1$ から出発して $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$ に従って時間発展する粒子は、厳密な周辺密度 $\mathbf{x}_t \sim p_t$ をすべての $t \in [0, 1]$ で持つ

<!--en:5a7e8b25-->
::: {.slide title="証明：周辺速度は連続の式を満たす"}

<!--en:bd547dcc-->
[条件付き分布の雲から周辺密度へ]{.kicker}

<!--en:5476d0df-->
- **ステップ 1（条件付き保存則）：**固定した各 $\mathbf{x}_0$ に対し、ガウス分布の雲 $p_t(\mathbf{x} \mid \mathbf{x}_0)$ は $\partial_t p_t(\mathbf{x} \mid \mathbf{x}_0) = -\nabla \cdot \big(p_t(\mathbf{x} \mid \mathbf{x}_0)\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_0)\big)$ を満たす

<!--en:12a5ca55-->
- **ステップ 2（ベイズの定理による平均化）：**$p(\mathbf{x}_0)p_t(\mathbf{x} \mid \mathbf{x}_0) = p_t(\mathbf{x})p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})$ を用いて期待値の外へ $p_t(\mathbf{x})$ をくくり出すと、線形性により $\partial_t p_t = -\nabla \cdot (p_t\mathbf{u}_t)$ が得られる

<!--en:c0ff42dd-->
::: {.slide title="ODE の軌跡が条件付き直線と交差しない理由"}

<!--en:239731d1-->
[確率フロー ODE の幾何学]{.kicker}

<!--en:d7717dce-->
- **条件付き直線はいたるところで交差：**異なるモードから伸びる直線の弦 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol\epsilon$ は $\mathbb{R}^d \times (0, 1)$ の全域で交わる

<!--en:9da3e09a-->
- **周辺 ODE の軌跡は決して交わらない：**$\mathbf{u}_t(\mathbf{x})$ は各 $(\mathbf{x}, t)$ にただ 1 つの速度ベクトルを割り当てるため、ピカール・リンデレーフの一意性により軌跡は交差点を避けて曲がり、滑らかな全単射 $\mathcal{N}(\mathbf{0}, \mathbf{I}) \leftrightarrow p_0$ をなす

<!--en:fc540905-->
::: {.slide title="オイラー法と 1 ステップでの平均への崩壊"}

<!--en:f8432f18-->
[確率フロー ODE の離散化]{.kicker}

<!--en:183fe896-->
- **オイラー離散化：**$\mathbf{x}_{t_{k-1}} = \mathbf{x}_{t_k} - \frac{1}{K}\mathbf{u}_{t_k}(\mathbf{x}_{t_k})$ を $K$ ステップにわたって反復

<!--en:c60ce480-->
- **一般のデータで $K = 1$ が崩壊する理由：**$t = 1$ では $\mathbf{x}_1 = \boldsymbol\epsilon$ は $\mathbf{x}_0$ と独立であるため、事後平均は大域的なデータ平均 $\mathbb{E}[\mathbf{x}_0]$ となり、1 回のオイラー更新で**すべての初期点**が $\mathbb{E}[\mathbf{x}_0] = (-0.50, -0.70)$ に集約される

<!--en:ab9a783f-->
::: {.slide title="1・10・100 オイラーステップにおける 5 つの初期点の追跡"}

<!--en:c3906c83-->
[厳密な速度場の軌跡]{.kicker}

<!--en:fc7bd0f5-->
混合分布の厳密な速度場 `mix.velocity` を、原点 $(0, 0)$ と $(\pm 1.5, 0), (0, \pm 1.5)$ の 5 つの固定初期点から積分すると、複数ステップの曲率が事前分布をどのように分割するかがわかる。

<!--en:5ffa2c6d-->
- **1 ステップ：**5 つの初期点すべてがモード間の空っぽの谷にある $\mathbb{E}[\mathbf{x}_0] = (-0.50, -0.70)$ へと一様に崩壊

<!--en:cc9c734c-->
- **10 対 100 ステップ：**10 ステップまでにすべての軌跡がいずれかのモードに定まる（100 ステップの終点から $\approx 0.2$ 以内で、大域平均側へわずかに偏る）

<!--en:8f664d33-->
- **質量を保存する吸引領域：**原点を含む 5 つの初期点のうち 3 つが最も重いモード（$\pi_1 = 0.5$、位置 $(-2.5, -1.5)$）に流入し、$(1.5, 0)$ はモード 2、$(0, 1.5)$ はモード 3 へと向かう

<!--en:d7075b8c-->
::: {.slide title="経路非依存の座標系：α_t による再スケーリング"}

<!--en:8278fb79-->
[スケジュール間での ODE 積分の統一]{.kicker}

<!--en:ba7a3e65-->
- **汎用 VE 座標系（Karras et al., 2022）：**$\mathbf{x}_t$ を $\alpha_t$ で割りノイズ比 $\bar\sigma_t = \sigma_t/\alpha_t = e^{-\lambda_t/2}$ を導入すると線形ドリフトが吸収され、すべての経路が $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$ に従う

<!--en:23682afb-->
- **経路非依存のオイラー更新（$\eta = 0$ の DDIM と一致）：**$\bar\sigma_t$ から $\bar\sigma_s$ へステップを進め $\alpha_s$ を掛けると $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$ が得られる

<!--en:beb70719-->
::: {.slide title="経路非依存の ODE サンプリングの実装"}

<!--en:9e22d402-->
[汎用予測器とオイラーループ]{.kicker}

<!--en:f9e034f5-->
同じ対数信号対雑音比で任意のネットワーク出力を $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ に変換する `make_pred` と `ode_sample` を組み合わせることで、任意のソルバが任意の訓練済みモデルを任意のスケジュール上で積分できる。

<!--en:e33a1273-->
- **純粋なノイズ（$\alpha_1 = 0, \bar\sigma_1 = \infty$）の扱い：**線形経路の $t = 1$ では $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ が無限大になるが、更新式 $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$ に必要なのは $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ の $(\mathbf{x}_1, \alpha_1 = 0, \sigma_1 = 1)$ における評価値のみ

<!--en:9ba36eb3-->
- **ノイズモデルに $\alpha_1 > 0$ が必要な理由：**`'eps'` は $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t\hat{\boldsymbol\epsilon})/\alpha_t$ によりクリーンな点を復元するため $\alpha_1 = 0$ では開始できないが、VP 経路上では $\alpha_1 \approx 0.007 > 0$ となる

<!--en:0f6ba40c-->
- **線形経路での同値性：**$\alpha_s = 1 - s$ と $\sigma_s = s$ を代入すると $\mathbf{x}_s = (1 - s)\hat{\mathbf{x}}_0 + s\hat{\boldsymbol\epsilon} = \mathbf{x}_t - (t - s)(\hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0)$ となり、$t$ に関する陽的オイラー法と一致する

<!--en:434a9f80-->
- **1 点データでの厳密性：**VP 経路における $t$ のオイラー法とは異なり、$\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$ は任意のステップサイズで離散化誤差なしに単一点データを積分する

<!--en:0cf4e516-->
::: {.slide title="1・10・100 ステップでの学習済み速度場からのサンプリング"}

<!--en:4c78025d-->
[ODE の実験的収束]{.kicker}

<!--en:2ca2669f-->
`ode_sample` を訓練済みの線形速度ネットワーク `ema_v` 上で $2{,}000$ 個のガウス初期点から実行すると、$K$ の増加に伴って離散化誤差が消失する様子が確認できる。

<!--en:31673eb0-->
::: {.slide title="離散化誤差とネットワーク誤差の切り分け"}

<!--en:a90dec1f-->
[10 ステップ実行と 100 ステップ実行の比較分析]{.kicker}

<!--en:07c50e5b-->
- **10 オイラーステップ（訓練済み MLP 対厳密なオラクル）：**どちらもモード内の広がりが狭まるため（$0.45$ および $0.46$ 対データの $0.48\text{--}0.49$）、この縮小はネットワーク誤差ではなくソルバの離散化によるもの。逸脱点はネットワークで $\approx 3\%$、オラクルで $2\text{--}3\%$ と、データの $1.1\text{--}1.5\%$ から増加

<!--en:47e7cf2d-->
- **100 オイラーステップ：**データのモード重み（標本誤差の範囲内）、広がり（$0.48$）、逸脱点の割合（$\approx 1.3\%$）と一致

<!--en:9d392751-->
::: {.slide title="DDPM 祖先サンプリング：一次分解"}

<!--en:29f6544b-->
[確率的逆マルコフ連鎖]{.kicker}

<!--en:bd265ff0-->
- **一次展開（$1/\sqrt{1-\beta} \approx 1 + \beta/2$）：**DDPM の逆ステップのドリフトを分解すると、同時に働く 2 つの更新が現れる

<!--en:32226540-->
- **構造的解釈：**すべての祖先ステップは**確率フロー ODE の 1 オイラーステップ**（$p_t \to p_s$ を輸送）と **1 ランジュバン MCMC ステップ**（逸脱を生むドリフト誤差を減衰）の和

<!--en:72ff9031-->
::: {.slide title="連続時間：逆 SDE 対確率フロー ODE"}

<!--en:858e602e-->
[Anderson の時間反転定理（1982）]{.kicker}

<!--en:a36fd7ae-->
- **逆時間 SDE（確率的）：**DDPM の祖先連鎖により離散化され、新たなブラウンノイズ $g(t)\,d\bar{\mathbf{W}}$ が軌跡を絶えず混合してスコア誤差を減衰させる

<!--en:70b7777f-->
- **確率フロー ODE（決定論的）：**$g\,d\bar{\mathbf{W}}$ を除いてスコア項を半分（$g^2 \to \frac{1}{2}g^2$）にすることで、可逆な全単射 $\mathbf{x}_1 \leftrightarrow \mathbf{x}_0$ を定めつつ厳密に同じ周辺分布 $p_t(\mathbf{x})$ を共有する

<!--en:98e69a0e-->
::: {.slide title="対応のあるベンチマーク：20,000 初期点対厳密なオラクル"}

<!--en:f8a88645-->
[祖先連鎖対確率フロー ODE]{.kicker}

<!--en:d2bf15aa-->
$200$ 回の祖先ステップ（VP 経路上の `ema_eps`）と $100$ 回のオイラー ODE ステップ（線形経路上の `ema_v`）を共通の $20{,}000$ 個のガウス初期点 $\mathbf{x}_1$ から実行し、同一の初期点と注入ノイズを用いた厳密な `oracle` の実行と対にして比較する。

<!--en:bf6d6c11-->
::: {.slide title="混合分布における ODE サンプラと祖先サンプラの比較"}

<!--en:5c0bad45-->
[20,000 サンプル実験の解釈]{.kicker}

<!--en:fbc777c4-->
サンプル数 $N = 20{,}000$ ではモード重みの二項標準誤差が $\sqrt{w(1-w)/N} \approx 0.0035$ となり、分布の精度を鋭敏に検証できる。

<!--en:61dff380-->
- **厳密なオラクルとのモード不一致は 1% 未満：**$99.5\%$〜$99.8\%$ の初期点（全 $20{,}000$ 点中）において、訓練済みニューラルネットワークと厳密な解析的事後平均は軌跡を**まったく同じモード**へと送る

<!--en:bfbca854-->
- **サンプリングノイズ範囲内のモード重み：**200 ステップの祖先連鎖と 100 ステップのオイラー ODE はいずれも、オラクルのモード重みに $\le 0.004$（$\approx 1\,\mathrm{SE}$）以内、モード内の広がりに $0.01$ 以内、逸脱率（$\approx 1.1\%$）でも一致

<!--en:37500ae2-->
- **決定論的写像対確率的補正：**ODE は各初期点 $\mathbf{x}_1$ を一意な $\mathbf{x}_0$ へと決定論的に写像し（§{sec_diffusion-samplers} の反転や潜在補間を可能にする）、祖先連鎖は毎ステップでランジュバンノイズを注入する

<!--en:aef88098-->
::: {.slide title="総括：第 §{num:chap_diffusion} 章の生成モデル"}

<!--en:52fa418b-->
[経路・ターゲット・サンプラの統一]{.kicker}

<!--en:0d4afdef-->
文献に登場するあらゆる拡散モデルとフローマッチングモデルは、順方向経路・回帰目標・逆時間積分器という 3 つのモジュール選択によって規定される。

<!--en:ffcd8459-->
| モデル族 | 順方向経路 | 訓練ターゲット | 逆時間サンプラ | 該当節 |

<!--en:aad6e6bc-->
| **単一レベル DSM** | 固定ぼかし $\mathbf{x}_0 + \sigma\boldsymbol\epsilon$ | スコア $-\boldsymbol\epsilon/\sigma$ | 固定 $\sigma$ ランジュバン MCMC | §{sec_diffusion-denoising} |

<!--en:93922094-->
| **NCSN**（Song & Ermon, 2019） | 幾何 VE ラダー $\tilde\sigma_i$ | スコア $-\boldsymbol\epsilon/\tilde\sigma_i$（重み $\tilde\sigma_i^2$） | アニーリング付きランジュバンラダー | §{sec_diffusion-gaussian-paths} |

<!--en:b77bb6a7-->
| **DDPM**（Ho et al., 2020） | 線形 $\beta$ VP（$\alpha_t^2 + \sigma_t^2 = 1$） | ノイズ $\boldsymbol\epsilon$ | 祖先連鎖（逆 SDE） | §{sec_diffusion-gaussian-paths}〜§{num:sec_diffusion-integration} |

<!--en:8c841b56-->
| **フローマッチング / Rectified Flow** | 線形 $(1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$ | 速度 $\boldsymbol\epsilon - \mathbf{x}_0$ | フロー ODE 上のオイラー / Heun 法 | §{sec_diffusion-gaussian-paths}〜§{num:sec_diffusion-samplers} |

<!--en:e0f5b193-->
| **$\mathbf{v}$ 予測**（Salimans & Ho） | コサイン VP（$\cos\frac{\pi t}{2}, \sin\frac{\pi t}{2}$） | 角速度 $\mathbf{v} = \alpha_t\boldsymbol\epsilon - \sigma_t\mathbf{x}_0$ | ODE または祖先サンプラ | §{sec_diffusion-gaussian-paths}、§{sec_diffusion-design-space} |

<!--en:9174e2c5-->
| **EDM**（Karras et al., 2022） | VE 座標 $\bar{\mathbf{x}} = \mathbf{x}_0 + \bar\sigma\boldsymbol\epsilon$ | 前処理付きクリーンな点 $\hat{\mathbf{x}}_0$ | $\bar\sigma^{1/\rho}$ グリッド上の 2 次 Heun 法 | §{sec_diffusion-samplers}〜§{num:sec_diffusion-design-space} |

<!--en:c2d81b72-->
::: {.slide title="概念チェック：輸送・オイラーステップ・SDE の分解"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:99f18c36-->
逆時間積分とサンプラの仕組みに関する理解を確認しよう。

<!--en:763156f3-->
1. **純粋なノイズからの 1 オイラーステップ：**平均 $(-2.5, -1.5)$、$(2.5, -1.5)$、$(0, 2.5)$、重み $(0.5, 0.3, 0.2)$ を持つ 3 モード混合分布において、1 回のオイラーステップ $\mathbf{x}_0 = \mathbf{x}_1 - \mathbf{u}_1(\mathbf{x}_1)$ は任意の初期点 $\mathbf{x}_1$ をどこへ送るか。

<!--en:277552ed-->
2. **純粋なノイズ付近での誤差増幅：**線形経路上では $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - t\hat{\boldsymbol\epsilon})/(1 - t)$ である。ノイズ予測器 $\hat{\boldsymbol\epsilon}$ が誤差 $\boldsymbol\delta$ を $t = 0.95$ で持つとき、速度 $\mathbf{u}_t = \hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0$ に生じる誤差はどれほどの大きさになるか。

<!--en:e37075dc-->
3. **逆 SDE と確率フロー ODE が周辺分布を共有する理由：**逆 SDE と確率フロー ODE の逆方向ステップはどの項だけ異なり、なぜその項は $p_t$ を変化させないのか。

<!--en:557b98b6-->
**解答：**

<!--en:a0401291-->
1. $\mathbf{u}_1(\mathbf{x}_1) = \mathbf{x}_1 - \mathbb{E}[\mathbf{x}_0]$ より、すべての初期点 $\mathbf{x}_1$ は大域的なデータ平均 $\mathbb{E}[\mathbf{x}_0] = 0.5(-2.5, -1.5) + 0.3(2.5, -1.5) + 0.2(0, 2.5) = \mathbf{(-0.50, -0.70)}$ へと崩壊する。

<!--en:1cd57152-->
2. $\mathbf{u}_t = (\hat{\boldsymbol\epsilon} - \mathbf{x}_t)/(1 - t)$ より、誤差 $\boldsymbol\delta$（$\hat{\boldsymbol\epsilon}$ の誤差）は $1/(1 - t) = 1/0.05 = \mathbf{20\times}$（$\mathbf{20\,\boldsymbol\delta}$）に拡大され、$t = 1$（$\alpha_1 = 0$）で発散する。

<!--en:e3350239-->
3. 両者の逆方向ステップの差は $\frac{1}{2}g(t)^2\nabla \log p_t(\mathbf{x})\,\Delta t + g(t)\sqrt{\Delta t}\,\mathbf{z}$ であり、これはステップサイズ $h = g(t)^2\Delta t$、定常分布 $p_t$ を持つ無限小の**ランジュバンステップ**である。

<!--en:4ef7ab1b-->
::: {.slide title="まとめ：逆時間積分の仕組み"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:90706fd0-->
| 概念 / サンプラ | 数式 | 構造的保証 | 実際の挙動 |

<!--en:1768ccc8-->
| **連続の式** | $\partial_t p_t = -\nabla \cdot (p_t \mathbf{u}_t)$ | 全確率質量を保存 | $\mathbf{u}_t = \mathbb{E}[\dot{\mathbf{x}}_t \mid \mathbf{x}_t]$ による正確なモード重みの復元を保証 |

<!--en:b5b44c7a-->
| **確率フロー ODE** | $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$（$t: 1 \to 0$） | 滑らかな決定論的全単射 $\mathbf{x}_1 \leftrightarrow \mathbf{x}_0$ | 再現可能なサンプル、潜在コード、厳密な尤度 |

<!--en:c9604a8b-->
| **1 ステップオイラー崩壊** | $\mathbf{x}_1 - \mathbf{u}_1(\mathbf{x}_1) = \mathbb{E}[\mathbf{x}_0]$ | $t = 1$ での接線が事前平均を向く | 全初期点を $\mathbb{E}[\mathbf{x}_0]$ へ崩壊させ、曲がった軌跡には $\ge 10$ ステップ必要 |

<!--en:2733e331-->
| **再スケーリング VE 座標** | $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t, \;\bar\sigma = \sigma_t/\alpha_t$ | 任意の経路で $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}, \bar\sigma)$ | オイラーステップ $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$（$\eta = 0$ の DDIM） |

<!--en:e223fbf7-->
| **DDPM 祖先ステップ** | $\frac{\mathbf{x}_t + \beta\nabla \log p_t}{\sqrt{1-\beta}} + \sqrt{\beta}\mathbf{z}$ | $= \text{Flow ODE Euler step} + \text{Langevin step}$ | Anderson の逆 SDE を離散化し、ドリフト誤差を減衰 |
