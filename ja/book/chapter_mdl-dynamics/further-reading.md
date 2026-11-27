<!--en:7902ddfd-->
# 関連文献

<!--en:e3a1d72f-->
これまでの 7 つの節では、1 つの点の軌跡から出発してノイズをデータへと変換するフローに至るまでをたどってきたが、その途上では数多くの問いに立ち止まることなく通り過ぎてきた。物理系におけるノイズを記述するのはどの確率積分（stochastic integral）か。モデル自体が SDE であるとき、随伴系はどうなるのか。以下の各項目では、そうした問いを 1 つずつ取り上げ、その答えがどこにあるかを示す。

<!--en:25736441-->
拡散モデル（diffusion model）の実践は :numref:`chap_diffusion` とその関連文献（:numref:`sec_diffusion-further-reading`）の守備範囲であるため、そこで扱われる手法については参照先を示すにとどめる。残る話題は数学である。まずは次の 3 つの項目から読むことを勧める。拡散サンプラが少数のステップで済む理由の 1 つである指数積分器（exponential integrator）、拡散する密度が何を下っていくのかを述べるワッサースタイン勾配フロー（Wasserstein gradient flow）、そして最適なペアリングの確率版にあたるシュレーディンガーブリッジ（Schrödinger bridge）である。残りの項目は、読者によって抱くかもしれない疑問に答えるものである。:numref:`sec_mdl-flow-matching-transport` で読み終えたとしても、本書の残りの部分で用いる内容を何ひとつ見落とすことはない。

<!--en:86e5b87a-->
## ソルバ

<!--en:96f8e5b4-->
### 指数積分器

<!--en:85c914c1-->
学習されたスコアを用いた VP 過程の確率フロー ODE（:numref:`sec_mdl-score-based-generative-modeling`）である $\dot{\mathbf x}=-\tfrac12\beta\,\mathbf x-\tfrac12\beta\,\mathbf s_{\boldsymbol\theta}(\mathbf x,t)$ は、閉じた形で既知の線形項と学習された部分の和からなる半線形（semilinear）方程式である。ルンゲ・クッタ法のステップは両方を離散化するため、線形項も誤差に寄与するうえ、学習された部分がどれほど滑らかであっても :numref:`sec_mdl-stiffness-implicit` の安定性限界によってステップ幅が制約される。線形項は厳密に解ける以上、これは無駄である。指数積分器は、学習された部分を駆動線形方程式（:numref:`subsec_mdl-ode-driven-linear`）への入力として扱う。すなわち、定数変化法（variation of constants）の公式によって線形項による状態の時間発展を厳密に運び、学習された部分の積分だけを近似するのである。

<!--en:a0ad20ba-->
ステップの間、学習された部分を一定に保てば、ネットワークの出力を入力とする同項のゼロ次ホールドになる。その出力が変化しなければこのステップは厳密になるが、実際には変化するため 1 次精度となる。実は本章でも、この名前こそ出さなかったものの同じステップを一度用いている。$\sqrt{\bar\alpha_t}$ で割る操作（:numref:`sec_mdl-ddim-guidance`）によって線形項が取り除かれ、DDIM は 1 ストライドにわたってノイズ予測を一定に保つ。つまり DDIM は 1 次の指数積分器である。

<!--en:6ddb12f6-->
この系統のサンプラは、どの出力（スコア、ノイズ予測、またはクリーンな点の予測）を一定に保つか、そしてどの時計を用いるかという点で異なる。より高次の手法では、定数の代わりに複数の予測値を通る多項式を用いる。その理論は :citet:`Hochbruck.Ostermann.2010` が概観している。これに基づく拡散サンプラには DEIS :cite:`Zhang.Chen.2023` や DPM-Solver :cite:`Lu.Zhou.Bao.ea.2022` がある。後者は演習問題 :exref:`ex_samplers-dpm-solver`（:numref:`sec_diffusion-samplers`）の題材であり、その後継手法は :numref:`sec_diffusion-further-reading` に挙げられている。SDE に対しても同じ考え方から、演習問題 :exref:`ex_mdl-sdes-exact-vp-discretization`（:numref:`sec_mdl-sdes`）の指数オイラー・丸山法のステップが得られる。

<!--en:c8d19ae2-->
### 幾何学的数値積分とリープフロッグ法

<!--en:a097bea8-->
摩擦のない振動子に対して前進オイラー法はステップごとにエネルギーを増加させ、後退オイラー法はエネルギーを減少させるため、長時間実行すると前者は発散し、後者は静止してしまう。より高次の手法を用いても、これを先延ばしにできるにすぎない。*幾何学的数値積分*（geometric integration）が手法に求めるのは別の性質、すなわち厳密なフローを精度よく追跡するかどうかとは無関係に、厳密なフローが持つ構造を保存することである。シンプレクティックオイラー法やリープフロッグ法のような*シンプレクティック*手法は、厳密なフローと同様に、位置と運動量の平面における面積を保存する（座標が増えれば面積は体積になる）。

<!--en:3c680ab6-->
さらに後退誤差解析（backward error analysis）によれば、このような手法は $1/h$ に関して指数関数的に小さい誤差を除いて、わずかに異なるエネルギーに対する厳密なフローを追跡する。したがって真のエネルギーにおける誤差は、ホライズンとともに増大するのではなく、非常に長い時間にわたって有界にとどまる。演習問題 :exref:`ex_mdl-odes-solvers-symplectic-euler`（:numref:`sec_mdl-euler-runge-kutta`）では、振動子に対するシンプレクティックオイラー法についてこの異なるエネルギーを求める。その等位集合は、真のエネルギーが描く円に近い楕円になる。勾配降下法にもエネルギーを損失に置き換えた同じ解析が成り立ち、損失に加わる項は :citet:`Barrett.Dherin.2021` の暗黙的勾配正則化項（implicit gradient regularizer）である（同節の演習問題 :exref:`ex_mdl-odes-solvers-implicit-gradient-regularization`）。この理論は :citet:`Hairer.Lubich.Wanner.2006` が展開している。

<!--en:88e5e63d-->
リープフロッグ法は、ハミルトニアンモンテカルロ法 :cite:`Duane.Kennedy.Pendleton.ea.1987,Neal.2011` の内部で用いられる数値積分器でもある。このサンプラは、目標分布の負の対数密度をポテンシャルエネルギーとして持つ摩擦のない粒子をシミュレーションすることで遠くへの移動を提案し、メトロポリス則（:numref:`sec_mdl-bayes-mcmc`）によって採択または棄却する。リープフロッグ法のエネルギー誤差が有界にとどまるため、遠くへの移動であっても採択され得る。さらに、リープフロッグ法のステップは可逆で体積を保存するため、この判定則は 2 つのエネルギーを比較するだけでよく、ヤコビ行列を必要としない。

<!--en:77bb31eb-->
## 連続時間モデル

<!--en:1902d89b-->
### 行列式を安価に計算できるフロー

<!--en:0bb35516-->
離散的な正規化フロー（normalizing flow）は、ヤコビ行列の行列式を安価に計算できるよう各層に制約を課す。:numref:`sec_mdl-continuous-normalizing-flows` では通常の制約として三角ヤコビ行列を挙げたが、層がどのようにしてそれを得るかは述べなかった。カップリング層 :cite:`dinh2014nice,dinh2017density` は、座標の半分をそのまま残し、もう半分を前半の座標の関数によってシフトおよびスケーリングする。そのヤコビ行列は三角行列となり、逆変換は減算と除算だけで計算できる。1 つの層では状態の半分しか変化しないため、多数の層を積み重ね、層ごとに役割を入れ替える。

<!--en:dbec08fc-->
可逆残差ネットワーク（invertible residual network）:cite:`Behrmann.Grathwohl.Chen.ea.2019` は別の対象に制約を課す。残差ブロックはオイラー法の 1 ステップであり（:numref:`sec_mdl-neural-odes`）、分岐のリプシッツ定数を 1 未満に抑えれば、リプシッツ連続なベクトル場のフローと同様に（:numref:`subsec_mdl-ode-flow-map`）、そのステップは可逆になる。ヤコビ行列は一般の形状を保ち、その対数行列式は演習問題 :exref:`ex_mdl-odes-solvers-residual-logdet`（:numref:`sec_mdl-neural-odes-cnf`）のべき級数によって推定される。:citet:`Papamakarios.Nalisnick.Rezende.ea.2021` はこれらのモデル族と、それぞれが何を犠牲にしているかを概観している。

<!--en:7b17430b-->
### 可逆ソルバと正則化フロー

<!--en:c4a80ee0-->
:numref:`sec_mdl-adjoint-method` の随伴法（adjoint method）は、逆方向の計算中に順方向の状態を再構成することでメモリを節約するが、縮小するフローではその再構成に伴う丸め誤差が増幅される。再構成には縮小がなくても生じる第 2 の誤差がある。ソルバの逆方向ステップは順方向ステップの逆写像に近いものの、厳密な逆写像ではない。そのため逆向きにたどって得られる状態は順方向に送り出した状態と一致せず、得られる勾配も実際に実行した計算の勾配ではなくなる。可逆ソルバ（reversible solver）は各ステップを代数的に元へ戻せるように構成されており、この第 2 の誤差を取り除く。ただし丸め誤差は依然として増幅されるため、硬い（stiff）ベクトル場に対してはチェックポイント保存（checkpointing）の方が引き続き安全な選択肢である。両者の比較は :citet:`Kidger.2022`（第 5 章）にある。

<!--en:6153f544-->
尤度によって訓練されたフローには、数値積分が安価になるべき理由がない。尤度は各点が最終的にどこへ到達するかだけで決まり、たどった経路には依存しないからである。そこで :citet:`Finlay.Jacobsen.Nurbekyan.ea.2020` は、訓練中に運動エネルギー $\int\mathbb{E}\|\mathbf f_{\boldsymbol\theta}\|^2\,dt$ にペナルティを課す。固定された 2 端点間では等速の直線経路が最小の運動エネルギーを持つため、このペナルティによって経路は直線に近づき、ソルバは直線経路を少数のステップで走破できる。フローマッチングは、直線的な回帰目標とノイズ・データのペアリング（:numref:`sec_mdl-ot-connection`）を通じて同じ目標を追求する。

<!--en:b9a0c8c8-->
### 潜在微分方程式と制御微分方程式

<!--en:2a366507-->
本章のモデルは初期状態を最終状態へと写像する。しかし患者の診療記録のような時系列データは不規則な時刻に到着するため、そのモデルには各観測が得られた時点でそれを順次取り込むことが求められる。潜在 ODE（latent ODE）:cite:`Rubanova.Chen.Duvenaud.2019` はその半分を実現する。すなわち、観測列を 1 つの初期状態へエンコードし、任意の時刻集合における軌跡をデコードするため、後から到着した観測が進路を変えることはできない。

<!--en:91cbfb6c-->
ニューラル制御微分方程式（neural controlled differential equation）:cite:`Kidger.Morrill.Foster.ea.2020` では、すべての観測が状態に作用できる。この手法は補間されたデータ経路 $\mathbf X_t$ そのものによって状態を駆動し、$d\mathbf z=\mathbf f_{\boldsymbol\theta}(\mathbf z)\,d\mathbf X_t$（ただし $\mathbf f_{\boldsymbol\theta}$ は行列値関数）と表される。この方程式はブラウン経路をデータに置き換えた SDE の形をしているが、補間された経路は滑らかであるため、積分は通常の積分になる。そのオイラーステップ $\mathbf z_{n+1}=\mathbf z_n+\mathbf f_{\boldsymbol\theta}(\mathbf z_n)(\mathbf X_{n+1}-\mathbf X_n)$ は、データの増分を読み取るリカレントネットワークにほかならない。

<!--en:5a42a7b9-->
### 相互作用する粒子としての Transformer

<!--en:80d6cc3a-->
残差ネットワークはオイラーステップによって 1 つの状態を動かす（:numref:`sec_mdl-neural-odes`）。一方、Transformer は系列内のすべてのトークンを同時に動かし、アテンションによって各トークンの速度が他のトークンの位置に依存するようになる。つまりトークンは相互作用する粒子である。:citet:`Geshkovski.Letrouit.Polyanskiy.ea.2023` は、層正規化によってすべてのトークンが単位球面上に保たれる理想化されたモデルでこの系を解析し、深さが増すにつれてトークンがクラスタへと集まることを示している。本章で扱った系との唯一の違いは、ベクトル場が自ら動かす密度に依存するため、連続の式（continuity equation）が密度に関してもはや線形ではなくなる点である。

<!--en:1842496b-->
## ノイズ

<!--en:30de2a42-->
### ワッサースタイン勾配フロー

<!--en:e36d2003-->
勾配フロー上の点は損失を下っていく（:numref:`sec_mdl-gd-as-solver`）。拡散過程の密度もまた、フォッカー・プランク方程式に従って決定論的に時間発展する。では、密度も何かを下っているのだろうか。ドリフトが勾配 $\mathbf f=-\nabla V$ であり、かつ $g$ が定数であるときには、確かに下っている。このとき密度は自由エネルギー（free energy）$\mathcal F(p)=\int Vp\,d\mathbf x+\tfrac{g^2}2\int p\log p\,d\mathbf x$ を下る。第 1 項は平均ポテンシャルエネルギーであり、すべての質量が $V$ の底に集まるとき最小になる。第 2 項は負のエントロピーであり、質量が広がっているとき最小になる。ドリフトは第 1 項に、ノイズは第 2 項に働き、両者は $e^{-2V/g^2}$ に比例する定常密度において釣り合う。

<!--en:ae635dea-->
最急降下方向は距離の測り方に依存し、ここで 2 つの密度間の距離として用いなければならないのは、一方を他方へ輸送するコストを測る 2 次ワッサースタイン距離である。距離がどこに現れるかは :numref:`sec_mdl-gd-as-solver` の後退オイラーステップを見ればわかる。凸な損失に対して、このステップは $\mathcal L(\mathbf x)+\tfrac1{2\eta}\|\mathbf x-\mathbf x_k\|^2$ を最小化する。JKO スキーム :cite:`Jordan.Kinderlehrer.Otto.1998` は、このステップの点を密度に、二乗距離を輸送コストに置き換えた $p_{k+1}=\arg\min_p\,\mathcal F(p)+\tfrac1{2\eta}W_2^2(p,p_k)$ である。$\eta\to0$ のとき、そのステップ列はフォッカー・プランク方程式の解へと収束する。

<!--en:b007d2c8-->
とりわけ記憶にとどめておくべき例がランジュバン動力学である。そのドリフトは目標分布のスコアの半分であり、ノイズは $g=1$ を持つ（:numref:`sec_mdl-langevin-sampling`）。目標分布を $p_\infty$ と書き、粒子群の密度を表す $p$ と区別しよう。すると $V=-\tfrac12\log p_\infty$ となり、自由エネルギーは $\tfrac12D_{\textrm{KL}}(p\,\|\,p_\infty)$ となる。すなわちランジュバン粒子の群れは、輸送によって距離を測ったもとで、目標分布に対する KL ダイバージェンス上の勾配降下法を実行しているのである。連続の式からベナムー・ブレニエ :cite:`Benamou.Brenier.2000` に至るまで、本章で輸送が繰り返し姿を現す理由はここにある。

<!--en:00509409-->
### 物理ノイズとウォン・ザカイの定理

<!--en:ce3236a7-->
:numref:`sec_mdl-ito-integral` では左端点を用いて伊藤積分を定義し、ストラトノビッチ積分を定義する中点則では通常の連鎖律が保たれることを述べた。では、物理系を記述するのはこの 2 つのうちどちらか。物理系におけるノイズはブラウン経路ではない。短い記憶と滑らかな経路を持っており、滑らかな経路に対しては通常の積分ただ 1 つしか存在しない。:citet:`Wong.Zakai.1965` は、そのような滑らかな経路がブラウン経路に近づくとき、通常の微分方程式の解がストラトノビッチ解へと収束することを証明した。1 次元において、$dX=f\,dt+g\,dW$ をストラトノビッチの意味で解釈した方程式は、ドリフトを $f+\tfrac12g\,\partial_xg$ とした伊藤方程式に等しい。

<!--en:c76997b9-->
拡散モデルの加法的ノイズ（additive noise）に対しては、この 2 つの解釈は一致する。乗法的ノイズ（multiplicative noise）に対しては両者は異なり、どちらを選ぶべきかをこの定理が教えてくれる。滑らかな物理ノイズを理想化した方程式はストラトノビッチとして解釈する。一方、離散時間で構築され、係数が自らの掛ける増分を先回りして見ることのできないモデルは伊藤として解釈する。確率的勾配降下法のステップはこの後者の種類に属する。

<!--en:5ec6a72e-->
### SDE としての確率的勾配降下法

<!--en:980ad8bf-->
勾配降下法は勾配フローに対する前進オイラー法である（:numref:`sec_mdl-gd-as-solver`）。では、確率的勾配降下法は何の離散化なのだろうか。ミニバッチ勾配は全勾配にノイズを加えたものであり、その共分散 $\boldsymbol\Sigma$ は $1/B$ に比例してバッチサイズ $B$ とともに縮小する。:citet:`Li.Tai.E.2017` は SDE として $d\boldsymbol\theta=-\nabla\mathcal L\,dt+\sqrt{\eta}\,\boldsymbol\Sigma^{1/2}d\mathbf W$ を提案している。長さ $\eta$ のオイラー・丸山法のステップをとってみよう。ノイズ係数にさらに $\sqrt\eta$ が掛かり、結果として学習率 $\eta$ の確率的勾配降下法の 1 ステップが得られる。

<!--en:8af7bb48-->
したがって学習率は、ステップ幅として、そしてノイズの内部に、計 2 回現れる。$\boldsymbol\Sigma$ がいたるところ単位行列の同じ定数倍 $c\,\mathbf I$ であるとき、先ほどの自由エネルギーが $V=\mathcal L$ および $g^2=\eta c$ としてそのまま当てはまり、損失に対してエントロピーの重みを定める。この意味において、学習率は温度である。二次損失の上では、反復列は OU 過程（:numref:`sec_mdl-ornstein-uhlenbeck`）の定常ガウス分布へと落ち着き、その分散は比 $\eta/B$ とともに増大する。:citet:`Mandt.Hoffman.Blei.2017` はこの性質を利用し、確率的勾配降下法を近似的な事後分布サンプラとして動かしている。ただしこのモデル化には小さな学習率とほぼガウス分布に従う勾配ノイズが必要であり、実用上はその両方の仮定が破れることがある。

<!--en:a5f93a21-->
### ニューラル SDE

<!--en:e056f93e-->
学習されるモデル自体が、学習されるドリフトと学習されるノイズ振幅 $g$ を持つ SDE であるとき、随伴系はどうなるのか。その場合、随伴系は同じブラウン経路で駆動される後向き SDE（backward SDE）となるため、そのブラウン経路を保存しておくか、乱数シードから再生成しなければならない :cite:`Li.Wong.Chen.ea.2020`。損失もまた変化する。このようなモデルの出力は確率的であるため、1 つの端点を 1 つの目標値と直接比較することはできない。訓練信号は、経路全体の法則（確率分布）間のダイバージェンス、あるいは経路同士を比較する識別器になる :cite:`Kidger.2022`。

<!--en:b376600f-->
ギルサノフの定理（Girsanov's theorem）:cite:`Oksendal.2003` は、同じ $g$ とドリフト $\mathbf f$ および $\mathbf f'$ を持つ 2 つの過程に対して、このダイバージェンスを閉じた形で与える。両者の経路法則間の KL ダイバージェンスは、初期法則間の KL ダイバージェンスに $\tfrac12\int\mathbb{E}\|\mathbf f-\mathbf f'\|^2/g^2\,dt$（期待値は第 1 の過程の経路に関するもの）を加えた値に等しい。ここで $g$ が一致するという条件は外せない。異なる $g$ を持つ 2 つの過程は、たった 1 本の経路からでもその二次変分（:numref:`sec_mdl-quadratic-variation`）によって見分けられるため、経路法則間の KL ダイバージェンスは無限大になる。

<!--en:68bbb405-->
:numref:`subsec_mdl-f-div-data-processing` のデータ処理不等式（data-processing inequality）により、端点の法則間の差は経路の法則間の差を超えられない。したがって、拡散モデルの厳密な逆ドリフトと学習された逆ドリフトにギルサノフの公式を適用すれば、演習問題 :exref:`ex_mdl-score-matching-diffusion-flow-likelihood-bound`（:numref:`sec_mdl-score-matching-diffusion-flow`）で経路法則を用いずに導出した尤度境界が得られる。

<!--en:88b03294-->
## 学習される生成器

<!--en:08b536d0-->
この方向で本章の先に続く内容の大半は :numref:`chap_diffusion` で扱う。予測対象、損失の重み付け、およびスケジュールについては :numref:`sec_diffusion-design-space` で解説しており、そこでは信号対雑音比（signal-to-noise ratio）の対数である $-2\log\bar\sigma_t$（:numref:`sec_mdl-ddim-guidance` の記法）によって各選択肢が整理される :cite:`Kingma.Salimans.Poole.ea.2021`。1 ステップまたは少数ステップでの生成は :numref:`sec_diffusion-few-step` で扱う。以下では、両章の範囲を超える 4 つのトピックを取り上げる。

<!--en:941c0c9a-->
### スライススコアマッチング

<!--en:b796e83f-->
:numref:`sec_mdl-score-matching` ではトレースの計算コストを理由に陰的スコアマッチング（implicit score matching）を脇に置き、そこでは採用しなかった抜け道として、ランダムな方向ごとに 1 回のベクトル・ヤコビ行列積で済む Hutchinson の推定値に言及した。この推定値を用いて訓練する手法はスライススコアマッチング（sliced score matching）と呼ばれる :cite:`Song.Garg.Shi.ea.2019`。この手法はノイズレベルを選ぶ必要がなく、データそのもののスコアを学習する。もっとも、拡散モデルにはこの利点の使い道がない。拡散モデルが必要とするのはあらゆるノイズレベルにおけるノイズ付きデータのスコアであり、ノイズ除去スコアマッチングはトレースなしでそれを提供するため、ノイズ除去が標準となったのである。

<!--en:dd71487c-->
### スコア誤差とサンプリング誤差

<!--en:c5634f2d-->
本章のほぼすべてのサンプラは厳密なスコアを用いて動かしたが、訓練されたスコアはどこにおいても厳密ではない。では、どれほど正確であればよいのだろうか。平均的に良ければ十分であり、それ以上は必要ない。:citet:`Chen.Chewi.Li.ea.2023` は、データ分布と離散化された逆 SDE の出力との間の全変動（total variation）を 3 つの項の和で上から抑えており、その各項はすでに本章で個別に登場している。$p_T$ とサンプラの出発点であるガウス分布との距離は $e^{-T}$ のオーダーで減衰し、離散化誤差は次元とともに増大しステップ幅とともに減少する。この 2 つは :numref:`sec_mdl-time-reversal` で測定した。第 3 の項はノイズ付きデータのもとでのスコアの二乗平均平方根誤差であり、これはノイズ除去損失がその下限からの超過分として測っている量にほかならない。この定理はそれ以外の項が一切入らないことを示しており、仮定もわずかである。すなわち、データの有限な二次モーメントと、ノイズ付き密度のスコアがリプシッツ連続であることだけを要求する。単峰性は不要であるため、:numref:`sec_mdl-langevin-sampling` でランジュバン動力学を阻んだ 2 つのモードも対象に含まれる。

<!--en:77190567-->
### シュレーディンガーブリッジとエントロピー輸送

<!--en:14cbdb7b-->
1931 年、Schrödinger は粒子に関するある問いを立てた。自由に拡散する粒子群が、現在ある形状で観測され、後の時刻に別の起こりそうにない形状で観測されたとき、そこへ至る最も確からしい道筋はどのようなものか、という問いである。独立な多数の粒子からなる群れでは、自由拡散とは異なる経路法則に従って群れが動くのを目にする確率は指数関数的に減少し、その減衰率は自由拡散に対するその経路法則の KL ダイバージェンスで与えられる。したがって答えは、与えられた 2 つの周辺分布を持つ経路法則のうち、KL ダイバージェンスの意味で自由拡散に最も近いものとなる。これを*シュレーディンガーブリッジ*と呼ぶ。

<!--en:636411c2-->
このブリッジを :numref:`sec_mdl-ot-connection` のミニバッチペアリングと並べてみよう。ミニバッチペアリングは各バッチ上での厳密な割り当てであり、訓練される生成器は ODE である。これに対しブリッジは、:numref:`subsec_mdl-sinkhorn` のシンクホーン反復によって計算されるエントロピー正則化最適輸送の計画 :cite:`Cuturi.2013,Peyre.Cuturi.2019` に従って両端を対応付け、その生成器は SDE となる。自由拡散のノイズが小さくなるにつれて、この輸送計画は最適輸送計画へと近づく。

<!--en:309ef511-->
拡散モデルと並べると、ブリッジの方が難しい問題である。拡散モデルは順過程を固定して逆過程だけを学習し、事前分布には近似的にしか到達しない。一方ブリッジは与えられた 2 つの法則を有限時間で結ぶため、両方向を学習しなければならない。:citet:`DeBortoli.Thornton.Heng.ea.2021` は順方向と逆方向の拡散を交互に適合させており、その交互反復の最初の 1 巡は拡散モデルの訓練に一致する。:citet:`Albergo.Boffi.VandenEijnden.2023` はブリッジ問題を解くことなく、ODE と SDE の両方の生成器に到達している。任意の 2 つの分布からのサンプル $\mathbf x_0$、$\mathbf x_1$ と、独立なガウス変数 $\mathbf w$ をとろう。経路 $a_t\mathbf x_0+b_t\mathbf x_1+c_t\mathbf w$（ただし両端で $c_t$ はゼロ）が持つ周辺分布は、1 つの ODE または SDE の族によって生成できる（演習問題 :exref:`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants`、:numref:`sec_mdl-flow-matching-transport`）。

<!--en:ea666aec-->
### 直線的カップリングと最適カップリング

<!--en:c1e1db73-->
Reflow によって変化しないペアリングは交差しない直線経路を持つが、2 次元以上ではそれが最適であるとは限らない。:numref:`sec_mdl-ot-connection` の例では、ガウス分布に従う点群の各点を、それを 4 分の 1 回転させた点と対応付けている。線分は決して交差しないが、コストが $4$ かかるのに対し、本来は $W_2^2=0$ である。このペアリングが持ち、最適なペアリングが持たないものは回転である。最適なフローの速度場は勾配であるが :cite:`Benamou.Brenier.2000`、点群を原点のまわりに回転させるベクトル場は勾配ではない。

<!--en:9c6d535b-->
そこで :citet:`Liu.2022` は速度場を勾配 $\nabla\phi_t$ に制限し、この制限付き Reflow によって変化しないペアリングが最適なペアリングになると論じた。しかし、この主張が成り立つにはそこで置かれた仮定以上の条件が必要である。:citet:`Hertrich.Chambolle.Delon.2025` は、移動中の点群があらゆる時刻で空間全体を覆っている場合にこれを証明する一方、点群が複数の離れた塊に分かれている場合の反例を与えている。したがってこの制限だけでは、:numref:`sec_mdl-ot-connection` が残した問い、すなわち訓練のみによって最適なペアリングをいかにして見つけるかという問いは決着しない。

<!--en:77fc72da-->
## ダイナミクスと生成フロー全般の参考資料

<!--en:d87a6a53-->
以下の文献を開く前に 1 点注意しておく。MIT の講義ノートと講義、Flow Matching Guide、およびフローマッチングに関する 2 編の論文では、時刻を $0$ のノイズから $1$ のデータへと進めるのに対し、本章ではその逆向きに進めている（:numref:`fig_mdl-dyn-time-conventions`）。

<!--en:cd0bff15-->
**書籍と講義ノート**

<!--en:89d51682-->
- 『[Nonlinear Dynamics and Chaos](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering)』（Steven Strogatz 著）：フロー、不動点とその安定性に関する幾何学的な入門書であり、:numref:`sec_mdl-odes-solvers` の内容を扱っている。

<!--en:a70411fb-->
- 『[On Neural Differential Equations](https://arxiv.org/abs/2202.02435)』（Patrick Kidger 著、2022 年）：ニューラル ODE、制御微分方程式、確率微分方程式を書籍規模で体系的に論じた文献であり、:numref:`sec_mdl-neural-odes-cnf` の随伴法の仕組みが余すところなく展開されているほか、:numref:`sec_mdl-euler-runge-kutta` のソルバの選び方に関する助言も収められている。

<!--en:5157c74d-->
- 『[Applied Stochastic Differential Equations](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf)』（Särkkä、Solin 著）：:numref:`sec_mdl-sdes` および :numref:`sec_mdl-fokker-planck-probability-flow` に対応する伊藤解析、SDE、フォッカー・プランク方程式を応用重視の視点から解説した書籍であり、著者らによって全文の PDF が公開されている。

<!--en:bdd1055b-->
- 『[Stochastic Differential Equations: An Introduction with Applications](https://link.springer.com/book/10.1007/978-3-642-14394-6)』（Bernt Øksendal 著）：:numref:`sec_mdl-sdes` の背景にあるブラウン運動、伊藤積分、および解の存在理論に関する厳密な基本文献である。

<!--en:cc2b7925-->
- 『[Numerical Solution of Stochastic Differential Equations](https://link.springer.com/book/10.1007/978-3-662-12616-5)』（Kloeden、Platen 著）：:numref:`sec_mdl-sdes` のシミュレーション部分に対応する、オイラー・丸山法、ミルスタイン法、より高次のスキーム、およびそれらの強収束次数・弱収束次数に関する基本文献である。

<!--en:21d945a4-->
- 「[An Introduction to Flow Matching and Diffusion Models (MIT 6.S184 lecture notes)](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)」（Holderrieth、Erives 著）：SDE と連続の式から拡散モデルとフローマッチングを導出する自己完結した講義ノートであり、:numref:`sec_mdl-score-matching-diffusion-flow` と :numref:`sec_mdl-flow-matching-transport` の副読本となる。

<!--en:e59c5bd6-->
**講義とチュートリアル**

<!--en:296e4fea-->
- 「[MIT 6.S184: Generative AI with Stochastic Differential Equations](https://diffusion.csail.mit.edu/)」：:numref:`sec_mdl-fokker-planck-probability-flow` から :numref:`sec_mdl-flow-matching-transport` への道筋に沿って第一原理から拡散モデルを構築する講義、スライド、および演習ラボであり、オープンライセンスで公開されている。

<!--en:d927865f-->
- 「[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)」（Calvin Luo 著、2022 年）：:numref:`sec_mdl-score-matching-diffusion-flow` に関連して、拡散モデルの変分的な見方とスコアベースの見方を並べて導出している。

<!--en:594ec7b6-->
- 「[Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)」（Lipman ら著、2024 年）：:numref:`sec_mdl-flow-matching-transport` 向けのフローマッチングに関する詳細な解説文献であり、PyTorch ライブラリ [`facebookresearch/flow_matching`](https://github.com/facebookresearch/flow_matching) と対になっている。

<!--en:d63f4e40-->
**基礎論文**

<!--en:c066a683-->
- 「[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)」（Chen ら著、NeurIPS 2018）：:numref:`sec_mdl-neural-odes-cnf` と同様に、ネットワークのダイナミクスを ODE としてパラメータ化し、随伴法を用いて訓練する。

<!--en:813c7ef8-->
- 「[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)」（Song、Ermon 著、NeurIPS 2019）：:numref:`sec_mdl-score-matching-diffusion-flow` に対応する、ノイズ除去スコアマッチングとアニーリング付きランジュバンサンプリングによるスコアベース生成を提案している。

<!--en:2265ceb8-->
- 「[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)」（Ho、Jain、Abbeel 著、NeurIPS 2020）：:numref:`sec_mdl-score-matching-diffusion-flow` で離散化された SDE として解釈した、離散時間の DDPM 定式化を与えている。

<!--en:cdc87fc5-->
- 「[Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)」（Song ら著、ICLR 2021）：スコアマッチングと拡散モデルを順方向・逆方向の SDE として統一し、:numref:`sec_mdl-fokker-planck-probability-flow` の確率フロー ODE を導入している。

<!--en:30f50e8f-->
- 「[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)」（Song、Meng、Ermon 著、ICLR 2021）：訓練済みの DDPM を再利用してより少ないステップ数で生成を行う、:numref:`sec_mdl-score-matching-diffusion-flow` の決定論的サンプラである。

<!--en:4f139db9-->
- 「[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)」（Lipman ら著、ICLR 2023）：条件付きベクトル場の回帰によって連続フローをシミュレーションなしで訓練する、:numref:`sec_mdl-flow-matching-transport` の手法である。

<!--en:b2ee27b6-->
- 「[Flow Straight and Fast: Rectified Flow](https://arxiv.org/abs/2209.03003)」（Liu、Gong、Liu 著、ICLR 2023）：:numref:`sec_mdl-flow-matching-transport` に対応する、少数ステップサンプリングのための直線補間経路と Reflow を提案している。

<!--en:959bee44-->
**ブログと視覚的解説**

<!--en:47132f47-->
- 「[Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)」（Yang Song 著）：スコアマッチング、ランジュバン動力学、および :numref:`sec_mdl-score-matching-diffusion-flow` の SDE による描像を図とともに結びつける著者自身の解説記事である。

<!--en:15e13d3e-->
- 「[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)」（Lilian Weng 著）：DDPM からスコア SDE やガイダンスに至る拡散モデルのサーベイ記事であり、2024 年まで更新されている（:numref:`sec_mdl-score-matching-diffusion-flow` に対応）。

<!--en:de445ead-->
- 「[Perspectives on Diffusion](https://sander.ai/2023/07/20/perspectives.html)」（Sander Dieleman 著）：拡散モデルがいかにしてノイズ除去器、スコア推定器、逆 SDE ソルバ、そしてフローという顔を併せ持つかを示す記事であり、:numref:`sec_mdl-score-velocity-dictionary` の対応辞書にあたる。

<!--en:b5b5d4a7-->
- 「[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)」（Hugging Face）：:numref:`sec_mdl-score-matching-diffusion-flow` の数学を実行可能なコードへと結びつける、PyTorch による DDPM の 1 行ごとの実装解説である。

<!--en:48a210aa-->
[ディスカッション](https://d2l.discourse.group/t/36870)
