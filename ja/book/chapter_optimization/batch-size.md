<!--en:cadd2e6d-->
# バッチサイズ

<!--en:2bacf06e-->
実際の現場では、ミニバッチサイズ $b$ はアクセラレータのメモリを埋めるように選ばれ、そのまま変更されないことが多い。しかし、訓練を複数の GPU にスケールさせたり、訓練の実経過時間と総計算量（FLOPs）をトレードオフしたりする場面では、根本的な問いに直面する。ミニバッチサイズ $b$ を 2 倍にして（ステップあたりの並列計算量を 2 倍にして）訓練したとき、半分のステップ数で目標の損失に到達するのだろうか。

<!--en:169b74b1-->
:numref:`sec_sgd` で見たように、$b$ 個の例からなるミニバッチを用いると、勾配のサンプリング分散は $\operatorname{tr}(\boldsymbol{\Sigma})/b$ に減少する。ここから、次の 2 つの異なる領域が導かれる。

<!--en:62d29a2a-->
1. **小バッチ（ノイズ支配）領域：**サンプリングノイズが真の勾配より大きいとき、$b$ を 2 倍にするとノイズ分散が半分になり、より大きな学習率を使えるようになる。その結果、必要なステップ数 $S(b)$ は半分に減り、処理される総データ例数 $E(b) = b\,S(b)$ は一定に保たれる（*完全な線形スケーリング*）。

<!--en:1c4d282c-->
2. **大バッチ（曲率支配）領域：**$b$ が十分に大きくなり、ミニバッチ勾配が真のフルバッチ勾配を精度よく近似するようになると、さらに $b$ を増やしても $S(b)$ の減少幅は逓減する一方、データと計算のコスト $E(b)$ は線形に増大する。

<!--en:754e990d-->
:citet:`McCandlish.Kaplan.Amodei.ea.2018` は、これら 2 つの領域の遷移が**臨界バッチサイズ**（critical batch size）$b_{\text{crit}}$ で生じること、そしてこの値が**勾配ノイズスケール**（gradient-noise scale）$b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$ によって予測できることを示した。本節では、$b_{\text{noise}}$ の 2 バッチ推定量を構築し、ステップ数とデータ量のトレードオフのフロンティア、ならびに SGD（$\eta \propto b$）と Adam（$\eta \propto \sqrt{b}$）における学習率のスケーリング則を導出し、その両方の予測を `TinyLM` と `FashionCNN` で検証する。

<!--en:0bfe3da5-->
## 勾配ノイズスケール

<!--en:dd283bf6-->
### 信号、ノイズ、2 バッチ推定量

<!--en:a7bb4e29-->
データセット全体の厳密な勾配を $\nabla f(\mathbf{x})$ とし、ミニバッチ勾配 $\hat{\mathbf{g}}_b$ を独立な $b$ 個の例にわたる平均（1 例あたりの勾配の共分散は $\boldsymbol{\Sigma}$）とする。$\mathbb{E}[\hat{\mathbf{g}}_b] = \nabla f$ かつ $\mathbb{E}[\|\hat{\mathbf{g}}_b - \nabla f\|^2] = \operatorname{tr}(\boldsymbol{\Sigma})/b$ であるため、ミニバッチ勾配の二乗ノルムの期待値は、信号パワー $\|\nabla f\|^2$ とノイズパワー $\operatorname{tr}(\boldsymbol{\Sigma})/b$ の和に分解される。

<!--en:d8fc180b-->
信号パワーとノイズパワーは、次の**単純勾配ノイズスケール**（simple gradient-noise scale）において等しくなる :cite:`McCandlish.Kaplan.Amodei.ea.2018`。

<!--en:c165652c-->
$\operatorname{tr}\boldsymbol{\Sigma}$ と $\|\nabla f\|^2$ はいずれも、データセット全体を走査しなくても推定できる。具体的には、二乗勾配ノルムの平均 $N_b = \mathbb{E}[\|\hat{\mathbf{g}}_b\|^2]$ を 2 つのバッチサイズ $b_{\text{s}} < b_{\text{l}}$ で測定し、2 元連立一次方程式 :eqref:`eq_sqnorm-bias` を解けばよい。

<!--en:829be3bc-->
まず WikiText-103 と Fashion-MNIST をアクセラレータに読み込み、ランダムなミニバッチサンプラ `sample_fashion(b)` と `sample_lm(b)` を定義する。

<!--en:395800a5-->
これらのサンプラを用いて、交差エントロピーの補助関数 `ce_loss`、全パラメータにわたる二乗勾配ノルム `grad_sq_norm`、および 2 バッチ推定量 `noise_scale` を `d2l` に保存する。

<!--en:4249b65b-->
それでは、`noise_scale` を `TinyLM`（$b_{\text{s}} = 8, b_{\text{l}} = 256$）に対して、初期化時点と $500$ ステップの Adam（$\eta = 3 \times 10^{-3}$）を実行した後の時点で評価してみよう。

<!--en:1fd356fa-->
画像に対する畳み込みネットワークのノイズスケールも同様に増大するのだろうか。デフォルトの $b_{\text{s}} = 16$ 枚および $b_{\text{l}} = 2{,}048$ 枚の画像を用いて、`FashionCNN` でも初期化時点と $200$ ステップの Adam（$\eta = 0.004$、後述のスイープで $b = 256$ のときに用いる学習率）を実行した後の時点で同じ測定を繰り返す。

<!--en:9c4ab989-->
初期化時点の `TinyLM` では、どのウィンドウも大まかなユニグラム分布への同じ調整を要求するため、信号 $\|\nabla f\|^2$ が大きく、$b_{\text{noise}} \approx 2$ 系列にとどまる。頻出バイトの統計量が学習されると（ステップ $500$ までには）、$\|\nabla f\|^2$ が低下する一方で系列間の分散 $\operatorname{tr}\boldsymbol{\Sigma}$ は残るため、$b_{\text{noise}}$ は 10 倍以上に増大して $\sim 50\text{--}150$ 系列となる。

<!--en:47ce57dd-->
`FashionCNN` はこれとは異なる挙動を示す。10 個のクラスが均衡しているため、未訓練のネットワークにはすべての画像で一致する調整方向が存在せず、ノイズスケールは最初のステップから $100$ 枚のオーダーになる。$200$ ステップの間、推定値の変動は実行ごとに増減いずれの方向にも 2 倍未満にとどまり、$100$ 枚のオーダーのままである。

<!--en:5183c8d4-->
## バッチサイズ、ステップ効率、学習率のスケーリング

<!--en:33ca62ec-->
### 最適ステップサイズと効率フロンティア

<!--en:c4201e54-->
$L$ 滑らかな降下上界 $f(\mathbf{x} - \eta\hat{\mathbf{g}}_b) - f(\mathbf{x}) \le -\eta\nabla f^\top \hat{\mathbf{g}}_b + \frac{L}{2}\eta^2\|\hat{\mathbf{g}}_b\|^2$ の期待値をとり、:eqref:`eq_sqnorm-bias` を代入すると次式が得られる。

<!--en:9dfd9547-->
右辺を $\eta$ について最小化すると、最適ステップサイズ $\eta_{\text{opt}}(b)$ と 1 ステップあたりの保証減少量が得られる。

<!--en:2485581c-->
*（スカラーの滑らかさの上界 $L\mathbf{I}$ を局所ヘッセ行列 $\mathbf{H}$ に置き換えても、曲率重み付きノイズスケール $\operatorname{tr}(\mathbf{H}\boldsymbol{\Sigma}) / (\nabla f^\top\mathbf{H}\nabla f)$ を用いたまったく同じ形の式が得られ :cite:`McCandlish.Kaplan.Amodei.ea.2018`、これに対して :eqref:`eq_noise-scale` の $b_{\text{noise}}$ はヘッセ行列不要の代理指標として機能する。）*

<!--en:b16a653f-->
バッチサイズ $b$ における各ステップは、厳密なフルバッチステップに対して $(1 + b_{\text{noise}}/b)^{-1}$ の割合だけ前進するため、フルバッチで $S_{\min}$ ステップを要する目標損失に到達するには、ステップ数 $S(b)$ と総訓練例数 $E(b) = b\,S(b)$ が次式のように必要となる。

<!--en:9f953a45-->
ここで、$E_{\min} = b_{\text{noise}} S_{\min}$ は小バッチ極限 $b \ll b_{\text{noise}}$ で達成される最小データ予算である。**臨界バッチサイズ** $b_{\text{crit}} = b_{\text{noise}}$ においては、ステップ数とデータ例数の双方が理論的最小値の 2 倍（$S = 2S_{\min}, E = 2E_{\min}$）となり、パレートフロンティア $((E/E_{\min} - 1)(S/S_{\min} - 1) = 1$) の屈曲点（knee）をなす。

<!--en:f2fe8d67-->
### 学習率の線形スケーリングと平方根スケーリング

<!--en:dcf4be1b-->
バッチサイズを $b_0$ から $b$ へ増やすとき、ノイズ支配領域（$b_0, b \ll b_{\text{noise}}$）では学習率 $\eta$ をどのようにスケーリングすべきだろうか。

<!--en:2d22cced-->
* **SGD とモメンタム（線形スケーリング）：**先の :eqref:`eq_best-step` より、$b \ll b_{\text{noise}}$ のとき $\eta_{\text{opt}}(b) \approx \frac{b}{L b_{\text{noise}}} \propto b$ となる。言い換えれば、$k$ 回のステップ（サイズ $\eta_0$、バッチサイズ $b_0$）は、サイズ $k\eta_0$、バッチサイズ $k b_0$ による 1 回のステップと一致し、同時に定常ノイズフロア $\propto \eta / b$ も保たれる :cite:`Goyal.Dollar.Girshick.ea.2017,Smith.Kindermans.Ying.ea.2018`。

<!--en:720c7ef6-->
* **Adam と AdamW（平方根スケーリング）：**ある座標の勾配が平均 $\bar{g}$ と標準偏差 $\sigma / \sqrt{b}$ を持つノイズ支配領域では、Adam は勾配をその RMS スケール $\sqrt{\hat{v}_t} \approx \sigma / \sqrt{b}$ で正規化するため、1 ステップあたりのドリフト $-\eta(\sqrt{b}/\sigma)\bar{g}$ とステップノイズ $-\eta z$（$z \sim \mathcal{N}(0, 1)$）が生じる。分母の $\sigma/\sqrt{b}$ がすでに $\sqrt{b/b_0}$ 倍の因子を与えるため、$k = b/b_0$ 回の小ステップのドリフトと累積分散に 1 回の大きなステップで一致させるには、$\eta$ を $\sqrt{b/b_0}$ 倍にスケーリングする必要がある :cite:`Malladi.Lyu.Panigrahi.ea.2022`。

<!--en:8470d842-->
$b$ が $b_{\text{crit}}$ を超えると、勾配ノイズはもはや分母で支配的ではなくなり、最適学習率は $b$ とともに増大し続けるのではなく、決定論的な曲率による上限で飽和する :cite:`Shallue.Lee.Antognini.ea.2019`。

<!--en:62c64e47-->
## 実験的検証：目標到達ステップ数とスケーリング則

<!--en:9e42aae0-->
固定したホールドアウトの目標損失に到達するまでのステップ数 $S(b)$ を測定するため、まず `eval_loss` を定義し、検証テンソル上の `ce_loss` を $4{,}096$ 例ごとのチャンクに分けて評価できるようにする。

<!--en:85cfdc94-->
停止ハーネス `train_to_target` は、`eval_loss` を `eval_every` ステップごとに `d2l.train_steps` 内で確認し、`target` に到達したステップ数を返す（`max_steps` を使い切った場合は `inf` を返す）。

<!--en:e0019e25-->
まず共通の初期化から、`train_to_target` を `TinyLM`（目標検証損失 $1.9$ nats/byte）に対して $b \in \{4, 16, 64, 256\}$ 系列で実行し、Adam の学習率を $\eta = 0.003\sqrt{b / 64}$ に従ってスケーリングする。

<!--en:6937fccb-->
同じ目標到達ステップ数のスイープを `FashionCNN`（目標損失 $0.35$）に対しても $b \in \{8, 32, 128, 512, 2048\}$ 枚の画像で繰り返し、学習率を $\eta = 0.004\sqrt{\min(b, 512)/256}$ として $b = 512$ までスケーリングする。

<!--en:ab6caecc-->
$S(b) = S_{\min}(1 + b_{\text{crit}}/b)$ は $1/b$ に関して線形であるため、`fit_steps` は重み付き最小二乗法によって両モデルの漸近的な最小ステップ数 $S_{\min}$ と実験的な臨界バッチサイズ $b_{\text{crit}}$ を抽出する。

<!--en:e9818edb-->
各実行のデータ例数 $E(b) = b\,S(b)$ を $E_{\min} = b_{\text{crit}} S_{\min}$ で正規化し、ステップ数を $S_{\min}$ で正規化することで、両モデルを普遍的なパレートフロンティア :eqref:`eq_steps-examples` と重ねてプロットする。

<!--en:f7017527-->
いずれのアーキテクチャも普遍曲線 $(E/E_{\min} - 1)(S/S_{\min} - 1) = 1$ の近くに並ぶ（$E = E_{\min}$ に近い点は曲線の急峻な壁の上に位置するため、測定されたステップ数のわずかな誤差によって上下に大きく動く）。当てはめられた臨界バッチサイズは、`TinyLM` では $b_{\text{crit}} \sim 100$ 系列、`FashionCNN` では $b_{\text{crit}} \sim 200\text{--}400$ 枚である。`TinyLM` では、この当てはめ結果は先ほど測定した $b_{\text{noise}}$ の 2 バッチ推定値（$100$ 系列をやや下回る値）とよく一致する。一方、`FashionCNN` では約 $100$ 枚という推定値の 2〜4 倍となっている。単純ノイズスケールは曲率を無視するため、$b_{\text{crit}}$ の正確な値ではなくオーダー（桁）を与えるのである。

<!--en:dbd79c7e-->
最後に、線形（$\eta \propto b$）と平方根（$\eta \propto \sqrt{b}$）の学習率スケーリング則 :eqref:`eq_lr-rules` を、`FashionCNN` 上で $12{,}800$ 枚（$1/5$ エポック）の固定データ予算のもと直接検証する。補助関数 `final_loss_at` は、`d2l.lr_sweep` を `num_examples // b` ステップだけバッチサイズ `b` で実行し、各実行の最後の 1/4 区間にわたって平均した最終損失を返す。

<!--en:898e4ff2-->
モメンタム付き SGD（$\beta = 0.9$）では、バッチサイズを $8\times$（$b = 8$ から $b = 64$ へ）増やして `sgd_lrs` にわたり探索すると、最適学習率は右へ $4\text{--}8\times$ シフトする（$b = 64$ では最終損失が $\eta = 0.05$ から $0.1$ の間でほぼ平坦になる）。

<!--en:cf749ca5-->
対照的に、Adam のもとでバッチサイズを $64\times$（$b = 2$ から $b = 128$ へ）増やして `adam_lrs` にわたり探索すると、最適学習率のシフトはわずか $4\text{--}8\times$（$2$ から $3$ 段階分の $2\times$ グリッドステップ）にとどまり、$\sqrt{64} = 8\times$ と一致して $64\times$ にはならない。

<!--en:439f8d23-->
### バッチサイズを選択する実践的手順

<!--en:de13f5a4-->
1. **まずハードウェアの演算器を飽和させる：**1 ステップあたりの実行時間が $b$ に対して線形に増加し始めるまで、少なくとも $b$ を大きくする（:numref:`sec_sgd`）。

<!--en:23ea1577-->
2. **ウォームアップ後に $b_{\text{noise}}$ を推定する：**初期の損失低下の後（たとえば訓練開始から数百ステップ後）に `noise_scale` を測定する。`TinyLM` で見られたように、$b_{\text{noise}}$ は初期化時点ではきわめて小さく、訓練中に $1\text{--}2$ 桁増大することがあるためである（これが GPT-3 などの大規模な事前学習でバッチサイズのウォームアップランプが用いられる理由である :cite:`brown2020language`）。

<!--en:e9f0228a-->
3. **$b$ を $b_{\text{crit}} \approx b_{\text{noise}}$ に照らして設定する：**計算量やデータがボトルネックである場合は $b \lesssim 0.3\,b_{\text{crit}}$ を選ぶ。並列アクセラレータを用いた実経過時間がボトルネックである場合は $b \approx b_{\text{crit}}$（このとき $S = 2S_{\min}$ かつ $E = 2E_{\min}$）を選び、$b \gg b_{\text{crit}}$ は避ける。

<!--en:01eccfbd-->
4. **$\eta$ を :eqref:`eq_lr-rules` に従ってスケーリングする：**SGD・モメンタムでは $\eta \propto b$、AdamW では $\eta \propto \sqrt{b}$ に従って $b_{\text{crit}}$ までスケーリングし、線形ウォームアップと組み合わせる（:numref:`subsec_scheduler-warmup`）。

<!--en:30ac03ff-->
## まとめ

<!--en:3bc7148b-->
**勾配ノイズスケール** $b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$ は、サンプリングのノイズパワーと信号パワーが等しくなるバッチサイズを表し、データセット全体の走査を行わずに 2 つのバッチサイズ $b_{\text{s}} < b_{\text{l}}$ における二乗勾配ノルムからオンラインで推定できる（:eqref:`eq_sqnorm-bias`）。適切に調整された学習率のもとでは、目標損失への到達に必要なステップ数 $S(b)$ とデータ例数 $E(b) = b\,S(b)$ は、双曲線型のトレードオフ $S(b)/S_{\min} = 1 + b_{\text{crit}}/b$ および $E(b)/E_{\min} = 1 + b/b_{\text{crit}}$ に従う。この関係は屈曲点 $b_{\text{crit}} \approx b_{\text{noise}}$ を境として、データ効率の高い小バッチ領域（$b \ll b_{\text{crit}}$）とステップ数が飽和する大バッチ領域（$b \gg b_{\text{crit}}$）とを分ける。$b_{\text{crit}}$ 未満では、最適学習率は SGD とモメンタムでは線形（$\eta \propto b$）に、Adam と AdamW では平方根（$\eta \propto \sqrt{b}$）に従ってスケーリングし、$b$ が $b_{\text{crit}}$ を超えると頭打ちになる。

<!--en:2294243a-->
## 演習問題

<!--en:788c2e66-->
1. **学習率を固定したままのバッチサイズ拡大。**本節の :eqref:`eq_noisy-step` と :eqref:`eq_best-step` を用いて、$\eta$ を $\eta_0 = \eta_{\text{opt}}(b_0)$ に固定したままバッチサイズを $b > b_0$ に増やすと、1 ステップあたりの保証減少量は $2 - (1 + b_{\text{noise}}/b)/(1 + b_{\text{noise}}/b_0) < 2$ 倍に改善されるものの、$b \to \infty$ としても決して 2 倍を超えないことを示せ。
   :exlabel:`ex_batch-size-fixed-rate`

<!--en:6bb1aa02-->
1. **時間最適バッチサイズと計算量最適バッチサイズ。**バッチサイズ $b$ における 1 ステップの実経過時間が $t(b) = t_0(1 + b/b_{\text{sat}})$ であるとき、:eqref:`eq_steps-examples` を用いて最適なバッチサイズ $b^\star$（総実経過時間 $T(b) = t(b)S(b)$ を最小化する値）を導出し、そのデータオーバーヘッド $E(b^\star)/E_{\min}$ を求めよ。
   :exlabel:`ex_batch-size-time-optimal-compute-optimal`

<!--en:6b6b083c-->
1. [code] **訓練に伴う信号とノイズの時間変化。**関数 `noise_scale` を変更して $\operatorname{tr}\boldsymbol{\Sigma}$、$\|\nabla f\|^2$、およびそれらの比 $b_{\text{noise}}$ を個別に返すようにし、`TinyLM` 上でステップ $0, 25, 50, 100, 200, 500, 1000, 2000, 3000$ においてこれら 3 つすべてを追跡せよ。そして、$100\times$ におよぶ $b_{\text{noise}}$ の増大が、主に信号 $\|\nabla f\|^2$ の低下（$\operatorname{tr}\boldsymbol{\Sigma}$ の増加ではなく）によって生じていることを確認せよ。
   :exlabel:`ex_batch-size-noise-scale-over-training`

<!--en:718b1d9a-->
1. [code] **臨界バッチサイズを超えたスケーリング。**本節の `TinyLM` に対する目標到達ステップ数のスイープ（:numref:`subsec_batch-size-steps`）を $b = 1{,}024$ まで拡張し、平方根則 $\eta \propto \sqrt{b}$ に従う場合と $\eta$ を $b = 256$ での値に固定する場合とを比較せよ。さらに、$\eta$ を $b_{\text{crit}}$ を超えてスケーリングし続けると収束が悪化する理由を説明せよ。
   :exlabel:`ex_batch-size-past-elbow`

<!--en:27eaf5d4-->
[ディスカッション](https://d2l.discourse.group/t/36765)

<!--en:d8dc56bb-->
バッチサイズ、勾配ノイズスケール、臨界バッチサイズ<br>**信号パワーとノイズパワー・2 バッチ推定量・ステップ数対データ例数のパレートフロンティア・臨界バッチサイズ・線形および √b 学習率則**

<!--en:b719757b-->
::: {.slide title="バッチサイズを 2 倍にするとステップ数はいつ半減するか？"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:ad38c601-->
- **小バッチ（$b: 4 \to 16$、$4\times$）：**ノイズ支配領域であり、平均化でサンプリング分散 $\operatorname{tr}(\boldsymbol{\Sigma})/b$ が減少して**ステップ数が $\approx 3\times$ 減少**し、データコストはほぼ一定（$\approx 1.1\text{--}1.3\times E(4)$）

<!--en:43fbae7a-->
- **大バッチ（$b: 64 \to 256$、$4\times$）：**曲率支配領域であり、ミニバッチ勾配はすでに $\nabla f$ とほぼ一致するため、**ステップ数の削減は $\approx 2\times$** にとどまり、データと FLOP のコストは 2 倍に増加

<!--en:96eb71f5-->
- **屈曲点（$b_{\text{crit}} \approx b_{\text{noise}}$）：**勾配ノイズスケール $b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$ によって決まる（McCandlish et al., 2018）

<!--en:263794ad-->
::: {.slide title="信号パワー、ノイズパワー、勾配ノイズスケール"}

<!--en:797916de-->
[ミニバッチの信号対雑音比]{.kicker}

<!--en:4195f4ab-->
- **二乗ノルムの期待値の分解：**
  $$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \underbrace{\|\nabla f\|^2}_{\text{Signal power}} + \underbrace{\frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}}_{\text{Noise power}}$$

<!--en:fe05393a-->
- **単純勾配ノイズスケール：**ノイズパワーと信号パワーが等しくなるバッチサイズ
  $$b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$$

<!--en:743d8572-->
- $b \ll b_{\text{noise}}$ では $\hat{\mathbf{g}}_b$ の大半がサンプリングノイズであり、$b \gg b_{\text{noise}}$ では $\hat{\mathbf{g}}_b \approx \nabla f$ となる

<!--en:07aa677e-->
::: {.slide title="データセット全体の走査を要しない b_noise の 2 バッチ推定量"}

<!--en:027e1295-->
[オンラインでのノイズスケール推定]{.kicker}

<!--en:19b86129-->
- $\|\nabla f\|^2$ を直接計算するには、データセット全体の走査が必要になる

<!--en:754f274c-->
- 代わりに、二乗勾配ノルムの平均 $N_b = \mathbb{E}[\|\hat{\mathbf{g}}_b\|^2]$ を 2 つのミニバッチサイズ $b_{\text{s}} < b_{\text{l}}$ で測定し、$2 \times 2$ の連立一次方程式 $N_b = \|\nabla f\|^2 + \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$ を解く
  $$\operatorname{tr}\boldsymbol{\Sigma} \approx \frac{N_{b_{\text{s}}} - N_{b_{\text{l}}}}{1/b_{\text{s}} - 1/b_{\text{l}}}$$

<!--en:add7615f-->
- 信号ノルムは $\|\nabla f\|^2 \approx \frac{b_{\text{l}} N_{b_{\text{l}}} - b_{\text{s}} N_{b_{\text{s}}}}{b_{\text{l}} - b_{\text{s}}}$ となる。たとえば $N_{16} = 0.50$、$N_{2048} = 0.103$ ならば、$\operatorname{tr}\boldsymbol{\Sigma} \approx 6.4$、$\|\nabla f\|^2 \approx 0.10$、$b_{\text{noise}} \approx 64$ である。

<!--en:713ef73f-->
::: {.slide title="TinyLM の訓練に伴う b_noise の増大"}

<!--en:1494161b-->
[ノイズスケールの動的な変化]{.kicker}

<!--en:cb84686c-->
- `noise_scale` を `TinyLM`（$b_{\text{s}} = 8, b_{\text{l}} = 256$ 系列）上で $500$ ステップの Adam（$\eta = 3 \times 10^{-3}$）の前後で測定
  - **初期化時点（ステップ 0）：**$b_{\text{noise}} \approx 2$ 系列。未訓練のどのウィンドウも同じ大域的なバイト頻度の補正を求めるため、揃った信号 $\|\nabla f\|^2$ が非常に大きい。
  - **500 ステップ後：**$b_{\text{noise}}$ は **$>30\times$ の $\sim 50\text{--}150$ 系列**へと増大（$\|\nabla f\|^2$ が低下する一方、系列間の不一致 $\operatorname{tr}\boldsymbol{\Sigma}$ が残るため）

<!--en:aa9de17e-->
::: {.slide title="ノイズ下での最適ステップサイズと 1 ステップの前進量"}

<!--en:8d7a0212-->
[ノイズ下での降下上界]{.kicker}

<!--en:0088a06a-->
- **ノイズ下での $L$ 滑らかな降下上界：**
  $$\mathbb{E}[f(\mathbf{x} - \eta\hat{\mathbf{g}}_b)] - f(\mathbf{x}) \le -\eta\|\nabla f\|^2 + \frac{L}{2}\eta^2\|\nabla f\|^2\Big(1 + \frac{b_{\text{noise}}}{b}\Big)$$

<!--en:622e330d-->
- **最適ステップサイズと 1 ステップの減少量：**
  $$\eta_{\text{opt}}(b) = \frac{1}{L\,(1 + b_{\text{noise}}/b)}, \quad \Delta f_{\text{opt}}(b) = \frac{\|\nabla f\|^2}{2L\,(1 + b_{\text{noise}}/b)}$$

<!--en:f02ba678-->
- $b = b_{\text{noise}}$ では、$\eta_{\text{opt}}$ と $\Delta f_{\text{opt}}$ はともにフルバッチ（$b \to \infty$）での値の**ちょうど半分**になる

<!--en:f3a41488-->
::: {.slide title="ステップ数・データ量のパレートフロンティアと臨界バッチサイズ"}

<!--en:7bcf4b44-->
[普遍的なトレードオフ則]{.kicker}

<!--en:b548adb0-->
- **ステップ数 $S(b)$ とデータ例数 $E(b) = b\,S(b)$（$E_{\min} = b_{\text{crit}}S_{\min}$）：**
  $$\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b}, \qquad \frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$$

<!--en:b8c8314d-->
- **小バッチ（$b = 0.1\,b_{\text{crit}}$）：**$11\,S_{\min}$ ステップ、データはわずか $1.10\,E_{\min}$

<!--en:4d16c020-->
- **臨界屈曲点（$b = b_{\text{crit}}$）：**バランスのとれた **$2\,S_{\min}$ ステップと $2\,E_{\min}$ のデータ**

<!--en:13532b42-->
- **過大なバッチ（$b = 10\,b_{\text{crit}}$）：**ステップ数の削減は $<50\%$（対 $b_{\text{crit}}$）にとどまり、データ量は $5.5\times$（$11\,E_{\min}$）に膨張

<!--en:cfd6e915-->
::: {.slide title="なぜ SGD は b_crit 未満で学習率の線形スケーリング（η ∝ b）を使うのか"}

<!--en:7a9e2359-->
[非正規化型オプティマイザの線形スケーリング則]{.kicker}

<!--en:59405d5a-->
- **小バッチ極限（$b \ll b_{\text{noise}}$）：**
  $$\eta_{\text{opt}}(b) = \frac{1}{L\,(1 + b_{\text{noise}}/b)} \approx \frac{b}{L\,b_{\text{noise}}} \implies \eta(b) = \frac{b}{b_0}\eta_0$$

<!--en:31a689fa-->
- **軌跡とノイズフロアの等価性（Goyal et al., 2017）：**$k = b/b_0$ 回の $(\eta_0, b_0)$ での小ステップは、$(k\eta_0, kb_0)$ での 1 ステップと等しく、$\eta/b$（および過剰損失 $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}}{4b}$）を一定に保つ

<!--en:f36d7d5d-->
- **$b_{\text{crit}}$ を超えた後の飽和：**$b \gg b_{\text{crit}}$ では $\eta_{\text{opt}} \to 1/L$ となり、$b_{\text{crit}}$ を超えて線形スケーリングを続けると $\eta < 2/L$ に違反する

<!--en:75643f1f-->
::: {.slide title="なぜ Adam は b_crit 未満で平方根スケーリング（η ∝ √b）を使うのか"}

<!--en:03b6e839-->
[適応的オプティマイザの平方根スケーリング則]{.kicker}

<!--en:d2a0edf1-->
なぜ **Adam / AdamW** は、ノイズ支配領域において線形スケーリングではなく**平方根スケーリング** $\eta(b) = \sqrt{b/b_0}\,\eta_0$ を必要とするのだろうか（Malladi et al., 2022）。

<!--en:253ba539-->
- ミニバッチ勾配が真の平均 $\bar{g}$ とサンプリングノイズ $z\,\frac{\sigma}{\sqrt{b}}$（$z \sim \mathcal{N}(0, 1)$）を持ち、$\frac{\sigma}{\sqrt{b}} \gg |\bar{g}|$ である座標を考える

<!--en:d16b6c14-->
- **Adam の分母がすでに $\sqrt{b}$ に応じて変化する：**$\sqrt{\hat{v}_t} \approx \sqrt{\bar{g}^2 + \sigma^2/b} \approx \frac{\sigma}{\sqrt{b}}$ であるため、$\sqrt{\hat{v}_t}$ で割ることで信号ドリフトが自動的に $\sqrt{b}$ 倍に増幅される
  $$\Delta x_t \approx -\eta\,\frac{\bar{g} + z\,\sigma/\sqrt{b}}{\sigma/\sqrt{b}} = \underbrace{-\eta\,\frac{\sqrt{b}}{\sigma}\,\bar{g}}_{\text{Deterministic drift}} \;-\; \underbrace{\eta\,z}_{\text{Unit-variance step noise}}$$

<!--en:d180120e-->
- **$k = b/b_0$ 回の小ステップに $1$ 回の大きなステップで一致させる：**$k$ 回の $(\eta_0, b_0)$ での小ステップにわたる累積ドリフトは $k \cdot \eta_0\frac{\sqrt{b_0}}{\sigma}\bar{g}$、累積ノイズ分散は $k\,\eta_0^2$ となる。バッチサイズ $b = k b_0$ の 1 ステップが累積ドリフトと累積分散の**両方**に一致するのは次のときである。
  $$\eta(b) = \sqrt{k}\,\eta_0 = \sqrt{\frac{b}{b_0}}\;\eta_0$$

<!--en:5ffde008-->
::: {.slide title="TinyLM における目標到達ステップ数の実験的スケーリング"}

<!--en:ea2b44d4-->
[言語モデルのスイープ]{.kicker}

<!--en:5814ab3c-->
- `TinyLM` を Adam（$\eta = 0.003\sqrt{b/64}$）でホールドアウト検証損失 $1.9$ nats まで、$b \in \{4, 16, 64, 256\}$ 系列にわたり訓練
  - **$b = 4 \to 16$（$4\times$）：**ステップ数は約 **$3\times$** 減少し（$\approx 4{,}000 \to 1{,}200$）、破線の完全スケーリング線（$S \propto 1/b$）によく追従
  - **$b = 64 \to 256$（$4\times$）：**ステップ数の減少は約 **$2\times$**（$\approx 430 \to 215$）にとどまり、処理する総系列数は倍増（$\approx 27{,}000 \to 55{,}000$、$b$ が $b_{\text{crit}} \sim 100$ を通過するため）

<!--en:0946aceb-->
::: {.slide title="FashionCNN における目標到達ステップ数の実験的スケーリング"}

<!--en:e864ea36-->
[画像分類器のスイープ]{.kicker}

<!--en:2a3e27f0-->
- `FashionCNN` を Adam で損失 $0.35$ まで、$b \in \{8, 32, 128, 512, 2048\}$ 枚の画像にわたり訓練
  - **$b = 8 \to 32 \to 128$：**ほぼ線形のステップ数削減（$2{,}300\text{--}3{,}000$ ステップから約 $700$、さらに約 $240$ へ）が得られ、総データ例数のコストは緩やかに増加（$\sim 19{,}000\text{--}32{,}000$ 枚）
  - **$b = 512 \to 2{,}048$（$4\times$）：**ステップ数の削減は約 3 分の 1（$\approx 120 \to 80$）にとどまり、必要データ量はほぼ 3 倍の $163{,}840$ 枚に増加

<!--en:252d2a8d-->
::: {.slide title="S_min と b_crit の 1/b に関する最小二乗法による当てはめ"}

<!--en:cafa72dc-->
[臨界バッチサイズの抽出]{.kicker}

<!--en:353e1abf-->
- $S(b) = S_{\min}\!\left(1 + \frac{b_{\text{crit}}}{b}\right) = S_{\min} + (S_{\min}b_{\text{crit}})\frac{1}{b}$ は $1/b$ に関して線形であるため、重み付き線形回帰によってステップ数から両方の漸近線を直接抽出できる
  - **切片：**フルバッチでの最小ステップ数の下限 $S_{\min}$
  - **傾き / 切片：**実験的な臨界バッチサイズ $b_{\text{crit}}$

<!--en:af54f005-->
- 当てはめた値（`TinyLM` で $b_{\text{crit}} \sim 80\text{--}110$ 系列、`FashionCNN` で $b_{\text{crit}} \sim 200\text{--}350$ 枚）は、2 バッチ勾配ノイズ推定量 $b_{\text{noise}}$ とオーダーが一致する

<!--en:8f8f847c-->
::: {.slide title="普遍的な正規化パレート曲線への重なり"}

<!--en:0bdf4442-->
[ステップ数対データ例数のフロンティア]{.kicker}

<!--en:6714c7de-->
- $b / b_{\text{crit}}$ を $\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b}$ と $\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$（ただし $E_{\min} = b_{\text{crit}}S_{\min}$）の間から消去すると、アーキテクチャに依存しない双曲線が得られる
  $$\left(\frac{E}{E_{\min}} - 1\right)\left(\frac{S}{S_{\min}} - 1\right) = 1$$

<!--en:242f807a-->
- `TinyLM` と `FashionCNN` をそれぞれ当てはめた $(S_{\min}, E_{\min})$ で再スケーリングすると、両モデルがこの単一の曲線に重なる
  - **左上の枝（$b \ll b_{\text{crit}}$）：**データ効率が高い（$E \approx E_{\min}$）が、多数の逐次ステップを要する
  - **屈曲点（$b = b_{\text{crit}}$）：**$(2E_{\min}, 2S_{\min})$
  - **右下の枝（$b \gg b_{\text{crit}}$）：**ステップ数が飽和し（$S \approx S_{\min}$）、データコストが急増する

<!--en:2f9104b8-->
::: {.slide title="SGD における線形スケーリング則（η ∝ b）の検証"}

<!--en:e4d53194-->
[FashionCNN 上でのモメンタム付き SGD]{.kicker}

<!--en:dd7a785c-->
- 固定データ予算 $E = 12{,}800$ 枚（$1/5$ エポック）のもと、モメンタム付き SGD（$\beta = 0.9$）を公比 $2$ の学習率グリッド上で $b = 8$（$1{,}600$ ステップ）と $b = 64$（$200$ ステップ）についてスイープ
  - バッチサイズを **$8\times$（$8 \to 64$）**に増やすと、U 字型の損失曲線とその最適学習率が右へ **$4\text{--}8\times$**（$0.0125 \to 0.05\text{--}0.10$）シフトする
  - $\eta$ を $0.0125$ のままスケーリングせずに $b = 64$ へ移行すると、$200$ ステップの実行ではステップ幅が大幅に不足する

<!--en:4739626c-->
::: {.slide title="Adam における平方根則（η ∝ √b）の検証"}

<!--en:83198386-->
[バッチサイズ 64× 増加に対する FashionCNN 上での Adam]{.kicker}

<!--en:362b9bed-->
- 同じ $12{,}800$ 枚の予算のもと、Adam でバッチサイズを **$64\times$（$b = 2 \to 128$）**に増やし、$13$ 点・公比 $2$ のグリッド上で探索
  - **スケーリングなし（$1\times$）：**$\eta$ が $4\text{--}8\times$ 小さすぎ（$b = 128$ において）、損失が約 $0.1$ nats 悪化する
  - **平方根則（$\sqrt{64} = 8\times = 2^3$、グリッドで $+3$ 段階）：**実験的な最適値（$\eta = 0.0064$）またはその 1 グリッド段階以内に到達する
  - **線形則（$64\times = 2^6$、グリッドで $+6$ 段階）：**不安定な右側の壁まで行き過ぎ、損失は 3 手法中で最悪となる。この学習率または 1 段階上の値では損失が $\log 10 \approx 2.3$ に張り付き、ネットワークは何も学習しない。

<!--en:c8d6d134-->
::: {.slide title="動的なバッチサイズランプと実践的な選択レシピ"}

<!--en:f3151170-->
[エンジニアリング実践指針]{.kicker}

<!--en:edf89931-->
- **なぜ訓練中に $b$ を段階的に増やすのか：**$b_{\text{noise}}$ は $10\text{--}100\times$ も増大し得るため（ステップ 0 から訓練中盤にかけて）、最先端の実行では $b$ を動的に引き上げる（例：Llama 3 では $4\text{M} \to 16\text{M}$ トークン）

<!--en:0ab6a809-->
- **4 段階の選択プロトコル：**
  1. GPU のテンソルコアが飽和するよう $b \ge b_{\text{sat}}$ に設定する
  2. ウォームアップ後の $b_{\text{noise}}$ をステップ $300\text{--}500$ で測定する
  3. $b \lesssim 0.3\,b_{\text{crit}}$（計算律速）または $b \approx b_{\text{crit}}$（実経過時間律速）を選ぶ
  4. 線形ウォームアップを併用し、$\eta \propto b$（SGD）または $\eta \propto \sqrt{b}$（AdamW）でスケーリングする

<!--en:7427cffb-->
::: {.slide title="理解度チェック：2 バッチ推定量とスケーリング則"}

<!--en:929b7671-->
[理解度の確認]{.kicker}

<!--en:c6382487-->
次の各バッチサイズのシナリオについて、定量的な予測を求めよ。

<!--en:644e8189-->
1. **2 バッチ推定量：**ステップ $500$ において、二乗勾配ノルムの平均を $b_{\text{s}} = 10$ と $b_{\text{l}} = 100$ で測定したところ、$N_{10} = 1.0$ と $N_{100} = 0.19$ が得られた。$\operatorname{tr}\boldsymbol{\Sigma}$、$\|\nabla f\|^2$、$b_{\text{noise}}$ はそれぞれいくらか。

<!--en:ce4492b4-->
2. **$b_{\text{crit}}$ 付近での GPU の追加：**`TinyLM` を AdamW で $b = 64$、$\eta = 3 \times 10^{-3}$ のもと訓練しており、$b_{\text{crit}} = 100$ であるとする。2 基の GPU を用いてバッチサイズを $b = 128$ に 2 倍化したとき、平方根則が推奨する学習率はいくらか。また、ステップ数 $S(b)$ は何倍の比率で減少し、総データ量 $E(b)$ は何倍に増加するか。

<!--en:3e4d835d-->
3. **大きな $b$ でウォームアップが必須となる理由：**訓練中盤で $b_{\text{crit}} = 256$ であるとしても、ウォームアップなしで $\sqrt{256/16} = 4\times$ に引き上げた学習率をステップ $t = 1$ から適用できないのはなぜか。

<!--en:557b98b6-->
**解答：**

<!--en:1fcd5d43-->
1. $\operatorname{tr}\boldsymbol{\Sigma} = \frac{1.0 - 0.19}{1/10 - 1/100} = \frac{0.81}{0.09} =$ **$9.0$**、$\|\nabla f\|^2 = 0.19 - \frac{9.0}{100} =$ **$0.10$**、**$b_{\text{noise}} = \frac{9.0}{0.10} = 90$**。

<!--en:1f81038b-->
2. 平方根スケーリングにより **$\eta = \sqrt{2} \times 3 \times 10^{-3} \approx 4.24 \times 10^{-3}$** となる。ステップ数は $\frac{1 + 100/64}{1 + 100/128} = \frac{2.5625}{1.78125} \approx$ **$1.44\times$** 減少し（2 基の GPU での実経過時間は $69\%$ に短縮）、処理される総データ量は $\frac{1 + 128/100}{1 + 64/100} = \frac{2.28}{1.64} \approx$ **$1.39\times$（データ量 $+39\%$）**に増加する。

<!--en:b41602a6-->
3. ステップ $t = 0$ では $b_{\text{noise}}(0) \approx 2 \ll 256$ であるため、$b = 256$ は初期段階では曲率支配領域の奥深く（$b \gg b_{\text{noise}}(0)$）にあり、$4\times$ に引き上げた学習率は初期の安定性の上限を超えてしまう。

<!--en:c660a564-->
::: {.slide title="まとめ：勾配ノイズスケールと臨界バッチサイズ"}

<!--en:c5a222f8-->
[要点のまとめ]{.kicker}

<!--en:70ad3065-->
| 量・法則 | 数式 | 領域・スケーリング挙動 | 実務上の意味 |

<!--en:0802a335-->
| **ミニバッチ二乗ノルム** | $\mathbb{E}\|\hat{\mathbf{g}}_b\|^2 = \|\nabla f\|^2 + \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$ | 信号パワー $+\; 1/b$ ノイズパワー | $\operatorname{tr}\boldsymbol{\Sigma}$ と $\|\nabla f\|^2$ を 2 つのバッチサイズ $b_{\text{s}} < b_{\text{l}}$ から推定可能 |

<!--en:69e5acbb-->
| **勾配ノイズスケール** | $b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$ | 訓練中に $10\text{--}100\times$ 増大し得る（$\|\nabla f\|^2$ の低下に伴う） | 臨界バッチサイズ $b_{\text{crit}}$ のヘッセ行列不要な予測指標 |

<!--en:0110c915-->
| **ステップ数・データ量のパレート曲線** | $\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b},\; \frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$ | 普遍双曲線 $(E/E_{\min}-1)(S/S_{\min}-1) = 1$ | $b = b_{\text{crit}}$ で $S = 2S_{\min}$ かつ $E = 2E_{\min}$、$b_{\text{crit}}$ 超では収穫逓減 |

<!--en:ed60f0a8-->
| **SGD の学習率スケーリング** | $\eta(b) = \frac{b}{b_0}\,\eta_0$（$b \ll b_{\text{crit}}$） | $b_{\text{crit}}$ まで線形に増加し、その後 $1/L$ で飽和 | $k$ 回の小ステップと $1$ 回の大きなステップを一定のノイズフロア $\eta/b$ で一致させる |

<!--en:9dab49bd-->
| **AdamW の学習率スケーリング** | $\eta(b) = \sqrt{\frac{b}{b_0}}\,\eta_0$（$b \ll b_{\text{crit}}$） | $\sqrt{\hat{v}_t} \approx \sigma/\sqrt{b}$ が $\sqrt{b}$ を補うため平方根で増加 | 小さな初期値 $b_{\text{noise}}(0)$ に対処するため線形ウォームアップ（またはバッチサイズランプ）と併用 |
