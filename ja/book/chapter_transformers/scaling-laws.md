<!--en:b7e6a72e-->
# スケーリング則

<!--en:c4c0d6c2-->
Transformer 言語モデル（:numref:`sec_gpt`）を事前学習（pretraining）する際、計算コストを決定する巨視的な量が 2 つある。**非埋め込みパラメータ $N$**（non-embedding parameters、$L$ 個の Transformer ブロック内のパラメータ）の個数と、オプティマイザに流し込まれる**訓練トークン数 $D$**（training tokens）である。各トークンはパラメータあたり予測可能な回数の浮動小数点演算（floating-point operations、FLOPs）を必要とするため、訓練の計算予算 $C$（単位は FLOPs）を固定すると、モデルサイズ $N$ とデータ予算 $D$ の間に直接的なトレードオフが生じる。より大きなモデルをより少ないステップ数で訓練すべきか、それともより小さなモデルをより多くのトークンで訓練すべきか。

<!--en:71ccda42-->
経験的な**ニューラルスケーリング則**（neural scaling laws）の研究 :cite:`kaplan2020scaling` により、検証交差エントロピー損失は滑らかなべき乗則（power law）の曲面 $\mathcal{L}(N, D)$ に従い、それが $N$ と $D$ の何桁にもわたる範囲で成り立つことが明らかになった。続いて :citet:`hoffmann2022training` は、固定された事前学習計算量 $C \approx 6 N D$ のもとで損失を最小化するには、モデルパラメータ $N^*(C)$ と訓練トークン数 $D^*(C)$ を**同じ比率**（$N^* \propto C^{1/2}, D^* \propto C^{1/2}$、パラメータあたり約 $D^* / N^* \approx 20$ トークン、:numref:`fig_mdl-transformers-chinchilla`）でスケーリングする必要があることを示した。本節では、パラメータ数の集計式 $N \approx 12 L d^2$ と FLOP 則 $C \approx 6 N D$ を導出して厳密な演算子カウンタに対して両者を検証し、15 モデルのスケーリンググリッド（5 種類のサイズと 3 種類のトークン予算の組み合わせ）を WikiText-103 上で実行したうえで、**推論コスト**（inference cost）を考慮すると現代のオープンモデルが Chinchilla の比率をはるかに超えてより小さなモデルを訓練する方向へシフトする理由を説明する。

<!--en:b9a7e99d-->
## パラメータ数（$N \approx 12 L d^2$）と FLOPs（$C \approx 6 N D$）の算出

<!--en:73aba09e-->
### ブロックと埋め込み

<!--en:3160c592-->
密な $L$ ブロックの `GPT` において、隠れ層の幅を $d$ とし、標準的なマルチヘッドアテンション（$H_{\textrm{kv}} = H$）および `SwiGLU` の幅 $d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$（:numref:`sec_transformer-block`）を用いる場合を考える。

<!--en:d6cebebe-->
- 各アテンションサブ層は 4 つの $d \times d$ 射影行列（$\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$）を持ち、パラメータ数は **$4 d^2$** 個である。

<!--en:fdce0f52-->
- 各 `SwiGLU` FFN は 3 つの行列（$\mathbf{W}_g, \mathbf{W}_1 \in \mathbb{R}^{d \times \frac{8}{3}d}$ および $\mathbf{W}_2 \in \mathbb{R}^{\frac{8}{3}d \times d}$）を持ち、パラメータ数は **$8 d^2$** 個である。

<!--en:6a6fc888-->
- したがって、各ブロックは **$12 d^2$** 個の行列パラメータ（および $2d$ 個の `RMSNorm` ゲイン）を持ち、$L$ 個のブロック全体では次式の個数の非埋め込み行列パラメータを持つ。

  $$N \;=\; 12 L d^2$$

  一方、重み共有されたトークン埋め込みテーブルが $V d$ 個のパラメータを追加する。

<!--en:8a9d89b6-->
非埋め込みパラメータ数が $12 L d^2$ に等しいこと（無視できる $O(Ld)$ 個の正規化のゲインおよびバイアスの範囲内で）を、6 ブロックのバイトレベル `GPT` と 12 ブロックの GPT-2 Small の両方で確認する。

<!--en:da12482e-->
### パラメータ・トークンあたり 6 FLOPs（$C \approx 6 N D$）

<!--en:ab1ec4af-->
なぜ 1 つの訓練トークンは、約 $6 N$ FLOPs を $N = 12 L d^2$ 個の線形射影（linear projection）パラメータ全体で消費するのだろうか。

<!--en:b42a6bac-->
1. **順伝播（トークンあたり $2 N$ FLOPs）：**トークンベクトル $\mathbf{x} \in \mathbb{R}^{d_{\textrm{in}}}$ に重み行列 $\mathbf{W} \in \mathbb{R}^{d_{\textrm{in}} \times d_{\textrm{out}}}$ を掛けると、$d_{\textrm{in}} d_{\textrm{out}}$ 回の積和演算 $= 2 P$ FLOPs が実行される（ここで $P = d_{\textrm{in}} d_{\textrm{out}}$ はこの行列のパラメータ数である）。全 $N$ 個の行列パラメータと $D$ トークンにわたって合計すると、順伝播の計算量は $2 N D$ FLOPs になる。

<!--en:3811d602-->
2. **逆伝播（トークンあたり $4 N$ FLOPs）：**$\mathbf{Y} = \mathbf{X}\mathbf{W}$ を通して誤差逆伝播法を実行するには、トークンあたり *2* 回のサイズ $P$ の行列積が必要になる。すなわち $\nabla_{\mathbf{X}}\mathcal{L} = (\nabla_{\mathbf{Y}}\mathcal{L})\mathbf{W}^\top$（$2P$ FLOPs）と $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top(\nabla_{\mathbf{Y}}\mathcal{L})$（$2P$ FLOPs）であり、合計で逆伝播の計算量は $4 N D$ FLOPs となる。

<!--en:6753554b-->
順伝播と逆伝播を足し合わせると、訓練計算量の基本的な近似式が得られる :cite:`kaplan2020scaling`。

<!--en:c4d8ba6a-->
（$Q K^\top$ と $A V$ のアテンションスコアのドット積をコンテキストウィンドウ長 $T$ にわたって含めると、$12 L T d D$ FLOPs が加わり、$6 N D$ は $1 + \frac{T}{6d}$ 倍になる。この係数は $1$ に近い値となる（$T \ll 6d$ が成り立つ限り）。因果アテンション行列のマスクされた半分をスキップする実装では、この補正項は半分になる。）

<!--en:e35a756b-->
### ハードウェアカウンタによる計算式の検証

<!--en:b0e9cd2e-->
:eqref:`eq_six_nd` は、フレームワークが計測する厳密な FLOP カウントと直接照らし合わせることで、`gpt(256, act='gelu')` の単一の訓練ステップ上で検証できる。

<!--en:be97e161-->
PyTorch の `FlopCounterMode` は、すべての演算子の FLOPs をカウントする。

<!--en:2f0d4996-->
XLA のコスト解析は、コンパイルされたプログラムの FLOPs を CPU 上でカウントする。

<!--en:3a148e0d-->
実測された逆伝播の計算量は順伝播の約 2 倍（$2.0\text{--}2.1\times$）であり、$6 N D$（重み共有出力ヘッドおよびアテンションスコア項 $12 L T d D$ を考慮したもの）は厳密な演算子カウントと数パーセント以内で一致する。

<!--en:84b18ab8-->
## WikiText-103 における小規模スケーリング実験

<!--en:be99b4fa-->
### 5 つのモデルサイズと 3 つのデータ予算

<!--en:1a4381eb-->
スケーリングの挙動を直接観察するため、**15 個のモデル**からなるグリッドを WikiText-103 上で訓練する。具体的には、非埋め込みパラメータ数が $N \in [0.3\textrm{M}, 14\textrm{M}]$ にわたる 5 種類のアーキテクチャ（`(d, L, H)` が `(96, 3, 4)` から `(384, 8, 8)` まで）と、3 種類のステップ予算 $s \in \{500, 1000, 2000\}$ ステップ（訓練バイト数 $D \in \{4.10\textrm{M}, 8.19\textrm{M}, 16.38\textrm{M}\}$ に相当）を掛け合わせる。

<!--en:6f3071c2-->
コーパスを読み込んだら、全 $5 \times 3 = 15$ 通りの `(size, budget)` 設定を、幅に応じてスケーリングしたピーク学習率 $\eta = 2 \times 10^{-3} \cdot (256 / d)$ で訓練し、各モデルの非埋め込みパラメータ数 `Ns` と検証交差エントロピー `val` を記録する。

<!--en:99d9b157-->
### モデルサイズ、データ予算、および計算量に対する損失

<!--en:544bdad0-->
検証交差エントロピーを、非埋め込みパラメータ数 $N$ の対数軸に対してトークン予算 $D$ ごとにプロットし、さらに総訓練計算量 $C = 6 N D$ に対してもプロットする。

<!--en:32fec6eb-->
同じ 15 個の検証損失を総訓練計算量 $C = 6 N D$ に対して再プロットすると（あわせて計算量が近い 3 つの実行、すなわち $C \approx 2\text{--}3 \times 10^{14}\textrm{ FLOPs}$ で互いに $2\times$ 以内に収まるものを出力すると）、FLOP 予算が増えるにつれて計算量最適（compute-optimal）なモデルサイズが上方へシフトする様子がわかる。

<!--en:9c2394cd-->
2 つのパターンが際立っている。

<!--en:800641f1-->
1. 最大のトークン予算（$2{,}000$ ステップ、$D = 16.4\textrm{M}$ バイト）では、検証損失は小型・中型モデルにわたって $\log N$ に対してほぼ線形に減少したのち、最大のモデル（$N = 14.16\textrm{M}$）で横ばいに曲がる。そこでは $2{,}000$ ステップ（パラメータあたり $D/N \approx 1.2$ バイト）であっても、$14\textrm{M}$ 個のパラメータに対してはいまだデータ不足の状態にある。

<!--en:0d8dc7f6-->
2. 全 15 回の実行を総訓練 FLOPs $C = 6 N D$ に対してプロットすると、その下側包絡線（lower envelope）が**パレート最適な計算フロンティア**（Pareto-optimal compute frontier）を描く。計算量が小さい領域では小型モデルが優位に立ち、計算量が大きい領域では十分なステップ数が与えられた時点で大型モデルが逆転する。

<!--en:0caaf108-->
## 計算量最適な配分（Chinchilla スケーリング）

<!--en:fa3d0698-->
### Chinchilla のパラメトリック法則

<!--en:dcf8a1fa-->
:citet:`hoffmann2022training`（および :citet:`Besiroglu.Erdil.Barnett.ea.2024` による再フィット）は、検証損失曲面 $\mathcal{L}(N, D)$ を、削減不能なテキストエントロピーの下限 $L_\infty$ と、$N$ および $D$ に関する 2 つのべき乗則ペナルティの和としてモデル化する。

<!--en:0bd0635c-->
### 計算量最適な分割（$N^*(C)$ と $D^*(C)$）

<!--en:e67a1825-->
:eqref:`eq_chinchilla_law` を、$N$ と $D$ に関して制約 $C = 6 N D$ のもとで最小化すると（$D = C / (6N)$ を代入して $\partial \mathcal{L} / \partial N = 0$ とおくと）、最適なモデルサイズ $N^*(C)$ と最適なトークン予算 $D^*(C)$ の両方についてべき乗則スケーリングが得られる。

<!--en:b48111c1-->
ここで、各定数は次のとおりである。

<!--en:3c95af28-->
![後述する Besiroglu ら（2024）の再フィット定数を用いた、パラメータ数と訓練トークン数にわたる Chinchilla 則の損失等高線。破線：等しい計算量 $C = 6ND$ の予算線、点：各予算における計算量最適モデル、オレンジ色の実線：計算量最適経路。Chinchilla を示す四角は自身の予算の最適点上に位置し、その点を覆っている。一方、Chinchilla に近い予算で訓練された Llama 3 8B は、最適サイズの 10 分の 1 の位置にあり、最適経路から大きく外れている。](../img/mdl-transformers-chinchilla.svg)

<!--en:fa5e0893-->
$\alpha \approx \beta \approx 0.34\text{--}0.37$ であるため、指数は $a \approx b \approx 0.5$ を満たす。すなわち、訓練計算量 $C$ が $100\times$ に増えると、モデルサイズ $N^*$ と訓練トークン数 $D^*$ はともに $\sim 10\times$ に増え、パラメータあたり約 $D^* / N^* \approx 20$ トークンという計算量最適な比率はほぼ一定に保たれる。

<!--en:c3d85ec4-->
これは、この問いに対する最初の答えではなかった。その 2 年前、:citet:`kaplan2020scaling` は同種のべき乗則を当てはめ、追加の計算量の大部分はモデルに投じるべきだと結論づけていた。すなわち $N^* \propto C^{0.73}$ であり、データには $D^* \propto C^{0.27}$ しか残らない（:numref:`fig_mdl-transformers-kaplan-chinchilla`）。この 2 つの研究の食い違いは、主に学習率スケジュール（learning rate schedule）に起因する。Kaplan らはすべての実行に 1 つのスケジュールを使い、より小さなトークン予算の損失を、まだ学習率が減衰していない中間チェックポイントから読み取っていた。これでは少ないトークンでの訓練が実際より悪く見えるため、データの効果が実際より小さく見積もられてしまう。また、扱ったモデルも小さく、高々 $1.5\textrm{B}$ パラメータであった。Hoffmann らは最大 $16\textrm{B}$ パラメータのモデルを訓練し、コサインスケジュールを各実行の長さに合わせた。前述のグリッド実験でもこれと同じ方法をとっており、15 回の実行のそれぞれが固有のコサインホライズンを持つのはそのためである（そうしない場合に何が起きるかは、演習問題 :exref:`ex_scaling-laws-frozen-learning-rate` で計測する）。

<!--en:6bd3bd5e-->
![公表された 2 つの指数、すなわち $0.73$（Kaplan ら）と $0.5$（Hoffmann ら）のもとでの訓練計算量に対する計算量最適モデルサイズ。いずれの直線も $10^{18}$ FLOPs において同じモデルを通るように引かれており、傾きのみが各論文に由来する。予算を $100\times$ に増やすと、第 1 の指数では $29\times$ 大きいモデルが、第 2 の指数では $10\times$ 大きいモデルが求められる。四角は Chinchilla を示す。](../img/mdl-transformers-kaplan-chinchilla.svg)

<!--en:d1633d65-->
以下の関数 `optimum` は、:eqref:`eq_chinchilla_optimum` を Chinchilla の訓練予算 $C = 5.76 \times 10^{23}$ FLOPs において、Hoffmann らが公表した定数と Besiroglu らによる再フィット定数のそれぞれについて評価する。

<!--en:5168d889-->
### 推論を見据えた Chinchilla 最適を超える訓練

<!--en:e7c562b1-->
なぜ Llama 3 8B（:numref:`fig_mdl-transformers-chinchilla`）は、8B パラメータのモデルを $15\textrm{T}$ トークン（パラメータあたり $D / N \approx 1{,}875$ トークンで、Chinchilla の比率のほぼ $100\times$）で訓練する道を選び、Chinchilla 最適な $70\textrm{B}$ モデルを $1.4\textrm{T}$ トークン（同じ事前学習計算量 $C \approx 7 \times 10^{23}$ FLOPs）で訓練しなかったのだろうか。

<!--en:1fcebce4-->
それは、Chinchilla が**事前学習の FLOPs $6 N D$ のみ**を最小化し、デプロイ後のモデルのサービングに費やされる $2 N D_{\textrm{inf}}$ の推論 FLOPs や KV キャッシュメモリを無視しているからである :cite:`Sardana.Portes.Doubov.ea.2024`。一定の損失をもつ等高線 $\mathcal{L}(N, D) = \mathcal{L}_{\textrm{target}}$ に沿って移動してみると、8B モデルを $15\textrm{T}$ トークンで過剰訓練（over-training）すれば、Chinchilla 配分の $30\textrm{B}\text{--}40\textrm{B}$ モデルに近い損失に到達しつつ、その後に生成するすべての推論トークンを $4\text{--}5\times$ 安くサービングできることがわかる。

<!--en:30ac03ff-->
## まとめ

<!--en:ac4cdbb6-->
$L$ 層、幅 $d$ で SwiGLU（$d_{\textrm{ff}} \approx \frac{8}{3}d$）を備えた密な Transformer は、**$N \approx 12 L d^2$** 個の非埋め込み行列パラメータ（アテンションに $4 L d^2$、FFN に $8 L d^2$）を含み、**$C \approx 6 N D$** の訓練 FLOPs を $D$ トークンにわたって消費する（順伝播で $2 N D$、逆伝播で $4 N D$、:eqref:`eq_six_nd`）。**Chinchilla スケーリング則** $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta}$（:eqref:`eq_chinchilla_law`）のもとでは、固定された事前学習計算量 $C = 6 N D$ で検証損失を最小化すると、パラメータ数とトークン数は指数 $a = \beta/(\alpha+\beta) \approx 0.5$ および $b = \alpha/(\alpha+\beta) \approx 0.5$（:eqref:`eq_chinchilla_optimum`、:eqref:`eq_chinchilla_allocation`）に従ってスケーリングし、パラメータあたり約 $D^* / N^* \approx 20$ トークンという比率が得られる。**生涯推論計算量**（lifetime inference compute、$2 N D_{\textrm{inf}}$）とサービング時のレイテンシを考慮すると、総計算量の最適点は等損失等高線に沿って、パラメータあたり数百〜数千トークンで過剰訓練されたより小さなモデル $N$ の側へとシフトする。

<!--en:2294243a-->
## 演習問題

<!--en:517384ad-->
1. **アテンションスコアの FLOP 補正（$1 + T / (6d)$）。**本文の :eqref:`eq_six_nd` は、アテンションスコアと値の積 $\mathbf{Q}\mathbf{K}^\top$ および $\mathbf{A}\mathbf{V}$、ならびに出力ヘッド $\mathbf{H}^{(L)}\mathbf{E}^\top$ を省略している。
   1. $L$ ブロック、幅 $d$、ウィンドウ長 $T$ のモデルにおいて、順伝播のアテンションのドット積がウィンドウあたり $4 L T^2 d$ FLOPs（トークンあたり $4 L T d$ FLOPs）を実行し、順伝播と逆伝播を合わせたアテンションのドット積がトークンあたり $12 L T d$ FLOPs（因果対称性が FlashAttention によって利用される場合は $6 L T d$）を実行することを示せ。
   2. $6 L T d$ と $6 N = 72 L d^2$ の比をとることで、因果アテンションが加える相対的な FLOP 比率が $T / (12 d)$（因果タイルスキップなしでは $T / (6 d)$）となることを示せ。コンテキスト長 $T^*$ がいくつのときに、アテンションの計算量は Llama 3 8B（$d = 4096$）の線形射影の FLOPs と一致するか。

<!--en:f84661f4-->
2. **スケーリンググリッドにおける総パラメータと非埋め込みパラメータ。**本節の 5 サイズにわたる WikiText-103 スイープの各モデルではバイト語彙（$V = 256$）を用いたため、$V d$ は高々約 $7\%$（対 $12 L d^2$ 比）にとどまっていた。`census` を同じ 5 つの `(d, L, H)` バックボーンに対して、代わりに BPE 語彙 $V = 128{,}256$ トークン（Llama 3）を用いた場合について実行せよ。最小モデル（`d=96, L=3`）と最大モデル（`d=384, L=8`）では、埋め込みテーブルが総パラメータのどれだけの割合を占めるか。また、なぜ :citet:`kaplan2020scaling` はべき乗則を $N_{\textrm{non-emb}} = 12 L d^2$ に対して当てはめ、$N_{\textrm{total}}$ に対しては当てはめなかったのか。
   :exlabel:`ex_scaling-laws-embedding-count`

<!--en:8e91ef28-->
3. **コサイン減衰のホライズンと固定ステップホライズン。**本節の 15 実行のスケーリング実験では、各予算 $s \in \{500, 1000, 2000\}$ が固有のコサインスケジュールを実行し、学習率をステップ $s$ でゼロまで減衰させた（スケジュールを実行の長さに合わせることは、:citet:`hoffmann2022training` による :citet:`kaplan2020scaling` への決定的な修正であった）。3 つの予算を中型モデル `(192, 4, 6)` について再実行せよ。ただし、単一の固定コサインホライズン `num_steps=2000` を `LMOptimizer` 内で用い、ステップ 500、1000、2000 における検証損失を単に読み取るものとする。$s = 500$ および $s = 1000$ で学習率を減衰させないことによって、短い実行の損失はどれだけ過大評価されるか。
   :exlabel:`ex_scaling-laws-frozen-learning-rate`

<!--en:a47d04bb-->
4. **Chinchilla 則の 15 点グリッドへの当てはめ。**`scipy.optimize.minimize`（またはグリッド探索 + 非線形最小二乗法）を用いて、パラメータ $(L_\infty, A, B, \alpha, \beta)$（:eqref:`eq_chinchilla_law`）を、15 個の `(N, D, val_loss)` 点（本節の WikiText-103 スイープで計測したもの）に当てはめよ。スケーリング指数 $a = \beta/(\alpha+\beta)$ と $b = \alpha/(\alpha+\beta)$ は、このバイトレベルの 15 点グリッドからどのような値として得られるか。
   :exlabel:`ex_scaling-laws-fit-grid`

<!--en:a20fe21b-->
5. **推論需要 $D_{\textrm{inf}}$ を伴う生涯計算量の最適化。**サイズ $N$ のモデルを $D$ トークンで訓練し、$D_{\textrm{inf}}$ 個の推論トークンをサービングするためにデプロイすると、消費される総生涯 FLOPs は $C_{\textrm{tot}}(N, D) = 6 N D + 2 N D_{\textrm{inf}}$ となる。
   1. 制約 $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta} = \mathcal{L}_{\textrm{target}}$ に対してラグランジュ乗数法を用い、生涯計算量の最小点 $(N^*, D^*)$ において限界比が次の関係を満たすことを示せ。

      $$\frac{\beta B (D^*)^{-\beta}}{\alpha A (N^*)^{-\alpha}} \;=\; \frac{6 N^* D^*}{6 N^* D^* + 2 N^* D_{\textrm{inf}}} \;=\; \frac{1}{1 + \frac{D_{\textrm{inf}}}{3 D^*}}.$$

   2. $D_{\textrm{inf}} = 0$ のときにこれが Chinchilla の均衡式 $\alpha A (N^*)^{-\alpha} = \beta B (D^*)^{-\beta}$ に帰着することを確かめよ。さらに $(N^*, D^*)$ を、$\mathcal{L}_{\textrm{target}} = \mathcal{L}(70\textrm{B}, 1.4\textrm{T})$ かつ $D_{\textrm{inf}} \in \{10\textrm{T}, 100\textrm{T}, 500\textrm{T}\}$ トークンの各場合について数値的に求めよ。
   :exlabel:`ex_scaling-laws-training-for-inference`

<!--en:ad025382-->
6. **活性化チェックポイントと $8 N D$ 則。**完全なブロック単位の活性化チェックポイントは、各ブロックへの入力 $\mathbf{X}^{(\ell-1)}$ のみを順伝播（$2 N D$ FLOPs）の間に保存し、誤差逆伝播法において逆方向の行列積（$4 N D$ FLOPs）を計算する前に各ブロックの順伝播を一度再計算する。活性化チェックポイントによって事前学習計算量の式が $C \approx 6 N D$ から $C \approx 8 N D$ に変わることを示せ。また、$6$ を $8$ に置き換えること（:eqref:`eq_chinchilla_optimum`）によって、$N^*(C)$ と $D^*(C)$ が固定された FLOP 予算 $C$ のもとでどのようにスケーリングするかを求めよ。

<!--en:128c43b6-->
[ディスカッション](https://d2l.discourse.group/t/36780)

<!--en:e771fb7a-->
ニューラルスケーリング則と計算量最適な訓練（Chinchilla）<br>**N ≈ 12Ld² と C ≈ 6ND・15 モデルのスケーリンググリッド・IsoFLOP 等高線・Chinchilla 最適・推論を見据えた過剰訓練**

<!--en:01b31777-->
::: {.slide title="なぜ 10²⁵ FLOP の実行は開始前の計画が不可欠か"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:57bce442-->
- 最先端の言語モデルの事前学習は、数千基の GPU を数か月稼働させるやり直しのきかない単発の設備投資（Llama 3 405B では $3.8 \times 10^{25}\text{ FLOPs}$）

<!--en:c7f196dd-->
- **根本的な配分の問い：**固定された訓練の計算予算 $C \approx 6 N D$ が与えられたとき、$C$ をモデルパラメータ $N$ と訓練トークン数 $D$ にどう分割すれば検証損失 $\mathcal{L}(N, D)$ を最小化できるか

<!--en:6e893d87-->
::: {.slide title="パラメータ数の集計：ブロック（12Ld²）と埋め込み（Vd）"}

<!--en:90b25bbf-->
[パラメータ数の内訳]{.kicker}

<!--en:4fe3c052-->
$L$ ブロック・幅 $d$ の密な Transformer（$H_{\text{kv}} = H$、`SwiGLU` の幅 $d_{\text{ff}} \approx \frac{8}{3}d$）では、各ブロックが $4d^2$ 個のアテンション重みと $8d^2$ 個の FFN 重みを持つ。

<!--en:c85e2d1a-->
- **小規模な語彙（$V = 256$ バイト）：**埋め込みテーブルは $1.4\%$ にすぎない（$4.79\text{M}$ パラメータのモデル、$N_{\text{blocks}} = 12.01\,L d^2$）

<!--en:d5514c9c-->
- **大規模な BPE 語彙（$V = 50{,}257$、GPT-2 Small）：**埋め込み・位置テーブルが $39.4\text{M}$（$31.6\%$）を占め（総数 $124.4\text{M}$ パラメータ）、12 個のブロックが $85.0\text{M}$（$12.02\,L d^2$）を保持する

<!--en:fd08bfb3-->
::: {.slide title="パラメータ数集計の実装"}

<!--en:90b25bbf-2-->
[パラメータ数の内訳]{.kicker}

<!--en:445b2c32-->
::: {.slide title="訓練にパラメータ・トークンあたり 6 FLOPs かかる理由"}

<!--en:65fdc80c-->
[C ≈ 6ND の法則]{.kicker}

<!--en:4998f802-->
- 各線形射影 $\mathbf{Y} = \mathbf{X}\mathbf{W}$（パラメータ数 $P = d_{\text{in}} d_{\text{out}}$）を $D$ トークンにわたり計算すると、全体の計算量は次のようになる。
  $$C \approx \underbrace{2 N D}_{\text{forward}} + \underbrace{4 N D}_{\text{backward}} = 6 N D$$

<!--en:3f5f8bbe-->
- **順伝播（トークンあたり $2P$ FLOPs）：**$\mathbf{Y} = \mathbf{X}\mathbf{W}$

<!--en:d3bf5a09-->
- **逆伝播（トークンあたり $4P$ FLOPs）：**活性化値の勾配 $\nabla_{\mathbf{X}}\mathcal{L} = (\nabla_{\mathbf{Y}}\mathcal{L})\mathbf{W}^\top$（$2P$）+ 重みの勾配 $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top(\nabla_{\mathbf{Y}}\mathcal{L})$（$2P$）

<!--en:80ec9019-->
::: {.slide title="アテンションスコアのドット積の考慮"}

<!--en:547e34fe-->
[アテンションの FLOP 項]{.kicker}

<!--en:b0e8eced-->
- 各ブロック内のアテンションでは $\mathbf{Q}\mathbf{K}^\top$ と $\mathbf{A}\mathbf{V}$ も計算され（順伝播でトークンあたり $4Td$ FLOPs、線形射影は $24d^2$ FLOPs）、相対補正項は $T / (6d)$（因果タイルスキップ時は $T / (12d)$）となる

<!--en:8416a8ec-->
- **短い文脈（$T \ll 6d$）：**$\approx +8\%$ にとどまる（バイトレベル GPT、$T = 128, d = 256$）

<!--en:5e7f7910-->
- **長文脈の交差点（$T \ge 6d$）：**$T = 6d = 24{,}576$ トークンで射影の FLOPs に並ぶ（Llama 3 8B、$d = 4{,}096$）

<!--en:04d9fdad-->
::: {.slide title="厳密なハードウェアカウンタによる C ≈ 6ND の検証" only="pytorch"}

<!--en:f795cc80-->
[実測 FLOP の検証]{.kicker}

<!--en:6eebfcbf-->
`gpt(256, act='gelu')` の順伝播 + 逆伝播のステップ全体を、バッチ $(64, 128)$（$D_{\text{step}} = 8{,}192$ トークン）に対して `FlopCounterMode` で計測すると、解析的な導出が裏づけられる。

<!--en:447e8b6d-->
行列積の厳密な FLOP カウントは **$1.082 \times 6 N D$** となり、$1 + \frac{4 L T d}{2 N} = 1.082$ と小数第 3 位まで一致し、逆伝播と順伝播の比は **$\mathbf{2.00}$**（$4 N D$ 対 $2 N D$）となる。

<!--en:1bfbea00-->
::: {.slide title="厳密なハードウェアカウンタによる C ≈ 6ND の検証" only="jax"}

<!--en:f795cc80-2-->
[実測 FLOP の検証]{.kicker}

<!--en:83d14f1a-->
`gpt(256, act='gelu')` の順伝播 + 逆伝播のステップ全体を、バッチ $(64, 128)$（$D_{\text{step}} = 8{,}192$ トークン）に対してコンパイル済みステップの HLO `cost_analysis()` で計測すると、解析的な導出が裏づけられる。

<!--en:518ebe35-->
HLO のコスト解析は **$1.116 \times 6 N D$**、逆伝播と順伝播の比を **$2.06$** と報告しており、行列積による予測値 $1 + \frac{4 L T d}{2 N} = 1.082$ および比率 $2$（$4 N D$ 対 $2 N D$）を数パーセント上回る。

<!--en:25750c16-->
::: {.slide title="15 モデルによる WikiText-103 の実証スケーリング実験"}

<!--en:e531ac0c-->
[スケーリンググリッドの設計]{.kicker}

<!--en:43d4b08f-->
ニューラルスケーリング則を直接観察するため、$5 \times 3$ グリッドの **15 個のバイトレベル GPT**（$43\times$ の範囲にわたる非埋め込みパラメータ $N$ と 3 種類のデータ予算 $D$）を訓練する。

<!--en:7e8a068b-->
| 実験の軸 | 15 実行グリッドでの設定値 | 範囲と目的 |

<!--en:bdfc8c0d-->
| **5 つのバックボーンサイズ `(d, L, H)`** | `(96, 3, 4)`、`(128, 3, 4)`、`(192, 4, 6)`、`(256, 6, 8)`、`(384, 8, 8)` | 非埋め込みパラメータ $N \in [0.33\text{M},\; 14.16\text{M}]$（$43\times$ の範囲） |

<!--en:b44b3ffd-->
| **3 つのステップ予算（$s$ ステップ）** | $500$、$1{,}000$、$2{,}000$ ステップ（$8{,}192$ バイト/ステップ） | $D \in \{4.1\text{M},\; 8.2\text{M},\; 16.4\text{M}\}$ バイト（比率 $\le 6\%$、$270\text{M}$ コーパス中） |

<!--en:0844045d-->
| **幅スケーリング付きピーク学習率** | $\eta_{\max}(d) = 2 \times 10^{-3} \cdot (256 / d)$ | AdamW の最適学習率は幅 $d$ に反比例してスケーリング |

<!--en:d94ec85a-->
| **実行長に合わせたコサインホライズン** | ウォームアップ $\max(100, s/10)$、コサイン減衰で $0$ に到達（ステップ $s$ 時点） | すべての実行を個別に減衰させたスケジュールの終端で評価 |

<!--en:c95a1b47-->
::: {.slide title="落とし穴：固定ホライズンによるスケーリング指数の歪み"}

<!--en:80b628f4-->
[実験手法の落とし穴]{.kicker}

<!--en:2db1feaa-->
- **Hoffmann ら（2022）による Kaplan ら（2020）への修正：**単一の $2{,}000$ ステップのコサインスケジュールでは、ステップ $500$ の学習率はまだピーク付近にある（$\eta_{500} \approx 0.9\,\eta_{\max}$）

<!--en:46964393-->
- **小予算に対する系統的バイアス：**未減衰のスケジュールの中間チェックポイントを読み取ると短い実行の損失が人為的に膨らみ、追加トークン数 $D$ の価値が追加パラメータ $N$ より低く見える

<!--en:9b405c36-->
::: {.slide title="グリッド実験の結果：N 単独拡大における収穫逓減"}

<!--en:3819a9f6-->
[実証スケーリング曲面]{.kicker}

<!--en:96a546c8-->
- **$N$ の増加（トークン数 $D$ 固定）：**モデルサイズを $0.59\text{M} \to 1.77\text{M}$ と 3 倍にすると損失は $0.05\text{--}0.09\text{ nats}$ 下がる一方、$4.72\text{M} \to 14.16\text{M}$ の 3 倍化では $0.02\text{--}0.04\text{ nats}$ の改善にとどまる（$500\text{--}2{,}000$ ステップでは $14.16\text{M}$ モデルがデータ不足になるため）

<!--en:c6ff6572-->
- **$D$ の増加（モデルサイズ $N$ 固定）：**トークン数を 4 倍（$4.1\text{M} \to 16.4\text{M}$）にすると、すべてのモデルサイズで検証損失が **$0.20\text{--}0.27\text{ nats}$** 低下する

<!--en:dc0bfcc0-->
::: {.slide title="IsoFLOP トレードオフ：長く訓練した小型モデルの優位"}

<!--en:cc89a756-->
[パレート計算フロンティア]{.kicker}

<!--en:4a2a1b2c-->
- **全 15 実行の総 FLOPs $C = 6 N D$ に対するプロット：**異なるモデルサイズの曲線が計算量 $C$ で重なる領域では、**より多くのトークンで訓練した小型モデル**が一貫して低い検証損失を達成する

<!--en:e41fb840-->
- **具体的な IsoFLOP 比較：**
  - $N = 14.16\text{M}$、$D = 4.1\text{M}$（$C = 3.5 \times 10^{14}$）：**$1.42\text{ nats}$**
  - $N = 4.72\text{M}$、$D = 8.2\text{M}$（$C = 2.3 \times 10^{14}$）：**$1.34\text{ nats}$**
  - $N = 1.77\text{M}$、$D = 16.4\text{M}$（$C = 1.7 \times 10^{14}$）：**$1.29\text{ nats}$**（*半分*の FLOPs で最小損失）

<!--en:aa7ee6b2-->
::: {.slide title="Chinchilla のパラメトリックスケーリング則"}

<!--en:ad7f80de-->
[パラメトリック損失曲面]{.kicker}

<!--en:87174072-->
- Hoffmann ら（2022）は検証損失 $\mathcal{L}(N, D)$ を 3 つの加法項に分解する。
  $$\mathcal{L}(N, D) = L_\infty + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

<!--en:d00ebf57-->
- **エントロピー下限（$L_\infty \approx 1.7\text{--}1.8$）：**自然言語の削減不能なエントロピー

<!--en:c0b133b6-->
- **モデル容量（$A / N^\alpha$）とデータサンプル（$B / D^\beta$）：**$\alpha \approx 0.35, \beta \approx 0.37$ のべき乗則ペナルティ

<!--en:65d4d0d7-->
::: {.slide title="計算量最適な分割 N*(C) と D*(C) の導出"}

<!--en:96ee8bde-->
[制約付き最適化]{.kicker}

<!--en:6a370cef-->
$\mathcal{L}(N, D)$ を固定された計算予算 $C = 6 N D$ のもとで最小化するため、$D = C / (6N)$ を代入して $\mathcal{L}(N, C/(6N))$ とし、$N$ に関する導関数をゼロとおく。

<!--en:73fcc85d-->
- **限界改善率の均衡（$\alpha A (N^*)^{-\alpha} = \beta B (D^*)^{-\beta}$）：**計算量最適点では、有限モデルのペナルティと有限データのペナルティが比例する速度で縮小する。
  $$N^*(C) = G\left(\frac{C}{6}\right)^a, \qquad D^*(C) = G^{-1}\left(\frac{C}{6}\right)^b$$
  ここで $a = \frac{\beta}{\alpha + \beta}$、$b = \frac{\alpha}{\alpha + \beta}$、$G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}$ である。

<!--en:293c5ae9-->
- **等比率スケーリング（$a \approx b \approx 0.5$）：**$\alpha \approx \beta$ であるため、計算量 $C$ を $100\times$ に拡大すると **$N^*$ と $D^*$ の両方が $\sim 10\times$ に拡大する**

<!--en:837229fa-->
::: {.slide title="IsoFLOP 等高線と計算量最適経路"}

<!--en:9914b045-->
[幾何学的直観]{.kicker}

<!--en:8e4846b4-->
- **IsoFLOP 予算線（破線、両対数軸で傾き $-1$）：**訓練計算量 $C = 6 N D$ が一定の直線

<!--en:8c24a580-->
- **計算量最適の尾根（オレンジ色の線、傾き $\approx +1$）：**各予算線が到達可能な最小の損失等高線に接する接点 $(N^*(C), D^*(C))$ を結ぶ直線

<!--en:a3700fa1-->
- **パラメータあたり 20 トークン則：**オレンジ色の経路に沿うと、パラメータあたり $D^* / N^* \approx 20$ トークン（`Chinchilla` 70B は $1.4\text{T}$ トークンで訓練）

<!--en:bc13df1d-->
::: {.slide title="IsoFLOP 等高線と計算量最適経路（続き）"}

<!--en:9914b045-2-->
[幾何学的直観]{.kicker}

<!--en:6c12c439-->
閉形式の最適解 $N^*(C), D^*(C)$ を、公表された 2 つのフィットについて Chinchilla の予算 $C = 5.76 \times 10^{23}$ FLOPs で評価する。

<!--en:9569ad11-->
::: {.slide title="Kaplan（2020）と Chinchilla（2022）の整合"}

<!--en:13b166b0-->
[指数が変化した理由]{.kicker}

<!--en:f3ded24d-->
- **Kaplan ら（2020）：**$N^* \propto C^{0.73}, D^* \propto C^{0.27}$ を導出（未減衰の学習率スケジュールから中間損失を読み取り、埋め込みパラメータ $V d$ を除外したため）

<!--en:019278e0-->
- **Chinchilla（2022）と追試研究：**コサイン減衰のホライズンを各予算 $s$ に合わせ、総パラメータ数と総 FLOPs を考慮することで $N^* \propto C^{0.50}, D^* \propto C^{0.50}$（$D^* / N^* \approx 20$）を得る

<!--en:d0c703e0-->
::: {.slide title="実用モデルが Chinchilla 比を大きく超えて訓練される理由"}

<!--en:acef1204-->
[推論を考慮したスケーリング]{.kicker}

<!--en:b34a3f48-->
近年のオープンウェイトモデルが Chinchilla の事前学習最適点に対してどこに位置するかを見てみよう。

<!--en:4f730c05-->
- **Chinchilla は事前学習コスト（$6 N D$）のみを最適化：**リリース後のモデルサービングに費やされる推論 FLOPs（生成トークンあたり $2 N$）や VRAM に**コストゼロ**を割り当てている

<!--en:f49b6d65-->
- **最適点付近の緩やかな損失等高線：**`Llama 3 8B` を $15\text{T}$ トークンで訓練すると（$1{,}875\text{ tokens/param}$、同予算での Chinchilla 最適サイズの $0.10\times$）、損失の悪化は $+0.054\text{ nats}$（Chinchilla 最適な $\sim 80\text{B}$ モデル比）にとどまる一方、**すべての推論トークンの生成が $\sim 10\times$ 安価になる**

<!--en:8f110e0c-->
::: {.slide title="生涯計算量：事前学習とサービングのバランス"}

<!--en:ffcf5006-->
[総所有コスト]{.kicker}

<!--en:dc1a06a2-->
- 訓練トークン数 $D$ と生成される推論トークン数 $D_{\text{inf}}$ にわたる総生涯計算量は次の式で表される。
  $$C_{\mathrm{lifetime}}(N, D) = 6 N D + 2 N D_{\mathrm{inf}}$$

<!--en:f82da7cc-->
- **Llama 3 8B（$D = 15\text{T}$）と同損失の 34B Chinchilla モデルの損益分岐点：**先行投資した追加の訓練計算量 $5.9 \times 10^{23}$ FLOPs により生成トークンあたり $5.2 \times 10^{10}\text{ FLOPs}$ が節約され、**推論トークン数 $D_{\text{inf}} \approx 11.3\text{T}$** で元が取れる

<!--en:3fd13d7a-->
::: {.slide title="概念チェック：FLOP の内訳と予算配分"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:025f4585-->
3 つの計算でスケーリング則の理解を確かめよう。

<!--en:684dcdda-->
1. **GPT-2 Small の計算量とアテンション補正：**GPT-2 Small は総パラメータ数 $N = 124\text{M}$（$d = 768$）を持ち、約 $D = 10\text{B}$ トークン、コンテキスト長 $T = 1{,}024$ で訓練された。$6ND$ の訓練 FLOPs とアテンションスコアの相対補正 $T / (6d)$ を見積もれ。

<!--en:26099519-->
2. **100× の計算予算拡大：**ある研究機関が事前学習の計算予算 $C$ を $100\times$（$10^{23}$ から $10^{25}\text{ FLOPs}$ へ）拡大する場合、モデルパラメータ $N^*$ と訓練トークン数 $D^*$ をそれぞれ何倍にスケーリングすれば Chinchilla の計算量最適経路（$a \approx b \approx 0.5$）にとどまれるか。

<!--en:86abd34e-->
3. **活性化チェックポイント（$8ND$）：**完全なブロック単位の活性化チェックポイントは、順伝播（$2ND$）を誤差逆伝播法（$4ND$）の最中に一度再計算する。これによりトークンあたりの訓練コストの式はどう変わるか。

<!--en:557b98b6-->
**解答：**

<!--en:8f73aac5-->
1. **$C \approx 6 \times (1.24 \times 10^8) \times 10^{10} \approx \mathbf{7.4 \times 10^{18}\text{ FLOPs}}$** であり、アテンション項は線形射影に対して $T / (6d) = 1024 / (6 \times 768) \approx \mathbf{22.2\%}$ を追加する

<!--en:59176b1f-->
2. $N^*$ と $D^*$ はともに **$100^{0.5} = \mathbf{10\times}$** にスケーリングし、パラメータあたり $D^* / N^* \approx 20$ トークンという比率は不変に保たれる

<!--en:055a1c49-->
3. 総訓練計算量は $33\%$ 増加して $C \approx 6 N D$ から **$C \approx (2 + 2 + 4) N D = \mathbf{8 N D}$** になり、引き換えに層あたりの活性化メモリが $O(1)$ になる

<!--en:e3222d32-->
::: {.slide title="まとめ：パラメータ数・FLOP・Chinchilla スケーリング則"}

<!--en:c5a222f8-->
[要点のまとめ]{.kicker}

<!--en:8702d295-->
| 法則・量 | 公式 | 実務上の意味と経験則 |

<!--en:e6dac5e6-->
| **非埋め込みパラメータ** | $N \approx 12 L d^2$（アテンション $4 L d^2$ $+ 8 L d^2$ FFN） | 密なモデルで誤差 $<1\%$、重み共有埋め込みには $V d$ を加算 |

<!--en:a9c049c2-->
| **訓練 FLOPs** | $C \approx 6 N D \cdot \bigl(1 + \frac{T}{6d}\bigr)$ | 順伝播 $2ND$ $+ 4ND$ 逆伝播、$T \gtrsim 6d$ のときのみアテンションが支配的 |

<!--en:9d4de3d2-->
| **Chinchilla 損失曲面** | $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta}$ | エントロピー下限、容量誤差、データ誤差への加法分解 |

<!--en:02db57ea-->
| **計算量最適な分割** | $N^*(C) \propto C^{\frac{\beta}{\alpha+\beta}} \approx C^{0.5},\; D^*(C) \propto C^{0.5}$ | $N$ と $D$ を $\sqrt{C}$ に比例して等しく拡大、$D^* / N^* \approx 20\text{ tokens/param}$ を維持 |

<!--en:93494dbf-->
| **推論最適へのシフト** | $6 N D + 2 N D_{\text{inf}}$ を $\mathcal{L}(N,D)=\mathcal{L}_0$ 上で最小化 | サービングコスト削減のため小型モデルを $200\text{--}2{,}000\text{ tokens/param}$ まで訓練 |
