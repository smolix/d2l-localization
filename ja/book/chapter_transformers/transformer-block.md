<!--en:f389f79e-->
# Transformer ブロック

<!--en:243267c5-->
自己アテンション（self-attention、:numref:`chap_attention`）は、系列の位置間で情報を混合する。すなわち、各トークンの出力は、アテンションを向ける位置の値ベクトルの重み付き平均となる。しかし、アテンション層単体では、その重み付き和の前後に線形射影（$W_Q, W_K, W_V, W_O$）を適用するにすぎない。位置ごとの非線形性、残差接続、および正規化がなければ、アテンション層を積み重ねてもトークンごとの階層的な特徴量を構築できず、数十層にわたって信号を安定して伝播させることもできない。

<!--en:dd0fd990-->
**Transformer ブロック** :cite:`Vaswani.Shazeer.Parmar.ea.2017` は、マルチヘッドアテンションと位置ごとの**フィードフォワードネットワーク（FFN）**を組み合わせ、各サブ層を加算的な**残差接続**（residual connection）で包み :cite:`He.Zhang.Ren.ea.2016`、**層正規化**（layer normalization）によって活性化値を安定化させる（:numref:`fig_transformer-block`）。現代のブロックと 2017 年の元の Transformer を分ける設計上の改良点は、次の 3 つである。

<!--en:b735c7dc-->
1. **正規化の配置：**正規化を残差ストリーム（residual stream）自体（**Post-Norm**）から各サブ層の入力（**Pre-Norm**）へと移すことで、深さ方向に減衰しない恒等経路が保たれる。

<!--en:e85f95c2-->
2. **正規化の形式：**平均中心化を伴う `LayerNorm` を二乗平均平方根（root-mean-square）による正規化（**`RMSNorm`**）と任意のヘッドごとの**クエリ・キー正規化**（query–key normalization、`QK-Norm`）に置き換えることで、メモリ帯域幅のオーバーヘッドを抑えつつ活性化値とアテンションロジットを有界に保つ。

<!--en:9135e9ae-->
3. **ゲート付きフィードフォワードネットワーク：**2 行列の ReLU・GELU MLP（$d_{\textrm{ff}} = 4d$）をゲート付きの **`SwiGLU`** ネットワーク（$d_{\textrm{ff}} \approx \frac{8}{3}d$）に置き換えることで、まったく同じ $8d^2$ のパラメータ数と FLOP 予算のまま表現の質を高める。

<!--en:36d515b2-->
## Transformer ブロックの構成要素

<!--en:4f960f94-->
Transformer ブロックは、共有された**残差ストリーム** $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$（:numref:`sec_what-attention-computes`）上で動作する。連続する 2 つのサブ層は $\mathbf{X}$ を上書きするのではなく、ストリームから値を読み取り、更新量を計算して、その結果をストリームに加算して戻す。

<!--en:dcd98a4b-->
- **マルチヘッドアテンション**（**トークン混合**、token mixing）は、系列の位置どうしが相互作用する唯一の箇所である。各トークンは系列にクエリを送り（:numref:`sec_multihead-attention`）、取得したコンテキストベクトルを自身のストリームに加算する。

<!--en:76a8fffb-->
- **位置ごとのフィードフォワードネットワーク**（**チャネル混合**、channel mixing）は、同じ MLP を各トークンベクトル $\mathbf{x}_i \in \mathbb{R}^d$ に独立に適用し、アテンションが集約した特徴量を変換する。密なブロックのパラメータ数の約 3 分の 2 は FFN に存在する。

<!--en:dcf28fc1-->
現代の Transformer で用いられる **Pre-Norm** 配置（:numref:`fig_transformer-block`）では、各サブ層は更新量を計算する前に自身の入力を正規化する。

<!--en:dc8e315a-->
![Transformer ブロック。アテンションと位置ごとの FFN が残差ストリームから値を読み取り、計算結果をストリームに加算して戻す。左は 2017 年の配置であり、正規化は各加算の後にストリーム自体の上に置かれる。右は現代の配置であり、正規化は各ブランチの上へと移り、ストリームは入力から出力まで途切れることなく通る。](../img/mdl-transformers-block-anatomy.svg)

<!--en:3e8d0762-->
## 正規化

<!--en:b8348ffb-->
### Pre-Norm と Post-Norm

<!--en:4e2ee620-->
2017 年の元の Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017` と BERT :cite:`Devlin.Chang.Lee.ea.2018` は **Post-Norm** を採用しており、各加算の後に*残差ストリーム自体の上*に正規化を置いていた。

<!--en:a76eb4ef-->
なぜ現代のアーキテクチャは :eqref:`eq_postln-block` から :eqref:`eq_preln-block` へと切り替わったのだろうか。最小限のブロック `MiniBlock`（フラグ `pre_norm` で 2 つの配置を切り替える）を使えば、1 ステップも訓練することなく初期化の段階でその違いを直接確認できる。

<!--en:bc2135a5-->
### 初期化時の信号伝播とトークン崩壊

<!--en:11e790d5-->
単位分散をもつランダムなトークンベクトルのバッチ $\mathbf{X}^{(0)} \in \mathbb{R}^{B \times n \times d}$ を、初期化したばかりの $L = 32$ 個のブロックに通すことを考えよう。以下の関数 `signal_stats` は、積層された各ブロックのすべての重み行列を、2017 年の Transformer と同じ Xavier 初期化で、あるいは $\mathcal{N}(0, \textrm{std}^2)$ の要素（`std` が指定された場合）で新たにサンプリングする。各ブロック $\ell \in \{1, \dots, L\}$ において、次の 3 つの量を追跡する。

<!--en:bdc8c7f7-->
1. **ストリーム標準偏差**（$\mathbf{X}^{(\ell)}$ の全要素にわたる標準偏差）。

<!--en:57f46e3e-->
2. **トークンの広がり**（token spread）。$n$ 個の系列位置にわたる標準偏差として測られ、$n$ 個すべてのトークンが同じベクトルへと崩壊すると消失する。

<!--en:50534d1d-->
3. **クエリ勾配ノルム** $\|\partial \mathcal{L} / \partial W_q^{(\ell)}\|_F$（スタック最上部の単位スカラー損失から逆伝播してくる勾配のノルム）。

<!--en:c29f371c-->
なぜ Post-Norm では深さとともにトークンの多様性が失われるのだろうか。各トークンを全トークンが共有する成分と固有の成分に分け、$\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$ と書く。ここで $\boldsymbol{\mu} = \frac{1}{n}\sum_j \mathbf{x}_j$ は $n$ 個の位置にわたる平均である（:numref:`fig_mdl-transformers-token-collapse`）。初期化時、アテンション重みはほぼ一様であるため、どのトークンもほぼ同じ更新量 $\mathbf{u} \approx \mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$ を受け取る。すなわち、偏差は平均化されて打ち消し合い、平均成分だけが先へ伝わる。2 つの射影はそれぞれ入力の長さの二乗を $g^2 = d\sigma^2$ 倍（$\sigma^2$ は重みの各要素の分散）にし、確率ベクトル $\mathbf{u}$ は $\boldsymbol{\mu}$ とほぼ直交する。したがって、共有成分のエネルギーは $1 + g^4$ 倍に増大する一方、偏差は元のままである。続く正規化ではこれを修復できない。すべてのトークンを同じ係数で再スケーリングするため、ストリームの長さは元に戻るものの、平均と偏差のバランスは戻らないからである。

<!--en:9a6bb650-->
![1 つの Post-Norm ブロックにおけるトークン崩壊。各トークンは共有平均 $\boldsymbol{\mu}$ と固有の偏差 $\boldsymbol{\delta}_i$ に分解される。ほぼ一様なアテンションがすべてのトークンに同じ更新量 $\mathbf{u} \approx \mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$ を加算するため、平均だけが増大する。続く正規化が全トークンを単一の係数で再スケーリングすることで、偏差が縮小する。Xavier 初期化（$g = 1$）の場合を図示しており、このとき $\mathbf{u}$ は $\boldsymbol{\mu}$ と同じ長さをもち、係数は $1/\sqrt{2}$ となる。](../img/mdl-transformers-token-collapse.svg)

<!--en:1e4e7366-->
共有エネルギーとトークン固有のエネルギーの比を $\rho = n\|\boldsymbol{\mu}\|^2 / \sum_i \|\boldsymbol{\delta}_i\|^2$ とする。各ブロックは $\rho$ を $1 + g^4$ 倍にする。ストリームのノルムのうち偏差が占める割合であるトークンの広がりは $s = (1 + \rho)^{-1/2}$ であり、$n$ 個の独立なランダムトークンは $\rho_0 = 1/(n-1)$ から出発する。したがって、$k$ ブロック後には次のようになる。

<!--en:f7e9a4c4-->
ひとたび平均が支配的になると、広がりはブロックごとに $1/\sqrt{1 + g^4}$ 倍に縮小し、その値は $1/\sqrt{2}$ となる（Xavier 初期化、すなわち $g = 1$ の場合）。なお、:eqref:`eq_collapse-rate` に FFN は現れない。FFN は各トークンに個別に作用し、平均と偏差を区別できないため、両者の比をほとんど変えないからである（演習問題 :exref:`ex_transformer-block-ffn-width-rounding`）。

<!--en:6e39a545-->
Pre-Norm（:eqref:`eq_preln-block`）のもとでは、残差ストリームが正規化の分母で割られることはない。$L$ 個のブロックを展開すると、直接の和 $\mathbf{X}^{(L)} = \mathbf{X}^{(0)} + \sum_{\ell=1}^{2L} \mathbf{U}^{(\ell)}$ となる。$2L$ 個の独立なサブ層出力がストリームに分散を加えるため、スケーリングなしの Pre-Norm ストリームの RMS は深さとともに $\sqrt{1 + 2\ell\sigma^2}$ のように増大する（全 $2L$ 個のサブ層が加える合計分散が $O(1)$ となるよう任意の深さ $L$ で抑えるため、GPT-2 :cite:`Radford.Wu.Child.ea.2019` は各ブロックの残差出力射影 `W_o` と `W_2` の初期化標準偏差を $1/\sqrt{2L}$ 倍にスケーリングしており、これは :numref:`sec_gpt` で実装し、演習問題 :exref:`ex_transformer-block-stream-variance` で検証する）。一方、Post-Norm において BERT :cite:`Devlin.Chang.Lee.ea.2018` は、:eqref:`eq_collapse-rate` の崩壊を遅らせるためにすべての線形重みをより小さな標準偏差 $\sigma = 0.02$ で初期化する（これにより $g^2 \approx 0.10$、$g^4 \approx 0.01$ となる。ただし $d = 256$ の場合）。これら 3 つの構成（`post-norm`、`pre-norm`、`post-norm, std 0.02`）を 32 ブロックにわたって比較してみよう。

<!--en:66cc7b98-->
ブロック $k = 1, \ldots, 32$ にわたるトークンの広がりの実測値を、2 つの Post-Norm 条件に対する :eqref:`eq_collapse-rate` の予測（点線、$n = 64$）と並べてプロットする。

<!--en:93166c88-->
標準的な Xavier 初期化による Post-Norm（`post-norm`）では、トークンの広がりはブロック 32 までにほぼゼロ（$\sim 10^{-4}$）へと崩壊し、:eqref:`eq_collapse-rate` の予測によく一致する。系列内のすべてのトークンが、ほぼ同一のベクトルへと押し潰されているのである。実測曲線が点線をわずかに下回っているのは、FFN による小さな寄与である。Post-Norm の初期化を縮小すると（`post-norm, std 0.02`）、ブロックごとの収縮が十分に遅くなり、32 ブロックにわたってトークンの広がりが保たれる。Pre-Norm（`pre-norm`）も無縁ではなく、アテンションサブ層は同様に共有平均へ加算する。しかしストリーム自体が深さとともに増大するため、各更新の相対的な重みは前の層より小さくなり、広がりの減衰は緩やかである。Xavier スケールであっても 32 ブロック後に 3 分の 1 以上が残っている（その代わり、ストリームの RMS は $\sqrt{L}$ のオーダーで増大するが、これは $1/\sqrt{2L}$ の出力射影スケーリングと組み合わせることで抑えられる。演習問題 :exref:`ex_transformer-block-stream-variance` を参照）。

<!--en:f45fd701-->
### 初期化時の勾配

<!--en:f7555db1-->
トークン崩壊（token collapse）は順伝播の性質であるが、どの重みが学習できるかも決定づける。クエリ射影の勾配ノルム $\|\partial \mathcal{L} / \partial W_q^{(\ell)}\|_F$ をブロック番号 $\ell \in \{1, \dots, 32\}$ にわたってプロットする。

<!--en:32430eb0-->
`post-norm` では、クエリ勾配はブロック 1 で最大（$10^{-2}$ 程度）となり、ブロック 32 では約 $10^{-9}$ まで落ち込む。これは、損失から最も遠い層ほど受け取る勾配が小さくなる深い MLP の勾配消失とは逆の現象である。勾配は逆伝播の途中で失われているのではない。深いブロックが順伝播で受け取る入力に原因があるのである。ブロックに入るトークンどうしがほぼ一致してしまうと、どのキーも同じスコアを受け取り、どの値も同じになるため、クエリを動かしても出力は何も変化しない。その結果、クエリ勾配はトークンの広がりの二乗に比例して縮小し（スコアが偏差 $\boldsymbol{\delta}_i$ を通じてしか異ならないことで 1 乗分、値も同様であることでさらに 1 乗分縮小する）、キー勾配もそれに伴って縮小する。他の重みは影響を受けない。ブロック 32 における `W_v` と `W_o` の勾配は、`post-norm` と `pre-norm` で同じ桁の大きさにある（上の各条件の出力の最終行）。

<!--en:8351bfcd-->
`pre-norm` では、恒等項 $I$（$\partial \mathbf{X}^{(L)} / \partial \mathbf{X}^{(\ell)} = I + \cdots$ に現れる項）が損失からの勾配をすべてのブロックへと運ぶ。こちらのクエリ勾配も一様ではなく、トークンの広がりが緩やかに減衰するにつれてスタック全体で約 30 分の 1 に低下する。それでも、Post-Norm が 6 桁以上も失うのに対し、$10^{-3}$ 以上を保っている。:citet:`xiong2020layer` は、初期化時の Post-Norm Transformer における別の不均衡、すなわち出力付近のパラメータに対する勾配が大きくなる現象を解析し、Post-Norm の訓練が学習率のウォームアップに依存する一方で Pre-Norm はそれなしで訓練できる理由がここにあることを示している。

<!--en:bb93ce31-->
### RMSNorm

<!--en:032a9029-->
標準的な LayerNorm（:numref:`sec_batch_norm`）は、各トークンベクトル $\mathbf{x} \in \mathbb{R}^d$ からその平均 $\mu = \frac{1}{d}\sum_j x_j$ を引いて中心化し、標準偏差でスケーリングしたうえで、学習されるゲイン $\boldsymbol{\gamma}$ とバイアス $\boldsymbol{\beta}$ を適用する。:citet:`Zhang.Sennrich.2019` は、平均中心化と加算バイアスは訓練の安定性にほとんど寄与せず、重要なのは二乗平均平方根の大きさを正規化することだと示した。$\mu$ と $\boldsymbol{\beta}$ を取り除くと **RMSNorm** が得られる。

<!--en:d9879676-->
ここで、$\boldsymbol{\gamma} \in \mathbb{R}^d$ は 1 で初期化され、$\epsilon = 10^{-5}$ である。RMSNorm はトークンごとに平均と分散の両方ではなく二乗平均の縮約 $\frac{1}{d}\sum_j x_j^2$ を 1 回計算するだけで済むため、LayerNorm と同等の正解率を保ちながらメモリ上の縮約演算を減らせる。

<!--en:cbbe1c47-->
### クエリ・キー正規化（QK-Norm）

<!--en:beb09a8e-->
Pre-Norm はアテンションサブ層に入る入力 $\mathbf{X}$ を正規化するが、アテンション内部ではクエリ $\mathbf{Q} = \textrm{Norm}(\mathbf{X})\mathbf{W}_q$ とキー $\mathbf{K} = \textrm{Norm}(\mathbf{X})\mathbf{W}_k$ に学習される射影行列 $\mathbf{W}_q, \mathbf{W}_k$ が掛けられる。訓練中に $\mathbf{W}_q$ と $\mathbf{W}_k$ の特異値が増大すると、スケール付きドット積ロジット $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ は重みのスケールに対して二次関数的に増大し、ソフトマックスがワンホットな飽和状態に追い込まれて勾配が消失する :cite:`Henry.Dachapally.Pawar.ea.2020,Dehghani.Djolonga.Mustafa.ea.2023`。

<!--en:37339a69-->
**QK-Norm** は、アテンションスコアを計算する直前に、ヘッドごとの幅 $d_h$ の `RMSNorm` を $\mathbf{Q}$ と $\mathbf{K}$ に適用する。コーシー・シュワルツの不等式により、単位 RMS をもつ 2 つのベクトルのドット積は大きさが高々 $d_h$ に抑えられるため、スケール付きロジット $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ の絶対値は $\sqrt{d_h}$ 以下に有界となり（さらにランダムな方向に対しては単位分散をもつ）、これは $\|\mathbf{W}_q\|$ や $\|\mathbf{W}_k\|$ がどれほど大きくなっても変わらない。

<!--en:3cc4f5fd-->
QK-Norm なしで $Q$ と $K$ を $\lambda = 4\times$ および $\lambda = 16\times$ にスケーリングすると、ロジットの標準偏差は二次関数的に跳ね上がり（$\approx 16$ および $\approx 256$）、ソフトマックスのエントロピーはゼロに向かって崩壊する。一方、`rms_norm` を $Q$ と $K$ に適用すれば、ロジットの広がりとアテンションのエントロピーは重みのスケールに対して完全に不変となる。

<!--en:ca6d3eb0-->
## フィードフォワードネットワーク：GELU から SwiGLU へ

<!--en:d639a04d-->
元の Transformer のフィードフォワードネットワークは、チャネル幅を $d$ から $d_{\textrm{ff}} = 4d$ へと拡大し、要素ごとの活性化関数（$\textrm{ReLU}$ が :citet:`Vaswani.Shazeer.Parmar.ea.2017` で、$\textrm{GELU}$ が GPT-2 と BERT で使われた）を適用してから $d$ へと射影し直す 2 層の MLP である。

<!--en:b0fa6e7c-->
これは $2 \cdot d \cdot 4d = 8d^2$ 個の重みパラメータを用いる。

<!--en:f916b73a-->
:citet:`Dauphin.Fan.Auli.ea.2017` および :citet:`Shazeer.2020` に従い、現代の Transformer は :eqref:`eq_ffn` をバイアスなしの **SwiGLU** ゲート付き線形ユニット（gated linear unit）に置き換える。これは、$\textrm{SiLU}$ で活性化したゲート射影 $\mathbf{x}\mathbf{W}_g$（$\textrm{SiLU}(z) = z \sigma(z)$）と線形アップ射影 $\mathbf{x}\mathbf{W}_1$ の要素ごとの積をとってから、$\mathbb{R}^d$ へと $\mathbf{W}_2$ で射影し直すものである。

<!--en:618dc43c-->
`SwiGLU` は形状 $d \times d_{\textrm{ff}}$（$\mathbf{W}_g, \mathbf{W}_1$）と $d_{\textrm{ff}} \times d$（$\mathbf{W}_2$）の *3* つの行列を使うため、$d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$ と設定すれば $3 d d_{\textrm{ff}} \approx 8d^2$ となり、$4d$ の GELU MLP とパラメータ数・FLOP 数の両方が一致する。

<!--en:68a0e4ca-->
## 設定可能な `TransformerBlock`

<!--en:8df01549-->
ここで `TransformerBlock` を組み立てよう。このクラスは、Pre-Norm または Post-Norm の配置（`pre_norm`）、`RMSNorm` または `LayerNorm`（`norm`）、`SwiGLU` または `GELU` の FFN（`act`）、バイアスの切り替え（`bias`）、および差し替え可能な `attn_factory` と `ffn_factory` のフックをひとまとめにしたものである。これらのフックにより、後の節ではブロックを書き直すことなく、:numref:`sec_gpt` の因果的 RoPE アテンションや :numref:`sec_moe` の混合エキスパート（mixture of experts）層をそのまま組み込める。

<!--en:bb6567af-->
### パラメータ数

<!--en:e774321e-->
デフォルトの現代的なブロック（`RMSNorm`、`SwiGLU`、`bias=False`）を 2017 年版・GPT-2 版の変種（`GELU`、`LayerNorm`、`bias=True`）と $d = 256, H = 8$ で比較すると、どの構成も $\approx 12 d^2 = 786{,}432$ 個のパラメータ（アテンションに $4d^2$、FFN に $8d^2$）をもつことが確認できる。

<!--en:30ac03ff-->
## まとめ

<!--en:d646c5ec-->
**Transformer ブロック**は、**トークン混合**（系列の位置間にわたるマルチヘッドアテンション）と**チャネル混合**（$d$ 個の特徴量にわたる位置ごとのフィードフォワードネットワーク）を分離し、両方の更新量を共有された残差ストリームに加算する。**Pre-Norm**（:eqref:`eq_preln-block`）は各サブ層への入力のみを正規化して残差ストリームを途切れのない恒等経路として残すのに対し、**Post-Norm**（:eqref:`eq_postln-block`）はストリーム自体を正規化する。Xavier スケールでは、深い Post-Norm スタックはトークン間の差異を指数関数的な速さで失い（:eqref:`eq_collapse-rate`）、上層ブロックのクエリ勾配とキー勾配を枯渇させる一方、各ブロックの残差出力射影（`W_o` と `W_2`）を $1/\sqrt{2L}$ 倍にスケーリングすれば、Pre-Norm のストリーム分散は $O(1)$ のまま任意の深さ $L$ にわたって保たれる。現代のブロックは、LayerNorm の平均減算と加算バイアスを省いた **RMSNorm**（:eqref:`eq_rmsnorm`）とヘッドごとの **QK-Norm** を組み合わせることで、射影重みの増大にかかわらずコーシー・シュワルツの不等式によってアテンションロジットを有界に保つ。最後に、**SwiGLU**（:eqref:`eq_swiglu`）は 2 行列・幅 $4d$ のフィードフォワードネットワークを、幅 $d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$ の 3 行列によるゲート付き積に置き換え、標準的な FFN と同じ $8d^2$ のパラメータ数および FLOP 予算と一致させる。

<!--en:2294243a-->
## 演習問題

<!--en:ede77121-->
1. **RMSNorm のスケール不変性とそのヤコビ行列。**$\mathbf{y} = \textrm{RMSNorm}(\mathbf{x})$ が :eqref:`eq_rmsnorm` に従い、$\epsilon = 0$ かつ $\boldsymbol{\gamma} = \mathbf{1}$ で作用するものとする。
   1. $\textrm{RMSNorm}(c\mathbf{x}) = \textrm{RMSNorm}(\mathbf{x})$ が任意のスカラー $c > 0$ に対して成り立つことを示せ。
   2. $\mathbf{y} = \sqrt{d}\, \mathbf{x} / \|\mathbf{x}\|_2$ を微分してヤコビ行列が $J(\mathbf{x}) = \frac{\sqrt{d}}{\|\mathbf{x}\|_2}\bigl(\mathbf{I} - \frac{\mathbf{x}\mathbf{x}^\top}{\|\mathbf{x}\|_2^2}\bigr)$ となることを示し、$J(\mathbf{x})\mathbf{x} = \mathbf{0}$ を確かめよ。$\mathbf{x}$ に平行な任意の動径方向の摂動が $J(\mathbf{x})$ の零空間に属さなければならないのはなぜか。

<!--en:c581ee64-->
2. **展開された残差ストリームと勾配の恒等経路。**$2L$ 個のサブ層（$L$ ブロックの Pre-Norm スタックを構成するサブ層）に $f_1, f_2, \dots, f_{2L}$ と番号を振る。ここで、奇数番号はアテンションサブ層、偶数番号は FFN サブ層とする。
   1. :eqref:`eq_preln-block` を展開し、$\mathbf{X}^{(L)}$ を $\mathbf{X}^{(0)} + \sum_{m=1}^{2L} f_m(\textrm{Norm}(\mathbf{X}^{(m-1)/2}))$ として陽に表せ。
   2. 連鎖律を適用して $\partial \mathbf{X}^{(L)} / \partial \mathbf{X}^{(0)}$ を $2L$ 個のヤコビ行列の積として書き下し、主要項を展開したうえで、得られる恒等項 $\mathbf{I}$ を、すべての因子に左から $J_{\textrm{Norm}}$ が掛かる Post-Norm の積と対比せよ。

<!--en:a9c38797-->
3. **`TransformerBlock` 内部の QK-Norm。**`attn_factory` を作成し、`d2l.MultiHeadAttention` をラップして、アテンションスコアの計算前にヘッドごとのクエリとキーへ幅 `d // h` の RMSNorm を適用するようにせよ。射影重みを $4\times$（または $16\times$）に定数倍した 16 ブロックのスタック上で、アテンションのエントロピーがゼロから離れた値に保たれることを確かめよ。

<!--en:9cad4238-->
4. **深さ方向のストリーム分散予算。**Pre-Norm 配置では、すべてのサブ層が残差ストリームに更新量を加算する。`signal_stats` を修正し、`W_o` と `W_2` を標準偏差 $0.02/\sqrt{2L}$ で、他のすべての行列を標準偏差 $0.02$ でサンプリングするようにして、すべての行列を標準偏差 $0.02$ とした場合と $L \in \{8, 16, 32, 64\}$ にわたって比較せよ。スケーリングなしでは最終的なストリームの RMS が $L$ とともに増大する一方、$1/\sqrt{2L}$ 則のもとではほぼ一定に保たれることを確かめよ。
   :exlabel:`ex_transformer-block-stream-variance`

<!--en:b6f90639-->
5. **トークン崩壊のレート。**:eqref:`eq_collapse-rate` の予測には FFN に対応する因子が含まれていないが、これはトークン間を混合するのはアテンションサブ層だけであり、FFN は位置ごとに作用するという議論に基づいている。
   1. `MiniBlock` を修正してアテンションサブ層をスキップし（$\mathbf{H} = \textrm{Norm}(\mathbf{X})$）、Post-Norm の FFN のみが実行されるようにしたうえで、`signal_stats(pre_norm=False)` を再実行せよ（アテンションの重みは使われなくなるため、勾配に関する行は削除せよ）。32 ブロック後にトークンの広がりはどれだけ残っているか。
   2. 次に FFN をスキップし、アテンションサブ層を残せ。実測される広がりは :eqref:`eq_collapse-rate` に $k = 8, 16, 32$ でどれほど近づくか。
   3. FFN のみのスタックでもわずかな収縮が起きる。位置ごとのネットワークが全トークンに共通する小さなベクトルを加算できてしまうのは、GELU のどの性質によるものか。
   :exlabel:`ex_transformer-block-ffn-width-rounding`

<!--en:3cce5cba-->
[ディスカッション](https://d2l.discourse.group/t/36774)

<!--en:26de144d-->
Transformer ブロック：Pre-Norm、SwiGLU、信号伝播<br>**残差ストリーム・Pre-Norm と Post-Norm・トークン崩壊・RMSNorm と QK-Norm・SwiGLU・12d² のパラメータ集計**

<!--en:ddf34141-->
::: {.slide title="なぜアテンションを積むだけでは深層で破綻するのか"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:3bca3ab5-->
- **トークンごとの非線形性の欠如：**アテンション層の間に位置ごとの FFN がなければ、線形射影は各トークンのチャネル上の単一な低ランク写像へと潰れる

<!--en:e7849f3e-->
- **トークン崩壊と勾配の枯渇：**Xavier 初期化の Post-Norm ブロックを $L = 32$ 層積むと、トークンの広がりは $\sim 10^{-4}$ に崩壊し、ブロック 32 のクエリ勾配 $\|\partial \mathcal{L} / \partial \mathbf{W}_q^{(32)}\|_F$ は $\sim 10^{-9}$ まで低下する

<!--en:fd196e07-->
::: {.slide title="残差ストリーム上のトークン混合とチャネル混合"}

<!--en:37774c41-->
[ブロックの構造]{.kicker}

<!--en:a9005933-->
- **共有される残差ストリーム $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$：**各サブ層は表現を上書きせず、ストリームから読み取って $\mathbb{R}^{B \times n \times d}$ の更新量を計算し、加算して戻す

<!--en:b15b71b1-->
- **マルチヘッドアテンション（トークン混合）：**系列位置 $1, \dots, n$ が情報をやり取りする*唯一*の演算（パラメータ数 $4d^2$）

<!--en:87e27c8f-->
- **位置ごとの FFN（チャネル混合）：**同一の MLP を各トークンベクトル $\mathbf{x}_i \in \mathbb{R}^d$ に独立に適用し、アテンションが集めた文脈を変換する（パラメータ数 $8d^2$、ブロック全体の 3 分の 2）

<!--en:03cf153e-->
::: {.slide title="正規化の配置：Post-Norm と Pre-Norm"}

<!--en:c0dae5ea-->
[正規化の配置]{.kicker}

<!--en:ba828e07-->
- **Post-Norm（2017 年・BERT）：**主たる残差ストリーム自体の上に正規化を置く
  $$\mathbf{H} = \mathrm{Norm}_1(\mathbf{X} + \mathrm{Attn}(\mathbf{X})), \quad \mathbf{Y} = \mathrm{Norm}_2(\mathbf{H} + \mathrm{FFN}(\mathbf{H}))$$

<!--en:62335f5d-->
- **Pre-Norm（GPT-2・Llama・Qwen）：**正規化を各サブ層の入力ブランチに移し、主ストリームを途切れなく通す
  $$\mathbf{H} = \mathbf{X} + \mathrm{Attn}(\mathrm{Norm}_1(\mathbf{X})), \quad \mathbf{Y} = \mathbf{H} + \mathrm{FFN}(\mathrm{Norm}_2(\mathbf{H}))$$

<!--en:755cfb88-->
::: {.slide title="Post-Norm と Pre-Norm の違いはコード 2 行"}

<!--en:ce0f4436-->
[実装]{.kicker}

<!--en:5d873470-->
診断用の `MiniBlock` は両配置で同一の重みを用い、`norm1` と `norm2` で残差和を包むかブランチ入力を包むかだけを切り替える。

<!--en:45f2a54f-->
未訓練の $L$ ブロックのスタックは決定論的なランダム関数であるため、信号の分散、トークン間の多様性、および勾配が初期化時（$t = 0$）に両配置でどう伝播するかを測定できる。

<!--en:8041dbf5-->
::: {.slide title="なぜ Post-Norm は深さ方向にトークンを押し潰すのか"}

<!--en:7b76940e-->
[トークン崩壊の仕組み]{.kicker}

<!--en:6d16f412-->
- 各トークンベクトル $\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$ を系列平均 $\boldsymbol{\mu} = \frac{1}{n}\sum_j \mathbf{x}_j$ とトークン固有の偏差 $\boldsymbol{\delta}_i$ に分解する

<!--en:b1fd1ba4-->
- **ランダムなアテンションが $\boldsymbol{\mu}$ を強める：**一様な初期重みは全トークンに $\mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$ を加算し、偏差 $\boldsymbol{\delta}_i$ は打ち消し合う

<!--en:517d8b6d-->
- **Post-Norm が下方に再スケーリング：**全トークンを同じ係数で割るとストリーム長は戻るが、毎層で $\boldsymbol{\delta}_i$ が $\boldsymbol{\mu}$ に対して縮小する（`LayerNorm` が中心化するのはトークンではなく*チャネル*）

<!--en:c5b14f15-->
::: {.slide title="トークン崩壊の指数的レートの予測"}

<!--en:de057a66-->
[数学的導出]{.kicker}

<!--en:ee777553-->
- 共通成分と固有成分のエネルギー比を $\rho = n\|\boldsymbol{\mu}\|^2 / \sum_{i=1}^n \|\boldsymbol{\delta}_i\|^2$ とおくと、相対的なトークンの広がりは $s = (1 + \rho)^{-1/2}$

<!--en:8ba6b756-->
- 射影ごとの分散ゲイン $g^2 = d\sigma^2$ のもとで、各 Post-Norm ブロックは $\rho \mapsto (1 + g^4)\rho$ と増幅させる
  $$s_k \approx \left(1 + \frac{(1 + g^4)^k}{63}\right)^{-1/2}$$

<!--en:f722f38b-->
::: {.slide title="トークンの広がりの実測値は解析則と一致"}

<!--en:63718f6d-->
[実験による検証]{.kicker}

<!--en:0a5054d9-->
- **Post-Norm + Xavier（`C0`）：**トークンの広がりは深さ方向に指数減衰し、ブロック 32 で $\sim 10^{-4}$ に低下する（FFN でも `GELU` の正の平均が小さな共有ベクトルを加えるため、アテンションのみの点線をわずかに下回る）

<!--en:c1be2f73-->
- **Post-Norm + $\mathcal{N}(0, 0.02^2)$（`C2`）：**小さな重みにより $g^4 \approx 0.01$ に抑えられ、トークンの広がりは $0.97$ のまま 32 層を通じて維持される（幅 $d = 256$ の場合）

<!--en:f14628e7-->
- **Pre-Norm + Xavier（`C1`）：**大きな Xavier 重み（$g = 1$）でも、Pre-Norm は指数的な崩壊を起こさず十分なトークンの多様性（$s_{32} \approx 0.37$）を保つ

<!--en:0c156638-->
::: {.slide title="落とし穴：固定重みスケール 0.02 はモデル幅とともに増大"}

<!--en:af5b852a-->
[幅スケーリングの落とし穴]{.kicker}

<!--en:131ee3a1-->
- 要素の分散を固定した $\mathbf{W} \in \mathbb{R}^{d \times d}$（$\sigma^2 = 0.02^2$）では、射影ごとの分散ゲインが幅 $d$ に比例して増大する
  $$g^2 = d \sigma^2 = 0.0004\,d$$

<!--en:42e34be6-->
- 固定の $\sigma = 0.02$ が小さいといえるのは $d \ll 2{,}500$ の範囲に限られ、最先端モデルの幅（$d = 4{,}096$）では $g^2 = 1.64 > 1$ となって Post-Norm スタックは Xavier よりも速く崩壊する

<!--en:ab1747ed-->
::: {.slide title="トークン崩壊によるクエリ・キー勾配の枯渇"}

<!--en:f410a91e-->
[逆方向の信号伝播]{.kicker}

<!--en:a02928ae-->
- **上層での $\mathbf{W}_q, \mathbf{W}_k$ の勾配消失：**ブロック 32 への入力トークン $\mathbf{x}_1, \dots, \mathbf{x}_n$ がほぼ同一になると、$\mathbf{W}_q$ や $\mathbf{W}_k$ を摂動させても全アテンションスコアが等しくシフトするだけでソフトマックスの重み付き平均は変わらない（$\|\nabla_{\mathbf{W}_q^{(32)}} \mathcal{L}\|_F \sim 10^{-9}$）

<!--en:96c2d2a6-->
- **逆伝播の減衰ではない：**Post-Norm の $\mathbf{W}_q$ 勾配はブロック 1 で最大（$\sim 10^{-2}$）となり、トークンの広がりの二乗に比例して縮小する一方、$\mathbf{W}_v$ と $\mathbf{W}_o$ のブロック 32 における勾配は Pre-Norm と同程度の桁を保つ。**Pre-Norm**（`C1`）の低下は 6 桁超ではなく、約 30 分の 1（32 ブロック全体）にとどまる

<!--en:887544c3-->
::: {.slide title="Pre-Norm の展開ストリームと 1/√(2L) 残差スケーリング"}

<!--en:ee2de8be-->
[Pre-Norm の分散制御]{.kicker}

<!--en:88fa45c7-->
- $L$ 個の Pre-Norm ブロックを展開すると、途切れのない恒等経路が現れる
  $$\mathbf{X}^{(L)} = \mathbf{X}^{(0)} + \sum_{\ell=1}^{L} \bigl(\mathrm{Attn}_\ell(\mathrm{Norm}(\mathbf{X}^{(\ell-1)})) + \mathrm{FFN}_\ell(\mathrm{Norm}(\mathbf{H}^{(\ell)}))\bigr)$$

<!--en:d265f579-->
- **GPT-2 の $1/\sqrt{2L}$ スケーリング則：**ブロックあたりの残差書き込み行列 2 つ（`W_o` と `W_2`）の初期標準偏差を $1/\sqrt{2L}$ 倍にスケーリングすると、累積ストリーム分散は $\sum_{m=1}^{2L} \frac{\sigma^2}{2L} = \sigma^2 = O(1)$ となり任意の深さ $L$ で一定に保たれる

<!--en:6bdb215e-->
::: {.slide title="RMSNorm：高速化のための平均中心化の省略"}

<!--en:d25cce71-->
[正規化の形式]{.kicker}

<!--en:2925b450-->
Zhang & Sennrich（2019）は、`LayerNorm` の安定性がチャネル平均 $\mu = \frac{1}{d}\sum_j x_j$ の中心化ではなく、二乗平均平方根の大きさの抑制に由来することを示した。

<!--en:10844088-->
**スケール不変性とシフト不変性：**いずれの正規化も正の定数倍（$\mathbf{x} \mapsto 3\mathbf{x}$）に対して不変である一方、定数シフトの加算（$\mathbf{x} \mapsto \mathbf{x} + 10$）は `RMSNorm` を変化させるが `LayerNorm` は変化させない。$\mu$ と $\boldsymbol{\beta}$ を省くことで、トークンあたり HBM 上の縮約パスを 1 回削減できる。

<!--en:36fa3566-->
::: {.slide title="クエリ・キー正規化（QK-Norm）によるロジットの上界"}

<!--en:0821ecab-->
[アテンションの安定性]{.kicker}

<!--en:f7c75717-->
- **ロジットの二次関数的爆発：**訓練中に $\mathbf{W}_q$ と $\mathbf{W}_k$ のスペクトルノルムが $\lambda$ 倍に増大すると、アテンションロジット $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ は **$\lambda^2$** 倍に増大し、ソフトマックスを飽和させる

<!--en:0d385a2e-->
- **ヘッドごとの `QK-Norm`：**幅 $d_h$ の `RMSNorm` をクエリとキーに適用すると $\|\hat{\mathbf{q}}_i\|_2 = \|\hat{\mathbf{k}}_j\|_2 = \sqrt{d_h}$ が成り立ち、厳密なコーシー・シュワルツの上界が課される
  $$\frac{|\hat{\mathbf{q}}_i^\top \hat{\mathbf{k}}_j|}{\sqrt{d_h}} \le \sqrt{d_h}$$

<!--en:71030f11-->
::: {.slide title="QK-Norm のもとでのロジットとエントロピーの不変性の検証"}

<!--en:7a0e5db5-->
[QK-Norm の検証]{.kicker}

<!--en:18b06421-->
$\mathbf{W}_q$ と $\mathbf{W}_k$ を $\lambda \in \{1, 4, 16\}$ でスケーリングし、$n = 64$ 個のランダムトークン（$d_h = 64$、一様分布のエントロピー $\ln 64 \approx 4.16$ nat）上でソフトマックスの崩壊と `QK-Norm` による回復を確認する。

<!--en:1175030b-->
::: {.slide title="QK-Norm はロジットの広がりとエントロピーを一定に維持"}

<!--en:7a0e5db5-2-->
[QK-Norm の検証]{.kicker}

<!--en:df5cff3c-->
`QK-Norm` なしでは、重みが $16\times$ に増大するとロジットの標準偏差は $16^2 = 256\times$ に膨張し、アテンションのエントロピーは $\approx 0.01\text{--}0.03$ nat（ワンホット）へと崩壊する。一方、`QK-Norm` ありでは、ロジットの広がり（$\approx 1.0$）とエントロピー（$\approx 3.7$）は完全に不変に保たれる。

<!--en:ef75e1ef-->
::: {.slide title="キー・値メモリとしてのフィードフォワードネットワーク"}

<!--en:d1fa7d93-->
[チャネル混合]{.kicker}

<!--en:30ec0071-->
- 2 層 FFN を列ベクトル $\mathbf{w}_{1,j} \in \mathbb{R}^d$（$\mathbf{W}_1$ の各列）と行ベクトル $\mathbf{w}_{2,j} \in \mathbb{R}^d$（$\mathbf{W}_2$ の各行）で書き直すと、内容アドレス指定メモリバンクとしての構造が現れる（Geva et al., 2021）
  $$\mathrm{FFN}_{\mathrm{GELU}}(\mathbf{x}) = \sum_{j=1}^{4d} \mathrm{GELU}(\mathbf{x}^\top \mathbf{w}_{1,j})\, \mathbf{w}_{2,j}$$

<!--en:62aa7037-->
- **パラメータの大半を占有：**2 つの $d \times 4d$ 行列が $8d^2$ 個のパラメータを保持する（GPT-2 Small ではブロックあたり $3{,}072$ 個の検出器スロット）

<!--en:c3d7788e-->
::: {.slide title="GELU MLP からゲート付き SwiGLU ネットワークへ"}

<!--en:05e292cd-->
[ゲート付きフィードフォワードネットワーク]{.kicker}

<!--en:0e6886c9-->
- 現代の Transformer は 2 行列の `GELU` MLP をバイアスなしのゲート付き **`SwiGLU`** ネットワークに置き換える（Shazeer, 2020）
  $$\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x}) = \bigl(\mathrm{SiLU}(\mathbf{x}\mathbf{W}_g) \odot (\mathbf{x}\mathbf{W}_1)\bigr)\mathbf{W}_2$$

<!--en:091bf24c-->
- **$8d^2$ 予算との一致：**隠れ層の幅を $4d$ から $d_{\text{ff}} = \operatorname{round}(\frac{8}{3}d)$ に縮小することで、3 行列のパラメータ数を $3 \times d \times \frac{8}{3}d = 8d^2$ に保つ

<!--en:37fb06bc-->
::: {.slide title="パラメータ数を揃えた単一クラスでの両 FFN の実装"}

<!--en:12e5b97f-->
[FeedForward の実装]{.kicker}

<!--en:aea5dd9f-->
`FeedForward` モジュールは、$4d$ の `GELU` MLP と $\frac{8}{3}d$ の `SwiGLU` ネットワークを、まったく同じ $8d^2$ のパラメータ数・FLOP 予算で簡潔に切り替える。

<!--en:9ff50588-->
$d = 256$ のとき、`act='gelu'` は $2 \times 256 \times 1024 = 524{,}288$ 個のパラメータを使うのに対し、`act='swiglu'` は $3 \times 256 \times 683 = 524{,}544$ 個のパラメータを使う（丸めによる $8.004\,d^2$、すなわち $\frac{8}{3}\cdot 256 = 682.67 \to 683$ のため）。

<!--en:61b15b2f-->
::: {.slide title="設定可能な単一の TransformerBlock とパラメータ集計"}

<!--en:3deca60b-->
[統合ブロックとパラメータ集計]{.kicker}

<!--en:a4a7966c-->
`TransformerBlock` は `norm`（`'rms'` または `'layer'`）、`act`（`'swiglu'` または `'gelu'`）、`pre_norm`、`bias`、および差し替え可能な `attn_factory`・`ffn_factory` フックを統合し、常に $\approx 12d^2$ 個のパラメータを保持する。

<!--en:c0bba112-->
**$12d^2$ の経験則：**$H_{\text{kv}} = H$ を満たすすべての密なブロックにおいて、マルチヘッドアテンションは **$4d^2$** 個のパラメータ（$\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$）を、FFN は **$8d^2$** 個のパラメータを保持し、1 層あたりの合計は **$12d^2$** 個となる（ブロックあたり $7.08\text{M}$ 個、GPT-2 の $d = 768$ の場合）。

<!--en:f4f3c16f-->
::: {.slide title="概念チェック：Transformer ブロックの設計"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:38a668a3-->
次の 3 つの設計上の問いを考えてみよう。

<!--en:361209a4-->
1. **不公平なアブレーションの罠：**ある研究者が幅 $4d$ の `GELU` FFN（$d = 768$）を `SwiGLU` FFN に置き換えたが、隠れ層の幅を $4d = 3{,}072$ から $\frac{8}{3}d = 2{,}048$ に変更し忘れた。ブロックあたり余分なパラメータを誤って何個追加したか。

<!--en:7a60198c-->
2. **`QK-Norm` のスケーリング：**`QK-Norm` なしのとき、$\mathbf{W}_q$ と $\mathbf{W}_k$ を $\lambda = 8$ 倍にするとアテンションロジットの標準偏差は何倍になるか。`QK-Norm` ありでは何倍か。

<!--en:77c416a9-->
3. **深さスケーリング：**32 ブロックの Pre-Norm モデルにおいて、GPT-2 は $\mathbf{W}_o$ と $\mathbf{W}_2$ の初期標準偏差を何倍にスケーリングするか。

<!--en:557b98b6-->
**解答：**

<!--en:ca381658-->
1. **ブロックあたり $+4d^2 \approx 2.36\text{M}$ 個の余分なパラメータ**（$12d^2 = 7.08\text{M}$ 対 $8d^2 = 4.72\text{M}$ で FFN サイズが $50\%$ 増加）となり、アーキテクチャの質とモデル容量が交絡してしまう

<!--en:fb855a40-->
2. **$\lambda^2 = 64\times$**（`QK-Norm` なし、ソフトマックスがワンホット化）、**$1\times$（不変）**（`QK-Norm` あり）

<!--en:8bcfa971-->
3. **$1 / \sqrt{2L} = 1 / \sqrt{64} = 1/8 = 0.125$** であり、全 $2L = 64$ 個の残差ブランチの分散和を $O(1)$ に保つ

<!--en:94e38d41-->
::: {.slide title="まとめ：Transformer ブロックの構造と安定性"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:315df129-->
- **トークン混合（$4d^2$）とチャネル混合（$8d^2$）：**マルチヘッドアテンションと `SwiGLU` FFN（$d_{\text{ff}} = \frac{8}{3}d$）でブロックあたり合計 $12d^2$ 個のパラメータ

<!--en:ad760aa0-->
- **Pre-Norm と $1/\sqrt{2L}$ 出力スケーリング：**恒等勾配経路を保ちつつ、累積ストリーム分散を $O(1)$ のまま深さ $L$ にわたって維持

<!--en:19742837-->
- **`RMSNorm` と `QK-Norm`：**高速なスケール正規化とコーシー・シュワルツによるロジット上界（$\le \sqrt{d_h}$）
