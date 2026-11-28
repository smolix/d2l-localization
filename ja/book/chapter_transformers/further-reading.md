<!--en:7902ddfd-->
# 関連文献

<!--en:66927834-->
本章で構築してきたアーキテクチャは、共通の設計を備えている。すなわち、Pre-Norm の残差ストリーム（:numref:`sec_transformer-block`）、因果または双方向のマルチヘッドアテンション（:numref:`sec_gpt`、:numref:`sec_transformer`）、ヘッド・トークン・チャネルにわたって圧縮されたキー・値キャッシュ（:numref:`sec_kv-cache`）、そして密に（:numref:`sec_scaling-laws`）またはスパースなエキスパートルーティングを通じて（:numref:`sec_moe`）スケールされたフィードフォワード層である。本節では、これらの基盤の上に築かれた重要なアーキテクチャ上のバリエーションと研究の方向性を概観する。

<!--en:caa58cfa-->
## 正規化の配置と残差ストリームの派生形

<!--en:85ee2203-->
Pre-Norm（各サブ層の前に `RMSNorm` を配置する方式）は、出力から最初の層まで減衰のない恒等経路を保つため、現代の大半の Transformer で標準となっている（:numref:`sec_transformer-block`）。しかし、非常に深いモデルや長時間の FP16/BF16 訓練では、正規化されない残差ストリームが深さ方向にわたって大きな活性化値ノルムを蓄積しうるほか、アテンションスコアの大きさもソフトマックスが飽和するまで上方へドリフトしかねない。この問題に対処する改良には 2 つの系統がある（:numref:`fig_norm-taxonomy`）。

<!--en:70d88e79-->
![サブ層 $f$ の前後に正規化を配置する 4 つの方法。Post-LN は残差ストリーム自体を正規化する。Pre-LN はサブ層の入力のみを正規化する。サブ層後正規化（OLMo 2）はストリームへ戻して加算する前にサブ層の出力のみを正規化する。Peri-LN（Sandwich-LN、Gemma 3）はサブ層の入力と出力の両方を正規化する。](../img/mdl-transformers-norm-taxonomy.svg)

<!--en:ee13b226-->
- **サブ層出力の正規化（Sandwich-LN / Peri-LN）と Post-Norm の派生形：**各サブ層の入力と出力の両方を正規化する構成（$x + \operatorname{Norm}(f(\operatorname{Norm}(x)))$）は、恒等スキップ接続（skip connection）を損なわずに各残差更新の大きさを有界に保つものであり、CogView :cite:`Ding.Yang.Hong.ea.2021` や Gemma 2/3 :cite:`Gemma.Team.2024,Gemma.Team.2025` で採用されている。一方、OLMo 2 :cite:`OLMo.2025` は RMSNorm を各サブ層の出力側へ移し（$x + \operatorname{Norm}(f(x))$）、これをクエリ・キー正規化（query–key normalization、QK-Norm）と組み合わせることで、加算後の正規化によって恒等経路を遮ることなく残差ストリームの増大を有界に抑えている。

<!--en:98fd5eb8-->
- **クエリ・キー正規化（QK-Norm）：**ヘッドごとにクエリとキーへ RMSNorm を適用してから $QK^\top / \sqrt{d_h}$ を計算すると（:numref:`subsec_qk-norm`）、残差ストリームの増大にかかわらず、コーシー・シュワルツの不等式によってソフトマックス前のロジットが有界に保たれる :cite:`Henry.Dachapally.Pawar.ea.2020,Dehghani.Djolonga.Mustafa.ea.2023`。この手法は ViT-22B、Gemma 3、Qwen3、OLMo 2 で標準となっている。

<!--en:e0c720ff-->
- **深さスケーリング付き Post-Norm とロジットのソフトキャッピング：**DeepNorm :cite:`Wang.Ma.Dong.ea.2024` は Post-LN ブロックを維持しつつ、明示的な深さ依存の定数によってスキップ経路の重みを大きくし（$\operatorname{Norm}(\alpha x + f(x))$、ただし $\alpha = (2L)^{1/4}$）、同時にブランチの重みの初期化を $\beta = (8L)^{-1/4}$ 倍に縮小する。この両方によってストリームに対する各更新が相対的に小さくなり、Post-LN Transformer を 1,000 層まで安定化できる。別の方法として、Gemma 2 は滑らかな tanh ソフトキャッピング（$c \tanh(z / c)$）によってアテンションロジットと最終ロジットを有界に抑えているが、より新しいモデルではアテンションロジットのソフトキャッピングはおおむね QK-Norm に取って代わられている。

<!--en:9ff7da3e-->
## フィードフォワードネットワークと活性化関数

<!--en:d954cc7d-->
元の Transformer は、中間幅 $d_{\textrm{ff}} = 4d$ をもつ 2 行列の ReLU フィードフォワードネットワークを用いていた :cite:`Vaswani.Shazeer.Parmar.ea.2017`。GPT-2 と BERT は ReLU を GELU に置き換えた :cite:`Hendrycks.Gimpel.2016`。:citet:`Shazeer.2020` 以降、現代のほぼすべての言語モデルはゲート付き線形ユニット（gated linear unit）:cite:`Dauphin.Fan.Auli.ea.2017`、すなわち **SwiGLU**（SiLU ゲート）または **GeGLU**（GELU ゲート）を採用しており、中間幅 $d_{\textrm{ff}} \approx \frac{8}{3}d$ をハードウェアに適した 128 または 256 の倍数に丸めることで、3 つの射影行列（$W_{\textrm{gate}}, W_{\textrm{up}}, W_{\textrm{down}}$）を $8d^2$ のパラメータ数・FLOP 予算（ゲートなしの幅 $4d$ の FFN と同量）に一致させている（:numref:`sec_transformer-block`）。極端に大きな幅では、**並列 Transformer ブロック**（PaLM :cite:`chowdhery2022palm`、ViT-22B :cite:`Dehghani.Djolonga.Mustafa.ea.2023`）が同じ正規化済み入力からアテンションと FFN を並列に計算するため（$x + \operatorname{Attn}(\operatorname{Norm}(x)) + \operatorname{FFN}(\operatorname{Norm}(x))$）、アテンションと FFN の入力射影を単一の大規模な行列積へと融合できる。

<!--en:0ca03e62-->
## 長文脈推論とキャッシュ圧縮

<!--en:f2465935-->
デコーディングはメモリ帯域幅律速であるため（:numref:`subsec_kv-cache-prefill-decode`）、キー・値キャッシュの通信量を削減することが効率的なサービングの中心課題となる。

<!--en:bfef77ed-->
- **グループ化クエリアテンションとマルチヘッド潜在アテンション：**マルチクエリアテンション（MQA）:cite:`Shazeer.2019` とグループ化クエリアテンション（grouped-query attention、GQA）:cite:`Ainslie.Lee-Thorp.Jong.ea.2023` は、クエリヘッドのグループ間でキーヘッドと値ヘッドを共有する（:numref:`sec_kv-cache-reduction`）。DeepSeek-V2 :cite:`DeepSeek-AI.2024` で導入され DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` でも引き継がれた**マルチヘッド潜在アテンション**（multi-head latent attention、MLA）は、低ランク KV 圧縮（:numref:`subsec_kv-cache-low-rank`）を一般化した手法である。キーと値を結合的に単一の低次元潜在ベクトル $c_t^{\textrm{KV}} \in \mathbb{R}^{d_c}$（$d_c \ll H d_h$）へ射影してメモリにキャッシュする一方、切り離された低次元の RoPE キーベクトルと対にすることで、デコーディング時の重み吸収（weight absorption）と回転位置エンコーディングの両立を保っている。

<!--en:0dafd3cd-->
- **スライディングウィンドウ、アテンションシンク、ハイブリッドスタック：**直近トークンの移動ウィンドウとあわせて先頭のプロンプトトークンを保持する手法（**StreamingLLM** :cite:`Xiao.Tian.Chen.ea.2024`、:numref:`subsec_kv-cache-sinks`）は、ソフトマックスの分母が余剰の確率質量を先頭の「シンク」トークンで吸収するため、ストリーミング推論中も生成品質を維持できる。近年のいくつかのアーキテクチャでは、局所的なスライディングウィンドウアテンション層と完全な大域的因果アテンション層を交互に配置したり（Gemma 3 では大域層 1 層につき局所層 5 層、gpt-oss では 1 層おき）、アテンション層と線形時間の状態空間層・リカレント層を交互に配置したりしている（:numref:`chap_modern_rnn`）。

<!--en:776959da-->
- **投機的デコーディング：**候補となる $m$ 個のトークンを並列に検証する処理は、単一トークンをデコードする場合とほぼ同じメモリ帯域幅コストで実行できる（:numref:`subsec_kv-cache-prefill-decode`）。そのため、**投機的デコーディング**（speculative decoding）:cite:`Leviathan.Kalman.Matias.2023,Chen.Borgeaud.Irving.ea.2023` は高速なドラフトモデル（またはターゲットモデルに付加した追加のマルチトークン予測ヘッド）を用いて $m$ 個のトークンを提案し、ターゲット分布を厳密に保つ棄却サンプリング（rejection sampling）規則によって、ターゲットモデルの 1 回の順伝播でそれらを検証する。

<!--en:777177e0-->
## エンコーダ、クロスアテンション、ビジョンバックボーン

<!--en:cdf6bdea-->
- **双方向エンコーダと検索：**オープンエンドな生成では因果デコーダが主流を占める一方、双方向エンコーダ（:numref:`sec_transformer`）は依然として密なテキスト埋め込み、リランキング、およびトークンレベルの分類における主力である。RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` と DeBERTa :cite:`He.Liu.Gao.ea.2021` は BERT のマスク言語モデリング（masked language modeling）による事前学習レシピを洗練させ、ModernBERT :cite:`Warner.Chaffin.Clavie.ea.2024` は現代的な Pre-Norm、RoPE、SwiGLU、およびパディングなしの FlashAttention スタックを双方向エンコーダへと導入している。

<!--en:b9998588-->
- **Vision Transformer と階層型ハイブリッド：**中程度のデータ規模で ViT を効果的に訓練するには、強力なデータ拡張と正則化（DeiT :cite:`touvron2021training`）、自己教師ありのマスクパッチ予測（MAE :cite:`he2022masked`）または自己蒸留（self-distillation、DINOv2 :cite:`Oquab.Darcet.Moutakanni.ea.2024`）、および 2D 回転位置埋め込み（rotary position embedding）が鍵となる。マルチスケールの特徴マップを必要とする密な予測タスク（検出、セグメンテーション）に対しては、Swin Transformer :cite:`liu2021swin` が階層型ピラミッドにわたってシフトされた局所 2D ウィンドウ内に自己アテンションを制限する一方、ConvNeXt :cite:`liu2022convnet` は Transformer ブロックの設計上の選択を純粋なデプスワイズ分離可能畳み込みネットワークへと還元している。

<!--en:66531f99-->
## スケーリング則と混合エキスパートの設計

<!--en:cfbf9027-->
- **計算量最適スケーリングと推論考慮型スケーリング：**:citet:`kaplan2020scaling` は、言語モデルの交差エントロピー損失がモデルサイズ $N$、データセットサイズ $D$、および計算量 $C$ に対してべき乗則に従ってスケールすることを初めて確立した。続いて :citet:`hoffmann2022training`（Chinchilla）は IsoFLOP 曲線を用い、一定の事前学習計算量のもとで訓練損失が最小化されるのは、$N$ と $D$ を同じ比率でスケールさせたとき（$D \approx 20 N$）であることを示した。現代のオープンウェイトモデル（15 兆トークンで訓練された Llama 3 8B や Qwen3 など）は、小さなモデルを Chinchilla 比（:numref:`sec_scaling-laws`）をはるかに超えてあえて過剰に訓練することで、運用期間全体の総推論コストに対して訓練計算量を償却している :cite:`Sardana.Portes.Doubov.ea.2024`。

<!--en:bca33215-->
- **スパース混合エキスパート：**Top-$K$ ノイジーゲーティング :cite:`Shazeer.Mirhoseini.Maziarz.ea.2017`、GShard :cite:`Lepikhin.Lee.Xu.ea.2021`、および Switch Transformer :cite:`fedus2022switch` を基盤とする現代の MoE モデルは、主に粒度において異なっている（:numref:`sec_moe`）。Mixtral 8x7B :cite:`Jiang.Sablayrolles.Roux.ea.2024` が全幅のエキスパート 8 個のうち 2 個へ各トークンをルーティングするのに対し、Qwen3 :cite:`Yang.Li.Yang.ea.2025` や DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` は 100 個を超える幅の狭いエキスパートのうち 8 個を選択する。さらに DeepSeek-V3 は常時活性な共有エキスパート :cite:`Dai.Deng.Zhao.ea.2024` を追加し、ごく小さな補助ペナルティだけを残して、主に動的なエキスパートごとのルータバイアス :cite:`Wang.Gao.Zeng.ea.2024` によって負荷を分散させている。また、**スパースアップサイクリング**（sparse upcycling）:cite:`Komatsuzaki.Puigcerver.Lee-Thorp.ea.2023` によって密なチェックポイントを MoE モデルへ変換することも可能であり、訓練済みの密な FFN から各エキスパートを初期化したうえで事前学習を継続する。

<!--en:1a8fd496-->
[ディスカッション](https://d2l.discourse.group/t/36853)
