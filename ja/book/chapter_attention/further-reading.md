<!--en:7902ddfd-->
# 関連文献

<!--en:7ca3a29b-->
ここまでの各節では、層の各構成要素について代表的な設計を 1 つずつ展開してきた。すなわち、ソフトマックスルックアップ（:numref:`sec_queries-keys-values`）、スケール付きドット積によるスコア計算とマスキング（:numref:`sec_attention-scoring-functions`）、マルチヘッドおよびグループ化クエリ自己アテンション（:numref:`sec_multihead-attention`）、位置エンコーディング（:numref:`sec_positional-information`）、タイル化オンラインソフトマックス（:numref:`sec_attention-at-scale`）、スライディングウィンドウアテンションと線形アテンション（:numref:`sec_windowed-linear-attention`）、そして機構的回路（:numref:`sec_what-attention-computes`）である。本節では、これらと密接に関連する派生手法、現代のオープンウェイトモデルにおけるアーキテクチャ上の改良、および一次文献を概観する。

<!--en:f9243964-->
## ルックアップとその正規化

<!--en:eacf96f0-->
### 連想記憶としてのアテンション

<!--en:7ac2ec98-->
:citet:`Ramsauer.Schafl.Lehner.ea.2021` は、ソフトマックスアテンションの 1 ステップが、キーと値の行を記憶パターンとする連続値*現代的ホップフィールドネットワーク*（modern Hopfield network）の 1 回の更新に対応することを示し、ソフトマックスアテンションを古典的な連想記憶（associative memory）と結びつけた。逆温度（カーネル回帰における $1/\sigma^2$、ドット積アテンションにおける $1/\sqrt{d}$）によって、不動点（fixed point）が個々の記憶パターンになるか（高逆温度）、複数のパターンにわたる準安定な平均になるか（低逆温度）が決まる。この視点から同研究は、BERT などの事前学習済みエンコーダにおいて、浅い層は主として広範な平均化レジームで動作し、深い層は鋭い検索レジームで動作することを観察している。適応的カーネルバンド幅 :cite:`Silverman86` は、クエリまたはヘッドに依存するソフトマックス温度に対応する。

<!--en:43012c03-->
### 単調アライメント

<!--en:f1a68df5-->
:citet:`Bahdanau.Cho.Bengio.2014` が機械翻訳向けに制約のない内容ベースのアライメントを導入する 1 年前、:citet:`Graves.2013` は手書き文字合成向けに微分可能な*単調*（monotone）位置ベースのアテンション機構を導入し、平均位置がペンストロークごとに単調に前進する混合ガウス窓をパラメータ化した。これは音声や手書き文字には適した帰納バイアス（inductive bias）であるが（:cite:`rabiner1993fundamentals`）、翻訳における語順の逆転に対しては制約が強すぎる（:numref:`subsec_learned-alignment`）。

<!--en:eeca9527-->
### 総和が 1 である必要のない重み

<!--en:2a2284eb-->
ソフトマックスの各行は和が 1 にならなければならないため、関連するキーを持たないクエリは、確率質量を先頭トークンへと集中させることが多い。これが :numref:`subsec_cross-attention-alignment` および :numref:`sec_what-attention-computes` で見た*アテンションシンク*である（:numref:`subsec_kv-cache-sinks` も参照）。:citet:`Sun.Chen.Kolter.ea.2024` は、アテンションシンクを残差ストリーム内で暗黙的バイアス（implicit bias）項として働く巨大な外れ値の活性化値と関連づけており、一方 :citet:`Barbero.Arroyo.Gu.ea.2025` は、シンクが無操作（no-op）のターゲットを提供することで、深い長文脈 Transformer における表現崩壊（representation collapse）を防いでいると論じている。

<!--en:ce9a76df-->
近年のいくつかの設計では、この総和 1 の制約を緩和し、入力トークンのシンクを使わずにヘッドを待機状態にできるようにしている。

<!--en:09c2859d-->
* **シグモイドアテンション** :cite:`Ramapuram.Danieli.Dhekane.ea.2025` は、行ごとのソフトマックスを $-\log n$ だけシフトした要素ごとのロジスティックシグモイドに置き換え、重みを $1/(n + 1)$ 付近に初期化する。

<!--en:749a28e7-->
* **ゲート付きアテンション**（gated attention、:cite:`Qiu.Wang.Zheng.ea.2025`）は、Qwen3-Next :cite:`Qwen.Team.2025` で採用されており、各ヘッドのソフトマックス出力に入力依存のシグモイドゲートを乗じる。

<!--en:20df9920-->
* **学習可能なシンクロジット**（learnable sink logits）は、`gpt-oss` :cite:`OpenAI.2025` で採用されており、対応する値ベクトルを持たない訓練可能な仮想キースコアをソフトマックスの分母に追加する。

<!--en:8a6f1e90-->
* **差分アテンション**（differential attention、:cite:`Ye.Dong.Xia.ea.2025`）は、ヘッドごとに 2 つのソフトマックス分布の差をとることで、無関係なキーに対するコモンモードの背景重みを相殺する。

<!--en:c599fa90-->
## 自己アテンションとヘッド

<!--en:56792267-->
### 起源とヘッドの冗長性

<!--en:98988fa9-->
自己アテンションは、*系列内アテンション*（intra-attention）を伴う再帰構造 :cite:`Cheng.Dong.Lapata.2016`、自然言語推論（natural language inference）向けの再帰を用いない分解可能アテンション :cite:`Parikh.Tackstrom.Das.ea.2016`、および多様性ペナルティを伴う複数行の文埋め込み :cite:`Lin.Feng.Santos.ea.2017` から発展し、その後に :citet:`Vaswani.Shazeer.Parmar.ea.2017` が Transformer において射影付きマルチヘッド自己アテンションとして統合した。

<!--en:c8e64bf8-->
単一混合のボトルネック（:numref:`subsec_one-mixture-bottleneck`）を克服するには複数のヘッドが不可欠であるが、訓練済みモデルはテスト時に大きなヘッドの冗長性を示すことが多い。実際、:citet:`Michel.Levy.Neubig.2019` および :citet:`Voita.Talbot.Moiseev.ea.2019` は、BERT や NMT モデルにおける大半のヘッドを最小限の性能劣化で事後的に枝刈りでき、位置・構文・低頻度語を担う中核的なヘッドだけが残ることを示した。現代のアーキテクチャでは、グループ化クエリアテンション（grouped-query attention、:numref:`subsec_kv-head-sharing`、:numref:`sec_kv-cache-reduction`）によって、訓練前の段階からパラメータとキャッシュの効率化が図られている。

<!--en:d5ac1b26-->
## 位置

<!--en:1d6cd3d9-->
### 学習される相対位置

<!--en:b7c78f35-->
RoPE の登場以前は、オフセットごとに訓練可能なキー埋め込み :cite:`shaw2018self`、Music Transformer におけるスキューインデックス付き相対ロジット :cite:`huang2018music`、あるいは T5 における対数距離バケットごとのスカラーバイアス :cite:`raffel2020exploring` を通じて相対オフセットが注入されていた。その後、ALiBi は学習されるバケットを固定の線形スロープに置き換え、RoPE はパラメータを持たない座標回転によって、オフセットのみに厳密に依存するスコアを実現した。

<!--en:fe39c899-->
### 訓練済みモデルにおける RoPE

<!--en:ac7d606f-->
オープンウェイトモデルは、より長いコンテキストに対応するため、RoPE の基底周波数 $b$ を段階的に引き上げてきた。具体的には、Llama 2 の $b = 10^4$ から、Llama 2 Long :cite:`Xiong.Liu.Molybog.ea.2024` や Llama 3 :cite:`Grattafiori.Dubey.Jauhri.ea.2024` では $5 \times 10^5$ へ、さらに Gemma 3 :cite:`Gemma.Team.2025` のグローバル層では $10^6$ へと拡大している。$b$ を大きくすると、波長が訓練ウィンドウを超える低速レジームへとより多くの座標ペアが移行する（:numref:`subsec_pos-context-extension`）。:citet:`Barbero.Vitvitskyi.Perivolaropoulos.ea.2025` は、訓練済みモデルが高周波の RoPE ペアを位置ヘッド（直前トークンヘッドなど）に用い、低周波のペアをほぼ位置不変な意味チャネルとして用いていることを示した。これが、極端な距離で意味チャネルがドリフトしないよう最低周波成分を完全にゼロにする *p-RoPE* の動機となっている。YaRN :cite:`Peng.Quesnelle.Fan.ea.2024` を超える事後学習でのコンテキスト拡張として、*LongRoPE* :cite:`Ding.Zhang.Zhang.ea.2024` は次元ごとの再スケーリング係数に対する進化的探索を用い、コンテキストウィンドウを最大 200 万トークンまで拡張している。

<!--en:7bbb09f7-->
### 長さ汎化と長文脈の活用

<!--en:0b71a641-->
:citet:`Chi.Fan.Rudnicky.ea.2023` は ALiBi の有効受容野を解析し、平坦なパープレキシティ外挿が長距離の検索ではなく局所的なウィンドウ処理の反映であるという :numref:`subsec_pos-experiment` の結果を裏づけている。位置エンコーディングを持たない因果モデル（NoPE、:cite:`Haviv.Ram.Press.ea.2022`）については、:citet:`Wang.Ji.Wu.ea.2024` が、系列長とともにアテンションエントロピーが増大すること、またヘッドごとの温度スケーリングによってこれを打ち消せることを示した。この仕組みは Llama 4 :cite:`Meta.2025` の NoPE グローバル層において推論時に採用されている。最後に、長文ドキュメント全体でのパープレキシティが低いからといって、ウィンドウ全体にわたる有効な検索が保証されるわけではない。実際、:citet:`Liu.Lin.Hewitt.ea.2024` は、関連する事実が長いコンテキストの先頭や末尾付近ではなく中央付近に現れると正解率が大幅に低下する*「中間での埋没」*（lost in the middle）現象を実証している（:numref:`sec_lm_long_context` 参照）。

<!--en:a4651e26-->
## コスト

<!--en:b1ef82f1-->
### 初期の線形計算量近似

<!--en:8871cfcf-->
FlashAttention の登場以前は、数多くのアーキテクチャが $n \times n$ のアテンション行列を近似していた。たとえば、*Performer* :cite:`Choromanski.Likhosherstov.Dohan.ea.2021` は正値直交ランダム特徴量を用いて指数カーネルを近似し、*Linformer* :cite:`Wang.Li.Khabsa.ea.2020` は低ランクの系列射影を適用し、*Reformer* :cite:`Kitaev.Kaiser.Levskaya.2020` は局所性鋭敏型ハッシュ（locality-sensitive hashing）を用い、*Longformer* :cite:`beltagy2020longformer` と *Sparse Transformer* :cite:`child2019generating` は局所ウィンドウにストライド付きトークンやグローバルトークンを組み合わせた（:citet:`Tay.Dehghani.Bahri.ea.2020` によるサーベイを参照）。

<!--en:d25e4a5d-->
### 公開モデルにおけるウィンドウと内容適応型スパース性

<!--en:b8f5bee0-->
Mistral 7B :cite:`Jiang.Sablayrolles.Mensch.ea.2023` はすべての層にスライディングウィンドウを適用していたが（後に :cite:`Mistral.AI.2023` で廃止）、後続のアーキテクチャでは局所ウィンドウと周期的なグローバル層を交互に配置している。たとえば、Gemma 2 :cite:`Gemma.Team.2024` では 1:1、Gemma 3（$w = 1{,}024$、:cite:`Gemma.Team.2025`）では 5:1、`gpt-oss` :cite:`OpenAI.2025` では $w = 128$ で 1:1 となっており、Llama 4 :cite:`Meta.2025` ではチャンク化された局所層と NoPE グローバル層を組み合わせている。

<!--en:c64b425b-->
固定の位置ウィンドウにとどまらず、長文脈の訓練とデコーディングに向けて、ハードウェアに適合した*内容ベースのスパースアテンション*（content-based sparse attention、:numref:`sec_lm_sparse_attention`）も登場している。その例として、*Native Sparse Attention*（NSA、:cite:`Yuan.Gao.Dai.ea.2025`）、*Mixture of Block Attention*（MoBA、:cite:`Lu.Jiang.Liu.ea.2025`）、および *DeepSeek Sparse Attention* :cite:`Liu.Mei.Lin.ea.2025` が挙げられる。

<!--en:97c18b6a-->
### FlashAttention-2、FlashAttention-3、FlexAttention

<!--en:dac281b1-->
FlashAttention :cite:`Dao.Fu.Ermon.ea.2022` に続く *FlashAttention-2* :cite:`Dao.2024` は、クエリの系列次元にわたって並列化し、キーループの最後まで正規化を遅延させる。Hopper GPU 上では、*FlashAttention-3* :cite:`Shah.Bikshandi.Zhang.ea.2024` が非同期の Tensor Core 行列積とワープ特化型のソフトマックス指数計算をオーバーラップさせ、インコヒーレントな直交事前回転によって FP8 をサポートする。*FlexAttention* :cite:`Dong.Feng.Guessous.ea.2025` は任意のスコア変更クロージャとブロックマスクを融合 Triton カーネルへとコンパイルし、一方で並列 KV キャッシュデコーディングカーネル（:numref:`sec_lm_decode_kernels`）はキー軸を分割して、オンラインソフトマックスにより部分状態 $(m_B, s_B, \mathbf{o}_B)$ を縮約する。

<!--en:9352878e-->
### ゲート付き再帰とハイブリッド

<!--en:aa39988b-->
現代の線形アテンションの派生手法は、:numref:`chap_modern_rnn` で扱う 2 つの改良によって記憶容量の課題に対処している。すなわち、（1）データ依存の忘却ゲート（*Gated Linear Attention*、:cite:`Yang.Wang.Shen.ea.2024`、:numref:`sec_matrix-state`）と、（2）誤差修正型のデルタ更新（*DeltaNet*、:cite:`Schlag.Irie.Schmidhuber.2021,Yang.Wang.Zhang.ea.2024`、および *Gated DeltaNet*、:cite:`Yang.Kautz.Hatamizadeh.2024`、:numref:`sec_deltanet`）である。実用アーキテクチャでは、厳密な連想想起を維持するために、これらのリカレント層と周期的な完全ソフトマックスアテンション層（通常は 4 層に 1 層から 8 層に 1 層の割合）を組み合わせている :cite:`Qwen.Team.2025,Kimi.Team.2025b,MiniMax.2025`（:numref:`sec_hybrids`）。

<!--en:8aefd574-->
## 大規模モデルにおける回路

<!--en:8e9f08c5-->
### 活性化パッチング、重ね合わせ、説明

<!--en:38cb8d5a-->
ゼロ・平均アブレーション（:numref:`subsec_head-ablation`）の先へ進む手法として、*活性化パッチング*（activation patching）はクリーンな実行の活性化値を破損させた実行へと代入することで因果経路を切り分ける :cite:`Meng.Bau.Andonian.ea.2022,Wang.Variengien.Conmy.ea.2023`。この手続きは ACDC :cite:`Conmy.MavorParker.Lynch.ea.2023` や 1 次の*帰属パッチング*（attribution patching、:cite:`Nanda.2023,Kramar.Lieberum.Shah.ea.2024,Zhang.Nanda.2024,Heimersheim.Nanda.2024`）によって自動化されている。特徴量の数が残差ストリームの次元数を上回る*重ね合わせ*（superposition、:cite:`Elhage.Hume.Olsson.ea.2022`）の状況では、*スパースオートエンコーダ*と帰属グラフが多義的（polysemantic）な方向を単一意味（monosemantic）の特徴量へと分離する :cite:`Bricken.Templeton.Batson.ea.2023,Cunningham.Ewart.Riggs.ea.2024,Ameisen.Lindsey.Pearce.ea.2025`（:numref:`sec_lm_interpretability`）。また、こうした介入的ツールは、アテンション重みが忠実な説明になり得るかをめぐる :citet:`Jain.Wallace.2019` と :citet:`Wiegreffe.Pinter.2019` の古典的な論争にも決着をつけている。

<!--en:04fc81e2-->
### アルゴリズムとしてのコンテキスト内学習

<!--en:6777c59d-->
誘導ヘッド（induction head）によるコピー :cite:`Olsson.Elhage.Nanda.ea.2022` と並び、コンテキスト内回帰の問題族で訓練された Transformer は、勾配降下法、リッジ回帰、および最小二乗法のソルバを順伝播の中で直接実装できる :cite:`Garg.Tsipras.Liang.ea.2022,Oswald.Niklasson.Randazzo.ea.2023,Akyurek.Schuurmans.Andreas.ea.2023`。ただし、一般的なテキスト事前学習がどの程度まで暗黙的な勾配降下法を誘導するかについては、依然として議論が続いている :cite:`Shen.Mishra.Khashabi.2024`。

<!--en:60b0333c-->
## 推奨される参考資料

<!--en:e8dd6d7a-->
* **視覚的な入門資料：**3Blue1Brown の [*Attention in Transformers, Step-by-Step*](https://www.3blue1brown.com/lessons/attention)（2024）、Jay Alammar の [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)（2018）、Peter Bloem の [*Transformers from Scratch*](https://peterbloem.nl/blog/transformers)（2019）、Lilian Weng の [*The Transformer Family Version 2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)（2023）。

<!--en:429e8694-->
* **基礎および回路に関する専門文献：**Elhage らの [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html)（2021）、Olsson らの [*In-Context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)（2022）、Weiss・Goldberg・Yahav の [*Thinking Like Transformers*](https://arxiv.org/abs/2106.06981)（2021）およびこれに対応する Sasha Rush の [*Transformer Puzzles*](https://github.com/srush/Transformer-Puzzles)。

<!--en:8beae255-->
[ディスカッション](https://d2l.discourse.group/t/36852)
