<!--en:7902ddfd-->
# 関連文献

<!--en:73a3f808-->
本章では、現代の長文脈モデルがアテンションと組み合わせるリカレント層を構築し、その起源を LSTM までさかのぼり、固定サイズの状態がどこで容量の限界に達するかを計測してきた。この分野の文献の大半は発表から 5 年未満のものであり、そのすべてが今後も残るとは限らない。本節では、本章で扱わなかった拡張手法や未解決問題とあわせて、参照実装およびチュートリアルを概観する。

<!--en:612fedb7-->
## 再帰とアテンションの組み合わせ

<!--en:1ef2b51a-->
### 実用におけるハイブリッドモデル

<!--en:aa35300a-->
:numref:`sec_hybrids` の演習問題では、ハイブリッドモデルにどれだけのアテンションが必要か、またスタック内のどこに配置すべきかを問うた。3 つ目の問いは、スライディングウィンドウアテンションで大域アテンション（global attention）を代替できるかという点である。条件を揃えた :citet:`Waleffe.Byeon.Riach.ea.2024` の 8B パラメータ比較では、24 個の Mamba-2 層と 4 個のアテンション層（MLP を含む全層の約 8%）を組み合わせ、位置埋め込み（positional embedding）が不要になるよう先頭に Mamba-2 層を置き、アテンション層を等間隔に配置している。この比率において、ハイブリッドモデルは 12 個の標準ベンチマークすべてで Transformer に匹敵するかそれを上回ったのに対し、純粋な Mamba モデルは 5-shot MMLU と電話帳ルックアップで遅れをとった。

<!--en:4989e43e-->
実用モデルにおける具体的な混合比率は多様である。Qwen3-Next :cite:`Qwen.Team.2025` と Kimi Linear :cite:`Kimi.Team.2025b` はデルタ則の漸化式（:numref:`sec_deltanet`）の周囲で 4 層に 1 層の割合でアテンション層を用い、Jamba :cite:`Lieber.Lenz.Bata.ea.2024,Jamba.Team.2024` は 8 層に 1 層を用い、Nemotron-H :cite:`NVIDIA.2025` は Mamba-2 層を維持している。Griffin :cite:`De.Smith.Fernando.ea.2024` は大域アテンションを完全に廃し、2 つのゲート付き対角漸化式層と 1 つのスライディングウィンドウアテンション層（:numref:`sec_windowed-linear-attention`）を交互に配置している。ただし、ウィンドウは直近の $w$ トークンしか保持しないため、コンテキストがウィンドウ幅を超えると厳密なルックアップ性能は劣化する。

<!--en:7c7b05c9-->
### Transformer からの蒸留

<!--en:0d2fecc1-->
リカレントモデルやハイブリッドモデルをゼロから事前学習する代わりに、事前学習済みの Transformer のアテンション射影から行列状態層を初期化し（:numref:`sec_matrix-state`）、蒸留（distillation）によって変換することもできる。:citet:`Wang.Paliotta.May.ea.2024` は一部のアテンション層を残してハイブリッドを構成する一方、:citet:`Bick.Li.Xing.ea.2024` は元の事前学習予算の 1% 未満にあたる 30〜50 億トークンを用いて、アテンションマップ、隠れ状態、および最終出力を段階的に整合させている。

<!--en:c931a33c-->
### 固定状態と完全なキャッシュの間

<!--en:b73440ae-->
固定サイズの状態と完全な KV キャッシュの間を補間する設計が 2 つある。Based :cite:`Arora.Eyuboglu.Zhang.ea.2024` は、$\exp(\mathbf{q}^\top\mathbf{k})$ に対する 2 次テイラー近似と厳密なスライディングウィンドウアテンションを組み合わせ、状態サイズと想起性能を直接トレードオフさせる。対数線形アテンション（log-linear attention、:cite:`Guo.Yang.Goel.ea.2025`）は、過去へさかのぼるにつれて長さが 2 倍ずつ増える 2 進（dyadic）履歴ブロックごとに 1 つの行列状態を保持するため、メモリは $O(\log T)$ で増大し、Mamba-2 や Gated DeltaNet にそのまま適用できる。

<!--en:7cbafdb7-->
## 行列メモリへの書き込み

<!--en:94200efd-->
### 固定状態モデルの表現力

<!--en:2546ce6c-->
系列タスクの中には、状態サイズをどれだけ大きくしても、対角または時不変な線形漸化式では解けないものがある。5 つのカップの下に隠されたボールを一連の入れ替えにわたって追跡する問題、すなわち対称群（symmetric group）$S_5$ における置換の合成を考えよう。:citet:`Merrill.Petty.Sabharwal.2024` は、対角遷移または固定遷移を持つ状態空間モデル（S4 や Mamba を含む）が、Transformer と同様に、定数深さしきい値回路クラス（constant-depth threshold circuit class）$\mathsf{TC}^0$ に属する関数しか計算できないことを証明している。5 要素以上の置換の合成は $\mathsf{NC}^1$（対数深さ回路）完全であり、したがって $\mathsf{TC}^0$ の外にある（ただし $\mathsf{TC}^0 = \mathsf{NC}^1$ でない限り）。（カップが 3 つまたは 4 つの場合は、対称群が可解群（solvable group）となるため $\mathsf{TC}^0$ の内部にとどまる。）

<!--en:f8e23517-->
デルタ則の遷移 $\mathbf{I} - \beta_t \mathbf{k}_t\mathbf{k}_t^\top$ は入力依存かつ非対角であるため、$\mathsf{TC}^0$ の障壁によっては排除されない。しかし :citet:`Grazzi.Siems.Franke.ea.2025` は、有限精度のもとでは、遷移の固有値が $[0, 1]$ 内に限定されるいかなる線形漸化式も、任意の長さにおいてパリティを計算できず、したがって入れ替えも追跡できないことを示している。標準の `GatedDeltaNet` はシグモイドによる $\beta_t \in (0, 1)$ を用いるため、この制約を受ける。$\beta_t \in (0, 2)$ を許容すれば、遷移は負の固有値をとれるようになる。実際、$\beta_t = 2$ かつキーが $(\mathbf{e}_i - \mathbf{e}_j)/\sqrt{2}$ のとき、$\mathbf{I} - 2\mathbf{k}_t\mathbf{k}_t^\top$ は座標 $i$ と $j$ を入れ替える鏡映（reflection）となる。:numref:`subsec_dn-state-tracking` では、この派生モデルをパリティ課題で訓練している。

<!--en:3f6b1545-->
### 一般化デルタ則、指数ゲート、ネットワークメモリ

<!--en:c04474bb-->
- **一般化デルタ則：**RWKV-7 :cite:`Peng.Zhang.Goldstein.ea.2025` は、各座標を固有のレートで減衰させ、あるキーベクトルに沿って消去しつつそれとは異なるキーベクトルに沿って書き込むことでデルタ則のランク 1 射影の共有を解き、任意の正規言語（regular language）を認識可能な非対称遷移を実現している。DeltaProduct :cite:`Siems.Carstensen.Zela.ea.2025` はトークンごとに複数の鏡映 $\prod_i (\mathbf{I} - \beta_i\mathbf{k}_i\mathbf{k}_i^\top)$ を合成し、1 ステップでより高ランクの置換を表現できるようにしている。

<!--en:bb0b5dc5-->
- **指数ゲート：**シグモイド入力ゲートは、単一トークンの書き込み重みを最大 $1$ に制限する。xLSTM :cite:`Beck.Poppel.Spanring.ea.2024,Beck.Poppel.Lippe.ea.2025` の mLSTM は、指数入力ゲートを逐次正規化項の状態および :numref:`sec_attention-at-scale` のオンライン対数最大値（log-max）安定化と組み合わせて用いる。

<!--en:2ccd6ad0-->
- **ネットワークメモリ：**TTT :cite:`Sun.Li.Dalal.ea.2024` と Titans :cite:`Behrouz.Zhong.Mirrokni.2025` は、:numref:`sec_deltanet` におけるオンライン勾配の視点を一般化し、行列 $\mathbf{S}_t$ をトークンごとの勾配ステップで更新される線形層や小規模な MLP に置き換える（Titans ではモメンタムと重み減衰も併用する）。非線形な高速重みが、同じパラメータ数の行列よりも状態の 1 バイトあたり多くの対応関係を想起できるかどうかは、依然として未解決の問いである。

<!--en:4a2e0640-->
## 状態空間モデルと長い畳み込み

<!--en:39f0ce63-->
### 多項式射影による記憶

<!--en:7f7109f7-->
:numref:`sec_ssm` では、S4D のレートを $\lambda_n = -n$ と初期化した。この選択は HiPPO :cite:`Gu.Dao.Ermon.ea.2020` に由来する。HiPPO は、次数 $(N-1)$ の直交多項式による入力履歴 $x(s)$（$s \le t$）の最良近似の係数を保持する手法である。これらの係数は、固定された下三角の HiPPO 行列 $\mathbf{A}$ を持つ線形常微分方程式（ordinary differential equation、ODE）を満たす。S4 :cite:`Gu.Goel.Re.2022` はこの系を時不変にし（指数関数的に時間伸縮された履歴上の Legendre 多項式に対応する :cite:`Gu.Johnson.Timalsina.ea.2023`）、高速なカーネル評価のために $\mathbf{A}$ を正規行列（normal matrix）と低ランク補正の和へと分解した。S4D :cite:`Gu.Gupta.Goel.Re.2022` は、対角成分（複素固有値、または実対角成分 $-1, \ldots, -N$）のみを残すだけで、はるかに単純な仕組みでほぼ同等の性能を回復できることを示した。S5 :cite:`Smith.Warrington.Linderman.2023` はチャネルごとのスカラー SSM を、並列スキャン（:numref:`subsec_parallel-scans`）で評価される単一の多入力・多出力対角 SSM に置き換えており、Gu の博士論文 :cite:`Gu.2023` がこの一連の研究を体系的にまとめている。

<!--en:b85a1f39-->
### Mamba の後継モデル、直接的な線形漸化式、長い畳み込み

<!--en:663c5fb6-->
- **Mamba-2 と Mamba-3：**Mamba-2 :cite:`Dao.Gu.2024` は各ヘッド内でスカラー減衰を共有することで、訓練を :numref:`sec_matrix-state` の行列積に基づくチャンク形式へと帰着させる。Mamba-3 :cite:`Lahoti.Li.Chen.ea.2026` は、台形離散化（trapezoidal discretization）、回転追跡のための複素状態、および自己回帰デコーディング時の演算密度（arithmetic intensity）を高める多入力更新を導入している。

<!--en:41f26e23-->
- **Linear Recurrent Unit（LRU）：**:citet:`Orvieto.Smith.Gu.ea.2023` は、単位円板内の複素固有値と入力正規化 $\sqrt{1 - |\lambda|^2}$ を用いて離散対角漸化式を直接パラメータ化し、Long Range Arena において連続時間 SSM に匹敵する性能を達成するとともに、Griffin で用いられている実数値版 RG-LRU の動機を与えた。

<!--en:f884f3ad-->
- **長い畳み込み：**H3 :cite:`Fu.Dao.Saab.ea.2023` はシフト SSM と乗法的連想 SSM を直列に連結し、Hyena :cite:`Poli.Massaroli.Nguyen.ea.2023` は SSM カーネルを完全に廃し、FFT によって $O(T \log T)$ で評価される陰的にパラメータ化された全長フィルタに置き換えた。一般の長い畳み込み（long convolution）は $O(1)$ デコーディングのためのコンパクトな再帰状態を持たないため、言語モデルでは主にリカレント SSM が選ばれてきたが、ゲノミクス分野では長い畳み込みが依然として広く用いられている。

<!--en:0a293908-->
## 言語を超えて

<!--en:93ad1fb9-->
言語モデリング以外にも、画像パッチ上の双方向スキャンが Vision Mamba :cite:`Zhu.Liao.Zhang.ea.2024` や VMamba :cite:`Liu.Tian.Zhao.ea.2024` を支えており、SaShiMi :cite:`Goel.Gu.Donahue.ea.2022` ではマルチスケールの S4 スタックが生の音声波形をモデル化している。さらに、100 万ヌクレオチド規模のゲノムコンテキストは、HyenaDNA :cite:`Nguyen.Poli.Faizi.ea.2023` では長い畳み込みにより、Caduceus :cite:`Schiff.Kao.Gokaslan.ea.2024` では逆相補鎖同変（reverse-complement-equivariant）な双方向 Mamba 層によってモデル化されている。

<!--en:30616fcb-->
## 参照実装とチュートリアル

<!--en:471ed40c-->
- **実装：**[The Annotated S4](https://srush.github.io/annotated-s4/)（Rush & Karamcheti, 2022）は、完全な S4 を 1 行ずつ構築する。[state-spaces/mamba](https://github.com/state-spaces/mamba)（Gu & Dao）は Mamba および Mamba-2 の `ssd_minimal` の参照実装を提供しており、[mamba-minimal](https://github.com/johnma2006/mamba-minimal) は単一ファイルの PyTorch 版 Mamba を与えている（:numref:`sec_mamba`）。[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) は、GLA、DeltaNet、および Gated DeltaNet 向けの Triton カーネルを提供している（:numref:`sec_matrix-state`、:numref:`sec_deltanet`）。[Gated DeltaNet for Linear Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet)（Raschka, 2025）は Gated DeltaNet ハイブリッドをゼロから構築するチュートリアルを提供しており、[zoology](https://github.com/HazyResearch/zoology) は :numref:`sec_matrix-state` および :numref:`sec_hybrids` で使用した合成連想想起スイートを提供している。

<!--en:46cc938d-->
- **解説とサーベイ：**[On the Tradeoffs of SSMs and Transformers](https://goombalab.github.io/blog/2025/tradeoffs/) および [State Space Duality (Mamba-2), parts I–IV](https://goombalab.github.io/blog/2024/mamba2-part1-model/)（Gu & Dao, 2024〜2025）、[A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)（Grootendorst, 2024）、[DeltaNet Explained, parts I–III](https://sustcsonglin.github.io/blog/2024/deltanet-1/)（Yang, 2024）、ならびに [Test-Time Regression](https://arxiv.org/abs/2501.12352)（Wang, Shi & Fox, 2025）がある。最後の文献は、:numref:`sec_deltanet` の末尾で扱ったメモリの統一的な回帰としての見方をまとめている。

<!--en:3ed01eb7-->
[ディスカッション](https://d2l.discourse.group/t/36786)
