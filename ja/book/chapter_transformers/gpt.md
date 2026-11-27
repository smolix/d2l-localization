<!--en:e6e75e3c-->
# デコーダのみの言語モデル（GPT）

<!--en:944e72b3-->
単一の `TransformerBlock`（:numref:`sec_transformer-block`）は、$d$ 次元のトークン表現の系列を同じ形状 $(B, T, d)$ の系列へと写像する。因果的な Transformer ブロックのスタックを自己回帰（autoregressive）型の **Generative Pre-trained Transformer (GPT)** :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language` に仕立てるには、$L$ 個のブロックを 2 つの境界層で挟み込む（:numref:`fig_gpt-stack`）。

<!--en:262e4135-->
1. **入力埋め込み：**トークン埋め込みテーブル $\mathbf{E} \in \mathbb{R}^{V \times d}$ は、離散トークン ID $x_1, \dots, x_T \in \{0, \dots, V-1\}$ を $\mathbb{R}^d$ のベクトルへと写像する。なお、アテンション内部で回転位置埋め込み（rotary position embedding）を使用しない場合は、オプションで学習可能な絶対位置埋め込み（absolute position embedding）$\mathbf{P} \in \mathbb{R}^{T_{\max} \times d}$ を加算する。

<!--en:d5a9b5f5-->
2. **最終正規化と言語モデリングヘッド：**$L$ 個の因果的な `TransformerBlock` 層を経た後、最終正規化層が残差ストリーム $\mathbf{H}^{(L)}$ を安定化し、線形ヘッドが各位置 $t$ を次トークンのロジット $\mathbf{Z} = \textrm{Norm}(\mathbf{H}^{(L)})\mathbf{E}^\top \in \mathbb{R}^{B \times T \times V}$ へと射影する。このとき、出力射影の重みは入力埋め込みテーブル $\mathbf{E}$ と**重み共有**（weight tying）する :cite:`Press.Wolf.2017,Inan.Khosravi.Socher.2017`。

<!--en:81ff1f71-->
## ブロックから言語モデルへ

<!--en:fd61d568-->
![下から上へと見た GPT の構造。テーブル $\mathbf{E}$ がトークン ID を残差ストリームに埋め込み、$L$ 個の因果的な `TransformerBlock` 層がストリームから読み出して書き込み、最終正規化と転置テーブル $\mathbf{E}^\top$ が各位置のベクトルを語彙上のロジットへと変換する。ここに示す現代的な構成では各アテンション層の内部で RoPE によりクエリとキーを回転させるが、GPT-2 では代わりに最下部で学習された位置ベクトルを一度だけ加算する。](../img/mdl-transformers-gpt-stack.svg)

<!--en:b237ddab-->
各ブロックの内部では、`CausalAttention` がクエリ・キー・値の射影を融合し、**グループ化クエリアテンション**（grouped-query attention、`num_kv_heads` $\le$ `num_heads`、:numref:`sec_multihead-attention`）をサポートする。さらに、オプションで **RoPE**（`rope=True`、:numref:`sec_positional-information`）によってクエリとキーを回転させ、位置 $t$ が位置 $s \le t$ のみにアテンションを向けるよう因果マスクを適用する。

<!--en:e7d6e73a-->
ここで定義する `GPT` クラスは、設定可能な `num_blks` 個の `TransformerBlock` モジュール（デフォルトでは `norm='rms'`、`act='swiglu'`、`pos='rope'`、`pre_norm=True` であり、GPT-2 の `norm='layer'`、`act='gelu'`、`pos='learned'`、`bias=True` もサポートする）を積み重ね、出力射影をトークン埋め込みテーブル `token_emb` と重み共有する。

<!--en:84d78175-->
GPT-2 :cite:`Radford.Wu.Child.ea.2019` および :numref:`subsec_signal-propagation` に従い、`init_weights` は線形層と埋め込み層の重みを $\mathcal{N}(0, 0.02^2)$ からサンプリングしつつ、各ブロックの 2 つの残差出力射影（`attention.W_o` と `ffn.W_2`）を $1 / \sqrt{2L}$ 倍にスケーリングする。

<!--en:fb50dd93-->
## WikiText-103 におけるバイトレベル GPT の訓練

<!--en:9cc86c11-->
ゼロからの実験を自己完結させ、トークナイザに起因するアーティファクトを排除するため、**WikiText-103** :cite:`Merity.Xiong.Bradbury.ea.2016` の生の UTF-8 バイト列（$V = 256$）上で直接訓練する。ここでは、ウィンドウ数 $B = 64$、長さ $T = 128$ バイトのミニバッチ（ステップあたり $8{,}192$ トークン）を用いる。

<!--en:814b7e8f-->
また、補助コンストラクタ `gpt(...)` も定義しておく。これは新しく初期化された、すぐに訓練可能な `GPT` を構築する関数である。

<!--en:866ed5c7-->
`gpt` はモデルを GPU に転送して `torch.compile` でコンパイルし、fp32 の行列積で tf32 テンソルコアを使用できるようにする。

<!--en:5817e1c3-->
`gpt` はモデルごとに `d2l.get_key()` から新しいランダムな重みをサンプリングする。

<!--en:733fa8be-->
### 自己回帰の目的関数

<!--en:95a106fa-->
入力ウィンドウ $\mathbf{x} = (x_1, \dots, x_T)$ とシフトされたターゲットウィンドウ $\mathbf{y} = (x_2, \dots, x_{T+1})$ が与えられたとき、因果マスクにより、出力ロジットの第 $t$ 行 $\mathbf{z}_t \in \mathbb{R}^V$ は $(x_1, \dots, x_t)$ のみに依存することが保証される。訓練では、全 $T$ 個の位置にわたる平均交差エントロピー損失を並列に最小化する。

<!--en:c6c365dc-->
初期化の段階（学習が一切進む前）では、モデルはすべてのバイトにほぼ一様な確率 $1/V$ を割り当てるため、初期の交差エントロピー損失は $\ln V = \ln 256 \approx 5.55$ nat/バイトに近い値となる。まず、シフトされた入力・ターゲットのウィンドウ対 `(X[0], Y[0])` を 1 つと、$5 \times 5$ の下三角因果マスクを確認してみよう。

<!--en:9498ccb7-->
新しく初期化した 6 ブロックの `GPT`（$\approx 4.79\textrm{M}$ パラメータ）を検証ミニバッチ上で評価すると、未訓練時の交差エントロピーが $\ln 256 \approx 5.55$ nat/バイトに近いことが確認できる。

<!--en:ce741cf0-->
### 訓練レシピ（`LMOptimizer`）

<!--en:e2373fd9-->
現代の LLM の事前学習では、**AdamW**（:numref:`sec_adam`、標準的なモメンタムパラメータとしてデフォルトでは $\beta_1 = 0.9, \beta_2 = 0.999$、大バッチでの実行では $\beta_2 = 0.95$ を使用）に、**選択的重み減衰**（埋め込み、バイアス、1D 正規化スケールを対象外とし、2D 線形射影行列にのみ重み減衰 $\lambda = 0.1$ を適用）、**線形ウォームアップとそれに続くコサイン学習率減衰**（:numref:`sec_scheduler`）、および**大域的勾配ノルムクリッピング**（$\|\nabla \mathcal{L}\|_2 \le 1.0$）を組み合わせる。

<!--en:eaaa29d2-->
6 ブロックのバイトレベル `GPT`（$d = 256, H = 8, L = 6$、$\approx 4.8\textrm{M}$ パラメータ）を 1,000 ステップ（$\approx 8.2\textrm{M}$ バイト）訓練すると、検証交差エントロピーは英語 Wikipedia テキストの経験ユニグラムエントロピー（$3.17$ nat/バイト）と経験バイグラムエントロピー（$2.45$ nat/バイト）の両方を大きく下回る。

<!--en:43e6d2c9-->
参考として、訓練コーパスの先頭 $10\textrm{M}$ バイトから経験ユニグラムおよびバイグラムのバイトエントロピーを直接計算し、6 ブロックの Transformer が局所的な文字統計からどれほど先へ進んでいるかを確かめる。

<!--en:37066329-->
## 統制された比較

<!--en:4f14bb99-->
### 実行間のばらつきと FFN 活性化関数（`SwiGLU` と `GELU`）

<!--en:f68c6a84-->
純粋なアーキテクチャ上の改善を乱数シードのノイズと切り分けるため、デフォルトの `SwiGLU` モデルを独立な 3 つのシードで訓練し、さらにパラメータ数を揃えた `GELU` モデル（`act='gelu'`）も 3 つのシードで訓練する。

<!--en:34a102dc-->
乱数シード間の実行ごとのばらつきはわずか $\sim 0.01$ nat/バイトにすぎないのに対し、`SwiGLU` は同じパラメータ数および FLOP 予算のもとで `GELU` を一貫して $\sim 0.06$ nat/バイト上回る。

<!--en:1f5e6636-->
この `train_lm` を使えば、:numref:`sec_transformer-block` におけるもう 1 つの設計上の選択である正規化の配置も検証できる。初期化の時点では、:numref:`subsec_signal-propagation` で見たように、`init_weights` が用いる重みスケール $0.02$ において Post-Norm に問題は生じなかった。トークン同士は離れたままであり、クエリの勾配も消失せずに残る。しかし、Post-Norm が同様にうまく*訓練できる*かどうかは別の問題である。そこで、同じモデルを Post-Norm ブロック（`pre_norm=False`）にして、ウォームアップを 500 ステップにした場合、本実験と同じ 100 ステップにした場合、およびウォームアップなしにした場合の 3 通りで訓練する。ピーク学習率はいずれの 3 通りでも本実験と同じである。

<!--en:538ce59d-->
その答えはウォームアップに依存する。Pre-Norm モデルを約 $1.34$ nat/バイトまで訓練できる 100 ステップのウォームアップでは、Post-Norm モデルは $3.17$ で止まり、ウォームアップなしの実行でも同じ結果になる。この数値はすでに上で見たものである。すなわち先ほど計算したユニグラムエントロピーであり、これら 2 つのモデルは各バイトがどれくらいの頻度で出現するかだけを学習し、直前にあるバイト列については何も学習していない。原因は初期の更新が大きすぎることにある。学習率がわずか数ステップのうちにピークへ達すると、アテンションと FFN の各サブ層の出力が、加算先の正規化された残差ストリームよりもはるかに大きく成長してしまう。その結果、正規化を経た後にはすべてのトークンがほぼ同一の最終ベクトルに行き着き、どの位置でも同じベクトルしか得られないため、バイトの出現頻度しか予測できなくなる。トークン同士を再び引き離すはずの勾配もトークンの広がりとともに縮小してしまうため（:numref:`subsec_signal-propagation`）、訓練が回復することはない。

<!--en:eae6523d-->
ウォームアップを 500 ステップにすれば、サブ層の出力はストリームより小さいまま保たれ、Post-Norm モデルも Pre-Norm モデルにわずかに及ばないものの約 $1.38$ nat/バイトまで訓練できる。したがって、Pre-Norm には十分なウォームアップであっても Post-Norm には不十分なのである。その境界がどこにあるかは一概には言えない。境界は明確ではなく、中間のウォームアップ長では訓練が進む実行もあれば崩壊する実行もある。これは :citet:`xiong2020layer` が元の Transformer について報告した挙動そのものである。すなわち、Post-Norm は長いウォームアップを必要とするのに対し、Pre-Norm はウォームアップなしでも訓練でき、より速く収束する。

<!--en:64302a93-->
### 条件を揃えたリカレントベースラインとの比較

<!--en:b3e05a88-->
**同じパラメータ数**（$d_{\textrm{rnn}} = 1920$、$\approx 4.7\textrm{M}$ パラメータ）と訓練ステップ数を与えたとき、6 ブロックの `GPT` はリカレント言語モデル（`d2l.RNNLM`、:numref:`sec_rnn-scratch`）と比べてどうなるだろうか。RNN の学習率を $\{10^{-3}, 2\times 10^{-3}, 4\times 10^{-3}\}$ にわたってスイープし、ウィンドウ内の位置 $t \in \{0, \dots, 127\}$ の関数として検証損失を比較する（`loss_by_position`）。

<!--en:ce65c29b-->
各候補のピーク学習率 $\{10^{-3}, 2\times 10^{-3}, 4\times 10^{-3}\}$ ごとに `RNNLM` を 2 つのシードで訓練し、最良のリカレントモデルの実行結果をウィンドウ位置 $t \in \{0, 1, 4, 16, 64, 127\}$ にわたって `GPT` と比較する。

<!--en:ff339a19-->
位置 $t = 0$（文脈として 1 バイトしか利用できない位置）では、`GPT` と `RNNLM` はほぼ同じ損失から出発する。しかし、利用可能なプレフィックスの文脈が増えるにつれて（$t \to 127$）、Transformer の検証損失は RNN よりも急激に低下する。これは、自己アテンションが任意の先行バイトを $O(1)$ ホップで直接読み取り、過去の履歴を $t$ 回の逐次的なリカレントステップを通して圧縮する必要がないからである。

<!--en:dd17b6a9-->
## モデルからのサンプリング

<!--en:0adf0f45-->
自己回帰生成（`GPT.generate`）では、一度に 1 トークンずつ末尾に追加していく。各ステップで現在のプレフィックスをモデルに入力し、最終位置のロジット $\mathbf{z}_t$ を**温度**（temperature）$\tau > 0$ で割り、オプションで上位の **top-$k$** 個を除くすべてのロジットを $-\infty$ にマスクしたうえで :cite:`Fan.Lewis.Dauphin.2018`、$\textrm{softmax}(\mathbf{z}_t / \tau)$ から次のトークンをサンプリングする。なお、極限 $\tau \to 0$ に対応する貪欲デコーディング（greedy decoding）は、`top_k=1` の場合にあたる。

<!--en:1411ac05-->
`'The history of the '` に続く 120 バイトの系列を 3 通りの `(temperature, top_k)` 設定でサンプリングすると、低い温度と top-$k$ フィルタリングがいかに多様性と引き換えに局所的な一貫性を高めるかがわかる。

<!--en:55654c67-->
## 事前学習済み GPT-2（`124M`）の読み込み

<!--en:f0f02b4c-->
本節の `GPT` クラスは `norm='layer'`、`act='gelu'`、`pos='learned'`、および `bias=True` をサポートしているため、OpenAI の 124M パラメータの **GPT-2 Small** アーキテクチャ（$V = 50{,}257, d = 768, H = 12, L = 12, T_{\max} = 1024$）をそのままインスタンス化し、公式の `model.safetensors` の重みとバイトレベル BPE トークナイザ（`gpt2_tokenizer`）を直接読み込める。

<!--en:7b9fd19b-->
:OpenAI の GPT-2 `safetensors` チェックポイントのキーと本節の `GPT` モジュールの属性との対応（`Conv1D` の重みは形状 `(in, out)` をもつ）。

<!--en:c5fb60ea-->
| チェックポイントのテンソル | `GPT` の属性 | 備考 |

<!--en:1ca501b3-->
| `wte.weight` | `token_emb` | 出力ヘッドと共有（`V x d`） |

<!--en:eb6ceffb-->
| `wpe.weight` | `pos_emb` | 1,024 個の学習済み位置ベクトル |

<!--en:313df5ab-->
| `h.i.ln_1`, `h.i.ln_2` | `blks[i].norm1`, `blks[i].norm2` | アテンション前および FFN 前の LayerNorm |

<!--en:e3970ec6-->
| `h.i.attn.c_attn` | `blks[i].attention.W_qkv` | 融合 QKV（`Conv1D`） |

<!--en:69291e63-->
| `h.i.attn.c_proj` | `blks[i].attention.W_o` | 出力射影（`Conv1D`） |

<!--en:003065c0-->
| `h.i.mlp.c_fc`, `h.i.mlp.c_proj` | `blks[i].ffn.W_1`, `blks[i].ffn.W_2` | 2 層 GELU MLP（`Conv1D`） |

<!--en:98bf1e30-->
| `ln_f` | `norm` | 出力ヘッド前の最終 LayerNorm |

<!--en:4824c297-->
`nn.Linear` は `(out, in)` の形状で格納するため、`load_gpt2` はすべての `Conv1D` の重みを転置する。

<!--en:8859f6c7-->
`nnx.Linear` のカーネルは `(in, out)` すなわち `Conv1D` と同じ配置であるため、`load_gpt2` は重みをそのままコピーする。

<!--en:b438f4fa-->
読み込んだチェックポイントを検証するため、『The Time Machine』の冒頭段落における交差エントロピー（$\approx 3.9$ nat/トークン、すなわちパープレキシティ $\approx 49$）を計算し、さらに続きのテキストを生成する。

<!--en:1c822607-->
事前学習済みの重みが検証できたところで、`gpt2.generate` にプロンプトを与え、貪欲デコーディング（`top_k=1`）によって決定論的な続きを生成してみよう。

<!--en:b394b9f9-->
確率的な top-$k$ サンプリング（`temperature=0.8, top_k=50`）に切り替えて 40 個のサブワードトークンを生成すると、流暢で多様な続きが得られる。

<!--en:02e337a2-->
## 現代的なレシピ

<!--en:2b297386-->
2019 年の `load_gpt2()` の構成を本節のデフォルトの `GPT` と比較すると、オープンウェイトである現代のデコーダのみの言語モデルを特徴づける 4 つの主要な改良点が浮かび上がる（:numref:`tab_modern-recipe`）。

<!--en:1a5e47de-->
1. **正規化：**バイアスなしの Pre-Norm `RMSNorm`（:numref:`subsec_rmsnorm`）を `LayerNorm` の代わりに用い、多くの場合ヘッドごとの **QK-Norm**（:numref:`subsec_qk-norm`）やサブ層出力の正規化（OLMo 2、Gemma 3）と組み合わせる。

<!--en:db5a8885-->
2. **位置表現：**各アテンション層の内部で**回転位置埋め込み**（RoPE、:numref:`sec_positional-information`）を、固定長の学習済み位置テーブル `wpe` の代わりに用いる。

<!--en:14c0d153-->
3. **フィードフォワードネットワーク：**バイアスなしのゲート付き **`SwiGLU`**（または `GeGLU`）を、2 層の `GELU` MLP の代わりに用いる。

<!--en:13a78e81-->
4. **推論キャッシュとモデル容量のスケーリング：**KV キャッシュを縮小する（:numref:`sec_kv-cache`）ための**グループ化クエリアテンション（GQA）**、**スライディングウィンドウアテンション**、または**低ランク潜在圧縮**（low-rank latent compression、MLA）と、トークンあたりの FLOP 数を増やさずに総パラメータ数を拡大する**混合エキスパート**（mixture of experts、MoE）層（:numref:`sec_moe`）。

<!--en:6e007488-->
:2023〜2025 年の技術報告書で報告された 7 つのオープンウェイトモデル族のアーキテクチャ。アテンションとキャッシュ：:numref:`sec_kv-cache-reduction`。正規化：:numref:`sec_transformer-block`。位置表現：:numref:`sec_positional-information`。混合エキスパート：:numref:`sec_moe`。

<!--en:ccbb7ecb-->
| モデル | アテンションとキャッシュ | 正規化 | 位置表現 | FFN | 密または MoE |

<!--en:bd862735-->
| Mistral 7B（2023）:cite:`Jiang.Sablayrolles.Mensch.ea.2023` | GQA $32{:}8$、スライディングウィンドウ 4096 | RMSNorm、Pre-Norm | RoPE、ベース $10^4$ | SwiGLU | 密 |

<!--en:da1f1d7c-->
| Llama 3（2024） | GQA $32{:}8$（8B）、$64{:}8$（70B）、$128{:}8$（405B） | RMSNorm、Pre-Norm | RoPE、ベース $5 \times 10^5$ | SwiGLU | 密 |

<!--en:dcc4d5fa-->
| Qwen3（2025）:cite:`Yang.Li.Yang.ea.2025` | GQA：$64{:}8$ 密（32B）、$64{:}4$ MoE（235B-A22B） | RMSNorm、Pre-Norm、QK-Norm | RoPE、ベース $10^6$、YaRN | SwiGLU | 密、および MoE：128 エキスパート中 8 個活性 |

<!--en:b5734b4e-->
| OLMo 2/3（2024/25）:cite:`OLMo.2025,OLMo3.2025` | MHA（7B）、GQA（32B）、OLMo 3：4 層中 3 層でウィンドウ 4096 | サブ層出力に RMSNorm、QK-Norm | RoPE | SwiGLU | 密 |

<!--en:b51c73e1-->
| DeepSeek-V3（2024）:cite:`Liu.Feng.Xue.ea.2024` | MLA：KV を 512 次元の潜在表現に圧縮 | RMSNorm、Pre-Norm | 分離型 RoPE（64 次元の共有キー） | SwiGLU | MoE：256 エキスパート + 共有 1、8 個活性 |

<!--en:d498fb36-->
| Gemma 3（2025）:cite:`Gemma.Team.2025` | GQA、局所対大域 $5{:}1$、ウィンドウ 1024 | RMSNorm（Pre・Post 両方）、QK-Norm | RoPE、ベースは大域 $10^6$、局所 $10^4$ | GeGLU | 密 |

<!--en:d9ddcc60-->
| gpt-oss（2025）:cite:`OpenAI.2025` | GQA $64{:}8$、交互の層でウィンドウ 128、学習型シンク | RMSNorm、Pre-Norm | RoPE、YaRN | クランプ付き SwiGLU | MoE：128 または 32 エキスパート、4 個活性 |

<!--en:30ac03ff-->
## まとめ

<!--en:db030636-->
**デコーダのみの GPT** は、$L$ 個の因果的な `TransformerBlock` 層をトークン埋め込みテーブル $\mathbf{E} \in \mathbb{R}^{V \times d}$ と最終正規化および重み共有された出力射影 $\mathbf{E}^\top$ の間に配置し、全 $T$ 個の系列位置における次トークンのロジットを 1 回の並列順伝播で予測する（:eqref:`eq_gpt-loss`）。**AdamW**（2D 行列にのみ選択的に重み減衰を適用）、線形ウォームアップ、コサイン減衰、および勾配クリッピング（`LMOptimizer`）を用いて訓練すると、4.8M パラメータのバイトレベル `GPT` は 1,000 ステップ以内に英語テキストのバイグラムエントロピーを大きく下回る。パラメータ数と計算量を揃えた統制された比較では、`SwiGLU` は `GELU` を約 $0.06$ nat/バイト上回り（シード間のばらつきは $0.01$ nat）、また `GPT` は条件を揃えた `RNNLM` に対して文脈位置 $t$ が深くなるほど着実に差を広げていく。現代のオープンウェイト LLM（:numref:`tab_modern-recipe`）はいずれもこの Pre-Norm の `RMSNorm` + `RoPE` + `SwiGLU`・`GeGLU` というレシピを共有しており、主な違いは KV キャッシュをどのように圧縮するか（:numref:`sec_kv-cache`）、そして密な FFN を混合エキスパートに置き換えるかどうか（:numref:`sec_moe`）にある。

<!--en:2294243a-->
## 演習問題

<!--en:57d0fb9f-->
1. **訓練レシピ（`LMOptimizer`）のアブレーション。**`LMOptimizer` は、選択的重み減衰（2D 重み行列のみに $\lambda = 0.1$）、線形ウォームアップ（100 ステップ）、コサイン減衰、および大域的勾配クリッピング（$\|\nabla \mathcal{L}\|_2 \le 1.0$）という 4 つの要素を組み合わせている。1 つずつ要素を無効化しながら、本節の 6 ブロックのバイトレベル `GPT` を 1,000 ステップ訓練せよ。(a) 埋め込みや RMSNorm のゲインを含む*すべて*のパラメータに重み減衰を適用する、(b) ウォームアップなし（`warmup=0`）、(c) 定数学習率（終始 `lr=2e-3`）、(d) 勾配クリッピングなし。検証損失を最も悪化させる単一の変更はどれか。
   :exlabel:`ex_gpt-recipe`

<!--en:15ac0867-->
2. **GPT-2 チェックポイントのテンソル対応表の検証。**GPT-2 の `model.safetensors` ファイルには 148 個の重み・バイアステンソル（および 12 個の旧式の因果マスクバッファ `h.i.attn.bias`）が格納されている。`d2l.num_params(gpt2)` がファイル内の 148 個のパラメータテンソルの総要素数（$124{,}439{,}808$）と一致することを確かめる短い検証コードを書き、さらに使用しているフレームワークの線形層の重み配置に基づいて、読み込み時に `c_attn.weight` を転置する必要があるかどうかを説明せよ。
   :exlabel:`ex_gpt-checkpoint-mismatches`

<!--en:99c816e7-->
3. **0 層の重み共有埋め込みベースライン（$L = 0$）。**`GPT`（`num_blks=0`）を考える。入力トークン $x_t$ は $\mathbf{e}_{x_t}$ として埋め込まれ、`RMSNorm` で正規化された後、$\mathbf{E}^\top$ との内積によってロジット $\mathbf{z}_t \propto \mathbf{e}_{x_t} \mathbf{E}^\top$ を生成する。
   1. 0 ブロックのモデルは、どのクラスの統計的言語モデル（$n$ グラムの次数）を表現できるか。
   2. 入力埋め込みと出力埋め込みを重み共有すると（$\mathbf{W}_{\textrm{LM}} = \mathbf{E}^\top$）、対称行列 $\mathbf{E}\mathbf{E}^\top$ が対角成分 $x_{t+1} = x_t$ に最大の成分を割り当ててしまうのはなぜか（ただし $\mathbf{E}$ のすべての行が等しいノルムをもつとする）。また、重みを共有しないモデルや 1 ブロックのモデルはどのようにしてこの対称性を破るか。
   :exlabel:`ex_gpt-tied-head`

<!--en:eafdc1b3-->
4. **トークナイザ間でのバイトあたりビット数の比較（バイトレベル GPT と BPE GPT-2）。**バイトレベルモデルは交差エントロピーを nat/バイトで報告するのに対し、GPT-2 は交差エントロピーを nat/サブワードトークンで報告する。
   1. 任意の $N_B$ バイトの UTF-8 テキスト（`enc.encode` により $N_T$ 個のサブワードトークンにトークン化されるもの）において、トークンあたりの平均の負の対数尤度が $\mathcal{L}_{\textrm{tok}}$（nat/トークン）であるサブワードモデルの厳密な**バイトあたりビット数（bpb）**が $\textrm{bpb} = \frac{N_T \, \mathcal{L}_{\textrm{tok}}}{N_B \ln 2}$ となることを示せ。
   2. WikiText-103 検証セットの先頭 10,000 バイト上で、本節の 4.8M パラメータのバイトレベル `model` と事前学習済みの 124M パラメータの `gpt2` の両方をバイトあたりビット数で評価せよ。
   :exlabel:`ex_gpt-bits-per-byte`

<!--en:25995c8e-->
5. **Top-$p$（ニュークリアス）サンプリング。**`GPT.generate` を拡張してオプション引数 `top_p`（$0 < p < 1$）を追加し :cite:`Holtzman.Buys.Du.ea.2020`、温度スケーリング後の確率を降順にソートしたうえで、累積確率が $p$ を超える最小のプレフィックスより後ろのトークンをマスクするようにせよ。`gpt2` にプロンプト `"Alan Turing theorized that computers would"` を与え、`top_p=0.9, temperature=0.8` を用いた場合とフィルタリングなしの `temperature=1.2` を用いた場合とで生成結果を比較せよ。

<!--en:637cae41-->
6. **QK-Norm による学習率の安定性。**本節の 6 ブロックのバイトレベル `GPT` を、ピーク学習率 $\eta \in \{2\times 10^{-3}, 6\times 10^{-3}, 1.5\times 10^{-2}\}$ にわたり、標準の `CausalAttention` を用いた場合と、`RMSNorm(d_h)` を `q` と `k` に適用する変種（`CausalAttention.qkv` の内部、:numref:`subsec_qk-norm`）を用いた場合とで訓練せよ。QK-Norm は安定に訓練できる最大学習率をどのように変化させるか。
   :exlabel:`ex_gpt-qk-norm-sensitivity`

<!--en:02c69a8b-->
[ディスカッション](https://d2l.discourse.group/t/36775)

<!--en:a300e618-->
デコーダのみの言語モデル（GPT）<br>**因果スタックと重み共有埋め込み・バイトレベル事前学習・SwiGLU と RNN ベースライン・サンプリング・GPT-2 の読み込み・現代的なレシピ**

<!--en:d95e6fda-->
::: {.slide title="長さ T の単一ウィンドウによる T 個の予測問題の訓練"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:c814502d-->
- 自己回帰言語モデルは連鎖律 $P(x_{1:T+1}) = \prod_{t=1}^T P(x_{t+1} \mid x_{1:t})$ により系列の確率を分解する

<!--en:8899f996-->
- **並列な教師強制：**$T$ 回の別々の順伝播を長さ $1, \dots, T$ の各プレフィックスに対して実行する代わりに、**デコーダのみの GPT** は因果マスクのもとで全 $T$ 個のプレフィックス予測のスコアを **1 回の並列順伝播**で計算する

<!--en:74e8b455-->
::: {.slide title="GPT アーキテクチャ：埋め込み、因果スタック、重み共有ヘッド"}

<!--en:c12b0ef7-->
[アーキテクチャの概要]{.kicker}

<!--en:aa875cfc-->
- **入力埋め込み（$\mathbf{E} \in \mathbb{R}^{V \times d}$）：**離散トークン ID $x_1, \dots, x_T$ を $\mathbb{R}^{B \times T \times d}$ の初期残差ストリームベクトルへ写像する

<!--en:2505e568-->
- **因果 Transformer スタック（$L$ ブロック）：**各 `TransformerBlock` が因果的自己アテンションと位置ごとの FFN をストリームに適用する

<!--en:29167e93-->
- **最終正規化と重み共有出力ヘッド（$\mathbf{E}^\top$）：**正規化されたストリームベクトルを語彙ロジット $\mathbf{Z} = \mathrm{Norm}(\mathbf{H}^{(L)})\mathbf{E}^\top \in \mathbb{R}^{B \times T \times V}$ へ射影し、$\mathbf{E}$ を再利用して $V d$ 個のパラメータ（GPT-2 Small で $38.6\text{M}$ 個）を節約する

<!--en:f6e0f154-->
::: {.slide title="RoPE と GQA を備えた融合型因果的自己アテンション"}

<!--en:d731f9ec-->
[因果アテンション]{.kicker}

<!--en:50e1ad04-->
- **融合 QKV 射影：**$\mathbf{X}$ に単一の行列 $\mathbf{W}_{qkv} \in \mathbb{R}^{d \times (d + 2 H_{\text{kv}} d_h)}$ を掛け、$H$ 個のクエリヘッドと $H_{\text{kv}} \le H$ 個のキー・値ヘッドに分割する

<!--en:87cd8898-->
- **RoPE + 因果マスク（`is_causal=True`）：**位置角度でクエリとキーを回転させ、$s > t$ を $-\infty$ にマスクして $\partial \mathbf{z}_t / \partial \mathbf{x}_s = \mathbf{0}$ とする

<!--en:05975dd2-->
::: {.slide title="位置情報：学習型テーブルと RoPE"}

<!--en:caa96def-->
[位置エンコーディング]{.kicker}

<!--en:55e78ab5-->
- **`pos='learned'`（GPT-2、2019）：**参照行 $\mathbf{p}_t \in \mathbb{R}^d$ を `pos_emb`（$T_{\max} \times d$）から取り出してスタック最下部で一度だけ加算し、$T_{\max} = 1{,}024$ が厳密な上限となる

<!--en:c7ff61d5-->
- **`pos='rope'`（デフォルト / Llama / Qwen）：**事前計算した `d2l.rope_tables(T, head_width)` を用いて*すべて*のアテンション層の内部で $(\mathbf{q}_t, \mathbf{k}_s)$ を回転させ、$32\text{K}$〜$1\text{M}$ トークンまでスケールする相対スコアを与える

<!--en:832273c2-->
::: {.slide title="10 行で書く GPT の完全な順伝播" only="pytorch"}

<!--en:cb6399b9-->
[順伝播の実装]{.kicker}

<!--en:3194b30b-->
トークン ID `X`（形状 `(batch, T)`）が与えられると、`GPT.forward` はすべての位置における次トークンのロジット（形状 `(batch, T, V)`）を同時に返す。

<!--en:d104e599-->
出力射影を `self.token_emb.weight` と重み共有することで、希少なトークンは入力プレフィックスに現れたときと、ソフトマックスの分母で対比されたときの両方で勾配更新を受ける。

<!--en:98568517-->
::: {.slide title="10 行で書く GPT の完全な順伝播" only="jax"}

<!--en:cb6399b9-2-->
[順伝播の実装]{.kicker}

<!--en:4f084454-->
トークン ID `X`（形状 `(batch, T)`）が与えられると、`GPT.__call__` はすべての位置における次トークンのロジット（形状 `(batch, T, V)`）を同時に返す。

<!--en:4b3ce629-->
出力射影を `self.token_emb.attend` を介して重み共有することで、希少なトークンは入力プレフィックスに現れたときと、ソフトマックスの分母で対比されたときの両方で勾配更新を受ける。

<!--en:1dd498d7-->
::: {.slide title="深さに応じた重みのスケーリング初期化（1/√(2L)）"}

<!--en:1580e77b-->
[初期化]{.kicker}

<!--en:ddce39a5-->
- `init_weights` は埋め込みと射影を $\mathcal{N}(0, 0.02^2)$ からサンプリングし、2 つの残差書き込み行列（`W_o`、`W_2`）を $1/\sqrt{2L}$ 倍にスケーリングする

<!--en:54bfb32a-->
- $2L$ 個のサブ層が Pre-Norm ストリームへ書き込むため、ブロック 1 以降も初期ストリームの分散が埋め込みの分散付近に保たれる

<!--en:63091fa0-->
::: {.slide title="自己回帰の交差エントロピー目的関数"}

<!--en:f6f8a32f-->
[訓練の目的関数]{.kicker}

<!--en:b5d924cb-->
入力ウィンドウ $\mathbf{x} = (x_1, \dots, x_T)$ とシフトされたターゲット $\mathbf{y} = (x_2, \dots, x_{T+1})$ が与えられたとき、訓練では全 $B \times T$ 個の位置にわたる平均の負の対数尤度を最小化する。

<!--en:d40cd8f4-->
::: {.slide title="WikiText-103 におけるバイトレベル言語モデリング"}

<!--en:327a54bd-->
[コーパスと初期化の確認]{.kicker}

<!--en:5174a181-->
6 ブロックのバイトレベル `GPT`（$V = 256, d = 256, H = 8, L = 6, T = 128$、$4.79\text{M}$ パラメータ）を、**WikiText-103** の生の UTF-8 バイト列（$270\text{M}$ バイト）上で訓練する。

<!--en:279360d5-->
**訓練前の妥当性確認：**ステップ $0$ では、ランダムなロジットがすべてのバイトにほぼ一様な確率 $1/256$ を割り当てるため、初期損失（$5.5\text{--}5.7\text{ nats/byte}$）は $\ln 256 \approx 5.55\text{ nats/byte}$ に近くなる。

<!--en:6116d67d-->
::: {.slide title="現代的な事前学習オプティマイザ（LMOptimizer）"}

<!--en:821003f6-->
[最適化レシピ]{.kicker}

<!--en:63fe4672-->
- `LMOptimizer` は AdamW（$\eta_{\max} = 2 \times 10^{-3}$）、線形ウォームアップ + コサイン減衰、および大域的勾配クリッピング（$\le 1.0$）を組み合わせる。
  $$\eta_t = \eta_{\max} \cdot \frac{1}{2}\Bigl(1 + \cos\bigl(\pi \tfrac{t - t_w}{S - t_w}\bigr)\Bigr)$$

<!--en:f55885ee-->
- **選択的重み減衰（2D 行列のみに $\lambda = 0.1$）：**埋め込み（`*emb*`）、1D の `RMSNorm` ゲイン、およびバイアスを対象外とする

<!--en:f9117ee3-->
::: {.slide title="訓練曲線と経験エントロピーの基準値"}

<!--en:cba18bb6-->
[訓練結果]{.kicker}

<!--en:995f03af-->
- **急速な収束：**$1{,}000$ ステップ（$8.2\text{M}$ バイト、コーパスの $\sim 3\%$）で、検証交差エントロピーは $\approx 5.6$ から **$1.34\text{ nats/byte}$**（$\approx 1.93\text{ bits/byte}$）まで低下する

<!--en:40c26715-->
- **同じバイト列上のカウントベース基準値：**
  - 一様予測（$\ln 256$）：**$5.55\text{ nats/byte}$**
  - 経験ユニグラム頻度：**$3.17\text{ nats/byte}$**
  - 経験バイグラム遷移：**$2.45\text{ nats/byte}$**

<!--en:fc6f082c-->
::: {.slide title="訓練曲線と経験エントロピーの基準値（続き）"}

<!--en:cba18bb6-2-->
[訓練結果]{.kicker}

<!--en:c9c6f0e7-->
ユニグラムとバイグラムの基準値は、訓練データの先頭 $10\text{M}$ バイトに対する単純な出現回数から求める。

<!--en:02569252-->
::: {.slide title="統制アブレーション：シード間のばらつきと SwiGLU の利得"}

<!--en:29a2cbee-->
[実験の厳密性]{.kicker}

<!--en:7f34a42f-->
純粋なアーキテクチャ上の改善をランダム初期化のノイズと区別するため、`act='swiglu'` を独立な 3 つのシードで、パラメータ数を揃えた `act='gelu'` を 3 つのシードで訓練する。

<!--en:3accbf68-->
- **高い実行間の再現性：**3 つの乱数シード間で、検証損失のばらつきは **$0.01\text{ nats/byte}$ 未満**に収まる

<!--en:9c1c50a4-->
- **統計的に明確な `SwiGLU` の優位性：**同一のパラメータ数（$4.79\text{M}$）とステップ時間で、`SwiGLU` は `GELU` を **$\sim 0.06\text{ nats/byte}$**（シード間のばらつきの $> 5\times$）上回る

<!--en:314fd56a-->
- **Post-Norm 条件：**同じ 100 ステップのウォームアップまたはウォームアップなしの `pre_norm=False` ブロックは、ユニグラムエントロピーである **$3.17\text{ nats/byte}$** で止まる。サブ層の出力が正規化されたストリームよりも大きく成長し、全トークンが同じベクトルに行き着いて訓練が回復しないためである。500 ステップのウォームアップでは Post-Norm も約 $1.38$ まで訓練でき、Pre-Norm（$1.34$）にわずかに遅れる程度となる。

<!--en:cab31162-->
::: {.slide title="パラメータ数を揃えたリカレントモデルとの比較"}

<!--en:05a118e8-->
[Transformer と RNNLM]{.kicker}

<!--en:d5a89296-->
パラメータ数を揃えた `d2l.RNNLM`（$d_{\text{rnn}} = 1{,}920$、$4.74\text{M}$ パラメータ）を 3 通りの学習率で訓練し、ウィンドウ位置 $t \in \{0, 1, 4, 16, 64, 127\}$ ごとの検証損失を調べる。

<!--en:2416d618-->
**なぜ文脈位置 $t$ とともに差が広がるか：**$t = 0$（文脈 1 バイト）では、両モデルのスコアは約 $2.5\text{ nats}$（バイグラムの下限は $2.45$）である。$t = 16 \dots 127$ になると、RNN は約 $1.4\text{ nats}$ で頭打ちになる（履歴を $t$ 回の逐次的な $\tanh$ ステップに押し込める必要があるため）のに対し、`GPT` は任意の先行バイトへ $O(1)$ ホップで直接アテンションを向ける（約 $1.3\text{ nats}$）。

<!--en:690e79ce-->
::: {.slide title="自己回帰デコーディング：温度と Top-k"}

<!--en:80006646-->
[サンプリングの仕組み]{.kicker}

<!--en:4a09287d-->
- 各デコードステップで、`GPT.generate` はロジット $\mathbf{z}_t \in \mathbb{R}^V$ を**温度 $\tau > 0$** でスケーリングし、**top-$k$** 候補に打ち切る。
  $$p_v = \frac{\exp(z_{t,v} / \tau)}{\sum_{u \in \mathcal{V}_k} \exp(z_{t,u} / \tau)}$$

<!--en:c56ebfd4-->
::: {.slide title="自己回帰デコーディング：温度と Top-k（続き）"}

<!--en:80006646-2-->
[サンプリングの仕組み]{.kicker}

<!--en:6e268d0e-->
$\tau = 0.7, k = 8$ では自然な英単語に確率質量が集中するのに対し、$\tau = 2.0$ では分布が平坦化して意味不明な文字列に近づく。

<!--en:64c4534e-->
::: {.slide title="事前学習済み OpenAI GPT-2 Small（124M）の読み込み"}

<!--en:532c94b7-->
[チェックポイントの読み込み]{.kicker}

<!--en:96082955-->
- GPT-2 の 2019 年の設定（`norm='layer'`、`act='gelu'`、`pos='learned'`、`bias=True`）を `d2l.GPT` に渡すと、厳密な **GPT-2 Small** アーキテクチャがインスタンス化される。

<!--en:a2d4aab0-->
::: {.slide title="落とし穴：PyTorch における Conv1D 重みの転置" only="pytorch"}

<!--en:704e16fe-->
[重み配置の落とし穴]{.kicker}

<!--en:3c51e097-->
OpenAI の元の TensorFlow 版 GPT-2 コードは `Conv1D` 層を用いて重み行列を `(in_features, out_features)` として格納していたのに対し、PyTorch の `nn.Linear` は `(out_features, in_features)` として格納する。

<!--en:99cc9585-->
**サイレントバグの危険性：**`c_proj` は正方形状 `(768, 768)` をもつため、`.T` を忘れてもすべての形状アサーションをエラーなしに通過してしまう。転置された行列を検出できるのは、実際のテキストに対するパープレキシティの確認だけである。

<!--en:4b5b80fb-->
::: {.slide title="JAX における Conv1D と nnx.Linear の重み配置の一致" only="jax"}

<!--en:704e16fe-2-->
[重み配置の落とし穴]{.kicker}

<!--en:0f5cbe6e-->
OpenAI の元の TensorFlow 版 GPT-2 コードは `Conv1D` 層を用いて重み行列を `(in_features, out_features)` として格納しており、これは Flax の `nnx.Linear` のカーネル配置と完全に一致する。

<!--en:bc0819c5-->
**サイレントバグの危険性：**フレームワークが線形層の重みを `(in, out)` と `(out, in)` のどちらで格納するかを忘れると、`c_proj` `(768, 768)` のような正方射影が形状エラーを出さずに暗黙のうちに破壊される。読み込んだチェックポイントは必ずホールドアウトのパープレキシティで検証すること。

<!--en:6ee29397-->
::: {.slide title="パープレキシティとテキスト生成による GPT-2 Small の検証"}

<!--en:23e12e5c-->
[チェックポイントの検証]{.kicker}

<!--en:ed3b2023-->
読み込んだ `124.4M` パラメータの `gpt2` モデルを『The Time Machine』上で評価すると、トークンあたりの交差エントロピーは $3.89\text{ nats}$（パープレキシティは $e^{3.89} \approx 49$ であり、一様予測の $50{,}257$ と対照的）となり、一貫性のある貪欲生成・サンプリング生成の続きが得られる。

<!--en:520559a1-->
::: {.slide title="GPT-2 Small による貪欲生成とサンプリング生成"}

<!--en:23e12e5c-2-->
[チェックポイントの検証]{.kicker}

<!--en:0a8c9a1a-->
::: {.slide title="現代のオープンウェイト LLM のアーキテクチャレシピ"}

<!--en:3ec0f888-->
[フロンティアモデルの比較]{.kicker}

<!--en:7901b569-->
- 主要なすべてのオープンウェイト・デコーダのみのモデル族（2023〜2025 年）は、本節のデフォルト `GPT` バックボーン（`RMSNorm`、`RoPE`、`SwiGLU`）を共有する

<!--en:2320b78c-->
- **差別化の 2 つのフロンティア軸：**
  1. **KV キャッシュ圧縮（§{sec_kv-cache}）：**GQA（$32{:}8$ または $64{:}8$）、$5{:}1$ の局所・大域ウィンドウ（Gemma 3）、または低ランク MLA（DeepSeek-V3）
  2. **FFN のスパース性（§{sec_moe}）：**密な `SwiGLU` と細粒度 MoE（$128$〜$256$ エキスパート）

<!--en:fa3aece6-->
::: {.slide title="概念チェック：パラメータ数、パープレキシティ、対称性"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:4ef5ca81-->
3 つの具体的な計算で理解度を確かめよう。

<!--en:63d671b9-->
1. **GPT-2 Medium のサイズ見積もり：**GPT-2 Medium は $L = 24$ ブロック、幅 $d = 1024$、語彙数 $V = 50{,}257$、学習済み位置数 $T_{\max} = 1024$、および重み共有ヘッドをもつ。総パラメータ数を見積もれ。

<!--en:eaf8bb5f-->
2. **実効分岐数としてのパープレキシティ：**本節のバイトレベル GPT は $1.35\text{ nats/byte}$ に達し、GPT-2 Small は『The Time Machine』で $3.89\text{ nats/token}$ のスコアとなる。それぞれの損失は、等確率な選択肢が何通りあることに対応するか。

<!--en:845adf55-->
3. **0 層での重み共有の落とし穴：**`num_blks=0` かつ $\mathbf{E}$ のすべての行が単位ノルムをもつとき、どの次トークン $x_{t+1}$ が $\mathbf{z}_t = \mathbf{e}_{x_t}\mathbf{E}^\top$ で常に最大のロジットを得るか。

<!--en:557b98b6-->
**解答：**

<!--en:41a969fa-->
1. **$\approx 355\text{M}$ パラメータ：**ブロック内に $24 \times 12(1024)^2 \approx 302.0\text{M}$、`token_emb` に $+ 50{,}257 \times 1024 \approx 51.5\text{M}$、`pos_emb` に $+ 1.0\text{M}$

<!--en:6f4822f9-->
2. バイトモデルでは **$e^{1.35} \approx 3.9$ バイト**（全 $256$ 通り中）、GPT-2 では **$e^{3.89} \approx 49$ サブワードトークン**（全 $50{,}257$ 通り中）

<!--en:c7bc7e4a-->
3. **$x_{t+1} = x_t$（現在のトークンの繰り返し）：**対称なグラム行列 $\mathbf{E}\mathbf{E}^\top$ は $\mathbf{e}_i^\top \mathbf{e}_j$ を対角成分 $j = i$ 上で最大化する。アテンションと FFN がこの対称性を破る。

<!--en:8312f4f3-->
::: {.slide title="まとめ：デコーダのみの GPT の構築と訓練"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:a933f8f2-->
- **因果スタックと重み共有ヘッド：**融合された $\mathbf{W}_{qkv}$、RoPE、下三角マスク、および共有された $\mathbf{E} \in \mathbb{R}^{V \times d}$ により、全 $T$ 個のプレフィックス予測を 1 回のパスで訓練

<!--en:5bf8b1b5-->
- **`1/√(2L)` 初期化と `LMOptimizer`：**2D 行列のみの重み減衰とウォームアップ + コサイン減衰を備えた AdamW により、$1.34\text{ nats/byte}$ に $1{,}000$ ステップで到達

<!--en:9c33501a-->
- **統制されたベースラインと GPT-2 の読み込み：**`SwiGLU` が `GELU` を $0.06\text{ nats}$ 上回り、公式の `124M` GPT-2 を直接読み込み可能
