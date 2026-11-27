<!--en:837bd13b-->
# アテンション

<!--en:86a1bc05-->
:numref:`chap_rnn` のリカレントモデルは、系列を 1 トークンずつ読み込み、それまでに見たすべてを固定サイズの隠れ状態に圧縮する。長い文書の末尾付近での予測が最初の段落の細部に依存する場合、リカレントネットワークはその細部が必要になると知るはるか前に、隠れ状態へ書き込んでおかなければならない。履歴を単一のベクトルに圧縮するのではなく、ネットワークがすべての位置に表現を保持し、現在の予測に必要なものだけを取り出すことはできないだろうか。

<!--en:07ae40d2-->
*アテンション*はまさにこれを実現する。各ルックアップでは、現在の位置が求めるものを表す*クエリ*ベクトルに対し、すべての候補位置にある*キー*ベクトルとの間でスコアを計算する。ソフトマックスがそれらのスコアを和が 1 になる非負の重みへと変換し、層はそれらの位置に格納された*値*（value）ベクトルの重み付き平均を返す。重み付き平均は滑らかであるため、勾配降下法によってクエリ、キー、値を生成する射影をエンドツーエンドで訓練できる。このようなルックアップを複数並列に実行すると、*マルチヘッドアテンション*が得られる。アテンションは、ニューラル機械翻訳でデコーダが原言語のすべての単語を直接参照できるように導入され :cite:`Bahdanau.Cho.Bengio.2014`、Transformer における唯一の位置混合演算となった :cite:`Vaswani.Shazeer.Parmar.ea.2017`。

<!--en:ffdd19f3-->
再帰構造を全ペア間のルックアップに置き換えると、新たに 3 つの課題が生じる。

<!--en:1cf1e970-->
1. **順序への不感性：**重み付き和は入力の置換に対して不変であるため、位置情報を明示的に注入しなければならない。

<!--en:b0f30b52-->
2. **二次スケーリング：**$n$ 個のトークンの全ペアに対してスコアを計算するには $O(n^2)$ の演算とメモリが必要であり、これが IO を考慮した厳密カーネルと準二次近似の両方の動機となる。

<!--en:4e253e03-->
3. **機構的解釈：**アテンション重みが示すのはヘッドが*どこを*読み取るかであり、層をまたいで*どのような*関数を計算するかではない。

<!--en:4bce203b-->
最初の 3 つの節では、微分可能なキー・値ルックアップ（:numref:`sec_queries-keys-values`）、スケール付きドット積スコアリングとブールマスキング（:numref:`sec_attention-scoring-functions`）、そしてマルチヘッド自己アテンション（:numref:`sec_multihead-attention`）によってこの層を構築する。続く :numref:`sec_positional-information` では、絶対、回転（RoPE）、および相対（ALiBi）の位置エンコーディングによって系列の順序を扱う。次の 2 つの節では計算スケールを取り上げる。:numref:`sec_attention-at-scale` では $n \times n$ のスコア行列を実体化しないタイル化オンラインソフトマックスカーネル（FlashAttention）を導出し、:numref:`sec_windowed-linear-attention` ではスライディングウィンドウアテンションと線形アテンションの変種を検討する。最後に、:numref:`sec_what-attention-computes` では残差ストリーム（residual stream）、QK/OV 分解、および 2 層の誘導ヘッド（induction head）を通じて訓練済みアテンション回路を解析し、:numref:`sec_attention-further-reading` では各種の拡張と一次文献を概観する。

<!--en:41cabfbb-->
:numref:`chap_transformers` へ直接進む読者は、最初の 4 つの節と :numref:`sec_what-attention-computes` 冒頭の残差ストリームの定式化だけを押さえればよい。計算コストに関する各節は、:numref:`sec_kv-cache` および :numref:`chap_modern_rnn` と併せて読むことができる。

<!--en:bb504911-->
アテンション機構：微分可能なコンテンツベース検索<br>**クエリ、キー、値・スケール付きドット積・マルチヘッドと GQA・RoPE と ALiBi・FlashAttention・線形アテンション・回路**

<!--en:359d614e-->
::: {.slide title="固定サイズ状態のボトルネックから直接ルックアップへ"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:451a4c71-->
- **再帰構造のボトルネック：**幅 $d = 768$ のモデル（GPT-2 small）は $n = 1{,}024$ トークンを読む際、$1{,}024 \times 768 \approx 786{,}000$ 個の履歴数値を $768$ 個の浮動小数点数へと $O(n)$ ステップの勾配連鎖にわたり押し込める

<!--en:4aaa32a2-->
- **コンテンツベース検索：**各位置に 1 つの表現を保持し、関連する文脈を必要に応じて取り出す

<!--en:68e80f47-->
- **微分可能なソフトルックアップ：**不連続なデータベース照合を格納された値のソフトマックス重み付き平均に置き換え、誤差逆伝播法でエンドツーエンドに訓練可能にする

<!--en:91807bca-->
::: {.slide title="基本プリミティブ：クエリ、キー、値"}

<!--en:053f73b8-->
[中核メカニズム]{.kicker}

<!--en:8abcaa58-->
- **3 つの異なる役割：**各ルックアップは**クエリ** $\mathbf{q} \in \mathbb{R}^d$（現在の位置が求めるもの）を $m$ 個の**キー** $\mathbf{k}_i \in \mathbb{R}^d$（各候補位置が提供するもの）と比較する

<!--en:79df554e-->
- **値の凸混合：**適合度スコア $a(\mathbf{q}, \mathbf{k}_i)$ はソフトマックスを通って非負の重み $\alpha_i$（総和 $1$）を形成する。
  $$\textrm{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\,\mathbf{v}_i$$

<!--en:92664e51-->
- **並列ヘッド：**1 回のルックアップが 1 つのアテンション**ヘッド**を成し、$H$ 個のヘッドを並べて実行することでトークンは複数の引数を同時に取り出せる

<!--en:058f7782-->
![ソフトルックアップはクエリと各キーとのスコアを計算し、値の凸結合を返す。](../img/mdl-attention-soft-lookup.svg){.nostretch width=100%}

<!--en:04721d10-->
::: {.slide title="全ペアルックアップが生む 3 つの課題"}

<!--en:a83731ae-->
[アーキテクチャ上のトレードオフ]{.kicker}

<!--en:1763c6c1-->
- **1. 順序への不感性：**重み付き和 $\sum_i \alpha_i \mathbf{v}_i$ は置換不変で語順を無視するため、絶対位置テーブル、**RoPE**、または **ALiBi** を要する（**§{sec_positional-information}**）

<!--en:d74f2d0e-->
- **2. 二次コスト：**$n$ トークンの全ペアのスコア計算には $\Theta(n^2 d)$ FLOPs と素朴な実装で $O(n^2)$ のメモリ（$64\text{ GiB}$、$n=128\text{K}$ 時）を要し、**FlashAttention**（**§{sec_attention-at-scale}**）、ウィンドウ、**線形アテンション**（**§{sec_windowed-linear-attention}**）の動機となる

<!--en:022fa63b-->
- **3. 機構的不透明性：**アテンション重みが示すのはヘッドが*どこを*見るかであり、**QK/OV 回路**（**§{sec_what-attention-computes}**）を介して残差ストリーム上で*どのような*関数を計算するかではない

<!--en:0ff6467b-->
::: {.slide title="第 §{num:chap_attention} 章の構成とロードマップ"}

<!--en:99e0030a-->
[章のロードマップ]{.kicker}

<!--en:c19c3523-->
- **パート 1：層の構築（§{sec_queries-keys-values}〜§{num:sec_multihead-attention}）：**ソフトなキー・値ルックアップ、Nadaraya–Watson カーネルプーリング、$1/\sqrt{d}$ スケール付きドット積スコアリング、ブールマスク代数、マルチヘッドおよびグループ化クエリアテンション

<!--en:6adc4d8f-->
- **パート 2：系列の順序（§{sec_positional-information}）：**置換同変性、正弦波回転、**RoPE**、**ALiBi**、ゼロショットのコンテキスト拡張

<!--en:7bce3df4-->
- **パート 3・パート 4：スケールと回路（§{sec_attention-at-scale}〜§{num:sec_what-attention-computes}）：**ブロック単位のオンラインソフトマックス（**FlashAttention**）、スライディングウィンドウ、線形アテンションの漸化式、2 層の**誘導ヘッド**

<!--en:72b72fbb-->
::: {.slide title="通しベンチマークと学習ガイド"}

<!--en:05adf916-->
[章を貫く学習の軸]{.kicker}

<!--en:5bb5449d-->
すべての節で実行可能なベンチマークによって主張を検証し、基準となる GPT-2 small のヘッド（$d = 768, H = 12, d_h = 64$）で正確な FLOPs とバイト数を数える。

<!--en:d79273b6-->
| 節 | 実行可能なベンチマーク / モデル | 主な定量的知見 |

<!--en:ba5609c8-->
| **§{sec_queries-keys-values}〜§{num:sec_attention-scoring-functions}** | Nadaraya–Watson 1D 回帰、`ToyAligner` 置換タスク | バンド幅 $\sigma$ と $1/\sqrt{d}$ が勾配飽和を防ぐソフトマックス温度として機能 |

<!--en:2e74e22e-->
| **§{sec_multihead-attention}** | *「Alice gave Bob the book」*、ガウス分布の **copy-both** タスク | 1 ヘッドは $\ge 1/2$ の相対 MSE を生じるが、2 ヘッドは $\sim 10^{-7}$ に到達（同一の $4d^2$ パラメータ） |

<!--en:d443cbff-->
| **§{sec_positional-information}** | WikiText-103 バイト列上の `TinyCharLM`（$n = 128$ $\to$ $512$） | RoPE が範囲内で最小のパープレキシティ（$7.3$）を達成し、低速ペア補間が $4\times$ のコンテキストを修復 |

<!--en:446d11f9-->
| **§{sec_attention-at-scale}〜§{num:sec_windowed-linear-attention}** | チャンク化オンラインソフトマックス、最大 $n = 32\text{K}$ の融合 Flash/FlexAttention | オンラインソフトマックスがメモリを $O(n^2)$ から $O(n)$ に削減し、ウィンドウ $w = 128$ が $> 30\times$ の高速化を達成 |

<!--en:e6b7ec00-->
| **§{sec_what-attention-computes}** | 可変周期の反復ランダムパターン上の `TinyCharLM` | 2 ブロックは文脈内誘導を学習（正解率 $\sim 90\%$）し、1 ブロックは $\sim 10\%$ で停滞 |

<!--en:fd9600fe-->
**第 §{num:chap_transformers} 章（Transformer）への最短ルート：**まず **§{sec_queries-keys-values}〜§{num:sec_positional-information}** と **§{sec_what-attention-computes}** 冒頭の残差ストリームを読み、**§{sec_attention-at-scale}〜§{num:sec_windowed-linear-attention}** は KV キャッシュおよび現代の RNN と併せて学ぶとよい。
