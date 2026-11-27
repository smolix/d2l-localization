<!--en:bd41d179-->
# 訓練状態のシャーディング

<!--en:4cf89bb8-->
:numref:`sec_memory_precision` で見たように、fp32 の Adam を用いた訓練では、重み、その勾配、および Adam の 2 つのモーメント推定値にそれぞれ 4 バイトずつ、パラメータあたり合計 16 バイトを保持する。18 ブロック・1B パラメータの GPT では、この固定状態だけで 1 トークンも読み込む前に 24 GiB カードのうち 15.1 GiB を占め、さらにデータ並列（data parallelism、:numref:`sec_multi_gpu`、:numref:`sec_multi_gpu_concise`）はこの 16 バイトすべてを各 GPU に複製する。GPT の深さを 2 倍の 36 ブロック（1.92B パラメータ、状態サイズ 28.6 GiB）にすると、1 基の GPU と同様、4 基の 24 GiB GPU によるデータ並列でも訓練できなくなる。

<!--en:f94066ad-->
1 枚のカードに収まらないモデルを $k$ 基の GPU で訓練できるだろうか。答えは可能である。なぜなら、GPU がブロックの重み全体を必要とするのはそのブロックの順伝播または逆伝播を計算している間だけであり、全勾配や完全なオプティマイザ状態（optimizer state）はそもそも一切必要としないからである。したがって、$k$ 基の各 GPU は重み、勾配、およびオプティマイザのモーメントを $1/k$ ずつ保持し、各ブロックの実行直前にその重みをオンザフライで集め、各ブロックの勾配が生成され次第すぐに reduce-scatter すればよい。

<!--en:8cc4f943-->
まず、形状のみを保持するメタデバイス上で、36 ブロックの GPT が持つ $16N$ バイトの訓練状態（training state）が単一の RTX 4090 の容量 24 GiB を超えることを確認しよう。

<!--en:4617892b-->
## オプティマイザ状態、勾配、重みの分割

<!--en:b6eaf86a-->
標準的なデータ並列では、$k$ 基すべての GPU が Adam のモーメント推定値の同一なコピーを保持し、同一のオプティマイザ更新を実行する。*ZeRO*（*Zero Redundancy Optimizer*）ファミリー :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` は、累積的な 3 つの段階によってこの冗長性を排除する（:numref:`tab_mgp-zero`）。

<!--en:c943fe60-->
* **ZeRO Stage 1** はオプティマイザ状態のみを分割する。各 GPU はパラメータの $1/k$ について Adam の 2 つのモーメントを保持する（パラメータあたり $8 + 8/k$ バイト）。allreduce（$=\text{reduce-scatter} + \text{all-gather}$、:numref:`subsec_mg-ring`）の代わりに、GPU 間で勾配を reduce-scatter することで、各 GPU が自身の $1/k$ シャードに対応する合算済み勾配を受け取り、自身のパラメータを更新したうえで、更新後の重みを all-gather する。移動するバイト数はデータ並列と完全に同じである。

<!--en:2371aa8f-->
* **ZeRO Stage 2** は勾配も分割する（パラメータあたり $4 + 12/k$ バイト）。逆伝播中に各勾配バケットが埋まり次第すぐに reduce-scatter され、自身が所有しない $1 - 1/k$ の勾配シャードは解放される。これもデータ並列に対する追加の通信バイトコストはゼロである。

<!--en:146dff2a-->
* **ZeRO Stage 3**、すなわち*完全シャーディングデータ並列*（Fully Sharded Data Parallelism、FSDP）:cite:`Zhao.Gu.Varma.ea.2023` は、重みもシャーディングする（パラメータあたり $16/k$ バイト）。順伝播または逆伝播でブロックを実行する直前に all-gather でそのブロックの重み全体を組み立て、ブロックが完了した直後に所有しない重みシャードを解放し、そのブロックの勾配は reduce-scatter する（:numref:`fig_fsdp_lifecycle`）。

<!--en:d07e7ede-->
:$k$ 基の GPU 上で fp32 の Adam を用いた場合の逆伝播終了時点における GPU あたりの訓練状態、および $k = 2$ における 36 ブロック・19.2 億パラメータの GPT での値。

<!--en:f160e345-->
| 各 GPU の保持内容 | 重み | 勾配 | Adam 状態 | パラメータあたりバイト数 | GPT（$k = 2$） |

<!--en:4978c5e7-->
| データ並列 | すべて | すべて | すべて | 16 | 28.6 GiB |

<!--en:6fb3025f-->
| ZeRO Stage 1 | すべて | すべて | $1/k$ | $8 + 8/k$ | 21.4 GiB |

<!--en:654ca3e8-->
| ZeRO Stage 2 | すべて | $1/k$ | $1/k$ | $4 + 12/k$ | 17.9 GiB |

<!--en:6891dea4-->
| ZeRO Stage 3（FSDP） | $1/k$ | $1/k$ | $1/k$ | $16/k$ | 14.3 GiB |

<!--en:b6fcf1f3-->
![1 基の GPU 上における FSDP 下での 1 ブロックの重みのライフサイクル。使用していない間、ブロックはシャーディングされており、各 GPU はその $1/k$ を保持する（他のブロックは常にシャーディングされたままである）。順伝播のために all-gather がブロック全体を組み立て、終了後に解放する。逆伝播のために 2 回目の all-gather が再びブロックを組み立てて終了後に解放し、reduce-scatter によって各 GPU には担当分の勾配が残る。](../img/mdl-perf-fsdp-lifecycle.svg)

<!--en:da477b88-->
FSDP はステップごとに $S = 4N$ バイトに対する 2 回の all-gather（順伝播と逆伝播）と 1 回の reduce-scatter を実行するため、リング allreduce が 2 回の集団通信（collective）フェーズで済むところを 3 フェーズ要し、**データ並列の $1.5\times$ の通信量**を移動させる。:numref:`tab_collectives` に並列訓練で使われる主要な 4 つの集団通信演算をまとめ、:numref:`fig_collectives` にそれぞれが 2 基の GPU 間で何を移動させるかを示す。

<!--en:51637ac0-->
![2 基の GPU 間における実行前後のデータ移動として見た 4 つの集団通信。GPU 0 はチャンク $a_0, a_1$、GPU 1 はチャンク $b_0, b_1$ を初期状態とし、all-gather では GPU $g$ は自身のシャードであるチャンク $g$ のみを初期状態とする。元のチャンクを青、和を緑で示す。](../img/mdl-perf-collectives.svg)

<!--en:48924402-->
:データ並列訓練およびシャーディング訓練を支える集団通信演算。

<!--en:cf05ac75-->
| 集団通信 | 各 GPU が最終的に得る結果 | 使用箇所 |

<!--en:e0ced2ce-->
| allreduce | 全体の和 | データ並列における勾配 |

<!--en:3aa03358-->
| reduce-scatter | 和の $1/k$ シャード 1 つ | ZeRO および FSDP における勾配 |

<!--en:3fbf4ab3-->
| all-gather | 全シャードを連結したもの | ZeRO および FSDP における重み |

<!--en:d1e3cb62-->
| all-to-all | 各 GPU からの異なるスライス | 混合エキスパート（:numref:`sec_moe`）におけるトークンルーティング |

<!--en:8c105a55-->
## メモリと通信

<!--en:4706d640-->
$k = 2$ 基の GPU 上で、GPU あたり 1,024 トークンの系列 2 本とブロックごとの活性化チェックポイント（activation checkpointing、:numref:`subsec_mp-checkpointing`）を用いて 36 ブロックの GPT を FSDP で実行したときのピークメモリとステップ時間を予測してみよう。

<!--en:3b593ca0-->
* **メモリ：**シャーディングされた訓練状態は GPU あたり $16N/2 = 14.3$ GiB を占める。チェックポイント保存された活性化値にトークンあたり約 1 MiB（2,048 トークンで約 2 GiB）を要し、さらにその時点で集められている 1〜2 個のブロック（各約 0.2 GB）を加えても、GPU あたりのピークメモリは高々**約 17 GiB** となり、24 GiB を十分に下回る。

<!--en:fa979f07-->
* **通信：**3 つの集団通信フェーズはそれぞれ GPU あたり $(k-1)S/k$ バイトを送信するため、合計の通信時間は次式となる。

  $$
  t_\textrm{comm} = \frac{3(k-1)}{k}\,S\,\beta_\textrm{link}.
  $$
  :eqlabel:`eq_fsdp_comm`

  FSDP の重要なスケーリング特性に注目しよう。**GPU を増やすと GPU あたりの状態メモリは $k$ で割られて減る一方、各 GPU がステップごとに送信するバイト数は $1.5S$（$k = 2$ のとき）から $3S$（$k$ が大きいとき）へと増加する**。1 ステップに占める通信の割合を減らせる手段は、GPU あたりに処理するトークン数を増やすか、より高速なインターコネクトを使うことだけである。

<!--en:c38baa04-->
まず、256 MiB の all-gather における $\beta_\textrm{link}$ を $k = 2$ 基の GPU 間で計測する。

<!--en:e71ed503-->
実測した $\beta_\textrm{link}$ と RTX 4090 の tf32 ピーク性能である 82.6 TFLOP/s（`peak_flops`）を用いて、GPU あたりの状態メモリ（$16N/k$）、通信時間（$3(k-1)S\beta_\textrm{link}/k$）、およびチェックポイント保存時の計算時間（$8NbT/P$）に関する机上予測を評価しよう。

<!--en:d8e4559b-->
ホストメモリ経由のバス帯域幅が約 2.2 GB/s の場合、ステップあたり $1.5S = 11.5$ GB のデータ移動には約 5 s かかるのに対し、2,048 トークン分の算術演算はわずか約 0.4 s で終わる。0.4 s の計算では 5 s の通信を隠蔽できないため、1 ステップには約 5 s かかるはずである。

<!--en:ccdcccea-->
## 2 基の GPU によるシャーディング訓練

<!--en:33e0eb2b-->
それでは、シャーディングされた 36 ブロック GPT のステップを 2 基の GPU 上で実行する。その際、どの単一 GPU も 28.6 GiB の状態全体を一度に実体化しないようモデルを構築する。

<!--en:104ab75c-->
PyTorch では、`'meta'` デバイス上でモデルを構築し、チェックポイント化した各ブロックと最上位モデルを `fully_shard`（`torch.distributed.fsdp`）でラップしたうえで、`model.to_empty(device=device)` によりこのランクのシャードのみを実体化し、インプレースで初期化する。

<!--en:982e1a82-->
続いて `TRAIN_FSDP` を $k$ 個のプロセスにわたって `d2l.torchrun` で起動し、ランク 0 からピーク確保メモリ（`peak_bytes`）、ステップ所要時間（`times`）、および NCCL 通信時間（`comm`）を収集する。

<!--en:b010f490-->
JAX では、`fsdp_sharding` が各パラメータテンソルを $k$ で割り切れる最初の軸に沿ってシャーディングし（`P('data')`）、`create` がこのシャーディング制約を適用することで、各 GPU は `model` と `opt` のうち自身のシャードのみを初期化する。

<!--en:f2a1ce6c-->
次に、活性化チェックポイントのために各ブロックを `nnx.remat` でラップし、`loss_fn` における各ブロックの活性化値の出力がバッチ方向にシャーディングされた状態（`data`）を保つよう制約して XLA が FSDP の all-gather と reduce-scatter を自動発行するようにし、さらに `model` と `opt` を `train_step` でバッファ譲渡（donate）して更新後の状態が古いシャードをインプレースで上書きするようにする。

<!--en:abe6c0d1-->
ステップを実行する前に、`train_step` をバッチ方向にシャーディングされたトークンバッチ `X`（形状 $(kb, T)$）に対してコンパイルし、`compiled.memory_analysis()` を呼び出して譲渡された状態と一時バッファの GPU あたりのメモリ使用量を確認する。

<!--en:e7939c2f-->
XLA が期待どおり FSDP の集団通信を発行したことを検証するため、`collective_megabytes` でコンパイル済み HLO テキストを解析し、各集団通信演算の結果バイト数を GPU あたりで集計する。

<!--en:273274b6-->
`compiled` 内の集団通信を集計すると、`all-gather`（ステップあたり 2 回）と `reduce-scatter`（ステップあたり 1 回）が予測どおりのバイト量を移動させていることが確認できる。

<!--en:166f61d5-->
ここで `train_step` をウォームアップしたうえで 3 回の反復実行を計測し、さらにステップをプロファイリングして NCCL 通信時間（`comm`）とデバイスのピークメモリ（`peak_bytes`）を記録する。

<!--en:1148a2a5-->
最後に、実測された GPU あたりのピークメモリ、ステップ所要時間の中央値、および GPU 0 が NCCL 通信カーネルに費やした時間を出力する。

<!--en:a083b8f3-->
両方の GPU はカードあたりピーク約 16 GiB（予測上界の 17 GiB 未満）で 36 ブロックのモデルを保持しており、各ステップの所要時間は約 5.5 s で、GPU 0 はステップの 90% 超を NCCL 通信カーネルに費やしている。これは机上予測である 5 s から 1/6 以内に収まっている。NVLink 5（片方向あたり 900 GB/s）を用いれば、同じ 11.5 GB の集団通信はわずか約 13 ms で完了し、0.4 s の計算時間の 1/30 未満となる。

<!--en:57aba994-->
この比較から、*いつ*シャーディングすべきかもわかる。訓練状態が妥当なサイズのマイクロバッチとともに単一 GPU に収まるモデル（18 ブロックの GPT など）では、データ並列と勾配累積（gradient accumulation、グローバルステップごとに 1 回だけ通信する）を組み合わせたほうが、PCIe 経由の FSDP（マイクロバッチごとに重みを 2 回再収集する）よりもはるかに高速に訓練できる。

<!--en:30ac03ff-->
## まとめ

<!--en:d634531c-->
データ並列は fp32 Adam の状態 $16N$ バイトすべてを各 GPU に複製する。ZeRO はこの状態を $k$ 基の GPU に分割する。Stage 1 はオプティマイザのモーメントをシャーディングし（パラメータあたり $8 + 8/k$ バイト）、Stage 2 は勾配もシャーディングするが（パラメータあたり $4 + 12/k$ バイト）、どちらもステップあたりに移動するデータ量はリング allreduce と完全に同じ $2(k-1)S/k$ バイトである。Stage 3（FSDP）は重みもシャーディングし（パラメータあたり $16/k$ バイト）、順伝播と逆伝播のたびに各ブロックの重みを収集し勾配を reduce-scatter することで、データ並列の $1.5\times$ の通信量（GPU あたり $3(k-1)S/k$ バイト）で動作する。シャーディングにより、単独のカードには収まらない 28.6 GiB の状態を持つ 36 ブロックの GPT を、2 基の 24 GiB GPU で訓練できるようになる。

<!--en:2294243a-->
## 演習問題

<!--en:93e21771-->
1. **シャーディングのコスト。** ● GPU あたり・ステップあたりの通信量を、fp32 の重みまたは勾配 1 セット分のバイト数である $S = 4N$ を単位として数えよ。
    1. ZeRO Stage 3 のある変種では、収集した各ブロックを順伝播から逆伝播が完了するまでメモリ上に保持し続ける。これによりどの集団通信が不要になり、:numref:`tab_mgp-zero` のメモリ削減量のうちどれだけを手放すことになるか。
    1. 混合精度 FSDP は、各 GPU のマスター重みと Adam シャードを fp32 に保ったまま、重みの all-gather と勾配の reduce-scatter を bf16 で実行する。ステップあたりの転送バイト数を fp32 データ並列と比較し、パラメータあたりのメモリ量を :numref:`tab_mgp-zero` の最終行と比較せよ。
    1. $k = 2$ 基の GPU 上での 36 ブロック GPT において、実測した $\beta_\textrm{link}$ のもとで、チェックポイント保存時の計算時間（83 TFLOP/s でトークンあたり $8N$ FLOPs）が通信時間 :eqref:`eq_fsdp_comm` と釣り合うには、1 ステップで GPU あたり何トークン必要か。その活性化値はカードに収まるか。また、FSDP がマイクロバッチごとに重みを再収集する場合に勾配累積が役に立たないのはなぜか。

<!--en:1f9dc34e-->
2. [extended] **オプティマイザ状態のシャーディング。** ● ZeRO Stage 1 :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` を、`TRAIN_DDP`（:numref:`sec_multi_gpu_concise`）および `torch.optim.Adam` を出発点として手作業で実装せよ。
    1. パラメータテンソルをランク間で分割し、各ランクがパラメータの約 $1/k$ を所有して自身の所有テンソルに対してのみ Adam を実行したうえで、更新後の各テンソルをその所有元から `dist.broadcast` でブロードキャストするようにせよ。数ステップ後にパラメータが通常の DDP Adam と一致することを検証し、`torch.distributed.optim.ZeroRedundancyOptimizer` と比較せよ。
    1. シャーディングの有無それぞれについて $k = 2$ および $k = 4$ での確保済み GPU メモリとピーク GPU メモリを計測し、DDP の allreduce と所有元からのブロードキャストによる総送信バイト数を、Stage 1 の reduce-scatter と all-gather による総送信バイト数と比較せよ。

<!--en:d13a5945-->
3. [code] **テンソル並列とデータ並列の比較。** ● MLP ブロック $\mathbf{Y} = \phi(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$（ただし $\mathbf{X} \in \mathbb{R}^{n \times d}$、$\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$、$\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$）を $k$ 基のデバイスに分散することを考える。テンソル並列（:numref:`fig_splitting`、:cite:`shoeybi2019megatron`）では、$\mathbf{W}_1$ を列方向に、$\mathbf{W}_2$ を行方向に分割する。
    1. 各デバイスが通信なしに $n \times d$ の部分出力を計算でき、その $k$ 個の部分出力を合算すると $\mathbf{Y}$ が得られることを示せ。順伝播にはどの集団通信が必要で、逆伝播で $\nabla_{\mathbf{X}}$ を求めるにはどの集団通信が必要か。
    1. デバイスあたり・ステップあたりに送信される値の個数をテンソル並列とデータ並列で比較し、テンソル並列のほうがデータ送信量が少なくなるトークン数 $n$ の上限を求めよ。それにもかかわらず、テンソル並列が高速なノード内 NVLink ドメインに限定されるのはなぜか。
    1. `parallelize_module` を `ColwiseParallel()` および `RowwiseParallel()` とともに 1D の `DeviceMesh` 上で用い、`CommDebugMode` によって順伝播と逆伝播の集団通信の回数を数えよ。
   :exlabel:`ex_mgp-tensor`

<!--en:8f3c0a63-->
2. [extended] **オプティマイザ状態のシャーディング。** ● ZeRO Stage 1 :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` を、`train_step`（:numref:`sec_multi_gpu_concise`）と `optax.adam` を出発点として宣言的に実装せよ。モデルパラメータは複製された状態（`P()`）に保ち、`nnx.state(opt)` のリーフを `fsdp_sharding` で配置せよ。
    1. 必要に応じて `train_step` の末尾で `jax.sharding.reshard` を用いてパラメータが複製状態を保つようにし、複数ステップにわたるパラメータ更新が複製版の Adam と一致することを検証せよ。
    1. コンパイル済みステップを `collective_megabytes` で調べ、その集団通信および $k = 2$ と $k = 4$ における使用中メモリ・ピークメモリを :numref:`tab_mgp-zero` と比較せよ。

<!--en:7d3b57ef-->
3. [code] **テンソル並列とデータ並列の比較。** ● MLP ブロック $\mathbf{Y} = \phi(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$（ただし $\mathbf{X} \in \mathbb{R}^{n \times d}$、$\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$、$\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$）を $k$ 基のデバイスに分散することを考える。テンソル並列（:numref:`fig_splitting`、:cite:`shoeybi2019megatron`）では、$\mathbf{W}_1$ を列方向（`P(None, 'data')`）に、$\mathbf{W}_2$ を行方向（`P('data', None)`）に分割する。
    1. 各デバイスが通信なしに $n \times d$ の部分出力を計算でき、その $k$ 個の部分出力を合算すると $\mathbf{Y}$ が得られることを示せ。順伝播にはどの集団通信が必要で、逆伝播で $\nabla_{\mathbf{X}}$ を求めるにはどの集団通信が必要か。
    1. デバイスあたり・ステップあたりに送信される値の個数をテンソル並列とデータ並列で比較し、テンソル並列のほうがデータ送信量が少なくなるトークン数 $n$ の上限を求めよ。それにもかかわらず、テンソル並列が高速なノード内 NVLink ドメインに限定されるのはなぜか。
    1. $\mathbf{W}_1$ を `P(None, 'data')`、$\mathbf{W}_2$ を `P('data', None)` で配置し、$\mathbf{X}$ を `Auto` メッシュ上で複製したうえで、順伝播と逆伝播を JIT コンパイルして発行される集団通信を調べよ。
   :exlabel:`ex_mgp-tensor`

<!--en:f98d8942-->
4. **次ブロックのプリフェッチ。** FSDP における順伝播で、1 ブロックの all-gather 時間を $g$、計算時間を $c$ とする。オーバーラップなしでは $L$ ブロックに $L(g + c)$ かかるが、プリフェッチではブロック $l + 1$ の all-gather をブロック $l$ の計算中に発行する。
    1. プリフェッチによって順伝播の時間が $g + (L-1)\max(g, c) + c$ に短縮されることを示し、どのようなスケジュールでも $L\max(g, c)$ より速くできない理由を説明せよ。
    1. $g$ と $c$ の値を 36 ブロック GPT の 1 ブロック（$50$M パラメータ、GPU あたり 2,048 トークン、83 TFLOP/s）について本節の PCIe リンクと NVLink 5（900 GB/s）のそれぞれで求め、各インターコネクトでプリフェッチがどれだけ時間を短縮するか述べよ。

<!--en:3ee51ae5-->
[ディスカッション](https://d2l.discourse.group/t/36886)

<!--en:214c5972-->
ZeRO と完全シャーディングデータ並列（FSDP）<br>**16N 状態のボトルネック・ZeRO Stage 1〜3・all-gather と reduce-scatter・FSDP のライフサイクル・通信とメモリ**

<!--en:39dbafcc-->
::: {.slide title="16N のパラメータ状態だけで GPU メモリを超えるとき"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:974ac1de-->
1B パラメータの GPT を $18$ ブロックから $36$ ブロック（幅 $d = 2048$）へ倍増させると $1.92\text{B}$ パラメータのモデルとなり、その `float32` Adam 状態（$16N$ バイト）は $24\text{ GiB}$ の RTX 4090 に収まらなくなる。

<!--en:842c633b-->
- **データ並列では解決できない理由：**標準的な DDP は $16\text{ B/param}$ すべて（$28.6\text{ GiB}$）を全 GPU に複製するため、$24\text{ GiB}$ カードを 4 枚使っても 1 枚のときと同様に 1 トークンも処理する前にメモリ不足になる

<!--en:05f1f61d-->
::: {.slide title="k 基の GPU 間で状態を複製することが冗長な理由"}

<!--en:5e1073f6-->
[核心となる着眼点]{.kicker}

<!--en:9cd5c2c2-->
- **重み（$\mathbf{W}_\ell$、$4\text{ B/param}$）：**ブロック $\ell$ が順伝播または逆伝播を実行している間だけ必要

<!--en:e8f295e3-->
- **勾配（$\nabla\mathbf{W}_\ell$、$4\text{ B/param}$）：**ブロック $\ell$ の逆伝播からオプティマイザの更新ステップまでの間だけ必要

<!--en:534d7df8-->
- **Adam のモーメント（$\mathbf{m}, \mathbf{v}$、$8\text{ B/param}$）：**ステップ末尾の**要素ごとの Adam 更新の内部でのみ**参照されるため、$k$ 基の各 GPU はパラメータの $1/k$ だけを所有・更新すればよい

<!--en:5d82e95a-->
::: {.slide title="ZeRO の 3 段階と GPU あたりメモリ"}

<!--en:8c793f18-->
[ZeRO の体系]{.kicker}

<!--en:fc8e69f6-->
- **ZeRO-1（$8 + 8/k\text{ B/param}$）：**Adam のモーメント $\mathbf{m}, \mathbf{v}$ を $k$ 基の GPU にシャーディング（$21.4\text{ GiB}$、$k=2$ での $1.92\text{B}$ GPT）

<!--en:2f13cd68-->
- **ZeRO-2（$4 + 12/k\text{ B/param}$）：**モーメントと勾配をシャーディングし（$17.9\text{ GiB}$、$k=2$ 時）、通信量は DDP と同じ $2\,\frac{k-1}{k}S$

<!--en:cf1ff427-->
- **ZeRO-3 / FSDP（$16/k\text{ B/param}$）：**重み・勾配・モーメントをシャーディング（$14.3\text{ GiB}$、$k=2$ 時、活性化値用に $\sim 10\text{ GiB}$ の余裕が残る）

<!--en:40665bcf-->
::: {.slide title="シャーディング訓練を支える集団通信プリミティブ"}

<!--en:67dff45d-->
[集団通信演算]{.kicker}

<!--en:66919a60-->
- **リング allreduce = reduce-scatter + all-gather：**$\frac{2(k-1)}{k}S$ の総和演算を、それぞれ $\frac{k-1}{k}S$ バイトを送る 2 つの半集団通信に分解

<!--en:cdd8936a-->
- **reduce-scatter（$\text{Full } S \to \text{Summed } S/k$）：**GPU 間で勾配を合算し、シャード $j$ を所有元の GPU $j$ に配信

<!--en:0cf1cef3-->
- **all-gather（$\text{Shard } S/k \to \text{Full } S$）：**層の実行直前またはシャーディングされたオプティマイザ更新後に、GPU 間で重み全体を再構築

<!--en:218963e7-->
::: {.slide title="ZeRO Stage 1 における追加通信なしでのモーメント分割"}

<!--en:f2456114-->
[ZeRO Stage 1]{.kicker}

<!--en:ae23fc50-->
- **allreduce の前後半の間に Adam を挿入：**(1) 勾配を reduce-scatter し（$\frac{k-1}{k}S$）、(2) 所有する $1/k$ の重みを所有する $1/k$ の Adam モーメントで更新し、(3) 更新後の重みを all-gather する（$\frac{k-1}{k}S$）

<!--en:cea32378-->
- **追加の通信バイト数はゼロ：**GPU あたりの総通信量は標準的な DDP と同一の $\frac{2(k-1)}{k}S$ のまま、Adam の状態メモリを $k\times$ 削減

<!--en:f2093eb9-->
::: {.slide title="ZeRO Stage 2 における勾配のストリーミング reduce-scatter"}

<!--en:8d7acada-->
[ZeRO Stage 2]{.kicker}

<!--en:6c2247a7-->
$4N$ バイトすべての勾配を逆伝播の最後まで保持する必要があるだろうか。各 GPU が更新するのはパラメータの $1/k$ だけである。そこで **ZeRO Stage 2** は勾配の解放を逆伝播に組み込む。

<!--en:f90afe4c-->
- **`backward()` 中のバケット単位の reduce-scatter：**誤差逆伝播法があるブロックまたはバケットの勾配計算を終え次第、GPU は直ちにそのバケットに対して **reduce-scatter** を起動

<!--en:9c3b5f14-->
- **非所有シャードの即時解放：**各 GPU は自身が所有する合算済みの $1/k$ 勾配スライスのみを保持し、次の逆伝播ブロックを実行する前に残りの $(k-1)/k$ 勾配スライスを直ちに解放

<!--en:27e2a04b-->
- **DDP と同じ通信量：**逆伝播全体とステップ後の重みの all-gather を通じて、各 GPU の総送信量は依然として $\frac{2(k-1)}{k}S$ バイトでありながら、$4N$ バイトの勾配テンソル全体を一度に実体化することはない

<!--en:f3e5680d-->
::: {.slide title="ZeRO Stage 3（FSDP）におけるブロック重みの収集と解放"}

<!--en:939754cb-->
[完全シャーディングデータ並列]{.kicker}

<!--en:8f34fb2e-->
- **重みのシャーディング（総状態サイズ $16N/k$）：**使用していない間、各 GPU はすべてのブロックの重みの $1/k$ のみを保持

<!--en:ecf1f68b-->
- **ブロックごとのライフサイクル：**ブロック $\ell$ が**順伝播**を実行する前に、**all-gather** が一時バッファに重み全体を組み立て、ブロック $\ell$ の終了直後にそれを解放する。**逆伝播**では 2 回目の **all-gather** がブロック $\ell$ の重みを再構築し、続いてブロック $\ell$ の勾配の **reduce-scatter** と重みの即時解放を行う。

<!--en:f3069360-->
::: {.slide title="FSDP の通信コストとデータ並列比 1.5× の通信量"}

<!--en:cd44412f-->
[通信モデル]{.kicker}

<!--en:84b3ef16-->
- **DDP の $1.5\times$ の通信量：**FSDP は順伝播後に収集した重みを解放するため、逆伝播で追加の all-gather が 1 回発生する（集団通信フェーズが $3$ 回対 $2$ 回）

<!--en:bd6d5aef-->
- **$k$ の増加に伴うスケーリング則：**GPU を増やすと定常状態のメモリは分割されて減る（$16N/k \downarrow$）一方、GPU あたりの通信量は $1.5\,S$（$k=2$）から $3S$（$k \to \infty$）へと増加

<!--en:d3dd03aa-->
::: {.slide title="2 基の RTX 4090 における all-gather 帯域幅の計測"}

<!--en:37794ebb-->
[リンクの較正]{.kicker}

<!--en:60816398-->
36 ブロックの GPT を $k = 2$ 基の GPU で実行する前に、2 基の RTX 4090 間で $\beta_{\textrm{link}}$ を $256\text{ MiB}$ の all-gather によって較正する。

<!--en:af7e39d8-->
- **ホスト経由リンク速度の一致：**各 GPU は $(k-1)/k = 1/2$ にあたる $256\text{ MiB}$ 出力の半分（$128\text{ MiB}$）をバス帯域幅 $\sim 2.2\text{--}2.3\text{ GB/s}$（$\beta_{\textrm{link}} \approx 0.43\text{--}0.45\text{ ns/B}$）で送信し、§{sec_multi_gpu_concise} で計測した素の allreduce のリンク速度と一致

<!--en:378ce7de-->
::: {.slide title="1.9B GPT におけるメモリ・通信・計算時間の予測"}

<!--en:649eb15a-->
[ステップ予測]{.kicker}

<!--en:d0078a9c-->
次に、36 ブロック、$1.92\text{B}$ パラメータの GPT（$S = 4N = 7.7\text{ GB}$）を、$k = 2$ 基の GPU 上で GPU あたり $b = 2$ 本・長さ $T = 1{,}024$ トークンの系列とブロックごとの活性化チェックポイントを用いて 1 ステップ FSDP 実行したときのピークメモリ・通信時間・計算時間を予測する。

<!--en:ffae7134-->
- **メモリ予測（$\le 17\text{ GiB} < 24\text{ GiB}$）：**シャーディング状態 $16N/2 = 14.3\text{ GiB}$ + チェックポイント保存された活性化値（$\sim 1\text{ MiB/token} \times 2{,}048 \approx 2\text{ GiB}$）+ 展開中の 1〜2 ブロック（$\sim 0.4\text{ GiB}$）$\approx 16.7\text{ GiB}$

<!--en:061ae86e-->
- **時間予測（通信 $\sim 5\text{ s}$ 対計算 $0.4\text{ s}$）：**$k = 2$ では、各 GPU が $1.5\,S = 11.5\text{ GB}$ を $2.2\text{ GB/s}$ の PCIe リンク経由で送信し（$\sim 5.0\text{ s}$）、`tf32` 算術演算の $0.4\text{ s}$（チェックポイント保存込みでトークンあたり $8N$ FLOPs）を大きく上回る

<!--en:297a373a-->
::: {.slide title="PyTorch におけるメタデバイス初期化と fully_shard" only="pytorch"}

<!--en:98bce545-->
[PyTorch での FSDP]{.kicker}

<!--en:43f4b79d-->
起動時にどの単一 GPU 上でも $28.6\text{ GiB}$ の状態全体を確保せずに済むよう、メモリ消費ゼロの `'meta'` デバイス上でモデルを構築し、各ブロックを `fully_shard` でシャーディングしてから、`to_empty` でローカルシャードのみを実体化する。

<!--en:9a4fa79a-->
- **`fully_shard` で各ブロックを個別にラップする理由：**ブロック境界でラップすることで、一度に all-gather される対象が $50\text{M}$ パラメータの 1 ブロック（$\sim 200\text{ MB}$）のみとなり、$1.92\text{B}$ パラメータのモデル全体を集めずに済む

<!--en:a01b2fe4-->
::: {.slide title="マルチプロセス FSDP 起動時のホスト OOM 回避" only="pytorch"}

<!--en:081c51a0-->
[適切な初期化手順]{.kicker}

<!--en:e6288fc8-->
`torchrun` が単一のホストマシン上で $k$ 個のワーカープロセスを起動するとき、素朴にモデルを初期化すると FSDP が始まる前にホスト RAM や GPU VRAM が枯渇してクラッシュし得る。

<!--en:d6b2012a-->
- **落とし穴 1（`fully_shard` より前の GPU 上での実体化）：**`d2l.GPT(**config).to(device)` を `fully_shard` の前に呼ぶとカード上に $4N = 7.7\text{ GB}$ が確保され、より大規模なモデル（$7\text{B+}$）では即座に CUDA OOM が発生する

<!--en:25bbfbd2-->
- **落とし穴 2（ホスト RAM 上の $k$ 個のコピー）：**$k = 8$ 個すべてのプロセスが $30\text{ GB}$ のモデルを CPU RAM 上で同時に構築・初期化すると、ホスト DRAM を $8 \times 30 = 240\text{ GB}$ 消費する

<!--en:9474ff4b-->
- **実用パターン：**`torch.device('meta')` 上で構築 $\to$ ブロックごとに `fully_shard` を適用 $\to$ ローカルの $1/k$ GPU ストレージを `model.to_empty(device=device)` で確保 $\to$ シャーディングされた `DTensor` に直接初期化またはチェックポイントをストリーミング読み込み

<!--en:bdf836d1-->
::: {.slide title="JAX における宣言的なパラメータ・活性化値シャーディング" only="jax"}

<!--en:4307d10f-->
[JAX での FSDP]{.kicker}

<!--en:9256a6b5-->
JAX では、FSDP のために手動で all-gather や reduce-scatter のコードを書く必要はない。各パラメータテンソルを $k$ で割り切れる最初の軸に沿ってシャーディングし（`fsdp_sharding`）、各ブロックの活性化値の出力がバッチ方向のシャーディング（`P('data')`）を維持するよう制約する。

<!--en:d7362e20-->
- ブロックの重みが `'data'` 上でシャーディングされ、ブロックの活性化値*も* `'data'` 上でシャーディングされているため、XLA は行列積の前に各ブロックの重みの all-gather を、逆伝播では重み勾配の reduce-scatter を自動発行する

<!--en:b42d90d7-->
- **通信量の厳密な一致：**HLO の結果サイズ（送信バイト数の $2 \times$、$k=2$ 時）で見ると、all-gather は $2S \approx 15.0\text{ GB}$ を生成し（送信量は $S = 7.5\text{ GB}$）、reduce-scatter は $S/2 \approx 3.8\text{ GB}$ を生成し（送信量は $3.8\text{ GB}$）、合計送信量は $11.3\text{ GB} \approx 1.5\,S$ となる

<!--en:c803ae9c-->
::: {.slide title="XLA の GPU あたりメモリ計画とバッファ譲渡の確認" only="jax"}

<!--en:6be3765f-->
[コンパイラのメモリ計画]{.kicker}

<!--en:4b51524b-->
1 ステップも実行する前に `compiled.memory_analysis()` を呼び出すことで、XLA の静的メモリ計画が GPU あたり $24\text{ GiB}$ 以内に十分収まることを確認できる。

<!--en:30c55f2f-->
- **バッファ譲渡（`donate_argnums=(0, 1)`）：**シャーディングされた古い重みと Adam モーメントのバッファ（$10.7\text{ GiB} = 12N/2$）を更新後の状態用にインプレースで再利用可能としてマークする。バッファ譲渡がなければ、オプティマイザ更新中に新旧両方の状態を同時に保持するため $2 \times 10.7 = 21.4\text{ GiB}$ が必要となり OOM が発生する。

<!--en:f7c44630-->
::: {.slide title="FSDP ステップの実測（ピーク 16 GiB・通信 90%）"}

<!--en:dbb2c465-->
[実験による検証]{.kicker}

<!--en:c62bb76c-->
36 ブロック、$1.92\text{B}$ パラメータの GPT ステップを 2 基の $24\text{ GiB}$ RTX 4090 上で実行すると、メモリと実行時間の両方の予測が裏付けられる。

<!--en:02b7140a-->
- **メモリは余裕をもって収まる：**各 GPU のピークは **$\sim 16\text{ GiB}$**（予測上界の $17\text{ GiB}$ 未満）であり、単独のカードには収まらない状態サイズ $28.6\text{ GiB}$ のモデルを訓練可能

<!--en:8f6ea900-->
- **PCIe 上では通信が支配的：**各ステップは **$\sim 5.5\text{ s}$** かかり、GPU 0 は**ステップの $90\%$ 超（$\sim 5.0\text{ s}$）を NCCL 通信カーネル内で**費やし、算術演算はわずか $\sim 0.4\text{ s}$

<!--en:b31c925b-->
::: {.slide title="ホスト経由 PCIe と NVLink 5 ファブリックでの FSDP 比較"}

<!--en:1c7e2089-->
[インターコネクトへの感度]{.kicker}

<!--en:92c65c19-->
- **ホスト経由 PCIe 4.0（$\sim 2.2\text{ GB/s}$）：**$1.5\,S = 11.5\text{ GB}$ の移動に $\sim 5.0\text{ s}$ かかるのに対し計算は $0.40\text{ s}$ であり、ステップは強いリンク律速となる（計算比率 $\sim 7\%$）

<!--en:580b942d-->
- **NVLink 4/5（$450\text{--}900\text{ GB/s}$）：**同じ $11.5\text{ GB}$ が $13\text{--}26\text{ ms}$ で完了し、それらの GPU におけるステップ全体の計算時間（B200/H100 の `bf16` ピーク時で $14\text{--}32\text{ ms}$）を下回る

<!--en:5f61af62-->
- **ブロックのプリフェッチ：**ブロック $\ell+1$ の all-gather をブロック $\ell$ の計算とオーバーラップさせることで、$g \le c$ のとき通信を完全に隠蔽

<!--en:bd7a3f83-->
::: {.slide title="シャーディングと勾配累積付き複製の使い分け"}

<!--en:69e9d074-->
[選択の指針]{.kicker}

<!--en:4c286512-->
- **$16N$ が収まるとき（例：1B GPT を $24\text{ GiB}$ で訓練）：**手法には **DDP + 勾配累積**を選ぶ。全 $m$ 個のマイクロバッチが終わった後、GPU は**グローバルステップあたり 1 回だけ**通信

<!--en:9c457b05-->
- **勾配累積ステップで PCIe 経由の FSDP が不利な理由：**FSDP はブロックごとに重みを解放するため、**マイクロバッチごとに毎回 2 回**全重みを再収集する必要が生じる（PCIe 通信量が $2m\times$ に増加）

<!--en:a9e82535-->
- **段階的な移行指針：**メモリ要件に応じて初めて DDP $\to$ ZeRO-1/2 $\to$ ZeRO-3（FSDP）へと段階を上げる

<!--en:82d02841-->
::: {.slide title="理解度チェック（ZeRO のメモリ・FSDP の通信量・リンク）"}

<!--en:929b7671-->
[理解度の確認]{.kicker}

<!--en:dedd14db-->
3 つのシャーディングのシナリオを計算してみよう（$S = 4N$ は重みまたは勾配 1 セット分の `float32` コピーのバイト数）。

<!--en:f15dc987-->
1. $N = 4 \times 10^9$ パラメータ（`float32` Adam 状態で $16N = 64\text{ GB}$）を $k = 4$ 基の GPU で扱うとき、**DDP**、**ZeRO-1**、**ZeRO-2**、**ZeRO-3（FSDP）**のそれぞれで逆伝播後に各 GPU が保持する状態は何 GB か。

<!--en:f7419112-->
2. 混合精度 FSDP は `float32` のシャードを保持しつつ、重みの all-gather と勾配の reduce-scatter を `bfloat16`（$2\text{ B}$）で実行する。`float32` の DDP と比べて、ステップあたり・GPU あたり何バイトを送信するか。

<!--en:a2522324-->
3. FSDP を $k = 2$ 基の GPU から $k = 64$ 基の GPU へスケールさせるとき、GPU あたりの状態メモリは何倍になり、ステップあたりの GPU ごとの通信量は何倍になるか。

<!--en:557b98b6-->
**解答：**

<!--en:3da1bef2-->
1. GPU あたり **DDP（$16N$）：**$\mathbf{64\text{ GB}}$、**ZeRO-1（$(8 + 8/k)N = 10N$）：**$\mathbf{40\text{ GB}}$、**ZeRO-2（$(4 + 12/k)N = 7N$）：**$\mathbf{28\text{ GB}}$、**ZeRO-3 / FSDP（$16N/k = 4N$）：**$\mathbf{16\text{ GB}}$

<!--en:c582273f-->
2. **`float32` DDP より少ない通信量：**集団通信の dtype を `bfloat16` に半減すると $S_{\textrm{comm}}$ が $4N \to 2N = S/2$ と半分になるため、FSDP の 3 フェーズが移動させるデータ量は $\frac{3(k-1)}{k} \frac{S}{2} = \mathbf{1.5\,\frac{k-1}{k}S}$ バイトとなる（$25\%$ *少ない*通信量となり、`float32` DDP の $\frac{2(k-1)}{k}S$ を下回る）

<!--en:89b4cc07-->
3. **メモリは $32\times$ 削減**され（$16N/2 \to 16N/64$）、一方で **GPU あたりの通信量はほぼ $2\times$ に増加**する。具体的には $3(1/2)S = 1.5\,S$（$k=2$）から $3(63/64)S \approx 2.95\,S$（$k=64$）へと増える。

<!--en:9a53155d-->
::: {.slide title="まとめ（ZeRO Stage 1〜3 と FSDP のトレードオフ）"}

<!--en:c5a222f8-->
[要点]{.kicker}

<!--en:d851a9dd-->
| 手法 | $k$ 基の GPU への分割対象 | メモリ / パラメータ（`fp32` Adam） | ステップあたりの集団通信 | 送信バイト数 / GPU（$S = 4N$） |

<!--en:2e6299c4-->
| **標準 DDP** | ミニバッチのみ | $16\text{ B}$ | 1 回の allreduce | $\frac{2(k-1)}{k}S < 2S$ |

<!--en:7123e5aa-->
| **ZeRO Stage 1** | オプティマイザのモーメント（$\mathbf{m}, \mathbf{v}$） | $8 + \frac{8}{k}\text{ B}$ | 1 回の reduce-scatter + 1 回の all-gather | $\frac{2(k-1)}{k}S < 2S$ |

<!--en:62977595-->
| **ZeRO Stage 2** | モーメント + 勾配（$\nabla \mathbf{W}$） | $4 + \frac{12}{k}\text{ B}$ | バケット単位の reduce-scatter + 1 回の all-gather | $\frac{2(k-1)}{k}S < 2S$ |

<!--en:4432a1a5-->
| **ZeRO Stage 3（FSDP）** | モーメント + 勾配 + 重み（$\mathbf{W}$） | **$\frac{16}{k}\text{ B}$** | 2 回の all-gather + 1 回の reduce-scatter | **$\frac{3(k-1)}{k}S < 3S$** |

<!--en:7014f3c6-->
| **`bf16` 通信 FSDP** | 全状態（マスターシャードは `fp32`） | $\frac{16}{k}\text{ B}$（+ 展開中ブロック） | 2 回の `bf16` all-gather + 1 回の `bf16` RS | $\frac{1.5(k-1)}{k}S < 1.5S$ |
