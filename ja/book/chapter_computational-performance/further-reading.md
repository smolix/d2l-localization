<!--en:7902ddfd-->
# 関連文献

<!--en:cfd5025b-->
本章の各節では、それぞれの考え方について代表的な例を 1 つずつ扱ってきた。すなわち、1 つのメモリ帯域幅を持つ 1 つのルーフライン、1 基の GPU とそのリンク、フレームワークごとに 1 つのコンパイラ、1 枚のカード上でメモリを節約する 3 つの方法、1 つの集合通信アルゴリズム（collective algorithm）、モデルの状態を GPU 間で分割する 1 つの方法、そして 1 台のマシン上で高速化した 1 つの Transformer である。オープンアクセスの 2 冊の書籍が、本章全体の格好の伴侶となる。『[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)』（Austin et al., 2025）は TPU と GPU におけるルーフライン、シャーディング記法、および集合通信のコストを導出しており、『[The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)』（Hugging Face, 2025）はメモリバジェット、データ並列、ZeRO、およびマルチノード並列性を実測値とともに概説している。:numref:`sec_perf_model` の 3 つのレジームに関する手短な入門としては、Horace He のエッセイ :cite:`He.2022` が参考になる。

<!--en:300df661-->
## 計測とモデル化

<!--en:f256a612-->
### より精緻な性能モデル

<!--en:896b73c9-->
:numref:`sec_perf_model` のルーフライン :cite:`Williams.Waterman.Patterson.2009` は、上下どちらの方向にも誤差を生じる。一部のカーネルがその帯域幅ルーフを上回って動作するのは、72 MB の L2 キャッシュや共有メモリに収まるテンソルが $\beta$ より速く到達するためである。キャッシュ考慮型ルーフライン（cache-aware roofline）:cite:`Ilic.Pratas.Sousa.2014` はメモリ階層の各レベルごとに 1 本の帯域幅ルーフを引き、:citet:`Yang.Wang.Farrell.ea.2020` は深層学習カーネルについてこれらのレベルごとのバイト数をハードウェアカウンタから読み取っている。逆に、同時に移動中のバイト数（bytes in flight）が少なすぎるためにルーフを下回るカーネルもある（リトルの法則、:numref:`subsec_hw-gpu`）。:citet:`Volkov.2016` は、1 つのスレッド内の独立した命令が追加のワープの代わりになり得ること、そして従来の GPU 性能モデルがリトルの法則を誤って適用した結果、スループットを最大 1.7 倍誤って予測していたことを示している。

<!--en:12f50225-->
### デバイスタイマとシステムプロファイラ

<!--en:977e410a-->
`d2l.Benchmark`（:numref:`sec_perf_model`）はクロックを読み取る前にキューを空にするため、実際の訓練ループでは速度低下を招く。CUDA イベント（`torch.cuda.Event`）はキューを空にすることなく GPU ストリーム上でタイムスタンプを記録するが、その区間には GPU がホストを待機していた空白時間も含まれる。:numref:`subsec_perf-profiler` のプロファイラは、1 つのプロセスについてビジー時間とアイドル時間を分離し、Perfetto 用のトレースを書き出す。NVIDIA の Nsight Systems は全プロセスにわたる OS スレッド、メモリコピー、NCCL 呼び出しを 1 つのタイムライン上に配置し、一方 Nsight Compute は単一のカーネルを両方のルーフに照らして計測する。

<!--en:e288fbfc-->
### 入力パイプライン

<!--en:e06cf234-->
画像や音声で訓練する読者は、Python のオーバーヘッドよりも先にデータローダの供給不足（starvation）に直面することが多い。次のミニバッチを CPU がデコードしてデータ拡張するあいだ、GPU が待機してしまうからである。ホストの空きコア数が GPU の必要とする数より少ない場合（:numref:`subsec_hw-cpu`）、デコード処理を GPU へ移すことができる（たとえば NVIDIA DALI を用いる）。:citet:`Mohan.Phanishayee.Raniwala.ea.2021` は複数の訓練ジョブにわたるデータストール（data stall）を分析し、訓練の特性を考慮したキャッシュ（training-aware cache）とハイパーパラメータスイープ間での前処理の共有によって、ストール時間の大部分を解消している。

<!--en:62ae0657-->
### 利用率と目標損失までの時間

<!--en:f311bb68-->
ケーススタディの締めくくりで行ったカーネルの内訳分析（:numref:`subsec_ft-account`）は、:citet:`Ivanov.Dryden.Ben-Nun.ea.2021` が BERT に対して実施したものであり、行列積のあいだにある帯域幅律速のカーネルを融合することでエンコーダ層を $1.3\times$ 高速化した。ケーススタディの `MFU` 列は、大規模な訓練実行で報告される標準的な評価指標である :cite:`chowdhery2022palm`。*ハードウェア FLOPs 利用率*（hardware FLOPs utilization、HFU）は再計算された FLOPs も数え上げるため、活性化チェックポイントのもとでは MFU を上回る :cite:`Korthikanti.Casper.Lym.ea.2022`。モデルが本来必要とする計算のみを評価対象とするため、システム間を公平に比較できるのは MFU だけである。

<!--en:7ebe969a-->
毎秒トークン数によるスループットは、たとえ収束により多くのステップを要する変更であっても、1 ステップを短縮するあらゆる変更を高く評価してしまう。そのため modded-nanoGPT スピードラン :cite:`Jordan.2024` では、8 基の GPU 上で GPT を一定の検証損失まで訓練するのにかかる実経過時間によって変更を順位付けしており、コンパイル済みカーネルや低精度形式を Muon オプティマイザ（:numref:`sec_muon`）と組み合わせている。

<!--en:a2dc8fea-->
## ハードウェアと数値形式

<!--en:a8b82c6f-->
### その他のアクセラレータ

<!--en:7cc73e82-->
ルーフライン解析は、:numref:`tab_gpu_specs` 以外のアクセラレータにもそのまま適用できる。AMD の Instinct MI355X、Google の TPU7x、Intel の Gaudi 3、AWS の Trainium2 はいずれも、積層 HBM と行列演算ユニットを組み合わせている。これらの bf16 におけるリッジポイントは約 230 FLOP/byte（Trainium2）から 450 超（Gaudi 3）に及び、本章の RTX 4090 における 164 FLOP/byte の $1.4\times$ から $3\times$ に達するため、演算融合やビット幅の狭い形式がいっそう重要になる。

<!--en:11d5e0c1-->
### ユニファイドメモリ

<!--en:261372a5-->
重みがメモリに収まる限り、シングルストリームのデコーディングは、メモリ帯域幅を重みのバイト数で割った速度を超えて動作することはない（:numref:`subsec_hw-two-workloads`）。Apple の M3 Ultra は、最大 512 GB のメモリを CPU と GPU のあいだで 819 GB/s の帯域幅で共有する。8 ビット表現の 70B モデル（約 70 GB）は 24 GB のカードには収まらないが、M3 Ultra 上では帯域幅の上限から毎秒約 12 トークンの生成が可能になる。このようなユニファイドメモリ搭載マシンは、計算ルーフの低さと引き換えに大容量を備えており、訓練よりもシングルストリームの推論に適している。

<!--en:12a312f2-->
### ブロックスケール形式の詳細

<!--en:9ca89418-->
:numref:`subsec_hw-gpu` のブロックスケールは、わずかなビット数のオーバーヘッドを伴う。OCP マイクロスケーリング形式 :cite:`Rouhani.Zhao.More.ea.2023` では、32 個の値で共有される 8 ビットの 2 のべき乗スケールにより、値あたり 0.25 ビットが追加される。NVIDIA の NVFP4 は、16 個の値で共有する fp8 スケール（値あたり 0.5 ビット）に加えてテンソルごとに 1 つの fp32 スケールを持つことで、2 倍単位の範囲内にとどまるのではなく、各ブロックの最大値を 6（fp4 の範囲の上限）の近くに写像できる。いずれの方式でも、4 ビットの重みや活性化値に必要なのは多くても 4.5 ビットである。

<!--en:4234eebd-->
## コンパイラとカーネル

<!--en:4246a638-->
### 融合カーネルの記述

<!--en:342bd9f6-->
:numref:`sec_compilation` のコンパイラは、要素ごとの演算の連鎖やリダクションを融合するが（PyTorch 2 については :cite:`Ansel.Yang.He.ea.2024`）、FlashAttention のようなタイル化を伴うアルゴリズム自体の書き換えまでは見つけ出せない。Triton :cite:`Tillet.Kung.Cox.2019`（および JAX の Pallas）は、テンソルタイルをプログラミングの単位とする。すなわち、実装者がタイリングとオンチップバッファリングを指定し、コンパイラがスレッド、ベクトル命令、共有メモリを管理する。FlashAttention-2 :cite:`Dao.2024` はクエリタイルにわたって並列化し、FlashAttention-3 :cite:`Shah.Bikshandi.Zhang.ea.2024` は Hopper GPU 上でデータ移動をテンソルコアの積演算および指数計算とオーバーラップさせるとともに、fp8 サポートを追加している。

<!--en:090623ed-->
### 大規模環境でのコンパイル時間

<!--en:d17843bc-->
小さなステップのコンパイル（:numref:`subsec_comp-wholestep`）は数秒で済むが、大規模なモデルのコンパイルにはプロセスおよび形状ごとに数分かかることがある。領域コンパイル（regional compilation）は、繰り返されるブロック（Transformer 層など）をすべてのコピーに対して 1 回だけコンパイルし（`torch.compiler.nested_compile_region`）、永続キャッシュはコンパイル済みアーティファクトをプロセス間で共有する（`torch.compiler.save_cache_artifacts`、`jax_compilation_cache_dir`）。オートチューニング（PyTorch の `mode='max-autotune'`、XLA では標準搭載）は、コンパイル時に複数の行列積タイリングをベンチマークし、初回のコンパイル時間と引き換えに各ステップを高速化する（:numref:`subsec_comp-wholestep`）。

<!--en:6f6ab4d5-->
### 訓練におけるカーネルリプレイ

<!--en:578fa2da-->
記録された CUDA グラフは、形状とメモリアドレスを固定する（:numref:`subsec_comp-overhead`）。`mode='reduce-overhead'` の背後にある CUDA グラフツリーは、共有メモリプール内で形状ごとに 1 回ずつ順伝播と逆伝播を記録する。ステップをまたいで保持されるテンソル（ログに記録した損失など）は、クローンしておかない限り次のリプレイ時に上書きされる。また、ランタイムがステップ境界を自動検出できない場合には、`torch.compiler.cudagraph_mark_step_begin` で明示的に印を付ける。

<!--en:f418163a-->
### エクスポートと初期のグラフ設計

<!--en:5ae5488c-->
キャプチャされたグラフは Python の外部でも実行できる。`torch.export` はグラフブレイクを一切許さずに厳密な全体グラフをキャプチャし、JIT コンパイルされた JAX 関数はモバイル、Web、およびサービング用ランタイムへのデプロイに向けて StableHLO へと変換（lowering）される。Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` や TensorFlow 1 :cite:`Abadi.Barham.Chen.ea.2016` といった初期のフレームワークでは、ユーザーが静的なシンボリックグラフを明示的に構築する必要があったが、トレーシングコンパイラは通常の即時実行（eager）コードから同じプログラム全体のグラフを復元する :cite:`Frostig.Johnson.Leary.2018`。

<!--en:243c0297-->
## メモリ

<!--en:ea815006-->
### より小さなオプティマイザ状態

<!--en:5899876d-->
:numref:`subsec_mp-anatomy` におけるパラメータあたり 16 バイトのうち、Adam の 2 つのモーメント推定値は 8 バイトを占める。これらを GPU 間でシャーディングする（:numref:`sec_sharding`）代わりに、8-bit Adam :cite:`Dettmers.Lewis.Shleifer.ea.2022` は両方のモーメントをブロックごとに量子化して正解率を損なうことなくパラメータあたり 2 バイトに抑える。一方、Adafactor :cite:`Shazeer.Stern.2018` は一次モーメントを省略し、二次モーメントを行和と列和へと分解（factorize）する。

<!--en:ed1ab25e-->
### ホストメモリとディスクへのオフロード

<!--en:11df9f73-->
ホスト DRAM は GPU メモリよりも容量が 1 桁大きく、かつ安価である。ZeRO-Offload :cite:`Ren.Rajbhandari.Aminabadi.ea.2021` は、fp32 のマスター重みと Adam のモーメントをホストメモリに保持してオプティマイザの更新を CPU 上で実行し、GPU 上には 16 ビットの重みと勾配のみを残す。さらに ZeRO-Infinity :cite:`Rajbhandari.Ruwase.Rasley.ea.2021` は、オフロード先を NVMe ドライブまで拡張する。

<!--en:f5658cf5-->
### 再計算対象の選択

<!--en:33201ec9-->
ブロックの入力のみを保存する手法（:numref:`subsec_mp-checkpointing`、:cite:`Chen.Xu.Zhang.ea.2016`）は、トレードオフ曲線上の 1 点にすぎない。二項チェックポイント法（binomial checkpointing）:cite:`Griewank.Walther.2008` は連鎖構造において保存する状態数を固定したもとで再計算量を最小化し、Checkmate :cite:`Jain.Jain.Nrusimha.ea.2020` は一般の計算グラフを整数線形計画法によって解く。現代のコンパイラもまた、選択的再計算を自動的に適用するか（PyTorch における AOTAutograd の最小カットパーティショナ）、明示的なポリシー（JAX の `jax.checkpoint_policies`）を通じて適用する。

<!--en:3062a30b-->
### 断片化と事前割り当て

<!--en:ef607908-->
ステップごとにテンソルの形状が変化すると、PyTorch のキャッシングアロケータが空きメモリを非連続な断片へと分割してしまうことがある（予約済みだが未割り当てを意味する「reserved but unallocated」）。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定すると、セグメントをインプレースで拡張できるようになる。JAX では、デフォルトのアロケータが起動時に GPU メモリの 75% を事前割り当てする（`XLA_PYTHON_CLIENT_PREALLOCATE=false` で事前割り当てを無効化でき、`XLA_PYTHON_CLIENT_ALLOCATOR=vmm` で CUDA の仮想メモリ管理を使用する）。

<!--en:afbbe1cd-->
## 通信とスケール

<!--en:850e4cd9-->
### リングを超える集合通信アルゴリズム

<!--en:7e0cc107-->
リング allreduce（:numref:`subsec_mg-ring`）は移動する総バイト数を最小化するが、$2(k-1)$ 回の逐次ホップを要する。ツリーアルゴリズムはホップ数を $\log_2 k$ のオーダーに抑え、NCCL は相補的な 2 つの二分木を組み合わせることで、すべてのデバイスが一方の木では葉となり、もう一方の木では内部ノードとなるようにしている。$k$ が 2 のべき乗のとき、再帰的半減・倍増法（recursive halving and doubling）はリングと同じバイト通信量を $2\log_2 k$ ステップで達成する :cite:`Thakur.Rabenseifner.Gropp.2005`。

<!--en:5f7128b4-->
### パラメータサーバーと非同期更新

<!--en:20c318c1-->
集合通信ライブラリが標準となる以前、分散訓練ではパラメータサーバー上で勾配を集約し :cite:`Dean.Corrado.Monga.ea.2012,Li.Andersen.Park.ea.2014`、非同期更新やローカル SGD を支えていた :cite:`Stich.2019`。古い勾配（stale gradient）は最適化のダイナミクスを変化させてしまうため、現在では密なモデルに対して同期型の allreduce や ZeRO（:cite:`Li.Zhao.Varma.ea.2020,Rajbhandari.Rasley.Ruwase.ea.2020`）が標準となっている一方、巨大な疎の推薦テーブル（:numref:`chap_recsys`）では依然としてパラメータサーバーが広く使われている。

<!--en:8defd137-->
### 並列化インターフェースと単一マシンを超える拡張

<!--en:31051db4-->
PyTorch は各並列化戦略をデバイスメッシュ上の組み合わせ可能な API（`DistributedDataParallel`、`fully_shard`、`torch.distributed.tensor.parallel`、`torch.distributed.pipelining`）として公開しているのに対し、JAX は GSPMD / Shardy :cite:`Xu.Lee.Chen.ea.2021` でコンパイルされる名前付きメッシュ上の配列配置によってそれらすべてを宣言的に表現する。マルチノードのテンソル並列・パイプライン並列・エキスパート並列 :cite:`shoeybi2019megatron,Narayanan.Shoeybi.Casper.ea.2021,Lepikhin.Lee.Xu.ea.2021`（:numref:`chap_lm_parallelism`）、fp8 訓練 :cite:`Micikevicius.Stosic.Burgess.ea.2022,Liu.Feng.Xue.ea.2024`（:numref:`sec_lm_low_precision`）、連続バッチングと PagedAttention :cite:`Yu.Jeong.Kim.ea.2022,Kwon.Li.Zhuang.ea.2023`（:numref:`chap_lm_serving`）、および重みの量子化（:numref:`chap_lm_compression`）については、「言語モデル」パートで詳しく扱う。

<!--en:191246a0-->
[ディスカッション](https://d2l.discourse.group/t/36854)
