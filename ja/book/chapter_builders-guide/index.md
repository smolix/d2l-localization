<!--en:f5b9081b-->
# 計算

<!--en:d5436961-->
ここまでは、モデルの構築、パラメータの管理、初期化、デバイス上での実行をフレームワークの裏側の処理に任せてきた。しかし、教科書レベルのプロトタイプから本番規模のシステムへと進むには、そのブラックボックスを開ける必要がある。80 億パラメータの言語モデルを 32 ビット浮動小数点の Adam で訓練する場合、活性化値を 1 つ保持する前の段階で、パラメータ、勾配、オプティマイザ状態（optimizer state）だけで 128 GB のメモリを要するうえ、数時間おきにハードウェア障害で訓練が中断される数千基の GPU 上で実行することになる。この規模では、フレームワークがモデルをどのように表現し、初期化し、計算に用い、永続化するかを理解することが不可欠である。

<!--en:6ddf6de1-->
本章のあらゆる内容は、1 つの中心的なデータ構造の上に築かれている。すなわち、ニューラルネットワークとは*モジュールの木*（tree of modules）であり、その葉がモデルの*状態*（state、訓練可能なパラメータと固定バッファの両方）を保持し、それぞれが根からのパスによって指定されるという構造である。本章の前半では、この構造的な抽象化を展開し、フレームワークの各操作がそれをどのように走査するかを示す。まず、モジュールの階層構造、コンテナへの登録、残差スキップ接続、形状の遅延決定（lazy shape inference）、設定駆動のビルダから始める（:numref:`sec_model_construction`）。続いて状態木を開き、訓練可能なパラメータと訓練対象外のバッファを切り分け、fp32 の Adam 訓練におけるパラメータあたり 16 バイトというメモリ下限を導出し、ストレージを確保せずにパラメータ数を数え上げ、重みの共有と凍結を扱う（:numref:`sec_parameters`）。次に、層ごとの分散規則から木全体および残差構造の初期化へと進み、スケーリングしない残差分岐が活性化値の分散を $(1 + g^2)^N$ 倍に増幅させる理由と、$1/\sqrt{N}$ またはゼロによる出力射影（output projection）のスケーリングが深い積層モデルを安定させる仕組みを示す（:numref:`sec_init_param`）。そのうえで、状態をもたない層、パラメータをもつ層（`RMSNorm`）、固定バッファをもつ層（`RandomFeatures`）、およびストレートスルー推定器（straight-through estimator）によるカスタム逆伝播規則を用いてフレームワークを拡張する（:numref:`sec_custom_layer`）。

<!--en:9959aaf9-->
後半では、モデルの構造から数値表現、永続化、ハードウェア実行、および診断へと目を向ける。まず fp32、fp16、bf16 を比較し、16 ビットの重みでは微小な更新が丸めによって消えてしまう理由を説明したうえで、混合精度訓練（mixed-precision training）がいかに fp32 のマスター重みと 16 ビット演算、および fp16 の損失スケーリング（loss scaling）を組み合わせるかを示す（:numref:`sec_numerics`）。次に、安全でない `pickle` アーカイブとコードを含まない `safetensors` ファイルを対比し、ビット単位で完全に再開できるアトミックな 5 要素の訓練チェックポイントを構築するとともに、事前学習済みバックボーンを読み込む際の欠落キーや予期しないキーを検査する（:numref:`sec_read_write`）。最後に、テンソルとモデルを GPU 上に配置し、非同期コマンドキューの実行時間を正確に計測し、$O(\sqrt{K})$ の活性化チェックポイント（activation checkpointing）とあわせて GPU メモリの内訳を詳しく調べ（:numref:`sec_use_gpu`）、擬似乱数シードの制御および GPU の決定論的実行を非侵襲的な順伝播・逆伝播フックと組み合わせる（:numref:`sec_repro`）。

<!--en:5dd576ec-->
## 参考資料と関連文献 {.unnumbered}

<!--en:a64691d2-->
以下の資料は本章の各節に対応しており、それぞれ関連する節を併記している。特記のない限り、すべてオンラインで無料公開されている。

<!--en:f3e47eb0-->
**書籍**

<!--en:d3974bde-->
- [Deep Learning with PyTorch (Stevens, Antiga & Viehmann)](https://web.archive.org/web/20211012030609/https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)：PyTorch チームによる無料の PDF 版（アーカイブ版）。第 1 部でテンソル、ストレージ、自動微分、`nn.Module` を :numref:`sec_model_construction` および :numref:`sec_parameters` と同じ水準で解説している。

<!--en:d790fd9e-->
- [Machine Learning Systems (Vijay Janapa Reddi)](https://mlsysbook.ai/)：オンラインで無料公開。フレームワーク、データパイプライン、訓練インフラストラクチャ、効率的なデプロイメントなど、本章を取り巻くシステム面の視点を扱っている（:numref:`sec_use_gpu`、:numref:`sec_repro`）。

<!--en:51311a3d-->
**講義コースと講義動画**

<!--en:c7ced7f4-->
- [CMU 10-414/714: Deep Learning Systems (Chen & Kolter)](https://dlsyscourse.org/)：無料の講義と課題。自動微分、モジュール、初期化、GPU サポートを備えた小型フレームワーク「needle」を受講生自ら構築する（:numref:`sec_model_construction`、:numref:`sec_init_param`、:numref:`sec_use_gpu`）。

<!--en:7dee2f99-->
- [Neural Networks: Zero to Hero (Andrej Karpathy)](https://karpathy.ai/zero-to-hero.html)：無料の動画シリーズ。標準の Python のみを用いて、自動微分、モジュール、訓練ループをゼロから構築する（:numref:`sec_model_construction`、:numref:`sec_custom_layer`）。

<!--en:b066c45d-->
- [fast.ai Part 2: Deep Learning Foundations (Howard et al.)](https://course.fast.ai/Lessons/part2.html)：無料の講義コース。モジュール、初期化、混合精度を含め、テンソル演算から積み上げて訓練フレームワークを再構築する（:numref:`sec_model_construction`、:numref:`sec_init_param`、:numref:`sec_numerics`）。

<!--en:b24c3731-->
**チュートリアル・解説記事・インタラクティブ教材**

<!--en:84cbfaef-->
- [PyTorch internals (Edward Yang)](http://blog.ezyang.com/2019/05/pytorch-internals/)：PyTorch 内部のテンソル、ストライド、ディスパッチ、自動微分を順を追って解説する記事であり、:numref:`sec_model_construction` より 1 段低い抽象度を扱っている。

<!--en:565210e5-->
- [JAX: The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)：純粋関数、明示的な PRNG キー、JIT の制約など、JAX の背後にある関数型設計を実践上の注意点としてまとめた解説である（:numref:`sec_parameters`、:numref:`sec_repro`）。

<!--en:599e8531-->
- [Train With Mixed Precision (NVIDIA)](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)：fp16/bf16 演算と損失スケーリングに関するベンダー公式ガイドであり、:numref:`sec_numerics` はその要点を凝縮したものである。

<!--en:57c07d87-->
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (David Goldberg)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)：あらゆる丸めやオーバーフローの問題（:numref:`sec_numerics`）の根底にある標準的な参考文献である。

<!--en:3dcc9c62-->
- [Making Deep Learning Go Brrrr From First Principles (Horace He)](https://horace.io/brrr_intro.html)：演算能力、メモリ帯域幅、フレームワークのオーバーヘッドを第一原理から解き明かす解説記事である（:numref:`sec_use_gpu`）。

<!--en:d8f41a16-->
- [safetensors (Hugging Face)](https://huggingface.co/docs/safetensors/index)：読み込み時にコードを実行せずに重みを共有するためのファイル形式と、それが pickle ベースのチェックポイントを置き換えた理由を解説するドキュメントである（:numref:`sec_read_write`）。

<!--en:00a668af-->
- [Reproducibility (PyTorch notes)](https://pytorch.org/docs/stable/notes/randomness.html)：決定論的実行のためのフラグ、その計算性能上のコスト、および限界についての解説であり、:numref:`sec_repro` を補完する。

<!--en:d63f4e40-->
**基礎となった論文**

<!--en:6dad5c08-->
- [Automatic Differentiation in Machine Learning: a Survey (Baydin, Pearlmutter, Radul & Siskind, 2018, *JMLR*)](https://jmlr.org/papers/v18/17-468.html)：フォワードモードおよびリバースモードの自動微分に関する決定版のサーベイ論文であり、:numref:`sec_custom_layer` のあらゆるカスタム勾配の背後にあるアルゴリズムを解説している。

<!--en:f8791fdd-->
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740)：fp16・マスター重み・損失スケーリングを組み合わせたレシピ（:numref:`sec_numerics`）の原典となった論文である。

<!--en:303a71d3-->
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library (Paszke et al., 2019)](https://arxiv.org/abs/1912.01703)：即時実行、自動微分、キャッシングアロケータなど、:numref:`sec_model_construction` および :numref:`sec_use_gpu` の命令型スタイルを支える設計思想を述べた論文である。

<!--en:8c19e674-->
深層学習の計算：プロトタイプコードから本番システムへ<br>**モジュール木・パラメータ状態・初期化・カスタム層・数値表現・チェックポイント・GPU・再現性**

<!--en:7ca5d243-->
::: {.slide title="4 配列の MLP から 8B 基盤モデルへのスケーリング"}

<!--en:961550ab-->
[システム面の課題]{.kicker}

<!--en:f970f620-->
- **メモリ容量（下限 `128 GB`）：**8B の重みの保持は `16 GB`（`bfloat16`）で済むが、`float32` の Adam 訓練では活性化値を 1 つも計算する前の段階でパラメータ、勾配、オプティマイザのモーメントに `128 GB` を要する

<!--en:c3e2c46f-->
- **数値安定性と精度：**スケーリングしない残差分岐は 32 ブロックで活性化値の分散を $2^{32}$ 倍に爆発させ、16 ビットの重みは微小な更新を暗黙のうちに丸め捨てる

<!--en:075308b7-->
- **耐障害性、セキュリティ、非同期 GPU：**アトミックなチェックポイント（`96 GB`）、`safetensors` と `pickle` の任意コード実行（RCE）リスクの対比、同期された GPU タイマー、非侵襲的な層フック

<!--en:798ec937-->
::: {.slide title="統一的な抽象化：モデルはモジュールの木"}

<!--en:fc6976eb-->
[中核となるメンタルモデル]{.kicker}

<!--en:127806f1-->
- **構造化されたパス指定：**すべてのパラメータと訓練対象外のバッファは、根モジュールからのパス（例：ブロック 1、層 0、重み）によって一意に指定される

<!--en:c82ab8fd-->
- **統一的な木の走査：**オプティマイザの設定、初期化、`safetensors` によるシリアライズ、dtype キャスト、GPU 配置、フックはいずれも登録済みの子モジュールを再帰的にたどる

<!--en:d30ce925-->
- **暗黙の登録不変条件：**モジュールレジストリ外の通常の Python 属性やリストに格納された層やテンソルは、オプティマイザ、チェックポイント、GPU から見えない

<!--en:347e847f-->
::: {.slide title="第 §{num:chap_computation} 章のロードマップ：アーキテクチャ・状態・実行"}

<!--en:8b8a96b7-->
[アーキテクチャとシステムのロードマップ]{.kicker}

<!--en:e20c1d9c-->
- **モデルの構造と状態（§{sec_model_construction}〜§{num:sec_custom_layer}）：**モジュール木と `__setattr__` による登録、$16\text{ B/param}$ の `fp32` Adam メモリ下限、重み結合、$(1+g^2)^N$ の残差分散に対するスケーリング（$g = 1/\sqrt{N}$）、`RMSNorm`、固定バッファ、ストレートスルー推定器

<!--en:392bb4e2-->
- **精度、I/O、ハードウェア（§{sec_numerics}〜§{num:sec_repro}）：**`fp32` マスター重みと `bf16`・`fp16` 演算、ゼロコピーの `safetensors` と 5 要素のアトミックなチェックポイント、非同期 GPU キュー、$O(\sqrt{K})$ の活性化チェックポイント、PRNG のシード設定、順伝播・逆伝播フック

<!--en:5b6075a6-->
::: {.slide title="本章を通じた具体的なベンチマーク"}

<!--en:418d9f93-->
[定量的な基準点]{.kicker}

<!--en:84dc51a2-->
- **パラメータとメモリのスケーリング（§{sec_model_construction}〜§{num:sec_parameters}、§{sec_numerics}）：**トイ 4 ブロック残差 MLP（$729{,}866$ パラメータ）と Llama 3 8B（$291$ テンソル、`16 GB` の `bf16` 推論、`128 GB` の `fp32` Adam 訓練メモリ下限）の対比

<!--en:18c93e45-->
- **信号の安定性と I/O（§{sec_init_param}、§{sec_read_write}）：**$g = 1/\sqrt{N}$ は 32 ブロックでの標準偏差の増大を $\sqrt{e} \approx 1.65\times$ に抑え（未スケーリング時は $65{,}536\times$）、5 要素からなる `96 GB` のチェックポイントはビット単位で完全に再開する

<!--en:7b14e43e-->
- **活性化メモリ（§{sec_use_gpu}）：**$K = 16$ ブロックを $s = \sqrt{K} = 4$ ごとに分割すると、ステップあたりのピーク RAM が $1.1\text{ GiB} \to 0.45\text{ GiB}$ に削減される

<!--en:15d021a2-->
::: {.slide title="本章の読み進め方"}

<!--en:72fdbd10-->
[学習とエンジニアリングのガイド]{.kicker}

<!--en:5ba38583-->
各節では、複数フレームワークで実行可能なコードと、気づきにくい訓練の失敗を防ぐシステム診断を組み合わせている。

<!--en:93729ff5-->
- **すべての読者向けの基本ルート：**まず **§{sec_model_construction}〜§{sec_init_param}** でモジュール登録、パラメータとバッファの計算、残差分散のスケーリングを習得し、続いて **§{sec_numerics}〜§{sec_use_gpu}** で混合精度、安全なチェックポイント保存、GPU メモリ管理を学ぶ

<!--en:39e92d3c-->
- **フレームワーク固有のイディオム：**使用するフレームワークがリスト型コンテナ、遅延形状、PRNG の状態、計算グラフのコンパイルをどのように扱うかに注意する

<!--en:4b6f6be4-->
- **すぐに身につけるべき診断習慣：**ジョブを投入する前にパラメータと活性化値のバイト数を計算し、1 回の順伝播で初期の活性化値の標準偏差を検証し、外部の重みを読み込む際は常に欠落キーや予期しないキーを検査する
