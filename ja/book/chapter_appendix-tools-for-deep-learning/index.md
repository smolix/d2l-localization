<!--en:8fb4d2dc-->
# 深層学習のためのツール

<!--en:52543bfb-->
大半の読者はまず、ノート PC 上やブラウザのタブ内で本書のコードを実行する。ノートブックを開き、セルを実行し、学習率を変更して再実行するといった具合である。しかし遅かれ早かれ、手元の環境には収まらない事態が生じる。モデルがノート PC の搭載量を超えるメモリを必要としたり、訓練の実行に 1 週間かかったり、週末のプロジェクト用にダウンロードしたオープンな言語モデルが毎秒 3 語しか書き出せなかったりするのである。これらはいずれも、どの製品を使うべきかという問いに見える。だがその根底では、同じ少数の問いが繰り返し現れる。ジョブはメモリに収まるか、何がその速度を制限しているか、そして完了した結果を得るのにどれだけのコストがかかるか。

<!--en:35dc7d83-->
製品は数か月のうちに移り変わるが、その答えはより長く通用する少数の原則に基づいている。そもそも実行できるかどうかを決めるのはメモリであるため、高速化を図る前にまずジョブがメモリに収まるかを確認する。言語モデルがテキストを生成する際には、新しいトークンごとに、使用するすべての重みをメモリから 1 回読み出すため、メモリ帯域幅（memory bandwidth）が生成速度を決める。時間あたりの料金が最も安いマシンであっても、ジョブの実行が遅かったり途中で失敗したりすれば最も安上がりな実験にはならないため、完了した結果あたりのコストを比較する。さらに、記録された内容から再実行できない結果は、検証することもその上に積み上げることもできないため、再現性（reproducibility）は最初のセルからワークフローに組み込まれていなければならない。:numref:`chap_performance` では、プログラムは算術演算、メモリ帯域幅、あるいは演算を発行するオーバーヘッドによって律速されていた。以上の原則は、何を実行し、借り、購入し、デプロイするかという判断にそのモデルを適用したものである。

<!--en:f8af5c3c-->
本章では、単一の対話型ノートブックから複数マシンでの訓練、そして本番サービングへと段階的に進む。まずは日常的にコードを実行する環境から始める。結果が隠れた実行履歴に左右されないよう Jupyter カーネルの状態とローカルの `uv` 環境を管理する方法（:numref:`sec_interactive_development`）、ローカルの環境構築なしで Colab や Kaggle 上でノートブックを実行する方法（:numref:`sec_hosted_notebooks`）、そしてレンタルしたクラウドアクセラレータの料金を完了した結果あたりの総コストで見積もる方法（:numref:`sec_cloud_instances`）を扱う。次に、ハードウェアと外部アーティファクトに目を向け、メモリ容量・メモリ帯域幅・演算性能の上限に基づいて購入すべき機材を選ぶ方法（:numref:`sec_hardware_buyers`）と、公開ハブにある事前学習済みモデルやデータセットを探索・評価・ピン留め（pinning）し、ライセンスを確認する方法（:numref:`sec_software_ecosystem`）を解説する。最後に、複数のマシンや貢献者へと規模を広げ、複数のアクセラレータに訓練を分割する手法（:numref:`sec_training_systems`）、目標レイテンシの下で同時接続ユーザーにサービングする手法（:numref:`sec_model_serving`）、そしてシングルソースの Markdown ファイルから本書自体がどのようにビルド・保守されているか（:numref:`sec_developers_guide`）を見ていく。

<!--en:7da7f1cb-->
価格、製品名、クォータ、ライブラリのバージョンは急速に変化する。これらを記載している各節では時点を明記しているため、利用前に最新の情報を確認されたい。

<!--en:604fe18a-->
生成されたユーティリティおよび `d2l` API のドキュメントは、引き続き検索可能な HTML リファレンスページとして提供されている。これらは通読する学習順序や PDF 版には含まれない。

<!--en:5dd576ec-->
## 参考資料と関連文献 {.unnumbered}

<!--en:71077584-->
以下の資料は、1 台のマシンでの作業から大規模な訓練とサービングに至るまで、本章の各節に対応している。いずれもオンラインで無料公開されている。

<!--en:1a925603-->
**システムとスケーリング**

<!--en:3474c6d3-->
- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/)：実測の前に性能を予測する方法を解説したオンライン書籍であり、ルーフライン、TPU と GPU の内部構造、並列性、および推論の算術を、:numref:`sec_hardware_buyers`、:numref:`sec_training_systems`、:numref:`sec_model_serving` に対応する具体的な数値とともに展開している。

<!--en:783b25e3-->
- [The Ultra-Scale Playbook — Hugging Face (2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：最大 512 基の GPU を用いた 4,000 回以上の実測ランに基づく GPU クラスタでの訓練ガイドであり、詳細なメモリバジェットと並列性のトレードオフによって :numref:`sec_training_systems` の内容を掘り下げている。

<!--en:1cb373fe-->
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)：トークン化から分散訓練、推論に至るまで、言語モデルをエンドツーエンドで構築する YouTube 上の講義と GitHub 上の課題であり、:numref:`sec_training_systems` および :numref:`sec_model_serving` の副教材となる講義コースである。

<!--en:98115f0f-->
- [GPU MODE lectures](https://github.com/gpu-mode/lectures)：CUDA の基礎から FlashAttention、量子化までを網羅した動画、スライド、およびノートブックであり、:numref:`sec_hardware_buyers` で限界を見積もる対象となるカーネルを自ら書きたい読者に適している。

<!--en:38908ea7-->
**計算性能の考え方**

<!--en:a4fc976e-->
- [Making Deep Learning Go Brrrr From First Principles — Horace He (2022)](https://horace.io/brrr_intro.html)：プログラムの実行時間を演算、メモリ帯域幅、およびオーバーヘッドに切り分ける短い論考であり、:numref:`sec_hardware_buyers` の土台となる考え方である。

<!--en:46a39da1-->
- [Transformer Inference Arithmetic — kipply (2022)](https://kipp.ly/p/transformer-inference-arithmetic)：KV キャッシュのサイズとデコーディングの帯域幅上限を手計算で導出しており、:numref:`sec_hardware_buyers` および :numref:`sec_model_serving` における見積もりの基礎となっている。

<!--en:c18a5abb-->
- [Accelerating Generative AI with PyTorch II: GPT, Fast — PyTorch team (2023)](https://pytorch.org/blog/accelerating-generative-ai-2/)：[gpt-fast](https://github.com/meta-pytorch/gpt-fast) のコードを伴う解説記事であり、コンパイル、量子化、投機的デコーディング、テンソル並列を用いて、1,000 行未満の PyTorch コードで 10 倍近い推論の高速化を段階的に構築している。

<!--en:e6855df9-->
- [Which GPU for Deep Learning? — Tim Dettmers (2023)](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)：最終更新が 2023 年 1 月であるためハードウェアの具体例は古くなっているが、ワークロードから GPU を選ぶ手法は :numref:`sec_hardware_buyers` を補完する。

<!--en:5d8b025f-->
**サーベイ論文**

<!--en:5f8b0eff-->
- [A Survey on Efficient Inference for Large Language Models — Zhou et al. (2024)](https://arxiv.org/abs/2404.14294)：データ・モデル・システムの各レベルにおける推論の最適化手法を体系的に分類し、代表的手法の比較実験をまとめたサーベイであり、:numref:`sec_model_serving` の背景知識となる。

<!--en:774f6e59-->
- [A Survey on Large Language Model Acceleration Based on KV Cache Management — Li et al. (2024)](https://arxiv.org/abs/2412.19442)：KV キャッシュ（:numref:`sec_model_serving` で導入）の保持、圧縮、および退避の手法を、精選された論文リストとともに解説している。

<!--en:eeeb051a-->
**最新動向の把握**

<!--en:bd3915f4-->
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) と [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions)：リリース報告、量子化の検証結果、ハードウェアの実測値、および再現の試みが集まる場であり、こうした情報源をどう吟味するかは :numref:`sec_software_ecosystem` で解説する。

<!--en:0b9f61a6-->
- [Pro Git — Chacon and Straub](https://git-scm.com/book/en/v2)：無料の書籍であり、その第 1〜3 章と第 6 章では :numref:`sec_developers_guide` が前提とする Git とプルリクエストのワークフローを解説している。

<!--en:e76fff19-->
深層学習のためのツール：ローカルのノートブックから分散クラスタとサービングへ<br>**再現可能なカーネル・ホスト型ランタイム・クラウドの経済性・ハードウェアのルーフライン・オープンウェイトのアーティファクト・分散訓練・推論エンジン**

<!--en:c169e67c-->
::: {.slide title="あらゆるツール選択の背後にある 4 つのシステム上の問い"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:6a02baab-->
- **1. メモリに収まるか：**訓練ではパラメータ、勾配、オプティマイザのモーメント（$16N$）、活性化値を VRAM に保持するため、FLOP/s が問題になる前に OOM で実行が停止する

<!--en:d07db932-->
- **2. 何がスループットを律速するか：**訓練とプロンプトのプレフィルは計算律速である一方、バッチサイズ 1 の自己回帰デコーディングはメモリ帯域幅律速になる

<!--en:63b97272-->
- **3. コストと再現性：**完了した結果あたりの総コスト（セットアップ、エグレス、スポット復旧を含む）を最適化し、最初のセルから環境と SHA を固定する

<!--en:517c847d-->
::: {.slide title="システムサイジングのための 3 つの第一原理の式"}

<!--en:b2d9898b-->
[定量的な原則]{.kicker}

<!--en:7fd2b1d8-->
- **訓練状態と演算量（§{sec_hosted_notebooks}、§{sec_hardware_buyers}、§{sec_training_systems}）：**混合精度の Adam は $16\text{ B/param}$ を消費し、演算量は $C \approx 6ND\text{ FLOPs}$（パラメータ数 $N$、トークン数 $D$）となる

<!--en:50b9507a-->
- **デコードとチェックポイント間隔（§{sec_hosted_notebooks}〜§{num:sec_model_serving}）：**バッチサイズ 1 のデコードは $B_w + B_{\text{kv}}$ を帯域幅 $\beta$ でストリーム読み出しし、Young の法則では $\tau^* = \sqrt{2\delta M}$ ごとにチェックポイントを保存する

<!--en:d694597f-->
::: {.slide title="第 §{num:chap_appendix_tools} 章のロードマップ：ノート PC から本番環境へ"}

<!--en:00be04d1-->
[章の構成]{.kicker}

<!--en:ad3d532b-->
- **§{sec_interactive_development}〜§{num:sec_cloud_instances} ノートブック、ホスト型 VM、クラウド：**カーネル状態の管理規律、`uv` ロックファイル、Colab・Kaggle での永続化、完了結果あたりのクラウドコスト

<!--en:46b2f579-->
- **§{sec_hardware_buyers}〜§{num:sec_software_ecosystem} ハードウェアとエコシステム：**容量と帯域幅のルーフライン、MoE 向けユニファイドメモリ、Arena Elo の分解能、`safetensors` と SHA によるピン留め

<!--en:931e7f99-->
- **§{sec_training_systems}〜§{num:sec_utils} スケーリング、サービング、D2L の内部構造：**DDP/FSDP2、3D 並列性、連続バッチング、PagedAttention、D2L のシングルソースビルド

<!--en:8142e7ce-->
::: {.slide title="本章を通じた具体的なベンチマーク例"}

<!--en:05adf916-->
[学習を貫く軸]{.kicker}

<!--en:06cebc8a-->
- **再現可能な環境構築と経済性（§{sec_interactive_development}〜§{num:sec_cloud_instances}）：**`uv pip sync`、環境のフィンガープリント、4090 / A100 / H100 にわたるエンジニア工数込みのクラウドコストモデル

<!--en:f26e36ed-->
- **サイジングと検証（§{sec_hardware_buyers}〜§{num:sec_software_ecosystem}）：**7B の訓練状態（フル訓練の $78\text{ GiB}$ 対 QLoRA の $\sim 4\text{ GiB}$）、デコードの上限、Bradley-Terry SE（$347/\sqrt{v}$）、固定されたマニフェスト

<!--en:e6bc6299-->
- **分散訓練、サービング、ビルド（§{sec_training_systems}〜§{num:sec_developers_guide}）：**8 GPU の FSDP2（$22\text{ GiB}$ 対 DDP の $95\text{ GiB}$）、連続バッチングのシミュレータ、プレフィックス出力のハッシュ化

<!--en:5ec7fbd8-->
::: {.slide title="本章の読み進め方と最新動向の把握"}

<!--en:123f19cf-->
[学習ガイド]{.kicker}

<!--en:33b7fe08-->
- **すぐ始める環境構築（§{sec_interactive_development}〜§{num:sec_hosted_notebooks}）：**一時的なセッションで作業内容を失うことなく、ローカルの `uv` 環境または Colab や Kaggle 上で本書のノートブックを実行するにはここから始める

<!--en:50ebd20b-->
- **インフラの意思決定（§{sec_cloud_instances}〜§{num:sec_software_ecosystem}）：**クラウド GPU のレンタル、ワークステーションの構築、あるいは Hugging Face からのオープンウェイトのチェックポイント採用の前にこれらの節を読む

<!--en:ecb18b1a-->
- **スケーリングと貢献（§{sec_training_systems}〜§{num:sec_utils}）：**マルチ GPU 訓練と LLM サービングには §{sec_training_systems}〜§{num:sec_model_serving} を第 §{num:chap_performance} 章（*計算性能*）と併読し、D2L への貢献には §{sec_developers_guide} を、`d2l` ユーティリティには §{sec_utils} を参照する

<!--en:c10e68dc-->
- **副読本・関連教材：**本章とあわせて DeepMind の *How to Scale Your Model*、Hugging Face の *Ultra-Scale Playbook*、および Stanford CS336 を活用する

<!--en:05480b67-->
**次節（§{sec_interactive_development} ノートブック）：**Jupyter カーネルの実行モデルの仕組み、隠れた状態が再現性を損なう理由、および `uv` を用いた決定論的なローカル環境の構築方法
