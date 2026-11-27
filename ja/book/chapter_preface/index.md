<!--en:3c0e38c7-->
# まえがき

<!--en:468f88d3-->
## 第 2 版へのまえがき

<!--en:4cd42ef5-->
振り返ってみると、初版はまるで別の世界で書かれたかのように感じられる。本文、図表、コードはすべて手作業で作られ、スタイルガイドは共著者たちによって熱心に徹底され、新しいフレームワークを追加するには熟練した専門家の手が必要であった。具体的には、Anirudh Dagar（PyTorch と JAX）および Yuan Tang（TensorFlow）の尽力によるものである。本書が 1,000 ページを超え、何百もの GitHub の Issue や PR、Discourse のコメントが寄せられるようになると、すべてを最新の状態に保つことはフルタイムの仕事になった。その結果、著者全員がそれぞれ別の場へと移った後、本書は 5 年間にわたりそのまま置かれることとなった。わかりやすいコード、例を通じた解説、ノートブック形式といった本書の前提の多くは、時の試練に耐え抜いた。これらは 2020 年当時と変わらず今も新鮮である。内容のおよそ半分は今なお色あせていないが、残りの部分はその後、はるかに魅力的な問題や手法、応用によって取って代わられた。ハードウェアとそれに伴うソフトウェアツールが飛躍的な進歩を遂げたため、コードについては D2L ライブラリの形式を除けば、そのまま通用するものはほとんど残らなかった。

<!--en:5da87d86-->
2026 年に起きた 2 つの出来事をきっかけに、このプロジェクトを再び手に取り、D2L を「改装」できるか試してみることにした。第 1 に、ギリシャの出版社から、本書をギリシャ語に翻訳して印刷・出版することに同意してもらえないかという依頼を受けたことである。最後に改良を加えてから 5 年が経ってもなお、新たに書籍として印刷したいと考える人がいるほどの価値が本書に残っていることを、この依頼は示していた。第 2 に、誰もが目の当たりにしてきたエージェント型 AI の急速な進歩により、Anthropic、OpenAI、Google のツールを使えば、以前なら到底手が回らなかった大規模な更新や修復を劇的に容易に行えるようになったことである。これらのツールのおかげで、コードを現代のフレームワークに合わせて刷新し、未対応だった GitHub の Issue や PR を取り込み、Discourse スレッドのバグ報告やコメントを確認し、コード・文章・実験・数式の整合性を検証して修正できた。出版スタックも刷新しており、現在本書では [Quarto](https://quarto.org) と [uv](https://docs.astral.sh/uv/) を使用し、ディスカッションフォーラムは [Discourse](https://d2l.discourse.group) 自体のサーバー上でホストしている。さらにエージェントの活用により、すべてのフレームワークが同じ問題に対して同等の性能を発揮するように揃えられた。すなわち、不適切な JIT コンパイルの利用、初期化の違い、最適化スケジュール、あるいは「魔法の」乱数シードの使用に起因する性能低下を解消したのである。これは以前には望むべくもなかった贅沢だが、全編にわたって各フレームワークのベストプラクティスを示せるという点で、教育上の質を大きく高めていると確信している。

<!--en:5e8f780f-->
こうした基盤の刷新と並行して、内容自体も大きく進化した。PDF で 3,000 ページを超える規模となった本書は、もはや単一巻の枠に収まらなくなり、現在はモジュール化された複数のパートで構成されている。[基礎](../chapter_basics/index.qmd)パートでは深層学習の中核となる土台を築き、入門講義にそのまま使える自己完結した伴走教材となるよう設計されている。[発展](../chapter_advanced/index.qmd)パートでは、[最適化](../chapter_optimization/index.qmd)、[状態空間モデル](../chapter_recurrent-modern/index.qmd)、[Transformer](../chapter_transformers/index.qmd)、[強化学習](../chapter_reinforcement-learning/index.qmd)、[GAN](../chapter_generative-adversarial-networks/index.qmd)、[拡散モデル](../chapter_diffusion-models/index.qmd)など多くの分野について、特定のテーマに深入りしすぎることなく現在の最先端に近い水準まで導く。これらの各分野における情報爆発を考えれば、すべてに深く踏み込むことは不可能だからである。[深層学習のための数学](../chapter_mathematics-for-deep-learning/index.qmd)パートも大幅に拡充され、[拡散とフロー](../chapter_mdl-dynamics/index.qmd)、[情報理論](../chapter_mdl-information-theory/index.qmd)、[最適化](../chapter_mdl-optimization/index.qmd)をより手厚く扱うようになった。このパートは本分野の入門的な数学講義の教科書としても十分に活用でき、論文（あるいは本書の一部の章）を読む際に当然の前提とされがちな背景知識を補う助けとなる。強化学習と拡散モデルの章を寄稿してくれた Rasool Fakoor に感謝する。

<!--en:2921bcf7-->
やるべきことはまだ数多く残っている。[言語モデル](../chapter_language-models/index.qmd)パートは未執筆であり、本書では現代のコンピュータビジョン（「画像モデル」パートは初版から引き継いだものである）、グラフィックス、オーディオ、音声、半教師あり学習をまだ扱っておらず、エージェントについての詳細な解説も備えていない。少しの時間と大量のトークン、そして惜しみないキュレーションがあれば、後者（エージェント）が前者（本書）を完成させる助けとなってくれるだろう。さらにその仕組みは、新たな科学的発見や新手法の発明があるたびに本書を最新の状態に保つ上でも役立つはずである。

<!--en:12544c61-->
*Alex Smola、2026 年 11 月*

<!--en:f8105d57-->
## 初版へのまえがき

<!--en:d55e23c5-->
ほんの数年前まで、大企業やスタートアップで知的な製品やサービスを開発する深層学習の研究者・技術者の大群など存在しなかった。筆者らがこの分野に足を踏み入れた頃、機械学習が日刊紙の見出しを飾ることはなかった。筆者らの両親は機械学習とは何かまったく知らず、まして医学や法曹の道よりもそちらを選ぶ理由など見当もつかなかった。機械学習は基礎研究志向の学術分野であり、産業上の重要性は音声認識やコンピュータビジョンなど限られた実世界の応用にとどまっていた。しかも、そうした応用の多くは膨大なドメイン知識を必要としたため、機械学習はそのごく一部を占めるにすぎない、まったく別の専門分野とみなされることが多かった。当時、本書で焦点を当てる深層学習手法の前身であるニューラルネットワークは、一般に時代遅れとみなされていたのである。

<!--en:477fd141-->
ところがわずか数年のうちに、深層学習は世界を驚かせ、コンピュータビジョン、自然言語処理、自動音声認識、強化学習、生物医学情報学といった多様な分野で急速な進歩を牽引するようになった。さらに、実用上重要な数々のタスクにおける深層学習の成功は、理論的な機械学習や統計学の発展さえも促している。こうした進歩を手にしたいま、かつてないほど高い自律性で走行する自動運転車（もっとも、一部の企業が宣伝するほどの自律性には達していないが）、確認の質問を投げかけながらコードをデバッグする対話システム、さらには囲碁のようなボードゲームで世界最高峰の人間の棋士を破るソフトウェアエージェント（かつては何十年も先のことと考えられていた偉業である）を構築できるようになった。すでにこれらのツールは産業と社会にますます幅広い影響を及ぼしており、映画制作や疾病診断のあり方を変え、天体物理学から気候モデリング、気象予測、生物医学に至るまで、基礎科学においても存在感を高めつつある。

<!--en:7fd586dc-->
*Aston Zhang、Zach Lipton、Mu Li、Alex Smola、2021 年*

<!--en:c9dfe567-->
## 本書について

<!--en:6e4bdc5c-->
本書は、*概念*・*文脈*・*コード*を 1 か所で学ぶことで、深層学習を親しみやすいものにする試みである。

<!--en:bc1c9331-->
### コード、数式、HTML を統合した単一の媒体

<!--en:67f95cfd-->
いかなるコンピューティング技術も、その真価を十分に発揮するには、よく理解され、十分に文書化され、成熟して適切に保守されたツールによって支えられていなければならない。核心となる考え方は明確に整理され、新たな実務者が最前線に追いつくまでの時間が最小限に抑えられているべきである。また、成熟したライブラリが定型的な作業を自動化し、模範となるコードによって、実務者が標準的なモデルを自身の目的に合わせて容易に改変・応用・拡張できなければならない。

<!--en:d078b25b-->
動的なウェブアプリケーションはその好例である。Amazon などの企業は 1990 年代にデータベース駆動型のウェブアプリケーションを開発して成功を収めたが、この技術が創造的な起業家を支援する幅広い可能性を解き放ったのは、続く 10 年間に強力で十分に文書化されたウェブフレームワークが登場してからのことであった。

<!--en:765e3149-->
深層学習の可能性を引き出すには特有の難しさがある。というのも、どの応用ひとつをとっても複数の専門分野が絡み合うからである。深層学習を応用するには、(i) 問題を特定の形で定式化する動機、(ii) 与えられたモデルの数学的形式、(iii) モデルをデータに適合させる最適化アルゴリズム、(iv) モデルが未知のデータにいつ汎化（generalization）すると期待できるかを示す統計的原理と、実際に汎化したことを検証する実践的手法、そして (v) 数値計算の落とし穴を回避し、利用可能なハードウェアを最大限に活かしてモデルを効率的に訓練するエンジニアリング技術、これらすべてを同時に理解する必要がある。問題を定式化するための批判的思考力、それを解くための数学、そしてその解法を実装するためのソフトウェアツールを 1 か所で教えることは容易ならざる挑戦である。本書における目標は、意欲ある実務者が迅速に実力を身につけられる統合的なリソースを提供することにある。

<!--en:d3a46cea-->
本書のプロジェクトを開始した当時、(i) 常に最新の状態に保たれ、(ii) 現代の機械学習の広がりを十分な技術的深さで網羅し、(iii) 教科書に求められる厳密な解説と実践的なチュートリアルに求められる明快で実行可能なコードとを織り交ぜたリソースは、単一のものとしては存在しなかった。特定の深層学習フレームワークの使い方（たとえば TensorFlow における基本的な行列計算）や、特定のアーキテクチャ（LeNet、AlexNet、ResNet など）の実装方法を示すコード例は、ブログ記事や GitHub リポジトリの随所に数多く見られた。しかし、そうした例の多くは手法を*どのように*実装するかは示していても、アルゴリズムの設計上*なぜ*その選択がなされたのかは省かれていた。[Distill](http://distill.pub) や個人ブログに掲載されたインタラクティブな記事は、特定のトピックを視覚的に掘り下げてはいたものの、扱う範囲は分野の一部に限られ、コードが伴わないことも多かった。一方、深層学習の基礎を包括的に概観した :citet:`Goodfellow.Bengio.Courville.2016` のように、当時登場していた深層学習の教科書は数学的な記述を動作する実装と結びつけておらず、理論とコードの橋渡しに戸惑う読者も少なくなかった。さらに、あまりに多くの学習リソースが商用講座のペイウォールの向こう側に閉ざされていた。

<!--en:77f09e2d-->
そこで本書では、次のようなリソースを作ることを目指した。すなわち、(i) 誰もが無料で利用でき、(ii) 応用機械学習の研究者・技術者へと飛躍するための足がかりとなる十分な技術的深さを備え、(iii) 実際に問題を*どのように*解くかを読者に示す実行可能なコードを含み、(iv) 著者陣およびコミュニティ全体による迅速な更新が可能であり、(v) 技術的な詳細や未解決の疑問について双方向に議論できる[ディスカッションフォーラム](https://d2l.discourse.group/c/5)によって補完されるリソースである。

<!--en:2fdf61b3-->
これらの目標はしばしば互いに相反した。数式、定理、引用の執筆と組版には LaTeX が最も適している。コードは Python で表現するのが最も自然である。そしてウェブページの本来の基盤は HTML と JavaScript である。さらに、実行可能なノートブック、印刷書籍、ダウンロード可能な PDF、インタラクティブなウェブサイトといういずれの形態でも同等にアクセスできるようにしたかった。これらすべての要求を満たす既存の出版ワークフローは存在しなかったため、独自の仕組みを構築した（:numref:`sec_how_to_contribute`）。ソースの管理とコミュニティからの貢献の調整には GitHub を、文章・数式・コードを混在させそこから Jupyter ノートブックを自動生成する単一のソース形式には Markdown を、レンダリングエンジンには Quarto を、そして議論のプラットフォームには Discourse を採用した。完璧なシステムは存在しないものの、これらの選択は相反する要求の間で効果的なバランスを実現している。*Dive into Deep Learning* は、このような統合ワークフローを用いて出版された最初の書籍であると考えられる。

<!--en:27b0d331-->
### 実践による学習

<!--en:cdf06e42-->
多くの教科書は概念を厳密な順序で提示し、各トピックを徹底的に掘り下げてから次へと進む。たとえば、:citet:`Bishop.2006` による古典的な教科書は各主題をきわめて網羅的に扱っているため、線形回帰（linear regression）の章にたどり着くだけでも相当な労力を要する。専門家が同書を高く評価するのはまさにその網羅性ゆえであるが、この構成は初心者向けの入門書としては敷居を高くしている。

<!--en:6dc8f980-->
本書では、ほとんどの概念を*ジャストインタイム*で導入する。言い換えれば、具体的で実践的な目標を達成するために必要となったまさにその瞬間に、新しい考え方に触れることになる。冒頭の短い章で線形代数や確率といった不可欠な基礎事項を扱い、さらに深く学びたい読者のために包括的な数学の付録も用意しているが、より抽象的な理論に頭を悩ませる前に、まずは実際に動く最初のモデルを訓練する達成感を味わってほしいと考えている。

<!--en:2c6f7130-->
基本的な数学的背景を短期集中で学ぶ冒頭の数本のノートブックを除けば、以降の各章では無理のない範囲の新しい概念を、実データセット上で動作する自己完結した例とともに導入する。この方針には構成上の難しさも伴った。論理的には 1 つのノートブックにまとめられるモデル群もあれば、複数のモデルを順に実行することで最もよく理解できる考え方もあるからである。それでも、*1 つの動作例につき 1 つのノートブック*という方針を貫くことには大きな実用的利点がある。本書のコードをそのまま土台にして、読者自身の研究プロジェクトや開発プロジェクトをかつてないほど容易に立ち上げられるのである。ノートブックをコピーし、書き換え始めるだけでよい。

<!--en:9e130586-->
本書全体を通して、実行可能なコードと理論的背景を必要に応じて織り交ぜている。原則として、仕組みの細部をすべて説明する前にまずツールを読者の手に渡し、実際に動く様子を確認してからより深い背景を補う方針をとる。たとえば、初期のモデルの訓練には*確率的勾配降下法*（stochastic gradient descent）を用いるが、それがなぜこれほど効果的に機能するのかを詳しく分析するのはずっと後の章である。このアプローチには、途中の教育上の選択をいくつか信頼してもらうというわずかな代償こそ伴うものの、実務者が素早く問題を解くために必要なツールをいち早く手にできるという利点がある。

<!--en:9087e740-->
また本書では、深層学習の中核となる概念をゼロから実装しながら学ぶ。現代の深層学習フレームワークが通常はユーザーから隠蔽しているモデルの細部にも、しばしば踏み込んで中身を確かめる。とりわけ基礎を扱う各章では、層、損失関数、オプティマイザの内部で起きていることすべてを理解できるよう、こうした解説を重視している。その際、2 つの実装を並べて示すのが通例である。1 つは基本的なテンソル演算と自動微分（automatic differentiation）のみに依拠してゼロから構築する実装であり、もう 1 つは深層学習フレームワークの高レベル API を用いて簡潔に記述した実装である。ある構成要素の仕組みを一から説明した後は、以降の章ではフレームワークの高レベル実装を利用する。

<!--en:02bb71cd-->
### 内容と構成

<!--en:c15a8c5a-->
本書は 3 つの短い導入章から始まる。:numref:`chap_installation` ではコードを実行するための環境構築の手順を説明し、:numref:`chap_notation` では本書の数学的記法をまとめ、:numref:`chap_introduction` では深層学習の全体像を概観する。本書の残りの部分は 7 つのパートで構成されており、中核をなす「基礎」と「発展」の 2 つのパートの構成を :numref:`fig_book_org` に示す。

<!--en:eb0b5fab-->
![本書の構成。ある章から別の章への矢印は、後者が前者を前提とすることを意味し、「基礎」パートから「発展」パートへの矢印は、「発展」パートのすべての章が「基礎」パートを前提とすることを意味する。「言語モデル」、「画像モデル」、「アーカイブ」の各パートは省略している。](../img/front-book-org.svg)

<!--en:8d86b3f8-->
* **基礎**。
:numref:`chap_preliminaries` では、データをテンソルとして格納・操作する方法、導関数を自動計算する方法、確率と不確実性を扱う考え方など、実践的な深層学習に欠かせない前提知識を手早く身につける。続く :numref:`chap_regression` と :numref:`chap_classification` では、線形モデルをデータに適合させながら、本書のほぼすべての手法に共通する 3 つの基本要素、すなわちパラメータ化されたモデル、その誤差を測る損失関数、そして損失を減らす最適化アルゴリズムを導入する。:numref:`chap_perceptrons` では隠れ層（hidden layer）を追加して深層ネットワークを構築し、これほど多くのパラメータを持つモデルがなぜ新しいデータに対しても汎化できるのかを問う。次に :numref:`chap_computation` では、再利用可能なモジュールからのモデルの組み立て、初期化、保存、GPU による高速化など、深層学習ソフトウェアの仕組みに目を向ける。本パートの最後の 3 章では、領域固有の構造をアーキテクチャに直接組み込む。すなわち、:numref:`chap_cnn` と :numref:`chap_modern_cnn` の畳み込みニューラルネットワーク（convolutional neural network、CNN）は画像の空間的対称性を活用し、:numref:`chap_rnn` のリカレント系列モデルはテキストや時系列の順序構造を捉える。最初の 5 章は順番に読むのが望ましいが、その後の画像に関する章と系列に関する章は互いに独立して読むことができる。

<!--en:bd4cacde-->
* **発展**。
これらの章では、「基礎」パートの土台の上に立ち、今日の最前線で用いられているアーキテクチャと訓練手法を展開する。読む順序は複数あり得、:numref:`fig_book_org` の矢印が章間の依存関係を示している。「基礎」パートではオプティマイザ、学習率スケジュール、バッチサイズを主に実用上のレシピとして扱ったのに対し、:numref:`chap_optimization` ではそれらがなぜ機能し、モデルの大規模化に伴ってどう適応させるべきかを検証する。:numref:`chap_attention` では、系列内の各位置が他の任意の位置から直接情報を取得できるようにする*アテンション*機構を導入し、:numref:`chap_transformers` ではこれらの層を積み重ねて、現代の大規模言語モデルを支える Transformer アーキテクチャを構築する。:numref:`chap_modern_rnn` では状態空間モデルを通じて再帰性を再考し、固定サイズの状態を保ちながらも系列のすべての位置にわたって並列に訓練する仕組みを学ぶ。:numref:`chap_performance` では Transformer を一貫した例として用い、単一 GPU および複数 GPU にまたがる訓練ステップの速度を何が律速するのかを探る。:numref:`chap_reinforcement_learning` と :numref:`chap_deep_rl` では教師あり学習から離れ、環境との相互作用を通じて報酬から学習するエージェントへと軸足を移し、厳密に解ける小さな表形式の問題から始めてニューラルネットワークによる大規模化へと進む。最後に :numref:`chap_gans` と :numref:`chap_diffusion` では生成モデルを扱い、生成器と識別器の間のミニマックスゲーム、あるいはデータ破壊過程を反復的に逆転させる学習によって新しいデータを合成する方法を学ぶ。

<!--en:b2703d68-->
* **言語モデルと画像モデル**。
現在準備中の「言語モデル」パートでは、「発展」パートの手法を現代の大規模言語モデルへと拡張し、エンコーダ・埋め込み・検索（:numref:`chap_lm_encoders`）、事前学習（:numref:`chap_lm_pretraining`）、ファインチューニング（:numref:`chap_lm_finetuning`）、選好と検証可能な報酬によるアライメント（:numref:`chap_lm_rl`）、エージェント（:numref:`chap_lm_agents`）、そして効率的なサービング（:numref:`chap_lm_serving`）を扱う。「画像モデル」パート（:numref:`chap_cv`）では物体検出、セマンティックセグメンテーション、ニューラルスタイル変換を取り上げており、初版から引き継がれた内容として今後全面的な改稿を予定している。

<!--en:2f126fdd-->
* **深層学習のための数学**。
この付録では、本編の根底にある数学的基礎を展開する。本編の各章では数学的な結果を直観とともに述べるにとどめ、完全な証明や導出、より深い背景についてはこの付録を参照する形をとっている。:numref:`chap_mdl-linear-algebra` では幾何学的な射影、固有値分解、特異値分解を扱い、:numref:`chap_mdl-calculus` では多変数の導関数、勾配、自動微分の仕組みを展開する。:numref:`chap_mdl-optimization` では、曲率がなぜ学習率の上限を定めるのかといった、:numref:`chap_optimization` で観察される経験的な振る舞いの背後にある理論的保証を与える。:numref:`chap_mdl-probability-statistics` では統計的推定、ベイズ推論、汎化限界を扱い、:numref:`chap_mdl-information-theory` では交差エントロピーが何を測っているのか、そしてダイバージェンスや最適輸送によって確率分布をどのように比較できるのかを説明し、:numref:`chap_mdl-dynamics` では拡散ベースおよびフローベースの生成モデルの背後にある微分方程式と確率過程を展開する。あるトピックにすでに精通している読者は該当する章を読み飛ばすか参考資料として使えばよく、それ以外の読者にとっては、この付録が自己完結した入門教科書として機能するよう設計されている。

<!--en:344d1818-->
* **深層学習のためのツール**。
:numref:`chap_appendix_tools` では、本書のノートブックをローカル環境やクラウド環境で実行する方法、GPU ハードウェアの選定とプロビジョニング、分散訓練のスケールアップ、本番環境でのモデルのサービング、本書への貢献方法といった実践的なエンジニアリングのワークフローを扱う。あわせて `d2l` ライブラリの API リファレンス（:numref:`sec_d2l`）も収録している。

<!--en:3c985ab8-->
* **アーカイブ**。
最後のパートには、ガウス過程、ハイパーパラメータ最適化、推薦システム（:numref:`chap_gp` から :numref:`chap_recsys` まで）など、初版から引き継がれ今版では積極的な拡充の対象外となった章が収められている。

<!--en:96104f9f-->
### コード

<!--en:b3a3b264-->
本書のほとんどの節には実行可能なコードが含まれている。深層学習における重要な直観の多くは、コードを少しずつ書き換えて結果がどう変化するかを観察するという、手を動かす実験を通じてこそ最もよく養われると考えられる。理想的には、包括的な数学理論によって、あらゆるアーキテクチャやハイパーパラメータの調整がどのような振る舞いをもたらすかを事前に知ることができればよいだろう。しかし現実には、深層学習の実務者は形式的な理論に先んじて試行錯誤しなければならない場面が多い。急速な進歩にもかかわらず、多くの手法がなぜ有効なのかについての完全な理論的説明はいまだ得られていない。強い非線形性を持つネットワークを特徴づける数学は容易ではなく、汎化は単純な定式化を拒む実世界データの構造的性質に左右されることが多く、現代のアーキテクチャに対する体系的な理論研究は経験的な発見を今なお追いかけている段階にある。深層学習の理論が成熟するにつれて、本書の将来の版により深い形式的知見を取り入れていくことを楽しみにしている。

<!--en:c86f10a2-->
本書のコードは最大 4 つの深層学習フレームワークで提供されており、それぞれ独立したインタラクティブなタブで表示される。「基礎」パートと付録「深層学習のための数学」は PyTorch、JAX、TensorFlow、MXNet で実装されており、「発展」パートは PyTorch と JAX で実装されている。フレームワークの選択と動作環境の構築については :numref:`chap_installation` が手引きとなる。すべてのノートブックは共通インポートをまとめた標準セルから始まり、その内容は :numref:`sec_preamble` で解説している。本文の焦点を絞るため、このプリアンブルのセルはそこでのみ一度だけ表示し、以降の節では扱うトピックに直接関係するコードからすぐに始められるようにしている。

<!--en:bb3d4b26-->
章ごとに定型コードを繰り返すのを避けるため、再利用可能な関数やクラスは軽量な `d2l` ライブラリにまとめている。本書全体を通して、`#@save` が付いたコードブロック（補助関数、基底クラス、標準インポートなど）は本文中で一度だけ定義・説明されると同時に `d2l` パッケージに保存され、後の章では `d2l` 経由で直接呼び出せるようになっている。これらの関数とクラスの完全な一覧は :numref:`sec_d2l` に掲載している。フレームワークに依存しない `d2l` ライブラリの中核部分が要するのは、以下の標準的な Python インポートのみである。

<!--en:4b9cff6b-->
現在表示されているのは本書の MXNet 版である。MXNet による実装は、「基礎」パートおよび付録「深層学習のための数学」で利用できる。:numref:`chap_optimization` から始まる「発展」パートは PyTorch と JAX で実装されているため、発展的な章を読む際にはいずれかのタブに切り替えることができる。MXNet のコードは、:numref:`chap_installation` の環境設定で固定された MXNet 2.0 ビルドを用いて実行している。バージョンの非互換性が生じた場合は、その固定された環境を使用されたい。`d2l` ライブラリの MXNet バックエンドでは、以下のインポートが追加される。

<!--en:d3fadea1-->
現在表示されているのは本書の PyTorch 版であり、すべての章にわたって実装が提供されている。コードは、:numref:`chap_installation` の環境設定で固定されたバージョンの PyTorch を用いて実行している。深層学習フレームワークは急速に進化するため、新しいリリースで非互換性が生じた場合は、その固定された環境の使用を勧める。`d2l` ライブラリの PyTorch バックエンドでは、以下のインポートが追加される。

<!--en:6a407cc8-->
現在表示されているのは本書の TensorFlow 版である。TensorFlow による実装は、「基礎」パートおよび付録「深層学習のための数学」で利用できる。:numref:`chap_optimization` から始まる「発展」パートは PyTorch と JAX で実装されているため、発展的な章を読む際にはいずれかのタブに切り替えることができる。コードは、:numref:`chap_installation` の環境設定で固定されたバージョンの TensorFlow を用いて実行している。深層学習フレームワークは急速に進化するため、新しいリリースで非互換性が生じた場合は、その固定された環境の使用を勧める。`d2l` ライブラリの TensorFlow バックエンドでは、以下のインポートが追加される。

<!--en:25e699d6-->
現在表示されているのは本書の JAX 版であり、「基礎」パート、「発展」パート、および付録「深層学習のための数学」の実装が提供されている。JAX は純粋な数値計算用の Python 関数を変換することで動作し、自動微分、コンパイル、ベクトル化を施す。本書全体を通して、ニューラルネットワークのモデルは Flax の `nnx` モジュールを用いて定義し、Optax のオプティマイザで訓練する。コードは、:numref:`chap_installation` の環境設定で固定された JAX エコシステムの各バージョンを用いて実行している。これらのライブラリは急速に進化するため、新しいリリースで非互換性が生じた場合は、その固定された環境の使用を勧める。`d2l` ライブラリの JAX バックエンドでは、以下のインポートが追加される。

<!--en:3b7e757c-->
### 対象読者

<!--en:3d43c720-->
本書は、現代の深層学習を実践的かつ着実に身につけたい学部生・大学院生、ソフトウェアエンジニア、研究者を対象に執筆されている。すべての概念を一から積み上げて解説するため、深層学習や機械学習に関する予備知識は必要ない。深層学習の手法を丁寧に説明するには数学とプログラミングの双方が欠かせないが、前提とするのは線形代数、微分積分、確率、および Python の基礎的な素養のみである。数学の前提知識を復習したい場合や、理論をより深く掘り下げたい場合には、付録の[深層学習のための数学](../chapter_mathematics-for-deep-learning/index.qmd)が本書で用いる数学のほぼすべてを網羅している。本編の各章では、形式的な数学的一般性よりも明快な直観と核心的な考え方を一貫して優先する。

<!--en:ab2899ee-->
数学的基礎をさらに深く学びたい読者には、いくつかの古典的な名著を強く勧める。:citet:`Bollobas.1999` による『Linear Analysis』は線形代数と関数解析を深く扱い、『All of Statistics』は統計的推論への簡潔で明快な入門書であり :cite:`Wasserman.2013`、確率に関する Joe Blitzstein の[教科書](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918)と[講義](https://projects.iq.harvard.edu/stat110/home)は教育的にも珠玉の内容である。最後に、Python に初めて触れる場合は、コードを読み進める前にインタラクティブな [Python チュートリアル](http://learnpython.org/)に取り組んでおくことを勧める。

<!--en:334b5836-->
### ノートブック、ウェブサイト、GitHub、フォーラム

<!--en:1cb5fffd-->
本書の実行可能なノートブックはすべて、[D2L.ai ウェブサイト](https://d2l.ai)および [GitHub リポジトリ](https://github.com/smolix/d2l-neu)からダウンロードできる。また、[d2l.discourse.group](https://d2l.discourse.group/c/5) では活発なコミュニティのディスカッションフォーラムも運営している。本書のどの節についても、疑問が生じた際にはノートブック末尾にある直通リンクから対応するディスカッションスレッドへアクセスできる。

<!--en:37aeb2c1-->
## 謝辞

<!--en:2ea4ef65-->
英語版と中国語版の双方に携わった何百人もの貢献者に深く感謝する。その尽力により内容は改善され、数々の貴重なフィードバックが寄せられた。本書は当初、MXNet を主たるフレームワークとして実装されていた。初期の MXNet コードの大部分をそれぞれ PyTorch 実装と TensorFlow 実装へ移植してくれた Anirudh Dagar と Yuan Tang に感謝する。2021 年 7 月以降、PyTorch を主たるフレームワークに据え、本書を PyTorch、MXNet、TensorFlow で再設計・再実装した。その新しい PyTorch コードの大部分を JAX 実装へ移植してくれた Anirudh Dagar に改めて感謝する。また、中国語版においてその PyTorch コードの大部分を PaddlePaddle 実装へ移植してくれた Baidu の Gaosheng Wu、Liujun Hu、Ge Zhang、Jiehang Xie に感謝する。出版社の LaTeX スタイルを PDF ビルドへ統合してくれた Shuai Zhang にも謝意を表する。

<!--en:5a5a5764-->
GitHub 上で初版をすべての読者にとってより良いものにしてくれた貢献者全員にも感謝する。その GitHub ID または氏名は次のとおりである（順不同）。alxnorden、avinashingit、bowen0701、brettkoonce、Chaitanya Prakash Bapat、cryptonaut、Davide Fiocco、edgarroman、gkutiel、John Mitro、Liang Pu、Rahul Agarwal、Mohamed Ali Jamaoui、Michael (Stu) Stewart、Mike Müller、NRauschmayr、Prakhar Srivastav、sad-、sfermigier、Sheng Zha、sundeepteki、topecongiro、tpdi、vermicelli、Vishaal Kapoor、Vishwesh Ravi Shrimali、YaYaB、Yuhong Chen、Evgeniy Smirnov、lgov、Simon Corston-Oliver、Igor Dzreyev、Ha Nguyen、pmuens、Andrei Lukovenko、senorcinco、vfdev-5、dsweet、Mohammad Mahdi Rahimi、Abhishek Gupta、uwsd、DomKM、Lisa Oakley、Bowen Li、Aarush Ahuja、Prasanth Buddareddygari、brianhendee、mani2106、mtn、lkevinzc、caojilin、Lakshya、Fiete Lüer、Surbhi Vijayvargeeya、Muhyun Kim、dennismalmgren、adursun、Anirudh Dagar、liqingnz、Pedro Larroy、lgov、ati-ozgur、Jun Wu、Matthias Blume、Lin Yuan、geogunow、Josh Gardner、Maximilian Böther、Rakib Islam、Leonard Lausen、Abhinav Upadhyay、rongruosong、Steve Sedlmeyer、Ruslan Baratov、Rafael Schlatter、liusy182、Giannis Pappas、ati-ozgur、qbaza、dchoi77、Adam Gerson、Phuc Le、Mark Atwood、christabella、vn09、Haibin Lin、jjangga0214、RichyChen、noelo、hansent、Giel Dops、dvincent1337、WhiteD3vil、Peter Kulits、codypenta、joseppinilla、ahmaurya、karolszk、heytitle、Peter Goetz、rigtorp、Tiep Vu、sfilip、mlxd、Kale-ab Tessera、Sanjar Adilov、MatteoFerrara、hsneto、Katarzyna Biesialska、Gregory Bruss、Duy–Thanh Doan、paulaurel、graytowne、Duc Pham、sl7423、Jaedong Hwang、Yida Wang、cys4、clhm、Jean Kaddour、austinmw、trebeljahr、tbaums、Cuong V. Nguyen、pavelkomarov、vzlamal、NotAnotherSystem、J-Arun-Mani、jancio、eldarkurtic、the-great-shazbot、doctorcolossus、gducharme、cclauss、Daniel-Mietchen、hoonose、biagiom、abhinavsp0730、jonathanhrandall、ysraell、Nodar Okroshiashvili、UgurKap、Jiyang Kang、StevenJokes、Tomer Kaftan、liweiwp、netyster、ypandya、NishantTharani、heiligerl、SportsTHU、Hoa Nguyen、manuel-arno-korfmann-webentwicklung、aterzis-personal、nxby、Xiaoting He、Josiah Yoder、mathresearch、mzz2017、jroberayalas、iluu、ghejc、BSharmi、vkramdev、simonwardjones、LakshKD、TalNeoran、djliden、Nikhil95、Oren Barkan、guoweis、haozhu233、pratikhack、Yue Ying、tayfununal、steinsag、charleybeller、Andrew Lumsdaine、Jiekui Zhang、Deepak Pathak、Florian Donhauser、Tim Gates、Adriaan Tijsseling、Ron Medina、Gaurav Saha、Murat Semerci、Lei Mao、Levi McClenny、Joshua Broyde、jake221、jonbally、zyhazwraith、Brian Pulfer、Nick Tomasino、Lefan Zhang、Hongshen Yang、Vinney Cavallo、yuntai、Yuanxiang Zhu、amarazov、pasricha、Ben Greenawald、Shivam Upadhyay、Quanshangze Du、Biswajit Sahoo、Parthe Pandit、Ishan Kumar、HomunculusK、Lane Schwartz、varadgunjal、Jason Wiener、Armin Gholampoor、Shreshtha13、eigen-arnav、Hyeonggyu Kim、EmilyOng、Bálint Mucsányi、Chase DuBois、Juntian Tao、Wenxiang Xu、Lifu Huang、filevich、quake2005、nils-werner、Yiming Li、Marsel Khisamutdinov、Francesco "Fuma" Fumagalli、Peilin Sun、Vincent Gurgul、qingfengtommy、Janmey Shukla、Mo Shan、Kaan Sancak、regob、AlexSauer、Gopalakrishna Ramachandra、Tobias Uelwer、Chao Wang、Tian Cao、Nicolas Corthorn、akash5474、kxxt、zxydi1992、Jacob Britton、Shuangchi He、zhmou、krahets、Jie-Han Chen、Atishay Garg、Marcel Flygare、adtygan、Nik Vaessen、bolded、Louis Schlessinger、Balaji Varatharajan、atgctg、Kaixin Li、Victor Barbaros、Riccardo Musto、Elizabeth Ho、azimjonn、Guilherme Miotto、Alessandro Finamore、Joji Joseph、Anthony Biel、Zeming Zhao、shjustinbaek、gab-chen、nantekoto、Yutaro Nishiyama、Oren Amsalem、Tian-MaoMao、Amin Allahyar、Gijs van Tulder、Mikhail Berkov、iamorphen、Matthew Caseres、Andrew Walsh、pggPL、RohanKarthikeyan、Ryan Choi、Likun Lei。

<!--en:0bec82b4-->
本書の執筆にあたり惜しみない支援を提供してくれた Amazon Web Services、とりわけ Wen-Ming Ye、George Karypis、Swami Sivasubramanian、Peter DeSantis、Adam Selipsky、Andrew Jassy に感謝する。十分な時間やリソース、同僚たちとの議論、そして絶え間ない励ましがなければ、本書が世に出ることはなかっただろう。出版に向けた準備の過程では、Cambridge University Press から素晴らしい支援を受けた。担当編集者である David Tranah の尽力とプロフェッショナリズムに深く感謝する。

<!--en:dc84a00f-->
### 第 2 版への貢献者

<!--en:3945e89f-->
また、過去 4 年間にわたり GitHub やディスカッションフォーラムを通じて誤植の報告、数学的・教育的な明確化の提案、リンク切れやコードのバグの修正、プルリクエストの投稿を寄せてくれた多くのコミュニティの参加者にも感謝する。その中には、Abhishek Gupta (\@abhishek-gupta2707)、Akshaya Bali (\@akshayabali)、Alankrit (\@alankrit02)、Alexhead (\@Alexhead)、Amirhossein (\@amirhosein-mesbah)、Andrey (\@v4lproik)、Andrii Oriekhov (\@andriyor)、Ankit Kumar Singh (\@ankitkumarsingh19091995)、Ashwinpathak20 (\@ashwinpathak20)、Avinash (\@avinashscs)、Baptiste (\@baptiste-pasquier)、Basti564 (\@Basti564)、Berke (\@berkeoral)、Bhavesh Laddagiri (\@bhaveshladdagiri)、Bjoernenz (\@bjoernenz)、Brian Bartoldson (\@bartoldson)、Catalin (\@catalin-sipos)、Chaitanya (\@chaitanyab2001)、Cuong V. Nguyen (\@ncv)、Daniel (\@dakenf)、Daniel (\@dani2442)、Danishjeet Singh (\@danishjeet)、dark_master (\@Dark_Master)、Dd3607 (\@Dd3607)、Denis Kazakov (\@Denis-Kazakov)、Dilip (\@DilipSCU)、doc_karl (\@Doc_Karl)、DomonkosNagy (\@DomonkosNagy)、Dongryeol Lee (\@dongryeol-lee)、Dr.rer.nat (\@Dr.rer.nat)、Duc (\@nguyenduc1102)、Ebad Sayed (\@ebadsayed)、Elvis (\@elvisesp)、Emmett (\@emmett9001)、EnesBerkSak (\@EnesBerkSak)、Eurus (\@Eurus-Holmes)、Evan (\@evanmiller)、Evgenii (\@kaiser179)、Fardin (\@fardinabbasi)、Fenilgandhi (\@fenilgandhi)、Fredrick (\@FriskyFred)、Fzyzcjy (\@fzyzcjy)、Georgy (\@goshik92)、Glenn (\@glennmatlin)、GoodMan (\@GoodMan666)、Hannes (\@hannesdornbusch)、Haocong Wang (\@HaocongWang)、HappyB1rd (\@HappyB1rd)、Harsh (\@Harsh8604)、HenryJia (\@HenryJia)、Himanshu (\@himanshurawlani)、HyeongjinKim (\@HyeongjinKim)、Inaki (\@InakiLPP)、James (\@james-Howard)、Jayanth (\@jayanthjayanth)、Jcbl1 (\@jcbl1)、Jiawei He (\@h-jiawei)、Jinyang (\@jinyangyuan)、Joe (\@joekakone)、Jon (\@jontonsoup4)、Jonathan (\@JonathanRayner)、Juanlu (\@Juanlu001)、Juha (\@kamiju)、kishan_kc (\@Kishan_KC)、L. Jia (\@L.Jia)、Legin (\@Legin2022)、Lieven (\@lievengovaerts)、Lin (\@Lin13579)、Lockon (\@Lockon007)、Lorenzoh (\@lorenzoh)、Luohao (\@luohao-brian)、Mahmood (\@mahmood-taghavi)、Manimaran (\@Manimaran-T)、Marc (\@MarcBresson)、Mark (\@Mark1001)、Martin (\@mjirik)、Matthieu (\@matthieubulte)、Mhamed (\@mhamedbenhadj)、Michael (\@michael-fonder)、Mieszko (\@Mieszko4)、Minato (\@Minato-namikaze)、Moises (\@moises-b2)、MrDiver (\@MrDiver)、Noam (\@noamgot)、Notplus (\@notplus)、OvO (\@OvO)、Pan (\@pan-x-c)、Phumtawan (\@Phumtawan)、Piotrek (\@piotrek-k)、prajwal_pitlehra (\@Prajwal_Pitlehra)、Qaq (\@qaq)、Ranis (\@ranisalt)、Robert (\@robert-abc)、Saketh (\@sakethram)、Sam (\@SamFancher)、Satvik (\@Satvik-Garg)、Sepehr (\@sepehr-gh)、Sepehr (\@sepehrnoey)、Shichao (\@shichaoy)、Shiva (\@shiva-sankaran)、Smsxgz (\@smsxgz)、Sravan (\@sravan-s)、Stest2012 (\@stest2012)、Tirth (\@Tirth27)、Tushar (\@tusharmaurya)、Verify (\@verify-code)、Victor (\@victor-iyi)、Wj (\@wj-Mcat)、Xinyu (\@xinyu-dev)、Yibo (\@yibotongxue) が名を連ねる。

<!--en:dcd4d01d-->
さらに、フォーラムやリポジトリにおいて活発な参加やフィードバック、技術的な議論を寄せてくれた以下の読者にも感謝する。

<!--en:51702d68-->
Amritansh (\@Amritansh)、Angad (\@Angad_Sandhu)、anh_tuan_bui (\@Anh_Tuan_Bui)、arijit_das (\@Arijit_Das)、Artyom (\@artyompal)、carlos_arriaga (\@Carlos_Arriaga)、Daniel、David (\@d_c)、Fan (\@Fan_Gu)、Hanker (\@Hanker)、Igor (\@igormorgado)、Jeet (\@Jeet)、Jonathan (\@jonathan-code-r)、mahmoud_taha (\@Mahmoud_Taha)、Marios (\@marios_g)、Mark (\@Mark_Hall)、max_ah (\@Max_Ah)、Maxe (\@Maxe)、Mugen (\@Mugen)、Nishant (\@Nishant_Thakur)、Porter (\@Porter)、Raj (\@Raj)、Rling (\@Rling)、Sansui (\@Sansui)。

<!--en:dda32b61-->
[ディスカッション](https://d2l.discourse.group/t/18)

<!--en:db81ad28-->
[ディスカッション](https://d2l.discourse.group/t/20)

<!--en:dedd2dcd-->
[ディスカッション](https://d2l.discourse.group/t/186)

<!--en:ef35fe23-->
[ディスカッション](https://d2l.discourse.group/t/17963)

<!--en:473ac740-->
::: {.slide title="D2L のコード基盤"}

<!--en:e6227312-->
D2L のノートブックでは、`d2l` パッケージの共通インポートと補助関数を使用する。`#@save` が付いたブロックは同パッケージへエクスポートされるため、以降のノートブックでは準備コードを繰り返さずモデルに集中できる。ライブラリのヘッダはインポートのみで構成され、モジュールが `d2l` という名前に束縛される。

<!--en:2cd285c2-->
::: {.slide title="フレームワーク固有のインポート"}

<!--en:c01956d1-->
各バックエンドのタブでは、本書全体で使用するテンソルライブラリ、ニューラルネットワークの層、データ変換、ユーティリティをインポートする。公開される `d2l` 名前空間は一貫したまま、タブによって実装が切り替わる。

<!--en:6a7286e9-->
::: {.slide title="以降のノートブックの読み方"}

<!--en:7b48f6d3-->
- `#@save` は、そのコードが再利用可能な `d2l` ライブラリの一部になることを示す

<!--en:761e50f3-->
- `#@tab` は、MXNet、PyTorch、TensorFlow、JAX それぞれのバックエンド固有のコードを表す

<!--en:1af7d3e0-->
- 大半の章はアルゴリズムを解説するが、このまえがきでは共通の名前とインポートの由来を説明する
