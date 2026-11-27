<!--en:325423f8-->
# Transformer

<!--en:e785897f-->
:numref:`chap_attention` では、マルチヘッドアテンション層、位置エンコーディング、および長文脈へスケールするアテンションの派生手法を構築した。しかし、アテンション単体ではトークン混合演算子（token-mixing operator）にすぎない。トークンごとの非線形変換、残差接続、および正規化がなければ、アテンション層を積み重ねても訓練可能な深層ネットワークは得られないからである。本章では、アテンションを完全な **Transformer** アーキテクチャへと組み上げ :cite:`Vaswani.Shazeer.Parmar.ea.2017`、その訓練・サービング・スケーリングの仕組みを考察する。

<!--en:2bb20a08-->
まず、因果的スタックを一から構築して訓練することから始める。:numref:`sec_transformer-block` では、Pre-Norm 残差ストリーム（`RMSNorm` と `SwiGLU`）の内部でマルチヘッドアテンションと位置ごとのフィードフォワードネットワークを対にし、残差射影のスケーリング（$1/\sqrt{2L}$）およびクエリ・キー正規化（query–key normalization）が深さ方向にわたって信号ノルムとアテンションロジットを有界に保つ理由を示す。:numref:`sec_gpt` では、これらの因果的ブロックをトークン埋め込みと重み共有（weight tying）を用いた言語モデリングヘッドの間に積み重ね、条件を揃えたリカレントベースラインと比較しながら WikiText-103 上でバイトレベルの GPT を訓練する。さらに、事前学習済みの GPT-2 の重み（`124M`）を読み込んで温度と top-$k$ サンプリングを検証し、現代のオープンウェイト LLM に共通するアーキテクチャのレシピを抽出する（:numref:`sec_modern-recipe`）。素朴な自己回帰生成ではステップごとにプレフィックス全体を再計算するため、:numref:`sec_kv-cache` ではキー・値キャッシュ（key–value cache、KV キャッシュ）を導入し、ハードウェアのルーフライン上で計算律速（compute-bound）なプレフィルとメモリ帯域幅律速（memory-bandwidth-bound）なデコードを対比させたうえで、ヘッド数（グループ化クエリアテンションとマルチクエリアテンション）、チャネル幅（低ランクのキー・値射影）、および系列長（アテンションシンクを伴うスライディングウィンドウ）にわたってキャッシュを圧縮する。

<!--en:77ccb35a-->
すべてのタスクが因果的なテキスト生成というわけではない。:numref:`sec_transformer` では、因果的デコーダを、マスク言語モデリング（masked language modeling）によって訓練される双方向エンコーダ :cite:`Devlin.Chang.Lee.ea.2018` や、クロスアテンションで結合された系列対系列のエンコーダ・デコーダ（固定サイズの潜在ボトルネックを含む :cite:`Jaegle.Gimeno.Brock.ea.2021`）と対比させる。続く :numref:`sec_vision-transformer` では、重複のないピクセルパッチをトークン埋め込みへ射影することで双方向エンコーダを画像に直接適用し :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`、学習された 2D 位置構造とデータ効率を、パラメータ数を揃えた ResNet と比較する。

<!--en:fe1fa0b4-->
最後に、スケーリングと条件付き計算（conditional computation）に目を向ける。:numref:`sec_scaling-laws` では、パラメータ数（$N \approx 12 L d^2$）と訓練 FLOPs（$C \approx 6 N D$）を数え上げ、15 モデルのグリッドを訓練し、公開されている Chinchilla のべき乗則フィット :cite:`kaplan2020scaling,hoffmann2022training` を用いて IsoFLOP 曲線と計算量最適（compute-optimal）な配分を描く。あわせて、運用期間全体の推論コスト（inference cost）を考慮すると、現代の事前学習がなぜ Chinchilla のパラメータあたりトークン比をはるかに超えて訓練された、より小さなモデルへと向かうのかを説明する。さらに :numref:`sec_moe` では、密なフィードフォワードネットワークをルーティングされるエキスパート群に置き換えることで保持パラメータ容量とトークンあたり FLOPs を切り離し :cite:`Shazeer.Mirhoseini.Maziarz.ea.2017,fedus2022switch`、補助損失による負荷分散と損失を用いないルータバイアス制御を比較する :cite:`Wang.Gao.Zeng.ea.2024`。そして :numref:`sec_transformers-further-reading` では、正規化の配置、潜在アテンション（latent attention）、および投機的デコーディング（speculative decoding）にわたる拡張手法を概観する。

<!--en:e6c857bf-->
Transformer：1 つのブロックから最先端アーキテクチャまで<br>**残差ストリーム・Pre-Norm と SwiGLU・GPT・KV キャッシュとルーフライン・エンコーダと ViT・スケーリング則・混合エキスパート**

<!--en:296b0aa1-->
::: {.slide title="なぜアテンションだけでは不十分か"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:8fb6c38e-->
- **トークンごとの非線形性の欠如：**位置ごとの FFN がなければ、アテンション層を積み重ねても各トークンベクトル内で階層的な非線形特徴量を合成できない

<!--en:5b46cad7-->
- **深さ方向の不安定性と帯域幅の壁：**深い積層には Pre-Norm 残差ストリームが不可欠であり、生成時には全 $12 L d^2$ 個のブロック重みと KV キャッシュを HBM からストリーミングする

<!--en:4068a1dd-->
::: {.slide title="第 §{num:chap_transformers} 章における 4 つのアーキテクチャの柱"}

<!--en:c7d201c1-->
[アーキテクチャのロードマップ]{.kicker}

<!--en:253ff008-->
- **1. 構築と訓練（§{sec_transformer-block}〜§{num:sec_gpt}）：**Pre-Norm、`RMSNorm`、`QK-Norm`、$1/\sqrt{2L}$ スケーリング、`SwiGLU`、デコーダのみの GPT 事前学習

<!--en:af465b30-->
- **2. サービングと圧縮（§{sec_kv-cache}）：**KV キャッシュ、ルーフライン解析、GQA、低ランク MLA、アテンションシンク

<!--en:a69656bd-->
- **3. 再配線とスケーリング（§{sec_transformer}〜§{num:sec_moe}）：**エンコーダ、クロスアテンション、ViT パッチ、Chinchilla スケーリング則、スパース混合エキスパート（MoE）

<!--en:bf119ab2-->
::: {.slide title="第 §{num:chap_transformers} 章を通じた具体的な単一 GPU ベンチマーク"}

<!--en:ff9e6bb1-->
[通しの実験例]{.kicker}

<!--en:b30403f7-->
- 本章のあらゆるアーキテクチャ機構を、単一 GPU 上で数分以内に訓練・プロファイリングできる再現可能なモデルで検証する

<!--en:f8ebc089-->
- **エンドツーエンドの検証：**32 層の初期化監査とバイトレベルの WikiText-103 事前学習から、RTX 4090 でのルーフラインプロファイリング、15 モデルのスケーリンググリッド、8 エキスパートの MoE ルーティングまで

<!--en:9913bfa1-->
::: {.slide title="2017 年の Transformer から現代のレシピへ"}

<!--en:688e3215-->
[現代のレシピ]{.kicker}

<!--en:aeced3fa-->
- 本章のすべてのモデルはパラメータ化された単一のコンストラクタのインスタンスであり、そのデフォルト設定は最先端のオープンウェイト LLM（`RMSNorm`、`RoPE`、`SwiGLU`、GQA・MLA、MoE）と一致する

<!--en:b3f79a21-->
- **次節（§{sec_transformer-block} Transformer ブロック）：**Pre-Norm 配置、`RMSNorm`、`QK-Norm`、`SwiGLU` がいかにしてトークン崩壊を防ぎ、深さ方向にわたって勾配の振る舞いを良好に保つか
