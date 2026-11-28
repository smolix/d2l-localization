<!--en:7902ddfd-->
# 関連文献

<!--en:243f93d1-->
本節では、本章で展開した手法を発展させる専門書、チュートリアル、参照実装、および一次文献をまとめる。

<!--en:21b2a03a-->
## 専門書、講義、コード

<!--en:08face5d-->
- [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) :cite:`Lai.Song.Kim.ea.2025` は、変分・スコアベース・フローベースの各経路に沿って拡散モデルを導出し、それらがどこで合流するかを示している。

<!--en:7dd3f191-->
- MIT の講義 6.S184 の[講義ノート、講義、ラボ](https://diffusion.csail.mit.edu/) :cite:`Holderrieth.Erives.2025` は、フローマッチング、潜在拡散 Transformer、および離散拡散（discrete diffusion）を扱っている。

<!--en:7224b94e-->
- [*Diffusion Meets Flow Matching*](https://diffusionflow.github.io/) :cite:`Gao.Hoogeboom.Heek.ea.2024` は、スケジュールの再スケーリングのもとで DDIM とオイラー法のフローマッチングサンプラが等価であることを、対話的な図を用いて解説している。

<!--en:94286aad-->
- [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) :cite:`Lipman.Havasi.Holderrieth.ea.2024` とその[ライブラリ](https://github.com/facebookresearch/flow_matching)は、:numref:`sec_diffusion-gaussian-paths` に対する完全な証明と参照コードを提供している。

<!--en:ecb3c42c-->
- [*Step-by-Step Diffusion*](https://arxiv.org/abs/2406.08929) :cite:`Nakkiran.Bradley.Zhou.ea.2025` は、確率解析（stochastic calculus）を用いず初等確率論のみから、決定論的サンプラと確率的サンプラ（:numref:`sec_diffusion-integration` および :numref:`sec_diffusion-samplers`）を導出している。

<!--en:292b57c7-->
- Sander Dieleman の[ブログ](https://sander.ai/)では、周波数空間でのノイズ除去 :cite:`Dieleman.2024`、潜在表現 :cite:`Dieleman.2025`、フロー写像 :cite:`Dieleman.2026`、および言語に対する連続拡散と離散拡散の比較 :cite:`Dieleman.2026b` が検討されている。

<!--en:f4863c1d-->
- 参照実装としては、[DDPM 公式リポジトリ](https://github.com/hojonathanho/diffusion) :cite:`ho2020denoising`、[Score SDE リポジトリ](https://github.com/yang-song/score_sde_pytorch) :cite:`song2021score`、および [MDLM リポジトリ](https://github.com/kuleshov-group/mdlm) :cite:`Sahoo.Arriola.Schiff.ea.2024` がある。

<!--en:c50e15d7-->
- [*Deep Generative Models* 講義](https://kuleshov-group.github.io/dgm-website/) :cite:`Kuleshov.2023`、Yang Song の[スコアベースモデルのチュートリアル](https://yang-song.net/blog/2021/score/) :cite:`Song.2021`、Lilian Weng の [*What Are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) :cite:`Weng.2021`、および :citet:`Luo.2022` は、スコアマッチングと変分拡散に関する平易な解説を提供している。

<!--en:bad89039-->
一次論文のなかでも、次の 5 編はとりわけ基礎をなすものである。:citet:`Karras.Aittala.Aila.ea.2022`（EDM）はサンプラ、前処理（preconditioning）、ノイズスケジュールを切り分け、:citet:`Esser.Kulal.Blattmann.ea.2024`（Stable Diffusion 3）はレクティファイドフローとマルチモーダル拡散 Transformer を組み合わせた。また、:citet:`Lu.Zhou.Bao.ea.2025`（DPM-Solver++）はガイダンス下での高次ソルバを解析し、:citet:`Lu.Song.2025`（sCM）は大規模環境における連続時間一貫性モデル（consistency model）を安定化し、:citet:`Sahoo.Arriola.Schiff.ea.2024`（MDLM）はマスク付き離散拡散の見通しのよい定式化を与えている。隣接するトピックは :numref:`sec_gan-further-reading`（敵対的蒸留および分布整合蒸留、FID 評価）ならびに :numref:`sec_mdl-dynamics-further-reading`（指数積分器、スコア誤差の上界、シュレーディンガーブリッジ）で扱う。

<!--en:0bde3a80-->
## エネルギーベースモデルとスコアマッチング

<!--en:358d6d8c-->
**エネルギーベースモデルの訓練：**:citet:`Song.Kingma.2021` は、正規化されていない密度に対する最尤法による訓練を概観している。コントラスティブダイバージェンスはデータ点を初期値として短い MCMC 連鎖を開始するため :cite:`Hinton.2002`、離れたモード間での混合が妨げられるが（:numref:`sec_diffusion-denoising`）、持続的コントラスティブダイバージェンス（persistent contrastive divergence）はパラメータ更新をまたいで連鎖の状態を保持する :cite:`Tieleman.2008,Du.Mordatch.2019`。

<!--en:2edb1f15-->
**ノイズ対照推定：**微分によって正規化定数を消去する代わりに、:citet:`Gutmann.Hyvarinen.2012` は正規化定数を学習可能なパラメータとして扱い、既知のノイズ分布に対するロジスティック回帰によってモデルを訓練する。

<!--en:49172c91-->
**スコアマッチングの限界：**低密度の谷を隔てたモードの重みに対してフィッシャーダイバージェンスが不感になる性質（:numref:`sec_diffusion-denoising`）は :citet:`Wenliang.Kanagawa.2020` によって解析されており、:citet:`Koehler.Heckett.Risteski.2022` は等周定数（isoperimetric constant）を用いてスコアマッチングの統計的効率に境界を与えている。

<!--en:843ff3c7-->
**ノイズラダーのヒューリスティクス：**:citet:`Song.Ermon.2020` は、幾何ノイズラダー（:numref:`sec_diffusion-gaussian-paths`）に対して 4 つの実践的な規則を与えた。すなわち、最大ノイズスケールを訓練データ間の最大ペアワイズ距離に設定すること、高確率シェルが重なり合うほど隣接レベルを十分に近づけて選ぶこと（演習問題 :exref:`ex_gaussian-paths-ladder-spacing`、:numref:`sec_diffusion-gaussian-paths`）、ランジュバン法のステップサイズを調整すること、および重みに指数移動平均を適用することである。

<!--en:43c71512-->
**経験ベイズ：**Tweedie の公式は経験ベイズ推定（empirical Bayes estimation）に端を発しており :cite:`Robbins.1956`、:citet:`Efron.2011` が高次元推定における選択バイアスの補正にこれを用いる手法を広めた。

<!--en:6b534ab1-->
## 画像ネットワークとガイダンス

<!--en:f06c3c9d-->
**大規模な U-Net アーキテクチャ：**:citet:`Dhariwal.Nichol.2021`（ADM）は、DDPM の U-Net に多重解像度アテンション、残差リサンプリング、および適応グループ正規化を追加した。:citet:`Karras.Aittala.Lehtinen.ea.2024`（EDM2）は、活性化値と重みの大きさを保つように U-Net を再設計し、周期的なチェックポイントからの事後的な EMA 再構成を導入した。

<!--en:6f1d843b-->
**飽和のないガイダンス：**大きなガイダンス強度はピクセル強度を飽和させることがある（演習問題 :exref:`ex_image-diffusion-thresholding`、:numref:`sec_diffusion-images`）。:citet:`Sadat.Hilliges.Weber.2025` は、ガイダンスによる更新を条件付き予測に平行な成分と直交する成分へ分解し、平行成分を減衰させることで飽和を防いでいる。

<!--en:8ecba538-->
**空間的条件付け：**ControlNet :cite:`Zhang.Rao.Agrawala.2023` は、ゼロ初期化された層を介して接続したエンコーダの訓練可能なコピーを用いることで、事前学習済みのテキスト画像生成モデルを空間マップ（エッジ、深度、姿勢）で条件付ける。

<!--en:d3bea0c1-->
## 評価、暗記、汎化

<!--en:a8fd78b6-->
**プロンプトアライメント：**フレシェ距離はプロンプトへの忠実度ではなく分布の類似度を測るため、CLIPScore :cite:`Hessel.Holtzman.Forbes.ea.2021` は共通埋め込みにおける画像とキャプションの類似度を測定し、一方 GenEval :cite:`Ghosh.Hajishirzi.Schmidt.2023` は物体検出器を用いて物体の数、位置、属性を検証する。

<!--en:26873971-->
**暗記：**:citet:`Carlini.Hayes.Nasr.ea.2023` は大規模なテキスト画像生成モデルから訓練画像とほぼ同一の重複画像を抽出しており、:citet:`Biroli.Bonnaire.deBortoli.ea.2024` は逆過程においてクラスの確定（種分化、speciation）と訓練データ点の暗記（崩壊、collapse）に対応する異なる力学的レジームを特定している。

<!--en:f95f5085-->
**ノイズ除去器の汎化：**ノイズ除去目的関数の厳密な経験最小点は有限の訓練セットを暗記してしまうが、:citet:`Bertrand.Gagneux.Massias.ea.2025` は目標ノイズだけでは汎化を説明できないことを示している。互いに素な部分集合で訓練されたノイズ除去器はほぼ同一のスコア場を学習し :cite:`Kadkhodaie.Guth.Simoncelli.ea.2024`、それは主として畳み込みの局所性と同変性によって規定される :cite:`Kamb.Ganguli.2025`。サーベイについては :citet:`Massias.Bertrand.2026` を参照のこと。

<!--en:d29b7e35-->
## サンプラ、編集、逆問題

<!--en:f945cc79-->
**ガイダンス付きサンプリングのためのソルバ：**大きなガイダンス強度は高次のノイズ予測ソルバを不安定にする（演習問題 :exref:`ex_samplers-dpm-solver`、:numref:`sec_diffusion-samplers`）。そこで DPM-Solver++ :cite:`Lu.Zhou.Bao.ea.2025` は代わりにクリーン画像の推定値を用いて多段更新をパラメータ化し、UniPC :cite:`Zhao.Bai.Rao.ea.2023` は追加計算コストのない修正子（corrector）ステップを加えている。

<!--en:278a52df-->
**ODE ソルバにおける確率性：**:citet:`Karras.Aittala.Aila.ea.2022` はホイン法の各ステップの前に少量のノイズを注入し、一方 Restart サンプリング :cite:`Xu.Deng.Cheng.ea.2023` は決定論的な ODE 積分とより大きな順方向ノイズのジャンプを交互に繰り返すことで、蓄積した離散化誤差を縮小させる。

<!--en:c02f2f27-->
**厳密な反転：**DDIM 反転（DDIM inversion、:numref:`sec_diffusion-samplers`）は分類器なしガイダンスのもとで離散化誤差を蓄積するため、ヌルテキスト反転（null-text inversion）:cite:`Mokady.Hertz.Aberman.ea.2023` はガイダンスなしの軌跡の周囲で無条件埋め込みを最適化し、一方 EDICT :cite:`Wallace.Gokul.Naik.2023` は結合されたアフィン変換を用いて厳密な可逆性を実現する。

<!--en:d5ace9b1-->
**再訓練なしの逆問題：**ブラー除去や超解像などの線形逆問題に対し、Diffusion Posterior Sampling（DPS）:cite:`Chung.Kim.Mccann.ea.2023` は事後平均 $\hat{\mathbf{x}}_0(\mathbf{x}_t)$ において測定尤度の勾配を近似し、DDRM :cite:`Kawar.Elad.Ermon.ea.2022` は測定作用素のスペクトル領域で動作する。

<!--en:01e8f000-->
## スケジュール、重み付け、尤度

<!--en:d3bf9c07-->
**学習される逆過程の分散：**コサインスケジュール（:numref:`sec_diffusion-gaussian-paths`）と並び、:citet:`Nichol.Dhariwal.2021` は逆過程の分散を $\tilde\beta_k$ と $\beta_k$ の間の学習される補間としてパラメータ化し、変分境界を用いて訓練している。

<!--en:ad9f0f6f-->
**Min-SNR 重み付け：**:citet:`Hang.Gu.Li.ea.2023` は、クリーン画像の二乗誤差に $\min(\mathrm{SNR}_t, \gamma)$ の重みを掛けることで（等価的にノイズ誤差へ $\min(1, \gamma e^{-\lambda})$ の重みを掛けることに相当）、低ノイズのステップが勾配更新を支配するのを防いでいる。

<!--en:189c8f12-->
**尤度最適化：**変分拡散モデル（variational diffusion model、VDM）:cite:`Kingma.Salimans.Poole.ea.2021` は ELBO 推定量の分散を最小化するようにノイズスケジュールを最適化する。また、:citet:`Song.Durkan.Murray.ea.2021` は連続時間スコアマッチングに対する尤度重み付けスケジュールを導出し、:citet:`Zheng.Lu.Chen.ea.2023` は確率フロー ODE に対する厳密な尤度訓練と評価を改善している。

<!--en:6e0a285c-->
## 潜在表現、テキスト、動画

<!--en:2b27eb53-->
**テキスト条件付け：**大規模なテキスト画像生成モデルは、凍結した CLIP :cite:`radford2021learning` や T5 :cite:`raffel2020exploring` のテキストエンコーダを用い、クロスアテンション（:numref:`sec_diffusion-latent`）または画像・テキスト結合自己アテンション :cite:`Esser.Kulal.Blattmann.ea.2024` を介して条件付けを行う。:citet:`saharia2022photorealistic` は、テキストエンコーダをスケールする方が、U-Net をスケールするよりもプロンプトアライメントを大きく向上させることを示した。

<!--en:e46427f5-->
**表現アライメントとオートエンコーダ：**REPA :cite:`Yu.Kwak.Jang.ea.2025` は、中間の隠れ状態を事前学習済みの視覚エンコーダと整合させることで潜在 DiT の訓練を高速化する。Deep Compression Autoencoders（DC-AE）:cite:`Chen.Cai.Chen.ea.2025` は空間方向のダウンサンプリングを $8\times$ 超へと拡大し、表現オートエンコーダ（representation autoencoder）:cite:`Zheng.Ma.Tong.ea.2026` は凍結したセマンティックエンコーダ（DINO など）と学習されるデコーダを組み合わせる。

<!--en:f2481849-->
**動画生成：**動画拡散モデルは、3D 時空間オートエンコーダと、空間および時間にわたる分解型または結合型のアテンションを適用する :cite:`Ho.Salimans.Gritsenko.ea.2022,Blattmann.Dockhorn.Kulal.ea.2023,Polyak.Zohar.Brown.ea.2024`。

<!--en:0456db4f-->
## 少数ステップ生成

<!--en:90de5be8-->
**漸進的蒸留とガイダンス蒸留：**漸進的蒸留（progressive distillation）:cite:`Salimans.Ho.2022` は教師モデルのステップ数を反復的に半減させる手法であり、低 SNR における数値安定性のために $\mathbf{v}$ 予測を導入した。:citet:`Meng.Rombach.Gao.ea.2023` は、分類器なしガイダンスをガイダンス強度 $\gamma$ で条件付けられた単一パスへと蒸留している。

<!--en:510cb2e1-->
**大規模環境における一貫性とフロー写像：**改良型一貫性訓練（improved consistency training、iCT）:cite:`Song.Dhariwal.2024` は離散時間において教師モデルを不要にし、一方 sCM :cite:`Lu.Song.2025` は三角関数経路と接線正規化（tangent normalization）を用いて ImageNet スケールでの連続時間一貫性訓練を安定化する。Improved MeanFlow :cite:`Geng.Lu.Wu.ea.2026` は瞬間速度へ直接回帰し、:citet:`Boffi.Albergo.VandenEijnden.2025` はオイラー形式、ラグランジュ形式、および合成形式のフロー写像恒等式にわたる自己蒸留を解析している。

<!--en:cc839eee-->
## 離散拡散

<!--en:ccc7c374-->
**代替カーネルと再マスキング：**離散フローマッチング :cite:`Gat.Remez.Shaul.ea.2024` は、単体上の確率経路を通じて離散生成を定式化する。一様破壊カーネルを用いれば任意のステップでトークンを修正でき、:citet:`Sahoo.Deschenaux.Gokaslan.ea.2025` は一様離散拡散をしきい値処理されたガウス拡散と結びつけている。ReMDM :cite:`Wang.Schiff.Sahoo.ea.2025` は吸収型拡散に再マスキングを導入し、:citet:`Kim.Shah.Kontonis.ea.2025` は数独などの推論タスクにおいて確信度順のアンマスキングが高い効果を発揮することを実証している。

<!--en:badadf1b-->
**拡散言語モデル：**LLaDA :cite:`Nie.Zhu.You.ea.2025` は 8B のマスク付き拡散言語モデルをゼロから訓練し、一方 Dream :cite:`Ye.Xie.Zheng.ea.2025` は自己回帰モデルのチェックポイントから初期化する。ブロック拡散（block diffusion）:cite:`Arriola.Gokaslan.Chiu.ea.2025` はブロック間の自己回帰生成とブロック内の並列拡散を組み合わせ、Fast-dLLM :cite:`Wu.Zhang.Xue.ea.2026` は KV キャッシュの再利用（:numref:`sec_kv-cache`）を追加している。なお :citet:`Zheng.Chen.Mao.ea.2025` は、32 ビットの Gumbel-max サンプリングがサンプリング温度を人為的に低下させ、サンプルエントロピーを犠牲にして報告される生成パープレキシティを実態より良く見せてしまうことに警鐘を鳴らしている。

<!--en:224e09cf-->
## 事後学習と視覚・テキスト以外の応用

<!--en:6f051c20-->
**強化学習と選好最適化：**DDPO :cite:`Black.Janner.Du.ea.2024` と Flow-GRPO :cite:`Liu.Liu.Liang.ea.2025` は、拡散モデルおよびフローモデルの確率的な逆遷移に対して方策勾配更新（:numref:`sec_rl_sequences`）を適用する。Diffusion-DPO :cite:`Wallace.Dang.Rafailov.ea.2024` は、計算困難なモデルの対数尤度を拡散モデルの ELBO で置き換えることにより、直接選好最適化を適合させている。

<!--en:aafea46a-->
**音声、ロボティクス、タンパク質：**DiffWave :cite:`Kong.Ping.Huang.ea.2021` は生の音声波形に拡散を適用する。拡散方策（diffusion policy）:cite:`Chi.Feng.Du.ea.2023` は視覚観測に条件付けられた多峰性のロボット行動軌跡をモデル化し、直接的な MSE 方策回帰におけるモード平均化の破綻を回避する。マルチモーダルなタンパク質モデル :cite:`Campbell.Yim.Barzilay.ea.2024` は、連続的な 3D 原子座標と離散的なアミノ酸配列を同時に拡散させる。

<!--en:f75d086a-->
[ディスカッション](https://d2l.discourse.group/t/36894)
