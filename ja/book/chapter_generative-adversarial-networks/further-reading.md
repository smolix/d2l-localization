<!--en:7902ddfd-->
# 関連文献

<!--en:34c6c7c0-->
本節では、本章の中核となる概念（密度比推定、ダイバージェンスによる定式化、相対論的目的関数、勾配ペナルティ、敵対的事後学習）を、より広範な文献およびオープンソース実装へと結びつける。

<!--en:2683e95e-->
## 推奨される入門文献とコードベース

<!--en:e2cac79e-->
1. **局所収束と勾配ペナルティ：**:citet:`Mescheder.Geiger.Nowozin.2018` は :numref:`sec_gan_convergence` で引用した局所収束定理を証明し、重みクリッピングや 1 中心ペナルティ（WGAN-GP）が破綻する均衡点近傍において、ゼロ中心勾配ペナルティ（$R_1$ と $R_2$）が交互勾配更新を安定化することを示した。著者らの [GAN stability リポジトリ](https://github.com/LMescheder/GAN_stability)には、Dirac-GAN および $R_1$ と $R_2$ の正則化項の実装が含まれている。

<!--en:16e9e296-->
2. **現代化された GAN ベースライン（R3GAN）：**:citet:`Huang.Gokaslan.Kuleshov.ea.2024` は、相対論的ペアリング損失を $R_1 + R_2$ ペナルティおよび現代的な畳み込みブロックと組み合わせ、適切に正則化された目的関数を用いれば初期の GAN で使われていた場当たり的なアーキテクチャ上の安定化手法の多くが不要になることを示した（[R3GAN 公式リポジトリ](https://github.com/brownvc/R3GAN)）。

<!--en:971361fe-->
3. **評価指標（適合率、再現率、FID、KID）：**:citet:`Kynkaanniemi.Karras.Laine.ea.2019` は、サンプルの忠実度（fidelity）とモード被覆（mode coverage）を切り分ける $k$-NN の適合率・再現率指標（:numref:`subsec_gan_precision_recall`）を導入した。[StyleGAN3 リポジトリ](https://github.com/NVlabs/stylegan3) :cite:`Karras.Aittala.Laine.ea.2021` は、フルスケールの FID、KID、適合率、再現率のリファレンス実装を提供している。

<!--en:fda0b759-->
4. **敵対的拡散蒸留：**DMD2 :cite:`Yin.Gharbi.Park.ea.2024` は、分布整合スコア蒸留（distribution-matching score distillation）と実画像に対して訓練された識別器を組み合わせることで、1 ステップの生徒生成器が複数ステップの拡散教師モデルを上回る性能を達成できるようにしている。

<!--en:e9b7586f-->
5. **対話的な可視化と歴史的サーベイ：**[NIPS 2016 チュートリアル](https://arxiv.org/abs/1701.00160) :cite:`Goodfellow.2016` と [ganhacks チェックリスト](https://github.com/soumith/ganhacks)には初期の GAN 訓練におけるヒューリスティクスがまとめられており、[GAN Lab](https://poloclub.github.io/ganlab/) :cite:`Kahng.Thorat.Chau.ea.2018` では 2D の生成器と識別器のダイナミクスをブラウザ上で可視化できる。また、大規模ベンチマークにおいて :citet:`Lucic.Kurach.Michalski.ea.2018` は、同等のハイパーパラメータ調整予算を与えれば、多くの敵対的目的関数（:numref:`sec_gan_objectives`）が同程度の FID を達成することを実証している。

<!--en:9a32726f-->
## 目的関数とミニマックスダイナミクス

<!--en:0ad069d1-->
- **最小二乗 GAN（LSGAN）の符号化スキーム：**:numref:`sec_gan_objectives` では、0-1 最小二乗識別器が $\tfrac14 \chi^2(p \,\|\, m)$ を評価することを示した。:citet:`Mao.Li.Xie.ea.2017` は一般の正解ラベル $b$（実データ）、$a$（偽データ）、および $c$（生成器の目標ラベル）を解析し、最適クリティックのもとで生成器の目的関数が $\tfrac14 (a + b - 2c)^2 + \tfrac14 (b - a)^2 \chi^2(p \,\|\, m)$ に帰着することを示した。これは同じ $\chi^2$ ダイバージェンスの正の定数倍である（任意の $b \neq a$ に対して成り立つ）。

<!--en:66c091ce-->
- **$f$-GAN の出力活性化関数：**クリティックの出力 $t$ が凸共役（convex conjugate）$f^*(t)$ の有限となる定義域に収まるようにするため（:numref:`sec_gan_objectives`）、:citet:`Nowozin.Cseke.Tomioka.2016` は定義域に応じた出力活性化関数 $t(v)$ をネットワークの生の出力 $v$ に適用している。具体的には、順方向 KL には恒等写像、逆方向 KL には $-e^{-v}$（$t < 0$）、イェンセン・シャノンには $\log 2 - \mathrm{softplus}(-v)$（$t < \log 2$）を用いる。

<!--en:5d940e57-->
- **一般の相対論的ダイバージェンス：**:citet:`Jolicoeur-Martineau.2020` は、相対論的ペアリング目的関数（:numref:`sec_gan_relativistic`）の較正された最適値が、任意の凹利得関数 $\ell$（ただし $\ell(0) = 0$、$\ell'(0) \neq 0$、$\sup_{t > 0} \ell(t) > 0$ を満たすもの）に対して正当なダイバージェンスを定義することを証明している。

<!--en:83ff3113-->
- **双線形ミニマックス問題に対する最適化アルゴリズム：**双線形問題 $\min_\theta \max_\psi \theta\psi$ において、ステップサイズ $h$ の同時勾配降下法はステップごとに原点からの距離を $\sqrt{1 + h^2}$ 倍に増幅させる。*外挿勾配*法（extragradient method）:cite:`Korpelevich.1976` は先読みした試行点で勾配を評価し、距離を $\sqrt{1 - h^2 + h^4} < 1$ 倍（$0 < h < 1$ のとき）へと縮小させる。*楽観的勾配降下法*（optimistic gradient descent）:cite:`Daskalakis.Ilyas.Syrgkanis.ea.2018` は直前ステップの勾配を用いてこの先読みを近似し、負のモメンタム（negative momentum）:cite:`Gidel.Askari-Hemmat.Pezeshki.ea.2019` は交互更新を安定化する。これらの手法がオプティマイザを変更するのに対し、大規模な画像 GAN では代わりに $R_1$/$R_2$ 勾配ペナルティによって目的関数を変更する（:numref:`sec_gan_convergence`）。

<!--en:6dbea5a1-->
## 固定された差異と学習される差異

<!--en:cef17807-->
クリティックネットワークを用いずに、固定された 2 サンプル間距離に対して生成器を直接訓練することはできるだろうか。Generative Moment Matching Networks :cite:`Li.Swersky.Zemel.2015` は固定カーネルの MMD（:numref:`sec_gan_objectives`）を最小化し、陰的最尤推定（Implicit Maximum Likelihood Estimation、IMLE）:cite:`Li.Malik.2018` はモード欠落を防ぐために各実データ例を最近傍の生成サンプルに対応させ、Sliced Wasserstein 生成器 :cite:`Deshpande.Zhang.Schwing.2018` はランダム射影にわたって 1D の最適輸送距離を平均する。しかし自然画像においては、ピクセル空間の固定カーネルやランダム射影では知覚的な差異を捉えることが難しく、これが MMD-GAN :cite:`Li.Chang.Cheng.ea.2017,Binkowski.Sutherland.Arbel.ea.2018` における敵対的カーネル学習の動機となった。

<!--en:acd8039a-->
## アーキテクチャ、特徴空間、データ効率

<!--en:cff84a2d-->
- **アーキテクチャの進化（2015〜2021 年）：**DCGAN :cite:`Radford.Metz.Chintala.2015` から R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` に至るまで、訓練の安定性と高解像度生成に向けたアーキテクチャ上の進歩が重ねられた。代表的なものとして、漸進的成長（progressive growing）:cite:`Karras.Aila.Laine.ea.2017`、SAGAN や BigGAN におけるスペクトル正規化（spectral normalization）と自己アテンション :cite:`Miyato.Kataoka.Koyama.ea.2018,Zhang.Goodfellow.Metaxas.ea.2019,Brock.Donahue.Simonyan.2019`、そして StyleGAN の潜在マッピングネットワーク、スタイル変調・復調、およびエイリアスフリーなリサンプリング :cite:`Karras.Laine.Aila.2019,Karras.Laine.Aittala.ea.2020,Karras.Aittala.Laine.ea.2021` が挙げられる。

<!--en:65c9d879-->
- **特徴空間と評価指標の落とし穴：**:citet:`Kynkaanniemi.Karras.Aittala.ea.2023` は、Inception-v3 の特徴量が ImageNet のクラス境界に強く影響されているのに対し、自己教師あり学習による DINOv2 の特徴量は人間の知覚的判断とより良く整合することを示した :cite:`Stein.Cresswell.Hosseinzadeh.ea.2023,Karras.Aittala.Kynkaanniemi.ea.2024`。さらに、エイリアシングを伴う画像リサイズは FID を歪めることがあり :cite:`Parmar.Zhang.Zhu.2022`、有限サンプルにおける FID の統計的バイアスはモデル依存の傾きをもって $O(1/n)$ でスケールする :cite:`Chong.Forsyth.2020`。

<!--en:4d1bda5c-->
- **クリティックにおける事前学習済み特徴：**Projected GAN :cite:`Sauer.Chitta.Muller.ea.2021` と StyleGAN-T :cite:`Sauer.Karras.Laine.ea.2023` は、凍結した事前学習済みの視覚バックボーンの上に軽量な識別器ヘッドを適用することで、収束を加速している。この手法を用いる場合、評価には必ず独立した特徴抽出器を用いるべきである。

<!--en:caecb887-->
- **勾配ペナルティの償却：**$R_1$ と $R_2$ はクリティックを通じた 2 次の誤差逆伝播法を必要とするため、StyleGAN2 は*遅延正則化*（lazy regularization、:citet:`Karras.Laine.Aittala.ea.2020`）として 16 ミニバッチごとに 1 回だけペナルティを評価する。一方、大規模な動画拡散モデルの事後学習では、微小なガウス入力摂動のもとでの有限差分（finite difference）によって $R_1$ を近似している :cite:`Lin.Xia.Ren.ea.2025`。

<!--en:42142a37-->
- **非リーク型データ拡張：**小規模なデータセットでは、クリティックは訓練画像を急速に暗記してしまう（:numref:`sec_dcgan`）。Adaptive Discriminator Augmentation (ADA) :cite:`Karras.Aittala.Hellsten.ea.2020` および DiffAugment :cite:`Zhao.Liu.Lin.ea.2020` は、勾配経路の内部で実画像と生成画像の両方に可逆かつ*非リーク*（non-leaking）な変換を適用することで、目標分布を損なうことなくクリティックの過学習を防ぐ。

<!--en:84f41cf8-->
## 条件付け、画像変換、現代のハイブリッドシステム

<!--en:2a5e6f9e-->
- **特徴変調層：**チャネルごとのアフィン変調 $a(c) \odot h + b(c)$（:numref:`sec_gan_conditional`）は、Conditional Batch Normalization :cite:`De-Vries.Strub.Mary.ea.2017`、FiLM :cite:`Perez.Strub.De-Vries.ea.2018`、スタイル変換のための Adaptive Instance Normalization (AdaIN) :cite:`Huang.Belongie.2017`、およびセマンティック画像生成のための空間的に変化する SPADE :cite:`Park.Liu.Wang.ea.2019` として現れる。

<!--en:d13d9eec-->
- **ペアあり・ペアなしの画像から画像への変換：**Pix2pix :cite:`Isola.Zhu.Zhou.ea.2017` は、U-Net 生成器とパッチレベルの識別器（PatchGAN）を組み合わせることで $L_1$ 再構成を鮮鋭化する。CycleGAN :cite:`Zhu.Park.Isola.ea.2017` はサイクル一貫性（cycle consistency、$F(G(x)) \approx x$）を課すことでペアなしドメインへと変換を拡張するが、ドメイン間でエントロピーが異なるとき、サイクル一貫性を持つ写像は入力元の情報を高周波のステガノグラフィ的摂動の中に隠蔽してしまうことがある :cite:`Chu.Zhmoginov.Sandler.2017`。

<!--en:900beb96-->
- **トークナイザ、少数ステップ蒸留、ボコーダ：**視覚トークナイザ（VQGAN、ViTok）では、デコーダの敵対的損失によってピクセルごとの PSNR と引き換えに知覚的な再構成品質を向上させる :cite:`Esser.Rombach.Ommer.2021,Hansen-Estruch.Yan.Chung.ea.2025,Hansen-Estruch.Chen.Ramanujan.ea.2026`。拡散蒸留においては、LADD :cite:`Sauer.Boesel.Dockhorn.ea.2024`（FLUX.1-schnell で採用、:citet:`BlackForestLabs.2024`）および SDXL-Lightning :cite:`Lin.Wang.Yang.2024` が潜在空間またはパッチ空間で直接識別器を適用する一方、MeanFlow :cite:`Geng.Deng.Bai.ea.2025` や連続時間一貫性モデル（continuous-time consistency model）:cite:`Lu.Song.2025` はクリティック不要の少数ステップ回帰目的関数を探求している。音声合成では、RAF :cite:`Lee.Choi.2026` が BigVGAN :cite:`Lee.Ping.Ginsburg.ea.2023` を :numref:`sec_gan_relativistic` の相対論的ペアリング目的関数で改良している。

<!--en:d598dcf4-->
- **陰的クリティックとしての尤度モデル（DDO）：**Direct Discriminative Optimization :cite:`Zheng.Chen.Chen.ea.2025` は、尤度モデルまたは拡散モデル $q_\theta$ を凍結した参照コピー $q_{\textrm{ref}}$ に対してファインチューニングする際、クリティック自体を対数尤度比 $D_\theta(x) = \beta \log(q_\theta(x) / q_{\textrm{ref}}(x))$ としてパラメータ化することで、別個の識別器ネットワークを不要にしている（演習問題 :exref:`ex_adversarial-losses-ddo-plays-log-loss`）。

<!--en:78ea4de4-->
[ディスカッション](https://d2l.discourse.group/t/36874)
