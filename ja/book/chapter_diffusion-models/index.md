<!--en:98cdb6cb-->
# 拡散モデル

<!--en:ae2dba21-->
:numref:`chap_gans` の敵対的な生成器（generator）は、ノイズをデータへ写像する生成器に直接的な回帰目標（regression target）がないため、識別器（discriminator）またはクリティックという 2 つ目のネットワークを必要とした。拡散モデル（diffusion model）とフローマッチングモデルは、各訓練例をガウスノイズで段階的に破壊することによって回帰目標を構築する。小さなガウスノイズのステップを逆転させるには、ノイズを含む観測が与えられたもとでのクリーンなデータの条件付き期待値（conditional expectation）を推定する必要がある。

<!--en:0904a9bc-->
この視点は、非平衡熱力学とノイズ除去拡散確率モデル（denoising diffusion probabilistic model、DDPM）:cite:`sohl2015deep,ho2020denoising`、スコアベース生成モデリング（score-based generative modeling）:cite:`song2019generative,song2021score`、およびフローマッチングや確率的補間（stochastic interpolant）:cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022,Albergo.Boffi.VandenEijnden.2023` という 3 つの研究系譜を統合するものである。これら 3 つの定式化のいずれにおいても、単一のニューラルネットワークがクリーンなデータ（または等価的に、付加されたノイズ、スコア関数 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$、あるいは経路に沿った速度）をあらゆるノイズレベル $t \in [0, 1]$ で予測するように学習し、数値サンプラがその予測を $t = 1$ の純粋なノイズから $t = 0$ のデータへと逆向きに積分する。

<!--en:d29c91cc-->
敵対的なミニマックス目的関数を回帰に置き換えることで、GAN の 3 つの限界が解消される。すなわち、訓練に必要なのは二乗誤差で最適化される単一のネットワークのみであり、重み付き回帰損失は対数尤度（log-likelihood）の変分境界（さらに確率フロー ODE を通じた厳密な対数尤度）を与え、尤度目的関数は欠落したモードにペナルティを科す。本章全体を通じて、デフォルトの定式化では DDPM における分散保持（variance-preserving）経路上のノイズ予測器と並行して、線形経路 $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$ に沿った速度を回帰する :cite:`Esser.Kulal.Blattmann.ea.2024`。

<!--en:bcfb3e03-->
最初の 3 節では、真の密度、スコア関数、事後平均（posterior mean）、速度場（velocity field）が閉形式で得られる 3 つのガウス分布の 2 次元混合分布を用いて、数学的・アルゴリズム的な中核を完全に構築する。:numref:`sec_diffusion-images` では Fashion-MNIST 画像へと進み、時間条件付き U-Net、フレシェ距離（Fréchet distance）による評価、および分類器なしガイダンス（classifier-free guidance）を導入する。続く 4 節では、サンプラ（:numref:`sec_diffusion-samplers`）、尤度境界と設計空間（:numref:`sec_diffusion-design-space`）、潜在拡散（latent diffusion）と Transformer バックボーン（:numref:`sec_diffusion-latent`）、およびフロー写像（flow map）による少数ステップ蒸留（:numref:`sec_diffusion-few-step`）を検討する。最後に :numref:`sec_diffusion-discrete` では、ガウスノイズをトークンマスキングに置き換えることで、拡散を離散トークンへと拡張する。より長い数学的証明は :numref:`chap_mdl-dynamics` に掲載されており、外部のチュートリアル、講義、参照実装は :numref:`sec_diffusion-further-reading` にまとめられている。

<!--en:ff904380-->
拡散・フローベース生成モデル<br>**ノイズ除去とスコア関数 · ガウス確率経路 · ODE と SDE の積分 · U-Net と DiT · 高速サンプラ · 離散マスキング**

<!--en:76656029-->
::: {.slide title="なぜ拡散モデルが敵対的ミニマックスゲームを置き換えるか"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:14851172-->
ランダムノイズをデータへ写像する生成器には直接の教師目標がないため、GAN は競合する識別器を訓練せざるを得ない。段階的なガウス破壊を導入すると、あらゆるノイズレベルで明示的な回帰目標が得られる。

<!--en:69e4cf4b-->
- **単一ネットワークによる回帰：**訓練例 $\mathbf{x}_0$ を $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ へと破壊することで、1 つのネットワークが二乗誤差によりクリーンなデータ点、付加ノイズ、スコア関数、または速度へ直接回帰できる

<!--en:d614b1a5-->
- **厳密な尤度と境界：**重み付きノイズ除去損失は負の対数尤度 $-\log p_{\boldsymbol\theta}(\mathbf{x}_0)$ の上界を与え、決定論的な確率フロー ODE は変数変換によって厳密な対数密度を導く

<!--en:9b7f684b-->
- **モード被覆の保証：**期待負の対数尤度 $\mathbb{E}_p[-\log p_{\boldsymbol\theta}] = H(p) + D_{\textrm{KL}}(p \,\|\, p_{\boldsymbol\theta})$ の最小化は、いかなるデータモードの欠落にも大きなペナルティを科す

<!--en:62dbfe0d-->
::: {.slide title="ガウス経路と共通の対数信号対雑音比クロック"}

<!--en:7dd17f06-->
[順方向破壊と時刻の規約]{.kicker}

<!--en:3958f321-->
本章のすべての連続モデルは、クリーンなデータ（$t = 0$）と標準ガウスノイズ $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（$t = 1$）の間を補間する。

<!--en:4e3aa045-->
- **線形経路（フローマッチングのデフォルト）：**$\alpha_t = 1 - t$、$\sigma_t = t$ とし、各訓練ペアを一定の条件付き速度 $\dot{\mathbf{x}}_t = \boldsymbol\epsilon - \mathbf{x}_0$ で移動させる

<!--en:e712d695-->
- **分散保持（VP）経路（DDPM）：**$\alpha_t^2 + \sigma_t^2 = 1$ とし、$T = 1{,}000$ 個の離散順方向ステップにわたり単位分散を保つ

<!--en:46f3992f-->
- **標準的な対数信号対雑音比クロック：**$\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$ は $+\infty$（$t = 0$）から $-\infty$（$t = 1$）へと減少し、サンプラは常に $t = 1$ から $t = 0$ へと逆向きに積分する

<!--en:172b8637-->
::: {.slide title="1 つの事後平均による 4 つの回帰目標の統一"}

<!--en:75ce520f-->
[数学的な中核]{.kicker}

<!--en:552bed7f-->
二乗誤差は入力が与えられたもとでの条件付き期待値によって最小化されるため、訓練例ごとの目標で訓練すると事後平均 $\hat{\mathbf{x}}_0(\mathbf{x}_t) = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$ が復元される。

<!--en:0c6810e1-->
- **トゥイーディーの恒等式：**クリーンなデータ点の事後平均とノイズの事後平均 $\hat{\boldsymbol\epsilon} = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$ から、周辺密度 $p_t$ のスコア関数が次のように定まる
  $$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{\hat{\boldsymbol\epsilon}(\mathbf{x}_t)}{\sigma_t}$$

<!--en:e7977c67-->
- **目標のアフィン等価性：**$\mathbf{x}_t = \alpha_t \hat{\mathbf{x}}_0 + \sigma_t \hat{\boldsymbol\epsilon}$ が成り立つため、クリーンなデータの予測 $\hat{\mathbf{x}}_0$、ノイズ $\hat{\boldsymbol\epsilon}$、スコア関数 $\nabla \log p_t$、周辺速度 $\mathbf{u}_t = \dot\alpha_t \hat{\mathbf{x}}_0 + \dot\sigma_t \hat{\boldsymbol\epsilon}$ は互いにアフィンな再パラメータ化の関係にある

<!--en:e8d2cc35-->
- **逆時間積分：**ODE または SDE の数値ソルバが、$\mathbf{u}_t(\mathbf{x})$ を $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ から $\mathbf{x}_0$ まで積分する

<!--en:9f3d2260-->
::: {.slide title="文献間における時刻の規約の変換"}

<!--en:7c962db8-->
[拡散モデルとフローマッチングの論文を読む]{.kicker}

<!--en:19cde150-->
拡散モデルの論文はクリーンなデータを $t = 0$ に置くのに対し、フローマッチングの多くの論文は時間軸を反転させてノイズを $t = 0$ に置く。

<!--en:9dc1a6ff-->
- **本書の規約（拡散モデルの標準）：**クリーンなデータを $t = 0$（$\mathbf{x}_0$）、純粋なノイズを $t = 1$（$\mathbf{x}_1 = \boldsymbol\epsilon$）に置き、生成時は $t = 1$ から $t = 0$ まで積分する

<!--en:9df74f98-->
- **フローマッチング論文からの変換：**時刻 $t_{\textrm{FM}}$ を $1 - t$ に置き換え、初期ノイズ $\mathbf{x}_0^{\textrm{FM}}$ を $\boldsymbol\epsilon$ に、目標データ $\mathbf{x}_1^{\textrm{FM}}$ を $\mathbf{x}_0$ に対応させ、すべての速度ベクトルの符号を反転させる（$\mathbf{v}_t^{\textrm{FM}} = -\mathbf{u}_{1-t}$）

<!--en:c84ca325-->
- **対数信号対雑音比による妥当性確認：**論文中の時刻 $t = 0.8$ が高い SNR（$\lambda > 0$、ほぼデータ）と低い SNR（$\lambda < 0$、ほぼノイズ）のどちらに対応するかを常に確認する

<!--en:7818f8e3-->
::: {.slide title="第 §{num:chap_diffusion} 章を通じた具体的なベンチマーク"}

<!--en:f3f40f68-->
[実証的な裏付け]{.kicker}

<!--en:12d83417-->
すべての理論的主張とアーキテクチャ上の選択を、厳密な基準値や較正された特徴量の評価指標が存在する制御されたベンチマーク上で検証する。

<!--en:675e2b20-->
- **2D 3 モード混合ガウス分布（§{sec_diffusion-denoising}〜§{num:sec_diffusion-integration}、§{sec_diffusion-design-space}、§{sec_diffusion-few-step}）：**重み $(0.5, 0.3, 0.2)$、幅 $s = 0.5$ を持ち、厳密な密度、スコア関数、事後平均、速度、エントロピー（$\approx 2.5$ nat）が閉形式で得られる

<!--en:8cfbe205-->
- **Fashion-MNIST のピクセルと潜在表現（§{sec_diffusion-images}〜§{num:sec_diffusion-few-step}）：**$28 \times 28$ 画像（値域 $[-1, 1]$）と $7 \times 7 \times 4$ の VAE 潜在表現を、有限標本の実データ間における基準下限に対するフレシェ距離（FD）で評価する

<!--en:086ba720-->
- **離散トークン系列（§{sec_diffusion-discrete}）：**量子化された $32 \times 32$ グリッドペアと 784 トークンの二値衣類データをトークンマスキング $[\textrm{M}]$ で破壊する

<!--en:abec5550-->
::: {.slide title="第 §{num:chap_diffusion} 章のロードマップ：スコアマッチングからトークンまで"}

<!--en:123f19cf-->
[学習ガイド]{.kicker}

<!--en:61afbee2-->
- **§{sec_diffusion-denoising}〜§{num:sec_diffusion-integration} 基礎理論：**ノイズ除去スコアマッチング、トゥイーディーの公式、ガウス経路（$\lambda_t$）、逆時間 ODE・SDE 輸送（訓練済み MLP が厳密な 2D オラクル重みに誤差 $0.004$ 以内で一致）

<!--en:a2ad194e-->
- **§{sec_diffusion-images}〜§{num:sec_diffusion-design-space} 画像・高速ソルバ・設計空間：**U-Net + CFG（クラス一致率 $\approx 90\%$）、20 NFE のホイン法および多段ソルバ、連続時間 ELBO（$w(\lambda) \equiv 1$）、EDM 前処理

<!--en:1d327088-->
- **§{sec_diffusion-latent}〜§{num:sec_diffusion-discrete} 潜在表現・少数ステップ・離散トークン：**KL オートエンコーダ + `adaLN-Zero` DiT（$>10\times$ 少ない FLOPs）、2 時刻フロー写像蒸留（$2$ 回のジャンプで $15$ オイラーステップに匹敵）、吸収型マスク付き拡散（$0.17$ bpp）
