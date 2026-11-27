<!--en:4c69064d-->
# 敵対的生成ネットワーク

<!--en:ff27bff3-->
100 次元の確率ベクトル $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ を抽出し、それをニューラルネットワーク $G(\mathbf{z})$ に通して画像を生成するとする。このような生成器ネットワークを記述するだけなら数行のコードで済むが、難しいのはその訓練である。観測されたラベルやトークンの対数尤度（log-likelihood）を最大化する分類器や自己回帰言語モデルとは異なり、画像を直接出力する生成器は、最大化すべき明示的な密度 $q(\mathbf{x})$ を持たない画像上の陰的な分布（implicit distribution）を定義する。サンプリングはできるが尤度を評価できないとき、生成モデル（generative model）をどのように訓練すればよいのだろうか。

<!--en:9cc971e5-->
分類器を用いれば、サンプルだけから 2 つの分布を比較できることはすでに見てきた。:numref:`subsec_covariate-shift-correction` で共変量シフト（covariate shift）を補正した際には、訓練入力とテスト入力を識別する二値分類器を訓練し、その最適ロジットから 2 つの分布間の対数密度比（log density ratio）を復元した。:citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` は、この 2 サンプル比較を生成モデルの訓練原理へと発展させた。すなわち、*敵対的生成ネットワーク*（generative adversarial network、GAN）では、実データと生成サンプルを識別するように*識別器*（discriminator、または*クリティック*）$D$ を訓練し、同時に、識別器が実データと見分けられないようなサンプルを生成するように*生成器*（generator）$G$ を訓練する。識別器の勾配を通じて、生成器はどちらの密度も明示的に書き下すことなく、自身のサンプルがデータ分布に対して過剰または過少になっている領域を学習する。

<!--en:3a0ca300-->
汎用的な画像生成においては、拡散モデルやフローマッチングモデル（:numref:`chap_diffusion`）が単体の GAN にほぼ取って代わったものの、敵対的目的関数（adversarial objective）は依然として現代の生成システムの中核をなす構成要素であり、潜在拡散モデル（latent diffusion model）を支える潜在オートエンコーダや、ニューラル音声ボコーダ、複数ステップの拡散教師モデルから蒸留された 1 ステップ生成器などの訓練に用いられている。

<!--en:97299268-->
本章では、敵対的モデリングを 3 段階で展開する。まずは目的関数とダイバージェンスから始める。:numref:`sec_basic_gan` では対数損失における最適識別器、そのイェンセン・シャノン・ダイバージェンス（Jensen–Shannon divergence）、および非飽和生成器損失（non-saturating generator loss）を導出する。続く :numref:`sec_gan_objectives` ではこの枠組みを $f$ ダイバージェンスと、積分確率指標（integral probability metric）である最大平均値差異（maximum mean discrepancy、MMD）および 1 次ワッサースタイン距離（Wasserstein-1 distance）へ一般化し、:numref:`sec_gan_relativistic` ではブラッドリー・テリー（Bradley–Terry）ランキングに基づく相対論的ペア型クリティック（RpGAN）を導入する。次に、:numref:`sec_gan_convergence` では Dirac-GAN 上でミニマックス訓練ダイナミクス（training dynamics）を調べ、正則化なしの更新が回転したり外側へ渦巻状に発散したりする理由と、ゼロ中心勾配ペナルティ（$R_1$ と $R_2$）が R3GAN のレシピをいかに安定化するかを示す :cite:`Huang.Gokaslan.Kuleshov.ea.2024`。最後に、:numref:`sec_dcgan` と :numref:`sec_gan_conditional` では、畳み込みアーキテクチャと射影条件付きアーキテクチャを、特徴空間の評価指標（FID、KID、$k$-NN の適合率と再現率）とあわせて画像生成へとスケールさせ、:numref:`sec_gan_beyond` では拡散蒸留、画像トークナイザ、音声ボコーダにおいて敵対的クリティックが条件付き平均によるぼけをいかに防ぐかを解説する。

<!--en:5f1d3268-->
## 記法 {.unnumbered}

<!--en:12d973c1-->
:numref:`tab_gan_notation` に、本章全体で用いる記法をまとめる。ダイバージェンスとエントロピーは nat を単位として測り、:numref:`chap_mdl-information-theory` に従う。すなわち、$D_{\textrm{KL}}$ はカルバック・ライブラー・ダイバージェンス、$\mathrm{JS}$ はイェンセン・シャノン・ダイバージェンス、$H$ は微分エントロピーまたは離散エントロピー、$H_b(t)$ はパラメータ $t$ のベルヌーイ分布の二値エントロピーを表す。

<!--en:bdf56af1-->
:本章で用いる記法。

<!--en:e31ebed0-->
| 記号 | 意味 | 初出 | よく使われる別記法 |

<!--en:800c3ddf-->
| $p$、$q$、$m = \tfrac12(p + q)$ | データ密度、生成器密度、均衡混合密度 | :numref:`sec_basic_gan` | $p_{\textrm{data}}$、$p_g$ |

<!--en:3f5823cd-->
| $x \sim p$、$x' \sim q$、$y \in \{0, 1\}$ | 実サンプル、生成サンプル、出所ラベル（$1$ = 実データ） | :numref:`sec_basic_gan` | |

<!--en:a1a1f280-->
| $z \in \mathbb{R}^{d_z}$、$G$ | 潜在ベクトルと生成器ネットワーク（$x' = G(z)$） | :numref:`sec_basic_gan` | |

<!--en:86426113-->
| $D$、$V(D)$ | クリティック・識別器の真実性ロジット、ミニマックス価値関数 | :numref:`sec_basic_gan` | |

<!--en:4ad45273-->
| $\rho = p/q$、$\lambda = \log(p/q)$ | 密度比と対数密度比 | :numref:`sec_basic_gan` | |

<!--en:b3a5ee6c-->
| $w_{\textrm{sat}}$、$w_{\textrm{ns}}$、$w_{\textrm{Rp}}$ | サンプルごとの生成器勾配重み（飽和型、非飽和型、相対論的） | :numref:`sec_basic_gan`、:numref:`sec_gan_relativistic` | |

<!--en:36ed181b-->
| $\eta = p/(p + q)$ | ベイズクラス事後確率 $P(y = 1 \mid x)$ | :numref:`sec_gan_objectives` | |

<!--en:6dccf4fe-->
| $\ell$、$L(\eta)$、$\Delta_\ell(p, q)$ | マージン利得、条件付きベイズリスク、ベイズリスクギャップ | :numref:`sec_gan_objectives` | |

<!--en:c0a0882c-->
| $D_f(p \,\|\, q)$、$k(x, x')$ | 凸関数 $f$ による $f$ ダイバージェンス、正定値カーネル | :numref:`sec_gan_objectives` | |

<!--en:a28ded29-->
| $\Phi(D)$、$d_{\mathrm{Rp}}(p, q)$ | 相対論的ペアリング目的関数とペアリングダイバージェンス | :numref:`sec_gan_relativistic` | |

<!--en:c7f6845e-->
| $R_1$、$R_2$、$\gamma$ | 実サンプルおよび生成サンプルにおけるゼロ中心勾配ペナルティ、ペナルティ重み | :numref:`sec_gan_convergence` | |

<!--en:27a93164-->
| $c$、$e_c$、$\phi(x)$ | 条件付け変数、条件埋め込み、特徴表現 | :numref:`sec_dcgan`、:numref:`sec_gan_conditional` | |

<!--en:61f230be-->
敵対的生成ネットワーク<br>**陰的密度推定 · ミニマックス 2 サンプルゲーム · $f$ ダイバージェンスと IPM · 回転ダイナミクスと勾配ペナルティ · 条件付き生成と知覚クリティック**

<!--en:84462dec-->
::: {.slide title="尤度を用いない陰的生成器の訓練"}

<!--en:86e68ef5-->
[中心的な問題]{.kicker}

<!--en:42ff4ccf-->
**陰的生成器**は、潜在ベクトル $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_z})$（$d_z = 100$）を抽出し、1 回の順伝播で画像 $\mathbf{x}' = G(\mathbf{z}) \in \mathbb{R}^{d_x}$ を出力することで、計算可能な密度 $q(\mathbf{x})$ を最大化することなくサンプリング手続きを定義する。

<!--en:65e2309c-->
- **低次元多様体上の台：**$100$ 個の潜在座標を $784$（Fashion-MNIST）または $12{,}288$（$64 \times 64$ RGB）ピクセルへ写像すると、生成サンプルはルベーグ密度を持たない薄い曲面上に集中する

<!--en:885abde9-->
- **尤度が誤解を招く理由（Theis et al., 2016）：**真の密度に $99\%$ のノイズを混ぜた $\tilde{p} = 0.01\,p_{\textrm{data}} + 0.99\,p_{\textrm{noise}}$ は $\log \tilde{p}(\mathbf{x}) \ge \log p_{\textrm{data}}(\mathbf{x}) - \log 100$ を満たし、最適モデルとの差はわずか $4.6$ nat（画像全体では数千 nat）に収まるにもかかわらず $99\%$ の確率で純粋なノイズを出力する

<!--en:c68320d4-->
- **中心アイデア：**尤度の評価を、生成サンプルを直接検査する学習済みの **2 サンプルクリティック** $D$ に置き換える

<!--en:94639566-->
::: {.slide title="密度比推定としての 2 サンプル分類"}

<!--en:fa562a5e-->
[中核となる仕組み]{.kicker}

<!--en:7d022792-->
Goodfellow et al. (2014) は、**生成器** $G$ と**識別器（クリティック）**$D$ を競わせることで、2 サンプル二値分類を生成モデルの訓練原理へと発展させた。

<!--en:f1fa3993-->
- **均衡 2 サンプルゲーム：**実サンプル $x \sim p$ に $y = 1$、生成サンプル $x' = G(z) \sim q$ に $y = 0$ のラベルをそれぞれ事前確率 $\frac{1}{2}$ で付与する

<!--en:798e5da2-->
- **最適クリティックによる対数密度比の復元：**二値対数損失を最小化するとベイズ事後確率 $\sigma(D^\star(x)) = \frac{p(x)}{p(x) + q(x)}$ が得られ、最適ロジットは次の式になる
  $$D^\star(x) = \log \frac{p(x)}{q(x)} = \lambda(x)$$

<!--en:88707ffb-->
- **密度を用いない勾配：**$D$ を通じた逆伝播により生成サンプル $x' = G(z)$ を $\nabla_x D(x')$ の上り方向へ押し出し、過剰に表現された領域（$q > p \implies D^\star < 0$）から過少に表現された領域（$p > q \implies D^\star > 0$）へと確率質量を移動させる

<!--en:c2d36363-->
::: {.slide title="あらゆる敵対的ゲームに共通する 3 つの設計軸"}

<!--en:69ffac88-->
[統一的な枠組み]{.kicker}

<!--en:657ede69-->
- **古典的ミニマックスが飽和する理由：**生成サンプルがデータの台を外すと、$\max_D V(D) = 2\,\mathrm{JS}(p, q) - 2\log 2$ は $\mathrm{JS}(p, q) = \log 2 \approx 0.6931$ nat で平坦化し、勾配消失が生じる

<!--en:5dfbaa42-->
- **直交する 3 つの改良軸：**あらゆる GAN の目的関数は、クリティックの推定対象（$D^\star(\lambda)$ 対証人関数 $T$）、サンプルの採点方法（単一サンプル対ブラッドリー・テリーのペア）、クリティック勾配の正則化方法（$R_1 + R_2$）のいずれかを変更する

<!--en:90fe55c9-->
- **現代化された R3GAN レシピ（Huang et al., 2024）：**ペアワイズランキング（`RpGAN`）とゼロ中心ペナルティ $R_1 + R_2$ を組み合わせ、BatchNorm なしで回転ダイナミクスを減衰させる

<!--en:adf10ab6-->
::: {.slide title="第 §{num:chap_gans} 章のロードマップ：理論・ダイナミクス・システム"}

<!--en:88710d83-->
[数理とアーキテクチャのロードマップ]{.kicker}

<!--en:f8aad7b5-->
- **パート I：目的関数とダイバージェンス（§{sec_basic_gan}〜§{num:sec_gan_relativistic}）：**対数損失による密度比推定（$D^\star = \log(p/q)$）および非飽和重み（$2\,D_{\textrm{KL}}(m \,\|\, p)$）から、$f$ ダイバージェンス、IPM（$W_1$、MMD）、相対論的ペアリング（$d_{\mathrm{Rp}}$）まで

<!--en:5d265cc3-->
- **パート II：ダイナミクスと収束（§{sec_gan_convergence}）：**Dirac-GAN 上でペナルティなしのミニマックス更新が周回または外側へ渦巻状に発散する理由と、ゼロ中心ペナルティ $R_1 + R_2$ がヤコビ行列の固有値を開左半平面へ移動させる仕組み

<!--en:03c5e235-->
- **パート III：画像・条件付け・応用（§{sec_dcgan}〜§{num:sec_gan_beyond}）：**DCGAN 対正規化なしバックボーン、FID・KID と $K$-NN の適合率・再現率、射影クリティック、拡散蒸留（ADD・DMD2）

<!--en:72b72fbb-->
::: {.slide title="通しベンチマークと学習ガイド"}

<!--en:05adf916-->
[章を貫く学習の流れ]{.kicker}

<!--en:2d86ef20-->
- **厳密密度を持つ 2D ガウス分布と混合分布（§{sec_basic_gan}〜§{num:sec_gan_relativistic}）：**クリティックのロジットを解析解 $\lambda(x) = \log(p(x)/q(x))$ と照合し、生成器の当てはまりを $D_{\textrm{KL}}(q \,\|\, p)$ と $\mathrm{JS}(p, q)$ の厳密な nat 値で測定する

<!--en:ea8a9616-->
- **25 個のガウス分布からなる $5 \times 5$ グリッド（§{sec_gan_convergence}）：**ペアワイズランキングと $R_1 + R_2$ 勾配ペナルティのどちらがモード欠落を解消するかを切り分ける

<!--en:4e8f8ee1-->
- **画像生成と条件付き生成（§{sec_basic_gan}、§{sec_dcgan}〜§{num:sec_gan_conditional}）：**Fashion-MNIST（$28 \times 28$）、Pokémon スプライト（$64 \times 64$）、10 クラス CIFAR-10（$32 \times 32$）

<!--en:9c0b1285-->
- **二峰性回帰と蒸留（§{sec_gan_beyond}）：**厳密な閉形式のノイズ除去教師モデルを備えた 2 バンドベンチマーク $x = c \pm 1 + 0.05\,\xi$

<!--en:e9d832ba-->
::: {.slide title="概念チェック：敵対的定式化の整理"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:d36a1bfa-->
本章の概要から、3 つの基本的なシナリオで直感を確かめてみよう。

<!--en:ba1c76c2-->
1. **最適クリティックのロジットの読み取り：**対数損失の最適クリティックが、$D^\star(x_1) = 1.2$（点 $x_1$）および $D^\star(x_2) = -2.0$（点 $x_2$）を返すとする。各点における密度比 $p(x)/q(x)$ はいくらか。また、生成器はどちらの点の近傍でサンプル密度を減らすべきか。

<!--en:97370dc0-->
2. **3 つの失敗モードの診断：**訓練中に次の現象が観測されたとする。(a) 生成サンプルがデータから遠く離れ、生成器の勾配がほぼ 0 になっている。(b) 両方の損失は均衡付近にとどまるが、生成サンプルが周期的に周回している。(c) *「鳥」*を指定して生成させると、常にまったく同じ典型的な鳥の画像が返される。それぞれの症状には、本章のどの仕組みで対処できるか。

<!--en:557b98b6-->
**解答：**

<!--en:0ece88d0-->
1. $D^\star(x) = \log(p(x)/q(x))$ より、$p(x_1)/q(x_1) = e^{1.2} \approx \mathbf{3.32}$（実データの密度が $3.3\times$ 高い）および $p(x_2)/q(x_2) = e^{-2.0} \approx \mathbf{0.135}$（生成器の密度が $\approx 7.4\times$ 過剰）となる。したがって、生成器は確率質量を **$x_2$** から **$x_1$** へと移す必要がある。

<!--en:11e01bae-->
2. **(a)** 棄却されたサンプル（$D(x') \ll 0$）が単位勾配重みを保つよう、**非飽和生成器損失**（§{sec_basic_gan}）または滑らか・ペナルティ付きクリティック（§{sec_gan_objectives}）を用いる。**(b)** **ゼロ中心勾配ペナルティ $R_1 + R_2$**（§{sec_gan_convergence}）と指数移動平均（**EMA**、§{sec_dcgan}）を加えて回転するミニマックスダイナミクスを減衰させる。**(c)** 高い分類器アライメントに隠れたクラス内モード崩壊を検出するため、**クラスごとの $K$-NN 再現率**（§{sec_gan_conditional}）を測定する。

<!--en:a8dec3da-->
::: {.slide title="まとめ：主要記法と本章の総合"}

<!--en:c5a222f8-->
[重要ポイント]{.kicker}

<!--en:203ebc5e-->
- **2 サンプル領域と対数密度比：**実サンプル $x \sim p$ と生成サンプル $x' = G(z) \sim q$ が均衡混合分布 $m = \frac{1}{2}(p + q)$ をなし、最適クリティックの目標は $\lambda(x) = \log(p(x)/q(x))$、ベイズ事後確率は $\eta(x) = \sigma(\lambda(x))$ となる

<!--en:6a8cae73-->
- **サンプルごとの生成器重み：**$\nabla_\theta D(G_\theta(z))$ を $w_{\textrm{sat}} = \sigma(D(x'))$、$w_{\textrm{ns}} = \sigma(-D(x'))$、またはシフト不変な $1 - w_{\textrm{Rp}} = \mathbb{E}_p[\sigma(D(x) - D(x'))]$ でスケーリングする

<!--en:be63adec-->
- **均衡点での減衰：**ゼロ中心ペナルティ $R_1 + R_2 = \frac{\gamma}{2}(\mathbb{E}_p\|\nabla_x D\|^2 + \mathbb{E}_q\|\nabla_{x'} D\|^2)$ は $q = p$（$\nabla_x D^\star = \mathbf{0}$）で消失しつつ、ミニマックス軌道を安定化させる
