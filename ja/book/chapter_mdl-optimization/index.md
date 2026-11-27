<!--en:16ebb42e-->
# 最適化

<!--en:38ad263e-->
本書に登場するすべてのニューラルネットワークは、同じ基本的なループで訓練される。すなわち、損失の勾配を計算し、その逆方向にステップを進め、これを繰り返す。:numref:`chap_optimization` では、実際のネットワーク上でモメンタム、適応的ステップサイズ、学習率スケジュールがこのループにどう作用するかを実験的に確認したが、そこでは 3 つの疑問が未解決のまま残された。なぜ学習率には越えられない上限があるのか。その上限を十分に下回る安全なステップであっても、進行が極端に遅くなりうるのはなぜか。そして、なぜ勾配ノイズ（gradient noise）によって訓練が落ち着かなくなるのか。

<!--en:8b34c7c6-->
最適化理論は、これらそれぞれの問いを目的関数の幾何構造と結びつける。最も急な方向の曲率が安全な最大ステップサイズを定め、最も急な曲率と最も緩やかな曲率の比が反復回数を定め、凸性（convexity）が停留点（stationary point）が大域的に最適かどうかを決定する。本章全体を通して、細長い鉢状の関数 $\tfrac12(x_1^2 + 1000\,x_2^2)$ に繰り返し立ち返る。この関数はある方向がもう一方の方向より 1,000 倍急になっており、急な壁に沿って安全を保てるステップサイズでは、平坦な谷に沿った方向には反復点をわずかにしか動かせない。:numref:`chap_optimization` が実践と計測に焦点を当てたのに対し、本章では自己完結した証明と背後にある幾何構造を与える。

<!--en:209bc91b-->
本章は 3 部構成で展開する。まず厳密な勾配が得られる設定のもとで、:numref:`sec_mdl-gradient-based-optimization` ではステップサイズの上限と反復回数を証明し、モメンタムによって曲率比がその平方根に置き換わる仕組みを示す。一方、:numref:`sec_mdl-convexity` では停留点が大域的最小点となる条件を確立する。両者の間に位置する :numref:`sec_mdl-curvature-methods` では、曲率を積極的にモデル化するニュートン法、準ニュートン法（quasi-Newton method）、信頼領域法（trust-region method）を扱う。次にノイズを含む勾配に目を転じ、:numref:`sec_mdl-stochastic-gradient-methods` では固定ステップの SGD がステップサイズに比例するノイズフロアで停滞する理由を示し、:numref:`sec_mdl-adaptive-stochastic-methods` では座標ごとのスケーリングと AdamW をヘビーボールモメンタムと比較検証する。最後に、:numref:`sec_mdl-constrained-optimization-duality` と :numref:`sec_mdl-lagrangian-duality` ではノルムバジェット（:numref:`sec_weight_decay`）の背後にある KKT 条件、射影、近接ステップ、および双対性（duality）を展開し、:numref:`sec_mdl-numerical-stability-conditioning` では浮動小数点精度、桁落ち、および条件性（conditioning）を解析する（:numref:`sec_numerics`）。

<!--en:a64aa2fc-->
本章では、勾配とヘッセ行列（:numref:`sec_mdl-multivariable_calculus`）、および固有値分解（:numref:`sec_mdl-eigendecompositions`）に関する知識を前提とする。深層ネットワークの損失は大域的に滑らかでも凸でもないため、本章の定理は局所的なモデルとして機能するものであり、その適用限界は各節で明示する。なお、内点法（interior-point method）と作用素分割法（operator-splitting method）については :numref:`sec_mdl-optimization-further-reading` で概観する。

<!--en:5f1d3268-->
## 記法 {.unnumbered}

<!--en:b82469ac-->
本章全体を通して、:numref:`tab_mdl-opt-notation` にまとめた記法の慣例に従う。各節ではその節で頻繁に用いる記号を改めて示し、単一の節に固有の記法は初出箇所で導入する。

<!--en:a81b9720-->
:最適化の章で用いる記号。星印 $\star$ は最適点・最適値を、アスタリスク $*$ は凸共役を表す。行列は太字とする。

<!--en:204b3fa2-->
| 記号 | 意味 |

<!--en:49e186c6-->
| $f$、$\ell(\mathbf{w})$ | 目的関数、重みの関数としての訓練損失 |

<!--en:df138c29-->
| $\mathbf{x}^\star$、$f^\star$ | 最小点と最小値 |

<!--en:24179639-->
| $\nabla f$、$\nabla^2 f$、$\mathbf{g}_k$ | 勾配、ヘッセ行列、ステップ $k$ で用いる勾配（または確率的勾配） |

<!--en:c5aa368b-->
| $L$、$\mu$ | 滑らかさの定数と強凸性の定数（$\mu\mathbf{I} \preceq \nabla^2 f \preceq L\mathbf{I}$） |

<!--en:95d715eb-->
| $\kappa = L/\mu$、$\kappa(\mathbf{A})$ | 問題の条件数、行列の条件数 $\sigma_{\max}(\mathbf{A})/\sigma_{\min}(\mathbf{A})$ |

<!--en:ed4d4734-->
| $\lambda_i$、$\lambda_{\max}(\mathbf{A})$ | ヘッセ行列または二次関数の行列 $\mathbf{A}$ の固有値 |

<!--en:e8b62e6e-->
| $\eta$、$\beta$ | ステップサイズ（学習率）、モメンタム係数 |

<!--en:fea0fdff-->
| $k$、$t$ | 反復の添字、適応的手法におけるオプティマイザのステップ |

<!--en:87b65958-->
| $\epsilon$ | 目標精度、Adam において除算を有限に保つ定数 |

<!--en:d63c6b79-->
| $n$、$d$、$\mathcal{B}$、$b$ | データ例の数、変数の数、ミニバッチとそのサイズ |

<!--en:5aaef151-->
| $\boldsymbol{\Sigma}$ | 各データ例の勾配の共分散 |

<!--en:78ad6c38-->
| $\mathbf{m}_t$、$\mathbf{v}_t$、$\beta_1$、$\beta_2$ | Adam の一次・二次モーメント推定値、およびそれらの減衰率 |

<!--en:d73d4dd2-->
| $\mathbf{P}$ | 前処理行列（更新ステップは $-\eta\,\mathbf{P}^{-1}\mathbf{g}$） |

<!--en:4f3048e3-->
| $g_i$、$h_j$ | 不等式制約 $g_i(\mathbf{x}) \le 0$、等式制約 $h_j(\mathbf{x}) = 0$ |

<!--en:ca9eefa8-->
| $\lambda$、$\nu$ | 不等式制約と等式制約の乗数（$\lambda$ は減衰項 $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ の重みも表す） |

<!--en:ff8754d1-->
| $\mathcal{L}$、$q$ | ラグランジアン、双対関数 |

<!--en:4cce05c1-->
| $p^\star$、$d^\star$ | 主問題と双対問題の最適値 |

<!--en:0578cae4-->
| $f^*$ | $f$ の凸共役 |

<!--en:a986ab6b-->
| $\Pi_C$、$\mathrm{prox}_\psi$ | 集合 $C$ へのユークリッド射影、$\psi$ の近接作用素 |

<!--en:d6b729a4-->
| $u$、$\varepsilon_{\mathrm{mach}}$ | 浮動小数点形式の丸めの単位と計算機イプシロン（$u = \varepsilon_{\mathrm{mach}}/2$） |

<!--en:4901a14e-->
深層学習のための数学：最適化<br>**滑らかさ・条件数・モメンタム・曲率・凸性・確率的ノイズ・前処理・双対性・数値計算**

<!--en:6c116032-->
::: {.slide title="すべての訓練ループに残された 3 つの問い"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:5e9d67b7-->
- **なぜステップサイズに越えられない上限があるのか：**最も急な方向で $2/L$ を超えて進むと、谷の壁を飛び越えて損失が増加する

<!--en:127c121a-->
- **なぜ安全なステップは這うように遅いのか：**曲率に $\kappa = L/\mu$ 倍の差があると、最も急な壁が $\eta$ の上限を抑え、最も平坦な方向が停滞する

<!--en:35f258cb-->
- **なぜミニバッチノイズは $f^\star$ の手前で止まるのか：**定数ステップの SGD は、損失 $\propto \eta\sigma^2/b$ の定常なノイズ球に落ち着く

<!--en:ee854171-->
::: {.slide title="共通ベンチマーク：悪条件な谷"}

<!--en:311edefe-->
[章を貫く軸]{.kicker}

<!--en:1bc16447-->
すべての定理を $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$（$\mu = 1, L = 1000, \kappa = 1000$）で検証する。

<!--en:278d6061-->
- **一次手法のスケーリング：**固定ステップの GD は $6{,}908$ ステップ（$O(\kappa)$）を要するが、ヘビーボールモメンタムとネステロフのモメンタムは $315$〜$508$ ステップ（$O(\sqrt{\kappa})$）で済む

<!--en:c7a78c17-->
- **前処理と回転：**厳密なニュートン法は $1$ ステップで完了する。軸に沿った問題では Adam は $f < 10^{-8}$ に $22$〜$48$ ステップで到達する（GD では $6{,}160$ ステップ）が、$45^\circ$ の回転で対角スケーリングは破綻する

<!--en:09955639-->
- **確率的ノイズフロア：**定数ステップの SGD（$\eta = 10^{-3}, \sigma^2 = 1$）は損失 $7.5 \times 10^{-4}$ で停滞する

<!--en:df538ea7-->
::: {.slide title="第 §{num:chap_mdl-optimization} 章の構成：3 つの主要部"}

<!--en:99e0030a-->
[章のロードマップ]{.kicker}

<!--en:e26d048f-->
- **第 1 部・厳密な勾配と幾何構造（§{sec_mdl-gradient-based-optimization}〜§{num:sec_mdl-convexity}）：**$L$ 滑らかさの降下補題、$\kappa$ 対 $\sqrt{\kappa}$ のモメンタム則、ニュートン法・BFGS・信頼領域、凸性による保証

<!--en:f19071c4-->
- **第 2 部・確率的手法と適応的手法（§{sec_mdl-stochastic-gradient-methods}〜§{num:sec_mdl-adaptive-stochastic-methods}）：**$1/b$ 分散則、定常ノイズフロア、Robbins–Monro スケジュール、AdamW、行列前処理（K-FAC、Muon）

<!--en:d86b44fa-->
- **第 3 部・制約、双対性、数値計算（§{sec_mdl-constrained-optimization-duality}〜§{num:sec_mdl-numerical-stability-conditioning}）：**KKT 条件、射影、近接 ISTA、Fenchel 双対性、浮動小数点の条件性

<!--en:8a49cc84-->
::: {.slide title="主要な数学量と記法"}

<!--en:edb10563-->
[記法リファレンス]{.kicker}

<!--en:77ffc341-->
- **曲率と条件性：**
  - $L$、$\mu$：滑らかさの上限と強凸性の下限（$\mu\mathbf{I} \preceq \nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$）
  - $\kappa = L/\mu$：問題の条件数。行列 $\mathbf{A}$ に対しては $\kappa(\mathbf{A}) = \sigma_{\max}(\mathbf{A})/\sigma_{\min}(\mathbf{A})$
  - $\mathbf{P}$：ステップを $-\eta \mathbf{P}^{-1}\mathbf{g}_k$ に再スケーリングする前処理行列

<!--en:931f118c-->
- **確率的・適応的状態：**
  - $\boldsymbol{\Sigma}$、$b$：各データ例の勾配の共分散とミニバッチサイズ（ノイズ電力 $\operatorname{tr}\boldsymbol{\Sigma}/b$）
  - $\mathbf{m}_t, \mathbf{v}_t$：減衰係数 $\beta_1, \beta_2$ による一次・二次モーメントの移動平均

<!--en:982b93c9-->
- **制約、双対性、作用素：**
  - $g_i(\mathbf{x}) \le 0$、$h_j(\mathbf{x}) = 0$：乗数 $\lambda_i \ge 0$ と $\nu_j$ を伴う不等式制約と等式制約
  - $\mathcal{L}$、$q$、$p^\star$、$d^\star$：ラグランジアン、双対関数、主最適値、双対最適値（$d^\star \le p^\star$）
  - $f^*$、$\Pi_C$、$\mathrm{prox}_\psi$：凸共役、$C$ へのユークリッド射影、近接作用素

<!--en:bbf53d98-->
- **浮動小数点精度：**
  - $\varepsilon_{\mathrm{mach}} = 2^{-p}$、$u = 2^{-p-1}$：仮数部 $p$ ビットに対する計算機イプシロンと丸めの単位

<!--en:15d021a2-->
::: {.slide title="本章の読み進め方"}

<!--en:123f19cf-->
[学習ガイド]{.kicker}

<!--en:b0bf7ff5-->
- **必須の基礎（§{sec_mdl-gradient-based-optimization}、§{sec_mdl-convexity}）：**まずはここから読み進め、$2/L$ のステップ上限、$\sqrt{\kappa}$ のモメンタム高速化、接線による下界、イェンセンの不等式、大域的収束率を押さえる

<!--en:cd160025-->
- **深層学習の主力手法（§{sec_mdl-stochastic-gradient-methods}、§{sec_mdl-adaptive-stochastic-methods}）：**次に読み、SGD のノイズフロア、コサイン・WSD スケジュール、Adam のスケール不変性、重み減衰を分離した AdamW、Muon を学ぶ

<!--en:a87c8e25-->
- **必要に応じた深掘り（§{sec_mdl-curvature-methods}、§{sec_mdl-constrained-optimization-duality}〜§{num:sec_mdl-numerical-stability-conditioning}）：**L-BFGS・信頼領域法、KKT・射影・ISTA、Fenchel 双対性、`bf16` の数値安定性について参照する
