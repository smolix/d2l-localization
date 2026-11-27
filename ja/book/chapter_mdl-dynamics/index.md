<!--en:25a68c1e-->
# ダイナミクス：微分方程式と生成フロー

<!--en:5b5ca6dc-->
残差ネットワーク（residual network）は層ごとに入力へ小さな更新を加え、勾配降下法は重みをステップごとに下り方向へ少しずつ動かし、拡散モデル（diffusion model）は学習されたベクトル場（vector field）に沿ってノイズを画像へと変換する。これら 3 つはいずれも、状態が変化する速さを定める規則である微分方程式（differential equation）を離散化したものである。ここからただちに 2 つの実践的な問いが生じる。サンプラはどれだけ少ないステップ数で実行できるのか。また、画像の例だけから速度やスコアの方向をどのように学習できるのだろうか。

<!--en:557aa4ba-->
訓練画像にノイズを徐々に加えていくことで、2 つ目の問いに逆方向から答えられる。ノイズを帯びた個々の粒子は予測不能にさまようが、粒子群全体の密度は決定論的に時間発展し、その時間発展を逆転させればガウスノイズをデータへと戻せる。これまでの章では、この仕組みまでは踏み込んでいない。誤差逆伝播法（backpropagation）は ODE の解ではなく固定された計算グラフを微分するものであり、静的な確率では移動する密度を追跡できず、$\sqrt{\Delta t}$ オーダーのブラウン増分は通常の連鎖律（chain rule）を破綻させるからである。

<!--en:67dc6f1b-->
本章は 3 つの段階で展開する。第 1 に、決定論的な ODE によって単一の状態を厳密かつ数値的に動かす。ここでは、最も速く減衰するモードが勾配降下法およびすべての陽的ルンゲ・クッタ法のソルバのステップサイズに上限を課す一方（:numref:`sec_mdl-euler-runge-kutta`）、より真っ直ぐな輸送経路によって少数ステップでの生成が可能になる（:numref:`sec_mdl-rectified-flow`）。その過程で、駆動される線形 ODE から状態空間モデル（state space model）の漸化式が導かれ（:numref:`subsec_mdl-ode-driven-linear`、:numref:`sec_ssm`）、深い残差ネットワーク（:numref:`sec_resnet`）はニューラル ODE および連続正規化フロー（continuous normalizing flow）へとつながる（:numref:`sec_mdl-neural-odes-cnf`）。第 2 に、確率解析（stochastic calculus）によって連鎖律を修復し（:numref:`sec_mdl-sdes`）、フォッカー・プランク方程式を通じて粒子群の密度を追跡することで、時間反転に必要なのは*スコア* $\nabla \log p_t$ だけであることを示す（:numref:`sec_mdl-fokker-planck-probability-flow`）。第 3 に、ノイズを加えた対や補間された対に対する最小二乗回帰（least-squares regression）により、場の正解ラベルなしにスコアや周辺速度（marginal velocity）を直接学習する（:numref:`sec_mdl-score-matching-diffusion-flow`、:numref:`sec_mdl-flow-matching-transport`）。

<!--en:518ab661-->
:numref:`chap_diffusion` へ直行したい読者は、最初の一読では連続随伴法（continuous adjoint method、:numref:`sec_mdl-adjoint-method`）と最適輸送（optimal transport、:numref:`sec_mdl-ot-connection`）を読み飛ばしてよい。本章のほとんどの実験では、離散化誤差を明確に切り分けられるよう、閉形式のベクトル場に対してサンプラを検証する。なお、固有値分解（:numref:`sec_mdl-eigendecompositions`）、ヤコビ行列（:numref:`subsec_mdl-jacobian`）、およびガウス分布の条件付けを前提とし、測度論（measure theory）は仮定しない。

<!--en:5f1d3268-->
## 記法 {.unnumbered}

<!--en:a2267486-->
:numref:`tab_mdl-dyn-notation` に、複数の節にわたって繰り返し現れる記号をまとめる。各節ではその節で必要な記号を改めて示し、一度しか使わない文字はその場で定義する。表中の文字が 2 つの意味を持つ場合は、それぞれの意味がどの箇所で成り立つかを該当する行に記している。

<!--en:78248bb3-->
:ダイナミクスの章で用いる記法。

<!--en:969fc472-->
| 記号 | 意味 | 初出箇所 |

<!--en:812595bb-->
| $\mathbf x(t)\in\mathbb R^d$、$\dot{\mathbf x}$ | 状態、その時間微分 | :numref:`sec_mdl-odes-solvers` |

<!--en:b5c29849-->
| $\mathbf x_0$ | $t=0$ における状態（軌跡の始点、ノイズ付加過程ではデータ点） | :numref:`sec_mdl-odes-solvers` |

<!--en:ad7aa4f0-->
| $\mathbf f(\mathbf x,t)$ | ODE のベクトル場、SDE のドリフト | :numref:`sec_mdl-odes-solvers`、:numref:`sec_mdl-sdes` |

<!--en:ffce713d-->
| $\Phi_t$、$\Phi_{s\to t}$ | フロー写像 $\Phi_t(\mathbf x_0)=\mathbf x(t)$、時刻 $s$ の状態から時刻 $t$ の状態への写像 | :numref:`sec_mdl-odes-solvers` |

<!--en:5ba1410a-->
| $L$ | $\mathbf f$ のリプシッツ定数 | :numref:`sec_mdl-odes-solvers` |

<!--en:eaf73320-->
| $T$ | 終端時刻（ホライズン） | :numref:`sec_mdl-odes-solvers` |

<!--en:1d00be30-->
| $\mathbf A$、$\lambda_j=a_j+\mathrm i\,b_j$、$\mathbf V$、$\boldsymbol\Lambda$ | 線形場 $\dot{\mathbf x}=\mathbf A\mathbf x$ の行列、その固有値、固有ベクトルを並べた行列、固有値の対角行列 | :numref:`sec_mdl-odes-solvers` |

<!--en:4fe1fce3-->
| $\lambda$ | $\mathbf A$ またはヤコビ行列の固有値。逆時間サンプラでは注入するノイズ振幅の定数倍 $\lambda g$ | :numref:`sec_mdl-odes-solvers`、:numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:6d5a6d1a-->
| $\mathbf K$ | 回転を表す行列 $\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$（$\dot{\mathbf x}=\mathbf K\mathbf x$） | :numref:`sec_mdl-odes-solvers` |

<!--en:0435ba08-->
| $\mathbf J$ | $\mathbf f$ のヤコビ行列 | :numref:`sec_mdl-odes-solvers` |

<!--en:010cff50-->
| $\mathbf x^\star$ | ベクトル場の不動点 $\mathbf f(\mathbf x^\star)=\mathbf 0$（勾配フローでは損失の臨界点） | :numref:`sec_mdl-odes-solvers` |

<!--en:f0264de4-->
| $\mathbf u(t)$、$\mathbf B$ | 入力とその行列（$\dot{\mathbf x}=\mathbf A\mathbf x+\mathbf B\mathbf u(t)$） | :numref:`sec_mdl-odes-solvers` |

<!--en:39bf118f-->
| $h$、$t_n=nh$、$\mathbf x_n$ | ステップサイズ（:numref:`sec_mdl-sdes` と :numref:`sec_mdl-fokker-planck-probability-flow` では $\Delta t$ と表記）、時間グリッド、$t_n$ における状態（厳密解またはソルバの反復点） | :numref:`sec_mdl-odes-solvers`、:numref:`sec_mdl-euler-runge-kutta` |

<!--en:1ec22aa7-->
| $\bar{\mathbf A}$、$\bar{\mathbf B}$ | 入力を区間幅 $h$ にわたり一定に保った厳密ステップ $\mathbf x_{n+1}=\bar{\mathbf A}\mathbf x_n+\bar{\mathbf B}\mathbf u_n$ | :numref:`sec_mdl-odes-solvers` |

<!--en:13ffa0e2-->
| $R(z)$、$z=h\lambda$ | ソルバの安定性関数 | :numref:`sec_mdl-euler-runge-kutta` |

<!--en:faee3d21-->
| $\varepsilon_n$ | 大域誤差 $\lVert\mathbf x_n-\mathbf x(t_n)\rVert$ | :numref:`sec_mdl-euler-runge-kutta` |

<!--en:abda2338-->
| $\mathcal L$、$\eta$、$\mathbf H$ | 損失、学習率、ヘッセ行列 | :numref:`sec_mdl-odes-solvers`、:numref:`sec_mdl-euler-runge-kutta` |

<!--en:303f859c-->
| $N$ | 終端時刻までのステップ数。ソルバのステップ $n=1,\ldots,N$（$Nh=T$、最初の 2 節ではこの総数にも $n$ を用いる）。また :numref:`sec_mdl-score-matching-diffusion-flow` では DDPM のノイズレベル $t=1,\ldots,N$ | :numref:`sec_mdl-neural-odes-cnf` |

<!--en:cd91af6e-->
| $\boldsymbol\theta$ | ネットワークのパラメータ | :numref:`sec_mdl-neural-odes-cnf` |

<!--en:80068c08-->
| $\mathbf a(t)$ | 随伴状態 | :numref:`sec_mdl-neural-odes-cnf` |

<!--en:7240586f-->
| $p_t$ | 時刻 $t$ における状態の密度 | :numref:`sec_mdl-neural-odes-cnf` |

<!--en:4bc87d8f-->
| $\boldsymbol\epsilon$ | :numref:`sec_mdl-neural-odes-cnf` ではトレース推定量のランダムプローブ。:numref:`sec_mdl-sdes` 以降では $\mathbf x_t=\alpha_t\mathbf x_0+\sigma_t\boldsymbol\epsilon$ における標準ガウスノイズ（:numref:`sec_mdl-flow-matching-transport` ではノイズ側端点 $\mathbf x_1$ とも表記） | :numref:`sec_mdl-neural-odes-cnf`、:numref:`sec_mdl-sdes` |

<!--en:5d65b758-->
| $\mathbf X_t$（1 次元では $X_t$） | 時刻 $t$ における確率過程の状態を表す確率変数（$\mathbf x$ はそのとる値） | :numref:`sec_mdl-sdes` |

<!--en:67e1a8a6-->
| $\mathbf W_t$ | ブラウン運動 | :numref:`sec_mdl-sdes` |

<!--en:38783c60-->
| $g$、$g(t)$、$\mathbf G$ | 定数または時間変化する SDE のノイズ振幅（拡散係数）、ノイズ行列 | :numref:`sec_mdl-sdes` |

<!--en:7745aba9-->
| $\theta$ | OU 過程 $dX=-\theta X\,dt+g\,dW$ のレート。:numref:`sec_mdl-neural-odes-cnf` のスカラー例では $\dot x=\theta x$ のパラメータ | :numref:`sec_mdl-neural-odes-cnf`、:numref:`sec_mdl-sdes` |

<!--en:b708859e-->
| $\alpha_t$、$\sigma_t$ | $\mathbf X_t$ の、$\mathbf X_0$ を条件とする平均係数と標準偏差（$\sigma_t$ はノイズレベル） | :numref:`sec_mdl-sdes` |

<!--en:9116a1ed-->
| $\phi$ | 状態の滑らかな関数。伊藤の補題における関数、およびフォッカー・プランク方程式の背後にあるテスト関数（他の箇所では角度） | :numref:`sec_mdl-sdes` |

<!--en:fcaf06cd-->
| $\psi$ | ベクトル解析の準備における滑らかなスカラー場 | :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:d47a0041-->
| $\mathbf s_t=\nabla\log p_t$、$\mathbf s_{\boldsymbol\theta}$ | スコア、学習されたスコア | :numref:`sec_mdl-fokker-planck-probability-flow`、:numref:`sec_mdl-score-matching-diffusion-flow` |

<!--en:0d1ec1c3-->
| $\mathbf u_t$、$\mathbf u_{\boldsymbol\theta}$ | $p_t$ を輸送する速度（確率フロー速度、入力 $\mathbf u(t)$ とは無関係）、学習された速度 | :numref:`sec_mdl-fokker-planck-probability-flow`、:numref:`sec_mdl-flow-matching-transport` |

<!--en:89139d0b-->
| $\mathbf j$ | 確率流 | :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:fff21276-->
| $\bar t=T-t$ | 逆向きの時刻 | :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:2518d8d3-->
| $w_k$、$r_k(\mathbf x)$、$v_k$ | 混合重み、負担率、各成分の分散 | :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:d7526930-->
| $\beta(t)$、$\beta_t$、$\bar\alpha_t$ | 分散保持 SDE のレート。DDPM のノイズスケジュール（およそ $\beta(t)$ とステップ幅の積であり、:numref:`sec_mdl-sdes` では $\beta_k$ と表記）とその累積積 $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$（本章の $\alpha_t^2$ に相当） | :numref:`sec_mdl-sdes`、:numref:`sec_mdl-score-matching-diffusion-flow` |

<!--en:c6687a99-->
| $\bar{\mathbf x}_t$、$\bar\sigma_t$ | DDIM 座標（リスケールされた点 $\mathbf x_t/\sqrt{\bar\alpha_t}$ とそのノイズ対信号比 $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$） | :numref:`sec_mdl-score-matching-diffusion-flow` |

<!--en:0175a508-->
| $w(t)$、$\gamma$ | 損失におけるノイズレベル $t$ の重み、ガイダンス強度 | :numref:`sec_mdl-score-matching-diffusion-flow` |

<!--en:32b140c1-->
| $\pi$ | カップリング（データとノイズの組の同時分布、ペアリングとも呼ぶ） | :numref:`sec_mdl-flow-matching-transport` |

<!--en:cc94fe75-->
| $\mathrm i$、$\mathbf I$ | 虚数単位（立体）、単位行列 | 章全体 |

<!--en:109c4cb5-->
ダイナミクス：微分方程式、確率フロー、輸送<br>**ODE とソルバ・ニューラル ODE・伊藤 SDE・フォッカー・プランク PDE・スコアマッチング・フローマッチング**

<!--en:525812b2-->
::: {.slide title="なぜ深層学習に連続時間ダイナミクスが必要なのか"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:739b4546-->
- **オイラー法のステップとしてのアーキテクチャとオプティマイザ：**ResNet ブロック $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathbf{f}(\mathbf{x}_l)$（$h=1$）と勾配降下法 $\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\nabla\mathcal{L}$（$h=\eta$）はいずれも ODE の前進オイラー法ステップであり、$\eta < 2/\lambda_{\max}(\mathbf{H})$ の上限を持つ

<!--en:d38a9c2d-->
- **ノイズの逆転による生成サンプリング：**ブラウン運動の揺らぎ（$\sim \sqrt{dt}$）を加えると、フォッカー・プランク PDE に従って $p_t(\mathbf{x})$ が決定論的に平滑化され、これを逆転させることで $\mathcal{N}(\mathbf{0}, \mathbf{I})$ から $p_{\text{data}}$ へと戻せる

<!--en:52e627fa-->
::: {.slide title="第 §{num:chap_mdl-dynamics} 章における 3 段階の数学的ロードマップ"}

<!--en:c8dfc67f-->
[本章の構成]{.kicker}

<!--en:6cc56c3f-->
- **ステージ 1：決定論的 ODE とフロー（§{sec_mdl-odes-solvers}〜§{num:sec_mdl-neural-odes-cnf}）：**ピカール・リンデレーフの一意性、行列指数関数 $e^{\mathbf{A}t}$、ルンゲ・クッタ法の安定性 $R(h\lambda)$、連続随伴 ODE、リウヴィルのトレース則 $\frac{d}{dt}\log p_t = -\operatorname{tr}\mathbf{J}$

<!--en:fd6d1ff6-->
- **ステージ 2：確率解析と密度フロー（§{sec_mdl-sdes}〜§{num:sec_mdl-fokker-planck-probability-flow}）：**二次変分 $(dW)^2 = dt$、伊藤の補題、OU ガウスカーネル、フォッカー・プランク PDE、Anderson の逆時間 SDE、確率フロー ODE

<!--en:6aa66c40-->
- **ステージ 3：生成ベクトル場の学習（§{sec_mdl-score-matching-diffusion-flow}〜§{num:sec_mdl-flow-matching-transport}）：**Vincent のノイズ除去スコアマッチング、Tweedie の公式、DDPM・DDIM、条件付きフローマッチング、Reflow

<!--en:23ef19e0-->
::: {.slide title="本章の軸となる具体的な通しベンチマーク"}

<!--en:05adf916-->
[章を貫く題材]{.kicker}

<!--en:eee4fa07-->
- **線形 ODE と硬い ODE：**平面の渦状沈点（$\lambda = -0.5 \pm \mathrm{i}$）と硬い 2 モード減衰（$\mathrm{diag}(-50, -1)$）により、$e^{\mathbf{A}t}$、RK4 の次数、随伴法、陽解法ステップ幅の上限を検証

<!--en:0f0792e5-->
- **確率的・多峰性ベンチマーク：**オルンシュタイン・ウーレンベック SDE（$dX = -\theta X\,dt + g\,dW$）と二峰性混合ガウス分布により、厳密な遷移カーネル、スコア、逆時間 SDE、DDIM のストライドを検証

<!--en:83aba0c8-->
- **2D 輸送ベンチマーク：**Two-moons と $\mathcal{N}(\mathbf{0}, \mathbf{I})$ の対により、交差する線分、CFM の下限 $C$、Reflow による直線化を検証

<!--en:3cbdd1b1-->
::: {.slide title="統一的な方程式：粒子、密度、時間反転"}

<!--en:39418420-->
[基本対応表]{.kicker}

<!--en:9e3e68f4-->
- **順時間・逆時間 SDE：**$d\mathbf{X} = \mathbf{f}\,dt + g\,d\mathbf{W}$ の時間反転は $d\mathbf{X} = [\mathbf{f} - g^2\nabla\log p_t]\,dt + g\,d\bar{\mathbf{W}}$ で与えられる

<!--en:30d78ab6-->
- **決定論的な確率フロー ODE：**$\dot{\mathbf{x}} = \mathbf{u}_t(\mathbf{x}) = \mathbf{f}(\mathbf{x}, t) - \tfrac12 g(t)^2\nabla\log p_t(\mathbf{x})$ はすべての周辺分布 $p_t(\mathbf{x})$ を共有する

<!--en:ea5f1305-->
- **鍵となる恒等式：**$\tfrac12 g^2\Delta p_t = \nabla\cdot(p_t \cdot \tfrac12 g^2\nabla\log p_t)$ と書き換えることで、ラプラシアン拡散がスコア場に沿った決定論的輸送に変わる

<!--en:09587a8c-->
::: {.slide title="第 §{num:chap_mdl-dynamics} 章の前提知識と学習パス"}

<!--en:123f19cf-->
[学習ガイド]{.kicker}

<!--en:dcabe51c-->
- **数学的前提知識：**固有値分解とスペクトルノルム（§{chap_mdl-linear-algebra}）、多変数のヤコビ行列・発散・部分積分（§{chap_mdl-calculus}）、およびガウス分布の条件付き期待値（§{chap_mdl-probability-statistics}）を前提とし、測度論は不要

<!--en:f4de0fea-->
- **最適化・SSM パス（§{sec_mdl-odes-solvers}〜§{num:sec_mdl-euler-runge-kutta}）：**状態空間モデル（Mamba/S4）の厳密なゼロ次ホールド離散化、モメンタムにおける非正規な過渡的増大、勾配降下法における数値的な硬さを扱う

<!--en:07b6eb34-->
- **拡散・フローモデルへの最短経路：**フローとルンゲ・クッタ法のソルバは **§{sec_mdl-odes-solvers}〜§{num:sec_mdl-euler-runge-kutta}**、伊藤の補題とオルンシュタイン・ウーレンベックのカーネルは **§{sec_mdl-sdes}**、逆時間 SDE と確率フロー ODE は **§{sec_mdl-fokker-planck-probability-flow}**、スコアマッチングとフローマッチングは **§{sec_mdl-score-matching-diffusion-flow}〜§{num:sec_mdl-flow-matching-transport}** を読む

<!--en:6fc2ec71-->
- **発展トピック（任意）：**画像生成を目指す最初の一読では、連続随伴法（**§{sec_mdl-neural-odes-cnf}**）とワッサーシュタイン最適輸送理論（**§{sec_mdl-flow-matching-transport}**）は読み飛ばしてよい
