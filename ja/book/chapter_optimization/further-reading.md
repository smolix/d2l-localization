<!--en:7902ddfd-->
# 関連文献

<!--en:c6951c42-->
本節では、本章の手法を発展させた基礎的な教科書、サーベイ論文、および近年の研究をまとめる。本文中で引用した収束率（convergence rate）の数学的証明は :numref:`chap_mdl-optimization` で与えられており（理論に関する追加の文献は :numref:`sec_mdl-optimization-further-reading` を参照）、自動ハイパーパラメータ最適化は :numref:`chap_hyperopt` で、複数デバイスによる分散訓練システムは :numref:`chap_performance` で扱っている。

<!--en:7e6a74bc-->
## 基礎的な教科書、サーベイ、プレイブック

<!--en:af678677-->
* **凸最適化と数値最適化：**:citet:`Boyd.Vandenberghe.2004`（『[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)』）は、条件付け（conditioning）、勾配降下法、ニュートン法（:numref:`sec_gd`）、および一般のノルムのもとでの最急降下法（steepest descent、:numref:`sec_muon`）に関する標準的な参考書である。:citet:`Nocedal.Wright.2006`（『Numerical Optimization』）は、ニュートン法、準ニュートン法、直線探索、信頼領域法を包括的に扱っている。

<!--en:06a2b213-->
* **確率的最適化：**:citet:`Bottou.Curtis.Nocedal.2018`（「[Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)」）は、確率的勾配のノイズ球、ステップサイズの減衰（:numref:`sec_sgd`）、および分散低減（variance reduction）を概説している。

<!--en:982305cd-->
* **対話型ガイドと実務者向けプレイブック：**Gabriel Goh による「[Why Momentum Really Works](https://distill.pub/2017/momentum/)」（Distill、2017 年）は、臨界減衰と $\sqrt{\kappa}$ 加速則（:numref:`sec_momentum`）の対話的な可視化を提供している。Jeremy Bernstein による「[Deriving Muon](https://jeremybernste.in/writing/deriving-muon)」および :citet:`Bernstein.Newhouse.2024` は、ノルム球にわたる最急降下法を統一的に導出している（:numref:`sec_muon`）。EleutherAI の「[Practitioner's Guide to Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/)」は $\mu\text{P}$ の座標チェック（:numref:`sec_scaling`）を詳述しており、Google Research の『[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)』:cite:`Godbole.Dahl.Gilmer.ea.2023` は、:numref:`sec_test_models` および :numref:`sec_practice` 全体で用いられる科学的（scientific）・局外（nuisance）・固定（fixed）ハイパーパラメータの区別を定式化している。

<!--en:efa04b17-->
## 損失曲面

<!--en:d26423de-->
### 高次元損失ランドスケープの可視化とモード連結性

<!--en:522ac3f4-->
ニューラルネットワークの損失曲面の低次元断面は、正の斉次性によるスケール不変性（$c\,\mathbf{W}_1$ と $c^{-1}\mathbf{W}_2$ は同じ ReLU ネットワークを計算する）を取り除くために摂動方向をフィルタごとに正規化しなければ、誤解を招くものになりかねない。フィルタ正規化したスライスを用いることで、:citet:`Li.Xu.Taylor.ea.2018` は残差接続（:numref:`sec_resnet`）がぎざぎざした非凸な曲面を滑らかな盆地へと変えることを示した。

<!--en:2382e906-->
独立に訓練された 2 つのネットワーク間を線形補間すると高い損失の障壁を横切ることになるが、:citet:`Garipov.Izmailov.Podoprikhin.ea.2018` および :citet:`Draxler.Veschgini.Salmhofer.ea.2018` は**モード連結性**（mode connectivity）を発見した。すなわち、1 回だけ折れ曲がる単純な区分線形曲線によって、訓練損失・テスト損失ともに一様に低い経路に沿って独立な極小点どうしを結ぶことができる。さらに :citet:`Entezari.Sedghi.Saukh.ea.2022` と :citet:`Ainsworth.Hayase.Srinivasa.2023` は、置換対称性（:numref:`sec_gd`）を用いて隠れユニットを揃えると、幅の広いネットワーク間の線形の障壁がほぼ消失することを実証した。

<!--en:47554bb1-->
### 平坦な極小点と Sharpness-Aware Minimization (SAM)

<!--en:d8f23678-->
より平坦な極小点ほど汎化性能が高いという仮説に基づき、**Sharpness-Aware Minimization (SAM)** :cite:`Foret.Kleiner.Mobahi.ea.2021` は、降下方向の勾配を評価する前に勾配上昇方向の摂動ステップを踏むことで、現在の重みの周囲の $\ell_2$ 球内における最悪ケースの損失を最小化する（1 回の更新につき $2\times$ の勾配コストがかかる）。ただし、そのままのヘッセ行列の鋭さ（sharpness）は層ごとの再パラメータ化に対して不変ではないことに注意が必要である :cite:`Dinh.Pascanu.Bengio.ea.2017`。

<!--en:afb51542-->
## 古典的な 2 次手法と分散低減手法

<!--en:31c92eb8-->
### 準ニュートン法、直線探索、信頼領域法

<!--en:e5c00a49-->
**準ニュートン法**（quasi-Newton method）は、ヘッセ行列の逆行列 $\mathbf{H}^{-1}$ を、1 次の勾配のみを用いた連続するセカント対 $(\mathbf{x}_{t+1} - \mathbf{x}_t, \mathbf{g}_{t+1} - \mathbf{g}_t)$ から近似する。記憶制限 BFGS（**L-BFGS**、:citet:`Liu.Nocedal.1989`）は直近 $m$ 個の対を $O(md)$ のメモリに保持し、決定論的で滑らかな問題に対する定番の手法であるが（:numref:`subsec_mdl-quasi-newton`）、ミニバッチ勾配の差分に対しては敏感である。**直線探索**（line search、Armijo の十分減少条件の判定により $\eta_t$ を動的に選ぶ手法）および**信頼領域法**（trust-region method、:numref:`subsec_mdl-trust-region`、Muon の信頼領域解析については :citet:`Kovalev.2025` を参照）も同様に、決定論的最適化において威力を発揮する。

<!--en:332f8348-->
### ヘッセ行列フリー法と鞍点フリーニュートン法

<!--en:77d9f5b1-->
$d \times d$ のヘッセ行列を陽に作らずに 2 次手法を深層ネットワークへ適用するため、**ヘッセ行列フリー最適化**（Hessian-free optimization）:cite:`Martens.2010` は、半正定値であるガウス・ニュートン行列上のヘッセ行列ベクトル積（:numref:`subsec_mdl-hvp`）を用いた共役勾配法によって $\mathbf{H}\mathbf{d} = -\mathbf{g}$ を近似的に解く。一方、**鞍点フリーニュートン法**（Saddle-Free Newton）:cite:`Dauphin.Pascanu.Gulcehre.ea.2014` は、負の曲率方向が反復点を引き寄せるのではなく反発させるよう、低次元のクリロフ部分空間上で $|\mathbf{H}|^{-1}$ による前処理を施す。

<!--en:c6c4d065-->
### 確率的収束率と分散低減

<!--en:13a971b4-->
一般の凸目的関数に対して、ミニバッチ SGD はミニマックス最適な $O(1/\sqrt{T})$ の収束率を達成し :cite:`Nemirovski.Yudin.1983`（:numref:`subsec_mdl-fixed-budget-rates`）、非凸で滑らかな目的関数に対しては $\min_{t \le T} \mathbb{E}[\|\nabla f(\mathbf{x}_t)\|^2] = O(1/\sqrt{T})$ を保証する :cite:`Ghadimi.Lan.2013`（:numref:`subsec_mdl-nonconvex-sgd`）。Polyak–Ruppert の反復平均化 :cite:`Polyak.Juditsky.1992` は、最適な漸近共分散を達成する（:numref:`subsec_practice-weight-averaging`）。有限のデータセット上では、**SVRG** :cite:`Johnson.Zhang.2013`（:numref:`sec_mdl-variance-reduction`）がデータセット全体にわたる周期的な制御変量（control variate）によって定常的なノイズ球を解消するが、深層学習においてはデータ拡張や急速に変化する特徴がその有効性を制限する。

<!--en:50312ace-->
## 行列オプティマイザと構造的オプティマイザ

<!--en:97ac58c1-->
* **自然勾配、K-FAC、SOAP：**出力分布の空間でステップ長を測る**自然勾配**（natural gradient）:cite:`Amari.1998` は、フィッシャー情報行列の逆行列によって更新を前処理する。**K-FAC** :cite:`Martens.Grosse.2015` は、層ごとのフィッシャー行列ブロックを入力の共分散と出力勾配の共分散のクロネッカー積として近似する（:numref:`subsec_mdl-preconditioning-ladder`）。一方、**SOAP** :cite:`Vyas.Morwani.Zhao.ea.2024` は、Shampoo の左右の前処理行列 :cite:`Gupta.Koren.Singer.2018` がなす固有基底の上で Adam を実行する。

<!--en:5efd2c20-->
* **Muon の理解とスケーリング：**:citet:`Wang.Zhang.Li.ea.2025` は、AdamW に対する Muon の高速化の要因が、裾の特徴（tail feature）における連想メモリ射影（アテンションの値・出力行列および MLP 行列）にあることを突き止めた。これは希少トークンにおける SGD に対する Adam の優位性 :cite:`Kunstner.Chen.Lavington.ea.2023,Kunstner.Yadav.Milligan.ea.2024` と対をなす結果である。また :citet:`Shah.Polloreno.Stratos.ea.2025` は、Muon が AdamW よりも大きなバッチサイズまでデータ効率を維持することを明らかにしている。**Scion** :cite:`Pethick.Xie.Antonakopoulos.ea.2025` は、AdamW にフォールバックすることなく、作用素ノルムによる最急降下法をすべての層（埋め込みや読み出しヘッドを含む）へと拡張する。

<!--en:88a4ea64-->
* **効率的・分散的な直交化：**まず **Polar Express** :cite:`Amsel.Persson.Musco.ea.2026` と :citet:`Cesista.You.Jordan.2025` は、ステップごとのニュートン・シュルツ多項式の係数（:numref:`fig_opt_ns_maps`）を最適化する。また :citet:`Kim.Oh.2026` は非厳密な $q$ ステップ直交化のもとでの収束を解析し、**Dion** :cite:`Ahn.Xu.Abreu.ea.2025` はシャーディングされた訓練におけるデバイス間通信を削減するため、低ランクのモメンタム近似を直交化する。

<!--en:aa166138-->
## スケジュール、平均化、大規模ベンチマーク

<!--en:1080d580-->
* **ウォームリスタート、スケジュールフリー最適化、モデルスープ：**コサインアニーリングはもともと周期的なウォームリスタートを伴う手法（**SGDR**、:citet:`Loshchilov.Hutter.2016`）として導入され、サイクル間で予測を平均する**スナップショットアンサンブル**（snapshot ensemble）:cite:`Huang.Li.Pleiss.ea.2017` を可能にした。これに対し、**モデルスープ**（model soup）:cite:`Wortsman.Ilharco.Gadre.ea.2022` は共通の初期化からファインチューニングされた複数のモデルの重みを平均する。**Schedule-Free SGD および AdamW** :cite:`Defazio.Yang.Mehta.ea.2024` は、現在の反復点と反復点の移動平均との補間点で勾配を評価することにより、あらかじめホライズン $T$ を定めるスケジュールを不要にする。

<!--en:13afdcce-->
* **バッチサイズランプとスケーリング則：**:citet:`Malladi.Lyu.Panigrahi.ea.2022` は、バッチサイズの増大に伴う Adam のモーメント $(\beta_1, \beta_2, \epsilon)$ に対する完全な確率微分方程式（SDE）スケーリング則を導出している。:citet:`Zhang.Morwani.Vyas.ea.2024` および :citet:`Merrill.Arora.Groeneveld.ea.2025` は、臨界バッチサイズがモデルサイズではなく主に処理済みトークン数に応じて増大することを示しており、:citet:`Dey.Gosal.Chen.ea.2023`、:citet:`Bi.Chen.Chen.ea.2024`、および :citet:`Meta.2025` は、実用言語モデルにおけるスケール間のハイパーパラメータ転移を報告している。

<!--en:44a96548-->
* **オプティマイザのベンチマークと評価プロトコル：**新しいオプティマイザ（LARS/LAMB :cite:`You.Gitman.Ginsburg.2017` や NorMuon :cite:`Li.Liu.Liang.ea.2026` など）による見かけ上の改善は、ベースライン側の局外ハイパーパラメータを同等の労力で調整すると、縮小または消失することが少なくない :cite:`Nado.Gilmer.Shallue.ea.2021,Schmidt.Schneider.Hennig.2021,Wen.Hall.Ma.ea.2025`。**AlgoPerf** :cite:`Dahl.Schneider.Nado.ea.2023,Kasimbeg.Schneider.Eschenhagen.ea.2025`、言語モデルのベンチマークスイート :cite:`Semenov.Pagliardini.Jaggi.2025`、および **NanoGPT speedrun** の最適化トラック :cite:`Jordan.2024,Jordan.2026` といった標準化されたベンチマークでは、明示的な調整予算と統計的有意性のしきい値が課されている。最後に、公開されている **OPT-175B 訓練ログブック** :cite:`zhang2022opt` には、大規模環境における訓練途中の損失スパイク、チェックポイントの巻き戻し、ハードウェア復旧のありのままの日次記録が残されている。

<!--en:744960d2-->
[ディスカッション](https://d2l.discourse.group/t/36851)
