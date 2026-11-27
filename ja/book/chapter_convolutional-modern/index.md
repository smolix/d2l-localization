<!--en:c3e528a8-->
# 現代の畳み込みニューラルネットワーク

<!--en:fcbebf97-->
畳み込み層と LeNet の仕組みを押さえたところで（:numref:`chap_cnn`）、本章では 2012 年の ImageNet における AlexNet のブレークスルーから、2020 年代初頭の畳み込みネットワークと Vision Transformer の対話に至るまで、10 年にわたる畳み込みアーキテクチャ設計の歩みをたどる。節目となった各モデルを孤立した産物として扱うのではなく、どのようなボトルネックがそれぞれの設計ステップを促したのかを問い、章全体を通して累積的なツールキットを引き継いでいく。

<!--en:b5e7c447-->
冒頭の各節では、個別に手作業で組まれた層からモジュール式の構成ブロックへと進む。AlexNet（:numref:`sec_alexnet`）は、大規模なラベル付きデータセットと GPU が結びついたことで、エンドツーエンドの特徴学習（feature learning）がいかに人手で設計されたビジョンパイプラインを凌駕したかを実証している。同モデルは ReLU 活性化関数、ドロップアウト、データ拡張（data augmentation）、および本章全体で再利用するウォームアップ付きコサイン学習率スケジュールを導入した一方で、畳み込みの計算量と全結合ヘッドのパラメータ数との間にある際立った不均衡も浮き彫りにした。続く VGG、Network in Network（NiN）、GoogLeNet（:numref:`sec_blocks`）は、繰り返し可能な $3 \times 3$ ブロックによるネットワークの構成、$1 \times 1$ 畳み込みによるピクセルごとのチャネル混合、巨大な全結合ヘッドからグローバル平均プーリング（global average pooling）への置き換え、そしてステム・ボディ・ヘッド構造の内部での並列マルチスケール分岐の組み合わせによって、この不均衡に対処する。

<!--en:4e79ff87-->
中盤の各節が扱うのは、深さとハードウェア効率のスケーリングである。正規化層（:numref:`sec_batch_norm`）は、ミニバッチ全体または個々のデータ例の内部で中間の活性化値を標準化し、隠れ層を重みのスケールに対して不感にすることで、実用可能な学習率の幅を何桁も広げる。残差ネットワーク（:numref:`sec_resnet`）は、加算ショートカット $\mathbf{x} + g(\mathbf{x})$ により $g = 0$ で恒等写像（identity mapping）を保つことで、深いプレーンな積層ネットワークにおける最適化の劣化を解消し、さらにグループ畳み込み（ResNeXt）および特徴の連結（DenseNet）によってブロックを拡張する。効率的な畳み込みネットワーク（:numref:`sec_efficient_cnns`）では、正解率と並ぶ重要な設計目標としてレイテンシ、メモリ、エネルギーを位置づけ、深さ方向分離可能な分解（depthwise-separable factorization）、逆ボトルネック（inverted bottleneck）、複合スケーリング（compound scaling）、Squeeze-and-Excitation ゲート、構造的再パラメータ化（structural re-parameterization）を導入するとともに、演算回数が GPU の実経過時間としばしば乖離する理由を明らかにする。

<!--en:e931cb7c-->
終盤の各節では、訓練手法とアーキテクチャを切り分ける。現代の訓練レシピ（:numref:`sec_training_recipes`）では、AdamW、コサインウォームアップ、ラベル平滑化（label smoothing）、Mixup、CutMix、確率的深さ（stochastic depth）、重み平均化といった現代的な最適化と正則化だけでも、アーキテクチャを変えない ResNet において ImageNet の正解率の数パーセントポイント分を担っていることを示す。このようなレシピを固定したうえで、ConvNeXt（:numref:`sec_convnext`）は ResNet を段階的に現代化し、アテンションを用いずに階層型 Vision Transformer に匹敵する性能を達成する。そして畳み込みネットワークの設計空間（:numref:`sec_cnn-design`）では、サンプリングされたモデル集団にわたる経験誤差の分布を比較することで、RegNet ファミリーの簡潔な構造規則を抽出する。

<!--en:e0422d3d-->
現代の畳み込みニューラルネットワーク<br>**AlexNet・ブロック・正規化・残差ショートカット・効率的な畳み込みネットワーク・訓練レシピ・ConvNeXt・RegNet**

<!--en:c9a3fb38-->
::: {.slide title="なぜ LeNet の層をさらに積み重ねても実用上うまくいかないか"}

<!--en:acc8203e-->
[中心的な問題]{.kicker}

<!--en:8e2134cc-->
- **信号ドリフトと劣化：**シグモイドは勾配を $\le 1/4^L$ 倍に縮小させ、ReLU + BN を用いても 56 層のプレーンなネットワークは 20 層より訓練性能が劣る（損失 $\approx 0.28$ 対 $0.20$）

<!--en:42c9fb0c-->
- **92% に及ぶ全結合ヘッドの肥大化と FLOPs 対レイテンシ：**AlexNet・VGG は重みの $>92\%$ をヘッドに抱え、深さ方向畳み込みは MAC 数で $8\times$ の削減を達成しても GPU のメモリ帯域幅の限界に突き当たる

<!--en:8dc95df8-->
- **訓練レシピの交絡：**現代的なレシピ（LAMB、BCE 損失、RandAugment・Mixup・CutMix、600 エポック）だけで、構造を変えない ResNet-50 が $4.3$ ポイント向上する（$76.1\% \to 80.4\%$）

<!--en:38b564de-->
::: {.slide title="本章を通じたアーキテクチャのツールキット"}

<!--en:99e0030a-->
[章のロードマップ]{.kicker}

<!--en:a7bdbd8d-->
- **§{sec_alexnet}〜§{num:sec_blocks}（AlexNet、VGG、NiN、GoogLeNet）：**ステム・ボディ・ヘッドのテンプレート内での ReLU、ドロップアウト、$3 \times 3$ ブロック、$1 \times 1$ ボトルネック、グローバル平均プーリング

<!--en:c5d48b77-->
- **§{sec_batch_norm}〜§{num:sec_resnet}（正規化と残差）：**BatchNorm、LayerNorm、GroupNorm、加算ショートカット $\mathbf{x} + g(\mathbf{x})$、グループ化 ResNeXt ブロック、DenseNet の連結

<!--en:136cab62-->
- **§{sec_efficient_cnns}〜§{num:sec_cnn-design}（効率性、レシピ、ConvNeXt、RegNet）：**深さ方向分解、RepVGG の融合、AdamW・Mixup・EMA、$7 \times 7$ ConvNeXt ブロック、集団の誤差 CDF

<!--en:a2a396a1-->
::: {.slide title="3 つの統一的な設計不変則"}

<!--en:acd261d6-->
[中核となるメンタルモデル]{.kicker}

<!--en:c98fd784-->
- **1. FLOP あたりの受容野と GAP：**積層した $3 \times 3$ 畳み込みはより多くの非線形性を伴って $5 \times 5$（$18c^2$ 対 $25c^2$）や $7 \times 7$（$27c^2$ 対 $49c^2$）をカバーし、GAP は 4300 万パラメータのヘッドを不要にする

<!--en:b524125d-->
- **2. スケール不変性と入れ子の関数クラス：**$\textrm{BN}(a\mathbf{w}^\top\mathbf{h}) = \textrm{BN}(\mathbf{w}^\top\mathbf{h})$ が成り立ち、$\mathbf{x} + g(\mathbf{x})$ は $g=0$ で恒等写像を保つ（$\mathcal{F} \subseteq \mathcal{F}'$）

<!--en:7d7f0815-->
- **3. 分解と演算強度：**深さ方向分離可能畳み込みは MAC 数を $1/c_\textrm{o} + 1/k^2 \approx 1/9$ に減らす一方、メモリ読み出しあたりでは $9$ 対 $9c_\textrm{o}$ MAC にとどまる

<!--en:e47006f5-->
::: {.slide title="条件を揃えた Fashion-MNIST 通しベンチマーク"}

<!--en:343fa0a9-->
[実証的な比較]{.kicker}

<!--en:e8dd206b-->
- **AlexNet $\to$ Quarter-VGG / ResNet-18：**正解率が $91\%$ から $93\text{--}94\%$ へ上昇する一方、パラメータ数は $46.8\textrm{M}$ から $11.2\textrm{M}$（GAP 使用時は $0.6\textrm{M}$）へ縮小する

<!--en:9616121d-->
- **BNLeNet と ResNet-56：**BatchNorm は実用可能な学習率の範囲を $300\times$ に広げ（$\eta_0 \in [0.03, 10]$）、残差スキップは 56 層での劣化を逆転させる

<!--en:684beff0-->
- **ConvNeXt と RegNetX32：**ResNet-18（$\approx 94\%$）と同等の正解率を、$3.38\textrm{M}$ と $0.52\textrm{M}$ のパラメータ数（$21\times$ 小規模）で達成する
