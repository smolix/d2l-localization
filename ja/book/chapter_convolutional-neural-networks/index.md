<!--en:ef761b00-->
# 畳み込みニューラルネットワーク

<!--en:029944cc-->
これまでに構築してきた画像分類器はすべて、画像の空間的な幾何構造を無視し、2 次元のピクセルグリッドを 1 次元のベクトルへと平坦化していた。:numref:`chap_perceptrons` の MLP は、$28 \times 28$ の Fashion-MNIST 写真を順序のない 784 個の数値のリストとして扱う。そのため、すべての画像のピクセルに同じ固定のランダム置換を施して再訓練しても、並べ替えられた画像が人間の目にはただの砂嵐にしか見えないにもかかわらず、MLP はまったく同じ正解率に到達する。人間はスニーカーを輪郭やテクスチャ、すなわち*隣接する*ピクセルが形作るパターンによって認識しており、輪郭はフレーム内のどこに現れても同じ意味を持つ。

<!--en:9838d1c2-->
初期の手がかりは神経科学から得られた。ネコとサルを用いた古典的な実験において、Hubel と Wiesel は初期視覚野のニューロンが視野内の局所的な領域、すなわち各ニューロンの*受容野*（receptive field）に現れる特定の向きを持つエッジやバーに応答することを発見した :cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968`。その後、:citet:`Field.1987` は画像上のあらゆる位置にわたって走査する少数の局所フィルタからなるバンクを用いて、自然画像に対するこれらの応答をモデル化した（:numref:`field_visual`）。

<!--en:fb4f9a8d-->
![6 つの受容野フィルタと自然画像に対するその応答（:citet:`Field.1987` より改変）。左のパネルはフィルタ、中央のパネルは入力画像、右のパネルは各フィルタのサイズに比例した間隔でサンプリングされたフィルタ応答を示す。異なるフィルタは、それぞれ異なる局所的なエッジパターンに応答する。](../img/field-visual.png)

<!--en:b6ce82d4-->
これらの知見に触発され、Fukushima の Neocognitron :cite:`Fukushima.1980` は局所的な検出器を空間全体にわたって繰り返される階層的な層へと組織化し、続いて LeCun らは手書きの郵便番号を認識するために、そのようなアーキテクチャを誤差逆伝播法（backpropagation）によってエンドツーエンドで訓練する方法を示した :cite:`LeCun.Boser.Denker.ea.1989,LeCun.Jackel.Bottou.ea.1995`。さらに、訓練された視覚ネットワークのより深い層は、霊長類の視覚経路における高次の段階と表現上の類似性さえ示す :cite:`Kuzovkin.Vicente.Petton.ea.2018`。もっとも、生物学が果たしたのはアーキテクチャ上の着想を与える役割であり、シリコン上の計算に対する設計図ではなかった。

<!--en:65cf35a0-->
これらのアーキテクチャの中核にあるのが*畳み込み層*（convolutional layer）であり、共有された小さな重みのカーネルを画像のあらゆる近傍にわたってスライドさせる。各出力ユニットは近くのピクセルのみに依存し、入力を平行移動させれば出力もそのまま平行移動する。これにより、局所性（locality）と平行移動同変性（translation equivariance）をネットワークに直接組み込みつつ、パラメータ数を何桁も削減できる。こうした*畳み込みニューラルネットワーク*（convolutional neural network、CNN）は、ImageNet における分水嶺となるブレークスルーを牽引し :cite:`Deng.Dong.Socher.ea.2009,Krizhevsky.Sutskever.Hinton.2012`、ビジョン Transformer（:numref:`chap_transformers`）と並んで今なおコンピュータビジョンの礎であり続けている。

<!--en:d976ee85-->
本章では、3 つのステップで第一原理から CNN を導出し構築する。まず、全結合層に局所性と平行移動同変性を課すと、線形演算としては畳み込み（convolution）が唯一の選択肢になることを示す（:numref:`sec_why-conv`）。続いて 2 次元の相互相関（cross-correlation）を実装し、勾配降下法（gradient descent）によってエッジ検出カーネルを学習させ、畳み込みを行列積へと書き換え（*im2col*）、層を積み重ねることで受容野がどのように広がるかをたどる（:numref:`sec_conv_layer`）。次に、パディング・ストライド・ダイレーション（:numref:`sec_padding`）、マルチチャネルのフィルタバンクと $1 \times 1$ ボトルネック（:numref:`sec_channels`）、および局所プーリングとグローバルプーリング（:numref:`sec_pooling`）を通じて、空間解像度、視野、特徴の表現容量を制御する。最後に、これらの要素を LeNet（:numref:`sec_lenet`）へと組み立て、Fashion-MNIST 上で MLP と比較し、学習された最初の層のフィルタを観察する。さらに、対称性から MLP の正解率は変化しないピクセルシャッフル画像上で LeNet を再訓練することで、空間的な帰納バイアス（inductive bias）の寄与を切り分ける。現代の深層畳み込みアーキテクチャは、続く :numref:`chap_modern_cnn` で扱う。

<!--en:5dd576ec-->
## 参考資料と関連文献 {.unnumbered}

<!--en:a64691d2-->
以下の資料は本章の各節に対応しており、それぞれ関連する節を併記している。特記のない限り、すべてオンラインで無料公開されている。

<!--en:f3e47eb0-->
**書籍**

<!--en:e68aa4cd-->
- [Deep Learning (Goodfellow, Bengio & Courville)](https://www.deeplearningbook.org/)：無料の HTML 版。第 9 章で疎な相互作用、パラメータ共有、同変性を詳しく展開している（:numref:`sec_why-conv`、:numref:`sec_pooling`）。

<!--en:e398d41c-->
- [Understanding Deep Learning (Simon J. D. Prince)](https://udlbook.github.io/udlbook/)：無料の PDF 版。第 10 章で畳み込み、ストライド、ダイレーション、受容野を明快な幾何学的図解とともに解説している（:numref:`sec_conv_layer`、:numref:`sec_padding`）。

<!--en:2ca5481e-->
- [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/chap6.html)：オンラインで無料公開。第 6 章で畳み込み層、共有された重み、プーリングを第一原理から導入している（:numref:`sec_conv_layer`、:numref:`sec_pooling`）。

<!--en:51311a3d-->
**講義コースと講義動画**

<!--en:3911fa8a-->
- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/convolutional-networks/)：無料の講義ノート。「Convolutional Networks」モジュール（層のサイズ計算、パラメータ数の計算、層の配置パターン）は、本章とあわせて広く利用されているオンライン副教材である（:numref:`sec_padding`、:numref:`sec_channels`）。

<!--en:ffafa570-->
- [Michigan EECS 498-007: Deep Learning for Computer Vision (Justin Johnson)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/)：無料の講義動画。本章と同じペースで畳み込みとプーリングの計算を順を追って解説している（:numref:`sec_conv_layer`、:numref:`sec_pooling`）。

<!--en:b24c3731-->
**チュートリアル・解説記事・インタラクティブ教材**

<!--en:40460908-->
- [A guide to convolution arithmetic for deep learning (Dumoulin & Visin, 2016)](https://arxiv.org/abs/1603.07285)：無料、付属の[アニメーション](https://github.com/vdumoulin/conv_arithmetic)あり。パディング、ストライド、ダイレーション、転置畳み込みの視覚的なリファレンスである（:numref:`sec_padding`）。

<!--en:058c280d-->
- [CNN Explainer (Wang et al., Polo Club)](https://poloclub.github.io/cnn-explainer/)：活性化値、カーネル、受容野をブラウザ上で対話的に確認できる畳み込みネットワークの可視化ツールである（:numref:`sec_channels`、:numref:`sec_lenet`）。

<!--en:a492243e-->
- [Computing Receptive Fields of Convolutional Neural Networks (Araujo, Norris & Sim, 2019, *Distill*)](https://distill.pub/2019/computing-receptive-fields/)：ストライドを含む場合や複数経路を持つ場合も含め、:numref:`sec_conv_layer` で導入する受容野の閉形式の計算式を導出している。

<!--en:6e881d9e-->
- [Conv Nets: A Modular Perspective (Chris Olah)](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/)：構造化された重み共有としての畳み込みについて直感を養う、簡潔な定番記事である（:numref:`sec_why-conv`）。

<!--en:f65c5934-->
- [Image Kernels Explained Visually (Victor Powell)](https://setosa.io/ev/image-kernels/)：人手で設計したカーネルを画像上でスライドさせて試せるインタラクティブツールであり、:numref:`sec_conv_layer` を補完する。

<!--en:0c736337-->
- [Feature Visualization (Olah, Mordvintsev & Schubert, 2017, *Distill*)](https://distill.pub/2017/feature-visualization/)：畳み込みネットワークの各ユニットが何に応答するかを視覚的に探求する記事であり、:numref:`sec_lenet` に対応する。

<!--en:d63f4e40-->
**基礎となった論文**

<!--en:d85d5677-->
- [Gradient-Based Learning Applied to Document Recognition (LeCun, Bottou, Bengio & Haffner, 1998, *Proc. IEEE*)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)：無料の PDF 版。:numref:`sec_lenet` の背景にある LeNet-5 の論文である。

<!--en:d46d3469-->
- [Backpropagation Applied to Handwritten Zip Code Recognition (LeCun et al., 1989, *Neural Computation*)](https://doi.org/10.1162/neco.1989.1.4.541)：誤差逆伝播法によってエンドツーエンドで訓練された最初の畳み込みネットワークの論文である（購読制限あり）（:numref:`sec_lenet`）。

<!--en:9410a7b8-->
- [Neocognitron (Fukushima, 1980, *Biological Cybernetics*)](https://doi.org/10.1007/BF00344251)：畳み込みとプーリングの段階を交互に重ねる構造の、誤差逆伝播法以前における先駆的研究である（購読制限あり）（:numref:`sec_pooling`）。

<!--en:fe39f3f3-->
- [Receptive fields, binocular interaction and functional architecture in the cat's visual cortex (Hubel & Wiesel, 1962, *J. Physiology*)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/)：無料公開（PMC）。局所受容野と階層的な特徴検出の生物学的起源となった論文である（:numref:`sec_why-conv`）。

<!--en:49541e54-->
畳み込みニューラルネットワーク：幾何構造を考慮した視覚層<br>**局所性・平行移動同変性・相互相関・パディング・チャネル・プーリング・LeNet**

<!--en:67402ace-->
::: {.slide title="画像の平坦化による空間的幾何構造の破壊"}

<!--en:ae166c83-->
[動機と帰納バイアス]{.kicker}

<!--en:d1c67e2f-->
- **置換に対する盲目性：**Fashion-MNIST の全 784 ピクセルを 1 つの固定置換で並べ替えても、MLP の正解率は変わらない（$\approx 87\%$）

<!--en:874a0bc3-->
- **大規模化に伴うパラメータ爆発：**1 メガピクセルの RGB 全結合層には $(3 \times 10^6)^2 = 9 \times 10^{12}$ 個の重みが必要になる（`float32` で **36 TB**）

<!--en:bdbc2d1f-->
- **幾何学的な事前知識：**層に**局所性**と**平行移動同変性**を課すことで、数兆個の重みが数百個のパラメータからなる共有カーネルへと集約される

<!--en:c2ffd101-->
::: {.slide title="生物とシリコンの視覚における受容野"}

<!--en:3ae29bb0-->
[視覚野から畳み込みネットワークへ]{.kicker}

<!--en:49025cdd-->
Hubel と Wiesel は、初期視覚野のニューロンが局所的な*受容野*内の特定方向のエッジに応答することを発見し、空間全体を走査する階層的なフィルタバンクの着想をもたらした。

<!--en:02cc4043-->
- **局所フィルタバンク（左 $\to$ 右）：**向きの異なる 6 つのエッジフィルタを自然画像に畳み込むと、異なる向きやスケールの局所的な輪郭を際立たせるスパースな活性化マップが得られる

<!--en:1169bf6b-->
- **エンドツーエンド学習：**畳み込みニューラルネットワーク（CNN）はフィルタバンクを人手で設計する代わりに、誤差逆伝播法によって全層のカーネルの重みを同時に学習する

<!--en:39d2dfa9-->
::: {.slide title="第 §{num:chap_cnn} 章の数理とアーキテクチャのロードマップ"}

<!--en:00be04d1-->
[章の構成]{.kicker}

<!--en:6e6c72e3-->
- **§{sec_why-conv}〜§{num:sec_conv_layer}（対称性と畳み込み層）：**平行移動同変性と局所性から 2D 相互相関 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ が導かれ、`im2col` による GEMM で実行される

<!--en:e2d5e914-->
- **§{sec_padding}〜§{num:sec_channels}（幾何構造とチャネル）：**パディング $p$、ストライド $s$、ダイレーション $d$、4D フィルタバンク $(c_\textrm{o}, c_\textrm{i}, k_\textrm{h}, k_\textrm{w})$、および $1 \times 1$ ボトルネック

<!--en:67976edf-->
- **§{sec_pooling}〜§{num:sec_lenet}（プーリングと LeNet）：**パラメータを持たないプーリングと、MLP と比較した LeNet のエンドツーエンド検証

<!--en:d8bc79ea-->
::: {.slide title="第 §{num:chap_cnn} 章を通じた具体的なベンチマーク"}

<!--en:418d9f93-->
[定量的な基準点]{.kicker}

<!--en:f9c20798-->
- **解像度に依存しない重み：**$3 \times 3$ の RGB から 64 チャネルへのステムは、任意の画像解像度で $1{,}792$ 個のパラメータを使用し、$1 \times 1$ ボトルネックは深い層における $3 \times 3$ のコストを $8.5\times$ 圧縮する

<!--en:f4667bb8-->
- **空間的な到達範囲と位置ずれ耐性：**ストライド $(2,2,1)$ またはダイレーション $(1,2,4)$ により 3 層の到達範囲は $r=7$ から $r=15$ へ倍増し、$2 \times 2$ プーリングは 1 px のずれの $50\%$ を吸収する

<!--en:e3569e2a-->
- **LeNet の計算資源の内訳：**畳み込み層が保持する重みは全体の $4.2\%$ にすぎないが、MAC 演算の $85.9\%$ を実行する

<!--en:2caa5954-->
::: {.slide title="概念チェックと本章の学習戦略"}

<!--en:6a3f295c-->
[確認と統合]{.kicker}

<!--en:3ed99e5d-->
- **1. 解像度のスケーリング（$224^2 \to 448^2$）：**高さと幅が倍になるとき、パラメータ数、MAC 数、活性化値の RAM 使用量はそれぞれどのようにスケールするか。

<!--en:81f49b72-->
- **2. 帰納バイアスの検証：**自然画像では CNN が MLP を上回るにもかかわらず、固定のランダムなピクセル置換のもとではその優位性を失うのはなぜか。

<!--en:e6db9f74-->
- **解答：**パラメータ数は **$1\times$** のまま変わらず（$c_\textrm{o}, c_\textrm{i}, k$ で固定）、MAC 数と活性化値は **$4\times$** にスケールする（面積 $H \times W$ に比例）。シャッフルは 2D の局所性を破壊する。
