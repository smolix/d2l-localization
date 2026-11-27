<!--en:5a7d784a-->
# 線形代数

<!--en:0c6eb0f1-->
本書に登場するすべてのニューラルネットワークは、行列積から組み立てられている。線形層は入力に重み行列を掛け、アテンションはクエリにキーを掛け、リカレントネットワークはステップごとに隠れ状態に同じ行列を掛ける。:numref:`sec_linear-algebra` では、行列が作用するベクトルに対して実際に*何をするのか*を問うことなく、これらの積を計算した。選択章である本章では、証明も含めてこの幾何学的な問いに取り組む。本文の内容は本章に一切依存しないが、行列積の背後にある幾何学が見えてくると、本書に登場するほぼすべてのモデルの読み解き方が変わるだろう。

<!--en:11da4ad8-->
784 ピクセルの Fashion-MNIST 画像を 256 個の特徴量へ写像することを考えよう。2 枚の画像はどれほど似ているか、この層はどれだけの情報を捨てるか、そして同じ線形変換（linear map）を 50 回続けて適用すると何が起きるだろうか。784 次元の空間に図を描くことはできないため、その答えは平面幾何を任意の次元へと拡張する代数から得られる。

<!--en:cde83739-->
複雑な行列も、適切な方向に沿って見れば見通しがよくなる。1810 年頃、Gauss は未知数より多くの観測値に対して最小二乗法（least squares）で小惑星パラスの軌道を当てはめたが、これは本章の最初の節で解き、最後の節で再び取り上げる問題である。正規直交軸のもとでは長さと角度がドット積に帰着し、固有ベクトル（eigenvector）に沿えば正方行列は各方向を 1 つのスカラー倍にし、1 組の特異基底のもとでは*任意の*行列が独立な伸縮へと分解される。アテンションがなぜスコアを $\sqrt{d}$ で割るのか（:numref:`sec_attention-scoring-functions`）、リカレント勾配がいつ爆発するのか（:numref:`sec_bptt`）、そして低ランクアダプタが何を表現できるのかは、これらの考え方から説明できる。

<!--en:f7f9125b-->
本章は 2 つの段階で進む。第 1 段階では、行列を 1 回だけ適用する。:numref:`sec_mdl-geometry-linear-algebraic-ops` では長さ、角度、および部分空間（subspace）内で最も近い点を与える直交射影（orthogonal projection）を高次元へ拡張し、:numref:`sec_mdl-linear-maps` ではどの方向が 0 に潰れるか、また写像がいつ可逆になるかを追跡する。第 2 段階では、行列に合わせた座標系を選ぶ。対称行列に対する固有ベクトルとスペクトル定理（:numref:`sec_mdl-eigendecompositions`）、固有値（eigenvalue）だけでは捉えきれない非正規な振る舞い（:numref:`sec_mdl-spectra-in-practice`）、そして任意の形状に対する特異ベクトル（singular vector）と最適な低ランク打ち切り（:numref:`sec_mdl-svd-low-rank`）を順に扱う。本章全体を通じて、安易な思い込みを戒める 2 つの小さな行列を一貫して用いる。複素固有値を持つ $\mathbf{A}=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ と、固有ベクトルの張る空間（span）が平面全体に満たないせん断（shear）$\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ である。

<!--en:3796f690-->
初読の際は、$n$ 次元の行列式（determinant）や、:numref:`sec_mdl-spectra-in-practice` における Gershgorin の円盤、PageRank、擬スペクトル（pseudospectrum）の議論は流し読みしてかまわない。数値的な固有値ソルバと線形ソルバについては :numref:`sec_mdl-linear-algebra-further-reading` で概観する。

<!--en:21595c10-->
線形代数：幾何学、スペクトル、低ランク構造<br>**射影・線形変換・固有値分解・スペクトル安定性・特異値分解**

<!--en:8e674e67-->
::: {.slide title="なぜ深層学習に幾何学的な線形代数が必要か"}

<!--en:e02fdbdc-->
[動機]{.kicker}

<!--en:499aa373-->
- **機械的な積の計算を超えて：**$\mathbf{X}\mathbf{W}$ を幾何学的に捉えることで、層が $\mathbb{R}^d$ 内で何を保ち、何を潰し、何を増幅するかが見えてくる

<!--en:fe24de8c-->
- **高次元の直感：**ドット積によって、長さ、角度、直交する影の概念を 784 ピクセルの画像や 4096 次元の埋め込みへそのまま持ち込める

<!--en:b186d65d-->
- **自然な軸の選択：**部分空間の正規直交基底、固有ベクトル、または特異ベクトルに沿うことで、行列の構造が見通しよくなる

<!--en:e9eba208-->
::: {.slide title="2 段階のロードマップ：単一の写像から自然な軸へ"}

<!--en:ddfd1d8f-->
[章の構成]{.kicker}

<!--en:a8f4aa03-->
- **第 1 段階：標準座標（§{sec_mdl-geometry-linear-algebraic-ops}〜§{num:sec_mdl-linear-maps}）：**類似度を測り、部分空間へ射影し（$\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$、$\mathbf{A} = \mathbf{Q}\mathbf{R}$）、残る次元と潰れる次元を追跡する（$\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n$、$\det\mathbf{A}$）

<!--en:39983e36-->
- **第 2 段階：行列に合わせた座標（§{sec_mdl-eigendecompositions}〜§{num:sec_mdl-svd-low-rank}）：**反復される正方変換を対角化し（$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$）、べき乗法と非正規な増大を解析し、任意の $m \times n$ 行列を SVD で分解する（$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$）

<!--en:632b51b4-->
::: {.slide title="第 §{num:chap_mdl-linear-algebra} 章の 3 つの重要定理"}

<!--en:2c5478a1-->
[中核となる理論]{.kicker}

<!--en:af4f3570-->
- **1. 直交射影（$\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$）：**$S$ 内の最近接点は残差 $\mathbf{y} - \mathbf{P}\mathbf{y} \perp S$ を残し、最小二乗法を直角三角形の幾何へと帰着させる

<!--en:5fa31978-->
- **2. スペクトル定理（$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$）：**すべての実対称行列は実固有値と正規直交固有基底を持つ

<!--en:824da38e-->
- **3. Eckart–Young–Mirsky の定理（$\mathbf{A}_k$）：**SVD をランク $k$ で打ち切ると、スペクトルノルムおよびフロベニウスノルムにおける最適なランク $k$ 近似が得られる

<!--en:f71b0cda-->
::: {.slide title="通しで用いる行列と実証ベンチマーク"}

<!--en:05adf916-->
[各節をつなぐ軸]{.kicker}

<!--en:3662cffc-->
- **具体的な $2 \times 2$ テスト行列：**通しで用いる行列 $\mathbf{A}$（複素固有値 $\lambda = 2 \pm i$、$\det = 5$）、欠陥を持つせん断 $\mathbf{T}$（$g = 1 < a = 2$、線形ドリフト）、およびランク 1 の特異行列 $\mathbf{B}$

<!--en:14c9a2bf-->
- **高次元の実データ：**$\mathbb{R}^{784}$ 上の Fashion-MNIST（正解率 $92\%$ の最近傍平均超平面、非等方なコサイン）と $400 \times 500$ の猫の写真（容量 $9\%$ のランク 20 SVD で誤差 $8\%$）

<!--en:15d021a2-->
::: {.slide title="本章の読み進め方"}

<!--en:123f19cf-->
[学習ガイド]{.kicker}

<!--en:8bad973b-->
- **初読で押さえるべき要点：**直交射影と $1/\sqrt{d}$ のアテンションスケーリング（§{sec_mdl-geometry-linear-algebraic-ops}）、列空間・零空間とランク・退化次数の定理（§{sec_mdl-linear-maps}）、スペクトル定理と正定値性（§{sec_mdl-eigendecompositions}）、べき乗法と過渡的増大（§{sec_mdl-spectra-in-practice}）、および SVD・LoRA・条件数のワークフロー全体（§{sec_mdl-svd-low-rank}）に重点を置く

<!--en:17f74952-->
- **初読で流し読みしてよい箇所：**置換による公理的な $n$ 次元行列式（§{sec_mdl-linear-maps}）、ならびに Gershgorin の円盤、PageRank、$\varepsilon$ 擬スペクトル（§{sec_mdl-spectra-in-practice}）は、本筋を見失うことなく読み飛ばせる

<!--en:fc711756-->
- **実空間と複素空間：**本章全体を通して実ベクトル空間 $\mathbb{R}^d$ で議論を進め、複素数 $\mathbb{C}$ は回転行列や $\mathbf{A}$ のような実行列から複素固有値が必然的に現れる場面にのみ導入する

<!--en:67b05a19-->
- **数値計算の視点：**すべての節で厳密な定理を浮動小数点演算の現実と対にし、なぜコサインを $[-1, 1]$ にクリップし、正規方程式の代わりに QR・SVD を使い、`solve` を呼んで `inv` を避け、微小な特異値をしきい値処理するのかを明らかにする

<!--en:43c83f81-->
**次の節（§{sec_mdl-geometry-linear-algebraic-ops} ベクトル、角度、射影）：**ドット積が平面上の長さ、角度、直交する影、分離超平面を 784 次元へどのように拡張するかを見る。
