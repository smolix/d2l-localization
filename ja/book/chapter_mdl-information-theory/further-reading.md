<!--en:7902ddfd-->
# 関連文献

<!--en:cb8b64be-->
本章の主要な節では、明快に証明でき、かつ小さな例で確認できる結果に対象を絞ってきた。以下の各小項では、それぞれの節の議論を引き継ぎ、本筋には収まらなかった周辺の考え方を取り上げる。時間がない場合は、日々の実践で特に直接現れる 3 つのトピック、すなわちモデルの較正（calibration）、スケーリング則（scaling law）における削減不能損失の定数、および相互情報量（mutual information）推定量のバイアス・分散のトレードオフから読み始めるとよい。ビッツバック符号化（bits-back coding）も、その仕掛け自体がきわめて巧妙であるため一読の価値がある。通信路容量（channel capacity）と情報幾何学（information geometry）は後回しにしても差し支えなく、本書の以降の箇所がこのページの内容に依存することはない。

<!--en:198fec87-->
## エントロピーと損失

<!--en:50387741-->
### 訓練後の較正

<!--en:b874e174-->
原理的には、母集団の対数損失（log-loss）を最小化すれば真の条件付きクラス確率が復元され（:numref:`subsec_mdl-irreducible-loss`）、予測は自動的に較正される。しかし、訓練済みのニューラルネットワークがその最適解に落ち着くことはまれである。:citet:`Guo.Pleiss.Sun.Weinberger.2017` が指摘したことで知られるように、現代のネットワークは特定の方向に偏って誤る、すなわち過信（overconfidence）に陥る傾向がある。その主な原因はワンホットの訓練ラベルにある。ワンホットラベルのもとでは、最大ロジットを他のロジットより無限に大きく引き離すほど損失が下がるため（:numref:`sec_mdl-information_theory`）、ネットワークが訓練セットを完全に分類できるようになった後は、ロジットどうしの差をさらに広げ続けることでしか損失を下げられない。訓練の終盤において同著者らは、ホールドアウトの分類誤差が横ばいのままであるにもかかわらず、ホールドアウトの対数損失が上昇し始めることを観察している。ネットワークは意思決定ではなく確率推定に対して過学習しているのである。温度スケーリング（temperature scaling、:numref:`sec_classification`）は、ホールドアウトの検証セット上で当てはめた単一のスカラーによって、この問題の大部分を解消する。これは蒸留（distillation）で用いられるものと同じ温度パラメータであり、ここでは過信した予測を軟化させる目的で転用されている。

<!--en:231481ef-->
このような事後的な再較正（post-hoc recalibration）はどこまで有効だろうか。$\bar{\mathbf{p}}(\mathbf{q})$ を、モデルが確率ベクトル $\mathbf{q}$ を出力するすべての入力における実際の経験クラス頻度とする。任意の交差エントロピーはエントロピーと KL ダイバージェンスの和に分解されるため、予測ベクトル $\mathbf{q}$ ごとに条件付きでこの恒等式を適用すると、ホールドアウトの対数損失は次の 2 つの項に分解される。

<!--en:9284eb3f-->
第 1 項は較正のずれ（miscalibration）を nat 単位で測るものであり、モデルの予測確率が主張するとおりの頻度で実際に事象が起こるとき、かつそのときに限り 0 になる。第 2 項は、モデルの予測を知った後でもなお真のラベルについて残る不確実性である。予測確率を事後的にどのように再写像しても、第 1 項を 0 に近づけることはできるが、第 2 項を減らすことは決してできない。温度スケーリングはこのような再写像の 1 パラメータ族にほかならず、それゆえきわめて簡潔であるが、一部のクラスでは過信し別のクラスでは過小確信（underconfidence）に陥っているモデルを 1 つのスカラーで修正することはできない。より一般に、あらゆる固有スコアリングルール（proper scoring rule）は、ダイバージェンス項とエントロピー項への同様の分解をもつ :cite:`Gneiting.Raftery.2007`。

<!--en:e798f114-->
### 参照に対する相対的なエントロピー

<!--en:dd615881-->
:numref:`subsec_mdl-differential-entropy` では、微分エントロピー（differential entropy）の厄介な性質を見た。単位をメートルからセンチメートルに変えるだけで、値が $\ln 100$ だけ増加してしまうのである。これに対し、KL ダイバージェンスはまったく変化しない。比の内部にある両方の密度が同じ係数でスケールされるからである。参照密度 $m$ に対して測り、$-\int p \log(p/m)\,dx = -D_{\textrm{KL}}(p\|m)$ と定義すれば、エントロピーにも同じ座標不変性を与えることができる。通常の微分エントロピー $h$ は、たまたま選んだ単位系における $m = 1$ という特別な場合にすぎない。1 メートルあたり 1 の一様密度と 1 センチメートルあたり 1 の一様密度は同じ測度ではないため、先の $\ln 100$ のずれの正体は隠れた参照測度の変更だったのである。:citet:`Jaynes.1968` は、次第に細かくなるグリッド上の離散エントロピーの極限としてこの相対形を導出しており、これを用いると最大エントロピー原理（maximum-entropy principle）は「制約が許す限り参照測度の近くにとどまれ」という自然な形をとる。符号の違いを除けば、変分オートエンコーダ（variational autoencoder、:numref:`subsec_mdl-variational-bottleneck`）における KL ペナルティは事前分布（prior）が $m$ の役割を果たすこの種のエントロピーであり、潜在空間（latent space）の滑らかな再パラメータ化のもとで不変に保たれる理由もそこにある。

<!--en:0b7a4e8f-->
## 符号と圧縮

<!--en:ee8c3a56-->
### モデルを用いない圧縮

<!--en:ca28237f-->
:numref:`sec_mdl-coding-compression` で扱ったすべての圧縮器は、次の記号を予測するために明示的な確率モデルに依拠している。レンペル・ジブ（Lempel–Ziv）族はその要件を完全に不要にする。LZ77 :cite:`Ziv.Lempel.1977` は繰り返し現れる任意の部分文字列をそれ以前の出現箇所を指す短いポインタに置き換え、LZ78 :cite:`Ziv.Lempel.1978` はファイルを走査しながらフレーズの辞書をオンザフライで構築していく。注目すべきことに、両アルゴリズムは*普遍*（universal）である。すなわち、任意の定常エルゴード情報源（stationary ergodic source、大まかに言えば 1 つの長いサンプルから長期的な統計的性質が明らかになる情報源）に対し、情報源に関する事前知識を一切与えなくても、ファイルが大きくなるにつれて平均符号長が真のエントロピー率（entropy rate）へと近づく。もっとも実際には、その極限への収束は遅い。『The Time Machine』に対して、（LZ77 とハフマン符号化を組み合わせた）`zlib` は 1 文字あたり $2.84$ ビットを要するのに対し、直前の 3 文字から各文字を予測する :numref:`subsec_mdl-lm-compressor` の単純なカウントモデルは、すでに $2.55$ ビットまで到達している。

<!--en:7e225c8f-->
### ビッツバック符号化

<!--en:1bdf9c43-->
変分オートエンコーダのような潜在変数モデル（latent-variable model）では、:numref:`subsec_mdl-arithmetic-coding` の算術符号器を直接駆動することはできない。符号器には厳密な周辺確率（marginal probability）$p(x)$ が必要であるのに対し、モデルが与えるのは ELBO（:numref:`sec_mdl-latent-em-elbo`）のみであり、これは $\log p(x)$ に対する、近似事後分布（approximate posterior）$q(z \mid x)$ から構成された下界だからである。驚くべきことに、圧縮ストリームのデコードは数学的にはサンプリングと同じであるという巧妙な双対性（duality）のおかげで、この下界もなお達成可能なファイル長に対応する。$q(z \mid x)$ 向けに設計されたデコーダにランダムなビット列を入力すると、デコーダは約 $\log_2 1/q(z \mid x)$ ビットを消費し、潜在サンプル $z$ を $q$ からの抽出結果として出力する。

<!--en:aedd0e83-->
*ビッツバック符号化*は、この知見を実用的な方式へと昇華させる。送信者がそれ以前のデータ点を収めた圧縮ファイルをすでに持っているとする。優れた圧縮は冗長性を取り除くため、そのファイル末尾のビット列はランダムなコイン投げと区別がつかないように見える。新しいデータ点 $x$ を符号化するため、送信者はファイル末尾から $\log_2 1/q(z \mid x)$ ビットを借り受け、それを $q(z \mid x)$ で「デコード」して潜在変数 $z$ を抽出する。次に送信者は、$x$ を $p(x \mid z)$ で符号化し、$z$ を事前分布 $p(z)$ で符号化して、その両方をファイルに追記する。$x$ を展開するため、受信者はこの手順を逆にたどる。まず $z$ を事前分布からデコードし、次に $x$ を $p(x \mid z)$ からデコードしたうえで、$z$ を $q(z \mid x)$ によって*再符号化*し、借りていたビット列を元の場所へとそっくり戻す。全体として、ファイルは $\log_2 1/p(x \mid z) + \log_2 1/p(z)$ ビットだけ増え、$\log_2 1/q(z \mid x)$ ビットだけ縮むため、$q$ のもとでの正味の期待増加量は負の ELBO となる。受信者は送信者の操作を後入れ先出し（last-in, first-out）の順序で取り消さなければならないため、基盤となる符号器はスタックのように振る舞う必要がある。ビッツバックの議論自体は :citet:`Wallace.1990` および :citet:`Hinton.Van-Camp.1993` に遡り、:citet:`Frey.Hinton.1997` はキューとして振る舞う従来の算術符号化（arithmetic coding）の上に最初の動作する符号器を構築した。非対称記数系（asymmetric numeral systems）:cite:`Duda.2013` は本来的にスタックベースであり、これを用いて :citet:`Townsend.Bird.Barber.2019` は、小規模な変分オートエンコーダで MNIST をその負の ELBO のすぐ近くのレートまで圧縮できることを実証した。

<!--en:71f3fe1f-->
### 2 部符号を超える記述長

<!--en:90b82ad3-->
:numref:`subsec_mdl-mdl` の 2 部符号（two-part code）は、選択した精度でモデルパラメータを明示的に符号化するため、総記述長（description length）がパラメータ空間をどれだけ細かく離散化するかに敏感になる。*逐次予測*（prequential）符号は、パラメータの送信を一切行わない。その代わり、:numref:`subsec_mdl-lm-compressor` の適応カウントモデルとちょうど同じように、それまでに観測したデータ点に当てはめたモデルを用いて各データ点を符号化するため、モデルの学習コストはデータの符号長に直接吸収される。:citet:`Blier.Ollivier.2018` は逐次予測符号化を用いて深層ニューラルネットワークの訓練ラベルを圧縮し、2 部符号や変分符号（ネットワークの重みにビッツバック符号化を適用する手法）よりもはるかに短い符号を得た。大幅に過剰パラメータ化されたネットワークであっても、この尺度で測ればラベルを確かに圧縮しているのである。:citet:`Grunwald.2007` は、逐次予測符号と正規化最尤法（normalized maximum likelihood、事後的に選んだクラス最良のモデルに対する最悪ケースのリグレットを最小化する符号）の双方を詳しく展開している。この圧縮の視点を理論的な極限まで推し進めると、データを出力する最短のコンピュータプログラムの長さであるコルモゴロフ複雑性（Kolmogorov complexity）へと至る。これは以上の考え方をすべて統一する概念であるが、計算不能であることでよく知られている :cite:`Li.Vitanyi.2019`。

<!--en:aa0d5d2d-->
### スケーリング則の定数

<!--en:8c0f0f75-->
:numref:`sec_scaling-laws` では、多様なスケールにわたる言語モデルのホールドアウト損失を、定数 $E$ と 2 つのべき乗則（power law）の項（一方はパラメータ数に応じて減衰し、もう一方は訓練トークン数に応じて減衰する項）の和としてモデル化している :cite:`hoffmann2022training`。$E$ を自然テキストの真のエントロピーと読みたくなるかもしれないし、実際、最良の場合には $E$ はトークンあたりのエントロピー率（:numref:`subsec_mdl-entropy-rate`）、すなわちいかなる期待損失も下回ることのできない下限に対応する。しかし実際には、$E$ はそれよりはるかに上にある損失から外挿して当てはめられた回帰定数にすぎず、エントロピー率の直接的な測定値ではない。先行研究である :citet:`kaplan2020scaling` は加法定数を一切含まない同様のスケーリング曲線を当てはめていたが、損失はエントロピーの下限を下回れないため、この近似は十分に大きなスケールではいずれ破綻せざるを得ない。また、$E$ がトークン単位で測られている点にも留意すべきである。異なるトークナイザのもとで当てはめられた定数は、双方をバイトあたりビット数（:numref:`subsec_bits-per-byte`）へ変換しない限り直接比較することはできない。

<!--en:91d5bbee-->
## 分布の比較

<!--en:41aa6e51-->
### レーニイダイバージェンスとプライバシー

<!--en:e2855dc6-->
乱択アルゴリズムを繰り返し実行したとき、1 件の訓練レコードについてどれだけの情報が漏洩するだろうか。差分プライバシー（differential privacy）は、その 1 件のレコードだけが異なる 2 つのデータセット上でのアルゴリズムの出力分布間のダイバージェンスとして、この情報漏洩を定式化する。差分プライバシー付き SGD は各勾配ステップでノイズを注入し何千回もの反復を実行するため、独立なステップにわたってきれいに合成できるダイバージェンスが必要になる。次数 $\alpha$ のレーニイダイバージェンス（Rényi divergence）:cite:`Renyi.1961` は次の式で定義される。

<!--en:09037f16-->
このダイバージェンスはその性質を備えている。f ダイバージェンスが和の内側で凸関数を適用するのに対し、ここでは対数が和の*外側*にあることがポイントである。独立な座標にわたって和が因数分解されるため、対数によってその積が和へと変換され、結果として $D_\alpha$ は KL ダイバージェンスと同様に、また $\chi^2$ ダイバージェンスや全変動距離（total variation distance）とは異なり、独立なステップにわたってきれいに加法的に合成される。レーニイ差分プライバシー（Rényi differential privacy）:cite:`Mironov.2017` は、ノイズを加えた 1 ステップあたりの $D_\alpha$ を上から抑え、その上界を訓練実行全体にわたって足し合わせる。KL ダイバージェンスもステップ間で加算できるが、対数密度比 $\log(p(x)/q(x))$ を平均の意味でしか制御できない。一方、$D_\alpha$ は $\alpha$ に関して単調に増加し、$\alpha \to 1$ での KL ダイバージェンスから $\alpha \to \infty$ での最悪ケースの対数比までを補間するため、より大きな次数 $\alpha$ を選べば、まれに起こる情報漏洩の大きい出力に対しても保護を与えられる。これらの極限と不等式の網羅的な整理は :citet:`Van-Erven.Harremoes.2014` にまとめられており、演習問題 :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`d（:numref:`sec_mdl-divergences-distances`）では $D_\alpha$ と $\alpha$ ダイバージェンスの関係を扱う。

<!--en:61740827-->
### 情報幾何学

<!--en:ca692b58-->
:numref:`sec_mdl-divergences-distances` では簡単なテイラー展開により、$q_{\boldsymbol\theta + \mathbf{h}}$ と $q_{\boldsymbol\theta}$ の間の任意の滑らかな f ダイバージェンスが、2 次の精度でフィッシャー情報行列（Fisher information matrix）に関する二次形式 $\tfrac12 f''(1)\, \mathbf{h}^\top \mathbf{I}(\boldsymbol\theta)\, \mathbf{h}$ に帰着することを見た。これは一見すると代数的な偶然のように思えるかもしれないが、チェンツォフの定理（Chentsov's theorem）は、それ以外の結果にはなり得なかったことを示している。正定値二次形式はあらゆる微小ステップ $\mathbf{h}$ に長さを割り当て、分布の空間をリーマン多様体（Riemannian manifold）へと変える（:numref:`sec_mdl-what-is-a-divergence` の距離が 2 つの分布間の大域的な距離を測るのに対し、こちらは局所的なステップを測る）。有限集合上の分布について、Chentsov は、生のデータを十分統計量（sufficient statistic）に置き換えたときに不変に保たれるリーマン計量（Riemannian metric）は、正の定数倍を除いてフィッシャー情報量*のみ*であることを証明した :cite:`Cencov.1982`。あらゆる f ダイバージェンスはこの不変性を共有している。$\boldsymbol\theta$ を知らなくても十分統計量からデータを再サンプリングできるため、:numref:`subsec_mdl-f-div-data-processing` のデータ処理不等式（data-processing inequality）が両方向に適用され、等号が強制されるからである。したがって、任意の滑らかな f ダイバージェンスの局所二次近似は、フィッシャー情報量の定数倍でなければならない。この幾何構造のもとで長さを測った勾配ステップを踏むと自然勾配（natural gradient）が得られ、K-FAC はこれを層あたり 1 つのクロネッカー積（Kronecker product）で近似する（:numref:`subsec_mdl-preconditioning-ladder`）。また、:numref:`sec_ppo` の KL 信頼領域（trust region）も、2 次の精度ではまさにこの計量における球である。:citet:`Amari.2016` はこの幾何学的視点を全面的に展開し、最尤法と EM アルゴリズム（:numref:`sec_mdl-latent-em-elbo`）の両ステップがいかに直交射影（orthogonal projection）として理解できるかを示している。

<!--en:e66547a1-->
### シンクホーンダイバージェンスとスライスワッサースタイン距離

<!--en:ae9603d6-->
最適輸送距離を高次元ミニバッチ上の実用的な損失へと変える際には、計算コストと統計的バイアス（statistical bias）という 2 つの障壁に直面する。エントロピー正則化（entropic regularization、:numref:`subsec_mdl-sinkhorn`）は計算上の問題を解決するものの、微妙なバイアスを持ち込む。ある分布からそれ自身への正則化輸送コストが厳密に非ゼロとなるため、これを最小化するよう訓練されたモデルはデータよりも狭い分布へと収縮しやすいのである。演習問題 :exref:`ex_mdl-divergences-distances-sinkhorn-divergences`（:numref:`sec_mdl-test-functions-transport`）のシンクホーンダイバージェンス（Sinkhorn divergence）は、2 つの自己輸送コストの平均を差し引くことでこの問題を解消する。その演習問題の先には、これを信頼できる訓練目的関数たらしめる定理がある。有界領域上でユークリッド距離またはその二乗を基礎コスト（ground cost）として用いると、シンクホーンダイバージェンスは非負であり、2 つの分布が一致するとき、かつそのときに限り 0 となり、さらにモデルがデータへと分布収束するとき、かつそのときに限り 0 に収束する :cite:`Feydy.Sejourne.Vialard.ea.2019`。

<!--en:04881087-->
*スライス*ワッサースタイン距離（sliced Wasserstein distance）は、まったく異なる道筋をとる。2 つの高次元サンプルを 1 本の 1 次元直線上に射影すれば、最適輸送は射影された値を単にソートするだけの操作に帰着する（:numref:`sec_mdl-optimal-transport`）。この 1 次元のコストを多数のランダムな方向にわたって平均したものが、スライスワッサースタイン距離である :cite:`Bonneel.Rabin.Peyre.ea.2015`。ただし高次元における難点は、2 つの分布が実際に異なっている部分空間にランダムな方向が重なることがめったにない点にある。これを回避するため、:citet:`Deshpande.Zhang.Schwing.2018` は識別器（discriminator）を訓練して有用な特徴量を抽出し、生のピクセルではなくその学習された特徴空間上でスライスワッサースタイン距離を計算している。

<!--en:1f2dd48b-->
### Wasserstein GAN におけるリプシッツクリティック

<!--en:295af25a-->
:numref:`sec_mdl-optimal-transport` のカントロヴィチ・ルビンシュタイン双対（Kantorovich–Rubinstein dual）はリプシッツ定数 1 の関数（1-Lipschitz function）全体にわたって最適化を行うが、通常のニューラルネットワークはそのままではこの制約を満たさない。:numref:`sec_gan_objectives` ではこれを課す 3 つの主要な手法（重みクリッピング、スペクトル正規化、勾配ペナルティ）を比較し、:numref:`sec_gan_convergence` では勾配ペナルティを詳しく調べている。とりわけ :citet:`Gulrajani.Ahmed.Arjovsky.ea.2017` のペナルティは、実サンプルと生成サンプルの間を補間したランダムな点においてクリティックの勾配ノルムを 1 に近づけるものであり、これは対応づけられた点を結ぶ輸送半直線（transport ray）に沿って最適クリティックが傾き 1 をもつという幾何学的事実を反映している（演習問題 :exref:`ex_mdl-test-functions-transport-optimal-critics`d、:numref:`sec_mdl-test-functions-transport`）。

<!--en:7f1cebf5-->
### スタイン変分勾配降下法

<!--en:bb75a17e-->
分布 $Q$ がそのスコア $s_Q$ を通じてのみ既知である場合、そこからサンプルを抽出するには通常、長い確率的ランダムウォークを実行する必要がある（:numref:`sec_mdl-scores`）。:numref:`sec_mdl-ksd` では、カーネルスタイン不一致度（kernel Stein discrepancy）が、サンプル分布 $P$ がそのようなモデル $Q$ からどれだけ離れているかを測ることを見た。スタイン変分勾配降下法（Stein variational gradient descent）:cite:`Liu.Wang.2016` は、この不一致度を、粒子の集合を $Q$ へ向かって能動的に輸送する決定論的アルゴリズムへと変換する。分布 $P$ に従う任意の初期点群から出発し、各粒子 $y$ は次の速度場（velocity field）に沿って微小ステップで移動する。

<!--en:c2cab908-->
期待値の内側にある 2 つの項は相補的な役割を果たす。第 1 項は $y$ の近傍におけるスコアのカーネル重み付き平均であり、確率密度のより高い領域へと粒子を上り方向に導く。第 2 項は $y$ を近傍の粒子から押し離す斥力であり、ランジュバン動力学におけるランダムノイズの役割を引き継いでいる。これがなければ、すべての粒子が単一のモードへと潰れてしまうだろう。:numref:`sec_mdl-ksd` の再生核ヒルベルト空間の単位球へと正規化すると、この速度場は $D_{\textrm{KL}}(P\|Q)$ に対する最急降下方向となり、KL ダイバージェンスを $\|v\| = \mathrm{KSD}(P, Q)$ の速さで減少させる。驚くべきことに、同じスタイン作用素（Stein operator）が適合度検定（goodness-of-fit test）と粒子輸送則の両方の役割を果たしているのである。依然として多数の微小ステップは必要であるものの、更新は決定論的であり粒子どうしが協調的に広がるため、元論文における小規模ニューラルネットワークの重みのように、わずか数十個の粒子で事後分布を近似する用途にこの手法は適している。

<!--en:0e8ee723-->
## 相互情報量

<!--en:c8b9966a-->
### 容量に迫る符号

<!--en:5b7fc8fe-->
シャノンの通信路符号化定理（Shannon's channel coding theorem、:numref:`subsec_mdl-channel-capacity`）は、通信路容量未満のあらゆるレートで高信頼な通信が可能であることを保証するが、Shannon の証明はデコードに指数時間を要するランダム符号帳（random codebook）に依拠している。その存在定理と実用的なハードウェアとの隔たりを埋めるには数十年を要した。疎な因子グラフ（factor graph）上でメッセージを伝播させることで効率的にデコードされる低密度パリティ検査（low-density parity-check、LDPC）符号は、実用上きわめて容量に近い性能を達成する :cite:`Richardson.Urbanke.2008`。ポーラ符号（polar code）:cite:`Arikan.2009` はさらに一歩進み、対称 2 元入力通信路において容量を達成することが証明された初の明示的な符号族であり、符号化とデコードの双方が $O(n \log n)$ 回の演算で動作する（$n$ はブロック長）。現代の 5G 無線規格では、これら両方の符号族が実用化されている。

<!--en:4cf7227a-->
### 学習型非可逆圧縮

<!--en:3a5bd207-->
:numref:`subsec_mdl-rate-distortion` では、ガウス情報源に対する厳密なレート歪み（rate–distortion）曲線を導出した。自然写真に対しては閉じた形のレート歪み関数が知られていないため、学習型の画像コーデックはいずれも、その未知のフロンティアより上方に位置する実用的な動作点を表すことになる。:citet:`Balle.Laparra.Simoncelli.2017` は、エンコーダ、デコーダ、および量子化された潜在表現上の学習された確率分布を定義する*エントロピーモデル*という 3 つの構成要素を同時に訓練する。丸め操作はほとんど至るところで勾配が 0 になるため、訓練中は量子化を加法的な一様ノイズに置き換える。その目的関数は $R + \lambda D$ を最小化する。ここで、レート $R$ はエントロピーモデルのもとでの潜在変数の算術符号化長であり（:numref:`sec_mdl-coding-compression`）、$D$ はデコードされた画像の再構成歪みである。$\lambda$ を変化させると傾き $-\lambda$ の点が選ばれ、コーデックのレート歪み曲線が描かれる。後続研究において :citet:`Balle.Minnen.Singh.ea.2018` は、*超事前分布*（hyperprior）と呼ばれる第 2 のより小さな潜在変数でエントロピーモデルを条件づけることにより、画像の複雑さの空間的な変動を捉え、同じ視覚品質のもとでビットレートを大幅に引き下げている。

<!--en:4071a501-->
## 相互情報量の推定

<!--en:9164835c-->
### 上界

<!--en:d22a5b6d-->
:numref:`sec_mdl-mi-estimation` で導出した変分推定量は、いずれも相互情報量の*下界*である。これは共有情報量を最大化したいときには有用だが、本来破棄すべき機微属性や保護属性（protected attribute）に関する情報を学習表現が*ほとんど*保持していないことを保証するには使えない。真の上界を得るには、確率的エンコーダが提供する条件付き密度 $p(z \mid x)$ を利用できる必要がある。変分情報ボトルネック（variational information bottleneck、:numref:`subsec_mdl-variational-bottleneck`）はそのような上界の 1 つであり、固定された周辺分布の近似 $r(z)$ を用いる。対照対数比上界（contrastive log-ratio upper bound、CLUB）:cite:`Cheng.Hao.Dai.ea.2020` は $r(z)$ の必要性を完全に取り除く手法であり、ミニバッチ内の対応ペアにわたる $\log p(z \mid x)$ の平均から、非対応ペアにわたる平均を差し引く。$p(z \mid x)$ が真のエンコーダ密度であるとき、CLUB は期待値の意味で上界となる。しかし、表現と保護属性の間には解析的な条件付き密度が存在しないため、代わりにニューラルネットワークを当てはめなければならない。その場合、当てはめた条件付き分布が周辺分布の積よりも真の同時分布（joint distribution）に対して KL ダイバージェンスの意味で近い場合に限り上界が成り立つが、この条件はそもそも未知であった当の密度なしには検証できない。

<!--en:5c8575b1-->
### 下界のバイアスと分散

<!--en:c8d02af2-->
厳密に最適なクリティックが与えられた場合でさえ、変分下界は鋭いトレードオフに直面する。NWJ は平均的には不偏であるが高い分散に悩まされるのに対し、InfoNCE は低分散である代わりに $\log N$ で頭打ちになる（:numref:`subsec_mdl-exact-critics`）。:citet:`Poole.Oord.Alemi.ea.2019` は、これら 2 つの下界の間を滑らかに補間する単一の混合パラメータを導入し、このトレードオフを連続的に調整できるようにしている。さらに深刻なことに、:citet:`Song.Ermon.2020b` は、分散の爆発が特定の目的関数に固有の副産物ではなく本質的な現象であることを証明した。すなわち、NWJ と Donsker–Varadhan の両下界において、たとえ厳密なクリティックを用いても、推定量の分散は真の相互情報量に対して指数関数的に増大する。同著者らが提案する対策はクリティックの密度比をクリップするものであり、統計的バイアスが生じる代償として分散を抑え込む。要するに、利用可能なあらゆる対策は、誤差を解消するのではなく分散とバイアスの間で移し替えているにすぎない。

<!--en:6ab2338b-->
実データでは正解の相互情報量がわかっていることはまれであるため、同著者らは既知の答えを必要としない自己無撞着性による妥当性確認（self-consistency sanity check）を提案している。信頼できる推定量であれば、一方の変数にその決定論的関数を付加しても推定値は変化せず、単一の対 $(X, Y)$ をその独立な 2 つのコピーに置き換えれば推定値は 2 倍になるはずである。画像データセット上では、同著者らが評価した標準的な推定量はいずれもこれらの基本的なテストに不合格となっている。実践上の経験則として、数 nat 規模の相互情報量の推定値が報告されている場合には必ずバッチサイズの対数と比較し、手元のデータでこれらの整合性チェックを通過することを確認したうえで初めて推定量を信頼すべきである。

<!--en:161096d4-->
### ノイズ付きネットワークとボトルネックの視点

<!--en:9cf578d6-->
:numref:`subsec_mdl-compression-debate` の圧縮論争は、:citet:`Tishby.Zaslavsky.2015` がニューラルネットワークの各層を情報ボトルネックのフロンティア（:numref:`sec_mdl-information-bottleneck`）上にプロットすることを提案したことに端を発する。このフロンティアは、達成可能な最大の $I(Y; Z)$ を与えられた $I(X; Z)$ ごとに追跡するものである。しかしその項で見たとおり、この着想は数学的な難点に突き当たる。連続入力を受け取る決定論的ネットワークでは $I(X; Z)$ が形式的に無限大となり、追跡すべき有限の圧縮曲線が存在しなくなるのである。:citet:`Goldfeld.Berg.Greenewald.ea.2019` は、隠れ層の活性化値に小さなガウスノイズを注入して $I(X; Z)$ を有限にすることで、この問いを良設定（well-posed）にした。このノイズ付きモデルのもとでは実際に訓練中に $I(X; Z)$ が減少し得るが、著者らはその減少の要因が、同一クラスの隠れ表現どうしが互いに引き寄せられる幾何学的なクラスタリングにあることを突き止めている。以前から決定論的ネットワークに対するビン分割ベースの相互情報量推定が捉えていた正体も、まさにこのクラスタリングにほかならないと同著者らは論じている。

<!--en:e8db4f80-->
## 情報理論全般の参考資料

<!--en:f3e47eb0-->
**書籍**

<!--en:5d2dbe48-->
- [Elements of Information Theory — Cover & Thomas](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X)：エントロピー、KL ダイバージェンス（同書では相対エントロピーと呼ばれる）、相互情報量、およびそれらの連鎖律を第一原理から展開するとともに、情報源符号化（source coding）、通信路容量、レート歪み理論を扱う大学院レベルの古典的標準書である。:numref:`sec_mdl-information_theory`、:numref:`sec_mdl-coding-compression`、:numref:`sec_mdl-mutual-information` に対応する。

<!--en:1c9c53df-->
- [Information Theory, Inference, and Learning Algorithms — David MacKay](https://www.inference.org.uk/mackay/itila/)：オンラインで無料公開されている書籍である。全編を通じて情報理論と機械学習を織り交ぜて論じており、:numref:`sec_mdl-coding-compression` に対応する記号符号（symbol code）とストリーム符号（算術符号化を含む）の章を備える。

<!--en:1fa9ddc6-->
- [Information Theory: From Coding to Learning — Polyanskiy & Wu](https://www.cambridge.org/9781108832908)：統計学と機械学習に軸足を置いた現代的な大学院向けテキストであり、f ダイバージェンス、変分表現、データ処理の扱いは :numref:`sec_mdl-divergences-distances` および :numref:`sec_mdl-mutual-information` に対応する。同書の母体となった [MIT 6.441 の講義ノート](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/)も無料公開されている。

<!--en:4940a6ad-->
- [Computational Optimal Transport — Peyré & Cuturi](https://optimaltransport.github.io/book/)：データサイエンス向けにワッサースタイン距離、エントロピー正則化、シンクホーン反復を解説しており、:numref:`sec_mdl-test-functions-transport` の最適輸送パートを補完する。全文は [arXiv](https://arxiv.org/abs/1803.00567) でも公開されている。

<!--en:ae356d1b-->
- [The Minimum Description Length Principle — Peter Grünwald](https://mitpress.mit.edu/9780262072816/the-minimum-description-length-principle/)：2 部符号と普遍符号を通じて統計学習をデータ圧縮として展開し、:numref:`subsec_mdl-mdl` の内容を掘り下げている。中心的な考え方は[無料公開されているチュートリアル](https://arxiv.org/abs/math/0406077)でも解説されている。

<!--en:b4a79945-->
**講義・チュートリアル・ソフトウェア**

<!--en:912d6a39-->
- [Stanford EE376A Information Theory — Tsachy Weissman](https://web.stanford.edu/class/ee376a/)：エントロピー、接頭符号（prefix code）、典型集合（typical set）を扱うわかりやすい講義であり、学生が筆記した公開[講義ノート](https://web.stanford.edu/class/ee376a/files/scribes/lecture_notes.pdf)が提供されている。:numref:`sec_mdl-information_theory` および :numref:`sec_mdl-coding-compression` に対応する。

<!--en:268070ac-->
- [Visual Information Theory — Christopher Olah](https://colah.github.io/posts/2015-09-Visual-Information/)：符号語長から出発してエントロピー、交差エントロピー、相互情報量へと積み上げていく直観的かつ視覚的な入門記事であり、:numref:`sec_mdl-information_theory`、:numref:`sec_mdl-coding-compression`、:numref:`sec_mdl-mutual-information` に対応する。

<!--en:ce625855-->
- [From GAN to WGAN — Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/)：生成モデルにおける KL、イェンセン・シャノン、ワッサースタインの各損失を比較する明快な解説であり、:numref:`sec_mdl-divergences-distances` および :numref:`sec_mdl-test-functions-transport` と並行した内容になっている。

<!--en:bc259aa4-->
- [POT: Python Optimal Transport — Flamary et al.](https://pythonot.github.io/)：厳密な最適輸送とエントロピー正則化付きシンクホーン反復の両方のソルバを提供するオープンソースの Python ライブラリであり、手元のデータセット上で :numref:`sec_mdl-optimal-transport` の距離を計算するのに役立つ。

<!--en:13e6eab6-->
- [Generative Modeling by Estimating Gradients of the Data Distribution — Yang Song](https://yang-song.net/blog/2021/score/)：スコア関数がなぜ正規化定数（normalizing constant）を回避できるのか、またスコアマッチングがいかにしてスコア関数をデータに当てはめるのか（:numref:`sec_mdl-scores` の範囲）を平易に解説したうえで、サンプリングと拡散モデル（diffusion model）へと進むチュートリアルである。

<!--en:d63f4e40-->
**基礎論文**

<!--en:ab57dafd-->
- [Language Modeling Is Compression — Delétang et al.](https://arxiv.org/abs/2309.10668)：算術符号器と言語モデルの次トークン確率を組み合わせる研究であり、まさに :numref:`subsec_mdl-lm-compressor` で実演した着想そのものを扱っている。

<!--en:6a1067d6-->
- [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization — Nowozin et al.](https://arxiv.org/abs/1606.00709)：:numref:`sec_mdl-f-gan-dual` で導出したとおり、任意の f ダイバージェンスがそのフェンシェル共役（Fenchel conjugate）による下界を通じて敵対的訓練（adversarial training）の目的関数を与えることを示している。

<!--en:ebed8555-->
- [A Kernel Two-Sample Test — Gretton et al.](https://jmlr.org/papers/v13/gretton12a.html)：:numref:`sec_mdl-ipm-mmd` で扱った最大平均値差異（maximum mean discrepancy）、その不偏 2 サンプル推定量、およびそれに基づく仮説検定を導入した論文である。

<!--en:ac6b4748-->
- [Wasserstein GAN — Arjovsky et al.](https://arxiv.org/abs/1701.07875)：サポートが重ならない場合でも勾配が有用な情報を保つよう、イェンセン・シャノン・ダイバージェンスを 1 次ワッサースタイン距離に置き換えた研究であり、:numref:`sec_mdl-optimal-transport` に対応する。

<!--en:d6f3fa17-->
- [Estimation of Non-Normalized Statistical Models by Score Matching — Aapo Hyvärinen](https://jmlr.org/papers/v6/hyvarinen05a.html)：対数密度の勾配をデータの勾配に一致させ、未知のデータスコアを部分積分（integration by parts）によって消去することで非正規化モデルを当てはめる手法を提案しており、これらが :numref:`sec_mdl-scores` の 2 つの重要なステップとなっている。

<!--en:58989f40-->
- [A Kernelized Stein Discrepancy for Goodness-of-fit Tests — Liu, Lee & Jordan](https://proceedings.mlr.press/v48/liub16.html)：:numref:`sec_mdl-ksd` で示したように、ペアごとのカーネル評価を用いて、スコアのみを通じて知られるモデルからサンプルが得られたかどうかを検定する。

<!--en:9c21094d-->
- [The Information Bottleneck Method — Tishby, Pereira & Bialek](https://arxiv.org/abs/physics/0004057)：$X$ を圧縮しつつ $Y$ に関する情報を保持するという問題を定式化しており、:numref:`sec_mdl-information-bottleneck` で学んだ枠組みの原典である。

<!--en:a3fdafab-->
- [Estimating Mutual Information — Kraskov, Stögbauer & Grassberger](https://arxiv.org/abs/cond-mat/0305641)：最近傍距離から連続変数の相互情報量を推定する :numref:`subsec_mdl-ksg` の推定量を提案した論文であり、scikit-learn の [`mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) もこれに基づいている。

<!--en:9fed735f-->
- [On Variational Bounds of Mutual Information — Poole et al.](https://arxiv.org/abs/1905.06922)：Barber–Agakov、Donsker–Varadhan、NWJ、InfoNCE の各下界を単一の枠組みで統一し、それらのバイアス・分散のトレードオフを解析しており、:numref:`sec_mdl-mi-variational-bounds` に対応する。

<!--en:e06d1664-->
- [Representation Learning with Contrastive Predictive Coding — van den Oord et al.](https://arxiv.org/abs/1807.03748)：:numref:`sec_mdl-infonce` で展開したとおり、自己教師あり表現学習（self-supervised representation learning）に向けて InfoNCE 損失を相互情報量の下界として導入した論文である。

<!--en:e67f7d89-->
[ディスカッション](https://d2l.discourse.group/t/36867)
