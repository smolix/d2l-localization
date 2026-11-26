<!--en:168ab129-->
# Notation

<!--en:dbaf2a9a-->
Eine klare, einheitliche Notation erleichtert es, mathematische Ideen mit Code zu verbinden. Das gilt für alle Themen dieses Buches, von linearer Algebra, Analysis, Wahrscheinlichkeitsrechnung und Optimierung bis hin zu modernen neuronalen Architekturen. Im ganzen Buch verwenden wir dieselben Schriftkonventionen. Kursive Kleinbuchstaben ($x$) bezeichnen Skalare, fette Kleinbuchstaben ($\mathbf{x}$) Vektoren und fette Großbuchstaben ($\mathbf{X}$) Matrizen. Serifenlose Großbuchstaben ($\mathsf{X}$) stehen für Tensoren höherer Ordnung, kalligrafische Buchstaben ($\mathcal{X}$) für Mengen und kursive Großbuchstaben ($X$) für Zufallsvariablen.

<!--en:3a7674bd-->
In den folgenden Übersichten kennzeichnet der unbestimmte Artikel *„ein“ oder „eine“* eine allgemeine Schreibkonvention. So bedeutet „$x$: ein Skalar“, dass kursive Kleinbuchstaben im Allgemeinen Skalare bezeichnen. Der bestimmte Artikel *„der“, „die“ oder „das“* kennzeichnet dagegen ein Symbol mit festgelegter Bedeutung, etwa „$\mathbb{R}$: die Menge der reellen Zahlen“.

<!--en:38fc2576-->
Das Alphabet ist endlich. Einige gebräuchliche Buchstaben haben daher je nach Fachgebiet unterschiedliche Bedeutungen. Die Schreibweise und der Kontext machen in jedem Fall deutlich, was gemeint ist. So steht bei der Sigmoidfunktion $\sigma(\cdot)$ das Argument in Klammern. Eine Standardabweichung $\sigma_X$ trägt eine Zufallsvariable als Index. Ein Singulärwert $\sigma_i(\mathbf{X})$ hat einen Index und bezieht sich auf eine Matrix. Beim Rauschplan (engl. noise schedule) $\sigma_t$ eines Diffusionsmodells bezeichnet der Index die Zeit $t$. Auf mögliche Verwechslungen weisen wir im Folgenden ausdrücklich hin. Kapitel, die viele eigene Fachbegriffe einführen, beginnen zudem mit einer eigenen Notationstabelle. Das gilt für Optimierung (:numref:`tab_mdl-opt-notation`), Informationstheorie (:numref:`tab_mdl-it-notation`), Dynamik (:numref:`tab_mdl-dyn-notation`), bestärkendes Lernen (:numref:`tab_rl_notation`), Generative Adversarial Networks (:numref:`tab_gan_notation`) und Diffusionsmodelle (:numref:`tab_diffusion-notation`). Die letzten Abschnitte dieses Kapitels fassen die wichtigsten Symbole für einige weiterführende Themen zusammen.

<!--en:b543a219-->
## Numerische Objekte

<!--en:1553b069-->
* $x$: ein Skalar

<!--en:6f7adca9-->
* $\mathbf{x}$: ein Vektor (standardmäßig ein Spaltenvektor)

<!--en:9039f3bc-->
* $(x_1, \ldots, x_n)$: der Spaltenvektor mit den Einträgen $x_1, \ldots, x_n$ in einer Zeile geschrieben. Für Matrixzeilen verwenden wir im Fließtext dieselbe Schreibweise, etwa „die Zeilen $(0, 1, 2)$ und $(3, 4, 5)$“

<!--en:c4155746-->
* $\mathbf{X}$: eine Matrix

<!--en:5382b6c0-->
* $\mathsf{X}$: ein Tensor mit mehr als zwei Achsen

<!--en:a6b889b3-->
* $\mathbf{I}$: die Einheitsmatrix, deren Dimension sich aus dem Kontext ergibt (eine quadratische Matrix mit $1$ auf der Hauptdiagonalen und $0$ an allen übrigen Stellen)

<!--en:8a2976ff-->
* $\mathbf{0}$, $\mathbf{1}$: der Nullvektor und der Einsvektor, deren Dimension sich aus dem Kontext ergibt

<!--en:e6f5ea3a-->
* $x_i$: das $i^\textrm{th}$ Element des Vektors $\mathbf{x}$

<!--en:fba12516-->
* $x_{ij}$, $[\mathbf{X}]_{i,j}$: das Element der Matrix $\mathbf{X}$ in Zeile $i$ und Spalte $j$

<!--en:fcd1b3c7-->
## Mengenlehre

<!--en:66d52d87-->
* $\mathcal{X}$: eine Menge

<!--en:53f0b6d2-->
* $\mathbb{R}$: die Menge der reellen Zahlen

<!--en:0bb19b3c-->
* $\mathbb{R}^n$: die Menge der $n$-dimensionalen reellen Vektoren

<!--en:a77cb2f0-->
* $\mathbb{R}^{a\times b}$: die Menge der reellen Matrizen mit $a$ Zeilen und $b$ Spalten

<!--en:2af64644-->
* $|\mathcal{X}|$: die Mächtigkeit (Anzahl der Elemente) der Menge $\mathcal{X}$

<!--en:add2aa69-->
* $\mathcal{A}\cup\mathcal{B}$: die Vereinigung der Mengen $\mathcal{A}$ und $\mathcal{B}$

<!--en:288bba92-->
* $\mathcal{A}\cap\mathcal{B}$: der Schnitt der Mengen $\mathcal{A}$ und $\mathcal{B}$

<!--en:b3f689c6-->
* $\mathcal{A}\setminus\mathcal{B}$: die Differenzmenge (die Elemente von $\mathcal{A}$, die nicht zu $\mathcal{B}$ gehören)

<!--en:c7827ae0-->
## Funktionen und Operatoren

<!--en:fb5d3c14-->
* $f(\cdot)$: eine Funktion

<!--en:e82af5d0-->
* $\log(\cdot)$: der natürliche Logarithmus (zur Basis $e$)

<!--en:d1fa7655-->
* $\log_2(\cdot)$: der Logarithmus zur Basis $2$

<!--en:102e2f62-->
* $\exp(\cdot)$: die Exponentialfunktion

<!--en:75fcc17d-->
* $\sigma(\cdot)$: die logistische Sigmoidfunktion, $\sigma(x) = 1/(1 + e^{-x})$

<!--en:3108b61c-->
* $\mathrm{softmax}(\mathbf{o})$: der Vektor der Softmax-Funktion mit den Einträgen $\exp(o_i) / \sum_j \exp(o_j)$

<!--en:5be09752-->
* $\mathbf{1}(\cdot)$: die Indikatorfunktion. Ihr Wert ist $1$, wenn ihr boolesches Argument wahr ist, andernfalls $0$. Am Argument erkennt man den Unterschied zum Einsvektor $\mathbf{1}$

<!--en:734d3617-->
* $\operatorname{argmax}_x f(x)$, $\operatorname{argmin}_x f(x)$: ein Wert von $x$, der $f(x)$ maximiert beziehungsweise minimiert

<!--en:83e60b16-->
* $\sum$: die Summe über eine Menge von Elementen

<!--en:06207a7e-->
* $\prod$: das Produkt über eine Menge von Elementen

<!--en:dcdf5e98-->
* $O(\cdot)$: eine asymptotische obere Schranke bis auf einen konstanten Faktor, etwa bei einem Aufwand von $O(n^2)$ Operationen

<!--en:6fe9b800-->
* $\propto$: proportional zu (beide Seiten unterscheiden sich um einen konstanten Faktor, der von der betrachteten Variablen unabhängig ist)

<!--en:1a4dc64e-->
* $\stackrel{\textrm{def}}{=}$: Gleichheit aufgrund der Definition des Symbols auf der linken Seite

<!--en:24e95e8f-->
## Lineare Algebra

<!--en:2b69e940-->
* $(\cdot)^\top$: die Transponierte eines Vektors oder einer Matrix

<!--en:8e257c98-->
* $\mathbf{X}^{-1}$: die Inverse der quadratischen Matrix $\mathbf{X}$

<!--en:f758a663-->
* $\mathbf{X}^{+}$: die Moore-Penrose-Pseudoinverse der Matrix $\mathbf{X}$

<!--en:8069dc7a-->
* $\langle \mathbf{x}, \mathbf{y} \rangle$: das innere Produkt (Skalarprodukt) der Vektoren $\mathbf{x}$ und $\mathbf{y}$, auch geschrieben als $\mathbf{x}^\top \mathbf{y}$

<!--en:cc0f7397-->
* $\odot$: das Hadamard-Produkt (elementweises Produkt)

<!--en:2e0e7e5c-->
* $[\cdot, \cdot]$: die Verkettung von Vektoren oder Matrizen

<!--en:f5af0971-->
* $\|\cdot\|_p$: die $\ell_p$-Norm eines Vektors

<!--en:b1629f49-->
* $\|\cdot\|$: die euklidische Norm ($\ell_2$-Norm)

<!--en:b6102a93-->
* $\|\mathbf{X}\|_2$: die Spektralnorm (Operatornorm) der Matrix $\mathbf{X}$, also ihr größter Singulärwert

<!--en:59374292-->
* $\|\mathbf{X}\|_F$: die Frobeniusnorm der Matrix $\mathbf{X}$, also die $\ell_2$-Norm ihrer Einträge

<!--en:c08cca10-->
* $\|\mathbf{X}\|_*$: die Nuklearnorm (Spurnorm) der Matrix $\mathbf{X}$, also die Summe ihrer Singulärwerte

<!--en:83a161a4-->
* $\operatorname{diag}(\mathbf{x})$: die Diagonalmatrix mit den Einträgen des Vektors $\mathbf{x}$ auf der Hauptdiagonalen

<!--en:67f2c341-->
* $\operatorname{rank}\mathbf{X}$, $\operatorname{tr}\mathbf{X}$, $\det\mathbf{X}$: der Rang, die Spur und die Determinante der Matrix $\mathbf{X}$

<!--en:02170e28-->
* $\lambda_i$, $\lambda_{\max}(\mathbf{X})$: ein Eigenwert einer quadratischen Matrix und der größte Eigenwert einer symmetrischen Matrix $\mathbf{X}$

<!--en:e198b385-->
* $\sigma_i(\mathbf{X})$: der $i^\textrm{th}$ größte Singulärwert der Matrix $\mathbf{X}$ (nicht zu verwechseln mit der Sigmoidfunktion $\sigma(\cdot)$)

<!--en:26003fb6-->
* $\kappa(\mathbf{X})$: die Konditionszahl der Matrix $\mathbf{X}$, also das Verhältnis ihres größten zu ihrem kleinsten von null verschiedenen Singulärwert

<!--en:6381befc-->
* $\rho(\mathbf{X})$: der Spektralradius einer quadratischen Matrix $\mathbf{X}$, also der größte Betrag unter ihren Eigenwerten (nicht zu verwechseln mit dem weiter unten aufgeführten Korrelationskoeffizienten $\rho(X, Y)$)

<!--en:c322f183-->
* $\mathbf{X}\succeq 0$, $\mathbf{X}\succ 0$: die symmetrische Matrix $\mathbf{X}$ ist positiv semidefinit ($\mathbf{X}\succeq 0$) beziehungsweise positiv definit ($\mathbf{X}\succ 0$)

<!--en:60cfb36d-->
## Analysis

<!--en:bdb91049-->
* $\frac{dy}{dx}$, $f'(x)$: die Ableitung von $y$ nach $x$ und die Ableitung von $f$ an der Stelle $x$

<!--en:f952525c-->
* $\frac{\partial y}{\partial x}$: die partielle Ableitung von $y$ nach $x$

<!--en:dcee0dfb-->
* $\nabla_{\mathbf{x}} y$: der Gradient des Skalars $y$ nach dem Vektor $\mathbf{x}$

<!--en:33e03308-->
* $\mathbf{J}$, $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$: die Jacobi-Matrix der vektorwertigen Funktion $\mathbf{f}$ bezüglich $\mathbf{x}$

<!--en:fb49d508-->
* $\nabla^2 f$: die Hesse-Matrix von $f$ mit den partiellen Ableitungen zweiter Ordnung

<!--en:36e0899e-->
* $\nabla \cdot \mathbf{f}$, $\Delta f$: die Divergenz des Vektorfeldes $\mathbf{f}$ und der Laplace-Operator angewandt auf die skalare Funktion $f$

<!--en:8785498d-->
* $\int_a^b f(x) \,dx$: das bestimmte Integral von $f$ in den Grenzen von $a$ bis $b$ bezüglich $x$

<!--en:7704e7ef-->
* $\int f(x) \,dx$: das unbestimmte Integral (die Stammfunktion) von $f$ bezüglich $x$

<!--en:41832806-->
## Wahrscheinlichkeit und Informationstheorie

<!--en:f4e1706c-->
* $X$: eine Zufallsvariable

<!--en:ff5c8c1d-->
* $P$: eine Wahrscheinlichkeitsverteilung

<!--en:7c064ec6-->
* $X \sim P$: die Zufallsvariable $X$ folgt der Verteilung $P$

<!--en:23431f0e-->
* $P(X=x)$: die Wahrscheinlichkeit des Ereignisses, dass die Zufallsvariable $X$ den Wert $x$ annimmt

<!--en:3e1deb21-->
* $P(X \mid Y)$: die bedingte Wahrscheinlichkeitsverteilung von $X$ gegeben $Y$

<!--en:e56da4d5-->
* $p(\cdot)$: eine Wahrscheinlichkeitsdichte (PDF) oder Wahrscheinlichkeitsfunktion (PMF) der Verteilung $P$

<!--en:c24c492c-->
* $\mathbb{E}[X]$: der Erwartungswert der Zufallsvariablen $X$

<!--en:ea850fe5-->
* $\mathbb{E}_{x \sim P}[f(x)]$: der Erwartungswert von $f(x)$, wenn wir $x$ aus der Verteilung $P$ ziehen

<!--en:cf0c9ffa-->
* $X \perp Y$: die Zufallsvariablen $X$ und $Y$ sind statistisch unabhängig

<!--en:a55dcb48-->
* $X \perp Y \mid Z$: die Zufallsvariablen $X$ und $Y$ sind bedingt unabhängig gegeben $Z$

<!--en:3fcde23b-->
* $\sigma$, $\sigma_X$: eine Standardabweichung im Allgemeinen und die Standardabweichung der Zufallsvariablen $X$ im Besonderen

<!--en:ffbcd637-->
* $\textrm{Var}(X)$: die Varianz der Zufallsvariablen $X$ mit dem Wert $\sigma^2_X$

<!--en:b0dc3197-->
* $\textrm{Cov}(X, Y)$: die Kovarianz der Zufallsvariablen $X$ und $Y$

<!--en:c0c96095-->
* $\boldsymbol{\Sigma}$: eine Kovarianzmatrix

<!--en:5956b357-->
* $\rho(X, Y)$: der Pearson-Korrelationskoeffizient zwischen $X$ und $Y$ mit dem Wert $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$

<!--en:c7936b1c-->
* $\mathcal{N}(\mu, \sigma^2)$: die univariate Gauß-Verteilung (Normalverteilung) mit Mittelwert $\mu$ und Varianz $\sigma^2$. Das Symbol $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ bezeichnet ihre multivariate Form

<!--en:72798c23-->
* $H(X)$, $h(X)$: die Shannon-Entropie einer diskreten Zufallsvariablen $X$ und die differentielle Entropie einer stetigen Zufallsvariablen $X$

<!--en:b99715fd-->
* $I(X; Y)$: die gegenseitige Information zwischen den Zufallsvariablen $X$ und $Y$

<!--en:0509fe36-->
* $D_{\textrm{KL}}(P\|Q)$: die Kullback-Leibler-Divergenz (KL-Divergenz, relative Entropie) von der Verteilung $P$ zur Verteilung $Q$

<!--en:1927e1b8-->
## Daten, Modelle und Training

<!--en:a07a11bd-->
* $n$: die Anzahl der Beispiele in einem Datensatz

<!--en:03fa9b0e-->
* $d$: die Anzahl der Eingabemerkmale (Dimension eines Beispiels)

<!--en:16ce3abd-->
* $\mathbf{x}^{(i)}$, $y^{(i)}$: der Merkmalsvektor und das Ziellabel des $i^\textrm{th}$ Beispiels

<!--en:0107d01d-->
* $\mathcal{D}$: ein Datensatz, etwa $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$

<!--en:75138db0-->
* $\mathbf{X} \in \mathbb{R}^{n \times d}$: eine Designmatrix oder Minibatchmatrix mit einem Beispiel pro Zeile

<!--en:7a656ac6-->
* $\mathcal{B}$: ein Minibatch von Beispielen mit der Größe $|\mathcal{B}|$

<!--en:c096f8d5-->
* $\hat{y}$: die Vorhersage eines Modells für die Zielgröße $y$

<!--en:63bf8d44-->
* $\mathbf{w}$, $b$: der Gewichtsvektor und der skalare Bias eines linearen Modells

<!--en:288190d2-->
* $\mathbf{W}^{(\ell)}$, $\mathbf{b}^{(\ell)}$: die Gewichtsmatrix und der Bias-Vektor der Schicht $\ell$, etwa $\mathbf{W}^{(1)}$

<!--en:18bbe889-->
* $\theta$, $\boldsymbol{\theta}$: die Parameter eines Modells (in Fettschrift als $\boldsymbol{\theta}$, wenn wir sie zu einem Vektor zusammenfassen)

<!--en:e3d2f287-->
* $l$, $\ell$: eine Verlustfunktion für ein einzelnes Beispiel

<!--en:c50e10e7-->
* $L$: der über einen Datensatz oder Minibatch gemittelte empirische Verlust

<!--en:bcd6dd8e-->
* $\eta$: die Lernrate (Schrittweite) eines gradientenbasierten Optimierers

<!--en:08a8c7c8-->
* $\lambda$: die Regularisierungsstärke des $\ell_2$-Strafterms für den Gewichtszerfall $\frac{\lambda}{2}\|\mathbf{w}\|^2$

<!--en:77407414-->
* $t$, $T$: ein diskreter oder kontinuierlicher Zeitschritt $t$ und die Gesamtlänge $T$ einer Sequenz

<!--en:6bdec9b0-->
* $\mathbf{x}_t$, $\mathbf{h}_t$: die Eingabe und der versteckte Zustand zum Zeitschritt $t$

<!--en:8eac3302-->
## Attention und Transformer

<!--en:f1492a82-->
Diese Symbole führen wir in :numref:`chap_attention` ein. Wir verwenden sie durchgehend in :numref:`chap_transformers` und in den darauf aufbauenden Kapiteln.

<!--en:e1817408-->
* $\mathbf{q}$, $\mathbf{k}_i$, $\mathbf{v}_i$: ein Abfragevektor sowie der Schlüsselvektor und der Wertvektor des Elements $i$ (:numref:`sec_queries-keys-values`)

<!--en:0dc2f7fe-->
* $a(\mathbf{q}, \mathbf{k})$: eine Bewertungsfunktion für Attention, etwa $\mathbf{q}^\top \mathbf{k} / \sqrt{d}$ bei skalierter Skalarprodukt-Attention

<!--en:80c27aca-->
* $\alpha(\mathbf{q}, \mathbf{k}_i)$: das Attention-Gewicht für Schlüssel $i$. Es ergibt sich aus der Softmax-Funktion über die Scores aller Schlüssel

<!--en:53bf7480-->
* $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$: die Matrizen aus $n$ Abfragen, $m$ Schlüsseln und $m$ Werten mit je einem Vektor pro Zeile. Die skalierte Skalarprodukt-Attention bildet diese Matrizen auf $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})\,\mathbf{V}$ ab (:numref:`sec_attention-scoring-functions`)

<!--en:f54ed6a4-->
* $d$, $d_v$: die Dimension (Breite) der Abfragen und Schlüssel ($d$) sowie die der Werte ($d_v$)

<!--en:3c50e9f1-->
* $H$, $d_h$: die Anzahl der Attention-Köpfe und die Dimension je Kopf, $d_h = d/H$ (:numref:`sec_multihead-attention`)

<!--en:c58b3d4a-->
## Zustandsraummodelle

<!--en:a2a7e4cc-->
Diese Symbole führen wir in :numref:`chap_modern_rnn` ein.

<!--en:5c0964d0-->
* $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, $D$: die Zustandsübergangsmatrix, die Eingabeprojektion, die Abbildung vom Zustand auf die Ausgabe und der Durchgriffsterm (direkte Verbindung) eines Zustandsraummodells (:numref:`subsec_state-space`)

<!--en:b0759425-->
* $N$: die Dimension des latenten Zustands pro Kanal

<!--en:403dc13d-->
* $\lambda_n$: der $n^\textrm{th}$ Diagonaleintrag (Rate) einer diagonalen Zustandsmatrix. Die zugehörige Zustandskomponente klingt pro Schritt um den Faktor $e^{\Delta \lambda_n}$ ab

<!--en:e3625723-->
* $\Delta$, $\Delta_t$: die Schrittweite der Diskretisierung (:numref:`subsec_zoh`) und ihr eingabeabhängiges (selektives) Gegenstück in Mamba (:numref:`sec_mamba`)

<!--en:c55e8f68-->
* $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$: ein matrixwertiger rekurrenter Zustand. Äußere Produkte $\mathbf{v}_t \mathbf{k}_t^\top$ aktualisieren ihn, eine Abfrage $\mathbf{q}_t$ liest ihn aus (:numref:`sec_matrix-state`). Hier bezeichnet $d_k$ die Dimension der Schlüssel und $d_v$ die Dimension der Werte

<!--en:34b5c20e-->
* $\beta_t$: die Stärke des Schreibschritts (Lernrate) der Delta-Regel (:numref:`sec_deltanet`)

<!--en:cbe6efec-->
## Bestärkendes Lernen

<!--en:c96e7fff-->
Diese Symbole verwenden wir in :numref:`chap_reinforcement_learning` und :numref:`chap_deep_rl`. Eine vollständige Liste mit dem Abschnitt, in dem wir jedes Symbol definieren, findet sich in :numref:`tab_rl_notation`.

<!--en:599d1165-->
* $\mathcal{S}$, $\mathcal{A}$: der Zustandsraum und der Aktionsraum

<!--en:001349d1-->
* $s_t$, $a_t$, $r_t$: der Zustand, die Aktion und die skalare Belohnung zum Zeitschritt $t$

<!--en:44054002-->
* $P(s' \mid s, a)$: der Markov-Übergangskern. Er gibt die Wahrscheinlichkeit an, den Zustand $s'$ zu erreichen, wenn man im Zustand $s$ die Aktion $a$ wählt

<!--en:06cd841a-->
* $r(s, a)$: die erwartete unmittelbare Belohnung für die Aktion $a$ im Zustand $s$

<!--en:279ca80d-->
* $\gamma$: der Diskontierungsfaktor, $\gamma \in [0, 1)$

<!--en:21544d8c-->
* $\tau$, $R(\tau)$: eine Trajektorie $(s_0, a_0, r_0, s_1, \ldots)$ und ihr diskontierter Ertrag $R(\tau) = \sum_t \gamma^t r_t$

<!--en:ac408b2c-->
* $\pi(a \mid s)$: eine stochastische Strategie, die die Wahrscheinlichkeit angibt, mit der man Aktion $a$ im Zustand $s$ wählt. Bei einer deterministischen Strategie bezeichnet $\pi(s)$ die gewählte Aktion

<!--en:5f6b338a-->
* $\pi_\theta$, $J(\theta)$: eine parametrisierte Strategie mit Parametern $\theta$ und ihr erwarteter diskontierter Ertrag

<!--en:2e31653a-->
* $V^\pi(s)$, $Q^\pi(s, a)$: die Zustandswertfunktion und die Aktionswertfunktion unter der Strategie $\pi$ (der erwartete Ertrag ab Zustand $s$ beziehungsweise der erwartete Ertrag, wenn man im Zustand $s$ zunächst die Aktion $a$ ausführt)

<!--en:a7e33bab-->
* $V^*$, $Q^*$: die optimale Zustandswertfunktion und die optimale Aktionswertfunktion

<!--en:32e96a75-->
* $A^\pi(s, a)$: die Vorteilsfunktion, $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$

<!--en:dc80fa03-->
* $\delta_t$: der Temporal-Difference-Fehler (TD-Fehler) im Schritt $t$

<!--en:c3a2bb6c-->
* $\epsilon$: die Explorationswahrscheinlichkeit bei der $\epsilon$-greedy Aktionswahl

<!--en:f351aaeb-->
## Generative Adversarial Networks

<!--en:f1166b6a-->
Diese Symbole gehören zu :numref:`chap_gans`. In :numref:`tab_gan_notation` stehen zu jedem Symbol der Abschnitt mit seiner Definition und gebräuchliche alternative Schreibweisen aus der Literatur.

<!--en:4bbfd87f-->
* $p$, $q$, $m$: die wahre Datendichte, die Modelldichte des Generators und ihre gleichgewichtete Mischung $m = \tfrac12(p + q)$

<!--en:823ac510-->
* $x \sim p$, $x' \sim q$: ein reales Beispiel aus der Datenverteilung und ein vom Generator gezogenes synthetisches Sample

<!--en:4d71ce16-->
* $y$: das binäre Label für die Herkunft eines Beispiels ($y = 1$ für reale Daten und $y = 0$ für generierte Samples)

<!--en:e87417c0-->
* $z$, $G$: die latente Rauschvariable und das Generatornetz, das $x' = G(z)$ erzeugt

<!--en:36103bee-->
* $D$: das Logit des Kritikers (oder Diskriminators) für die Echtheit eines Beispiels. Damit ist $\sigma(D(x))$ die geschätzte Wahrscheinlichkeit, dass das Beispiel $x$ echt ist

<!--en:08d42f7f-->
* $V$: die Wertfunktion des Minimax-Spiels mit logarithmischem Verlust

<!--en:e8ec8891-->
* $\rho = p/q$, $\lambda = \log \rho$: das Likelihood-Verhältnis (Dichteverhältnis) und sein Logarithmus

<!--en:20f2df45-->
* $\eta = p/(p + q)$: die Bayes-optimale A-posteriori-Wahrscheinlichkeit dafür, dass ein Beispiel zur Klasse der realen Daten gehört (in :numref:`chap_gans` schreiben wir die Schrittweite des Optimierers als $h$ statt als $\eta$)

<!--en:180c1813-->
* $f$, $D_f$: eine Divergenzfunktion und ihre $f$-Divergenz

<!--en:2d61659e-->
* $\Phi$, $d_{\mathrm{Rp}}$: die relativistische Zielfunktion für paarweise Vergleiche und ihr Wert beim optimalen Kritiker

<!--en:f0ad3ecb-->
* $\gamma$, $R_1$, $R_2$: das Gewicht der Gradientenstrafe und die um null zentrierten Gradientenstrafen, ausgewertet an realen Daten ($R_1$) und an generierten Samples ($R_2$)

<!--en:29b79ea9-->
* $c$: eine Variable zur Konditionierung, etwa ein Klassenlabel oder ein Text-Prompt

<!--en:f23c6307-->
## Diffusionsmodelle

<!--en:03ec2647-->
Diese Symbole gehören zu :numref:`chap_diffusion` und folgen den Konventionen aus :numref:`chap_mdl-dynamics` (:numref:`tab_mdl-dyn-notation`). Im ganzen Buch verwenden wir dieselbe Zeitrichtung. Bei $t = 0$ liegen unverrauschte Daten vor, bei $t = 1$ reines Rauschen. Zur Generierung von Samples integrieren die Verfahren rückwärts von $t = 1$ bis $t = 0$. (Einige Arbeiten zum Flow-Matching verwenden die umgekehrte Konvention, siehe :numref:`sec_diffusion-gaussian-paths`.) Eine Übersicht der Symbole dieses Kapitels und der entsprechenden Schreibweisen in der Literatur bietet :numref:`tab_diffusion-notation`.

<!--en:4467f835-->
* $\mathbf{x}_0$, $\boldsymbol{\epsilon}$: ein unverrauschter Datenpunkt und ein davon unabhängiger Vektor aus standardnormalverteiltem Rauschen, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

<!--en:e2a8f576-->
* $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$: der verrauschte Zustand zur Zeit $t \in [0, 1]$ auf einem gaußschen Wahrscheinlichkeitspfad

<!--en:cafaa6cb-->
* $\alpha_t$, $\sigma_t$: die zeitabhängigen Koeffizienten für Signal und Rauschen entlang des Pfads (zu unterscheiden von einer Lernrate oder der Standardabweichung der Daten)

<!--en:65ffe1b2-->
* $\bar\alpha_t = \alpha_t^2$, $\beta(t)$: der quadrierte Signalkoeffizient des varianzerhaltenden Pfads und seine Rate (:numref:`sec_diffusion-gaussian-paths`)

<!--en:fe3627af-->
* $k$, $T$, $\beta_k$: ein Schritt der DDPM-Kette, die Anzahl der Schritte und die im Schritt $k$ hinzugefügte Varianz (:numref:`sec_diffusion-gaussian-paths`)

<!--en:1806198a-->
* $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$: das logarithmische Signal-Rausch-Verhältnis (Log-SNR). Es fällt monoton von $+\infty$ bei unverrauschten Daten ($t=0$) auf $-\infty$ bei reinem Rauschen ($t=1$)

<!--en:83b101f7-->
* $p_t$, $\nabla \log p_t$: die Randdichte von $\mathbf{x}_t$ zur Zeit $t$ und ihre Score-Funktion

<!--en:871a4b2a-->
* $\hat{\mathbf{x}}_0$, $\hat{\boldsymbol{\epsilon}}$: die von einem Netz zur Entrauschung vorhergesagten A-posteriori-Erwartungswerte $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$ und $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$

<!--en:3d2c85b6-->
* $\mathbf{u}_t$, $\mathbf{u}_{\boldsymbol{\theta}}$, $\hat{\mathbf{u}}$: das Geschwindigkeitsfeld des Wahrscheinlichkeitsflusses, das Samples entlang des Pfads transportiert, ein mit $\boldsymbol{\theta}$ parametrisiertes neuronales Netz, das dieses Feld approximieren lernt, und die vom Netz vorhergesagte Geschwindigkeit

<!--en:1a3753de-->
* $\bar{\mathbf{x}} = \mathbf{x}_t / \alpha_t$, $\bar\sigma = \sigma_t / \alpha_t$: die Koordinaten, in denen jeder Gauß-Pfad die Form $\bar{\mathbf{x}} = \mathbf{x}_0 + \bar\sigma \boldsymbol{\epsilon}$ hat und die Sampling-Verfahren ihre Schritte ausführen (:numref:`sec_diffusion-integration`). Der Querstrich steht hier für die Division durch $\alpha_t$. In $\bar\alpha_t = \alpha_t^2$ bezeichnet er dagegen das Quadrat des Signalkoeffizienten

<!--en:90803d42-->
* $\mathbf{v} = \alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_0$: der Zielvektor bei der $\mathbf{v}$-Vorhersage (:numref:`sec_diffusion-gaussian-paths`)

<!--en:ee4944bd-->
* $\tilde{\mathbf{u}}$: das durch Classifier-Free Guidance gesteuerte Geschwindigkeitsfeld (:numref:`subsec_diffusion-images-guided`)

<!--en:853446e5-->
* $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$, $\mathbf{u}_{\textrm{ref}}$: die Durchschnittsgeschwindigkeit zwischen den Zeitpunkten $t$ und $s$ und die Referenzgeschwindigkeit für die Verlustfunktionen der Destillation auf wenige Schritte (entweder die Vorhersage $\hat{\mathbf{u}}$ eines Lehrermodells oder die bedingte Geschwindigkeit $\boldsymbol{\epsilon} - \mathbf{x}_0$) (:numref:`sec_diffusion-few-step`)

<!--en:0c6982e3-->
* $c$, $\gamma$: ein Signal zur Konditionierung und die Guidance-Skala (:numref:`subsec_diffusion-images-guided`)

<!--en:75498650-->
[Diskussion](https://d2l.discourse.group/t/25)
