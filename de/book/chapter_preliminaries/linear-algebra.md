<!--en:5a7d784a-->
# Lineare Algebra

<!--en:a2ee5897-->
Teilen wir einen Minibatch von Bildern durch $255$, skalieren wir jedes Pixel für sich. Ein Vorhersagemodell muss dagegen Hunderte oder Tausende Eingabemerkmale zu einer Entscheidung verknüpfen. Die *lineare Algebra* ist die mathematische Sprache für solche gewichteten Kombinationen. Mit ihr lässt sich der Vorwärtsdurchlauf einer ganzen Schicht für einen Minibatch in einer kurzen Gleichung ausdrücken. Daraus können wir schon vorab die Form der Ausgabe und die Anzahl der Gleitkommaoperationen ablesen.

<!--en:461edbe0-->
Betrachten wir einen linearen Bildklassifikator, der $28 \times 28$ große Graustufenbilder von Kleidungsstücken (:numref:`chap_classification`) in $10$ Kategorien einteilt. Formen wir jedes Bild zu einem Vektor um, erhalten wir $28 \times 28 = 784$ Pixel. Für einen Minibatch schreiben wir $256$ solcher Bildvektoren als Zeilen in eine Datenmatrix $\mathbf{X} \in \mathbb{R}^{256 \times 784}$. Die $784$ Gewichte für jede der $10$ Kategorien bilden die Spalten von $\mathbf{W} \in \mathbb{R}^{784 \times 10}$. Die $10$ kategorienspezifischen Verschiebungen fassen wir in $\mathbf{b} \in \mathbb{R}^{10}$ zusammen (insgesamt $7{,}850$ Parameter). Für alle $256 \times 10 = 2{,}560$ Kategoriescores des Minibatchs gilt

<!--en:81a6ff34-->
Dabei addieren wir $\mathbf{b}$ mittels Broadcasting zu jeder Zeile (:numref:`fig_la_matmul`).

<!--en:937a7676-->
![Scores eines linearen Klassifikators für einen Minibatch aus 256 Bildern und 10 Kategorien. Zeile $i$ von $\mathbf{X}$ enthält Bild $i$, Spalte $j$ von $\mathbf{W}$ die Gewichte der Kategorie $j$. Beide gehen in den Score $o_{ij}$ ein.](../img/linear-algebra-matmul.svg)

<!--en:d4361331-->
In diesem Abschnitt erarbeiten wir $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ Schritt für Schritt. Von Skalaren, Vektoren, Matrizen und Tensoren gelangen wir über Reduktionen und Skalarprodukte zu Matrixprodukten. Den Abschluss bilden *Normen* und *Eigenwerte*. Eine Übersicht der Schreibweisen bietet :numref:`chap_notation`.

<!--en:2a9f53fc-->
## Skalare, Vektoren, Matrizen und Tensoren

<!--en:c7b8c99c-->
Arrays der Ordnung $0^\textrm{th}$, $1^\textrm{st}$, $2^\textrm{nd}$ und $4^\textrm{th}$ haben wir in :numref:`sec_ndarray` bereits erzeugt und umgeformt. Hier ordnen wir jeder Ordnung (im Code `ndim`) die übliche mathematische Bezeichnung und Schreibweise zu.

<!--en:3899e623-->
### Skalare

<!--en:3968eed2-->
Einzelne Zahlen heißen *Skalare*. Wir bezeichnen sie mit gewöhnlichen kursiven Kleinbuchstaben ($x, y, z$). Die Schreibweise $x \in \mathbb{R}$ bedeutet, dass $x$ eine reelle Zahl ist. Im Code ist ein Skalar ein Tensor der Ordnung $0^\textrm{th}$ mit leerer Form `()`:

<!--en:c5566523-->
### Vektoren

<!--en:051856e6-->
Ein *Vektor* ist ein eindimensionales Array fester Länge aus Skalaren, seinen *Elementen* oder *Komponenten*. Wir bezeichnen Vektoren mit fetten Kleinbuchstaben ($\mathbf{x}, \mathbf{y}, \mathbf{w}$). Die Schreibweise $\mathbf{x} \in \mathbb{R}^n$ bezeichnet einen $n$-dimensionalen reellen Vektor. Standardmäßig verstehen wir darunter einen *Spaltenvektor*:

<!--en:51083aca-->
Im Fließtext schreiben wir ihn als $\mathbf{x} = (x_1, \ldots, x_n)$. Im Code ist ein Vektor ein Tensor der Ordnung $1^\textrm{st}$:

<!--en:ccce12fe-->
In der mathematischen Schreibweise beginnen die Indizes bei 1 ($x_1, \ldots, x_n$), in Python bei 0. Auf die dritte Komponente $x_3$ greifen wir daher mit `x[2]` zu:

<!--en:4a1cb6f9-->
Die *Dimension* $n$ eines Vektors ist die Anzahl seiner Komponenten. In Python liefert sie die eingebaute Funktion `len(x)`. Sie ist auch der einzige Eintrag des 1-Tupels `x.shape`, das die Form angibt:

<!--en:4d0ee512-->
Um die Mehrdeutigkeit des umgangssprachlich verwendeten Wortes „Dimension“ zu vermeiden, bezeichnen wir die Anzahl der Achsen eines Tensors als *Ordnung* (`ndim`) und die Anzahl der Komponenten eines Vektors als *Dimension*.

<!--en:ffb9bb37-->
### Matrizen

<!--en:12b176ce-->
*Matrizen* sind Tensoren der Ordnung $2^\textrm{nd}$. Wir bezeichnen sie mit fetten Großbuchstaben ($\mathbf{A}, \mathbf{X}, \mathbf{W}$). Die Schreibweise $\mathbf{A} \in \mathbb{R}^{m \times n}$ steht für eine Matrix mit $m$ Zeilen und $n$ Spalten (gilt $m = n$, heißt $\mathbf{A}$ *quadratisch*). Der Eintrag $a_{ij}$ steht in Zeile $i$ und Spalte $j$:

<!--en:5c7f3f90-->
Minibatchmatrizen $\mathbf{X} \in \mathbb{R}^{n \times d}$ aus tabellarischen Daten oder aus zu Vektoren umgeformten Beispielen enthalten ein Beispiel pro Zeile und ein Merkmal pro Spalte. Wir erzeugen eine $2 \times 3$-Matrix `A`:

<!--en:fd02dfe6-->
Wenn wir die Zeilen und Spalten einer Matrix vertauschen, erhalten wir ihre *Transponierte* $\mathbf{A}^\top \in \mathbb{R}^{n \times m}$. Dabei gilt $[\mathbf{A}^\top]_{ij} = a_{ji}$. Im Code verwenden wir `.T`:

<!--en:ca183af4-->
In TensorFlow verwenden wir `tf.transpose(A)`.

<!--en:1907be66-->
Eine quadratische Matrix heißt *symmetrisch*, wenn $\mathbf{S} = \mathbf{S}^\top$ gilt ($s_{ij} = s_{ji}$ für alle $i, j$):

<!--en:643e0fff-->
### Tensoren höherer Ordnung

<!--en:0a1e06b5-->
Tensoren mit drei oder mehr Achsen bezeichnen wir mit serifenlosen Großbuchstaben ($\mathsf{X}, \mathsf{Y}, \mathsf{Z}$). Wir erzeugen einen synthetischen vierdimensionalen Bild-Minibatch `images` der Form $(256, 1, 28, 28)$ mit Einträgen in $[0, 1]$. Jedes Bild formen wir zu einem Vektor um und erhalten so die zweidimensionale Matrix $\mathbf{X} \in \mathbb{R}^{256 \times 784}$:

<!--en:0311aff7-->
## Elementweise Operationen und Reduktionen

<!--en:0c83abb3-->
### Elementweise Operationen und Hadamard-Produkt

<!--en:cff3b51a-->
Das elementweise Produkt zweier Matrizen $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$ derselben Form heißt *Hadamard-Produkt* $\mathbf{A} \odot \mathbf{B}$. Seine Einträge sind $[\mathbf{A} \odot \mathbf{B}]_{ij} = a_{ij} b_{ij}$. Im Code berechnen wir es mit `*`:

<!--en:73c269a9-->
### Reduktion entlang von Achsen

<!--en:6efc58b8-->
Übergeben wir `axis=0` an `.sum()`, summieren wir über die Zeilen und erhalten einen Vektor der Spaltensummen ($\sum_{i=1}^m a_{ij}$). Mit `axis=1` summieren wir über die Spalten und erhalten einen Vektor der Zeilensummen ($\sum_{j=1}^n a_{ij}$). Ohne `axis` summieren wir sämtliche Einträge (:numref:`fig_la_reduce`):

<!--en:acbfb0a9-->
![Reduktionen einer $2 \times 3$-Matrix. Beim Summieren über Achse 0 bleibt eine Summe je Spalte, bei Achse 1 eine Summe je Zeile. Die reduzierte Achse verschwindet aus der Form. Mit `keepdims=True` behalten die Zeilensummen die Form $(2, 1)$. Durch Broadcasting gelten sie damit für alle drei Spalten, und nach der Division `A / sum_A` summiert sich jede Zeile zu 1.](../img/linear-algebra-reduce-axes.svg)

<!--en:474bb3c2-->
### Reduzierte Achse für Broadcasting beibehalten

<!--en:0e6ab631-->
Um jede Zeile von `A` (Form `(2, 3)`) durch ihre Zeilensumme zu normieren, genügt `A.sum(axis=1)` (Form `(2,)`) nicht. Broadcasting mit `(2, 3)` scheitert, weil die letzten Dimensionen $2$ und $3$ nicht zusammenpassen. Mit `keepdims=True` bleibt die reduzierte Achse mit Länge $1$ erhalten, sodass `sum_A` die Form `(2, 1)` hat:

<!--en:8140f768-->
PyTorch akzeptiert sowohl `dim`/`keepdim` als auch die aus NumPy bekannten Namen `axis`/`keepdims`.

<!--en:7ffcb41b-->
TensorFlow-Tensoren haben keine Methoden namens `.sum` oder `.mean`. Verwenden Sie stattdessen `tf.reduce_sum(A, axis=1)` und `tf.reduce_mean(X, axis=1)`.

<!--en:5519a55b-->
Da `sum_A` die Form `(2, 1)` hat, verteilt Broadcasting bei der Division `A / sum_A` jede Zeilensumme auf alle drei Spalten. Jede Zeile der resultierenden $2 \times 3$-Matrix summiert sich damit zu $1$:

<!--en:0e6774b4-->
Bei unserem $256 \times 784$-Minibatch $\mathbf{X}$ funktioniert Broadcasting direkt, wenn wir für jedes Pixel den Spaltenmittelwert abziehen (`axis=0`, Form `(784,)`). Wollen wir dagegen für jedes Bild den Zeilenmittelwert abziehen (`axis=1`), brauchen wir `keepdims=True` (Form `(256, 1)`):

<!--en:f4df3712-->
Vorsicht bei einer quadratischen Matrix $\mathbf{X}$ ($784 \times 784$): Lassen wir `keepdims=True` in `X - X.mean(axis=1)` weg, erhalten wir keine Fehlermeldung. Stattdessen wird der Mittelwert von Bild $j$ unbemerkt von Spalte $j$ jedes Bildes abgezogen (Übung :exref:`ex_linear-algebra-shape-prediction`).

<!--en:a7d9b6e0-->
## Produkte

<!--en:e3d2789e-->
### Skalarprodukte

<!--en:7ffcd610-->
Das *Skalarprodukt* (oder *innere Produkt*) zweier Vektoren $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ schreiben wir als $\mathbf{x}^\top \mathbf{y}$ oder $\langle \mathbf{x}, \mathbf{y} \rangle$. Es summiert die Produkte der jeweils entsprechenden Einträge:

<!--en:d52c7195-->
Im Code berechnen wir das Skalarprodukt der beiden eindimensionalen Vektoren `x` und `y` direkt mit der Funktion `dot` unseres Frameworks:

<!--en:c060483d-->
In TensorFlow berechnen wir es mit `tf.tensordot(x, y, axes=1)`.

<!--en:0c4b0fe5-->
Gleichwertig können wir `x * y` elementweise multiplizieren und die Produkte mit `sum` summieren (das ergibt $\sum_i x_i = 0 + 1 + 2 = 3$ für $\mathbf{y} = \mathbf{1}$):

<!--en:ca72f865-->
### Matrix-Vektor-Produkte

<!--en:d2aaa11d-->
Zerlegen wir $\mathbf{A} \in \mathbb{R}^{m \times n}$ in ihre $m$ Zeilenvektoren $\mathbf{a}_1^\top, \ldots, \mathbf{a}_m^\top \in \mathbb{R}^n$, so ergibt das *Matrix-Vektor-Produkt* $\mathbf{A}\mathbf{x}$ mit $\mathbf{x} \in \mathbb{R}^n$ einen $m$-dimensionalen Vektor. Seine Einträge sind die Skalarprodukte mit den einzelnen Zeilen (:numref:`fig_la_matvec`):

<!--en:b9b0d86b-->
![Das Matrix-Vektor-Produkt $\mathbf{A}\mathbf{x}$ berechnet für jede Zeile von $\mathbf{A}$ ein Skalarprodukt. Eine $2\times3$-Matrix bildet somit einen Vektor der Länge 3 auf einen Vektor der Länge 2 ab. Die erste Zeile ergibt $0 \cdot 0 + 1 \cdot 1 + 2 \cdot 2 = 5$.](../img/linear-algebra-matvec.svg)

<!--en:6932d4c5-->
Der Infixoperator `@` in Python berechnet sowohl Matrix-Vektor- als auch Matrix-Matrix-Produkte:

<!--en:eae97a09-->
In TensorFlow benötigt `@` zweidimensionale Matrizen. Für Matrix-Vektor-Produkte verwenden wir deshalb `tf.linalg.matvec(A, x)`:

<!--en:4376ef49-->
Spaltenweise betrachtet ist $\mathbf{A}\mathbf{x} = \sum_{j=1}^n x_j (\mathbf{A}\mathbf{e}_j)$ eine Linearkombination der Spalten von $\mathbf{A}$. Die $j^\textrm{th}$ Spalte $\mathbf{A}\mathbf{e}_j$ ist das Bild des $j^\textrm{th}$ Standardbasisvektors $\mathbf{e}_j$. Im $\mathbb{R}^2$ bildet eine $2 \times 2$-Matrix das von $\mathbf{e}_1, \mathbf{e}_2$ aufgespannte Einheitsquadrat auf das von ihren Spalten aufgespannte Parallelogramm ab (:numref:`fig_la_maps`).

<!--en:329f7bbe-->
![Eine $2 \times 2$-Matrix bildet das Einheitsquadrat (gestrichelt) auf das von ihren Spalten $\mathbf{A}\mathbf{e}_1$ und $\mathbf{A}\mathbf{e}_2$ aufgespannte Parallelogramm ab. Von links nach rechts: Drehung um $90^\circ$, Skalierung um den Faktor 2 entlang $\mathbf{e}_1$ und um den Faktor $1/2$ entlang $\mathbf{e}_2$ sowie eine Scherung.](../img/mdl-prelim-linear-maps.svg)

<!--en:0ecd0b86-->
### Matrix-Matrix-Produkte

<!--en:9c0abb6f-->
Für die Matrizen $\mathbf{A} \in \mathbb{R}^{m \times k}$ und $\mathbf{B} \in \mathbb{R}^{k \times n}$ ist der Eintrag von $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$ an der Position $(i, j)$ gleich dem Skalarprodukt von Zeile $i$ der Matrix $\mathbf{A}$ mit Spalte $j$ der Matrix $\mathbf{B}$:

<!--en:32fdda9b-->
Wir können unseren linearen Klassifikator für 10 Klassen $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ nun auf den $256 \times 784$-Minibatch `X` anwenden. Mit `argmax` entlang der Achse $1$ wählen wir für jedes Bild die Kategorie mit dem höchsten Score aus:

<!--en:38effaa7-->
Beim Multiplizieren einer $m \times k$-Matrix mit einer $k \times n$-Matrix berechnen wir $mn$ Skalarprodukte der Länge $k$. Das erfordert $\approx 2mkn$ FLOPs ($2 \times 256 \times 784 \times 10 \approx 4 \times 10^6$ FLOPs für `X @ W` beziehungsweise $2n^3$ FLOPs bei nur $3n^2$ Speichereinträgen für zwei $n \times n$-Matrizen, siehe :numref:`chap_performance`). Zwei besondere quadratische Matrizen begegnen uns in der linearen Algebra immer wieder. Die *Einheitsmatrix* $\mathbf{I} \in \mathbb{R}^{n \times n}$ ($1$ auf der Hauptdiagonalen, sonst $0$) lässt jeden Vektor unverändert ($\mathbf{I}\mathbf{x} = \mathbf{x}$). Die *Inverse* $\mathbf{A}^{-1}$ einer invertierbaren Matrix $\mathbf{A}$ ist durch $\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ definiert (:numref:`subsec_mdl-inverses`).

<!--en:7e144df5-->
Spätere Modelle multiplizieren einen ganzen Stapel von Matrizen auf einmal, eine für jedes Beispiel im Minibatch. Der Operator `@` erledigt das ohne Schleife. Bei Tensoren mit mehr als zwei Achsen multipliziert er entlang der letzten beiden Achsen und verwendet Broadcasting für die vorangehenden (:numref:`subsec_broadcasting`). Wo `@` nicht ausreicht, können wir mit `einsum` die Summe aus :eqref:`eq_matmul` Index für Index angeben. Die Zeichenkette `'bil,lj->bij'` benennt die Achsen der Faktoren und des Ergebnisses. Über einen Index, der im Ergebnis fehlt, summieren wir, hier also über `l`. Wir stapeln drei Vielfache von `A` und multiplizieren sie alle mit `B`:

<!--en:de5eb4a1-->
Das Ergebnis enthält drei $2 \times 4$-Produkte, eines pro Matrix im Stapel. Das Ergebnis von `einsum` stimmt in jedem Eintrag mit dem von `@` überein.

<!--en:5c071cc4-->
## Normen

<!--en:9cdebe33-->
*Normen* messen die Größe eines Vektors oder einer Matrix. Im maschinellen Lernen dienen sie sowohl als *Verlustfunktionen* (engl. loss functions), um den Vorhersagefehler $\hat{\mathbf{y}} - \mathbf{y}$ zu messen, als auch als *Regularisierungsterme*, um die Größe der Gewichte zu beschränken.

<!--en:5343bea0-->
### Euklidische Länge und Winkel

<!--en:b17eba08-->
Die *$\ell_2$-Norm* (oder *euklidische Norm*) von $\mathbf{x} \in \mathbb{R}^n$ ist

<!--en:fc956649-->
Wir schreiben dafür oft kurz $\|\mathbf{x}\|$. Für zwei Vektoren $\mathbf{x}, \mathbf{y}$ in der Ebene mit eingeschlossenem Winkel $\theta$ setzen wir den Kosinussatz $\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ mit der bilinearen Entwicklung $(\mathbf{x} - \mathbf{y})^\top(\mathbf{x} - \mathbf{y}) = \|\mathbf{x}\|^2 - 2\mathbf{x}^\top\mathbf{y} + \|\mathbf{y}\|^2$ gleich und erhalten (:numref:`fig_la_dot`)

<!--en:9894b2b0-->
In beliebiger Dimension $d$ lesen wir :eqref:`eq_dot_cos` von rechts nach links und definieren so die *Kosinusähnlichkeit* (engl. cosine similarity) zweier von null verschiedener Vektoren:

<!--en:a0f8c0a5-->
Sie hängt nur von der Richtung ab und reicht von $+1$ (gleichgerichtet) über $0$ (orthogonal) bis $-1$ (entgegengesetzt gerichtet) (:numref:`fig_la_cosine`).

<!--en:86550ed3-->
![Das Skalarprodukt $\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ ist die Länge von $\mathbf{x}$ mal der vorzeichenbehafteten Länge $\|\mathbf{y}\|\cos\theta$ der Projektion von $\mathbf{y}$ auf die Gerade durch $\mathbf{x}$.](../img/linear-algebra-dot.svg)

<!--en:c8ee7197-->
![Zwei Einheitsvektoren, die den Winkel $\theta$ einschließen. Ihr Skalarprodukt ist gleich $\cos\theta$ und reicht von $1$ (gleichgerichtet) über $0$ (senkrecht) bis $-1$ (entgegengesetzt gerichtet).](../img/mdl-prelim-cosine.svg)

<!--en:d016a4b3-->
Die *Cauchy-Schwarz-Ungleichung* garantiert $|\mathbf{x}^\top\mathbf{y}| \leq \|\mathbf{x}\|\|\mathbf{y}\|$ (also $\cos\theta \in [-1, 1]$). Für die Einheitsvektoren $\mathbf{u} = \mathbf{x}/\|\mathbf{x}\|$ und $\mathbf{v} = \mathbf{y}/\|\mathbf{y}\|$ ergibt das Ausmultiplizieren von $0 \leq \|\mathbf{u} \mp \mathbf{v}\|^2 = 2 \mp 2\mathbf{u}^\top\mathbf{v}$ die Ungleichung $|\mathbf{u}^\top\mathbf{v}| \leq 1$.

<!--en:b97a9d32-->
### $\ell_1$-Norm, $\ell_p$-Norm und Frobeniusnorm

<!--en:9ffe2932-->
Die *$\ell_1$-Norm* (oder *Manhattan-Abstand* vom Ursprung) summiert die Beträge der Komponenten (:numref:`fig_la_norms`):

<!--en:1f37d52b-->
![Die $\ell_2$-Norm von $\mathbf{x}$ ist die Länge des geraden Weges nach $\mathbf{x}$. Die $\ell_1$-Norm ist die Länge des Weges entlang des Gitters.](../img/linear-algebra-norms.svg)

<!--en:626727cd-->
Bei $\ell_2$ werden die Komponenten quadriert, bei $\ell_1$ nicht. Daher reagiert der $\ell_1$-Verlust robuster auf große Fehler durch Ausreißer, und $\ell_1$-Strafterme für die Gewichte können Gewichte genau auf null setzen (Sparsität). Beide Normen gehören zur Familie der *$\ell_p$-Normen* $\|\mathbf{x}\|_p = (\sum_{i=1}^n |x_i|^p)^{1/p}$ ($p \geq 1$) mit $\|\mathbf{x}\|_\infty = \max_i |x_i|$ für $p \to \infty$. Jede Norm erfüllt (i) die absolute Homogenität $\|\alpha\mathbf{x}\| = |\alpha|\|\mathbf{x}\|$, (ii) die Dreiecksungleichung $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$ und (iii) die positive Definitheit ($\|\mathbf{x}\| > 0$ für $\mathbf{x} \neq \mathbf{0}$).

<!--en:2913f1cb-->
Für eine $m \times n$-Matrix $\mathbf{A}$ berechnet die *Frobeniusnorm* die $\ell_2$-Norm ihrer $mn$ Einträge:

<!--en:3f7ac19a-->
PyTorch unterscheidet `torch.linalg.vector_norm` und `torch.linalg.matrix_norm`. Letztere Funktion behandelt einen dreidimensionalen Tensor als Batch zweidimensionaler Matrizen.

<!--en:6149495a-->
## Eigenvektoren und Eigenwerte

<!--en:d5e0b636-->
Was geschieht, wenn ein Netz mit 50 Schichten einen Eingabevektor $\mathbf{x}$ wiederholt mit Matrizen multipliziert? Ein Vektor $\mathbf{v} \neq \mathbf{0}$ ungleich null ist ein *Eigenvektor* von $\mathbf{A} \in \mathbb{R}^{n \times n}$ zum skalaren *Eigenwert* $\lambda$, wenn die Multiplikation mit $\mathbf{A}$ den Vektor $\mathbf{v}$ lediglich skaliert, ohne ihn aus seiner Geraden zu drehen:

<!--en:e072f8c3-->
Für unsere symmetrische Matrix $\mathbf{S} = \left[\begin{smallmatrix} 2 & 1 \\ 1 & 2 \end{smallmatrix}\right]$ ist $(1, 1)$ ein Eigenvektor mit $\lambda = 3$ und $(-1, 1)$ ein Eigenvektor mit $\lambda = 1$ (:numref:`fig_la_eigen`).

<!--en:248b1828-->
![Links: $\mathbf{S}$ bildet den Einheitskreis (gestrichelt) auf eine Ellipse ab. Ihre Achsen liegen in den Eigenvektorrichtungen mit den Streckfaktoren $\lambda = 3$ und $\lambda = 1$. Der Vektor $\mathbf{e}_1$ geht in $\mathbf{S}\mathbf{e}_1 = (2, 1)$ über. Rechts: Die Richtungen von $\mathbf{S}^k\mathbf{e}_1$ für $k = 0, \ldots, 4$ nähern sich dem Eigenvektor $(1, 1)$ zum größten Eigenwert.](../img/mdl-prelim-eigen.svg)

<!--en:d62cfc4f-->
Die Eigenwerte und normierten Eigenvektoren einer reellen symmetrischen Matrix berechnen wir mit `linalg.eigh`:

<!--en:ef21e9f5-->
Nach dem *Spektralsatz* (:numref:`sec_mdl-eigendecompositions`) hat jede reelle symmetrische $n \times n$-Matrix $\mathbf{A}$ genau $n$ reelle Eigenwerte und eine vollständige Orthonormalbasis aus Eigenvektoren $\mathbf{v}_1, \ldots, \mathbf{v}_n$. Entwickeln wir einen beliebigen Vektor $\mathbf{x} = \sum_{i=1}^n c_i \mathbf{v}_i$ in dieser Basis (mit $c_i = \mathbf{v}_i^\top \mathbf{x}$), so ergibt sich

<!--en:9921cce4-->
Mit wachsendem $k$ dominiert der Term mit dem größten $|\lambda_i|$. Der Vektor $\mathbf{A}^k\mathbf{x}$ richtet sich am dominanten Eigenvektor aus und wächst bei jedem Schritt um den Faktor $\max_i |\lambda_i|$ (*Potenziteration*). Für eine symmetrische Matrix ist diese Wachstumsrate zugleich der größte Faktor, um den sie einen Vektor strecken kann, ihre *Spektralnorm* $\|\mathbf{A}\|_2 = \max_{\mathbf{x}\neq\mathbf{0}} \|\mathbf{A}\mathbf{x}\|/\|\mathbf{x}\|$.

<!--en:3abbe95e-->
Kehren wir zum Gedankenexperiment mit 50 Schichten zurück. Die Matrix $\mathbf{M} = \left[\begin{smallmatrix} 1 & 0.1 \\ 0.1 & 1 \end{smallmatrix}\right]$ hat die Eigenwerte $\lambda_1 = 1.1$ und $\lambda_2 = 0.9$. Sie weichen von $1$ um nur $10\%$ ab. Multiplizieren wir die beiden normierten Eigenvektoren mit $\mathbf{M}$, und zwar $50$-mal, wächst der erste um den Faktor $1.1^{50} \approx 117.4$ und der zweite schrumpft auf die Länge $0.9^{50} \approx 0.0052$:

<!--en:dd011050-->
Die Gewichtsmatrizen eines Netzes sind nicht symmetrisch und die meisten sind nicht einmal quadratisch. Verraten die Eigenwerte trotzdem, wie stark eine solche Matrix einen Vektor streckt? Nein. Beide Eigenwerte von $\mathbf{A} = \left[\begin{smallmatrix} 0 & 2 \\ 0 & 0 \end{smallmatrix}\right]$ sind $0$, und es gilt $\mathbf{A}^2 = \mathbf{0}$. Dennoch bildet $\mathbf{A}$ den Vektor $\mathbf{e}_2$ auf $2\mathbf{e}_1$ ab und verdoppelt damit seine Länge. Wegen $\|\mathbf{A}\mathbf{x}\|_2^2 = \mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}$ bestimmt die symmetrische Matrix $\mathbf{A}^\top\mathbf{A}$ die Streckung. Die Quadratwurzeln ihrer Eigenwerte, $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$, heißen *Singulärwerte* von $\mathbf{A}$. Es gibt sie für jede Matrix, ob quadratisch oder nicht. Berechnen wir sie für $\mathbf{S}$ und für $\mathbf{A}$:

<!--en:01e9fe22-->
Für die symmetrische Matrix $\mathbf{S}$ sind die Singulärwerte bis auf Rundungsfehler $3$ und $1$, also die Beträge ihrer Eigenwerte. Dieser Zusammenhang gilt für jede symmetrische Matrix. Für $\mathbf{A}$ sind die Singulärwerte $2$ und $0$. Die Quadratwurzeln der Eigenwerte von $\mathbf{A}^\top\mathbf{A}$ liefern dieselben beiden Zahlen in aufsteigender Reihenfolge. Bei jeder Matrix ist der größte Singulärwert ihre Spektralnorm. Die Anzahl ihrer von null verschiedenen Singulärwerte heißt *Rang*. Er gibt an, wie viele unabhängige Richtungen nach der Multiplikation erhalten bleiben. Für $\mathbf{A}$ ist es eine. Beides folgt aus der Singulärwertzerlegung (:numref:`sec_mdl-svd-low-rank`).

<!--en:4f7d783d-->
Hält die Gewichtsmatrix jeder Schicht die Vektorlängen im Mittel nicht nahe bei $1$, *explodieren* oder *verschwinden* Aktivierungen und Gradienten exponentiell mit der Tiefe (:numref:`sec_numerical_stability`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:2339a129-->
Mit linearer Algebra lassen sich Modelle für ganze Minibatches kompakt ausdrücken. Fassen wir $256$ zu Vektoren umgeformte Bilder in $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ zusammen, ergeben sich alle $2{,}560$ Kategoriescores aus $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$. Reduktionen fassen Einträge entlang bestimmter Achsen zusammen (`keepdims=True` behält die Achse mit Länge $1$ für Broadcasting bei). Skalarprodukte $\mathbf{x}^\top\mathbf{y} = \sum_i x_i y_i$ bilden die Grundlage für Matrix-Vektor-Produkte $\mathbf{A}\mathbf{x}$ und Matrix-Matrix-Produkte $\mathbf{A}\mathbf{B}$ (mit einem Aufwand von $\approx 2mkn$ FLOPs). Normen ($\ell_2$-, $\ell_1$-, $\ell_p$-, Frobenius- und Spektralnorm) messen die Größe von Vektoren und Matrizen. Normierte Skalarprodukte ergeben die Kosinusähnlichkeit $\cos\theta = \mathbf{x}^\top\mathbf{y}/(\|\mathbf{x}\|\|\mathbf{y}\|)$. Eigenvektoren ($\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$) erklären, warum wiederholte Matrixmultiplikation die Komponenten mit $\lambda^k$ skaliert. Bei einer nicht symmetrischen Matrix beschreiben die Singulärwerte die Streckung. Der größte davon ist die Spektralnorm.

<!--en:2294243a-->
## Übungen

<!--en:e403b4d6-->
1. **Transponierte und Matrixprodukte.** Beweisen Sie die folgenden algebraischen Identitäten:
    1. Zeigen Sie, dass sich beim Transponieren eines Matrixprodukts die Reihenfolge der Faktoren umkehrt. Es gilt $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$, sofern $\mathbf{A}\mathbf{B}$ definiert ist. (Schreiben Sie den Eintrag an der Position $(i, j)$ auf beiden Seiten mit :eqref:`eq_matmul` aus.)
    1. Zeigen Sie für *jede* rechteckige Matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$, dass das Produkt $\mathbf{X}\mathbf{X}^\top$ stets quadratisch und symmetrisch ist. Dasselbe gilt für $\mathbf{X}^\top\mathbf{X}$. Welche Form hat für unseren $256 \times 784$-Bild-Minibatch $\mathbf{X}$ das Produkt $\mathbf{X}\mathbf{X}^\top$, und was misst sein Eintrag an der Position $(i, j)$?

<!--en:6a7bcb2e-->
2. [code] **Formen bei Reduktion und Broadcasting vorhersagen.** Gegeben seien `images` mit der Form $(256, 1, 28, 28)$ und `X` mit der Form $(256, 784)$. Sagen Sie für jeden der folgenden Ausdrücke die Form oder den Fehler vorher, ohne Code auszuführen. Prüfen Sie Ihre Antworten anschließend im Code.
    1. Was liefert `len(images)`? Welche Formen hat `images.sum(axis=(2, 3))` mit `keepdims=False` beziehungsweise `keepdims=True`? Welches der beiden Ergebnisse eignet sich direkt als Divisor für `images` mittels Broadcasting, und was berechnet `images / images.sum(axis=(2, 3), keepdims=True)`?
    1. Was geschieht, wenn Sie `X - X.mean(axis=1)` auf unsere $256 \times 784$-Matrix `X` anwenden, und wie behebt `keepdims=True` das Problem?
    1. Ihr Minibatch enthalte nun zufällig $n = 784$ Bilder, sodass `X` eine quadratische $784 \times 784$-Matrix ist. Zeigen Sie, dass `X - X.mean(axis=1)` *ohne Fehlermeldung* läuft. Geben Sie an, was der Eintrag an der Position $(i, j)$ von `X - X.mean(axis=1)` aus den Einträgen von `X` tatsächlich berechnet. Erklären Sie, warum ein unbemerkter Broadcasting-Fehler bei einer quadratischen Matrix wesentlich gefährlicher ist als eine Fehlermeldung wegen inkompatibler Formen.
   :exlabel:`ex_linear-algebra-shape-prediction`

<!--en:16b2fba5-->
2. [code] **Formen bei Reduktion und Broadcasting vorhersagen.** Gegeben seien `images` mit der Form $(256, 1, 28, 28)$ (umgewandelt in `float32`) und `X` mit der Form $(256, 784)$. Sagen Sie für jeden der folgenden Ausdrücke die Form oder den Fehler vorher, ohne Code auszuführen. Prüfen Sie Ihre Antworten anschließend im Code.
    1. Was liefert `len(images)`? Welche Formen hat `tf.reduce_sum(images, axis=(2, 3))` mit `keepdims=False` beziehungsweise `keepdims=True`? Welches der beiden Ergebnisse eignet sich direkt als Divisor für `images` mittels Broadcasting, und was berechnet `images / tf.reduce_sum(images, axis=(2, 3), keepdims=True)`?
    1. Was geschieht, wenn Sie `X - tf.reduce_mean(X, axis=1)` auf unsere $256 \times 784$-Matrix `X` anwenden, und wie behebt `keepdims=True` das Problem?
    1. Ihr Minibatch enthalte nun zufällig $n = 784$ Bilder, sodass `X` eine quadratische $784 \times 784$-Matrix ist. Zeigen Sie, dass `X - tf.reduce_mean(X, axis=1)` *ohne Fehlermeldung* läuft. Geben Sie an, was der Eintrag an der Position $(i, j)$ von `X - tf.reduce_mean(X, axis=1)` aus den Einträgen von `X` tatsächlich berechnet. Erklären Sie, warum ein unbemerkter Broadcasting-Fehler bei einer quadratischen Matrix wesentlich gefährlicher ist als eine Fehlermeldung wegen inkompatibler Formen.
   :exlabel:`ex_linear-algebra-shape-prediction`

<!--en:ea2255e8-->
3. **Wie stumpf können die Winkel zwischen Vektoren paarweise sein?** Untersuchen Sie, wie negativ alle paarweisen Skalarprodukte von $k$ Vektoren zugleich sein können.
    1. Können drei Vektoren $\mathbf{u}, \mathbf{v}, \mathbf{w} \in \mathbb{R}^2$ in der Ebene gleichzeitig $\mathbf{u}^\top\mathbf{v} < 0$, $\mathbf{v}^\top\mathbf{w} < 0$ und $\mathbf{u}^\top\mathbf{w} < 0$ erfüllen? Geben Sie ein konkretes Zahlenbeispiel an oder beweisen Sie, dass keines existiert.
    1. Es seien $k \geq 2$ Einheitsvektoren $\mathbf{u}_1, \ldots, \mathbf{u}_k \in \mathbb{R}^d$ ($d \geq k - 1$) symmetrisch angeordnet, sodass für jedes Paar verschiedener Vektoren dasselbe Skalarprodukt $\mathbf{u}_i^\top\mathbf{u}_j = c$ für alle $i \neq j$ gilt. Multiplizieren Sie $0 \leq \|\mathbf{u}_1 + \cdots + \mathbf{u}_k\|_2^2$ aus und beweisen Sie damit $c \geq -\frac{1}{k - 1}$. Was folgt daraus für die paarweise Kosinusähnlichkeit von $10$ Klassenvorlagen der Länge eins in einem Klassifikator für 10 Klassen, wenn alle $10$ Vorlagen paarweise denselben Winkel einschließen?

    *Nach MIT 18.06, [Aufgabenblatt 1](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/22297c2a6dcf06d82e93ee4af115e91a_MIT18_06S10_pset1_s10_soln.pdf), Abschnitt 1.2, Aufgabe 28.*

<!--en:24c49717-->
1. [code] **Normen über mehrere Parametertensoren.** Die hier eingeführten Funktionen für Vektornormen lassen sich auch auf Tensoren höherer Ordnung anwenden.
    1. Berechnen Sie die Vektornorm mit Standardargumenten für einen dreidimensionalen Tensor der Form $(2, 3, 4)$. Beschreiben Sie die verwendete mathematische Formel in Worten. Prüfen Sie Ihre Antwort, indem Sie selbst die Einträge quadrieren, die Quadrate summieren und die Wurzel ziehen.
    1. Ein tiefes neuronales Netz speichert seine lernbaren Parameter in einer Liste von Gewichts- und Bias-Tensoren unterschiedlicher Ordnung und Form. Beim Überwachen des Trainings oder beim Begrenzen der Gradienten (:numref:`sec_rnn-scratch`) benötigen wir häufig eine einzige globale $\ell_2$-Norm aller zu einem großen Vektor verketteten Parameter. Drücken Sie diese globale $\ell_2$-Norm durch die einzelnen Tensornormen aus, ohne die Tensoren in Vektoren umzuformen oder zu verketten. Prüfen Sie Ihre Formel im Code an drei Tensoren der Formen $(3, 4)$, $(4,)$ und $(2, 3, 5)$.

<!--en:d06354fd-->
1. **Assoziativität der Matrixmultiplikation und Rechenaufwand.** Die Matrixmultiplikation ist assoziativ, mathematisch gilt also $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$. Trotzdem kann die Klammerung einer Kette von Matrixprodukten den Rechenaufwand um Größenordnungen verändern. Gegeben seien $\mathbf{A} \in \mathbb{R}^{1000 \times 50000}$, $\mathbf{B} \in \mathbb{R}^{50000 \times 20}$ und $\mathbf{C} \in \mathbb{R}^{20 \times 10000}$. Berechnen Sie für $(\mathbf{A}\mathbf{B})\mathbf{C}$ und $\mathbf{A}(\mathbf{B}\mathbf{C})$ jeweils die Gesamtzahl der FLOPs und die Anzahl der Einträge in der Zwischenmatrix. Um welche Faktoren unterscheiden sich Rechenaufwand und Speicherbedarf für das Zwischenergebnis bei den beiden Reihenfolgen?

<!--en:561876fb-->
1. [code] **Matrix aller paarweisen euklidischen Abstände ohne Schleifen.** ● Gegeben seien zwei Sammlungen von $50$ Vektoren im $\mathbb{R}^8$. Sie stehen als Zeilen in $\mathbf{X} \in \mathbb{R}^{50 \times 8}$ und $\mathbf{Y} \in \mathbb{R}^{50 \times 8}$. Wir wollen die $50 \times 50$-Matrix $\mathbf{D}$ der quadrierten euklidischen Abstände $D_{ij} = \|\mathbf{x}_i - \mathbf{y}_j\|_2^2$ ohne Python-Schleifen berechnen:
    1. Berechnen Sie $\mathbf{D}$ zunächst durch dreidimensionales Broadcasting. Formen Sie $\mathbf{X}$ zu $(50, 1, 8)$ und $\mathbf{Y}$ zu $(1, 50, 8)$ um, subtrahieren Sie, quadrieren Sie und summieren Sie über die letzte Achse. Berechnen Sie anschließend $\mathbf{D}$ durch Ausmultiplizieren des Quadrats gemäß $\|\mathbf{x}_i - \mathbf{y}_j\|_2^2 = \|\mathbf{x}_i\|_2^2 + \|\mathbf{y}_j\|_2^2 - 2\mathbf{x}_i^\top\mathbf{y}_j$. Dafür benötigen Sie nur zeilenweise Normberechnungen, zweidimensionales Broadcasting und ein einziges Matrixprodukt $\mathbf{X}\mathbf{Y}^\top$. Prüfen Sie, ob beide Ergebnisse innerhalb der Gleitkommagenauigkeit übereinstimmen.
    1. Wenn $\mathbf{X}$ und $\mathbf{Y}$ jeweils $10{,}000$ Vektoren im $\mathbb{R}^{128}$ enthalten, wie viele Gleitkommazahlen muss das Verfahren mit dreidimensionalem Broadcasting im Vergleich zum Matrixprodukt als Zwischenwerte speichern? Welches Verfahren eignet sich für große Datenmengen?
    1. Erklären Sie, weshalb Rundungsfehler bei der Gleitkommaberechnung von $\|\mathbf{x}_i\|_2^2 + \|\mathbf{y}_j\|_2^2 - 2\mathbf{x}_i^\top\mathbf{y}_j$ gelegentlich kleine negative Zahlen (etwa $-10^{-7}$) liefern, wenn $\mathbf{x}_i = \mathbf{y}_j$ gilt. Zeigen Sie, wie sich die Ausgabe vor dem Wurzelziehen nach unten begrenzen lässt.

    *Nach Stanford CS231n, [Aufgabenblatt 1](https://cs231n.github.io/assignments2026/assignment1/) (Abstände der $k$ nächsten Nachbarn), und fast.ai, [Deep Learning from the Foundations](https://github.com/fastai/course-v3/blob/master/files/dl-2019/notes/notes-2-8.md), Lektion 8.*

<!--en:ffed3a72-->
1. [code] **Potenziteration und Konvergenzrate.** ● Implementieren Sie die normierte Potenziteration mit der wiederholten Aktualisierung $\mathbf{v}_{k+1} = \mathbf{A}\mathbf{v}_k / \|\mathbf{A}\mathbf{v}_k\|_2$. Beginnen Sie mit einem zufälligen Einheitsvektor $\mathbf{v}_0$ und verwenden Sie die symmetrische $3 \times 3$-Matrix

   $$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 0 & 4 \\ 3 & 4 & 5 \end{bmatrix}.$$

    1. Halten Sie die erste Iteration $k$ fest, bei der der Betrag der Kosinusähnlichkeit $|\mathbf{v}_k^\top \mathbf{v}_{k-1}|$ zwischen aufeinanderfolgenden Näherungen größer als $0.999$ ist. Vergleichen Sie den nach Konvergenz erhaltenen Vektor $\mathbf{v}_k$ mit dem dominanten Eigenvektor aus `eigh(A)`. Erklären Sie, warum sich beide Vektoren um einen Vorzeichenfaktor $\pm 1$ unterscheiden können.
    1. Messen Sie in jedem Schritt den Sinus des Winkels zwischen $\mathbf{v}_k$ und dem tatsächlichen dominanten Eigenvektor $\mathbf{u}_{\max}$, also $\sin\theta_k = \sqrt{1 - (\mathbf{v}_k^\top \mathbf{u}_{\max})^2}$. Berechnen Sie den Quotienten $\sin\theta_{k+1} / \sin\theta_k$, der angibt, wie stark der Sinus bei jedem Schritt schrumpft. Drücken Sie seinen asymptotischen Wert durch die Eigenwerte von $\mathbf{A}$ aus. Verwenden Sie dazu :eqref:`eq_matrix_power` und erklären Sie damit, warum die Potenziteration in so wenigen Schritten konvergiert ist.
    1. Führen Sie die Potenziteration nun mit der verschobenen Matrix $\mathbf{B} = \mathbf{A} - 9\mathbf{I}$ aus. Gegen welchen Eigenwert und Eigenvektor von $\mathbf{A}$ konvergiert die Potenziteration jetzt? Was geschieht mit dem Vorzeichen aufeinanderfolgender Näherungen $\mathbf{v}_k$ und $\mathbf{v}_{k+1}$, und warum?

    *Nach MIT 18.06, [Aufgabenblatt 1](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/22297c2a6dcf06d82e93ee4af115e91a_MIT18_06S10_pset1_s10_soln.pdf), Abschnitt 2.1, Aufgaben 29–30.*

<!--en:1743a7e9-->
[Diskussion](https://d2l.discourse.group/t/30)

<!--en:d3a8635e-->
[Diskussion](https://d2l.discourse.group/t/31)

<!--en:a8812ecb-->
[Diskussion](https://d2l.discourse.group/t/196)

<!--en:cf6c98e6-->
[Diskussion](https://d2l.discourse.group/t/17968)

<!--en:375e7c9a-->
Lineare Algebra als rechnerische Grundlage des Deep Learning<br>**Tensoren und Symmetrie · Reduktion entlang von Achsen · Projektionen und Matrixprodukte · Normen · spektrale Dynamik**

<!--en:e3ca9289-->
::: {.slide title="Warum Deep Learning auf linearer Algebra beruht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:17e9bb10-->
- **Lineare und nichtlineare Abbildungen im Wechsel:** In vollständig verbundenen Schichten, Faltungs- und Attention-Schichten wechseln sich **lineare Transformationen** ($\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$) und **nichtlineare Aktivierungen** ab.

<!--en:4d810f12-->
- **Von einzelnen Vektoren zu Batches für die Hardware:** Fassen wir $256$ zu Vektoren umgeformte Fashion-MNIST-Bilder in $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ und $10$ Klassenvorlagen in $\mathbf{W} \in \mathbb{R}^{784 \times 10}$ zusammen, lassen sich $256$ Auswertungen als **ein Matrixprodukt** auf den Tensor Cores der GPU berechnen.

<!--en:3eae8dd3-->
- **Form und Aufwand vorhersagen:** Ausgabeform $(256, 10)$ und Rechenaufwand ($2 \times 256 \times 784 \times 10 \approx 4.0 \times 10^6\text{ FLOPs}$) stehen vor der Ausführung fest.

<!--en:5ee600cb-->
::: {.slide title="Skalare, Vektoren, Vektordimension und Tensorordnung"}

<!--en:b47479c8-->
[Grundbegriffe]{.kicker}

<!--en:3cca4604-->
- **Skalare ($x \in \mathbb{R}$):** Tensoren der Ordnung $0^\textrm{th}$ mit leerer Form `()` (z. B. Verlust $\mathcal{L}$ oder Lernrate $\eta$).

<!--en:a2ecb13e-->
- **Vektoren ($\mathbf{x} \in \mathbb{R}^d$):** Tensoren der Ordnung $1^\textrm{st}$ und Form `(d,)`, mathematisch als Spaltenvektoren $\mathbf{x} = [x_1, \ldots, x_d]^\top$ aufgefasst.

<!--en:e6868c4a-->
- **Begriffe unterscheiden:** Die **Dimension** $d$ eines Vektors ist seine Länge entlang Achse 0 (`len(x)`). Die **Ordnung (Rang)** eines Tensors ist die Anzahl seiner Achsen (`ndim`). Ein $784$-dimensionaler Vektor ist ein Tensor der Ordnung $1^\textrm{st}$.

<!--en:b2da0c9f-->
::: {.slide title="Matrizen, Transposition und Symmetrie"}

<!--en:a1481f02-->
[Zweidimensionale Darstellungen]{.kicker}

<!--en:1815c456-->
- **Designmatrizen ($\mathbf{X} \in \mathbb{R}^{n \times d}$):** Ein Beispiel je Zeile $i$, ein Merkmal je Spalte $j$.

<!--en:889ecfb2-->
- **Transponierte ($[\mathbf{A}^\top]_{ij} = a_{ji}$):** Spiegelt die Einträge an der Hauptdiagonalen und vertauscht die Dimensionen: $(m, n) \to (n, m)$.

<!--en:daf1a779-->
- **Symmetrische Matrizen ($\mathbf{S} = \mathbf{S}^\top$):** Kovarianzmatrizen $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$ und Hesse-Matrizen $\nabla^2 \mathcal{L}$.

<!--en:36dfe6ca-->
::: {.slide title="Tensoren höherer Ordnung und elementweises Rechnen mit dem Hadamard-Produkt"}

<!--en:8b9aed76-->
[Tensoren und Hadamard-Produkt]{.kicker}

<!--en:52cb22fd-->
- **Tensoren höherer Ordnung:** Dreidimensionale Tokenfolgen `(B, T, d)` und vierdimensionale Bilder `(256, 1, 28, 28)` lassen sich mit `reshape(256, -1)` in zweidimensionale Matrizen umformen.

<!--en:d579cab2-->
- **Hadamard-Produkt ($\mathbf{A} \odot \mathbf{B}$, `A * B`):** Multipliziert einander entsprechende Einträge $[\mathbf{A} \odot \mathbf{B}]_{ij} = a_{ij} b_{ij}$, etwa zur Steuerung durch Gates (LSTMs, SwiGLU). Das Matrixprodukt `A @ B` ist eine andere Operation.

<!--en:57b5e3d0-->
::: {.slide title="Reduktion entlang von Achsen"}

<!--en:412006fb-->
[Reduktionen]{.kicker}

<!--en:5350aca3-->
- Reduktionen (`.sum()`, `.mean()`) fassen Einträge entlang einer Achse zusammen und **entfernen diese Achse aus der Form**:
  - **Alle Achsen (`A.sum()`):** `(2, 3) -> ()` ergibt einen Skalar.
  - **Über die Zeilen (`axis=0`):** $c_j = \sum_{i=1}^m a_{ij}$ ergibt Spaltensummen der Form **`(3,)`**.
  - **Über die Spalten (`axis=1`):** $r_i = \sum_{j=1}^n a_{ij}$ ergibt Zeilensummen der Form **`(2,)`**.

<!--en:4d4b8410-->
::: {.slide title="Summen mit erhaltener Achse (keepdims=True) und Normierung durch Broadcasting"}

<!--en:de5a2274-->
[Reduktion und Broadcasting]{.kicker}

<!--en:a3d926ef-->
- **Warum `A / A.sum(axis=1)` scheitert:** Bei der Division von `(2, 3)` durch `(2,)` werden die Formen rechtsbündig als `(2, 3)` und `(1, 2)` ausgerichtet. Die letzten Dimensionen $3 \neq 2$ sind inkompatibel.

<!--en:89e78bf5-->
- **`keepdims=True` erhält die Achse mit Länge 1 (`(2, 1)`):** Bei `A / sum_A` greift Broadcasting über die Spalten. Jede Zeile summiert sich zu $1$ (**Softmax-Funktion**, **Attention**).

<!--en:92140e49-->
- **$\mathbf{X} \in \mathbb{R}^{256 \times 784}$ zentrieren:** Pixelmittelwerte (`axis=0`, `(784,)`) eignen sich direkt für Broadcasting. Bildmittelwerte (`axis=1`, `(256, 1)`) erfordern `keepdims=True` (**LayerNorm**).

<!--en:4a076fd7-->
::: {.slide title="Skalarprodukt: gewichtete Summen und geometrische Projektion"}

<!--en:fc1c769f-->
[Innere Produkte]{.kicker}

<!--en:7f04b0d4-->
- Für $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ ist das **Skalarprodukt** (innere Produkt) die Summe der Produkte der entsprechenden Koordinaten:
  $$\mathbf{x}^\top \mathbf{y} = \langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d} x_i y_i = \|\mathbf{x}\|_2 \|\mathbf{y}\|_2 \cos\theta$$

<!--en:57eda7b1-->
- **Bedeutung im Deep Learning:** Die Voraktivierung $\mathbf{w}^\top \mathbf{x} + b$ eines einzelnen Neurons ist ein Skalarprodukt mit addiertem Bias. Gelten $w_i \geq 0$ und $\sum_i w_i = 1$, berechnet $\mathbf{w}^\top \mathbf{x}$ ein konvex gewichtetes Mittel.

<!--en:a866b490-->
::: {.slide title="Kosinusähnlichkeit und Cauchy-Schwarz-Ungleichung"}

<!--en:3dda3eba-->
[Winkel in hohen Dimensionen]{.kicker}

<!--en:5e1a83b4-->
- Die Division durch die euklidischen Längen definiert die **Kosinusähnlichkeit** in jeder Dimension $d$:
  $$\cos\theta = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2} \in [-1, 1]$$

<!--en:3a97012c-->
- **Skaleninvarianz:** Eine Änderung der Helligkeit oder der Länge eines Einbettungsvektors (`2 * x`) lässt $\cos\theta$ unverändert. Daher dient die Kosinusähnlichkeit häufig zum Vergleich von Token-, Bild- und Retrieval-Einbettungen.

<!--en:2e7af27c-->
- **Cauchy-Schwarz ($|\mathbf{x}^\top\mathbf{y}| \leq \|\mathbf{x}\|_2\|\mathbf{y}\|_2$):** Folgt direkt aus $0 \leq \|\mathbf{u} \mp \mathbf{v}\|_2^2 = 2 \mp 2\mathbf{u}^\top\mathbf{v}$ für die Einheitsvektoren $\mathbf{u}, \mathbf{v}$.

<!--en:5361ae52-->
::: {.slide title="Matrix-Vektor-Produkte: lineare Abbildungen auf Merkmalsräumen"}

<!--en:43ed7a0c-->
[Matrix-Vektor-Produkte]{.kicker}

<!--en:d0b78abd-->
- Eine Matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ definiert eine lineare Abbildung $\mathbb{R}^n \to \mathbb{R}^m$. Es gibt zwei einander ergänzende Sichtweisen:
  1. **Zeilenweise ($m$ parallele Skalarprodukte):** $[\mathbf{A}\mathbf{x}]_i = \mathbf{a}_i^\top \mathbf{x}$ projiziert die Eingabe $\mathbf{x}$ auf jede der $m$ Zeilenvorlagen von $\mathbf{A}$.
  2. **Spaltenweise (Basiskombination):** $\mathbf{A}\mathbf{x} = \sum_{j=1}^n x_j \mathbf{a}_{:,j}$ ist eine Linearkombination der Spalten von $\mathbf{A}$ mit den Koordinaten von $\mathbf{x}$ als Gewichten.

<!--en:c18351a7-->
::: {.slide title="Geometrische Sicht: Wie Matrizen den Raum transformieren"}

<!--en:34234143-->
[Lineare Geometrie]{.kicker}

<!--en:f40b1d19-->
- Da $\mathbf{A}\mathbf{e}_j$ die $j^\textrm{th}$ Spalte von $\mathbf{A}$ auswählt, zeigen die Spalten, wohin die Standardbasisvektoren $\mathbf{e}_1, \mathbf{e}_2$ abgebildet werden:
  - **Drehungen, Skalierungen entlang der Achsen und Scherungen** verformen das gestrichelte Einheitsquadrat zu dem von $\mathbf{A}\mathbf{e}_1$ und $\mathbf{A}\mathbf{e}_2$ aufgespannten Parallelogramm.
  - In den Schichten eines neuronalen Netzes dreht, streckt und projiziert jede Gewichtsmatrix $\mathbf{W}$ den Merkmalsraum, sodass zunächst vermischte Klassen linear trennbar werden.

<!--en:5a5f4987-->
::: {.slide title="Matrix-Matrix-Multiplikation: lineare Schichten für ganze Batches"}

<!--en:ec5e4916-->
[Matrix-Matrix-Produkte]{.kicker}

<!--en:44e5aac2-->
- Für $\mathbf{A} \in \mathbb{R}^{m \times k}$ und $\mathbf{B} \in \mathbb{R}^{k \times n}$ berechnet $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$ (`A @ B`) insgesamt $mn$ Skalarprodukte der Länge $k$:
  $$c_{ij} = \mathbf{a}_i^\top \mathbf{b}_{:,j} = \sum_{l=1}^{k} a_{il}\, b_{lj}$$

<!--en:14b7fe90-->
- **Formregel:** $(m, k) \times (k, n) \to (m, n)$. Die inneren Dimensionen $k$ müssen übereinstimmen und entfallen durch die Summation.

<!--en:a67cd923-->
- **Ein Klassifikator für 10 Klassen auf einem Batch ($\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$):** `(256, 784) @ (784, 10) + (10,) -> (256, 10)` berechnet alle $2{,}560$ Logits auf einmal. Mit `argmax` entlang Achse 1 erhalten wir die vorhergesagte Klasse jedes Bildes:

<!--en:0bdf452c-->
::: {.slide title="Rechenaufwand und arithmetische Intensität der Matrixmultiplikation"}

<!--en:e7fe691c-->
[Hardware und FLOPs]{.kicker}

<!--en:311c2dee-->
- **Anzahl der FLOPs:** Bei der Multiplikation $(m, k) \times (k, n)$ entstehen $mn$ Skalarprodukte der Länge $k$. Das erfordert **$\approx 2mkn\text{ FLOPs}$**.

<!--en:8d62184e-->
- **$O(n^2)$ Byte gegenüber $O(n^3)$ FLOPs:** Bei elementweisen Operationen ($0.33\text{ FLOP/el}$) und Matrix-Vektor-Produkten mit einem einzigen Vektor ($2\text{ FLOP/el}$) begrenzt die HBM-Bandbreite den Durchsatz.

<!--en:652c5047-->
- **Warum Batches?** Fassen wir Vektoren zu einer Matrix $\mathbf{X}\mathbf{W}$ zusammen, steigt die arithmetische Intensität auf $O(n)$ ($2{,}730\text{ FLOP/el}$ bei $n=4096$). So lassen sich die Tensor Cores der GPU auslasten.

<!--en:8631da31-->
::: {.slide title="Verständnisfrage: Formen bei Reduktion, Broadcasting und Matrixmultiplikation"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:2d84dbac-->
Gegeben seien der zu Zeilenvektoren umgeformte Minibatch `X` der Form `(256, 784)` und die zeilenweise gespeicherte Gewichtsmatrix `W_T` der Form `(10, 784)`:

<!--en:5fc52908-->
1. Warum scheitert `X @ W_T`, und welcher Ausdruck berechnet die `(256, 10)`-Logit-Matrix?

<!--en:0788f94f-->
2. Welche Formen haben `X.sum(axis=0)` und `X.sum(axis=1, keepdims=True)`?

<!--en:304dc4e8-->
3. Ein Minibatch enthalte zufällig $n = 784$ Bilder, sodass `X` quadratisch ist (`(784, 784)`). Was ergibt `X - X.mean(axis=1)` *ohne* `keepdims=True`?

<!--en:557b98b6-->
**Antworten:**

<!--en:1039b8e9-->
1. Die inneren Dimensionen $784 \neq 10$ passen nicht zusammen. Transponieren mit **`X @ W_T.T`** bringt `(256, 784) @ (784, 10) -> (256, 10)` zur Deckung.

<!--en:e627917e-->
2. `X.sum(axis=0)` hat die Form **`(784,)`** (Summe je Pixel), `X.sum(axis=1, keepdims=True)` die Form **`(256, 1)`** (Summe je Bild).

<!--en:4c5a1955-->
3. **Unbemerkter Broadcasting-Fehler:** `X.mean(axis=1)` hat die Form `(784,)`. Rechtsbündig ausgerichtet ergibt diese `(1, 784)`. Damit wird *der Mittelwert von Bild $j$ von Pixel $j$ jedes Bildes* abgezogen, ohne dass ein Fehler gemeldet wird.

<!--en:7d06e264-->
::: {.slide title="Vektor- und Matrixnormen als Größenmaße"}

<!--en:b39d105e-->
[Normen]{.kicker}

<!--en:a19b37a7-->
- Jede Norm $\|\cdot\|$ erfüllt $\|\alpha\mathbf{x}\| = |\alpha|\|\mathbf{x}\|$, $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$ und $\|\mathbf{x}\| > 0$ für $\mathbf{x} \neq \mathbf{0}$:
  - **Euklidische Norm ($\ell_2$):** $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ (für $(3, -4)$ gilt $\|\mathbf{x}\|_2 = 5$).
  - **Manhattan-Norm ($\ell_1$):** $\|\mathbf{x}\|_1 = \sum_i |x_i|$ (für $(3, -4)$ gilt $\|\mathbf{x}\|_1 = 7$).
  - **$\ell_p$- und Frobeniusnorm:** $\|\mathbf{x}\|_p = (\sum_i |x_i|^p)^{1/p}$, $\|\mathbf{A}\|_\textrm{F} = \sqrt{\sum_{i,j} a_{ij}^2}$.

<!--en:e2f67264-->
::: {.slide title="Wie die Norm Ausreißer, Gewichtszerfall und Sparsität beeinflusst"}

<!--en:e6689021-->
[Normen im maschinellen Lernen]{.kicker}

<!--en:25657f5a-->
- **Ausreißer ($\ell_2^2$- gegenüber $\ell_1$-Verlust):** Bei $\mathbf{r} = (1, -1, 1, -1, 10)$ macht der einzelne Ausreißer $10$ **$96\%$** von $\|\mathbf{r}\|_2^2$ aus (Steigung $2r$), aber nur **$71\%$** von $\|\mathbf{r}\|_1$ (beschränkte Steigung $\pm 1$).

<!--en:f2189057-->
- **Gewichtszerfall ($\frac{\lambda}{2}\|\mathbf{W}\|_\textrm{F}^2$):** Die runde $\ell_2$-Kugel zieht Gewichte stetig gegen null.

<!--en:130d50e6-->
- **Sparsität ($\lambda\|\mathbf{w}\|_1$):** Die spitzen Ecken der $\ell_1$-Raute auf den Achsen treiben irrelevante Gewichte **genau auf null**.

<!--en:a04776c7-->
::: {.slide title="Eigenwertzerlegung: invariante Richtungen und Spektralsatz"}

<!--en:54cce930-->
[Eigenwertzerlegung]{.kicker}

<!--en:1465fe79-->
- Ein Vektor $\mathbf{v}$ ungleich null ist ein **Eigenvektor** von $\mathbf{A} \in \mathbb{R}^{n \times n}$ zum **Eigenwert** $\lambda$, wenn $\mathbf{A}\mathbf{v} = \lambda \mathbf{v} \Longrightarrow \mathbf{A}^k \mathbf{v} = \lambda^k \mathbf{v}$ gilt.

<!--en:d2bdf815-->
- **Spektralsatz:** Jede reelle symmetrische Matrix $\mathbf{S} = \mathbf{S}^\top$ hat $n$ **reelle Eigenwerte** und eine **Orthonormalbasis aus Eigenvektoren** (`linalg.eigh`).

<!--en:8efc277b-->
::: {.slide title="Potenziteration und Spektralnorm"}

<!--en:bfa75699-->
[Wiederholte Matrixmultiplikation]{.kicker}

<!--en:5ef7357f-->
- Die Entwicklung einer beliebigen Eingabe $\mathbf{x} = \sum_{i=1}^n c_i \mathbf{v}_i$ in der Orthonormalbasis aus Eigenvektoren zeigt die Wirkung wiederholter Multiplikation:
  $$\mathbf{A}^k \mathbf{x} = \sum_{i=1}^n c_i \lambda_i^k \mathbf{v}_i$$

<!--en:4bc6f80d-->
- **Potenziteration:** Mit wachsendem $k$ dominiert der Term mit dem größten $|\lambda_i|$. Der Vektor $\mathbf{A}^k\mathbf{x}$ richtet sich am dominanten Eigenvektor aus und seine Norm wächst pro Schritt um den Faktor $\max_i |\lambda_i|$.

<!--en:345f04ce-->
- Für symmetrisches $\mathbf{A}$ ist $\max_i |\lambda_i|$ gleich der **Spektralnorm** $\|\mathbf{A}\|_2 = \max_{\mathbf{x}\neq\mathbf{0}} \|\mathbf{A}\mathbf{x}\|_2 / \|\mathbf{x}\|_2$ (maximaler Streckfaktor).

<!--en:ca91fb4d-->
::: {.slide title="Spektrale Dynamik tiefer Netze: explodierende und verschwindende Signale"}

<!--en:e4de5c27-->
[Warum Initialisierung und Normierung wichtig sind]{.kicker}

<!--en:e5f6e025-->
- Über $L = 50$ Schichten führt schon eine Eigenwertabweichung von $10\%$ ($\lambda_1 = 1.1$, $\lambda_2 = 0.9$) zu exponentiellem Wachstum wie $1.1^{50} \approx \mathbf{117.4\times}$ oder zum Abklingen wie $0.9^{50} \approx \mathbf{0.0052\times}$.

<!--en:4e5674a7-->
- Die Streckfaktoren (Singulärwerte) jeder Schicht sollen nahe bei $1$ bleiben. Das motiviert die **Xavier- und He-Initialisierung**, **Residualverbindungen** und **LayerNorm**.

<!--en:dc9c827f-->
::: {.slide title="Zusammenfassung: Operationen der linearen Algebra, Formen und Anwendungen"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:78c9afef-->
| Operation | Ausdruck | Formregel | FLOPs | Bedeutung im Deep Learning |

<!--en:34b7c196-->
| **Hadamard-Produkt** | $\mathbf{A} \odot \mathbf{B}$ (`A * B`) | $(m, n) \to (m, n)$ | $mn$ | Steuerung durch Gates in LSTMs und SwiGLU, Masken |

<!--en:df28d43e-->
| **Achsenreduktion** | `sum(axis=1, keepdims=True)` | $(m, n) \to (m, 1)$ | $mn$ | Softmax-Normalisierung, LayerNorm |

<!--en:27eb2724-->
| **Skalarprodukt und Kosinus** | $\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|_2\|\mathbf{y}\|_2\cos\theta$ | $(d,), (d,) \to ()$ | $2d$ | Logit eines Neurons, Ähnlichkeit von Abfrage und Schlüssel |

<!--en:b49749e0-->
| **Matrix-Vektor-Produkt** | $\mathbf{A}\mathbf{x}$ (`A @ x`) | $(m, n) \times (n,) \to (m,)$ | $2mn$ | Lineare Abbildung eines Beispiels |

<!--en:20ce1c9b-->
| **Matrix-Matrix-Produkt** | $\mathbf{C} = \mathbf{X}\mathbf{W}$ (`X @ W`) | $(n, k) \times (k, m) \to (n, m)$ | $2nkm$ | Vollständig verbundene Schichten und Attention-Schichten für Batches |

<!--en:2b77318a-->
| **Normen ($\ell_2, \ell_1, \textrm{F}$)** | $\|\mathbf{x}\|_2, \|\mathbf{x}\|_1, \|\mathbf{W}\|_\textrm{F}$ | Beliebiger Tensor $\to ()$ | $2N$ | MSE-/MAE-Verlust, Gewichtszerfall, Begrenzung |

<!--en:555bc743-->
| **Eigenwertzerlegung** | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v} \Rightarrow \mathbf{A}^k\mathbf{v} = \lambda^k\mathbf{v}$ | $(n, n) \to (\lambda_i, \mathbf{v}_i)$ | $O(n^3)$ | Explodierende oder verschwindende Signale, Krümmung |
