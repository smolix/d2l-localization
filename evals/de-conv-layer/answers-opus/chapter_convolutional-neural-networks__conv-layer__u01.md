<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=de rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Faltungen für Bilder

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
In :numref:`sec_why-conv` haben wir die Faltungsschicht (engl. convolutional
layer) auf dem Papier aus zwei geometrischen Prinzipien hergeleitet:
Translationsäquivarianz und Lokalität. Nun sehen wir uns an, was diese Formel
auf einem Bild tatsächlich berechnet und ob der Gradientenabstieg nützliche
Faltungskerne (engl. kernel) automatisch aus Daten finden kann.

<!--en:55f0a41b kind=paragraph-->
Consider locating vertical edges where a bright region meets a dark one.
Subtracting each pixel's right-hand neighbor ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)
yields zero inside uniform regions, $+1$ where white transitions to black, and
$-1$ where black returns to white. Sweeping the two-weight detector $(1, -1)$
across every valid image location produces an *edge map*. In this section, we
implement two-dimensional cross-correlation, test our hand-crafted edge
detector, let gradient descent *learn* $(1, -1)$ directly from data, show how
a convolution becomes a single matrix multiplication (*im2col*), and
derive how stacking layers widens a unit's receptive field.
<!--tr:55f0a41b-->
Nehmen wir an, wir wollen vertikale Kanten finden, an denen ein heller Bereich
auf einen dunklen trifft. Ziehen wir von jedem Pixel seinen rechten Nachbarn ab
($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), so erhalten wir null innerhalb
gleichförmiger Bereiche, $+1$ dort, wo Weiß in Schwarz übergeht, und
$-1$ dort, wo Schwarz wieder in Weiß übergeht. Schieben wir den Detektor $(1, -1)$
mit seinen zwei Gewichten über jede gültige Position im Bild, so entsteht eine
*Kantenkarte* (engl. edge map). In diesem Abschnitt implementieren wir die
zweidimensionale Kreuzkorrelation, testen unseren von Hand entworfenen
Kantendetektor und lassen den Gradientenabstieg $(1, -1)$ direkt aus Daten
*lernen*. Außerdem zeigen wir, wie eine Faltung zu einer einzigen
Matrizenmultiplikation wird (*im2col*), und leiten her, wie das Stapeln von
Schichten das rezeptive Feld (engl. receptive field) einer Einheit vergrößert.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Kreuzkorrelation

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### Die Operation

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Lassen wir Kanäle vorerst beiseite. Die Eingabe $\mathbf{X}$ sei eine
$n_\textrm{h} \times n_\textrm{w}$-Matrix und der Faltungskern $\mathbf{K}$ eine
$k_\textrm{h} \times k_\textrm{w}$-Matrix. Der Faltungskern definiert ein
gleitendes *Fenster* der Form $k_\textrm{h} \times k_\textrm{w}$. Legen wir die
linke obere Ecke des Fensters auf $(i, j)$, so multiplizieren wir den
überdeckten Ausschnitt der Eingabe elementweise mit $\mathbf{K}$ und summieren
die Produkte. Das ergibt den Ausgabeeintrag an der Stelle $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Das ist die *zweidimensionale Kreuzkorrelation* von $\mathbf{X}$ mit
$\mathbf{K}$, geschrieben $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$, mit
nullbasierten Indizes wie im Code. In :numref:`sec_why-conv` lagen die
Versätze symmetrisch um die Mitte, von $-\Delta$ bis $\Delta$. Die Verankerung
an der linken oberen Ecke unterscheidet sich davon nur durch eine feste
Verschiebung um $\Delta$. Dafür kommt sie ganz natürlich mit Faltungskernen
gerader Größe zurecht (etwa einem $2 \times 2$-Faltungskern oder unserem
$1 \times 2$-Kantendetektor), die kein mittleres Pixel haben.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Zweidimensionale Kreuzkorrelation. Die schattierten Teile sind das erste Ausgabeelement sowie die Elemente von Eingabe und Faltungskern, aus denen es berechnet wird: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
In :numref:`fig_correlation` schieben wir einen $2 \times 2$-Faltungskern
pixelweise über eine $3 \times 3$-Eingabe, von links nach rechts und von oben
nach unten. Das Fenster passt an genau vier Positionen vollständig hinein, und
so ergibt sich die $2 \times 2$-Ausgabe:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Die Ausgabe schrumpft dabei entlang jeder Achse etwas, weil der Faltungskern
vollständig innerhalb der Eingabe bleiben muss. Ein Fenster der Breite $k$
passt in eine Länge $n$ an $n - k + 1$ gültigen Positionen. Die Ausgabe hat
daher die Form

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Eine $3 \times 3$-Eingabe mit einem $2 \times 2$-Faltungskern ergibt also
$2 \times 2$, ein $6 \times 8$-Bild mit einem $1 \times 2$-Faltungskern ergibt
$6 \times 7$ und ein $28 \times 28$-Bild aus Fashion-MNIST mit einem
$5 \times 5$-Faltungskern ergibt $24 \times 24$. Wollen wir die räumlichen
Dimensionen erhalten, umgeben wir die Eingabe mit einem Rand aus Nullen
(*Padding*). Das Padding untersuchen wir zusammen mit Strides und Dilatation in
:numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Implementierung

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Wir implementieren :eqref:`eq_corr2d` direkt in `corr2d`, mit zwei Schleifen
über alle Ausgabekoordinaten.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Um `corr2d` mit unserer Handrechnung aus :numref:`fig_correlation` abzugleichen,
legen wir den $3 \times 3$-Eingabetensor `X` und den $2 \times 2$-Faltungskern
`K` an und werten `corr2d(X, K)` aus.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Die Ausgabe reproduziert :numref:`fig_correlation` Eintrag für Eintrag. (Wie
:numref:`sec_why-conv` gezeigt hat, lernt die streng mathematische Faltung mit
einem gespiegelten Faltungskern $\mathbf{K}'$ genau dieselbe Familie von
Schichten. Daher folgen wir der üblichen Terminologie des Deep Learning und
nennen die Kreuzkorrelation „Faltung“.)

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Faltungsschichten

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Die Schicht

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Eine zweidimensionale Faltungsschicht bildet die Kreuzkorrelation ihrer Eingabe
mit einem gelernten Faltungskern und addiert einen skalaren Bias. Wie die
Gewichte unseres MLP in :numref:`chap_perceptrons` initialisieren wir den
Faltungskern mit Werten aus $\mathcal{N}(0, 0.01^2)$ und den Bias mit null. Die
Klasse `Conv2D` unten registriert beide Parameter und ruft in ihrem
Vorwärtsdurchlauf `corr2d` auf. Eine Schicht mit einem Faltungskern der Höhe
$h$ und der Breite $w$ heißt $h \times w$-Faltungsschicht.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Anders als eine vollständig verbundene Schicht ist eine Faltungsschicht nicht an
eine feste Eingabeauflösung gebunden, denn ihre Parameter hängen nur von der
Kerngröße ab. Genau dieselbe $2 \times 2$-Schicht (insgesamt 5 Parameter)
verarbeitet sowohl unseren $3 \times 3$-Tensor `X` als auch ein
$6 \times 8$-Bild und liefert Ausgaben der Formen $2 \times 2$ und
$5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Kantenerkennung

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Nun testen wir den Detektor für vertikale Kanten aus der Einleitung an einem
synthetischen $6 \times 8$-Bild, dessen mittlere vier Spalten schwarz (`0`) und
dessen äußere Spalten weiß (`1`) sind.

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
Wir definieren den $1 \times 2$-Faltungskern

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
An jeder Position $(i, j)$ berechnet er die horizontale *endliche Differenz*
(engl. finite difference) $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$. Sie ist
(bis auf das Vorzeichen) eine diskrete Näherung der horizontalen Ableitung der
Helligkeit: In gleichförmigen Bereichen verschwindet sie, und überall dort, wo
die Intensität springt, schlägt sie aus.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Die Kreuzkorrelation des $6 \times 8$-Streifenbilds `X` mit dem
$1 \times 2$-Faltungskern `K` für horizontale Differenzen liefert über
`corr2d(X, K)` die $6 \times 7$-Kantenkarte `Y`:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Tatsächlich liefert `Y` den Wert `+1` entlang der Grenze von Weiß nach Schwarz,
`-1` entlang der Grenze von Schwarz nach Weiß und `0` überall sonst. Wenden wir
genau denselben Faltungskern auf das transponierte Bild an (aus dem vertikalen
Streifen wird so ein horizontaler), erhalten wir lauter Nullen:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Offensichtlich reagiert `[1, -1]` ausschließlich auf vertikale Kanten. Um
horizontale Kanten, Diagonalen, Ecken oder Texturen zu erkennen, brauchen wir
ganz andere Faltungskerne.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Lernen eines Faltungskerns

<!--en:61cacfaf kind=paragraph-->
Hand-crafting `[1, -1]` is easy when we know the exact feature we want, yet no
engineer can specify thousands of intermediate filters across a deep network by
hand. Can gradient descent recover the kernel `[1, -1]` from the input–output
pair `(X, Y)` alone? We test this using each framework's built-in 2D
convolutional layer (with bias omitted and weights initialized from
$\mathcal{N}(0, 0.01^2)$), minimizing squared error summed across all 42 output
entries over 10 gradient steps at learning rate $0.03$. Because built-in layers
expect 4D tensors including batch and channel axes (:numref:`sec_channels`), we
reshape `X` and `Y` accordingly.
<!--tr:61cacfaf-->
Den Faltungskern `[1, -1]` von Hand zu entwerfen ist leicht, wenn wir genau
wissen, welches Merkmal wir suchen. Tausende von Filtern in den Zwischenschichten
eines tiefen Netzes kann jedoch niemand von Hand festlegen. Kann der
Gradientenabstieg den Faltungskern `[1, -1]` allein aus dem Eingabe-Ausgabe-Paar
`(X, Y)` wiederfinden? Das testen wir mit der eingebauten 2D-Faltungsschicht
des jeweiligen Frameworks (ohne Bias, die Gewichte initialisiert mit Werten aus
$\mathcal{N}(0, 0.01^2)$). Wir minimieren den quadratischen Fehler, summiert
über alle 42 Ausgabeeinträge, mit 10 Gradientenschritten bei der Lernrate
$0.03$. Da eingebaute Schichten 4D-Tensoren mit Batch- und Kanalachse erwarten
(:numref:`sec_channels`), bringen wir `X` und `Y` in die passende Form.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Anfangs liegt der Verlust bei etwa 12 (der zufällige Faltungskern nahe null
verfehlt alle 12 Kantenpixel mit Wert $\pm 1$ um jeweils rund 1), und schon
innerhalb von neun Schritten fällt er unter 0.01. Ein Blick auf den gelernten
Faltungskern bestätigt, was der Gradientenabstieg gefunden hat:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Beide Einträge liegen bis auf rund 0.02 bei `[1, -1]`. Nichts in dieser
Schleife wusste irgendetwas über Kanten: Mit einem Verlustsignal hat der
Gradientenabstieg den Filter selbstständig gefunden. Hier haben wir die
einzelne Schicht direkt mit `Y` überwacht. In einem tiefen CNN lernt die
Backpropagation die Faltungskerne aller Schichten durchgängig, allein aus dem
Verlust der Aufgabe am Ende des Netzes.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Faltungen für Bilder<br>**2D-Kreuzkorrelation · Kantenerkennung · gelernte Faltungskerne · im2col · rezeptive Felder**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Lokale Kanten im ganzen Bild erkennen"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Motivation und zentrales Denkmodell]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Synthetisches $6 \times 8$-Testbild:** Weiße äußere Spalten (`1`) umgeben ein 4 Spalten breites, schwarzes vertikales Band (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Horizontale endliche Differenz:** $\mathbf{K} = [1, -1]$ über das Bild geschoben berechnet $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$: `+1` an Kanten von Weiß nach Schwarz, `-1` an Kanten von Schwarz nach Weiß.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Lernen statt Handentwurf:** Tiefe CNNs finden Tausende solcher Faltungskerne automatisch per Gradientenabstieg.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="Die 2D-Kreuzkorrelation"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Innere Produkte im gleitenden Fenster]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Für eine $n_\textrm{h} \times n_\textrm{w}$-Eingabematrix $\mathbf{X}$ und einen $k_\textrm{h} \times k_\textrm{w}$-Faltungskern $\mathbf{K}$ legt die 2D-Kreuzkorrelation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ die linke obere Ecke des Fensters auf $(i, j)$ und summiert die elementweisen Produkte:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Anker links oben:** Nullbasierte Indizes $a, b \ge 0$ entsprechen der Array-Indizierung im Code und bewältigen Faltungskerne gerader Größe (etwa $2 \times 2$ oder $1 \times 2$) problemlos.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Schattierte Ausgabe links oben $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Gültige Fensterpositionen und räumliche Form der Ausgabe"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Warum Faltungen ohne Padding das Raster verkleinern]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Gültige Fensterpositionen:** Ein $k_\textrm{h} \times k_\textrm{w}$-Fenster passt in ein $n_\textrm{h} \times n_\textrm{w}$-Raster an $n - k + 1$ Positionen pro Achse.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Beispiele für Formen:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Die 2D-Kreuzkorrelation im Code"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Direkte Implementierung mit verschachtelten Schleifen]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
Die Funktion `corr2d` legt den $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$-Ausgabetensor an und berechnet jeden Eintrag als Summe von `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Jede Iteration schneidet ein $k_\textrm{h} \times k_\textrm{w}$-Teilfenster aus `X` aus, multipliziert es elementweise mit `K` und reduziert es mit `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Prüfung von corr2d und Äquivalenz zum gespiegelten Faltungskern"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Kreuzkorrelation vs. mathematische Faltung]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Um 180° gespiegelter Faltungskern $\mathbf{K}'$:** Die Kreuzkorrelation mit $\mathbf{K}$ ist gleich der Faltung mit $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Gleiche gelernte Klasse:** Frameworks berechnen die Kreuzkorrelation und nennen sie Faltung.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Ein Modul für die 2D-Faltungsschicht"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Lernbare Gewichte des Faltungskerns plus skalarer Bias]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Eine Faltungsschicht umschließt `corr2d` mit einem lernbaren $k_\textrm{h} \times k_\textrm{w}$-Gewichtsparameter (initialisiert aus $\mathcal{N}(0, 0.01^2)$) und einem skalaren Bias (initialisiert mit $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Auflösungsunabhängigkeit von Faltungsschichten"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Feste Parameter für beliebige Eingabegrößen]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Unabhängig von $H \times W$:** Genau dieselben 5 Parameter ($2 \times 2$-Faltungskern + 1 Bias) verarbeiten $3 \times 3$-, $6 \times 8$- oder Megapixel-Eingaben, ohne die Gewichte umzuformen.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Ein synthetisches Bild mit vertikalen Kanten"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Ein binäres 6×8-Testmuster]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Das $6 \times 8$-Testbild `X` erhält Einsen am linken und rechten Rand und Nullen in den mittleren vier Spalten (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Stückweise konstante Spalten:** Konstante Intensität (`1` oder `0`) innerhalb der Spalten $0\text{--}1$, $2\text{--}5$ und $6\text{--}7$, Sprünge nur zwischen den Spalten $1 \to 2$ und $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Der Faltungskern für horizontale endliche Differenzen"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Vertikale Kanten erkennen mit K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
Die Kreuzkorrelation von `X` mit dem $1 \times 2$-Faltungskern $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ berechnet $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ für jedes horizontale Paar:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Kantenkarte mit Vorzeichen:** Spalte 1 liefert `+1` (Übergang von Weiß nach Schwarz), Spalte 5 liefert `-1` (Übergang von Schwarz nach Weiß), alle konstanten inneren Spalten ergeben `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Orientierungsselektivität von Kantendetektoren"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Warum ein Faltungskern nicht alle Orientierungen erkennt]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Blind für horizontale Kanten:** Nach dem Transponieren von `X` (`X.T`) sind horizontale Nachbarn gleich, daher gibt $[1, -1]$ nur Nullen aus.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Filterbänke (§{sec_channels}):** Schichten für Bilddaten lernen parallele Kanäle für horizontale, vertikale und diagonale Orientierungen.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Einen Kantendetektor per Gradientenabstieg lernen"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Eine eingebaute Faltungsschicht auf (X, Y) überwacht trainieren]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4D-Tensoren des Frameworks:** `X` und `Y` in 4D-Tensoren umformen (ein Beispiel, ein Kanal) und $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ bei $\eta = 0.03$ minimieren.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Ein Blick auf die gelernten Gewichte des Faltungskerns"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Warum der Verlust bei 12 beginnt und der Faltungskern gegen [1, -1] konvergiert]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Warum der Verlust bei $12.0$ beginnt:** Gewichte nahe null sagen $\hat{Y} \approx 0$ vorher und verfehlen die zwölf Kantenpixel mit $\pm 1$ in `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Konvergenz in 9 Schritten:** SGD findet $[0.99, -0.99]$ automatisch aus den Daten.

=== DECISIONS ===
- "edge map" → „Kantenkarte“ (die); "edge detector" → „Kantendetektor“; "edge detection" → „Kantenerkennung“
- "window" (sliding window of a convolution) → „Fenster“, "sliding" → „gleitend“
- "patch" (input region covered by a window) → „Ausschnitt“; "patch matrix" → „Matrix der Ausschnitte“ (no compound, no loan)
- "dense layer" → „vollständig verbundene Schicht“
- "unit" (a neuron/activation in a layer) → „Einheit“
- "vertical/horizontal edge" → „vertikale/horizontale Kante“ (not senkrecht/waagerecht)
- "end to end" (adverb) → „durchgängig“; in compounds End-to-End- (style §3.4)
- "kernel" in this page always the convolution sense → „Faltungskern“, also in slides; "kernel size" → „Kerngröße“ (glossary)
- "framework" → „das Framework“, plural „die Frameworks“ (loan, no German term in use)
- English glosses on this page so far: Faltungsschicht (engl. convolutional layer), Faltungskern (engl. kernel), Kantenkarte (engl. edge map), rezeptives Feld (engl. receptive field), endliche Differenz (engl. finite difference); later units do not repeat them

=== NOTES ===
- 7e4ef44b: the English kicker says the loss "converges to [1, -1]"; it is the kernel that converges. Translated the evident meaning ("der Faltungskern gegen [1, -1] konvergiert").
