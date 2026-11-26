<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=de rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Faltung als Matrizenmultiplikation

<!--en:5bdffcd7 kind=paragraph-->
While the nested loops in `corr2d` make the math transparent, visiting one
output pixel at a time in Python is far too slow for real workloads. A classic
remedy, and still one of the algorithms that libraries choose from, is to recast
cross-correlation as a single large matrix multiplication, the exact operation
GPUs execute at peak speed. Notice that every
output entry in :numref:`fig_correlation` is simply the dot product of the
flattened $k_\textrm{h} k_\textrm{w}$-element kernel with the corresponding
flattened input window. Extracting all
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ windows and
stacking them as rows yields a patch matrix; multiplying it by the flattened
kernel computes the entire cross-correlation in one shot. This standard layout
transformation is called *im2col* ("image to columns", or rows here).
:numref:`fig_conv_im2col` carries it out on the example of
:numref:`fig_correlation`.
<!--tr:5bdffcd7-->
Die verschachtelten Schleifen in `corr2d` machen die Rechnung anschaulich. Für den praktischen Einsatz ist es aber viel zu langsam, in Python jedes Ausgabepixel einzeln zu berechnen. Ein klassischer Ansatz besteht darin, die Kreuzkorrelation als eine einzige große Matrizenmultiplikation zu formulieren. Diese Operation führen GPUs mit höchster Geschwindigkeit aus. Bibliotheken nutzen diesen Algorithmus bis heute neben anderen Verfahren. Jeder Ausgabeeintrag in :numref:`fig_correlation` ist das Skalarprodukt (engl. dot product) des zum Vektor abgeflachten Faltungskerns mit $k_\textrm{h} k_\textrm{w}$ Elementen und des entsprechenden, ebenfalls abgeflachten Eingabeausschnitts. Wir entnehmen der Eingabe alle $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ Fensterausschnitte und ordnen sie als Zeilen einer Ausschnittsmatrix (engl. patch matrix) an. Multiplizieren wir diese Matrix mit dem abgeflachten Faltungskern, erhalten wir die gesamte Kreuzkorrelation auf einmal. Diese übliche Umordnung heißt *im2col* („Bild in Spalten“, hier allerdings in Zeilen). :numref:`fig_conv_im2col` zeigt sie am Beispiel aus :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![Die Kreuzkorrelation aus :numref:`fig_correlation` als einziges Matrixprodukt. Jeder der vier $2 \times 2$-Eingabeausschnitte bildet eine Zeile der Ausschnittsmatrix. Die Multiplikation mit dem abgeflachten Faltungskern ergibt die Ausgabe als Vektor. Die schattierten Zellen zeigen, wie der erste Ausschnitt in seine Zeile eingeht und daraus der Ausgabewert 19 entsteht. Das mittlere Pixel mit dem Wert 4 liegt in allen vier Fenstern und erscheint daher in jeder Zeile.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Dafür brauchen wir zusätzlichen Arbeitsspeicher, denn jedes innen liegende Pixel wird in bis zu $k_\textrm{h} k_\textrm{w}$ Ausschnitte kopiert. Am Beispiel aus :numref:`fig_correlation` prüfen wir unten, ob die Berechnung mit einem einzigen Matrixprodukt genau mit `corr2d` übereinstimmt.

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Die Abweichung ist überall null. Ein einziges Matrixprodukt liefert genau dasselbe Ergebnis wie `corr2d`.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Dieselbe Rechnung lässt sich auch aufstellen, indem wir den *Faltungskern* anstelle der Eingabe in einer dünn besetzten Bandmatrix $\mathbf{W}$ anordnen. Diese Matrix multiplizieren wir mit dem als Vektor geschriebenen Bild. Das entspricht in zwei Dimensionen genau Teil (c) von :numref:`fig_why_conv_weight_sharing`, in dem sich dieselben wenigen Faltungsgewichte in jeder Zeile wiederholen. Damit sehen wir die Faltung als vollständig verbundene Schicht mit besonderer Struktur und sind wieder bei der Herleitung aus :numref:`sec_why-conv`. In den Übungen konstruieren wir $\mathbf{W}$ explizit.

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Merkmalskarten und rezeptive Felder

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Eine einzelne Faltungsschicht sieht pro Ausgabeeintrag nur einen Ausschnitt von der Größe ihres Faltungskerns. Um ganze Objekte zu erkennen, muss ein Netz Informationen aus deutlich größeren Bereichen zusammenführen. Aus :numref:`subsec_why-conv-channels` kennen wir die Ausgabe einer Faltungsschicht als *Merkmalskarte* (engl. feature map). Ein Beispiel ist unsere Kantenkarte `Y` für vertikale Kanten. Das *rezeptive Feld* eines Elements $x$ in einer Merkmalskarte ist der Bereich im Eingabebild oder in einer früheren Schicht, dessen Werte $x$ im Vorwärtsdurchlauf beeinflussen können. Der Begriff stammt aus der Neurophysiologie, siehe :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Übereinanderliegende Schichten

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
Der schattierte Ausgabeeintrag in :numref:`fig_correlation` ist `19`. Sein rezeptives Feld der Größe $2 \times 2$ umfasst die vier Pixel links oben in $\mathbf{X}$. Darauf folgt eine *zweite* $2 \times 2$-Faltungsschicht. Aus der $2 \times 2$-Ausgabe $\mathbf{Y}$ berechnet sie einen einzelnen Skalar $z$. Dieser Wert $z$ hängt von allen vier Elementen von $\mathbf{Y}$ ab. Zusammen hängen diese von allen neun Elementen von $\mathbf{X}$ ab. Das rezeptive Feld wächst so stetig mit der Anzahl der Schichten.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Wie schnell wächst das Feld? Jede $k \times k$-Faltungsschicht erweitert das rezeptive Feld gegenüber dem eines einzelnen Eingabeelements um $k - 1$ Pixel pro Achse. Schalten wir $L$ Schichten mit den Kerngrößen $k_1, \ldots, k_L$ hintereinander, ergibt sich daraus die Seitenlänge des rezeptiven Feldes.

<!--en:b302e3ef kind=paragraph-->
For a stack of $L$ consecutive $3 \times 3$ layers, each layer adds 2 pixels,
covering $(2L + 1) \times (2L + 1)$ inputs: $5 \times 5$ after two layers and
$7 \times 7$ after three (:numref:`fig_conv_receptive_field`). This linear
growth is actually a blessing rather than a reason to reach for huge kernels:
three stacked $3 \times 3$ layers achieve the same $7 \times 7$ receptive field
as a single $7 \times 7$ layer using far fewer parameters ($3 \times 9 = 27$
weights vs. 49 per channel pair) while interleaving three nonlinearities instead
of one.
<!--tr:b302e3ef-->
Bei $L$ aufeinanderfolgenden $3 \times 3$-Schichten verlängert jede Schicht die Seiten des rezeptiven Feldes um 2 Pixel. Damit umfasst es $(2L + 1) \times (2L + 1)$ Eingabeelemente, also $5 \times 5$ nach zwei Schichten und $7 \times 7$ nach drei Schichten (siehe :numref:`fig_conv_receptive_field`). Dieses lineare Wachstum ist gerade ein Vorteil, denn es macht sehr große Faltungskerne entbehrlich. Drei aufeinanderfolgende $3 \times 3$-Schichten erreichen dasselbe rezeptive Feld der Größe $7 \times 7$ wie eine einzige $7 \times 7$-Schicht. Sie brauchen dafür deutlich weniger Parameter ($3 \times 9 = 27$ Gewichte gegenüber 49 pro Kanalpaar) und verknüpfen drei Nichtlinearitäten anstelle einer einzigen.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Rezeptives Feld einer Einheit nach drei eindimensionalen Schichten mit Kerngröße 3. Die schattierten Zellen markieren die Einheiten, von denen die oberste Einheit abhängt. Linien verbinden diese Einheiten jeweils mit ihren Eingaben. Jede Schicht vergrößert das rezeptive Feld um 2, sodass es schließlich 7 Eingaben umfasst.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Effektive rezeptive Felder

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Die Größe $r$ beschreibt das *theoretische* rezeptive Feld. Dazu gehört jedes Eingabepixel, das über *irgendeinen* Pfad mit der Ausgabeeinheit verbunden ist. Wie stark ein Pixel tatsächlich beiträgt, geht daraus nicht hervor. :citet:`Luo.Li.Urtasun.ea.2016` haben die Gradienten nach der Eingabe direkt gemessen. Dabei zeigten sie, dass der Einfluss mit wachsendem Abstand von der Mitte des *effektiven* rezeptiven Feldes stark abnimmt, ungefähr entlang einer gaußschen Glockenkurve. Das gilt für untrainierte wie für trainierte Netze. Einheiten in tiefen Schichten können theoretisch große Teile des Bildes erfassen. Besonders empfindlich reagieren sie aber auf Änderungen nahe der Mitte.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Die Anzahl der Pfade erklärt die Glockenform. In einer Dimension führt vom Rand des rezeptiven Feldes nur ein Pfad zur Ausgabeeinheit. Von der Mitte führen dagegen viele Pfade durch die dazwischenliegenden Schichten. :numref:`fig_conv_rf_paths` zählt sie für die Schichten aus :numref:`fig_conv_receptive_field`. Die mittlere Eingabe erreicht die oberste Einheit auf sieben Pfaden, jede Eingabe am Rand auf einem. Haben alle Gewichte einen ähnlichen Betrag, wächst der Einfluss einer Eingabe mit der Anzahl ihrer Pfade.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Anzahl der Pfade von jeder Einheit zur obersten Einheit aus :numref:`fig_conv_receptive_field` (drei eindimensionale Schichten mit Kerngröße 3). Jede Zahl ist die Summe der Pfadzahlen der Einheiten, die ihre Eingabe von dieser Zelle erhalten. Dunklere Zellen haben mehr Pfade als die übrigen Zellen derselben Schicht. Die Balken geben erneut die Pfadzahlen der Eingabezeile an: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Bei unserer einfachen Schicht bleiben zwei praktische Probleme. Faltungen ohne Padding verkleinern die räumlichen Abmessungen und berücksichtigen Randpixel seltener. Zudem erkennt ein einzelner Faltungskern nur ein Muster. Die räumlichen Abmessungen und die Behandlung der Ränder besprechen wir als Nächstes in :numref:`sec_padding`, Merkmalskarten mit mehreren Kanälen in :numref:`sec_channels`.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Zusammenfassung

<!--en:92688a0c kind=paragraph-->
A convolutional layer applies two-dimensional cross-correlation with a learned
kernel plus a scalar bias, sliding the kernel across every valid window of the
input. A simple $1 \times 2$ finite-difference kernel `[1, -1]` detects vertical
edges anywhere in an image, and gradient descent recovers those exact weights
from input–output data alone, just as deep CNNs learn entire hierarchies of
filters end to end. Mathematical convolution flips the kernel, making no
difference once kernels are learned. A convolution can be evaluated as a single
matrix multiplication (*im2col*), and stacking $L$ convolutional
layers widens each unit's receptive field linearly by $k_i - 1$ per layer.
<!--tr:92688a0c-->
Eine Faltungsschicht berechnet die zweidimensionale Kreuzkorrelation mit einem gelernten Faltungskern und addiert einen skalaren Bias. Dazu verschiebt sie den Faltungskern über alle gültigen Fensterpositionen der Eingabe. Ein einfacher $1 \times 2$-Faltungskern für endliche Differenzen, `[1, -1]`, erkennt vertikale Kanten überall im Bild. Allein aus Eingabe- und Ausgabedaten kann der Gradientenabstieg genau diese Gewichte ermitteln. Auf ähnliche Weise lernen tiefe CNNs ganze Hierarchien von Filtern vom Eingang bis zur Ausgabe. Bei der mathematischen Faltung wird der Faltungskern umgedreht. Für die lernbaren Faltungskerne spielt das keine Rolle. Eine Faltung lässt sich mit einer einzigen Matrizenmultiplikation (*im2col*) berechnen. Schalten wir $L$ Faltungsschichten hintereinander, wächst das rezeptive Feld jeder Einheit linear um $k_i - 1$ pro Schicht.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Schnelle Faltung durch Matrizenmultiplikation"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Die im2col-Umordnung vom Bild in Spalten]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Eine dichte Matrizenmultiplikation (GEMM):** Jedes $k_\textrm{h} \times k_\textrm{w}$-Fenster als Zeile von `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ schreiben und die Matrix mit $\textrm{vec}(\mathbf{K})$ multiplizieren.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Geschwindigkeit gegen Speicherbedarf:** Höchstgeschwindigkeit auf GPU-Tensor-Cores bei $k_\textrm{h} k_\textrm{w}\times$ so viel Speicher für die kopierten Ausschnitte.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Alternative Darstellung des Faltungskerns als Bandmatrix"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Multiplikation mit einer Toeplitz-Gewichtsmatrix]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Bei dieser Darstellung den *Faltungskern* als dünn besetzte Bandmatrix $\mathbf{W}$ schreiben und mit dem Eingabevektor $\textrm{vec}(\mathbf{X})$ multiplizieren. So muss das Eingabebild nicht in überlappende Ausschnitte zerlegt werden.

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Matrix-Vektor-Form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, wobei $\mathbf{W}$ genau $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ Zeilen und $n_\textrm{h} n_\textrm{w}$ Spalten hat.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Bezug zu §{sec_why-conv} und Backpropagation:** Jede Zeile von $\mathbf{W}$ enthält dieselben $k_\textrm{h} k_\textrm{w}$ Einträge des Faltungskerns, spaltenweise verschoben (Teil c). Für den Rückwärtsdurchlauf nach $\mathbf{X}$ genügt die Multiplikation mit $\mathbf{W}^\top$ (eine transponierte Faltung).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Merkmalskarten und rezeptive Felder in tiefen Schichten"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Lineares Wachstum des rezeptiven Feldes über mehrere Schichten]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
Das **rezeptive Feld** einer Einheit ist der Bereich im ursprünglichen Eingabebild, dessen Pixel ihre Aktivierung im Vorwärtsdurchlauf beeinflussen können.

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Eine Schicht ($L=1$):** Ein $k_1 \times k_1$-Faltungskern sieht einen $k_1 \times k_1$-Eingabeausschnitt.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **$L$ Schichten mit Stride 1:** Schicht $i$ mit Kerngröße $k_i$ reicht pro Achse $k_i - 1$ Pixel weiter als ein einzelnes Element der Schicht $i-1$.
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ aufeinanderfolgende $3 \times 3$-Schichten:** Pro Schicht kommen $3 - 1 = 2$ Pixel zur Seitenlänge hinzu. Das ergibt $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Vorteile mehrerer 3×3-Faltungskerne gegenüber einem großen Faltungskern"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Weniger Parameter und höhere Ausdruckskraft]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Gleiches rezeptives Feld von $7 \times 7$:** Drei aufeinanderfolgende $3 \times 3$-Schichten ($r = 1 + 2 + 2 + 2 = 7$) sehen genau denselben $7 \times 7$-Ausschnitt wie eine $7 \times 7$-Schicht.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% weniger Parameter:** $3 \times 9 c^2 = 27 c^2$ Gewichte statt $49 c^2$ bei $c$ Kanälen.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Höhere Ausdruckskraft:** **3 nichtlineare Aktivierungen** hintereinander statt 1 (das Grundprinzip von VGG in §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Theoretische und effektive rezeptive Felder"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Gaußförmige Abnahme der Empfindlichkeit mit dem Abstand zur Mitte]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Theoretisches rezeptives Feld ($r = 2L+1$):** Begrenzungsrahmen aller Pixel mit *mindestens einem* Pfad zur Ausgabeeinheit.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Zentraler Grenzwertsatz:** Bei Pixeln in der Mitte wächst die Pfadzahl exponentiell mit der Tiefe. Eckpixel haben nur **1 Pfad**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Effektives rezeptives Feld (Luo et al., 2016):** Der Betrag des Eingabegradienten $\partial y / \partial \mathbf{X}$ nimmt nach außen entlang einer gaußschen Glockenkurve mit Standardabweichung $\sigma \propto \sqrt{L}$ ab.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Verständnisfragen zu Kreuzkorrelation und rezeptiven Feldern"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Aufgaben zu Ausgabeformen, `im2col` und zur Größe rezeptiver Felder

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Matrixabmessungen bei im2col:** Ein einkanaliger $3 \times 3$-Faltungskern wird ohne Padding auf ein $10 \times 10$-Bild angewandt. Dazu dient `im2col`. Welche räumliche Form hat die Ausgabe? Welche Abmessungen haben die Matrix `patches` und der abgeflachte Faltungskern?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Schichttiefe für Fashion-MNIST:** Wie viele $3 \times 3$-Schichten sind bei Stride 1 nötig, damit das theoretische rezeptive Feld einer Einheit ein ganzes $28 \times 28$-Bild abdeckt? Wie viele bei $5 \times 5$-Schichten?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Anfangsverlust bei verschobener Kante:** Warum lag der Anfangsverlust beim Lernen vertikaler Kanten im $6 \times 8$-Bild bei $\approx 12$? Wie groß wäre er, wenn der schwarze Streifen bis an den rechten Bildrand reichte (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Lösungen:**
(1) Die Ausgabe hat die Form $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ Ausgabewerte). Die Matrix `patches` hat die Form $(64, 9)$ und wird mit dem $(9, 1)$-Faltungskernvektor multipliziert.
(2) Für $3 \times 3$ gilt $2L + 1 \ge 28 \implies L = 14$ Schichten. Für $5 \times 5$ gilt $4L + 1 \ge 28 \implies L = 7$ Schichten.
(3) Jede der 6 Zeilen hatte zwei Kanten mit Werten von $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Reicht der schwarze Streifen bis an den rechten Rand, entfällt der rechte Übergang von Schwarz zu Weiß. Pro Zeile bleibt eine Kante. Der Anfangsverlust halbiert sich auf $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Zusammenfassung der wichtigsten Ergebnisse"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Überblick über den Abschnitt]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2D-Kreuzkorrelation und Lernen von Kantenfiltern:** $\mathbf{K}$ über alle gültigen Fenster schieben ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). SGD nähert sich $[1, -1]$ in 9 Schritten.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Matrizenmultiplikation mit im2col:** $k_\textrm{h} k_\textrm{w}\times$ so viel Speicher für kopierte Ausschnitte ermöglicht maximalen GEMM-Durchsatz auf der GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Rezeptive Felder über mehrere Schichten:** $r = 1 + \sum (k_i - 1) = 2L+1$ für aufeinanderfolgende $3 \times 3$-Schichten. Die Empfindlichkeit nimmt von der Mitte aus gaußförmig ab.

=== DECISIONS ===
- „patch matrix“ → „Ausschnittsmatrix“, entsprechend „patch“ → „Ausschnitt“.
- „banded matrix“ → „Bandmatrix“, „sparse“ → „dünn besetzt“.
- „input gradient“ → „Gradient nach der Eingabe“ oder als Kurzform „Eingabegradient“.

=== NOTES ===
- eed4e2a0: Ein einzelner Pfad vom Rand zur Ausgabe gilt für das hier gemeinte eindimensionale Beispiel. In zwei Dimensionen gilt das für die Ecken.
- 405695e8: Die Aussage über schrumpfende Dimensionen und seltener erfasste Randpixel bezieht sich auf Faltungskerne, die in der betreffenden Dimension größer als 1 sind.
- 92688a0c, 00011fb7: Das Experiment nähert die Zielgewichte an. Die Aussage über die exakten Gewichte beschreibt den Zielwert, keine Gleichheit nach endlich vielen Schritten. Zum Ausdruck „SGD“ siehe die Anmerkung zu 456670c5.
- 16976d18, 250d6877: Die angegebenen Tiefen ergeben theoretisch Seitenlänge 29. Damit eine Ausgabe mit einem so großen rezeptiven Feld bei einer Eingabe mit Seitenlänge 28 existiert, braucht es Padding.
