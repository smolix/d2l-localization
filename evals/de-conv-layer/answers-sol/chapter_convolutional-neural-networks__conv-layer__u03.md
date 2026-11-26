<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=de rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Übungen

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->
1. [code] **Von Hand entworfene Faltungskerne.** Entwerfen Sie einige Faltungskerne von Hand.
    1. Gegeben sei der Richtungsvektor $\mathbf{v} = (v_1, v_2)$. Leiten Sie einen Faltungskern her, der Kanten orthogonal zu $\mathbf{v}$ erkennt, das heißt Kanten in Richtung $(v_2, -v_1)$.
    1. Leiten Sie aus endlichen Differenzen einen Operator für die zweite Ableitung her. Bei welchen Bildstrukturen fällt seine Antwort am stärksten aus?
    1. Wie groß muss ein Faltungskern mindestens sein, um eine Ableitung der Ordnung $d$ anzunähern?
    1. Wie würden Sie einen Faltungskern zum Weichzeichnen entwerfen? Wozu könnte er dienen?
    1. Laden Sie ein Foto und wandeln Sie es in ein einkanaliges Bild um. Reduzieren Sie die Auflösung auf etwa $100 \times 100$ Pixel (`corr2d` arbeitet mit Python-Schleifen). Wenden Sie den Kantendetektor, den Faltungskern zum Weichzeichnen und den Faltungskern zum Schärfen („zweimal die Identität minus den Weichzeichner“) an. Zeigen Sie jedes Ergebnis neben dem Original. Was zeigt die Kantenkarte des Fotos, das die Kantenkarte des schwarzen Streifens nicht zeigen konnte?

    *Nach dem Notebook ConvolutionalNetworks aus Stanford CS231n,
    [Aufgabe 2](https://cs231n.github.io/assignments2024/assignment2/).*

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->
2. [code] **Faltung als Bandmatrix.** Die Ausschnittsmatrix aus :numref:`subsec_conv_matmul` ordnet die Eingabe neu an. Beim umgekehrten Aufbau ordnen wir den Faltungskern neu an. Sei $\mathbf{X}$ eine $4 \times 4$-Eingabe und $\mathbf{K}$ ein $3 \times 3$-Faltungskern. Die Operation $\textrm{vec}(\cdot)$ reiht die Zeilen zu einem Vektor aneinander.
    1. Konstruieren Sie die $4 \times 16$-Matrix $\mathbf{W}$ mit $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$. Dabei ist $\mathbf{Y}$ die Kreuzkorrelation von $\mathbf{X}$ mit $\mathbf{K}$. Wie viele von null verschiedene Einträge hat jede Zeile, und wie viele verschiedene Werte kommen in $\mathbf{W}$ vor?
    1. Erzeugen Sie $\mathbf{W}$ im Code aus `K` und überprüfen Sie, ob das Produkt dasselbe Ergebnis wie `corr2d(X, K)` liefert.
    1. Betrachten Sie eine $n \times n$-Eingabe und einen $k \times k$-Faltungskern. Wie viele Einträge speichert $\mathbf{W}$ als dichtes Feld? Wie viele Einträge von $\mathbf{W}$ sind ungleich null, und wie viele Einträge hat die Ausschnittsmatrix? Wie wächst die jeweilige Anzahl mit $n$?
    1. Behalten Sie das Sparsitätsmuster von $\mathbf{W}$ bei, geben Sie aber jeder Zeile ihre eigenen $k^2$ Werte. So entsteht eine lokal verbundene Schicht. Ihr entspricht der Tensor vierter Ordnung $\mathsf{V}$ aus :numref:`sec_why-conv` in $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$. Er ist auf ein $k \times k$-Fenster der Versätze $(a, b)$ beschränkt. Die Gewichte sind noch unabhängig, bevor die Translationsäquivarianz sie miteinander verknüpft. Bestimmen Sie für eine $n \times n$-Eingabe jeweils die Anzahl der Parameter dieser Schicht, einer vollständig verbundenen Schicht mit denselben Ein- und Ausgabeformen und der Faltung.

    *In Anlehnung an CMU 11-785,
    [Übungsblatt 2, Teil 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->
3. [code] **Effektives rezeptives Feld.** ● Das theoretische rezeptive Feld $r$ aus diesem Abschnitt grenzt ein, welche Eingaben eine Ausgabe beeinflussen *können*. Das effektive rezeptive Feld misst, wie stark sie die Ausgabe tatsächlich beeinflussen :cite:`Luo.Li.Urtasun.ea.2016`. Schalten Sie $L$ einkanalige $3 \times 3$-Faltungsschichten ohne Padding, Bias oder Nichtlinearität hintereinander. Verwenden Sie eine $32 \times 32$-Eingabe und die eingebaute Schicht aus :numref:`subsec_learning_kernel`. Ziehen Sie jeden Eintrag der Faltungskerne unabhängig aus einer Standardnormalverteilung.
    1. Ziehen Sie für $L = 3$ die Faltungskerne einmal. Setzen Sie den Gradienten an einer Ausgabeeinheit nahe der Mitte der letzten Schicht auf eins und propagieren Sie ihn bis zur Eingabe zurück. Plotten Sie den Betrag des Eingabegradienten. Vergleichen Sie seinen Träger mit dem theoretischen rezeptiven Feld der Größe $7 \times 7$. Geben Sie an, wo der Betrag am größten ist.
    1. Mitteln Sie den quadrierten Eingabegradienten über 500 Ziehungen der Faltungskerne, jeweils für $L = 3$ und für $L = 10$. Summieren Sie jede gemittelte Karte über ihre Zeilen und normieren Sie das Spaltenprofil auf die Summe eins. Wie ändert sich das Verhältnis der Standardabweichung des Profils zur theoretischen Seitenlänge $2L + 1$ mit der Tiefe?
    1. :numref:`fig_conv_rf_paths` zählt die Pfade für $L = 3$ in einer Dimension. Zeigen Sie, dass für beliebiges $L$ der über die Zeilen summierte Erwartungswert des quadrierten Eingabegradienten proportional zur Faltung von $L$ Kopien des Vektors $(1, 1, 1)$ ist. Leiten Sie her, wie die Standardabweichung des normierten Profils mit $L$ wächst.

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->
4. [code] **Rückwärtsdurchlauf einer Faltung.** ● Sei $\mathbf{Y}$ die einkanalige Kreuzkorrelation der Eingabe $\mathbf{X}$ mit dem Faltungskern $\mathbf{K}$, berechnet mit `corr2d`. Sei $\ell$ ein skalarer Verlust, der von $\mathbf{X}$ und $\mathbf{K}$ nur über $\mathbf{Y}$ abhängt.
    1. Leiten Sie $\partial \ell/\partial \mathbf{K}$ und $\partial \ell/\partial \mathbf{X}$ aus $\partial \ell/\partial \mathbf{Y}$ her. Orientieren Sie sich an den Gleichungen für die Backpropagation aus :numref:`sec_backprop`.
    1. Implementieren Sie Ihren Ausdruck für $\partial \ell/\partial \mathbf{K}$ ohne automatisches Differenzieren für den Verlust $\ell = \sum_{i,j} G_{ij} Y_{ij}$. Dabei ist $\mathbf{G}$ eine feste Zufallsmatrix mit derselben Form wie $\mathbf{Y}$. Vergleichen Sie den berechneten Gradienten mit dem Gradienten aus zentralen endlichen Differenzen mit der Schrittweite $10^{-2}$. Verwenden Sie kleine, zufällig erzeugte Matrizen `X` und `K` und geben Sie den größten relativen Fehler an.
    1. Vergleichen Sie Ihren Ausdruck für $\partial \ell/\partial \mathbf{K}$ mit dem Gradienten nach dem Faltungskern, den das automatische Differenzieren im Experiment aus :numref:`subsec_learning_kernel` liefert.

    *In Anlehnung an Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Kapitel 6, sowie an das Notebook ConvolutionalNetworks aus Stanford CS231n,
    [Aufgabe 2](https://cs231n.github.io/assignments2024/assignment2/).*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Diskussion](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Diskussion](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Diskussion](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Diskussion](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- „locally connected layer“ → „lokal verbundene Schicht“.
- „sparsity pattern“ → „Sparsitätsmuster“.
- „support“ (einer Funktion) → „Träger“.
- „blur kernel“, „sharpening kernel“ → „Faltungskern zum Weichzeichnen“, „Faltungskern zum Schärfen“.

=== NOTES ===
- 6f4ddec6: „obtain a derivative“ meint im Kontext der endlichen Differenzen eine Näherung der Ableitung.
- 6ab5b67c: Die Glossarwarnung zu „Neural Networks“ betrifft den unveränderten Buchtitel *Neural Networks and Deep Learning*.
