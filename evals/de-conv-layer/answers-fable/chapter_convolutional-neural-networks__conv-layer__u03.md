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
    1. Gegeben sei ein Richtungsvektor $\mathbf{v} = (v_1, v_2)$. Leiten Sie
       einen Faltungskern zur Kantenerkennung her, der Kanten orthogonal zu
       $\mathbf{v}$ erkennt, das heißt Kanten in Richtung $(v_2, -v_1)$.
    1. Leiten Sie einen Operator endlicher Differenzen für die zweite
       Ableitung her. Welche Strukturen in Bildern sprechen am stärksten
       darauf an?
    1. Wie groß muss ein Faltungskern mindestens sein, um eine Ableitung der
       Ordnung $d$ zu erhalten?
    1. Wie würden Sie einen Weichzeichnungskern entwerfen? Wozu könnten Sie
       einen solchen Faltungskern einsetzen wollen?
    1. Laden Sie eine Fotografie, wandeln Sie sie in einen Kanal um und
       verkleinern Sie sie auf etwa $100 \times 100$ Pixel (`corr2d`
       durchläuft Schleifen in Python). Wenden Sie den Kantendetektor, den
       Weichzeichnungskern und den Schärfungskern „zweimal die Identität
       minus Weichzeichnung“ an und zeigen Sie jedes Ergebnis neben dem
       Original. Was zeigt die Kantenkarte einer Fotografie, das die
       Kantenkarte des schwarzen Bandes nicht zeigen konnte?

    *Nach Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    Notebook ConvolutionalNetworks.*

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
2. [code] **Faltung als Bandmatrix.** Die Ausschnittsmatrix aus
   :numref:`subsec_conv_matmul` zerlegt die Eingabe in Ausschnitte. Die
   umgekehrte Konstruktion rollt den Faltungskern aus. Sei $\mathbf{X}$ eine
   $4 \times 4$-Eingabe, $\mathbf{K}$ ein $3 \times 3$-Faltungskern und
   $\textrm{vec}(\cdot)$ das zeilenweise Abflachen.
    1. Konstruieren Sie die $4 \times 16$-Matrix $\mathbf{W}$ mit
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       wobei $\mathbf{Y}$ die Kreuzkorrelation von $\mathbf{X}$ mit
       $\mathbf{K}$ ist. Wie viele von null verschiedene Einträge hat jede
       Zeile, und wie viele verschiedene Werte kommen in $\mathbf{W}$ vor?
    1. Bauen Sie $\mathbf{W}$ im Code aus `K` auf und bestätigen Sie, dass
       das Produkt `corr2d(X, K)` reproduziert.
    1. Zählen Sie für eine $n \times n$-Eingabe und einen
       $k \times k$-Faltungskern die Einträge von $\mathbf{W}$ bei
       Speicherung als dichtes Array, die von null verschiedenen Einträge
       von $\mathbf{W}$ und die Einträge der Ausschnittsmatrix. Wie wächst
       jede dieser Anzahlen mit $n$?
    1. Behalten Sie das Besetzungsmuster von $\mathbf{W}$ bei, geben Sie
       aber jeder Zeile ihre eigenen $k^2$ Werte. Das Ergebnis ist eine
       lokal verbundene Schicht: der Tensor vierter Stufe $\mathsf{V}$ aus
       :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       eingeschränkt auf ein $k \times k$-Fenster von Versätzen $(a, b)$,
       bevor die Translationsäquivarianz die Gewichte koppelt. Zählen Sie
       für eine $n \times n$-Eingabe die Parameter dieser Schicht, die
       einer vollständig verbundenen Schicht mit denselben Eingabe- und
       Ausgabeformen und die der Faltung.

    *Nach CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

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
3. [code] **Effektives rezeptives Feld.** ● Das theoretische rezeptive
   Feld $r$ dieses Abschnitts begrenzt, welche Eingaben eine Ausgabe
   beeinflussen *können*; das effektive rezeptive Feld misst, wie stark sie
   es tatsächlich tun :cite:`Luo.Li.Urtasun.ea.2016`. Bauen Sie einen Stapel
   aus $L$ einkanaligen $3 \times 3$-Faltungsschichten ohne Padding, ohne
   Bias und ohne Nichtlinearität auf einer $32 \times 32$-Eingabe auf und
   verwenden Sie dazu die eingebaute Schicht aus
   :numref:`subsec_learning_kernel`. Ziehen Sie jeden Kerneintrag unabhängig
   aus einer Standardnormalverteilung.
    1. Propagieren Sie für $L = 3$ und eine einzelne Ziehung der
       Faltungskerne einen Einheitsgradienten von einer Ausgabeeinheit nahe
       der Mitte der letzten Schicht zurück zur Eingabe und plotten Sie den
       Betrag des Eingabegradienten. Vergleichen Sie seinen Träger mit dem
       theoretischen rezeptiven Feld von $7 \times 7$ und geben Sie an, wo
       der Betrag am größten ist.
    1. Mitteln Sie den quadrierten Eingabegradienten über 500 Ziehungen der
       Faltungskerne, für $L = 3$ und für $L = 10$. Summieren Sie jede
       gemittelte Karte über ihre Zeilen und normieren Sie das entstehende
       Spaltenprofil so, dass es sich zu eins summiert. Wie ändert sich das
       Verhältnis der Standardabweichung des Profils zur theoretischen
       Seitenlänge $2L + 1$ mit der Tiefe?
    1. :numref:`fig_conv_rf_paths` zählt die Pfade für $L = 3$ in einer
       Dimension. Zeigen Sie, dass für allgemeines $L$ der Erwartungswert
       des quadrierten Eingabegradienten, summiert über die Zeilen,
       proportional zur Faltung von $L$ Kopien des Vektors $(1, 1, 1)$ ist.
       Leiten Sie her, wie die Standardabweichung des normierten Profils
       mit $L$ wächst.

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
4. [code] **Rückwärtsdurchlauf einer Faltung.** ● Sei $\mathbf{Y}$ die
   einkanalige Kreuzkorrelation einer Eingabe $\mathbf{X}$ mit einem
   Faltungskern $\mathbf{K}$, wie `corr2d` sie berechnet, und sei $\ell$ ein
   skalarer Verlust, der von $\mathbf{X}$ und $\mathbf{K}$ nur über
   $\mathbf{Y}$ abhängt.
    1. Leiten Sie $\partial \ell/\partial \mathbf{K}$ und
       $\partial \ell/\partial \mathbf{X}$ her, ausgedrückt durch
       $\partial \ell/\partial \mathbf{Y}$, im Stil der
       Backpropagation-Gleichungen aus :numref:`sec_backprop`.
    1. Implementieren Sie Ihren Ausdruck für
       $\partial \ell/\partial \mathbf{K}$ ohne automatisches Differenzieren
       für den Verlust $\ell = \sum_{i,j} G_{ij} Y_{ij}$,
       wobei $\mathbf{G}$ eine feste Zufallsmatrix von der Form von
       $\mathbf{Y}$ ist. Vergleichen Sie ihn mit einem Gradienten aus
       zentralen endlichen Differenzen mit der Schrittweite $10^{-2}$ auf
       kleinen zufälligen `X` und `K` und geben Sie den größten relativen
       Fehler an.
    1. Prüfen Sie Ihren Ausdruck für $\partial \ell/\partial \mathbf{K}$
       anhand des Gradienten des Faltungskerns, den das automatische
       Differenzieren im Experiment aus :numref:`subsec_learning_kernel`
       berechnet.

    *Nach Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Kapitel 6, und Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    Notebook ConvolutionalNetworks.*

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
- "Adapted from …" (exercise credit) → "Nach …"; titles of external assignments and books stay in English ("Assignment 2", "Homework 2 Part 1"); "X notebook" → "Notebook X"
- "[code]" and "●" exercise tags are copied unchanged
- "blur kernel" → der Weichzeichnungskern; "sharpening kernel" → der Schärfungskern; "blur" (the operation) → die Weichzeichnung
- "finite difference operator" → Operator endlicher Differenzen; "centered finite difference" → zentrale endliche Differenz
- "nonzero entries" → von null verschiedene Einträge; "sparsity pattern" → das Besetzungsmuster (established term; glossary "Sparsität" is for the property)
- "locally connected layer" → lokal verbundene Schicht; "fourth-order tensor" → Tensor vierter Stufe
- "to tie weights" → Gewichte koppeln
- "single-channel" → einkanalig; "draw" (of random kernels) → die Ziehung, "to draw" → ziehen
- "support" (of a function) → der Träger; "input gradient" → der Eingabegradient; "unit gradient" → der Einheitsgradient
- "standard normal distribution" → die Standardnormalverteilung; "to normalize to sum to one" → normieren
- "Discussions" (forum link) → Diskussion

=== NOTES ===
- 6f4ddec6: "i.e." is written out as "das heißt" (style.md §2: no abbreviations in body text).
