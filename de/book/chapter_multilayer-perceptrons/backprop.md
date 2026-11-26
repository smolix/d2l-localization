<!--en:dfe88535-->
# Backpropagation

<!--en:93f840b7-->
In :numref:`sec_mlp-implementation` haben wir alle $203{,}530$ Parameter unseres Fashion-MNIST-MLP trainiert, ohne eine einzige Ableitung von Hand zu bestimmen. Automatisches Differenzieren (engl. automatic differentiation) hat uns sämtliche Gradienten geliefert. Das Prinzip kennen wir aus :numref:`sec_autograd`. Beim automatischen Differenzieren im Rückwärtsmodus (engl. reverse mode) zeichnen wir den Vorwärtsdurchlauf (engl. forward pass) als Berechnungsgraphen (engl. computational graph) auf. Ausgehend vom skalaren Verlust wenden wir die Kettenregel rückwärts an.

<!--en:3c8bfbcf-->
Sehen wir uns an, wie das bei einem mehrschichtigen Perzeptron (engl. multilayer perceptron, MLP) funktioniert. Für das gesamte MLP genügen **drei lokale Ableitungsregeln für Matrizen**. Je eine Regel gilt für die Matrizenmultiplikation, die Addition eines Bias durch Broadcasting und die elementweise Aktivierung. Mit ihnen können wir drei für die Praxis zentrale Fragen beantworten.

<!--en:6130e6f7-->
1. Wie kann ein einziger Rückwärtsdurchlauf (engl. backward pass) Hunderttausende partielle Ableitungen berechnen, wenn seine Rechenkosten höchstens etwa $2\times$ so hoch sind wie die eines Vorwärtsdurchlaufs?

<!--en:68bc102c-->
2. Warum braucht das Training so viel mehr GPU-Speicher als die Inferenz, und warum wächst der Speicherbedarf linear mit der Minibatchgröße $n$ und der Tiefe $K$?

<!--en:6ee47c78-->
3. Wie entstehen durch die Rückwärtsrekursion tote ReLU-Einheiten sowie verschwindende oder explodierende Gradienten?

<!--en:7c4f2d09-->
## Berechnungsgraphen

<!--en:f5edf441-->
### Ein kleiner Graph

<!--en:113b7acd-->
Rufen wir uns das Differenzieren im Rückwärtsmodus an einem Ausdruck mit drei Skalaren in Erinnerung. Wir betrachten $e = (a + b)\,c$ für $a = 2, b = 1, c = -3$ (:numref:`fig_mdl-mlp-backprop-small`). Der *Vorwärtsdurchlauf* (von links nach rechts) berechnet $u = a + b = 3$ und $e = u\,c = -9$. Der *Rückwärtsdurchlauf* (von rechts nach links) beginnt mit $\partial e / \partial e = 1$ und verwendet an jedem Knoten eine lokale Ableitung.

<!--en:02aae982-->
- Am Multiplikationsknoten $e = u\,c$ gilt $\partial e / \partial u = c = -3$ und $\partial e / \partial c = u = 3$.

<!--en:34bcecc4-->
- Am Additionsknoten $u = a + b$ gilt $\partial u / \partial a = \partial u / \partial b = 1$ und damit $\partial e / \partial a = \partial e / \partial b = -3$.

<!--en:d9319349-->
![Berechnungsgraph von $e = (a + b)\,c$ für $a = 2, b = 1, c = -3$ mit den Werten aus dem Vorwärtsdurchlauf (schwarz) und den Ableitungen $\partial e / \partial (\cdot)$ aus dem Rückwärtsmodus (rot).](../img/mdl-mlp-backprop-small.svg)

<!--en:dc037565-->
Schon hier erkennen wir zwei allgemeine Regeln. (i) Ein Additionsknoten gibt den eingehenden Gradienten unverändert an jeden Summanden weiter. Ein Multiplikationsknoten multipliziert ihn jeweils mit dem *anderen* Faktor. Dazu müssen wir im Vorwärtsdurchlauf $u$ und $c$ zwischenspeichern. (ii) Der Rückwärtsdurchlauf besucht die Knoten des Graphen in *umgekehrter topologischer Reihenfolge*.

<!--en:2abc79ed-->
### Der Berechnungsgraph eines MLP

<!--en:f1ab404a-->
Betrachten wir nun unser MLP mit einer versteckten Schicht (:numref:`fig_forward`) für einen Minibatch $\mathbf{X} \in \mathbb{R}^{n \times d}$. Die Parameter sind $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$, $\mathbf{b}^{(1)} \in \mathbb{R}^h$, $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ und $\mathbf{b}^{(2)} \in \mathbb{R}^q$. Für den $\ell_2$-Gewichtszerfall (engl. weight decay) verwenden wir den Strafterm $s = \frac{\lambda}{2}(\|\mathbf{W}^{(1)}\|_{\textrm{F}}^2 + \|\mathbf{W}^{(2)}\|_{\textrm{F}}^2)$. Damit ergibt sich

<!--en:0d0c3a5d-->
Dabei ist $L$ der über den Minibatch gemittelte Kreuzentropieverlust und $\|\mathbf{W}\|_{\textrm{F}}^2 = \sum_{r,j} w_{rj}^2$ das Quadrat der Frobeniusnorm.

<!--en:d0836c3f-->
![Berechnungsgraph des Vorwärtsdurchlaufs für ein MLP mit einer versteckten Schicht und Gewichtszerfall ($d = 784, h = 256, q = 10$). Die Parameterknoten sind schattiert. Jede Gewichtsmatrix geht sowohl in ihre affine Schicht als auch in den Regularisierungsterm $s$ ein.](../img/mdl-mlp-forward-graph.svg)

<!--en:17b008a9-->
## Backpropagation durch ein MLP

<!--en:798c3056-->
### Lokale Regeln für Matrizen

<!--en:0f7bdb09-->
Für eine beliebige Matrix $\mathbf{A}$ bezeichnen wir mit $\partial J / \partial \mathbf{A}$ die Matrix der partiellen Ableitungen $(\partial J / \partial a_{ij})$. Sie hat *genau dieselbe Form* wie $\mathbf{A}$. Drei lokale Regeln für Vektor-Jacobi-Produkte (:numref:`subsec_mdl-layer-vjp`) genügen für alle Schichten eines MLP.

<!--en:7fd937bb-->
1. **Matrizenmultiplikation ($\mathbf{A} = \mathbf{X}\mathbf{W}$ mit $\mathbf{X} \in \mathbb{R}^{n \times d}, \mathbf{W} \in \mathbb{R}^{d \times h}$):** Der eingehende Gradient sei $\mathbf{G} = \partial J / \partial \mathbf{A} \in \mathbb{R}^{n \times h}$. Wegen $a_{ij} = \sum_r x_{ir} w_{rj}$ liefert die Kettenregel $\partial J / \partial w_{rj} = \sum_i x_{ir} g_{ij}$ und $\partial J / \partial x_{ir} = \sum_j g_{ij} w_{rj}$. In Matrixform gilt damit
   $$
   \frac{\partial J}{\partial \mathbf{W}} = \mathbf{X}^\top \mathbf{G}, \qquad
   \frac{\partial J}{\partial \mathbf{X}} = \mathbf{G}\,\mathbf{W}^\top.
   $$
   :eqlabel:`eq_backprop-matmul`
   Die Formen passen genau zusammen. Die Form von $\mathbf{X}^\top\mathbf{G}$ ist $(d \times n)(n \times h) = d \times h$ und stimmt mit der von $\mathbf{W}$ überein. Die Form von $\mathbf{G}\mathbf{W}^\top$ ist $(n \times h)(h \times d) = n \times d$ und stimmt mit der von $\mathbf{X}$ überein. Wie bei der Multiplikation von Skalaren brauchen wir für $\partial J / \partial \mathbf{W}$ die zwischengespeicherte Eingabe $\mathbf{X}$. Bei der Berechnung von $\partial J / \partial \mathbf{X}$ geben wir das Fehlersignal über $\mathbf{W}^\top$ rückwärts weiter.

<!--en:786b77d2-->
2. **Bias-Addition durch Broadcasting ($\mathbf{Z} = \mathbf{A} + \mathbf{b}$ mit $\mathbf{b} \in \mathbb{R}^h$):** Wir addieren $b_j$ in jeder Zeile $i$ der Spalte $j$. Daher reicht die Addition den Gradienten $\partial J / \partial \mathbf{A} = \mathbf{G}$ unverändert durch. Für $\partial J / \partial \mathbf{b} = \mathbf{G}^\top\mathbf{1}_n$ summieren wir $\mathbf{G}$ über die $n$ Beispiele (Zeilen).

<!--en:32d98056-->
3. **Elementweise Aktivierung ($\mathbf{H} = \phi(\mathbf{Z})$):** Da $h_{ij} = \phi(z_{ij})$ nur vom Eintrag $(i, j)$ abhängt, multiplizieren wir den eingehenden Gradienten eintragsweise (Hadamard-Produkt $\odot$) mit der lokalen Steigung der Aktivierungsfunktion. Damit gilt
   $$\frac{\partial J}{\partial \mathbf{Z}} = \frac{\partial J}{\partial \mathbf{H}} \odot \phi'(\mathbf{Z}).$$

<!--en:e2e02f72-->
### Der Rückwärtsdurchlauf

<!--en:a55b8c8b-->
Wir beginnen mit $\partial J / \partial \mathbf{O} = \frac{1}{n}(\hat{\mathbf{Y}} - \mathbf{Y})$ (:numref:`subsec_softmax_and_derivatives`, mit $\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{O})$ und der One-Hot-Labelmatrix $\mathbf{Y} \in \{0, 1\}^{n \times q}$) und $\partial s / \partial \mathbf{W}^{(k)} = \lambda \mathbf{W}^{(k)}$. Wenden wir unsere drei Regeln in umgekehrter topologischer Reihenfolge von der Ausgabe bis zur Eingabe an, erhalten wir den vollständigen Rückwärtsdurchlauf des MLP.

<!--en:aa31ea54-->
Für ein allgemeines MLP mit $K$ Schichten ($\mathbf{Z}^{(k)} = \mathbf{H}^{(k-1)}\mathbf{W}^{(k)} + \mathbf{b}^{(k)}, \mathbf{H}^{(k)} = \phi(\mathbf{Z}^{(k)})$) durchläuft jede Schicht $k = K-1, \ldots, 1$ dieselben zwei Schritte.

<!--en:7bf4dd3f-->
### Ein Rechenbeispiel

<!--en:b248a355-->
Rechnen wir :eqref:`eq_backprop-mlp` für ein kleines Netz von Hand durch ($n = 1, d = h = 2, q = 1$, ReLU-Aktivierung, Bias-Terme gleich null, $\lambda = 0$ und quadratischer Verlust $L = \frac{1}{2}(o - y)^2$).

<!--en:46a61340-->
Im Vorwärtsdurchlauf (:numref:`fig_mdl-mlp-backprop-graph`) erhalten wir $\mathbf{Z} = \mathbf{X}\mathbf{W}^{(1)} = \begin{bmatrix} -1 & 2 \end{bmatrix}$, $\mathbf{H} = \mathrm{ReLU}(\mathbf{Z}) = \begin{bmatrix} 0 & 2 \end{bmatrix}$ (Einheit 1 ist inaktiv), $o = \mathbf{H}\mathbf{W}^{(2)} = -2$ und $L = \frac{1}{2}(-2)^2 = 2$. Im Rückwärtsdurchlauf gilt $\partial L / \partial o = o - y = -2$ und

<!--en:e83fafed-->
Die gesamte erste Spalte von $\partial L / \partial \mathbf{W}^{(1)}$ ist null. Wegen $z_1 = -1 < 0$ gilt $\phi'(z_1) = 0$. Der eingehende Gradient $\partial L / \partial h_1 = -4$ erreicht daher keines der Gewichte, die in Einheit 1 eingehen.

<!--en:2fd42c2f-->
![Werte im Vorwärtsdurchlauf (schwarz) und Gradienten im Rückwärtsdurchlauf (rot) für das Rechenbeispiel :eqref:`eq_backprop-example`. Wegen $z_1 = -1 < 0$ setzt die ReLU sowohl $\partial L / \partial z_1$ als auch die erste Spalte von $\partial L / \partial \mathbf{W}^{(1)}$ auf null.](../img/mdl-mlp-backprop-graph.svg)

<!--en:4f5d3d74-->
### Prüfung des Rückwärtsdurchlaufs an einem Minibatch

<!--en:70adbda6-->
Implementieren wir :eqref:`eq_backprop-mlp` in `mlp_backward`. Zur Prüfung vergleichen wir die Ergebnisse in 64-Bit-Gleitkommaarithmetik mit denen des automatischen Differenzierens im jeweiligen Framework. Dazu verwenden wir einen zufälligen Minibatch der Form $256 \times 784$ ($h = 256, q = 10, \lambda = 10^{-3}$).

<!--en:118d7b60-->
Erzeugen wir einen zufälligen Minibatch mit 256 Beispielen. Wir übernehmen die Dimensionen und die Initialisierung genau aus :numref:`sec_mlp-implementation` und setzen zusätzlich $\lambda = 10^{-3}$. Dann vergleichen wir `mlp_backward` mit dem automatischen Differenzieren im jeweiligen Framework. Wie bei Gradientenprüfungen üblich, rechnen wir mit 64-Bit-Gleitkommazahlen. Ihre Rundungsfehler liegen in der Größenordnung von $10^{-16}$, sodass selbst kleine algebraische Fehler klar hervortreten. Für jeden Parametertensor geben wir die größte relative Abweichung an.

<!--en:b1140cd7-->
Jeder Parametergradient stimmt bis auf einen relativen Fehler von höchstens $10^{-15}$ mit dem Ergebnis des automatischen Differenzierens überein. Das entspricht einem kleinen Vielfachen des Maschinenepsilons bei doppelter Genauigkeit ($2.2 \times 10^{-16}$).

<!--en:08580c55-->
## Speicherbedarf und Rechenkosten

<!--en:7fa4ab64-->
### Gespeicherte Aktivierungen und Parameterspeicher

<!--en:9269bfbf-->
Aus :eqref:`eq_backprop-mlp` geht hervor, welche Werte wir für $\partial J / \partial \mathbf{W}^{(2)}$ und $\partial J / \partial \mathbf{W}^{(1)}$ im Rückwärtsdurchlauf benötigen. Der Vorwärtsdurchlauf muss $\mathbf{X}$, $\mathbf{H}$ (worin bei ReLU auch die Maske $\mathbf{Z} > 0$ kodiert ist) und $\hat{\mathbf{Y}}$ speichern. In unserem Fashion-MNIST-MLP ($d = 784, h = 256, q = 10$) enthalten diese zwischengespeicherten Tensoren bei einer Minibatchgröße von $n = 256$ insgesamt

<!--en:266b7ecd-->
also bereits mehr Werte als die $203{,}530$ Parameter ($\approx 0.8\textrm{ MB}$). In einem Netz mit $K$ Schichten wächst der Aktivierungsspeicher (engl. activation memory) linear *sowohl* mit der Batchgröße $n$ *als auch* mit der Tiefe $K$. Bei der Inferenz hingegen können wir die Aktivierungen jeder Schicht verwerfen, sobald wir die nächste Schicht berechnet haben.

<!--en:bb419b69-->
### Rechenkosten

<!--en:45c6d9f1-->
Im Vorwärtsdurchlauf fallen pro Beispiel $d h + h q = 784 \times 256 + 256 \times 10 = 203{,}264$ Multiplikationen mit Addition an. Im Rückwärtsdurchlauf erfordert jedes Matrixprodukt $\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}$ im Inneren des Netzes *zwei* Matrixprodukte derselben Größe ($\mathbf{H}^{(k-1)\top}\mathbf{G}$ für die Gewichte und $\mathbf{G}\mathbf{W}^{(k)\top}$ für die Aktivierungen der darunterliegenden Schicht). Für die erste Schicht genügt $\mathbf{X}^\top\mathbf{G}_Z$, da wir $\partial J / \partial \mathbf{X}$ nicht benötigen. Der Rückwärtsdurchlauf unseres zweischichtigen MLP erfordert daher $d h + 2 h q = 205{,}824$ Multiplikationen mit Addition, also etwa so viele wie der Vorwärtsdurchlauf. Fast der gesamte Rechenaufwand entfällt auf die erste Schicht, für die rückwärts ein einziges Matrixprodukt genügt. In einem tiefen Netz liegen fast alle Schichten im Inneren. Daher kostet der Rückwärtsdurchlauf $\sim 2\times$ so viel wie der Vorwärtsdurchlauf und ein vollständiger Trainingsschritt rund $3\times$ so viel wie die Inferenz. In beiden Fällen entsprechen die Rechenkosten für alle $203{,}530$ Ableitungen denen von ein oder zwei zusätzlichen Vorwärtsdurchläufen. Bei einer einseitigen endlichen Differenz (engl. finite difference) kostet dagegen schon ein *einziger* skalarer Parameter einen zusätzlichen Vorwärtsdurchlauf.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:c3f7335d-->
Für die Backpropagation durch ein mehrschichtiges Perzeptron genügen drei lokale Matrixregeln, die wir in umgekehrter topologischer Reihenfolge anwenden. Wir berechnen $\partial J / \partial \mathbf{W} = \mathbf{X}^\top\mathbf{G}$ und $\partial J / \partial \mathbf{X} = \mathbf{G}\mathbf{W}^\top$ für Matrixprodukte $\mathbf{A} = \mathbf{X}\mathbf{W}$. Für die Bias-Addition durch Broadcasting summieren wir über die Zeilen ($\mathbf{G}^\top\mathbf{1}$), für Aktivierungen gewichten wir den Gradienten elementweise ($\mathbf{G} \odot \phi'(\mathbf{Z})$). Geht ein Parameter sowohl in den Datenverlust als auch in den Gewichtszerfall ein, summieren wir die eingehenden Gradienten. Da $\mathbf{X}^\top\mathbf{G}$ von den Eingaben der jeweiligen Schicht im Vorwärtsdurchlauf abhängt, müssen wir beim Training die Zwischenaktivierungen über den Vorwärtsdurchlauf hinweg speichern. Der Speicherbedarf beträgt $O(nKh)$ und wächst linear mit Batchgröße und Tiefe. Dafür liefert ein einziger Rückwärtsdurchlauf alle Parametergradienten mit höchstens etwa doppelt so vielen FLOPs wie ein Vorwärtsdurchlauf.

<!--en:2294243a-->
## Übungen

<!--en:449eeca1-->
1. **Jacobi-Matrix einer Schicht.** Der Gradient des Skalars $J$ nach $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ hat die Form von $\mathbf{W}^{(2)}$. Die Ableitung der Matrix $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} \in \mathbb{R}^{n \times q}$ nach $\mathbf{W}^{(2)}$ ist ein größeres Objekt. Sie enthält für jede Kombination der vier Indizes einen Eintrag $\partial o_{ij} / \partial w_{rt}$.
    1. Wie viele Einträge hat $\partial \mathbf{O} / \partial \mathbf{W}^{(2)}$, und wie viele davon können von null verschieden sein?
    1. Werten Sie die Kettenregel $\partial J / \partial w_{rt} = \sum_{i, j} (\partial J / \partial o_{ij})(\partial o_{ij} / \partial w_{rt})$ aus, um wieder $\mathbf{H}^\top \partial J / \partial \mathbf{O}$ zu erhalten.
    1. Wie viel Speicher würde $\partial \mathbf{O} / \partial \mathbf{W}^{(2)}$ im 32-Bit-Gleitkommaformat für das MLP aus :numref:`sec_mlp-implementation` mit $n = 256$ belegen? Erklären Sie, warum Backpropagation diese Ableitung nie vollständig bildet.

<!--en:808a22a0-->
1. **Speicherbedarf eines tieferen Netzes.** Betrachten Sie ein MLP mit $d$ Eingaben, $q$ Ausgaben und $K$ Schichten. Die $K - 1$ versteckten Schichten sind ReLU-Schichten der Breite $h$. Das Training verwendet Minibatches mit je $n$ Beispielen.
    1. Listen Sie anhand des im Text beschriebenen Rekursionsschritts für $K$ Schichten die Werte auf, die der Rückwärtsdurchlauf liest. Zählen Sie diese Werte und die Parameter.
    1. Bei der Vorhersage kann jeder Wert entfallen, sobald die nächste Schicht ihn verwendet hat. Bestimmen Sie die größte Anzahl von Werten, die bei der Vorhersage gleichzeitig im Speicher liegen.
    1. Bestimmen Sie für $d = 784$, $h = 256$ und $q = 10$ die kleinste Minibatchgröße, bei der die Anzahl der gespeicherten Werte die Parameterzahl übersteigt. Betrachten Sie dazu $K = 2$ und $K = 11$.

<!--en:1db58f6f-->
1. [code] **Gradientenprüfung.** Bevor automatisches Differenzieren zur Verfügung stand, prüfte man einen selbst implementierten Rückwärtsdurchlauf mit endlichen Differenzen.
    1. Wählen Sie einen Eintrag $w$ von $\mathbf{W}^{(1)}$ und vergleichen Sie seinen Gradienten aus `mlp_backward` mit der zentralen Differenz $\bigl(J(w + \epsilon) - J(w - \epsilon)\bigr) / (2\epsilon)$ für $\epsilon = 10^{-1}, 10^{-2}, \ldots, 10^{-7}$. Rechnen Sie einmal im 32-Bit- und einmal im 64-Bit-Gleitkommaformat. Tragen Sie den relativen Fehler gegen $\epsilon$ auf und skalieren Sie beide Achsen logarithmisch.
    1. Erklären Sie die Form der Kurven. Welcher Fehler nimmt mit $\epsilon$ ab, welcher nimmt zu, und wo sind beide Fehler bei der jeweiligen Genauigkeit gleich groß?
    1. Die Prüfung kann trotz korrektem Gradienten fehlschlagen, wenn eine Präaktivierung höchstens $\epsilon$ von null entfernt ist. Erklären Sie, wie es dazu kommt und wie sich das vermeiden lässt.

<!--en:5fc8b664-->
1. **Vorwärtsmodus.** Backpropagation beginnt an der Ausgabe mit $\partial J / \partial J = 1$ und durchläuft den Graphen rückwärts. Beim Differenzieren im Vorwärtsmodus beginnen wir dagegen mit $\partial w / \partial w = 1$ an einem einzelnen gewählten skalaren Parameter $w$ und durchlaufen den Graphen vorwärts. Jeder Knoten berechnet die Ableitung seines Werts nach $w$ aus den Ableitungen seiner Eingaben.
    1. Berechnen Sie im Vorwärtsmodus erneut $\partial L / \partial \mathbf{W}^{(1)}$ für das Netz in :eqref:`eq_backprop-example`. Verwenden Sie einen Durchlauf pro Eintrag von $\mathbf{W}^{(1)}$.
    1. Wie viele Durchläufe benötigt jeder Modus, um alle sechs Gewichtsgradienten dieses Netzes zu bestimmen? Welche Produkte berechnet der Vorwärtsmodus bei jedem Durchlauf neu, während der Rückwärtsmodus sie nur einmal berechnet?

    *Nach Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), Aufgaben 7.12 und 7.13.*

<!--en:346234ed-->
1. [extended] **Automatisches Differenzieren selbst implementieren.** ● Entwickeln Sie ein kleines System zum automatischen Differenzieren. Berechnen Sie damit erneut die Gradienten des Netzes in :eqref:`eq_backprop-example`.
    1. Schreiben Sie eine Klasse `Value` für Skalare, die zu jedem Ergebnis die Eingaben und die zugehörige Operation festhält und so den Berechnungsgraphen aufzeichnet. Die Klasse soll `+`, `*` und `relu` unterstützen. (*Hinweis:* Implementieren Sie `__add__` und `__mul__` so, dass sie jeweils ein neues `Value`-Objekt zurückgeben. Es enthält Verweise auf seine Vorgänger und eine kleine Funktion, die den Gradienten einen Schritt rückwärts weiterleitet.)
    1. Implementieren Sie `backward()`. Sortieren Sie den Graphen topologisch, setzen Sie den Gradienten der Ausgabe auf $1$ und besuchen Sie die Knoten in umgekehrter Reihenfolge. Jeder Knoten gibt dabei seinen Gradienten an seine Vorgänger weiter. Achten Sie darauf, dass sich die Gradienten *aufsummieren* (`+=` statt `=`), wenn ein Wert mehrfach eingeht.
    1. Rechnen Sie mit Ihrem System den Graphen aus :numref:`fig_mdl-mlp-backprop-small` und das Rechenbeispiel :eqref:`eq_backprop-example` erneut durch. Zerlegen Sie dazu die Matrixprodukte in skalare Operationen. Vergleichen Sie die Gradienten mit denen, die wir in diesem Abschnitt hergeleitet haben.
    1. Betrachten Sie folgende Kette. Aus einer skalaren Eingabe $x$ entstehen $v_i = f_i(x)$ für $i = 1, 2, 3$, der Zwischenwert $m = v_1 + v_2 + v_3$, die Werte $p_j = g_j(m)$ für $j = 1, 2, 3$ und der Verlust $L = p_1 + p_2 + p_3$. Schreiben Sie $\alpha_i = f_i'(x)$ und $\beta_j = g_j'(m)$. Die Kettenregel als Summe über die Pfade zählt $3 \times 3 = 9$ Pfade von $x$ nach $L$ auf. Jeder Pfad trägt das Produkt $\alpha_i \beta_j$ bei. Zeigen Sie, dass Ihr System stattdessen $(\alpha_1 + \alpha_2 + \alpha_3)(\beta_1 + \beta_2 + \beta_3)$ berechnet, indem es die Gradientenbeiträge an $m$ nur einmal summiert. Erklären Sie, warum diese Wiederverwendung den Rechenaufwand der Backpropagation in tiefen Netzen begrenzt.

    *Nach Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd), `micrograd/engine.py`.*
   :exlabel:`ex_backprop-autograd-engine`

<!--en:d24f2249-->
[Diskussionen](https://d2l.discourse.group/t/102)

<!--en:cefa09b6-->
Backpropagation und Berechnungsgraphen in MLPs<br>**DAGs im Rückwärtsmodus · lokale Matrixregeln · MLP-Rückwärtsdurchlauf in 5 Zeilen · Aktivierungsspeicher und Rechenkosten in FLOPs**

<!--en:ffe91e05-->
::: {.slide title="Warum ein Rückwärtsdurchlauf alle 203 530 Gradienten liefert"}

<!--en:00b9344a-->
[Ein Blick ins Innere]{.kicker}

<!--en:150ba285-->
- **Endliche Differenzen und Rückwärtsmodus:** Jedes skalare Gewicht einzeln zu variieren ($w \pm \epsilon$) erfordert $203{,}530$ Vorwärtsdurchläufe pro Schritt. Der Rückwärtsmodus berechnet *alle* Parametergradienten mit **höchstens $\sim 2\times$ so vielen FLOPs wie ein Vorwärtsdurchlauf**.

<!--en:7dad2e69-->
- **Gemeinsames Fehlersignal je Schicht:** Backpropagation berechnet die eingehende Sensitivität $\partial J / \partial \mathbf{Z}^{(k)}$ jeder Schicht von oben nach unten *einmal*. Ein einziges Matrixprodukt liefert dann $\partial J / \partial \mathbf{W}^{(k)}$.

<!--en:dae56e1b-->
- **Speicherbedarf beim Training:** Um $\mathbf{H}^{(k-1)\top}\mathbf{G}_{\mathbf{Z}^{(k)}}$ zu bilden, muss der Vorwärtsdurchlauf die Aktivierungen aller versteckten Schichten zwischenspeichern. Das benötigt $O(nKh)$ GPU-Speicher.

<!--en:80281870-->
::: {.slide title="Vorwärts und rückwärts durch einen Berechnungsgraphen mit drei Knoten"}

<!--en:c166b87e-->
[Berechnungsgraphen · Ein skalares Beispiel]{.kicker}

<!--en:f8e88bd0-->
Das Differenzieren im Rückwärtsmodus am einfachen Beispiel $e = (a + b)\,c$ für $a = 2, b = 1, c = -3$ nachvollziehen:

<!--en:1c1dfb36-->
- **Vorwärtsdurchlauf (von links nach rechts):** Zwischenwerte berechnen und speichern, zuerst $u = a + b = 3$, dann $e = u\,c = -9$.

<!--en:7d09f9a6-->
- **Rückwärtsdurchlauf (von rechts nach links):** Mit $\partial e / \partial e = 1$ beginnen und in umgekehrter topologischer Reihenfolge an jedem Knoten eine lokale Ableitung anwenden.
  - **Multiplikationsknoten ($e = u\,c$):** Die Eingaben vertauschen und den Gradienten jeweils mit der *anderen* gespeicherten Eingabe skalieren. Das ergibt $\partial e / \partial u = c = -3$ und $\partial e / \partial c = u = 3$.
  - **Additionsknoten ($u = a + b$):** Der eingehende Gradient fließt unverändert in beide Zweige. Damit gilt $\partial e / \partial a = \partial e / \partial b = -3$.

<!--en:31d7c373-->
::: {.slide title="Berechnungsgraph eines regularisierten MLP"}

<!--en:c632b454-->
[Berechnungsgraphen · DAG für Matrizen]{.kicker}

<!--en:eadc9beb-->
Für einen Minibatch $\mathbf{X} \in \mathbb{R}^{n \times d}$ bildet das MLP mit einer versteckten Schicht und $\ell_2$-Gewichtszerfall $s = \frac{\lambda}{2}(\|\mathbf{W}^{(1)}\|_{\textrm{F}}^2 + \|\mathbf{W}^{(2)}\|_{\textrm{F}}^2)$ einen gerichteten azyklischen Graphen.

<!--en:5a589966-->
- **Rechenschritte im Vorwärtsdurchlauf:** $\mathbf{Z} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$ $\to$ $\mathbf{H} = \phi(\mathbf{Z})$ $\to$ $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$:
  $$J = L(\mathbf{O}, \mathbf{Y}) + s$$

<!--en:885d08d9-->
- **Verzweigte Parameterknoten (schattiert):** Jede Gewichtsmatrix $\mathbf{W}^{(k)}$ hat *zwei* Nachfolger, ihre affine Schicht und den Strafterm $s$ mit der quadrierten Frobeniusnorm.

<!--en:a7ab9a20-->
- **Mehrdimensionale Kettenregel an Verzweigungen:** Führen von einem Knoten mehrere Pfade weiter, ist sein Gradient im Rückwärtsdurchlauf die **Summe** (`+=`) der Gradienten, die über alle ausgehenden Kanten zurückfließen.

<!--en:fb6750af-->
::: {.slide title="Lokale Regel 1 für die Matrizenmultiplikation A = XW"}

<!--en:304ee12b-->
[Lokale Ableitungsregeln für Matrizen]{.kicker}

<!--en:0c4a8f41-->
- **Eintragsweise Betrachtung:** Der Eintrag $w_{rj}$ beeinflusst jedes Beispiel $i$ in Spalte $j$ ($\sum_i x_{ir} g_{ij}$). Der Eintrag $x_{ir}$ beeinflusst jede Ausgabeeinheit $j$ in Zeile $i$ ($\sum_j g_{ij} w_{rj}$).

<!--en:8f36b97e-->
- **Prüfung der Formen:** $\mathbf{X}^\top\mathbf{G}$ ist die *einzige* Möglichkeit, zwei Matrizen der Formen $(n \times d)$ und $(n \times h)$ so zu multiplizieren, dass die Form $(d \times h)$ entsteht. Dazu ist die gespeicherte Eingabe $\mathbf{X}$ aus dem Vorwärtsdurchlauf nötig.

<!--en:00b8514b-->
::: {.slide title="Lokale Regeln 2 und 3 für Bias-Addition durch Broadcasting und Aktivierungen"}

<!--en:304ee12b-2-->
[Lokale Ableitungsregeln für Matrizen]{.kicker}

<!--en:e634be2f-->
- **Bias-Addition durch Broadcasting ($\mathbf{Z} = \mathbf{A} + \mathbf{b}$):** Der Gradient $\mathbf{G}$ fließt unverändert zu $\mathbf{A}$. Für den Bias wird er über die Batchachse summiert (`G.sum(axis=0)`).

<!--en:93d29a19-->
- **Elementweise Aktivierung ($\mathbf{H} = \phi(\mathbf{Z})$):** Jeden Eintrag des eingehenden Gradienten mit der jeweiligen lokalen Steigung multiplizieren (Hadamard-Produkt $\odot$ mit $\phi'(z_{ij})$).

<!--en:b68ad3e4-->
::: {.slide title="Der vollständige Rückwärtsdurchlauf eines MLP in fünf Zeilen"}

<!--en:f7d3012f-->
[Die Schritte des Rückwärtsdurchlaufs]{.kicker}

<!--en:f545a8c8-->
- **Fehlerzuordnung von oben nach unten:** Mit $\mathbf{G}_{\mathbf{O}} = \frac{1}{n}(\hat{\mathbf{Y}} - \mathbf{Y})$ an den Logits der Softmax-Funktion beginnen. Daraus ergibt sich $\nabla\mathbf{W}^{(2)} = \mathbf{H}^\top\mathbf{G}_{\mathbf{O}} + \lambda\mathbf{W}^{(2)}$.

<!--en:08e384b5-->
- **Fehlersignal für die versteckte Schicht:** Mit $\mathbf{G}_{\mathbf{O}}\mathbf{W}^{(2)\top}$ das Signal auf die versteckten Einheiten zurückprojizieren, mit $\odot\,\phi'(\mathbf{Z})$ elementweise gewichten und $\nabla\mathbf{W}^{(1)} = \mathbf{X}^\top\mathbf{G}_{\mathbf{Z}} + \lambda\mathbf{W}^{(1)}$ bilden.

<!--en:21e3f219-->
::: {.slide title="Rückwärtsrekursion für K Schichten"}

<!--en:66910ece-->
[Rekursion in tiefen MLPs]{.kicker}

<!--en:ae9382f3-->
- **Zwei Schritte pro Schicht:** Jede versteckte Schicht $k = K - 1, \ldots, 1$ projiziert den eingehenden Fehler mit $\mathbf{W}^{(k+1)\top}$ zurück und gewichtet ihn elementweise mit $\odot\,\phi'(\mathbf{Z}^{(k)})$. Daraus ergibt sich $\mathbf{H}^{(k-1)\top}\mathbf{G}_{\mathbf{Z}^{(k)}}$.

<!--en:2140e1df-->
- **Produkt über die Schichten:** Die wiederholte Multiplikation mit $\mathbf{W}^{(k)\top}$ und $\phi'(\mathbf{Z}^{(k)})$ erklärt, warum Gradienten in tiefen Netzen verschwinden oder explodieren.

<!--en:0fe6ad9f-->
::: {.slide title="Vorwärts- und Rückwärtsdurchlauf von Hand"}

<!--en:0a27393b-->
[Rechenbeispiel · Zahlen]{.kicker}

<!--en:f5c368f1-->
- **Ausgangswerte ($2 \to 2 \to 1$ mit ReLU):** $\mathbf{X} = [1, 2]$, $\mathbf{W}^{(1)} = \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix}$, $\mathbf{W}^{(2)} = [2, -1]^\top$, $y = 0$, $L = \frac{1}{2}(o - y)^2$.

<!--en:de1aec6c-->
- **Vorwärts:** $\mathbf{Z} = [-1, 2] \to \mathbf{H} = [0, 2] \to o = -2 \to L = 2$.

<!--en:ad645a55-->
- **Rückwärts:** $\partial L/\partial o = -2 \implies \nabla\mathbf{H} = [-4, 2] \implies \nabla\mathbf{Z} = [-4, 2] \odot [0, 1] = [0, 2]$. Damit wird Spalte 1 von $\nabla\mathbf{W}^{(1)}$ null.

<!--en:a3be2b57-->
::: {.slide title="Die tote ReLU im Berechnungsgraphen des Rückwärtsdurchlaufs"}

<!--en:b9fee103-->
[Rechenbeispiel · Graph]{.kicker}

<!--en:d8c8e724-->
Die Werte im Vorwärtsdurchlauf (schwarz) und die Ableitungen im Rückwärtsmodus (rot) zeigen im Berechnungsgraphen, wie eine **tote ReLU-Einheit** das Lernen zum Stillstand bringt.

<!--en:b0688692-->
- **Nullspalte für die eingehenden Gewichte:** Einheit 1 hat eine große eingehende Sensitivität $\partial L / \partial h_1 = -4$. Wegen ihrer negativen Präaktivierung $z_1 = -1 < 0$ beträgt die Steigung der ReLU dennoch $\phi'(z_1) = 0$.

<!--en:63815ea4-->
- **Dauerhafter Stillstand bei Inaktivität im gesamten Datensatz:** Wegen $\partial L / \partial z_1 = 0$ ist die gesamte erste Spalte von $\partial L / \partial \mathbf{W}^{(1)}$ gleich null ($\begin{bmatrix} 0 & 0 \end{bmatrix}^\top$). Spalte 1 von $\mathbf{W}^{(1)}$ ändert sich daher nie.

<!--en:550d1a10-->
::: {.slide title="Implementierung des Rückwärtsdurchlaufs in fünf Zeilen"}

<!--en:871c4e3e-->
[Prüfung der Backpropagation mit Autograd]{.kicker}

<!--en:7cd802aa-->
Die fünf Matrixgleichungen lassen sich direkt in `mlp_backward(X, Y, W1, b1, W2, b2, lambd)` umsetzen.

<!--en:ade79fab-->
Bei ReLU genügt es, $\mathbf{H}$ (oder $\mathbf{Z}$) zu speichern, um die binäre Maske `(Z > 0)` in der Zeile `G_Z = (G_O @ W2.T) * (Z > 0)` auszuwerten.

<!--en:5d96c8cb-->
::: {.slide title="Gradientenprüfung mit Float64 – Übereinstimmung mit Autograd bis auf 10⁻¹⁵"}

<!--en:871c4e3e-2-->
[Prüfung der Backpropagation mit Autograd]{.kicker}

<!--en:25b33e0d-->
Der Vergleich von `mlp_backward` mit `autograd` aus dem jeweiligen Framework in 64-Bit-Gleitkommaarithmetik (`float64`, Maschinenepsilon $\approx 2.2 \times 10^{-16}$) auf einem zufälligen Minibatch der Form $256 \times 784$ bestätigt die Herleitung bis auf einen relativen Fehler von $10^{-15}$.

<!--en:14347382-->
::: {.slide title="Warum das Training mehr Speicher braucht als die Inferenz"}

<!--en:24f0683f-->
[Rechen- und Speicheraufwand · Aktivierungsspeicher]{.kicker}

<!--en:cb40bdc7-->
- **Inferenz ($O(nh)$ RAM):** Der Aktivierungspuffer jeder Schicht lässt sich überschreiben, sobald die nächste Schicht fertig ist.

<!--en:b197aa82-->
- **Training ($O(nKh)$ RAM):** $\mathbf{X}$ ($0.80\textrm{ MB}$), $\mathbf{H}$ ($0.26\textrm{ MB}$) und $\hat{\mathbf{Y}}$ ($0.01\textrm{ MB}$) müssen im Speicher bleiben, bis der Rückwärtsdurchlauf $\mathbf{H}^{(k-1)\top}\mathbf{G}_{\mathbf{Z}^{(k)}}$ berechnet hat.

<!--en:27d97f97-->
- **Einfluss der Batchgröße:** Schon bei einem MLP mit 1 versteckten Schicht und $n = 256$ belegen die gespeicherten Aktivierungen ($1.08\textrm{ MB}$) mehr Speicher als die Modellparameter ($0.81\textrm{ MB}$).

<!--en:c5d73f20-->
::: {.slide title="Rechenkosten – warum der Rückwärtsdurchlauf in tiefen Netzen rund 2× so teuer ist wie der Vorwärtsdurchlauf"}

<!--en:1848d3ca-->
[Rechen- und Speicheraufwand · FLOPs]{.kicker}

<!--en:58a4b519-->
- **Vorwärtsdurchlauf ($203{,}264$ MACs pro Bild):** Pro Schicht fällt ein Matrixprodukt an ($\mathbf{X}\mathbf{W}^{(1)}$ und $\mathbf{H}\mathbf{W}^{(2)}$).

<!--en:92762c7a-->
- **Rückwärtsdurchlauf ($205{,}824$ MACs pro Bild):** Jedes Matrixprodukt $\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}$ im Inneren des Netzes erfordert rückwärts **zwei** Matrixprodukte mit jeweils derselben Anzahl an FLOPs ($\mathbf{H}^{(k-1)\top}\mathbf{G}$ und $\mathbf{G}\mathbf{W}^{(k)\top}$).

<!--en:799c3afa-->
- **3× als Faustregel:** Beim gezeigten MLP mit 2 Schichten benötigen Rückwärts- und Vorwärtsdurchlauf etwa gleich viele FLOPs (in der ersten Schicht entfällt $\partial J/\partial \mathbf{X}$). In tiefen Netzen liegen fast alle Schichten im Inneren. Daher benötigt der Rückwärtsdurchlauf $\approx 2\times$ so viele FLOPs wie der Vorwärtsdurchlauf. Ein vollständiger Trainingsschritt benötigt also $\approx 3\times$ so viele FLOPs wie die Inferenz.

<!--en:043907fd-->
::: {.slide title="Rückwärtsmodus, Vorwärtsmodus und endliche Differenzen im Vergleich"}

<!--en:9420b48e-->
[Rechen- und Speicheraufwand · Algorithmen im Vergleich]{.kicker}

<!--en:54396114-->
Warum nutzt Deep Learning für einen skalaren Verlust $J: \mathbb{R}^P \to \mathbb{R}$ ($P = 203{,}530$) Backpropagation im Rückwärtsmodus und nicht den Vorwärtsmodus oder numerische endliche Differenzen?

<!--en:01373feb-->
| Differenzierverfahren | Ergebnis eines Durchlaufs | Durchläufe für $\nabla J \in \mathbb{R}^P$ | Aktivierungsspeicher | FLOPs pro Schritt insgesamt ($P = 203{,}530$) |

<!--en:e3dcea2a-->
| **Endliche Differenzen** $\frac{J(w+\epsilon)-J(w-\epsilon)}{2\epsilon}$ | $1$ skalare Richtungsableitung | $P$ (bei zentralen Differenzen $2P$) | $O(nh)$ (ohne Zwischenspeicher) | $\sim 203{,}530 \times$ Vorwärtsaufwand ($\sim 4 \times 10^{10}$) |

<!--en:c19f1d9c-->
| **Vorwärtsmodus** (Startwert $\dot{w}_i = 1$) | Tangenten $\partial (\cdot) / \partial w_i$ für 1 Gewicht | $P$ Vorwärtsdurchläufe | $O(nh)$ (ohne Zwischenspeicher) | $\sim 203{,}530 \times$ Vorwärtsaufwand |

<!--en:ef868ba6-->
| **Rückwärtsmodus (Backpropagation)** (Startwert $\bar{J} = 1$) | Adjungierte $\partial J / \partial (\cdot)$ für **alle** Gewichte | **$1$ Rückwärtsdurchlauf** | **$O(nKh)$ zwischengespeichert** | **$\sim 3 \times$ Vorwärtsaufwand ($\sim 6 \times 10^5$)** |

<!--en:f75f89af-->
Der Rückwärtsmodus benötigt $O(1)$ Durchläufe für skalare Ausgaben mit vielen Eingaben ($P \gg 1$). Der Vorwärtsmodus ist nur dann günstiger, wenn es mehr Ausgaben als Eingaben gibt ($q \gg P$).

<!--en:b73cda75-->
::: {.slide title="Verständnisfragen zu Formen, Speicherbedarf und Jacobi-Matrizen"}

<!--en:ffcb46a8-->
[Verständnisfragen]{.kicker}

<!--en:4281f3c3-->
Wie funktioniert Backpropagation, und wie wächst der Ressourcenbedarf?

<!--en:3b454001-->
1. **Summe äußerer Produkte:** Im Fashion-MNIST-MLP mit Batchgröße $n = 256$ hat $\mathbf{X}^\top\mathbf{G}_{\mathbf{Z}}$ die Form $(784, 256) \times (256, 256) = (784, 256)$. Welche Rang-1-Matrix trägt jedes Beispiel zur Summe über die Beispiele $i = 1, \ldots, 256$ bei?

<!--en:16d46932-->
2. **Aktivierungsspeicher für 10 Schichten:** Das MLP erhält $10$ versteckte Schichten der Breite $256$ ($n = 256$). Wie viele Megabyte belegen die zwischengespeicherten `float32`-Aktivierungen pro Minibatch beim Training, und wie groß ist der Parameterspeicher?

<!--en:e6a84cac-->
3. **Warum die vollständige Jacobi-Matrix nicht bilden?** Wie viel Speicher wäre im Format `float32` nötig, um den vierdimensionalen Tensor $\partial \mathbf{O} / \partial \mathbf{W}^{(2)}$ der Form $(n, q, h, q) = (256, 10, 256, 10)$ explizit zu speichern?

<!--en:557b98b6-->
**Lösungen:**

<!--en:53e1aa1d-->
1. **Summe äußerer Produkte:** $\mathbf{X}^\top\mathbf{G}_{\mathbf{Z}} = \sum_{i=1}^{256} \mathbf{x}_i \mathbf{g}_{\mathbf{z}, i}^\top$. Dabei ist $\mathbf{x}_i \in \mathbb{R}^{784}$ die Eingabe von Beispiel $i$ und $\mathbf{g}_{\mathbf{z}, i} \in \mathbb{R}^{256}$ der Gradient bezüglich seiner Präaktivierung.

<!--en:f3de1c92-->
2. **Speicherbedarf bei Tiefe 10:** $\mathbf{X}$ ($200{,}704$) + $10 \times \mathbf{H}^{(k)}$ ($10 \times 65{,}536$) + $\hat{\mathbf{Y}}$ ($2{,}560$) $= 858{,}624$ Gleitkommazahlen, also $\approx 3.43\textrm{ MB}$ (mehr als die $795{,}658$ Parameter mit $\approx 3.18\textrm{ MB}$).

<!--en:d32644ad-->
3. **Speicherbedarf der vollständigen Jacobi-Matrix:** $256 \times 10 \times 256 \times 10 \times 4\textrm{ B} \approx 26.2\textrm{ MB}$ allein für die kleine Ausgabeschicht. Vektor-Jacobi-Produkte liefern durch die Kontraktion $\mathbf{H}^\top\mathbf{G}_{\mathbf{O}}$ direkt eine Matrix der Form $256 \times 10$ ($10\textrm{ KB}$).

<!--en:d95790c1-->
::: {.slide title="Zusammenfassung der Backpropagation in mehrschichtigen Perzeptronen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:5df37c66-->
| Operation im Vorwärtsdurchlauf | Gespeicherte Tensoren | Gradient bezüglich der Parameter | Gradient bezüglich der Eingabe | Rechenkosten (FLOPs) und Speicherbedarf |

<!--en:80ccfd86-->
| **Linear $\mathbf{A} = \mathbf{X}\mathbf{W}$** | $\mathbf{X} \in \mathbb{R}^{n \times d}$, $\mathbf{W} \in \mathbb{R}^{d \times h}$ | $\frac{\partial J}{\partial \mathbf{W}} = \mathbf{X}^\top\mathbf{G}$ | $\frac{\partial J}{\partial \mathbf{X}} = \mathbf{G}\mathbf{W}^\top$ | $2 \times n d h$ Multiplikationen mit Addition im Rückwärtsdurchlauf |

<!--en:04c254f1-->
| **Bias-Addition durch Broadcasting $\mathbf{Z} = \mathbf{A} + \mathbf{b}$** | Keine | $\frac{\partial J}{\partial \mathbf{b}} = \mathbf{G}^\top\mathbf{1}_n$ (Summe über die Zeilen) | $\frac{\partial J}{\partial \mathbf{A}} = \mathbf{G}$ | $O(nh)$ für die Summe, kein zusätzlicher Speicher |

<!--en:6abceaea-->
| **Aktivierung $\mathbf{H} = \phi(\mathbf{Z})$** | $\mathbf{Z}$ (bei ReLU auch $\mathbf{H}$) | Keiner | $\frac{\partial J}{\partial \mathbf{Z}} = \mathbf{G} \odot \phi'(\mathbf{Z})$ | $O(nh)$ für elementweise Operationen, inaktiv bei $\phi' = 0$ |

<!--en:05c831e6-->
| **Gewichtszerfall $s = \frac{\lambda}{2}\|\mathbf{W}\|_{\textrm{F}}^2$** | $\mathbf{W}$ | Addiert $+\lambda\mathbf{W}$ zu $\frac{\partial J}{\partial \mathbf{W}}$ | – | Verzweigung im DAG summiert eingehende Gradienten |

<!--en:382af61b-->
| **Vollständiger Durchlauf durch ein MLP mit $K$ Schichten** | $\mathbf{X}, \mathbf{H}^{(1 \dots K-1)}, \hat{\mathbf{Y}}$ | Alle $\mathbf{W}^{(k)}, \mathbf{b}^{(k)}$ in 1 Rückwärtsdurchlauf | Gibt $\mathbf{G}\mathbf{W}^{(k)\top} \odot \phi'$ weiter | $\sim 2\times$ so viele FLOPs wie im Vorwärtsdurchlauf, $O(nKh)$ RAM |
