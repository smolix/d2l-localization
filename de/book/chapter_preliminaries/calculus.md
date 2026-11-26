<!--en:5ed86b2f-->
# Analysis

<!--en:4220625a-->
Im antiken Griechenland berechnete Archimedes die Fläche eines Kreises, indem er ihm regelmäßige Vielecke mit immer mehr Seiten einschrieb, $n \to \infty$ (:numref:`fig_circle_area`). Je schmaler die Dreiecke werden, desto näher kommt ihre Höhe dem Radius $r$ und die Summe ihrer Grundseiten dem Kreisumfang $2\pi r$. So ergibt sich die Fläche $\frac{1}{2} r (2\pi r) = \pi r^2$. Gekrümmte Formen durch lineare Teilstücke anzunähern und zum Grenzwert kleiner Schritte überzugehen ist der Grundgedanke sowohl der Integralrechnung als auch der *Differentialrechnung*.

<!--en:6972b253-->
![Einem Kreis einbeschriebene regelmäßige Vielecke mit 4, 5, 6 und 8 Seiten. Mit wachsender Seitenzahl nähert sich die Fläche des Vielecks der Kreisfläche.](../img/polygon-circle.svg)

<!--en:4737c3fd-->
Im Deep Learning ermöglicht die Differentialrechnung die *Optimierung*. Ein Modell zu trainieren heißt, seine Parameter so anzupassen, dass eine skalare Verlustfunktion möglichst klein wird. Sie misst, wie schlecht das Modell zu den Trainingsbeispielen passt. Wollten wir bei Millionen oder Milliarden Parametern jeden einzeln ausprobieren, bräuchten wir für jede Aktualisierung Millionen von Vorwärtsdurchläufen. Die *Ableitung* gibt an, wie schnell sich der Verlust bei einer Änderung des jeweiligen Parameters im betrachteten Punkt ändert. Fassen wir alle partiellen Ableitungen zum *Gradienten* zusammen, erhalten wir die Richtung des steilsten Anstiegs. Die *Kettenregel* (engl. chain rule) verknüpft die Gradienten ineinandergeschachtelter Schichten in einem einzigen Rückwärtsdurchlauf.

<!--en:5fdcc53c-->
In diesem Abschnitt entwickeln wir Ableitungen, Gradienten und die Kettenregel anhand eines durchgängigen Beispiels zur linearen Regression. Es ist klein genug, um jede Zahl von Hand nachzurechnen. In :numref:`sec_autograd` sehen wir, wie Deep-Learning-Frameworks diese Rechnungen für beliebige Programme automatisieren.

<!--en:da5733be-->
## Ableitungen

<!--en:410f022f-->
### Die Ableitung als Grenzwert

<!--en:c07f7586-->
Angenommen, ein Modell mit einem einzigen Parameter sagt für jedes Beispiel denselben Wert $w$ vorher. Wir werten es auf drei Trainingsbeispielen mit den Labels $1$, $1$ und $0$ aus. Die Summe der quadratischen Fehler ergibt

<!--en:1f3f0fd4-->
Für $w = 1$ beträgt der Verlust $L(1) = 1$:

<!--en:49143084-->
Um zu sehen, wie $L(w)$ auf eine kleine Störung $h \neq 0$ reagiert, berechnen wir den Differenzenquotienten $\frac{L(1+h) - L(1)}{h}$ für Werte von $h$, die von $10^{-1}$ auf $10^{-5}$ abnehmen:

<!--en:4dbac01b-->
Ausmultiplizieren liefert $L(1+h) = 1 + 2h + 3h^2$. Damit gilt $\frac{L(1+h) - L(1)}{h} = 2 + 3h \to 2$ für $h \to 0$. Formal ist für eine Funktion $f: \mathbb{R} \rightarrow \mathbb{R}$ die *Ableitung* von $f$ an der Stelle $x$ der Grenzwert

<!--en:8965ea4d-->
Geometrisch ist der Differenzenquotient die Steigung der *Sekante* durch $P = (x, f(x))$ und $Q = (x+h, f(x+h))$. Für $h \to 0$ dreht sich die Sekante in die *Tangente* im Punkt $P$ mit der Steigung $f'(x)$ (:numref:`fig_calc_secant_tangent`).

<!--en:fc6c518d-->
![Der Differenzenquotient $(f(x+h)-f(x))/h$ ist die Steigung der Sekante durch $P$ und $Q$. Für $h \to 0$ rückt $Q$ an $P$ heran. Die Sekante geht dabei in die Tangente im Punkt $P$ mit der Steigung $f'(x)$ über.](../img/calculus-secant-tangent.svg)

<!--en:440f161e-->
Nicht jede Funktion ist überall differenzierbar. Die Funktion $|x|$ hat bei $x = 0$ einen Knick (Steigung $-1$ von links, $+1$ von rechts). Noch ungünstiger verhält sich die diskrete Klassifikationsgenauigkeit. Sie ist fast überall flach ($f'(x) = 0$) und springt an den Stellen, an denen sich eine Vorhersage ändert. Sie liefert also kein Gradientensignal. Deshalb trainieren wir Klassifikatoren, indem wir glatte Ersatzverluste wie die Kreuzentropie minimieren (:numref:`fig_calc_surrogate`).

<!--en:0c1ae89c-->
![Der 0–1-Fehler ($1$ für eine falsche Vorhersage, $0$ für eine richtige) und ein glatter Ersatzverlust als Funktionen eines Scores $z$, der bei richtiger Klassifikation positiv ist. Der 0–1-Fehler ist beiderseits des Sprungs konstant, seine Steigung also ohne Aussagekraft. Der Ersatzverlust fällt überall und gibt damit die Richtung für eine Änderung der Parameter vor.](../img/calculus-surrogate-loss.svg)

<!--en:f6f622c0-->
### Ableitungsregeln

<!--en:7edc5a78-->
Für $y = f(x)$ schreiben wir $f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x)$. Häufig verwendete elementare Ableitungen sind

<!--en:d202ca95-->
und für Kombinationen differenzierbarer Funktionen $f, g$ gilt

<!--en:91fc6fb9-->
:numref:`fig_calc_product_rule` veranschaulicht die Produktregel geometrisch anhand des Flächeninhalts eines Rechtecks mit den Seitenlängen $f(x) \times g(x)$ (vollständige Beweise finden sich in :numref:`sec_mdl-single_variable_calculus`). Wenden wir diese Regeln auf $L(w) = 3w^2 - 4w + 2$ an, erhalten wir direkt $L'(w) = 6w - 4$ und damit $L'(1) = 2$.

<!--en:05e15158-->
![Die Produktregel am Flächeninhalt eines Rechtecks. Wächst $x$ um $h$, so wachsen die Seiten $f(x)$ und $g(x)$ um ungefähr $f'(x) h$ und $g'(x) h$. Zur Fläche kommen zwei schmale Streifen und ein Eckstück mit der Fläche $f'(x) g'(x) h^2$ hinzu, das für kleine $h$ vernachlässigbar ist.](../img/calculus-product-rule.svg)

<!--en:aa98099f-->
### Lineare Näherung erster Ordnung und Minima

<!--en:ecac162d-->
Multiplizieren wir $\frac{f(x+h) - f(x)}{h} \approx f'(x)$ mit $h$, erhalten wir die *lineare Näherung erster Ordnung*

<!--en:973eeb45-->
Wegen $L'(1) = 2 > 0$ sinkt der Verlust, wenn wir $w$ verkleinern. Ein Schritt entgegen der Ableitung gemäß $w \leftarrow w - \eta L'(w)$ mit der Schrittweite $\eta = 0.1$ verschiebt $w$ von $1$ auf $1 - 0.1(2) = 0.8$. Dabei fällt $L$ von $1$ auf $L(0.8) = 0.72$, nahe an der Vorhersage der Tangente $1 + 2(-0.2) = 0.60$ (die Differenz von $0.12$ stammt vom vernachlässigten quadratischen Krümmungsterm $3h^2$). An jedem glatten lokalen Minimum muss die Tangente waagerecht sein ($f'(x) = 0$). Aus $L'(w) = 6w - 4 = 0$ ergibt sich der eindeutige globale Minimierer $w^* = 2/3$, der Stichprobenmittelwert der drei Labels $1, 1, 0$.

<!--en:4658ee63-->
## Partielle Ableitungen und Gradienten

<!--en:7133f178-->
### Partielle Ableitungen

<!--en:74b944d0-->
Nun ordnen wir unseren drei Beispielen die skalaren Eingaben $0$, $1$ und $2$ mit den zugehörigen Labels $1$, $1$ und $0$ zu und passen die affine Gerade $w_1 + w_2 \cdot \textrm{input}$ an:

<!--en:7dbf47ff-->
Bei einer Funktion mehrerer Variablen $y = f(x_1, \ldots, x_n)$ verändern wir für die *partielle Ableitung* nach $x_i$ nur die Koordinate $i$ und halten alle übrigen Variablen fest:

<!--en:279daf91-->
Betrachten wir den Punkt $(w_1, w_2) = (1, 0)$. Halten wir $w_2 = 0$ fest, ergibt sich $L(w_1, 0) = 3w_1^2 - 4w_1 + 2$ und damit $\frac{\partial L}{\partial w_1}(1, 0) = 2$. Halten wir dagegen $w_1 = 1$ fest, erhalten wir $L(1, w_2) = 5w_2^2 + 4w_2 + 1$ mit der Ableitung $10w_2 + 4$, also $\frac{\partial L}{\partial w_2}(1, 0) = 4$ (:numref:`fig_calc_partial_slices`).

<!--en:991c3fdd-->
![Der Verlust $L(w_1, w_2)$ des Geradenmodells in der Nähe von $\mathbf{w} = (1, 0)$ (schwarzer Punkt). Für festes $w_2 = 0$ ergibt sich die orangefarbene Kurve, der Verlust des konstanten Modells. Für festes $w_1 = 1$ ergibt sich die grüne Kurve. Die Tangenten am markierten Punkt (gestrichelt) haben die Steigungen $\partial L / \partial w_1 = 2$ und $\partial L / \partial w_2 = 4$.](../img/calculus-partial-slices.svg)

<!--en:4716713d-->
Für die Matrix-Vektor-Darstellung (:numref:`sec_linear-algebra`) schreiben wir jedes Beispiel als Zeile $(1, \textrm{input}_i)$ von $\mathbf{X}$ und erhalten

<!--en:29578a16-->
Wir können diese partiellen Ableitungen numerisch prüfen. Dazu definieren wir `line_loss(w)` und die Hilfsfunktion `numerical_gradient(f, w, h=1e-6)` für endliche Differenzen. Sie verändert jeweils eine Koordinate von `w` um $h = 10^{-6}$ und rechnet in `float64`:

<!--en:c3351956-->
Die Auswertung von `numerical_gradient(line_loss, w)` bei $\mathbf{w} = (1, 0)$ liefert unsere analytisch berechneten partiellen Ableitungen $(2, 4)$ bis auf den Verfahrensfehler der Größenordnung $O(h) \approx 10^{-6}$:

<!--en:f69ccbce-->
Die Funktion `numerical_gradient` benötigt $n$ zusätzliche Auswertungen von $f$ für $n$ Parameter. Bei $n = 10^8$ ist der Aufwand zu hoch.

<!--en:72af8e7f-->
### Genauigkeitsgrenzen endlicher Differenzen

<!--en:94c23282-->
Ist $h$ zu klein, leiden endliche Differenzen zudem unter Auslöschung in der Gleitkommaarithmetik. Bei endlicher Genauigkeit (Maschinenepsilon $\epsilon \approx 2.2 \times 10^{-16}$ für `float64`, $\epsilon \approx 1.2 \times 10^{-7}$ für `float32`) fallen bei der Subtraktion fast gleicher Zahlen $f(x+h) - f(x)$ führende Stellen weg (*katastrophale Auslöschung*, engl. catastrophic cancellation). Der verbleibende Rundungsfehler der Größenordnung $\sim \epsilon$ wird durch $h$ dividiert. Damit wirken *Verfahrensfehler* ($\propto h$) und *Rundungsfehler* ($\propto \epsilon / h$) gegeneinander:

<!--en:950c97dc-->
Wir vergleichen die 64-Bit-Arithmetik von NumPy mit dem im Deep Learning üblichen 32-Bit-Gleitkommaformat. Dazu wandeln wir das Gitter der Schrittweiten `h` in einen Framework-Tensor vom Typ `float32` mit dem Namen `h32` um:

<!--en:17a8b425-->
Nun werten wir `quotient_error(loss, 1, 2, ...)` bei $w = 1$ aus (dort ist die exakte Ableitung $L'(1) = 2$). Für alle $61$ Schrittweiten rechnen wir sowohl in `float64` (`h`) als auch in `float32` (`h32`). Zum Vergleich zeichnen wir die Gerade $3h$ für den Verfahrensfehler und die Asymptote des Rundungsfehlers in `float32` mit der Gleichung $\epsilon / h$ ein:

<!--en:0937cbaa-->
Setzen wir beide Fehler gleich, $3h \approx \epsilon / h$, ergibt sich die optimale Schrittweite $h^* \approx \sqrt{\epsilon / 3}$. Dabei geht die Hälfte der Stellen des Zahlenformats verloren (kleinster Fehler $\sim 10^{-8}$ bei $h \approx 10^{-8}$ in `float64` und $\sim 10^{-4}$ bei $h \approx 10^{-4}$ in `float32`). Analytisches Differenzieren mit der Kettenregel vermeidet sowohl den Aufwand von $O(n)$ als auch den Genauigkeitsverlust der Größenordnung $\sqrt{\epsilon}$. Endliche Differenzen in `float64` dienen dann dazu, Gradienten in Softwaretests zu prüfen.

<!--en:4f6e3e89-->
### Der Gradient und der steilste Abstieg

<!--en:14b6cb3a-->
Fassen wir alle $n$ partiellen Ableitungen von $f: \mathbb{R}^n \to \mathbb{R}$ zusammen, erhalten wir den *Gradienten* $\nabla_{\mathbf{x}} f(\mathbf{x}) = (\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n})^\top$. Dieser Vektor hat dieselbe Form wie $\mathbf{x}$. Damit verallgemeinern wir :eqref:`eq_calculus-linear-1d` auf gleichzeitige Störungen $\mathbf{h} \in \mathbb{R}^n$:

<!--en:5c5f70c8-->
Für einen Schritt $\mathbf{h} = t\mathbf{d}$ der Länge $t > 0$ entlang des Einheitsvektors $\mathbf{d}$ ($\|\mathbf{d}\|_2 = 1$) folgt aus :eqref:`eq_dot_cos` die Näherung $f(\mathbf{x} + t\mathbf{d}) - f(\mathbf{x}) \approx t\,\|\nabla f(\mathbf{x})\|\cos\theta$. Also zeigt $\nabla f(\mathbf{x})$ in die Richtung des *steilsten Anstiegs* ($\theta = 0^\circ$) und $-\nabla f(\mathbf{x})$ in die Richtung des *steilsten Abstiegs* ($\theta = 180^\circ$). Zudem steht $\nabla f(\mathbf{x})$ senkrecht auf der Niveaumenge durch $\mathbf{x}$ ($\theta = 90^\circ$) (:numref:`fig_calc_gradient_field`).

<!--en:b4eb400a-->
![Niveaulinien des Verlusts $L(w_1, w_2)$ des Geradenmodells. Die Gradienten (rot) stehen senkrecht auf der Niveaulinie durch ihren Fußpunkt und zeigen bergauf. Vom Punkt $\mathbf{w} = (1, 0)$ (schwarz) startet der Gradientenabstieg mit $\eta = 0.1$ (orange) in Richtung $-\nabla L$. Diese Richtung zeigt nicht zum Minimum, dem sich der Gradientenabstieg im Zickzack nähert.](../img/calculus-gradient-field.svg)

<!--en:5b0e695d-->
Daraus ergibt sich der *Gradientenabstieg* (engl. gradient descent), $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$. Ausgehend von $\mathbf{w} = (1, 0)$ führt ein Schritt mit der Lernrate $\eta = 0.1$ zum Punkt $(1, 0) - 0.1(2, 4) = (0.8, -0.4)$ und senkt den Verlust von $1.0$ auf $0.4$. Warum verläuft der steilste Abstieg im Zickzack durch das Tal in :numref:`fig_calc_gradient_field`, statt direkt zum globalen Minimum $\mathbf{w}^* = (7/6, -1/2)$ zu führen? „Steil“ bezieht sich hier auf die Änderung pro Einheit der euklidischen Schrittlänge $\|\mathbf{h}\|_2$. Diese hängt von den Maßeinheiten der Koordinaten ab. Eine Streckung entlang einer Merkmalsachse flacht die Niveaulinien ab und kippt $-\nabla L$ vom Minimum weg. Darum sind die Standardisierung der Merkmale (:numref:`sec_pandas`) und die Vorkonditionierung anhand der Krümmung (:numref:`chap_optimization`) in der Praxis so wichtig.

<!--en:dba3e234-->
### Rechenregeln der Matrixanalysis

<!--en:48b96484-->
Drei Regeln für Vektorgradienten (:numref:`sec_mdl-matrix-calculus`) ermöglichen es uns, Vektorausdrücke abzuleiten, ohne die Koordinaten auszuschreiben:

<!--en:43c872a0-->
Multiplizieren wir $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = \mathbf{w}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{w} - 2(\mathbf{X}^\top\mathbf{y})^\top\mathbf{w} + \mathbf{y}^\top\mathbf{y}$ aus und wenden diese Regeln an, erhalten wir

<!--en:67a5b0f2-->
Jede Komponente des Gradienten ist also das Doppelte des Skalarprodukts der entsprechenden Merkmalsspalte von $\mathbf{X}$ mit dem Residuenvektor $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$. Im Optimum gilt $\nabla L(\mathbf{w}^*) = \mathbf{0}$, und die Vorhersagefehler stehen senkrecht auf jeder Merkmalsspalte.

<!--en:ce6ecec8-->
## Die Kettenregel

<!--en:6f8e0b0d-->
Tiefe Netze verketten viele ineinandergeschachtelte Schichten $y = f_L(f_{L-1}(\cdots f_1(\mathbf{x})\cdots))$. Mit der *Kettenregel* differenzieren wir solche Verkettungen, ohne sie auszumultiplizieren.

<!--en:ec5a11eb-->
Für skalare Funktionen $z = f(u)$ und $u = g(x)$ multiplizieren sich die Änderungsraten: $\frac{dz}{dx} = \frac{dz}{du}\frac{du}{dx}$. Allgemeiner gelte $z = f(u_1, \ldots, u_m)$ mit $u_j = g_j(x_1, \ldots, x_n)$ für jedes Zwischenergebnis. Dann multiplizieren sich die Änderungsraten entlang jedes Pfades $x_i \to u_j \to z$, und die Beiträge aller parallelen Pfade addieren sich:

<!--en:04b6fd58-->
Dabei ist $\mathbf{J} \in \mathbb{R}^{m \times n}$ die *Jacobi-Matrix* (engl. Jacobian matrix) von $\mathbf{u}$ nach $\mathbf{x}$ mit den Einträgen $J_{ji} = \frac{\partial u_j}{\partial x_i}$. Für eine affine Abbildung $\mathbf{u} = \mathbf{A}\mathbf{x} - \mathbf{y}$ ist die Jacobi-Matrix gleich $\mathbf{J} = \mathbf{A}$.

<!--en:c0949c8c-->
Rechnen wir einen kleinen Fall von Hand durch. Es sei $z = u^2 + v$ mit $u = x_1 x_2$ und $v = x_2^2$, ausgewertet bei $(x_1, x_2) = (1, 2)$. Von der Eingabe $x_2$ führen zwei Pfade zu $z$, einer über $u$ und einer über $v$ (:numref:`fig_calc_chain_paths`). Jeder Pfad trägt das Produkt seiner lokalen Ableitungen bei. Beide Beiträge addieren sich zu $\partial z / \partial x_2 = 4 \cdot 1 + 1 \cdot 4 = 8$.

<!--en:02554915-->
![Die Kettenregel für $z = u^2 + v$ mit $u = x_1 x_2$ und $v = x_2^2$ bei $(x_1, x_2) = (1, 2)$. Jeder Kasten zeigt den Wert einer Variablen (schwarz) und die Ableitung von $z$ nach dieser Variablen (rot). Jeder rote Pfeil trägt eine lokale Ableitung. Entlang eines Pfades multiplizieren sich die Ableitungen. Wo zwei Pfade zusammentreffen, etwa bei $x_2$, addieren sich ihre Beiträge.](../img/calculus-parallel-dag.svg)

<!--en:f63a3677-->
Zerlegen wir den Verlust unseres Geradenmodells $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ in zwei Stufen. (1) Die innere Funktion liefert die Residuen $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ mit Jacobi-Matrix $\mathbf{J} = \mathbf{X}$. (2) Die äußere Funktion liefert die quadrierte Norm $L = \|\mathbf{r}\|^2$ mit Gradient $\nabla_{\mathbf{r}} L = 2\mathbf{r}$. Mit der mehrdimensionalen Kettenregel erhalten wir $\nabla_{\mathbf{w}} L = \mathbf{X}^\top(2\mathbf{r})$ in zwei Durchläufen:

<!--en:6936402f-->
Der *Vorwärtsdurchlauf* (engl. forward pass) berechnet $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ und $L = \mathbf{r}^\top\mathbf{r}$. Der *Rückwärtsdurchlauf* (engl. backward pass, auch *Backpropagation*) multipliziert $2\mathbf{r}$ mit $\mathbf{X}^\top$. So erhält man den exakten Gradienten nach allen Parametern mit ungefähr dem Aufwand einer Vorwärtsauswertung. In :numref:`sec_autograd` übernimmt das Framework diesen schematischen Rückwärtsdurchlauf durch automatisches Differenzieren.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:825f904e-->
Die Ableitung $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ gibt die Steigung der Tangente und die lokale lineare Näherung $f(x+h) \approx f(x) + f'(x)h$ an. Für $f: \mathbb{R}^n \to \mathbb{R}$ fasst der Gradient $\nabla_{\mathbf{x}} f(\mathbf{x})$ alle partiellen Ableitungen zusammen. Er erfüllt $f(\mathbf{x}+\mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{h}$ und zeigt in die Richtung des steilsten Anstiegs, senkrecht zu den Niveaumengen. Daraus ergibt sich der Gradientenabstieg $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$. Endliche Differenzen benötigen $O(n)$ Auswertungen und verlieren durch katastrophale Auslöschung an Genauigkeit ($\sim \sqrt{\epsilon}$). Mit der mehrdimensionalen Kettenregel $\nabla_{\mathbf{x}} z = \mathbf{J}^\top \nabla_{\mathbf{u}} z$ lassen sich dagegen in einem einzigen Rückwärtsdurchlauf exakte Gradienten vom Verlust bis zu allen Parametern weiterreichen (*Backpropagation*).

<!--en:2294243a-->
## Übungen

<!--en:11ffcb1a-->
1. **Ableitung von $x^x$ bei variabler Basis und variablem Exponenten.** Betrachten Sie für $x > 0$ die Funktion $f(x) = x^x$, in der $x$ sowohl in der Basis als auch im Exponenten vorkommt.
    1. Erklären Sie, warum weder die Potenzregel allein (bei konstant gehaltenem Exponenten ergibt sie $x \cdot x^{x-1} = x^x$) noch die Exponentialregel allein (bei konstant gehaltener Basis ergibt sie $x^x \log x$) die richtige Ableitung liefert.
    1. Schreiben Sie $x^x = e^{x \log x}$ um und bestimmen Sie $f'(x)$ mit der Kettenregel und der Produktregel. Die richtige Ableitung ist die *Summe* der beiden naiven Antworten aus Teil 1. Erklären Sie dies mit der Kettenregel für zwei Variablen, angewandt auf $g(u, v) = u^v$ entlang von $u = x, v = x$.

<!--en:e70f34e9-->
1. **Warum der quadratische Fehler den Mittelwert liefert und der absolute Fehler den Median.** Ein Modell mit einem einzigen Parameter sage den konstanten Wert $w \in \mathbb{R}$ für $n$ Trainingsbeispiele mit skalaren Labels $y_1, \ldots, y_n$ vorher.
    1. Leiten Sie die Summe der quadratischen Fehler $L_2(w) = \sum_{i=1}^n (w - y_i)^2$ nach $w$ ab und setzen Sie die Ableitung gleich null. Beweisen Sie, dass der *Mittelwert* der Stichprobe $w^* = \frac{1}{n}\sum_{i=1}^n y_i$ der eindeutige Minimierer ist.
    1. Betrachten Sie nun die Summe der absoluten Fehler $L_1(w) = \sum_{i=1}^n |w - y_i|$. Zeigen Sie, dass an jeder Stelle $w$, die von allen Labels $y_i$ verschieden ist, für die Ableitung $L_1'(w)$ Folgendes gilt: Sie ist gleich der Anzahl der Labels, die strikt kleiner als $w$ sind, abzüglich der Anzahl der Labels, die strikt größer als $w$ sind. An welchen Stellen $w$ existiert $L_1'(w)$ nicht? Folgern Sie daraus: $L_1(w)$ ist an jedem *Median* der Stichprobe $\{y_1, \ldots, y_n\}$ minimal.
    1. Berechnen Sie für die vier Labels $\{1, 1, 0, 10\}$ den Minimierer von $L_2$ (Mittelwert) und den Minimierer von $L_1$ (Median). Was geschieht mit den beiden Minimierern, wenn das Ausreißerlabel $10$ auf $100$ verfälscht wird?

<!--en:8d227e94-->
1. **Gradient der euklidischen Norm.** Berechnen Sie den Gradienten $\nabla_{\mathbf{x}} f(\mathbf{x})$ der unquadrierten euklidischen Norm $f(\mathbf{x}) = \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$ für jeden von null verschiedenen Vektor $\mathbf{x} \neq \mathbf{0}$. Prüfen Sie, dass $\|\nabla_{\mathbf{x}} f(\mathbf{x})\|_2 = 1$ überall gilt, wo der Gradient existiert. Was geschieht mit dem Gradienten im Ursprung $\mathbf{x} = \mathbf{0}$ und warum?

<!--en:5253e0cf-->
1. **Stabilität des Gradientenabstiegs in Abhängigkeit von der Schrittweite.** Betrachten Sie die Minimierung der isotropen quadratischen Schale $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ mit dem Gradientenabstieg $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)$ und der Lernrate $\eta > 0$.
    1. Berechnen Sie ausgehend von $\mathbf{x}_0 = (1, 1)^\top$ den Punkt $\mathbf{x}_1$ und den neuen Zielfunktionswert $f(\mathbf{x}_1)$ nach einem einzigen Schritt für die drei Lernraten $\eta = 0.1$, $\eta = 1.0$ und $\eta = 2.0$.
    1. Geben Sie $\mathbf{x}_{k+1}$ in geschlossener Form als Funktion von $\mathbf{x}_k$ und $\eta$ an. Für welchen Bereich von Lernraten $\eta > 0$ senkt ein Gradientenschritt den Wert von $f(\mathbf{x})$ für jeden von null verschiedenen Startpunkt $\mathbf{x}_k \neq \mathbf{0}$ strikt? Bei welchem speziellen Wert von $\eta$ erreicht man das globale Minimum $\mathbf{x}^* = \mathbf{0}$ in einem einzigen Schritt, und was passiert für $\eta > 1$?
    1. Charakterisieren Sie die stationären Punkte $\mathbf{x}$, an denen ein Gradientenschritt den Parametervektor unverändert lässt ($\mathbf{x}_{k+1} = \mathbf{x}_k$), sowohl für $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ als auch für eine allgemeine differenzierbare Zielfunktion $f$.

<!--en:2aa487b4-->
1. [code] **Zentrale Differenzen und Vorwärtsdifferenzen.** ● Die in diesem Abschnitt untersuchte einseitige *Vorwärtsdifferenz* $\frac{f(x+h) - f(x)}{h}$ hat den Verfahrensfehler $O(h)$. Die symmetrische *zentrale Differenz*

   $$\frac{f(x+h) - f(x-h)}{2h}$$

   wertet $f$ bei $x+h$ und $x-h$ aus, sodass sich die quadratischen Terme der Ordnung $O(h)$ aufheben. Schreiben Sie eine Variante von `quotient_error` mit zentralen Differenzen und werten Sie sie auf unserem logarithmischen Gitter von $h$-Werten in `float64` aus:
    1. Zeichnen Sie für den quadratischen Verlust $L(w) = 3w^2 - 4w + 2$ bei $w = 1$ die Fehlerkurven der Vorwärtsdifferenz und der zentralen Differenz in dasselbe doppelt logarithmische Diagramm. Warum sinkt der Fehler der zentralen Differenz bei einer quadratischen Funktion direkt auf das Maschinenepsilon ($\sim 10^{-16}$), auch wenn $h = 10^{-1}$ groß ist?
    1. Wiederholen Sie den Vergleich mit einer nichtquadratischen Funktion, nämlich $f(x) = e^x$ bei $x = 0$ (mit $f'(0) = 1$). Vergleichen Sie im Bereich großer $h$ die Steigung der Fehlerkurve der zentralen Differenz mit der Steigung $1$ bei der Vorwärtsdifferenz. Bestimmen Sie für beide Verfahren die optimale Schrittweite $h^*$ und den kleinsten erreichbaren Fehler. Erklären Sie beides mit der Taylor-Entwicklung $e^{\pm h} = 1 \pm h + \frac{h^2}{2} \pm \frac{h^3}{6} + \cdots$.

<!--en:d457ddb3-->
1. [code] **Gradientenprüfung in `float64` und `float32`.** ● Endliche Differenzen sind für das Training viel zu langsam. Zentrale Differenzen sind jedoch das Referenzverfahren, um von Hand abgeleitete oder in eigenen Kerneln implementierte Gradienten in Softwaretests zu prüfen.
    1. Schreiben Sie eine Funktion `central_gradient(f, x, h=1e-5)`, die $\nabla f(\mathbf{x})$ für $f: \mathbb{R}^n \to \mathbb{R}$ mit zentralen Differenzen schätzt. Prüfen Sie damit in `float64` die Gradientenformel für quadratische Formen $\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ an einer zufälligen *nicht symmetrischen* $5 \times 5$-Matrix $\mathbf{A}$ und einem zufälligen Vektor $\mathbf{x} \in \mathbb{R}^5$. Geben Sie den maximalen relativen Fehler über die fünf Koordinaten an.
    1. Prüfen Sie nun auf dieselbe Weise die *fehlerhafte* Formel $2\mathbf{A}\mathbf{x}$. Dieser Fehler tritt häufig auf, wenn man übersieht, dass $\mathbf{A}$ nicht symmetrisch ist. Wie groß ist der relative Fehler jetzt?
    1. Wiederholen Sie die Prüfung der richtigen Formel $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ in 32-Bit-Arithmetik (`float32`) mit Framework-Tensoren über $h \in \{10^{-1}, 10^{-2}, \ldots, 10^{-6}\}$. Welches $h$ liefert den kleinsten relativen Fehler in `float32`? Warum unterscheidet es sich vom Optimum $h^* \approx 10^{-4}$ für die Vorwärtsdifferenz? Welche Schwelle für den relativen Fehler trennt bei `float32` eine korrekte Implementierung des Gradienten zuverlässig von einer fehlerhaften?

<!--en:f162ccbe-->
[Diskussion](https://d2l.discourse.group/t/32)

<!--en:c02ea5db-->
[Diskussion](https://d2l.discourse.group/t/33)

<!--en:6e9d2b15-->
[Diskussion](https://d2l.discourse.group/t/197)

<!--en:1575f8ea-->
[Diskussion](https://d2l.discourse.group/t/17969)

<!--en:0cc0fd39-->
Differentialrechnung: Ableitungen, Gradienten, Jacobi-Matrizen und Kettenregel<br>**Lokale lineare Näherungen · Tangentensteigungen · steilster Abstieg · azyklische Berechnungsgraphen · Grenzen endlicher Differenzen**

<!--en:9b0361c4-->
::: {.slide title="Wie die Differentialrechnung dem Lernen die Richtung weist"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:b0b2ea51-->
- **Glatte Ersatzverluste für flache 0–1-Genauigkeit:** Der stufenförmige Fehler hat fast überall Steigung null. Glatte Verluste (Kreuzentropie, MSE) liefern eine lokale Tangente $f(x+h) \approx f(x) + f'(x)h$.

<!--en:7233b752-->
- **Ein Gradient oder $d+1$ blinde Versuche:** $d = 10^9$ Gewichte einzeln zu ändern erfordert $10^9$ Vorwärtsdurchläufe. Dagegen zeigt $\nabla_{\mathbf{w}}\mathcal{L}$ nach einem Rückwärtsdurchlauf für alle $d$ Koordinaten die Aufstiegsrichtung.

<!--en:83777c05-->
::: {.slide title="Von der Sekantensteigung zur Ableitung"}

<!--en:bd012d73-->
[Analysis einer Variablen]{.kicker}

<!--en:fbf2101b-->
- Die **Ableitung** $f'(x)$ ist die momentane Änderungsrate, der Grenzwert der **Sekantensteigung** zwischen $(x, f(x))$ und $(x+h, f(x+h))$ für $h \to 0$:
  $$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

<!--en:fd6e97a0-->
- Für $L(w) = 3w^2 - 4w + 2$ bei $w = 1$ ergibt Ausmultiplizieren $L(1+h) = 1 + 2h + 3h^2$ und damit $\frac{L(1+h)-L(1)}{h} = 2 + 3h \to 2$:

<!--en:2de1039e-->
::: {.slide title="Die Tangente als lokale lineare Näherung"}

<!--en:2e4fe053-->
[Lokale Linearität]{.kicker}

<!--en:9b1d2771-->
- **Taylor-Entwicklung erster Ordnung um $a$:**
  $$f(x) \approx f(a) + f'(a)(x - a)$$
  Bei $w = 1$ ($L(1) = 1$, $L'(1) = 2$) lautet die Tangente $T(w) = 1 + 2(w - 1) = 2w - 1$.

<!--en:53adeff8-->
- **Warum lokale Linearität den Gradientenabstieg begründet:** Für einen kleinen Schritt $\Delta w = -\eta L'(w)$ mit Lernrate $\eta > 0$ gilt:
  $$L\big(w - \eta L'(w)\big) \approx L(w) - \eta \big[L'(w)\big]^2 < L(w)$$

<!--en:c87398d1-->
- Ein Schritt von $w = 1$ mit $\eta = 0.1$ ergibt $w = 1 - 0.1(2) = 0.8$ und senkt $L$ von $1.0$ auf $0.72$ (Vorhersage der Tangente: $0.60$). Aus $L'(w^*) = 6w^* - 4 = 0$ folgt das globale Minimum $w^* = 2/3$.

<!--en:d422adfc-->
::: {.slide title="Elementare Ableitungsregeln und die Produktregel"}

<!--en:8761149c-->
[Analytisches Differenzieren]{.kicker}

<!--en:65cfcb23-->
| Funktion | Ableitung $\frac{d}{dx}$ | Regel | Formel |

<!--en:3ae81821-->
| Konstante $C$ | $0$ | **Linearität** | $(\alpha u + \beta v)' = \alpha u' + \beta v'$ |

<!--en:48112488-->
| Potenz $x^n$ | $n x^{n-1}$ | **Produkt** | $(u\,v)' = u'v + u\,v'$ |

<!--en:3def97e7-->
| Exponentialfunktion $e^x$ | $e^x$ | **Quotient** | $(u/v)' = (u'v - u\,v')/v^2$ |

<!--en:df4c84c5-->
| Logarithmus $\ln x$ | $1/x$ | **$3w^2 - 4w + 2$** | $3(2w) - 4(1) = 6w - 4$ |

<!--en:f94a93c3-->
- **Geometrie der Produktregel:** Wächst ein Rechteck der Fläche $f(x)g(x)$ bei einem Zuwachs von $h$, kommen zwei Streifen mit der Fläche $g(x)f'(x)h + f(x)g'(x)h$ und eine Ecke der Ordnung $O(h^2)$ hinzu. Ihr Beitrag verschwindet für $h \to 0$.

<!--en:ef667715-->
::: {.slide title="Partielle Ableitungen an Schnitten durch eine Fläche"}

<!--en:a36b8745-->
[Analysis mehrerer Variablen]{.kicker}

<!--en:f0c122a0-->
- Für $f: \mathbb{R}^n \to \mathbb{R}$ hält die **partielle Ableitung** $\frac{\partial f}{\partial x_i}$ alle Koordinaten außer $x_i$ fest. Sie misst die Steigung in einem eindimensionalen Schnitt:
  $$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

<!--en:9747374f-->
- Für den Verlust des Geradenmodells $L(w_1, w_2) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ bei $\mathbf{w} = (1, 0)$ ergeben die Schnitte $L(w_1, 0) = 3w_1^2 - 4w_1 + 2$ und $L(1, w_2) = 5w_2^2 + 4w_2 + 1$, also $\frac{\partial L}{\partial w_1} = 2$ und $\frac{\partial L}{\partial w_2} = 4$:

<!--en:4ea1b61f-->
::: {.slide title="Der Gradient, die steilsten Richtungen und die Niveaulinien"}

<!--en:3353e0e3-->
[Geometrie des Gradienten]{.kicker}

<!--en:74437d8b-->
- Die $n$ partiellen Ableitungen bilden zusammen den **Gradienten** $\nabla_{\mathbf{x}} f(\mathbf{x}) = \big[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\big]^\top \in \mathbb{R}^n$.

<!--en:0f24fd93-->
- Nach Cauchy-Schwarz beträgt die Richtungsableitung entlang des Einheitsvektors $\mathbf{u}$ im Winkel $\theta$ zu $\nabla f(\mathbf{x})$ genau $D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x})^\top \mathbf{u} = \|\nabla f(\mathbf{x})\|_2 \cos\theta$:
  - **Steilster Anstieg ($\theta = 0^\circ$):** $\nabla f(\mathbf{x})$ zeigt direkt bergauf.
  - **Steilster Abstieg ($\theta = 180^\circ$):** $-\nabla f(\mathbf{x})$ zeigt direkt bergab.
  - **Senkrecht zur Niveaulinie ($\theta = 90^\circ$):** $D_{\mathbf{u}} f = 0$ entlang der Niveaulinien $f(\mathbf{x}) = c$.

<!--en:17539517-->
::: {.slide title="Lineare Näherung in mehreren Variablen und Gradientenabstieg"}

<!--en:106df48d-->
[Optimierung erster Ordnung]{.kicker}

<!--en:8cf03165-->
Werden alle $n$ Koordinaten gleichzeitig um $\boldsymbol{\epsilon} \in \mathbb{R}^n$ verändert, addieren sich die linearen Änderungen entlang der einzelnen Achsen:

<!--en:8a82567b-->
- **Garantierte lokale Abnahme:** Für den Schritt beim Gradientenabstieg $\boldsymbol{\epsilon} = -\eta \nabla_{\mathbf{x}} f(\mathbf{x})$ mit kleiner Lernrate $\eta > 0$ gilt:
  $$f\big(\mathbf{x} - \eta \nabla_{\mathbf{x}} f(\mathbf{x})\big) \approx f(\mathbf{x}) - \eta \big\|\nabla_{\mathbf{x}} f(\mathbf{x})\big\|_2^2 \le f(\mathbf{x})$$

<!--en:2421b398-->
- Solange $\nabla_{\mathbf{x}} f(\mathbf{x}) \neq \mathbf{0}$ gilt und $\eta$ für die lineare Näherung klein genug ist, sinkt $f$ mit jedem Schritt strikt.

<!--en:623ff18a-->
- **Warum der Gradientenabstieg im Zickzack verläuft:** Beim Geradenmodell beträgt der Gradient an der Stelle $\mathbf{w}_0 = [1, 0]^\top$ $\nabla L(\mathbf{w}_0) = [2, 4]^\top$. Ein Schritt mit $\eta = 0.1$ ergibt:
  $$\mathbf{w}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} - 0.1 \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 0.8 \\ -0.4 \end{bmatrix}, \quad L(\mathbf{w}_1) = 0.40 < 1.0$$

<!--en:37669c17-->
- Da $-\nabla L$ senkrecht auf den *elliptischen* Niveaulinien steht, statt zum Zentrum $\mathbf{w}^* = [7/6, -1/2]^\top$ zu zeigen, verläuft die Bahn im Zickzack talwärts.

<!--en:a3bdabb4-->
::: {.slide title="Die mehrdimensionale Kettenregel und azyklische Berechnungsgraphen"}

<!--en:f5f26c40-->
[Verkettungen und gerichtete azyklische Graphen (DAGs)]{.kicker}

<!--en:073f9d6b-->
- **Kettenregel für einen Pfad:** Für $z = f(u)$ und $u = g(x)$ multiplizieren sich die lokalen Änderungsraten entlang der Kantenfolge $x \to u \to z$:
  $$\frac{dz}{dx} = \frac{dz}{du}\frac{du}{dx}$$

<!--en:c1ea5a0b-->
- **Mehrdimensionale Kettenregel auf einem DAG:** Hängt $y = f(\mathbf{u})$ von Zwischenknoten $u_1, \ldots, u_m$ ab und jedes $u_i = g_i(\mathbf{x})$ von den Eingaben $x_1, \ldots, x_n$, gilt:
  $$\frac{\partial y}{\partial x_j} = \sum_{i=1}^m \frac{\partial y}{\partial u_i} \frac{\partial u_i}{\partial x_j} \quad \Longleftrightarrow \quad \nabla_{\mathbf{x}} y = \mathbf{J}^\top \nabla_{\mathbf{u}} y$$

<!--en:3f877869-->
- **Graphenregel:** Kantengewichte (lokale Ableitungen) auf jedem gerichteten Pfad von $x_j$ nach $y$ multiplizieren, dann **über alle parallelen Pfade summieren**.

<!--en:ef967a43-->
::: {.slide title="Rechenbeispiel zur Summe über parallele Pfade im Berechnungsgraphen"}

<!--en:554e0ed7-->
[Die mehrdimensionale Kettenregel in der Praxis]{.kicker}

<!--en:f5ceab83-->
Für $z = u^2 + v$ mit $u = x_1 x_2$ und $v = x_2^2$ an der Stelle $(x_1, x_2) = (1, 2)$ gilt:

<!--en:701e3104-->
- **Vorwärts (blau):** $u = 2,\; v = 4 \implies z = 2^2 + 4 = 8$.

<!--en:0f568de6-->
- **Rückwärts (rot):** Aus dem nachfolgenden Knoten kommen die adjungierten Größen $\frac{\partial z}{\partial u} = 2u = 4$ und $\frac{\partial z}{\partial v} = 1$:
  $$\frac{\partial z}{\partial x_1} = 4 \cdot 2 = 8, \quad \frac{\partial z}{\partial x_2} = 4(1) + 1(4) = 8$$

<!--en:6098f9b6-->
- Die Wiederverwendung von $\frac{\partial z}{\partial u} = 4$ erspart das symbolische Ausmultiplizieren von $z(x_1, x_2)$.

<!--en:710b680e-->
::: {.slide title="Rechenregeln der Matrixanalysis und Jacobi-Matrizen vektorwertiger Funktionen"}

<!--en:08e5232d-->
[Vektor- und Matrixanalysis]{.kicker}

<!--en:ed87852b-->
| Ausdruck | Funktion $f(\mathbf{x})$ | Gradient / Jacobi-Matrix | Anwendung im Deep Learning |

<!--en:f7baacb3-->
| **Linearform** | $\mathbf{a}^\top \mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{a}^\top \mathbf{x}) = \mathbf{a}$ | Lineares Neuron $\mathbf{w}^\top \mathbf{x} + b$ |

<!--en:0569b947-->
| **Quadratische Form** | $\mathbf{x}^\top \mathbf{A} \mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ | Terme der Hesse-Matrix ($2\mathbf{A}\mathbf{x}$ für $\mathbf{A} = \mathbf{A}^\top$) |

<!--en:1cd38ae2-->
| **Quadrierte $L_2$-Norm** | $\|\mathbf{x}\|_2^2 = \mathbf{x}^\top \mathbf{x}$ | $\nabla_{\mathbf{x}}\|\mathbf{x}\|_2^2 = 2\mathbf{x}$ | Gewichtszerfall und MSE-Residuen |

<!--en:ad10c17c-->
| **Kleinste Quadrate** | $\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$ | $\nabla_{\mathbf{w}} L = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ | Batchgradient der linearen Regression |

<!--en:441ef7f1-->
| **Vektorwertige Abbildung** | $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ | $\mathbf{J}_{\mathbf{f}}(\mathbf{x}) \in \mathbb{R}^{m \times n},\; (\mathbf{J}_{\mathbf{f}})_{ij} = \frac{\partial f_i}{\partial x_j}$ | Verkettung von Schichten: $\mathbf{J}_{\mathbf{f} \circ \mathbf{g}} = \mathbf{J}_{\mathbf{f}}\mathbf{J}_{\mathbf{g}}$ |

<!--en:4c9b0701-->
Für jede affine Abbildung $\mathbf{u} = \mathbf{A}\mathbf{x} + \mathbf{b}$ gilt $\mathbf{J}_{\mathbf{u}}(\mathbf{x}) = \mathbf{A}$. Beim Rückwärtsdurchlauf wird $\nabla_{\mathbf{u}} y$ daher mit $\mathbf{A}^\top$ multipliziert: $\nabla_{\mathbf{x}} y = \mathbf{A}^\top \nabla_{\mathbf{u}} y$.

<!--en:dc5877d2-->
::: {.slide title="Von der Kettenregel zur Backpropagation mit Vektor-Jacobi-Produkten"}

<!--en:37c3ee3d-->
[Vorwärts- und Rückwärtsdurchlauf]{.kicker}

<!--en:abcf1fe8-->
- **Graph mit zwei Stufen ($\mathbf{w} \to \mathbf{r} \to L$):** Der Vorwärtsdurchlauf berechnet $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ und $L = \|\mathbf{r}\|_2^2$.

<!--en:2c7c60c4-->
- **Vektor-Jacobi-Produkt rückwärts:** Aus $\nabla_{\mathbf{r}} L = 2\mathbf{r}$ wird $\nabla_{\mathbf{w}} L = \mathbf{X}^\top (2\mathbf{r})$ mit $O(mn)$ FLOPs:

<!--en:67634904-->
::: {.slide title="Warum genügen numerische endliche Differenzen nicht?"}

<!--en:01e101ba-->
[Numerisches und analytisches Differenzieren]{.kicker}

<!--en:2a67ea66-->
Die Näherung $f'(x) \approx \frac{f(x+h) - f(x)}{h}$ hat im Deep Learning **zwei entscheidende Nachteile**:

<!--en:45c0f3f8-->
1. **V-förmige numerische Fehlerkurve:**
   - **Zu großes $h$:** Der Verfahrensfehler der Taylor-Näherung wächst wie $O(h)$ (bei zentralen Differenzen wie $O(h^2)$).
   - **Zu kleines $h$:** Die Subtraktion fast gleicher Gleitkommazahlen $f(x+h) - f(x)$ löscht Mantissenbits aus (**katastrophale Auslöschung**, Fehler $\sim \epsilon / h$, für `float32` bleiben höchstens $\sim 4$ gültige Stellen).

<!--en:b6cf0bff-->
2. **Rechenaufwand $O(d)$:** Das Stören von $d = 10^9$ Gewichten erfordert **$10^9 + 1$ Vorwärtsdurchläufe** pro Schritt. Automatisches Differenzieren im Rückwärtsmodus (§{sec_autograd}) braucht **einen Rückwärtsdurchlauf**.

<!--en:8e717126-->
::: {.slide title="Verständnisfragen zu Gradienten, Niveaulinien und Kettenregel"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:fc0740c9-->
1. **Geometrie des Gradienten:** Bei $\mathbf{x} = (1, 2)$ gilt $\nabla f(\mathbf{x}) = [3, -4]^\top$. Wie lauten der Einheitsvektor $\mathbf{u}^*$ des **steilsten Abstiegs**, die Änderungsrate $D_{\mathbf{u}^*} f(\mathbf{x})$ und ein Tangentenvektor an die Niveaulinie?

<!--en:db4a526f-->
2. **Fallstrick der Matrixanalysis:** Für nicht symmetrisches $\mathbf{A} = [[1, 2], [0, 1]]$ und $\mathbf{x} = [1, 1]^\top$, wie lautet $\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x})$? Warum ist $2\mathbf{A}\mathbf{x}$ falsch?

<!--en:f068524e-->
3. **Skalierung auf tiefe Netze:** Wie viele Vorwärtsdurchläufe benötigen zentrale Differenzen pro Schritt bei $d = 10^8$ Parametern, wie viele die Backpropagation?

<!--en:557b98b6-->
**Lösungen:**

<!--en:13e7aad7-->
1. $\|\nabla f\|_2 = 5 \implies \mathbf{u}^* = \mathbf{[-3/5,\, 4/5]^\top}$ mit Änderungsrate $-\|\nabla f\|_2 = \mathbf{-5}$. Tangentenvektor: $\mathbf{[4, 3]^\top}$ ($3(4) - 4(3) = 0$).

<!--en:c9c83500-->
2. $\nabla_{\mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x} = \mathbf{[4, 4]^\top}$ (wegen $\mathbf{A} \neq \mathbf{A}^\top$ fehlt bei $2\mathbf{A}\mathbf{x} = [6, 2]^\top$ der transponierte Term der Produktregel).

<!--en:06c13fb1-->
3. Zentrale Differenzen benötigen **$2d = 2 \times 10^8$ Vorwärtsdurchläufe**, die Backpropagation **1 Vorwärts- und 1 Rückwärtsdurchlauf** ($\approx 2\text{--}3\times$ ein Vorwärtsdurchlauf).

<!--en:0ce49791-->
::: {.slide title="Analysis im Überblick: Größen, Formen, Geometrie und Optimierung"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:4db68606-->
| Größe | Formel | Form | Geometrische Bedeutung | Rolle im Deep Learning |

<!--en:3d998d7e-->
| **Ableitung** $f'(x)$ | $\lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$ | `()` | Tangentensteigung bei $x$ | Lineare 1D-Näherung $f(x+h) \approx f(x) + f'(x)h$ |

<!--en:66674809-->
| **Partielle Ableitung** $\frac{\partial f}{\partial x_i}$ | Ableitung bei festem $x_{j \neq i}$ | `()` | Steigung im Schnitt entlang Achse $i$ | Empfindlichkeit des Verlusts gegenüber einem Gewicht $w_i$ |

<!--en:4d8c76bd-->
| **Gradient** $\nabla_{\mathbf{x}} f(\mathbf{x})$ | $[\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^\top$ | **`(n,)`** | Zeigt bergauf, $\perp$ zu Niveaulinien | Abstiegsschritt $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}\mathcal{L}$ |

<!--en:2d9a0cbf-->
| **Jacobi-Matrix** $\mathbf{J}_{\mathbf{f}}(\mathbf{x})$ | $(\mathbf{J}_{\mathbf{f}})_{ij} = \partial f_i / \partial x_j$ | **`(m, n)`** | Lokale lineare Abbildung $\mathbf{J}\mathbf{h}$ | Matrix der Empfindlichkeiten zwischen Schichten |

<!--en:8b4f33b9-->
| **Kettenregel (VJP)** | $\nabla_{\mathbf{x}} y = \mathbf{J}_{\mathbf{u}}(\mathbf{x})^\top \nabla_{\mathbf{u}} y$ | `(n,)` | Summe über alle DAG-Pfade | Backpropagation im Rückwärtsmodus (§{sec_autograd}) |

<!--en:bde749dc-->
| **Endliche Differenz** | $\frac{f(x+h)-f(x-h)}{2h}$ | $2d$ Auswertungen | Sekantensteigung auf $[x-h, x+h]$ | Gradientenprüfung in Softwaretests mit `float64` |
