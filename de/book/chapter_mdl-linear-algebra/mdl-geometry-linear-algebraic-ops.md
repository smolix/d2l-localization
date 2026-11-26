<!--en:9c9d630d-->
# Vektoren, Winkel und Projektionen

<!--en:5bc50d68-->
In den Grundlagenkapiteln haben wir jede Eingabe als Vektor dargestellt. Aus einem Bild wurde eine Liste von Pixelwerten, aus einem Wort eine Einbettung (engl. embedding). Im Kern führen Modelle des maschinellen Lernens mit diesen Vektoren drei geometrische Operationen aus. Sie *vergleichen* Vektoren, etwa wenn wir messen, wie ähnlich sich zwei Einbettungen sind. Sie *passen* Vektoren *an*, etwa wenn wir die Kombination von Merkmalen suchen, die gegebene Zielwerte am besten trifft. Und sie *trennen* Vektoren, etwa wenn ein Klassifikator entscheidet, auf welcher Seite einer Grenze eine Eingabe liegt. In der zweidimensionalen Ebene kennen wir für jede dieser Operationen ein anschauliches Bild mit Längen, Winkeln und Schatten. Reale Eingaben liegen jedoch in Räumen mit Hunderten oder Tausenden von Dimensionen, die niemand zeichnen kann. Haben die geometrischen Bilder dort noch eine Bedeutung?

<!--en:7ee4d288-->
Das lässt sich an einer konkreten Aufgabe prüfen. Wir wollen T-Shirts von Hosen unterscheiden. Jedes $28 \times 28$ große Bild aus Fashion-MNIST (:numref:`sec_fashion_mnist`) ist eine Liste von $784$ Pixelintensitäten, also ein Punkt in $\mathbb{R}^{784}$. Betrachten wir die denkbar einfachste Regel. Wir berechnen das mittlere T-Shirt-Bild und das mittlere Hosenbild. Jedes neue Bild ordnen wir dem näher gelegenen Mittelwert zu. In der Ebene sehen wir mit bloßem Auge, was „näher“ bedeutet. In $\mathbb{R}^{784}$ genügt dafür ein einziges Skalarprodukt (engl. dot product) pro Bild. Bei den $2{,}000$ Testbildern dieser beiden Klassen liegt die Regel in etwa $92\%$ der Fälle richtig. Dass eine so einfache Regel so gut funktioniert, zeigt, wie weit die beiden Punktwolken auseinanderliegen. Geometrische Begriffe wie der Abstand behalten also auch dort ihre Bedeutung, wo wir sie nicht mehr veranschaulichen können.

<!--en:cbcdeee9-->
Das Skalarprodukt überträgt die Geometrie der Ebene auf beliebig viele Dimensionen. Längen sind Quadratwurzeln aus Skalarprodukten. Den Schatten eines Vektors auf einen anderen bestimmen wir über einen Quotienten aus Skalarprodukten, und den Winkel zwischen zwei Vektoren definieren wir direkt über Skalarprodukte. Wir entwickeln diese Geometrie in drei Schritten. Zuerst betrachten wir zwei Vektoren, die Projektion auf eine Gerade, den Winkel zwischen den Vektoren und die Cauchy-Schwarz-Ungleichung. In :numref:`sec_linear-algebra` haben wir diese Ungleichung algebraisch bewiesen. Hier beweisen wir sie erneut anhand eines rechtwinkligen Dreiecks. Dann führen wir die Begriffe für Unterräume ein, die wir im weiteren Kapitel brauchen (Spann, lineare Unabhängigkeit, Basis und Dimension). Wir lösen das Problem der kleinsten Quadrate (engl. least squares) durch Projektion und verwenden für die Berechnung die QR-Zerlegung. Schließlich wenden wir diese Ideen auf hochdimensionale Bilddaten an. Wir untersuchen, wie ähnlich einander inhaltlich unverbundene Vektoren in $\mathbb{R}^d$ sind, und deuten einen linearen Klassifikator als Hyperebene.

<!--en:0182e39c-->
Drei Ergebnisse sind besonders wichtig. Das Residuum einer Projektion steht senkrecht auf der Geraden oder dem Unterraum, auf den wir projizieren. Deshalb liefert die Projektion den nächstgelegenen Punkt. Die Cauchy-Schwarz-Ungleichung stellt sicher, dass Winkel in jeder Dimension wohldefiniert sind. Und der Kosinus hat typischerweise die Größenordnung $1/\sqrt{d}$, wenn wir zufällige Richtungen in $\mathbb{R}^d$ vergleichen. Bis auf das Laden des Datensatzes verwendet der Code in diesem Abschnitt ausschließlich NumPy und läuft in allen Frameworks gleich.

<!--en:43823348-->
## Vektoren und ihre Geometrie

<!--en:d7802703-->
### Punkte und Richtungen

<!--en:cb61033d-->
Wir können uns einen Vektor auf zwei Arten vorstellen, die einander ergänzen. Als *Punkt* markiert $\mathbf{v} = (3, 2)$ eine Stelle, die $3$ Einheiten rechts vom *Ursprung* und $2$ Einheiten über ihm liegt. So wird ein ganzer Datensatz zu einer Punktwolke. Die Klassifikation von T-Shirts und Hosen wird zur geometrischen Aufgabe, zwei Punktwolken zu trennen. Als *Richtung* (oder Verschiebung) gibt derselbe Vektor an, dass wir $3$ Schritte nach rechts und $2$ Schritte nach oben gehen. Da eine Verschiebung überall beginnen kann, stellen alle parallelen Pfeile in :numref:`fig_mdl-la-vectors` denselben Vektor dar.

<!--en:95cb6fca-->
![Zwei Sichtweisen auf einen Vektor. Als *Punkt* hat er die Koordinaten $(x, y)$ (links). Als *Richtung* ist er ein Pfeil, der überall beginnen kann. Daher stellen alle gezeigten Pfeile denselben Vektor $(3, 2)$ dar (rechts).](../img/mdl-la-vectors.svg)

<!--en:6427ec43-->
Deuten wir Vektoren als Richtungen, ist die Addition unmittelbar anschaulich. Wir folgen dem ersten Pfeil und von seiner Spitze aus dem zweiten, wie :numref:`fig_mdl-la-vector-add` zeigt.

<!--en:32984767-->
![Addition zweier Vektoren. Wir folgen dem ersten Pfeil und setzen am Ende des ersten den zweiten an.](../img/mdl-la-vector-add.svg)

<!--en:5e069ed1-->
Die Subtraktion funktioniert ebenso. Wegen $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$ ist die Differenz $\mathbf{u}-\mathbf{v}$ der Verschiebungspfeil, der vom Punkt $\mathbf{v}$ zum Punkt $\mathbf{u}$ führt.

<!--en:63536ed4-->
Alles Weitere in diesem Abschnitt beruht auf einer einzigen Operation, dem *Skalarprodukt* aus :numref:`sec_linear-algebra`. Für zwei Vektoren $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ multipliziert es jeweils die einander entsprechenden Einträge und summiert die Produkte:

<!--en:d4c9e49e-->
Das Skalarprodukt ist symmetrisch, es gilt also $\mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u}$. Häufig schreiben wir es als $\mathbf{u}\cdot\mathbf{v}$. Den Punkt verwenden wir, wenn wir an zwei Pfeile denken, die Transponierte, wenn das Produkt in einem Matrixausdruck steht. Für $\mathbf{u} = (1, 3)$ und $\mathbf{v} = (2, 1)$ gilt $\mathbf{u}\cdot\mathbf{v} = 2 + 3 = 5$, und das Längenquadrat von $\mathbf{u}$ beträgt $\mathbf{u}\cdot\mathbf{u} = 1 + 9 = 10$.

<!--en:4f07e707-->
### Projektion auf eine Gerade

<!--en:fb98743b-->
Welcher Anteil eines Vektors $\mathbf{v}$ weist entlang eines anderen Vektors $\mathbf{w}$? Gleichbedeutend damit ist die Frage, welches skalare Vielfache $t\mathbf{w}$ dem Vektor $\mathbf{v}$ am nächsten liegt. Das ist das einfachste Ausgleichsproblem der linearen Algebra. Wir nähern $\mathbf{v}$ mit einer einzigen Richtung an und suchen den besten Koeffizienten $t$.

<!--en:7fb2c9c3-->
Probieren wir für unser Vektorpaar $\mathbf{v} = (1, 3)$ und $\mathbf{w} = (2, 1)$ einige Werte von $t$ aus. Bei $t = 1$ hat der Fehler $\mathbf{v} - t\mathbf{w} = (-1, 2)$ das Längenquadrat $5$. Bei $t = 0.5$ erhalten wir den Fehler $(0, 2.5)$ mit dem Längenquadrat $6.25$, bei $t = 1.5$ den Fehler $(-2, 1.5)$ mit demselben Längenquadrat $6.25$. Das Minimum liegt bei $t = 1$ und liefert den nächstgelegenen Punkt $(2, 1)$. Den verbleibenden Fehler nennen wir *Residuum*. Hier beträgt das Residuum $\mathbf{r} = (-1, 2)$. Betrachten wir sein Skalarprodukt mit $\mathbf{w}$. Es gilt $\mathbf{r}\cdot\mathbf{w} = -2 + 2 = 0$. Der kürzeste Fehlervektor trifft die Gerade im rechten Winkel. Mehr noch, für die Längenquadrate gilt dieselbe Beziehung wie im rechtwinkligen Dreieck. Es gilt $\|\mathbf{v}\|^2 = 10$, der nächstgelegene Punkt hat das Längenquadrat $5$, das Residuum das Längenquadrat $5$, und es gilt $10 = 5 + 5$.

<!--en:78ef8837-->
Diese Beziehung gilt in jeder Dimension.

<!--en:f11f6e46-->
**Satz (orthogonale Projektion).** *Sei $\mathbf{w} \neq \mathbf{0}$. Das Vielfache von $\mathbf{w}$, das $\mathbf{v}$ am nächsten liegt, ist*

<!--en:b4cb8a0e-->
*und für das Residuum $\mathbf{r} = \mathbf{v} - \operatorname{proj}_{\mathbf{w}}\mathbf{v}$ gilt $\mathbf{r}\cdot\mathbf{w} = 0$.*

<!--en:7392742b-->
**Beweis.** Wir minimieren den quadratischen Fehler als Funktion von $t$. Multiplizieren wir das Skalarprodukt von $\mathbf{v} - t\mathbf{w}$ mit sich selbst aus, erhalten wir eine quadratische Funktion von $t$,

<!--en:ad464355-->
Da der Leitkoeffizient $\|\mathbf{w}\|^2$ positiv ist, hat $q(t)$ ein eindeutiges Minimum an der Nullstelle der Ableitung $q'(t) = 2\|\mathbf{w}\|^2 t - 2\,\mathbf{v}\cdot\mathbf{w}$, also bei $t^\star = (\mathbf{v}\cdot\mathbf{w}) / (\mathbf{w}\cdot\mathbf{w})$. Setzen wir $t^\star$ ein, erhalten wir :eqref:`eq_mdl-projection`. Für das Residuum ergibt sich $\mathbf{r}\cdot\mathbf{w} = \mathbf{v}\cdot\mathbf{w} - t^\star\,\mathbf{w}\cdot\mathbf{w} = 0$. $\blacksquare$

<!--en:64f6be81-->
Für unser Beispiel erhalten wir $q(t) = 5t^2 - 10t + 10$. Diese Funktion erreicht ihren kleinsten Wert $5$ bei $t^\star = 5/5 = 1$, wie unsere Proberechnung nahelegt. Wir nennen $\operatorname{proj}_{\mathbf{w}}\mathbf{v}$ die *Projektion* oder den *Schatten* von $\mathbf{v}$ auf $\mathbf{w}$. Die Projektion erfasst den Anteil von $\mathbf{v}$ entlang von $\mathbf{w}$, das Residuum den dazu senkrechten Anteil. Damit wird auch die Voraussetzung $\mathbf{w} \neq \mathbf{0}$ verständlich. Der Nullvektor legt keine Richtung fest, und mit $\mathbf{w} = \mathbf{0}$ würden wir durch null teilen.

<!--en:3c8c7e7e-->
Wegen dieses rechten Winkels addieren sich die Längenquadrate. Zwei Vektoren $\mathbf{a}$ und $\mathbf{b}$ heißen *orthogonal*, wenn $\mathbf{a}\cdot\mathbf{b} = 0$ gilt. Multiplizieren wir dann das Längenquadrat ihrer Summe aus, erhalten wir den *Satz von Pythagoras*,

<!--en:17fcc80b-->
In jeder Dimension addieren sich die Längenquadrate orthogonaler Komponenten ohne gemischten Term. Für die Zerlegung $\mathbf{v} = \operatorname{proj}_{\mathbf{w}}\mathbf{v} + \mathbf{r}$ ergibt sich damit $\|\mathbf{v}\|^2 = \|\operatorname{proj}_{\mathbf{w}}\mathbf{v}\|^2 + \|\mathbf{r}\|^2$, in unserem Beispiel also $10 = 5 + 5$.

<!--en:6bd8ab0a-->
Wir brauchen außerdem die Länge des Schattens mit einem Vorzeichen, das angibt, ob er in dieselbe Richtung wie $\mathbf{w}$ oder in die Gegenrichtung weist. Multiplizieren wir den Koeffizienten $t^\star$ mit $\|\mathbf{w}\|$, erhalten wir diese *vorzeichenbehaftete Länge*:

<!--en:e2b6348d-->
In unserem Beispiel beträgt sie $5/\sqrt{5} = \sqrt{5}$, die Länge von $(2, 1)$. Am Ende des Abschnitts begegnet uns diese Länge wieder, und zwar als vorzeichenbehafteter Abstand eines Punkts von einer Hyperebene. :numref:`fig_mdl-la-projection` veranschaulicht die geometrischen Zusammenhänge.

<!--en:68c6ab69-->
![(a) Die Projektion (der Schatten) von $\mathbf{v}=(1, 3)$ auf $\mathbf{w}=(2, 1)$ ist $(2, 1)$. Das Residuum $(-1, 2)$ steht senkrecht auf $\mathbf{w}$, also gilt $\|\mathbf{v}\|^2 = 10 = 5 + 5$. (b) Gleichheit in der Cauchy-Schwarz-Ungleichung. Hier ist $\mathbf{v}$ kollinear zu $\mathbf{w}$ und das Residuum verschwindet.](../img/mdl-la-projection.svg)

<!--en:5157fe18-->
### Die Cauchy-Schwarz-Ungleichung und der Winkel

<!--en:5034ed7d-->
In $784$ Dimensionen gibt es keinen Winkelmesser. Hat der „Winkel zwischen zwei Bildern“ dort noch eine Bedeutung? Überraschenderweise liefert eine viel einfachere Frage die Antwort. Kann der Schatten eines Vektors länger sein als der Vektor selbst? Teil (a) von :numref:`fig_mdl-la-projection` zeigt, dass dies unmöglich ist. Der Schatten ist eine Kathete eines rechtwinkligen Dreiecks mit der Hypotenuse $\mathbf{v}$. Mit dem Satz von Pythagoras wird aus diesem Bild ein Beweis für jede Dimension.

<!--en:20fd32c9-->
**Satz (Cauchy-Schwarz-Ungleichung).** *Für alle Vektoren $\mathbf{v}, \mathbf{w}$ gilt*

<!--en:c2873e0f-->
*wobei Gleichheit genau dann gilt, wenn $\mathbf{v}$ und $\mathbf{w}$ kollinear sind (einer ist ein skalares Vielfaches des anderen).*

<!--en:9c1f295e-->
**Beweis.** Für $\mathbf{w} = \mathbf{0}$ verschwinden beide Seiten und die Vektoren sind kollinear. Andernfalls zerlegen wir $\mathbf{v}$ in seine Projektion auf $\mathbf{w}$ und sein orthogonales Residuum $\mathbf{r}$. Nach dem Satz von Pythagoras :eqref:`eq_mdl-pythagoras` gilt

<!--en:cd096625-->
Multiplizieren wir beide Seiten mit $\|\mathbf{w}\|^2$ und ziehen die Quadratwurzel, erhalten wir die Ungleichung. Gleichheit gilt genau dann, wenn $\mathbf{r} = \mathbf{0}$, also $\mathbf{v} = t^\star\mathbf{w}$ gilt. Gilt umgekehrt $\mathbf{v} = t\mathbf{w}$, so folgt $|\mathbf{v}\cdot\mathbf{w}| = |t|\,\|\mathbf{w}\|^2 = \|\mathbf{v}\|\,\|\mathbf{w}\|$. $\blacksquare$

<!--en:fa7a1120-->
Anschaulich ist der Schatten eines Vektors nie länger als der Vektor selbst. Er ist nur dann genauso lang, wenn das Residuum verschwindet. In :numref:`sec_linear-algebra` haben wir die Cauchy-Schwarz-Ungleichung algebraisch bewiesen. Hier folgt sie direkt aus Teil (a) von :numref:`fig_mdl-la-projection`. Teil (b) zeigt den Fall der Gleichheit.

<!--en:ad1703e5-->
Nun können wir Winkel in jeder Dimension definieren. In der Ebene liefert der Kosinussatz $\mathbf{v}\cdot\mathbf{w} = \|\mathbf{v}\|\|\mathbf{w}\|\cos\theta$ (:numref:`sec_linear-algebra`). In $\mathbb{R}^d$ haben wir kein konkretes Dreieck, an dem wir messen könnten. Daher verwenden wir diese Identität als Definition. Für zwei von null verschiedene Vektoren $\mathbf{v}, \mathbf{w} \in \mathbb{R}^d$ ist der *Winkel* zwischen ihnen gegeben durch

<!--en:6d39c8db-->
Da $\arccos$ nur auf $[-1, 1]$ definiert ist, ergibt diese Formel nur dann Sinn, wenn der Quotient in diesem Intervall liegt. Genau das garantiert die Cauchy-Schwarz-Ungleichung. Damit ist der Winkel in jeder Dimension wohldefiniert, und die Formel für das Skalarprodukt in der Ebene gilt unmittelbar auch in beliebigen Dimensionen:

<!--en:e027dd9b-->
Die beiden Fälle der Gleichheit in der Cauchy-Schwarz-Ungleichung entsprechen $\theta = 0$ (gleichgerichtete Vektoren) und $\theta = \pi$ (entgegengesetzt gerichtete Vektoren).

<!--en:fd46839c-->
Stimmt diese algebraische Definition mit dem Winkel überein, den ein Winkelmesser tatsächlich anzeigen würde? Ja. Zwei Vektoren liegen stets in einer gemeinsamen Ebene (oder auf einer Geraden, wenn sie kollinear sind). In dieser Ebene gilt die gewöhnliche Geometrie. Durch Projektion erhalten wir rechtwinklige Koordinaten für die Ebene. Setzen wir $\mathbf{q}_1 = \mathbf{w}/\|\mathbf{w}\|$ und, sofern $\mathbf{r} \neq \mathbf{0}$ gilt, $\mathbf{q}_2 = \mathbf{r}/\|\mathbf{r}\|$. Diese beiden Vektoren haben die Länge eins und sind orthogonal. Daher verhalten sich Skalarprodukte ihrer Linearkombinationen genauso wie Skalarprodukte in $\mathbb{R}^2$. In diesem Koordinatensystem gilt $\mathbf{w} = (\|\mathbf{w}\|, 0)$ und $\mathbf{v} = (\mathbf{v}\cdot\mathbf{w}/\|\mathbf{w}\|,\ \|\mathbf{r}\|)$. Der mit dem Winkelmesser zwischen diesen beiden Pfeilen in der Ebene gemessene Winkel hat den Kosinus $(\mathbf{v}\cdot\mathbf{w}/\|\mathbf{w}\|)/\|\mathbf{v}\|$, in Übereinstimmung mit :eqref:`eq_mdl-angle_formula`. Für unser Beispiel mit $\mathbf{v} = (1, 3)$ und $\mathbf{w} = (2, 1)$ erhalten wir $\cos\theta = 5/(\sqrt{10}\sqrt{5}) = 1/\sqrt{2}$, also $\theta = 45^\circ$.

<!--en:c37d00ee-->
Für Vektoren ungleich null bedeutet Orthogonalität $\theta = \pi/2$. Die Definition über $\mathbf{v}\cdot\mathbf{w} = 0$ hat den kleinen Vorteil, dass sie auch den Nullvektor erfasst. Er ist orthogonal zu jedem Vektor, obwohl für ihn kein Winkel definiert ist.

<!--en:5ee71a04-->
Für die Umsetzung von :eqref:`eq_mdl-angle_formula` in Code genügen zwei Zeilen. Dabei lohnt sich ein Blick auf eine praktische Vorkehrung. Testen wir den Code mit unserem Beispiel und mit einem bewusst heikel gewählten kollinearen Paar, $\mathbf{v} = (10^8 + 1, 3, 7)$ und $3\mathbf{v}$. Dessen Kosinus beträgt exakt $1$ und sein Winkel exakt $0$. Für dieses Paar geben wir zusätzlich den direkt berechneten Kosinus abzüglich $1$ und das Ergebnis von `arccos` ohne vorherige Begrenzung aus.

<!--en:3f9251d9-->
Die erste Zahl ist $0.785 = \pi/4$ und entspricht dem Winkel von $45^\circ$ aus unserem Beispiel. Die übrigen drei Werte zeigen, warum `np.clip` unverzichtbar ist. Bei exakter Rechnung hält die Cauchy-Schwarz-Ungleichung den Quotienten innerhalb von $[-1, 1]$. In der Gleitkommaarithmetik kann er durch Rundung um eine Einheit der letzten Stelle über die Grenze rutschen. Hier ergibt sich für das kollineare Paar $1 + 2.2 \times 10^{-16}$, sodass `np.arccos` den Wert `nan` zurückgibt. Begrenzen wir den Quotienten auf $[-1, 1]$, erhalten wir das richtige Ergebnis $0$.

<!--en:7e89e81d-->
Aus der Cauchy-Schwarz-Ungleichung folgt auch, dass die Länge einer Summe höchstens so groß ist wie die Summe der Längen. Wir multiplizieren $\|\mathbf{v} + \mathbf{w}\|^2$ aus und schätzen den gemischten Term $2\,\mathbf{v}\cdot\mathbf{w}$ nach oben durch $2\,\|\mathbf{v}\|\|\mathbf{w}\|$ ab, wie :eqref:`eq_mdl-cauchy-schwarz` erlaubt. So erhalten wir die *Dreiecksungleichung* $\|\mathbf{v} + \mathbf{w}\| \le \|\mathbf{v}\| + \|\mathbf{w}\|$ wie in :numref:`sec_linear-algebra`.

<!--en:1b1046c8-->
## Unterräume und kleinste Quadrate

<!--en:d3495f0e-->
### Spann, Basen und Unterräume

<!--en:419d5f9e-->
Angenommen, wir messen eine Größe zu drei Zeitpunkten $t = 0, 1, 2$ und suchen die Gerade $C + Dt$, die am besten zu unseren Daten passt. Wir fassen die Messwerte im Vektor $\mathbf{y} \in \mathbb{R}^3$ und die Zeitpunkte in $\mathbf{t} = (0, 1, 2)$ zusammen und setzen $\mathbf{1} = (1, 1, 1)$. Werten wir die Gerade an den drei Zeitpunkten aus, erhalten wir den Vektor $C\mathbf{1} + D\mathbf{t}$. Die beste Gerade zu finden bedeutet also, unter allen Vektoren der Form $C\mathbf{1} + D\mathbf{t}$ denjenigen zu finden, der $\mathbf{y}$ am nächsten liegt. Wenn $C$ und $D$ alle reellen Zahlen durchlaufen, überstreichen diese Vektoren eine Ursprungsebene in $\mathbb{R}^3$. Bei der Projektion auf eine Gerade hatten wir es mit einer einzigen Richtung zu tun. Für Ebenen und ihre höherdimensionalen Entsprechungen brauchen wir einige Grundbegriffe. Wenn Sie mit Spann, Basen und Unterräumen bereits vertraut sind, können Sie direkt zum nächsten Unterabschnitt springen. Dort bestimmen wir die gesuchte Gerade.

<!--en:bef40482-->
Eine gewichtete Summe $a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k$ heißt *Linearkombination* von $\mathbf{v}_1, \ldots, \mathbf{v}_k$. Die Menge aller solchen Kombinationen ist ihr *Spann*,

<!--en:a9e21417-->
In $\mathbb{R}^2$ (:numref:`fig_mdl-la-span`) ist der Spann eines einzelnen Vektors ungleich null die Ursprungsgerade in seiner Richtung. Zwei Vektoren, die auf keiner gemeinsamen Ursprungsgeraden liegen, spannen dagegen die gesamte Ebene auf. Bei unserem Ausgleichsproblem bilden die möglichen Anpassungen die Menge $\operatorname{span}(\mathbf{1}, \mathbf{t})$.

<!--en:e12cb16d-->
![Links bildet der Spann eines einzelnen Vektors $\mathbf{v}$ ungleich null die Menge aller seiner skalaren Vielfachen. Diese Ursprungsgerade ist ein eindimensionaler Unterraum. Rechts spannen zwei Vektoren, die auf keiner gemeinsamen Ursprungsgeraden liegen, die gesamte Ebene auf. Das gestrichelte Parallelogramm zeigt die Zerlegung von $\mathbf{x}$ als $2\mathbf{u} + \mathbf{w}$. Die Koordinaten sind eindeutig, weil $\mathbf{u}$ und $\mathbf{w}$ linear unabhängig sind.](../img/mdl-la-span.svg)

<!--en:b5eefee7-->
Ein Spann ist stets unter Addition und Skalarmultiplikation abgeschlossen. Addieren wir zwei Linearkombinationen oder multiplizieren wir eine mit einer reellen Zahl, erhalten wir wieder eine Linearkombination. Jede nichtleere Vektormenge mit diesen beiden Eigenschaften heißt *Unterraum*. Jeder Unterraum enthält den Nullvektor $\mathbf{0}$, denn wir können jedes seiner Elemente mit $0$ multiplizieren. Eine Gerade oder Ebene, die nicht durch den Ursprung verläuft, ist also kein Unterraum. In $\mathbb{R}^2$ sind die einzigen Unterräume der Ursprung allein, die Ursprungsgeraden und $\mathbb{R}^2$ selbst. In $\mathbb{R}^3$ kommen die Ursprungsebenen hinzu.

<!--en:db1a9cbc-->
Eine Liste von Vektoren, die einen Unterraum aufspannt, kann überflüssige Vektoren enthalten. Liegt ein Vektor bereits im Spann der übrigen, können wir ihn streichen, ohne den Spann zu verkleinern. Eine Liste von Vektoren heißt *linear unabhängig*, wenn keiner überflüssig ist. Dann ist die einzige Linearkombination $a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k$ mit dem Wert $\mathbf{0}$ die triviale Kombination, für die alle Koeffizienten $a_i = 0$ erfüllen. Unsere Vektoren $\mathbf{1}$ und $\mathbf{t}$ sind linear unabhängig, denn $a_1\mathbf{1} + a_2\mathbf{t} = (a_1,\, a_1 + a_2,\, a_1 + 2a_2)$ verschwindet nur für $a_1 = a_2 = 0$.

<!--en:67c38f91-->
Eine *Basis* eines Unterraums ist eine Liste linear unabhängiger Vektoren, die ihn aufspannen. Die vertrauten Vektoren $\mathbf{e}_1 = (1, 0)$ und $\mathbf{e}_2 = (0, 1)$ bilden die *Standardbasis* von $\mathbb{R}^2$. Das schräg liegende Vektorpaar in :numref:`fig_mdl-la-span` ist jedoch eine ebenso gute Basis. Ein grundlegender Satz der linearen Algebra, den wir ohne Beweis verwenden, besagt, dass jede Basis eines gegebenen Unterraums dieselbe Anzahl von Vektoren hat. Diese Anzahl ist die *Dimension* des Unterraums. Damit erhält das $d$ in „$d$-dimensionaler Raum“ eine genaue Bedeutung. $\mathbb{R}^d$ hat die Dimension $d$, weil $\mathbf{e}_1, \ldots, \mathbf{e}_d$ eine Basis ist. Ebenso bilden die möglichen Anpassungen in unserem Geradenbeispiel einen zweidimensionalen Unterraum von $\mathbb{R}^3$.

<!--en:317ca8f8-->
Die lineare Unabhängigkeit sichert jedem Punkt eines Unterraums eindeutige Koordinaten. Ist $\mathbf{v}_1, \ldots, \mathbf{v}_k$ eine Basis eines Unterraums $S$, so lässt sich jedes $\mathbf{x} \in S$ auf genau eine Weise als $\mathbf{x} = a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k$ schreiben. Mindestens eine Darstellung existiert, weil die Basis $S$ aufspannt. Gäbe es zwei Darstellungen mit den Koeffizienten $a_i$ und $b_i$, so ergäbe ihre Differenz $\sum_i (a_i - b_i)\,\mathbf{v}_i = \mathbf{0}$. Aus der linearen Unabhängigkeit folgt dann $a_i = b_i$ für jedes $i$.

<!--en:ad2b1e90-->
Haben wir eine Basis gewählt, verhält sich ein $k$-dimensionaler Unterraum wie $\mathbb{R}^k$. Jeden Punkt beschreiben seine Koeffizienten $(a_1, \ldots, a_k)$ eindeutig. Für unsere Ausgleichsgerade sind diese Koordinaten der Achsenabschnitt $C$ und die Steigung $D$.

<!--en:629b53ef-->
### Projektion auf einen Unterraum

<!--en:381ab7b6-->
Kehren wir mit konkreten Messwerten $\mathbf{y} = (6, 0, 0)$ zur Ausgleichsgeraden zurück. Gesucht ist der Punkt $C\mathbf{1} + D\mathbf{t}$ in der Ebene $\operatorname{span}(\mathbf{1}, \mathbf{t})$, der $\mathbf{y}$ am nächsten liegt. Bei der Projektion auf eine Gerade stand das Residuum zum nächstgelegenen Punkt senkrecht auf der Geraden. Bei der Projektion auf eine Ebene sollte das Residuum auf der gesamten Ebene senkrecht stehen, also sowohl auf $\mathbf{1}$ als auch auf $\mathbf{t}$. Das beweisen wir gleich. Setzen wir beide Skalarprodukte gleich null, erhalten wir zwei lineare Gleichungen mit zwei Unbekannten:

<!--en:223d0bdd-->
Ziehen wir die erste Gleichung von der zweiten ab, erhalten wir $2D = -6$, also $D = -3$ und $C = 5$. Die Ausgleichsgerade ist $5 - 3t$. An den Zeitpunkten $t = 0, 1, 2$ liefert sie die Vorhersagen $(5, 2, -1)$, sodass das Residuum $\mathbf{r} = \mathbf{y} - (5, 2, -1) = (1, -2, 1)$ bleibt. Beide Orthogonalitätsbedingungen sind erfüllt, denn es gilt $\mathbf{r}\cdot\mathbf{1} = 1 - 2 + 1 = 0$ und $\mathbf{r}\cdot\mathbf{t} = 0 - 2 + 2 = 0$. Der Satz von Pythagoras zerlegt das Längenquadrat von $\mathbf{y}$ wieder in eine Summe, $36 = 30 + 6$. Projizieren wir zum Vergleich $\mathbf{y}$ allein auf $\mathbf{1}$, erhalten wir die konstante Anpassung $\mathbf{y}\cdot\mathbf{1}/\mathbf{1}\cdot\mathbf{1} = 2$ (den Stichprobenmittelwert). Ihr Residuum $(4, -2, -2)$ hat das Längenquadrat $24$. Lassen wir eine Steigung entlang von $\mathbf{t}$ zu, sinkt der quadratische Fehler von $24$ auf $6$. :numref:`fig_mdl-la-least-squares` zeigt diese Projektion in $\mathbb{R}^3$.

<!--en:5881c1fd-->
![Die Methode der kleinsten Quadrate als Projektion. Die Messwerte $\mathbf{y}=(6, 0, 0)$ liegen außerhalb der Ebene der möglichen Geraden $C\mathbf{1}+D\mathbf{t}$. Der nächstgelegene Punkt in der Ebene ist $(5, 2, -1)$ (die Gerade $5-3t$). Das Residuum $(1, -2, 1)$ steht sowohl auf $\mathbf{1}$ als auch auf $\mathbf{t}$ senkrecht.](../img/mdl-la-least-squares.svg)

<!--en:806b6db2-->
Um das allgemeine Ergebnis zu formulieren und zu beweisen, wählen wir zweckmäßig eine Basis aus paarweise orthogonalen Vektoren der Länge eins. Eine Basis $\mathbf{q}_1, \ldots, \mathbf{q}_k$ eines Unterraums $S \subseteq \mathbb{R}^n$ heißt *orthonormal*, wenn für jeden Vektor $\|\mathbf{q}_i\| = 1$ und für verschiedene Vektoren $\mathbf{q}_i\cdot\mathbf{q}_j = 0$ gilt. Aus jedem endlichen Erzeugendensystem lässt sich eine solche Basis gewinnen. In :numref:`subsec_mdl-gram-schmidt` konstruieren wir sie allein durch Projektion auf Geraden. Schreiben wir die Basisvektoren als Spalten einer $n \times k$-Matrix $\mathbf{Q}$, nimmt die Orthonormalität die kompakte Form $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}_k$ an. Denn der Eintrag $(i, j)$ von $\mathbf{Q}^\top\mathbf{Q}$ ist $\mathbf{q}_i\cdot\mathbf{q}_j$. In diesem Kapitel steht der Buchstabe $\mathbf{Q}$ stets für eine Matrix mit orthonormalen Spalten. Auf den folgenden Satz greifen wir im weiteren Verlauf des Kapitels am häufigsten zurück.

<!--en:1065ae07-->
**Satz (Projektion auf einen Unterraum).** *Die Matrix $\mathbf{Q}$ habe orthonormale Spalten, die $S$ aufspannen, und es sei $\mathbf{P} =
\mathbf{Q}\mathbf{Q}^\top$. Dann gilt für jedes $\mathbf{x} \in \mathbb{R}^n$:* (i) *Es gilt $\mathbf{P}\mathbf{x} \in S$, und das Residuum $\mathbf{x} -
\mathbf{P}\mathbf{x}$ ist orthogonal zu jedem Vektor aus $S$.* (ii) *$\mathbf{P}\mathbf{x}$ ist der eindeutig bestimmte Punkt in $S$ mit dem kleinsten Abstand zu $\mathbf{x}$.* (iii) *Es gilt $\mathbf{P}^2 = \mathbf{P}$ und $\mathbf{P}^\top = \mathbf{P}$.*

<!--en:ac543032-->
**Beweis.** Zu (i). Der Vektor $\mathbf{P}\mathbf{x} = \mathbf{Q}(\mathbf{Q}^\top\mathbf{x})$ ist eine Linearkombination der Spalten von $\mathbf{Q}$ und liegt daher in $S$. Multiplizieren wir das Residuum von links mit $\mathbf{Q}^\top$, erhalten wir

<!--en:0d321c20-->
Damit ist das Residuum orthogonal zu jeder Spalte $\mathbf{q}_i$ und somit zu jeder Linearkombination dieser Spalten. Für (ii) wählen wir ein beliebiges $\mathbf{s} \in S$ und schreiben $\mathbf{x} - \mathbf{s} = (\mathbf{x} - \mathbf{P}\mathbf{x}) + (\mathbf{P}\mathbf{x} - \mathbf{s})$. Da $\mathbf{P}\mathbf{x} - \mathbf{s}$ in $S$ liegt und $\mathbf{x} - \mathbf{P}\mathbf{x}$ orthogonal zu $S$ ist, liefert der Satz von Pythagoras :eqref:`eq_mdl-pythagoras`

<!--en:70de726d-->
wobei Gleichheit nur für $\mathbf{s} = \mathbf{P}\mathbf{x}$ gilt. Für (iii) erhalten wir $\mathbf{P}^2 = \mathbf{Q}(\mathbf{Q}^\top\mathbf{Q})\mathbf{Q}^\top =
\mathbf{Q}\mathbf{Q}^\top = \mathbf{P}$ und $\mathbf{P}^\top = (\mathbf{Q}^\top)^\top\mathbf{Q}^\top = \mathbf{P}$. $\blacksquare$

<!--en:bbf24b27-->
Umgekehrt kennzeichnet die Orthogonalitätsbedingung in (i) den nächstgelegenen Punkt eindeutig. Sei $\mathbf{s} \in S$ ein Punkt, dessen Residuum $\mathbf{x} - \mathbf{s}$ orthogonal zu $S$ ist. Dann liegt die Differenz $\mathbf{P}\mathbf{x} - \mathbf{s} = (\mathbf{x} - \mathbf{s}) - (\mathbf{x} - \mathbf{P}\mathbf{x})$ in $S$ und ist zugleich orthogonal zu $S$. Sie steht also senkrecht auf sich selbst, woraus $\mathbf{s} = \mathbf{P}\mathbf{x}$ folgt. Deshalb mussten die beiden Orthogonalitätsgleichungen für unsere Ausgleichsgerade den nächstgelegenen Punkt liefern.

<!--en:6b0a66d2-->
Beide Teile der Eigenschaft (iii) haben eine klare geometrische Bedeutung. Nach der Identität $\mathbf{P}^2 = \mathbf{P}$ ändert eine zweite Projektion nichts, weil das Ergebnis der ersten bereits in $S$ liegt. Die Symmetrie $\mathbf{P}^\top = \mathbf{P}$ macht die Projektion *orthogonal*. Eine Matrix, für die $\mathbf{P}^2 = \mathbf{P}$ gilt, die aber nicht symmetrisch ist, ist ein *schiefer* Projektor. Er bildet Vektoren entlang einer schrägen Richtung auf seinen Spaltenraum ab und verfehlt im Allgemeinen den nächstgelegenen Punkt (:exref:`ex_mdl-geometry-linear-algebraic-ops-projectors`). Im eindimensionalen Fall $k = 1$ genügt als einzige Spalte der Einheitsvektor $\mathbf{q} = \mathbf{w}/\|\mathbf{w}\|$. Damit vereinfacht sich $\mathbf{P} = \mathbf{q}\mathbf{q}^\top$ direkt zu unserer früheren Projektion auf eine Gerade :eqref:`eq_mdl-projection`.

<!--en:b116e9c4-->
Unsere Ausgleichsgerade ist ein Beispiel für die *Methode der kleinsten Quadrate*. Das zugehörige Optimierungsproblem bildet den Kern der linearen Regression. Sei $\mathbf{A}$ eine Datenmatrix, deren Zeilen Beispiele und deren Spalten Merkmale darstellen, und sei $\mathbf{y}$ der Vektor der Zielwerte. Die Methode der kleinsten Quadrate sucht den Koeffizientenvektor $\mathbf{x}$, der den quadratischen Fehler minimiert:

<!--en:2ea316d1-->
Durchläuft $\mathbf{x}$ alle Koeffizientenvektoren, so durchläuft $\mathbf{A}\mathbf{x}$ den Spann der Spalten von $\mathbf{A}$, den sogenannten *Spaltenraum* $\operatorname{col}(\mathbf{A})$. Bei der Methode der kleinsten Quadrate suchen wir also den Punkt in $\operatorname{col}(\mathbf{A})$, der $\mathbf{y}$ am nächsten liegt. Nach unserem Projektionssatz ist das der Punkt, dessen Residuum $\mathbf{y} - \mathbf{A}\hat{\mathbf{x}}$ orthogonal zu jeder Spalte von $\mathbf{A}$ ist. Schreiben wir diese Orthogonalitätsbedingung als $\mathbf{A}^\top(\mathbf{y} - \mathbf{A}\hat{\mathbf{x}}) = \mathbf{0}$, erhalten wir die *Normalgleichungen*,

<!--en:24a0a268-->
Das ist die Matrixform des $2 \times 2$-Systems, das wir von Hand gelöst haben. Für die Ausgleichsgerade liefert $\mathbf{A} = [\mathbf{1}\ \mathbf{t}]$ die Ausdrücke $\mathbf{A}^\top\mathbf{A} = \left(\begin{smallmatrix}3&3\\3&5\end{smallmatrix}\right)$ und $\mathbf{A}^\top\mathbf{y} = (6, 0)$. Sind die Spalten von $\mathbf{A}$ linear unabhängig, so ist $\mathbf{A}^\top\mathbf{A}$ invertierbar und die optimalen Koeffizienten $\hat{\mathbf{x}}$ sind eindeutig bestimmt. Selbst bei linear abhängigen Spalten bleibt der angepasste Vektor $\mathbf{A}\hat{\mathbf{x}}$ eindeutig, denn er ist weiterhin der nächstgelegene Punkt in $\operatorname{col}(\mathbf{A})$. Allerdings können dann mehrere Koeffizientenvektoren $\hat{\mathbf{x}}$ denselben Punkt liefern.

<!--en:e863250c-->
Im Code können wir die Anpassung $5 - 3t$ daher auf zwei Wegen berechnen. Wir können die Normalgleichungen direkt lösen oder mit $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$ projizieren. Für $\mathbf{Q}$ rufen wir `np.linalg.qr` auf. Die Funktion konstruiert eine Orthonormalbasis des Spaltenraums einer Matrix mit linear unabhängigen Spalten. Im nächsten Unterabschnitt konstruieren wir eine solche Basis von Hand.

<!--en:9b7ccddf-->
Beide Wege liefern dieselben Ergebnisse wie die Rechnung von Hand, nämlich die Koeffizienten $C = 5$ und $D = -3$, die projizierten Werte $(5, 2, -1)$ und das Residuum $(1, -2, 1)$. Die beiden zur Kontrolle berechneten Normen in der zweiten Zeile verschwinden bis auf Gleitkomma-Rundungsfehler. Das bestätigt $\mathbf{P}^2 = \mathbf{P}$ und $\mathbf{Q}^\top\mathbf{r} = \mathbf{0}$.

<!--en:f30c7218-->
Bei einem so kleinen, gut konditionierten Problem stimmen beide Methoden vollständig überein. Sind die Spalten von $\mathbf{A}$ jedoch nahezu linear abhängig, quadriert der Übergang zu $\mathbf{A}^\top\mathbf{A}$ die Konditionszahl des Problems. Sie beschreibt, wie empfindlich das Problem auf Rundungsfehler reagiert (siehe :numref:`subsec_mdl-condition-number`). Deshalb vermeiden praxistaugliche Verfahren für die Methode der kleinsten Quadrate die Normalgleichungen und arbeiten mit einer Orthonormalbasis des Spaltenraums.

<!--en:1e921c1c-->
### Gram-Schmidt-Verfahren und QR-Zerlegung

<!--en:96f1ccae-->
Wie gewinnen wir eine Orthonormalbasis? Durch wiederholte Projektion. Bei zwei linear unabhängigen Vektoren behalten wir die Richtung des ersten bei. Vom zweiten ziehen wir seinen Schatten entlang des ersten ab. Nach unserem Projektionssatz steht der verbleibende Vektor senkrecht auf dem ersten.

<!--en:792335e2-->
Führen wir die Schritte für $\mathbf{a}_1 = (3, 4)$ und $\mathbf{a}_2 = (1, 0)$ aus. Normieren wir $\mathbf{a}_1$, erhalten wir $\mathbf{q}_1 = \mathbf{a}_1/5 = (0.6, 0.8)$. Der Schatten von $\mathbf{a}_2$ auf $\mathbf{q}_1$ ist $(\mathbf{q}_1\cdot\mathbf{a}_2)\,\mathbf{q}_1 = 0.6\,(0.6, 0.8) = (0.36, 0.48)$. Ziehen wir ihn ab, bleibt $(1, 0) - (0.36, 0.48) = (0.64, -0.48)$, ein Vektor der Länge $0.8$, der tatsächlich senkrecht auf $\mathbf{q}_1$ steht ($0.384 - 0.384 = 0$). Normieren wir diesen Rest, erhalten wir $\mathbf{q}_2 = (0.8, -0.6)$. Lesen wir die Schritte rückwärts, ergibt sich für jeden ursprünglichen Vektor eine Darstellung in der neuen Orthonormalbasis:

<!--en:c1400eeb-->
In Matrixschreibweise lautet dies $\mathbf{A} = \mathbf{Q}\mathbf{R}$. Dabei gilt $\mathbf{A} = [\mathbf{a}_1\ \mathbf{a}_2]$ und $\mathbf{Q} = [\mathbf{q}_1\ \mathbf{q}_2]$, und $\mathbf{R} = \left(\begin{smallmatrix}5&0.6\\0&0.8\end{smallmatrix}\right)$ enthält die Skalarprodukte $\mathbf{q}_i\cdot\mathbf{a}_j$. Weshalb ist $\mathbf{R}$ eine obere Dreiecksmatrix? Nach Konstruktion hängt jedes $\mathbf{a}_j$ nur von $\mathbf{q}_1, \ldots, \mathbf{q}_j$ ab.

<!--en:bf691021-->
Übertragen wir diese Idee auf eine beliebige Liste linear unabhängiger Vektoren $\mathbf{a}_1, \ldots, \mathbf{a}_k$, erhalten wir das *Gram-Schmidt-Verfahren*. Wir gehen der Reihe nach vor. Von jedem $\mathbf{a}_j$ ziehen wir die Projektionen :eqref:`eq_mdl-projection` auf die bereits konstruierten Einheitsvektoren $\mathbf{q}_1, \ldots, \mathbf{q}_{j-1}$ ab und normieren den Rest. So erhalten wir $\mathbf{q}_j$. Das Ergebnis ist zugleich eine Orthonormalbasis für $\operatorname{span}(\mathbf{a}_1, \ldots, \mathbf{a}_k)$ und die *QR-Zerlegung* $\mathbf{A} = \mathbf{Q}\mathbf{R}$. Dabei hat $\mathbf{Q}$ orthonormale Spalten, und $\mathbf{R}$ ist eine obere Dreiecksmatrix mit positiven Diagonaleinträgen. Für diese Zerlegung ist die lineare Unabhängigkeit wesentlich. Liegt ein Vektor bereits im Spann seiner Vorgänger, bleibt der Nullvektor übrig, der sich nicht normieren lässt. Brauchen wir nur eine Orthonormalbasis für den Spann, überspringen wir jeden Vektor mit Rest null. Somit liefert jedes endliche Erzeugendensystem eine Orthonormalbasis, wie wir zuvor vorausgesetzt haben.

<!--en:9979f561-->
Numerische Bibliotheken berechnen die QR-Zerlegung mit einem anderen Algorithmus, der auf Spiegelungen beruht und sich in der Gleitkommaarithmetik deutlich besser verhält. Sein Ergebnis stimmt nur bis auf die Vorzeichen mit dem Gram-Schmidt-Verfahren überein. Wechseln wir das Vorzeichen einer Spalte von $\mathbf{Q}$ und der zugehörigen Zeile von $\mathbf{R}$, bleibt das Produkt $\mathbf{Q}\mathbf{R}$ gleich. Hat $\mathbf{A}$ linear unabhängige Spalten, sind diese Vorzeichenwechsel die einzige Mehrdeutigkeit. Vergleichen wir unsere zweistufige Rechnung nach Gram-Schmidt mit `np.linalg.qr` und prüfen wir, ob die Spalten bis auf das Vorzeichen übereinstimmen.

<!--en:78f737a8-->
Das Gram-Schmidt-Verfahren liefert $(0.6, 0.8)$ und $(0.8, -0.6)$. Dagegen gibt `np.linalg.qr` beide Spalten mit umgekehrtem Vorzeichen zurück. Die Diagonaleinträge lauten $-5$ und $-0.8$ in $\mathbf{R}$. Bis auf das Vorzeichen erhalten wir also dieselbe Basis und dieselben von Hand berechneten Skalarprodukte $5$, $0.6$ und $0.8$. In der Gleitkommaarithmetik kommt es auch auf die Reihenfolge der Subtraktionen an. Bei nahezu linear abhängigen Spalten geht in der obigen Formulierung, dem *klassischen* Gram-Schmidt-Verfahren, die Orthogonalität allmählich verloren (:exref:`ex_mdl-geometry-linear-algebraic-ops-gram-schmidt`).

<!--en:fb799edf-->
## Ähnlichkeit in hohen Dimensionen

<!--en:3e11721f-->
Bisher konnten wir jedes Ergebnis durch ein Bild in $\mathbb{R}^2$ oder $\mathbb{R}^3$ veranschaulichen. Nun wenden wir uns Räumen zu, die wir nicht mehr zeichnen können. Wie ähnlich sind sich zwei reale Bilder, und wie viel Ähnlichkeit ist bei Vektoren ohne inhaltlichen Zusammenhang allein durch Zufall zu erwarten? Wir laden den Datensatz Fashion-MNIST (:numref:`sec_fashion_mnist`) mit `d2l.FashionMNIST` als NumPy-Arrays. Jedes Bild formen wir zu einer Zeile mit $784$ Pixeln um und skalieren die Werte auf $[0, 1]$.

<!--en:9a18cdc3-->
Jedes dieser $70{,}000$ Bilder ist nun ein Punkt in $\mathbb{R}^{784}$.

<!--en:cdacda60-->
### Kosinusähnlichkeit

<!--en:baaf30d4-->
Der Quotient in :eqref:`eq_mdl-angle_formula` ist die *Kosinusähnlichkeit* (:numref:`sec_linear-algebra`),

<!--en:dd159002-->
Die Kosinusähnlichkeit vergleicht Richtungen und lässt Längen außer Acht. Ein Bild und eine auf $10\%$ seiner Helligkeit abgedunkelte Kopie haben einen großen euklidischen Abstand. Ihre Kosinusähnlichkeit beträgt dennoch $1$.

<!--en:f92b2d05-->
Dank dieser Skaleninvarianz ist die Kosinusähnlichkeit das Standardmaß für die Suche nach nächsten Nachbarn unter Einbettungen. Dasselbe gilt für kontrastive Lernziele wie bei SimCLR :cite:`Chen.Kornblith.Norouzi.ea.2020`. Diese richten die Einbettungen zweier Ansichten desselben Bildes in dieselbe Richtung aus. Die Attention-Schicht des ursprünglichen Transformers geht anders vor. Sie verwendet das *unnormierte* Skalarprodukt mit dem Skalierungsfaktor $1/\sqrt{d}$, den wir gleich herleiten. Hier bleiben die Vektorlängen also aussagekräftig. Viele spätere Modelle normieren zuerst die Abfragen und Schlüssel (:numref:`subsec_qk-norm`). Damit ist der Score wieder ein skalierter Kosinus.

<!--en:1aa17ceb-->
### Nahezu orthogonale Zufallsrichtungen

<!--en:301e8003-->
Angenommen, wir wählen zwei Bilder oder zwei Einbettungen ohne inhaltlichen Zusammenhang. Welche Kosinusähnlichkeit ist allein durch Zufall zu erwarten? Ohne eine Baseline können wir nicht beurteilen, ob ein beobachteter Kosinus auffällig groß ist. Die einfachste Baseline liefert eine zufällige Richtung, die auf der Einheitssphäre gleichverteilt ist und daher keinen Bereich des Raums bevorzugt.

<!--en:3b1a9ff9-->
**Satz (nahezu orthogonale Zufallsrichtungen).** *Sei $\mathbf{u} \in
\mathbb{R}^d$ ein fester Einheitsvektor. Wir ziehen $\mathbf{v}$ gleichverteilt von der Einheitssphäre in $\mathbb{R}^d$. Dann gilt für $\cos\theta = \mathbf{u}\cdot\mathbf{v}$*

<!--en:fe1ec783-->
**Beweis.** Die Gleichverteilung auf der Sphäre ist rotationsinvariant. Wir dürfen die Koordinaten daher so drehen, dass $\mathbf{u} = \mathbf{e}_1$ gilt. Dann ist $\cos\theta = v_1$. Auch ein Vorzeichenwechsel $\mathbf{v} \mapsto -\mathbf{v}$ lässt die Verteilung unverändert, also gilt $\mathbb{E}[v_1] = 0$. Aus Symmetriegründen hat jede Koordinate dasselbe zweite Moment $\mathbb{E}[v_i^2]$. Wegen $\sum_{i=1}^d v_i^2 = \|\mathbf{v}\|^2 = 1$ ergibt die Bildung des Erwartungswerts $d\,\mathbb{E}[v_1^2] = \mathbb{E}\bigl[\sum_{i=1}^d v_i^2\bigr] = 1$ und damit $\textrm{Var}(\cos\theta) = \mathbb{E}[v_1^2] = 1/d$. Die Schranke für die Überschreitungswahrscheinlichkeit folgt direkt aus der Chebyshev-Ungleichung $\Pr(|X - \mathbb{E}[X]| \ge \varepsilon) \le \textrm{Var}(X)/\varepsilon^2$ (:numref:`sec_prob`). $\blacksquare$

<!--en:caf824ff-->
Der typische Kosinus zwischen zwei zufälligen Richtungen hat also die Größenordnung $1/\sqrt{d}$. Für $d = 784$ beträgt die Standardabweichung $1/28 \approx 0.036$. Zwei zufällige Richtungen im Bildraum weichen damit fast immer nur um wenige Grad vom rechten Winkel ab. Nach der Chebyshev-Ungleichung ist die Wahrscheinlichkeit für $|\cos\theta| \ge 0.1$ höchstens $1/(784 \cdot 0.01) \approx 0.13$. Tatsächlich liegt sie unter $0.01$. Die Chebyshev-Ungleichung liefert eine grobe Schranke, weil sie nur die Varianz verwendet. Schärfere exponentielle Schranken lernen wir in :numref:`sec_mdl-concentration-generalization` kennen.

<!--en:9f0ccbbb-->
In niedrigen Dimensionen sieht die Verteilung ganz anders aus (:numref:`fig_mdl-la-cosine-highd`). In der Ebene ($d = 2$) ist der Winkel zwischen einer festen und einer zufälligen Richtung auf $[0, \pi]$ gleichverteilt. Eine Abweichung von höchstens $0.1$ Radiant vom rechten Winkel ist daher ebenso wahrscheinlich wie eine Abweichung von höchstens $0.1$ Radiant von der Kollinearität (jeweils etwa $6\%$). Die Werte des *Kosinus* häufen sich dagegen nahe $\pm 1$, weil $\cos\theta$ nahe $\theta = 0$ und $\theta = \pi$ flach verläuft. Die Wahrscheinlichkeit für $|\cos\theta| > 0.9$ beträgt etwa $0.29$, gegenüber nur $0.06$ für $|\cos\theta| < 0.1$. Für $d = 3$ ist der Kosinus auf $[-1, 1]$ gleichverteilt, ein klassisches Ergebnis, das schon Archimedes kannte. Ab $d = 4$ hat die Dichte ihr Maximum bei $0$. Bei $d = 10$ hat sie die Form einer Glocke der Breite $1/\sqrt{10} \approx 0.32$, bei $d = 1000$ die einer schmalen Spitze der Breite von ungefähr $0.03$.

<!--en:bacff0e4-->
![Histogramme des Kosinus zwischen unabhängigen zufälligen Einheitsvektoren für die Dimensionen $d = 2$, $10$ und $1000$. Bei $d = 2$ häufen sich die Kosinuswerte nahe $\pm 1$, obwohl der Winkel gleichverteilt ist. Bei $d = 10$ ergibt sich eine Glocke um $0$ und bei $d = 1000$ eine schmale Spitze der Breite $\approx 1/\sqrt{d}$, wie es der Satz vorhersagt.](../img/mdl-la-cosine-highd.svg)

<!--en:3d0eff3b-->
Prüfen wir diese isotrope Baseline für $d = 784$ durch eine Simulation. Da die Verteilung eines standardnormalen Zufallsvektors rotationsinvariant ist, liefert seine Normierung einen auf der Einheitssphäre gleichverteilten Vektor. In derselben Codezelle schätzen wir auch eine Varianz, die wir gleich für Attention brauchen, nämlich die Varianz von $\mathbf{q}\cdot\mathbf{k}$ für unabhängige standardnormale Zufallsvektoren in $64$ Dimensionen.

<!--en:d5b6876d-->
Die empirische Standardabweichung stimmt mit $1/\sqrt{784} \approx 0.036$ überein. Die Varianz des unnormierten Skalarprodukts liegt nahe bei $64$, der Dimension des Raums.

<!--en:a55c9d3d-->
Reale Daten sind allerdings selten isotrop. Verhalten sich zwei Fashion-MNIST-Bilder ohne inhaltlichen Zusammenhang wie nahezu orthogonale Vektoren? Ziehen wir $5{,}000$ zufällige Paare von Testbildern und messen wir ihre Kosinuswerte, zunächst für die rohen Pixelwerte und dann nach Abzug des mittleren Trainingsbildes.

<!--en:bdc7c618-->
Die Abweichung ist deutlich. Für die Rohbilder beträgt der mittlere Kosinus etwa $0.6$ bei einer Standardabweichung von ungefähr $0.18$. Das liegt weit entfernt von $0 \pm 0.036$. Der Grund liegt in den nichtnegativen Pixelintensitäten. Alle Bilder liegen im selben positiven Orthanten von $\mathbb{R}^{784}$, sodass negative Skalarprodukte unmöglich sind. Nach Abzug des mittleren Bildes liegen die Kosinuswerte um $0$. Ihre Standardabweichung von etwa $0.34$ beträgt aber immer noch fast das Zehnfache des isotropen Werts. Nach unserem Satz entspricht eine Streuung von $0.34$ zufälligen Richtungen in nur acht oder neun Dimensionen ($1/0.34^2$ liegt zwischen $8$ und $9$). Gemessen an ihren Kosinuswerten verhalten sich die zentrierten Bilder wie eine Punktwolke in weniger als zehn Dimensionen statt in $784$. Darin erkennen wir bereits die Struktur mit niedrigem Rang, die wir in :numref:`sec_mdl-svd-low-rank` untersuchen. In der Praxis gilt ein Kosinus also erst dann als groß, wenn er sich deutlich von der Streuung unzusammenhängender Paare aus *demselben* Datensatz abhebt. Durch das Training eines Einbettungsmodells heben sich zusammengehörige Paare von diesem Hintergrund ab.

<!--en:c078f0e2-->
### Größenordnung der Attention-Scores

<!--en:aa1c0d22-->
Bei der Skalarprodukt-Attention (:numref:`sec_attention-scoring-functions`) erhält eine Abfrage $\mathbf{q} \in \mathbb{R}^d$ für jeden Schlüssel $\mathbf{k}_1, \mathbf{k}_2, \ldots$ den Score $\mathbf{q}\cdot\mathbf{k}_j/\sqrt{d}$. Diese Scores gehen in eine Softmax-Funktion ein :cite:`Vaswani.Shazeer.Parmar.ea.2017`. Eine kurze Varianzrechnung erklärt, woher der Faktor $\sqrt{d}$ stammt. Angenommen, alle $2d$ Einträge von $\mathbf{q}$ und $\mathbf{k}$ sind voneinander unabhängig und haben den Mittelwert $0$ und die Varianz $1$. Jedes Produkt $q_i k_i$ hat dann den Mittelwert $0$ und die Varianz $\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1$. Verschiedene Terme sind unkorreliert, denn es gilt $\mathbb{E}[(q_i k_i)(q_j k_j)] = \mathbb{E}[q_i]\,\mathbb{E}[k_i]\,\mathbb{E}[q_j]\,\mathbb{E}[k_j] = 0$ für $i \neq j$. Die Varianzen der $d$ Terme addieren sich also:

<!--en:b15a3f27-->
Die unskalierten Skalarprodukte haben somit die Standardabweichung $\sqrt{d}$, die mit der Dimension des Attention-Kopfs wächst. Die Division durch $\sqrt{d}$ stellt die Varianz eins wieder her. Ohne diese Skalierung gerät die Softmax-Funktion in die Sättigung und liefert nahezu One-Hot-kodierte Gewichte. Für $d = 64$ haben die unskalierten Scores die Standardabweichung $8$. Zwei Scores mit der Differenz $8$ erhalten die Softmax-Gewichte $0.9997$ und $0.0003$. Nach Division durch $\sqrt{d} = 8$ sind die Gewichte mit $0.73$ und $0.27$ wieder ausgewogener. Die Geometrie liefert genau dieselbe Skalierung. Jeder Vektor hat ungefähr die Länge $\sqrt{d}$, und der Kosinus zwischen unzusammenhängenden Richtungen hat etwa die Größenordnung $1/\sqrt{d}$. Damit hat ihr Skalarprodukt typischerweise den Betrag $\sqrt{d}\cdot\sqrt{d}\cdot (1/\sqrt{d}) = \sqrt{d}$.

<!--en:4e03ff86-->
Trotz der beinahe rechten Winkel fallen irrelevante Schlüssel ins Gewicht, wenn es viele davon gibt. Jeder einzelne Kosinus zu einem irrelevanten Schlüssel hat zwar die Größenordnung $1/\sqrt{d}$, doch der *größte* Kosinus unter $n$ unabhängigen Schlüsseln wächst in führender Ordnung wie $\sqrt{2\ln n / d}$. In einem großen Retrieval-Index ist der stärkste Zufallstreffer daher um ein Mehrfaches größer als $1/\sqrt{d}$. Ein relevanter Schlüssel muss einen entsprechend höheren Wert übertreffen. Dafür sorgt das Training (in :exref:`ex_mdl-geometry-linear-algebraic-ops-random-cosines` berechnen wir konkrete Zahlen für diesen Effekt).

<!--en:f9f996c7-->
## Hyperebenen und Entscheidungsgrenzen

<!--en:09326fa0-->
Ein linearer Klassifikator stellt zwei geometrische Fragen zu einer Eingabe $\mathbf{x}$. Auf welcher Seite der Entscheidungsgrenze (engl. decision boundary) liegt sie, und wie weit ist sie von dieser Grenze entfernt? In diesem Abschnitt schreiben wir die Grenze als $\mathbf{w}\cdot\mathbf{x} + b = 0$.

<!--en:267a7007-->
### Hyperebenen und vorzeichenbehafteter Abstand

<!--en:ded15950-->
In der Ebene beschreibt diese Gleichung eine Gerade. Ein kurzes Beispiel zeigt die Bedeutung von $\mathbf{w}$ und $b$. Wählen wir $\mathbf{w} = (2, 1)$ und $b = -1$. Die Punkte, die $2x + y - 1 = 0$ erfüllen, bilden die Gerade $y = 1 - 2x$. Für jeden Punkt $\mathbf{x}$ auf dieser Geraden gilt $\mathbf{w}\cdot\mathbf{x} = -b$. Daher haben alle Schatten auf $\mathbf{w}$ dieselbe vorzeichenbehaftete Länge gemäß :eqref:`eq_mdl-signed-length`:

<!--en:de579ee2-->
Geometrisch steht die Gerade senkrecht auf $\mathbf{w}$ und schneidet die zugehörige Achse an der vorzeichenbehafteten Position $-b/\|\mathbf{w}\|$. Ihr Abstand zum Ursprung beträgt $|b|/\|\mathbf{w}\| = 1/\sqrt{5} \approx 0.45$.

<!--en:5b710d5e-->
In höheren Dimensionen gilt dasselbe. Eine *Hyperebene* in $\mathbb{R}^d$ ist eine Menge $\{\mathbf{x} : \mathbf{w}\cdot\mathbf{x} + b = 0\}$ mit $\mathbf{w} \neq \mathbf{0}$. Wir nennen $\mathbf{w}$ ihren *Normalenvektor* und $b$ ihren *Offset*. Sie entsteht durch Verschiebung des Unterraums $\mathbf{w}\cdot\mathbf{x} = 0$ der Dimension $d - 1$ (:numref:`subsec_mdl-rank`). In $\mathbb{R}^2$ ist sie eine Gerade, in $\mathbb{R}^3$ eine Ebene und im Bildraum eine $783$-dimensionale ebene Fläche. Die Hyperebene trennt $\mathbb{R}^d$ in zwei *Halbräume*, $\mathbf{w}\cdot\mathbf{x} + b > 0$ und $\mathbf{w}\cdot\mathbf{x} + b < 0$. Ändern wir $b$, verschiebt sich die Hyperebene parallel zu sich selbst entlang von $\mathbf{w}$, ohne sich zu neigen.

<!--en:b183d8d7-->
Wie weit ist ein beliebiger Punkt $\mathbf{x}$ von der Hyperebene entfernt? Wählen wir einen beliebigen Punkt $\mathbf{x}_0$ auf der Hyperebene, sodass $\mathbf{w}\cdot\mathbf{x}_0 = -b$ gilt. Den Abstand von $\mathbf{x}$ zur Hyperebene erhalten wir aus dem Schatten der Verschiebung $\mathbf{x} - \mathbf{x}_0$ auf den Normalenvektor $\mathbf{w}$. Seine vorzeichenbehaftete Länge liefert den *vorzeichenbehafteten Abstand*:

<!--en:6df46c96-->
Der Punkt $\mathbf{x}_0$ fällt vollständig weg. Der vorzeichenbehaftete Abstand hängt nur von $\mathbf{x}$, $\mathbf{w}$ und $b$ ab. Er ist positiv auf der Seite, in die $\mathbf{w}$ weist, negativ auf der Gegenseite und null auf der Hyperebene. In unserem Beispiel in der Ebene hat der Ursprung den vorzeichenbehafteten Abstand $-1/\sqrt{5}$. Der Punkt $(1, 1)$ hat den Score $\mathbf{w}\cdot\mathbf{x} + b = 2$ und liegt somit im vorzeichenbehafteten Abstand $+2/\sqrt{5} \approx 0.89$ auf der positiven Seite. Der ihm nächstgelegene Punkt auf der Geraden ist $(1, 1) - (2/5)(2, 1) = (0.2, 0.6)$. Der Punkt $(0, 1)$ hat den Score $0$ und liegt direkt auf der Geraden (:numref:`fig_mdl-la-hyperplane`).

<!--en:9318899b-->
![Die Hyperebene $\mathbf{w}\cdot\mathbf{x}+b=0$ mit Normalenvektor $\mathbf{w}$. Links ist die Hyperebene für $\mathbf{w}=(2, 1)$ und $b=-1$ die Gerade $2x+y=1$ im Abstand $1/\sqrt{5}$ vom Ursprung. Der Punkt $(1, 1)$ liegt im vorzeichenbehafteten Abstand $2/\sqrt{5}$ auf der Seite, in die $\mathbf{w}$ weist. Schattiert ist die Halbebene $\mathbf{w}\cdot\mathbf{x}+b>0$. Rechts definiert dieselbe Gleichung in drei Dimensionen eine Ebene.](../img/mdl-la-hyperplane.svg)

<!--en:07751cce-->
Bei einem Beispiel $(\mathbf{x}, y)$ mit Label $y \in \{-1, +1\}$ interessiert uns auch, ob es auf der *richtigen* Seite liegt. Multiplizieren wir den vorzeichenbehafteten Abstand mit dem Label, erhalten wir die *Margin* $y\,(\mathbf{w}\cdot\mathbf{x} + b)/\|\mathbf{w}\|$. Sie ist bei richtiger Klassifikation positiv und wächst, je tiefer der Punkt in seinem eigenen Halbraum liegt. Eine Skalierung $(\mathbf{w}, b) \to (c\mathbf{w}, cb)$ lässt für jedes $c > 0$ die Hyperebene und die Margin unverändert, multipliziert aber den unskalierten Score $\mathbf{w}\cdot\mathbf{x} + b$ mit $c$. Jedes Lernziel, das diesen Score maximiert, muss daher die Größe von $\mathbf{w}$ beschränken.

<!--en:18cd2177-->
### Klassifikation nach dem nächsten Klassenmittelwert

<!--en:894b69eb-->
Welche Hyperebene wählen wir, um zwei Klassen zu trennen? Die einfachste Regel kommt ganz ohne iteratives Training aus. Wir ordnen $\mathbf{x}$ der Klasse zu, deren Mittelwert näher liegt. Bemerkenswerterweise lässt sich diese Regel durch eine einzige Hyperebene beschreiben. Seien $\bar{\mathbf{x}}_0$ und $\bar{\mathbf{x}}_1$ die beiden Klassenmittelwerte. Multiplizieren wir beide Abstandsquadrate aus, fällt der quadratische Term $\|\mathbf{x}\|^2$ auf beiden Seiten weg:

<!--en:9317fa38-->
Mit der Identität $\|\mathbf{a}\|^2 - \|\mathbf{c}\|^2 = (\mathbf{a} - \mathbf{c})\cdot(\mathbf{a} + \mathbf{c})$ wird die letzte Ungleichung zu $\mathbf{w}\cdot\mathbf{x} + b > 0$ mit

<!--en:3f565447-->
Die Grenze für die Zuordnung zum nächsten Klassenmittelwert ist also die Mittelsenkrechte in beliebiger Dimension. Ihr Normalenvektor weist von einem Mittelwert zum anderen, und die Hyperebene verläuft durch den Mittelpunkt der Verbindungsstrecke. Wir müssen keinen Schwellenwert von Hand einstellen. Die Regel ist zudem invariant gegenüber einer einheitlichen Skalierung der Pixelwerte. Multiplizieren wir jeden Pixelwert mit $c > 0$, so gilt $\mathbf{w} \to c\mathbf{w}$ und $b \to c^2 b$. Jeder Score wird mit $c^2$ multipliziert, und alle Vorhersagen bleiben gleich. (In :exref:`ex_mdl-geometry-linear-algebraic-ops-nearest-centroid` erweitern wir diese Invarianz auf Drehungen und Verschiebungen und konstruieren eine pixelweise Skalierung, die Vorhersagen *tatsächlich ändert*.)

<!--en:bed05b0b-->
Bevor wir diesen Klassifikator an T-Shirts und Hosen testen, betrachten wir die beiden Klassenmittelwerte und berechnen den Winkel zwischen ihnen.

<!--en:caaa38a9-->
Beide Mittelwertbilder sind unscharf, die Kleidungsstücke aber sofort erkennbar. Der Winkel zwischen den Bildern beträgt nur etwa $33^\circ$ (Kosinus $0.84$). Sie sind also weit von der Orthogonalität entfernt, weil bei beiden Kleidungsstücken die Bildmitte hell ist. Der kleine Winkel zwischen den Mittelwerten könnte uns skeptisch stimmen. Die Klassifikation nach dem nächsten Klassenmittelwert erfordert jedoch keine orthogonalen Mittelwerte. Es kommt nur auf ihre *Differenz* an. Werten wir den Klassifikator auf allen $2{,}000$ Testbildern von T-Shirts und Hosen aus.

<!--en:25abcf77-->
Die Regel klassifiziert etwa $92\%$ der Testbilder richtig. Nur $169$ der $2{,}000$ Bilder landen auf der falschen Seite. Dazu haben wir lediglich den Mittelwert jeder Klasse berechnet und geprüft, auf welcher Seite einer einzigen Hyperebene jedes Testbild liegt.

<!--en:302a4735-->
Um die verbleibenden Fehler zu verstehen, reduzieren wir jedes Testbild auf eine Zahl, seinen vorzeichenbehafteten Abstand :eqref:`eq_mdl-signed-distance` von der Grenze. Für jede Klasse zeichnen wir ein Histogramm.

<!--en:ede3ffaa-->
Die beiden Klassen bilden deutlich getrennte Gipfel, deren Ausläufer über die Grenze reichen. Die T-Shirts bilden eine breite Kuppe um $-4$, die Hosen einen schmaleren Gipfel nahe $+4$ mit einem langen Ausläufer in Richtung null. Im Mittel liegt jede Klasse nahe beim vorzeichenbehafteten Abstand ihres Trainingsmittelwerts, also bei $\mp\|\mathbf{w}\|/2 \approx \mp 3.2$. Um das Bild in Zahlen zu fassen, berechnen wir für jede Klasse die Standardabweichung entlang von $\mathbf{w}$ und zählen die Beispiele auf der falschen Seite.

<!--en:f7018da8-->
Entlang von $\mathbf{w}$ haben die T-Shirts eine Standardabweichung von etwa $2.4$ und die Hosen von etwa $1.7$. Insgesamt sind $101$ T-Shirts und $68$ Hosen falsch klassifiziert. Da jedes Klassenzentrum weniger als zwei Standardabweichungen von der Grenze entfernt liegt, ist eine Fehlerrate von mehreren Prozent zu erwarten. Hier zeigt sich eine grundsätzliche Einschränkung der Regel, die die Grenze durch den Mittelpunkt legt. Sie berücksichtigt nur die beiden Klassenmittelwerte und erfasst weder die Größe noch die Form der Punktwolken. Was geschieht bei schwierigeren Paaren, die einander viel ähnlicher sehen, etwa T-Shirts und Hemden oder Pullovern und Mänteln?

<!--en:12d45569-->
Die Genauigkeit sinkt deutlich. Bei T-Shirts und Hemden fällt sie auf etwa $79\%$, bei Pullovern und Mänteln auf etwa $72\%$. Die übrigen Spalten zeigen den Grund. Jeder Trainingsmittelwert liegt im Abstand $\|\mathbf{w}\|/2$ von der Grenze. Für die Überlappung ist entscheidend, wie viele Standardabweichungen der jeweiligen Klasse entlang von $\mathbf{w}$ diesem Abstand entsprechen. Bei T-Shirts und Hosen sind es $1.4$ bis $1.9$ Standardabweichungen. Bei den beiden schwierigeren Paaren sind es nur noch $0.6$ bis $1.0$ Standardabweichungen. Eine Grenze durch den Mittelpunkt schneidet daher mitten durch beide Punktwolken. Abhilfe schafft eine Drehung von $\mathbf{w}$ weg von Richtungen, in denen die Klassen stark streuen. Das ist die Idee hinter der linearen Fisher-Diskriminante (:exref:`ex_mdl-geometry-linear-algebraic-ops-fisher-direction`). Die Softmax-Regression geht einen Schritt weiter und lernt sowohl die Neigung als auch den Offset direkt aus den Daten. Ein tiefes Netz transformiert die Eingaben selbst in eine Darstellung, in der eine einzige Hyperebene die Klassen fast vollständig trennt.

<!--en:9e773092-->
### Schichten als Stapel von Hyperebenen

<!--en:dba2c88b-->
Eine lineare Schicht eines neuronalen Netzes prüft die Lage zu vielen Hyperebenen parallel. Der Eintrag $i$ des Matrix-Vektor-Produkts $\mathbf{A}\mathbf{x}$ ist das Skalarprodukt der Zeile $i$ von $\mathbf{A}$ mit $\mathbf{x}$, also ein Score $\mathbf{a}_i\cdot\mathbf{x}$ bezüglich des Normalenvektors $\mathbf{a}_i$. Addieren wir einen Bias-Vektor, erhalten wir für jeden Eintrag den Score zu einer Hyperebene. Die Softmax-Schicht für die zehn Klassen von Fashion-MNIST aus :numref:`sec_softmax` speichert zum Beispiel zehn Normalenvektoren $\mathbf{w}_1, \ldots, \mathbf{w}_{10}$ und zehn Offsets. Sie sagt die Klasse mit dem höchsten Score vorher. Die Entscheidungsgrenze zwischen den Klassen $i$ und $j$ ist daher die Hyperebene $(\mathbf{w}_i - \mathbf{w}_j)\cdot\mathbf{x} + (b_i - b_j) = 0$, auf der beide Scores gleich sind.

<!--en:f52205a0-->
Betrachten wir dazu die in diesem Kapitel wiederkehrende Matrix $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ und die Eingabe $\mathbf{x} = (2, -1)$. Die erste Zeile liefert $1\cdot 2 + 2\cdot(-1) = 0$. Somit liegt $\mathbf{x}$ direkt auf der Ursprungshyperebene mit Normalenvektor $(1, 2)$. Die zweite Zeile liefert $-2 - 3 = -5$. Daher liegt $\mathbf{x}$ auf der negativen Seite der Hyperebene mit Normalenvektor $(-1, 3)$, im vorzeichenbehafteten Abstand $-5/\sqrt{10} \approx -1.58$. Fassen wir beide Ergebnisse zu einem Vektor zusammen, erhalten wir $\mathbf{A}\mathbf{x} = (0, -5)$.

<!--en:18123cee-->
Betrachten wir $\mathbf{A}\mathbf{x}$ zeilenweise, erkennen wir in einer linearen Schicht eine Reihe paralleler Tests an Hyperebenen. Zwei wesentliche Fragen bleiben offen. Welche Eingaben bildet die Transformation auf null ab, und wann lässt sie sich umkehren? In :numref:`sec_mdl-linear-maps` beantworten wir beide Fragen, indem wir dasselbe Produkt spaltenweise lesen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:95f216a3-->
Mit dem Skalarprodukt lassen sich Längen, Winkel und Projektionen aus der Ebene mit genau denselben Formeln auf beliebig viele Dimensionen übertragen. Die Projektion auf eine Gerade oder einen Unterraum liefert den eindeutig bestimmten nächstgelegenen Punkt. Sein Residuum steht senkrecht auf dem Zielraum. Allein aus dieser geometrischen Tatsache folgen die Cauchy-Schwarz-Ungleichung, die Winkel auch in hohen Dimensionen wohldefiniert macht, und die Normalgleichungen der Methode der kleinsten Quadrate. Das Gram-Schmidt-Verfahren verwandelt jedes endliche Erzeugendensystem in eine Orthonormalbasis und liefert so den Faktor $\mathbf{Q}$ der QR-Zerlegung. Zufällige Richtungen in $\mathbb{R}^d$ sind nahezu orthogonal, mit typischen Kosinuswerten der Größenordnung $1/\sqrt{d}$. Reale Bilddatensätze sind dagegen weit von der Isotropie entfernt. Ein beobachteter Kosinus ist daher nur im Vergleich mit unzusammenhängenden Paaren aus derselben Verteilung aussagekräftig. Die Klassifikation nach dem nächsten Klassenmittelwert definiert eine einzige Hyperebene, die die Verbindungsstrecke der beiden Klassenmittelwerte senkrecht halbiert. Bei T-Shirts und Hosen erreicht sie etwa $92\%$ Genauigkeit. Bei optisch ähnlichen Paaren sinkt die Genauigkeit auf $72\%$ bis $79\%$. Hier kommt es auf die Streuung der Klassen entlang des Normalenvektors der Grenze an.

<!--en:2294243a-->
## Übungen

<!--en:2f3f25a5-->
1. [code] **Kosinuswerte zwischen Zufallsvektoren.**
    1. Ziehen Sie $1{,}000$ Paare unabhängiger zufälliger Einheitsvektoren in $d = 10{,}000$ Dimensionen, indem Sie standardnormale Zufallsvektoren normieren. Berechnen Sie den Kosinus für jedes Paar. Vergleichen Sie die empirische Standardabweichung mit der Vorhersage $1/\sqrt{d} = 0.01$ aus dem Satz über nahezu orthogonale Zufallsrichtungen. Geben Sie den größten beobachteten Betrag $|\cos\theta|$ an.
    1. Ein Retrieval-System vergleicht eine Abfrage mit $n$ Schlüsseln und gibt den Schlüssel mit dem größten Kosinus zurück. Modellieren Sie die Abfrage und die Schlüssel als unabhängige zufällige Einheitsvektoren in $d = 256$ Dimensionen. Berechnen Sie das Maximum der $n$ Kosinuswerte für $n = 10$, $10^3$ und $10^5$. Vergleichen Sie es mit $\sqrt{2\ln n / d}$, der Schätzung führender Ordnung für das Maximum von $n$ unabhängigen normalverteilten Zufallswerten mit Varianz $1/d$. Welchen Kosinus erreicht ein irrelevanter Schlüssel als bester Treffer unter $10^5$ Schlüsseln? Wie sollte der Schwellenwert, ab dem zwei Einbettungen als zusammengehörig gelten, mit $n$ und $d$ skalieren?
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-random-cosines`

<!--en:ea01aada-->
1. **Nächste Klassenmittelwerte und Invarianz.** Die Regel des nächsten Klassenmittelwerts ordnet $\mathbf{x}$ der Klasse $1$ zu, wenn $\mathbf{w}\cdot\mathbf{x} + b > 0$ gilt. Dabei sind $\mathbf{w} = \bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_0$ und $b = -\mathbf{w}\cdot\tfrac12(\bar{\mathbf{x}}_0 + \bar{\mathbf{x}}_1)$.
    1. Angenommen, wir unterziehen jedes Trainings- und Testbild derselben Transformation $\mathbf{x} \mapsto c\mathbf{Q}\mathbf{x} + \mathbf{z}$. Dabei gilt $c > 0$, die Matrix $\mathbf{Q}$ ist quadratisch mit $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$, und $\mathbf{z}$ ist ein fester Vektor. Anschließend berechnen wir die Klassenmittelwerte und $b$ neu. Zeigen Sie, dass alle Vorhersagen gleich bleiben. Stellen Sie den Zusammenhang mit der im Text besprochenen Skalierung $\mathbf{w} \to c\mathbf{w}$, $b \to c^2 b$ her.
    1. Die Standardisierung jedes Pixels auf Varianz eins ersetzt $\mathbf{x}$ durch $\mathbf{D}\mathbf{x}$ mit einer positiven Diagonalmatrix $\mathbf{D}$. Konstruieren Sie zwei Klassenmittelwerte und einen Testpunkt in $\mathbb{R}^2$, dessen vorhergesagte Klasse sich unter einer solchen Transformation mit $\mathbf{D}$ ändert.
    1. Betrachten Sie die Regel des nächsten Klassenmittelwerts für $q$ Klassen. Zeigen Sie, dass das Gebiet der Klasse $i$ der Schnitt der $q - 1$ Halbräume $(\bar{\mathbf{x}}_i - \bar{\mathbf{x}}_j)\cdot\mathbf{x} + b_{ij} > 0$ für $j \neq i$ ist, und bestimmen Sie $b_{ij}$. Vergleichen Sie das Ergebnis mit den paarweisen Grenzen $(\mathbf{w}_i - \mathbf{w}_j)\cdot\mathbf{x} + (b_i - b_j) = 0$ einer Softmax-Schicht. Für welche Gewichte und Bias-Terme liefert die Softmax-Schicht denselben Klassifikator wie die Regel des nächsten Klassenmittelwerts?
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-nearest-centroid`

<!--en:21bae1e5-->
1. **Orthogonale und schiefe Projektoren.**
    1. Die Matrix $\mathbf{P}$ erfülle $\mathbf{P}^2 = \mathbf{P}$ und $\mathbf{P}^\top = \mathbf{P}$. Zeigen Sie für jedes $\mathbf{x}$, dass $\mathbf{P}\mathbf{x}$ im Spaltenraum von $\mathbf{P}$ liegt und $\mathbf{x} - \mathbf{P}\mathbf{x}$ orthogonal zu diesem Raum ist. Folgern Sie mit dem Satz von Pythagoras, dass $\mathbf{P}\mathbf{x}$ der nächstgelegene Punkt im Spaltenraum ist, ohne eine Orthonormalbasis zu verwenden.
    1. Die Matrix $\mathbf{P}_{\mathrm{o}} = \left(\begin{smallmatrix}1&1\\0&0\end{smallmatrix}\right)$ erfüllt $\mathbf{P}_{\mathrm{o}}^2 = \mathbf{P}_{\mathrm{o}}$, ist aber nicht symmetrisch. Berechnen Sie für $\mathbf{x} = (1, 1)$ sowohl $\mathbf{P}_{\mathrm{o}}\mathbf{x}$ als auch die orthogonale Projektion von $\mathbf{x}$ auf denselben Spaltenraum (die erste Koordinatenachse). Bestimmen Sie jeweils den Abstand zu $\mathbf{x}$. Entlang welcher Richtung projiziert $\mathbf{P}_{\mathrm{o}}$ die Punkte?
    1. Zeigen Sie, dass jede Matrix $\mathbf{P}$ mit $\mathbf{P}^2 = \mathbf{P}$ jeden Vektor $\mathbf{x}$ als $\mathbf{P}\mathbf{x} + (\mathbf{x} - \mathbf{P}\mathbf{x})$ zerlegt. Dabei liegt der erste Term im Spaltenraum von $\mathbf{P}$ und der zweite in ihrem Nullraum $\{\mathbf{z} : \mathbf{P}\mathbf{z} = \mathbf{0}\}$. Zeigen Sie, dass diese Unterräume genau dann orthogonal zueinander sind, wenn $\mathbf{P}^\top = \mathbf{P}$ gilt.
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-projectors`

<!--en:eef7d563-->
1. [code] **Die Grenze neigen.** ● Der Normalenvektor $\boldsymbol{\Delta} = \bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_0$ ist die Differenz der Mittelwerte. Er berücksichtigt nicht, wie die Bilder jeder Klasse um ihren Mittelwert streuen. Sei $\mathbf{C}$ der Mittelwert der beiden klasseninternen Kovarianzmatrizen auf dem Trainingsdatensatz. Die lineare Fisher-Diskriminante :cite:`Fisher.1936` wählt den Normalenvektor $\mathbf{w}$ so, dass er den folgenden Quotienten maximiert:

   $$
   J(\mathbf{w}) = \frac{(\mathbf{w}\cdot\boldsymbol{\Delta})^2}{\mathbf{w}^\top\mathbf{C}\mathbf{w}},
   $$

   Im Zähler steht das Abstandsquadrat der projizierten Klassenmittelwerte, im Nenner die klasseninterne Varianz der Projektionen. Ist $\mathbf{C}$ positiv definit ($\mathbf{v}^\top\mathbf{C}\mathbf{v} > 0$ für alle $\mathbf{v} \neq \mathbf{0}$), so existiert ihre Inverse $\mathbf{C}^{-1}$ und für den Maximierer gilt $\mathbf{w} \propto \mathbf{C}^{-1}\boldsymbol{\Delta}$. (Diese Aufgabe verwendet die inverse Matrix aus :numref:`subsec_mdl-inverses`, die positive Definitheit aus :numref:`subsec_mdl-psd` und die Konditionszahl aus :numref:`subsec_mdl-condition-number`.)
    1. Zeigen Sie, dass $J(c\mathbf{w}) = J(\mathbf{w})$ für jedes $c \neq 0$ gilt. Zeigen Sie für $\mathbf{C} = s^2\mathbf{I}$ mithilfe der Cauchy-Schwarz-Ungleichung, dass $\boldsymbol{\Delta}$ die Funktion $J$ maximiert.
    1. Berechnen Sie mit den Arrays `X0` und `X1` für T-Shirts und Hosen aus dem Text die Matrix $\mathbf{C}$. Sei $\bar s^2 = \operatorname{tr}(\mathbf{C})/784$ die mittlere Pixelvarianz. Für diese Bilder hat $\mathbf{C}$ eine Konditionszahl von etwa $10^9$, weil einige Randpixel nahezu konstant sind. Bilden Sie für $\gamma \in \{10^{-4}, 10^{-2}, 1, 10^{2}, 10^{4}\}$ den regularisierten Normalenvektor $\mathbf{w}_\gamma = (\mathbf{C} + \gamma\bar s^2\mathbf{I})^{-1}\boldsymbol{\Delta}$. Legen Sie die Grenze durch den Mittelpunkt zwischen den beiden Mittelwerten und bestimmen Sie die Trainings- und die Testgenauigkeit. Legen Sie anschließend $1{,}000$ Trainingsbilder je Klasse für die Validierung zurück. Berechnen Sie $\mathbf{C}$ und die Klassenmittelwerte aus den übrigen Bildern neu. Wählen Sie $\gamma$ anhand der Validierungsgenauigkeit auf den zurückgelegten Bildern. Zeichnen Sie das Histogramm der vorzeichenbehafteten Abstände mit dem so bestimmten $\mathbf{w}_\gamma$ erneut.
    1. Warum richtet sich $\mathbf{w}_\gamma$ an $\boldsymbol{\Delta}$ aus, wenn $\gamma \to \infty$ gilt? Warum skalieren wir den Regularisierungsterm mit $\bar s^2$, anstatt ein festes Vielfaches von $\mathbf{I}$ zu addieren?
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-fisher-direction`

<!--en:a770551a-->
1. [code] **Gram-Schmidt-Verfahren in der Gleitkommaarithmetik.** ● Die Schritte des Gram-Schmidt-Verfahrens lassen sich in zwei algebraisch gleichwertigen Reihenfolgen ausführen. Das klassische Gram-Schmidt-Verfahren (CGS) berechnet $\mathbf{q}_j$ durch Normierung von $\mathbf{a}_j - \sum_{i<j}(\mathbf{q}_i\cdot\mathbf{a}_j)\,\mathbf{q}_i$ wie in unserem Beispiel mit zwei Vektoren. Das modifizierte Gram-Schmidt-Verfahren (MGS) beginnt mit $\mathbf{v} = \mathbf{a}_j$, führt die Aktualisierung $\mathbf{v} \leftarrow \mathbf{v} - (\mathbf{q}_i\cdot\mathbf{v})\,\mathbf{q}_i$ nacheinander für $i = 1, \ldots, j-1$ aus und normiert anschließend $\mathbf{v}$.
    1. Zeigen Sie, dass CGS und MGS bei exakter Rechnung dieselben Vektoren liefern.
    1. Implementieren Sie beide Algorithmen und wenden Sie sie auf die Hilbert-Matrizen $\mathbf{H}_n$ mit Einträgen $1/(i + j - 1)$ für $n = 4, 6, 8, 10, 12$ an. Ihre Spalten sind nahezu linear abhängig. Tragen Sie die Frobeniusnorm $\|\mathbf{Q}^\top\mathbf{Q} - \mathbf{I}\|_F$ gegen $n$ auf, jeweils für CGS, MGS und `np.linalg.qr`. Ab welchem $n$ wirkt $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$ beim jeweiligen Verfahren nicht mehr als Projektor, gemessen an $\|\mathbf{P}^2 - \mathbf{P}\|_F$?
    1. Wenden Sie den Subtraktionsschritt von CGS für jede Spalte zweimal an. Beginnen Sie den zweiten Durchlauf mit dem Vektor, der nach dem ersten übrig bleibt. Wie nahe kommen die resultierenden Spalten der Orthonormalität im Vergleich zu einem einzigen CGS-Durchlauf und zu MGS? Erklären Sie die Verbesserung, indem Sie die Größe der Komponenten entlang von $\mathbf{q}_1, \ldots, \mathbf{q}_{j-1}$ zu Beginn des ersten und des zweiten Durchlaufs vergleichen.
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-gram-schmidt`

<!--en:582ab067-->
1. [extended] **Packungen nahezu orthogonaler Richtungen.** ●● In $\mathbb{R}^d$ gibt es höchstens $d$ paarweise orthogonale Richtungen. Es finden aber weit mehr Richtungen Platz, deren paarweise Kosinuswerte klein bleiben. :citet:`Elhage.Hume.Olsson.ea.2022` erklären mit dieser Geometrie, wie eine Schicht der Breite $d$ mehr als $d$ Merkmale in *Überlagerung* als nahezu orthogonale Richtungen darstellen kann. Ihre Merkmale sind *dünn besetzt*. Jedes wird nur bei einem kleinen Teil der Eingaben aktiv, sodass nur wenige gleichzeitig ausgelesen werden müssen. Seien $\mathbf{u}_1, \ldots, \mathbf{u}_n$ Einheitsvektoren in $\mathbb{R}^d$, die wir als Zeilen der Matrix $\mathbf{U} \in \mathbb{R}^{n \times d}$ anordnen. Sei $\mu = \max_{i \neq j}|\mathbf{u}_i\cdot\mathbf{u}_j|$ ein Maß für ihre stärkste paarweise Interferenz.
    1. Schreiben Sie $\sum_{i,j}(\mathbf{u}_i\cdot\mathbf{u}_j)^2$ in Koordinaten aus und zeigen Sie, dass der Ausdruck gleich $\sum_{k,l}\bigl(\sum_i u_{ik}u_{il}\bigr)^2$ ist, der Summe der Eintragsquadrate der $d \times d$-Matrix $\mathbf{U}^\top\mathbf{U}$. Schätzen Sie diese Summe mithilfe ihrer $d$ Diagonalterme und der Cauchy-Schwarz-Ungleichung in der Form $\sum_{k=1}^d a_k^2 \ge \bigl(\sum_k a_k\bigr)^2/d$ nach unten ab. Beachten Sie dabei, dass $\sum_k (\mathbf{U}^\top\mathbf{U})_{kk} = n$ gilt. Schätzen Sie die ursprüngliche Summe nach oben durch $n + n(n-1)\mu^2$ ab, um die *Welch-Schranke* :cite:`Welch.1974` herzuleiten:

       $$
       \mu^2 \ge \frac{n - d}{d\,(n - 1)}.
       $$

       Werten Sie diese Schranke für $d = 32$ mit $n = 64$ und $n = 512$ aus.
    1. Die boolesche Ungleichung für alle Paare zeigt, dass für unabhängige zufällige Richtungen $\mu \le \varepsilon$ gilt, wenn $n$ die Größenordnung $e^{d\varepsilon^2/4}$ hat (:numref:`sec_mdl-concentration-generalization`). Berechnen Sie für $d = 32$ und $n \in \{64, 128, 256, 512\}$ den Wert $\mu$ für zufällige Einheitsvektoren und vergleichen Sie ihn mit der Welch-Schranke.
    1. Verringern Sie $\mu$ direkt durch Optimierung. Minimieren Sie $\sum_{i \neq j}(\mathbf{u}_i\cdot\mathbf{u}_j)^{p}$ für einen geraden Exponenten wie $p = 10$ durch Gradientenschritte bezüglich $\mathbf{U}$. Normieren Sie jede Zeile nach jedem Schritt erneut. Tragen Sie für $d \in \{16, 32, 48, 64\}$ und $n \in \{d, 2d, 4d, 8d, 16d\}$ die Werte von $\mu$ gegen $n$ auf, und zwar für zufällige Vektoren, optimierte Vektoren und die Welch-Schranke.
    1. Schätzen Sie anhand dieser Kurven für jedes $d$ das größte $n$, für das die optimierten Richtungen $\mu \le 0.2$ erfüllen. Wie wächst diese Kapazität mit $d$ im Vergleich zur Rate $e^{d\varepsilon^2/4}$ für zufällige Richtungen? Wie würden sich Ihre Schlüsse ändern, wenn manche Merkmale wichtiger wären als andere oder wenn viele Merkmale gleichzeitig aktiv wären?
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-packing-directions`

<!--en:95b92a21-->
[Diskussionen](https://d2l.discourse.group/t/410)

<!--en:c42dea04-->
[Diskussionen](https://d2l.discourse.group/t/1084)

<!--en:b4721c01-->
[Diskussionen](https://d2l.discourse.group/t/1085)

<!--en:b4721c01-2-->
[Diskussionen](https://d2l.discourse.group/t/1085)

<!--en:15467fc6-->
Vektoren, Winkel und Projektionen<br>**Skalarprodukte · orthogonale Projektionen als Schatten · Cauchy-Schwarz-Ungleichung · Unterräume und QR-Zerlegung · Kosinuswerte in hohen Dimensionen · Hyperebenen**

<!--en:a9c1ec6a-->
::: {.slide title="Ein Skalarprodukt klassifiziert 92% der Testbilder richtig"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:47056570-->
- **Drei geometrische Operationen im maschinellen Lernen:** Modelle *vergleichen* Darstellungen (Ähnlichkeit), *passen* Vorhersagen an Zielwerte *an* (Projektion nach der Methode der kleinsten Quadrate) und *trennen* Klassen (Hyperebenen als Grenzen).

<!--en:fa0bc20e-->
- **Ein konkreter Test in $\mathbb{R}^{784}$:** Jedes $28 \times 28$ große Fashion-MNIST-Bild ist ein Punkt in einem $784$-dimensionalen Raum, den wir nicht zeichnen können.

<!--en:415ac22a-->
- **Der einfachste Klassifikator:** Wir mitteln alle T-Shirt-Bilder und alle Hosenbilder im Trainingsdatensatz getrennt. Jedes Testbild ordnen wir dem näheren Klassenmittelwert zu. Dafür genügt ein Skalarprodukt, die **Testgenauigkeit beträgt $92\%$**.

<!--en:cd796efe-->
::: {.slide title="Vektoren als Punkte und als Verschiebungen"}

<!--en:bbf06bea-->
[Vektorgeometrie]{.kicker}

<!--en:4b544592-->
- **Als Punkt:** Ein Vektor $\mathbf{v} = (3, 2)$ gibt eine feste Position relativ zum Ursprung $\mathbf{0}$ an. Ein Datensatz aus $n$ Beispielen wird so zu einer Wolke aus $n$ Punkten in $\mathbb{R}^d$.

<!--en:c0b51f40-->
- **Als Richtung:** Derselbe Vektor beschreibt eine Verschiebung ($3$ Schritte nach rechts, $2$ nach oben), die überall beginnen kann. Alle gleichgerichteten Pfeile gleicher Länge stellen denselben Vektor dar.

<!--en:ec195854-->
- **Das Skalarprodukt als Grundlage:** Jede geometrische Größe in $\mathbb{R}^d$ ergibt sich aus einer symmetrischen Operation ($|\mathbf{u}| = \sqrt{\mathbf{u}\cdot\mathbf{u}}$):
  $$\mathbf{u}^\top\mathbf{v} = \mathbf{u}\cdot\mathbf{v} = \sum_{i=1}^d u_i v_i$$

<!--en:6a5e329f-->
::: {.slide title="Addition von Pfeilen und relative Verschiebung"}

<!--en:bbf06bea-2-->
[Vektorgeometrie]{.kicker}

<!--en:721efa3d-->
- **Vektoraddition ($\mathbf{u} + \mathbf{v}$):** Um zwei Verschiebungen zu addieren, folgen wir erst dem Pfeil $\mathbf{u}$ und von seiner Spitze aus dem Pfeil $\mathbf{v}$.

<!--en:8ee882f2-->
- **Vektorsubtraktion ($\mathbf{u} - \mathbf{v}$):** Wegen $\mathbf{u} = \mathbf{v} + (\mathbf{u} - \mathbf{v})$ ist die Differenz $\mathbf{u} - \mathbf{v}$ genau der Verschiebungspfeil vom Punkt $\mathbf{v}$ zum Punkt $\mathbf{u}$.

<!--en:be628cf2-->
- **Euklidischer Abstand:** Der Abstand zwischen zwei Datenpunkten $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ ist die Länge ihres Differenzvektors, also $\|\mathbf{u} - \mathbf{v}\| = \sqrt{(\mathbf{u} - \mathbf{v})\cdot(\mathbf{u} - \mathbf{v})}$.

<!--en:99d89c25-->
::: {.slide title="Projektion auf eine Gerade und nächstgelegenes Vielfaches"}

<!--en:0b4c6a7e-->
[Orthogonale Projektion]{.kicker}

<!--en:51eb74eb-->
- **Eindimensionales Ausgleichsproblem:** Gegeben sind $\mathbf{v} = (1, 3)$ und eine Richtung $\mathbf{w} = (2, 1)$. Welches skalare Vielfache $t\mathbf{w}$ liegt $\mathbf{v}$ am nächsten?

<!--en:7481d346-->
- **Quadratische Fehlerkurve:** Ausmultiplizieren des quadratischen Fehlers ergibt eine konvexe Parabel in $t$:
  $$q(t) = \|\mathbf{v} - t\mathbf{w}\|^2 = \|\mathbf{w}\|^2 t^2 - 2(\mathbf{v}\cdot\mathbf{w})t + \|\mathbf{v}\|^2$$

<!--en:eb149cb2-->
- **Eindeutiges Minimum ($q'(t^\star) = 0$):** Aus $2\|\mathbf{w}\|^2 t - 2(\mathbf{v}\cdot\mathbf{w}) = 0$ folgt $t^\star = (\mathbf{v}\cdot\mathbf{w})/(\mathbf{w}\cdot\mathbf{w})$. Für unser Paar gilt $t^\star = 5/5 = 1$, mit Schatten $(2, 1)$ und Residuum $\mathbf{r} = (-1, 2)$.

<!--en:aa1da602-->
::: {.slide title="Orthogonalität und der Satz von Pythagoras in jeder Dimension"}

<!--en:0b4c6a7e-2-->
[Orthogonale Projektion]{.kicker}

<!--en:32adf452-->
- **Formel für die orthogonale Projektion:** Für $\mathbf{w} \neq \mathbf{0}$ gilt: Das Vielfache dieses Vektors, das $\mathbf{v}$ am nächsten liegt, ist $\operatorname{proj}_{\mathbf{w}}\mathbf{v} = \frac{\mathbf{v}\cdot\mathbf{w}}{\mathbf{w}\cdot\mathbf{w}}\mathbf{w}$. Es bleibt das Residuum $\mathbf{r} = \mathbf{v} - \operatorname{proj}_{\mathbf{w}}\mathbf{v} \perp \mathbf{w}$.

<!--en:75c44c55-->
- **Satz von Pythagoras ($\mathbf{a}\cdot\mathbf{b} = 0$):** Die gemischten Terme verschwinden, also gilt $\|\mathbf{v}\|^2 = \|\operatorname{proj}_{\mathbf{w}}\mathbf{v}\|^2 + \|\mathbf{r}\|^2$ ($10 = 5 + 5$).

<!--en:dd179828-->
- **Vorzeichenbehaftete Schattenlänge:** Multiplikation von $t^\star$ mit $\|\mathbf{w}\|$ ergibt $(\mathbf{v}\cdot\mathbf{w})/\|\mathbf{w}\|$.

<!--en:e74383dd-->
::: {.slide title="Die Cauchy-Schwarz-Ungleichung am rechtwinkligen Dreieck"}

<!--en:3dda3eba-->
[Winkel in hohen Dimensionen]{.kicker}

<!--en:735768a9-->
- **Cauchy-Schwarz-Ungleichung:** Für alle $\mathbf{v}, \mathbf{w} \in \mathbb{R}^d$ gilt $|\mathbf{v}\cdot\mathbf{w}| \le \|\mathbf{v}\|\,\|\mathbf{w}\|$. Gleichheit gilt genau dann, wenn $\mathbf{v}$ und $\mathbf{w}$ kollinear sind.

<!--en:c9875454-->
- **Beweis in einer Zeile:** Wegen $\|\mathbf{r}\|^2 \ge 0$ dürfen wir das Längenquadrat des Residuums im Satz von Pythagoras weglassen und erhalten $\|\mathbf{v}\|^2 \ge \|\operatorname{proj}_{\mathbf{w}}\mathbf{v}\|^2 = (\mathbf{v}\cdot\mathbf{w})^2/\|\mathbf{w}\|^2$.

<!--en:56b84b5e-->
- **Dreiecksungleichung:** Die Abschätzung $2\mathbf{v}\cdot\mathbf{w} \le 2\|\mathbf{v}\|\|\mathbf{w}\|$ liefert $\|\mathbf{v} + \mathbf{w}\| \le \|\mathbf{v}\| + \|\mathbf{w}\|$.

<!--en:0ad26fec-->
::: {.slide title="Winkel definieren und Rundungsfehler abfangen"}

<!--en:3dda3eba-2-->
[Winkel in hohen Dimensionen]{.kicker}

<!--en:b9cf4381-->
- **Winkel in hohen Dimensionen:** Nach der Cauchy-Schwarz-Ungleichung liegt der Kosinusquotient in $[-1, 1]$. Damit ist $\theta = \arccos\!\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right) \in [0, \pi]$ definiert.

<!--en:468639e4-->
- **Schutz vor Rundungsfehlern:** Bei kollinearen Vektoren kann Rundung $1 + 2.2\times 10^{-16}$ ergeben. Den Quotienten stets mit `np.clip(cos, -1, 1)` begrenzen.

<!--en:964e9d13-->
::: {.slide title="Spann, lineare Unabhängigkeit, Basis und Dimension"}

<!--en:7ad73785-->
[Unterräume]{.kicker}

<!--en:9fdf5347-->
- **Spann und Unterräume:** Die Menge aller Linearkombinationen $\operatorname{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{\sum_{i=1}^k a_i \mathbf{v}_i\}$ ist unter Addition und Skalierung abgeschlossen. Sie bildet einen *Unterraum* durch den Ursprung $\mathbf{0}$.

<!--en:0d82d0cf-->
- **Lineare Unabhängigkeit:** Die Vektoren $\mathbf{v}_1, \ldots, \mathbf{v}_k$ sind linear unabhängig, wenn keiner überflüssig ist. Aus $\sum_i a_i \mathbf{v}_i = \mathbf{0}$ folgt dann $a_1 = \cdots = a_k = 0$.

<!--en:72814f4c-->
- **Basis und eindeutige Koordinaten:** Eine Basis eines Unterraums $S$ ist ein linear unabhängiges Erzeugendensystem. Die Anzahl seiner Vektoren ist die *Dimension* $\dim S$. Jedes $\mathbf{x} \in S$ hat eindeutige Koordinaten $(a_1, \ldots, a_k)$.

<!--en:6a09de9d-->
::: {.slide title="Eine Ausgleichsgerade durch Projektion auf eine Ebene bestimmen"}

<!--en:9d5ad7d6-->
[Kleinste Quadrate]{.kicker}

<!--en:acc27e82-->
- **Unterraum der möglichen Anpassungen in $\mathbb{R}^3$:** Passen wir $C + Dt$ an die Werte $\mathbf{y} = (6, 0, 0)$ zu den Zeitpunkten $\mathbf{t} = (0, 1, 2)$ an, suchen wir den nächstgelegenen Vektor $C\mathbf{1} + D\mathbf{t}$ in der zweidimensionalen Ebene $S = \operatorname{span}(\mathbf{1}, \mathbf{t})$.

<!--en:a3266051-->
- **Orthogonalitätsgleichungen:** Das Residuum $\mathbf{r} = \mathbf{y} - C\mathbf{1} - D\mathbf{t}$ muss auf der gesamten Ebene $S$ senkrecht stehen, also auf beiden Basisvektoren. Daraus folgen $\mathbf{1}\cdot\mathbf{r} = 6 - 3C - 3D = 0$ und $\mathbf{t}\cdot\mathbf{r} = -3C - 5D = 0$:
  $$3C + 3D = 6 \implies C = 5, \; D = -3$$

<!--en:331334c2-->
- **Lösung und Fehlerzerlegung:** Die beste Gerade $5 - 3t$ liefert den Vorhersagevektor $(5, 2, -1)$ und das Residuum $\mathbf{r} = (1, -2, 1)$ mit $\|\mathbf{y}\|^2 = 36 = 30 + 6$.

<!--en:a60abbaa-->
::: {.slide title="Die Projektionsmatrix P = QQᵀ für einen Unterraum"}

<!--en:9d5ad7d6-2-->
[Kleinste Quadrate]{.kicker}

<!--en:37ced6b3-->
- **Matrix einer Orthonormalbasis $\mathbf{Q} \in \mathbb{R}^{n \times k}$:** Die Spalten $\mathbf{q}_1, \ldots, \mathbf{q}_k$ spannen $S$ auf und erfüllen $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}_k$.

<!--en:642e838f-->
- **Eindeutiger nächstgelegener Punkt:** $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$ projiziert auf $S$ mit $\mathbf{x} - \mathbf{P}\mathbf{x} \perp S$. Das Minimum folgt aus $\|\mathbf{x} - \mathbf{s}\|^2 = \|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2 + \|\mathbf{P}\mathbf{x} - \mathbf{s}\|^2$.

<!--en:cac1d6b3-->
- **Eigenschaften des Projektors:** Idempotent ($\mathbf{P}^2 = \mathbf{P}$) und symmetrisch ($\mathbf{P}^\top = \mathbf{P}$).

<!--en:e0ac8e4d-->
::: {.slide title="Normalgleichungen und orthogonale Projektion im Vergleich"}

<!--en:9d5ad7d6-3-->
[Kleinste Quadrate]{.kicker}

<!--en:7568ce8f-->
- **Normalgleichungen:** $\mathbf{A}^\top(\mathbf{y} - \mathbf{A}\hat{\mathbf{x}}) = \mathbf{0} \implies \mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{y}$.

<!--en:8bc4ca03-->
- **Warum numerische Verfahren QR bevorzugen:** Die Bildung von $\mathbf{A}^\top\mathbf{A}$ quadriert $\kappa(\mathbf{A})$, während $\mathbf{A} = \mathbf{Q}\mathbf{R}$ die Kondition erhält.

<!--en:ff4ca9b1-->
::: {.slide title="Das Gram-Schmidt-Verfahren und die QR-Zerlegung"}

<!--en:3c63ed48-->
[Orthonormalisierung]{.kicker}

<!--en:7a6282f0-->
- **Schrittweise Projektion:** Projektionen auf $\mathbf{q}_1, \ldots, \mathbf{q}_{j-1}$ abziehen und den Rest normieren: $\mathbf{v}_j = \mathbf{a}_j - \sum_{i<j}(\mathbf{q}_i\cdot\mathbf{a}_j)\mathbf{q}_i$, $\mathbf{q}_j = \mathbf{v}_j/\|\mathbf{v}_j\|$.

<!--en:daa4d4fb-->
- **Rechenbeispiel für ein $2 \times 2$-Paar:** $\mathbf{a}_1 = (3, 4) \implies \mathbf{q}_1 = (0.6, 0.8)$ und $\mathbf{a}_2 = (1, 0) \implies \mathbf{v}_2 = (0.64, -0.48) \implies \mathbf{q}_2 = (0.8, -0.6)$.

<!--en:6da00f6f-->
- **Obere Dreiecksform in $\mathbf{A} = \mathbf{Q}\mathbf{R}$:** Lesen wir die Schritte rückwärts, erhalten wir die Einträge $r_{ij} = \mathbf{q}_i\cdot\mathbf{a}_j$ für $i \le j$.

<!--en:286e457d-->
::: {.slide title="Gram-Schmidt von Hand und QR-Zerlegung nach Householder"}

<!--en:3c63ed48-2-->
[Orthonormalisierung]{.kicker}

<!--en:7b10f373-->
Vergleichen wir unser zweistufiges Gram-Schmidt-Verfahren mit `np.linalg.qr`, erhalten wir bis auf die Vorzeichen der Spalten dieselbe Orthonormalbasis:

<!--en:b997392c-->
**Spiegelungen in der Gleitkommaarithmetik:** Numerische Bibliotheken berechnen $\mathbf{A} = \mathbf{Q}\mathbf{R}$ mit Householder-Spiegelungen. So bleibt die Orthogonalität selbst bei nahezu kollinearen Spalten bis auf Maschinengenauigkeit erhalten. Das klassische Gram-Schmidt-Verfahren arbeitet mit Subtraktionen.

<!--en:dad93edf-->
::: {.slide title="Kosinusähnlichkeit im hochdimensionalen Raum"}

<!--en:2c8c006e-->
[Hochdimensionale Geometrie]{.kicker}

<!--en:fcad341a-->
- **Skaleninvariantes Richtungsmaß:** Die Kosinusähnlichkeit $\cos\theta = (\mathbf{v}\cdot\mathbf{w})/(\|\mathbf{v}\|\|\mathbf{w}\|) \in [-1, 1]$ misst allein die Übereinstimmung der Richtungen.

<!--en:58e0fc94-->
- **Invarianz gegenüber Helligkeit:** Dimmen auf $10\%$ der Bildhelligkeit verändert den euklidischen Abstand. Es gilt weiterhin $\cos\theta = 1$.

<!--en:bbc6f58c-->
- **Retrieval und Attention:** Kontrastives Lernen (SimCLR, CLIP) normiert die Vektoren. Attention verwendet $\mathbf{q}\cdot\mathbf{k}/\sqrt{d}$.

<!--en:f448c7b3-->
::: {.slide title="Zufällige Richtungen in ℝᵈ sind nahezu orthogonal"}

<!--en:b61e1bca-->
[Maßkonzentration]{.kicker}

<!--en:4ad33aaa-->
- **Isotrope Baseline:** Wir fixieren einen Einheitsvektor $\mathbf{u} \in \mathbb{R}^d$ und ziehen $\mathbf{v}$ gleichverteilt von der Einheitssphäre. Wegen der Rotationsinvarianz dürfen wir $\mathbf{u} = \mathbf{e}_1$ setzen, sodass $\cos\theta = v_1$ gilt.

<!--en:6c0b2989-->
- **Mittelwert und Varianz:** Aus der Vorzeichensymmetrie folgt $\mathbb{E}[\cos\theta] = \mathbb{E}[v_1] = 0$, aus der Koordinatensymmetrie und $\sum_{i=1}^d v_i^2 = 1$ folgt $d\,\mathbb{E}[v_1^2] = 1$:
  $$\operatorname{Var}(\cos\theta) = \mathbb{E}[v_1^2] = \frac{1}{d}$$

<!--en:f6f01709-->
- **Konzentration um null:** Typische Kosinuswerte zufälliger Richtungen skalieren wie $1/\sqrt{d}$ ($\approx 0.036$ für $\mathbb{R}^{784}$). Sie häufen sich in einer schmalen Spitze um $0$ ($90^\circ$).

<!--en:da8845d5-->
::: {.slide title="Simulation mit Zufallsvektoren und Kosinuswerte realer Bilder"}

<!--en:b61e1bca-2-->
[Maßkonzentration]{.kicker}

<!--en:9e0232b9-->
Normierte standardnormale Zufallsvektoren bestätigen die Streuung von $1/\sqrt{784} \approx 0.036$. Reale Testbilder aus Fashion-MNIST verhalten sich ganz anders:

<!--en:0b9f0e27-->
::: {.slide title="Simulation mit Zufallsvektoren und Kosinuswerte realer Bilder (Fortsetzung)"}

<!--en:b61e1bca-3-->
[Maßkonzentration]{.kicker}

<!--en:afc91a62-->
::: {.slide title="Warum reale Daten von der isotropen Baseline abweichen"}

<!--en:dd90cc28-->
[Geometrie der Daten]{.kicker}

<!--en:e90f7b36-->
- **Verschiebung durch den positiven Orthanten ($\approx 0.60 \pm 0.18$):** Rohe Pixelwerte in $[0, 1]^{784}$ liegen im nichtnegativen Orthanten. Dort sind Skalarprodukte nie negativ.

<!--en:ec842e69-->
- **Niedrige effektive Dimension nach Zentrierung ($\approx 0.00 \pm 0.34$):** Nach Abzug des Mittelwertbildes liegen die Kosinuswerte um $0$, ihre Streuung $0.34$ entspricht aber einer effektiven Dimension von $d_{\text{eff}} = 1/0.34^2 \approx 8.6$.

<!--en:d4d1d8aa-->
- **Vergleichsmaßstab für Retrieval:** Kosinuswerte stets mit denen zufälliger Paare aus *derselben* Datenverteilung vergleichen.

<!--en:7afc305b-->
::: {.slide title="Warum Skalarprodukt-Attention durch √d dividiert"}

<!--en:ab6d2e14-->
[Geometrie des Transformers]{.kicker}

<!--en:eab1533c-->
- **Wachstum der Varianz ($\operatorname{Var}(\mathbf{q}\cdot\mathbf{k}) = d$):** Bei unabhängigen Einträgen mit Varianz eins hat die Summe von $d$ unkorrelierten Produkten $q_i k_i$ die Standardabweichung $\sqrt{d}$.

<!--en:1120da0b-->
- **Sättigung der Softmax-Funktion vermeiden:** Bei $d = 64$ gilt für unskalierte Scores $\sigma = 8$. Eine Differenz von $1\sigma$ ergibt die Gewichte $(0.9997, 0.0003)$. Division durch $\sqrt{d} = 8$ liefert wieder $(0.73, 0.27)$.

<!--en:6e319573-->
- **Retrieval-Schwelle bei $n$ Schlüsseln:** Der größte zufällige Kosinus unter $n$ Schlüsseln wächst wie $\sqrt{2\ln n / d}$.

<!--en:eb0e7757-->
::: {.slide title="Hyperebenen und vorzeichenbehafteter Abstand zur Grenze"}

<!--en:a376ac07-->
[Lineare Entscheidungsgrenzen]{.kicker}

<!--en:2e7febb2-->
- **Gleichung der Hyperebene:** In $\mathbb{R}^d$ beschreibt $\{\mathbf{x} : \mathbf{w}\cdot\mathbf{x} + b = 0\}$ mit Normalenvektor $\mathbf{w} \neq \mathbf{0}$ und Offset $b$ eine $(d-1)$-dimensionale ebene Fläche, die den Raum in zwei Halbräume teilt ($\gtrless 0$).

<!--en:c2267dda-->
- **Vorzeichenbehafteter Abstand:** Wählen wir ein beliebiges $\mathbf{x}_0$ auf der Hyperebene ($\mathbf{w}\cdot\mathbf{x}_0 = -b$) und projizieren $\mathbf{x} - \mathbf{x}_0$ auf $\mathbf{w}$, erhalten wir:
  $$\text{dist}(\mathbf{x}) = \frac{\mathbf{w}\cdot(\mathbf{x} - \mathbf{x}_0)}{\|\mathbf{w}\|} = \frac{\mathbf{w}\cdot\mathbf{x} + b}{\|\mathbf{w}\|}$$

<!--en:09a59ef3-->
- **Geometrische Margin:** Für das Label $y \in \{-1, +1\}$ ist die Margin $y(\mathbf{w}\cdot\mathbf{x} + b)/\|\mathbf{w}\|$ bei richtiger Klassifikation positiv und unter $(\mathbf{w}, b) \mapsto (c\mathbf{w}, cb)$ skaleninvariant.

<!--en:82739b85-->
::: {.slide title="Eine Hyperebene für die Klassifikation nach dem nächsten Klassenmittelwert"}

<!--en:61e48d66-->
[Lineare Klassifikatoren]{.kicker}

<!--en:8dbf045f-->
- **Wegfall des quadratischen Terms:** In $\|\mathbf{x} - \bar{\mathbf{x}}_1\|^2 < \|\mathbf{x} - \bar{\mathbf{x}}_0\|^2$ fällt $\|\mathbf{x}\|^2$ auf beiden Seiten weg.

<!--en:5eb1a43a-->
- **Mittelsenkrechte:** Mit dem Normalenvektor $\mathbf{w} = \bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_0$ und dem Offset $b = -\mathbf{w}\cdot\tfrac{1}{2}(\bar{\mathbf{x}}_0 + \bar{\mathbf{x}}_1)$ für die Grenze durch den Mittelpunkt erreichen wir bei T-Shirts und Hosen eine Genauigkeit von $92\%$.

<!--en:5470f574-->
::: {.slide title="Histogramm der vorzeichenbehafteten Abstände und Überlappung der Klassen"}

<!--en:445e312d-->
[Fehleranalyse]{.kicker}

<!--en:c26972f0-->
- **Eindimensionale Projektion entlang von $\mathbf{w}$:** Die Projektion aller $2{,}000$ Testbilder auf den Normalenvektor $\mathbf{w} = \bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_0$ reduziert jedes Bild mit $784$ Pixeln auf seinen vorzeichenbehafteten Abstand $(\mathbf{w}\cdot\mathbf{x} + b)/\|\mathbf{w}\|$ von der Grenze.

<!--en:981006e8-->
- **Klassenmittelwerte bei $\mp\|\mathbf{w}\|/2 \approx \mp 3.2$ (Gipfel nahe $\mp 4$):** Die T-Shirts bilden auf der negativen Seite eine breite Kuppe ($\sigma \approx 2.4$), die Hosen auf der positiven Seite einen schmaleren Gipfel ($\sigma \approx 1.7$).

<!--en:5ac9a725-->
- **Überlappende Ausläufer ($169$ Fehler):** Die Grenze liegt nur $1.4\sigma$ bis $1.9\sigma$ von den Klassenzentren entfernt. Die Ausläufer reichen über null hinaus ($101$ T-Shirts und $68$ Hosen falsch klassifiziert).

<!--en:2e2673f0-->
::: {.slide title="Warum der nächste Klassenmittelwert bei stark streuenden Klassen versagt"}

<!--en:fecb6f23-->
[Grenzen des Klassifikators]{.kicker}

<!--en:6621ae01-->
- **Verhältnis von Signal zu Streuung:** Die Regel mit der Grenze im Mittelpunkt berücksichtigt nur die Klassenzentren $\bar{\mathbf{x}}_0, \bar{\mathbf{x}}_1$. Sie ignoriert, wie stark jede Punktwolke entlang von $\mathbf{w}$ streut:

<!--en:36b1ca05-->
- **Von $92\%$ auf $72\%$:** Bei ähnlich aussehenden Kleidungsstücken (T-Shirt und Hemd, Pullover und Mantel) sinkt der halbe Abstand $\|\mathbf{w}\|/2$ auf $0.6\sigma$ bis $1.0\sigma$. Die Grenze schneidet mitten durch beide Punktwolken.

<!--en:f189fd46-->
- **Schichten als Stapel von Hyperebenen:** Eine lineare Schicht $\mathbf{A}\mathbf{x} + \mathbf{b}$ berechnet $m$ Scores für Hyperebenen parallel (einen pro Zeile). Das Lernen mit der Fisher-Diskriminante, der Softmax-Regression oder tiefen Darstellungen dreht die Normalenvektoren von Richtungen hoher Varianz weg.

<!--en:7ed1e7fb-->
::: {.slide title="Verständnisfragen zu Projektionen, Kosinuswerten und Hyperebenen"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:6bb59b70-->
1. **Projektion und Satz von Pythagoras:** Projizieren Sie $\mathbf{v} = (3, 2)$ auf $\mathbf{w} = (2, 1)$, bestimmen Sie das Residuum $\mathbf{r}$ und überprüfen Sie $\|\mathbf{v}\|^2 = \|\operatorname{proj}_{\mathbf{w}}\mathbf{v}\|^2 + \|\mathbf{r}\|^2$.

<!--en:327acaa2-->
2. **Skalierung bei Attention:** Ein Transformer-Kopf hat die Dimension $d = 100$. Die Einträge der Abfragen und Schlüssel haben Varianz eins. Welche Standardabweichung haben die Skalarprodukte $\mathbf{q}\cdot\mathbf{k}$ vor der Skalierung?

<!--en:fc0cdb87-->
3. **Hyperebenen aus Matrixzeilen:** Gegeben seien $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ und $\mathbf{x} = (2, -1)$. Welche vorzeichenbehafteten Abstände hat $\mathbf{x}$ zu den beiden Ursprungshyperebenen, die die Zeilen von $\mathbf{A}$ definieren?

<!--en:c334d885-->
**Antworten:**
**1.** Es gilt $t^\star = \frac{8}{5} = 1.6$, also $\operatorname{proj}_{\mathbf{w}}\mathbf{v} = (3.2, 1.6)$ und $\mathbf{r} = (-0.2, 0.4)$ mit $\mathbf{r}\cdot\mathbf{w} = 0$ und $13 = 12.8 + 0.2$.
**2.** Es gilt $\operatorname{Std}(\mathbf{q}\cdot\mathbf{k}) = \sqrt{d} = 10$. Daher dividiert Attention die Scores durch $\sqrt{100} = 10$.
**3.** Zeile 1 liefert $\mathbf{a}_1\cdot\mathbf{x} = 0$ (der Punkt liegt auf Hyperebene 1). Zeile 2 liefert $\mathbf{a}_2\cdot\mathbf{x} = -5$, also beträgt der vorzeichenbehaftete Abstand $-5/\sqrt{10} \approx -1.58$.

<!--en:a1a0ae70-->
::: {.slide title="Zusammenfassung zu Vektoren, Winkeln und Projektionen"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:f51ab8b6-->
| Begriff | Mathematische Formel | Wesentliche Eigenschaft oder Rolle im Deep Learning |

<!--en:c3d67178-->
| **Projektion auf eine Gerade** | $\operatorname{proj}_{\mathbf{w}}\mathbf{v} = \frac{\mathbf{v}\cdot\mathbf{w}}{\mathbf{w}\cdot\mathbf{w}}\mathbf{w}$ | Residuum $\mathbf{r} = \mathbf{v} - \operatorname{proj}_{\mathbf{w}}\mathbf{v} \perp \mathbf{w}$, Satz von Pythagoras $\|\mathbf{v}\|^2 = \|\operatorname{proj}\|^2 + \|\mathbf{r}\|^2$ |

<!--en:f162a735-->
| **Cauchy-Schwarz-Ungleichung und Winkel** | $|\mathbf{v}\cdot\mathbf{w}| \le \|\mathbf{v}\|\|\mathbf{w}\|$, $\cos\theta = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}$ | Kosinus liegt in jeder Dimension in $[-1, 1]$. In der Gleitkommaarithmetik auf $[-1, 1]$ begrenzen. |

<!--en:8937a78b-->
| **Projektor auf einen Unterraum** | $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$ ($\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}_k$) | Eindeutiger nächstgelegener Punkt in $S$. Idempotent ($\mathbf{P}^2 = \mathbf{P}$) und symmetrisch ($\mathbf{P}^\top = \mathbf{P}$). |

<!--en:e358e2c3-->
| **Kleinste Quadrate und QR-Zerlegung** | $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{y}$, $\mathbf{A} = \mathbf{Q}\mathbf{R}$ | Die QR-Zerlegung nach Gram-Schmidt oder Householder vermeidet das Quadrieren von $\kappa(\mathbf{A})$ in den Normalgleichungen. |

<!--en:70268dc5-->
| **Kosinuswerte zufälliger Richtungen in $\mathbb{R}^d$** | $\mathbb{E}[\cos\theta] = 0$, $\operatorname{Var}(\cos\theta) = 1/d$ | Typische Abweichung des Kosinus von null bei zufälligen Richtungen: $\pm 1/\sqrt{d}$. Attention skaliert mit $1/\sqrt{d}$. |

<!--en:fc4fcee6-->
| **Hyperebene und nächster Klassenmittelwert** | $\text{dist}(\mathbf{x}) = \frac{\mathbf{w}\cdot\mathbf{x} + b}{\|\mathbf{w}\|}$, $\mathbf{w} = \bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_0$ | Die Grenze durch den Mittelpunkt erreicht $92\%$ Genauigkeit bei T-Shirts und Hosen. Lineare Schichten kombinieren $m$ Hyperebenen. |
