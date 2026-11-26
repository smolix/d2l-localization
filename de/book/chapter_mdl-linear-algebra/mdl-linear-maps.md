<!--en:ec9b42f4-->
# Matrizen als lineare Abbildungen

<!--en:8d05172a-->
In :numref:`sec_mdl-geometry-linear-algebraic-ops` haben wir das Matrix-Vektor-Produkt $\mathbf{A}\mathbf{x}$ zeilenweise gelesen. Der Eintrag $i$ ist das Skalarprodukt der Zeile $i$ mit $\mathbf{x}$ und entspricht einem einzelnen Hyperebenentest. Diese Sicht zeigt, was eine Matrix für einen einzelnen Eingabevektor berechnet. Wie sie auf den gesamten Eingaberaum wirkt, bleibt jedoch verborgen. Insbesondere erkennen wir so weder, ob die Abbildung Information erhält oder verwirft, noch, ob sich ihre Wirkung rückgängig machen lässt.

<!--en:7ea5b28d-->
Diese Frage stellt sich, sobald eine Schicht eines neuronalen Netzes schmaler ist als ihre Eingabe. Eine vollständig verbundene Schicht, die die 784 Pixel eines Fashion-MNIST-Bildes auf 256 versteckte Merkmale abbildet, entspricht einer $256 \times 784$-Matrix. Da ihre Ausgabe weniger Zahlen enthält als ihre Eingabe, gehen zwangsläufig Dimensionen verloren. Wie wir gleich genauer herleiten, bildet sie mindestens $784 - 256 = 528$ unabhängige Eingaberichtungen auf null ab. Verschiedene Bilder können also identische Merkmale ergeben, noch bevor eine Nichtlinearität auf sie wirkt.

<!--en:bc99ff47-->
Selbst eine quadratische Matrix kann Information verlieren. Betrachten wir die Beispielmatrix dieses Kapitels, $\mathbf{A} = \left(\begin{smallmatrix}1 & 2\\ -1 & 3\end{smallmatrix}\right)$. Sie bildet $(2, -1)$ auf $(0, -5)$ ab, und eine andere Matrix bildet $(0, -5)$ wieder auf $(2, -1)$ ab. Tatsächlich kehrt die zweite Matrix die Wirkung von $\mathbf{A}$ für jede Eingabe um. Damit verliert $\mathbf{A}$ keine Information. Anders verhält sich $\mathbf{B} = \left(\begin{smallmatrix}2 & 1\\ 4 & 2\end{smallmatrix}\right)$. Diese Matrix bildet sowohl $(1, 1)$ als auch $(0, 3)$ auf $(3, 6)$ ab. Kennen wir nur die Ausgabe $(3, 6)$, können wir die ursprüngliche Eingabe nicht bestimmen. Daher kann keine Matrix die Wirkung von $\mathbf{B}$ umkehren.

<!--en:bdcd5eb9-->
Ob eine lineare Abbildung umkehrbar ist, entscheidet darüber, ob eine Schicht Information verwirft, ob ein lineares Gleichungssystem eine eindeutige Lösung hat und wie sich eine Wahrscheinlichkeitsdichte bei einem Variablenwechsel skaliert (:numref:`subsec_mdl-change-of-variables`). Um diese Frage zu beantworten, betrachten wir nun die Spalten. Spalte $j$ einer Matrix gibt das Bild des Einheitsvektors auf der $j$-ten Koordinatenachse an. Diese Bilder bestimmen bereits die gesamte Abbildung. Die maximale Anzahl linear unabhängiger Spalten verrät uns, wie viele Eingabedimensionen erhalten bleiben. Das orientierte Volumen des transformierten Einheitswürfels zeigt mit einer einzigen Zahl an, ob eine Dimension verloren geht. Wir entwickeln beide Ideen zunächst in zwei Dimensionen, wo sich jede Aussage zeichnen lässt, und erweitern sie dann auf $n$ Dimensionen.

<!--en:7ed4149e-->
Drei Ergebnisse gliedern diesen Abschnitt. Wir bestimmen, wie die Anzahlen erhaltener und verlorener Dimensionen zusammenhängen, wann genau eine quadratische Matrix eine Inverse besitzt und wie sich die Volumenskalierung bei der Verkettung zweier Abbildungen verhält. Die allgemeine $n$-dimensionale Konstruktion der Determinante (:numref:`subsec_mdl-determinant-general`) ist der technisch anspruchsvollste Teil. Bei der ersten Lektüre genügt es, ihn zu überfliegen.

<!--en:0aa5f6f0-->
## Lineare Abbildungen

<!--en:d7834f5d-->
### Spalten als Bilder der Achsen

<!--en:2c4655a8-->
Eine Matrix wirkt auf unendlich viele Eingabevektoren, obwohl wir sie als endliche Zahlentabelle speichern. Wie können vier Zahlen die Ausgabe für jeden Vektor der Ebene festlegen? Wenden wir eine $2 \times 2$-Matrix $\mathbf{A}$ mit den Einträgen $p, q, r, s$ auf $\mathbf{x} = (x, y)$ an. Fassen wir die Terme mit dem Faktor $x$ und die mit dem Faktor $y$ jeweils zusammen, ergibt sich

<!--en:4c9c261d-->
Damit ist $\mathbf{A}\mathbf{x}$ eine gewichtete Summe der beiden Spalten von $\mathbf{A}$. Als Gewichte dienen die Koordinaten von $\mathbf{x}$. Wegen $\mathbf{A}\mathbf{e}_1 = (p, r)$ und $\mathbf{A}\mathbf{e}_2 = (q, s)$ sind die Spalten die Bilder der Standardbasisvektoren $\mathbf{e}_1 = (1, 0)$ und $\mathbf{e}_2 = (0, 1)$. Kennen wir die Bilder dieser beiden Achsenvektoren, so liegt das Bild jedes anderen Vektors der Ebene fest.

<!--en:9a5dd830-->
Bei unserer Beispielmatrix ($p = 1$, $q = 2$, $r = -1$, $s = 3$) sind die Bilder der beiden Achsenvektoren $\mathbf{A}\mathbf{e}_1 = (1, -1)$ und $\mathbf{A}\mathbf{e}_2 = (2, 3)$. Da sich die Eingabe $\mathbf{v} = (2, -1)$ als $2\mathbf{e}_1 - \mathbf{e}_2$ schreiben lässt, ist ihr Bild genau die entsprechende Linearkombination der Spalten:

<!--en:de1ab8bb-->
Die zeilenweise Rechnung bestätigt das Ergebnis, denn $(1, 2)\cdot(2, -1) = 0$ und $(-1, 3)\cdot(2, -1) = -5$. Beide Sichtweisen liefern dasselbe Produkt. An den Spalten erkennen wir darüber hinaus, wie $\mathbf{A}$ auf die gesamte Ebene wirkt. :numref:`fig_mdl-la-linear-map` veranschaulicht die Geometrie. Das von $\mathbf{e}_1$ und $\mathbf{e}_2$ aufgespannte Einheitsquadrat geht in das von den beiden Spalten aufgespannte Parallelogramm über. Der Gitterpunkt $\mathbf{v}$ geht in den entsprechenden Punkt $\mathbf{A}\mathbf{v}$ des transformierten Gitters über.

<!--en:b98fd90b-->
![Wirkung der Beispielmatrix $\mathbf{A} = \left(\begin{smallmatrix}1 & 2\\ -1 & 3\end{smallmatrix}\right)$ auf die Ebene. (a) Eingabegitter mit schattiertem Einheitsquadrat, Basisvektoren und Eingabe $\mathbf{v} = (2, -1)$. (b) Das Bild. Dabei geht $\mathbf{e}_1$ in $(1, -1)$ über, $\mathbf{e}_2$ in $(2, 3)$ und $\mathbf{v}$ in $(0, -5)$. Geraden bleiben Geraden, der Ursprung bleibt fest und gleichmäßig verteilte Punkte bleiben gleichmäßig verteilt. Das schattierte Quadrat geht in das schattierte Parallelogramm über, dessen Flächeninhalt zur Determinante führt. Der Maßstab in (b) beträgt ein Drittel des Maßstabs in (a).](../img/mdl-la-linear-map.svg)

<!--en:77bad75b-->
Diese geometrische Deutung der Spalten gilt in jeder Dimension. Eine $m \times n$-Matrix bildet Vektoren aus $\mathbb{R}^n$ auf Vektoren in $\mathbb{R}^m$ ab. Die Abbildung ist *linear*:

<!--en:9256b810-->
In Worten ist das Bild einer gewichteten Summe die mit denselben Gewichten gebildete Summe der Bilder. Schreiben wir eine beliebige Eingabe als $\mathbf{x} = \sum_j x_j \mathbf{e}_j$, so ergibt sich $\mathbf{A}\mathbf{x} = \sum_j x_j\,\mathbf{A}\mathbf{e}_j$. Dabei wählt die Multiplikation mit $\mathbf{e}_j$ die Spalte $j$ aus. Umgekehrt lässt sich jede Abbildung $f : \mathbb{R}^n \to \mathbb{R}^m$, die diese Linearitätsregel erfüllt, als Matrix schreiben. Ihre $j$-te Spalte ist $f(\mathbf{e}_j)$, denn es gilt $f(\mathbf{x}) = f(\sum_j x_j\mathbf{e}_j) = \sum_j x_j f(\mathbf{e}_j)$. Lineare Abbildungen von $\mathbb{R}^n$ nach $\mathbb{R}^m$ und $m \times n$-Matrizen beschreiben also dasselbe Objekt.

<!--en:9027c71d-->
Die Linearität erklärt auch, weshalb das Gitter in :numref:`fig_mdl-la-linear-map` seine Gitterstruktur behält. Jede Gerade $\mathbf{p} + t\mathbf{u}$ geht in $\mathbf{A}\mathbf{p} + t\,\mathbf{A}\mathbf{u}$ über, also wieder in eine Gerade (oder in einen einzigen Punkt, falls $\mathbf{A}\mathbf{u} = \mathbf{0}$). Der Ursprung bleibt stets fest, denn $\mathbf{A}\mathbf{0} = \mathbf{0}$.

<!--en:07b295aa-->
Die Matrix muss dafür nicht quadratisch sein. Eine $2 \times 3$-Matrix bildet $\mathbb{R}^3$ in den von ihren drei Spalten aufgespannten Raum in $\mathbb{R}^2$ ab. Sind mindestens zwei Spalten linear unabhängig, füllt das Bild die gesamte Ausgabeebene aus. In umgekehrter Richtung bildet eine $3 \times 2$-Matrix die Ebene in $\mathbb{R}^3$ ab. Je nach ihren beiden Spalten bilden die Ausgaben dort eine Ebene durch den Ursprung, eine Gerade oder nur den Ursprung. Jede vollständig verbundene Schicht eines neuronalen Netzes führt eine solche Abbildung mit einer rechteckigen Matrix aus. Darauf folgen eine Verschiebung um den Bias und eine Nichtlinearität.

<!--en:29ed95bc-->
Zur Veranschaulichung zeigt :numref:`fig_mdl-la-map-gallery`, wie sechs $2 \times 2$-Matrizen auf das Einheitsquadrat wirken. Eine Drehung um $90^\circ$ gegen den Uhrzeigersinn bildet $\mathbf{e}_1$ auf $\mathbf{e}_2$ und $\mathbf{e}_2$ auf $-\mathbf{e}_1$ ab. Ihre Spalten sind also $(0, 1)$ und $(-1, 0)$. Eine horizontale Scherung lässt $\mathbf{e}_1$ fest und kippt $\mathbf{e}_2$ nach $(1, 1)$. Dabei verschiebt sich jede horizontale Gerade seitlich proportional zu ihrer Höhe. In den letzten beiden Teilbildern geschieht etwas grundsätzlich anderes. Eine Dimension geht verloren, und das gesamte Quadrat fällt mit der ganzen Ebene auf eine einzige Gerade zusammen.

<!--en:670bef09-->
![Sechs Möglichkeiten, wie eine $2 \times 2$-Matrix das gestrichelte Einheitsquadrat transformiert. In jedem Teilbild zeigen die Pfeile die Spalten, also die Bilder von $\mathbf{e}_1$ (blau) und $\mathbf{e}_2$ (orange). Die schattierte Fläche ist das Bild des Quadrats. In (e) und (f) fällt die Ebene auf eine Gerade zusammen. In (f) sind beide Spalten gleich $(1, 1)$.](../img/mdl-la-map-gallery.svg)

<!--en:66696db6-->
Eine vertraute Operation fehlt unter diesen Beispielen, die Verschiebung $\mathbf{x} \mapsto \mathbf{x} + \mathbf{c}$ mit $\mathbf{c} \neq \mathbf{0}$. Da sie den Ursprung verschiebt, ist sie nicht linear. Eine affine Schicht $\mathbf{W}\mathbf{x} + \mathbf{b}$ in einem neuronalen Netz besteht also aus einer linearen Abbildung mit anschließender Verschiebung.

<!--en:1ff253c3-->
### Verkettung

<!--en:e33bdbbe-->
Zwei aufeinanderfolgende lineare Schichten ohne Nichtlinearität dazwischen lassen sich zu einer einzigen linearen Abbildung zusammenfassen. Welche Matrix stellt diese Abbildung dar? Untersuchen wir das an zwei konkreten Abbildungen aus :numref:`fig_mdl-la-map-gallery`, der Drehung um $90^\circ$ mit der Matrix $\mathbf{R}_{90}$ und der Scherung $\mathbf{T}$ (benannt nach dem englischen Wort für das Kippen der vertikalen Achse):

<!--en:8c001347-->
Beiden Matrizen begegnen wir in diesem Kapitel wieder. Verfolgen wir für beide Reihenfolgen die Bilder der Standardbasisvektoren. Scheren wir zuerst, bleibt $\mathbf{e}_1$ bei $(1, 0)$, und die Drehung bildet diesen Vektor auf $(0, 1)$ ab. Dagegen geht $\mathbf{e}_2$ durch die Scherung in $(1, 1)$ und durch die Drehung in $(-1, 1)$ über. „Erst scheren, dann drehen“ ergibt also die Spalten $(0, 1)$ und $(-1, 1)$. In umgekehrter Reihenfolge bildet die Drehung $\mathbf{e}_1$ auf $(0, 1)$ ab, und die Scherung verschiebt diesen Vektor nach $(1, 1)$. Die Drehung bildet $\mathbf{e}_2$ auf $(-1, 0)$ ab, und die Scherung lässt diesen Vektor unverändert. Die beiden Verkettungen lauten somit

<!--en:f81e810c-->
Wie :numref:`fig_mdl-la-composition` zeigt, sind dies zwei verschiedene Abbildungen.

<!--en:f255d43c-->
![Bei der Verkettung kommt es auf die Reihenfolge an. (a) Die Scherung $\mathbf{T}$ (gepunkteter Umriss) mit anschließender Drehung $\mathbf{R}_{90}$ ergibt $\mathbf{R}_{90}\mathbf{T}$. (b) Die Drehung (gepunkteter Umriss) mit anschließender Scherung ergibt $\mathbf{T}\mathbf{R}_{90}$. Die Beschriftungen geben die endgültigen Bilder von $\mathbf{e}_1$ (blau) und $\mathbf{e}_2$ (orange) an. Matrixprodukte liest man stets von rechts nach links.](../img/mdl-la-composition.svg)

<!--en:55b4de63-->
Die Matrizenmultiplikation liefert die Rechenregel für solche Verkettungen. Betrachten wir zwei beliebige Matrizen $\mathbf{A}$ und $\mathbf{B}$ mit passenden Dimensionen. Wenden wir $\mathbf{B}$ auf $\mathbf{v}$ an, dann $\mathbf{A}$ auf das Ergebnis und betrachten wir den Eintrag $i$:

<!--en:5d2f384c-->
Beide Abbildungen nacheinander anzuwenden entspricht der Anwendung der einzigen Matrix $\mathbf{A}\mathbf{B}$. Ihr Eintrag an der Stelle $(i, k)$ ist das Skalarprodukt der Zeile $i$ von $\mathbf{A}$ mit der Spalte $k$ von $\mathbf{B}$. Beachten wir die Reihenfolge. Das Produkt $\mathbf{A}\mathbf{B}$ bedeutet „zuerst $\mathbf{B}$ anwenden, dann $\mathbf{A}$“. Wie die Funktionsverkettung $f(g(x))$ lesen wir es von rechts nach links. Setzen wir $\mathbf{v} = \mathbf{e}_j$ in :eqref:`eq_mdl-composition` ein, erhalten wir außerdem eine Regel für die Spalten. Spalte $j$ von $\mathbf{A}\mathbf{B}$ ist $\mathbf{A}\mathbf{b}_j$, also das Bild unter $\mathbf{A}$ der Spalte $j$ von $\mathbf{B}$. Darauf greifen wir bei der Determinante zurück.

<!--en:e63e4e3e-->
Die Deutung als Verkettung erklärt die zunächst ungewöhnliche Definition der Multiplikation. Die Formel „Zeile mal Spalte“ ergibt sich unmittelbar aus der Verkettung zweier linearer Abbildungen. Auch die algebraischen Eigenschaften folgen direkt. Die Matrizenmultiplikation ist assoziativ, $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$, weil die Verkettung von Funktionen assoziativ ist. Beide Seiten wenden zuerst $\mathbf{C}$, dann $\mathbf{B}$ und zuletzt $\mathbf{A}$ an. Im Allgemeinen ist die Multiplikation nicht kommutativ, denn zwei geometrische Operationen in umgekehrter Reihenfolge führen meist zu unterschiedlichen Ergebnissen. NumPy verwendet für die Matrizenmultiplikation den Operator `@`. Prüfen wir, ob er beide Verkettungen von Drehung und Scherung liefert (im folgenden Code als `R90` und `S` gespeichert).

<!--en:38117d12-->
Manche Matrizenpaare kommutieren durchaus. Zwei Drehungen in der Ebene kommutieren, denn eine Drehung um $\phi$ und eine weitere um $\psi$ ergeben in beiden Reihenfolgen eine Drehung um $\phi + \psi$. Auch zwei Diagonalmatrizen kommutieren, da jede die Koordinatenachsen unabhängig voneinander skaliert. Aufgabe :exref:`ex_mdl-linear-maps-commute-rotation` charakterisiert alle $2 \times 2$-Matrizen, die mit $\mathbf{R}_{90}$ kommutieren.

<!--en:3972634c-->
### Orthogonale Matrizen

<!--en:de43d63c-->
Manche linearen Abbildungen bewegen den Raum starr und erhalten dabei alle Längen und Winkel. Welche Matrizen haben diese Eigenschaft? Betrachten wir eine Drehung in der Ebene um den Winkel $\phi$ gegen den Uhrzeigersinn. Sie bildet $\mathbf{e}_1$ auf $(\cos\phi, \sin\phi)$ und $\mathbf{e}_2$ auf $(-\sin\phi, \cos\phi)$ ab. Ihre Matrix lautet daher

<!--en:298911c8-->
Ihre Spalten haben die Länge eins, $\sqrt{\cos^2\phi + \sin^2\phi} = 1$, und ihr Skalarprodukt ist null, $-\cos\phi\sin\phi + \sin\phi\cos\phi = 0$. Sie sind also orthonormal. Diese Bedingung haben wir in :numref:`sec_mdl-geometry-linear-algebraic-ops` als $\mathbf{R}_\phi^\top\mathbf{R}_\phi = \mathbf{I}$ geschrieben. Für $\phi = 90^\circ$ erhalten wir wieder $\mathbf{R}_{90}$. Diese Matrix bildet $(3, 2)$ auf $(-2, 3)$ ab. Beide Vektoren haben die Länge $\sqrt{13}$. Auch die Spiegelung $\operatorname{diag}(1, -1)$ an der $x$-Achse hat orthonormale Spalten.

<!--en:9bea61dc-->
In jeder Dimension heißt eine quadratische Matrix $\mathbf{Q}$ *orthogonal*, wenn ihre Spalten orthonormal sind, wenn also gilt:

<!--en:479c4f8d-->
Zwei Feinheiten dieser Definition verdienen Beachtung. Erstens genügt es nicht, dass die Spalten senkrecht aufeinander stehen. So hat $\operatorname{diag}(1, 10)$ zueinander senkrechte Spalten der Längen $1$ und $10$. Dennoch verändert die Matrix den Winkel von $90^\circ$ zwischen $(1, 1)$ und $(1, -1)$ auf $\arccos(-99/101) \approx 169^\circ$ (:exref:`ex_mdl-geometry-linear-algebraic-ops-angle-preserving`). Zweitens ist die Bezeichnung „orthogonal“ quadratischen Matrizen vorbehalten. Eine hohe $n \times k$-Matrix mit $k < n$ orthonormalen Spalten erhält zwar Längen. Sie bettet aber $\mathbb{R}^k$ in einen größeren Raum $\mathbb{R}^n$ ein und hat daher keine beidseitige Inverse (:numref:`subsec_mdl-inverses`).

<!--en:9cf2615b-->
Eine orthogonale Abbildung erhält jedes Skalarprodukt:

<!--en:cb5fdbe3-->
Setzen wir $\mathbf{y} = \mathbf{x}$, erhalten wir $\|\mathbf{Q}\mathbf{x}\| = \|\mathbf{x}\|$. Längen bleiben also erhalten. Weil Winkel nur von Skalarprodukten und Längen abhängen, bleiben auch sie erhalten. Drehen wir eine ganze Menge von Einbettungsvektoren mit derselben orthogonalen Matrix, so bleibt jede paarweise Kosinusähnlichkeit unverändert.

<!--en:b044b8c2-->
Die Definitionsgleichung $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ besagt, dass $\mathbf{Q}^\top$ die Wirkung von $\mathbf{Q}$ aufhebt, wenn wir die transponierte Matrix *danach* anwenden. Für eine quadratische Matrix $\mathbf{Q}$ hebt $\mathbf{Q}^\top$ die Wirkung von $\mathbf{Q}$ auch dann auf, wenn wir die transponierte Matrix *davor* anwenden. Es gilt also $\mathbf{Q}\mathbf{Q}^\top = \mathbf{I}$. Damit sind auch die Zeilen von $\mathbf{Q}$ orthonormal, und es gilt $\mathbf{Q}^{-1} =
\mathbf{Q}^\top$. Das ergibt sich nicht unmittelbar aus der Definition. Es folgt aus einem allgemeinen Satz über quadratische Matrizen, den wir in :numref:`subsec_mdl-inverses` beweisen.

<!--en:95304685-->
In zwei Dimensionen sind Drehungen und Spiegelungen die einzigen orthogonalen Matrizen. Die erste Spalte muss ein Einheitsvektor $(\cos\phi, \sin\phi)$ sein. Für die zweite Spalte bleiben nur zwei dazu senkrechte Einheitsvektoren, nämlich $\pm(-\sin\phi, \cos\phi)$. Das Pluszeichen ergibt die Drehung $\mathbf{R}_\phi$. Das Minuszeichen ergibt $\mathbf{R}_\phi\operatorname{diag}(1, -1)$, eine Spiegelung an der $x$-Achse mit anschließender Drehung. Zusammen entsprechen sie wiederum einer Spiegelung an einer Ursprungsgeraden. Später unterscheiden wir Drehungen und Spiegelungen auf einen Blick am Vorzeichen der Determinante (:numref:`subsec_mdl-multiplicativity`).

<!--en:41701e97-->
### Tensoren und einsteinsche Summenkonvention

<!--en:e42e6072-->
In einem Transformer vergleicht die Self-Attention jede Abfrage mit jedem Schlüssel, und zwar in jedem Attention-Kopf und für jede Sequenz eines Batchs. Bezeichnen wir den Batch mit $b$, den Kopf mit $h$, die Abfrageposition mit $i$, die Schlüsselposition mit $t$ und die Merkmalsdimension innerhalb eines Kopfs mit $j$. Dann lauten die Attention-Scores

<!--en:7728644c-->
Dabei sind $q_{bhij}$ und $k_{bhtj}$ Einträge der vierachsigen Abfrage- und Schlüsseltensoren (:numref:`sec_ndarray`). Jeder Score ist ein Skalarprodukt entlang der Merkmalsachse $j$. Wir berechnen es für jede Kombination der vier übrigen Indizes. In der üblichen Matrixnotation lässt sich diese Kontraktion mit vier Ausgabeindizes nur umständlich ausdrücken. Die Bezeichnung „batchweise Matrizenmultiplikation mit Transposition der letzten beiden Achsen“ verschleiert, über welche Achse wir summieren.

<!--en:245002e0-->
Betrachten wir die bisherigen Produkte im Zusammenhang, erkennen wir stets dasselbe Muster. Für das Skalarprodukt, das Matrix-Vektor-Produkt, das Matrixprodukt und die *Spur* $\operatorname{tr}(\mathbf{A})$ (die Summe der Diagonaleinträge einer quadratischen Matrix) gilt:

<!--en:6ab0e659-->
Die Spur lernen wir hier erstmals kennen. In :numref:`sec_mdl-eigendecompositions` begegnet sie uns als Summe der Eigenwerte einer Matrix wieder. Jede der vier Formeln multipliziert Einträge miteinander und summiert über einen doppelt auftretenden Index. Die *einsteinsche Summenkonvention* macht dieses Muster zur Regel. Wir lassen das Summenzeichen weg und summieren automatisch über jeden wiederholten Index. Damit schreiben wir $\mathbf{v}\cdot\mathbf{w} = v_i w_i$ und $(\mathbf{A}\mathbf{B})_{ik} = a_{ij}b_{jk}$.

<!--en:35c936fa-->
Die Attention-Scores zeigen eine Grenze der klassischen Konvention. Die Indizes $b$ und $h$ für Batch und Kopf treten in beiden Faktoren auf. Wir wollen sie jedoch beibehalten, statt über sie zu summieren. Numerikbibliotheken lösen dies in `einsum` durch die ausdrückliche Angabe der beizubehaltenden Indizes. Eine Zeichenkette enthält die Eingabeindizes, den Pfeil `->` und die Ausgabeindizes. Über jeden Index, der auf der rechten Seite fehlt, wird summiert. Drücken wir unsere vier grundlegenden Produkte so aus und prüfen wir sie an unseren Beispielmatrizen.

<!--en:5dac6180-->
Alle Ausgaben stimmen mit unseren bisherigen Rechnungen von Hand überein, auch $\operatorname{tr}\mathbf{A} = 1 + 3 = 4$. In derselben Notation genügt für die Attention-Scores eine Zeile. Im Folgenden berechnen wir sie mit `einsum`, prüfen die Übereinstimmung mit dem batchweisen Matrixprodukt mit den transponierten Schlüsseln und zählen alle Multiplikationsadditionen.

<!--en:7d187bba-->
Die Ausgabe hat die Form $(2, 3, 4, 5)$. Das ergibt pro Sequenz und Kopf eine $4 \times 5$-Tabelle von Scores. Beide Ausdrücke stimmen überein. Die Kontraktion benötigt $2 \cdot 3 \cdot 4 \cdot 5 \cdot 8 = 960$ Multiplikationsadditionen, also das Produkt der Größen aller fünf Indexbereiche.

<!--en:1e54b0a7-->
Auch außerhalb des Codes ermöglicht die Indexnotation kurze Beweise in einer Zeile. Für beliebige $\mathbf{A} \in \mathbb{R}^{m \times n}$ und $\mathbf{B} \in \mathbb{R}^{n \times m}$ sind sowohl $\mathbf{A}\mathbf{B}$ als auch $\mathbf{B}\mathbf{A}$ quadratisch. Für ihre Spuren gilt

<!--en:bd2c4877-->
Im mittleren Schritt vertauschen wir nur zwei Skalare, die Indizes bleiben unverändert. Überraschenderweise hängt die Spur eines Produkts nicht von der Reihenfolge der beiden Faktoren ab. Das gilt sogar dann, wenn $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ gilt und die beiden Produkte unterschiedliche Dimensionen haben. Eine unmittelbare Folgerung brauchen wir später für die Eigenwerte. Hat $\mathbf{V}$ eine Inverse $\mathbf{V}^{-1}$ (:numref:`subsec_mdl-inverses`), so gilt $\operatorname{tr}(\mathbf{V}\mathbf{A}\mathbf{V}^{-1}) =
\operatorname{tr}(\mathbf{A}\mathbf{V}^{-1}\mathbf{V}) =
\operatorname{tr}(\mathbf{A})$. Da $\mathbf{V}\mathbf{A}\mathbf{V}^{-1}$ dieselbe lineare Abbildung $\mathbf{A}$ in einem anderen Koordinatensystem darstellt, ist die Spur eine Eigenschaft der Abbildung selbst und unabhängig von der Wahl der Koordinaten. Auch die Determinante besitzt diese Invarianz (:numref:`subsec_mdl-multiplicativity`).

<!--en:33c60a7f-->
Für die Praxis ist eines zu beachten. Verknüpft ein `einsum`-Ausdruck drei oder mehr Tensoren, legt die Zeichenkette zwar das mathematische Ergebnis fest, aber nicht die Reihenfolge der paarweisen Kontraktionen. Diese Reihenfolge kann die Rechengeschwindigkeit erheblich beeinflussen. Bei $n \times n$-Matrizen benötigt die Berechnung von $(\mathbf{A}\mathbf{B})\mathbf{v}$ etwa $n^3$ Multiplikationsadditionen, die von $\mathbf{A}(\mathbf{B}\mathbf{v})$ dagegen nur $2n^2$.

<!--en:b62cae67-->
## Rang und Invertierbarkeit

<!--en:81442481-->
### Spaltenraum und Nullraum

<!--en:6520a88f-->
Welche Ausgaben kann eine Matrix überhaupt erzeugen, und welche Eingaben bildet sie auf null ab? Beides sehen wir an unserem singulären Beispiel

<!--en:8ddfbd47-->
Die Spalten sind $\mathbf{b}_1 = (2, 4)$ und $\mathbf{b}_2 = (1, 2)$. Die eine ist also das Doppelte der anderen, $\mathbf{b}_1 = 2\mathbf{b}_2$. Da jede Ausgabe von $\mathbf{B}$ eine gewichtete Summe der Spalten ist, ergibt sich stets ein skalares Vielfaches von $\mathbf{b}_2$:

<!--en:dfd3a224-->
Damit bildet $\mathbf{B}$ die gesamte zweidimensionale Ebene auf die einzige Gerade $y = 2x$ durch $(1, 2)$ ab. Auf dieser Geraden fallen zwangsläufig Bilder verschiedener Eingaben zusammen. Sowohl $(1, 1)$ als auch $(0, 3)$ gehen in $(3, 6)$ über. Wegen der Linearität muss die Differenz $(1, -2)$ auf die Differenz der Ausgaben abgebildet werden, also auf $\mathbf{B}(1, -2) = \mathbf{0}$. Das lässt sich auch an den Zeilen erkennen. Beide Zeilen von $\mathbf{B}$, nämlich $(2, 1)$ und $(4, 2)$, stehen senkrecht auf $(1, -2)$. Daher liefern beide Hyperebenentests null. :numref:`fig_mdl-la-null-collapse` zeigt die beiden Geraden nebeneinander.

<!--en:cffb5146-->
![Die Matrix $\mathbf{B} = \left(\begin{smallmatrix}2 & 1\\ 4 & 2\end{smallmatrix}\right)$ bildet die Ebene auf eine Gerade ab. (a) In der Eingabeebene steht die gestrichelte Nullraumgerade durch $(1, -2)$ senkrecht auf der Zeilenrichtung $(2, 1)$. Markiert sind drei Eingaben, $\mathbf{x}_1 = (1, 1)$, $\mathbf{x}_2 = (0, 3)$ und der Punkt $\mathbf{x}_3 = (1, -2)$ auf der Nullraumgeraden. (b) In der Ausgabeebene (gleicher Maßstab) liegen alle Ausgaben im Spaltenraum, auf der Geraden $y = 2x$, die $\mathbf{b}_1$ und $\mathbf{b}_2$ aufspannen. Sowohl $\mathbf{x}_1$ als auch $\mathbf{x}_2$ gehen in $(3, 6)$ über, während $\mathbf{x}_3$ auf den Ursprung fällt. Da verschiedene Eingaben dasselbe Bild haben, kann es keine Inverse geben.](../img/mdl-la-null-collapse.svg)

<!--en:b18cd841-->
Dieses Beispiel veranschaulicht zwei grundlegende Unterräume, die zu jeder $m \times n$-Matrix $\mathbf{A}$ gehören. Der *Spaltenraum*, geschrieben $\operatorname{col}\mathbf{A}$, ist der von den Spalten von $\mathbf{A}$ aufgespannte Raum. Da $\mathbf{A}\mathbf{x}$ eine gewichtete Kombination der Spalten mit den Gewichten $x_1, \ldots, x_n$ ist, umfasst der Spaltenraum genau alle erreichbaren Ausgaben. Der *Nullraum*, auch *Kern* genannt und als $\ker\mathbf{A}$ geschrieben, besteht aus allen Eingaben, die $\mathbf{A}$ auf null abbildet:

<!--en:8e1d0ad9-->
Er ist ein Unterraum von $\mathbb{R}^n$, denn aus $\mathbf{A}\mathbf{x} = \mathbf{A}\mathbf{y} = \mathbf{0}$ folgt $\mathbf{A}(\alpha\mathbf{x} + \beta\mathbf{y}) = \alpha\mathbf{A}\mathbf{x} +
\beta\mathbf{A}\mathbf{y} = \mathbf{0}$. Bei $\mathbf{B}$ ist der Spaltenraum die Gerade $y = 2x$ in der Ausgabeebene und der Nullraum die Ursprungsgerade durch $(1, -2)$ in der Eingabeebene. (Beide liegen nur deshalb in $\mathbb{R}^2$, weil $\mathbf{B}$ quadratisch ist. Im Allgemeinen gelten $\ker\mathbf{A} \subseteq
\mathbb{R}^n$ und $\operatorname{col}\mathbf{A} \subseteq \mathbb{R}^m$.)

<!--en:b1f0e692-->
Aus diesem Beispiel lassen sich drei allgemeine Aussagen ableiten:

<!--en:66a70d10-->
1. *Der Nullraum besteht aus allen Eingaben, die senkrecht auf jeder Zeile stehen.* Zeilenweise betrachtet ist $(\mathbf{A}\mathbf{x})_i$ das Skalarprodukt der Zeile $i$ mit $\mathbf{x}$. Alle Einträge von $\mathbf{A}\mathbf{x}$ verschwinden also genau dann, wenn $\mathbf{x}$ zu jeder Zeile orthogonal ist.

<!--en:1dd63d5f-->
2. *Die Spalten sind genau dann linear abhängig, wenn der Nullraum einen Vektor ungleich null enthält.* Eine nichttriviale Abhängigkeit $\sum_j x_j\mathbf{a}_j = \mathbf{0}$ zwischen den Spalten $\mathbf{a}_j$ besagt gerade, dass $\mathbf{A}\mathbf{x} = \mathbf{0}$ für einen Vektor $\mathbf{x}$ ungleich null gilt. Umgekehrt folgt aus der Gleichung die Abhängigkeit. Für $\mathbf{B}$ liefert die Beziehung $\mathbf{b}_1 - 2\mathbf{b}_2 = \mathbf{0}$ den Nullraumvektor $(1, -2)$.

<!--en:26344982-->
3. *Zwei Eingaben ergeben genau dann dieselbe Ausgabe, wenn ihre Differenz im Nullraum liegt.* Tatsächlich ist $\mathbf{A}\mathbf{x} = \mathbf{A}\mathbf{y}$ gleichbedeutend mit $\mathbf{A}(\mathbf{x} - \mathbf{y}) = \mathbf{0}$. Alle Eingaben, die $\mathbf{B}$ auf $(3, 6)$ abbildet, bilden die verschobene Gerade $(1, 1) + t(1, -2)$. Sie verläuft durch $(0, 3)$ bei $t = -1$. Sobald $\ker\mathbf{A}$ einen Vektor ungleich null enthält, haben verschiedene Eingaben dasselbe Bild und es kann keine Inverse geben.

<!--en:95d5b50a-->
### Rang

<!--en:d2570392-->
Wie viele Eingabedimensionen bleiben unter einer linearen Abbildung erhalten? Der *Rang* einer Matrix ist die Dimension ihres Spaltenraums oder, gleichbedeutend damit, die maximale Anzahl linear unabhängiger Spalten:

<!--en:af8230d1-->
Unsere Matrix $\mathbf{B}$ hat Rang $1$. Ihre beiden Spalten sind abhängig und beide sind ungleich null. Betrachten wir als etwas umfangreicheres Beispiel die $4 \times 5$-Matrix

<!--en:83f02ff8-->
Sie hat Rang $2$. Ihre ersten beiden Spalten sind linear unabhängig, doch jedes der $\binom{5}{3} = 10$ möglichen Spaltentripel ist linear abhängig.

<!--en:6ba10b7a-->
Bemerkenswerterweise ergeben die Anzahl der erhaltenen Dimensionen und die Anzahl der auf null abgebildeten Dimensionen zusammen stets die Dimension des Eingaberaums.

<!--en:dba456de-->
**Proposition (Dimensionssatz).** *Für eine $m \times n$-Matrix $\mathbf{A}$ gilt*

<!--en:8a3871ca-->
**Beweis.** Wählen wir eine Basis $\mathbf{w}_1, \ldots, \mathbf{w}_k$ von $\ker\mathbf{A}$ und ergänzen wir sie zu einer Basis $\mathbf{w}_1, \ldots,
\mathbf{w}_k, \mathbf{u}_1, \ldots, \mathbf{u}_{n-k}$ von $\mathbb{R}^n$. (Wir verwenden ohne Beweis zwei bekannte Tatsachen. Jede linear unabhängige Familie lässt sich zu einer Basis ergänzen, und alle Basen eines Unterraums haben gleich viele Elemente.) Die Bilder $\mathbf{A}\mathbf{u}_1, \ldots,
\mathbf{A}\mathbf{u}_{n-k}$ spannen $\operatorname{col}\mathbf{A}$ auf, denn jede Eingabe $\mathbf{x} \in \mathbb{R}^n$ ist eine Linearkombination der gesamten Basis und die $\mathbf{w}_j$ verschwinden unter $\mathbf{A}$. Die Bilder sind auch linear unabhängig. Gilt $\sum_i c_i \mathbf{A}\mathbf{u}_i = \mathbf{0}$, so liegt $\sum_i c_i \mathbf{u}_i$ in $\ker\mathbf{A}$ und lässt sich daher als Linearkombination der $\mathbf{w}_1, \ldots, \mathbf{w}_k$ schreiben. Innerhalb einer Basis ist das nur möglich, wenn für alle Koeffizienten $c_i = 0$ gilt. Daher hat $\operatorname{col}\mathbf{A}$ die Dimension $n - k$. $\blacksquare$

<!--en:62aaeace-->
:numref:`fig_mdl-la-rank-nullity` zeigt die im Beweis konstruierte Basis. Jeder ihrer $n$ Vektoren verschwindet entweder unter $\mathbf{A}$ oder trägt eine Dimension zum Spaltenraum bei. Kein Vektor tut beides.

<!--en:b3dd1291-->
![Die Basis im Beweis des Dimensionssatzes. Die $k$ Vektoren $\mathbf{w}_1, \ldots, \mathbf{w}_k$ spannen $\ker\mathbf{A}$ auf und werden auf $\mathbf{0}$ abgebildet. Die $n-k$ Vektoren $\mathbf{u}_1, \ldots, \mathbf{u}_{n-k}$ ergänzen sie zu einer Basis von $\mathbb{R}^n$. Ihre Bilder bilden eine Basis von $\operatorname{col}\mathbf{A}$.](../img/mdl-la-rank-nullity.svg)

<!--en:dbe21616-->
Prüfen wir die Erhaltung der Dimension an unseren Beispielen. Für $\mathbf{B}$ erhalten wir $1 + 1 = 2$. Bei der eingangs betrachteten $256 \times 784$-Schicht für Fashion-MNIST ist der Rang höchstens so groß wie die Anzahl $256$ der Ausgabekoordinaten. Der Nullraum hat also mindestens die Dimension $784 - 256 = 528$. Bei $\mathbf{C}$ muss der Nullraum die Dimension $5 - 2 = 3$ haben. Je drei linear unabhängige Nullraumvektoren bilden daher eine Basis von $\ker\mathbf{C}$. Im Folgenden prüfen wir, dass $(3, -1, 3, 0, 0)$, $(1, 0, 0, 1, 0)$ und $(0, 0, 1, 0, 1)$ unter $\mathbf{C}$ alle auf null abgebildet werden. Anschließend prüfen wir den Rang der Matrix `Z`, deren Spalten diese Vektoren sind.

<!--en:fd672a89-->
Für die Matrizen $\mathbf{B}$ und $\mathbf{C}$ erhalten wir die Ränge $1$ und $2$. Der größte Eintrag von $|\mathbf{C}\mathbf{Z}|$ ist null, und die drei Spalten von `Z` haben Rang $3$. Das bestätigt $2 + 3 = 5$.

<!--en:f24b82e8-->
Was ergibt sich, wenn wir eine Matrix zeilenweise statt spaltenweise lesen? Ihr *Zeilenrang*, die maximale Anzahl linear unabhängiger Zeilen, gibt die Anzahl unabhängiger Hyperebenentests an, die sie ausführt. In $\mathbf{C}$ ist Zeile 3 die Summe der Zeilen 1 und 2, Zeile 4 die Differenz von Zeile 1 und Zeile 2. Der Zeilenrang ist also $2$ und stimmt mit dem Spaltenrang überein. Diese Übereinstimmung gilt allgemein.

<!--en:4459dcaf-->
**Proposition (Zeilenrang gleich Spaltenrang).** *Für jede Matrix $\mathbf{A}$ ist die maximale Anzahl linear unabhängiger Zeilen gleich der maximalen Anzahl linear unabhängiger Spalten.*

<!--en:bb6e9dc3-->
**Beweis.** Sei $r = \operatorname{rank}\mathbf{A}$. Schreiben wir die Vektoren einer Basis von $\operatorname{col}\mathbf{A}$ in die Spalten einer $m \times r$-Matrix $\mathbf{G}$. Jede Spalte von $\mathbf{A}$ ist eine Linearkombination der Spalten von $\mathbf{G}$. Schreiben wir die Koeffizienten für die Spalte $j$ von $\mathbf{A}$ in die Spalte $j$ einer $r \times n$-Matrix $\mathbf{H}$, so erhalten wir die Faktorisierung

<!--en:06c0e30f-->
Lesen wir dasselbe Produkt nun zeilenweise. Die Zeile $i$ von $\mathbf{A}$ ist $\sum_k g_{ik}\,(\textrm{row } k \textrm{ of } \mathbf{H})$. Also liegt jede Zeile von $\mathbf{A}$ in dem Raum, den die $r$ Zeilen von $\mathbf{H}$ aufspannen. Der Zeilenrang von $\mathbf{A}$ beträgt daher höchstens $r$. Wiederholen wir das Argument für $\mathbf{A}^\top$, deren Spalten die Zeilen von $\mathbf{A}$ sind, folgt, dass der Spaltenrang höchstens gleich dem Zeilenrang ist. Somit sind beide gleich. $\blacksquare$

<!--en:a38c1a44-->
Die Faktorisierung $\mathbf{A} = \mathbf{G}\mathbf{H}$ dient nicht nur dem Beweis. Sie zeigt, dass jede Matrix vom Rang $r$ ihre Eingabe durch einen Flaschenhals der Breite $r$ leitet. Zunächst komprimiert $\mathbf{H}$ den Vektor $\mathbf{x} \in \mathbb{R}^n$ auf $r$ Zahlen. Anschließend erweitert $\mathbf{G}$ diese $r$ Zahlen wieder zu einem Vektor in $\mathbb{R}^m$. Ist $r$ wesentlich kleiner als $m$ und $n$, benötigen wir zum Speichern von $\mathbf{G}$ und $\mathbf{H}$ nur $r(m + n)$ Parameter statt $mn$. Das ist die Grundidee der Aktualisierungen mit niedrigem Rang in :numref:`subsec_mdl-low-rank-updates`.

<!--en:1e9b5d6d-->
In der Praxis würde es exponentiell lange dauern, den Rang durch das Testen von Spaltenteilmengen zu bestimmen. Von Hand verwendet man die Gauß-Elimination, die wir in diesem Kapitel nicht behandeln. Numerikbibliotheken zählen beispielsweise mit `np.linalg.matrix_rank`, wie viele *Singulärwerte* (:numref:`sec_mdl-svd-low-rank`) eine kleine Toleranzschwelle überschreiten. Eine Toleranz ist unumgänglich, denn der Rang hängt nicht stetig von den Matrixeinträgen ab. Schon eine winzige Störung kann einer rangdefizienten Matrix vollen Rang verleihen (:numref:`subsec_mdl-det-limits`).

<!--en:ac9be38f-->
### Inverse Matrizen

<!--en:4a94a451-->
Wann können wir aus einer gegebenen Ausgabe $\mathbf{A}\mathbf{x}$ zuverlässig die Eingabe $\mathbf{x}$ rekonstruieren? Eine *Inverse* einer quadratischen Matrix $\mathbf{A}$ ist eine Matrix $\mathbf{A}^{-1}$, die die Wirkung von $\mathbf{A}$ auf beiden Seiten aufhebt:

<!--en:15deea9f-->
Damit gilt $\mathbf{x} = \mathbf{A}^{-1}(\mathbf{A}\mathbf{x})$ für jede Eingabe $\mathbf{x}$. Für eine $2 \times 2$-Matrix mit den Einträgen $p, q, r, s$ und der Eigenschaft $ps - qr \neq 0$ gibt es eine explizite Formel:

<!--en:cd540cf2-->
Das lässt sich durch Nachrechnen bestätigen. Multiplizieren wir $\left(\begin{smallmatrix}s & -q\\ -r & p\end{smallmatrix}\right)$ mit $\mathbf{A}$, erhalten wir in beiden Reihenfolgen $(ps - qr)\,\mathbf{I}$. Nach Division durch $ps - qr$ bleibt $\mathbf{I}$. Die Bedingung $ps - qr \neq 0$ ist unverzichtbar. Gilt $ps - qr = 0$, so ist das Produkt die Nullmatrix. Die Multiplikation beider Seiten mit einer angenommenen Inversen $\mathbf{A}^{-1}$ würde $\left(\begin{smallmatrix}s & -q\\ -r & p\end{smallmatrix}\right) =
\mathbf{0}$ erzwingen. Daraus folgte $\mathbf{A} = \mathbf{0}$, und die Nullmatrix hat offensichtlich keine Inverse. Der Größe $ps - qr$ begegnen wir in :numref:`subsec_mdl-parallelogram-area` als orientiertem Flächeninhalt wieder. Für unsere Beispielmatrix gilt $ps - qr = 3 + 2 = 5$, also

<!--en:dae5ae27-->
Tatsächlich kehrt $\mathbf{A}^{-1}$ die Rechnung aus :numref:`subsec_mdl-column-view` um.

<!--en:6affe084-->
Existiert eine Inverse, so ist sie eindeutig. Es gilt sogar, dass jede Linksinverse und jede Rechtsinverse derselben Matrix übereinstimmen. Gelten $\mathbf{L}\mathbf{A} = \mathbf{I}$ und $\mathbf{A}\mathbf{X} = \mathbf{I}$, so folgt aus der Assoziativität

<!--en:5f4903b1-->
Wann existiert im Allgemeinen eine Inverse? Für eine quadratische Matrix genügen der Nullraum und der Dimensionssatz, um die Frage zu beantworten. Auf dieses Ergebnis stützen wir uns im weiteren Kapitel häufiger als auf jedes andere dieses Abschnitts.

<!--en:4dbfadfc-->
**Proposition (invertierbare Matrizen).** *Für eine $n \times n$-Matrix $\mathbf{A}$ sind folgende Aussagen äquivalent:* (i) *$\mathbf{A}$ ist invertierbar.* (ii) *Die Spalten von $\mathbf{A}$ sind linear unabhängig.* (iii) *$\ker\mathbf{A} = \{\mathbf{0}\}$.* (iv) *$\operatorname{rank}\mathbf{A} = n$.*

<!--en:080621ee-->
**Beweis.** Die Äquivalenz von (ii) und (iii) folgt aus Aussage 2 in :numref:`subsec_mdl-column-null-space`, die von (iii) und (iv) aus dem Dimensionssatz. Ist $\mathbf{A}$ invertierbar und gilt $\mathbf{A}\mathbf{x} =
\mathbf{0}$, so ergibt die Multiplikation von links mit $\mathbf{A}^{-1}$ die Gleichung $\mathbf{x} = \mathbf{A}^{-1}\mathbf{A}\mathbf{x} = \mathbf{0}$. Also folgt (iii) aus (i).

<!--en:57fd92da-->
Die Umkehrung erfordert mehr Arbeit, da wir die Inverse konstruieren müssen. Nach (iv) ist der Spaltenraum ein $n$-dimensionaler Unterraum von $\mathbb{R}^n$ und muss daher mit ganz $\mathbb{R}^n$ übereinstimmen. Andernfalls könnten wir eine Basis von $\operatorname{col}\mathbf{A}$ um einen Vektor außerhalb des Spaltenraums ergänzen. Das ergäbe $n + 1$ linear unabhängige Vektoren in $\mathbb{R}^n$, was unmöglich ist. Somit erreichen wir jeden Zielvektor in $\mathbb{R}^n$. Nach (iii) und Aussage 3 haben keine zwei verschiedenen Eingaben dieselbe Ausgabe. Wir können daher als Spalte $j$ einer Matrix $\mathbf{X}$ die eindeutige Lösung von $\mathbf{A}\mathbf{x} =
\mathbf{e}_j$ wählen. Nach Konstruktion gilt $\mathbf{A}\mathbf{X} = \mathbf{I}$. Um zu sehen, dass $\mathbf{X}$ auch von links wirkt, betrachten wir ein beliebiges $\mathbf{x} \in
\mathbb{R}^n$. Da $\mathbf{A}(\mathbf{X}\mathbf{A}\mathbf{x}) =
(\mathbf{A}\mathbf{X})\mathbf{A}\mathbf{x} = \mathbf{A}\mathbf{x}$ gilt und verschiedene Eingaben verschiedene Bilder haben, folgt $\mathbf{X}\mathbf{A}\mathbf{x} = \mathbf{x}$. Somit gilt auch $\mathbf{X}\mathbf{A} = \mathbf{I}$, und wir erhalten $\mathbf{X} =
\mathbf{A}^{-1}$. $\blacksquare$

<!--en:1a3131cd-->
Damit ist die offene Behauptung aus :numref:`subsec_mdl-orthogonal-matrices` bewiesen. Betrachten wir eine quadratische orthogonale Matrix $\mathbf{Q}$, für die also $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ gilt. Aus $\mathbf{Q}\mathbf{c} = \mathbf{0}$ folgt $\mathbf{c} = \mathbf{Q}^\top\mathbf{Q}\mathbf{c} = \mathbf{0}$ und damit $\ker\mathbf{Q} = \{\mathbf{0}\}$. Nach dem Satz ist $\mathbf{Q}$ invertierbar. Ihre Linksinverse $\mathbf{Q}^\top$ muss mit $\mathbf{Q}^{-1}$ übereinstimmen. Daher gilt $\mathbf{Q}\mathbf{Q}^\top = \mathbf{I}$.

<!--en:80f54214-->
Eine quadratische Matrix ohne Inverse heißt *singulär*. Unser Standardbeispiel ist $\mathbf{B}$. Nichtquadratische Matrizen können grundsätzlich keine beidseitige Inverse haben. Der Rang einer $m \times n$-Matrix beträgt höchstens $\min(m, n)$. Für $m > n$ hat der Spaltenraum höchstens die Dimension $n < m$. Die Abbildung ist also nicht *surjektiv*, denn manche Zielvektoren in $\mathbb{R}^m$ sind unerreichbar. Für $n > m$ erzwingt der Dimensionssatz einen Nullraum der Dimension mindestens $n - m > 0$. Die Abbildung ist also nicht *injektiv*, denn verschiedene Eingaben haben dasselbe Bild. Für die Umkehrbarkeit brauchen wir beide Eigenschaften. Das Besondere am obigen Satz ist, dass bei quadratischen Matrizen jede der beiden Eigenschaften die andere garantiert.

<!--en:80ba2e75-->
### Lösung linearer Gleichungssysteme in der Praxis

<!--en:93ac79a0-->
Die Inverse ist für algebraische Rechnungen und Beweise unentbehrlich. Bei numerischen Rechnungen bilden wir sie jedoch fast nie explizit. Um $\mathbf{A}\mathbf{x} = \mathbf{b}$ zu lösen, verwenden Sie `np.linalg.solve(A, b)` anstelle von `np.linalg.inv(A) @ b`.

<!--en:afc834f5-->
Dafür gibt es zwei Hauptgründe, Geschwindigkeit und Genauigkeit. Erstens zerlegt `solve` die Matrix $\mathbf{A}$ einmal in Dreiecksmatrizen (eine *LU-Zerlegung*, die etwa $n^3/3$ Multiplikationsadditionen kostet) und löst anschließend zwei Dreieckssysteme mit geringem Aufwand. Die explizite Berechnung von $\mathbf{A}^{-1}$ kostet etwa $n^3$ Multiplikationsadditionen, also dreimal so viel Arbeit, noch vor der Multiplikation mit $\mathbf{b}$. Zweitens ist `solve` *rückwärtsstabil*. Die berechnete Lösung löst ein leicht gestörtes System exakt, dessen Einträge nur im Bereich der Gleitkommarundung von $\mathbf{A}$ und $\mathbf{b}$ abweichen. Das Residuum $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|$ bleibt daher in der Größenordnung der Maschinengenauigkeit multipliziert mit $\|\mathbf{A}\|\,\|\mathbf{x}\|$ (mit der Spektralnorm aus :numref:`subsec_lin-algebra-norms`). Für die Berechnung von `inv(A) @ b` gibt es keine solche Garantie. Das Residuum kann um einen Faktor bis zur *Konditionszahl* $\kappa(\mathbf{A})$ größer sein. Diese misst, wie nahe $\mathbf{A}$ der Singularität kommt (:numref:`subsec_mdl-condition-number`).

<!--en:e22b1ab1-->
Wie nahe eine Matrix der Singularität kommt, ist eine Eigenschaft des Problems selbst und unabhängig vom Algorithmus. Bei doppelter Genauigkeit (etwa sechzehn Dezimalstellen) können beide Methoden bis zu etwa $\log_{10}\kappa(\mathbf{A})$ Stellen Genauigkeit in der Lösung $\mathbf{x}$ verlieren. Das ist eine Schranke für den ungünstigsten Fall, keine genaue Vorhersage für eine einzelne rechte Seite. Sehen wir uns an, wie nahe ein berüchtigtes Beispiel dieser Schranke kommt. Die *Hilbert-Matrix* mit den Einträgen $h_{ij} = 1/(i + j - 1)$ ist ein klassischer Testfall für eine nahezu singuläre Matrix. Wir wählen $\mathbf{x}_{\textrm{true}} = \mathbf{1}$, bilden $\mathbf{b} = \mathbf{H}\mathbf{1}$ für die $10 \times 10$-Hilbert-Matrix $\mathbf{H}$ und versuchen, $\mathbf{x}$ auf beiden Wegen zu rekonstruieren.

<!--en:38fe2c05-->
Die Konditionszahl beträgt $\kappa(\mathbf{H}) \approx 1.6 \cdot 10^{13}$. Wir müssen also mit dem Verlust von bis zu dreizehn Stellen rechnen. Wie erwartet hält `solve` das Residuum nahe der Maschinengenauigkeit. Bei `inv` ist es dagegen rund elf Größenordnungen größer, im Einklang mit dem Faktor $\kappa$. Ein winziges Residuum garantiert jedoch keine genaue Lösung. Auf beiden Wegen gehen mindestens zwölf der sechzehn Stellen verloren, fast so viele wie im ungünstigsten Fall. Dabei schneidet `inv` noch um mehr als eine Größenordnung schlechter ab.

<!--en:54f467c6-->
Bei dünn besetzten Matrizen kommt ein dritter, entscheidender Grund hinzu, die Inverse nicht zu bilden. Eine Matrix mit je einer Million Zeilen und Spalten und nur fünf Millionen Einträgen ungleich null passt bequem in den Speicher. Ihre Inverse ist dagegen meist vollständig dicht besetzt und benötigt rund $10^{12}$ Einträge.

<!--en:9bb07281-->
## Die Determinante

<!--en:5ebbc395-->
### Flächeninhalt eines Parallelogramms

<!--en:f15aecbb-->
Mit welchem Faktor skaliert eine Matrix Flächeninhalte? Betrachten wir noch einmal :numref:`fig_mdl-la-linear-map`. Unsere Beispielmatrix $\mathbf{A}$ bildet das Einheitsquadrat (Kanten $\mathbf{e}_1$ und $\mathbf{e}_2$, Flächeninhalt $1$) auf das Parallelogramm mit den Kanten $\mathbf{a}_1 = (1, -1)$ und $\mathbf{a}_2 = (2, 3)$ und den Ecken $(0, 0)$, $(1, -1)$, $(3, 2)$ und $(2, 3)$ ab. Dieses Parallelogramm passt genau in das umschließende Rechteck $[0, 3] \times [-1, 3]$ mit Flächeninhalt $12$. Der übrige Teil des Rechtecks besteht aus vier rechtwinkligen Dreiecken mit den Katheten $1$ und $1$, $2$ und $3$, $1$ und $1$ sowie $2$ und $3$. Ihre Flächeninhalte ergeben zusammen $\tfrac12 + 3 + \tfrac12 + 3 = 7$. Ziehen wir diese Summe ab, bleibt ein Flächeninhalt von $12 - 7 = 5$ (:numref:`fig_mdl-la-parallelogram-area`, Teilbild (a)). Die Matrix $\mathbf{A}$ multipliziert den Flächeninhalt des Einheitsquadrats mit $5$.

<!--en:129271ae-->
![Das Bild des Einheitsquadrats unter $\mathbf{A}$ hat den Flächeninhalt $5$. (a) Umschließendes Rechteck $[0, 3] \times [-1, 3]$ mit Flächeninhalt $12$ abzüglich vier rechtwinkliger Dreiecke mit Gesamtfläche $7$. (b) Grundseite mal Höhe. Die Grundseite $\mathbf{a}_1 = (1, -1)$ hat die Länge $\sqrt{2}$. Der Einheitsnormalenvektor $\mathbf{n} = (1, 1)/\sqrt{2}$ steht senkrecht auf ihr. Die Höhe von $\mathbf{a}_2 = (2, 3)$ über der Grundlinie beträgt $\mathbf{a}_2\cdot\mathbf{n} = 5/\sqrt{2}$.](../img/mdl-la-parallelogram-area.svg)

<!--en:c6b28fb4-->
Für diese eine Zeichnung können wir die Dreiecke ausschneiden. Jede neue Anordnung der Ecken würde jedoch eine neue Zeichnung erfordern. Die Formel „Grundseite mal Höhe“ erfasst alle Fälle zugleich. Die Kanten des Parallelogramms seien $\mathbf{u} = (p, r)$ und $\mathbf{w} = (q, s)$, die beiden Spalten einer allgemeinen $2 \times 2$-Matrix. Die Grundseite hat die Länge $\|\mathbf{u}\|$. Drehen wir $\mathbf{u}$ gegen den Uhrzeigersinn um $90^\circ$, erhalten wir $\mathbf{R}_{90}\mathbf{u} = (-r, p)$. Damit ist

<!--en:a641a54d-->
ein Einheitsvektor senkrecht zur Grundseite. Aus :numref:`sec_mdl-geometry-linear-algebraic-ops` wissen wir, dass die vorzeichenbehaftete Komponente von $\mathbf{w}$ in Richtung eines Einheitsvektors $\mathbf{n}$ gleich dem Skalarprodukt $\mathbf{w}\cdot\mathbf{n}$ ist. Geometrisch ist dies die orientierte Höhe des Parallelogramms über seiner Grundseite. Sie ist positiv, wenn $\mathbf{w}$ von $\mathbf{u}$ aus gegen den Uhrzeigersinn liegt, und negativ, wenn die Kante im Uhrzeigersinn liegt. Multiplizieren wir die Länge der Grundseite mit der orientierten Höhe, ergibt sich

<!--en:1922899e-->
Die Längen $\|\mathbf{u}\|$ kürzen sich heraus. Übrig bleibt ein Polynom in den vier Matrixeinträgen, das ohne Fallunterscheidung nach dem Vorzeichen auskommt. Für unsere Beispielmatrix gelten $\|\mathbf{u}\| = \sqrt{2}$, $\mathbf{n} = (1, 1)/\sqrt{2}$ und $\mathbf{w}\cdot\mathbf{n} = 5/\sqrt{2}$. Das ergibt den orientierten Flächeninhalt $5$ (Teilbild (b)). Er ist positiv, da $\mathbf{a}_2$ von $\mathbf{a}_1$ aus gegen den Uhrzeigersinn liegt.

<!--en:de3a2cca-->
Wir definieren die *Determinante* einer $2 \times 2$-Matrix als den orientierten Flächeninhalt des Bildes des Einheitsquadrats:

<!--en:0d456609-->
Ihr Betrag $|\det\mathbf{A}|$ ist der Faktor, mit dem die Abbildung Flächeninhalte skaliert. Ihr Vorzeichen gibt an, ob die Bilder von $\mathbf{e}_1$ und $\mathbf{e}_2$ ihre Orientierung gegen den Uhrzeigersinn beibehalten. Für $\mathbf{A}$ ergibt sich $1 \cdot 3 - 2 \cdot (-1) = 5$. Vertauschen wir die beiden Spalten von $\mathbf{A}$, erhalten wir $\left(\begin{smallmatrix}2 & 1\\ 3 & -1\end{smallmatrix}\right)$. Diese Matrix spannt dasselbe Parallelogramm mit umgekehrter Kantenreihenfolge auf. Ihre Determinante wechselt das Vorzeichen und beträgt $-2 - 3 = -5$. Für die singuläre Matrix $\mathbf{B}$ erhalten wir $2 \cdot 2 - 1 \cdot 4 = 0$. Das Einheitsquadrat fällt auf die Strecke von $\mathbf{0}$ nach $(3, 6)$ zusammen, deren Flächeninhalt null ist.

<!--en:848ef966-->
:numref:`fig_mdl-la-determinant` zeigt die drei Fälle im Vergleich. Bei positiver Determinante bleibt die Orientierung erhalten, bei negativer kehrt sie sich um, und bei Determinante null fällt das Quadrat zusammen.

<!--en:4a9f4d35-->
![Die Determinante als orientierter Flächeninhalt für unsere drei $2 \times 2$-Beispiele. Das gestrichelte Einheitsquadrat geht in das von den Spalten aufgespannte Gebiet über. (a) $\mathbf{A}$ mit Determinante $5$. Die Spalten behalten ihre Reihenfolge gegen den Uhrzeigersinn bei (gebogener Pfeil). (b) $\mathbf{A}$ mit vertauschten Spalten und Determinante $-5$. Das Parallelogramm bleibt gleich, die Orientierung kehrt sich um. (c) $\mathbf{B}$ mit Determinante $0$. Das Quadrat fällt auf einen Abschnitt der Geraden $y = 2x$ zusammen. Der Maßstab in (c) ist halb so groß wie in (a) und (b).](../img/mdl-la-determinant.svg)

<!--en:17e36f2d-->
Mit „Grundseite mal Höhe“ erfassen wir die Ebene. Für den Übergang zu $n$ Dimensionen brauchen wir das orientierte Volumen eines $n$-dimensionalen Parallelotops.

<!--en:19deaac8-->
### Die allgemeine Determinante

<!--en:50bf9252-->
Wie definieren wir das orientierte Volumen des Parallelotops, das $n$ Spalten in $\mathbb{R}^n$ aufspannen? Statt hochdimensionale Körper zur Volumenberechnung zu zerlegen, wählen wir einen direkteren Weg. Wir formulieren drei geometrische Eigenschaften, die jedes orientierte Volumen erfüllen muss. Diese Eigenschaften bestimmen die Formel eindeutig. Als Funktion der Spalten $\mathbf{a}_1, \ldots, \mathbf{a}_n$ einer $n \times n$-Matrix hat die Determinante folgende Eigenschaften:

<!--en:01dbaee4-->
1. **Multilinear** bedeutet linear in jeder einzelnen Spalte, wenn wir die übrigen Spalten festhalten. (Skalieren wir eine Kante des Parallelotops, skaliert sein Volumen mit demselben Faktor. Schreiben wir eine Kante als Summe, zerfällt das Volumen entsprechend.)

<!--en:e03a4d28-->
2. **Alternierend** bedeutet, dass die Determinante null ist, sobald zwei Spalten übereinstimmen. (Fallen zwei Kanten zusammen, verflacht das Parallelotop zu einem Körper kleinerer Dimension mit Volumen null.)

<!--en:956b805b-->
3. **Normiert** bedeutet $\det \mathbf{I} = 1$. (Der Einheitswürfel hat das Volumen $1$.)

<!--en:b6e0b864-->
Prüfen Sie, dass unsere $2 \times 2$-Formel $ps - qr$ alle drei Eigenschaften erfüllt. Sie ist linear in $(p, r)$ und in $(q, s)$, verschwindet für $(p, r) = (q, s)$ und nimmt den Wert $1$ auf $\mathbf{I}$ an. Allein aus diesen drei Axiomen folgen unmittelbar zwei wichtige Ergebnisse. Wir erhalten eine Regel für Spaltenvertauschungen und eine explizite Formel, die zeigt, dass höchstens eine Funktion alle drei Axiome erfüllen kann.

<!--en:e81896dc-->
Erstens **kehrt jede Vertauschung zweier Spalten das Vorzeichen der Determinante um**. Setzen wir dazu $\mathbf{u} + \mathbf{w}$ an zwei Spaltenpositionen ein und entwickeln wir mithilfe der Multilinearität:

<!--en:fb1d0fad-->
Die linke Seite, der erste und der letzte Term haben jeweils zwei gleiche Spalten. Da die Determinante alternierend ist, verschwinden alle drei. Übrig bleibt

<!--en:1985a9c6-->
Vertauschen wir zwei Kanten des Parallelotops, kehrt sich seine Orientierung um, wie in Teilbild (b) von :numref:`fig_mdl-la-determinant`.

<!--en:d0745cfc-->
Zweitens legen die drei Eigenschaften die Formel vollständig fest. Entwickeln wir jede Spalte in der Standardbasis als $\mathbf{a}_j = \sum_i a_{ij}\,\mathbf{e}_i$ und wenden wir spaltenweise die Multilinearität an. Dadurch wird $\det\mathbf{A}$ zu einer großen Summe von Termen $a_{i_1 1} a_{i_2 2} \cdots a_{i_n n}\, \det(\mathbf{e}_{i_1}, \ldots,
\mathbf{e}_{i_n})$, je einem für jede Wahl eines Zeilenindex $i_j$ in jeder Spalte $j$. Stimmen zwei gewählte Zeilenindizes überein, verschwindet der Term, da die Determinante alternierend ist. Sind alle $n$ Zeilenindizes verschieden, so ist die Liste $\mathbf{e}_{i_1}, \ldots, \mathbf{e}_{i_n}$ eine Umordnung der Spalten von $\mathbf{I}$. Durch wiederholtes Vertauschen von Spalten lässt sich $\det(\mathbf{e}_{i_1}, \ldots, \mathbf{e}_{i_n})$ also auf $\pm 1$ zurückführen.

<!--en:5410d3d0-->
Im $2 \times 2$-Fall mit den Spalten $\mathbf{a}_1 = p\,\mathbf{e}_1 +
r\,\mathbf{e}_2$ und $\mathbf{a}_2 = q\,\mathbf{e}_1 + s\,\mathbf{e}_2$ umfasst die Entwicklung vier Terme. Die beiden Terme mit wiederholtem Basisvektor entfallen. Übrig bleibt

<!--en:6c7fa41c-->
Damit erhalten wir wieder den von Hand hergeleiteten orientierten Flächeninhalt. Allgemein ist $\det\mathbf{A}$ eine Summe mit Vorzeichen über alle Möglichkeiten, aus jeder Zeile und jeder Spalte genau einen Eintrag zu wählen. Bezeichnen wir mit $\sigma(j)$ die in Spalte $j$ gewählte Zeile, erhalten wir die *Permutationsformel*:

<!--en:b9681263-->
Dabei durchläuft $\sigma$ alle $n!$ Permutationen von $\{1, \ldots, n\}$. Das Vorzeichen $\operatorname{sgn}(\sigma) = \det(\mathbf{e}_{\sigma(1)}, \ldots,
\mathbf{e}_{\sigma(n)})$ beträgt $+1$, wenn $\sigma$ eine gerade Anzahl von Vertauschungen erfordert, und $-1$ bei einer ungeraden Anzahl. Da jede Funktion, die die drei Axiome erfüllt, mit dieser Formel übereinstimmen muss, gibt es höchstens eine Determinante.

<!--en:51ea03c0-->
Den Existenzbeweis führen wir hier nicht vollständig aus. Entscheidend ist, dass $\operatorname{sgn}(\sigma)$ wohldefiniert ist. Keine Permutation lässt sich sowohl durch eine gerade als auch durch eine ungerade Anzahl von Vertauschungen erzeugen. Diese bekannte Tatsache übernehmen wir aus :cite:`Horn.Johnson.2012`. Mit dieser Aussage über die Parität erfüllt die Formel alle drei Axiome. Jeder Term enthält einen Eintrag aus jeder Spalte (Multilinearität). Für $\mathbf{I}$ trägt nur die identische Permutation bei (Normierung). Die alternierende Eigenschaft folgt aus demselben Paritätsargument, das wir hier nicht ausführen.

<!--en:2a8a6a19-->
Die Normierung kam erst ganz am Ende der Herleitung ins Spiel. Daraus gewinnen wir ein stärkeres Ergebnis für den nächsten Unterabschnitt. Ist $f$ *irgendeine* multilineare und alternierende Funktion der Spalten, so gelten dieselbe Vertauschungsregel und dieselbe Entwicklung. Jeder Faktor $f(\mathbf{e}_{\sigma(1)}, \ldots, \mathbf{e}_{\sigma(n)})$ wird zu $\operatorname{sgn}(\sigma)\,f(\mathbf{I})$. Folglich gilt

<!--en:4377af8b-->
Jede multilineare, alternierende Funktion der Spalten ist also ein skalares Vielfaches der Determinante.

<!--en:17f8019e-->
Für kleine Rechnungen von Hand gruppieren wir die Terme in :eqref:`eq_mdl-det-permutation` nach der in Spalte $1$ gewählten Zeile. So erhalten wir die *Kofaktorentwicklung*. Jeder Eintrag $a_{i1}$ wird mit der Determinante der $(n-1) \times (n-1)$-Untermatrix multipliziert, die durch Streichen der Zeile $i$ und der Spalte $1$ entsteht. Hinzu kommt das alternierende Vorzeichen $(-1)^{i+1}$. Für die $3 \times 3$-Matrix

<!--en:651fa5cf-->
ergibt die Entwicklung nach der ersten Spalte

<!--en:4d867e72-->
Damit bildet $\mathbf{M}$ den Einheitswürfel in $\mathbb{R}^3$ auf ein Parallelotop mit Volumen $8$ ab.

<!--en:438e7575-->
Bei einer Dreiecksmatrix vereinfacht sich die Determinante zum Produkt der Diagonaleinträge. Unter allen Rechenregeln für Determinanten ist diese besonders wichtig. Sei $\mathbf{U}$ eine obere Dreiecksmatrix, es gelte also $u_{ij} = 0$ für $i > j$. In der Permutationsformel muss ein Term ungleich null aus jeder Spalte einen Eintrag wählen, jeweils in verschiedenen Zeilen. In Spalte $1$ kommt nur der Eintrag in Zeile $1$ infrage. In Spalte $2$ ist Zeile $1$ bereits belegt, und die Zeilen $3, \ldots, n$ enthalten Nullen. Es bleibt nur Zeile $2$. Fahren wir spaltenweise so fort, bleibt nur die Permutation entlang der Diagonalen übrig. Sie erfordert keine Vertauschung, also gilt

<!--en:bffb2d81-->
Für untere Dreiecksmatrizen führen wir dasselbe Argument rückwärts von Spalte $n$ aus. Es lässt sich sogar auf Blöcke übertragen. Betrachten wir eine *obere Blockdreiecksmatrix* $\left(\begin{smallmatrix}\mathbf{D}_1 & *\\ \mathbf{0} &
\mathbf{D}_2\end{smallmatrix}\right)$, deren quadratische Diagonalblöcke $\mathbf{D}_1$ und $\mathbf{D}_2$ selbst keine Dreiecksmatrizen sein müssen. Jeder Permutationsterm ungleich null muss in den Spalten von $\mathbf{D}_1$ Einträge aus den Zeilen von $\mathbf{D}_1$ wählen, weil darunter nur Nullen stehen. Damit sind alle Zeilen von $\mathbf{D}_1$ belegt. Für die übrigen Spalten bleiben nur die Zeilen von $\mathbf{D}_2$. Der Block $*$ außerhalb der Diagonalen trägt nichts bei. Jede verbleibende Permutation zerfällt in zwei unabhängige Permutationen auf den beiden Blöcken. Ihre Vorzeichen multiplizieren sich, und die Summe zerfällt in ein Produkt:

<!--en:3f8d92cb-->
:numref:`fig_mdl-la-triangular-det` zeigt, welche Einträge in beiden Fällen beitragen.

<!--en:b0f5f31c-->
![Einträge, die zur Determinante beitragen. (a) Bei einer oberen Dreiecksmatrix schließen die Nullen unterhalb der Diagonalen jede Permutation außer der entlang der Diagonalen aus. Die Determinante ist daher das Produkt der Diagonaleinträge. (b) Bei einer oberen Blockdreiecksmatrix beschränkt der Nullblock jede beitragende Permutation auf die Diagonalblöcke $\mathbf{D}_1$ und $\mathbf{D}_2$. Die mit $*$ markierten Einträge tragen nie bei.](../img/mdl-la-triangular-det.svg)

<!--en:fb686b2e-->
Diese beiden Regeln sind auch im modernen maschinellen Lernen unverzichtbar. Ein normalisierender Fluss transformiert eine einfache Zufallsvariable mithilfe eines invertierbaren neuronalen Netzes. Bei jedem Schritt muss er die Log-Determinante der Jacobi-Matrix des Netzes berechnen, denn bei einem Variablenwechsel wird die Wahrscheinlichkeitsdichte durch den lokalen Volumenskalierungsfaktor dividiert (:numref:`subsec_mdl-change-of-variables`). Deshalb konstruiert man Flüsse so, dass jede Schicht eine dreieckige oder blockdreieckige Jacobi-Matrix hat. Die Log-Determinante reduziert sich dann auf eine Summe von $n$ skalaren Logarithmen. Damit entfallen die rund $n^3/3$ Multiplikationsadditionen, die eine allgemeine Matrix erfordert.

<!--en:35df33fe-->
Eine weitere nützliche Symmetrie besteht darin, dass Transposition die Determinante erhält, also $\det\mathbf{A}^\top = \det\mathbf{A}$. In der Permutationsformel für $\mathbf{A}^\top$ lautet jeder Term $\operatorname{sgn}(\sigma)\prod_j a_{j\sigma(j)}$. Mit $j = \sigma^{-1}(i)$ lässt sich das Produkt als $\prod_i a_{\sigma^{-1}(i)\, i}$ schreiben. Dies ist der Term für $\mathbf{A}$ zur inversen Permutation $\sigma^{-1}$. Um eine Folge von Vertauschungen rückgängig zu machen, brauchen wir genau dieselbe Anzahl von Vertauschungen. Daher gilt $\operatorname{sgn}(\sigma^{-1}) = \operatorname{sgn}(\sigma)$. Beide Summen enthalten also genau dieselben Terme. Jede Aussage über die Spalten einer Determinante gilt somit auch für ihre Zeilen.

<!--en:d9c2034b-->
Mit $n!$ Termen dient die Permutationsformel als theoretisches Hilfsmittel. Numerisch kommt sie allenfalls für winzige Matrizen infrage. Bibliotheken zerlegen die Matrix zur Berechnung der Determinante in Dreiecksfaktoren (wie bei `solve`), multiplizieren deren Diagonaleinträge und passen das Vorzeichen an die Zeilenvertauschungen an. Der Aufwand beträgt rund $n^3/3$ Multiplikationsadditionen.

<!--en:7bcbc3b1-->
### Multiplikativität

<!--en:baa6705a-->
Skaliert $\mathbf{B}$ orientierte Flächeninhalte mit $\det\mathbf{B}$ und $\mathbf{A}$ mit $\det\mathbf{A}$, so sollte die Anwendung von $\mathbf{B}$ und anschließend $\mathbf{A}$ die Flächeninhalte mit dem Produkt beider Faktoren skalieren. Prüfen wir dies an zwei Beispielen. Für unsere Beispielmatrix ergibt sich $\mathbf{A}^2 =
\left(\begin{smallmatrix}-1 & 8\\ -4 & 7\end{smallmatrix}\right)$ mit $\det\mathbf{A}^2 = -7 + 32 = 25 = 5^2$. Für die Verkettung aus :numref:`subsec_mdl-composition` gilt $\det(\mathbf{R}_{90}\mathbf{T}) = 0 \cdot
1 - (-1) \cdot 1 = 1$, in Übereinstimmung mit $\det\mathbf{R}_{90} \cdot \det\mathbf{T} =
1 \cdot 1$.

<!--en:b1af69b0-->
Eine Zerlegung in kleine Quadrate veranschaulicht, warum dies allgemein gelten sollte. Überziehen wir ein beliebiges ebenes Gebiet mit einem feinen Gitter aus winzigen Quadraten. Da eine lineare Abbildung überall im Raum gleich wirkt, bildet sie jedes Quadrat auf ein Parallelogramm ab. Dessen Flächeninhalt beträgt das $|\det\mathbf{A}|$-Fache des ursprünglichen Flächeninhalts. Folglich skaliert eine lineare Abbildung den Flächeninhalt *jeder* Figur mit $|\det\mathbf{A}|$. Wenden wir $\mathbf{B}$ auf ein Gebiet mit Flächeninhalt $V$ an, erhalten wir den Flächeninhalt $|\det\mathbf{B}|\,V$. Die anschließende Anwendung von $\mathbf{A}$ skaliert ihn auf $|\det\mathbf{A}|\,|\det\mathbf{B}|\,V$. Da die Verkettung $\mathbf{A}\mathbf{B}$ selbst eine lineare Abbildung ist, muss ihr Skalierungsfaktor gleich diesem Produkt sein. Mit der Charakterisierung aus :numref:`subsec_mdl-determinant-general` übersetzen wir diese Anschauung in einen dreizeiligen Beweis, der auch die Vorzeichen erfasst.

<!--en:d7840a6a-->
**Proposition (Multiplikativität der Determinante).** *Für quadratische Matrizen $\mathbf{A}$ und $\mathbf{B}$ gleicher Größe gilt*

<!--en:4f47ae0f-->
**Beweis.** Halten wir $\mathbf{A}$ fest und betrachten wir $D(\mathbf{B}) = \det(\mathbf{A}\mathbf{B})$ als Funktion der Spalten von $\mathbf{B}$. Nach :numref:`subsec_mdl-composition` ist Spalte $j$ von $\mathbf{A}\mathbf{B}$ gleich $\mathbf{A}\mathbf{b}_j$. Weil $\mathbf{A}$ linear und $\det$ multilinear ist, ist $D$ multilinear in den Spalten von $\mathbf{B}$. Stimmen zwei Spalten von $\mathbf{B}$ überein, so stimmen auch die entsprechenden Spalten von $\mathbf{A}\mathbf{B}$ überein. Damit ist $D(\mathbf{B}) = 0$ und die Funktion $D$ alternierend. Jede multilineare, alternierende Funktion der Spalten ist ein skalares Vielfaches der Determinante (:numref:`subsec_mdl-determinant-general`). Das ergibt

<!--en:1260df23-->
Vier Folgerungen aus :eqref:`eq_mdl-det-multiplicative` begegnen uns im weiteren Verlauf des Kapitels immer wieder:

<!--en:7f4c4709-->
1. *Determinanten kommutieren auch dann, wenn die Matrizen nicht kommutieren.* Da Determinanten gewöhnliche Skalare sind, gilt $\det(\mathbf{A}\mathbf{B}) =
   \det\mathbf{A}\,\det\mathbf{B} = \det\mathbf{B}\,\det\mathbf{A} =
   \det(\mathbf{B}\mathbf{A})$. Für unsere Drehung und Scherung gilt $\mathbf{R}_{90}\mathbf{T} \neq \mathbf{T}\mathbf{R}_{90}$, doch beide Produkte haben die Determinante $1$.

<!--en:9a085680-->
2. *Invarianz unter Ähnlichkeit.* Sei $\mathbf{V}$ eine beliebige invertierbare Matrix. Wenden wir Folgerung 1 auf die Faktoren $\mathbf{V}\mathbf{A}$ und $\mathbf{V}^{-1}$ an, ergibt sich $\det(\mathbf{V}\mathbf{A}\mathbf{V}^{-1}) =
   \det(\mathbf{V}^{-1}\mathbf{V}\mathbf{A}) = \det\mathbf{A}$. Die Matrix $\mathbf{V}\mathbf{A}\mathbf{V}^{-1}$ stellt dieselbe Abbildung $\mathbf{A}$ in dem Koordinatensystem dar, dessen Achsen die Spalten von $\mathbf{V}^{-1}$ vorgeben. Ein Koordinatenwechsel kann den Volumenskalierungsfaktor nicht verändern.

<!--en:b785ed27-->
3. *Inverse.* Ist $\mathbf{A}$ invertierbar, so ergibt sich mit $\mathbf{B} =
   \mathbf{A}^{-1}$ die Beziehung $\det\mathbf{A}\,\det\mathbf{A}^{-1} =
   \det\mathbf{I} = 1$. Daher gelten für jede invertierbare Matrix $\det\mathbf{A} \neq 0$ und $\det\mathbf{A}^{-1} = 1/\det\mathbf{A}$ (für unsere Beispielmatrix gilt $1/5 = 0.2$).

<!--en:e90cc334-->
4. *Orthogonale Matrizen.* Verbinden wir die Multiplikativität mit $\det\mathbf{Q}^\top = \det\mathbf{Q}$ und wenden beides auf $\mathbf{Q}^\top\mathbf{Q} =
   \mathbf{I}$ an, erhalten wir

   $$
   \det(\mathbf{Q})^2 = \det(\mathbf{Q}^\top)\det(\mathbf{Q})
    = \det(\mathbf{Q}^\top\mathbf{Q}) = \det(\mathbf{I}) = 1,
   $$

   und damit $\det\mathbf{Q} = \pm 1$. Starre Bewegungen erhalten stets das Volumen. In der Ebene unterscheidet das Vorzeichen orientierungserhaltende Drehungen ($\det = +1$) von orientierungsumkehrenden Spiegelungen ($\det = -1$) (:numref:`subsec_mdl-orthogonal-matrices`).

<!--en:12aa97d1-->
### Invertierbarkeit und Determinante

<!--en:d5dc1dea-->
Die Determinante ergänzt den Satz über invertierbare Matrizen aus :numref:`subsec_mdl-inverses` um eine fünfte äquivalente Bedingung. Als einzige lässt sie sich durch eine einzelne Zahl ausdrücken. In zwei Dimensionen ist die Äquivalenz unmittelbar zu sehen. Der orientierte Flächeninhalt $ps - qr$ verschwindet genau dann, wenn beide Spalten auf derselben Ursprungsgeraden liegen (oder eine von ihnen $\mathbf{0}$ ist), also genau dann, wenn die Spalten linear abhängig sind. Dasselbe gilt in jeder Dimension.

<!--en:e980c79b-->
**Proposition (Invertierbarkeit und Determinante).** *Eine $n \times n$-Matrix $\mathbf{A}$ ist genau dann invertierbar, wenn $\det\mathbf{A} \neq 0$ gilt. Gleichbedeutend damit gilt $\det\mathbf{A} = 0$ genau dann, wenn die Spalten von $\mathbf{A}$ linear abhängig sind.*

<!--en:b77b5779-->
**Beweis.** Ist $\mathbf{A}$ invertierbar, so gilt nach Folgerung 3 aus der Multiplikativität $\det\mathbf{A}\,\det\mathbf{A}^{-1} = 1$ und damit $\det\mathbf{A} \neq 0$. Nehmen wir umgekehrt an, $\mathbf{A}$ sei nicht invertierbar. Nach dem Satz in :numref:`subsec_mdl-inverses` sind ihre Spalten linear abhängig. Also ist eine Spalte eine Linearkombination der anderen, $\mathbf{a}_j = \sum_{i \neq j} c_i\mathbf{a}_i$. Entwickeln wir Spalte $j$ mithilfe der Multilinearität, ergibt sich $\det\mathbf{A} = \sum_{i \neq j} c_i\det(\ldots, \mathbf{a}_i, \ldots)$. Dabei tritt $\mathbf{a}_i$ sowohl an Position $i$ als auch an Position $j$ auf. Jeder Term hat zwei identische Spalten und verschwindet daher, weil die Determinante alternierend ist. $\blacksquare$

<!--en:b934f937-->
Für unsere Beispielmatrix belegt $\det\mathbf{A} = 5 \neq 0$ die Invertierbarkeit. Für $\mathbf{B}$ bestätigt $\det\mathbf{B} = 0$ die Singularität. Beachten wir jedoch, was der Satz *nicht* verspricht. Eine Determinante ungleich null garantiert die mathematische Existenz von $\mathbf{A}^{-1}$. Ob sich die Inverse numerisch zuverlässig berechnen lässt, ist eine andere Frage (:numref:`subsec_mdl-det-limits`).

<!--en:66d12e21-->
### Blockmatrizen und Aktualisierungen mit niedrigem Rang

<!--en:b05022a0-->
Oft kennen wir die Inverse bereits für einen Teil einer Matrix, etwa für einen Block oder für die Matrix vor einer kleinen Änderung. Können wir sie wiederverwenden? Die Antwort liefert ein einziger Eliminationsschritt, bei dem wir mit Blöcken statt mit Zahlen rechnen. Zerlegen wir eine quadratische Matrix $\mathbf{M}$ in vier Blöcke $\mathbf{M}_{11}$, $\mathbf{M}_{12}$, $\mathbf{M}_{21}$ und $\mathbf{M}_{22}$, wobei $\mathbf{M}_{11}$ quadratisch und invertierbar sei. Ziehen wir das Produkt von $\mathbf{M}_{21}\mathbf{M}_{11}^{-1}$ mit der ersten Blockzeile von der zweiten ab, so verschwindet der linke untere Block. Ein entsprechender Schritt mit den Spalten lässt den rechten oberen Block verschwinden. Als Produkt geschrieben lautet dies

<!--en:bdd5a9e5-->
wie das Ausmultiplizieren der drei Faktoren bestätigt. Die Matrix $\mathbf{S}$ heißt *Schur-Komplement* von $\mathbf{M}_{11}$. Sie ist der Teil von $\mathbf{M}_{22}$, der nach der Elimination des ersten Blocks übrig bleibt. Für unsere Beispielmatrix mit $1 \times 1$-Blöcken gilt $\mathbf{S} = 3 - (-1) \cdot 1^{-1} \cdot 2
= 5$.

<!--en:8ef1edc6-->
Die beiden äußeren Faktoren sind Dreiecksmatrizen mit Einsen auf der Diagonalen und haben daher jeweils die Determinante $1$. Aus der Multiplikativität und der Blockregel folgt $\det\mathbf{M} = \det\mathbf{M}_{11}\,\det\mathbf{S}$ (im Beispiel $1 \cdot 5$). Außerdem ist $\mathbf{M}$ genau dann invertierbar, wenn $\mathbf{S}$ invertierbar ist. Einen äußeren Faktor invertieren wir, indem wir das Vorzeichen seines Blocks außerhalb der Diagonalen umkehren. Invertieren wir die drei Faktoren in umgekehrter Reihenfolge, so ergibt sich

<!--en:a6773a5a-->
Der rechte untere Block von $\mathbf{M}^{-1}$ ist $\mathbf{S}^{-1}$, der linke obere Block ist $\mathbf{M}_{11}^{-1} +
\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{S}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1}$. Wir müssen jedoch nicht links oben beginnen. Eliminieren wir zuerst $\mathbf{M}_{22}$, vertauschen sich die Rollen der Blöcke. Der linke obere Block von $\mathbf{M}^{-1}$ ergibt sich dann als Inverse des anderen Schur-Komplements, $\mathbf{M}_{11} -
\mathbf{M}_{12}\mathbf{M}_{22}^{-1}\mathbf{M}_{21}$. Diese Matrix begegnet uns in :numref:`subsec_mdl-mvn-conditioning` als bedingte Kovarianz einer Gauß-Verteilung wieder.

<!--en:6a979864-->
Die beiden Ausdrücke für den linken oberen Block müssen übereinstimmen. Daraus folgt ein überraschend nützliches Ergebnis. Angenommen, wir kennen $\mathbf{A}^{-1}$ für eine $n \times n$-Matrix $\mathbf{A}$. Nun ändere sich $\mathbf{A}$ um eine Matrix vom Rang $k$, die wir als $\mathbf{U}\mathbf{W}^\top$ mit $\mathbf{U}, \mathbf{W} \in \mathbb{R}^{n \times k}$ schreiben. Wählen wir die Blöcke $\mathbf{M}_{11} = \mathbf{A}$, $\mathbf{M}_{12} = -\mathbf{U}$, $\mathbf{M}_{21} = \mathbf{W}^\top$ und $\mathbf{M}_{22} = \mathbf{I}_k$. Das eine Schur-Komplement ist dann $\mathbf{A} + \mathbf{U}\mathbf{W}^\top$, das andere die $k \times k$-Matrix $\mathbf{I}_k +
\mathbf{W}^\top\mathbf{A}^{-1}\mathbf{U}$. Setzen wir die beiden Ausdrücke für den linken oberen Block gleich, erhalten wir die *Woodbury-Identität*

<!--en:adcadb91-->
Ändert sich eine Matrix um eine Matrix vom Rang $k$, so ändert sich ihre Inverse um eine Matrix mit Rang höchstens $k$. Um diese Änderung zu bestimmen, müssen wir nur eine $k \times k$-Matrix invertieren. Kennen wir $\mathbf{A}^{-1}$ bereits, kostet die Aktualisierung Multiplikationsadditionen in der Größenordnung $n^2k$ statt $n^3$ bei erneuter Inversion. Ist $\mathbf{A}$ diagonal, kostet das Lösen eines Systems mit $\mathbf{A} + \mathbf{U}\mathbf{W}^\top$ nur Aufwand der Größenordnung $nk^2$. Die beiden Eliminationswege liefern auch zwei Ausdrücke für $\det\mathbf{M}$ und damit $\det(\mathbf{A} + \mathbf{U}\mathbf{W}^\top)
= \det\mathbf{A}\,\det(\mathbf{I}_k +
\mathbf{W}^\top\mathbf{A}^{-1}\mathbf{U})$. Für $k = 1$ ist die innere Inversion eine Division. Die Identität heißt dann *Sherman-Morrison-Formel*:

<!--en:303f7d9e-->
Diese Formel erklärt, warum sich bei einer Anpassung nach der Methode der kleinsten Quadrate einzelne Beispiele hinzufügen oder entfernen lassen. Jedes Beispiel ändert $\mathbf{X}^\top\mathbf{X}$ um einen Term vom Rang eins. Aufgabe :exref:`ex_mdl-geometry-linear-algebraic-ops-det-rank-one-update` wendet sowohl die Determinante als auch die Inverse auf einen normalisierenden Fluss an.

<!--en:0d8038f6-->
### Grenzen der Determinante

<!--en:2e9891f6-->
Aus $\det\mathbf{A} \neq 0$ folgt, dass eine Matrix invertierbar ist. Wie *nahe* sie daran ist, Dimensionen zu verlieren, oder in welchen Richtungen das geschieht, erfahren wir daraus nicht. Zwei extreme Beispiele zeigen dies von entgegengesetzten Seiten. Erstens ist die skalierte Einheitsmatrix $0.1\,\mathbf{I}_{400}$ mit der Inversen $10\,\mathbf{I}_{400}$ optimal konditioniert. Ihre Determinante $10^{-400}$ ist jedoch so winzig, dass sie unterhalb der kleinsten positiven Gleitkommazahl mit doppelter Genauigkeit liegt und ein Unterlauf auftritt. Zweitens hat die extreme Scherung $\left(\begin{smallmatrix}1 & 10^8\\ 0 & 1\end{smallmatrix}\right)$ die Determinante $1$. Dennoch genügt es, ihren linken unteren Eintrag um $10^{-8}$ zu ändern, um sie singulär zu machen (:exref:`ex_mdl-geometry-linear-algebraic-ops-det-near-singular`).

<!--en:ee7d19fe-->
Im Folgenden berechnen wir beide Determinanten und die Singulärwerte der Scherung. Für die erste Determinante verwenden wir `slogdet`, um Vorzeichen und Logarithmus des Betrags zuverlässig zu erhalten. Welchen Rang erwarten Sie für die *unveränderte* Scherung, bevor Sie die NumPy-Zelle ausführen?

<!--en:7f5e9e4c-->
Wie erwartet liefert `det` für $0.1\,\mathbf{I}_{400}$ wegen eines Unterlaufs `0.0`, während `slogdet` problemlos $400 \ln 0.1 \approx -921.03$ berechnet. Die unveränderte Scherung hat die Determinante $1$, die gestörte bis auf Rundung die Determinante $0$. Trotzdem meldet `matrix_rank` den Rang $1$ für *beide*. Bei exakter Rechnung hat die unveränderte Scherung Rang $2$, doch ihre beiden Singulärwerte betragen etwa $10^8$ und $10^{-8}$. NumPy behandelt einen Singulärwert als null, sobald er unter das Produkt aus dem größten Singulärwert, der Matrixgröße und der Maschinengenauigkeit fällt, hier $10^8 \cdot 2 \cdot 2.2 \cdot 10^{-16} \approx 4 \cdot
10^{-8}$. Die Einträge der Scherung lassen sich zwar exakt speichern, doch Gleitkommarechnungen mit Zahlen der Größenordnung $10^8$ führen zu Rundungsfehlern um $10^{-8}$. Numerisch lässt sich die Scherung daher nicht von einer singulären Matrix unterscheiden.

<!--en:9b1a240e-->
Eine einzelne Zahl kann nicht erfassen, wie eine Matrix den Raum in verschiedenen Richtungen streckt und staucht. Um diese Richtungen zu untersuchen, brauchen wir Eigenwerte und Singulärwerte. Unsere Beispielmatrix $\mathbf{A}$ hat die Eigenwerte $2 \pm i$ mit dem Produkt $5 = \det\mathbf{A}$ (:numref:`sec_mdl-eigendecompositions`). Ihre Singulärwerte betragen etwa $3.618$ und $1.382$ mit dem Produkt $5 = |\det\mathbf{A}|$ (:numref:`sec_mdl-svd-low-rank`). Tatsächlich ist die Determinante jeder quadratischen Matrix das Produkt ihrer Eigenwerte, mit Vielfachheit gezählt und einschließlich komplexer Paare. Der Betrag $|\det\mathbf{A}|$ ist das Produkt ihrer Singulärwerte. Wie nahe eine Matrix daran ist, eine Dimension zu verlieren, misst das *Verhältnis* des größten zum kleinsten Singulärwert. Die Determinante gibt darüber keine Auskunft. Als Nächstes bestimmen wir diese richtungsabhängigen Faktoren.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:fe527adf-->
Eine Matrix stellt eine lineare Abbildung dar. Ihre Spalten geben die Bilder der Standardbasisvektoren an und legen so die gesamte Abbildung fest. Die Matrizenmultiplikation entspricht der von rechts nach links gelesenen Verkettung linearer Abbildungen. Die einsteinsche Summenkonvention und `einsum` erfassen Skalarprodukte, Matrixprodukte, Spuren und mehrachsige Tensorkontraktionen wie die Attention im Transformer mit einer einzigen Indexregel.

<!--en:65a75cea-->
Zu jeder $m \times n$-Matrix gehören der Spaltenraum aller erreichbaren Ausgaben und der Nullraum aller Eingaben mit Bild null. Der Dimensionssatz $\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n$ verknüpft beide Räume. Eine $256 \times 784$-Schicht verliert daher zwangsläufig mindestens $528$ Eingabedimensionen. Bei einer quadratischen Matrix sind unabhängige Spalten, ein trivialer Nullraum, voller Rang, eine Determinante ungleich null und die Existenz einer Inversen äquivalent. Die Determinante gibt den orientierten Faktor an, mit dem eine Abbildung Volumen skaliert. Bei Verkettung multiplizieren sich die Determinanten. Für jede Dreiecksmatrix ist die Determinante gleich dem Produkt der Diagonaleinträge. Ein Schritt der Blockelimination liefert das Schur-Komplement. Daraus ergeben sich die Determinante und die Inverse einer Blockmatrix sowie über die Woodbury-Identität die Inverse nach einer Änderung mit niedrigem Rang. Eine Determinante ungleich null garantiert zwar die Existenz einer Inversen. Wie nahe eine Matrix der Singularität kommt, lässt sich daraus jedoch nicht ablesen. Diese Frage führt direkt zu Eigenwerten und Singulärwerten.

<!--en:2294243a-->
## Übungen

<!--en:ef3b01db-->
1. **Winkelerhaltende Abbildungen.** Drehen wir jeden Einbettungsvektor mit derselben orthogonalen Matrix, bleiben alle Kosinusähnlichkeiten erhalten. In dieser Aufgabe bestimmen Sie alle linearen Abbildungen mit dieser Eigenschaft.
    1. Bestimmen Sie die Bedingung an die Spalten einer quadratischen Matrix $\mathbf{A}$, unter der $\mathbf{A}\mathbf{e}_1$ und $\mathbf{A}\mathbf{e}_2$ orthogonal sind. Zeigen Sie, dass $\mathbf{A} = \operatorname{diag}(1, 10)$ diese Bedingung erfüllt, aber den Winkel zwischen $(1, 1)$ und $(1, -1)$ verändert.
    1. Angenommen, eine invertierbare Matrix $\mathbf{A}$ erhält den Winkel zwischen je zwei Vektoren ungleich null. Wenden Sie dies für die Paare $\mathbf{e}_i, \mathbf{e}_j$ und $\mathbf{e}_i + \mathbf{e}_j, \mathbf{e}_i - \mathbf{e}_j$ mit $i \neq j$ an. Zeigen Sie damit, dass $\mathbf{A}^\top\mathbf{A} = c^2\mathbf{I}$ für ein $c > 0$ gilt, also $\mathbf{A} = c\mathbf{Q}$ mit einer orthogonalen Matrix $\mathbf{Q}$. Zeigen Sie umgekehrt, dass jede solche Matrix alle Kosinusähnlichkeiten erhält.
    1. Nehmen Sie stattdessen an, dass $\|\mathbf{A}\mathbf{x}\| = \|\mathbf{x}\|$ für jedes $\mathbf{x}$ gilt. Multiplizieren Sie $\|\mathbf{x} + \mathbf{y}\|^2$ und $\|\mathbf{x} - \mathbf{y}\|^2$ aus, um zu zeigen, dass $\mathbf{A}$ jedes Skalarprodukt erhält. Folgern Sie, dass $\mathbf{A}$ orthogonal ist.
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-angle-preserving`

<!--en:a04e760e-->
1. **Rang eines Produkts.** Ein linearer Autoencoder leitet eine $d$-dimensionale Eingabe durch eine versteckte Schicht der Breite $k < d$. Seine Abbildung lautet $\mathbf{x} \mapsto \mathbf{W}_2\mathbf{W}_1\mathbf{x}$ mit $\mathbf{W}_1 \in \mathbb{R}^{k \times d}$ und $\mathbf{W}_2 \in \mathbb{R}^{d \times k}$.
    1. Sei $\mathbf{A}$ eine $m \times n$-Matrix und $\mathbf{B}$ eine $n \times p$-Matrix. Zeigen Sie, dass der Spaltenraum von $\mathbf{A}\mathbf{B}$ im Spaltenraum von $\mathbf{A}$ liegt und der Nullraum von $\mathbf{B}$ im Nullraum von $\mathbf{A}\mathbf{B}$. Beweisen Sie $\operatorname{rank}(\mathbf{A}\mathbf{B}) \le
       \min(\operatorname{rank}\mathbf{A}, \operatorname{rank}\mathbf{B})$ mithilfe der ersten Aussage sowie des Dimensionssatzes in Verbindung mit der zweiten Aussage.
    1. Folgern Sie, dass $\mathbf{W}_2\mathbf{W}_1$ höchstens Rang $k$ hat. Wie klein kann die Dimension des Nullraums sein? Welche Dimension kann ein Unterraum höchstens haben, auf dem der Autoencoder jede Eingabe exakt rekonstruiert?
    1. Konstruieren Sie eine $2 \times 3$-Matrix und eine $3 \times 2$-Matrix, beide vom Rang zwei, deren $2 \times 2$-Produkt Rang eins hat.
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-rank-of-product`

<!--en:7cb82b50-->
1. **Matrizen, die mit einer Drehung kommutieren.** Im Allgemeinen ist die Matrizenmultiplikation nicht kommutativ. In dieser Aufgabe bestimmen Sie die Ausnahmen für $\mathbf{R}_{90}$.
    1. Bestimmen Sie alle $2 \times 2$-Matrizen $\mathbf{X}$ mit $\mathbf{X}\mathbf{R}_{90} = \mathbf{R}_{90}\mathbf{X}$ und zeigen Sie, dass sie die Form $\left(\begin{smallmatrix}p & -q\\ q & p\end{smallmatrix}\right)
       = p\mathbf{I} + q\mathbf{R}_{90}$ haben.
    1. Zeigen Sie mithilfe von $\mathbf{R}_{90}^2 = -\mathbf{I}$, dass $(p\mathbf{I} + q\mathbf{R}_{90})(r\mathbf{I} + s\mathbf{R}_{90}) =
       (pr - qs)\,\mathbf{I} + (ps + qr)\,\mathbf{R}_{90}$ gilt. Diese Matrizen multiplizieren sich also wie die komplexen Zahlen $p + qi$.
    1. Zeigen Sie für $(p, q) \neq (0, 0)$, dass $p\mathbf{I} + q\mathbf{R}_{90} =
       \sqrt{p^2 + q^2}\,\mathbf{R}_\phi$ für den Winkel $\phi$ mit $(\cos\phi, \sin\phi) = (p, q)/\sqrt{p^2 + q^2}$ gilt. Es handelt sich also um eine skalierte Drehung. Berechnen Sie ihre Determinante und folgern Sie, dass sie außer im Fall $p = q = 0$ invertierbar ist.
    1. Kommutiert unsere Beispielmatrix $\mathbf{A}$ mit $\mathbf{R}_{90}$? Berechnen Sie beide Produkte.
   :exlabel:`ex_mdl-linear-maps-commute-rotation`

<!--en:05419465-->
1. **Householder-Spiegelungen.** Sei $\mathbf{u}$ ein Einheitsvektor in $\mathbb{R}^n$ und sei $\mathbf{H} = \mathbf{I} - 2\mathbf{u}\mathbf{u}^\top$.
    1. Zeigen Sie, dass $\mathbf{H}$ symmetrisch und orthogonal ist, dass $\mathbf{H}\mathbf{u} = -\mathbf{u}$ gilt und dass $\mathbf{H}\mathbf{x} =
       \mathbf{x}$ für jedes $\mathbf{x}$ senkrecht zu $\mathbf{u}$ gilt. Damit spiegelt $\mathbf{H}$ an der Hyperebene mit Normalenvektor $\mathbf{u}$.
    1. Schreiben Sie $\mathbf{H} = \mathbf{I} - 2\mathbf{P}$ mit $\mathbf{P} = \mathbf{u}\mathbf{u}^\top$ und erklären Sie die Formel anhand der Projektion auf $\mathbf{u}$ aus :numref:`sec_mdl-geometry-linear-algebraic-ops`. Wie wirkt $\mathbf{x} - 2\mathbf{P}\mathbf{x}$ auf die Komponente von $\mathbf{x}$ in Richtung $\mathbf{u}$?
    1. Beweisen Sie $\det\mathbf{H} = -1$. Ergänzen Sie $\mathbf{u}$ zu einer Orthonormalbasis und schreiben Sie deren Vektoren als Spalten einer orthogonalen Matrix $\mathbf{Q} = [\mathbf{u}\
       \mathbf{q}_2 \cdots \mathbf{q}_n]$. Berechnen Sie $\mathbf{Q}^\top\mathbf{H}\mathbf{Q}$ und nutzen Sie die Multiplikativität.
    1. Bestimmen Sie einen Einheitsvektor $\mathbf{u}$, dessen Spiegelung $(3, 4)$ auf $(5, 0)$ abbildet (Hinweis: $\mathbf{u}$ ist parallel zu $(3, 4) - (5, 0)$). Geben Sie $\mathbf{H}$ an und prüfen Sie $\det\mathbf{H} = -1$. Mithilfe solcher spaltenweise angewandten Spiegelungen zerlegen Bibliotheken eine Matrix in einen orthogonalen Faktor und einen Dreiecksfaktor.
   :exlabel:`ex_mdl-linear-maps-householder`

<!--en:c1a671fe-->
1. [code] **Determinante und Nähe zur Singularität.** Eine quadratische Matrix ist genau dann invertierbar, wenn ihre Determinante ungleich null ist. Wie gut misst der Betrag der Determinante, wie nahe eine Matrix der Singularität kommt?
    1. Beweisen Sie mithilfe der Multilinearität die Identität $\det(c\mathbf{A}) = c^n\det\mathbf{A}$ für eine $n \times n$-Matrix $\mathbf{A}$ und einen Skalar $c$. Prüfen Sie die Identität im Code für die Matrix $\mathbf{M}$ aus :eqref:`eq_mdl-det-3x3` mit $c = 2$.
    1. Erzeugen Sie eine untere $n \times n$-Dreiecksmatrix mit unabhängigen standardnormalverteilten Einträgen auf und unter der Diagonalen. Wählen Sie dafür jeweils $n = 20$, $100$ und $1000$. Vergleichen Sie den von `np.linalg.slogdet` zurückgegebenen Logarithmus mit der Summe der Logarithmen der Beträge der Diagonaleinträge. Nach der Dreiecksregel ist diese Summe bis auf Rundung exakt. Ab welchem $n$ weichen beide Werte voneinander ab, und wie groß ist die Abweichung bei $n = 1000$? Berechnen Sie `np.linalg.cond` für $n = 20$ und $n = 100$ und erklären Sie die Abweichung. (Die Routine nutzt die Dreiecksstruktur nicht. Sie zerlegt die Matrix wie jede andere mit Zeilenvertauschungen.)
    1. Laut Text macht eine Änderung um $10^{-8}$ im linken unteren Eintrag die Scherung $\left(\begin{smallmatrix}1 & 10^8\\ 0 &
       1\end{smallmatrix}\right)$ singulär. Prüfen Sie dies und zeigen Sie, dass keine kleinere Änderung eines einzelnen Eintrags genügt.
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-det-near-singular`

<!--en:054f5081-->
1. **Determinante einer Rang-eins-Aktualisierung.** ● Ein *planarer Fluss* :cite:`rezende2015variational` transformiert $\mathbf{z} \in \mathbb{R}^d$ gemäß $f(\mathbf{z}) = \mathbf{z} + \mathbf{u}\,h(\mathbf{w}\cdot\mathbf{z} + b)$ für Vektoren $\mathbf{u}, \mathbf{w}$, einen Skalar $b$ und eine skalare Nichtlinearität $h$. Um die Dichte von $f(\mathbf{z})$ zu berechnen, brauchen wir die Determinante der Jacobi-Matrix $\mathbf{I} + h'(\mathbf{w}\cdot\mathbf{z} + b)\,\mathbf{u}\mathbf{w}^\top$.
    1. Zeigen Sie mithilfe der Multilinearität und der alternierenden Eigenschaft, dass $\det(\mathbf{u}\mathbf{w}^\top) = 0$ für $n \times n$-Matrizen mit $n \ge 2$ gilt.
    1. Prüfen Sie die beiden Faktorisierungen
       $$\begin{aligned}\begin{bmatrix}\mathbf{I} & -\mathbf{u}\\ \mathbf{w}^\top & 1\end{bmatrix} &= \begin{bmatrix}\mathbf{I} & \mathbf{0}\\ \mathbf{w}^\top & 1\end{bmatrix}\begin{bmatrix}\mathbf{I} & -\mathbf{u}\\ \mathbf{0}^\top & 1 + \mathbf{w}\cdot\mathbf{u}\end{bmatrix} \\ &= \begin{bmatrix}\mathbf{I} & -\mathbf{u}\\ \mathbf{0}^\top & 1\end{bmatrix}\begin{bmatrix}\mathbf{I} + \mathbf{u}\mathbf{w}^\top & \mathbf{0}\\ \mathbf{w}^\top & 1\end{bmatrix}.\end{aligned}$$
       Zeigen Sie mithilfe der Permutationsformel :eqref:`eq_mdl-det-permutation`, dass eine Matrix mit letzter Spalte $\mathbf{e}_{n+1}$ dieselbe Determinante hat wie ihr linker oberer $n \times n$-Block. Leiten Sie $\det(\mathbf{I} + \mathbf{u}\mathbf{w}^\top) = 1 + \mathbf{w}\cdot\mathbf{u}$ aus :eqref:`eq_mdl-det-multiplicative` und der Dreiecksregel ab.
    1. Geben Sie den Logarithmus des Betrags der Jacobi-Determinante eines planaren Flusses an. Vergleichen Sie den Rechenaufwand mit dem für eine allgemeine $d \times d$-Determinante. Sei $h = \tanh$. Für welche Werte von $\mathbf{u}\cdot\mathbf{w}$ bleibt die Jacobi-Matrix an jeder Stelle $\mathbf{z}$ invertierbar?
    1. Geben Sie mit der Sherman-Morrison-Formel :eqref:`eq_mdl-sherman-morrison` die Inverse der Jacobi-Matrix $\mathbf{I} + c\,\mathbf{u}\mathbf{w}^\top$ an, wobei $c = h'(\mathbf{w}\cdot\mathbf{z} + b)$ gilt. Prüfen Sie, dass die Formel genau bei den Werten von $c$ versagt, für die die Determinante verschwindet. Zählen Sie die Multiplikationsadditionen, die nötig sind, um die Inverse auf einen Vektor anzuwenden. Die inverse Jacobi-Matrix kehrt die Funktion $f$ selbst nicht um. Zeigen Sie, dass sich die Rekonstruktion von $\mathbf{z}$ aus $f(\mathbf{z})$ auf eine skalare Gleichung für $\mathbf{w}\cdot\mathbf{z}$ zurückführen lässt.
   :exlabel:`ex_mdl-geometry-linear-algebraic-ops-det-rank-one-update`

<!--en:55a1533f-->
1. [code] **Einsteinsche Summenkonvention für Attention.** Der Abfrage-, der Schlüssel- und der Wertetensor haben die Formen $(n_b, n_h, n_q, d_h)$, $(n_b, n_h, n_k, d_h)$ und $(n_b, n_h, n_k, d_h)$.
    1. Geben Sie die `einsum`-Zeichenketten für die Scores $s_{bhit} = \sum_j
       q_{bhij}k_{bhtj}$ und für die gewichtete Summe der Werte $o_{bhij} =
       \sum_t p_{bhit}\,v_{bhtj}$ an, wobei $p_{bhit}$ die Attention-Gewichte sind. Prüfen Sie beide Ergebnisse durch Vergleich mit `@` und `swapaxes`.
    1. Geben Sie die Anzahl der Multiplikationsadditionen jeder Kontraktion als Produkt der Dimensionsgrößen an. Vergleichen Sie sie mit der Anzahl der Multiplikationsadditionen, die eine lineare $d_h \times d_h$-Abbildung für alle Abfragen zusammen benötigt. Ab welcher Anzahl von Schlüsseln $n_k$ ist die Berechnung der Scores teurer?
    1. Schreiben Sie $\operatorname{tr}(\mathbf{A}^\top\mathbf{A})$, $\operatorname{tr}(\mathbf{A}\mathbf{A}^\top)$ und $\sum_{ij}
       a_{ij}^2$ jeweils als einen `einsum`-Ausdruck für eine zufällige $3 \times 5$-Matrix und prüfen Sie, dass alle drei übereinstimmen. Welche der beiden Gleichheiten ist ein Spezialfall von :eqref:`eq_mdl-trace-cyclic`, und weshalb gilt die andere?
   :exlabel:`ex_mdl-linear-maps-einsum-attention`

<!--en:883f609c-->
[Diskussion](https://d2l.discourse.group/t/36855)

<!--en:3a462572-->
Matrizen als lineare Abbildungen<br>**Geometrie der Spalten · Verkettung und einsum · Dimensionssatz · Invertierbarkeit · Determinanten und Volumenskalierung**

<!--en:2ccd0091-->
::: {.slide title="Wann verliert eine lineare Schicht Information?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:a4515305-->
- **Von Zeilen zu Spalten:** Zeilenweise liefert $\mathbf{A}\mathbf{x}$ die Ergebnisse von $m$ Hyperebenentests. Die Spalten zeigen die Bilder der Achsenvektoren des Eingaberaums.

<!--en:7dd06f83-->
- **Engpass durch die Dimension:** Eine $256 \times 784$-Schicht für Fashion-MNIST gibt weniger Zahlen aus, als sie aufnimmt. Mindestens $784 - 256 = 528$ Richtungen gehen auf null.

<!--en:f88f65a6-->
- **Invertierbar oder singulär:** Unsere Beispielmatrix $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ ist umkehrbar. Dagegen bildet $\mathbf{B} = \left(\begin{smallmatrix}2&1\\4&2\end{smallmatrix}\right)$ die beiden Eingaben $(1, 1)$ und $(0, 3)$ auf dieselbe Ausgabe $(3, 6)$ ab.

<!--en:7a50eed7-->
::: {.slide title="Spalten als Bilder der Koordinatenachsen"}

<!--en:7378b465-->
[Geometrie der Spalten]{.kicker}

<!--en:c95d81c5-->
- **Wirkung auf Basisvektoren:** Aus $\mathbf{x} = x\mathbf{e}_1 + y\mathbf{e}_2$ folgt, dass $\mathbf{A}\mathbf{x}$ eine gewichtete Kombination der Spalten von $\mathbf{A}$ ist:
  $$\mathbf{A}\mathbf{x} = x\,\mathbf{A}\mathbf{e}_1 + y\,\mathbf{A}\mathbf{e}_2$$

<!--en:5c606fcd-->
- **Beispielmatrix $\mathbf{A}$:** Spalte 1 ist $\mathbf{A}\mathbf{e}_1 = (1, -1)$ und Spalte 2 ist $\mathbf{A}\mathbf{e}_2 = (2, 3)$. Daher geht $\mathbf{v} = (2, -1)$ in $2(1, -1) - (2, 3) = (0, -5)$ über.

<!--en:f7483625-->
- **Erhaltung der Gitterstruktur:** Wegen $\mathbf{A}(\alpha\mathbf{x} + \beta\mathbf{y}) = \alpha\mathbf{A}\mathbf{x} + \beta\mathbf{A}\mathbf{y}$ bleiben Geraden Geraden, der Ursprung bleibt fest ($\mathbf{A}\mathbf{0} = \mathbf{0}$) und das Einheitsquadrat geht in ein Parallelogramm über.

<!--en:6856f3c0-->
::: {.slide title="Sechs geometrische Abbildungen des Einheitsquadrats"}

<!--en:0dc7da6f-->
[Beispiele für lineare Abbildungen]{.kicker}

<!--en:0e50d297-->
- **Abbildungen mit vollem Rang und mit Rangdefizit:** Skalierung, Drehung $\mathbf{R}_{90} = \left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$, Scherung $\mathbf{T} = \left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ und Spiegelung bilden das Einheitsquadrat auf ein zweidimensionales Parallelogramm ab. Bei der Projektion (e) und bei abhängigen Spalten (f) fällt die ganze Ebene auf eine eindimensionale Gerade zusammen.

<!--en:cde0f5df-->
- **Affine Schichten ($\mathbf{W}\mathbf{x} + \mathbf{b}$):** Ein Bias ungleich null verschiebt den Ursprung ($\mathbf{0} \mapsto \mathbf{b}$). Eine affine Schicht entspricht einer linearen Abbildung mit anschließender Verschiebung.

<!--en:92d5d6a9-->
::: {.slide title="Matrizenmultiplikation als Verkettung von Abbildungen"}

<!--en:bfe139b3-->
[Verkettung]{.kicker}

<!--en:997c5c2b-->
- **Verkettung von rechts nach links:** Zuerst $\mathbf{B}$ und dann $\mathbf{A}$ auf $\mathbf{v}$ anzuwenden entspricht dem Produkt $\mathbf{A}\mathbf{B}$. Dessen $j$-te Spalte ist $\mathbf{A}\mathbf{b}_j$:
  $$\bigl(\mathbf{A}(\mathbf{B}\mathbf{v})\bigr)_i = \sum_k \Bigl(\sum_j a_{ij} b_{jk}\Bigr) v_k = \bigl((\mathbf{A}\mathbf{B})\mathbf{v}\bigr)_i$$

<!--en:1c505a47-->
- **Die Reihenfolge zählt ($\mathbf{R}_{90}\mathbf{T} \neq \mathbf{T}\mathbf{R}_{90}$):** Erst scheren, dann drehen ergibt $\mathbf{e}_1 \mapsto (0, 1)$. Erst drehen, dann scheren ergibt $\mathbf{e}_1 \mapsto (1, 1)$.

<!--en:3dc4cd27-->
- **Assoziativität:** $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$ folgt unmittelbar aus der Assoziativität der Funktionsverkettung.

<!--en:de1a609b-->
::: {.slide title="Orthogonale Matrizen erhalten Längen und Winkel"}

<!--en:06b2e4b5-->
[Starre Bewegungen]{.kicker}

<!--en:a4cbebd2-->
- **Orthonormale Spalten ($\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$):** Die quadratische Matrix $\mathbf{Q}$ erhält Skalarprodukte $(\mathbf{Q}\mathbf{x})\cdot(\mathbf{Q}\mathbf{y}) = \mathbf{x}^\top\mathbf{Q}^\top\mathbf{Q}\mathbf{y} = \mathbf{x}\cdot\mathbf{y}$, Längen und Winkel.

<!--en:a02cd1ee-->
- **Beidseitige Inverse ($\mathbf{Q}^{-1} = \mathbf{Q}^\top$):** In $\mathbb{R}^2$ sind nur Drehungen $\mathbf{R}_\phi$ und Spiegelungen orthogonal.

<!--en:d8d9bdc3-->
- **Gleiche Längen sind entscheidend:** $\operatorname{diag}(1, 10)$ hat zueinander senkrechte Spalten, verändert aber den Winkel von $90^\circ$ auf $169^\circ$.

<!--en:ce3a1bc9-->
::: {.slide title="Einsteinsche Summenkonvention für Tensorkontraktionen"}

<!--en:a85bae55-->
[Indexnotation]{.kicker}

<!--en:c58ab825-->
- **Eine Regel für alle Produkte:** Einträge multiplizieren und über die Indizes summieren, die nach `->` fehlen.

<!--en:ceabed54-->
- **Zyklische Invarianz der Spur:** $\operatorname{tr}(\mathbf{A}\mathbf{B}) = \sum_{i,j} a_{ij}b_{ji} = \operatorname{tr}(\mathbf{B}\mathbf{A})$, also $\operatorname{tr}(\mathbf{V}\mathbf{A}\mathbf{V}^{-1}) = \operatorname{tr}(\mathbf{A})$.

<!--en:303d5503-->
::: {.slide title="Kontraktionen für Multi-Head-Attention mit einsum"}

<!--en:1a9930ea-->
[Indexnotation in der Praxis]{.kicker}

<!--en:5bfc4938-->
- **Beibehaltene und summierte Achsen:** `'bhij,bhtj->bhit'` behält Batch $b$, Kopf $h$, Abfrage $i$ und Schlüssel $t$ bei und summiert über die Merkmalsdimension $j$.

<!--en:a865184c-->
- **Kosten der Kontraktionsreihenfolge:** $\mathbf{A}(\mathbf{B}\mathbf{v})$ benötigt $2n^2$ Operationen gegenüber $n^3$ für $(\mathbf{A}\mathbf{B})\mathbf{v}$.

<!--en:9fa0b371-->
::: {.slide title="Spaltenraum und Nullraum einer Matrix"}

<!--en:43a8f470-->
[Grundlegende Unterräume]{.kicker}

<!--en:5dac562f-->
- **Spaltenraum ($\operatorname{col}\mathbf{A} \subseteq \mathbb{R}^m$):** Der von den Spalten von $\mathbf{A}$ aufgespannte Raum umfasst genau die erreichbaren Ausgaben $\{\mathbf{A}\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$.

<!--en:fffa38c6-->
- **Nullraum oder Kern ($\ker\mathbf{A} \subseteq \mathbb{R}^n$):** Der Unterraum der Eingaben mit Bild null, $\{\mathbf{x} \in \mathbb{R}^n : \mathbf{A}\mathbf{x} = \mathbf{0}\}$, besteht aus allen Vektoren, die zu jeder Zeile von $\mathbf{A}$ orthogonal sind.

<!--en:9562901d-->
- **Singuläres Beispiel $\mathbf{B} = \left(\begin{smallmatrix}2&1\\4&2\end{smallmatrix}\right)$:** Wegen $\mathbf{b}_1 = 2\mathbf{b}_2$ ist $\operatorname{col}\mathbf{B}$ die Gerade $y = 2x$ und $\ker\mathbf{B}$ die Ursprungsgerade durch $(1, -2)$. Die Eingaben $(1, 1)$ und $(0, 3)$ unterscheiden sich um $(1, -2)$ und haben dasselbe Bild $(3, 6)$.

<!--en:4f686ff0-->
::: {.slide title="Der Dimensionssatz und die Erhaltung der Dimension"}

<!--en:dcd9166e-->
[Rang und Dimension]{.kicker}

<!--en:510fc8af-->
- **Erhaltung der Dimension:** Für jede Matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ gilt $\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n$.

<!--en:0daece81-->
- **Beweis durch Basisergänzung:** Die Ergänzung einer Basis $\mathbf{w}_1, \ldots, \mathbf{w}_k$ von $\ker\mathbf{A}$ um $\mathbf{u}_1, \ldots, \mathbf{u}_{n-k}$ liefert unabhängige Bilder $\mathbf{A}\mathbf{u}_i$, die $\operatorname{col}\mathbf{A}$ aufspannen.

<!--en:0fe5068b-->
- **Schichten mit Flaschenhals:** Für eine $256 \times 784$-Schicht gilt $\operatorname{rank}\mathbf{A} \le 256$ und damit zwangsläufig $\dim\ker\mathbf{A} \ge 528$.

<!--en:feed87e7-->
::: {.slide title="Zeilenrang, Spaltenrang und Faktorisierung mit Flaschenhals"}

<!--en:68730bc0-->
[Rang und Flaschenhals]{.kicker}

<!--en:666c3ae6-->
- **Faktorisierung vom Rang $r$ ($\mathbf{A} = \mathbf{G}\mathbf{H}$):** $\mathbf{G} \in \mathbb{R}^{m \times r}$ enthält eine Basis des Spaltenraums, $\mathbf{H} \in \mathbb{R}^{r \times n}$ die zugehörigen Koeffizienten, deren Zeilen den Zeilenraum aufspannen. Daraus folgt die Gleichheit von Zeilenrang und Spaltenrang.

<!--en:da617ab1-->
- **Speicherbedarf bei niedrigem Rang:** $\mathbf{G}\mathbf{H}$ benötigt $r(m+n)$ Zahlen statt $mn$.

<!--en:a677fb01-->
::: {.slide title="Inverse Matrizen und die explizite 2×2-Formel"}

<!--en:f61c5022-->
[Invertierbarkeit]{.kicker}

<!--en:2a2425ac-->
- **Beidseitige Inverse ($\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$):** Über $\mathbf{x} = \mathbf{A}^{-1}(\mathbf{A}\mathbf{x})$ lässt sich die Eingabe eindeutig rekonstruieren.

<!--en:3803a2fc-->
- **Explizite $2 \times 2$-Formel:** Für $ps - qr \neq 0$ gilt $\mathbf{A}^{-1} = \frac{1}{ps - qr}\left(\begin{smallmatrix}s & -q \\ -r & p\end{smallmatrix}\right)$.

<!--en:2273acbd-->
- **Beispielmatrix $\mathbf{A}$:** Aus $ps - qr = 5$ ergibt sich $\mathbf{A}^{-1} = \frac{1}{5}\left(\begin{smallmatrix}3&-2\\1&1\end{smallmatrix}\right)$ mit der Wirkung $(0, -5) \mapsto (2, -1)$.

<!--en:c0c943b4-->
::: {.slide title="Invertierbarkeit quadratischer Matrizen"}

<!--en:f61c5022-2-->
[Invertierbarkeit]{.kicker}

<!--en:67096cfd-->
Für eine quadratische $n \times n$-Matrix $\mathbf{A}$ sind Injektivität und Surjektivität auf $\mathbb{R}^n$ äquivalent. Damit hängen auch unsere Bedingungen an die Unterräume zusammen:

<!--en:41657c37-->
**Proposition (invertierbare Matrizen).** Für eine $n \times n$-Matrix $\mathbf{A}$ sind folgende Aussagen äquivalent: (i) $\mathbf{A}$ ist invertierbar. (ii) Die Spalten von $\mathbf{A}$ sind linear unabhängig. (iii) $\ker\mathbf{A} = \{\mathbf{0}\}$. (iv) $\operatorname{rank}\mathbf{A} = n$.

<!--en:b4af9dcc-->
- **Spaltenweise Konstruktion von $\mathbf{A}^{-1}$:** Aus $\operatorname{rank}\mathbf{A} = n$ folgen $\operatorname{col}\mathbf{A} = \mathbb{R}^n$ und $\ker\mathbf{A} = \{\mathbf{0}\}$. Daher hat $\mathbf{A}\mathbf{x} = \mathbf{e}_j$ für jede Achse eine eindeutige Lösung $\mathbf{x}_j$. Das Zusammensetzen von $\mathbf{X} = [\mathbf{x}_1 \cdots \mathbf{x}_n]$ ergibt $\mathbf{A}\mathbf{X} = \mathbf{I}$ und $\mathbf{X}\mathbf{A} = \mathbf{I}$.

<!--en:75e16ae6-->
- **Warum nichtquadratische Matrizen keine beidseitige Inverse haben:** Für $m > n$ (hohe Matrix) gilt $\operatorname{rank}\mathbf{A} \le n < m$, die Abbildung ist also nie surjektiv. Für $n > m$ (breite Matrix) gilt $\dim\ker\mathbf{A} \ge n - m > 0$, sodass es stets verschiedene Eingaben mit demselben Bild gibt.

<!--en:f6ba87ed-->
::: {.slide title="Numerische Regel für lineare Systeme: direkt lösen"}

<!--en:23e7d6f0-->
[Numerische lineare Algebra]{.kicker}

<!--en:6014bc63-->
- **Geschwindigkeit und Rückwärtsstabilität:** `np.linalg.solve(A, b)` zerlegt $\mathbf{A}$ mit LU in $\sim n^3/3$ Operationen und garantiert ein Residuum $\|\mathbf{A}\hat{\mathbf{x}} - \mathbf{b}\|$ im Bereich der Maschinengenauigkeit. Dagegen kostet `np.linalg.inv(A) @ b` etwa $\sim n^3$ Operationen und kann das Residuum um einen Faktor bis zu $\kappa(\mathbf{A})$ vergrößern:

<!--en:bf2bbb4a-->
**Kondition und Stabilität:** Für die schlecht konditionierte $10 \times 10$-Hilbert-Matrix ($\kappa \approx 1.6 \times 10^{13}$) erreicht `solve` ein Residuum von $10^{-16}$, während `inv` ein Residuum von $10^{-5}$ liefert. Bei dünn besetzten Matrizen zerstört `inv` zudem die dünne Besetzung und füllt den Speicher mit einer dicht besetzten Matrix.

<!--en:ba3adce1-->
::: {.slide title="Die 2×2-Determinante als orientierter Flächeninhalt eines Parallelogramms"}

<!--en:a2f35155-->
[Die Determinante]{.kicker}

<!--en:f86ef4d2-->
- **Grundseite mal orientierte Höhe:** Für die Spalten $\mathbf{u} = (p, r)$ und $\mathbf{w} = (q, s)$ liefert die Drehung von $\mathbf{u}$ um $+90^\circ$ den Einheitsnormalenvektor $\mathbf{n} = (-r, p)/\|\mathbf{u}\|$.

<!--en:4928f5e1-->
- **Die Längen kürzen sich:** Die Projektion von $\mathbf{w}$ auf $\mathbf{n}$ liefert die orientierte Höhe. Damit beträgt der Flächeninhalt
  $$\det\begin{bmatrix}p & q \\ r & s\end{bmatrix} = \|\mathbf{u}\|(\mathbf{w}\cdot\mathbf{n}) = ps - qr$$

<!--en:1410ea70-->
- **Beispielmatrix ($\det\mathbf{A} = 5$):** Vom Flächeninhalt $12$ des umschließenden Rechtecks den Flächeninhalt $7$ der äußeren rechtwinkligen Dreiecke abziehen. Es bleibt $5$, in Übereinstimmung mit $1(3) - 2(-1) = 5$.

<!--en:d63303c8-->
::: {.slide title="Orientierung, Vorzeichen und Volumenverlust"}

<!--en:fd5c72fe-->
[Geometrische Fälle]{.kicker}

<!--en:5e7fd4a8-->
- **Drei geometrische Fälle:** $|\det\mathbf{A}|$ ist der Skalierungsfaktor für Flächeninhalte. Bei $\det > 0$ bleibt die Orientierung gegen den Uhrzeigersinn erhalten (a). Bei $\det < 0$ kehrt sich die Orientierung durch den Spaltentausch um (b). Bei $\det = 0$ fällt das Quadrat auf eine Strecke mit Flächeninhalt null zusammen (c).

<!--en:fd5eae36-->
::: {.slide title="Drei Axiome definieren die n-dimensionale Determinante"}

<!--en:ca855df1-->
[Allgemeine Determinante]{.kicker}

<!--en:db49ba99-->
- **1. Linear in jeder Spalte:** Skalierung und Addition in Spalte $j$ übertragen sich auf das orientierte Volumen.

<!--en:8598049d-->
- **2. Alternierend ($\det = 0$ bei gleichen Spalten):** Stimmen zwei Spalten überein, fällt der Körper zusammen. Das Vertauschen zweier Spalten kehrt das Vorzeichen um.

<!--en:b22da505-->
- **3. Normiert ($\det\mathbf{I} = 1$):** Damit ergibt sich eindeutig eine Summe über $n!$ Permutationen, nämlich $\det\mathbf{A} = \sum_\sigma \operatorname{sgn}(\sigma)\prod_j a_{\sigma(j)j}$.

<!--en:da3abd08-->
::: {.slide title="Dreiecksmatrizen, Blockregeln und Kofaktoren"}

<!--en:7b3213d5-->
[Berechnung von Determinanten]{.kicker}

<!--en:21a305e5-->
- **Rechenregel für Dreiecksmatrizen:** Nur $\sigma(j) = j$ vermeidet Nullfaktoren. Daher gelten $\det\mathbf{U} = \prod_i u_{ii}$ und $\det\left(\begin{smallmatrix}\mathbf{D}_1 & * \\ \mathbf{0} & \mathbf{D}_2\end{smallmatrix}\right) = \det\mathbf{D}_1\det\mathbf{D}_2$.

<!--en:41328772-->
- **Invarianz unter Transposition:** $\det\mathbf{A}^\top = \det\mathbf{A}$.

<!--en:70de0145-->
- **Normalisierende Flüsse:** Bei dreieckigen Jacobi-Matrizen lässt sich $\log|\det\mathbf{J}| = \sum_i \log|J_{ii}|$ als Summe mit Aufwand $O(d)$ statt $O(d^3)$ berechnen.

<!--en:3f46eb86-->
::: {.slide title="Multiplikativität der Volumenskalierung bei Verkettung"}

<!--en:17bc2979-->
[Multiplikativität]{.kicker}

<!--en:b86c4527-->
- **Multiplikativität ($\det(\mathbf{A}\mathbf{B}) = \det\mathbf{A}\det\mathbf{B}$):** $D(\mathbf{B}) = \det(\mathbf{A}\mathbf{B})$ ist multilinear und alternierend in den Spalten von $\mathbf{B}$. Daher gilt $D(\mathbf{B}) = D(\mathbf{I})\det\mathbf{B}$.

<!--en:453372a8-->
- **Wichtige Folgerungen:** Kommutativität der Skalarmultiplikation $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{B}\mathbf{A})$, Invarianz unter Ähnlichkeit $\det(\mathbf{V}\mathbf{A}\mathbf{V}^{-1}) = \det\mathbf{A}$ sowie $\det(\mathbf{A}^{-1}) = 1/\det\mathbf{A}$ und $\det\mathbf{Q} = \pm 1$.

<!--en:c27bee65-->
::: {.slide title="Invertierbarkeitskriterium: det A ≠ 0"}

<!--en:49dee9dc-->
[Invertierbarkeit und Volumen]{.kicker}

<!--en:2639c398-->
**Proposition (Invertierbarkeit und Determinante).** Eine $n \times n$-Matrix $\mathbf{A}$ ist genau dann invertierbar, wenn $\det\mathbf{A} \neq 0$ gilt. Gleichbedeutend damit gilt $\det\mathbf{A} = 0$ genau dann, wenn die Spalten von $\mathbf{A}$ linear abhängig sind.

<!--en:70050730-->
- **Beweis in beiden Richtungen:**
  - Ist $\mathbf{A}$ invertierbar, gilt $\det(\mathbf{A})\det(\mathbf{A}^{-1}) = \det\mathbf{I} = 1$. Das schließt $\det\mathbf{A} = 0$ aus.
  - Ist $\mathbf{A}$ nicht invertierbar, ist eine Spalte eine Linearkombination der anderen, $\mathbf{a}_j = \sum_{i \neq j} c_i \mathbf{a}_i$. Die Entwicklung von Spalte $j$ mithilfe der Multilinearität liefert eine Summe von Determinanten mit jeweils zwei identischen Spalten $\mathbf{a}_i$. Sie verschwinden also alle.

<!--en:74baa303-->
- **Fünf äquivalente Bedingungen für $n \times n$-Matrizen:** invertierbar $\iff$ unabhängige Spalten $\iff$ $\ker\mathbf{A} = \{\mathbf{0}\}$ $\iff$ $\operatorname{rank}\mathbf{A} = n$ $\iff$ $\det\mathbf{A} \neq 0$.

<!--en:3db4b4b5-->
::: {.slide title="Warum der Betrag der Determinante die Kondition nicht misst"}

<!--en:a6dc8563-->
[Numerische Fallstricke]{.kicker}

<!--en:4503e8cf-->
- **Zwei gegensätzliche Fälle:** Eine winzige Determinante bedeutet nicht zwangsläufig, dass eine Matrix nahezu singulär ist. Ebenso garantiert $\det = 1$ keine gute Kondition:

<!--en:53e7fd3c-->
- **Unterlauf und schlechte Kondition:** $0.1\,\mathbf{I}_{400}$ ist optimal konditioniert ($\kappa = 1$, Inverse $10\,\mathbf{I}_{400}$). Dennoch liefert `det` durch Unterlauf `0.0` ($10^{-400}$ erfordert `slogdet`). Die extreme Scherung $\left(\begin{smallmatrix}1&10^8\\0&1\end{smallmatrix}\right)$ hat $\det = 1$, aber Singulärwerte von etwa $10^8$ und $10^{-8}$. Nur $10^{-8}$ trennen sie von einer singulären Matrix.

<!--en:281d13cb-->
::: {.slide title="Ausblick auf Eigenwerte und Singulärwerte"}

<!--en:84a4f31e-->
[Ausblick]{.kicker}

<!--en:bd5b86d1-->
- **Warum ein Skalar nicht genügt:** Die Determinante fasst die Streckfaktoren aller $n$ Dimensionen zu einem Produkt zusammen. Ob eine Matrix alle Achsen gleichmäßig streckt oder eine Achse stark staucht und eine andere stark streckt, bleibt verborgen.

<!--en:6919740e-->
- **Beispielmatrix $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$:**
  - Ihre Eigenwerte sind $\lambda_{1,2} = 2 \pm i$ mit dem Produkt $(2+i)(2-i) = 5 = \det\mathbf{A}$ (§{sec_mdl-eigendecompositions}).
  - Ihre Singulärwerte sind $\sigma_1 \approx 3.618$ und $\sigma_2 \approx 1.382$ mit dem Produkt $3.618 \times 1.382 = 5 = |\det\mathbf{A}|$ (§{sec_mdl-svd-low-rank}).

<!--en:d5ad3b48-->
- **Nähe zur Singularität:** Das Verhältnis $\kappa(\mathbf{A}) = \sigma_1/\sigma_n$ misst, wie nahe eine Matrix daran ist, eine Dimension zu verlieren. Die Determinante $\det\mathbf{A}$ erfasst dies nicht.

<!--en:bb2516b3-->
::: {.slide title="Verständnisfragen zu linearen Abbildungen, Rang und Determinanten"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:104562fe-->
1. **Kommutativität:** Kommutieren $\mathbf{D} = \operatorname{diag}(2, 1)$ und die Drehung um $90^\circ$ mit der Matrix $\mathbf{R}_{90} = \left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$? Beide Produkte an $\mathbf{e}_1 = (1, 0)$ prüfen.

<!--en:1dff751f-->
2. **Anwendung des Dimensionssatzes:** Für $\mathbf{M} = \left(\begin{smallmatrix}1&1\\2&2\end{smallmatrix}\right)$ den Spaltenraum, einen Basisvektor des Nullraums, den Rang und die Determinante bestimmen.

<!--en:099cff9e-->
3. **Flaschenhals im Autoencoder:** Ein linearer Autoencoder $\mathbf{W}_2\mathbf{W}_1$ bildet wie folgt ab: $\mathbb{R}^{100} \to \mathbb{R}^{16} \to \mathbb{R}^{100}$. Wie klein kann die Dimension von $\ker(\mathbf{W}_2\mathbf{W}_1)$ sein?

<!--en:7cbf5152-->
**Lösungen:**
**1.** Nein, denn $\mathbf{R}_{90}\mathbf{D}\mathbf{e}_1 = \mathbf{R}_{90}(2, 0) = (0, 2)$, während $\mathbf{D}\mathbf{R}_{90}\mathbf{e}_1 = \mathbf{D}(0, 1) = (0, 1)$ gilt.
**2.** Beide Spalten sind $(1, 2)$. Daher ist $\operatorname{col}\mathbf{M}$ die Gerade $y = 2x$, und es gelten $\ker\mathbf{M} = \operatorname{span}\{(1, -1)\}$, $\operatorname{rank}\mathbf{M} = 1$ und $\det\mathbf{M} = 2 - 2 = 0$.
**3.** Der Zwischenraum hat die Dimension $16$. Daher gelten $\operatorname{rank}(\mathbf{W}_2\mathbf{W}_1) \le 16$ und $\dim\ker(\mathbf{W}_2\mathbf{W}_1) \ge 100 - 16 = 84$.

<!--en:1d9ec1c6-->
::: {.slide title="Zusammenfassung: Matrizen als lineare Abbildungen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:af6f9890-->
| Begriff | Mathematische Regel | Geometrische Bedeutung und Rolle im Deep Learning |

<!--en:629cc03b-->
| **Spaltenweise Betrachtung** | $\mathbf{A}\mathbf{x} = \sum_{j=1}^n x_j \mathbf{a}_j$, $\mathbf{a}_j = \mathbf{A}\mathbf{e}_j$ | Die Spalten sind Bilder der Achsenvektoren und legen die Wirkung auf das gesamte Eingabegitter fest |

<!--en:8bfd1242-->
| **Verkettung und `einsum`** | $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$ | Zuerst $\mathbf{B}$ anwenden, dann $\mathbf{A}$. Mit `einsum` lassen sich Kontraktionen auf vierdimensionale Attention-Tensoren verallgemeinern |

<!--en:2e0f8046-->
| **Orthogonale Matrix** | $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I} \iff \mathbf{Q}^{-1} = \mathbf{Q}^\top$ | Starre Drehung oder Spiegelung erhält Skalarprodukte, euklidische Normen und Winkel |

<!--en:b3423e44-->
| **Dimensionssatz** | $\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n$ | Erhaltene Dimensionen ($\operatorname{col}\mathbf{A}$) und verlorene Dimensionen ($\ker\mathbf{A}$) ergeben zusammen die Eingabebreite $n$ |

<!--en:43c9ccb4-->
| **Invertierbarkeit und `solve`** | $\ker\mathbf{A} = \{\mathbf{0}\} \iff \operatorname{rank}\mathbf{A} = n \iff \det\mathbf{A} \neq 0$ | Quadratische Abbildung ist injektiv und surjektiv. Stets `solve(A, b)` ($n^3/3$) verwenden, niemals `inv(A)` |

<!--en:291b4b39-->
| **Determinante und Volumen** | $\det(\mathbf{A}\mathbf{B}) = \det\mathbf{A}\det\mathbf{B}$, $\det\mathbf{U} = \prod_i u_{ii}$ | Orientierter Volumenskalierungsfaktor. Dreieckige Jacobi-Matrizen ermöglichen in Flüssen Aufwand $O(n)$ für `slogdet` |
