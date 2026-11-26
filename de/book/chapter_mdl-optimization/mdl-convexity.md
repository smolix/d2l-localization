<!--en:75c07670-->
# Konvexität

<!--en:ff8ee6bf-->
Wenn ein Trainingslauf endet, wissen wir meist nur eines über seinen Endpunkt: Der Verlust sinkt dort nicht weiter. Mit einer hinreichend kleinen Schrittweite erreicht der Gradientenabstieg zuverlässig einen solchen Punkt, an dem der Gradient nahezu null ist (:numref:`sec_mdl-gradient-based-optimization`). Das Newton-Verfahren gelangt schneller dorthin, selbst wenn sich der stationäre Punkt als Sattelpunkt erweist (:numref:`sec_mdl-curvature-methods`). Keine der beiden Garantien verrät uns, ob es anderswo einen wesentlich besseren Punkt gibt. Wann ist ein stationärer Punkt tatsächlich der bestmögliche Punkt, und wie schnell konvergiert der Gradientenabstieg gegen ihn?

<!--en:3c25e1d1-->
Ein einfaches eindimensionales Experiment zeigt, warum sich diese Frage stellt. Wir führen dieselbe Schleife für den Gradientenabstieg von $500$ zufälligen Startpunkten aus, zuerst für die parabelförmige Schale $f(x) = (x - \tfrac12)^2$ und dann für das geneigte Doppelmuldenpotential $\phi(x) = (x^2 - 1)^2 + x/2$. Bei der Schale erreicht jeder Lauf denselben Minimierer. Beim Doppelmuldenpotential bleibt fast die Hälfte der Läufe im flacheren Tal stecken. Dort liegt der Verlust um fast eine ganze Einheit über dem globalen Minimum. Auch dort ist der Gradient null. Mehr verspricht die Stationaritätsgarantie nicht, und der Gradient allein verrät nichts über das tiefere Tal in der Nähe.

<!--en:2e54a59c-->
Die beiden Funktionen unterscheiden sich in einer geometrischen Eigenschaft. Jede Sehne zwischen zwei Punkten der Schale liegt über ihrem Graphen. Beim Doppelmuldenpotential verlaufen manche Sehnen dagegen teilweise unterhalb der Kurve. Funktionen, deren Sehnen stets auf oder über ihrem Graphen liegen, heißen *konvex*. Für sie hat unsere Ausgangsfrage eine einfache Antwort: Jedes lokale Minimum ist ein globales Minimum. Der Beweis braucht nur drei Zeilen (:numref:`subsec_mdl-local-global`). Dahinter steht eine noch nützlichere Tatsache: Jede Tangente an eine konvexe Funktion bildet im gesamten Definitionsbereich eine untere Schranke für den Graphen. Die lokale Information an einem einzigen Punkt beschränkt also die Funktion überall. Summieren wir diese Tangentenschranke über die Iterierten des Gradientenabstiegs, erhalten wir Konvergenzraten, die für jeden Startpunkt gelten. Mitteln wir sie über eine Zufallsvariable, erhalten wir die Jensen-Ungleichung, die wir in den Kapiteln über Wahrscheinlichkeit und Informationstheorie immer wieder nutzen.

<!--en:7a9de611-->
Dieser Abschnitt behandelt viele Ergebnisse von unterschiedlicher Tragweite. Drei davon sollten Sie sich merken: die Tangente als untere Schranke, die globale Optimalität lokaler Minima und die lineare Konvergenzrate des Gradientenabstiegs. Der Beweis dieser Rate beruht auf einer einzigen Ungleichung, die auch ohne Konvexität gelten kann. Wir beginnen mit konvexen Mengen und drei gleichwertigen Kriterien für konvexe Funktionen. Dann leiten wir die wichtigsten Folgerungen ab und stellen einige Rechenregeln zusammen, mit denen sich konvexe Verlustfunktionen erkennen lassen. Da die Verlustfunktionen tiefer Netze nicht konvex sind, fragen wir zum Schluss, welche Teile der Theorie sich auf sie übertragen lassen. Wenn Sie wenig Zeit haben, lesen Sie die Sehnen- und Tangentenbedingungen in :numref:`subsec_mdl-three-lenses` und springen Sie direkt zu :numref:`subsec_mdl-why-convexity-matters`. Zwei Beispiele begleiten uns durch den Abschnitt: An der quadratischen Funktion $\tfrac12(x_1^2 + 1000\,x_2^2)$, die wir im ganzen Kapitel verwenden, messen wir Konvergenzraten. Das Doppelmuldenpotential $\phi$ wiederum besteht keines unserer Konvexitätskriterien.

<!--en:3896d366-->
## Konvexe Mengen

<!--en:b850c450-->
Die Definition konvexer Funktionen betrachtet Verbindungsstrecken zwischen Punkten ihres Definitionsbereichs. Wir beginnen deshalb mit den Mengen, in denen diese Punkte liegen. Der Definitionsbereich einer Verlustfunktion, der zulässige Bereich eines Problems mit Nebenbedingungen und die Menge aller Minimierer sind Beispiele dafür. Wir brauchen nur eine Definition, drei typische Familien von Beispielen und eine Regel, mit der wir aus bekannten konvexen Mengen neue bilden.

<!--en:21ba3fb9-->
### Definition und Beispiele

<!--en:89f8bc2d-->
Geometrisch heißt eine Menge konvex, wenn die gerade Verbindungsstrecke zwischen je zwei ihrer Punkte vollständig in der Menge liegt. Formal heißt eine Menge $C \subseteq \mathbb{R}^d$ von $d$-dimensionalen Punkten **konvex**, wenn

<!--en:274230d2-->
Läuft $\theta$ von $1$ bis $0$, so durchläuft die Kombination $\theta\mathbf{x} + (1-\theta)\mathbf{y}$ die Strecke von $\mathbf{x}$ nach $\mathbf{y}$. :numref:`fig_mdl-opt-convex-vs-nonconvex-set` stellt einer konvexen Menge eine nichtkonvexe Sichel gegenüber und zeigt die beiden häufigsten konvexen Mengen im Deep Learning: den Wahrscheinlichkeitssimplex und einen Halbraum.

<!--en:eb0bdfd1-->
![Konvexe und nichtkonvexe Mengen. Links eine konvexe Menge, in der jede Verbindungsstrecke zwischen zwei ihrer Punkte liegt. In der Mitte eine nichtkonvexe Sichel, bei der eine Sehne zwischen zwei ihrer Punkte die Menge verlässt. Rechts der Wahrscheinlichkeitssimplex $\{\mathbf{p}\succeq0,\ \mathbf{1}^\top\mathbf{p}=1\}$ und ein Halbraum $\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}\leq b\}$, beide konvex.](../img/mdl-opt-convex-vs-nonconvex-set.svg)

<!--en:f2f3f9c0-->
Was schließt die Definition aus? Betrachten wir den Kreisring $\{\mathbf{x} : 1 \le \|\mathbf{x}\| \le 2\}$. Er enthält sowohl $(\tfrac32, 0)$ als auch $(-\tfrac32, 0)$. Der Mittelpunkt ihrer Verbindungsstrecke ist aber der Ursprung, der im Loch liegt. Ein einziger fehlender Mittelpunkt genügt, um :eqref:`eq_mdl-opt-convex-set` zu verletzen.

<!--en:4fa3d563-->
Drei Familien konvexer Mengen decken fast alle Fälle in diesem Kapitel ab. Erstens sind jede Hyperebene $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} = b\}$ und jeder Halbraum $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \le b\}$ konvex, denn $\mathbf{a}^\top\mathbf{x}$ ist linear. Sind zwei Zahlen gleich $b$ (oder höchstens $b$), so ist auch ihr gewichteter Mittelwert gleich $b$ (oder höchstens $b$). Zweitens ist jede Normkugel $\{\mathbf{x} : \|\mathbf{x}\| \le r\}$ konvex. Aus Dreiecksungleichung und Homogenität folgt $\|\theta\mathbf{x} + (1-\theta)\mathbf{y}\| \le \theta\|\mathbf{x}\| + (1-\theta)\|\mathbf{y}\| \le r$. Damit erfassen wir sowohl die $\ell_2$-Kugeln beim Gewichtszerfall und beim Gradienten-Clipping als auch die $\ell_1$-Kugeln bei der Regularisierung, die dünn besetzte Lösungen fördert. Drittens gibt es den Wahrscheinlichkeitssimplex $\Delta = \{\mathbf{p} \in \mathbb{R}^d : \mathbf{p} \succeq 0,\ \mathbf{1}^\top\mathbf{p} = 1\}$. Dabei bedeutet $\mathbf{p} \succeq 0$, dass jeder Eintrag nichtnegativ ist. Attention-Gewichte, Klassenwahrscheinlichkeiten und Mischgewichte liegen alle auf $\Delta$. Seine Konvexität folgt unmittelbar aus unserer nächsten Regel.

<!--en:3df53121-->
### Neue konvexe Mengen aus bekannten Mengen

<!--en:e068cc9e-->
Der Schnitt beliebig vieler konvexer Mengen ist stets konvex, auch bei unendlich vielen Mengen. Liegen zwei Punkte im Schnitt, so gehören sie zu jeder einzelnen Menge. Ihre Verbindungsstrecke liegt also in jeder Menge und damit im Schnitt. Anschaulich hat eine konvexe Menge keine Einbuchtungen. Auch wenn man sie mit einer anderen Menge ohne Einbuchtungen schneidet, entsteht keine. Für Vereinigungen gibt es keine solche Garantie: $[0,1] \cup [2,3]$ ist die Vereinigung zweier konvexer Intervalle, doch die Strecke von $1$ bis $2$ verlässt die Menge.

<!--en:f6a475ab-->
Mit dieser Schnittregel weisen wir die Konvexität zweier wichtiger Mengen nach. Der Wahrscheinlichkeitssimplex ist der Schnitt einer Hyperebene ($\mathbf{1}^\top\mathbf{p} = 1$) mit $d$ Halbräumen ($-p_i \le 0$) und damit konvex. Auch die Menge der symmetrischen positiv semidefiniten Matrizen (PSD) ist konvex, denn sie ist der Schnitt unendlich vieler Halbräume im Raum der symmetrischen Matrizen. Für jeden festen Vektor $\mathbf{z}$ ist die Nebenbedingung $\mathbf{z}^\top \mathbf{A} \mathbf{z} \ge 0$ eine einzelne lineare Ungleichung in den Einträgen von $\mathbf{A}$.

<!--en:44646dcb-->
Wir können eine konvexe Menge auch von innen aufbauen. Die **konvexe Hülle** einer Menge $S$ besteht aus allen gewichteten Mittelwerten $\sum_i \theta_i \mathbf{x}_i$ von Punkten $\mathbf{x}_i \in S$ mit $\theta_i \ge 0$ und $\sum_i \theta_i = 1$. Sie ist konvex und sogar die kleinste konvexe Menge, die $S$ enthält (Aufgabe :exref:`ex_mdl-convexity-prove-jensen-s-finite`). So gesehen ist der Wahrscheinlichkeitssimplex die konvexe Hülle der Standardbasisvektoren $\mathbf{e}_1, \ldots, \mathbf{e}_d$: Jede Wahrscheinlichkeitsverteilung ist ein gewichteter Mittelwert von Verteilungen, bei denen jeweils ein Ergebnis sicher ist.

<!--en:533dffd2-->
## Konvexe Funktionen

<!--en:f9e4bff9-->
So wie eine konvexe Menge keine Einbuchtungen hat, krümmt sich der Graph einer konvexen Funktion nach oben, ohne in ein zweites Tal abzufallen. Wir entwickeln drei gleichwertige Konvexitätskriterien, die keine, eine beziehungsweise zwei Ableitungen benötigen. Danach verfeinern wir den Begriff in zwei Richtungen: Starke Konvexität bestimmt, wie schnell der Gradientenabstieg konvergiert, und Subgradienten erfassen Verlustfunktionen mit Knicken. Wenn Sie sich nur eines der drei Kriterien merken, wählen Sie das zweite: Die Tangente als untere Schranke trägt fast jeden Beweis in diesem Abschnitt.

<!--en:8ef3421c-->
### Die Sehnenbedingung

<!--en:e91f5187-->
Das erste Kriterium kommt ganz ohne Ableitungen aus. Es gilt deshalb für die Betragsfunktion und den Hinge-Verlust genauso wie für glatte Zielfunktionen. Eine Funktion $f : C \to \mathbb{R}$ auf einem konvexen Definitionsbereich $C$ heißt **konvex**, wenn ihr Graph auf oder unter jeder Sehne zwischen zwei seiner Punkte liegt:

<!--en:9857a3e8-->
Die linke Seite wertet $f$ an einem Punkt der Strecke zwischen $\mathbf{x}$ und $\mathbf{y}$ aus. Die rechte Seite gibt die Höhe der geraden Sehne zwischen $(\mathbf{x}, f(\mathbf{x}))$ und $(\mathbf{y}, f(\mathbf{y}))$ genau über diesem Punkt an. Prüfen wir das an einem Beispiel: Für $f(x) = x^2$ mit $x = 0$, $y = 2$ und $\theta = \tfrac12$ liegt der Graph auf Höhe $f(1) = 1$ und die Sehne auf Höhe $\tfrac12 \cdot 0 + \tfrac12 \cdot 4 = 2$. Tatsächlich gilt $1 \le 2$. Für beliebige $x$, $y$ und $\theta$ ergibt das Ausmultiplizieren der Quadrate folgenden Höhenunterschied zwischen Sehne und Graph:

<!--en:6fdc76d6-->
Damit ist $x^2$ konvex, und der Höhenunterschied wächst quadratisch mit dem Abstand der beiden Punkte. Genau dieser Ausdruck begegnet uns gleich wieder als Varianz in der Jensen-Ungleichung.

<!--en:ba992571-->
Zwei verwandte Begriffe können wir schon jetzt benennen. Gilt die Ungleichung :eqref:`eq_mdl-opt-chord` für $\mathbf{x} \neq \mathbf{y}$ und $\theta \in (0, 1)$ stets strikt, so heißt $f$ **strikt konvex**. Ist $-f$ konvex, so heißt $f$ **konkav**. Eine lineare Funktion $f(x) = x$ ist konvex (und konkav), aber nicht strikt konvex, denn jede Sehne liegt genau auf ihrem Graphen. Auch die Konvexität des Definitionsbereichs $C$ hat einen Grund: Ohne diese Voraussetzung läge der Zwischenpunkt $\theta\mathbf{x} + (1-\theta)\mathbf{y}$ womöglich außerhalb des Bereichs, in dem $f$ definiert ist.

<!--en:6cff0c2c-->
Konvexe Funktionen und konvexe Mengen hängen geometrisch eng zusammen. Der **Epigraph** von $f$ ist der Bereich auf und über dem Graphen, also $\{(\mathbf{x}, t) : t \ge f(\mathbf{x})\}$. Eine Funktion $f$ ist genau dann konvex, wenn ihr Epigraph eine konvexe Menge ist. Denn eine Sehne zwischen zwei Punkten des Epigraphen bleibt genau dann auf oder über dem Graphen, wenn :eqref:`eq_mdl-opt-chord` gilt. Auch über die **Subniveaumengen** $\{\mathbf{x} \in C : f(\mathbf{x}) \le c\}$ überträgt sich Konvexität von Funktionen auf Mengen: Gelten $f(\mathbf{x}) \le c$ und $f(\mathbf{y}) \le c$, so folgt aus der Sehnenungleichung unmittelbar $f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \le \theta c + (1-\theta) c = c$. Deshalb beschreibt jede Nebenbedingung der Form $g(\mathbf{x}) \le 0$ mit konvexem $g$ einen konvexen zulässigen Bereich. :numref:`fig_mdl-opt-epigraph-sublevel`(a) zeigt beide Mengen für eine Parabel.

<!--en:51ced664-->
![Epigraphen und Subniveaumengen. (a) Der Epigraph der konvexen Funktion $f(x) = \tfrac12 x^2 + \tfrac14$ (schattiert und nach oben unbeschränkt) ist eine konvexe Menge. Die Sehne zwischen je zwei seiner Punkte bleibt also im Epigraphen. Der Schnitt auf Höhe $c$ ergibt nach Projektion auf die $x$-Achse die Subniveaumenge $\{f \le c\}$. (b) Jede Subniveaumenge von $f(x) = \sqrt{|x|}$ ist ein Intervall. Der Epigraph ist dennoch nicht konvex, denn der Mittelpunkt der Sehne liegt unter dem Graphen.](../img/mdl-opt-epigraph-sublevel.svg)

<!--en:cfa27d6b-->
Die Umkehrung gilt allerdings nicht. Jede Subniveaumenge von $f(x) = \sqrt{|x|}$ ist ein konvexes Intervall $\{|x| \le c^2\}$, obwohl $f$ selbst nicht konvex ist. Für $x = 0$, $y = 1$ und $\theta = \tfrac12$ liegt der Graph auf Höhe $\sqrt{1/2} \approx 0.71$ und die Sehne darunter auf Höhe $0.5$ (:numref:`fig_mdl-opt-epigraph-sublevel`(b)).

<!--en:2a5e20d5-->
### Lokale Minima sind global

<!--en:11cd2ece-->
Bemerkenswerterweise beantwortet schon die Sehnenbedingung unsere Ausgangsfrage. :numref:`fig_mdl-opt-local-equals-global` stellt die beiden Landschaften gegenüber: Die konvexe Zielfunktion hat ein einziges Einzugsgebiet, das Doppelmuldenpotential zwei konkurrierende Einzugsgebiete.

<!--en:4d0516ac-->
![(a) Bei einer konvexen Zielfunktion erreicht der Gradientenabstieg mit kleiner fester Schrittweite von beiden Startpunkten aus den globalen Minimierer. (b) Beim Doppelmuldenpotential führt dieselbe Aktualisierung von verschiedenen Startpunkten zu verschiedenen Minima. Die Einzugsgebiete grenzen am lokalen Maximum aneinander.](../img/mdl-opt-local-equals-global.svg)

<!--en:14171149-->
**Proposition (lokale Minima sind global).** *Ist $f$ auf einer konvexen Menge $C$ konvex, so ist jeder lokale Minimierer von $f$ auf $C$ ein globaler Minimierer.*

<!--en:14a5c097-->
**Beweis.** Sei $\mathbf{x}^\star$ ein lokaler Minimierer und $\mathbf{y} \in C$ ein beliebiger anderer Punkt. Für hinreichend kleines $\theta > 0$ liegt der Punkt $\mathbf{z}_\theta = (1-\theta)\mathbf{x}^\star + \theta\mathbf{y}$ auf der Strecke in Richtung $\mathbf{y}$ in der Umgebung, in der $\mathbf{x}^\star$ minimal ist. Aus lokaler Minimalität und Sehnenungleichung folgt

<!--en:b27b891b-->
Wir subtrahieren $(1-\theta)f(\mathbf{x}^\star)$ auf beiden Seiten und dividieren durch $\theta > 0$. Damit erhalten wir $f(\mathbf{x}^\star) \le f(\mathbf{y})$. $\blacksquare$

<!--en:6c1143fe-->
Geometrisch ist der Zusammenhang ebenso klar: Gäbe es einen echt besseren Punkt $\mathbf{y}$ irgendwo in $C$, so fiele die Sehne von $\mathbf{x}^\star$ nach $\mathbf{y}$ sofort ab. Dann läge der Graph beliebig nahe am Ausgangspunkt unter dem Wert $f(\mathbf{x}^\star)$, im Widerspruch zur lokalen Minimalität bei $\mathbf{x}^\star$.

<!--en:f62d9851-->
Führen wir nun das Experiment vom Anfang durch, um die Aussage des Satzes numerisch zu prüfen. Wir verwenden denselben Gradientenabstieg (Schrittweite $0.05$, $600$ Schritte) für die Schale $(x - \tfrac12)^2$ und das Doppelmuldenpotential $\phi$. In beiden Fällen starten wir an denselben $500$ Punkten, die wir gleichverteilt aus $[-1.5, 1.5]$ ziehen. Dank der kleinen Schrittweite führt jede Iteration bergab, ohne die mittlere Erhebung zu überspringen. Ein Lauf auf $\phi$ sollte also in der Mulde auf seiner Seite des lokalen Maximums enden. Wir berechnen dieses Maximum als Nullstelle von $\phi'(x) = 4x^3 - 4x + \tfrac12$ und vergleichen zwei Anteile: Wie viele Startpunkte liegen rechts davon, und wie viele Läufe enden in der flacheren rechten Mulde?

<!--en:46aa4227-->
Bei der Schale konvergieren alle $500$ Läufe bis auf Rundungsfehler gegen $x = 0.5$. Beim Doppelmuldenpotential teilen sich die Läufe in zwei Gruppen: Die eine landet bei $x \approx -1.0575$ mit $\phi \approx -0.515$, die andere bei $x \approx 0.9304$ mit $\phi \approx 0.483$ – fast eine ganze Einheit schlechter. Das lokale Maximum zwischen den Einzugsgebieten liegt bei $x \approx 0.127$. Damit liegt ein Anteil von $(1.5 - 0.127)/3 \approx 0.458$ der gleichverteilten Startpunkte rechts davon. Bei wiederholter Zufallsziehung weicht der beobachtete Anteil nur um wenige Prozentpunkte von $46\%$ ab. Das entspricht der Vorhersage bis auf die Stichprobenrauschen bei $500$ Startpunkten.

<!--en:8ec25432-->
Beide Experimente verwenden denselben Optimierer, nur die Zielfunktion ändert sich. Konvexität schließt ein zweites Einzugsgebiet aus. Damit der Gradientenabstieg den Minimierer erreicht, brauchen wir zusätzlich Glattheit und eine hinreichend kleine Schrittweite wie in :numref:`subsec_mdl-gd-smoothness`.

<!--en:a849caf9-->
### Die Bedingung erster Ordnung

<!--en:65c97375-->
Die Sehnenbedingung kommt ohne Ableitungen aus, doch Beweise in der Optimierung arbeiten mit Gradienten. Ist $f$ differenzierbar, so lässt sich Konvexität gleichwertig durch eine Bedingung erster Ordnung ausdrücken: Die tangentiale Hyperebene an jedem Punkt ist eine *globale untere Schranke* für die Funktion,

<!--en:84f06534-->
Prüfen wir dies für $f(x) = x^2$ an der Stelle $x = 1$: Die Tangente lautet $1 + 2(y - 1) = 2y - 1$, und der vertikale Abstand zwischen Graph und Tangente beträgt $y^2 - (2y - 1) = (y - 1)^2 \ge 0$ für jedes $y$. Wie weit :eqref:`eq_mdl-opt-first-order` reicht, lohnt einen Moment der Betrachtung: Ein einziger Gradient an einem einzigen Punkt $\mathbf{x}$ beschränkt $f$ an jedem anderen Punkt des Definitionsbereichs nach unten. Fast alle weiteren Beweise in diesem Abschnitt beruhen auf dieser Ungleichung. :numref:`fig_mdl-opt-chord-above-graph` stellt die ersten beiden Kriterien nebeneinander: Sehnen liegen über dem Graphen, Tangenten darunter.

<!--en:992465aa-->
![Zwei gleichwertige Bedingungen. (a) Die Sehne zwischen zwei Punkten des Graphen liegt über dem Graphen: $f(\theta\mathbf{x}+(1-\theta)\mathbf{y})\le\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})$. (b) Die Tangente an einem beliebigen Punkt liegt unter dem Graphen: $f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$.](../img/mdl-opt-chord-above-graph.svg)

<!--en:995af2b5-->
### Die Bedingung zweiter Ordnung

<!--en:237024ae-->
Für glatte Zielfunktionen lässt sich Konvexität meist am schnellsten mit einem Matrixkriterium nachweisen. Ist $f$ auf einem **offenen** konvexen Definitionsbereich $C$ zweimal differenzierbar, so ist $f$ genau dann konvex, wenn die Hesse-Matrix überall positiv semidefinit ist:

<!--en:767c0037-->
In einer Dimension ergibt sich die bekannte Regel $f''(x) \ge 0$: Die Kurve krümmt sich nach oben. In $d$ Dimensionen heißt das, dass sich jeder eindimensionale Schnitt durch $\mathbf{x}$ nach oben krümmt. Denn $\mathbf{v}^\top \nabla^2 f(\mathbf{x})\, \mathbf{v}$ ist die zweite Ableitung von $f$ entlang der Geraden in Richtung $\mathbf{v}$ (:numref:`sec_mdl-multivariable_calculus`).

<!--en:11f5ca83-->
Mit dem Matrixkriterium lassen sich drei wichtige Beispiele sofort behandeln. Erstens hat die quadratische Funktion $\tfrac12\mathbf{x}^\top \mathbf{A} \mathbf{x} + \mathbf{b}^\top\mathbf{x}$ mit symmetrischem $\mathbf{A}$ die konstante Hesse-Matrix $\mathbf{A}$. Sie ist also genau dann konvex, wenn $\mathbf{A} \succeq 0$ gilt. Zweitens hat die Verlustfunktion der kleinsten Quadrate $\tfrac12\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ für eine Designmatrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ ($n$ Beispiele, $d$ Gewichte) die Hesse-Matrix $\mathbf{X}^\top \mathbf{X}$. Wegen $\mathbf{v}^\top \mathbf{X}^\top\mathbf{X} \mathbf{v} = \|\mathbf{X}\mathbf{v}\|^2 \ge 0$ für jedes $\mathbf{v}$ ist das Problem der kleinsten Quadrate stets konvex (:numref:`subsec_mdl-psd`). Drittens sind die skalaren Funktionen $e^x$, $-\log x$ und $x \log x$ auf ihren jeweiligen Definitionsbereichen konvex, da ihre zweiten Ableitungen $e^x$, $1/x^2$ und $1/x$ überall positiv sind.

<!--en:16be4dec-->
Warum verlangen wir, dass der Definitionsbereich $C$ offen ist? Auf einem niederdimensionalen Definitionsbereich zählen nur Richtungen, die in $C$ bleiben. Beispielsweise ist $x^2 - y^2$ auf der Geraden $y = 0$ konvex, obwohl die volle zweidimensionale Hesse-Matrix indefinit ist.

<!--en:cd477d12-->
Zeigen wir nun, dass alle drei Kriterien genau dieselben Funktionen kennzeichnen. Der Beweis hat vier kurze Teile. Besonders der zweite Teil ist einprägsam: Wenden wir die Tangentenschranke am *Zwischenpunkt* einer Sehne an, heben sich die Gradiententerme auf. Diesen Kunstgriff verwenden wir auch beim Beweis der Jensen-Ungleichung.

<!--en:986a4d51-->
**Proposition (Äquivalenz der drei Kriterien).** *Sei $f$ auf einem offenen konvexen Definitionsbereich $C$ differenzierbar. Dann sind die Sehnenbedingung :eqref:`eq_mdl-opt-chord` und die Bedingung erster Ordnung :eqref:`eq_mdl-opt-first-order` äquivalent. Ist $f$ zweimal differenzierbar, so sind beide äquivalent zu $\nabla^2 f(\mathbf{x}) \succeq 0$ auf $C$.*

<!--en:e4bcfe29-->
**Beweis.** *(Sehnenbedingung $\Rightarrow$ Bedingung erster Ordnung.)* Wir halten $\mathbf{x}, \mathbf{y} \in C$ und $\theta \in (0, 1]$ fest. Am Zwischenpunkt $\mathbf{x} + \theta(\mathbf{y} - \mathbf{x})$ wenden wir die Sehnenungleichung an, subtrahieren auf beiden Seiten $f(\mathbf{x})$ und dividieren durch $\theta$:

<!--en:d53a5641-->
Für $\theta \downarrow 0$ nähert sich die linke Seite der Richtungsableitung $\nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})$. Damit folgt :eqref:`eq_mdl-opt-first-order`.

<!--en:e0c7e4fb-->
*(Bedingung erster Ordnung $\Rightarrow$ Sehnenbedingung.)* Wir setzen $\mathbf{z} = \theta\mathbf{x} + (1-\theta)\mathbf{y}$ und wenden die Tangentenschranke bei $\mathbf{z}$ zweimal an, einmal in Richtung jedes Endpunkts:

<!--en:82d30ac3-->
Wir multiplizieren die erste Ungleichung mit $\theta$, die zweite mit $1 - \theta$ und addieren. Die linearen Terme ergeben zusammen $\nabla f(\mathbf{z})^\top(\theta\mathbf{x} + (1-\theta)\mathbf{y} - \mathbf{z}) = 0$. Es bleibt :eqref:`eq_mdl-opt-chord`.

<!--en:477e3ca3-->
*(Bedingung zweiter Ordnung $\Rightarrow$ Bedingung erster Ordnung.)* Wir beschränken $f$ auf die Verbindungsstrecke zwischen $\mathbf{x}$ und $\mathbf{y}$, indem wir $g(t) = f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))$ definieren. Die zweite Ableitung lautet $g''(t) = (\mathbf{y} - \mathbf{x})^\top \nabla^2 f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))\,(\mathbf{y} - \mathbf{x}) \ge 0$. Nach dem Satz von Taylor mit Lagrange-Restglied (:numref:`sec_mdl-single_variable_calculus`) gilt $g(1) = g(0) + g'(0) + \tfrac12 g''(\tau)$ für ein $\tau \in (0, 1)$. Lassen wir das nichtnegative Restglied weg, erhalten wir :eqref:`eq_mdl-opt-first-order`.

<!--en:9e1372f8-->
*(Bedingung erster Ordnung $\Rightarrow$ Bedingung zweiter Ordnung.)* Nehmen wir zum Widerspruch an, es gelte $\mathbf{v}^\top \nabla^2 f(\mathbf{x})\, \mathbf{v} < 0$ an einem Punkt $\mathbf{x} \in C$ in einer Richtung $\mathbf{v}$. Entwickeln wir $f$ entlang $\mathbf{v}$, so erhalten wir

<!--en:27160f2f-->
Für hinreichend kleines $t > 0$ überwiegt der negative quadratische Term das Restglied $o(t^2)$. Also fällt $f(\mathbf{x} + t\mathbf{v})$ strikt unter den Tangentenwert $f(\mathbf{x}) + t \nabla f(\mathbf{x})^\top\mathbf{v}$. Das widerspricht :eqref:`eq_mdl-opt-first-order`. $\blacksquare$

<!--en:ef309e68-->
### Numerische Überprüfung der drei Kriterien

<!--en:1d016cbd-->
Wenden wir alle drei Kriterien numerisch auf zwei Verlustfunktionen an: auf die Zielfunktion der kleinsten Quadrate $f(\mathbf{w}) = \tfrac12\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ mit $n = 20$ zufälligen Beispielen und $d = 2$ Gewichten sowie auf das Doppelmuldenpotential $\phi$ aus unserem ersten Experiment. Für die kleinsten Quadrate kennen wir bereits die genauen Ausdrücke für mögliche Verletzungen. Multiplizieren wir die Quadrate wie bei $x^2$ aus, erhalten wir als Sehnendifferenz (Graph minus Sehne) $-\tfrac12\theta(1-\theta)\|\mathbf{X}(\mathbf{u} - \mathbf{v})\|^2$ und als Tangentendifferenz (Tangente minus Graph) $-\tfrac12\|\mathbf{X}(\mathbf{v} - \mathbf{u})\|^2$. Beide sind nichtpositiv. Für jede Verlustfunktion zieht der folgende Code $1{,}000$ zufällige Sehnen und Tangenten in $[-1.5, 1.5]^d$, gibt für jedes Kriterium die größte Verletzung aus und prüft den kleinsten Eigenwert der Hesse-Matrix über die gezogenen Punkte.

<!--en:8db38376-->
Bei den kleinsten Quadraten fällt keine der gezogenen Sehnen unter den Graphen, keine Tangente steigt darüber, und die Eigenwerte der Hesse-Matrix bleiben positiv. Genau das sagt die Rechnung voraus. Das Doppelmuldenpotential verletzt alle drei Kriterien: Die ungünstigste Sehne zwischen Punkten in den beiden Tälern liegt fast eine Einheit unter dem Graphen. Die Tangenten schneiden die Kurve, und die zweite Ableitung fällt an der mittleren Erhebung auf nahezu $-4$, wo $\phi''(0) = -4$ gilt. Hier zeigt sich eine Asymmetrie: Zufallsstichproben können Konvexität *widerlegen* – schon eine einzige ungeeignete Sehne liefert für $\phi$ ein Gegenbeispiel –, aber niemals *beweisen*. Bei den kleinsten Quadraten haben wir lediglich keine Verletzung gefunden. Den Beweis liefert das Kriterium für die Hesse-Matrix, denn $\mathbf{X}^\top \mathbf{X}$ ist an jedem Punkt $\mathbf{w}$ dieselbe positiv semidefinite Matrix.

<!--en:f2f22fcf-->
### Strikte und starke Konvexität

<!--en:a6b7b66f-->
Konvexität allein erlaubt beliebig flache Abschnitte. Dort kann der Gradientenabstieg beliebig langsam vorankommen. Schnelle Konvergenz setzt hier eine Krümmung voraus, die überall durch eine positive Konstante nach unten beschränkt ist. Drei einfache Funktionen veranschaulichen die Abstufung in :numref:`fig_mdl-opt-convexity-hierarchy`(a). Die Gerade $f(x) = x$ ist konvex, aber nicht strikt konvex, denn ihre Sehnen liegen auf ihrem Graphen. Die Funktion vierten Grades $x^4$ ist strikt konvex: Ihre zweite Ableitung $12x^2$ verschwindet nur bei $0$, sodass ihre Steigung überall streng wächst. Wegen $f''(0) = 0$ ist sie am Minimierer dennoch völlig flach. Die quadratische Funktion $x^2$ hingegen hat an jedem Punkt die konstante Krümmung $f''(x) = 2$.

<!--en:ffcfe37f-->
Starke Konvexität fasst diese Beobachtung in eine quantitative Schranke. Für $\mu > 0$ heißt eine Funktion $f$ **$\mu$-stark konvex**, wenn nach Abzug einer quadratischen Funktion der Krümmung $\mu$ eine konvexe Funktion bleibt, wenn also $h(\mathbf{x}) = f(\mathbf{x}) - \tfrac{\mu}{2}\|\mathbf{x}\|^2$ konvex ist. Ist $f$ differenzierbar, wenden wir die Bedingung erster Ordnung auf $h$ an. Mit dem Gradienten $\nabla f(\mathbf{x}) - \mu\mathbf{x}$ und der Identität $\|\mathbf{y}\|^2 - \|\mathbf{x}\|^2 - 2\mathbf{x}^\top(\mathbf{y} - \mathbf{x}) = \|\mathbf{y} - \mathbf{x}\|^2$ erhalten wir die äquivalente Ungleichung

<!--en:a585a5b1-->
Geometrisch muss der Graph von $f$ über seiner Tangente an der Stelle $\mathbf{x}$ und sogar über einer dort tangential anliegenden Parabel der Krümmung $\mu$ liegen. Die Bedingung zweiter Ordnung für $h$ liefert das Matrixkriterium: Ein zweimal differenzierbares $f$ ist genau dann $\mu$-stark konvex, wenn überall $\nabla^2 f(\mathbf{x}) \succeq \mu\mathbf{I}$ gilt. Dabei ist $\mathbf{I}$ die Einheitsmatrix. Also ist $x^2$ zwar $2$-stark konvex, doch $x^4$ ist in keiner Umgebung von $0$ stark konvex, denn es gilt $f''(0) = 0 < \mu$ für jedes $\mu > 0$. Daraus folgt auch ein praktischer Vorteil: Addieren wir den Ridge-Strafterm $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ zu einer beliebigen konvexen Verlustfunktion, so ist die Summe sofort $\lambda$-stark konvex. Ziehen wir nämlich $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ wieder ab, erhalten wir die ursprüngliche konvexe Verlustfunktion. Gewichtszerfall verkleinert also die Parameter und garantiert zugleich starke Konvexität.

<!--en:93c9238c-->
![(a) Konvex, strikt konvex, stark konvex: Bei $x$ liegen die Sehnen auf dem Graphen, $x^4$ ist am Ursprung flach, $x^2$ hat überall Krümmung. (b) Eine $\mu$-stark konvexe, $L$-glatte Funktion, hier $f(x)=\tfrac12x^2+3\log\cosh x$, liegt zwischen zwei Parabeln, die sie bei $x_0=1$ berühren. Ihre Krümmungen sind $\mu=1$ und $L=4$, ihr Verhältnis ist $\kappa$.](../img/mdl-opt-convexity-hierarchy.svg)

<!--en:d5ffa2c7-->
Starke Konvexität ist das natürliche Gegenstück zur Glattheit. Erinnern wir uns an :numref:`subsec_mdl-gd-smoothness`: Eine Funktion $f$ heißt **$L$-glatt**, wenn ihr Gradient $L$-Lipschitz-stetig ist, also $\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \le L\|\mathbf{x} - \mathbf{y}\|$ gilt. Für eine konvexe, zweimal differenzierbare Funktion bedeutet das $\nabla^2 f \preceq L\mathbf{I}$. Glattheit liefert eine quadratische obere Schranke für den Graphen, $f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \tfrac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$, starke Konvexität eine quadratische untere Schranke :eqref:`eq_mdl-opt-strong-convexity`. Zusammen schließen sie $f$ zwischen zwei tangential anliegenden Parabeln der Krümmungen $\mu$ und $L$ ein. :numref:`fig_mdl-opt-convexity-hierarchy`(b) zeigt dies für $f(x) = \tfrac12 x^2 + 3\log\cosh x$, dessen zweite Ableitung $f''(x) = 1 + 3/\cosh^2 x$ zwischen $\mu = 1$ und $L = 4$ liegt.

<!--en:fed5d6cb-->
Das Verhältnis $\kappa = L/\mu$ heißt **Konditionszahl**. Es verallgemeinert das Verhältnis der Eigenwerte, das wir für quadratische Funktionen in :numref:`subsec_mdl-quadratic-model` kennengelernt haben. In unserem durchgehenden Beispiel $\tfrac12(x_1^2 + 1000\,x_2^2)$ sind die extremen Krümmungen $\mu = 1$ und $L = 1000$. Damit ist $\kappa = 10^3$.

<!--en:c8455b4b-->
### Subgradienten

<!--en:69d13674-->
Wichtige Verlustfunktionen wie die $\ell_1$-Norm $\|\mathbf{x}\|_1$ und der Hinge-Verlust $\max(0, 1 - z)$ haben Knicke. Dort gibt es keinen Gradienten, und :eqref:`eq_mdl-opt-first-order` lässt sich in dieser Form nicht angeben. Wir behalten die Tangentenschranke bei und machen die *Aufgabe* des Gradienten zur Definition: Er liefert eine globale lineare untere Schranke. Ein Vektor $\mathbf{g}$ ist ein **Subgradient** von $f$ an der Stelle $\mathbf{x}$, wenn

<!--en:46e4a657-->
Die Menge aller Subgradienten an der Stelle $\mathbf{x}$ heißt **Subdifferential**. Wir schreiben dafür $\partial f(\mathbf{x})$. Überall dort, wo eine konvexe Funktion $f$ differenzierbar ist, besteht das Subdifferential aus einem einzigen Vektor: $\{\nabla f(\mathbf{x})\}$. An einem Knick umfasst es dagegen eine ganze Menge von Steigungen stützender Geraden (:numref:`fig_mdl-opt-subgradient-fan`). Für $f(x) = |x|$ an der Stelle $x = 0$ erfüllt jede Steigung $g \in [-1, 1]$ die Ungleichung $|y| \ge g y$ für alle $y$. Also gilt $\partial f(0) = [-1, 1]$. Ähnlich erhalten wir für den Hinge-Verlust $\max(0, 1 - z)$ am Knick $z = 1$ das Subdifferential $\partial f(1) = [-1, 0]$.

<!--en:1c09a385-->
![Am Knick von $f(x) = |x|$ existiert kein Gradient. Das Subdifferential ist dort eine Menge: Jede Steigung $g \in \left(-1, 1\right)$ definiert eine Stützgerade unter der V-förmigen Kurve, und die beiden extremen Steigungen $\pm 1$ gehören zu den Ästen selbst. Also ist $\partial f(0)$ das gesamte Intervall von $-1$ bis $1$. Die Gerade mit Steigung null (orange) liefert das Optimalitätskriterium $0 \in \partial f(0)$: Der Knick ist auch ohne Differenzierbarkeit nachweislich ein Minimum.](../img/mdl-opt-subgradient-fan.svg)

<!--en:656c7b8d-->
Damit haben wir sofort ein Optimalitätskriterium ohne Ableitungen: $\mathbf{x}^\star$ minimiert $f$ genau dann, wenn $\mathbf{0} \in \partial f(\mathbf{x}^\star)$ gilt. Der Beweis folgt unmittelbar aus der Definition: Setzen wir $\mathbf{g} = \mathbf{0}$ in :eqref:`eq_mdl-opt-subgradient` ein, erhalten wir $f(\mathbf{y}) \ge f(\mathbf{x}^\star)$ für alle $\mathbf{y}$. Genau das definiert einen globalen Minimierer.

<!--en:e58b5bd5-->
Besitzt eine konvexe Funktion immer einen Subgradienten? An jedem inneren Punkt ihres Definitionsbereichs ja. Darauf stützen wir den Beweis der Jensen-Ungleichung. In einer Dimension ist der Grund leicht zu sehen. Für drei beliebige Punkte $a < b < c$ in einem offenen Intervall, auf dem $f$ konvex ist, ergibt die Sehnenungleichung bei $b$ nach Umstellen

<!--en:c948b032-->
Die Sehnensteigungen nehmen also von links nach rechts nicht ab. Folglich ist der Differenzenquotient $q(x) = (f(x) - f(x_0))/(x - x_0)$ für jeden inneren Punkt $x_0$ monoton wachsend in $x$. Seine beiden einseitigen Grenzwerte, die links- und rechtsseitigen Ableitungen $f'_-(x_0) \le f'_+(x_0)$, existieren also und sind endlich. Jede Steigung $g \in [f'_-(x_0), f'_+(x_0)]$ ist ein Subgradient: Für $y > x_0$ gilt $q(y) \ge f'_+(x_0) \ge g$. Umgestellt ergibt das $f(y) \ge f(x_0) + g(y - x_0)$. Der Fall $y < x_0$ folgt symmetrisch.

<!--en:4db7a88a-->
Die Aussage gilt auch in $d$ Dimensionen: An jedem inneren Punkt ihres Definitionsbereichs besitzt eine konvexe Funktion mindestens einen Subgradienten. Unter ihren Graphen passt also stets eine stützende Hyperebene (:citet:`Rockafellar.1970`, :citet:`Boyd.Vandenberghe.2004`, Abschnitt 2.5). Wir verwenden dieses Ergebnis ohne Beweis. Als Suchrichtung leistet ein Subgradient allerdings deutlich weniger als ein Gradient: Ein Schritt entlang $-\mathbf{g}$ muss $f$ keineswegs verringern. Am Minimum $x = 0$ von $|x|$ führt ein Schritt mit dem zulässigen Subgradienten $g = \tfrac12$ zu $-\eta/2$ und erhöht den Verlust. Deshalb benötigt das Subgradientenverfahren abnehmende Schrittweiten und konvergiert langsam. Der Fehler fällt nur wie $1/\sqrt{k}$.

<!--en:06502333-->
## Folgen der Konvexität

<!--en:e1c57cd9-->
Im weiteren Verlauf des Buchs nutzen wir drei wichtige Folgen der Konvexität. Alle drei haben denselben geometrischen Ursprung: Eine Stützgerade oder Stützhyperebene beschränkt die Funktion überall nach unten. Mitteln wir diese Schranke über eine Zufallsvariable, erhalten wir die Jensen-Ungleichung. Werten wir sie an einem stationären Punkt aus, weist sie globale Optimalität nach und verschärft damit das Ergebnis, das :numref:`subsec_mdl-local-global` allein aus der Sehnenbedingung ableitet. Summieren wir sie über die Schritte des Gradientenabstiegs, erhalten wir aus dem lokalen Fortschritt globale Konvergenzraten. Die Jensen-Ungleichung lässt sich unabhängig davon lesen. Wenn es Ihnen nur um Optimierung geht, können Sie direkt zum nächsten Unterabschnitt springen.

<!--en:89da2f99-->
### Die Jensen-Ungleichung

<!--en:7cef0fc3-->
Bei erwarteten Verlusten, mittleren Log-Likelihoods und KL-Divergenzen stellt sich dieselbe Frage: Wie verhält sich der Funktionswert am Mittelwert zum Mittelwert der Funktionswerte? Schon ein Beispiel mit zwei Punkten zeigt die Richtung der Ungleichung. Die Zufallsvariable $Z$ nehme die Werte $0$ und $2$ mit Wahrscheinlichkeit $\tfrac12$ an, und es sei $f(x) = x^2$. Werten wir $f$ am Mittelwert aus, erhalten wir $f(\mathbb{E}[Z]) = f(1) = 1$. Mitteln wir dagegen $f$ über die möglichen Ergebnisse, erhalten wir $\mathbb{E}[f(Z)] = \tfrac12 \cdot 0 + \tfrac12 \cdot 4 = 2$. Die Differenz von $1$ ist die Varianz von $Z$. Tatsächlich ist für $f(x) = x^2$ die Differenz $\mathbb{E}[Z^2] - (\mathbb{E}[Z])^2$ immer die Varianz. Schauen wir noch einmal auf :numref:`fig_mdl-opt-chord-above-graph`(a): Für eine Zweipunktverteilung mit Gewichten $\theta$ und $1 - \theta$ beschreibt die Sehnenungleichung genau diesen Vergleich. Die Differenz $\theta(1-\theta)(x - y)^2$, die wir für $x^2$ berechnet haben, ist die Varianz dieser Verteilung.

<!--en:865080bc-->
Das Argument gilt auch für mehr als zwei Punkte. Wenden wir :eqref:`eq_mdl-opt-chord` induktiv an (Aufgabe :exref:`ex_mdl-convexity-prove-jensen-s-finite`), erhalten wir die endliche Form

<!--en:1c0a0085-->
Nichtnegative Gewichte mit Summe eins definieren eine diskrete Wahrscheinlichkeitsverteilung auf den Punkten $\mathbf{x}_i$. Es liegt also nahe, den endlichen Mittelwert durch einen beliebigen Erwartungswert zu ersetzen.

<!--en:ec16afbb-->
**Proposition (Jensen-Ungleichung).** *Sei $f$ konvex auf einer konvexen Menge $C \subseteq \mathbb{R}^d$. Sei ferner $\mathbf{Z}$ ein Zufallsvektor mit Werten in $C$, für den $\mathbb{E}\|\mathbf{Z}\| < \infty$ gilt und dessen Mittelwert $\mathbf{m} = \mathbb{E}[\mathbf{Z}]$ im Inneren von $C$ liegt. Dann gilt*

<!--en:3485b54c-->
*Ist $f$ strikt konvex, so gilt Gleichheit genau dann, wenn $\mathbf{Z} = \mathbf{m}$ mit Wahrscheinlichkeit eins gilt.*

<!--en:4cb30ec0-->
**Beweis.** Da $\mathbf{m}$ im Inneren von $C$ liegt, besitzt $f$ dort mindestens einen Subgradienten $\mathbf{g} \in \partial f(\mathbf{m})$. Die Stützhyperebene bei $\mathbf{m}$ beschränkt $f$ für alle möglichen Ergebnisse zugleich nach unten: Nach :eqref:`eq_mdl-opt-subgradient` gilt

<!--en:7ff0f251-->
Bilden wir auf beiden Seiten den Erwartungswert. Wegen $\mathbb{E}[\mathbf{Z} - \mathbf{m}] = \mathbf{0}$ verschwindet der lineare Term, und es bleibt $\mathbb{E}[f(\mathbf{Z})] \ge f(\mathbf{m})$. Ist $f$ strikt konvex, so berührt die Stützhyperebene den Graphen *nur* bei $\mathbf{m}$. Berührte sie ihn auch bei einem $\mathbf{x} \neq \mathbf{m}$, so ergäbe sich aus strikter Konvexität $f\bigl(\tfrac{\mathbf{x} + \mathbf{m}}{2}\bigr) < \tfrac12 f(\mathbf{x}) + \tfrac12 f(\mathbf{m})$. Am Mittelpunkt läge der Graph dann strikt unter der Stützhyperebene, ein Widerspruch. Somit gilt die punktweise Ungleichung strikt, sobald $\mathbf{Z} \neq \mathbf{m}$ gilt. Gleichheit der Erwartungswerte setzt daher voraus, dass $\mathbf{Z} = \mathbf{m}$ mit Wahrscheinlichkeit eins gilt. $\blacksquare$

<!--en:8ec1afc7-->
Aus derselben punktweisen Schranke folgt außerdem, dass eine integrierbare Zufallsvariable $f(\mathbf{Z})$ nach unten beschränkt. Somit ist $\mathbb{E}[f(\mathbf{Z})]$ stets wohldefiniert (möglicherweise gleich $+\infty$, dann gilt die Ungleichung unmittelbar). Anschaulich kann sich der Erwartungswert einer konvexen Funktion nur erhöhen, wenn wir die Zufallsvariable bei festem Mittelwert stärker streuen. Für eine konkave Funktion $f$ kehrt sich die Ungleichung um: $\mathbb{E}[f(\mathbf{Z})] \le f(\mathbb{E}[\mathbf{Z}])$.

<!--en:12ebdb6d-->
Im maschinellen Lernen dient die Jensen-Ungleichung am häufigsten zum Beweis, dass die KL-Divergenz niemals negativ ist. In :numref:`sec_mdl-information_theory` begegnet uns dasselbe Ergebnis als Gibbs-Ungleichung.

<!--en:9f86bada-->
**Korollar (Nichtnegativität der KL-Divergenz).** *Für Wahrscheinlichkeitsverteilungen $p$ und $q$ auf demselben endlichen Alphabet gelte $q(x) > 0$ überall dort, wo $p(x) > 0$ gilt. Dann gilt*

<!--en:3dbf6202-->
*mit Gleichheit genau dann, wenn $p = q$ gilt.*

<!--en:5e9dad9b-->
**Beweis.** Wir ziehen $Z \sim p$ und wenden die Jensen-Ungleichung auf die strikt konvexe Funktion $-\log$ und die positive Zufallsvariable $R = q(Z)/p(Z)$ an:

<!--en:3c3d5ca9-->
Im letzten Schritt verwenden wir $\sum_{x : p(x) > 0} q(x) \le 1$. Nach dem Gleichheitskriterium der Jensen-Ungleichung gilt Gleichheit genau dann, wenn $R$ mit Wahrscheinlichkeit eins konstant ist und $q$ außerhalb des Trägers von $p$ keine Masse hat, also $q = c\,p$ mit $c = 1$ gilt. $\blacksquare$

<!--en:b001c2ef-->
Wenden wir dieselbe Idee auf die konkave Funktion $\log$ an, erhalten wir die klassische **Ungleichung vom arithmetischen und geometrischen Mittel**: Für beliebige positive Zahlen $x_1, \ldots, x_N$ folgt aus der Konkavität $\log\bigl(\tfrac1N \sum_i x_i\bigr) \ge \tfrac1N \sum_i \log x_i$. Exponentieren wir beide Seiten, erhalten wir

<!--en:7a07b059-->
mit Gleichheit genau dann, wenn alle $x_i$ gleich sind, denn $\log$ ist streng konkav. Auch die untere Evidenzschranke (ELBO) aus :numref:`sec_mdl-latent-em-elbo` ergibt sich direkt aus der Jensen-Ungleichung für $\log$. Sie beschränkt den Logarithmus eines Erwartungswerts nach unten.

<!--en:5f110595-->
Prüfen wir die Differenz in der Jensen-Ungleichung und den Gleichheitsfall der KL-Divergenz numerisch. Erstens können wir für $Z \sim \mathcal{N}(0, 1)$ und $f(x) = e^x$ beide Seiten in geschlossener Form berechnen: $\mathbb{E}[e^Z] = \sqrt{e} \approx 1.649$ beziehungsweise $e^{\mathbb{E}[Z]} = 1$. Wie gut trifft die Monte-Carlo-Schätzung diese Werte? Zweitens mischen wir $q$ in $p$ gemäß $(1 - \varepsilon)p + \varepsilon q$. Wie schnell geht dabei die KL-Divergenz gegen null: linear in $\varepsilon$ oder schneller?

<!--en:1ca47878-->
Die Monte-Carlo-Mittelwerte liegen nahe bei $1.65$ beziehungsweise $1.00$. Mehr Stichproben verringern das Schätzrauschen, doch die durch die Jensen-Ungleichung bedingte Differenz bleibt bestehen. Im KL-Experiment geht die Divergenz wesentlich schneller gegen null als $\varepsilon$: Verkleinern wir $\varepsilon$ um den Faktor zehn, so sinkt der Median der Divergenz etwa um den Faktor hundert. Die Entwicklung des Logarithmus erklärt das Verhalten: Es gilt $D_{\textrm{KL}}(p \,\|\, (1-\varepsilon)p + \varepsilon q) \approx \tfrac{\varepsilon^2}{2}\sum_x (q(x) - p(x))^2/p(x)$, sobald $\varepsilon\,q(x)/p(x)$ für jedes $x$ klein ist. (Paare, bei denen $p$ einem Symbol eine winzige Wahrscheinlichkeit zuweist, erreichen diesen quadratischen Bereich zuletzt. Deshalb geben wir den Median an und nicht das Maximum.) Die Annäherung an den Gleichheitsfall verläuft quadratisch. Gleichheit tritt erst im Grenzfall $q \to p$ ein.

<!--en:5a805be6-->
### Lokale und globale Minima

<!--en:ef4ab970-->
Wir haben bereits bewiesen, dass jedes lokale Minimum einer konvexen Funktion global ist (:numref:`subsec_mdl-local-global`). Drei verwandte Aussagen folgen ebenso leicht, jede in einer Zeile. Erstens ist die Menge aller globalen Minimierer konvex: Nehmen $\mathbf{x}$ und $\mathbf{y}$ beide den Minimalwert $f^\star$ an, so beträgt der Funktionswert nach der Sehnenungleichung an jedem Punkt dazwischen höchstens $f^\star$. Zweitens folgt für ein differenzierbares konvexes $f$ auf einer offenen Menge aus $\nabla f(\mathbf{x}) = \mathbf{0}$ unmittelbar $f(\mathbf{x}) = f^\star$. Denn mit Gradient null lautet die Bedingung erster Ordnung $f(\mathbf{y}) \ge f(\mathbf{x})$ für alle $\mathbf{y}$. Drittens garantiert ein kleiner Gradient eine kleine Differenz zum Optimalwert: Wenden wir die Bedingung erster Ordnung in Richtung eines Minimierers $\mathbf{x}^\star$ an, ergibt sich $f(\mathbf{x}) - f^\star \le \nabla f(\mathbf{x})^\top(\mathbf{x} - \mathbf{x}^\star) \le \|\nabla f(\mathbf{x})\|\,\|\mathbf{x} - \mathbf{x}^\star\|$. Nach dem Stationaritätsergebnis aus :numref:`subsec_mdl-gd-smoothness` treibt der Gradientenabstieg mit $\eta = 1/L$ die kleinste Gradientennorm gegen null. Für konvexe Zielfunktionen erhalten wir damit eine Garantie für den *Verlustwert* selbst, sofern der Abstand der Iterierten zu $\mathbf{x}^\star$ beschränkt bleibt. In :numref:`subsec_mdl-convergence-rates` beweisen wir diese Beschränktheit und bestimmen die genaue Rate.

<!--en:d8e6b35a-->
Zwei Eigenschaften garantiert gewöhnliche Konvexität *nicht*: Ein Minimierer muss weder eindeutig sein (eine konvexe Funktion kann auf einer ganzen Strecke konstant sein) noch überhaupt existieren ($e^x$ nähert sich $0$ für $x \to -\infty$, ohne den Wert je anzunehmen). Strikte Konvexität schließt mehrere Minimierer aus. Starke Konvexität garantiert Existenz und Eindeutigkeit.

<!--en:f07bf119-->
**Proposition (Eindeutigkeit).** *Eine strikt konvexe Funktion $f$ hat höchstens einen Minimierer. Eine stetige, $\mu$-stark konvexe Funktion $f$ auf einer nichtleeren abgeschlossenen konvexen Menge $C$ mit nichtleerem Inneren besitzt genau einen.*

<!--en:10ba11d5-->
**Beweis.** Angenommen, $\mathbf{x} \neq \mathbf{y}$ wären beide globale Minimierer mit Wert $f^\star$. Nach der strikten Sehnenungleichung für $\theta = \tfrac12$ gälte an ihrem Mittelpunkt $f\bigl(\tfrac{\mathbf{x} + \mathbf{y}}{2}\bigr) < f^\star$, ein Widerspruch. Da starke Konvexität strikte Konvexität impliziert, bleibt nur die Existenz zu zeigen. Wir beweisen sie für ein differenzierbares $f$ auf $C = \mathbb{R}^d$. Das deckt die Verlustfunktionen ohne Nebenbedingungen in diesem Buch ab. Im allgemeinen Fall führt man dieselben Schritte mit einem Subgradienten von $f - \tfrac{\mu}{2}\|\cdot\|^2$ an einem inneren Punkt von $C$ aus. Wir wählen einen beliebigen Startpunkt $\mathbf{x}_0$. In :eqref:`eq_mdl-opt-strong-convexity` beträgt der lineare Term mindestens $-\|\nabla f(\mathbf{x}_0)\|\,\|\mathbf{y} - \mathbf{x}_0\|$. Daher gilt $f(\mathbf{y}) > f(\mathbf{x}_0)$, sobald $\|\mathbf{y} - \mathbf{x}_0\| > 2\|\nabla f(\mathbf{x}_0)\|/\mu$ erfüllt ist. Die Subniveaumenge $\{\mathbf{y} : f(\mathbf{y}) \le f(\mathbf{x}_0)\}$ ist also beschränkt und wegen der Stetigkeit von $f$ auch abgeschlossen. Jede stetige Funktion nimmt auf einer nichtleeren abgeschlossenen und beschränkten Menge ihr Minimum an. Das Minimum auf dieser Subniveaumenge ist zugleich das globale Minimum auf $\mathbb{R}^d$. $\blacksquare$

<!--en:53e948e4-->
Die technischen Voraussetzungen an $C$ und die Stetigkeit schließen lediglich pathologische Funktionen aus, die am Rand ihres Definitionsbereichs nach oben springen. Solche Funktionen treten bei den Verlustfunktionen in diesem Buch nicht auf. Dass wie bei $e^x$ kein Minimierer existiert, kommt dagegen auch in der Praxis vor: Aufgabe :exref:`ex_mdl-convexity-minimizer-existence` zeigt diesen Zusammenhang für die logistische Regression mit linear separierbaren Daten.

<!--en:268ca032-->
### Konvergenzraten

<!--en:ff0caaf1-->
Konvexität sagt uns, *wo* der Gradientenabstieg endet, aber nicht, *wie schnell* er dorthin gelangt. Über die Geschwindigkeit sagt gewöhnliche Konvexität überraschend wenig aus. Vergleichen wir $f(x) = x^2$ und $f(x) = x^4$: Beide sind strikt konvex mit eindeutigem Minimierer $0$, aber nur $x^2$ ist stark konvex. Wir führen für beide Funktionen denselben Gradientenabstieg aus ($\eta = \tfrac18$, Start bei $x_0 = 1$) und beobachten über $10^4$ Schritte, wie unterschiedlich sich die beiden Läufe entwickeln.

<!--en:b0ddb527-->
Der Unterschied ist groß. Bei $x^2$ multipliziert jeder Schritt die Iterierte mit dem konstanten Faktor $1 - 2\eta = 0.75$. Unter $10^{-6}$ fällt sie bereits nach $49$ Schritten. Der Verlust fällt fast sofort unter den dargestellten Bereich. Bei $x^4$ dagegen multipliziert jeder Schritt $x_k$ mit $1 - 4\eta x_k^2$. Dieser Kontraktionsfaktor rückt immer näher an $1$ heran, während sich die Iterierte dem flachen Tal bei $0$ nähert. Die ausgegebenen Produkte $x_k\sqrt{k}$ nähern sich $1$. Also nimmt $x_k$ nur wie $1/\sqrt{k}$ ab und beträgt noch $0.01$ nach $10{,}000$ Schritten. In doppelt logarithmischer Darstellung beschreibt die Differenz zum minimalen Verlust $x_k^4$ deshalb eine Gerade mit Steigung $-2$. Strikte Konvexität schließt mehrere Minimierer aus, garantiert aber keine schnelle Konvergenz. Für eine geometrische Konvergenzgarantie brauchen wir $\mu > 0$.

<!--en:ccc25186-->
Die nächsten beiden Propositionen verallgemeinern diesen Vergleich: Bei jeder glatten, stark konvexen Funktion konvergiert der Gradientenabstieg geometrisch wie bei $x^2$. Ohne $\mu > 0$ können wir nur eine Abnahme wie $O(1/k)$ garantieren. Beide Beweise beginnen mit dem **Abstiegslemma** aus :numref:`subsec_mdl-gd-smoothness`: Für eine $L$-glatte Funktion $f$ und den Gradientenschritt $\mathbf{x}_{k+1} = \mathbf{x}_k - \tfrac1L\nabla f(\mathbf{x}_k)$ gilt

<!--en:cd59d500-->
Jeder Schritt verringert die Zielfunktion um mindestens $\|\nabla f(\mathbf{x}_k)\|^2/(2L)$. Bei einer allgemeinen nichtkonvexen Funktion zeigt das nur, dass der Gradient gegen null geht. Konvexität stellt die fehlende Verbindung zwischen dem Gradienten und der Differenz zum Optimalwert her: Solange $f(\mathbf{x}_k) - f^\star$ groß ist, muss der Gradient groß genug sein, um weiteren Fortschritt zu ermöglichen. Die Iterierten können also nicht weit vom Optimum zum Stillstand kommen.

<!--en:acabd28d-->
**Proposition (Gradientenabstieg für glatte konvexe Funktionen).** *Sei $f$ konvex und $L$-glatt mit einem Minimierer $\mathbf{x}^\star$. Für den Gradientenabstieg mit $\eta = 1/L$ gilt für jedes $k \ge 1$*

<!--en:c746bd44-->
Die Differenz zum minimalen Verlust nimmt also mindestens so schnell ab wie $1/k$. Den Vorfaktor bestimmen die Glattheitskonstante und das Quadrat des Anfangsabstands zum Minimierer. Für $L = 1$ und $\|\mathbf{x}_0 - \mathbf{x}^\star\| = 10$ beträgt die Schranke beispielsweise $50/k$. Sie garantiert eine Differenz von höchstens $10^{-2}$ nach $5{,}000$ Schritten.

<!--en:ede5f310-->
**Beweis.** Wir kombinieren das Abstiegslemma mit der Bedingung erster Ordnung :eqref:`eq_mdl-opt-first-order` an der Stelle $\mathbf{x}_k$ in Richtung des Optimums, also mit $f(\mathbf{x}_k) - f(\mathbf{x}^\star) \le \nabla f(\mathbf{x}_k)^\top(\mathbf{x}_k - \mathbf{x}^\star)$:

<!--en:1bda333e-->
Nun ergänzen wir zum Quadrat. Dadurch wird die rechte Seite zur Differenz zweier quadrierter Abstände zu $\mathbf{x}^\star$:

<!--en:357c2142-->
Die zweite Norm ist gerade $\|\mathbf{x}_{k+1} - \mathbf{x}^\star\|^2$. Summieren wir über $j = 0, \ldots, k - 1$, so heben sich auf der rechten Seite die Zwischenterme auf:

<!--en:7528e2f5-->
Nach dem Abstiegslemma wächst $f(\mathbf{x}_j)$ nie. Daher ist der letzte Summand $f(\mathbf{x}_k) - f(\mathbf{x}^\star)$ der kleinste der Summe, und $k\left(f(\mathbf{x}_k) - f(\mathbf{x}^\star)\right)$ ist höchstens so groß wie die gesamte Summe. Division durch $k$ liefert :eqref:`eq_mdl-opt-rate-convex`. $\blacksquare$

<!--en:c6a596ad-->
Zwei Eigenschaften dieses Beweises verdienen Beachtung. Erstens kommt die Dimension $d$ nirgends vor. Für ein Modell mit einer Milliarde Gewichten gilt bei gleichem $L$ und gleichem Anfangsabstand genau dieselbe Garantie für den ungünstigsten Fall. Zweitens ist vor der Summation die linke Seite der Ungleichung für jeden Schritt nichtnegativ: $f(\mathbf{x}_{k+1}) - f^\star \ge 0$. Daraus folgt $\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| \le \|\mathbf{x}_k - \mathbf{x}^\star\|$. Der Abstand zu jedem Minimierer nimmt also bei keinem Schritt zu.

<!--en:1cab575b-->
Starke Konvexität verbessert diese Abnahme wie $1/k$ zu einer geometrischen Rate.

<!--en:0dbd6b5d-->
**Proposition (lineare Rate bei starker Konvexität).** *Sei $f$ eine $\mu$-stark konvexe und $L$-glatte Funktion mit Minimierer $\mathbf{x}^\star$. Für den Gradientenabstieg mit $\eta = 1/L$ gilt*

<!--en:01106eb8-->
Jede Iteration baut also mindestens den Anteil $\mu/L = 1/\kappa$ der verbleibenden Differenz zum minimalen Verlust ab.

<!--en:3e2f2a65-->
**Beweis.** Der Beweis besteht aus zwei kurzen Schritten. Zunächst liefert starke Konvexität eine untere Schranke für die quadrierte Gradientennorm in Abhängigkeit von der Differenz zum Optimalwert. Wir minimieren beide Seiten von :eqref:`eq_mdl-opt-strong-convexity` über $\mathbf{y}$: Links ergibt sich das Minimum $f(\mathbf{x}^\star)$. Rechts steht eine quadratische Funktion in $\mathbf{y}$ mit Minimierer $\mathbf{y} = \mathbf{x} - \tfrac{1}{\mu}\nabla f(\mathbf{x})$ und Minimalwert $f(\mathbf{x}) - \tfrac{1}{2\mu}\|\nabla f(\mathbf{x})\|^2$. Umstellen liefert

<!--en:98fe0429-->
Im zweiten Schritt setzen wir :eqref:`eq_mdl-opt-pl` direkt in das Abstiegslemma ein:

<!--en:99bbb3a5-->
Wenden wir diese Kontraktion $k$-mal an, ist der Beweis abgeschlossen. $\blacksquare$

<!--en:d5ce0f53-->
Der zweite Teil des Beweises nutzt an keiner Stelle Konvexität, sondern nur die Ungleichung :eqref:`eq_mdl-opt-pl`. Auf diese überraschende Tatsache kommen wir in :numref:`subsec_mdl-beyond-convexity` zurück. Beachten Sie auch den großen Geschwindigkeitsgewinn gegenüber dem Fall ohne starke Konvexität: Für eine relative Genauigkeit von $\epsilon$ brauchen wir jetzt ungefähr $\kappa \log(1/\epsilon)$ Schritte statt $O(1/\epsilon)$. In unserem quadratischen Beispiel mit $\kappa = 10^3$ erhalten wir nach dieser Schranke sechs Stellen Genauigkeit, also $(1 - 10^{-3})^k \le 10^{-6}$, in etwa $13{,}800$ Schritten.

<!--en:b382bfd5-->
Wie scharf ist :eqref:`eq_mdl-opt-rate-strong` in der Praxis? Messen wir das an einem Problem der kleinsten Quadrate, dessen Hesse-Matrix $\mathbf{X}^\top\mathbf{X}$ konstruktionsbedingt die Eigenwerte $1$ und $10$ hat ($\mu = 1$, $L = 10$, $\kappa = 10$). Wir führen Gradientenabstieg mit $\eta = 1/L$ aus und verfolgen das tatsächliche Verhältnis der Differenzen zum minimalen Verlust in aufeinanderfolgenden Schritten, also $(f(\mathbf{w}_{k+1}) - f^\star)/(f(\mathbf{w}_k) - f^\star)$.

<!--en:5bc5eb4b-->
Unsere Schranke garantiert pro Schritt den Kontraktionsfaktor $0.9$. Ab dem zweiten Schritt messen wir jedoch das Verhältnis $0.81$, also $(1 - \mu/L)^2$. (Das erste Verhältnis hängt von der zufälligen Startrichtung ab und schwankt von Lauf zu Lauf.) In den Eigenvektorkoordinaten von $\mathbf{X}^\top\mathbf{X}$ lässt sich das erklären. Bei $\eta = 1/L$ multipliziert der erste Schritt die steife Eigenkomponente von $\mathbf{w} - \mathbf{w}^\star$ mit $1 - L/L = 0$ und löscht sie vollständig aus. In zwei Dimensionen bleibt nur die langsame Komponente. Ihr Abstand zu $\mathbf{w}^\star$ schrumpft pro Schritt um den Faktor $1 - \mu/L$. Die Differenz zum minimalen Verlust hängt quadratisch von diesem Abstand ab und schrumpft deshalb um den Faktor $(1 - \mu/L)^2$. In höheren Dimensionen steigt das Verhältnis gegen $(1 - \mu/L)^2$, während die schnelleren Eigenkomponenten abklingen.

<!--en:98c55b1d-->
In unserem kurzen Beweis ging ein Quadrat im Kontraktionsfaktor verloren. Eine schärfere Analyse zeigt, dass der Gradientenabstieg mit $\eta = 1/L$ die Differenz $f - f^\star$ pro Schritt um den Faktor $(1 - \mu/L)^2$ verringert, und zwar für *jede* $L$-glatte, $\mu$-stark konvexe Funktion :cite:`Taylor.Hendrickx.Glineur.2018`. Quadratische Funktionen erreichen die Schranke. In Aufgabe :exref:`ex_mdl-convexity-distance-contraction` beweisen Sie die eng verwandte Schranke $\kappa(1 - \mu/L)^{2k}$. Kehren wir zu unserem quadratischen Beispiel mit $\kappa = 10^3$ zurück. Sechs Stellen Genauigkeit in der Differenz zum minimalen Verlust (drei Stellen im Abstand zum Minimierer) kosten nach unserer einfachen Schranke rund $13{,}800$ Schritte. Bei der tatsächlichen Rate sind es etwa $6{,}900$ Schritte mit dem Faktor $(1 - \mu/L)^2$. Mit der optimalen festen Schrittweite reichen halb so viele, nämlich $3{,}454$ Schritte bei $\eta = 2/(\mu + L)$. Das ist die Hälfte der $6{,}908$ Schritte, die :numref:`sec_mdl-gradient-based-optimization` bei derselben Schrittweite für sechs Stellen Genauigkeit im Abstand gezählt hat. Der erste Faktor zwei erklärt sich aus der Ungenauigkeit unseres Beweises, der zweite aus der Verdopplung der Schrittweite.

<!--en:f4bde6c4-->
## Konvexität erkennen

<!--en:83281031-->
Globale Optimalität und klare Konvergenzraten nützen uns nur, wenn wir erkennen können, ob eine Verlustfunktion überhaupt konvex ist. Hesse-Matrizen von Hand zu berechnen wird schnell mühsam und scheitert an Knicken ganz. In der Praxis prüfen wir Konvexität daher selten von Grund auf. Wir erkennen konvexe Zielfunktionen daran, wie sie sich aus einfachen konvexen Funktionen zusammensetzen.

<!--en:3cd93f17-->
### Rechenregeln für konvexe Funktionen

<!--en:28eb4318-->
Vier elementare Operationen erhalten Konvexität. Fast jede konvexe Verlustfunktion im maschinellen Lernen lässt sich mit diesen vier Regeln aus konvexen Grundfunktionen aufbauen.

<!--en:332dc3eb-->
1. **Summen mit nichtnegativen Gewichten.** Sind $f_1, \ldots, f_m$ konvex und gilt $w_i \ge 0$, so ist $\sum_i w_i f_i$ konvex: Wir multiplizieren die Sehnenungleichung für jedes $f_i$ mit $w_i$ und addieren. Ist die Verlustfunktion für jedes einzelne Beispiel konvex, so ergibt insbesondere ihr Mittel über einen Trainingsdatensatz stets eine konvexe Zielfunktion.

<!--en:892140b2-->
2. **Verkettung mit einer affinen inneren Abbildung.** Ist $f$ konvex, so auch $\mathbf{x} \mapsto f(\mathbf{A}\mathbf{x} + \mathbf{b})$. Eine affine Abbildung überführt die Strecke zwischen $\mathbf{x}$ und $\mathbf{y}$ in die Strecke zwischen ihren Bildern, und zwar mit demselben Gewicht $\theta$. Die Sehnenungleichung überträgt sich also direkt. Ist ein Verlust konvex in den *Vorhersagen* eines Modells, so ist er folglich auch konvex in den *Gewichten eines linearen Modells*, das diese Vorhersagen liefert.

<!--en:d752cea1-->
3. **Punktweises Maximum und Supremum.** Ist jedes $f_i$ konvex, so auch $\max_i f_i$. Der kürzeste Beweis nutzt Epigraphen: Der Epigraph des punktweisen Maximums ist der Schnitt der einzelnen Epigraphen, und Schnitte konvexer Mengen sind konvex. Deshalb ist die obere Einhüllende jeder Familie affiner Funktionen konvex, unabhängig von der Größe der Familie.

<!--en:f0dfff30-->
4. **Monotone Verkettung.** Ist $g$ konvex und $h$ konvex *und monoton wachsend*, so ist die Verkettung $h \circ g$ konvex. Zwei Zeilen genügen zum Beweis:

   $$
   h\bigl(g(\theta\mathbf{x} + (1-\theta)\mathbf{y})\bigr)
   \;\le\; h\bigl(\theta g(\mathbf{x}) + (1-\theta) g(\mathbf{y})\bigr)
   \;\le\; \theta\, h(g(\mathbf{x})) + (1-\theta)\, h(g(\mathbf{y})),
   $$

   Die erste Ungleichung nutzt die Konvexität von $g$ und die Monotonie von $h$, die zweite die Konvexität von $h$. Die Monotonie von $h$ ist wesentlich: $h(t) = t^2$ ist konvex, aber nicht monoton wachsend. Die Verkettung mit der konvexen Funktion $g(x) = x^2 - 1$ ergibt das nichtkonvexe Doppelmuldenpotential $(x^2 - 1)^2$ (daraus entsteht $\phi$ durch Addition von $x/2$).

<!--en:90e53495-->
Mit diesen vier Regeln weisen wir die Konvexität fast aller gängigen konvexen Verlustfunktionen auf einmal nach. Der Hinge-Verlust $\max(0, 1 - y\,\mathbf{w}^\top\mathbf{x})$ ist das Maximum zweier affiner Funktionen von $\mathbf{w}$ (Regeln 3 und 2). Die $\ell_1$-Norm $\|\mathbf{w}\|_1 = \sum_i \max(w_i, -w_i)$ ist eine Summe von Maxima affiner Funktionen (Regeln 1, 3 und 2). Der logistische Verlust $\log(1 + e^{-y\,\mathbf{w}^\top\mathbf{x}})$ geht von der skalaren Funktion $t \mapsto \log(1 + e^t)$ aus. Sie ist konvex, denn ihre zweite Ableitung $e^t/(1 + e^t)^2$ ist strikt positiv. Wir schalten ihr eine affine Abbildung vor (Regel 2) und summieren über die Trainingsbeispiele (Regel 1). Addieren wir zu einer dieser Funktionen den Ridge-Strafterm $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$, erhalten wir eine stark konvexe Zielfunktion. Regel 3 liefert aber nicht automatisch starke Konvexität: $\max(x^2, 1 - x)$ ist konvex, stimmt jedoch auf dem ganzen Intervall mit der affinen Funktion $1 - x$ überein. Das Intervall $(-1.618, 0.618)$ liegt zwischen den beiden Nullstellen von $x^2 = 1 - x$. Die Funktion ist also nicht stark konvex.

<!--en:08fc6ea5-->
In unserer Liste fehlt die Verkettung mit einer inneren Abbildung, die weder affin noch konvex ist. Als Funktion seiner Gewichte ist ein zweischichtiges neuronales Netz $\mathbf{W}_2\,\mathrm{relu}(\mathbf{W}_1\mathbf{x})$ genau eine solche Verkettung. An dieser Stelle geht die Konvexität verloren (:numref:`subsec_mdl-beyond-convexity`).

<!--en:1c77ea07-->
### Log-Sum-Exp und die Softmax-Kovarianz

<!--en:97aa70eb-->
Eine grundlegende Verlustfunktion erfassen die vier Regeln nicht unmittelbar: die Log-Sum-Exp-Funktion $\mathrm{lse}(\mathbf{x}) = \log \sum_{i=1}^d e^{x_i}$, auf der jeder Softmax-Kreuzentropieverlust beruht. Geometrisch ist $\mathrm{lse}(\mathbf{x})$ eine glatte Näherung an $\max_i x_i$ von oben, die den Wert um höchstens $\log d$ übersteigt. Ihre Konvexität können wir direkt anhand ihrer Hesse-Matrix nachweisen. Diese hat eine besonders einfache wahrscheinlichkeitstheoretische Deutung.

<!--en:433510e8-->
**Satz (Log-Sum-Exp ist konvex).** *Die Hesse-Matrix von $\mathrm{lse}$ an der Stelle $\mathbf{x}$ ist*

<!--en:b75a86fd-->
*Dies ist die Kovarianzmatrix der One-Hot-Kodierung einer Klasse, die wir gemäß $\mathbf{s}$ ziehen. Da jede Kovarianzmatrix positiv semidefinit ist, ist $\mathrm{lse}$ konvex.*

<!--en:b7f541a3-->
**Beweis.** Einmaliges Ableiten ergibt $\partial\, \mathrm{lse} / \partial x_i = e^{x_i} / \sum_k e^{x_k} = s_i$. Der Gradient von $\mathrm{lse}$ ist also der Softmax-Vektor $\mathbf{s}$. Leiten wir $s_i = e^{x_i}/\sum_k e^{x_k}$ mit der Quotientenregel ein weiteres Mal ab, erhalten wir

<!--en:ec5af53f-->
Dies ist der Eintrag $(i, j)$ von $\mathrm{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$ (dieselbe Jacobi-Matrix der Softmax-Funktion wie in :numref:`sec_mdl-matrix-calculus`). Sei nun $Y$ ein zufälliger Klassenindex mit Wahrscheinlichkeiten $P(Y = i) = s_i$ und $\mathbf{e}_Y$ sein One-Hot-Vektor. Dann gelten $\mathbb{E}[\mathbf{e}_Y] = \mathbf{s}$ und $\mathbb{E}[\mathbf{e}_Y \mathbf{e}_Y^\top] = \mathrm{diag}(\mathbf{s})$. Die Hesse-Matrix ist also tatsächlich $\textrm{Cov}(\mathbf{e}_Y) = \mathrm{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$. Jede Kovarianzmatrix ist positiv semidefinit. Für eine beliebige Richtung $\mathbf{v}$ gilt nämlich

<!--en:ec06593b-->
Die Krümmung von Log-Sum-Exp in Richtung $\mathbf{v}$ ist also gerade die Varianz der Koordinate $v_Y$ unter der Softmax-Verteilung.

<!--en:5658f425-->
Aus dieser Formel folgen unmittelbar drei praktische Aussagen. Erstens ist der Kreuzentropieverlust als Funktion der Logits, also $\ell(\mathbf{z}) = \mathrm{lse}(\mathbf{z}) - z_y$ für die wahre Klasse $y$, gleich Log-Sum-Exp abzüglich eines linearen Terms und damit konvex. Zweitens setzt die Softmax-Regression (:numref:`sec_softmax`) die affinen Logits $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ in $\ell(\mathbf{z})$ ein. Nach Regel 2 ist der Verlust also konvex in den Gewichten: Einen linearen Klassifikator mit festen Merkmalen zu trainieren ist immer ein konvexes Problem. Drittens folgt aus der Varianzformel sofort der *Eigenwert null* in Richtung des Einsvektors $\mathbf{1}$, denn es gilt $\textrm{Var}(v_Y) = 0$, wenn alle Einträge $v_i$ gleich sind. Diese flache Richtung entspricht der Verschiebungsinvarianz $\mathrm{lse}(\mathbf{x} + c\mathbf{1}) = \mathrm{lse}(\mathbf{x}) + c$, die eine numerisch stabile Softmax-Berechnung ermöglicht. Dieselbe Jacobi-Matrix $\mathrm{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$ bestimmt auch den Rückwärtsdurchlauf jedes Attention-Kopfs. Seine Attention-Gewichte ergeben sich aus der Anwendung der Softmax-Funktion auf die Scores zwischen Abfragen und Schlüsseln (:numref:`sec_attention-scoring-functions`).

<!--en:c0f7d4e9-->
Wie groß kann die Krümmung von Log-Sum-Exp werden? Wie Sie in Aufgabe :exref:`ex_mdl-convexity-smoothing-maximum` zeigen, übersteigt kein Eigenwert der Hesse-Matrix jemals $\tfrac12$, unabhängig von den Logits. Also ist $\tfrac12$ eine globale Glattheitskonstante. Wo nähert sich die Krümmung dieser Schranke, und wie stark ist die Verlustfunktion bei typischen Logits gekrümmt? Im Folgenden ziehen wir $10{,}000$ zufällige Logit-Vektoren für sechs Klassen, skalieren sie mit einem Temperaturfaktor $c$ und geben sowohl den größten Eigenwert der Hesse-Matrix über alle Ziehungen als auch den Median der größten Eigenwerte an.

<!--en:bd729257-->
Wie vorhergesagt übersteigt der größte Eigenwert nie $\tfrac12$. Das bestätigt, dass die Softmax-Kreuzentropie bei jeder Temperatur $\tfrac12$-glatt in den Logits ist. Bei kleinen Logits ($c = 0.1$) ist die Softmax-Verteilung über die sechs Klassen nahezu gleichförmig und die Krümmung liegt knapp über $\tfrac16$. Bei mittlerer Skalierung erreicht die typische Krümmung ihren höchsten Wert, mit einem Median nahe $0.3$. Vergrößern wir $c$ weiter, gehen Median und Maximum auseinander. Bei einer typischen Ziehung konzentriert sich die Vorhersage fast ganz auf eine Klasse (nahezu One-Hot). Der Median der Krümmung fällt daher gegen null: In der Umgebung einer sicheren Vorhersage ist die Verlustfunktion beinahe flach. Das *Maximum* über die $10{,}000$ Ziehungen steigt hingegen gegen $\tfrac12$. Es stammt von den seltenen Ziehungen, bei denen die beiden größten Logits fast gleich sind und die Softmax-Funktion die Wahrscheinlichkeit nahezu gleichmäßig auf diese zwei Klassen verteilt.

<!--en:b5b089fc-->
## Jenseits der Konvexität

<!--en:d7a42409-->
Tiefe neuronale Netze haben keine konvexen Verlustlandschaften. In diesem letzten Unterabschnitt zeigen wir den Grund an nur zwei skalaren Gewichten. Wir bestimmen eine schwächere Bedingung, unter der die lineare Konvergenzrate des Gradientenabstiegs auch ohne Konvexität gilt, und fragen, wie ein Optimierer unter unendlich vielen globalen Minimierern eine Lösung auswählt.

<!--en:c66d598f-->
### Nichtkonvexe Netze

<!--en:6a58ade7-->
Betrachten wir $f(a, b) = (ab - 1)^2$: ein lineares Modell mit zwei hintereinandergeschalteten skalaren Gewichten und quadratischem Verlust, das einfachste denkbare „tiefe“ Netz. Wie :numref:`fig_mdl-opt-factored-model` zeigt, bilden seine globalen Minimierer die Hyperbel $\{ab = 1\}$. In :numref:`subsec_mdl-why-convexity-matters` haben wir bewiesen, dass die Menge der Minimierer einer konvexen Funktion konvex sein muss. Eine Hyperbel ist jedoch offensichtlich nicht konvex: Der Mittelpunkt der beiden Minimierer $(1, 1)$ und $(-1, -1)$ ist der Ursprung $(0, 0)$. Dort liegt $f(0, 0) = 1$ deutlich über dem Minimalwert $0$. Der Ursprung ist ein stationärer Punkt, aber kein Minimum: Seine Hesse-Matrix $\nabla^2 f(0, 0)$ hat die Eigenwerte $\pm 2$, also ist er ein Sattelpunkt.

<!--en:e46f10de-->
![Das Modell $(ab-1)^2$ mit zwei Gewichten. Seine Minimierer bilden die nichtkonvexe Hyperbel $ab=1$: Der Mittelpunkt zwischen $(1,1)$ und $(-1,-1)$ ist der Ursprung. Dort gilt $f=1$, und die Hesse-Matrix hat die Eigenwerte $\pm2$.](../img/mdl-opt-factored-model.svg)

<!--en:90b1a8e7-->
Bei neuronalen Netzen aus der Praxis geht die Konvexität aus einem eng verwandten strukturellen Grund verloren. Vertauschen wir die versteckten Einheiten einer beliebigen Schicht zusammen mit ihren ein- und ausgehenden Gewichten, bleibt die Abbildung von Eingaben auf Ausgaben gleich. Zu jedem Minimierer gehören also zahlreiche symmetrische Kopien, verstreut über den Parameterraum. Im Allgemeinen sind diese Kopien verschieden. Mitteln wir zwei davon, ändert sich die Funktion und der Verlust steigt, wie uns schon das Mittel aus $(1, 1)$ und $(-1, -1)$ zum Sattelpunkt am Ursprung geführt hat. Enthält eine Menge von Minimierern zwei verschiedene Punkte, aber nicht deren Mittelpunkt, so kann die Zielfunktion nicht konvex sein.

<!--en:d0bed95d-->
Schon ein winziges Netz mit zwei ReLU-Einheiten zeigt diesen Effekt. Der Parametervektor $(w_1, v_1, w_2, v_2) = (1, 1, -1, 1)$ in $v_1\,\mathrm{relu}(w_1 x) + v_2\,\mathrm{relu}(w_2 x)$ berechnet $|x|$. Vertauschen wir die beiden Einheiten, erhalten wir genau dieselbe Funktion $|x|$. Mitteln wir jedoch die beiden Parametervektoren, heben sich die Eingabegewichte auf. Das Netz berechnet dann die Nullfunktion.

<!--en:5593b62f-->
### Die Polyak-Łojasiewicz-Bedingung

<!--en:7fdb4db3-->
Erinnern wir uns an den Beweis der linearen Konvergenzrate bei starker Konvexität: Er nutzte Konvexität *nur* im ersten Schritt, um :eqref:`eq_mdl-opt-pl` herzuleiten. Der zweite Schritt kam ganz ohne Konvexität aus. Die Ungleichung aus dem ersten Schritt hat einen eigenen Namen. Eine Funktion $f$ mit Infimum $f^\star$ erfüllt die **Polyak-Łojasiewicz-Ungleichung (PL-Ungleichung)** mit Konstante $\mu > 0$, wenn $\tfrac12\|\nabla f(\mathbf{x})\|^2 \ge \mu\,(f(\mathbf{x}) - f^\star)$ für alle $\mathbf{x}$ gilt. Polyak und Łojasiewicz führten diese Bedingung 1963 unabhängig voneinander ein :cite:`Polyak.1963,Lojasiewicz.1963`. Anschaulich darf der Gradient nur dann klein sein, wenn der Verlust bereits nahe am Optimalwert liegt.

<!--en:de626095-->
**Satz (lineare Rate unter der PL-Bedingung).** *Sei $f$ eine $L$-glatte, nach unten beschränkte Funktion mit $f^\star = \inf f$, die der PL-Ungleichung mit Konstante $\mu$ genügt. Dann gilt für den Gradientenabstieg mit $\eta = 1/L$ die Schranke $f(\mathbf{x}_k) - f^\star \le (1 - \mu/L)^k\,(f(\mathbf{x}_0) - f^\star)$.*

<!--en:a470dc02-->
**Beweis.** Wir wiederholen den zweiten Teil des Beweises für starke Konvexität unverändert: Aus Abstiegslemma und PL-Ungleichung folgt $f(\mathbf{x}_{k+1}) - f^\star \le (1 - \mu/L)(f(\mathbf{x}_k) - f^\star)$ :cite:`Karimi.Nutini.Schmidt.2016`. Wenden wir diese Abschätzung $k$-mal an, ist der Beweis abgeschlossen. $\blacksquare$

<!--en:48619841-->
Die PL-Bedingung ist deshalb so attraktiv, weil sie viel Spielraum für Nichtkonvexität lässt. Eine PL-Funktion kann mehrere Minimierer, flache Mannigfaltigkeiten von Minimierern und Bereiche negativer Krümmung haben. Zwei Eigenschaften sind jedoch zwingend: Jeder stationäre Punkt muss ein globaler Minimierer sein (denn aus $\nabla f(\mathbf{x}) = \mathbf{0}$ folgt $f(\mathbf{x}) = f^\star$), *und* $f - f^\star$ muss mit dem Abstand von der Menge der Minimierer mindestens quadratisch wachsen. Die Funktion vierten Grades $x^4$ zeigt, warum die zweite Forderung zählt. Ihr einziger stationärer Punkt ist ihr globales Minimum. Trotzdem verletzt sie die PL-Bedingung, denn $\tfrac12 f'(x)^2 = 8x^6$ kann nicht über $\mu x^4$ bleiben, wenn $x \to 0$ gilt, gleich welches $\mu > 0$ wir wählen. Genau deshalb kam der Gradientenabstieg bei $x^4$ in :numref:`subsec_mdl-convergence-rates` so langsam voran. Starke Konvexität impliziert also die PL-Bedingung, wie wir bereits bewiesen haben. Die Umkehrung ist falsch.

<!--en:6c7efbd6-->
Das klassische Beispiel einer nichtkonvexen PL-Funktion ist $f(x) = x^2 + 3\sin^2 x$. Ihre zweite Ableitung $2 + 6\cos 2x$ nimmt Werte von $-4$ bis $L = 8$ an. Schätzen wir ihre PL-Konstante numerisch auf $[-5, 5]$ und führen wir den Gradientenabstieg vom Startpunkt $x_0 = 4.5$ mit Schrittweite $\eta = \tfrac1{16} = 1/(2L)$ aus. (Wir wählen $1/(2L)$ anstelle von $1/L$, weil bei $\eta = 1/L$ der lokale Kontraktionsfaktor $1 - f''(0)\,\eta$ am Ursprung verschwindet und die asymptotische Rate verdeckt.) Bei $\eta = 1/(2L)$ garantiert die quadratische obere Schranke pro Schritt eine Abnahme um $\tfrac{3}{8L} f'(x_k)^2$. Daraus ergibt sich mit der PL-Ungleichung der Kontraktionsfaktor $1 - \tfrac{3\mu}{4L}$.

<!--en:efad8e5b-->
Die PL-Konstante auf $[-5, 5]$ beträgt $\mu \approx 0.1755$, und das Minimum von $f''$ liegt bei $-4$: Die Funktion ist nichtkonvex und erfüllt dennoch die PL-Bedingung. Das untersuchte Intervall reicht aus, denn außerhalb bleibt das Verhältnis $\tfrac12 f'(x)^2 / f(x)$ deutlich über diesem Wert und strebt gegen $2$ für $|x| \to \infty$. Aufgabe :exref:`ex_mdl-convexity-show-that-polyak-ojasiewicz` liefert den Beweis der Schranke.

<!--en:cb1e8a8a-->
Mit diesem $\mu$ beträgt der garantierte Faktor pro Schritt etwa $0.98$. Das größte gemessene Verhältnis von etwa $0.978$ stammt von $x \approx 2.2$, wo $\tfrac12 f'(x)^2/f(x)$ sein Minimum $\mu$ annimmt. Dort ist die Garantie also fast scharf. Nahe dem Minimierer nähert sich das Verhältnis $0.25$. Diesen Grenzwert erklärt die lokale quadratische Näherung: Nahe $0$ gilt $f \approx \tfrac12 f''(0)\,x^2$ mit $f''(0) = 8$. Jeder Schritt multipliziert $x$ mit $1 - 8/16 = \tfrac12$ und die in $x$ quadratische Differenz zum Optimalwert mit $\tfrac14$.

<!--en:a3c12568-->
Die PL-Bedingung kann auch nur in einem Teil des Raums gelten. Für $(ab - 1)^2$ erfüllt der Gradient die Beziehung $\tfrac12\|\nabla f\|^2 = 2(a^2 + b^2) f$. Daher gilt die PL-Bedingung mit Konstante $2r^2$ im Bereich $\{a^2 + b^2 \ge r^2\}$. In der Nähe des Sattelpunkts am Ursprung gibt es dagegen keine solche Konstante. Für breite Netze lassen sich solche lokalen PL-Ungleichungen in der Nähe interpolierender Lösungen nachweisen :cite:`Liu.Zhu.Belkin.2022`. Daraus folgt eine Rate für den Trainingsverlust, aber keine Aussage über die Generalisierung.

<!--en:f10c4872-->
### Impliziter Bias

<!--en:df0273ff-->
Hat eine Verlustfunktion viele globale Minimierer, entscheidet der Optimierer, welchen er erreicht, unabhängig davon, ob die Funktion konvex ist. Diese Bevorzugung bestimmter Lösungen heißt **impliziter Bias**. Für Deep Learning ist sie von praktischer Bedeutung: Ein Netz mit mehr Gewichten als Trainingsbeispielen kann die Daten auf viele Arten anpassen. Alle Lösungen liefern denselben Trainingsverlust, aber andernorts unterschiedliche Vorhersagen. Welche Lösung der Optimierer wählt, bestimmt also, wie das Modell generalisiert. Das einfachste Beispiel ist ein unterbestimmtes Problem der kleinsten Quadrate mit weniger Beispielen als Gewichten und einer Matrix $\mathbf{X}$ von vollem Zeilenrang. Jeder Gradient $\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ liegt im Zeilenraum von $\mathbf{X}$. Iterierte mit Startpunkt $\mathbf{w}_0 = \mathbf{0}$ verlassen diesen Raum also nie. Ihr Grenzwert ist deshalb der einzige Interpolant im Zeilenraum, der Interpolant *kleinster Norm*, den die Pseudoinverse berechnet (Aufgabe :exref:`ex_mdl-svd-low-rank-ridgeless-gradient-descent` in :numref:`sec_mdl-svd-low-rank`). Bei der logistischen Regression für separierbare Daten hat die Verlustfunktion überhaupt keinen Minimierer. Dennoch konvergiert die Richtung der Iterierten gegen die Maximum-Margin-Trennebene, die Lösung der Hard-Margin-Support-Vector-Machine :cite:`Soudry.Hoffer.Nacson.ea.2018` (Aufgabe :exref:`ex_mdl-convexity-max-margin-bias`). Welchen Minimierer ein Verfahren auswählt, hängt von Modell, Verlustfunktion, Startpunkt und Aktualisierungsregel ab. Ein fester diagonaler Vorkonditionierer $\mathbf{D}$ führt beispielsweise zu dem Interpolanten, der $\mathbf{w}^\top\mathbf{D}^{-1}\mathbf{w}$ statt $\|\mathbf{w}\|^2$ minimiert (Aufgabe :exref:`ex_mdl-convexity-implicit-bias-verified`). Die koordinatenweisen Schritte von Adam (:numref:`sec_adam`) müssen also nicht dieselbe Lösung auswählen wie der Gradientenabstieg.

<!--en:4557e09f-->
Alle Ergebnisse dieses Abschnitts setzen den exakten Gradienten und Gewichte voraus, die ganz $\mathbb{R}^d$ durchlaufen dürfen. Beim Training verwenden wir Minibatch-Gradienten, die nur im Mittel stimmen (:numref:`sec_mdl-stochastic-gradient-methods`). Ist ein Gewichtsvektor auf eine Kugel, einen Simplex oder einen Halbraum beschränkt, brauchen wir die Projektionen und Multiplikatoren aus :numref:`sec_mdl-constrained-optimization-duality`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3b1d6e3c-->
Eine konvexe Funktion liegt unter ihren Sehnen. Bei differenzierbaren Funktionen ist das gleichbedeutend damit, dass der Graph über jeder Tangente liegt. Bei zweimal differenzierbaren Funktionen ist es gleichbedeutend mit einer positiv semidefiniten Hesse-Matrix. An einem Knick tritt ein Subgradient an die Stelle des Gradienten. Da ein einziger Gradient die Funktion überall nach unten beschränkt, ist jeder stationäre Punkt ein globaler Minimierer. Das Abstiegslemma liefert dann globale Konvergenzraten für den Gradientenabstieg: $L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2/(2k)$ für glatte konvexe Funktionen $f$ und bei starker Konvexität einen Faktor $1 - \mu/L$ pro Schritt, also etwa $\kappa\log(1/\epsilon)$ Schritte. Die Jensen-Ungleichung ist die Sehnenungleichung für Erwartungswerte. Aus ihr folgt $D_{\textrm{KL}} \ge 0$. Vier Rechenregeln zeigen die Konvexität des Hinge-Verlusts, der $\ell_1$-Norm als Verlustfunktion und des logistischen Verlusts. Log-Sum-Exp und damit die Softmax-Kreuzentropie sind konvex, weil die Hesse-Matrix von Log-Sum-Exp eine Kovarianzmatrix ist. Die Verlustfunktionen neuronaler Netze sind im Allgemeinen nichtkonvex. Wo eine PL-Ungleichung gilt, bleibt die lineare Rate auch ohne Konvexität erhalten. Unter vielen Minimierern wählt die Dynamik des Optimierers einen aus.

<!--en:2294243a-->
## Übungen

<!--en:f452c86c-->
1. **Subdifferentiale von Normen und Maxima.** Berechnen Sie jedes Subdifferential aus der Definition :eqref:`eq_mdl-opt-subgradient`.
    1. Berechnen Sie $\partial \|\mathbf{x}\|_2$ für alle $\mathbf{x} \in \mathbb{R}^d$, einschließlich $\mathbf{x} = \mathbf{0}$.
    1. Sei $f(\mathbf{x}) = \max_i x_i$ und sei $\mathcal{A}(\mathbf{x}) = \{i : x_i = f(\mathbf{x})\}$ die Menge der aktiven Koordinaten. Zeigen Sie, dass $\partial f(\mathbf{x})$ die konvexe Hülle von $\{\mathbf{e}_i : i \in \mathcal{A}(\mathbf{x})\}$ ist. Bei der Backpropagation durch Max-Pooling geht der eingehende Gradient an eine einzige maximale Eingabe. Das Verfahren nutzt also einen Eckpunkt $\mathbf{e}_i$ dieser Hülle. Nehmen Sie an, dass zwei oder mehr Eingaben das Maximum annehmen. Zeigen Sie, dass ein kleiner Schritt entlang $-\mathbf{e}_i$ den Wert von $f$ nicht verringert. Für welche $\mathbf{g} \in \partial f(\mathbf{x})$ ist $-\mathbf{g}$ eine Abstiegsrichtung? Welcher dieser Subgradienten liefert die kleinste, also am stärksten negative Richtungsableitung von $f$ entlang $-\mathbf{g}/\|\mathbf{g}\|$?
    1. Nutzen Sie (a) und das Optimalitätskriterium $\mathbf{0} \in \partial(\cdot)$, um $\lambda\|\mathbf{w}\|_2 + \tfrac12\|\mathbf{w} - \mathbf{z}\|^2$ über $\mathbf{w} \in \mathbb{R}^d$ zu minimieren. So erhalten Sie den Proximaloperator von $\lambda\|\cdot\|_2$. Vergleichen Sie die Lage der Nullen in der Lösung mit dem Ergebnis des koordinatenweisen weichen Schwellenwertverfahrens $w_i = \mathrm{sign}(z_i)\max(|z_i| - \lambda, 0)$. Wie wirkt der Strafterm, wenn $\mathbf{w}$ die ausgehenden Gewichte einer versteckten Einheit enthält?
   :exlabel:`ex_mdl-convexity-compute-full-subdifferential-f`

<!--en:ea60ac33-->
1. **Konvexkombinationen und Jensen-Ungleichung.** Schreiben Sie $\sum_{i=1}^N \theta_i \mathbf{x}_i$ mit $\theta_i \ge 0$ und $\sum_i \theta_i = 1$ für eine Konvexkombination von $N$ Punkten.
    1. Zeigen Sie durch Induktion über $N$, dass eine konvexe Menge $C$ jede Konvexkombination von Punkten aus $C$ enthält. Diese Inklusion vervollständigt die Beschreibung der konvexen Hülle in :numref:`subsec_mdl-convex-sets`.
    1. Wenden Sie (a) auf den Epigraphen einer konvexen Funktion $f$ an, um die endliche Form :eqref:`eq_mdl-opt-jensen-finite` der Jensen-Ungleichung herzuleiten.
    1. Sei $f$ zweimal differenzierbar mit $\alpha \le f'' \le \beta$ auf einem Intervall, das die Werte einer Zufallsvariablen $Z$ mit Mittelwert $m$ und Varianz $s^2$ enthält. Zeigen Sie, dass die Differenz $\mathbb{E}[f(Z)] - f(m)$ in der Jensen-Ungleichung zwischen $\alpha s^2/2$ und $\beta s^2/2$ liegt. Prüfen Sie die Schranken am Zweipunktbeispiel aus dem Text, $Z \in \{0, 2\}$ mit $f(x) = x^2$, für das $\alpha = \beta = 2$ gilt. Wenden Sie das Ergebnis dann auf $f = -\log$ an, um $\log \mathbb{E}[Z] - \mathbb{E}[\log Z]$ für eine Zufallsvariable $Z$ mit Werten in $[a, b] \subset (0, \infty)$ abzuschätzen.
   :exlabel:`ex_mdl-convexity-prove-jensen-s-finite`

<!--en:79416eb2-->
1. **Existenz von Minimierern.** Sei $\ell(\mathbf{w}) = \sum_i \log\bigl(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}_i}\bigr)$ der logistische Verlust für die Labels $y_i \in \{\pm 1\}$ und $\mathbf{X}$ die Matrix mit Zeilen $\mathbf{x}_i^\top$. Die Eindeutigkeitsaussage aus :numref:`subsec_mdl-why-convexity-matters` besagt, dass eine strikt konvexe Funktion höchstens einen Minimierer hat. Eine stetige, stark konvexe Funktion auf einer nichtleeren abgeschlossenen konvexen Menge mit nichtleerem Inneren hat genau einen.
    1. Für ein $\mathbf{w}$ gelte $y_i \mathbf{w}^\top \mathbf{x}_i > 0$ für jedes $i$. Die Daten lassen sich also durch eine Hyperebene durch den Ursprung linear trennen. Zeigen Sie, dass $\inf_{\mathbf{w}} \ell(\mathbf{w}) = 0$ gilt und dass $\ell$ keinen Minimierer hat.
    1. Zeigen Sie, dass $\ell$ strikt konvex ist, sobald $\mathbf{X}$ vollen Spaltenrang hat. Warum widerspricht das nicht (a)?
    1. Zeigen Sie, dass $\ell(\mathbf{w}) + \tfrac{\lambda}{2}\|\mathbf{w}\|^2$ für jedes $\lambda > 0$ und jeden Datensatz genau einen Minimierer hat. Welche Voraussetzung der Eindeutigkeitsaussage stellt der Strafterm sicher?
   :exlabel:`ex_mdl-convexity-minimizer-existence`

<!--en:bedbd322-->
1. **Die PL-Bedingung ohne Konvexität.** Die Codezelle für $f(x) = x^2 + 3\sin^2 x$ misst die PL-Konstante $\mu \approx 0.1755$ auf dem Intervall $[-5, 5]$.
    1. Schätzen Sie $|f'(x)|$ nach unten und $f(x)$ nach oben ab, wenn $|x| \ge 5$ gilt. Folgern Sie, dass die PL-Ungleichung :eqref:`eq_mdl-opt-pl` auf ganz $\mathbb{R}$ mit der auf dem Intervall gemessenen Konstante gilt.
    1. Sei $\mathbf{X} \in \mathbb{R}^{n \times d}$ mit $n < d$ von vollem Zeilenrang, und sei $f(\mathbf{w}) = \tfrac12\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$. Zeigen Sie, dass $f$ nicht stark konvex ist und unendlich viele Minimierer besitzt, aber die PL-Ungleichung mit $\mu = \lambda_{\min}(\mathbf{X}\mathbf{X}^\top)$ erfüllt, dem kleinsten Eigenwert von $\mathbf{X}\mathbf{X}^\top$. Welche lineare Rate erreicht dann der Gradientenabstieg mit $\eta = 1/L$, und wie hängt $L$ mit $\mathbf{X}\mathbf{X}^\top$ zusammen?
   :exlabel:`ex_mdl-convexity-show-that-polyak-ojasiewicz`

<!--en:f2de82c9-->
1. [code] **Kontraktion des Abstands.** ● Die Rate :eqref:`eq_mdl-opt-rate-strong` garantiert den Faktor $1 - \mu/L$ pro Schritt für $f - f^\star$. Die Codezelle zur Prüfung der Rate misst jedoch $(1 - \mu/L)^2$. Sei $f$ zweimal stetig differenzierbar mit $\mu \mathbf{I} \preceq \nabla^2 f \preceq L \mathbf{I}$, und sei $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)$. Verwenden Sie zwei Ergebnisse aus :numref:`sec_mdl-gradient-based-optimization`: die quadratische obere Schranke $f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \tfrac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$ und die Schrittweite $\eta^\star = 2/(\mu + L)$, die beste feste Schrittweite für eine quadratische Funktion.
    1. Schreiben Sie $\nabla f(\mathbf{x}_k) = \bar{\mathbf{H}}_k (\mathbf{x}_k - \mathbf{x}^\star)$ mit $\bar{\mathbf{H}}_k = \int_0^1 \nabla^2 f\bigl(\mathbf{x}^\star + s(\mathbf{x}_k - \mathbf{x}^\star)\bigr)\,ds$. Zeigen Sie, dass

       $$\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| \le \max\bigl(|1 - \eta\mu|,\, |1 - \eta L|\bigr)\, \|\mathbf{x}_k - \mathbf{x}^\star\|.$$

    1. Setzen Sie $\eta = 1/L$ und $\kappa = L/\mu$. Kombinieren Sie (a) mit der unteren Schranke :eqref:`eq_mdl-opt-strong-convexity` und der quadratischen oberen Schranke, beide an der Stelle $\mathbf{x}^\star$, um $f(\mathbf{x}_k) - f^\star \le \kappa\,(1 - \mu/L)^{2k}\,(f(\mathbf{x}_0) - f^\star)$ zu zeigen. Ab welchem $k$ ist diese Schranke schärfer als :eqref:`eq_mdl-opt-rate-strong`?
    1. Sagen Sie die Kontraktion von $f - f^\star$ pro Schritt für die Konstruktion mit $\kappa = 10$ aus der Codezelle zur Prüfung der Rate bei $\eta^\star = 2/11$ voraus. (Hinweis: Beide Eigenkomponenten werden mit Zahlen vom Betrag $9/11$ multipliziert.) Prüfen Sie Ihre Vorhersage, indem Sie die Schrittweite in der Codezelle ändern. Erklären Sie, warum das erste Verhältnis anders als bei $\eta = 1/L$ keine Sonderrolle mehr spielt.
   :exlabel:`ex_mdl-convexity-distance-contraction`

<!--en:dd06d228-->
1. **Glättung des Maximums.** ● Das Maximum affiner Funktionen, wie es bei Zielfunktionen vom Hinge-Typ oder für den ungünstigsten Fall auftritt, hat einen Knick, wo zwei verschiedene Teilfunktionen gemeinsam das Maximum annehmen. Für eine Temperatur $\tau > 0$ sei $\mathrm{lse}_\tau(\mathbf{z}) = \tau\, \mathrm{lse}(\mathbf{z}/\tau)$ für $\mathbf{z} \in \mathbb{R}^m$. Dieses Glättungsverfahren geht auf :citet:`Nesterov.2005` zurück.
    1. Zeigen Sie $\max_j z_j \le \mathrm{lse}(\mathbf{z}) \le \max_j z_j + \log m$ und folgern Sie daraus $\max_j z_j \le \mathrm{lse}_\tau(\mathbf{z}) \le \max_j z_j + \tau\log m$.
    1. Zeigen Sie mit :eqref:`eq_mdl-opt-lse-hessian`, dass der größte Eigenwert von $\nabla^2 \mathrm{lse}_\tau(\mathbf{z})$ höchstens $1/(2\tau)$ beträgt. Schätzen Sie dafür $\textrm{Var}(v_Y)$ durch $\mathbb{E}[(v_Y - c)^2]$ mit einer geeignet gewählten Konstante $c$ nach oben ab. Wird die Schranke erreicht?
    1. Sei $F(\mathbf{w}) = \max_j (\mathbf{A}\mathbf{w} + \mathbf{r})_j$ und $F_\tau(\mathbf{w}) = \mathrm{lse}_\tau(\mathbf{A}\mathbf{w} + \mathbf{r})$. Nehmen Sie an, dass $F_\tau$ einen Minimierer im Abstand von höchstens $R$ zu $\mathbf{w}_0$ hat. Zeigen Sie mit (b), dass $F_\tau$ eine $L$-glatte Funktion mit $L = \|\mathbf{A}\|_2^2/(2\tau)$ ist und dass $k$ Schritte Gradientenabstieg auf $F_\tau$ mit $\eta = 1/L$ folgende Schranke liefern:

       $$F(\mathbf{w}_k) - \min F \le \tau \log m + \frac{\|\mathbf{A}\|_2^2 R^2}{4\tau k}.$$

       Wählen Sie $\tau$ so, dass diese Schranke minimal wird. Geben Sie die Schrittzahl an, die eine Genauigkeit von $\epsilon$ garantiert. Wie ändert sie sich, wenn Sie den Gradientenabstieg durch ein beschleunigtes Verfahren mit Fehler $O(LR^2/k^2)$ ersetzen?
   :exlabel:`ex_mdl-convexity-smoothing-maximum`

<!--en:6f2dfe68-->
1. [code] **Impliziter Bias und Optimierer.** ● In Aufgabe :exref:`ex_mdl-svd-low-rank-ridgeless-gradient-descent` aus :numref:`sec_mdl-svd-low-rank` haben Sie rechnerisch gezeigt, dass der Gradientenabstieg für $\tfrac12\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ mit $\mathbf{X} \in \mathbb{R}^{n \times d}$ von vollem Zeilenrang und $n < d$ bei Start in $\mathbf{w}_0 = \mathbf{0}$ stets im Zeilenraum von $\mathbf{X}$ bleibt. Er konvergiert gegen den Interpolanten kleinster Norm $\mathbf{X}^+\mathbf{y}$. Hier bleibt der Startpunkt $\mathbf{w}_0 = \mathbf{0}$ fest und wir ändern den Algorithmus. Wählen Sie ein zufälliges $4 \times 10$-System und vergleichen Sie für jedes Verfahren den Grenzwert und seine Norm mit dem Ergebnis von `np.linalg.pinv(X) @ y` und dessen Norm.
    1. Führen Sie Gradientenabstieg mit $\eta = 1/\lambda_{\max}(\mathbf{X}^\top \mathbf{X})$ aus und verfolgen Sie die Norm der Komponente von $\mathbf{w}_k$, die orthogonal zum Zeilenraum ist. Berechnen Sie sie mit dem Projektor $\mathbf{I} - \mathbf{X}^+\mathbf{X}$.
    1. Heavy-Ball-Momentum verwendet die Aktualisierung $\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\nabla f(\mathbf{w}_k) + \beta(\mathbf{w}_k - \mathbf{w}_{k-1})$. Zeigen Sie, dass die Iterierten im Zeilenraum bleiben, und sagen Sie ihren Grenzwert voraus. Prüfen Sie die Vorhersage mit der Schrittweite aus (a) und $\beta = 0.9$.
    1. Ersetzen Sie die Aktualisierung durch den diagonal vorkonditionierten Schritt $\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\, \mathbf{D} \mathbf{X}^\top (\mathbf{X}\mathbf{w}_k - \mathbf{y})$. Dabei ist $\mathbf{D}$ eine feste Diagonalmatrix, deren Einträge Sie gleichverteilt aus $[0.1, 10]$ ziehen, und es gilt $\eta = 1/\lambda_{\max}(\mathbf{D}^{1/2} \mathbf{X}^\top \mathbf{X} \mathbf{D}^{1/2})$. Bestimmen Sie den Grenzwert mit der Substitution $\mathbf{u} = \mathbf{D}^{-1/2}\mathbf{w}$ in geschlossener Form und prüfen Sie ihn numerisch. Ist er weiterhin der Interpolant kleinster Norm? Was legt dies über den impliziten Bias adaptiver Verfahren wie Adam (:numref:`sec_adam`) nahe, deren koordinatenweise Schrittweiten wie $\mathbf{D}$ wirken?
    1. Führen Sie 1000 Durchläufe des exakten zyklischen Koordinatenabstiegs aus. Verwenden Sie $\mathbf{A} = \mathbf{X}^\top \mathbf{X}$, $\mathbf{b} = \mathbf{X}^\top \mathbf{y}$ und die Aktualisierung $w_j \leftarrow (b_j - \sum_{i \ne j} A_{ji} w_i)/A_{jj}$ nacheinander für $j = 1, \ldots, d$. Prüfen Sie, ob der Grenzwert die Daten interpoliert. Ist er der Interpolant kleinster Norm? Welcher Schritt im Argument zum Zeilenraum scheitert? Wiederholen Sie den Lauf dreimal mit gleichverteilt zufällig gezogenen Koordinaten. Hängt der Grenzwert von der Ziehung ab?
   :exlabel:`ex_mdl-convexity-implicit-bias-verified`

<!--en:e208cbb4-->
1. [extended] **Bevorzugung der maximalen Margin beim Gradientenabstieg.** ●● Auf Daten, die eine Hyperebene durch den Ursprung trennt, hat der logistische Verlust keinen Minimierer (Teil (a) von Aufgabe :exref:`ex_mdl-convexity-minimizer-existence`). :citet:`Soudry.Hoffer.Nacson.ea.2018` zeigen, dass der Gradientenabstieg dennoch in einer bestimmten Richtung divergiert: Die Iterierten erfüllen $\mathbf{w}_k = \hat{\mathbf{w}} \log k + \boldsymbol{\rho}_k$, wobei $\boldsymbol{\rho}_k$ für fast jeden Datensatz beschränkt bleibt. Dabei ist

   $$\hat{\mathbf{w}} = \mathop{\mathrm{argmin}}_{\mathbf{w}} \|\mathbf{w}\| \quad \textrm{subject to} \quad y_i \mathbf{w}^\top \mathbf{x}_i \ge 1 \textrm{ for all } i$$

   die Lösung der Hard-Margin-SVM mit Margin $\gamma = 1/\|\hat{\mathbf{w}}\|$. Der Vektor $\hat{\mathbf{w}}$ ist durch $\hat{\mathbf{w}} = \sum_i \alpha_i y_i \mathbf{x}_i$ mit $\alpha_i \ge 0$ charakterisiert. Dabei gilt $\alpha_i = 0$, außer wenn $y_i \hat{\mathbf{w}}^\top \mathbf{x}_i = 1$ erfüllt ist. Die Punkte, für die Gleichheit gilt, sind die Stützvektoren.
    1. Leiten Sie die Richtung heuristisch her. Ersetzen Sie den logistischen Verlust für große Werte der Margin durch seine Näherung $e^{-q}$ mit $q = y_i \mathbf{w}^\top \mathbf{x}_i$. Nehmen Sie an, es gelte $\mathbf{w}_k = g(k)\,\mathbf{u} + O(1)$ mit $\|\mathbf{u}\| = 1$ und $g(k) \to \infty$. Zeigen Sie, dass der negative Gradient asymptotisch eine nichtnegative Linearkombination der Punkte $y_i\mathbf{x}_i$ mit kleinster Margin $y_i \mathbf{u}^\top \mathbf{x}_i$ ist. Folgern Sie, dass $\mathbf{u}$ proportional zu $\hat{\mathbf{w}}$ ist.
    1. Leiten Sie die Wachstumsrate her. Berücksichtigen Sie im Gradientenfluss $\dot{\mathbf{w}} = -\nabla \ell(\mathbf{w})$ nur die Stützvektoren. Zeigen Sie, dass $\dot g \approx c\, e^{-\gamma g}$ für eine Konstante $c > 0$ gilt, und folgern Sie daraus $\|\mathbf{w}(t)\| \approx \|\hat{\mathbf{w}}\| \log t$.
    1. Wählen Sie die Stützvektoren $\mathbf{x}_1 = (0.5, 1.5)$ und $\mathbf{x}_2 = (1.5, 0.5)$ mit Label $+1$, deren Negative mit Label $-1$ sowie 12 Zufallspunkte, 6 je Klasse, mit $y_i \hat{\mathbf{w}}^\top \mathbf{x}_i > 1$ für $\hat{\mathbf{w}} = (\tfrac12, \tfrac12)$. Sei $\mathbf{X}$ die $16 \times 2$-Matrix mit Zeilen $\mathbf{x}_i^\top$. Führen Sie Gradientenabstieg auf dem summierten logistischen Verlust ab $\mathbf{w}_0 = \mathbf{0}$ mit $\eta = 1/\|\mathbf{X}\|_2^2$ für $10^6$ Schritte aus. In vektorisiertem NumPy dauert das nur Sekunden. Tragen Sie gegen $k$ auf einer logarithmischen Achse die Norm $\|\mathbf{w}_k\|$, den Trainingsverlust, den Winkel zwischen $\mathbf{w}_k$ und $\hat{\mathbf{w}}$ sowie die Differenz zur maximalen Margin $\gamma - \min_i y_i \mathbf{w}_k^\top \mathbf{x}_i / \|\mathbf{w}_k\|$ auf. Vergleichen Sie die Steigung von $\|\mathbf{w}_k\|$ als Funktion von $\log k$ mit (b).
    1. Der Trainingsfehler erreicht schon nach wenigen Schritten null. Um welchen Faktor ändern sich die Differenz zur maximalen Margin und der Trainingsverlust zwischen $k = 10^3$ und $k = 10^6$ jeweils? Was folgt aus dem Vergleich für die Entscheidung, das Training zu beenden, sobald der Trainingsfehler null erreicht?
   :exlabel:`ex_mdl-convexity-max-margin-bias`

<!--en:aee20456-->
1. **Konvexität der Log-Partitionsfunktion.** Diese Aufgabe verallgemeinert die Aussage über Log-Sum-Exp auf die Log-Partitionsfunktionen, auf denen :numref:`sec_mdl-exponential-family` aufbaut. Für eine endliche Menge $\mathcal{X}$, Gewichte $h(x) > 0$, Statistiken $\mathbf{T}(x) \in \mathbb{R}^p$ und natürliche Parameter $\boldsymbol{\eta} \in \mathbb{R}^p$ (hier keine Schrittweite) seien $A(\boldsymbol{\eta}) = \log\sum_{x \in \mathcal{X}} h(x)\exp(\boldsymbol{\eta}^\top\mathbf{T}(x))$ und $p_{\boldsymbol{\eta}}(x) = h(x)\exp(\boldsymbol{\eta}^\top\mathbf{T}(x) - A(\boldsymbol{\eta}))$ gegeben.
    1. Zeigen Sie $\nabla A(\boldsymbol{\eta}) = \mathbb{E}_{\boldsymbol{\eta}}[\mathbf{T}]$ und $\nabla^2 A(\boldsymbol{\eta}) = \textrm{Cov}_{\boldsymbol{\eta}}[\mathbf{T}]$, wobei sich Erwartungswert und Kovarianz auf $p_{\boldsymbol{\eta}}$ beziehen.
    1. Folgern Sie, dass $A$ konvex ist. Zeigen Sie, dass die Funktion genau dann strikt konvex ist, wenn es keinen von null verschiedenen Vektor $\mathbf{v}$ gibt, für den $\mathbf{v}^\top\mathbf{T}(x)$ auf $\mathcal{X}$ konstant ist.
    1. Leiten Sie :eqref:`eq_mdl-opt-lse-hessian` mit $\mathcal{X} = \{1, \ldots, d\}$, $\mathbf{T}(x) = \mathbf{e}_x$ und $h \equiv 1$ erneut her und benennen Sie die Richtung, in der $A$ flach ist.
    1. Gegeben seien Beobachtungen $x_1, \ldots, x_n$ mit mittlerer Statistik $\bar{\mathbf{T}} = \tfrac1n\sum_i \mathbf{T}(x_i)$. Zeigen Sie, dass die mittlere negative Log-Likelihood $A(\boldsymbol{\eta}) - \boldsymbol{\eta}^\top\bar{\mathbf{T}} - \tfrac1n\sum_i \log h(x_i)$ konvex in $\boldsymbol{\eta}$ ist und dass für einen Minimierer $\mathbb{E}_{\boldsymbol{\eta}}[\mathbf{T}] = \bar{\mathbf{T}}$ gilt (Momentenabgleich). Warum gibt es im Fall der Softmax-Funktion aus (c) keinen Minimierer, wenn eine Klasse nie beobachtet wird?
   :exlabel:`ex_mdl-convexity-log-partition`

<!--en:0773990d-->
[Diskussion](https://d2l.discourse.group/t/36830)

<!--en:80446db9-->
Konvexität<br>**Konvexe Mengen · Sehnen, Tangenten und Hesse-Matrizen · Subgradienten · Jensen-Ungleichung · Konvergenzraten · PL-Bedingung**

<!--en:9cd731ab-->
::: {.slide title="Wann ist ein stationärer Punkt global optimal?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:fff4d3fb-->
Mit $\eta = 1/L$ erreicht der Gradientenabstieg zuverlässig einen Punkt mit $\nabla f(\mathbf{x}) \approx \mathbf{0}$. Das Newton-Verfahren gelangt noch schneller dorthin. Lokale Stationarität allein verrät jedoch nichts darüber, ob anderswo ein wesentlich tieferes Tal liegt:

<!--en:b07e8d6b-->
- **Experiment ($500$ zufällige Startpunkte in $[-1.5, 1.5]$):** Derselbe Gradientenabstieg ($600$ Schritte, $\eta = 0.05$) für zwei eindimensionale Funktionen:
  - **Quadratische Schale** $f(x) = (x - \tfrac12)^2$: Jeder Lauf konvergiert gegen das eindeutige globale Minimum $x^\star = 0.5$.
  - **Geneigtes Doppelmuldenpotential** $\phi(x) = (x^2 - 1)^2 + \tfrac12 x$: Rund $46\%$ der Läufe bleiben bei $x \approx 0.93$ ($\phi \approx 0.48$) stecken, fast eine ganze Einheit über dem globalen Minimum bei $x \approx -1.06$ ($\phi \approx -0.52$).

<!--en:3f48f90c-->
- **Geometrischer Unterschied:** Jede Sehne zwischen zwei Punkten der Schale liegt über ihrem Graphen. Sehnen über die mittlere Erhebung des Doppelmuldenpotentials verlaufen teilweise unter der Kurve.

<!--en:ad64379f-->
- **Folge der Konvexität:** Jede tangentiale Hyperebene einer konvexen Funktion liefert eine globale untere Schranke für die Zielfunktion. Damit garantiert $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ globale Optimalität.

<!--en:b195e95c-->
::: {.slide title="Konvexe Mengen enthalten jede Verbindungsstrecke"}

<!--en:a8826c43-->
[Konvexe Mengen]{.kicker}

<!--en:f0dd9e3b-->
Eine Menge $C \subseteq \mathbb{R}^d$ heißt **konvex**, wenn die Verbindungsstrecke zwischen je zwei Punkten $\mathbf{x}, \mathbf{y} \in C$ vollständig in $C$ liegt, also für alle $\theta \in [0, 1]$ gilt:

<!--en:ddf9d996-->
- **Nichtkonvexes Gegenbeispiel:** Der Kreisring $\{1 \le \|\mathbf{x}\|_2 \le 2\}$ enthält $(\pm\tfrac32, 0)$, doch der Mittelpunkt $\mathbf{0}$ ihrer Verbindungsstrecke liegt im Loch.

<!--en:33ed726c-->
- **Drei wichtige Familien:**
  1. **Hyperebenen und Halbräume:** $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} = b\}$ und $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \le b\}$ (wegen der Linearität).
  2. **Normkugeln:** $\{\mathbf{x} : \|\mathbf{x}\| \le r\}$ für jede Norm (wegen Dreiecksungleichung und Homogenität).
  3. **Wahrscheinlichkeitssimplex:** $\Delta = \{\mathbf{p} \succeq \mathbf{0} : \mathbf{1}^\top\mathbf{p} = 1\}$.

<!--en:916d6604-->
::: {.slide title="Neue Mengen durch Schnitte und konvexe Hüllen"}

<!--en:a8826c43-2-->
[Konvexe Mengen]{.kicker}

<!--en:b13e7b3f-->
- **Beliebige Schnitte erhalten Konvexität:** Ist jedes $C_i$ konvex, so ist auch $\bigcap_{i \in \mathcal{I}} C_i$ konvex (Beispiele sind der Simplex $\Delta$ und der PSD-Kegel $\mathbb{S}_+^d$).

<!--en:e99a2684-->
- **Konvexe Hülle (von innen aufgebaut):** Die kleinste konvexe Menge, die $S$ enthält, besteht aus allen Konvexkombinationen:
  $$\mathrm{conv}(S) = \left\{\sum_{i=1}^m \theta_i \mathbf{x}_i : \mathbf{x}_i \in S,\; \theta_i \ge 0,\; \sum_{i=1}^m \theta_i = 1\right\}$$

<!--en:e2ff44f4-->
::: {.slide title="Kriterium nullter Ordnung: Sehnen liegen über dem Graphen"}

<!--en:4015f8bd-->
[Konvexe Funktionen]{.kicker}

<!--en:f2ee8071-->
Eine Funktion $f : C \to \mathbb{R}$ auf einer konvexen Menge $C$ heißt **konvex**, wenn jede Sehne auf oder über ihrem Graphen liegt ($\theta \in [0, 1]$):

<!--en:d8437643-->
- **Epigraphen:** $f$ ist genau dann konvex, wenn $\mathrm{epi}(f) = \{(\mathbf{x}, t) : t \ge f(\mathbf{x})\}$ eine konvexe Menge ist.

<!--en:02ecbaff-->
- **Subniveaumengen:** Jede Subniveaumenge $\{\mathbf{x} : f(\mathbf{x}) \le c\}$ einer konvexen Funktion ist konvex (die Umkehrung scheitert jedoch bei $\sqrt{|x|}$).

<!--en:3fa07f14-->
::: {.slide title="Jedes lokale Minimum einer konvexen Funktion ist global"}

<!--en:b100cd5e-->
[Globale Optimalität]{.kicker}

<!--en:4ac05efc-->
Die Sehnenungleichung nullter Ordnung schließt lokale Minima oberhalb des globalen Minimums unmittelbar aus:

<!--en:b06df74f-->
- **Satz:** Ist $f$ auf einer konvexen Menge $C$ konvex, so ist jeder lokale Minimierer $\mathbf{x}^\star$ von $f$ auf $C$ ein globaler Minimierer.

<!--en:d36cceac-->
- **Beweis in drei Zeilen:** Für jedes $\mathbf{y} \in C$ und hinreichend kleines $\theta > 0$ liegt der Punkt $\mathbf{z}_\theta = (1 - \theta)\mathbf{x}^\star + \theta\mathbf{y}$ in der Umgebung, in der $\mathbf{x}^\star$ minimal ist:
  $$f(\mathbf{x}^\star) \le f(\mathbf{z}_\theta) \le (1 - \theta)f(\mathbf{x}^\star) + \theta f(\mathbf{y})$$

<!--en:84c3b383-->
- $(1-\theta)f(\mathbf{x}^\star)$ subtrahieren und durch $\theta > 0$ dividieren. Damit gilt $f(\mathbf{x}^\star) \le f(\mathbf{y})$ für alle $\mathbf{y} \in C$.

<!--en:20076d86-->
::: {.slide title="Konkurrierende Einzugsgebiete beim geneigten Doppelmuldenpotential"}

<!--en:b100cd5e-2-->
[Globale Optimalität]{.kicker}

<!--en:f32d4238-->
Nach $600$ Schritten Gradientenabstieg ($\eta = 0.05$) von $500$ gleichverteilt gezogenen Startpunkten in $[-1.5, 1.5]$ zeigt sich der Unterschied zwischen dem einen Einzugsgebiet von $(x - \tfrac12)^2$ und den beiden von $\phi(x) = (x^2 - 1)^2 + \tfrac12 x$:

<!--en:40b9f993-->
- Das lokale Maximum zwischen den Einzugsgebieten von $\phi$ ist die mittlere Nullstelle von $\phi'(x) = 4x^3 - 4x + \tfrac12 = 0$ bei $x_{\max} \approx 0.127$. Daher endet ein Anteil von $(1.5 - 0.127)/3 \approx 45.8\%$ der Läufe in der suboptimalen rechten Mulde.

<!--en:4a1cec23-->
::: {.slide title="Kriterium erster Ordnung: Tangenten beschränken f nach unten"}

<!--en:4015f8bd-2-->
[Konvexe Funktionen]{.kicker}

<!--en:76b2b73b-->
Ist $f$ auf einer konvexen Menge $C$ differenzierbar, so gilt Konvexität genau dann, wenn die tangentiale Hyperebene an jedem Punkt $\mathbf{x} \in C$ eine **globale untere Schranke** für $f$ bei allen $\mathbf{y} \in C$ liefert:

<!--en:9744116a-->
- **Anwendung in Optimierungsbeweisen:** Ein *einziger* Gradient an der Stelle $\mathbf{x}$ beschränkt $f(\mathbf{y})$ im gesamten Definitionsbereich nach unten.

<!--en:53dacb90-->
- **Unmittelbares Optimalitätskriterium:** Gilt $\nabla f(\mathbf{x}^\star) = \mathbf{0}$, so folgt $f(\mathbf{y}) \ge f(\mathbf{x}^\star) + \mathbf{0}^\top(\mathbf{y} - \mathbf{x}^\star) = f(\mathbf{x}^\star)$ für jedes $\mathbf{y} \in C$.

<!--en:20ee7bd0-->
::: {.slide title="Kriterium zweiter Ordnung: positiv semidefinite Hesse-Matrix"}

<!--en:4015f8bd-3-->
[Konvexe Funktionen]{.kicker}

<!--en:e89c6ac0-->
Ist $f$ auf einer *offenen* konvexen Menge $C$ zweimal differenzierbar, so ist $f$ genau dann konvex, wenn die Hesse-Matrix überall positiv semidefinit ist:

<!--en:0f766e7c-->
- **Geometrische Deutung:** Jeder eindimensionale Schnitt $g(t) = f(\mathbf{x} + t\mathbf{v})$ hat eine nichtnegative zweite Ableitung $g''(0) = \mathbf{v}^\top \nabla^2 f(\mathbf{x})\mathbf{v} \ge 0$. Die Fläche krümmt sich also in jeder Richtung nach oben.

<!--en:01ffc71f-->
- **Wichtige Beispiele mit kurzem Nachweis:**
  - *Quadratische Funktionen:* Für $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{x}$ gilt $\nabla^2 f(\mathbf{x}) = \mathbf{A}$, also ist die Funktion konvex $\iff \mathbf{A} \succeq \mathbf{0}$.
  - *Lineare kleinste Quadrate:* Für $f(\mathbf{w}) = \tfrac12\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ gilt $\nabla^2 f(\mathbf{w}) = \mathbf{X}^\top\mathbf{X}$, und es gilt $\mathbf{v}^\top\mathbf{X}^\top\mathbf{X}\mathbf{v} = \|\mathbf{X}\mathbf{v}\|^2 \ge 0$ für alle $\mathbf{v}$.
  - *Skalare Grundfunktionen:* $e^x$ ($f'' = e^x > 0$), $-\log x$ auf $x > 0$ ($f'' = 1/x^2 > 0$) und $x\log x$ auf $x > 0$ ($f'' = 1/x > 0$).

<!--en:10f50a00-->
::: {.slide title="Beweis der Äquivalenz der drei Kriterien"}

<!--en:7a33f583-->
[Äquivalenzbeweis]{.kicker}

<!--en:12728b35-->
Alle drei Kriterien kennzeichnen dieselbe Klasse glatter Funktionen:

<!--en:23c30b95-->
- **Nullte $\Rightarrow$ erste Ordnung:** Die Sehnenungleichung bei $\mathbf{x} + \theta(\mathbf{y} - \mathbf{x})$ nach Subtraktion des Ausgangswerts durch $\theta$ dividieren und zum Grenzwert $\theta \downarrow 0$ übergehen.

<!--en:26d83a91-->
- **Erste $\Rightarrow$ nullte Ordnung:** Die Tangentenschranke bei $\mathbf{z} = \theta\mathbf{x} + (1-\theta)\mathbf{y}$ in Richtung $\mathbf{x}$ und $\mathbf{y}$ anwenden und die gewichtete Summe bilden.

<!--en:c5b90c0c-->
- **Erste $\iff$ zweite Ordnung:** Nach eindimensionaler Taylor-Entwicklung entlang $\mathbf{x} + t(\mathbf{y}-\mathbf{x})$ beträgt der Abstand über der Tangente $\tfrac12(\mathbf{y}-\mathbf{x})^\top\nabla^2 f(\mathbf{z}_\tau)(\mathbf{y}-\mathbf{x}) \ge 0$.

<!--en:559519a0-->
::: {.slide title="Numerische Prüfung mit Sehnen, Tangenten und Hesse-Matrizen"}

<!--en:63718f6d-->
[Numerische Überprüfung]{.kicker}

<!--en:b3fb47aa-->
$1{,}000$ zufällige Punktepaare in $[-1.5, 1.5]^d$ dienen zur Prüfung aller drei Bedingungen für die kleinsten Quadrate $\tfrac12\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ und das Doppelmuldenpotential $\phi(x)$:

<!--en:ea5f41fe-->
**Asymmetrie numerischer Prüfungen:** Eine einzige positive Sehnen- oder Tangentendifferenz *widerlegt* die Konvexität des Doppelmuldenpotentials ($\phi''(0) = -4 < 0$). Für die kleinsten Quadrate ist dagegen der analytische Nachweis erforderlich: $\mathbf{X}^\top\mathbf{X} \succeq \mathbf{0}$.

<!--en:f4e8803e-->
::: {.slide title="Strikte und starke Konvexität und die quadratische untere Schranke"}

<!--en:4ee0f87f-->
[Starke Konvexität]{.kicker}

<!--en:87a72c3c-->
Selbst strikte Konvexität erlaubt beliebig flache Bereiche, in denen der Gradientenabstieg kaum vorankommt (etwa bei $f(x) = x^4$ nahe $x = 0$, wo $f''(0) = 0$ gilt). **$\mu$-starke Konvexität** ($\mu > 0$) verlangt, dass $f(\mathbf{x}) - \tfrac{\mu}{2}\|\mathbf{x}\|^2$ konvex ist:

<!--en:cd37f07f-->
- **Matrixkriterium:** Für zweimal differenzierbares $f$ gilt $\mu$-starke Konvexität genau dann, wenn überall $\nabla^2 f(\mathbf{x}) \succeq \mu\mathbf{I}$ gilt (jeder Eigenwert der Hesse-Matrix beträgt mindestens $\mu$).

<!--en:ec5e6b50-->
- **Existenz und Eindeutigkeit:** Jede stetige $\mu$-stark konvexe Funktion auf einer nichtleeren abgeschlossenen konvexen Menge hat beschränkte Subniveaumengen und damit einen *eindeutigen* globalen Minimierer $\mathbf{x}^\star$.

<!--en:f78a8d75-->
- **Warum Gewichtszerfall die Optimierung erleichtert:** Die Addition von $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ zu einer konvexen Verlustfunktion macht die Zielfunktion $\lambda$-stark konvex.

<!--en:16d3105b-->
::: {.slide title="Krümmungsschranken und Konditionszahl"}

<!--en:4ee0f87f-2-->
[Starke Konvexität]{.kicker}

<!--en:f1ca05d7-->
Zusammen schließen $\mu$-starke Konvexität und $L$-Glattheit ($\mu\mathbf{I} \preceq \nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$) die Funktion $f(\mathbf{y})$ an jedem Punkt $\mathbf{x}$ zwischen zwei tangential anliegenden Parabeln ein:

<!--en:92cbef3c-->
- **Quadratische untere Schranke ($\mu$):**
  $$f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) + \tfrac{\mu}{2}\|\mathbf{y}-\mathbf{x}\|^2$$

<!--en:8dd0cb9b-->
- **Quadratische obere Schranke ($L$):**
  $$f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) + \tfrac{L}{2}\|\mathbf{y}-\mathbf{x}\|^2$$

<!--en:16168754-->
- **Allgemeine Konditionszahl:** $\kappa = L/\mu \ge 1$ (rechts gilt $\mu = 1, L = 4 \implies \kappa = 4$ für $\tfrac12 x^2 + 3\log\cosh x$).

<!--en:0f324e80-->
::: {.slide title="Subgradienten: stützende Hyperebenen an Knicken"}

<!--en:50f1e4aa-->
[Nichtglatte konvexe Funktionen]{.kicker}

<!--en:0e1c0b7d-->
Nichtglatte konvexe Verlustfunktionen wie $\|\mathbf{x}\|_1$ und der Hinge-Verlust $\max(0, 1 - z)$ haben Knicke, an denen kein Gradient existiert. Ein Vektor $\mathbf{g} \in \mathbb{R}^d$ heißt **Subgradient** von $f$ an der Stelle $\mathbf{x}$, wenn er für alle $\mathbf{y}$ eine globale Stützhyperebene definiert:

<!--en:a940a468-->
- **Subdifferential $\partial f(\mathbf{x})$:** Menge aller Subgradienten an der Stelle $\mathbf{x}$ (z. B. $\partial|x|(0) = [-1, 1]$).

<!--en:509d0cb4-->
- **Universelles Optimalitätskriterium:** $\mathbf{x}^\star$ ist genau dann ein globaler Minimierer von $f$, wenn $\mathbf{0} \in \partial f(\mathbf{x}^\star)$ gilt.

<!--en:d7b945fa-->
::: {.slide title="Jensen-Ungleichung: Erwartungswerte konvexer Funktionen"}

<!--en:fcd9e0bd-->
[Folgen der Konvexität]{.kicker}

<!--en:f0f71b66-->
Für jede konvexe Funktion $f : C \to \mathbb{R}$ und jeden Zufallsvektor $\mathbf{Z} \in C$ mit Mittelwert $\mathbf{m} = \mathbb{E}[\mathbf{Z}]$ gilt:

<!--en:0e181874-->
- **Beweis in drei Zeilen:** Einen Subgradienten $\mathbf{g} \in \partial f(\mathbf{m})$ wählen, sodass $f(\mathbf{Z}) \ge f(\mathbf{m}) + \mathbf{g}^\top(\mathbf{Z} - \mathbf{m})$ gilt. Beim Bilden des Erwartungswerts verschwindet der lineare Term $\mathbf{g}^\top\mathbb{E}[\mathbf{Z} - \mathbf{m}] = 0$.

<!--en:239bfef5-->
- **Gleichheitsfall bei strikter Konvexität:** Für strikt konvexes $f$ gilt Gleichheit genau dann, wenn $\mathbf{Z} = \mathbf{m}$ fast sicher gilt.

<!--en:a7b455d2-->
::: {.slide title="Aus der Jensen-Ungleichung folgt KL-Divergenz ≥ 0"}

<!--en:e3be9381-->
[Informationstheorie]{.kicker}

<!--en:51b32d5b-->
Die Jensen-Ungleichung liefert grundlegende Ungleichungen für Wahrscheinlichkeitstheorie, Statistik und Variationsinferenz:

<!--en:284e65a3-->
- **Nichtnegativität der KL-Divergenz (Gibbs-Ungleichung):** Für Verteilungen $p, q$ mit gleichem Träger, $R = q(Z)/p(Z)$ und $Z \sim p$ gilt die Jensen-Ungleichung für die strikt konvexe Funktion $-\log$:
  $$D_{\mathrm{KL}}(p \,\|\, q) = \mathbb{E}_p\big[-\log R\big] \ge -\log\big(\mathbb{E}_p[R]\big) = -\log\sum_x p(x)\frac{q(x)}{p(x)} = -\log 1 = 0$$
  Wegen der strengen Konvexität von $-\log$ gilt $D_{\mathrm{KL}}(p \,\|\, q) = 0 \iff R = 1$ fast sicher $\iff p = q$.

<!--en:3bf2bb19-->
- **Ungleichung vom arithmetischen und geometrischen Mittel:** Die Jensen-Ungleichung für die konkave Funktion $\log$ liefert $\frac{1}{N}\sum_{i=1}^N x_i \ge \big(\prod_{i=1}^N x_i\big)^{1/N}$ für $x_i > 0$.

<!--en:582a684b-->
- **Untere Evidenzschranke (ELBO):** Die Abschätzung $\log \mathbb{E}_{q(z)}[p(x, z)/q(z)] \ge \mathbb{E}_{q(z)}[\log(p(x, z)/q(z))]$ liegt Variationsautoencodern und dem EM-Verfahren zugrunde.

<!--en:96f818c0-->
::: {.slide title="Monte-Carlo-Prüfung der Jensen-Ungleichung und der Abnahme der KL-Divergenz"}

<!--en:e3be9381-2-->
[Informationstheorie]{.kicker}

<!--en:d1fb8799-->
$10^6$ standardnormalverteilte Stichproben $Z \sim \mathcal{N}(0, 1)$ bestätigen die feste Differenz in der Jensen-Ungleichung für $f(x) = e^x$. Beim Mischen von $q$ in $p$ zeigt sich die quadratische Konvergenz $O(\varepsilon^2)$ der KL-Divergenz:

<!--en:a21b110e-->
- Für $Z \sim \mathcal{N}(0, 1)$ gilt $\mathbb{E}[e^Z] = e^{1/2} \approx 1.6487 > 1 = e^{\mathbb{E}[Z]}$.

<!--en:542a5403-->
- Eine Verkleinerung um den Faktor $10\times$ bei $\varepsilon$ verringert die Divergenz $D_{\mathrm{KL}}(p \,\|\, (1-\varepsilon)p + \varepsilon q) \approx \frac{\varepsilon^2}{2}\sum_x \frac{(q(x)-p(x))^2}{p(x)}$ um den Faktor $100\times$.

<!--en:780a0661-->
::: {.slide title="Warum Konvexität allein langsamen Abstieg erlaubt: x² und x⁴"}

<!--en:9050d033-->
[Konvergenzraten]{.kicker}

<!--en:e3c6fb43-->
Sowohl $f(x) = x^2$ als auch $f(x) = x^4$ sind glatt und strikt konvex mit eindeutigem globalem Minimierer $x^\star = 0$. Nur $x^2$ ist stark konvex. Beim Gradientenabstieg ($\eta = 1/8$, $x_0 = 1$) zeigt sich ein großer Geschwindigkeitsunterschied:

<!--en:2796c0b6-->
- Bei $x^2$ ($\mu = 2$) fällt $x_k = (0.75)^k$ unter $10^{-6}$, und zwar in $49$ Schritten. Bei $x^4$ ($f''(0) = 0$) nähert sich der Kontraktionsfaktor der Eins, $1 - 4\eta x_k^2 \to 1$, und es gilt $x_k \sim 1/\sqrt{k}$ ($f(x_k) \sim 1/k^2$).

<!--en:165fdd8a-->
::: {.slide title="Beweis der Rate O(1/k) für glatte konvexe Funktionen"}

<!--en:92320639-->
[Konvergenzbeweis]{.kicker}

<!--en:81d7acc1-->
Für jede konvexe, $L$-glatte Funktion $f$ mit Minimierer $\mathbf{x}^\star$ erfüllt der Gradientenabstieg mit $\eta = 1/L$ die Schranke

<!--en:d162cd14-->
- **Schritt 1: Abstiegslemma und Tangentenschranke kombinieren:** Wegen $f(\mathbf{x}_k) - f^\star \le \nabla f(\mathbf{x}_k)^\top(\mathbf{x}_k - \mathbf{x}^\star)$ gilt
  $$f(\mathbf{x}_{j+1}) - f^\star \le \nabla f(\mathbf{x}_j)^\top(\mathbf{x}_j - \mathbf{x}^\star) - \frac{1}{2L}\|\nabla f(\mathbf{x}_j)\|^2$$

<!--en:34ed45c4-->
- **Schritt 2: Zum Quadrat ergänzen und die Abnahme des quadrierten Abstands ablesen:**
  $$f(\mathbf{x}_{j+1}) - f^\star \le \frac{L}{2}\Big(\|\mathbf{x}_j - \mathbf{x}^\star\|^2 - \big\|\mathbf{x}_j - \tfrac{1}{L}\nabla f(\mathbf{x}_j) - \mathbf{x}^\star\big\|^2\Big) = \frac{L}{2}\Big(\|\mathbf{x}_j - \mathbf{x}^\star\|^2 - \|\mathbf{x}_{j+1} - \mathbf{x}^\star\|^2\Big)$$

<!--en:bb5f595e-->
- **Schritt 3: Teleskopsumme über $j = 0, \ldots, k-1$ bilden:** Da $f(\mathbf{x}_j)$ monoton fällt, gilt $k(f(\mathbf{x}_k) - f^\star) \le \sum_{j=0}^{k-1}(f(\mathbf{x}_{j+1}) - f^\star) \le \frac{L}{2}\|\mathbf{x}_0 - \mathbf{x}^\star\|^2$ unabhängig von der Dimension $d$.

<!--en:52dd19f1-->
::: {.slide title="Beweis der linearen Rate bei starker Konvexität"}

<!--en:92320639-2-->
[Konvergenzbeweis]{.kicker}

<!--en:73099295-->
Mit $\mu$-starker Konvexität verbessert sich die sublineare Rate $O(1/k)$ zu geometrischer (linearer) Konvergenz:

<!--en:ea749ebe-->
- **Schritt 1: Gradientennorm mithilfe der Differenz zum Optimalwert nach unten beschränken:** Beide Seiten der quadratischen unteren Schranke $f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) + \tfrac{\mu}{2}\|\mathbf{y}-\mathbf{x}\|^2$ über $\mathbf{y}$ minimieren (die rechte Seite ist bei $\mathbf{y} = \mathbf{x} - \tfrac{1}{\mu}\nabla f(\mathbf{x})$ minimal):
  $$f^\star \ge f(\mathbf{x}) - \frac{1}{2\mu}\|\nabla f(\mathbf{x})\|^2 \implies \frac12\|\nabla f(\mathbf{x})\|^2 \ge \mu\big(f(\mathbf{x}) - f^\star\big)$$

<!--en:2526f1eb-->
- **Schritt 2: In das Abstiegslemma für $\eta = 1/L$ einsetzen:**
  $$f(\mathbf{x}_{k+1}) - f^\star \le \big(f(\mathbf{x}_k) - f^\star\big) - \frac{1}{2L}\|\nabla f(\mathbf{x}_k)\|^2 \le \left(1 - \frac{\mu}{L}\right)\big(f(\mathbf{x}_k) - f^\star\big)$$

<!--en:582851f3-->
- **Geometrische Kontraktion:** Nach $k$ Schritten gilt $f(\mathbf{x}_k) - f^\star \le (1 - 1/\kappa)^k(f(\mathbf{x}_0) - f^\star)$. Die relative Genauigkeit $\epsilon$ ist nach $O(\kappa \log(1/\epsilon))$ Iterationen erreicht.

<!--en:4d93f125-->
::: {.slide title="Messung des tatsächlichen Kontraktionsfaktors bei quadratischen Funktionen"}

<!--en:63718f6d-2-->
[Numerische Überprüfung]{.kicker}

<!--en:2f4003d3-->
Beim Gradientenabstieg mit $\eta = 1/L$ für ein Problem der kleinsten Quadrate mit Eigenwerten $\mu = 1$ und $L = 10$ ($\kappa = 10$) schrumpft die Differenz zum minimalen Verlust tatsächlich um den Faktor $(1 - \mu/L)^2 = 0.81$ statt $1 - \mu/L = 0.90$:

<!--en:3928dd22-->
- **Warum der Faktor im Quadrat steht:** Bei $\eta = 1/L$ multipliziert der erste Schritt die Fehlerkomponente in der steifen Eigenrichtung mit $1 - L/L = 0$. Der verbleibende Abstand in der langsamen Richtung schrumpft pro Schritt um den Faktor $1 - \mu/L = 0.9$, die quadratische Differenz zum minimalen Verlust daher um den Faktor $(1 - \mu/L)^2 = 0.81$.

<!--en:0c6ba39e-->
::: {.slide title="Vier Rechenregeln zum Erkennen konvexer Verlustfunktionen"}

<!--en:1f29e515-->
[Rechenregeln für Konvexität]{.kicker}

<!--en:8afceaec-->
- **Nichtnegative Summen und affine Eingaben:** $\sum_{i=1}^m w_i f_i(\mathbf{x})$ ($w_i \ge 0$) und $\mathbf{w} \mapsto f(\mathbf{X}\mathbf{w} + \mathbf{b})$ erhalten Konvexität.

<!--en:18639ae6-->
- **Punktweises Maximum:** $\max_i f_i(\mathbf{x})$ ist konvex, denn $\mathrm{epi}(\max_i f_i) = \bigcap_i \mathrm{epi}(f_i)$ (damit sind der Hinge-Verlust und $\|\mathbf{w}\|_1$ konvex).

<!--en:b03d5ac7-->
- **Monotone Verkettung:** $h(g(\mathbf{x}))$ ist konvex, wenn $g$ konvex und $h$ konvex *und monoton wachsend* ist (ohne Monotonie ergibt sich etwa das nichtkonvexe Doppelmuldenpotential $(x^2 - 1)^2$).

<!--en:8d7280ca-->
::: {.slide title="Log-Sum-Exp: die Softmax-Kovarianz als Hesse-Matrix"}

<!--en:1f29e515-2-->
[Rechenregeln für Konvexität]{.kicker}

<!--en:d9ebe4a6-->
Bei der Kreuzentropie für mehrere Klassen hat $\mathrm{lse}(\mathbf{x}) = \log\sum_{i=1}^d e^{x_i}$ den Gradienten $\nabla\mathrm{lse}(\mathbf{x}) = \mathbf{s} = \mathrm{softmax}(\mathbf{x})$. Die Hesse-Matrix ist die Kovarianzmatrix eines One-Hot-Klassenvektors $\mathbf{e}_Y$ aus der Verteilung $Y \sim \mathbf{s}$:

<!--en:94281aa2-->
- Da $v_Y$ beschränkt ist, gilt überall $\lambda_{\max}(\nabla^2\mathrm{lse}(\mathbf{x})) \le \tfrac12$ ($\tfrac12$-glatt). In Richtung $\mathbf{1}$ verschwindet die Krümmung (Verschiebungsinvarianz). Bei sicheren Vorhersagen ist sie nahezu null.

<!--en:d6629923-->
::: {.slide title="Warum tiefe neuronale Netze nichtkonvex sind"}

<!--en:c74a1608-->
[Jenseits der Konvexität]{.kicker}

<!--en:ad1e3d46-->
Schon bei einem skalaren Netz mit zwei Gewichten $f(a, b) = (ab - 1)^2$ geht die Konvexität verloren: Die Verkettung mit einer nichtaffinen inneren Schicht erfüllt unsere Rechenregeln nicht:

<!--en:5befc1c4-->
- **Nichtkonvexe Menge der Minimierer:** Die globalen Minimierer bilden die Hyperbel $\{ab = 1\}$. Der Mittelpunkt zwischen $(1, 1)$ und $(-1, -1)$ ist der Ursprung $(0, 0)$, ein Sattelpunkt mit $f(0, 0) = 1$ und den Eigenwerten $\pm 2$ der Hesse-Matrix.

<!--en:4dfbf226-->
- **Permutationssymmetrie in tiefen Netzen:** Eine Permutation der $m$ versteckten Einheiten einer Schicht liefert insgesamt $m!$ verschiedene Parametervektoren mit identischem Verlust. An ihren Mittelpunkten fallen die versteckten Merkmale zusammen und der Verlust steigt.

<!--en:7339282e-->
::: {.slide title="Die Polyak-Łojasiewicz-Bedingung und impliziter Bias"}

<!--en:c74a1608-2-->
[Jenseits der Konvexität]{.kicker}

<!--en:65b591a7-->
Im Beweis für starke Konvexität haben wir Konvexität *nur* zur Herleitung der **Polyak-Łojasiewicz-Ungleichung (PL-Ungleichung)** verwendet:

<!--en:48dedfb4-->
- **Nichtkonvexe PL-Landschaften:** Für $f(x) = x^2 + 3\sin^2 x$ gilt $f''(x) \in [-4, 8]$ (nichtkonvex). Trotzdem erfüllt die Funktion die PL-Bedingung mit $\mu \approx 0.1755$, und der Gradientenabstieg konvergiert linear. Breite überparametrisierte Netze erfüllen lokale PL-Bedingungen um Mannigfaltigkeiten interpolierender Minimierer. Dort wählt der **implizite Bias** des Optimierers eine bestimmte Lösung aus (z. B. kleinste $\ell_2$-Norm oder maximale Margin).

<!--en:c812d48e-->
::: {.slide title="Verständnisfragen zu Konvexität, Konvergenzraten und Subgradienten"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:a36280b3-->
Fragen zu Konvexitätskriterien, Subgradienten und Konvergenzraten:

<!--en:180ee966-->
1. Gegeben sei $f(x) = x\log x$ auf $x > 0$. Ist $f$ auf $(0, \infty)$ konvex? Ist die Funktion $\mu$-stark konvex auf $(0, 1]$? Ist sie $\mu$-stark konvex auf $(0, \infty)$?

<!--en:ee1df048-->
2. Wie lautet das Subdifferential $\partial f(z)$ des Hinge-Verlusts $f(z) = \max(0, 1 - z)$ am Knick $z = 1$?

<!--en:5a73af4e-->
3. Wie viele Schritte braucht der Gradientenabstieg bei $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ mit $\eta = 1/L = 10^{-3}$, um $f(\mathbf{x}_k) - f^\star$ auf das $10^{-6}$-Fache zu verringern? Wie unterscheiden sich die Vorhersagen der allgemeinen Schranke $(1 - \mu/L)^k$ und der exakten Rate $(1 - \mu/L)^{2k}$ für diese quadratische Funktion?

<!--en:ca5ab381-->
**Lösungen:** **(1)** Wegen $f''(x) = 1/x > 0$ ist $f$ auf $(0, \infty)$ strikt konvex. Auf $(0, 1]$ gilt $f''(x) \ge 1$, also ist die Funktion dort $1$-stark konvex. Auf $(0, \infty)$ gilt $f''(x) = 1/x \to 0$ für $x \to \infty$, sodass kein positives $\mu > 0$ eine globale untere Schranke für $f''(x)$ ist. **(2)** Die linksseitige Ableitung ist $f'_-(1) = -1$ und die rechtsseitige $f'_+(1) = 0$, also gilt $\partial f(1) = [-1, 0]$. **(3)** Die allgemeine Schranke $(1 - 10^{-3})^k \le 10^{-6}$ ergibt $k \approx 13{,}815$ Schritte. Mit der exakten quadratischen Kontraktion $(1 - 10^{-3})^{2k}$ halbiert sich die Zahl auf $\approx 6{,}908$ Schritte (mit der optimalen Schrittweite $\eta^\star = 2/(\mu+L)$ nochmals auf $3{,}454$).

<!--en:cb6f66c4-->
::: {.slide title="Zusammenfassung: Konvexität und Optimierungsgarantien"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:66c27d60-->
| Funktionenklasse | Definierende mathematische Bedingung | Eigenschaften der Minimierer | Rate des Gradientenabstiegs ($\eta = 1/L$) | Typische Beispiele |

<!--en:ce0bffd7-->
| **Konvex ($L$-glatt)** | $f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$ ($\nabla^2 f \succeq \mathbf{0}$) | Lokale Minima sind global, die Menge der Minimierer ist konvex | $f(\mathbf{x}_k) - f^\star \le \frac{L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2}{2k}$ | Kleinste Quadrate, logistischer Verlust und Softmax-Kreuzentropie |

<!--en:6c5c2cde-->
| **Streng konvex** | Strikte Sehnen- beziehungsweise Tangentenungleichung für $\mathbf{x} \neq \mathbf{y}$ | Höchstens ein globaler Minimierer | Sublinear bei $f''(\mathbf{x}^\star) = 0$ (z. B. $x^4 \sim 1/k^2$) | $x^4$, $e^x$, KL-Divergenz $D_{\mathrm{KL}}(p \,\|\, q)$ |

<!--en:36cf1f76-->
| **$\mu$-stark konvex** | $\nabla^2 f(\mathbf{x}) \succeq \mu\mathbf{I} \succ \mathbf{0}$ ($\kappa = L/\mu$) | Eindeutiger globaler Minimierer $\mathbf{x}^\star$ existiert stets | $f(\mathbf{x}_k) - f^\star \le (1 - 1/\kappa)^k(f_0 - f^\star)$ | Beliebiger konvexer Verlust mit Ridge-Gewichtszerfall $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ |

<!--en:9812987c-->
| **Nichtglatt und konvex** | $\mathbf{g} \in \partial f(\mathbf{x})$ definiert eine Stützhyperebene | $\mathbf{x}^\star$ optimal $\iff \mathbf{0} \in \partial f(\mathbf{x}^\star)$ | $O(1/\sqrt{k})$ bei abnehmenden Subgradientenschrittweiten | $\ell_1$-Strafterm $\|\mathbf{w}\|_1$, Hinge-Verlust der SVM |

<!--en:90b66439-->
| **Nichtkonvex mit PL-Bedingung** | $\tfrac12\|\nabla f(\mathbf{x})\|^2 \ge \mu(f(\mathbf{x}) - f^\star)$ | Jeder stationäre Punkt ist ein globales Minimum | $f(\mathbf{x}_k) - f^\star \le (1 - \mu/L)^k(f_0 - f^\star)$ | $x^2 + 3\sin^2 x$, lokal auch breite interpolierende Netze |
