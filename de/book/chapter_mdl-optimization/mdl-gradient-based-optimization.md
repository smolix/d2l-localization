<!--en:df8214b0-->
# Gradientenabstieg und Krümmung

<!--en:384ebabe-->
In der Praxis wählen wir die Lernrate meist durch Ausprobieren. Wir setzen einen Wert an, beobachten den Verlust und senken die Lernrate, wenn das Training divergiert. Da jeder Schritt entgegen dem Gradienten des Verlusts verläuft, legt der Gradient die Richtung fest. Die Lernrate bestimmt nur, wie weit wir gehen. Wählen wir die Schrittlänge ungünstig, wird das Training quälend langsam oder instabil. Wie weit können wir einem Gradientenschritt trauen, und wie viele Iterationen zwingt uns diese Grenze auf?

<!--en:3cda0bea-->
Schon eine Funktion zweier Variablen zeigt das Problem. Betrachten wir die elliptische Mulde $f(x, y) = \tfrac12(x^2 + 10\,y^2)$. In $y$-Richtung ist ihre Krümmung zehnmal so groß wie in $x$-Richtung. Wir starten bei $(2, 1)$ mit $f = 7$ und $\nabla f = (2, 10)$. Ein Schritt der Weite $\eta$ entgegen dem Gradienten führt zu $(2 - 2\eta,\, 1 - 10\eta)$. Mit $\eta = 0.01$ sinkt der Verlust kaum, nämlich auf $6.01$. Mit $\eta = 0.1$ fällt er deutlich auf $1.62$. Mit $\eta = 0.25$ *steigt* er auf $12.4$. Alle drei Schritte zeigten bergab. Nur ihre Länge war verschieden.

<!--en:ab4f8d17-->
Die Ursache ist die Krümmung. Ein Schritt, der entlang der steilen $y$-Achse kurz genug ist, bringt uns entlang der flachen $x$-Achse kaum voran. Selbst mit der besten festen Lernrate brauchen wir $69$ Iterationen, um den Fehler um den Faktor $10^6$ zu verkleinern und damit sechs Stellen Genauigkeit zu gewinnen. Erhöhen wir das Krümmungsverhältnis auf $1000$ wie im wiederkehrenden Beispiel dieses Kapitels, $\tfrac12(x_1^2 + 1000\,x_2^2)$, braucht der Gradientenabstieg $6{,}908$ Schritte. Mit Momentum sinkt die Zahl auf $315$. Wir rechnen von Hand mit der weniger steilen Mulde. An ihrer Hesse-Matrix $\mathrm{diag}(1, 10)$ lassen sich alle Zahlen leicht nachprüfen. Für das steifere Beispiel geben wir jeweils die entsprechenden Schrittzahlen an.

<!--en:8bf0f06f-->
Unsere Analyse hat drei Teile. Zunächst definieren wir Glattheit. Damit gewinnen wir aus der lokalen linearen Näherung in :numref:`sec_mdl-multivariable_calculus` eine Garantie für endliche Schritte, das Abstiegslemma :eqref:`eq_mdl-opt-descent-lemma`. Dann berechnen wir für quadratische Funktionen den Verlauf des Gradientenabstiegs exakt. Das Verhältnis der Krümmungen erweist sich als Konditionszahl und bestimmt die optimale Schrittweite :eqref:`eq_mdl-opt-optimal-step`. Schließlich führen wir Momentum ein und leiten das Konvergenzgesetz mit $\sqrt\kappa$ her, siehe :eqref:`eq_mdl-opt-hb-rate`. Die Verfahren in diesem Abschnitt berücksichtigen die Krümmung, ohne sie zu modellieren. Verfahren, die sie direkt modellieren, etwa das Newton-Verfahren, behandeln wir in :numref:`sec_mdl-curvature-methods`. Hier setzen wir außerdem exakte Gradienten voraus. Die verrauschten Gradienten des Minibatch-Trainings untersuchen wir in :numref:`sec_mdl-stochastic-gradient-methods`. Der folgende Code verwendet ausschließlich NumPy. So lässt sich jede Aktualisierung direkt mit ihrer Gleichung vergleichen.

<!--en:7ad2277a-->
## Abstiegsrichtungen und Glattheit

<!--en:c6f64149-->
Bei jedem Gradientenschritt entscheiden wir, in welche Richtung wir gehen und wie weit. Die erste Antwort liefert unmittelbar die Taylor-Entwicklung erster Ordnung. Für die zweite brauchen wir eine Annahme darüber, wie schnell sich der Gradient selbst ändern kann. Damit wird aus einer lokalen linearen Näherung eine Garantie für Schritte endlicher Länge.

<!--en:2badfefb-->
### Die Richtung des steilsten Abstiegs

<!--en:ffdb048f-->
Im Eingangsbeispiel sind wir in Richtung $-\nabla f$ gegangen, ohne zu fragen, ob eine andere Richtung besser wäre. Warum der negative Gradient naheliegt, zeigt die Entwicklung von $f : \mathbb{R}^d \to \mathbb{R}$ bis zur ersten Ordnung. Ein kleiner Schritt der Weite $\eta > 0$ in Richtung $\mathbf{d}$ ändert den Funktionswert um

<!--en:085324e4-->
In erster Ordnung ist die Änderung gleich $\eta$ mal der Richtungsableitung $\nabla f(\mathbf{x})^\top \mathbf{d}$. Wir nennen $\mathbf{d}$ eine **Abstiegsrichtung** (engl. descent direction) im Punkt $\mathbf{x}$, wenn $\nabla f(\mathbf{x})^\top \mathbf{d} < 0$ gilt.

<!--en:f16c6ba1-->
**Satz (Abstiegsrichtungen und Richtung des steilsten Abstiegs).** *Sei $f$ im Punkt $\mathbf{x}$ differenzierbar mit $\nabla f(\mathbf{x}) \neq \mathbf{0}$.*

<!--en:f51e7d57-->
1. *Gilt $\nabla f(\mathbf{x})^\top \mathbf{d} < 0$, so gilt $f(\mathbf{x} + \eta\,\mathbf{d}) < f(\mathbf{x})$ für alle hinreichend kleinen $\eta > 0$.*

<!--en:915e23af-->
2. *Unter allen Einheitsvektoren minimiert allein der normierte negative Gradient die Richtungsableitung:*

<!--en:9fda6ef0-->
**Beweis.** Für die erste Aussage setzen wir $\varphi(\eta) = f(\mathbf{x} + \eta\,\mathbf{d})$. Nach der Kettenregel gilt $\varphi'(0) = \nabla f(\mathbf{x})^\top \mathbf{d} < 0$. Der Differenzenquotient $(\varphi(\eta) - \varphi(0))/\eta$ ist somit für alle hinreichend kleinen $\eta > 0$ strikt negativ. Daraus folgt $\varphi(\eta) < \varphi(0)$. Für die zweite Aussage liefert die Cauchy-Schwarz-Ungleichung $\nabla f(\mathbf{x})^\top \mathbf{d} \ge -\|\nabla f(\mathbf{x})\|\,\|\mathbf{d}\| = -\|\nabla f(\mathbf{x})\|$ für jeden Einheitsvektor $\mathbf{d}$. Gleichheit gilt genau dann, wenn $\mathbf{d}$ ein negatives skalares Vielfaches von $\nabla f(\mathbf{x})$ ist. Der einzige solche Einheitsvektor ist $-\nabla f(\mathbf{x})/\|\nabla f(\mathbf{x})\|$, und er erreicht die Schranke. $\blacksquare$

<!--en:11b66a64-->
In unserer Mulde weist am Punkt $(2, 1)$ der Einheitsvektor $-(2, 10)/\sqrt{104} = (-0.196, -0.981)$ in die steilste Abstiegsrichtung, mit der Steigung $-\|\nabla f\| = -10.198$. Auch die Koordinatenrichtung $(-1, 0)$ weist bergab, allerdings mit der deutlich flacheren Steigung $-2$. Die Abstiegsrichtungen füllen sogar den ganzen offenen Halbraum der Vektoren aus, die mit $-\nabla f$ einen spitzen Winkel bilden. Der negative Gradient gibt unter ihnen die steilste Richtung an. Dieses Ergebnis betrifft Steigungen, also den Grenzfall verschwindend kurzer Schritte. Prüfen wir für vier Einheitsrichtungen, wie genau ihre Steigungen die tatsächliche Abnahme bei einem kleinen, endlichen Schritt mit $\eta = 0.01$ vorhersagen.

<!--en:ca1de743-->
Bei $\eta = 0.01$ weicht die tatsächliche Änderung in allen vier Fällen um höchstens ein Prozent vom Produkt aus $\eta$ und der jeweiligen Steigung ab. Das Modell erster Ordnung ist also für kurze Schritte verlässlich. Die Differenzierbarkeit garantiert nur, dass der Funktionswert bei hinreichend kleinen Schritten sinkt. Wie klein „hinreichend klein“ ist, untersuchen wir als Nächstes.

<!--en:9cc17f4a-->
### Glattheit und Abstiegslemma

<!--en:da9c59ce-->
Der **Gradientenabstieg** (engl. gradient descent) wiederholt den Schritt in Richtung des steilsten Abstiegs mit fester **Schrittweite** (oder Lernrate) $\eta > 0$:

<!--en:7a5226f7-->
In diesem Abschnitt bezeichnet $k$ den Iterationsindex, und wir schreiben kurz $\mathbf{g}_k = \nabla f(\mathbf{x}_k)$. Die Taylor-Entwicklung erster Ordnung begründet die Wahl der Richtung $-\mathbf{g}_k$. Wie weit wir ihr trauen können, sagt sie jedoch nicht. Bewegen wir uns so weit, dass sich der Gradient merklich ändert, beschreibt $\mathbf{g}_k$ die Funktion $f$ nicht mehr zuverlässig. Um eine endliche Schrittweite abzusichern, brauchen wir eine Schranke dafür, wie schnell sich der Gradient ändern kann.

<!--en:6293e642-->
**Definition ($L$-Glattheit).** Eine differenzierbare Funktion $f$ heißt **$L$-glatt**, wenn sich ihr Gradient pro Längeneinheit um höchstens $L$ ändert:

<!--en:c415e265-->
Ist $f$ zweimal stetig differenzierbar, so ist $L$-Glattheit gleichbedeutend damit, dass alle Eigenwerte der Hesse-Matrix in $[-L, L]$ liegen. Aus diesen Eigenwertschranken folgt :eqref:`eq_mdl-opt-smoothness` durch Integration der Hesse-Matrix längs der Strecke von $\mathbf{y}$ nach $\mathbf{x}$. Die Umkehrung folgt durch Grenzübergang im Differenzenquotienten (Aufgabe :exref:`ex_mdl-gradient-based-optimization-smoothness-constants`a). Somit begrenzt $L$ die Krümmung in jeder Richtung. Unsere Mulde hat die konstante Hesse-Matrix $\mathrm{diag}(1, 10)$, also gilt $L = 10$. Wächst die Krümmung einer Funktion hingegen unbeschränkt, gibt es keine globale Glattheitskonstante. Für $\tfrac14 x^4$ gilt $f''(x) = 3x^2$, also existiert kein endliches globales $L$. Diesen Fall behandeln wir gleich mit der Backtracking-Liniensuche.

<!--en:71281cbf-->
Bei einer glatten Funktion erhalten wir aus der lokalen Tangentialebene eine globale quadratische obere Schranke:

<!--en:3edee3cc-->
Geometrisch kann der Graph von $f$ seine Tangentialebene am Punkt $\mathbf{x}$ höchstens um $\tfrac{L}{2}$ mal das Quadrat des Abstands überragen. Bevor wir diese Ungleichung beweisen, setzen wir einen Gradientenschritt $\mathbf{y} = \mathbf{x} - \eta\,\mathbf{g}$ ein und betrachten die beiden gegenläufigen Terme. Die lineare Abnahme erster Ordnung $\eta\,\|\mathbf{g}\|^2$ wächst proportional zu $\eta$. Der quadratische Beitrag der Krümmung $\tfrac{L}{2}\eta^2\|\mathbf{g}\|^2$ wächst hingegen wie $\eta^2$. Für kleine $\eta$ überwiegt die lineare Abnahme. Bei $\eta = 2/L$ heben sich beide Terme genau auf.

<!--en:018bd01e-->
**Satz (Abstiegslemma).** *Ist $f$ eine $L$-glatte Funktion, so gilt für den Gradientenschritt :eqref:`eq_mdl-opt-gd-step`*

<!--en:0e55d43e-->
*Insbesondere nimmt $f$ strikt ab, wenn $0 < \eta < 2/L$ und $\mathbf{g}_k \neq \mathbf{0}$ gelten. Die garantierte Abnahme ist bei $\eta = 1/L$ am größten. Dann gilt $f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \tfrac{1}{2L}\|\mathbf{g}_k\|^2$.*

<!--en:ef0294e1-->
**Beweis.** Zuerst zeigen wir die quadratische obere Schranke :eqref:`eq_mdl-opt-quadratic-ceiling`. Wir beschränken $f$ auf die Strecke zwischen $\mathbf{x}$ und $\mathbf{y}$, indem wir $\psi(t) = f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))$ für $t \in [0, 1]$ setzen. Damit gilt $\psi'(t) = \nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x}))^\top (\mathbf{y} - \mathbf{x})$. Aus dem Hauptsatz der Differential- und Integralrechnung, der Cauchy-Schwarz-Ungleichung und der Lipschitz-Bedingung :eqref:`eq_mdl-opt-smoothness` folgt

<!--en:6ce317e1-->
Setzen wir den Gradientenschritt $\mathbf{y} = \mathbf{x}_k - \eta\,\mathbf{g}_k$ ein, erhalten wir

<!--en:97b8ac03-->
Der Faktor $\eta(1 - L\eta/2)$ ist genau dann strikt positiv, wenn $0 < \eta < 2/L$ gilt. Sein Maximum erreicht er bei $\eta = 1/L$. $\blacksquare$

<!--en:21f0cd1a-->
Im Beweis vergleichen wir den Gradienten bei $\mathbf{x}_k$ nur mit Gradienten auf der Strecke nach $\mathbf{x}_{k+1}$. Es genügt also, dass $\nabla f$ auf dieser Strecke $L$-Lipschitz-stetig ist. Prüfen wir das Lemma an unserer Mulde mit $\|\mathbf{g}\|^2 = 104$ und $L = 10$. Bei $\eta = 1/L = 0.1$ garantiert die Schranke $f \le 7 - 104/20 = 1.8$. Bei $\eta = 0.01$ garantiert sie $f \le 7 - 104 \cdot 0.01 \cdot 0.95 = 6.012$. Bei $\eta = 0.25 > 2/L$ wird der Faktor $1 - L\eta/2$ negativ. Die Schranke liegt dann über dem Ausgangswert und garantiert keine Abnahme mehr. Wie nahe liegen die Schranken an den Werten, die wir mit den drei Schritten tatsächlich erreichen?

<!--en:509e9a9b-->
Für die beiden stabilen Schritte sind die Schranken erstaunlich scharf. Der Wert $6.0102$ liegt knapp unter $6.012$, der Wert $1.62$ deutlich unter $1.8$. Bei $\eta = 0.25$ liegt die Schranke $13.5$ weit über dem Ausgangswert $f = 7$. Tatsächlich schießt der Schritt über das Ziel hinaus und erreicht $12.375$. In :numref:`fig_mdl-opt-descent-lemma-slice` sind beide Kurven als Funktionen von $\eta$ dargestellt. Entlang der Suchrichtung gilt $\mathbf{g}^\top\mathrm{diag}(1, 10)\,\mathbf{g} = 1004$. Die tatsächliche Zielfunktion ist also die Parabel $7 - 104\,\eta + 502\,\eta^2$ und ihre quadratische obere Schranke die Parabel $7 - 104\,\eta + 520\,\eta^2$. Die beiden liegen so dicht beieinander, weil der Gradient fast entlang der steifen Achse zeigt. Dort ist die Richtungskrümmung nahezu gleich $L$.

<!--en:9a138444-->
![Das Abstiegslemma am Eingangsbeispiel. In Richtung $-\mathbf{g}$ vom Punkt $(2,1)$ aus beschreibt die untere Parabel die tatsächliche Zielfunktion, die obere ihre quadratische obere Schranke $f-\eta\|\mathbf{g}\|^2+\tfrac{L}{2}\eta^2\|\mathbf{g}\|^2$. Bei $\eta=0$ berühren sie sich. Die Schranke ist bei $\eta=1/L=0.1$ minimal und garantiert $f\le 1.8$. Der Schritt erreicht $1.62$. Jenseits von $2/L=0.2$ steigt die Schranke über den Ausgangswert und garantiert keine Abnahme mehr. Bei $\eta=0.25$ steigt die Zielfunktion auf $12.4$.](../img/mdl-opt-descent-lemma-slice.svg)

<!--en:34f134c7-->
Die Schrittweite $\eta = 1/L$ minimiert also die quadratische obere Schranke. Der Wert von $f$ übersteigt sie garantiert nicht. In :numref:`sec_mdl-single_variable_calculus` entwickeln wir dasselbe geometrische Argument für eine Variable.

<!--en:c7df1c87-->
### Stationäre Punkte ohne Konvexität

<!--en:190a3f20-->
Erstaunlicherweise liefert das Abstiegslemma allein, ganz ohne Konvexitätsannahme, bereits eine quantitative Schranke für lange Iterationsfolgen. Der Gradient kann nicht beliebig lange groß bleiben.

<!--en:340e8d12-->
**Satz (Gradientenabstieg findet näherungsweise stationäre Punkte).** *Sei $f$ eine $L$-glatte, durch $f_{\mathrm{low}}$ nach unten beschränkte Funktion. Wir führen :eqref:`eq_mdl-opt-gd-step` mit $\eta = 1/L$ aus. Dann gilt für jedes $K \ge 1$*

<!--en:03e6b2b2-->
**Beweis.** Für $\eta = 1/L$ lässt sich das Abstiegslemma zu $\|\mathbf{g}_k\|^2 \le 2L\,(f(\mathbf{x}_k) - f(\mathbf{x}_{k+1}))$ umformen. Summieren wir über $k = 0, \ldots, K-1$, ergibt die Teleskopsumme auf der rechten Seite $2L\,(f(\mathbf{x}_0) - f(\mathbf{x}_K)) \le 2L\,(f(\mathbf{x}_0) - f_{\mathrm{low}})$. Das Minimum von $K$ Zahlen ist höchstens so groß wie ihr Mittelwert. Division durch $K$ liefert daher die Schranke. $\blacksquare$

<!--en:15085452-->
Unter den ersten $K$ Iterierten gibt es also mindestens eine, deren quadrierte Gradientennorm von der Ordnung $O(1/K)$ ist. Dafür brauchen wir lediglich eine untere Schranke für $f$ und eine Glattheitskonstante. Das Ergebnis setzt ein globales $L$ voraus. ReLU-Netze sind nicht glatt. Für sie dient die Schranke daher als idealisiertes Modell und bietet keine formale Garantie.

<!--en:80db5be2-->
Ein kleiner Gradient garantiert allerdings kein Minimum. Betrachten wir $f(x, y) = \tfrac12 x^2 + \cos y$. Diese Funktion ist durch $-1$ nach unten beschränkt und $1$-glatt, denn die Eigenwerte ihrer Hesse-Matrix $\mathrm{diag}(1, -\cos y)$ liegen in $[-1, 1]$. Starten wir den Gradientenabstieg mit $\eta = 1$ bei $(1, 0)$, fällt $x$ in einem Schritt auf $0$. Die Koordinate $y$ ändert sich nie, denn $\partial f/\partial y = -\sin y$ verschwindet bei $y = 0$. Der Algorithmus hält bei $(0, 0)$ mit $f = 1$ an. Das ist ein Sattelpunkt, an dem sich $f$ entlang $x$ nach oben und entlang $y$ nach unten krümmt, weit entfernt vom tatsächlichen Minimalwert $-1$ bei $(0, \pi)$. Der Satz gilt weiterhin, denn der Gradient verschwindet am Sattel. Er sagt uns jedoch nicht, welche Art von stationärem Punkt wir erreicht haben. Dass das Verfahren dort stehen bleibt, liegt am Startpunkt. Starten wir bei $(1, y_0)$ mit einem beliebigen kleinen $y_0 \neq 0$, so entfernt jeder Schritt $y \leftarrow y + \eta \sin y$ die Koordinate $y$ weiter von $0$, bis das Verfahren gegen ein Minimum konvergiert. Allgemein bilden die Startpunkte, von denen der Gradientenabstieg gegen einen Sattel mit einer Richtung negativer Krümmung konvergiert, eine Menge vom Maß null (:numref:`subsec_opt-intro-saddles`). Bei Zufallsinitialisierung vermeiden wir sie daher fast sicher, obwohl die Iterierten lange in der Nähe des Sattels bleiben können. Bei konvexen Zielfunktionen stellt sich diese Frage nicht. Jeder stationäre Punkt ist dort ein globales Minimum (:numref:`sec_mdl-convexity`).

<!--en:0e939d12-->
### Backtracking-Liniensuche

<!--en:4462ad16-->
Die Wahl $\eta = 1/L$ setzt voraus, dass wir $L$ kennen. Das ist in der Praxis selten der Fall. Selbst wenn ein globales $L$ existiert, variiert die lokale Krümmung in der Verlustlandschaft. Eine einzige globale Konstante ist dann zu vorsichtig. Eine **Liniensuche** (engl. line search) passt die Schrittweite zur Laufzeit an. Sie akzeptiert einen Probeschritt $\eta$ nur, wenn die tatsächliche Abnahme mindestens einen festen Anteil $c \in (0, 1)$ der Abnahme erreicht, die das Modell erster Ordnung vorhersagt. Das ist die **Armijo-Bedingung** oder Bedingung für hinreichende Abnahme :cite:`Armijo.1966`:

<!--en:dc35ef3d-->
Die rechte Seite ist die lineare Taylor-Näherung, deren Steigung wir mit dem Faktor $c$ verkleinern. Beim **Backtracking** beginnen wir mit einer großzügig gewählten Schrittweite $\eta_0$ und halbieren $\eta$ so oft, bis :eqref:`eq_mdl-opt-armijo` erfüllt ist. Nach dem Abstiegslemma gelingt der Test spätestens dann, wenn $1 - L\eta/2 \ge c$ oder gleichbedeutend $\eta \le 2(1-c)/L$ gilt. Die Schleife endet daher nach endlich vielen Halbierungen und liefert eine Schrittweite $\eta \ge \min\big(\eta_0,\, (1-c)/L\big)$. So erreichen wir die theoretische Schwelle bis auf einen Faktor zwei, ohne $L$ vorab kennen zu müssen.

<!--en:f8be104c-->
Dafür genügt sogar ein $L$ für das Gebiet, das die Iterierten tatsächlich durchlaufen. Betrachten wir die Funktion vierten Grades $f(x) = \tfrac14 x^4$ ohne globales $L$. Wir starten bei $x_0 = 3$ mit $c = \tfrac12$. Da Backtracking den Wert von $f$ nie erhöht, bleiben alle Iterierten im Intervall $[-3, 3]$. Dort gilt $f''(x) = 3x^2 \le 27$. Ein Probeschritt mit $\eta \le 1/27$ führt von $x$ nach $x(1 - \eta x^2)$, also zu einem Punkt zwischen $\tfrac23 x$ und $x$. Damit bleibt die ganze Strecke in $[-3, 3]$, und lokal genügt $L = 27$. Weit vom Ursprung ist die Krümmung groß, in seiner Nähe geht sie gegen null. Eine feste Schrittweite eignet sich daher kaum für die gesamte Trajektorie. Vergleichen wir vom Startpunkt $x_0 = 3$ aus eine feste Schrittweite $\eta = 0.3$, die nahe dem Ursprung völlig sicher wäre, mit Backtracking ab $\eta_0 = 1$.

<!--en:b9516343-->
Mit fester Schrittweite divergiert das Verfahren sofort. Schon nach sechs Iterationen gilt $|x_6| \approx 6.5 \times 10^{103}$. Die Aktualisierung $x \mapsto x(1 - \eta x^2)$ verkleinert $|x|$ genau dann, wenn $\eta x^2 < 2$ gilt. Vom Startpunkt $x_0 = 3$ aus konvergiert das Verfahren daher bei fester Schrittweite $\eta < 2/9 \approx 0.22$ und divergiert oberhalb dieser Schwelle. Die Stabilitätsschwelle liegt also bei $2/9$ und nicht bei $2/f''(x_0) = 2/27$. Die Regel $2/f''(x_0)$ gilt für quadratische Funktionen. Bei einem Schritt zum Ursprung zählt die mittlere Krümmung von $f''$ zwischen $0$ und $x$, also $x^2$, anstelle des Spitzenwerts $3x^2$. Das Abstiegslemma muss die größte Krümmung auf der Strecke berücksichtigen und sichert daher nur ein Drittel des tatsächlich stabilen Bereichs ab.

<!--en:8bbb6eb0-->
Backtracking akzeptiert im ersten Schritt hingegen $0.031$. Die zuvor erprobte Schrittweite $0.0625$ scheitert am Armijo-Test. Für diese Funktion vierten Grades gelingt er genau dann, wenn $\eta x^2 \le 0.456$ gilt (ungefähr $\eta \le 0.05$ für $x_0 = 3$). Je näher die Iterierte dem Ursprung kommt und je geringer die Krümmung wird, desto größer werden die akzeptierten Schrittweiten ($0.0625, 0.125, 0.25, 0.5, 1.0$, jeweils nahe bei $0.3/x^2$). Die Liniensuche prüft die lokale Krümmung bei jedem Schritt. Beim Training tiefer Netze kommt Backtracking selten zum Einsatz. Der Minibatch-Verlust ist zu stark verrauscht, um zwei Probeschritte zuverlässig zu vergleichen, und jeder Versuch kostet einen zusätzlichen Vorwärtsdurchlauf. Meist gibt stattdessen ein Lernratenplan die Schrittweite vor (:numref:`sec_scheduler`).

<!--en:c2afc66a-->
### Abgeschwächte Glattheit und Gradienten-Clipping

<!--en:74d2d8d9-->
Ein Lernratenplan kann nicht reagieren, wenn eine Iterierte in ein Gebiet hoher Krümmung gelangt. Beim Training schützt davor das Gradienten-Clipping. Unsere Funktion vierten Grades zeigt, warum es wirkt. Die Krümmung $f''(x) = 3x^2$ ist unbeschränkt, aber nur dort groß, wo auch der Gradient $f'(x) = x^3$ groß ist. Es gilt $3x^2 \le 4 + |x|^3$ für jedes $x$. Eine zweimal differenzierbare Funktion heißt **$(L_0, L_1)$-glatt**, wenn ihr Gradient die Krümmung in diesem Sinne beschränkt :cite:`Zhang.He.Sra.ea.2020`:

<!--en:a264463f-->
Dabei ist die Norm auf der linken Seite der größte Betrag eines Eigenwerts der Hesse-Matrix. Mit $L_1 = 0$ erhalten wir wieder die $L$-Glattheit. Unsere Funktion vierten Grades erfüllt die Bedingung mit $L_0 = 4$ und $L_1 = 1$. Unter dieser Bedingung ist die Schrittweite, die das Abstiegslemma absichert, keine Konstante $1/L$ mehr. Sie hat die Größenordnung $1/(L_0 + L_1\|\mathbf{g}\|)$. Bei kleinem Gradienten ist die Schrittweite fest, bei großem Gradienten die zurückgelegte *Strecke* $\eta\,\|\mathbf{g}\|$ von der Größenordnung $1/L_1$. Gradienten-Clipping ersetzt $\mathbf{g}$ durch $\min(1, c/\|\mathbf{g}\|)\,\mathbf{g}$ mit einem Schwellenwert $c$ und erzielt ohne zusätzliche Kosten genau solche Schritte. Für unsere Funktion vierten Grades mit $c = 1$ verschiebt die zuvor divergente Schrittweite $\eta = 0.3$ die Iterierte $x$ nun pro Iteration um $0.3$, bis $|x| \le 1$ gilt. Von dort aus konvergiert das Verfahren (Aufgabe :exref:`ex_mdl-gradient-based-optimization-relaxed-smoothness`).

<!--en:544a7560-->
## Das quadratische Modell und die Konditionszahl

<!--en:59dcafc2-->
Zu Beginn haben wir gefragt, warum wir für die eine Mulde $69$ Schritte brauchen und für die andere $6{,}908$. Beide Zielfunktionen sind quadratisch. Für quadratische Funktionen lässt sich der Gradientenabstieg in geschlossener Form berechnen. Erstaunlicherweise hängt die Iterationszahl nur von einer Größe ab, dem Verhältnis der größten zur kleinsten Krümmung.

<!--en:0ecb335e-->
### Schlecht konditionierte Täler

<!--en:17992a6e-->
Sehen wir uns vor der Rechnung :numref:`fig_mdl-opt-gd-bowl-vs-valley` an. Die Abbildung vergleicht den Gradientenabstieg für zwei quadratische Funktionen. In einer runden Mulde laufen die Iterierten fast geradewegs zum Minimum. In einem lang gestreckten Tal sind die Flanken steil, der Talboden ist flach. Jeder Schritt nahe der Stabilitätsgrenze, die die Flanken vorgeben, schießt quer über den Talboden hinaus. Die Iterierten springen von Flanke zu Flanke und kommen entlang des Talbodens kaum voran. So entsteht die klassische Zickzackbahn. (Für das dargestellte Tal wählen wir $\mathrm{diag}(1, 20)$ mit $\eta = 0.097$, knapp unter der Stabilitätsgrenze $0.1$. Daraus ergeben sich die in der Bildunterschrift angegebenen Kontraktionsfaktoren.)

<!--en:4d3bf1db-->
![Gradientenabstieg für zwei quadratische Funktionen $f(\mathbf{x})=\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$. Links laufen die Iterierten in einer gut konditionierten Mulde fast geradewegs zum Minimum. Rechts schießt in einem schlecht konditionierten Tal die steile Eigenkomponente bei einer Schrittweite nahe der Stabilitätsgrenze über das Ziel hinaus und wechselt das Vorzeichen (Faktor $-0.94$). Entlang der langsamen Achse beträgt der Kontraktionsfaktor pro Schritt nur $0.90$.](../img/mdl-opt-gd-bowl-vs-valley.svg)

<!--en:5aed6a80-->
Quadratische Funktionen verdienen eine eingehende Analyse, denn jede glatte Funktion sieht in der Nähe eines strikten Minimums näherungsweise quadratisch aus. Um einen Minimierer $\mathbf{x}^\star$ mit positiv definiter Hesse-Matrix $\mathbf{A}$ nähert die Taylor-Entwicklung zweiter Ordnung $f$ durch $f(\mathbf{x}^\star) + \tfrac12 (\mathbf{x}-\mathbf{x}^\star)^\top \mathbf{A}\, (\mathbf{x}-\mathbf{x}^\star)$ an. Verschieben wir den Minimierer in den Ursprung und lassen die additive Konstante weg, erhalten wir unser Modellproblem:

<!--en:41e611fb-->
Der Gradient ist $\mathbf{A}\mathbf{x}$ und die Hesse-Matrix ist $\mathbf{A}$. Die Glattheitskonstante beträgt daher einfach $L = \lambda_{\max}$. Das Verhältnis der größten zur kleinsten Krümmung

<!--en:ac051a53-->
heißt **Konditionszahl** (engl. condition number) des Problems. Sie beträgt $10$ für unsere erste Mulde $\mathrm{diag}(1, 10)$ und $10^3$ für das wiederkehrende Beispiel des Kapitels. Da $\mathbf{A}$ symmetrisch und positiv definit ist, stimmt $\kappa$ mit dem Verhältnis der Singulärwerte $\kappa(\mathbf{A})$ aus :numref:`subsec_mdl-condition-number` überein.

<!--en:b2a1dc71-->
### Kontraktion der Eigenkomponenten und Stabilitätsgrenze

<!--en:2a04ace3-->
Der Zickzackverlauf in :numref:`fig_mdl-opt-gd-bowl-vs-valley` folgt unmittelbar aus der Schrittweitengrenze, die die steilste Richtung erzwingt. Um die Grenze genau zu bestimmen und zu sehen, wie stark sie uns in der flachen Richtung bremst, berechnen wir den Gradientenabstieg für eine quadratische Funktion exakt. Wie gewohnt beginnen wir mit einer Variablen.

<!--en:d3250905-->
Betrachten wir $f(u) = \tfrac{\lambda}{2} u^2$ mit $\lambda > 0$. Ein Gradientenschritt führt von $u_k$ nach $u_{k+1} = u_k - \eta\lambda u_k$. Also gilt

<!--en:82e5f3fd-->
Jede Iteration multipliziert den Fehler mit demselben skalaren Faktor $1 - \eta\lambda$. Konvergenz liegt genau dann vor, wenn $|1 - \eta\lambda| < 1$ gilt, also $0 < \eta < 2/\lambda$. Bei der optimalen Wahl $\eta = 1/\lambda$ erreicht die Iterierte in einem Schritt den Punkt $0$. Für $1/\lambda < \eta < 2/\lambda$ ist der Faktor negativ. Daher wechselt $u_k$ bei jedem Schritt das Vorzeichen und nähert sich dabei null. Für $\lambda = 10$ beispielsweise ergibt $\eta = 0.1$ den Faktor $0$, bei $\eta = 0.19$ erhalten wir $-0.9$ und bei $\eta = 0.21$ den Faktor $-1.1$, mit dem die Iteration divergiert.

<!--en:99c3e351-->
Die $d$-dimensionale quadratische Funktion hat genau dieselbe Struktur. Wegen $\nabla f(\mathbf{x}) = \mathbf{A}\mathbf{x}$ ergibt die Aktualisierung :eqref:`eq_mdl-opt-gd-step`

<!--en:3b20366c-->
Jeder Schritt wendet denselben linearen Operator an. Nach $k$ Schritten gilt also $\mathbf{x}_k = (\mathbf{I} - \eta\mathbf{A})^k\mathbf{x}_0$. Da $\mathbf{A}$ symmetrisch ist, besitzt die Matrix eine Orthonormalbasis aus Eigenvektoren (:numref:`sec_mdl-eigendecompositions`). Stellen wir $\mathbf{x}_k$ in dieser Eigenbasis mit den Koordinaten $u_i^{(k)}$ dar. Die Matrix $\mathbf{I} - \eta\mathbf{A}$ hat dieselben Eigenvektoren mit den Eigenwerten $1 - \eta\lambda_i$. Damit entkoppeln sich die Koordinaten vollständig und entwickeln sich unabhängig voneinander:

<!--en:fde67aa1-->
Der Gradientenabstieg für eine $d$-dimensionale quadratische Funktion besteht also aus $d$ unabhängigen Kopien der Rekursion für eine Variable, je einer pro Eigenwert bzw. **Eigenkomponente** (engl. mode). Aus den Faktoren $1 - \eta\lambda_i$ folgen unmittelbar zwei Aussagen:

<!--en:e66e3b1a-->
* **Konvergenz** erfordert $|1 - \eta \lambda_i| < 1$ für jede Eigenkomponente. Das vereinfacht sich zu $0 < \eta < 2/\lambda_{\max} = 2/L$. Die obere Grenze heißt **Stabilitätsgrenze** (engl. stability ceiling). Für quadratische Funktionen ist die nach dem Abstiegslemma hinreichende Schranke $2/L$ auch notwendig. Erhöhen wir $\eta$ über diese Grenze hinaus, fällt der Faktor der steifsten Komponente unter $-1$. Diese Koordinate schwingt mit wachsender Amplitude, selbst wenn alle anderen Komponenten abklingen.

<!--en:83bb6276-->
* **Geschwindigkeit.** Die langsamste Eigenkomponente gibt das Tempo vor. Die Fehlernorm fällt wie $\rho(\eta)^k$. Dabei ist $\rho(\eta) = \max_i |1 - \eta\lambda_i|$ der **Spektralradius** von $\mathbf{I} - \eta\mathbf{A}$, also der größte Betrag eines Eigenwerts.

<!--en:368941cf-->
Für $\mathbf{A} = \mathrm{diag}(1, 10)$ liegt die Stabilitätsgrenze bei $\eta = 0.2$. Welche Schrittweite hinterlässt nach $60$ Schritten vom Startpunkt $(1, 1)$ den kleinsten Fehler? Geben Sie vor der folgenden Versuchsreihe eine Vorhersage ab. Ist es die vorsichtige Wahl $0.02$, die nach dem Abstiegslemma günstigste Wahl $0.1 = 1/L$ oder eine Schrittweite näher an der Grenze?

<!--en:253a9101-->
Die vorsichtige Wahl $\eta = 0.02$ ist sicher, aber langsam ($\rho = 0.98$, Restfehler $0.3$ nach $60$ Schritten). Bei $\eta = 0.1$ verschwindet die steife Komponente in einem Schritt. Die langsame Komponente begrenzt die Gesamtrate jedoch weiterhin auf $\rho = 0.9$. Interessanterweise ergibt $\eta = 0.19$ dieselbe Rate $\rho = 0.9$, diesmal aus dem entgegengesetzten Grund. Jetzt schwingt die steife Komponente mit dem Faktor $-0.9$. Dazwischen liegt $\eta = 2/11 \approx 0.182$. Diese Schrittweite gleicht beide Extreme aus und erzielt einen um zwei Größenordnungen kleineren Fehler als jede andere untersuchte Wahl ($\rho = 0.818$, Fehler $8.3 \times 10^{-6}$). Genau an der Grenze $\eta = 0.2$ hat die steife Komponente den Faktor $-1$. Sie wechselt unbegrenzt zwischen $\pm 1$, sodass $\|\mathbf{x}_{60}\| = 1.0$ gilt. Gehen wir knapp über die Grenze auf $\eta = 0.21$, wächst der Fehler in sechzig Iterationen auf $3 \times 10^{2}$.

<!--en:6d5bee82-->
### Die optimale Schrittweite

<!--en:4ae17cd2-->
Mit welcher festen Schrittweite $\eta$ im stabilen Bereich konvergiert der Gradientenabstieg am schnellsten? Die Versuchsreihe legt die Antwort nahe. Wir müssen die beiden äußersten Eigenkomponenten ins Gleichgewicht bringen. Für festes $\eta$ ist der Graph der Abbildung $\lambda \mapsto |1 - \eta\lambda|$ V-förmig mit Scheitel bei $\lambda = 1/\eta$. Auf dem Intervall $[\lambda_{\min}, \lambda_{\max}]$ nimmt eine V-förmige Funktion ihr Maximum an einem der Randpunkte an. Also gilt $\rho(\eta) = \max\big(|1 - \eta\lambda_{\min}|,\, |1 - \eta\lambda_{\max}|\big)$. Sobald $\eta$ den Wert $1/\lambda_{\max}$ übersteigt, beschleunigt eine weitere Erhöhung die langsame Komponente bei $\lambda_{\min}$, bremst aber die steife bei $\lambda_{\max}$.

<!--en:4f4ad9d4-->
**Satz (optimale feste Schrittweite für eine quadratische Funktion).** *Für $f(\mathbf{x}) = \tfrac12 \mathbf{x}^\top \mathbf{A} \mathbf{x}$ nimmt der Konvergenzfaktor $\rho(\eta) = \max_i |1 - \eta\lambda_i|$ sein Minimum an bei*

<!--en:865c33dc-->
**Beweis.** Für jedes $\eta$ gilt $\rho(\eta) \ge \max(1 - \eta\lambda_{\min},\, \eta\lambda_{\max} - 1)$. Das Maximum zweier Zahlen ist mindestens so groß wie jede ihrer Konvexkombinationen. Wählen wir die Gewichte $\lambda_{\max}/(\lambda_{\min}+\lambda_{\max})$ und $\lambda_{\min}/(\lambda_{\min}+\lambda_{\max})$, heben sich die Terme mit $\eta$ auf:

<!--en:2b9d00d6-->
Gleichheit gilt nur, wenn die beiden Terme übereinstimmen, also $1 - \eta\lambda_{\min} = \eta\lambda_{\max} - 1$. Auflösen ergibt $\eta = \eta^\star$. Bei $\eta^\star$ haben beide Randpunkte den Wert $(\kappa - 1)/(\kappa + 1)$. Wegen der V-Form liegen die Werte an allen inneren Eigenwerten darunter. Damit ist die Schranke erreicht. $\blacksquare$

<!--en:03a7eea2-->
Die Abbildung :numref:`fig_mdl-opt-eta-tent` veranschaulicht diesen Beweis für $\mathrm{diag}(1, 10)$. Beim optimalen V liegen die beiden Endpunkte auf derselben Höhe. Hier gilt $\eta^\star = 2/11$ und $\rho = 9/11 = 0.818$. Für sechs Stellen Genauigkeit brauchen wir daher $\log 10^6/\log(11/9) \approx 69$ Schritte. Beim wiederkehrenden Beispiel des Kapitels ($\kappa = 1000$) gilt $\rho = 999/1001$. Dieselben sechs Stellen erfordern dort $6{,}908$ Schritte.

<!--en:6c436804-->
![Die Kontraktionsfaktoren $|1-\eta\lambda|$ der Eigenkomponenten bilden ein V mit Scheitel bei $\lambda = 1/\eta$. Der Konvergenzfaktor $\rho(\eta)$ ist der größere der beiden Werte an den äußersten Eigenwerten $\lambda_{\min} = 1$ und $\lambda_{\max} = 10$. Die Wahl $\eta = 0.1$ beseitigt die steife Komponente sofort. Die langsame schrumpft jedoch mit dem Faktor $0.9$. Die optimale Schrittweite $\eta^\star = 2/(\lambda_{\min} + \lambda_{\max})$ gleicht beide Randwerte bei $(\kappa-1)/(\kappa+1) = 9/11$ an. Sinkt ein Randwert, steigt der andere.](../img/mdl-opt-eta-tent.svg)

<!--en:65536caa-->
Für $\kappa = 1$ (eine vollkommen runde Mulde) gilt $\rho = 0$, und ein einziger Schritt erreicht das Minimum. Für große $\kappa$ gilt $(\kappa-1)/(\kappa+1) \approx 1 - 2/\kappa$ und damit $\log(1/\rho) \approx 2/\kappa$. Um den Fehler um den Faktor $\epsilon$ zu verkleinern, brauchen wir ungefähr $\tfrac{\kappa}{2}\log\tfrac1\epsilon$ Iterationen. **Die Iterationszahl des Gradientenabstiegs wächst linear mit der Konditionszahl.** Der Grund lässt sich kurz zusammenfassen. Damit der Schritt stabil bleibt, muss er sich nach der steifsten Komponente richten, $\eta \lesssim 2/\lambda_{\max}$. In der flachsten Richtung kommen wir daher pro Iteration nur um $\eta\,\lambda_{\min} \approx 2/\kappa$ voran.

<!--en:31255647-->
Bei $\eta^\star$ sagt der Faktor $\rho$ sogar mehr aus als eine asymptotische Rate. Für $\mathrm{diag}(1, 10)$ lauten die Faktoren der beiden Komponenten $+9/11$ und $-9/11$ (Zeile für $\eta = 2/11$ in der Versuchsreihe). Die langsame Komponente schrumpft monoton, die steife wechselt das Vorzeichen. Da beide Koordinaten betragsmäßig im selben Verhältnis schrumpfen, sinkt $\|\mathbf{x}_k\|$ bei jedem einzelnen Schritt um den Faktor $(\kappa-1)/(\kappa+1)$.

<!--en:44b82a2f-->
### Am Rand der Stabilität

<!--en:2d254bb2-->
Die Stabilitätsgrenze legt für neuronale Netze ein einfaches Vorgehen nahe. Wir berechnen die **Schärfe** (engl. sharpness), den größten Eigenwert $\lambda_{\max}(\nabla^2 f)$ der Hesse-Matrix, und setzen $\eta < 2/\lambda_{\max}$. Beim Training eines neuronalen Netzes ändert sich die Hesse-Matrix allerdings entlang der Trajektorie. Beim Gradientenabstieg auf dem gesamten Datensatz wächst die Schärfe typischerweise bis etwa $2/\eta$ und pendelt dann um diese Schwelle. Der Verlust sinkt dabei langfristig, obwohl er bei vielen einzelnen Schritten sprunghaft steigt :cite:`Cohen.Kaur.Li.ea.2021`. In :numref:`subsec_gd-edge-of-stability` führen wir diesen Versuch an einem kleinen $\tanh$-Netz durch und untersuchen ihn im Detail. Hier fragen wir genauer, welchen Teil dieses Verhaltens unser quadratisches Modell erklärt.

<!--en:6f4c89ba-->
Etwa die Hälfte. In der Eigenbasis der Hesse-Matrix hat die Komponente zum größten Eigenwert den Faktor $1 - \eta\lambda_{\max}$. Dieser fällt genau dann unter $-1$, wenn $\lambda_{\max} > 2/\eta$ gilt. Die Koordinate wechselt dann mit wachsender Amplitude das Vorzeichen. Das erklärt, warum ein langfristig sinkender Verlust bei einzelnen Schritten in die Höhe schnellen kann. Bei einer festen quadratischen Funktion würde die Amplitude jedoch unbegrenzt wachsen. Was die Schärfe wieder senkt, liegt außerhalb der quadratischen Näherung. Der dort vernachlässigte Taylor-Term dritter Ordnung bewirkt, dass die Schwingung die Iterierten allmählich in flachere Gebiete geringerer Schärfe führt :cite:`Damian.Nichani.Lee.2023`. Aufgabe :exref:`ex_mdl-gradient-based-optimization-edge-of-stability-momentum` zeigt, dass Momentum die Stabilitätsschwelle auf $(2 + 2\beta)/\eta$ verschiebt.

<!--en:8dcf4f5e-->
### Von quadratischen zu konvexen Funktionen

<!--en:1e769f29-->
Die lineare Abhängigkeit von $\kappa$ bleibt auch über quadratische Funktionen hinaus bestehen. Eine differenzierbare Funktion ist konvex, wenn sie auf oder über jeder ihrer Tangentialebenen liegt. Sie ist $\mu$-stark konvex, wenn sie diese Ebenen um mindestens einen quadratischen Term überragt, also $f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \tfrac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$ für alle $\mathbf{x}, \mathbf{y}$ gilt. Beide Begriffe entwickeln wir in :numref:`sec_mdl-convexity`. Ist $f$ sowohl $L$-glatt als auch $\mu$-stark konvex, so liegt die Funktion an jedem Punkt zwischen zwei quadratischen Funktionen mit den Krümmungen $\mu$ und $L$. Das Verhältnis $\kappa = L/\mu$ tritt an die Stelle von $\lambda_{\max}/\lambda_{\min}$. Für den Gradientenabstieg mit $\eta = 1/L$ gilt dann

<!--en:733f0037-->
Damit brauchen wir wiederum $O(\kappa\log\tfrac1\epsilon)$ Iterationen. Ist $f$ glatt und konvex, aber nicht stark konvex, konvergiert der Gradientenabstieg weiterhin zum globalen Optimum, allerdings mit der langsameren sublinearen Rate $O(1/k)$. Ohne Konvexität bleibt nur die Stationaritätsschranke :eqref:`eq_mdl-opt-stationarity-rate`. Die auf Konvexität gestützten Beweise führen wir in :numref:`subsec_mdl-convergence-rates`. Dort vergleichen wir diese allgemeinen Schranken auch mit der schärferen Rate für quadratische Funktionen.

<!--en:49d722bd-->
## Momentum und Beschleunigung

<!--en:a30d1252-->
Der Gradientenabstieg braucht $O(\kappa)$ Iterationen, weil dieselbe Schrittweite sowohl zur steifsten als auch zur flachsten Eigenkomponente passen muss. Auch mit Momentum werten wir pro Schritt nur einen Gradienten aus. Die geänderte Aktualisierungsregel erlaubt es jedoch beiden Komponenten, mit derselben Rate zu schrumpfen.

<!--en:3d92a1fc-->
### Heavy-Ball-Momentum

<!--en:5b5bdcfc-->
Die Geometrie des Tals legt nahe, was wir ändern können. An den steilen Flanken zeigen aufeinanderfolgende Gradienten in entgegengesetzte Richtungen (der Zickzackverlauf in :numref:`fig_mdl-opt-gd-bowl-vs-valley`). Mitteln wir die jüngsten Gradienten, heben sich die Schwingungen auf. Entlang des flachen Talbodens sind aufeinanderfolgende Gradienten klein, zeigen aber stets in dieselbe Richtung. Addieren wir sie, gewinnen wir an Geschwindigkeit. Beides erreichen wir, indem wir der Iterierten eine Geschwindigkeit mit Gedächtnis geben. **Heavy-Ball-Momentum** :cite:`Polyak.1964` ersetzt den einfachen Gradientenschritt durch

<!--en:c166b355-->
Dabei ist $\beta \in [0, 1)$ der Momentum-Parameter. Die Geschwindigkeit $\mathbf{v}_k$ ist eine Summe vergangener Gradienten mit exponentiell abklingenden Gewichten. Wegen $\mathbf{v}_k = \mathbf{x}_k - \mathbf{x}_{k-1}$ können wir $\mathbf{v}$ eliminieren und erhalten die äquivalente Zweischrittform

<!--en:a6ffccb9-->
Bei jeder Aktualisierung gehen wir einen Gradientenschritt und addieren das $\beta$-Fache der letzten Verschiebung. Setzen wir $\beta = 0$, erhalten wir den gewöhnlichen Gradientenabstieg.

<!--en:72aeae36-->
Für eine quadratische Funktion entkoppeln sich die Eigenkomponenten erneut. Für eine einzelne Komponente mit Gradient $\lambda u_k$ lautet die Zweischrittform $u_{k+1} = (1 + \beta - \eta\lambda)\, u_k - \beta\, u_{k-1}$. Jede Iterierte hängt also von ihren beiden Vorgängerinnen ab. Eine solche Rekurrenz zweiter Ordnung ist das diskrete Gegenstück zu einem gedämpften harmonischen Oszillator. Anschaulich rollt eine schwere Kugel unter Reibung in einer Mulde $1 - \beta$. Dabei bestimmt $\beta$ die Dämpfung.

<!--en:3e27f3a6-->
In :numref:`fig_mdl-opt-momentum-damping` sehen wir die drei möglichen Verläufe für $\mathrm{diag}(1, 10)$, alle mit der für den Gradientenabstieg optimalen Schrittweite $\eta = 2/11$. Bei zu wenig Momentum verläuft die Bahn ähnlich zickzackförmig wie beim gewöhnlichen Gradientenabstieg. Bei richtig gewähltem Momentum schwingt die Kugel einmal über und kommt dann gleichmäßig zur Ruhe. Bei zu viel Momentum trägt die Trägheit sie spiralförmig um das Minimum. Bestimmen wir nun die genauen Konvergenzraten in allen drei Fällen. Dabei zeigt sich, dass die schnellstmögliche Konvergenz auch eine größere Schrittweite erfordert.

<!--en:551c4325-->
![Heavy-Ball-Momentum im Tal mit $\kappa=10$ und Zielfunktion $\tfrac12(x^2+10y^2)$. Alle drei Läufe verwenden die Schrittweite $\eta=2/11$ des Gradientenabstiegs. Bei zu wenig Momentum entsteht ein ähnlicher Zickzackverlauf wie beim Gradientenabstieg. Der Koeffizient $1+\beta-\eta\lambda_{\max}$ der steifen Eigenkomponente ist negativ und ergibt eine negative reelle Wurzel. Am schnellsten ist bei dieser Schrittweite die Wahl von $\beta$ für den aperiodischen Grenzfall. Bei zu viel Momentum verläuft die Bahn spiralförmig um das Minimum. Die optimale $\sqrt\kappa$-Rate von $0.52$ erfordert die größere Schrittweite $\eta_{\mathrm{HB}}$ aus :eqref:`eq_mdl-opt-hb-tuning`, bei der das Verfahren mit $\beta=0.05$ divergieren würde.](../img/mdl-opt-momentum-damping.svg)

<!--en:05b2d0a1-->
### Das $\sqrt{\kappa}$-Gesetz

<!--en:dbe43576-->
Mit dem passenden Parameterpaar $(\eta, \beta)$ schrumpfen, wie wir nun zeigen, alle Eigenkomponenten mit derselben Rate $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$. Für $\kappa = 10$ lautet das optimale Paar $\beta^\star \approx 0.27$ und $\eta_{\mathrm{HB}} \approx 0.23$. Damit sinkt der Kontraktionsfaktor von $0.82$ beim Gradientenabstieg auf $0.52$. Sechs Stellen Genauigkeit erreichen wir in $27$ statt in $69$ Schritten. Erstaunlicherweise folgt das gesamte Ergebnis aus einer einzigen algebraischen Eigenschaft einer einzelnen Komponente.

<!--en:9f2c41b3-->
**Satz (Heavy-Ball-Momentum kontrahiert jede Eigenkomponente mit Rate $\sqrt\beta$).** *Sei $0 < \beta < 1$. Für $f(u) = \tfrac{\lambda}{2}u^2$ erfüllen die Heavy-Ball-Iterierten die Gleichung $u_{k+1} = (1 + \beta - \eta\lambda)u_k - \beta u_{k-1}$. Gilt $(1 - \sqrt\beta)^2 \le \eta\lambda \le (1 + \sqrt\beta)^2$, so haben beide Wurzeln des charakteristischen Polynoms den Betrag $\sqrt\beta$, und es gilt $|u_k| \le C\,k\,\beta^{k/2}$ für $k \ge 1$ mit einer Konstanten $C$, die von $u_0$, $u_1$ und $\eta\lambda$ abhängt. Den linearen Faktor $k$ brauchen wir nur an den beiden Randpunkten, an denen eine doppelte Wurzel auftritt.*

<!--en:85b2d681-->
**Beweis.** Setzen wir den Ansatz $u_k = z^k$ in die Rekurrenz ein und dividieren durch $z^{k-1}$, erhalten wir die charakteristische Gleichung

<!--en:07bb9106-->
Das Produkt der beiden Wurzeln ist $\beta$. Sie sind konjugiert komplex, wenn die Diskriminante $(1 + \beta - \eta\lambda)^2 - 4\beta$ negativ ist. Das bedeutet $|1 + \beta - \eta\lambda| < 2\sqrt\beta$ oder, umgeformt, $(1 - \sqrt\beta)^2 < \eta\lambda < (1 + \sqrt\beta)^2$. Für konjugiert komplexe Zahlen $z, \bar z$ gilt $|z|^2 = z\bar z = \beta$. Beide Wurzeln haben daher den Betrag $\sqrt\beta$ unabhängig von $\lambda$, und die allgemeine Lösung $u_k = a z^k + b \bar z^k$ erfüllt $|u_k| \le (|a| + |b|)\beta^{k/2}$. An beiden Randpunkten verschwindet die Diskriminante, und wir erhalten eine doppelte reelle Wurzel $(1 + \beta - \eta\lambda)/2 = \pm\sqrt\beta$. Bei einer doppelten Wurzel $z$ lautet die allgemeine Lösung $u_k = (a + bk)z^k$. Das erklärt den zusätzlichen Faktor $k$. $\blacksquare$

<!--en:035c2353-->
**Korollar (Parameterwahl).** Wir legen die beiden äußersten Eigenkomponenten genau an die Grenzen des Intervalls komplexer Wurzeln, also $\eta\lambda_{\min} = (1 - \sqrt\beta)^2$ und $\eta\lambda_{\max} = (1 + \sqrt\beta)^2$. Alle dazwischenliegenden Eigenwerte fallen damit ins Innere des Intervalls, sodass jede Komponente Wurzeln vom Betrag $\sqrt\beta$ hat. Durch Wurzelziehen und Bilden von Differenz und Summe erhalten wir $\sqrt{\eta\lambda_{\max}} - \sqrt{\eta\lambda_{\min}} = 2\sqrt\beta$ und $\sqrt{\eta\lambda_{\max}} + \sqrt{\eta\lambda_{\min}} = 2$. Lösen wir die zweite Gleichung nach $\eta$ auf und dividieren die erste durch die zweite, ergibt sich $\sqrt\beta = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$. Mit der Parameterwahl

<!--en:0d2ce7d5-->
kontrahiert Heavy-Ball-Momentum jede Eigenkomponente mit der Rate

<!--en:a3abe1f7-->
Vergleichen wir dies mit :eqref:`eq_mdl-opt-optimal-step`. Momentum ersetzt $\kappa$ durch $\sqrt\kappa$. Damit sinkt der Iterationsaufwand von $O(\kappa \log \tfrac1\epsilon)$ auf $O(\sqrt{\kappa} \log \tfrac1\epsilon)$ :cite:`Polyak.1964`. Für endliches $k$ entsteht durch die doppelten Wurzeln der beiden äußersten Komponenten ein geringer logarithmischer Mehraufwand. Der Fehler fällt wie $k\rho^k$. Mit $\log(1/\rho) \approx 2/\sqrt\kappa$ ergibt sich die Schrittzahl für die Genauigkeit $\epsilon$ daher aus $k \approx \tfrac{\sqrt\kappa}{2}\big(\log\tfrac1\epsilon + \log k\big)$. Zum asymptotischen Gesetz kommt ein Korrekturterm der Ordnung $\sqrt\kappa\log\kappa$ hinzu.

<!--en:e297c5dc-->
Die Abbildung :numref:`fig_mdl-opt-hb-eigenvalues` veranschaulicht den Beweis für $\kappa = 10$. Links bleibt der Betrag der dominanten Wurzel im gesamten Intervall komplexer Wurzeln konstant bei $\sqrt\beta$. Rechts liegt das Wurzelpaar jeder Komponente in der komplexen Ebene auf dem Kreis mit Radius $\sqrt{\beta^\star} = 0.52$. Die reellen Faktoren des Gradientenabstiegs reichen hingegen bis $\pm 0.82$. Mit Momentum dürfen die Wurzeln die reelle Achse verlassen. Dadurch können steife und flache Komponenten denselben kleinen Wurzelbetrag haben, während die flache Komponente beim Gradientenabstieg nur mit der Rate $1 - 2/\kappa$ vorankommt.

<!--en:edabf38e-->
Der Satz erklärt auch die genauen Raten der drei Trajektorien in :numref:`fig_mdl-opt-momentum-damping`. Dort legt $\eta = 2/11$ die beiden Komponenten auf $\eta\lambda = 0.18$ und $1.82$. Bei $\beta = 0.85$ liegen beide im Inneren des Intervalls. Die Kontraktionsrate ist $\sqrt{0.85} = 0.92$, und die komplexen Wurzeln erzeugen das spiralförmige Überschwingen. Das kleinste $\beta$, dessen Intervall die langsame Komponente noch enthält, ist $\beta = (1 - \sqrt{2/11})^2 \approx 0.33$. Damit erreichen wir für diese Schrittweite die schnellste Rate $0.57$. Bei $\beta = 0.05$ liegen beide Komponenten außerhalb des Intervalls $[0.60, 1.50]$. Die langsame hat eine reelle Wurzel $0.81$. Die steife hat eine negative reelle Wurzel, sodass die Komponente bei jedem Schritt das Vorzeichen wechselt. Ein zeitkontinuierlicher physikalischer Oszillator kann einen solchen schrittweisen Vorzeichenwechsel nicht erzeugen.

<!--en:60006109-->
![Heavy-Ball-Momentum mit der Parameterwahl aus :eqref:`eq_mdl-opt-hb-tuning` für das Beispiel mit $\kappa=10$. (a) Bei festem $\beta$ bleibt der größere der beiden Wurzelbeträge konstant bei $\sqrt\beta$, solange die Wurzeln komplex sind. Die optimale Parameterwahl legt die beiden äußersten Eigenkomponenten an die Intervallgrenzen. (b) In der komplexen Ebene liegt das Wurzelpaar jeder Komponente (mit ihrem Eigenwert $\lambda$ beschriftet) auf dem Kreis mit Radius $\sqrt{\beta^\star}=0.52$. Die Faktoren des Gradientenabstiegs erstrecken sich auf der reellen Achse über $[-0.82,0.82]$. Die doppelten Wurzeln der äußersten Komponenten führen zum Faktor $k$ im Fehler.](../img/mdl-opt-hb-eigenvalues.svg)

<!--en:3134804e-->
Dieser Satz über $\sqrt\kappa$ gilt nur für quadratische Funktionen. :citet:`Lessard.Recht.Packard.2016` konstruierten eine eindimensionale, glatte, stark konvexe Funktion, für die Heavy-Ball-Momentum mit der Parameterwahl :eqref:`eq_mdl-opt-hb-tuning` bei bestimmten Initialisierungen nicht konvergiert. Die Iterierten bleiben in einem stabilen Grenzzyklus gefangen (Aufgabe :exref:`ex_mdl-gradient-based-optimization-heavy-ball-limit-cycle`). Der verbreitete Einsatz von Heavy-Ball-Momentum im Deep Learning :cite:`Sutskever.Martens.Dahl.ea.2013` stützt sich also auf die lokale quadratische Näherung und empirische Erfolge. Ein globaler Konvergenzsatz liegt ihm nicht zugrunde.

<!--en:a3c89752-->
### Nesterovs vorausschauender Schritt

<!--en:80323b27-->
Eine kleine Änderung liefert wieder eine globale Garantie. Wir werten den Gradienten statt an der aktuellen Position $\mathbf{x}_k$ an dem vorausberechneten Punkt $\mathbf{x}_k + \beta\,\mathbf{v}_k$ aus, zu dem das Momentum uns führen würde. **Nesterovs beschleunigtes Gradientenverfahren** :cite:`Nesterov.1983` verwendet die Aktualisierung

<!--en:bfc4090c-->
Dieser Wechsel des Auswertungspunkts ist der einzige Unterschied zum Heavy-Ball-Verfahren. Würde das Momentum die Iterierte über das Minimum hinaustragen, wirkt der am vorausberechneten Punkt berechnete Gradientenschritt der Bewegung bereits entgegen. Er bremst die Iterierte, bevor sie überschwingt.

<!--en:87987171-->
**Satz (Nesterovs Verfahren für quadratische Funktionen).** *Sei $\kappa = L/\mu$. Mit $\eta = 1/L$ und $\beta = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ haben die charakteristischen Wurzeln jeder Eigenkomponente einer quadratischen Funktion mit Eigenwerten in $[\mu, L]$ höchstens den Betrag $1 - 1/\sqrt\kappa$. Bei der langsamen Komponente $\lambda = \mu$ tritt eine doppelte Wurzel auf. Folglich ist der Fehler durch $C\,k\,(1 - 1/\sqrt\kappa)^k$ beschränkt.*

<!--en:2e92f57f-->
**Beweis.** Für eine Eigenkomponente mit Eigenwert $\lambda$ schreiben wir $a = 1 - \eta\lambda \in [0, 1 - 1/\kappa]$. Für die Geschwindigkeit ergibt sich die Aktualisierung $v_{k+1} = a\beta v_k - \eta\lambda u_k$, also $u_{k+1} = a u_k + a\beta v_k$. Setzen wir $v_k = u_k - u_{k-1}$ ein, erhalten wir die Rekurrenz $u_{k+1} = (1 + \beta)a\,u_k - \beta a\,u_{k-1}$. Ihr charakteristisches Polynom $z^2 - (1 + \beta)a\,z + \beta a$ hat die Diskriminante $a\,[(1 + \beta)^2 a - 4\beta]$. Für unsere Wahl von $\beta$ gilt $(1 + \beta)^2 = 4\kappa/(\sqrt\kappa + 1)^2$. Bei der langsamen Komponente $a = (\kappa - 1)/\kappa$ hat der Klammerausdruck den Wert $4(\kappa - 1)/(\sqrt\kappa + 1)^2 - 4\beta = 4\beta - 4\beta = 0$. Damit liegt eine doppelte Wurzel bei $(1 + \beta)a/2 = (\sqrt\kappa - 1)/\sqrt\kappa = 1 - 1/\sqrt\kappa$. Jede steifere Komponente hat ein kleineres $a$. Der Klammerausdruck ist dann strikt negativ, und die Wurzeln sind komplex (oder liegen bei $0$, wenn $a = 0$ gilt). Ihr Betrag erfüllt $\sqrt{\beta a} < \sqrt{\beta(1 - 1/\kappa)} = 1 - 1/\sqrt\kappa$. $\blacksquare$

<!--en:82d7778b-->
Für quadratische Funktionen beträgt Nesterovs Kontraktionsfaktor $0.684$ bei $\kappa = 10$, $0.900$ bei $\kappa = 100$ und $0.968$ bei $\kappa = 1000$. Damit ist das Verfahren etwas langsamer als Heavy-Ball-Momentum mit $0.519$, $0.818$ und $0.939$, behält aber dieselbe Abhängigkeit von $\sqrt\kappa$ bei. Anders als beim Heavy-Ball-Verfahren gelten für Nesterovs Verfahren auch jenseits quadratischer Funktionen Garantien. Wir geben sie hier ohne Beweis an :cite:`Nesterov.2018`. Für jede $\mu$-stark konvexe, $L$-glatte Funktion erfüllt Nesterovs Aktualisierung mit demselben konstanten $\beta$ die Schranke $f(\mathbf{x}_k) - f^\star \le C\,(1 - 1/\sqrt\kappa)^k$. (Beachten Sie die betrachtete Größe. Die Schranke gilt für die Differenz der Funktionswerte $f - f^\star$, die wie das Quadrat des Abstands skaliert. Ihr Exponent ist also halb so groß wie der entsprechende Exponent für die Funktionswerte im quadratischen Fall.) Für glatte konvexe Funktionen ohne starke Konvexität ergibt ein Parameterplan $\beta_k \to 1$ die Rate $f(\mathbf{x}_k) - f^\star = O(1/k^2)$ und verbessert damit die Rate $O(1/k)$ des Gradientenabstiegs. Im Allgemeinen lässt sich diese Geschwindigkeit nicht übertreffen. Kein Verfahren, das seine Iterierten allein aus bisherigen Gradienten aufbaut, kann im ungünstigsten Fall schneller als $O(1/k^2)$ konvergieren oder die Abhängigkeit von $\sqrt\kappa$ verbessern, sofern die Dimension im Verhältnis zur Iterationszahl groß ist :cite:`Nemirovski.Yudin.1983`. In Aufgabe :exref:`ex_mdl-gradient-based-optimization-first-order-lower-bound` konstruieren Sie die Funktion, die diese Grenze im ungünstigsten Fall belegt.

<!--en:b3f46b77-->
Überprüfen wir nun die zu Beginn des Abschnitts genannten Iterationszahlen. Für $\mathrm{diag}(1, \kappa)$ mit Startpunkt $(1, 1)$ sollte der Gradientenabstieg rund $\tfrac{\kappa}{2}\log 10^6 \approx 6.9\,\kappa$ Schritte für sechs Stellen Genauigkeit brauchen und das Heavy-Ball-Verfahren etwas mehr als $6.9\sqrt\kappa$.

<!--en:e88f2e82-->
Die Tabelle bestätigt alle Vorhersagen. Beim Gradientenabstieg bleibt die Schrittzahl geteilt durch $\kappa$ konstant bei $6.9$. Das ergibt $6{,}908$ Iterationen bei $\kappa = 1000$, während das Heavy-Ball-Verfahren nur $315$ braucht. Beim Heavy-Ball-Verfahren steigt das Verhältnis $k/\sqrt\kappa$ leicht an ($8.5$, $9.3$, $10.0$), weil bei beiden äußersten Komponenten eine doppelte Wurzel auftritt. Ihre $2\times 2$-Iterationsmatrix hat die Form eines Jordan-Blocks. Zum mehrfachen Eigenwert gehört nur ein linear unabhängiger Eigenvektor (:numref:`subsec_mdl-jordan`), und die $k$-te Potenz bringt den Faktor $k$ mit sich. Die verfeinerte Vorhersage $k \approx \tfrac{\sqrt\kappa}{2}(\log 10^6 + \log k)$ stimmt gut mit den Messwerten überein ($8.6$, $9.2$, $9.8$). Bei $\kappa = 10^4$ benötigt das Heavy-Ball-Verfahren $1{,}056$ Schritte, was einem Verhältnis von $10.6$ entspricht (Aufgabe :exref:`ex_mdl-gradient-based-optimization-write-heavy-ball-eqref`). Nesterovs $508$ Iterationen bei $\kappa = 1000$ liegen zwischen den beiden anderen Werten, wie sein Kontraktionsfaktor $0.968$ gegenüber $0.939$ beim Heavy-Ball-Verfahren erwarten lässt. Alle drei Verfahren werten pro Schritt nur einen Gradienten aus. Die Beschleunigung durch Momentum kostet also keine zusätzlichen Gradientenauswertungen. Zugleich ist $\sqrt\kappa$ die Grenze dessen, was Gradienten allein leisten können. Nach der unteren Schranke für Verfahren erster Ordnung kann kein Verfahren, das allein auf bisherigen Gradienten beruht, diese Abhängigkeit in hohen Dimensionen verbessern. Um die Abhängigkeit von $\kappa$ ganz zu beseitigen, muss der Schritt die Krümmung ausdrücklich modellieren (:numref:`sec_mdl-curvature-methods`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:f3afa704-->
Die Glattheit begrenzt, wie weit wir einem Gradientenschritt trauen können. Für eine $L$-glatte Zielfunktion ist der Abstieg für positive Schrittweiten unterhalb der Grenze $\eta = 2/L$ garantiert. Bei einer quadratischen Funktion schrumpft der Fehler mit der besten festen Schrittweite in jeder Iteration um den Faktor $(\kappa - 1)/(\kappa + 1)$. Die Schrittzahl wächst daher linear mit der Konditionszahl $\kappa$. Die steifste Eigenkomponente bestimmt die Stabilität, die flachste den Fortschritt. Mit Momentum schrumpfen alle Komponenten mit derselben Rate $\sqrt\beta$. Bei optimaler Parameterwahl senkt der Faktor $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ die Iterationszahl von $O(\kappa)$ auf $O(\sqrt\kappa)$, ohne zusätzliche Gradientenauswertungen zu erfordern. Bei neuronalen Netzen steigt die Schärfe bis zur Stabilitätsgrenze $2/\eta$ der gewählten Schrittweite. Das quadratische Modell erklärt dieses Phänomen nur zum Teil. Drei Fragen bleiben offen. Was kann ein Optimierer erreichen, wenn er die Krümmung direkt modelliert (:numref:`sec_mdl-curvature-methods`)? Wann ist ein stationärer Punkt garantiert ein globales Minimum (:numref:`sec_mdl-convexity`)? Und was ändert sich bei verrauschten Gradienten (:numref:`sec_mdl-stochastic-gradient-methods`)?

<!--en:2294243a-->
## Übungen

<!--en:460959d6-->
1. **Vorkonditionierte Abstiegsrichtungen.** Im Satz aus :numref:`subsec_mdl-descent-directions` messen wir die Länge eines Richtungsvektors mit der euklidischen Norm. Eine andere Norm ergibt eine andere Richtung des steilsten Abstiegs.
    1. Zeigen Sie, dass $\mathbf{d} = -\mathbf{P}^{-1}\nabla f(\mathbf{x})$ für jede positiv definite Matrix $\mathbf{P}$ eine Abstiegsrichtung ist, sofern der Gradient nicht verschwindet.
    1. Messen Sie die Länge durch $\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P}\, \mathbf{d}}$. Zeigen Sie, dass der Minimierer von $\nabla f(\mathbf{x})^\top \mathbf{d}$ unter der Nebenbedingung $\|\mathbf{d}\|_{\mathbf{P}} = 1$ proportional zu $-\mathbf{P}^{-1}\nabla f(\mathbf{x})$ ist. Bestimmen Sie die Matrix $\mathbf{P}$, die die Newton-Richtung $-\big(\nabla^2 f(\mathbf{x})\big)^{-1}\nabla f(\mathbf{x})$ aus :numref:`sec_mdl-curvature-methods` ergibt. Die Variablentransformation $\mathbf{u} = \mathbf{P}^{1/2}\mathbf{d}$ führt das Problem auf den euklidischen Fall zurück.
   :exlabel:`ex_mdl-gradient-based-optimization-show-that-is-de`

<!--en:3a1f2474-->
1. [code] **Glattheitskonstanten.** Die Glattheitskonstante $L$ aus :eqref:`eq_mdl-opt-smoothness` beschränkt die Krümmung von $f$. Für einen logistischen Verlust lässt sie sich direkt aus den Daten berechnen.
    1. Sei $f$ zweimal stetig differenzierbar und $L$-glatt. Zeigen Sie, dass $\|\nabla^2 f(\mathbf{x})\,\mathbf{v}\| \le L\,\|\mathbf{v}\|$ für alle $\mathbf{x}$ und $\mathbf{v}$ gilt. Folgern Sie daraus, dass jeder Eigenwert der Hesse-Matrix in $[-L, L]$ liegt. Damit ist die Rückrichtung der Äquivalenz im Anschluss an :eqref:`eq_mdl-opt-smoothness` bewiesen.
    1. Betrachten Sie

       $$
       f(\mathbf{w}) = \frac1n \sum_{i=1}^n \log\big(1 + \exp(-y_i\, \mathbf{x}_i^\top \mathbf{w})\big) + \frac{\nu}{2}\|\mathbf{w}\|^2
       $$

       mit Labels $y_i = \pm 1$ und $\nu = 0.1$. Das ist die Funktion `loss`, die wir in der Datenzelle von :numref:`subsec_mdl-why-not-newton` verwenden. Dabei ist $\mathbf{x}_i^\top$ die $i$-te Zeile von $\mathbf{X} \in \mathbb{R}^{n \times d}$ ($n = 200$, $d = 5$, mit unabhängigen standardnormalverteilten Einträgen). Zeigen Sie, dass $L = \lambda_{\max}(\mathbf{X}^\top \mathbf{X})/(4n) + \nu$ eine gültige Glattheitskonstante ist und die Hesse-Matrix diese Schranke bei $\mathbf{w} = \mathbf{0}$ erreicht.
    1. Berechnen Sie dieses $L$ für eine $200 \times 5$-Matrix $\mathbf{X}$ mit standardnormalverteilten Einträgen und Labels $y_i \in \{\pm 1\}$, die Sie mit der Wahrscheinlichkeit $P(y_i = 1) = 1/(1 + \exp(-\mathbf{x}_i^\top \mathbf{w}_{\mathrm{true}}))$ für $\mathbf{w}_{\mathrm{true}} = (1, -2, 1.5, 0, 2)$ ziehen. Bei jeder Ziehung ergibt sich ein Wert nahe $0.4$. Erklären Sie dies mithilfe der Näherung $\lambda_{\max}(\mathbf{X}^\top\mathbf{X})/n \approx (1 + \sqrt{d/n})^2$ für standardnormalverteilte Einträge. Zählen Sie, wie viele Schritte die Verfahren mit $\eta = 0.5$ (der Referenzschrittweite) und $\eta = 1/L$ jeweils vom Startpunkt $\mathbf{w} = \mathbf{0}$ bis zum Erreichen von $\|\nabla f(\mathbf{w})\| \le 10^{-10}$ brauchen. Welches ist schneller? Bleibt $\eta = 0.5$ unter $2/L$?
   :exlabel:`ex_mdl-gradient-based-optimization-smoothness-constants`

<!--en:93c3beac-->
1. [code] **Clipping bei einer Funktion vierten Grades.** Betrachten Sie $f(x) = \tfrac14 x^4$ und die Aktualisierung mit Gradienten-Clipping $x \leftarrow x - \eta \min(1, c/|g|)\, g$ für $g = x^3$.
    1. Zeigen Sie, dass $3x^2 \le 4 + |x|^3$ für alle $x$ gilt, mit Gleichheit bei $|x| = 2$. Damit erfüllt $f$ die Bedingung :eqref:`eq_mdl-opt-relaxed-smoothness` mit $L_0 = 4$ und $L_1 = 1$. Zeigen Sie außerdem, dass kein $L_0$ für $L_1 = 0$ genügt.
    1. Führen Sie die Aktualisierung mit Clipping ab $x_0 = 3$ mit $\eta = 0.3$ und $c = 1$ aus. Bei dieser Schrittweite erreichte die Iteration ohne Clipping in sechs Schritten den Betrag $6.5 \times 10^{103}$. Bei wie vielen Schritten greift das Clipping ein, und um wie viel verschiebt jeder dieser Schritte die Iterierte $x$?
    1. Zeigen Sie, dass die Aktualisierung mit Clipping bei jedem Schritt mit von null verschiedener Iterierter den Betrag $|x|$ verringert, für jeden Startpunkt $x_0 \neq 0$, sofern $\eta\, c^{2/3} < 2$ gilt. Behandeln Sie den Fall $|x| > c^{1/3}$, in dem Clipping eingreift, und den Fall ohne Clipping getrennt. Warum kann keine feste Schrittweite ohne Clipping diese Eigenschaft haben?
   :exlabel:`ex_mdl-gradient-based-optimization-relaxed-smoothness`

<!--en:b1602e0b-->
1. **Steilster Abstieg in anderen Normen.** ● In Aufgabe :exref:`ex_mdl-gradient-based-optimization-show-that-is-de`b messen wir die Länge mit einer Norm $\|\cdot\|_{\mathbf{P}}$, die von einem inneren Produkt induziert wird. Normen, die nicht von einem inneren Produkt stammen, ergeben steilste Abstiegsrichtungen ganz anderer Form.
    1. Bestimmen Sie die Minimierer von $\nabla f(\mathbf{x})^\top \mathbf{d}$ unter den Nebenbedingungen $\|\mathbf{d}\|_\infty = 1$ bzw. $\|\mathbf{d}\|_1 = 1$ sowie die zugehörigen minimalen Steigungen. Geben Sie an, wann der jeweilige Minimierer eindeutig ist.
    1. Der Parameter sei eine Matrix, deren Gradient $\mathbf{G}$ die reduzierte Singulärwertzerlegung $\mathbf{G} = \mathbf{U}\, \mathrm{diag}(\sigma_i)\, \mathbf{V}^\top$ besitzt. Die Steigung in Richtung $\mathbf{D}$ sei $\operatorname{tr}(\mathbf{G}^\top \mathbf{D})$. Messen Sie die Länge mit der Spektralnorm $\|\mathbf{D}\|_2$. Zeigen Sie, dass die minimale Steigung unter der Bedingung $\|\mathbf{D}\|_2 \le 1$ gleich $-\sum_i \sigma_i$ ist. Sie wird bei $\mathbf{D} = -\mathbf{U} \mathbf{V}^\top$ erreicht. Dies ist der polare Faktor, den Muon auf seine Momentum-Matrix anwendet (:numref:`sec_muon`). Schreiben Sie als ersten Schritt $\operatorname{tr}(\mathbf{G}^\top \mathbf{D}) = \sum_i \sigma_i\, \mathbf{u}_i^\top \mathbf{D}\, \mathbf{v}_i$. Diese Sicht auf das Verfahren als steilsten Abstieg stammt von :citet:`Bernstein.Newhouse.2024`.
   :exlabel:`ex_mdl-gradient-based-optimization-steepest-other-norms`

<!--en:a6a20331-->
1. [code] **Die doppelte Wurzel und der Faktor $k$ beim Heavy-Ball-Verfahren.** ● Wenden Sie das Heavy-Ball-Verfahren :eqref:`eq_mdl-opt-heavy-ball` auf $f(u) = \tfrac{\lambda}{2} u^2$ an.
    1. Schreiben Sie eine Iteration als lineare Abbildung des Zustands $(u_k, u_{k-1})$. Zeigen Sie, dass das charakteristische Polynom der zugehörigen $2 \times 2$-Matrix durch :eqref:`eq_mdl-opt-hb-char-poly` gegeben ist.
    1. Bei der Parameterwahl :eqref:`eq_mdl-opt-hb-tuning` haben die beiden äußersten Eigenkomponenten die doppelten Eigenwerte $\sqrt{\beta^\star}$ und $-\sqrt{\beta^\star}$. Zeigen Sie, dass die jeweilige Matrix dann nur einen linear unabhängigen Eigenvektor besitzt und dass die $k$-te Potenz des Jordan-Blocks $\left(\begin{smallmatrix}\rho&1\\0&\rho\end{smallmatrix}\right)$ über der Diagonalen den Eintrag $k\rho^{k-1}$ hat.
    1. Verwenden Sie $\|\mathbf{e}_k\| \approx k\rho^k$ und $\log(1/\rho) \approx 2/\sqrt\kappa$. Lösen Sie $k = \tfrac{\sqrt\kappa}{2}\big(\log\tfrac1\epsilon + \log k\big)$ numerisch für $\epsilon = 10^{-6}$ und $\kappa \in \{10, 100, 1000\}$. Vergleichen Sie $k/\sqrt\kappa$ mit den Werten der entsprechenden Spalte der Momentum-Codezelle, also $8.5$, $9.3$ und $10.0$.
    1. Führen Sie das Heavy-Ball-Verfahren aus der Momentum-Codezelle mit $\kappa = 10^4$ aus und vergleichen Sie seine Schrittzahl mit Ihrer Vorhersage. Erklären Sie, warum das Verhältnis $k/\sqrt\kappa$ mit wachsendem $\kappa$ nicht gegen eine Konstante konvergiert.
   :exlabel:`ex_mdl-gradient-based-optimization-write-heavy-ball-eqref`

<!--en:743268cd-->
1. [code] **Das Heavy-Ball-Verfahren jenseits quadratischer Funktionen.** ● :citet:`Lessard.Recht.Packard.2016` definieren eine eindimensionale Funktion über ihren Gradienten:

   $$
   \nabla f(x) = \begin{cases} 25x, & x < 1, \\ x + 24, & 1 \le x < 2, \\ 25x - 24, & x \ge 2. \end{cases}
   $$

    1. Zeigen Sie, dass $f$ sowohl $\mu$-stark konvex mit $\mu = 1$ als auch $L$-glatt mit $L = 25$ ist.
    1. Führen Sie das Heavy-Ball-Verfahren mit der Parameterwahl :eqref:`eq_mdl-opt-hb-tuning` für $300$ Schritte ab $x_0 = 3.3$ mit Anfangsgeschwindigkeit null aus. Starten Sie am selben Punkt Nesterovs Verfahren :eqref:`eq_mdl-opt-nesterov` mit $\eta = 1/L$ und $\beta = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$, wobei $\kappa = L/\mu$ gilt. Tragen Sie für beide Verfahren $x_k$ gegen $k$ auf.
    1. Die Heavy-Ball-Iterierten nähern sich einem Zyklus der Periode drei. Er durchläuft den Bereich $x < 1$ zweimal und den Bereich $x \ge 2$ einmal. Stellen Sie die Heavy-Ball-Aktualisierungen über diese Bereiche als lineares Gleichungssystem für die drei Punkte des Zyklus dar. Lösen Sie es und vergleichen Sie Ihre Lösung mit der Simulation.
    1. Wiederholen Sie den Heavy-Ball-Lauf für ein Gitter von Startpunkten auf $[-6, 6]$. Geben Sie an, welcher Anteil auf dem Zyklus landet. In jedem Teilbereich ist $f$ eine quadratische Funktion mit einer Krümmung in $[\mu, L]$. Warum lässt sich die Analyse der einzelnen Eigenkomponenten aus :eqref:`eq_mdl-opt-hb-rate` nicht auf $f$ anwenden?
   :exlabel:`ex_mdl-gradient-based-optimization-heavy-ball-limit-cycle`

<!--en:f6efc0e8-->
1. [code] **Eine untere Schranke für Verfahren erster Ordnung.** ● In dieser Aufgabe leiten Sie den $O(1/k^2)$-Teil der Schranke von :citet:`Nemirovski.Yudin.1983` aus :numref:`subsec_mdl-momentum-acceleration` für den glatten konvexen Fall her. Folgen Sie dabei der Konstruktion von :citet:`Nesterov.2018`. Als Verfahren erster Ordnung gilt hier jeder Algorithmus, dessen Iterierte die Bedingung $\mathbf{x}_k \in \mathbf{x}_0 + \mathrm{span}\{\nabla f(\mathbf{x}_0), \ldots, \nabla f(\mathbf{x}_{k-1})\}$ erfüllen. Setzen Sie $\mathbf{x}_0 = \mathbf{0}$. Für $p \le d$ sei $\mathbf{A}_p \in \mathbb{R}^{d \times d}$ eine Matrix, deren Einträge bis auf den linken oberen $p \times p$-Block null sind. Dieser Block ist tridiagonal mit $2$ auf der Diagonalen und $-1$ auf beiden Nebendiagonalen. Definieren Sie

   $$
   f_p(\mathbf{x}) = \frac{L}{4} \left( \tfrac12\, \mathbf{x}^\top \mathbf{A}_p\, \mathbf{x} - x_1 \right).
   $$

    1. Zeigen Sie, dass $f_p$ konvex und $L$-glatt ist.
    1. Zeigen Sie, dass für einen Vektor $\mathbf{x}$, dessen Träger in den ersten $j$ Koordinaten liegt, auch der Träger von $\nabla f_p(\mathbf{x})$ in den ersten $j + 1$ Koordinaten liegt. Folgern Sie, dass die $k$-te Iterierte jedes Verfahrens erster Ordnung nur in den ersten $k$ Koordinaten von null verschieden sein kann.
    1. Zeigen Sie, dass der Minimierer von $f_p$ die Einträge $x_i^\star = 1 - i/(p+1)$ für $i \le p$ und sonst nur Nullen hat, dass der Minimalwert $-\frac{L}{8}\big(1 - \frac{1}{p+1}\big)$ beträgt und dass $\|\mathbf{x}^\star\|^2 \le (p+1)/3$ gilt.
    1. Sei $2k + 1 \le d$. Die Funktionen $f_{2k+1}$ und $f_k$ stimmen für alle Vektoren überein, deren Träger in den ersten $k$ Koordinaten liegt. Zeigen Sie damit, dass für jedes Verfahren erster Ordnung, angewandt auf $f_{2k+1}$, mit $\mathbf{x}^\star$ als Minimierer von $f_{2k+1}$ gilt:

       $$
       f_{2k+1}(\mathbf{x}_k) - \min f_{2k+1} \;\ge\; \frac{3L\, \|\mathbf{x}_0 - \mathbf{x}^\star\|^2}{32\,(k+1)^2}.
       $$

       Da die Konstruktion $d \ge 2k + 1$ erfordert, gilt die Schranke für ungefähr die ersten $d/2$ Iterationen. In fester Dimension können lange Läufe asymptotisch besser abschneiden. Bei neuronalen Netzen mit $d$ in Millionenhöhe fällt diese Einschränkung jedoch nicht ins Gewicht.
    1. Führen Sie für $k \in \{5, 10, 25, 50, 100, 200, 400\}$ jeweils $k$ Schritte des Gradientenabstiegs mit $\eta = 1/L$ und von Nesterovs Aktualisierung :eqref:`eq_mdl-opt-nesterov` mit $\eta = 1/L$ und $\beta_j = (j-1)/(j+2)$ im Schritt $j = 1, 2, \ldots$ aus. Verwenden Sie die Funktion $f_{2k+1}$ mit $d = 2k + 1$. Dividieren Sie jeweils die Differenz zum optimalen Funktionswert durch die untere Schranke. Welches Verhältnis bleibt bei wachsendem $k$ beschränkt?
   :exlabel:`ex_mdl-gradient-based-optimization-first-order-lower-bound`

<!--en:c7a7842b-->
1. [code] **Am Rand der Stabilität mit Momentum.** ●● :citet:`Cohen.Kaur.Li.ea.2021` berichten, dass sich die Schärfe mit Momentum an einer anderen Schwelle als $2/\eta$ einpendelt.
    1. Zeigen Sie mit der Iterationsmatrix aus Aufgabe :exref:`ex_mdl-gradient-based-optimization-write-heavy-ball-eqref`a, dass das Heavy-Ball-Verfahren :eqref:`eq_mdl-opt-heavy-ball` für $f(x) = \tfrac{\lambda}{2} x^2$ mit $0 \le \beta < 1$ genau dann konvergiert, wenn $0 < \eta\lambda < 2 + 2\beta$ gilt. Leiten Sie das entsprechende Stabilitätsintervall für Nesterovs Aktualisierung :eqref:`eq_mdl-opt-nesterov` her.
    1. Ändern Sie den Versuch zum Rand der Stabilität aus :numref:`subsec_gd-edge-of-stability` so ab, dass Sie mit dem Heavy-Ball-Verfahren und mit Nesterovs Aktualisierung für $\beta \in \{0.25, 0.5\}$ trainieren. Wählen Sie für jeden Lauf eine Schrittweite, deren vorhergesagte Schwelle zwischen $5$ und $12$ liegt. Verfolgen Sie die Schärfe im Vergleich zu $2/\eta$ und zu Ihrer Schwelle aus (a).
    1. An welcher Schwelle pendelt sich die Schärfe ein? Was folgt daraus für die Wahl von $\eta$ anhand der bei der Initialisierung gemessenen Schärfe, wenn Sie mit Momentum trainieren?
   :exlabel:`ex_mdl-gradient-based-optimization-edge-of-stability-momentum`

<!--en:5e70b3c6-->
[Diskussionen](https://d2l.discourse.group/t/36828)

<!--en:5b12aebe-->
Gradientenabstieg und Krümmung<br>**Steilster Abstieg · L-Glattheit · Abstiegslemma · Konditionszahl · Heavy-Ball- und Nesterov-Momentum**

<!--en:8adc0246-->
::: {.slide title="Eine Abstiegsrichtung, drei Schrittlängen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:9d674010-->
Für $f(x, y) = \tfrac12(x^2 + 10y^2)$ mit Startpunkt $\mathbf{x}_0 = (2, 1)$ ($f = 7, \nabla f = (2, 10)$) bestimmt $-\nabla f$ die Richtung und $\eta$ die Schrittlänge:

<!--en:f335525c-->
- **Zu zaghaft ($\eta = 0.01$):** Der Schritt führt sicher, aber sehr langsam zu $(1.98, 0.90)$ mit $f = 6.01$.

<!--en:54bcab67-->
- **Genau richtig ($\eta = 0.10 = 1/L$):** Ein Schritt beseitigt die steile $y$-Komponente ($f = 1.62$). Die flache $x$-Komponente schrumpft pro Schritt nur mit dem Faktor $0.9\times$.

<!--en:d07795f0-->
- **Zu groß ($\eta = 0.25 > 2/L$):** Der Schritt führt über das Ziel hinaus zur steilen Gegenflanke bei $(1.50, -1.50)$ ($f = 12.38$). Das Verfahren divergiert.

<!--en:785823e8-->
::: {.slide title="Taylor-Modell erster Ordnung und Abstiegsrichtungen"}

<!--en:1558a39d-->
[Steilster Abstieg]{.kicker}

<!--en:4ad15dd3-->
Entwicklung von $f$ um $\mathbf{x}$ entlang eines Einheitsvektors mit $\|\mathbf{d}\| = 1$ für kleine $\eta > 0$:

<!--en:b83b23ef-->
- **Halbraum der Abstiegsrichtungen:** Jede Richtung mit $\nabla f(\mathbf{x})^\top \mathbf{d} < 0$ verringert $f$ für hinreichend kleine $\eta > 0$.

<!--en:0a446eeb-->
- **Optimalität nach Cauchy-Schwarz:** Es gilt $\nabla f(\mathbf{x})^\top \mathbf{d} \ge -\|\nabla f(\mathbf{x})\|$. Nur für $\mathbf{d}^\star = -\nabla f(\mathbf{x}) / \|\nabla f(\mathbf{x})\|$ tritt Gleichheit ein.

<!--en:295b0f85-->
::: {.slide title="Richtungsableitungen und endliche Schritte im Vergleich"}

<!--en:1558a39d-2-->
[Steilster Abstieg]{.kicker}

<!--en:d5a70980-->
Für $f(x, y) = \tfrac12(x^2 + 10y^2)$ im Punkt $(2, 1)$ ist der normierte negative Gradient $(-0.196, -0.981)$ mit der Richtungsableitung $-\|\nabla f\| = -10.198$. Bei einem kurzen Schritt mit $\eta = 0.01$ stimmt die tatsächliche Verluständerung mit $\eta \nabla f^\top \mathbf{d}$ bis auf $1\%$ überein:

<!--en:03f82dcf-->
Differenzierbarkeit allein garantiert einen Abstieg nur für *infinitesimal* kleine Schritte. Um eine endliche Schrittweite $\eta > 0$ abzusichern, ist eine Schranke für die Änderung des Gradienten entlang des Schritts nötig.

<!--en:134dbb59-->
::: {.slide title="L-Glattheit und globale quadratische obere Schranke"}

<!--en:b16fe7c7-->
[Glattheit]{.kicker}

<!--en:622308cc-->
Eine differenzierbare Funktion $f$ heißt **$L$-glatt**, wenn ihr Gradient $L$-Lipschitz-stetig ist ($\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \le L\|\mathbf{x} - \mathbf{y}\|$):

<!--en:65229589-->
- **Kriterium für die Hesse-Matrix:** Für $C^2$-Funktionen gilt $L$-Glattheit genau dann, wenn alle Eigenwerte von $\nabla^2 f(\mathbf{x})$ in $[-L, L]$ liegen.

<!--en:fa0b58ec-->
- **Geometrische Garantie:** Der Graph von $f$ überragt seine Tangentialebene höchstens um den Wert der isotropen quadratischen Funktion $\tfrac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$.

<!--en:f63f3e6b-->
::: {.slide title="Abstiegslemma und Schrittweitengrenze 2/L"}

<!--en:b16fe7c7-2-->
[Glattheit]{.kicker}

<!--en:32fa87b0-->
Einsetzen der Aktualisierung des Gradientenabstiegs $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\mathbf{g}_k$ mit $\mathbf{g}_k = \nabla f(\mathbf{x}_k)$ in die quadratische obere Schranke liefert das **Abstiegslemma**:

<!--en:52f4830a-->
- **Lineare Abnahme und Krümmung:** Die Abnahme erster Ordnung $-\eta\|\mathbf{g}_k\|^2$ überwiegt bei kleinem $\eta$. Der größtmögliche Krümmungsbeitrag $+\tfrac{L}{2}\eta^2\|\mathbf{g}_k\|^2$ wächst quadratisch mit $\eta$.

<!--en:d348b669-->
- **Bereich strikter Abnahme:** Der Faktor $\eta(1 - L\eta/2)$ ist genau dann strikt positiv, wenn $0 < \eta < 2/L$ gilt. Bei $\eta = 2/L$ heben sich lineare Abnahme und Krümmungsbeitrag auf.

<!--en:f670969c-->
- **Optimaler Schritt im ungünstigsten Fall:** Die Schrittweite $\eta = 1/L$ minimiert die quadratische obere Schranke und garantiert pro Schritt eine Abnahme um mindestens $f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \tfrac{1}{2L}\|\mathbf{g}_k\|^2$.

<!--en:3cb4d066-->
::: {.slide title="Die quadratische obere Schranke entlang eines Schritts"}

<!--en:b16fe7c7-3-->
[Glattheit]{.kicker}

<!--en:9803d732-->
Für $f(x,y) = \tfrac12(x^2 + 10y^2)$ mit Startpunkt $(2, 1)$ gilt $L = 10$ und $\|\mathbf{g}_0\|^2 = 104$:

<!--en:f4683ca5-->
- Bei $\eta = 1/L = 0.1$ garantiert die Schranke $f \le 1.8$. Der Schritt erreicht $1.62$.

<!--en:91263d23-->
- Jenseits von $2/L = 0.2$ übersteigt die Schranke $f(\mathbf{x}_0) = 7$. Bei $\eta = 0.25$ steigt der tatsächliche Verlust auf $12.375$.

<!--en:d608d0b7-->
::: {.slide title="Stationäre Punkte ohne Konvexität finden"}

<!--en:1f829d82-->
[Garantien ohne Konvexität]{.kicker}

<!--en:1b650644-->
Selbst in einer nichtkonvexen Landschaft kann die Gradientennorm nach dem Abstiegslemma nicht beliebig lange groß bleiben, sofern $f$ durch $f_{\mathrm{low}}$ nach unten beschränkt ist:

<!--en:f6d74da4-->
- **Abnahme als Teleskopsumme:** Für $\eta = 1/L$ das Abstiegslemma schreiben als $\|\mathbf{g}_k\|^2 \le 2L\big(f(\mathbf{x}_k) - f(\mathbf{x}_{k+1})\big)$ und summieren über $k = 0, \ldots, K-1$:
  $$\sum_{k=0}^{K-1} \|\nabla f(\mathbf{x}_k)\|^2 \le 2L\big(f(\mathbf{x}_0) - f(\mathbf{x}_K)\big) \le 2L\big(f(\mathbf{x}_0) - f_{\mathrm{low}}\big)$$

<!--en:0f81be53-->
- **Sublineare Rate für die Stationarität:** Das Minimum über $K$ Schritte ist höchstens so groß wie der Mittelwert, also gilt
  $$\min_{0 \le k < K} \|\nabla f(\mathbf{x}_k)\|^2 \le \frac{2L\big(f(\mathbf{x}_0) - f_{\mathrm{low}}\big)}{K}$$

<!--en:e7121176-->
Unter den ersten $K$ Iterierten hat mindestens eine die Gradientennorm $\|\nabla f(\mathbf{x}_k)\| = O(1/\sqrt{K})$. Dafür genügen $L$-Glattheit und eine untere Schranke für die Zielfunktion.

<!--en:164aba61-->
::: {.slide title="Warum Stationarität kein Minimum garantiert"}

<!--en:4585c242-->
[Schwierigkeiten ohne Konvexität]{.kicker}

<!--en:877ea37d-->
Ohne Konvexität lässt sich anhand von $\nabla f(\mathbf{x}) = \mathbf{0}$ ein Minimum nicht von einem Sattelpunkt unterscheiden:

<!--en:3287f79a-->
- **Am Sattel gefangen:** Die Funktion $f(x, y) = \tfrac12 x^2 + \cos y$ ist durch $-1$ nach unten beschränkt und global $1$-glatt ($\nabla^2 f = \mathrm{diag}(1, -\cos y)$).

<!--en:da125a4d-->
- **Stillstand nach einem Schritt:** Vom Startpunkt $(1, 0)$ mit Schrittweite $\eta = 1/L = 1$ gelangt der Gradientenabstieg direkt zu $(0, 0)$, wo $\nabla f(0, 0) = \mathbf{0}$ gilt.

<!--en:8f60dbb2-->
- **Suboptimales Plateau:** Der Gradientenabstieg bleibt dauerhaft am Sattel $f(0, 0) = +1$ stehen, zwei Einheiten über dem globalen Minimum $f(0, \pi) = -1$.

<!--en:0e41449b-->
- **Ausnahme vom Maß null:** Nur Startpunkte auf der Geraden $y = 0$ enden hier. Bei jeder kleinen Auslenkung $y_0 \neq 0$ entkommt der Gradientenabstieg zu einem Minimum.

<!--en:aa97340c-->
::: {.slide title="Backtracking-Liniensuche und Armijo-Regel"}

<!--en:f4b57e7c-->
[Adaptive Schrittweiten]{.kicker}

<!--en:a65dfbd5-->
Ist $L$ unbekannt oder variiert die lokale Konstante (etwa bei $f(x) = \tfrac14 x^4$), halbiert die **Backtracking-Liniensuche** einen Probeschritt $\eta \leftarrow \eta/2$ so oft, bis die Armijo-Bedingung gilt ($c \in (0, 1)$):

<!--en:510cf596-->
- **Hinreichende Abnahme:** Akzeptiert wird $\eta$, sobald der Verlust um mindestens den Anteil $c$ der linear vorhergesagten Abnahme sinkt.

<!--en:3554e19a-->
- **Endlich viele Versuche:** Sobald $\eta \le 2(1-c)/L_{\mathrm{loc}}$ gilt, gelingt der Test garantiert. Er liefert $\eta \ge \min(\eta_0, (1-c)/L_{\mathrm{loc}})$ ohne vorherige Kenntnis von $L_{\mathrm{loc}}$.

<!--en:5397076d-->
::: {.slide title="Backtracking und feste Schrittweite bei einer Funktion vierten Grades"}

<!--en:f4b57e7c-2-->
[Adaptive Schrittweiten]{.kicker}

<!--en:9d711c29-->
Für $f(x) = \tfrac14 x^4$ mit Startpunkt $x_0 = 3$ und $c = \tfrac12$ verkleinert die Aktualisierung $x \mapsto x(1 - \eta x^2)$ den Betrag $|x|$ genau dann, wenn $\eta x_0^2 < 2$ gilt ($\eta < 2/9 \approx 0.22$). Die lokale Krümmung bei $x_0 = 3$ beträgt hingegen $f''(3) = 27$:

<!--en:86185a52-->
Mit der festen Schrittweite $\eta = 0.3$, die nahe dem Ursprung sicher ist, wächst der Betrag auf $6.5 \times 10^{103}$ in $6$ Schritten ab $x_0 = 3$. Backtracking beginnt bei $\eta = 0.031 < 1/27$ und verdoppelt $\eta$ automatisch, wenn die Iterierte in flachere Bereiche nahe null gelangt.

<!--en:46e40197-->
::: {.slide title="Quadratisches Modell und Konditionszahl"}

<!--en:d5f2d7c0-->
[Geometrie quadratischer Funktionen]{.kicker}

<!--en:2ce7919d-->
Nahe jedem strikten lokalen Minimierer $\mathbf{x}^\star$ mit positiv definiter Hesse-Matrix $\mathbf{A} = \nabla^2 f(\mathbf{x}^\star) \succ \mathbf{0}$ ergibt sich das quadratische Modell in Normalform durch Verschiebung von $\mathbf{x}^\star$ in den Ursprung:

<!--en:c5c1f4de-->
- **Spektrum:** $0 < \lambda_{\min} = \lambda_1 \le \cdots \le \lambda_d = \lambda_{\max} = L$.

<!--en:5d5cf9bd-->
- **Konditionszahl:** $\kappa = \lambda_{\max} / \lambda_{\min} \ge 1$ ist das Verhältnis der größten zur kleinsten Krümmung ($\kappa = 10$ für unsere flachere Mulde, $\kappa = 1000$ für das Vergleichsbeispiel des Kapitels).

<!--en:7a6770fd-->
::: {.slide title="Entkopplung in skalare Rekursionen durch Eigenzerlegung"}

<!--en:d5f2d7c0-2-->
[Geometrie quadratischer Funktionen]{.kicker}

<!--en:b3282647-->
Wegen $\nabla f(\mathbf{x}) = \mathbf{A}\mathbf{x}$ wendet der Gradientenabstieg den linearen Operator $\mathbf{x}_{k+1} = (\mathbf{I} - \eta\mathbf{A})\mathbf{x}_k$ an. Die Darstellung von $\mathbf{x}_k$ in der orthonormalen Eigenbasis von $\mathbf{A}$ mit Koordinaten $u_i^{(k)}$ entkoppelt alle $d$ Variablen:

<!--en:655459b9-->
- **Exakte Stabilitätsgrenze:** Jede Eigenkomponente schrumpft genau dann, wenn $|1 - \eta\lambda_i| < 1$ für alle $i$ gilt. Das erfordert $0 < \eta < 2/\lambda_{\max} = 2/L$. Die hinreichende Grenze $2/L$ aus dem Abstiegslemma ist für quadratische Funktionen also auch *notwendig*.

<!--en:380f7444-->
- **Der Spektralradius bestimmt das Tempo:** Im ungünstigsten Fall fällt die Fehlernorm wie $\rho(\eta)^k$, wobei die beiden äußersten Eigenwerte $\rho(\eta) = \max_i |1 - \eta\lambda_i| = \max\big(|1 - \eta\lambda_{\min}|,\, |1 - \eta\lambda_{\max}|\big)$ bestimmen.

<!--en:c47d427b-->
::: {.slide title="Schrittweiten beiderseits der Stabilitätsgrenze"}

<!--en:d5f2d7c0-3-->
[Geometrie quadratischer Funktionen]{.kicker}

<!--en:2d4d976f-->
Für $\mathbf{A} = \mathrm{diag}(1, 10)$ mit Startpunkt $\mathbf{x}_0 = (1, 1)$ liegt die Stabilitätsgrenze bei $2/\lambda_{\max} = 0.20$:

<!--en:0f2f8d0d-->
- Bei $\eta = 0.10 = 1/L$ verschwindet die steife Komponente in $1$ Schritt ($1 - 10\eta = 0$), doch die langsame schrumpft nur mit dem Faktor $1 - \eta = 0.90$.

<!--en:7acf753a-->
- Bei $\eta = 2/11 \approx 0.182$ schrumpfen beide Komponenten mit dem Faktor $9/11 \approx 0.818$ (Vorzeichenfaktoren $+9/11$ und $-9/11$). Der Fehler nach $60$ Schritten sinkt dadurch um zwei Größenordnungen.

<!--en:e19e9cb5-->
::: {.slide title="Die optimale feste Schrittweite gleicht die äußersten Eigenkomponenten aus"}

<!--en:d5f2d7c0-4-->
[Geometrie quadratischer Funktionen]{.kicker}

<!--en:d1d8cfda-->
Da $\lambda \mapsto |1 - \eta\lambda|$ auf $[\lambda_{\min}, \lambda_{\max}]$ V-förmig ist, erreicht $\rho(\eta)$ sein Minimum bei gleichen Beträgen der Kontraktionsfaktoren an den Randpunkten:

<!--en:e2cebd80-->
- **Optimale Kontraktionsrate:**
  $$\rho(\eta^\star) = \frac{\kappa - 1}{\kappa + 1} \approx 1 - \frac{2}{\kappa}$$

<!--en:2f34e9ee-->
- **Linearer Iterationsaufwand:** Für $\|\mathbf{x}_k\| \le \epsilon\|\mathbf{x}_0\|$ genügen $k \approx \frac{\kappa}{2}\log\frac{1}{\epsilon}$ Schritte ($69$ bei $\kappa=10$, $6{,}908$ bei $\kappa=1000$).

<!--en:db51c521-->
::: {.slide title="Training neuronaler Netze am Rand der Stabilität"}

<!--en:1977ae3e-->
[Dynamik beim Deep Learning]{.kicker}

<!--en:db920d74-->
- **Fortschreitende Zunahme der Schärfe:** Beim Gradientenabstieg mit dem gesamten Datensatz steigt die Schärfe $\lambda_{\max}(\nabla^2 f(\mathbf{x}_k))$ tiefer Netze bis zur Stabilitätsgrenze $2/\eta$ und pendelt sich dort ein.

<!--en:87430ab3-->
- **Ursache der Verlustspitzen:** Sobald $\lambda_{\max} > 2/\eta$ gilt, schwingt die Komponente zum größten Eigenwert mit dem Faktor $1 - \eta\lambda_{\max} < -1$ und wachsender Amplitude.

<!--en:995af7ff-->
- **Selbststabilisierung jenseits quadratischer Funktionen:** Taylor-Terme dritter Ordnung koppeln die Schwingung an die veränderliche Krümmung. So gelangt die Iterierte in flachere Gebiete, in denen $\lambda_{\max} \approx 2/\eta$ gilt.

<!--en:d5878e84-->
::: {.slide title="Wie Konvexität die Konvergenzgarantien verbessert"}

<!--en:612a075e-->
[Konvergenzraten]{.kicker}

<!--en:94274138-->
Mit stärkeren Annahmen an $f$ garantiert der Gradientenabstieg ($\eta = 1/L$) zunächst lokale Stationarität, dann globale Optimalität und schließlich geometrische Konvergenz:

<!--en:7b13f436-->
| Klasse der Zielfunktion | Mathematische Annahme | Garantierte Rate bei $\eta = 1/L$ | Schritte für Genauigkeit $\epsilon$ |

<!--en:c1468507-->
| **Glatt, nichtkonvex** | $\nabla f$ ist $L$-Lipschitz-stetig, $f \ge f_{\mathrm{low}}$ | $\min_{k < K}\|\nabla f(\mathbf{x}_k)\|^2 \le \frac{2L(f_0 - f_{\mathrm{low}})}{K}$ | $O(1/\epsilon^2)$ für $\|\nabla f\| \le \epsilon$ |

<!--en:b5cf2347-->
| **Glatt und konvex** | $L$-glatt und $\nabla^2 f(\mathbf{x}) \succeq \mathbf{0}$ | $f(\mathbf{x}_k) - f^\star \le \frac{L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2}{2k}$ | $O(1/\epsilon)$ für $f - f^\star \le \epsilon$ |

<!--en:71e414ea-->
| **Stark konvex** | $\mu\mathbf{I} \preceq \nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$ ($\kappa = L/\mu$) | $f(\mathbf{x}_k) - f^\star \le (1 - 1/\kappa)^k(f_0 - f^\star)$ | $O(\kappa \log(1/\epsilon))$ |

<!--en:8e11488b-->
| **Quadratisch bei $\eta^\star$** | $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$, $\eta^\star = \frac{2}{\mu + L}$ | $f(\mathbf{x}_k) - f^\star \le \bigl(\frac{\kappa - 1}{\kappa + 1}\bigr)^{2k}(f_0 - f^\star)$ | $\approx \frac{\kappa}{4}\log(1/\epsilon)$ für $f - f^\star$ |

<!--en:b81f2fe0-->
::: {.slide title="Heavy-Ball-Momentum als Geschwindigkeit mit Gedächtnis"}

<!--en:331a968b-->
[Momentum]{.kicker}

<!--en:e24ed384-->
Der Gradientenabstieg braucht $O(\kappa)$ Schritte. Ohne Gedächtnis kann er steile Schwingungen nur dämpfen, indem er auch flache Richtungen bremst. Polyaks **Heavy-Ball-Momentum** führt eine exponentiell gewichtete Geschwindigkeit $\mathbf{v}_k$ mit und aktualisiert $\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}$:

<!--en:faff339e-->
- **Zweischrittrekursion:** Eliminieren von $\mathbf{v}_k = \mathbf{x}_k - \mathbf{x}_{k-1}$ ergibt $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\nabla f(\mathbf{x}_k) + \beta(\mathbf{x}_k - \mathbf{x}_{k-1})$.

<!--en:48d9cbf3-->
- **Doppelte Wirkung:** An den steilen Flanken heben sich entgegengesetzte Gradienten auf. Entlang des flachen Talbodens erhöhen gleichgerichtete Gradienten die Geschwindigkeit.

<!--en:87de0bb8-->
::: {.slide title="Charakteristisches Polynom einer Eigenkomponente mit Momentum"}

<!--en:e4b5ea71-->
[Analyse von Momentum]{.kicker}

<!--en:918df085-->
In der Eigenbasis einer quadratischen Funktion $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ erfüllt jede Eigenkomponente $u_k$ zum Eigenwert $\lambda$ die lineare Rekurrenz zweiter Ordnung $u_{k+1} = (1 + \beta - \eta\lambda)u_k - \beta u_{k-1}$. Der Ansatz $u_k = z^k$ liefert die charakteristische Gleichung:

<!--en:7c0f438d-->
- **Bereich konjugiert komplexer Wurzeln:** Die Diskriminante $(1 + \beta - \eta\lambda)^2 - 4\beta$ ist nichtpositiv, wenn gilt:
  $$(1 - \sqrt{\beta})^2 \le \eta\lambda \le (1 + \sqrt{\beta})^2$$

<!--en:0d95b1e5-->
- **Gleicher Wurzelbetrag:** Sind die Wurzeln $z, \bar{z}$ konjugiert komplex, ist ihr Produkt gleich dem konstanten Term $z\bar{z} = |z|^2 = \beta$. **Beide Wurzeln haben dann den Betrag $|z| = \sqrt{\beta}$, unabhängig von $\lambda$.**

<!--en:fbc84172-->
- **Doppelte Wurzeln am Rand:** An den beiden Grenzen $\eta\lambda = (1 \pm \sqrt{\beta})^2$ führt eine doppelte reelle Wurzel $z = \pm\sqrt{\beta}$ zum linearen Vorfaktor $|u_k| \le C\,k\,\beta^{k/2}$.

<!--en:276c8073-->
::: {.slide title="Optimale Heavy-Ball-Parameter und das √κ-Gesetz"}

<!--en:e4b5ea71-2-->
[Analyse von Momentum]{.kicker}

<!--en:532cab75-->
Liegen die äußersten Eigenwerte $\lambda_{\min}$ und $\lambda_{\max}$ genau an den Grenzen des Intervalls komplexer Wurzeln, $\eta\lambda_{\min} = (1 - \sqrt{\beta})^2$ und $\eta\lambda_{\max} = (1 + \sqrt{\beta})^2$, liegen die Wurzeln jeder Komponente auf dem Kreis mit Radius $\sqrt{\beta^\star}$. Die Schrittweite beträgt $\eta_{\mathrm{HB}} = 4 / (\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}})^2$:

<!--en:f57e51b6-->
- **Beschleunigte Kontraktion:** Jede Komponente schrumpft mit der Rate
  $$\rho_{\mathrm{HB}} = \sqrt{\beta^\star} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \approx 1 - \frac{2}{\sqrt{\kappa}}$$

<!--en:ffa67498-->
- **Beschleunigung:** Der Iterationsaufwand sinkt von $O(\kappa\log\tfrac{1}{\epsilon})$ auf $O(\sqrt{\kappa}\log\tfrac{1}{\epsilon})$.

<!--en:7b77af16-->
::: {.slide title="Nesterov-Beschleunigung und globale Garantien"}

<!--en:fbc4ca8c-->
[Beschleunigte Verfahren]{.kicker}

<!--en:9d86986d-->
**Nesterovs beschleunigtes Gradientenverfahren** wertet $\nabla f$ am vorausberechneten Punkt $\tilde{\mathbf{x}}_k = \mathbf{x}_k + \beta\mathbf{v}_k$ aus:

<!--en:52deeb4f-->
- **Vorausschauendes Bremsen:** Trägt das Momentum die Iterierte bergauf, wirkt der Schritt entgegen $\nabla f(\tilde{\mathbf{x}}_k)$ bremsend, und zwar *vor* dem Überschwingen.

<!--en:6fbfbee4-->
- **Minimax-optimale Raten:** Das Verfahren erreicht global $O(\sqrt{\kappa}\log(1/\epsilon))$ für alle $L$-glatten, $\mu$-stark konvexen Zielfunktionen und $O(1/k^2)$ für glatte konvexe Zielfunktionen. Damit erreicht es die untere Schranke von Nemirovski und Yudin.

<!--en:9f8aa561-->
::: {.slide title="Wie die gemessene Iterationszahl mit κ wächst"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:58a88bb3-->
Die Zahl der Iterationen bis $\|\mathbf{x}_k\| \le 10^{-6}\|\mathbf{x}_0\|$ für $\mathrm{diag}(1, \kappa)$ vom Startpunkt $(1, 1)$ aus bestätigt das lineare Wachstum mit $\kappa$ bzw. $\sqrt{\kappa}$:

<!--en:68e0af14-->
- Beim **Gradientenabstieg** bleibt $k/\kappa \approx 6.9$ konstant ($6{,}908$ Schritte bei $\kappa = 1000$).

<!--en:349bfd98-->
- Das **Heavy-Ball-Verfahren** braucht $315$ Schritte bei $\kappa = 1000$. Das Verhältnis $k/\sqrt{\kappa}$ steigt leicht ($8.5$ $\to$ $10.0$). Der Jordan-Block zur doppelten Wurzel erklärt dies durch den Fehlerterm $k\rho_{\mathrm{HB}}^k$, im Einklang mit $k \approx \frac{\sqrt{\kappa}}{2}(\log 10^6 + \log k)$.

<!--en:feabfc93-->
::: {.slide title="Verständnisfragen zu Schrittweiten und Momentum-Raten"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:500bb39c-->
Gegeben ist die quadratische Zielfunktion $f(x_1, x_2) = \tfrac12(x_1^2 + 100\,x_2^2)$ mit den Eigenwerten $\lambda_{\min} = 1$ und $\lambda_{\max} = 100$ ($\kappa = 100$):

<!--en:afa3b209-->
1. Wie groß sind die Stabilitätsgrenze der Lernrate beim Gradientenabstieg, die optimale feste Schrittweite $\eta^\star$ und der Kontraktionsfaktor $\rho(\eta^\star)$ pro Schritt?

<!--en:3abbe8ed-->
2. Wie viele Schritte braucht der Gradientenabstieg ungefähr, um $\|\mathbf{x}_k\|$ auf das $10^{-6}$-Fache zu verkleinern?

<!--en:78cc9e20-->
3. Welchen Kontraktionsfaktor $\rho_{\mathrm{HB}}$ erreicht Heavy-Ball-Momentum bei optimaler Parameterwahl? Wie viele Schritte braucht es ungefähr, einschließlich des logarithmischen Terms für die doppelte Wurzel?

<!--en:fa08144f-->
**Lösungen:** **(1)** Stabilitätsgrenze: $2/L = 0.02$. Optimale Schrittweite: $\eta^\star = 2/(1 + 100) = 2/101 \approx 0.0198$. Kontraktionsfaktor: $\rho(\eta^\star) = 99/101 \approx 0.9802$. **(2)** $\log(10^6)/\log(101/99) \approx \frac{100}{2}\ln(10^6) \approx 691$ Schritte. **(3)** Mit $\sqrt{\kappa} = 10$ gilt $\rho_{\mathrm{HB}} = (10 - 1)/(10 + 1) = 9/11 \approx 0.818$. Rein geometrisches Abklingen ergibt $69$ Schritte. Durch den Faktor $k\rho_{\mathrm{HB}}^k$ der doppelten Wurzel kommen $\frac{\sqrt{\kappa}}{2}\ln k \approx 24$ Schritte hinzu. Das entspricht den gemessenen $93$ Schritten.

<!--en:fdd970e4-->
::: {.slide title="Zusammenfassung zu Verfahren erster Ordnung und Krümmung"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:96a945e3-->
| Verfahren oder Begriff | Schrittregel oder Formel | Kontraktion oder Garantie | Kernaussage |

<!--en:0a0b4b5e-->
| **Abstiegslemma** | $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\mathbf{g}_k$, $0 < \eta < 2/L$ | $\Delta f \le -\eta(1 - L\eta/2)\|\mathbf{g}_k\|^2$ | Die größte Krümmung $L$ begrenzt sichere Schrittweiten auf Werte unter $2/L$ |

<!--en:af29671f-->
| **Backtracking (Armijo)** | $\eta$ halbieren, bis $f(\mathbf{x} - \eta\mathbf{g}) \le f(\mathbf{x}) - c\eta\|\mathbf{g}\|^2$ gilt | Liefert $\eta \ge \min(\eta_0, (1-c)/L_{\mathrm{loc}})$ | Passt sich ohne Kenntnis von $L$ an die lokale Krümmung an |

<!--en:afbea15c-->
| **Gradientenabstieg mit optimaler fester Schrittweite** | $\eta^\star = \frac{2}{\lambda_{\min} + \lambda_{\max}}$ | $\rho_{\mathrm{GD}} = \frac{\kappa - 1}{\kappa + 1} \approx 1 - \frac{2}{\kappa}$ | Iterationszahl wächst linear mit $\kappa = L/\mu$ |

<!--en:e05fbbf9-->
| **Heavy-Ball-Momentum** | $\mathbf{v}_{k+1} = \beta^\star\mathbf{v}_k - \eta_{\mathrm{HB}}\mathbf{g}_k$ | $\rho_{\mathrm{HB}} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \approx 1 - \frac{2}{\sqrt{\kappa}}$ | Komplexe Wurzeln gleichen die Komponenten auf denselben Wurzelbetrag $\sqrt{\beta^\star}$ an |

<!--en:f12bf86d-->
| **Nesterovs vorausschauender Schritt** | $\mathbf{v}_{k+1} = \beta\mathbf{v}_k - \frac{1}{L}\nabla f(\mathbf{x}_k + \beta\mathbf{v}_k)$ | $\rho_{\mathrm{NAG}} = 1 - \frac{1}{\sqrt{\kappa}}$ ($O(1/k^2)$ bei $\mu=0$) | Erreicht die optimale Ordnung $O(\sqrt{\kappa})$ global auch jenseits quadratischer Funktionen |
