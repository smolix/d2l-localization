<!--en:9042de8d-->
# Analysis mehrerer Variablen

<!--en:d665c952-->
Im vorigen Abschnitt haben wir den Verlust als Funktion eines einzigen Gewichts untersucht und alle anderen Gewichte festgehalten. Beim Training ändern sich die Gewichte jedoch gemeinsam. Jeder Schritt des Gradientenabstiegs verändert alle Gewichte zugleich ein wenig. Wir müssen also wissen, wie der Verlust auf einen solchen Schritt reagiert und bei welchem Schritt er am schnellsten sinkt. Eine Million einzelner Steigungen, eine pro Gewicht, beantwortet für sich genommen keine der beiden Fragen.

<!--en:9609d42b-->
Unter einer recht schwachen Voraussetzung beantworten diese Steigungen jedoch beide Fragen. Ist der Verlust differenzierbar, so sagt der Vektor der Steigungen, der *Gradient*, die Wirkung jedes kleinen gemeinsamen Schritts bis zur ersten Ordnung voraus. Er zeigt in die Richtung des steilsten Anstiegs. Um darüber hinaus ein lokales Minimum von einem Sattelpunkt zu unterscheiden, brauchen wir eine symmetrische Matrix aus zweiten Ableitungen, die *Hesse-Matrix* (engl. Hessian). Sie beschreibt, wie sich die Fläche krümmt. In diesem Abschnitt untersuchen wir, was diese beiden Größen über eine Funktion vieler Variablen aussagen. Wie ein Programm den Gradienten für ein großes Netz berechnet, ist eine eigene Frage. Ihr widmet sich als Nächstes :numref:`sec_mdl-chain-rule-graphs`.

<!--en:e339dc63-->
In :numref:`sec_calculus` haben wir bereits die Ableitungsregeln angegeben und den Gradienten eingesetzt. Wie in :numref:`sec_mdl-single_variable_calculus` für eine Variable erarbeiten wir hier die Grundlagen. Wir beweisen, wann die lineare Näherung gilt und wie wir kritische Punkte anhand zweiter Ableitungen klassifizieren können. Zwei Ergebnisse ziehen sich durch den gesamten Abschnitt: das lineare Modell für mehrere Variablen und das Kriterium der zweiten Ableitung. Den kurzen Unterabschnitt zu Tangentialebenen und den abschließenden Teil zu Lagrange-Multiplikatoren können Sie beim ersten Lesen überspringen. Jede Überlegung prüfen wir am selben Beispiel, der Log-Sum-Exp-Funktion zweier Variablen $f(x, y) = \log(e^x + e^y)$ an der Stelle $(0, \log 2)$. Als ihr Gradient ergibt sich die bekannte Softmax-Funktion.

<!--en:b50c0c6c-->
## Partielle Ableitungen und der Gradient

<!--en:7133f178-->
### Partielle Ableitungen

<!--en:5dc90f87-->
Halten wir alle Gewichte bis auf $w_1$ fest, so wird der Verlust zu einer Funktion einer einzigen Variablen. Ihre gewöhnliche Ableitung ist die *partielle Ableitung* (engl. partial derivative) $\partial\ell/\partial w_1$ aus :numref:`sec_calculus`. Geometrisch beschreibt sie die Steigung des eindimensionalen Schnitts durch den Graphen von $\ell$, den wir erhalten, wenn wir nur $w_1$ variieren. Wie bei Funktionen einer Variablen sagt diese Steigung die Wirkung einer kleinen Änderung $\epsilon_1$ allein in $w_1$ voraus:

<!--en:75602cac-->
Berechnen wir die partiellen Ableitungen für unser Beispiel $f(x, y) = \log(e^x + e^y)$. Halten wir $y$ fest und leiten nach $x$ ab, erhalten wir $\partial f/\partial x = e^x/(e^x+e^y)$ und aus Symmetriegründen $\partial f/\partial y = e^y/(e^x+e^y)$. Am Punkt $(0, \log 2)$ gilt $e^x = 1$ und $e^y = 2$. Somit erhalten wir $f(0, \log 2) = \log 3 \approx 1.0986$, $\partial f/\partial x = \tfrac13$ und $\partial f/\partial y = \tfrac23$. Erhöhen wir also nur $y$ ein wenig, so steigt $f$ doppelt so schnell wie bei einer gleich großen Erhöhung von $x$ allein.

<!--en:7a74358f-->
### Differenzierbarkeit und Gradient

<!--en:21c17cf5-->
Partielle Ableitungen geben die Steigung entlang jeder einzelnen Koordinatenachse an. Daraus folgt noch nicht, was passiert, wenn sich alle Koordinaten zugleich ändern, etwa bei einem Schritt von $(0, \log 2)$ um $\boldsymbol{\epsilon} = (0.01, -0.03)$. Es liegt nahe, die Änderungen entlang der beiden Koordinaten zu addieren, sodass sich $\tfrac13 \cdot 0.01 + \tfrac23 \cdot (-0.03)$ ergibt. Diese Vermutung ist eine weitreichende Aussage über das Verhalten von $f$ in der Nähe des Punktes und erfordert eine Voraussetzung.

<!--en:5cf303bd-->
Wir setzen voraus, dass sich in der Nähe von $\mathbf{w}$ die Funktion $\ell$ bis auf einen Fehler linear verhält, der schneller gegen null geht als die Schrittweite. Formal heißt $\ell$ an der Stelle $\mathbf{w}$ *differenzierbar*, wenn es eine lineare Abbildung $\Lambda$ gibt, für die gilt:

<!--en:866dcc0f-->
Dabei bezeichnet $o(\|\boldsymbol{\epsilon}\|)$ einen Restterm, dessen Quotient mit $\|\boldsymbol{\epsilon}\|$ für $\boldsymbol{\epsilon} \to \mathbf{0}$ gegen null geht.

<!--en:5f3438e4-->
Existiert eine solche lineare Abbildung $\Lambda$, so legen die partiellen Ableitungen sie vollständig fest. Setzen wir $\boldsymbol{\epsilon} = h\mathbf{e}_i$ in :eqref:`eq_mdl-differentiable` ein, teilen beide Seiten durch $h$ und bilden den Grenzwert für $h \to 0$, so folgt $\Lambda(\mathbf{e}_i) = \partial\ell/\partial w_i$. Da $\Lambda$ linear ist, steht damit ihr Wert für jeden gemeinsamen Schritt $\boldsymbol{\epsilon}$ fest:

<!--en:9f29497a-->
Diesen Spaltenvektor $\nabla_{\mathbf{w}}\ell$ der partiellen Ableitungen nennen wir den *Gradienten* von $\ell$. Setzen wir ihn in :eqref:`eq_mdl-differentiable` ein, erhalten wir die lineare Näherung für mehrere Variablen:

<!--en:036a5e0b-->
Vergleichen wir dies mit der Näherung $f(x+\epsilon) \approx f(x) + \epsilon f'(x)$ für eine Variable. Die Form ist dieselbe: An die Stelle der skalaren Steigung tritt ein Vektor, an die Stelle der gewöhnlichen Multiplikation ein Skalarprodukt. Der Gradient ist der eindeutig bestimmte Vektor, dessen Skalarprodukt mit einem Schritt die Änderung von $\ell$ in erster Ordnung ergibt. Von den verschiedenen Sichtweisen auf den Gradienten sollten Sie sich vor allem diese merken. In diesem Buch fassen wir den Gradienten stets als Spaltenvektor auf.

<!--en:9064f264-->
In unserem Beispiel gilt am Punkt $(0, \log 2)$ die Beziehung $\nabla f = [\tfrac13, \tfrac23]^\top$. Setzen wir $\boldsymbol{\epsilon} = (0.01, -0.03)$ in :eqref:`eq_mdl-nabla_use` ein, erhalten wir die Vorhersage $f(0.01, \log 2 - 0.03) \approx \log 3 + \tfrac13 \cdot 0.01 - \tfrac23 \cdot 0.03 \approx 1.081946$. An *jedem* Punkt $(x, y)$ sind die beiden Einträge von $\nabla f$ positiv und ergeben in der Summe eins. Der Gradient von Log-Sum-Exp ist die Softmax-Funktion $\mathbf{p} = \operatorname{softmax}(x, y)$ mit den Einträgen $e^x/(e^x+e^y)$ und $e^y/(e^x+e^y)$.

<!--en:95cb5786-->
Warum verlangt Differenzierbarkeit mehr als die bloße Existenz partieller Ableitungen? Betrachten wir $f(x, y) = 2xy/(x^2+y^2)$ mit $f(0, 0) = 0$. Auf beiden Koordinatenachsen ist $f$ überall null. Daher haben beide partiellen Ableitungen im Ursprung den Wert $0$, und als lineares Modell kommt nur die konstante Funktion $0$ infrage. Auf der Diagonalen $x = y \neq 0$ gilt jedoch $f(x, x) = 1$, wie nahe wir dem Ursprung auch kommen. Dort kann es also keine lineare Näherung geben. In diesem Beispiel ist $f$ im Ursprung nicht einmal stetig. Aufgabe :exref:`ex_mdl-multivariable-calculus-no-gradient` zeigt, dass sogar eine stetige Funktion alle partiellen Ableitungen besitzen kann, ohne differenzierbar zu sein.

<!--en:4bb642c6-->
Zum Glück müssen wir :eqref:`eq_mdl-differentiable` selten direkt prüfen. Es gibt eine praktische hinreichende Bedingung: Alle partiellen Ableitungen existieren in einer Umgebung von $\mathbf{w}$ und sind an der Stelle $\mathbf{w}$ stetig :cite:`Rudin.1976`. Da die partiellen Ableitungen von Log-Sum-Exp überall stetig sind, ist die Funktion überall differenzierbar.

<!--en:00caad63-->
Die Definition der Differenzierbarkeit garantiert für den Näherungsfehler lediglich $o(\|\boldsymbol{\epsilon}\|)$. Er geht also schneller gegen null als $\|\boldsymbol{\epsilon}\|$. Wie viel schneller geht das in der Praxis? Schätzen Sie vor dem Ausführen der nächsten Zelle, um welchen Faktor der Fehler des linearen Modells schrumpft, wenn wir den Schritt $\boldsymbol{\epsilon} = (0.01, -0.03)$ auf ein Zehntel verkleinern. Das folgende Experiment prüft dies allein mit NumPy.

<!--en:e161ad41-->
Bei $\|\boldsymbol{\epsilon}\| \approx 0.03$ weicht das lineare Modell um $1.79 \times 10^{-4}$ vom wahren Wert ab (erst in der vierten Nachkommastelle). Verkleinern wir den Schritt jeweils auf ein Zehntel, so sinkt die Abweichung jedes Mal auf ein Hundertstel. Der Fehler ist also von zweiter Ordnung in der Schrittweite. Das ist eine wesentlich schärfere Aussage, als die Differenzierbarkeit allein erlaubt. Mithilfe der Hesse-Matrix können wir diese Abweichung bald genau vorhersagen (:numref:`subsec_mdl-second-order-taylor`).

<!--en:f2122ebd-->
### Richtungsableitungen

<!--en:fab1b612-->
Bisher haben wir mit dem Gradienten die Wirkung kleiner Schritte vorhergesagt. Er liefert auch die genaue momentane Steigung in jeder beliebigen Richtung, einschließlich der Koordinatenrichtungen. Wählen wir einen beliebigen Einheitsvektor $\mathbf{u}$, setzen $\boldsymbol{\epsilon} = h\mathbf{u}$ in :eqref:`eq_mdl-nabla_use` ein, teilen durch $h$ und bilden den Grenzwert für $h \to 0$. Der Restterm der Ordnung $o(\|h\mathbf{u}\|)$ trägt dann nichts mehr bei, und es bleibt

<!--en:66f78010-->
Der Grenzwert auf der linken Seite ist die *Richtungsableitung* (engl. directional derivative) von $\ell$ in Richtung $\mathbf{u}$. Wie die rechte Seite zeigt, ist sie die skalare Projektion des Gradienten auf $\mathbf{u}$. Ein einziger Vektor beschreibt also die Steigung in jeder möglichen Richtung.

<!--en:5d5cd1b1-->
Drei Richtungen in unserem Beispiel am Punkt $(0, \log 2)$ zeigen das ganze Spektrum. Entlang der Diagonalen $\mathbf{u} = (1, 1)/\sqrt{2}$ beträgt die Steigung $(\tfrac13 + \tfrac23)/\sqrt{2} = 1/\sqrt{2} \approx 0.707$. Diese Steigung ist an *jedem* Punkt gleich, denn es gilt $f(x+t, y+t) = \log\bigl(e^t(e^x+e^y)\bigr) = f(x, y) + t$. Legen wir auf der Diagonalen die Strecke $\sqrt{2}\,t$ zurück, so steigt $f$ stets um $t$. Gehen wir dem Gradienten genau entgegen, also in Richtung $\mathbf{u} = -(1, 2)/\sqrt{5}$, beträgt die Steigung $-(\tfrac13 + \tfrac43)/\sqrt{5} = -\sqrt{5}/3 \approx -0.745$. In Richtung $\mathbf{u} = (2, -1)/\sqrt{5}$ schließlich ist die Steigung $(\tfrac23 - \tfrac23)/\sqrt{5} = 0$. In erster Ordnung ändert sich $f$ in dieser Richtung also überhaupt nicht.

<!--en:a4ad7a20-->
## Die Geometrie des Gradienten

<!--en:500b6622-->
### Steilster Abstieg

<!--en:d27dcf6d-->
Der Gradientenabstieg (:numref:`sec_linear_regression`) verschiebt die Parameter ein kleines Stück in die Richtung, in der der Verlust am schnellsten fällt. Nach :eqref:`eq_mdl-directional-derivative` suchen wir den Einheitsvektor $\mathbf{v}$, für den die Projektion $\mathbf{v}\cdot\nabla\ell$ möglichst negativ ist. In :numref:`sec_calculus` haben wir diese Richtung aus dem Kosinus des Winkels zwischen $\mathbf{v}$ und dem Gradienten bestimmt. Die Cauchy-Schwarz-Ungleichung liefert einen noch einfacheren Beweis und bestätigt, dass die Richtung eindeutig ist.

<!--en:e8815def-->
**Proposition (steilster Anstieg und Abstieg).** *An jedem Punkt mit $\nabla \ell \neq \mathbf{0}$ gilt unter allen Einheitsvektoren $\mathbf{v}$: Die Richtungsableitung $\mathbf{v}\cdot\nabla \ell$ nimmt für $\mathbf{v} = \nabla\ell/\|\nabla\ell\|$ ihren größten und für $\mathbf{v} = -\nabla\ell/\|\nabla\ell\|$ ihren kleinsten Wert an. Damit ist $+\nabla \ell$ die Richtung des steilsten Anstiegs und $-\nabla \ell$ die des steilsten Abstiegs.*

<!--en:d5b6b0f0-->
**Beweis.** Nach der Cauchy-Schwarz-Ungleichung (:numref:`sec_mdl-geometry-linear-algebraic-ops`) gilt $|\mathbf{v}\cdot\mathbf{g}| \le \|\mathbf{v}\|\|\mathbf{g}\|$, wobei Gleichheit genau dann eintritt, wenn $\mathbf{v}$ und $\mathbf{g}$ kollinear sind. Für einen Einheitsvektor $\mathbf{v}$ und $\mathbf{g} = \nabla \ell$ ergibt das

<!--en:e1cca1c3-->
Jede Schranke erreicht genau ein zu $\nabla \ell$ kollinearer Einheitsvektor: die obere Schranke $\mathbf{v} = \nabla \ell/\|\nabla \ell\|$ und die untere $\mathbf{v} = -\nabla \ell/\|\nabla \ell\|$. $\blacksquare$

<!--en:46f8d59e-->
In unserem Beispiel ist am Punkt $(0, \log 2)$ die Richtung des steilsten Abstiegs $-(1, 2)/\sqrt{5}$ mit der Steigung $-\|\nabla f\| = -\sqrt{5}/3 \approx -0.745$. Dies ist die kleinste der drei Steigungen, die wir gerade berechnet haben. Ein Schritt in Richtung $-\nabla_{\mathbf{w}} \ell$ ergibt die bekannte Aktualisierungsregel des Gradientenabstiegs:

<!--en:20cbf4ef-->
Bei dieser Aktualisierung normieren wir den Gradienten nicht. Die Schrittlänge beträgt $\eta\|\nabla\ell\|$ und nimmt damit von selbst ab, wenn der Gradient in der Nähe eines Minimums kleiner wird.

<!--en:5472a183-->
Leicht übersieht man, dass „Einheitsvektor“ im Satz einen Vektor der *euklidischen* Länge eins bezeichnet. Das ist eine Modellierungsentscheidung und kein Naturgesetz. Beschränken wir stattdessen jede Koordinate des Schritts auf einen Betrag von höchstens eins, ergibt sich der Schritt des steilsten Abstiegs durch komponentenweise Anwendung von $-\operatorname{sign}(\nabla \ell)$. Am Punkt $(0, \log 2)$ lautet dieser Schritt $(-1, -1)$ mit der Steigung $-(\tfrac13 + \tfrac23) = -1$. Adaptive Optimierer wie Adam teilen jede Koordinate der Aktualisierung durch eine laufende Schätzung ihrer Größenordnung (:numref:`chap_optimization`). Ihr Verhalten ähnelt diesem vorzeichenbasierten Schritt viel stärker als dem euklidischen steilsten Abstieg. Aufgabe :exref:`ex_mdl-gradient-based-optimization-steepest-other-norms` untersucht den steilsten Abstieg für andere Normen.

<!--en:73afaf3f-->
### Gradienten und Niveaumengen

<!--en:7da5b9ec-->
Wir können den Gradienten auch direkt aus einem Höhenlinienbild ablesen, ganz ohne Schritte zu betrachten. Eine *Niveaumenge* (engl. level set) von $\ell$ besteht aus allen Punkten, an denen $\ell$ einen festen Wert $c$ annimmt. Wenn wir uns auf einer Niveaumenge bewegen, bleibt $\ell$ konstant. Die Richtungsableitung sollte also in jeder Tangentialrichtung verschwinden.

<!--en:dde884f0-->
**Proposition (Gradienten stehen senkrecht auf Niveaumengen).** *Sei $\ell$ an der Stelle $\mathbf{x}$ differenzierbar. Dann steht $\nabla \ell(\mathbf{x})$ senkrecht auf jeder Richtung, die an der Stelle $\mathbf{x}$ tangential zur Niveaumenge $\{\mathbf{x}' : \ell(\mathbf{x}') = c\}$ durch $\mathbf{x}$ verläuft.*

<!--en:2e27f53d-->
**Beweis.** Ein Vektor $\mathbf{v}$ ist im Punkt $\mathbf{x}$ *tangential* zur Niveaumenge, wenn $\mathbf{v} = \boldsymbol{\gamma}'(0)$ für eine differenzierbare Kurve $\boldsymbol{\gamma}(t)$ gilt, die ganz in der Niveaumenge liegt und $\boldsymbol{\gamma}(0) = \mathbf{x}$ erfüllt. Entlang einer solchen Kurve gilt $\ell(\boldsymbol{\gamma}(t)) = c$ für alle $t$. Die Ableitung von $t \mapsto \ell(\boldsymbol{\gamma}(t))$ an der Stelle $t = 0$ muss also null sein. In erster Ordnung gilt $\boldsymbol{\gamma}(t) = \mathbf{x} + t\mathbf{v} + o(t)$. Nach :eqref:`eq_mdl-nabla_use` ist die Ableitung daher gleich $\mathbf{v}\cdot\nabla \ell(\mathbf{x})$. Folglich gilt $\mathbf{v}\cdot\nabla \ell(\mathbf{x}) = 0$. $\blacksquare$

<!--en:5183af45-->
Prüfen wir dies an unserem Beispiel. Die Niveaumenge von $f$ durch $(0, \log 2)$ ist die Kurve $e^x + e^y = 3$. Ableiten entlang dieser Kurve ergibt $e^x\,dx + e^y\,dy = 0$. Eine Tangentialrichtung am Punkt $(0, \log 2)$ ist somit $(e^y, -e^x) = (2, -1)$. Das Skalarprodukt mit dem Gradienten bestätigt $(2, -1)\cdot(\tfrac13, \tfrac23) = 0$. Dies ist die Richtung mit Steigung null, die wir zuvor gefunden haben.

<!--en:af3ecb54-->
In einem Höhenlinienbild schneidet der Gradient also jede Niveaulinie im rechten Winkel und zeigt in Richtung höherer Werte. Unterscheiden sich die Werte benachbarter Niveaulinien um $\Delta c$ und liegen diese im Abstand $d$, so beträgt die Steigung quer zu ihnen ungefähr $\Delta c/d$. Der Gradient ist daher dort am längsten, wo die Niveaulinien am dichtesten beieinanderliegen (:numref:`fig_mdl-cal-gradient-field`).

<!--en:6bb54c29-->
![Niveaulinien eines Verlusts $\ell$ (grau) mit $\nabla \ell$ an mehreren Punkten. Jeder Gradient steht senkrecht auf seiner Niveaulinie und zeigt zu größeren Werten. Am längsten ist er dort, wo die Niveaulinien am dichtesten beieinanderliegen.](../img/mdl-cal-gradient-field.svg)

<!--en:52d86457-->
### Tangentialebenen

<!--en:e52694e6-->
Lassen wir den Fehlerterm in der linearen Näherung :eqref:`eq_mdl-nabla_use` weg und bezeichnen die vorhergesagte Höhe mit $z$, erhalten wir die *Tangentialebene* (engl. tangent plane) an den Graphen von $f$ an der Stelle $\mathbf{x}_0$:

<!--en:08cb7848-->
Für unser Beispiel am Punkt $(0, \log 2)$ lautet die Ebenengleichung $z = \log 3 + \tfrac13 x + \tfrac23 (y - \log 2)$. Schreiben wir :eqref:`eq_mdl-tangent_plane` in der Form $\nabla f(\mathbf{x}_0)\cdot(\mathbf{x}-\mathbf{x}_0) - (z - f(\mathbf{x}_0)) = 0$, sehen wir sofort, dass der Vektor $[\nabla f(\mathbf{x}_0), -1]^\top$ im $(\mathbf{x}, z)$-Raum senkrecht auf der Tangentialebene steht. Lassen wir seine Höhenkoordinate weg, bleibt $\nabla f(\mathbf{x}_0)$ als Normale zur Niveaulinie in der Grundebene (:numref:`fig_mdl-tangent-plane`).

<!--en:ae25551e-->
![Eine Fläche $z = f(\mathbf{x})$ (blau) mit ihrer Tangentialebene in einem Punkt (orange). Der Vektor $[-\nabla f, 1]$ ist der Gegenvektor zu $[\nabla f, -1]$ und steht senkrecht auf der Ebene (schwarzer Pfeil). Ohne die Höhenkoordinate bleibt ein Vielfaches von $\nabla f$. Dieser Vektor in der Grundebene (oranger Pfeil) steht senkrecht auf der Niveaulinie durch den Punkt (schwarz).](../img/mdl-cal-tangent-plane.svg)

<!--en:bba23847-->
### Kritische Punkte

<!--en:786f5445-->
Verlustfunktionen im Deep Learning sind viel zu kompliziert, um ihre Minima in geschlossener Form zu bestimmen. Das lineare Modell liefert dennoch eine notwendige Bedingung für jedes Minimum. Genau dort, wo sie erfüllt ist, kommt der Gradientenabstieg zum Stillstand. Im Folgenden sei $f$ auf ganz $\mathbb{R}^n$ definiert und an der Stelle $\mathbf{x}_0$ differenzierbar.

<!--en:063e4186-->
Gilt $\nabla f(\mathbf{x}_0) \neq \mathbf{0}$, können wir einen Schritt entgegen dem Gradienten gehen, indem wir $\boldsymbol{\epsilon} = -\eta\nabla f(\mathbf{x}_0)$ mit $\eta > 0$ wählen. Nach :eqref:`eq_mdl-nabla_use` ändert sich dadurch $f$ um $-\eta\|\nabla f(\mathbf{x}_0)\|^2 + o(\eta)$. Für jedes hinreichend kleine $\eta$ ist diese Änderung strikt negativ. Also kann $\mathbf{x}_0$ keine lokale Minimalstelle sein. Wiederholen wir das Argument mit $+\eta$, scheidet auch eine lokale Maximalstelle aus. Somit gilt **an jeder lokalen Minimal- oder Maximalstelle $\nabla f(\mathbf{x}_0) = \mathbf{0}$.** Punkte, an denen der Gradient verschwindet, heißen *kritische Punkte* (engl. critical points). Diese Bedingung erster Ordnung ist notwendig, aber nicht hinreichend. Das nächste Beispiel zeigt, was wir darüber hinaus brauchen, um ein globales Extremum nachzuweisen.

<!--en:547e3db0-->
Betrachten wir $\psi(x, y) = x e^{-x^2-y^2}$. Die Fläche bildet für $x > 0$ einen Hügel mit positiven Werten und für $x < 0$ eine entsprechende Senke mit negativen Werten. Ihre partiellen Ableitungen lauten

<!--en:7971326b-->
Setzen wir die zweite der beiden partiellen Ableitungen gleich null, muss entweder $x = 0$ oder $y = 0$ gelten. Für $x = 0$ hat die erste partielle Ableitung den Wert $e^{-y^2} \neq 0$. Es muss also $y = 0$ gelten, und dann folgt aus $1 - 2x^2 = 0$ die Bedingung $x = \pm 1/\sqrt{2}$. An den beiden kritischen Punkten nimmt die Funktion die Werte $\psi(\mp 1/\sqrt{2}, 0) = \mp e^{-1/2}/\sqrt{2} \approx \mp 0.4289$ an.

<!--en:04843eea-->
Um zu schließen, dass $(-1/\sqrt{2}, 0)$ die *globale* Minimalstelle ist, müssen wir allerdings wissen, dass überhaupt ein globales Minimum existiert. Außerhalb der Kreisscheibe mit Radius $R = 2$ gilt wegen des exponentiellen Abfalls $|\psi| \le R e^{-R^2} < 0.04$. Im Inneren gilt dagegen $\psi(-1/\sqrt{2}, 0) \approx -0.43$. Da $\psi$ stetig ist, nimmt die Funktion auf der abgeschlossenen Kreisscheibe ihr Minimum an. Dieses Minimum liegt unter $-0.04$. Die Minimalstelle kann also nicht auf dem Rand liegen und muss eine globale Minimalstelle von $\psi$ auf ganz $\mathbb{R}^2$ sein, mithin ein kritischer Punkt. Vergleichen wir unsere beiden Kandidaten, finden wir $(-1/\sqrt{2}, 0)$ als globale Minimalstelle und mit demselben Argument $(+1/\sqrt{2}, 0)$ als globale Maximalstelle. Ohne den Existenznachweis versagt der Schluss: Die Funktion $x^3$ hat genau einen kritischen Punkt, aber nirgends ein Minimum.

<!--en:1bd324ae-->
Hier sehen wir auch die Grenze der Information erster Ordnung. An der Minimalstelle und an der Maximalstelle verschwindet der Gradient, sodass wir sie mit ihm allein nicht unterscheiden können. Dazu betrachten wir nun zweite Ableitungen (:numref:`subsec_mdl-second-derivative-test`).

<!--en:02e03698-->
## Die Hesse-Matrix

<!--en:2afa2b9f-->
### Symmetrie der Hesse-Matrix

<!--en:b49b02b8-->
Das lineare Modell erfasst nur das Verhalten erster Ordnung. Ob eine Funktion an einem kritischen Punkt ein Minimum, ein Maximum oder keines von beiden hat, hängt davon ab, wie sich die Fläche von ihrer Tangentialebene wegkrümmt. Diese Krümmung messen wir mit zweiten Ableitungen. Eine Funktion von $n$ Variablen hat $n^2$ zweite partielle Ableitungen,

<!--en:0c3bccd8-->
die wir zur *Hesse-Matrix* zusammenfassen:

<!--en:82a7531b-->
Der Eintrag $(i, j)$ misst, wie schnell sich die $j$-te Steigung ändert, wenn wir uns entlang der Koordinate $x_i$ bewegen. Für die Hesse-Matrix schreiben wir auch $\nabla^2 f$ oder kurz $\mathbf{H}$, wenn Funktion und Auswertungspunkt aus dem Zusammenhang klar sind.

<!--en:134409d4-->
Schreiben wir für Log-Sum-Exp $(x_1, x_2) = (x, y)$ und leiten die Einträge der Softmax-Funktion $p_i = e^{x_i}/(e^{x_1}+e^{x_2})$ noch einmal ab. Die Quotientenregel liefert $\partial p_i/\partial x_j = p_i\delta_{ij} - p_ip_j$. Dabei hat $\delta_{ij}$ den Wert $1$ für $i = j$ und sonst den Wert $0$. In Matrixform ergibt das

<!--en:75497027-->
Diese $2 \times 2$-Matrix hat den Eigenwert $0$ in Richtung $(1, 1)$ und den Eigenwert $\tfrac49$ in Richtung $(1, -1)$. Der Eigenwert null passt zu unserer früheren Beobachtung: Entlang der Diagonalen $(1, 1)$ ist die Funktion gemäß $f(x+t, y+t) = f(x, y) + t$ linear und hat keinerlei Krümmung. Auf diese positiv semidefinite, singuläre Hesse-Matrix kommen wir beim Kriterium der zweiten Ableitung zurück.

<!--en:92ea69bb-->
Die $n^2$ Einträge einer Hesse-Matrix sind nicht unabhängig voneinander. Ist $f$ zweimal stetig differenzierbar, so spielt die Reihenfolge der Ableitungen keine Rolle und die Hesse-Matrix ist symmetrisch. Dieses klassische Ergebnis ist als Satz von Schwarz oder Clairaut–Schwarz bekannt :cite:`Rudin.1976`. Die Symmetrie ist entscheidend: Nach dem Spektralsatz (:numref:`sec_mdl-eigendecompositions`) garantiert sie reelle Eigenwerte und eine Basis aus orthogonalen Eigenvektoren. Das ermöglicht das Kriterium der zweiten Ableitung.

<!--en:103ee04b-->
**Proposition (Symmetrie der Hesse-Matrix nach Clairaut–Schwarz).** *Die ersten partiellen Ableitungen von $f$ und die gemischten partiellen Ableitungen $\frac{\partial^2 f}{\partial x_i \partial x_j}$ und $\frac{\partial^2 f}{\partial x_j \partial x_i}$ sollen in einer Umgebung von $\mathbf{x}$ existieren. Sind die gemischten partiellen Ableitungen an der Stelle $\mathbf{x}$ stetig, so gilt*

<!--en:ee15da65-->
*Gilt dies für alle $i, j$, so ist die Hesse-Matrix symmetrisch: $\mathbf{H}_f = \mathbf{H}_f^\top$.*

<!--en:e3e96e43-->
**Beweis.** Wir stellen beide gemischten partiellen Ableitungen als Grenzwerte *derselben* Größe dar, nämlich der symmetrischen zweiten Differenz

<!--en:c6a7b59c-->
Vertauschen wir $i$ und $j$, ändert sich ihr Wert offenbar nicht. Wir wollen $\Delta_h$ als gemischte partielle Ableitung an einem benachbarten Punkt ausdrücken. Definieren wir dazu die Funktion einer Variablen $\varphi(t) = f(\mathbf{x}+t\mathbf{e}_i+h\mathbf{e}_j) - f(\mathbf{x}+t\mathbf{e}_i)$, sodass $h^2\Delta_h = \varphi(h)-\varphi(0)$ gilt. Wir wenden den Mittelwertsatz (:numref:`sec_mdl-mvt`) zweimal an: zunächst auf $\varphi$ und dann entlang $x_j$ auf die Differenz der ersten partiellen Ableitungen in $\varphi'$. So erhalten wir Zahlen $\tau$ und $\sigma$ zwischen $0$ und $h$, für die gilt:

<!--en:272b6d9c-->
Damit gilt $\Delta_h = \frac{\partial^2 f}{\partial x_j \partial x_i}(\boldsymbol{\xi}_h)$ für einen Punkt $\boldsymbol{\xi}_h$ im Quadrat mit den gegenüberliegenden Ecken $\mathbf{x}$ und $\mathbf{x} + h\mathbf{e}_i + h\mathbf{e}_j$. Sein Abstand beträgt höchstens $\sqrt{2}\,|h|$ von $\mathbf{x}$. Vertauschen wir $i$ und $j$, bleibt $\Delta_h$ gleich, und wir erhalten $\Delta_h = \frac{\partial^2 f}{\partial x_i \partial x_j}(\boldsymbol{\xi}'_h)$ an einem anderen Punkt desselben Quadrats. Für $h \to 0$ konvergieren die beiden Ausdrücke wegen der Stetigkeit der gemischten partiellen Ableitungen an der Stelle $\mathbf{x}$ gegen die jeweiligen gemischten partiellen Ableitungen an der Stelle $\mathbf{x}$. Da die Ausdrücke für jedes $h$ übereinstimmen, stimmen auch ihre Grenzwerte überein. $\blacksquare$

<!--en:11f2dc33-->
Auf die Stetigkeitsvoraussetzung können wir nicht verzichten. Es gibt Funktionen, deren gemischte partielle Ableitungen überall existieren, aber im Ursprung unstetig sind und dort verschiedene Werte haben.

<!--en:1dff931d-->
### Die Taylor-Näherung zweiter Ordnung

<!--en:c69bc4fe-->
So wie der Gradient die beste lineare Näherung an $f$ liefert, liefert die Hesse-Matrix die beste *quadratische* Näherung. Woher die Koeffizienten kommen, sehen wir am einfachsten an einem quadratischen Polynom. Betrachten wir $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$. Für Funktionswert, Gradient und Hesse-Matrix :eqref:`eq_mdl-hess_def` erhalten wir im Ursprung

<!--en:07fb403d-->
Setzen wir diese Größen zusammen, erhalten wir genau das Ausgangspolynom: $f(\mathbf{x}) = f(\mathbf{0}) + \nabla f(\mathbf{0}) \cdot \mathbf{x} + \tfrac12\mathbf{x}^\top \mathbf{H}_f(\mathbf{0})\, \mathbf{x}$. Der Faktor $\tfrac12$ hebt den Faktor $2$ auf, der beim Ableiten eines Quadrats entsteht.

<!--en:16751e39-->
Für eine allgemeine Funktion $f$ ist das quadratische Modell nicht mehr exakt. Wie genau es ist, sehen wir, wenn wir $f$ auf eine Strecke einschränken. Sei $f$ in einer Umgebung der Strecke von $\mathbf{x}_0$ nach $\mathbf{x}$ zweimal stetig differenzierbar. Bezeichnen wir die Verschiebung mit $\boldsymbol{\delta} = \mathbf{x} - \mathbf{x}_0$ und betrachten wir den eindimensionalen Schnitt $\varphi(t) = f(\mathbf{x}_0 + t\boldsymbol{\delta})$. Nach dem linearen Modell :eqref:`eq_mdl-nabla_use` ist seine erste Ableitung $\varphi'(t) = \sum_i \frac{\partial f}{\partial x_i}(\mathbf{x}_0 + t\boldsymbol{\delta})\,\delta_i$. Da jede partielle Ableitung in dieser Summe selbst differenzierbar ist, liefert derselbe Schritt die zweite Ableitung: $\varphi''(t) = \sum_{i,j} \delta_j\frac{\partial^2 f}{\partial x_j\partial x_i}(\mathbf{x}_0 + t\boldsymbol{\delta})\,\delta_i = \boldsymbol{\delta}^\top \mathbf{H}_f(\mathbf{x}_0 + t\boldsymbol{\delta})\,\boldsymbol{\delta}$.

<!--en:ef94d952-->
Wenden wir nun den eindimensionalen Satz von Taylor mit Lagrange-Restglied (:numref:`subsec_mdl-taylor-series`) auf das Taylorpolynom ersten Grades von $\varphi$ zwischen $0$ und $1$ an. Es gibt ein $\tau \in (0, 1)$ mit $\varphi(1) = \varphi(0) + \varphi'(0) + \tfrac12\varphi''(\tau)$. Für die ursprüngliche Funktion $f$ ergibt das

<!--en:a0630ce3-->
Diese Gleichung ist exakt, doch müssen wir dafür die Hesse-Matrix an einem unbekannten Zwischenpunkt $\boldsymbol{\xi}$ auf der Strecke auswerten statt am Ausgangspunkt $\mathbf{x}_0$. Da die zweiten partiellen Ableitungen stetig sind, ändert sich der quadratische Term höchstens um den folgenden Betrag, wenn wir $\mathbf{H}_f(\boldsymbol{\xi})$ durch $\mathbf{H}_f(\mathbf{x}_0)$ ersetzen: $\tfrac12\|\mathbf{H}_f(\boldsymbol{\xi}) - \mathbf{H}_f(\mathbf{x}_0)\|\,\|\boldsymbol{\delta}\|^2 = o(\|\boldsymbol{\delta}\|^2)$. So erhalten wir die *Taylor-Näherung zweiter Ordnung* (engl. second-order Taylor approximation):

<!--en:e7499c49-->
Der Fehler dieses quadratischen Modells gegenüber $f$ geht schneller gegen null als das Quadrat der Schrittweite. Es ist das einzige quadratische Polynom mit dieser Eigenschaft. Hat $f$ drei stetige Ableitungen, ist der Fehler von dritter Ordnung in der Schrittweite.

<!--en:90c65b03-->
Entlang einer beliebigen Richtung $\mathbf{v}$ ergibt ein Schritt $t\mathbf{v}$ den quadratischen Term $\tfrac{t^2}{2}\mathbf{v}^\top\mathbf{H}\mathbf{v}$. Die Zahl $\mathbf{v}^\top\mathbf{H}\mathbf{v} = \tfrac{d^2}{dt^2}f(\mathbf{x}_0 + t\mathbf{v})\big|_{t=0}$ heißt *zweite Richtungsableitung*. Sie misst die Krümmung des Schnitts durch den Graphen von $f$ in Richtung $\mathbf{v}$.

<!--en:7e6611aa-->
Nun können wir die Abweichung des linearen Modells in unserer ersten Codezelle erklären. Für den Schritt $\boldsymbol{\epsilon} = (0.01, -0.03)$ vom Punkt $(0, \log 2)$ aus sagt die Hesse-Matrix :eqref:`eq_mdl-lse-hessian` eine Korrektur zweiter Ordnung von $\tfrac12\boldsymbol{\epsilon}^\top\mathbf{H}\boldsymbol{\epsilon} = \tfrac19(\epsilon_1 - \epsilon_2)^2 = \tfrac19 (0.04)^2 \approx 1.78 \times 10^{-4}$ voraus. Das kommt der gemessenen Abweichung von $1.786 \times 10^{-4}$ sehr nahe. Prüfen wir die Eigenwerte von $\mathbf{H}$ und die vorhergesagten Abweichungen im Code.

<!--en:8acdc02d-->
Sowohl die Matrix als auch ihre Eigenwerte $0$ und $\tfrac49$ stimmen mit :eqref:`eq_mdl-lse-hessian` überein. Die vorhergesagten Abweichungen $1.7778 \times 10^{-4}$ und $1.7778 \times 10^{-6}$ stimmen auf zwei bis drei signifikante Ziffern mit den Messwerten $1.786 \times 10^{-4}$ und $1.779 \times 10^{-6}$ überein. Die verbleibende kleine Differenz von etwa $10^{-6}$ bei der ersten Abweichung stammt vom Term dritter Ordnung. Verkleinern wir den Schritt auf ein Zehntel, sinkt die Differenz ungefähr auf ein Tausendstel und ist in den ausgegebenen Stellen nicht mehr sichtbar.

<!--en:bb925f99-->
In unserem Log-Sum-Exp-Beispiel war eine Richtung völlig flach. Betrachten wir nun eine Funktion, die sich in beiden Richtungen krümmt: $\psi(x, y) = xe^{-x^2-y^2}$ am Entwicklungspunkt $\mathbf{x}_0 = (-1, 0)$. Aus den zuvor berechneten partiellen Ableitungen und dem Funktionswert erhalten wir $\psi(-1, 0) = -e^{-1}$ und $\nabla \psi(-1, 0) = (-e^{-1}, 0)$. Noch einmal ableiten ergibt $\mathbf{H}_\psi(-1, 0) = 2e^{-1}\mathbf{I}$. Setzen wir diese Größen in :eqref:`eq_mdl-second_taylor` ein, erhalten wir das quadratische Modell $q(x, y) = e^{-1}\bigl(-1 - (x+1) + (x+1)^2 + y^2\bigr)$. Wie :numref:`fig_mdl-taylor-quadratic` zeigt, sind die Fläche und ihre quadratische Näherung in der Nähe von $(-1, 0)$ kaum zu unterscheiden. Erst weiter entfernt treten die Unterschiede hervor.

<!--en:6b6b4742-->
![Die Fläche $z = xe^{-x^2-y^2}$ (blau) und ihr Taylorpolynom zweiter Ordnung am Entwicklungspunkt $(-1, 0)$ (orange). Dort stimmen Funktionswert, Steigung und Krümmung überein. In der Nähe des Entwicklungspunkts sind die Flächen daher kaum zu unterscheiden, weiter außen entfernen sie sich voneinander.](../img/mdl-cal-taylor-quadratic.svg)

<!--en:c8a7b475-->
Ist der Fehler dieses quadratischen Modells tatsächlich von dritter Ordnung, sollte er bei doppelter Schrittweite etwa achtmal so groß sein. Zur Prüfung wertet die nächste Zelle $\psi$ und $q$ für Schritte der Weite $d$ in jeder Koordinate vom Punkt $(-1, 0)$ aus.

<!--en:db07343c-->
Wie erwartet verschwindet die Abweichung am Entwicklungspunkt. Verdoppeln wir die Schrittweite von $0.05$ auf $0.1$, wächst die Abweichung fast auf das Achtfache (von $0.000119$ auf $0.000921$). Das entspricht einem Restterm dritter Ordnung. Passen wir in jedem Schritt dieses lokale quadratische Modell an und springen direkt zu *seiner* Minimalstelle, erhalten wir das Newton-Verfahren. In :numref:`subsec_mdl-newton` haben wir es für eine Variable eingeführt, in :numref:`subsec_gd-newton` entwickeln wir es als Optimierungsverfahren für mehrere Variablen.

<!--en:9c91f467-->
### Das Kriterium der zweiten Ableitung

<!--en:dc9e1acd-->
An einem kritischen Punkt $\mathbf{x}_0$ verschwindet der lineare Term in :eqref:`eq_mdl-second_taylor`. In der Nähe von $\mathbf{x}_0$ beschreibt der quadratische Term die Änderung der Funktion daher bis auf einen Fehler der Ordnung $o(\|\mathbf{x} - \mathbf{x}_0\|^2)$. Ob $f$ steigt oder fällt, wenn wir uns in Richtung $\mathbf{v}$ entfernen, hängt also vom Vorzeichen von $\mathbf{v}^\top\mathbf{H}\mathbf{v}$ ab, sofern diese Krümmung von null verschieden ist.

<!--en:c8fc0703-->
Die Eigenwerte von $\mathbf{H}$ geben Aufschluss über diese Vorzeichen. Da $\mathbf{H}$ symmetrisch ist, garantiert der Spektralsatz (:numref:`sec_mdl-eigendecompositions`) eine Orthonormalbasis aus Eigenvektoren $\mathbf{q}_i$ mit reellen Eigenwerten $\lambda_i$. In dieser Basis ist die quadratische Form eine gewichtete Summe von Quadraten:

<!--en:17c69ad4-->
An dieser Formel können wir die lokale Geometrie direkt ablesen. Gilt für jeden Eigenwert $\lambda_i > 0$, so ist die Krümmung in jeder Richtung strikt positiv und beträgt mindestens $\lambda_{\min}\|\mathbf{v}\|^2$. Haben die Eigenwerte verschiedene Vorzeichen, krümmt sich die Fläche entlang einiger Eigenvektoren nach oben und entlang anderer nach unten.

<!--en:d90e5c7f-->
**Proposition (Bedingungen zweiter Ordnung).** *Sei $f$ in einer Umgebung eines kritischen Punktes $\mathbf{x}_0$ zweimal stetig differenzierbar. Wir schreiben $\mathbf{H} = \mathbf{H}_f(\mathbf{x}_0)$.*

<!--en:e2a9cbcb-->
1. *Ist $\mathbf{x}_0$ eine lokale Minimalstelle, so gilt $\mathbf{H} \succeq 0$ (alle Eigenwerte sind $\ge 0$).*

<!--en:cb1fc083-->
2. *Gilt $\mathbf{H} \succ 0$ (alle Eigenwerte sind $> 0$), so ist $\mathbf{x}_0$ eine strikte lokale Minimalstelle.*

<!--en:6bd2b7a4-->
3. *Hat $\mathbf{H}$ Eigenwerte beider Vorzeichen, so ist $\mathbf{x}_0$ weder eine lokale Minimalstelle noch eine lokale Maximalstelle, sondern ein Sattelpunkt. Das gilt auch dann, wenn zusätzlich der Eigenwert null auftritt.*

<!--en:c7980fe2-->
*Die entsprechenden Aussagen für Maxima erhalten wir, indem wir den Satz auf $-f$ anwenden. Ist $\mathbf{H}$ singulär und semidefinit, so trifft weder Fall 2 noch Fall 3 zu. Das Kriterium der zweiten Ableitung erlaubt dann keine Entscheidung.*

<!--en:e74fe08c-->
**Beweis.** (1) Nehmen wir im Gegenteil an, dass $\mathbf{v}^\top\mathbf{H}\mathbf{v} < 0$ für eine Richtung $\mathbf{v}$ gilt. Für den eindimensionalen Schnitt $\varphi(t) = f(\mathbf{x}_0 + t\mathbf{v})$ gilt $\varphi'(0) = \nabla f(\mathbf{x}_0)\cdot\mathbf{v} = 0$ und $\varphi''(0) = \mathbf{v}^\top\mathbf{H}\mathbf{v} < 0$. Nach dem Kriterium der zweiten Ableitung für eine Variable hat $\varphi$ bei $t = 0$ ein striktes lokales Maximum. Also nimmt $f$ Werte strikt unter $f(\mathbf{x}_0)$ beliebig nahe bei $\mathbf{x}_0$ an. Das widerspricht der Annahme, dass $\mathbf{x}_0$ eine lokale Minimalstelle ist.

<!--en:268cc3ef-->
(2) Dieser Schritt verlangt besondere Sorgfalt, denn wir müssen den Restterm für alle Richtungen zugleich gleichmäßig abschätzen. Wegen $\nabla f(\mathbf{x}_0) = \mathbf{0}$ liefert die exakte Taylorformel die Gleichung $f(\mathbf{x}_0 + \boldsymbol{\delta}) - f(\mathbf{x}_0) = \tfrac12 \boldsymbol{\delta}^\top\mathbf{H}_f(\boldsymbol{\xi})\boldsymbol{\delta}$ für ein $\boldsymbol{\xi}$ auf der Strecke von $\mathbf{x}_0$ nach $\mathbf{x}_0 + \boldsymbol{\delta}$. Zerlegen wir die Hesse-Matrix gemäß $\mathbf{H}_f(\boldsymbol{\xi}) = \mathbf{H} + \bigl(\mathbf{H}_f(\boldsymbol{\xi}) - \mathbf{H}\bigr)$. Nach :eqref:`eq_mdl-quadform-eig` gilt für den führenden Term $\boldsymbol{\delta}^\top\mathbf{H}\boldsymbol{\delta} \ge \lambda_{\min}\|\boldsymbol{\delta}\|^2$. Aus der Stetigkeit der zweiten partiellen Ableitungen folgt $\|\mathbf{H}_f(\boldsymbol{\xi}) - \mathbf{H}\| \le \tfrac12\lambda_{\min}$ für alle hinreichend kleinen $\boldsymbol{\delta}$. Zusammengenommen ergeben die beiden Schranken $f(\mathbf{x}_0 + \boldsymbol{\delta}) - f(\mathbf{x}_0) \ge \tfrac12\bigl(\lambda_{\min} - \tfrac12\lambda_{\min}\bigr)\|\boldsymbol{\delta}\|^2 = \tfrac14\lambda_{\min}\|\boldsymbol{\delta}\|^2 > 0$ für alle hinreichend kleinen $\boldsymbol{\delta} \neq \mathbf{0}$.

<!--en:982c362b-->
(3) Entlang eines normierten Eigenvektors zum Eigenwert $\lambda > 0$ erfüllt der Schnitt $\varphi$ die Bedingungen $\varphi'(0) = 0$ und $\varphi''(0) = \lambda > 0$. Daher nimmt $f$ Werte strikt über $f(\mathbf{x}_0)$ beliebig nahe bei $\mathbf{x}_0$ an. Entlang eines Eigenvektors zum Eigenwert $\lambda < 0$ nimmt die Funktion strikt kleinere Werte an. $\blacksquare$

<!--en:38887e85-->
Von diesen drei Aussagen sollten Sie sich vor allem die zweite merken. Nur ihr Beweis erfordert etwas mehr Arbeit. Auch der Fall, in dem das Kriterium keine Entscheidung erlaubt, ist praktisch bedeutsam: Entlang eines Eigenvektors zum Eigenwert null ist das quadratische Modell völlig flach. Dort entscheiden die Terme höherer Ordnung. So haben $x^2 + y^4$ und $x^2 - y^4$ im Ursprung dieselbe Hesse-Matrix $\operatorname{diag}(2, 0)$. Die erste Funktion hat dort dennoch ein striktes Minimum, die zweite einen Sattelpunkt (Aufgabe :exref:`ex_mdl-multivariable-calculus-flat` führt dieses Phänomen noch weiter). :numref:`fig_mdl-cal-second-derivative-test` veranschaulicht alle vier Fälle anhand der quadratischen Modelle unserer Beispiele.

<!--en:c2d40d5c-->
![Niveaulinien des quadratischen Modells $\tfrac12\mathbf{v}^\top\mathbf{H}\mathbf{v}$ (grau), Richtungen seiner Eigenvektoren mit den zugehörigen Eigenwerten (gestrichelt) und Abstiegspfeile $-\mathbf{H}\mathbf{v}$ (rot). (a) $\mathbf{H}\succ 0$: ein Minimum, alle Abstiegspfeile zeigen nach innen. (b) $\mathbf{H}\prec 0$: ein Maximum. (c) Indefinit: ein Sattelpunkt. Der Abstieg verläuft entlang eines Eigenvektors nach innen, entlang des anderen nach außen. (d) Singulär und semidefinit: Entlang des Eigenvektors zum Eigenwert null ist das quadratische Modell flach. Die zweite Ordnung erlaubt keine Entscheidung.](../img/mdl-cal-second-derivative-test.svg)

<!--en:70d2d1a3-->
In :numref:`fig_mdl-cal-second-derivative-test` zeigt Teil (a) die Hesse-Matrix von $\psi$ am Punkt $(-1/\sqrt{2}, 0)$. Sie ist gleich $e^{-1/2}\operatorname{diag}(2\sqrt{2}, \sqrt{2})$. Teil (b) zeigt am Punkt $(+1/\sqrt{2}, 0)$ die Matrix mit umgekehrtem Vorzeichen. Teil (c) zeigt die Situation im Ursprung für $x^4 + y^4 - 4xy$ mit der Hesse-Matrix $\left(\begin{smallmatrix} 0 & -4 \\ -4 & 0 \end{smallmatrix}\right)$. Teil (d) verwendet die Hesse-Matrix von Log-Sum-Exp aus :eqref:`eq_mdl-lse-hessian` nur als Beispielmatrix. Log-Sum-Exp selbst hat keinen kritischen Punkt, da sein Gradient, die Softmax-Funktion, nirgends verschwindet.

<!--en:668d5c21-->
Was passiert, wenn wir das Kriterium der zweiten Ableitung numerisch umsetzen und auf eine flache Richtung anwenden? Eine aus finiten Differenzen berechnete Hesse-Matrix liefert fast nie einen Eigenwert von exakt null. Wir müssen daher eine Toleranz wählen, unterhalb derer wir den Betrag eines Eigenwerts als null behandeln. In der nächsten Zelle ist die zentrierte gemischte Differenz `fxy` die zweite Differenz aus :eqref:`eq_mdl-second-difference` mit Schrittweite $2h$ und Ausgangspunkt $(x-h, y-h)$.

<!--en:b10167e7-->
Am Punkt $(-1/\sqrt{2}, 0)$ erhalten wir die Eigenwerte $0.858$ und $1.716$ (entsprechend $\sqrt{2}e^{-1/2}$ und $2\sqrt{2}e^{-1/2}$). Sie bestätigen das zuvor gefundene globale Minimum. Am Punkt $(+1/\sqrt{2}, 0)$ wechseln beide ihr Vorzeichen und bestätigen damit ein Maximum. Im Ursprung hat die Funktion $x^4 + y^4 - 4xy$ die Eigenwerte $\pm 4$. Es liegt also ein Sattelpunkt vor, dessen Anstiegs- und Abstiegsrichtungen $(1, -1)$ und $(1, 1)$ gegenüber den Koordinatenachsen gedreht sind. Besonders aufschlussreich ist der vierte Punkt. Im Ursprung von $x^2 + y^4$ liefern finite Differenzen entlang der flachen Richtung den Wert $2h^2 = 2 \times 10^{-8}$. Er liegt unterhalb unserer Toleranz. Das Programm meldet daher zu Recht „keine Entscheidung“, obwohl der Ursprung eine strikte Minimalstelle ist. Die Information zweiter Ordnung kann den Anstieg vierter Ordnung nicht erfassen. Nur die Toleranz verhindert, dass das numerische Kriterium das Gegenteil nahelegt.

<!--en:f99c801e-->
### Sattelpunkte in hohen Dimensionen

<!--en:fd6c0cb3-->
In $n$ Dimensionen müssen an einer lokalen Minimalstelle alle $n$ Eigenwerte der Hesse-Matrix nichtnegativ sein. An einer nicht entarteten Minimalstelle müssen alle $n$ strikt positiv sein. Verhielten sich die Vorzeichen der Eigenwerte an einem zufälligen kritischen Punkt wie unabhängige Münzwürfe, so wären alle $n$ mit Wahrscheinlichkeit $2^{-n}$ positiv. Schon das ist für große $n$ verschwindend wenig.

<!--en:f6774257-->
Überraschenderweise *überschätzt* diese Münzwurfheuristik die Wahrscheinlichkeit, ein Minimum zu finden. Betrachten wir zufällige symmetrische Matrizen aus dem gaußschen orthogonalen Ensemble. Ihre Diagonaleinträge haben die doppelte Varianz der Einträge außerhalb der Diagonalen. Bereits für $n = 2$ beträgt die exakte Wahrscheinlichkeit, dass beide Eigenwerte positiv sind, nur $(1 - 1/\sqrt{2})/2 \approx 0.146$ statt $0.25$ (Aufgabe :exref:`ex_mdl-multivariable-calculus-random-hessians`). Für große $n$ sinkt die Wahrscheinlichkeit wesentlich schneller als exponentiell, nämlich wie $\exp\bigl(-\tfrac{\ln 3}{4}n^2\bigr)$ :cite:`Dean.Majumdar.2006`. Die Eigenwerte einer zufälligen symmetrischen Matrix stoßen einander ab und liegen fast nie alle auf derselben Seite von null.

<!--en:f3f27090-->
In zufälligen hochdimensionalen Landschaften sind kritische Punkte bei typischen (hohen) Funktionswerten also fast ausschließlich Sattelpunkte. Erst bei niedrigen Funktionswerten verschiebt sich das Spektrum der Hesse-Matrix zu positiven Eigenwerten hin, sodass Minima überwiegen :cite:`Bray.Dean.2007`. Neuronale Netze zeigen dasselbe Verhalten: Bestimmt man mit dem Newton-Verfahren kritische Punkte auf den Verlustflächen kleiner Netze, so sind die kritischen Punkte mit hohem Verlust Sattelpunkte :cite:`Dauphin.Pascanu.Gulcehre.ea.2014`. Kommt ein Optimierer bei hohem Verlust kaum voran, ist ein ungünstiges lokales Minimum in der Praxis die unwahrscheinlichste Erklärung. Möglicherweise verweilt der Optimierer in der Nähe eines Sattelpunkts. Noch häufiger durchquert er ein schlecht konditioniertes Gebiet, in dem der Gradient keineswegs klein ist.

<!--en:74573f56-->
## Optimierung unter Nebenbedingungen und Lagrange-Multiplikatoren

<!--en:92f59f8e-->
Bisher konnten sich unsere Variablen in jede beliebige Richtung bewegen. Viele Probleme des maschinellen Lernens beschränken die Parameter dagegen auf eine Fläche. Beispiele sind Gewichtsvektoren der Länge eins, Wahrscheinlichkeiten mit Summe eins oder die Suche nach dem Punkt einer Kurve mit dem kürzesten Abstand zum Ursprung. In allen diesen Fällen wollen wir $f(\mathbf{x})$ unter einer Nebenbedingung $g(\mathbf{x}) = c$ minimieren. Die Bedingung $\nabla f = \mathbf{0}$ für das Problem ohne Nebenbedingung gilt dann nicht mehr. Am besten zulässigen Punkt muss $\nabla f$ nicht verschwinden. Es genügt, dass kein Schritt auf der zulässigen Fläche eine Abnahme in erster Ordnung bewirkt. Was an die Stelle von $\nabla f = \mathbf{0}$ tritt, folgt unmittelbar aus der Geometrie der Niveaumengen, ganz ohne zweite Ableitungen.

<!--en:20b5305b-->
Seien $f$ und $g$ differenzierbar. An einem zulässigen Punkt entsprechen die zulässigen infinitesimalen Schritte den Richtungen $\mathbf{v}$, die tangential zur Fläche $\{g = c\}$ verlaufen. An einer lokalen Minimalstelle $\mathbf{x}^\star$ unter der Nebenbedingung kann kein zulässiger Schritt $f$ in erster Ordnung verringern. Es gilt also $\mathbf{v}\cdot\nabla f(\mathbf{x}^\star) \ge 0$ für jede Tangentialrichtung $\mathbf{v}$. Da auch $-\mathbf{v}$ eine Tangentialrichtung ist, muss Gleichheit gelten: $\nabla f(\mathbf{x}^\star)$ steht senkrecht auf jeder Tangentialrichtung. Nach unserem Satz über Niveaumengen steht auch $\nabla g(\mathbf{x}^\star)$ senkrecht auf jeder Tangentialrichtung. Die beiden Gradienten müssen also unter der folgenden Regularitätsvoraussetzung parallel sein.

<!--en:af1fec9d-->
**Proposition (Lagrange-Bedingung).** *Sei $\mathbf{x}^\star$ eine lokale Minimalstelle von $f$ auf $\{\mathbf{x} : g(\mathbf{x}) = c\}$. Die Funktion $f$ sei an der Stelle $\mathbf{x}^\star$ differenzierbar, $g$ sei in einer Umgebung von $\mathbf{x}^\star$ stetig differenzierbar und es gelte $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$. Dann gibt es einen Skalar $\mu$, den sogenannten Lagrange-Multiplikator, für den gilt:*

<!--en:f5ce028e-->
**Beweis.** Der Beweis beruht auf einer geometrischen Eigenschaft der zulässigen Menge. Unter den letzten beiden Voraussetzungen ist $\{g = c\}$ in der Nähe von $\mathbf{x}^\star$ eine glatte Fläche. Jeder Vektor $\mathbf{v}$ mit $\nabla g(\mathbf{x}^\star)\cdot\mathbf{v} = 0$ ist der Geschwindigkeitsvektor $\boldsymbol{\gamma}'(0)$ einer differenzierbaren Kurve $\boldsymbol{\gamma}$, die in $\{g = c\}$ verläuft und $\boldsymbol{\gamma}(0) = \mathbf{x}^\star$ erfüllt :cite:`Rudin.1976`. Entlang einer solchen Kurve hat die Funktion einer Variablen $t \mapsto f(\boldsymbol{\gamma}(t))$ bei $t = 0$ ein lokales Minimum. Ihre Ableitung muss dort also verschwinden. Wie im Beweis für Niveaumengen ist diese Ableitung gleich $\mathbf{v}\cdot\nabla f(\mathbf{x}^\star)$. Folglich steht $\nabla f(\mathbf{x}^\star)$ senkrecht auf der gesamten Hyperebene aller Vektoren $\mathbf{v}$, die orthogonal zu $\nabla g(\mathbf{x}^\star)$ sind. Nur die skalaren Vielfachen von $\nabla g(\mathbf{x}^\star)$ stehen senkrecht auf dieser Hyperebene. Also gilt $\nabla f(\mathbf{x}^\star) = \mu\nabla g(\mathbf{x}^\star)$. $\blacksquare$

<!--en:2568b0a5-->
Geometrisch sind an einer Optimalstelle unter der Nebenbedingung die Gradienten von $f$ und $g$ parallel. Die Niveaumenge von $f$ durch $\mathbf{x}^\star$ berührt also die zulässige Fläche tangential. Algebraisch bilden die $n$ Gleichungen aus :eqref:`eq_mdl-lagrange-condition` zusammen mit der Nebenbedingung $g(\mathbf{x}) = c$ ein System von $n + 1$ Gleichungen in den $n + 1$ Unbekannten, den Komponenten von $\mathbf{x}$ und dem Multiplikator $\mu$.

<!--en:6f721060-->
Bestimmen wir als Beispiel die Punkte der Hyperbel $xy = 1$ mit dem kürzesten Abstand zum Ursprung. Wir wollen $f(x, y) = x^2 + y^2$ unter der Nebenbedingung $g(x, y) = xy = 1$ minimieren. Die Gradienten lauten $\nabla f = (2x, 2y)$ und $\nabla g = (y, x)$, und überall auf der Hyperbel gilt $\nabla g \neq \mathbf{0}$. Die Lagrange-Bedingung liefert $2x = \mu y$ und $2y = \mu x$. Setzen wir die zweite Gleichung in die erste ein, erhalten wir $4x = \mu^2 x$ oder gleichbedeutend $x(4 - \mu^2) = 0$. Da $x = 0$ auf der Hyperbel $xy = 1$ unmöglich ist, muss $\mu = \pm 2$ gelten. Die Wahl $\mu = -2$ erzwingt $y = -x$ und damit $xy = -x^2 = 1$, was keine reelle Lösung hat. Es bleiben $\mu = 2$ und $x = y$. Daraus ergeben sich die beiden Kandidaten $(1, 1)$ und $(-1, -1)$, jeweils mit $f = 2$. Da $f = x^2 + 1/x^2$ für $x \to 0$ oder $|x| \to \infty$ auf jedem Ast unbegrenzt wächst, muss ein globales Minimum existieren. Diese beiden Punkte liegen dem Ursprung also tatsächlich am nächsten. Im Punkt $(1, 1)$ berührt der Kreis $x^2 + y^2 = 2$ die Hyperbel tangential, und es gilt $\nabla f = (2, 2) = 2\,\nabla g$ (:numref:`fig_mdl-lagrange-tangency`).

<!--en:4fce73a6-->
![Der dem Ursprung nächstgelegene Punkt auf $xy = 1$ im dargestellten Ausschnitt. Im Punkt $(1, 1)$ berührt der Kreis $x^2+y^2=2$ die Hyperbel tangential. Es gilt $\nabla f = (2,2) = 2\nabla g$ und damit $\mu = 2$. Die spiegelbildliche Lösung $(-1,-1)$ liegt außerhalb des Bildes. An einem zulässigen, aber nicht optimalen Punkt zeigen die beiden Gradienten in verschiedene Richtungen, und $\nabla f$ hat eine Komponente tangential zur Hyperbel.](../img/mdl-cal-lagrange-tangency.svg)

<!--en:a8a6a271-->
An der Hyperbel lässt sich die Geometrie leicht zeichnen. Dasselbe Verfahren führt uns auch direkt zu unserem durchgehenden Beispiel zurück. Maximieren wir $\mathbf{z}\cdot\mathbf{p} - \sum_i p_i\log p_i$ über alle Wahrscheinlichkeitsvektoren $\mathbf{p}$ unter der Nebenbedingung $\sum_i p_i = 1$. Ausgeschrieben ergibt die Lagrange-Bedingung $z_i - \log p_i - 1 = \mu$ für jede Koordinate $i$, woraus $p_i \propto e^{z_i}$ folgt. Die optimale Verteilung ergibt sich also aus der Softmax-Funktion, und der erreichte Maximalwert ist Log-Sum-Exp (:numref:`sec_mdl-lagrangian-duality`).

<!--en:d64b21bc-->
Auf die Regularitätsvoraussetzung $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$ können wir nicht verzichten. Ohne sie existiert möglicherweise überhaupt kein Lagrange-Multiplikator (Aufgabe :exref:`ex_mdl-multivariable-calculus-rayleigh` liefert ein einfaches Gegenbeispiel). Später erweitert :numref:`sec_mdl-constrained-optimization-duality` diese Bedingung mit den KKT-Bedingungen auf Ungleichungsnebenbedingungen, und :numref:`sec_mdl-lagrangian-duality` leitet das duale Problem her.

<!--en:f272a6fc-->
Bei allen Ergebnissen in diesem Abschnitt haben wir den Gradienten als bekannt vorausgesetzt. Für eine Formel mit zwei Variablen an der Tafel lässt er sich mühelos von Hand berechnen. Bei einem neuronalen Netz mit einer Million Gewichten liegt die Schwierigkeit gerade darin, den Gradienten effizient zu berechnen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:c5a25a9a-->
Die Änderung einer differenzierbaren Funktion ist in erster Ordnung gleich dem Skalarprodukt des Schritts mit ihrem Gradienten. Die bloße Existenz partieller Ableitungen garantiert dies nicht. Geometrisch zeigt der negative Gradient in die Richtung des steilsten Abstiegs, und der Gradient steht senkrecht auf jeder Niveaumenge. Eine Ordnung höher bestimmt die symmetrische Hesse-Matrix die lokale quadratische Näherung. An einem kritischen Punkt lassen sich anhand der Vorzeichen ihrer Eigenwerte Minima, Maxima und Sattelpunkte unterscheiden. Bei einer singulären semidefiniten Hesse-Matrix erlaubt das Kriterium keine Entscheidung. Beschränken wir die Variablen auf die zulässige Fläche $g(\mathbf{x}) = c$, so tritt aufgrund der Geometrie der Niveaumengen an die Stelle von $\nabla f = \mathbf{0}$ die Lagrange-Bedingung $\nabla f = \mu\nabla g$.

<!--en:2294243a-->
## Übungen

<!--en:9541909e-->
1. **Richtungsableitungen ohne Gradienten.** Sei $f(x, y) = x^2 y / (x^2 + y^2)$ für $(x, y) \neq (0, 0)$ und $f(0, 0) = 0$.
    1. Zeigen Sie, dass $f$ im Ursprung stetig ist.
    1. Berechnen Sie die beiden partiellen Ableitungen von $f$ im Ursprung sowie dort die Richtungsableitung entlang jedes Einheitsvektors $\mathbf{u} = [\cos\theta, \sin\theta]^\top$.
    1. Zeigen Sie, dass es keinen Vektor $\mathbf{g}$ gibt, mit dem sich diese Richtungsableitungen als $\mathbf{u}\cdot\mathbf{g}$ schreiben lassen, sodass $f$ im Ursprung nicht differenzierbar ist. Welche Voraussetzung der hinreichenden Bedingung für Differenzierbarkeit in :numref:`subsec_mdl-gradient` ist verletzt? Was zeigt diese Funktion, was das dortige Gegenbeispiel $2xy/(x^2+y^2)$ nicht zeigt?
   :exlabel:`ex_mdl-multivariable-calculus-no-gradient`

<!--en:472b9572-->
1. **Gradienten einer lang gestreckten Schale.** Sei $f(x, y) = x^2 + \kappa y^2$ mit $\kappa \ge 1$. Die Minimalstelle ist der Ursprung. Der Parameter $\kappa$ misst, wie viel stärker sich $f$ entlang der $y$-Achse krümmt als entlang der $x$-Achse.
    1. Bestimmen Sie eine Tangentialrichtung zur Niveaulinie von $f$ durch $(1, 1)$ und prüfen Sie, dass sie senkrecht auf $\nabla f(1, 1)$ steht. Berechnen Sie den Winkel zwischen $-\nabla f(1, 1)$ und der Richtung $-[1, 1]^\top$ von $(1, 1)$ zur Minimalstelle. An welchen Punkten $(x, y) \neq (0, 0)$ zeigt der negative Gradient zur Minimalstelle?
    1. Bestimmen Sie den kleinsten Wert des Kosinus dieses Winkels über alle Punkte $(x, y) \neq (0, 0)$ sowie die Punkte, an denen er angenommen wird. (Hinweis: Setzen Sie $\rho = y/x$ und minimieren Sie über $\rho$.) Berechnen Sie den Wert für $\kappa = 10$. Was folgt daraus für einen Schritt des Gradientenabstiegs auf einer solchen Verlustfläche, wenn $\kappa$ wächst?

<!--en:01af43f9-->
1. **Flache Richtungen.**
    1. Geben Sie drei Funktionen von $(x, y)$ mit einem kritischen Punkt im Ursprung und der dortigen Hesse-Matrix $\operatorname{diag}(2, 0)$ an: eine mit einem strikten lokalen Minimum, eine mit einem lokalen, aber nicht strikten Minimum und eine ohne lokales Minimum an dieser Stelle.
    1. Auch die Funktion $f(x, y) = (x - y^2)(x - 2y^2)$ hat im Ursprung die Hesse-Matrix $\operatorname{diag}(2, 0)$. Zeigen Sie, dass ihre Einschränkung auf jede Gerade durch den Ursprung dort ein striktes lokales Minimum hat. Skizzieren Sie das Gebiet der Ebene mit $f < 0$ und zeigen Sie, dass der Ursprung dennoch keine lokale Minimalstelle von $f$ ist. Was folgt daraus für die Prüfung eines kritischen Punktes, wenn man den Verlust nur in einigen wenigen Richtungen untersucht?
    1. Der *Affensattel* $f(x, y) = x^3 - 3xy^2$ hat einen kritischen Punkt im Ursprung. Berechnen Sie dort seine Hesse-Matrix, schreiben Sie $f$ in Polarkoordinaten und bestimmen Sie die Richtungen, in denen $f$ abnimmt. Wie hängt $\|\nabla f\|$ im Vergleich zum Sattel $x^2 - y^2$ vom Abstand zum Ursprung ab? Was folgt daraus für den Gradientenabstieg, wenn er in der Nähe des jeweiligen Sattelpunkts startet?
   :exlabel:`ex_mdl-multivariable-calculus-flat`

<!--en:31a5798a-->
1. [code] **Zufällige Hesse-Matrizen.** ● An einer lokalen Minimalstelle müssen alle $n$ Eigenwerte der Hesse-Matrix nichtnegativ sein, an einer nicht entarteten sogar alle positiv. Für den zweiten Fall ergibt die Münzwurfheuristik aus diesem Abschnitt die Wahrscheinlichkeit $2^{-n}$. Prüfen Sie diese Heuristik an zufälligen symmetrischen Matrizen $\mathbf{A} = (\mathbf{G} + \mathbf{G}^\top)/\sqrt{2}$, wobei die $n \times n$-Matrix $\mathbf{G}$ unabhängige standardnormalverteilte Einträge hat.
    1. Zeigen Sie für $n = 2$, dass $s = (a_{11} + a_{22})/2$, $d = (a_{11} - a_{22})/2$ und $a_{12}$ unabhängige standardnormalverteilte Zufallsvariablen sind und dass $\mathbf{A} \succ 0$ genau dann gilt, wenn $s > \sqrt{d^2 + a_{12}^2}$. Nutzen Sie die Rotationssymmetrie der dreidimensionalen Standardnormalverteilung, um $P(\mathbf{A} \succ 0)$ exakt zu berechnen, und vergleichen Sie das Ergebnis mit $2^{-2}$.
    1. Schätzen Sie $P(\mathbf{A} \succ 0)$ mit der Monte-Carlo-Methode für $n = 1, \ldots, 6$. Verwenden Sie $10^6$ Ziehungen für $n \le 5$ und $10^7$ Ziehungen in Batches für $n = 6$. Tragen Sie $\log P$ gegen $n$ auf, zeichnen Sie auch $\log 2^{-n}$ ein und passen Sie ein quadratisches Polynom in $n$ an Ihre Schätzwerte an. :citet:`Dean.Majumdar.2006` beweisen, dass $\log P \approx -\tfrac{\ln 3}{4} n^2$ für große $n$ gilt. Vergleichen Sie den dortigen Koeffizienten mit dem quadratischen Koeffizienten Ihres Polynoms. Überschätzt oder unterschätzt die Heuristik die Wahrscheinlichkeit eines Minimums, und welche Eigenschaft der Eigenwerte lässt sie außer Acht?
    1. Für große $n$ liegt der kleinste Eigenwert von $\mathbf{A}$ nahe bei $-2\sqrt{n}$. Bestimmen Sie für $n = 5, 10, 20$ die Verschiebung $\beta$, bei der die Hälfte der Matrizen $\mathbf{A} + \beta\mathbf{I}$ positiv definit ist, und vergleichen Sie sie mit $2\sqrt{n}$. In den zufälligen Landschaften von :citet:`Bray.Dean.2007` verhält sich die Hesse-Matrix an einem kritischen Punkt wie eine solche verschobene Matrix. Die Verschiebung wird umso größer, je kleiner der Wert der Zufallsfunktion am kritischen Punkt ist. Was legt Ihr Ergebnis für kritische Punkte mit niedrigem Verlust nahe?
   :exlabel:`ex_mdl-multivariable-calculus-random-hessians`

<!--en:c1f37814-->
1. [extended] **Sattelpunkte eines linearen Netzes.** ●● Sei $\mathbf{A}$ eine symmetrische $n \times n$-Matrix mit den Eigenwerten $\lambda_1 > \lambda_2 > \cdots > \lambda_n > 0$ und den orthonormalen Eigenvektoren $\mathbf{q}_1, \ldots, \mathbf{q}_n$. Ein lineares Netz mit einer versteckten Einheit, den Eingabegewichten $\mathbf{v}$ und den Ausgabegewichten $\mathbf{u}$ bildet $\mathbf{x}$ auf $\mathbf{u}\mathbf{v}^\top\mathbf{x}$ ab. Beim Training mit durch Whitening transformierten Eingaben ($\mathbb{E}[\mathbf{x}\mathbf{x}^\top] = \mathbf{I}$) und Zielwerten $\mathbf{A}\mathbf{x}$ beträgt die Hälfte des erwarteten quadratischen Fehlers

   $$\ell(\mathbf{u}, \mathbf{v}) = \tfrac12 \|\mathbf{A} - \mathbf{u}\mathbf{v}^\top\|_F^2.$$

   :citet:`Baldi.Hornik.1989` zeigten, dass die lokalen Minima solcher linearen Netze global sind. In dieser Aufgabe bestimmen Sie alle kritischen Punkte und vergleichen das Ergebnis mit dem Zusammenhang zwischen Verlust und Index, den :citet:`Dauphin.Pascanu.Gulcehre.ea.2014` für nichtlineare Netze gemessen haben. Der *Index* eines kritischen Punktes ist der Anteil negativer Eigenwerte seiner Hesse-Matrix.
    1. Berechnen Sie $\nabla_{\mathbf{u}} \ell$ und $\nabla_{\mathbf{v}} \ell$. Zeigen Sie, dass die kritischen Punkte der Ursprung und die Punkte $\mathbf{u} = \alpha\mathbf{q}_i$, $\mathbf{v} = (\lambda_i / \alpha)\mathbf{q}_i$ mit $\alpha \neq 0$ sind. Berechnen Sie $\ell$ an jedem dieser Punkte.
    1. Berechnen Sie die Eigenwerte der $2n \times 2n$-Hesse-Matrix am ausgeglichenen Punkt $\mathbf{u} = \mathbf{v} = \sqrt{\lambda_i}\,\mathbf{q}_i$ und im Ursprung. (In der Eigenvektorbasis zerfällt die Hesse-Matrix in $2 \times 2$-Blöcke, je einer für das Koordinatenpaar in Richtung $\mathbf{q}_j$ in $\mathbf{u}$ und in $\mathbf{v}$.) Wie viele Eigenwerte sind am jeweiligen kritischen Punkt negativ? Erklären Sie den Eigenwert null anhand einer Symmetrie von $\ell$ und folgern Sie, dass jedes lokale Minimum global ist.
    1. Wählen Sie $n = 4$, $(\lambda_1, \ldots, \lambda_4) = (4, 3, 2, 1)$ und eine zufällige Orthonormalbasis aus Eigenvektoren. Führen Sie das Newton-Verfahren von vielen zufälligen Startpunkten unterschiedlicher Größenordnung aus. Erfassen Sie für jeden erreichten kritischen Punkt den Verlust und den Index. An jedem kritischen Punkt außer dem Ursprung ist die Hesse-Matrix singulär. Berechnen Sie daher jeden Newton-Schritt mit der Methode der kleinsten Quadrate. Tragen Sie den Verlust wie in Abbildung 1 von :citet:`Dauphin.Pascanu.Gulcehre.ea.2014` gegen den Index auf. Wie oft endet das Newton-Verfahren an einem kritischen Punkt der jeweiligen Art, und wo endet der Gradientenabstieg bei kleinen zufälligen Anfangswerten?
    1. Ersetzen Sie $\mathbf{u}\mathbf{v}^\top$ durch $\mathbf{U}\mathbf{V}^\top$ mit zwei versteckten Einheiten, sodass $\mathbf{U}$ und $\mathbf{V}$ nun $4 \times 2$-Matrizen sind. Sagen Sie anhand der Analyse für Rang eins die Verlustwerte an den kritischen Punkten voraus und bestätigen Sie sie mit dem Newton-Verfahren. Wie viele Eigenwerte null hat die Hesse-Matrix an den kritischen Punkten, und welche Symmetrie erklärt sie? Ist der Index eine monotone Funktion des Verlusts?

<!--en:477fa8f6-->
1. **Lagrange-Multiplikatoren und Rayleigh-Quotient.** Sei $\mathbf{A}$ eine symmetrische $n \times n$-Matrix. Maximieren Sie $f(\mathbf{x}) =
   \mathbf{x}^\top\mathbf{A}\mathbf{x}$ unter der Nebenbedingung $g(\mathbf{x}) =
   \|\mathbf{x}\|^2 = 1$.
    1. Schreiben Sie die Lagrange-Bedingung auf und zeigen Sie, dass sie $\mathbf{A}\mathbf{x} = \mu\mathbf{x}$ lautet: Jeder kritische Punkt unter der Nebenbedingung ist ein normierter Eigenvektor, und sein Multiplikator ist der Eigenwert.
    1. Zeigen Sie, dass an jedem kritischen Punkt unter der Nebenbedingung $f(\mathbf{x}) = \mu$ gilt. Warum existiert unter der Nebenbedingung ein Maximum? Folgern Sie, dass das Maximum $\lambda_{\max}(\mathbf{A})$ und das Minimum $\lambda_{\min}(\mathbf{A})$ ist.
    1. Bestimmen Sie für $\mathbf{A} = \left(\begin{smallmatrix} 2 & 1 \\ 1 & 2 \end{smallmatrix}\right)$ die Maximal- und Minimalstellen sowie die zugehörigen Funktionswerte. Prüfen Sie an einer Maximalstelle, dass die Niveaumenge von $f$ durch diesen Punkt den Einheitskreis tangential berührt.
    1. Welche Voraussetzung des Satzes über die Lagrange-Bedingung erfüllt $g(\mathbf{x}) = \|\mathbf{x}\|^2$ an jedem Punkt der Einheitssphäre? Minimieren Sie nun die lineare Funktion $\mathbf{c}^\top\mathbf{x}$ mit $\mathbf{c} \neq \mathbf{0}$ unter der Nebenbedingung $\|\mathbf{x}\|^2 = 0$. Wo liegt die Minimalstelle, und warum gibt es keinen Multiplikator $\mu$, der dort $\nabla f = \mu\nabla g$ erfüllt?
   :exlabel:`ex_mdl-multivariable-calculus-rayleigh`

<!--en:a8c7d176-->
[Diskussion](https://d2l.discourse.group/t/413)

<!--en:1ac47d7c-->
[Diskussion](https://d2l.discourse.group/t/1090)

<!--en:f7019250-->
[Diskussion](https://d2l.discourse.group/t/1091)

<!--en:f7019250-2-->
[Diskussion](https://d2l.discourse.group/t/1091)

<!--en:f2a77a06-->
Analysis mehrerer Variablen<br>**partielle Ableitungen · Gradient · Richtungsableitungen · Niveaumengen · Hesse-Matrix · kritische Punkte · Lagrange-Multiplikatoren**

<!--en:deb05fe5-->
::: {.slide title="Gemeinsame Schritte für Millionen von Gewichten"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:f565375c-->
- **Vom Schnitt zum gemeinsamen Schritt:** Der Gradientenabstieg aktualisiert alle $N = 10^6$ Gewichte gleichzeitig um $\boldsymbol{\epsilon} \in \mathbb{R}^N$.

<!--en:6b5f5e37-->
- **Ein Vektor für die Steigungen erster Ordnung:** Der **Gradient** $\nabla_{\mathbf{w}}\ell$ ist ein Spaltenvektor. Über $\boldsymbol{\epsilon}\cdot\nabla_{\mathbf{w}}\ell$ sagt er die Wirkung jedes kleinen gemeinsamen Schritts voraus. Er zeigt in Richtung des steilsten Anstiegs.

<!--en:720887e0-->
- **Eine Matrix für die Flächenkrümmung:** Die symmetrische **Hesse-Matrix** $\mathbf{H}_\ell$ erfasst die Krümmung in jeder Richtung und unterscheidet Minima von Sattelpunkten.

<!--en:242317d8-->
::: {.slide title="Partielle Ableitungen am Beispiel von Log-Sum-Exp"}

<!--en:965259cb-->
[Schnitte entlang der Koordinatenachsen]{.kicker}

<!--en:57a9f6f1-->
- **Ableitung im Koordinatenschnitt:** Bleiben alle Koordinaten bis auf $w_i$ fest, ergibt sich die partielle Ableitung $\partial\ell/\partial w_i$.

<!--en:cf40847b-->
- **Log-Sum-Exp $f(x, y) = \log(e^x + e^y)$:** $\frac{\partial f}{\partial x} = \frac{e^x}{e^x+e^y}$ und $\frac{\partial f}{\partial y} = \frac{e^y}{e^x+e^y}$ bilden den Wahrscheinlichkeitsvektor $\mathbf{p} = \operatorname{softmax}(x, y)$.

<!--en:34153e63-->
- **Am Punkt $(0, \log 2)$:** $f = \log 3$ und $\nabla f = (\tfrac{1}{3}, \tfrac{2}{3})^\top$.

<!--en:619a6ab1-->
::: {.slide title="Differenzierbarkeit in mehreren Variablen und der Gradient"}

<!--en:2e3fe40a-->
[Lineares Modell erster Ordnung]{.kicker}

<!--en:e28c9c4b-->
- **Differenzierbarkeit in mehreren Variablen:** $\ell(\mathbf{w} + \boldsymbol{\epsilon}) = \ell(\mathbf{w}) + \boldsymbol{\epsilon}\cdot\nabla_{\mathbf{w}}\ell(\mathbf{w}) + o(\|\boldsymbol{\epsilon}\|)$.

<!--en:980d5a1e-->
- **Wie partielle Ableitungen die lineare Abbildung festlegen:** Für Schritte $\boldsymbol{\epsilon} = h\mathbf{e}_i$ folgt zwingend $\Lambda(\mathbf{e}_i) = \partial\ell/\partial w_i$, also $\Lambda(\boldsymbol{\epsilon}) = \sum_i \epsilon_i \frac{\partial\ell}{\partial w_i}$.

<!--en:0f527002-->
- **Vorhersage am Punkt $(0, \log 2)$:** Für den gemeinsamen Schritt $\boldsymbol{\epsilon} = (0.01, -0.03)$ ergibt sich $\Delta f \approx \tfrac{1}{3}(0.01) + \tfrac{2}{3}(-0.03) = -0.01667$.

<!--en:2ed3f7f0-->
::: {.slide title="Warum partielle Ableitungen allein nicht genügen"}

<!--en:2287b2d2-->
[Eine Hürde für die Differenzierbarkeit]{.kicker}

<!--en:31485fc9-->
- **Pathologischer Grat $f(x, y) = \frac{2xy}{x^2+y^2}$ ($f(0,0)=0$):** Auf beiden Achsen verschwindet die Funktion identisch ($\partial_x f(0,0) = \partial_y f(0,0) = 0$), auf der Diagonalen hat sie dagegen den Wert $1$ für $y = x \neq 0$.

<!--en:9897e685-->
- **Hinreichende $C^1$-Bedingung:** Existieren alle partiellen Ableitungen in einer Umgebung von $\mathbf{w}$ und sind sie an der Stelle $\mathbf{w}$ **stetig**, so ist $\ell$ an der Stelle $\mathbf{w}$ differenzierbar.

<!--en:041912a5-->
::: {.slide title="Numerische Prüfung: Der Fehler der linearen Näherung fällt wie O(‖ε‖²)"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:68524fd6-->
Differenzierbarkeit garantiert für den Fehler der linearen Näherung nur $o(\|\boldsymbol{\epsilon}\|)$. Bei glatten Funktionen wie Log-Sum-Exp sinkt der Fehler jedoch auf ein Hundertstel, wenn der Schritt $\boldsymbol{\epsilon} = (0.01, -0.03)$ auf ein Zehntel schrumpft:

<!--en:6487f384-->
Bei $\|\boldsymbol{\epsilon}\| = 3.2 \times 10^{-2}$ beträgt die Abweichung des linearen Modells $1.786 \times 10^{-4}$. Schrumpft $\|\boldsymbol{\epsilon}\|$ um die Faktoren $10\times$ und $100\times$, sinkt sie auf $1.779 \times 10^{-6}$ bzw. $1.778 \times 10^{-8}$. Der Fehler nimmt also quadratisch mit der Schrittweite ab ($O(\|\boldsymbol{\epsilon}\|^2)$).

<!--en:c81fbf5f-->
::: {.slide title="Richtungsableitungen: Steigungen in beliebigen Richtungen"}

<!--en:4d2b930e-->
[Projektionen des Gradienten]{.kicker}

<!--en:3328504c-->
- **Formel für die Richtungsableitung:** Für jeden Einheitsvektor $\mathbf{u}$ ($\|\mathbf{u}\|_2 = 1$) gilt $D_{\mathbf{u}}\ell(\mathbf{w}) = \lim_{h \to 0}\frac{\ell(\mathbf{w}+h\mathbf{u})-\ell(\mathbf{w})}{h} = \mathbf{u}\cdot\nabla_{\mathbf{w}}\ell(\mathbf{w})$.

<!--en:47c9c10f-->
- **Drei Richtungen am Punkt $(0, \log 2)$ mit $\nabla f = (\tfrac{1}{3}, \tfrac{2}{3})^\top$:**
  - Gleichmäßige Verschiebung $(1, 1)/\sqrt{2}$ ergibt $+1/\sqrt{2} \approx +0.707$.
  - Steilster Abstieg $-(1, 2)/\sqrt{5}$ ergibt $-\sqrt{5}/3 \approx -0.745$.
  - Tangente an die Niveaulinie $(2, -1)/\sqrt{5}$ ergibt $0$.

<!--en:2b09798f-->
::: {.slide title="Steilster Abstieg mit der Cauchy-Schwarz-Ungleichung"}

<!--en:106df48d-->
[Optimierung erster Ordnung]{.kicker}

<!--en:f33723a9-->
- **Cauchy-Schwarz-Schranken für Richtungsableitungen:** Für jeden Einheitsvektor $\mathbf{v}$ ($\|\mathbf{v}\|_2 = 1$) gilt
  $$-\|\nabla \ell\|_2 \le \mathbf{v} \cdot \nabla \ell \le \|\nabla \ell\|_2$$

<!--en:c5c10359-->
- **Eindeutige Richtungen der stärksten Änderung:** Nur $\mathbf{v} = +\nabla \ell / \|\nabla \ell\|_2$ erreicht die obere Schranke (steilster Anstieg), nur $\mathbf{v} = -\nabla \ell / \|\nabla \ell\|_2$ die untere (steilster Abstieg).

<!--en:650d5635-->
- **Abhängigkeit von der Norm (SGD und Adam):** Welche Richtung „am steilsten“ ist, hängt von der Norm ab, die den Schritt beschränkt. Unter der $\ell_\infty$-Boxbeschränkung $\|\mathbf{v}\|_\infty \le 1$ ergibt sich der steilste Abstieg komponentenweise als $-\operatorname{sign}(\nabla \ell)$. Das ähnelt den normierten Aktualisierungen von Adam.

<!--en:180fe76d-->
::: {.slide title="Gradienten stehen senkrecht auf Niveaumengen"}

<!--en:33e43f76-->
[Geometrie der Niveaulinien]{.kicker}

<!--en:a4a0c32b-->
- **Beweis mit Kurven in der Niveaumenge:** Sei $\boldsymbol{\gamma}(t)$ eine Kurve in $\{\ell(\mathbf{x}) = c\}$ mit $\boldsymbol{\gamma}(0) = \mathbf{x}$ und Geschwindigkeitsvektor $\mathbf{v} = \boldsymbol{\gamma}'(0)$. Ableiten von $\ell(\boldsymbol{\gamma}(t)) = c$ ergibt $0 = \mathbf{v}\cdot\nabla\ell(\mathbf{x})$.

<!--en:5a4c17df-->
- **Log-Sum-Exp am Punkt $(0, \log 2)$:** Die Niveaulinie $e^x + e^y = 3$ hat den Tangentialvektor $\mathbf{v} = (e^y, -e^x) = (2, -1)$. Er ist orthogonal zu $\nabla f = (\tfrac{1}{3}, \tfrac{2}{3})^\top$.

<!--en:ad11918a-->
::: {.slide title="Tangentialebenen an Verlustflächen"}

<!--en:d3267e4f-->
[Linearisierung einer Fläche]{.kicker}

<!--en:5c66016f-->
- **Graph der linearen Näherung:** Ohne den Restterm $o(\|\mathbf{x} - \mathbf{x}_0\|)$ beschreibt die Entwicklung erster Ordnung die **Tangentialebene** an die Fläche $z = f(\mathbf{x})$ an der Stelle $\mathbf{x}_0$:
  $$z = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0) \cdot (\mathbf{x} - \mathbf{x}_0)$$

<!--en:7fcb4898-->
- **Flächennormale in 3D und Normale zur Niveaulinie in 2D:** Die Ebenengleichung $\nabla f(\mathbf{x}_0) \cdot (\mathbf{x} - \mathbf{x}_0) - (z - f(\mathbf{x}_0)) = 0$ zeigt, dass $[\nabla f(\mathbf{x}_0), -1]^\top$ (oder $[-\nabla f(\mathbf{x}_0), 1]^\top$) im $(\mathbf{x}, z)$-Raum senkrecht auf der Tangentialebene steht.

<!--en:fa432f02-->
- **Horizontaler Schnitt:** Bei der Projektion der 3D-Normalen auf die Grundebene entfällt die $z$-Komponente. Übrig bleibt $\nabla f(\mathbf{x}_0)$ als Normale zur Niveaulinie.

<!--en:b49993f4-->
::: {.slide title="Kritische Punkte und Existenz globaler Extrema"}

<!--en:1bcd4dc3-->
[Optimalitätsbedingung erster Ordnung]{.kicker}

<!--en:753637cc-->
- **Kritische Punkte ($\nabla f(\mathbf{x}_0) = \mathbf{0}$):** Für $\psi(x, y) = x e^{-x^2 - y^2}$ folgt aus $\nabla\psi = \mathbf{0}$ zwingend $y = 0$ und $x = \pm 1/\sqrt{2}$ ($\psi \approx \mp 0.4289$).

<!--en:20d50000-->
- **Warum ein globaler Existenznachweis nötig ist:** $\nabla f = \mathbf{0}$ allein beweist noch nicht, dass ein Minimum existiert (Beispiel: $x^3$ bei $0$).

<!--en:b841664c-->
- **Abschätzung außerhalb einer kompakten Kreisscheibe:** Außerhalb des Radius $R = 2$ gilt $|\psi| \le 2e^{-4} < 0.04$. Mit dem Satz vom Minimum und Maximum folgt, dass $(-1/\sqrt{2}, 0)$ die globale Minimalstelle ist.

<!--en:dad70a1a-->
::: {.slide title="Die Hesse-Matrix und die Krümmung von Log-Sum-Exp"}

<!--en:c0496c30-->
[Zweite partielle Ableitungen]{.kicker}

<!--en:3e5b35a0-->
- **Einträge der Hesse-Matrix:** $[\mathbf{H}_f(\mathbf{x})]_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}(\mathbf{x})$ misst, wie sich die Steigung in Richtung $j$ entlang der Achse $i$ ändert.

<!--en:135dcc7e-->
- **Hesse-Matrix von Log-Sum-Exp:** Ableiten von $p_i = e^{x_i}/\sum_k e^{x_k}$ ergibt $\mathbf{H}_f(\mathbf{x}) = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$.

<!--en:b190d2df-->
- **Spektrum am Punkt $(0, \log 2)$:** Eigenwert $0$ in Richtung $(1, 1)^\top$ (flache Verschiebungsrichtung) und $\tfrac{4}{9}$ in Richtung $(1, -1)^\top$.

<!--en:02a899d7-->
::: {.slide title="Symmetrie der Hesse-Matrix: Satz von Schwarz"}

<!--en:eb70e62e-->
[Gemischte partielle Ableitungen]{.kicker}

<!--en:4f61e472-->
- **Satz von Schwarz (Clairaut–Schwarz):** Sind die gemischten partiellen Ableitungen $\partial^2_{ij}f$ und $\partial^2_{ji}f$ an der Stelle $\mathbf{x}$ stetig, so stimmen sie überein. Folglich gilt $\mathbf{H}_f(\mathbf{x}) = \mathbf{H}_f(\mathbf{x})^\top$.

<!--en:e775a1aa-->
- **Beweis über die symmetrische zweite Differenz:** Beide Ableitungsreihenfolgen liefern Grenzwerte desselben unter $i \leftrightarrow j$ symmetrischen Differenzenquotienten $\Delta_h$ über vier Eckpunkte.

<!--en:57e24984-->
- **Folgerung aus dem Spektralsatz:** Bei jeder $C^2$-Funktion hat die Hesse-Matrix reelle Eigenwerte und eine Orthonormalbasis aus Eigenvektoren.

<!--en:eb0abae3-->
::: {.slide title="Taylorentwicklung zweiter Ordnung für mehrere Variablen"}

<!--en:88aa9790-->
[Quadratische Näherung]{.kicker}

<!--en:3d4f3430-->
- **Rückführung auf einen eindimensionalen Schnitt ($\varphi(t) = f(\mathbf{x}_0 + t\boldsymbol{\delta})$):** $\varphi'(t) = \nabla f\cdot\boldsymbol{\delta}$ und $\varphi''(t) = \boldsymbol{\delta}^\top\mathbf{H}_f\boldsymbol{\delta}$.

<!--en:82879b0a-->
- **Exakte Lagrange-Form:** Es gilt $f(\mathbf{x}_0 + \boldsymbol{\delta}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)\cdot\boldsymbol{\delta} + \tfrac{1}{2}\boldsymbol{\delta}^\top\mathbf{H}_f(\boldsymbol{\xi})\boldsymbol{\delta}$ für einen Punkt $\boldsymbol{\xi}$ auf der Verbindungsstrecke.

<!--en:8fe5c80c-->
- **Lokale quadratische Näherung:** Tritt an die Stelle von $\mathbf{H}_f(\boldsymbol{\xi})$ die Matrix $\mathbf{H}_f(\mathbf{x}_0)$, bleibt ein Restterm der Ordnung $o(\|\boldsymbol{\delta}\|^2)$ (sogar $O(\|\boldsymbol{\delta}\|^3)$ bei $C^3$-Funktionen).

<!--en:9b515222-->
::: {.slide title="Vorhersage des linearen Näherungsfehlers mit der Hesse-Matrix"}

<!--en:fc917b85-->
[Genauigkeit zweiter Ordnung]{.kicker}

<!--en:63962d6f-->
Für Log-Sum-Exp am Punkt $(0, \log 2)$ und den Schritt $\boldsymbol{\epsilon} = (0.01, -0.03)$ sagt der quadratische Term $\tfrac{1}{2}\boldsymbol{\epsilon}^\top \mathbf{H}_f \boldsymbol{\epsilon} = \tfrac{1}{9}(\epsilon_1 - \epsilon_2)^2$ $= \tfrac{1}{9}(0.04)^2$ $\approx 1.7778 \times 10^{-4}$ die gemessene Abweichung $1.786 \times 10^{-4}$ auf zwei Stellen genau voraus (relativer Fehler unter 0.5%):

<!--en:0c325ce8-->
Die kleine Restdifferenz ($\approx 8 \times 10^{-7}$) zwischen der tatsächlichen Abweichung und $\tfrac{1}{2}\boldsymbol{\epsilon}^\top \mathbf{H}_f \boldsymbol{\epsilon}$ stammt vom Taylor-Term dritter Ordnung $O(\|\boldsymbol{\epsilon}\|^3)$. Sie sinkt auf etwa ein Tausendstel, wenn $\|\boldsymbol{\epsilon}\|$ auf ein Zehntel schrumpft.

<!--en:a055ef9d-->
::: {.slide title="Eine gekrümmte Fläche und ihr Taylor-Paraboloid"}

<!--en:a5996e51-->
[Quadratische Näherung in 3D]{.kicker}

<!--en:208711af-->
- **Quadratisches Modell von $\psi(x, y) = x e^{-x^2 - y^2}$ am Punkt $(-1, 0)$:** Mit $\psi(-1, 0) = -e^{-1}$, $\nabla \psi(-1, 0) = (-e^{-1}, 0)^\top$ und $\mathbf{H}_\psi(-1, 0) = 2e^{-1}\mathbf{I}$ lautet das Taylorpolynom zweiter Ordnung
  $$q(x, y) = e^{-1}\bigl(-1 - (x + 1) + (x + 1)^2 + y^2\bigr)$$

<!--en:e4142b94-->
- **Geometrische Übereinstimmung:** Am Punkt $(-1, 0)$ stimmen die Fläche $z = \psi(x, y)$ (blau) und ihr Taylor-Paraboloid $z = q(x, y)$ (orange) in Wert, Gradient und Krümmung überein.

<!--en:568bfc03-->
::: {.slide title="Kubische Abnahme des Restterms der quadratischen Näherung"}

<!--en:8283c3dc-->
[Restterm dritter Ordnung]{.kicker}

<!--en:06e1dc4b-->
Der Vergleich von $\psi(x, y)$ und seiner quadratischen Näherung $q(x, y)$ an den Punkten $(-1 + d, d)$, ausgehend von $(-1, 0)$, bestätigt für den Restterm die Größenordnung $O(d^3)$:

<!--en:e8bd4623-->
Verdoppelt sich der Koordinatenschritt $d$ von $0.05$ auf $0.10$, wächst die Abweichung $|\psi - q|$ fast um den Faktor $8 = 2^3$ ($0.000119 \to 0.000921$). Wer in jedem Schritt direkt zur Minimalstelle des lokalen quadratischen Modells springt, erhält das Newton-Verfahren für mehrere Variablen.

<!--en:2b9d7c63-->
::: {.slide title="Das Kriterium der zweiten Ableitung anhand der Eigenwerte"}

<!--en:4115ea21-->
[Klassifikation kritischer Punkte]{.kicker}

<!--en:df1d8a8c-->
An einem kritischen Punkt gilt $\nabla f(\mathbf{x}_0) = \mathbf{0}$. Die Darstellung von $\mathbf{v}$ in der Orthonormalbasis zu den Eigenpaaren $\{(\lambda_i, \mathbf{q}_i)\}$ der symmetrischen Hesse-Matrix $\mathbf{H} = \mathbf{H}_f(\mathbf{x}_0)$ bringt die quadratische Form auf Diagonalgestalt:

<!--en:b6fb7f74-->
**Proposition (Optimalitätsbedingungen zweiter Ordnung bei $\nabla f(\mathbf{x}_0) = \mathbf{0}$):**

<!--en:f1a17e96-->
1. **Notwendige Bedingung für ein lokales Minimum:** Ist $\mathbf{x}_0$ eine lokale Minimalstelle, so gilt $\mathbf{H} \succeq 0$ (alle $\lambda_i \ge 0$).

<!--en:5437e4c3-->
2. **Hinreichende Bedingung für ein striktes lokales Minimum:** Gilt $\mathbf{H} \succ 0$ (alle $\lambda_i > 0$), so gilt für kleine $\boldsymbol{\delta} \ne \mathbf{0}$ die Abschätzung $f(\mathbf{x}_0 + \boldsymbol{\delta}) - f(\mathbf{x}_0) \ge \tfrac{1}{4}\lambda_{\min}\|\boldsymbol{\delta}\|^2 > 0$. Somit ist $\mathbf{x}_0$ eine strikte lokale Minimalstelle (bei $\mathbf{H} \prec 0$ eine strikte lokale Maximalstelle).

<!--en:050c2b4f-->
3. **Sattelpunkt:** Hat $\mathbf{H}$ sowohl strikt positive als auch strikt negative Eigenwerte, so steigt $f$ entlang der Eigenvektoren zu positiven Eigenwerten und fällt entlang der Eigenvektoren zu negativen Eigenwerten. Damit ist $\mathbf{x}_0$ ein **Sattelpunkt**.

<!--en:e1914550-->
::: {.slide title="Vier Krümmungsfälle an kritischen Punkten"}

<!--en:b8d63b39-->
[Geometrische Klassifikation]{.kicker}

<!--en:bf2979a1-->
- **(a) $\mathbf{H} \succ 0$ (lokales Minimum):** Alle Eigenwerte sind positiv ($\psi$ am Punkt $(-1/\sqrt{2}, 0)$). Die Niveaulinien sind Ellipsen und $-\mathbf{H}\mathbf{v}$ zeigt nach innen.

<!--en:9688b44b-->
- **(b) $\mathbf{H} \prec 0$ (lokales Maximum):** Alle Eigenwerte sind negativ ($\psi$ am Punkt $(+1/\sqrt{2}, 0)$). Die Fläche krümmt sich in allen Richtungen nach unten.

<!--en:cc85f50d-->
- **(c) Indefinite Matrix $\mathbf{H}$ (Sattelpunkt):** Die Eigenwerte haben verschiedene Vorzeichen ($\lambda = \pm 4$ für $x^4 + y^4 - 4xy$ bei $\mathbf{0}$). Die Niveaulinien sind Hyperbeln. Der Abstieg führt entlang einer Achse nach innen und entlang der anderen nach außen.

<!--en:92161b68-->
- **(d) Singuläre semidefinite Matrix $\mathbf{H}$ (keine Entscheidung):** Beim Eigenwert null entscheiden Terme höherer Ordnung. Im Ursprung hat $x^2 + y^4$ ein Minimum und $x^2 - y^4$ einen Sattelpunkt, obwohl für beide $\mathbf{H}(\mathbf{0}) = \operatorname{diag}(2, 0)$ gilt.

<!--en:2ab5fd70-->
::: {.slide title="Numerische Klassifikation bei flachen Richtungen"}

<!--en:de9e7a8c-->
[Hesse-Matrizen aus finiten Differenzen]{.kicker}

<!--en:c3d2a13b-->
Aus zentrierten zweiten Differenzen ergibt sich eine $2 \times 2$-Hesse-Matrix. Die Klassifikation kritischer Punkte anhand ihrer Eigenwerte erfordert eine positive Toleranz `tol` für den Vergleich mit null. Sonst erscheint der Diskretisierungsfehler in flachen Richtungen mit Anstieg vierter Ordnung als echte Krümmung:

<!--en:27a54583-->
In der Talsohle von $x^2 + y^4$ am Punkt $(0, 0)$ hat die zweite partielle Ableitung $\partial_{yy} f(0, 0)$ den Wert $0$. Finite Differenzen mit der Schrittweite $h = 10^{-4}$ liefern jedoch $2h^2 = 2 \times 10^{-8} > 0$. Mit `tol=1e-6` meldet das Kriterium der zweiten Ableitung zu Recht **keine Entscheidung**.

<!--en:22b609ce-->
::: {.slide title="Warum Sattelpunkte in hochdimensionalen Landschaften überwiegen"}

<!--en:35911c1c-->
[Theorie der Zufallsmatrizen]{.kicker}

<!--en:4684577f-->
- **Abstoßung der Eigenwerte zufälliger Hesse-Matrizen:** In $n$ Dimensionen müssen an einer nicht entarteten lokalen Minimalstelle alle $n$ Eigenwerte positiv sein. Für symmetrische Zufallsmatrizen aus dem gaußschen orthogonalen Ensemble (GOE) gilt $P(\mathbf{H} \succ 0) \sim \exp(-\frac{\ln 3}{4}n^2) \ll 2^{-n}$.

<!--en:2c53a462-->
- **Verlust und Index (Dauphin et al., 2014):** Kritische Punkte mit hohem Verlust sind fast ausschließlich Sattelpunkte (hoher Anteil negativer $\lambda_i$). Lokale Minima häufen sich nahe dem globalen Verlustminimum.

<!--en:4cd049ec-->
::: {.slide title="Optimierung unter Nebenbedingungen und Lagrange-Multiplikatoren"}

<!--en:ef374ef9-->
[Flächen zulässiger Punkte]{.kicker}

<!--en:3dc97455-->
- **Beschränkung von $\mathbf{x}$ auf $\{g(\mathbf{x}) = c\}$:** An einer lokalen Optimalstelle $\mathbf{x}^\star$ unter der Nebenbedingung muss für jeden tangentialen Schritt $\mathbf{v}$ auf der zulässigen Fläche die Änderung von $f$ in erster Ordnung null sein. Also gilt $\mathbf{v} \cdot \nabla f(\mathbf{x}^\star) = 0$.

<!--en:289f71f3-->
- **Parallele Gradienten im Berührpunkt:** Da $\nabla g(\mathbf{x}^\star) \ne \mathbf{0}$ einen Normalenvektor zur zulässigen Fläche angibt, steht jeder zu $\nabla g(\mathbf{x}^\star)$ orthogonale Vektor auch senkrecht auf $\nabla f(\mathbf{x}^\star)$. Daraus folgt die **Lagrange-Bedingung**:
  $$\nabla f(\mathbf{x}^\star) = \mu\,\nabla g(\mathbf{x}^\star)$$

<!--en:319f81f9-->
- **Beispiel ($\min x^2 + y^2$ unter $xy = 1$):** Aus $(2x, 2y) = \mu(y, x)$ und $xy = 1$ folgen $\mu = 2$ und die nächstgelegenen Punkte $\pm(1, 1)$ mit dem quadrierten Abstand $2$ zum Ursprung.

<!--en:ba3d41c5-->
::: {.slide title="Verständnisfragen zur Analysis mehrerer Variablen"}

<!--en:b6d9a92f-->
[Wiederholung und Übungen]{.kicker}

<!--en:2102d872-->
Verständnis von Richtungsableitungen, Eigenwerten der Hesse-Matrix und Lagrange-Multiplikatoren prüfen:

<!--en:f9c4ee5f-->
1. **Richtungsableitung und Niveaumengen:** Am Punkt $(0, \log 2)$ hat Log-Sum-Exp den Gradienten $\nabla f = (\tfrac{1}{3}, \tfrac{2}{3})^\top$. Wie groß ist die maximale Richtungsableitung $D_{\mathbf{u}}f$ über alle Einheitsvektoren $\mathbf{u}$, und für welchen Einheitsvektor $\mathbf{u}$ wird sie erreicht?

<!--en:44261c89-->
2. **Klassifikation mit der Hesse-Matrix:** Es gelte $\nabla f(\mathbf{x}_0) = \mathbf{0}$ und $\mathbf{H}_f(\mathbf{x}_0) = \begin{bmatrix} 1 & 3 \\ 3 & 1 \end{bmatrix}$. Ist $\mathbf{x}_0$ eine lokale Minimalstelle, eine lokale Maximalstelle oder ein Sattelpunkt?

<!--en:37c65462-->
3. **Lagrange-Multiplikator bei maximaler Entropie:** Die Maximierung von $\mathbf{z} \cdot \mathbf{p} - \sum_i p_i \log p_i$ unter der Nebenbedingung $g(\mathbf{p}) = \sum_i p_i = 1$ ergibt $z_i - \log p_i - 1 = \mu$. Welche Wahrscheinlichkeitsverteilung $\mathbf{p}^\star$ löst diese Gleichung?

<!--en:557b98b6-->
**Lösungen:**

<!--en:d4e69fb7-->
1. Nach der Cauchy-Schwarz-Ungleichung beträgt die maximale Richtungsableitung $\|\nabla f\|_2 = \sqrt{(1/3)^2 + (2/3)^2} = \sqrt{5}/3 \approx 0.745$. Sie wird in Richtung $\mathbf{u} = \nabla f / \|\nabla f\|_2 = \frac{1}{\sqrt{5}}(1, 2)^\top$ erreicht.

<!--en:a0acc007-->
2. $\mathbf{H}_f(\mathbf{x}_0)$ hat die Spur $2$ und die Determinante $1 - 9 = -8 < 0$ (Eigenwerte $\lambda_1 = 4 > 0$ in Richtung $(1,1)^\top$ und $\lambda_2 = -2 < 0$ in Richtung $(1,-1)^\top$). Da die Eigenwerte verschiedene Vorzeichen haben, ist $\mathbf{x}_0$ trotz der positiven Diagonaleinträge ein **Sattelpunkt**.

<!--en:53acb240-->
3. Auflösen von $z_i - \log p_i - 1 = \mu$ ergibt $p_i = e^{z_i - 1 - \mu} \propto e^{z_i}$. Die Normierung auf $\sum_i p_i = 1$ liefert die **Softmax-Verteilung** $p_i^\star = e^{z_i} / \sum_k e^{z_k}$.

<!--en:7555a790-->
::: {.slide title="Zusammenfassung: Werkzeuge der Analysis mehrerer Variablen"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:3eff081a-->
| Mathematischer Begriff oder Regel | Formel und Bedingung | Geometrische Bedeutung und Bezug zum Deep Learning |

<!--en:23f07d28-->
| **Lineares Modell für mehrere Variablen** | $\ell(\mathbf{w}+\boldsymbol{\epsilon}) = \ell(\mathbf{w}) + \boldsymbol{\epsilon}\cdot\nabla_{\mathbf{w}}\ell(\mathbf{w}) + o(\|\boldsymbol{\epsilon}\|)$ | Gilt bei stetigen partiellen Ableitungen ($C^1$) und sagt die Wirkung gemeinsamer Gewichtsänderungen voraus |

<!--en:1b16d638-->
| **Richtungsableitung** | $D_{\mathbf{u}}\ell(\mathbf{w}) = \mathbf{u} \cdot \nabla_{\mathbf{w}}\ell(\mathbf{w})$ ($\|\mathbf{u}\|_2 = 1$) | Momentane Steigung in Richtung $\mathbf{u}$ im Intervall $[-\|\nabla\ell\|_2, +\|\nabla\ell\|_2]$ |

<!--en:68cf66de-->
| **Steilster Abstieg und Niveaulinien** | $\mathbf{u}^\star = -\nabla\ell / \|\nabla\ell\|_2$; $\mathbf{v}_{\text{tangent}} \cdot \nabla\ell = 0$ | $-\nabla\ell$ zeigt senkrecht zu den Niveaumengen $\{\ell = c\}$ bergab |

<!--en:5ccef131-->
| **Symmetrische Hesse-Matrix $\mathbf{H}_f$** | $[\mathbf{H}_f]_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$ | Erfasst die Flächenkrümmung. Für Log-Sum-Exp gilt $\mathbf{H} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ |

<!--en:d9329394-->
| **Taylor-Modell zweiter Ordnung** | $f(\mathbf{x}_0+\boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f\cdot\boldsymbol{\delta} + \tfrac{1}{2}\boldsymbol{\delta}^\top\mathbf{H}_f\boldsymbol{\delta}$ | Lokale quadratische Näherung, die die Abweichung der Tangentialebene von der Ordnung $O(\|\boldsymbol{\delta}\|^2)$ erfasst |

<!--en:0045b466-->
| **Kriterium der zweiten Ableitung** | $\mathbf{v}^\top\mathbf{H}\mathbf{v} = \sum_i \lambda_i (\mathbf{q}_i^\top\mathbf{v})^2$ bei $\nabla f(\mathbf{x}_0)=\mathbf{0}$ | Alle $\lambda_i > 0 \Rightarrow$ striktes Minimum, alle $\lambda_i < 0 \Rightarrow$ striktes Maximum, verschiedene Vorzeichen $\Rightarrow$ Sattelpunkt |

<!--en:5a5a07b6-->
| **Lagrange-Multiplikatoren** | $\nabla f(\mathbf{x}^\star) = \mu\,\nabla g(\mathbf{x}^\star)$ unter $g(\mathbf{x}) = c$ | Die Niveaumenge von $f$ berührt die Fläche der zulässigen Punkte tangential ($\nabla g(\mathbf{x}^\star) \ne \mathbf{0}$) |
