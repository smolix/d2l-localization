<!--en:2867bcfd-->
# Beschränkte Optimierung

<!--en:0e94fa30-->
Bisher konnte jeder Parameter bei der Optimierung jeden reellen Wert annehmen. Um ein Minimum zu finden, suchten wir daher einen Punkt, an dem der Gradient verschwindet. Viele Größen im Deep Learning haben diese Freiheit nicht. Attention-Gewichte müssen nichtnegativ sein und in der Summe eins ergeben. Gradienten-Clipping begrenzt die Norm eines Aktualisierungsschritts auf einen Schwellenwert. Eine Normschranke begrenzt die Länge eines Gewichtsvektors, und bei einer Support Vector Machine muss jede Margin im Training mindestens eins betragen. Müssen die Variablen im zulässigen Bereich (engl. feasible set) bleiben, verschwindet der Gradient im Optimum nur selten. Das Kriterium aus den vorigen Abschnitten versagt dann. Woran erkennen wir einen optimalen Punkt unter Nebenbedingungen, und wie berechnen wir ihn?

<!--en:40c9fac7-->
Das einfachste Beispiel liefert das Gradienten-Clipping. Angenommen, wir wollen den Vektor $\mathbf{y} = (3, 4)$ auf eine Norm von höchstens $1$ begrenzen. Skalieren wir ihn auf Einheitslänge, erhalten wir $(0.6, 0.8)$. Das ist der Punkt der Einheitskreisscheibe, der $\mathbf{y}$ am nächsten liegt. Mit anderen Worten minimiert $(0.6, 0.8)$ den quadrierten Abstand $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ auf der Kreisscheibe $\|\mathbf{x}\| \le 1$ und erreicht den Optimalwert $f = 8$. Dennoch ist der Gradient dort $(-2.4, -3.2)$, also weit von null entfernt. Der Punkt $(0.6, 0.8)$ ist optimal, weil keine *zulässige* Richtung $f$ verringert. Jede Bewegung, bei der $\|\mathbf{x}\| \le 1$ gilt, erhöht entweder $f$ oder lässt die Funktion in erster Ordnung unverändert. Für solche Probleme brauchen wir ein neues Kriterium anstelle des verschwindenden Gradienten. Auch den gewöhnlichen Gradientenabstieg müssen wir ersetzen, denn sein Schritt würde hier aus dem zulässigen Bereich hinausführen.

<!--en:e2aca00f-->
An die Stelle der Bedingung „Der Gradient ist null“ tritt eine Bedingung mit einem Skalar pro Nebenbedingung. Dieser *Multiplikator* misst, wie stark das Optimum gegen die jeweilige Grenze drückt. Bei $(0.6, 0.8)$ wirkt der Rand mit der Stärke $4$ entlang der äußeren Normalen entgegen, sodass $(-2.4, -3.2) + 4\,(0.6, 0.8) = \mathbf{0}$ gilt. Mit den Multiplikatoren erhalten wir Optimalitätsbedingungen erster Ordnung, nämlich die Lagrange-Bedingung für Gleichungsnebenbedingungen und die Karush-Kuhn-Tucker-Bedingungen (KKT-Bedingungen) für Ungleichungen. Die Projektion von $(3, 4)$ auf $(0.6, 0.8)$ legt zugleich einen Algorithmus nahe. Wir machen einen gewöhnlichen Gradientenschritt und projizieren dann auf den nächstgelegenen zulässigen Punkt zurück. Ersetzen wir eine feste Grenze durch einen Strafterm mit Knick, etwa den Betragsterm, mit dem das Lasso jedes Gewicht bestraft, verallgemeinert sich die Projektion zu einem *Proximalschritt*. Er setzt kleine Gewichte exakt auf null. In diesem Abschnitt sorgt die Konvexität (:numref:`sec_mdl-convexity`) dafür, dass die Bedingungen globale Optimalität garantieren und unsere Algorithmen konvergieren.

<!--en:c13c2743-->
Drei Ideen stehen im Mittelpunkt dieses Abschnitts: die KKT-Bedingungen, die bei konvexen Problemen hinreichend sind, die Ungleichung für den stumpfen Winkel, die eine Projektion charakterisiert, und das weiche Schwellenwertverfahren (engl. soft-thresholding).

<!--en:44420906-->
## Gleichungsnebenbedingungen und Lagrange-Multiplikatoren

<!--en:4d911d1c-->
### Berührung und Lagrange-Bedingung

<!--en:22daf9e8-->
Gleichungsnebenbedingungen sind einfacher als Ungleichungsnebenbedingungen. Ersetzen wir daher die Kreisscheibe aus dem Clipping-Beispiel zunächst durch ihren Rand. Wir minimieren $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ mit $\mathbf{y} = (3, 4)$ auf dem Einheitskreis, beschrieben durch $h(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) = 0$. Allgemeiner wollen wir eine differenzierbare Zielfunktion $f : \mathbb{R}^n \to \mathbb{R}$ unter einer einzigen Gleichungsnebenbedingung minimieren:

<!--en:e4132a97-->
Da der zulässige Bereich eine Fläche ist (in zwei Dimensionen eine Kurve), fehlt ihm eine Dimension gegenüber dem umgebenden Raum. Jede zulässige Bewegung muss auf der Fläche bleiben.

<!--en:2cdfffbb-->
:numref:`fig_mdl-opt-lagrange-tangency` zeigt, woran wir ein Optimum geometrisch erkennen. Betrachten wir einen zulässigen Punkt, an dem eine Niveaumenge von $f$ den Kreis *schneidet*. Dann hat $\nabla f$ eine von null verschiedene Komponente entlang des Kreises. Bewegen wir uns auf dem Kreis entgegen dieser Komponente, verringern wir $f$, ohne den zulässigen Bereich zu verlassen. Ein solcher Punkt kann kein Optimum sein. Ein Optimum ist nur dort möglich, wo die Niveaumenge den Kreis *berührt*. Dort lässt jede Bewegung entlang des Kreises $f$ in erster Ordnung unverändert. Da sich berührende Kurven parallele Normalenvektoren haben, muss $\nabla f$ im Optimum ein skalares Vielfaches von $\nabla h$ sein.

<!--en:c119099e-->
![Minimierung von $f=\tfrac12\|\mathbf{x}-\mathbf{y}\|^2$ mit $\mathbf{y}=(3,4)$ auf dem Einheitskreis $h(\mathbf{x})=0$. Wo eine Niveaumenge den Kreis schneidet, hat $\nabla f$ eine Komponente entlang des Kreises. Damit gibt es eine zulässige Abstiegsrichtung. Bei $\mathbf{x}^\star=(0.6,0.8)$ berührt die Niveaumenge den Kreis, und es gilt $\nabla f=-4\,\nabla h$. Auch der gegenüberliegende Punkt $(-0.6,-0.8)$ erfüllt die Berührbedingung, mit $\nu=-6$. Er ist das Maximum.](../img/mdl-opt-lagrange-tangency.svg)

<!--en:00542f2e-->
**Proposition (Lagrange-Bedingung).** *Sei $\mathbf{x}^\star$ ein lokales Minimum von $f$ auf $\{\mathbf{x} : h(\mathbf{x}) = 0\}$. Die Funktion $f$ sei bei $\mathbf{x}^\star$ differenzierbar, $h$ sei in einer Umgebung von $\mathbf{x}^\star$ stetig differenzierbar, und es gelte $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$. Dann gibt es genau einen Skalar $\nu^\star$, den **Lagrange-Multiplikator**, für den gilt:*

<!--en:a8e53cd0-->
Ein Vielfaches des Normalenvektors der Nebenbedingung gleicht also den Gradienten der Zielfunktion aus. Dieselbe Lagrange-Bedingung haben wir in :numref:`subsec_mdl-lagrange` bewiesen. Dort lautet die Nebenbedingung $g(\mathbf{x}) = c$ und die Lagrange-Bedingung $\nabla f = \mu\,\nabla g$. Mit $h = g - c$ und $\nu = -\mu$ gehen die beiden Formen ineinander über. Wir verwenden hier das Pluszeichen, weil es dem Vorzeichen in der nun folgenden Lagrange-Funktion entspricht.

<!--en:8c81616a-->
Am Berührpunkt $(0.6, 0.8)$ in :numref:`fig_mdl-opt-lagrange-tangency` gilt $\nabla f = \mathbf{x} - \mathbf{y} = (-2.4, -3.2)$ und $\nabla h = \mathbf{x} = (0.6, 0.8)$. Die beiden Vektoren sind tatsächlich parallel. Aus $\nabla f = -4\,\nabla h$ folgt $\nu^\star = 4$.

<!--en:45f64196-->
Die Annahme $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$ ist unser erstes Beispiel für eine **Regularitätsbedingung** (engl. constraint qualification). Warum sie wesentlich ist, sehen wir, wenn wir $f(\mathbf{x}) = x_1$ unter der Nebenbedingung $h(\mathbf{x}) = x_1^2 + x_2^2 = 0$ minimieren. Der zulässige Bereich besteht aus dem einzigen Punkt $\mathbf{0}$, der damit automatisch das Minimum ist. Es gilt jedoch $\nabla f(\mathbf{0}) = (1, 0)$ und zugleich $\nabla h(\mathbf{0}) = \mathbf{0}$. Daher kann kein Multiplikator $\nu^\star$ die Gleichung :eqref:`eq_mdl-opt-lagrange-condition` erfüllen. Da der zulässige Bereich auf einen einzigen Punkt ohne Tangentialrichtungen geschrumpft ist, greift das Argument über die Berührung hier nicht.

<!--en:9cb507a0-->
### Die Lagrange-Funktion

<!--en:075d0b3e-->
Wir haben zwei Unbekannte, $\mathbf{x}$ und $\nu$, und zwei Anforderungen: Die Gradienten müssen parallel und der Punkt muss zulässig sein. Beides können wir als Stationarität einer einzigen Funktion von $(\mathbf{x}, \nu)$ ausdrücken. Diese **Lagrange-Funktion** addiert zur Zielfunktion die mit ihrem Multiplikator gewichtete Nebenbedingung:

<!--en:d2e34577-->
Setzen wir alle partiellen Ableitungen von $\mathcal{L}$ gleich null, erhalten wir

<!--en:0e95aa19-->
und damit zugleich :eqref:`eq_mdl-opt-lagrange-condition` und die Gleichungsnebenbedingung. So wird aus einem beschränkten Optimierungsproblem in $\mathbf{x}$ ein Stationaritätsproblem ohne Nebenbedingungen in $(\mathbf{x}, \nu)$.

<!--en:150fe69f-->
Lösen wir das Problem auf dem Einheitskreis mithilfe der Lagrange-Funktion. Stationarität bezüglich $\mathbf{x}$ ergibt $\nabla_{\mathbf{x}}\mathcal{L} = (\mathbf{x} - \mathbf{y}) + \nu\mathbf{x} = \mathbf{0}$. Der Punkt $\mathbf{x} = \mathbf{y}/(1 + \nu)$ muss also auf der Geraden durch den Ursprung und $\mathbf{y}$ liegen. Die Zulässigkeitsbedingung $\|\mathbf{x}\| = 1$ verlangt dann $|1 + \nu| = \|\mathbf{y}\| = 5$. Daraus ergeben sich zwei Lösungen:

<!--en:ee3fa255-->
* $\nu = 4$: $\mathbf{x} = (0.6, 0.8)$ im Abstand $4$ von $\mathbf{y}$ mit $f = 8$,

<!--en:8446d07f-->
* $\nu = -6$: $\mathbf{x} = (-0.6, -0.8)$ im Abstand $6$ von $\mathbf{y}$ mit $f = 18$.

<!--en:d9631d07-->
Die zweite Lösung ist der Punkt auf dem Kreis mit dem *größten Abstand* zu $\mathbf{y}$. Das zeigt eine wichtige Einschränkung: Die Lagrange-Bedingung ist nur notwendig. Sie gilt an jedem stationären Punkt von $f$ auf der Fläche, also auch an Maxima und Sattelpunkten. Hier unterscheidet das Vorzeichen des Multiplikators das Minimum vom Maximum. Bei den Ungleichungsnebenbedingungen im nächsten Abschnitt ergibt sich aus den KKT-Bedingungen eine allgemeine Vorzeichenregel.

<!--en:6b4aea98-->
Bei mehreren Gleichungsnebenbedingungen $h_j(\mathbf{x}) = 0$ für $j = 1, \ldots, p$ erhält jede ihren eigenen Multiplikator. Die Lagrange-Funktion lautet dann $\mathcal{L} = f + \sum_j \nu_j h_j$. Sind die Gradienten $\nabla h_j(\mathbf{x}^\star)$ der Nebenbedingungen linear unabhängig, so folgt mit demselben Argument, dass $\nabla f(\mathbf{x}^\star)$ im *Spann* der Normalenvektoren der Nebenbedingungen liegt. Eine Linearkombination der Nebenbedingungsgradienten gleicht also den Gradienten der Zielfunktion aus.

<!--en:290942e6-->
### Der nächstgelegene Punkt auf einer Hyperebene

<!--en:a7816eb9-->
Das einfachste Optimierungsproblem mit linearer Nebenbedingung fragt nach dem Punkt einer Hyperebene, der dem Ursprung am nächsten liegt. Wir minimieren $f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|^2$ unter der Nebenbedingung $h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = 0$. Setzen wir den Gradienten von $\mathcal{L} = \tfrac12\|\mathbf{x}\|^2 + \nu(\mathbf{a}^\top\mathbf{x} - b)$ gleich null, erhalten wir $\mathbf{x} + \nu\mathbf{a} = \mathbf{0}$. Der nächstgelegene Punkt liegt also auf der Geraden in Richtung des Normalenvektors $\mathbf{a}$. Setzen wir $\mathbf{x} = -\nu\mathbf{a}$ in die Nebenbedingung ein, erhalten wir $-\nu\|\mathbf{a}\|^2 = b$ und damit $\nu^\star = -b/\|\mathbf{a}\|^2$. Für die Lösung und den Optimalwert gilt daher

<!--en:904be79b-->
Geometrisch erreichen wir den nächstgelegenen Punkt, indem wir vom Ursprung um $b/\|\mathbf{a}\|$ in Richtung der Einheitsnormalen $\mathbf{a}/\|\mathbf{a}\|$ gehen. Das entspricht der Projektionsformel aus :numref:`sec_mdl-geometry-linear-algebraic-ops`.

<!--en:d86f540d-->
Für $\mathbf{a} = (3, 4)$ und $b = 5$ erhalten wir erneut $(0.6, 0.8)$, diesmal mit $\nu^\star = -0.2$ und $p^\star = \tfrac12$. Die Gerade $3x_1 + 4x_2 = 5$ berührt den Einheitskreis bei $(0.6, 0.8)$. Der Multiplikator dient nicht nur als algebraische Hilfsgröße. Erhöhen wir $b$ von $5$ auf $5.1$, so rückt die Gerade etwas weiter vom Ursprung weg. Der Optimalwert steigt dabei um etwa $0.02 = -\nu^\star \cdot 0.1$. Wie :numref:`subsec_mdl-shadow-prices` allgemein zeigt, misst ein Multiplikator stets, wie empfindlich der Optimalwert auf eine Verschiebung der zugehörigen Nebenbedingung reagiert.

<!--en:845d66e9-->
## Ungleichungsnebenbedingungen und KKT-Bedingungen

<!--en:6d439bac-->
### Aktive und inaktive Nebenbedingungen

<!--en:a58f414a-->
Die meisten Nebenbedingungen im maschinellen Lernen sind einseitig. Eine Norm darf höchstens $r$ betragen, eine Wahrscheinlichkeit muss mindestens $0$ sein, oder ein Ressourcenbudget setzt eine Obergrenze. Wir schreiben Ungleichungsnebenbedingungen in der Standardform

<!--en:a2168a9b-->
Damit ist jedes $g_i$ im Inneren seines Bereichs negativ und am Rand null. An einer Lösung $\mathbf{x}^\star$ gibt es für jede Ungleichung zwei Fälle. Gilt $g_i(\mathbf{x}^\star) < 0$, ist die Nebenbedingung **inaktiv**. Die Lösung liegt im Inneren des Bereichs, sodass die Nebenbedingung lokal keine Wirkung hat. Gilt $g_i(\mathbf{x}^\star) = 0$, ist die Nebenbedingung **aktiv**. Die Lösung liegt genau auf dem Rand, wo die Ungleichung wie eine Gleichung wirkt.

<!--en:a34f5462-->
Am Gradienten-Clipping lassen sich beide Fälle veranschaulichen. Wir projizieren einen Vektor $\mathbf{y}$ auf die euklidische Kugel mit Radius $r$, indem wir $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ unter der Nebenbedingung $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - r^2) \le 0$ minimieren. Führen wir einen Multiplikator $\lambda$ ein, liefert die Stationarität von $f + \lambda g$ die Gleichung $(\mathbf{x} - \mathbf{y}) + \lambda\mathbf{x} = \mathbf{0}$. Damit ist $\mathbf{x} = \mathbf{y}/(1 + \lambda)$ stets eine radiale Skalierung von $\mathbf{y}$. Welcher Fall eintritt, hängt von $\|\mathbf{y}\|$ ab:

<!--en:9c6e1853-->
* Gilt $\|\mathbf{y}\| \le r$, ist $\mathbf{y}$ bereits zulässig und minimiert $f$. Im Inneren ist die Nebenbedingung inaktiv. Es gilt $\lambda^\star = 0$ und $\mathbf{x}^\star = \mathbf{y}$. So bleibt etwa $\mathbf{y} = (0.3, 0.4)$ bei $r = 1$ unverändert.

<!--en:4e5d4527-->
* Gilt $\|\mathbf{y}\| > r$, kann kein innerer Punkt optimal sein. Jedes Minimum im Inneren müsste auch ein Minimum von $f$ ohne Nebenbedingung sein. Das einzige solche Minimum ist jedoch der unzulässige Punkt $\mathbf{y}$. Die Nebenbedingung muss daher aktiv sein. Aus $\|\mathbf{x}^\star\| = r$ folgt $\lambda^\star = \|\mathbf{y}\|/r - 1 > 0$. Für $\mathbf{y} = (3, 4)$ und $r = 1$ erhalten wir $\lambda^\star = 4$, $\mathbf{x}^\star = (0.6, 0.8)$ und $p^\star = \tfrac12 \cdot 4^2 = 8$.

<!--en:85e5d559-->
Beide Fälle lassen sich in einer Formel zusammenfassen,

<!--en:7b0983c6-->
die kurze Vektoren unverändert lässt und lange auf die Norm $r$ skaliert. Auf einen Gradientenvektor angewandt, ergibt dies das *Gradienten-Clipping* :cite:`Pascanu.Mikolov.Bengio.2013`. Im Code genügt für die Fallunterscheidung ein einziges `max`. Am so berechneten Punkt können wir direkt prüfen, dass sich der Gradient von $f$ und das Produkt aus $\lambda^\star$ und dem Nebenbedingungsgradienten $\nabla g(\mathbf{x}) = \mathbf{x}$ gegenseitig aufheben.

<!--en:84fda57e-->
Obwohl $\nabla f$ weit von null entfernt ist, gilt Stationarität. Der Gradient der Zielfunktion $(-2.4, -3.2)$ und die Gegenkraft des Randes $(2.4, 3.2)$ addieren sich zu null. Der Multiplikator $\lambda^\star = 4$ gibt das Verhältnis der Länge des Zielfunktionsgradienten zu der des Nebenbedingungsgradienten an.

<!--en:c8995615-->
Auch das Vorzeichen des Multiplikators hat eine geometrische Bedeutung. Da $g(\mathbf{x})$ beim Verlassen der Kugel wächst, zeigt der Gradient $\nabla g = \mathbf{x}$ nach außen. Schreiben wir die Stationarität als $-\nabla f = \lambda\nabla g$ mit $\lambda > 0$, so sehen wir, dass die Abstiegsrichtung $-\nabla f$ nach außen weist und der Rand entgegenwirkt. Würde $-\nabla f$ nach innen zeigen, könnten wir durch einen Schritt ins Kugelinnere $f$ verringern. Der Punkt wäre also nicht optimal. Ein negativer Multiplikator scheidet damit aus. Für $\lambda = -6$ ergäbe sich der entfernte Kreispunkt $(-0.6, -0.8)$, das Maximum auf der Kreisscheibe. Anders als eine Gleichungsnebenbedingung wirkt eine Ungleichungsnebenbedingung nur in eine Richtung.

<!--en:1430fa5b-->
### Die Karush-Kuhn-Tucker-Bedingungen

<!--en:3d991241-->
Kombinieren wir Ungleichungs- und Gleichungsnebenbedingungen, erhalten wir das allgemeine Optimierungsproblem

<!--en:cd35ba8a-->
Seine **Lagrange-Funktion** ordnet jeder Nebenbedingung einen Multiplikator zu. Die Multiplikatoren der Ungleichungen müssen nichtnegativ sein:

<!--en:06da237d-->
An jedem lokalen Minimum $\mathbf{x}^\star$, an dem eine Regularitätsbedingung gilt, gibt es Multiplikatoren, die die vier **Karush-Kuhn-Tucker-Bedingungen (KKT-Bedingungen)** erfüllen :cite:`Karush.1939,Kuhn.Tucker.1951`. Welche Regularitätsbedingungen dafür ausreichen, sehen wir im nächsten Unterabschnitt.

<!--en:abfae1b8-->
Die ersten drei Bedingungen kennen wir aus unseren Beispielen. Die Gradienten stehen im Gleichgewicht, die Nebenbedingungen selbst sind erfüllt, und die Multiplikatoren der Ungleichungen unterliegen einer Vorzeichenbeschränkung. Die vierte Bedingung, der **komplementäre Schlupf** (engl. complementary slackness), besagt, dass für jedes $i$ mindestens eine der beiden Größen $\lambda_i^\star$ und $g_i(\mathbf{x}^\star)$ verschwinden muss. Jede inaktive Nebenbedingung ($g_i(\mathbf{x}^\star) < 0$) erhält somit automatisch den Multiplikator $\lambda_i^\star = 0$ und entfällt in der Stationaritätsgleichung. Die Umkehrung gilt nur eingeschränkt. Ein positiver Multiplikator garantiert eine aktive Nebenbedingung. Eine aktive Nebenbedingung kann aber $\lambda_i^\star = 0$ haben, wenn ihr Rand den Punkt $\mathbf{x}^\star$ berührt, ohne eine Kraft auszuüben. Bei der Projektion auf die Kugel beschreibt der komplementäre Schlupf genau die Fallunterscheidung, die wir von Hand vorgenommen haben. Bei $(0.6, 0.8)$ gilt $\lambda^\star g(\mathbf{x}^\star) = 4 \cdot 0 = 0$.

<!--en:bbd23a40-->
:numref:`fig_mdl-opt-kkt-active-set` veranschaulicht, was Stationarität und duale Zulässigkeit zusammen bedeuten. Im Optimum muss die Abstiegsrichtung $-\nabla f$ eine Linearkombination der äußeren Normalen der aktiven Nebenbedingungen mit nichtnegativen Koeffizienten sein. Gleichbedeutend damit liegt $-\nabla f$ in dem Kegel, den diese äußeren Normalen aufspannen (dem *Normalenkegel*, engl. normal cone). Inaktive Nebenbedingungen tragen nichts bei.

<!--en:426e7ac3-->
![Geometrie der KKT-Bedingungen in einem durch zwei Ungleichungen beschriebenen zulässigen Bereich. In (a) ist bei $\mathbf{x}^\star$ eine Nebenbedingung aktiv und die andere inaktiv. Der Vektor $-\nabla f$ zeigt in Richtung der aktiven äußeren Normalen, mit $\lambda_1 > 0$, während $\lambda_2 = 0$ gilt. In (b) sind an einer Ecke beide Nebenbedingungen aktiv. Daher muss $-\nabla f$ im schattierten Normalenkegel liegen, den beide äußeren Normalen aufspannen. Orangefarbene Bögen zeigen Niveaumengen von $f$ um den Minimierer ohne Nebenbedingungen (+).](../img/mdl-opt-kkt-active-set.svg)

<!--en:98f28c69-->
Wie die Lagrange-Bedingung sind die KKT-Bedingungen Bedingungen erster Ordnung und für sich genommen nicht hinreichend. Minimieren wir etwa $f(x) = -x^2$ auf $[-1, 1]$ mit den Nebenbedingungen $x - 1 \le 0$ und $-1 - x \le 0$. Bei $x = 0$ verschwindet der Gradient, beide Nebenbedingungen sind inaktiv, und mit zwei Multiplikatoren gleich null sind alle vier KKT-Bedingungen erfüllt. Dennoch ist $x = 0$ das *Maximum*. Die tatsächlichen Minima $x = \pm 1$ sind ebenfalls KKT-Punkte, jeweils mit dem Multiplikator $2$ für die aktive Nebenbedingung.

<!--en:3ee3f851-->
### Regularitätsbedingungen und hinreichende Optimalitätsbedingungen

<!--en:7a5416b2-->
Damit stellen sich zwei einander ergänzende Fragen. Wann erfüllt ein lokales Minimum *garantiert* die KKT-Bedingungen? Und wann ist ein KKT-Punkt *garantiert* ein Minimum?

<!--en:166c0f5e-->
Die Antwort auf die erste Frage ähnelt der für Gleichungen. Gilt eine Regularitätsbedingung, muss ein Minimum die KKT-Bedingungen erfüllen. Am gebräuchlichsten ist die *Regularitätsbedingung der linearen Unabhängigkeit* (engl. linear independence constraint qualification, LICQ). Sei $\mathbf{x}^\star$ ein lokales Minimum von :eqref:`eq_mdl-opt-standard-problem`, und seien $f$, $g_i$ und $h_j$ stetig differenzierbar. Betrachten wir die Gradienten aller aktiven Ungleichungen zusammen mit denen aller Gleichungen, also $\{\nabla g_i(\mathbf{x}^\star) : g_i(\mathbf{x}^\star) = 0\} \cup \{\nabla h_j(\mathbf{x}^\star) : j = 1, \ldots, p\}$. Sind diese Gradienten linear unabhängig, gibt es Multiplikatoren, die :eqref:`eq_mdl-opt-kkt` erfüllen (:citet:`Nocedal.Wright.2006`, Kapitel 12). LICQ verallgemeinert unmittelbar die Bedingung $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$. Aufgabe :exref:`ex_mdl-constrained-degenerate-constraints` zeigt, was ohne diese Voraussetzung schiefgehen kann.

<!--en:98df1be5-->
Für konvexe Probleme genügt eine einfachere Prüfung. Wir müssen nur einen einzigen geeigneten zulässigen Punkt finden, ohne die Gradienten an der unbekannten Lösung zu untersuchen. Wir nennen das Problem :eqref:`eq_mdl-opt-standard-problem` *konvex*, wenn $f$ und alle $g_i$ konvex und alle $h_j$ affin sind. Ein konvexes Problem erfüllt die **Slater-Bedingung** :cite:`Slater.1950`, wenn es einen Punkt $\bar{\mathbf{x}}$ gibt, für den gilt:

<!--en:4aad0c89-->
Der Punkt $\bar{\mathbf{x}}$ muss also zulässig sein und jede nichtaffine Ungleichung strikt erfüllen. Sind alle Nebenbedingungen affin, genügt für die Slater-Bedingung bereits die Zulässigkeit. (Ist $f$ oder eines der $g_i$ nur auf einer Teilmenge von $\mathbb{R}^n$ definiert, muss $\bar{\mathbf{x}}$ außerdem im relativen Inneren dieses Definitionsbereichs liegen. Dieser Fall tritt hier nicht auf.) Für die euklidische Kugel eignet sich $\bar{\mathbf{x}} = \mathbf{0}$ unmittelbar, weil $g(\mathbf{0}) = -r^2/2 < 0$ gilt. Erfüllt ein konvexes Problem mit differenzierbaren Funktionen $f$ und $g_i$ die Slater-Bedingung, so ist jedes Minimum ein KKT-Punkt :cite:`Boyd.Vandenberghe.2004`. Der tiefere Grund ist die starke Dualität, die wir in :numref:`subsec_mdl-strong-duality` aus derselben Bedingung herleiten.

<!--en:aa6929d5-->
Konvexität beantwortet auch die zweite Frage und liefert damit das für die Anwendung wichtigste Ergebnis dieses Abschnitts.

<!--en:e34b34f1-->
**Proposition (Hinreichende KKT-Bedingungen bei Konvexität).** *Im Problem :eqref:`eq_mdl-opt-standard-problem` seien $f$ und alle $g_i$ konvex und differenzierbar, alle $h_j$ seien affin. Erfüllt $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ die KKT-Bedingungen :eqref:`eq_mdl-opt-kkt`, so ist $\mathbf{x}^\star$ ein globales Minimum.*

<!--en:5342166d-->
**Beweis.** Setzen wir $\varphi(\mathbf{x}) = \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Wegen $\lambda_i^\star \ge 0$ ist $\varphi$ eine Linearkombination konvexer Funktionen mit nichtnegativen Koeffizienten zuzüglich affiner Terme. Also ist $\varphi$ konvex. Aus der Stationarität folgt $\nabla\varphi(\mathbf{x}^\star) = \mathbf{0}$. Damit ist $\mathbf{x}^\star$ ein *globaler* Minimierer von $\varphi$ auf ganz $\mathbb{R}^n$. Wählen wir nun einen beliebigen zulässigen Punkt $\mathbf{x}$. Wegen $\lambda_i^\star \ge 0$, $g_i(\mathbf{x}) \le 0$ und $h_j(\mathbf{x}) = 0$ gilt

<!--en:9c7ce807-->
Die letzte Gleichheit folgt aus dem komplementären Schlupf ($\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$) und der primalen Zulässigkeit ($h_j(\mathbf{x}^\star) = 0$). $\blacksquare$

<!--en:5fe45b2c-->
In diesem kurzen Beweis steckt ein allgemeineres Prinzip. Für *beliebige* feste Multiplikatoren mit $\boldsymbol{\lambda} \ge 0$ gilt die erste Ungleichung für jeden zulässigen Punkt $\mathbf{x}$. Minimieren wir die Lagrange-Funktion über $\mathbf{x}$, erhalten wir daher stets eine untere Schranke für den Optimalwert unter Nebenbedingungen. Bei den KKT-Multiplikatoren ist diese Schranke exakt. In :numref:`sec_mdl-lagrangian-duality` leiten wir aus dieser Beobachtung die gesamte Dualitätstheorie her. Bei konvexen Problemen sind die KKT-Bedingungen also immer hinreichend. Unter der Slater-Bedingung sind sie auch notwendig und charakterisieren damit die globalen Minima vollständig.

<!--en:1763ee2b-->
## Projektionen und projizierter Gradientenabstieg

<!--en:bf130831-->
Mit den KKT-Bedingungen können wir ein Optimum *erkennen* und bei einfachen Nebenbedingungen von Hand bestimmen. Um Lösungen auch in allgemeineren Fällen zu *berechnen*, kombinieren wir den Gradientenabstieg mit der Suche nach dem nächstgelegenen zulässigen Punkt. Wir machen einen gewöhnlichen Gradientenschritt. Führt er aus dem zulässigen Bereich hinaus, projizieren wir den Punkt wieder hinein.

<!--en:84098937-->
### Projektion auf eine konvexe Menge

<!--en:8c34800f-->
Sei $C \subseteq \mathbb{R}^n$ eine nichtleere abgeschlossene konvexe Menge. Die **Projektion** eines Punktes $\mathbf{y}$ auf $C$ ist der eindeutig bestimmte Punkt aus $C$ mit dem kleinsten Abstand zu $\mathbf{y}$,

<!--en:a4eeda9a-->
Warum gibt es immer genau einen Minimierer? Wir können $C$ mit einer hinreichend großen abgeschlossenen Kugel um $\mathbf{y}$ schneiden, ohne das Optimum auszuschließen. Der Schnitt ist abgeschlossen und beschränkt, also kompakt. Die stetige Zielfunktion nimmt darauf ihr Minimum an. Da der quadrierte Abstand strikt konvex ist, können keine zwei Punkte das Minimum annehmen. Damit ist $\Pi_C$ wohldefiniert.

<!--en:78c5d641-->
Zwei Projektionen kennen wir bereits in geschlossener Form: die Projektion auf eine Hyperebene (der nächstgelegene Punkt) und die auf eine Kugel (die Clipping-Formel). Die Projektion auf einen Halbraum $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \ge b\}$ folgt unmittelbar aus der Formel für die Hyperebene. Erfüllt $\mathbf{y}$ die Ungleichung bereits, bleibt der Punkt unverändert. Andernfalls verschieben wir ihn geradlinig entlang $\mathbf{a}$ auf die begrenzende Hyperebene:

<!--en:a472c276-->
Projektionen auf abgeschlossene konvexe Mengen lassen sich geometrisch besonders einfach charakterisieren. Sei $\hat{\mathbf{x}} = \Pi_C(\mathbf{y})$ der Minimierer von $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ auf $C$. Für jeden anderen Punkt $\mathbf{x} \in C$ liegt wegen der Konvexität die gesamte Strecke von $\hat{\mathbf{x}}$ nach $\mathbf{x}$ in $C$. Wenn wir uns von $\hat{\mathbf{x}}$ auf $\mathbf{x}$ zubewegen, kann die Zielfunktion daher in erster Ordnung nicht abnehmen. Ihre Richtungsableitung ist also nichtnegativ: $(\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}}) \ge 0$. Ein Vorzeichenwechsel liefert die **Ungleichung für den stumpfen Winkel** (auch Variationsungleichung):

<!--en:d655ce84-->
Geometrisch bildet der *Residuenvektor* $\mathbf{y} - \Pi_C(\mathbf{y})$ einen Winkel von mindestens $90^\circ$ mit jedem Vektor von $\Pi_C(\mathbf{y})$ zu einem Punkt $\mathbf{x} \in C$ (:numref:`fig_mdl-opt-projection-inequality`, Teil (a)). Anders ausgedrückt ist die Hyperebene durch $\Pi_C(\mathbf{y})$ mit der Normalen $\mathbf{y} - \Pi_C(\mathbf{y})$ eine *Stützhyperebene* von $C$. Die gesamte Menge $C$ liegt auf der vom Ausgangspunkt abgewandten Seite oder auf der Hyperebene selbst.

<!--en:d2333d33-->
![(a) Das Residuum $\mathbf{y}-\Pi_C(\mathbf{y})$ bildet einen Winkel von mindestens $90^\circ$ mit jeder Richtung in die Menge $C$. Daher liegt $C$ ganz auf der abgewandten Seite der Stützgeraden durch $\Pi_C(\mathbf{y})$. (b) Die Projektion auf eine abgeschlossene konvexe Menge vergrößert Abstände nie: $\|\mathbf{u}-\mathbf{v}\|\le\|\mathbf{x}-\mathbf{x}'\|$.](../img/mdl-opt-projection-inequality.svg)

<!--en:b44cc86a-->
Prüfen wir die Ungleichung an unserem Beispiel. Projizieren wir $\mathbf{y} = (3, 4)$ auf die Einheitskreisscheibe, erhalten wir das Residuum $(2.4, 3.2) = 4\,(0.6, 0.8)$, also wieder das Produkt aus Multiplikator und äußerer Normalen. Für jedes $\mathbf{x}$ mit $\|\mathbf{x}\| \le 1$ gilt

<!--en:9d0550fe-->
wobei die erste Ungleichung aus der Cauchy-Schwarz-Ungleichung $\mathbf{u}^\top\mathbf{v} \le \|\mathbf{u}\|\,\|\mathbf{v}\|$ folgt. Die Stützgerade ist hier $3x_1 + 4x_2 = 5$, also genau die Hyperebene aus unserem Beispiel zum nächstgelegenen Punkt.

<!--en:e201122f-->
Ebenso wichtig ist, dass die Ungleichung für den stumpfen Winkel auch *hinreichend* ist. Ein Punkt $\hat{\mathbf{x}} \in C$ erfülle :eqref:`eq_mdl-opt-proj-variational`. Setzen wir $\varphi(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$. Der Gradient dieser Funktion am Punkt $\hat{\mathbf{x}}$ ist $\hat{\mathbf{x}} - \mathbf{y}$. Da eine konvexe differenzierbare Funktion stets oberhalb ihrer Tangentialebenen liegt (:numref:`sec_mdl-convexity`), gilt $\varphi(\mathbf{x}) \ge \varphi(\hat{\mathbf{x}}) + (\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}})$. Nach Voraussetzung ist der letzte Term für jedes $\mathbf{x} \in C$ nichtnegativ. Also gilt $\varphi(\mathbf{x}) \ge \varphi(\hat{\mathbf{x}})$ auf ganz $C$, womit $\hat{\mathbf{x}} = \Pi_C(\mathbf{y})$ bewiesen ist. Somit *charakterisiert* :eqref:`eq_mdl-opt-proj-variational` die Projektion *vollständig*. Aus der Ungleichung folgen alle Eigenschaften, die wir für Verfahren mit Projektionen brauchen. Zunächst zeigen wir, dass Projektionen Abstände niemals vergrößern.

<!--en:642a6c04-->
**Proposition (Projektionen sind nichtexpansiv).** *Für jede nichtleere abgeschlossene konvexe Menge $C$ und alle $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ gilt*

<!--en:2d947fd4-->
**Beweis.** Setzen wir $\mathbf{u} = \Pi_C(\mathbf{x})$ und $\mathbf{v} = \Pi_C(\mathbf{y})$. Wir wenden :eqref:`eq_mdl-opt-proj-variational` zweimal an, zunächst bei $\mathbf{x}$ mit $\mathbf{v} \in C$, dann bei $\mathbf{y}$ mit $\mathbf{u} \in C$. Das ergibt

<!--en:f0a903c8-->
Addieren wir die beiden Ungleichungen, erhalten wir $(\mathbf{x} - \mathbf{y} - (\mathbf{u} - \mathbf{v}))^\top(\mathbf{v} - \mathbf{u}) \le 0$. Durch Umstellen folgt

<!--en:13a2457c-->
wobei wir die Cauchy-Schwarz-Ungleichung verwendet haben. Division durch $\|\mathbf{u} - \mathbf{v}\|$ schließt den Beweis ab (für $\|\mathbf{u} - \mathbf{v}\| = 0$ gilt die Behauptung unmittelbar). $\blacksquare$

<!--en:002a81d0-->
Teil (b) von :numref:`fig_mdl-opt-projection-inequality` veranschaulicht das Ergebnis. Zwei Punkte außerhalb der Kreisscheibe liegen mindestens so weit auseinander wie ihre Projektionen. Die Konvexität von $C$ ist dabei wesentlich. Betrachten wir die nichtkonvexe Zweipunktmenge $C = \{-1, 1\}$. Dort wird $y = 0.01$ auf $1$ projiziert und $y = -0.01$ auf $-1$. Aus einem Abstand von $0.02$ wird ein Abstand von $2$. Bei $y = 0$ ist die Projektion nicht einmal eindeutig. Die Projektion auf eine nichtkonvexe Menge kann also sowohl unstetig als auch mehrwertig sein.

<!--en:a93e0f23-->
### Projizierter Gradientenabstieg

<!--en:6b02b539-->
Beim **projizierten Gradientenabstieg** (engl. projected gradient descent, PGD) wird die Projektion zum Bestandteil eines iterativen Lösungsverfahrens. Wir projizieren nach jedem Gradientenschritt:

<!--en:5fab0a77-->
Damit ist jeder Iterationspunkt zulässig. Mehr noch: Die Fixpunkte von PGD sind genau die stationären Punkte unter den Nebenbedingungen. Sei $f$ differenzierbar, $C$ abgeschlossen und konvex und $\eta > 0$ die Schrittweite. Dann gilt die Fixpunktgleichung $\mathbf{x}^\star = \Pi_C(\mathbf{x}^\star - \eta \nabla f(\mathbf{x}^\star))$ genau dann, wenn $\mathbf{x}^\star \in C$ die Ungleichung für den stumpfen Winkel :eqref:`eq_mdl-opt-proj-variational` mit $\mathbf{y} = \mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star)$ erfüllt. Kürzen wir $\mathbf{x}^\star$ heraus und teilen durch $\eta > 0$, so wird daraus

<!--en:e01cc0fe-->
Mit anderen Worten: Keine zulässige Richtung ist eine Abstiegsrichtung. Das ist die allgemeine Bedingung erster Ordnung für beschränkte Optimierung. Sie vereinfacht sich zu $\nabla f(\mathbf{x}^\star) = \mathbf{0}$, wenn $C = \mathbb{R}^n$ gilt. Bei konvexem $f$ folgt aus der Tangentialebenenungleichung unmittelbar, dass die Bedingung für ein globales Minimum hinreicht. Bei nichtkonvexem $f$ weist sie Stationarität nach. Gilt $C = \{\mathbf{x} : g_i(\mathbf{x}) \le 0\}$ für differenzierbare konvexe $g_i$ und ist die Slater-Bedingung erfüllt, so ist sie zu den KKT-Bedingungen äquivalent.

<!--en:8764d31a-->
Verlangsamt die zusätzliche Projektion den Gradientenabstieg? Erstaunlicherweise nicht. Dank der Nichtexpansivität genügen drei Schritte für den Beweis.

<!--en:7a541b5f-->
**Proposition (PGD kontrahiert mindestens so schnell wie der Gradientenabstieg ohne Nebenbedingungen).** *Sei $C$ nichtleer, abgeschlossen und konvex. Die Funktion $f$ sei zweimal stetig differenzierbar mit $\mu\mathbf{I} \preceq \nabla^2 f \preceq L\mathbf{I}$ für gewisse $0 < \mu \le L$. Sei $\mathbf{x}^\star$ der Minimierer von $f$ auf $C$, und es gelte $\eta > 0$. Dann gilt für die Iterationspunkte von :eqref:`eq_mdl-opt-pgd`*

<!--en:203438c2-->
**Beweis.** (i) Da $\mathbf{x}^\star$ ein Fixpunkt von PGD ist, können wir $\mathbf{x}^\star = \Pi_C(\mathbf{b})$ mit $\mathbf{b} = \mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star)$ schreiben. (ii) Diese Wahl von $\mathbf{b}$ ist entscheidend. Wenden wir die Nichtexpansivität auf $\mathbf{x}_k - \eta\nabla f(\mathbf{x}_k)$ und $\mathbf{b}$ an, entfällt die Projektion vollständig:

<!--en:ad13a624-->
(iii) Nach dem Hauptsatz der Differential- und Integralrechnung gilt entlang der Strecke von $\mathbf{x}^\star$ nach $\mathbf{x}_k$ die Gleichung $\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star) = \bar{\mathbf{H}}(\mathbf{x}_k - \mathbf{x}^\star)$ mit $\bar{\mathbf{H}} = \int_0^1 \nabla^2 f(\mathbf{x}^\star + t(\mathbf{x}_k - \mathbf{x}^\star))\,dt$. Diese gemittelte Hesse-Matrix ist symmetrisch, und alle ihre Eigenwerte liegen in $[\mu, L]$. Jeder Eigenwert von $\mathbf{I} - \eta\bar{\mathbf{H}}$ hat daher die Form $1 - \eta h$ mit $h \in [\mu, L]$ und dem Betrag höchstens $\rho$. Die rechte Seite ist folglich höchstens $\rho\,\|\mathbf{x}_k - \mathbf{x}^\star\|$. $\blacksquare$

<!--en:79b0690d-->
Nachdem in Schritt (ii) die Projektion entfallen ist, bleibt nur die Kontraktion des Gradientenabstiegs ohne Nebenbedingungen. Für eine quadratische Funktion $f = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ gilt $\bar{\mathbf{H}} = \mathbf{A}$. Schritt (iii) liefert dann für jede Eigenkomponente genau den Faktor $|1 - \eta\lambda_i(\mathbf{A})|$ aus :numref:`sec_mdl-gradient-based-optimization`. Dieser ist strikt kleiner als $1$, sofern $0 < \eta < 2/L$ gilt. Ist die konvexe, $L$-glatte Funktion $f$ nicht stark konvex, behält PGD mit $\eta = 1/L$ ebenfalls die Rate $O(1/k)$ des Gradientenabstiegs für den Zielfunktionswert bei :cite:`Nesterov.2018`.

<!--en:732dfb84-->
Wie scharf ist der Kontraktionsfaktor $\rho$ in der Praxis? Betrachten wir die schlecht konditionierte quadratische Funktion $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ mit Konditionszahl $\kappa = 10^3$. Wir minimieren sie auf dem Halbraum $x_1 + x_2 \ge 1$ (geschrieben als $g(\mathbf{x}) = 1 - x_1 - x_2 \le 0$). Die KKT-Bedingungen lassen sich von Hand lösen. Der Minimierer ohne Nebenbedingung $\mathbf{0}$ ist unzulässig, also muss die Nebenbedingung aktiv sein. Die Stationarität $(x_1, 1000\,x_2) - \lambda(1, 1) = \mathbf{0}$ liefert $x_1 = \lambda$ und $x_2 = \lambda/1000$. Setzen wir beides in $x_1 + x_2 = 1$ ein, erhalten wir $\lambda^\star = 1000/1001 \approx 0.999$, $\mathbf{x}^\star = (1000/1001,\; 1/1001) \approx (0.999, 0.001)$ und $p^\star = 500/1001 \approx 0.4995$.

<!--en:91a77350-->
Führen wir PGD nun mit der Schrittweite $\eta = 0.0015$ aus, die deutlich unter der Stabilitätsgrenze $2/L = 0.002$ liegt. Die Proposition garantiert pro Schritt einen Faktor von höchstens $\rho = \max(1 - \eta,\ |1 - 1000\,\eta|) = 0.9985$. Starten wir bei $(3, 4)$ im Abstand $4.47$ von $\mathbf{x}^\star$, so genügen nach dieser Schranke höchstens $\log(4.47 \times 10^6)/\log(1/\rho) \approx 10{,}200$ Schritte, um den Abstand $10^{-6}$ zu erreichen. Der folgende Code führt PGD aus und ermittelt das größte Abstandsverhältnis eines einzelnen Schritts sowie die tatsächliche Schrittzahl. Außerdem untersuchen wir, ab wann jeder Schritt den Rand erreicht, wie schnell der Abstand entlang des Randes schrumpft und welchen Wert der Gradient am Ende hat.

<!--en:c50285cf-->
Das größte Abstandsverhältnis eines Schritts, $0.9977$, liegt nahe bei $\rho = 0.9985$. Die Schranke pro Schritt ist also nahezu scharf. Die Gesamtzahl der Schritte fällt jedoch überraschend klein aus: $738$ statt $10{,}200$. Die ausgegebenen Werte zeigen, warum. Die Trajektorie durchläuft zwei Phasen. In der ersten Phase bleibt die Nebenbedingung nach dem Anfangsschritt inaktiv. Die Koordinate $x_2$ in der steilen Richtung schrumpft innerhalb weniger Dutzend Iterationen fast auf null. Die Koordinate $x_1$ in der flachen Richtung bewegt sich wie beim Gradientenabstieg ohne Nebenbedingungen langsam mit dem Faktor $1 - \eta$ pro Schritt nach innen. Von $x_1 = 3$ aus dauert es etwa $\log 3 / \eta \approx 730$ Schritte, bis wir die Randgerade erreichen. Ab Schritt $733$ trifft jeder Aktualisierungsschritt auf den Rand und wird auf die Gerade $x_1 + x_2 = 1$ projiziert. Entlang der Geraden hat die quadratische Funktion die Krümmung $(1 + 1000)/2 = 500.5$. Der Abstand zu $\mathbf{x}^\star$ nimmt daher pro Schritt um den Faktor $1 - 500.5\,\eta = 0.249$ ab, genau mit der Rate des eindimensionalen Gradientenabstiegs entlang der Geraden. Schon nach wenigen weiteren Schritten endet der Lauf. Während $\rho$ die ungünstigste Krümmung in der gesamten Ebene berücksichtigen muss, zählt bei aktiver Nebenbedingung nur die Krümmung *entlang des Randes*. Betrachten wir schließlich den Gradienten am Ende: $(0.999, 0.999) = \lambda^\star(1, 1) = -\lambda^\star\nabla g$. Er ist weit von null entfernt und liefert uns wieder den von Hand berechneten Multiplikator $\lambda^\star = 1000/1001$.

<!--en:1be411be-->
PGD kommt häufig zum Einsatz, wenn sich $\Pi_C$ mit geringem Aufwand berechnen lässt. Bei Nichtnegativitätsbedingungen projizieren wir auf den nichtnegativen Orthanten, indem wir negative Einträge auf null setzen. Bei einer Obergrenze für die Norm der Zeilen einer Gewichtsmatrix skalieren wir jede Zeile nach jedem Schritt bei Bedarf auf den Rand einer Kugel. Die wohl bekannteste Anwendung im Deep Learning ist die Erzeugung adversarieller Beispiele durch projizierten Gradientenaufstieg bezüglich einer Eingabestörung $\boldsymbol{\delta}$ mit $\|\boldsymbol{\delta}\|_\infty \le \epsilon$. Die Projektion begrenzt dabei jeden Eintrag auf $[-\epsilon, \epsilon]$ :cite:`Madry.Makelov.Schmidt.ea.2018`. Auch Gradienten-Clipping verwendet eine Kugelprojektion, projiziert aber den *Gradientenvektor* und nicht die Parameter. Es entspricht also keinem PGD für eine Nebenbedingung an die Gewichte (Aufgabe :exref:`ex_mdl-constrained-projected-fixed-points` verdeutlicht den Unterschied). Das Clipping liefert einen stabilen Schritt, wenn die Krümmung mit dem Gradienten wächst (:numref:`subsec_mdl-relaxed-smoothness`).

<!--en:09ad72b6-->
### Projektion auf den Simplex

<!--en:80593b7c-->
Attention-Gewichte und Klassenwahrscheinlichkeiten liegen auf dem **Wahrscheinlichkeitssimplex** $\Delta = \{\mathbf{x} : \mathbf{x} \ge 0,\; \sum_i x_i = 1\}$. Wie finden wir zu einem Vektor reeller Scores $\mathbf{y}$ den nächstgelegenen Punkt auf $\Delta$? Das Problem

<!--en:e2a099d1-->
ist ein kleines **quadratisches Programm** (QP) mit konvexer quadratischer Zielfunktion und affinen Nebenbedingungen. Mit den KKT-Bedingungen müssen wir nur noch einen einzigen Skalar bestimmen. Führen wir einen Multiplikator $\tau$ für die Summenbedingung und Multiplikatoren $\lambda_i \ge 0$ für die Nichtnegativitätsbedingungen $-x_i \le 0$ ein. Die Stationarität ergibt $x_i - y_i + \tau - \lambda_i = 0$, und der komplementäre Schlupf teilt die Koordinaten in zwei Gruppen. Wo $x_i > 0$ gilt, muss $\lambda_i = 0$ und damit $x_i = y_i - \tau$ gelten. Wo $x_i = 0$ gilt, liefert die Stationarität $\lambda_i = \tau - y_i$, und aus der dualen Zulässigkeit $\lambda_i \ge 0$ folgt $y_i \le \tau$. Zusammen ergibt das

<!--en:a7567159-->
Zur Projektion auf den Simplex ziehen wir also von jedem Score denselben Schwellenwert $\tau$ ab und setzen negative Ergebnisse auf null. Bemerkenswert ist, dass diese Formel für *jeden* Wert von $\tau$ automatisch Stationarität, komplementären Schlupf, duale Zulässigkeit und Nichtnegativität gewährleistet. Allein die Summenbedingung $\sum_i x_i = 1$ legt $\tau$ fest. Die linke Seite $\varphi(\tau) = \sum_i \max(y_i - \tau, 0)$ ist stetig, stückweise linear und überall dort streng fallend, wo sie positiv ist. Daher gibt es genau ein $\tau$, das $\varphi(\tau) = 1$ löst.

<!--en:511c9c0a-->
Eine kurze Rechnung zeigt, wie wir $\tau$ finden. Betrachten wir den bereits sortierten Vektor $\mathbf{y} = (0.9, 0.5, -0.2)$ und nehmen wir an, dass die $k$ größten Scores nach der Projektion positiv bleiben. Dann liefert die Summenbedingung $\sum_{j \le k}(y_j - \tau) = 1$ den möglichen Schwellenwert $\tau_k = (\sum_{j \le k} y_j - 1)/k$. Er ist nur dann mit der Annahme vereinbar, wenn der $k$-te Score strikt über $\tau_k$ liegt:

<!--en:4174c0d5-->
* $k = 1$: $\tau_1 = -0.1$ und $y_1 = 0.9 > \tau_1$,

<!--en:458a2c11-->
* $k = 2$: $\tau_2 = 0.2$ und $y_2 = 0.5 > \tau_2$,

<!--en:ab7ce4bb-->
* $k = 3$: $\tau_3 \approx 0.067$, aber $y_3 = -0.2 < \tau_3$.

<!--en:caabe2a3-->
Der größte gültige Wert ist $k = 2$. Damit erhalten wir $\tau = 0.2$ und $\mathbf{x}^\star = (0.7, 0.3, 0)$ (:numref:`fig_mdl-opt-simplex-threshold`). Zum Vergleich liefert die Softmax-Funktion für dieselben Scores $(0.50, 0.33, 0.17)$. Hier bleibt jeder Eintrag strikt positiv.

<!--en:88237bd5-->
![(a) Projektion von $\mathbf{y}=(0.9,0.5,-0.2)$ auf den Wahrscheinlichkeitssimplex. Von jedem Score ziehen wir den Schwellenwert $\tau=0.2$ ab und begrenzen das Ergebnis nach unten auf null. Dabei ist $\tau$ durch $\sum_i x_i=1$ eindeutig bestimmt. (b) Bei der Softmax-Funktion bleiben für dieselben Scores alle Koordinaten strikt positiv.](../img/mdl-opt-simplex-threshold.svg)

<!--en:4054936f-->
Zwei Beobachtungen führen von dieser Rechnung zu einem allgemeinen Algorithmus. Erstens umfasst der *Träger* $\{i : x_i^\star > 0\}$ stets die Indizes der größten Scores. Gilt $y_i > y_j$ und $x_j^\star > 0$, so folgt $x_i^\star \ge y_i - \tau > y_j - \tau = x_j^\star > 0$. (Die *aktiven* Vorzeichenbedingungen gehören zu den übrigen, auf null gesetzten Koordinaten.) Zweitens gilt nach dem Sortieren der Scores gemäß $u_1 \ge \cdots \ge u_n$ die Testbedingung $u_k > \tau_k$ für jedes $k$ bis zur tatsächlichen Größe des Trägers und danach nicht mehr. Stellen wir $u_k > \tau_k$ um, erhalten wir $\sum_{j < k}(u_j - u_k) < 1$, also gerade $\varphi(u_k) < 1 = \varphi(\tau)$. Da $\varphi$ im positiven Bereich streng fällt, ist dies äquivalent zu $u_k > \tau$. Wir sortieren also $\mathbf{y}$, wählen das größte $k$ mit $u_k > \tau_k$ und setzen $\tau = \tau_k$. Die Laufzeit beträgt $O(n \log n)$ :cite:`Held.Wolfe.Crowder.1974,Duchi.Shalev-Shwartz.Singer.ea.2008`.

<!--en:460011a7-->
Bei Attention-Scores heißt diese Simplexprojektion *Sparsemax* :cite:`Martins.Astudillo.2016`, weil Positionen mit niedrigen Scores exakt das Gewicht null erhalten. Die Softmax-Funktion und Sparsemax unterscheiden sich nur in einem Regularisierungsterm. Multiplizieren wir den quadrierten Abstand aus, sehen wir, dass Sparsemax $\mathbf{y}^\top\mathbf{x} - \tfrac12\|\mathbf{x}\|^2$ auf $\Delta$ maximiert. Die Softmax-Funktion maximiert dagegen $\mathbf{y}^\top\mathbf{x} + H(\mathbf{x})$ und ersetzt so den quadratischen Strafterm durch die Shannon-Entropie (:numref:`sec_mdl-lagrangian-duality`).

<!--en:8734eafe-->
Wenn wir eine Projektion implementieren, sollten wir ihr Ergebnis prüfen. Als erster Test bieten sich die KKT-Residuen an. Dabei kommt es darauf an, welche Residuen eine falsche Lösung tatsächlich erkennen. Die folgende Codezelle prüft sechs zufällige Scores sowohl beim berechneten $\tau$ als auch bei einem gestörten Wert $\tau + 0.3$. Anschließend prüft sie das Optimalitätskriterium aus der Ungleichung für den stumpfen Winkel :eqref:`eq_mdl-opt-proj-variational`. Die linke Seite von :eqref:`eq_mdl-opt-proj-variational` ist affin in $\mathbf{x}$ und nimmt ihr Maximum auf $\Delta$ daher an einer Ecke an. Es genügt also, die $n$ Standardbasisvektoren $\mathbf{e}_i$ zu prüfen, um die Optimalität auf dem gesamten Simplex nachzuweisen.

<!--en:b2effafc-->
Meist werden drei bis fünf der sechs Koordinaten exakt null. Die entsprechenden Nichtnegativitätsbedingungen sind aktiv und haben jeweils den positiven Multiplikator $\tau - y_i$. Hier ist bei der Diagnose Vorsicht geboten. Stationarität, komplementärer Schlupf und duale Zulässigkeit sind bis auf Rundungsfehler *sowohl* beim richtigen $\tau$ als auch beim falschen Wert erfüllt. Nur die Summenbedingung deckt den Fehler auf. Die drei Residuen allein können die Lösung also nicht bestätigen. Die Prüfung an den Ecken kann das: Gilt $\max_i (\mathbf{y} - \mathbf{x}^\star)^\top(\mathbf{e}_i - \mathbf{x}^\star) \le 0$ bis auf Rundungsfehler, liegt kein Punkt auf dem Simplex näher an $\mathbf{y}$.

<!--en:1529c3d8-->
## Proximaloperatoren und Koordinatenabstieg

<!--en:3f65a794-->
### Weiches Schwellenwertverfahren und Proximaloperator

<!--en:c18103f7-->
Um Gewichte zu begrenzen, brauchen wir nicht unbedingt eine feste Schranke. Das **Lasso** :cite:`Tibshirani.1996` ersetzt sie durch einen $\ell_1$-Strafterm mit Knick, den wir zum Verlust der kleinsten Quadrate addieren,

<!--en:3f97adc0-->
wobei der Minimierer typischerweise viele Koordinaten exakt auf null setzt. Wir bezeichnen das Gewicht des Strafterms bewusst mit $\lambda$, denn es wirkt wie der Lagrange-Multiplikator einer Normbeschränkung $\|\mathbf{w}\|_1 \le c$. Jeder Minimierer von $F$ minimiert auch den Verlust der kleinsten Quadrate unter der $\ell_1$-Schranke $c$, die er ausschöpft. Umgekehrt entspricht jeder Schranke $c > 0$ ein Strafgewicht $\lambda$. In :numref:`subsec_mdl-weight-decay-duality` beweisen wir beide Richtungen sowohl für das Lasso als auch für den $\ell_2$-Gewichtszerfall.

<!--en:97537f1d-->
Untersuchen wir das numerisch. Wir geben einen dünn besetzten Vektor $\mathbf{w}$ in $\mathbb{R}^8$ mit fünf Nullkoordinaten vor, erzeugen $n = 100$ verrauschte lineare Beobachtungen und setzen $\lambda = 3$. Als Schrittweite wählen wir $\eta = 1/L$. Dabei ist $L = \lambda_{\max}(\mathbf{A}^\top\mathbf{A})$ der größte Eigenwert von $\mathbf{A}^\top\mathbf{A}$ und zugleich die Glattheitskonstante des Terms der kleinsten Quadrate.

<!--en:06eb0212-->
Hier liegt $L$ meist zwischen $130$ und $190$, während die quadrierten Spaltennormen $\|\mathbf{a}_j\|^2$ von $\mathbf{A}$ ungefähr $100$ betragen.

<!--en:d8237e1b-->
Als erster Versuch bietet sich der Subgradientenabstieg an. Wie wir aus :numref:`sec_mdl-convexity` wissen, sind die Subgradienten von $|w|$ durch $\mathrm{sign}(w)$ für $w \neq 0$ gegeben. Sie umfassen das gesamte Intervall $[-1, 1]$ am Knick $w = 0$. Wählen wir in jedem Schritt $\mathrm{sign}(\mathbf{w})$, erhalten wir die Aktualisierung $\mathbf{w} \leftarrow \mathbf{w} - \eta\left(\mathbf{A}^\top(\mathbf{A}\mathbf{w} - \mathbf{b}) + \lambda\,\mathrm{sign}(\mathbf{w})\right)$. Finden wir damit die fünf Nullen wieder?

<!--en:bd403d4e-->
Das Verfahren erzeugt keine einzige exakte Null. Keine Koordinate trifft null, und die fünf *Nullkoordinaten* (deren vorgegebener Wert null war) wechseln bei fast jeder Iteration das Vorzeichen. Sie pendeln mit einer Amplitude von etwa $\eta\lambda$ um null. Der Grund ist der Knick bei null. An der tatsächlichen Lasso-Lösung ist der Betrag des glatten Gradienten in einer Nullkoordinate strikt kleiner als $\lambda$. In der Nähe von null dominiert also der Strafterm $-\eta\lambda\,\mathrm{sign}(w_j)$ den Schritt und stößt die Koordinate jedes Mal um etwa $\eta\lambda$ über null hinaus. Der Subgradientenabstieg legt sich bei jedem Schritt auf eine einzelne Steigung aus dem Intervall $[-\lambda, \lambda]$ fest. Bei fester Schrittweite kann er sich daher nicht genau auf die Steigung einpendeln, die den glatten Gradienten ausgleicht.

<!--en:2a6d8839-->
Das lässt sich beheben, indem wir den Strafterm mit Knick *exakt* behandeln, statt ihn zu linearisieren. Beginnen wir in einer Dimension und minimieren wir $\lambda|x| + \tfrac12(x - z)^2$ über $x \in \mathbb{R}$. Da die Zielfunktion konvex ist, ist $x$ genau dann optimal, wenn $0$ zur Menge ihrer Subgradienten gehört:

<!--en:2b775dc5-->
* Gilt $x \neq 0$, ist die Zielfunktion bei $x$ differenzierbar, also gilt $\lambda\,\mathrm{sign}(x) + x - z = 0$. Für $x > 0$ folgt daraus $x = z - \lambda$, was $z > \lambda$ voraussetzt. Für $x < 0$ folgt $x = z + \lambda$, was $z < -\lambda$ voraussetzt.

<!--en:1e8a04c0-->
* Gilt $x = 0$, bilden die Subgradienten das Intervall $[-\lambda, \lambda] - z$. Es enthält $0$ genau dann, wenn $|z| \le \lambda$ gilt.

<!--en:88aca6e5-->
Diese beiden Fälle zerlegen die reelle Zahlengerade. Zusammen ergeben sie das **weiche Schwellenwertverfahren**,

<!--en:4cb4e898-->
Jede Eingabe mit Betrag höchstens $\lambda$ wird exakt null. Größere Beträge verringern sich um $\lambda$. Für $\lambda = 0.7$ etwa wird $z = 0.5$ auf $0$ abgebildet, $z = 2$ auf $1.3$ und $z = -2$ auf $-1.3$ (:numref:`fig_mdl-opt-soft-threshold`). Gerade diese Totzone $[-\lambda, \lambda]$ sorgt dafür, dass die $\ell_1$-Regularisierung exakte Sparsität erzeugt. Die $\ell_2$-Regularisierung hingegen verkleinert Gewichte nur, ohne sie jemals auf null zu setzen.

<!--en:124f61ce-->
![Weiches Schwellenwertverfahren $\mathrm{prox}_{\lambda|\cdot|}(z)$ für $\lambda = 0.7$ (durchgezogen). Eingaben in der Totzone $[-\lambda, \lambda]$ werden auf $0$ abgebildet, etwa $z = 0.5$. Jede andere Eingabe rückt in Richtung $0$ um den Betrag $\lambda$, etwa $z = 2$. Die gestrichelte Gerade ist die Identität. Die gepunktete Gerade zeigt die entsprechende Abbildung für den quadratischen Strafterm $\tfrac12 x^2$. Sie halbiert jede Eingabe und bildet nur $z = 0$ auf null ab.](../img/mdl-opt-soft-threshold.svg)

<!--en:ba42acd8-->
Das weiche Schwellenwertverfahren ist ein Spezialfall einer allgemeineren Operation. Der **Proximaloperator** einer Funktion $\psi$ wägt zwei Ziele gegeneinander ab: den Wert von $\psi(\mathbf{x})$ zu verringern und nahe an einem Referenzpunkt $\mathbf{z}$ zu bleiben.

<!--en:6ceb6e5b-->
Diese Abbildung ist für jede *eigentliche, abgeschlossene, konvexe* Funktion $\psi$ wohldefiniert. Das heißt, die Funktion ist konvex, nimmt Werte in $\mathbb{R} \cup \{+\infty\}$ an, ist an mindestens einer Stelle endlich und hat einen abgeschlossenen Epigraphen. Unter diesen Bedingungen ist die Zielfunktion in :eqref:`eq_mdl-opt-prox` eine $1$-stark konvexe Funktion und hat daher genau einen Minimierer :cite:`Parikh.Boyd.2014`.

<!--en:93a4d220-->
Drei Spezialfälle zeigen, wie der Proximaloperator bereits bekannte Ideen verbindet:

<!--en:88cc9f1d-->
* Für $\psi = 0$ ist der Proximaloperator die identische Abbildung.

<!--en:3c483480-->
* Ist $\psi$ die *Indikatorfunktion* einer abgeschlossenen konvexen Menge $C$ (mit Wert $0$ auf $C$ und $+\infty$ außerhalb), muss jeder Minimierer in $C$ liegen. Der Proximaloperator ist dann die euklidische Projektion $\mathrm{prox}_\psi = \Pi_C$. Proximaloperatoren verallgemeinern also Projektionen von Mengen auf Funktionen.

<!--en:d11e8a0c-->
* Für $\psi = \lambda\|\cdot\|_1$ zerfällt die Zielfunktion in unabhängige Terme für die einzelnen Koordinaten. Der Proximaloperator wendet daher das weiche Schwellenwertverfahren :eqref:`eq_mdl-opt-soft-threshold` auf jede Koordinate einzeln an.

<!--en:a61c63d7-->
In Aufgabe :exref:`ex_mdl-constrained-group-lasso-prox` bestimmen Sie den Proximaloperator der euklidischen Norm, der einen ganzen Block von Gewichten auf einmal auf null setzt. Aufgabe :exref:`ex_mdl-convexity-moreau-decomposition` zeigt, dass die Summe aus dem Proximaloperator einer Norm und der Projektion auf die zugehörige duale Einheitskugel die Identität ergibt.

<!--en:62e1f35c-->
### Proximaler Gradientenabstieg

<!--en:bdb3a728-->
Die Lasso-Zielfunktion hat die zusammengesetzte Form $F = s + \psi$. Sie verbindet einen glatten Verlust $s$ mit einem nichtglatten Strafterm $\psi$, dessen Proximaloperator sich mit geringem Aufwand auswerten lässt. Sei $s$ konvex mit $L$-Lipschitz-stetigem Gradienten, sei $\psi$ eigentlich, abgeschlossen und konvex, und es gelte $0 < \eta \le 1/L$. Nach dem Abstiegslemma (:numref:`sec_mdl-gradient-based-optimization` mit $1/\eta \ge L$) ist $s$ nach oben durch das quadratische Modell $s(\mathbf{x}_k) + \nabla s(\mathbf{x}_k)^\top(\mathbf{x} - \mathbf{x}_k) + \tfrac{1}{2\eta}\|\mathbf{x} - \mathbf{x}_k\|^2$ beschränkt. Statt auch $\psi$ zu linearisieren, minimieren wir diese quadratische obere Schranke für $s$ zusammen mit dem *exakten* Strafterm $\psi$. Durch quadratische Ergänzung erhalten wir den Minimierer

<!--en:35f13d69-->
Das ist ein gewöhnlicher Gradientenschritt für den glatten Anteil, gefolgt von einem Proximalschritt für den Anteil mit Knick. Wiederholen wir diesen Schritt,

<!--en:e22965fb-->
so erhalten wir den **proximalen Gradientenabstieg**. Für $\psi = \lambda\|\cdot\|_1$ heißt er **ISTA** (Iterative Shrinkage-Thresholding Algorithm). Er umfasst sowohl den projizierten Gradientenabstieg (mit $\psi$ als Indikatorfunktion von $C$) als auch den gewöhnlichen Gradientenabstieg (mit $\psi = 0$).

<!--en:ede157e7-->
Obwohl $F$ nicht differenzierbar ist, erreicht der proximale Gradientenabstieg mit $\eta = 1/L$ dieselbe Rate $O(1/k)$ wie der Gradientenabstieg für glatte Funktionen: $F(\mathbf{x}_k) - F^\star \le L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2/(2k)$ (:citet:`Beck.Teboulle.2009`, Satz 3.1). Ist $s$ zudem zweimal stetig differenzierbar und $\mu$-stark konvex ($\mu\mathbf{I} \preceq \nabla^2 s$), können wir den Kontraktionsbeweis für PGD wörtlich übernehmen. Er braucht nur zwei Eigenschaften: $\mathbf{x}^\star$ ist ein Fixpunkt, und $\mathrm{prox}_{\eta\psi}$ ist ebenso nichtexpansiv wie $\Pi_C$ (siehe den Beweis in Aufgabe :exref:`ex_mdl-convexity-accelerated-proximal-gradient`).

<!--en:d427314f-->
Beim Lasso ist der Proximalschritt das weiche Schwellenwertverfahren mit Schwellenwert $\eta\lambda$. Jede Koordinate, deren Abstand zu null höchstens $\eta\lambda$ beträgt, wird unmittelbar null. Dieser Schwellenwert entspricht genau der Amplitude, mit der der Subgradientenabstieg um null pendelte. ISTA sollte die Nullkoordinaten daher exakt bei null halten. Prüfen wir das.

<!--en:fce9cfda-->
Nach denselben fünfzig Iterationen findet ISTA bei gleichem Aufwand pro Schritt wie beim Subgradientenabstieg den Träger exakt wieder. Alle fünf Nullkoordinaten sind null, die drei aktiven Koordinaten bleiben von null verschieden. Diese drei Einträge weichen um etwa $0.05$ oder weniger von den vorgegebenen Werten ab. Wesentlich näher können sie ihnen nicht kommen, denn der $\ell_1$-Strafterm verschiebt selbst die tatsächliche Lasso-Lösung um etwa $\lambda/\|\mathbf{a}_j\|^2 \approx 0.03$ in Richtung null.

<!--en:4d9571de-->
### Koordinatenabstieg

<!--en:d78978af-->
Bei der ersten Lektüre können Sie diesen Unterabschnitt überspringen. Der Koordinatenabstieg (engl. coordinate descent) ist das Standardverfahren für das Lasso, beim Training neuronaler Netze kommt er jedoch nicht zum Einsatz.

<!--en:d357e652-->
Minimieren wir beim Lasso entlang einer einzigen Koordinate $w_j$ und halten alle anderen fest, erhalten wir genau das eben gelöste eindimensionale Problem. Sei $\mathbf{r}_{-j} = \mathbf{b} - \sum_{i \neq j}\mathbf{a}_i w_i$ das ohne die Koordinate $j$ berechnete Residuum. Das $j$-te Teilproblem $\tfrac12\|\mathbf{a}_j w_j - \mathbf{r}_{-j}\|^2 + \lambda|w_j|$ stimmt bis auf eine additive Konstante mit $\|\mathbf{a}_j\|^2\left(\tfrac12(w_j - c_j/\|\mathbf{a}_j\|^2)^2 + (\lambda/\|\mathbf{a}_j\|^2)|w_j|\right)$ überein, wobei $c_j = \mathbf{a}_j^\top\mathbf{r}_{-j}$ gilt. Das weiche Schwellenwertverfahren mit Schwellenwert $\lambda/\|\mathbf{a}_j\|^2$ liefert die Lösung in geschlossener Form:

<!--en:c804bd73-->
Hat die Korrelation $c_j$ eines Merkmals mit dem aktuellen Residuum einen Betrag von höchstens $\lambda$, wird sein Gewicht sofort null. Die Koordinaten zyklisch auf diese Weise zu aktualisieren, ist der Algorithmus hinter dem verbreiteten Paket `glmnet` :cite:`Friedman.Hastie.Tibshirani.2010`. Passen wir das Residuum nach jeder Koordinatenänderung schrittweise an, kostet ein vollständiger Durchlauf über alle $d$ Koordinaten so viel wie eine einzige Gradientenauswertung.

<!--en:e2cf5614-->
Dasselbe Prinzip gilt für jede differenzierbare Zielfunktion $f$ mit koordinatenweisen Glattheitskonstanten $L_j$,

<!--en:a3a5856d-->
wobei die Minimierung der quadratischen oberen Schranke über $t$ den Schritt des **Koordinatenabstiegs** ergibt:

<!--en:fef6bcd5-->
Da $L_j$ die Krümmung entlang einer einzelnen Achse misst, kann sie wesentlich kleiner sein als die globale Glattheitskonstante $L$. Das erlaubt größere Schritte. In unserem Lasso-Problem gilt $L_j = \|\mathbf{a}_j\|^2 \approx 100$ gegenüber $L \in [130, 190]$. Für eine positiv definite quadratische Funktion $f(\mathbf x)=\tfrac12\mathbf x^\top\mathbf{A}\mathbf x-\mathbf b^\top\mathbf x$ ist die obere Schranke mit $L_j = A_{jj}$ exakt, und der Schritt

<!--en:af85d9c1-->
minimiert $f$ entlang der Koordinate $j$ exakt. Durchlaufen wir alle Koordinaten zyklisch, erhalten wir das klassische Gauß-Seidel-Verfahren für $\mathbf{A}\mathbf x = \mathbf b$.

<!--en:3ac6ec87-->
Wie viel schneller ist ein Durchlauf über alle Koordinaten als ein gleich teurer ISTA-Schritt? Zählen wir für unser Lasso-Problem, wie viele Datendurchläufe jedes Verfahren benötigt, bis die Differenz $F(\mathbf{w}) - F^\star$ zum optimalen Zielfunktionswert unter $10^{-8}$ fällt.

<!--en:a811257b-->
Im logarithmischen Diagramm ergeben beide Verfahren Geraden, sie konvergieren also linear. Die Gerade des Koordinatenabstiegs fällt jedoch wesentlich steiler ab. Meist erreicht er eine Differenz von $10^{-8}$ in vier bis sechs Durchläufen, während ISTA zwölf bis fünfundzwanzig Schritte benötigt. Aus den Beweisen dieses Abschnitts lässt sich dieser genaue Geschwindigkeitsgewinn nicht vorhersagen. Die schärfsten Konvergenzraten für den Koordinatenabstieg gelten bei zufälliger Wahl der Koordinaten :cite:`Nesterov.2012`. Auch für zyklische Durchläufe wie unsere ist eine Konvergenzrate garantiert, allerdings mit schwächeren Konstanten :cite:`Beck.Tetruashvili.2013`.

<!--en:4bd45453-->
Der Koordinatenabstieg hat eine wichtige Einschränkung. Lässt sich ein nichtglatter Term mit Knick nicht nach Koordinaten trennen, kann das Verfahren weit vom Optimum entfernt stehen bleiben. Betrachten wir $f(x, y) = \max(|x|, |y|)$ am Punkt $(1, 1)$. Weder entlang $x$ allein noch entlang $y$ allein lässt sich $f$ verringern, obwohl das globale Minimum $0$ im Ursprung liegt. Beim Lasso funktioniert der Koordinatenabstieg gerade deshalb, weil sich sein nichtglatter Strafterm $\lambda\sum_j |w_j|$ nach Koordinaten trennen lässt :cite:`Tseng.2001`. Für dünn besetzte lineare Modelle und blockweise alternierende Minimierung bleibt er das Verfahren der Wahl. Beim Training dicht besetzter neuronaler Netze spielt er jedoch keine Rolle, weil moderne Beschleuniger auf parallele Matrizenmultiplikationen ausgelegt sind und skalare Schritte hier nacheinander ablaufen.

<!--en:7d63bb3a-->
Der gesamte Abschnitt folgt einem gemeinsamen Muster. Wenn wir einen Multiplikator suchten, haben wir zuerst $\mathbf{x}$ bestimmt. In :numref:`sec_mdl-lagrangian-duality` kehren wir die Frage um. Wie können wir die Multiplikatoren *direkt* bestimmen? Und wie lässt sich mit jeder Wahl von Multiplikatoren nachweisen, dass kein zulässiger Punkt eine bestimmte untere Schranke unterschreitet? Den Einstieg bildet unser Clipping-Beispiel mit $\lambda^\star = 4$ und $p^\star = 8$.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:b098ac35-->
An einem Optimum unter Nebenbedingungen muss der Gradient nicht verschwinden. Entscheidend ist, dass keine zulässige Richtung die Zielfunktion verringert. Für eine einzelne Gleichungsnebenbedingung mit $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$ bedeutet dies, dass $\nabla f + \nu\nabla h = \mathbf{0}$ für einen Lagrange-Multiplikator $\nu$ gilt. Er misst, wie stark die Lösung gegen die Grenze drückt. Nehmen wir Ungleichungsnebenbedingungen hinzu, erhalten wir die vier KKT-Bedingungen. Die duale Zulässigkeit ($\lambda_i \ge 0$) stellt sicher, dass der Rand nur in eine Richtung wirkt. Der komplementäre Schlupf ($\lambda_i g_i = 0$) setzt den Multiplikator jeder inaktiven Nebenbedingung auf null. Bei konvexen Problemen ist jeder KKT-Punkt ein globales Minimum. Unter einer Regularitätsbedingung (etwa LICQ oder der Slater-Bedingung) ist auch jedes Minimum ein KKT-Punkt.

<!--en:6f2ced4d-->
Um Lösungen unter Nebenbedingungen iterativ zu berechnen, schließt der projizierte Gradientenabstieg an jeden Gradientenschritt eine euklidische Projektion $\Pi_C$ an. Die Projektion auf eine abgeschlossene konvexe Menge ist eindeutig, durch die Ungleichung für den stumpfen Winkel charakterisiert und nichtexpansiv. Deshalb behält der projizierte Gradientenabstieg die Konvergenzrate des Gradientenabstiegs ohne Nebenbedingungen bei. Bei der Projektion auf eine Kugel, einen Halbraum oder den Wahrscheinlichkeitssimplex genügt es jeweils, einen einzigen skalaren Multiplikator zu bestimmen. Der Proximaloperator schließlich verallgemeinert die Projektion von Mengen auf nichtglatte Funktionen. Das weiche Schwellenwertverfahren ist der Proximaloperator von $\lambda|\cdot|$. Sowohl ISTA als auch der Koordinatenabstieg lösen damit das Lasso und liefern exakte Nullen, die der Subgradientenabstieg nicht erreicht.

<!--en:2294243a-->
## Übungen

<!--en:f75c9832-->
1. **Entartete Nebenbedingungen.** Die Lagrange-Bedingung :eqref:`eq_mdl-opt-lagrange-condition` und die KKT-Bedingungen :eqref:`eq_mdl-opt-kkt` sind an einem Minimum nur unter einer Regularitätsbedingung garantiert. Jedes der folgenden Probleme verletzt eine solche Bedingung.
    1. Minimieren Sie $f(\mathbf{x}) = x_1 + x_2^2$ auf $\mathbb{R}^3$ unter den Nebenbedingungen $h_1(\mathbf{x}) = x_3 = 0$ und $h_2(\mathbf{x}) = x_3 - x_1^2 = 0$. Zeigen Sie, dass der zulässige Bereich eine Gerade ist, bestimmen Sie den Minimierer und zeigen Sie, dass beide Nebenbedingungsgradienten überall von null verschieden sind. Zeigen Sie, dass keine Multiplikatoren $\nu_1, \nu_2$ am Minimierer die Gleichung $\nabla f + \nu_1 \nabla h_1 + \nu_2 \nabla h_2 = \mathbf{0}$ erfüllen. Beschreiben Sie dieselbe Gerade durch zwei andere Gleichungsnebenbedingungen, für die Multiplikatoren existieren. Ist eine Regularitätsbedingung eine Eigenschaft des zulässigen Bereichs oder der Funktionen, die ihn beschreiben?
    1. Minimieren Sie $f(x) = x$ auf $\mathbb{R}$ unter der Nebenbedingung $x^2 \le 0$. Bestimmen Sie $p^\star$ und zeigen Sie, dass kein $\lambda \ge 0$ am Minimierer die KKT-Stationaritätsbedingung erfüllt. Welche der beiden Bedingungen, LICQ und Slater-Bedingung, sind verletzt? Begründen Sie Ihre Antwort für jede Bedingung.
    1. Minimieren Sie $f(x) = x$ auf $\mathbb{R}$, wobei die Nebenbedingung $-x \le 0$ zweimal vorkommt, also $g_1(x) = g_2(x) = -x$. Zeigen Sie, dass LICQ verletzt und die Slater-Bedingung erfüllt ist. Bestimmen Sie alle Paare $(\lambda_1^\star, \lambda_2^\star)$, die am Minimierer die KKT-Bedingungen erfüllen. Berechnen Sie unter den gelockerten Nebenbedingungen $-x \le u_1$ und $-x \le u_2$ den Optimalwert $p^\star(u_1, u_2)$. Prüfen Sie nach, dass für jedes KKT-Paar die Ungleichung $p^\star(\mathbf{u}) \ge p^\star(\mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u}$ für alle $\mathbf{u}$ gilt. Ist $p^\star$ bei $\mathbf{0}$ differenzierbar, und was tritt an die Stelle der Beziehung $\boldsymbol{\lambda}^\star = -\nabla p^\star(\mathbf{0})$?
   :exlabel:`ex_mdl-constrained-degenerate-constraints`

<!--en:ad250440-->
1. **Fixpunkte des projizierten Gradientenabstiegs.** Für eine abgeschlossene konvexe Menge $C$ sind die Fixpunkte des projizierten Gradientenabstiegs :eqref:`eq_mdl-opt-pgd` genau die stationären Punkte unter der Nebenbedingung, unabhängig von der Schrittweite $\eta > 0$.
    1. Sei $C = \{\mathbf{x} : \mathbf{x} \ge 0\}$. Zeigen Sie, dass $\mathbf{x}^\star$ genau dann ein Fixpunkt ist, wenn für jede Koordinate $i$ entweder $\partial f / \partial x_i (\mathbf{x}^\star) = 0$ und $x_i^\star \ge 0$ oder $\partial f / \partial x_i (\mathbf{x}^\star) > 0$ und $x_i^\star = 0$ gilt. Zeigen Sie, dass dies die KKT-Bedingungen für die Nebenbedingungen $-x_i \le 0$ sind, und drücken Sie die Multiplikatoren durch $\nabla f(\mathbf{x}^\star)$ aus.
    1. Gradienten-Clipping verwendet den Schritt $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\Pi_{B_c}(\nabla f(\mathbf{x}_k))$ mit $B_c = \{\mathbf{z} : \|\mathbf{z}\| \le c\}$. Bestimmen Sie die Fixpunkte dieser Aktualisierung und vergleichen Sie sie mit denen des projizierten Gradientenabstiegs auf der Parameterkugel $\{\mathbf{x} : \|\mathbf{x}\| \le r\}$. Welches der beiden Verfahren löst ein beschränktes Optimierungsproblem?
    1. Binäre Gewichte beschränken einen Parameter auf die nichtkonvexe Menge $C = \{-1, 1\}$. Die Projektion auf $C$ rundet auf das nähere Element (bei Gleichstand auf $1$). Sei $f(x) = \tfrac12 (x - a)^2$ mit $0 < a < 1$. Für welche Schrittweiten ist $x = 1$ ein Fixpunkt von :eqref:`eq_mdl-opt-pgd` und für welche ist $x = -1$ ein Fixpunkt? Wie verhalten sich die Iterationspunkte, wenn $\eta$ oberhalb beider Bereiche liegt? Welcher Punkt minimiert $f$ auf $C$? Erfüllt einer der beiden Punkte die Ungleichung $f'(x^\star)(x - x^\star) \ge 0$ für alle $x \in C$? Welcher Schritt im Fixpunktargument versagt für diese Menge?
    1. Führen Sie die PGD-Codezelle für den Halbraum mit $\eta = 2.2/1000$ anstelle von $0.0015$ aus. Warum divergiert der Lauf trotz der Nichtexpansivität der Projektion? Welcher Faktor in der Proposition zur Kontraktion übersteigt $1$?
   :exlabel:`ex_mdl-constrained-projected-fixed-points`

<!--en:96f784cf-->
1. [code] **Sparsemax und die $\ell_1$-Kugel.** Die Funktion `project_simplex` berechnet die Simplexprojektion :eqref:`eq_mdl-opt-simplex`, also Sparsemax. Ihr erster Rückgabewert ist der projizierte Vektor.
    1. Zeigen Sie, dass die Projektion unverändert bleibt, wenn Sie zu jedem Score dieselbe Konstante addieren. Teilen Sie die Scores durch eine Temperatur $T > 0$. Seien $u_1 > u_2$ die beiden größten Einträge von $\mathbf{y}$. Zeigen Sie, dass die Projektion von $\mathbf{y} / T$ genau dann ein One-Hot-Vektor ist, wenn $T \le u_1 - u_2$ gilt. Bestimmen Sie ihren Grenzwert für $T \to \infty$. Tragen Sie für zehn standardnormalverteilte Scores die Anzahl der von null verschiedenen Einträge der Projektion von $\mathbf{y} / T$ gegen $T$ auf.
    1. Sei $S$ der Träger von $\mathbf{x}^\star$, und kein $y_i$ sei gleich dem Schwellenwert $\tau$. Leiten Sie die Jacobi-Matrix $\partial \mathbf{x}^\star / \partial \mathbf{y}$ aus :eqref:`eq_mdl-opt-simplex` her :cite:`Martins.Astudillo.2016` und prüfen Sie das Ergebnis mit zentralen endlichen Differenzen. Wie viele Rechenoperationen benötigt der Rückwärtsdurchlauf durch Sparsemax bei einer Eingabe der Länge $n$?
    1. Zeigen Sie, dass die Projektion auf die $\ell_1$-Kugel $\{\mathbf{x} : \|\mathbf{x}\|_1 \le z\}$ die Vorzeichen der von null verschiedenen projizierten Einträge gegenüber $\mathbf{y}$ beibehält und für $\|\mathbf{y}\|_1 > z$ durch $\mathrm{sign}(\mathbf{y}) \odot z\,\Pi_\Delta(|\mathbf{y}| / z)$ gegeben ist :cite:`Duchi.Shalev-Shwartz.Singer.ea.2008`. Implementieren Sie die Projektion mit `project_simplex`. Weisen Sie die Optimalität des Ergebnisses mit der Variationsungleichung $(\mathbf{y} - \hat{\mathbf{x}})^\top(\mathbf{x} - \hat{\mathbf{x}}) \le 0$ nach, indem Sie diese an den $2n$ Punkten $\pm z\,\mathbf{e}_i$ prüfen. Warum genügen diese Punkte?

<!--en:4f2ce876-->
1. **Bedingungen zweiter Ordnung auf der Kugeloberfläche.** ● Die Lagrange-Bedingung ist eine Bedingung erster Ordnung und kann Minima nicht von Sattelpunkten unterscheiden. Ein Test zweiter Ordnung muss die Krümmung der Nebenbedingung berücksichtigen.
    1. Seien $f$ und $h$ zweimal stetig differenzierbar. Der Punkt $\mathbf{x}^\star$ erfülle :eqref:`eq_mdl-opt-lagrange-condition` mit dem Multiplikator $\nu^\star$. Zeigen Sie für jede Kurve $\mathbf{x}(t)$ auf $\{h = 0\}$ mit $\mathbf{x}(0) = \mathbf{x}^\star$ und $\dot{\mathbf{x}}(0) = \mathbf{v}$, dass

       $$\left.\frac{d^2}{dt^2} f(\mathbf{x}(t))\right|_{t=0} = \mathbf{v}^\top \nabla^2_{\mathbf{x}} \mathcal{L}(\mathbf{x}^\star, \nu^\star)\, \mathbf{v},$$

       gilt, wobei die Lagrange-Funktion gegeben ist durch :eqref:`eq_mdl-opt-lagrangian-eq`. Folgern Sie, dass $\mathbf{v}^\top \nabla^2_{\mathbf{x}} \mathcal{L}\, \mathbf{v} \ge 0$ für jede Tangentialrichtung $\mathbf{v}$ an einem lokalen Minimum gilt. Warum ist die Hesse-Matrix von $f$ allein die falsche Matrix für diesen Test? Umgekehrt folgt aus positiver Definitheit auf dem Tangentialraum ein striktes lokales Minimum (:citet:`Nocedal.Wright.2006`, Kapitel 12).
    1. Sei $\mathbf{A}$ symmetrisch mit paarweise verschiedenen Eigenwerten $a_1 < a_2 < \cdots < a_n$ und normierten Eigenvektoren $\mathbf{q}_1, \ldots, \mathbf{q}_n$. Minimieren Sie $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x}$ unter der Nebenbedingung $\mathbf{x}^\top \mathbf{x} = 1$. Zeigen Sie, dass die Lagrange-Punkte $\pm \mathbf{q}_i$ mit $\nu^\star = -a_i$ sind. Bestimmen Sie die Eigenwerte von $\nabla^2_{\mathbf{x}} \mathcal{L}$ auf dem Tangentialraum bei $\mathbf{q}_i$. Welche Lagrange-Punkte sind lokale Minima, welche lokale Maxima und welche Sattelpunkte? Wie viele linear unabhängige Abstiegsrichtungen hat $\mathbf{q}_i$?
    1. Der projizierte Gradientenabstieg auf der Kugeloberfläche, $\mathbf{x} \leftarrow (\mathbf{x} - \eta \nabla f(\mathbf{x})) / \|\mathbf{x} - \eta \nabla f(\mathbf{x})\|$, projiziert auf eine nichtkonvexe Menge. Zeigen Sie, dass er für diese Funktion $f$ der Potenziteration für $\mathbf{I} - 2\eta \mathbf{A}$ entspricht. Nehmen Sie $a_1 + a_n > 0$ an. Wie verhalten sich die Iterationspunkte $\mathbf{x}_k$ und die Werte $f(\mathbf{x}_k)$ bei zufälligem Start? Wie hängt die Antwort von $\eta > 0$ ab?

<!--en:d77d2b57-->
1. **Blockweises weiches Schwellenwertverfahren (Gruppen-Lasso).** Der Strafterm $\lambda\|\mathbf{w}\|_2$ für einen ganzen Vektor hat nur bei $\mathbf{w} = \mathbf{0}$ einen Knick. Seine Subgradienten ergeben sich aus $\mathbf{w}/\|\mathbf{w}\|$ für $\mathbf{w} \neq \mathbf{0}$ und aus allen Vektoren der Einheitskugel bei $\mathbf{0}$, jeweils multipliziert mit $\lambda$.
    1. Zeigen Sie, dass $\mathrm{prox}_{\lambda\|\cdot\|_2}(\mathbf{z}) = \max\left(0,\; 1 - \lambda/\|\mathbf{z}\|\right)\mathbf{z}$ gilt.
    1. Vergleichen Sie das Muster der Nulleinträge mit dem des koordinatenweisen weichen Schwellenwertverfahrens :eqref:`eq_mdl-opt-soft-threshold`. Wann liefert der blockweise Proximaloperator exakt $\mathbf{0}$? Kann er eine einzelne Koordinate eines Blocks auf null setzen?
    1. Wie wirkt sich der Strafterm $\lambda\sum_u\|\mathbf{w}_u\|_2$ auf ein Netz aus, wenn wir über die Vektoren $\mathbf{w}_u$ der ausgehenden Gewichte der versteckten Einheiten $u$ summieren?
    1. [code] Prüfen Sie (a) durch vollständige Suche nach dem Minimum auf einem 2-D-Gitter für $\mathbf{z} = (0.3, 0.4)$ und $\mathbf{z} = (3, 4)$ mit $\lambda = 1$. Die Ergebnisse sind $\mathbf{0}$ und $(2.4, 3.2)$. Wo ist $(2.4, 3.2)$ in diesem Abschnitt schon einmal aufgetreten?
   :exlabel:`ex_mdl-constrained-group-lasso-prox`

<!--en:48a48c5d-->
1. **Moreau-Zerlegung.** ● Die *konvex Konjugierte* von $\psi$ ist $\psi^*(\mathbf{y}) = \sup_{\mathbf{x}}\left(\mathbf{y}^\top\mathbf{x} - \psi(\mathbf{x})\right)$. Für eine eigentliche, abgeschlossene, konvexe Funktion $\psi$ stellen zwei Aussagen die Verbindung zum Proximaloperator :eqref:`eq_mdl-opt-prox` her. Es gilt $\mathbf{x} = \mathrm{prox}_\psi(\mathbf{z})$ genau dann, wenn $\mathbf{z} - \mathbf{x} \in \partial \psi(\mathbf{x})$ gilt (Zugehörigkeit zur Menge der Subgradienten), und $\mathbf{y} \in \partial \psi(\mathbf{x})$ genau dann, wenn $\mathbf{x} \in \partial \psi^*(\mathbf{y})$ gilt :cite:`Parikh.Boyd.2014`.
    1. Leiten Sie die Moreau-Zerlegung $\mathbf{z} = \mathrm{prox}_\psi(\mathbf{z}) + \mathrm{prox}_{\psi^*}(\mathbf{z})$ her.
    1. Die Konjugierte von $\psi = \lambda\|\cdot\|_1$ ist die Indikatorfunktion des Quaders $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le \lambda\}$. Zeigen Sie, dass die Zerlegung das weiche Schwellenwertverfahren :eqref:`eq_mdl-opt-soft-threshold` als Differenz aus $\mathbf{z}$ und einer durch Clipping begrenzten Kopie von $\mathbf{z}$ darstellt.
    1. Die Konjugierte von $\psi = \lambda\|\cdot\|_2$ ist die Indikatorfunktion der $\ell_2$-Kugel mit Radius $\lambda$. Ihr Proximaloperator projiziert auf die Kugel, begrenzt also wie beim Gradienten-Clipping die Norm. Leiten Sie mithilfe der Zerlegung aus der Clipping-Formel die Formel für das blockweise weiche Schwellenwertverfahren aus der Aufgabe zum Gruppen-Lasso her. Welche beiden Vektoren aus diesem Abschnitt ergeben für $\mathbf{z} = (3, 4)$ und $\lambda = 1$ in der Summe $\mathbf{z}$?
   :exlabel:`ex_mdl-convexity-moreau-decomposition`

<!--en:78145982-->
1. [code] **Beschleunigtes Proximalgradientenverfahren.** ● FISTA :cite:`Beck.Teboulle.2009` ergänzt den ISTA-Schritt :eqref:`eq_mdl-opt-ista` für $F = s + \psi$ um Momentum. Mit $\mathbf{v}_1 = \mathbf{x}_0$ und $t_1 = 1$ gelten für $k \ge 1$ die Iterationsschritte

   $$\begin{aligned}\mathbf{x}_k &= \mathrm{prox}_{\eta \psi}\bigl(\mathbf{v}_k - \eta \nabla s(\mathbf{v}_k)\bigr), \\ t_{k+1} &= \tfrac12\Bigl(1 + \sqrt{1 + 4t_k^2}\Bigr), \\ \mathbf{v}_{k+1} &= \mathbf{x}_k + \frac{t_k - 1}{t_{k+1}}\,(\mathbf{x}_k - \mathbf{x}_{k-1}).\end{aligned}$$

   Für $\eta = 1/L$ erfüllt die Differenz zum optimalen Zielfunktionswert die Schranke $F(\mathbf{x}_k) - F^\star \le 2L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2/(k+1)^2$ gegenüber $O(1/k)$ bei ISTA.
    1. Zeigen Sie, dass der Proximaloperator nichtexpansiv ist, also $\|\mathrm{prox}_{\eta\psi}(\mathbf{u}) - \mathrm{prox}_{\eta\psi}(\mathbf{v})\| \le \|\mathbf{u} - \mathbf{v}\|$ gilt. Addieren Sie dazu wie im Beweis für Projektionen die beiden Subgradientenungleichungen, die $\mathrm{prox}_{\eta\psi}(\mathbf{u})$ und $\mathrm{prox}_{\eta\psi}(\mathbf{v})$ charakterisieren.
    1. Ergänzen Sie die ISTA-Codezelle um FISTA für dasselbe Lasso-Problem. Tragen Sie $F(\mathbf{x}_k) - F^\star$ für beide Verfahren über 100 Schritte auf einer logarithmischen Achse auf. Bestimmen Sie $F^\star$ aus $10^4$ ISTA-Schritten. Welches Verfahren liegt nach 10 Schritten vorn, welches nach 30? Erklären Sie den geradlinigen Verlauf für ISTA mit der Proposition zur Kontraktion von PGD und Teil (a).
    1. Verläuft $F(\mathbf{x}_k)$ entlang der FISTA-Iterationspunkte monoton?
    1. Ersetzen Sie $\mathbf{A}$ durch eine gaußsche $30 \times 100$-Matrix, geben Sie dieselben drei von null verschiedenen Koeffizienten vor und führen Sie beide Verfahren für 1000 Schritte aus. Für welchen Bereich von $k$ liegt FISTA nun vorn? Warum endet ISTA auch hier mit einer schnellen linearen Phase, obwohl $\mathbf{A}^\top\mathbf{A}$ singulär ist?
   :exlabel:`ex_mdl-convexity-accelerated-proximal-gradient`

<!--en:6b68d4f6-->
[Diskussion](https://d2l.discourse.group/t/36831)

<!--en:27bad0b5-->
Beschränkte Optimierung<br>**Lagrange-Multiplikatoren · KKT-Bedingungen · euklidische Projektion und Simplexprojektion · Proximaloperatoren und ISTA · Koordinatenabstieg**

<!--en:3fc41ac8-->
::: {.slide title="Wenn der Gradient im Optimum von null verschieden ist"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:97a948a1-->
Die Projektion von $\mathbf{y} = (3, 4)$ auf die Einheitskreisscheibe $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|_2^2 - 1) \le 0$ minimiert $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|_2^2$:

<!--en:a15989bd-->
- **Gradient ungleich null bei $\mathbf{x}^\star = (0.6, 0.8)$:** $\nabla f(\mathbf{x}^\star) = (-2.4, -3.2) \neq \mathbf{0}$.

<!--en:672eed55-->
- **Kräftegleichgewicht:** Keine zulässige Bewegung verringert $f$, denn die äußere Normale $\nabla g(\mathbf{x}^\star)$ wirkt mit dem Multiplikator $\lambda^\star = 4$ entgegen.

<!--en:e9075685-->
::: {.slide title="Gleichungsnebenbedingungen und Berührung von Niveaumengen"}

<!--en:b87e66e5-->
[Lagrange-Multiplikatoren]{.kicker}

<!--en:e1612161-->
Zu minimieren ist eine differenzierbare Zielfunktion $f(\mathbf{x})$ auf der durch die Gleichung $h(\mathbf{x}) = 0$ beschriebenen Fläche (etwa auf dem Einheitskreis $h(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) = 0$):

<!--en:63c21f14-->
- **Am Schnittpunkt ist kein Optimum möglich:** Schneidet eine Niveaumenge von $f$ die Fläche $\{h = 0\}$, so hat $\nabla f$ eine von null verschiedene Tangentialkomponente. Damit gibt es eine zulässige Abstiegsrichtung.

<!--en:76fdca57-->
- **Berührbedingung:** An jedem lokalen Extremum $\mathbf{x}^\star$ mit $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$ berührt die Niveaumenge von $f$ die Fläche $\{h = 0\}$. Ihre Normalenvektoren sind daher parallel:
  $$\nabla f(\mathbf{x}^\star) + \nu^\star\,\nabla h(\mathbf{x}^\star) = \mathbf{0}$$

<!--en:75a136d9-->
::: {.slide title="Lagrange-Funktion und Stationarität auf dem Kreis"}

<!--en:b87e66e5-2-->
[Lagrange-Multiplikatoren]{.kicker}

<!--en:6e363a63-->
Beide Anforderungen, parallele Normalen und Zulässigkeit $h(\mathbf{x}) = 0$, beschreibt eine einzige Funktion von $(\mathbf{x}, \nu)$, die **Lagrange-Funktion**:

<!--en:942b48ff-->
- **Lösung auf dem Einheitskreis ($\mathbf{y} = (3, 4)$):**
  1. Aus der Stationarität $\nabla_{\mathbf{x}}\mathcal{L} = (\mathbf{x} - \mathbf{y}) + \nu\mathbf{x} = \mathbf{0}$ folgt, dass $\mathbf{x} = \frac{\mathbf{y}}{1 + \nu}$ auf der Geraden durch den Ursprung und $\mathbf{y}$ liegt.
  2. Die Zulässigkeit $\|\mathbf{x}\|_2 = 1$ verlangt $|1 + \nu| = \|\mathbf{y}\|_2 = 5$. Es gibt zwei Lösungen:
     - **$\nu^\star = +4$:** $\mathbf{x}^\star = (0.6, 0.8)$ im Abstand $4$ von $\mathbf{y}$ ($f = 8$, das **globale Minimum**).
     - **$\nu = -6$:** $\mathbf{x} = (-0.6, -0.8)$ im Abstand $6$ von $\mathbf{y}$ ($f = 18$, das **globale Maximum**).

<!--en:5c616901-->
- **Warum $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$ wichtig ist:** Beim Problem $\min x_1$ unter der Nebenbedingung $h(x_1, x_2) = x_1^2 + x_2^2 = 0$ gilt am einzigen zulässigen Punkt $\mathbf{0}$ zwar $\nabla f(\mathbf{0}) = (1, 0)$, aber $\nabla h(\mathbf{0}) = \mathbf{0}$. Daher gibt es keinen Multiplikator $\nu$.

<!--en:71a31778-->
::: {.slide title="Nächstgelegener Punkt auf einer Hyperebene und Schattenpreise"}

<!--en:b87e66e5-3-->
[Lagrange-Multiplikatoren]{.kicker}

<!--en:f3911b8f-->
Die Minimierung von $f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|_2^2$ auf der Hyperebene $h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = 0$ liefert $\mathbf{x} = -\nu\mathbf{a}$ und damit

<!--en:235ccc85-->
- **Schattenpreis:** Die Ableitung von $p^\star(b)$ nach dem Schwellenwert $b$ ergibt $\frac{\partial p^\star(b)}{\partial b} = -\nu^\star$.

<!--en:0b83bc9f-->
::: {.slide title="Ungleichungen als aktive oder inaktive Nebenbedingungen"}

<!--en:337be85b-->
[Ungleichungsnebenbedingungen]{.kicker}

<!--en:1a5c79be-->
Bei der Minimierung von $\tfrac12\|\mathbf{x} - \mathbf{y}\|_2^2$ auf der abgeschlossenen Kugel $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|_2^2 - r^2) \le 0$ ergeben sich zwei Fälle:

<!--en:2b5a606c-->
- **Inaktiv ($g(\mathbf{x}^\star) < 0$, $\|\mathbf{y}\| < r$):** $\mathbf{x}^\star = \mathbf{y}$ liegt im Inneren der Kugel, und es gilt $\lambda^\star = 0$.

<!--en:46fd58d2-->
- **Aktiv ($g(\mathbf{x}^\star) = 0$, $\|\mathbf{y}\| \ge r$):** $\mathbf{x}^\star = r\mathbf{y}/\|\mathbf{y}\|$ liegt auf der Kugeloberfläche, und es gilt $\lambda^\star \ge 0$.

<!--en:2ba6b73b-->
::: {.slide title="Kräftegleichgewicht beim Gradienten-Clipping"}

<!--en:337be85b-2-->
[Ungleichungsnebenbedingungen]{.kicker}

<!--en:4f206d35-->
Für $\mathbf{y} = (3, 4)$ und Radius $r = 1$ liefert die Formel für die aktive Nebenbedingung $\lambda^\star = 5/1 - 1 = 4$ und $\mathbf{x}^\star = (0.6, 0.8)$:

<!--en:a3f667c5-->
**Warum Multiplikatoren von Ungleichungen nichtnegativ sein müssen ($\lambda^\star \ge 0$):** Da $\nabla g(\mathbf{x}^\star)$ aus dem zulässigen Bereich $\{g \le 0\}$ *nach außen* zeigt, weist nach der Stationaritätsbedingung $-\nabla f(\mathbf{x}^\star) = \lambda^\star \nabla g(\mathbf{x}^\star)$ mit $\lambda^\star \ge 0$ auch die Abstiegsrichtung $-\nabla f$ nach außen gegen den Rand. Bei $\lambda^\star < 0$ würde $-\nabla f$ *nach innen* zeigen. Ein Schritt ins Innere würde also $f$ verringern.

<!--en:e2295582-->
::: {.slide title="Die vier Karush-Kuhn-Tucker-Bedingungen (KKT-Bedingungen)"}

<!--en:03246263-->
[KKT-Bedingungen]{.kicker}

<!--en:5e79d6f2-->
Für $\min_{\mathbf{x}} f(\mathbf{x})$ unter den Nebenbedingungen $g_i(\mathbf{x}) \le 0$ und $h_j(\mathbf{x}) = 0$ lautet die Lagrange-Funktion $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})$:

<!--en:b2d6f59f-->
- **Zulässigkeit:** primal ($g_i(\mathbf{x}^\star) \le 0, h_j(\mathbf{x}^\star) = 0$) und dual ($\lambda_i^\star \ge 0$).

<!--en:93edc294-->
- **Komplementärer Schlupf:** $\lambda_i^\star g_i(\mathbf{x}^\star) = 0$ für jedes $i$ (inaktive Nebenbedingungen üben keine Kraft aus).

<!--en:88c8e0a1-->
::: {.slide title="Normalenkegel aktiver Nebenbedingungen"}

<!--en:03246263-2-->
[KKT-Bedingungen]{.kicker}

<!--en:6148fd1d-->
Wegen des komplementären Schlupfs gilt bei inaktiven Nebenbedingungen ($g_i(\mathbf{x}^\star) < 0$) stets $\lambda_i^\star = 0$. Sie entfallen in der Summe der Stationaritätsbedingung. Sei $\mathcal{A}(\mathbf{x}^\star) = \{i : g_i(\mathbf{x}^\star) = 0\}$ die Menge der aktiven Indizes, mit $\lambda_i^\star \ge 0$:

<!--en:de7c92a7-->
- **Normalenkegel:** Im Optimum muss die Richtung des steilsten Abstiegs $-\nabla f(\mathbf{x}^\star)$ in dem konvexen Kegel liegen, den die äußeren Normalen $\nabla g_i(\mathbf{x}^\star)$ der aktiven Nebenbedingungen aufspannen.

<!--en:0f8b3ad5-->
- **Im nichtkonvexen Fall:** Bei $\min_{x \in [-1, 1]} -x^2$ erfüllt der innere Punkt $x = 0$ mit $\lambda_1 = \lambda_2 = 0$ alle vier KKT-Bedingungen. Dennoch ist $x = 0$ das globale *Maximum*.

<!--en:00a52566-->
::: {.slide title="Regularitätsbedingungen nach LICQ und Slater"}

<!--en:449fc0b5-->
[Notwendigkeit und Hinlänglichkeit der KKT-Bedingungen]{.kicker}

<!--en:561d1d6d-->
Zwei Regularitätsbedingungen garantieren jeweils, dass jedes lokale Minimum ein KKT-Punkt ist. Sie schließen entartete Spitzen am Rand aus, an denen die Nebenbedingungsgradienten zusammenfallen:

<!--en:5ab442db-->
- **1. Regularitätsbedingung der linearen Unabhängigkeit (LICQ, allgemeine glatte Probleme):** Die Gradienten aller aktiven Ungleichungen und aller Gleichungen am Punkt $\mathbf{x}^\star$, also
  $$\big\{\nabla g_i(\mathbf{x}^\star) : i \in \mathcal{A}(\mathbf{x}^\star)\big\} \cup \big\{\nabla h_j(\mathbf{x}^\star) : j = 1, \ldots, p\big\},$$
  sind linear unabhängig.

<!--en:3174f87d-->
- **2. Slater-Bedingung (konvexe Probleme, $f, g_i$ konvex, $h_j$ affin):** Es gibt einen strikt zulässigen Punkt $\bar{\mathbf{x}}$ mit
  $$g_i(\bar{\mathbf{x}}) < 0 \;\text{ (for all non-affine } g_i\text{)}, \qquad g_i(\bar{\mathbf{x}}) \le 0 \;\text{ (for affine } g_i\text{)}, \qquad h_j(\bar{\mathbf{x}}) = 0 \;\forall\,j$$
  Für die Prüfung genügt ein einziger innerer Punkt (z. B. $\bar{\mathbf{x}} = \mathbf{0}$ bei jeder Normkugel mit Radius $r > 0$). Der Punkt $\mathbf{x}^\star$ muss vorab nicht bekannt sein.

<!--en:6d26d651-->
::: {.slide title="Warum die KKT-Bedingungen bei Konvexität hinreichend sind"}

<!--en:449fc0b5-2-->
[Notwendigkeit und Hinlänglichkeit der KKT-Bedingungen]{.kicker}

<!--en:71287f67-->
Bei konvexen Problemen ($f$ und $g_i$ konvex und differenzierbar, $h_j$ affin) sind die KKT-Bedingungen für globale Optimalität **immer hinreichend**. Eine Regularitätsbedingung ist dafür überflüssig:

<!--en:048fea72-->
- **Proposition:** Erfüllt $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ die KKT-Bedingungen eines konvexen Problems, so gilt $f(\mathbf{x}^\star) \le f(\mathbf{x})$ für jeden zulässigen Punkt $\mathbf{x}$.

<!--en:ea213e18-->
- **Beweis in drei Schritten:**
  1. Die KKT-Multiplikatoren festhalten und folgende Funktion betrachten: $\varphi(\mathbf{x}) = \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) = f(\mathbf{x}) + \sum_i \lambda_i^\star g_i(\mathbf{x}) + \sum_j \nu_j^\star h_j(\mathbf{x})$. Wegen $\lambda_i^\star \ge 0$ ist $\varphi(\mathbf{x})$ eine Summe nichtnegativ gewichteter konvexer Funktionen und affiner Funktionen. Also ist $\varphi$ auf $\mathbb{R}^n$ konvex.
  2. Aus der KKT-Stationarität folgt $\nabla\varphi(\mathbf{x}^\star) = \mathbf{0}$. Somit ist $\mathbf{x}^\star$ ein globaler Minimierer von $\varphi$ ohne Nebenbedingungen. Es gilt $\varphi(\mathbf{x}) \ge \varphi(\mathbf{x}^\star)$ für alle $\mathbf{x} \in \mathbb{R}^n$.
  3. Für jeden zulässigen Punkt $\mathbf{x}$ ($g_i(\mathbf{x}) \le 0, h_j(\mathbf{x}) = 0$) folgt aus dem komplementären Schlupf ($\lambda_i^\star g_i(\mathbf{x}^\star) = 0$):
     $$f(\mathbf{x}) \ge f(\mathbf{x}) + \sum_{i=1}^m \lambda_i^\star g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j^\star h_j(\mathbf{x}) = \varphi(\mathbf{x}) \ge \varphi(\mathbf{x}^\star) = f(\mathbf{x}^\star)$$

<!--en:c81b9aec-->
::: {.slide title="Euklidische Projektion auf eine abgeschlossene konvexe Menge"}

<!--en:ffb1cc92-->
[Projektionen]{.kicker}

<!--en:db2b6ee0-->
Für jede nichtleere abgeschlossene konvexe Menge $C \subseteq \mathbb{R}^n$ bildet die **euklidische Projektion** $\Pi_C(\mathbf{y})$ den Punkt $\mathbf{y} \in \mathbb{R}^n$ auf den eindeutig bestimmten nächstgelegenen Punkt in $C$ ab:

<!--en:874ec675-->
- **Existenz und Eindeutigkeit:** Der Schnitt von $C$ mit einer hinreichend großen abgeschlossenen Kugel um $\mathbf{y}$ ist kompakt. Wegen der Stetigkeit existiert daher ein Minimum. Die $1$-starke Konvexität von $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ garantiert seine Eindeutigkeit.

<!--en:f497f485-->
- **Geschlossene Formeln für elementare Projektionen:**
  - *Euklidische Kugel $\{ \|\mathbf{x}\|_2 \le r \}$:* $\Pi_C(\mathbf{y}) = \frac{\mathbf{y}}{\max(1,\, \|\mathbf{y}\|_2 / r)}$.
  - *Achsenparalleler Quader $[a_i, b_i]$:* $\Pi_C(\mathbf{y})_i = \min\big(b_i, \max(a_i, y_i)\big)$ (koordinatenweises Begrenzen).
  - *Halbraum $\{ \mathbf{a}^\top\mathbf{x} \ge b \}$:* $\Pi_C(\mathbf{y}) = \mathbf{y} + \frac{\max(0,\, b - \mathbf{a}^\top\mathbf{y})}{\|\mathbf{a}\|_2^2}\,\mathbf{a}$.

<!--en:ba489448-->
::: {.slide title="Die Variationsungleichung für den stumpfen Winkel"}

<!--en:ffb1cc92-2-->
[Projektionen]{.kicker}

<!--en:9c649b53-->
Ein Punkt $\hat{\mathbf{x}} \in C$ ist genau dann die Projektion $\Pi_C(\mathbf{y})$, wenn der Residuenvektor $\mathbf{y} - \hat{\mathbf{x}}$ einen stumpfen oder rechten Winkel $\ge 90^\circ$ mit jeder zulässigen Sehne $\mathbf{x} - \hat{\mathbf{x}}$ in die Menge $C$ bildet ($\forall\,\mathbf{x} \in C$):

<!--en:48925e4e-->
- **Notwendigkeit:** Eine Bewegung entlang $(1-\theta)\hat{\mathbf{x}} + \theta\mathbf{x} \in C$ kann $\varphi(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ nicht verringern, also gilt $\nabla\varphi(\hat{\mathbf{x}})^\top(\mathbf{x} - \hat{\mathbf{x}}) = (\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}}) \ge 0$.

<!--en:5a320c57-->
- **Hinlänglichkeit:** Aus der Tangentenschranke für die konvexe Funktion $\varphi$ folgt $\varphi(\mathbf{x}) \ge \varphi(\hat{\mathbf{x}}) + (\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}}) \ge \varphi(\hat{\mathbf{x}})$.

<!--en:19d25f80-->
::: {.slide title="Warum Projektionen nichtexpansiv sind"}

<!--en:ffb1cc92-3-->
[Projektionen]{.kicker}

<!--en:eafd7bae-->
Bei der Projektion zweier Punkte auf eine abgeschlossene konvexe Menge $C$ nimmt ihr euklidischer Abstand niemals zu:

<!--en:ba347460-->
- **Beweis in drei Schritten:** Die Summe der Ungleichungen für den stumpfen Winkel $(\mathbf{x} - \mathbf{u})^\top(\mathbf{v} - \mathbf{u}) \le 0$ und $(\mathbf{y} - \mathbf{v})^\top(\mathbf{u} - \mathbf{v}) \le 0$ (mit $\mathbf{u} = \Pi_C(\mathbf{x}), \mathbf{v} = \Pi_C(\mathbf{y})$) liefert $\|\mathbf{u} - \mathbf{v}\|_2^2 \le (\mathbf{x} - \mathbf{y})^\top(\mathbf{u} - \mathbf{v})$. Dann die Cauchy-Schwarz-Ungleichung anwenden.

<!--en:f7328b02-->
::: {.slide title="Projizierter Gradientenabstieg und Fixpunkte"}

<!--en:a5da397f-->
[Projizierter Gradientenabstieg]{.kicker}

<!--en:9457b46a-->
Beim **projizierten Gradientenabstieg (PGD)** wechseln sich ein Gradientenschritt und eine euklidische Projektion auf $C$ ab:

<!--en:aadb6a96-->
- **Fixpunktkriterium:** $\mathbf{x}^\star = \Pi_C(\mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star))$ gilt genau dann, wenn $\nabla f(\mathbf{x}^\star)^\top(\mathbf{x} - \mathbf{x}^\star) \ge 0$ für alle $\mathbf{x} \in C$ gilt (keine zulässige Abstiegsrichtung, bei konvexem $f$ globales Optimum).

<!--en:618d29f5-->
::: {.slide title="Warum PGD so schnell kontrahiert wie der Gradientenabstieg ohne Nebenbedingungen"}

<!--en:a5da397f-2-->
[Projizierter Gradientenabstieg]{.kicker}

<!--en:407b2fae-->
Verlangsamt eine Projektion nach jedem Gradientenschritt die Konvergenz? Für eine $\mu$-stark konvexe, $L$-glatte Funktion ($\mu\mathbf{I} \preceq \nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$) garantiert die Nichtexpansivität im ungünstigsten Fall *genau denselben* Kontraktionsfaktor $\rho = \max(|1 - \eta\mu|, |1 - \eta L|)$ wie beim Gradientenabstieg ohne Nebenbedingungen:

<!--en:8502c261-->
- **Schritt 1. Darstellung von $\mathbf{x}^\star$ als Fixpunkt:** $\mathbf{x}^\star = \Pi_C\big(\mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star)\big)$.

<!--en:4e5e3f43-->
- **Schritt 2. Wegen der Nichtexpansivität entfällt $\Pi_C$:**
  $$\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| = \big\|\Pi_C(\mathbf{x}_k - \eta\nabla f(\mathbf{x}_k)) - \Pi_C(\mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star))\big\| \le \big\|(\mathbf{x}_k - \mathbf{x}^\star) - \eta\big(\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star)\big)\big\|$$

<!--en:52b8d0d8-->
- **Schritt 3. Anwendung des Hauptsatzes der Differential- und Integralrechnung:** Mit $\bar{\mathbf{H}} = \int_0^1 \nabla^2 f(\mathbf{x}^\star + t(\mathbf{x}_k - \mathbf{x}^\star))\,dt$ ($\mu\mathbf{I} \preceq \bar{\mathbf{H}} \preceq L\mathbf{I}$) gilt
  $$\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| \le \big\|(\mathbf{I} - \eta\bar{\mathbf{H}})(\mathbf{x}_k - \mathbf{x}^\star)\big\| \le \max\big(|1 - \eta\mu|,\, |1 - \eta L|\big)\,\|\mathbf{x}_k - \mathbf{x}^\star\|$$

<!--en:09fb2a26-->
::: {.slide title="Zwei Phasen des PGD im Tal mit κ = 1000"}

<!--en:a5da397f-3-->
[Projizierter Gradientenabstieg]{.kicker}

<!--en:f3d08a2a-->
Zu minimieren ist $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ auf dem Halbraum $x_1 + x_2 \ge 1$ ($\mathbf{x}^\star = (\frac{1000}{1001}, \frac{1}{1001})$, $\lambda^\star = \frac{1000}{1001}$) mittels PGD mit $\eta = 1.5 \times 10^{-3}$ vom Startpunkt $(3, 4)$ aus:

<!--en:b2b5a5ad-->
- **Warum $738$ Schritte statt $10{,}200$?** Nach der ersten Projektion bewegt sich der Iterationspunkt bis zum Schritt $732$ mit der langsamen Rate $1 - \eta = 0.9985$ durch das Innere. Erreicht er die Randgerade $x_1 + x_2 = 1$ bei Schritt $733$, hat die auf die Gerade eingeschränkte quadratische Funktion die Krümmung $500.5$. Der Abstand schrumpft dann mit dem Faktor $1 - 500.5\eta = 0.249$ pro Schritt.

<!--en:8745beec-->
::: {.slide title="Projektion auf den Wahrscheinlichkeitssimplex (Sparsemax)"}

<!--en:8ffaa442-->
[Simplexprojektion]{.kicker}

<!--en:80f6d60d-->
Zur Projektion eines Score-Vektors $\mathbf{y} \in \mathbb{R}^n$ auf den Wahrscheinlichkeitssimplex $\Delta = \{\mathbf{x} \succeq \mathbf{0} : \mathbf{1}^\top\mathbf{x} = 1\}$ ist das folgende Problem zu lösen: $\min_{\mathbf{x} \in \Delta}\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$:

<!--en:538d112d-->
- **KKT-Stationarität und komplementärer Schlupf:** Mit dem Multiplikator $\tau$ für $\sum_i x_i = 1$ und $\lambda_i \ge 0$ für $-x_i \le 0$ vereinfachen sich die Stationarität $x_i - y_i + \tau - \lambda_i = 0$ und die Bedingung $\lambda_i x_i = 0$ zu $x_i^\star = \max(y_i - \tau,\, 0)$. Dabei ist $\tau$ durch die Summenbedingung eindeutig bestimmt:
  $$\sum_{i=1}^n \max(y_i - \tau,\, 0) = 1$$

<!--en:b39abe67-->
- **Exaktes Schwellenwertverfahren in $O(n\log n)$:** Sortieren von $u_1 \ge \cdots \ge u_n$, Berechnen von $\tau_k = \frac{1}{k}\big(\sum_{j=1}^k u_j - 1\big)$ und Wahl des größten $k$ mit $u_k > \tau_k$.

<!--en:b3b793df-->
::: {.slide title="Nachweis der Simplexprojektion durch Prüfung an den Ecken"}

<!--en:8ffaa442-2-->
[Simplexprojektion]{.kicker}

<!--en:9338f552-->
Da $x_i(\tau) = \max(y_i - \tau, 0)$ und $\lambda_i(\tau) = \max(0, \tau - y_i)$ für *jeden* Skalar $\tau$ die Stationarität, die duale Zulässigkeit und den komplementären Schlupf erfüllen, kann ein falsches $\tau$ bei alleiniger Prüfung dieser drei KKT-Residuen unentdeckt bleiben:

<!--en:a3bd6bd8-->
- Erst die Summenbedingung $\sum_i x_i = 1$ und die Ungleichung für den stumpfen Winkel an den $n$ Simplexecken $\mathbf{e}_i$, nämlich $\max_{1 \le i \le n}(\mathbf{y} - \mathbf{x}^\star)^\top(\mathbf{e}_i - \mathbf{x}^\star) \le 0$, weisen nach, dass $\mathbf{x}^\star = \Pi_\Delta(\mathbf{y})$ gilt.

<!--en:a7174e79-->
::: {.slide title="Warum der Subgradientenabstieg keine Sparsität erzeugt"}

<!--en:f48e9910-->
[Proximaloperatoren]{.kicker}

<!--en:b9c976c0-->
Das **Lasso** minimiert $F(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|_2^2 + \lambda\|\mathbf{w}\|_1$ ($\mathbf{A} \in \mathbb{R}^{100 \times 8}$, $5$ vorgegebene Nullgewichte, $\lambda = 3$). Was geschieht beim Subgradientenabstieg $\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\big(\mathbf{A}^\top(\mathbf{A}\mathbf{w}_k - \mathbf{b}) + \lambda\,\mathrm{sign}(\mathbf{w}_k)\big)$ mit $\eta = 1/L$?

<!--en:a33a045c-->
**Warum Subgradientenschritte um null pendeln:** An einer vorgegebenen Nullkoordinate hat der glatte Gradient den Betrag $< \lambda$. Der Subgradientenschritt $\pm\eta\lambda$ überwiegt und kehrt das Vorzeichen der Koordinate bei fast jedem Schritt um. Damit verbleiben $0$ exakte Nullen.

<!--en:f53f0da1-->
::: {.slide title="Weiches Schwellenwertverfahren und Proximaloperator"}

<!--en:f48e9910-2-->
[Proximaloperatoren]{.kicker}

<!--en:d30e774d-->
Der **Proximaloperator** minimiert einen nichtglatten konvexen Strafterm $\psi(\mathbf{x})$ zusammen mit einem quadratischen Abstandsterm, der die Lösung nahe bei $\mathbf{z}$ hält:

<!--en:ea1bbf69-->
- **Eindimensionales weiches Schwellenwertverfahren ($\psi(x) = \lambda|x|$):** Nach der Fermat-Regel $0 \in \lambda\partial|x| + (x - z)$ gilt $\mathrm{prox}_{\lambda|\cdot|}(z) = \mathrm{sign}(z)\max(|z| - \lambda, 0)$. Für $|z| \le \lambda$ gibt es eine **Totzone**, in der das Ergebnis exakt null ist.

<!--en:e65d17ab-->
::: {.slide title="Proximaler Gradientenabstieg (ISTA) findet exakte Nullen"}

<!--en:f48e9910-3-->
[Proximaloperatoren]{.kicker}

<!--en:e94d170e-->
Für eine zusammengesetzte Zielfunktion $F(\mathbf{w}) = s(\mathbf{w}) + \psi(\mathbf{w})$ sei $s$ konvex und $L$-glatt und $\psi$ eigentlich, abgeschlossen und konvex. Der **proximale Gradientenabstieg** minimiert die aus der $L$-Glattheit folgende quadratische obere Schranke von $s$ zusammen mit dem *exakten* Strafterm $\psi$:

<!--en:1e49517a-->
- Für $\psi(\mathbf{w}) = \lambda\|\mathbf{w}\|_1$ ergibt dieser Schritt **ISTA** (Iterative Shrinkage-Thresholding Algorithm). Seine Totzone $[-\eta\lambda, \eta\lambda]$ deckt genau die Amplitude ab, mit der der Subgradientenabstieg um null pendelt. So findet ISTA alle $5$ vorgegebenen Nullen wieder.

<!--en:32fc2098-->
::: {.slide title="Zyklischer Koordinatenabstieg beim Lasso"}

<!--en:07d2c9a4-->
[Koordinatenabstieg]{.kicker}

<!--en:81f2c977-->
Da $\|\mathbf{w}\|_1 = \sum_{j=1}^d |w_j|$ nach Koordinaten zerfällt, lässt sich das Lasso entlang eines einzelnen Gewichts $w_j$ in geschlossener Form minimieren (mit partiellem Residuum $\mathbf{r}_{-j} = \mathbf{b} - \sum_{i \neq j}\mathbf{a}_i w_i$ und Korrelation $c_j = \mathbf{a}_j^\top\mathbf{r}_{-j}$):

<!--en:a4b62c49-->
- **Warum Koordinatendurchläufe schneller sind als ISTA:** Jeder Koordinatenschritt nutzt die Krümmung $L_j = \|\mathbf{a}_j\|_2^2 \approx 100$ der jeweiligen Spalte anstelle der globalen Eigenwertschranke $L = \lambda_{\max}(\mathbf{A}^\top\mathbf{A}) \approx 160$. Damit genügen für $10^{-8}$ bereits $4$ bis $6$ Durchläufe gegenüber $12$ bis $25$ bei ISTA.

<!--en:6e44ef82-->
::: {.slide title="Verständnisfragen zu KKT, Simplexprojektion und Proximaloperator"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:e713842f-->
Drei Rechnungen mit den KKT-Bedingungen und den Regeln für Proximaloperatoren:

<!--en:1b806048-->
1. Projektion von $\mathbf{y}_1 = (1.5, 2.0)$ und $\mathbf{y}_2 = (0.3, 0.4)$ auf die Einheitskreisscheibe $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|_2^2 - 1) \le 0$. Wie lauten jeweils der optimale Punkt $\mathbf{x}^\star$ und der KKT-Multiplikator $\lambda^\star$?

<!--en:9426e669-->
2. Projektion des Score-Vektors $\mathbf{y} = (0.9, 0.5, -0.2)$ auf den Wahrscheinlichkeitssimplex $\Delta$. Welche Koordinaten bleiben positiv ($k$), wie groß ist der Schwellenwert $\tau$, und was ergibt sich für $\Pi_\Delta(\mathbf{y})$?

<!--en:66611403-->
3. Den Operator $\mathrm{prox}_{\lambda|\cdot|}(z)$ des weichen Schwellenwertverfahrens mit $\lambda = 0.7$ bei $z = 0.5$ und bei $z = -2.0$ auswerten.

<!--en:59d4060d-->
**Lösungen:** **(1)** $\|\mathbf{y}_1\|_2 = 2.5 > 1$ (aktiv): $\lambda^\star = 2.5 - 1 = \mathbf{1.5}$ und $\mathbf{x}^\star = \mathbf{y}_1 / 2.5 = \mathbf{(0.6, 0.8)}$. $\|\mathbf{y}_2\|_2 = 0.5 < 1$ (inaktiv): $\lambda^\star = \mathbf{0}$ und $\mathbf{x}^\star = \mathbf{(0.3, 0.4)}$. **(2)** Mögliche Schwellenwerte $\tau_k = (\sum_{j \le k} y_j - 1)/k$: $\tau_1 = -0.1 < 0.9$ (gültig), $\tau_2 = (1.4 - 1)/2 = 0.2 < 0.5$ (gültig), $\tau_3 = 0.2/3 \approx 0.067 \not< -0.2$ (ungültig). Also gilt $k = 2$, $\tau = \mathbf{0.2}$ und $\Pi_\Delta(\mathbf{y}) = \mathbf{(0.7, 0.3, 0.0)}$. **(3)** Bei $z = 0.5$ liegt der Betrag wegen $|0.5| \le 0.7$ in der Totzone, also gilt $\mathrm{prox}_{0.7|\cdot|}(0.5) = \mathbf{0}$. Bei $z = -2.0$ gilt $\mathrm{prox}_{0.7|\cdot|}(-2.0) = -(2.0 - 0.7) = \mathbf{-1.3}$.

<!--en:c6788622-->
::: {.slide title="Zusammenfassung der beschränkten Optimierung und ihrer Operatoren"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:fc612d2e-->
| Problem oder Verfahren | Mathematische Bedingung oder Aktualisierung | Optimalität oder garantierte Rate | Rolle im Deep Learning und maschinellen Lernen |

<!--en:f2b24e1f-->
| **Lagrange-Bedingung für Gleichungen** | $\nabla f(\mathbf{x}^\star) + \sum_j \nu_j^\star \nabla h_j(\mathbf{x}^\star) = \mathbf{0}$, $h_j = 0$ | Unter LICQ notwendig, $\nu_j^\star = -\partial p^\star/\partial b_j$ | Projektion auf Hyperebenen, Normierungsbedingungen |

<!--en:5747d380-->
| **KKT-Bedingungen für Ungleichungen** | Stationarität, primale und duale Zulässigkeit ($\lambda_i^\star \ge 0$), $\lambda_i^\star g_i(\mathbf{x}^\star) = 0$ | Bei konvexen Problemen hinreichend, unter der Slater-Bedingung notwendig | Gradienten-Clipping, SVMs, Regularisierung durch Normbeschränkung |

<!--en:82ba748b-->
| **Euklidische Projektion $\Pi_C$** | $(\mathbf{y} - \Pi_C(\mathbf{y}))^\top(\mathbf{x} - \Pi_C(\mathbf{y})) \le 0\;\forall\mathbf{x}\in C$ | Eindeutig und nichtexpansiv ($\|\Pi_C(\mathbf{x}) - \Pi_C(\mathbf{y})\| \le \|\mathbf{x}-\mathbf{y}\|$) | PGD $\mathbf{x}_{k+1} = \Pi_C(\mathbf{x}_k - \eta\nabla f(\mathbf{x}_k))$ kontrahiert mindestens so schnell wie der Gradientenabstieg ohne Nebenbedingungen (gleiches $\rho$) |

<!--en:950ef665-->
| **Simplexprojektion** | $x_i^\star = \max(y_i - \tau, 0)$ mit $\sum_i x_i^\star = 1$ | Exaktes Sortier- und Schwellenwertverfahren in $O(n\log n)$ | `sparsemax`-Attention-Gewichte mit exakten Nulleinträgen |

<!--en:4db68f42-->
| **Proximalgradient (ISTA)** | $\mathbf{w}_{k+1} = \mathrm{prox}_{\eta\psi}(\mathbf{w}_k - \eta\nabla s(\mathbf{w}_k))$ | $O(1/k)$ bei Konvexität, linear mit $(1-\mu/L)^k$ bei starker Konvexität | Das weiche Schwellenwertverfahren $\mathrm{prox}_{\lambda\|\cdot\|_1}$ erzeugt exakte Sparsität der Gewichte |
