<!--en:572c847e-->
# Dualität

<!--en:e7119089-->
Wenn ein iterativer Löser keine Fortschritte mehr macht und Konvergenz meldet, stellt sich die Frage: Ist der gefundene Punkt tatsächlich optimal? Bei einer konvexen Funktion ohne Nebenbedingungen genügt zur Antwort ein verschwindender Gradient (:numref:`sec_mdl-convexity`). Unter Nebenbedingungen liegt das Optimum jedoch meist auf dem Rand, wo der Gradient von null verschieden ist (:numref:`sec_mdl-constrained-optimization-duality`). Damit entfällt dieser einfache Test. Für numerische Löser wie für mathematische Beweise brauchen wir deshalb eine verlässliche untere Schranke für den Optimalwert. In diesem Abschnitt konstruieren wir solche Schranken systematisch und untersuchen, was die Lagrange-Multiplikatoren der KKT-Bedingungen eigentlich messen.

<!--en:52215460-->
Betrachten wir noch einmal das Clipping-Problem aus :numref:`subsec_mdl-kkt-conditions`. Dort haben wir den quadrierten Abstand $\tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ zum Zielpunkt $\mathbf{x}_0 = (3, 4)$ auf der Einheitskreisscheibe $\|\mathbf{x}\| \le 1$ minimiert. Das Optimum lag bei $\mathbf{x}^\star = (0.6, 0.8)$, der Optimalwert betrug $8$. (Hier schreiben wir $\mathbf{x}_0$ statt $\mathbf{y}$. Das Symbol $\mathbf{y}$ verwenden wir für Labels und später für das Argument der konvex Konjugierten (engl. convex conjugate).) Angenommen, ein iterativer Löser hält bei $\mathbf{x} = (0.55, 0.835)$ an. Wegen $\|\mathbf{x}\|^2 = 0.9997 \le 1$ ist dieser Punkt zulässig, und sein Zielfunktionswert beträgt $8.01$. Woher wissen wir, ob $8.01$ nahezu optimal ist oder ob ein anderer Punkt der Kreisscheibe den Wert $7$ erreicht? Der Gradient am tatsächlichen Optimum beträgt $(-2.4, -3.2)$ und ist damit weit von null entfernt. Die unendlich vielen Punkte der Kreisscheibe einzeln zu prüfen, kommt nicht infrage.

<!--en:777003f0-->
Die Idee besteht darin, die Nebenbedingung in die Zielfunktion aufzunehmen. Dazu addieren wir ein nichtnegatives Vielfaches $\lambda \ge 0$ der Nebenbedingungsfunktion $\tfrac12(\|\mathbf{x}\|^2 - 1)$ zur Zielfunktion. An jedem zulässigen Punkt ist der zusätzliche Term nichtpositiv. Minimieren wir die veränderte Zielfunktion über *ganz* $\mathbb{R}^2$, kann ihr Minimum also höchstens so groß sein wie das Optimum unter der Nebenbedingung. Jede Wahl von $\lambda \ge 0$ liefert so durch unbeschränkte Minimierung eine untere Schranke. Für $\lambda = 4$ beträgt sie $8$. Damit wissen wir sofort, dass der vom Löser gefundene Wert $8.01$ höchstens um $0.01$ vom Optimum abweicht. Die Suche nach der schärfsten solchen Schranke definiert ein zweites Optimierungsproblem, das *duale Problem* (engl. dual problem). Bemerkenswerterweise ist das duale Problem immer konvex, selbst wenn das ursprüngliche Problem nicht konvex ist. Bei einem konvexen Problem mit einem strikt zulässigen Punkt stimmt der duale Optimalwert mit dem Optimum unter den Nebenbedingungen überein. Die optimalen Multiplikatoren geben an, wie schnell sich der Optimalwert verbessert, wenn wir die jeweilige Nebenbedingung lockern.

<!--en:5ccb2b35-->
Mit dieser Schranke weisen wir zunächst die Optimalität beim Clipping-Problem nach und formulieren die *schwache Dualität* (engl. weak duality), die sich für jedes Optimierungsproblem in zwei Zeilen begründen lässt. Dann beweisen wir die *starke Dualität* (engl. strong duality) für konvexe Probleme. In diesem Beweis steckt der einzige geometrisch knifflige Schritt des Abschnitts. Wir deuten die optimalen Multiplikatoren als *Schattenpreise* (engl. shadow prices) und stellen damit eine direkte Verbindung zwischen Gewichtszerfall (engl. weight decay) und Normbeschränkungen her. Anschließend führen wir die *konvex Konjugierte* ein, mit der sich duale Probleme bei linearen Nebenbedingungen bequem berechnen lassen. Diesen Teil können Sie bei der ersten Lektüre überfliegen. Wir rechnen drei Beispiele durch: die Support Vector Machine, die Leistungsverteilung durch Water-Filling und ein nichtkonvexes Problem, dessen beste duale Schranke strikt unter dem Optimum liegt. Für alle Rechnungen genügt NumPy, die Grafiken erstellen wir mit `d2l`.

<!--en:40c031e5-->
## Das duale Problem

<!--en:e47938fc-->
Zunächst wollen wir beweisen, dass kein zulässiger Punkt des Clipping-Problems einen Wert unter $8$ erreicht, ohne die Punkte einzeln zu prüfen. Alle unteren Schranken in diesem Abschnitt beruhen auf einer einfachen Beobachtung: An jedem zulässigen Punkt sind die Nebenbedingungsfunktionen aller Ungleichungen nichtpositiv.

<!--en:67511456-->
### Eine untere Schranke aus einem einzigen Multiplikator

<!--en:57d90043-->
Beim Clipping-Problem minimieren wir $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ mit $\mathbf{x}_0 = (3, 4)$ unter der Nebenbedingung $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$. Der Faktor $\tfrac12$ in $g$ sorgt lediglich dafür, dass der Gradient gleich $\mathbf{x}$ ist. Aus den KKT-Bedingungen kennen wir bereits die Lösung $\mathbf{x}^\star = \mathbf{x}_0 / \|\mathbf{x}_0\| = (0.6, 0.8)$ mit dem Optimalwert $p^\star = \tfrac12(5 - 1)^2 = 8$ und dem optimalen Multiplikator $\lambda^\star = 4$, der sich aus der Stationaritätsbedingung $\nabla f(\mathbf{x}^\star) + \lambda^\star \mathbf{x}^\star = (-2.4, -3.2) + 4\,(0.6, 0.8) = \mathbf{0}$ ergibt.

<!--en:0bb19240-->
Wählen wir einen beliebigen Multiplikator $\lambda \ge 0$ und einen beliebigen zulässigen Punkt $\mathbf{x}$. Wegen $g(\mathbf{x}) \le 0$ und $\lambda \ge 0$ ist ihr Produkt $\lambda g(\mathbf{x})$ nichtpositiv, also gilt $f(\mathbf{x}) + \lambda g(\mathbf{x}) \le f(\mathbf{x})$. Minimieren wir die linke Seite über ganz $\mathbb{R}^2$, kann ihr Wert nur noch kleiner werden. Das Minimum ohne Nebenbedingung beträgt somit höchstens $f(\mathbf{x})$. Das gilt für jedes zulässige $\mathbf{x}$, insbesondere für das beste zulässige $\mathbf{x}$:

<!--en:12379109-->
Berechnen wir die linke Seite explizit. Wir minimieren die Lagrange-Funktion $\mathcal{L}(\mathbf{x}, \lambda) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2 + \tfrac\lambda2(\|\mathbf{x}\|^2 - 1)$. Für jedes $\lambda \ge 0$ ist sie eine konvexe quadratische Funktion von $\mathbf{x}$ mit der Hesse-Matrix $(1 + \lambda)\mathbf{I}$. Ihr eindeutiger Minimierer liegt dort, wo der Gradient $(\mathbf{x} - \mathbf{x}_0) + \lambda \mathbf{x}$ verschwindet, also bei $\mathbf{x}(\lambda) = \mathbf{x}_0 / (1 + \lambda)$. Je größer $\lambda$ ist, desto stärker schrumpft $\mathbf{x}_0$ in Richtung Ursprung. Setzen wir $\mathbf{x}(\lambda)$ wieder in die Lagrange-Funktion ein und nutzen $\mathbf{x}(\lambda) - \mathbf{x}_0 = -\lambda\mathbf{x}_0 / (1 + \lambda)$ und $\|\mathbf{x}_0\|^2 = 25$, erhalten wir $\tfrac12\|\mathbf{x}(\lambda) - \mathbf{x}_0\|^2 = \tfrac{25}{2}\lambda^2/(1+\lambda)^2$ und $\tfrac\lambda2\|\mathbf{x}(\lambda)\|^2 = \tfrac{25}{2}\lambda/(1+\lambda)^2$. Beim Addieren der beiden Brüche kürzt sich ein Faktor $(1 + \lambda)$ heraus. Für den Minimalwert bleibt eine einfache Funktion von $\lambda$:

<!--en:8ea99047-->
Die beiden Terme wirken in entgegengesetzte Richtungen. Mit wachsendem $\lambda$ steigt der erste Term gegen $12.5$. Der zweite Term stammt von der Konstanten $-\tfrac12$ in $g$ und fällt linear gegen minus unendlich. Die schärfste untere Schranke liegt dort, wo sich beide Effekte ausgleichen.

<!--en:bf076653-->
Setzen wir einige Zahlen ein: $q(0) = 0$, $q(1) = 5.75$, $q(2) = 7.33$, $q(3) = 7.875$, $q(4) = 8$, $q(5) = 7.92$ und $q(8) = 7.11$. Jeder dieser Werte ist eine gültige untere Schranke für $p^\star$. Die Ableitung $q'(\lambda) = \tfrac{25}{2(1 + \lambda)^2} - \tfrac12$ verschwindet für $(1 + \lambda)^2 = 25$, also für $\lambda = 4$. Wegen $q''(\lambda) = -25/(1 + \lambda)^3 < 0$ liegt bei $\lambda = 4$ das eindeutige Maximum mit $q(4) = 10 - 2 = 8 = p^\star$. Die beste untere Schranke stimmt mit dem tatsächlichen Optimum überein, und der zugehörige Multiplikator ist der KKT-Multiplikator. Damit ist nachgewiesen, dass der vom Löser gefundene Wert $8.01$ höchstens um $0.01$ vom Optimum abweicht. Schon ein Schätzwert wie $\lambda = 3$ liefert ohne Optimierung eine garantierte Abweichung von höchstens $0.135$.

<!--en:f09aa88d-->
Warum treffen wir mit $\lambda = 4$ genau das Optimum? Betrachten wir den Wert der Nebenbedingungsfunktion am Minimierer der Lagrange-Funktion: $g(\mathbf{x}(\lambda)) = \tfrac12\bigl(25/(1 + \lambda)^2 - 1\bigr)$. Für $\lambda < 4$ ist der Multiplikator zu klein, um den Minimierer in die Einheitskreisscheibe zu ziehen (so hat $\mathbf{x}(1) = (1.5, 2)$ die Norm $2.5$). Es gilt also $g(\mathbf{x}(\lambda)) > 0$, und die Schranke ergibt sich aus einem unzulässigen Punkt. Für $\lambda > 4$ zieht der Multiplikator den Minimierer zu weit ins Innere (so hat $\mathbf{x}(8)$ die Norm $\tfrac59$). Dann drückt $\lambda g(\mathbf{x}(\lambda)) < 0$ die Schranke wieder nach unten. Genau bei $\lambda = 4$ liegt der Minimierer $\mathbf{x}(4) = (0.6, 0.8)$ auf dem Randkreis, der Strafterm $\lambda g(\mathbf{x}(\lambda))$ verschwindet, und die Schranke entspricht dem Zielfunktionswert eines zulässigen Punktes.

<!--en:ff322069-->
Tragen wir $q(\lambda)$ zusammen mit $p^\star$ für ein Gitter von $\lambda$-Werten auf, um den Verlauf der Schranken zu sehen.

<!--en:820973a4-->
Die Kurve ist konkav und verläuft unterhalb der horizontalen Geraden $p^\star = 8$, die sie nur bei $\lambda = 4$ berührt. Überraschenderweise hängt die Konkavität hier nicht von der Geometrie der Kreisscheibe ab. Wie wir nun zeigen, ist die Dualfunktion bei *jedem* Optimierungsproblem konkav.

<!--en:0f078303-->
### Die Lagrange-Dualfunktion

<!--en:dc07b7a5-->
Unsere Herleitung der Schranke hat keine Besonderheit des Clipping-Problems genutzt. Wenden wir dieselbe Konstruktion auf das allgemeine Optimierungsproblem mit Nebenbedingungen aus :numref:`subsec_mdl-kkt-conditions` an,

<!--en:8fc1451c-->
mit dem Optimalwert $p^\star$, dem Infimum von $f$ über alle zulässigen Punkte (mit $p^\star = +\infty$, falls kein zulässiger Punkt existiert). Dabei bezeichnet $\mathcal{D}$ den Bereich, auf dem $f$ und alle Nebenbedingungsfunktionen definiert sind. Er ist einfach $\mathbb{R}^n$, sofern wir nicht die Zugehörigkeit zu einer Menge als harte Nebenbedingung beibehalten. So verfahren wir am Ende dieses Abschnitts mit $[0, 1]$ im Beispiel zur Dualitätslücke. Die Lagrange-Funktion führt für jede Nebenbedingung einen Multiplikator ein,

<!--en:593d2a74-->
mit $\lambda_i \ge 0$ für Ungleichungen und $\nu_j \in \mathbb{R}$ mit beliebigem Vorzeichen für Gleichungen. Mit $\boldsymbol{\lambda} \succeq 0$ meinen wir, dass $\lambda_i \ge 0$ für jede Koordinate $i$ gilt.

<!--en:d1c47070-->
Minimieren wir die Lagrange-Funktion bei festen Multiplikatoren über $\mathbf{x}$, erhalten wir unsere untere Schranke als Funktion von $(\boldsymbol{\lambda}, \boldsymbol{\nu})$. Die **Lagrange-Dualfunktion** lautet

<!--en:49672262-->
In Worten gibt $q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ an, wie klein $f$ werden kann, wenn wir die harten Nebenbedingungen durch lineare Strafterme mit den Gewichten $\boldsymbol{\lambda}$ und $\boldsymbol{\nu}$ ersetzen. Für das Clipping-Problem ergab das :eqref:`eq_mdl-opt-ball-dual`. Im Allgemeinen muss das Infimum nicht angenommen werden, und es kann sogar $-\infty$ sein. Minimieren wir zum Beispiel $x$ unter der Nebenbedingung $-x \le 0$, so ist die Lagrange-Funktion $(1 - \lambda)x$ nach unten unbeschränkt, außer wenn $\lambda = 1$ gilt. Damit ist $q(1) = 0$ und $q(\lambda) = -\infty$ für alle anderen $\lambda$. Die Schranke $-\infty$ gilt trivialerweise, hilft uns aber nicht weiter. Die Menge der Multiplikatoren mit $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) > -\infty$ heißt *effektiver Definitionsbereich* (engl. effective domain) von $q$. Wenn wir weiter unten das duale Problem der Support Vector Machine herleiten, wird aus dieser impliziten Einschränkung des Definitionsbereichs eine explizite Gleichungsnebenbedingung.

<!--en:1eca180d-->
Zwei grundlegende Eigenschaften von $q$ gelten ganz ohne Annahmen an $f$, $g_i$ oder $h_j$. Nicht einmal Konvexität oder Differenzierbarkeit ist erforderlich.

<!--en:f61675fa-->
**Proposition (Konkavität der Dualfunktion).** *Für jedes primale Problem ist $q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ aus :eqref:`eq_mdl-opt-dual-function` konkav.*

<!--en:a6b2ddff-->
**Beweis.** Halten wir ein beliebiges $\mathbf{x} \in \mathcal{D}$ fest. Da die Multiplikatoren linear in $\mathcal{L}$ eingehen, ist die Abbildung $(\boldsymbol{\lambda}, \boldsymbol{\nu}) \mapsto \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ affin. Die Dualfunktion $q$ ist das punktweise Infimum dieser Familie affiner Funktionen. Ihr **Hypograph**, also die Menge aller Punkte auf oder unter ihrem Graphen, ist daher der Schnitt der Halbräume unter den Graphen dieser affinen Funktionen. Da jeder Schnitt konvexer Mengen konvex ist, ist $q$ konkav. $\blacksquare$

<!--en:b449d6fc-->
:numref:`fig_mdl-opt-dual-envelope` zeigt diese Konstruktion für das Clipping-Problem. Jede gestrichelte Gerade stellt die Lagrange-Funktion für ein festes $\mathbf{x}$ als Funktion von $\lambda$ dar. Die Kurve $q$ liegt unter allen diesen Geraden oder berührt sie. Die Zielfunktion des Clipping-Problems ist zwar konvex, doch das haben wir im Beweis nirgends verwendet. Die untere Einhüllende von Geraden ist unabhängig von der Form von $f$ und $g$ konkav.

<!--en:8dd7c903-->
![Die Dualfunktion des Clipping-Problems als untere Einhüllende. Jedes feste $\mathbf{x}$ liefert eine Gerade $\lambda \mapsto \mathcal{L}(\mathbf{x}, \lambda)$ (gestrichelt), und $q(\lambda)$ ist der kleinste Wert, den eine dieser Geraden bei $\lambda$ annimmt. Die gezeichneten Geraden gehören zu den Minimierern $\mathbf{x}(\lambda) = \mathbf{x}_0/(1+\lambda)$ für verschiedene Werte von $\lambda$. Jede berührt $q$ beim zugehörigen $\lambda$ (Punkte). Die Gerade zu $\mathbf{x}^\star = \mathbf{x}(4)$ verläuft wegen $g(\mathbf{x}^\star) = 0$ horizontal. Sie berührt $q$ im Maximum $d^\star = q(4) = p^\star = 8$.](../img/mdl-opt-dual-envelope.svg)

<!--en:4cc68c5a-->
**Proposition (schwache Dualität).** *Ist $\mathbf{x}$ zulässig und gilt $\boldsymbol{\lambda} \succeq 0$, so ist $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le f(\mathbf{x})$. Folglich gilt*

<!--en:480fe1c0-->
**Beweis.** Für jedes zulässige $\mathbf{x}$ und jedes $\boldsymbol{\lambda} \succeq 0$ gilt $\lambda_i g_i(\mathbf{x}) \le 0$ (wegen $\lambda_i \ge 0$ und $g_i(\mathbf{x}) \le 0$) sowie $\nu_j h_j(\mathbf{x}) = 0$. Daher gilt

<!--en:1bcd02d2-->
Nehmen wir links das Supremum über $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ mit $\boldsymbol{\lambda} \succeq 0$ und rechts das Infimum über alle zulässigen $\mathbf{x}$, erhalten wir :eqref:`eq_mdl-opt-weak-duality`. $\blacksquare$

<!--en:21660043-->
Mit dieser kurzen Ungleichung der schwachen Dualität haben wir beim Clipping-Problem jedes $q(\lambda)$ als untere Schranke nachgewiesen. Auf demselben Argument beruht in :numref:`subsec_mdl-kkt-conditions` der Beweis, dass die KKT-Bedingungen hinreichend sind. Setzen wir die KKT-Multiplikatoren ein, werden dank des komplementären Schlupfs beide Ungleichungen zu Gleichungen.

<!--en:569722bc-->
Die Maximierung von $q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ über $\boldsymbol{\lambda} \succeq 0$ und $\boldsymbol{\nu}$ heißt **duales Problem**. Da wir eine konkave Funktion über einer konvexen Menge maximieren, ist das duale Problem *immer* ein konvexes Optimierungsproblem, selbst wenn das primale Problem stark von der Konvexität abweicht. Sein Optimalwert $d^\star$ ist die schärfste untere Schranke, die diese Konstruktion liefern kann. Jedes dual zulässige Paar $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ ergibt eine gültige untere Schranke. Die nichtnegative Differenz $p^\star - d^\star \ge 0$ heißt **Dualitätslücke** (engl. duality gap). Beim Clipping-Problem war die Lücke null, im letzten Rechenbeispiel ist sie strikt positiv. Wann können wir schon im Voraus garantieren, dass die Lücke verschwindet?

<!--en:8d7df980-->
## Starke Dualität

<!--en:3c57ca05-->
Beim Clipping-Problem stimmte die beste duale Schranke genau mit dem primalen Optimum überein. Bei einem nichtkonvexen Problem kann sie strikt darunterliegen und damit nur einen Teil der Antwort liefern. Glücklicherweise genügen zwei schwache Voraussetzungen, um die Lücke zu schließen: Konvexität und ein einziger strikt zulässiger Punkt. Eine geometrische Betrachtung der erreichbaren Werte der Nebenbedingungs- und der Zielfunktion zeigt, wozu wir beide brauchen.

<!--en:9ad8f1db-->
### Die Slater-Bedingung

<!--en:ca1f9704-->
**Proposition (aus der Slater-Bedingung folgt starke Dualität).** *Das Problem sei konvex ($f$ und alle $g_i$ seien konvex, alle $h_j$ affin) und erfülle die Slater-Bedingung (:numref:`subsec_mdl-constraint-qualifications`): Es gebe einen zulässigen Punkt $\bar{\mathbf{x}}$ im relativen Inneren von $\mathcal{D}$ mit $g_i(\bar{\mathbf{x}}) < 0$ für jede nichtaffine Ungleichungsnebenbedingung $g_i$. Dann gilt $d^\star = p^\star$, und bei endlichem $p^\star$ wird das duale Optimum angenommen.*

<!--en:82193ac2-->
Das ist dieselbe Regularitätsbedingung für konvexe Probleme wie in :numref:`subsec_mdl-kkt-conditions`: Mindestens ein Punkt muss für alle gekrümmten Ungleichungsnebenbedingungen strikt im Inneren liegen. (Den vollständigen Beweis für mehrere Dimensionen gibt :citet:`Boyd.Vandenberghe.2004`, Abschnitt 5.3.2.) Beim Clipping-Problem ist die Slater-Bedingung leicht zu prüfen: Sowohl $f$ als auch $g$ sind konvex, und für den Ursprung $\mathbf{0}$ gilt $g(\mathbf{0}) = -\tfrac12 < 0$. Damit stand $q(4) = p^\star$ schon fest, bevor wir eine Ableitung berechnet hatten.

<!--en:5b53975c-->
Um zu verstehen, *warum* Konvexität und strikte Zulässigkeit die Lücke schließen, betrachten wir ein Problem mit einer einzigen Ungleichung $g(\mathbf{x}) \le 0$. Wir bilden jeden Punkt $\mathbf{x} \in \mathcal{D}$ auf das Wertepaar $(g(\mathbf{x}), f(\mathbf{x}))$ aus Nebenbedingungs- und Zielfunktion ab und nehmen alle Punkte hinzu, die oberhalb und rechts von einem solchen Paar liegen:

<!--en:13a2b897-->
In dieser $(u, t)$-Ebene entsprechen zulässige Punkte der Bedingung $u \le 0$. Das primale Optimum $p^\star$ ist also die geringste Höhe $t$, die $\mathcal{A}$ auf der vertikalen Achse $u = 0$ erreicht. Halten wir nun $\lambda \ge 0$ fest. Unter allen $(u, t)$, die ein gegebenes $\mathbf{x}$ beiträgt, nimmt die lineare Funktion $t + \lambda u$ ihr Minimum bei $(g(\mathbf{x}), f(\mathbf{x}))$ an. Damit gilt $q(\lambda) = \inf_{(u, t) \in \mathcal{A}} (t + \lambda u)$. Geometrisch verschieben wir eine Gerade $t + \lambda u = c$ mit Steigung $-\lambda$ von unten nach oben, bis sie $\mathcal{A}$ gerade stützt. Ihr Achsenabschnitt auf der vertikalen Achse $u = 0$ ist $q(\lambda)$. Da die gesamte Menge $\mathcal{A}$ auf oder über dieser Stützgeraden liegt, kann der Achsenabschnitt $q(\lambda)$ nie größer als $p^\star$ sein. Hier sehen wir die schwache Dualität unmittelbar (:numref:`fig_mdl-opt-primal-dual-gap`).

<!--en:63c7230c-->
![Dualität im Bild der Stützgeraden. Jedes $\mathbf{x}$ wird auf $(g(\mathbf{x}),f(\mathbf{x}))$ abgebildet. Die schattierte Menge $\mathcal{A}$ enthält diese Paare und alle Punkte oberhalb und rechts davon. Zur Auswertung der Dualfunktion bei $\lambda$ wandert eine Gerade mit Steigung $-\lambda$ nach oben, bis sie $\mathcal{A}$ von unten stützt. Ihr Achsenabschnitt bei $u=0$ ist $q(\lambda)$. (a) Clipping-Problem: $\mathcal{A}$ ist konvex und $\mathbf{0}$ strikt zulässig. Daher berührt die optimale Stützgerade die Menge bei $(0,p^\star)$ mit Steigung $-4$, und es gilt $d^\star=p^\star=8$. (b) Minimierung von $-x^2$ auf $[0,1]$ unter $x\le\tfrac12$: $\mathcal{A}$ ist nicht konvex. Die höchste Stützgerade verläuft deshalb strikt unter $(0,p^\star)$, sodass $d^\star=-\tfrac12<-\tfrac14=p^\star$ gilt.](../img/mdl-opt-primal-dual-gap.svg)

<!--en:fd143e5f-->
Mit diesem geometrischen Bild lässt sich die starke Dualität für eine einzelne Ungleichungsnebenbedingung unmittelbar beweisen.

<!--en:84ceb212-->
**Beweis (eine Ungleichungsnebenbedingung, $g(\bar{\mathbf{x}}) < 0$, $p^\star$ endlich).** Zunächst ist $\mathcal{A}$ konvex. Wählen wir zwei beliebige Punkte $(u_1, t_1)$ und $(u_2, t_2)$ in $\mathcal{A}$, die von $\mathbf{x}_1$ und $\mathbf{x}_2$ stammen, sowie ein beliebiges $\theta \in [0, 1]$. Wegen der Konvexität von $g$ gilt für den Zwischenpunkt $\mathbf{x}_\theta = \theta\mathbf{x}_1 + (1 - \theta)\mathbf{x}_2$ die Ungleichung $g(\mathbf{x}_\theta) \le \theta g(\mathbf{x}_1) + (1 - \theta) g(\mathbf{x}_2) \le \theta u_1 + (1 - \theta) u_2$. Die entsprechende Ungleichung gilt auch für $f$. Somit gehört die Konvexkombination von $(u_1, t_1)$ und $(u_2, t_2)$ zu $\mathcal{A}$.

<!--en:1b9651b0-->
Zweitens liegt $(0, p^\star)$ auf dem Rand von $\mathcal{A}$. Der Punkt ist Grenzwert einer Folge von Punkten aus $\mathcal{A}$, während $(0, p^\star - \varepsilon) \notin \mathcal{A}$ für jedes $\varepsilon > 0$ gilt. Durch jeden Randpunkt einer konvexen Menge verläuft eine Stützgerade: Es gibt einen Vektor $(\lambda, \gamma) \neq (0, 0)$ mit $\gamma t + \lambda u \ge \gamma p^\star$ für alle $(u, t) \in \mathcal{A}$ (:citet:`Boyd.Vandenberghe.2004`, Abschnitt 2.5.2). Auf derselben geometrischen Tatsache beruhen die Subgradienten in :numref:`sec_mdl-convexity`. Da $\mathcal{A}$ nach oben und rechts unbeschränkt ist, müssen $\gamma \ge 0$ und $\lambda \ge 0$ gelten.

<!--en:c053643c-->
Könnte $\gamma = 0$ gelten? Dann wäre die Stützgerade vertikal und es gälte $\lambda > 0$. Somit müsste $\lambda u \ge 0$ (und damit $u \ge 0$) überall in $\mathcal{A}$ gelten. Das widerspricht der Existenz des strikt zulässigen Punktes $\bar{\mathbf{x}}$, der $u = g(\bar{\mathbf{x}}) < 0$ beiträgt. Also gilt $\gamma > 0$. Division durch $\gamma$ ergibt $t + (\lambda/\gamma)\,u \ge p^\star$ für alle $(u, t) \in \mathcal{A}$, also $q(\lambda/\gamma) \ge p^\star$. Zusammen mit der schwachen Dualität folgt $q(\lambda/\gamma) = p^\star$: Das duale Optimum wird angenommen und ist gleich $p^\star$. $\blacksquare$

<!--en:50f6400b-->
Beachten wir, wie sich die beiden Voraussetzungen im Beweis ergänzen. Die Konvexität brauchen wir genau einmal: Sie macht $\mathcal{A}$ konvex, sodass es bei $(0, p^\star)$ eine Stützgerade gibt. Auch die Slater-Bedingung brauchen wir nur einmal: Sie schließt eine vertikale Stützgerade aus, die einem unendlichen Multiplikator entspräche. In Teilbild (a) von :numref:`fig_mdl-opt-primal-dual-gap` zum Clipping-Problem beträgt für Punkte der Norm $\|\mathbf{x}\| = r$ der kleinste Zielfunktionswert $\tfrac12(5 - r)^2$ und der Wert der Nebenbedingungsfunktion $u = \tfrac12(r^2 - 1)$. Der untere Rand von $\mathcal{A}$ ist daher die Kurve $t = \tfrac12\bigl(5 - \sqrt{1 + 2u}\bigr)^2$. Ihre Steigung bei $u = 0$ ist $-4$ und stimmt mit der Steigung der optimalen Stützgeraden, $-\lambda^\star = -4$, überein. Teilbild (b) greift dem nichtkonvexen Beispiel am Ende des Abschnitts vor. Dort krümmt sich $\mathcal{A}$ in die falsche Richtung, sodass jede Gerade unter $\mathcal{A}$ strikt unterhalb von $(0, p^\star)$ verläuft.

<!--en:fc55be48-->
### Sattelpunkte

<!--en:5d43e006-->
Starke Dualität lässt sich auch allein mit der Lagrange-Funktion formulieren, ohne die Dualfunktion zu erwähnen. Dazu schreiben wir das primale Problem als Minimax-Problem. Für jedes $\mathbf{x} \in \mathcal{D}$ gilt

<!--en:66d24b5c-->
Verletzt $\mathbf{x}$ eine Nebenbedingung, können wir den zugehörigen Multiplikator so skalieren, dass $\mathcal{L}$ gegen $+\infty$ geht. Ist $\mathbf{x}$ zulässig, erreichen wir mit den Multiplikatoren bestenfalls, dass für jede nichtaktive Nebenbedingung $\lambda_i = 0$ gilt. Dann bleibt $f(\mathbf{x})$. Folglich gilt $p^\star = \inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \mathcal{L}$, während nach Definition $d^\star = \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \inf_{\mathbf{x}} \mathcal{L}$ gilt. Schwache Dualität ist schlicht die Ungleichung $\sup \inf \le \inf \sup$, die für jede Funktion zweier Variablen gilt. Starke Dualität bedeutet, dass wir Minimierung und Maximierung vertauschen können, ohne den Wert zu ändern. Werden beide Optima angenommen, so ist diese Vertauschbarkeit gleichbedeutend mit der Existenz eines Sattelpunkts der Lagrange-Funktion.

<!--en:165d5173-->
**Proposition (starke Dualität und Sattelpunkt).** *Es gelte starke Dualität, und beide Optima seien angenommen, im primalen Optimum $\mathbf{x}^\star$ und im dualen Optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Dann ist $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ ein* **Sattelpunkt** *der Lagrange-Funktion: Für alle $\mathbf{x} \in \mathcal{D}$, alle $\boldsymbol{\lambda} \succeq 0$ und alle $\boldsymbol{\nu}$ gilt*

<!--en:37ec584e-->
*Umgekehrt schließt jeder Sattelpunkt von $\mathcal{L}$ über $\mathbf{x} \in \mathcal{D}$ und $(\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu})$ die Dualitätslücke, und der Sattelwert ist $p^\star = d^\star$.*

<!--en:b00567e3-->
**Beweis.** Aus der Definition von $q$ und der schwachen Dualität folgt

<!--en:c316cfea-->
wobei die letzte Ungleichung gilt, weil bei $\lambda_i^\star \ge 0$ und $g_i(\mathbf{x}^\star) \le 0$ jeder Strafterm nichtpositiv ist. Bei starker Dualität ($d^\star = p^\star$) müssen in dieser Kette überall Gleichheitszeichen stehen. Die Gleichheit links bedeutet, dass $\mathbf{x}^\star$ die Funktion $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ minimiert. Das ergibt die rechte Ungleichung in :eqref:`eq_mdl-opt-saddle-point`. Aus der Gleichheit rechts folgt $\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$, also komplementärer Schlupf. Für beliebige $\boldsymbol{\lambda} \succeq 0$ und $\boldsymbol{\nu}$ gilt deshalb $\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x}^\star) + \sum_i \lambda_i g_i(\mathbf{x}^\star)
\le f(\mathbf{x}^\star)
= \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Das liefert die linke Ungleichung in :eqref:`eq_mdl-opt-saddle-point`.

<!--en:22beb46f-->
Ist umgekehrt $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ ein Sattelpunkt, dann gilt

<!--en:8b6455d8-->
und damit $p^\star \le d^\star$. Die umgekehrte Ungleichung folgt aus der schwachen Dualität, also gilt $p^\star = d^\star$. $\blacksquare$

<!--en:6b69060b-->
In unserem Clipping-Beispiel ist $(\mathbf{x}^\star, \lambda^\star) = ((0.6, 0.8), 4)$ der Sattelpunkt: Bewegen wir $\mathbf{x}$ von $\mathbf{x}^\star$ weg, wächst $\mathcal{L}(\cdot, 4)$. Variieren wir hingegen $\lambda$, bleibt $\mathcal{L}(\mathbf{x}^\star, \lambda) = 8$ wegen $g(\mathbf{x}^\star) = 0$ konstant.

<!--en:ec0ee7dc-->
Starke Dualität hat drei praktische Vorteile:

<!--en:a38b5087-->
- *Das duale Problem kann einfacher oder kleiner sein als das primale.* Das duale Problem der Support Vector Machine hat eine Variable pro Datenpunkt, einfache Vorzeichenbedingungen und höchstens eine Gleichung. Das duale Problem der Ridge-Regression in Aufgabe :exref:`ex_mdl-lagrangian-duality-ridge-dual` reduziert sich auf ein lineares Gleichungssystem, dessen Größe der Anzahl der Beispiele statt der Anzahl der Merkmale entspricht.

<!--en:ea7c6fd1-->
- *Jeder dual zulässige Punkt liefert ein Abbruchkriterium.* Für jedes primal zulässige $\mathbf{x}$ und jedes dual zulässige $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ gibt die Lücke $f(\mathbf{x}) - q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ eine obere Schranke dafür an, wie weit $f(\mathbf{x})$ von $p^\star$ entfernt sein kann.

<!--en:a84c9e43-->
- *Aus der dualen Lösung lässt sich die primale Lösung gewinnen.* Nach :eqref:`eq_mdl-opt-saddle-point` gehört $\mathbf{x}^\star$ *zu den* Minimierern von $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Ist der Minimierer eindeutig (etwa wenn $f$ strikt konvex ist), erhalten wir sofort $\mathbf{x}^\star$, wie beim Clipping-Problem mit $\mathbf{x}(4) = \mathbf{x}_0 / 5 = \mathbf{x}^\star$. Andernfalls wählen Zulässigkeit und komplementärer Schlupf aus der Menge der Minimierer ein $\mathbf{x}^\star$ aus. Bei einem linearen Programm ist $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star)$ konstant in $\mathbf{x}$. Bei der SVM mit straftermfreiem Achsenabschnitt weiter unten bleibt der Achsenabschnitt nach der Minimierung der Lagrange-Funktion unbestimmt. Der komplementäre Schlupf legt ihn fest.

<!--en:a4c21721-->
Die Proposition über Sattelpunkte klärt auch die Notwendigkeit der KKT-Bedingungen, die sich in :numref:`subsec_mdl-kkt-conditions` für konvexe Probleme als hinreichend erwiesen haben. Ein konvexes Problem mit Funktionen $f$, $g_i$ und $h_j$, die auf $\mathcal{D} = \mathbb{R}^n$ differenzierbar sind, erfülle die Slater-Bedingung und nehme sein Optimum bei $\mathbf{x}^\star$ an. Dann gilt starke Dualität, und das duale Optimum wird bei $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ angenommen. Da $\mathbf{x}^\star$ die Funktion $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ auf ganz $\mathbb{R}^n$ minimiert, verschwindet dort ihr Gradient $\nabla_{\mathbf{x}} \mathcal{L}$. Wie wir gerade gesehen haben, gilt auch komplementärer Schlupf. Zusammen mit primaler und dualer Zulässigkeit sind das genau die vier KKT-Bedingungen. Unter der Slater-Bedingung sind sie sowohl notwendig als auch hinreichend.

<!--en:d4685887-->
Die Minimax-Struktur tritt sogar bei Deep-Learning-Problemen ohne explizite Nebenbedingungen auf. Generative adversariale Netze (:numref:`sec_basic_gan`) und adversariales Training lassen sich direkt als $\min_{\boldsymbol{\theta}} \max_{\boldsymbol{\phi}}$-Spiele formulieren :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014,Madry.Makelov.Schmidt.ea.2018`. Ohne Sattelpunkt können die Werte von $\min\max$ und $\max\min$ auseinanderliegen. Selbst wenn ein Sattelpunkt existiert, konvergiert das Verfahren nicht unbedingt, wenn wir gleichzeitig in der zu minimierenden Variablen einen Gradientenabstiegsschritt und in der zu maximierenden Variablen einen Aufstiegsschritt machen. Bei der einfachen bilinearen Zielfunktion $xy$ mit Sattelpunkt im Ursprung führt dieses Vorgehen für jede positive Schrittweite auf einer Spirale nach außen (Aufgabe :exref:`ex_mdl-lagrangian-duality-descent-ascent`).

<!--en:d3317854-->
## Multiplikatoren als Preise

<!--en:235d3159-->
Was ist eine Nebenbedingung wert? Wie viel näher kommen wir beim Clipping-Problem an $\mathbf{x}_0$ heran, wenn wir die Einheitskreisscheibe etwas vergrößern? Der optimale Lagrange-Multiplikator gibt darauf unmittelbar Antwort. Dieselbe Sensitivitätsbetrachtung zeigt sogar, dass bei konvexem Verlust Gewichtszerfall und eine Normbeschränkung der Gewichte genau dieselbe Familie von Lösungen liefern.

<!--en:1ef82dda-->
### Schattenpreise

<!--en:97163e32-->
Um den Wert jeder Nebenbedingung zu messen, ändern wir ihre rechte Seite und verfolgen, wie sich der optimale Zielfunktionswert verändert. Für Störvektoren $\mathbf{u} \in \mathbb{R}^m$ und $\mathbf{v} \in \mathbb{R}^p$ definieren wir den gestörten Optimalwert

<!--en:bf4ceac2-->
Bei $(\mathbf{0}, \mathbf{0})$ erhalten wir das ursprüngliche Optimum $p^\star(\mathbf{0}, \mathbf{0})$. Mit $u_i > 0$ lockern wir die $i$-te Ungleichungsnebenbedingung. Geometrisch kennen wir diese Funktion bereits: Für eine einzige Ungleichung ist $p^\star(u)$ die geringste Höhe $t$, die die Menge $\mathcal{A}$ aus :numref:`fig_mdl-opt-primal-dual-gap` bei der horizontalen Koordinate $u$ erreicht. Der untere Rand von $\mathcal{A}$ ist der Graph von $p^\star(u)$. Die Stützgerade mit Steigung $-\lambda^\star$ liegt überall auf oder unter diesem Graphen. Der folgende Satz verallgemeinert diese geometrische Ungleichung auf beliebig viele Nebenbedingungen.

<!--en:c38c6829-->
**Proposition (Multiplikatoren als Schattenpreise).** *Es gelte starke Dualität mit dualem Optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Dann gilt für alle $\mathbf{u}$ und $\mathbf{v}$*

<!--en:84e66b9f-->
*und wenn $p^\star$ bei $(\mathbf{0}, \mathbf{0})$ differenzierbar ist, gelten dort $\nabla_{\mathbf{u}} p^\star = -\boldsymbol{\lambda}^\star$ und $\nabla_{\mathbf{v}} p^\star = -\boldsymbol{\nu}^\star$.*

<!--en:278a6978-->
**Beweis.** Sei $\mathbf{x}$ ein beliebiger zulässiger Punkt des gestörten Problems. Dann gelten $g_i(\mathbf{x}) \le u_i$ und $h_j(\mathbf{x}) = v_j$. Aus der starken Dualität, der Definition von $q$ als Infimum, der Bedingung $\lambda_i^\star \ge 0$ und den gestörten Nebenbedingungen erhalten wir

<!--en:c4a5b3e5-->
Nehmen wir das Infimum über alle für das gestörte Problem zulässigen $\mathbf{x}$, erhalten wir :eqref:`eq_mdl-opt-shadow-price`. Ist $p^\star$ im Ursprung differenzierbar, so nimmt die nichtnegative Funktion $p^\star(\mathbf{u}, \mathbf{v}) - p^\star(\mathbf{0}, \mathbf{0}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v}$ ihr globales Minimum $0$ bei $(\mathbf{0}, \mathbf{0})$ an. Dort muss ihr Gradient also verschwinden. $\blacksquare$

<!--en:1ef9ea18-->
Der optimale Multiplikator $\lambda_i^\star$ heißt **Schattenpreis** der Nebenbedingung $i$. Er misst, wie schnell sich der Optimalwert verbessert, wenn wir die Nebenbedingung lockern. Die Schranke :eqref:`eq_mdl-opt-shadow-price` gilt global, also für Störungen *beliebiger* Größe. Lockern wir die Nebenbedingung $i$ um $u_i > 0$, kann der Optimalwert um höchstens $\lambda_i^\star u_i$ sinken. Verschärfen wir sie um denselben Betrag, steigt er um mindestens $\lambda_i^\star u_i$. Ist $p^\star$ differenzierbar, beträgt die Änderung in erster Ordnung $-\lambda_i^\star u_i$. Bei einer nichtaktiven Nebenbedingung gilt wegen des komplementären Schlupfs $\lambda_i^\star = 0$. Das leuchtet ein: Eine Nebenbedingung zu lockern, die ohnehin nicht bindet, kann das Optimum nicht verbessern.

<!--en:92c79868-->
Prüfen wir das am Clipping-Problem. Lockern wir $g(\mathbf{x}) \le 0$ zu $g(\mathbf{x}) \le u$, wächst der Radius der Kreisscheibe auf $r = \sqrt{1 + 2u}$. Damit ergibt sich $p^\star(u) = \tfrac12\bigl(5 - \sqrt{1 + 2u}\bigr)^2$, der bereits berechnete untere Rand von $\mathcal{A}$. Die Ableitung bei $u = 0$ ist tatsächlich $-4 = -\lambda^\star$. Lockern wir die Nebenbedingung um $u = \tfrac12$ (Kreisscheibe mit Radius $\sqrt2$), beträgt das neue Optimum $p^\star(\tfrac12) = \tfrac12(5 - \sqrt2)^2 \approx 6.43$. Es liegt damit über der globalen unteren Schranke $8 - 4 \cdot \tfrac12 = 6$, die :eqref:`eq_mdl-opt-shadow-price` garantiert.

<!--en:b2f67a97-->
Dabei kommt es auf die Skalierung der Nebenbedingung an. Hier ist $\lambda^\star = 4$ der Schattenpreis für $u$, also für das *halbe* Quadrat des Radius und nicht für den Radius $r$ selbst. Pro Radiuseinheit ergibt die Kettenregel $\mathrm{d}p^\star / \mathrm{d}r = -(5 - r)$. Das entspricht dem $r$-Fachen der Ableitung nach $u$ und stimmt nur wegen $r = 1$ mit $-\lambda^\star$ überein. Skalieren wir eine Nebenbedingung um, ändert sich ihr Multiplikator entsprechend: Schreiben wir $\|\mathbf{x}\|^2 - 1 \le 0$ ohne den Faktor $\tfrac12$, halbiert sich der Multiplikator auf $2$.

<!--en:3a522141-->
Das Problem des nächstgelegenen Punktes aus :numref:`subsec_mdl-closest-point` veranschaulicht den Fall einer Gleichung. Dort ist das Minimum von $\tfrac12\|\mathbf{x}\|^2$ auf der Hyperebene $\mathbf{a}^\top\mathbf{x} = b$ gleich $p^\star(b) = b^2 / (2\|\mathbf{a}\|^2)$. Die Beziehung $\partial p^\star / \partial b = -\nu^\star$ ergibt sich unmittelbar aus :eqref:`eq_mdl-opt-shadow-price` für die Störung $h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = v$, die die Hyperebene nach $b + v$ verschiebt. Das erklärt den Anstieg um $0.02 = 0.2 \cdot 0.1$, den wir in jenem Abschnitt numerisch beobachtet haben.

<!--en:cb335928-->
### Gewichtszerfall und Normbeschränkungen

<!--en:7c9c65b7-->
Im Deep Learning begrenzen wir die Modellkomplexität häufig, indem wir den Trainingsverlust $\ell(\mathbf{w})$ um einen Term für Gewichtszerfall (:numref:`sec_weight_decay`) erweitern oder die Norm der Gewichte durch eine harte Nebenbedingung beschränken. Mit der in diesem Buch verwendeten Konvention lauten die beiden Probleme

<!--en:4688b0a2-->
An einem stationären Punkt des Problems mit Strafterm gilt $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$. Führen wir für das beschränkte Problem einen Multiplikator $\lambda \ge 0$ zur Nebenbedingung $g(\mathbf{w}) = \tfrac12\|\mathbf{w}\|^2 - \tfrac12 r^2$ ein, ergibt sich die KKT-Stationaritätsbedingung $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$. Das ist dieselbe Gleichung, wobei der Koeffizient des Gewichtszerfalls die Rolle des Lagrange-Multiplikators übernimmt.

<!--en:450b5476-->
Ist der Verlust $\ell$ *konvex*, gilt die Äquivalenz in beiden Richtungen. Gehen wir zunächst vom Strafterm zur Nebenbedingung über. Sei $\mathbf{w}_\lambda$ ein Minimierer der Zielfunktion mit Strafterm. Wir wählen das Normbudget $r = \|\mathbf{w}_\lambda\|$. Dann ist $\mathbf{w}_\lambda$ zulässig, die Nebenbedingung ist aktiv (also gilt komplementärer Schlupf), und die KKT-Stationaritätsbedingung ist automatisch erfüllt. Da die KKT-Bedingungen für konvexe Probleme hinreichend sind (:numref:`subsec_mdl-kkt-conditions`), löst $\mathbf{w}_\lambda$ das beschränkte Problem.

<!--en:bce1c70c-->
Für den umgekehrten Weg wählen wir ein beliebiges Budget $r > 0$. Der Ursprung $\mathbf{w} = \mathbf{0}$ ist strikt zulässig, also gilt die Slater-Bedingung. Wird das beschränkte Minimum bei $\mathbf{w}^\star$ angenommen, so gibt es wegen der Notwendigkeit der KKT-Bedingungen einen Multiplikator $\lambda^\star(r) \ge 0$, und $\mathbf{w}^\star$ minimiert die konvexe Zielfunktion mit Strafterm $\ell(\mathbf{w}) + \tfrac{\lambda^\star}{2}\|\mathbf{w}\|^2$. Für $\lambda^\star > 0$ macht der quadratische Term die Zielfunktion strikt konvex, sodass $\mathbf{w}^\star$ ihr eindeutiger Minimierer ist. Wird das Budget nicht ausgeschöpft, gilt $\lambda^\star = 0$, und wir erhalten das Minimierungsproblem ohne Regularisierung. Nach der Proposition über Schattenpreise gilt $\lambda^\star(r) = -\partial p^\star / \partial(\tfrac12 r^2)$: Der Koeffizient des Gewichtszerfalls misst die marginale Abnahme des Trainingsverlusts pro Einheit der halben quadrierten Norm.

<!--en:325d14fc-->
Tatsächlich ist keine der beiden Richtungen an die $\ell_2$-Norm gebunden. Wir können $\tfrac12\|\mathbf{w}\|^2$ durch jeden konvexen Strafterm $\Omega(\mathbf{w})$ ersetzen, etwa durch die beim Lasso verwendete $\ell_1$-Norm $\|\mathbf{w}\|_1$ (:numref:`subsec_mdl-proximal-operators`). Jeder Minimierer $\mathbf{w}_\lambda$ von $\ell + \lambda\Omega$ löst das beschränkte Problem mit Budget $\Omega(\mathbf{w}) \le \Omega(\mathbf{w}_\lambda)$, denn für jedes $\mathbf{w}$ innerhalb dieses Budgets gilt $\ell(\mathbf{w}) \ge \ell(\mathbf{w}_\lambda) + \lambda\,(\Omega(\mathbf{w}_\lambda) - \Omega(\mathbf{w})) \ge \ell(\mathbf{w}_\lambda)$. Umgekehrt gilt für konvexes $\ell$ und jedes Budget strikt oberhalb von $\inf \Omega$ die Slater-Bedingung. Die Proposition über Sattelpunkte garantiert dann, dass jeder Minimierer des beschränkten Problems auch $\ell + \lambda^\star\Omega$ minimiert.

<!--en:b0e393c0-->
Für die vollständige Äquivalenz ist die Konvexität von $\ell$ jedoch wesentlich. Bei nichtkonvexem Verlust ist ein stationärer Punkt des Problems mit Strafterm weiterhin ein KKT-Punkt des zugehörigen beschränkten Problems. Die Mengen der globalen Lösungen können sich aber unterscheiden, und manche Normbudgets lassen sich durch *kein* Gewicht $\lambda$ des Strafterms erreichen (Aufgabe :exref:`ex_mdl-lagrangian-duality-penalty-nonconvex`).

<!--en:96d1aa2c-->
Derselbe quadratische Term lässt eine dritte Deutung zu. Bei quadratischem Verlust mit Rauschvarianz $\sigma^2$ und einem isotropen Gauß-Prior $\mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ für die Gewichte ist der Strafterm $\tfrac\lambda2\|\mathbf{w}\|^2$ mit $\lambda = \sigma^2/\tau^2$ bis auf eine additive Konstante und einen gemeinsamen Skalierungsfaktor die negative logarithmierte A-priori-Dichte (:numref:`sec_mdl-maximum_likelihood`). Eine einzige Zahl $\lambda$ lässt sich somit dreifach deuten: als Koeffizient des Gewichtszerfalls, als Lagrange-Multiplikator für ein Normbudget und als Verhältnis von Rauschvarianz zu A-priori-Varianz.

<!--en:16d30b1e-->
Prüfen wir die Äquivalenz numerisch. Für den konvexen Kleinste-Quadrate-Verlust $\ell(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2$ auf zufälligen Daten lösen wir das Problem mit Strafterm in geschlossener Form über $(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^\top\mathbf{b}$ und setzen $r = \|\mathbf{w}_\lambda\|$. Dann lösen wir das beschränkte Problem durch projizierten Gradientenabstieg mit Schrittweite $1/\lambda_{\max}(\mathbf{A}^\top\mathbf{A})$. Die Projektion auf die $r$-Kugel erfolgt durch Clipping. Liefert der Löser für das beschränkte Problem dieselben Gewichte? Stimmt der Multiplikator, den wir aus der Stationarität über das innere Produkt von $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ mit $\mathbf{w}$ als $\lambda = -\mathbf{w}^\top\nabla\ell(\mathbf{w}) / r^2$ erhalten, mit $\lambda$ überein?

<!--en:ba5f24c4-->
In jeder Zeile stimmt die Lösung des beschränkten Problems bis auf Maschinengenauigkeit mit der Lösung des Problems mit Strafterm überein. Der aus der Stationarität berechnete Multiplikator entspricht in allen sechs ausgegebenen Nachkommastellen $\lambda$ (die genauen Budgets $r$ hängen von der Zufallsstichprobe ab). Strafterm und Nebenbedingung parametrisieren denselben Lösungspfad. Der Lagrange-Multiplikator vermittelt zwischen den beiden Darstellungen.

<!--en:1a6399a9-->
## Die konvex Konjugierte

<!--en:483b3ac5-->
Um die Dualfunktion $q$ auszuwerten, müssen wir normalerweise für jede Wahl der Multiplikatoren ein neues Minimierungsproblem ohne Nebenbedingungen lösen. Bei linearen Nebenbedingungen können wir die gemeinsame Rechnung jedoch ein für alle Mal erledigen. Für die Nebenbedingungen $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$ lautet die Lagrange-Funktion $f(\mathbf{x}) + \boldsymbol{\lambda}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$, also gilt

<!--en:a119b51f-->
Das Infimum hängt von $f$ und dem einen Vektor $\mathbf{A}^\top\boldsymbol{\lambda}$ ab, nicht von $\mathbf{A}$ oder $\mathbf{b}$ einzeln. Berechnen wir für jede gängige Zielfunktion $f$ einmal den Wert von $\inf_{\mathbf{x}} [f(\mathbf{x}) - \mathbf{y}^\top\mathbf{x}]$ als Funktion von $\mathbf{y}$, können wir das duale Problem für *jedes* linear beschränkte Problem mit Zielfunktion $f$ direkt aus einer Tabelle ablesen. Diese Zuordnung ist die konvex Konjugierte.

<!--en:9e0ca218-->
### Stützgeraden

<!--en:7ffc00d6-->
Beginnen wir in einer Dimension mit $f(x) = \tfrac12 x^2$ und wählen eine Steigung $y = 2$. Welche Geraden $t = 2x + c$ mit Steigung $2$ liegen überall auf oder unter der Parabel? Wir brauchen $2x + c \le \tfrac12 x^2$ für alle $x$, also $c \le \min_x (\tfrac12 x^2 - 2x) = -2$, wobei das Infimum bei $x = 2$ angenommen wird. Die höchste solche Gerade, $t = 2x - 2$, berührt die Parabel tangential bei $(2, 2)$. Dort gilt $f'(2) = 2$. Verschieben wir eine Gerade mit Steigung $2$ nach unten, bis sie den Graphen gerade berührt, liegt ihr vertikaler Achsenabschnitt bei $-2$. Die Zahl $\max_x (2x - \tfrac12 x^2) = 2$ ist dieser Achsenabschnitt mit umgekehrtem Vorzeichen, wie :numref:`fig_mdl-opt-conjugate-intercept`(a) zeigt.

<!--en:f906f08d-->
![Die konvex Konjugierte im Bild der Stützgeraden. (a) Für $f(x)=\tfrac12x^2$ und Steigung $2$ wandert die gestrichelte Gerade mit Steigung $2$ nach unten, bis sie den Graphen bei $x=2$ berührt. Ihr Achsenabschnitt ist $-2$, also gilt $f^*(2)=2$. (b) Für $f(x)=|x|$ stützt eine Gerade mit Steigung $\tfrac12$ den Graphen im Ursprung mit Achsenabschnitt $0$. Eine Gerade mit Steigung $2$ steigt hingegen unabhängig vom Achsenabschnitt schneller als $|x|$. Somit gilt $f^*(\tfrac12)=0$ und $f^*(2)=+\infty$. Die Konjugierte ist also genau für die Steigungen endlich, für die eine Stützgerade unter dem Graphen existiert.](../img/mdl-opt-conjugate-intercept.svg)

<!--en:4d2a6fe7-->
Die **konvex Konjugierte** (auch Fenchel-Legendre-Transformierte) einer Funktion $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ gibt für jeden Steigungsvektor $\mathbf{y}$ diesen Achsenabschnitt mit umgekehrtem Vorzeichen an (wir verwenden einen Asterisk $*$ für Konjugierte und einen Stern $\star$ für Optima):

<!--en:ff5caa31-->
Geometrisch ist $f^*(\mathbf{y})$ der mit minus eins multiplizierte vertikale Achsenabschnitt der höchsten affinen Funktion mit Steigung $\mathbf{y}$, die überall unter dem Graphen von $f$ oder auf ihm liegt. Statt $f$ durch einzelne Funktionswerte anzugeben, beschreibt die Konjugierte $f$ durch die Familie ihrer Stützgeraden. Für $f(x) = \tfrac12 x^2$ und jede Steigung $y$ wird das Supremum bei $x = y$ angenommen. Das ergibt $f^*(y) = \tfrac12 y^2$: Diese quadratische Funktion ist zu sich selbst konjugiert.

<!--en:a533e28a-->
Wenn wir für $f$ auch den Wert $+\infty$ zulassen, können wir einen eingeschränkten Definitionsbereich einfach darstellen. Punkte mit $f(\mathbf{x}) = +\infty$ tragen nie zum Supremum bei. Eine solche Funktion mit Werten in den erweiterten reellen Zahlen ist genau dann konvex, wenn ihr Epigraph konvex ist.

<!--en:2b703676-->
Auch das Supremum selbst kann $+\infty$ sein. Betrachten wir $f(x) = |x|$ und die Steigung $y = 2$. Für $x > 0$ wächst $2x - |x| = x$ unbegrenzt. Keine Gerade mit Steigung $2$ kann unter dem V-förmigen Graphen bleiben, wie tief ihr Achsenabschnitt auch liegt. Gilt hingegen $|y| \le 1$, so ist $yx - |x| \le 0$ mit Gleichheit bei $x = 0$. Die Ursprungsgerade mit Steigung $y$ stützt also den Graphen. Daher gilt $f^*(y) = 0$ für $|y| \le 1$ und sonst $f^*(y) = +\infty$ (:numref:`fig_mdl-opt-conjugate-intercept`(b)). Die Konjugierte der Betragsfunktion ist also die *Indikatorfunktion* des Intervalls $[-1, 1]$: Sie hat innerhalb der Menge den Wert $0$ und außerhalb den Wert $+\infty$.

<!--en:a5ea470b-->
Aus :eqref:`eq_mdl-opt-conjugate` folgen unmittelbar zwei Eigenschaften. Erstens ist $f^*$ *immer* konvex, unabhängig davon, ob $f$ konvex ist. Für jedes feste $\mathbf{x}$ ist der Ausdruck $\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ affin in $\mathbf{y}$. Das punktweise Supremum affiner Funktionen ist konvex, denn sein Epigraph ist der Schnitt der Halbräume oberhalb ihrer Graphen. Das ist genau das Spiegelbild unseres Arguments für die Konkavität von $q$.

<!--en:f61859de-->
Zweitens ist $f^*(\mathbf{y})$ das Supremum über alle $\mathbf{x}$. Daher gilt $f^*(\mathbf{y}) \ge \mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ für alle $\mathbf{x}$ und $\mathbf{y}$. Umstellen ergibt die **Fenchel-Young-Ungleichung**,

<!--en:42eae714-->
Gleichheit gilt genau dann, wenn $\mathbf{x}$ das Supremum in :eqref:`eq_mdl-opt-conjugate` annimmt. Ist $f$ konvex und differenzierbar, so ist das genau für $\mathbf{y} = \nabla f(\mathbf{x})$ der Fall, denn dann ist $\mathbf{x} \mapsto \mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ konkav und erreicht sein Maximum dort, wo sein Gradient verschwindet. Auf die Konvexität kommt es an: Für das nichtkonvexe Doppelmuldenpotential $\phi_0(x) = (x^2 - 1)^2$ gilt $\phi_0'(0) = 0$, und doch ist $\phi_0^*(0) = -\min \phi_0 = 0$ und $\phi_0(0) + \phi_0^*(0) = 1 > 0$. Denn $x = 0$ ist ein lokales Maximum und kein Punkt, in dem eine horizontale Gerade den Graphen von unten stützt. Für das zu sich selbst konjugierte Paar $\tfrac12\|\mathbf{x}\|^2$ erhalten wir aus der Fenchel-Young-Ungleichung die bekannte Ungleichung $\mathbf{y}^\top\mathbf{x} \le \tfrac12\|\mathbf{x}\|^2 + \tfrac12\|\mathbf{y}\|^2$.

<!--en:acbc5454-->
Geht beim Übergang zur Konjugierten Information verloren? Jede konvexe Funktion $f$, die *eigentlich* (nirgends $-\infty$ und nicht identisch $+\infty$) und *abgeschlossen* ist (ihr Epigraph ist eine abgeschlossene Menge), ist durch ihre Stützgeraden vollständig bestimmt. Bilden wir die Konjugierte ein zweites Mal, erhalten wir wieder $f$: $f^{**} = f$ :cite:`Rockafellar.1970`. Ist $f$ nicht konvex, aber durch mindestens eine affine Funktion nach unten beschränkt, so ist $f^{**}$ die größte abgeschlossene konvexe Funktion unterhalb von $f$, also ihre konvexe Einhüllende (Aufgabe :exref:`ex_mdl-convexity-conjugate-pairs`d).

<!--en:95b3ffb0-->
In diesem Abschnitt und den zugehörigen Aufgaben verwenden wir vier Paare konjugierter Funktionen:

<!--en:b2bb5740-->
- $\tfrac12\|\mathbf{x}\|^2$ und $\tfrac12\|\mathbf{y}\|^2$, das zu sich selbst konjugierte Paar,

<!--en:81105514-->
- $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ und $\tfrac12\mathbf{y}^\top\mathbf{A}^{-1}\mathbf{y}$ für $\mathbf{A} \succ 0$ (angenommen bei $\mathbf{x} = \mathbf{A}^{-1}\mathbf{y}$),

<!--en:157c6b6b-->
- $e^x$ und $y \log y - y$ für $y > 0$, mit dem Wert $0$ bei $y = 0$ und $+\infty$ für $y < 0$ (angenommen bei $x = \log y$, also beispielsweise $f^*(1) = -1$ und $f^*(e) = 0$),

<!--en:9df6acd0-->
- $\lambda\|\mathbf{x}\|_1$ und die Indikatorfunktion des Quaders $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le \lambda\}$, das mehrdimensionale Gegenstück zur Betragsfunktion von oben.

<!--en:59df5aa5-->
### Duale Probleme bei linearen Nebenbedingungen

<!--en:5a8221ce-->
Kehren wir zu dem linear beschränkten Problem zurück, das uns auf die Konjugierte geführt hat: Wir minimieren $f(\mathbf{x})$ unter $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$. Ziehen wir ein Minuszeichen vor das Infimum, erhalten wir unmittelbar das Supremum aus der Definition von $f^*$:

<!--en:4184aaa8-->
Für Gleichungsnebenbedingungen $\mathbf{A}\mathbf{x} = \mathbf{b}$ ergibt dieselbe dreizeilige Rechnung $q(\boldsymbol{\nu}) = -f^*(-\mathbf{A}^\top\boldsymbol{\nu}) - \mathbf{b}^\top\boldsymbol{\nu}$, wobei $\boldsymbol{\nu}$ ein beliebiges Vorzeichen haben darf. Damit haben wir eine zweite Erklärung für die Konkavität von $q$: Die Funktion $f^*$ ist konvex, ihr Argument ist linear in $\boldsymbol{\lambda}$, und durch das Minuszeichen wird aus Konvexität Konkavität.

<!--en:b2742340-->
Erproben wir :eqref:`eq_mdl-opt-dual-conjugate` am kleinstmöglichen Beispiel. Wir suchen in einer Halbebene den Punkt, der dem Ursprung am nächsten liegt: $\min_{\mathbf{x}} \tfrac12\|\mathbf{x}\|^2$ unter $\mathbf{a}^\top\mathbf{x} \le b$ mit $\mathbf{a} = (1, 2)$ und $b = -1$. Da der Ursprung die Nebenbedingung verletzt ($\mathbf{a}^\top\mathbf{0} = 0 > -1$), muss diese im Optimum aktiv sein. Setzen wir $f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|^2$ und $\|\mathbf{a}\|^2 = 5$ in :eqref:`eq_mdl-opt-dual-conjugate` ein, erhalten wir die Dualfunktion in einem Schritt:

<!--en:53adc8f5-->
Aus $q'(\lambda) = 1 - 5\lambda = 0$ folgt $\lambda^\star = \tfrac15$ und $d^\star = \tfrac15 - \tfrac1{10} = \tfrac1{10}$. Der Minimierer der Lagrange-Funktion bei $\lambda^\star$ ist der Punkt, an dem das Supremum der Konjugierten angenommen wird, also $\mathbf{x} = -\lambda^\star\mathbf{a} = (-0.2, -0.4)$. Wir prüfen direkt nach: $\mathbf{a}^\top\mathbf{x} = -1 = b$ und $f(\mathbf{x}) = \tfrac12(0.04 + 0.16) = \tfrac1{10} = d^\star$. Damit folgt aus der schwachen Dualität, dass $\mathbf{x}$ optimal ist. Das stimmt mit der Projektionsformel $(b / \|\mathbf{a}\|^2)\,\mathbf{a}$ aus :numref:`subsec_mdl-closest-point` überein, hier für eine andere Hyperebene. Das primale Problem hatte zwei Variablen, das duale nur eine, und zu seiner Lösung genügte eine Zeile.

<!--en:28adf05c-->
### Log-Sum-Exp und Entropie

<!--en:f28e32a3-->
Im Deep Learning verwenden wir die Softmax-Funktion, um aus rohen Scores durch Exponenzieren und Normieren Wahrscheinlichkeiten zu berechnen. Warum gerade die Exponentialfunktion und keine andere positive Funktion? Die Konjugierte der Log-Sum-Exp-Funktion $\mathrm{lse}(\mathbf{x}) = \log \sum_i e^{x_i}$ gibt darauf eine einfache Antwort. Dabei bezeichne $\Delta = \{\mathbf{p} : \mathbf{p} \succeq 0,\ \mathbf{1}^\top\mathbf{p} = 1\}$ den Wahrscheinlichkeitssimplex und $H(\mathbf{p}) = -\sum_i p_i \log p_i$ die Shannon-Entropie.

<!--en:9985e750-->
Um $\mathrm{lse}^*(\mathbf{p}) = \sup_{\mathbf{x}} \bigl(\mathbf{p}^\top\mathbf{x} - \mathrm{lse}(\mathbf{x})\bigr)$ zu berechnen, prüfen wir zunächst, wann das Supremum endlich sein kann. Ist eine Koordinate $p_i$ negativ und gilt $x_i \to -\infty$, so strebt $\mathbf{p}^\top\mathbf{x}$ gegen $+\infty$, während $\mathrm{lse}(\mathbf{x})$ beschränkt bleibt. Das Supremum ist also $+\infty$. Gilt $\mathbf{1}^\top\mathbf{p} \neq 1$, so ergibt die Wahl des konstanten Vektors $\mathbf{x} = c\mathbf{1}$ den Ausdruck $c(\mathbf{1}^\top\mathbf{p} - 1) - \log n$ (wegen $\mathrm{lse}(c\mathbf{1}) = c + \log n$), der für $c \to \pm\infty$ wiederum unbegrenzt wächst. Somit ist $\mathrm{lse}^*(\mathbf{p})$ nur auf dem Wahrscheinlichkeitssimplex $\Delta$ endlich.

<!--en:4eea0f23-->
Liegt $\mathbf{p}$ im Inneren von $\Delta$ (alle $p_i > 0$), setzen wir den Gradienten nach $\mathbf{x}$ gleich null und erhalten $\mathrm{softmax}(\mathbf{x}) = \mathbf{p}$. Die Lösung lautet bis auf eine additive Konstante $\mathbf{x} = \log\mathbf{p}$ und liefert den Wert $\sum_i p_i \log p_i$. Liegt $\mathbf{p}$ auf dem Rand von $\Delta$ und gilt für mindestens eine Koordinate $p_i = 0$, existiert kein endlicher Maximierer, weil die Softmax-Funktion nur strikt positive Werte liefert. Im Grenzübergang $x_i \to -\infty$ für diese Koordinaten nähern wir uns dennoch demselben Wert. Mit der üblichen Konvention $0 \log 0 = 0$ setzt sich die Formel daher stetig auf ganz $\Delta$ fort. Da $\mathrm{lse}$ konvex und überall endlich ist, gilt $\mathrm{lse}^{**} = \mathrm{lse}$. Damit erhalten wir das konjugierte Paar

<!--en:5a8c8d02-->
In Worten besagt die zweite Zeile von :eqref:`eq_mdl-opt-lse-entropy`: Log-Sum-Exp ist das Maximum des erwarteten Scores $\mathbf{p}^\top\mathbf{x}$ zuzüglich eines Entropiebonus $H(\mathbf{p})$ über alle Verteilungen auf $n$ Ergebnissen. Welche Verteilung erreicht dieses Maximum? Nach der Gleichheitsbedingung der Fenchel-Young-Ungleichung ist es $\mathbf{p} = \nabla\mathrm{lse}(\mathbf{x})$. Ihr $i$-ter Eintrag lautet $\partial\,\mathrm{lse}/\partial x_i = e^{x_i} / \sum_j e^{x_j}$. Das ist genau die Softmax-Funktion (die zugehörige Hesse-Matrix haben wir in :numref:`sec_mdl-convexity` berechnet). Anders gesagt: Unter allen Verteilungen mit demselben erwarteten Score $\mathbf{p}^\top\mathbf{x}$ hat die Softmax-Verteilung die größte Entropie. Das entspricht der Herleitung über die maximale Entropie in :numref:`subsec_mdl-max-entropy-derivation` (in Aufgabe :exref:`ex_mdl-lagrangian-duality-max-entropy` leiten Sie die Softmax-Funktion aus dieser Nebenbedingung her). Ohne den Entropieterm legt die Maximierung von $\mathbf{p}^\top\mathbf{x}$ über $\Delta$ die gesamte Masse auf $\arg\max_i x_i$. Durch Hinzufügen der Entropie wird aus dieser harten Auswahl eine glatte Wahrscheinlichkeitsverteilung. Wenden wir dieselbe Konstruktion auf die konvexe erzeugende Funktion einer $f$-Divergenz an, erhalten wir die Variationsschranken für das Training von $f$-GANs (:numref:`sec_mdl-f-gan-dual`).

<!--en:20a08b01-->
## Rechenbeispiele

<!--en:320f6403-->
Drei Beispiele zeigen, was Dualität in der Praxis leistet. Bei der Support Vector Machine beschränkt der komplementäre Schlupf die Lösung auf wenige Datenpunkte, und in das duale Problem gehen die Daten nur über innere Produkte ein. Beim Water-Filling hat der Lagrange-Multiplikator eine konkrete technische Einheit: Übertragungsrate pro Leistungseinheit. Schließlich zeigt ein eindimensionales nichtkonvexes Problem die strikt positive Dualitätslücke, die Konvexität hier verhindert.

<!--en:f15cda19-->
### Das duale Problem der Support Vector Machine

<!--en:c216616e-->
Für linear trennbare Trainingsdaten $(\mathbf{x}_i, y_i)$ mit Labels $y_i \in \{\pm 1\}$ sucht die Support Vector Machine :cite:`Boser.Guyon.Vapnik.1992` die trennende Hyperebene $\mathbf{w}^\top\mathbf{x} + b = 0$ mit der größtmöglichen Margin. Ihr primales Problem lautet

<!--en:d5ee7aa7-->
Der Abstand eines Punktes $\mathbf{x}$ von der Hyperebene beträgt $|\mathbf{w}^\top\mathbf{x} + b| / \|\mathbf{w}\|$ (:numref:`sec_mdl-geometry-linear-algebraic-ops`). Die beiden Randhyperebenen $\mathbf{w}^\top\mathbf{x} + b = \pm 1$ der Margin liegen daher auf beiden Seiten der Entscheidungsgrenze im Abstand $1/\|\mathbf{w}\|$. Ihr Abstand voneinander beträgt also $2/\|\mathbf{w}\|$. Minimieren wir $\tfrac12\|\mathbf{w}\|^2$, maximieren wir folglich die Margin. Wir haben ein konvexes quadratisches Programm mit einer affinen Ungleichung pro Trainingspunkt. Bei affinen Nebenbedingungen verlangt die Slater-Bedingung lediglich einen zulässigen Punkt. Diesen erhalten wir, indem wir ein beliebiges trennendes Paar $(\mathbf{w}, b)$ so skalieren, dass die Margin jedes Punktes mindestens $1$ beträgt. Damit ist starke Dualität garantiert.

<!--en:e4ee6636-->
Für jede Nebenbedingung führen wir einen nichtnegativen Multiplikator $\alpha_i \ge 0$ ein und bilden die Lagrange-Funktion

<!--en:df442b4c-->
Um die Dualfunktion $q(\boldsymbol{\alpha}) = \inf_{\mathbf{w}, b} \mathcal{L}$ zu berechnen, beachten wir zunächst, dass $\mathcal{L}$ affin im straftermfreien Achsenabschnitt $b$ ist, mit Koeffizient $-\sum_i \alpha_i y_i$. Verschwindet dieser Koeffizient nicht, so können wir durch $b \to \pm\infty$ die Lagrange-Funktion $\mathcal{L}$ gegen $-\infty$ treiben. Also gilt $q(\boldsymbol{\alpha}) = -\infty$, sobald $\sum_i \alpha_i y_i \neq 0$. Hier sehen wir, wie sich der effektive Definitionsbereich von :eqref:`eq_mdl-opt-dual-function` auswirkt: Die Bedingung $\sum_i \alpha_i y_i = 0$ für einen endlichen Funktionswert wird zur expliziten Gleichungsnebenbedingung im dualen Problem. Auf diesem Bereich entfällt der $b$-Term. Übrig bleibt eine strikt konvexe quadratische Funktion von $\mathbf{w}$ mit eindeutigem Minimum dort, wo $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0}$ gilt:

<!--en:1afe8d33-->
Der optimale Gewichtsvektor ist also eine Linearkombination der Trainingseingaben mit ihren jeweiligen Multiplikatoren als Gewichten. Setzen wir diesen Ausdruck wieder in $\mathcal{L}$ ein, erhalten wir das **duale Problem der SVM**:

<!--en:8653be82-->
Schon bevor wir einen Löser wählen, ergeben sich zwei Konsequenzen. Erstens folgt aus dem komplementären Schlupf $\alpha_i (1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)) = 0$, dass jeder Punkt mit $\alpha_i > 0$ die Gleichung $y_i(\mathbf{w}^\top\mathbf{x}_i + b) = 1$ erfüllt und damit genau auf einer Randhyperebene der Margin liegt. Diese Punkte heißen **Stützvektoren** (engl. support vectors). In :numref:`fig_mdl-opt-svm-margin` kommen dafür nur die vier eingekreisten Punkte infrage. Ihre Nebenbedingungen sind aktiv, und ihre Normalen gleichen $\nabla f$ aus, wie in :numref:`fig_mdl-opt-kkt-active-set`. Dagegen fällt jeder Punkt mit $\alpha_i = 0$ aus $\mathbf{w}$ vollständig heraus. Zweitens gehen die Trainingseingaben in :eqref:`eq_mdl-opt-svm-dual` ausschließlich über innere Produkte $\mathbf{x}_i^\top\mathbf{x}_j$ ein. Ersetzen wir diese durch eine Kernel-Funktion $k(\mathbf{x}_i, \mathbf{x}_j)$, erhalten wir unmittelbar einen nichtlinearen Klassifikator. In der primalen Formulierung bleibt diese Struktur verborgen.

<!--en:191df6e3-->
![Maximum-Margin-Trennung der acht Punkte aus der folgenden Codezelle. Die durchgezogene Gerade ist die Entscheidungsgrenze $\mathbf{w}^{\star\top}\mathbf{x} + b^\star = 0$ mit $\mathbf{w}^\star = (\tfrac47, \tfrac47)$ und $b^\star = -\tfrac37$. Die gestrichelten Geraden sind die Randhyperebenen $\mathbf{w}^{\star\top}\mathbf{x} + b^\star = \pm 1$ der Margin mit Abstand $2/\|\mathbf{w}^\star\|$ voneinander. Nur bei den vier eingekreisten Punkten auf einer Randhyperebene kann $\alpha_i > 0$ gelten.](../img/mdl-opt-svm-margin.svg)

<!--en:8796a539-->
Die Gleichungsnebenbedingung $\sum_i \alpha_i y_i = 0$ entstand allein dadurch, dass die Lagrange-Funktion wegen des straftermfreien Achsenabschnitts $b$ in einer Richtung affin blieb. Im Code lösen wir eine eng verwandte Variante, bei der der dual zulässige Bereich einfach der nichtnegative Orthant ist. Dort setzt die Projektion lediglich negative Einträge auf null. Dazu nehmen wir den Achsenabschnitt in die Gewichte auf: Wir erweitern jede Eingabe um ein konstantes Merkmal gemäß $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$, setzen $\tilde{\mathbf{w}} = (\mathbf{w}, b)$ und minimieren $\tfrac12\|\tilde{\mathbf{w}}\|^2$ unter $y_i \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}_i \ge 1$. Da der Strafterm jetzt auch $b$ erfasst, kann die Entscheidungsgrenze geringfügig von der in :eqref:`eq_mdl-opt-svm-primal` abweichen. Dafür ist die Lagrange-Funktion in jeder primalen Variablen strikt konvex, und die Gleichungsnebenbedingung entfällt. Mit $\mathbf{Q}_{ij} = y_i y_j \tilde{\mathbf{x}}_i^\top \tilde{\mathbf{x}}_j$ erhalten wir als duales Problem

<!--en:b25ab82b-->
die Maximierung einer konkaven quadratischen Funktion auf dem nichtnegativen Orthanten. Beim projizierten Gradientenaufstieg kehren wir im Aktualisierungsschritt aus :numref:`subsec_mdl-projected-gd` das Vorzeichen der Schrittweite um und iterieren $\boldsymbol{\alpha} \leftarrow \max(\mathbf{0},\, \boldsymbol{\alpha} + \eta\,(\mathbf{1} - \mathbf{Q}\boldsymbol{\alpha}))$ mit der sicheren Schrittweite $\eta = 1/\lambda_{\max}(\mathbf{Q})$. Wenden wir den Aufstieg auf acht Punkte in der Ebene an und prüfen drei Vorhersagen: Trennt die so gewonnene Hyperebene die Daten? Gehören alle positiven Multiplikatoren zu Punkten mit Margin $1$? Stimmt der duale Wert mit dem primalen überein?

<!--en:9455b74d-->
Alle drei Vorhersagen treffen zu. Vier Multiplikatoren sind positiv, und genau diese vier Punkte haben Margin $1$. Die übrigen vier haben Margins zwischen $1.29$ und $2.43$ mit $\alpha_i = 0$. Die ausgegebenen Ziffern stimmen mit der exakten Lösung $\tilde{\mathbf{w}}^\star = (\tfrac47, \tfrac47, -\tfrac37)$ überein, also mit $\mathbf{w}^\star = (\tfrac47, \tfrac47)$ und $b^\star = -\tfrac37$. Dabei gilt $p^\star = d^\star = \tfrac{41}{98} \approx 0.418367$, und die Dualitätslücke liegt nahe $10^{-16}$. (Die Gleichheit von primalem und dualem Wert folgt auch unmittelbar aus der Identität $\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ im Optimum, siehe Aufgabe :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`a.)

<!--en:130d0530-->
Die Ausgabe zeigt auch, worauf es bei der Deutung dualer Lösungen ankommt. Weil die primale Zielfunktion $\tfrac12\|\tilde{\mathbf{w}}\|^2$ strikt konvex ist, ist das primale Optimum $\tilde{\mathbf{w}}^\star$ eindeutig. Das duale Optimum $\boldsymbol{\alpha}^\star$ ist es *nicht*. Hier gilt $\mathbf{Q} = \mathbf{G}\mathbf{G}^\top$, wobei $\mathbf{G}$ die $8 \times 3$-Matrix mit Zeilen $y_i\tilde{\mathbf{x}}_i^\top$ ist. Daher hat $\mathbf{Q}$ höchstens Rang $3$. Die vier Stützvektoren entsprechen vier Vektoren in $\mathbb{R}^3$ und sind somit linear abhängig. Es gibt also einen Vektor $\mathbf{d}$ ungleich null, der nur an diesen vier Indizes von null verschieden sein kann und $\mathbf{G}^\top\mathbf{d} = \mathbf{0}$ erfüllt. Verschieben wir $\boldsymbol{\alpha}^\star$ in Richtung $\mathbf{d}$, ändert sich der quadratische Term wegen $\mathbf{Q}\mathbf{d} = \mathbf{0}$ nicht. Auch der lineare Term bleibt gleich, denn jeder Stützvektor hat Margin $1$: $\mathbf{1}^\top\mathbf{d} = \sum_i d_i\, y_i\tilde{\mathbf{w}}^{\star\top}\tilde{\mathbf{x}}_i = \tilde{\mathbf{w}}^{\star\top}\mathbf{G}^\top\mathbf{d} = 0$. Somit ist eine ganze Strecke von Multiplikatorvektoren optimal. Alle liefern denselben Vektor $\tilde{\mathbf{w}}^\star$ und denselben dualen Wert. Der projizierte Aufstieg mit Startwert $\mathbf{0}$ liefert lediglich einen Punkt dieser Strecke. An jedem ihrer Endpunkte gilt für einen der vier Punkte mit Margin $1$ tatsächlich $\alpha_i = 0$ (Aufgabe :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`c). Der komplementäre Schlupf liefert also nur eine Implikation: Aus $\alpha_i > 0$ folgt Margin $1$, doch aus Margin $1$ folgt nicht zwingend $\alpha_i > 0$.

<!--en:c48a714f-->
### Water-Filling

<!--en:5b6754a6-->
Unser zweites Rechenbeispiel liefert eine geschlossene Lösung mit einem einprägsamen physikalischen Bild. Ein Sender soll eine Gesamtleistung $P$ auf $n$ unabhängige Übertragungskanäle verteilen. Kanal $i$ hat eine Hintergrundrauschleistung $N_i > 0$. Weisen wir ihm die Leistung $P_i \ge 0$ zu, erreichen wir eine Übertragungsrate von $\log(1 + P_i / N_i)$ Nat pro Kanalnutzung :cite:`Cover.Thomas.1999`. (Cover und Thomas verwenden für einen reellen Gauß-Kanal einen Faktor $\tfrac12$. Dadurch ändert sich die Skalierung des folgenden Lagrange-Multiplikators, die optimale Verteilung bleibt jedoch gleich.) Um die Gesamtrate zu maximieren, lösen wir

<!--en:70620c6d-->
Die Maximierung der konkaven Funktion $U$ entspricht der Minimierung der konvexen Funktion $-U$ unter affinen Nebenbedingungen. Da es eine strikt positive zulässige Verteilung gibt, gilt die Slater-Bedingung. Die KKT-Bedingungen sind also notwendig und hinreichend. Führen wir einen Multiplikator $\nu$ für die Budgetgleichung $\sum_i P_i - P = 0$ und Multiplikatoren $\lambda_i \ge 0$ für die Nichtnegativitätsbedingungen $-P_i \le 0$ ein, so lautet die Stationaritätsbedingung der Lagrange-Funktion zu $-U$

<!--en:4e29a72a-->
Nun lassen sich die beiden Fälle ablesen. Erhält Kanal $i$ positive Leistung ($P_i > 0$), folgt aus dem komplementären Schlupf $\lambda_i = 0$, also $N_i + P_i = 1/\nu$. Alle aktiven Kanäle erreichen genau denselben Gesamtpegel $1/\nu$. Erhält Kanal $i$ keine Leistung ($P_i = 0$), so verlangt die duale Zulässigkeit $\lambda_i = \nu - 1/N_i \ge 0$, dass $N_i \ge 1/\nu$ gilt. Bezeichnen wir die gemeinsame Schwelle mit $\theta = 1/\nu$, so erhält die optimale Verteilung die kompakte Form

<!--en:b5214aa9-->
Dies ist die klassische **Water-Filling**-Regel. Stellen wir uns jeden Kanal als Becken vor, dessen Boden auf Höhe $N_i$ liegt. In diese Becken gießen wir insgesamt das Wasservolumen $P$. Wie :numref:`fig_mdl-opt-water-filling` zeigt, stellt sich ein gemeinsamer Wasserspiegel $\theta$ ein. Die rauschärmsten Kanäle erhalten die meiste Leistung, und jedes Becken, dessen Boden über $\theta$ liegt, bleibt trocken. Die Funktion $\sum_i \max(\theta - N_i, 0)$ ist stetig und stückweise linear. Für $\theta \le \min_i N_i$ ist sie null, für $\theta > \min_i N_i$ wächst sie streng monoton. Deshalb bestimmt jedes Budget $P > 0$ einen eindeutigen Wasserspiegel $\theta$, den wir leicht durch Bisektion finden. (Die Simplexprojektion in :numref:`subsec_mdl-projected-gd` verwendet dieselbe Schwellenstruktur in umgekehrter Richtung: Von jeder Koordinate ziehen wir einen gemeinsamen Pegel ab und setzen negative Ergebnisse auf null.)

<!--en:b05e09bd-->
![Water-Filling für die Rauschpegel und das Budget aus der folgenden Codezelle. Bei der Verteilung von $P=3$ Leistungseinheiten auf Becken mit Bodenhöhen $N_i$ füllen sich die drei rauschärmsten Kanäle bis zum gemeinsamen Pegel $\theta=1/\nu^\star\approx1.43$. Jeder erhält die Leistung $P_i^\star=\theta-N_i$. Die beiden Becken mit Böden oberhalb von $\theta$ erhalten nichts, wie es der komplementäre Schlupf verlangt.](../img/mdl-opt-water-filling.svg)

<!--en:f6a6baeb-->
Die Proposition über Schattenpreise gibt $\nu^\star$ eine unmittelbare technische Bedeutung. Die Maximierung von $U$ entspricht der Minimierung von $-U$. Lockern wir das Budget von $P$ auf $P + v$, ändert sich $-U^\star$ mit der Rate $-\nu^\star$, also gilt $\mathrm{d}U^\star / \mathrm{d}P = \nu^\star = 1/\theta$. Der optimale Multiplikator gibt somit an, wie viel zusätzliche Übertragungsrate wir pro zusätzlicher Leistungseinheit gewinnen. Berechnen wir die Water-Filling-Verteilung für fünf Kanäle und prüfen diese Vorhersage des Schattenpreises mit endlichen Differenzen bei $P \pm 10^{-4}$.

<!--en:41b0307d-->
Das Leistungsbudget ist bis auf Maschinengenauigkeit ausgeschöpft. Die drei aktiven Kanäle erreichen alle den gemeinsamen Pegel $\theta = (3 + 0.1 + 0.4 + 0.8)/3 \approx 1.4333$. Die rauschstarken Kanäle bei $1.6$ und $2.5$ bleiben trocken, wie es die duale Zulässigkeit verlangt. Außerdem stimmt der Multiplikator $\nu^\star = 1/\theta = 0.697674$ in allen sechs ausgegebenen Ziffern mit der aus endlichen Differenzen berechneten Sensitivität $\mathrm{d}U^\star / \mathrm{d}P$ überein.

<!--en:9e5deb0e-->
### Eine explizite Dualitätslücke

<!--en:9c3ae3c3-->
Schwache Dualität gilt für jedes Problem, starke Dualität hingegen nicht. Ein eindimensionales nichtkonvexes Beispiel macht die Lücke sichtbar. Auf dem Definitionsbereich $\mathcal{D} = [0, 1]$ betrachten wir

<!--en:5b958b5c-->
Da $f(x) = -x^2$ konkav und nicht konvex ist, ist das Problem nicht konvex. Der zulässige Bereich ist $[0, \tfrac12]$, und das Minimum liegt am rechten Endpunkt: $p^\star = f(\tfrac12) = -\tfrac14$. Wir dualisieren nur die Nebenbedingung $g(x) \le 0$ und behalten $\mathcal{D} = [0, 1]$ als harte Beschränkung des Definitionsbereichs bei. Das Infimum in :eqref:`eq_mdl-opt-dual-function` erstreckt sich also über $[0, 1]$ (die schwache Dualität gilt unverändert). Für jedes $\lambda \ge 0$ ist die Lagrange-Funktion $\mathcal{L}(x, \lambda) = -x^2 + \lambda(x - \tfrac12)$ konkav in $x$. Ihr Minimum auf $[0, 1]$ muss daher an einem der beiden Endpunkte liegen: $\mathcal{L}(0, \lambda) = -\lambda/2$ oder $\mathcal{L}(1, \lambda) = \lambda/2 - 1$. Damit gilt

<!--en:8f787b2a-->
wie erwartet eine stückweise lineare, konkave Funktion. Die beiden Geraden schneiden sich bei $\lambda^\star = 1$, wo $d^\star = -\tfrac12 < -\tfrac14 = p^\star$ gilt. Es bleibt eine strikt positive Dualitätslücke von $\tfrac14$. Tragen wir $q(\lambda)$ zusammen mit $p^\star$ auf, um diese Dualfunktion für das nichtkonvexe Problem mit der Kurve des Clipping-Problems vom Anfang des Abschnitts zu vergleichen.

<!--en:4f77df99-->
Die Kurve des Clipping-Problems stieg glatt an, bis sie $p^\star$ berührte. Hier liegt die Spitze der zeltförmigen Kurve bei $-\tfrac12$, ein ganzes Viertel unter dem primalen Optimum. Einen strikt zulässigen Punkt gibt es durchaus: $x = \tfrac14$ liegt im Inneren von $\mathcal{D}$ und erfüllt $g(\tfrac14) = -\tfrac14 < 0$. Die Lücke entsteht also allein durch die fehlende Konvexität und nicht durch fehlende strikte Zulässigkeit im Sinne der Slater-Bedingung. Betrachten wir wieder Teilbild (b) von :numref:`fig_mdl-opt-primal-dual-gap`. Die Bildkurve $\{(u, -(u + \tfrac12)^2) : u \in [-\tfrac12, \tfrac12]\}$ krümmt sich nach unten. Deshalb kann keine Stützgerade unterhalb von $\mathcal{A}$ den Punkt $(0, p^\star)$ erreichen. Die höchste Stützgerade mit Steigung $-1$ liefert nur die Schranke $-\tfrac12$.

<!--en:95b911ce-->
Zwei Schwierigkeiten bleiben. Erstens liefert das duale Problem bei fehlender Konvexität eine Schranke für das Optimum, muss es aber nicht erreichen. Zweitens können gleichzeitige Aktualisierungen von $\mathbf{x}$ und $\boldsymbol{\lambda}$ selbst dann die Konvergenz verfehlen, wenn ein Sattelpunkt existiert. Das ist eine zentrale Schwierigkeit beim Minimax-Training. Das Training eines ganzen tiefen Netzes ist zwar nicht konvex, viele seiner Teilprobleme sind es jedoch: Projektionen, durch Clipping begrenzte Aktualisierungen, die Anpassung der letzten Schicht und konvexe Trust-Region-Teilprobleme. Auf sie können wir die Ergebnisse dieses Abschnitts direkt anwenden. Bei allen Herleitungen in diesem Kapitel sind wir bisher von exakter reeller Arithmetik ausgegangen. Im nächsten Abschnitt untersuchen wir, was passiert, wenn die endliche Gleitkommagenauigkeit das Ergebnis bestimmt und nicht der Optimierungsalgorithmus selbst.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:859aafdf-->
Bei jedem beschränkten Optimierungsproblem ist die Lagrange-Dualfunktion $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ konkav, und es gilt die schwache Dualität $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le p^\star$. Jede Wahl von $\boldsymbol{\lambda} \succeq 0$ liefert somit eine untere Schranke für das primale Optimum. Ist das primale Problem konvex und erfüllt es die Slater-Bedingung (ein strikt zulässiger Punkt für alle nichtaffinen Ungleichungen), so schließt die starke Dualität die Lücke ($d^\star = p^\star$). Werden beide Optima angenommen, bilden die primale und die duale Lösung gemeinsam einen Sattelpunkt der Lagrange-Funktion. Bei linearen Nebenbedingungen lässt sich die Dualfunktion durch die konvex Konjugierte ausdrücken: $q(\boldsymbol{\lambda}) = -f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}$. Aus konjugierten Paaren wie Log-Sum-Exp und negativer Entropie ergeben sich so unmittelbar duale Formulierungen. Bei starker Dualität wirken die optimalen Multiplikatoren als Schattenpreise ($\lambda_i^\star = -\partial p^\star / \partial u_i$ und $\nu_j^\star = -\partial p^\star / \partial v_j$ überall dort, wo $p^\star$ differenzierbar ist). Das erklärt, warum Gewichtszerfall bei konvexem Verlust einer Normbeschränkung entspricht, warum die SVM-Multiplikatoren außerhalb der Stützvektoren verschwinden und warum der Water-Filling-Multiplikator $1/\theta$ die zusätzliche Übertragungsrate pro Leistungseinheit misst. Ohne Konvexität kann eine strikt positive Dualitätslücke bleiben, wie die Minimierung von $-x^2$ auf $[0, 1]$ unter $x \le \tfrac12$ zeigt.

<!--en:2294243a-->
## Übungen

<!--en:3cb5eba8-->
1. **Konjugierte Paare.** Berechnen Sie die folgenden Konjugierten mit :eqref:`eq_mdl-opt-conjugate`.
    1. $f(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$ mit $\lambda > 0$.
    1. $f(\mathbf{x}) = \tfrac12 \mathbf{x}^\top \mathbf{A} \mathbf{x}$ mit $\mathbf{A} \succ 0$. Was besagt die Fenchel-Young-Ungleichung :eqref:`eq_mdl-opt-fenchel-young` für dieses Paar?
    1. Die Softplus-Funktion $f(t) = \log(1 + e^t)$ auf $\mathbb{R}$. Die binäre Kreuzentropie eines Logits $t$ bezüglich eines weichen Labels $p \in [0, 1]$ ist $\log(1 + e^t) - pt$. Zeigen Sie mit der Fenchel-Young-Ungleichung, dass sie mindestens so groß ist wie die binäre Entropie $H(p) = -p\log p - (1-p)\log(1-p)$. Gleichheit gilt genau dann, wenn $p = 1/(1 + e^{-t})$ gilt, also beim Wert der Sigmoidfunktion an der Stelle $t$. Schreiben Sie die Differenz als KL-Divergenz.
    1. Das Doppelmuldenpotential $\phi_0(x) = (x^2 - 1)^2$ aus :numref:`subsec_mdl-recognizing-convexity` (die dortige Funktion $\phi$ ohne den Neigungsterm $x/2$) ist nicht konvex. Bestimmen Sie seine konvexe Einhüllende $\phi_0^{**}$ und die Menge, auf der $\phi_0^{**} < \phi_0$ gilt.

   *Nach Boyd und Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), Beispiele 3.22 und 3.26.*
   :exlabel:`ex_mdl-convexity-conjugate-pairs`

<!--en:9843fc77-->
1. **Maximale Entropie und Temperatur.** Die Entropie eines Wahrscheinlichkeitsvektors $\mathbf{p} \in \mathbb{R}^n$ ist $H(\mathbf{p}) = -\sum_i p_i \log p_i$. In dieser Aufgabe leiten Sie die Softmax-Funktion aus den Nebenbedingungen her.
    1. Maximieren Sie $H$ unter $\sum_i p_i = 1$ mit einem Lagrange-Multiplikator. Lassen Sie dabei die Vorzeichenbedingungen $p_i \ge 0$ außer Acht. Zeigen Sie, dass die Gleichverteilung der einzige stationäre Punkt ist. Warum ändert sich das Ergebnis nicht, wenn Sie die Vorzeichenbedingungen außer Acht lassen?
    1. Wählen Sie Scores $s_1, \ldots, s_n$, die nicht alle gleich sind, und einen Zielwert $\bar s$ strikt zwischen $\min_i s_i$ und $\max_i s_i$. Fügen Sie die Nebenbedingung $\sum_i p_i s_i = \bar s$ hinzu. Zeigen Sie, dass der Maximierer $\mathbf{p} = \mathrm{softmax}(\beta\mathbf{s})$ ist, wobei der Skalar $\beta$ durch den Multiplikator der Score-Nebenbedingung bestimmt ist. Zeigen Sie, dass der mittlere Score unter $\mathrm{softmax}(\beta\mathbf{s})$ mit $\beta$ streng monoton wächst, sodass genau ein $\beta$ den Zielwert liefert.
    1. Sei $H^\star(\bar s)$ die maximale Entropie. Zeigen Sie, dass $\mathrm{d}H^\star / \mathrm{d}\bar s = -\beta$ gilt. Damit gibt der Multiplikator den Schattenpreis der Score-Nebenbedingung an, so wie der Multiplikator aus :numref:`subsec_mdl-closest-point` den Schattenpreis der Verschiebung $b$ angibt. Für welche Zielwerte $\bar s$ ist $\beta$ negativ? Was geschieht mit der Softmax-Temperatur $1/\beta$, wenn $\bar s$ gegen $\max_i s_i$ geht?
   :exlabel:`ex_mdl-lagrangian-duality-max-entropy`

<!--en:86524813-->
1. **Ein duales Zertifikat für die $\ell_1$-Minimierung.** Das Verfahren Basis Pursuit,

   $$\min_{\mathbf{x}}\; \|\mathbf{x}\|_1 \quad \textrm{subject to} \quad \mathbf{A}\mathbf{x} = \mathbf{b},$$

   mit $\mathbf{A} \in \mathbb{R}^{m \times n}$ und $m < n$ wählt eine dünn besetzte Lösung eines unterbestimmten linearen Gleichungssystems. Die Nebenbedingungen sind affin. Für die Slater-Bedingung genügt also ein zulässiger Punkt, und es gilt $d^\star = p^\star$, sobald das System eine Lösung hat. Die Konjugierte von $\|\cdot\|_1$ ist die Indikatorfunktion des Quaders $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le 1\}$.
    1. Zeigen Sie mithilfe der Gleichungsvariante von :eqref:`eq_mdl-opt-dual-conjugate`: Ersetzt man $\boldsymbol{\nu}$ durch $-\boldsymbol{\nu}$, so lautet das duale Problem $\max_{\boldsymbol{\nu}} \mathbf{b}^\top \boldsymbol{\nu}$ unter $\|\mathbf{A}^\top \boldsymbol{\nu}\|_\infty \le 1$.
    1. Lösen Sie beide Probleme für $\mathbf{A} = (1\;\; 2)$ und $b = 2$ von Hand und prüfen Sie $d^\star = p^\star$.
    1. Seien $\mathbf{x}^\star$ und $\boldsymbol{\nu}^\star$ optimal. Zeigen Sie, dass überall dort $x_i^\star = 0$ gilt, wo $|(\mathbf{A}^\top \boldsymbol{\nu}^\star)_i| < 1$ ist, und dass $(\mathbf{A}^\top \boldsymbol{\nu}^\star)_i = \mathrm{sign}(x_i^\star)$ gilt, wo $x_i^\star \neq 0$ ist. Gegeben sei ein Kandidat $\mathbf{x}$ mit $\mathbf{A}\mathbf{x} = \mathbf{b}$. Welche Bedingungen muss ein Vektor $\boldsymbol{\nu}$ erfüllen, damit er die Optimalität von $\mathbf{x}$ nachweist, ohne dass Sie das Problem lösen müssen?

<!--en:0d13d622-->
1. **Stützvektoren und dualer Wert.** Für das duale SVM-Problem mit einbezogenem Achsenabschnitt :eqref:`eq_mdl-opt-svm-dual-folded` sei $\boldsymbol{\alpha}^\star$ optimal, und es gelte $\tilde{\mathbf{w}}^\star = \sum_i \alpha_i^\star y_i \tilde{\mathbf{x}}_i$.
    1. Zeigen Sie $\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$, indem Sie die KKT-Bedingungen des dualen Problems auf dem Orthanten aufstellen. Folgern Sie, dass der duale Optimalwert mit dem primalen Wert $\tfrac12 \|\tilde{\mathbf{w}}^\star\|^2$ übereinstimmt.
    1. Zeigen Sie: Entfernen Sie einen Trainingspunkt mit $\alpha_j^\star = 0$, bleiben die übrigen Multiplikatoren für das kleinere Problem optimal, und $\tilde{\mathbf{w}}^\star$ bleibt unverändert. Bei der Leave-one-out-Validierung wird jeder der $n$ Punkte mit der SVM klassifiziert, die auf den jeweils anderen $n - 1$ Punkten trainiert wurde. Folgern Sie, dass die Leave-one-out-Fehlerrate für jedes optimale $\boldsymbol{\alpha}^\star$ höchstens so groß ist wie der Anteil der Punkte mit $\alpha_i^\star > 0$. Welches optimale $\boldsymbol{\alpha}^\star$ liefert die beste Schranke?
    1. [code] Die Matrix $\mathbf{Q}$ aus der SVM-Codezelle hat Rang $3$. Sei $S$ die Menge der vier Indizes mit $\alpha_i^\star > 0$ und $\mathbf{G}_S$ die Matrix aus den zugehörigen Zeilen von `G`. Bestimmen Sie einen Vektor $\mathbf{d} \in \mathbb{R}^4$ ungleich null mit $\mathbf{G}_S^\top\mathbf{d} = \mathbf{0}$ (wenden Sie `np.linalg.svd` auf $\mathbf{G}_S^\top$ an). Durchlaufen Sie die Gerade $\boldsymbol{\alpha}^\star_S + t\mathbf{d}$ in jeder Richtung, bis eine Koordinate null erreicht. Prüfen Sie, dass beide Endpunkte dual zulässig sind, den dualen Wert $\tfrac{41}{98}$ haben und dasselbe $\tilde{\mathbf{w}}^\star$ liefern. Für welchen Punkt mit Margin $1$ gilt am jeweiligen Endpunkt $\alpha_i = 0$? Was bedeutet das für die Frage, ob sich Stützvektoren an den Multiplikatoren erkennen lassen? (Sie sollten die Endpunkte $(0, 0.204, 0, 0, 0, 0.192, 0, 0.441)$ und $(0.204, 0, 0, 0, 0, 0.273, 0, 0.359)$ erhalten.)
   :exlabel:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`

<!--en:f06790ef-->
1. [code] **Gradientenabstieg und -aufstieg.** ● Primal-duale Verfahren und das GAN-Training aktualisieren gleichzeitig die zu minimierende Variable durch Gradientenabstieg und die zu maximierende Variable durch Gradientenaufstieg.
    1. Schreiben Sie für das bilineare Spiel $\min_x \max_y\, xy$ mit Sattelpunkt im Ursprung den gleichzeitigen Abstieg und Aufstieg $x \leftarrow x - \eta y$ und $y \leftarrow y + \eta x$ als lineare Abbildung. Zeigen Sie, dass sich die Iterierten von jedem Startpunkt außer dem Ursprung für jedes $\eta > 0$ spiralförmig vom Sattelpunkt entfernen. Zeigen Sie, dass die alternierende Variante, bei der die Aktualisierung von $y$ bereits das neue $x$ verwendet, eine lineare Abbildung mit Determinante $1$ ist. Wie verhalten sich ihre Iterierten für $\eta < 2$?
    1. Wenden Sie den gleichzeitigen Abstieg und Aufstieg auf die Lagrange-Funktion $\mathcal{L}(\mathbf{x}, \nu) = \tfrac12\mathbf{x}^\top \mathbf{x} + \nu(\mathbf{a}^\top \mathbf{x} - b)$ des Problems des nächstgelegenen Punktes aus :numref:`subsec_mdl-closest-point` an. Verwenden Sie $\mathbf{a} = (1, 2)$ und $b = 1$ mit den Startwerten $\mathbf{x} = \mathbf{0}$ und $\nu = 0$. Bestimmen Sie die Schrittweiten, für die die Iterierten gegen $(\mathbf{x}^\star, \nu^\star)$ konvergieren. Leiten Sie diese aus den Eigenwerten der Aktualisierungsabbildung ab, die auf das Paar $(\mathbf{a}^\top \mathbf{x} / \|\mathbf{a}\|, \nu)$ und auf die zu $\mathbf{a}$ orthogonalen Komponenten von $\mathbf{x}$ wirkt.
    1. Addieren Sie $\tfrac{\epsilon}{2} x^2$ mit $0 < \epsilon \le 2$ zur Zielfunktion des bilinearen Spiels. Für welche Schrittweiten konvergieren gleichzeitiger Abstieg und Aufstieg jetzt? Welcher Term der Lagrange-Funktion aus der vorherigen Teilaufgabe übernimmt die Rolle von $\epsilon$?
   :exlabel:`ex_mdl-lagrangian-duality-descent-ascent`

<!--en:1194be41-->
1. [code] **Strafterm und Nebenbedingung ohne Konvexität.** ● Für einen konvexen Verlust zeigt :numref:`subsec_mdl-weight-decay-duality`, dass Gewichtszerfall und Normbeschränkung denselben Lösungspfad liefern. Betrachten Sie stattdessen den nichtkonvexen eindimensionalen Verlust $\ell(w) = (w - 1)^2 (w - 3)^2 - w$ mit lokalen Minima nahe $w = 1.16$ und $w = 3.11$.
    1. Berechnen Sie auf einem feinen Gitter für $w$ die Minimierer $w_\lambda = \mathrm{argmin}_w\, \ell(w) + \tfrac\lambda2 w^2$ der Zielfunktion mit Strafterm für Werte von $\lambda$ auf einem logarithmischen Gitter sowie das beschränkte Optimum $p^\star(r) = \min \{\ell(w) : w^2 \le r^2\}$ für $r \in [0, 3.2]$. Zeichnen Sie die Punkte $(w_\lambda^2, \ell(w_\lambda))$ in das Diagramm der Kurve $(r^2, p^\star(r))$ ein. Welche Budgets $r$ treten nie als Norm $|w_\lambda|$ eines Minimierers der Zielfunktion mit Strafterm auf?
    1. Zeigen Sie, dass $w_\lambda$ genau dann $\ell(w) + \tfrac\lambda2 w^2$ minimiert, wenn die Gerade mit Steigung $-\tfrac\lambda2$ durch $(w_\lambda^2, \ell(w_\lambda))$ unter allen Punkten der Menge $\{(w^2, \ell(w)) : w \in \mathbb{R}\}$ liegt oder sie berührt. Erklären Sie damit und mit dem Bild der Stützgeraden in :numref:`fig_mdl-opt-primal-dual-gap` die fehlenden Budgets.
    1. Wählen Sie im fehlenden Bereich ein Budget $r$, bei dem die Nebenbedingung aktiv ist. Berechnen Sie den KKT-Multiplikator $\lambda$ der beschränkten Lösung. Schreiben Sie dazu die Nebenbedingung wie im Text als $\tfrac12 w^2 \le \tfrac12 r^2$. Finden Sie einen Punkt, an dem die Zielfunktion mit diesem Strafgewicht $\lambda$ kleiner ist. Berechnen Sie die Dualitätslücke $p^\star(r) - d^\star(r)$ des beschränkten Problems.
   :exlabel:`ex_mdl-lagrangian-duality-penalty-nonconvex`

<!--en:c7b7aec4-->
1. [extended] **Entropisch regularisierter optimaler Transport.** ●● Für gegebene Wahrscheinlichkeitsvektoren $\mathbf{p} \in \mathbb{R}^n$ und $\mathbf{r} \in \mathbb{R}^m$ und eine Kostenmatrix $\mathbf{C} \in \mathbb{R}^{n \times m}$ sucht der optimale Transport den günstigsten Plan $\boldsymbol{\Gamma}$, der die Masse $\mathbf{p}$ in die Masse $\mathbf{r}$ überführt:

   $$\mathrm{OT} = \min_{\boldsymbol{\Gamma} \ge 0}\; \sum_{i,j} C_{ij} \Gamma_{ij} \quad \textrm{subject to} \quad \boldsymbol{\Gamma} \mathbf{1} = \mathbf{p}, \;\; \boldsymbol{\Gamma}^\top \mathbf{1} = \mathbf{r},$$

   Dies ist ein lineares Programm mit $nm$ Variablen. Die entropische Regularisierung :cite:`Cuturi.2013` addiert $\varepsilon \sum_{i,j} \Gamma_{ij} (\log \Gamma_{ij} - 1)$ mit $\varepsilon > 0$ zur Zielfunktion. Sei $\boldsymbol{\Gamma}_\varepsilon$ das regularisierte Optimum. Seine Berechnung mit Sinkhorn-Iterationen dient zum Vergleich von Verteilungen :cite:`Peyre.Cuturi.2019`.
    1. Bilden Sie die Lagrange-Funktion mit den Termen $\boldsymbol{\phi}^\top (\mathbf{p} - \boldsymbol{\Gamma} \mathbf{1}) + \boldsymbol{\psi}^\top (\mathbf{r} - \boldsymbol{\Gamma}^\top \mathbf{1})$ für die Nebenbedingungen an die Randverteilungen. Zeigen Sie $\boldsymbol{\Gamma}_\varepsilon = \mathrm{diag}(\mathbf{u})\, \mathbf{K}\, \mathrm{diag}(\mathbf{v})$ mit $K_{ij} = e^{-C_{ij} / \varepsilon}$, $u_i = e^{\phi_i / \varepsilon}$ und $v_j = e^{\psi_j / \varepsilon}$. Warum können Sie die Vorzeichenbedingungen $\Gamma_{ij} \ge 0$ in der Lagrange-Funktion weglassen?
    1. Zeigen Sie, dass die Dualfunktion

       $$q(\boldsymbol{\phi}, \boldsymbol{\psi}) = \mathbf{p}^\top \boldsymbol{\phi} + \mathbf{r}^\top \boldsymbol{\psi} - \varepsilon \sum_{i,j} e^{(\phi_i + \psi_j - C_{ij}) / \varepsilon},$$

       lautet und dass ihre exakte Maximierung über $\boldsymbol{\phi}$ bei festem $\boldsymbol{\psi}$ die elementweise Sinkhorn-Aktualisierung $\mathbf{u} = \mathbf{p} / (\mathbf{K} \mathbf{v})$ liefert. Folgern Sie, dass der Sinkhorn-Algorithmus einen blockweisen Koordinatenaufstieg auf einer konkaven Funktion ausführt und dass sein dualer Wert nie sinkt. Warum ist jeder duale Wert eine untere Schranke für $\mathrm{OT}$?
    1. Sei $H(\boldsymbol{\Gamma}) = -\sum_{i,j} \Gamma_{ij} \log \Gamma_{ij}$. Verwenden Sie ohne Beweis, dass für jeden zulässigen Plan $\max(H(\mathbf{p}), H(\mathbf{r})) \le H(\boldsymbol{\Gamma}) \le H(\mathbf{p}) + H(\mathbf{r})$ gilt. Zeigen Sie $0 \le \sum_{i,j} C_{ij} (\boldsymbol{\Gamma}_\varepsilon)_{ij} - \mathrm{OT} \le \varepsilon \min(H(\mathbf{p}), H(\mathbf{r}))$.
    1. Die Kosten liegen in $[0, 1]$ liegen. In float64 wird $e^{-x}$ für Werte von $x$ oberhalb von etwa $745$ auf null gerundet. Für welche $\varepsilon$ können Einträge von $\mathbf{K}$ verschwinden, und was geschieht dann bei den multiplikativen Aktualisierungen $\mathbf{u} = \mathbf{p} / (\mathbf{K}\mathbf{v})$? Erklären Sie, warum sich das Problem vermeiden lässt, wenn Sie $\boldsymbol{\phi}$ und $\boldsymbol{\psi}$ direkt aktualisieren und anstelle der Produkte mit $\mathbf{K}$ Log-Sum-Exp verwenden.
   :exlabel:`ex_mdl-lagrangian-duality-entropic-ot`

<!--en:35adf513-->
1. [code] **Das duale Problem der Ridge-Regression.** Schreiben Sie $\min_{\mathbf{w}}\tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2 + \tfrac\lambda2\|\mathbf{w}\|^2$ mit $\mathbf{A} \in \mathbb{R}^{n \times d}$ als $\min_{\mathbf{w}, \mathbf{z}}\tfrac12\|\mathbf{z}\|^2 + \tfrac\lambda2\|\mathbf{w}\|^2$ unter der Nebenbedingung $\mathbf{A}\mathbf{w} - \mathbf{b} - \mathbf{z} = \mathbf{0}$ mit Multiplikator $\boldsymbol{\alpha} \in \mathbb{R}^n$.
    1. Minimieren Sie die Lagrange-Funktion über $\mathbf{z}$ und $\mathbf{w}$, um $\mathbf{z} = \boldsymbol{\alpha}$, $\mathbf{w} = -\mathbf{A}^\top\boldsymbol{\alpha}/\lambda$ und die Dualfunktion $q(\boldsymbol{\alpha}) = -\tfrac12\|\boldsymbol{\alpha}\|^2 - \tfrac1{2\lambda}\boldsymbol{\alpha}^\top\mathbf{A}\mathbf{A}^\top\boldsymbol{\alpha} - \mathbf{b}^\top\boldsymbol{\alpha}$ zu erhalten. Warum gilt starke Dualität?
    1. Maximieren Sie $q$, um $\boldsymbol{\alpha}^\star = -\lambda(\mathbf{A}\mathbf{A}^\top + \lambda\mathbf{I})^{-1}\mathbf{b}$ und $\mathbf{w}^\star = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top + \lambda\mathbf{I})^{-1}\mathbf{b}$ zu erhalten. Erklären Sie, warum die Lösung eine Linearkombination der Zeilen von $\mathbf{A}$ ist, so wie der SVM-Gewichtsvektor eine Linearkombination der Datenpunkte ist. Warum gehen die Daten nur über $\mathbf{A}\mathbf{A}^\top$ ein?
    1. Prüfen Sie für $n = 20$ Beispiele und $d = 50$ Merkmale numerisch, dass die Lösung des primalen Systems $(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^\top\mathbf{b}$ und die duale Formel bis auf etwa $10^{-14}$ übereinstimmen und dass die primalen und dualen Werte gleich sind. Welcher Lösungsweg ist für $n \ll d$ günstiger? Was erhalten Sie, wenn Sie $\mathbf{A}\mathbf{A}^\top$ durch eine Kernel-Matrix ersetzen?
   :exlabel:`ex_mdl-lagrangian-duality-ridge-dual`

<!--en:b81ac181-->
[Diskussion](https://d2l.discourse.group/t/36860)

<!--en:5c6046df-->
Dualität<br>**Lagrange-Dualfunktion · schwache und starke Dualität · Slater-Bedingung · Schattenpreise · Fenchel-Konjugierte · SVM und Water-Filling**

<!--en:540d09bf-->
::: {.slide title="Optimalitätsnachweis bei nichtverschwindendem Gradienten"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:0bb3660b-->
Ein iterativer Löser für $\min_{\|\mathbf{x}\|_2 \le 1}\tfrac12\|\mathbf{x} - \mathbf{x}_0\|_2^2$ ($\mathbf{x}_0 = (3, 4)$) halte am zulässigen Punkt $\mathbf{x} = (0.55, 0.835)$ mit $f(\mathbf{x}) = 8.01 \ge p^\star$ an:

<!--en:f701a7a6-->
- **Warum der Test ohne Nebenbedingungen versagt:** Am beschränkten Optimum $\mathbf{x}^\star = (0.6, 0.8)$ gilt $\nabla f(\mathbf{x}^\star) = (-2.4, -3.2) \neq \mathbf{0}$.

<!--en:3e61db22-->
- **Nachweis durch eine duale untere Schranke:** Für jedes $\lambda \ge 0$ gilt auf der Kreisscheibe $\lambda g(\mathbf{x}) \le 0$. Die Minimierung von $f(\mathbf{x}) + \lambda g(\mathbf{x})$ über ganz $\mathbb{R}^2$ liefert somit die garantierte untere Schranke $q(\lambda) \le p^\star$.

<!--en:f595dfb0-->
::: {.slide title="Untere Schranken durch Einbeziehen der Nebenbedingungen"}

<!--en:1ca48427-->
[Das duale Problem]{.kicker}

<!--en:09442e20-->
Für das Clipping-Problem ($f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$, $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$, $\|\mathbf{x}_0\|^2 = 25$) ein beliebiges $\lambda \ge 0$ festhalten:

<!--en:b0f4da25-->
- **Geschlossene Lösung ohne Nebenbedingung:** Aus $\nabla_{\mathbf{x}}\mathcal{L}(\mathbf{x}, \lambda) = (\mathbf{x} - \mathbf{x}_0) + \lambda\mathbf{x} = \mathbf{0}$ ergibt sich der eindeutige Minimierer $\mathbf{x}(\lambda) = \frac{\mathbf{x}_0}{1 + \lambda}$.

<!--en:a6f7d250-->
- **Auswertung der unteren Schranke $q(\lambda) = \mathcal{L}(\mathbf{x}(\lambda), \lambda)$:**
  $$q(\lambda) = \frac{25}{2}\frac{\lambda^2}{(1+\lambda)^2} + \frac{\lambda}{2}\left(\frac{25}{(1+\lambda)^2} - 1\right) = \frac{25}{2}\frac{\lambda}{1 + \lambda} - \frac{\lambda}{2}$$

<!--en:396871de-->
- **Schärfste untere Schranke:** Aus $q'(\lambda) = \frac{25}{2(1+\lambda)^2} - \frac12 = 0$ folgt $\lambda^\star = 4$. Dort gilt $q(4) = 10 - 2 = \mathbf{8 = p^\star}$. Somit weicht $8.01$ höchstens um $0.01$ vom Optimum ab.

<!--en:d8f6f97d-->
::: {.slide title="Die Dualfunktion für die Einheitskreisscheibe"}

<!--en:1ca48427-2-->
[Das duale Problem]{.kicker}

<!--en:ef3dec2e-->
Jedes $\lambda \ge 0$ liefert eine gültige untere Schranke $q(\lambda) \le p^\star = 8$:

<!--en:b9addcaf-->
- Für $\lambda < 4$ liegt $\mathbf{x}(\lambda)$ außerhalb der Kreisscheibe ($g > 0$).

<!--en:4a46c5ed-->
- Für $\lambda > 4$ wandert $\mathbf{x}(\lambda)$ ins Innere der Kreisscheibe ($g < 0$).

<!--en:37b63ff7-->
- Bei $\lambda^\star = 4$ liegt $\mathbf{x}(4) = (0.6, 0.8)$ auf dem Randkreis ($g = 0$).

<!--en:f521a4e5-->
::: {.slide title="Die Lagrange-Dualfunktion ist immer konkav"}

<!--en:1ca48427-3-->
[Das duale Problem]{.kicker}

<!--en:0187b497-->
Die **Lagrange-Dualfunktion** ist das Infimum von $\mathcal{L}$ über $\mathbf{x} \in \mathcal{D}$ ohne weitere Nebenbedingungen:

<!--en:a91621ee-->
- **Konkavität in jedem Fall:** Für jedes feste $\mathbf{x}$ ist $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ affin in $(\boldsymbol{\lambda}, \boldsymbol{\nu})$. Das punktweise Infimum affiner Funktionen ist immer konkav. Deshalb **ist $q$ selbst bei einem nichtkonvexen primalen Problem konkav**.

<!--en:488fad6a-->
::: {.slide title="Schwache Dualität und Dualitätslücke"}

<!--en:1ca48427-4-->
[Das duale Problem]{.kicker}

<!--en:e15cea5a-->
Die Maximierung der konkaven Dualfunktion über die dual zulässigen Multiplikatoren ($\boldsymbol{\lambda} \succeq \mathbf{0}$) definiert das konvexe **duale Problem**:

<!--en:875372b1-->
- **Kurzbeweis der schwachen Dualität ($d^\star \le p^\star$):** Für jedes primal zulässige $\mathbf{x}$ ($g_i(\mathbf{x}) \le 0, h_j(\mathbf{x}) = 0$) und jedes dual zulässige $(\boldsymbol{\lambda} \succeq \mathbf{0}, \boldsymbol{\nu})$ gilt für die Strafterme $\lambda_i g_i(\mathbf{x}) \le 0$ und $\nu_j h_j(\mathbf{x}) = 0$:
  $$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{z} \in \mathcal{D}}\mathcal{L}(\mathbf{z}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x}) \le f(\mathbf{x})$$

<!--en:78108cda-->
- **Dualitätslücke und Abbruchkriterium:** Die nichtnegative Differenz $p^\star - d^\star \ge 0$ heißt **Dualitätslücke**. Jedes dual zulässige Paar $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ begrenzt für jeden primal zulässigen Kandidaten $\mathbf{x}$ sofort die Abweichung vom Optimum gemäß $f(\mathbf{x}) - p^\star \le f(\mathbf{x}) - q(\boldsymbol{\lambda}, \boldsymbol{\nu})$.

<!--en:f79c54ae-->
::: {.slide title="Slater-Bedingung und starke Dualität"}

<!--en:ba03859a-->
[Starke Dualität]{.kicker}

<!--en:e4ac947e-->
- **Slater-Bedingung ($\Rightarrow d^\star = p^\star$):** Ist das primale Problem konvex ($f, g_i$ konvex, $h_j$ affin) und gibt es einen Punkt $\bar{\mathbf{x}} \in \mathrm{relint}(\mathcal{D})$ mit $g_i(\bar{\mathbf{x}}) < 0$ (für nichtaffine $g_i$), $g_i(\bar{\mathbf{x}}) \le 0$ (für affine $g_i$) und $h_j(\bar{\mathbf{x}}) = 0$, so **gilt starke Dualität ($d^\star = p^\star$)**, und es gibt dual optimale Multiplikatoren $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$.

<!--en:7a1b5090-->
- **Vereinfachung im affinen Fall:** Sind alle Ungleichungen linear oder affin (lineare Programme, SVMs), genügt die Existenz eines zulässigen Punktes.

<!--en:142df958-->
::: {.slide title="Stützgeraden und starke Dualität"}

<!--en:ba03859a-2-->
[Starke Dualität]{.kicker}

<!--en:861c196b-->
Für eine einzige Ungleichung $g(\mathbf{x}) \le 0$ ergibt sich aus den erreichbaren Werten der Nebenbedingungs- und der Zielfunktion folgender Epigraph:

<!--en:6a1a37d9-->
- **Geometrische Deutung von $p^\star$ und $q(\lambda)$:** $p^\star$ ist die kleinste Höhe $t$ von $\mathcal{A}$ auf der vertikalen Achse $u = 0$. Für $q(\lambda) = \inf_{(u,t)\in\mathcal{A}}(t + \lambda u)$ eine Gerade mit Steigung $-\lambda$ nach oben verschieben, bis sie $\mathcal{A}$ von unten stützt. Ihr Achsenabschnitt bei $u = 0$ ist $q(\lambda)$.

<!--en:40b4247a-->
- **Warum Konvexität und Slater-Bedingung genügen:** Dank der Konvexität ist $\mathcal{A}$ konvex und hat bei $(0, p^\star)$ eine Stützgerade. Der Slater-Punkt $u = g(\bar{\mathbf{x}}) < 0$ schließt eine vertikale Stützgerade aus.

<!--en:0bc188e7-->
::: {.slide title="Sattelpunkte und Vertauschbarkeit von Minimum und Maximum"}

<!--en:ba03859a-3-->
[Starke Dualität]{.kicker}

<!--en:ef6d22c8-->
Der primale und der duale Optimalwert ergeben sich aus den beiden Zugreihenfolgen eines Minimax-Spiels mit $\mathcal{L}$:

<!--en:11695b42-->
- **Äquivalenz zum Sattelpunkt:** Bei angenommenen Optima gilt $d^\star = p^\star$ genau dann, wenn $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ ein Sattelpunkt ist ($\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) \le \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$).

<!--en:369a91ec-->
- **Notwendigkeit der KKT-Bedingungen:** Minimierung über $\mathbf{x}$ liefert $\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0}$, und aus $d^\star = p^\star$ folgt $\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$.

<!--en:ed1deffa-->
::: {.slide title="Minimax-Spiele im Deep Learning: GANs und Robustheit"}

<!--en:ba03859a-4-->
[Starke Dualität]{.kicker}

<!--en:5ccc136e-->
- **Primale Lösung und kleinere Dimension:** Bei strikt konvexem $f$ liefert $\mathbf{x}^\star = \mathrm{argmin}_{\mathbf{x}}\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ die primale Lösung aus dem dualen Problem mit $n$ Variablen (z. B. bei Kernel-SVMs für $n \ll d$).

<!--en:d5426d44-->
- **Rotation bei GANs und adversarialem Training:** Im bilinearen Spiel $\min_x\max_y xy$ wächst der Radius bei gleichzeitigem Gradientenabstieg und -aufstieg in jedem Schritt um den Faktor $\sqrt{1 + \eta^2} > 1$. Ohne Stabilisierung durch ein Extragradientenverfahren oder Dämpfung entsteht eine nach außen laufende Spirale.

<!--en:55b6a930-->
::: {.slide title="Lagrange-Multiplikatoren als Schattenpreise"}

<!--en:cdaeec6a-->
[Sensitivität und Preise]{.kicker}

<!--en:50ad3a7f-->
Wie stark verbessert sich der Optimalwert bei einer Lockerung der Nebenbedingungen zu $g_i(\mathbf{x}) \le u_i$ und $h_j(\mathbf{x}) = v_j$? Sei $p^\star(\mathbf{u}, \mathbf{v})$ der gestörte Optimalwert:

<!--en:028e0d65-->
- **Globale untere Sensitivitätsschranke:** Bei starker Dualität mit dualem Optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ gilt für jedes $\mathbf{x}$, das für $(\mathbf{u}, \mathbf{v})$ zulässig ist:
  $$p^\star(\mathbf{0}, \mathbf{0}) = q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) \le f(\mathbf{x}) + \sum_{i=1}^m \lambda_i^\star g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j^\star h_j(\mathbf{x}) \le f(\mathbf{x}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v}$$
  Das Infimum über alle für das gestörte Problem zulässigen $\mathbf{x}$ liefert:
  $$p^\star(\mathbf{u}, \mathbf{v}) \ge p^\star(\mathbf{0}, \mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u} - \boldsymbol{\nu}^{\star\top}\mathbf{v}$$

<!--en:2ee71d49-->
- **Schattenpreis als Ableitung:** Ist $p^\star(\mathbf{u}, \mathbf{v})$ an der Stelle $(\mathbf{0}, \mathbf{0})$ differenzierbar, gilt
  $$\lambda_i^\star = -\frac{\partial p^\star}{\partial u_i}(\mathbf{0}, \mathbf{0}), \qquad \nu_j^\star = -\frac{\partial p^\star}{\partial v_j}(\mathbf{0}, \mathbf{0})$$

<!--en:da50e5fc-->
::: {.slide title="Äquivalenz von Gewichtszerfall und Normbeschränkung"}

<!--en:cdaeec6a-2-->
[Sensitivität und Preise]{.kicker}

<!--en:75e10d57-->
Bei konvexem Verlust $\ell(\mathbf{w})$ liefert ein $\ell_2$-Strafterm denselben Lösungspfad wie die Beschränkung der Gewichte auf eine Normkugel:

<!--en:17c7c1c0-->
- **Gleiche KKT-Stationaritätsbedingung:** Für beide Probleme gilt $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$.

<!--en:99d55019-->
- **Wechselseitige Zuordnung:** Der Minimierer $\mathbf{w}_\lambda$ der Zielfunktion mit Strafterm löst das Problem auf der Kugel mit Radius $r = \|\mathbf{w}_\lambda\|_2$, und für den zugehörigen Multiplikator gilt $\lambda^\star = -\partial p^\star/\partial(\tfrac12 r^2)$.

<!--en:18ad2d61-->
::: {.slide title="Numerischer Vergleich von Strafterm und Projektion auf die Kugel"}

<!--en:cdaeec6a-3-->
[Sensitivität und Preise]{.kicker}

<!--en:754fc022-->
Für den konvexen Kleinste-Quadrate-Verlust $\ell(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|_2^2$ ($30 \times 5$) das Ridge-Problem mit Strafterm für $\lambda \in \{0.1, 1.0, 10.0\}$ in geschlossener Form lösen, $r = \|\mathbf{w}_\lambda\|_2$ setzen und projizierten Gradientenabstieg auf der $r$-Kugel anwenden:

<!--en:2c96b715-->
- Der projizierte Gradientenabstieg auf der Normkugel liefert die Lösung $\mathbf{w}_\lambda$ des Problems mit Strafterm bis auf Maschinengenauigkeit ($10^{-16}$). Der aus $-\mathbf{w}^\top\nabla\ell(\mathbf{w})/r^2$ berechnete Multiplikator stimmt in allen sechs Nachkommastellen mit $\lambda$ überein.

<!--en:96041f54-->
::: {.slide title="Die konvex Konjugierte nach Fenchel: Stützgeraden"}

<!--en:d2ab5ffe-->
[Konvex Konjugierte]{.kicker}

<!--en:65a75c17-->
Die **konvex Konjugierte** (auch Fenchel-Konjugierte) $f^*(\mathbf{y})$ beschreibt eine Funktion $f(\mathbf{x})$ durch die Achsenabschnitte ihrer Stützhyperebenen mit Steigung $\mathbf{y}$:

<!--en:28214bb8-->
- **Konvexität und Bikonjugation:** $f^*$ ist ein punktweises Supremum affiner Funktionen von $\mathbf{y}$ und somit immer konvex. Ist $f$ eigentlich, abgeschlossen und konvex, gilt $f^{**} = f$.

<!--en:dd0b2ccc-->
- **Fenchel-Young-Ungleichung ($\forall\,\mathbf{x}, \mathbf{y}$):**
  $$\mathbf{y}^\top\mathbf{x} \le f(\mathbf{x}) + f^*(\mathbf{y})$$
  Für konvexes, differenzierbares $f$ gilt Gleichheit genau dann, wenn $\mathbf{y} = \nabla f(\mathbf{x})$.

<!--en:6e5df341-->
::: {.slide title="Vier wichtige Paare Fenchel-Konjugierter"}

<!--en:d2ab5ffe-2-->
[Konvex Konjugierte]{.kicker}

<!--en:1847b3ff-->
- **Quadratische Funktionen:** $\tfrac12\|\mathbf{x}\|_2^2$ ist zu sich selbst konjugiert ($\tfrac12\|\mathbf{y}\|_2^2$), und $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ ($\mathbf{A} \succ \mathbf{0}$) hat die Konjugierte $\tfrac12\mathbf{y}^\top\mathbf{A}^{-1}\mathbf{y}$.

<!--en:7af9ec43-->
- **Exponentialfunktion $\leftrightarrow$ Entropie:** $f(x) = e^x$ hat die Konjugierte $f^*(y) = y\log y - y$ für $y > 0$ (angenommen bei $x^\star = \log y$).

<!--en:149fc90c-->
- **$\ell_1$-Strafterm $\leftrightarrow$ $\ell_\infty$-Quader:** $f(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$ hat die Konjugierte $I_{\{\|\mathbf{y}\|_\infty \le \lambda\}}(\mathbf{y})$.

<!--en:775bb9c9-->
::: {.slide title="Duale Probleme bei linearen Nebenbedingungen in einer Zeile"}

<!--en:d2ab5ffe-3-->
[Konvex Konjugierte]{.kicker}

<!--en:61bf9ddb-->
Bei jedem Problem $\min_{\mathbf{x}} f(\mathbf{x})$ mit linearer Nebenbedingung $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$ (oder $\mathbf{A}\mathbf{x} = \mathbf{b}$) vereinfacht sich die Dualfunktion $q(\boldsymbol{\lambda})$ unmittelbar zur Darstellung durch die Fenchel-Konjugierte $f^*$:

<!--en:8760bba9-->
- **Rechenbeispiel (nächstgelegener Punkt in einer Halbebene):** $f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|_2^2$ unter $\mathbf{a}^\top\mathbf{x} \le b$ mit $\mathbf{a} = (1, 2)$ ($\|\mathbf{a}\|^2 = 5$) und $b = -1$ minimieren.

<!--en:8f97ead0-->
- **Duales Problem in einer Zeile mit $f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|_2^2$:**
  $$q(\lambda) = -f^*(-\lambda\mathbf{a}) - b\lambda = -\tfrac12\lambda^2\|\mathbf{a}\|_2^2 - b\lambda = \lambda - \tfrac52\lambda^2 \quad (\lambda \ge 0)$$

<!--en:742ac6eb-->
- Aus $q'(\lambda) = 1 - 5\lambda = 0$ folgen $\lambda^\star = \tfrac15$, $d^\star = p^\star = \tfrac{1}{10}$ und das primale Optimum $\mathbf{x}^\star = -\lambda^\star\mathbf{a} = (-0.2, -0.4)$.

<!--en:f59a4c1d-->
::: {.slide title="Log-Sum-Exp und negative Shannon-Entropie"}

<!--en:d2ab5ffe-4-->
[Konvex Konjugierte]{.kicker}

<!--en:c9be4cad-->
Die Fenchel-Konjugierte von $\mathrm{lse}(\mathbf{x}) = \log\sum_{i=1}^n e^{x_i}$ ist die negative Shannon-Entropie auf dem Wahrscheinlichkeitssimplex $\Delta$:

<!--en:fd0dbf9d-->
- **Variationsdarstellung durch Bikonjugation ($\mathrm{lse}^{**} = \mathrm{lse}$):** $\mathrm{lse}(\mathbf{x}) = \max_{\mathbf{p} \in \Delta}(\mathbf{p}^\top\mathbf{x} + H(\mathbf{p}))$, eindeutig angenommen bei $\mathbf{p}^\star = \mathrm{softmax}(\mathbf{x})$.

<!--en:8ef88646-->
::: {.slide title="Herleitung des dualen Problems der Support Vector Machine (SVM)"}

<!--en:158c57bf-->
[Duale Probleme: SVM]{.kicker}

<!--en:f2f8fe4e-->
Für linear trennbare Daten $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ mit $y_i \in \{\pm 1\}$ maximiert die Hard-Margin-SVM die geometrische Margin $2/\|\mathbf{w}\|_2$ durch Lösung des konvexen quadratischen Programms:

<!--en:57672f7c-->
- **Lagrange-Funktion aufstellen und minimieren ($\alpha_i \ge 0$):**
  $$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \tfrac12\|\mathbf{w}\|_2^2 + \sum_{i=1}^n \alpha_i\big(1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\big)$$
  1. **Minimierung über $b$:** $\partial_b\mathcal{L} = -\sum_{i=1}^n \alpha_i y_i = 0$ (sonst $\inf_b \mathcal{L} = -\infty$).
  2. **Minimierung über $\mathbf{w}$:** $\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = \mathbf{0} \implies \mathbf{w}^\star = \sum_{i=1}^n \alpha_i^\star y_i \mathbf{x}_i$.

<!--en:c1d32d3d-->
::: {.slide title="Stützvektoren und Kernel-Trick"}

<!--en:158c57bf-2-->
[Duale Probleme: SVM]{.kicker}

<!--en:3b063b7b-->
Einsetzen von $\mathbf{w}^\star = \sum_i \alpha_i^\star y_i \mathbf{x}_i$ und $\sum_i \alpha_i y_i = 0$ in $\mathcal{L}$ liefert das **duale Problem der SVM**:

<!--en:3d50d0bf-->
- **Wenige Stützvektoren:** Wegen des komplementären Schlupfs gilt $\alpha_i^\star > 0$ nur für aktive Punkte auf dem Rand der Margin $y_i(\mathbf{w}^{\star\top}\mathbf{x}_i + b^\star) = 1$.

<!--en:f780dc02-->
- **Kernel-Trick:** Training und Vorhersage hängen von $\mathbf{x}_i$ ausschließlich über innere Produkte $\mathbf{x}_i^\top\mathbf{x}_j \mapsto k(\mathbf{x}_i, \mathbf{x}_j)$ ab.

<!--en:3207b5fb-->
::: {.slide title="Lösung des dualen SVM-Problems durch projizierten Gradientenaufstieg"}

<!--en:158c57bf-3-->
[Duale Probleme: SVM]{.kicker}

<!--en:0ed65246-->
Der Achsenabschnitt $b$ geht in $\tilde{\mathbf{w}} = (\mathbf{w}, b)$ ein. Mit $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$ ergibt sich das duale Problem $\max_{\boldsymbol{\alpha} \succeq \mathbf{0}}\big(\mathbf{1}^\top\boldsymbol{\alpha} - \tfrac12\boldsymbol{\alpha}^\top\mathbf{Q}\boldsymbol{\alpha}\big)$ auf dem nichtnegativen Orthanten. Es lässt sich durch projizierten Gradientenaufstieg lösen:

<!--en:b00c4e70-->
- Alle $4$ positiven Multiplikatoren gehören zu Punkten mit Margin $1.0000$. Es gilt $p^\star = d^\star = 41/98 \approx 0.418367$ mit Lücke $10^{-16}$.

<!--en:c10b61ee-->
::: {.slide title="Leistungsverteilung auf Kanäle durch Water-Filling"}

<!--en:bc8a5ac0-->
[Duale Probleme: Water-Filling]{.kicker}

<!--en:c11bcb6d-->
Die Gesamtleistung $P > 0$ so auf $n$ Kanäle mit Rauschpegeln $N_i > 0$ verteilen, dass die gesamte Shannon-Kapazität maximal wird:

<!--en:c63d1530-->
- **KKT-Stationarität:** Mit Multiplikator $\nu$ für $\sum_i P_i = P$ und $\lambda_i \ge 0$ für $-P_i \le 0$ ($\lambda_i P_i = 0$) verlangt die Stationarität $\frac{1}{N_i + P_i} = \nu - \lambda_i$.

<!--en:d9cce500-->
- **Gemeinsamer Wasserspiegel $\theta = 1/\nu^\star$:** Aktive Kanäle füllen sich bis $\theta$ ($P_i^\star = \max(\theta - N_i,\, 0)$), wobei $\theta$ die folgende Gleichung löst:
  $$\sum_{i=1}^n \max(\theta - N_i,\, 0) = P$$

<!--en:142662f4-->
::: {.slide title="Water-Filling und Schattenpreis im numerischen Test"}

<!--en:bc8a5ac0-2-->
[Duale Probleme: Water-Filling]{.kicker}

<!--en:325c41bb-->
Bei Rauschpegeln $\mathbf{N} = (0.1, 0.4, 0.8, 1.6, 2.5)$ und Gesamtbudget $P = 3.0$ liefert Bisektion den Wasserspiegel $\theta$. Ein numerischer Vergleich zeigt, ob der Lagrange-Multiplikator $\nu^\star = 1/\theta$ der marginalen Kapazität $\mathrm{d}U^\star / \mathrm{d}P$ entspricht:

<!--en:3292990f-->
- Die drei rauschärmsten Kanäle füllen sich bis $N_i + P_i^\star = \theta = 1.433333$, die beiden rauschstarken Kanäle ($N_i > \theta$) bleiben trocken ($P_4^\star = P_5^\star = 0$), und $\nu^\star = 1/\theta = 0.697674$ stimmt auf sechs Stellen mit $\mathrm{d}U^\star/\mathrm{d}P$ überein.

<!--en:23951907-->
::: {.slide title="Wenn starke Dualität versagt: eine explizite Dualitätslücke"}

<!--en:87f9b99b-->
[Nichtkonvexe Dualität]{.kicker}

<!--en:70c8d4fc-->
Um die Rolle der Konvexität für starke Dualität zu sehen, die *konkave* Zielfunktion $f(x) = -x^2$ auf $\mathcal{D} = [0, 1]$ unter der Nebenbedingung $g(x) = x - \tfrac12 \le 0$ minimieren ($p^\star = f(\tfrac12) = -\tfrac14$):

<!--en:e61145f4-->
- Obwohl $x = 1/4$ strikt zulässig ist ($g(1/4) = -1/4 < 0$), bleibt wegen der fehlenden Konvexität eine strikt positive Dualitätslücke $p^\star - d^\star = 1/4$.

<!--en:72a18dda-->
::: {.slide title="Verständnisfragen: duale Schranken, Schattenpreise und Wasserspiegel"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:aad51b96-->
Fragen zu dualen Zertifikaten, Schattenpreisen und Water-Filling:

<!--en:24bd50c0-->
1. Beim Clipping auf die Kreisscheibe ($p^\star = 8$) hält ein Löser an einem zulässigen Punkt mit $f(\mathbf{x}) = 8.01$ an. Welche Schranke für die Abweichung vom Optimum lässt sich ohne Kenntnis von $\lambda^\star = 4$ mit dem nicht optimierten Multiplikator $\lambda = 3$ ($q(3) = 7.875$) nachweisen?

<!--en:fa0e6d78-->
2. Beim Clipping von $(3, 4)$ auf die Einheitskreisscheibe ($\tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$) gilt $p^\star(0) = 8$ mit optimalem Multiplikator $\lambda^\star = 4$. Welche untere Schranke für $p^\star(\tfrac12)$ garantiert der Schattenpreis bei einer Lockerung um $u = \tfrac12$ (Vergrößerung der Kreisscheibe auf Radius $r = \sqrt{2}$)? Wie verhält sie sich zum tatsächlichen Wert $p^\star(\tfrac12)$?

<!--en:4a34f4e3-->
3. Wie lässt sich beim Water-Filling mit Rauschpegeln $(0.1, 0.4, 0.8, 1.6, 2.5)$ der Wasserspiegel $\theta = 1.433333$ für $P = 3$ von Hand berechnen?

<!--en:e41de942-->
**Lösungen:** **(1)** Die schwache Dualität garantiert $p^\star \ge q(3) = 7.875$, also unmittelbar $f(\mathbf{x}) - p^\star \le 8.01 - 7.875 = \mathbf{0.135}$. **(2)** Die globale Sensitivitätsschranke garantiert $p^\star(\tfrac12) \ge p^\star(0) - \lambda^\star u = 8 - 4(\tfrac12) = \mathbf{6.00}$. Das neue Optimum beträgt genau $\tfrac12(5 - \sqrt{2})^2 \approx \mathbf{6.43} > 6.00$. **(3)** Sind die $3$ rauschärmsten Kanäle aktiv, folgt $\sum_{i=1}^3(\theta - N_i) = 3\theta - (0.1 + 0.4 + 0.8) = 3 \implies 3\theta = 4.3 \implies \theta = \mathbf{4.3/3 \approx 1.4333}$. Das ist wegen $0.8 < 1.4333 < 1.6$ mit der Annahme vereinbar.

<!--en:2c677b90-->
::: {.slide title="Zusammenfassung: Lagrange-Dualität und Konjugierte"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:ac4e390e-->
| Begriff oder Satz | Mathematische Aussage | Voraussetzungen | Praktische Bedeutung |

<!--en:3f9e48a0-->
| **Lagrange-Dualfunktion $q(\boldsymbol{\lambda}, \boldsymbol{\nu})$** | $\inf_{\mathbf{x} \in \mathcal{D}}\big(f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})\big)$ | Keine | Immer konkav in $(\boldsymbol{\lambda}, \boldsymbol{\nu})$, auch bei nichtkonvexem primalem Problem |

<!--en:8503bd9b-->
| **Schwache Dualität** | $d^\star = \sup_{\boldsymbol{\lambda} \succeq \mathbf{0}, \boldsymbol{\nu}} q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le p^\star$ | Keine | Jedes dual zulässige Paar $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ liefert eine untere Schranke für $p^\star$ |

<!--en:32eadba3-->
| **Starke Dualität (Slater)** | $d^\star = p^\star$ und Sattelpunkt von $\mathcal{L}$ | Konvexes primales Problem und strikt zulässiger innerer Punkt $\bar{\mathbf{x}}$ | Dualitätslücke null, KKT-Bedingungen notwendig und hinreichend |

<!--en:f2c8c8ba-->
| **Schattenpreise** | $p^\star(\mathbf{u}, \mathbf{v}) \ge p^\star(\mathbf{0}, \mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u} - \boldsymbol{\nu}^{\star\top}\mathbf{v}$ | Starke Dualität ($d^\star = p^\star$) | $\lambda_i^\star = -\partial p^\star/\partial u_i$ verbindet Gewichtszerfall und Normbudget |

<!--en:25c79312-->
| **Fenchel-Konjugierte $f^*(\mathbf{y})$** | $\sup_{\mathbf{x}}(\mathbf{y}^\top\mathbf{x} - f(\mathbf{x}))$ und $\mathbf{y}^\top\mathbf{x} \le f(\mathbf{x}) + f^*(\mathbf{y})$ | $f^{**} = f$ bei eigentlicher, abgeschlossener, konvexer Funktion | Bei linearen Nebenbedingungen lautet die Dualfunktion $-f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}$. Es gilt $\mathrm{lse}^* = -H$ |
