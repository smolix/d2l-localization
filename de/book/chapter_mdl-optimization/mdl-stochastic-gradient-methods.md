<!--en:124938d3-->
# Stochastische Gradientenverfahren

<!--en:68dc6425-->
Beim Training eines neuronalen Netzes berechnen wir fast nie den exakten Gradienten der Verlustfunktion. Statt für jedes Update den gesamten Datensatz zu durchlaufen, berechnen wir den Gradienten in jedem Schritt auf einem zufälligen Minibatch aus einigen Hundert Beispielen. Bisher haben wir in jedem Abschnitt exakte Gradienten vorausgesetzt. Dann bestimmt die Krümmung die Anzahl der Schritte, und bei Konvexität ist jeder stationäre Punkt ein globaler Minimierer. Ein Minibatchgradient zeigt im Mittel in die richtige Richtung, ist aber verrauscht. Damit ändert sich die zentrale Frage der Optimierung. Gefragt ist nun, wie nahe ein Lauf dem Minimierer kommen kann. Die Zahl der nötigen Schritte allein reicht als Maß nicht mehr aus. Wir untersuchen, was uns das Gradientenrauschen kostet und wie wir es überwinden können.

<!--en:106f3fd0-->
Schon die einfachste Verlustfunktion zeigt, was das Rauschen kostet. Betrachten wir $f(x) = \tfrac12 x^2$ mit dem Minimierer $0$. Wir addieren zu jedem Gradienten unabhängiges Rauschen mit Varianz eins und wählen die konstante Schrittweite $\eta = 0.1$. Mit exakten Gradienten würde jeder Schritt $x$ um ein Zehntel verkleinern. Die Iterierte käme $0$ beliebig nahe. Bei verrauschten Gradienten kommt die Iterierte nie zur Ruhe. Ihr mittleres Quadrat konvergiert gegen $0.1/1.9 \approx 0.053$. Ihr typischer Abstand vom Minimierer beträgt also ungefähr $0.23$, egal wie viele Schritte wir machen. Wie weit der Verlust sinken kann, hängt von drei Größen ab: der Stärke des Rauschens, der Schrittweite und der Anzahl der Schritte, die wir uns leisten können.

<!--en:aad476a7-->
Zunächst bestimmen wir die Stärke des Rauschens eines Minibatchgradienten. Für eine quadratische Zielfunktion berechnen wir dann, wo der Abstieg bei fester Schrittweite zum Stillstand kommt und wie viele Schritte er bis dahin braucht. Anschließend zeigen wir, wie abnehmende Schrittweiten diese Grenze überwinden und welche Konstante ein $1/k$-Schrittweitenplan erfordert. Für ein festes Rechenbudget leiten wir Fehlerschranken für konvexe und nichtkonvexe Verlustfunktionen her, wenn wir die konstante Schrittweite auf das Budget abstimmen. Schließlich zeigen wir, wie ein gespeicherter Referenzgradient das Rauschen auf einem endlichen Trainingsdatensatz aufheben kann. Drei Ergebnisse begleiten uns dabei. Die rauschbedingte Untergrenze ist proportional zur Schrittweite. Ein $1/k$-Schrittweitenplan erreicht seine Rate nur mit einer hinreichend großen Konstanten. Bei einem festen Budget von $K$ Schritten sollte die Schrittweite proportional zu $1/\sqrt{K}$ sein. Den letzten Unterabschnitt zur Varianzreduktion können Sie beim ersten Lesen überspringen.

<!--en:b8c3634a-->
## Minibatchgradienten

<!--en:3bf0eee5-->
### Rechenaufwand für einen vollständigen Gradienten

<!--en:3712bc78-->
Eine typische Zielfunktion für das Training mittelt den Verlust über $n$ Beispiele,

<!--en:ee182e18-->
wobei $f_i$ der Verlust für Beispiel $i$ ist und $\mathbf{x} \in \mathbb{R}^d$ die Modellparameter enthält. Auch der Gradient $\nabla f = \tfrac1n \sum_i \nabla f_i$ ist ein Mittelwert. Um ihn zu berechnen, müssen wir alle $n$ Beispiele auswerten. Für $n = 10^9$ erfordert ein einziger Schritt des Gradientenabstiegs einen vollständigen Durchlauf durch den Datensatz. Typischerweise braucht der Gradientenabstieg Tausende solcher Schritte. Schon im durchgehenden Beispiel dieses Kapitels, $\tfrac12(x_1^2 + 1000\,x_2^2)$, unterscheiden sich die Krümmungen um den Faktor $1000$. Selbst mit optimaler fester Schrittweite sind etwa $7{,}000$ Iterationen nötig, um den Fehler auf ein Millionstel zu senken (:numref:`sec_mdl-gradient-based-optimization`). Der **stochastische Gradientenabstieg** (SGD) vermeidet diesen Aufwand. Er ersetzt den vollständigen Mittelwert durch den Mittelwert über einen zufälligen **Minibatch** $\mathcal{B}$ aus $b$ Indizes,

<!--en:74d18f65-->
Jeder Schritt berücksichtigt nun nur $b$ Beispiele statt $n$. Ein Minibatch mit $b = 256$ von $10^9$ Beispielen benötigt ungefähr ein Viermillionstel des Aufwands eines vollständigen Durchlaufs. In diesem Abschnitt zählt $k$ die Schritte des Optimierers. Die Schrittweite $\eta_k$ darf von $k$ abhängen. Für unsere Analyse ziehen wir in jedem Schritt einen neuen Minibatch. Seine $b$ Indizes sind unabhängig und auf $\{1, \ldots, n\}$ gleichverteilt (Ziehen mit Zurücklegen). Beim Training mischt man den Datensatz meist und zieht ohne Zurücklegen. Dadurch erhält die unten hergeleitete Gradientenvarianz den Faktor $(n-b)/(n-1)$. Dieser liegt nahe bei eins, wenn $b \ll n$ gilt (:numref:`sec_sgd`).

<!--en:9a7d8915-->
### Erwartungstreue und das $1/b$-Gesetz

<!--en:479aa5c1-->
Wie genau ist ein Minibatchgradient? Zwei Größen kennzeichnen seine Güte. Sein Erwartungswert zeigt, ob er im Mittel in die richtige Richtung weist. Seinen mittleren quadratischen Abstand vom wahren Gradienten nennen wir *Rauschleistung*. Sie hängt davon ab, wie stark die Gradienten der einzelnen Beispiele um ihren Mittelwert streuen. Diese Streuung beschreibt ihre Kovarianzmatrix.

<!--en:7cd31abb-->
**Proposition (Minibatchgradienten sind erwartungstreu mit Rauschleistung $\operatorname{tr}\boldsymbol{\Sigma}/b$).** *Wir halten $\mathbf{x}$ fest. Sei*

<!--en:d2f03285-->
*die Kovarianz des Gradienten eines einzelnen, gleichverteilt gezogenen Beispiels. Sind die $b$ Indizes von $\mathcal{B}$ unabhängig und auf $\{1, \ldots, n\}$ gleichverteilt, so gilt*

<!--en:646b3c13-->
**Beweis.** Für einen einzelnen gleichverteilten Index $i$ gilt $\mathbb{E}[\nabla f_i(\mathbf{x})] = \tfrac1n \sum_i \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x})$. Jeder Summand im Minibatch hat also den Erwartungswert $\nabla f(\mathbf{x})$, und damit auch der Mittelwert der Summanden. Die $b$ Indizes sind unabhängig, und jeder Summand hat die Kovarianz $\boldsymbol{\Sigma}$. Die Kovarianz der Summe beträgt daher $b\boldsymbol{\Sigma}$. Die Division durch $b$ skaliert die Kovarianz mit $1/b^2$ und ergibt $\boldsymbol{\Sigma}/b$. Schließlich ist für jeden Zufallsvektor mit Erwartungswert null die erwartete quadrierte Norm gleich der Spur seiner Kovarianzmatrix. $\blacksquare$

<!--en:b8659d9e-->
Während der Optimierung ist $\mathbf{x}$ die aktuelle Iterierte $\mathbf{x}_k$. Da wir den Minibatch unabhängig von den bisherigen Schritten ziehen, gelten beide Gleichungen auch bedingt auf $\mathbf{x}_k$. Die Erwartungstreue setzt gleichverteiltes Ziehen voraus: Ziehen wir Beispiel $i$ mit Wahrscheinlichkeit $p_i$, müssen wir seinen Gradienten mit $1/(np_i)$ gewichten, damit der Schätzer erwartungstreu bleibt.

<!--en:c887a8a8-->
Da die Rauschleistung wie $1/b$ abnimmt, sinkt die effektive Rauschamplitude nur wie $1/\sqrt{b}$. Um das Gradientenrauschen auf ein Zehntel zu verringern, brauchen wir pro Schritt hundertmal so viele Beispiele. Das wahre Gradientensignal hat unabhängig von der Batchgröße die quadrierte Norm $\|\nabla f\|^2$. Bei der Batchgröße $b_{\textrm{noise}} = \operatorname{tr}\boldsymbol{\Sigma}/\|\nabla f\|^2$ haben Signal und Rauschen dieselbe Leistung. Im Haupttext heißt diese Größe *Skala des Gradientenrauschens* (:numref:`sec_batch_size`). In Aufgabe :exref:`ex_mdl-gradient-based-optimization-use-eqref-eq-mdl` leiten Sie das mit der Krümmung gewichtete Gegenstück her.

<!--en:2f5eeac2-->
Um die Stärke des Rauschens an konkreten Daten zu bestimmen, betrachten wir ein kleines Problem der logistischen Regression. Es dient uns später auch für das Experiment zur Schrittweite. Es umfasst $n = 200$ Beispiele $\mathbf{a}_i \in \mathbb{R}^5$ mit standardnormalverteilten Merkmalen. Die Labels $y_i \in \{-1, +1\}$ stammen aus einem logistischen Modell mit den wahren Gewichten $(1, -2, 1.5, 0, 2)$. Die regularisierte Verlustfunktion lautet

<!--en:dc84b34d-->
Der quadratische Strafterm macht $f$ stark konvex mit $\mu \ge 0.1$. Der Gradient für Beispiel $i$ lautet $\nabla f_i(\mathbf{x}) = -y_i s_i \mathbf{a}_i + 0.1\,\mathbf{x}$. Dabei ist $s_i = 1/(1 + e^{y_i \mathbf{a}_i^\top \mathbf{x}})$ die Wahrscheinlichkeit, die das Modell dem falschen Label zuweist. Die folgende Codezelle erzeugt den Datensatz und definiert die Verlustfunktion sowie eine vektorisierte Hilfsfunktion. Diese gibt alle $n$ Beispielgradienten als Zeilen einer Matrix der Form $n \times d$ zurück.

<!--en:d50682b6-->
Da die Proposition exakt gilt, dient die Simulation vor allem der Plausibilitätsprüfung. Außerdem zeigt sie, wie groß $\operatorname{tr}\boldsymbol{\Sigma}$ für dieses Problem tatsächlich ist. Wir werten für jede Batchgröße $10{,}000$ zufällige Minibatches bei $\mathbf{x} = \mathbf{0}$ aus und vergleichen den empirischen Fehler mit $\operatorname{tr}\boldsymbol{\Sigma}/b$.

<!--en:a4f13276-->
Die Rauschleistung $\operatorname{tr}\boldsymbol{\Sigma}$ eines einzelnen Beispiels beträgt etwa $1.1$. Bei allen vier Batchgrößen stimmt der gemessene Fehler bis auf Stichprobenschwankungen von wenigen Prozent mit $\operatorname{tr}\boldsymbol{\Sigma}/b$ überein. Nun kennen wir das Rauschen eines einzelnen Schritts. Wie wirkt es sich über einen ganzen Optimierungslauf aus?

<!--en:349f11f8-->
### Die rauschbedingte Untergrenze

<!--en:f875d0eb-->
Mittelt sich das Gradientenrauschen über viele Schritte heraus, oder summiert es sich? In :numref:`sec_sgd` haben wir die Punktwolke vermessen, in der sich SGD in einem Tal einpendelt, und ihren Radius hergeleitet. Hier bestimmen wir die gesamte Bahn in geschlossener Form. Daraus ergibt sich auch, wie viele Schritte wir bis zu dieser Wolke brauchen. Der einfachste Fall zeigt bereits den ganzen Mechanismus: eine einzelne Koordinate einer quadratischen Funktion, $f(x) = \tfrac12 \lambda x^2$ mit Krümmung $\lambda > 0$ und Minimierer $0$. Wir modellieren den stochastischen Gradienten als Summe aus dem wahren Gradienten $\lambda x_k$ und einem Rauschterm $\xi_k$. Dieser hat bedingt auf $x_k$ den Erwartungswert null und die Varianz $\sigma^2$. Dabei entspricht $\sigma^2$ einem Diagonaleintrag von $\boldsymbol{\Sigma}/b$. Jede zu $\sigma^2$ proportionale Größe skaliert also wie $1/b$. Ein SGD-Schritt lautet

<!--en:2d999506-->
In jedem Schritt zieht der deterministische Anteil die Iterierte mit dem Faktor $\rho$ zum Minimierer, genau wie beim exakten Gradientenabstieg. Der stochastische Anteil versetzt ihr einen zufälligen Stoß $-\eta\xi_k$. Um die Größe der Iterierten zu verfolgen, betrachten wir ihr zweites Moment $e_k = \mathbb{E}[x_k^2]$. Wir quadrieren die Aktualisierungsgleichung und bilden den Erwartungswert. Der gemischte Term $-2\rho\eta\,\mathbb{E}[x_k \xi_k]$ verschwindet, da $\xi_k$ bedingt auf $x_k$ den Erwartungswert null hat. Damit bleibt

<!--en:4ee61622-->
Jeder Schritt verkleinert das zweite Moment um den Faktor $\rho^2$ und fügt die Rauschleistung $\eta^2\sigma^2$ hinzu. Fast alle Ergebnisse dieses Abschnitts lassen sich aus dieser einen Rekursion ablesen. Eine lineare Rekursion mit konstantem inhomogenem Term lösen wir über ihren Fixpunkt $V$, an dem $V = \rho^2 V + \eta^2\sigma^2$ gilt. Ziehen wir die Fixpunktgleichung von der Rekursion ab, sehen wir, dass der Rest $e_k - V$ in jedem Schritt um den Faktor $\rho^2$ schrumpft. Damit erhalten wir

<!--en:aea47cfe-->
mit der **rauschbedingten Untergrenze**

<!--en:d78d590b-->
Die Untergrenze wächst linear mit der Schrittweite und der Rauschvarianz und ist umgekehrt proportional zur Krümmung. Beide Formeln setzen $0 < \eta < 2/\lambda$ voraus. Dieselbe Stabilitätsgrenze gilt für den exakten Gradientenabstieg auf einer quadratischen Funktion. Oberhalb der Grenze divergiert das zweite Moment sogar ohne Rauschen.

<!--en:96613c93-->
Die Bahn durchläuft zwei Phasen. Der transiente Anteil $\rho^{2k}(e_0 - V)$ klingt wie beim deterministischen Gradientenabstieg geometrisch ab. Die Untergrenze $V$ entsteht allein durch das Rauschen. Weit vom Minimierer dominiert der transiente Anteil, und SGD verhält sich ähnlich wie der Gradientenabstieg. In der Nähe des Minimierers wird die Rückstellwirkung $\eta\lambda x$ schwächer. Die zufälligen Stöße behalten dagegen die Varianz $\eta^2\sigma^2$, bis sich bei $e = V$ beide Effekte ausgleichen. Die Iterierte konvergiert dann auch auf Dauer nicht. Ihre Verteilung wird stationär und bildet eine **Kugel um das Minimum** (engl. noise ball). Ihr quadratischer Mittelradius beträgt ungefähr $\sqrt{\eta\sigma^2/(2\lambda)}$ und skaliert wie $\sqrt{\eta}$. :numref:`fig_mdl-opt-sgd-noise-ball` zeigt beide Phasen in einer zweidimensionalen quadratischen Mulde.

<!--en:b7a105de-->
![Gradientenabstieg und SGD mit fester Schrittweite auf $\tfrac12\|\mathbf{x}\|^2$ bei $\eta=0.15$ und Gradientenrauschen mit Standardabweichung $0.6$. Mit dem vollständigen Batch konvergiert die Bahn zum Minimierer. SGD steigt zunächst wie der Gradientenabstieg ab und schwankt dann innerhalb einer Kugel um das Minimum. Gestrichelte Kreise kennzeichnen diese Kugel bei der gegebenen und bei halber Schrittweite. Damit sie die Punktwolke umschließen, entsprechen ihre Radien dem Dreifachen des quadratischen Mittelabstands $\sqrt{\eta\sigma^2/(2\lambda)}$ pro Koordinate. Die Radien unterscheiden sich um den Faktor $\sqrt{2}$.](../img/mdl-opt-sgd-noise-ball.svg)

<!--en:8766c256-->
In der Praxis interessiert uns der erwartete Verlust mehr als der quadrierte Abstand. Wegen $\mathbb{E} f(x_k) = \tfrac12 \lambda e_k$ beträgt die Untergrenze des Verlusts

<!--en:cbc1ce30-->
Für kleine Schritte kürzt sich die Krümmung $\lambda$ überraschenderweise vollständig heraus. In einer flacheren Richtung kann sich die Iterierte weiter entfernen, und zwar um genau den Faktor, um den die geringere Krümmung den Beitrag zum Verlust senkt.

<!--en:016b8601-->
Bei einer $d$-dimensionalen quadratischen Funktion entkoppeln die Koordinaten entlang der Eigenvektoren der Hesse-Matrix. Es ergeben sich $d$ unabhängige Kopien der skalaren Rekursion, jede mit eigenem Eigenwert $\lambda_i$ und eigener Rauschvarianz $\sigma_i^2$. Die Untergrenze des erwarteten Gesamtverlusts ist daher die Summe $\sum_i \eta\sigma_i^2/(2(2 - \eta\lambda_i))$. Die $\sigma_i^2$ sind die Diagonaleinträge von $\boldsymbol{\Sigma}/b$ in der Eigenbasis und summieren sich zu $\operatorname{tr}\boldsymbol{\Sigma}/b$. Gilt für jede Richtung $\eta\lambda_i \ll 1$, vereinfacht sich die Untergrenze zu ungefähr $\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$. Sie hängt von Schrittweite und Batchgröße nur über deren Verhältnis $\eta/b$ ab. Eine Halbierung der Schrittweite senkt die Untergrenze daher um denselben Faktor wie eine Verdopplung der Batchgröße. Auf diesem Zusammenhang beruhen die Regeln für die Lernrate bei großen Batches in :numref:`sec_batch_size`.

<!--en:86ad2857-->
Wie schnell erreicht ein Lauf seine Untergrenze? Wählen wir wie im Eingangsbeispiel $\lambda = \sigma = 1$. Dort ergab $\eta = 0.1$ den Wert $V = 0.053$. Erhöhen wir die Schrittweite auf $\eta = 0.3$, steigt die Untergrenze auf $V = 0.176$. Sie ist damit etwas mehr als dreimal so hoch, und die Untergrenze des Verlusts beträgt $0.088$. Starten wir bei $x_0 = 3$ ($e_0 = 9$), so sinkt der transiente Anteil $e_k - V$ unter ein Zehntel von $V$. Dafür brauchen wir $36$ Schritte bei $\eta = 0.1$ ($\rho^2 = 0.81$), aber nur $9$ Schritte bei $\eta = 0.3$ ($\rho^2 = 0.49$). Eine Verringerung der Schrittweite auf ein Drittel senkt die Untergrenze auf ungefähr ein Drittel. Um diese zu erreichen, brauchen wir allerdings viermal so viele Schritte. Die übrigen Verfahren dieses Abschnitts gehen auf unterschiedliche Weise mit diesem Zielkonflikt zwischen Geschwindigkeit und Genauigkeit um.

<!--en:639cb270-->
Kehren wir zu unserem durchgehenden Beispiel $\tfrac12(x_1^2 + 1000\,x_2^2)$ zurück. Wir modellieren das Rauschen direkt und addieren in jedem Schritt zu jeder Koordinate des Gradienten unabhängiges Rauschen mit Mittelwert null und Varianz $\sigma^2 = 1$. Der Gradient lautet $(x_1, 1000\,x_2)$. Die beiden Koordinaten folgen der skalaren Rekursion mit $\lambda_1 = 1$ und $\lambda_2 = 1000$. Wir setzen $\eta = 1/L = 10^{-3}$, also die Hälfte der Stabilitätsgrenze $2/1000$. Für die steife Koordinate gilt $\rho_2 = 0$. Ein einziger Schritt beseitigt ihren transienten Anteil vollständig. Sie liegt damit an ihrer Untergrenze $V_2 = 10^{-3}/1000 = 10^{-6}$. Für die langsame Koordinate gilt $\rho_1 = 0.999$ mit der Untergrenze $V_1 = 10^{-3}/1.999 = 5.0 \times 10^{-4}$. Ihr transienter Anteil braucht $\log 1000/(-2\log 0.999) \approx 3{,}450$ Schritte, um auf ein Tausendstel zu sinken. Die Untergrenze des Gesamtverlusts beträgt $7.5 \times 10^{-4}$. Davon entfallen $2.5 \times 10^{-4}$ auf die langsame und $5 \times 10^{-4}$ auf die steife Koordinate. Ausgerechnet die steife Koordinate, die der Gradientenabstieg in einem Schritt eliminiert, trägt zwei Drittel zur Untergrenze des Verlusts bei. Denn bei $\eta\lambda_2 = 1$ beträgt ihre Untergrenze $\eta\sigma^2/2$ und damit das Doppelte des Werts $\eta\sigma^2/4$ für kleine Schritte. Die schlechte Konditionierung bremst den Abstieg in der flachen Richtung, während das Rauschen den Verlust in beiden Richtungen erhöht.

<!--en:253c1ab5-->
Unsere geschlossene Formel für die Untergrenze gilt exakt nur für quadratische Funktionen mit additivem Rauschen. Als Nächstes prüfen wir, ob die Untergrenze auch beim nichtquadratischen logistischen Verlust proportional zu $\eta$ ist. Außerdem untersuchen wir, wie abnehmende Schrittweiten die Untergrenze überwinden.

<!--en:86123d71-->
## Abnehmende Schrittweiten

<!--en:27d5de3d-->
### Die Bedingungen von Robbins und Monro

<!--en:f15c05a1-->
Da die rauschbedingte Untergrenze linear mit $\eta$ skaliert, können wir den Minimierer nur erreichen, wenn die Schrittweite gegen null geht. Sinkt sie jedoch zu schnell, bleibt die Iterierte vor dem Ziel stehen. Bei Gradienten der Norm eins kann die insgesamt zurückgelegte Strecke höchstens $\sum_k \eta_k$ betragen. Die klassischen Bedingungen von :citet:`Robbins.Monro.1951` erfassen beide Anforderungen:

<!--en:4e2d00f5-->
Die erste Bedingung stellt sicher, dass sich die Iterierte beliebig weit vom Startpunkt entfernen kann. Die zweite hält die durch das Rauschen eingebrachte Gesamtvarianz $\sigma^2 \sum_k \eta_k^2$ endlich. Der Schrittweitenplan $\eta_k = c/k$ erfüllt beide Bedingungen, während $\eta_k = c/\sqrt{k}$ und eine konstante Schrittweite nur die erste erfüllen. Nehmen wir zusätzlich erwartungstreue Gradienten an, deren Varianz höchstens quadratisch mit dem Abstand zum Minimierer wächst. Dann garantieren die beiden Bedingungen, dass SGD bei jeder stark konvexen Verlustfunktion zum Minimierer konvergiert :cite:`Robbins.Monro.1951,Bottou.Curtis.Nocedal.2018`.

<!--en:204d3a0c-->
Bei unserer skalaren quadratischen Funktion genügen schwächere Voraussetzungen für die Konvergenz im quadratischen Mittel. Durch wiederholtes Einsetzen in die Rekursion erhalten wir $e_k \to 0$, sofern $\eta_k \to 0$ und $\sum_k \eta_k = \infty$ gelten (mit $\eta_k < 2/\lambda$). Insbesondere ergibt $\eta_k = c/\sqrt{k}$ die Beziehung $e_k \approx c\sigma^2/(2\lambda\sqrt{k})$. Das zweite Moment folgt also der Untergrenze für die jeweils aktuelle Schrittweite. Der allgemeine Satz braucht die stärkere Forderung der Quadratsummierbarkeit, um die Konvergenz entlang fast jedes einzelnen Zufallspfads zu garantieren.

<!--en:6d72e248-->
Für eine nichtquadratische Zielfunktion bleiben zwei Fragen offen. Skaliert die rauschbedingte Untergrenze weiterhin linear mit $\eta$? Und kann ein abnehmender Schrittweitenplan diese Grenze unterschreiten? Für unser Problem der logistischen Regression berechnen wir zunächst das Referenzoptimum $f^\star$ mit $4{,}000$ Schritten des vollständigen Gradientenabstiegs. Dann definieren wir eine Schleife für Minibatch-SGD, die in jedem Schritt den Abstand $f(\mathbf{x}_k) - f^\star$ zum optimalen Verlust aufzeichnet.

<!--en:4bd8e5d1-->
Wenn das quadratische Modell eine gute Näherung liefert, sollte eine Halbierung der festen Schrittweite auch den verbleibenden Abstand zum optimalen Verlust halbieren. Wir führen $4{,}000$ Schritte mit $b = 8$ aus und vergleichen die konstanten Schrittweiten $\eta = 0.8$ und $\eta = 0.4$ mit dem abnehmenden Schrittweitenplan $\eta_k = 80/(k + 100) = 0.8/(1 + k/100)$, der bei $0.8$ beginnt. Die Codezelle gibt für jeden Lauf den mittleren Abstand über die letzten $1{,}000$ Schritte sowie die Verhältnisse dieser Mittelwerte aus. Sie zeigt außerdem die gleitenden Mittelwerte über je $100$ Schritte in doppeltlogarithmischer Darstellung.

<!--en:b78def87-->
Bei beiden Läufen mit fester Schrittweite kommt der Abstieg zum Stillstand. Bei $\eta = 0.8$ liegt der mittlere Abstand über die letzten $1{,}000$ Schritte zwischen etwa $1$ und $2 \times 10^{-2}$. Eine Halbierung der Schrittweite halbiert den Abstand ungefähr. Über mehrere Läufe schwankt das Verhältnis zwischen etwa $1.7$ und $2.8$, mit dem vom quadratischen Modell vorhergesagten Faktor $2$ als Mittelpunkt, obwohl der logistische Verlust nicht quadratisch ist. Mit dem abnehmenden Schrittweitenplan liegt der Abstand am Ende etwa um den Faktor $20$ bis $90$ unter der Grenze für $\eta = 0.8$. Trotz der Minibatchschwankungen fällt die Kurve in der doppeltlogarithmischen Darstellung weiter. Die großen Anfangsschritte legen die Strecke zum Minimierer schnell zurück, die kleinen späteren Schritte verkleinern die Kugel um das Minimum.

<!--en:ae41e77f-->
### Die Konstante eines $1/k$-Schrittweitenplans

<!--en:2063355a-->
Der Schrittweitenplan $\eta_k = c/(k + k_0)$ erfüllt für jedes $c > 0$ beide Bedingungen von Robbins und Monro. Die Verschiebung $k_0$ hält die ersten Schritte sicher unterhalb der Stabilitätsgrenze. Erreichen wir mit jedem Wert von $c$ die Rate $O(1/k)$? Die skalare quadratische Funktion mit Krümmung $\lambda = \mu$ liefert eine exakte Antwort. Mit $t = k + k_0$ lautet die Rekursion für das zweite Moment

<!--en:13054f88-->
Hier wird die Kontraktion wie $1/t$ schwächer, während die zugeführte Rauschleistung wie $1/t^2$ abnimmt. Welcher Effekt dominiert, hängt entscheidend von $c$ ab. Ist $c$ zu klein, sinkt die Schrittweite schneller, als sich die Iterierte dem Minimierer nähern kann. Dann begrenzt der Schrittweitenplan selbst die Konvergenz stärker als das Rauschen.

<!--en:3f79e238-->
**Proposition (Konstante eines $1/k$-Schrittweitenplans).** *Im skalaren Modell mit Krümmung $\mu$ und Schrittweiten $\eta_k = c/(k + k_0)$ folgt aus $2c\mu > 1$ die Beziehung $k\, e_k \to A$ mit*

<!--en:657a49ca-->
*Falls $2c\mu < 1$ gilt, fällt $e_k$ wie $k^{-2c\mu}$ und damit strikt langsamer als $1/k$. Im Grenzfall $2c\mu = 1$ fällt es wie $\log k / k$. Für jedes $c > 0$ gilt $e_k \to 0$.*

<!--en:3bf90d9d-->
Wir verzichten hier auf den vollständigen Beweis. Die Konstante $A$ leiten wir durch Vergleich der führenden Terme her. Die beiden anderen Fälle skizzieren wir. Aufgabe :exref:`ex_mdl-gradient-based-optimization-one-over-k-constant` führt zum vollständigen Beweis: oberhalb des Schwellenwerts durch Induktion, unterhalb durch wiederholtes Einsetzen in die Rekursion.

<!--en:9ad4b89a-->
**Herleitung der Konstanten.** Wir wählen den asymptotischen Ansatz $e_k \approx A/t$. Die linke Seite der Rekursion lautet $A/(t+1) \approx A/t - A/t^2$. Auf der rechten Seite entwickeln wir $(1 - c\mu/t)^2 \approx 1 - 2c\mu/t$ und erhalten $A/t - 2c\mu A/t^2 + c^2\sigma^2/t^2$ bis auf Terme der Ordnung $t^{-3}$. Die Terme mit $1/t$ stimmen bereits überein. Der Vergleich der Koeffizienten von $1/t^2$ ergibt $-A = -2c\mu A + c^2\sigma^2$ oder gleichbedeutend $A(2c\mu - 1) = c^2\sigma^2$. Entscheidend ist ein leicht zu übersehender Term auf der linken Seite. Der Übergang von $t$ zu $t+1$ liefert den Beitrag $-A/t^2$. Er hat dieselbe Ordnung wie der Kontraktionsterm und führt zum Schwellenwert $2c\mu > 1$. Ohne diese Verschiebung würden wir fälschlich $A = c\sigma^2/(2\mu)$ für jedes $c > 0$ vorhersagen.

<!--en:b3e3f3b7-->
**Unterhalb des Schwellenwerts (Beweisskizze).** Für $2c\mu < 1$ bestimmt die kumulierte Kontraktion, wie schnell der Anfangsfehler und das Rauschen aus den ersten Schritten abklingen. Wegen $\log(1 - u) \approx -u$ und $\sum_{j \le t} 1/j \approx \log t$ verhält sich das Produkt der Kontraktionsfaktoren wie $\prod_{j}(1 - c\mu/(j + k_0))^2 \approx (k_0/t)^{2c\mu}$. Es fällt also langsamer als $1/t$. Im Grenzfall $2c\mu = 1$ wirkt das in Schritt $j$ hinzugekommene Rauschen in Schritt $t$ mit einem Gewicht von ungefähr $c^2\sigma^2/(jt)$ nach. Die Summe über $j$ liefert den zusätzlichen Faktor $\log k$.

<!--en:945084b5-->
Wie empfindlich das Ergebnis ist, zeigen konkrete Zahlen für $\mu = \sigma = 1$. Setzen wir $c = 1$, so erhalten wir $A = 1$. Iterieren wir die Rekursion ab $e_0 = 1$ mit $k_0 = 10$, ergibt sich $k\, e_k = 1.0001$ bei $k = 10^6$. Eine Verdopplung auf $c = 2$ erhöht die Konstante nur wenig, auf $A = 4/3$. Ein Unterschreiten der Schwelle wirkt sich viel stärker aus. Für $c = 0.25$ beträgt die Steigung im doppeltlogarithmischen Diagramm $-0.50$ zwischen $k = 10^5$ und $10^6$. Für $c = 0.1$ beträgt sie nur $-0.20$. Minimieren wir $A$ über $c$, erhalten wir die optimale Konstante $c = 1/\mu$ mit $A = \sigma^2/\mu^2$ und $e_k \approx \sigma^2/(\mu^2 k)$. Bemerkenswerterweise erreicht der Mittelwert der Iterierten bei konstanter Schrittweite genau dieselbe optimale Konstante, ohne dass die Schrittweite sinkt (Aufgabe :exref:`ex_mdl-stochastic-gradient-methods-iterate-averaging`). Wegen der scharfen Schwelle bei $c = 1/(2\mu)$ sind $1/k$-Schrittweitenpläne in der Praxis empfindlich. Die kleinste Krümmung $\mu$ ist selten bekannt. Unterschätzen wir $c$, verlieren wir die Konvergenzrate selbst. Deshalb nehmen Lernraten in der Praxis häufig langsamer ab als $1/k$.

<!--en:8887ac2d-->
In unserem durchgehenden Beispiel $\tfrac12(x_1^2 + 1000\,x_2^2)$ bringt die Verschiebung $k_0$ eine zweite Schwierigkeit mit sich. Stabilität schon im ersten Schritt verlangt $c/k_0 < 2/L$, also $k_0 > cL/2$. Die optimale Konstante $c = 1/\mu = 1$ für die langsame Koordinate erzwingt daher $k_0 > 500$. Bei $k_0 = 1000$ und $\mathbf{x}_0 = (1, 1)$ beträgt der transiente Anteil der langsamen Koordinate noch den Bruchteil $(k_0/(k + k_0))^2 \approx 0.008$ ihres Anfangsfehlers, selbst nach $k = 10^4$ Schritten. Der erwartete Verlust liegt bei $4.2 \times 10^{-3}$ und damit mehr als fünfmal so hoch wie die Untergrenze $7.5 \times 10^{-4}$ bei konstanter Schrittweite. Die steife Koordinate erzwingt kleine Anfangsschritte und bremst damit die langsame Koordinate. In $k_0$ tritt die Konditionszahl erneut auf.

<!--en:e50c62a9-->
### Vergleich von Schrittweitenplänen mit der exakten Rekursion

<!--en:03db2ab2-->
Beim heutigen Training fällt die Schrittweite selten wie $1/k$. Häufiger sind eine kosinusförmige Abnahme und Warmup–Stable–Decay (Aufwärmphase, konstante Phase und Abnahme), die wir in :numref:`sec_scheduler` an neuronalen Netzen vergleichen. Für tiefe Netze gibt es keinen allgemeinen Satz, der diese Pläne nach ihrer Güte ordnet. Für unser quadratisches Modell lässt sich der erwartete Verlust jedoch für *jeden* Schrittweitenplan exakt berechnen, ohne Stichproben und ohne Zufallsrauschen. Unsere Herleitung der Rekursion für das zweite Moment gilt für jeden einzelnen Schritt, auch bei variablem $\eta_k$. Damit erhalten wir

<!--en:83ba5d3f-->
Durch Iteration berechnen wir die exakte Kurve des erwarteten Verlusts über ein Budget von $K$ Schritten mit $K$ Multiplikationen. Für dasselbe Budget $K$ vergleichen wir vier in :numref:`fig_mdl-opt-schedule-zoo` dargestellte Verläufe: eine konstante Schrittweite $\eta_0$, den $1/k$-Schrittweitenplan $c/(k + k_0)$ mit $c = \eta_0 k_0$, der ebenfalls bei $\eta_0$ beginnt, die **kosinusförmige Abnahme** :cite:`Loshchilov.Hutter.2016` und **Warmup–Stable–Decay** (WSD) :cite:`Hu.Tu.Han.ea.2024`. Bei WSD bleibt $\eta_0$ über die ersten $80\%$ des Budgets konstant und sinkt dann linear auf null:

<!--en:f4b96de3-->
Sowohl der Kosinusplan als auch WSD beginnen bei $\eta_0$ und enden bei null. Beim Kosinusplan sinkt die Schrittweite während des gesamten Laufs stetig, bei WSD erst im letzten Fünftel. Wir lassen die kurze Aufwärmphase zu Beginn (siehe :numref:`subsec_mdl-schedules-warmup`) weg, da das quadratische Modell schon ab Schritt null bei $\eta_0$ stabil ist. WSD ist beim Pre-Training großer Modelle aus praktischen Gründen beliebt. Von jedem Checkpoint der konstanten Phase aus kann die Schrittweite bis zu einem frei gewählten Endzeitpunkt sinken. Das Gesamtbudget muss also nicht im Voraus feststehen. Mit unserem quadratischen Modell bewerten wir den erwarteten Endverlust. Die Flexibilität des Verfahrens geht dabei nicht ein.

<!--en:f6b32a42-->
![Schrittweitenpläne für ein gemeinsames Budget $K$: konstante Schrittweite, $1/k$-Abnahme (mit $c/k$ bezeichnet), Kosinusplan und Warmup–Stable–Decay (WSD). Jedem Plan geht eine kurze lineare Aufwärmphase voraus, die in der folgenden Berechnung entfällt. Die Abbildung zeigt die Schrittweitenfolgen. Die anschließende Rechnung liefert den erwarteten Verlust für jeden Plan im quadratischen Modell.](../img/mdl-opt-schedule-zoo.svg)

<!--en:0f81b771-->
Welcher der vier Pläne erreicht den niedrigsten Endverlust? Versuchen Sie eine Vorhersage, bevor Sie die Rechnung ausführen. Die folgende Codezelle iteriert die exakte Rekursion mit $\lambda = \sigma = 1$, $K = 2000$, $\eta_0 = 0.3$, $x_0 = 3$ und $k_0 = 50$ für den $1/k$-Schrittweitenplan. Sie gibt den erwarteten Verlust bei $80\%$ des Budgets und am Ende aus und zeichnet anschließend $\mathbb{E} f(x_k)$.

<!--en:bdf5ba8a-->
Da die Rekursion deterministisch ist, ergeben sich bei jedem Lauf dieselben Zahlen. Bei konstanter Schrittweite nähert sich der Verlust seiner Untergrenze $0.0882$. Ab Schritt $9$ beträgt die Abweichung höchstens ein Zehntel der Untergrenze. Bis zum Beginn der Abnahmephase *ist* WSD der Lauf mit konstanter Schrittweite. Bei $80\%$ des Budgets liegt der Verlust also an genau derselben Grenze. Erst in den letzten $400$ Schritten sinkt er um den Faktor $14.4$ auf $6.1 \times 10^{-3}$. Aus diesen Schritten stammt der gesamte Gewinn. Die kosinusförmige Abnahme und die $1/k$-Abnahme enden nahe beieinander bei $1.7 \times 10^{-3}$ und $1.9 \times 10^{-3}$. Das ist ungefähr ein Drittel bis ein Viertel des WSD-Endverlusts.

<!--en:774ed9a5-->
Der Endwert beim $1/k$-Schrittweitenplan stimmt gut mit unserer Proposition überein. Hier gilt $c = 0.3 \cdot 50 = 15$, woraus $A = 225/29 \approx 7.8$ und $\mathbb{E} f(x_K) \approx \tfrac12 A/(K + k_0) = 1.9 \times 10^{-3}$ folgen. Diese Rangfolge gilt exakt für die gewählten Parameter des quadratischen Modells. Auf neuronale Netze lässt sie sich nicht ohne Weiteres übertragen. Beim quadratischen Modell verbessert eine konstante Anfangsphase den Endverlust nicht. Verlängern wir die Abnahmephase von WSD, sinkt der Endverlust monoton. Bei linearer Abnahme über den gesamten Lauf erreicht er $2.7 \times 10^{-3}$.

<!--en:70947fb4-->
## Konvergenzraten bei festem Budget

<!--en:0b9296e6-->
Ein abnehmender Schrittweitenplan eignet sich für einen Lauf mit offenem Ende. In der Praxis kennen wir unser Rechenbudget $K$ jedoch meist im Voraus. Damit stellt sich die Frage anders: Welche konstante Schrittweite sollten wir für $K$ Schritte wählen, und welche Genauigkeit garantiert sie? Ohne starke Konvexität gibt es weder einen geometrischen Kontraktionsfaktor noch eine exakte rauschbedingte Untergrenze. Daher leiten wir obere Schranken her. Die beiden folgenden Schranken haben dieselbe Struktur. Ein transienter Term fällt wie $1/(\eta K)$, ein Rauschterm wächst wie $\eta$. Setzen wir die Terme gleich, erhalten wir $\eta \propto 1/\sqrt{K}$, wie :numref:`fig_mdl-opt-budget-tradeoff` zeigt.

<!--en:c3535668-->
![Die Schranke $D^2/(2\eta K)+\eta G^2/2$ aus :eqref:`eq_mdl-opt-convex-averaged` als Funktion der Schrittweite $\eta$ für $D=G=1$ und zwei Budgets. Mit wachsendem $\eta$ fällt der transiente Anteil, während die Untergrenze steigt. Ihre Summe erreicht das Minimum $DG/\sqrt{K}$ bei $\eta^\star=D/(G\sqrt{K})$. Ein um den Faktor $100\times$ größeres Budget senkt also die optimale Schrittweite und die Schranke um den Faktor $10\times$.](../img/mdl-opt-budget-tradeoff.svg)

<!--en:fb89b6ef-->
### Gemittelte Iterierte bei konvexen Verlustfunktionen

<!--en:2242c28f-->
Unsere erste Schranke setzt nur Konvexität voraus. Wir nutzen sie über die Subgradientenungleichung aus :numref:`sec_mdl-convexity`: Es gilt $f(\mathbf{y}) \ge f(\mathbf{x}) + \mathbf{s}^\top(\mathbf{y} - \mathbf{x})$ für alle $\mathbf{y}$ und jeden Subgradienten $\mathbf{s} \in \partial f(\mathbf{x})$. Dieser ist gleich $\nabla f(\mathbf{x})$, wo immer $f$ differenzierbar ist.

<!--en:f2105440-->
**Proposition (SGD bei konvexer Verlustfunktion, gemittelte Iterierte).** *Sei $f$ konvex mit einem Minimierer $\mathbf{x}^\star$ und dem Optimalwert $f^\star = f(\mathbf{x}^\star)$. Wir iterieren $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \mathbf{g}_k$ ab einem Startpunkt mit $\|\mathbf{x}_0 - \mathbf{x}^\star\| \le D$. Jeder stochastische Gradient erfülle $\mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k] \in \partial f(\mathbf{x}_k)$ und $\mathbb{E}[\|\mathbf{g}_k\|^2 \mid \mathbf{x}_k] \le G^2$. Dann gilt für die gemittelte Iterierte $\bar{\mathbf{x}}_K = \frac1K \sum_{k=0}^{K-1} \mathbf{x}_k$*

<!--en:eb25d4cd-->
*und die Wahl $\eta = D/(G\sqrt{K})$ liefert $\mathbb{E}[f(\bar{\mathbf{x}}_K)] - f^\star \le DG/\sqrt{K}$.*

<!--en:d5458c98-->
**Beweis.** Wir betrachten den quadrierten Abstand zum Minimierer. Multiplizieren wir ihn nach einem Update aus, erhalten wir

<!--en:9145b8b2-->
Wir bilden den bedingten Erwartungswert bei gegebenem $\mathbf{x}_k$. Mit $\mathbf{s}_k = \mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k]$ wird der gemischte Term zu $-2\eta\, \mathbf{s}_k^\top(\mathbf{x}_k - \mathbf{x}^\star)$. Aus der Subgradientenungleichung an der Stelle $\mathbf{y} = \mathbf{x}^\star$ folgt $\mathbf{s}_k^\top(\mathbf{x}_k - \mathbf{x}^\star) \ge f(\mathbf{x}_k) - f^\star$. Auf diesem Schritt beruht das gesamte Argument. Wenn der Verlust weit über dem Optimum liegt, hat der Gradient eine große Komponente in Richtung der Verschiebung gegenüber $\mathbf{x}^\star$. Der Abstieg zieht die Iterierte also näher zum Minimierer. Wir beschränken den letzten Term durch $\eta^2 G^2$, bilden den unbedingten Erwartungswert und setzen $r_k = \mathbb{E}\|\mathbf{x}_k - \mathbf{x}^\star\|^2$. Damit erhalten wir

<!--en:8ce85456-->
Summieren über $k = 0, \ldots, K-1$ ergibt eine Teleskopsumme der quadrierten Abstände. Wegen $r_0 - r_K \le r_0 \le D^2$ gilt $2\eta \sum_{k<K} \mathbb{E}[f(\mathbf{x}_k) - f^\star] \le D^2 + K\eta^2 G^2$. Wir dividieren durch $2\eta K$ und wenden die Jensensche Ungleichung auf die konvexe Funktion $f$ an. Mit $f(\bar{\mathbf{x}}_K) \le \frac1K \sum_{k<K} f(\mathbf{x}_k)$ erhalten wir :eqref:`eq_mdl-opt-convex-averaged`. Für $\eta = D/(G\sqrt{K})$ sind die beiden Terme gleich groß und betragen jeweils $DG/(2\sqrt{K})$. $\blacksquare$

<!--en:fb34963d-->
Die beiden Terme lassen sich direkt deuten. Der transiente Anteil $D^2/(2\eta K)$ gibt an, was es kostet, die Strecke $D$ in $K$ Schritten der Weite $\eta$ zurückzulegen. Der Term $\eta G^2/2$ beschreibt die durch das Rauschen in jedem Schritt verursachte Untergrenze. Der Beweis setzt keinerlei Glattheit voraus. Die Schranke gilt also auch für nichtglatte Verlustfunktionen mit Knicken wie den Hinge-Verlust. Multiplizieren wir das Budget mit $100$, sinken die optimale Schrittweite und der garantierte Fehler jeweils um den Faktor $10$. Die Rate $O(1/\sqrt{K})$ lässt sich im ungünstigsten Fall nicht verbessern. Betrachtet man die gesamte Problemklasse der Proposition, kann kein Algorithmus, der nur erwartungstreue stochastische Gradienten mit $\mathbb{E}\|\mathbf{g}\|^2 \le G^2$ erhält, eine bessere Schranke garantieren als ein konstantes Vielfaches von $DG/\sqrt{K}$ :cite:`Nemirovski.Yudin.1983`.

<!--en:3cdda828-->
Starke Konvexität verändert die Skalierung erheblich. Bei unserer skalaren quadratischen Funktion erreicht die optimale konstante Schrittweite für ein Budget $K$ die Rate $e_K = \Theta(\log K / K)$. Denn bei starker Konvexität klingt der transiente Anteil geometrisch wie $\rho^{2K}$ ab und nicht nur algebraisch wie $1/(\eta K)$.

<!--en:77f44c7c-->
Zwei Einschränkungen sind wichtig. Erstens gilt die Schranke für die *gemittelte* Iterierte $\bar{\mathbf{x}}_K$. Für die letzte Iterierte gilt eine schwächere Schranke. Bei bekanntem Budget fällt der im ungünstigsten Fall optimale Schrittweitenplan für die letzte Iterierte linear auf null :cite:`Defazio.Cutkosky.Mehta.ea.2023`. Zweitens schließt die Annahme $\mathbb{E}\|\mathbf{g}_k\|^2 \le G^2$ Verlustfunktionen mit unbeschränktem Gradienten aus, etwa eine quadratische Funktion auf ganz $\mathbb{R}^d$. Sie gilt jedoch für Lipschitz-stetige Verlustfunktionen mit beschränkten Merkmalen (etwa die unregularisierte logistische Regression) oder wenn die Iterierten in einer beschränkten Menge bleiben. Die Schranke deckt den ungünstigsten Fall ab und kann bei gutartigen Problemen entsprechend grob ausfallen. Beschränken wir unser Problem der logistischen Regression auf eine Kugel, überschätzt sie den tatsächlichen Abstand zum optimalen Verlust um etwa eine Größenordnung.

<!--en:08ae0cc9-->
### Zufällig gewählte Iterierte bei nichtkonvexen Verlustfunktionen

<!--en:3e6e76c0-->
Bei neuronalen Netzen haben wir es zugleich mit nichtkonvexen Verlustfunktionen *und* verrauschten Gradienten zu tun. Bisher haben wir beide Schwierigkeiten einzeln behandelt. Betrachten wir eine Funktion $f$ mit $L$-Lipschitz-stetigem Gradienten (eine $L$-glatte Funktion). Ihre Verlustfläche liegt überall unter einer tangierenden Parabel der Krümmung $L$:

<!--en:509f5529-->
Mit exakten Gradienten und $\eta = 1/L$ liefert diese quadratische obere Schranke ohne Konvexitätsannahme die Beziehung $\min_{k<K}\|\nabla f(\mathbf{x}_k)\|^2 \le 2L\Delta/K$. Dabei ist $f^\star$ eine untere Schranke für $f$, und es gilt $\Delta = f(\mathbf{x}_0) - f^\star$ (:numref:`sec_mdl-gradient-based-optimization`). Mit verrauschten Gradienten auf einer quadratischen Funktion bleibt der Abstieg bei konstanter Schrittweite an einer rauschbedingten Untergrenze stehen. Der Satz von :citet:`Ghadimi.Lan.2013` verbindet beide Ergebnisse. Wir modellieren jeden SGD-Schritt als $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \mathbf{g}_k$. Für den stochastischen Gradienten gelte

<!--en:21b82625-->
Der stochastische Gradient ist also erwartungstreu, und seine Rauschleistung ist gleichmäßig durch $\sigma^2$ beschränkt. Nach :eqref:`eq_mdl-opt-variance` erfüllt ein Minibatchgradient beide Annahmen mit $\sigma^2 = \sup_{\mathbf{x}} \operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x})/b$, sofern dieses Supremum endlich ist. Beim nachstehenden beschränkten $\tanh$-Modell ist es endlich. Bei der Methode der kleinsten Quadrate auf ganz $\mathbb{R}^d$ ist es dagegen unendlich, da das Gradientenrauschen mit dem Abstand zum Minimierer wächst.

<!--en:05131c66-->
**Proposition (SGD findet näherungsweise stationäre Punkte).** *Sei $f$ eine $L$-glatte, nach unten durch $f^\star$ beschränkte Funktion. Wir setzen $\Delta = f(\mathbf{x}_0) - f^\star$ und führen SGD mit stochastischen Gradienten gemäß :eqref:`eq_mdl-opt-sgd-oracle` und konstanter Schrittweite $\eta \le 1/L$ aus. Dann gilt*

<!--en:987691d9-->
*und bei der Wahl $\eta = \min\bigl(1/L,\ \sqrt{2\Delta/(L\sigma^2 K)}\bigr)$ gilt für einen Index $R$, den wir unabhängig vom Lauf gleichverteilt aus $\{0, \ldots, K-1\}$ ziehen,*

<!--en:1c68af5a-->
**Beweis.** Wir wenden die obere Schranke für $L$-glatte Funktionen auf den tatsächlichen Schritt $\mathbf{y} - \mathbf{x} = -\eta\, \mathbf{g}_k$ an:

<!--en:1835e6e4-->
Der bedingte Erwartungswert bei gegebenem $\mathbf{x}_k$ vereinfacht zwei Terme zugleich. Wegen der Erwartungstreue wird der mittlere Term zu $-\eta\,\|\nabla f(\mathbf{x}_k)\|^2$. Die Varianzzerlegung teilt den quadratischen Term in Signal und Rauschen auf: $\mathbb{E}[\|\mathbf{g}_k\|^2 \mid \mathbf{x}_k] = \|\nabla f(\mathbf{x}_k)\|^2 + \mathbb{E}[\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2 \mid \mathbf{x}_k] \le \|\nabla f(\mathbf{x}_k)\|^2 + \sigma^2$. Fassen wir die Terme zusammen, ergibt sich

<!--en:e9dbe493-->
Dies ist das übliche Abstiegslemma mit einem zusätzlichen Varianzterm $\tfrac12 L\eta^2\sigma^2$ pro Schritt. Für $\eta \le 1/L$ gilt $1 - \tfrac{L\eta}{2} \ge \tfrac12$. Wir bilden den unbedingten Erwartungswert und summieren über $k = 0, \ldots, K-1$. Die Teleskopsumme der Verluste ergibt $\mathbb{E}[f(\mathbf{x}_K)] - f(\mathbf{x}_0) \ge -\Delta$. Division beider Seiten durch $\eta K/2$ liefert :eqref:`eq_mdl-opt-ghadimi-lan`. Die linke Seite ist genau $\mathbb{E}[\|\nabla f(\mathbf{x}_R)\|^2]$ für einen gleichverteilt gezogenen Index $R$. Bei der Schrittweite, die beide Terme ausgleicht, unterscheiden wir zwei Fälle. Gilt $\sqrt{2\Delta/(L\sigma^2 K)} \le 1/L$, so stimmen die beiden Terme in :eqref:`eq_mdl-opt-ghadimi-lan` überein. Jeder beträgt $\sigma\sqrt{2L\Delta/K}$. Andernfalls gelten $\eta = 1/L$ und $\sigma^2 < 2L\Delta/K$, und die Schranke lautet $2L\Delta/K + \sigma^2 \le 2L\Delta/K + \sigma\sqrt{2L\Delta/K}$. In beiden Fällen gilt die obere Schranke :eqref:`eq_mdl-opt-ghadimi-lan-balanced`. $\blacksquare$

<!--en:29cba9ab-->
Für $\sigma = 0$ verschwindet der zweite Term, und wir erhalten wieder die deterministische Rate $2L\Delta/K$. Bei vorhandenem Rauschen und einem hinreichend großen Budget mit $K > 2L\Delta/\sigma^2$ (also $\eta < 1/L$) dominiert der Rauschterm. Er fällt nur wie $K^{-1/2}$. Bei $L = \Delta = \sigma = 1$ und $K = 10^4$ beispielsweise ergibt der Ausgleich der Terme die Schrittweite $\eta \approx 0.014$ und die Schranke $2 \times 10^{-4} + 0.028$. Der Rauschterm ist um den Faktor $140$ größer als der deterministische Term. Beachten Sie, dass die Schranke für die *quadrierte* Gradientennorm gilt. Ein um den Faktor $100$ größeres Budget verringert $\mathbb{E}\|\nabla f\|^2$ um den Faktor $10$. Um die Gradientennorm selbst um den Faktor $10$ zu verkleinern, brauchen wir also $10^4$-mal so viele Schritte. Bei einem verzerrten Gradientenschätzer bleibt im Beweis ein zusätzliches Skalarprodukt stehen. Es liefert eine konstante Untergrenze, die auch mit wachsendem $K$ nie verschwindet (Aufgabe :exref:`ex_mdl-stochastic-gradient-methods-biased-estimators`).

<!--en:a36a82af-->
Warum gibt die Proposition eine *zufällig gewählte* Iterierte $\mathbf{x}_R$ zurück und nicht die beste? Bei exakten Gradienten kennen wir $\|\nabla f(\mathbf{x}_k)\|$ in jedem Schritt und können die Iterierte mit der kleinsten Gradientennorm wählen. Bei stochastischen Gradienten kennt der Algorithmus die wahre Gradientennorm nicht. Einen Index $R$ gleichverteilt zu ziehen ist dagegen eine umsetzbare Regel. Der zeitliche Mittelwert in :eqref:`eq_mdl-opt-ghadimi-lan` entspricht genau der erwarteten quadrierten Gradientennorm der so gewählten Iterierten.

<!--en:0bbf68e5-->
Gegenüber dem konvexen Fall ändern sich zwei Dinge. Wir beschränken die quadrierte Gradientennorm anstelle des Abstands zum optimalen Verlust und werten sie an einer zufällig gewählten Iterierten statt am Mittelwert aus. Ohne Konvexität kann ein Verfahren erster Ordnung nur einen kleinen Gradienten garantieren, da ein stationärer Punkt kein globaler Minimierer sein muss. Wie im konvexen Fall ist diese Rate $K^{-1/2}$ im ungünstigsten Fall optimal für jedes Verfahren, das nur auf stochastische Gradienten gemäß :eqref:`eq_mdl-opt-sgd-oracle` zugreift :cite:`Arjevani.Carmon.Duchi.ea.2023`.

<!--en:54e7bbe8-->
### Die Rate bei einem einzelnen $\tanh$-Neuron

<!--en:26448d82-->
Die Proposition liefert eine obere Schranke. Ein Experiment kann daher nicht beweisen, dass die Rate scharf ist. Es kann nur zeigen, wie sich tatsächliche Läufe im Verhältnis zur oberen Schranke $K^{-1/2}$ verhalten. Im Folgenden messen wir die Gradientennorm sowohl für die zufällig gewählte als auch für die beste Iterierte bei einem kleinen nichtkonvexen Problem. Wir passen ein einzelnes $\tanh$-Neuron mit der Methode der kleinsten Quadrate an $n = 64$ verrauschte Beispiele in fünf Dimensionen an:

<!--en:24085b56-->
Jeder Summand ist glatt und beschränkt, aber keiner ist konvex. Denn $\tanh$ geht für große Beträge des Arguments in flache Sättigungsbereiche über. Die folgende Codezelle erzeugt die Daten und definiert eine vektorisierte Hilfsfunktion. Für einen Batch von Parametervektoren, die zeilenweise in `X` stehen, berechnet sie die exakten vollständigen Gradienten.

<!--en:83c756c9-->
Wir führen SGD mit je einem Beispiel für die Budgets $K \in \{125, 500, 2000, 8000\}$ aus und verwenden die Skalierung $\eta = 0.5/\sqrt{K}$ aus dem Satz. Die $20$ unabhängigen Läufe berechnen wir parallel als Zeilen von `X`. Für jedes Budget zeichnet die Codezelle den Mittelwert von $\|\nabla f(\mathbf{x}_k)\|^2$ über den Lauf (das Maß für die zufällig gewählte Iterierte) sowie das Minimum (für die beste Iterierte) auf. Sie gibt die Steigungen beider Größen als Funktionen von $K$ in doppeltlogarithmischer Darstellung aus und zeichnet die Kurve für die zufällig gewählte Iterierte zusammen mit einer Referenzgeraden der Steigung $-1/2$.

<!--en:c2182db9-->
Vergrößern wir das Budget um den Faktor $64$, sinkt das Maß für die zufällig gewählte Iterierte um etwa den Faktor $12$ (bei wiederholten Läufen meist zwischen $9$ und $16$). Das entspricht einer Steigung nahe $-0.6$ in der doppeltlogarithmischen Darstellung. Wie zu erwarten, fällt die Kurve etwas steiler als die obere Schranke $-1/2$ des Satzes. Der Wert für die beste Iterierte fällt mit einer Steigung von ungefähr $-1$ oder noch steiler deutlich schneller. Wie oben gesehen, kann ein Algorithmus diese Iterierte jedoch nicht erkennen, wenn er nur verrauschte Gradienten beobachtet.

<!--en:57adbd24-->
## Varianzreduktion

<!--en:50a04b06-->
### Kontrollvariaten

<!--en:44a8c81e-->
Mit abnehmender Schrittweite überwinden wir die rauschbedingte Untergrenze, indem wir $\eta$ verkleinern. Dafür verlangsamt sich die Konvergenz auf eine algebraische Rate. Ist die Zielfunktion eine *endliche* Summe über $n$ Beispiele, gibt es eine zweite Möglichkeit: Wir beseitigen das Gradientenrauschen selbst. Am Minimierer $\mathbf{x}^\star$ verschwindet der vollständige Gradient, die einzelnen Beispielgradienten $\nabla f_i(\mathbf{x}^\star)$ dagegen im Allgemeinen nicht. Genau diese von null verschiedenen Komponentengradienten verursachen das Rauschen, das bei konstanter Schrittweite fortwährend hinzukommt.

<!--en:4502628c-->
In einem wichtigen Fall brauchen wir überhaupt keine Korrektur. Interpoliert ein Modell den Trainingsdatensatz, passt es also jedes einzelne Beispiel mit Verlust null an, so verschwindet jeder Gradient $\nabla f_i(\mathbf{x}^\star)$ am Minimierer. SGD mit konstanter Schrittweite konvergiert dann bei einer stark konvexen Zielfunktion linear ohne rauschbedingte Untergrenze (Aufgabe :exref:`ex_mdl-stochastic-gradient-methods-interpolation`). Große neuronale Netze interpolieren ihre Trainingsdaten häufig. Das erklärt zum Teil, warum SGD mit konstanter Schrittweite bei ihnen viel besser abschneidet, als die Untergrenze für additives Rauschen erwarten lässt.

<!--en:ed1bdd42-->
Verschwinden die Komponentengradienten bei $\mathbf{x}^\star$ nicht, müssen wir ihre Varianz direkt ausgleichen. Dazu dient ein gängiges Monte-Carlo-Verfahren. Vom verrauschten Schätzer $\nabla f_i(\mathbf{x})$ ziehen wir eine korrelierte Zufallsvariable ab, deren Erwartungswert wir genau kennen, und addieren diesen Erwartungswert wieder. Eine solche Korrektur heißt **Kontrollvariate** (engl. control variate). Sie begegnet uns bei der Baseline für Policy-Gradienten wieder (:numref:`sec_baselines`). Bei einer endlichen Summe bietet sich als Kontrollvariate der Gradient $\nabla f_i(\tilde{\mathbf{x}})$ desselben Beispiels an einem gespeicherten Referenzpunkt $\tilde{\mathbf{x}}$ an. Sein Mittelwert über alle $i$ ist der vollständige Gradient $\nabla f(\tilde{\mathbf{x}})$, den wir einmal pro Durchlauf durch den Datensatz berechnen. Das ergibt den **SVRG**-Schätzer (stochastic variance-reduced gradient) :cite:`Johnson.Zhang.2013`,

<!--en:006d52a0-->
SVRG geht also vom exakten vollständigen Gradienten am Referenzpunkt aus. Es korrigiert ihn um die Änderung des Gradienten eines einzelnen Beispiels zwischen $\tilde{\mathbf{x}}$ und $\mathbf{x}$. Für einen gleichverteilt gezogenen Index $i$ bleibt der Schätzer erwartungstreu,

<!--en:f2883d79-->
während seine Varianz gegen null geht, wenn Iterierte und Referenzpunkt einander näherkommen.

<!--en:5381b9e7-->
**Proposition (der Abstand zum Referenzpunkt beschränkt die Varianz von SVRG).** *Hat jeder Summand $f_i$ einen $L_i$-Lipschitz-stetigen Gradienten und gilt $L_{\max} = \max_i L_i$, so folgt für einen gleichverteilt gezogenen Index $i$*

<!--en:5a05afbe-->
**Beweis.** Die linke Seite ist die Varianz des Zufallsvektors $\nabla f_i(\mathbf{x}) - \nabla f_i(\tilde{\mathbf{x}})$ mit Erwartungswert $\nabla f(\mathbf{x}) - \nabla f(\tilde{\mathbf{x}})$. Die erste Ungleichung folgt unmittelbar daraus, dass die Varianz eines Zufallsvektors höchstens so groß wie sein zweites Moment ist. Wenden wir die $L_i$-Lipschitz-Schranke auf jeden Summanden des zweiten Moments an, erhalten wir die zweite Ungleichung. $\blacksquare$

<!--en:7fa77dc6-->
Ein Vielfaches von $\|\mathbf{x} - \tilde{\mathbf{x}}\|^2$ beschränkt die Rauschleistung nach oben. Nähern sich $\mathbf{x}$ und $\tilde{\mathbf{x}}$ dem Minimierer $\mathbf{x}^\star$, geht die Rauschleistung daher von selbst gegen null. Damit entfällt die rauschbedingte Untergrenze selbst bei konstanter Schrittweite. Anschaulich enthalten die Gradienten desselben Beispiels an zwei benachbarten Punkten fast dasselbe Rauschen. Bilden wir die Differenz, hebt sich das gemeinsame Rauschen auf, während die kleine tatsächliche Änderung des Gradienten erhalten bleibt.

<!--en:3214065c-->
### SVRG und SAGA

<!--en:3c760daa-->
Damit die Schranke :eqref:`eq_mdl-opt-svrg-variance` weiter sinkt, müssen wir den Referenzpunkt regelmäßig an die Iterierte anpassen. In jedem Schritt den vollständigen Gradienten neu zu berechnen, würde jedoch den Vorteil stochastischer Optimierung zunichtemachen. SVRG löst dieses Problem durch eine Einteilung in Epochen. Zu Beginn jeder Epoche speichert es die aktuelle Iterierte als Referenzpunkt $\tilde{\mathbf{x}}$ und berechnet dort den vollständigen Gradienten $\nabla f(\tilde{\mathbf{x}})$. Dann führt es $m$ innere Updates $\mathbf{x} \leftarrow \mathbf{x} - \eta\, \tilde{\mathbf{g}}_i(\mathbf{x})$ mit jeweils neu gezogenen Indizes aus. Die letzte innere Iterierte wird zum nächsten Referenzpunkt. Eine Epoche kostet also $n + 2m$ Auswertungen von Beispielgradienten. :citet:`Johnson.Zhang.2013` (Satz 1) beweisen folgende Aussage. Jede Funktion $f_i$ sei konvex mit einem $L_{\max}$-Lipschitz-stetigen Gradienten, und $f$ sei $\mu$-stark konvex. Ferner gelte $\eta < 1/(4L_{\max})$, und $m$ sei groß genug, damit der Kontraktionsfaktor pro Epoche unter eins liegt. Dann fällt der erwartete Abstand $\mathbb{E}[f(\tilde{\mathbf{x}})] - f^\star$ zum optimalen Verlust geometrisch mit der Zahl der Epochen. (Der Satz wählt eine zufällige innere Iterierte als nächsten Referenzpunkt. Unsere Implementierung verwendet die letzte Iterierte, die sich in der Praxis ähnlich verhält.)

<!--en:6153c599-->
Wir testen SVRG an einer Ridge-Regression, einer stark konvexen endlichen Summe mit $f_i(\mathbf{x}) = \tfrac12(\mathbf{a}_i^\top\mathbf{x} - y_i)^2 + \tfrac{0.1}{2}\|\mathbf{x}\|^2$, $n = 300$ Beispielen und $d = 12$ Parametern. Der exakte Minimierer löst die Normalgleichungen. Jeder Komponentengradient ist $L_i$-Lipschitz-stetig mit $L_i = \|\mathbf{a}_i\|^2 + 0.1$. Die folgende Codezelle erzeugt die Daten, definiert die Komponentengradienten und den vollständigen Gradienten und gibt $L_{\max}$ sowie die theoretische Obergrenze $1/(4L_{\max})$ für die Schrittweite aus.

<!--en:99ec77a2-->
Bei wiederholtem zufälligem Ziehen der Daten liegt $L_{\max}$ meist zwischen $25$ und $40$. Damit liegt die theoretische Obergrenze $1/(4L_{\max})$ für die Schrittweite ungefähr zwischen $0.006$ und $0.01$.

<!--en:d800ba14-->
Als Nächstes vergleichen wir SVRG ($m = n$) mit SGD für je ein Beispiel bei derselben konstanten Schrittweite $\eta = 0.02$. Sie ist mehr als doppelt so hoch wie die Schranke des Satzes. Wir führen $12$ Epochen mit je $3n$ Auswertungen von Beispielgradienten aus. Außerdem prüfen wir für SGD unsere Formel für die Untergrenze bei quadratischen Funktionen. Im stationären Zustand sollte der Abstand zum optimalen Verlust gleich der Summe von $\eta\sigma_i^2/(2(2 - \eta\lambda_i))$ über die Eigenvektoren der Hesse-Matrix sein. Dabei ist $\sigma_i^2$ die Varianz der Beispielgradienten bei $\mathbf{x}^\star$ entlang des Eigenvektors $i$.

<!--en:8c40b589-->
Bei SGD sinkt der Verlust zunächst schnell und bleibt dann bei einigen Vielfachen von $10^{-2}$, genau dort, wo unsere Formel die Untergrenze vorhersagt. Über mehrere Läufe liegt das gemessene Plateau ungefähr zwischen $10\%$ unter und $40\%$ über der Vorhersage, häufiger leicht darüber als darunter. Denn die Formel wertet die Gradientenvarianz nur bei $\mathbf{x}^\star$ aus, während die Beispielgradienten mit zunehmendem Abstand vom Minimierer etwas stärker streuen. SVRG liegt anfangs zurück, endet aber weit vorn. Nach der ersten Epoche mit nur $n$ inneren Schritten und dem Startpunkt als Referenz ist der Abstand zum optimalen Verlust noch größer als bei SGD. Ab Epoche $2$ schrumpft er jedoch pro Epoche ungefähr um den Faktor acht (je nach Lauf zwischen fünf und dreizehn). Er erreicht $10^{-13}$ bis $10^{-8}$ in Epoche $12$. Diese lineare Konvergenz gelingt mit einer Schrittweite, die mehr als doppelt so hoch wie die Schranke des Satzes ist. Die theoretische Bedingung an die Schrittweite ist also hinreichend, aber nicht notwendig. Dieser Vorteil beruht auf drei Voraussetzungen: einer endlichen Summe, mehreren Durchläufen durch die Summe und starker Konvexität für die lineare Rate. Bei einem Datenstrom, in dem jedes Beispiel nur einmal vorkommt, fehlt die endliche Summe für den Referenzgradienten.

<!--en:25b7e2e5-->
**SAGA** :cite:`Defazio.Bach.Lacoste-Julien.2014` vermeidet die regelmäßigen vollständigen Durchläufe ganz. An die Stelle des Referenzpunkts tritt eine Tabelle $\mathbf{t}_1, \ldots, \mathbf{t}_n$ mit dem zuletzt berechneten Gradienten jedes Beispiels. In jedem Schritt bildet SAGA den Schätzer $\nabla f_i(\mathbf{x}) - \mathbf{t}_i + \tfrac1n\sum_j \mathbf{t}_j$ und überschreibt dann $\mathbf{t}_i$ mit $\nabla f_i(\mathbf{x})$. SAGA berechnet pro Schritt nur einen neuen Beispielgradienten, speichert dafür aber $n \times d$ Zahlen (bei verallgemeinerten linearen Modellen genügt ein Skalar pro Beispiel, siehe Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-svrg-inner-loop`).

<!--en:41a6b8c6-->
Bei tiefen neuronalen Netzen haben weder SVRG noch SAGA verlässlich Vorteile gebracht. Die Kontrollvariate veraltet schnell, wenn sich die Verlustlandschaft rasch ändert oder die Datenaugmentation jedes Beispiel bei jedem Durchlauf verändert. Eine zweite Einschränkung begleitet uns seit dem schlecht konditionierten Tal. Alle Verfahren in diesem Abschnitt verwenden dieselbe globale Schrittweite für alle Koordinaten. In unserem durchgehenden Beispiel begrenzt die steifste Richtung deshalb $\eta$ auf $2/1000$. Im nächsten Abschnitt überwinden wir diese Einschränkung. Jede Koordinate passt dann ihre eigene Schrittweite anhand der bisher beobachteten Gradienten an.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:c8f63e5d-->
Der Gradient eines Minibatchs der Größe $b$ ist ein erwartungstreuer Schätzer von $\nabla f$ mit der Rauschleistung $\operatorname{tr}\boldsymbol{\Sigma}/b$. Bei einer quadratischen Funktion und fester Schrittweite $\eta$ nähert sich das zweite Moment jeder Koordinate mit der geometrischen Rate $(1 - \eta\lambda)^2$ der rauschbedingten Untergrenze $\eta\sigma^2/(\lambda(2 - \eta\lambda))$. Die Verlustuntergrenzen $\eta\sigma^2/(2(2 - \eta\lambda))$ der einzelnen Koordinaten addieren sich über alle Dimensionen. Die Untergrenze wächst linear mit $\eta$, während der transiente Anteil bei größerem $\eta$ schneller abklingt. Keine konstante Schrittweite führt also bis zum Minimierer. Jedes Verfahren dieses Abschnitts geht mit diesem Zielkonflikt um. Ein Schrittweitenplan nach Robbins und Monro beseitigt die Untergrenze. Dabei erreicht $\eta_k = c/k$ die Rate $O(1/k)$ nur für $c > 1/(2\mu)$. Bei einem festen Budget von $K$ Schritten garantiert die konstante Schrittweite $\eta \propto 1/\sqrt{K}$ die Schranke $DG/\sqrt{K}$ für die gemittelte Iterierte bei konvexen Verlustfunktionen. Bei glatten nichtkonvexen Verlustfunktionen garantiert sie $O(\sigma\sqrt{L\Delta/K})$ für die erwartete quadrierte Gradientennorm an einer zufällig gewählten Iterierten. Bei einer endlichen Summe bleibt der SVRG-Schätzer mit Kontrollvariate erwartungstreu. Seine Varianz beträgt höchstens $L_{\max}^2\|\mathbf{x} - \tilde{\mathbf{x}}\|^2$. Damit ist bei stark konvexen Problemen lineare Konvergenz mit fester Schrittweite möglich.

<!--en:2294243a-->
## Übungen

<!--en:00e82327-->
1. [code] **Rauschskala und kritische Batchgröße.** ● Um wie viel senkt ein Minibatchschritt den Verlust, und wie hängt die Verbesserung von $b$ ab? Setzen Sie $\mathbf{g} = \nabla f(\mathbf{x})$ und $\mathbf{H} = \nabla^2 f(\mathbf{x})$. Nähern Sie den Verlust nach dem Schritt $\mathbf{x} - \eta\,\hat{\mathbf{g}}_b$ aus :eqref:`eq_mdl-opt-minibatch` durch seine Taylor-Entwicklung zweiter Ordnung um $\mathbf{x}$ an. Die Analyse folgt :citet:`McCandlish.Kaplan.Amodei.ea.2018`.
    1. Verwenden Sie die Kovarianz $\boldsymbol{\Sigma}/b$ von $\hat{\mathbf{g}}_b$ aus dem Beweis von :eqref:`eq_mdl-opt-variance`. Zeigen Sie, dass $\eta_{\textrm{opt}}(b) = \eta_{\max} / (1 + B_{\textrm{noise}}/b)$ den erwarteten Verlust nach dem Schritt minimiert und die erwartete Verbesserung $\Delta_{\max} / (1 + B_{\textrm{noise}}/b)$ beträgt. Dabei gilt

       $$
       \eta_{\max} = \frac{\|\mathbf{g}\|^2}{\mathbf{g}^\top \mathbf{H} \mathbf{g}},
       \qquad
       \Delta_{\max} = \frac{\|\mathbf{g}\|^4}{2\,\mathbf{g}^\top \mathbf{H} \mathbf{g}},
       \qquad
       B_{\textrm{noise}} = \frac{\operatorname{tr}(\mathbf{H} \boldsymbol{\Sigma})}{\mathbf{g}^\top \mathbf{H} \mathbf{g}}.
       $$

    1. Die lineare Skalierungsregel von :citet:`Goyal.Dollar.Girshick.ea.2017` multipliziert $\eta$ mit demselben Faktor wie $b$. Für welche Batchgrößen folgt diese Regel aus $\eta_{\textrm{opt}}(b)$? Welchen Gewinn pro Schritt bringt ein größerer Batch, sobald $b \gg B_{\textrm{noise}}$ gilt?
    1. Ist $\mathbf{H}$ proportional zur Einheitsmatrix, vereinfacht sich $B_{\textrm{noise}}$ zu $B_{\textrm{simple}} = \operatorname{tr}\boldsymbol{\Sigma} / \|\mathbf{g}\|^2$, der Skala des Gradientenrauschens $b_{\textrm{noise}}$ aus dem Text. Berechnen Sie $B_{\textrm{simple}}$ und $B_{\textrm{noise}}$ für die logistische Regression dieses Abschnitts bei $\mathbf{x} = \mathbf{0}$ und nach $5$, $10$ und $20$ Schritten mit vollständigem Gradienten und Schrittweite $\eta = 0.5$. Verwenden Sie `example_grads` und die Hesse-Matrix $\mathbf{H}(\mathbf{x}) = \frac1n \sum_i s_i(1 - s_i)\,\mathbf{a}_i \mathbf{a}_i^\top + 0.1\,\mathbf{I}$ mit $s_i = 1/(1 + e^{y_i \mathbf{a}_i^\top \mathbf{x}})$. Wie verändert sich die Rauschskala, wenn sich die Iterierten dem Optimum nähern? Betrachten Sie $B_{\textrm{noise}}$ als die kritische Batchgröße, die die beiden Fälle aus (b) trennt. Wie sollte sich die Batchgröße im Lauf des Trainings ändern?
   :exlabel:`ex_mdl-gradient-based-optimization-use-eqref-eq-mdl`

<!--en:a4e9acd2-->
1. [code] **Die Konstante eines $1/k$-Schrittweitenplans.** ●● Im Text haben wir die Konstante $A$ aus :eqref:`eq_mdl-opt-one-over-k-constant` in führender Ordnung bestimmt. Beweisen Sie das Ergebnis und übertragen Sie es auf unser durchgehendes Beispiel.
    1. Sei $t = k + k_0$, und es gelte $u_{k+1} \le (1 - a/t)\, u_k + q/t^2$ für alle $k \ge 0$ mit $a > 1$, $q > 0$, $u_k \ge 0$ und $k_0 \ge a$. Beweisen Sie durch Induktion, dass $u_k \le \nu/(k + k_0)$ mit $\nu = \max\{q/(a - 1),\ k_0 u_0\}$ gilt. (Hinweis: Es gilt $1 - a/t \ge 0$ und $1/t - 1/t^2 \le 1/(t + 1)$.)
    1. Wenden Sie das Lemma auf die Rekursion $e_{k+1} = (1 - c\mu/t)^2 e_k + c^2\sigma^2/t^2$ an. Zeigen Sie, dass $(1 - c\mu/t)^2 \le 1 - (2c\mu - (c\mu)^2/k_0)/t$ für $t \ge k_0$ gilt. Damit folgt für jedes $2c\mu > 1$ die Rate $e_k = O(1/k)$, sobald $k_0$ groß genug ist. Bestimmen Sie $a$ und $q$ und zeigen Sie, dass $q/(a - 1) \to A$ für $k_0 \to \infty$ gilt.
    1. Zeigen Sie für $2c\mu < 1$ die Beziehung $e_k = \Theta(k^{-2c\mu})$ anhand des Produkts $\prod_{j<k}(1 - c\mu/(j + k_0))^2$ und der Summe der Rauschbeiträge aus der aufgelösten Rekursion.
    1. Iterieren Sie die Rekursion mit $\mu = \sigma = 1$, $k_0 = 10$ und $e_0 = 1$ bis $k = 10^6$ für $c \in \{0.1, 0.25, 0.5, 1, 2\}$. Geben Sie die Steigung im doppeltlogarithmischen Diagramm zwischen $k = 10^5$ und $10^6$ sowie den Wert von $k\, e_k$ bei $10^6$ an. Welche Wahl der Konstanten kostet mehr: das Doppelte oder die Hälfte des Minimierers von $A$?
    1. Im durchgehenden Beispiel $\tfrac12(x_1^2 + 1000\,x_2^2)$ mit Einheitsrauschen pro Koordinate erfordert Stabilität im ersten Schritt $c/k_0 < 2/1000$. Wählen Sie $c = 1/\mu = 1$, $k_0 = 1000$, $\mathbf{x}_0 = (1, 1)$ und $K = 10^4$. Berechnen Sie den erwarteten Verlust mithilfe der Rekursion für beide Koordinaten (er beträgt etwa $4.2 \times 10^{-3}$). Vergleichen Sie ihn mit dem Verlust bei der konstanten Schrittweite $\eta = 10^{-3}$ ($7.5 \times 10^{-4}$) und bei kosinusförmiger Abnahme ab demselben $\eta_0$ ($1.0 \times 10^{-4}$). Welche Koordinate kommt bei der $1/k$-Abnahme zu kurz, und warum taucht die Konditionszahl in $k_0$ wieder auf?
   :exlabel:`ex_mdl-gradient-based-optimization-one-over-k-constant`

<!--en:113e341b-->
1. [code] **Innere Schleife von SVRG und SAGA.** Verwenden Sie die Daten der Ridge-Regression und die Funktionen `component_grad`, `full_grad_sum` und `objective` dieses Abschnitts. Das Budget beträgt $36n = 10{,}800$ Auswertungen von Beispielgradienten und entspricht den $12$ Epochen der SVRG-Codezelle. Zählen Sie für jeden Lauf die Auswertungen, bis der Abstand zum optimalen Verlust unter $10^{-10}$ sinkt. Prüfen Sie den Abstand einmal pro Epoche.
    1. Lassen Sie die innere Schleife $m$ Schritte laufen, sodass eine Epoche $n + 2m$ Auswertungen kostet. Variieren Sie $m \in \{10, 30, 75, 300, 1200\}$ bei $\eta \in \{0.01, 0.02\}$. Welcher Wert von $m$ ist für die jeweilige Schrittweite am besten? Erklären Sie den Zielkonflikt: Je mehr Schritte sich die Kosten für den Referenzgradienten teilen, desto stärker veraltet die Kontrollvariate. Warum verändert sich der beste Wert von $m$ mit $\eta$?
    1. Implementieren Sie SAGA. Speichern Sie für jedes Beispiel den zuletzt berechneten Gradienten in einer Tabelle. Füllen Sie diese am Punkt $\mathbf{x}_0$ mit $n$ Auswertungen. Verwenden Sie $\nabla f_i(\mathbf{x}) - \mathbf{t}_i + \tfrac1n \sum_j \mathbf{t}_j$ als Schätzer, wobei $\mathbf{t}_i$ der Tabelleneintrag für Beispiel $i$ ist. Vergleichen Sie SAGA bei beiden Schrittweiten mit dem jeweils besten SVRG-Lauf. Wie viele Zahlen enthält die Tabelle bei diesem Problem? Wie viele würden genügen, wenn Sie pro Beispiel nur den Skalar $\mathbf{a}_i^\top\mathbf{x} - y_i$ speicherten?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-svrg-inner-loop`

<!--en:2fa32718-->
1. **Verzerrte Gradientenschätzer.** ● Behalten Sie die Varianzschranke aus :eqref:`eq_mdl-opt-sgd-oracle` bei, messen Sie die Varianz nun aber um den bedingten Erwartungswert. Ersetzen Sie die Erwartungstreue durch $\mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k] = \nabla f(\mathbf{x}_k) + \boldsymbol{\delta}_k$ mit $\|\boldsymbol{\delta}_k\| \le \delta$.
    1. Führen Sie den Beweis von :eqref:`eq_mdl-opt-ghadimi-lan` für $\eta \le 1/L$ erneut und zeigen Sie

       $$\frac{1}{K} \sum_{k=0}^{K-1} \mathbb{E}\left[\|\nabla f(\mathbf{x}_k)\|^2\right] \;\le\; \frac{2\Delta}{\eta K} + L\eta\sigma^2 + \delta^2.$$

       Wenden Sie $L\eta \le 1$ auf den Term $\tfrac{L\eta^2}{2}\|\nabla f(\mathbf{x}_k) + \boldsymbol{\delta}_k\|^2$ an, bevor Sie ihn ausmultiplizieren.
    1. Zeigen Sie, dass der Term $\delta^2$ unverzichtbar ist. Geben Sie eine eindimensionale Funktion $f$ und eine konstante Verzerrung an, bei denen SGD mit $\sigma = 0$ zu einem Punkt mit $|f'|^2 = \delta^2$ konvergiert.
    1. Der Gradient eines einzelnen gleichverteilt und ohne Zurücklegen gezogenen Minibatchs ist weiterhin ein erwartungstreuer Schätzer des vollständigen Gradienten. Welche der folgenden Schätzer sind in diesem Sinne verzerrt? Betrachten Sie den Beispielgradienten bei einem Schritt mit zufälliger Neuanordnung (:numref:`sec_sgd`), bei der jedes Beispiel einmal pro Epoche in einer neuen Zufallsreihenfolge vorkommt, einen stochastischen Gradienten mit auf einen festen Höchstwert begrenzter Norm und den Gradienten an der Iterierten von $\tau$ Schritten zuvor wie beim asynchronen Training.
   :exlabel:`ex_mdl-stochastic-gradient-methods-biased-estimators`

<!--en:e19e5fb3-->
1. [code] **Mittelung der Iterierten.** ● :citet:`Polyak.Juditsky.1992` zeigten, dass die Mittelung der Iterierten bei langsam abnehmender oder konstanter Schrittweite die bestmögliche Konstante liefert. Betrachten Sie das skalare Modell mit konstanter Schrittweite $\eta \in (0, 2/\lambda)$ und einem Startwert aus der stationären Verteilung.
    1. Zeigen Sie, dass $\textrm{Cov}(x_j, x_k) = V\rho^{|j - k|}$ gilt.
    1. Zeigen Sie, dass für den Mittelwert $\bar{x}_K = \frac1K \sum_{k<K} x_k$ die Beziehung $\mathbb{E}[\bar{x}_K^2] = \frac{V}{K}\cdot\frac{1 + \rho}{1 - \rho} - O(K^{-2})$ gilt und dass $V(1 + \rho)/(1 - \rho) = \sigma^2/\lambda^2$ für jede solche Schrittweite $\eta$ gilt. In führender Ordnung hängt das zweite Moment der gemittelten Iterierten also nicht von der Schrittweite ab.
    1. Prüfen Sie dies durch Simulation mit $K = 2000$ und $4{,}000$ Läufen bei $\eta = 0.3$ und $\eta = 0.1$ mit $\lambda = \sigma = 1$. In beiden Fällen sollten Sie $\mathbb{E}[\bar{x}_K^2] \approx 5 \times 10^{-4}$ erhalten, gegenüber den Untergrenzen $0.18$ und $0.053$ für die letzte Iterierte.
    1. Starten Sie stattdessen bei $x_0 = 3$, wirkt der transiente Anteil im Mittelwert nach. Zeigen Sie, dass sein Beitrag zu $\mathbb{E}[\bar{x}_K^2]$ ungefähr $(x_0/(\eta\lambda K))^2$ beträgt. Entscheiden Sie, wann es besser ist, nur über die zweite Hälfte des Laufs zu mitteln.
    1. Vergleichen Sie das Ergebnis mit dem besten $1/k$-Schrittweitenplan, dessen Konstante $A = \sigma^2/\mu^2$ beträgt. Welche Voraussetzungen brauchen die beiden Wege zur Rate $\sigma^2/(\mu^2 K)$ jeweils?
   :exlabel:`ex_mdl-stochastic-gradient-methods-iterate-averaging`

<!--en:7e4cfcab-->
1. [code] **Interpolation und lineare Konvergenz.** ●● Jede Funktion $f_i$ sei konvex mit einem $L_i$-Lipschitz-stetigen Gradienten, und $f$ sei $\mu$-stark konvex. Alle Funktionen $f_i$ mögen ihr Minimum im selben Punkt $\mathbf{x}^\star$ annehmen, wie bei einem Modell, das jedes Trainingsbeispiel exakt anpasst. Die Analyse folgt :citet:`Vaswani.Bach.Schmidt.2019`.
    1. Zeigen Sie, dass $\mathbb{E}_i\|\nabla f_i(\mathbf{x})\|^2 \le 2L_{\max}(f(\mathbf{x}) - f^\star)$ gilt. (Hinweis: Für eine konvexe $L$-glatte Funktion $h$ mit Minimierer $\mathbf{x}^\star$ gilt $\|\nabla h(\mathbf{x})\|^2 \le 2L(h(\mathbf{x}) - h(\mathbf{x}^\star))$.)
    1. Folgern Sie, dass SGD mit je einem Beispiel und konstanter Schrittweite $\eta \le 1/L_{\max}$ die Schranke $\mathbb{E}\|\mathbf{x}_{k+1} - \mathbf{x}^\star\|^2 \le (1 - \eta\mu)\,\mathbb{E}\|\mathbf{x}_k - \mathbf{x}^\star\|^2$ erfüllt, also ohne rauschbedingte Untergrenze linear konvergiert. Welchen Wert hat $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)$ in diesem Fall?
    1. Prüfen Sie das Ergebnis mit der Methode der kleinsten Quadrate für $n = 50$ Beispiele und $d = 100$ Parameter bei $\eta = 1/L_{\max}$ und $\mathbf{x}_0 = \mathbf{0}$. Die Iterierten bleiben im Zeilenraum von $\mathbf{A}$. Dort ist $f$ stark konvex, sodass (b) gilt. Zeigen Sie, dass der Abstand zum optimalen Verlust sowohl bei konsistenten Labels als auch bei durch Rauschen mit Standardabweichung $0.5$ gestörten Labels linear fällt. Erklären Sie, warum das Rauschen hier keine Rolle spielt.
    1. Wiederholen Sie den Versuch mit $n = 200$ und $d = 100$. Jetzt führen verrauschte Labels wieder zu einer rauschbedingten Untergrenze. Warum?
   :exlabel:`ex_mdl-stochastic-gradient-methods-interpolation`

<!--en:f363d5b5-->
[Diskussion](https://d2l.discourse.group/t/36859)

<!--en:a462aad3-->
Stochastische Gradientenverfahren<br>**Minibatchvarianz · stationäre Rauschgrenze · Schrittweiten nach Robbins und Monro · Raten bei festem Budget · Varianzreduktion**

<!--en:d371db32-->
::: {.slide title="Wie Minibatchrauschen die Frage der Optimierung verändert"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ce6c5c4d-->
Beim Training eines Basismodells mit $n = 10^9$ Beispielen entfällt die Berechnung des exakten Gradienten $\nabla f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x})$:

<!--en:0284c14c-->
- **Aufwand für exakte Gradienten:** Sechs Stellen Genauigkeit auf $\tfrac12(x_1^2 + 1000\,x_2^2)$ erfordern $\approx 7{,}000$ Schritte mit vollständigem Batch und damit $7 \times 10^{12}$ Beispielauswertungen.

<!--en:e6d49404-->
- **Abkürzung über Minibatches:** Der Mittelwert über einen zufälligen Minibatch $\mathcal{B}$ der Größe $b = 256$ senkt den Aufwand pro Schritt etwa um den Faktor $4 \times 10^6$ und zeigt im Mittel in die richtige Richtung.

<!--en:e5ddccb6-->
- **Preis des Gradientenrauschens:** Auf $f(x) = \tfrac12 x^2$ mit Gradientenrauschen der Varianz eins und konstanter Schrittweite $\eta = 0.1$ konvergiert die Iterierte nie gegen $x^\star = 0$.

<!--en:2312c6a9-->
- Ihr mittleres Quadrat pendelt sich bei $\mathbb{E}[x_k^2] \to \frac{0.1}{1.9} \approx 0.053$ ein. Die Iterierte schwankt dauerhaft im typischen Abstand $\approx 0.23$ um das Minimum.

<!--en:09729137-->
Durch das Gradientenrauschen verschiebt sich die Frage: von der Zahl der nötigen Schritte (*wie viele Schritte?*) zum erreichbaren Abstand bei gegebener Schrittweite und gegebenem Budget (*wie nahe am Minimum?*).

<!--en:c40a7d86-->
::: {.slide title="Minibatchgradienten und das 1/b-Gesetz der Varianz"}

<!--en:6c734e16-->
[Minibatchstatistik]{.kicker}

<!--en:32a924db-->
Für $b$ unabhängige Beispielgradienten mit Kovarianz $\boldsymbol{\Sigma}$ je Beispiel ist der Minibatchmittelwert $\hat{\mathbf{g}}_b(\mathbf{x}) = \frac{1}{b}\sum_{i \in \mathcal{B}}\nabla f_i(\mathbf{x})$ erwartungstreu. Seine Varianz beträgt:

<!--en:73030be0-->
- **Wurzelgesetz für die Amplitude:** Die effektive Rauschamplitude fällt wie $1/\sqrt{b}$. Ihre Verringerung um den Faktor $10\times$ erfordert um den Faktor $100\times$ größere Batches.

<!--en:b8f12646-->
- **Skala des Gradientenrauschens:** Signal und Rauschen haben bei $b_{\mathrm{noise}} = \operatorname{tr}\boldsymbol{\Sigma}/\|\nabla f(\mathbf{x})\|^2$ dieselbe Leistung.

<!--en:742f5caa-->
::: {.slide title="Empirische Prüfung des 1/b-Gesetzes für das Rauschen"}

<!--en:6c734e16-2-->
[Minibatchstatistik]{.kicker}

<!--en:5d112051-->
Bei der $\ell_2$-regularisierten logistischen Regression ($n = 200$ Beispiele, $d = 5$ Merkmale) beträgt die Rauschleistung eines Beispiels am Punkt $\mathbf{x} = \mathbf{0}$ $\operatorname{tr}\boldsymbol{\Sigma} \approx 1.1$. Für $10{,}000$ zufällige Minibatches je Batchgröße stimmen die Messwerte bei allen Größen mit $\operatorname{tr}\boldsymbol{\Sigma}/b$ überein:

<!--en:59e01eb8-->
Ziehen ohne Zurücklegen (Mischen in jeder Epoche) multipliziert diese Varianz mit dem Korrekturfaktor $(n - b)/(n - 1)$ für eine endliche Grundgesamtheit. Er liegt nahe bei $1$, wenn $b \ll n$ gilt.

<!--en:7c59f2d0-->
::: {.slide title="Exakte Rekursion des zweiten Moments bei Gradientenrauschen"}

<!--en:ee17cedc-->
[Die rauschbedingte Untergrenze]{.kicker}

<!--en:93bb3ed2-->
Für eine skalare Eigenrichtung mit $f(x) = \tfrac12\lambda x^2$ und verrauschtem Gradienten $g_k = \lambda x_k + \xi_k$ ($\mathbb{E}[\xi_k \mid x_k] = 0, \mathbb{E}[\xi_k^2 \mid x_k] = \sigma^2$) erfüllt das zweite Moment $e_k = \mathbb{E}[x_k^2]$ folgende Gleichung ($\rho = 1 - \eta\lambda$):

<!--en:e775cb25-->
- **Orthogonaler Rauschbeitrag:** Der gemischte Term $\mathbb{E}[x_k\xi_k] = 0$ verschwindet nach dem Gesetz der iterierten Erwartungswerte.

<!--en:58698df9-->
- **Physikalisches Gleichgewicht:** Jeder Schritt verringert die bestehende Varianz um den Faktor $\rho^2$ und fügt neue Rauschleistung $\eta^2\sigma^2$ hinzu.

<!--en:895d9304-->
::: {.slide title="Geschlossene Formel für die Bahn und stationäre Kugel um das Minimum"}

<!--en:ee17cedc-2-->
[Die rauschbedingte Untergrenze]{.kicker}

<!--en:2a555e82-->
Für $0 < \eta < 2/\lambda$ ($|\rho| < 1$) hat die lineare Rekursion $e_{k+1} = \rho^2 e_k + \eta^2\sigma^2$ genau einen stationären Fixpunkt $V = \rho^2 V + \eta^2\sigma^2$:

<!--en:fc29bf0b-->
- **Stationäre rauschbedingte Untergrenze:**
  $$V = \frac{\eta^2\sigma^2}{1 - \rho^2} = \frac{\eta\,\sigma^2}{\lambda(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{2\lambda}$$

<!--en:d55cb483-->
- **Zwei Phasen:** Der deterministische transiente Anteil $\rho^{2k}(e_0 - V)$ klingt geometrisch ab. Zurück bleibt eine stationäre **Kugel um das Minimum** mit quadratischem Mittelradius $\sqrt{V} \propto \sqrt{\eta}$.

<!--en:413c96fb-->
::: {.slide title="Addition der Verlustuntergrenzen und lineare Skalierung der Batchgröße"}

<!--en:ee17cedc-3-->
[Die rauschbedingte Untergrenze]{.kicker}

<!--en:dc11521d-->
Beim Deep Learning zählt der erwartete Abstand $\mathbb{E}[f(\mathbf{x}_k)] - f^\star$ zum optimalen Verlust mehr als der Abstand im Parameterraum:

<!--en:eab309cc-->
- **Die Krümmung kürzt sich heraus:** Für eine skalare Eigenrichtung $f(x) = \tfrac12\lambda x^2$ ergibt sich der Verlust durch Multiplikation von $V$ mit $\tfrac12\lambda$. Dabei kürzt sich $\lambda$ für kleine Schritte ($\eta\lambda \ll 1$) heraus:
  $$\mathbb{E}[f(x_\infty)] = \tfrac12\lambda V = \frac{\eta\,\sigma^2}{2(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{4}$$
  In flacheren Richtungen wächst der mittlere quadratische Abstand ($\propto 1/\lambda$) genau um den Faktor, um den die geringere Krümmung den Verlust senkt ($\propto \lambda$).

<!--en:e409ef6b-->
- **Summe über $d$ Eigenrichtungen:** Auf $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ mit Minibatchkovarianz $\boldsymbol{\Sigma}/b$ summieren sich die Varianzen zu $\sum_{i=1}^d \sigma_i^2 = \operatorname{tr}\boldsymbol{\Sigma}/b$. Die Untergrenze des erwarteten Gesamtverlusts lautet:
  $$\mathbb{E}[f(\mathbf{x}_\infty)] = \sum_{i=1}^d \frac{\eta\,\sigma_i^2}{2(2 - \eta\lambda_i)} \approx \frac{\eta\,\operatorname{tr}\boldsymbol{\Sigma}}{4b}$$

<!--en:8d130fd2-->
- **Lineare Skalierungsregel:** Die Untergrenze hängt von $(\eta, b)$ nur über $\eta/b$ ab. Die Verdopplung der Batchgröße $b$ und die Halbierung der Lernrate $\eta$ senken die rauschbedingte Untergrenze daher um genau denselben Faktor.

<!--en:e1e2f17c-->
::: {.slide title="Geschwindigkeit und Genauigkeit im Tal mit κ = 1000"}

<!--en:44c0ed7b-->
[Durchgehendes Beispiel]{.kicker}

<!--en:e5afd9ac-->
Auf $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ mit $\eta = 1/L = 10^{-3}$ und Einheitsrauschen ($\sigma_1^2 = \sigma_2^2 = 1$):

<!--en:13ffae0c-->
- **Flache Richtung ($\lambda_1 = 1$):** Der Kontraktionsfaktor beträgt $\rho_1 = 0.999$. Nach $\approx 3{,}450$ Schritten ist der transiente Anteil bis zur Untergrenze $\tfrac12\lambda_1 V_1 = 2.5 \times 10^{-4}$ abgeklungen ($33\%$).

<!--en:0a75a11d-->
- **Steile Richtung ($\lambda_2 = 1000$):** Schon $1$ Schritt ($\rho_2 = 0$) beseitigt den transienten Anteil. Wegen $2 - \eta\lambda_2 = 1$ trägt diese Richtung aber $\tfrac12\lambda_2 V_2 = 5.0 \times 10^{-4}$ bei ($67\%$ der gesamten Verlustuntergrenze $7.5 \times 10^{-4}$).

<!--en:97154522-->
::: {.slide title="Unterhalb der Rauschgrenze mit den Bedingungen von Robbins und Monro"}

<!--en:b4e3c4d2-->
[Abnehmende Schrittweiten]{.kicker}

<!--en:63b1de23-->
Um die Rauschgrenze einer konstanten Schrittweite zu überwinden, muss ein abnehmender Schrittweitenplan $\eta_k \to 0$ zwei Reihen in Einklang bringen:

<!--en:6dfdf5d9-->
- **Unbegrenzte Reichweite ($\sum \eta_k = \infty$):** Die Iterierte kann sich beliebig weit von $\mathbf{x}_0$ entfernen, um $\mathbf{x}^\star$ zu erreichen.

<!--en:1fafee8b-->
- **Endlicher Rauschbeitrag ($\sum \eta_k^2 < \infty$):** Die insgesamt zugeführte Varianz $\sigma^2\sum \eta_k^2$ bleibt endlich (erfüllt für $\eta_k = c/(k + k_0)$).

<!--en:8825d460-->
::: {.slide title="Feste Schrittweite und 1/k-Abnahme bei logistischer Regression"}

<!--en:b4e3c4d2-2-->
[Abnehmende Schrittweiten]{.kicker}

<!--en:4fcddd80-->
Vergleich bei der $\ell_2$-regularisierten logistischen Regression ($n = 200, d = 5, b = 8$): konstante Schrittweiten $\eta = 0.8$ und $\eta = 0.4$ gegenüber dem abnehmenden Schrittweitenplan $\eta_k = 0.8 / (1 + k/100)$:

<!--en:580c606e-->
- Sinkt die konstante Schrittweite von $0.8$ auf $0.4$, halbiert sich das asymptotische Verlustniveau. Das bestätigt $V \propto \eta$ für eine nichtquadratische Zielfunktion. Der $1/k$-Schrittweitenplan erreicht um den Faktor $20\times$ bis $90\times$ niedrigere Werte und fällt weiter.

<!--en:b3c76e56-->
::: {.slide title="Warum die Konstante c im c/k-Schrittweitenplan entscheidend ist"}

<!--en:b4e3c4d2-3-->
[Abnehmende Schrittweiten]{.kicker}

<!--en:a5256da1-->
Einsetzen von $\eta_k = c/t$ ($t = k + k_0$) in die Rekursion für das zweite Moment ergibt:

<!--en:042f1b7d-->
- **Kritische Schwelle ($2c\mu > 1$):** Der Koeffizientenvergleich mit $e_k \approx A/t$ ergibt $A = \frac{c^2\sigma^2}{2c\mu - 1}$ mit Minimum bei $c^\star = 1/\mu$ ($e_k \approx \frac{\sigma^2}{\mu^2 k}$).

<!--en:7270ed55-->
- **Drastische Verlangsamung ($2c\mu < 1$):** Die Schrittweite sinkt schneller, als sich die Iterierte dem Minimierer nähert. Die Rate verschlechtert sich auf $\Theta(k^{-2c\mu})$.

<!--en:2f44c245-->
::: {.slide title="Empfindlichkeit von 1/k-Schrittweitenplänen und Konditionszahl"}

<!--en:b4e3c4d2-4-->
[Abnehmende Schrittweiten]{.kicker}

<!--en:2632880f-->
Die Schwelle $2c\mu > 1$ erklärt, warum eine reine $1/k$-Abnahme beim Deep Learning empfindlich ist. Moderne Trainingsverfahren verwenden daher häufig langsamer abnehmende Schrittweiten oder mitteln die Iterierten:

<!--en:c29bfdf7-->
- **Optimale Konstante $c^\star = 1/\mu$:** Die Minimierung von $A(c) = \frac{c^2\sigma^2}{2c\mu - 1}$ über $c > \frac{1}{2\mu}$ liefert $c^\star = \frac{1}{\mu}$ und damit die Minimax-Rate $e_k \approx \frac{\sigma^2}{\mu^2 k}$.

<!--en:565d2ec2-->
- **Asymmetrische Empfindlichkeit:** Beim Doppelten des optimalen Werts von $c$ ($c = 2/\mu$) steigt $A$ nur um $33\%$ ($A = \frac{4}{3}\frac{\sigma^2}{\mu^2}$). Beträgt $c$ nur ein Viertel des optimalen Werts ($c = 1/(4\mu)$), verschlechtert sich die Rate von $O(k^{-1})$ auf $O(k^{-1/2})$.

<!--en:bb1d19ed-->
- **Schlechte Konditionierung über $k_0$:** Auf $\tfrac12(x_1^2 + 1000\,x_2^2)$ ist $c = 1/\mu = 1$ für die flache Richtung optimal. Die Stabilitätsbedingung $\eta_0 = c/k_0 < 2/L = 2/1000$ für die steife Richtung erzwingt dann $k_0 > 500$. Dadurch macht die flache Richtung anfangs kaum Fortschritte.

<!--en:1b54064e-->
::: {.slide title="Moderne Schrittweitenpläne mit Kosinus und WSD"}

<!--en:11d04aca-->
[Schrittweitenpläne in der Praxis]{.kicker}

<!--en:1020a938-->
Bei festem Trainingshorizont von $K$ Schritten und einer anfänglichen Lernrate $\eta_0$ bevorzugt man beim heutigen Pre-Training Pläne mit anfangs großen Schrittweiten, die bis $k = K$ stetig auf null sinken:

<!--en:126fc50f-->
- **Kosinusförmige Abnahme:**
  $$\eta_k^{\mathrm{cos}} = \tfrac12\eta_0\left(1 + \cos\frac{\pi k}{K}\right)$$

<!--en:175ff53d-->
- **Warmup–Stable–Decay (WSD):** $\eta_0$ bleibt für $80\%$ des Budgets konstant und sinkt in den letzten $20\%$ linear. Von jedem Checkpoint auf dem Plateau aus lässt sich so bis zu einem beliebigen Horizont $K$ weitertrainieren.

<!--en:2cfdde8d-->
::: {.slide title="Rangfolge der Schrittweitenpläne aus der exakten Rekursion"}

<!--en:11d04aca-2-->
[Schrittweitenpläne in der Praxis]{.kicker}

<!--en:8e43cead-->
Da $e_{k+1} = (1 - \eta_k\lambda)^2 e_k + \eta_k^2\sigma^2$ in jedem Schritt für jede zeitlich veränderliche Folge $\eta_k$ gilt, lässt sich die Kurve des erwarteten Verlusts $\mathbb{E}[f(x_k)] = \tfrac12\lambda e_k$ über $K = 2000$ Schritte exakt und ohne Stichprobenrauschen berechnen:

<!--en:99c47225-->
- WSD verbleibt auf der Untergrenze für konstante Schrittweite ($0.0882$), bis die ersten $1{,}600$ Schritte verstrichen sind. Während der linearen Abnahme sinkt der Verlust um den Faktor $14.4\times$ in $400$ Schritten. Der Kosinusplan und die $c/(k+k_0)$-Abnahme fallen stetig bis auf $\approx 1.8 \times 10^{-3}$.

<!--en:8216e6dd-->
::: {.slide title="Festes Rechenbudget und Gleichgewicht zwischen transientem Anteil und Rauschen"}

<!--en:c0824b57-->
[Raten bei festem Budget]{.kicker}

<!--en:e25f4053-->
Das Rechenbudget $K$ steht im Voraus fest, und es gilt $\mu = 0$ (keine starke Konvexität). Welche konstante Lernrate $\eta$ minimiert den Endfehler?

<!--en:b3b02a38-->
- **Allgemeine Schranke aus zwei Termen:**
  $$\text{Error}(K, \eta) \le \underbrace{\frac{\text{Initial Gap}}{\eta K}}_{\text{transient progress}} + \underbrace{\eta \cdot (\text{Noise Scale})}_{\text{stationary noise floor}}$$

<!--en:f510a72e-->
- **Optimale Skalierung mit dem Budget:** Gleichsetzen der beiden Terme liefert stets $\eta^\star \propto 1/\sqrt{K}$ und damit insgesamt die Konvergenzrate $O(1/\sqrt{K})$.

<!--en:966e63b1-->
::: {.slide title="Beweis der Rate O(1/√K) für konvexes SGD"}

<!--en:071badcc-->
[Beweis bei festem Budget und Konvexität]{.kicker}

<!--en:0cf0e045-->
Für konvexes $f$ mit $\|\mathbf{x}_0 - \mathbf{x}^\star\| \le D$ und $\mathbb{E}[\|\mathbf{g}_k\|^2 \mid \mathbf{x}_k] \le G^2$ bilden die Differenzen der Abstandsquadrate eine Teleskopsumme. Die Jensensche Ungleichung für $\bar{\mathbf{x}}_K = \frac{1}{K}\sum_{k=0}^{K-1}\mathbf{x}_k$ liefert:

<!--en:57187788-->
- **Minimax-Schrittweite bei festem Budget:** Die Wahl $\eta^\star = \frac{D}{G\sqrt{K}}$ gleicht die beiden Terme aus und liefert $\mathbb{E}[f(\bar{\mathbf{x}}_K)] - f^\star \le \frac{DG}{\sqrt{K}}$ ohne Glattheitsannahme.

<!--en:16621c28-->
::: {.slide title="Nichtkonvexes SGD und die Stationaritätsschranke von Ghadimi und Lan"}

<!--en:6c18aace-->
[Beweis bei festem Budget ohne Konvexität]{.kicker}

<!--en:1b191a1e-->
Bei nichtkonvexen neuronalen Netzen sei $f$ eine $L$-glatte Funktion mit $\Delta = f(\mathbf{x}_0) - f^\star$. Für die stochastischen Gradienten gelte $\mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k] = \nabla f(\mathbf{x}_k)$ und $\mathbb{E}[\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2 \mid \mathbf{x}_k] \le \sigma^2$:

<!--en:ce00f77f-->
- **Schritt 1: Stochastisches Abstiegslemma:** $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\mathbf{g}_k$ in die quadratische obere Schranke für $L$-glatte Funktionen einsetzen und $\mathbb{E}[\|\mathbf{g}_k\|^2 \mid \mathbf{x}_k] \le \|\nabla f(\mathbf{x}_k)\|^2 + \sigma^2$ verwenden:
  $$\mathbb{E}\big[f(\mathbf{x}_{k+1}) \mid \mathbf{x}_k\big] \le f(\mathbf{x}_k) - \eta\left(1 - \frac{L\eta}{2}\right)\|\nabla f(\mathbf{x}_k)\|^2 + \frac{L\eta^2\sigma^2}{2}$$

<!--en:3c3d4e79-->
- **Schritt 2: Teleskopsumme für $\eta \le 1/L$ (also $1 - L\eta/2 \ge \tfrac12$).** Summieren über $k = 0, \ldots, K-1$ und Division durch $\eta K / 2$ ergeben eine Schranke für die mittlere quadrierte Gradientennorm:
  $$\frac{1}{K}\sum_{k=0}^{K-1}\mathbb{E}\big[\|\nabla f(\mathbf{x}_k)\|^2\big] \le \frac{2\Delta}{\eta K} + L\eta\sigma^2$$

<!--en:58446edf-->
::: {.slide title="Optimale Schrittweite ohne Konvexität und Stichprobenkomplexität"}

<!--en:6c18aace-2-->
[Beweis bei festem Budget ohne Konvexität]{.kicker}

<!--en:1a805b41-->
Für eine gleichverteilt gezogene Ausgabeiterierte $\mathbf{x}_R$ ($R \sim \mathrm{Uniform}\{0, \ldots, K-1\}$) gilt bei Schrittweite $\eta^\star = \min\!\big(\frac{1}{L}, \sqrt{\frac{2\Delta}{L\sigma^2 K}}\big)$:

<!--en:f8f93e0b-->
- **Warum zufällig stoppen?** SGD kennt die exakte Norm $\|\nabla f(\mathbf{x}_k)\|$ nicht und kann daher die beste Iterierte nicht auswählen. Durch Ziehen von $R$ wird aus dem Mittelwert ein Erwartungswert.

<!--en:7f8afc9f-->
- **Stichprobenkomplexität $O(1/\epsilon^4)$:** Eine Verringerung von $\mathbb{E}\|\nabla f\|^2$ um den Faktor $100\times$ (der quadratische Mittelwert der Gradientennorm sinkt um den Faktor $10\times$) erfordert um den Faktor $10^4\times$ mehr Schritte.

<!--en:fd43fb6a-->
::: {.slide title="Messung der nichtkonvexen Rate an einem tanh-Neuron"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:332a7917-->
Beim Training des nichtkonvexen Regressionsmodells $f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n \big(\tanh(\mathbf{a}_i^\top\mathbf{x}) - y_i\big)^2$ mit einem Neuron ($n = 64, d = 5$) und $\eta = 0.5/\sqrt{K}$ über die Budgets $K \in \{125, 500, 2000, 8000\}$ zeigt sich das theoretische $K^{-1/2}$-Gesetz:

<!--en:40a5f0df-->
- Das Maß $\mathbb{E}\|\nabla f(\mathbf{x}_R)\|^2$ für die zufällig gewählte Iterierte fällt im doppeltlogarithmischen Diagramm mit Steigung $\approx -0.60$, nahe an der für den ungünstigsten Fall garantierten Rate $-0.50$.

<!--en:0094a7b9-->
::: {.slide title="Endliche Summen, Interpolation und Rauschen"}

<!--en:b789c329-->
[Varianzreduktion]{.kicker}

<!--en:4a2b9b1a-->
Warum erreicht SGD mit konstanter Schrittweite eine rauschbedingte Untergrenze? Am globalen Minimierer $\mathbf{x}^\star$ verschwindet der mittlere Gradient ($\nabla f(\mathbf{x}^\star) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x}^\star) = \mathbf{0}$). Die einzelnen Beispielgradienten $\nabla f_i(\mathbf{x}^\star)$ verschwinden im Allgemeinen nicht:

<!--en:4fc00667-->
- **Interpolationsfall ($\nabla f_i(\mathbf{x}^\star) = \mathbf{0}$ für alle $i$):** Passt ein überparametrisiertes Netz jedes Trainingsbeispiel mit Verlust null an, verschwindet jeder Komponentengradient bei $\mathbf{x}^\star$. Damit verschwindet die Rauschvarianz $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) = 0$ am Optimum von selbst. SGD konvergiert bei konstanter Schrittweite linear ohne rauschbedingte Untergrenze.

<!--en:5b2143c3-->
- **Endliche Summen ohne Interpolation ($\nabla f_i(\mathbf{x}^\star) \neq \mathbf{0}$):** Stimmen die Beispielgradienten bei $\mathbf{x}^\star$ nicht überein, lässt sich ihre bleibende Varianz mit einer **Kontrollvariate** aus der Monte-Carlo-Methode aufheben. Dazu dient die Subtraktion eines korrelierten Gradienten, dessen exakter Mittelwert über die $n$ Beispiele bekannt ist.

<!--en:5fa884bb-->
::: {.slide title="SVRG senkt die Varianz mit einem Referenzpunkt und Kontrollvariaten"}

<!--en:b789c329-2-->
[Varianzreduktion]{.kicker}

<!--en:7552b150-->
**SVRG** speichert regelmäßig einen Referenzpunkt $\tilde{\mathbf{x}}$ mit dem exakten vollständigen Gradienten $\nabla f(\tilde{\mathbf{x}})$ und bildet den erwartungstreuen Schätzer mit Kontrollvariate:

<!--en:0a22239d-->
- **Verschwindende Varianz:** Aus der $L_{\max}$-Glattheit jeder Funktion $f_i$ folgt $\mathbb{E}_i\|\tilde{\mathbf{g}}_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2 \le L_{\max}^2\|\mathbf{x} - \tilde{\mathbf{x}}\|^2$.

<!--en:f30c382d-->
- **Lineare Konvergenz bei festem $\eta$:** Für $\mathbf{x}, \tilde{\mathbf{x}} \to \mathbf{x}^\star$ verschwindet das Gradientenrauschen, ohne dass $\eta$ sinkt.

<!--en:e0d0bbd9-->
::: {.slide title="SVRG und SGD bei der Ridge-Regression"}

<!--en:b789c329-3-->
[Varianzreduktion]{.kicker}

<!--en:0de87909-->
Vergleich von SGD mit je einem Beispiel und SVRG bei gleicher konstanter Schrittweite $\eta = 0.02$ und gleichem Budget von $3n$ Beispielgradienten pro Epoche bei der Ridge-Regression ($n = 300, d = 12$):

<!--en:c3ea56e4-->
- **SGD** bleibt dauerhaft an der exakten theoretischen Rauschgrenze $\sum_i \frac{\eta\sigma_i^2}{2(2-\eta\lambda_i)} \approx 2 \times 10^{-2}$. Bei **SVRG** fällt der Abstand zum optimalen Verlust geometrisch um den Faktor $\sim 8\times$ pro Epoche bis zur Maschinengenauigkeit ($10^{-11}$).

<!--en:3f1f0a04-->
::: {.slide title="Verständnisfragen zu Rauschgrenzen, Schrittweitenplänen und Budgets"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:68138b72-->
Fragen zur Dynamik stochastischer Gradientenverfahren:

<!--en:9be41fd0-->
1. Welche Untergrenze hat der erwartete Verlust einer skalaren quadratischen Funktion mit $\lambda = \sigma = 1$ bei $\eta = 0.3$? Um welchen Faktor sinkt sie, wenn sich gleichzeitig $\eta$ auf $0.15$ halbiert und die Batchgröße $b$ verdoppelt wird?

<!--en:1235f100-->
2. Eine stark konvexe Zielfunktion habe die kleinste Krümmung $\mu = 0.25$. Welche Bedingung muss $c$ erfüllen, damit der Schrittweitenplan $\eta_k = c/(k + k_0)$ die Rate $O(1/k)$ erreicht? Welche Wahl von $c$ minimiert die asymptotische Konstante?

<!--en:9ea21079-->
3. Eine glatte nichtkonvexe Verlustfunktion habe $L = \Delta = \sigma = 1$, das Budget betrage $K = 10^6$. Welche Schrittweite $\eta$ und welche Stationaritätsschranke liefert der Satz von Ghadimi und Lan?

<!--en:fe1b9715-->
**Lösungen:** **(1)** Bei $\eta = 0.3$ beträgt die Verlustuntergrenze $\frac{\eta\sigma^2}{2(2 - \eta\lambda)} = \frac{0.3}{3.4} \approx 0.0882$. Halbiert sich $\eta$ und verdoppelt sich $b$, sinkt $\eta/b$ um den Faktor **$4\times$**. Die Untergrenze sinkt etwas stärker, ungefähr um den Faktor $\approx 4.3\times$, weil $2 - \eta\lambda$ von $1.7$ auf $1.85$ steigt. **(2)** Die Bedingung $2c\mu > 1$ mit $\mu = 0.25$ verlangt **$c > 2$**. Unterhalb von $c = 2$ verlangsamt sich die Konvergenz auf $O(k^{-c/2})$. Die optimale Konstante ist $c^\star = 1/\mu = \mathbf{4}$. **(3)** Die Schrittweite $\eta = \sqrt{2\Delta/(L\sigma^2 K)} = \sqrt{2} \times 10^{-3} \approx \mathbf{0.00141}$ garantiert $\mathbb{E}\|\nabla f(\mathbf{x}_R)\|^2 \le 2 \times 10^{-6} + 2\sqrt{2} \times 10^{-3} \approx \mathbf{0.00283}$.

<!--en:af620c2d-->
::: {.slide title="Stochastische Gradientenverfahren und ihre Konvergenzraten im Überblick"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:a5f23339-->
| Fall / Algorithmus | Schrittweite / Schätzer | Konvergenzrate / Untergrenze | Wirkungsweise und praktische Folgerung |

<!--en:8cc0715e-->
| **SGD mit konstanter Schrittweite** | Festes $\eta \in (0, 2/L)$, Minibatchgröße $b$ | Transienter Anteil $\rho^{2k} \to$ Verlustuntergrenze $\approx \frac{\eta \operatorname{tr}\boldsymbol{\Sigma}}{4b}$ | Anfangs schneller Abstieg, dann Schwankungen in einer Kugel um das Minimum $\propto \eta/b$ |

<!--en:41e774e8-->
| **Abnahme nach Robbins und Monro** | $\eta_k = \frac{c}{k + k_0}$ mit $2c\mu > 1$ | $\mathbb{E}\|\mathbf{x}_k - \mathbf{x}^\star\|^2 \approx \frac{c^2\sigma^2}{(2c\mu - 1)k}$ | Überwindet die rauschbedingte Untergrenze. Die Rate sinkt auf $O(k^{-2c\mu})$ für $c < \frac{1}{2\mu}$ |

<!--en:3fe24b99-->
| **Konvex, festes Budget** | Festes $\eta = \frac{D}{G\sqrt{K}}$, gemittelte Iterierte $\bar{\mathbf{x}}_K$ | $\mathbb{E}[f(\bar{\mathbf{x}}_K)] - f^\star \le \frac{DG}{\sqrt{K}}$ | Ohne Glattheit minimax-optimal mit Rate $O(K^{-1/2})$ |

<!--en:360c82e9-->
| **Nichtkonvex, festes Budget** | Festes $\eta = \sqrt{\frac{2\Delta}{L\sigma^2 K}}$, zufällig gewählte Iterierte $\mathbf{x}_R$ | $\mathbb{E}\|\nabla f(\mathbf{x}_R)\|^2 \le \frac{2L\Delta}{K} + 2\sigma\sqrt{\frac{2L\Delta}{K}}$ | Garantiert näherungsweise Stationarität in $O(1/\epsilon^4)$ Schritten |

<!--en:baf08874-->
| **SVRG / SAGA** | $\nabla f_i(\mathbf{x}) - \nabla f_i(\tilde{\mathbf{x}}) + \nabla f(\tilde{\mathbf{x}})$ | Varianz $\le L_{\max}^2\|\mathbf{x} - \tilde{\mathbf{x}}\|^2 \to 0$, lineare Rate | Beseitigt bei endlichen Summen die Rauschgrenze bei konstantem $\eta$ |
