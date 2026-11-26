<!--en:f28333bc-->
# Adaptive Verfahren

<!--en:b4d32036-->
Wie die meisten großen neuronalen Netze trainieren wir die Transformer in diesem Buch mit Adam oder seiner Variante AdamW (:numref:`sec_adamw`). Ein wesentlicher Grund, diese Verfahren dem einfachen stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD) vorzuziehen, ist die Schrittweite (engl. step size). Die Verfahren aus :numref:`sec_mdl-stochastic-gradient-methods` verwenden dieselbe Schrittweite für alle Koordinaten und passen sie nur im Laufe der Zeit an. Ein modernes Netz hat jedoch $10^7$ bis $10^{11}$ Parameter, deren Krümmungen (engl. curvature) sich um Größenordnungen unterscheiden und im Voraus unbekannt sind. Bei gemeinsamer Schrittweite begrenzt die steilste Richtung den Schritt, und in den flachen Richtungen kommen wir kaum voran. Wie kann ein Optimierer allein aus den Gradienten für jede Koordinate eine eigene Schrittweite bestimmen, und wo stößt dieses Vorgehen an Grenzen?

<!--en:6cd120e1-->
Unser durchgängiges Beispiel zeigt, wie viel davon abhängt. Betrachten wir das schmale Tal $f(\mathbf{x}) = \tfrac12\bigl(x_1^2 + 1000\, x_2^2\bigr)$ mit Startpunkt $(1, 1)$. Der Gradientenabstieg mit der optimalen gemeinsamen Schrittweite $\eta = 2/1001$ braucht 6160 Schritte, bis $f$ unter $10^{-8}$ fällt. Ein Schritt, der klein genug ist, um $x_2$ stabil zu halten, ist für $x_1$ um den Faktor tausend zu klein. Das Heavy-Ball-Verfahren aus :numref:`sec_mdl-gradient-based-optimization` addiert zu jedem neuen Schritt einen Bruchteil des vorherigen und braucht nur 296 Schritte. (Beide Zahlen liegen etwas unter den dort genannten $6{,}908$ und $315$ Schritten, weil dort der Abstand zum Minimierer über den Abbruch entscheidet und hier der Wert von $f$.) Ein Optimierer, der für jede Koordinate eine eigene Schrittweite allein aus den beobachteten Gradienten schätzt, braucht nur wenige Dutzend Schritte. Noch bemerkenswerter ist, wovon diese Zahl kaum abhängt. Machen wir das Tal noch einmal hundertmal schmaler, kommen nur sechs Schritte hinzu.

<!--en:3879fea3-->
Solche eindrucksvollen Zahlen lassen sich leicht erzeugen und ebenso leicht missverstehen. Deshalb widmen wir einen großen Teil dieses Abschnitts einem fairen Vergleich. Wir verwenden für das Heavy-Ball-Verfahren die theoretisch optimalen Einstellungen, testen für Adam eine Reihe von Schrittweiten und drehen das Tal außerdem um $45^\circ$. Unter diesen Bedingungen sieht Adams Vorteil ganz anders aus, als die erste Zahl vermuten lässt.

<!--en:bf050975-->
Der Mechanismus ist einfach. AdaGrad, RMSProp und Adam teilen den Schritt jeder Koordinate durch das quadratische Mittel (engl. root mean square, RMS) ihrer bisherigen Gradienten. Zunächst beweisen wir, dass Adam durch diese Normierung unempfindlich gegenüber der Skalierung jeder einzelnen Koordinate wird. Dann untersuchen wir die Grenzen dieser Symmetrie anhand einer Drehung des Tals, einer periodischen Folge konvexer Verlustfunktionen, des Gewichtszerfalls (engl. weight decay), der ersten Trainingsschritte und der Matrixstruktur einer Schicht. Besonders wichtig sind die Proposition zur Skaleninvarianz und die Fixpunktanalyse von AdamW. Beim ersten Lesen können Sie die Regret-Schranken für AdaGrad und die strukturierten Vorkonditionierer am Ende überspringen. Durchgehend nummeriert $t$ die Schritte des Optimierers, $h_i$ bezeichnet die Krümmung der Koordinate $i$ und $\lambda$ den Koeffizienten des Gewichtszerfalls.

<!--en:b8cb0084-->
## Schrittweiten für einzelne Koordinaten

<!--en:048b9652-->
### Eine diagonale quadratische Funktion

<!--en:9f17ce41-->
Bei einer quadratischen Funktion braucht der Gradientenabstieg eine zur Konditionszahl $\kappa = h_{\max}/h_{\min}$ proportionale Anzahl von Schritten, weil die gemeinsame Schrittweite unter $2/h_{\max}$ bleiben muss (:numref:`sec_mdl-gradient-based-optimization`). Wie sich das Problem lösen lässt, zeigt im einfachsten Fall eine diagonale quadratische Funktion in zwei Variablen,

<!--en:40c33980-->
Da jede partielle Ableitung nur von der eigenen Koordinate abhängt, können wir die beiden Schrittweiten unabhängig wählen. Bei eigener Schrittweite $\eta_i$ für jede Koordinate geht $x_i$ in $x_i - \eta_i h_i x_i = (1 - \eta_i h_i)\, x_i$ über. Mit $\eta_i = 1/h_i$ erreichen alle Koordinaten in einem einzigen Schritt null. Für $h = (1, 1000)$ sind die idealen Schrittweiten $1$ und $10^{-3}$. Das entspricht dem Newton-Verfahren für eine diagonale Hesse-Matrix. Die Anzahl der Schritte hängt dann nicht mehr von der Konditionszahl ab.

<!--en:9c090f49-->
Zwei Hindernisse trennen diesen Idealfall von der Praxis. Die Krümmungen $h_i$ sind unbekannt und ändern sich im Laufe des Trainings. Außerdem beobachten wir nur verrauschte Gradientenschätzungen aus Minibatches. Wir müssen die Schrittweite jeder Koordinate daher allein aus den Gradienten ableiten und verwenden deren Betrag als Ersatz für die Krümmung. Bei der diagonalen quadratischen Funktion hat die Gradientenkomponente $g_i = h_i x_i$ den Betrag $h_i |x_i|$. Sind die Koordinaten ähnlich groß, hat eine Koordinate mit großen Gradienten auch eine hohe Krümmung. Die Division durch den Gradientenbetrag entspricht dann bis auf einen Faktor der Division durch die Krümmung. Unterscheiden sich die Größenordnungen der Koordinaten dagegen stark, hat $g_i/|g_i|$ weiterhin den Betrag eins. Die Normierung legt den pro Schritt zurückgelegten Weg auf $\eta$ fest, anstatt den tatsächlichen Newton-Schritt $x_i$ zu machen.

<!--en:6bfb8f01-->
### AdaGrad

<!--en:e90d9fc2-->
Unterschiedliche Krümmungen sind ein Grund für koordinatenweise Schrittweiten. Ein zweiter, unabhängiger Grund ist die sehr unterschiedliche Häufigkeit, mit der wir einzelne Koordinaten aktualisieren. Eine global abnehmende Schrittweite wie $\eta/\sqrt{t}$, die klassische Wahl bei der konvexen Online-Optimierung und der nichtglatten konvexen Optimierung, behandelt alle Koordinaten gleich. Angenommen, Koordinate 1 erhält bei jedem Schritt einen Gradienten von $\pm 1$, Koordinate 2 dagegen nur einmal in $10^4$ Schritten, wie etwa das Gewicht eines seltenen Wortes in einem Bag-of-Words-Modell. Bis $t = 10^4$ sinken beide Schrittweiten nach dem globalen Plan auf $\eta/100$. Die einzige Beobachtung für die seltene Koordinate verschiebt sie also nur um ein Hundertstel von $\eta$. Summiert jede Koordinate dagegen ihre eigenen quadrierten Gradienten, ergeben sich die laufenden Summen $10^4$ und $1$. Teilen wir $\eta$ durch die jeweiligen Quadratwurzeln, erhält die häufig aktualisierte Koordinate die Schrittweite $\eta/100$, während die selten aktualisierte bei $\eta$ bleibt.

<!--en:521c1141-->
Das ist **AdaGrad** :cite:`Duchi.Hazan.Singer.2011`. Wir schreiben $\mathbf{g}_t \odot \mathbf{g}_t$ für den Vektor der quadrierten Gradientenkomponenten (im Folgenden kurz $\mathbf{g}_t^2$). Auch Quadratwurzeln und Divisionen von Vektoren verstehen wir elementweise. AdaGrad führt die laufende Summe $\mathbf{s}_t$ der quadrierten Gradienten mit und teilt den Schritt durch ihre Quadratwurzel:

<!--en:04ee4944-->
mit Startwert $\mathbf{s}_0 = \mathbf{0}$. Eine kleine Konstante $\epsilon$ verhindert die Division durch null (in :numref:`sec_adam` im Haupttext steht $\epsilon$ unter der Wurzel, was hier qualitativ keinen Unterschied macht). Die Schrittweite jeder Koordinate nimmt nun entsprechend ihren eigenen bisherigen Gradienten ab und hängt nicht mehr von der globalen Schrittzahl ab.

<!--en:a5e0521d-->
*Aus Sicht der Metrik.* Der steilste Abstieg hängt davon ab, wie wir die Schrittlänge messen. In der gewichteten Norm $\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P}\, \mathbf{d}}$ mit positiv definiter Matrix $\mathbf{P}$ ist $-\mathbf{P}^{-1}\nabla f$ die Richtung des steilsten Abstiegs (Aufgabe :exref:`ex_mdl-gradient-based-optimization-show-that-is-de` in :numref:`sec_mdl-gradient-based-optimization`). AdaGrad entspricht dem steilsten Abstieg in der diagonalen Metrik $\mathbf{P}_t = \mathrm{diag}(\sqrt{\mathbf{s}_t})$. Eine Bewegung entlang der Koordinate $i$ wird umso stärker bestraft, je größer das aus ihren bisherigen Gradienten berechnete Maß $\sqrt{\sum_{\tau \le t} g_{\tau,i}^2}$ ist. Eine Matrix wie $\mathbf{P}_t$, die den Gradienten vor dem Schritt umskaliert, heißt **Vorkonditionierer** (engl. preconditioner). Während das Newton-Verfahren die tatsächliche Krümmung $\mathrm{diag}(h_i)$ verwendet, greift AdaGrad auf die bisherigen Gradienten zurück (Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-adagrad-metric-view`).

<!--en:4309c785-->
*Aus Sicht des Regrets.* AdaGrad wurde ursprünglich für die **konvexe Online-Optimierung** entwickelt. Hier treffen konvexe Verlustfunktionen $f_1, f_2, \ldots$ nacheinander ein, und das Lernverfahren muss $\mathbf{x}_t$ wählen, bevor es $f_t$ kennt. Die Leistung über $T$ Runden misst der **Regret** $\sum_{t=1}^T f_t(\mathbf{x}_t) - \min_{\mathbf{x}} \sum_{t=1}^T f_t(\mathbf{x})$. Er gibt an, wie viel Verlust sich zusätzlich gegenüber dem im Nachhinein besten fest gewählten Punkt angesammelt hat. Ein Algorithmus lernt in diesem Sinne, wenn sein durchschnittlicher Regret (Regret geteilt durch $T$) gegen null geht.

<!--en:69eba129-->
Auf einer konvexen Menge mit euklidischem Durchmesser $D_2$ und beschränkten Gradienten $\|\mathbf{g}_t\| \le G$ erreicht der Online-Gradientenabstieg mit passend gewählter globaler Schrittweite einen Regret der Größenordnung $D_2 G \sqrt{T}$ (Beweis in Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-regret-ogd`). AdaGrad erreicht dagegen einen Regret der folgenden Größenordnung :cite:`Duchi.Hazan.Singer.2011`:

<!--en:500e447f-->
Dabei ist $D_\infty$ der Durchmesser der Menge in der $\ell_\infty$-Norm. Für jede der $d$ Koordinaten zählen nur ihre eigenen bisherigen Gradienten und nicht die größtmögliche Gradientennorm über alle Koordinaten.

<!--en:d259a5e5-->
Auf dem Quader $[-1, 1]^d$ gilt $D_2 = \sqrt{d}\, D_\infty$. Konzentriert sich die Gradientenmasse auf wenige Koordinaten, ist die Schranke von AdaGrad um bis zu einen Faktor $\sqrt{d}$ kleiner. Verteilt sich die Masse gleichmäßig auf die Koordinaten, stimmen die beiden Schranken überein. Für One-Hot-Gradienten, die Koordinate $i$ mit Wahrscheinlichkeit $p_i$ treffen, beträgt das erwartete Verhältnis der beiden Schranken höchstens $\sum_i \sqrt{p_i}/\sqrt{d}$ (Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-concentrated-gradients`). Bei gleichförmigen $p_i$ beträgt dieses Verhältnis $1.00$, obwohl jeder einzelne Gradient dünn besetzt ist. Für Worthäufigkeiten nach dem Zipfschen Gesetz mit Exponent $1$ bei $d = 10^4$ sinkt es dagegen auf $0.63$. Der Vorteil von AdaGrad beruht also auf der *Konzentration* der Gradientenmasse auf bestimmte Koordinaten. Sparsität allein reicht nicht aus.

<!--en:150aae49-->
Die Quadratwurzel in :eqref:`eq_mdl-opt-adagrad` hat einen bestimmten Grund. Ihr Exponent bringt die beiden gegenläufigen Terme im Beweis der Regret-Schranke genau ins Gleichgewicht (Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-regret-ogd`c).

<!--en:3820097b-->
Außerhalb der konvexen Online-Optimierung wird die unbegrenzte Summation jedoch zur größten Schwäche von AdaGrad. Bei anhaltendem Gradientenrauschen mit Varianz $\sigma^2$ wächst die Summe gemäß $s_{t,i} \approx \sigma^2 t$, sodass die effektive Schrittweite wie $\eta/(\sigma\sqrt{t})$ abnimmt. Damit sinkt die Schrittweite bei AdaGrad von sich aus mit der Schrittzahl, unabhängig vom Fortschritt der Optimierung. Die Rate $1/\sqrt{t}$ ist für den Regret bei konvexen Problemen optimal. Bei langen nichtkonvexen Trainingsläufen nimmt die Schrittweite jedoch unaufhaltsam ab, ob der Verlust noch sinkt oder nicht. Erreicht der Optimierer spät im Training ein Plateau, ist die Schrittweite bereits fast null.

<!--en:edc5ebc3-->
### RMSProp

<!--en:5ed10ad8-->
**RMSProp** :cite:`Tieleman.Hinton.2012` verhindert diese stetige Abnahme, indem es die Summe über alle Schritte durch einen exponentiellen gleitenden Mittelwert (engl. exponential moving average, EMA) ersetzt. So reagiert die Schätzung der Größenordnung auf die jüngsten Gradienten:

<!--en:b940d53b-->
Entwickeln wir die Rekursion, sehen wir, dass $\mathbf{v}_t$ dem quadrierten Gradienten von vor $k$ Schritten das Gewicht $(1 - \beta_2)\beta_2^k$ zuweist. Das Gewicht alter Gradienten halbiert sich also ungefähr alle $\log 2/(1 - \beta_2)$ Schritte, beim üblichen RMSProp-Wert $\beta_2 = 0.9$ alle 7 Schritte und bei $\beta_2 = 0.999$ alle 693 Schritte. Die Schrittweiten gehen nun nicht mehr zwangsläufig gegen null. Dafür gerät auch jeder einzelne große Gradient auf derselben Zeitskala in Vergessenheit. Diese Schwäche nutzen wir weiter unten für ein Gegenbeispiel.

<!--en:2f22123d-->
### Adam und die Bias-Korrektur

<!--en:7c73fcf4-->
**Adam** :cite:`Kingma.Ba.2014` kombiniert die Schätzung des zweiten Moments aus RMSProp mit einem exponentiellen gleitenden Mittelwert des Gradienten selbst. Beide Mittelwerte korrigiert Adam für die Initialisierung bei null. In der Notation von :numref:`sec_adam` lautet die Aktualisierung

<!--en:533062a7-->
mit $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$ und den Standardwerten $\beta_1 = 0.9$, $\beta_2 = 0.999$ und $\epsilon = 10^{-8}$. Das erste Moment $\mathbf{m}_t$ ist die Heavy-Ball-Geschwindigkeit aus :numref:`subsec_mdl-momentum-acceleration`, skaliert mit $1 - \beta_1$. Dadurch wird aus einer laufenden Summe ein gewichteter Mittelwert. Er glättet das Minibatch-Rauschen, dessen Richtung einfaches RMSProp unmittelbar folgt.

<!--en:37466fe4-->
Die Division durch $1 - \beta_2^t$ gleicht die Initialisierung mit $\mathbf{v}_0 = \mathbf{0}$ aus. Entwickeln wir die Rekursion, erhalten wir $\mathbf{v}_t = (1 - \beta_2) \sum_{\tau \le t} \beta_2^{\,t-\tau}\, \mathbf{g}_\tau^2$. Ist die Größenordnung der quadrierten Gradienten stationär, gilt also $\mathbb{E}[\mathbf{g}_\tau^2] = \overline{\mathbf{g}^2}$ für jedes $\tau$, so ergibt die Summe der geometrischen Reihe

<!--en:7dfcc360-->
Unter Stationarität ist $\hat{\mathbf{v}}_t$ somit eine erwartungstreue Schätzung von $\overline{\mathbf{g}^2}$ in jedem Schritt $t$; im selben Rechengang folgt aus $\beta_1$ für $\hat{\mathbf{m}}_t$ eine erwartungstreue Schätzung des mittleren Gradienten.

<!--en:7e2584b3-->
Ohne diese Korrektur ist das Defizit durch die Initialisierung groß und langlebig. Beim Standardwert $\beta_2 = 0.999$ erfasst $\mathbf{v}_{10}$ nur etwa $1\%$ seines stationären Zielwerts, und selbst bei $t = 1000$ sind erst $63\%$ erreicht (:numref:`sec_adam`). Noch bemerkenswerter ist der zeitliche Verlauf des Fehlers. Ohne beide Bias-Korrekturen ist die Aktualisierung $\mathbf{m}_t/\sqrt{\mathbf{v}_t}$ um den Faktor $(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$ falsch skaliert. Dieser beginnt bei $3.16$ für $t = 1$ und steigt überraschenderweise noch bis zum Höchstwert $6.6$ bei $t = 12$. Der Mittelwert des ersten Moments ist nach etwa zehn Schritten aufgebaut, der des zweiten braucht etwa tausend. :numref:`fig_mdl-opt-bias-correction` zeigt das Defizit und die daraus folgende Vergrößerung der Schritte.

<!--en:fcba4800-->
![Bias-Korrektur bei Adam. Links erfasst der bei null gestartete gleitende Mittelwert $\mathbf{v}_t$ nur den Anteil $1-\beta_2^t$ der Größenordnung der quadrierten Gradienten ($63\%$ bei $t=1000$ für $\beta_2=0.999$). Die Division durch $1-\beta_2^t$ stellt diese Größenordnung in jedem Schritt wieder her. Rechts vergrößert sich die Aktualisierung $\mathbf{m}_t/\sqrt{\mathbf{v}_t}$ ohne Bias-Korrektur um den Faktor $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$. Er beginnt bei $3.16$ für $t=1$ und erreicht bei einem Wert von $6.6$ für $t=12$ sein Maximum, weil sich der Zähler in zehn Schritten aufbaut, der Nenner aber tausend braucht. Zu $1$ kehrt er erst auf der Zeitskala $1/(1-\beta_2)$ zurück. Adam macht ohne Korrektur also gerade dann die größten Schritte, wenn die wenigsten Gradienten in seinen Vorkonditionierer eingegangen sind.](../img/mdl-opt-bias-correction.svg)

<!--en:c1db5e82-->
Selbst ein erwartungstreues $\hat{\mathbf{v}}_t$ macht die Schrittweite selbst noch nicht erwartungstreu. Adam teilt durch $\sqrt{\hat{\mathbf{v}}_t}$ statt durch $\hat{\mathbf{v}}_t$. Die Konkavität der Quadratwurzel verzerrt den Nenner daher nach unten. Nach der Jensen-Ungleichung gilt $\mathbb{E}\sqrt{\hat{v}_t} < \sqrt{\mathbb{E}[\hat{v}_t]}$, sofern $\hat{v}_t$ nicht konstant ist. Der erste Schritt ist der Extremfall. Für $t = 1$ gilt $\hat{m}_1 = g_1$ und $\sqrt{\hat{v}_1} = |g_1|$. Der Quotient ist also $\mathrm{sign}(g_1)$, und der Schritt hat in jeder Koordinate unabhängig vom Gradienten den Betrag $\eta$ anstelle von $\eta\,|g_1|/\sigma$ für einen Gradienten der Größenordnung $\sigma$. Wie stark sich das auswirkt, untersuchen wir in :numref:`subsec_mdl-schedules-warmup`.

<!--en:b651f2b7-->
### Skaleninvarianz

<!--en:a4db9f9c-->
In unserem Einstiegsbeispiel gilt $h_2 = 1000\, h_1$. Dennoch durchquert Adam das Tal, als gäbe es diesen Faktor tausend nicht. Der Grund ist eine exakte Symmetrie der Aktualisierung. Sie ist das zentrale Ergebnis dieses Abschnitts.

<!--en:cc559ffa-->
**Proposition (Adam ist unempfindlich gegenüber der Skalierung jeder einzelnen Koordinate).** *Es sei $\epsilon = 0$ und es gelte $\mathbf{C} = \mathrm{diag}(c_1, \ldots, c_d)$ mit $c_i > 0$ für alle Koordinaten. Die Gradienten zweier Probleme seien durch $\nabla \tilde{f}(\mathbf{x}) = \mathbf{C}\, \nabla f(\mathbf{x})$ für jedes $\mathbf{x}$ verknüpft. Dann liefert Adam bei gleichem $\beta_1$, gleichem $\beta_2$, gleichem Schrittweitenplan und gleichem Startpunkt für beide Probleme identische Iterierte, solange die Aktualisierungen definiert bleiben.*

<!--en:2fc71354-->
Im zweiten Problem ist also der Gradient jeder Koordinate $i$ gegenüber dem ersten Problem mit einer positiven Konstanten $c_i$ multipliziert. Seine Größen kennzeichnen wir mit einer Tilde.

<!--en:e02a93e7-->
**Beweis.** Wir führen eine Induktion über $t$. Die Iterierten mögen bis $\mathbf{x}_t$ übereinstimmen. Dann gilt $\tilde{\mathbf{g}}_\tau = \mathbf{C}\mathbf{g}_\tau$ für jedes $\tau \le t$. Da die Rekursionen für $\mathbf{m}$ und $\mathbf{v}$ linear in $\mathbf{g}_\tau$ beziehungsweise $\mathbf{g}_\tau^2$ sind, erhalten wir $\tilde{m}_{t,i} = c_i m_{t,i}$ und $\tilde{v}_{t,i} = c_i^2 v_{t,i}$. Die Nenner der Bias-Korrektur sind in beiden Problemen dieselben Skalare. Damit ergibt der korrigierte Quotient $\hat{m}_{t,i}/\sqrt{\hat{v}_{t,i}}$ des zweiten Problems $c_i \hat{m}_{t,i}/(c_i \sqrt{\hat{v}_{t,i}}) = \hat{m}_{t,i}/\sqrt{\hat{v}_{t,i}}$. Die Aktualisierungen stimmen also überein, und es folgt $\tilde{\mathbf{x}}_{t+1} = \mathbf{x}_{t+1}$. $\blacksquare$

<!--en:8b8b3462-->
Bei der diagonalen quadratischen Funktion ist $\nabla f(\mathbf{x}) = \mathrm{diag}(h)\,\mathbf{x}$ einfach der mit den jeweiligen Faktoren skalierte Gradient der rotationssymmetrischen Schale $\tfrac12\|\mathbf{x}\|^2$. Diese Faktoren sind $\mathbf{C} = \mathrm{diag}(h)$. Daher beschreibt Adam im Tal mit $1000{:}1$ Schritt für Schritt genau dieselbe Trajektorie wie auf der rotationssymmetrischen Schale. Dort bleiben zwei Koordinaten mit gleichem Startwert stets gleich. Jede Koordinate folgt derselben eindimensionalen Iteration für $\tfrac12 x^2$,

<!--en:b56dc9eb-->
wobei $\hat{v}_t$ die bisherigen Quadrate $x_\tau^2$ für $\tau \le t$ mittelt. Da die früheren Iterierten größer waren, übersteigt $\sqrt{\hat{v}_t}$ ab dem zweiten Schritt $|x_t|$. Jede Aktualisierung legt daher weniger als $\eta$ zurück und bewirkt eine geometrische Kontraktion um den Faktor $1 - \eta/\sqrt{\hat{v}_t}$. Für $\eta = 0.1$ sinkt dieser Faktor von $0.9$ bei $t = 1$ auf $0.67$ bei $t = 48$, während sich $\hat{v}_t$ den kleiner werdenden Iterierten annähert. Die Iterierten selbst sind völlig unabhängig von $h_2/h_1$. Nur das Abbruchkriterium hängt vom Krümmungsverhältnis ab, weil $f(\mathbf{x}_t) = \tfrac12(h_1 + h_2)\, x_t^2$ die gemeinsame Iterierte mit den Krümmungen gewichtet. Selbst ein wesentlich schmaleres Tal sollte daher nur wenige zusätzliche geometrische Kontraktionsschritte erfordern und die Schrittzahl keinesfalls um den Faktor $\kappa$ erhöhen.

<!--en:1e4bfbe8-->
Wovon die Schrittzahl *tatsächlich* abhängt, ist der anfängliche Abstand in Einheiten von $\eta$, denn die ersten Schritte haben unabhängig von der Krümmung je Koordinate ungefähr die Länge $\eta$. Ein Experiment im nächsten Unterabschnitt prüft beide Vorhersagen direkt.

<!--en:b77b3ded-->
Adam misst also jede Koordinate in Einheiten ihres eigenen Gradientenbetrags. Damit bleibt eine koordinatenweise Skalierung für das Verfahren unsichtbar. Diese Invarianz hat drei Grenzen. Erstens bricht $\epsilon > 0$ die Symmetrie, sobald der Gradient einer Koordinate auf die Größenordnung von $\epsilon$ sinkt. Zweitens sagt die Proposition nichts über einen Basiswechsel aus, der die Koordinaten mischt. Das prüfen wir als Nächstes am gedrehten Tal. Drittens folgt Adam im Allgemeinen nicht nur dem Vorzeichen des Gradienten. Seine Aktualisierung ergibt nur in drei Sonderfällen den **Vorzeichenschritt** $\eta\,\mathrm{sign}(\mathbf{g}_t)$: bei $t = 1$, für $\beta_1 = \beta_2 = 0$ (Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`d) und bei $\beta_1 = 0$ immer dann, wenn der Gradientenbetrag jeder Koordinate über den Zeitraum konstant bleibt, den $\hat{\mathbf{v}}$ berücksichtigt.

<!--en:599b401d-->
## Koordinatenweise Skalierung und ihre Grenzen

<!--en:19e92af2-->
### Achsenparallele und gedrehte Täler

<!--en:5b375ea8-->
Wie viele der 6160 Schritte des Gradientenabstiegs spart die koordinatenweise Skalierung allein ein, und bleibt der Gewinn erhalten, wenn wir das Tal drehen? Um die Wirkung der Skalierung von der des Momentums zu unterscheiden, vergleichen wir vier Verfahren: den Gradientenabstieg mit optimaler gemeinsamer Schrittweite, das Heavy-Ball-Verfahren mit optimaler Schrittweite und optimalem Momentum, Adam mit $\beta_1 = 0$, das nur skaliert, und Adam mit dem Standard-Momentum $\beta_1 = 0.9$. Für eine Drehung um den Winkel $\theta$ ersetzen wir $\mathrm{diag}(h)$ durch $\mathbf{A}_\theta = \mathbf{R}_\theta\, \mathrm{diag}(h)\, \mathbf{R}_\theta^\top$, wobei $\mathbf{R}_\theta$ die $2 \times 2$-Drehmatrix ist. Wir drehen auch den Startpunkt mit, $\mathbf{x}_0 = \mathbf{R}_\theta (1, 1)^\top$, damit seine Komponenten entlang beider Krümmungsachsen gleich bleiben. Für $\theta = \pi/4$ ergibt sich $\mathbf{A} = \left(\begin{smallmatrix}500.5 & -499.5\\ -499.5 & 500.5\end{smallmatrix}\right)$. Der folgende Code definiert das Tal, eine Schleife, die die Schritte bis zum Erreichen der Zieltoleranz zählt, und die drei Aktualisierungsregeln.

<!--en:4908b919-->
Bevor wir die Schrittzahlen betrachten, lohnt sich ein Blick auf die Informationen, die jedes Verfahren erhält. Für Gradientenabstieg und Heavy-Ball-Verfahren berechnen wir die theoretisch optimalen Einstellungen in geschlossener Form aus den tatsächlichen Krümmungen $1$ und $1000$. Das ist ein großzügiger Vorsprung, denn in der Praxis kennt niemand die extremalen Krümmungen eines neuronalen Netzes. Für Adam gibt es keine geschlossene Formel. Wir gehen daher wie in der Praxis vor und testen sieben logarithmisch verteilte Schrittweiten von $0.002$ bis $2$ (jeweils etwa um den Faktor drei auseinander). Pro Zeile betrachten wir den besten Eintrag. Der Wert $\eta = 1$ fehlt absichtlich im Raster. Adams erster Schritt hat in jeder Koordinate die Länge $\eta$ (:numref:`subsec_mdl-schedules-warmup`). Vom Startpunkt $(1, 1)$ aus würden wir mit $\eta = 1$ in einem Schritt genau im Ursprung landen. Das läge an der zufälligen Wahl des Startpunkts und wäre kein fairer Maßstab für die Geschwindigkeit des Optimierers.

<!--en:3e6314ec-->
Betrachten wir zunächst das achsenparallele Tal. Die koordinatenweise Skalierung allein ($\beta_1 = 0$) braucht bei ihrer besten Schrittweite nur 22 Schritte, der Gradientenabstieg 6160 und das Heavy-Ball-Verfahren 296. Der Optimierer ohne Kenntnis der beiden Krümmungen braucht also nur ein Dreizehntel der Schritte des Heavy-Ball-Verfahrens, dem beide bekannt sind. Diese Zeile zeigt auch, was wir mit Adams Schrittweite tatsächlich einstellen. Die Schrittzahl skaliert ungefähr wie $1/\eta$, denn die ersten Schritte haben etwa die Länge $\eta$. Entscheidend ist somit vor allem der Abstand zum Minimierer, gemessen in Schrittlängen. Bei den beiden größten Schrittweiten konvergiert die Skalierung allein jedoch nie. Die Iterierte schießt über das Minimum hinaus und gerät in einen Zyklus, in dem jede Koordinate zwischen $+\eta/2$ und $-\eta/2$ pendelt. Adam hat also eine Unbekannte, die Krümmung, gegen eine andere eingetauscht: den Abstand zum Minimierer in Einheiten von $\eta$.

<!--en:65d8a2e1-->
Mit Adams Standard-Momentum ($\beta_1 = 0.9$) liegen die Schrittzahlen im gesamten Schrittweitenbereich von $0.02$ bis $2$ zwischen 165 und 205. Zweierlei fällt auf. Erstens ist das Verfahren robust. Eine Änderung von $\eta$ um den Faktor hundert ändert die Schrittzahl nur um ein Viertel. Auch bei den beiden größten Schrittweiten, die ohne Momentum zu endlosen Zyklen führten, konvergiert es nun, weil die Mittelung im Momentum die Vorzeichenwechsel dämpft. Das Heavy-Ball-Verfahren hat keinen solchen Spielraum. Seine optimale Schrittweite liegt weniger als $0.1\%$ von der Stabilitätsgrenze $(2 + 2\beta)/\lambda_{\max}$ entfernt (Aufgabe :exref:`ex_mdl-gradient-based-optimization-edge-of-stability-momentum`a in :numref:`sec_mdl-gradient-based-optimization`). Das Ergebnis von 296 Schritten setzt also voraus, dass wir die größte Krümmung fast exakt kennen. Zweitens hat die Robustheit ihren Preis. Mit Momentum braucht Adam etwa siebenmal so lange wie mit Skalierung allein. Der Grund ist die Kontraktionsgrenze $\sqrt\beta$ aus :numref:`subsec_mdl-momentum-acceleration`. Eine Iteration mit Momentum $\beta$ kontrahiert jede Eigenkomponente mit komplexen Wurzeln pro Schritt mit der Rate $\sqrt\beta$ und nicht schneller. Nach der koordinatenweisen Skalierung bewegen sich beide Koordinaten effektiv auf einer runden Schale mit Konditionszahl eins, für die das optimale Momentum null ist. Bei $\beta_1 = 0.9$ beträgt der Kontraktionsfaktor $\sqrt{0.9} \approx 0.95$ pro Schritt. Das lässt bis zum Erreichen der Toleranz etwa 230 Schritte erwarten, was gut mit den beobachteten Werten übereinstimmt. Ein Momentum, das in einem schlecht konditionierten Tal den Abstieg beschleunigt, bremst ihn auf einer runden Schale. Adam setzt es in der Standardeinstellung dennoch immer ein.

<!--en:7bd57a4e-->
Wenden wir uns dem gedrehten Tal zu. Gradientenabstieg und Heavy-Ball-Verfahren brauchen genau so viele Schritte wie zuvor, denn ihre Aktualisierungen drehen sich mit dem Problem mit. Bei Adam ist das anders. Die Skalierung allein verfehlt die Toleranz bei jeder einzelnen Schrittweite des Rasters. Adam mit Momentum konvergiert zwar, aber erst am oberen Ende des Rasters zügig. Im besten Fall braucht es 559 Schritte bei $\eta = 2$, fast doppelt so viele wie die 296 des Heavy-Ball-Verfahrens. Bei der kleinsten Schrittweite sind es 12 862 Schritte, doppelt so viele wie beim einfachen Gradientenabstieg. Auch die Robustheit gegenüber der Wahl der Schrittweite geht verloren. Die Schrittzahlen im Raster unterscheiden sich jetzt um mehr als den Faktor zwanzig.

<!--en:6ef861f0-->
Ein fairer Vergleich ergibt also ein zweiteiliges Bild. Stimmen die Koordinatenachsen mit den Hauptkrümmungsrichtungen überein, schlägt die koordinatenweise Skalierung ein optimal eingestelltes Heavy-Ball-Verfahren um eine Größenordnung und benötigt dafür wesentlich weniger Vorwissen. Drehen wir das Tal, verschwindet der Vorteil der diagonalen Skalierung vollständig. Die verbleibende Konvergenz verdankt Adam dem Momentum, und mit gut eingestellten Parametern ist das Heavy-Ball-Verfahren dabei doppelt so schnell. Sehen wir uns beide Fälle genauer an.

<!--en:acd76d3c-->
Für das achsenparallele Tal sagt unsere Proposition zur Skaleninvarianz identische Iterierte für beide Koordinaten voraus, obwohl die steife Koordinate eine tausendmal kleinere effektive Schrittweite erhält. Im Folgenden geben wir die Iterierten und die effektiven Schrittweiten $\eta/\sqrt{\hat{\mathbf{v}}_t}$ für Adam mit reiner Skalierung ($\epsilon = 0$, $\eta = 0.1$) aus. Mit den letzten beiden Zeilen prüfen wir, wovon die Schrittzahl tatsächlich abhängt. Dazu variieren wir das Krümmungsverhältnis über fünf Größenordnungen und verlegen den Startpunkt hundertmal so weit vom Ursprung weg.

<!--en:99fcdd29-->
Schon ab dem ersten Schritt stehen die effektiven Schrittweiten exakt im Verhältnis $1000{:}1$. Das tatsächliche Krümmungsverhältnis ergibt sich allein aus den Gradientenbeträgen, ohne je eine Hesse-Matrix zu bilden. Beide Koordinaten stimmen durchgehend bis auf Gleitkommagenauigkeit überein. Die effektive Schrittweite in der flachen Koordinate wächst von $0.1$ auf $0.33$, während sich $\hat{v}_t$ den kleiner werdenden Gradienten anpasst. Erhöhen wir das Krümmungsverhältnis von $1$ über $10^3$ auf $10^5$, steigt die Schrittzahl lediglich von 40 über 48 auf 54. Eine um den Faktor hunderttausend schlechtere Kondition kostet also nur vierzehn zusätzliche Schritte. Vergrößern wir dagegen den anfänglichen Abstand um den Faktor hundert, steigt die Schrittzahl auf 4145. Sie misst den anfänglichen Abstand in Einheiten von $\eta$ und hängt kaum von der Krümmung ab.

<!--en:41e7682b-->
Warum macht eine Drehung um $45^\circ$ all das zunichte? Die Diagonaleinträge der gedrehten Matrix liegen nahe $500$, und ihre steile Richtung $(-1, 1)/\sqrt{2}$ projiziert sich zu gleichen Teilen auf beide Koordinatenachsen. Die steile Eigenkomponente dominiert also beide Gradientenkomponenten. Daher ergeben sich für beide Koordinaten fast identische Skalenschätzungen (ihre $\hat{v}$-Werte stimmen in den Läufen bis auf ein Prozent überein). Ein koordinatenweiser Vorkonditionierer weist jeder Koordinate nur einen Skalar zu. Er kann daher den steilen Anteil ihres Gradienten nur zusammen mit dem flachen Anteil verkleinern. Bei gleicher Skalierung beider Koordinaten reduziert sich der Vorkonditionierer auf ein skalares Vielfaches der Einheitsmatrix. Reine Skalierung mit Adam ($\beta_1 = 0$) entspricht dann wieder einem Gradientenabstieg, dessen Schrittweite die steile Eigenkomponente vorgibt. Die flache Eigenkomponente kommt kaum voran. Adam mit $\beta_1 = 0.9$ verhält sich wieder wie das Heavy-Ball-Verfahren. Sein Momentum liegt nahe am Optimum $0.88$, doch die effektive Schrittweite $\eta/\sqrt{\hat{v}_t}$ richtet sich nach den bisherigen Gradienten und nicht nach der Krümmung. Selbst am besten Rasterpunkt braucht das Verfahren daher doppelt so lange wie das optimal eingestellte Heavy-Ball-Verfahren.

<!--en:311747ef-->
Ohne Momentum endet der Lauf im gedrehten Tal in einer Schwingung. Während die steile Komponente anfangs schrumpft, nimmt $\hat{\mathbf{v}}$ ab und die effektive Schrittweite wächst, bis sie die Stabilitätsgrenze $2/h_{\max}$ der steilen Eigenkomponente überschreitet. Bei $\eta = 0.2$, der optimalen Schrittweite für das achsenparallele Tal, sinkt der Verlust auf ungefähr $10^{-3}$ um den Zeitpunkt $t = 1{,}700$. Dann beginnt die steile Eigenkomponente zu schwingen. Die anhaltende Schwingung hält den Gradientenbetrag konstant, sodass $\hat{\mathbf{v}}_t \to \mathbf{g}_t^2$ gilt. Damit reduziert sich die Aktualisierung auf den Vorzeichenschritt $\eta\,\mathrm{sign}(\mathbf{g}_t)$ der Länge $\eta\sqrt{2}$ quer zum Tal. Die Iterierte bleibt für immer in einem Zweierzyklus mit Amplitude $\eta/\sqrt{2}$ und Verlust $f = 250\,\eta^2 = 10$ gefangen (Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity` überprüft die Formel für das gesamte Schrittweitenraster). Momentum mittelt die gegenläufigen Schritte aus. Daher erreichen die Läufe mit $\beta_1 = 0.9$ schließlich die Zieltoleranz. :numref:`fig_mdl-opt-per-coordinate` zeigt alle drei Verhaltensweisen an einem Tal, das für die Darstellung flach genug ist.

<!--en:2ce1db85-->
![Vergleich in einem zweidimensionalen Tal ($\kappa=20$, im Text gilt $\kappa=10^3$ mit zu schmalen Höhenlinien für die Darstellung) bei gleichem Startpunkt. (a) Der Gradientenabstieg mit gemeinsamer Schrittweite nahe der Stabilitätsgrenze oszilliert in der steilen Richtung und kommt in der flachen Richtung kaum voran. (b) Die koordinatenweise Normierung (Adam mit $\beta_1=0$) bewegt beide Koordinaten gleich schnell und nähert sich dem Minimum auf einer glatten Kurve. (c) Nach einer Drehung von Tal und Startpunkt um $45^\circ$ stimmen die Koordinatenachsen nicht mehr mit den Krümmungsrichtungen überein. Die Iterierte kommt auf der Talsohle nur langsam voran ($f=0.19$ nach 40 Schritten gegenüber $4\times10^{-7}$ in (b)).](../img/mdl-opt-per-coordinate.svg)

<!--en:310c0576-->
**Koordinatenweise Skalierung beseitigt die schlechte Kondition genau dann, wenn die Koordinatenachsen mit den Eigenvektoren der Krümmungsmatrix übereinstimmen.** Die Skalierung ändert die Metrik durch eine Diagonalmatrix. Keine Diagonalmatrix kann die Krümmung in Richtungen erfassen, die mehrere Koordinaten mischen. Eine volle Matrix aus äußeren Produkten $\mathbf{g}_\tau\mathbf{g}_\tau^\top$ dreht sich mit dem Problem mit und vermeidet diese Schwäche. Für große neuronale Netze wäre es jedoch viel zu aufwendig, sie zu speichern und zu invertieren. Bei Gewichtsmatrizen begegnen die strukturierten Vorkonditionierer am Ende dieses Abschnitts dieser Einschränkung mit kompakten Faktoren, die Koordinaten mischen.

<!--en:2a362872-->
### Ein konvexes Gegenbeispiel für Adam

<!--en:26a8ad8c-->
RMSProp und Adam vergessen frühere Gradienten, während AdaGrad sie unbegrenzt berücksichtigt. Kann dieses Vergessen selbst ein einfaches konvexes Problem unlösbar machen? :citet:`Reddi.Kale.Kumar.2019` zeigen anhand einer eindimensionalen periodischen Folge konvexer Verlustfunktionen, dass das möglich ist.

<!--en:7ab5b8c6-->
**Proposition (Adam konvergiert nicht zwangsläufig).** *Es sei $C > 2$. Betrachten wir die Online-Optimierung auf $x \in [-1, 1]$ mit den periodischen Verlustfunktionen*

<!--en:218e7b84-->
*Wir verwenden Adam ohne Bias-Korrektur mit $\beta_1 = 0$, $\beta_2 = 1/(1 + C^2)$ und Schrittweiten $\eta_t = \eta/\sqrt{t}$, für die $\eta < \sqrt{1 - \beta_2}$ gilt. Jede Iterierte projizieren wir zurück auf $[-1, 1]$, der Startwert sei $x_1 = 1$. Dann gilt $x_t > 0$ für jedes $t$ und $x_{3k+1} = 1$ für jedes $k \ge 0$. Jede Periode aus drei Schritten erhöht den Regret um mindestens $2C - 4 > 0$. Der durchschnittliche Regret geht also nicht gegen null.* :cite:`Reddi.Kale.Kumar.2019`

<!--en:0e799958-->
Die Projektion auf $[-1, 1]$ beschränkt hier einfach den Wert auf das Intervall (:numref:`sec_mdl-constrained-optimization-duality` behandelt Projektionen allgemein). Da sich die Gradienten in jeder Periode aus drei Schritten zu $C - 2 > 0$ summieren, ist $x^\star = -1$ im Nachhinein der beste fest gewählte Punkt. Dennoch kehrt Adam zu Beginn jeder Periode zu $x = +1$ zurück, dem schlechtesten Punkt des Intervalls. Jede Periode kostet Adam mindestens $C - 2$ (im ersten Schritt entstehen Kosten von $C$ bei $x = 1$, in jedem der beiden anderen mindestens $-1$). Dagegen kostet die Wahl von $x^\star = -1$ nur $-(C - 2)$, sodass der Regret pro Periode um mindestens $2C - 4$ steigt. Der Beweis führt eine Induktion über die Perioden. Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-reddi-by-hand` behandelt den entscheidenden Schritt, in dem $\eta < \sqrt{1 - \beta_2}$ sicherstellt, dass der Abwärtsschritt in $(0, 1]$ bleibt. Das Scheitern liegt auch nicht allein an der Wahl $\beta_1 = 0$. Dieselbe Arbeit konstruiert für jedes feste Paar mit $\beta_1 < \sqrt{\beta_2}$ sowohl eine konvexe Online-Folge als auch ein stochastisches konvexes Problem, bei denen Adam nicht konvergiert.

<!--en:cfeb61e0-->
Den Mechanismus können wir direkt an einer einzelnen Periode mit $C = 4$ und $\beta_2 = 1/17$ ablesen, sobald $v$ seinen stationären periodischen Verlauf erreicht hat. Bei der Gradientenfolge $+4, -1, -1$ nimmt die Schätzung des zweiten Moments $v$ die Werte $15.12$, $1.83$ und $1.05$ an. Für $\sqrt{v}$ ergeben sich also $3.89$, $1.35$ und $1.02$. Die drei Verschiebungen $-\eta g_t/\sqrt{v_t}$ lauten

<!--en:e2f22645-->
und ergeben pro Periode eine Nettoverschiebung um $+0.69\,\eta$ in Richtung $+1$. Die Normierung begrenzt die Wirkung des seltenen großen Gradienten $+C$ auf etwa $\eta$, kaum mehr als bei einem Gradienten vom Betrag $1$. Außerdem vergisst $v$ die Spitze $C^2$ innerhalb zweier Schritte. Zusammen schieben die beiden Gradienten $-1$ daher $x$ weiter nach oben, als der einzelne Gradient $+C$ den Wert nach unten zieht. Einfacher SGD mit Schrittweite $\eta g_t/C$ bewegt sich dagegen um $-\eta$, $+0.25\,\eta$ und $+0.25\,\eta$, insgesamt also um $-0.5\,\eta$ in Richtung des Optimums. Anders als der durch das Rauschen bedingte Restfehler bei SGD mit konstanter Schrittweite (:numref:`sec_mdl-stochastic-gradient-methods`) verschwindet Adams Drift in die falsche Richtung auch bei abnehmender Schrittweite nicht. Sie zeigt bei jeder Schrittweite zu $+1$.

<!--en:c89fcc42-->
Als Abhilfe schlagen :citet:`Reddi.Kale.Kumar.2019` **AMSGrad** vor. Es ersetzt $\mathbf{v}_t$ im Nenner durch das laufende Maximum $\tilde{\mathbf{v}}_t = \max(\tilde{\mathbf{v}}_{t-1}, \mathbf{v}_t)$ des unkorrigierten Mittelwerts. Die effektive Schrittweite $\eta_t/\sqrt{\tilde{v}_t}$ kann dann nicht mehr zunehmen. Ein großer Gradient gerät also nie in Vergessenheit. In unserem periodischen Beispiel bleibt $\tilde{v}$ durchgehend bei $15.12$. Die drei Verschiebungen betragen nun $-1.03\,\eta$, $+0.26\,\eta$ und $+0.26\,\eta$, insgesamt also $-0.51\,\eta$ in Richtung des Optimums. Adam sollte demnach gegen $+1$ konvergieren, AMSGrad und SGD dagegen gegen $-1$. Die folgende Codezelle testet alle drei mit derselben Abnahme gemäß $1/\sqrt{t}$. Wir starten bei $x = 0$, der Intervallmitte, sodass Adam nicht einmal am ungünstigen Endpunkt beginnt.

<!--en:9b327bb1-->
Hier übernimmt `alpha` die Rolle von $\eta$ in der Proposition. Wegen $\eta = 0.5 < \sqrt{16/17}$ ist die Bedingung an die Schrittweite erfüllt. Selbst von der Intervallmitte aus läuft Adam geradewegs zu $+1$. Ab $t \approx 300$ kehrt es in jeder Periode zu $+1$ zurück und fällt nie unter $0.97$. AMSGrad und SGD nähern sich dagegen stetig dem Optimum $-1$ und stimmen auf zwei Dezimalstellen überein. Das Gegenbeispiel beweist, dass Adam für konvexe Probleme keine allgemeine Konvergenzgarantie besitzt. Es ist jedoch ein Existenzbeweis und sagt kein Scheitern bei typischen Deep-Learning-Problemen voraus.

<!--en:23148934-->
## Entkoppelter Gewichtszerfall

<!--en:b3edf4e9-->
### Gekoppelter und entkoppelter Gewichtszerfall

<!--en:724c667f-->
Fast jedes moderne Trainingsverfahren verwendet Gewichtszerfall. Bei einem adaptiven Optimierer stellt sich damit die Frage, ob der Zerfallsterm den Vorkonditionierer durchlaufen soll. Bei SGD gibt es diesen Unterschied nicht. Addieren wir einen $\ell_2$-Strafterm $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ zum Verlust, kommt zum Gradienten $\lambda \mathbf{w}$ hinzu. Die Aktualisierung $\mathbf{w}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{w}_t) = (1 - \eta\lambda)\, \mathbf{w}_t - \eta\, \mathbf{g}_t$ ist zugleich ein Gradientenschritt für den Verlust und eine gleichmäßige Schrumpfung der Gewichte (in :numref:`sec_adamw` steht $\mathbf{x}$ für die Gewichte). Bei Adam fallen diese beiden Mechanismen jedoch auseinander. Setzen wir den Gradienten einschließlich Strafterm in :eqref:`eq_mdl-opt-adam` ein und isolieren den Zerfallsterm (der Übersicht halber mit $\beta_1 = 0$, sodass die Aktualisierung $-\eta\,(\mathbf{g}_t + \lambda\mathbf{w}_t)/(\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ lautet), dann ergibt sich für die Schrumpfung der Koordinate $i$ pro Schritt

<!--en:de0b11f2-->
Bei gekoppeltem Adam teilt der Vorkonditionierer sowohl den Gradienten des Verlusts als auch den des Strafterms durch denselben Wert. Die Schrumpfungsrate einer Koordinate ist daher umgekehrt proportional zur Größenordnung ihres Gradienten. Gewichte mit großen Gradienten schrumpfen kaum, solche mit kleinen Gradienten dagegen stark. **AdamW** :cite:`Loshchilov.Hutter.2019` entkoppelt die beiden Operationen. Nur der Verlustgradient durchläuft den Vorkonditionierer, während der Gewichtszerfall direkt auf die Parameter wirkt:

<!--en:302a6c6f-->
Wie bei SGD schrumpft jede Koordinate nun pro Schritt um denselben Faktor $1 - \eta\lambda$, und in die gleitenden Momente geht nur der Verlustgradient ein.

<!--en:56064a30-->
`torch.optim.Adam(weight_decay=...)` addiert $\lambda\mathbf{w}$ zum Gradienten und implementiert damit die gekoppelte Regel. `torch.optim.AdamW` implementiert die entkoppelte Regel :eqref:`eq_mdl-opt-adamw-decoupled`.

<!--en:12fe1b77-->
`mx.optimizer.Adam(wd=...)` addiert $\lambda\mathbf{w}$ zum Gradienten und implementiert damit die gekoppelte Regel. `mx.optimizer.AdamW` wendet den Gewichtszerfall außerhalb des Vorkonditionierers an.

<!--en:dcfaee23-->
In Keras ist `weight_decay` ein Argument jedes Optimierers. Der Gewichtszerfall wirkt außerhalb des Vorkonditionierers auf die Gewichte. Daher implementieren `keras.optimizers.Adam(weight_decay=...)` und `keras.optimizers.AdamW` beide die entkoppelte Regel. Für die gekoppelte Regel müssen wir der Verlustfunktion explizit einen $\ell_2$-Strafterm hinzufügen.

<!--en:b8ef462e-->
`optax.adam` hat kein Argument für den Gewichtszerfall. `optax.adamw` implementiert die entkoppelte Regel. Für die gekoppelte Regel brauchen wir einen expliziten Strafterm in der Verlustfunktion oder `optax.chain(optax.add_decayed_weights(lam), optax.adam(eta))`. Letzteres addiert $\lambda\mathbf{w}$ zum Gradienten, bevor Adam ihn erhält.

<!--en:20363763-->
Am deutlichsten lässt sich :eqref:`eq_mdl-opt-adamw` an einem Gewicht prüfen, dessen Verlustgradient aus reinem Rauschen mit Mittelwert null besteht. Nur der Gewichtszerfall bewirkt dann eine systematische Drift. Hat das Rauschen die Standardabweichung $\sigma$, so gilt $\sqrt{\hat{v}} \approx \sigma$. Wir erwarten daher, dass gekoppeltes Adam das Gewicht auf ungefähr $e^{-\eta\lambda T/\sigma}$ verkleinert, wenn $T$ Schritte vergangen sind. AdamW sollte dagegen jedes Gewicht auf $(1 - \eta\lambda)^T$ verkleinern, unabhängig von $\sigma$. Im Folgenden initialisieren wir $10^4$ Gewichte mit Rauschskala $\sigma = 10$ und weitere $10^4$ mit $\sigma = 0.1$, alle beim Wert $w = 1$. Wir führen für jedes Verfahren 4000 Schritte mit $\eta = 10^{-3}$ und $\lambda = 0.1$ aus und vergleichen die Gruppenmittelwerte mit den Vorhersagen.

<!--en:d367a808-->
Bei ein und demselben $\lambda$ ergeben sich für die beiden Gruppen gekoppelte Zerfallsraten von $10^{-5}$ und $10^{-3}$ pro Schritt. Sie unterscheiden sich um den Faktor hundert. Wie vorhergesagt behalten die stark verrauschten Gewichte $0.96$ ihres anfänglichen Betrags. Die schwach verrauschten Gewichte sinken auf etwa $0.03$ und liegen damit etwas über dem Schätzwert $0.018$. Der etwas schwächere Zerfall hat einen einfachen Grund. In der Nähe von $w = 1$ ist der Gradient des Strafterms $\lambda w = 0.1$ bei einem schwach verrauschten Gewicht ähnlich groß wie das Rauschen. Er erhöht damit sein eigenes $\hat{v}$ und bremst so den anfänglichen Zerfall. Bei AdamW enden dagegen beide Gruppen genau bei $0.67 = (1 - \eta\lambda)^{4000}$. Für AdamW gilt die Vorhersage im Erwartungswert exakt, weil die normierten Rauschschritte den Mittelwert null haben und der multiplikative Zerfall in jedem Schritt gleich ist.

<!--en:fffcc2e8-->
### Fixpunkte

<!--en:31a51a35-->
Das Experiment mit reinem Rauschen hat keinen stationären Punkt, weil sich seine Gradienten nie stabilisieren. Bei exakten Gradienten kann ein Optimierer jedoch zur Ruhe kommen. An einem Fixpunkt hat $\lambda$ eine überraschend andere Bedeutung.

<!--en:a83d47ae-->
**Proposition (Fixpunkte bei gekoppeltem und entkoppeltem Gewichtszerfall).** *Der Gradient $\mathbf{g}(\mathbf{w}) = \nabla\ell(\mathbf{w})$ einer Verlustfunktion $\ell$ sei exakt (rauschfrei), und es gelte $\epsilon > 0$. Wir nennen $\mathbf{w}$ einen Fixpunkt einer Aktualisierungsregel, wenn die Aktualisierung dort verschwindet, sobald sich die Momentschätzungen stabilisiert haben. (a) Adam hat für $\ell + \tfrac{\lambda}{2}\|\mathbf{w}\|^2$ genau dann einen Fixpunkt bei $\mathbf{w}$, wenn $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ gilt. Das ist derselbe stationäre Punkt wie bei SGD für die Verlustfunktion mit Strafterm. (b) AdamW hat genau dann einen Fixpunkt bei $\mathbf{w}$, wenn für jede Koordinate $i$ gilt:*

<!--en:b4d018d0-->
*Damit liegt jeder Fixpunkt von AdamW strikt im Inneren des Quaders $|w_i| < 1/\lambda$. Für $\epsilon \to 0$ ergeben sich zwei Möglichkeiten. Entweder gilt $g_i = 0$, oder es gilt $|w_i| = 1/\lambda$, wobei $w_i$ und $g_i$ entgegengesetzte Vorzeichen haben.*

<!--en:0dbecf85-->
**Beweis.** An einem Fixpunkt ist der Vektor $\mathbf{u}$, der in die gleitenden Momente eingeht, konstant. Somit gilt $\hat{\mathbf{m}}_t \to \mathbf{u}$ und $\hat{\mathbf{v}}_t \to \mathbf{u}^2$. (a) Für gekoppeltes Adam gilt $\mathbf{u} = \mathbf{g} + \lambda\mathbf{w}$, und die Aktualisierung der Koordinate $i$ strebt gegen $-\eta\, u_i/(|u_i| + \epsilon)$. Dieser Ausdruck verschwindet genau dann, wenn $u_i = 0$ gilt. (b) Für AdamW gilt $\mathbf{u} = \mathbf{g}$. Die Aktualisierung der Koordinate $i$ strebt also gegen $-\eta\lambda w_i - \eta\, g_i/(|g_i| + \epsilon)$. Setzen wir diesen Ausdruck gleich null und multiplizieren beide Seiten mit $(|g_i| + \epsilon)/\eta$, erhalten wir :eqref:`eq_mdl-opt-adamw-fixed-point`. Bilden wir die Beträge, folgt $|w_i| = |g_i|/\bigl(\lambda(|g_i| + \epsilon)\bigr) < 1/\lambda$. Für $\epsilon \to 0$ wird die Bedingung zu $\lambda w_i |g_i| = -g_i$. Das gilt genau dann, wenn $g_i = 0$ oder $\lambda w_i = -\mathrm{sign}(g_i)$ gilt. $\blacksquare$

<!--en:05dcd101-->
Die Proposition charakterisiert Fixpunkte. Sie garantiert weder die Existenz eines Fixpunkts noch die Konvergenz der Iterierten dorthin. Auch die Fallunterscheidung ist nicht nur im Grenzwert interessant. Wegen $1 - \lambda|w_i| = \epsilon/(|g_i| + \epsilon)$ gilt bei $\epsilon = 10^{-8}$ für jede Koordinate: Entweder liegt ihr relativer Abstand zum Rand des Quaders innerhalb von $\epsilon/|g_i|$, oder ihr Gradient hat die Größenordnung $\epsilon$.

<!--en:f4d7c4be-->
Ein eindimensionales Beispiel macht den Unterschied deutlich. Betrachten wir $\ell(w) = \tfrac12 (w - 3)^2$ mit dem unbeschränkten Minimierer $w = 3$. Bei $\lambda = \tfrac14$ kommt gekoppeltes Adam am Minimierer der Funktion mit Strafterm $3/(1 + \lambda) = 2.4$ zur Ruhe, AdamW dagegen bei $3.0$. Die Beschränkung auf den Quader $|w| \le 4$ ist nicht aktiv, der entkoppelte Zerfall lässt den Minimierer also unberührt. Bei $\lambda = 1$ kommt gekoppeltes Adam bei $1.5$ zur Ruhe, während AdamW bei $1.0$ genau auf dem Rand des Quaders bleibt.

<!--en:097017df-->
Die beiden Deutungen von $\lambda$ unterscheiden sich also grundsätzlich. Dominiert im Training das Rauschen, verkleinert AdamW alle Koordinaten pro Schritt gleichmäßig, während gekoppeltes Adam die Schrumpfung je nach Größenordnung des Gradienten verzerrt. An Fixpunkten mit exakten Gradienten kehren sich die Rollen um. Gekoppeltes Adam erhält die Maximum-a-posteriori-Lösung (MAP) für eine gaußsche A-priori-Verteilung der Gewichte (:numref:`sec_mdl-maximum_likelihood`). Bei AdamW legt $\lambda$ dagegen den Radius $1/\lambda$ eines $\ell_\infty$-Quaders mit der Nebenbedingung :cite:`Xie.Li.2024` fest. Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-adamw-linf` stellt den direkten Zusammenhang mit der Optimierung unter Nebenbedingungen her.

<!--en:8ba51c39-->
## Aufwärmphase

<!--en:662df56c-->
Im Schritt $t = 1$ lauten Adams bias-korrigierte Momente $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ und $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$. Die Aktualisierung ist also einfach $\eta\,\mathrm{sign}(\mathbf{g}_1)$, ein Schritt der vollen Länge $\eta$ in jeder Koordinate. Auf Grundlage eines einzigen Minibatches bewegt sich eine Koordinate mit Gradient $10^{-6}$ genauso weit wie eine mit Gradient $10^3$. In der **Aufwärmphase** (engl. warmup) steigt die Schrittweite während der ersten Trainingsschritte von einem Wert nahe null allmählich an :cite:`Goyal.Dollar.Girshick.ea.2017`. So bleiben die ersten Aktualisierungen klein. Es liegt nahe, die anfängliche Instabilität dem Vorkonditionierer $\hat{\mathbf{v}}_t$ zuzuschreiben, der bei $t = 1$ auf einer einzigen Beobachtung beruht. Ein einfaches Experiment zeigt, dass etwas anderes dahintersteckt.

<!--en:91368d6f-->
Ist ein Schritt der Länge $\eta$ überhaupt zu groß? Für eine Koordinate mit gleichbleibendem Gradienten ist er angemessen, denn auch im stationären Zustand beträgt ihre Schrittlänge $\eta$. Besteht der Gradient einer Koordinate jedoch aus reinem Rauschen, ist die Bewegung nutzlos. Um ihren Umfang zu messen, ziehen wir in jedem Schritt $2 \times 10^4$ unabhängige Gradientenkomponenten aus $\mathcal{N}(0, 1)$, verwenden Adam mit Standardeinstellungen und verfolgen das quadratische Mittel der Aktualisierung in Einheiten von $\eta$. Wir vergleichen vier Kurven: gewöhnliches Adam, Adam mit $\hat{\mathbf{v}}_t$ ersetzt durch die exakte Varianz $\sigma^2 = 1$ (wodurch der Schätzfehler des Vorkonditionierers vollständig entfällt), Adam ohne die beiden Bias-Korrekturen und Adam mit einem über die ersten $20$ Schritte linear ansteigenden Aufwärmfaktor.

<!--en:8f93d401-->
Bei gewöhnlichem Adam beträgt das quadratische Mittel der Aktualisierung anfangs $1.00\,\eta$, sinkt auf $0.33\,\eta$ bei $t = 10$ und pendelt sich innerhalb von etwa fünfzig Schritten bei $\sqrt{(1 - \beta_1)/(1 + \beta_1)}\,\eta = 0.23\,\eta$ ein. Das ist die stationäre Größenordnung eines exponentiellen gleitenden Mittelwerts von unabhängigen Rauschwerten mit Einheitsvarianz. In einer reinen Rauschkoordinate sind Adams erste Schritte damit etwa viermal so groß wie im stationären Zustand. Dieser Übergang dauert etwa zehn Schritte.

<!--en:1f64076c-->
Ist der verrauschte Vorkonditionierer dafür verantwortlich? Die zweite Kurve spricht dagegen. Ersetzen wir $\hat{\mathbf{v}}_t$ durch die exakte Varianz, bleibt das quadratische Mittel der Aktualisierung in jedem Schritt bis auf $0.01$ gleich (das Stichprobenrauschen bei $2 \times 10^4$ Koordinaten). Die vierfache Vergrößerung entsteht bei der Mittelung im *Momentum*. Es braucht etwa $1/(1 - \beta_1) = 10$ Schritte, um unabhängiges Rauschen auszumitteln. Der Vorkonditionierer gleicht bei $t = 1$ tatsächlich die Größenordnungen unterschiedlicher Koordinaten an. Dieser Effekt bleibt unsichtbar, wenn für alle Koordinaten $\sigma = 1$ gilt.

<!--en:0a3d43a1-->
Die Bias-Korrektur hat wesentlich größeren Einfluss. Ohne sie springt die erste Aktualisierung auf $3.16\,\eta$, passend zum Faktor $(1 - \beta_1)/\sqrt{1 - \beta_2}$ in :numref:`fig_mdl-opt-bias-correction`. Der Übergang dauert nun Hunderte statt zehn Schritte. Die Aktualisierung beträgt noch etwa $0.75\,\eta$ bei $t = 100$ und $0.29\,\eta$ bei $t = 1000$, weil sich das unkorrigierte $\mathbf{v}_t$ auf der langen Zeitskala $1/(1 - \beta_2) = 1000$ aufbaut. Ein Optimierer ohne Korrektur bräuchte schon tausend Aufwärmschritte, nur um seine eigene Initialisierung bei null auszugleichen. Mit Bias-Korrektur genügt ein linearer Anstieg über nur $20$ Schritte, um die verbleibende Überhöhung zu dämpfen. Die Aufwärmkurve bleibt dabei durchgehend unter $0.26\,\eta$.

<!--en:03533d22-->
Das erklärt Aufwärmphasen von einigen Dutzend Schritten. Beim Training großer Modelle sind jedoch oft Tausende üblich. Solche längeren Aufwärmphasen berücksichtigen einen zweiten Mechanismus, der in unserem Rauschmodell fehlt: die Krümmung. Eine Schrittweite ist nur unterhalb der Grenze $2/\lambda_{\max}$ stabil, die die steilste Krümmungsrichtung vorgibt (:numref:`subsec_mdl-quadratic-model`). Unmittelbar nach der zufälligen Initialisierung kann diese Krümmung ungewöhnlich groß sein und sich schnell ändern. Eine später im Training unbedenkliche Schrittweite kann daher anfangs instabil sein. :numref:`subsec_scheduler-warmup` untersucht beide Mechanismen an einem tatsächlichen neuronalen Netz und misst die anfänglichen Schrittweiten direkt.

<!--en:88072fff-->
## Strukturierte Vorkonditionierer

<!--en:60b912af-->
### Kronecker-Faktoren

<!--en:ddb91a09-->
An unserem gedrehten Tal scheiterten alle diagonalen Vorkonditionierer. Alle Verfahren dieses Abschnitts haben die vorkonditionierte Form $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \mathbf{P}_t^{-1} \mathbf{g}_t$ mit einer positiv definiten Matrix $\mathbf{P}_t$. Der Gradientenabstieg verwendet $\mathbf{P} = \mathbf{I}$, das Newton-Verfahren die Hesse-Matrix $\nabla^2 f$ und Adam eine aus den bisherigen Gradienten geschätzte Diagonalmatrix. Eine volle Krümmungsmatrix erfasst die Steilheit in jeder Richtung, auch abseits der Koordinatenachsen. Sie ist aber viel zu groß, um sie zu speichern. Schon eine einzelne $4096 \times 4096$-Gewichtsmatrix hat $1.7 \times 10^7$ Parameter. Ihre volle Krümmungsmatrix hätte $2.8 \times 10^{14}$ Einträge und bräuchte bei doppelter Genauigkeit ungefähr zwei Petabyte.

<!--en:5459c0b2-->
Ein Mittelweg ist möglich, weil die Parameter neuronaler Netze in Matrizen angeordnet sind. Betrachten wir eine lineare Schicht $\mathbf{y} = \mathbf{W}\mathbf{a}$ mit Gewichtsmatrix $\mathbf{W} \in \mathbb{R}^{m \times n}$, Eingabe $\mathbf{a} \in \mathbb{R}^n$ und Ausgabegradient $\boldsymbol{\delta} = \partial\ell/\partial\mathbf{y} \in \mathbb{R}^m$. Der Gradient bezüglich $\mathbf{W}$ ist das äußere Produkt $\boldsymbol{\delta}\mathbf{a}^\top$ vom Rang eins. Bezeichnen wir mit $\mathrm{vec}(\mathbf{V})$ den Vektor, der durch Stapeln der Spalten einer Matrix $\mathbf{V}$ entsteht, so gilt $\mathrm{vec}(\boldsymbol{\delta}\mathbf{a}^\top) = \mathbf{a} \otimes \boldsymbol{\delta}$. Dabei ist das **Kronecker-Produkt** $\mathbf{B} \otimes \mathbf{D}$ die Blockmatrix, deren Block an Position $(i, j)$ gleich $b_{ij}\mathbf{D}$ ist.

<!--en:142468bc-->
Die aus dem äußeren Produkt gewonnene Krümmungsmatrix dieser Schicht ist der Block der Fisher-Informationsmatrix $\mathbb{E}[\nabla\log p\,\nabla\log p^\top]$ (die wir in :numref:`sec_mdl-maximum_likelihood` untersuchen), der zu $\mathbf{W}$ gehört. Er lautet somit $\mathbb{E}\bigl[(\mathbf{a}\mathbf{a}^\top) \otimes (\boldsymbol{\delta}\boldsymbol{\delta}^\top)\bigr]$. **K-FAC** :cite:`Martens.Grosse.2015` nähert diesen Erwartungswert durch das Kronecker-Produkt $\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}$ an. Dabei ist $\mathbf{S}_{\mathrm{in}} = \mathbb{E}[\mathbf{a}\mathbf{a}^\top]$ eine $n \times n$-Matrix und $\mathbf{S}_{\mathrm{out}} = \mathbb{E}[\boldsymbol{\delta}\boldsymbol{\delta}^\top]$ eine $m \times m$-Matrix. Wären Eingaben und Ausgabegradienten statistisch unabhängig, wäre diese Faktorisierung exakt. Bei einem trainierten Netz ist das kaum zu erwarten. Wir verwenden die Näherung dennoch, weil sie sich mit geringem Aufwand invertieren lässt.

<!--en:886d89a0-->
Die Inverse eines Kronecker-Produkts wirkt nämlich direkt auf den Gradienten in seiner ursprünglichen Matrixform. Für symmetrische invertierbare Faktoren gilt

<!--en:bdd447e3-->
Um die Identität nachzuprüfen, schreiben wir die Spalte $j$ des Produkts $\mathbf{D}\mathbf{V}\mathbf{B}$ als $\sum_k b_{kj}\, \mathbf{D}\mathbf{v}_k$, wobei $\mathbf{v}_k$ die Spalte $k$ von $\mathbf{V}$ ist. Stapeln wir diese Spalten, ergibt sich das Blockprodukt $(\mathbf{B}^\top \otimes \mathbf{D})\,\mathrm{vec}(\mathbf{V})$, also $\mathrm{vec}(\mathbf{D}\mathbf{V}\mathbf{B}) = (\mathbf{B}^\top \otimes \mathbf{D})\,\mathrm{vec}(\mathbf{V})$. Aus $(\mathbf{B} \otimes \mathbf{D})^{-1} = \mathbf{B}^{-1} \otimes \mathbf{D}^{-1}$ und der Symmetrie der Faktoren folgt mit $\mathbf{B} = \mathbf{S}_{\mathrm{in}}^{-1}$ und $\mathbf{D} = \mathbf{S}_{\mathrm{out}}^{-1}$ die Identität :eqref:`eq_mdl-opt-kronecker`. So können wir eine Krümmungsmatrix mit $(mn)^2$ Einträgen mithilfe zweier kleinerer Inversen der Größen $n$ und $m$ invertieren und anwenden. Die folgende Codezelle vergleicht Speicher- und Rechenaufwand für unsere $4096 \times 4096$-Schicht.

<!--en:65faa8c8-->
Die Faktorisierung senkt den Speicherbedarf von $2.8 \times 10^{14}$ auf $3.4 \times 10^7$ Einträge und den Aufwand für die Inversion von $4.7 \times 10^{21}$ auf ungefähr $1.4 \times 10^{11}$ Operationen.

<!--en:fa8c223c-->
### K-FAC, Shampoo und Muon

<!--en:39747b8b-->
Die Kronecker-Identität lässt zwei Entscheidungen offen. Welche Matrizen faktorisieren wir, und welchen Exponenten verwenden wir für die Faktoren? Moderne strukturierte Optimierer nutzen eine beidseitige Vorkonditionierung der Form $\mathbf{G} \mapsto \mathbf{P}_{\mathrm{l}}^{-a}\mathbf{G}\,\mathbf{P}_{\mathrm{r}}^{-a}$ für die Gradientenmatrix $\mathbf{G}$ einer Schicht (fett gesetzt zur Unterscheidung von der skalaren Gradientenschranke $G$ der Regret-Analyse). Zusammen bilden diese Verfahren einen natürlichen Übergang von Adams diagonaler Skalierung zur Normierung ganzer Matrizen.

<!--en:0d6c4c71-->
K-FAC bildet seine Faktoren aus den zweiten Momenten der Schichteingaben und der Ausgabegradienten und verwendet den Exponenten $a = 1$. Es approximiert die Vorkonditionierung mit der Fisher-Informationsmatrix, den **natürlichen Gradienten** :cite:`Amari.1998`. Das entspricht der metrischen Sicht auf AdaGrad mit der Fisher-Matrix als Metrik. Bei Softmax-Kreuzentropie und quadratischem Verlust stimmt die Fisher-Matrix (mit aus dem Modell gezogenen Labels) mit der Gauß-Newton-Matrix überein. Diese positiv semidefinite Näherung der Hesse-Matrix lässt den Term mit den zweiten Ableitungen des Netzes weg. Für diese Verlustfunktionen nähert K-FAC also die tatsächliche Krümmung an.

<!--en:0cf51a1e-->
**Shampoo** :cite:`Gupta.Koren.Singer.2018` bildet seinen linken und rechten Faktor dagegen im Sinne von AdaGrad direkt aus den über die Schritte angesammelten Gradientenmatrizen. Es setzt $\mathbf{P}_{\mathrm{l}} = \sum_t \mathbf{G}_t\mathbf{G}_t^\top$ und $\mathbf{P}_{\mathrm{r}} = \sum_t \mathbf{G}_t^\top\mathbf{G}_t$ mit $a = \tfrac14$. **Muon** :cite:`Jordan.Jin.Boza.ea.2024` lässt sich als Shampoo ohne Gedächtnis auffassen. Verwenden wir nur die aktuelle Matrix $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ (als quadratisch und invertierbar vorausgesetzt), so gilt

<!--en:5f530062-->
Das ist der Polarfaktor von $\mathbf{G}$. Er erhält die Singulärvektoren von $\mathbf{G}$ und setzt jeden Singulärwert auf eins. Während Adam die Koordinaten der Aktualisierung angleicht, gleicht Muon deren Singulärrichtungen an. Anders als die koordinatenweise Normierung vertauscht diese Abbildung mit orthogonalen Drehungen der Zeilen und Spalten (Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`d). Muon als praktischen Optimierer, der diese Orthogonalisierung auf die Momentummatrix anwendet, behandeln wir im Haupttext (:numref:`sec_muon`). Die Newton-Schulz-Iteration, die den Polarfaktor allein durch Matrixmultiplikationen berechnet, untersuchen wir in :numref:`subsec_mdl-orthogonalized-updates`.

<!--en:3bb1f9b7-->
Keines der drei Verfahren bildet das vollständige Newton-Verfahren nach. Muon speichert keine Matrizen zweiter Momente und braucht nur wenige Matrixprodukte pro Schritt. K-FAC und Shampoo invertieren dagegen ihre beiden Faktoren oder ziehen deren Matrixwurzeln. Diesen Aufwand verteilen sie üblicherweise über mehrere Schritte. Außerdem setzen alle positiven Ergebnisse dieses Abschnitts voraus, dass die Krümmung entweder an den Koordinatenachsen oder an der Kronecker-Struktur der einzelnen Schichten ausgerichtet ist. Die tatsächliche Krümmung eines tiefen neuronalen Netzes ist weder diagonal noch konstant. Warum diese adaptiven Verfahren in der Praxis so gut funktionieren, bleibt daher eine offene Forschungsfrage.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ab35c525-->
AdaGrad, RMSProp und Adam teilen die Aktualisierung jeder Koordinate durch das quadratische Mittel ihrer bisherigen Gradienten. Damit entfällt die Abhängigkeit von der Konditionszahl, wenn die Krümmungsrichtungen mit den Koordinatenachsen übereinstimmen. Für $\epsilon = 0$ sind Adams Iterierte exakt invariant gegenüber einer koordinatenweisen Skalierung. In einem achsenparallelen Tal braucht Skalierung allein deshalb eine Größenordnung weniger Schritte als ein optimal eingestelltes Heavy-Ball-Verfahren. Drehen wir das Tal jedoch um $45^\circ$, verschwindet dieser Vorteil. Die Skalierung allein konvergiert nie, und Adam mit Momentum braucht ungefähr doppelt so viele Schritte wie das Heavy-Ball-Verfahren.

<!--en:ca71d1bd-->
Auch das Vergessen früherer Gradienten hat Folgen. Bei einer periodischen Folge konvexer Verlustfunktionen kann das kurze Gedächtnis für das zweite Moment Adam zum schlechtesten Punkt des Definitionsbereichs treiben. AMSGrad behebt das Problem mit seinem laufenden Maximum. Beim Gewichtszerfall bleibt der MAP-Fixpunkt erhalten, wenn ein $\ell_2$-Strafterm Adam durchläuft. Die Schrumpfung jeder Koordinate ist dann allerdings umgekehrt proportional zur Größenordnung ihres Gradienten. AdamW entkoppelt den Gewichtszerfall, sodass alle Koordinaten pro Schritt gleichmäßig schrumpfen. An Fixpunkten mit exakten Gradienten bestimmt $\lambda$ dadurch den Radius $1/\lambda$ eines $\ell_\infty$-Quaders.

<!--en:9896e968-->
In Koordinaten mit reinem Rauschen sind Adams erste Aktualisierungen etwa viermal so groß wie im stationären Zustand. Der Grund ist die Mittelung im Momentum und nicht der Vorkonditionierer. Ohne Bias-Korrektur fallen die Schritte über Hunderte von Schritten noch deutlich größer aus. Längere Aufwärmphasen in der Praxis berücksichtigen dagegen die anfängliche Krümmung. Über die diagonale Skalierung hinaus erfassen Vorkonditionierer mit Kronecker-Faktoren wie K-FAC, Shampoo und Muon mithilfe zweier kompakter Faktoren Zusammenhänge zwischen den Koordinaten einer Gewichtsmatrix. Muon gleicht dabei die Singulärrichtungen der Aktualisierung einer Schicht an.

<!--en:2294243a-->
## Übungen

<!--en:915d4364-->
1. **AdaGrad aus Sicht der Metrik.** In Aufgabe :exref:`ex_mdl-gradient-based-optimization-show-that-is-de`b in :numref:`sec_mdl-gradient-based-optimization` haben wir gezeigt, dass die Richtung des steilsten Abstiegs bezüglich der Norm $\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P} \mathbf{d}}$ proportional zu $-\mathbf{P}^{-1}\nabla f$ ist.
    1. Setzen Sie $\mathbf{P}_t = \mathrm{diag}\bigl(\sqrt{\mathbf{s}_t}\bigr) + \epsilon \mathbf{I}$ mit $\mathbf{s}_t = \sum_{\tau \le t} \mathbf{g}_\tau^2$ und leiten Sie daraus :eqref:`eq_mdl-opt-adagrad` her.
    1. Ersetzen Sie die Zielfunktion $f$ durch $cf$ mit einer Konstanten $c > 0$ und setzen Sie $\epsilon = 0$. Wie ändert sich die Aktualisierung von AdaGrad? Wie würde sie sich ändern, wenn Sie $\mathbf{P}_t$ aus $\mathbf{s}_t$ statt aus $\sqrt{\mathbf{s}_t}$ bilden würden? Vergleichen Sie beide Varianten mit dem Gradientenabstieg und dem Newton-Verfahren.
    1. Die Gradienten einer Koordinate seien unabhängig und identisch verteilt mit $\mathbb{E}[g_{t,i}^2] = \sigma^2$. Wie schnell nimmt die effektive Schrittweite dieser Koordinate in den beiden Varianten mit $t$ ab, und wie hängt sie von $\sigma$ ab?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-adagrad-metric-view`

<!--en:ddefe05e-->
1. **Konzentrierte Gradienten und die Schranke von AdaGrad.** Bis auf konstante Faktoren erreicht der projizierte Online-Gradientenabstieg mit passend gewählter globaler Schrittweite einen Regret von $D_2 G \sqrt{T}$ auf einer konvexen Menge mit euklidischem Durchmesser $D_2$, wenn $\|\mathbf{g}_t\| \le G$ gilt (Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-regret-ogd`). Diagonales AdaGrad erreicht einen Regret von $D_\infty \sum_{i=1}^{d} \sqrt{\sum_{t=1}^{T} g_{t,i}^2}$, wobei $D_\infty$ der Durchmesser der Menge in der $\ell_\infty$-Norm ist :cite:`Duchi.Hazan.Singer.2011`. Betrachten Sie den Quader $[-1, 1]^d$ und One-Hot-Gradienten. In jedem Schritt erhält eine Koordinate $i$, die unabhängig mit Wahrscheinlichkeit $p_i$ gezogen wird, den Gradienten $+1$ oder $-1$, alle anderen Koordinaten erhalten $0$.
    1. Berechnen Sie beide Schranken. Schätzen Sie für AdaGrad den Erwartungswert jeder Quadratwurzel mit der Jensen-Ungleichung ab. Zeigen Sie, dass das Verhältnis der AdaGrad-Schranke zur Schranke des Gradientenabstiegs höchstens $\sum_i \sqrt{p_i}/\sqrt{d}$ beträgt.
    1. Berechnen Sie das Verhältnis für gleichförmige $p_i$ und für Zipf-Häufigkeiten $p_i \propto i^{-\alpha}$ mit $d = 10^4$ und $\alpha \in \{1, 2, 3\}$.
    1. Worthäufigkeiten in natürlichsprachlichen Texten folgen dem Zipfschen Gesetz mit $\alpha \approx 1$. Ihr Verhältnis beträgt dort etwa $0.63$. Wie viel des Faktors $\sqrt{d}$ aus :numref:`subsec_mdl-per-coordinate` erreicht ein Bag-of-Words-Modell? Mit welcher Verteilung der Gradientenmasse ließe sich der ganze Faktor erreichen?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-concentrated-gradients`

<!--en:d7f40a1b-->
1. **Reddis Beispiel von Hand.** Verwenden Sie $C = 4$, $\beta_1 = 0$ und wie in der Proposition $\beta_2 = 1/(1 + C^2) = 1/17$. Verzichten Sie wie in `online` auf die Bias-Korrektur.
    1. Die Schrittweite $\eta$ sei fest, die Iterierte liege im Inneren von $[-1, 1]$ und $v_t$ habe seinen periodischen Verlauf erreicht. Schreiben Sie $v_a$ für den Wert von $v$ nach dem $+C$-Schritt. Lösen Sie die Rekursion über drei Schritte für $v_a$ in geschlossener Form und leiten Sie die periodischen Werte $15.12$, $1.83$, $1.05$ sowie die drei Verschiebungen aus dem Text her. Berechnen Sie die drei SGD-Schritte $\eta g_t/C$. Welche Schritte vergrößert Adam gegenüber SGD und warum?
    1. AMSGrad ersetzt $v_t$ durch $\tilde{v}_t = \max(\tilde{v}_{t-1}, v_t)$. Zeigen Sie, dass die effektive Schrittweite $\eta_t/\sqrt{\tilde{v}_t}$ mit $t$ nicht zunimmt, sofern $\eta_t$ nicht zunimmt. Berechnen Sie die drei Verschiebungen im periodischen Zustand und zeigen Sie, dass die Nettoverschiebung $-0.51\,\eta$ beträgt.
    1. Die Proposition setzt $\eta < \sqrt{1 - \beta_2}$ für die Schrittweiten $\eta_t = \eta/\sqrt{t}$ voraus. Zeigen Sie, an welcher Stelle diese Voraussetzung eingeht. Beginnen Sie eine Periode bei $x = 1$. Nutzen Sie die Ungleichung $v \ge (1-\beta_2)C^2$, die nach dem Gradienten $+C$ gilt, um zu zeigen, dass der erste Schritt der Periode in $(0, 1)$ endet und die Projektion ihn daher unverändert lässt.
   :exlabel:`ex_mdl-adaptive-stochastic-methods-reddi-by-hand`

<!--en:2e9f42d0-->
1. **Regret des Online-Gradientenabstiegs.** ● Der projizierte Online-Gradientenabstieg wählt $\mathbf{x}_{t+1} = \Pi_{\mathcal{C}}(\mathbf{x}_t - \eta_t \mathbf{g}_t)$ mit $\mathbf{g}_t = \nabla f_t(\mathbf{x}_t)$. Dabei ist $\Pi_{\mathcal{C}}$ die euklidische Projektion auf eine abgeschlossene konvexe Menge $\mathcal{C}$ mit Durchmesser $D$. Es gilt $\|\mathbf{g}_t\| \le G$, und die Schrittweiten $\eta_t$ nehmen nicht zu :cite:`Zinkevich.2003`.
    1. Beweisen Sie für den Regret die Schranke

       $$\sum_{t=1}^{T} f_t(\mathbf{x}_t) - \min_{\mathbf{x} \in \mathcal{C}} \sum_{t=1}^{T} f_t(\mathbf{x}) \;\le\; \frac{D^2}{2\eta_T} + \frac{G^2}{2} \sum_{t=1}^{T} \eta_t.$$

       Gehen Sie von $f_t(\mathbf{x}_t) - f_t(\mathbf{x}) \le \mathbf{g}_t^\top(\mathbf{x}_t - \mathbf{x})$ aus und nutzen Sie, dass die Projektion die Abstände zu Punkten von $\mathcal{C}$ nicht vergrößert.
    1. Zeigen Sie, dass $\eta_t = D/(G\sqrt{t})$ einen Regret von höchstens $\tfrac32 DG\sqrt{T}$ ergibt und dass die Schranke bei $\eta_t = c/t$ linear in $T$ wächst, und zwar für jedes $c > 0$.
    1. Kombinieren Sie diese beiden Schranken mit den Abnahmeraten aus Aufgabe :exref:`ex_mdl-adaptive-stochastic-methods-adagrad-metric-view`c. Erklären Sie so, warum AdaGrad durch $\sqrt{\mathbf{s}_t}$ und nicht durch $\mathbf{s}_t$ teilt.
   :exlabel:`ex_mdl-adaptive-stochastic-methods-regret-ogd`

<!--en:235dcc95-->
1. [code] **Drehabhängigkeit.** ● Verwenden Sie `rotated`, `run` und die Funktionen zum Erzeugen der Aktualisierungsregeln aus der Zelle `#adaptive-stochastic-methods-adam-vs-gd`.
    1. Zeigen Sie, dass die Iterierten des Gradientenabstiegs und des Heavy-Ball-Verfahrens auf $\mathbf{A}_\theta$ vom Startpunkt $\mathbf{R}_\theta(1, 1)^\top$ aus gleich dem Produkt aus $\mathbf{R}_\theta$ und den jeweiligen Iterierten im achsenparallelen Tal sind. Ihre Schrittzahlen hängen also nicht von $\theta$ ab.
    1. Geben Sie für $\beta_1 = 0$ bei $\theta = \pi/4$ den Wert von $f$ nach $40{,}000$ Schritten für jedes $\eta$ im Raster der Tabelle aus. Vergleichen Sie ihn mit dem Wert $250\,\eta^2$ aus dem Text. Welchem Bestandteil von Adam verdankt das Verfahren seinen verbleibenden Vorteil gegenüber dem Gradientenabstieg?
    1. Geben Sie Adams effektive koordinatenweise Schrittweiten bei $\theta = \pi/16$ aus. Warum geht schon bei dieser kleinen Drehung der größte Teil des Schrittweitenverhältnisses verloren, das Adam im achsenparallelen Tal aufbaut?
    1. Aufgabe :exref:`ex_mdl-gradient-based-optimization-steepest-other-norms` in :numref:`sec_mdl-gradient-based-optimization` bestimmt $-\mathrm{sign}(\mathbf{g})$ als Richtung des steilsten Abstiegs bezüglich der $\ell_\infty$-Norm und $-\mathbf{U}\mathbf{V}^\top$ als Richtung des steilsten Abstiegs bezüglich der Spektralnorm, wobei $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ gilt. Zeigen Sie, dass Adam mit $\beta_1 = \beta_2 = 0$ und $\epsilon = 0$ in Richtung $-\mathrm{sign}(\mathbf{g}_t)$ geht. Nun seien $\mathbf{Q}_1$ und $\mathbf{Q}_2$ orthogonal. Zeigen Sie, dass die Abbildung $\mathbf{G} \mapsto \mathbf{U}\mathbf{V}^\top$ mit $\mathbf{G} \mapsto \mathbf{Q}_1 \mathbf{G} \mathbf{Q}_2$ vertauscht. Geben Sie ein $2 \times 2$-Beispiel an, in dem die Vorzeichenabbildung das nicht tut. Was lässt sich daraus über die Drehabhängigkeit von Muon im Vergleich zu Adam schließen?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`

<!--en:990611fd-->
1. [extended] **Die implizite $\ell_\infty$-Nebenbedingung von AdamW.** ●● Nach der Proposition in :numref:`subsec_mdl-decoupled-weight-decay` liegen die Fixpunkte von AdamW im Quader $|w_i| < 1/\lambda$. Für $\epsilon \to 0$ liegen sie überall dort auf seinem Rand, wo der Gradient nicht verschwindet. :citet:`Xie.Li.2024` beweisen: Konvergiert AdamW bei Verwendung des gesamten Datensatzes mit $\beta_1 \le \beta_2$ und nicht zunehmenden Schrittweiten, deren Summe divergiert, dann ist sein Grenzwert ein KKT-Punkt von $\min_{\mathbf{w}} \ell(\mathbf{w})$ unter der Nebenbedingung $\|\mathbf{w}\|_\infty \le 1/\lambda$.
    1. Ein zulässiges $\mathbf{w}$ ist ein KKT-Punkt dieses Problems (:numref:`sec_mdl-constrained-optimization-duality` leitet die Bedingungen allgemein her), wenn für jede Koordinate $\partial_i \ell(\mathbf{w}) = 0$ bei $|w_i| < 1/\lambda$ und $w_i\, \partial_i \ell(\mathbf{w}) \le 0$ bei $|w_i| = 1/\lambda$ gilt. Zeigen Sie, dass an einem solchen Punkt keine zulässige Richtung $\ell$ in erster Ordnung verringert.
    1. Setzen Sie wie in der Proposition $\epsilon = 0$ in der AdamW-Bedingung :eqref:`eq_mdl-opt-adamw-fixed-point`. Zeigen Sie, dass eine Koordinate mit $\partial_i \ell \ne 0$ die KKT-Bedingungen erfüllt. Warum folgt aus der Bedingung allein für eine Koordinate mit verschwindendem Gradienten noch nicht $|w_i| \le 1/\lambda$?
    1. Erzeugen Sie linear separierbare Daten für die logistische Regression mit $n = 100$, $d = 5$, standardnormalverteilten Merkmalen und Labels $y_i = \mathrm{sign}(\mathbf{x}_i^\top \mathbf{w}_{\mathrm{true}})$, sodass die unregularisierte Verlustfunktion keinen endlichen Minimierer hat. Verwenden Sie AdamW auf dem gesamten Datensatz mit $\epsilon = 10^{-16}$ für $5 \times 10^4$ Schritte. Lassen Sie $\eta$ nach einem Kosinusplan von $10^{-2}$ auf $0$ fallen und verwenden Sie $\lambda \in \{0.05, 0.1, 0.2\}$. Vergleichen Sie das Ergebnis mit der Lösung des Problems unter der Quaderbeschränkung durch projizierten Gradientenabstieg, der hier nach jedem Gradientenschritt jede Koordinate auf $[-1/\lambda, 1/\lambda]$ beschränkt, sowie mit Adam für $\ell + \tfrac{\lambda}{2}\|\mathbf{w}\|^2$.
    1. Wiederholen Sie die AdamW-Läufe mit Minibatch-Gradienten für Batchgrößen von $10$ und $50$. Wo liegt die Iterierte am Ende relativ zum Quader und warum?
    1. Was steuert $\lambda$ diesen Ergebnissen zufolge bei AdamW, und wie unterscheidet sich das von der Deutung von $\lambda$ über eine gaußsche A-priori-Verteilung in :numref:`sec_mdl-maximum_likelihood`?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-adamw-linf`

<!--en:a60e1f77-->
1. **Die Kronecker-Identität und ihr Rechenaufwand.** Stapeln Sie für $\mathrm{vec}$ die Spalten.
    1. Beweisen Sie $\mathrm{vec}(\mathbf{D}\mathbf{V}\mathbf{B}) = (\mathbf{B}^\top \otimes \mathbf{D})\,\mathrm{vec}(\mathbf{V})$, indem Sie die Spalte $j$ des Produkts ausschreiben. Beweisen Sie außerdem $(\mathbf{B} \otimes \mathbf{D})(\mathbf{B}' \otimes \mathbf{D}') = (\mathbf{B}\mathbf{B}') \otimes (\mathbf{D}\mathbf{D}')$.
    1. Leiten Sie :eqref:`eq_mdl-opt-kronecker` für symmetrische invertierbare Faktoren her. Welche Formel gilt für nicht symmetrische Faktoren, welche beim Stapeln der Zeilen?
    1. Bestimmen Sie für eine $m \times n$-Schicht die Anzahl der Operationen, um die volle $mn \times mn$-Matrix zu invertieren und anzuwenden. Vergleichen Sie diese mit der Anzahl der Operationen, um die beiden Faktoren zu invertieren und gemäß :eqref:`eq_mdl-opt-kronecker` anzuwenden ($O(m^3 + n^3 + mn(m + n))$). Berechnen Sie beide Anzahlen für $m = n = 4096$ sowie für $m = 4096$, $n = 16384$.
    1. Setzen Sie für einen einzelnen Gradienten $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ die Faktoren $\mathbf{P}_{\mathrm{l}} = \mathbf{G}\mathbf{G}^\top$ und $\mathbf{P}_{\mathrm{r}} = \mathbf{G}^\top\mathbf{G}$. Zeigen Sie, dass der Shampoo-Schritt $\mathbf{P}_{\mathrm{l}}^{-1/4}\mathbf{G}\,\mathbf{P}_{\mathrm{r}}^{-1/4}$ gleich $\mathbf{U}\mathbf{V}^\top$ ist. Wie ist die inverse vierte Wurzel zu verstehen, wenn $\mathbf{G}$ nicht quadratisch ist, und welche Singulärvektoren bleiben im Ergebnis erhalten?

<!--en:f53f2a34-->
[Diskussion](https://d2l.discourse.group/t/36829)

<!--en:b694fbe2-->
Adaptive Verfahren<br>**AdaGrad · RMSProp · Adam und Bias-Korrektur · Skaleninvarianz · Drehabhängigkeit · AdamW · Aufwärmphase · K-FAC und Muon**

<!--en:9654e43b-->
::: {.slide title="Wie eine globale Lernrate neuronale Netze ausbremst"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:25d8c212-->
Moderne neuronale Netze haben $10^7$ bis $10^{11}$ Parameter. Zwischen Schichten und Koordinaten unterscheiden sich die Krümmungen um Größenordnungen. Für $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ von $(1, 1)$ bis $f(\mathbf{x}) < 10^{-8}$ gilt:

<!--en:250c7aad-->
- **Gradientenabstieg mit optimaler gemeinsamer Schrittweite ($\eta = 2/1001$):** $6{,}160$ Schritte.

<!--en:c790cd3d-->
- **Heavy-Ball-Momentum mit optimierten Parametern:** $296$ Schritte.

<!--en:2382bcd5-->
- **Koordinatenweise RMS-Skalierung (achsenparallel):** **$22$ Schritte**. Steigt $\kappa$ von $10^3$ auf $10^5$, kommen nur $6$ Schritte hinzu.

<!--en:dfb2be84-->
- **Drehung um $45^\circ$:** Skalierung allein erreicht $10^{-8}$ **nie**.

<!--en:1b295193-->
::: {.slide title="Ideale diagonale Vorkonditionierung und Näherung über Gradienten"}

<!--en:a292733c-->
[Koordinatenweise Skalierung]{.kicker}

<!--en:c764f46d-->
Auf einer diagonalen quadratischen Schale $f(\mathbf{x}) = \tfrac12\sum_i h_i x_i^2$ gilt für jede Koordinate die Aktualisierung $x_{t+1,i} = (1 - \eta_i h_i)x_{t,i}$:

<!--en:65f5dfc6-->
- **Diagonaler Newton-Schritt:** Mit $\eta_i = 1/h_i$ ist null in 1 Schritt erreicht. Damit entfällt $\kappa = h_{\max}/h_{\min}$.

<!--en:19092a99-->
- **Unbekannte, verrauschte Krümmungen:** Im Deep Learning ist $h_i$ unbekannt und durch Minibatch-Rauschen verdeckt.

<!--en:090e1d6e-->
- **Quadratisches Mittel als Ersatz:** Wegen $|g_i| = h_i|x_i|$ erhält jede steilere Achse automatisch kleinere Schritte, wenn man jeden Schritt durch das gleitende quadratische Mittel der bisherigen Gradienten $\sqrt{v_{t,i}}$ teilt.

<!--en:b1927b21-->
::: {.slide title="AdaGrad und die kumulative koordinatenweise Normierung"}

<!--en:260a9800-->
[AdaGrad]{.kicker}

<!--en:25a4f19e-->
**AdaGrad** summiert die quadrierten Gradienten $\mathbf{s}_t$ elementweise und teilt jeden Schritt durch $\sqrt{s_{t,i}}$:

<!--en:a57928d1-->
- **Seltene und häufige Merkmale:** Nach $10^4$ Schritten erhält ein häufig aktives Merkmal ($g_{t,1} = \pm 1$) die Schrittweite $\eta/100$, eine nur einmal aktive Einbettung eines seltenen Tokens dagegen die volle Schrittweite $\eta$.

<!--en:46b4d4ab-->
- **Diagonale Metrik:** Steilster Abstieg bezüglich $\mathbf{P}_t = \mathrm{diag}(\sqrt{\mathbf{s}_t})$.

<!--en:fa9992cb-->
::: {.slide title="Regret bei der konvexen Online-Optimierung und abnehmende Schrittweiten bei AdaGrad"}

<!--en:260a9800-2-->
[AdaGrad]{.kicker}

<!--en:20310b10-->
AdaGrad wurde für die **konvexe Online-Optimierung** entwickelt und minimiert den kumulativen Regret $\sum_{t=1}^T f_t(\mathbf{x}_t) - \min_{\mathbf{x}} \sum_{t=1}^T f_t(\mathbf{x})$:

<!--en:b5ac5605-->
- **Koordinatenabhängige Regret-Schranke:** Auf einem Quader mit $\ell_\infty$-Durchmesser $D_\infty$ ($D_2 = \sqrt{d}\,D_\infty$) erreicht der Online-Gradientenabstieg den Regret $O(D_2 G\sqrt{T})$. Für AdaGrad gilt dagegen:
  $$\text{Regret}_T(\text{AdaGrad}) = O\!\left(D_\infty \sum_{i=1}^d \sqrt{\sum_{t=1}^T g_{t,i}^2}\right)$$
  Konzentriert sich die Gradientenmasse auf wenige Koordinaten (etwa bei Worthäufigkeiten nach dem Zipfschen Gesetz), ist die Schranke von AdaGrad um bis zu einen Faktor $\sqrt{d}$ kleiner.

<!--en:97e879f1-->
- **Warum AdaGrad bei tiefen Netzen stockt:** Bei anhaltendem Minibatch-Rauschen mit Varianz $\sigma^2$ wächst die kumulative Summe linear ($s_{t,i} \approx \sigma^2 t$). Dadurch sinkt die effektive Schrittweite $\eta/(\sigma\sqrt{t})$ unaufhaltsam gegen null, selbst wenn der Optimierer noch weit von einem Minimum entfernt ist.

<!--en:92b46ebf-->
::: {.slide title="RMSProp und der exponentielle gleitende Mittelwert der Quadrate"}

<!--en:574f0b23-->
[RMSProp]{.kicker}

<!--en:347d2c90-->
**RMSProp** ersetzt die Summe über alle bisherigen Schritte aus AdaGrad durch einen exponentiellen gleitenden Mittelwert (EMA) der quadrierten Gradienten:

<!--en:226fb179-->
- **Begrenztes Gedächtnis:** Das Gewicht früherer quadrierter Gradienten halbiert sich alle $k_{1/2} = \frac{\log 2}{1 - \beta_2}$ Schritte ($\approx 7$ Schritte bei $\beta_2 = 0.9$, $\approx 693$ Schritte bei $\beta_2 = 0.999$).

<!--en:e128daba-->
- **Dauerhaft positive Schrittweiten:** Bei stationärem Rauschen bleiben die effektiven Schrittweiten von null weg beschränkt und folgen Änderungen der Krümmung.

<!--en:b7ef2396-->
::: {.slide title="Adam kombiniert Momentum mit RMS-Normierung"}

<!--en:4fdf6ec1-->
[Adam]{.kicker}

<!--en:fc28ac09-->
**Adam** kombiniert den EMA des zweiten Moments $\mathbf{v}_t$ aus RMSProp mit einem Momentum-EMA des ersten Moments $\mathbf{m}_t$ und korrigiert beide für die Initialisierung bei null:

<!--en:e75da66f-->
- **Standardwerte:** $\beta_1 = 0.9$ (Halbwertszeit $\approx 7$ Schritte), $\beta_2 = 0.999$ (Halbwertszeit $\approx 693$ Schritte), $\epsilon = 10^{-8}$.

<!--en:31d19444-->
- **Zwei Wirkungen:** $\hat{\mathbf{m}}_t$ mittelt hochfrequente Schwingungen aus, bevor die Skalierung mit $\sqrt{\hat{\mathbf{v}}_t}$ erfolgt.

<!--en:e318e5f1-->
::: {.slide title="Warum die Initialisierung bei null eine Bias-Korrektur erfordert"}

<!--en:4fdf6ec1-2-->
[Adam]{.kicker}

<!--en:d0e9eca6-->
Bei stationärer Größenordnung der quadrierten Gradienten ($\mathbb{E}[\mathbf{g}_\tau^2] = \overline{\mathbf{g}^2}$) ergibt die Summe der geometrischen Reihe:

<!--en:b13cf763-->
- Bei $\beta_2 = 0.999$ enthält $\mathbf{v}_{10}$ nur $1\%$ von $\overline{\mathbf{g}^2}$, und $\mathbf{v}_{1000}$ enthält nur $63\%$.

<!--en:8fdff086-->
- **Überhöhte Schrittweite ohne Korrektur:** Ohne beide Korrekturen vergrößert sich die Aktualisierung um $\frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}}$. Der Faktor beginnt bei $3.16\times$ ($t=1$) und erreicht maximal **$6.6\times$** ($t=12$), weil sich $\mathbf{m}_t$ um den Faktor $100\times$ schneller aufbaut als $\mathbf{v}_t$.

<!--en:4cada730-->
::: {.slide title="Exakte Invarianz von Adam gegenüber koordinatenweiser Skalierung"}

<!--en:3142969a-->
[Skaleninvarianz]{.kicker}

<!--en:02aab5fa-->
Für $\epsilon = 0$ und $\nabla\tilde{f}(\mathbf{x}) = \mathrm{diag}(c_1, \ldots, c_d)\nabla f(\mathbf{x})$ ($c_i > 0$) ergibt die Linearität der gleitenden Mittelwerte $\tilde{m}_{t,i} = c_i m_{t,i}$ und $\tilde{v}_{t,i} = c_i^2 v_{t,i}$:

<!--en:1a77bd6b-->
- **Identische Trajektorien:** Auf $\tfrac12(x_1^2 + 1000\,x_2^2)$ folgt Adam vom Startpunkt $(1, 1)$ aus genau derselben Bahn $x_{t,1} = x_{t,2}$ wie auf der rotationssymmetrischen Schale $\tfrac12(x_1^2 + x_2^2)$.

<!--en:54c5cecd-->
::: {.slide title="Folgen und Grenzen der Skaleninvarianz"}

<!--en:3142969a-2-->
[Skaleninvarianz]{.kicker}

<!--en:7a7a8004-->
Die Proposition zur Skaleninvarianz erklärt am Beispiel $f(\mathbf{x}) = \tfrac12\sum_{i=1}^d h_i x_i^2$ sowohl Adams Stärke in achsenparallelen Tälern als auch die strukturellen Grenzen des Verfahrens:

<!--en:cb22dd60-->
- **Jedes achsenparallele Tal verhält sich wie eine runde Schale:** Da $\nabla f(\mathbf{x}) = \mathrm{diag}(h)\,\mathbf{x}$ der Gradient der isotropen quadratischen Funktion $\tfrac12\|\mathbf{x}\|^2$ mit Skalierung $c_i = h_i$ ist, beschreibt Adam auf $\tfrac12(x_1^2 + 1000\,x_2^2)$ vom Startpunkt $(1, 1)$ aus *genau dieselbe Trajektorie* $x_{t,1} = x_{t,2}$ wie Adam auf $\tfrac12(x_1^2 + x_2^2)$.

<!--en:c0ace18b-->
- **Abstand statt Krümmung:** Die ersten Schritte verschieben jede Koordinate um ungefähr $\pm\eta$, unabhängig von $h_i$. Daher skaliert die Schrittzahl mit dem anfänglichen Abstand $\|\mathbf{x}_0\|_\infty / \eta$ und nicht mit $\kappa = h_{\max}/h_{\min}$.

<!--en:a5b0c075-->
- **Drei Grenzen der Invarianz:**
  1. $\epsilon > 0$ bricht die Invarianz, sobald $|g_{t,i}| \lesssim \epsilon$.
  2. Im Allgemeinen folgt Adam *nicht* allein dem Vorzeichen. Die Aktualisierung ergibt $-\eta\,\mathrm{sign}(\mathbf{g}_t)$ nur bei $t=1$, wenn $\beta_1 = \beta_2 = 0$ gilt oder wenn $|g_{t,i}|$ über das Gedächtnis von $\hat{v}_t$ hinweg konstant bleibt.
  3. Die Invarianz gilt nur für *diagonale* Skalierungen der Koordinaten. Bei orthogonalen Drehungen geht sie verloren.

<!--en:d49581f7-->
::: {.slide title="Fairer Vergleich im achsenparallelen und im um 45° gedrehten Tal"}

<!--en:61096002-->
[Drehabhängigkeit]{.kicker}

<!--en:069f4ab7-->
Vergleich des optimal eingestellten Gradientenabstiegs und Heavy-Ball-Verfahrens (mit bekannten exakten Eigenwerten $\mu = 1, L = 1000$) mit Adam bei 7 Lernraten ($\beta_1 = 0$ für reine Skalierung, verglichen mit $\beta_1 = 0.9$). Beide Tests laufen im achsenparallelen und im um $\theta = 45^\circ$ gedrehten Tal:

<!--en:3e8670ce-->
::: {.slide title="Robustheit und Drehabhängigkeit im Vergleich"}

<!--en:61096002-2-->
[Drehabhängigkeit]{.kicker}

<!--en:d4c5362d-->
- **Achsenparallel ($\theta = 0^\circ$):** Mit reiner Skalierung ($\beta_1 = 0$) braucht Adam **$22$ Schritte**, mit Standard-Momentum ($\beta_1 = 0.9$) zwischen $165$ und $205$ Schritte über einen Bereich, der den Faktor $100\times$ in $\eta$ abdeckt.

<!--en:e5f02348-->
- **Gedreht ($\theta = 45^\circ$):** Gradientenabstieg ($6{,}160$ Schritte) und Heavy-Ball-Verfahren ($296$ Schritte) sind drehinvariant. Adam mit reiner Skalierung gerät in einen 2-Zyklus, Adam mit Standardeinstellungen braucht $559$ bis $12{,}862$ Schritte.

<!--en:2264e929-->
- **Warum $45^\circ$ die diagonale Skalierung aushebelt:** Die steile Eigenkomponente $(-1, 1)/\sqrt{2}$ projiziert sich gleichmäßig auf beide Achsen. Damit gilt $\hat{v}_{t,1} \approx \hat{v}_{t,2}$, und der diagonale Vorkonditionierer reduziert sich auf einen Skalar.

<!--en:0440115c-->
::: {.slide title="Effektive Schrittweiten bei unterschiedlichen Krümmungsverhältnissen"}

<!--en:61096002-3-->
[Drehabhängigkeit]{.kicker}

<!--en:0107200e-->
Im achsenparallelen Tal ($\beta_1 = 0, \eta = 0.1, \epsilon = 0$) erkennt Adam das Krümmungsverhältnis $1000{:}1$ schon in Schritt 1 allein aus den Gradientenbeträgen:

<!--en:eac28b69-->
- Steigt $\kappa$ um den Faktor $10^5\times$ ($1$ $\to$ $10^3$ $\to$ $10^5$), kommen nur $14$ Schritte hinzu ($40$ $\to$ $48$ $\to$ $54$). Wächst dagegen der Abstand von $\mathbf{x}_0$ um den Faktor $100\times$ ($1$ $\to$ $100$), steigt die Schrittzahl um den Faktor $\approx 86\times$ ($48$ $\to$ $4{,}145$).

<!--en:31d4da85-->
::: {.slide title="Reddis konvexes Gegenbeispiel zur Konvergenz von Adam"}

<!--en:b72f7060-->
[Konvergenzgrenzen]{.kicker}

<!--en:5e7d86f7-->
Auf $x \in [-1, 1]$ gelten die konvexen Verlustfunktionen $(+4x, -x, -x)$ mit Periode 3. Ihre Steigungen summieren sich pro Periode zu $+2$, das eindeutige Optimum ist $x^\star = -1$:

<!--en:a75ecaf2-->
- **Kurzes EMA-Gedächtnis ($\beta_2 = 1/17, \beta_1 = 0$):** Bei der Folge $(+4, -1, -1)$ pendelt sich $\sqrt{v_t}$ auf $(3.89, 1.35, 1.02)$ ein. Die normierten Schritte $-1.03\eta, +0.74\eta, +0.98\eta$ ergeben eine Nettoverschiebung von **$+0.69\eta > 0$** zum schlechtesten Punkt $x = +1$.

<!--en:57a5cb31-->
- **Abhilfe durch AMSGrad:** Mit dem laufenden Maximum $\tilde{v}_t = \max(\tilde{v}_{t-1}, v_t)$ bleibt $\sqrt{\tilde{v}_t} = 3.89$ fest. Die Nettoverschiebung beträgt wieder $-0.51\eta$ in Richtung $x^\star = -1$.

<!--en:43db8fef-->
::: {.slide title="AMSGrad stellt mit laufenden Maxima die Konvergenz wieder her"}

<!--en:b72f7060-2-->
[Konvergenzgrenzen]{.kicker}

<!--en:de40e535-->
**AMSGrad** löst das Problem in Reddis Gegenbeispiel. Es ersetzt $\mathbf{v}_t$ im Nenner durch sein laufendes Maximum $\tilde{\mathbf{v}}_t = \max(\tilde{\mathbf{v}}_{t-1}, \mathbf{v}_t)$ und garantiert so, dass die effektiven Schrittweiten $\eta_t / \sqrt{\tilde{\mathbf{v}}_t}$ nicht zunehmen:

<!--en:0fa73d66-->
- Bleibt $\tilde{v}_t$ bei $15.12$, ergeben sich die drei Schritte $-1.03\eta, +0.26\eta, +0.26\eta$ (Nettoverschiebung um $-0.51\eta$ in Richtung $x^\star = -1$). Das entspricht SGD, während gewöhnliches Adam bei $x = +1$ hängen bleibt.

<!--en:1cb52370-->
::: {.slide title="Gewichtszerfall bei adaptiven Verfahren mit Kopplung und mit AdamW"}

<!--en:1fdef517-->
[Entkoppelter Gewichtszerfall]{.kicker}

<!--en:de06a05d-->
Bei einfachem SGD entspricht ein $\ell_2$-Strafterm $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ in der Verlustfunktion ($\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda\mathbf{w}_t$) genau der Multiplikation der Gewichte mit $(1 - \eta\lambda)$ in jedem Schritt. Bei Adam durchläuft $\lambda\mathbf{w}_t$ den Vorkonditionierer $\sqrt{\hat{\mathbf{v}}_t}$. Damit hängt der Gewichtszerfall von der Größenordnung des Gradienten ab:

<!--en:8073ff85-->
- **Nachteil des gekoppelten $\ell_2$-Zerfalls:** Koordinaten mit großen bisherigen Gradienten ($\sqrt{\hat{v}_{t,i}} \gg 1$) erfahren fast keinen Gewichtszerfall, während die Gewichte in ruhigen Koordinaten ($\sqrt{\hat{v}_{t,i}} \ll 1$) stark schrumpfen.

<!--en:c0526a26-->
- **Aktualisierung von AdamW (Loshchilov und Hutter, 2019):**
  $$\mathbf{w}_{t+1} = (1 - \eta\lambda)\,\mathbf{w}_t - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$
  Nur der Gradient des Datenverlusts geht in $\mathbf{m}_t$ und $\mathbf{v}_t$ ein. Die multiplikative Schrumpfung $(1 - \eta\lambda)$ wirkt gleichmäßig auf alle Parameter.

<!--en:339f1e6f-->
::: {.slide title="Wie gekoppelter Gewichtszerfall die Schrumpfung bei Rauschen verzerrt"}

<!--en:1fdef517-2-->
[Entkoppelter Gewichtszerfall]{.kicker}

<!--en:096e57bb-->
Simulation mit zwei Gruppen von je $10^4$ Gewichten mit Startwert $w_0 = 1$. Bei Rauschskalen von $\sigma_1 = 10$ und $\sigma_2 = 0.1$ erhalten sie über $T = 4{,}000$ Schritte reines gaußsches Gradientenrauschen mit Mittelwert null ($\eta = 10^{-3}, \lambda = 0.1$):

<!--en:c86a3eec-->
- Bei gekoppeltem Adam schrumpfen die schwach verrauschten Gewichte um den Faktor $100\times$ schneller ($10^{-3}$ gegenüber $10^{-5}$ pro Schritt, Endwerte etwa $0.03$ gegenüber $0.96$). AdamW verkleinert beide Gruppen gleichmäßig auf $(1 - \eta\lambda)^T \approx \exp(-0.4) \approx 0.670$.

<!--en:b5ab4f4f-->
::: {.slide title="Fixpunkte bei exakten Gradienten: MAP und ℓ∞-Quader"}

<!--en:1fdef517-3-->
[Entkoppelter Gewichtszerfall]{.kicker}

<!--en:fed116f0-->
Wenn sich die exakten Gradienten stabilisieren ($\hat{\mathbf{m}}_t \to \mathbf{u}, \hat{\mathbf{v}}_t \to \mathbf{u}^2$), konvergieren gekoppeltes Adam und AdamW gegen unterschiedliche stationäre Punkte:

<!--en:2b28feef-->
- **Gekoppeltes Adam:** Die Aktualisierung verschwindet genau dann, wenn $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ gilt (gaußsche MAP-Lösung).

<!--en:01558ae1-->
- **Entkoppeltes AdamW ($\epsilon \to 0$):** Erzwingt die KKT-Bedingungen für eine implizite $\ell_\infty$-Quaderbeschränkung $\min_{\|\mathbf{w}\|_\infty \le 1/\lambda} \ell(\mathbf{w})$.

<!--en:050a06db-->
::: {.slide title="Die ersten Schritte und die Aufwärmphase der Lernrate"}

<!--en:88464b8d-->
[Dynamik der Aufwärmphase]{.kicker}

<!--en:fae3482b-->
Im Schritt $t = 1$ liefert die Bias-Korrektur $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ und $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$. Adams allererste Aktualisierung ist daher in jeder Koordinate ein voller Vorzeichenschritt:

<!--en:ba615cc3-->
- **Warum reine Rauschkoordinaten um den Faktor $4\times$ weiter springen, wenn $t = 1$ gilt:** Besteht der Gradient einer Koordinate aus unabhängigem Rauschen $g_t \sim \mathcal{N}(0, \sigma^2)$ mit Mittelwert null, hat der Schritt bei $t = 1$ den quadratischen Mittelwert $1.00\,\eta$ für seinen Betrag.

<!--en:b389e874-->
- **Stationäre Varianz des Momentum-EMA:** Sobald $\mathbf{m}_t = (1 - \beta_1)\sum_{\tau \le t}\beta_1^{t-\tau}g_\tau$ über $\sim \frac{1}{1-\beta_1} = 10$ unabhängige Stichproben gemittelt hat, sinkt seine Varianz auf $\frac{1 - \beta_1}{1 + \beta_1}\sigma^2$. Im stationären Zustand beträgt das quadratische Mittel der Aktualisierung also nur:
  $$\text{RMS}\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}\right) \to \sqrt{\frac{1 - \beta_1}{1 + \beta_1}} = \sqrt{\frac{0.1}{1.9}} \approx 0.23$$

<!--en:0f7f8b3e-->
::: {.slide title="Anfängliche Instabilität durch Momentum und Verzerrung"}

<!--en:88464b8d-2-->
[Dynamik der Aufwärmphase]{.kicker}

<!--en:aca4ddae-->
Das quadratische Mittel der Aktualisierung (in Einheiten von $\eta$) über $20{,}000$ unabhängige Rauschkoordinaten zeigt getrennt die Wirkung des Momentumfensters, der Bias-Korrektur und der linearen Aufwärmphase:

<!--en:e787d57a-->
::: {.slide title="Anfängliche Instabilität durch Momentum und Verzerrung (Fortsetzung)"}

<!--en:88464b8d-3-->
[Dynamik der Aufwärmphase]{.kicker}

<!--en:dbfd61e5-->
- Ersetzt man $\hat{\mathbf{v}}_t$ durch die exakte Varianz $\sigma^2 = 1$, bleibt die $1.00 \to 0.23$-Kurve unverändert. Die anfangs zu großen Schritte in verrauschten Koordinaten entstehen durch das Momentumfenster von $10$ Schritten (eine Aufwärmphase von $20$ Schritten dämpft sie). Längere Aufwärmphasen in tiefen Netzen schützen vor hoher anfänglicher Krümmung $\lambda_{\max}$ der Hesse-Matrix.

<!--en:e154f377-->
::: {.slide title="Über die diagonale Skalierung hinaus: Metriken mit Kronecker-Faktoren"}

<!--en:03593d93-->
[Strukturierte Vorkonditionierer]{.kicker}

<!--en:603ac64b-->
In gedrehten Tälern mischt die Krümmung die Koordinaten. Um dort ohne den Speicheraufwand $O(d^2)$ eines vollen $d \times d$-Vorkonditionierers auszukommen, nutzen strukturierte Verfahren die **Matrixform** der Schichten neuronaler Netze:

<!--en:78c115b8-->
- **Schichtgradienten vom Rang eins:** Für eine lineare Schicht $\mathbf{y} = \mathbf{W}\mathbf{a}$ mit $\mathbf{W} \in \mathbb{R}^{m \times n}$, Eingabe $\mathbf{a} \in \mathbb{R}^n$ und Ausgabegradient $\boldsymbol{\delta} = \partial\ell/\partial\mathbf{y} \in \mathbb{R}^m$ ist der Gewichtsgradient $\mathbf{G} = \boldsymbol{\delta}\mathbf{a}^\top$. In gestapelter Vektorform lautet er $\mathrm{vec}(\mathbf{G}) = \mathbf{a} \otimes \boldsymbol{\delta}$.

<!--en:27c59ab3-->
- **Kronecker-Faktorisierung:** Nähert man die $(mn) \times (mn)$-Krümmungsmatrix aus dem äußeren Produkt $\mathbb{E}[(\mathbf{a}\mathbf{a}^\top) \otimes (\boldsymbol{\delta}\boldsymbol{\delta}^\top)]$ durch $\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}$ an (mit $\mathbf{S}_{\mathrm{in}} = \mathbb{E}[\mathbf{a}\mathbf{a}^\top] \in \mathbb{R}^{n \times n}$ und $\mathbf{S}_{\mathrm{out}} = \mathbb{E}[\boldsymbol{\delta}\boldsymbol{\delta}^\top] \in \mathbb{R}^{m \times m}$), lässt sich der Vorkonditionierer mit zwei kleinen Matrixprodukten invertieren und anwenden:
  $$\big(\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}\big)^{-1}\mathrm{vec}(\mathbf{V}) = \mathrm{vec}\big(\mathbf{S}_{\mathrm{out}}^{-1}\,\mathbf{V}\,\mathbf{S}_{\mathrm{in}}^{-1}\big)$$

<!--en:d1ed3954-->
::: {.slide title="Speicher- und Rechenersparnis durch Kronecker-Faktorisierung"}

<!--en:03593d93-2-->
[Strukturierte Vorkonditionierer]{.kicker}

<!--en:9822a076-->
Für eine einzelne $4096 \times 4096$-Gewichtsmatrix eines Transformers ($d = mn \approx 1.68 \times 10^7$ Parameter) die volle $(mn) \times (mn)$-Krümmungsmatrix mit zwei $4096 \times 4096$-Kronecker-Faktoren vergleichen:

<!--en:6514b2d5-->
- Die Kronecker-Faktorisierung verringert den Speicherbedarf um den Faktor **$8.4 \times 10^6\times$** (von $2.8 \times 10^{14}$ Einträgen auf $3.4 \times 10^7$) und den Aufwand für die Inversion um den Faktor **$3.4 \times 10^{10}\times$** (von $4.7 \times 10^{21}$ FLOPs auf $1.4 \times 10^{11}$ FLOPs).

<!--en:9548be14-->
::: {.slide title="Vorkonditionierung mit K-FAC, Shampoo und Muon"}

<!--en:03593d93-3-->
[Strukturierte Vorkonditionierer]{.kicker}

<!--en:c4bc3293-->
Moderne Matrixoptimierer wenden die beidseitige Vorkonditionierung $\mathbf{G} \mapsto \mathbf{P}_{\mathrm{l}}^{-a}\mathbf{G}\mathbf{P}_{\mathrm{r}}^{-a}$ auf eine $m \times n$-Gradienten- oder Momentummatrix $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ an:

<!--en:79e6fe78-->
- **K-FAC ($a = 1$) und Shampoo ($a = 1/4$):** Nähern die Fisher-Faktoren beziehungsweise die Faktoren von Matrix-AdaGrad über Zeilen- und Spaltenkovarianzen an.

<!--en:14ca6977-->
- **Muon (Polarfaktor $\mathbf{U}\mathbf{V}^\top$):** Setzt mit schnellen Newton-Schulz-Iterationen alle Singulärwerte auf $1$ und vertauscht mit beliebigen orthogonalen Drehungen.

<!--en:c798e3a3-->
::: {.slide title="Verständnisfragen zu Skaleninvarianz, AdamW und Muon"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:d5f3fdb4-->
Das Verständnis adaptiver und mit Matrizen vorkonditionierter Optimierer prüfen:

<!--en:7e3f3b1c-->
1. Wie weit bewegt Adam mit Bias-Korrektur im Schritt $t = 1$ bei $\epsilon = 10^{-8}$ eine Koordinate mit Gradient $g_{1,i} = 10^{-6}$, und nach wie vielen Schritten halbiert sich bei $\beta_2 = 0.999$ das Gewicht eines früheren quadrierten Gradienten?

<!--en:3b6989ab-->
2. Wo liegt für die eindimensionale Verlustfunktion $\ell(w) = \tfrac12(w - 3)^2$ mit exakten Gradienten und $\lambda = 1$ der stationäre Fixpunkt von gekoppeltem Adam ($\ell_2$-Strafterm), wo der von entkoppeltem AdamW ($\epsilon \to 0$)?

<!--en:6fce0983-->
3. Warum verliert Adams diagonale Skalierung bei einer Drehung des $1000{:}1$-Tals um $45^\circ$ ihre Wirkung, während Muons Polarfaktor $\mathbf{U}\mathbf{V}^\top$ äquivariant bleibt?

<!--en:399b9ffa-->
**Lösungen:** **(1)** Bei $t = 1$ gilt $\hat{m}_{1,i}/\sqrt{\hat{v}_{1,i}} = \mathrm{sign}(g_{1,i})$, die Schrittweite beträgt also $\eta \frac{10^{-6}}{10^{-6} + 10^{-8}} \approx \mathbf{0.99\,\eta}$. Die Halbwertszeit für $\beta_2 = 0.999$ ist $\frac{\log 2}{1 - 0.999} \approx \mathbf{693\text{ steps}}$. **(2) Gekoppeltes Adam** löst $(w - 3) + \lambda w = 0 \implies w^\star = \frac{3}{1+1} = \mathbf{1.5}$. **AdamW** begrenzt den unbeschränkten Minimierer $w = 3$ auf den $\ell_\infty$-Quader $|w| \le 1/\lambda = 1$ und kommt bei $w^\star = \mathbf{1.0}$ zur Ruhe. **(3)** Eine Drehung um $45^\circ$ mischt die Koordinaten. Die steile Eigenkomponente wirkt dann auf beide Achsen ($\hat{v}_1 \approx \hat{v}_2$), und Adams diagonaler Vorkonditionierer reduziert sich auf ein skalares Vielfaches der Einheitsmatrix. Bei orthogonalen Drehungen $\mathbf{Q}_1\mathbf{G}\mathbf{Q}_2^\top = (\mathbf{Q}_1\mathbf{U})\boldsymbol{\Sigma}(\mathbf{Q}_2\mathbf{V})^\top$ geht der Polarfaktor dagegen in $\mathbf{Q}_1(\mathbf{U}\mathbf{V}^\top)\mathbf{Q}_2^\top$ über. Er dreht sich also mit der Basis mit.

<!--en:912429bf-->
::: {.slide title="Zusammenfassung: adaptive und strukturierte Vorkonditionierer"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:b5b0979a-->
| Optimierer | Vorkonditionierer / Aktualisierung | Invarianz | Zusätzlicher Speicherbedarf | Stärken und Grenzen |

<!--en:96274ca8-->
| **AdaGrad** | $\mathbf{s}_t = \sum_{\tau \le t}\mathbf{g}_\tau^2$, Schritt $\propto \mathbf{s}_t^{-1/2}\mathbf{g}_t$ | Diagonale Koordinatenskalierung | $1$ Vektor ($d$) | Optimaler Regret bei dünn besetzten konvexen Problemen. Abnahme gemäß $1/\sqrt{t}$ bremst tiefe Netze aus |

<!--en:b5b66ecb-->
| **RMSProp** | $\mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$ | Diagonale Koordinatenskalierung | $1$ Vektor ($d$) | Dauerhaft positive Schritte. Ohne Momentum-Glättung verrauscht |

<!--en:d7397442-->
| **Adam / AMSGrad** | Mit Bias-Korrektur $\hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ | Exakte Invarianz bei diagonaler Skalierung ($\epsilon=0$) | $2$ Vektoren ($2d$) | Schnell und robust in achsenparallelen Tälern. Versagt bei Drehung um $45^\circ$ |

<!--en:a17d0641-->
| **AdamW** | Entkoppelt: $(1-\eta\lambda)\mathbf{w}_t - \eta\hat{\mathbf{m}}_t/\sqrt{\hat{\mathbf{v}}_t}$ | Gleichmäßiger Zerfall bei jeder Skalierung | $2$ Vektoren ($2d$) | Standard für Transformer. Fixpunkte im $\ell_\infty$-Quader $\|\mathbf{w}\|_\infty \le 1/\lambda$ |

<!--en:1a64a295-->
| **K-FAC / Shampoo** | Beidseitig: $\mathbf{P}_{\mathrm{l}}^{-a}\mathbf{G}\mathbf{P}_{\mathrm{r}}^{-a}$ ($m \times n$-Schicht) | Zeilen- und Spaltentransformationen der Schicht | $m^2 + n^2$ pro Schicht | Erfasst Korrelationen zwischen Koordinaten. Periodische Berechnung von Matrixwurzeln beziehungsweise Inversen |

<!--en:6fe15f13-->
| **Muon** | Orthogonaler Polarfaktor $\mathbf{U}\mathbf{V}^\top$ von $\mathbf{M}_t$ | Orthogonale Zeilen- und Spaltendrehungen | $1$ Matrix ($mn$) | Gleicht Singulärwerte durch schnelle Matrixmultiplikationen nach Newton-Schulz an |
