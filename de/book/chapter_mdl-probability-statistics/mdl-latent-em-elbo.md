<!--en:99d95690-->
# Latente Variablen, EM und die ELBO

<!--en:de44f430-->
Jede Likelihood in :numref:`sec_mdl-maximum_likelihood` war ein Produkt mit einem Faktor pro Beobachtung. Durch Logarithmieren erhielten wir eine Summe, die wir gliedweise ableiten konnten. Dabei haben wir stillschweigend vorausgesetzt, dass wir alles beobachten, was das Modell erzeugt.

<!--en:86822ac4-->
In der Praxis fehlt uns oft ein entscheidender Teil der Daten. Stellen Sie sich vor, Sie erheben $500$ Messwerte aus zwei verschiedenen Populationen, $300$ aus der einen und $200$ aus der anderen. Danach stellen Sie fest, dass niemand vermerkt hat, welcher Messwert aus welcher Population stammt. Mit diesen Labels ließe sich das Modell in einer Zeile pro Population anpassen: Man bräuchte nur den Mischungsanteil, den Mittelwert und die Varianz. Ohne die Labels ergibt sich die Wahrscheinlichkeit jedes Messwerts als Summe über die beiden Populationen, aus denen er stammen könnte. Der Logarithmus einer Summe lässt sich jedoch nicht zerlegen. Jeder Parameter geht in jeden Summanden ein, und wenn wir den Gradienten gleich null setzen, erhalten wir Gleichungen ohne geschlossene Lösung. Auf dieselbe Schwierigkeit stoßen wir immer dann, wenn ein Modell Beobachtungen durch verborgene Strukturen erklärt. Beispiele sind die Clusterzugehörigkeit eines Kunden, das Thema eines Dokuments oder der zufällige Code, aus dem ein Generator ein Bild erzeugt. Solche unbeobachteten Größen heißen *latente Variablen*.

<!--en:55750c2b-->
Die Grundidee dieses Abschnitts ist einfach und überraschend wirksam. Was wir nicht beobachten können, schätzen wir durch Wahrscheinlichkeiten statt durch feste Labels. Dann passen wir das Modell so an, als wären diese weichen Zuordnungen beobachtete Daten. Das funktioniert selbst dann, wenn unsere Schätzung nicht stimmt. Für *jede* probabilistische Schätzung bildet die so gewonnene einfache Zielfunktion eine untere Schranke für die wahre Log-Likelihood. Sie berührt die Log-Likelihood genau dann, wenn die Schätzung mit der A-posteriori-Verteilung (engl. posterior distribution) der verborgenen Labels übereinstimmt. Diese untere Schranke ist das zentrale Ergebnis. Der Expectation-Maximization-Algorithmus (EM) erhöht sie direkt. Ist die exakte A-posteriori-Verteilung unzugänglich, erhöhen die variationelle Inferenz (engl. variational inference) und Variationsautoencoder (engl. variational autoencoders) die Schranke näherungsweise.

<!--en:ba5cfc29-->
Wir verfolgen diese Idee von Anfang bis Ende. Zuerst formulieren wir Mischmodelle (engl. mixture models). Dann leiten wir die untere Schranke her und bestimmen ihren Abstand zur Log-Likelihood. Schließlich wenden wir den so gewonnenen Algorithmus auf unsere $500$ Punkte an. Von einem Startpunkt aus findet er beide Populationen wieder, bei einem symmetrischen Start bleibt er stecken.

<!--en:7861a543-->
## Mischmodelle

<!--en:430bb585-->
Drücken wir das Problem der fehlenden Labels in Formeln aus. Ein probabilistisches Modell mit einer latenten Variablen $z$ gibt eine gemeinsame Verteilung für die verborgenen und die beobachteten Größen an,

<!--en:0ea4afaa-->
von denen wir nur $x$ beobachten. Um die Likelihood der tatsächlich beobachteten Daten zu berechnen, summieren wir über alle möglichen Werte der verborgenen Variablen (oder integrieren über sie, wenn $z$ stetig ist). So erhalten wir die *marginale* Likelihood:

<!--en:5bae3ffe-->
Die Wahrscheinlichkeit für $x$ ergibt sich also aus allen Möglichkeiten, wie das Modell $x$ erzeugen könnte. Jeder Wert der verborgenen Variablen trägt einen Summanden bei.

<!--en:394f126e-->
Das klassische Beispiel ist das **Gauß-Mischmodell** (engl. Gaussian mixture model, GMM). Hier wählt die latente Variable $z \in \{1,\ldots,K\}$ die Komponente $k$ mit der Mischungswahrscheinlichkeit $\pi_k$. Die Beobachtung ziehen wir dann aus der Gauß-Verteilung dieser Komponente, $x \mid z{=}k \sim \mathcal{N}(\mu_k, \sigma_k^2)$. Für die $n$ Beobachtungen zusammen ergibt sich die negative Log-Likelihood (NLL)

<!--en:53ef384b-->
Die Schwierigkeit liegt allein darin, dass die Summe *innerhalb* des Logarithmus steht. Der Logarithmus wirkt auf die Mischung als Ganzes. Auf die einzelnen Gauß-Dichten angewandt, würde er die Exponentialfunktion aufheben, sodass nur ein quadratischer Ausdruck übrig bliebe. Betrachten wir zum Vergleich das Problem mit *vollständigen Daten*. Würde uns ein Orakel jedes Label $z_i$ verraten, ließe sich die Likelihood nach Komponenten zerlegen. Die Maximum-Likelihood-Schätzung würde auf bekannte Schätzer hinauslaufen. Jedes Mischungsgewicht $\pi_k$ wäre einfach der Anteil der Punkte, die zur Komponente $k$ gehören, ganz wie beim Schätzer $\hat\theta = n_H/n$ für die Kopfwahrscheinlichkeit einer Münze in :numref:`sec_mdl-maximum_likelihood`. Für $\mu_k$ und $\sigma_k^2$ erhielten wir den üblichen Stichprobenmittelwert und die Stichprobenvarianz dieser Punkte. Weil das Problem nur bei fehlenden Labels schwierig ist, liegt es nahe, für die fehlenden Labels Wahrscheinlichkeiten zu schätzen. Anschließend können wir das einfache Problem für vollständige Daten lösen.

<!--en:b828adc4-->
Wie sieht eine solche probabilistische Schätzung für einen einzelnen Punkt aus? Betrachten wir eine Beobachtung bei $x = 0.5$ und zwei gleich gewichtete Komponenten, $\mathcal{N}(-1, 1)$ und $\mathcal{N}(1, 1)$. Ihre Dichten bei $x = 0.5$ sind proportional zu $e^{-(1.5)^2/2} = e^{-1.125}$ beziehungsweise $e^{-(0.5)^2/2} = e^{-0.125}$. Nach dem Satz von Bayes erhält die linke Komponente daher die A-posteriori-Wahrscheinlichkeit

<!--en:8a5b272b-->
Der Satz von Bayes verlangt keine harte Entscheidung. Er ordnet den Punkt zu $27\%$ der linken und zu $73\%$ der rechten Komponente zu. Verschieben wir die Beobachtung weit nach rechts, etwa zu $x = 3$, so gehört sie mit einer A-posteriori-Wahrscheinlichkeit von $99.8\%$ zur rechten Komponente. In der Mitte bei $x = 0$ sind beide Anteile mit $50\%$ und $50\%$ gleich groß. Wo die Daten eindeutig sind, fällt auch die weiche Zuordnung eindeutig aus. Wo sich die Komponenten überlappen, spiegelt sie die Unsicherheit wider (:numref:`fig_mdl-prob-soft-resp`).

<!--en:2570b233-->
![Weiche Zuordnung anhand des Verhältnisses der Dichtewerte für zwei gleich gewichtete Komponenten $\mathcal{N}(-1,1)$ und $\mathcal{N}(1,1)$. (a) Bei $x=0.5$ haben die Dichten die Werte $h_1$ und $h_2$. Auf die linke Komponente entfällt der Anteil $r_1=h_1/(h_1+h_2)=0.27$. (b) Die beiden A-posteriori-Wahrscheinlichkeiten $r_1(x)$ und $r_2(x)$ sind bei $x=0$ gleich groß. Sie betragen $0.27$ und $0.73$ bei $x=0.5$. Auf die rechte Komponente entfällt ein Anteil von $0.998$ bei $x=3$.](../img/mdl-prob-soft-resp.svg)

<!--en:fad3c258-->
## Die untere Evidenzschranke

<!--en:8d71fdf4-->
Wie gewinnen wir aus dieser intuitiven Strategie einen Algorithmus mit mathematischer Garantie? Wir konstruieren eine untere Schranke für die Log-Likelihood, bei der der Logarithmus wieder innerhalb der Summe steht. Sei $q(z)$ eine *beliebige* Wahrscheinlichkeitsverteilung der latenten Variablen, die unsere weiche Zuordnung beschreibt. In der marginalen Likelihood erweitern wir jeden Summanden mit $q(z)$. Dann wenden wir die Jensen-Ungleichung an. Da der Logarithmus konkav ist, gilt $\log \mathbb{E}[\cdot] \ge \mathbb{E}[\log(\cdot)]$ (:numref:`subsec_mdl-jensen`). Damit erhalten wir

<!--en:15a82d7d-->
Hier laufen die Summen über alle $z$ mit $q(z) > 0$. Weist $q$ einem Wert $z$ die Wahrscheinlichkeit null zu, obwohl die gemeinsame Verteilung dort positiv ist, wird aus dem ersten Gleichheitszeichen ein $\ge$. Die Schranke gilt also weiterhin. Die marginale Likelihood $p(x\mid\boldsymbol{\theta})$ spielt für die latente Variable $z$ dieselbe Rolle wie die Evidenz $P(X)$ (der Nenner im Satz von Bayes in :numref:`subsec_mdl-map`) für die Modellparameter. Sie gibt die Wahrscheinlichkeit der beobachteten Daten an, nachdem wir über alle unbeobachteten Größen summiert haben. Daher heißt sie auch *Evidenz*. Die rechte Seite $\mathcal{L}(q, \boldsymbol{\theta})$ heißt **untere Evidenzschranke** (engl. evidence lower bound, **ELBO**). Zerlegen wir den Logarithmus des Quotienten, so erhalten wir zwei leicht verständliche Terme für die ELBO:

<!--en:b7a8d92b-->
Der erste Term ist die erwartete Log-Likelihood der *vollständigen Daten*, also unsere einfache Zielfunktion, gemittelt über die weiche Zuordnung $q$. Der zweite Term ist die Entropie dieser Zuordnung. Sie hängt überhaupt nicht von $\boldsymbol{\theta}$ ab. Wir können sogar den Abstand zwischen der ELBO und der wahren Log-Likelihood exakt angeben. Er ist gleich einer KL-Divergenz. Auf dieser einen Identität beruhen sowohl EM als auch die variationelle Inferenz.

<!--en:6d599390-->
**Proposition (der Abstand zwischen ELBO und Log-Likelihood ist eine KL-Divergenz).** *Für jedes $\boldsymbol{\theta}$ und jede Verteilung $q$, deren gesamte Masse auf dem Träger der A-posteriori-Verteilung $p(z\mid x,\boldsymbol{\theta})$ liegt, gilt*

<!--en:1b8e7035-->
*Damit gilt $\mathcal{L}(q,\boldsymbol{\theta}) \le \log p(x\mid\boldsymbol{\theta})$. Gleichheit gilt genau dann, wenn $q$ mit der A-posteriori-Verteilung $p(z\mid x,\boldsymbol{\theta})$ übereinstimmt.*

<!--en:d52523c1-->
**Beweis.** Setzen wir den Satz von Bayes, $p(z\mid x,\boldsymbol{\theta}) = p(x,z\mid\boldsymbol{\theta})/p(x\mid\boldsymbol{\theta})$, direkt in die Definition der KL-Divergenz ein:

<!--en:42c05bfb-->
Dabei lässt sich $\log p(x\mid\boldsymbol{\theta})$ aus dem Erwartungswert herausziehen, weil der Ausdruck nicht von $z$ abhängt. Umstellen liefert :eqref:`eq_mdl-elbo-gap`. Nach der Gibbs-Ungleichung (:numref:`subsec_mdl-max-entropy-derivation`) ist die KL-Divergenz stets nichtnegativ. Sie verschwindet genau dann, wenn ihre beiden Argumente übereinstimmen. $\blacksquare$

<!--en:f9d09604-->
Prüfen wir diese Identität am Beispiel $x = 0.5$ aus dem vorigen Unterabschnitt. Dort beträgt die wahre Log-Evidenz $\log p(x) = -1.42$. Gewichten wir mit $q = (\tfrac{1}{2}, \tfrac{1}{2})$ beide Komponenten gleich, so ergibt sich für die ELBO der Wert $-1.54$. Die fehlenden $0.12$ Nat entsprechen genau der KL-Divergenz von $(\tfrac{1}{2}, \tfrac{1}{2})$ zur A-posteriori-Verteilung $(0.27, 0.73)$. Die Schranke liegt also nur um den Betrag unter der Log-Evidenz, um den unsere Schätzung im Sinne der KL-Divergenz von der wahren A-posteriori-Verteilung abweicht. :numref:`fig_mdl-prob-elbo` zeigt, wie sich die untere Schranke verhält, wenn wir $\boldsymbol{\theta}$ variieren.

<!--en:2d31f30c-->
![Die Log-Evidenz und ihre untere Schranke. Für eine feste Schätzung $q$ liegt die ELBO $\mathcal{L}(q,\theta)$ auf oder unter der Kurve der Log-Evidenz $\log p(x\mid\theta)$, und zwar für jedes $\theta$. Der vertikale Abstand zwischen den Kurven ist die KL-Divergenz von $q$ zur A-posteriori-Verteilung $p(z\mid x,\theta)$. Im E-Schritt von EM erhält $q$ den Wert der A-posteriori-Verteilung am aktuellen Iterationspunkt $\theta^{(t)}$. Damit verschwindet der Abstand, und die Schranke berührt dort die Log-Evidenz. Der M-Schritt führt anschließend zum Maximum der Schranke bei $\theta^{(t+1)}$. Da die Schranke die Log-Evidenz bei $\theta^{(t)}$ berührt und nirgends über ihr liegt, ist die Log-Evidenz bei $\theta^{(t+1)}$ mindestens so hoch wie bei $\theta^{(t)}$.](../img/mdl-prob-elbo.svg)

<!--en:c259d428-->
## Expectation-Maximization

<!--en:896b4d10-->
Die ELBO-Identität macht aus unserem schwierigen Optimierungsproblem einen **Koordinatenaufstieg** (engl. coordinate ascent) für $\mathcal{L}(q, \boldsymbol{\theta})$. Wir maximieren bezüglich eines Arguments und halten das andere fest. Dann tauschen wir die Rollen und wiederholen das Verfahren. Für einen Datensatz $X = \{x_1,\ldots,x_n\}$ mit unabhängigen Beobachtungen gilt die Schranke für jede Beobachtung einzeln. Wir addieren die Schranken über den Datensatz und verwenden pro Datenpunkt eine Schätzung $q_i$. Der **Expectation-Maximization-Algorithmus** (EM) wechselt zwischen diesen beiden Koordinaten :cite:`Dempster.Laird.Rubin.1977`:

<!--en:787bd987-->
* **E-Schritt.** Wir halten $\boldsymbol{\theta}^{(t)}$ fest und maximieren über die Schätzungen $q_i$. Aus der Bedingung für Gleichheit in :eqref:`eq_mdl-elbo-gap` folgt, dass die exakte A-posteriori-Verteilung jedes Punkts optimal ist: $q_i^{(t)}(z) = p(z \mid x_i, \boldsymbol{\theta}^{(t)})$. Damit verschwindet jede KL-Divergenz, und die Schranke wird bei den aktuellen Parametern *angenommen*: $\sum_i \mathcal{L}(q_i^{(t)}, \boldsymbol{\theta}^{(t)}) = \log P(X\mid\boldsymbol{\theta}^{(t)})$.

<!--en:00938340-->
* **M-Schritt.** Wir halten die Schätzungen $q_i^{(t)}$ fest und maximieren über die Parameter $\boldsymbol{\theta}$:

  $$
  \boldsymbol{\theta}^{(t+1)} = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}} \sum_i \mathbb{E}_{q_i^{(t)}}\big[\log p(x_i, z\mid\boldsymbol{\theta})\big],
  $$

  Die Entropieterme $H(q_i^{(t)})$ entfallen, weil sie nicht von $\boldsymbol{\theta}$ abhängen. Damit haben wir wieder die einfache Zielfunktion für vollständige Daten. Wahrscheinlichkeiten treten an die Stelle der fehlenden Labels, die uns ein Orakel liefern müsste.

<!--en:090700c3-->
Warum können wir uns auf diese abwechselnden Schritte verlassen? Die Geometrie in :numref:`fig_mdl-prob-elbo` garantiert, dass jeder einzelne Schritt die wahre Log-Likelihood erhöht oder sie im ungünstigsten Fall unverändert lässt.

<!--en:8be9f0e1-->
**Proposition (EM verringert die Likelihood nie).** *Für jede EM-Iteration gilt $\log P(X\mid\boldsymbol{\theta}^{(t+1)}) \ge \log P(X\mid\boldsymbol{\theta}^{(t)})$.*

<!--en:84ab2315-->
**Beweis.** Wir verknüpfen drei einfache Beziehungen:

<!--en:2c65b5d8-->
Die erste Ungleichung folgt aus der unteren Evidenzschranke :eqref:`eq_mdl-elbo` bei den neuen Parametern. Die zweite gilt, weil der M-Schritt die Schranke über $\boldsymbol{\theta}$ maximiert. Die abschließende Gleichheit gilt, weil der E-Schritt den Abstand auf null gebracht hat. $\blacksquare$

<!--en:96bcb481-->
Für unser Gauß-Mischmodell haben beide Schritte eine geschlossene Form. Im E-Schritt berechnen wir die **Zugehörigkeitswahrscheinlichkeiten** (engl. responsibilities). Sie geben a posteriori an, mit welcher Wahrscheinlichkeit Punkt $i$ von Komponente $k$ stammt. Für $x = 0.5$ haben wir das bereits von Hand berechnet:

<!--en:b89ce092-->
Im M-Schritt passen wir jede Komponente neu an. Beim Zählen und bei der Mittelwertbildung gewichten wir jeden Punkt mit seiner Zugehörigkeitswahrscheinlichkeit. Schreiben wir $n_k = \sum_i r_{ik}$ für die effektive Anzahl der Punkte, die wir der Komponente $k$ zuordnen, so erhalten wir

<!--en:5823e2ee-->
Die Formeln kennen wir bereits. Es sind die üblichen Schätzer für vollständige Daten (Mischungsanteil, Stichprobenmittelwert und Stichprobenvarianz), jetzt mit weichen Zuordnungen. Jeder Datenpunkt gibt seine Stimme anteilig an die Komponenten ab, entsprechend seinen Zugehörigkeitswahrscheinlichkeiten. Bei einer harten Zuordnung erhielte eine einzige Komponente die ganze Stimme. Geben wir allen Komponenten dieselbe feste Varianz $\sigma^2$ vor und betrachten den Grenzfall $\sigma^2 \to 0$, so werden aus den Zugehörigkeitswahrscheinlichkeiten harte $0/1$-Zuordnungen zum nächsten Mittelwert. Damit geht EM direkt in den K-means-Algorithmus über.

<!--en:49ccbd05-->
Der gesamte Algorithmus umfasst nur ein Dutzend Codezeilen. Erzeugen wir $300$ Punkte aus $\mathcal{N}(-2, 0.8^2)$ und $200$ aus $\mathcal{N}(1.5, 0.6^2)$. Wir schreiben eine kurze Funktion `em` für eine eindimensionale Mischung aus $K$ Gauß-Verteilungen. Sie zeichnet die Log-Likelihood in jedem Schritt und die zuletzt geschätzten Parameter auf.

<!--en:5829deca-->
Unsere Proposition garantiert, dass die Log-Likelihood nie sinkt. Wie schnell sie mit EM steigt, bleibt offen. Beginnen wir mit einer recht ungünstigen Schätzung: zwei stark überlappenden Komponenten mit Varianz eins und Mittelwerten bei $\pm 0.5$. Wir führen $30$ Iterationen aus und betrachten die ersten sechs Werte der Log-Likelihood sowie den Wert im letzten Schritt.

<!--en:65eaf735-->
Die Log-Likelihood steigt um rund $400$ Nat, von etwa $-1{,}280$ auf etwa $-860$ (beide Werte schwanken je nach Zufallsstichprobe um einige Dutzend Nat). Fast der gesamte Anstieg entfällt auf die ersten Schritte. Schon nach sechs Schritten liegt die Log-Likelihood bis auf etwa ein Nat bei ihrem Endwert. Die geschätzten Gewichte, Mittelwerte und Standardabweichungen weichen um höchstens etwa $0.2$ von den wahren Parametern $(0.6, 0.4)$, $(-2, 1.5)$ und $(0.8, 0.6)$ ab, mit denen wir die Daten erzeugt haben.

<!--en:1e99e125-->
Trotz dieser schnellen Konvergenz hat EM vier Einschränkungen, die wir kennen sollten. Die erste ist harmlos. Vertauschen wir die Labels der $K$ Komponenten, so erhalten wir $K!$ gleichwertige Kopien jedes Maximums. Dieser *Labeltausch* (engl. label switching) ist dieselbe Art der Nichtidentifizierbarkeit, die beim Vertauschen versteckter Einheiten in einem neuronalen Netz auftritt. Alle so entstehenden Lösungen sind gleich gut.

<!--en:6b398a63-->
Die zweite Einschränkung hat größere Folgen. EM bleibt am ersten stationären Punkt stehen, den es erreicht, unabhängig davon, ob dort ein globales Maximum liegt. Besonders deutlich zeigt das eine symmetrische Initialisierung. Überlegen Sie vor dem Ausführen der nächsten Zelle: Was geschieht, wenn beide Komponenten mit genau demselben Mittelwert und derselben Standardabweichung starten?

<!--en:9f455298-->
Die beiden Komponenten trennen sich nie. Da sie identisch starten, erhalten sie an jedem einzelnen Datenpunkt dieselben Zugehörigkeitswahrscheinlichkeiten. Auch nach dem M-Schritt bleiben sie daher identisch. EM findet die Lösung für eine einzige Gauß-Verteilung mit dem Mittelwert und der Standardabweichung des gesamten Datensatzes. Ihre Log-Likelihood liegt mehr als $100$ Nat unter der Lösung mit zwei Clustern. Jeder Fixpunkt von EM ist ein stationärer Punkt der Likelihood, aber nicht unbedingt ein Maximum. In der Praxis startet man EM deshalb mehrmals mit verschiedenen Zufallsinitialisierungen. Auch unter den versteckten Einheiten einer Schicht eines neuronalen Netzes bricht die Zufallsinitialisierung genau diese Symmetrie (:numref:`sec_numerical_stability`). Einheiten mit gleichen Anfangswerten erhalten gleiche Gradienten und bleiben bei jedem Schritt identisch.

<!--en:9f7906c0-->
Die dritte Einschränkung ist eine Besonderheit stetiger Mischverteilungen. Die Likelihood selbst ist nach oben unbeschränkt. Zentrieren wir die zweite Komponente auf dem ersten Datenpunkt, $\mu_2 = x_1$, mit Gewicht $\pi_2 = 1/n$ und Standardabweichung $\epsilon$. Ihr gewichteter Dichtebeitrag bei $x_1$ beträgt $\tfrac{1}{n}(2\pi)^{-1/2}\epsilon^{-1}$. Die erste Komponente deckt weiterhin die restlichen $n - 1$ Punkte ab. Damit gilt

<!--en:a851b98a-->
wobei die Konstante den Beitrag der übrigen $n - 1$ Punkte zusammenfasst. Für $\epsilon \to 0$ fällt die negative Log-Likelihood gegen $-\infty$. Die globale Maximum-Likelihood-„Lösung“ ist eine unendlich schmale Dichtespitze über einem einzigen Datenpunkt (:numref:`fig_mdl-prob-gmm-spikes`). Eine positive Untergrenze für die Varianzen oder eine geeignete A-priori-Verteilung für sie verhindert solche entarteten Spitzen. Ist die Menge der Parameter, deren Zielfunktionswert mindestens so gut ist wie der Startwert, kompakt, so ist jeder Häufungspunkt von EM ein stationärer Punkt des regularisierten Problems :cite:`Wu.1983`.

<!--en:3f1be359-->
![Das entartete Maximum der Likelihood eines Gauß-Mischmodells. Die erste Komponente (blau) deckt $n-1$ Datenpunkte (Punkte) ab. Die zweite (orange) sitzt auf dem verbleibenden Punkt, $\mu_2=x_1$, und hat die Standardabweichung $\sigma_2=\epsilon$. Sinkt $\epsilon$ von $0.3$ über $0.1$ auf $0.03$, so wächst ihre Dichte bei $x_1$ wie $1/\epsilon$. Mit ihr wächst die Likelihood über alle Grenzen.](../img/mdl-prob-gmm-spikes.svg)

<!--en:ec44357c-->
Die vierte Einschränkung führt uns über EM hinaus. Für den Monotoniebeweis mussten wir im E-Schritt die *exakte* A-posteriori-Verteilung einsetzen. Sobald wir diesen Schritt nur näherungsweise ausführen können, entfällt die Garantie. Ein *Variationsautoencoder* :cite:`Kingma.Welling.2014` verwendet beispielsweise dieselbe ELBO als Zielfunktion. Da ein tiefer nichtlinearer Generator keine A-posteriori-Verteilung in geschlossener Form liefert, ersetzt ein neuronales Netz als Encoder $q_{\boldsymbol{\phi}}(z\mid x)$ die exakte Verteilung für jeden Punkt. Wir trainieren den Encoder gemeinsam mit dem Decoder. Das Verfahren erhöht die ELBO mit stochastischen Gradienten statt mit exakten EM-Schritten.

<!--en:8e333f01-->
Die Schranke reicht also über den Algorithmus hinaus, mit dem wir sie eingeführt haben. Die ELBO verlangt nirgends, dass die verborgene Variable $z$ ein Clusterlabel ist. Setzen wir an die Stelle von $z$ die Modellparameter selbst, so benötigt der E-Schritt die A-posteriori-Verteilung der Parameter. Genau diese Verteilung fehlte uns am Ende von :numref:`sec_mdl-maximum_likelihood`. Auch sie liegt selten in geschlossener Form vor. In :numref:`sec_mdl-bayesian-computation` untersuchen wir daher praktische Verfahren, mit denen wir sie annähern und Stichproben aus ihr ziehen können.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:eeae3131-->
Enthält ein Modell verborgene Variablen, so steht beim Logarithmieren der marginalen Likelihood eine Summe innerhalb des Logarithmus. Aus einfachem Zählen und Mitteln wird dadurch ein Optimierungsproblem ohne geschlossene Lösung. Mit einer probabilistischen Schätzung $q$ für die verborgenen Variablen gewinnen wir die einfache Zielfunktion für vollständige Daten als untere Schranke zurück. Die ELBO liegt genau um die KL-Divergenz von $q$ zur wahren A-posteriori-Verteilung der latenten Variablen unter der Log-Evidenz. EM bringt diesen Abstand im E-Schritt auf null und maximiert im anschließenden M-Schritt die so gewonnene Schranke über die Parameter. In keiner Iteration sinkt die Likelihood, doch EM kann an einem stationären Punkt stehen bleiben, der kein Optimum ist. Bei stetigen Mischverteilungen braucht EM eine Untergrenze oder eine A-priori-Verteilung für die Varianzen, um entartete Dichtespitzen zu vermeiden. Die Monotonie ist nur bei exakter A-posteriori-Verteilung garantiert.

<!--en:2294243a-->
## Übungen

<!--en:567c5f51-->
1. [code] **Geschwindigkeit von EM und fehlende Information.** ● Betrachten Sie die Mischung $p(x\mid\pi)=\pi f_0(x)+(1-\pi)f_1(x)$ mit den bekannten Komponenten $f_0=\mathcal N(\cdot\,;0,1)$ und $f_1=\mathcal N(\cdot\,;\Delta,1)$. Nur das Mischungsgewicht $\pi$ ist unbekannt.
    1. Zeigen Sie, dass EM das Gewicht gemäß $\pi^{(t+1)}=M(\pi^{(t)})$ aktualisiert, mit $M(\pi)=\tfrac1n\sum_i r_i(\pi)$ und den Zugehörigkeitswahrscheinlichkeiten $r_i(\pi)=\pi f_0(x_i)/p(x_i\mid\pi)$.
    1. In der Nähe eines Fixpunkts $\hat\pi=M(\hat\pi)$ multipliziert jede Iteration den Fehler näherungsweise mit $M'(\hat\pi)$. Definieren Sie die beobachtete Information $I_{\textrm{obs}}=-\frac{d^2}{d\pi^2}\sum_i\log p(x_i\mid\pi)$ an der Stelle $\hat\pi$ und die Information der vollständigen Daten $I_{\textrm{com}}=n/(\hat\pi(1-\hat\pi))$, die bei beobachteten Labels vorläge (siehe :numref:`sec_mdl-maximum_likelihood`). Zeigen Sie, dass gilt:

       $$M'(\hat\pi)=1-\frac{I_{\textrm{obs}}}{I_{\textrm{com}}}.$$

       Drücken Sie beide Seiten durch die Zugehörigkeitswahrscheinlichkeiten aus und verwenden Sie $\sum_i r_i(\hat\pi)=n\hat\pi$. :citet:`Dempster.Laird.Rubin.1977` beweisen die allgemeine Form dieses Ergebnisses. Die Konvergenzrate von EM hängt davon ab, welcher Anteil der Information durch die latenten Variablen verborgen bleibt.
    1. Ziehen Sie $n=10{,}000$ Punkte mit $\pi=0.3$ für jedes $\Delta\in\{0.5,1,2,4\}$. Starten Sie EM bei $\pi^{(0)}=0.5$ und vergleichen Sie das gemessene Verhältnis aufeinanderfolgender Fehler mit der Vorhersage aus (b). Bestimmen Sie $\hat\pi$ aus einem langen Lauf. Wie viele Iterationen braucht EM für jedes $\Delta$, bis der Fehler unter $10^{-6}$ liegt?
    1. Was folgt aus dem Verhältnis $I_{\textrm{obs}}/I_{\textrm{com}}$ für den Standardfehler von $\hat\pi$ im Vergleich zu einem gleich großen Datensatz mit beobachteten Labels?
   :exlabel:`ex_mdl-maximum-likelihood-em-missing-information`

<!--en:6e7f89ec-->
1. **Die Schranke von Hand berechnen.** Verwenden Sie die einzelne Beobachtung $x=0.5$ und die beiden gleich gewichteten Komponenten $\mathcal N(-1,1)$ und $\mathcal N(1,1)$ aus dem Text.
    1. Berechnen Sie $\log p(x)$ sowie die ELBO :eqref:`eq_mdl-elbo` für die Gleichverteilung $q=(\tfrac12,\tfrac12)$ und für die A-posteriori-Verteilung $q=(0.27,0.73)$. Überprüfen Sie damit die Werte aus dem Text.
    1. Überprüfen Sie :eqref:`eq_mdl-elbo-gap` für die Gleichverteilung, indem Sie die KL-Divergenz direkt berechnen.
    1. Eine harte Zuordnung legt die gesamte Masse von $q$ auf die rechte Komponente. Berechnen Sie die zugehörige ELBO und ihren Abstand zur Log-Evidenz. Ein Algorithmus, der harte Zuordnungen und M-Schritte abwechselt, verringert die ELBO nach wie vor nie. Warum ist ein Anstieg seiner Log-Likelihood nicht mehr garantiert?
   :exlabel:`ex_mdl-latent-elbo-by-hand`

<!--en:141dc39e-->
1. **K-means als Grenzfall.** Alle $K$ Komponenten eines Gauß-Mischmodells sollen gleiche Gewichte und dieselbe feste Varianz $\sigma^2$ haben, sodass Sie nur die Mittelwerte anpassen.
    1. Schreiben Sie die Zugehörigkeitswahrscheinlichkeiten $r_{ik}$ als Softmax-Funktion von $-(x_i-\mu_k)^2/(2\sigma^2)$ über $k$. Welche Größe spielt die Rolle der Softmax-Temperatur? Zeigen Sie, dass für $\sigma^2\to0$ die Zugehörigkeitswahrscheinlichkeit zum nächsten Mittelwert gegen $1$ geht, sofern dieser nächste Mittelwert eindeutig ist.
    1. Zeigen Sie, dass der M-Schritt :eqref:`eq_mdl-gmm-mstep` dann jeden Mittelwert auf den Durchschnitt der ihm zugeordneten Punkte setzt. Zeigen Sie außerdem, dass die beiden Schritte zusammen $\sum_i (x_i-\mu_{z_i})^2$ nie erhöhen. Dabei bezeichnet $z_i$ die Komponente, die dem Punkt $i$ zugeordnet ist.
    1. Beim symmetrischen Start aus dem Text beginnen beide Mittelwerte am selben Punkt. Wie verhält sich K-means bei einem solchen Start? Was folgt daraus für die Entscheidung zwischen gleich weit entfernten Zentren in einer Implementierung?
   :exlabel:`ex_mdl-latent-kmeans-limit`

<!--en:3bdcae96-->
[Diskussion](https://d2l.discourse.group/t/36881)

<!--en:188cac3e-->
Latente Variablenmodelle, EM und die ELBO<br>**Mischmodelle · marginale Log-Likelihood · weiche Zuordnungen · untere Evidenzschranke · Koordinatenaufstieg · Fallstricke**

<!--en:e2108015-->
::: {.slide title="500 Punkte aus zwei Populationen ohne Labels"}

<!--en:25d9aa8a-->
[Motivation und Problemstellung]{.kicker}

<!--en:947d039a-->
- **Mit beobachteten Labels (vollständige Daten):** Für $K$ Gauß-Populationen genügen der Stichprobenanteil, der Mittelwert und die Varianz pro Cluster. Alle drei haben geschlossene Formeln.

<!--en:c2a79c8f-->
- **Ohne Labels (latente Variablen):** Die Wahrscheinlichkeit jeder Beobachtung ist eine Summe über die verborgenen Populationen. Diese Summe steht im Logarithmus der Likelihood.

<!--en:3063ee5a-->
- **Gemeinsame Strategie:** Fehlende Labels durch weiche A-posteriori-Wahrscheinlichkeiten $q(z)$ ersetzen und die **untere Evidenzschranke (ELBO)** maximieren.

<!--en:1a8aa33b-->
::: {.slide title="Gemeinsame und marginale Likelihood bei latenten Variablen"}

<!--en:2eaee2f0-->
[Generative Struktur]{.kicker}

<!--en:0aea5f6e-->
- **Datenerzeugung ($z \to x$):** Zuerst die verborgene Ursache gemäß $z \sim p(z \mid \boldsymbol{\theta})$ ziehen, dann die Beobachtung gemäß $x \sim p(x \mid z, \boldsymbol{\theta})$ erzeugen.

<!--en:b1c517f4-->
- **Marginale Likelihood (Evidenz):** Nur $x$ ist beobachtet. Daher ergibt sich $p(x \mid \boldsymbol{\theta})$ durch Summieren oder Integrieren über alle möglichen latenten Zustände $z$.

<!--en:9a9fb89b-->
::: {.slide title="Vollständige Daten mit Orakel und NLL der beobachteten Mischung"}

<!--en:34af3598-->
[Gauß-Mischmodelle]{.kicker}

<!--en:e7920b93-->
- **Orakel für vollständige Daten ($z_i$ bekannt):** Der Logarithmus wirkt direkt auf $\mathcal{N}(x_i; \mu_{z_i}, \sigma_{z_i}^2)$. Es entstehen $K$ unabhängige quadratische Ausdrücke mit geschlossener Lösung.

<!--en:631cc901-->
- **Beobachtete Mischung ($z_i$ verborgen):** Die Summe $\sum_{k=1}^K$ steht *innerhalb* von $\log(\cdot)$. Dadurch sind alle $3K$ Parameter an jedem Datenpunkt miteinander gekoppelt.

<!--en:3165b219-->
::: {.slide title="Weiche Zuordnung mit dem Satz von Bayes"}

<!--en:900d509d-->
[Zugehörigkeitswahrscheinlichkeiten]{.kicker}

<!--en:c9e33f82-->
- **Anteilige Zuordnung:** Bei $0.5\,\mathcal{N}(-1, 1) + 0.5\,\mathcal{N}(1, 1)$ ordnet der Satz von Bayes den Punkt $x = 0.5$ beiden Komponenten anteilig zu ($27\%$ der Komponente $1$, $73\%$ der Komponente $2$).

<!--en:1cb2d43c-->
- **Stetiger Übergang:** Die Zugehörigkeitswahrscheinlichkeiten liegen bei $50/50$ für $x = 0$ und verschieben sich auf $0.2\% / 99.8\%$ für $x = 3$.

<!--en:99d26885-->
::: {.slide title="Herleitung der unteren Evidenzschranke (ELBO)"}

<!--en:19a1ff5f-->
[Jensen-Ungleichung]{.kicker}

<!--en:2183e6e3-->
- **$\log$ innerhalb der Summe:** Für *jede* Verteilung $q(z)$ über latente Zustände liefert die Konkavität von $\log$ ($\log \mathbb{E}_q[Y] \ge \mathbb{E}_q[\log Y]$) eine globale untere Schranke.

<!--en:2c20e460-->
- **Berechenbare gemeinsame Log-Dichte:** In $\mathcal{L}(q, \boldsymbol{\theta})$ wirkt der Logarithmus direkt auf das Produkt $p(x, z \mid \boldsymbol{\theta})$ für vollständige Daten.

<!--en:e76ab7cd-->
::: {.slide title="Die ELBO als Summe aus Energie und Entropie"}

<!--en:dd6762a2-->
[Zerlegung der unteren Schranke]{.kicker}

<!--en:e2820c9c-->
- **Erwartete Log-Likelihood der vollständigen Daten ($\mathbb{E}_q[\log p(x, z \mid \boldsymbol{\theta})]$):** Einfach berechenbare Zielfunktion des Orakels, gewichtet mit den Wahrscheinlichkeiten $q(z)$

<!--en:0c723a23-->
- **Bonus für die Shannon-Entropie ($H(q) = -\sum_z q(z)\log q(z)$):** Begünstigt breit gestreute Verteilungen $q(z)$ und entfällt bei der Optimierung über $\boldsymbol{\theta}$.

<!--en:53a4db3c-->
::: {.slide title="Die grundlegende Zerlegung in ELBO und KL-Divergenz"}

<!--en:521417a9-->
[Exakte Formel für den Abstand]{.kicker}

<!--en:ffc068f0-->
- **Exakte Identität für den Abstand:** Einsetzen von $p(x, z \mid \boldsymbol{\theta}) = p(z \mid x, \boldsymbol{\theta})\,p(x \mid \boldsymbol{\theta})$ in $\mathcal{L}(q, \boldsymbol{\theta})$ ergibt
  $$\log p(x \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + D_{\mathrm{KL}}\bigl(q(z) \,\|\, p(z \mid x, \boldsymbol{\theta})\bigr)$$

<!--en:341d5d53-->
- **Nichtnegativität und Gleichheit:** Wegen $D_{\mathrm{KL}} \ge 0$ gilt überall $\mathcal{L}(q, \boldsymbol{\theta}) \le \log p(x \mid \boldsymbol{\theta})$. Gleichheit ($\mathcal{L} = \log p$) gilt genau dann, wenn $q(z) = p(z \mid x, \boldsymbol{\theta})$.

<!--en:2981fb5f-->
- **Gemeinsame Tangente:** Erhält $q$ den Wert der A-posteriori-Verteilung bei $\boldsymbol{\theta}^{(t)}$, so berührt die Kurve der unteren Schranke die wahre Log-Likelihood bei $\boldsymbol{\theta}^{(t)}$ tangential.

<!--en:6076c2e7-->
::: {.slide title="Rechenbeispiel zum Abstand zwischen ELBO und Log-Evidenz"}

<!--en:32af94d5-->
[Numerische Überprüfung bei x = 0.5]{.kicker}

<!--en:056db6a5-->
- **Exakte Identität bei $x = 0.5$ ($\log p(x) = -1.42$):** Für jede Schätzung $q(z)$ gilt $\mathcal{L}(q) + D_{\mathrm{KL}}(q \,\|\, p(z \mid x)) = -1.42$ Nat.

<!--en:fb7e4fa2-->
- **Exakte A-posteriori-Verteilung $q = (0.27, 0.73)$:** Die ELBO erreicht $\mathcal{L} = -1.42$. Die KL-Divergenz ist null.

<!--en:9dc1a733-->
- **Warum die harte Zuordnung $(0, 1)$ mehr als doppelt so viel verliert wie die Gleichverteilung $(0.5, 0.5)$:** Bei $q = (0, 1)$ entfällt der Entropiebonus ($H(q) = 0$). Die KL-Divergenz beträgt $0.31$ Nat gegenüber $0.12$ Nat bei $(0.5, 0.5)$.

<!--en:47d3322a-->
::: {.slide title="Expectation-Maximization als Koordinatenaufstieg"}

<!--en:2c8c2152-->
[Der EM-Algorithmus]{.kicker}

<!--en:d3c9beb3-->
- **E-Schritt (Erwartungswert):** $\boldsymbol{\theta}^{(t)}$ festhalten und für jeden Punkt $q_i^{(t)}(z) = p(z \mid x_i, \boldsymbol{\theta}^{(t)})$ setzen. Die KL-Divergenz fällt auf null, sodass die ELBO die Kurve von $\log P(X \mid \boldsymbol{\theta}^{(t)})$ berührt.

<!--en:568ce410-->
- **M-Schritt (Maximierung):** Die Zugehörigkeitswahrscheinlichkeiten $q_i^{(t)}$ festhalten und die erwartete Log-Likelihood der vollständigen Daten über $\boldsymbol{\theta}$ maximieren:
  $$\boldsymbol{\theta}^{(t+1)} = \mathop{\mathrm{argmax}}_{\boldsymbol{\theta}} \sum_{i=1}^n \mathbb{E}_{q_i^{(t)}(z)}\bigl[\log p(x_i, z \mid \boldsymbol{\theta})\bigr]$$

<!--en:e866e8ac-->
::: {.slide title="Satz zum monotonen Anstieg der Likelihood bei EM"}

<!--en:bdad95e5-->
[Konvergenzgarantie]{.kicker}

<!--en:0e2652b2-->
Warum kann ein EM-Schritt die wahre Log-Likelihood $\log P(X \mid \boldsymbol{\theta})$ der beobachteten Daten nie verringern? Drei Beziehungen ergeben zusammen einen einzeiligen Beweis der Monotonie:

<!--en:18313e36-->
1. **Globale untere Schranke ($\ge$):** Die ELBO ist eine untere Schranke für $\log P(X \mid \boldsymbol{\theta})$ bei den neuen Parametern $\boldsymbol{\theta}^{(t+1)}$, und zwar für jedes $q_i^{(t)}$.

<!--en:ffae6963-->
2. **Optimierung im M-Schritt ($\ge$):** Die Wahl von $\boldsymbol{\theta}^{(t+1)}$ als Maximalstelle der Schranke garantiert $\sum_i \mathcal{L}(q_i^{(t)}, \boldsymbol{\theta}^{(t+1)}) \ge \sum_i \mathcal{L}(q_i^{(t)}, \boldsymbol{\theta}^{(t)})$.

<!--en:8c05bab0-->
3. **Gleichheit im E-Schritt ($=$):** Da $q_i^{(t)}$ gleich der exakten A-posteriori-Verteilung bei $\boldsymbol{\theta}^{(t)}$ gesetzt wurde, ist der Abstand bei $\boldsymbol{\theta}^{(t)}$ null.

<!--en:d015de11-->
**Warum die exakte A-posteriori-Verteilung zählt:** Ersetzt ein Algorithmus den E-Schritt durch harte $0/1$-Zuordnungen, steigt die ELBO weiterhin monoton. Im dritten Beweisschritt gilt jedoch keine Gleichheit mehr. Daher kann die wahre Log-Likelihood $\log P(X \mid \boldsymbol{\theta})$ sinken.

<!--en:b74b414b-->
::: {.slide title="Geschlossene EM-Formeln für Gauß-Mischmodelle"}

<!--en:2def85f6-->
[Anteilige Zählung und Mittelwertbildung]{.kicker}

<!--en:5df931b0-->
Für ein Gauß-Mischmodell mit $K$ Komponenten liefern E- und M-Schritt einfache Stichprobenkennzahlen, gewichtet mit den Zugehörigkeitswahrscheinlichkeiten:

<!--en:1bc5666c-->
- **E-Schritt (Zugehörigkeitswahrscheinlichkeiten $r_{ik} = q_i(z=k)$):**
  $$r_{ik} = \frac{\pi_k\,\mathcal{N}(x_i;\, \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j\,\mathcal{N}(x_i;\, \mu_j, \sigma_j^2)}$$

<!--en:5e80c9b4-->
- **M-Schritt (gewichtete Anzahlen, Mittelwerte und Varianzen mit $n_k = \sum_{i=1}^n r_{ik}$):**
  $$\pi_k \leftarrow \frac{n_k}{n}, \qquad \mu_k \leftarrow \frac{1}{n_k}\sum_{i=1}^n r_{ik}\,x_i, \qquad \sigma_k^2 \leftarrow \frac{1}{n_k}\sum_{i=1}^n r_{ik}(x_i - \mu_k)^2$$

<!--en:6cbd0af5-->
- **K-means als Grenzfall verschwindender Varianz:** Für $\pi_k = 1/K$ und eine gemeinsame, vorgegebene Varianz $\sigma^2 \to 0$ gehen die Zugehörigkeitswahrscheinlichkeiten der Softmax-Funktion $r_{ik} \propto \exp\bigl(-(x_i-\mu_k)^2/(2\sigma^2)\bigr)$ in $0/1$-Indikatoren für das nächstgelegene Zentrum über.

<!--en:c412c457-->
::: {.slide title="EM für eindimensionale Gauß-Mischmodelle in NumPy"}

<!--en:a4fbb6b9-->
[Implementierung]{.kicker}

<!--en:d39d141d-->
Mit vektorisierten NumPy-Operationen genügen für die gesamte EM-Schleife bei $n = 500$ Beobachtungen aus $0.6\,\mathcal{N}(-2.0, 0.8^2) + 0.4\,\mathcal{N}(1.5, 0.6^2)$ ein Dutzend Zeilen:

<!--en:8ec3411d-->
::: {.slide title="EM für eindimensionale Gauß-Mischmodelle in NumPy (Fortsetzung)"}

<!--en:a4fbb6b9-2-->
[Implementierung]{.kicker}

<!--en:045782e9-->
Durch Broadcasting von `x[:, None]` mit der Form `(n, 1)` und Parameterarrays der Form `(K,)` entsteht in einem Schritt die Matrix der gemeinsamen Dichten der Form `(n, K)`, gespeichert in `joint`. Zeilenweises Normieren ergibt `r` (E-Schritt), spaltenweises Summieren ergibt `n_k` (M-Schritt).

<!--en:50ac8a6a-->
::: {.slide title="Schnelle Konvergenz bei asymmetrischer Initialisierung"}

<!--en:cebd2814-->
[Empirischer Verlauf]{.kicker}

<!--en:da7d1025-->
Zwei stark überlappende Komponenten starten mit Varianz eins, Mittelwerten $\mu = (-0.5, 0.5)$ und gleichen Gewichten $\pi = (0.5, 0.5)$. Darauf folgen $30$ EM-Iterationen:

<!--en:7db0e148-->
- **Monotoner Anstieg:** Die Log-Likelihood steigt um rund $400$ Nat (von $\approx -1{,}280$ auf $\approx -860$), fast ausschließlich in den ersten $6$ Schritten.

<!--en:2d60c465-->
- **Fehlende Information bestimmt das Tempo:** In der Nähe eines Fixpunkts beträgt die lineare Konvergenzrate von EM $1 - I_{\mathrm{obs}}/I_{\mathrm{com}}$. Das ist der Anteil der Fisher-Information, der durch die fehlenden Labels verborgen bleibt. Bei gut getrennten Clustern ($I_{\mathrm{obs}} \approx I_{\mathrm{com}}$) genügen wenige Schritte bis zur Konvergenz.

<!--en:6eea4939-->
::: {.slide title="Fallstrick 1: Labeltausch und symmetrische Startwerte"}

<!--en:be1900bb-->
[Nichtidentifizierbarkeit und Sattelpunkte]{.kicker}

<!--en:0e9c1716-->
Vertauschen der $K$ Komponentenindizes ergibt $K!$ gleichwertige globale Maxima (*Labeltausch*). Weit problematischer sind identische Startwerte für zwei Komponenten, etwa $\mu = (0, 0)$ und $\sigma = (1, 1)$:

<!--en:1037ad2c-->
- **Unveränderliche Zugehörigkeitswahrscheinlichkeiten:** Wegen der identischen Startwerte ergeben sich für beide Komponenten gleiche Zugehörigkeitswahrscheinlichkeiten $r_{i1} = r_{i2} = 0.5$ für jede Beobachtung $x_i$. Der M-Schritt liefert daher stets den gleichen Mittelwert und die gleiche Standardabweichung des gesamten Datensatzes für beide Komponenten.

<!--en:f5a5af4d-->
- **Stationärer Punkt ohne Maximum:** EM bleibt am Sattelpunkt einer einzelnen Gauß-Verteilung stecken, mehr als $100$ Nat unter der Lösung mit zwei Clustern. Dieselbe Symmetrie entsteht, wenn die versteckten Einheiten einer Schicht eines neuronalen Netzes mit identischen Gewichten starten.

<!--en:7f65d636-->
::: {.slide title="Fallstrick 2: Unbeschränkte Likelihood und Dichtespitzen"}

<!--en:31bbd596-->
[Entartete globale Maxima]{.kicker}

<!--en:4726169b-->
- **Entstehung einer Dichtespitze:** Liegt das Zentrum der Komponente $2$ auf einem einzigen Punkt ($\mu_2 = x_1$) und gilt $\sigma_2 = \epsilon \to 0$, so wächst die Dichte gemäß $\frac{1}{\sqrt{2\pi}\epsilon} \to \infty$. Dasselbe Phänomen tritt bei der NLL einer heteroskedastischen Regression auf, wenn $\hat{y}_i = y_i$ und $\hat{\sigma}_i \to 0$ gelten.

<!--en:759883df-->
- **Praktische Abhilfe:** Eine untere Varianzgrenze $\sigma_k^2 \ge \sigma_{\min}^2 > 0$ vorgeben oder für $\sigma_k^2$ eine inverse Gamma-Verteilung als A-priori-Verteilung wählen.

<!--en:73785b50-->
::: {.slide title="Von exaktem EM zu Variationsautoencodern (VAEs)"}

<!--en:b96fafd7-->
[Amortisierte variationelle Inferenz]{.kicker}

<!--en:5fbc5ccd-->
- **Schwer berechenbare A-posteriori-Verteilung:** Durchlaufen stetige latente Codes $z \in \mathbb{R}^d$ einen tiefen nichtlinearen Decoder $p_{\boldsymbol{\theta}}(x \mid z)$, so hat die exakte A-posteriori-Verteilung $p(z \mid x, \boldsymbol{\theta})$ keine geschlossene Form.

<!--en:194cc345-->
- **Amortisierter Encoder $q_{\boldsymbol{\phi}}(z \mid x)$:** Ein neuronales Netz sagt $(\boldsymbol{\mu}_{\boldsymbol{\phi}}(x), \log\boldsymbol{\sigma}_{\boldsymbol{\phi}}(x))$ direkt aus $x$ vorher.

<!--en:e75730f9-->
- **Gemeinsamer stochastischer Gradientenabstieg auf der negativen ELBO:** Die Reparametrisierung $z = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ ermöglicht das gemeinsame Training von $(\boldsymbol{\theta}, \boldsymbol{\phi})$ mittels Backpropagation für $\sum_i \mathcal{L}(q_{\boldsymbol{\phi}}(\cdot \mid x_i), \boldsymbol{\theta})$.

<!--en:5357bf55-->
::: {.slide title="EM, VAEs und bayessche variationelle Inferenz"}

<!--en:c7a2ac3a-->
[Drei Anwendungen derselben Schranke]{.kicker}

<!--en:39d253d9-->
Die Zerlegung $\log \text{Evidence} = \mathrm{ELBO}(q) + D_{\mathrm{KL}}(q \,\|\, \text{Posterior})$ verbindet drei wichtige Algorithmen aus dem statistischen Lernen und der Modellierung mit tiefen generativen Netzen:

<!--en:045f7bb5-->
| Methode | Unbeobachtete Variable | Variationelle Verteilung $q$ | Maximierung der $\mathrm{ELBO}$ | Sinkt der KL-Abstand auf $0$? |

<!--en:6d6f8fe1-->
| **EM-Algorithmus** | Diskrete Clusterzugehörigkeit $z_i \in \{1,\dots,K\}$ pro Beispiel | Exakte tabellarische Verteilung $q_i(z_i)$ pro Punkt | Koordinatenaufstieg (geschlossene Formeln für E- und M-Schritt) | **Ja** ($q_i = p(z_i \mid x_i, \boldsymbol{\theta}^{(t)})$ nach jedem E-Schritt) |

<!--en:edbf40ca-->
| **Variationsautoencoder** | Stetiger Code $\mathbf{z}_i \in \mathbb{R}^d$ pro Beispiel | Amortisiertes Encoder-Netz $q_{\boldsymbol{\phi}}(\mathbf{z}_i \mid \mathbf{x}_i)$ | Gemeinsamer stochastischer Gradientenabstieg über $(\boldsymbol{\theta}, \boldsymbol{\phi})$ durch Reparametrisierung | **Nein** (begrenzt durch die Kapazität der Encoder-Familie) |

<!--en:b2661f87-->
| **Bayessche variationelle Inferenz (§{sec_mdl-bayesian-computation})** | Globale Modellparameter $\boldsymbol{\theta} \in \mathbb{R}^p$ | Parametrische Familie $q_{\boldsymbol{\psi}}(\boldsymbol{\theta})$ (z. B. Gauß-Verteilungen) | Gradientenaufstieg auf $\mathcal{L}(\boldsymbol{\psi})$ bezüglich der Variationsparameter $\boldsymbol{\psi}$ | **Nein** (minimiert die umgekehrte KL-Divergenz $D_{\mathrm{KL}}(q_{\boldsymbol{\psi}} \,\|\, p(\boldsymbol{\theta}\mid\mathcal{D}))$) |

<!--en:8830c36d-->
::: {.slide title="Verständnisfragen zu latenten Variablen, EM und ELBO"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:7b469893-->
1. **Entropiebonus der ELBO:** Bei $x = 0.5$ ist die wahre A-posteriori-Verteilung $(0.27, 0.73)$. Warum ergibt eine harte Zuordnung $q = (0, 1)$ einen größeren Abstand zur Log-Evidenz ($0.31$ Nat) als eine unentschiedene $50/50$-Zuordnung ($0.12$ Nat)?

<!--en:e71a0fb5-->
2. **Grenzfall der Softmax-Temperatur bei EM für Gauß-Mischmodelle:** Alle $K$ Gauß-Komponenten haben gleiche Mischungsgewichte $\pi_k = 1/K$ und eine gemeinsame feste Varianz $\sigma^2$. Wie verhalten sich die Zugehörigkeitswahrscheinlichkeiten $r_{ik}$ im E-Schritt und die Mittelwerte $\mu_k$ im M-Schritt für $\sigma^2 \to 0$?

<!--en:f99b0e5b-->
3. **Warum hartes EM die Likelihood verringern kann:** Ersetzt ein Algorithmus den E-Schritt durch harte Zuordnungen $q_i(z) = \mathbf{1}\{z = k^*(x_i)\}$, so erhöht er die ELBO weiterhin monoton. Warum kann die wahre Log-Likelihood $\log P(X \mid \boldsymbol{\theta})$ sinken?

<!--en:557b98b6-->
**Lösungen:**

<!--en:ea5990ad-->
1. Die harte Zuordnung gibt Komponente $1$ die Masse null ($H(q) = 0$). Daraus folgen $D_{\mathrm{KL}} = 1 \cdot \log\frac{1}{0.73} = 0.31$ Nat. Bei der $50/50$-Zuordnung bleiben $H(q) = \log 2 = 0.69$ Nat Entropie erhalten, und die KL-Divergenz zu $(0.27, 0.73)$ ist geringer.

<!--en:68a459a1-->
2. Die Zugehörigkeitswahrscheinlichkeiten $r_{ik} = \frac{\exp(-(x_i-\mu_k)^2/(2\sigma^2))}{\sum_j \exp(-(x_i-\mu_j)^2/(2\sigma^2))}$ ergeben sich aus einer Softmax-Funktion mit der Temperatur $2\sigma^2$. Für $\sigma^2 \to 0$ gilt $r_{ik} \to 1$ für das nächstgelegene Zentrum $\mu_k$. Damit geht EM direkt in **K-means** über.

<!--en:3b9017a6-->
3. Eine harte Verteilung $q_i$ stimmt nie mit der weichen A-posteriori-Verteilung $p(z \mid x_i, \boldsymbol{\theta}^{(t)})$ überein. Daher verschwindet der KL-Abstand $D_{\mathrm{KL}}(q_i \,\|\, p) > 0$ bei $\boldsymbol{\theta}^{(t)}$ nicht. Die für die Berührung nötige Gleichheit $\mathcal{L}(q^{(t)}, \boldsymbol{\theta}^{(t)}) = \log P(X \mid \boldsymbol{\theta}^{(t)})$ ist verletzt.

<!--en:082acb9f-->
::: {.slide title="Zusammenfassung zu latenten Variablen, EM und ELBO"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:f932efc8-->
| Begriff | Mathematische Form | Bedeutung für Statistik und Deep Learning |

<!--en:be05e650-->
| **Marginale Log-Likelihood** | $\log p(x \mid \boldsymbol{\theta}) = \log \sum_z p(z \mid \boldsymbol{\theta})\,p(x \mid z, \boldsymbol{\theta})$ | Summe innerhalb von $\log$ koppelt alle Parameter und verhindert eine geschlossene Lösung für die Maximum-Likelihood-Schätzung |

<!--en:fbed4be2-->
| **Untere Evidenzschranke** | $\mathcal{L}(q, \boldsymbol{\theta}) = \mathbb{E}_q[\log p(x, z \mid \boldsymbol{\theta})] + H(q)$ | Die Jensen-Ungleichung ermöglicht es, $\log$ wieder auf die gemeinsame Dichte anzuwenden |

<!--en:db6f7504-->
| **Identität für den KL-Abstand** | $\log p(x \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + D_{\mathrm{KL}}\bigl(q(z) \,\|\, p(z \mid x, \boldsymbol{\theta})\bigr)$ | Überall gilt $\mathcal{L} \le \log p$. Gleichheit gilt genau dann, wenn $q$ die wahre A-posteriori-Verteilung ist |

<!--en:9be3cd47-->
| **Koordinatenaufstieg bei EM** | E-Schritt: $q_i^{(t)} = p(z \mid x_i, \boldsymbol{\theta}^{(t)})$, M-Schritt: $\operatorname{argmax}_{\boldsymbol{\theta}} \sum_i \mathcal{L}$ | Monotone Erhöhung von $\log P(X \mid \boldsymbol{\theta})$ mit jeder Iteration |

<!--en:863db697-->
| **Anteilige Zählung im Gauß-Mischmodell** | $r_{ik} \propto \pi_k \mathcal{N}(x_i; \mu_k, \sigma_k^2)$ und $\mu_k = \frac{1}{n_k}\sum_i r_{ik}x_i$ | Weiche Zuordnungen in den Orakelschätzern. Grenzfall K-means bei gemeinsamer Varianz $\sigma^2 \to 0$ |

<!--en:c0ba73e2-->
| **Fallstricke bei EM** | Symmetrische Startwerte ($\mu_1 = \mu_2$) und Dichtespitzen ($\sigma_k \to 0$) | Zufällige Neustarts sowie Untergrenzen oder A-priori-Verteilungen für $\sigma_k^2$ erforderlich |
