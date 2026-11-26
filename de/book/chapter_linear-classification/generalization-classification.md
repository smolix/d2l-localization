<!--en:fed8961a-->
# Generalisierung bei der Klassifikation

<!--en:903e50a8-->
Unser Fashion-MNIST-Klassifikator erreichte eine Genauigkeit von rund $83\%$ auf $10{,}000$ Validierungsbildern. Wie verlässlich sagt diese Zahl voraus, wie gut er künftige Kleidungsstücke erkennt? Dafür gibt es zwei grundlegende Einschränkungen:

<!--en:1f7578d1-->
1. **Unsicherheit bei endlichen Stichproben und wiederholte Nutzung des Testdatensatzes:** Selbst ein neuer Testdatensatz aus $n = 10{,}000$ Bildern ist nur eine endliche Zufallsstichprobe. In der Praxis kommt hinzu, dass wir beim Modellvergleich und bei der Hyperparameteroptimierung immer wieder denselben Teildatensatz auswerten.

<!--en:7875c791-->
2. **Auswendiglernen oder Lernen:** In hohen Dimensionen unterscheidet sich jedes Trainingsbild von allen anderen. Ein Modell, das die Trainingsbilder samt Label in einer Tabelle speichert, erreicht eine Trainingsgenauigkeit von $100\%$ und rät bei jedem neuen Bild. Warum lässt ein kleiner Trainingsfehler bei einem *linearen* Klassifikator auf einen kleinen Populationsfehler (engl. population error) schließen, wenn selbst ein Trainingsfehler von null beim bloßen Auswendiglernen nichts aussagt?

<!--en:81b11a8a-->
In :numref:`sec_generalization_basics` haben wir Überanpassung (engl. overfitting) und die Aufteilung in Trainings- und Validierungsdaten für die Regression qualitativ beschrieben. Für die Klassifikation untersuchen wir die Generalisierung nun quantitativ. Wir leiten exakte und asymptotische Konfidenzintervalle für einen Testdatensatz der Größe $n$ her. Wir berechnen, wie stark Selektionsbias und adaptive Überanpassung die Scores bei wiederholter Nutzung des Testdatensatzes erhöhen. Aus der statistischen Lerntheorie führen wir die *Vapnik-Chervonenkis-Dimension (VC-Dimension)* und die *gleichmäßige Konvergenz* (engl. uniform convergence) ein.

<!--en:a222c2a2-->
## Der Testdatensatz

<!--en:d30250f2-->
### Empirischer Fehler und Populationsfehler

<!--en:9610918a-->
Wir legen einen einzelnen Klassifikator $f$ fest, *bevor* wir einen unabhängigen Testdatensatz $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ ziehen. Seine Beispiele sind unabhängig und identisch nach der Populationsverteilung $P(X, Y)$ verteilt. Der *empirische Fehler* auf $\mathcal{D}$ und der wahre *Populationsfehler* sind gegeben durch

<!--en:b4bbb134-->
Da $f$ im Voraus feststeht, entspricht jeder Fehlerindikator $\mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)})$ einem unabhängigen Münzwurf mit Verteilung $\mathrm{Bernoulli}(\epsilon(f))$, Mittelwert $\epsilon(f)$ und Varianz $\epsilon(f)(1 - \epsilon(f)) \leq 1/4$. Der Testfehler $\epsilon_{\mathcal{D}}(f)$ ist daher ein erwartungstreuer Schätzer (engl. unbiased estimator) des Populationsfehlers ($\mathbb{E}[\epsilon_{\mathcal{D}}(f)] = \epsilon(f)$). Sein Standardfehler beträgt

<!--en:7c00662d-->
Für unseren Fashion-MNIST-Klassifikator ($\epsilon(f) \approx 0.17$) beträgt der Standardfehler bei $n = 10{,}000$ Testbildern $\sqrt{0.17 \times 0.83 / 10{,}000} \approx 0.0038$ ($0.38$ Prozentpunkte). Nach diesem Skalierungsgesetz nimmt er proportional zu $1/\sqrt{n}$ ab. Um die Messunsicherheit zu halbieren, benötigen wir also *viermal* so viele Testbeispiele.

<!--en:160e6ce3-->
### Konfidenzintervalle und die Hoeffding-Ungleichung

<!--en:1e53d7f2-->
Um aus dem Standardfehler ein $95\%$-Konfidenzintervall $\epsilon_{\mathcal{D}}(f) \pm r$ zu gewinnen, stehen uns zwei Verfahren zur Verfügung:

<!--en:a91d8c3b-->
1. **Asymptotisches Normalintervall (zentraler Grenzwertsatz):** Für große $n$ ist $\epsilon_{\mathcal{D}}(f)$ näherungsweise normalverteilt. Die Abweichung beträgt höchstens $1.96$ Standardfehler vom wahren Wert $\epsilon(f)$, und zwar mit Wahrscheinlichkeit $\approx 0.95$:
   $$P\left(|\epsilon_\mathcal{D}(f) - \epsilon(f)| \leq 1.96\sqrt{\frac{\epsilon(f)(1 - \epsilon(f))}{n}}\right) \approx 0.95.$$
   Bei $\epsilon \approx 0.17$ und $n = 10{,}000$ hat das $95\%$-Normalintervall den Radius $1.96 \times 0.0038 \approx 0.007$ ($\pm 0.7$ Prozentpunkte). Im ungünstigsten Fall ($\epsilon = 1/2$) sind für einen $95\%$-Konfidenzradius $\leq 0.01$ nach der Normalapproximation $n \geq (1.96 \times 0.5 / 0.01)^2 = 9{,}604$ Beispiele nötig. Damit lässt sich der Umfang der Testdatensätze von MNIST, Fashion-MNIST und CIFAR-10 einordnen: Mit je $10{,}000$ Beispielen lässt sich die Genauigkeit bis auf etwa einen Prozentpunkt bestimmen.

<!--en:2a25c91b-->
2. **Hoeffding-Intervall für endliche Stichproben:** Liegt $\epsilon(f)$ sehr nahe bei $0$, kann das Normalintervall den wahren Wert viel zu selten enthalten. Auf einem kleinen Testdatensatz tritt dann womöglich kein einziger Fehler auf. Setzt man diese Fehlerrate in die Normalapproximation ein, schrumpft der Radius auf $0$. Die *Hoeffding-Ungleichung* :cite:`Hoeffding.1963` setzt lediglich voraus, dass jeder Fehlerindikator in $[0, 1]$ liegt. Sie liefert eine exakte Schranke für jedes $n$ und jede Verteilung (:numref:`sec_mdl-concentration-generalization`):
   $$
   P\left(|\epsilon_\mathcal{D}(f) - \epsilon(f)| \geq t\right) \leq 2\exp\left(-2nt^2\right).
   $$
   :eqlabel:`eq_generalization-classification-hoeffding`
   Setzen wir die rechte Seite gleich $\delta = 0.05$, so erhalten wir den $95\%$-Konfidenzradius nach Hoeffding, $r = \sqrt{\log(2/\delta)/(2n)}$. Wir benötigen $n \geq \log(40)/(2 \times 0.01^2) \approx 18{,}445$ Beispiele für $r \leq 0.01$, rund doppelt so viele wie nach der Normalapproximation. Beide Rechnungen gehen von der maximalen Varianz $1/4$ aus. Die exponentielle Schranke von Hoeffding für die Randwahrscheinlichkeit gilt jedoch für jedes $n$ und ist weniger scharf als das exakte Quantil der Normalverteilung.

<!--en:24b3b403-->
Simulieren wir jeweils $10{,}000$ unabhängige Testdatensätze für verschiedene Größen $n \in [100, 10{,}000]$ und zwei Klassifikatoren (einen mit $\epsilon = 0.17$ und einen mit $\epsilon = 0.01$). Wir tragen auf, wie oft die beiden $95\%$-Intervalle den wahren Fehler enthalten, also ihre empirische Überdeckung (engl. coverage):

<!--en:67475f41-->
Bei $\epsilon = 0.17$ liegt die Überdeckung des Normalintervalls mit eingesetzter Fehlerschätzung nahe bei $95\%$, und zwar für jedes $n$. Bei $\epsilon = 0.01$ und kleinem $n$ kommen Testdatensätze ohne Fehler vor. Dadurch sinkt die Überdeckung des Normalintervalls auf bis zu $\sim 63\%$. Das konservative Hoeffding-Intervall erreicht dagegen überall eine Überdeckung von mehr als $95\%$.

<!--en:1494224d-->
## Wiederholte Nutzung des Testdatensatzes

<!--en:dfe8084b-->
### Auswahl unter vielen Klassifikatoren

<!--en:ac3fb4b5-->
Was geschieht, wenn wir $k$ vorab festgelegte Klassifikatoren $f_1, \ldots, f_k$ auf *demselben* Testdatensatz $\mathcal{D}$ auswerten und nur das beste Ergebnis angeben? Selbst wenn alle $k$ Kandidaten feststehen, bevor wir $\mathcal{D}$ ziehen, wählen wir mit dem besten Score gerade den Kandidaten aus, der auf $\mathcal{D}$ am meisten Glück hatte. Wir verwenden die *boolesche Ungleichung* (engl. union bound): Die Wahrscheinlichkeit, dass mindestens eines von $k$ unerwünschten Ereignissen eintritt, ist höchstens so groß wie die Summe ihrer Einzelwahrscheinlichkeiten. Auf :eqref:`eq_generalization-classification-hoeffding` angewandt ergibt das

<!--en:16125953-->
Setzen wir die rechte Seite gleich $\delta$, so erhalten wir ein simultanes Konfidenzniveau von $1 - \delta$ für alle $k$ Modelle. Dafür wächst der Konfidenzradius auf $\sqrt{\log(2k/\delta)/(2n)}$. Mit $\sqrt{\log k}$ wächst er nur langsam: Beim Vergleich von $k = 1{,}000$ vorab festgelegten Modellen nimmt der Radius nur um einen logarithmischen Faktor zu. Betrachten wir dennoch $n = 1{,}000$ per Münzwurf erzeugte Testlabel und $k = 10{,}000$ Klassifikatoren, die selbst nur Münzen werfen. Jeder einzelne hat eine wahre Genauigkeit von $0.50$. Die beste Testgenauigkeit steigt mit der Anzahl der Kandidaten stetig entlang der Kurve $0.5 + \sqrt{\log(k)/(2n)}$ bis auf $\sim 0.56$:

<!--en:80071a3e-->
### Adaptive Überanpassung

<!--en:cd305990-->
Der Selektionsbias wirkt sich viel stärker aus, wenn wir neue Kandidaten *adaptiv* anhand früherer Testergebnisse auswählen :cite:`Dwork.Feldman.Hardt.ea.2015`. Sobald die Testergebnisse den Modellentwurf beeinflussen, sind die Klassifikatoren nicht mehr unabhängig von $\mathcal{D}$. Die aus der booleschen Ungleichung abgeleitete Schranke :eqref:`eq_generalization-classification-union` gilt dann nicht mehr. Betrachten wir wie :citet:`Blum.Hardt.2015` nur die ersten $1{,}000$ Klassifikatoren aus unserem Münzexperiment. Wir wählen diejenigen aus, deren Genauigkeit auf dem Testdatensatz zufällig über $0.50$ lag. Für jedes Testbeispiel lassen wir die Mehrheit dieser Klassifikatoren entscheiden:

<!--en:b29a84a8-->
Ohne ein einziges Testlabel zu kennen, fragen wir nur $1{,}000$ skalare Genauigkeitswerte ab. Schon steigt die Testgenauigkeit der Münzwürfe per Mehrheitsentscheid auf $\sim 70\%$. Auf neuen Daten bleibt die Populationsgenauigkeit bei $50\%$. (Mit $O(n)$ Abfragen lassen sich sogar sämtliche Testlabel rekonstruieren, siehe Übung :exref:`ex_generalization-classification-leaking-test-set`.) Für die Hyperparameteroptimierung und das Erproben von Architekturen verwenden wir deshalb einen eigenen Validierungsdatensatz (Übung :exref:`ex_generalization-classification-validation-split`). Den Testdatensatz werten wir ein einziges Mal zum Abschluss aus.

<!--en:a894358c-->
Wie sehr hat ein Jahrzehnt wiederholter Nutzung den gängigen Benchmarks geschadet? Als :citet:`Recht.Roelofs.Schmidt.ea.2019` neue Testdatensätze für CIFAR-10 und ImageNet zusammenstellten, sank die Genauigkeit um $3$ bis $15$ Prozentpunkte. Das Ranking der Modelle änderte sich jedoch kaum. Der oben beschriebene Extremfall ist möglich, doch in der üblichen Modellentwicklung war die Überanpassung an diese Testdatensätze weit geringer, als es die Theorie zulässt. Der Rückgang selbst lässt sich besser durch eine kleine Änderung der Datenverteilung erklären. Damit befasst sich :numref:`sec_environment-and-distribution-shift`.

<!--en:ac5d0d9e-->
## Statistische Lerntheorie

<!--en:f8c39c74-->
### Gleichmäßige Konvergenz

<!--en:10f0f195-->
Wie lässt sich *a priori* garantieren, dass ein Klassifikator $f_{\mathcal{S}}$ auf die Population generalisiert? Wir wählen ihn aus einer Hypothesenklasse $\mathcal{F}$, indem wir den Trainingsfehler $\epsilon_{\mathcal{S}}(f)$ auf einem Trainingsdatensatz $\mathcal{S}$ der Größe $n$ minimieren. Der Lernalgorithmus wählt $f_{\mathcal{S}} \in \mathcal{F}$ erst, nachdem er $\mathcal{S}$ gesehen hat. Deshalb muss der Trainingsfehler *gleichzeitig für jedes $f \in \mathcal{F}$* nahe am Populationsfehler liegen (*gleichmäßige Konvergenz*). Ist $\mathcal{F}$ endlich ($|\mathcal{F}| < \infty$), wenden wir die boolesche Ungleichung :eqref:`eq_generalization-classification-union` auf alle $k = |\mathcal{F}|$ Hypothesen auf $\mathcal{S}$ an. Mit Wahrscheinlichkeit mindestens $1 - \delta$ gilt dann

<!--en:6fc7ac3c-->
### VC-Dimension und VC-Schranke

<!--en:c98e15eb-->
Bei Familien mit kontinuierlichen Parametern wie den linearen Halbräumen $f(\mathbf{x}) = \operatorname{sign}(\mathbf{w}^\top\mathbf{x} + b)$ auf $\mathbb{R}^d$ gilt $|\mathcal{F}| = \infty$. Damit verliert :eqref:`eq_generalization-classification-finite` ihre Aussagekraft. Doch wie :citet:`Vapnik.Chervonenkis.1971` beobachteten, gilt für jede endliche Menge von $n$ Punkten: Zwei Klassifikatoren, die genau dieselbe binäre Zuweisung der Label $\pm 1$ erzeugen, lassen sich auf $\mathcal{S}$ nicht unterscheiden. Wir sagen, $\mathcal{F}$ *zertrümmert* (engl. shatters) eine Menge von $m$ Punkten, wenn Funktionen aus $\mathcal{F}$ alle $2^m$ möglichen binären Labelzuweisungen auf diesen $m$ Punkten realisieren können. Die *Vapnik-Chervonenkis-Dimension (VC-Dimension)* $v = \mathrm{VC}(\mathcal{F})$ ist die Größe der größten Punktmenge, die $\mathcal{F}$ zertrümmern kann:

<!--en:05281bbc-->
- **Geraden in der Ebene ($\mathbb{R}^2$) haben VC-Dimension $3$ (:numref:`fig_mdl-clf-shattering`):** Beliebige drei nicht kollineare Punkte lassen sich durch Geraden auf alle $2^3 = 8$ Arten aufteilen. Keine vier Punkte in $\mathbb{R}^2$ lassen sich hingegen zertrümmern. In einem konvexen Viereck schneiden sich die Diagonalen, sodass keine Gerade die Paare gegenüberliegender Ecken trennen kann. Enthält eine Halbebene alle drei Ecken eines Dreiecks, so enthält sie auch jeden Punkt in seinem Inneren.

<!--en:1e20371c-->
- **Lineare Halbräume in $\mathbb{R}^d$ haben VC-Dimension $d + 1$ (Übung :exref:`ex_generalization-classification-linear-vc`):** Sie zertrümmern die Menge aus dem Ursprung und den $d$ Standardbasisvektoren. Nach dem Satz von Radon lässt sich keine Menge von $d + 2$ Punkten in $\mathbb{R}^d$ zertrümmern.

<!--en:92f4365d-->
- **Für auswendig lernende Modelle mit beliebig großer Nachschlagetabelle gilt $v = \infty$:** Sie zertrümmern jede endliche Menge verschiedener Punkte.

<!--en:0afc182b-->
![Geraden in $\mathbb{R}^2$ zertrümmern drei nicht kollineare Punkte (links, alle $2^3 = 8$ Labelzuweisungen). Keine Menge aus vier Punkten lässt sich so zertrümmern (Mitte und rechts).](../img/mdl-clf-shattering.svg)

<!--en:28f11a0e-->
Nach dem *Lemma von Sauer* wächst für $n \geq v$ die Anzahl der verschiedenen Labelzuweisungen, die eine Klasse mit VC-Dimension $v$ auf $n$ Punkten realisieren kann, nur polynomiell. Sie beträgt höchstens $(en/v)^v$ statt $2^n$. Wenden wir eine symmetrisierte Form der booleschen Ungleichung auf diese Labelzuweisungen an (:numref:`sec_mdl-generalization-bounds`), erhalten wir die *VC-Schranke* (engl. VC bound). Für eine absolute Konstante $c$ gilt mit Wahrscheinlichkeit mindestens $1 - \delta$ über die zufällige Wahl des Trainingsdatensatzes $\mathcal{S}$:

<!--en:f34cb051-->
Dieser Satz beantwortet unsere Ausgangsfrage. Lineare Klassifikatoren in $\mathbb{R}^d$ haben endliche VC-Dimension $v = d + 1$. Bei $n \gg d$ garantiert ein kleiner Trainingsfehler daher einen kleinen Populationsfehler für *jede* Verteilung $P$. Für ein auswendig lernendes Modell mit unendlicher VC-Dimension gibt es keine solche Schranke. Allerdings müssen Schranken für die gleichmäßige Konvergenz im ungünstigsten Fall selbst für gezielt ungünstig gewählte Verteilungen gelten. Daher fällt :eqref:`eq_generalization-classification-vc` für natürliche Bilder numerisch grob aus. Für einen einzelnen Halbraum, der eine Klasse vom Rest trennt, ergibt sich bereits $\sqrt{785 / 60{,}000} \approx 0.11$ vor Multiplikation mit der Konstanten $c$. Die Kapazität aller zehn Klassen zusammen ist noch größer. Bei Fashion-MNIST beobachten wir dagegen nur eine Differenz von $0.01$ zwischen Trainings- und Testfehler. Für überparametrisierte tiefe neuronale Netze, die zufällige Label auswendig lernen können, verliert die Schranke jede Aussagekraft ($v \geq n$) :cite:`zhang2021understanding`. Auf diese Frage kommen wir in :numref:`sec_generalization_deep` zurück. Für präzise empirische Messungen verwenden wir zurückgehaltene Testdatensätze.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:21da077a-->
Auf einem unberührten Testdatensatz mit $n$ Beispielen hat der empirische Fehler $\epsilon_{\mathcal{D}}(f)$ eines festen Klassifikators den Standardfehler $\sqrt{\epsilon(1-\epsilon)/n} \leq 0.5/\sqrt{n}$. Um die Genauigkeit bei $95\%$ Konfidenz bis auf einen Prozentpunkt zu bestimmen, benötigen wir nach der Normalapproximation rund $10{,}000$ Testbeispiele und nach der Hoeffding-Schranke für endliche Stichproben rund $18{,}500$. Verwenden wir denselben Testdatensatz zur Auswahl unter $k$ festen Modellen, wächst der Unsicherheitsradius nur langsam gemäß $\sqrt{\log(2k/\delta)/(2n)}$. Entwerfen wir Modelle dagegen adaptiv anhand früherer Testergebnisse, können schon weit weniger Abfragen zur Überanpassung an den Testdatensatz führen. Um allein aus dem Trainingsfehler eine Generalisierungsschranke abzuleiten, misst die statistische Lerntheorie die Kapazität einer Hypothesenklasse durch die VC-Dimension $v$. Sie gibt die Größe der größten Punktmenge an, die die Klasse zertrümmern kann (für lineare Halbräume gilt $v = d + 1$ in $\mathbb{R}^d$). Daraus folgt eine gleichmäßige Schranke der Ordnung $O(\sqrt{v/n})$. Sie erklärt, warum lineare Modelle große Datensätze nicht auswendig lernen können. Für überparametrisierte tiefe Netze ist sie jedoch ohne Aussagekraft.

<!--en:2294243a-->
## Übungen

<!--en:81b67333-->
1. **Zwei Modelle, ein Testdatensatz.** Laut einer Veröffentlichung erreicht Modell A auf einem Testdatensatz von 10 000 Bildern eine Genauigkeit von 91.2%, Modell B erreicht 91.6%.
    1. Berechnen Sie für beide Genauigkeiten den Standardfehler und den 95%-Konfidenzradius der Normalapproximation. Ist der Unterschied von 0.4 Prozentpunkten aussagekräftig, wenn die Modelle auf zwei unabhängigen Testdatensätzen dieser Größe ausgewertet wurden?
    1. Tatsächlich wurden beide Modelle auf denselben 10 000 Bildern ausgewertet. Erklären Sie, warum es nun auf die Bilder ankommt, bei denen die Modelle zu unterschiedlichen Ergebnissen gelangen. Wie würden Sie die Frage entscheiden, wenn die Veröffentlichung für jedes Testbild angäbe, welche Modelle es richtig klassifiziert haben?
    1. Wie groß müsste ein Testdatensatz sein, damit eine Differenz von 0.4 Prozentpunkten zwischen zwei unabhängig ausgewerteten Modellen außerhalb des Konfidenzradius liegt?

<!--en:92a83d64-->
1. **Keine Fehler auf dem Testdatensatz.** Ein Klassifikator macht auf einem Testdatensatz mit $n$ Beispielen keinen Fehler.
    1. Geben Sie das Intervall der Normalapproximation an, wenn Sie $\epsilon_\mathcal{D}(f)$ anstelle von $\epsilon(f)$ einsetzen. Geben Sie auch das Hoeffding-Intervall an.
    1. Zeigen Sie, dass ein Klassifikator mit Populationsfehler $\epsilon$ auf $n$ unabhängigen Testbeispielen mit Wahrscheinlichkeit $(1 - \epsilon)^n \leq e^{-n\epsilon}$ keinen Fehler macht. Folgern Sie daraus, dass für $\epsilon \geq 3/n$ ein Testdatensatz ohne Fehler mit einer Wahrscheinlichkeit unter 0.05 auftritt (die „Dreierregel“).
    1. Vergleichen Sie $3/n$ mit dem Hoeffding-Radius für $n = 100$ und $n = 10{,}000$. Erklären Sie den Unterschied.

<!--en:004bc261-->
1. **Preisgabe der Testlabel.** Eine andere Person verwaltet einen Testdatensatz $\mathcal{D}$ mit $n$ Beispielen und binären Labeln. Sie gibt nur die Eingaben frei. Sie dürfen ein beliebiges Modell $f$ ohne Einschränkung der Modellklasse einreichen und erfahren nur seinen Testfehler $\epsilon_\mathcal{D}(f)$.
    1. Geben Sie eine Strategie an, mit der Sie jedes Testlabel rekonstruieren. Damit erhalten Sie unabhängig vom Populationsfehler ein Modell mit Testfehler $0$. Wie viele Einreichungen braucht Ihre Strategie?
    1. Jede Antwort $\epsilon_\mathcal{D}(f)$ nimmt einen von $n+1$ Werten an. Leiten Sie daraus eine untere Schranke für die Zahl der Einreichungen ab, die eine beliebige Strategie benötigt, um alle $2^n$ Labelzuweisungen zu unterscheiden.
   :exlabel:`ex_generalization-classification-leaking-test-set`

<!--en:a64688b0-->
1. [code] **Ein Validierungsdatensatz für Fashion-MNIST.** In diesem Kapitel haben wir die Modelle anhand der 10 000 Testbilder ausgewählt. Halten Sie stattdessen 10 000 der 60 000 Trainingsbilder für die Validierung zurück und trainieren Sie die Softmax-Regression auf den übrigen 50 000 Bildern.
    1. Wählen Sie die Lernrate aus $\{0.01, 0.03, 0.1, 0.3, 1\}$ anhand der Genauigkeit auf dem Validierungsdatensatz.
    1. Werten Sie das ausgewählte Modell einmal auf den Testbildern aus und vergleichen Sie seine Testgenauigkeit mit seiner Validierungsgenauigkeit. Liegt die Differenz innerhalb des Radius des 95%-Intervalls für 10 000 Beispiele?
    1. Erklären Sie, warum die einmal nach der Auswahl gemessene Testgenauigkeit eine erwartungstreue Schätzung der Genauigkeit des gewählten Modells liefert, die beste Validierungsgenauigkeit hingegen nicht.
   :exlabel:`ex_generalization-classification-validation-split`

<!--en:e5b9e697-->
1. **VC-Dimension linearer Klassifikatoren.** ● Betrachten Sie die linearen Klassifikatoren $f(\mathbf{x}) = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)$ auf $\mathbb{R}^d$.
    1. Zeigen Sie, dass sie die $d+1$ Punkte $\mathbf{0}, \mathbf{e}_1, \ldots, \mathbf{e}_d$ zertrümmern. Konstruieren Sie zu beliebigen Labeln $y_0, y_1, \ldots, y_d \in \{\pm 1\}$ Parameter $b$ und $\mathbf{w}$, die diese Label erzeugen. Hinweis: $f(\mathbf{0})$ hängt allein von $b$ ab. Wählen Sie also zuerst $b$.
    1. Seien $\mathbf{x}_1, \ldots, \mathbf{x}_{d+2} \in \mathbb{R}^d$ gegeben. Zeigen Sie, dass die $d + 2$ Vektoren $[\mathbf{x}_i, 1] \in \mathbb{R}^{d+1}$ linear abhängig sind. Sie entstehen, indem man an jedes $\mathbf{x}_i$ eine 1 anhängt. Daher gilt $\sum_i a_i [\mathbf{x}_i, 1] = \mathbf{0}$ für Koeffizienten $a_i$, die nicht alle null sind. Sei $I$ die Menge der Indizes mit $a_i > 0$ und $J$ die Menge der Indizes mit $a_i < 0$. Zeigen Sie, dass beide Mengen nicht leer sind und dass kein linearer Klassifikator den Punkten mit Indizes in $I$ das Label $+1$ und denen mit Indizes in $J$ das Label $-1$ zuweist. Hinweis: Werten Sie $\sum_i a_i (\mathbf{w}^\top \mathbf{x}_i + b)$ auf zwei Arten aus.
    1. Folgern Sie daraus, dass die VC-Dimension $d + 1$ beträgt. Zeigen Sie, dass sich die konvexen Hüllen von $\{\mathbf{x}_i : i \in I\}$ und $\{\mathbf{x}_j : j \in J\}$ schneiden.
   :exlabel:`ex_generalization-classification-linear-vc`

<!--en:c95a590c-->
1. **VC-Dimension von Polynomen.** Welche VC-Dimension hat die Klasse der Klassifikatoren $f(x) = \operatorname{sign}(p(x))$, wobei $p$ ein Polynom vom Grad höchstens fünf in $x \in \mathbb{R}$ ist? Wie groß ist sie für $\mathbf{x} \in \mathbb{R}^d$? Hinweis: Ein solcher Klassifikator ist ein linearer Klassifikator auf den Monomen von $\mathbf{x}$. Geben Sie anschließend an, welche Aussage die VC-Schranke anhand des Trainingsfehlers eines Polynomklassifikators fünften Grades auf Fashion-MNIST erlaubt und warum niemand einen solchen Klassifikator trainiert.

<!--en:a5698e8c-->
[Diskussionen](https://d2l.discourse.group/t/6829)

<!--en:f9b96ebe-->
Generalisierung bei der Klassifikation: Fehlerbalken, wiederholte Testnutzung und VC-Theorie<br>**empirischer Fehler und Populationsfehler · Intervalle nach dem zentralen Grenzwertsatz und nach Hoeffding · boolesche Ungleichung · adaptive Überanpassung · Zertrümmern und VC-Dimension**

<!--en:f58e5766-->
::: {.slide title="Zwei Hürden zwischen gemessener und wahrer Genauigkeit"}

<!--en:cf19954c-->
[Warum Generalisierung eine quantitative Theorie braucht]{.kicker}

<!--en:855e2278-->
Der Fashion-MNIST-Klassifikator erreicht $\sim 83\%$ auf $10{,}000$ Validierungsbildern und $\sim 84\%$ auf $60{,}000$ Trainingsbildern. Wie verlässlich sind diese Zahlen für künftige Kleidungsstücke?

<!--en:526f9544-->
- **Endliche Stichproben und wiederholte Testnutzung:** Selbst bei einem unberührten Testdatensatz mit $n = 10{,}000$ Bildern hat die Messung statistische Fehlerbalken. Wiederholte Abfragen zur Modellauswahl schwächen die Garantie.

<!--en:c9bc5577-->
- **Auswendiglernen oder Lernen:** Ein Modell mit Nachschlagetabelle erreicht $100\%$ Trainingsgenauigkeit ($0\%$ Fehler) und rät bei neuen Bildern. Warum spricht ein kleiner Trainingsfehler für ein *lineares* Modell, wenn beim Auswendiglernen selbst ein Fehler von null nichts aussagt?

<!--en:139e5e2a-->
::: {.slide title="Empirischer Fehler und Populationsfehler eines festen Klassifikators"}

<!--en:41523816-->
[Testfehler als Mittelwert von Bernoulli-Versuchen]{.kicker}

<!--en:4b317773-->
Einen einzelnen Klassifikator $f$ festlegen, **bevor** die unabhängigen, identisch verteilten Beispiele des Testdatensatzes $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n \sim P(X, Y)$ gezogen werden:

<!--en:b8b8637e-->
- **Erwartungstreuer Bernoulli-Mittelwert:** Jeder Indikator entspricht einem unabhängigen Versuch mit Verteilung $\mathrm{Bernoulli}(\epsilon(f))$. Für den Mittelwert gilt $\mathbb{E}[\epsilon_{\mathcal{D}}(f)] = \epsilon(f)$ mit Standardfehler
  $$\mathrm{SE}\big(\epsilon_{\mathcal{D}}(f)\big) = \sqrt{\frac{\epsilon(f)(1 - \epsilon(f))}{n}} \le \frac{0.5}{\sqrt{n}}$$

<!--en:f2557567-->
- **Benchmark ($\epsilon \approx 0.17, n = 10\text{k}$):** $\mathrm{SE} \approx 0.38\%$. Eine Halbierung erfordert $4\times$ so viele Testdaten.

<!--en:86d1d0d4-->
::: {.slide title="Asymptotische 95%-Intervalle nach dem zentralen Grenzwertsatz"}

<!--en:63957883-->
[Was n = 10 000 Testbeispiele messen]{.kicker}

<!--en:d2ce999e-->
Nach dem zentralen Grenzwertsatz weicht $\epsilon_{\mathcal{D}}(f)$ um höchstens $1.96$ Standardfehler von $\epsilon(f)$ ab, und zwar mit Wahrscheinlichkeit $\approx 0.95$:

<!--en:a729746a-->
- **Radius für Fashion-MNIST ($\epsilon = 0.17, n = 10\text{k}$):** $r_{\text{normal}} = 1.96 \times 0.0038 \approx \pm 0.7\%$.

<!--en:952fbd7f-->
- **Bedeutung von $n = 10{,}000$:** Im ungünstigsten Fall ($\epsilon = 1/2$) erfordert ein Radius $\le \pm 1\%$ mindestens so viele Testbilder, wie $n \ge (1.96 \times 0.5 / 0.01)^2 = 9{,}604$ angibt. Mit den je $10{,}000$ Testbildern von MNIST, Fashion-MNIST und CIFAR-10 lässt sich die Genauigkeit also bis auf etwa einen Prozentpunkt bestimmen.

<!--en:6a035b84-->
::: {.slide title="Garantien für endliche Stichproben durch die Hoeffding-Ungleichung"}

<!--en:65acd850-->
[Verteilungsfreie Konzentration]{.kicker}

<!--en:b0400748-->
Gilt auf einem kleinen Testdatensatz $\epsilon_{\mathcal{D}}(f) = 0$, schrumpft der Radius der Normalapproximation mit eingesetzter Fehlerschätzung auf $0$. Die **Hoeffding-Ungleichung** liefert eine exakte Schranke für endliche Stichproben bei jedem $n$ und $\epsilon(f) \in [0, 1]$:

<!--en:18299ba2-->
- **Exakter $(1 - \delta)$-Konfidenzradius:**
  $$r_{\text{Hoeffding}} = \sqrt{\frac{\log(2/\delta)}{2n}}$$

<!--en:fe60825b-->
- **Kosten der Garantie für endliche Stichproben:** Bei $95\%$ Konfidenz ($\delta = 0.05$) und $r \le 0.01$ verlangt Hoeffding $n \ge 18{,}445$ ($\approx 2\times$ so viele Beispiele wie bei der Normalapproximation mit $9{,}604$). Beide Rechnungen setzen die Varianz $1/4$ an, doch die exponentielle Schranke für die Randwahrscheinlichkeit ist weniger scharf als das Quantil der Normalverteilung.

<!--en:0ba3bbef-->
::: {.slide title="Empirische Überdeckung von Normal- und Hoeffding-Intervallen"}

<!--en:85cf3902-->
[Monte-Carlo-Simulation für verschiedene Testdatensatzgrößen]{.kicker}

<!--en:9aa86c3a-->
Je $10{,}000$ unabhängige Testdatensätze pro Größe $n \in [100, 10{,}000]$ für $\epsilon = 0.17$ und $\epsilon = 0.01$ simulieren und die tatsächliche Überdeckung der $95\%$-Intervalle vergleichen:

<!--en:40aea809-->
- **Mäßiger Fehler ($\epsilon = 0.17$, blau):** Das Normalintervall mit eingesetzter Fehlerschätzung (durchgezogen) erreicht annähernd $95\%$ Überdeckung bei jedem $n$. Das Hoeffding-Intervall (gestrichelt) ist konservativ (Überdeckung über $99.9\%$).

<!--en:9af34a50-->
- **Fehler nahe null ($\epsilon = 0.01$, orange):** Bei $n \le 200$ tritt in $\sim 13\%\text{--}37\%$ der Testdatensätze kein Fehler auf ($\epsilon_{\mathcal{D}} = 0$). Dadurch sinkt die Überdeckung des Normalintervalls mit eingesetzter Fehlerschätzung auf bis zu $\sim 63\%$.

<!--en:52ec2579-->
- **Hoeffdings Garantie gilt überall:** Die Überdeckung bleibt stets über der gepunkteten Linie bei $0.95$.

<!--en:8c575b0b-->
::: {.slide title="Vergleich von k festen Klassifikatoren mit der booleschen Ungleichung"}

<!--en:0f8b499a-->
[Selektionsbias bei gemeinsamem Testdatensatz]{.kicker}

<!--en:1db6ed10-->
Werden $k$ feste Modelle $f_1, \ldots, f_k$ auf *demselben* Testdatensatz $\mathcal{D}$ ausgewertet, steigt die Wahrscheinlichkeit einer Schrankenverletzung höchstens um den Faktor $k$ (**boolesche Ungleichung**):

<!--en:f3ceab64-->
Die rechte Seite gleich $\delta$ setzen. Daraus ergibt sich der simultane Konfidenzradius:

<!--en:5c827fce-->
- **Langsames Wachstum mit $\sqrt{\log k}$:** Bei $n = 10{,}000, \delta = 0.05$ wächst der Radius von $1.36\%$ ($k = 1$) auf $2.30\%$ ($k = 10^3$) und $2.96\%$ ($k = 10^6$).

<!--en:22651b46-->
::: {.slide title="Wie Selektionsbias den besten von k Münzwerfern begünstigt"}

<!--en:ad84eaa1-->
[Auswahl des Modells mit dem meisten Glück]{.kicker}

<!--en:14cadb7c-->
Selbst wenn alle $k$ Modelle vor der Auswertung auf $\mathcal{D}$ feststehen: Wer nur $\max_{1 \le j \le k} \text{acc}_{\mathcal{D}}(f_j)$ angibt, wählt das Modell mit dem meisten Glück aus.

<!--en:00596e8b-->
- **Experiment ($n = 1{,}000$ Label, $k = 10{,}000$ zufällig ratende Klassifikatoren):** Jeder Klassifikator wirft unabhängig eine faire Münze (wahre Populationsgenauigkeit $= 0.50$).

<!--en:097e731a-->
- **Logarithmischer Anstieg:** Der beste Testscore unter den ersten $k$ ratenden Klassifikatoren steigt stetig entlang $0.5 + \sqrt{\frac{\log k}{2n}}$ und erreicht **$\approx 0.56$** bei $k = 10{,}000$. Keiner hat ein Signal gelernt.

<!--en:2cfadfe8-->
::: {.slide title="Warum die Schranke bei adaptiver Überanpassung versagt"}

<!--en:d3c98ad0-->
[Wenn frühere Testscores den nächsten Modellentwurf bestimmen]{.kicker}

<!--en:288dfc5c-->
Fließen Ergebnisse auf dem Testdatensatz in den Entwurf des nächsten Modells ein (*adaptive Wiederverwendung*), versagt die $\sqrt{\log k}$-Schranke durch Überanpassung:

<!--en:5c87aca6-->
- **Von $50\%$ auf $\sim 70\%$ mit $1{,}000$ Abfragen:** Der Mehrheitsentscheid der Münzwerfer mit Testgenauigkeit $> 0.50$ erreicht **$\sim 70\%$ Testgenauigkeit**. Auf neuen Daten bleibt die Genauigkeit bei $50\%$.

<!--en:e683c45c-->
- **Dreiteilung der Daten:** Auf `train` trainieren, auf `val` auswählen und `test` **einmal** auswerten.

<!--en:31371f15-->
::: {.slide title="Gleichmäßige Konvergenz auf einer Hypothesenklasse"}

<!--en:4f546895-->
[Statistische Lerntheorie]{.kicker}

<!--en:d1b8d511-->
Das trainierte Modell $f_{\mathcal{S}} \in \mathcal{F}$ hängt vom Trainingsdatensatz $\mathcal{S}$ ab. Eine Garantie für $\epsilon(f_{\mathcal{S}})$ erfordert daher eine Schranke für die größte Abweichung, die gleichzeitig für *alle* $f \in \mathcal{F}$ gilt (**gleichmäßige Konvergenz**):

<!--en:2534a8c8-->
- **Endliche Hypothesenklassen ($|\mathcal{F}| < \infty$):** Nach der booleschen Ungleichung für $k = |\mathcal{F}|$ Hypothesen gilt mit Wahrscheinlichkeit $\ge 1 - \delta$ für alle $f \in \mathcal{F}$:
  $$\epsilon(f) \le \epsilon_{\mathcal{S}}(f) + \sqrt{\frac{\log(2|\mathcal{F}|/\delta)}{2n}}$$

<!--en:411af2f0-->
::: {.slide title="Von unendlichen Funktionenklassen zu endlich vielen Labelzuweisungen"}

<!--en:c0868215-->
[Warum Auswendiglernen scheitert und lineare Modelle generalisieren]{.kicker}

<!--en:f0fc2a88-->
Was geschieht bei $|\mathcal{F}| = \infty$, etwa für lineare Halbräume $f(\mathbf{x}) = \operatorname{sign}(\mathbf{w}^\top\mathbf{x} + b)$ mit kontinuierlichen Parametern auf $\mathbb{R}^d$?

<!--en:e164f8eb-->
- **Endlich viele Dichotomien auf $n$ Punkten:** Auf beliebigen $n$ Punkten $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}$ gibt es höchstens $2^n$ binäre Zuweisungen der Label $\pm 1$ (Vapnik und Chervonenkis, 1971).

<!--en:769d64d2-->
- **Zertrümmern und VC-Dimension:** $\mathcal{F}$ **zertrümmert** $m$ Punkte, wenn die Klasse darauf alle $2^m$ binären Labelzuweisungen realisiert. Die **VC-Dimension** $v = \mathrm{VC}(\mathcal{F})$ ist die größte Anzahl $m$ von Punkten, die $\mathcal{F}$ zertrümmern kann ($v = \infty$ für uneingeschränkt auswendig lernende Modelle).

<!--en:9d0a6625-->
::: {.slide title="Zertrümmern in der Ebene: Geraden haben VC-Dimension 3"}

<!--en:0899425e-->
[Geometrische Deutung der VC-Dimension]{.kicker}

<!--en:efcd8496-->
Orientierte Halbebenen $f(\mathbf{x}) = \operatorname{sign}(w_1 x_1 + w_2 x_2 + b)$ in $\mathbb{R}^2$ betrachten:

<!--en:96638d54-->
- **$\mathrm{VC} \ge 3$ (links):** Beliebige $3$ nicht kollineare Punkte in $\mathbb{R}^2$ lassen sich durch eine Gerade auf alle $2^3 = 8$ Arten aufteilen.

<!--en:96f2fe70-->
- **$\mathrm{VC} < 4$ (Mitte und rechts):** *Keine* Menge von $4$ Punkten in $\mathbb{R}^2$ lässt sich zertrümmern. Bilden die $4$ Punkte ein konvexes Viereck (Mitte), kann keine Gerade die beiden Paare diagonal gegenüberliegender Ecken trennen (`+` bzw. `-`). Liegt ein Punkt im Dreieck der anderen drei (rechts), enthält jede Halbebene mit den drei Ecken auch den inneren Punkt. Also gilt $\mathrm{VC} = 3$.

<!--en:3e944268-->
::: {.slide title="VC-Dimension linearer Klassifikatoren in d Dimensionen"}

<!--en:0e00b297-->
[Kapazität jenseits der Parameteranzahl]{.kicker}

<!--en:8dab4b63-->
In $\mathbb{R}^d$ haben lineare Halbräume $f(\mathbf{x}) = \operatorname{sign}(\mathbf{w}^\top\mathbf{x} + b)$ genau die folgende VC-Dimension:

<!--en:27298152-->
- **Untere Schranke ($v \ge d + 1$):** Dem Ursprung $\mathbf{0}$ und den $d$ Standardbasisvektoren $\mathbf{e}_1, \ldots, \mathbf{e}_d$ lassen sich unabhängig voneinander beliebige Label zuweisen.

<!--en:f6458173-->
- **Obere Schranke ($v \le d + 1$, Satz von Radon):** Beliebige $d + 2$ erweiterte Punkte $[\mathbf{x}_i, 1] \in \mathbb{R}^{d+1}$ sind linear abhängig. Sie zerfallen in zwei Teilmengen, deren konvexe Hüllen sich schneiden. Keine Hyperebene kann sie trennen.

<!--en:60dfb2eb-->
- **VC-Dimension $\neq$ Parameteranzahl:** $f_\omega(x) = \operatorname{sign}(\sin(\omega x))$ hat nur $1$ Parameter, aber $v = \infty$.

<!--en:40cd4c21-->
::: {.slide title="Das Lemma von Sauer und die VC-Generalisierungsschranke"}

<!--en:5b4a6d22-->
[Polynomielles Wachstum der Anzahl der Dichotomien]{.kicker}

<!--en:88c0594a-->
Warum garantiert eine endliche VC-Dimension $v < \infty$ gleichmäßige Konvergenz?

<!--en:a0bc6be6-->
- **Lemma von Sauer und Shelah:** Für $n \ge v$ ist die Anzahl der binären Labelzuweisungen, die $\mathcal{F}$ auf *beliebigen* $n$ Punkten realisieren kann, durch $\sum_{i=0}^v \binom{n}{i} \le (en/v)^v = O(n^v)$ beschränkt statt durch $2^n$.

<!--en:e05cb2f2-->
- **VC-Generalisierungsschranke:** Mit Wahrscheinlichkeit $\ge 1 - \delta$ gilt gleichzeitig für **jedes** $f \in \mathcal{F}$:
  $$\epsilon(f) \;\le\; \epsilon_{\mathcal{S}}(f) + c\sqrt{\frac{v + \log(1/\delta)}{n}}$$

<!--en:0123cd33-->
::: {.slide title="Stärken und Grenzen der VC-Theorie im Deep Learning"}

<!--en:0d5e7de8-->
[Grobe verteilungsfreie Schranken und Auswertung auf separaten Testdaten]{.kicker}

<!--en:4466992c-->
Was besagt die VC-Schranke für Fashion-MNIST und für tiefe Netze?

<!--en:0b9cc0d1-->
- **Warum lineare Modelle bei $n \gg d$ nicht auswendig lernen können:** Für einen Halbraum mit $d = 784, v = 785, n = 60{,}000$ ist $\sqrt{v/n} \approx 0.114$ eine Schranke für die Abweichung unter *jeder* Verteilung, bis auf die Konstante $c$ (beobachtete Abweichung sogar nur $\sim 0.01$).

<!--en:f1d76ac9-->
- **Keine aussagekräftige Schranke für tiefe Netze:** Überparametrisierte Netze können zufällige Label auf $n$ Punkten lernen ($v \ge n, \sqrt{v/n} \ge 1$). Scharfe Garantien im Deep Learning liefert **nach dem Training** ein bis dahin unberührter Testdatensatz ($\pm 0.7\%$ bei $n = 10{,}000$).

<!--en:9b37182e-->
::: {.slide title="Verständnisfragen zu Stichprobengröße, null Fehlern und VC-Theorie"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:b86a3cbe-->
Drei Aufgaben zur Statistik von Testdatensätzen und zur VC-Theorie:

<!--en:f5a9eb3d-->
1. **Halbierung des Konfidenzradius:** Wie viele Testbeispiele $n$ sind für einen $95\%$-Konfidenzradius von höchstens $\pm 0.5$ Prozentpunkten ($t = 0.005$) im ungünstigsten Fall ($\epsilon = 1/2$) nötig, einmal nach der Normalapproximation und einmal nach der Hoeffding-Ungleichung?

<!--en:2c94a961-->
2. **Die „Dreierregel“ bei null Testfehlern (Übung 2):** Ein Klassifikator macht **$0$ Fehler** bei $n = 100$ Testbeispielen. Warum führt das Normalintervall mit eingesetzter Fehlerschätzung $0 \pm 0$ in die Irre? Für welchen Populationsfehler $\epsilon$ gilt $(1 - \epsilon)^n \le e^{-n\epsilon} \le 0.05$?

<!--en:26cba7c6-->
3. **Quadratische Entscheidungsgrenzen:** Welche VC-Dimension haben quadratische Klassifikatoren $f(\mathbf{x}) = \operatorname{sign}(w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2)$ in der Ebene $\mathbb{R}^2$?

<!--en:dedf2694-->
**Lösungen:** **(1)** Halbiert sich der Radius von $0.01$ auf $0.005$, wächst $n$ um den Faktor $4$. Die Normalapproximation verlangt $(1.96 \times 0.5 / 0.005)^2 = 38{,}416$ Beispiele, die Hoeffding-Ungleichung $\log(40)/(2 \times 0.005^2) \approx 73{,}778$. **(2)** Bei $0$ beobachteten Fehlern verschwindet die aus den Daten geschätzte Varianz. Aus $e^{-n\epsilon} \le 0.05$ $\iff$ $\epsilon \ge \frac{\log 20}{n} \approx \frac{3}{n}$ folgt die **Dreierregel mit Intervall $[0,\; 3/n]$** ($[0, 3\%]$ bei $n = 100$, schärfer als Hoeffdings $O(1/\sqrt{n})$-Schranke von $\approx 13.6\%$). **(3)** In den $5$ Dimensionen der Merkmalsabbildung $\phi(x_1, x_2) = (x_1, x_2, x_1^2, x_1 x_2, x_2^2)$ ist der Klassifikator ein linearer Halbraum mit Bias $w_0$. Also gilt $\mathrm{VC} = 5 + 1 = 6$.

<!--en:f546fe7a-->
::: {.slide title="Generalisierungsgarantien bei der Klassifikation im Überblick"}

<!--en:27de615d-->
[Zusammenfassung des Abschnitts]{.kicker}

<!--en:5db3ac71-->
| Situation | Fehlerschranke / Radius ($95\%$ Konfidenz, $\delta = 0.05$) | Skalierungsgesetz und praktische Bedeutung |

<!--en:6139ddee-->
| **Ein festes Modell (Normalapproximation)** | $\big|\epsilon_{\mathcal{D}}(f) - \epsilon(f)\big| \le 1.96\sqrt{\epsilon(1-\epsilon)/n} \le 0.98/\sqrt{n}$ | $\pm 0.7\%$ bei Fashion-MNIST ($n = 10\text{k}$), $n \ge 9{,}604$ für $\pm 1\%$ im ungünstigsten Fall |

<!--en:fdcdeefb-->
| **Ein festes Modell (Hoeffding)** | $\big|\epsilon_{\mathcal{D}}(f) - \epsilon(f)\big| \le \sqrt{\log(2/\delta)/(2n)}$ | Exakt für alle Stichprobengrößen und Fehlerraten ($n, \epsilon \in [0, 1]$), $n \ge 18{,}445$ für $\pm 1\%$ ($\approx 2\times$ so großes $n$ wie bei der Normalapproximation) |

<!--en:0df281e3-->
| **Bestes von $k$ festen Modellen (boolesche Ungleichung)** | $\max_{1\le j\le k}\big|\epsilon_{\mathcal{D}}(f_j) - \epsilon(f_j)\big| \le \sqrt{\log(2k/\delta)/(2n)}$ | Langsames Wachstum mit $\sqrt{\log k}$, der beste von $10\text{k}$ Klassifikatoren mit fairer Münze erreicht $\approx 0.56$ |

<!--en:a82b325a-->
| **Adaptive Wiederverwendung des Testdatensatzes** | Schranke aus der booleschen Ungleichung versagt (Überanpassung mit $O(n)$ Abfragen möglich) | Mehrheitsentscheid unter $1\text{k}$ vorselektierten ratenden Klassifikatoren erreicht $\sim 0.70$, Hyperparameter nur auf Validierungsdaten optimieren |

<!--en:45886bd1-->
| **Endliche Hypothesenklasse $\mathcal{F}$** | $\sup_{f\in\mathcal{F}}\big|\epsilon_{\mathcal{S}}(f) - \epsilon(f)\big| \le \sqrt{\log(2|\mathcal{F}|/\delta)/(2n)}$ | Gleichmäßige Konvergenz für alle $f \in \mathcal{F}$ einschließlich des trainierten Modells $f_{\mathcal{S}}$ |

<!--en:9308bae3-->
| **VC-Dimension $v = \mathrm{VC}(\mathcal{F})$** | $\epsilon(f) \le \epsilon_{\mathcal{S}}(f) + c\sqrt{(v + \log(1/\delta))/n}$ | $v = d + 1$ für lineare Klassifikatoren in $\mathbb{R}^d$, $v = \infty$ beim Auswendiglernen, für tiefe Netze ohne Aussagekraft |
