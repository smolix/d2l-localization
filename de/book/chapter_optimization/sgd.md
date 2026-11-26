<!--en:8fd1c5f0-->
# Stochastischer Gradientenabstieg und Minibatch-Gradientenabstieg

<!--en:f655d9b5-->
In :numref:`sec_gd` haben wir für jedes Parameterupdate den exakten Gradienten der Zielfunktion verwendet. Dazu müssen wir den Verlust bei jedem Schritt über den gesamten Trainingsdatensatz auswerten. Im Deep Learning ist der Gradientenabstieg auf dem gesamten Trainingssatz zu rechenaufwendig. Trainieren wir ResNet-50 auf den $1.28$ Millionen Bildern von ImageNet über $90$ Epochen mit Minibatches von je $256$ Bildern, benötigen wir $450{,}000$ Updates :cite:`Goyal.Dollar.Girshick.ea.2017`. Dagegen würden $450{,}000$ Schritte auf dem gesamten Trainingssatz $450{,}000$ vollständige Durchläufe durch den Datensatz erfordern, also den $5{,}000$-fachen Rechenaufwand.

<!--en:446ef162-->
Der **stochastische Gradientenabstieg** (engl. stochastic gradient descent, SGD) schätzt den Gradienten anhand eines zufällig gezogenen Beispiels oder eines kleinen **Minibatches** aus $b$ Beispielen (:numref:`subsec_linear-regression-minibatch-sgd`). Die Schätzung ist erwartungstreu, und ihre Varianz sinkt proportional zu $1/b$. Das Rauschen durch die Zufallsauswahl verändert jedoch den Verlauf der Optimierung in der Spätphase grundlegend. Bei konstanter Lernrate $\eta$ konvergieren die Iterierten nicht gegen einen einzelnen Punkt. Sie schwanken innerhalb einer stationären **Kugel um das Minimum** (engl. noise ball).

<!--en:a96a3133-->
In diesem Abschnitt untersuchen wir Gradientenrauschen bei ungleichmäßiger Krümmung. Am zweidimensionalen Tal aus :numref:`sec_optimization-intro` leiten wir die stationäre Varianz von SGD mit konstanter Schrittweite für jede Krümmungsrichtung her. Die daraus folgende Vorhersage für den Überschussverlust (engl. excess loss) prüfen wir an einem realen Regressionsdatensatz. Wir untersuchen, wie das Ziehen ohne Zurücklegen und die Parallelität der Hardware die Wahl der Minibatches beeinflussen. Daraus ergibt sich die Motivation für den Lernratenabfall (:numref:`sec_scheduler`) und für Momentum (:numref:`sec_momentum`).

<!--en:726e1d90-->
## Stochastischer Gradientenabstieg

<!--en:09ceefe5-->
### Gradienten aus Stichproben im Tal

<!--en:70986b6a-->
Für ein empirisches Risiko $f(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n f_i(\mathbf{x})$ über $n$ Trainingsbeispiele ist der exakte Gradient der Mittelwert der Gradienten der einzelnen Beispiele:

<!--en:8670e650-->
Bei SGD mit Einzelbeispielen ziehen wir in jedem Schritt einen Index $i \in \{1, \ldots, n\}$ aus der Gleichverteilung und aktualisieren den Parametervektor nach der Vorschrift

<!--en:6acd284e-->
Da $\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$ gilt, lässt sich der stochastische Gradient als $\nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}) + \boldsymbol{\xi}$ schreiben. Das Gradientenrauschen $\boldsymbol{\xi}$ hat den Erwartungswert null.

<!--en:baf9cd19-->
Um zu sehen, wie $\boldsymbol{\xi}$ mit der Krümmung zusammenwirkt, definieren wir zunächst `sgd_2d`. Die Funktion addiert zu beiden Gradientenkoordinaten unabhängiges gaußsches Rauschen mit Erwartungswert null und Varianz eins:

<!--en:48ddbe2b-->
Anschließend führen wir $300$ Schritte mit `sgd_2d` im Tal $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ (:numref:`sec_optimization-intro`) aus. Wir wählen die Lernrate $\eta = 0.1$ und starten bei $(-5, -2)$:

<!--en:445a78a4-->
In den ersten $\sim 100$ Schritten dominiert der deterministische Gradient, und die Iterierte nähert sich dem Ursprung. In dessen Nähe wird $\|\nabla f(\mathbf{x})\|$ klein, während die Rauschamplitude konstant bleibt. Die Iterierte bleibt dadurch in einer länglichen Punktwolke entlang der flachen Achse $x_1$.

<!--en:1f96e2a1-->
### Die Kugel um das Minimum

<!--en:53bc1bf4-->
Die Ausdehnung dieser Wolke lässt sich analytisch berechnen. Betrachten wir einen Eigenvektor mit Krümmung $\lambda > 0$ und unabhängiges Gradientenrauschen $\xi_t$ mit Erwartungswert null und Varianz $\sigma^2$. Entlang dieses Eigenvektors gilt für einen SGD-Schritt

<!--en:270b04d1-->
Quadrieren wir beide Seiten und bilden den Erwartungswert, erhalten wir mit $\mathbb{E}[x_t \xi_t] = 0$ die Rekursion für das zweite Moment:

<!--en:1bbcbfde-->
Für jede stabile Schrittweite $0 < \eta < 2/\lambda$ fällt der Erwartungswert $\mathbb{E}[x_t]$ auf null. Das zweite Moment konvergiert gegen die stationäre Varianz. Dort gleichen sich die deterministische Kontraktion und das neu hinzukommende Rauschen aus. Aus $\mathbb{E}[x_\infty^2] = (1 - \eta\lambda)^2 \mathbb{E}[x_\infty^2] + \eta^2 \sigma^2$ folgt:

<!--en:c8f86ee2-->
:eqref:`eq_sgd-noise-ball` beschreibt die **Kugel um das Minimum** (die vollständige Trajektorie für endliche Zeiten leiten wir in :numref:`subsec_mdl-stochastic-gradients` her):

<!--en:d5b94bb6-->
1. **Die Stabilitätsgrenze bleibt erhalten:** Für $\eta \to 2/\lambda$ geht der Nenner gegen null ($(2 - \eta\lambda) \to 0$), und die Varianz divergiert.

<!--en:130b7049-->
2. **Inverse Skalierung mit der Krümmung:** Die stationäre Varianz ist umgekehrt proportional zur Krümmung $\lambda$, solange die Schrittweite hinreichend klein ist. Im Tal beträgt die stationäre Varianz bei $\eta = 0.1$ und $\sigma^2 = 1$ etwa $0.1 / (0.2 \times 1.98) \approx 0.2525$ entlang der flachen Koordinate $x_1$ ($\text{std} \approx 0.50$) und $0.1 / (4 \times 1.6) \approx 0.0156$ entlang der steilen Koordinate $x_2$ ($\text{std} \approx 0.125$). Flache Richtungen haben zwei Nachteile. In der Anfangsphase ist die Konvergenz langsam, und im stationären Zustand sind die Schwankungen groß.

<!--en:fc91a634-->
3. **Lineare Skalierung mit der Schrittweite:** Für $\eta\lambda \ll 1$ halbiert eine Halbierung der Lernrate $\eta$ die stationäre Varianz in jeder Richtung. Das bestätigt eine Simulation mit $10{,}000$ unabhängigen Trajektorien:

<!--en:0affc5da-->
### Abnehmende Lernraten

<!--en:0ee29bdd-->
Bei SGD mit konstanter Schrittweite besteht ein Zielkonflikt. Eine kleinere Lernrate $\eta$ verringert die Varianz innerhalb der Kugel um das Minimum ($\propto \eta$). Zugleich verlangsamt sie die anfängliche Konvergenz, weil die Kontraktionsrate $\eta\lambda$ pro Schritt sinkt. Damit die rauschbedingte Untergrenze verschwindet, muss die Lernrate $\eta_t$ mit der Zeit abnehmen. :citet:`Robbins.Monro.1951` zeigten für glatte, stark konvexe Zielfunktionen mit beschränkter Gradientenvarianz: SGD konvergiert gegen den exakten Minimierer, wenn die Schrittweiten folgende Bedingungen erfüllen:

<!--en:8f847d8f-->
Die erste Bedingung sichert ein unendliches Schrittweitenbudget, sodass die Iterierte das Optimum von jeder Initialisierung aus erreichen kann. Die zweite Bedingung stellt sicher, dass die kumulierte Varianz des hinzukommenden Rauschens $\propto \sum_t \eta_t^2$ endlich bleibt. Polynomiale Lernratenpläne wie $\eta_t = c / (t + t_0)$ mit $c > 1/(2\lambda_{\min})$ erfüllen beide Bedingungen. Der erwartete quadratische Fehler sinkt dabei gemäß $\mathbb{E}[\|\mathbf{x}_t - \mathbf{x}^\star\|^2] = O(1/t)$ (:numref:`subsec_mdl-step-size-decay` und Übung :exref:`ex_sgd-one-over-t`). Bei einem festen Trainingsbudget von $T$ Schritten verwendet man im Deep Learning anstelle asymptotischer $1/t$-Pläne in der Praxis kosinusförmige oder lineare Lernratenpläne oder Pläne mit Aufwärmphase, konstanter Lernrate und anschließendem Abfall (:numref:`sec_scheduler`).

<!--en:a300530f-->
## Minibatches

<!--en:1232eb14-->
### Gradientenvarianz und Batchgröße

<!--en:d3b8bee2-->
Bei Minibatch-SGD verwenden wir statt eines einzelnen Beispiels eine zufällige Teilmenge $\mathcal{B}_t$ aus $b$ Beispielen und mitteln deren Gradienten:

<!--en:179c1b29-->
Sei $\boldsymbol{\Sigma}(\mathbf{x}) = \mathbb{E}_i\big[(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))^\top\big]$ die Kovarianzmatrix der Gradienten der einzelnen Beispiele. Ihre Spur $\operatorname{tr}\boldsymbol{\Sigma} = \mathbb{E}_i\|\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2$ ist die über alle Koordinaten summierte Gradientenvarianz pro Beispiel. Ziehen wir $b$ Beispiele unabhängig mit Zurücklegen, so teilt die Mittelung die Kovarianz durch $b$:

<!--en:cddb18dc-->
Setzen wir $\sigma^2 \leftarrow \sigma^2 / b$ in :eqref:`eq_sgd-noise-ball` ein, erhalten wir für Minibatches die stationäre Varianz in einer Richtung mit Krümmung $\lambda$:

<!--en:7feea5a7-->
Bei kleinen Schritten ($\eta\lambda \ll 1$) bestimmen Lernrate und Batchgröße die stationäre Kugel um das Minimum allein über ihr Verhältnis **$\eta / b$**. Eine Verdopplung der Batchgröße $b$ hat also denselben Effekt auf die rauschbedingte Untergrenze wie eine Halbierung der Lernrate $\eta$.

<!--en:480d07a3-->
### Überschussverlust an realen Daten

<!--en:b3dd1a58-->
Um wie viel übersteigt der Trainingsverlust durch die Schwankungen innerhalb der Kugel um das Minimum seinen Minimalwert $f(\mathbf{x}^\star)$? Betrachten wir den $k$-ten Eigenvektor der Hesse-Matrix $\mathbf{H}$ mit Eigenwert $\lambda_k$ und Gradientenvarianz $\sigma_k^2$ pro Beispiel. Eine stationäre Auslenkung mit Varianz $v_k \approx \frac{\eta \sigma_k^2}{2 \lambda_k b}$ in dieser Richtung trägt $\frac{1}{2}\lambda_k v_k$ zum erwarteten quadratischen Verlust bei:

<!--en:6c3053f2-->
Die Krümmung $\lambda_k$ kürzt sich heraus. In flachen Richtungen streuen die Parameter stärker (Varianz $\propto 1/\lambda_k$), doch eine Auslenkung um eine Einheit erhöht den Verlust dort weniger ($\propto \lambda_k$). Summieren wir die Beiträge aller $d$ orthogonalen Eigenvektoren ($\sum_{k=1}^d \sigma_k^2 = \operatorname{tr}\boldsymbol{\Sigma}$), erhalten wir den stationären **Überschussverlust**:

<!--en:9671d8be-->
Wir prüfen :eqref:`eq_sgd-excess-loss` am Regressionsdatensatz Airfoil Self-Noise der NASA. Er enthält $n = 1{,}500$ Windkanalmessungen mit $5$ standardisierten Merkmalen. Pro Messung gibt es $1$ standardisierten Zielwert für den Schalldruck:

<!--en:ff97bc25-->
Für die lineare Regression mit quadratischem Verlust fassen wir den Bias als $6$. Gewicht zu einem konstanten Merkmal auf. Wir berechnen die exakte Hesse-Matrix $\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$, den Minimierer in geschlossener Form $\mathbf{w}^\star$, die größte Krümmung $L = \lambda_{\max}(\mathbf{H}) \approx 2.10$ und die Gradientenvarianz pro Beispiel $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) \approx 3.24$:

<!--en:db43ec2e-->
Um den empirischen Überschussverlust mit dieser Vorhersage zu vergleichen, simuliert `sgd_runs` mit `num_runs=400` mehrere Minibatch-SGD-Trajektorien in vektorisierter Form. Die Beispiele ziehen wir mit Zurücklegen. Die Funktion `excess_loss` berechnet $\frac{1}{2}(\mathbf{w} - \mathbf{w}^\star)^\top \mathbf{H} (\mathbf{w} - \mathbf{w}^\star)$:

<!--en:e0c003da-->
Wir starten bei $\mathbf{w}^\star$ mit der Batchgröße $b = 10$ und mitteln `excess_loss` über die dritte Epoche ($150$ Schritte). Für die Lernraten $\eta \in \{0.05, 0.025\}$ vergleichen wir die Ergebnisse mit $\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$:

<!--en:836db8ca-->
Bei $b = 10$ stimmen die gemessenen Überschussverluste (etwa $0.004$ bei $\eta = 0.05$ und $0.002$ bei $\eta = 0.025$) gut mit der Vorhersage $\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$ überein. Im **interpolierenden Regime** (engl. interpolating regime) hat das Modell genug Kapazität, um jedes Trainingsbeispiel exakt anzupassen. Dann gilt $\nabla f_i(\mathbf{x}^\star) = \mathbf{0}$ für alle $i$. In diesem Grenzfall gilt $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) = 0$. Die Kugel um das Minimum schrumpft auf einen Punkt, wenn sich die Iterierte $\mathbf{x}^\star$ nähert, und SGD mit konstanter Schrittweite konvergiert geometrisch gegen den exakten Minimierer. Überparametrisierte Klassifikatoren nähern sich diesem Regime beim Training über viele Epochen. Große Sprachmodelle hingegen durchlaufen beim Training riesige Korpora nur einmal und bleiben im rauschbegrenzten Regime mit $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) > 0$.

<!--en:197bb3a8-->
### Ziehen ohne Zurücklegen

<!--en:48966da6-->
In der Praxis ziehen Datenlader Minibatches nicht unabhängig mit Zurücklegen. Sie mischen den Datensatz zu Beginn jeder Epoche zufällig und teilen ihn in disjunkte Minibatches der Größe $b$ auf. Ziehen wir $b$ verschiedene Beispiele ohne Zurücklegen aus einer endlichen Grundgesamtheit der Größe $n$, sind die Gradientenfehler der einzelnen Beispiele negativ korreliert, denn es gilt $\sum_{i=1}^n (\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})) = \mathbf{0}$. Für die Varianz des Minibatch-Gradienten aus :eqref:`eq_sgd-variance` ergibt sich dadurch ein Korrekturfaktor für die endliche Grundgesamtheit (Übung :exref:`ex_sgd-finite-population`):

<!--en:d40c0ffd-->
Für $b \ll n$ ($10$ von $1{,}500$ Beispielen) liegt dieser Faktor nahe bei eins ($0.994$). Für $b = n$ verschwindet die Varianz, wie es beim Gradientenabstieg auf dem gesamten Trainingssatz sein muss. Das zufällige Mischen bewirkt außerdem, dass sich die Fehler der Minibatch-Gradienten über die $n/b$ Schritte einer Epoche in erster Ordnung aufheben. Bei kleinen Lernraten ist der stationäre Fehler deshalb geringer als beim Ziehen mit Zurücklegen :cite:`Mishchenko.Khaled.Richtarik.2020`.

<!--en:beedd71c-->
### Batchgröße und Hardwaredurchsatz

<!--en:9ac2de5a-->
Sei $\eta\lambda \ll 1$, und halten wir $\eta / b$ konstant. Ein Schritt mit einem Minibatch der Größe $b$ und der Lernrate $\eta$ verringert den erwarteten Fehler auf das $(1 - \eta\lambda)$-Fache. Er führt in dieselbe Kugel um das Minimum wie $b$ Schritte mit je einem Beispiel und der Lernrate $\eta / b$, denn es gilt $(1 - \eta\lambda / b)^b \approx 1 - \eta\lambda$. Zählen wir nur die Gradientenauswertungen für einzelne Beispiele, so beschleunigt oder verlangsamt eine Vergrößerung von $b$ in diesem linearen Bereich das Lernen nicht.

<!--en:d3283d1c-->
Minibatches sind unverzichtbar, weil **sich die Fixkosten der Hardware auf mehrere Beispiele verteilen**. Bei jedem Parameterupdate fallen unabhängig von $b$ feste Kosten für die Aufrufsteuerung in Python und die Kernelstarts an. Vektorprozessoren (CPUs und GPUs) verarbeiten $b$ Beispiele parallel und lesen die Modellgewichte dabei nur einmal aus dem Speicher (:numref:`chap_performance`). Am Airfoil-Datensatz messen wir sowohl die Laufzeit pro Epoche als auch den Verlust in Abhängigkeit von der Zahl der verarbeiteten Beispiele. Dazu trainiert `train_sgd` ein lineares Regressionsmodell mit Minibatch-SGD ab $\mathbf{w} = \mathbf{0}$. Die Funktion gibt den Verlust auf dem gesamten Datensatz nach jedem Update sowie die Sekunden pro Epoche zurück:

<!--en:6ab9e743-->
Wir führen `train_sgd` für $b \in \{1, 10, 100\}$ mit festem Verhältnis $\eta / b = 0.005$ aus. Zum Vergleich verwenden wir den Gradientenabstieg auf dem gesamten Trainingssatz ($b = 1{,}500$ bei $\eta = 0.9 < 2/L \approx 0.95$):

<!--en:70bea9ba-->
Tragen wir den Trainingsverlust gegen die Gesamtzahl der verarbeiteten Beispiele ($b \times \text{updates}$) auf, können wir die Dateneffizienz aller vier Batchgrößen in derselben Grafik vergleichen:

<!--en:3d53bde8-->
Ein Update mit $b = 100$ Beispielen dauert ungefähr so lange wie eines mit $b = 1$. Bei $b = 100$ läuft eine Epoche dadurch fast $50\text{--}100\times$ so schnell ab, und der Verlust folgt pro verarbeitetem Beispiel derselben Kurve. Hier zeigt sich auch die Grenze der linearen Skalierung $\eta \propto b$. Bei festem Verhältnis $\eta / b = 0.005$ steigt die Lernrate für $b = 100$ auf $\eta = 0.5$. Damit ist $\eta L = 0.5 \times 2.10 = 1.05$ nicht mehr klein gegenüber $2$, und der Faktor $(2 - \eta\lambda)^{-1}$ in :eqref:`eq_sgd-noise-ball` wächst. Jede Batchgröße oberhalb von $b \approx 190$ verletzt die Stabilitätsgrenze $\eta < 2/L \approx 0.95$. Den Übergang an dieser kritischen Batchgröße untersuchen wir in :numref:`sec_batch_size` systematisch.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:73a77c60-->
Minibatch-SGD schätzt den Gradienten anhand von $b$ Beispielen. Die Stichprobenvarianz sinkt dadurch auf $\operatorname{tr}(\boldsymbol{\Sigma})/b$, beim Ziehen ohne Zurücklegen kommt der Faktor $(n-b)/(n-1)$ hinzu. Zugleich verteilen sich die Kosten für die Aufrufsteuerung und die beanspruchte Speicherbandbreite pro Schritt auf den ganzen Minibatch. Bei konstanter Lernrate $\eta$ nähern sich die Iterierten einer stationären **Kugel um das Minimum**. Die Koordinatenvarianz in einer Richtung mit Krümmung $\lambda$ beträgt $\frac{\eta\sigma^2}{b\lambda(2 - \eta\lambda)} \approx \frac{\eta\sigma^2}{2\lambda b}$. Die stärksten Schwankungen treten in den flachsten Richtungen auf. Summieren wir die Beiträge zum quadratischen Verlust über alle Eigenrichtungen, kürzt sich $\lambda$ heraus. Zurück bleibt ein erwarteter stationärer **Überschussverlust** von $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$. Er verschwindet nur, wenn das Modell im Optimum jedes Trainingsbeispiel interpoliert. Für $\eta L \ll 1$ hängen sowohl der anfängliche Fortschritt pro Beispiel als auch die stationäre rauschbedingte Untergrenze von $\eta$ und $b$ nur über deren Verhältnis $\eta/b$ ab. Diese Skalierung stößt an ihre Grenze, wenn wir $b$ mit $\eta \propto b$ so weit erhöhen, dass sich die Lernrate der deterministischen Stabilitätsgrenze $\eta < 2/L$ nähert.

<!--en:2294243a-->
## Übungen

<!--en:62bf73a2-->
1. **Rauschen als zufälliger Datensatz.** Zeigen Sie, dass `sgd_2d` auf $f(\mathbf{x}) = 0.1 x_1^2 + 2x_2^2$ mit gaußschem Gradientenrauschen der Varianz eins äquivalent zu SGD mit Einzelbeispielen auf $f(\mathbf{x}, \mathbf{w}) = 0.1 (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ mit $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{C})$ ist. Bestimmen Sie die Kovarianzmatrix $\mathbf{C}$.
   :exlabel:`ex_sgd-noise-dataset`

<!--en:7a1d1555-->
1. [code] **Die Konstante eines $1/t$-Lernratenplans.** ● Betrachten Sie eine eindimensionale quadratische Funktion mit Krümmung $\lambda$, Gradientenrauschen der Varianz $\sigma^2$ und Lernrate $\eta_t = c / (t + t_0)$.
   :exlabel:`ex_sgd-one-over-t`
    1. Geben Sie die exakte Rekursion für $e_t = \mathbb{E}[x_t^2]$ an. Zeigen Sie mit dem asymptotischen Ansatz $e_t \approx A/t$, dass $A = c^2 \sigma^2 / (2 c \lambda - 1)$ gilt, sofern $c > 1/(2\lambda)$ erfüllt ist. Welcher Wert von $c$ minimiert $A$?
    1. Iterieren Sie die deterministische Rekursion für $e_t$ bei $\lambda = 0.2, \sigma = 1$ ab dem Anfangswert $e_0 = 25$ mit $t_0 = 20$ und $c \in \{1, 2.5, 5, 10\}$ bis $t = 10^6$. Tragen Sie $t\,e_t$ gegen $t$ in einem doppeltlogarithmischen Diagramm auf. Erklären Sie, warum kein Wert von $c$ gleichzeitig für $\lambda_1 = 0.2$ und $\lambda_2 = 4$ optimal ist.

<!--en:f290484c-->
1. [code] **Korrekturfaktor für die endliche Grundgesamtheit und zufälliges Mischen.**
    1. Sei $\mathbf{e}_i = \nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})$. Zeigen Sie mit $\sum_{i=1}^n \mathbf{e}_i = \mathbf{0}$, dass $\mathbb{E}_{i \neq j}[\mathbf{e}_i^\top \mathbf{e}_j] = -\operatorname{tr}\boldsymbol{\Sigma}/(n - 1)$ gilt, und leiten Sie :eqref:`eq_sgd-finite-population` her. Überprüfen Sie die Formel empirisch an den Gradienten `G` der einzelnen Beispiele des Airfoil-Datensatzes. Ziehen Sie dazu jeweils $b \in \{10, 100, 500, 1000\}$ Beispiele mit und ohne Zurücklegen.
       :exlabel:`ex_sgd-finite-population`
    1. Ziehen Sie $n = 100$ Vektoren $\mathbf{w}_i$ aus der Verteilung in Übung :exref:`ex_sgd-noise-dataset`. Vergleichen Sie die SGD-Trajektorien über 20 Epochen beim Ziehen mit Zurücklegen und beim zufälligen Mischen zu Beginn jeder Epoche für $\eta \in \{0.1, 0.01\}$.
       :exlabel:`ex_sgd-reshuffling`

<!--en:2ed4333a-->
1. [code] **Gradientenvarianz während des Trainings.** ●
   :exlabel:`ex_sgd-variance-training`
    1. Vergleichen Sie für das lineare Modell auf den Airfoil-Daten die Werte von $\operatorname{tr}\boldsymbol{\Sigma}$ bei $\mathbf{w} = \mathbf{0}$ und bei $\mathbf{w} = \mathbf{w}^\star$.
    1. Trainieren Sie ein zweischichtiges MLP mit ReLU-Aktivierung ($64$ versteckte Einheiten) auf den Airfoil-Daten über $10$ Epochen ($b = 10, \eta = 0.05$). Vergleichen Sie $\operatorname{tr}\boldsymbol{\Sigma}$, aufgeschlüsselt nach `W1`, `b1`, `W2` und `b2`, bei der Initialisierung und nach dem Training.

<!--en:9477e62a-->
1. [code] **Batchgröße und Skalierung der Lernrate.** ● Setzen Sie für das lineare Regressionsmodell auf den Airfoil-Daten $\eta = 0.05$ bei $b = 10$. Testen Sie $b \in \{1, 10, 100, 1500\}$ sowohl mit linearer Skalierung ($\eta \propto b$) als auch mit Skalierung nach der Quadratwurzel ($\eta \propto \sqrt{b}$). Erklären Sie anhand von $L = 2.10$, bei welchen Einstellungen das Verfahren divergiert. Berechnen Sie den exakten stationären Überschussverlust über alle $6$ Eigenvektoren von `H` mithilfe von :eqref:`eq_sgd-noise-ball`. Erklären Sie damit, warum die rauschbedingte Untergrenze bei $b = 100, \eta = 0.5$ etwas höher liegt als bei $b = 10, \eta = 0.05$.
   :exlabel:`ex_sgd-batch-lr`

<!--en:4f71e3ad-->
[Diskussion](https://d2l.discourse.group/t/497)

<!--en:57b439c6-->
[Diskussion](https://d2l.discourse.group/t/1067)

<!--en:b0d9b1a2-->
Stochastischer Gradientenabstieg und Minibatch-Gradientenabstieg<br>**Gradienten aus Stichproben · stationäre Kugel um das Minimum · Lernratenabfall nach Robbins und Monro · Minibatch-Varianz · Überschussverlust · Hardwaredurchsatz**

<!--en:f7d80d4e-->
::: {.slide title="Warum der Gradientenabstieg auf dem gesamten Trainingssatz zu aufwendig ist"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ecfa25ed-->
- **Rechenkosten auf dem gesamten Trainingssatz:** Die Auswertung von $\nabla f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x})$ auf ImageNet ($n = 1.28\text{M}$) für $450{,}000$ Updates erfordert $450{,}000$ vollständige Durchläufe durch den Datensatz ($5{,}000\times$ so viel Rechenaufwand wie $90$ Epochen mit $b = 256$).

<!--en:62baf0b6-->
- **Effizienz von Minibatches ($b = 256$):** Die Schätzung der Abstiegsrichtung kostet pro Schritt nur $1/5{,}000$ der Rechenkosten eines Schritts auf dem gesamten Trainingssatz.

<!--en:bb8d9199-->
- **Der grundlegende Zielkonflikt:** Das Ziehen von $b \ll n$ Beispielen erzeugt **Gradientenrauschen**. Bei konstanter Lernrate schwanken die Iterierten dadurch in einer stationären Kugel um das Minimum.

<!--en:4d9a9224-->
::: {.slide title="Erwartungstreue Gradientenschätzung und Zerlegung in Signal und Rauschen"}

<!--en:46ec1b9c-->
[Stochastische Gradienten]{.kicker}

<!--en:b20edfdf-->
- **Erwartungstreuer Schätzer:** Für $i \sim \text{Uniform}\{1, \ldots, n\}$ gilt $\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$.

<!--en:4961d2a9-->
- **Zerlegung in Signal und Rauschen:**
  $$\nabla f_i(\mathbf{x}_t) = \nabla f(\mathbf{x}_t) + \boldsymbol{\xi}_t, \qquad \mathbb{E}[\boldsymbol{\xi}_t \mid \mathbf{x}_t] = \mathbf{0}$$

<!--en:36725e0a-->
- **Bleibendes Rauschen bei $\mathbf{x}^\star$:** Selbst wenn $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ gilt, sind einzelne Gradienten $\nabla f_i(\mathbf{x}^\star) \neq \mathbf{0}$, sofern die Beispiele unterschiedliche Gradienten haben. Bei konstanter Lernrate entfernen sich die Iterierten daher immer wieder von $\mathbf{x}^\star$.

<!--en:9e8359a3-->
::: {.slide title="SGD im Tal: vom anfänglichen Abstieg zur Punktwolke um das Minimum"}

<!--en:b6f5ea38-->
[Simulation im Tal]{.kicker}

<!--en:d2043201-->
- Unabhängiges gaußsches Rauschen $\xi_1, \xi_2 \sim \mathcal{N}(0, 1)$ mit Varianz eins überlagert den Gradienten im Tal $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$ ($\lambda_1 = 0.2, \lambda_2 = 4.0$). Bei $\eta = 0.1$ zeigen sich über $300$ Schritte zwei Phasen:
  1. **Anfangsphase (Schritte 1–100):** Der deterministische Gradient mit Norm $\|\nabla f(\mathbf{x})\|$ dominiert das Rauschen $\boldsymbol{\xi}$ und bewegt die Iterierte von $(-5, -2)$ in Richtung $(0, 0)$.
  2. **Stationäre Phase (ab Schritt 100):** Nahe dem Ursprung sinkt $\|\nabla f(\mathbf{x})\|$ auf den Rauschpegel. Die Iterierte bleibt in einer länglichen Wolke entlang der flachen Achse $x_1$.

<!--en:267e1984-->
::: {.slide title="Herleitung der stationären Varianz in der Kugel um das Minimum"}

<!--en:d187213b-->
[Rekursion für das zweite Moment]{.kicker}

<!--en:8dfc6501-->
- **Entwicklung mit Rauschen in einer Dimension ($x_{t+1} = (1 - \eta\lambda)x_t - \eta\xi_t$):** Quadrieren und Erwartungswertbildung ($\mathbb{E}[x_t\xi_t] = 0$) trennen die deterministische Kontraktion vom hinzukommenden Rauschen:
  $$\mathbb{E}\big[x_{t+1}^2\big] = (1 - \eta\lambda)^2\,\mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2$$

<!--en:d0335cd1-->
- **Stationäres Gleichgewicht ($0 < \eta < 2/\lambda$):** Aus $\mathbb{E}[x_{t+1}^2] = \mathbb{E}[x_t^2] = \mathbb{E}[x_\infty^2]$ folgt die Varianz in der **Kugel um das Minimum**:
  $$\mathbb{E}\big[x_\infty^2\big] = \frac{\eta\,\sigma^2}{\lambda\,(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{2\lambda}$$

<!--en:0f26254b-->
::: {.slide title="Drei Gesetzmäßigkeiten für die stationäre Kugel um das Minimum"}

<!--en:f3086122-->
[Geometrische Deutung]{.kicker}

<!--en:ebec19d4-->
- **1. Die Stabilitätsgrenze bleibt erhalten:** Es gilt $(2 - \eta\lambda) \to 0$ für $\eta \to 2/\lambda$. An der deterministischen Grenze divergiert die Varianz (im Tal muss $\eta < 0.5$ gelten).

<!--en:df1a209b-->
- **2. Inverse Skalierung mit der Krümmung ($\propto 1/\lambda$):** In flachen Richtungen klingt der Fehler anfangs langsamer ab ($\eta\lambda$), und die stationären Schwankungen sind stärker ($16\times$ so große Varianz für $\lambda_1 = 0.2$ wie für $\lambda_2 = 4.0$).

<!--en:48aa7909-->
- **3. Lineare Skalierung mit $\eta/b$:** Für $\eta\lambda \ll 1$ skaliert die stationäre Varianz gemäß $\frac{\eta\sigma^2}{2\lambda b}$. Eine Halbierung von $\eta$ oder eine Verdopplung von $b$ halbiert die Varianz in der Kugel um das Minimum.

<!--en:bfa50525-->
::: {.slide title="10 000 Simulationsläufe bestätigen die Formel für die Kugel um das Minimum"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:2c6eef42-->
- Die Simulation umfasst $10{,}000$ unabhängige SGD-Trajektorien im Tal ($\lambda_1 = 0.2, \lambda_2 = 4.0, \sigma^2 = 1$) über $2{,}000$ Schritte bei $\eta = 0.1$ und $\eta = 0.05$:
  - Bei $\eta = 0.1$ weichen die gemessenen Varianzen von $\frac{\eta}{\lambda(2-\eta\lambda)} = (0.2525, 0.0156)$ um höchstens $1\%$ ab.
  - Bei $\eta = 0.05$ halbiert sich die Varianz in der flachen Richtung nahezu auf $0.1256$. In der steilen Richtung sinkt sie um etwas mehr als die Hälfte ($0.0156 \to 0.0069$), weil $2 - \eta\lambda_2$ zunimmt ($1.6 \to 1.8$).

<!--en:b85b14de-->
::: {.slide title="Wie die rauschbedingte Untergrenze verschwindet: die Robbins-Monro-Bedingungen"}

<!--en:1ce1ae4f-->
[Abnehmende Schrittweiten]{.kicker}

<!--en:ca14110b-->
- **Robbins-Monro-Bedingungen (1951)** für exakte Konvergenz $\mathbf{x}_t \to \mathbf{x}^\star$ unter Rauschen:
  $$\sum_{t=1}^{\infty} \eta_t = \infty \qquad \text{and} \qquad \sum_{t=1}^{\infty} \eta_t^2 < \infty$$

<!--en:95f182c6-->
- **Unendliches Schrittweitenbudget ($\sum \eta_t = \infty$):** Damit ist $\mathbf{x}^\star$ von jeder Initialisierung aus erreichbar. Der geometrische Abfall $\eta_0\gamma^t$ verletzt diese Bedingung, und der Abstieg kommt vorzeitig zum Stillstand.

<!--en:fc1a40a4-->
- **Endliche kumulierte Rauschvarianz ($\sum \eta_t^2 < \infty$):** Die Varianz der Iterierten geht langfristig gegen null. Dies gelingt mit $\eta_t = \frac{c}{t+t_0}$, wobei der mittlere quadratische Fehler mit Rate $O(1/t)$ abnimmt.

<!--en:f20670c4-->
::: {.slide title="Mittelung im Minibatch senkt die Varianz proportional zu 1/b"}

<!--en:9c96161c-->
[Minibatch-SGD]{.kicker}

<!--en:198e994a-->
Minibatch-SGD mittelt die Gradienten der $b$ Beispiele im Minibatch $\mathcal{B}_t$, gezogen in Schritt $t$:

<!--en:147ba75f-->
- **Kovarianz der Gradienten einzelner Beispiele:** Sei $\boldsymbol{\Sigma}(\mathbf{x}) = \mathbb{E}_i\big[(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))^\top\big]$ die Kovarianzmatrix. Die Summe der Varianzen ist $\operatorname{tr}\boldsymbol{\Sigma} = \mathbb{E}_i\|\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2$.

<!--en:9688de18-->
- **Die Varianz skaliert mit $1/b$:** Bei $b$ unabhängigen Ziehungen mit Zurücklegen teilt die Mittelung die Kovarianz durch $b$:
  $$\mathbb{E}\big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}_t)}{b}$$

<!--en:d8e155e8-->
- **Invarianz bei festem $\eta / b$:** Die Substitution $\sigma^2 \leftarrow \sigma^2 / b$ in der Formel für die Kugel um das Minimum ergibt $\mathbb{E}[x_\infty^2] \approx \frac{\eta}{b}\cdot\frac{\sigma^2}{2\lambda}$ für $\eta\lambda \ll 1$. **Eine Verdopplung von $b$ hat in dieser Näherung genau dieselbe Wirkung auf die stationäre Kugel wie eine Halbierung von $\eta$.**

<!--en:87dbd50d-->
::: {.slide title="Stationärer Überschussverlust: Warum sich die Krümmung herauskürzt"}

<!--en:423e5cf1-->
[Wo das Training zum Stillstand kommt]{.kicker}

<!--en:7f25f273-->
- **Koordinatenvarianz und Krümmung der Verlustfunktion:** Entlang des Eigenvektors $k$ (Krümmung $\lambda_k$, Rauschvarianz $\sigma_k^2$) beträgt die stationäre Parametervarianz $v_k \approx \frac{\eta\sigma_k^2}{2\lambda_k b}$.

<!--en:50224b03-->
- **Der Faktor $\lambda_k$ kürzt sich exakt heraus:** Durch Multiplikation mit $\frac{1}{2}\lambda_k$ im quadratischen Verlust entfällt $\lambda_k$ in jeder Eigenrichtung: $\frac{1}{2}\lambda_k v_k = \frac{\eta\sigma_k^2}{4b}$.

<!--en:015bf6a1-->
- **Gesamter Überschussverlust unabhängig von der Krümmung:** Die Summe über alle $d$ Eigenrichtungen ($\sum_k \sigma_k^2 = \operatorname{tr}\boldsymbol{\Sigma}$) ergibt:
  $$\mathbb{E}\big[f(\mathbf{x}_\infty)\big] - f(\mathbf{x}^\star) \approx \frac{\eta\,\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$$

<!--en:ffc1f205-->
::: {.slide title="Überschussverlust am Airfoil-Datensatz der NASA"}

<!--en:c4757c7b-->
[Exakte Größen aus den Daten]{.kicker}

<!--en:c6d27dc6-->
- Auf dem standardisierten NASA-Datensatz Airfoil Self-Noise ($n = 1{,}500$ Windkanalmessungen, $5$ Merkmale $+ 1$ Bias $= 6$ Gewichte) lassen sich bei linearer Regression mit quadratischem Verlust die Hesse-Matrix und der Minimierer $\mathbf{w}^\star$ exakt in geschlossener Form angeben:
  $$\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$$

<!--en:df5be1be-->
- Der größte Eigenwert ist $L = \lambda_{\max}(\mathbf{H}) \approx 2.10$. Die Auswertung aller $1{,}500$ Einzelgradienten $\mathbf{g}_i = (\mathbf{x}_i^\top\mathbf{w}^\star - y_i)\mathbf{x}_i$ bei $\mathbf{w}^\star$ ergibt $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) \approx 3.24$.

<!--en:4d263c37-->
::: {.slide title="Der gemessene Überschussverlust stimmt mit η tr Σ / (4b) überein"}

<!--en:1915288f-->
[Theorie und Experiment]{.kicker}

<!--en:42cb59b7-->
- Für die Batchgröße $b = 10$ und $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) = 3.24$ liefert unsere Formel folgende Vorhersagen:
  - Bei $\eta = 0.05$ liegt der Verlust um $\frac{0.05 \times 3.24}{4 \times 10} \approx 0.0041$ über $f(\mathbf{w}^\star) \approx 0.242$.
  - Bei $\eta = 0.025$ beträgt der Überschuss $\frac{0.025 \times 3.24}{4 \times 10} \approx 0.0020$.

<!--en:6ae7b80d-->
- Das Mittel über $400$ unabhängige Minibatch-SGD-Läufe in der dritten Epoche stimmt bis auf $2 \times 10^{-4}$ mit der Vorhersage überein.

<!--en:c57986b6-->
::: {.slide title="Das interpolierende und das rauschbegrenzte Regime"}

<!--en:e1123082-->
[Wann verschwindet die Kugel um das Minimum?]{.kicker}

<!--en:85d7b249-->
Die Formel für den Überschussverlust $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ zeigt zwei grundverschiedene Trainingsregime. Entscheidend ist, ob die Gradienten der einzelnen Beispiele bei $\mathbf{x}^\star$ übereinstimmen:

<!--en:bf8da6ba-->
| Trainingsregime | Gradienten einzelner Beispiele bei $\mathbf{x}^\star$ | Spur der Rauschkovarianz $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)$ | SGD mit konstanter Lernrate | Beispiel aus dem Deep Learning |

<!--en:6dc0a6f1-->
| **Unterparametrisierung oder verrauschte Labels** | Gradienten unterscheiden sich ($\nabla f_i(\mathbf{x}^\star) \neq \mathbf{0}$, nur im Mittel gilt $\frac{1}{n}\sum_i \nabla f_i(\mathbf{x}^\star) = \mathbf{0}$) | $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) > 0$ | Verharrt beim Überschussverlust $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$. **Lernratenabfall $\eta_t \to 0$ erforderlich** | Regression auf Airfoil-Daten, LLM-Pre-Training mit nur einem Durchlauf durch riesige Korpora |

<!--en:e7b9fa1b-->
| **Interpolierendes Regime** | Das Modell passt jedes Trainingsbeispiel exakt an ($\nabla f_i(\mathbf{x}^\star) = \mathbf{0}$ für alle $i$) | $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) = 0$ | Die Kugel um das Minimum schrumpft von selbst für $\mathbf{x}_t \to \mathbf{x}^\star$. SGD **konvergiert geometrisch**. | Überparametrisierte Klassifikatoren, deren Verlust nach vielen Trainingsepochen auf null sinkt |

<!--en:a7f174fd-->
::: {.slide title="Ziehen ohne Zurücklegen und zufälliges Mischen"}

<!--en:dfb30e84-->
[Datenlader in der Praxis]{.kicker}

<!--en:ac1fa555-->
Datenlader ziehen Minibatches in der Praxis ohne Zurücklegen. In jeder Epoche mischen sie die $n$ Beispiele zufällig und teilen sie in disjunkte Minibatches der Größe $b$ auf:

<!--en:e094aaa5-->
- **Varianzkorrektur für endliche Grundgesamtheiten:** Die $n$ Gradientenfehler der einzelnen Beispiele summieren sich zu null ($\sum_{i=1}^n (\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})) = \mathbf{0}$). Daher sind die Fehler verschiedener Beispiele im Minibatch leicht negativ korreliert ($\mathbb{E}_{i \neq j}[\mathbf{e}_i^\top \mathbf{e}_j] = -\frac{\operatorname{tr}\boldsymbol{\Sigma}}{n-1}$). Die Varianz erhält einen Korrekturfaktor:
  $$\mathbb{E}\big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}\cdot\frac{n - b}{n - 1}$$

<!--en:db5bdb52-->
- **Prüfung der Grenzfälle:** Für $b = 10, n = 1{,}500$ beträgt der Faktor $0.994 \approx 1$. Für $b = n$ verschwindet die Varianz ($0$), wie es beim Gradientenabstieg auf dem gesamten Trainingssatz sein muss.

<!--en:107558b7-->
- **Fehlerausgleich über eine Epoche:** Durch das zufällige Mischen summieren sich die Gradientenfehler über alle $n/b$ Schritte einer Epoche in erster Ordnung zu null. Bei kleinem $\eta$ verkleinert dies die stationäre Kugel um das Minimum weiter (Mishchenko et al., 2020).

<!--en:d3382b4a-->
::: {.slide title="Fixkosten der Hardware: 100 Beispiele kosten so viel wie 1 Beispiel"}

<!--en:c3911eda-->
[Durchsatzmessung]{.kicker}

<!--en:90a213a1-->
- Jedes Update des Optimierers verursacht unabhängig von $b$ feste Kosten für die Aufrufsteuerung in Python, den Autograd-Graphen, die Kernelstarts und das Lesen der Gewichte aus dem Speicher.

<!--en:af3a9895-->
- Auf dem Airfoil-Datensatz mit $1{,}500$ Beispielen bei festem $\eta / b = 0.005$ gilt:
  - Für $b = 1$ ($1{,}500$ Updates/Epoche) und $b = 100$ ($15$ Updates/Epoche) beträgt die Laufzeit jeweils etwa eine Millisekunde *pro Update*. Mit $b = 100$ läuft das Training **pro Epoche $50\text{--}100\times$ so schnell**.

<!--en:7298dff2-->
::: {.slide title="Dateneffizienz und Grenzen der linearen Skalierung η ∝ b"}

<!--en:f1f4a44f-->
[Fortschritt pro Beispiel]{.kicker}

<!--en:ac28137c-->
- **Warum sich die Kurven kleiner Batches decken:** Bei festem $\eta / b = 0.005$ und $\eta\lambda \ll 1$ verkleinern $b$ Schritte mit Lernrate $\eta/b$ den Fehler um den Faktor $(1 - \eta\lambda/b)^b \approx 1 - \eta\lambda$. Pro verarbeitetem Beispiel ergibt sich derselbe Fortschritt wie bei einem Minibatch-Schritt der Größe $\eta$.

<!--en:bc88d1be-->
- **Grenze der Skalierung nahe der Krümmungsschranke $2/L$:**
  - Für $b = 100$ ergibt $\eta = 0.5$ bereits $\eta L = 0.5 \times 2.10 = 1.05$. Dadurch wächst der Faktor $(2 - \eta L)^{-1}$ in der Varianzformel, und der Verlust in der letzten Epoche steigt leicht ($0.248$ gegenüber $0.246$).
  - Oberhalb von $b_{\max} = \frac{2/L}{0.005} \approx 190$ verletzt die linear skalierte Lernrate die Bedingung $\eta < 2/L \approx 0.95$, und das Verfahren divergiert.

<!--en:de4be7ab-->
::: {.slide title="Verständnisfragen: Kugel um das Minimum, Überschussverlust und Batchgröße"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:fdbfa154-->
Für jedes Minibatch-SGD-Szenario das Ergebnis berechnen:

<!--en:31577dd3-->
1. **Vervierfachung der Batchgröße:** Bei kleinen Schritten ($\eta\lambda \ll 1$) steigt die Batchgröße von $b = 16$ auf $b = 64$, während $\eta$ konstant bleibt. Um welchen Faktor ändert sich der stationäre Überschussverlust? Was geschieht, wenn zusätzlich $\eta$ um den Faktor $4\times$ steigt?

<!--en:f0898de7-->
2. **Anisotrope Kugel um das Minimum und isotroper Überschussverlust:** Im Tal ($\lambda_1 = 0.2, \lambda_2 = 4.0$) sei die Varianz des Gradientenrauschens in beiden Richtungen gleich ($\sigma_1^2 = \sigma_2^2 = 1$). Wie groß ist das Verhältnis der stationären Koordinatenvarianzen $\mathbb{E}[x_{\infty,1}^2] / \mathbb{E}[x_{\infty,2}^2]$ für kleine $\eta$? In welchem Verhältnis stehen ihre Beiträge zum Überschussverlust?

<!--en:9fba46c1-->
3. **Grenze der linearen Skalierung:** Eine Verlustlandschaft hat die maximale Krümmung $L = 5.0$. Die gewählte Lernrate beträgt $\eta = 0.02$ bei $b = 8$ ($\eta/b = 0.0025$). Bei welcher Batchgröße $b$ erreicht die lineare Skalierung $\eta = 0.0025\,b$ die deterministische Stabilitätsgrenze $2/L$?

<!--en:7715016c-->
::: {.slide title="Verständnisfragen: Kugel um das Minimum, Überschussverlust und Batchgröße (Antworten)"}

<!--en:929b7671-2-->
[Verständniskontrolle]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:3122394b-->
1. Bei festem $\eta$ sinkt der Überschussverlust $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}}{4b}$ **um den Faktor $4\times$**. Steigt auch $\eta$ um den Faktor $4\times$, bleibt $\eta / b$ konstant. Der Überschussverlust bleibt **unverändert**, erreicht seine Untergrenze aber mit $4\times$ weniger Updates.

<!--en:e619d524-->
2. Die Koordinatenvarianz skaliert mit $1/\lambda$. Daher gilt $\mathbb{E}[x_{\infty,1}^2]/\mathbb{E}[x_{\infty,2}^2] \approx 4.0 / 0.2 =$ **$20\times$ zugunsten der flachen Richtung** $x_1$. Im Überschussverlust $\frac{1}{2}\lambda_k \mathbb{E}[x_{\infty,k}^2] \approx \frac{\eta\sigma_k^2}{4b}$ kürzt sich $\lambda_k$ heraus. Beide Achsen tragen **gleich viel bei ($1:1$)**.

<!--en:4cecfc85-->
3. Die Stabilitätsgrenze liegt bei $2/L = 2/5.0 = 0.4$. Die lineare Skalierung erreicht sie bei **$b = 0.4 / 0.0025 = 160$ Beispielen**.

<!--en:faf2dff9-->
::: {.slide title="Zusammenfassung: Stochastischer Gradientenabstieg und Minibatch-Gradientenabstieg"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:65666ad7-->
| Größe oder Gesetz | Mathematische Formel | Skalierung | Praktische Bedeutung |

<!--en:c779ed4a-->
| **Varianz des Minibatch-Gradienten** | $\mathbb{E}\|\mathbf{g}_t - \nabla f\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}\frac{n-b}{n-1}$ | Varianz $\propto 1/b$, Standardabweichung $\propto 1/\sqrt{b}$ | Ein $100\times$ so großer Batch senkt die Standardabweichung des Gradientenfehlers um den Faktor $10\times$ |

<!--en:190a72fa-->
| **Stationäre Kugel um das Minimum** | $\mathbb{E}[x_\infty^2] = \frac{\eta\sigma^2}{b\lambda(2 - \eta\lambda)} \approx \frac{\eta\sigma^2}{2\lambda b}$ | Skaliert mit dem Verhältnis $\eta/b$ und umgekehrt proportional zur Krümmung $\lambda$ | In flachen Richtungen schwanken die Parameter am stärksten |

<!--en:4f616e47-->
| **Stationärer Überschussverlust** | $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ | Die Krümmung $\lambda_k$ kürzt sich heraus. Der Überschussverlust ist proportional zu $\eta/b$ und $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)$. | Verschwindet bei interpolierenden Modellen ($\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)=0$), andernfalls muss die Lernrate abnehmen |

<!--en:fb727913-->
| **Robbins-Monro-Bedingungen** | $\sum_t \eta_t = \infty,\; \sum_t \eta_t^2 < \infty$ | $\eta_t = c/(t+t_0)$ ergibt $\mathbb{E}\|\mathbf{x}_t - \mathbf{x}^\star\|^2 = O(1/t)$ | Unendliches Schrittweitenbudget und endliche kumulierte Rauschvarianz |

<!--en:1bb7e23e-->
| **Bereich linearer Skalierung** | $\eta(b) \propto b$ für $\eta L \ll 2$ | Bei festem $\eta/b$ gleicher Fortschritt pro Beispiel und gleiche rauschbedingte Untergrenze | Auf paralleler Hardware verteilen sich die Fixkosten pro Schritt bis zur Batchgröße $b_{\text{crit}}$ auf immer mehr Beispiele (§{sec_batch_size}) |
