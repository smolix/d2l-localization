<!--en:61d6eacc-->
# Dropout

<!--en:a0b2a3fa-->
Trainieren wir ein überparametrisiertes Netz über viele Epochen mit wenigen oder verrauschten Daten, lernt es schließlich Eigenheiten der Trainingsdaten und verfälschte Labels auswendig (:numref:`sec_generalization_deep`). Der Gewichtszerfall (engl. weight decay) bestraft große Gewichte (:numref:`sec_weight_decay`). Early Stopping beendet das Training, bevor das Netz in den späten Epochen mit dem Auswendiglernen beginnt (:numref:`subsec_early_stopping`). *Dropout* regularisiert das Netz, indem es beim Training multiplikatives Rauschen direkt in die versteckten Schichten einbringt :cite:`Hinton.Srivastava.Krizhevsky.ea.2012,Srivastava.Hinton.Krizhevsky.ea.2014`. Bei jedem Vorwärtsdurchlauf setzt es jede versteckte Einheit unabhängig von den anderen mit Wahrscheinlichkeit $p$ auf null. So muss jede Einheit Merkmale lernen, die auch ohne bestimmte Partnereinheiten zuverlässig funktionieren. Das wirkt einer gegenseitigen Spezialisierung (*Koadaptation*) entgegen. Beim Test ist Dropout ausgeschaltet und das vollständige Netz arbeitet deterministisch.

<!--en:5efc8186-->
In diesem Abschnitt formulieren wir die Maskierung und Skalierung beim invertierten Dropout. Wir implementieren das Verfahren für ein MLP mit zwei versteckten Schichten auf Fashion-MNIST, zunächst von Grund auf und dann in kompakter Form. In einem Experiment mit starkem Labelrauschen zeigen wir, wie Dropout das Auswendiglernen verlangsamt und wie viel Validierungsgenauigkeit dadurch erhalten bleibt.

<!--en:4d7e7e87-->
## Rauschen in versteckten Schichten

<!--en:0f0f89be-->
### Maskierung und Skalierung beim invertierten Dropout

<!--en:45a6bf3d-->
Fügen wir den Eingaben beim Training zufälliges Rauschen hinzu, begünstigen wir glatte Funktionen, die robust gegenüber Störungen sind :cite:`Bishop.1995`. Betrachten wir ein lineares Modell $\mathbf{w}^\top\mathbf{x} + b$ mit quadratischem Fehler. Addieren wir zu den Eingaben unabhängiges Rauschen mit Mittelwert null und Varianz $s^2$, so erhöht sich der erwartete Verlust um $s^2\|\mathbf{w}\|_2^2$. Das entspricht einem $\ell_2$-Gewichtszerfall. Dropout überträgt dieses Prinzip auf versteckte Schichten, indem es zufällige binäre Masken verwendet.

<!--en:975cc0ee-->
Für eine versteckte Aktivierung $h_j$ und die *Dropout-Wahrscheinlichkeit* $p \in [0, 1)$ ziehen wir in jedem Trainingsschritt unabhängig eine Bernoulli-Maske $\xi_j \in \{0, 1\}$ mit $P(\xi_j = 0) = p$ und $P(\xi_j = 1) = 1 - p$. Wir ersetzen $h_j$ durch

<!--en:c01ee068-->
Teilen wir die verbleibenden Aktivierungen beim Training durch $1 - p$ (*invertiertes Dropout*), bleibt der Erwartungswert unverändert:

<!--en:0d43bfcb-->
Da beim Training $\mathbb{E}[h_j'] = h_j$ gilt, **schalten wir Dropout bei der Inferenz aus ($h_j' = h_j$), ohne die Gewichte zu skalieren**. (Bei der linearen Regression mit Dropout auf den Eingaben folgt aus der Varianzformel $\mathrm{Var}[x_j'] = \frac{p}{1-p}x_j^2$, dass zum quadratischen Verlust ein adaptiver Ridge-Strafterm hinzukommt. Der erwartete quadratische Verlust beträgt dann $\mathbb{E}[(\mathbf{w}^\top\mathbf{x}' + b - y)^2] = (\mathbf{w}^\top\mathbf{x} + b - y)^2 + \frac{p}{1-p}\sum_j x_j^2 w_j^2$. In einem tiefen nichtlinearen Netz gilt $\mathbb{E}[\mathrm{ReLU}(z)] \neq \mathrm{ReLU}(\mathbb{E}[z])$. Das deterministische Netz liefert beim Test eine wirksame Näherung für das Ensemble der $2^h$ ausgedünnten Teilnetze. Siehe dazu die Übungen :exref:`ex_dropout-monte-carlo-dropout` und :exref:`ex_dropout-geometric-mean`.)

<!--en:50c31334-->
## Implementierung

<!--en:3a469f20-->
### Die Dropout-Schicht

<!--en:86fe8cc3-->
Unsere eigene Funktion `dropout_layer(X, dropout)` erzeugt einen Tensor aus gleichverteilten Zufallszahlen in $[0, 1)$. Sie behält die Eingabeeinträge, deren Zufallszahl $\geq$ `dropout` ist, und teilt sie durch `1.0 - dropout`. Für `dropout == 1` gibt sie Nullen zurück:

<!--en:fef4737c-->
Testen wir `dropout_layer` an einem $2 \times 8$-Tensor mit den Dropout-Wahrscheinlichkeiten $0$, $0.5$ und $1$. Außerdem wenden wir Dropout mit $p = 0.2$ auf einen Vektor aus einer Million Einsen an. Wir prüfen, ob sein empirischer Mittelwert und seine empirische Varianz mit den theoretischen Werten $1$ und $p/(1 - p) = 0.25$ übereinstimmen.

<!--en:d4963cb5-->
### Definition und Training des eigenen Modells

<!--en:2d29f689-->
Wir wenden Dropout direkt nach der ReLU-Aktivierung auf die Ausgaben jeder versteckten Schicht an (:numref:`fig_dropout2`). Es ist *nur* während des Trainings aktiv. Wir definieren nun unseren Klassifikator `DropoutMLPScratch` ($784 \to 256 \to 256 \to 10$) mit zwei versteckten Schichten, He-Initialisierung und getrennten Dropout-Wahrscheinlichkeiten `dropout_1` und `dropout_2`:

<!--en:b5317627-->
![Dropout schaltet in jedem Trainingsschritt eine zufällige Teilmenge der versteckten Einheiten (hier $h_2$ und $h_5$) sowie deren ein- und ausgehende Verbindungen aus.](../img/dropout2.svg)

<!--en:0cb00fbd-->
Die Schichten prüfen `autograd.is_training()`. Der Wert ist nur innerhalb von `autograd.record()` wahr.

<!--en:ae0b4b2d-->
Die Schichten prüfen `self.training`. Beim Aufruf von `model.eval()` setzt der Trainer diesen Wert auf `False`.

<!--en:04c950dd-->
Die Schichten prüfen `self.training`. Unsere Methode `Module.call` setzt diesen Wert anhand des Arguments `training`, das Keras übergibt.

<!--en:ff8410a7-->
Die Schichten prüfen `self.deterministic`. Für die Auswertung setzt der Trainer diesen Wert auf `True`.

<!--en:a63aea76-->
Wir erzeugen nun eine Instanz von `DropoutMLPScratch` mit `dropout_1 = 0.2` nach der ersten versteckten Schicht, `dropout_2 = 0.5` nach der zweiten und der Lernrate `lr = 0.1`. Die Hyperparameter fassen wir in `hparams` zusammen, um sie später wiederzuverwenden. Wir trainieren das Modell über $20$ Epochen auf Fashion-MNIST:

<!--en:c0d81a71-->
In den ersten Epochen liegt der Trainingsverlust *über* dem Validierungsverlust. Beim Berechnen des Trainingsverlusts sind die zufälligen Dropout-Masken aktiv. Für den Validierungsverlust verwenden wir das vollständige deterministische Netz.

<!--en:41f09a2d-->
### Kompakte Implementierung

<!--en:8ff0824f-->
Mit den `Dropout`-Schichten des Frameworks in `Sequential` lässt sich dieselbe Architektur mit zwei versteckten Schichten in wenigen Zeilen ausdrücken:

<!--en:47353cd0-->
## Wirkung von Dropout bei Labelrauschen

<!--en:f05f1e41-->
Auf dem vollständigen Fashion-MNIST-Datensatz mit $60{,}000$ Bildern kommt es bei einem unregularisierten MLP nach 20 Epochen kaum zur Überanpassung (engl. overfitting). Dropout kann hier also wenig bewirken. Um zu beobachten, wie Dropout ausgeprägtem Auswendiglernen entgegenwirkt, greifen wir auf den Benchmark mit $1{,}000$ Bildern und verrauschten Labels aus :numref:`subsec_early_stopping` zurück. Für $40\%$ der Trainingslabels ziehen wir mit `d2l.corrupt` zufällig neue Labels ($\sim 36\%$ falsche Labels, ausgewählt durch die boolesche Maske `wrong = labels != y`):

<!--en:f0109cff-->
Wir trainieren `DropoutMLP` für $200$ Epochen auf `(X, labels)`. In beiden versteckten Schichten verwenden wir zunächst die Dropout-Wahrscheinlichkeit `p = 0` (ohne Dropout) und zum Vergleich `p = 0.5`. Mit `d2l.Tracker` erfassen wir die Genauigkeit auf den falsch beschrifteten Trainingsbeispielen `X[wrong]` sowie die Validierungsgenauigkeit auf den zurückgehaltenen Daten `X_val`:

<!--en:ede111f2-->
Betrachten wir die beiden Kurvenverläufe:

<!--en:97b673e3-->
- **Ohne Dropout ($p = 0$):** Bis zur 100. Epoche hat das MLP $100\%$ der verfälschten Trainingslabels auswendig gelernt. Die Validierungsgenauigkeit fällt gegenüber ihrem frühen Höchstwert um $\sim 10$ Prozentpunkte.

<!--en:47caeb0e-->
- **Mit Dropout ($p = 0.5$):** Das MLP lernt falsche Labels langsamer auswendig. In Epoche 200 gibt es erst etwa $70$ bis $85\%$ davon wieder. In den letzten 10 Epochen liegt die Validierungsgenauigkeit **3 bis 6 Prozentpunkte höher** als beim unregularisierten Netz. Die Kurve für die falschen Labels steigt jedoch weiter. Dropout verzögert den Schaden, verhindert ihn aber nicht. Deshalb kombiniert man es mit Early Stopping.

<!--en:51962aae-->
In der Praxis bringt Dropout den größten Nutzen, wenn die Parameteranzahl die Zahl der Beispiele im Datensatz weit übersteigt und das Training viele Epochen dauert. Das war etwa bei Bildklassifikatoren der frühen 2010er-Jahre wie AlexNet in :numref:`sec_alexnet` der Fall. Beim Pre-Training von Basismodellen mit nur einem Datendurchlauf oder direkt vor Schichten zur Batchnormalisierung (:numref:`sec_batch_norm`) verwendet man meist weniger oder gar kein Dropout. Vor der Batchnormalisierung erhöht Dropout die Varianz beim Training um den Faktor $1/(1-p)$. Dadurch unterscheidet sich die Varianz zwischen Training und Test :cite:`Li.Chen.Hu.ea.2019`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:a75b41b0-->
Invertiertes Dropout setzt beim Training jede versteckte Aktivierung unabhängig mit Wahrscheinlichkeit $p$ auf null und skaliert die verbleibenden Aktivierungen mit dem Faktor $1/(1 - p)$. Damit bleibt $\mathbb{E}[h_j'] = h_j$ erhalten, während das Rauschen die Varianz $\frac{p}{1 - p}h_j^2$ einbringt. Beim Test arbeitet das vollständige Netz daher deterministisch, ohne Masken und ohne Skalierung der Gewichte. Bei kleinen oder verrauschten Datensätzen mit weitaus mehr Parametern als Beispielen wirkt Dropout einer störanfälligen Koadaptation der versteckten Einheiten entgegen. Es verlangsamt das Auswendiglernen verfälschter Labels und erhöht die Genauigkeit auf zurückgehaltenen Daten um mehrere Prozentpunkte. Da Dropout das Auswendiglernen nur verzögert, wirkt es am besten zusammen mit Early Stopping und Gewichtszerfall.

<!--en:2294243a-->
## Übungen

<!--en:75cbaf74-->
1. [code] **Dropout-Wahrscheinlichkeiten.** Das Modell aus diesem Abschnitt schaltet Einheiten nach der ersten versteckten Schicht mit Wahrscheinlichkeit 0.2 aus, nach der zweiten mit Wahrscheinlichkeit 0.5.
    1. Trainieren Sie das Modell zunächst mit vertauschten Wahrscheinlichkeiten. Setzen Sie dann beide auf 0.5 und in einem weiteren Lauf beide auf 0.8. Geben Sie für jeden Lauf die beste Validierungsgenauigkeit an. Welche Schicht verträgt die höhere Wahrscheinlichkeit?
    1. Der `Trainer` berechnet den dargestellten Trainingsverlust bei eingeschalteten Masken. Bestimmen Sie nach dem Training den Verlust auf den Trainingsbildern bei ausgeschalteten Masken. Vergleichen Sie die Differenz zwischen Trainingsverlust und Validierungsverlust für die Netze, die Sie mit und ohne Dropout trainiert haben.

<!--en:d9ad732f-->
1. **Dropout und He-Initialisierung.** Bei der He-Initialisierung aus :numref:`subsec_he_init` ist die Varianz der Gewichte so gewählt, dass der Mittelwert der quadrierten Präaktivierungen von Schicht zu Schicht gleich bleibt.
    1. Um welchen Faktor verändert Dropout mit Wahrscheinlichkeit $p$ beim Training den Mittelwert der quadrierten Präaktivierungen der nächsten Schicht? Geben Sie den Faktor für jede der beiden Dropout-Schichten des Modells aus diesem Abschnitt an.
    1. Betrachten Sie ein Netz mit $K - 1$ versteckten Schichten, auf die jeweils Dropout mit Wahrscheinlichkeit $p$ folgt. Wie wächst der Faktor mit $K$? Sollte die Initialisierung ihn ausgleichen, wenn man bedenkt, dass das Netz beim Test ohne Dropout arbeitet? Stellen Sie einen Bezug zur Abweichung zwischen Training und Test her, die Dropout vor einer Schicht zur Batchnormalisierung verursacht.

<!--en:6f5aa4d6-->
1. [code] **Monte-Carlo-Dropout.** Lassen Sie Dropout im trainierten Modell `DropoutMLPScratch` beim Test eingeschaltet. Berechnen Sie für jedes Validierungsbild $T = 20$ Vorwärtsdurchläufe und mitteln Sie die Ausgaben der Softmax-Funktion :cite:`Gal.Ghahramani.2016`. Vergleichen Sie die Genauigkeit und die Kalibrierung mit denen eines einzelnen Durchlaufs des vollständigen Netzes. Stellen Sie für die Kalibrierung den Mittelwert der jeweils größten Softmax-Wahrscheinlichkeiten der Genauigkeit gegenüber. Wie ändert sich die Genauigkeit, wenn $T$ von 1 auf 20 steigt?
   :exlabel:`ex_dropout-monte-carlo-dropout`

<!--en:db8d98d7-->
1. [code] **Dropout und Gewichtszerfall.** Gewichtszerfall und Dropout regularisieren das Modell aus diesem Abschnitt auf unterschiedliche Weise. Bewerten Sie jeden Lauf anhand der besten Validierungsgenauigkeit und der Differenz zwischen Trainingsverlust und Validierungsverlust. Bestimmen Sie beide Verluste bei ausgeschalteten Masken.
    1. Setzen Sie beide Dropout-Wahrscheinlichkeiten auf null. Verwenden Sie Gewichtszerfall wie in :numref:`sec_weight_decay` mit den Stärken $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}\}$. Vergleichen Sie diese Läufe mit den Läufen mit und ohne Dropout aus diesem Abschnitt.
    1. Behalten Sie die Dropout-Wahrscheinlichkeiten 0.2 und 0.5 bei und ergänzen Sie Gewichtszerfall mit denselben Stärken. Addieren sich die Wirkungen der beiden Regularisierungsverfahren, nimmt der zusätzliche Nutzen ab oder heben sich die Wirkungen gegenseitig auf?

<!--en:3505f5e8-->
1. **Dropout beim Test.** ● Setzen Sie `dropout_1` auf null, sodass nur die zweite versteckte Schicht maskiert wird. Dabei durchlaufe $\boldsymbol{\xi}$ alle Masken dieser Schicht. Es seien $\mathbf{o}(\boldsymbol{\xi})$ und $\hat{\mathbf{y}}(\boldsymbol{\xi})$ die Logits und die Ausgabe der Softmax-Funktion für die Maske $\boldsymbol{\xi}$, die mit Wahrscheinlichkeit $P(\boldsymbol{\xi})$ auftritt. Zeigen Sie, dass die Logits des vollständigen Netzes gleich $\sum_{\boldsymbol{\xi}} P(\boldsymbol{\xi})\, \mathbf{o}(\boldsymbol{\xi})$ sind. Zeigen Sie außerdem, dass die Ausgabe seiner Softmax-Funktion dem normierten gewichteten geometrischen Mittel entspricht:

    $$\tilde{y}_j = \frac{\prod_{\boldsymbol{\xi}} \hat{y}_j(\boldsymbol{\xi})^{P(\boldsymbol{\xi})}}{\sum_k \prod_{\boldsymbol{\xi}} \hat{y}_k(\boldsymbol{\xi})^{P(\boldsymbol{\xi})}}.$$

    Erklären Sie, warum beide Gleichungen im Allgemeinen ihre Gültigkeit verlieren, sobald `dropout_1` von null verschieden ist.
   :exlabel:`ex_dropout-geometric-mean`

<!--en:f1aa5423-->
[Diskussion](https://d2l.discourse.group/t/100)

<!--en:0f39d062-->
[Diskussion](https://d2l.discourse.group/t/101)

<!--en:02567678-->
[Diskussion](https://d2l.discourse.group/t/261)

<!--en:a723aec0-->
[Diskussion](https://d2l.discourse.group/t/17987)

<!--en:abfa3783-->
Dropout zur Regularisierung durch multiplikatives Rauschen in versteckten Schichten<br>**Koadaptation · invertiertes Dropout $1/(1-p)$ · Verhalten bei Training und Test · langsameres Auswendiglernen falscher Labels**

<!--en:c85aadba-->
::: {.slide title="Warum Rauschen in versteckte Einheiten einbringen?"}

<!--en:fe8bdd90-->
[Motivation und Koadaptation]{.kicker}

<!--en:17115cf4-->
- **Drei einander ergänzende Regularisierungsverfahren:** Gewichtszerfall bestraft große Gewichtsnormen $\|\mathbf{W}\|_{\textrm{F}}^2$. Early Stopping beendet das Training, bevor das Netz das Rauschen auswendig lernt. **Dropout** (Srivastava et al., 2014) setzt in jedem Trainingsschritt jede versteckte Einheit mit Wahrscheinlichkeit $p$ auf null.

<!--en:d367b0d1-->
- **Koadaptation versteckter Einheiten verhindern:** Sind Einheiten immer gemeinsam aktiv, können Gruppen von Neuronen gezielt einzelne Trainingsfehler korrigieren. Zufällige Maskierung zwingt jede Einheit, Merkmale zu extrahieren, die auch mit einer zufälligen Auswahl von Partnereinheiten nützlich sind.

<!--en:a48c14f4-->
::: {.slide title="Zum Einstieg: Eingaberauschen entspricht L2-Gewichtszerfall"}

<!--en:677b1197-->
[Rauschen als Regularisierung]{.kicker}

<!--en:a4dc89de-->
- **Äquivalenz nach Bishop (1995):** Bei Eingaberauschen $\boldsymbol{\epsilon}$ mit Mittelwert null und Varianz $s^2$ verschwindet der gemischte Term ($\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$). Übrig bleibt genau der $\ell_2$-Ridge-Strafterm $s^2\|\mathbf{w}\|_2^2$.

<!--en:9b65913a-->
- **Vom additiven zum multiplikativen Rauschen:** Dropout bringt Rauschen durch multiplikative binäre Masken in jede versteckte Schicht ein.

<!--en:1d5a6fda-->
::: {.slide title="Invertiertes Dropout: Maskierung und Skalierung mit 1/(1 − p)"}

<!--en:6f14e397-->
[Mathematische Formulierung]{.kicker}

<!--en:feb83ec5-->
- **Unveränderter Erwartungswert beim Training ($\mathbb{E}[h_j'] = h_j$):** Teilt man die verbleibenden Aktivierungen beim Training durch $1 - p$, gilt $\mathbb{E}[h_j'] = p \cdot 0 + (1 - p)\frac{h_j}{1 - p} = h_j$.

<!--en:c8f9ea6a-->
- **Kein Mehraufwand beim Test:** Bei der Inferenz bleibt Dropout ausgeschaltet ($h_j' = h_j$). Die Gewichte brauchen keine Skalierung.

<!--en:d6d78254-->
::: {.slide title="Erhöhte Varianz bei invertiertem Dropout"}

<!--en:e608a2ca-->
[Skalierung des zweiten Moments]{.kicker}

<!--en:79b09a6e-->
- **Stärke des multiplikativen Rauschens:** Bei $p = 0.5$ verdoppeln sich die verbleibenden Aktivierungen ($2\times$). Die Standardabweichung des Rauschens beträgt $100\%$ von $|h_j|$. Bei $p = 0.8$ steigt der Skalierungsfaktor auf $5\times$ ($\textrm{std} = 2|h_j|$).

<!--en:3a4d2ed7-->
- **Änderung des zweiten Moments:** Beim Training wächst $\mathbb{E}[(h_j')^2]$ um den Faktor $\frac{1}{1-p}$. Liegt Dropout direkt vor der Batchnormalisierung, weicht deren laufende Varianzschätzung von der Varianz bei der Inferenz ab.

<!--en:2ab8cadc-->
::: {.slide title="Adaptiver Ridge-Strafterm und Abweichung bei Nichtlinearität im Test"}

<!--en:9e4ccb43-->
[Exakt im linearen Fall, Näherung bei ReLU]{.kicker}

<!--en:ec10d4d7-->
Wie hängt das deterministische Netz beim Test ($h_j' = h_j$) mit den $2^h$ Teilnetzen zusammen, aus denen beim Training zufällig eines gezogen wird?

<!--en:2c5c5429-->
**1. Exakter adaptiver Ridge-Strafterm für eine lineare Schicht:**

<!--en:e6075348-->
- Der Strafterm gewichtet jedes Gewichtsquadrat $w_j^2$ proportional zur Energie $x_j^2$ des zugehörigen Merkmals (stärkere Regularisierung häufiger Merkmale).

<!--en:e19a6e3b-->
**2. Abweichung der Erwartungswerte bei Nichtlinearität ($\mathbb{E}[\mathrm{ReLU}(z)] \neq \mathrm{ReLU}(\mathbb{E}[z])$):**

<!--en:487e66ce-->
| $(h_1', h_2')$ bei $p = 0.5$ | $(0, 0)$ | $(2, 0)$ | $(0, 2)$ | $(2, 2)$ |

<!--en:f2de7e52-->
| Präaktivierung $z = h_1' - h_2'$ | $0$ | $+2$ | $-2$ | $0$ |

<!--en:75365b9c-->
- Der Mittelwert der ReLU-Ausgaben über die ausgedünnten Netze beträgt $\mathbb{E}[\mathrm{ReLU}(z)] = 0.5$. Das deterministische Netz liefert beim Test dagegen $\mathrm{ReLU}(\mathbb{E}[z]) = \mathrm{ReLU}(0) = 0$.

<!--en:552dc9c0-->
::: {.slide title="Invertiertes Dropout in drei Codezeilen"}

<!--en:e0151bb4-->
[Eigene Implementierung der Schicht]{.kicker}

<!--en:9cfc761a-->
Die Funktion `dropout_layer(X, dropout)` zieht gleichverteilte Zufallszahlen aus $[0, 1)$, behält Eingabeeinträge mit Zufallszahlen $\ge p$ und skaliert sie mit $1 / (1 - p)$:

<!--en:fedcb087-->
::: {.slide title="Position von Dropout in einem MLP mit mehreren Schichten"}

<!--en:beee0837-->
[Architektur und Wechsel des Betriebsmodus]{.kicker}

<!--en:75453b18-->
Dropout folgt unmittelbar auf die ReLU-Aktivierung jeder versteckten Schicht. Es ist **nur beim Training** aktiv und wird bei Validierung und Inferenz übersprungen:

<!--en:f920b335-->
- **MLP mit zwei versteckten Schichten für Fashion-MNIST:** $784 \to 256 \to 256 \to 10$ ($269{,}322$ Parameter, He-Initialisierung).

<!--en:4ee54bf3-->
- **Dropout je nach Schicht:** Schwächeres Dropout nahe der Eingabe (`dropout_1 = 0.2`) erhält das Signal der ursprünglichen Merkmale. In tieferen Schichten ist das Risiko der Koadaptation am höchsten, daher ist dort stärkeres Dropout sinnvoll (`dropout_2 = 0.5`).

<!--en:3d66c908-->
- **Automatische Maskierung im Rückwärtsdurchlauf:** Wegen $h_j' = \frac{\xi_j}{1 - p}h_j$ multipliziert autograd die Gradienten an ausgeschalteten Einheiten mit $0$ und an den verbleibenden mit $\frac{1}{1 - p}$.

<!--en:dc3fb122-->
::: {.slide title="Definition und Training von DropoutMLPScratch"}

<!--en:f6f46dbb-->
[Eigene Implementierung des Modells]{.kicker}

<!--en:9676c0e6-->
In `DropoutMLPScratch` hängt jeder Aufruf von `dropout_layer` davon ab, ob das Framework den Trainingsmodus meldet:

<!--en:b32bb124-->
::: {.slide title="Trainingskurve über 20 Epochen auf ganz Fashion-MNIST"}

<!--en:0b6411a5-->
[Benchmark auf dem vollständigen Datensatz]{.kicker}

<!--en:af67f020-->
Nach dem Training von `DropoutMLPScratch` über $20$ Epochen auf allen $60{,}000$ Bildern beträgt die Validierungsgenauigkeit $\sim 87\%$:

<!--en:0523388a-->
- **Warum der Trainingsverlust anfangs *über* dem Validierungsverlust liegt:** `train_loss` beruht auf zufällig ausgedünnten Netzen ($p_1 = 0.2, p_2 = 0.5$). Für `val_loss` dient das vollständige Netz ohne Maskierung und Rauschen als Näherung für das Ensemble.

<!--en:a6022661-->
- **Kaum Überanpassung auf $60{,}000$ Bildern mit korrekten Labels:** Über $20$ Epochen zeigt selbst ein unregularisiertes MLP auf dem vollständigen Datensatz kaum Überanpassung. Ein Test mit Labelrauschen zeigt die Wirkung von Dropout deutlicher.

<!--en:5262f50c-->
::: {.slide title="Kompakte Implementierung mit Dropout-Schichten des Frameworks"}

<!--en:05952cca-->
[High-Level-API]{.kicker}

<!--en:2d4e70db-->
In den `Sequential`-Containern des Frameworks steht je ein `Dropout(p)`-Modul hinter jeder `ReLU`. Beim Wechsel zwischen `train` und `eval` schaltet `Trainer` die Masken automatisch ein oder aus:

<!--en:0933aeea-->
::: {.slide title="Auswendiglernen bei 40% zufällig neu gezogenen Labels eindämmen"}

<!--en:9c425cb9-->
[Belastungstest mit Labelrauschen]{.kicker}

<!--en:b5b23e5d-->
Im Benchmark mit $1{,}000$ Bildern werden $40\%$ der Labels neu gezogen ($\sim 36\%$ falsche Labels). Das Training von `DropoutMLP` läuft über $200$ Epochen mit $p = 0$ bzw. $p = 0.5$:

<!--en:ec5ac706-->
- **Ohne Dropout ($p = 0$, blau):** Bis zur Epoche 100 hat das Netz $100\%$ der falschen Labels auswendig gelernt (`C0--`). Die Validierungsgenauigkeit (`C0-`) fällt dadurch um $\sim 10$ Prozentpunkte gegenüber ihrem frühen Höchstwert.

<!--en:fdefe227-->
- **Mit Dropout ($p = 0.5$, orange):** Das Netz lernt falsche Labels langsamer auswendig (`C1--`, $70$ bis $85\%$ in Epoche 200, weiter steigend). Die Validierungsgenauigkeit liegt am Ende **3 bis 6 Prozentpunkte höher** (`C1-`).

<!--en:0821d58e-->
::: {.slide title="Drei einander ergänzende Sichtweisen auf Dropout"}

<!--en:2d703e6e-->
[Theoretische Perspektiven]{.kicker}

<!--en:1ef0f1c3-->
- **Adaptive Regularisierung durch Rauschen:** Multiplikatives Rauschen erzeugt die Varianz $\mathrm{Var}[h_j'] = \frac{p}{1-p}h_j^2$. Für die Merkmalsgewichte ergibt sich ein Strafterm $\propto \frac{p}{1-p}x_j^2 w_j^2$ und damit eine stärkere Regularisierung bei energiereichen Merkmalen.

<!--en:6901205e-->
- **Schutz vor Koadaptation (Hinton et al.):** Zwei bestimmte Einheiten sind mit Wahrscheinlichkeit $(1 - p)^2$ gemeinsam aktiv ($25\%$ bei $p = 0.5$). Dadurch können sich mehrere Einheiten beim Lernen verrauschter Labels weniger stark aufeinander verlassen.

<!--en:0bb787ad-->
- **Implizites Ensemble von Teilnetzen:** $2^h$ ausgedünnte Teilnetze teilen sich dieselben Gewichte ($2^{512} \approx 10^{154}$ Masken im MLP mit 2 versteckten Schichten).

<!--en:1ed0b0a3-->
::: {.slide title="Wann ist Dropout sinnvoll und wann sollte es entfallen?"}

<!--en:57bd6a5d-->
[Empfehlungen für die Praxis]{.kicker}

<!--en:69ef322e-->
- **Vollständig verbundene Schichten bei kleinen oder verrauschten Datensätzen ($p \in [0.2, 0.5]$):** Ein hohes Verhältnis $p/n$ und viele Trainingsepochen begünstigen Koadaptation.

<!--en:e85e86fe-->
- **Transformer mit mehreren Epochen ($p \approx 0.1$) und LLMs mit einer Epoche ($p = 0.0$):** Beim Pre-Training mit einem Durchlauf über Billionen von Token liefert jeder Schritt neue Daten. Auswendiglernen durch wiederholte Durchläufe entfällt.

<!--en:f5039e95-->
- **Direkt vor der Batchnormalisierung auf Dropout verzichten ($p = 0.0$):** Beim Training steigt das zweite Moment um den Faktor $\frac{1}{1-p}$. Dadurch weicht die laufende Varianzschätzung der Batchnormalisierung von der Varianz beim Test ab (Li et al., 2019).

<!--en:b7ce1573-->
::: {.slide title="Verständnisfragen: invertiertes Dropout, Partnereinheiten und Varianz"}

<!--en:ffcb46a8-->
[Selbsttest]{.kicker}

<!--en:79ad73f0-->
Fragen zu den mathematischen Eigenschaften von Dropout und zum Verhalten des Netzes:

<!--en:847acf37-->
1. **Extremes Dropout ($p = 0.8$):** Eine versteckte Einheit hat die Aktivierung $h_j = 1.0$. Welchen Wert nimmt sie bei invertiertem Dropout mit $p = 0.8$ an, wenn sie aktiv bleibt? Wie groß sind $\mathbb{E}[h_j']$ und $\mathrm{Var}[h_j']$?

<!--en:31170022-->
2. **Wahrscheinlichkeit gemeinsamer Aktivität:** Wie viele der 256 Einheiten einer versteckten Schicht bleiben bei $p = 0.5$ im Mittel pro Beispiel aktiv? Bei welchem Anteil der Trainingsschritte sind drei bestimmte Partnereinheiten $(j_1, j_2, j_3)$ *alle* gleichzeitig aktiv?

<!--en:f60f0533-->
3. **Versehentlich aktives Dropout beim Test:** Woran ist zu erkennen, dass das Modell bei der Validierung versehentlich im Trainingsmodus geblieben ist?

<!--en:557b98b6-->
**Lösungen:**

<!--en:fac36778-->
1. **Bei $p = 0.8$:** Eine weiterhin aktive Einheit nimmt den Wert $1 / (1 - 0.8) = 5.0$ an (in $20\%$ der Fälle). Der Mittelwert beträgt $\mathbb{E}[h_j'] = 0.2 \times 5.0 = 1.0$ und die Varianz $\mathrm{Var}[h_j'] = \frac{0.8}{0.2}(1.0)^2 = 4.0$ (Standardabweichung $2.0 = 2|h_j|$).

<!--en:a2538629-->
2. **Gemeinsam aktive Partnereinheiten:** Im Mittel bleiben $256 \times 0.5 = 128$ Einheiten aktiv. Drei bestimmte Einheiten $(j_1, j_2, j_3)$ sind nur in $(1 - 0.5)^3 = 1/8 = 12.5\%$ der Schritte gemeinsam aktiv.

<!--en:de046b62-->
3. **Fehler beim Test:** Wiederholte Aufrufe liefern bei der Validierung unterschiedliche Vorhersagen. Ein einzelner Durchlauf liefert eine geringere Genauigkeit, es sei denn, man mittelt die Vorhersagen über $T \ge 20$ Monte-Carlo-Durchläufe.

<!--en:fdbcfd4c-->
::: {.slide title="Zusammenfassung: Regularisierung durch Dropout"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:f2929934-->
- **Erwartungstreue invertierte Maske:** Mit $h_j' = \frac{\xi_j h_j}{1 - p}$ gilt beim Training weiterhin $\mathbb{E}[h_j'] = h_j$. Beim Test kommt das Netz ganz ohne Skalierung aus ($h_j' = h_j$).

<!--en:dcaa35e8-->
- **Adaptive Regularisierung:** Multiplikatives Rauschen erzeugt die Varianz $\mathrm{Var}[h_j'] = \frac{p}{1-p}h_j^2$. Bei linearen Schichten entspricht das einem nach Merkmalsenergie gewichteten $\ell_2$-Strafterm.

<!--en:13511773-->
- **Wirkung im Experiment:** Werden $40\%$ der Labels in Fashion-MNIST ($n = 1{,}000$) neu gezogen, verlangsamt $p = 0.5$ das Auswendiglernen falscher Labels und erhöht die Validierungsgenauigkeit um $3$ bis $6$ Prozentpunkte.
