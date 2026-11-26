<!--en:7cc9ced7-->
# Bewertung von Klassifikatoren

<!--en:121a3d5c-->
Unsere Softmax-Regression für Fashion-MNIST klassifiziert rund $83\%$ der Validierungsbilder richtig. Ist ein Klassifikator mit höherer Genauigkeit (engl. accuracy) immer besser? Nur wenn jede Klasse gleich wichtig ist und jede Art von Fehler genau gleich viel kostet. In der Praxis gibt es drei häufige Gründe, warum diese Symmetrie verloren geht:

<!--en:7a43f517-->
1. **Klassenungleichgewicht** (engl. class imbalance): Oft ist die positive Klasse, die uns interessiert, selten. So haben etwa nur $1\%$ der untersuchten Personen eine bestimmte Krankheit, oder nur $0.1\%$ der Transaktionen sind betrügerisch. Ein nutzloser Klassifikator, der immer die häufigste Klasse vorhersagt, erreicht dann eine Genauigkeit von $99\%$ beziehungsweise $99.9\%$, ohne einen einzigen positiven Fall zu erkennen.

<!--en:d8351fee-->
2. **Asymmetrische Fehlerkosten:** Eine lebensbedrohliche Krankheit zu übersehen (ein falsch negatives Ergebnis) kostet weitaus mehr, als einen Folgetest zu veranlassen (bei einem falsch positiven Ergebnis).

<!--en:a8597bfb-->
3. **Entscheidungen anhand von Wahrscheinlichkeiten:** Nachgelagerte Entscheidungsregeln verwenden unmittelbar die Zahlenwerte der vorhergesagten Wahrscheinlichkeiten $\hat{\mathbf{y}}$. Auf diese Werte können wir uns nur verlassen, wenn der Klassifikator *kalibriert* ist.

<!--en:14a27c94-->
Für diese drei Fälle leiten wir in diesem Abschnitt Kennzahlen und Diagnoseverfahren her. Wir betrachten *Präzision* und *Trefferquote* (engl. recall), *kostensensitive Entscheidungsschwellen*, *Zuverlässigkeitsdiagramme* (engl. reliability diagrams) und den *erwarteten Kalibrierungsfehler* (engl. expected calibration error, ECE) sowie die nachträgliche *Temperaturskalierung* (engl. temperature scaling).

<!--en:a1da9ee3-->
## Grenzen der Genauigkeit

<!--en:417aa5c5-->
### Klassenungleichgewicht

<!--en:2bd57e26-->
Angenommen, wir untersuchen $100{,}000$ Menschen auf eine Krankheit mit einer Prävalenz von $1\%$ ($1{,}000$ Erkrankte und $99{,}000$ Gesunde). Eine einfache Regel ignoriert alle Testergebnisse und sagt immer *gesund* vorher. Damit erreicht sie eine Genauigkeit von $99\%$, ohne einen einzigen Krankheitsfall zu erkennen. Vergleichen wir sie mit einem tatsächlichen Screeningtest, der $90\%$ der Erkrankten erkennt und bei $5\%$ der Gesunden fälschlich positiv ausfällt:

<!--en:a1277acc-->
| | krank (positiv) | gesund (negativ) |

<!--en:604bf2a8-->
| **als krank eingestuft** | $900$ (TP) | $4{,}950$ (FP) |

<!--en:1ec73e35-->
| **als gesund eingestuft** | $100$ (FN) | $94{,}050$ (TN) |

<!--en:5177f649-->
In dieser binären $2 \times 2$-Konfusionsmatrix stehen die vier Zellen für *richtig positive* ($\textrm{TP} = 900$), *falsch positive* ($\textrm{FP} = 4{,}950$), *falsch negative* ($\textrm{FN} = 100$) und *richtig negative* Ergebnisse ($\textrm{TN} = 94{,}050$). Aus der Summe der Diagonaleinträge ergibt sich eine Genauigkeit von rund $94{,}950 / 100{,}000 = 95\%$. Nach der Genauigkeit allein schneidet die nutzlose Regel „immer gesund“ ($99\%$) also besser ab als ein lebensrettender Test ($95\%$).

<!--en:e532ec36-->
Um die Güte gezielt für die positive Klasse zu beurteilen, betrachten wir zwei Anteile. Die *Trefferquote* (auch *Sensitivität* oder *Richtig-Positiv-Rate*) gibt an, welchen Anteil der tatsächlich positiven Fälle der Test erkennt. Die *Präzision* (auch *positiver Vorhersagewert*) gibt an, welcher Anteil der positiv getesteten Personen tatsächlich erkrankt ist:

<!--en:7295cc2f-->
Aus der ersten Spalte ergibt sich $\textrm{recall} = 900 / 1{,}000 = 0.90$ (gegenüber $0$ bei „immer gesund“). Aus der ersten Zeile ergibt sich $\textrm{precision} = 900 / 5{,}850 \approx 0.15$. Nur $15\%$ der positiv getesteten Personen sind also tatsächlich krank. Bei der großen Mehrheit der Gesunden reichen nämlich schon $5\%$ falsch positive Ergebnisse aus, damit die Fehlalarme ($4{,}950$) zahlreicher sind als die $900$ richtig positiven Ergebnisse. Diesen Effekt kennen wir vom Satz von Bayes aus :numref:`subsec_probability_hiv_app`. Für ein zweistufiges klinisches Screening sind $90\%$ Trefferquote bei $15\%$ Präzision ideal. Der günstige erste Test erkennt fast alle Erkrankten. Nur die $5{,}850$ positiv getesteten Personen benötigen einen Folgetest zur endgültigen Abklärung.

<!--en:6e6fd21c-->
Halten wir die für unseren Test charakteristische Trefferquote ($0.90$) und die Falsch-Positiv-Rate ($0.05$) fest und variieren wir die Prävalenz der Krankheit über die Werte $\{0.1\%, 1\%, 10\%, 50\%\}$:

<!--en:d197867b-->
Was ändert sich, was bleibt gleich? Trefferquote und Falsch-Positiv-Rate hängen *allein vom Klassifikator* ab. Die Präzision hängt dagegen stark von der *Prävalenz in der Bevölkerung* ab.

<!--en:95def482-->
### Asymmetrische Kosten

<!--en:ba3b1ab3-->
Angenommen, eine Erkrankung zu übersehen kostet $\kappa_{\textrm{FN}} = 100$, ein Fehlalarm kostet $\kappa_{\textrm{FP}} = 1$ und eine richtige Entscheidung kostet $0$. Gewichten wir die Fehler mit diesen Kosten, so betragen die erwarteten Kosten der Regel „immer gesund“ $100 \times 1{,}000 / 100{,}000 = 1.00$ pro Person. Für unseren Screeningtest ergeben sich $(100 \times 100 + 1 \times 4{,}950) / 100{,}000 \approx 0.15$ pro Person, also nur etwa ein Siebtel.

<!--en:fb21cc52-->
Schätzt ein Klassifikator für eine Person die Wahrscheinlichkeit $p = P(\textrm{sick} \mid \mathbf{x})$, so ergeben sich allgemein für die Vorhersage *gesund* erwartete Kosten von $p\,\kappa_{\textrm{FN}}$ und für die Vorhersage *krank* erwartete Kosten von $(1 - p)\,\kappa_{\textrm{FP}}$. Die Vorhersage *krank* minimiert die erwarteten Kosten, wenn folgende Bedingung gilt:

<!--en:4be2540d-->
Bei gleichen Kosten ($\kappa_{\textrm{FN}} = \kappa_{\textrm{FP}}$) ergibt sich die bekannte Schwelle $1/2$ (Klassifikation per Argmax). Bei $\kappa_{\textrm{FN}} = 100$ und $\kappa_{\textrm{FP}} = 1$ sinkt die optimale Schwelle hingegen auf $1/101 \approx 0.01$. Schon bei einem vorhergesagten Risiko von $2\%$ lohnt es sich also, den Fall abzuklären. Eine niedrigere Schwelle erhöht die Trefferquote auf Kosten der Präzision. Sind die Kosten nicht vorgegeben, fasst man diesen Zielkonflikt in der Praxis häufig durch das harmonische Mittel aus Präzision und Trefferquote zusammen, den *F1-Wert* $= 2\,\textrm{precision}\cdot\textrm{recall}/(\textrm{precision} + \textrm{recall})$. Alternativ betrachtet man mit ROC-Kurven und Präzisions-Trefferquoten-Kurven alle Schwellen (siehe :exref:`ex_classification-roc-curves` und :exref:`ex_classification-auc-ranking-probability`).

<!--en:fc7e81a1-->
## Kalibrierung

<!--en:0901b61b-->
### Zuverlässigkeitsdiagramme

<!--en:9a6aa76c-->
Kostensensitive Entscheidungsregeln wie :eqref:`eq_cost-threshold` verwenden unmittelbar den Zahlenwert von $p$. Sie setzen daher voraus, dass der Klassifikator *kalibriert* ist. Von allen Vorhersagen, denen er eine Konfidenz (engl. confidence) von $\hat{p} \approx 0.80$ zuweist, sollten etwa $80\%$ tatsächlich richtig sein.

<!--en:7637c572-->
Bei einer Aufgabe mit $q$ Klassen prüfen wir die Kalibrierung der jeweils wahrscheinlichsten Klasse. Dazu erfassen wir für jede Vorhersage auf den Validierungsdaten die *Konfidenz* $\hat{p}^{(i)} = \max_j \hat{y}^{(i)}_j$ und mit dem Indikator $c^{(i)} = \mathbf{1}(\operatorname{argmax}_j \hat{y}^{(i)}_j = y^{(i)})$, ob die Vorhersage richtig ist. Wir gruppieren die Beispiele nach ihrer Konfidenz in die Intervalle $[0, 0.1), [0.1, 0.2), \ldots, [0.9, 1.0]$. Für jedes Intervall vergleichen wir die mittlere Konfidenz $\bar{p}_b$ mit der empirischen Genauigkeit $a_b$. Tragen wir $a_b$ gegen $\bar{p}_b$ auf, erhalten wir ein *Zuverlässigkeitsdiagramm*. Bei einem perfekt kalibrierten Modell liegen die Punkte auf der Diagonalen $a_b = \bar{p}_b$. Der *erwartete Kalibrierungsfehler* (ECE) ist der Mittelwert der absoluten Differenzen, gewichtet nach der Anzahl der Beispiele je Intervall:

<!--en:f2496eb4-->
Trainieren wir unser kompaktes Modell `SoftmaxRegression` auf Fashion-MNIST und sammeln wir für die Validierungsdaten die Logits `O` und die Labels `y`. Damit berechnen wir die Kennzahlen für jedes Intervall sowie den ECE und erstellen das Zuverlässigkeitsdiagramm:

<!--en:360c46ef-->
Um die Konfidenz jeder Vorhersage zu berechnen, wenden wir in NumPy die numerisch stabile Softmax-Funktion mit verschobenen Logits an (:numref:`sec_softmax_scratch`). In jeder Zeile wählen wir die größte Wahrscheinlichkeit.

<!--en:0fa013f2-->
Anschließend gruppiert `reliability(conf, correct, num_bins=10)` die Vorhersagen für die Validierungsdaten mit `np.bincount` nach ihrer Konfidenz in zehn gleich breite Intervalle. Die Funktion liefert für jedes Intervall die Anzahl der Beispiele `n`, die mittlere Konfidenz `mean_conf` und die empirische Genauigkeit `acc` sowie den nach der Anzahl der Beispiele gewichteten `ece`:

<!--en:8dd1aa37-->
Wir berücksichtigen nur Intervalle mit mindestens $100$ Vorhersagen (`used = n >= 100`). So verzerren zufällige Schwankungen der Anteile bei kleinen Stichproben die Kurve nicht. Wir tragen `acc[used]` gegen `mean_conf[used]` auf und zeichnen zum Vergleich die ideale Diagonale $y = x$ ein:

<!--en:568a4892-->
In allen ausreichend besetzten Intervallen ($n_b \geq 100$) weicht die empirische Genauigkeit um höchstens acht Prozentpunkte von der mittleren Konfidenz ab. Die Kurve liegt leicht *oberhalb* der Diagonalen (mittlere Konfidenz $\sim 81\%$ gegenüber einer Genauigkeit von $\sim 83\%$, bei einem ECE unter $0.03$). Tiefe überparametrisierte Netze geben häufig viel zu hohe Konfidenzen aus. Die flache Softmax-Regression auf Fashion-MNIST ist dagegen bereits gut kalibriert, ihre Konfidenzen sind nur etwas zu niedrig.

<!--en:3b6a1d9b-->
### Temperaturskalierung

<!--en:5d23e99b-->
Zeigt ein Zuverlässigkeitsdiagramm eine Fehlkalibrierung, lässt sie sich nachträglich mit einem einzigen Parameter korrigieren. Dazu verwenden wir die *Temperaturskalierung* :cite:`Guo.Pleiss.Sun.Weinberger.2017`. Wir halten die trainierten Gewichte fest und teilen die Logits durch eine skalare Softmax-Temperatur $T > 0$ (:numref:`sec_softmax`). Dann passen wir $T$ so an, dass die Kreuzentropie auf einem zurückgehaltenen Teil der Validierungsdaten minimal wird. Da die Division durch $T > 0$ die Reihenfolge der Logits erhält, ändern sich weder die vorhergesagten Klassen noch die Genauigkeit. Nur die Konfidenz ändert sich. Mit $T > 1$ dämpfen wir zu hohe Konfidenzen, mit $T < 1$ heben wir zu niedrige an. Passen wir $T$ auf der ersten Hälfte unseres Validierungsdatensatzes für Fashion-MNIST an und prüfen wir den ECE auf der zweiten Hälfte:

<!--en:1fafa900-->
Wie die leicht zu niedrigen Konfidenzen im Zuverlässigkeitsdiagramm erwarten lassen, liegt die angepasste Softmax-Temperatur knapp unter $1$ ($T \approx 0.9$). Der ohnehin sehr kleine ECE ändert sich dadurch kaum.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ac2f57c3-->
Die Genauigkeit allein ist nur dann ein geeignetes Gütemaß, wenn die Klassen gleich häufig sind und alle Fehler gleich viel kosten. Sind positive Beispiele selten, dominieren die richtig negativen Ergebnisse die Gesamtgenauigkeit. Für die Erkennung positiver Fälle betrachten wir deshalb zwei Größen: die *Trefferquote* ($\textrm{TP}/(\textrm{TP}+\textrm{FN})$), die allein vom Klassifikator abhängt, und die *Präzision* ($\textrm{TP}/(\textrm{TP}+\textrm{FP})$), die nach dem Satz von Bayes von der Prävalenz in der Bevölkerung abhängt. Haben falsch negative und falsch positive Ergebnisse unterschiedliche Kosten $\kappa_{\textrm{FN}}$ und $\kappa_{\textrm{FP}}$, so minimieren wir die erwarteten Kosten, indem wir bei $p > \kappa_{\textrm{FP}}/(\kappa_{\textrm{FP}} + \kappa_{\textrm{FN}})$ die positive Klasse vorhersagen. Diese Regel setzt voraus, dass die vorhergesagten Wahrscheinlichkeiten kalibriert sind. Mit Zuverlässigkeitsdiagrammen und dem erwarteten Kalibrierungsfehler prüfen wir, wie gut diese Wahrscheinlichkeiten mit den beobachteten Häufigkeiten übereinstimmen. Passen wir nachträglich eine einzige Softmax-Temperatur $T > 0$ auf zurückgehaltenen Daten an, skalieren wir die Konfidenz neu, ohne die Genauigkeit zu ändern.

<!--en:2294243a-->
## Übungen

<!--en:54de36c7-->
1. **Optimale Entscheidungen und Enthaltung.** Bei einer Aufgabe mit mehreren Klassen bezeichne $\kappa_{ij}$ die Kosten für die Vorhersage der Klasse $i$, wenn die tatsächliche Klasse $j$ ist. Sei $p(j \mid x)$ die Wahrscheinlichkeit der Klasse $j$ bei gegebener Eingabe $x$.
    1. Drücken Sie die erwarteten Kosten der Vorhersage $i$ durch die Kosten der einzelnen Fälle und $p(j \mid x)$ aus. Geben Sie die Regel zur Auswahl des optimalen $i$ an. Prüfen Sie, ob sich für zwei Klassen :eqref:`eq_cost-threshold` ergibt.
    1. Eine richtige Vorhersage koste $0$, eine falsche $\lambda_s$. Der Klassifikator darf sich *der Entscheidung enthalten*. Dafür fallen feste Kosten von $\lambda_r < \lambda_s$ an. Leiten Sie die Schwelle für die Wahrscheinlichkeit der wahrscheinlichsten Klasse her, oberhalb derer eine Vorhersage günstiger ist als eine Enthaltung.
    1. Bei einer medizinischen Diagnose kostet eine Überweisung zur fachärztlichen Abklärung wenig im Vergleich zu einer Fehldiagnose. Nehmen Sie $\lambda_r/\lambda_s = 0.05$ an. Berechnen Sie die Schwelle und geben Sie an, wie der Klassifikator bei einer Vorhersage mit einer Konfidenz von $0.9$ entscheiden sollte. Beschreiben Sie anschließend einen Fall, in dem die vollständige Verteilung $p(y \mid x)$ für den behandelnden Arzt hilfreicher ist als eine einzelne Vorhersage oder eine Enthaltung.

    *Nach Kevin Murphy,
    [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html),
    Aufgabe 5.1.*

<!--en:172e14ac-->
1. **Klassengewichteter Verlust.** Im Screeningbeispiel aus :numref:`subsec_beyond-accuracy` kostet es wesentlich mehr, eine Erkrankung zu übersehen, als einen Fehlalarm auszulösen. Weisen Sie jeder Klasse $j$ ein Gewicht $w_j > 0$ zu und trainieren Sie $f$ mit dem klassengewichteten mittleren Verlust

   $$\frac{1}{n} \sum_{i=1}^n w_{y^{(i)}}\, l\big(f(\mathbf{x}^{(i)}), y^{(i)}\big),$$

   wobei $l$ der Kreuzentropieverlust und $y^{(i)}$ die Klasse des Beispiels $i$ ist. Jedes Beispiel der Klasse $j$ zählt also $w_j$-fach.
    1. Die Trainingsbeispiele seien gemäß der A-priori-Verteilung $p(y)$ der Klassen und den klassenbedingten Dichten $p(\mathbf{x} \mid y)$ gezogen. Zeigen Sie, dass der erwartete klassengewichtete Verlust proportional zum erwarteten ungewichteten Verlust unter folgender Verteilung ist: Die klassenbedingten Dichten bleiben gleich und die A-priori-Verteilung der Klassen ist proportional zu $w_y\, p(y)$. Wie können Sie denselben Effekt durch eine Änderung der Trainingsdaten erzielen, ohne den Verlust zu ändern?
    1. Betrachten Sie eine feste Eingabe $\mathbf{x}$ mit den Klassenwahrscheinlichkeiten $p(j \mid \mathbf{x})$. Das Modell könne jede beliebige Verteilung $\hat{\mathbf{y}}$ ausgeben. Zeigen Sie, dass der erwartete gewichtete Verlust
       $$-\sum_j w_j\, p(j \mid \mathbf{x}) \log \hat{y}_j$$
       für $\hat{y}_j \propto w_j\, p(j \mid \mathbf{x})$ minimal wird.
    1. Bestimmen Sie für die beiden Klassen „gesund“ und „krank“ die Schwelle für $p(\textrm{sick} \mid \mathbf{x})$, oberhalb derer $\arg\max$ für dieses $\hat{\mathbf{y}}$ die Klasse „krank“ liefert. Wählen Sie die Gewichte so, dass die Schwelle mit der aus :eqref:`eq_cost-threshold` übereinstimmt. Erklären Sie, warum die Wahl $w_\textrm{sick} > w_\textrm{healthy}$ die Trefferquote erhöht und wie sie sich auf die Anzahl der falsch positiven Ergebnisse auswirkt.

<!--en:eacb1094-->
1. **Kalibrierung und Nutzen.** Bei einer Aufgabe mit zehn Klassen gibt es in jeder Klasse gleich viele Beispiele.
    1. Ein Klassifikator ignoriert seine Eingabe und gibt für jede Klasse die Wahrscheinlichkeit $0.1$ aus. Bei Gleichstand entscheidet er zufällig. Zeigen Sie, dass er kalibriert ist, und berechnen Sie seine Genauigkeit.
    1. Ein anderer Klassifikator liegt bei 95% der Beispiele richtig und gibt stets die Konfidenz $1$ an. Zeichnen Sie sein Zuverlässigkeitsdiagramm und berechnen Sie seinen erwarteten Kalibrierungsfehler.
    1. Erklären Sie, warum Kalibrierung und Genauigkeit unterschiedliche Eigenschaften messen. Warum kann keine der beiden Größen die andere ersetzen, wenn die Entscheidung auf der Schwelle :eqref:`eq_cost-threshold` beruht?

<!--en:c44b4167-->
1. [code] **ROC-Kurven.** Ein binärer Klassifikator gibt einen Score $s \in (0, 1)$ für die positive Klasse aus. Als Schwelle eignet sich jedes $\tau \in [0, 1]$, nicht nur $\frac{1}{2}$. Der Klassifikator sagt die positive Klasse vorher, wenn $s > \tau$ gilt. Die *ROC-Kurve* (Receiver Operating Characteristic) stellt die Richtig-Positiv-Rate (Trefferquote) der Falsch-Positiv-Rate $\textrm{FP}/(\textrm{FP}+\textrm{TN})$ gegenüber, während $\tau$ von $1$ auf $0$ sinkt. Simulieren Sie $1000$ positive und $1000$ negative Beispiele mit Scores $s = \sigma(z)$. Dabei ist $\sigma(z) = 1/(1 + \exp(-z))$ die Sigmoidfunktion. Für positive Beispiele gilt $z \sim \mathcal{N}(\mu, 1)$ und für negative $z \sim \mathcal{N}(0, 1)$.
    1. Berechnen und zeichnen Sie die ROC-Kurve für $\mu \in \{0, 1, 2\}$. Geben Sie an, welche Punkte im Diagramm den beiden Grenzfällen $\tau = 1$ und $\tau = 0$ entsprechen.
    1. Zeigen Sie, dass sich im Erwartungswert die Diagonale ergibt, wenn die Scores des Klassifikators unabhängig von den Labels sind.
    1. Zeichnen Sie für $\mu = 2$ die *Präzisions-Trefferquoten-Kurve*. Variieren Sie dazu $\tau$ und tragen Sie die Präzision gegen die Trefferquote auf. Verwenden Sie einmal gleich viele positive wie negative Beispiele und einmal nur 1% positive Beispiele (behalten Sie 1000 negative Beispiele bei und ziehen Sie 10 positive oder passen Sie die Gewichte an). Erklären Sie, warum die ROC-Kurve bei dieser Änderung der Prävalenz gleich bleibt, während die Präzisions-Trefferquoten-Kurve sinkt. Welche der beiden Kurven beschreibt den Screeningtest aus diesem Abschnitt besser?
   :exlabel:`ex_classification-roc-curves`

<!--en:12f1181f-->
1. [code] **AUC als Wahrscheinlichkeit eines korrekten Rankings.** ● Verwenden Sie für diese Aufgabe die ROC-Kurven und die simulierten Scores aus :exref:`ex_classification-roc-curves`.
    1. Zeigen Sie, dass die Fläche unter der ROC-Kurve bei paarweise verschiedenen Scores folgender Wahrscheinlichkeit entspricht: Ein zufällig gewähltes positives Beispiel erhält einen höheren Score als ein zufällig gewähltes negatives Beispiel. Damit beschreibt die Fläche die Güte des Rankings unabhängig von der Wahl einer Schwelle.
    1. Berechnen Sie diese Wahrscheinlichkeit für die simulierten Scores in geschlossener Form. Vergleichen Sie sie mit der Fläche, die Sie für jede Kurve mit der Trapezregel erhalten.
    1. Bestimmen Sie die ROC-Kurve und die AUC der trainierten Softmax-Regression für die Validierungsbilder zweier Klassen $a$ und $b$. Verwenden Sie $a$ als positive Klasse und als Score $s = \sigma(o_a - o_b)$, also die Sigmoidfunktion der Differenz der beiden Logits. Betrachten Sie die Paare „Turnschuh“ und „Sandale“ sowie „Hemd“ und „T-Shirt“. Vergleichen Sie die Reihenfolge der beiden AUC-Werte mit den Einträgen der Konfusionsmatrix $\mathbf{C}$ aus :numref:`sec_softmax_scratch` für dieselben Paare. Erklären Sie, warum die beiden Verfahren zu unterschiedlichen Bewertungen kommen können.
   :exlabel:`ex_classification-auc-ranking-probability`

<!--en:a409f106-->
1. [code] **Temperaturskalierung.** Im Text passen wir $T$ auf der ersten Hälfte der Validierungsbilder an.
    1. Zeichnen Sie für die zweite Hälfte die Zuverlässigkeitsdiagramme vor und nach der Skalierung. Zeigen Sie, dass sich die Genauigkeit nicht ändert.
    1. Passen Sie $T$ nun auf den Trainingsbildern an und vergleichen Sie den Wert mit dem Wert für $T$ aus dem Text. Erklären Sie, warum Sie für die Anpassung Daten verwenden sollten, die das Modell beim Training nicht gesehen hat.
    1. Berechnen Sie den erwarteten Kalibrierungsfehler des unskalierten Modells mit 5, 10 und 20 Intervallen. Wie stark hängt der Wert von der Wahl der Intervalle ab?

<!--en:9e600c28-->
[Diskussionen](https://d2l.discourse.group/t/6808)

<!--en:a3a9a641-->
[Diskussionen](https://d2l.discourse.group/t/6809)

<!--en:b6d8a399-->
[Diskussionen](https://d2l.discourse.group/t/6810)

<!--en:9edc49e3-->
[Diskussionen](https://d2l.discourse.group/t/17981)

<!--en:e439b3a5-->
Bewertung von Klassifikatoren: Präzision, Trefferquote, Kosten und Kalibrierung<br>**Klassenungleichgewicht · Trefferquote und Präzision · Prävalenz im Satz von Bayes · kostensensitive Schwellen · Zuverlässigkeitsdiagramme · ECE · Temperaturskalierung**

<!--en:c21a7ab4-->
::: {.slide title="Wann bedeutet höhere Genauigkeit einen schlechteren Klassifikator?"}

<!--en:62a3c34a-->
[Grenzen der Genauigkeit als alleiniger Kennzahl]{.kicker}

<!--en:2dc8e637-->
Unser Fashion-MNIST-Klassifikator erreichte eine Genauigkeit von $\sim 83\%$ bei $10$ gleich häufigen Klassen und gleichen Kosten für jeden Fehler. Drei Bedingungen heben diese Symmetrie auf:

<!--en:7f2be955-->
- **Starkes Klassenungleichgewicht:** Bei einer Prävalenz von $1\%$ erreicht die Vorhersage *gesund* für alle eine Genauigkeit von $99\%$, erkennt aber **keinen einzigen** positiven Fall.

<!--en:b5c7af0a-->
- **Asymmetrische Fehlerkosten:** Eine Krankheit zu übersehen ($\mathrm{FN}$) kostet weitaus mehr als ein zusätzlicher Bestätigungstest ($\mathrm{FP}$).

<!--en:f5adff1c-->
- **Entscheidungen anhand von Wahrscheinlichkeiten:** Schwellenregeln verwenden $\hat{p} = P(y \mid \mathbf{x})$ direkt. Die vorhergesagte Konfidenz muss daher mit der empirischen Genauigkeit übereinstimmen (*Kalibrierung*).

<!--en:66b01190-->
::: {.slide title="Die binäre Konfusionsmatrix bei Klassenungleichgewicht"}

<!--en:4b57af78-->
[Vergleich bei 1% Prävalenz]{.kicker}

<!--en:38dd4b89-->
Screening von $100{,}000$ Menschen bei einer Prävalenz von $1\%$ ($1{,}000$ Erkrankte, $99{,}000$ Gesunde). Der Test erkennt $90\%$ der Erkrankten ($\mathrm{TP}=900, \mathrm{FN}=100$) und fällt bei $5\%$ der Gesunden falsch positiv aus ($\mathrm{FP}=4{,}950, \mathrm{TN}=94{,}050$):

<!--en:90f3dd51-->
- **Das Genauigkeitsparadox:** Aus der Summe der Diagonaleinträge ergibt sich eine Genauigkeit von $(900 + 94{,}050) / 100{,}000 = 94.95\%$. Damit liegt die nutzlose Regel *„immer gesund“* ($99.0\%$) vor einem Test, der $90\%$ der Erkrankten erkennt.

<!--en:2a0dba2f-->
- **Warum die Genauigkeit versagt:** Richtig negative Ergebnisse ($\mathrm{TN} = 94{,}050$) dominieren den Nenner. Die $1{,}000$ Erkrankten tragen höchstens $1\%$ bei.

<!--en:8780283e-->
::: {.slide title="Erkennung positiver Fälle anhand von Trefferquote und Präzision"}

<!--en:c641e01c-->
[Anteile in Spalte und Zeile]{.kicker}

<!--en:5c68dd5e-->
Zwei Anteile ergänzen einander bei der Bewertung der positiven Fälle. Die zahlreichen richtig negativen Ergebnisse ($\mathrm{TN}$) gehen in keinen der beiden Anteile ein:

<!--en:7da917ce-->
- **Trefferquote ($900/1{,}000 = 0.90$):** Anteil der Erkrankten, die der Test erkennt (gegenüber $0.00$ bei *„immer gesund“*).

<!--en:67d72881-->
- **Präzision ($900/5{,}850 \approx 0.154$):** Anteil der positiv getesteten Personen, die tatsächlich erkrankt sind. Nur $5.85\%$ der Bevölkerung benötigen einen Bestätigungstest.

<!--en:af5f1a62-->
::: {.slide title="Wie hängt die Präzision von der Prävalenz in der Bevölkerung ab?"}

<!--en:333579bb-->
[Der Satz von Bayes bei der Bewertung von Klassifikatoren]{.kicker}

<!--en:7f7cf0ff-->
Testspezifische Trefferquote ($0.90$) und Falsch-Positiv-Rate ($\mathrm{FPR} = 0.05$) festhalten, Prävalenz der Krankheit $\pi = P(\text{sick})$ variieren:

<!--en:59c56994-->
::: {.slide title="Der Satz von Bayes erklärt die Abhängigkeit der Präzision von der Prävalenz"}

<!--en:333579bb-2-->
[Der Satz von Bayes bei der Bewertung von Klassifikatoren]{.kicker}

<!--en:0450a6a6-->
- **Testspezifisch oder prävalenzabhängig:** Trefferquote und $\mathrm{FPR}$ hängen *allein vom Klassifikator* ab. Für die Präzision gilt
  $$P(\text{sick} \mid \text{flagged}) = \frac{\text{recall}\cdot\pi}{\text{recall}\cdot\pi + \mathrm{FPR}\cdot(1-\pi)}$$
  und damit hängt sie gemäß dem Satz von Bayes von der Prävalenz $\pi$ in der Bevölkerung ab.

<!--en:85831bb4-->
::: {.slide title="Asymmetrische Fehlerkosten und erwartetes Risiko"}

<!--en:cb88d5b6-->
[Kostensensitive Bewertung]{.kicker}

<!--en:3a866b7c-->
Haben falsch negative und falsch positive Ergebnisse unterschiedliche Kosten $\kappa_{\mathrm{FN}}$ und $\kappa_{\mathrm{FP}}$, zählen für die Bewertung der Klassifikatoren die **erwarteten Kosten pro Person**:

<!--en:35b887c3-->
- **Zahlenbeispiel ($\kappa_{\mathrm{FN}} = 100,\; \kappa_{\mathrm{FP}} = 1$):**
  - *„Immer gesund“* ($\mathrm{FN} = 1{,}000,\; \mathrm{FP} = 0$): Kosten von $1.00$ pro Person.
  - *Screeningtest* ($\mathrm{FN} = 100,\; \mathrm{FP} = 4{,}950$): Kosten von $0.1495$ pro Person. Die **Kosten sind $6.7\times$ niedriger**, obwohl die Genauigkeit um $4$ Prozentpunkte niedriger ist.

<!--en:8e1b91e4-->
::: {.slide title="Herleitung der Bayes-optimalen Entscheidungsschwelle"}

<!--en:076d49bb-->
[Von Wahrscheinlichkeiten zu optimalen Entscheidungen]{.kicker}

<!--en:bbbca929-->
Bei einer kalibrierten Wahrscheinlichkeit $p = P(\text{sick} \mid \mathbf{x})$ betragen die erwarteten Kosten für die Vorhersage *gesund* $p\,\kappa_{\mathrm{FN}}$ und für die Vorhersage *krank* $(1 - p)\,\kappa_{\mathrm{FP}}$:

<!--en:ea54bb2a-->
- **Gleiche Kosten ($\kappa_{\mathrm{FN}} = \kappa_{\mathrm{FP}}$):** Es ergibt sich die übliche Schwelle $\tau^* = 1/2$ ($\operatorname{argmax}$).

<!--en:49482fe1-->
- **Kostenverhältnis 100:1:** Die Schwelle sinkt auf $\tau^* = 1/101 \approx 0.01$. Bereits ein vorhergesagtes Risiko von $2\%$ rechtfertigt eine weitere Abklärung.

<!--en:43f709e7-->
::: {.slide title="Zielkonflikte bei der Schwellenwahl: Präzision, Trefferquote, F1 und ROC"}

<!--en:c13417d3-->
[Variation der Entscheidungsschwelle]{.kicker}

<!--en:e9f2b293-->
Die Wahl der Entscheidungsschwelle $\tau \in [0, 1]$ bestimmt den Zielkonflikt zwischen Trefferquote, Präzision und Zahl der Fehlalarme:

<!--en:591bc0a4-->
- **Senkung von $\tau$ ($\tau \downarrow$):** Die **Trefferquote steigt** monoton ($\mathrm{TP} \uparrow$), während die **Präzision sinkt** ($\mathrm{FP} \uparrow$).

<!--en:73bcfe64-->
- **F1-Wert:** Harmonisches Mittel $\mathrm{F1} = \frac{2\,\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}$. Es ist klein, sobald die Präzision oder die Trefferquote klein ist.

<!--en:1f622393-->
- **Kurven über alle Schwellen:** **ROC-Kurve** ($\mathrm{TPR}$ gegen $\mathrm{FPR}$, prävalenzunabhängige $\mathrm{AUC}$) und **Präzisions-Trefferquoten-Kurve** (bei nur $1\%$ positiven Fällen weitaus aussagekräftiger).

<!--en:2c5195d4-->
::: {.slide title="Wann ist ein Klassifikator kalibriert?"}

<!--en:6537ad3b-->
[Verlässliche Wahrscheinlichkeitsschätzungen]{.kicker}

<!--en:a4dd37e4-->
Für einen Klassifikator mit $q$ Klassen sind **Konfidenz** und **Korrektheit** jeder Vorhersage wie folgt definiert:

<!--en:a9c4539b-->
- **Kalibrierung der wahrscheinlichsten Klasse:** Für Beispiele mit vorhergesagter Konfidenz $\hat{p} \approx \alpha$ erfüllt die empirische Genauigkeit die Gleichung $P(c = 1 \mid \hat{p} = \alpha) = \alpha$.

<!--en:35cce9da-->
- **Unabhängig von der Genauigkeit:** Die konstante Vorhersage $(0.1, \ldots, 0.1)$ für $10$ gleich häufige Klassen ist perfekt kalibriert ($10\%$ Genauigkeit bei $0.10$ Konfidenz), aber ohne Aussagekraft.

<!--en:727e01a7-->
::: {.slide title="Konfidenzbereiche und erwarteter Kalibrierungsfehler (ECE)"}

<!--en:75da51a1-->
[Fehlkalibrierung messen]{.kicker}

<!--en:562bb15b-->
$[0, 1]$ in $B = 10$ Konfidenzbereiche $I_b = [\frac{b-1}{B}, \frac{b}{B})$ zerlegen. Für jeden Bereich die mittlere Konfidenz $\bar{p}_b$ mit der empirischen Genauigkeit $a_b$ vergleichen:

<!--en:5d769cde-->
- **Leere untere Bereiche:** Nach dem Schubfachprinzip gilt $\max_j \hat{y}_j \ge 1/10$. Für das Intervall $[0, 0.1)$ gilt daher immer $n_1 = 0$.

<!--en:7c3b4d31-->
::: {.slide title="Berechnung der Kennzahlen je Konfidenzbereich und des ECE"}

<!--en:75da51a1-2-->
[Fehlkalibrierung messen]{.kicker}

<!--en:ab55c32d-->
::: {.slide title="Prüfung der Kalibrierung bei der Softmax-Regression"}

<!--en:9383ef3c-->
[Zuverlässigkeit je Konfidenzbereich auf Fashion-MNIST]{.kicker}

<!--en:59705f61-->
Auswertung mit `reliability(conf, correct)` für die $10{,}000$ Vorhersagen auf den Validierungsdaten von Fashion-MNIST:

<!--en:2eea5ac8-->
::: {.slide title="Wie liest man die Zuverlässigkeitstabelle?"}

<!--en:9383ef3c-2-->
[Zuverlässigkeit je Konfidenzbereich auf Fashion-MNIST]{.kicker}

<!--en:b33beed6-->
- **Auswertung der besetzten Intervalle:** In den Intervallen von $[0.4, 0.5)$ bis $[0.9, 1.0]$ (etwa $97\%$ der Vorhersagen) weicht die empirische Genauigkeit um höchstens etwa $6$ Prozentpunkte von der mittleren Konfidenz ab. Insgesamt ergibt sich $\mathrm{ECE} < 0.03$.

<!--en:13df292b-->
::: {.slide title="Darstellung im Zuverlässigkeitsdiagramm"}

<!--en:3ba326c5-->
[Zu hohe und zu niedrige Konfidenz]{.kicker}

<!--en:3dfc2f94-->
- **Zuverlässigkeitsdiagramm:** $a_b$ gegen $\bar{p}_b$ auftragen, für alle Intervalle mit $n_b \ge 100$

<!--en:6b4db8e0-->
- **Diagonale als Referenz ($a_b = \bar{p}_b$):** Bei einem perfekt kalibrierten Klassifikator liegen die Punkte auf der gepunkteten Diagonalen.

<!--en:2a0712af-->
- **Lage zur Diagonalen:** Punkte *oberhalb* der Diagonalen zeigen **zu niedrige Konfidenz** an ($a_b > \bar{p}_b$), Punkte *unterhalb* zeigen **zu hohe Konfidenz** an ($a_b < \bar{p}_b$).

<!--en:817f3cb1-->
- **Flache und tiefe Netze:** Unser lineares Modell liegt knapp oberhalb der Diagonalen (Konfidenz $\approx 81\%$, Genauigkeit $\approx 83\%$). Tiefe Netze liegen meist deutlich darunter und benötigen $T > 1$.

<!--en:ed540d09-->
::: {.slide title="Nachträgliche Kalibrierung durch Temperaturskalierung"}

<!--en:e1319a8c-->
[Konvexe Kalibrierung mit einem Parameter]{.kicker}

<!--en:7caab417-->
**Temperaturskalierung** (Guo et al., 2017): Trainierte Gewichte festhalten. Einen einzigen Skalar $T > 0$ so anpassen, dass die negative Log-Likelihood (NLL) von $\mathrm{softmax}(\mathbf{o} / T)$ auf zurückgehaltenen Validierungsdaten minimal wird.

<!--en:98a62738-->
- **Warum zurückgehaltene Daten und warum kein Verlust an Genauigkeit?** Die Anpassung von $T$ auf Validierungsdaten berücksichtigt die Margins bei der Generalisierung. Da $T > 0$ das Ranking der Logits erhält, bleiben alle vorhergesagten Klassen und die Gesamtgenauigkeit **zu 100% unverändert**.

<!--en:8c3d38a9-->
::: {.slide title="Anpassung der Temperatur auf zurückgehaltenen Daten"}

<!--en:e1319a8c-2-->
[Konvexe Kalibrierung mit einem Parameter]{.kicker}

<!--en:65f6cbbe-->
::: {.slide title="Verständnisfragen: Präzision beim Screening, Enthaltung und ECE"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:fa8481cd-->
Drei Aufgaben zu Diagnose und Entscheidung lösen:

<!--en:d6457292-->
1. **Weniger Fehlalarme beim Screening:** In unserem Beispiel mit $1\%$ Prävalenz ($1{,}000$ Erkrankte, $99{,}000$ Gesunde) erreicht ein verbesserter Test weiterhin $90\%$ Trefferquote ($\mathrm{TP} = 900$). Bei den Gesunden sinkt die Falsch-Positiv-Rate von $5\%$ auf $1\%$. Wie hoch sind jetzt Genauigkeit und Präzision? Übertrifft der Test bei der Genauigkeit die Regel *„immer gesund“* ($99.0\%$)?

<!--en:6d0eab8e-->
2. **Enthaltung und fachärztliche Abklärung (Übung 1):** Eine richtige Diagnose koste $0$, eine Fehldiagnose $\lambda_s = 100$ und die Überweisung eines unklaren Falls zur fachärztlichen Abklärung $\lambda_r = 5$. Ab welcher Konfidenz $\hat{p}$ der wahrscheinlichsten Klasse sollte der automatische Klassifikator selbst eine Diagnose stellen?

<!--en:917f4e5b-->
3. **Änderung der Temperatur:** Unser Fashion-MNIST-Modell liefert leicht zu niedrige Konfidenzen. Was geschieht mit seiner Validierungsgenauigkeit und seinem ECE, bei Division aller Logits durch $T = 2$?

<!--en:3ce78fc4-->
**Lösungen:** **(1)** Aus $\mathrm{FP} = 0.01 \times 99{,}000 = 990$ und $\mathrm{TN} = 98{,}010$ ergibt sich eine Genauigkeit von $(900 + 98{,}010)/100{,}000 = 98.91\%$ (weiterhin unter $99.0\%$). Die Präzision steigt dagegen von $15.4\%$ auf $900 / (900 + 990) \approx 47.6\%$. **(2)** Bei einer eigenen Entscheidung betragen die erwarteten Fehlerkosten $(1 - \hat{p})\lambda_s$. Die Entscheidung ist günstiger als die Enthaltung, wenn $(1 - \hat{p})\lambda_s < \lambda_r$ $\iff$ $\hat{p} > 1 - \lambda_r/\lambda_s = 1 - 5/100 = 0.95$. **(3)** Die Genauigkeit bleibt gleich, da die Logits ihre Reihenfolge behalten. Mit $T = 2$ sinken die Konfidenzen weiter in Richtung $0.10$. Damit unterschätzt das Modell seine Treffsicherheit noch stärker und der ECE steigt.

<!--en:7ea8b25c-->
::: {.slide title="Zusammenfassung: Kennzahlen, Entscheidungen und Kalibrierung"}

<!--en:27de615d-->
[Zusammenfassung des Abschnitts]{.kicker}

<!--en:ff2d443b-->
| Kennzahl oder Verfahren | Mathematische Definition | Aussage oder Zweck |

<!--en:542b9593-->
| **Trefferquote (Sensitivität / TPR)** | $\mathrm{TP} / (\mathrm{TP} + \mathrm{FN}) = P(\hat{y}=1 \mid y=1)$ | Anteil der erkannten Fälle unter den tatsächlich positiven Fällen, unabhängig von der Prävalenz der Klasse |

<!--en:049239ba-->
| **Präzision (positiver Vorhersagewert, PPV)** | $\mathrm{TP} / (\mathrm{TP} + \mathrm{FP}) = P(y=1 \mid \hat{y}=1)$ | Anteil der tatsächlich positiven Fälle unter den positiv eingestuften Fällen, abhängig von der Prävalenz gemäß dem Satz von Bayes |

<!--en:50b503c3-->
| **Bayes-Schwelle für Fehlerkosten** | Positive Vorhersage genau dann, wenn $p > \frac{\kappa_{\mathrm{FP}}}{\kappa_{\mathrm{FP}} + \kappa_{\mathrm{FN}}}$ | Minimiert die erwarteten Kosten bei asymmetrischen Fehlerkosten ($\kappa_{\mathrm{FN}}=100, \kappa_{\mathrm{FP}}=1$ $\implies$ $\tau^* = 1/101$) |

<!--en:4a4c17d5-->
| **Zuverlässigkeitsdiagramm und ECE** | $\mathrm{ECE} = \sum_{b=1}^B \frac{n_b}{n}\big|a_b - \bar{p}_b\big|$ | Prüfen, ob die vorhergesagte Konfidenz $\bar{p}_b$ mit der empirischen Genauigkeit $a_b$ im jeweiligen Intervall übereinstimmt |

<!--en:5abb9706-->
| **Temperaturskalierung** | $\hat{\mathbf{y}}(T) = \mathrm{softmax}(\mathbf{o}/T),\; T^* = \operatorname{argmin}_T \mathrm{NLL}_{\mathrm{val}}(T)$ | Eindimensionale konvexe Nachkalibrierung auf zurückgehaltenen Daten, bei unveränderter Genauigkeit |
