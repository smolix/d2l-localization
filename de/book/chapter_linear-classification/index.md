<!--en:768360f4-->
# Lineare Klassifikation

<!--en:0bb62bb9-->
In :numref:`chap_regression` wollten wir eine stetige Größe vorhersagen, etwa den Verkaufspreis eines Hauses. Bei vielen Aufgaben ist hingegen eine diskrete Kategorie gefragt. Ein Foto im Onlinekatalog zeigt ein Hemd, einen Mantel oder eine Sandale, aber niemals „5.5“ von irgendetwas. Die Kategorien von 1 bis 10 zu nummerieren und eine lineare Regression anzuwenden, führt zu schlechten Ergebnissen. Die Nummerierung erzwingt eine beliebige Reihenfolge, als läge ein Pullover genau zwischen einer Hose und einem Kleid. Zugleich kann das Modell seine Unsicherheit nicht kalibriert ausdrücken („höchstwahrscheinlich ein Hemd, möglicherweise ein Pullover“).

<!--en:586df2de-->
Fast alles, was wir zum Training für die Regression brauchen, können wir direkt für die *Klassifikation* übernehmen. Dazu gehören Datenlader für Minibatches, vektorisierte Vorwärtsdurchläufe, automatisches Differenzieren und stochastischer Gradientenabstieg (engl. stochastic gradient descent, SGD) mit Minibatches. Nur drei Dinge ändern sich. Erstens stellen wir ungeordnete Kategorien durch *One-Hot-Vektoren* dar. Zweitens berechnen wir für jede Kategorie einen linearen Score (*Logit*). Mit der *Softmax-Funktion* normalisieren wir diese Scores zu einer Wahrscheinlichkeitsverteilung. Drittens ersetzen wir den quadratischen Fehler durch den *Kreuzentropieverlust* (engl. cross-entropy loss). Er ergibt sich nach dem Maximum-Likelihood-Prinzip aus einer kategorialen Verteilung der Labels, so wie sich der quadratische Fehler aus gaußschem Rauschen ergab. Fast alle neuronalen Klassifikatoren in diesem Buch, bis hin zu großen Sprachmodellen, enden mit dieser Softmax-Schicht. Beim Training minimieren wir diesen Verlust.

<!--en:c5bbaa25-->
Zunächst formulieren wir die Softmax-Regression und den Kreuzentropieverlust. Wir leiten den Gradienten des Verlusts her, der dem Residuum $\hat{\mathbf{y}} - \mathbf{y}$ entspricht, und stellen den Zusammenhang mit der Informationstheorie von Shannon her (:numref:`sec_softmax`). Anschließend erstellen wir eine wiederverwendbare Bildpipeline für `FashionMNIST` (:numref:`sec_fashion_mnist`). Für die Softmax-Regression schreiben wir sowohl eine eigene Implementierung von Grund auf als auch eine kompakte Implementierung (:numref:`sec_softmax_scratch`). Mit Konfusionsmatrizen (engl. confusion matrices) und erlernten Schablonen der Größe $28 \times 28$ untersuchen wir Fehlklassifikationen. Überlauf und Unterlauf in der Gleitkommaarithmetik vermeiden wir, indem wir den Verlust mit einer kombinierten Log-Sum-Exp-Operation berechnen.

<!--en:3a0e3086-->
Bei der *Auswertung* (engl. evaluation) und beim *Einsatz* (engl. deployment) unterscheidet sich die Klassifikation deutlich von der Regression. Beiden Themen widmen wir die zweite Kapitelhälfte. Die Genauigkeit (engl. accuracy) allein führt in die Irre, wenn positive Beispiele selten sind oder unterschiedliche Fehler unterschiedliche Kosten verursachen. Deshalb betrachten wir Präzision (engl. precision), Trefferquote (engl. recall), kostenabhängige Schwellenwerte und Verfahren zur Prüfung der Kalibrierung (:numref:`sec_classification`). Selbst bei ausgewogenen Klassen sind Ergebnisse auf einem endlichen Testdatensatz mit statistischer Unsicherheit behaftet. Die zugehörigen Fehlerbalken verlieren an Aussagekraft, wenn wir denselben Testdatensatz wiederholt verwenden. Das führt uns zu Hoeffding-Schranken, adaptiver Überanpassung (engl. adaptive overfitting) und der Theorie von Vapnik und Chervonenkis (VC-Theorie) (:numref:`chap_classification_generalization`). In :numref:`sec_environment-and-distribution-shift` untersuchen wir schließlich, was geschieht, wenn die Verteilung der Daten im Einsatz zunehmend von der Trainingsverteilung abweicht. Wir korrigieren Kovariatenverschiebungen (engl. covariate shift) durch Wichtigkeitsgewichtung (engl. importance weighting), schätzen Labelverteilungsverschiebungen (engl. label shift) anhand der Konfusionsmatrix und entwickeln Maßnahmen gegen Rückkopplungsschleifen, die sich immer weiter verstärken.

<!--en:53a3d02e-->
Lineare neuronale Netze für die Klassifikation: von Logits zu verlässlichen Entscheidungen<br>**One-Hot-Labels · Softmax-Funktion · Kreuzentropie · Fashion-MNIST · Kalibrierung · Generalisierung · Verteilungsverschiebung**

<!--en:f8833c6b-->
::: {.slide title="Warum die Klassifikation eine eigene Verlustfunktion und eigene Diagnoseverfahren braucht"}

<!--en:1f3b154f-->
[Die zentrale Herausforderung]{.kicker}

<!--en:6846f0a9-->
Die Vorhersage einer ungeordneten diskreten Kategorie ($y \in \{1, \ldots, q\}$) unterscheidet sich in drei wesentlichen Punkten von der Regression mit stetiger Zielgröße:

<!--en:a88f331f-->
- **Beliebige ganzzahlige Codes verzerren die Geometrie:** Tragen zehn Kleidungsstücke die Nummern $1$ bis $10$, bestraft der quadratische Verlust die Verwechslung T-Shirt ($1$) $\to$ Stiefel ($10$) $81\times$ so stark wie T-Shirt ($1$) $\to$ Hose ($2$).

<!--en:8b78f045-->
- **Genauigkeit allein verschleiert einseitige Fehler:** Bei einer Krankheitsprävalenz von $1\%$ erreicht die konstante Vorhersage *gesund* eine Genauigkeit von $99\%$ und erkennt $0\%$ der Erkrankten.

<!--en:3f28b9a2-->
- **Testdatensätze sind nur für ihre eigene Verteilung aussagekräftig:** Erreicht ein lineares Modell auf der Ausgangsverteilung $85\%$ Genauigkeit, kann diese bei einer Verteilungsverschiebung im Einsatz auf $\sim 50\%$ fallen.

<!--en:5ab41206-->
::: {.slide title="Was sich beim Übergang von der Regression zur Klassifikation ändert"}

<!--en:a1c79e72-->
[Modellierung in drei Schritten]{.kicker}

<!--en:416c0c4a-->
Die Trainingspipeline für die Regression mit Minibatch-Datenladern, vektorisierter GEMM, automatischem Differenzieren und SGD lässt sich mit drei Änderungen übernehmen:

<!--en:b8e2a924-->
- **One-Hot-Ziel und Ausgabeschicht mit mehreren Logits:** Die Skalare $y \in \mathbb{R}$ und $\hat{y}$ durch den One-Hot-Vektor $\mathbf{y} \in \{0, 1\}^q$ und $q$ affine Logits $\mathbf{o} = \mathbf{W}^\top\mathbf{x} + \mathbf{b}$ ersetzen. Die Logits mit $\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})$ normalisieren.

<!--en:e3dd7b1d-->
- **Kreuzentropieverlust aus der Likelihood:** Den quadratischen Fehler des Gauß-Modells durch die negative Log-Likelihood der kategorialen Verteilung $l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^q y_j \log \hat{y}_j = -\log \hat{y}_y$ ersetzen

<!--en:92befe33-->
- **Gradient in derselben Residuenform:** Der Gradient nach den Logits vereinfacht sich zu $\nabla_{\mathbf{o}}\, l = \hat{\mathbf{y}} - \mathbf{y}$ (Vorhersage minus Zielwert).

<!--en:5ae0c3fe-->
::: {.slide title="Aufbau und Themen von Kapitel §{num:chap_classification}"}

<!--en:99e0030a-->
[Kapitelüberblick]{.kicker}

<!--en:7e143e5a-->
Die sechs Abschnitte führen von der mathematischen Formulierung und den Benchmark-Daten über die Implementierung und Kennzahlen für Entscheidungen bis zu statistischen Garantien und zur Robustheit im Einsatz:

<!--en:c8693da4-->
- **Modell und Pipeline (§{sec_softmax}–§{sec_softmax_scratch}):** Wahrscheinlichkeiten aus der Softmax-Funktion, Kreuzentropieverlust, `DataModule` für Fashion-MNIST ($60\text{k} + 10\text{k}$ Kleidungsstücke), Basisklasse `Classifier`, Konfusionsmatrizen der Größe $10\times 10$ und Verlustberechnung mit kombinierter Log-Sum-Exp-Operation

<!--en:7f5ddf9f-->
- **Entscheidungen, Schranken und Verteilungsverschiebungen (§{sec_classification}–§{sec_environment-and-distribution-shift}):** Präzision und Trefferquote, kostenabhängige Bayes-Schwellenwerte, Prüfung der Kalibrierung mit ECE, Fehlerbalken für den Testdatensatz ($\le 0.5/\sqrt{n}$), adaptive Überanpassung, VC-Dimension und Korrektur von Kovariaten- und Labelverteilungsverschiebungen

<!--en:268ead54-->
::: {.slide title="Durchgängige Benchmarks in allen Abschnitten"}

<!--en:4ed205ed-->
[Zahlen aus den Experimenten]{.kicker}

<!--en:f91443c1-->
Jedes Konzept in diesem Kapitel lässt sich anhand konkreter Zahlen und reproduzierbarer Experimente nachvollziehen:

<!--en:58cb1508-->
- **Grundmodell (§{sec_softmax}–§{sec_softmax_scratch}):** $2\times 2$-Beispiel mit Logits und zugehörigen Wahrscheinlichkeiten $(1.0, 2.2, 0.3) \to (0.21, 0.69, 0.10)$ ($0.37\text{ nats}$). Der Fashion-MNIST-Klassifikator mit $7{,}850$ Parametern erreicht eine Validierungsgenauigkeit von $\sim 83\%$ in $10$ Epochen.

<!--en:a0955624-->
- **Modellprüfung und Verteilungsverschiebung (§{sec_classification}–§{sec_environment-and-distribution-shift}):** Früherkennung bei $1\%$ Prävalenz ($90\%$ Trefferquote $\to 15.4\%$ Präzision, $\tau^* \approx 1\%$ bei einem Kostenverhältnis von $100:1$). Intervallradius nach Normalapproximation $\pm 0.7\%$, nach Hoeffding $\pm 1.4\%$ bei $n=10\text{k}$. Wichtigkeitsgewichtung ($\sim 55\% \to 95\%$) und Rekonstruktion der Klassenanteile mit $\mathbf{C}^{-1}\boldsymbol{\mu}$ bei Labelverteilungsverschiebung.

<!--en:55017918-->
::: {.slide title="Verständnisfragen und Lernhilfe zum Kapitel"}

<!--en:7a972241-->
[Selbsteinschätzung]{.kicker}

<!--en:73302fd2-->
Vor dem Einstieg in §{sec_softmax} die eigene Intuition anhand von drei Grundfragen des Kapitels prüfen:

<!--en:0b028eac-->
1. **Anzahl der Ausgaben:** Warum braucht ein linearer Klassifikator mit 10 Klassen für Bilder der Größe $28 \times 28$ insgesamt $7{,}850$ Parameter, während die lineare Regression mit $785$ Parametern auskommt?

<!--en:9c4d1a39-->
2. **Zahlenbereich:** Warum ergibt die Gleitkommaberechnung mit einfacher Genauigkeit (`float32`) für $\exp(1000) / (\exp(1000) + \exp(0) + \exp(0))$ den Wert `NaN`, obwohl der Quotient mathematisch gleich $1$ ist?

<!--en:4deeeb78-->
3. **Genauigkeit der Auswertung:** Zwei Modelle unterscheiden sich um $0.3$ Prozentpunkte auf einem Testdatensatz mit $10{,}000$ Bildern bei einer Genauigkeit von rund $83\%$. Lässt sich dieser Unterschied statistisch von zufälligen Stichprobenschwankungen unterscheiden?

<!--en:c5f3f3eb-->
**Lösungen:** **(1)** Jede der $q = 10$ Klassen braucht für ihre $784$ Pixel einen eigenen Gewichtsvektor $\mathbf{w}_j$ und einen eigenen Bias $b_j$, um die Klasse $j$ gegenüber allen anderen zu bewerten ($784 \times 10 + 10 = 7{,}850$). **(2)** Bei `float32` führt ein Überlauf zu `inf`, sobald der Wert etwa $\exp(88.7) \approx 3.4 \times 10^{38}$ erreicht. Der Quotient wird damit zu $\infty / \infty = \texttt{NaN}$. Zieht man zuvor $\max_k o_k = 1000$ ab, bleiben alle Exponenten $\le 0$. **(3)** Nein. Bei $\epsilon = 0.17$ und $n = 10{,}000$ beträgt der Standardfehler eines einzelnen Modells $\sqrt{0.17 \times 0.83 / 10{,}000} \approx 0.38$ Prozentpunkte ($95\%$-Intervall mit Radius $\pm 0.74$ Prozentpunkten).
