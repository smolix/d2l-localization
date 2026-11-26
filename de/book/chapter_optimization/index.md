<!--en:5ef9f633-->
# Optimierungsalgorithmen

<!--en:267e3a3d-->
In den vorangegangenen Kapiteln haben wir jedes tiefe Netz nach einem bewährten Trainingsrezept trainiert. Wir haben beispielsweise den stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD) mit Minibatches und einem Momentum von $0.9$ verwendet oder AdamW mit linearer Aufwärmphase (engl. warmup) und Kosinus-Abfall (engl. cosine decay) (:numref:`sec_training_recipes`). Divergiert ein Trainingslauf und liefert `nan`, senken wir die Lernrate. Stockt der Fortschritt, passen wir den Lernratenplan oder die Batchgröße an. In diesem Kapitel untersuchen wir, wie ein Optimierer arbeitet. Dabei beantworten wir drei praktische Fragen. Wovon hängen die höchste stabile Lernrate und die Geschwindigkeit der Konvergenz bei niedrigeren Lernraten ab? Wie verändert jeder Bestandteil eines modernen Trainingsrezepts den Aktualisierungsschritt? Wie müssen wir diese Hyperparameter skalieren, wenn Modelle und Batchgrößen wachsen?

<!--en:a1cfb999-->
Der Gradientenabstieg ist älter als der moderne Computer. Augustin-Louis Cauchy schlug ihn 1847 zur Berechnung von Planetenbahnen vor. 1951 untersuchten Herbert Robbins und Sutton Monro, wie sich Nullstellen bei verrauschten Beobachtungen mit abnehmenden Schrittweiten finden lassen. Sowohl in klassischen als auch in modernen Anwendungen stößt jeder gradientenbasierte Optimierer auf zwei grundlegende Hindernisse:

<!--en:20853ff6-->
1. **Schlechte Konditionierung (ungleichmäßige Krümmung):** Die Verlustlandschaft (engl. loss landscape) kann in manchen Richtungen stark und in anderen nur schwach gekrümmt sein. Wählen wir die Lernrate so klein, dass das Verfahren in der steilsten Richtung stabil bleibt, kommen wir in den flachsten Richtungen nur sehr langsam voran.

<!--en:6d77785d-->
2. **Gradientenrauschen:** Das Gradientenrauschen (engl. gradient noise) entsteht, weil wir den Gradienten aus endlichen Minibatches schätzen, also nur aus einem Teil des Datensatzes. Daher unterliegt jeder Aktualisierungsschritt einer Stichprobenvarianz. Bei konstanter Schrittweite verhindert sie, dass die Iterierten in einem Minimum zur Ruhe kommen.

<!--en:7f987165-->
![Die beiden Hindernisse in einem Tal, dessen Krümmung quer zum Tal $20$-mal so groß ist wie in Längsrichtung ($\kappa = 20$). Links: Der Gradientenabstieg führt im Zickzack über die steilen Talflanken und kommt in der flachen Längsrichtung nur langsam voran. Rechts: Bei verrauschten Gradienten und konstanter Schrittweite erreichen die Iterierten das Minimum, kommen dort aber nie zur Ruhe. Sie bewegen sich innerhalb einer Rauschkugel (gestrichelte Ellipse: zwei Standardabweichungen).](../img/mdl-opt-two-obstacles.svg)

<!--en:2eb95fef-->
:numref:`fig_opt_two_obstacles` zeigt beide Hindernisse im selben zweidimensionalen Tal. Jede Entwurfsentscheidung bei einem modernen Optimierer zielt auf eines dieser Hindernisse. Dafür gibt es drei Ansatzpunkte:

<!--en:02b01f48-->
* **Schrittrichtung:** Momentum (:numref:`sec_momentum`), koordinatenweise adaptive Verfahren wie Adam und AdamW (:numref:`sec_adam`) und Verfahren wie Muon (:numref:`sec_muon`), die ganze Matrizen orthogonalisieren, ersetzen den ursprünglichen Gradienten durch eine vorkonditionierte oder geglättete Richtung. Damit kommen sie in schlecht konditionierten Tälern schneller voran.

<!--en:26068e71-->
* **Schrittweitenplan:** Mit einer Aufwärmphase für die Lernrate, einem Kosinus-Abfall oder einem linearen Abfall sowie mit dreiphasigen Plänen (Aufwärmphase, konstante Lernrate, Abfall, engl. warmup–stable–decay, WSD) (:numref:`sec_scheduler`) wägen wir zwei Ziele gegeneinander ab. Anfangs möchten wir rasch vorankommen und am Ende das Rauschen gering halten.

<!--en:52fc5741-->
* **Minibatch-Größe und Skalierung:** Die kritische Batchgröße (engl. critical batch size) (:numref:`sec_batch_size`) gibt an, ab wann größere Minibatches die Zahl der Optimierungsschritte nicht weiter verringern. Mit der Maximal Update Parametrization ($\mu\text{P}$, :numref:`sec_scaling`) können wir Lernraten an einem kleinen Modell abstimmen und direkt auf wesentlich breitere Netze übertragen.

<!--en:30161f0b-->
Für jedes Verfahren betrachten wir ein analytisches Modell, ein zweidimensionales quadratisches Tal mit der Konditionszahl $\kappa = 20$. Wir ergänzen die Analyse durch systematische Messungen an zwei neuronalen Netzen. In :numref:`sec_test_models` stellen wir beide Testmodelle vor: den konvolutionären Klassifikator `FashionCNN` und das Transformer-Sprachmodell `TinyLM`, das auf Byte-Ebene arbeitet. Zum Abschluss verbinden wir in :numref:`sec_practice` Gradienten-Clipping, Gewichtsmittelung und die Skalierung von Hyperparametern zu einem durchgängigen Verfahren. Damit können wir große Trainingsläufe konfigurieren und Probleme diagnostizieren.

<!--en:4583ac54-->
Dieses Kapitel setzt Kenntnisse über Gradienten, Eigenwerte symmetrischer Matrizen und die Grundlagen von Minibatch-SGD voraus (:numref:`subsec_linear-regression-minibatch-sgd`). Wo eine Herleitung unmittelbar bei der Wahl der Hyperparameter hilft, führen wir sie kurz aus. Längere mathematische Beweise finden sich in :numref:`chap_mdl-optimization`. Kommentierte Hinweise auf Lehrbücher, Forschungsarbeiten und Praxisleitfäden stehen in :numref:`sec_optimization-further-reading`.

<!--en:544e580a-->
Optimierungsalgorithmen: Geometrie, Stochastik und Skalierung<br>**schlechte Konditionierung · Gradientenrauschen · Momentum · AdamW und Muon · Lernratenpläne · kritische Batchgröße · μP**

<!--en:1ff56a59-->
::: {.slide title="Zwei Hindernisse für jeden Optimierer im Deep Learning"}

<!--en:acc8203e-->
[Das zentrale Problem]{.kicker}

<!--en:b9b351be-->
- **Hindernis 1 – schlechte Konditionierung ($\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$):**
  - Für $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$ ($\lambda_1 = 0.2, \lambda_2 = 4, \kappa = 20$) erfordert Stabilität in Richtung $x_2$ eine Lernrate von $\eta < 2/\lambda_2 = 0.5$. Dadurch sinkt der Betrag von $x_1$ pro Schritt um $<10\%$ ($47$ Schritte bis auf $1\%$ Fehler).

<!--en:550499f4-->
- **Hindernis 2 – Gradientenrauschen bei Minibatches ($\operatorname{tr}(\boldsymbol{\Sigma})/b$):**
  - Beim Sampling von $b$ Beispielen bleiben die Iterierten bei konstanter Lernrate in einer stationären **Rauschkugel** mit Varianz $\eta\sigma^2 / (2\lambda b)$.

<!--en:26dd0777-->
- **These des Kapitels:** Jede Stellgröße eines modernen Optimierers setzt bei einem dieser beiden Engpässe an.

<!--en:bdab91cc-->
::: {.slide title="Drei Stellgrößen eines modernen Trainingsrezepts"}

<!--en:49387932-->
[Aufbau des Trainingsrezepts]{.kicker}

<!--en:80315afc-->
- **1. Schrittrichtung und Vorkonditionierung (§§{sec_momentum}–§{num:sec_muon}):**
  - $-\nabla f(\mathbf{x})$ durch den Geschwindigkeitsvektor des Momentum-Verfahrens ($\mathbf{v}_t$), durch koordinatenweise RMS-Normierung (AdamW) oder durch spektrale Orthogonalisierung ($\mathbf{U}\mathbf{V}^\top$, Muon) ersetzen

<!--en:d4d47dfd-->
- **2. Schrittweitenplan $\eta_t$ (§{sec_scheduler}):**
  - $\eta_t$ in der Aufwärmphase allmählich erhöhen, damit die Krümmung unter $2/\eta_0$ bleibt. Eine große Summe der Schrittweiten $B_T = \sum_t \eta_t$ beibehalten und die Lernrate zum Ende hin senken ($\eta_T \to 0$).

<!--en:347a23e7-->
- **3. Batchgröße, Skalierung mit der Netzbreite und Schutzmaßnahmen (§§{sec_batch_size}–§{num:sec_practice}):**
  - $b$ an $b_{\text{noise}}$ anpassen, Lernraten mittels $\mu\text{P}$ auf andere Netzbreiten übertragen ($\eta/m$) und das Training durch Clipping der globalen Gradientennorm und exponentiell gleitende Mittelung der Gewichte absichern

<!--en:af68ee7b-->
::: {.slide title="Von exakt lösbaren Modellen zu Tests an neuronalen Netzen"}

<!--en:b7a658bf-->
[Vorgehen im Kapitel]{.kicker}

<!--en:be94bbbe-->
- **Quadratische Funktionen und lineare Modelle auf Daten (§§{sec_gd}–§{num:sec_sgd}):**
  - Stabilitätsgrenzen, kritische Dämpfung mit $\sqrt{\kappa}$-Skalierung und Überschussverlust $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}}{4b}$ exakt bestimmen: am Tal mit $\kappa=20$ und am NASA-Datensatz zu Tragflächenprofilen ($n=1{,}500, d=6$)

<!--en:8cabdac7-->
- **Zwei einander ergänzende neuronale Netze (§§{sec_test_models}–§{num:sec_practice}):**
  - **`FashionCNN` ($421\text{K}$ Parameter):** Bildklassifikation über mehrere Epochen ($8.5$ Epochen in $2{,}000$ Schritten bei $b = 256$), Genauigkeit auf separaten Testdaten
  - **`TinyLM` ($471\text{K}$ / $3.3\text{M}$ Parameter):** Transformer auf Byte-Ebene, ein Durchlauf durch WikiText-103 (Verhältnis der größten zur kleinsten Gradientengröße über die Schichten: $40\times$)

<!--en:5ce1636c-->
::: {.slide title="Kapitel §{num:chap_optimization} im Überblick: Abschnitte und zentrale Zusammenhänge"}

<!--en:d1861b99-->
[Kapitelübersicht]{.kicker}

<!--en:6dc634f1-->
- **Grundlagen und Vergleichstests (§§{sec_gd}–§{num:sec_test_models}):**
  - Stabilität $\eta < 2/\lambda_{\max}$, Stabilitätsgrenze $\lambda_{\max} \approx 2/\eta$, Rauschkugel $\frac{\eta\sigma^2}{2\lambda b}$, Vergleichswerte aus Häufigkeiten und $3\times$-Schwellenwerte für Läufe mit verschiedenen Zufallsstartwerten

<!--en:46a6adb8-->
- **Vorkonditionierte Richtungen (§§{sec_momentum}–§{num:sec_muon}):**
  - Effektive Schrittweite bei Momentum $\eta_{\text{eff}} = \frac{\eta}{1-\beta}$ ($O(\sqrt{\kappa})$), Zeitskala von AdamW $\tau = \frac{1}{\eta\lambda}$ und Orthogonalisierung bei Muon mit 5 Newton-Schulz-Schritten für die Aktualisierung $-\eta\mathbf{U}\mathbf{V}^\top$

<!--en:4f271e46-->
- **Lernratenpläne, Skalierung und Trainingsbetrieb (§§{sec_scheduler}–§{num:sec_practice}):**
  - Aufwärmphase und Schrittweitenbudget $B_T$ bei WSD- oder Kosinusplänen, kritische Batchgröße $b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$, $\mu\text{P}$-Lernrate $\eta/m$ für verborgene Schichten, Gradienten-Clipping ($\theta=1$) und exponentiell gleitende Mittelung der Gewichte

<!--en:223f6751-->
::: {.slide title="Wegweiser durch Kapitel §{num:chap_optimization} und den mathematischen Anhang"}

<!--en:123f19cf-->
[Leseleitfaden]{.kicker}

<!--en:70b0b512-->
- **Grundlagen für die Praxis (§{sec_gd}–§{sec_adam}, §{sec_scheduler}–§{sec_batch_size}, §{sec_practice}):**
  - Die beiden Engpässe (**§{sec_gd}–§{sec_sgd}**), experimentelle Vergleichstests (**§{sec_test_models}**), Momentum und AdamW (**§{sec_momentum}–§{sec_adam}**), Lernratenpläne (**§{sec_scheduler}**), die kritische Batchgröße (**§{sec_batch_size}**) und die Diagnose im Trainingsbetrieb (**§{sec_practice}**) erarbeiten

<!--en:3da22afa-->
- **Weiterführende Themen zur Skalierung und zu Matrixverfahren (§{sec_muon}, §{sec_scaling}):**
  - **§{sec_muon} (Muon)** behandelt den steilsten Abstieg bezüglich der Spektralnorm mithilfe von Newton-Schulz-Polynomen. **§{sec_scaling} (Skalierung)** zeigt, wie sich Lernraten mittels $\mu\text{P}$ ohne erneute Abstimmung auf andere Netzbreiten übertragen lassen.

<!--en:5b1bf5f3-->
- **Ausblick und Beweise:**
  - `TinyLM`, der zugehörige Datenlader für WikiText-103 auf Byte-Ebene und der Trainingscode für AdamW und Lernratenpläne kommen in den Kapiteln über Attention, Transformer, Zustandsraummodelle und Rechenleistung erneut zum Einsatz.
  - Kurze Herleitungen, die bei der Wahl der Hyperparameter helfen, stehen direkt im Text. Vollständige Konvergenzbeweise und die konvexe Analysis finden sich in **Kapitel §{num:chap_mdl-optimization} (*Mathematik für Deep Learning: Optimierung*)**.

<!--en:f508964c-->
**Als Nächstes (§{sec_gd} Verlustlandschaften und Gradientenabstieg):** Wie die richtungsabhängige Krümmung und die Konditionszahl $\kappa = \lambda_{\max}/\lambda_{\min}$ die größte stabile Schrittweite und die Stabilitätsgrenze bestimmen.
