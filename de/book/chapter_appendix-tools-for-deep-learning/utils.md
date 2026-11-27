<!--en:b3547dfe-->
# Hilfsfunktionen und Hilfsklassen

<!--en:716ee7e7-->
Dieser Abschnitt enthält die Implementierungen der Hilfsfunktionen und Hilfsklassen, die wir in diesem Buch verwenden. Jede mit `#@save` markierte Zelle wird in das Paket `d2l` kopiert. So können wir ihre Funktionen in jedem Abschnitt als `d2l.plot`, `d2l.set_figsize` und so weiter aufrufen.

<!--en:d31c7403-->
## Hilfsfunktionen für Hyperparameter

<!--en:2152a294-->
## Fortschrittsanzeige und Diagramme

<!--en:b4b4870b-->
Die meisten Diagramme in diesem Buch entstehen mit `d2l.plot`. Die Funktion zeichnet eine oder mehrere Kurven in dasselbe Koordinatensystem. Die erste Zelle legt Format und Größe der Abbildung fest (`use_svg_display`, `set_figsize`) sowie Beschriftung, Wertebereiche und Skalierung der Achsen (`set_axes`). Auf einer logarithmischen Achse beschriftet `set_log_ticks` alle Hauptteilstriche in einem einheitlichen Format.

<!--en:eecf68ce-->
Der größte Teil des Codes von `plot` bringt die Eingaben in die Form einer Liste von Kurven.

<!--en:99808721-->
Das `ProgressBoard` aus :numref:`sec_oo-design` zeichnet im aufrufenden Thread. `show` zeichnet alle Kurven in eine neue Abbildung und ersetzt damit die vorherige Ansicht desselben Boards in der Notebook-Ausgabe. Beim Erstellen des Buchs ohne grafische Oberfläche zeichnet nur der letzte Aufruf eine Abbildung. Das ausgeführte Notebook enthält damit eine Abbildung pro Board. Die linke Achse zeigt den Verlust. Kommt die Genauigkeit hinzu, erhält sie rechts eine zweite Achse. Die Legende rückt dann über das Diagramm, damit sie keine der Kurven verdeckt. Punkte markieren die Validierungswerte und alle Kurven, die aus einem einzigen Wert bestehen. So bleibt auch ein Lauf mit nur einer Epoche sichtbar.

<!--en:f857136c-->
`show_trace_2d` zeichnet die Iterationspunkte einer zweidimensionalen Aktualisierungsregel, wie sie `d2l.train_2d` aus :numref:`sec_gd` zurückgibt, über die Niveaulinien der Zielfunktion `f`. Der Bildausschnitt umfasst den Startpunkt $(-5, -2)$ und das Minimum im Ursprung. Er vergrößert sich so weit, dass der gesamte Pfad hineinpasst. Beide Achsen haben denselben Maßstab, sofern der Pfad nicht entlang einer Achse divergiert.

<!--en:5d93bf09-->
`plot_intervals` zeichnet Schätzwerte mit ihren Intervallen, etwa Genauigkeiten mit Fehlerbalken aus dem Bootstrap-Verfahren. `series` ordnet jeder Bezeichnung drei Sequenzen derselben Länge wie `x` zu. Sie enthalten die Schätzwerte sowie die unteren und oberen Intervallgrenzen. Bei `nan` entfällt der betreffende Punkt. Mit `kind='errorbar'` erscheint jeder Schätzwert als Symbol mit Fehlerbalken. Die Datenreihen stehen an jeder Position `x` nebeneinander, damit sich die Balken nicht verdecken. Mit `kind='band'` erscheint jede Reihe als Linie mit schattiertem Band. `x` enthält Zahlen oder Kategorienamen. `reference` zeichnet eine gestrichelte horizontale Linie, etwa bei der Genauigkeit für zufälliges Raten oder bei null. Für `xscale` stehen `'linear'`, `'log'` und `'log2'` zur Wahl. Die Reihe $i$ erhält die Farbe `Ci`. Die Funktion gibt die Achsen zurück, sodass der aufrufende Code farblich passende Linien ergänzen kann. `marks` fügt den Reihen weitere Punkte hinzu, etwa die eines zweiten Schätzers. Dazu ordnet es einem Namen ein Matplotlib-Format (`'--'`, `'D'`) und für jede Reihe die Punkte `(xs, ys)` zu. Die Punkte erscheinen in der jeweiligen Reihenfarbe, ihr Name einmal in Schwarz in der Legende.

<!--en:8c7c700c-->
## Hilfsfunktionen für Reinforcement Learning

<!--en:699fc9ea-->
`run_seeds` führt einen Python-Generator für das Training einmal pro Zufallsstartwert aus. Die ausgegebenen Kurven fasst die Funktion zu einem Array der Form (Zufallsstartwert, Schritt, ...) zusammen, wie es `plot_curves` erwartet. Off-Policy-Agenten liefern einmal pro Episode Messwerte. Je nach Zufallsstartwert schließen sie unterschiedlich viele Episoden ab. `on_grid` überträgt diese Messwerte auf ein gemeinsames Raster von Umgebungsschritten, sodass sie sich ebenfalls in einem Array zusammenfassen und zeichnen lassen.

<!--en:896c846c-->
Die JAX-Implementierungen der Agenten für Reinforcement Learning wählen für jeweils einen Zustand eine Aktion. Für die Aktualisierungen verwenden sie Batches, deren Länge von Batch zu Batch variiert. Welche kompilierte Fassung zum Einsatz kommt, hängt von der Netzstruktur, den Formen der Eingaben und dem Optimierer ab. Drei Hilfsfunktionen halten die Zahl der Kompilierungen klein. `forward_np` kompiliert pro Netzstruktur einen Vorwärtsdurchlauf und liest bei jedem Aufruf die aktuellen Netzparameter. Ein Optimierungsschritt erfordert daher keine erneute Kompilierung. `_pad` und `_bucket` füllen einen Batch bis zur nächsten passenden Zweierpotenz auf (mindestens 64 Zeilen). So lässt sich eine kompilierte Aktualisierung für Batches unterschiedlicher Länge verwenden. `padded` füllt mehrere Arrays eines Batches gemeinsam auf und ergänzt eine Maske für die tatsächlichen Datenzeilen. Bei der Aktualisierung lässt sich das Padding so aus den Mittelwerten ausschließen. `shared_adam` hält pro Lernrate eine Instanz von `optax.adam` vor. Agenten mit derselben Lernrate nutzen daher dieselben kompilierten Aktualisierungen. `plain_sgd` leistet dasselbe für den einfachen stochastischen Gradientenabstieg (SGD), den einige Experimente anstelle von Adam verwenden.

<!--en:5c6c9a8f-->
## Merkmalsnetze für die Bildbewertung

<!--en:614c641d-->
In den Kapiteln zu generativen Modellen bewerten wir Bilder anhand der Merkmale eines kleinen CIFAR-10-Klassifikators (:numref:`sec_dcgan`). `load_cifar10_arrays` gibt die Trainings- und Testbilder von CIFAR-10, skaliert auf $[-1, 1]$, mit ihren Labels zurück. Der Klassifikator `FeatureCNN` besteht aus drei Faltungsblöcken mit je einer Batchnormalisierung, einer ReLU und einem $2 \times 2$-Max-Pooling. Darauf folgen Global-Average-Pooling und ein linearer Ausgabekopf. Die Methode `features` gibt die durch Pooling zusammengefassten Merkmale zurück. Das Argument `resolution` bestimmt, auf welche Größe der Klassifikator seine Eingaben zunächst skaliert. Beim Verkleinern wendet er einen Tiefpassfilter an. `train_feature_cnn` trainiert den Klassifikator mit Adam und einem Lernratenplan mit einem Zyklus. Dabei kommen zufällige horizontale Spiegelungen und auf Wunsch auch zufällige Verschiebungen zum Einsatz.

<!--en:65ea0bc6-->
## Hilfsfunktionen für das Training

<!--en:5cefc575-->
Aus Gründen der Abwärtskompatibilität behalten wir die folgenden älteren Hilfsfunktionen bei:

<!--en:6d02b03e-->
`show_heatmaps` stellt Matrizen als Raster farbiger Zellen dar, mit einem Teilbild pro Matrix. Das Argument hat die Form (Teilbildzeilen, Teilbildspalten, Höhe, Breite). Standardmäßig haben alle Teilbilder dieselbe Farbskala und einen gemeinsamen Farbbalken. Mit `cbar='each'` erhält jedes Teilbild eine eigene Skala und einen eigenen Farbbalken. Bei `cbar=False` entfällt der Farbbalken. Teilstriche markieren die ganzzahligen Zeilen- und Spaltenindizes. Mit `xticklabels` und `yticklabels` lassen sie sich beschriften, etwa mit dem jeweiligen Token. Die Funktion zeigt die Konfusionsmatrix in :numref:`sec_softmax_scratch` und die Attention-Gewichte in :numref:`chap_attention`.

<!--en:8ef60936-->
## Hilfsfunktionen für Sequenzen und Attention

<!--en:d129838e-->
### Einbettungsgradienten für kleine Vokabulare

<!--en:1ef3045b-->
Beim Berechnen des Gradienten eines Einbettungszugriffs wird der Gradient jeder Position zur Zeile des dortigen Tokens addiert. XLA führt diese Additionen mit atomaren Operationen aus. In einem Minibatch aus Text stehen an Tausenden von Positionen dieselben wenigen häufigen Token, etwa die Bytes für das Leerzeichen und für `e`. Die Additionen in derselben Zeile müssen aufeinander warten. Auf einer GPU kann die Summierung dadurch mehrere Dutzend Mal so lange dauern wie bei zufälligen Token. Bei Vokabularen mit bis zu 1024 Token berechnet `Embed` dieselbe Summe als ein einziges Matrixprodukt aus den One-Hot-Zeilen der Token und den Gradienten. Ansonsten verhält sich die Klasse wie `nnx.Embed`. Die JAX-Modelle des Buchs verwenden sie für ihre Token-Tabellen.

<!--en:d06b6a5e-->
## Verfügbarkeit in den einzelnen Frameworks

<!--en:f2d42956-->
Die älteren Hilfsfunktionen und -klassen in diesem Abschnitt (`train_ch6`, `train_seq2seq`, `predict_seq2seq`, `MaskedSoftmaxCELoss`) erhalten die Funktionalität der ursprünglichen D2L-Implementierung. Diese entstand vor der einheitlichen Klasse `d2l.Trainer`, die wir in dieser Ausgabe einführen. Für JAX stehen die älteren Hilfsfunktionen und -klassen nicht zur Verfügung. Im JAX-Reiter verwenden die betreffenden Kapitel durchgängig `d2l.Trainer.fit(model, data)`. Die Verarbeitung der einzelnen Batches übernehmen `Trainer.fit_epoch` und die zugehörigen Erweiterungen mit `@d2l.add_to_class`. Die Funktion `evaluate_accuracy` gibt es in allen vier Frameworks. PyTorch und JAX definieren `evaluate_accuracy(model, batches)` in :numref:`sec_test_models`. Die Reiter für MXNet und TensorFlow verwenden weiterhin die oben gezeigte ältere Funktion `evaluate_accuracy(net, data_iter)`.

<!--en:756f8ff0-->
[Diskussion](https://d2l.discourse.group/t/36849)

<!--en:7d8a4697-->
Hilfsfunktionen und Hilfsklassen im Paket `d2l`<br>**HyperParameters · plot und ProgressBoard · Unsicherheitsbänder · Zusammenfassung von RL-Läufen und Größenklassen für JAX-JIT · FeatureCNN · Embed mit custom_vjp**

<!--en:ac41e0bf-->
::: {.slide title="Zweck und Aufbau der d2l-Bibliothek"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ec2b78d6-->
- **Der Code ist einsehbar:** Jede Hilfsfunktion und Hilfsklasse in `d2l` (`plot`, `ProgressBoard`, `Trainer`, `FeatureCNN`, `Embed`) steht in einer ausführbaren Notebook-Zelle mit der Markierung `#@save`.

<!--en:12f8cbb3-->
- **Automatische Extraktion (`make lib`):** Erzeugt `d2l/torch.py`, `d2l/jax.py`, `d2l/tensorflow.py` und `d2l/mxnet.py` direkt aus den Quelldateien der Kapitel.

<!--en:b8f09b09-->
- **Aktuelle und ältere Hilfsfunktionen:** §{sec_utils} enthält zentrale PyTorch- und JAX-Funktionen für Diagramme, Reinforcement Learning und Bildverarbeitung sowie ältere Trainingsschleifen für MXNet und TF (`train_ch6`).

<!--en:5b76091b-->
::: {.slide title="Wie HyperParameters Konstruktorargumente automatisch ermittelt"}

<!--en:c834f204-->
[Funktionsweise der Basisklasse]{.kicker}

<!--en:ba9c4cc4-->
`d2l.Module`, `d2l.DataModule` und `d2l.Trainer` erben alle von `d2l.HyperParameters`. Dadurch können Unterklassen Konstruktorargumente als Attribute binden, ohne wiederholt Zuweisungen wie `self.x = x` zu schreiben:

<!--en:82a4415a-->
- **Inspektion des Stackframes (`inspect.currentframe().f_back`):**
  - Liest das Wörterbuch der lokalen Variablen der aufrufenden Methode `__init__` mit `inspect.getargvalues(frame)` aus.
  - Filtert `'self'`, private Namen mit dem Präfix `'_'` und alle in `ignore` aufgeführten Namen heraus.
  - Speichert das bereinigte Wörterbuch in `self.hparams` und bindet jeden Schlüssel mit `setattr(self, k, v)` als Instanzattribut.

<!--en:229da384-->
::: {.slide title="Publikationsreife Diagramme mit d2l.plot und logarithmischen Achsen"}

<!--en:d365bbe0-->
[Grundfunktionen für Diagramme]{.kicker}

<!--en:e38d1696-->
- **Einheitliche Darstellung verschiedener Kurveneingaben (`d2l.plot`):** Verwendet ein gemeinsames `X` für mehrere `Y`-Kurven. Bei `Y is None` dienen die Indizes als x-Werte. Ab 5 Kurven kombiniert die Funktion die 4 vorgegebenen Linienformate automatisch mit den Farben `C0` bis `C9`.

<!--en:8da75a69-->
- **Logarithmische Achsen mit kleinem Wertebereich (`set_log_ticks`):** Setzt Hauptteilstriche bei **$1, 2, 5$ pro Dekade**, wenn $\text{hi} < 100 \cdot \text{lo}$. So bleibt auch bei kleinem Wertebereich mehr als eine Achsenbeschriftung sichtbar.

<!--en:5a8c1cba-->
::: {.slide title="Kurven während des Trainings mit ProgressBoard.show und _plot"}

<!--en:df56dac6-->
[Messwerte beim Training]{.kicker}

<!--en:f7e294a5-->
- **Zwei Achsen für Verlust und Genauigkeit (`_plot`):** Zeichnet den Verlust auf der linken Achse (`ax`) und alle Reihen mit der Endung `'acc'` auf einer zweiten Achse rechts (`ax.twinx()`). Die gemeinsame Legende steht oben.

<!--en:b2003c72-->
- **Punktmarkierungen und Bucherstellung (`show`):** Zeichnet `marker='o'` bei Reihen mit nur 1 Punkt und bei `val_*`-Reihen. Bei `D2L_NB_CAPTURE=1` entfallen Zwischenansichten, sodass beim Erstellen des Buchs nur die letzte Ansicht gespeichert wird.

<!--en:1143ae47-->
::: {.slide title="Optimierungspfade in 2D und Raster aus Bildern oder Heatmaps" only="pytorch,jax"}

<!--en:e3a4d4a8-->
[Räumliche Darstellung]{.kicker}

<!--en:2f14ccf6-->
- **`show_trace_2d(f, results)`:** Wertet `f` auf einem automatisch erweiterten $100 \times 100$-Raster für Niveaulinien aus. Beide Achsen haben denselben Maßstab (`set_aspect('equal')`), sodass Gradienten senkrecht zu den Niveaulinien erscheinen.

<!--en:f7f0b5bd-->
- **`show_images` und `show_heatmaps`:** Wandeln Tensoren mit `d2l.numpy` um. Für 4D-Heatmap-Raster der Form `(rows, cols, H, W)` gelten eine **gemeinsame Farbskala mit `Normalize(min, max)`** und ein einziger Farbbalken (`cbar=True`).

<!--en:c6e3e1d3-->
::: {.slide title="Darstellung statistischer Unsicherheit mit plot_intervals" only="pytorch"}

<!--en:23d58a44-->
[Konfidenzintervalle als Balken und Bänder]{.kicker}

<!--en:77b9e6e4-->
- **Intervallprüfung und zwei Darstellungen (`kind`):** Prüft $\text{lo} \le \text{mean} \le \text{hi}$. Zeichnet mit `fill_between` durchscheinende Bänder (`kind='band'`) oder horizontal versetzte Fehlerbalken auf linearer oder logarithmischer Skala (`kind='errorbar'`).

<!--en:c44b361c-->
- **Referenzwerte und weitere Schätzer:** Unterstützt gestrichelte Referenzlinien mit `reference`, automatisch gedrehte Kategorienamen an der x-Achse und zusätzliche Markierungen mit `marks`.

<!--en:1b576db0-->
::: {.slide title="Diagramme für Reinforcement Learning und Auswertung mehrerer Zufallsstartwerte" only="pytorch,jax"}

<!--en:53350ec9-->
[Hilfsfunktionen für das Kapitel zum Reinforcement Learning]{.kicker}

<!--en:f17039da-->
- **Auswertung mehrerer Zufallsstartwerte (`run_seeds` $\to$ `on_grid` $\to$ `plot_curves`):** Führt $K$ Läufe mit vorgegebenen Zufallsstartwerten aus und interpoliert Episoden unterschiedlicher Länge mit kausaler Glättung auf ein gemeinsames Schrittraster. Zeigt den **Median über die Läufe** und ein Band **zwischen den Quartilen (25–75%)** oder **zwischen Minimum und Maximum**.

<!--en:6dab85d5-->
- **Verteilungen und Gitterwelten:** `plot_hist` legt Histogramme der Erträge übereinander. `show_grid` zeigt $V(s)$ als Heatmap mit Aktionspfeilen für die Greedy-Strategie $\pi(s)$.

<!--en:160241a2-->
::: {.slide title="Weniger XLA-Neukompilierungen beim Reinforcement Learning mit JAX" only="jax"}

<!--en:9626b84e-->
[JIT-Kompilierung in JAX]{.kicker}

<!--en:2b0f28ac-->
- **NumPy-Inferenz ohne Neukompilierung (`forward_np`):** Zerlegt `nnx.Module` einmal in `(graph, tree)` und liest bei jedem Schritt mit `get_raw_value()` die aktuellen Werte der Blätter aus.

<!--en:393009bd-->
- **Batchgrößen als Zweierpotenzen (`_bucket`, `padded`):** Füllt unterschiedlich lange Episoden-Batches mit Nullen auf eine der Längen $\{64, 128, 256, 512, 1024\}$ auf. Mit der binären float32-Maske `mask` genügen **höchstens 5 XLA-Kernel** für alle Batchlängen $1\dots 1{,}024$.

<!--en:0280ddcc-->
::: {.slide title="Bildbewertung anhand von Wahrnehmungsmerkmalen mit FeatureCNN auf CIFAR-10" only="pytorch,jax"}

<!--en:c4160b29-->
[Bewertung generativer Modelle]{.kicker}

<!--en:870baa3f-->
- **Geprüfter Datensatz auf der GPU (`load_cifar10_arrays`):** Prüft SHA-1-Prüfsummen, skaliert Pixelwerte auf $[-1, 1]$ und legt CIFAR-10 direkt im Bildformat des Frameworks auf der GPU ab.

<!--en:331fb100-->
- **128-dimensionale Wahrnehmungsmerkmale (`FeatureCNN`):** Skaliert die Eingaben auf $32 \times 32$ und verarbeitet sie in 3 Blöcken aus Faltung, Batchnormalisierung, ReLU und Max-Pooling ($[32, 64, 128]$). Das Training läuft über 15 Epochen mit Adam, einem Lernratenzyklus und getrennten lokalen Folgen von Pseudozufallszahlen.

<!--en:f3935fd8-->
::: {.slide title="Einbettungsgradienten für kleine Vokabulare mit eigenem Vektor-Jacobi-Produkt" only="jax"}

<!--en:7b6efe39-->
[Konkurrierende Zugriffe im GPU-Kernel]{.kicker}

<!--en:2b8f960b-->
- **Atomare Scatter-Add-Operationen als Flaschenhals:** In Byte- und Zeichenmodellen ($V \le 1{,}024$) konkurrieren Tausende von Token wie `' '` oder `'e'` bei `backward()` um atomare Zugriffe auf dieselbe Einbettungszeile. Die Gradientenberechnung dauert dadurch $10\text{--}50\times$ so lange.

<!--en:02a0f8cf-->
- **Eigenes Vektor-Jacobi-Produkt mit One-Hot-GEMM (`d2l.Embed`):** Für $V \le 1{,}024$ ersetzt eine parallele dichte Kontraktion auf Tensor Cores (`einsum`) die atomaren Additionen. Die Klasse greift auf `nnx.Embed` zurück, wenn $V > 1{,}024$.

<!--en:243bd110-->
::: {.slide title="Datencache, Sequenzmaskierung und Abwärtskompatibilität"}

<!--en:d4ed1d1e-->
[Ein-/Ausgabe und unterstützte Frameworks]{.kicker}

<!--en:bd9e0394-->
- **Atomarer Datensatz-Cache mit SHA-1 (`download_extract`):** Schreibt den Download fortlaufend in eine mit `tempfile.mkstemp` angelegte Datei und prüft die in Blöcken von 1 MB berechnete SHA-1-Prüfsumme anhand von `DATA_HUB`. Ersetzt die Zieldatei atomar mit `os.replace` und schreibt eine Markierungsdatei mit der Endung `.extracted`.

<!--en:9a0bc8f2-->
- **Maskierung und ältere Trainingsschleifen:** `sequence_mask` und `MaskedSoftmaxCELoss` setzen die Beiträge der Padding-Token auf null. Ältere Hilfsfunktionen (`train_ch6`, `train_seq2seq`) sichern die Kompatibilität mit den älteren Reitern für MXNet und TF.

<!--en:5e0c30b8-->
::: {.slide title="Verständnisfragen zur Funktionsweise der d2l-Hilfsfunktionen" only="jax"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:c15f1f46-->
Prüfen Sie Ihr Verständnis der Implementierung und der numerischen Verfahren in `d2l`:

<!--en:0c2f1323-->
1. Warum prüft die Methode `d2l.ProgressBoard.show()` die Bedingung `os.environ.get('D2L_NB_CAPTURE') == '1'` und überspringt bei `final=False` die Darstellung?

<!--en:7a4580ad-->
2. Beim Reinforcement Learning mit JAX ruft ein Agent `d2l.padded(states, actions, rewards)` für drei aufeinanderfolgende Episoden-Batches der Längen $n = 19$, $n = 47$ und $n = 58$ auf. Wie viele unterschiedliche Arrayformen entstehen durch das Padding, und warum muss die zurückgegebene Maske `mask` an die Verlustfunktion übergeben werden?

<!--en:d0f6a0a6-->
3. Warum wechselt `d2l.Embed` vom One-Hot-Matrixprodukt zurück zur üblichen Scatter-Add-Implementierung `nnx.Embed`, wenn der Umfang des Vokabulars $1{,}024$ übersteigt?

<!--en:034be31e-->
::: {.slide title="Verständnisfragen zur Funktionsweise der d2l-Hilfsfunktionen (Fortsetzung)" only="jax"}

<!--en:929b7671-2-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:80b52324-->
1. Im interaktiven Notebook aktualisiert `show(final=False)` die angezeigte SVG-Abbildung nach jeweils wenigen Minibatches. Beim automatischen Erstellen des Buchs (`D2L_NB_CAPTURE=1`) entfallen die Zwischenansichten. Dadurch muss Matplotlib nicht Hunderte Male pro Epoche zeichnen, und die gespeicherte Notebook-Ausgabe enthält nur **eine fertige Abbildung pro Board**.

<!--en:b4e1bc19-->
2. **Nur 1 Form (`size = 64`).** Da `_bucket(n) = 1 << max(6, (n - 1).bit_length())` jedes $n \in [1, 64]$ auf $64$ aufrundet, kompiliert XLA den Aktualisierungsschritt **nur einmal**. Die binäre Maske `mask` (1 für die tatsächlichen Datenzeilen $0 \dots n-1$, 0 für die Padding-Zeilen $n \dots 63$) verhindert, dass die mit Nullen aufgefüllten Zeilen den Batchmittelwert oder den TD-Verlust verfälschen.

<!--en:3fc327ed-->
3. Ein expliziter One-Hot-Tensor $\mathbf{O}$ der Form `(B, T, V)` erfordert $O(BTV)$ Speicher und $O(BTVd)$ FLOPs. Bei $V = 256\text{--}1024$ ist dieser Aufwand gering (hier konkurrieren besonders viele atomare Zugriffe). Bei $V = 32{,}000\text{--}128{,}000$ ist er untragbar (hier verteilen sich die Token-Vorkommen auch breiter über die Zeilen).

<!--en:6d3f6571-->
::: {.slide title="Verständnisfragen zur Funktionsweise der d2l-Hilfsfunktionen" except="jax"}

<!--en:929b7671-3-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:c15f1f46-2-->
Prüfen Sie Ihr Verständnis der Implementierung und der numerischen Verfahren in `d2l`:

<!--en:ea8bb0fd-->
Warum prüft die Methode `d2l.ProgressBoard.show()` die Bedingung `os.environ.get('D2L_NB_CAPTURE') == '1'` und überspringt bei `final=False` die Darstellung?

<!--en:b09bdd3e-->
**Antwort:** Im interaktiven Notebook aktualisiert `show(final=False)` die angezeigte SVG-Abbildung nach jeweils wenigen Minibatches. Beim automatischen Erstellen des Buchs (`D2L_NB_CAPTURE=1`) entfallen die Zwischenansichten. Dadurch muss Matplotlib nicht Hunderte Male pro Epoche zeichnen, und die gespeicherte Notebook-Ausgabe enthält nur **eine fertige Abbildung pro Board**.

<!--en:9bf3fd5a-->
::: {.slide title="Zusammenfassung der zentralen Hilfsfunktionen in §{num:sec_utils}"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:14d9315f-->
| Kategorie | Zentrale Bezeichner (`d2l.*`) | Funktionsweise |

<!--en:292e2397-->
| **Introspektion** | `HyperParameters.save_hyperparameters` | Untersucht den Stackframe `f_back` des Aufrufers und speichert die Argumente von `__init__` in `self.hparams` |

<!--en:e830811c-->
| **Diagramme und Achsen** | `plot`, `set_axes`, `set_log_ticks`, `ProgressBoard` | SVG-Ausgabe, logarithmische Teilstriche bei $1,2,5$ für kleine Wertebereiche, zwei Achsen für Verlust und Genauigkeit, nur die letzte Ansicht bei der Bucherstellung |

<!--en:e3cb32ee-->
| **Räumliche Darstellungen und Intervalle** | `show_trace_2d`, `show_images`, `show_heatmaps`, `plot_intervals` | Gemeinsame Farbskalierung für Heatmaps, adaptive Raster für Niveaulinien, nebeneinanderliegende Fehlerbalken und Bänder auf logarithmischer oder linearer Skala |

<!--en:7c6a2c5d-->
| **Reinforcement Learning und JIT in JAX** | `run_seeds`, `on_grid`, `plot_curves`, `show_grid`, `forward_np`, `padded` | Median und Quartilbänder über mehrere Zufallsstartwerte, Strategiepfeile in Gitterwelten, Batchgrößen als Zweierpotenzen zur Wiederverwendung kompilierten XLA-Codes |

<!--en:c6bbed7a-->
| **Bild- und Sprachverarbeitung** | `FeatureCNN`, `Embed`, `sequence_mask`, `download_extract` | Wahrnehmungsmerkmale auf CIFAR-10, One-Hot-Rückwärtsdurchlauf für Einbettungen mit `@jax.custom_vjp`, atomarer Cache mit SHA-1 |
