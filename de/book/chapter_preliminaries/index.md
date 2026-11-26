<!--en:99f029dc-->
# Handwerkszeug

<!--en:1c878d5c-->
Für den Einstieg in Deep Learning brauchen Sie einige grundlegende Fertigkeiten. Dazu gehören (i) Verfahren, um mehrdimensionale Daten im Speicher abzulegen und zu verarbeiten (:numref:`sec_ndarray`), und (ii) Werkzeuge, um tabellarische Rohdaten aus Dateien einzulesen und zu bereinigen, auch wenn Einträge fehlen oder Textfelder Kategorien enthalten (:numref:`sec_pandas`). Hinzu kommen (iii) Sicherheit im Umgang mit Operationen der linearen Algebra, die hochdimensionale Darstellungen transformieren und deren Größe messen (:numref:`sec_linear-algebra`), und (iv) Differentialrechnung, um zu bestimmen, in welche Richtung wir jeden Parameter ändern müssen, damit die Verlustfunktion (engl. loss function) sinkt (:numref:`sec_calculus`). Wir brauchen außerdem (v) automatisches Differenzieren (engl. automatic differentiation), damit das Framework die Kettenregel über Millionen von Parametern hinweg für uns anwendet (:numref:`sec_autograd`). Schließlich benötigen wir (vi) Grundkenntnisse in Wahrscheinlichkeitsrechnung und Statistik. Mit ihrer Hilfe ziehen wir Schlüsse unter Unsicherheit und unterscheiden ein echtes Signal von zufälligen Stichprobenschwankungen (:numref:`sec_prob`).

<!--en:b260a88f-->
Was Deep Learning der klassischen linearen Algebra, der Analysis und der Wahrscheinlichkeitsrechnung hinzufügt, ist der *Maßstab*: Wir wenden sie auf Millionen oder Milliarden von Variablen an, für die eine Rechnung von Hand ausscheidet. Das ermöglichen zwei Softwarebausteine, die alle folgenden Kapitel voraussetzen. *Tensorbibliotheken* wenden kompilierte Operationen parallel auf ganze Zahlenfelder auf CPUs und GPUs an. *Automatisches Differenzieren* berechnet die Ableitungen beliebiger numerischer Programme mechanisch.

<!--en:f80eaf99-->
Dieses Kapitel führt Sie anhand praktischer Beispiele im Schnelldurchgang in diese Voraussetzungen ein. Wer mit NumPy, linearer Algebra und der Analysis mehrerer Variablen vertraut ist, kann die bekannten Teile überfliegen. Wir empfehlen dennoch :numref:`sec_autograd` zum automatischen Differenzieren und :numref:`sec_prob` zu Standardfehlern und statistischer Schätzung. Die späteren Kapitel greifen immer wieder auf beides zurück. Eine theoretische Vertiefung mit vollständigen Beweisen und Herleitungen bietet der Anhang *Mathematik für Deep Learning*. Er behandelt lineare Algebra (:numref:`chap_mdl-linear-algebra`), Analysis (:numref:`chap_mdl-calculus`) sowie Wahrscheinlichkeitsrechnung und Statistik (:numref:`chap_mdl-probability-statistics`) ausführlich.

<!--en:f95cd064-->
Mathematische Grundlagen und Softwarebausteine des Deep Learning<br>**Tensoren · Datenvorverarbeitung · lineare Algebra · Analysis · automatisches Differenzieren · Wahrscheinlichkeit**

<!--en:6cf1b367-->
::: {.slide title="Warum Deep Learning zwei neue Softwarebausteine braucht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ae10ee56-->
- **Skalarer Code stößt an Grenzen:** Ein Batch mit 256 Bildern aus Fashion-MNIST enthält $200{,}704$ Pixel. Grundlagenmodelle aktualisieren $10^9\text{--}10^{12}$ Gewichte pro Schritt.

<!--en:a3dfb076-->
- **Baustein 1, Tensor-Kernel für zusammenhängende Daten:** Daten ohne Objekthüllen in RAM-/VRAM-Puffern ablegen und ganze Batches parallel auf SIMD-/GPU-Recheneinheiten verarbeiten.

<!--en:4f0d8e66-->
- **Baustein 2, Autograd im Rückwärtsmodus:** Den Ausführungsgraphen (DAG) aufzeichnen und alle Parametergradienten $\nabla_{\mathbf{w}} L$ in einem Rückwärtsdurchlauf berechnen.

<!--en:b7e16f22-->
- **Statistische Grundlage:** Unsicherheit auf dem Testdatensatz begrenzen ($\sigma/\sqrt{n}$) und Einschätzungen mit dem Satz von Bayes aktualisieren.

<!--en:0b98ef17-->
::: {.slide title="Ablauf einer vollständigen Iteration beim maschinellen Lernen"}

<!--en:23989a1a-->
[Die Lernschleife]{.kicker}

<!--en:9a266d27-->
- **Von Daten zu Tensoren:** Rohtabellen einlesen, Lücken ohne Datenleck ergänzen und Batches in zusammenhängenden Puffern vom Typ `float32` / `bfloat16` ablegen.

<!--en:f26dd743-->
- **Vorwärtsdurchlauf:** Eingaben durch Matrixprodukte für ganze Batches auf Klassenlogits abbilden: $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$.

<!--en:23471840-->
- **Verlust und Rückwärtsdurchlauf:** Skalaren Verlust $L$ berechnen und Vektor-Jacobi-Produkte rückwärts weitergeben, um $\nabla_{\mathbf{W}} L$ zu bestimmen.

<!--en:5ce4354d-->
- **Schritt und statistische Auswertung:** Parameter aktualisieren und Fehlerbalken für endliche Stichproben bestimmen.

<!--en:1505df49-->
::: {.slide title="Durchgehende Beispiele der einzelnen Abschnitte"}

<!--en:05adf916-->
[Roter Faden]{.kicker}

<!--en:a48d1fdf-->
- **Tensoren und lineare Algebra (Batch `images`):** Den Fashion-MNIST-Minibatch `(256, 1, 28, 28)` zu $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ umformen und mit $\mathbf{W} \in \mathbb{R}^{784 \times 10}$ multiplizieren.

<!--en:e733e01c-->
- **Datenvorverarbeitung (`house_tiny.csv`):** 10 Häuser mit fehlenden Zimmerangaben, Dachtypen als Kategorien und Größenordnungen, die sich um den Faktor $500\times$ unterscheiden.

<!--en:ab706a3d-->
- **Analysis und Autograd (Geradenanpassung und Kontrollfluss):** Den Verlust $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$ für 3 Punkte minimieren und durch Python-Schleifen differenzieren.

<!--en:3f8a21a9-->
- **Wahrscheinlichkeit (Münzwürfe und Früherkennung):** Konfidenzbänder der Größenordnung $1/\sqrt{n}$ und zweistufige bayessche medizinische Diagnostik.

<!--en:caba10a0-->
::: {.slide title="Orientierung in diesem Kapitel und im mathematischen Anhang"}

<!--en:123f19cf-->
[Lesehinweise]{.kicker}

<!--en:9b3735d4-->
- **Praktischer Schwerpunkt:** Zu jedem Konzept gehört ausführbarer Framework-Code für Formen, Datentypen, gemeinsame Speicherbereiche und Gradienten.

<!--en:094e4495-->
- **Besonders wichtige Abschnitte:** Auch mit Vorkenntnissen in NumPy und Analysis lohnt sich **der Autograd-Abschnitt** zur Semantik von Ausführungsgraphen (`detach`, Risiken bei Änderungen am selben Speicherort) und **der Abschnitt zur Wahrscheinlichkeit** zu Standardfehlern endlicher Stichproben.

<!--en:168f56e1-->
- **Vertiefende Beweise:** **Kapitel §{num:chap_mdl-linear-algebra}, §{num:chap_mdl-calculus} und §{num:chap_mdl-probability-statistics} (mathematischer Anhang)** behandeln die Singulärwertzerlegung, Rechenregeln der Matrixanalysis und die Asymptotik von Maximum-Likelihood-Schätzern.
