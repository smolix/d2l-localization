<!--en:9cff79a7-->
# Analysis und automatisches Differenzieren

<!--en:dd6404e4-->
In diesem Buch trainieren wir neuronale Netze immer nach demselben Muster. Wir berechnen den Verlust, lassen das Framework den Gradienten bestimmen und verändern jedes Gewicht um einen kleinen Schritt in Richtung geringeren Verlusts. Bisher haben wir diese Schleife verwendet, ohne zwei naheliegenden Fragen nachzugehen. Nehmen wir erstens an, wir verdoppeln die Lernrate und das Training divergiert plötzlich. Der Gradient ist weiterhin korrekt. Der Schritt führt uns jedoch aus der kleinen Umgebung heraus, in der die lokale Steigung den Verlust zuverlässig vorhersagt. Wie weit können wir dieser linearen Näherung vertrauen? Zweitens sind die Kosten für die Berechnung des Gradienten erstaunlich gering. Würden wir hundert Millionen Gewichte einzeln verändern, um jede Steigung zu schätzen, bräuchten wir hundert Millionen Vorwärtsdurchläufe. Ein einziger Rückwärtsdurchlauf liefert alle Steigungen und kostet ungefähr so viel wie wenige Vorwärtsdurchläufe. Wie ist das möglich?

<!--en:5fec5b3b-->
Beide Fragen gehören zur Analysis. Sie entstand im 17. Jahrhundert, als Newton und Leibniz fragten, wie sich stetige Änderungen messen und unendlich viele winzige Beiträge aufsummieren lassen. Ihre beiden Werkzeuge, das Differenzieren und das Integrieren, erweisen sich als Umkehroperationen. Im Deep Learning liefert die Ableitung das beste lineare Modell für die Verlustlandschaft in der Nähe der aktuellen Gewichte. Außer dem Verlustwert selbst steht dem Optimierer nur die Ableitung zur Verfügung. Das Integral kommt ins Spiel, sobald Daten oder Gewichte zufällig sind. Eine Wahrscheinlichkeit oder ein erwarteter Verlust entspricht einer Fläche unter einer Kurve. Verfahren wie der Policy-Gradient (:numref:`sec_policygradient`) müssen dieses Integral nach den Parametern genau der Verteilung ableiten, aus der sie ihre Stichprobenwerte ziehen.

<!--en:5fa04170-->
Die Grundregeln des Differenzierens haben wir bereits angegeben (:numref:`sec_calculus`). Das automatische Differenzieren haben wir als fertiges Werkzeug eingesetzt (:numref:`sec_autograd`). Für die Hauptkapitel genügt das. Dieses Kapitel im Anhang ist für alle gedacht, die verstehen möchten, warum die Verfahren funktionieren und worauf ihre Garantien beruhen. Drei Ergebnisse tragen den größten Teil der Argumentation. Erstens liefert das Abstiegslemma aus einer Schranke für die Krümmung die Regel für sichere Schrittweiten aus :numref:`sec_gd`. Zweitens lässt sich die Kettenregel als Produkt von Jacobi-Matrizen schreiben. Multiplizieren wir sie vom Verlust aus rückwärts, sinkt der Rechenaufwand erheblich. Daraus ergibt sich die Backpropagation (:numref:`sec_backprop`). Drittens gibt es zwei einander ergänzende Schätzer für den Gradienten eines Erwartungswerts. Bei der Wahl zwischen ihnen wägen wir allgemeine Anwendbarkeit und geringe Varianz gegeneinander ab.

<!--en:5d924947-->
Das Kapitel besteht aus drei Teilen. Zunächst nähern wir eine Funktion linear und quadratisch an, erst in einer Variablen (:numref:`sec_mdl-single_variable_calculus`), dann in mehreren (:numref:`sec_mdl-multivariable_calculus`). So können wir Schranken für sichere Schrittweiten bestimmen und ein Minimum von einem Sattelpunkt unterscheiden. Anschließend verfolgen wir die Kettenregel von einer Summe über Pfade in einem Berechnungsgraphen (:numref:`sec_mdl-chain-rule-graphs`) bis zu Produkten der Jacobi-Matrizen einzelner Schichten (:numref:`sec_mdl-matrix-calculus`). Daraus ergeben sich Algorithmen, die diese Matrizen automatisch multiplizieren (:numref:`sec_mdl-matrix-calculus-autodiff`). Der letzte Teil führt in die Integralrechnung ein (:numref:`sec_mdl-integral_calculus`). Dort verbinden wir Sampling und Differenzieren, um Gradienten von Erwartungswerten zu schätzen (:numref:`sec_mdl-expectation-gradients`).

<!--en:7d2d7fbb-->
Wenn Sie mit den Grundlagen der Analysis bereits vertraut sind, können Sie die einführenden Seiten zu Beginn mehrerer Abschnitte überfliegen. Aus der Vektoranalysis behandeln wir nur den Gradienten. Die Divergenz und den Satz von Stokes lassen wir ebenso aus wie die Maßtheorie. Zu Differentialgleichungen siehe :numref:`chap_mdl-dynamics`.

<!--en:fe9487c3-->
Analysis und automatisches Differenzieren – von lokalen Modellen zur Backpropagation<br>**Taylor-Näherungen · Abstiegslemma · Gradienten und Hesse-Matrizen · Berechnungsgraphen · Matrixdifferentialrechnung · automatisches Differenzieren · Gradienten von Erwartungswerten**

<!--en:e98fefaa-->
::: {.slide title="Zwei Grundfragen zu jedem Trainingsschritt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:138c191a-->
- **1. Wie weit können wir der lokalen Steigung vertrauen?** Der Gradient $\nabla\ell(\mathbf{w})$ bestimmt das lineare Modell erster Ordnung. Wie schnell der Verlust aufgrund der Krümmung davon abweicht, bestimmt, welche Lernraten $\eta$ sicher sind.

<!--en:5a2f8c69-->
- **2. Warum lässt sich der Gradient für $10^8$ Gewichte so schnell berechnen?** Alle $10^8$ Gewichte einzeln zu verändern, würde rund 12 Tage dauern. Ein Rückwärtsdurchlauf liefert alle $10^8$ exakten partiellen Ableitungen in rund 30 ms.

<!--en:8b9709e9-->
::: {.slide title="Drei Grundprinzipien der Analysis im Deep Learning"}

<!--en:445a79e0-->
[Gemeinsame Grundlagen]{.kicker}

<!--en:0896dab4-->
- **1. Abstiegslemma (§§{sec_mdl-single_variable_calculus}–§{num:sec_mdl-multivariable_calculus}):** Mit dem Lagrange-Restglied der Taylor-Näherung erhalten wir aus der Krümmungsschranke $|f''| \le L$ sichere Schrittweiten $0 < \eta < 2/L$ ($\eta^\star = 1/L$).

<!--en:809fe1e1-->
- **2. Produkte von Jacobi-Matrizen im Rückwärtsmodus (§§{sec_mdl-chain-rule-graphs}–§{num:sec_mdl-matrix-calculus-autodiff}):** Werten wir das Produkt $((\mathbf{u}^\top\mathbf{J}_K)\mathbf{J}_{K-1})\cdots\mathbf{J}_1$ von der Ausgabe aus rückwärts aus, erhalten wir alle Gradienten in $O(1)$ Durchläufen.

<!--en:3893726c-->
- **3. Erwartungswerte differenzieren (§§{sec_mdl-integral_calculus}–§{num:sec_mdl-expectation-gradients}):** Bei der Wahl zwischen Score-Funktionsschätzer (REINFORCE) und pfadweiser Reparametrisierung wägen wir allgemeine Anwendbarkeit gegen geringe Monte-Carlo-Varianz ab.

<!--en:45784cea-->
::: {.slide title="Kapitel §{num:chap_mdl-calculus} im Überblick – von eindimensionalen Schnitten zum Sampling"}

<!--en:b514d0e1-->
[Aufbau des Kapitels]{.kicker}

<!--en:e8b513fb-->
- **Teil I. Lokale Modelle (§§{sec_mdl-single_variable_calculus}–§{num:sec_mdl-multivariable_calculus}):** Taylor-Polynome, Lagrange-Restglieder, Abstiegslemma, Gradienten $\nabla f \perp$ Niveaumengen und symmetrische Hesse-Matrizen $\mathbf{H}_f$.

<!--en:acfcff02-->
- **Teil II. Kettenregel und automatisches Differenzieren (§§{sec_mdl-chain-rule-graphs}–§{num:sec_mdl-matrix-calculus-autodiff}):** Pfadsummen auf gerichteten azyklischen Graphen, Rechnen mit Differentialen $df = \langle \nabla f, d\mathbf{X}\rangle$, duale Zahlen, Wengert-Liste, $2\sqrt{T}$-Checkpointing, Hesse-Vektor-Produkte und implizites Differenzieren.

<!--en:dafbee57-->
- **Teil III. Integrale und Erwartungswerte (§§{sec_mdl-integral_calculus}–§{num:sec_mdl-expectation-gradients}):** Riemann-Summen, Hauptsatz der Differential- und Integralrechnung, Satz von Fubini, $|\det\mathbf{J}_{\boldsymbol{\phi}}|$ bei Flussmodellen, Leibniz-Regel, REINFORCE und pfadweise Gradienten.

<!--en:742191d0-->
::: {.slide title="Rechenbeispiele für die einzelnen Abschnitte"}

<!--en:a077b1cf-->
[Durchgehende Beispiele]{.kicker}

<!--en:7f1efce0-->
- **Lokale Modelle und Matrixmodelle (§§{sec_mdl-single_variable_calculus}–§{num:sec_mdl-matrix-calculus}):** Logistischer Verlust $\ell(w) = \log(1+e^{-w})$ ($L = 1/4$), Log-Sum-Exp an der Stelle $(0, \log 2)$ ($\nabla f = (\tfrac{1}{3}, \tfrac{2}{3})^\top$), gerichteter azyklischer Graph mit ganzzahliger Rechnung und 4 Eingaben ($\bar{w} = 73{,}472$) und mehrschichtiges Perzeptron mit ReLU und 2-2-1-Architektur.

<!--en:d70237ab-->
- **Automatisches Differenzieren und stochastische Analysis (§§{sec_mdl-matrix-calculus-autodiff}–§{num:sec_mdl-expectation-gradients}):** Eigene Implementierungen mit `Dual` und `Var` ($802$ gegenüber $80{,}200$ Operationen), Gauß-Integral in Polarkoordinaten $\sqrt{2\pi}$ und Varianz der Schätzer für den Gradienten von $\mathbb{E}[X^2]$ ($\sigma = 2.0$ gegenüber $7.18$).

<!--en:c74ef8f3-->
::: {.slide title="Lesehinweise zu Kapitel §{num:chap_mdl-calculus}"}

<!--en:123f19cf-->
[Lernleitfaden]{.kicker}

<!--en:8c69f975-->
- **Wenn Sie die Grundlagen der Analysis kennen:** Überfliegen Sie die Ableitungstabelle in §{sec_mdl-single_variable_calculus} und die Wiederholung zu Riemann-Summen und Stammfunktionen in §{sec_mdl-integral_calculus}. Gehen Sie direkt zu **Mittelwertsatz, Lagrange-Restglied und Abstiegslemma** (§{sec_mdl-single_variable_calculus}) sowie zur **Variablentransformation im $\mathbb{R}^n$** (§{sec_mdl-integral_calculus}).

<!--en:deb5217b-->
- **Für Theorie und Herleitungen im Deep Learning:** Machen Sie sich mit dem **Rechnen mit Differentialen** $df = \langle \nabla f, d\mathbf{X}\rangle$ und dem zyklischen Vertauschen von Faktoren unter der Spur in §{sec_mdl-matrix-calculus} vertraut. Damit erübrigen sich Tabellen mit hundert Matrixableitungen, und Sie vermeiden bei quadratischen Matrizen unbemerkte Fehler beim Transponieren.

<!--en:8f5ffabd-->
- **Für Systeme und modernes maschinelles Lernen:** In **§{sec_mdl-matrix-calculus-autodiff} (Automatisches Differenzieren)** erfahren Sie, warum der Rückwärtsmodus etwa so viel kostet wie 3 Vorwärtsdurchläufe. Sie sehen, wie Aktivierungs-Checkpointing den Speicherbedarf von $O(T)$ auf $2\sqrt{T}$ senkt und wie der Vorwärtsmodus über dem Rückwärtsmodus $\mathbf{H}\mathbf{v}$ ohne eine 4 TB große Hesse-Matrix berechnet. **§{sec_mdl-expectation-gradients}** behandelt die Gradientenvarianz bei Monte-Carlo-Schätzungen.

<!--en:e680ad79-->
**Als Nächstes (§{sec_mdl-single_variable_calculus} Analysis in einer Variablen):** Wie wir mit ersten und zweiten Ableitungen lokale lineare und quadratische Modelle aufstellen, den Näherungsfehler begrenzen und sichere Schrittweiten für den Gradientenabstieg nachweisen.
