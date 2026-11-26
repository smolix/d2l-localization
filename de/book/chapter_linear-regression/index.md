<!--en:a44c40a3-->
# Lineare Regression

<!--en:6c9e1436-->
*Regression* ist das Standardwerkzeug der quantitativen Vorhersage, wenn die Zielgröße ein aussagekräftiger Zahlenwert ist und keine diskrete Kategorie. So können wir den Verkaufspreis eines Hauses anhand von Wohnfläche und Alter schätzen, aus Vitalzeichen auf die Dauer eines Krankenhausaufenthalts schließen oder den Strombedarf im nächsten Quartal vorhersagen. Regression eignet sich auch hervorragend, um die Verfahren des Deep Learning zu erproben. Fast alles, was wir zum Training neuronaler Netze brauchen, finden wir in der *linearen Regression* bereits in seiner einfachsten und anschaulichsten Form: parametrisierte Vorwärtsdurchläufe (engl. forward passes), Verlustfunktionen (engl. loss functions) auf Grundlage des Maximum-Likelihood-Prinzips, stochastischer Gradientenabstieg mit Minibatches, wiederverwendbarer Trainingscode, Verfahren zur Beurteilung der Generalisierung und Regularisierung über die Norm der Gewichte.

<!--en:6749e303-->
In :numref:`sec_linear_regression` formulieren wir zunächst das affine Modell und den quadratischen Verlust. Wir lösen das Problem in geschlossener Form durch orthogonale Projektion und iterativ durch stochastischen Gradientenabstieg mit Minibatches (engl. stochastic gradient descent, SGD). Außerdem leiten wir den Zusammenhang zwischen quadratischem Verlust und Maximum-Likelihood-Schätzung bei gaußschem Rauschen her. Bevor wir mit realen Daten arbeiten, erstellen wir in :numref:`sec_synthetic-regression-data` einen synthetischen Benchmark mit bekannten wahren Parametern $(\mathbf{w}^*, b^*)$ und bekanntem Rauschpegel $\sigma$. Damit haben wir einen genauen Vergleichswert für unseren Code. In den nächsten drei Abschnitten bauen wir das Trainingsverfahren schrittweise auf: zunächst allein mit Tensoren und automatischem Differenzieren (engl. automatic differentiation, :numref:`sec_linear_scratch`), dann mit den wiederverwendbaren Klassen `Module`, `DataModule` und `Trainer`, die wir im restlichen Buch einsetzen (:numref:`sec_oo-design`). Schließlich genügen mit den vorgefertigten Schichten und Optimierern des Frameworks wenige Zeilen Code (:numref:`sec_linear_concise`).

<!--en:210521c9-->
Mit der Anpassung an die Trainingsdaten ist erst die Hälfte der Aufgabe gelöst. Am Beispiel der Kurvenanpassung mit Polynomen unterscheiden wir in :numref:`sec_generalization_basics` zwischen Trainingsfehler und Generalisierungsfehler. Wir zerlegen den erwarteten Testfehler in quadrierte Verzerrung (engl. squared bias), Varianz und Rauschen. Wir zeigen, wie wir mit getrennten Validierungsdaten und $K$-facher Kreuzvalidierung die Modellkomplexität wählen, ohne die Testdaten einzusehen. In :numref:`sec_weight_decay` begrenzen wir schließlich die Überanpassung (engl. overfitting) in hohen Dimensionen durch einen Strafterm für die quadrierte $\ell_2$-Norm. Dabei zeigt sich, wie der Gewichtszerfall (engl. weight decay) gezielt die Komponenten in jenen Richtungen schrumpfen lässt, die durch die Trainingseingaben nur ungenau bestimmt sind.

<!--en:19c19f7b-->
Lineare neuronale Netze für die Regression<br>
**affine Modelle · quadratischer Verlust · Normalgleichungen · Minibatch-SGD · Generalisierung · Gewichtszerfall**

<!--en:a5fce629-->
::: {.slide title="Lineare Regression: Deep Learning im Kleinen"}

<!--en:678b9332-->
[Ein Modell zum Erproben]{.kicker}

<!--en:2a45b287-->
- **Parametrisierter Vorwärtsdurchlauf und Verlust:** Vorhersage mit $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$, Bewertung der Abweichung mit $L(\mathbf{w}, b) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|_2^2$. Grundlage ist das Maximum-Likelihood-Prinzip bei gaußschem Rauschen.

<!--en:ade204c0-->
- **Exakte Lösung und skalierbare Iteration:** Die Normalgleichungen $\mathbf{X}^\top\mathbf{X}\mathbf{w}^* = \mathbf{X}^\top\mathbf{y}$ liefern die orthogonale Projektion von $\mathbf{y}$ in $O(nd^2 + d^3)$. Minibatch-SGD braucht $O(|\mathcal{B}|d)$ pro Schritt.

<!--en:03fa90b0-->
- **Zwei typische Fehler:** Trotz eines unbemerkten Formfehlers ($(32,) - (32, 1) \to 32 \times 32$) sinkt der Verlust. Ein Polynom hohen Grades, das 20 verrauschte Punkte interpoliert, versagt bei neuen Daten.

<!--en:536c67b9-->
::: {.slide title="Von den Grundprinzipien zum wiederverwendbaren Trainingsverfahren"}

<!--en:c7d201c1-->
[Aufbau der Implementierung]{.kicker}

<!--en:4004a8f2-->
- **Theorie und synthetische Referenz:** Affines Modell, quadratischen Verlust, Normalgleichungen und Minibatch-SGD formulieren. Anschließend mit den wahren Parametern $(\mathbf{w}^*, b^*)$ und der rauschbedingten Verlustuntergrenze $\sigma^2/2$ vergleichen.

<!--en:ecdd1c6c-->
- **Drei Implementierungsstufen:** Trainingsschleife mit Tensoren und Autograd selbst schreiben, in `Module` / `DataModule` / `Trainer` aufteilen und die vollständig verbundenen Schichten und Optimierer des Frameworks einsetzen.

<!--en:6686eb27-->
- **Generalisierung und Gewichtszerfall:** Unteranpassung und Überanpassung anhand von Verzerrung, Varianz und $K$-facher Kreuzvalidierung erkennen. Rauschen bei Kollinearität durch $\ell_2$-Ridge-Schrumpfung dämpfen.

<!--en:5b6075a6-->
::: {.slide title="Durchgehende Vergleichsbeispiele des Kapitels"}

<!--en:418d9f93-->
[Vergleichswerte]{.kicker}

<!--en:acad204f-->
- **Projektion und synthetischer Test:** Die Geradenanpassung an 3 Punkte $(0,1),(1,1),(2,0)$ ergibt $\hat{\mathbf{w}} = (7/6, -1/2)^\top$ mit $\mathbf{X}^\top\mathbf{r} = \mathbf{0}$. Bei $n = 1{,}000$ Beispielen mit $\mathbf{w}^* = (2, -3.4)^\top, b^* = 4.2, \sigma = 0.01$ fällt der Verlust von $16.6$ auf $\sigma^2/2 = 5 \times 10^{-5}$.

<!--en:f8c01af4-->
- **Kurvenanpassung mit Polynomen ($n = 20$):** Verrauschte Werte einer kubischen Funktion ($\sigma = 0.1$). Grad $1$ führt zur Unteranpassung, Grad $3$ erreicht $\sigma^2 = 0.01$, bei Grad $19$ explodiert der Fehler.

<!--en:7d373300-->
- **Hochdimensionale Ridge-Regression ($d = 200, n = 300$):** Der Strafterm aus der A-priori-Verteilung mit $\lambda = \sigma^2/(n\tau^2) = 1/3$ reduziert die effektive Parameterzahl auf $\operatorname{df} \approx 127$.

<!--en:5ce72ec9-->
::: {.slide title="Die vollständige Trainingsschleife beim überwachten Lernen"}

<!--en:33b69155-->
[Gesamtüberblick]{.kicker}

<!--en:061cdd10-->
Vom mehrschichtigen Perzeptron bis zum Transformer verwendet jedes weitere Modell in diesem Buch genau das Trainingsverfahren aus diesem Kapitel:

<!--en:9797c31e-->
- **Vorwärtsdurchlauf und Verlust (`Module`):** Minibatch-Eingaben $\mathbf{X} \in \mathbb{R}^{|\mathcal{B}| \times d}$ auf Vorhersagen $\hat{\mathbf{y}}$ abbilden und den empirischen Verlust mit Strafterm $L_{\mathcal{B}}(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$ auswerten.

<!--en:b3aa3992-->
- **Rückwärtsdurchlauf und Aktualisierung (`Trainer`):** $L_{\mathcal{B}}$ mit Autograd im Rückwärtsmodus differenzieren und die Parameter mit $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \nabla L_{\mathcal{B}}$ aktualisieren.

<!--en:40a2f71b-->
- **Validierung und Auswahl (`DataModule`):** Nach jeder Epoche das Risiko auf zurückgehaltenen Daten verfolgen und damit die Hyperparameter $(\eta, |\mathcal{B}|, \lambda)$ anpassen, ohne dass Testdaten einfließen.

<!--en:462821b0-->
Als neuronales Netz betrachtet besteht die lineare Regression aus einer einzigen vollständig verbundenen Schicht mit $d$ Eingaben und $1$ linearem Ausgabeneuron.
