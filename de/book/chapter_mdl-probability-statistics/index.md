<!--en:42325983-->
# Wahrscheinlichkeitstheorie und statistisches Lernen

<!--en:cb182177-->
Im Haupttext trainieren wir Regressionsmodelle mit dem quadratischen Fehler und Klassifikatoren mit der Kreuzentropie (:numref:`sec_linear_regression`, :numref:`sec_softmax`). Wir geben die Genauigkeit auf endlichen Testdatensätzen an und wählen Modelle anhand ihrer Validierungsergebnisse aus (:numref:`sec_generalization_basics`). Bei jedem dieser Schritte ziehen wir Schlüsse aus einer endlichen Stichprobe. Betrachten wir eine Münze, die $9$-mal bei $13$ Würfen Kopf zeigt. Die relative Häufigkeit von Kopf beträgt $9/13 \approx 0.69$. Doch selbst bei einer fairen Münze fällt in mehr als einem Achtel solcher Versuchsreihen mindestens neunmal Kopf. Die Genauigkeit auf dem Testdatensatz verhält sich genauso, als würfen wir für jedes Testbeispiel einmal diese Münze. In diesem Kapitel erarbeiten wir die mathematischen Grundlagen dafür, wann endliche Stichproben verlässliche Schlüsse erlauben und wann sie uns in die Irre führen.

<!--en:529dfcd4-->
Um die Unsicherheit zu verstehen, die mit einer Stichprobe einhergeht, brauchen wir vier Begriffe. Eine *Verteilung* beschreibt, wie die Ergebnisse schwanken. Ein *Schätzer* (engl. estimator) ordnet einer Stichprobe eine Zahl zu. Wie diese Zahl über hypothetische Stichproben hinweg streut, ist eine andere Frage als die nach der *A-posteriori-Verteilung* (engl. posterior), die auf die eine vorliegende Stichprobe bedingt ist. Eine *Generalisierungsschranke* (engl. generalization bound) gibt an, wann ein kleiner Fehler auf einer endlichen Stichprobe auch einen kleinen Fehler auf neuen Daten garantiert.

<!--en:1003eeff-->
Wir behandeln diese vier Begriffe der Reihe nach. Zunächst untersuchen wir Dichten stetiger Zufallsvariablen (:numref:`sec_mdl-random_variables`), gemeinsame Verteilungen und die Transformation von Dichten mithilfe der Jacobi-Determinante (:numref:`sec_mdl-joint-distributions`). Danach betrachten wir die gängigen Verteilungen (:numref:`sec_mdl-distributions`) und ihre einheitliche Darstellung in der Exponentialfamilie (:numref:`sec_mdl-exponential-family`). Diese Darstellung erklärt, warum der Gradient bei linearer, logistischer und Softmax-Regression dieselbe Form hat: „Vorhersage minus Beobachtung“. In :numref:`sec_mdl-statistics` behandeln wir Fehlerbalken, Hypothesentests, Konfidenzintervalle und das Bootstrap-Verfahren. In :numref:`sec_mdl-maximum_likelihood` leiten wir unsere Trainingsverluste aus dem Maximum-Likelihood-Prinzip ab und verknüpfen die Varianz der Parameterschätzer mit der durch die Fisher-Information beschriebenen Krümmung. Latente Variablen führen uns zur unteren Evidenzschranke (engl. evidence lower bound, ELBO) und zum Expectation-Maximization-Algorithmus (:numref:`sec_mdl-latent-em-elbo`). Für A-posteriori-Verteilungen ohne geschlossene Darstellung betrachten wir die Laplace-Approximation, Monte-Carlo-Verfahren, Markov-Chain-Monte-Carlo und variationelle Näherungen (:numref:`sec_mdl-bayesian-computation`). In :numref:`sec_mdl-concentration-generalization` beweisen wir für endliche Stichproben Schranken für die Wahrscheinlichkeit großer Abweichungen und zeigen, warum sich die Wahrscheinlichkeitsmasse in hohen Dimensionen auf einer dünnen Kugelschale konzentriert. In :numref:`sec_mdl-generalization-bounds` erweitern wir diese Ergebnisse auf die gleichmäßige Konvergenz (engl. uniform convergence) nach der Modellauswahl (:numref:`chap_classification_generalization`). In :numref:`sec_mdl-naive_bayes` wenden wir Verteilungen, Glättung und Kalibrierung auf die Bildklassifikation an.

<!--en:97b2c763-->
Wenn Sie mit :numref:`sec_prob` vertraut sind, können Sie die einleitenden Rechenregeln für Momente überfliegen. :numref:`sec_mdl-bayesian-computation` und :numref:`sec_mdl-naive_bayes` können Sie beim ersten Lesen überspringen. Die späteren Abschnitte setzen beide nicht voraus. Die Theorie der Entropie und der Divergenzen vertiefen wir in :numref:`chap_mdl-information-theory`. Einen Überblick über Maßtheorie, Zeitreihen und nichtparametrische Erweiterungen gibt :numref:`sec_mdl-probability-statistics-further-reading`.

<!--en:cc62f985-->
Wahrscheinlichkeitstheorie und statistisches Lernen: Grundlagen des Schließens unter Unsicherheit<br>**Dichten und Jacobi-Matrizen · Exponentialfamilien · Schätzen und Testen · MLE und ELBO · bayessche Rechenverfahren · Konzentration und Generalisierung**

<!--en:08c18ba1-->
::: {.slide title="Warum Deep Learning statistische Grundlagen braucht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:8d7c7f89-->
- **Unser Münzbeispiel ($9$-mal Kopf bei $13$ Würfen):** $\hat{\theta} = 9/13 \approx 0.692$. Doch auch bei einer fairen Münze ($\theta = 0.5$) fällt in $\ge 9$ von 13 Würfen Kopf, und zwar in $13.3\%$ der Versuchsreihen. Ebenso schwankt die Genauigkeit auf einem endlichen, für den Test zurückgehaltenen Datensatz.

<!--en:94bc2259-->
- **Zwei Richtungen des Schließens:** Die *Wahrscheinlichkeitstheorie* (§{sec_mdl-random_variables}–§{num:sec_mdl-exponential-family}) leitet aus einem bekannten Modell Häufigkeiten für die möglichen Ergebnisse ab. Die *Statistik* (§{sec_mdl-statistics}–§{num:sec_mdl-bayesian-computation}) schließt umgekehrt aus endlich vielen Beobachtungen auf Schätzwerte, Fehlerbalken und A-posteriori-Verteilungen.

<!--en:f4b9d48f-->
- **Garantien nach der Modellauswahl (§{sec_mdl-concentration-generalization}–§{num:sec_mdl-generalization-bounds}):** Wer aus mehreren Validierungsläufen das beste Modell auswählt, testet implizit mehrere Hypothesen und braucht Schranken für die gleichmäßige Konvergenz.

<!--en:2246f51c-->
::: {.slide title="Vier Grundlagen des probabilistischen Lernens"}

<!--en:00be04d1-->
[Kapitelaufbau]{.kicker}

<!--en:499c9244-->
- **1. Verteilungen und ihre einheitliche Darstellung (§{sec_mdl-random_variables}–§{num:sec_mdl-exponential-family}):** Dichten, Jacobi-Matrizen, gängige Verteilungen und die Exponentialfamilie $h(\mathbf{x})e^{\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})}$.

<!--en:ab9f460e-->
- **2. Schätzung, Tests und Likelihood (§{sec_mdl-statistics}–§{num:sec_mdl-maximum_likelihood}):** Zielkonflikt zwischen Verzerrung und Varianz, $p$-Werte, Konfidenzintervalle, Maximum-Likelihood-Schätzung als KL-Projektion und Fisher-Information.

<!--en:f71e27f4-->
- **3. Modelle mit latenten Variablen und A-posteriori-Verteilungen (§{sec_mdl-latent-em-elbo}–§{num:sec_mdl-bayesian-computation}):** logarithmierte marginale Likelihood $\log p(\mathbf{x})$, Jensen-Ungleichung, ELBO, EM, Laplace-Approximation, variationelle Inferenz (VI) und MCMC.

<!--en:374da574-->
- **4. Konzentration und Generalisierung (§{sec_mdl-concentration-generalization}–§{num:sec_mdl-naive_bayes}):** Chernoff- und Hoeffding-Schranken, Geometrie von Kugelschalen bei großer Dimension $d$, Rademacher-Komplexität und Naive Bayes.

<!--en:4311b0a9-->
::: {.slide title="Von Verteilungen zu A-posteriori-Verteilungen (§{sec_mdl-random_variables}–§{num:sec_mdl-bayesian-computation})"}

<!--en:d623abd2-->
[Abschnittsübersicht (Teil I)]{.kicker}

<!--en:e65eaf49-->
- **§{sec_mdl-random_variables}–§{num:sec_mdl-joint-distributions} (Rechnen mit stetigen Zufallsvariablen):** $F'(x) = p(x)$, Inversionsmethode $X = F^{-1}(U)$, Satz von der totalen Varianz und Volumenfaktor $|\det J_g|^{-1}$ aus der Jacobi-Determinante.

<!--en:3b5ddc86-->
- **§{sec_mdl-distributions}–§{num:sec_mdl-exponential-family} (Verteilungen und Exponentialfamilie):** Poisson-Grenzwertsatz für die Binomialverteilung, zentraler Grenzwertsatz, bedingte Gauß-Verteilungen über das Schur-Komplement, Ableitungen der Kumulantenfunktion $\nabla A = \mathbb{E}[T]$ und Aktualisierung konjugierter Verteilungen.

<!--en:baaec0ef-->
- **§{sec_mdl-statistics}–§{num:sec_mdl-bayesian-computation} (Inferenz und latente Variablen):** $\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$, gepaarte Tests, Bootstrap-Verfahren, Krümmung gemäß der Fisher-Information $nI(\theta)$, EM über die ELBO und Näherungen der A-posteriori-Verteilung mit MCMC und VI.

<!--en:a3fcca6b-->
::: {.slide title="Garantien und Anwendung (§{sec_mdl-concentration-generalization}–§{num:sec_mdl-naive_bayes})"}

<!--en:1bfcc6b1-->
[Abschnittsübersicht (Teil II)]{.kicker}

<!--en:f75f0a61-->
- **§{sec_mdl-concentration-generalization} Konzentrationsungleichungen:** Mit den Verfahren von Chernoff und Hoeffding verbessern wir die Tschebyscheff-Schranke $1/(n\varepsilon^2)$ auf einen exponentiellen Abfall $\exp(-2n\varepsilon^2)$. Wir zeigen, warum sich die Masse einer Gauß-Verteilung bei großer Dimension $d$ auf einer dünnen Kugelschale mit Radius $\sqrt{d}$ konzentriert.

<!--en:0a4c09c1-->
- **§{sec_mdl-generalization-bounds} Generalisierungsschranken:** Bei der Modellsuche untersuchen wir Schranken für den größtmöglichen Generalisierungsfehler $\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)|$ mithilfe der booleschen Ungleichung, der Rademacher-Komplexität und der Theorie der gutartigen Überanpassung.

<!--en:72e02ec4-->
- **§{sec_mdl-naive_bayes} Naive-Bayes-Klassifikator:** Aus Likelihoods der kategorialen Verteilung oder der Bernoulli-Verteilung und der Laplace-Glättung durch Addition von eins entsteht ein vollständiger generativer Bildklassifikator.

<!--en:c8eafc5f-->
::: {.slide title="Lesewege durch Kapitel §{num:chap_mdl-probability-statistics}"}

<!--en:123f19cf-->
[Lesehinweise]{.kicker}

<!--en:988773de-->
- **Theoretische Grundlagen:** Lesen Sie **§{sec_mdl-random_variables}–§{num:sec_mdl-maximum_likelihood}** der Reihe nach. Dort sehen Sie, wie sich alle Verlustfunktionen des überwachten Lernens und alle statistischen Fehlerbalken im Haupttext durch stetige Dichten und die Exponentialfamilie einheitlich beschreiben lassen.

<!--en:6ec8a3b2-->
- **Für Leser mit Vorkenntnissen:** Wenn Sie mit §{sec_prob} (Grundlagen der diskreten Wahrscheinlichkeitstheorie) vertraut sind, überfliegen Sie die Rechenregeln für Momente in §{sec_mdl-random_variables}–§{num:sec_mdl-joint-distributions}. Konzentrieren Sie sich auf die Inversionsmethode, den Satz von der totalen Varianz, Jacobi-Determinanten und bedingte Gauß-Verteilungen.

<!--en:9024af89-->
- **Weiterführende Themen:** **§{sec_mdl-bayesian-computation} (Bayessche Rechenverfahren)** und **§{sec_mdl-naive_bayes} (Naive Bayes)** sind in sich abgeschlossen. Sie können beide beim ersten Lesen überspringen, da spätere Abschnitte sie nicht voraussetzen.

<!--en:d64cd66b-->
- **Verbindungen zu anderen mathematischen Kapiteln:** Spektren und positive Definitheit kennen wir aus der linearen Algebra in **Kapitel §{num:chap_mdl-linear-algebra}**, Jacobi-Matrizen für mehrere Variablen und Konvexität aus **Kapitel §{num:chap_mdl-calculus}–§{num:chap_mdl-optimization}**. Die Theorie der Entropie und der KL-Divergenz vertiefen wir in **Kapitel §{num:chap_mdl-information-theory}**.

<!--en:c9e005f9-->
**Als Nächstes (§{sec_mdl-random_variables} Zufallsvariablen):** Warum haben einzelne Werte bei stetigen Zufallsvariablen die Wahrscheinlichkeit null? Wie hängen Dichte und Verteilungsfunktion über $F'(x) = p(x)$ zusammen? Warum werden Stichprobenmittelwerte bei schweren Verteilungsrändern unzuverlässig?
