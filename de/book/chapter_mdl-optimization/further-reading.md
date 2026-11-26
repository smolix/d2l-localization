<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:02430e52-->
Dieser Abschnitt führt über das Kapitel hinaus. Nicht alle Themen sind gleich dringlich. Fünf betreffen unmittelbar das Training neuronaler Netze und stehen deshalb in den jeweiligen Themenbereichen an erster Stelle. Es sind der Rand der Stabilität (engl. edge of stability), die Mittelung der Iterierten (engl. iterate averaging), die Interpolation, die Fisher-Matrix und die Sattelpunktverfahren. Die übrigen Themen stammen aus der klassischen Optimierung und der numerischen Mathematik. Den Abschluss bilden allgemeine Lehrbücher, Kurse und Übersichtsarbeiten. Einen Überblick über die Praxis des Trainings gibt :numref:`sec_optimization-further-reading`. Da kein anderer Teil des Buchs auf diesem Abschnitt aufbaut, können Sie sich die Themen aussuchen, die Sie interessieren.

<!--en:7edc1d9b-->
## Gradientenverfahren

<!--en:192ccab0-->
### Selbststabilisierung am Rand der Stabilität

<!--en:15ea91fa-->
Beim Training eines neuronalen Netzes mit Gradientenabstieg auf dem gesamten Trainingssatz geschieht etwas Überraschendes. Die Schärfe (engl. sharpness) des Verlusts (sein größter Hesse-Eigenwert) steigt bis zur Stabilitätsgrenze $2/\eta$ und schwankt dann um diesen Wert, statt zu divergieren :cite:`Cohen.Kaur.Li.ea.2021`. Das quadratische Modell in :numref:`subsec_mdl-edge-of-stability` erklärt den ersten Teil dieses Verhaltens. Sobald die Schärfe die Grenze $2/\eta$ überschreitet, wächst die Komponente in Richtung des führenden Eigenvektors. Ein rein quadratisches Modell sagt jedoch unbegrenztes Wachstum voraus. Was die Schärfe wieder sinken lässt, erklärt es nicht. :citet:`Damian.Nichani.Lee.2023` lösen das Rätsel, indem sie einen weiteren Term der Taylor-Entwicklung berücksichtigen, den kubischen Term in Richtung des führenden Eigenvektors. Sobald sich die Schwingungen in dieser Richtung aufbauen, verschiebt die kubische Kopplung die Iterierte entgegen dem Gradienten der Schärfe selbst. Dadurch sinkt die Schärfe wieder unter $2/\eta$. Die Schwingung klingt ab, bis der Zyklus von Neuem beginnt. Ohne den kubischen Term würde die Schwingung ungebremst wachsen.

<!--en:949ee687-->
### Optimale Polynome und konjugierte Gradienten

<!--en:17086891-->
Betrachten wir die Minimierung einer quadratischen Funktion $f = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$. Die Eigenwerte der Matrix liegen in $[\mu, L]$, der Minimierer ist $\mathbf{0}$. Bleibt bei einem Verfahren erster Ordnung der $k$-te Schritt im Spann der bisherigen Gradienten, so hat die Iterierte die Form $\mathbf{x}_k = p_k(\mathbf{A})\mathbf{x}_0$. Dabei ist $p_k$ ein Polynom vom Grad höchstens $k$ mit $p_k(0) = 1$. Ein schnelles Gradientenverfahren für quadratische Funktionen zu entwerfen heißt demnach, ein Polynom zu wählen, dessen Betrag auf dem Intervall $[\mu, L]$ möglichst klein bleibt. Den kleinsten Maximalbetrag auf dem gesamten Intervall erreicht ein skaliertes Chebyshev-Polynom. Sein Maximalbetrag ist $2\rho^k/(1 + \rho^{2k}) \approx 2\rho^k$, mit demselben Kontraktionsfaktor $\rho = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ wie beim Heavy-Ball-Verfahren (:numref:`subsec_mdl-momentum-acceleration`). Ist die Dimension $d$ groß gegenüber $k$, können sich die Eigenwerte über das gesamte Intervall verteilen. Im ungünstigsten Fall kann also kein Verfahren dieser Familie eine bessere Rate erreichen. In der Praxis schneidet das Verfahren der konjugierten Gradienten noch besser ab. In jedem Schritt findet es automatisch das Polynom, das $f$ für die *tatsächlich vorliegende* Matrix $\mathbf{A}$ und den Startpunkt $\mathbf{x}_0$ minimiert. Dazu benötigt es weder $\mu$ noch $L$. Bei exakter Arithmetik endet es nach höchstens $d$ Schritten. :citet:`Nocedal.Wright.2006` (Kap. 5) leiten sowohl das Standardverfahren als auch seine vorkonditionierte Variante her.

<!--en:35b51398-->
### Nachweis von Konvergenzraten durch semidefinite Programme

<!--en:085b6ec1-->
Scharfe Konvergenzraten von Hand zu beweisen ist bekanntlich schwierig. Unser Beweis in :numref:`subsec_mdl-convergence-rates` ergab für jeden Schritt den Kontraktionsfaktor $1 - \mu/L$. Im selben Abschnitt haben wir den exakten Faktor $(1 - \mu/L)^2$ angegeben, mit dem sich die benötigte Schrittzahl halbiert. Die schärfere Schranke liefert eine Methode namens *Performance Estimation*. Die Suche nach einem geschickten algebraischen Trick umgehen :citet:`Drori.Teboulle.2014` mit einem Optimierungsproblem. Sie betrachten das Verhalten eines Algorithmus im ungünstigsten Fall über alle $L$-glatten konvexen Funktionen. Die Optimierungsvariable ist die Gram-Matrix der Iterierten und Gradienten, also die Tabelle aller paarweisen inneren Produkte. Um den Optimalwert dieses Problems abzuschätzen, genügt ein kleines semidefinites Programm. Dessen duale Lösung liefert unmittelbar einen von Hand überprüfbaren Beweis. Auf genau diesem Weg erhalten :citet:`Taylor.Hendrickx.Glineur.2018` den quadrierten Faktor. Aus regelungstheoretischer Sicht betrachten :citet:`Lessard.Recht.Packard.2016` den Gradienten als unbekannte Nichtlinearität in einem linearen dynamischen System. Dabei sind lediglich die Schranken $\mu$ und $L$ vorgegeben. So weisen sie Konvergenzraten mit einem semidefiniten Programm nach, dessen Größe von der Schrittzahl unabhängig ist. Bemerkenswerterweise liefert dieselbe Technik auch Gegenbeispiele. So entdeckten sie die Funktion, für die das Heavy-Ball-Verfahren nicht konvergiert (:numref:`subsec_mdl-momentum-acceleration`).

<!--en:eaaf3ff2-->
## Stochastische und adaptive Verfahren

<!--en:3b625f59-->
### Mittelung bei konstanter Schrittweite

<!--en:34a54772-->
Wie wir in :numref:`subsec_mdl-stochastic-gradients` gesehen haben, erreicht der stochastische Gradientenabstieg (engl. stochastic gradient descent, SGD) bei konstanter Schrittweite eine rauschbedingte Untergrenze (engl. noise floor), die proportional zur Schrittweite $\eta$ ist. Verringern wir $\eta$ im Laufe der Zeit, verschwindet diese Untergrenze. Allerdings verlangsamt sich dann der Fortschritt gegen Ende, und der Verlauf der Schrittweite erfordert eine sorgfältige Abstimmung. Die Mittelung der Iterierten bietet einen einfacheren Ausweg. Die einzelnen Iterierten schwanken weiterhin um das Minimum. In ihrem fortlaufenden Mittelwert hebt sich das Rauschen mit Erwartungswert null auf. :citet:`Polyak.Juditsky.1992` zeigen, dass die Mittelung der Iterierten die asymptotisch optimale Kovarianz erreicht, wenn die Schrittweiten langsamer als $1/k$ abnehmen. Für die Methode der kleinsten Quadrate gehen :citet:`Bach.Moulines.2013` noch einen Schritt weiter. Sie zeigen, dass schon eine *konstante* Schrittweite genügt. Der Fehler der gemittelten Iterierten liegt nach Verarbeitung der Beispiele in der Größenordnung $1/n$, wobei $n$ die Anzahl der Beispiele ist. Das gilt auch ohne starke Konvexität.

<!--en:d7549745-->
### Interpolation

<!--en:1535d3a4-->
Die klassische stochastische Optimierung geht davon aus, dass einzelne Trainingsbeispiele die Parameter am Optimum in unterschiedliche, einander entgegenwirkende Richtungen ziehen. Dadurch entsteht eine unvermeidliche rauschbedingte Untergrenze. Bei modernen überparametrisierten neuronalen Netzen ist die Situation oft eine ganz andere. Passt ein Modell exakt zu jedem Trainingsbeispiel, verschwindet am Minimierer der Gradient jedes einzelnen Beispiels. Damit entfällt die rauschbedingte Untergrenze. Bei einer stark konvexen Verlustfunktion konvergiert SGD mit konstanter Schrittweite auch ohne Varianzreduktion linear (:numref:`sec_mdl-variance-reduction`). :citet:`Ma.Bassily.Belkin.2018` untersuchen diesen Interpolationsfall für die Methode der kleinsten Quadrate. Sie bestimmen die kritische Batchgröße, ab der größere Minibatches das Training nicht mehr beschleunigen. Interessanterweise entscheidet über diese Schwelle das Verhältnis der Krümmung pro Beispiel zur Krümmung auf dem gesamten Trainingssatz und nicht das Gradientenrauschen. Unter einer verwandten *starken Wachstumsbedingung* (engl. strong growth condition) sinkt die Gradientenvarianz proportional zur quadrierten Norm des vollständigen Gradienten. Dann lässt sich SGD sogar so weit beschleunigen, dass es die Konvergenzraten von Momentum-Verfahren erreicht :cite:`Vaswani.Bach.Schmidt.2019`.

<!--en:dda69add-->
### Fisher-Matrix und Gauß-Newton-Matrix

<!--en:a1e581a9-->
Näherungsverfahren zweiter Ordnung wie K-FAC (:numref:`subsec_mdl-preconditioning-ladder`) approximieren die Fisher-Informationsmatrix $\mathbf{F} = \mathbb{E}[\nabla\log p\,\nabla\log p^\top]$. Für den Erwartungswert zieht man die Label aus der prädiktiven Verteilung des Modells selbst :cite:`Amari.1998,Martens.Grosse.2015`. Bei üblichen Verlustfunktionen wie der Kreuzentropie mit Softmax-Funktion und dem quadratischen Fehler stimmt $\mathbf{F}$ mit der Gauß-Newton-Matrix $\mathbf{J}^\top\mathbf{H}_{\mathrm{out}}\mathbf{J}$ überein. Dabei ist $\mathbf{H}_{\mathrm{out}}$ die Hesse-Matrix des Verlusts bezüglich der Netzausgabe. Allgemeiner gilt diese Identität immer dann, wenn die Netzausgabe den natürlichen Parameter einer Likelihood aus der Exponentialfamilie parametrisiert (:numref:`sec_mdl-exponential-family`). Ein häufiger Fehler in der Praxis ist, stattdessen die *empirische* Fisher-Matrix einzusetzen. Sie mittelt dasselbe äußere Produkt über die Ground-Truth-Label des Datensatzes, statt Label aus dem Modell zu ziehen. Ihre Berechnung ist günstiger, und im Code heißt sie oft „Fisher“. Im Allgemeinen stimmt sie jedoch weder mit der tatsächlichen Fisher-Matrix noch mit der Gauß-Newton-Matrix überein :cite:`Martens.2020`.

<!--en:afea960d-->
### Mirror Descent

<!--en:e4c6580f-->
In der metrischen Sicht auf AdaGrad (:numref:`subsec_mdl-per-coordinate`) messen wir Schrittlängen mit einer quadratischen euklidischen Norm. Diese Geometrie passt zu Räumen ohne Nebenbedingungen wie $\mathbb{R}^d$, in denen alle Richtungen gleichwertig sind. Für den Wahrscheinlichkeitssimplex ist sie weniger geeignet. Dort müssen die Koordinaten nichtnegativ bleiben und sich zu eins summieren. Multiplikative Updates liegen deshalb viel näher. Mirror Descent passt die Geometrie des Gradientenabstiegs an. In der proximalen Form des Gradientenschritts ersetzt das Verfahren den quadrierten Abstand $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ durch eine Bregman-Divergenz $D_\phi(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x}) - \phi(\mathbf{y}) - \nabla\phi(\mathbf{y})^\top(\mathbf{x} - \mathbf{y})$ :cite:`Nemirovski.Yudin.1983,Beck.Teboulle.2003`. Wählen wir für $\phi$ die negative Entropie, so erhalten wir die Update-Regel mit exponentiertem Gradienten $x_i \leftarrow x_i e^{-\eta g_i} / \sum_j x_j e^{-\eta g_j}$. Sind die Gradienten koordinatenweise beschränkt, verbessert diese Wahl den in :numref:`subsec_mdl-per-coordinate` definierten Regret. Auf dem $d$-dimensionalen Simplex wächst er beim projizierten Gradientenabstieg wie $\sqrt{d}$, mit der negativen Entropie dagegen nur wie $\sqrt{\log d}$.

<!--en:6cf784ca-->
## Nebenbedingungen und Dualität

<!--en:ee84a291-->
### Sattelpunktverfahren

<!--en:4952e533-->
Die Sattelpunkte aus der Lagrange-Dualität (:numref:`subsec_mdl-strong-duality`) treten auch unmittelbar als Gleichgewichte in Zweipersonenspielen auf, etwa beim GAN-Training (:numref:`sec_basic_gan`). Der naheliegende Ansatz, für einen Spieler einen Gradientenabstieg und gleichzeitig für den anderen einen Gradientenaufstieg auszuführen, hat einen Nachteil. Schon im einfachen bilinearen Spiel $xy$ bewegt sich die Iterierte bei jeder positiven Schrittweite spiralförmig vom Sattelpunkt weg (Aufgabe :exref:`ex_mdl-lagrangian-duality-descent-ascent`). Das Extragradientenverfahren (engl. extragradient method) durchbricht diese Spirale, indem es einen Schritt vorausblickt. Es macht einen Probeschritt, berechnet dort den Gradienten und führt mit diesem Gradienten den Schritt von der ursprünglichen Iterierten aus. Bei konvex-konkaven Problemen mit Lipschitz-Gradienten konvergieren die gemittelten Iterierten mit der Rate $O(1/k)$ :cite:`Nemirovski.2004`. Optimistische Verfahren erzielen eine ähnliche Stabilisierung bei halb so hohen Kosten für die Gradientenberechnung. Als Näherung für den Gradienten am Probepunkt verwenden sie den Gradienten aus dem vorigen Schritt erneut. Diesen Ansatz nutzen :citet:`Daskalakis.Ilyas.Syrgkanis.ea.2018` erfolgreich, um das GAN-Training zu stabilisieren.

<!--en:abb5d03e-->
### Problemklassen und Innere-Punkte-Verfahren

<!--en:f31a59a4-->
Die dualen Probleme, die wir in :numref:`subsec_mdl-worked-duals` von Hand gelöst haben, sind quadratische Programme. Sie gehören zu einer bekannten Hierarchie ineinander geschachtelter Problemklassen der konvexen Optimierung. Auf lineare Programme folgen quadratische Programme, dann Kegelprogramme zweiter Ordnung und schließlich semidefinite Programme. Moderne Innere-Punkte-Verfahren (engl. interior-point methods) lösen Probleme all dieser Klassen mit hoher Genauigkeit in Polynomialzeit. Die Ungleichungen $g_i(\mathbf{x}) \le 0$ berücksichtigen sie indirekt, indem sie zur Zielfunktion eine glatte logarithmische Barriere $-\tfrac1s\sum_i \log(-g_i(\mathbf{x}))$ addieren. Mit wachsendem $s$ verfolgen sie die Minimierer des Problems ohne Nebenbedingungen. Diese nähern sich auf dem *zentralen Pfad* dem Rand. An jedem Punkt dieses Pfads bilden die Größen $-1/(s\,g_i(\mathbf{x}))$ automatisch dual zulässige Multiplikatoren. Damit weisen sie eine Dualitätslücke von $m/s$ bei $m$ Nebenbedingungen nach. So erhält man automatisch Optimalitätszertifikate der Art, die wir in :numref:`sec_mdl-lagrangian-duality` von Hand hergeleitet haben :cite:`Boyd.Vandenberghe.2004,Nesterov.2018`.

<!--en:26a2b22f-->
### Operatoraufspaltung und ADMM

<!--en:e35817f6-->
Die Proximalgradientenschritte aus :numref:`subsec_mdl-proximal-operators` eignen sich für die Summe aus einer glatten Verlustfunktion und einem einzigen einfachen, nichtglatten Strafterm. Viele strukturierte Probleme haben dagegen die zusammengesetzte Form $\min_{\mathbf{x}} f(\mathbf{x}) + h(\mathbf{A}\mathbf{x})$. Beide Terme sind konvex und können nichtglatt sein. Das zugehörige duale Problem nach Fenchel lautet $\max_{\mathbf{y}} -f^*(-\mathbf{A}^\top\mathbf{y}) - h^*(\mathbf{y})$ :cite:`Rockafellar.1970`. Um die beiden Terme getrennt behandeln zu können, führen wir eine Hilfsvariable $\mathbf{z} = \mathbf{A}\mathbf{x}$ ein und ergänzen die Lagrange-Funktion um einen quadratischen Strafterm $\tfrac{\tau}{2}\|\mathbf{A}\mathbf{x} - \mathbf{z}\|^2$. Wir minimieren abwechselnd über $\mathbf{x}$ und über $\mathbf{z}$. Die zweite Minimierung reduziert sich auf einen proximalen Schritt für $h$. Abschließend führen wir einen dualen Aufstiegsschritt für $\mathbf{y}$ aus. So erhalten wir die Alternating Direction Method of Multipliers (ADMM), die für jedes $\tau > 0$ konvergiert. :citet:`Boyd.Parikh.Chu.ea.2011` zeigen, wie sich große Optimierungsprobleme durch diese Aufspaltung auf mehrere Rechner verteilen lassen. :citet:`Parikh.Boyd.2014` bieten eine praktische Übersicht über Proximaloperatoren in geschlossener Form für gängige Regularisierungsterme und Nebenbedingungen.

<!--en:a5d96aa1-->
### Frank-Wolfe-Verfahren

<!--en:023bffd4-->
Beim projizierten Gradientenabstieg müssen wir in jeder Iteration auf die zulässige Menge projizieren. Für einen Quader oder eine $\ell_2$-Kugel ist das einfach. Bei einer Kugel bezüglich der Nuklearnorm erfordert schon eine Projektion eine vollständige Singulärwertzerlegung (SVD). Diese Kugel enthält alle Matrizen, deren Singulärwerte sich zu höchstens einer vorgegebenen Schranke summieren. Das Frank-Wolfe-Verfahren :cite:`Frank.Wolfe.1956` kommt ganz ohne Projektionen aus. Es minimiert eine lineare Näherung der Zielfunktion über der zulässigen Menge. Dazu bestimmt es $\mathbf{s}_k = \operatorname{argmin}_{\mathbf{s} \in C} \nabla f(\mathbf{x}_k)^\top\mathbf{s}$ und bildet die Konvexkombination $\mathbf{x}_{k+1} = (1 - \gamma_k)\mathbf{x}_k + \gamma_k\mathbf{s}_k$ mit der Schrittweite $\gamma_k = 2/(k + 2)$. Auf der Kugel bezüglich der Nuklearnorm benötigt man für dieses lineare Teilproblem nur das Singulärvektorpaar zum größten Singulärwert des Gradienten. Auf dem Wahrscheinlichkeitssimplex genügt es, die kleinste Koordinate zu finden. Das Verfahren konvergiert mit der Rate $O(LD^2/k)$ auf einer Menge mit Durchmesser $D$. Zudem liefert die Größe $\nabla f(\mathbf{x}_k)^\top(\mathbf{x}_k - \mathbf{s}_k)$ in jedem Schritt ohne zusätzlichen Rechenaufwand eine obere Schranke für $f(\mathbf{x}_k) - f^\star$ :cite:`Jaggi.2013`.

<!--en:a567f2b2-->
## Gleitkommaarithmetik

<!--en:ae19a077-->
### Probabilistische Analyse von Rundungsfehlern

<!--en:0c9c491c-->
Die Schranken für den Summationsfehler im ungünstigsten Fall aus :numref:`subsec_mdl-catastrophic-cancellation` wachsen linear wie $nu$, wenn wir $n$ Terme nacheinander addieren. In der Praxis fallen die gemessenen Rundungsfehler meist viel kleiner aus. Die Analyse des ungünstigsten Falls setzt nämlich voraus, dass jeder einzelne Rundungsfehler in dieselbe ungünstige Richtung wirkt. :citet:`Higham.Mary.2019` modellieren die einzelnen Rundungsfehler dagegen als unabhängige Zufallsvariablen mit Erwartungswert null. Sie beweisen Schranken der Größenordnung $\sqrt{n}\,u$, die mit hoher Wahrscheinlichkeit gelten. Das passt zu der Beobachtung, dass sich die Fehler in Experimenten typischerweise wie bei einer Irrfahrt teilweise aufheben. Ebenso wichtig ist, dass die Analyse erklärt, wann die günstige Abschätzung $\sqrt{n}\,u$ versagt. Das geschieht immer dann, wenn die Rundungsfehler stark korreliert sind. Ein Beispiel ist die Summe aus $10^7$ Summanden mit dem Wert $0.1$ im genannten Abschnitt. Dort rundet die Arithmetik bei vielen aufeinanderfolgenden Additionen in dieselbe Richtung.

<!--en:8f03209a-->
### Wachstumsfaktoren und iterative Nachbesserung

<!--en:9bbef36c-->
In :numref:`subsec_mdl-conditioning-revisited` haben wir festgehalten, dass die LU-Zerlegung mit partieller Pivotisierung in der Praxis rückwärtsstabil ist. Genauer gesagt ist ihr Rückwärtsfehler durch ein Polynom niedrigen Grades in $n$ mal $\rho_n u$ beschränkt. Der *Wachstumsfaktor* $\rho_n$ ist das Verhältnis zwischen dem betragsgrößten Eintrag, der während der Elimination entsteht, und dem betragsgrößten Eintrag von $\mathbf{A}$. Im ungünstigsten Fall steigt $\rho_n$ bis auf $2^{n-1}$, wie Wilkinsons berühmtes pathologisches Matrixbeispiel zeigt. Für fast alle Matrizen, die in der Praxis vorkommen, bleibt $\rho_n$ jedoch klein. Wer eine uneingeschränkte Garantie braucht, kann mit der QR-Zerlegung exponentielles Wachstum ganz ausschließen. Dafür sind etwa doppelt so viele Rechenoperationen nötig :cite:`Higham.2002,Trefethen.Bau.1997`. Mit iterativer Nachbesserung (engl. iterative refinement) in gemischter Genauigkeit lässt sich eine hohe Genauigkeit erzielen, ohne überall mit doppelter Genauigkeit zu rechnen. Dazu zerlegen wir $\mathbf{A}$ kostengünstig in fp32, berechnen das Residuum in fp64 und lösen mit den Faktoren niedriger Genauigkeit das Gleichungssystem für die Korrektur. In der letzten Aufgabe von :numref:`sec_mdl-numerical-stability-conditioning` bestimmen Sie, wie schlecht $\mathbf{A}$ konditioniert sein darf, bevor diese Schleife keinen Fortschritt mehr macht. Verwendet man die Faktoren niedriger Genauigkeit als Vorkonditionierer in GMRES, einem iterativen Lösungsverfahren für allgemeine lineare Gleichungssysteme, gelingt die Nachbesserung in gemischter Genauigkeit auch bei wesentlich schlechter konditionierten Matrizen :cite:`Carson.Higham.2018`.

<!--en:06ac78b6-->
## Allgemeine Literatur und Lehrmaterialien zur Optimierung

<!--en:f3e47eb0-->
**Bücher**

<!--en:416818bc-->
- [Convex Optimization – Boyd und Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) behandelt konvexe Mengen und Funktionen, Dualität und Innere-Punkte-Verfahren. Das Buch ist kostenlos als [PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) erhältlich und dient als Standardwerk für :numref:`sec_mdl-convexity`, :numref:`sec_mdl-constrained-optimization-duality` und :numref:`sec_mdl-lagrangian-duality`.

<!--en:3bf8eea1-->
- [Numerical Optimization – Nocedal und Wright (2. Auflage, Springer)](https://doi.org/10.1007/978-0-387-40065-5) behandelt Liniensuche, Trust-Regionen, Quasi-Newton-Verfahren, Verfahren der konjugierten Gradienten und beschränkte Optimierung ausführlicher als :numref:`sec_mdl-gradient-based-optimization`, :numref:`sec_mdl-curvature-methods` und :numref:`sec_mdl-constrained-optimization-duality`.

<!--en:520d06d2-->
- [Lectures on Convex Optimization – Nesterov (2. Auflage, Springer)](https://doi.org/10.1007/978-3-319-91578-4) behandelt Komplexitätsschranken und beschleunigte Verfahren erster Ordnung. Hier finden sich die allgemeinen Beweise, auf die sich :numref:`sec_mdl-gradient-based-optimization` und :numref:`sec_mdl-convexity` beziehen.

<!--en:27dd5000-->
- [Optimization for Data Analysis – Wright und Recht (Cambridge, 2022)](https://www.cambridge.org/core/books/optimization-for-data-analysis/C02C3708905D236AA354D1CE1739A6A2) behandelt Gradientenverfahren, beschleunigte und stochastische Verfahren sowie Koordinatenverfahren für die Datenanalyse. Das Buch eignet sich als Begleitlektüre zu :numref:`sec_mdl-gradient-based-optimization` bis :numref:`sec_mdl-adaptive-stochastic-methods`.

<!--en:59b32a2b-->
- [Accuracy and Stability of Numerical Algorithms – Higham (2. Auflage, SIAM)](https://doi.org/10.1137/1.9780898718027) behandelt die Analyse von Rundungsfehlern, die Summation sowie die Kondition linearer Gleichungssysteme und von Problemen der kleinsten Quadrate (:numref:`sec_mdl-numerical-stability-conditioning`).

<!--en:83560036-->
- [Mathematics for Machine Learning – Deisenroth, Faisal und Ong](https://mml-book.github.io/) enthält ein kostenloses, in sich geschlossenes Kapitel zu Gradientenabstieg, beschränkter Optimierung und Konvexität.

<!--en:b3edb95b-->
**Kurse und Vorlesungsvideos**

<!--en:f47090d9-->
- [Stanford EE364a: Convex Optimization I – Boyd](https://web.stanford.edu/class/ee364a/) bietet Folien, Übungsaufgaben und Software zum Lehrbuch von Boyd und Vandenberghe sowie [Vorlesungsvideos von 2023](https://www.youtube.com/playlist?list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h) (:numref:`sec_mdl-convexity`, :numref:`sec_mdl-lagrangian-duality`).

<!--en:f78d1b4d-->
- [CMU 10-725: Convex Optimization – Tibshirani](https://www.stat.cmu.edu/~ryantibs/convexopt/) bietet Folien und Vorlesungsmitschriften mit Schwerpunkt auf maschinellem Lernen. Sie behandeln Subgradienten, proximale und stochastische Verfahren, Dualität und ADMM. Dazu gibt es [Vorlesungsvideos aus dem Herbst 2016](https://www.youtube.com/playlist?list=PLjbUi5mgii6AVdvImLB9-Hako68p9MpIC) (:numref:`sec_mdl-stochastic-gradient-methods`, :numref:`sec_mdl-constrained-optimization-duality`, :numref:`sec_mdl-lagrangian-duality`).

<!--en:ec1d8b8a-->
**Anleitungen, Skripte und Übersichtsarbeiten**

<!--en:acdfd421-->
- [Why Momentum Really Works – Goh (Distill)](https://distill.pub/2017/momentum/) zeigt anhand interaktiver Abbildungen, wie Momentum bei konvexen quadratischen Funktionen wirkt (:numref:`subsec_mdl-momentum-acceleration`).

<!--en:698eab79-->
- [Optimization Methods for Large-Scale Machine Learning – Bottou, Curtis und Nocedal](https://arxiv.org/abs/1606.04838) behandelt die Theorie stochastischer Gradientenverfahren für maschinelles Lernen in großem Maßstab (:numref:`sec_mdl-stochastic-gradient-methods`).

<!--en:444c6f03-->
- [An overview of gradient descent optimization algorithms – Ruder](https://www.ruder.io/optimizing-gradient-descent/) stellt Momentum, Nesterov-Beschleunigung, AdaGrad, RMSProp und Adam einander gegenüber (:numref:`sec_mdl-adaptive-stochastic-methods`).

<!--en:8f9c5162-->
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic – Goldberg](https://doi.org/10.1145/103162.103163) behandelt das IEEE-Format, Rundung und Auslöschung (:numref:`sec_mdl-numerical-stability-conditioning`).

<!--en:1370a5e9-->
- [Optimization (Kapitel 5), Patterns, Predictions, and Actions – Hardt und Recht](https://mlstory.org/optimization.html) gibt einen knappen Überblick über die Optimierung beim überwachten Lernen (:numref:`sec_mdl-gradient-based-optimization`, :numref:`sec_mdl-stochastic-gradient-methods`).

<!--en:cce82a9e-->
[Diskussionen](https://d2l.discourse.group/t/36861)
