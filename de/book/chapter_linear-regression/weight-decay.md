<!--en:03f9ff0c-->
# Gewichtszerfall

<!--en:d2c240c4-->
In :numref:`sec_generalization_basics` haben wir die Modellkomplexität auf denkbar grobe Weise gesteuert: Wir haben die Anzahl $k + 1$ der polynomialen Merkmale verändert. Eine solche diskrete Merkmalsauswahl stößt schnell an Grenzen. Bei hochdimensionalen Problemen mit Hunderten oder Tausenden von Merkmalen gibt es $2^d$ mögliche Teilmengen. Erschwerend kommt hinzu, was wir bei der Überanpassung (engl. overfitting) eines Polynoms beobachtet haben: Seine Gewichte $w_j$ erreichen Millionenwerte und mehr, damit sich die Beiträge fast kollinearer Merkmale gegenseitig aufheben.

<!--en:5617c51f-->
Betrachten wir das Problem in seiner einfachsten Form mit zwei Merkmalen. Wir möchten die Raumtemperatur anhand der Messwerte zweier nebeneinanderstehender Thermometer vorhersagen. Die Messwerte $x_1$ und $x_2$ weichen jeweils nur um ein kleines, unabhängiges Rauschen von der wahren Temperatur ab. Vergleichen wir zwei lineare Modelle, deren Gewichte sich zu $2$ summieren:

<!--en:302b7fd5-->
- **Modell A** setzt $\mathbf{w} = (1, 1)^\top$. Es gewichtet beide Messwerte gleich. Bezogen auf ihren Mittelwert halbiert sich damit die Rauschvarianz.

<!--en:64d1378c-->
- **Modell B** setzt $\mathbf{w} = (501, -499)^\top$. Stimmen die beiden Thermometer überein, liefert $501 x_1 - 499 x_2 = 2 x_1$ denselben Wert wie Modell A. Jede noch so kleine Differenz $x_1 - x_2$ zwischen den Messwerten verstärkt Modell B jedoch um den Faktor $500$.

<!--en:248ab259-->
In einer kleinen Trainingsstichprobe kann $x_1 - x_2$ zufällig leicht mit dem Rauschen der Labels korrelieren. Dann wählt die unregularisierte Methode der kleinsten Quadrate (engl. least squares) $(501, -499)^\top$, um den Trainingsfehler geringfügig zu senken. Bei neuen Messwerten versagt das Modell. Wie können wir alle $d$ Merkmale im Modell behalten und zugleich verhindern, dass die Gewichte in schlecht bestimmten Richtungen zu groß werden?

<!--en:62a4fbf6-->
Die klassische Lösung heißt in der Statistik *Ridge-Regression* oder *Tikhonov-Regularisierung* :cite:`Hoerl.Kennard.1970,tikhonov1977solutions` und im Deep Learning *Gewichtszerfall* (engl. weight decay). Sie addiert direkt zur Verlustfunktion (engl. loss function) einen glatten Strafterm, der von der quadrierten euklidischen Norm $\|\mathbf{w}\|_2^2$ abhängt. Wir leiten den Gewichtszerfall aus vier einander ergänzenden Sichtweisen her: Optimierung mit Strafterm, multiplikativer Zerfall bei jedem Schritt des stochastischen Gradientenabstiegs (engl. stochastic gradient descent, SGD), bayessche Modellierung mit einem Gauß-Prior und spektrale Schrumpfung (engl. shrinkage) entlang der Eigenvektoren von $\mathbf{X}^\top\mathbf{X}$. Für alle vier Frameworks erstellen wir sowohl eine eigene als auch eine kompakte Implementierung.

<!--en:25b1b282-->
## Strafterme auf Basis von Normen

<!--en:a655a4ce-->
### Der Verlust mit Strafterm

<!--en:aabc9f3c-->
Wir können die effektive Kapazität eines linearen Modells steuern, ohne die Anzahl der Merkmale zu beschränken. Dazu bestrafen wir die quadrierte $\ell_2$-Norm seines Gewichtsvektors, also $\|\mathbf{w}\|_2^2 = \sum_{j=1}^d w_j^2$. Als Verlust verwenden wir den mittleren quadratischen Fehler $L(\mathbf{w}, b)$ und addieren den Strafterm mit einer nichtnegativen *Regularisierungskonstanten* $\lambda \geq 0$ als Gewicht. So erhalten wir die regularisierte Zielfunktion:

<!--en:1876425b-->
Der Faktor $\frac{1}{2}$ hebt wieder die $2$ auf, die beim Ableiten von $\|\mathbf{w}\|_2^2$ entsteht. Betrachten wir drei grundlegende Eigenschaften von :eqref:`eq_wd-objective`:

<!--en:4eb7520a-->
1. **Abwägung über $\lambda$:** Mit $\lambda = 0$ erhalten wir die unregularisierte Methode der kleinsten Quadrate. Je größer $\lambda$ ist, desto stärker bestraft die Zielfunktion große Gewichte und desto näher rückt $\mathbf{w}$ an $\mathbf{0}$. Es gilt $\mathbf{w} \to \mathbf{0}$ für $\lambda \to \infty$.

<!--en:ea57fc3f-->
2. **Warum der Bias $b$ keinen Strafterm erhält:** Der Bias $b$ verschiebt lediglich das mittlere Ausgabeniveau auf den Mittelwert der Zielgröße $y$. Würden wir $b$ gegen null ziehen, wären alle Vorhersagen in Richtung null verzerrt, wenn wir $y$ etwa in Kelvin statt in Grad Celsius messen. Die Empfindlichkeit des Modells gegenüber Störungen der Eingabe nähme dadurch keineswegs ab.

<!--en:73450303-->
3. **Die Skalierung der Merkmale zählt:** $\sum_j w_j^2$ summiert die quadrierten Koeffizienten aller Koordinaten ohne weitere Skalierung. Geben wir ein Merkmal in Millimetern statt in Metern an, so teilen wir sein Gewicht $w_j$ durch $1{,}000$. Sein Beitrag zum Strafterm sinkt dadurch um den Faktor $10^6$. In der Praxis standardisiert man die Merkmale auf vergleichbare Skalen (Mittelwert null, Varianz eins), bevor man einen isotropen Strafterm mit $\|\mathbf{w}\|_2^2$ anwendet.

<!--en:a4462d77-->
Warum bestrafen wir die quadrierte $\ell_2$-Norm? Bei unseren beiden Thermometern gilt für $(1, 1)^\top$ die Beziehung $\|\mathbf{w}\|_2^2 = 2$, für $(501, -499)^\top$ dagegen $\|\mathbf{w}\|_2^2 = 500{,}002$. Selbst ein winziges $\lambda > 0$ bevorzugt also deutlich die gleichmäßige Gewichtung der beiden Messwerte gegenüber ihrer Subtraktion.

<!--en:b540d608-->
### Gewichtszerfall im Gradientenschritt

<!--en:6d47223b-->
Wie verändert der Strafterm $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ die Aktualisierungsregel für Minibatch-SGD? Wegen $\nabla_{\mathbf{w}}\left(\frac{\lambda}{2}\|\mathbf{w}\|_2^2\right) = \lambda \mathbf{w}$ lautet der Gradientenschritt mit Strafterm

<!--en:64f716a8-->
Betrachten wir den ersten Term auf der rechten Seite: **Vor dem üblichen Gradientenschritt $-\eta\,\partial L / \partial \mathbf{w}$ auf den Daten multiplizieren wir den Gewichtsvektor mit $(1 - \eta\lambda) < 1$. Bei jeder Aktualisierung schrumpft er dadurch ein wenig in Richtung null.** Wegen dieses multiplikativen Schrumpfungsfaktors $(1 - \eta\lambda)$ heißt die $\ell_2$-Regularisierung im Deep Learning *Gewichtszerfall*. (Bei einfachem SGD ist es algebraisch gleichwertig, $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ zum Verlust zu addieren oder die Gewichte mit $(1 - \eta\lambda)$ zu multiplizieren. Bei adaptiven Optimierern wie Adam in :numref:`chap_optimization` macht es hingegen einen wesentlichen Unterschied, den multiplikativen Gewichtszerfall zu entkoppeln, wie es *AdamW* tut.)

<!--en:1d00da26-->
### Der $\ell_1$-Strafterm und Sparsität

<!--en:3cd5388f-->
Was geschieht, wenn wir die quadrierte $\ell_2$-Norm $\|\mathbf{w}\|_2^2 = \sum_j w_j^2$ durch die $\ell_1$-Norm $\|\mathbf{w}\|_1 = \sum_j |w_j|$ ersetzen? In der Statistik heißt dieses Verfahren *Lasso* :cite:`Tibshirani.1996`. Die Geometrie der zulässigen Mengen (:numref:`fig_ridge_geometry`) zeigt einen wesentlichen Unterschied. Minimieren wir den Datenverlust $L(\mathbf{w})$ unter der Nebenbedingung $\|\mathbf{w}\|_2^2 \leq r^2$, suchen wir innerhalb einer Kreisscheibe mit glattem Rand (oder einer Kugel in $\mathbb{R}^d$). Die elliptischen Niveaulinien des Verlusts berühren den Rand fast immer dort, wo *jede* Koordinate $w_j$ klein, aber von null verschieden ist. Minimieren wir dagegen $L(\mathbf{w})$ unter der Nebenbedingung $\|\mathbf{w}\|_1 \leq r$, suchen wir innerhalb einer Raute mit spitzen Ecken (eines Kreuzpolytops). Ihre Ecken liegen direkt auf den Koordinatenachsen. Eine elliptische Niveaulinie trifft häufig zuerst auf eine dieser Ecken. Dadurch werden viele Koordinaten $w_j$ *exakt null*. Das Verfahren wählt also automatisch Merkmale aus (*Sparsität*).

<!--en:62e1b98b-->
![Geometrie der Nebenbedingung bei der $\ell_2$-Ridge-Regression (links) und der $\ell_1$-Lasso-Regression (rechts). Vom unregularisierten Minimum aus werden die elliptischen Niveaulinien des Datenverlusts $L(\mathbf{w})$ um $\hat{\mathbf{w}}$ immer größer. Den glatten $\ell_2$-Kreis berühren sie meist abseits der Achsen (beide Gewichte bleiben ungleich null). Die $\ell_1$-Raute treffen sie oft an einer spitzen Ecke auf einer Achse (damit gilt $w_2 = 0$).](../img/mdl-linreg-ridge-geometry.svg)

<!--en:90e0766f-->
### Ein Gauß-Prior für die Gewichte

<!--en:ff0dd985-->
In :numref:`subsec_normal_distribution_and_squared_loss` ergab sich der quadratische Datenverlust aus dem Maximum-Likelihood-Prinzip bei gaußschem Beobachtungsrauschen. Die bayessche Statistik liefert eine ebenso klare probabilistische Deutung des Gewichtszerfalls. Nehmen wir vor der Beobachtung der Daten für die Gewichte eine isotrope gaußsche *A-priori-Verteilung* (engl. prior distribution) mit Mittelwert null an. Die Gewichte folgen unabhängig voneinander der Verteilung $w_j \sim \mathcal{N}(0, \tau^2)$, sodass $p(\mathbf{w}) \propto \exp(-\|\mathbf{w}\|_2^2 / (2\tau^2))$ gilt. Für die Labels gilt $y \mid \mathbf{x}, \mathbf{w}, b \sim \mathcal{N}(\mathbf{w}^\top\mathbf{x} + b, \sigma^2)$.

<!--en:8a07b62f-->
Nach dem Satz von Bayes maximieren wir die *A-posteriori-Dichte* $p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, b)\,p(\mathbf{w})$, indem wir ihren negativen Logarithmus minimieren (*Maximum-a-posteriori-Schätzung*, kurz MAP-Schätzung):

<!--en:d0e19a01-->
Dividieren wir durch $n / \sigma^2$, erhalten wir wieder unsere Zielfunktion für den Gewichtszerfall :eqref:`eq_wd-objective` mit $\lambda = \sigma^2 / (n\tau^2)$. Dieses Verhältnis hat eine anschauliche Bedeutung. Je stärker wir im Voraus kleine Gewichte erwarten (kleineres $\tau^2$) oder je stärker die Beobachtungen rauschen (größeres $\sigma^2$), desto stärker sollte die Regularisierung $\lambda$ sein. Wächst dagegen der Stichprobenumfang unbegrenzt, also $n \to \infty$, so gilt für die Regularisierungskonstante $\lambda = \sigma^2 / (n\tau^2) \to 0$. Bei hinreichend vielen Daten sprechen die Beobachtungen für sich.

<!--en:d961af34-->
## Eigene Implementierung des Gewichtszerfalls

<!--en:43535bbc-->
Wie hilft Gewichtszerfall einem Modell mit fast ebenso vielen Gewichten wie Trainingsbeispielen? Konstruieren wir dazu ein hochdimensionales synthetisches Regressionsproblem, bei dem die Dimension $d = 200$ nahe am Umfang der Trainingsstichprobe $n_{\textrm{train}} = 300$ liegt. Wir ziehen Eingaben $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{200})$ und erzeugen die Labels gemäß

<!--en:57c56608-->
mit $300$ Trainingsbeispielen (in Minibatches der Größe 20) und $1{,}000$ Validierungsbeispielen. Dazu verwenden wir `SyntheticRegressionData` aus :numref:`sec_oo-design`. Die wahren Gewichte $\mathbf{w}^* = (0.1, \ldots, 0.1)^\top$ haben in jeder Koordinate die Größenordnung $\tau = 0.1$. Das Rauschen ist beträchtlich: Seine Standardabweichung $\sigma = 1$ ist vergleichbar mit der des rauschfreien Signals, $\|\mathbf{w}^*\| = \sqrt{2} \approx 1.4$. Zwar reichen 300 Beispiele grundsätzlich aus, um alle 200 Gewichte zu bestimmen. Wie wir gleich sehen, sind viele ihrer Linearkombinationen jedoch nur schwach durch die Daten bestimmt.

<!--en:72b723fe-->
Legen wir vor dem Training zwei Vergleichswerte für den Validierungsverlust fest. Für unabhängige standardnormalverteilte Eingaben $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ und Rauschen $\epsilon \sim \mathcal{N}(0, \sigma^2)$ beträgt der erwartete Verlust an der Stelle $(\mathbf{w}, b^*)$ genau $\frac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|_2^2 + \frac{1}{2}\sigma^2$ (:numref:`sec_linear_scratch`). Ein konstantes Modell, das alle 200 Eingabewerte ignoriert und stets $b^*$ vorhersagt, hat daher den erwarteten Verlust $\frac{1}{2}(\|\mathbf{w}^*\|^2 + \sigma^2) = \frac{1}{2}(2 + 1) = 1.5$. Die wahren Parameter $(\mathbf{w}^*, b^*)$ erreichen hingegen $\sigma^2/2 = 0.5$. Ein trainiertes Modell muss deutlich unter 1.5 liegen, um seine 200 Gewichte zu rechtfertigen.

<!--en:c647d9d5-->
### Das Modell mit Strafterm

<!--en:e4a9ef3a-->
Wir definieren `l2_penalty(w)` so, dass die Funktion $\frac{1}{2}\|\mathbf{w}\|_2^2$ zurückgibt:

<!--en:7c914706-->
Als Basisklasse verwenden wir `LinearRegressionScratch` aus :numref:`sec_oo-design`, unsere objektorientierte Fassung des selbst implementierten Modells aus :numref:`sec_linear_scratch`. Nur `training_step` müssen wir ändern: Die Methode soll für die Ableitung und die Optimierung die Zielfunktion mit Strafterm :eqref:`eq_wd-objective` zurückgeben. Entscheidend ist, dass `validation_step` weiterhin den quadratischen Verlust ohne Strafterm ausgibt. So können wir die Genauigkeit der Vorhersagen für verschiedene Werte von $\lambda$ direkt anhand des Validierungsverlusts vergleichen.

<!--en:22cdfdb8-->
### Training mit und ohne Strafterm

<!--en:568dc56d-->
Schreiben wir die Funktion `train_scratch(lambd, data)`. Sie trainiert `WeightDecayScratch` über 50 Epochen mit der Lernrate $\eta = 0.05$ (750 SGD-Schritte) und zeichnet die Trainings- und Validierungskurven. Sie gibt das angepasste Modell, die Validierungsverluste je Epoche und die Norm $\|\mathbf{w}\|_2$ des gelernten Gewichtsvektors zurück. Zum Vergleich: Für den wahren Gewichtsvektor $\mathbf{w}^* = (0.1, \ldots, 0.1)^\top \in \mathbb{R}^{200}$ gilt $\|\mathbf{w}^*\|_2 = \sqrt{200 \times 0.1^2} \approx 1.4$.

<!--en:94afbb21-->
Wir beginnen mit `lambd = 0` und schalten damit die Regularisierung vollständig aus.

<!--en:15941bf7-->
Der Trainingsverlust sinkt auf etwa 0.2–0.3 und damit weit unter die rauschbedingte Untergrenze von 0.5, die die wahren Parameter erreichen. Das bestätigt, dass das Modell das Rauschen in seinen 300 Trainingslabels auswendig lernt. Der Validierungsverlust erreicht dagegen schon nach zwei oder drei Epochen sein Minimum von 0.8–1.0. Danach steigt er stetig an und liegt am Ende meist zwischen 1.1 und 1.9. Nach 50 Epochen hat das Modell mit seinen 200 Gewichten einen Großteil seines anfänglichen Vorsprungs verloren. Oft schneidet es nur so gut ab wie die konstante Vorhersage $b^*$ mit einem Verlust von 1.5, manchmal sogar schlechter. Das ist ein Musterbeispiel für Überanpassung. Warum liegt das Minimum des Validierungsverlusts so früh, nach nur 30 bis 45 Gradientenschritten? Die Antwort ergibt sich, wenn wir untersuchen, welche Richtungen im Gewichtsraum der Gradientenabstieg zuerst lernt.

<!--en:2098c14c-->
Nun schalten wir den Gewichtszerfall ein. Statt $\lambda$ zu erraten, nutzen wir die Formel für den Gauß-Prior. Für $\sigma = 1$, $n = 300$ und die wahre Größenskala der Gewichte $\tau = 0.1$ ergibt die MAP-Schätzung $\lambda = \sigma^2/(n\tau^2) = \frac{1}{3}$.

<!--en:3e068915-->
Die Trainingskurve zeigt jetzt den Wert der Zielfunktion mit Strafterm. Er bleibt deutlich über 0.5, denn der Strafterm hindert das Modell daran, sich an das Rauschen der Trainingslabels anzupassen. Der Validierungsverlust pendelt sich bei etwa 0.75–1.05 ein und bleibt damit deutlich unter dem Vergleichswert von 1.5 für die konstante Vorhersage. Die Gewichtsnorm schrumpft von rund 2 auf etwa 1.2 und ist damit sogar *kleiner* als die wahre Norm $\|\mathbf{w}^*\| \approx 1.4$. Obwohl der regularisierte Gewichtsvektor kürzer ist als der wahre, generalisiert er wesentlich besser als der unregularisierte.

<!--en:346aa533-->
Wie viel haben wir tatsächlich gewonnen? Gegenüber dem Modell ohne Strafterm, das wir volle 50 Epochen trainiert haben, senkt der Gewichtszerfall den Validierungsverlust um ein Viertel bis zur Hälfte. Beenden wir das Training desselben Modells dagegen nach seiner besten Epoche, ist der Gewinn gering: Beide Modelle erreichen etwa 0.8–1.0. Der praktische Unterschied liegt in der Stabilität. Gewichtszerfall verändert die Zielfunktion selbst. Deshalb bleibt die Lösung gut, egal wie lange wir trainieren. Ohne Strafterm müssen wir das Training zum richtigen Zeitpunkt beenden, den der Trainingsverlust allein nicht erkennen lässt. *Warum* sich Regularisierung und Early Stopping so verhalten, erkennen wir am Gewichtsvektor. Welche seiner Komponenten hat der Strafterm tatsächlich entfernt?

<!--en:7008eeb7-->
## Schrumpfung entlang der Eigenvektoren

<!--en:ba87bc4b-->
Warum senkt der zusätzliche Term $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ den Validierungsfehler so stark, statt jedes Gewicht unterschiedslos zu verkleinern? Eine Antwort in geschlossener Form erhalten wir, wenn wir den exakten Minimierer der Ridge-Zielfunktion in der Eigenbasis der Kovarianzmatrix der Merkmale betrachten.

<!--en:89d8783c-->
### Der Ridge-Schätzer

<!--en:b2d9375f-->
Nehmen wir an, die Merkmale und Labels seien zentriert, sodass $b = 0$ gilt. Dann schreiben wir die Ridge-Zielfunktion :eqref:`eq_wd-objective` in Matrixform:

<!--en:bdbbe0c5-->
Setzen wir den Gradienten gleich null, also $\nabla_{\mathbf{w}} L_{\lambda}(\mathbf{w}) = \frac{1}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w} = \mathbf{0}$, erhalten wir die regularisierten Normalgleichungen:

<!--en:a04b02fa-->
Daraus folgt unmittelbar eine algebraische Eigenschaft. Selbst wenn $\mathbf{X}^\top\mathbf{X}$ singulär ist, können wir das Gleichungssystem eindeutig lösen. Eine singuläre Matrix erhalten wir immer dann, wenn es weniger Beispiele als Merkmale gibt: Bei $n = 20$ und $d = 200$ hat $\mathbf{X}^\top\mathbf{X}$ im betrachteten Beispiel 180 Eigenwerte gleich null (Aufgabe :exref:`ex_weight-decay-more-weights`). Addieren wir $n\lambda\mathbf{I}$, so steigt für jedes $\lambda > 0$ jeder Eigenwert um $n\lambda > 0$. Dadurch ist $\mathbf{X}^\top\mathbf{X} + n\lambda\mathbf{I}$ strikt positiv definit und invertierbar.

<!--en:405eb427-->
### Schrumpfungsfaktoren

<!--en:34461345-->
Wie wirkt :eqref:`eq_wd-ridge-normal` entlang der einzelnen Richtungen im Merkmalsraum? Sei $\mathbf{S} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ die $d \times d$-Matrix der empirischen zweiten Momente der Merkmale (bei zentrierten Daten die Kovarianzmatrix). Da $\mathbf{S}$ symmetrisch und positiv semidefinit ist (:numref:`sec_linear-algebra`), besitzt sie eine orthonormale Eigenwertzerlegung $\mathbf{S} = \sum_{j=1}^d \mu_j \mathbf{v}_j \mathbf{v}_j^\top$ mit Eigenwerten $\mu_1 \geq \mu_2 \geq \cdots \geq \mu_d \geq 0$ und orthonormalen Eigenvektoren $\mathbf{v}_1, \ldots, \mathbf{v}_d$. Dabei misst $\mu_j = \frac{1}{n}\|\mathbf{X}\mathbf{v}_j\|_2^2$ die empirische Varianz (Streuung) der Trainingseingaben entlang der Richtung $\mathbf{v}_j$.

<!--en:35373209-->
In dieser Eigenbasis wirkt $\mathbf{S}$ auf $\mathbf{v}_j$ wie eine Multiplikation mit $\mu_j$. Dagegen wirkt $(\mathbf{S} + \lambda\mathbf{I})^{-1}$ auf $\mathbf{v}_j$ wie eine Division durch $\mu_j + \lambda$. Für $\mu_j > 0$ ergibt der Vergleich der Ridge-Komponente $\mathbf{v}_j^\top \hat{\mathbf{w}}_{\lambda}$ mit der unregularisierten Kleinste-Quadrate-Komponente $\mathbf{v}_j^\top \hat{\mathbf{w}}_0$ die Beziehung

<!--en:1d54113c-->
Für jede Eigenrichtung $\mathbf{v}_j$ gilt also ein eigener *Schrumpfungsfaktor* (:numref:`fig_ridge_shrinkage`):

<!--en:393d6b6f-->
![Schrumpfungsfaktor $\mu / (\mu + \lambda)$ der Ridge-Regression entlang eines Eigenvektors mit Eigenwert $\mu$ in Abhängigkeit von der Regularisierungskonstanten $\lambda$ (logarithmische Skala). Eine durch die Daten gut bestimmte Richtung bleibt für $\lambda \ll \mu$ fast unverändert und schrumpft für $\lambda \gg \mu$ stark gegen null. Der Faktor beträgt $\frac{1}{2}$ bei $\lambda = \mu$ (Punkte). Komponenten in Richtungen mit $\mu = 0$ sind für jedes $\lambda > 0$ gleich null (gestrichelt).](../img/mdl-linreg-ridge-shrinkage.svg)

<!--en:e71d251e-->
Darin liegt die Wirkung der Ridge-Regression: **Wie stark der Gewichtszerfall eine Richtung dämpft, hängt von ihrer Varianz ab. Die Schrumpfung ist also richtungsabhängig.**

<!--en:a1f86f4a-->
- In **gut bestimmten Richtungen** streuen die Daten stark ($\mu_j \gg \lambda$). Die Beobachtungen legen die Steigung genau fest, und der Schrumpfungsfaktor $\mu_j / (\mu_j + \lambda) \approx 1$ lässt die Kleinste-Quadrate-Schätzung nahezu unverändert.

<!--en:ce283d24-->
- In **schlecht bestimmten Richtungen** streuen die Daten kaum ($\mu_j \ll \lambda$). Ein Beispiel ist die Differenz $x_1 - x_2$ zwischen den Messwerten unserer beiden fast identischen Thermometer. Hier dividiert die unregularisierte Methode der kleinsten Quadrate das Rauschen durch einen winzigen Eigenwert $\mu_j$. Dadurch kann die Schätzung sehr groß werden (die Varianz des Koeffizienten beträgt $\sigma^2 / (n \mu_j)$). Die Ridge-Regression multipliziert die Schätzung dagegen mit $\mu_j / (\mu_j + \lambda) \approx \mu_j / \lambda \ll 1$ und dämpft die verrauschte Komponente.

<!--en:fe31a261-->
- In **unbeobachteten Richtungen im Nullraum** ($\mu_j = 0$) setzt die Ridge-Regression die Gewichtskomponente exakt auf $0$.

<!--en:b5c33d76-->
### Welche Richtungen unsere Daten bestimmen

<!--en:867f67b7-->
Prüfen wir diese spektrale Betrachtung an unserem Datensatz mit $d = 200$ Dimensionen, `data`. Zunächst berechnen wir die Eigenwerte $\mu_j$ von $\mathbf{S} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ für die 300 Trainingseingaben sowie die Schrumpfungsfaktoren bei $\lambda = \frac{1}{3}$. Wir lassen die Eingaben unzentriert, was das Spektrum kaum beeinflusst. Wie in :numref:`sec_linear-algebra` liefert `eigh` die Eigenwerte in aufsteigender Reihenfolge. Die Spalten der zweiten Rückgabe sind die zugehörigen Eigenvektoren. Wir zählen auch die Eigenwerte unterhalb von $\lambda$. Sie kennzeichnen die schlecht bestimmten Richtungen, in denen die Ridge-Regression den Betrag der Kleinste-Quadrate-Koordinate um mehr als die Hälfte verringert:

<!--en:c00fdb38-->
Keiner der Eigenwerte ist null: 300 Beispiele reichen aus, um alle 200 Dimensionen des Gewichtsraums aufzuspannen. Manche Richtungen sind durch die Daten jedoch viel genauer bestimmt als andere. Die Eigenwerte erstrecken sich über fast zwei Größenordnungen, von wenigen Hundertsteln bis etwa 3.2. Rund ein Viertel (50 bis 60 Richtungen) liegt unter $\lambda = \frac{1}{3}$. Die zugehörigen Schrumpfungsfaktoren reichen insgesamt von etwa 0.1 bis 0.9. Ihre Summe definiert die *effektive Parameterzahl* $\textrm{df}(\lambda) = \sum_j \mu_j/(\mu_j + \lambda)$. Hier gilt $\textrm{df}(\frac{1}{3}) \approx 127$: Trotz seiner 200 Gewichte verhält sich das regularisierte Modell so, als hätte es nur etwa 127 Parameter.

<!--en:ab762a42-->
Wie stark beeinträchtigt ein winziger Eigenwert die Kleinste-Quadrate-Schätzung? Entlang des Eigenvektors $\mathbf{v}_j$ addiert das Rauschen der Labels den Term $\frac{1}{n}(\mathbf{X}\mathbf{v}_j)^\top\boldsymbol{\epsilon}$ zu $\frac{1}{n}\mathbf{v}_j^\top\mathbf{X}^\top\mathbf{y}$. Diese zufällige Störung hat die Standardabweichung $\sigma\sqrt{\mu_j/n}$, denn $\|\mathbf{X}\mathbf{v}_j\|^2 = n\mu_j$. Die Methode der kleinsten Quadrate dividiert durch $\mu_j$. Dadurch verschiebt sich die Koordinate $\mathbf{v}_j^\top\hat{\mathbf{w}}_0$ um etwa $\sigma/\sqrt{n\mu_j}$, also um rund 0.06 bei $\mu_j = 1$, aber um ganze 0.3 bei $\mu_j = 0.04$. Die wahren Koordinaten haben nur einen Betrag von etwa $\|\mathbf{w}^*\|/\sqrt{200} = 0.1$. In den Richtungen der kleinsten Eigenwerte überwiegt bei der Kleinste-Quadrate-Schätzung also das Rauschen. Die Validierungseingaben folgen einer isotropen Standardnormalverteilung. Da $\|\mathbf{w} - \mathbf{w}^*\|_2^2 = \sum_j c_j^2$ unter orthogonalen Transformationen invariant ist, erhöht ein Fehler $c_j$ entlang von $\mathbf{v}_j$ den erwarteten Validierungsverlust um $c_j^2/2$, unabhängig von $\mu_j$. Prüfen wir das direkt: Wir projizieren den Parameterfehler $\mathbf{w} - \mathbf{w}^*$ beider trainierter Modelle auf die Eigenvektoren (`V.T @ (w - w_star)`). Dann vergleichen wir die Summe der quadrierten Fehler in den $k$ schlecht bestimmten Richtungen mit der Summe in den übrigen Richtungen:

<!--en:97e1e654-->
Die Aufteilung zeigt, wo der Fehler entsteht. Ohne Regularisierung entfällt mehr als die Hälfte des gesamten quadrierten Parameterfehlers auf etwa ein Viertel der Eigenvektoren: meist zwischen 0.8 und 2.2 gegenüber nur rund 0.5 in allen übrigen Richtungen zusammen. Mit Gewichtszerfall sinkt der Fehler in den schlecht bestimmten Richtungen um einen Faktor von zwei bis sechs, in den übrigen nur wenig. In jeder Gruppe bleiben etwa 0.3 bis 0.5 übrig. Selbst in den besser bestimmten Richtungen ist das Rauschen der Labels im Verhältnis zu den wahren Koordinaten beträchtlich. Eine Schrumpfung mit Faktoren von 0.5 bis 0.9 entfernt daher auch dort mehr Rauschen als Signal. Zur Kontrolle addieren wir zum halben quadrierten Gesamtfehler der Parameter $\sigma^2/2 = 0.5$. Das Ergebnis stimmt bei beiden Trainingsläufen ungefähr mit dem beobachteten Validierungsverlust überein. Der Gewichtszerfall hat also gleichzeitig in rund fünfzig schlecht konditionierten Richtungen dieselbe selektive Dämpfung bewirkt wie in der Differenzrichtung $x_1 - x_2$ unserer beiden Thermometer.

<!--en:b41b9fdf-->
Die spektrale Betrachtung beantwortet auch unsere frühere Frage: Warum war der Validierungsverlust ohne Strafterm schon nach zwei oder drei Epochen am niedrigsten? Beim Gradientenabstieg ohne Strafterm verringert jeder Schritt den Abstand zwischen der aktuellen Koordinate entlang von $\mathbf{v}_j$ und ihrer Kleinste-Quadrate-Lösung um den Faktor $1 - \eta\mu_j$. Schon nach einer Epoche mit 15 Schritten bei $\eta = 0.05$ hat sich der Abstand in einer Richtung mit großem Eigenwert ($\mu_j = 3.2$) um $1 - 0.84^{15} \approx 93\%$ verringert, in einer Richtung mit kleinem Eigenwert ($\mu_j = 0.04$) dagegen nur um $1 - 0.998^{15} \approx 3\%$. Nach zwei Epochen ist der Gradientenabstieg in den gut bestimmten Richtungen praktisch konvergiert. In den verrauschten Richtungen haben sich die Koordinaten kaum verändert. Darin ähnelt das Ergebnis dem Ridge-Schätzer. Diese Ähnlichkeit lässt sich beziffern. Bei Anfangsgewichten nahe null erreichen wir nach $t$ Schritten den Anteil $1 - (1 - \eta\mu_j)^t$ der Kleinste-Quadrate-Koordinate entlang von $\mathbf{v}_j$. Dieser Anteil kommt dem Ridge-Faktor $\mu_j/(\mu_j + \lambda)$ mit $\lambda \approx 1/(\eta t)$ nahe: Beide sind fast $1$ für $\mu_j \gg 1/(\eta t)$ und liegen nahe bei $\eta t \mu_j$ für $\mu_j \ll 1/(\eta t)$. Unser bester Abbruchzeitpunkt liegt bei 30 bis 45 Schritten mit $\eta = 0.05$. Damit liegt $1/(\eta t)$ zwischen $0.45$ und $0.67$, also in derselben Größenordnung wie $\lambda = \frac{1}{3}$ aus der A-priori-Verteilung. Trainieren wir weiter, passen sich die schlecht bestimmten Komponenten zunehmend an das Rauschen der Labels an, und der Validierungsverlust steigt wieder. *Early Stopping* ist somit eine eigenständige Form der Regularisierung und erreicht in diesem Experiment dieselbe Genauigkeit wie der Gewichtszerfall. In der Praxis ist es jedoch schwieriger einzusetzen: Der optimale Abbruchzeitpunkt hängt von der Lernrate und vom Minibatch-Rauschen ab. Um ihn zu finden, müssen wir bei jedem Trainingslauf durchgehend den Validierungsverlust verfolgen (Aufgabe :exref:`ex_weight-decay-early-stopping`).

<!--en:d52bb9d8-->
## Kompakte Implementierung

<!--en:ca6763fe-->
Gewichtszerfall ist beim Training neuronaler Netze so verbreitet, dass alle gängigen Deep-Learning-Frameworks die Aktualisierung :eqref:`eq_wd-update` dem Optimierer überlassen. Die Verlustfunktion braucht also keine Änderung. Das ist auch rechnerisch günstig: Der Optimierer verarbeitet ohnehin jeden Parameter einmal pro Schritt. Die Multiplikation mit dem Zerfallsfaktor verursacht daher kaum zusätzlichen Aufwand. Die Frameworks unterstützen zudem eigene Aktualisierungsregeln für einzelne Parameter. So können wir den Gewichtszerfall auf die Gewichte beschränken und die Bias-Parameter ausnehmen.

<!--en:46a15796-->
Wir leiten von `d2l.LinearRegression` die Klasse `WeightDecay` ab und übergeben den Hyperparameter `wd` für den Gewichtszerfall in `configure_optimizers` an den SGD-Optimierer des Frameworks. Den Bias nehmen wir aus, sodass entsprechend :eqref:`eq_wd-objective` nur die Gewichte $\mathbf{w}$ in den Strafterm eingehen.

<!--en:90d19965-->
In Gluon übergeben wir den Hyperparameter für den Gewichtszerfall direkt als `wd`, wenn wir eine Instanz von `Trainer` erzeugen. Gluon skaliert den Zerfall für jeden Parameter mit dessen Attribut `wd_mult`. Setzen wir beim Bias `wd_mult = 0`, ist er vom Zerfall ausgenommen.

<!--en:2d516e0f-->
Die Optimierer von PyTorch akzeptieren das Argument `weight_decay`. Wir bilden zwei Parametergruppen und wenden `weight_decay` nur auf `net.weight` an. Dadurch bleibt `net.bias` unregularisiert. Wie in :numref:`sec_linear_concise` initialisieren wir die Gewichte aus $\mathcal{N}(0, 0.01^2)$ und setzen den Bias auf null. Dazu rufen wir `weight_init` nach dem ersten Vorwärtsdurchlauf auf, in dem die Schicht ihre Parameter anlegt.

<!--en:024a5fac-->
Die Optimierer von Keras akzeptieren das Argument `weight_decay` und multiplizieren jede Variable vor jeder Aktualisierung mit $1 - \eta \cdot$ `weight_decay`. Mit dem Aufruf von `exclude_from_weight_decay` nehmen wir den Bias von der Schrumpfung aus.

<!--en:a2ebde8c-->
`optax.sgd` enthält selbst keinen Gewichtszerfall. In Optax können wir jedoch Optimierer zusammensetzen, indem wir Transformationen des Gradienten verketten. Hier addiert `add_decayed_weights` das `wd`-Fache jedes Parameters zu seinem Gradienten. Mit `masked` beschränken wir diese Transformation auf Parameter, die eine Bedingung erfüllen: Es muss sich um Arrays mit mehr als einer Dimension handeln. Damit erfassen wir die Gewichtsmatrix und nehmen den Bias-Vektor aus.

<!--en:e2aa15a0-->
Um `train_scratch(lambd, data)` exakt nachzubilden, müssen wir beide Hyperparameter umrechnen. Der eingebauten quadratischen Verlustfunktion fehlt unser Faktor $\frac{1}{2}$. Deshalb halbieren wir wie in :numref:`sec_linear_concise` die Lernrate auf 0.025. Den Zerfallsterm $\lambda\mathbf{w}$ addiert jedoch der Optimierer direkt zum Gradienten. Er wird also nicht mit dem Verlustgradienten mitskaliert. Wenn wir die Lernrate halbieren, müssen wir den Gewichtszerfall im Optimierer folglich auf $2\lambda = \frac{2}{3}$ verdoppeln. Nur so bleibt der Kontraktionsfaktor $1 - 0.025 \cdot \frac{2}{3} = 1 - \frac{1}{60}$ bei jedem Schritt exakt gleich.

<!--en:f3701270-->
Die resultierende Gewichtsnorm von rund 1.2 stimmt mit dem Ergebnis unserer eigenen Implementierung für $\lambda = \frac{1}{3}$ überein. Auch der Validierungsverlust, der etwa doppelt so hoch ausfällt, stimmt überein, wenn wir den Faktor 2 im mittleren quadratischen Fehler des Frameworks berücksichtigen. Die Trainingskurve zeigt jetzt den mittleren quadratischen Fehler ohne Strafterm, denn der Gewichtszerfall ist im Optimierer implementiert und damit von der Verlustfunktion getrennt.

<!--en:7ef42fab-->
In der Praxis genügt es also, dem Optimierer diesen einen Hyperparameter zu übergeben, um den Gewichtszerfall einzuschalten. Außerhalb synthetischer Experimente kennen wir den Rauschpegel $\sigma$ oder die Skala $\tau$ der A-priori-Verteilung selten im Voraus. Daher wählen wir $\lambda$ meist anhand der Validierungsdaten, indem wir mehrere Trainingsläufe vergleichen. Unsere Betrachtung über die Norm gilt unmittelbar für lineare Modelle, deren Gewichte die Steigungen der Funktion selbst sind. Bei nichtlinearen Funktionen ist es schwieriger, die Komplexität mathematisch genau zu erfassen. Hilberträume mit reproduzierendem Kernel (engl. reproducing kernel Hilbert spaces, RKHS) ermöglichen es, normbasierte Strafterme auf nichtlineare Funktionen zu übertragen. Klassische Kernel-Methoden eignen sich allerdings nur bedingt für sehr große Datensätze. Dennoch gehört der Gewichtszerfall für die Gewichte in den Schichten neuronaler Netze nach wie vor zu den zuverlässigsten Regularisierungsverfahren im modernen Deep Learning.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:142c3a5e-->
Addieren wir den quadratischen $\ell_2$-Strafterm $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ zur Zielfunktion für das Training, verhindert er, dass hochdimensionale Modelle das Rauschen entlang fast kollinearer Merkmalskombinationen verstärken. Der Bias $b$ bleibt dabei ohne Strafterm. Bei Minibatch-SGD bewirkt der Strafterm, dass wir $\mathbf{w}$ vor jedem Gradientenschritt mit $(1 - \eta\lambda)$ multiplizieren. Daher stammt der Name *Gewichtszerfall*. Dies entspricht einer Maximum-a-posteriori-Schätzung unter einem isotropen Gauß-Prior $w_j \sim \mathcal{N}(0, \tau^2)$ mit $\lambda = \sigma^2/(n\tau^2)$. Die spitzen Ecken der Normkugel beim $\ell_1$-Strafterm setzen Gewichte exakt auf null. Der glatte $\ell_2$-Strafterm wirkt hingegen als richtungsabhängiger Filter in der Eigenbasis von $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$. Jede Kleinste-Quadrate-Komponente erhält den Faktor $\mu_j/(\mu_j + \lambda)$. Dadurch bleiben gut durch die Daten bestimmte Richtungen ($\mu_j \gg \lambda$) erhalten, verrauschte Richtungen ($\mu_j \ll \lambda$) werden stark gedämpft und die Komponenten in unbeobachteten Richtungen des Nullraums ($\mu_j = 0$) verschwinden.

<!--en:2294243a-->
## Übungen

<!--en:f004ea9f-->
1. **Die $\ell_1$-Aktualisierung.** Ersetzen Sie den Strafterm $\frac{\lambda}{2}\|\mathbf{w}\|^2$ durch $\lambda \sum_j |w_j|$.
    1. Leiten Sie die Aktualisierungsregel für $\mathbf{w}$ bei Minibatch-SGD her. Verwenden Sie den Subgradienten $\operatorname{sign}(w_j)$ aus Aufgabe :exref:`ex_linear-regression-laplace-noise` anstelle der Ableitung von $|w_j|$.
    1. Vergleichen Sie, wie stark der Strafterm allein $w_j$ in einem Schritt nach dieser Regel und in einem Schritt gemäß :eqref:`eq_wd-update` verändert. Betrachten Sie große und kleine $|w_j|$.
    1. Die Aktualisierung aus (a) setzt $w_j$ fast nie exakt auf null. Zerlegen Sie sie in einen Gradientenschritt für $L$ und einen anschließenden Schritt für den Strafterm. Ändern Sie den zweiten Schritt so, dass er $w_j$ auf null setzt, sobald er das Vorzeichen von $w_j$ ändern würde. Erklären Sie den Zusammenhang mit der Merkmalsauswahl beim Lasso. Der modifizierte Schritt ist das *weiche Schwellenwertverfahren*, der proximale Schritt der $\ell_1$-Norm (:numref:`subsec_mdl-proximal-operators`).

<!--en:45bb8ce5-->
1. **Schrumpfung und Strafterm im Vergleich.** Gewichtszerfall lässt sich durch Schrumpfung der Gewichte vor dem Gradientenschritt oder durch Addition von $\frac{\lambda}{2}\|\mathbf{w}\|^2$ zum Verlust $L$ umsetzen.
    1. Zeigen Sie, dass die Aktualisierung $\mathbf{w} \leftarrow (1-\eta\lambda)\mathbf{w} - \eta\nabla L$ mit einem Gradientenschritt für den Verlust mit Strafterm $L + \frac{\lambda}{2}\|\mathbf{w}\|^2$ identisch ist.
    1. Nehmen Sie an, der Gradient werde vor dem Schritt koordinatenweise mit festen Konstanten $a_j > 0$ skaliert. Schrumpfen Sie zuerst die Gewichte und machen Sie dann den Gradientenschritt, erhalten Sie die erste Aktualisierung. Ein Gradientenschritt für den Verlust mit Strafterm ergibt die zweite:

       $$w_j \leftarrow (1-\eta\lambda)\, w_j - \eta a_j \,\partial_{w_j} L
       \qquad \textrm{and} \qquad
       w_j \leftarrow w_j - \eta a_j \left(\partial_{w_j} L + \lambda w_j\right).$$

       Bestimmen Sie, für welche $a_j$ die beiden Aktualisierungen übereinstimmen. Mit welchem Faktor multipliziert die zweite Aktualisierung $w_j$ in einem Schritt mit $\partial_{w_j} L = 0$? Viele Optimierer im Deep Learning skalieren den Gradienten auf diese Weise. Bei welcher der beiden Aktualisierungen bleibt $\lambda$ für alle Gewichte dieselbe Zerfallsrate?

    *Nach Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), Aufgabe 9.5.*
   :exlabel:`ex_weight-decay-shrinkage-vs-penalty`

<!--en:790c804a-->
1. [code] **Ridge-Regression als erweitertes Kleinste-Quadrate-Problem.** Zentrieren Sie die 300 Trainingseingaben und ihre Labels, sodass der Bias entfällt. Berechnen Sie den Ridge-Schätzer für $\lambda = \frac{1}{3}$ auf zwei Arten.
    1. Lösen Sie das lineare Gleichungssystem :eqref:`eq_wd-ridge-normal` direkt, zum Beispiel mit `np.linalg.solve`.
    1. Wenden Sie die gewöhnliche Methode der kleinsten Quadrate mit `lstsq` an. Verwenden Sie als Eingabematrix $\mathbf{X}$ mit darunter angefügter Matrix $\sqrt{n\lambda}\,\mathbf{I}$ und als Labels die Einträge von $\mathbf{y}$, gefolgt von $d$ Nullen.

    Prüfen Sie, ob die Ergebnisse bis auf Rundungsfehler übereinstimmen, und begründen Sie dies. Vergleichen Sie dann die Norm des Ergebnisses mit der von `train_scratch(lambd, data)` zurückgegebenen Norm. Welche Unterschiede zwischen den beiden Verfahren könnten eine Abweichung erklären?

    *Nach Hastie, Tibshirani und Friedman, [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), Aufgabe 3.12.*
   :exlabel:`ex_weight-decay-augmented-least-squares`

<!--en:293e151d-->
1. [code] **Versuchsreihe zu $\lambda$.** Trainieren Sie `WeightDecayScratch` auf `data` für $\lambda \in \{0.01, 0.1, 0.3, 1, 3, 10, 30\}$.
    1. Tragen Sie den abschließenden Validierungsverlust und die Norm von $\mathbf{w}$ gegen $\lambda$ auf einer logarithmischen Achse auf. Zeichnen Sie die Werte von `train_scratch(0, data)` als horizontale Linien ein und geben Sie das $\lambda$ mit dem niedrigsten Validierungsverlust an.
    1. Berechnen Sie für dieselben Werte $\textrm{df}(\lambda)$ aus den Eigenwerten aus :numref:`subsec_wd-shrinkage`. Tragen Sie den Validierungsverlust gegen $\textrm{df}(\lambda)$ auf. Vergleichen Sie die Form der Kurve mit :numref:`fig_capacity_vs_error`.
    1. Liegt der aus der A-priori-Verteilung abgeleitete Wert $\lambda = \frac{1}{3}$ nahe am besten Wert der Versuchsreihe? Erzeugen Sie Daten mit Rauschen $\sigma = 0.5$ und mit $\sigma = 2$. Berechnen Sie jeweils das $\lambda$ aus der A-priori-Verteilung neu und prüfen Sie, ob es der Verschiebung des besten Werts einer Versuchsreihe folgt.

<!--en:3fb0d4b3-->
1. [code] **Early Stopping und Gewichtszerfall im Vergleich.** Der Trainingslauf ohne Strafterm erreicht in diesem Abschnitt seinen niedrigsten Validierungsverlust nach zwei oder drei Epochen. Der Verlust ist dann etwa so niedrig wie am Ende des Trainingslaufs mit $\lambda = \frac{1}{3}$.
    1. Beginnt der Gradientenabstieg bei Gewichten nahe null, multipliziert er beim Verlust ohne Strafterm den Abstand der Koordinate entlang von $\mathbf{v}_j$ zu ihrem Kleinste-Quadrate-Wert bei jedem Schritt mit $1 - \eta\mu_j$. Nach $t$ Schritten hat er folglich den Anteil $1 - (1 - \eta\mu_j)^t$ dieser Koordinate erreicht. Tragen Sie diesen Faktor gegen $\mu_j$ für die Eigenwerte aus :numref:`subsec_wd-shrinkage` auf. Wählen Sie $t$ entsprechend einer, zwei und fünfzig Epochen mit je 15 Schritten bei $\eta = 0.05$. Stellen Sie dem den Ridge-Faktor $\mu_j/(\mu_j + \lambda)$ bei $\lambda = \frac{1}{3}$ gegenüber. Für welches $t$ ist die Annäherung an die Ridge-Regression am besten?
    1. Trainieren Sie `WeightDecayScratch` mit $\lambda = 0$ für zwei Epochen. Zerlegen Sie den Fehler wie in :numref:`subsec_wd-shrinkage` entlang der Eigenvektoren. Vergleichen Sie den quadratischen Fehler in den $k$ schlecht bestimmten Richtungen und den übrigen Richtungen mit den dort ausgegebenen Werten nach 50 Epochen bei $\lambda = 0$ und $\lambda = \frac{1}{3}$.
    1. Wiederholen Sie den Trainingslauf ohne Strafterm bei $\eta = 0.01$ und bei $\eta = 0.1$. Lesen Sie die beste Epoche aus `trainer.history` ab. Erklären Sie, wie sie sich verschiebt. Was erfordert Early Stopping in der Praxis, das beim Strafterm entfällt?
   :exlabel:`ex_weight-decay-early-stopping`

<!--en:0f14076e-->
1. [code] **Mehr Gewichte als Beispiele.** ● Erzeugen Sie Daten mit 200 Merkmalen und nur 20 Trainingsbeispielen in Minibatches der Größe 5. Verwenden Sie als wahre Gewichte jeweils $0.01$, als Bias $0.05$ und als Rauschen $\sigma = 0.01$.
    1. Berechnen Sie die Eigenwerte von $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$ für die 20 Trainingseingaben. Wie viele sind null und warum verändert der Gradientenabstieg die Gewichte entlang der zugehörigen Eigenvektoren nie?
    1. Trainieren Sie mit $\lambda = 0$ und mit $\lambda = 3$ jeweils 30 Epochen lang bei einer Lernrate von 0.01. Beginnen Sie jeweils einmal mit der Zufallsinitialisierung von `WeightDecayScratch` und einmal mit $\mathbf{w} = \mathbf{0}$ (übergeben Sie `sigma=0`). Sagen Sie vor dem Training die vier Validierungsverluste anhand der Norm der Anfangsgewichte in den unbestimmten Richtungen vorher. Vergleichen Sie anschließend Ihre Vorhersagen mit den Ergebnissen.
    1. Vergleichen Sie alle vier Verluste mit dem Verlust eines Modells, das stets $b^*$ vorhersagt. Was hat der Gewichtszerfall hier bewirkt und was lässt sich mit keinem Wert von $\lambda$ erreichen?
   :exlabel:`ex_weight-decay-more-weights`

<!--en:844ede37-->
[Diskussion](https://d2l.discourse.group/t/98)

<!--en:caa16659-->
[Diskussion](https://d2l.discourse.group/t/99)

<!--en:564dca37-->
[Diskussion](https://d2l.discourse.group/t/236)

<!--en:6b4232f4-->
[Diskussion](https://d2l.discourse.group/t/17979)

<!--en:c0d49fbb-->
Gewichtszerfall und spektrale Regularisierung<br>
**ℓ₂-Strafterm bei Ridge · multiplikativer Zerfall · Sparsität durch ℓ₁-Lasso · Gauß-Prior bei der MAP-Schätzung · Schrumpfung in den Eigenrichtungen**

<!--en:4119b181-->
::: {.slide title="Warum Merkmalsauswahl bei kollinearen Merkmalen scheitert"}

<!--en:9cc4c16d-->
[Das Problem der beiden Thermometer]{.kicker}

<!--en:35b2b139-->
- **Modell A ($\mathbf{w} = (1, 1)^\top$):** Gewichtet die Messwerte $x_1 \approx x_2$ mit $\|\mathbf{w}\|_2^2 = 2$ gleich und halbiert so die Rauschvarianz ihres Mittelwerts.

<!--en:3f744e10-->
- **Modell B ($\mathbf{w} = (501, -499)^\top$):** Hat die quadrierte Norm $\|\mathbf{w}\|_2^2 = 500{,}002$ und passt sich selbst geringstem Rauschen in $x_1 - x_2$ an. Eine Sensordrift von $2^\circ$ ergibt einen zusätzlichen Ausschlag von $+1{,}000$.

<!--en:a66da479-->
- **Stufenlose Kapazitätssteuerung:** Ein Strafterm für $\|\mathbf{w}\|_2^2$ erhält alle $d$ Merkmale und dämpft schlecht bestimmte Richtungen stufenlos.

<!--en:2cf057f6-->
::: {.slide title="Die Zielfunktion mit ℓ₂-Strafterm (Ridge-Regression)"}

<!--en:f724b79a-->
[Empirisches Risiko mit Strafterm]{.kicker}

<!--en:a2293c59-->
- **Lipschitz-Schranke für die Empfindlichkeit:** Eine Schranke für $\|\mathbf{w}\|_2$ begrenzt die Änderung der Ausgabe gemäß $|\mathbf{w}^\top(\mathbf{x} - \mathbf{x}')| \leq \|\mathbf{w}\|_2\|\mathbf{x} - \mathbf{x}'\|_2$.

<!--en:95ba93f3-->
- **Bias ohne Strafterm und standardisierte Merkmale:** Der Achsenabschnitt $b$ verschiebt nur das Ausgabeniveau und bleibt ohne Strafterm. Die Merkmale sollten auf Varianz eins standardisiert sein.

<!--en:a5866a01-->
::: {.slide title="Warum die ℓ₂-Regularisierung „Gewichtszerfall“ heißt"}

<!--en:e991f98c-->
[Multiplikative Schrumpfung bei jedem Schritt]{.kicker}

<!--en:6503debe-->
- **Multiplikative Kontraktion:** Vor dem Gradientenschritt $-\eta\nabla_{\mathbf{w}} L_{\mathcal{B}}$ auf den Daten schrumpft $\mathbf{w}$ bei jeder Aktualisierung in Richtung $\mathbf{0}$, und zwar mit dem Faktor $(1 - \eta\lambda) < 1$.

<!--en:8ea25100-->
- **Geometrischer Zerfall ohne Verstärkung:** Wo $\nabla_{\mathbf{w}} L_{\mathcal{B}} = \mathbf{0}$ gilt, nehmen die Gewichte exponentiell gemäß $(1 - \eta\lambda)^t$ ab.

<!--en:49101ecf-->
::: {.slide title="Geometrie der Nebenbedingungen bei ℓ₂-Ridge und Sparsität durch ℓ₁-Lasso"}

<!--en:d00aec2a-->
[Warum Koordinaten bei Lasso null werden und bei Ridge ungleich null bleiben]{.kicker}

<!--en:fd70f1b9-->
Nach der Lagrange-Dualität ist die Minimierung von $L(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|_p^p$ gleichwertig mit der Minimierung von $L(\mathbf{w})$ innerhalb der Normkugel $\|\mathbf{w}\|_p \leq r$:

<!--en:e83f8235-->
- **$\ell_2$-Ridge (glatte Kreisscheibe $\|\mathbf{w}\|_2 \leq r$):** Die elliptischen Niveaulinien von $L(\mathbf{w})$ um $\hat{\mathbf{w}}$ berühren den Kreis beim Anwachsen abseits der Achsen. Alle Koordinaten bleiben ungleich null, $(1, 1)^\top$ wird gegenüber $(2, 0)^\top$ bevorzugt.

<!--en:b2d90ce3-->
- **$\ell_1$-Lasso (Raute mit spitzen Ecken $\|\mathbf{w}\|_1 \leq r$):** Die $\ell_1$-Kugel $\sum_j |w_j| \leq r$ hat spitze Ecken auf den Koordinatenachsen. Die Niveaulinien des Verlusts treffen häufig zuerst eine Ecke, sodass $w_1 = 0$ gilt (**Sparsität** / Merkmalsauswahl).

<!--en:01998736-->
::: {.slide title="Bayessche Sicht auf die MAP-Schätzung mit Gauß-Prior"}

<!--en:93ba63e2-->
[Probabilistische Herleitung von λ]{.kicker}

<!--en:47ad64fc-->
- **Gauß-Prior mit Mittelwert null $w_j \sim \mathcal{N}(0, \tau^2)$:** Maximieren der A-posteriori-Dichte ergibt die Ridge-Regression mit $\lambda = \sigma^2 / (n\tau^2)$ ($1/3$ für $\sigma = 1, n = 300, \tau = 0.1$).

<!--en:e504a23b-->
::: {.slide title="Hochdimensionaler Versuch mit 200 Merkmalen und 300 Beispielen"}

<!--en:e315e401-->
[Versuchsaufbau mit eigener Implementierung]{.kicker}

<!--en:c46e546c-->
- **Vergleichswerte:** Die wahren Parameter $(\mathbf{w}^*, b^*)$ erreichen den Verlust $\frac{1}{2}\sigma^2 = 0.50$, der konstante Prädiktor mit $\mathbf{w} = \mathbf{0}$ hingegen $\frac{1}{2}(\|\mathbf{w}^*\|_2^2 + \sigma^2) = 1.50$.

<!--en:d7956d5f-->
::: {.slide title="Ohne Regularisierung (λ = 0) und mit Gewichtszerfall gemäß Prior (λ = 1/3)"}

<!--en:89652da5-->
[Gewichtszerfall gegen Überanpassung]{.kicker}

<!--en:74f6b6bb-->
Für $\sigma = 1.0, n = 300, \tau = 0.1$ ergibt die Formel des Gauß-Priors $\lambda = \sigma^2 / (n\tau^2) = 1 / (300 \times 0.01) = \frac{1}{3}$:

<!--en:75036030-->
- **Ohne Strafterm ($\lambda = 0$):** Der Trainingsverlust fällt auf $\sim 0.25$ (weit unter die Rauschgrenze von $0.50$). Der Validierungsverlust steigt auf $1.1$–$1.9$ und ist oft ebenso hoch wie beim konstanten Prädiktor mit $1.50$.

<!--en:bf43ddd6-->
- **Mit Strafterm ($\lambda = 1/3$):** Der Validierungsverlust stabilisiert sich bei $0.8$–$1.0$, und die Gewichtsnorm bleibt klein ($\|\hat{\mathbf{w}}\|_2 \approx 1.2$).

<!--en:993f4194-->
::: {.slide title="Spektrale Schrumpfung in der Eigenbasis von XᵀX/n"}

<!--en:680c30fc-->
[Warum Gewichtszerfall richtungsabhängig filtert]{.kicker}

<!--en:ec37b8fe-->
- **Varianzabhängige Filterung:** Gut bestimmte Richtungen ($\mu_j \gg \lambda$) bleiben erhalten, verrauschte kollineare Richtungen ($\mu_j \ll \lambda$) werden stark gedämpft, Komponenten im Nullraum ($\mu_j = 0$) auf null gesetzt.

<!--en:665447b2-->
::: {.slide title="Der richtungsabhängige Schrumpfungsfaktor μ/(μ + λ)"}

<!--en:343e4f78-->
[Varianzabhängige Filterung]{.kicker}

<!--en:3c7644a3-->
- **Schwelle für die Halbierung ($\mu_j = \lambda$):** Entspricht die empirische Merkmalsvarianz in einer Eigenrichtung genau $\lambda$, halbiert sich ihr Kleinste-Quadrate-Koeffizient ($\frac{\mu_j}{\mu_j + \lambda} = \frac{1}{2}$).

<!--en:e3497bb9-->
- **Komponenten im Nullraum ($\mu_j = 0$):** In jeder unbeobachteten Richtung (bei $d > n$ oder identischen Merkmalen) ist die Komponente exakt $0$.

<!--en:4ac8d736-->
- **Die beiden Thermometer:** Für $x_1 \approx x_2$ hat die Summenrichtung $\mathbf{v}_+ = (1, 1)^\top/\sqrt{2}$ den Eigenwert $\mu_+ \approx 2 \gg \lambda$ und bleibt erhalten. Die Differenzrichtung $\mathbf{v}_- = (1, -1)^\top/\sqrt{2}$ mit $\mu_- \approx 10^{-6} \ll \lambda$ wird nahezu ausgelöscht, sodass $\hat{w}_1 \approx \hat{w}_2 \approx 1$ gilt.

<!--en:510ac963-->
::: {.slide title="Spektrum und Fehlerverteilung bei 200 Merkmalen"}

<!--en:f29b11c5-->
[In welchen Richtungen der Fehler entsteht]{.kicker}

<!--en:6a2c0ce3-->
- **Selektive Rauschunterdrückung:** Etwa $k \approx 55$ Eigenwerte liegen unter $\lambda = 1/3$ ($\operatorname{df} \approx 127$). Mit $\lambda = 1/3$ sinkt der quadratische Fehler in diesen verrauschten Richtungen um den Faktor $2\times$–$6\times$.

<!--en:3f1b23b9-->
::: {.slide title="Warum SGD ohne Regularisierung früh am besten abschneidet: Early Stopping"}

<!--en:136eff50-->
[Spektrale Dynamik des Gradientenabstiegs]{.kicker}

<!--en:726e8dea-->
- **Tempo je nach Eigenwert:** Entlang von $\mathbf{v}_j$ multipliziert jeder Schritt den Abstand zu $\hat{w}_{0,j}$ mit $(1 - \eta\mu_j)$. Nach 1 Epoche ($15$ Schritte bei $\eta = 0.05$) ist der Abstand für $\mu_j = 3.2$ um $93\%$ gesunken, für $\mu_j = 0.04$ erst um $3\%$.

<!--en:96c00333-->
- **Early Stopping als spektraler Filter:** Ein Abbruch nach $T$ Schritten wendet den Filter $1 - (1 - \eta\mu_j)^T \approx \mu_j / (\mu_j + 1/(\eta T))$ an und nähert so die Wirkung der Ridge-Regression ohne Strafterm an.

<!--en:d03d2765-->
::: {.slide title="Kompakte Implementierung des Gewichtszerfalls im Optimierer"}

<!--en:e9edcc83-->
[Parametergruppen und Masken in den Frameworks]{.kicker}

<!--en:2b37fcbb-->
- **Ausnahme für den Bias:** Der Optimierer lässt die Gewichte schrumpfen, während der Bias ohne Strafterm bleibt.

<!--en:ca7208b3-->
::: {.slide title="Anpassung der Hyperparameter bei mittlerem quadratischem Fehler ohne Faktor 1/2"}

<!--en:40f83385-->
[Überprüfung des Trainings mit der kompakten Implementierung]{.kicker}

<!--en:a7b7a9a6-->
Beim mittleren quadratischen Fehler des Frameworks fehlt der Faktor $\frac{1}{2}$ im Datenverlust. Dadurch verdoppelt sich der Datengradient ($2\nabla L$). Beim Gewichtszerfall addiert der Optimierer `wd` $\cdot\,\mathbf{w}$ direkt zum Gradienten, ohne Umweg über den Verlust:

<!--en:357f75db-->
- **`lr` halbieren, `wd` verdoppeln:** Der Gradientenschritt auf den Daten $-\eta \nabla L$ und der multiplikative Zerfallsfaktor je Schritt $(1 - \eta\lambda) = 1 - 0.05 \times \frac{1}{3} = 1 - \frac{1}{60}$ bleiben bei `lr = 0.025` ($\eta/2$) und `wd = 2 * lambd` ($2/3$) erhalten.

<!--en:797efa52-->
- **Identische Lösung:** Die kompakte Implementierung erreicht dieselbe Gewichtsnorm $\|\hat{\mathbf{w}}\|_2 \approx 1.2$ und dieselbe Genauigkeit auf den Validierungsdaten.

<!--en:7dbe50c1-->
::: {.slide title="Verständnisfragen zu Prior, Schrumpfung und adaptivem Gewichtszerfall"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:807bdf5d-->
Drei Fragen zum Nachrechnen:

<!--en:d05c6f63-->
1. **Verlust durch Schrumpfung der wahren Gewichte:** Welcher Validierungsverlust ergibt sich im Versuch mit 200 Merkmalen ($\|\mathbf{w}^*\|_2^2 = 2.0$, erwarteter Validierungsverlust $\frac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|_2^2 + 0.5$) für $\mathbf{w} = 0.8\,\mathbf{w}^*$?

<!--en:6849d372-->
2. **Effektive Parameterzahl bei $\lambda = \mu_j$:** Eine Entwurfsmatrix mit $10$ Merkmalen hat die Eigenwerte $\mu = 2.0$ ($5\times$), $0.5$ ($3\times$) und $0$ ($2\times$). Wie groß ist $\operatorname{df}(\lambda)$ bei $\lambda = 0.5$?

<!--en:545920f3-->
3. **Entkoppelter Gewichtszerfall (AdamW gegenüber $\ell_2$-Strafterm):** Ein adaptiver Optimierer konditioniert den Gradienten jeder Koordinate mit $a_j > 0$ vor. Die entkoppelte Aktualisierung $w_j \leftarrow (1 - \eta\lambda)w_j - \eta a_j \partial_{w_j} L$ und die gekoppelte Aktualisierung $w_j \leftarrow w_j - \eta a_j (\partial_{w_j} L + \lambda w_j)$ vergleichen: Bei welcher zerfallen alle Gewichte unabhängig von $a_j$ mit derselben Rate?

<!--en:557b98b6-->
**Lösungen:**

<!--en:297edb21-->
1. Schrumpft $\mathbf{w}^*$ um $20\%$, ergibt sich $\frac{1}{2}\|(0.8 - 1)\mathbf{w}^*\|_2^2 + 0.5 = \frac{1}{2}(0.04)(2.0) + 0.5 = 0.54$. Das liegt nur $0.04$ über der Untergrenze $0.50$ bei wahren Parametern. Die Anpassung an das Rauschen ohne Regularisierung kostete dagegen $> 0.70$.

<!--en:9e6a6bf3-->
2. $\operatorname{df}(0.5) = 5 \cdot \frac{2.0}{2.0 + 0.5} + 3 \cdot \frac{0.5}{0.5 + 0.5} + 2 \cdot \frac{0}{0 + 0.5} = 5(0.8) + 3(0.5) + 0 = 5.5$ effektive Parameter von insgesamt $10$.

<!--en:300dc7f0-->
3. Die **entkoppelte Aktualisierung** (`AdamW`) multipliziert jedes Gewicht mit demselben Faktor $(1 - \eta\lambda)$. Die gekoppelte Aktualisierung multipliziert Koordinate $j$ mit $(1 - \eta a_j \lambda)$. Koordinaten mit großen Gradienten in früheren Schritten ($a_j \ll 1$) schrumpfen dadurch zu wenig.

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung der wichtigsten Ergebnisse"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:5a8c4df5-->
| Sichtweise auf den Gewichtszerfall | Mathematische Formulierung | Wirkungsweise |

<!--en:9a18c861-->
| **Optimierung mit Strafterm** | $\min_{\mathbf{w}, b} L(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$ | Beschränkt die Lipschitz-Konstante $\|\mathbf{w}\|_2$. Der Bias $b$ bleibt stets ohne Strafterm. Die Merkmale müssen standardisiert sein. |

<!--en:d74fc69a-->
| **Multiplikativer Schritt bei SGD** | $\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \eta \nabla_{\mathbf{w}} L_{\mathcal{B}}$ | Bei jedem Schritt schrumpft $\mathbf{w}$ in Richtung $\mathbf{0}$ mit dem Faktor $1 - \eta\lambda$, bevor der Datengradient zum Einsatz kommt. |

<!--en:00b36fae-->
| **Geometrie der Nebenbedingungen ($\ell_2$ und $\ell_1$)** | Glatte Kugel $\|\mathbf{w}\|_2 \leq r$ gegenüber Raute $\|\mathbf{w}\|_1 \leq r$ | $\ell_2$-Ridge verteilt das Gewicht auf korrelierte Merkmale $(1,1)$. Beim $\ell_1$-Lasso liegen die getroffenen Ecken auf den Achsen (Sparsität). |

<!--en:68a148ea-->
| **Gauß-Prior bei der MAP-Schätzung** | $w_j \sim \mathcal{N}(0, \tau^2),\, \epsilon \sim \mathcal{N}(0, \sigma^2) \Rightarrow \lambda = \frac{\sigma^2}{n\tau^2}$ | Die optimale Strafe ist proportional zum Verhältnis von Rausch- zu Signalvarianz $\sigma^2/\tau^2$ und nimmt wie $1/n$ ab. |

<!--en:158b27bf-->
| **Spektrale Schrumpfung** | $\mathbf{v}_j^\top\hat{\mathbf{w}}_\lambda = \frac{\mu_j}{\mu_j + \lambda}(\mathbf{v}_j^\top\hat{\mathbf{w}}_0)$, $\operatorname{df}(\lambda) = \sum_j \frac{\mu_j}{\mu_j+\lambda}$ | Gut bestimmte Richtungen ($\mu_j \gg \lambda$) bleiben erhalten, Rauschen in schlecht konditionierten Richtungen ($\mu_j \ll \lambda$) wird stark gedämpft. |
