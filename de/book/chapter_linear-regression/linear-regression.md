<!--en:a44c40a3-->
# Lineare Regression

<!--en:1fb08372-->
Angenommen, Sie möchten den Verkaufspreis eines Hauses aus Wohnfläche und Alter schätzen. Die Klassifikation ordnet eine Eingabe einer diskreten Kategorie zu. Die *Regression* sagt dagegen eine stetige numerische Größe vorher. Die Suche nach einer Regel, die quantitative Eingaben auf eine numerische Ausgabe abbildet, hat eine mehr als zweihundertjährige Geschichte. Im Jahr 1805 veröffentlichte Adrien-Marie Legendre die *Methode der kleinsten Quadrate* (engl. method of least squares), um aus verrauschten astronomischen Beobachtungen Kometenbahnen zu bestimmen :cite:`Legendre.1805`. Vier Jahre später gab Carl Friedrich Gauß dem Verfahren eine wahrscheinlichkeitstheoretische Begründung. Er fragte, für welche Verteilung der Messfehler die Lösung der kleinsten Quadrate am wahrscheinlichsten ist, und gelangte so zur Gauß-Verteilung :cite:`Gauss.1809`. Bereits 1801 hatte er die Methode genutzt, um den neu entdeckten Zwergplaneten Ceres wiederzufinden, nachdem ihn das Sonnenlicht überstrahlt hatte.

<!--en:efe96852-->
Heute reicht die Regression von der klinischen Risikobewertung und der Nachfrageprognose bis zu den Ausgabeschichten von Objektdetektoren, die Begrenzungsrahmen vorhersagen. An der *linearen Regression* lässt sich zudem fast jeder Grundbegriff des Deep Learning in einem Rahmen einführen, den wir vollständig von Hand analysieren können. Die Vorhersage ist hier eine affine Funktion der Eingaben. Als Berechnungsgraph betrachtet besteht die lineare Regression aus einem einzigen künstlichen Neuron, also einer *vollständig verbundenen Schicht* (engl. fully connected layer) mit einer Ausgabeeinheit. In diesem Abschnitt formulieren wir das Modell und den quadratischen Verlust. Wir lösen das Optimierungsproblem analytisch mit den Normalgleichungen und iterativ mit dem *stochastischen Gradientenabstieg mit Minibatches*, dem meistgenutzten Optimierungsverfahren im Deep Learning. Wir messen die Geschwindigkeit vektorisierter Arrayoperationen im Vergleich zu Python-Schleifen und stellen den Zusammenhang zwischen Verlustfunktion und Maximum-Likelihood-Schätzung bei verschiedenen Rauschverteilungen her.

<!--en:12b218dc-->
## Modell und Verlust

<!--en:2d4aff8a-->
### Merkmale und Labels

<!--en:aa614115-->
Um den Verkaufspreis eines Hauses anhand seiner Wohnfläche (in Quadratmetern) und seines Alters (in Jahren) zu schätzen, erstellen wir einen *Trainingsdatensatz* (engl. training dataset) aus früheren Verkäufen. Jedes erfasste Haus ist ein *Beispiel* (auch *Datenpunkt*, *Stichprobenwert* oder *Instanz*). Die Größe, die wir vorhersagen möchten, hier der Preis, heißt *Label* (oder *Zielwert*). Die messbaren Eigenschaften, auf die wir die Vorhersage stützen (Wohnfläche und Alter), heißen *Merkmale* (engl. features, auch *Kovariaten*).

<!--en:09a4dd4d-->
Unser Datensatz enthalte $n$ Beispiele. Wir nummerieren sie mit $i \in \{1, \ldots, n\}$. Dabei bezeichnet $\mathbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)})^\top \in \mathbb{R}^2$ den Merkmalsvektor des $i^{\textrm{th}}$ Hauses und $y^{(i)} \in \mathbb{R}$ seinen beobachteten Verkaufspreis. Beschreiben allgemeiner $d$ Merkmale jedes Beispiel, so gilt $\mathbf{x}^{(i)} \in \mathbb{R}^d$, und der vollständige Datensatz besteht aus den Paaren $\{(\mathbf{x}^{(1)}, y^{(1)}), \ldots, (\mathbf{x}^{(n)}, y^{(n)})\}$.

<!--en:80932ea8-->
### Das lineare Modell

<!--en:c7eea090-->
Jedes Verfahren des überwachten Lernens geht von einer *Modellklasse* (engl. model class) aus, einer parametrisierten Familie von Funktionen, die Merkmale auf Vorhersagen abbilden. Die Linearitätsannahme besagt, dass sich der Zielwert durch eine gewichtete Summe der Merkmale zuzüglich einer Konstanten annähern lässt:

<!--en:868111f1-->
Dabei sind $w_{\textrm{area}}$ und $w_{\textrm{age}}$ die *Gewichte* (engl. weights), und $b$ ist der *Bias* (auch *Offset* oder *Achsenabschnitt*). Jedes Gewicht bestimmt den marginalen Einfluss seines Merkmals, etwa um wie viel sich der vorhergesagte Preis bei gleichem Alter pro zusätzlichem Quadratmeter Wohnfläche ändert. Der Bias $b$ legt die Vorhersage fest, wenn alle Merkmale null sind. Auch wenn kein echtes Haus eine Wohnfläche von null hat, ist $b$ unverzichtbar. Ohne ihn muss die angepasste Ebene durch den Ursprung verlaufen, was die vom Modell darstellbaren Funktionen stark einschränkt. Streng genommen ist :eqref:`eq_price-area` eine *affine* Funktion der Merkmale, also eine lineare Abbildung mit anschließender konstanter Verschiebung. Im maschinellen Lernen spricht man dennoch durchweg von einem linearen Modell.

<!--en:a2a0b3c9-->
Bei $d$ Merkmalen hat die Vorhersage des Modells $\hat{y}$ (das Dach unterscheidet die Schätzung vom beobachteten Label $y$) die Form

<!--en:0c9479d2-->
Dabei fasst $\mathbf{w} = (w_1, \ldots, w_d)^\top \in \mathbb{R}^d$ die Gewichte zu einem Spaltenvektor zusammen. Um das Modell für alle $n$ Beispiele auf einmal auszuwerten, schreiben wir die Merkmalsvektoren $\mathbf{x}^{(i)}$ untereinander als Zeilen der *Entwurfsmatrix* (engl. design matrix) $\mathbf{X} \in \mathbb{R}^{n \times d}$. Die Vorhersagen $\hat{y}^{(i)}$ fassen wir zum Spaltenvektor $\hat{\mathbf{y}} \in \mathbb{R}^n$ zusammen:

<!--en:4a372166-->
Durch Broadcasting addieren wir den skalaren Bias $b$ zu jeder Komponente von $\mathbf{X}\mathbf{w}$ (:numref:`subsec_broadcasting`). :numref:`fig_linreg_design_matrix` zeigt die beteiligten Formen. Um den besten linearen Prädiktor zu finden, suchen wir nun die Parameter $(\mathbf{w}, b) \in \mathbb{R}^d \times \mathbb{R}$, für die $\hat{\mathbf{y}}$ möglichst gut mit den beobachteten Labels $\mathbf{y} = (y^{(1)}, \ldots, y^{(n)})^\top$ übereinstimmt. Oft ist der wahre Zusammenhang zwischen Merkmalen und Labels nur näherungsweise linear. Selbst wenn nicht erfasste Einflussgrößen und Messrauschen hinzukommen, liefert ein lineares Modell häufig eine schnelle, interpretierbare und überraschend konkurrenzfähige Baseline.

<!--en:e6dfb56a-->
![Die Formen in $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$ für $n = 6$ Beispiele und $d = 3$ Merkmale. Jede Zeile von $\mathbf{X}$ steht für ein Beispiel und liefert einen Eintrag von $\hat{\mathbf{y}}$ (schattiert). Der Vektor $\mathbf{w}$ enthält ein Gewicht pro Spalte. Durch Broadcasting wird der Skalar $b$ zu allen $n$ Einträgen addiert.](../img/mdl-linreg-design-matrix.svg)

<!--en:99405d98-->
### Der quadratische Verlust

<!--en:6bbf30e3-->
Bevor wir die besten Parameter suchen können, brauchen wir ein quantitatives Maß dafür, wie weit ein Kandidat $(\mathbf{w}, b)$ die beobachteten Labels verfehlt. Eine *Verlustfunktion* (engl. loss function) ordnet der Abweichung zwischen Vorhersage $\hat{y}$ und wahrem Wert $y$ eine nichtnegative Strafe zu. Stimmen beide überein, ist sie null. Bei der Regression verwenden wir meist den *quadratischen Fehler* (engl. squared error):

<!--en:84734df4-->
Der Faktor $\frac{1}{2}$ vereinfacht nur die Rechnung. Er kürzt sich mit dem Faktor $2$, der beim Ableiten des Quadrats entsteht. Die Lage des Minimums bleibt unverändert. Da die Differenz $\hat{y}^{(i)} - y^{(i)}$ quadriert wird, zählen Über- und Unterschätzungen desselben Betrags gleich stark. Große Abweichungen wiegen jedoch viel schwerer als kleine. Eine Vorhersage mit einer Abweichung um $3$ verursacht neunmal so viel Verlust wie eine Abweichung um $1$. Geometrisch betrachtet (:numref:`fig_fit_linreg`) ist jedes Residuum $\hat{y}^{(i)} - y^{(i)}$ der vorzeichenbehaftete vertikale Abstand zwischen Datenpunkt und Regressionsgerade. Der Wert $2\,l^{(i)}$ ist die Fläche des Quadrats über dieser vertikalen Strecke.

<!--en:be0f99a5-->
![Anpassung einer Geraden an Daten durch Minimierung der Summe der quadrierten vertikalen Residuen.](../img/fit-linreg.svg)

<!--en:1f3914e9-->
Um $(\mathbf{w}, b)$ auf dem gesamten Trainingsdatensatz mit $n$ Beispielen zu bewerten, mitteln wir die Verluste der einzelnen Beispiele. So erhalten wir den empirischen *Verlust* (auch *Kostenfunktion* oder *Zielfunktion*):

<!--en:972a26e2-->
Durch Division durch $n$ beziehen wir die Zielfunktion auf ein einzelnes Beispiel. Verdoppelt sich der Umfang des Datensatzes, verdoppeln sich daher weder der Zahlenwert des Verlusts noch der seiner Gradienten automatisch. Das Modell zu trainieren bedeutet, folgendes Optimierungsproblem zu lösen:

<!--en:08c6aeae-->
Da wir die Fehler quadrieren, zieht eine Vorhersage mit einer Abweichung um $10$ eine hundertmal so hohe Strafe nach sich wie eine Abweichung um $1$. Ein einziges verfälschtes Label kann daher die Anpassung beherrschen. Um das Ausmaß zu sehen, legen wir zwanzig Punkte auf die Gerade $y = 2x$ und ändern ein einziges Label auf $y^{(6)} = 10{,}000$, wie es bei einem Tippfehler passieren könnte. Dann passen wir auf zwei Arten eine Ursprungsgerade $\hat{y} = wx$ an. Zuerst verwenden wir den in geschlossener Form gegebenen Minimierer des quadratischen Verlusts $w = \sum_i x^{(i)} y^{(i)} / \sum_i (x^{(i)})^2$. Danach minimieren wir den *mittleren absoluten Fehler* (engl. mean absolute error) $\frac{1}{n}\sum_i |w x^{(i)} - y^{(i)}|$ auf einem feinen Gitter möglicher Steigungen:

<!--en:5520ff09-->
Ein einziger verfälschter Punkt von zwanzig verschiebt die Steigung beim quadratischen Verlust von $2.00$ bis auf $22.88$. Beim absoluten Fehler bleibt die Steigung unverändert bei $2.00$. Bei quadratischem Verlust wiegt ein einzelner Fehler von fast $10{,}000$ schwerer als alle neunzehn unverfälschten Punkte zusammen. Beim absoluten Fehler wächst sein Einfluss nur linear, und die neunzehn übrigen Punkte geben weiterhin den Ausschlag. Welches Verhalten erwünscht ist, hängt davon ab, welche Art von Rauschen wir bei unseren Labels erwarten. Dieser Frage gehen wir in :numref:`subsec_normal_distribution_and_squared_loss` von Grund auf nach.

<!--en:c632279b-->
## Minimierung des Verlusts

<!--en:d165ae79-->
Wie finden wir $(\mathbf{w}^*, b^*)$ bei $d$ Merkmalen und $n$ Beispielen? Die lineare Regression gehört zu den wenigen Modellen im maschinellen Lernen, für die es sowohl eine exakte Lösung in geschlossener Form als auch ein skalierbares iteratives Optimierungsverfahren gibt. Der Vergleich zeigt, warum im Deep Learning iterative Gradientenverfahren bei Weitem überwiegen.

<!--en:e0f552db-->
### Die Normalgleichungen

<!--en:336a4300-->
Zuerst können wir den skalaren Bias $b$ in den Gewichtsvektor einbeziehen. Wir ergänzen die Entwurfsmatrix um eine Spalte, deren Einträge alle $1$ sind. Jede Eingabe wird damit zu $(\mathbf{x}^{(i)}, 1)^\top \in \mathbb{R}^{d+1}$ und der Parametervektor zu $(\mathbf{w}, b)^\top \in \mathbb{R}^{d+1}$. Gleichwertig können wir $\mathbf{X}$ und $\mathbf{y}$ um ihre Stichprobenmittelwerte zentrieren (siehe Aufgabe :exref:`ex_linear-regression-centering`). Schreiben wir $\mathbf{X}$ und $\mathbf{w}$ in dieser erweiterten Darstellung, lautet der Verlust

<!--en:767b83e5-->
Das globale Minimum lässt sich durch quadratische Ergänzung rein algebraisch bestimmen, ganz ohne Ableitung, sofern $\mathbf{X}^\top\mathbf{X}$ invertierbar ist. Das ist genau dann der Fall, wenn die Spalten von $\mathbf{X}$ linear unabhängig sind, wofür $n \geq d$ gelten muss. Wir definieren $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$, sodass $\mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^\top\mathbf{y}$ gilt. Addieren und subtrahieren wir $\mathbf{X}\hat{\mathbf{w}}$ innerhalb der Norm und multiplizieren das Quadrat aus, ergibt sich

<!--en:3ea1556e-->
Betrachten wir den gemischten Term am Ende:

<!--en:9d1e0b82-->
Er verschwindet identisch für *jedes* $\mathbf{w}$. Übrig bleibt eine Summe aus zwei nichtnegativen quadrierten Normen. Der erste Term $\|\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}\|_2^2$ hängt überhaupt nicht von $\mathbf{w}$ ab. Der zweite Term $\|\mathbf{X}(\hat{\mathbf{w}} - \mathbf{w})\|_2^2$ ist genau dann null, wenn $\mathbf{w} = \hat{\mathbf{w}}$ gilt, denn $\mathbf{X}$ hat vollen Spaltenrang. Somit ist $\hat{\mathbf{w}}$ der eindeutige globale Minimierer. Dieselbe Gleichung erhalten wir mit Differentialrechnung, indem wir den Gradienten $\nabla_{\mathbf{w}} L(\mathbf{w}) = -\frac{1}{n}\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$ gleich null setzen. Das liefert die klassischen *Normalgleichungen* (engl. normal equations):

<!--en:c69aa201-->
Sind die Spalten von $\mathbf{X}$ linear abhängig (etwa bei zwei redundanten Merkmalen oder wenn $d > n$), ist $\mathbf{X}^\top\mathbf{X}$ singulär. Ein Minimierer im Sinne der kleinsten Quadrate existiert weiterhin, doch unendlich viele Gewichtsvektoren erzielen denselben minimalen Verlust (Aufgabe :exref:`ex_linear-regression-singular-design-matrix`).

<!--en:9b7f0e80-->
Im Code sollten Sie $(\mathbf{X}^\top\mathbf{X})^{-1}$ fast nie explizit bilden. Spezialisierte Löser für kleinste Quadrate (`lstsq`) verwenden QR- oder Singulärwertzerlegungen, die bei ähnlichem Aufwand numerisch deutlich stabiler sind. Prüfen wir die Normalgleichungen am Geradenmodell mit drei Punkten aus :numref:`sec_calculus`. Die Eingaben sind $0, 1, 2$, die Labels $1, 1, 0$ und die erweiterten Zeilen $(1, \textrm{input})$. In $\hat{\mathbf{w}} = (w_1, w_2)^\top = (7/6, -1/2)^\top$ steht also zuerst der Achsenabschnitt, dann die Steigung. Wir prüfen außerdem, ob für den Residuenvektor $\mathbf{r} = \mathbf{X}\hat{\mathbf{w}} - \mathbf{y} = (1/6, -1/3, 1/6)^\top$ die Gleichung $\mathbf{X}^\top\mathbf{r} = \mathbf{0}$ im Rahmen der Genauigkeit von `float32` gilt:

<!--en:0e7c244a-->
### Kleinste Quadrate als Projektion

<!--en:73ad7360-->
Woher kommt das Wort *normal* in Normalgleichungen? Die Identität :eqref:`eq_linreg-pythagoras` mit verschwindendem gemischtem Term ist genau der Satz des Pythagoras in $\mathbb{R}^n$ (:numref:`fig_linreg_projection`). Durchläuft $\mathbf{w}$ den Raum $\mathbb{R}^d$, überstreichen die möglichen Vorhersagevektoren $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ einen $d$-dimensionalen Unterraum von $\mathbb{R}^n$, den Spaltenraum von $\mathbf{X}$, also den Spann der Spalten. Der beobachtete Labelvektor $\mathbf{y} \in \mathbb{R}^n$ liegt meist außerhalb dieses Unterraums. Denn keine Linearkombination der $d$ Merkmalsspalten gibt alle $n$ verrauschten Beobachtungen genau wieder.

<!--en:86d49cf6-->
![Kleinste Quadrate als orthogonale Projektion in $\mathbb{R}^n$. Der Spaltenraum von $\mathbf{X}$ enthält alle Vorhersagevektoren $\mathbf{X}\mathbf{w}$. Der nächstgelegene Punkt zu $\mathbf{y}$ ist die orthogonale Projektion $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$. Ihr Residuum $\mathbf{y} - \hat{\mathbf{y}}$ steht senkrecht (normal) auf jeder Spalte von $\mathbf{X}$.](../img/mdl-linreg-projection.svg)

<!--en:d6457e04-->
Um $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$ zu minimieren, suchen wir den Punkt $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$ im Spaltenraum von $\mathbf{X}$ mit dem kleinsten euklidischen Abstand zu $\mathbf{y}$. Aus der elementaren Geometrie wissen wir, dass der nächstgelegene Punkt in einem ebenen Unterraum die *orthogonale Projektion* von $\mathbf{y}$ auf diesen Unterraum ist. Sie ist der eindeutige Punkt, dessen Differenzvektor $\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}$ senkrecht (*normal*) auf jeder Spalte von $\mathbf{X}$ steht. Die Forderung, dass $\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}$ zu allen Spalten von $\mathbf{X}$ orthogonal ist, lautet $\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}) = \mathbf{0}$. Genau das sind die Normalgleichungen.

<!--en:f52725f6-->
### Stochastischer Gradientenabstieg mit Minibatches

<!--en:0732eeac-->
Wenn die analytische Lösung der linearen Regression in eine Zeile passt, warum machen wir dann weiter? Zwei Hindernisse stehen Lösungen in geschlossener Form im modernen Deep Learning entgegen:

<!--en:dc824a33-->
1. **Skalierung des Rechenaufwands:** Zur Bildung von $\mathbf{X}^\top\mathbf{X} \in \mathbb{R}^{d \times d}$ sind $O(nd^2)$ Operationen nötig. Das Lösen des $d \times d$-Systems kostet $O(d^3)$. Bei Millionen von Beispielen und Millionen oder Milliarden Modellparametern kommt es nicht infrage, eine $d \times d$-Matrix zu speichern oder zu invertieren.

<!--en:8908cc97-->
2. **Nichtlinearität:** Sobald wir den quadratischen Verlust durch eine andere Zielfunktion ersetzen (etwa die Kreuzentropie in :numref:`chap_classification`) oder nichtlineare versteckte Schichten stapeln (:numref:`chap_perceptrons`), existiert keine Lösung in geschlossener Form mehr.

<!--en:e7daf21a-->
Stattdessen trainieren wir fast alle Deep-Learning-Modelle iterativ mit *Gradientenabstieg* (engl. gradient descent) und seinen stochastischen Varianten. Wie wir aus :numref:`sec_calculus` wissen, zeigt der Gradient $\nabla L(\mathbf{w}, b)$ in die Richtung des steilsten Anstiegs des Verlusts. Ein Schritt in die *entgegengesetzte* Richtung verringert ihn. Die Berechnung des vollständigen Gradienten $\nabla L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \nabla l^{(i)}(\mathbf{w}, b)$ erfordert jedoch für einen einzigen Parameterschritt einen Durchlauf über alle $n$ Trainingsbeispiele. Bei $n = 10^7$ ist der Gradientenabstieg mit dem gesamten Datensatz extrem langsam und statistisch verschwenderisch. Der mittlere Gradient von 100 zufällig ausgewählten Beispielen liefert bereits eine brauchbare Schätzung des Mittelwerts über 10 Millionen. Die Genauigkeit der Schätzung hängt vom Umfang der Stichprobe ab und nicht von der Größe des Datensatzes, aus dem sie stammt.

<!--en:78262fff-->
Der *stochastische Gradientenabstieg mit Minibatches* (engl. minibatch stochastic gradient descent, kurz Minibatch-SGD) nutzt diese statistische Redundanz. In jedem Schritt $t$ ziehen wir einen kleinen zufälligen *Minibatch* $\mathcal{B}_t \subset \{1, \ldots, n\}$ mit $|\mathcal{B}|$ Beispielen, mitteln die Verlustgradienten nur über $\mathcal{B}_t$ und aktualisieren die Parameter mit einer positiven *Lernrate* (engl. learning rate, auch Schrittweite) $\eta > 0$:

<!--en:b6a9571e-->
Da wir $\mathcal{B}_t$ gleichverteilt aus dem Trainingsdatensatz ziehen, ist der Minibatchgradient ein *erwartungstreuer Schätzer* (engl. unbiased estimator) des wahren Gradienten über den gesamten Datensatz. Sein Standardfehler nimmt wie $1/\sqrt{|\mathcal{B}|}$ ab. Für die lineare Regression mit dem quadratischen Verlust :eqref:`eq_mse` erhalten wir durch explizites Ableiten von $l^{(i)} = \frac{1}{2}(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)})^2$ die Aktualisierungsregel

<!--en:c4521f1e-->
Betrachten wir die einzelnen Terme. Der Skalar $\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}$ ist der vorzeichenbehaftete Vorhersagefehler für Beispiel $i$. Überschätzt das Modell den Zielwert ($\hat{y}^{(i)} > y^{(i)}$), subtrahiert die Aktualisierung ein Vielfaches von $\mathbf{x}^{(i)}$ vom Gewichtsvektor $\mathbf{w}$ und verringert $b$ ein wenig. Dadurch rückt die nächste Vorhersage näher an $y^{(i)}$. Wichtig ist auch folgende Unterscheidung: $\mathbf{w}$ und $b$ sind *Modellparameter*, die wir durch Optimierung lernen. Die Minibatchgröße $|\mathcal{B}|$ und die Lernrate $\eta$ sind dagegen *Hyperparameter*. Wir wählen sie vor Beginn des Trainings und stimmen sie anhand zurückgehaltener Validierungsdaten ab, wie wir in :numref:`sec_generalization_basics` besprechen.

<!--en:8d67b747-->
### Gradientenrauschen und Lernrate

<!--en:949f19e0-->
Wie unterscheidet sich die Bahn des Minibatch-SGD von der des Gradientenabstiegs mit dem gesamten Datensatz? Betrachten wir die Niveaulinien des empirischen Verlusts $L(\mathbf{w})$ um die Lösung der kleinsten Quadrate $\hat{\mathbf{w}}$ (:numref:`fig_linreg_gd_sgd`). Bei jedem Schritt berechnet der Gradientenabstieg den Gradienten von $L$ über alle $n$ Beispiele. Er bewegt sich stetig über die elliptischen Niveaulinien zum Minimum $\hat{\mathbf{w}}$, sofern $\eta$ nicht so groß ist, dass das Verfahren überschießt und divergiert.

<!--en:3dcf5ae0-->
Minibatch-SGD nimmt einen anderen Weg. Selbst im globalen Minimum $\hat{\mathbf{w}}$, wo der mittlere Gradient über alle $n$ Beispiele null ist, haben einzelne verrauschte Beispiele noch Residuen $\hat{y}^{(i)} - y^{(i)} \neq 0$ ungleich null. Bei einem zufälligen Minibatch $\mathcal{B}_t$ heben sich die Gradienten der einzelnen Beispiele fast nie genau auf. Bei konstanter Lernrate $\eta$ steigt Minibatch-SGD daher rasch in eine kleine Umgebung von $\hat{\mathbf{w}}$ ab und schwankt dann fortwährend in einer *Kugel um das Minimum* (engl. noise ball) $\hat{\mathbf{w}}$. Ihr quadrierter Radius skaliert wie $\eta / |\mathcal{B}|$.

<!--en:2656cc7a-->
![Der Gradientenabstieg mit dem gesamten Datensatz (schwarz) folgt gleichmäßig der Abstiegsrichtung von $L$ bis zum Minimum $\hat{\mathbf{w}}$. Jeder Schritt durchläuft alle $n$ Beispiele. Minibatch-SGD (farbig) macht pro Durchlauf $n/|\mathcal{B}|$ kostengünstige Schritte mit verrauschten Gradientenschätzungen und erreicht rasch eine kleine Kugel um $\hat{\mathbf{w}}$. Bei einem Viertel der Lernrate (rechts) schrumpft die Kugel.](../img/mdl-linreg-gd-sgd.svg)

<!--en:d36fbf7d-->
Halbieren wir $\eta$ oder verdoppeln wir $|\mathcal{B}|$, schrumpft die Kugel, in der die Iterierten schließlich schwanken. Verringern wir $\eta \to 0$ im Lauf des Trainings, können sich die Iterierten bei $\hat{\mathbf{w}}$ einpendeln (:numref:`chap_optimization`). In der Praxis ist es allerdings selten sinnvoll, das Minimum des Trainingsverlusts auf zehn Nachkommastellen genau zu treffen. Letztlich wollen wir einen kleinen Fehler auf *ungesehenen* Daten. Schon $\hat{\mathbf{w}}$ selbst weicht aufgrund des statistischen Schätzfehlers in der Größenordnung $1/\sqrt{n}$ von den wahren Parametern der Grundgesamtheit ab. Kleine Schwankungen bei der Optimierung innerhalb dieser statistischen Unsicherheit kosten daher keine Generalisierungsgüte.

<!--en:8723eb32-->
### Höhere Geschwindigkeit durch Vektorisierung

<!--en:98621d3d-->
Wenn Sie die Minibatch-Aktualisierungen :eqref:`eq_linreg_batch_update` implementieren, sollten Sie die Beispiele $i \in \mathcal{B}_t$ nie einzeln in einer Python-Schleife durchlaufen. Stattdessen *vektorisieren* wir die Rechnung. Wir formulieren den gesamten Vorwärtsdurchlauf und den Gradientenschritt für einen Minibatch als Array- und Matrixoperationen. C++- oder CUDA-Bibliotheken führen diese parallel auf SIMD-Vektoreinheiten oder GPU-Kernen aus.

<!--en:f02b9c5e-->
Messen wir den Unterschied, indem wir die Labels von $n = 1{,}000$ Beispielen mit den Gewichten $\mathbf{w} = (2, -3.4)^\top$ und dem Bias $b = 4.2$ vorhersagen. Zuerst berechnen wir ein Beispiel nach dem anderen in einer `for`-Schleife in Python, dann alle $1{,}000$ Beispiele gleichzeitig mit einem einzigen Matrixprodukt `X @ w + b`:

<!--en:8ba88358-->
Unsere Hilfsfunktion zur Zeitmessung `seconds` führt zunächst einen Probelauf aus, damit einmalige Kosten für die Einrichtung und die JIT-Kompilierung nicht in die Messung eingehen. Bevor sie die Uhr anhält, ruft sie `wait` auf. Frameworks für GPUs und andere Beschleuniger stoßen Operationen asynchron an und können die Kontrolle an Python zurückgeben, während die Hardware noch rechnet.

<!--en:30075075-->
Beide Varianten führen dieselben Rechenoperationen aus. Dennoch ist die Berechnung aller $1{,}000$ Vorhersagen in einem vektorisierten Aufruf Hunderte Male schneller. Bei der Schleife fallen für jedes einzelne Beispiel Aufwand für den Python-Interpreter und Latenz beim Kernel-Start an. Bei der Matrixmultiplikation fällt dieser Aufwand nur einmal an, und der gesamte Batch geht an eine einzige kompilierte Routine der linearen Algebra. In diesem Buch ist die Vektorisierung über Minibatches die wichtigste Gewohnheit für schnellen Deep-Learning-Code: Aus $n$ Skalarprodukten $\mathbf{w}^\top\mathbf{x}^{(i)}$ wird ein Matrix-Vektor-Produkt $\mathbf{X}\mathbf{w}$.

<!--en:928a6b07-->
## Quadratischer Verlust und gaußsches Rauschen

<!--en:5060e633-->
Bisher haben wir den quadratischen Verlust $\frac{1}{2}(\hat{y} - y)^2$ mit seiner anschaulichen Geometrie und seinen einfachen Ableitungen begründet. Stellen wir nun die weiterführende Frage von Gauß: *Unter welcher probabilistischen Annahme über die Daten ist es statistisch optimal, den quadratischen Fehler zu minimieren?*

<!--en:72bae4b6-->
### Die Normalverteilung

<!--en:e56cf8da-->
Erinnern wir uns an die Normalverteilung (oder Gauß-Verteilung) mit Mittelwert $\mu$ und Varianz $\sigma^2$ (Standardabweichung $\sigma$). Ihre Wahrscheinlichkeitsdichte lautet

<!--en:2630cf8a-->
Die Dichte bildet eine symmetrische Glockenkurve um $\mu$. Ihre Breite bestimmt die Standardabweichung $\sigma$. Etwa 68% der Wahrscheinlichkeit liegen in einem Abstand von höchstens $\sigma$ vom Mittelwert, etwa 95% in einem Abstand von höchstens $2\sigma$. Da die *quadrierte* Abweichung $(x - \mu)^2$ mit negativem Vorzeichen im Exponenten steht, sind extreme Abweichungen verschwindend unwahrscheinlich. Im Abstand $5\sigma$ vom Mittelwert beträgt die Dichte das $e^{-12.5} \approx 4 \times 10^{-6}$-Fache ihres Spitzenwerts. Mit $\epsilon \sim \mathcal{N}(\mu, \sigma^2)$ bezeichnen wir eine normalverteilte Zufallsvariable. Summen vieler kleiner unabhängiger Störungen streben einer Normalverteilung zu (:numref:`subsec_mdl-gaussian-clt`). Daher ist die Gauß-Verteilung das naheliegende Ausgangsmodell für gewöhnliches Messrauschen.

<!--en:0bcbd8c4-->
Die lineare Regression mit additivem gaußschem Rauschen nimmt formal an, dass jedes beobachtete Label die Summe aus dem linearen Signal und einem unabhängigen normalverteilten Rauschterm ist:

<!--en:2fefe4bb-->
In diesem Modell ist der bedingte Mittelwert $\mathbb{E}[y \mid \mathbf{x}]$ tatsächlich gleich $\mathbf{w}^\top \mathbf{x} + b$, in Übereinstimmung mit :numref:`subsec_linear_model`. Die Labels streuen symmetrisch um die Gerade, und das Rauschen hat dieselbe Stärke $\sigma$ bei jeder Eingabe $\mathbf{x}$ (:numref:`fig_linreg_gaussian_noise`).

<!--en:2201da38-->
![Das gaußsche Rauschmodell. Für jede Eingabe $x$ wird das Label $y$ gemäß der Normaldichte $p(y \mid x)$ gezogen. Sie hat ihr Zentrum auf der Geraden $wx + b$ und ist überall gleich breit.](../img/mdl-linreg-gaussian-noise.svg)

<!--en:73a5f16e-->
### Maximum-Likelihood-Schätzung

<!--en:bde000ad-->
Unter diesem additiven gaußschen Rauschmodell ist bei gegebenen Merkmalen $\mathbf{x}$ die bedingte Verteilung des beobachteten Labels $y$ eine Normalverteilung mit Mittelwert $\mathbf{w}^\top\mathbf{x} + b$ und Varianz $\sigma^2$:

<!--en:9e960224-->
Nach dem *Maximum-Likelihood-Prinzip* sollten wir $(\mathbf{w}, b)$ so wählen, dass die bedingte Wahrscheinlichkeitsdichte des gesamten beobachteten Datensatzes $\{y^{(1)}, \ldots, y^{(n)}\}$ bei gegebenem $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$ maximal wird. Wurden die Beispiele unabhängig gezogen, zerfällt die Likelihood des Datensatzes in das Produkt $\prod_{i=1}^n P(y^{(i)} \mid \mathbf{x}^{(i)})$. Ein positives Produkt zu maximieren ist gleichbedeutend damit, seinen negativen Logarithmus zu minimieren, die *negative Log-Likelihood*:

<!--en:075e341c-->
Betrachten wir die beiden Terme auf der rechten Seite. Der erste Term $\frac{n}{2}\log(2\pi\sigma^2)$ hängt weder von $\mathbf{w}$ noch von $b$ ab. Der zweite Term ist $\frac{n}{\sigma^2} L(\mathbf{w}, b)$, also unser mittlerer quadratischer Verlust multipliziert mit der positiven Konstanten $n/\sigma^2$. Wer den quadratischen Verlust minimiert, erhält folglich bei unabhängigem homoskedastischem gaußschem Rauschen den *Maximum-Likelihood-Schätzer* für $(\mathbf{w}, b)$, unabhängig vom Wert von $\sigma^2$.

<!--en:994708f3-->
### Wahl der Verlustfunktion anhand eines Rauschmodells

<!--en:d0a5fce2-->
Was bewirkt der quadratische Verlust, wenn das Rauschen *nicht* gaußsch ist? Die Antwort kommt ganz ohne Likelihood aus. Halten wir eine Eingabe $\mathbf{x}$ fest und fragen wir, welche einzelne Zahl $c$ den erwarteten quadratischen Fehler $\mathbb{E}[(y - c)^2 \mid \mathbf{x}]$ minimiert. Setzen wir die Ableitung $2(c - \mathbb{E}[y \mid \mathbf{x}])$ gleich null, erhalten wir den bedingten Mittelwert $c = \mathbb{E}[y \mid \mathbf{x}]$, und zwar für jede Rauschverteilung mit endlicher Varianz. Den erwarteten absoluten Fehler $\mathbb{E}[|y - c| \mid \mathbf{x}]$ minimiert dagegen der bedingte *Median*. Gewichten wir den Fehler asymmetrisch, sodass jede Einheit der Unterschätzung $q$ und jede Einheit der Überschätzung $1 - q$ kostet, so ist das $q$-Quantil der Minimierer. In der Prognose, wo das 90%-Quantil der Nachfrage oft wichtiger ist als ihr Mittelwert, heißt diese Funktion *Pinball-Verlust*. Das verfälschte Label in :numref:`subsec_linear-regression-loss-function` hat den Unterschied gezeigt: Ein einziger sehr großer Wert verschiebt den Mittelwert, während der Median ihn ignoriert. Mit der Verlustfunktion wählen wir also, welche Kenngröße von $y$ das Modell vorhersagt, ganz gleich, wie das Rauschen tatsächlich verteilt ist.

<!--en:5c0b6ed5-->
Der Zusammenhang mit der Maximum-Likelihood-Schätzung liefert ein systematisches Verfahren zum Entwurf von Verlustfunktionen, wenn gaußsches Rauschen *nicht* das passende Modell für die Wirklichkeit ist (:numref:`fig_linreg-loss-menu`):

<!--en:0f1fd16e-->
| Art der Ausgabe | Rauschmodell | Verlust (negative Log-Likelihood) | Vorhergesagte Kenngröße |

<!--en:63c4f684-->
| reellwertig, gewöhnliches Rauschen | Gauß-Verteilung | quadratischer Fehler (dieser Abschnitt) | Mittelwert von $y$ |

<!--en:bb73e177-->
| reellwertig, Ausreißer oder schwere Verteilungsränder | Laplace-Verteilung | absoluter Fehler (Aufgabe :exref:`ex_linear-regression-laplace-noise`) | Median von $y$ |

<!--en:58dcb058-->
| positiv, multiplikative Schwankungen (Preise) | Gauß-Verteilung für $\log y$ | quadratischer Fehler für $\log y$ (Aufgabe :exref:`ex_linear-regression-beyond-gaussian-noise`) | Mittelwert von $\log y$ |

<!--en:8dde0231-->
| Zähldaten (Verkäufe, Ankünfte) | Poisson-Verteilung | $\nu(\mathbf{x}) - y \log \nu(\mathbf{x})$ (Aufgabe :exref:`ex_linear-regression-counting-apples`) | Mittelwert von $y$ |

<!--en:0c768553-->
- **Ausreißer bei schweren Verteilungsrändern (Laplace-Rauschen $\rightarrow$ $\ell_1$-Verlust):** Führen vereinzelte fehlerhafte Messungen zu extremen Ausreißern, kann ein einziger falscher Datenpunkt die angepasste Gerade zu sich ziehen, weil $(\hat{y} - y)^2$ quadratisch wächst. Modellieren wir das Rauschen stattdessen mit den schwereren, exponentiell abfallenden Rändern der Laplace-Verteilung $p(\epsilon) = \frac{1}{2\tau}\exp(-|\epsilon|/\tau)$, so ergibt sich aus der negativen Log-Likelihood der *absolute Fehler* ($\ell_1$) als Verlust $|\hat{y} - y|$ (Aufgabe :exref:`ex_linear-regression-laplace-noise`). Sein Gradient hat unabhängig vom Abstand des Ausreißers den konstanten Betrag $\pm 1$.

<!--en:e2e5578b-->
- **Glattes Optimum und Robustheit gegenüber Ausreißern (Huber-Verlust):** Bei null hat $|\hat{y} - y|$ einen scharfen Knick. Die Ableitung springt dort zwischen $-1$ und $+1$. In der Praxis verwendet man daher häufig den *Huber-Verlust*. Er verbindet eine quadratische Schale für kleine Residuen ($|\hat{y} - y| \leq \delta$) mit linear ansteigenden Ästen für große Residuen ($|\hat{y} - y| > \delta$).

<!--en:dc8379fd-->
- **Zähldaten (Poisson-Rauschen):** Ist der Zielwert $y \in \{0, 1, 2, \ldots\}$ eine diskrete Anzahl, etwa die Zahl der täglich eintreffenden Kunden, führt eine Poisson-Likelihood auf eine eigene konvexe negative Log-Likelihood (Aufgabe :exref:`ex_linear-regression-counting-apples`).

<!--en:76fe0f6d-->
![Drei Verlustfunktionen für die Regression (rechts) und die Rauschdichten $p(\epsilon) \propto \exp(-l(\epsilon))$, denen die ersten beiden bei der Maximum-Likelihood-Schätzung entsprechen (links): quadratischer Fehler (Gauß-Verteilung) und absoluter Fehler (Laplace-Verteilung). Dazwischen liegt der Huber-Verlust. Der Ausschnitt in logarithmischer Skalierung zeigt die schwereren Ränder der Laplace-Verteilung.](../img/mdl-linreg-loss-menu.svg)

<!--en:f32be0ae-->
## Lineare Regression als neuronales Netz

<!--en:795cd6b5-->
Um die Verbindung zu den tiefen Architekturen späterer Kapitel herzustellen, stellen wir die lineare Regression als einschichtiges neuronales Netz dar (:numref:`fig_single_neuron`). Im Diagramm bilden die Eingabemerkmale $x_1, \ldots, x_d$ die Knoten der *Eingabeschicht* (engl. input layer). Jeder Knoten ist mit seinem eigenen synaptischen Gewicht $w_1, \ldots, w_d$ an dieselbe Recheneinheit $o_1$ in der *Ausgabeschicht* (engl. output layer) angeschlossen. Diese summiert die gewichteten Eingaben, addiert den Bias $b$ und gibt $\hat{y} = o_1$ aus.

<!--en:886e9ec2-->
![Die lineare Regression ist ein einschichtiges neuronales Netz mit einem einzigen künstlichen Neuron.](../img/singleneuron.svg)

<!--en:a61d2166-->
Nach üblicher Konvention zählen wir nur Schichten mit einstellbaren Parametern. Die lineare Regression ist daher ein *einschichtiges* Netz. Da jedes Eingabemerkmal mit dem Ausgabeknoten verbunden ist, heißt diese Schicht *vollständig verbundene Schicht* (auch *dichte Schicht*). Ähnlich führen einzelne Neuronen der Großhirnrinde auf ihren Dendriten gewichtete Signale von Tausenden Synapsen zusammen. Diese Analogie inspirierte frühe Pioniere wie McCulloch und Pitts :cite:`McCulloch.Pitts.1943`. Unser Interesse gilt in diesem Buch jedoch der Mathematik und der Technik, mit denen wir solche Schichten zu tiefen Netzen verbinden.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:e0306239-->
Die lineare Regression sagt einen stetigen Zielwert als affine Funktion der Merkmale vorher, $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$. Sie bewertet die Abweichung durch den mittleren quadratischen Fehler $L(\mathbf{w}, b) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|_2^2$. Geometrisch projiziert die Lösung der Normalgleichungen $\mathbf{X}^\top\mathbf{X}\mathbf{w}^* = \mathbf{X}^\top\mathbf{y}$ den Vektor $\mathbf{y}$ orthogonal auf den Spaltenraum von $\mathbf{X}$. Das Aufstellen und Lösen dieses $d \times d$-Systems kostet jedoch $O(nd^2 + d^3)$ Operationen, und das Verfahren lässt sich nicht auf nichtlineare Netze übertragen. Der stochastische Gradientenabstieg mit Minibatches aktualisiert die Parameter stattdessen durch vektorisierte Matrixoperationen auf kleinen zufälligen Batches $\mathcal{B}_t$. Mit einem Aufwand von $O(|\mathcal{B}|d)$ pro Schritt erreicht er rasch eine kleine Kugel um das Optimum. Statistisch entspricht die Minimierung des quadratischen Fehlers der Maximum-Likelihood-Schätzung bei additivem gaußschem Rauschen. Ersetzt man das Gauß-Modell durch Verteilungen mit schwereren Rändern oder durch diskrete Verteilungen, erhält man andere Verlustfunktionen, etwa den absoluten Fehler oder die negative Poisson-Log-Likelihood.

<!--en:2294243a-->
## Übungen

<!--en:f495c64b-->
1. **Achsenabschnitt und Zentrierung.** Ergänzen wir jede Eingabe um ein konstantes Merkmal $1$, wird das affine Modell $\mathbf{w}^\top \mathbf{x} + b$ zu einem linearen Modell in $(\mathbf{x}, 1)$. Durch Zentrieren der Daten lässt sich der Bias auf andere Weise entfernen. Seien $\bar{\mathbf{x}}$ und $\bar{y}$ die Mittelwerte der Merkmale und der Labels über den Trainingsdatensatz.
    1. Zeigen Sie, dass jeder Minimierer $(\hat{\mathbf{w}}, \hat{b})$ des quadratischen Verlusts die Gleichung $\hat{b} = \bar{y} - \hat{\mathbf{w}}^\top \bar{\mathbf{x}}$ erfüllt. Folgern Sie daraus, dass die Summe der Fehler $\hat{y}^{(i)} - y^{(i)}$ bei einer Anpassung nach kleinsten Quadraten mit Bias null ist.
    1. Zeigen Sie, dass $\hat{\mathbf{w}}$ eine Lösung nach kleinsten Quadraten ohne Bias für die zentrierten Daten $(\mathbf{x}^{(i)} - \bar{\mathbf{x}},\, y^{(i)} - \bar{y})$ ist.
   :exlabel:`ex_linear-regression-centering`

<!--en:9959b933-->
1. **Laplace-Rauschen.** Das additive Rauschen $\epsilon$ folge der Laplace-Verteilung mit Skalenparameter $1$ (siehe :numref:`sec_mdl-distributions`). Es gelte also $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    1. Schreiben Sie die negative Log-Likelihood der Daten unter diesem Modell, $-\log p(\mathbf y \mid \mathbf X)$, explizit aus.
    1. Bestimmen Sie, ob eine Lösung in geschlossener Form existiert.
    1. Geben Sie für diesen Verlust die Aktualisierung von $(\mathbf{w}, b)$ durch Minibatch-SGD in der Form von :eqref:`eq_linreg_batch_update` an. Verwenden Sie den Wert $0$ an der Stelle null, wo $|\cdot|$ keine Ableitung besitzt, und andernorts die Steigung $\operatorname{sign}(\cdot) = \pm 1$. Einen solchen Ersatz nennt man *Subgradient*. Vergleichen Sie, wie der Betrag des Beitrags jedes Beispiels bei diesem Verlust und beim quadratischen Verlust von seinem Fehler abhängt.
    1. [code] Wenden Sie diese Aktualisierung auf die Daten mit dem verfälschten Label aus :numref:`subsec_linear-regression-loss-function` an. Verwenden Sie eine Ursprungsgerade, in jedem Schritt den gesamten Datensatz, die Lernrate 0.002 und 2000 Schritte ab $w = 0$. Der Minimierer des absoluten Fehlers ist genau $w = 2$. Wo landen die Iterierten? Erklären Sie, warum sie nicht beim Minimierer zur Ruhe kommen, und schlagen Sie eine Abhilfe vor.
   :exlabel:`ex_linear-regression-laplace-noise`

<!--en:d43e1c94-->
1. **Jenseits des gaußschen Rauschens.** Angenommen, Sie schätzen mit Regression die Preise von Häusern oder Aktien.
    1. Modellieren Sie den Preis direkt als $y = \mathbf{w}^\top \mathbf{x} + b + \epsilon$ mit $\epsilon \sim \mathcal{N}(0, \sigma^2)$ und wählen Sie $\sigma = 100{,}000$ US-Dollar. Berechnen Sie die Wahrscheinlichkeit, die das Modell einem negativen Preis für ein Haus mit einem vorhergesagten Preis von $50{,}000$ US-Dollar zuweist. Erklären Sie, warum kein einzelnes $\sigma$ sowohl für dieses Haus als auch für ein Haus mit einem vorhergesagten Preis von $5{,}000{,}000$ US-Dollar geeignet ist.
    1. Führen Sie die Regression nun für den Logarithmus des Preises durch: $\log \textrm{price} = \mathbf{w}^\top \mathbf{x} + b + \epsilon$. Dabei wirkt $\epsilon \sim \mathcal{N}(0, \sigma^2)$ auf der logarithmischen Skala. Berechnen Sie den bedingten Mittelwert und die bedingte Standardabweichung des Preises und zeigen Sie, dass ihr Verhältnis unabhängig von $\mathbf{x}$ ist. Bestimmen Sie, ob $\exp(\mathbf{w}^\top \mathbf{x} + b)$ eine erwartungstreue Vorhersage des Preises ist. Dieselbe Modellierungsentscheidung liegt dem Black-Scholes-Modell zur Bewertung von Optionen zugrunde :cite:`Black.Scholes.1973`.
   :exlabel:`ex_linear-regression-beyond-gaussian-noise`

<!--en:dcaedd89-->
1. **Äpfel zählen.** Angenommen, wir möchten mit Regression die *Anzahl* der Äpfel schätzen, die ein Lebensmittelgeschäft verkauft.
    1. Benennen Sie die Probleme eines additiven gaußschen Rauschmodells für einen solchen Zielwert.
    1. Die [Poisson-Verteilung](https://en.wikipedia.org/wiki/Poisson_distribution) modelliert Zähldaten. Sie ist gegeben durch $p(k \mid \nu) = \nu^k e^{-\nu}/k!$. Dabei ist $\nu > 0$ die Rate und $k \in \{0, 1, 2, \ldots\}$ die beobachtete Anzahl. Beweisen Sie $\mathbb{E}[k] = \nu$.
    1. Leiten Sie die negative Log-Likelihood einer Anzahl $k$ bei einer von den Merkmalen abhängigen Rate $\nu(\mathbf{x})$ her. Lassen Sie Terme weg, die nicht von den Parametern abhängen, und vergleichen Sie mit der Tabelle in :numref:`subsec_linreg-loss-menu`.
    1. Vergleichen Sie die Parametrisierungen $\nu(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ und $\log \nu(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$. Prüfen Sie für jede, ob $\nu(\mathbf{x}) > 0$ für alle Eingaben und Parameter gilt. Was folgt daraus für den Verlust aus Teil (c)? Geben Sie diesen Verlust für die zweite Parametrisierung als Funktion von $(\mathbf{w}, b)$ an.
   :exlabel:`ex_linear-regression-counting-apples`

<!--en:6eb7459c-->
1. [code] **Größe der Kugel um das Minimum.** Erzeugen Sie $n = 1{,}000$ Beispiele mit je zwei aus $\mathcal{N}(0, 1)$ gezogenen Merkmalen und den Labels $y = \mathbf{x}^\top \mathbf{w}^* + \epsilon$, wobei $\mathbf{w}^* = (2, -3.4)$ und $\epsilon \sim \mathcal{N}(0, 1)$ gelten. Berechnen Sie die Lösung der kleinsten Quadrate $\hat{\mathbf{w}}$ mit `lstsq`.
    1. Führen Sie Minibatch-SGD ohne Bias aus. Starten Sie bei $\hat{\mathbf{w}}$ und verwenden Sie Minibatches der Größe 10 und $\eta = 0.1$ für 20 000 Schritte. Schätzen Sie den mittleren quadrierten Abstand $\mathbb{E}\|\mathbf{w} - \hat{\mathbf{w}}\|^2$ der Iterierten von $\hat{\mathbf{w}}$.
    1. Wiederholen Sie den Versuch für $\eta \in \{0.05, 0.025\}$ und für die Minibatchgrößen 20 und 40. Prüfen Sie, ob der mittlere quadrierte Abstand ungefähr proportional zu $\eta / |\mathcal{B}|$ ist.
    1. Warum hängt der Abstand nicht davon ab, wie weit $\hat{\mathbf{w}}$ von $\mathbf{w}^*$ entfernt ist? Was würde sich ändern, wenn die Labels völlig rauschfrei wären?

<!--en:fbf14b38-->
1. **Singuläre Entwurfsmatrix.** ● Bei vollem Spaltenrang ist der Minimierer im Sinne der kleinsten Quadrate eindeutig. Nehmen Sie stattdessen an, dass $\mathbf{X}^\top \mathbf{X}$ singulär ist.
    1. Zeigen Sie, dass weiterhin ein Minimierer von $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$ existiert, und charakterisieren Sie die Menge aller Minimierer.
    1. Schreiben Sie die Singulärwertzerlegung als $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ mit orthogonalen Matrizen $\mathbf{U}$ und $\mathbf{V}$ und einer rechteckigen Diagonalmatrix $\boldsymbol{\Sigma}$ mit nichtnegativen Singulärwerten. Die Moore-Penrose-Pseudoinverse lautet

       $$\mathbf{X}^{\dagger} = \mathbf{V}\boldsymbol{\Sigma}^{\dagger}\mathbf{U}^\top,$$

       Dabei entsteht $\boldsymbol{\Sigma}^{\dagger}$ durch Transponieren von $\boldsymbol{\Sigma}$ und Kehrwertbildung bei den von null verschiedenen Einträgen. Zeigen Sie, dass $\mathbf{X}^{\dagger}\mathbf{y}$ innerhalb der Menge aus Teil (a) der Minimierer mit der kleinsten Norm ist.
    1. Gegen welchen Minimierer konvergiert der Gradientenabstieg beim Start in $\mathbf{w} = \mathbf{0}$? Geben Sie an, ob sich die Antwort für den stochastischen Gradientenabstieg mit Minibatches ändert.
   :exlabel:`ex_linear-regression-singular-design-matrix`

<!--en:d3b799d2-->
[Diskussion](https://d2l.discourse.group/t/40)

<!--en:d23194e9-->
[Diskussion](https://d2l.discourse.group/t/258)

<!--en:6b5d1ed8-->
[Diskussion](https://d2l.discourse.group/t/259)

<!--en:6a82bf78-->
[Diskussion](https://d2l.discourse.group/t/17973)

<!--en:c362f1f2-->
Lineare Regression: Theorie, Geometrie und stochastische Optimierung<br>
**Affines Modell · quadratischer Verlust · Normalgleichungen · orthogonale Projektion · Minibatch-SGD · Maximum-Likelihood-Schätzung bei gaußschem Rauschen**

<!--en:a4930c50-->
::: {.slide title="Vorhersage stetiger Größen aus Merkmalen"}

<!--en:c6ba802c-->
[Motivation und Aufgabenstellung]{.kicker}

<!--en:6a260407-->
Regression sagt einen stetigen numerischen Zielwert aus beobachteten Kovariaten vorher:

<!--en:9c582b5e-->
- **Konkrete Aufgabe:** Den Verkaufspreis $y \in \mathbb{R}$ eines Hauses ($150\text{ m}^2$, gebaut vor $20\text{ years}$) anhand von Trainingsdaten aus $n$ früheren Verkäufen $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ schätzen.

<!--en:1c298496-->
- **Merkmale und Labels:** Zu jedem Beispiel $i$ gehören ein Merkmalsvektor $\mathbf{x}^{(i)} = (x_1^{(i)}, \ldots, x_d^{(i)})^\top \in \mathbb{R}^d$ und ein beobachteter skalarer Zielwert $y^{(i)} \in \mathbb{R}$.

<!--en:74986b90-->
- **Grundannahme:** Im Mittel ist der Zielwert eine affine Funktion der Merkmale. Hinzu kommt nicht modelliertes Messrauschen.

<!--en:b375a749-->
Eine an frühere Verkäufe angepasste Ebene liefert eine explizite Preisformel, die sich direkt auf jedes bisher ungesehene Haus $\mathbf{x}$ anwenden lässt.

<!--en:1eb82e9e-->
::: {.slide title="Das affine Modell: Gewichte und Bias"}

<!--en:9a30ca2b-->
[Die Modellklasse]{.kicker}

<!--en:20d43693-->
- **Marginale Steigungen ($\mathbf{w} \in \mathbb{R}^d$):** Bei $w_{\textrm{area}} = 3{,}000\text{ \$/m}^2$ und $w_{\textrm{age}} = -1{,}000\text{ \$/yr}$ erhöht jeder zusätzliche Quadratmeter den Preis um $\$3{,}000$. Jedes weitere Jahr senkt ihn um $\$1{,}000$.

<!--en:b367a0e9-->
- **Achsenabschnitt / Bias ($b \in \mathbb{R}$):** Mit $b = \$50{,}000$ kostet das Haus mit $150\text{ m}^2$ im Alter von $20$ Jahren $\$480{,}000$. Durch $b$ muss die angepasste Hyperebene nicht durch den Ursprung $\mathbf{x} = \mathbf{0}$ verlaufen.

<!--en:3ad7d4ac-->
::: {.slide title="Vektorisierung über den Datensatz mit der Entwurfsmatrix"}

<!--en:a276c29b-->
[Matrixdarstellung]{.kicker}

<!--en:f6d5945c-->
- **Passende Formen:** Das Produkt aus Entwurfsmatrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ und Spaltenvektor $\mathbf{w} \in \mathbb{R}^{d \times 1}$ ergibt $\mathbf{X}\mathbf{w}$ mit der Form $(n, 1)$. Broadcasting addiert $b$ in allen $n$ Zeilen.

<!--en:835ae5b7-->
- **Bias einbeziehen:** Eine zusätzliche Einserspalte $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}] \in \mathbb{R}^{n \times (d+1)}$ führt die affine Abbildung auf ein einziges Matrixprodukt $\tilde{\mathbf{X}}\tilde{\mathbf{w}}$ zurück.

<!--en:1f79084f-->
::: {.slide title="Der quadratische Fehler als Verlustfunktion"}

<!--en:39883335-->
[Messung der Abweichung]{.kicker}

<!--en:6fec90e6-->
Wie gut die Vorhersagen $\hat{y}^{(i)}$ zu den beobachteten Zielwerten $y^{(i)}$ passen, misst der Mittelwert der halben quadrierten vertikalen Residuen auf dem Trainingsdatensatz:

<!--en:9373ca6a-->
- **Residuum ($r^{(i)} = \hat{y}^{(i)} - y^{(i)}$):** Vorzeichenbehafteter vertikaler Abstand zwischen dem vorhergesagten Punkt auf der Geraden und dem beobachteten Datenpunkt.

<!--en:33aba9b8-->
- **Warum der Faktor $1/2$?** Er vereinfacht die Rechnung, weil er den beim Ableiten entstehenden Faktor $2$ kürzt. Dabei bleibt $\operatorname{argmin}_{\mathbf{w}, b} L(\mathbf{w}, b)$ unverändert.

<!--en:9c1433cd-->
- **Warum durch $n$ teilen?** Die Normierung mit dem Stichprobenumfang hält die Größenordnung von Verlust und Gradienten bei wechselnder Datensatzgröße konstant.

<!--en:07265484-->
::: {.slide title="Fallstrick: Empfindlichkeit des quadratischen Verlusts gegenüber Ausreißern"}

<!--en:d1e48a27-->
[Statistisches Fehlerverhalten]{.kicker}

<!--en:5d02493e-->
- **Quadratisch wachsender Einfluss oder begrenzte Stimme:** Ein einziges auf $10{,}000$ verfälschtes Label verschiebt die Steigung bei kleinsten Quadraten von $2.00$ auf $22.88$. Beim mittleren absoluten Fehler ($\frac{1}{n}\sum_i |\hat{y}^{(i)} - y^{(i)}|$) bleibt der Subgradient jedes Ausreißers auf $\pm 1$ begrenzt und die Steigung bei $w = 2.00$.

<!--en:426b58f6-->
::: {.slide title="Lösung in geschlossener Form: Die Normalgleichungen"}

<!--en:22c3b826-->
[Analytische Optimierung]{.kicker}

<!--en:dd62b50a-->
In erweiterter Schreibweise $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}] \in \mathbb{R}^{n \times (d+1)}$ ist der empirische Verlust eine konvexe quadratische Schale $L(\tilde{\mathbf{w}}) = \frac{1}{2n}\|\tilde{\mathbf{X}}\tilde{\mathbf{w}} - \mathbf{y}\|_2^2$:

<!--en:73692f61-->
- **Zerlegung nach Pythagoras:** Für jeden Kandidaten $\tilde{\mathbf{w}}$ ergibt die Entwicklung von $\|\mathbf{y} - \tilde{\mathbf{X}}\tilde{\mathbf{w}}\|_2^2$ um eine Lösung $\hat{\tilde{\mathbf{w}}}$ die Identität $\|\mathbf{y} - \tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}}\|_2^2 + \|\tilde{\mathbf{X}}(\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}})\|_2^2$, weil der gemischte Term $(\mathbf{y} - \tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}})^\top\tilde{\mathbf{X}}$ identisch verschwindet.

<!--en:97280049-->
- **Eindeutigkeit:** Sind die $d + 1$ Spalten von $\tilde{\mathbf{X}}$ linear unabhängig (dafür ist $n \geq d + 1$ erforderlich), ist die Gram-Matrix $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ strikt positiv definit und $\hat{\tilde{\mathbf{w}}} = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\mathbf{y}$ eindeutig.

<!--en:e1970266-->
- **Kollinearität ($d > n$ oder doppelte Merkmale):** Ist $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ singulär, erzielen unendlich viele Gewichtsvektoren genau denselben minimalen Verlust. `lstsq` liefert über QR-Zerlegung oder SVD die Lösung mit kleinster Norm $\tilde{\mathbf{X}}^\dagger \mathbf{y}$.

<!--en:dd979125-->
::: {.slide title="Rechenbeispiel und numerische Tücken der Matrixinversion"}

<!--en:caa0b322-->
[Normalgleichungen im Code]{.kicker}

<!--en:1e14db6d-->
- **Warum `inv(X.T @ X)` vermeiden?** Beim Produkt $\mathbf{X}^\top\mathbf{X}$ quadriert sich die Konditionszahl: $\kappa(\mathbf{X}^\top\mathbf{X}) = \kappa(\mathbf{X})^2$. Spezialisierte Löser (`lstsq`) zerlegen $\mathbf{X}$ direkt mittels QR oder SVD in $O(nd^2)$ Gleitkommaoperationen.

<!--en:b703ae1b-->
::: {.slide title="Kleinste Quadrate als orthogonale Projektion"}

<!--en:8117e0d0-->
[Woher das Wort „normal“ in den Normalgleichungen kommt]{.kicker}

<!--en:edbc804a-->
Im Beobachtungsraum $\mathbb{R}^n$ überstreichen die Vorhersagen bei veränderlichem $\tilde{\mathbf{w}} \in \mathbb{R}^{d+1}$ einen ebenen $(d+1)$-dimensionalen Unterraum, den Spaltenraum $\operatorname{col}(\tilde{\mathbf{X}})$:

<!--en:2e76abe9-->
- **Nächstgelegener Punkt im Unterraum:** Der verrauschte Vektor $\mathbf{y} \in \mathbb{R}^n$ liegt fast nie in $\operatorname{col}(\tilde{\mathbf{X}})$. Die Minimierung von $\|\mathbf{y} - \tilde{\mathbf{X}}\tilde{\mathbf{w}}\|_2^2$ liefert den nächstgelegenen Punkt $\hat{\mathbf{y}} = \tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}}$.

<!--en:29dceb9a-->
- **Orthogonalität des Residuums:** Die kürzeste Strecke von $\mathbf{y}$ zum Unterraum steht senkrecht (*normal*) auf ihm. Daher erfüllt $\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y}$ die Gleichung:
  $$\tilde{\mathbf{X}}^\top \mathbf{r} = \mathbf{0}$$

<!--en:fa193c3d-->
- **Summe der Residuen gleich null:** Da die Bias-Spalte $\mathbf{1}$ in $\operatorname{col}(\tilde{\mathbf{X}})$ liegt, gilt $\mathbf{1}^\top\mathbf{r} = \sum_{i=1}^n r^{(i)} = 0$ für jede Anpassung mit kleinsten Quadraten und Achsenabschnitt.

<!--en:1037f102-->
::: {.slide title="Warum Deep Learning Minibatch-SGD verwendet"}

<!--en:433872aa-->
[Von der geschlossenen Lösung zur iterativen Optimierung]{.kicker}

<!--en:92d06aa5-->
- **Kubischer Aufwand in der Parameteranzahl ($O(nd^2 + d^3)$):** Bei $d = 10^8$ Parametern bräuchte eine $d \times d$-Gram-Matrix $40\text{ petabytes}$ Arbeitsspeicher.

<!--en:74092151-->
- **Redundanz im Datensatz ($n \gg |\mathcal{B}|$):** Der Gradientenabstieg mit dem gesamten Datensatz liest vor jedem Schritt alle $n = 10^7$ Beispiele. Schon $|\mathcal{B}| = 64$ zufällige Beispiele liefern fast dieselbe Abstiegsrichtung.

<!--en:75bff399-->
- **Keine geschlossene Lösung für tiefe Modelle:** Bei Kreuzentropie und nichtlinearen versteckten Schichten gibt es keine geschlossene Lösung über Normalgleichungen.

<!--en:ec5f0e38-->
::: {.slide title="Funktionsweise des stochastischen Gradientenabstiegs mit Minibatches"}

<!--en:f067ead7-->
[Die Aktualisierungsregel in der Praxis]{.kicker}

<!--en:e9cb879b-->
- **Korrektur anhand des Fehlers:** Jedes Beispiel $i \in \mathcal{B}_t$ zieht $\mathbf{w}$ entlang von $-\mathbf{x}^{(i)}$ proportional zu seinem Residuum $r^{(i)} = \hat{y}^{(i)} - y^{(i)}$.

<!--en:92995534-->
- **Erwartungstreuer Schätzer:** Gleichverteiltes Ziehen ergibt $\mathbb{E}[\nabla L_{\mathcal{B}_t}] = \nabla L$ mit Standardfehler $\propto 1/\sqrt{|\mathcal{B}|}$. Dabei sind $(\mathbf{w}, b)$ gelernte *Parameter* und $(\eta, |\mathcal{B}|)$ *Hyperparameter*.

<!--en:aca87dd4-->
::: {.slide title="Schwankungen um das Minimum und Skalierung der Lernrate"}

<!--en:ae9d4454-->
[Stochastische Dynamik]{.kicker}

<!--en:a0b4445f-->
Selbst am exakten empirischen Minimierer $\hat{\mathbf{w}}$ mit vollständigem Gradienten $\nabla L(\hat{\mathbf{w}}) = \mathbf{0}$ haben einzelne verrauschte Beispiele von null verschiedene Residuen $r^{(i)} \neq 0$:

<!--en:36fdc64a-->
- **Gradientenabstieg mit dem gesamten Datensatz (schwarz):** Berechnet in jedem Schritt den exakten Gradienten über alle $n$ Beispiele und bewegt sich gleichmäßig über elliptische Niveaulinien zu $\hat{\mathbf{w}}$.

<!--en:bb83bc42-->
- **Minibatch-SGD (farbig):** Steigt mit $n / |\mathcal{B}|$ kostengünstigen Schritten pro Epoche rasch ab und schwankt dann fortwährend in einer **Kugel** um $\hat{\mathbf{w}}$:
  $$\mathbb{E}\bigl[\|\mathbf{w}_t - \hat{\mathbf{w}}\|_2^2\bigr] \propto \frac{\eta\,\sigma^2}{|\mathcal{B}|}$$

<!--en:2f1571a1-->
- **Einfluss auf den Radius:** Die Verringerung auf ein Viertel der Lernrate $\eta$ (rechts) oder die Erhöhung auf das Vierfache der Minibatchgröße $|\mathcal{B}|$ halbiert den Radius. Mit abklingendem $\eta_t \to 0$ konvergieren die Iterierten gegen $\hat{\mathbf{w}}$.

<!--en:93fa4585-->
::: {.slide title="Vektorisierung: Python-Schleifen und Matrix-Kernel"}

<!--en:4a0103de-->
[Rechengeschwindigkeit]{.kicker}

<!--en:8e9180bf-->
- **Aufwand beim Start einsparen:** Bei einer Python-Schleife fallen Interpreteraufwand und Latenz beim Kernel-Start 1000-mal an. Dagegen startet `X @ w + b` einen einzigen kompilierten BLAS- oder GPU-Kernel (bei Zeitmessungen asynchrone GPU-Datenströme synchronisieren).

<!--en:66f50582-->
::: {.slide title="Probabilistische Grundlage: Das gaußsche Rauschmodell"}

<!--en:e113601a-->
[Warum der quadratische Fehler?]{.kicker}

<!--en:51d21ae0-->
Im Jahr 1809 fragte Gauß, bei welcher Wahrscheinlichkeitsverteilung der Messfehler $\epsilon$ die Schätzung nach kleinsten Quadraten die wahrscheinlichste ist:

<!--en:4d9e2c78-->
- **Additives gaußsches Rauschen ($y = \mathbf{w}^\top \mathbf{x} + b + \epsilon$ mit $\epsilon \sim \mathcal{N}(0, \sigma^2)$):**
  $$p(y \mid \mathbf{x}; \mathbf{w}, b) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\Bigl(-\frac{(y - \mathbf{w}^\top \mathbf{x} - b)^2}{2\sigma^2}\Bigr)$$

<!--en:4425bea4-->
- **Homoskedastische Normalverteilung:** Jede bedingte Dichte $p(y \mid \mathbf{x})$ ist eine Gauß-Glocke mit Zentrum auf der Regressionsgeraden $\mathbf{w}^\top\mathbf{x} + b$ und konstanter Standardabweichung $\sigma$.

<!--en:4f363537-->
- **Schnell abfallende Ränder:** Weil $(y - \hat{y})^2$ im Exponenten von $\exp(-\cdot / 2\sigma^2)$ steht, beträgt die Dichte bei einem Residuum von $5\sigma$ nur $e^{-12.5} \approx 4 \times 10^{-6}$ des Spitzenwerts.

<!--en:0c9d61a6-->
::: {.slide title="Maximum-Likelihood-Schätzung bei gaußschem Rauschen liefert kleinste Quadrate"}

<!--en:7467c2a2-->
[Statistische Herleitung]{.kicker}

<!--en:73e52668-->
Für $n$ unabhängige Beobachtungen $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ zerfällt die bedingte Likelihood des Datensatzes in das Produkt $\prod_{i=1}^n p(y^{(i)} \mid \mathbf{x}^{(i)}; \mathbf{w}, b)$. Die Maximierung der Likelihood entspricht der Minimierung der negativen Log-Likelihood (NLL):

<!--en:707bbd16-->
- **Unabhängigkeit von $\sigma^2$:** Der erste Term $\frac{n}{2}\log(2\pi\sigma^2)$ ist eine von $(\mathbf{w}, b)$ unabhängige Konstante.

<!--en:7b598a7e-->
- **Exakte Äquivalenz:** Der zweite Term lautet $\frac{n}{\sigma^2} L(\mathbf{w}, b)$, also der empirische quadratische Verlust multipliziert mit der positiven Konstanten $n / \sigma^2$.

<!--en:6c234556-->
**Verlustfunktion aus der Maximum-Likelihood-Schätzung:** Minimierung des quadratischen Fehlers liefert bei additivem homoskedastischem gaußschem Rauschen exakt den Maximum-Likelihood-Schätzer $(\hat{\mathbf{w}}_{\textrm{MLE}}, \hat{b}_{\textrm{MLE}})$, unabhängig von der Rauschvarianz $\sigma^2$.

<!--en:c5f5b271-->
::: {.slide title="Verlustfunktionen aus anderen Rauschverteilungen"}

<!--en:dc6329d9-->
[Jenseits des gaußschen Rauschens]{.kicker}

<!--en:efe7a1c2-->
Ersetzt man die gaußsche Rauschdichte $p(\epsilon) \propto \exp(-l(\epsilon))$, lassen sich mit $-\log p(y \mid \mathbf{x})$ systematisch robuste und anwendungsspezifische Regressionsverluste herleiten:

<!--en:c7196f25-->
- **Laplace-Rauschen $\to$ $\ell_1$ absoluter Fehler:** Die langsam exponentiell abfallenden Ränder $p(\epsilon) = \frac{1}{2\tau}e^{-|\epsilon|/\tau}$ liefern $l(\hat{y}, y) = |\hat{y} - y|$. Der Subgradient $\operatorname{sign}(\hat{y} - y) \in \{\pm 1\}$ begrenzt den Einfluss jedes Ausreißers.

<!--en:33325d51-->
- **Huber-Verlust (glatt und robust):** Verbindet eine quadratische Schale $\frac{1}{2}r^2$ für kleine Residuen ($|r| \leq \delta$) mit linearen Ästen $\delta(|r| - \delta/2)$ für große Residuen ($|r| > \delta$).

<!--en:094347f1-->
- **Poisson-Zähldaten ($y \in \{0, 1, \ldots\}$):** Die Modellierung der Rate $\log \nu(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b$ liefert die konvexe NLL $l = \nu(\mathbf{x}) - y\log\nu(\mathbf{x})$.

<!--en:b0e2814e-->
::: {.slide title="Lineare Regression als einschichtiges neuronales Netz"}

<!--en:5491dcb0-->
[Verbindung zum Deep Learning]{.kicker}

<!--en:993159c2-->
In der Sprache neuronaler Netze ist die lineare Regression ein einzelnes künstliches Neuron, eine **vollständig verbundene (dichte) Schicht** mit $d$ Eingaben und $1$ linearer Ausgabeeinheit:

<!--en:503cf8b6-->
- **Eingabeschicht ($x_1, \ldots, x_d$):** Enthält die $d$ unverarbeiteten Merkmalskoordinaten. Da sie keine einstellbaren Parameter hat, zählt sie bei der Tiefe nicht mit.

<!--en:4931dcb8-->
- **Synaptische Gewichte und Ausgabe ($o_1$):** Jeder Eingabeknoten $x_j$ ist mit dem Ausgabeknoten $o_1$ über das Gewicht $w_j$ verbunden. Der Ausgabeknoten berechnet $o_1 = \sum_{j=1}^d w_j x_j + b$.

<!--en:b263a641-->
- **Warum später Schichten stapeln?** Mehrere affine Schichten ohne Nichtlinearitäten ergeben zusammen wieder eine einzige affine Abbildung. Elementweise Aktivierungen ermöglichen im Kapitel zum MLP die universelle Approximation.

<!--en:bae3fa0a-->
::: {.slide title="Verständnisfragen zu Orthogonalität, Schwankungen und Likelihood"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:bc0b0d8f-->
Diese drei Fragen vor dem Aufdecken der Lösungen aus den Grundprinzipien beantworten:

<!--en:28805c1b-->
1. **Summe der Residuen gleich null:** Warum erfüllen die Residuen $r^{(i)} = \hat{y}^{(i)} - y^{(i)}$ *jeder* Anpassung mit kleinsten Quadraten und Bias $b$ die Bedingung $\sum_{i=1}^n r^{(i)} = 0$?

<!--en:bc66f459-->
2. **Batchgröße und Schwankungen:** Die Minibatchgröße steigt von $|\mathcal{B}| = 32$ auf $|\mathcal{B}| = 128$ bei fester Lernrate $\eta$. Wie verändert sich der quadrierte Abstand im stationären Zustand $\mathbb{E}[\|\mathbf{w}_t - \hat{\mathbf{w}}\|_2^2]$? Welche Lernrate stellt die ursprüngliche Schwankungsbreite wieder her?

<!--en:179659b0-->
3. **Likelihood-Verhältnis bei Ausreißern:** Um welchen Faktor ist bei Rauschen mit Skalenparameter eins ein Residuum von $\epsilon = 10$ weniger plausibel als eines von $\epsilon = 1$? Was ergibt sich bei $\mathcal{N}(0, 1)$ und bei der Laplace-Verteilung $p(\epsilon) = \frac{1}{2}e^{-|\epsilon|}$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:e4ed34ca-->
1. Die erweiterte Entwurfsmatrix $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}]$ enthält die Einserspalte $\mathbf{1}$. Daher erzwingen die Normalgleichungen $\tilde{\mathbf{X}}^\top\mathbf{r} = \mathbf{0}$ die Bedingung $\mathbf{1}^\top\mathbf{r} = \sum_{i=1}^n r^{(i)} = 0$.

<!--en:dbbab666-->
2. Wegen $\mathbb{E}[\|\mathbf{w}_t - \hat{\mathbf{w}}\|_2^2] \propto \eta / |\mathcal{B}|$ verkleinert eine Vervierfachung von $|\mathcal{B}|$ den quadrierten Abstand um den Faktor $4\times$ und halbiert den Kugelradius. Die linear mitvergrößerte Lernrate $4\eta$ stellt das ursprüngliche Verhältnis $\eta / |\mathcal{B}|$ wieder her.

<!--en:a007591a-->
3. Verhältnis bei gaußschem Rauschen: $e^{-(10^2 - 1^2)/2} = e^{-49.5} \approx 3 \times 10^{-22}$ (praktisch unmöglich, daher kippt die Gerade). Verhältnis bei Laplace-Rauschen: $e^{-(10 - 1)} = e^{-9} \approx 1.2 \times 10^{-4}$ (selten, aber plausibel, sodass die Steigung vom Ausreißer unbeeinflusst bleibt).

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und zentrale Ergebnisse"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:d8eae15c-->
| Begriff | Mathematische Form | Rechenaufwand / Skalierung | Bedeutung in der Praxis |

<!--en:be3ea6ce-->
| **Affines Modell** | $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b = \tilde{\mathbf{X}}\tilde{\mathbf{w}}$ | $O(nd)$ Gleitkommaoperationen pro Durchlauf durch den Datensatz | Eine vollständig verbundene Schicht. Durch den Bias $b$ ist die Hyperebene nicht an den Ursprung gebunden. |

<!--en:efb15270-->
| **Quadratischer Verlust** | $L(\mathbf{w}, b) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|_2^2$ | Reduktion mit Aufwand $O(n)$ über die Residuen | Exakter Maximum-Likelihood-Schätzer bei Rauschen $\mathcal{N}(0, \sigma^2)$, empfindlich gegenüber Ausreißern bei schweren Verteilungsrändern |

<!--en:1d045464-->
| **Normalgleichungen** | $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}\hat{\tilde{\mathbf{w}}} = \tilde{\mathbf{X}}^\top\mathbf{y}$ | $O(nd^2 + d^3)$ mit QR/SVD (`lstsq`) | Orthogonale Projektion von $\mathbf{y}$ auf $\operatorname{col}(\tilde{\mathbf{X}})$, bei großem $d$ nicht praktikabel |

<!--en:f3d0a309-->
| **Minibatch-SGD** | $\tilde{\mathbf{w}} \leftarrow \tilde{\mathbf{w}} - \frac{\eta}{|\mathcal{B}|}\tilde{\mathbf{X}}_{\mathcal{B}}^\top(\tilde{\mathbf{X}}_{\mathcal{B}}\tilde{\mathbf{w}} - \mathbf{y}_{\mathcal{B}})$ | $O(|\mathcal{B}|d)$ Gleitkommaoperationen pro Schritt, unabhängig von $n$ | Erwartungstreuer Gradient mit Varianz $\propto 1/|\mathcal{B}|$, schwankt schließlich in einer Kugel mit quadriertem Radius $\propto \eta/|\mathcal{B}|$ |

<!--en:8d329c26-->
| **Robuste Alternativen** | $\ell_1$: $\|\hat{\mathbf{y}} - \mathbf{y}\|_1 / n$, Huber $l_\delta$ | $O(|\mathcal{B}|d)$ pro Subgradientenschritt | Aus der Maximum-Likelihood-Schätzung bei Laplace-Rauschen folgt eine Begrenzung des Gradientenbeitrags jedes Ausreißers auf $\pm 1$ |
