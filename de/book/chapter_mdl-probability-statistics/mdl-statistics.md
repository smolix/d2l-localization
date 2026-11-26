<!--en:7b3eabfb-->
# Statistik

<!--en:3e4889cc-->
Jedes Modell in diesem Buch beurteilen wir letztlich anhand einer Zahl, die wir aus zurückgehaltenen Daten berechnen: der Testgenauigkeit, dem Validierungsverlust oder einem Benchmark-Score. Wir betrachten diese Zahl oft als Eigenschaft des Modells. Sie hängt aber auch davon ab, welche Beispiele zufällig in den Testdatensatz gelangt sind. Angenommen, ein Klassifikator beantwortet 912 von 1000 Testfragen richtig. Seine Genauigkeit beträgt also 91.2%. Bei 1000 anderen Fragen aus derselben Quelle läge sie etwas höher oder niedriger. Die 91.2% sind also eine Realisierung einer Zufallsvariablen. Nun beantwortet ein zweites Modell 921 derselben Fragen richtig. Ist es tatsächlich besser, oder kamen ihm gerade diese Fragen zufällig entgegen?

<!--en:b700e944-->
Die Statistik beantwortet solche Fragen, indem sie untersucht, wie stark ein Schätzwert über all die anderen Stichproben hinweg schwanken würde, die wir nie gezogen haben. Diese Schwankungen folgen einer Verteilung, der *Stichprobenverteilung* (engl. sampling distribution) des Schätzers. Alle Verfahren in diesem Abschnitt beschreiben diese Verteilung oder ziehen aus ihr Schlüsse. Ihre Lage und Streuung zeigen, ob ein Schätzer sein Ziel systematisch verfehlt und wie weit die Schätzwerte auseinanderliegen. Ein Hypothesentest prüft, ob ein beobachteter Unterschied im Rahmen der Streuung liegt, die der Zufall allein erklären würde. Ein Konfidenzintervall umfasst alle Parameterwerte, die sich anhand der Daten nicht ausschließen lassen. Gibt es für die Stichprobenverteilung keine geschlossene Formel, simuliert das Bootstrap-Verfahren sie direkt aus den vorliegenden Daten.

<!--en:44be3e11-->
Im Deep Learning hängen fast alle technischen Entscheidungen vom Vergleich zufällig schwankender Zahlen ab. Welches Modell setzen wir ein? Welche Hyperparametereinstellung schneidet am besten ab? Hat eine neue Idee geholfen? Entscheiden wir anhand eines Unterschieds, der kleiner ist als seine eigene zufällige Schwankung, könnten wir ebenso gut eine Münze werfen. Bisher haben wir in diesem Kapitel untersucht, wie sich Daten bei bekannter Verteilung verhalten. Nun gehen wir den umgekehrten Weg und gewinnen aus beobachteten Daten Schätzwerte mit aussagekräftigen Fehlerbalken. Die Daten müssen nicht einmal aus der Außenwelt stammen. Nähern wir einen Erwartungswert durch Mittelung am Computer erzeugter Stichprobenwerte an (eine Monte-Carlo-Schätzung), erhalten wir wieder einen Stichprobenmittelwert. Dieselbe Theorie liefert also seinen Fehlerbalken (:numref:`subsec_mdl-monte-carlo`). Ebenso verhält sich die Testgenauigkeit wie der Kopfanteil bei einem Münzwurf pro Frage. Die Münze aus den früheren Abschnitten (9-mal Kopf bei 13 Würfen) begegnet uns deshalb wieder zur Plausibilitätsprüfung.

<!--en:9d7c5d8a-->
Wenn Sie sich nur drei Gedanken aus diesem Abschnitt merken, dann diese: Der mittlere quadratische Fehler (engl. mean squared error, MSE) ist die Summe aus dem Quadrat der Verzerrung (engl. bias) und der Varianz. Ein $p$-Wert ist unter der Nullhypothese gleichverteilt. Ein Konfidenzintervall ist die Menge der Parameterwerte, die ein Hypothesentest nicht verwirft. Beim ersten Lesen können Sie die Ausführungen zur Konsistenz und zur Delta-Methode überfliegen. Behalten Sie dabei eine Voraussetzung im Blick: Alle folgenden Garantien setzen voraus, dass wir den Schätzer oder Test *vor* der Betrachtung der Daten wählen. Am Ende des Abschnitts sehen wir, wie leicht diese Garantien sonst hinfällig werden.

<!--en:9627cba8-->
## Schätzer und ihre Stichprobenverteilung

<!--en:43ee10e4-->
Wie sehr können wir der Genauigkeit von 91.2% vertrauen? Dazu brauchen wir einen präzisen Begriff für die Rechenregel, die diesen Wert liefert, und ein einfaches Wahrscheinlichkeitsmodell für die Auswahl der Testfragen.

<!--en:14eb52b4-->
### Schätzer und Stichprobenanteil

<!--en:c27cf192-->
Ein *Schätzer* (engl. estimator) ist eine Regel, die einer Stichprobe $x_1,\ldots,x_n$ eine Zahl $\hat\theta_n$ zuordnet, um einen unbekannten Parameter $\theta$ anzunähern. Der Stichprobenmittelwert, der Stichprobenmedian und der Anteil richtiger Antworten auf einem Testdatensatz sind Schätzer. Da die Daten zufällig sind, ist auch $\hat\theta_n$ zufällig. Seine Verteilung bei wiederholter Ziehung von Stichproben des Umfangs $n$ heißt *Stichprobenverteilung*.

<!--en:fe018916-->
Um die Testgenauigkeit unseres Klassifikators zu modellieren, setzen wir $x_i=1$, wenn das Modell Frage $i$ richtig beantwortet, und andernfalls $x_i=0$. Wir nehmen an, dass wir die Testfragen unabhängig aus einem großen Fragenbestand ziehen. Beträgt die wahre Genauigkeit des Klassifikators auf diesem Bestand $\theta$, so sind die Indikatoren $x_i$ unabhängig und identisch verteilte Bernoulli-Zufallsvariablen mit Erwartungswert $\theta$. Unser Schätzwert ist der Stichprobenanteil $\hat\theta$ richtiger Antworten unter den $n$ Fragen. Aus der Linearität des Erwartungswerts folgt $\mathbb{E}[\hat\theta]=\theta$. Wie in :numref:`sec_prob` beträgt die Varianz $\textrm{Var}(\hat\theta)=\theta(1-\theta)/n$. Der Stichprobenanteil liegt also im Mittel genau bei der wahren Genauigkeit. Sein typischer Fehler nimmt mit der Rate $1/\sqrt n$ ab. Setzen wir $\theta=0.912$ und $n=1{,}000$ ein, erhalten wir die Standardabweichung $\sqrt{0.912\cdot0.088/1000}\approx0.0090$, also etwa 0.9 Prozentpunkte. Nach dem zentralen Grenzwertsatz nähert eine Gauß-Verteilung die Stichprobenverteilung gut an. Auf rund 95% der Testdatensätze liegt die Genauigkeit daher im Bereich von $\pm1.8$ Prozentpunkten um $\theta$.

<!--en:2374dd4c-->
Diese Stichprobenverteilung können wir auch direkt beobachten. Simulieren wir 10 000 unabhängige Testdatensätze mit je 1000 Fragen für einen Klassifikator, dessen wahre Genauigkeit $0.912$ beträgt. Prüfen wir dann Mittelwert und Streuung der gemessenen Genauigkeiten.

<!--en:3cf80c7f-->
Wie von der Formel vorhergesagt, liegen die simulierten Genauigkeiten um $0.912$ mit einer Standardabweichung von $0.0090$. Diese Wolke aus 10 000 Zahlen steht für die Genauigkeiten auf hypothetischen Testdatensätzen, die wir nie gezogen haben. Auf sie beziehen sich alle Verfahren dieses Abschnitts.

<!--en:4793216c-->
Kehren wir zu unserem zweiten Modell zurück, das 921 Fragen richtig beantwortet hat. Hätten wir seinen Score auf einem eigenen, unabhängigen Testdatensatz mit 1000 Fragen gemessen, hätte der beobachtete Unterschied von $0.009$ zwischen den Genauigkeiten eine Standardabweichung von $\sqrt{0.0090^2+0.0085^2}\approx0.012$. Dabei ergibt sich $0.0085$ aus derselben Formel für die Genauigkeit $0.921$. Ein Unterschied von $0.009$ beträgt nur etwa das 0.7-Fache seines Standardfehlers (engl. standard error). Das reicht bei Weitem nicht aus, um die Modelle zu unterscheiden. Hier haben wir beide Modelle aber auf *denselben* 1000 Fragen ausgewertet. In :numref:`subsec_mdl-paired-comparisons` sehen wir, wie viel genauer wir sie dank des gemeinsamen Testdatensatzes vergleichen können.

<!--en:78261cde-->
### Verzerrung und Varianz

<!--en:1ce619a2-->
Ein Schätzer kann den wahren Wert auf zweierlei Weise verfehlen. Er kann im Mittel danebenliegen oder stark um seinen eigenen Mittelwert streuen. Die *Verzerrung* von $\hat\theta_n$ misst die erste Fehlerart, die systematische Abweichung des Erwartungswerts des Schätzers vom wahren Parameter:

<!--en:f0deca55-->
Dabei bezieht sich der Erwartungswert auf die zufällige Ziehung der Stichprobe. Ist die Verzerrung für jeden möglichen Wert von $\theta$ null, so heißt der Schätzer *erwartungstreu* (engl. unbiased). Die *Varianz* misst die zweite Fehlerart, also wie stark $\hat\theta_n$ um seinen eigenen Erwartungswert schwankt. Ihre Quadratwurzel ist der *Standardfehler*:

<!--en:ee3ce9a8-->
Die Verzerrung gibt an, wo die Schätzwerte im Mittel liegen. Der Standardfehler gibt an, wie weit ein einzelner Schätzwert typischerweise von diesem Mittelwert abweicht. In unserem Beispiel zur Genauigkeit hat der Stichprobenanteil die Verzerrung $0$ und den Standardfehler $0.009$. Die Varianz misst die Streuung um den eigenen Mittelwert des Schätzers und nicht um $\theta$. Eine starre Regel, die sämtliche Daten ignoriert und immer $0.5$ ausgibt, hat die Varianz null. Ihr gesamter Fehler geht auf die Verzerrung zurück.

<!--en:3a027d2c-->
### Stichprobenvarianz

<!--en:93d6d93e-->
Der Stichprobenanteil ist erwartungstreu. Man könnte daher meinen, Erwartungstreue sei selbstverständlich. Doch schon beim Versuch, einen Standardfehler aus Daten zu berechnen, begegnet uns ein verzerrter Schätzer. Der Standardfehler eines Stichprobenmittelwerts hängt von der Varianz $\sigma^2$ der Grundgesamtheit ab. Diese kennen wir fast nie und müssen sie aus derselben Stichprobe schätzen. Naheliegend ist die *Plug-in-Varianz*, die mittlere quadratische Abweichung vom Stichprobenmittelwert $\bar x$:

<!--en:0b64bdaa-->
Überraschenderweise ist $s_0^2$ systematisch zu klein. Weshalb? Weil $\bar x$ der einzige Punkt ist, der für *diese* Stichprobe die Summe der quadrierten Abweichungen minimiert. Messen wir die Abweichungen vom wahren Mittelwert $\mu$ und liegt dieser an einer anderen Stelle, ist die Summe stets größer. Ersetzen wir also $\mu$ durch $\bar x$, verkleinert sich der Zähler.

<!--en:082a5c5c-->
An der kleinsten nichttrivialen Stichprobe mit $n=2$ sehen wir unmittelbar, wie stark der Zähler schrumpft. Der Stichprobenmittelwert liegt genau in der Mitte zwischen $x_1$ und $x_2$. Die beiden Abweichungen sind daher $\pm(x_1-x_2)/2$, und es gilt

<!--en:f07e6b6e-->
Da $x_1$ und $x_2$ unabhängig sind und jeweils den Erwartungswert $\mu$ und die Varianz $\sigma^2$ haben, hat ihre Differenz $x_1-x_2$ den Erwartungswert $0$ und die Varianz $2\sigma^2$. Die rechte Seite hat somit den Erwartungswert $\sigma^2$. Teilen wir durch $n=2$, erhalten wir im Mittel $\sigma^2/2$, also nur die Hälfte der wahren Varianz. Teilen wir dagegen durch $n-1=1$, erhalten wir im Mittel $\sigma^2$. Bemerkenswerterweise beseitigt dieselbe Korrektur mit $n-1$ die Verzerrung bei jedem Stichprobenumfang $n$.

<!--en:64e20ec2-->
**Proposition (erwartungstreue Stichprobenvarianz).** *Seien $x_1,\ldots,x_n$ unabhängig und identisch verteilt mit Erwartungswert $\mu$ und endlicher Varianz $\sigma^2$, und sei $n\ge2$. Dann gilt*

<!--en:44a6ea0b-->
**Beweis.** Wir vergleichen die Abweichungen von $\bar x$ mit denen von $\mu$. Dazu schreiben wir $x_i-\bar x=(x_i-\mu)-(\bar x-\mu)$ und multiplizieren das Quadrat aus. Mit $\sum_i(x_i-\mu)=n(\bar x-\mu)$ vereinfachen wir den gemischten Term und erhalten

<!--en:7ad3f523-->
Der gemischte Term ist doppelt so groß wie der letzte Term und hat das entgegengesetzte Vorzeichen. Zentrieren wir um $\bar x$ statt um $\mu$, verringert sich die Quadratsumme also um $n(\bar x-\mu)^2$. Bilden wir nun auf beiden Seiten den Erwartungswert. Jeder Term $\mathbb{E}[(x_i-\mu)^2]$ ist gleich $\sigma^2$, also hat die erste Summe den Erwartungswert $n\sigma^2$. Der Stichprobenmittelwert hat die Varianz $\sigma^2/n$, folglich gilt $\mathbb{E}[n(\bar x-\mu)^2]=\sigma^2$. Die Summe der quadrierten Abweichungen hat damit den Erwartungswert $(n-1)\sigma^2$. Division durch $n-1$ ergibt $\mathbb{E}[s^2]=\sigma^2$. $\blacksquare$

<!--en:d4f11f51-->
Der Faktor lässt sich mithilfe der *Freiheitsgrade* (engl. degrees of freedom) merken. Die $n$ Abweichungen $x_i-\bar x$ müssen sich zu null addieren. Nur $n-1$ von ihnen können also frei variieren: Die Schätzung von $\bar x$ aus den Daten kostet einen Freiheitsgrad. Die Plug-in-Varianz $s_0^2$ hat die Verzerrung $\frac{n-1}{n}\sigma^2-\sigma^2=-\sigma^2/n$. Diese fällt bei kleinem $n$ ins Gewicht und verschwindet mit wachsendem Stichprobenumfang.

<!--en:8eb90549-->
Eine Einschränkung ist wichtig: Nichtlineare Transformationen erhalten die Erwartungstreue im Allgemeinen nicht. Aus der Jensen-Ungleichung für die konkave Wurzelfunktion folgt $\mathbb{E}[s]\le\sqrt{\mathbb{E}[s^2]}=\sigma$. Die Ungleichung ist strikt, wenn $s$ nicht konstant ist. Für gaußverteilte Daten und $n=2$ beträgt die Stichprobenstandardabweichung $s=|x_1-x_2|/\sqrt2$. Ihr Erwartungswert ist $\mathbb{E}[s]=\sigma\sqrt{2/\pi}\approx0.80\,\sigma$. Ziehen wir aus einer erwartungstreuen Varianzschätzung die Quadratwurzel, unterschätzen wir die Standardabweichung hier also im Mittel um 20%.

<!--en:b9cc5622-->
In numerischen Bibliotheken wählen Sie den Nenner über einen Parameter für die Korrektur der Freiheitsgrade: `np.var(x, ddof=0)` teilt durch $n$, während `np.var(x, ddof=1)` durch $n-1$ teilt.

<!--en:76077a21-->
### Zerlegung in Verzerrung und Varianz

<!--en:3b8be72c-->
Um zu sehen, wie Verzerrung und Varianz zusammenwirken, konstruieren wir einen Schätzer, bei dem wir beide gezielt gegeneinander abwägen. Angenommen, wir beobachten $n=30$ verrauschte Messwerte $x_i\sim\mathcal N(\theta,\sigma^2)$ mit $\theta=1$ und $\sigma=4$. Der gewöhnliche Stichprobenmittelwert $\bar x$ ist erwartungstreu mit Varianz $\sigma^2/n=16/30\approx0.533$ (Standardfehler $0.73$). Betrachten wir nun den *geschrumpften Mittelwert* (engl. shrunk mean) $c\,\bar x$ mit einem Faktor $0<c<1$. Er zieht den Stichprobenmittelwert in Richtung $0$. Die Schrumpfung gegen null führt zu einer Verzerrung von $c\theta-\theta=(c-1)\theta$, verringert aber die Varianz auf $c^2\sigma^2/n$. Bei $c=0.65$ beträgt die Verzerrung $-0.35$, während der Standardfehler deutlich von $0.73$ auf $0.47$ fällt. :numref:`fig_mdl-sampling-distribution` zeigt beide Stichprobenverteilungen nebeneinander.

<!--en:a261a77d-->
![Stichprobenverteilungen zweier Schätzer für $\theta=1$ aus 30 Ziehungen aus $\mathcal N(1,16)$. Links der Stichprobenmittelwert $\bar x$ mit Verzerrung 0 und Standardfehler 0.73. Rechts der geschrumpfte Mittelwert $0.65\,\bar x$ mit Verzerrung $-0.35$ und Standardfehler 0.47. Die Verzerrung ist die Abweichung des Zentrums von $\theta$, der Standardfehler misst die Breite.](../img/mdl-prob-sampling-distribution.svg)

<!--en:e5f6cee1-->
Welcher Schätzer ist besser? Der geschrumpfte Mittelwert liegt systematisch daneben. Weil er aber deutlich weniger streut, liegt ein einzelner Schätzwert oft näher an $\theta$ als $\bar x$. Für den Vergleich brauchen wir ein Maß, das beide Fehler zugleich erfasst. Der *mittlere quadratische Fehler* (MSE) ist der mittlere quadrierte Abstand zwischen Schätzer und wahrem Parameter:

<!--en:583bd0ff-->
Der MSE lässt sich in die beiden gerade definierten Größen zerlegen.

<!--en:d7c15436-->
**Proposition (Zerlegung in Verzerrung und Varianz).** *Hat $\hat\theta_n$ endliche Varianz, so gilt*

<!--en:17beb657-->
**Beweis.** Sei $m=\mathbb{E}[\hat\theta_n]$ der Erwartungswert des Schätzers. Im Quadrat addieren und subtrahieren wir $m$:

<!--en:53116c37-->
Nun bilden wir gliedweise den Erwartungswert. Der erste Term ergibt $\textrm{Var}(\hat\theta_n)$, der dritte ist die Konstante $\textrm{Bias}(\hat\theta_n)^2$. Der gemischte Term verschwindet, weil $m-\theta$ konstant ist und $\mathbb{E}[\hat\theta_n-m]=0$ nach Definition von $m$ gilt. $\blacksquare$

<!--en:52a64ccb-->
Geometrisch betrachtet ist diese Identität der Satz des Pythagoras. Wir fassen Zufallsvariablen als Vektoren mit dem inneren Produkt $\langle U,V\rangle=\mathbb{E}[UV]$ auf. Dann ist die zentrierte Schwankung $\hat\theta_n-m$ orthogonal zu jeder Konstanten $a$, denn $\langle\hat\theta_n-m,\,a\rangle=a\,\mathbb{E}[\hat\theta_n-m]=0$. Schreiben wir den Gesamtfehler $\hat\theta_n-\theta$ als Summe dieser Schwankung und der konstanten Verzerrung $m-\theta$, erhalten wir zwei senkrecht zueinander stehende Katheten mit den Längen $\textrm{se}$ und $|\textrm{Bias}|$. Die Summe ihrer Längenquadrate ist das Hypotenusenquadrat $\textrm{MSE}$. Aus dieser Sicht ist $\mathbb{E}[X]$ die orthogonale Projektion von $X$ auf den Unterraum der Konstanten. Dieser Projektion sind wir bereits in :numref:`sec_mdl-geometry-linear-algebraic-ops` begegnet. Hier übernimmt der Erwartungswert die Rolle des Skalarprodukts. :numref:`fig_mdl-stat-mse-triangle` zeigt das rechtwinklige Dreieck.

<!--en:9fec67e3-->
![Die Zerlegung in Verzerrung und Varianz als Satz des Pythagoras für Schätzer von $\theta=1$ aus 30 Ziehungen aus $\mathcal N(1,16)$. Jeder Schätzer hat die horizontale Koordinate $|\textrm{bias}|$ und die vertikale Koordinate $\textrm{se}$. Sein Abstand vom Ursprung ist daher $\sqrt{\textrm{MSE}}$. Die Schrumpfung von $\bar x$ auf $0.65\,\bar x$ fügt eine horizontale Kathete der Länge 0.35 hinzu, verkürzt aber die vertikale Kathete von 0.73 auf 0.47. Dadurch sinkt die Hypotenusenlänge auf 0.59. Der Schätzer liegt nun innerhalb des gestrichelten Kreises, auf dem alle ebenso genauen Schätzer wie $\bar x$ liegen. Der Stichprobenmedian ist erwartungstreu, liegt aber außerhalb des Kreises.](../img/mdl-prob-stat-mse-triangle.svg)

<!--en:4cc01fb7-->
Mit dieser Zerlegung können wir Schätzer quantitativ vergleichen. Für unser Beispiel mit 30 Beobachtungen ergibt die Formel beim Stichprobenmittelwert einen MSE von $0.533$ (ausschließlich Varianz) und beim geschrumpften Mittelwert einen MSE von $0.123+0.225=0.348$ für den Schätzer $0.65\,\bar x$. Wie schneidet der Stichprobenmedian ab? Wegen der Symmetrie der Gauß-Verteilung ist er erwartungstreu, doch eine Formel für seine Varianz haben wir noch nicht. Simulieren wir 10 000 Datensätze vom Umfang 30 und vergleichen wir die drei Schätzer empirisch.

<!--en:c57421f9-->
Für jeden Schätzer in `estimates` berechnen wir aus den 10 000 simulierten Datensätzen das Quadrat der empirischen Verzerrung, die Varianz und den gesamten MSE. Die Summe $\textrm{Bias}^2 + \textrm{Var}$ vergleichen wir mit unseren analytischen Vorhersagen:

<!--en:e1edcad2-->
Die Simulation bestätigt die Formeln. Beim Stichprobenmittelwert liegt der MSE nahe $0.53$, beim geschrumpften Mittelwert bei $0.12+0.23=0.35$. Der Stichprobenmedian schneidet mit etwa $0.80$ schlechter ab. Obwohl er erwartungstreu ist, ist seine Varianz etwa anderthalbmal so groß wie die von $\bar x$. Ein verzerrter Schätzer kann also deutlich besser abschneiden als ein erwartungstreuer. Erwartungstreue allein garantiert keine Genauigkeit.

<!--en:8bbb1b7c-->
### Der geschrumpfte Mittelwert und die Abwägung zwischen Verzerrung und Varianz

<!--en:8071f20b-->
Der Vergleich wirft eine praktische Frage auf: Wie viel Verzerrung sollten wir für eine geringere Varianz in Kauf nehmen? Beim geschrumpften Mittelwert $c\,\bar x$ können wir diese Frage in geschlossener Form beantworten. Setzen wir seine Verzerrung und Varianz in :eqref:`eq_mdl-bias-variance` ein, ergibt sich

<!--en:7a17c401-->
Ziehen wir $c$ in Richtung null, wächst das Quadrat der Verzerrung, während die Varianz fällt. Leiten wir nach $c$ ab und setzen $-2(1-c)\theta^2+2c\,\sigma^2/n=0$, erhalten wir den optimalen Schrumpfungsfaktor und den zugehörigen minimalen Fehler:

<!--en:5918f931-->
Die optimale Schrumpfung schneidet stets besser ab als der ungeschrumpfte Stichprobenmittelwert. Sie hilft besonders dann, wenn das Rauschen $\sigma^2/n$ im Verhältnis zum Signal $\theta^2$ groß ist. Für unsere Beispielwerte ($\theta=1$, $\sigma=4$, $n=30$) liefert die Formel $c^\star=1/1.533\approx0.65$, also den bereits gewählten Faktor. Der MSE sinkt von $0.533$ auf $0.348$. :numref:`fig_mdl-bias-variance-u-curve` zeigt beide Fehleranteile und ihre Summe in Abhängigkeit von $c$. Die klassische U-Form ergibt sich hier exakt als Summe eines fallenden und eines steigenden Parabelastes.

<!--en:7d91e984-->
![Quadrat der Verzerrung und Varianz des geschrumpften Mittelwerts $c\,\bar x$ als Funktionen von $c$ für $\theta=1$, $\sigma=4$ und $n=30$. Ihre Summe, der MSE, erreicht bei $c^\star\approx0.65$ ihr Minimum von 0.35 gegenüber 0.53 für den ungeschrumpften Stichprobenmittelwert bei $c=1$.](../img/mdl-prob-bias-variance-u-curve.svg)

<!--en:d95323c6-->
Allerdings hängt $c^\star$ vom unbekannten Parameter $\theta$ ab. Wir können den optimalen Faktor also nicht direkt aus den Daten berechnen. Schlimmer noch: Einen festen Schrumpfungsfaktor ohne Kenntnis von $\theta$ zu wählen, kann den Fehler drastisch erhöhen. Läge der wahre Mittelwert bei $\theta=10$, ergäbe derselbe Faktor $c=0.65$ eine Verzerrung von $-3.5$ und einen MSE von $12.25+0.23\approx12.5$, mehr als das Zwanzigfache des MSE von $0.533$ für $\bar x$. Eine Schrumpfung in Richtung $0$ hilft nur, wenn der wahre Wert tatsächlich nahe bei $0$ liegt. In Übung :exref:`ex_mdl-statistics-shrinking-mean` untersuchen Sie, was geschieht, wenn Sie einen Schätzwert für $\theta$ in $c^\star$ einsetzen. Übung :exref:`ex_mdl-statistics-stein` zeigt ein berühmtes, überraschendes Ergebnis: In drei oder mehr Dimensionen schlägt eine ähnliche, datengestützte Schrumpfung den Stichprobenmittelwert für *jeden* Wert von $\theta$.

<!--en:f6ac78f2-->
Dieselbe Abwägung zwischen Verzerrung und Varianz bestimmt die überwachte Vorhersage. Bei einer festen Eingabe $x$ ist die Vorhersage $\hat f(x)$ eines auf einem zufälligen Datensatz trainierten Modells ein Schätzer der wahren Regressionsfunktion $f^*(x)$. Der erwartete quadratische Fehler bei einem neuen Label ist der MSE dieses Schätzers zuzüglich der irreduziblen Varianz des Labelrauschens (:numref:`sec_generalization_basics`). In vielen Modellfamilien senkt zusätzliche Flexibilität die Verzerrung und erhöht die Varianz. Regularisierung wirkt ähnlich wie der Schrumpfungsfaktor $c$ :cite:`Geman.Bienenstock.Doursat.1992`. Allerdings verläuft der Testfehler bei modernen überparametrisierten Modellen nicht immer U-förmig. Jenseits der Interpolationsschwelle, an der das Modell seine Trainingsdaten exakt anpasst, kann der Testfehler ein zweites Mal sinken (:numref:`subsec_mdl-double-descent-example`).

<!--en:5f5d149f-->
### Konsistenz und Effizienz

<!--en:8807af35-->
Mit Verzerrung und Varianz beurteilen wir einen Schätzer bei festem Stichprobenumfang $n$. Was geschieht, wenn wir immer mehr Daten sammeln? Nähert sich der Schätzwert schließlich dem wahren Wert? Ein Schätzer $\hat\theta_n$ heißt *konsistent*, wenn er *in Wahrscheinlichkeit konvergiert*, und zwar gegen $\theta$, das heißt, wenn

<!--en:26b82330-->
Wir schreiben dafür $\hat\theta_n\xrightarrow{P}\theta$. Anschaulich bedeutet das: Für jede Toleranz $\varepsilon>0$ geht die Wahrscheinlichkeit, den wahren Wert $\theta$ um mehr als $\varepsilon$ zu verfehlen, mit wachsendem Stichprobenumfang gegen null.

<!--en:53908f02-->
Mit der Zerlegung in Verzerrung und Varianz können wir Konsistenz schnell prüfen. Wenden wir die Markov-Ungleichung (:numref:`sec_mdl-random_variables`) auf die nichtnegative Zufallsvariable $(\hat\theta_n-\theta)^2$ mit Schwelle $\varepsilon^2$ an, erhalten wir aus dem MSE direkt eine obere Schranke für die Wahrscheinlichkeit eines großen Fehlers:

<!--en:13ec8c50-->
Gehen also Verzerrung und Varianz für $n\to\infty$ beide gegen null, ist der Schätzer konsistent. Keine der beiden Bedingungen reicht allein aus. Verwenden wir nur die erste Beobachtung, $\hat\theta_n=x_1$, ist der Schätzer für jedes $n$ erwartungstreu. Doch seine Varianz bleibt bei $\sigma^2$, und er konvergiert nicht gegen den wahren Wert. Umgekehrt hat $\bar x+1$ die Varianz $\sigma^2/n\to0$, konvergiert aber gegen den falschen Wert. Die Plug-in-Varianz $s_0^2$ zeigt, wie beides zusammenwirkt. Zwar ist sie bei jedem endlichen $n$ verzerrt, doch sowohl ihre Verzerrung $-\sigma^2/n$ als auch ihre Varianz (bei endlichem viertem Moment) gehen mit wachsendem $n$ gegen null. Sie ist also konsistent.

<!--en:4c98a95a-->
Wenden wir diese Schranke auf den Stichprobenmittelwert an, erhalten wir das grundlegende Resultat, das erklärt, weshalb Mittelung funktioniert.

<!--en:2822b65a-->
**Proposition (schwaches Gesetz der großen Zahlen).** *Sind $x_1,\ldots,x_n$ unabhängig und identisch verteilt mit Erwartungswert $\mu$ und endlicher Varianz $\sigma^2$, so gilt für den Stichprobenmittelwert $\bar x_n=\frac1n\sum_{i=1}^n x_i$*

<!--en:f36d634f-->
**Beweis.** Aus der Linearität des Erwartungswerts folgt $\mathbb{E}[\bar x_n]=\mu$. Die Verzerrung ist also null. Da die Variablen unabhängig sind, ist die Varianz ihrer Summe gleich der Summe ihrer Varianzen. Damit gilt $\textrm{Var}(\bar x_n)=n\sigma^2/n^2=\sigma^2/n$. Setzen wir $\textrm{MSE}=\sigma^2/n$ in die obige Markov-Schranke ein, erhalten wir die Ungleichung. Ihre rechte Seite geht für $n\to\infty$ gegen null. $\blacksquare$

<!--en:708e6d6f-->
Tatsächlich gilt das Gesetz bereits unter der schwächeren Voraussetzung eines endlichen Erwartungswerts $\mathbb{E}|x_1|<\infty$, der Beweis ist dann allerdings länger :cite:`Wasserman.2013`. Dieser einfache Satz bildet eine Grundlage des maschinellen Lernens. Er erklärt, warum eine Monte-Carlo-Schätzung eines Integrals konvergiert, warum die Genauigkeit auf einem großen Testdatensatz die wahre Genauigkeit widerspiegelt und warum der mittlere Trainingsverlust den erwarteten Verlust annähert. Die Schranke selbst ist allerdings recht grob. Bei unserem Klassifikator mit 91.2% Genauigkeit und 1000 Fragen liefert sie für eine Abweichung der Testgenauigkeit vom wahren Wert $\theta$ um mehr als 3 Prozentpunkte lediglich die obere Wahrscheinlichkeitsschranke $0.0803/(1000\cdot0.03^2)\approx0.09$. Die Gauß-Approximation liefert eine Wahrscheinlichkeit nahe $0.001$. Die Stärke der Schranke liegt in ihrer Allgemeinheit: Sie setzt nur endliche Varianz voraus. In :numref:`sec_mdl-concentration-generalization` leiten wir unter stärkeren Annahmen über die Verteilungsränder deutlich engere Schranken her.

<!--en:b667d19f-->
Konsistenz ist eine Mindestanforderung. Sie hilft uns nicht, zwischen zwei konsistenten Schätzern zu wählen. Von zwei erwartungstreuen Schätzern ist der mit der kleineren Varianz *effizienter*: Er erreicht dieselbe Präzision mit weniger Beobachtungen. Bei gaußverteilten Daten sind Stichprobenmittelwert und Stichprobenmedian beide erwartungstreu für $\mu$. Für große $n$ beträgt die Varianz des Medians jedoch ungefähr $\frac{\pi}{2}\,\sigma^2/n$ (Übung :exref:`ex_mdl-statistics-median-se`). Der Median braucht also etwa $\pi/2\approx1.57$-mal so viele Daten wie der Mittelwert, um denselben Standardfehler zu erreichen. Seine Effizienz beträgt damit $2/\pi\approx64\%$. Unsere Simulation für $n=30$ ließ dieses Verhältnis bereits erkennen: $0.80$ gegenüber $0.53$. Der Stichprobenanteil dagegen erreicht die kleinstmögliche Varianz unter allen erwartungstreuen Schätzern eines Bernoulli-Parameters. Diese fundamentale Untergrenze heißt Cramér-Rao-Schranke (:numref:`subsec_mdl-cramer-rao`).

<!--en:855dbc7b-->
### Monte-Carlo-Schätzung

<!--en:3fc181ac-->
Bisher stammten unsere Stichproben aus externen Daten: Testfragen, verrauschten Messungen oder Trainingsläufen. Für das Gesetz der großen Zahlen spielt die Herkunft der Stichproben jedoch keine Rolle. Das macht den Stichprobenmittelwert zu einem leistungsfähigen Rechenwerkzeug. Die meisten Erwartungswerte im Deep Learning, etwa ein erwarteter Verlust, eine Normierungskonstante oder einer ihrer Gradienten, sind Integrale ohne geschlossene Darstellung. Oft erstrecken sie sich über Tausende Dimensionen, in denen numerische Gitter viel zu aufwendig sind. Selbst wenn wir kein Gitter aufbauen können, können wir meist Stichproben ziehen. Die Idee stammt aus der Zeit der ersten elektronischen Rechner :cite:`Metropolis.Ulam.1949`. Um $\mathbb{E}[g(X)]$ für eine Zufallsvariable $X$ mit Dichte $p$ anzunähern, ziehen wir unabhängige Stichprobenwerte $x_1,\dots,x_n$ aus $p$ und mitteln die Funktionswerte:

<!--en:791ea291-->
Eine **Monte-Carlo**-Schätzung lässt sich als Riemann-Summe mit zufällig gewählten Stützstellen und gleichen Gewichten $1/n$ auffassen. Dadurch liegen automatisch mehr Stützstellen dort, wo die Dichte $p$ hoch ist. Nach dem Gesetz der großen Zahlen :eqref:`eq_mdl-lln` konvergiert dieser erwartungstreue Mittelwert gegen $\mathbb{E}[g(X)]$ für $n\to\infty$. Seine Varianz bestimmt die Konvergenzrate. Sind $X_1,\dots,X_n$ unabhängig und identisch verteilt mit Dichte $p$ und ist $\sigma^2=\textrm{Var}(g(X))$ endlich, so gilt

<!--en:9a72057b-->
Mathematisch kennen wir das bereits: Es ist die Formel $\sigma^2/n$ für die Varianz eines Stichprobenmittelwerts aus dem Gesetz der großen Zahlen. Dabei übernimmt $g(X_i)$ die Rolle von $x_i$ (:numref:`sec_mdl-joint-distributions`). Bemerkenswert ist die Konsequenz. Die Wurzel des mittleren quadratischen Fehlers fällt mit $\sigma/\sqrt n$. Das erscheint langsam: Für eine zusätzliche Dezimalstelle Genauigkeit brauchen wir 100-mal so viele Stichprobenwerte. Doch die Dimension von $x$ kommt im Exponenten nicht vor. Genau diese Dimensionsunabhängigkeit erklärt, warum die Stichprobenziehung in hohen Dimensionen funktioniert. Entscheidend ist eine endliche Varianz. Wie wir in :numref:`sec_mdl-random_variables` gesehen haben, können schwere Verteilungsränder die Mittelung unzuverlässig machen (siehe Übung :exref:`ex_mdl-random-variables-heavy-tail-averages`).

<!--en:bd13e2ba-->
Vergleichen wir dies mit einem deterministischen Gitter. Für die Gitterweite $\epsilon$ in $d$ Dimensionen brauchen wir $N=\epsilon^{-d}$ Stützstellen. Bei der linken Rechteckregel aus :numref:`sec_mdl-integral_calculus` ist der Fehler proportional zu $\epsilon=N^{-1/d}$. Eine Regel zweiter Ordnung wie die *Mittelpunktsregel*, die jede Zelle in ihrem Mittelpunkt auswertet, verbessert dies auf $\epsilon^2=N^{-2/d}$. Die Monte-Carlo-Schätzung behält dagegen in jeder Dimension ihre Rate $N^{-1/2}$. In einer Dimension ist das Mittelpunktgitter klar überlegen. Bei $d=4$ sind die Exponenten gleich. Bei $d=10$ verlangsamt sich die Konvergenz des Gitters auf $N^{-1/5}$, während das Monte-Carlo-Verfahren weiterhin mit $N^{-1/2}$ konvergiert. Der Zähler $\sigma$ kann zwar mit $d$ wachsen, der Exponent verschlechtert sich aber nie (Übung :exref:`ex_mdl-integral-calculus-grids-monte-carlo`).

<!--en:fc6c3a9c-->
Betrachten wir diese Rate $1/\sqrt n$ an einem konkreten Integral: $\int_0^1e^{-x^2}dx=\mathbb{E}[e^{-U^2}]$ für $U$ gleichverteilt auf $[0,1]$. Der exakte Wert ist $\tfrac{\sqrt\pi}2\operatorname{erf}(1)=0.746824$. Im Folgenden verfolgen wir den laufenden Mittelwert bei Zehnerpotenzen des Stichprobenumfangs. Wir vergleichen den tatsächlichen Fehler mit der vorhergesagten Größenordnung $\sigma/\sqrt n$, wobei wir $\sigma=\mathrm{sd}(e^{-U^2})\approx0.2$ aus der Stichprobe schätzen.

<!--en:1e476f83-->
Die beobachteten Fehler stimmen gut mit der vorhergesagten Größenordnung $\sigma/\sqrt n$ überein, obwohl jede Zeile nur eine zufällige Realisierung darstellt. Ein einzelner Fehler übersteigt $\sigma/\sqrt n$ selten um mehr als den Faktor zwei oder drei und liegt gelegentlich weit darunter. Um den Fehlerverlauf über die Stichprobenumfänge hinweg zu sehen, zeichnen wir den absoluten Fehler eines laufenden Mittelwerts für 200 logarithmisch abgestufte Werte von $n$. Daneben zeichnen wir die Wurzel des mittleren quadratischen Fehlers über 200 unabhängige Läufe, also genau die Größe, die :eqref:`eq_mdl-mc-variance` vorhersagt.

<!--en:1bb9ef08-->
Die beiden Kurven unterscheiden sich deutlich. Der absolute Fehler eines einzelnen laufenden Mittelwerts schwankt um die theoretische Linie, weil benachbarte Werte von $n$ auf fast denselben Stichprobenwerten beruhen. Bei jedem Nulldurchgang des Fehlers fällt die Kurve steil ab. Die Wurzel des mittleren quadratischen Fehlers über 200 unabhängige Läufe weicht dagegen bei jedem Stichprobenumfang um höchstens etwa 20% von $\sigma/\sqrt n$ ab. Über alle $n$-Werte und vier Größenordnungen hinweg fällt sie gleichmäßig mit der Steigung $-\tfrac12$.

<!--en:3eb1c7e2-->
## Hypothesentests

<!--en:ae38a93e-->
Ein Fehlerbalken gibt an, wie unsicher ein einzelner Schätzwert ist. Bei den meisten Entscheidungen im maschinellen Lernen vergleichen wir jedoch zwei Schätzwerte. Schneidet eine neue Architektur, ein anderer Optimierer oder eine neue Form der Regularisierung tatsächlich besser ab als die Baseline? In diesem Unterabschnitt entwickeln wir zunächst mithilfe einer Simulation einen Vergleichstest. Anschließend leiten wir sein Gegenstück in geschlossener Form her und fragen, wie viele Daten ein Experiment braucht, um eine echte Verbesserung zuverlässig nachzuweisen.

<!--en:5b2aff34-->
### Vergleich zweier Modelle

<!--en:65471322-->
Das Training eines neuronalen Netzes unterliegt Zufallseinflüssen: Die Initialisierung der Gewichte, die Reihenfolge der gemischten Daten und die Dropout-Masken hängen alle vom Zufallsstartwert ab. Führen wir dasselbe Trainingsskript zweimal aus, erhalten wir daher leicht unterschiedliche Testgenauigkeiten. Angenommen, wir trainieren Modell A und Modell B mit jeweils 20 Zufallsstartwerten und halten ihre Testgenauigkeiten fest. Der Zufallsstartwert verschiebt das Ergebnis eines einzelnen Laufs um etwa einen Prozentpunkt, während Modell B tatsächlich um 0.8 Prozentpunkte besser ist. Können wir allein anhand dieser 40 Zahlen erkennen, dass B besser ist? Als Maß bietet sich die Differenz der beiden mittleren Genauigkeiten an. Sie dient uns als *Teststatistik* (engl. test statistic).

<!--en:920bacf2-->
Jeder Hypothesentest beginnt mit einer skeptischen Behauptung, die er zu widerlegen versucht. Hier besagt die *Nullhypothese* $H_0$, dass beide Listen von Genauigkeiten aus derselben Verteilung stammen. Die Modelle schneiden also im Mittel gleich gut ab. Die *Alternativhypothese* $H_A$ besagt, dass ihre Verteilungen verschieden sind. Wir fragen nun: Wenn $H_0$ zuträfe, wie überraschend wäre ein mindestens so großer Unterschied wie der beobachtete? Wäre ein solcher Unterschied durch Zufall allein sehr unwahrscheinlich, *verwerfen* wir $H_0$. Andernfalls *verwerfen wir die Hypothese nicht*. Diese asymmetrische Formulierung ist Absicht. Dass wir $H_0$ nicht verwerfen, beweist ebenso wenig die Gleichwertigkeit der Modelle, wie ein Freispruch vor Gericht die Unschuld beweist.

<!--en:1f4c3ada-->
### Permutationstest

<!--en:21d68828-->
Um zu beurteilen, ob ein beobachteter Unterschied unter $H_0$ überrascht, brauchen wir seine Verteilung, wenn $H_0$ gilt. Sie heißt *Nullverteilung* (engl. null distribution). Mit einem *Permutationstest* können wir diese Verteilung direkt aus den Daten konstruieren, ohne eine Gauß-Verteilung oder eine andere parametrische Form vorauszusetzen :cite:`Fisher.1935`. Stammen beide Listen tatsächlich aus derselben Verteilung, enthalten die Gruppenbezeichnungen „A“ und „B“ keinerlei Information. Jede Aufteilung der 40 Zahlen in zwei Gruppen zu je 20 wäre ebenso wahrscheinlich wie die beobachtete. Wir mischen die Gruppenbezeichnungen, berechnen die Differenz der Mittelwerte neu und wiederholen das $R$-mal. So sehen wir, welche Unterschiede durch reinen Zufall entstehen. Die folgende Hilfsfunktion verarbeitet alle $R$ Permutationen vektorisiert und zählt, wie oft die neu berechnete Differenz betragsmäßig mindestens so groß ist wie die beobachtete.

<!--en:8dc0739c-->
Zählen wir die beobachtete Zuordnung als eine der $R+1$ Permutationen mit, erhalten wir den *$p$-Wert*:

<!--en:36c7221a-->
Er schätzt die Wahrscheinlichkeit, unter $H_0$ einen mindestens so extremen Unterschied wie den beobachteten zu erhalten. Ein sehr kleiner $p$-Wert bedeutet, dass ein so großer Unterschied selten wäre, wenn die Gruppenbezeichnungen bedeutungslos wären. Dieser $p$-Wert ist *zweiseitig*: Große Unterschiede in beide Richtungen sprechen gegen $H_0$. (Interessierte uns nur, ob B besser als A ist, würde ein einseitiger Test nur permutierte Differenzen zählen, die mindestens so groß sind wie die beobachtete positive Differenz.)

<!--en:882d0c79-->
Bevor wir die Daten betrachten, legen wir ein *Signifikanzniveau* $\alpha$ fest, üblicherweise $\alpha=0.05$. Wir vereinbaren, $H_0$ zu verwerfen, wenn $p\le\alpha$ gilt. Simulieren wir mit den obigen Zahlen ein Experiment, um den Test zu beobachten.

<!--en:ee82cc90-->
Da wir in jedem Experiment neue Zufallsstartwerte ziehen, schwankt der beobachtete Unterschied von Lauf zu Lauf um den wahren Wert $0.008$. Meist fällt $p$ unter $0.05$ und der Test verwirft $H_0$. In einer merklichen Minderheit der Läufe bleibt die Ablehnung jedoch aus. Das kann auch hier passieren. Bei Schwankungen von einem ganzen Prozentpunkt reichen 20 Zufallsstartwerte je Modell nicht immer aus, um eine Verbesserung um 0.8 Prozentpunkte zu erkennen. Gleich messen wir, wie oft das geschieht. Wir sehen dann, dass es am Stichprobenumfang des Experiments liegt und kein Mangel des Tests ist.

<!--en:e0791c8d-->
Zeichnen wir ein Histogramm der permutierten Differenzen und tragen wir die beobachtete Differenz sowie ihr negatives Spiegelbild ein, wird die Nullverteilung sichtbar.

<!--en:4cc8a21b-->
Obwohl der Permutationstest keine Verteilungsform voraussetzt, bilden die permutierten Differenzen eine Glockenkurve um null. Die beobachtete Differenz liegt typischerweise weit außen in einem der beiden Randbereiche. Dasselbe Verfahren funktioniert für *jede* zusammenfassende Statistik zweier Gruppen, etwa für die Differenz ihrer Mediane, Fehlerraten oder Validierungsverluste. (Für den Mittelwertvergleich in geschlossener Form ist der $t$-Test nach Welch das klassische Gegenstück.) Jeder Hypothesentest folgt derselben Reihenfolge: Wir formulieren $H_0$ und $H_A$, legen $\alpha$ fest und wählen den Stichprobenumfang so groß, dass ein relevanter Effekt nachweisbar ist. Erst dann erheben wir die Daten, berechnen den $p$-Wert und verwerfen $H_0$, falls $p\le\alpha$ gilt.

<!--en:2ade0f15-->
### Der $z$-Test

<!--en:951e309a-->
Durch Permutation konstruieren wir die Nullverteilung empirisch. Nehmen wir dagegen gaußverteilte Daten mit bekannter Varianz an, erhalten wir geschlossene Formeln für den $p$-Wert und die Wahrscheinlichkeit, einen Effekt nachzuweisen. Seien $x_1,\ldots,x_n$ unabhängig und identisch gemäß $\mathcal N(\mu,\sigma^2)$ verteilt, wobei $\sigma$ bekannt sei. Wir wollen $H_0\!:\mu=0$ testen. Dazu standardisieren wir den Stichprobenmittelwert mit seinem Standardfehler und erhalten die Teststatistik

<!--en:cca82d0e-->
Unter $H_0$ gilt $\bar x\sim\mathcal N(0,\sigma^2/n)$, also ist $T$ standardnormalverteilt. Mit $\Phi$ bezeichnen wir die Verteilungsfunktion der Standardnormalverteilung und mit $z_q$ ihr $q$-Quantil. Es ist durch $\Phi(z_q)=q$ definiert (zum Beispiel $z_{0.975}\approx1.96$). Der zweiseitige $p$-Wert beträgt $2(1-\Phi(|T|))$. Daher ist die Bedingung $p\le\alpha$ äquivalent zu $|T|\ge z_{1-\alpha/2}$. Bei $\alpha=0.05$ verwerfen wir $H_0$, wenn der Stichprobenmittelwert mehr als $1.96$ Standardfehler von $0$ entfernt liegt. :numref:`fig_mdl-statistical_significance` hebt die beiden Randbereiche hervor, in denen wir die Nullhypothese verwerfen.

<!--en:922bf473-->
![Ablehnungsbereich des zweiseitigen $z$-Tests bei $\alpha=0.05$. Unter $H_0$ folgt die standardisierte Teststatistik einer Standardnormalverteilung. Der mittlere Bereich hat die Wahrscheinlichkeit $1-\alpha=0.95$, die beiden Randbereiche jenseits von $\pm1.96$ zusammen die Wahrscheinlichkeit $\alpha$. Ein Wert der Teststatistik in einem der Randbereiche wäre unter $H_0$ selten. Daher wird $H_0$ verworfen.](../img/mdl-prob-significance.svg)

<!--en:4d7732eb-->
Für unseren Modellvergleich gibt es eine entsprechende Zweistichprobenvariante. Bei $k$ unabhängigen Zufallsstartwerten je Modell und einer Standardabweichung von $\sigma$ über die Startwerte hat die Differenz der Mittelwerte $\bar x_B-\bar x_A$ die Varianz $2\sigma^2/k$. Somit ist $T=(\bar x_B-\bar x_A)/(\sigma\sqrt{2/k})$ unter $H_0$ standardnormalverteilt. Mit $\sigma=0.01$ und $k=20$ ergibt sich für die Differenz ein Standardfehler von $0.0032$. Eine wahre Verbesserung von $0.008$ liegt damit $2.5$ Standardfehler über null, also jenseits der Schwelle von $1.96$, aber nur knapp.

<!--en:827424d9-->
### Zwei Fehlerarten und Teststärke

<!--en:becf5cb2-->
Bei jedem Hypothesentest können zwei Arten von Fehlern auftreten. Eine gute Versuchsplanung hält beide unter Kontrolle. Ein *Fehler erster Art* (falsch positives Ergebnis) liegt vor, wenn wir eine wahre Nullhypothese $H_0$ verwerfen und damit eine Verbesserung behaupten, die es nicht gibt. Ein *Fehler zweiter Art* (falsch negatives Ergebnis) liegt vor, wenn wir eine falsche Nullhypothese $H_0$ nicht verwerfen und eine echte Verbesserung übersehen. Die Wahrscheinlichkeiten dieser Fehler sind

<!--en:27e11e54-->
Dabei berechnen wir $\beta$ für eine bestimmte Alternative, etwa einen wahren Unterschied von 0.8 Prozentpunkten. Die Gegenwahrscheinlichkeit $1-\beta$ heißt *Teststärke* (engl. statistical power). Sie gibt an, mit welcher Wahrscheinlichkeit das Experiment den tatsächlich vorhandenen Effekt erkennt. :numref:`fig_mdl-type-i-ii-matrix` fasst die vier möglichen Ergebnisse zusammen.

<!--en:abc13f8b-->
![$2\times 2$-Entscheidungsmatrix eines Hypothesentests. Auf der Diagonalen liegen die richtigen Entscheidungen, daneben der Fehler erster Art (Rate $\alpha$) und der Fehler zweiter Art (Rate $\beta$). Die Zelle für die korrekte Ablehnung gibt die Teststärke ($1-\beta$) an.](../img/mdl-prob-type-i-ii-matrix.svg)

<!--en:ad311da2-->
Warum garantiert die Regel „verwerfen, wenn $p\le\alpha$“ eine Fehlerwahrscheinlichkeit erster Art von genau $\alpha$? Weil unter $H_0$ der $p$-Wert auf $[0,1]$ gleichverteilt ist. Der Beweis lohnt sich: Er verdeutlicht, was ein Signifikanzniveau bedeutet, und hilft uns später, das multiple Testen zu verstehen.

<!--en:1c5acfb3-->
**Proposition (Gleichverteilung der $p$-Werte unter der Nullhypothese).** *Die Teststatistik $T$ habe die stetige Verteilungsfunktion $F$ unter $H_0$. Sei $p=1-F(T)$ (einseitig) oder $p=2\min\{F(T),1-F(T)\}$ (zweiseitig). Dann gilt $P_{H_0}(p\le\alpha)=\alpha$ für jedes $\alpha\in[0,1]$.*

<!--en:1e71fea8-->
**Beweis.** Zuerst zeigen wir, dass $U=F(T)$ auf $[0,1]$ gleichverteilt ist. Für jedes $u\in(0,1)$ sei $t_u$ der größte Wert von $t$ mit $F(t)=u$. Er existiert, weil $F$ stetig ist und von $0$ bis $1$ verläuft. Da $F$ monoton wächst, gilt $F(T)\le u$ genau dann, wenn $T\le t_u$ gilt. Daraus folgt $P(U\le u)=F(t_u)=u$. Für den einseitigen $p$-Wert ergibt sich unmittelbar $P(1-U\le\alpha)=P(U\ge1-\alpha)=\alpha$. Für den zweiseitigen $p$-Wert bedeutet das Ereignis $2\min\{U,1-U\}\le\alpha$, dass entweder $U\le\alpha/2$ oder $U\ge1-\alpha/2$ gilt. Diese beiden disjunkten Ereignisse haben jeweils die Wahrscheinlichkeit $\alpha/2$. Ihre Vereinigung hat also die Wahrscheinlichkeit $\alpha$. $\blacksquare$

<!--en:adbccce2-->
Für den $p$-Wert des Permutationstests liefert ein einfacheres Zählargument dieselbe Garantie. Unter $H_0$ sind die beobachtete Zuordnung und die $R$ zufälligen Permutationen austauschbar. Der beobachtete Unterschied nimmt also jeden der $R+1$ Ränge mit gleicher Wahrscheinlichkeit ein. Folglich gilt $p\le\alpha$ nur, wenn der beobachtete Unterschied zu den $\lfloor\alpha(R+1)\rfloor$ betragsmäßig größten gehört. Damit folgt $P_{H_0}(p\le\alpha)\le\alpha$ (Rangbindungen können $p$ nur erhöhen). Deshalb ist das $+1$ in Zähler und Nenner wichtig: Es stellt sicher, dass ein Test zum Niveau $\alpha$ eine wahre Nullhypothese höchstens in einem Anteil $\alpha$ der Fälle verwirft.

<!--en:7ac76fb0-->
Wie sind die $p$-Werte verteilt, wenn $H_0$ falsch ist? Um den Unterschied zu sehen, simulieren wir 2000 Zweistichproben-$z$-Tests mit 20 Zufallsstartwerten je Modell. Zuerst gibt es keinen wahren Unterschied, dann beträgt er wie in unserem Beispiel $0.008$. Die resultierenden $p$-Werte tragen wir in ein Histogramm mit zehn gleich breiten Klassen ein.

<!--en:aa79aa03-->
Gilt $H_0$ (Unterschied $0.000$), ist das Histogramm flach. Etwa ein Zehntel der $p$-Werte fällt in jede Klasse, rund 5% liegen bei oder unter $0.05$. Bei einem wahren Unterschied von $0.008$ häufen sich die $p$-Werte nahe null. Rund 80% liegen unter $0.1$ und 70% bis 74% bei oder unter $0.05$. Dieser Anteil von Ablehnungen schätzt die Teststärke, für die unsere geschlossene Formel weiter unten $0.72$ vorhersagt.

<!--en:38e047ed-->
Bevor wir die Formel herleiten, betrachten wir zwei verbreitete Fehlinterpretationen eines $p$-Werts :cite:`Wasserstein.Lazar.2016`. Erstens ist ein $p$-Wert die Wahrscheinlichkeit, mindestens so extreme Daten zu beobachten, *unter der Annahme, dass $H_0$ gilt*. Er ist *nicht* die Wahrscheinlichkeit, dass $H_0$ gilt (dafür bräuchten wir eine A-priori-Verteilung und den Satz von Bayes). Zweitens beweist ein großer $p$-Wert nicht, dass $H_0$ gilt. Er besagt lediglich, dass das Experiment keinen Effekt nachweisen konnte. Vielleicht war seine Teststärke zu gering.

<!--en:5380804c-->
Berechnen wir die Teststärke für den Einstichproben-$z$-Test explizit. Beträgt der wahre Mittelwert $\mu\ne0$, ist die Statistik $T$ normalverteilt mit Erwartungswert $\sqrt n\,\mu/\sigma$ und Varianz 1. Wir definieren den *standardisierten Effekt* $\delta=|\mu|/\sigma$ als Effektgröße in Einheiten der Standardabweichung des Rauschens. Der Test verwirft die Nullhypothese, wenn $|T|\ge z_{1-\alpha/2}$ gilt. Dies geschieht mit der Wahrscheinlichkeit

<!--en:35d69eea-->
Dabei vernachlässigen wir die sehr kleine Wahrscheinlichkeit im gegenüberliegenden Randbereich $\Phi(-\delta\sqrt n-z_{1-\alpha/2})$. Die Teststärke wächst also mit der Effektgröße $\delta$ und mit $\sqrt n$. Bei einem großen Effekt $\delta=1$ liefern bereits $n=8$ Beobachtungen die Teststärke $\Phi(2.83-1.96)\approx0.81$.

<!--en:9cdad69b-->
Bei unserem Zweistichprobenvergleich der Modelle hat die Teststatistik den Erwartungswert $\delta\sqrt{k/2}$ mit $\delta=0.008/0.010=0.8$. Die Teststärke des $z$-Tests mit $k$ Zufallsstartwerten je Modell ist also $\Phi(0.8\sqrt{k/2}-1.96)$. Sie beträgt etwa $0.72$ bei 20 Startwerten und nur $0.24$ bei 5 Startwerten. Diese Formel setzt voraus, dass wir $\sigma$ im Voraus kennen. Prüfen wir nun mit je 400 neuen simulierten Experimenten für beide Anzahlen von Startwerten, wie der Permutationstest abschneidet, der keine vorherige Schätzung erfordert.

<!--en:93e44433-->
Über 400 Läufe verwirft der Permutationstest die Nullhypothese bei 20 Zufallsstartwerten in rund 70% der Experimente und bei 5 Startwerten nur in rund 20% (beide Anteile schwanken von Durchgang zu Durchgang um etwa zwei Prozentpunkte). Beide Werte liegen etwas unter der Vorhersage für den $z$-Test, weil der Permutationstest das Rauschen aus den Daten schätzt. Das kostet etwas Teststärke. Selbst bei 20 Startwerten übersieht unser Experiment eine echte Verbesserung um 0.8 Prozentpunkte also in etwa drei von zehn Fällen, bei 5 Startwerten in vier von fünf Fällen. Bleibt bei einer Teststärke von $0.2$ die Ablehnung aus, sagt uns das fast nichts. In der Praxis strebt man meist eine Teststärke von mindestens $0.8$ an.

<!--en:f53d258d-->
### Stichprobenumfang

<!--en:68ce6eb1-->
Statt erst im Nachhinein festzustellen, dass die Teststärke zu gering war, können wir den Stichprobenumfang im Voraus planen. Setzen wir die Formel für die Teststärke gleich $1-\beta$, erhalten wir $\delta\sqrt n-z_{1-\alpha/2}=z_{1-\beta}$. Auflösen nach $n$ liefert den Stichprobenumfang, den wir brauchen, um einen standardisierten Effekt $\delta$ beim Signifikanzniveau $\alpha$ mit der Teststärke $1-\beta$ zu erkennen:

<!--en:1df798ad-->
Beachten Sie die starke Abhängigkeit von der Effektgröße gemäß $1/\delta^2$. Halbieren wir die nachzuweisende Effektgröße, brauchen wir viermal so viele Daten. Für die üblichen Vorgaben $\alpha=0.05$ und Teststärke $0.8$ gilt $z_{0.975}\approx1.960$ und $z_{0.8}\approx0.842$. Daraus folgt die praktische Faustregel $n\approx7.85/\delta^2$. :numref:`fig_mdl-power` zeigt die resultierenden Kurven der Teststärke für verschiedene Effektgrößen.

<!--en:3554721a-->
![Teststärke des zweiseitigen $z$-Tests bei $\alpha=0.05$ in Abhängigkeit vom Stichprobenumfang $n$, jeweils für einen standardisierten Effekt $\delta$. Jede Kurve beginnt nahe $\alpha$ (gepunktete Linie), wo der Test die Nullhypothese nur zufällig verwirft, und steigt gegen $1$. Die gestrichelte Linie markiert die angestrebte Teststärke $0.8$. Diese wird bei $n\propto 1/\delta^2$ erreicht: etwa 8 Beobachtungen für $\delta=1$ und bis zu etwa 78 500 für $\delta=0.01$.](../img/mdl-prob-power.svg)

<!--en:cf257f58-->
Vergleichen wir zwei unabhängige Gruppen mit je $k$ Beobachtungen und gemeinsamer Standardabweichung $\sigma$, verdoppelt sich die Varianz der Differenz auf $\textrm{Var}(\bar x_B-\bar x_A)=2\sigma^2/k$. Entsprechend verdoppelt sich auch der erforderliche Stichprobenumfang je Gruppe:

<!--en:07f3e8f8-->
Für unseren Modellvergleich mit $\delta=0.8$ erfordert eine Teststärke von 80% somit $k=2\cdot7.85/0.64\approx25$ Zufallsstartwerte je Modell. Deshalb verfehlten wir mit 20 Startwerten das Ziel knapp.

<!--en:1f74badb-->
### Gepaarte Vergleiche

<!--en:e6d5e44d-->
Kehren wir zur Eingangsfrage zurück: Zwei Klassifikatoren beantworteten 912 beziehungsweise 921 von 1000 Fragen richtig. Behandeln wir ihre Ergebnisse als unabhängig, geht der Unterschied im Rauschen unter. Entscheidend ist jedoch, dass beide Modelle *dieselben* 1000 Fragen beantwortet haben. Manche Fragen sind für beide Modelle leicht, andere für beide schwer. Bei einer ungepaarten Analyse geht diese gemeinsame Schwierigkeit vollständig ins Rauschen ein.

<!--en:863ae1d5-->
Ein *gepaarter* Vergleich eliminiert diese gemeinsame Variation. Dazu betrachten wir für jede Frage direkt die Differenz $d_i\in\{-1,0,1\}$, also den Richtigkeitsindikator von Modell B abzüglich desjenigen von Modell A bei Frage $i$. Stimmen die Ergebnisse beider Modelle überein, gilt $d_i=0$. Angenommen, sie unterscheiden sich bei 41 Fragen: Bei 25 liegt nur Modell B richtig, bei 16 nur Modell A. Das ergibt einen Nettovorsprung von $+9$ (0.9 Prozentpunkte). Unter der Nullhypothese, dass die Modelle austauschbar sind, gleicht jeder Unterschied einem fairen Münzwurf, der mit gleicher Wahrscheinlichkeit A oder B begünstigt. Eine passende Simulation unter der Nullhypothese wechselt daher für jedes von null verschiedene $d_i$ zufällig das Vorzeichen. Dieses Verfahren heißt *Permutationstest mit Vorzeichenwechsel* (engl. sign-flip permutation test). Wenden wir ihn auf die 41 Unterschiede an.

<!--en:744450f8-->
Der Test mit Vorzeichenwechsel liefert $p\approx0.21$ (ersetzen wir die zufälligen Vorzeichenwechsel durch die exakte Binomialwahrscheinlichkeit, erhalten wir den McNemar-Test). Die ungepaarte Analyse derselben Anzahlen ergibt dagegen eine Differenz von nur $0.7$ Standardfehlern und $p\approx0.47$. Die Paarung halbiert den Standardfehler der Differenz von $0.012$ auf $\sqrt{41}/1000\approx0.006$, da die 959 Fragen mit gleichem Ergebnis nicht mehr zum Rauschen beitragen. Dennoch ist eine Aufteilung von 25 zu 16 bei 41 Unterschieden zu wenig überzeugend: Neun Fragen Vorsprung bei einem Test mit 1000 Fragen belegen nicht, dass Modell B besser ist.

<!--en:022490bf-->
Besonders wichtig ist die Paarung, wenn Sie eine Auswertung von Grund auf planen. Angenommen, zwei Modelle haben wahre Genauigkeiten von 90% und 91%. Bei 4% der Fragen unterscheiden sich ihre Ergebnisse (nur B liegt bei 2.5% richtig, nur A bei 1.5%). Dann hat $d_i$ den Erwartungswert $0.01$ und die Varianz $\mathbb{E}[d_i^2]-0.01^2=0.04-0.0001=0.0399$. Daraus ergibt sich ein standardisierter Effekt von $\delta=0.01/\sqrt{0.0399}\approx0.050$. Nach :eqref:`eq_mdl-power-sample-size` erreicht ein gepaarter Test mit etwa $7.85\cdot0.0399/0.01^2\approx3{,}100$ Fragen eine Teststärke von 80%. Werten wir die beiden Modelle dagegen auf getrennten, unabhängigen Testdatensätzen aus, bleibt je Fragenpaar eine ungepaarte Varianz von $0.9\cdot0.1+0.91\cdot0.09=0.1719$. Wir brauchen dann rund $7.85\cdot0.1719/0.01^2\approx13{,}500$ Fragen *je Modell*.

<!--en:95a533fa-->
In diesem Beispiel verkleinert die Paarung jeden Testdatensatz um mehr als den Faktor vier und den gesamten Aufwand für die Vergabe der Labels um fast den Faktor neun. Stimmen die Modelle noch häufiger überein, wächst die Einsparung weiter. Werten Sie konkurrierende Modelle daher stets auf demselben Testdatensatz aus und analysieren Sie die Differenzen je Frage. Dasselbe Prinzip gilt für Zufallsstartwerte: Verwenden zwei Modelle dieselben Startwerte beim Training, wechseln Sie im Test die Vorzeichen der Differenzen je Startwert (Übung :exref:`ex_mdl-statistics-paired-comparisons`).

<!--en:450378e3-->
### Multiples Testen

<!--en:e3c7ede7-->
Alle bisherigen Signifikanzgarantien, $P(\textrm{reject}\mid H_0)=\alpha$, setzen einen *einzigen* im Voraus festgelegten Test voraus. Im maschinellen Lernen laufen bei der Hyperparametersuche jedoch regelmäßig Dutzende Vergleiche gleichzeitig. Angenommen, Sie vergleichen 20 neue Konfigurationen mit einer Baseline, obwohl keine tatsächlich besser ist, und testen jede zum Niveau $\alpha=0.05$. Bei unabhängigen Tests beträgt die Wahrscheinlichkeit, dass mindestens eine Konfiguration fälschlich signifikant erscheint, $1-0.95^{20}\approx0.64$. Wer nur die Gewinner auswählt oder eine Analyse so lange anpasst, bis $p\le\alpha$ gilt (*$p$-Hacking*), erzielt selbst mit reinem Rauschen zuverlässig scheinbar publikationsreife Verbesserungen.

<!--en:4a4db39f-->
Die einfachste Absicherung ist die *Bonferroni-Korrektur* :cite:`Bonferroni.1936`: Bei $m$ Tests senken wir jede einzelne Schwelle auf $\alpha/m$. Sei $A_j$ das Ereignis, dass Test $j$ eine wahre Nullhypothese fälschlich verwirft. Dann liefert die Vereinigungsabschätzung

<!--en:01e5ef4b-->
Diese Schranke ist bemerkenswert allgemein. Sie begrenzt die Wahrscheinlichkeit, *auch nur eine* falsche Entdeckung zu machen, auf $\alpha$, unabhängig davon, wie die $m$ Tests miteinander korrelieren. Bei 20 unabhängigen Tests zum Niveau $0.05/20$ sinkt die Wahrscheinlichkeit mindestens eines falsch positiven Ergebnisses von $0.64$ auf $1-(1-0.0025)^{20}\approx0.049$.

<!--en:397c9b86-->
Geht $m$ jedoch in die Hunderte oder Tausende, wird die Bonferroni-Schwelle durch die Division durch $m$ so streng, dass echte Entdeckungen untergehen. Bei umfangreichen Reihenuntersuchungen begrenzt man stattdessen die *Falschentdeckungsrate* (engl. false discovery rate), den erwarteten Anteil fälschlich verworfener Hypothesen unter allen verworfenen Hypothesen. Das Benjamini-Hochberg-Verfahren :cite:`Benjamini.Hochberg.1995` hält diese Rate bei oder unter $\alpha$, wenn die $p$-Werte unabhängig oder positiv abhängig sind. Übung :exref:`ex_mdl-statistics-multiple-comparisons` führt die Regel ein und prüft sie in einer Simulation.

<!--en:b5917577-->
## Konfidenzintervalle

<!--en:8d83ce42-->
Ein $p$-Wert beantwortet eine Ja-nein-Frage zu einem einzelnen möglichen Parameterwert. In der Praxis interessiert uns meist mehr: der gesamte Bereich der mit den Daten vereinbaren Parameterwerte, direkt angegeben als Fehlerbalken um unseren Schätzwert.

<!--en:6bbade3f-->
### Definition und Interpretation

<!--en:6fdf365e-->
Ein *Konfidenzintervall* entsteht nach einer Regel, die den festen, unbekannten Parameter bei wiederholten Experimenten mit einer vorgegebenen Häufigkeit erfasst :cite:`Neyman.1937`. Formal ist ein $1-\alpha$-Konfidenzintervall ein aus den Daten berechnetes zufälliges Intervall $C_n$ mit der Eigenschaft

<!--en:4365ef00-->
Wir schreiben bewusst $C_n\ni\theta$ statt $\theta\in C_n$, um zu betonen, welche Größe zufällig ist: $\theta$ ist eine feste Zahl, während sich das Intervall $C_n$ von Datensatz zu Datensatz verschiebt. Die Wahrscheinlichkeit $1-\alpha$ ist eine Eigenschaft des *Verfahrens*. Über viele unabhängige Experimente hinweg enthalten mindestens 95% der so gewonnenen 95%-Konfidenzintervalle den Wert $\theta$. Jedes einzelne berechnete Intervall enthält $\theta$ oder verfehlt den Wert. Am Intervall selbst lässt sich nicht ablesen, welcher Fall vorliegt. :numref:`fig_mdl-prob-coverage` veranschaulicht dies mit 100 Intervallen aus 100 unabhängigen Stichproben. Viele in der Praxis verwendete Intervalle erfüllen diese Garantie nur im Grenzfall großer Stichproben. Für ein *asymptotisches* $1-\alpha$-Intervall gilt $\liminf_{n\to\infty}P_\theta(C_n\ni\theta)\ge1-\alpha$ für jedes $\theta$.

<!--en:80e741db-->
![Hundert 95%-Konfidenzintervalle, jeweils aus einer eigenen unabhängigen Stichprobe, im Vergleich zum festen wahren Mittelwert (gestrichelte Linie). Die meisten Intervalle enthalten den wahren Wert, die orangefarbenen verfehlen ihn. Die 95%-Garantie beschreibt die langfristige Überdeckung des Verfahrens. Einem einzelnen Intervall ist nicht anzusehen, ob es den wahren Wert enthält.](../img/mdl-prob-coverage.svg)

<!--en:b06b64ac-->
### Das Gauß-Intervall

<!--en:a089bb5c-->
Beginnen wir mit dem einfachsten Fall, in dem jeder Schritt exakt ist. Seien $x_1,\ldots,x_n$ unabhängig und identisch gemäß $\mathcal N(\mu,\sigma^2)$ verteilt, wobei $\sigma$ bekannt sei. Wegen $\bar x\sim\mathcal N(\mu,\sigma^2/n)$ können wir die halbe Intervallbreite auf $h=z_{1-\alpha/2}\,\sigma/\sqrt n$ setzen. Dann liegt $\bar x$ höchstens im Abstand $h$ von $\mu$, und zwar mit Wahrscheinlichkeit $1-\alpha$. Stellen wir nun die Ungleichung zu einer Aussage über $\mu$ um:

<!--en:b3e94417-->
Somit gilt $P_\mu(\mu\in[\bar x-h,\bar x+h])=1-\alpha$ für jedes $\mu$. Hinter einem Konfidenzintervall steckt genau diese algebraische Umformung. Bei $\alpha=0.05$ lautet das Intervall $\bar x\pm1.96\,\sigma/\sqrt n$.

<!--en:fe74aa78-->
In der Praxis kennen wir $\sigma$ selten. Deshalb setzen wir die Stichprobenstandardabweichung $s$ aus :eqref:`eq_mdl-unbiased-var` ein. Sobald wir $s$ aus den Daten schätzen, ist die studentisierte Statistik $T=(\bar x-\mu)/(s/\sqrt n)$ nicht mehr normalverteilt. Für gaußverteilte Daten folgt sie der Student-$t$-Verteilung mit $n-1$ Freiheitsgraden :cite:`Student.1908`. Deren schwerere Verteilungsränder tragen der zusätzlichen Zufälligkeit von $s$ Rechnung. (Wir übernehmen die Verteilung hier ohne Beweis. Übung :exref:`ex_mdl-distributions-student-t` leitet sie als Gauß-Verteilung mit zufälliger Präzision her.) Ersetzen wir $z_{1-\alpha/2}$ durch das $t$-Quantil $t_{n-1,1-\alpha/2}$, erhalten wir

<!--en:3c86f07e-->
Dieses Intervall erreicht für gaußverteilte Daten die Überdeckung $1-\alpha$ bei jedem $n\ge2$. Bei kleinen Stichproben macht die Wahl des Quantils einen großen Unterschied: $t_{n-1,0.975}$ beträgt $12.71$ für $n=2$, $2.776$ für $n=5$ und $1.984$ für $n=100$. Das Quantil nähert sich $1.96$, wenn $n$ wächst. Das in der Praxis häufig verwendete Intervall setzt direkt diesen Grenzwert für große Stichproben ein:

<!--en:d304f64c-->
Dies ist ein asymptotisches 95%-Intervall, das für jedes endliche $n$ etwas zu schmal ist. Bei gaußverteilten Daten beträgt seine tatsächliche Überdeckung $P(|t_{n-1}|\le1.96)$. Sie liegt bei $n=100$ nahe am Sollwert ($0.947$), fällt aber auf $0.878$ bei $n=5$.

<!--en:d35d7447-->
Was geschieht, wenn die Daten nicht gaußverteilt sind? Nach dem zentralen Grenzwertsatz (:numref:`subsec_mdl-gaussian-clt`) konvergiert $\sqrt n(\bar x-\mu)/\sigma$ bei endlicher Varianz in Verteilung gegen eine Standardnormalverteilung. Außerdem konvergiert $s$ nach dem Gesetz der großen Zahlen gegen $\sigma$. Somit bleibt :eqref:`eq_mdl-gauss_confidence` ein asymptotisches 95%-Intervall für den Mittelwert *jeder* Verteilung mit endlicher Varianz. Wie groß $n$ sein muss, hängt von der Verteilungsform ab. Schiefe Verteilungen erfordern mehr Beobachtungen (Übung :exref:`ex_mdl-statistics-coverage-small-n`). Bei Genauigkeiten nahe 100% brauchen wir noch wesentlich mehr, weil sich auf vielen Testdatensätzen eine perfekte Genauigkeit mit Stichprobenvarianz null ergibt (Übung :exref:`ex_mdl-statistics-delta-boundary`). Die halbe Intervallbreite fällt in allen Fällen wie $1/\sqrt n$. Für einen halb so großen Fehlerbalken brauchen wir also viermal so viele Daten.

<!--en:70b1e2c5-->
Die Testgenauigkeit ist der Mittelwert von 0/1-Indikatoren mit $s^2\approx\hat\theta(1-\hat\theta)$. Dieselbe Formel liefert daher $\hat\theta\pm1.96\sqrt{\hat\theta(1-\hat\theta)/n}$, das sogenannte *Wald-Intervall*. Berechnen wir es für unseren Klassifikator mit 1000 Fragen und für unser Münzbeispiel (9-mal Kopf bei 13 Würfen).

<!--en:54ee8e6e-->
Für den Klassifikator erhalten wir $91.2\%\pm1.8$ Prozentpunkte. Für die Münze schließt das Intervall $(0.44,0.94)$ den Wert $\tfrac12$ klar ein. 13 Würfe reichen also bei Weitem nicht aus, um auf eine unfaire Münze zu schließen. Um ihre Abweichung von einer fairen Münze bis auf $\pm0.05$ einzugrenzen, bräuchten wir rund $1.96^2\cdot0.69\cdot0.31/0.05^2\approx330$ Würfe.

<!--en:9626b63a-->
Vergleichen wir dies mit einem bayesschen *Glaubwürdigkeitsintervall* (engl. credible interval) für dieselbe Münze. Die bayessche Analyse behandelt $\theta$ selbst als zufällig, gibt dafür eine A-priori-Verteilung vor und bestimmt anhand der beobachteten Würfe ein Intervall mit A-posteriori-Wahrscheinlichkeit $1-\alpha$. Bei einer gleichförmigen A-priori-Verteilung lautet die A-posteriori-Verteilung $\mathrm{Beta}(10,5)$ (:numref:`sec_mdl-exponential-family`). Ihr zentrales 95%-Glaubwürdigkeitsintervall zwischen dem 2.5%- und dem 97.5%-Quantil beträgt $(0.42,0.87)$. Obwohl die Zahlen hier ähnlich aussehen, treffen die Intervalle grundverschiedene Aussagen. Nach dem Glaubwürdigkeitsintervall liegt $\theta$ unter der gewählten A-priori-Verteilung im Intervall $(0.42,0.87)$, und zwar mit Wahrscheinlichkeit $0.95$. Beim Konfidenzintervall schließt das Konstruktionsverfahren den festen wahren Wert $\theta$ in mindestens 95% der wiederholten Experimente ein, unabhängig vom Wert von $\theta$. Das Wald-Intervall erfüllt seine Garantie allerdings nur asymptotisch. Bei 13 Würfen beträgt seine tatsächliche Überdeckung rund 91% für $\theta=\frac12$ und fällt unter 50% für $\theta=0.05$.

<!--en:7439411d-->
Da die Überdeckung eine Häufigkeit bei wiederholten Experimenten ist, können wir sie direkt durch Simulation prüfen. Ziehen wir 1000 unabhängige Datensätze aus je $n=100$ standardnormalverteilten Beobachtungen. Für jeden berechnen wir das asymptotische Intervall :eqref:`eq_mdl-gauss_confidence` und zählen, wie viele Intervalle den wahren Mittelwert $0$ enthalten.

<!--en:67357641-->
Das erste Intervall enthält $0$ in etwa 95% der Läufe. Die Gesamtzahl der Treffer liegt um 947 (typischerweise innerhalb von $\pm14$) statt um 950. Dass im Mittel drei Treffer fehlen, ist keine zufällige Schwankung. Bei $n=100$ beträgt das exakte $t$-Quantil $t_{99,0.975}=1.984$ statt $1.96$. Die wahre Überdeckung des Intervalls mit Faktor 1.96 ist daher $P(|t_{99}|\le1.96)=0.947$. Die Streuung von $\pm14$ um 947 entspricht zwei Standardabweichungen der Binomialverteilung, $2\sqrt{1000\cdot0.947\cdot0.053}\approx14$. Genauso erhalten wir den Fehlerbalken der mittleren Genauigkeit über $k$ Zufallsstartwerte beim Training: Wir teilen die Standardabweichung über die Startwerte durch $\sqrt k$ und multiplizieren mit dem passenden Quantil.

<!--en:fb282910-->
### Tests und Intervalle

<!--en:8b92aac1-->
Die Zahl $1.96$ ist gerade zweimal aufgetaucht: als Ablehnungsschwelle des $z$-Tests und als Faktor im Gauß-Konfidenzintervall. Das ist kein Zufall. Tests und Konfidenzintervalle sind zwei Seiten derselben Medaille.

<!--en:398dd4a9-->
Betrachten wir gaußverteilte Daten mit bekanntem $\sigma$. Ein zweiseitiger $z$-Test von $H_0\!:\mu=\mu_0$ verwirft $H_0$ genau dann nicht, wenn $|T|\le z_{1-\alpha/2}$ gilt, wobei $T=\sqrt n(\bar x-\mu_0)/\sigma$. Stellen wir die Ungleichung um, erhalten wir $|\bar x-\mu_0|\le h$ mit $h=z_{1-\alpha/2}\,\sigma/\sqrt n$. Das bedeutet, dass $\mu_0\in[\bar x-h,\,\bar x+h]$ gilt. Kurz gesagt: *Ein 95%-Konfidenzintervall umfasst genau die Parameterwerte, die ein Hypothesentest zum Niveau 5% nicht verwirft*. Seine Überdeckungsgarantie folgt direkt aus der Fehlerwahrscheinlichkeit erster Art des Tests. Für jedes $\mu$ gilt $P_\mu(\mu\in[\bar x-h,\bar x+h])=1-P_\mu(\textrm{the test of }H_0\!:\mu\textrm{ rejects})=1-\alpha$.

<!--en:5b739877-->
Diese Dualität ist besonders nützlich, weil sie für *jede* Familie von Hypothesentests gilt. Haben wir für jeden möglichen Wert $\theta_0$ einen Test zum Niveau $\alpha$ für $H_0\!:\theta=\theta_0$, so fassen wir alle Werte $\theta_0$ zusammen, die der Test nicht verwirft. Nach demselben kurzen Argument bilden sie eine $1-\alpha$-Konfidenzmenge. Fehlt eine explizite Formel oder liefert sie unbrauchbare Ergebnisse, können wir stattdessen einen Test umkehren. Beantwortet ein Modell etwa 29 von 30 Fragen richtig, reicht das Wald-Intervall unsinnigerweise über die Genauigkeit $1$ hinaus. Die Umkehrung eines exakten Binomialtests liefert dagegen ein gültiges Intervall innerhalb von $[0,1]$ (Übung :exref:`ex_mdl-statistics-delta-boundary`).

<!--en:5f7258e1-->
Aus dieser Dualität ergibt sich auch eine aussagekräftigere Darstellung für den Modellvergleich: Geben Sie ein Konfidenzintervall für die Differenz an, etwa $\bar x_B-\bar x_A\pm1.96\,\sigma\sqrt{2/k}$ über $k$ Zufallsstartwerte. Es schließt $0$ genau dann aus, wenn der Hypothesentest die Nullhypothese verwirft. Zugleich zeigt es, wie groß die Verbesserung tatsächlich sein könnte.

<!--en:f0ab668e-->
### Die Delta-Methode

<!--en:ad317554-->
Oft interessiert uns anstelle von $\hat\theta$ selbst eine glatte Funktion $g(\hat\theta)$ davon, etwa die Log-Odds einer Genauigkeit, der Quotient zweier Verluste oder die relative Verringerung eines Fehlers. Wie überträgt sich unser Fehlerbalken auf den Wert von $g$? Ist $g$ an der Stelle $\theta$ differenzierbar, ergibt eine Taylor-Entwicklung erster Ordnung

<!--en:8daad20d-->
In der Umgebung von $\theta$ skaliert die Funktion $g$ kleine Abweichungen mit ihrer Steigung $|g'(\theta)|$. Bilden wir die Standardabweichungen und setzen wir $\hat\theta$ für den unbekannten Wert $\theta$ in $g'$ ein, erhalten wir die *Delta-Methode* :cite:`Wasserman.2013`:

<!--en:1e68caed-->
Probieren wir das mit konkreten Zahlen aus. Eine Genauigkeit von $\hat\theta=0.90$ bei $n=1{,}000$ Fragen hat den Standardfehler $\textrm{se}(\hat\theta)=\sqrt{0.9\cdot0.1/1000}\approx0.0095$. Die Log-Odds $g(\hat\theta)=\log\frac{\hat\theta}{1-\hat\theta}\approx2.20$ haben die Ableitung $g'(\hat\theta)=1/(\hat\theta(1-\hat\theta))\approx11.1$. Ihr Standardfehler beträgt somit ungefähr $11.1\cdot0.0095\approx0.105$.

<!--en:ecf0dc88-->
Dieselbe Linearisierung lässt sich direkt auf einen Vektor von Schätzern $\hat{\boldsymbol\theta}$ mit Kovarianzmatrix $\boldsymbol\Sigma$ übertragen. Sie ergibt $\textrm{Var}(g(\hat{\boldsymbol\theta}))\approx\nabla g^\top\boldsymbol\Sigma\,\nabla g$. Dabei sind die Kovarianzen außerhalb der Diagonalen von $\boldsymbol\Sigma$ wichtig: Bilden Sie den Quotienten zweier auf demselben Testdatensatz gemessener Verluste, so schwanken Zähler und Nenner gemeinsam.

<!--en:ef525813-->
Wie jede lineare Approximation hat die Delta-Methode klare Grenzen. Erstens setzt sie $g'(\theta)\ne0$ voraus. An einer stationären Stelle, etwa bei der Auswertung von $g(\theta)=(\theta-\frac12)^2$ an der Stelle $\theta=\frac12$, verschwindet der lineare Term. Dann schwankt $g(\hat\theta)$ mit der schnelleren Rate $1/n$ statt mit $1/\sqrt n$. Zweitens muss $\hat\theta$ in einem Bereich bleiben, in dem $g$ annähernd linear ist. Bei einer Genauigkeit sehr nahe an 100% ist das nicht mehr der Fall (Übung :exref:`ex_mdl-statistics-delta-boundary`). Hat eine Statistik überhaupt keine leicht berechenbare Ableitung, brauchen wir einen anderen Ansatz: das wiederholte Ziehen aus den beobachteten Daten.

<!--en:43400867-->
## Das Bootstrap-Verfahren

<!--en:9fdf1c09-->
Bisher haben wir jeden Standardfehler mit einer analytischen Formel berechnet. Viele praktisch wichtige Größen schwanken jedoch ebenso wie die Genauigkeit von Testdatensatz zu Testdatensatz, ohne dass es eine einfache Formel für ihre Varianz gibt. Beispiele sind der Stichprobenmedian, die AUC und der BLEU-Score. Das Bootstrap-Verfahren löst dieses Problem, indem es die Stichprobenverteilung direkt aus den Daten simuliert. Zuerst prüfen wir das Verfahren an einem Beispiel mit bekanntem Ergebnis. Dann wenden wir es dort an, wo eine einfache Formel fehlt. Schließlich untersuchen wir zwei aufschlussreiche Fälle, in denen es versagt.

<!--en:573bedc0-->
### Das Plug-in-Prinzip

<!--en:3956cf74-->
Kennten wir die wahre Verteilung $F$, die die Daten erzeugt, könnten wir die Stichprobenverteilung *jeder* Statistik leicht bestimmen: Wir würden Tausende neue Datensätze aus $F$ ziehen und die Statistik für jeden neu berechnen. Allerdings kennen wir $F$ nicht. Das *Bootstrap-Verfahren* :cite:`Efron.1979` wagt deshalb eine Ersetzung. An die Stelle der unbekannten Verteilung $F$ tritt die *empirische Verteilung* $\hat F_n$, die die Wahrscheinlichkeit $1/n$ auf jeden der $n$ beobachteten Datenpunkte legt. Eine neue Stichprobe vom Umfang $n$ aus $\hat F_n$ zu ziehen, heißt also, $n$ Punkte *mit Zurücklegen* aus unseren Daten zu ziehen. Der Algorithmus umfasst drei Schritte:

<!--en:1baebcb1-->
1. Wir ziehen $n$ Punkte mit Zurücklegen aus der beobachteten Stichprobe. So entsteht eine *Bootstrap-Stichprobe* (engl. bootstrap resample). Manche ursprünglichen Punkte kommen mehrfach vor, andere gar nicht.

<!--en:36ea29fa-->
2. Wir berechnen die Statistik $\hat\theta^*$ für diese Stichprobe neu.

<!--en:62fb47b1-->
3. Wir wiederholen dies $R$-mal und erhalten die Replikate $\hat\theta^*_1,\ldots,\hat\theta^*_R$.

<!--en:3f250d62-->
Die Standardabweichung dieser $R$ Replikate schätzt $\textrm{se}(\hat\theta)$. Ihr $\alpha/2$- und ihr $1-\alpha/2$-Quantil bilden die Grenzen des *Perzentil-Konfidenzintervalls* (engl. percentile confidence interval). :numref:`fig_mdl-bootstrap` veranschaulicht den Ablauf. Das einfache Verfahren beruht auf zwei wesentlichen Annahmen: Die Beobachtungen sind unabhängig und identisch verteilt, und die Statistik ändert sich glatt bei kleinen Änderungen der Daten.

<!--en:5f64c7ae-->
![Das gewöhnliche Bootstrap-Verfahren. Aus einer beobachteten Stichprobe (oben links) entstehen durch Ziehen mit Zurücklegen viele gleich große Bootstrap-Stichproben (unten links, Strichhöhen geben die Häufigkeit jeder Beobachtung an). Für jede wird die Statistik, hier der Median, neu berechnet. Das Histogramm der Replikate (rechts) nähert die Stichprobenverteilung um $\hat\theta$ (gestrichelte Linie) an. Seine Standardabweichung schätzt den Standardfehler, seine zentralen Quantile begrenzen ein Perzentilintervall. Das Histogramm wirkt unregelmäßig, weil für den Median nur wenige beobachtete Werte in Betracht kommen. Beim Maximum führt dieser Effekt zum vollständigen Versagen des Verfahrens.](../img/mdl-prob-bootstrap.svg)

<!--en:22c5f655-->
### Bootstrap-Standardfehler

<!--en:872c5b08-->
Bevor wir dem Bootstrap-Verfahren bei einer schwierigen Statistik vertrauen, prüfen wir es anhand einer bekannten Formel. Unser erster Testdatensatz enthielt 912 richtige und 88 falsche Antworten, mit theoretischem Standardfehler $0.0090$. Ziehen wir aus diesen 1000 binären Indikatoren 5000 Bootstrap-Stichproben und prüfen wir, ob wir denselben Wert erhalten.

<!--en:77f544fe-->
Die beiden Schätzwerte stimmen bis auf wenige Prozent überein. Wenden wir uns nun einer Statistik zu, für deren Standardfehler es keine einfach anwendbare geschlossene Formel gibt: dem Median von 200 Ziehungen aus der Exponentialverteilung mit Mittelwert 1. Diese Verteilung ist rechtsschief, ihr wahrer Median $\log2\approx0.69$ liegt unter ihrem Mittelwert. Für den Standardfehler des Medians gibt es zwar die asymptotische Formel $1/(2f(m)\sqrt n)$, wobei $f(m)$ die Dichte am Median $m$ ist (Übung :exref:`ex_mdl-statistics-median-se`), aber in der Praxis kennen wir $f(m)$ nicht. In unserem künstlichen Experiment *kennen* wir sie: $f(m)=e^{-\log2}=\frac12$. Damit erhalten wir den theoretischen Vergleichswert $1/\sqrt{200}\approx0.071$ für das Bootstrap-Verfahren.

<!--en:9d070250-->
Auch ohne die Dichte zu kennen, liefert das Bootstrap-Verfahren meist einen Wert nahe $0.07$. Über verschiedene Datensätze schwankt der Stichprobenmedian um $\log2$ (meist um höchstens $0.15$). Der Bootstrap-Standardfehler selbst schwankt etwa zwischen $0.05$ und $0.11$. Woher kommen diese großen Unterschiede? Der Median einer Bootstrap-Stichprobe hängt von den wenigen ursprünglichen Datenpunkten nahe der Stichprobenmitte ab. Deshalb ist die Bootstrap-Verteilung unregelmäßig und empfindlich gegenüber den zufälligen Abständen dieser Punkte. Das Perzentilintervall ist folglich etwas grob und schließt $\log2$ gelegentlich nur knapp ein. Da die zugrunde liegende Dichte am Median strikt positiv ist, ist der Median dennoch glatt genug, damit das Bootstrap-Verfahren gültig bleibt.

<!--en:76da25c7-->
### Grenzen des Bootstrap-Verfahrens

<!--en:f592ff93-->
Ebenso wichtig ist es, die Grenzen des Bootstrap-Verfahrens zu kennen. Seine beiden zentralen Annahmen, Glattheit und Unabhängigkeit, können in der Praxis verletzt sein.

<!--en:7cd7f950-->
Erstens versagt das Bootstrap-Verfahren, wenn eine Statistik zu empfindlich von extremen Datenpunkten abhängt. Das klassische Gegenbeispiel ist das Stichprobenmaximum. Seien $x_1,\ldots,x_n$ unabhängig und identisch gleichverteilt auf $[0,\theta]$ mit dem Schätzer $\hat\theta=\max_i x_i$. Jede Bootstrap-Stichprobe hat genau dasselbe Maximum $\hat\theta$, außer wenn alle $n$ Ziehungen die einzige größte Beobachtung verfehlen. Damit gilt

<!--en:8a81c505-->
Die Bootstrap-Verteilung konzentriert also 63% ihrer Wahrscheinlichkeitsmasse in einem einzigen Punkt, obwohl die wahre Stichprobenverteilung von $\hat\theta$ stetig ist. Schlimmer noch: Keine Bootstrap-Stichprobe kann ein Maximum über $\hat\theta$ liefern. Dieser Wert liegt strikt unter $\theta$. Die Wahrscheinlichkeit, dass das Perzentilintervall $\theta$ enthält, ist also null.

<!--en:c0815b8f-->
Zweitens versagt das Bootstrap-Verfahren, wenn wir abhängige Einheiten einzeln neu ziehen. Das ist ein häufiger Fallstrick bei modernen Benchmarks für Sprach- und Bildverarbeitung. Betrachten wir einen Benchmark zum Leseverständnis mit $G$ Textpassagen und $m$ Fragen je Passage. Missversteht ein Modell eine Passage, beantwortet es häufig mehrere zugehörige Fragen falsch. Angenommen, die Richtigkeitsindikatoren für zwei Fragen zur selben Passage haben die Korrelation $\rho$, während verschiedene Passagen unabhängig sind. Dann hat die Gesamtgenauigkeit die Varianz

<!--en:f54447ef-->
Sie ist um den Faktor $1+(m-1)\rho$ größer als bei $Gm$ unabhängigen Fragen. Ein naives Bootstrap-Verfahren, das einzelne Fragen zieht, ignoriert die Korrelation innerhalb jeder Passage und unterschätzt den Standardfehler um den Faktor $\sqrt{1+(m-1)\rho}$. Schon bei $m=20$ und einer mäßigen Korrelation von $\rho=0.1$ ist der Fehlerbalken um den Faktor $\sqrt{2.9}\approx1.7$ zu klein :cite:`Miller.2024`. Die Lösung ist einfach: Wir ziehen ganze Passagen anstelle einzelner Fragen (Übung :exref:`ex_mdl-statistics-clustered`).

<!--en:dbb196a0-->
Am Ende dieses Abschnitts bleiben zwei wichtige Fragen offen. Erstens haben wir gelernt, einen gegebenen Schätzer zu beurteilen und mit Fehlerbalken zu versehen. Wie aber *konstruieren* wir einen guten Schätzer, und wie genau lässt sich ein Parameter überhaupt schätzen? :numref:`sec_mdl-maximum_likelihood` beantwortet beides anhand der Likelihood-Funktion und ihrer Krümmung. Zweitens setzten alle Garantien in diesem Abschnitt einen einzigen Schätzer oder Hypothesentest voraus, den wir *vor* der Betrachtung der Daten festlegen. Wählen wir ein Modell aufgrund des höchsten Scores auf einem Validierungsdatensatz, ist diese Voraussetzung verletzt: Die Wahl des Gewinners aus vielen Kandidaten ist eine Form des multiplen Testens. In :numref:`sec_mdl-generalization-bounds` fragen wir, wie groß die Stichprobe sein muss, damit unsere Garantien für eine ganze Modellfamilie gleichzeitig gelten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:e854554b-->
Jede aus zufälligen Daten berechnete Zahl ist selbst eine Zufallsvariable. Wie weit wir ihr vertrauen können, bestimmt ihre Stichprobenverteilung über hypothetische andere Datensätze. Der mittlere quadratische Fehler zerfällt in das Quadrat der Verzerrung und die Varianz. Deshalb kann ein verzerrter Schätzer wie der geschrumpfte Mittelwert besser abschneiden als der erwartungstreue Stichprobenmittelwert. Fassen wir ein Integral als Erwartungswert auf, liefert der Stichprobenmittelwert eine Monte-Carlo-Schätzung. Ihre Fehlerrate $\sigma/\sqrt n$ verschlechtert sich anders als bei deterministischen Gittern mit wachsender Dimension nicht. Ein Hypothesentest verwirft die Nullhypothese, wenn ein beobachteter Unterschied durch Zufall allein selten wäre. Da ein $p$-Wert unter der Nullhypothese gleichverteilt ist, begrenzt die Ablehnung bei $p\le\alpha$ die Falsch-Positiv-Rate auf $\alpha$. Die Teststärke wächst mit dem Produkt aus standardisiertem Effekt und $\sqrt n$. Sie steigt erheblich, wenn wir konkurrierende Modelle auf denselben Fragen gepaart vergleichen. Ein Konfidenzintervall umfasst alle Parameterwerte, die ein Hypothesentest nicht verwirft. Fehlt eine analytische Formel für den Standardfehler, simuliert das Bootstrap-Verfahren die Stichprobenverteilung durch erneutes Ziehen aus den Daten. Bei glatten Statistiken unabhängiger Beobachtungen funktioniert das gut, bei Extremwerten wie dem Maximum und bei Testdatensätzen mit Gruppenstruktur versagt das gewöhnliche Verfahren.

<!--en:2294243a-->
## Übungen

<!--en:bdbee143-->
1. [code] **Schrumpfung des Stichprobenmittelwerts.** Wie im Text gezeigt, hat der geschrumpfte Mittelwert $c\,\bar x$ für $x_1,\ldots,x_n$ unabhängig und identisch gemäß $\mathcal N(\theta,\sigma^2)$ verteilte Beobachtungen den kleinsten MSE bei $c^\star=\theta^2/(\theta^2+\sigma^2/n)$.
    1. Erweitern Sie die Simulation von Stichprobenmittelwert, geschrumpftem Mittelwert und Median aus dem Text auf ein Gitter von Faktoren $c\in[0,1]$ bei $\theta=1$, $\sigma=4$ und $n=30$. Tragen Sie das Quadrat der Verzerrung, die Varianz und den MSE gegen $c$ auf. Verwenden Sie dazu `d2l.plot` und vergleichen Sie den empirisch optimalen Faktor mit $c^\star$.
    1. Schrumpfen Sie in Richtung eines Punktes $a$ statt gegen $0$. Verwenden Sie dazu den Schätzer $a+c(\bar x-a)$. Berechnen Sie seinen MSE und den optimalen Faktor. Welches $a$ würden Sie wählen, und was bedeutet das für die Wahl von $0$ im Text?
    1. $c^\star$ ist kein Schätzer, weil der Ausdruck $\theta$ enthält. Setzen Sie $\bar x$ für $\theta$ und $s^2$ für $\sigma^2$ ein. Sie erhalten $\hat c=\bar x^2/(\bar x^2+s^2/n)$. Bestimmen Sie den MSE von $\hat c\,\bar x$ durch Simulation bei $\theta\in\{0,1,3,10\}$. Wo schlägt dieser Ansatz $\bar x$, und wo schneidet er schlechter ab? Übung :exref:`ex_mdl-statistics-stein` zeigt, dass eine ähnliche Ersetzung in drei oder mehr Dimensionen für jedes $\theta$ überlegen ist.

   *Hinweis zu (c):* Bei $\theta=0$ unterliegt der Plug-in-Faktor selbst zufälligen Schwankungen. Betrachten Sie seine Verteilung.
   :exlabel:`ex_mdl-statistics-shrinking-mean`

<!--en:ecfe9faf-->
1. [code] **Multiple Vergleiche.** Bei einer Hyperparametersuche vergleichen wir viele Konfigurationen mit einer Baseline und führen damit viele Tests gleichzeitig aus.
    1. Zwanzig Modelle, von denen keines besser als die Baseline ist, werden jeweils in einem unabhängigen Test zum Niveau $\alpha=0.05$ mit ihr verglichen. Berechnen Sie die Wahrscheinlichkeit, dass mindestens ein Vergleich signifikant ausfällt. Berechnen Sie dieselbe Wahrscheinlichkeit für die Bonferroni-Schwelle $\alpha/20$. Zeigen Sie, dass die Bonferroni-Korrektur die Wahrscheinlichkeit mindestens eines falsch positiven Ergebnisses selbst bei abhängigen Tests auf höchstens $\alpha$ begrenzt.
    1. Das Benjamini-Hochberg-Verfahren :cite:`Benjamini.Hochberg.1995` sortiert die $m$ $p$-Werte, $p_{(1)}\le\cdots\le p_{(m)}$, bestimmt das größte $k$ mit $p_{(k)}\le k\alpha/m$ und verwirft die $k$ Hypothesen mit den kleinsten $p$-Werten. Simulieren Sie eine Suche über $m=200$ Konfigurationen, von denen $m_0=180$ keinen Effekt haben. Ziehen Sie die $z$-Statistiken für diese Konfigurationen aus $\mathcal N(0,1)$ und aus $\mathcal N(3,1)$ für die übrigen $20$. Berechnen Sie die zweiseitigen $p$-Werte `2 * norm.sf(np.abs(z))`. Geben Sie über $1{,}000$ Wiederholungen die mittlere Anzahl wahrer und falscher Entdeckungen und den mittleren Anteil falscher Entdeckungen an (Anteil fälschlich verworfener Hypothesen unter den verworfenen Hypothesen, bei keiner Ablehnung $0$). Vergleichen Sie die unkorrigierte Schwelle $0.05$, die Bonferroni-Korrektur und das Benjamini-Hochberg-Verfahren. Bei unabhängigen Tests garantiert letzteres einen erwarteten Anteil falscher Entdeckungen von höchstens $\alpha m_0/m$. Stimmt die Simulation damit überein?
   :exlabel:`ex_mdl-statistics-multiple-comparisons`

<!--en:152aca4a-->
1. [code] **Überdeckung bei kleinem $n$.** Das Intervall :eqref:`eq_mdl-gauss_confidence` verwendet das Gauß-Quantil $1.96$ statt des $t$-Quantils und setzt annähernd gaußverteilte Stichprobenmittelwerte voraus.
    1. Führen Sie die Simulation zur Überdeckung aus dem Text (1000 Intervalle aus standardnormalverteilten Daten) mit $n\in\{2,5,10,30,100\}$ und $100{,}000$ Versuchen aus. Halten Sie die Überdeckung des Intervalls fest. Ersetzen Sie dann $1.96$ durch das $97.5\%$-Quantil der $t$-Verteilung mit $n-1$ Freiheitsgraden. Es beträgt $12.71$, $2.776$, $2.262$, $2.045$ beziehungsweise $1.984$ für die betrachteten Werte von $n$. Erfassen Sie die Überdeckung erneut.
    1. Setzen Sie $n=2$ und ersetzen Sie $1.96$ durch $1.0$, das $75\%$-Quantil der $t$-Verteilung mit einem Freiheitsgrad. Damit hat das Intervall eine Überdeckung von $50\%$. Berechnen Sie die Überdeckung getrennt für Intervalle unterhalb und oberhalb der medianen Breite. Widerspricht der Unterschied der Aussage in :eqref:`eq_mdl-confidence`? Würden Sie ein schmales Intervall als Hinweis auf eine präzise Schätzung werten?
    1. Ziehen Sie Daten aus der Exponentialverteilung mit Mittelwert $1$ bei $n=10$ und behalten Sie das $t$-Quantil $2.262$ bei. Messen Sie die Überdeckung und zählen Sie getrennt die Intervalle, die vollständig unterhalb beziehungsweise oberhalb des wahren Mittelwerts liegen. Erklären Sie die Asymmetrie anhand der Verteilungsform.
   :exlabel:`ex_mdl-statistics-coverage-small-n`

<!--en:9c3b232b-->
1. [code] **Delta-Methode nahe einer Grenze.** Nach :eqref:`eq_mdl-delta-method` hat eine auf Testbeispielen gemessene Genauigkeit von $\hat\theta=0.90$ bei einem Stichprobenumfang von $n=1{,}000$ die Log-Odds $\log\frac{\hat\theta}{1-\hat\theta}$ mit einem Standardfehler von etwa $0.105$.
    1. Simulieren Sie $100{,}000$ Testdatensätze, indem Sie die Anzahl richtiger Vorhersagen aus der Binomialverteilung mit $n=1{,}000$ und $\theta=0.9$ ziehen. Vergleichen Sie die Standardabweichung der Log-Odds mit dem Wert der Delta-Methode.
    1. Wiederholen Sie dies mit $\theta=0.999$. Schätzen Sie den Anteil der Testdatensätze mit $\hat\theta=1$, vergleichen Sie ihn mit seinem exakten Wert und geben Sie die Log-Odds eines solchen Testdatensatzes an.
    1. Welche Annahme der Entwicklung erster Ordnung ist bei $\theta=0.999$ verletzt? Berechnen Sie die erwartete Anzahl falscher Antworten, $n(1-\theta)$, für beide Werte von $\theta$. Erklären Sie, warum diese Anzahl aussagekräftiger ist als $n$ allein, wenn Sie die Zuverlässigkeit der Delta-Methode und des Wald-Intervalls beurteilen.
    1. Ein Benchmark umfasst 30 Fragen, von denen ein Modell 29 richtig beantwortet. Berechnen Sie das Wald-Intervall $\hat\theta\pm1.96\sqrt{\hat\theta(1-\hat\theta)/30}$ und geben Sie an, wo es endet. Kehren Sie dann einen Test um: Berechnen Sie für jedes $\theta_0$ auf einem feinen Gitter in $[0.5,1]$ den exakten zweiseitigen $p$-Wert von $H_0\!:\theta=\theta_0$, also $\min\{1,\,2\min\{P_{\theta_0}(K\le29),\,P_{\theta_0}(K\ge29)\}\}$ mit $K\sim\mathrm{Binomial}(30,\theta_0)$. Sammeln Sie alle Werte $\theta_0$, die der Test bei $\alpha=0.05$ nicht verwirft. Vergleichen Sie beide Intervalle und ihre Überdeckung bei $\theta=0.95$ über $100{,}000$ simulierte Benchmarks.
   :exlabel:`ex_mdl-statistics-delta-boundary`

<!--en:9447cb74-->
1. [code] **Gepaarte Vergleiche.** ● Bei Auswertung auf denselben Testbeispielen oder mit denselben Zufallsstartwerten ist ein Teil des Rauschens beiden Modellen gemeinsam. Ein gepaarter Test eliminiert diesen Anteil.
    1. Im Text unterscheiden sich die Modelle bei 41 Fragen, mit einer Aufteilung von 25 zu 16. Unter $H_0$ folgt die Anzahl der Fragen, die nur B richtig beantwortet, der Verteilung $\mathrm{Binomial}(41,\tfrac12)$. Berechnen Sie den exakten zweiseitigen $p$-Wert. Vergleichen Sie ihn mit dem Monte-Carlo-Wert aus der Codezelle mit Vorzeichenwechsel in :numref:`subsec_mdl-paired-comparisons`. Halten Sie nun den Nettovorsprung bei 9 und variieren Sie die Anzahl unterschiedlicher Ergebnisse: Berechnen Sie $p$ für 9 zu 0, 17 zu 8, 25 zu 16 und 55 zu 46. Was entscheidet über das Urteil, der Vorsprung oder die Anzahl unterschiedlicher Ergebnisse?
    1. Modellieren Sie die Genauigkeiten je Zufallsstartwert mit einem beiden Modellen gemeinsamen Effekt: Für den Startwert $s$ betragen die Genauigkeiten $0.850+u_s+e_{A,s}$ und $0.858+u_s+e_{B,s}$, wobei $u_s\sim\mathcal N(0,0.01^2)$ und $e_{A,s},e_{B,s}\sim\mathcal N(0,0.003^2)$ unabhängig sind. Implementieren Sie den Test mit Vorzeichenwechsel für die Differenzen je Startwert als exakten Test. Zählen Sie für $k$ Startwerte alle $2^k$ Vorzeichenmuster auf, statt zufällige Muster zu ziehen. Vergleichen Sie seine Teststärke über $1{,}000$ Experimente mit $10$ Startwerten mit der des ungepaarten Permutationstests `perm_p`.
    1. Wiederholen Sie den Versuch mit $5$ Startwerten. Welchen kleinsten zweiseitigen $p$-Wert kann der exakte Test mit Vorzeichenwechsel liefern, und was bedeutet das bei $\alpha=0.05$? Eine Monte-Carlo-Variante, die $999$ zufällige Vorzeichenmuster zieht, wie `perm_p` zufällige Permutationen zieht, verwirft die Nullhypothese in einigen Prozent dieser Experimente. Warum?
   :exlabel:`ex_mdl-statistics-paired-comparisons`

<!--en:96b6c1d2-->
1. **Standardfehler des Medians.** ● Seien $x_1,\ldots,x_n$ unabhängig und identisch verteilt mit Verteilungsfunktion $F$, Median $m$ und Dichte $f(m)>0$ am Median. Der untere Stichprobenmedian $\hat m$ beträgt genau dann höchstens $m+t$, wenn mindestens die Hälfte der Beobachtungen höchstens $m+t$ beträgt.
    1. Die empirische Verteilungsfunktion $\hat F_n(m+t)$ hat den Erwartungswert $F(m+t)$ und die Varianz $F(m+t)\bigl(1-F(m+t)\bigr)/n$. Zeigen Sie mit dem zentralen Grenzwertsatz, dass $P(\hat m\le m+t)\approx\Phi\bigl(2\sqrt n\,f(m)\,t\bigr)$ für kleine $t$ gilt. Folgern Sie daraus $\textrm{se}(\hat m)\approx 1/\bigl(2f(m)\sqrt n\bigr)$. Setzen Sie den Faktor $1/f(m)$ zur Steigung $g'$ in :eqref:`eq_mdl-delta-method` in Beziehung.
    1. Werten Sie die Näherung für die Exponentialverteilung mit Mittelwert $1$ bei $n=200$ aus. Vergleichen Sie sie mit dem Bootstrap-Standardfehler, den Ihr Lauf des Bootstrap-Verfahrens für den Median aus dem Text ausgibt. Führen Sie die Zelle mehrmals aus. Die Ausgabe schwankt etwa zwischen $0.05$ und $0.11$. Warum schwankt der mit dem Bootstrap-Verfahren geschätzte Standardfehler des Medians so stark von Datensatz zu Datensatz, während die entsprechende Schätzung für den Mittelwert viel stabiler ist?
    1. Berechnen Sie den Quotienten $\textrm{Var}(\hat m)/\textrm{Var}(\bar x)$ für große $n$ bei gaußverteilten und bei Laplace-verteilten Daten. Welchen Lageschätzer würden Sie bei Daten mit schweren Verteilungsrändern wählen?

   *Hinweis zu (b):* Der Median einer Bootstrap-Stichprobe hängt von wenigen beobachteten Werten nahe $\hat m$ ab. Ihre Abstände sind zufällig.
   :exlabel:`ex_mdl-statistics-median-se`

<!--en:519fcf6b-->
1. [code] **Gitter und Monte-Carlo-Verfahren im Vergleich.** ● :numref:`subsec_mdl-monte-carlo` vergleicht Exponenten: Bei $N$ Auswertungen in $d$ Dimensionen hat ein Gitter zweiter Ordnung einen Fehler der Ordnung $N^{-2/d}$ und das Monte-Carlo-Verfahren einen Fehler der Ordnung $N^{-1/2}$. Prüfen Sie diesen Vergleich für $f(\mathbf x)=\exp(-\|\mathbf x\|^2)$ auf $[0,1]^d$. Nach der Produktregel von Fubini ist das Integral gleich $\bigl(\tfrac{\sqrt\pi}{2}\operatorname{erf}(1)\bigr)^d$ (:numref:`sec_mdl-integral_calculus`).
    1. Implementieren Sie die Mittelpunktsregel auf dem vollständigen Gitter aus $m^d$ Würfeln und den Monte-Carlo-Mittelwert :eqref:`eq_mdl-monte-carlo` aus $N=m^d$ gleichverteilten Ziehungen. Tragen Sie für $d\in\{1,2,4,5,6\}$ und $N$ bis etwa $10^6$ den relativen Fehler gegen $N$ in doppelt logarithmischer Darstellung auf. Verwenden Sie für das Monte-Carlo-Verfahren die Wurzel des mittleren quadratischen Fehlers über $20$ Wiederholungen. Stimmen die Steigungen mit den vorhergesagten Exponenten überein?
    1. Bei $d=4$ sind die Exponenten gleich. Welches Verfahren ist dort genauer? Was bestimmt den Vorfaktor jeder Rate, und wie ändert er sich jeweils mit $d$?
    1. Beide Fehler lassen sich vorhersagen. Die Mittelpunktsumme von $f$ ist ein Produkt eindimensionaler Mittelpunktsummen. Beim Monte-Carlo-Verfahren beträgt die Wurzel des mittleren quadratischen Fehlers $\mathrm{sd}\bigl(f(\mathbf U)\bigr)/\sqrt N$ nach :eqref:`eq_mdl-mc-variance` für $\mathbf U$ gleichverteilt auf $[0,1]^d$. Die Produktregel zerlegt $\mathbb{E}\bigl[f(\mathbf U)^2\bigr]$ in eindimensionale Integrale. Berechnen Sie beide vorhergesagten relativen Fehler für $d\in\{4,5,6\}$ bei $N\approx 10^5$ und vergleichen Sie sie mit Ihren Messungen. Bei welchem $d$ sind beide Verfahren gleichauf? Wie zuverlässig entscheiden $20$ Wiederholungen den Vergleich bei diesem $d$?
   :exlabel:`ex_mdl-integral-calculus-grids-monte-carlo`

<!--en:ab324b88-->
1. [code] **Testdatensätze mit Gruppenstruktur.** ● Auswertungsdatensätze bestehen oft aus Gruppen zusammengehöriger Beispiele, etwa mehreren Fragen zu einer Textpassage. Das Bootstrap-Verfahren für unabhängige, identisch verteilte Beobachtungen unterschätzt dann den Standardfehler :cite:`Miller.2024`. Ein Testdatensatz umfasst $G$ Gruppen mit je $m$ Beispielen. Die erwartete Genauigkeit beträgt $\theta$. Die Richtigkeitsindikatoren zweier Beispiele derselben Gruppe haben die Korrelation $\rho$, und Beispiele aus verschiedenen Gruppen sind unabhängig.
    1. Zeigen Sie, dass der Mittelwert von $m$ Variablen mit gemeinsamer Varianz $v$ und paarweiser Korrelation $\rho$ die Varianz $v\bigl(1+(m-1)\rho\bigr)/m$ hat. Folgern Sie daraus, dass die Genauigkeit $\hat\theta$ auf dem Testdatensatz folgende Varianz hat:
       $$\textrm{Var}(\hat\theta)=\frac{\theta(1-\theta)}{Gm}\bigl(1+(m-1)\rho\bigr).$$
       1. Simulieren Sie $G=100$ und $m=20$. Ziehen Sie für jede Gruppe eine Genauigkeit $\theta_g\sim\mathrm{Beta}(6.3,2.7)$ und anschließend Beispiele, die bedingt auf $\theta_g$ unabhängig sind. Diese Wahl ergibt $\theta=0.7$ und $\rho=1/(6.3+2.7+1)=0.1$. Vergleichen Sie über $300$ Testdatensätze die Standardabweichung von $\hat\theta$ mit zwei mittleren Bootstrap-Standardfehlern. Ziehen Sie für den ersten wie beim Bootstrap-Verfahren für die Genauigkeit im Text einzelne Beispiele unter der Unabhängigkeitsannahme, für den zweiten ganze Gruppen wie beim Cluster-Bootstrap. Verwenden Sie jeweils $R=1{,}000$ Bootstrap-Stichproben. Um welchen Faktor unterschätzt das Verfahren mit Unabhängigkeitsannahme den Standardfehler? Vergleichen Sie den Faktor mit der Vorhersage aus (a).
   :exlabel:`ex_mdl-statistics-clustered`

<!--en:abd24a90-->
1. [code] **Stein-Paradoxon.** ●● Beobachten Sie $\mathbf x\sim\mathcal
   N(\boldsymbol\theta,\mathbf I_d)$ und beurteilen Sie einen Schätzer anhand seines Risikos $\mathbb{E}\|\hat{\boldsymbol\theta}-\boldsymbol\theta\|^2$, also des MSE mit quadrierten Normen anstelle der Quadrate. Der Schätzer $\mathbf x$ hat das Risiko $d$ für jedes $\boldsymbol\theta$. :citet:`James.Stein.1961` zeigten, dass für $d\ge3$ der Schätzer
   $$\hat{\boldsymbol\theta}_{\textrm{JS}}=\left(1-\frac{d-2}{\|\mathbf x\|^2}\right)\mathbf x$$
   für jedes $\boldsymbol\theta$ ein kleineres Risiko hat.
    1. Beweisen Sie das Lemma von Stein :cite:`Stein.1981`: Gilt $z\sim\mathcal N(\theta,1)$ und ist $h$ differenzierbar mit $\mathbb{E}|h'(z)|<\infty$, so folgt $\mathbb{E}[(z-\theta)h(z)]=\mathbb{E}[h'(z)]$. Integrieren Sie partiell.
    1. Betrachten Sie einen Schätzer $\mathbf x+\mathbf g(\mathbf x)$ mit schwach differenzierbarem $\mathbf g$ (hier: differenzierbar außerhalb des Ursprungs, mit $\mathbb{E}\|\mathbf g(\mathbf x)\|^2<\infty$). Zeigen Sie, dass sein Risiko gleich
       $$d+\mathbb{E}\|\mathbf g(\mathbf x)\|^2+2\sum_{i=1}^d\mathbb{E}\left[\frac{\partial g_i}{\partial x_i}(\mathbf x)\right].$$
       ist. Zeigen Sie mit dieser Identität, dass $\hat{\boldsymbol\theta}_{\textrm{JS}}$ das Risiko $d-(d-2)^2\,\mathbb{E}\bigl[1/\|\mathbf x\|^2\bigr]$ hat. An welcher Stelle benötigt das Argument $d\ge3$?
    1. Simulieren Sie die Risiken von $\mathbf x$, von $\hat{\boldsymbol\theta}_{\textrm{JS}}$ und des auf den positiven Teil beschränkten Schätzers $\max\bigl(0,1-(d-2)/\|\mathbf x\|^2\bigr)\mathbf x$ für $d=10$ und $\|\boldsymbol\theta\|\in\{0,1,2,5,10,20\}$. Tragen Sie die Risiken gegen $\|\boldsymbol\theta\|$ auf.
    1. Geben Sie den optimalen Faktor $c^\star$ für $c\,\mathbf x$ in $d$ Dimensionen an, das Gegenstück zu $c^\star$ im Text. Welche Funktion der Daten ersetzt die unbekannte Größe in $\hat{\boldsymbol\theta}_{\textrm{JS}}$, und warum wird dieser Ersatz mit wachsendem $d$ genau? Vergleichen Sie mit Übung :exref:`ex_mdl-statistics-shrinking-mean`c.
   :exlabel:`ex_mdl-statistics-stein`

<!--en:6818aac5-->
1. [extended] **Zwischenauswertungen bei einem laufenden Test.** ●● Häufig beobachtet man einen A/B-Test oder eine Auswertung während des Trainings schon beim Eintreffen der Daten und bricht ab, sobald das Ergebnis signifikant erscheint :cite:`Johari.Koomen.Pekelis.ea.2017`. Seien $x_1,x_2,\ldots$ unabhängig und identisch gemäß $\mathcal N(\mu,1)$ verteilt, sei $H_0\!:\mu=0$ und sei $S_n=\sum_{i\le n}x_i$. Der $z$-Test zum Zeitpunkt $n$ verwirft die Nullhypothese, wenn $|S_n|/\sqrt n\ge1.96$ gilt.
    1. Simulieren Sie $4{,}000$ Läufe der Länge $N=10{,}000$ unter $H_0$. Schätzen Sie für $K$ gleichmäßig verteilte Auswertungszeitpunkte die Wahrscheinlichkeit, dass mindestens eine Auswertung die Nullhypothese verwirft. Verwenden Sie $K\in\{1,2,5,10,100,1000,10000\}$. Nach dem Gesetz des iterierten Logarithmus geht diese Wahrscheinlichkeit bei unbegrenzt fortgesetzter Beobachtung gegen eins. Wie schnell wächst sie in Ihrer Tabelle?
    1. Legen Sie die A-priori-Verteilung $\mathcal N(0,\tau^2)$ für $\mu$ fest und definieren Sie den Likelihood-Quotienten der Mischung
       $$\Lambda_n=\int\prod_{i=1}^n\frac{p(x_i\mid\mu)}{p(x_i\mid0)}\,\mathcal N(\mu;0,\tau^2)\,d\mu.$$
       Zeigen Sie, dass $\Lambda_n=(1+n\tau^2)^{-1/2}\exp\bigl(\tau^2S_n^2/(2(1+n\tau^2))\bigr)$ gilt und dass der Quotient unter $H_0$ ein Martingal ist: $\mathbb{E}[\Lambda_{n+1}\mid x_1,\ldots,x_n]=\Lambda_n$.
    1. Die Ville-Ungleichung, die wir hier ohne Beweis verwenden, besagt, dass für ein nichtnegatives Martingal mit $\Lambda_0=1$ die Ungleichung $P(\sup_n\Lambda_n\ge1/\alpha)\le\alpha$ gilt. Folgern Sie, dass die Ablehnung beim ersten $n$ mit $\Lambda_n\ge1/\alpha$ eine Fehlerwahrscheinlichkeit erster Art von höchstens $\alpha$ hat, unabhängig davon, wie oft die Daten betrachtet werden. Leiten Sie die Ablehnungsgrenze für $|S_n|$ her und vergleichen Sie ihr Wachstum in $n$ mit dem der vorab festgelegten Grenze $1.96\sqrt n$.
    1. Prüfen Sie den Fehler erster Art des sequenziellen Tests durch Simulation für $\tau^2\in\{0.01,0.1,1\}$. Setzen Sie dann $\mu=0.2$. Verwenden Sie als Referenz den $z$-Test mit festem Stichprobenumfang, dessen $n$ sich aus :eqref:`eq_mdl-power-sample-size` für die Teststärke $0.8$ ergibt. Geben Sie für jeden der drei Werte von $\tau^2$ die Teststärke des sequenziellen Tests bis zu diesem $n$, den Median seiner Stoppzeit und seine Teststärke bis $N$ an. Wie wirkt sich die Wahl von $\tau^2$ auf diese Zahlen aus? Was gibt ein Test auf, der zu jedem Auswertungszeitpunkt gültig ist, und was gewinnt er?
   :exlabel:`ex_mdl-statistics-peeking`

<!--en:512137e9-->
[Diskussion](https://d2l.discourse.group/t/419)

<!--en:b6902b17-->
[Diskussion](https://d2l.discourse.group/t/1102)

<!--en:faa63821-->
[Diskussion](https://d2l.discourse.group/t/1103)

<!--en:faa63821-2-->
[Diskussion](https://d2l.discourse.group/t/1103)

<!--en:3c3e83e3-->
Punktschätzung, Hypothesentests, Konfidenzintervalle und Bootstrap-Verfahren<br>**Stichprobenverteilungen · Dreieck aus Verzerrung und Standardfehler · Monte Carlo · Permutations- und $z$-Tests · Teststärke · Konfidenzintervalle · Bootstrap-Verfahren**

<!--en:4e407765-->
::: {.slide title="Sind 921/1000 wirklich besser als 912/1000?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:03707427-->
- **Testdatensatz mit 1000 Fragen:** Klassifikator A erreicht $912/1000$ ($91.2\%$), Klassifikator B erreicht $921/1000$ ($92.1\%$). Der Standardfehler je Modell beträgt $\mathrm{se} \approx 0.90\%$.

<!--en:bf7bada8-->
- **Ungepaarter und gepaarter Vergleich:** Bei zwei unabhängigen Testdatensätzen geht der Unterschied von $+0.9\%$ im Standardfehler $\mathrm{se} = 1.24\%$ unter. Die Paarung auf *denselben* Fragen eliminiert deren gemeinsame Schwierigkeit. Bei Genauigkeiten von $90\%$ und $91\%$ und unterschiedlichen Ergebnissen auf $4\%$ der Fragen sinkt die Zahl benötigter Labels etwa um den Faktor $9\times$.

<!--en:0c5840a3-->
::: {.slide title="Schätzer und ihre Stichprobenverteilung"}

<!--en:a14ea64e-->
[Schwankungen über hypothetische Datensätze]{.kicker}

<!--en:2d463b78-->
Ein **Schätzer** $\hat{\theta}_n = T(x_1, \dots, x_n)$ ordnet jeder Zufallsstichprobe vom Umfang $n$ einen Schätzwert für einen Parameter zu. Seine Verteilung über wiederholte Ziehungen des gesamten Datensatzes ist seine **Stichprobenverteilung**:

<!--en:8ae3f390-->
- **Testgenauigkeit als Stichprobenanteil:** $x_i \in \{0, 1\}$ gibt an, ob die Antwort auf Frage $i$ richtig ist ($x_i \stackrel{\text{i.i.d.}}{\sim} \mathrm{Bernoulli}(\theta)$). Für die Genauigkeit $\hat{\theta}_n = \frac{1}{n}\sum_{i=1}^n x_i$ in der Stichprobe gelten die exakten Momente:
  $$\mathbb{E}[\hat{\theta}_n] = \theta, \qquad \mathrm{Var}(\hat{\theta}_n) = \frac{\theta(1 - \theta)}{n}, \qquad \mathrm{se}(\hat{\theta}_n) = \sqrt{\frac{\theta(1 - \theta)}{n}}$$

<!--en:97971079-->
- Simulation von $10{,}000$ unabhängigen Testdatensätzen vom Umfang $n = 1{,}000$ bei wahrer Genauigkeit $\theta = 0.912$:

<!--en:ade687ea-->
**Grundgedanke der frequentistischen Statistik:** Beobachtet wird nur *ein* Testdatensatz. Wie zuverlässig die Schätzregel ist, zeigt ihre Streuung über die $10{,}000$ hypothetischen Testdatensätze, die nie gezogen wurden.

<!--en:53c9e663-->
::: {.slide title="Verzerrung und Standardfehler: Verschiebung und Streuung"}

<!--en:2e47a789-->
[Zwei Arten von Schätzfehlern]{.kicker}

<!--en:85e72173-->
- **Verzerrung (systematische Abweichung des Zentrums von $\theta$):**
  $$\mathrm{Bias}(\hat{\theta}_n) = \mathbb{E}[\hat{\theta}_n] - \theta \quad (\text{unbiased if } \mathrm{Bias} = 0\ \forall\theta)$$

<!--en:62407025-->
- **Varianz und Standardfehler (Streuung um $\mathbb{E}[\hat{\theta}_n]$):**
  $$\mathrm{Var}(\hat{\theta}_n) = \mathbb{E}\Bigl[\bigl(\hat{\theta}_n - \mathbb{E}[\hat{\theta}_n]\bigr)^2\Bigr], \quad \mathrm{se}(\hat{\theta}_n) = \sqrt{\mathrm{Var}(\hat{\theta}_n)}$$

<!--en:871aa989-->
- **Benchmark ($n = 30$ Ziehungen aus $\mathcal{N}(1, 16)$):**
  - Stichprobenmittelwert $\bar{x}$: $\mathrm{Bias} = 0$, $\mathrm{se} = 4/\sqrt{30} \approx 0.73$.
  - Geschrumpfter Mittelwert $0.65\,\bar{x}$: $\mathrm{Bias} = -0.35$, $\mathrm{se} = 0.65(0.73) \approx 0.47$.

<!--en:a8cde197-->
::: {.slide title="Warum die Division durch n die Varianz unterschätzt"}

<!--en:18a0dfca-->
[Bessel-Korrektur mit n − 1 und Freiheitsgrade]{.kicker}

<!--en:991fcaac-->
- **Warum $\bar{x}$ die Residuen verkleinert:** Der Stichprobenmittelwert $\bar{x}$ minimiert $\sum_i (x_i - c)^2$ für die vorliegende Stichprobe. Deshalb ist $\sum_i (x_i - \bar{x})^2$ kleiner als $\sum_i (x_i - \mu)^2$, und zwar genau um $n(\bar{x} - \mu)^2$.

<!--en:60bef330-->
- **Bessel-Korrektur (`ddof=1`):** Division durch die $n - 1$ Freiheitsgrade macht $s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$ erwartungstreu ($\mathbb{E}[s^2] = \sigma^2$).

<!--en:3c20f7f4-->
::: {.slide title="Zerlegung in Verzerrung und Varianz nach Pythagoras"}

<!--en:216f68fa-->
[Orthogonale Fehlerzerlegung]{.kicker}

<!--en:886b3925-->
- **Mittlerer quadratischer Fehler (MSE):** mittlerer quadrierter Abstand zum wahren Wert $\theta$:
  $$\mathrm{MSE}(\hat{\theta}_n) = \mathbb{E}\bigl[(\hat{\theta}_n - \theta)^2\bigr] = \mathrm{Bias}(\hat{\theta}_n)^2 + \mathrm{Var}(\hat{\theta}_n)$$

<!--en:41825d2d-->
- **Beweis und Geometrie:** $m = \mathbb{E}[\hat{\theta}_n]$ setzen und $\hat{\theta}_n - \theta = (\hat{\theta}_n - m) + (m - \theta)$ zerlegen. Beim inneren Produkt $\langle U, V \rangle = \mathbb{E}[UV]$ steht die zentrierte Schwankung $\hat{\theta}_n - m$ senkrecht auf der konstanten Verzerrung $m - \theta$:
  $$\mathbb{E}\bigl[(\hat{\theta}_n - m)(m - \theta)\bigr] = (m - \theta)\,\underbrace{\mathbb{E}[\hat{\theta}_n - m]}_{0} = 0$$

<!--en:3ee89434-->
- **Hypotenuse nach Pythagoras:** $\sqrt{\mathrm{MSE}} = \sqrt{\mathrm{Bias}^2 + \mathrm{se}^2}$ (für $0.65\bar{x}$ beträgt die Hypotenusenlänge $0.59 < 0.73$).

<!--en:bc59e313-->
::: {.slide title="Empirischer Vergleich: Der geschrumpfte Mittelwert schlägt den erwartungstreuen"}

<!--en:ae909843-->
[Erwartungstreue allein garantiert keine Genauigkeit]{.kicker}

<!--en:d8a3761e-->
Vergleich von Stichprobenmittelwert $\bar{x}$, geschrumpftem Mittelwert $0.65\,\bar{x}$ und Stichprobenmedian über $10{,}000$ Datensätze mit je $n = 30$ Ziehungen aus $\mathcal{N}(1, 16)$:

<!--en:49ec8272-->
**Relative Effizienz des Medians gegenüber dem Mittelwert:** Bei gaußverteilten Daten ist der Stichprobenmedian wegen der Symmetrie erwartungstreu. Seine asymptotische Varianz beträgt jedoch $\frac{\pi}{2}\frac{\sigma^2}{n} \approx 1.57\frac{\sigma^2}{n}$ ($0.80$ gegenüber $0.53$ bei $n=30$). Damit erreicht er eine Effizienz von $2/\pi \approx 64\%$.

<!--en:9faca0a6-->
::: {.slide title="Optimale Schrumpfung und Abwägung zwischen Verzerrung und Varianz"}

<!--en:73bce54c-->
[Regularisierung im Kleinen]{.kicker}

<!--en:267298d2-->
- Beim geschrumpften Schätzer $c\,\bar{x}$ ($0 < c < 1$) ist der MSE die Summe zweier Parabeln in $c$:
  $$\mathrm{MSE}(c) = (1 - c)^2\theta^2 + c^2\frac{\sigma^2}{n}$$

<!--en:8504d7fd-->
- Aus $\frac{d}{dc}\mathrm{MSE}(c) = 0$ ergibt sich die optimale Schrumpfung:
  $$c^\star = \frac{\theta^2}{\theta^2 + \sigma^2/n} < 1, \quad \mathrm{MSE}(c^\star) = \frac{\theta^2\,\sigma^2/n}{\theta^2 + \sigma^2/n} < \frac{\sigma^2}{n}$$

<!--en:0c23623d-->
- **Einschränkung:** $c^\star$ hängt vom unbekannten $\theta$ ab (bei $\theta=1$ liefert $c^\star \approx 0.65$ den MSE $\mathrm{MSE} = 0.35$, bei $\theta=10$ liefert $c=0.65$ den MSE $\mathrm{MSE} = 12.5 \gg 0.53$). In $d \ge 3$ Dimensionen schlägt die datengestützte **James-Stein-Schrumpfung** $\left(1 - \frac{d-2}{\|\mathbf{x}\|^2}\right)\mathbf{x}$ den Schätzer $\mathbf{x}$ für *jedes* $\boldsymbol{\theta}$.

<!--en:c315a078-->
::: {.slide title="Konsistenz und schwaches Gesetz der großen Zahlen"}

<!--en:67c2f1b9-->
[Asymptotische Konvergenz für n → ∞]{.kicker}

<!--en:b4cfcd7b-->
Ein Schätzer $\hat{\theta}_n$ ist **konsistent** ($\hat{\theta}_n \xrightarrow{P} \theta$), wenn $P(|\hat{\theta}_n - \theta| > \varepsilon) \to 0$ für $n \to \infty$ bei jeder Toleranz $\varepsilon > 0$ gilt. Die Markov-Ungleichung für $(\hat{\theta}_n - \theta)^2 > \varepsilon^2$ verbindet Konsistenz direkt mit dem MSE:

<!--en:0f2f0896-->
- **Hinreichende Bedingung für Konsistenz:** Gelten $\mathrm{Bias}(\hat{\theta}_n) \to 0$ und $\mathrm{Var}(\hat{\theta}_n) \to 0$ für $n \to \infty$, so folgt $\hat{\theta}_n \xrightarrow{P} \theta$.
  - *Konsistent trotz Verzerrung bei endlichem $n$:* Die Plug-in-Varianz $s_0^2 = \frac{n-1}{n}s^2$ hat die Verzerrung $-\sigma^2/n \to 0$ und die Varianz $O(1/n) \to 0$.
  - *Inkonsistent trotz Erwartungstreue:* Für die erste Beobachtung $\hat{\theta}_n = x_1$ gilt $\mathrm{Bias} = 0$, aber $\mathrm{Var} = \sigma^2 \not\to 0$.

<!--en:1b51fa4d-->
- **Proposition (schwaches Gesetz der großen Zahlen):** Sind $x_1, \dots, x_n$ unabhängig und identisch verteilt mit Erwartungswert $\mu$ und Varianz $\sigma^2 < \infty$, so hat $\bar{x}_n$ die Eigenschaften $\mathrm{Bias} = 0$ und $\mathrm{Var}(\bar{x}_n) = \sigma^2/n$. Also gilt
  $$P\bigl(|\bar{x}_n - \mu| > \varepsilon\bigr) \le \frac{\sigma^2}{n\,\varepsilon^2} \xrightarrow{n \to \infty} 0 \implies \bar{x}_n \xrightarrow{P} \mu$$

<!--en:172de98a-->
::: {.slide title="Monte-Carlo-Schätzung und deterministische Gitter"}

<!--en:af920fdf-->
[Überwindung des Fluchs der Dimensionalität]{.kicker}

<!--en:66ac4047-->
- **Fluch der Dimensionalität bei Gittern:** Ein Mittelpunktgitter zweiter Ordnung mit $N = \epsilon^{-d}$ Punkten hat den Fehler $O(\epsilon^2) = O(N^{-2/d})$. In hohen Dimensionen konvergiert es daher sehr langsam.

<!--en:098e702a-->
- **Dimensionsunabhängige Monte-Carlo-Schätzung:** Der Mittelwert aus $N$ unabhängigen, identisch verteilten Ziehungen konvergiert mit der Rate $\sigma N^{-1/2}$ in *jeder* Dimension $d$. Bei $d = 4$ erreicht er die Rate des Mittelpunktgitters, für alle $d > 4$ ist er schneller.

<!--en:1372cfdd-->
::: {.slide title="Monte-Carlo-Konvergenz: Einzellauf und Wurzel des mittleren quadratischen Fehlers"}

<!--en:45fe54f7-->
[Empirische Prüfung von σ / √n]{.kicker}

<!--en:4fdeeb8a-->
- Schätzung von $\int_0^1 e^{-x^2}\,dx = \mathbb{E}[e^{-U^2}] = \frac{\sqrt{\pi}}{2}\mathrm{erf}(1) \approx 0.746824$ mit $U \sim \mathrm{Unif}(0, 1)$ ($\sigma = \mathrm{sd}(e^{-U^2}) \approx 0.201$):
  - $n = 10^2$: $\sigma/\sqrt{n} \approx 2.0 \times 10^{-2}$
  - $n = 10^4$: $\sigma/\sqrt{n} \approx 2.0 \times 10^{-3}$

<!--en:b61b3946-->
- **Einzellauf und quadratisches Mittel über mehrere Läufe:** Ein einzelner laufender Mittelwert schwankt. Bei jedem Nulldurchgang seines Fehlers fällt die Fehlerkurve steil ab. Die Wurzel des mittleren quadratischen Fehlers über $200$ unabhängige Läufe folgt dagegen über vier Größenordnungen der theoretischen Linie $\sigma/\sqrt{n}$ mit Steigung $-1/2$.

<!--en:9955a31a-->
::: {.slide title="Hypothesentests: Nullverteilung und p-Werte"}

<!--en:333a1bde-->
[Modellvergleich über Zufallsstartwerte]{.kicker}

<!--en:94acf86c-->
Modell A und Modell B werden mit je $k = 20$ Zufallsstartwerten trainiert (Standardabweichung über die Startwerte $\sigma = 1.0\%$, wahre Verbesserung $\mu_B - \mu_A = +0.8\%$). Lässt sich aus den $40$ Zahlen erkennen, dass Modell B besser ist?

<!--en:d44556dc-->
- **Nullhypothese ($H_0$):** Beide Gruppen von Genauigkeiten stammen aus derselben Verteilung ($\mu_A = \mu_B$).

<!--en:e35bf283-->
- **Alternativhypothese ($H_A$):** Die Verteilungen unterscheiden sich ($\mu_A \ne \mu_B$).

<!--en:fc1bc442-->
- **Permutationstest (Fisher, 1935):** Unter $H_0$ sind die Gruppenbezeichnungen `"A"` und `"B"` bedeutungslos. Alle $40$ Scores zusammenfassen, in zwei Gruppen zu je $20$ aufteilen, dies für $R$ Permutationen wiederholen und den zweiseitigen **$p$-Wert** berechnen:
  $$p = \frac{1 + \#\bigl\{r : |\text{gap}_r| \ge |\text{gap}_{\text{obs}}|\bigr\}}{R + 1}$$

<!--en:99939037-->
::: {.slide title="Vektorisierter Permutationstest im Code"}

<!--en:cc438e6c-->
[Verteilungsfreier Modellvergleich]{.kicker}

<!--en:076ccef5-->
$R$ zufällige Neuzuordnungen der zusammengefassten Genauigkeiten vektorisieren und mit $k = 20$ Zufallsstartwerten je Modell testen:

<!--en:e0cb3965-->
**Warum $+1$ in Zähler und Nenner zählt:** Die beobachtete Zuordnung ist eine von $R + 1$ unter $H_0$ austauschbaren Anordnungen. Damit ist der Rang der beobachteten Differenz auf $\{1, \dots, R+1\}$ gleichverteilt. Dies garantiert $P_{H_0}(p \le \alpha) \le \alpha$.

<!--en:b55a8d4c-->
::: {.slide title="Parametrischer z-Test und Ablehnungsbereiche"}

<!--en:8abd4abf-->
[Hypothesentests in geschlossener Form]{.kicker}

<!--en:94085212-->
- Für $x_1, \dots, x_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$ mit bekanntem $\sigma$ verwendet der Test von $H_0 : \mu = 0$ die standardisierte Teststatistik:
  $$T = \frac{\bar{x}}{\sigma / \sqrt{n}} = \frac{\sqrt{n}\,\bar{x}}{\sigma} \;\underset{H_0}{\sim}\; \mathcal{N}(0, 1)$$

<!--en:fa9d8187-->
- **Zweiseitiger $p$-Wert und kritische Schwelle:**
  $$p = 2\bigl(1 - \Phi(|T|)\bigr) \le \alpha \iff |T| \ge z_{1 - \alpha/2}$$
  Bei $\alpha = 0.05$ gilt $z_{0.975} \approx 1.96$. Ablehnung von $H_0$, wenn $\bar{x}$ mehr als $1.96$ Standardfehler von $0$ entfernt liegt.

<!--en:4d3824c5-->
- **Zweistichprobenvergleich über Zufallsstartwerte:** Die Teststatistik lautet $T = \frac{\bar{x}_B - \bar{x}_A}{\sigma\sqrt{2/k}}$. Für die Differenz der Mittelwerte gilt $\mathrm{se} = 0.01\sqrt{2/20} \approx 0.00316$ ($+0.008$ entspricht $2.53\,\mathrm{se}$).

<!--en:11339572-->
::: {.slide title="Fehler erster und zweiter Art und Teststärke"}

<!--en:9f2cee16-->
[Die 2×2-Entscheidungsmatrix eines Experiments]{.kicker}

<!--en:31fea5d0-->
- **Fehler erster Art (falsch positiv, Rate $\alpha$):** Ablehnung von $H_0$, obwohl $H_0$ gilt (Behauptung einer Verbesserung, die es nicht gibt).

<!--en:4d11b122-->
- **Fehler zweiter Art (falsch negativ, Rate $\beta$):** Keine Ablehnung von $H_0$, obwohl $H_A$ gilt (eine echte Verbesserung bleibt unentdeckt).

<!--en:f45785af-->
- **Teststärke ($1 - \beta$):** Wahrscheinlichkeit, einen wahren Effekt der Größe $\delta$ nachzuweisen:
  $$1 - \beta = P(\text{reject } H_0 \mid H_A \text{ true})$$

<!--en:8b685596-->
- **Zwei verbreitete Fehlschlüsse beim $p$-Wert:**
  1. $p$ ist $P(\text{data this extreme} \mid H_0)$, **nicht** $P(H_0 \mid \text{data})$.
  2. $p > 0.05$ beweist **nicht**, dass $H_0$ gilt. Ein Test mit geringer Teststärke ($k = 5$ Zufallsstartwerte) übersieht häufig echte Verbesserungen.

<!--en:72953427-->
::: {.slide title="Warum p-Werte unter der Nullhypothese gleichverteilt sind"}

<!--en:6c412572-->
[Exakte Kalibrierung der Falsch-Positiv-Rate]{.kicker}

<!--en:077187ae-->
**Proposition:** Hat $T$ eine stetige Verteilungsfunktion $F$ unter $H_0$, so folgen der einseitige $p$-Wert $p = 1 - F(T)$ und der zweiseitige $p$-Wert $p = 2\min\{F(T), 1 - F(T)\}$ beide der Verteilung $\mathrm{Unif}(0, 1)$ unter $H_0$. Daher gilt $P_{H_0}(p \le \alpha) = \alpha$.

<!--en:aa54b46a-->
- **Beweis:** Nach der Umkehrung der Inversionsmethode erfüllt $U = F(T)$ die Gleichung $P(U \le u) = P(T \le F^{-1}(u)) = F(F^{-1}(u)) = u$. Also gilt $U \sim \mathrm{Unif}(0, 1)$. Somit folgt $P(2\min\{U, 1-U\} \le \alpha) = P(U \le \alpha/2) + P(U \ge 1 - \alpha/2) = \alpha/2 + \alpha/2 = \alpha$. $\blacksquare$

<!--en:7d49a341-->
::: {.slide title="Teststärke und Planung des Stichprobenumfangs"}

<!--en:36bf7723-->
[Warum 5 Zufallsstartwerte echte Verbesserungen übersehen und der Umfang wie 1/δ² wächst]{.kicker}

<!--en:c32c6ba3-->
- Unter $H_A$ mit standardisiertem Effekt $\delta = |\mu|/\sigma$ gilt $T \sim \mathcal{N}(\delta\sqrt{n}, 1)$. Zur Berechnung der Teststärke genügt näherungsweise der Ablehnungsbereich $T \ge z_{1-\alpha/2}$. Daraus folgt die Teststärke:
  $$1 - \beta \approx \Phi\bigl(\delta\sqrt{n} - z_{1-\alpha/2}\bigr)$$

<!--en:78eda780-->
- **Stichprobenumfang für die angestrebte Teststärke $1 - \beta$ ($\alpha = 0.05$, Teststärke $0.80$ $\implies$ $z_{0.975} + z_{0.80} = 1.96 + 0.842 = 2.802$):**
  $$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2} \approx \frac{7.85}{\delta^2}$$
  (Beim Zweistichprobentest verdoppelt sich der Umfang auf $k \approx \frac{15.7}{\delta^2}$ je Gruppe.)

<!--en:69baef28-->
- Bei $\delta = 0.8$ erfordert eine Teststärke von $80\%$ insgesamt $k = 25$ Zufallsstartwerte je Modell ($k = 20$ ergibt die Teststärke $72\%$, $k = 5$ nur $24\%$).

<!--en:89656cdb-->
::: {.slide title="Gepaarte Vergleiche: gemeinsames Rauschen herausrechnen"}

<!--en:a3709aea-->
[Warum konkurrierende Modelle denselben Testdatensatz brauchen]{.kicker}

<!--en:2a41f6e1-->
Der Vergleich von Klassifikator A ($912/1000$) und Klassifikator B ($921/1000$) auf *denselben* $1{,}000$ Fragen ermöglicht die Analyse der Differenz $d_i = x_{B,i} - x_{A,i} \in \{-1, 0, +1\}$ je Frage. Bei $959$ Übereinstimmungen gilt $d_i = 0$. Nur bei $41$ Fragen unterscheiden sich die Ergebnisse ($25$ für B, $16$ für A). Der **Permutationstest mit Vorzeichenwechsel** wechselt die Vorzeichen dieser $41$ Differenzen:

<!--en:40544343-->
- **Deutlich kleinerer Stichprobenumfang ($9\times$ weniger Labels):** Wegen $\mathrm{Var}(X_B - X_A) = \mathrm{Var}(X_B) + \mathrm{Var}(X_A) - 2\,\mathrm{Cov}(X_B, X_A)$ verringert positive Korrelation über dieselben Fragen das Rauschen erheblich. Um Genauigkeiten von $91\%$ und $90\%$ bei unterschiedlichen Ergebnissen auf $4\%$ der Fragen ($\mathrm{Var}(d_i) \approx 0.0399$) zu unterscheiden, genügen **$n \approx 3{,}100$ gepaarte Fragen** gegenüber **$13{,}500$ ungepaarten Fragen je Modell** (insgesamt $27{,}000$).

<!--en:1b840b79-->
::: {.slide title="Multiples Testen, p-Hacking und Bonferroni-Korrektur"}

<!--en:a78a65f4-->
[Kontrolle falscher Entdeckungen bei der Hyperparametersuche]{.kicker}

<!--en:1897ab91-->
- **Häufung falsch positiver Ergebnisse:** Bei $m = 20$ Modellen unter der Nullhypothese und Niveau $\alpha = 0.05$ beträgt die Wahrscheinlichkeit mindestens einer falschen Entdeckung $1 - 0.95^{20} \approx \mathbf{64.2\%}$.

<!--en:ad5d1cd0-->
- **Bonferroni-Korrektur ($\alpha/m$):** Eine Ablehnung nur bei $p_j \le \alpha/m$ garantiert nach der Vereinigungsabschätzung $\mathrm{FWER} \le m(\alpha/m) = \alpha$, selbst bei korrelierten Tests.

<!--en:d9df1a6f-->
- **Benjamini-Hochberg-Verfahren (Falschentdeckungsrate $\le \alpha$):** $p_{(1)} \le \dots \le p_{(m)}$ sortieren und das größte $k$ mit $p_{(k)} \le \frac{k}{m}\alpha$ suchen. Das erhält die Teststärke bei umfangreichen Vergleichen ($m \gg 100$).

<!--en:33745637-->
::: {.slide title="Konfidenzintervalle: Überdeckung durch das Verfahren und wahrer Wert"}

<!--en:77603ac8-->
[Frequentistische Überdeckungsgarantie]{.kicker}

<!--en:3e646292-->
- Ein **$1 - \alpha$-Konfidenzintervall** ist ein zufälliges Intervall $C_n = [L(x_{1:n}), U(x_{1:n})]$, dessen Konstruktionsverfahren den festen Parameter $\theta$ mit Wahrscheinlichkeit mindestens $1 - \alpha$ einschließt:
  $$P_\theta(C_n \ni \theta) \ge 1 - \alpha \quad \text{for every } \theta$$

<!--en:19560925-->
- **Was ist zufällig?** $\theta$ ist fest, das Intervall $C_n$ ändert sich von Stichprobe zu Stichprobe. In $100$ unabhängigen Experimenten enthalten $\sim 95$ Intervalle den Wert $\theta$, und $\sim 5$ verfehlen ihn.

<!--en:4fd4e704-->
- **Konfidenz- und Glaubwürdigkeitsintervall (Münze mit $9/13$):**
  - *Frequentistisches $95\%$-Wald-Intervall $(0.44, 0.94)$:* Das Verfahren strebt bei wiederholten Experimenten eine Überdeckung von $95\%$ für den festen Parameter $\theta$ an, erreicht sie aber nur asymptotisch. Bei $13$ Würfen überdeckt es den wahren Wert in etwa $91\%$ der Fälle, wenn $\theta = 1/2$ gilt.
  - *Bayessches $95\%$-Glaubwürdigkeitsintervall $(0.42, 0.87)$:* Bei einer A-priori-Verteilung $\mathrm{Beta}(1,1)$ gilt $P(\theta \in (0.42, 0.87) \mid x=9) = 0.95$.

<!--en:77b75a6a-->
::: {.slide title="Konfidenzintervalle nach Gauß, Student und Wald"}

<!--en:8cf243c6-->
[Umkehrung der Stichprobenverteilung]{.kicker}

<!--en:d4036fae-->
Umstellen der Wahrscheinlichkeitsaussage $|\bar{x} - \mu| \le h \iff \mu \in [\bar{x} - h, \bar{x} + h]$ liefert das Konfidenzintervall:

<!--en:23a83e45-->
- **Bekanntes $\sigma$ und Schätzwert $s$:** Bei bekanntem $\sigma$ gilt $h = z_{1-\alpha/2}\frac{\sigma}{\sqrt{n}}$ ($1.96\frac{\sigma}{\sqrt{n}}$ bei $95\%$). Ersetzen von $\sigma$ durch die Stichprobenstandardabweichung $s$ bringt zusätzliches Rauschen in den Nenner. Für $(\bar{x} - \mu)/(s/\sqrt{n}) \sim t_{n-1}$ braucht es daher das **Quantil der Student-$t$-Verteilung** $t_{n-1, 0.975}$ mit ihren schwereren Rändern ($12.71$ bei $n=2$, $2.776$ bei $n=5$, $1.984$ bei $n=100 \to 1.96$).

<!--en:851910d2-->
- **Wald-Intervall für die Genauigkeit ($\hat{\theta} \pm 1.96\sqrt{\hat{\theta}(1-\hat{\theta})/n}$)** und empirische Überdeckung über $1{,}000$ Versuche bei $n = 100$:

<!--en:86b8f981-->
::: {.slide title="Dualität von Tests und Intervallen und die Delta-Methode"}

<!--en:a42d5c2f-->
[Tests umkehren und Fehlerbalken auf Funktionswerte übertragen]{.kicker}

<!--en:ae8faeec-->
**1. Dualität von Tests und Konfidenzmengen**

<!--en:61bf988c-->
- Ein zweiseitiger Test zum Niveau $\alpha$ für $H_0 : \mu = \mu_0$ verwirft $H_0$ genau dann nicht, wenn $|\bar{x} - \mu_0| \le 1.96\frac{\sigma}{\sqrt{n}} \iff \mu_0 \in [\bar{x} - h, \bar{x} + h]$ gilt.

<!--en:be52aa5a-->
- **Allgemeiner Umkehrsatz:** Für *jeden* Test zum Niveau $\alpha$ für $H_0 : \theta = \theta_0$ bilden die nicht verworfenen Parameterwerte
  $$C_n = \bigl\{\theta_0 : \text{test of } H_0:\theta=\theta_0 \text{ does not reject}\bigr\}$$
  eine exakte $1 - \alpha$-Konfidenzmenge.

<!--en:8390ec66-->
- **Einhaltung der Grenzen ($29/30$ richtig):** Das Wald-Intervall $(0.90, 1.03)$ reicht über $1.0$ hinaus. Die Umkehrung des exakten Binomialtests ergibt $(0.83, 0.999) \subset [0, 1]$.

<!--en:a5cd39ba-->
**2. Delta-Methode für glatte Funktionen $g(\hat{\theta})$**

<!--en:5a846553-->
- Wie überträgt sich die Unsicherheit in $\hat{\theta}$ auf $g(\hat{\theta})$ (z. B. Log-Odds)? Bei Linearisierung von $g(\hat{\theta}) \approx g(\theta) + g'(\theta)(\hat{\theta} - \theta)$ skaliert die lokale Steigung den Standardfehler:
  $$\mathrm{se}\bigl(g(\hat{\theta})\bigr) \approx \bigl|g'(\hat{\theta})\bigr|\,\mathrm{se}(\hat{\theta})$$
  (oder $\mathrm{Var}(g(\hat{\boldsymbol{\theta}})) \approx \nabla g^\top \boldsymbol{\Sigma}\nabla g$ für Vektoren).

<!--en:893def4d-->
- **Log-Odds von $\hat{\theta} = 0.90$ ($n = 1{,}000$, $\mathrm{se} = 0.0095$):** Da $g(\theta) = \log\frac{\theta}{1-\theta}$ die Ableitung $g'(0.9) = \frac{1}{0.9 \times 0.1} \approx 11.11$ hat, gilt $\mathrm{se}(\text{log-odds}) \approx 11.11 \times 0.0095 \approx \mathbf{0.105}$.

<!--en:d87e18b0-->
::: {.slide title="Nichtparametrisches Bootstrap-Verfahren: erneutes Ziehen nach dem Plug-in-Prinzip"}

<!--en:5f86a0c6-->
[Fehlerbalken ohne geschlossene Formel]{.kicker}

<!--en:86640d5a-->
- **Plug-in-Prinzip nach Efron (1979):** Die unbekannte wahre Verteilung $F$ durch die empirische Verteilung $\hat{F}_n$ ersetzen (Masse $1/n$ an jedem beobachteten Punkt $x_i$).

<!--en:df620a7e-->
- **Algorithmus in drei Schritten:**
  1. $n$ Punkte aus $\{x_1, \dots, x_n\}$ **mit Zurücklegen** ziehen.
  2. Die Statistik $\hat{\theta}^*_r$ auf der Bootstrap-Stichprobe $r$ neu berechnen.
  3. $R$-mal wiederholen. $\mathrm{sd}(\hat{\theta}^*_{1:R})$ als $\mathrm{se}(\hat{\theta})$ verwenden. Das $2.5\%$- und das $97.5\%$-Quantil bilden die Grenzen des $95\%$-Perzentilintervalls.

<!--en:d584aebf-->
- Funktioniert bei komplexen Metriken (Median, AUC, F1, BLEU), wenn die Beobachtungen unabhängig und identisch verteilt sind und das Funktional glatt ist.

<!--en:80abbed3-->
::: {.slide title="Prüfung des Bootstrap-Verfahrens und zwei klassische Gegenbeispiele"}

<!--en:917df73b-->
[Wann das erneute Ziehen funktioniert und wann es versagt]{.kicker}

<!--en:fe6fe719-->
- **Erfolg bei glatten Statistiken:** Reproduziert den exakten Standardfehler $0.0090$ der Genauigkeit und schätzt den Standardfehler des Medians von $\mathrm{Exp}(1)$ ($\approx 0.071$) ohne Kenntnis der Dichte $f(m)$:

<!--en:34938505-->
- **Grenze 1: Nichtglatte Extremwerte ($\hat{\theta} = \max_i x_i$ auf $U(0, \theta)$):** Für jede Bootstrap-Stichprobe gilt $\hat{\theta}^* \le \hat{\theta} < \theta$ (Intervallüberdeckung $0\%$). Am Stichprobenmaximum häuft sich Masse:
  $$P(\hat{\theta}^* = \hat{\theta}) = 1 - \left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} 1 - \frac{1}{e} \approx 0.632$$

<!--en:e923e480-->
- **Grenze 2: Benchmarks mit Gruppenstruktur ($G$ Passagen $\times$ $m$ Fragen, Korrelation innerhalb einer Passage $\rho$):**
  $$\mathrm{Var}(\hat{\theta}) = \frac{\theta(1 - \theta)}{Gm}\bigl(1 + (m - 1)\rho\bigr)$$
  Bei $m = 20, \rho = 0.1$ unterschätzt das Ziehen einzelner Fragen $\mathrm{se}$ um den Faktor $\sqrt{2.9} \approx \mathbf{1.7\times}$. **Abhilfe:** Beim Cluster-Bootstrap ganze Passagen ziehen.

<!--en:48d8fa51-->
::: {.slide title="Verständnisfragen: Bessel-Korrektur, Teststärke und Bootstrap-Verfahren"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:36bf4d3d-->
Drei praktische Fragen zur statistischen Versuchsplanung:

<!--en:bc9846f7-->
1. **Stichprobenumfang für eine vorgegebene Effektgröße:** Wie viele unabhängige Beobachtungen sind nach $n \approx 7.85/\delta^2$ ($\alpha = 0.05$, Teststärke $80\%$) erforderlich, um einen standardisierten Effekt von $\delta = 0.5$ zu erkennen? Wie viele, wenn $\delta$ auf ein Fünftel, also auf $\delta = 0.1$, schrumpft?

<!--en:35d602e6-->
2. **Fehlerbalken bei Benchmarks mit Gruppenstruktur:** Ein Bildverarbeitungsbenchmark umfasst $G = 200$ Videos mit je $m = 16$ Einzelbildern. Die Vorhersagen für Einzelbilder desselben Videos haben die Korrelation $\rho = 0.2$. Um welchen Faktor unterschätzt ein naives Bootstrap-Verfahren auf Einzelbildebene den wahren Standardfehler?

<!--en:502d61aa-->
3. **Warum das Stichprobenmaximum in $63.2\%$ der Bootstrap-Stichproben wiederkehrt:** Wie wahrscheinlich ist es bei $n$ verschiedenen Datenpunkten, dass eine bestimmte Beobachtung $x_i$ in einer Bootstrap-Stichprobe vom Umfang $n$ für $n \to \infty$ *kein einziges Mal* gezogen wird?

<!--en:557b98b6-->
**Lösungen:**

<!--en:c0df709d-->
1. Bei $\delta = 0.5$ sind $n \approx 7.85 / 0.25 \approx \mathbf{32}$ Beobachtungen nötig. Sinkt $\delta$ um den Faktor $5\times$ auf $0.1$, steigt der Datenbedarf um den Faktor $5^2 = 25\times$ auf $n \approx 7.85 / 0.01 = \mathbf{785}$ Beobachtungen.

<!--en:8771e334-->
2. Die Varianz steigt um den Faktor $1 + (m - 1)\rho = 1 + 15(0.2) = 4.0$. Das naive Bootstrap-Verfahren auf Einzelbildebene unterschätzt den wahren Standardfehler also um den Faktor **$\sqrt{4.0} = 2.0\times$**.

<!--en:7aa09c20-->
3. Jede der $n$ Ziehungen verfehlt $x_i$ mit Wahrscheinlichkeit $1 - 1/n$. Alle $n$ Ziehungen verfehlen $x_i$ daher mit Wahrscheinlichkeit **$(1 - 1/n)^n \to e^{-1} \approx 36.8\%$** (jede Bootstrap-Stichprobe enthält also im Mittel $\approx 63.2\%$ der verschiedenen ursprünglichen Punkte).

<!--en:b2b06a82-->
::: {.slide title="Zusammenfassung: Statistische Schätzung und Hypothesentests"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:f1bb0327-->
| Statistisches Verfahren | Mathematische Formel oder Regel | Vergleichswert aus §{sec_mdl-statistics} | Anwendung im Deep Learning |

<!--en:76fe15f8-->
| **Stichprobenanteil $\hat{\theta}_n$** | $\mathbb{E}[\hat{\theta}_n] = \theta$, $\mathrm{se}(\hat{\theta}_n) = \sqrt{\theta(1-\theta)/n}$ | $912/1000 \implies 91.2\% \pm 0.90\%$ ($\mathrm{se}$) | Fehlerbalken für die Testgenauigkeit bei Klassifikation |

<!--en:2abfd26e-->
| **Erwartungstreue Varianzschätzung $s^2$** | $s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$ (`ddof=1`) | Division durch $n$ ergibt die Verzerrung $-\sigma^2/n$ | Schätzung der Varianz über Zufallsstartwerte bei kleinem $k$ |

<!--en:249b9232-->
| **Dreieck aus Verzerrung und Standardfehler** | $\mathrm{MSE}(\hat{\theta}_n) = \mathrm{Bias}(\hat{\theta}_n)^2 + \mathrm{Var}(\hat{\theta}_n)$ | Geschrumpfter Mittelwert $0.65\bar{x}$ ($0.35$) schlägt $\bar{x}$ ($0.53$) | Regularisierung, Schrumpfung und Modellkomplexität |

<!--en:ade237bc-->
| **Monte-Carlo-Integration** | $\mathrm{RMSE}(\frac{1}{N}\sum_{i=1}^N g(\mathbf{x}_i)) = \sigma / \sqrt{N}$ | Schneller als Gitter der Ordnung $2$ ($N^{-2/d}$) für $d > 4$ | Schätzung von Verlust, ELBO und Gradienten bei hohem $d$ |

<!--en:1d8dcdce-->
| **Permutationstests und gepaarte Tests** | $p = \frac{1 + \#\{\|\text{gap}_r\| \ge \|\text{gap}_{\text{obs}}\|\}}{R+1}$; $P_{H_0}(p \le \alpha) \le \alpha$ | Paarung senkt den Aufwand für $90\%\text{ vs. }91\%$ um den Faktor $9\times$ | Modellvergleich auf gemeinsamen Testdatensätzen oder mit gemeinsamen Zufallsstartwerten |

<!--en:f5df41be-->
| **Teststärke und Stichprobenumfang** | $1 - \beta \approx \Phi(\delta\sqrt{n} - z_{1-\alpha/2})$; $n \approx 7.85/\delta^2$ | $\delta = 0.8$: $k=25$ Zufallsstartwerte ($80\%$), $k=5$ ($24\%$) | Aufwandsplanung für Ablationsstudien und Bonferroni-Korrektur ($\alpha/m$) |

<!--en:7d364e7e-->
| **Konfidenzintervalle und Bootstrap-Verfahren** | $P_\theta(C_n \ni \theta) \ge 1-\alpha$; $n$ Werte mit Zurücklegen ziehen | $\bar{x} \pm t_{n-1, 0.975}s/\sqrt{n}$; ganze Gruppen neu ziehen | Gültige Unsicherheitsangaben für Genauigkeit, Log-Odds und F1/BLEU |
