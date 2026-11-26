<!--en:daab1405-->
# Mehrschichtige Perzeptronen

<!--en:e2e253a5-->
In diesem Kapitel lernen wir die ersten *tiefen* neuronalen Netze kennen. In :numref:`chap_classification` wies die Softmax-Regression jedem der 784 Pixel eines Modebilds aus Fashion-MNIST ein festes Gewicht zu und summierte die Beiträge. Damit zählte die Stimme eines Pixels für eine Klasse immer gleich viel, unabhängig von seiner Umgebung. Ein heller Pixel kann jedoch zu einem Ärmel, einem Handtaschenriemen oder einem Kragen gehören. Erst die benachbarten Pixel zeigen, wozu er gehört. Schon bei den vier möglichen Kombinationen zweier Bits kann eine einzelne lineare Schicht das exklusive Oder (XOR) nicht berechnen. Auf diese Grenze von Frank Rosenblatts einschichtigem *Perzeptron* wiesen Marvin Minsky und Seymour Papert 1969 hin.

<!--en:92bec8c6-->
*Mehrschichtige Perzeptronen* (engl. multilayer perceptron, MLP) überwinden diese Grenze linearer Modelle. Vor der Ausgabeschicht liegen eine oder mehrere vollständig verbundene *versteckte Schichten* (engl. hidden layers). Jede verbindet eine affine Transformation mit einer nichtlinearen Aktivierungsfunktion. Mit *Backpropagation* trainieren wir alle Schichten gemeinsam :cite:`Rumelhart.Hinton.Williams.ea.1988`. Zunächst zeigen wir, warum lineare Modelle an Wechselwirkungen zwischen Merkmalen wie beim XOR scheitern. Wir untersuchen, wie versteckte Schichten den Eingaberaum falten und dabei mit zunehmender Tiefe exponentiell viele lineare Bereiche erzeugen. Außerdem betrachten wir die Ableitungen gängiger Aktivierungsfunktionen (ReLU, GELU, Sigmoidfunktion und $\tanh$) (:numref:`sec_mlp`). Für Fashion-MNIST schreiben wir ein ReLU-MLP mit 256 Einheiten in zwei Varianten, einer eigenen und einer kompakten Implementierung (:numref:`sec_mlp-implementation`). Bei fast gleicher Laufzeit pro Epoche auf der GPU vermeidet das MLP rund ein Fünftel der Klassifikationsfehler der Softmax-Regression ($\sim 87\%$ gegenüber $\sim 84\%$ Genauigkeit). Es lernt, lokale Kontraste zu erkennen, statt unscharfe Schablonen für ganze Kleidungsstücke zu bilden.

<!--en:774825a8-->
Versteckte Schichten werfen auch drei Grundfragen des Deep Learning auf, denen wir im Rest des Kapitels nachgehen. Erstens leiten wir in :numref:`sec_backprop` drei Regeln der Matrixrechnung her, mit denen wir ein MLP im Rückwärtsmodus (engl. reverse mode) differenzieren. Wir prüfen, ob die Ergebnisse bis auf Maschinengenauigkeit mit denen des automatischen Differenzierens übereinstimmen. Außerdem bestimmen wir den Aktivierungsspeicher und die Zahl der Gleitkommaoperationen (FLOPs), die das Training benötigt. Zweitens verfolgen wir in :numref:`sec_numerical_stability` die Signal- und Gradientenvarianz über $K$ Schichten hinweg. Wir leiten die Initialisierungen nach *Xavier* und *He* her, die in einem MLP mit 10 Schichten verschwindende und explodierende Gradienten verhindern. Wir zeigen auch, warum wir zufällige Gewichte brauchen, um die Symmetrie zwischen versteckten Einheiten zu brechen. Drittens untersuchen wir in :numref:`sec_generalization_deep` und :numref:`sec_dropout`, warum selbst stark überparametrisierte Netze generalisieren. Dabei betrachten wir das Auswendiglernen zufälliger Labels, den doppelten Abstieg (engl. double descent), den impliziten Bias des Gradientenabstiegs, Early Stopping und die Regularisierung durch multiplikatives Rauschen beim *Dropout*. In :numref:`sec_kaggle_house` setzen wir schließlich alle diese Werkzeuge beim Regressionsbenchmark Ames Housing auf Kaggle ein.

<!--en:c34a1e9b-->
Mehrschichtige Perzeptronen überwinden die Grenzen linearer Modelle<br>
**Versteckte Schichten · Backpropagation · varianzerhaltende Initialisierung · Generalisierung · Dropout · Kaggle**

<!--en:74015760-->
::: {.slide title="Warum lineare Klassifikatoren bei praktischen Aufgaben an Grenzen stoßen"}

<!--en:acc8203e-->
[Das Kernproblem]{.kicker}

<!--en:ef9c3705-->
- **Keine Wechselwirkungen zwischen Merkmalen (XOR):** Kippt das Eingabebit $x_1$ beim exklusiven Oder, hängt die Richtung des Ausgabewechsels von $x_2$ ab. Keine einzelne Hyperebene kann die vier Ecken von $\{0, 1\}^2$ nach Klassen trennen.

<!--en:32fb9b82-->
- **Pixelgewichte ohne Kontext:** Ein heller Pixel trägt in Fashion-MNIST immer gleich viel zu einer Klasse bei, ob seine Nachbarn einen Kragen, einen Ärmel oder einen Handtaschenriemen bilden. Das begrenzt die Genauigkeit linearer Modelle auf $\sim 84\%$.

<!--en:ef3358bf-->
- **Die Lösung im Deep Learning:** Wir fügen versteckte Schichten $\mathbf{H} = \phi(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ ein. Sie lernen gemeinsam mit dem Klassifikator an der Ausgabe nichtlineare Merkmalskombinationen ($\sim 87\%$ Genauigkeit).

<!--en:c494cd83-->
::: {.slide title="Drei grundlegende Herausforderungen tiefer Netze"}

<!--en:d70cdc74-->
[Was sich mit der Tiefe ändert]{.kicker}

<!--en:7a9116a5-->
1. **Effiziente Gradientenberechnung:** Endliche Differenzen erfordern $203{,}530$ Vorwärtsdurchläufe pro Schritt. Die *Backpropagation* speichert die Aktivierungen aus dem Vorwärtsdurchlauf im GPU-Speicher zwischen und berechnet alle Gradienten in einem Rückwärtsdurchlauf.

<!--en:12095650-->
2. **Numerische Stabilität über viele Schichten:** Über $K$ Schichten wirkt auf Signale und Gradienten ein Produkt aus $K$ Gewichts- und Steigungsmatrizen. Das erfordert varianzerhaltende Initialisierungen wie die nach *Xavier* und *He*.

<!--en:e95fbf07-->
3. **Generalisierung jenseits der Interpolation:** Überparametrisierte MLPs können zufälliges Rauschen auswendig lernen. Der implizite Bias von SGD, Early Stopping und *Dropout* helfen ihnen dennoch, auf realen Daten zu generalisieren.

<!--en:4b951351-->
::: {.slide title="Fashion-MNIST als Vergleichsmaßstab im gesamten Kapitel"}

<!--en:1e178678-->
[Durchgängiges Beispiel]{.kicker}

<!--en:22a276e5-->
- **Softmax-Regression als Referenz ($784 \to 10$):** Mit $7{,}850$ Parametern ($0.8\textrm{ MB}$ für Aktivierungen bei $n = 256$) stagniert die Validierungsgenauigkeit bei $\sim 84\%$.

<!--en:2167195f-->
- **ReLU-MLPs mit 1 oder 2 versteckten Schichten:** Die Zahl der Parameter steigt auf $203{,}530$ bis $269{,}322$. Die Genauigkeit erreicht $\sim 87\%$ bei nahezu gleicher Laufzeit pro Epoche auf der GPU.

<!--en:9c6db4fd-->
- **Tiefentest mit 10 Schichten ($795{,}658$ Parameter):** Die Genauigkeit verharrt auf dem Zufallsniveau von $10\%$, wenn wir mit $\mathcal{N}(0, 0.01^2)$ initialisieren. Mit He-Initialisierung erreicht das Netz beim Training zuverlässig $85\text{--}87\%$ ($\tau = \sqrt{2/n_{\textrm{in}}}$).

<!--en:4c503820-->
::: {.slide title="Architektur und Mathematik im Überblick"}

<!--en:99e0030a-->
[Kapitelüberblick]{.kicker}

<!--en:390bf35c-->
- **Darstellung und Merkmale:** Nichtlineare Aktivierungen verhindern, dass sich das Netz auf eine lineare Abbildung reduziert. Mit zunehmender Tiefe faltet es den Raum in $2^m$ Bereiche und lernt Kontrastmuster anstelle von Schablonen für ganze Kleidungsstücke.

<!--en:f5cb7a9d-->
- **Optimierung und Stabilität:** Die drei lokalen Regeln für Vektor-Jacobi-Produkte ($\mathbf{X}^\top\mathbf{G}$, $\mathbf{G}\mathbf{W}^\top$, $\odot\,\phi'$) benötigen $\sim 2\times$ so viele FLOPs wie der Vorwärtsdurchlauf. Die Bedingung $h\tau^2\mathbb{E}[\phi'(z)^2] = 1$ verhindert, dass die Produkte verschwinden oder explodieren.

<!--en:f146a231-->
- **Generalisierung und maschinelles Lernen mit Tabellendaten:** Doppelter Abstieg, impliziter Bias zugunsten von Lösungen minimaler Norm, invertiertes Dropout $h' = \xi h / (1 - p)$ und 5-fache Kreuzvalidierung ohne Datenleckage auf Kaggle.

<!--en:380a87fc-->
::: {.slide title="Verständnisfragen und Lernleitfaden zum Kapitel"}

<!--en:6a3f295c-->
[Verständnis prüfen und Zusammenhänge erkennen]{.kicker}

<!--en:c04d3292-->
Prüfen Sie vor dem ersten Abschnitt Ihre Intuition anhand zweier Grundfragen, die die folgenden Abschnitte miteinander verbinden:

<!--en:0d9160f3-->
1. **Reduktion auf eine lineare Abbildung:** Wir schalten 10 vollständig verbundene Schichten der Breite $256$ ohne Aktivierungsfunktionen hintereinander ($\phi(z) = z$). Hat das Netz mehr Ausdruckskraft als eine einzelne lineare Schicht der Größe $784 \times 10$?

<!--en:c3236889-->
2. **Speicherbedarf von Aktivierungen und Parametern:** Wir trainieren ein tiefes Netz mit Minibatches der Größe $n = 256$. Warum kann der Aktivierungsspeicher den Parameterspeicher leicht übersteigen, selbst wenn jede versteckte Schicht schmaler ist als die Eingabe?

<!--en:557b98b6-->
**Antworten:**

<!--en:c9289f5b-->
1. **Nein:** Das Produkt $(\mathbf{X}\mathbf{W}^{(1)})\mathbf{W}^{(2)}\cdots\mathbf{W}^{(10)}$ lässt sich zu einer einzigen Matrix $\mathbf{X}\mathbf{W}$ zusammenfassen. Ihr Rang beträgt höchstens $256$ und an der Ausgabe höchstens $10$. Nur nichtlineare Aktivierungen verhindern diese algebraische Reduktion.

<!--en:58804622-->
2. **Einfluss der Batchgröße:** Eine Gewichtsmatrix $\mathbf{W}^{(k)} \in \mathbb{R}^{h \times h}$ speichert unabhängig von der Batchgröße $h^2$ Gleitkommazahlen. Die zwischengespeicherte Aktivierung $\mathbf{H}^{(k)} \in \mathbb{R}^{n \times h}$ aus dem Vorwärtsdurchlauf belegt dagegen $nh$ Gleitkommazahlen pro Schicht. Der Aktivierungsspeicher wächst also linear sowohl mit der Batchgröße $n$ als auch mit der Tiefe $K$.
