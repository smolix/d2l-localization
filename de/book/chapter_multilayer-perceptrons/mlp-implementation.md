<!--en:afc72d25-->
# Implementierung

<!--en:ac1d7b3f-->
In :numref:`sec_mlp` haben wir die Gewichte eines ReLU-Netzes mit zwei Einheiten von Hand so gewählt, dass es XOR berechnet. Nun lernt das Netz mit stochastischem Gradientenabstieg (engl. stochastic gradient descent, SGD) auf Minibatches versteckte Merkmale direkt aus Fashion-MNIST-Bildern. Zwischen die Eingabepixel und die Klassenlogits fügen wir eine einzige versteckte Schicht (engl. hidden layer) mit $h = 256$ ReLU-Einheiten ein. Bei $d = 784$ Eingabepixeln und $q = 10$ Klassenlogits steigen die Parameteranzahl und die Anzahl der Gleitkommaoperationen (FLOPs) pro Bild auf das 26-Fache ($203{,}530$ Parameter gegenüber $7{,}850$ bei der Softmax-Regression).

<!--en:880fe95e-->
Für dieses mehrschichtige Perzeptron (engl. multilayer perceptron, MLP) mit zwei Schichten schreiben wir eine eigene Implementierung (`MLPScratch`) und eine kompakte Implementierung (`MLP`) mit den `Sequential`-Containern der Frameworks. Vor dem Training prüfen wir das untrainierte Netz mit drei kurzen Plausibilitätstests. Wir vergleichen seine Lernkurve und die Laufzeit pro Epoche auf der GPU direkt mit denen der Softmax-Regression. Anschließend betrachten wir die Gewichtsmuster, die die versteckten Einheiten (engl. hidden units) in $\mathbf{W}^{(1)}$ lernen.

<!--en:6275dcac-->
## Eigene Implementierung

<!--en:c720f4d1-->
### Parameter und Vorwärtsdurchlauf

<!--en:71f690af-->
Wir formen jedes Bild der Größe $28 \times 28$ zu einem Vektor mit $d = 784$ Merkmalen um. Unser MLP hat eine einzige versteckte Schicht mit $h = 256$ versteckten Einheiten und $q = 10$ Ausgaben (:numref:`fig_mdl-mlp-arch`). Dafür legt es vier Parametertensoren an:

<!--en:316e64bc-->
![Architektur des MLP für Fashion-MNIST. Die $n$ Bilder eines Minibatchs bilden nach dem Umformen eine Matrix der Form $n \times 784$. Eine affine Schicht mit anschließender ReLU erzeugt daraus eine versteckte Darstellung der Form $n \times 256$. Diese Darstellung $\mathbf{H}$ wird auf $n \times 10$ Klassenlogits $\mathbf{O}$ projiziert.](../img/mdl-mlp-arch.svg)

<!--en:6ddb187d-->
Damit für den direkten Vergleich mit der Softmax-Regression dieselben Bedingungen gelten, ziehen wir die Einträge beider Gewichtsmatrizen aus $\mathcal{N}(0, 0.01^2)$ und setzen beide Bias-Vektoren auf $0$. In :numref:`sec_numerical_stability` sehen wir, warum die Zufallsinitialisierung notwendig ist, um die Symmetrie zwischen versteckten Einheiten zu brechen, und welchen Vorteil die He-Initialisierung gegenüber `std=0.01` bietet.

<!--en:bb2849d0-->
Nun implementieren wir die elementweise Aktivierung `relu(X)` selbst. Wir multiplizieren `X` mit der booleschen Maske `X > 0`. Dadurch bleiben positive Einträge erhalten und negative werden null:

<!--en:4514ddef-->
TensorFlow erlaubt keine Multiplikation eines Gleitkommatensors mit einem booleschen Tensor. Deshalb wandeln wir die Maske zunächst in den Datentyp `X.dtype` um.

<!--en:bf8bd2d7-->
Die Berechnung der versteckten Schicht lagern wir in die Methode `hidden(self, X)` aus. Sie bringt jeden Minibatch von Bildern in die Form `(-1, self.num_inputs)` und berechnet $\mathbf{H} = \mathrm{relu}(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$. Damit gibt `forward(self, X)` direkt die unnormalisierten Klassenlogits $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ zurück, die `Classifier.loss` unverändert übernimmt:

<!--en:c74c4a87-->
### Prüfungen vor dem Training

<!--en:46264d0b-->
Bevor wir ein langes Training starten, schicken wir einen Slice mit nur 8 Bildern durch das *untrainierte* Netz. So prüfen wir innerhalb von Millisekunden drei Invarianten:

<!--en:2bb1d429-->
1. **Formen:** `H` hat die Form `(8, 256)` und `O` die Form `(8, 10)`.

<!--en:f11edd71-->
2. **Anfangsverlust:** Gewichte nahe null ergeben Logits nahe null und damit fast gleiche Klassenwahrscheinlichkeiten ($\hat{y}_j \approx 1/10$). Der anfängliche Kreuzentropieverlust (engl. cross-entropy loss) liegt daher nahe bei $\log 10 \approx 2.303$.

<!--en:90efbaf9-->
3. **Anteil aktiver ReLU-Einheiten:** Sind die Bias-Terme null und die Gewichte symmetrisch verteilt, so sind etwa $50\%$ der Voraktivierungen in der versteckten Schicht positiv.

<!--en:70d292d9-->
### Training

<!--en:b7ccf386-->
Nun trainieren wir `MLPScratch` für $30$ Epochen auf Fashion-MNIST mit der Minibatchgröße $256$ und der Lernrate $\eta = 0.1$. Wir übergeben `num_gpus=1`, damit `Trainer` eine GPU nutzt, sofern eine verfügbar ist:

<!--en:d52bb9d8-->
## Kompakte Implementierung

<!--en:d76a6124-->
Mit dem `Sequential`-Container und den eingebauten linearen Schichten und ReLU-Schichten des jeweiligen Frameworks reichen wenige Zeilen für genau dieselbe Architektur. Die Parameterarrays müssen wir dabei nicht selbst anlegen. Wir ziehen die Gewichte aus $\mathcal{N}(0, 0.01^2)$ und setzen die Bias-Terme auf $0$, genau wie bei `MLPScratch`:

<!--en:ee350d94-->
## Vergleich mit der Softmax-Regression

<!--en:04762695-->
### Genauigkeit

<!--en:607f56cb-->
Trainieren wir die kompakte Implementierung `MLP` und `d2l.SoftmaxRegression` für $30$ Epochen mit denselben Einstellungen. Für jedes Modell speichern wir die mittlere Laufzeit pro Epoche in `seconds` und den Verlauf der Validierungsgenauigkeit (engl. validation accuracy) in `curves`:

<!--en:b53d9f1c-->
Wir zeichnen alle drei Verläufe der Validierungsgenauigkeit aus `curves` für die Epochen 1 bis 30. So können wir unsere eigene und die kompakte MLP-Implementierung direkt mit der Softmax-Regression als linearer Baseline vergleichen:

<!--en:1fe7fd7f-->
Um Schwankungen zwischen einzelnen Epochen zu glätten, geben wir für jedes Modell den Mittelwert der Validierungsgenauigkeit über die letzten fünf Epochen aus (`acc[-5:]`). Daneben geben wir die Genauigkeit von `MLPScratch` auf dem Trainingsdatensatz aus:

<!--en:60933eae-->
An den Lernkurven fallen zwei Muster auf:

<!--en:396b95c1-->
1. **Langsamer Start durch `std=0.01`:** In Epoche 1 liegen beide MLPs tatsächlich hinter der Softmax-Regression. Die Einträge von $\mathbf{W}^{(1)}$ und $\mathbf{W}^{(2)}$ liegen anfangs bei etwa $0.01$. Dadurch werden die versteckten Aktivierungen $\mathbf{H}$ und die durch Backpropagation berechneten Gradienten für $\mathbf{W}^{(1)}$ mit dem Faktor $0.01$ gedämpft, bis die Gewichte in $\mathbf{W}^{(2)}$ wachsen. Diesen Flaschenhals beseitigen wir in :numref:`sec_numerical_stability` mit der He-Initialisierung.

<!--en:7f86d6f6-->
2. **Ein Fünftel weniger Klassifikationsfehler:** In den Epochen 26–30 stagniert die Validierungsgenauigkeit der Softmax-Regression bei rund $84\%$ ($16\%$ Fehlerrate). Beide MLPs erreichen dagegen eine Validierungsgenauigkeit von $\sim 87\%$ ($\sim 13\%$ Fehlerrate und $87$–$90\%$ Trainingsgenauigkeit). Damit vermeiden sie rund $20\%$ der Fehler der Softmax-Regression.

<!--en:4ba51b55-->
Welche Fehler bleiben? In :numref:`sec_fashion_mnist` haben wir gesehen, dass die vier Arten von Oberteilen (T-Shirt, Pullover, Mantel und Hemd) ähnlich aussehen. Verwechslungen liegen hier also nahe. Für jedes der beiden gerade trainierten Modelle zählen wir die Validierungsfehler und ermitteln, wie viele davon auf Verwechslungen zwischen Oberteilen entfallen:

<!--en:0be159e8-->
Bei beiden Modellen machen Verwechslungen zwischen zwei Oberteilen mehr als die Hälfte aller Fehler aus. Dem MLP unterlaufen weniger solche Fehler als der Softmax-Regression, doch die Oberteile bleiben die größte Schwierigkeit bei Fashion-MNIST.

<!--en:3b95e448-->
### Die erlernten Merkmale

<!--en:a55c1deb-->
In :numref:`sec_softmax_scratch` musste bei der Softmax-Regression jede der 10 Spalten der Gewichtsmatrix $\mathbf{W}$ als Schablone für ein ganzes Kleidungsstück dienen. Was haben die 256 versteckten Einheiten unseres MLP stattdessen gelernt? Dazu lesen wir die trainierten Gewichtsmatrizen $\mathbf{W}^{(1)}$ und $\mathbf{W}^{(2)}$ als NumPy-Arrays aus:

<!--en:8c834513-->
Wir wählen die 10 versteckten Einheiten aus, deren Gewichtsvektoren die größte euklidische Norm haben, und stellen ihre Gewichte als Bilder der Größe $28 \times 28$ dar. Jedes Bild beschriften wir mit der Kleidungsklasse, zu der die Einheit das größte positive Ausgangsgewicht hat:

<!--en:d5a5f120-->
Diese Bilder sehen anders aus als Schablonen für ganze Kleidungsstücke. Jede versteckte Einheit mit großer Gewichtsnorm gewichtet bestimmte Bildbereiche positiv (rot) und benachbarte Bereiche negativ (blau). Sie reagiert also auf einen **Kontrast** zwischen hellen Pixeln in einem Bereich und dunklen Pixeln direkt daneben, etwa entlang der Schultern oder seitlich am Oberkörper. Die meisten Muster erstrecken sich immer noch über einen großen Teil des Kleidungsstücks. Es wäre daher falsch, sie als kleine Kantendetektoren zu deuten. Die Ausgabeschicht kombiniert $256$ solcher Kontraste zu ihren zehn Scores.

<!--en:e9f803dc-->
### Parameter, Rechenaufwand und Laufzeit

<!--en:a6e3d9a8-->
| Modell | Parameter | Multiplikationsadditionen pro Bild | Validierungsgenauigkeit (Epochen 26–30) |

<!--en:389a140e-->
| Softmax-Regression | $7{,}850$ | $7{,}840$ | $\sim 84\%$ |

<!--en:299ffad7-->
| MLP ($h = 256$ ReLU-Einheiten) | $203{,}530$ | $203{,}264$ | $\sim 87\%$ |

<!--en:8ce67db2-->
Unser MLP führt pro Bild 26-mal so viele Multiplikationsadditionen (engl. multiply-adds) aus wie die Softmax-Regression. Vergleichen wir nun die tatsächlichen Laufzeiten pro Epoche:

<!--en:143eead1-->
Auf moderner Hardware, insbesondere auf einer GPU, brauchen beide Modelle pro Epoche fast gleich lange. Bei einer Breite von $h = 256$ entfällt der größte Teil der Laufzeit auf das Laden der Daten und den Zusatzaufwand durch Python oder Kernelstarts. Obwohl der Rechenaufwand für die dichten Matrizen auf das 26-Fache steigt, kostet das kaum zusätzliche Zeit. In Übung :exref:`ex_mlp-implementation-width` untersuchen Sie, ab welcher Breite die Matrixoperationen mehr Zeit beanspruchen als das Laden der Daten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:29b0ad35-->
Vor dem Training genügt ein Vorwärtsdurchlauf mit einem kleinen Batch, um die Formen der Zwischenergebnisse und des Ausgabetensors zu prüfen. Dabei sehen wir auch, ob der anfängliche Kreuzentropieverlust nahe bei $\log q \approx 2.30$ liegt und ob etwa die Hälfte der versteckten ReLU-Einheiten aktiv ist. Bei Fashion-MNIST erhöht eine einzige versteckte Schicht mit 256 ReLU-Einheiten die Validierungsgenauigkeit von $\sim 84\%$ auf $\sim 87\%$. Die Fehlerrate sinkt damit um fast ein Fünftel, bei nahezu gleicher Laufzeit pro Epoche auf der GPU. Die versteckten Einheiten lernen anstelle unscharfer Schablonen für ganze Kleidungsstücke Kontrastmuster. Sie gewichten benachbarte Bildbereiche mit entgegengesetzten Vorzeichen. Die Ausgabeschicht kombiniert diese Muster.

<!--en:2294243a-->
## Übungen

<!--en:9fee2929-->
1. [code] **Eine versteckte Schicht ohne Aktivierungsfunktion.** Ersetzen Sie `relu` durch die Identitätsfunktion in `MLPScratch` und trainieren Sie das Modell wie in diesem Abschnitt.
    1. Vergleichen Sie seine Validierungsgenauigkeit nach 30 Epochen mit der des MLP und der Softmax-Regression.
    1. Berechnen Sie nach dem Training $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ und $\mathbf{b} = \mathbf{W}^{(2)\top}\mathbf{b}^{(1)} + \mathbf{b}^{(2)}$. Prüfen Sie an einem Minibatch, ob $\mathbf{X}\mathbf{W} + \mathbf{b}$ die Logits des Modells reproduziert.
    1. Wie Übung :exref:`ex_mlp-linear-networks` gezeigt hat, berechnet ein solches Netz genau diejenigen affinen Abbildungen, bei denen der Rang der Gewichtsmatrix höchstens so groß ist wie die Breite der versteckten Schicht. Erklären Sie, warum diese Einschränkung hier keine Rolle spielt. Unterhalb welcher Breite der versteckten Schicht würde sie die Darstellungsfähigkeit des Modells auf Fashion-MNIST einschränken?

<!--en:e720c305-->
1. [code] **Eine einzige versteckte Einheit.** Setzen Sie `num_hiddens=1` in `MLPScratch`. Die versteckte Schicht enthält dann nur eine ReLU-Einheit mit den Gewichten $\mathbf{w}^{(1)} \in \mathbb{R}^{784 \times 1}$ und dem Bias $b^{(1)} \in \mathbb{R}$. Jede Eingabe $\mathbf{x} \in \mathbb{R}^{1 \times 784}$ reduziert sich damit auf den Skalar $t = \operatorname{ReLU}(\mathbf{x}\mathbf{w}^{(1)} + b^{(1)}) \geq 0$. Die Logits sind $\mathbf{o} = t\, \mathbf{w}^{(2)} + \mathbf{b}^{(2)}$ mit $\mathbf{w}^{(2)} \in \mathbb{R}^{1 \times 10}$ und $\mathbf{b}^{(2)} \in \mathbb{R}^{10}$.
    1. Zeigen Sie, dass das Modell jede Klasse auf höchstens einem Intervall von $t$ vorhersagt. Zeigen Sie auch, dass die Reihenfolge dieser Intervalle auf der $t$-Achse durch die Einträge von $\mathbf{w}^{(2)}$ festliegt. Beschreiben Sie die Bereiche des Eingaberaums, die das Modell den Klassen zuordnet. Was folgt daraus für die Trennung der zehn Fashion-MNIST-Klassen?
    1. Angenommen, nach einer Aktualisierung gilt $\mathbf{x}\mathbf{w}^{(1)} + b^{(1)} < 0$ für jedes Trainingsbeispiel. Welche Parameter erhalten von da an Gradienten ungleich null und was sagt das Netz vorher? Warum schadet es wenig, wenn dies bei einer der 256 versteckten Einheiten aus diesem Abschnitt geschieht?
    1. Trainieren Sie das Modell fünfmal. Geben Sie für jeden Lauf die Validierungsgenauigkeit und den Anteil der Validierungsbeispiele mit $t > 0$ an. Den Wert des Skalars liefert `model.hidden`. Vergleichen Sie jeden Lauf, bei dem dieser Anteil null ist, mit Ihrer Antwort auf (b).

<!--en:1d66cb42-->
1. [code] **Breite.** Trainieren Sie `MLP` wie in diesem Abschnitt für 30 Epochen mit der Lernrate 0.1. Verwenden Sie für die versteckte Schicht die Breiten $h \in \{16, 64, 256, 1024, 4096\}$ und erfassen Sie jeweils die abschließende Validierungsgenauigkeit und die Laufzeit pro Epoche.
    1. Tragen Sie beide Größen gegen $h$ auf und skalieren Sie die Breitenachse logarithmisch. Ab welcher Breite ist der Zuwachs an Genauigkeit höchstens noch so groß wie die Schwankungen von Epoche zu Epoche?
    1. Die Anzahl der Multiplikationsadditionen wächst proportional zu $h$. Messen Sie die Dauer eines Durchlaufs durch die Trainingsdaten, bei dem Sie ohne Modell nur die Minibatches einlesen. Erklären Sie damit den Teil der Laufzeitkurve, der mit $h$ nicht ansteigt. Ab welcher Breite entfällt auf Ihrer Hardware der größte Teil der Laufzeit auf die Rechenoperationen?
    1. Wiederholen Sie das Training für die Breiten 64 und 1024 mit den Lernraten $\{0.03, 0.1, 0.3\}$. Hängt die beste Lernrate von der Breite ab?
   :exlabel:`ex_mlp-implementation-width`

<!--en:0df17ad9-->
[Diskussion](https://d2l.discourse.group/t/92)

<!--en:3f951718-->
[Diskussion](https://d2l.discourse.group/t/93)

<!--en:40276b30-->
[Diskussion](https://d2l.discourse.group/t/227)

<!--en:54f2c178-->
[Diskussion](https://d2l.discourse.group/t/17985)

<!--en:2f0e5344-->
Implementierung mehrschichtiger Perzeptrons für Fashion-MNIST<br>**Parameterregistrierung · Prüfungen vor dem Training · Sequential im Framework · Genauigkeit und erlernte Merkmale**

<!--en:7a7ea72c-->
::: {.slide title="Von der XOR-Lösung von Hand zu erlernten Bildmerkmalen"}

<!--en:4c8c7a7a-->
[Die empirische Frage]{.kicker}

<!--en:0304f462-->
- **Lineare Baseline:** Die Softmax-Regression nutzt $784 \times 10 + 10 = 7{,}850$ Parameter. Ihre Validierungsgenauigkeit stagniert bei etwa $84\%$. Sie verwechselt Oberteile mit ähnlichem Umriss.

<!--en:1bbec3be-->
- **ReLU-MLP mit 256 Einheiten:** Eine versteckte Schicht der Breite $h = 256$ erhöht die Kapazität auf $203{,}530$ Parameter ($26\times$ so viele). Die Zahl der Klassifikationsfehler sinkt um $\sim 20\%$ (Genauigkeit $\sim 87\%$).

<!--en:636ba8cb-->
- **Vier praktische Ziele:** Das MLP selbst und mit `Sequential` implementieren, Invarianten vor dem Training prüfen, die Geschwindigkeit auf der GPU messen und die erlernten Filter in $\mathbf{W}^{(1)}$ darstellen

<!--en:f012420d-->
::: {.slide title="Architektur und Parameteranzahl des MLP"}

<!--en:6ff7077b-->
[Architektur]{.kicker}

<!--en:9900729a-->
Jedes Bild der Größe $28 \times 28$ im Minibatch der Größe $n$ bildet eine Zeile von $\mathbf{X} \in \mathbb{R}^{n \times 784}$. Das zweischichtige MLP legt vier Parametertensoren an:

<!--en:8dc905f6-->
- **Versteckte Schicht ($d = 784 \to h = 256$):** $\mathbf{W}^{(1)} \in \mathbb{R}^{784 \times 256}$ ($200{,}704$ Gewichte) und $\mathbf{b}^{(1)} \in \mathbb{R}^{256}$, zusammen $98.7\%$ aller Parameter

<!--en:1f70d77a-->
- **Ausgabeschicht ($h = 256 \to q = 10$):** $\mathbf{W}^{(2)} \in \mathbb{R}^{256 \times 10}$ ($2{,}560$ Gewichte) und $\mathbf{b}^{(2)} \in \mathbb{R}^{10}$

<!--en:853981db-->
- **Insgesamt:** $785 \times 256 + 257 \times 10 = 203{,}530$ trainierbare Gleitkommazahlen ($\approx 0.81\textrm{ MB}$ im Format `float32`)

<!--en:90b8de83-->
::: {.slide title="Warum die Anfangsgewichte klein und zufällig sein müssen"}

<!--en:0d1326f7-->
[Invarianten der Initialisierung]{.kicker}

<!--en:f94f0700-->
- **Symmetriebrechung erfordert Zufall:** Bei identischen Anfangsspalten von $\mathbf{W}^{(1)}$ würde jede versteckte Einheit stets dieselbe Aktivierung berechnen und denselben Gradienten erhalten. Damit würden $256$ Einheiten wie $1$ einzige wirken.

<!--en:7d7c4c4d-->
- **Kleine Anfangsstreuung ($\sigma = 0.01$):** Die Logits liegen nahe bei $0$. Die Softmax-Funktion liefert anfangs nahezu gleiche Wahrscheinlichkeiten ($\hat{y}_j \approx 1/10$, Verlust $\approx \log 10 \approx 2.303$).

<!--en:48d04bd7-->
- **Neue Zufallswerte:** Die Gewichte für jede Schicht neu ziehen, damit sich die Zufallswerte nicht in jeder Schicht wiederholen

<!--en:c8136373-->
::: {.slide title="Parameter selbst anlegen und registrieren"}

<!--en:c8bf2621-->
[Eigene Implementierung]{.kicker}

<!--en:161ca92f-->
Jedes Array wird als Instanz der Parameterklasse des Frameworks registriert. So findet und aktualisiert der SGD-Optimierer von `Trainer` alle vier Tensoren automatisch:

<!--en:7ac04429-->
::: {.slide title="Elementweise ReLU und Vorwärtsdurchlauf in zwei Zeilen"}

<!--en:beb6cc59-->
[Vorwärtsdurchlauf]{.kicker}

<!--en:7deedf5f-->
- `relu(X)` nutzt die boolesche Maske in `X * (X > 0)`. Der Vorwärtsdurchlauf gibt die unnormalisierten Klassenlogits $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ zurück.

<!--en:b8c439bb-->
::: {.slide title="Drei Plausibilitätstests mit acht Bildern vor dem Training"}

<!--en:b4dc3bb5-->
[Prüfungen vor dem Training]{.kicker}

<!--en:d848e9ca-->
Vor einem mehrminütigen Training genügt ein Slice mit $8$ Bildern. Beim Durchlauf durch das *untrainierte* Netz lassen sich in Millisekunden drei Invarianten prüfen:

<!--en:94f91575-->
1. **Tensorformen:** `H` hat die Form `(8, 256)` und `O` die Form `(8, 10)`. Transponierte Gewichtsmatrizen oder fehlende Aufrufe von `Flatten` fallen sofort auf.

<!--en:5b7ca693-->
2. **Anfangsverlust nahe $\log 10 \approx 2.303$:** Das bestätigt, dass die Logits anfangs nahe bei $0$ liegen und der Kreuzentropieverlust unnormalisierte Logits statt Wahrscheinlichkeiten erhält.

<!--en:b680454d-->
3. **Anteil aktiver ReLU-Einheiten nahe $0.50$:** Bei Bias-Termen von null und symmetrisch normalverteilten Gewichten ist etwa die Hälfte aller Voraktivierungen $z_{ij}$ der versteckten Schicht in Schritt $0$ positiv.

<!--en:4112bdfc-->
::: {.slide title="Training von MLPScratch über 30 Epochen auf Fashion-MNIST"}

<!--en:15f39798-->
[Trainingskurve]{.kicker}

<!--en:f7604670-->
Training von `MLPScratch` über $30$ Epochen mit der Minibatchgröße $n = 256$ und der Lernrate $\eta = 0.1$:

<!--en:a081ee93-->
- **Gleichmäßige Konvergenz:** Der Trainingsverlust sinkt stetig von $2.30$ auf unter $0.35$. Die Trainingsgenauigkeit steigt auf $87\text{--}90\%$.

<!--en:8509005b-->
- **Plateau bei der Validierung:** Die Genauigkeit auf den zurückgehaltenen Daten erreicht $\sim 87\%$ (gegenüber $\sim 84\%$ bei der Softmax-Regression). In einzelnen Epochen kann sie einige Prozentpunkte tiefer liegen.

<!--en:dfe96a73-->
::: {.slide title="Kompakte Implementierung mit Sequential und den Schichten des Frameworks"}

<!--en:05952cca-->
[High-Level-API]{.kicker}

<!--en:a648cfa2-->
Mit dem `Sequential`-Container und den eingebauten linearen Schichten und ReLU-Schichten des jeweiligen Frameworks reichen wenige Zeilen für dieselbe Architektur. Parameterarrays von Hand anzulegen erübrigt sich:

<!--en:e880be00-->
::: {.slide title="Anpassung der Standardinitialisierung für Gewichte"}

<!--en:bab693bc-->
[Standardwerte und explizite Initialisierung]{.kicker}

<!--en:e393d77f-->
- **Standardwerte:** Die vollvernetzte Schicht des Frameworks zieht die Gewichte aus einer Gleich- oder Normalverteilung. Die Anzahl der Eingänge bestimmt die Skalierung, hier mit $\sigma \approx 0.02$–$0.04$.

<!--en:753ad2cb-->
- **Kontrollierter Vergleich:** Mit Gewichten aus $\mathcal{N}(0, 0.01^2)$ und Bias-Termen von null ergibt sich für `MLP` der gleiche Trainingsverlauf wie für `MLPScratch`.

<!--en:f23912b5-->
- **Ausblick auf die numerische Stabilität:** Für tiefe Netze mit ReLU-Schichten tritt an die Stelle von $\sigma = 0.01$ die varianzerhaltende He-Initialisierung ($\sigma = \sqrt{2/n_{\textrm{in}}}$).

<!--en:a446d072-->
::: {.slide title="Genauigkeit von MLP und Softmax-Regression im direkten Vergleich"}

<!--en:343fa0a9-->
[Empirischer Vergleich]{.kicker}

<!--en:d4082825-->
Training beider MLPs und der Softmax-Regression über $30$ Epochen mit identischen Einstellungen:

<!--en:1bd743af-->
- **Rückstand in Epoche 1 durch $\sigma = 0.01$:** In Epoche 1 liegen beide MLPs hinter der Softmax-Regression (Trainingsverlust $\sim 1.04$ gegenüber $\sim 0.79$). Bei der Backpropagation werden die Gradienten für $\mathbf{W}^{(1)}$ mit $\mathbf{W}^{(2)} \sim 0.01$ multipliziert, bis die Gewichte in $\mathbf{W}^{(2)}$ wachsen.

<!--en:5898dc8a-->
- **$20\%$ weniger Fehler bis Epoche 30:** Die Validierungsgenauigkeit beider MLPs nähert sich $\sim 87\%$ gegenüber $\sim 84\%$ bei der Softmax-Regression. Damit entfällt fast ein Fünftel aller Klassifikationsfehler.

<!--en:755fdb8f-->
::: {.slide title="Mittlere Validierungsgenauigkeit über die letzten fünf Epochen"}

<!--en:f2cb0bac-->
[Ergebnisse in Zahlen]{.kicker}

<!--en:8609efb3-->
Der Mittelwert der Validierungsgenauigkeit über die Epochen 26–30 glättet die SGD-bedingten Schwankungen von Epoche zu Epoche. Er bestätigt die Übereinstimmung der eigenen und der kompakten Implementierung:

<!--en:d58a836d-->
Bei $87\text{--}90\%$ Trainingsgenauigkeit und $\sim 87\%$ Validierungsgenauigkeit zeigt das auf $60{,}000$ Bildern trainierte MLP mit $203{,}530$ Parametern nach $30$ Epochen nur eine kleine Generalisierungslücke von wenigen Prozentpunkten.

<!--en:ee96356b-->
::: {.slide title="26× so viele FLOPs bei fast gleicher Laufzeit pro Epoche auf der GPU"}

<!--en:4a0103de-->
[Rechenleistung]{.kicker}

<!--en:d2f9f247-->
Der Vergleich von Parameteranzahl, Multiplikationsadditionen pro Bild und gemessener Laufzeit in Sekunden pro Epoche zeigt, welche Rolle die Hardware spielt:

<!--en:243d187d-->
| Modell | Parameter | Multiplikationsadditionen pro Bild | Validierungsgenauigkeit (Epochen 26–30) |

<!--en:681f2ffb-->
| Softmax-Regression ($784 \to 10$) | $7{,}850$ | $7{,}840$ | $\sim 84\%$ |

<!--en:eabb8e0e-->
| MLP ($784 \to 256 \to 10$) | $203{,}530$ | $203{,}264$ | $\sim 87\%$ |

<!--en:9487d631-->
Warum kostet der $26\times$ so hohe Aufwand für Matrixoperationen auf einer GPU kaum zusätzliche Zeit? Bei einer Breite von $h = 256$ bestimmen das Laden der Minibatches, die Python-Schleife und die Latenz beim CUDA-Kernelstart die Laufzeit. Die Tensor-Kerne der GPU berechnen das Matrixprodukt $(256 \times 784) \times (784 \times 256)$ in Mikrosekunden.

<!--en:34a3772b-->
::: {.slide title="Erlernte Merkmale: Kontrastmuster"}

<!--en:58de2e79-->
[Gewichte der ersten Schicht sichtbar machen]{.kicker}

<!--en:3c680729-->
Bei der Softmax-Regression dienten die $10$ Gewichtsspalten als unscharfe Schablonen für ganze Kleidungsstücke. Die $10$ Spalten von $\mathbf{W}^{(1)} \in \mathbb{R}^{784 \times 256}$ mit der größten Norm lassen sich zu Bildern der Größe $28 \times 28$ umformen. Diese zeigen, wie die versteckten Einheiten zusammenwirken:

<!--en:a12268b7-->
- **Kontrastmuster:** Rote (positive, erregende) und benachbarte blaue (negative, hemmende) Bereiche lassen eine Einheit auf helle Stellen neben dunklen reagieren, etwa entlang der Schultern oder seitlich am Oberkörper.

<!--en:4e16e066-->
- **Kombination in der Ausgabeschicht:** Viele versteckte Einheiten mit großer Gewichtsnorm tragen zu den Scores der vier ähnlichen Oberteile bei (Hemd, T-Shirt, Pullover, Mantel). Die Ausgabeschicht kombiniert $256$ solcher Kontraste zu ihren zehn Scores.

<!--en:6ffe8aba-->
::: {.slide title="Verständnisfragen zu Prüfungen vor dem Training und zur Skalierung"}

<!--en:ffcb46a8-->
[Verständnisfragen]{.kicker}

<!--en:d13d104a-->
Fragen zur Implementierung und Diagnose eines MLP:

<!--en:51a93b64-->
1. **Fehlerhafte Initialisierung:** Die Prüfung vor dem Training mit $8$ Bildern ergibt einen anfänglichen Kreuzentropieverlust von $14.8$ statt $2.30$ und einen Anteil aktiver ReLU-Einheiten von $0.50$. Welcher Fehler liegt als Erstes nahe?

<!--en:e2b4ac5e-->
2. **Inaktive Schicht:** Was bedeutet ein Anfangsverlust von $2.30$, wenn der Anteil aktiver ReLU-Einheiten im gesamten Batch bei $0.00$ liegt?

<!--en:551ab74c-->
3. **Skalierung auf $h = 1{,}024$:** Wie viele Parameter und wie viele Multiplikationsadditionen pro Bild hat das MLP für Fashion-MNIST, wenn die Breite der versteckten Schicht von $256$ auf $1{,}024$ Einheiten steigt?

<!--en:557b98b6-->
**Lösungen:**

<!--en:289757b2-->
1. **Die Anfangsstreuung $\sigma$ der Gewichte ist zu groß** (oder die Pixel wurden unskaliert als `uint8` im Bereich $[0, 255]$ statt als Gleitkommazahlen im Bereich $[0, 1]$ übergeben). Die anfänglichen Logits sind dadurch extrem groß und die wahre Klasse erhält eine Wahrscheinlichkeit nahe null.

<!--en:643e8e0f-->
2. **Negative anfängliche Bias-Terme** (z. B. $\mathbf{b}^{(1)}$ mit $-5$ statt $0$ initialisiert) drücken jede Voraktivierung unter null ($z_{ij} < 0$). Damit sind alle ReLU-Einheiten ab Schritt $0$ inaktiv.

<!--en:db4085ec-->
3. **Skalierung um den Faktor 4:** $785 \times 1024 + 1025 \times 10 = 814{,}090$ Parameter und $784 \times 1024 + 1024 \times 10 = 813{,}056$ Multiplikationsadditionen ($4\times$ so viele wie beim MLP mit 256 Einheiten, $\sim 104\times$ so viele wie bei der Softmax-Regression)

<!--en:ab75e325-->
::: {.slide title="Zusammenfassung: Implementierung mehrschichtiger Perzeptrons"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:e69a50c4-->
- **Eigene und kompakte Implementierung:** Die registrierten Parameter $\mathbf{W}^{(k)} \sim \mathcal{N}(0, 0.01^2)$ und $\mathbf{b}^{(k)} = \mathbf{0}$ brechen durch ihre Initialisierung die Symmetrie der versteckten Einheiten. Die Architektur entspricht `Sequential(Flatten, Linear(256), ReLU, Linear(10))`.

<!--en:dbf6d464-->
- **Prüfung vor dem Training:** Mit `(8, 256)` und `(8, 10)` als erwarteten Formen, $\text{loss} \approx \log 10 \approx 2.303$ und $50\%$ aktiven ReLU-Einheiten bei $8$ Bildern lassen sich Fehler in $<1\textrm{ ms}$ erkennen.

<!--en:3d3977d0-->
- **Geschwindigkeit und Merkmale:** Bei $26\times$ so vielen FLOPs braucht die GPU pro Epoche $\sim 1\times$ so lange wie bei der Softmax-Regression. Die in $\mathbf{W}^{(1)}$ erlernten Kontrastmuster erhöhen die Genauigkeit von $84\%$ auf $87\%$.
