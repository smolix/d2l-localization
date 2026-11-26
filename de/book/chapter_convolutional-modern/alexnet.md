<!--en:166ede32-->
# AlexNet und gelernte Bildrepräsentationen

<!--en:9aae5d5b-->
In :numref:`sec_lenet` haben wir LeNet durchgängig anhand roher Pixel trainiert, von den Faltungsfiltern bis zum abschließenden Klassifikator. Dennoch trennte die Computer Vision noch zwei Jahrzehnte nach LeCuns Ergebnissen von 1989 meist die Merkmalsextraktion (engl. feature extraction) von der Klassifikation. Von Hand entworfene Deskriptoren wie SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, HOG :cite:`Dalal.Triggs.2005` und der Bag-of-Visual-Words-Ansatz :cite:`Sivic.Zisserman.2003` bildeten Pixel auf feste Merkmalsvektoren ab. Diese dienten einem linearen oder kernelbasierten Klassifikator als Eingabe. Da sich beim Minimieren des Trainingsverlusts nur der Klassifikator änderte, hing der Fortschritt davon ab, welche Merkmale Menschen entwarfen.

<!--en:40e28344-->
Die Wende kam beim ImageNet-Wettbewerb 2012 mit 1000 Objektkategorien. Als Maß diente der *Top-5-Fehler* (engl. top-5 error), also der Anteil der Validierungsbilder, deren wahres Label unter den fünf wahrscheinlichsten Vorhersagen des Modells fehlt. Bei *AlexNet* :cite:`Krizhevsky.Sutskever.Hinton.2012` wurde ein konvolutionäres Netz mit acht Schichten durchgängig anhand roher Pixel trainiert. Der Wettbewerbsbeitrag, ein Ensemble aus sieben solchen Netzen, senkte den Top-5-Fehler auf einen Schlag von 26.2% (beste von Hand entworfene Pipeline) auf 15.3%. Ein einzelnes Netz erreichte bereits rund 18%. Im Aufbau ist AlexNet eine vergrößerte Version von LeNet. Der plötzliche Erfolg lässt sich auf drei Entwicklungen zurückführen, die 2012 zusammenkamen. ImageNet lieferte einen Datensatz mit einer Million Bildern mit Labels. GPUs konnten mehrere zehn Millionen Gewichte in Tagen statt Monaten trainieren. ReLU-Aktivierungen, Dropout und Datenaugmentierung (engl. data augmentation) ermöglichten das Training tiefer Netze und wirkten dem Auswendiglernen entgegen.

<!--en:b4c8f338-->
## Gelernte Merkmale, Daten und Hardware

<!--en:fd231cf1-->
Was fand die Backpropagation auf ImageNet, das von Hand entworfene Deskriptoren übersahen? :numref:`fig_filters` zeigt die 96 gelernten $11 \times 11$-Faltungskerne (engl. kernels) der ersten Faltungsschicht von AlexNet. Einige erkennen Kanten und Frequenzmuster bestimmter Orientierung, andere reagieren auf Flecken in Gegenfarben. Diese Filter entstehen allein durch Minimierung des Klassifikationsverlusts. Tiefere Schichten kombinieren ihre Antworten zu Texturen, Objektteilen und Objektkategorien.

<!--en:32ac17c4-->
![Die 96 auf ImageNet gelernten Faltungskerne der ersten Faltungsschicht von AlexNet. Abdruck mit freundlicher Genehmigung von :citet:`Krizhevsky.Sutskever.Hinton.2012`.](../img/filters.png)

<!--en:983bf7f9-->
### Daten und Hardware

<!--en:dd27076e-->
Modelle mit 60 Millionen Parametern lernen kleine Benchmarks wie PASCAL VOC (rund $10^4$ Bilder) schnell auswendig. Der Trainingsdatensatz von ImageNet für 2012 :cite:`Deng.Dong.Socher.ea.2009,russakovsky2015imagenet` umfasste 1.2 Millionen Fotos mit Labels in 1000 Kategorien. Durch Datenaugmentierung vervielfachte AlexNet die Vielfalt der Trainingsbeispiele. Aus $224 \times 224$ Pixel großen Ausschnitten, die zufällig aus Bildern der Größe $256 \times 256$ gewählt und horizontal gespiegelt wurden, ergaben sich bis zu $32 \cdot 32 \cdot 2 = 2{,}048$ Ansichten pro Foto.

<!--en:9cc6347b-->
Den zweiten Teil des Durchbruchs ermöglichte die Hardware. Wer einen Filter über alle räumlichen Positionen und Kanäle faltet, führt Millionen unabhängiger Multiplikationsadditionen (engl. multiply-adds, MACs) aus. Genau dafür sind Grafikprozessoren (GPUs) mit Hunderten paralleler Kerne und hoher Speicherbandbreite ausgelegt. Jede der beiden für AlexNet verwendeten NVIDIA GTX 580 lieferte rund 1.6 TFLOPS und 192 GB/s Speicherbandbreite. Eine damalige Vierkern-CPU erreichte rund 0.2 TFLOPS und 20 GB/s. Ein Vorwärtsdurchlauf (engl. forward pass) von AlexNet für ein RGB-Bild der Größe $224 \times 224$ kostet rund $10^9$ Multiplikationsadditionen. Veranschlagen wir für den Rückwärtsdurchlauf das Doppelte und pro Multiplikationsaddition 2 FLOPs, so erfordern 90 Epochen mit je 1.2 Millionen Bildern

<!--en:9993a536-->
– mit den eigens von Krizhevsky entwickelten CUDA-Kerneln ([cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)) entspricht das fünf bis sechs Tagen auf zwei GTX 580, gegenüber deutlich über einem Monat auf einer CPU.

<!--en:d8a63500-->
## Die Architektur von AlexNet

<!--en:be22b458-->
:numref:`fig_alexnet` stellt LeNet (:numref:`sec_lenet`) der AlexNet-Variante gegenüber, die wir weiter unten auf Fashion-MNIST trainieren. Beide folgen demselben Grundaufbau. Auf mehrere Faltungsstufen folgt ein Klassifikationskopf aus vollständig verbundenen Schichten (engl. fully connected layers). AlexNet vergrößert jedoch alle Dimensionen und modernisiert drei wesentliche Komponenten, nämlich die Aktivierungsfunktion, das Pooling und die Regularisierung.

<!--en:e76f9d90-->
![LeNet aus :numref:`sec_lenet` (links) und AlexNet aus diesem Abschnitt (rechts), beide für einkanalige Fashion-MNIST-Bilder und zehn Klassen. Die Eingabe steht unten. AlexNet ist tiefer und breiter, verwendet ReLU-Aktivierungen und Max-Pooling und regularisiert seine vollständig verbundenen Schichten mit Dropout.](../img/arch-alexnet.svg)

<!--en:38ff34c9-->
### Tiefe, Breite, ReLU und Dropout

<!--en:01994f99-->
AlexNet erweitert die fünf Schichten mit Parametern von LeNet auf acht, nämlich fünf Faltungsschichten und drei vollständig verbundene Schichten (:numref:`fig_alexnet`). Da Eingaben der Größe $224 \times 224$ achtmal so breit sind wie Miniaturbilder der Größe $28 \times 28$, verwendet Schicht 1 einen $11 \times 11$-Faltungskern mit Stride 4. Schicht 2 verwendet einen $5 \times 5$-Faltungskern, die Schichten 3–5 verwenden $3 \times 3$-Faltungskerne. Nach der ersten, zweiten und fünften Faltung folgt jeweils überlappendes $3 \times 3$-Max-Pooling mit Stride 2. Die Kanalzahl steigt von 96 auf 384 (gegenüber 6 und 16 bei LeNet). Die beiden versteckten vollständig verbundenen Schichten umfassen je 4096 Einheiten (gegenüber 120 und 84). Unsere Implementierung verzichtet auf zwei Besonderheiten aus dem Jahr 2012, die Aufteilung der Schichten auf zwei GPUs mit je 3 GB und die lokale Antwortnormalisierung. Letztere wurde durch die Batchnormalisierung aus :numref:`sec_batch_norm` abgelöst.

<!--en:58e3b4f0-->
Drei Entscheidungen tragen dazu bei, dass dieses größere Netz gut funktioniert. Die ersten beiden stammen aus dem ursprünglichen AlexNet, die dritte führen wir hier ein.

<!--en:1f8fbf30-->
1. **ReLU-Aktivierungen:** Bei sättigenden Sigmoidfunktionen schwächt die Ableitung $\le 1/4$ die Gradienten über viele Schichten um den Faktor $4^{-K}$ ab. Mit $\textrm{ReLU}(x) = \max(x, 0)$ beträgt die Ableitung bei aktiven Einheiten dagegen 1. So erreicht das Netz auf CIFAR-10 einen Trainingsfehler von 25% sechsmal so schnell wie mit $\tanh$ :cite:`Krizhevsky.Sutskever.Hinton.2012`.

<!--en:acd2e8d2-->
2. **Dropout:** Nach jeder der beiden vollständig verbundenen Schichten mit je 4096 Einheiten verhindert Dropout ($p = 0.5$, :numref:`sec_dropout`), dass der Klassifikationskopf den Trainingsdatensatz auswendig lernt.

<!--en:8ce5e2f1-->
3. **He-Initialisierung:** AlexNet zog seine Gewichte aus einer Gauß-Verteilung mit der festen Standardabweichung 0.01. Wir verwenden stattdessen die Initialisierung, die :citet:`He.Zhang.Ren.ea.2015` drei Jahre später für ReLU-Netze herleiteten. Weil ReLU rund die Hälfte der Eingaben auf null setzt, ziehen wir die Gewichte aus $\mathcal{N}(0, 2/n_{\textrm{in}})$ (:numref:`subsec_he_init`, bei Faltungen mit $n_{\textrm{in}} = c_{\textrm{i}} k^2$) und initialisieren die Bias-Terme mit null.

<!--en:04ec5e6e-->
`init.MSRAPrelu('in', 0)` zieht normalverteilte Gewichte mit Varianz $2/n_\textrm{in}$. Das entspricht der He-Initialisierung für eine ReLU (eine PReLU mit Steigung 0). Das Netz ist ein `HybridSequential`. Mit `hybridize()` wird es zu einem einzigen Graphen kompiliert.

<!--en:ef1062f1-->
Die verzögert initialisierten Schichten haben erst dann Gewichte, wenn der erste Minibatch deren Formen festlegt. Wie bei LeNet ruft der `Trainer` anschließend die Methode `weight_init` des Modells für jede Schicht auf.

<!--en:8a38b88f-->
Jede Schicht erhält einen eigenen Initialisierer, da ein Keras-Initialisierer ohne expliziten Startwert für den Zufallsgenerator bei jedem Aufruf dieselben Werte zurückgibt.

<!--en:b97d36cf-->
NNX-Schichten ziehen ihre Gewichte im Konstruktor. Daher erhält jede Schicht den Initialisierer.

<!--en:b95a4310-->
Um zu prüfen, ob unser Code :numref:`fig_alexnet` entspricht, übergeben wir ein einkanaliges Testbild der Größe $224 \times 224$ an `layer_summary` und betrachten die Form jedes Ausgabetensors.

<!--en:caa71645-->
Alle Dimensionen stimmen mit den Formeln aus :numref:`sec_padding` überein. Die erste Faltung mit Stride 4 verkleinert die Seitenlänge von 224 auf $\lfloor (224 - 11)/4 \rfloor + 1 = 54$. Jede überlappende $3 \times 3$-Max-Pooling-Schicht mit Stride 2 bildet $n$ auf $\lfloor (n - 3)/2 \rfloor + 1$ ab ($54 \to 26 \to 12 \to 5$). Die Faltungen dazwischen erhalten dank Padding die räumlichen Dimensionen. Die Ausgabe der letzten Faltungsstufe lässt sich daher zu einem Vektor mit $256 \times 5 \times 5 = 6400$ Elementen für den vollständig verbundenen Klassifikationskopf umformen.

<!--en:51e9e5ff-->
### Parameter und Multiplikationsadditionen

<!--en:5cfc6b21-->
Nachdem wir die Formen der Schichten geprüft haben, können wir wie bei LeNet (:numref:`sec_lenet`) die Parameter zählen und den Rechenaufwand bestimmen. Eine Faltungsschicht verwendet jedes ihrer $c_\textrm{i} c_\textrm{o} k^2$ Gewichte an jeder räumlichen Ausgabeposition erneut. Eine vollständig verbundene Schicht multipliziert jedes Gewicht nur einmal pro Beispiel. :numref:`tab_alexnet_cost` fasst die Ergebnisse für unsere AlexNet-Variante mit einem Eingabekanal und zehn Klassen zusammen.

<!--en:565b1a22-->
:Parameter und Multiplikationsadditionen der AlexNet-Schichten pro Beispiel für ein einkanaliges $224 \times 224$-Bild und zehn Klassen. Pooling-, Aktivierungs- und Dropout-Schichten haben keine Parameter.

<!--en:ca5784dc-->
| Schicht | Ausgabe | Parameter | Multiplikationsadditionen |

<!--en:26751744-->
| $11\times11$-Faltung, 1 auf 96, Stride 4 | 96@54×54 | $11{,}712$ | $33.9$ Mio. |

<!--en:be50f70f-->
| $5\times5$-Faltung, 96 auf 256 | 256@26×26 | $614{,}656$ | $415.3$ Mio. |

<!--en:d3af0869-->
| $3\times3$-Faltung, 256 auf 384 | 384@12×12 | $885{,}120$ | $127.4$ Mio. |

<!--en:011e42b9-->
| $3\times3$-Faltung, 384 auf 384 | 384@12×12 | $1{,}327{,}488$ | $191.1$ Mio. |

<!--en:265c9f4b-->
| $3\times3$-Faltung, 384 auf 256 | 256@12×12 | $884{,}992$ | $127.4$ Mio. |

<!--en:73625dfb-->
| vollständig verbunden, 6400 auf 4096 | 4096 | $26{,}218{,}496$ | $26.2$ Mio. |

<!--en:9ee71496-->
| vollständig verbunden, 4096 auf 4096 | 4096 | $16{,}781{,}312$ | $16.8$ Mio. |

<!--en:6b2a11c6-->
| vollständig verbunden, 4096 auf 10 | 10 | $40{,}970$ | $0.04$ Mio. |

<!--en:651e3113-->
| Gesamt | | $46{,}764{,}746$ | $938.1$ Mio. |

<!--en:942fa9c7-->
Parameter und Rechenaufwand sind noch ungleicher verteilt als bei LeNet. Die fünf Faltungsschichten enthalten nur 8% der 47 Millionen Parameter von AlexNet, führen aber 95% der Multiplikationsadditionen aus. Auf die drei vollständig verbundenen Schichten entfallen 92% der Parameter (rund 170 MB bei 32-Bit-Gleitkommadarstellung), aber nur 5% der Rechenoperationen. LeNet hat 61 706 Parameter und benötigt 416 520 Multiplikationsadditionen pro Bild. Bei AlexNet steigt die Parameterzahl um den Faktor $760\times$ und der Rechenaufwand um den Faktor $2{,}250\times$, jeweils näherungsweise. Auf ImageNet (3 RGB-Kanäle und 1000 Klassen) erreicht das vollständige Modell rund 60 Millionen Parameter. Der weitaus größte Teil entfällt weiterhin auf dieselben vollständig verbundenen Schichten.

<!--en:f22fafe5-->
## Training auf Fashion-MNIST

<!--en:b739f70b-->
Das Training von AlexNet auf ImageNet dauert auf einer modernen GPU Stunden. Es erfordert einen Datensatz von rund 150 GB und eine schnelle Eingabepipeline, die der GPU laufend Daten liefert. Damit wir interaktiv experimentieren können, trainieren wir das Netz auf Fashion-MNIST. Die ursprünglichen $28 \times 28$-Bilder von Fashion-MNIST sind für eine erste $11 \times 11$-Schicht mit Stride 4 und drei anschließende Pooling-Stufen zu klein. Deshalb vergrößern wir sie durch Upsampling auf $224 \times 224$. Dazu verwenden wir das Argument `resize` von `d2l.FashionMNIST`. In realen Anwendungen liefert eine 64-mal so große Pixelzahl keine neue Bildinformation, vervielfacht aber den Rechenaufwand. Hier nehmen wir diesen Aufwand allein deshalb in Kauf, damit wir AlexNet in seiner ursprünglichen Architektur ausführen können.

<!--en:1e998ac0-->
Wir trainieren mit stochastischem Gradientenabstieg auf Minibatches (engl. stochastic gradient descent, SGD) bei einer Batchgröße von 128 für 10 Epochen, ein Drittel der 30 Epochen von LeNet. Dank der ReLU-Aktivierungen fließen die Gradienten ohne die für Sigmoidfunktionen typische Abschwächung. Daher genügt eine wesentlich kleinere Lernrate als der Wert 2.0 bei LeNet.

<!--en:be465fee-->
### Lernratenplan

<!--en:9a3afc3c-->
Eine feste Lernrate ist selten sowohl zu Beginn als auch am Ende des Trainings optimal. Anfangs sind die Gewichte zufällig, und ein zu großer Schritt kann die Optimierung destabilisieren. Gegen Ende springen die Parameter bei zu großen Schritten um ein Minimum herum, ohne sich ihm weiter zu nähern. Die Autoren von AlexNet senkten die Lernrate von Hand. Sobald der Validierungsfehler stagnierte, teilten sie die Lernrate durch 10. Wir verwenden einen Lernratenplan (engl. learning rate schedule) mit glattem Verlauf, der ohne Eingriffe auskommt und sich inzwischen als Standard etabliert hat. Damit sich die Modelle gut vergleichen lassen, verwenden wir im restlichen Kapitel genau denselben Plan. In einer linearen *Aufwärmphase* (engl. warmup) steigt die Lernrate von 0 auf ihren Höchstwert $\eta_0$. Das dauert die ersten $t_\textrm{w}$ Schritte. Danach folgt die Lernrate einer halben Kosinusperiode und fällt bis zum letzten Schritt $T$ auf 0:

<!--en:b960d7fa-->
Bei $t = t_\textrm{w}$ ist der Kosinusterm gleich 1 und es gilt $\eta_t = \eta_0$. Nach der Hälfte der Abklingphase beträgt die Lernrate $\eta_0/2$, bei $t = T$ erreicht sie 0. Durch den allmählichen Anstieg können wir einen höheren Spitzenwert wählen, als ab Schritt 1 sicher wäre. Hier trainieren wir AlexNet mit einer maximalen Lernrate von $\eta_0 = 0.03$. Auch für jede Architektur in :numref:`sec_blocks` wählen wir den höchsten Spitzenwert, bei der das Training zuverlässig gelingt. Die folgende Funktion `cosine_warmup` berechnet den multiplikativen Faktor für die Lernrate im Schritt $t$. Wir speichern sie in `d2l`, um sie im gesamten Kapitel zu verwenden. Lernratenpläne untersuchen wir systematisch in :numref:`sec_scheduler`. :numref:`fig_cosine_warmup` zeigt den Verlauf der Lernrate.

<!--en:a3913f2f-->
![Lernratenplan für das gesamte Kapitel. In der linearen Aufwärmphase steigt die Lernrate von 0 auf ihren Höchstwert $\eta_0$. Diese Phase umfasst die ersten $t_\textrm{w}$ Schritte, hier ein Zehntel des Trainings. Danach senkt die halbe Kosinusperiode aus :eqref:`eq_cosine_warmup` die Lernrate bis zum letzten Schritt $T$ auf 0. Auf halbem Weg beträgt sie $\eta_0/2$.](../img/arch-cosine-warmup.svg)

<!--en:5a96159f-->
`AlexNet` aktualisiert die Lernrate nach jedem Minibatch gemäß diesem Plan. `num_steps` gibt die Gesamtzahl $T$ der Schritte im Trainingslauf an. Die Aufwärmphase umfasst das erste Zehntel des Trainings, also eine von zehn Epochen.

<!--en:6f605338-->
### Vergleich mit LeNet

<!--en:672d7973-->
Welches Ergebnis können wir von der Trainingsschleife erwarten? Fashion-MNIST enthält zu wenige Bilder und zu wenig visuelle Vielfalt, um die 47 Millionen Parameter von AlexNet auszuschöpfen. Gegenüber LeNet erwarten wir deshalb nur einen geringen Gewinn an Genauigkeit. Hier wollen wir die Implementierung prüfen, den Rechenaufwand der Schichten untersuchen und herausfinden, was die erste Schicht lernt. Dazu speichern wir vor dem Training die 96 Faltungskerne der ersten Schicht. So können wir genau messen, wie die Optimierung sie verändert.

<!--en:62f70a0f-->
Die anfänglichen $11 \times 11$-Faltungskerne sind in `K0` gespeichert. Nun trainieren wir `AlexNet` für 10 Epochen auf dem Fashion-MNIST-Datensatz `data` mit den vergrößerten Bildern.

<!--en:ec13376c-->
Nach 10 Epochen erreicht AlexNet rund 91% Validierungsgenauigkeit. Der Trainingsverlust liegt bei etwa 0.22, der Validierungsverlust bei etwa 0.25. Durch den Kosinus-Abfall nähert sich die Lernrate in den letzten Epochen null. Am Ende schwankt die Genauigkeit um weniger als einen Prozentpunkt. Die letzte Epoche ist daher für den Trainingslauf aussagekräftig. LeNet erreicht nach 30 Epochen rund 88–90%. AlexNet liegt mit einem Drittel der Epochen um einen oder zwei Prozentpunkte vorn, benötigt dafür aber $2{,}250\times$ so viele Rechenoperationen pro Bild. Selbst dieser bescheidene Gewinn ist den ReLU-Aktivierungen, der He-Initialisierung und dem Lernratenplan ebenso zuzuschreiben wie der größeren Modellkapazität.

<!--en:041edcb8-->
### Was die erste Schicht lernt

<!--en:438c0f44-->
Die $5 \times 5$-Faltungskerne von LeNet waren zu klein, um ihre Muster mit dem Auge zu erkennen (:numref:`sec_lenet`). Die 96 Faltungskerne von AlexNet mit ihrer Größe von $11 \times 11$ lassen sich dagegen direkt als kleine Bilder interpretieren, wie wir für ImageNet in :numref:`fig_filters` gesehen haben. Um zu sehen, was unser Modell auf Fashion-MNIST gelernt hat, ziehen wir die zufälligen Anfangsgewichte `K0` von den trainierten Faltungskernen der ersten Schicht ab. Für jeden Faltungskern stellen wir die Gewichtsdifferenzen auf einer eigenen Graustufenskala dar.

<!--en:54055dc7-->
In zehn Epochen ändern sich die Gewichte der ersten Schicht nur um rund 4% ihres anfänglichen Betrags. Bilder der trainierten Faltungskerne allein würden daher immer noch weitgehend wie das anfängliche Rauschen aussehen. Die *Gewichtsänderungen* selbst zeigen dagegen deutliche Strukturen. Sanfte Helligkeitsverläufe und breite, ausgerichtete Balken sind die niederfrequenten Gegenstücke zu den scharf konturierten Kantenfiltern für ImageNet in :numref:`fig_filters`. Dass feine, hochfrequente Kanten und Farbflecken fehlen, ergibt sich unmittelbar aus den Eingabedaten. Unsere Bilder enthalten nur Graustufen. Nach der achtfachen Vergrößerung sind die feinsten echten Details etwa acht Pixel breit. Ein $11 \times 11$-Faltungskern sieht hier nur einen sanften Helligkeitsverlauf. Die erste Schicht passt sich also genau an die räumliche Statistik ihres Datensatzes an.

<!--en:22605a92-->
Wie :numref:`tab_alexnet_cost` gezeigt hat, liegen 92% der Parameter von AlexNet im großen vollständig verbundenen Klassifikationskopf. Er trägt wenig zum Rechenaufwand bei, erfordert aber starkes Dropout, um das Auswendiglernen einzudämmen. Im nächsten Abschnitt untersuchen wir, wie sich tiefe Architekturen systematisch aus wiederverwendbaren Blöcken aufbauen lassen und ob ein so großer Klassifikationskopf überhaupt nötig ist.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:661ef86f-->
Im Jahr 2012 übertrafen durchgängig gelernte Bildrepräsentationen die Verfahren mit von Hand entworfenen Merkmalen. Dafür kamen drei Faktoren zusammen: große Datensätze mit Labels (ImageNet), massiv parallele GPUs, die mehrere zehn Millionen Parameter in wenigen Tagen trainieren konnten, und moderne Trainingsverfahren (ReLU-Aktivierungen, Dropout und Datenaugmentierung). Mit diesen Mitteln übertrug AlexNet den Grundaufbau von LeNet auf ein größeres Netz. Der Wettbewerbsbeitrag senkte den Top-5-Fehler auf ImageNet von 26.2% auf 15.3%. Die Bilanz der Schichten zeigt eine ungleiche Verteilung. Auf die Faltungsschichten entfallen 95% der Rechenoperationen, auf den vollständig verbundenen Kopf 92% der Parameter. Fashion-MNIST schöpft diese große Kapazität nicht aus. Hier übertrifft AlexNet mit einem Drittel der Epochen die Genauigkeit von LeNet um ein bis zwei Prozentpunkte, benötigt aber rund $2{,}250\times$ so viele Rechenoperationen pro Bild.

<!--en:2294243a-->
## Übungen

<!--en:4665b383-->
1. **Speicherbedarf und Rechenaufwand.** Ergänzen Sie :numref:`tab_alexnet_cost` um den Speicherbedarf von `AlexNet` bei 32-Bit-Gleitkommadarstellung.
    1. Berechnen Sie für jede Schicht, wie viele Aktivierungen der Vorwärtsdurchlauf für ein einzelnes $224 \times 224$-Bild speichert. Zählen Sie dazu die Ausgaben jeder Faltungsschicht, jeder Pooling-Schicht und jeder vollständig verbundenen Schicht. Welche Schichten benötigen den meisten Aktivierungsspeicher, welche den meisten Speicher für Parameter?
    1. Beim Training müssen die Parameter, ihre Gradienten und die gespeicherten Aktivierungen eines ganzen Minibatchs in den Speicher passen. Ab welcher Batchgröße übersteigt der Aktivierungsspeicher den Speicherbedarf für Parameter und Gradienten zusammen?
    1. Zählen Sie wie in :numref:`tab_alexnet_cost` die Parameter für die ImageNet-Version (3 Eingabekanäle, 1000 Klassen, Eingaben der Größe $227 \times 227$, sodass die letzte Faltungsstufe die räumlichen Dimensionen $6 \times 6$ statt $5 \times 5$ hat). Vergleichen Sie Ihre Summe mit den 60 Millionen Parametern, die :citet:`Krizhevsky.Sutskever.Hinton.2012` angeben.
   :exlabel:`ex_alexnet-memory-compute-ledger`

<!--en:41667d5b-->
1. **Rezeptive Felder der frühen Schichten.** In einem Stapel von Schichten mit Kerngrößen $k_i$ und Strides $s_i$ hat eine Einheit der obersten Schicht ein rezeptives Feld der Seitenlänge $r = 1 + \sum_{i} (k_i - 1) \prod_{j<i} s_j$ (:numref:`sec_padding`). Berechnen Sie das rezeptive Feld einer Einheit nach jeder der fünf Faltungsschichten von `AlexNet`. Berücksichtigen Sie dabei für jede Max-Pooling-Schicht deren eigene Kerngröße und deren Stride. Welchen Anteil der Seitenlänge der $224 \times 224$-Eingabe deckt das jeweilige rezeptive Feld ab? Gibt es eine Faltungsschicht, in der eine Einheit das gesamte Bild sieht?

<!--en:9479cb04-->
1. **Arithmetische Intensität.** ● Die *arithmetische Intensität* einer Schicht gibt an, wie viele Multiplikationsadditionen sie pro Byte an Parametern ausführt, das sie aus dem Speicher liest. Verwenden Sie :numref:`tab_alexnet_cost` für die folgenden Teilaufgaben.
    1. Berechnen Sie die arithmetische Intensität jeder Schicht von `AlexNet` für einen Batch mit 1 Bild und für einen Batch mit 128 Bildern. Nehmen Sie an, dass jede Schicht jeden Parameter pro Batch einmal liest.
    1. Betrachten Sie einen Beschleuniger, der in der Zeit für das Lesen eines Bytes aus dem Speicher 10 Multiplikationsadditionen ausführt. Die Laufzeit jeder Schicht entspricht dem Maximum aus ihrer Rechenzeit und der Lesezeit für ihre Parameter. Was beschleunigt die Inferenz für ein einzelnes Bild ($B = 1$) stärker: doppelter Rechendurchsatz oder doppelte Speicherbandbreite? Was beschleunigt das Training bei $B = 128$ stärker?

<!--en:2593248d-->
1. [code] **Batchgröße, Durchsatz und Speicherbedarf.** ● Trainieren Sie `AlexNet` wie in diesem Abschnitt und variieren Sie nur die Batchgröße.
    1. Schätzen Sie anhand der Aufgabe :exref:`ex_alexnet-memory-compute-ledger` die größte Batchgröße, die beim Training in den Speicher Ihrer GPU passt.
    1. Beginnen Sie mit Batchgröße 32 und verdoppeln Sie die Batchgröße schrittweise. Bestimmen Sie die größte Batchgröße, bei der der Speicher für das Training ausreicht. Falls das Ergebnis von Ihrer Vorhersage abweicht, erklären Sie den Hauptgrund dafür.
    1. Messen Sie den Trainingsdurchsatz (Bilder pro Sekunde) für mindestens vier Batchgrößen bis zu dieser Grenze. Ab welcher Batchgröße steigt der Durchsatz nicht mehr, und warum?

<!--en:96178d0b-->
[Diskussion](https://d2l.discourse.group/t/75)

<!--en:e8b31413-->
[Diskussion](https://d2l.discourse.group/t/76)

<!--en:85e75fe7-->
[Diskussion](https://d2l.discourse.group/t/276)

<!--en:5de5d3ba-->
[Diskussion](https://d2l.discourse.group/t/18001)

<!--en:84c209f6-->
AlexNet und gelernte Bildrepräsentationen<br>**durchgängig gelernte Merkmale · Daten und GPUs · ReLU und Dropout · Parameter und Rechenaufwand · Kosinusverlauf mit Aufwärmphase**

<!--en:2c07535b-->
::: {.slide title="Grenzen von Hand entworfener Verfahren auf ImageNet"}

<!--en:00b15d4f-->
[Motivation und früherer Engpass]{.kicker}

<!--en:0b37144d-->
- **Feste Deskriptoren (1990er-Jahre bis 2011):** Bei SIFT/HOG mit SVM änderte sich nur der lineare Klassifikator. Die Merkmale blieben fest, auf ImageNet blieb es bei **26.2% Top-5-Fehler**.

<!--en:f5026552-->
- **Durchbruch mit AlexNet 2012:** End-to-End-Training von 8 Schichten auf den Pixeln von 1.2 Mio. Bildern senkte den Top-5-Fehler auf **15.3%** (Ensemble aus sieben Netzen, einzelnes Netz $\approx 18\%$).

<!--en:fc35e159-->
- **Drei Voraussetzungen:** Eine Million Bilder, parallele Berechnung auf GPUs und Optimierung mit der nicht sättigenden ReLU

<!--en:80dfeb4e-->
::: {.slide title="Orientierte Kanten und Farbflecken durch Backpropagation"}

<!--en:3d101cf0-->
[Gelernte Repräsentationen]{.kicker}

<!--en:8e157bb0-->
Allein die Minimierung des Kreuzentropieverlusts zwingt die ersten Faltungsschichten, räumliche Filter zu lernen, die an biologisches Sehen und klassische Wavelets erinnern:

<!--en:c0d75130-->
- **Filter der ersten Schicht ($11\times 11$):** Die 96 gelernten Faltungskerne bilden zwei Gruppen: orientierte, Gabor-ähnliche Kanten- und Frequenzdetektoren sowie Filter für Flecken in Gegenfarben.

<!--en:a0d26948-->
- **Hierarchische Zusammensetzung:** Tiefere Schichten kombinieren diese lokalen Grundmuster zu Texturen (Schicht 2), Objektteilen (Schichten 3–4) und Vorlagen für ganze Objektkategorien (Schicht 5).

<!--en:8a5db882-->
- **Aufgabenspezifische Merkmale:** Anders als bei SIFT passt sich jedes Filtergewicht gemeinsam mit dem Klassifikator an die Klassengrenzen des Datensatzes an.

<!--en:45db8300-->
::: {.slide title="Voraussetzung 1: Größe von ImageNet und Datenaugmentierung zur Laufzeit"}

<!--en:cefd1cad-->
[Datenmenge und Modellkapazität]{.kicker}

<!--en:6dbc0cb9-->
- **Umfang von ILSVRC-2012:** $1.2 \times 10^6$ Trainingsbilder in $1{,}000$ Kategorien ($\sim 1{,}200$ pro Klasse)

<!--en:adbb222c-->
- **Geometrische Datenaugmentierung:** Zufällige $224 \times 224$-Bildausschnitte und horizontale Spiegelungen von $256 \times 256$-Bildern ergeben $32^2 \times 2 = 2{,}048$ Ansichten pro Bild.

<!--en:45dadd71-->
- **Fotometrische Variation mit PCA:** Zufällige Änderungen der RGB-Intensitäten entlang der Eigenvektoren des Datensatzes fördern die Invarianz gegenüber der Beleuchtung.

<!--en:dad0ddf7-->
::: {.slide title="Voraussetzung 2: GPU-Durchsatz und Rechenaufwand"}

<!--en:ebc9fa4f-->
[Skalierung der Hardware]{.kicker}

<!--en:54722b4a-->
- **Kosten pro Bild:** Ein Vorwärtsdurchlauf für ein $224 \times 224$-Bild benötigt $\approx 10^9$ MACs ($2 \times 10^9$ FLOPs).

<!--en:17fc5dc9-->
- **Rechenaufwand beim Training über 90 Epochen:** Mit Rückwärtsdurchläufen (insgesamt $3\times$ Aufwand des Vorwärtsdurchlaufs) ergeben sich bei $1.2\textrm{M}$ Bildern $\approx 6.5 \times 10^{17}$ FLOPs.

<!--en:280a848f-->
- **CPU und GPU:** Zwei GTX 580 (je $1.6\textrm{ TFLOPS}$ und $192\textrm{ GB/s}$) verkürzen das Training von deutlich über einem Monat auf einer CPU ($\approx 38$ Tage bei Spitzendurchsatz) auf **5–6 Tage**.

<!--en:120aaa9f-->
::: {.slide title="LeNet und AlexNet im direkten Vergleich"}

<!--en:d3f1407c-->
[Grundaufbau]{.kicker}

<!--en:5b2044e8-->
AlexNet übernimmt den Grundaufbau von LeNet: Faltungsschichten extrahieren Merkmale für einen vollständig verbundenen Klassifikationskopf. Tiefe, Breite und Eingabeauflösung steigen:

<!--en:0e31cc44-->
- **Starkes frühes Downsampling:** $224\times 224$-Eingaben haben $64\times$ so viele Pixel wie $28\times 28$-Miniaturbilder. Schicht 1 verwendet einen $11\times 11$-Faltungskern mit Stride 4, gefolgt von $3\times 3$-Max-Pooling mit Stride 2.

<!--en:c467659f-->
- **Drei Faltungen in Folge:** Auf eine $5\times 5$-Faltung und ein zweites Max-Pooling folgen drei $3\times 3$-Faltungen mit Padding (384 $\to$ 384 $\to$ 256 Kanäle) und ein abschließendes Max-Pooling.

<!--en:0bede016-->
- **Breiter Klassifikationskopf:** Die $256\times 5\times 5$ Merkmale bilden einen Vektor für zwei vollständig verbundene Schichten mit je 4096 Einheiten und Dropout ($p=0.5$).

<!--en:c661283a-->
::: {.slide title="Architekturen von LeNet und AlexNet im Vergleich"}

<!--en:5ee09b97-->
[Änderungen am Aufbau]{.kicker}

<!--en:3c73d9cc-->
- **Größenverhältnisse:** Eingabeauflösung: $64\times$ ($28^2 \to 224^2$), Parameterzahl: $760\times$ ($61.7\textrm{K} \to 46.8\textrm{M}$), Anzahl der MACs: $2{,}250\times$ ($0.42\textrm{M} \to 938\textrm{M}$)

<!--en:1d9e1095-->
- **Neue Bausteine:** Überlappendes $3 \times 3$-Max-Pooling mit Stride 2 ersetzt das Mittelwert-Pooling. Die nicht sättigende **ReLU** ersetzt die Sigmoidfunktion. **Dropout ($p=0.5$)** regularisiert beide vollständig verbundenen Schichten mit je 4096 Einheiten.

<!--en:ea73def2-->
::: {.slide title="Voraussetzung 3: ReLU und Dropout, ergänzt um die He-Initialisierung"}

<!--en:b93e3b0f-->
[Optimierung und Regularisierung]{.kicker}

<!--en:948d8324-->
- **1. Nicht sättigende ReLU:** Die Steigung $\textrm{ReLU}'(x) = 1$ für $x>0$ verhindert die Abschwächung um den Faktor $4^{-K}$ durch die Sigmoidfunktion ($6\times$ schnellere Konvergenz).

<!--en:0fd172b3-->
- **2. Varianzskalierung nach He (2015, hier ergänzt):** $W \sim \mathcal{N}(0, 2/(c_\textrm{i} k^2))$ gleicht aus, dass ReLU die Hälfte der Aktivierungen auf null setzt. Das ursprüngliche AlexNet verwendete $\mathcal{N}(0, 0.01^2)$.

<!--en:36bbc3db-->
- **3. Dropout ($p=0.5$):** Setzt beim Training die Hälfte der 4096 Einheiten der vollständig verbundenen Schicht auf null und skaliert die übrigen mit $2\times$.

<!--en:c799eb72-->
::: {.slide title="Implementierung von AlexNet"}

<!--en:c79b0487-->
[Modelldefinition]{.kicker}

<!--en:8257720e-->
Fünf Faltungsschichten, drei überlappende Max-Pooling-Stufen und zwei durch Dropout regularisierte vollständig verbundene Schichten:

<!--en:b276ceb6-->
::: {.slide title="Räumliche Dimensionen im Encoder" layout="tall-output"}

<!--en:517b8f22-->
[Prüfung der Tensorformen]{.kicker}

<!--en:3194ce35-->
Ein Testtensor der Form $(1, 1, 224, 224)$ zeigt mit `layer_summary`, wie die räumlichen Dimensionen abnehmen und die Kanalzahl wächst:

<!--en:4eeeaf0d-->
- **Erste Schicht mit Stride 4:** Die Seitenlänge sinkt gemäß $224 \to \lfloor(224 - 11)/4\rfloor + 1 = 54$. Die Ausgabe hat die Form $96 \times 54 \times 54$.

<!--en:702e7ebe-->
- **Überlappendes $3\times 3$-Pooling mit Stride 2:** Jede Stufe verkleinert die Seitenlänge gemäß $n \to \lfloor(n - 3)/2\rfloor + 1$. Die Auflösung sinkt schrittweise: $54$ $\to$ $26$ $\to$ $12$ $\to$ $5$.

<!--en:b287cb7b-->
- **Faltungen mit Padding:** Padding $p = (k-1)/2$ ($p=2$ für $5\times 5$, $p=1$ für $3\times 3$) erhält die räumlichen Dimensionen zwischen den Pooling-Stufen.

<!--en:b5797e25-->
- **Umformung zum Vektor:** Die letzte Merkmalskarte der Größe $256 \times 5 \times 5$ ergibt $6{,}400$ Merkmale pro Bild.

<!--en:ed660c11-->
::: {.slide title="Parameter und Rechenaufwand je Schicht"}

<!--en:e859c495-->
[Ressourcenbedarf]{.kicker}

<!--en:cc83ab10-->
Eine Faltungsschicht verwendet jedes ihrer $c_{\textrm{i}} c_{\textrm{o}} k^2$ Gewichte für alle $h_{\textrm{o}} w_{\textrm{o}}$ Ausgabepixel. Eine vollständig verbundene Schicht verwendet jedes Gewicht nur einmal pro Bild:

<!--en:fab4af05-->
| Schicht | Ausgabeform | Parameter ($c_{\textrm{i}} c_{\textrm{o}} k^2 + c_{\textrm{o}}$) | Multiplikationsadditionen ($h_{\textrm{o}} w_{\textrm{o}} c_{\textrm{i}} c_{\textrm{o}} k^2$) |

<!--en:5b763994-->
| Conv1 ($11\times 11$, $1\to 96$, $s=4$) | $96 \times 54 \times 54$ | $11{,}712$ | $33.9\text{M}$ |

<!--en:09037f70-->
| Conv2 ($5\times 5$, $96\to 256$, $p=2$) | $256 \times 26 \times 26$ | $614{,}656$ | $415.3\text{M}$ |

<!--en:2d348c4f-->
| Conv3 ($3\times 3$, $256\to 384$, $p=1$) | $384 \times 12 \times 12$ | $885{,}120$ | $127.4\text{M}$ |

<!--en:6fa3c807-->
| Conv4 ($3\times 3$, $384\to 384$, $p=1$) | $384 \times 12 \times 12$ | $1{,}327{,}488$ | $191.1\text{M}$ |

<!--en:dd023610-->
| Conv5 ($3\times 3$, $384\to 256$, $p=1$) | $256 \times 12 \times 12$ | $884{,}992$ | $127.4\text{M}$ |

<!--en:09fc5a21-->
| Dense1 ($6400 \to 4096$) + Dense2 ($4096 \to 4096$) | $4096 \to 4096$ | $26.22\text{M} + 16.78\text{M}$ | $26.2\text{M} + 16.8\text{M}$ |

<!--en:5c071ebf-->
| Dense3 ($4096 \to 10$) | $10$ | $40{,}970$ | $0.04\text{M}$ |

<!--en:be16500f-->
| **Gesamt** | | **$46{,}764{,}746$** | **$938.1\text{M}$** |

<!--en:8a603cad-->
::: {.slide title="Ungleiche Verteilung von Parametern und Rechenaufwand"}

<!--en:2fc3255e-->
[Engpass des Systems]{.kicker}

<!--en:f7aeca79-->
- **Faltungen (8% der Parameter, 95.4% der MACs):** Fünf Faltungsschichten enthalten $3.72\textrm{M}$ Gewichte ($15\textrm{ MB}$) und führen $895.1\textrm{M}$ MACs aus.

<!--en:bc6d642b-->
- **Vollständig verbundener Kopf (92% der Parameter, 4.6% der MACs):** $43.04\textrm{M}$ Gewichte ($172\textrm{ MB}$) für nur $43.0\textrm{M}$ MACs ($0.25$ MACs/Byte bei $B=1$, stark durch die Speicherbandbreite begrenzt)

<!--en:b6a88d6e-->
::: {.slide title="Kosinus-Lernratenplan mit linearer Aufwärmphase"}

<!--en:28b05c62-->
[Ablauf der Optimierung]{.kicker}

<!--en:035dd722-->
- **Aufwärmphase ($t < t_\textrm{w}$):** Linearer Anstieg $0 \to \eta_0$ über $T/10$ Schritte

<!--en:5c26eb66-->
- **Abfall entlang einer halben Kosinusperiode:** Stetige Abnahme $\eta_t \to 0$ bis zum Schritt $T$

<!--en:bc018297-->
::: {.slide title="Anpassung der Lernrate nach jedem Minibatch"}

<!--en:455c6b5d-->
[Konfiguration des Optimierers]{.kicker}

<!--en:a874b7e2-->
Aktualisierung von $\eta_t$ nach jedem Minibatch ($T = 10 \times 469 = 4{,}690$ Schritte für 10 Epochen bei Batchgröße 128) mit maximaler Lernrate $\eta_0 = 0.03$:

<!--en:6587f722-->
ReLU vermeidet die Abschwächung der Gradienten um den Faktor $4^{-K}$ durch die Sigmoidfunktionen von LeNet. Daher trainiert AlexNet stabil mit der maximalen Lernrate $\eta_0 = 0.03$ und benötigt keine so hohe Lernrate wie LeNet mit $\eta = 2.0$.

<!--en:b49371ea-->
::: {.slide title="Training von AlexNet auf vergrößerten Fashion-MNIST-Bildern"}

<!--en:3078e51d-->
[Versuchsergebnisse]{.kicker}

<!--en:ff419ecc-->
$28\times 28$-Bilder von Fashion-MNIST auf $224\times 224$ vergrößern, damit sie für die erste Schicht mit Stride 4 und drei Pooling-Stufen ausreichen. Anschließend 10 Epochen trainieren:

<!--en:dd76aa9c-->
- **Schnellere Konvergenz als bei LeNet:** AlexNet erreicht $\approx 91\%$ Validierungsgenauigkeit in 10 Epochen (ein Drittel der 30 Epochen von LeNet), mit Trainingsverlust $\approx 0.22$ und Validierungsverlust $\approx 0.25$.

<!--en:567ddc7c-->
- **Upsampling für die Untersuchung:** $64\times$ so viele Pixel liefern keine zusätzliche Bildinformation. Pro Bild fallen $2{,}250\times$ so viele Rechenoperationen an wie bei LeNet. Auf dem einfachen Datensatz mit 10 Klassen erklären ReLU, He-Initialisierung und Kosinus-Abfall den Genauigkeitsgewinn von 1–2 Prozentpunkten ebenso wie die Modellkapazität.

<!--en:e398de76-->
::: {.slide title="Gelernte Filter der ersten Schicht auf Fashion-MNIST"}

<!--en:947e1e93-->
[Untersuchung der Gewichtsänderungen]{.kicker}

<!--en:868db9c2-->
Die zufälligen Anfangsgewichte `K0` von den trainierten $11\times 11$-Filtern der ersten Schicht abziehen. Die Differenz zeigt, wie sich die Repräsentation an die Statistik des Datensatzes anpasst:

<!--en:365da7d2-->
- **Geringe relative Änderung:** In zehn Epochen ändern sich die Gewichte der ersten Schicht um nur $\approx 4\%$ ihres anfänglichen Betrags nach He-Initialisierung mit Normalverteilung. Die Gewichte selbst sehen im Bild weiterhin verrauscht aus.

<!--en:b4de2b17-->
- **Niederfrequente Struktur der Änderungen:** Die Differenzen $\mathbf{K}_{\text{trained}} - \mathbf{K}_0$ zeigen sanfte Helligkeitsverläufe und breite, ausgerichtete Balken.

<!--en:3b501b73-->
- **Auflösung passend zu den Daten:** Nach dem Upsampling ($8\times$) von $28\times 28$-Graustufenbildern sind echte Details mindestens 8 Pixel breit. Daher lernen $11\times 11$-Filter sanfte Helligkeitsverläufe statt scharfer Kanten wie auf ImageNet.

<!--en:cd77233b-->
::: {.slide title="Verständnisfragen zu rezeptiven Feldern und Speicherbedarf"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:3aa51f18-->
Zwei Fragen zur Hardware und zur Architektur beantworten:

<!--en:f2645529-->
1. **Arithmetische Intensität der ersten Schicht:** Conv1 hat $11{,}712$ Parameter im Format float32 ($46.8\text{ KB}$) und führt pro Bild $33.9\text{M}$ Multiplikationsadditionen aus. Bei Dense1 sind es $26.22\text{M}$ Parameter im Format float32 ($104.9\text{ MB}$) und $26.22\text{M}$ Multiplikationsadditionen. Wie verhalten sich bei Batchgröße $B=1$ die Multiplikationsadditionen pro gelesenem Byte der Gewichte von Conv1 und Dense1 zueinander?

<!--en:544642c2-->
2. **Räumliche Dimensionen der Ausgabe:** Ein $227\times 227$-RGB-Bild (wie beim vollständigen AlexNet für ImageNet) durchläuft Conv1 ($11\times 11$, Stride 4, Padding 0). Welches räumliche Raster erreicht die erste Max-Pooling-Schicht? Wie lang ist der Vektor nach der Umformung hinter der dritten Max-Pooling-Schicht?

<!--en:557b98b6-->
**Lösungen:**

<!--en:7a13a8ed-->
1. **Conv1:** $33.9\text{M} / 46.8\text{ KB} \approx 724$ MACs/Byte (stark rechenbegrenzt). **Dense1:** $26.22\text{M} / 104.9\text{ MB} = 0.25$ MACs/Byte (um einen Faktor von fast $2{,}900\times$ geringere arithmetische Intensität, stark durch die Speicherbandbreite begrenzt).

<!--en:e0f3aa8d-->
2. Conv1 verkleinert die Seitenlänge gemäß $227 \to \lfloor(227 - 11)/4\rfloor + 1 = 55$. Die anschließenden $3\times 3$-Pooling-Stufen mit Stride 2 verkleinern sie weiter: $55$ $\to$ $27$ $\to$ $13$ $\to$ $6$. Die letzte $256\times 6\times 6$-Merkmalskarte ergibt einen Vektor mit $9{,}216$ Einheiten. Dense1 hat damit $9216 \times 4096 = 37.7\text{M}$ Gewichte.

<!--en:7f134bd0-->
::: {.slide title="Zusammenfassung der Ergebnisse zu AlexNet"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:93d75aff-->
| Konzept oder Komponente | Mathematische Form oder Aufbau | Funktion und Auswirkung auf das System |

<!--en:b561cecf-->
| **Durchgängiges Repräsentationslernen** | Backpropagation aktualisiert Conv1–Conv5 gemeinsam mit Dense1–3 | Der AlexNet-Beitrag ersetzte SIFT/HOG und senkte den Top-5-Fehler auf ImageNet von 26.2% auf 15.3% |

<!--en:bbacd6a5-->
| **ReLU mit He-Initialisierung (2015)** | $\max(x, 0)$ mit $W \sim \mathcal{N}(0, 2/(c_{\textrm{i}}k^2))$ | Die Ableitung $1$ bei aktiven Einheiten verhindert die Abschwächung des Gradienten um den Faktor $4^{-K}$ durch die Sigmoidfunktion |

<!--en:56d23347-->
| **Überlappendes Max-Pooling** | $3\times 3$-Fenster, Stride $2$: $n \mapsto \lfloor(n-3)/2\rfloor + 1$ | Downsampling $224$ $\to$ $54$ $\to$ $26$ $\to$ $12$ $\to$ $5$ unter Erhalt der stärksten Aktivierungen |

<!--en:e1879c70-->
| **Verteilung von Rechenaufwand und Parametern** | Faltungen: 3.7 Mio. Parameter (8%), 895 Mio. MACs (95%)<br>Vollständig verbundener Kopf: 43.0 Mio. Parameter (92%), 43 Mio. MACs (5%) | Faltungen verwenden Gewichte an vielen räumlichen Positionen. Der Kopf benötigt viel Speicher und Dropout ($0.5$). |

<!--en:a6c4c949-->
| **Kosinus-Lernratenplan mit Aufwärmphase** | Linearer Anstieg auf $\eta_0$ über $t_{\textrm{w}}$ Schritte, dann $\frac{\eta_0}{2}(1 + \cos\frac{\pi(t-t_{\textrm{w}})}{T-t_{\textrm{w}}})$ | Stabilisiert die anfangs zufälligen Gewichte und lässt die Parameter gegen Ende zur Ruhe kommen |
