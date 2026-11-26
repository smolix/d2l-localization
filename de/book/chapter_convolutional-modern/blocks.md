<!--en:14015428-->
# Blöcke, Flaschenhälse und Zweige in VGG, NiN und GoogLeNet

<!--en:734692bf-->
AlexNet (:numref:`sec_alexnet`) wurde Schicht für Schicht entworfen. Hier ein $11 \times 11$-Faltungskern, dort 384 Kanäle, dazu Max-Pooling nach der ersten, zweiten und fünften Faltung (engl. convolution). Bei einem Modell mit acht Schichten lässt sich jede Schicht noch von Hand abstimmen. Für zwanzig oder mehr Schichten brauchen wir systematische Entwurfsregeln. Ebenso auffällig ist die Verteilung der Parameter. Im vollvernetzten Klassifikationskopf (engl. classification head) stecken 92% der 47 Millionen Parameter von AlexNet. Dieser Kopf beansprucht nur 5% des Rechenaufwands und braucht starkes Dropout, damit er den Trainingsdatensatz nicht auswendig lernt. In den Jahren 2013 und 2014 griffen drei wegweisende Architekturen diese Engpässe nacheinander auf. Die ursprünglichen Modelle kommen heute in der Praxis nur noch selten zum Einsatz. Ihre Entwurfsprinzipien finden sich jedoch in fast jedem modernen konvolutionären Netz.

<!--en:59f11b2c-->
Den Anfang machte *VGG* :cite:`Simonyan.Zisserman.2014`. Wiederholbare *Blöcke* traten an die Stelle einzeln entworfener Schichten. Jeder Block besteht aus kleinen $3 \times 3$-Faltungen mit anschließendem Pooling. So lässt sich die gesamte Architektur durch eine kurze Liste von Blockbeschreibungen angeben. Den übergroßen vollvernetzten Kopf beseitigte *Network in Network* (NiN) vollständig :cite:`Lin.Chen.Yan.2013`. $1 \times 1$-Faltungen mischen die Kanäle an jedem Pixel. Global-Average-Pooling verdichtet die letzten Merkmalskarten (engl. feature maps) direkt zu Klassen-Scores. Als dritte Architektur folgte *GoogLeNet* :cite:`Szegedy.Liu.Jia.ea.2015`. Es verwendet in jedem Block mehrere Filtergrößen parallel. So entfällt die Festlegung auf eine einzige Kerngröße. Flaschenhälse (engl. bottlenecks) mit $1 \times 1$-Faltungen verringern die Kanalzahl und begrenzen dadurch den Rechenaufwand der größeren Faltungen. Im Folgenden bauen wir alle drei Architekturen auf und zählen wie bei AlexNet ihre Parameter und Multiplikationsadditionen (engl. multiply-adds). Für einen direkten Vergleich trainieren wir sie auf Fashion-MNIST mit demselben Kosinusplan für die Lernrate.

<!--en:debe1453-->
## VGG und der Block als Entwurfseinheit

<!--en:dd8fad6a-->
### Gestapelte $3 \times 3$-Faltungen

<!--en:3f6b9c8a-->
Folgte auf jede Faltungsschicht ein Pooling mit Stride 2, ließe eine Eingabe der Breite $d$ höchstens $\lfloor\log_2 d\rfloor$ Schichten zu, bevor die räumliche Auflösung auf $1 \times 1$ fällt ($224 \to 112 \to 56 \to 28 \to 14 \to 7 \to 3 \to 1$). Für tiefere Netze stapeln wir bei jeder räumlichen Auflösung mehrere formerhaltende Faltungen mit Stride 1, bevor wir die Auflösung durch Downsampling verringern.

<!--en:71ea682c-->
Bei VGG ist die Kerngröße einheitlich :cite:`Simonyan.Zisserman.2014`. Das Netz verwendet durchgehend $3 \times 3$-Faltungskerne. Dies ist der kleinste symmetrische Filter, der die Nachbarn links, rechts, oben und unten erfasst. Zwei gestapelte $3 \times 3$-Schichten decken dasselbe rezeptive Feld (engl. receptive field) der Größe $5 \times 5$ ab wie eine einzelne $5 \times 5$-Schicht. Dazu brauchen sie $2 \cdot 9c^2 = 18c^2$ statt $25c^2$ Gewichte. Drei gestapelte $3 \times 3$-Schichten erfassen ein $7 \times 7$-Fenster mit $27c^2$ statt $49c^2$ Gewichten und bringen drei ReLU-Nichtlinearitäten statt einer ein. Ein *VGG-Block* stapelt `num_convs` $3 \times 3$-Faltungen mit Padding und ReLU. Den Abschluss bildet ein $2 \times 2$-Max-Pooling mit Stride 2.

<!--en:0dbd5afd-->
### Das VGG-Netz

<!--en:76991338-->
Ein vollständiges VGG-Netz reiht diese Blöcke aneinander. Darauf folgt der vollvernetzte Klassifikationskopf von AlexNet. Er enthält zwei versteckte Schichten mit je 4096 Einheiten und Dropout, gefolgt von der Ausgabeschicht. :numref:`fig_vgg` stellt VGG-11 und AlexNet gegenüber. Fünf Pooling-Stufen halbieren die Seitenlängen einer $224 \times 224$-Eingabe schrittweise bis auf $7 \times 7$. Die Kanalzahl verdoppelt sich von Stufe zu Stufe, bis sie 512 erreicht.

<!--en:078f5220-->
![AlexNet (links) und VGG-11 (rechts) für einkanalige $224 \times 224$-Bilder und zehn Klassen. Die Eingabe liegt unten. Bei AlexNet ist jede Schicht einzeln gewählt. VGG wiederholt mit wachsender Kanalzahl denselben Block aus mehreren $3 \times 3$-Faltungen und anschließendem Max-Pooling.](../img/arch-vgg.svg)

<!--en:eb54da00-->
Zwei ganze Zahlen legen jeden VGG-Block vollständig fest, nämlich die Anzahl seiner Faltungen und die seiner Ausgabekanäle. Daher können wir den gesamten Faltungsteil als Liste von `(num_convs, out_channels)`-Tupeln übergeben (`arch`). VGG beschreibt damit eine *parametrisierte Familie* von Architekturen. VGG-11, VGG-16 und VGG-19 verwenden dieselbe Klasse und unterscheiden sich nur in `arch`. Die Netztopologie durch modulare Daten zu beschreiben, statt sie von Hand im Code festzulegen, ist der nachhaltigste Beitrag von VGG. Wie in :numref:`sec_alexnet` initialisieren wir die Gewichte aller Netze in diesem Abschnitt nach He mit einer Normalverteilung und setzen die Bias-Terme auf null.

<!--en:4da14f49-->
Wie bei AlexNet bauen wir die Netze aus `HybridSequential`-Containern und `HybridBlock`-Bausteinen auf. `hybridize()` kompiliert sie zu einem einzigen Graphen. Die `Dense`-Schicht von Gluon formt ihre Eingabe in einen Vektor um. Der Kopf braucht dafür also keine eigene Schicht.

<!--en:6bf815dd-->
Wie bei AlexNet ruft der `Trainer` für jede Schicht `weight_init` auf, sobald der erste Minibatch die Formen der verzögert initialisierten Schichten festgelegt hat.

<!--en:c38ad8f5-->
Mit `he()` erhält jede Schicht einen eigenen Initialisierer, denn ein Keras-Initialisierer ohne expliziten Startwert für den Zufallsgenerator liefert bei jedem Aufruf dieselben Werte.

<!--en:29e469ea-->
Das Wörterbuch `init` übergibt jeder Schicht den Initialisierer für die Gewichte und die Zufallszahlenströme.

<!--en:211ab761-->
`VGG` erhält die Bildgröße als `input_shape`, weil NNX-Schichten ihre Eingabegrößen schon beim Aufbau benötigen. Die erste vollständig verbundene Schicht liest $h w c$ Zahlen, wenn die letzte Merkmalskarte $h \times w$ Pixel und $c$ Kanäle hat.

<!--en:ce5d79ce-->
VGG-11, die kleinste Variante der Originalarbeit, stapelt fünf Blöcke mit (1, 1, 2, 2, 2) Faltungsschichten und (64, 128, 256, 512, 512) Kanälen. Acht Faltungsschichten und drei vollständig verbundene Schichten ergeben insgesamt 11 parametrisierte Schichten. Betrachten wir die Formen der Zwischentensoren für ein einzelnes $224 \times 224$-Bild.

<!--en:c53876b3-->
Der letzte Faltungsblock gibt einen $512 \times 7 \times 7$-Tensor aus. Als Vektor liefert er dem vollvernetzten Kopf 25 088 Merkmale. Zählen wir wie bei AlexNet die Parameter und Rechenschritte der einzelnen Schichten, kommen wir bei VGG-11 auf 128.8 Millionen Parameter und 7.5 Milliarden Multiplikationsadditionen pro Bild. Das ist der achtfache Rechenaufwand von AlexNet. Wieder konzentrieren sich die Parameter in einem ganz anderen Teil des Netzes als die Rechenarbeit. Der vollvernetzte Kopf enthält 93% aller Gewichte. Allein auf seine erste Matrix der Größe $25{,}088 \times 4096$ entfallen 103 Millionen Parameter.

<!--en:70d292d9-->
### Training

<!--en:7fb7b362-->
Für eine Demonstration auf Fashion-MNIST wäre der achtfache Rechenaufwand von AlexNet unnötig hoch. Deshalb trainieren wir VGG-11 mit einem Viertel der ursprünglichen Breite und den Kanalzahlen 16, 32, 64, 128 und 128. Mit 512 Millionen Multiplikationsadditionen pro Bild benötigt es nur etwa halb so viele wie AlexNet mit 938 Millionen. Alle anderen Trainingseinstellungen übernehmen wir aus :numref:`sec_alexnet`. Wir vergrößern die Bilder auf $224 \times 224$ und trainieren mit Batchgröße 128 über 10 Epochen mit SGD. Für den Kosinusplan mit Aufwärmphase aus :numref:`sec_alexnet` verwenden wir wieder die Methode `configure_optimizers` von AlexNet.

<!--en:bec8842c-->
Die maximale Lernrate ist der einzige Hyperparameter, den wir in diesem Abschnitt je nach Netz anders wählen. Bei einem Höchstwert von 0.1 sind mehrere der folgenden Architekturen ohne Normalisierung kaum noch stabil. Gelegentlich divergiert das Training oder die Genauigkeit bleibt auf Zufallsniveau. Diese Modelle trainieren wir mit 0.03. Unser VGG mit einem Viertel der Breite lässt sich bei 0.1 stabil trainieren. Seine Genauigkeit liegt dann um 1.5 bis 2 Prozentpunkte höher als bei 0.03. Daher verwenden wir für dieses Netz 0.1.

<!--en:be5ef793-->
Mit dem Kosinusplan und seiner Aufwärmphase trainieren wir unser `VGG` mit einem Viertel der Breite für 10 Epochen auf $224 \times 224$ Pixel großen Bildern von Fashion-MNIST. Den Verlauf der Validierungsgenauigkeit speichern wir in `val_acc['VGG']`.

<!--en:53d06342-->
Unser VGG mit einem Viertel der Breite erreicht rund 93% Validierungsgenauigkeit. Damit liegt es bei etwa halbem Rechenaufwand zwei Prozentpunkte vor AlexNet. Allerdings erklärt die höhere Lernrate einen großen Teil des Vorsprungs. Mit der maximalen Lernrate von AlexNet, also 0.03, landet dasselbe VGG knapp unter 92%, wie wir gleich überprüfen. Ein Problem löst der schmalere Faltungsteil jedoch nicht. Die Faltungen kommen mit nur 0.6 Millionen Parametern aus, doch der vollvernetzte Kopf bildet weiterhin $128 \times 7 \times 7 = 6272$ Aktivierungen auf 4096 versteckte Einheiten ab. Auf ihn entfallen 42.5 Millionen der 43.1 Millionen Modellparameter, also erstaunliche 99%. Die Blöcke geben dem Hauptteil des Netzes eine regelmäßige Struktur. Der Kopf bleibt so groß wie zuvor.

<!--en:bfb24bf3-->
## NiN mit $1 \times 1$-Faltungen und Global-Average-Pooling

<!--en:6abc2384-->
Ein vollvernetzter Klassifikationskopf benötigt mehrere zehn Millionen Parameter, weil jede räumliche Position jedes Kanals mit jeder versteckten Einheit durch ein eigenes Gewicht verbunden ist. Diese gesamte Kapazität dient einer einzigen Aufgabe. Der Kopf fasst die letzten Merkmalskarten zu Klassen-Logits zusammen. Können wir ganz auf den vollvernetzten Kopf verzichten und zugleich die Nichtlinearität jedes Faltungsblocks ausdrucksstärker machen? Dieser Frage geht *Network in Network* (NiN) nach. Erinnern wir uns, was eine gewöhnliche Faltung mit anschließender ReLU leistet. Sie wendet auf jeden lokalen Bildausschnitt lediglich einen linearen Filter mit nachgeschaltetem Schwellenwert an. Um auf demselben räumlichen Fenster eine komplexere nichtlineare Funktion zu berechnen, können wir mehrere Schichten stapeln, die an jedem Pixel die Kanäle mischen. Der Faltungskern muss dafür nicht größer werden. Genau das leisten die $1 \times 1$-Faltungen aus :numref:`subsec_1x1`.

<!--en:4831b688-->
### NiN-Blöcke

<!--en:195f8123-->
Eine $1 \times 1$-Faltung wendet an jedem Pixel unabhängig dieselbe vollständig verbundene Schicht an. Lassen wir auf eine räumliche Faltung zwei $1 \times 1$-Faltungen mit je einer ReLU folgen, sitzt an jeder räumlichen Position ein kleines MLP mit einer versteckten Schicht. Daher der Name *Network in Network*. :numref:`fig_nin` stellt dem vollvernetzten Kopf von VGG den Abschluss von NiN gegenüber.

<!--en:4eb38e1a-->
![Der Kopf von VGG (links) und der Abschluss von NiN (rechts), jeweils für zehn Klassen. Die Eingabe liegt unten. VGG formt seine letzte Merkmalskarte zu einem Vektor für die vollständig verbundenen Schichten um. Bei NiN folgt auf das Dropout der letzte Block mit einer $3 \times 3$-Faltung und zwei $1 \times 1$-Faltungen. Nach der letzten Faltung entfällt die ReLU. Der Block gibt einen Kanal pro Klasse aus. Global-Average-Pooling berechnet aus jedem Kanal einen Score.](../img/arch-nin.svg)

<!--en:7d7f64a2-->
### Global-Average-Pooling

<!--en:84f24cf0-->
Im Grundaufbau übernimmt NiN die Kerngrößen und Kanalzahlen der frühen Schichten von AlexNet. Die Blöcke beginnen mit $11 \times 11$-, $5 \times 5$- und $3 \times 3$-Faltungen mit 96, 256 beziehungsweise 384 Kanälen. Auf jeden Block folgt ein $3 \times 3$-Max-Pooling mit Stride 2. Am oberen Ende des Netzes weicht NiN jedoch grundlegend von AlexNet und VGG ab. Auf das Dropout folgt ein vierter NiN-Block mit 384 Kanälen. Er endet mit einer $1 \times 1$-Faltung, die genau einen Ausgabekanal pro Klasse erzeugt (10 bei Fashion-MNIST). *Global-Average-Pooling* mittelt jede dieser zehn Merkmalskarten über alle räumlichen Koordinaten. So erhalten wir direkt die zehn Klassen-Logits. Da der Pooling-Kopf keine Parameter hat, muss der letzte Block selbst lernen, im Kanal der richtigen Klasse einen hohen räumlichen Mittelwert zu erzeugen.

<!--en:4b90a4d9-->
Ein Detail ist für die Implementierung wichtig. Übergeben wir `num_classes` an `nin_block`, entfällt die ReLU-Aktivierung nach der letzten $1 \times 1$-Faltung, weil diese Faltung die rohen Klassen-Logits ausgibt. Eine ReLU auf den Logits könnte eine Klasse dauerhaft ausschalten. Würden die Werte eines Klassenkanals im gesamten Datensatz an jeder Position negativ, wären sein Logit und sein Gradient überall null. SGD könnte die Klasse dann nie wieder aktivieren.

<!--en:9700d406-->
Global-Average-Pooling hat neben der Parameterersparnis einen weiteren Vorteil. Faltungen sind translationsäquivariant, und der Mittelwert über alle räumlichen Koordinaten ist invariant gegenüber Permutationen. Bis auf Rand- und Stride-Effekte hängen die Klassen-Scores daher nicht davon ab, *wo* ein Objekt im Bild erscheint. Ein vollvernetzter Kopf mit Vektoreingabe weist dagegen jeder räumlichen Position eigene Gewichte zu. Er muss jedes Objekt an jeder Position neu lernen.

<!--en:c178d69b-->
Verfolgen wir die Tensorformen für eine einzelne $224 \times 224$-Eingabe.

<!--en:f99ca3ba-->
Wie bei AlexNet sinken die räumlichen Abmessungen von 224 schrittweise auf 54, 26, 12 und 5. Dann gibt der letzte NiN-Block zehn $5 \times 5$-Score-Karten aus, die Global-Average-Pooling zu zehn Logits verdichtet. Ohne den vollvernetzten Kopf sinkt die Parameteranzahl auf nur 3.4 Millionen (7% der Parameter von AlexNet). Mit 798 Millionen Multiplikationsadditionen pro Bild bleibt der Rechenaufwand jedoch mit dem von AlexNet vergleichbar (938 Millionen). Die zusätzlichen $1 \times 1$-Faltungen verlagern die Rechenarbeit in den Faltungsteil.

<!--en:513da5f5-->
### Training von NiN

<!--en:def88d1b-->
Wir trainieren NiN mit demselben Lernratenplan wie VGG. Als maximale Lernrate wählen wir 0.03, weil das Training bei 0.1 gelegentlich divergiert.

<!--en:385f30aa-->
Nach 10 Epochen erreicht NiN rund 89% Validierungsgenauigkeit, etwa zwei Prozentpunkte weniger als AlexNet und vier weniger als unser VGG mit einem Viertel der Breite. Der Trainingsverlust liegt am Ende bei etwa 0.30 (gegenüber rund 0.22 bei AlexNet). Der Validierungsverlust folgt ihm eng. Das deutet nach 10 Epochen auf eine leichte Unteranpassung (engl. underfitting) ohne Überanpassung (engl. overfitting) hin. Für sich genommen erzielt NiN gemessen am Rechenaufwand ein bescheidenes Ergebnis. Seine bleibende Bedeutung verdankt es zwei wiederverwendbaren Bausteinen, die es verbreitet hat. Das sind $1 \times 1$-Faltungen zur Kanalmischung und Köpfe mit Global-Average-Pooling.

<!--en:e79ddcb0-->
### Ein Pooling-Kopf für VGG

<!--en:09a0e42f-->
NiN verändert den Faltungsteil und den Klassifikationskopf zugleich. An seiner Genauigkeit allein lässt sich daher der Beitrag des vollvernetzten Kopfes nicht ablesen. Um ihn isoliert zu untersuchen, behalten wir den Hauptteil von VGG mit einem Viertel der Breite unverändert bei und überschreiben nur `VGG.head`. In `PooledVGG` fasst Global-Average-Pooling die 128 Ausgabekanäle des letzten Blocks zusammen. Eine einzige lineare Schicht bildet diese 128 Merkmale direkt auf die zehn Klassen ab. So entfallen 42.5 Millionen der 43.1 Millionen Netzparameter und ebenso viele Multiplikationsadditionen.

<!--en:52776463-->
Weil das Training von `PooledVGG` wie das von `NiN` bei 0.1 divergieren oder zusammenbrechen kann, trainieren wir `VGG` und `PooledVGG` mit derselben maximalen Lernrate `lr=0.03`. Ihre endgültigen Validierungsgenauigkeiten speichern wir in `heads`.

<!--en:54984f27-->
`PooledVGG` erreicht am Ende 89–90% Validierungsgenauigkeit, etwa zwei Prozentpunkte weniger als `VGG` mit vollvernetztem Kopf (91–92% bei derselben maximalen Lernrate von 0.03). Auch der Trainingsverlust liegt höher (rund 0.25 gegenüber rund 0.20). Bei diesem schmalen Faltungsteil und 10 Trainingsepochen auf Fashion-MNIST bringt die zusätzliche Kapazität des vollvernetzten Kopfes also rund zwei Prozentpunkte mehr Genauigkeit. Dafür beansprucht der Kopf 99% der Modellparameter.

<!--en:ad6b848c-->
## GoogLeNet und Blöcke mit mehreren Zweigen

<!--en:91b674ce-->
Bei allen bisherigen Architekturen mussten wir für jede Schicht eine einzige Kerngröße wählen. Ein $1 \times 1$-Kern mischt Kanäle lokal, ein $3 \times 3$-Kern erfasst feine Texturen und ein $5 \times 5$-Kern überblickt größere Umgebungen. Auf Fotos kommen Objekte aber in allen Größen vor. GoogLeNet umgeht diese Wahl. Sein *Inception-Block* verarbeitet dieselbe Eingabe parallel mit mehreren Kerngrößen und verkettet die Ausgaben entlang der Kanalachse. Ohne weitere Vorkehrungen würden mit mehreren Filtergrößen zugleich die Parameteranzahl und der Rechenaufwand stark steigen. Deshalb schaltet GoogLeNet jedem aufwendigen räumlichen Filter eine $1 \times 1$-Faltung nach dem Vorbild von NiN vor. Sie dient als Flaschenhals zur Dimensionsreduktion.

<!--en:c0b25dcc-->
### Der Inception-Block

<!--en:de521f68-->
Wie :numref:`fig_inception` zeigt, verarbeitet jeder Inception-Block seine Eingabe in vier parallelen Zweigen. (1) Eine $1 \times 1$-Faltung. (2) Eine $1 \times 1$-Faltung zur Kanalreduktion mit anschließender $3 \times 3$-Faltung. (3) Eine $1 \times 1$-Faltung zur Kanalreduktion mit anschließender $5 \times 5$-Faltung. (4) Eine $3 \times 3$-Max-Pooling-Schicht mit Stride 1 und anschließender $1 \times 1$-Faltung. Jeder Zweig verwendet Padding, um Höhe und Breite der Eingabe zu erhalten. So lassen sich die vier Ausgaben entlang der Kanaldimension verketten. Über die Kanalzahl pro Zweig lässt sich die Kapazität auf verschiedene räumliche Maßstäbe verteilen. (Der Name des Blocks spielt auf den Film *Inception* und das Deep-Learning-Meme „Wir müssen tiefer gehen“ an.)

<!--en:fd7b1972-->
![Der Inception-Block mit der Eingabe unten. Vier parallele Zweige verarbeiten dieselbe Eingabe mit verschiedenen Kerngrößen. $1 \times 1$-Faltungen verringern die Kanalzahl vor den $3 \times 3$- und $5 \times 5$-Faltungen. Die Ausgaben der Zweige werden entlang der Kanaldimension verkettet. Die Formen gelten für den ersten Inception-Block von GoogLeNet bei einer $224 \times 224$-Eingabe.](../img/arch-inception-block.svg)

<!--en:9c4879f5-->
Am ersten Inception-Block von GoogLeNet lässt sich die Ersparnis durch $1 \times 1$-Faltungen zur Kanalreduktion gut ablesen. Bei einer $224 \times 224$-Eingabe erhält der Block 192 Kanäle mit der Auflösung $28 \times 28$ (:numref:`fig_inception`). Mit den Zweigbreiten $c_1 = 64$, $c_2 = (96, 128)$, $c_3 = (16, 32)$ und $c_4 = 32$ gibt er $64 + 128 + 32 + 32 = 256$ Kanäle mit derselben Auflösung $28 \times 28$ aus. Eine direkte $5 \times 5$-Faltung von 192 auf 32 Kanäle würde $25 \cdot 192 \cdot 32 = 153{,}600$ Gewichte benötigen. Projizieren wir die 192 Kanäle zuerst auf 16, brauchen wir nur $192 \cdot 16 = 3{,}072$ Gewichte für die $1 \times 1$-Reduktion und $25 \cdot 16 \cdot 32 = 12{,}800$ für den $5 \times 5$-Filter. Das ergibt insgesamt 15 872 Gewichte. Die Parameteranzahl und die Zahl der Multiplikationsadditionen sinken damit fast um den Faktor zehn. Eine solche schmale $1 \times 1$-Projektion vor einer aufwendigen räumlichen Faltung heißt *Flaschenhals*. Sie gehört inzwischen zu den verbreitetsten Bausteinen tiefer Netze für die Bildverarbeitung und begegnet uns in :numref:`subsec_resnext` und :numref:`sec_efficient_cnns` wieder.

<!--en:95ddcd32-->
### Eingangsteil, Hauptteil und Kopf

<!--en:8fbced55-->
GoogLeNet etablierte die Gliederung in *Eingangsteil, Hauptteil und Kopf*, die fast allen modernen Grundnetzen für die Bildverarbeitung gemeinsam ist. Der *Eingangsteil* (engl. stem) verarbeitet die rohen Pixel mit wenigen gewöhnlichen Faltungen und Pooling-Schichten, hier drei Faltungen und zweimal Max-Pooling. Dabei sinkt die räumliche Auflösung rasch um den Faktor acht, während die Kanalzahl auf 192 steigt. Der *Hauptteil* (engl. body) reiht neun Inception-Blöcke in drei Stufen mit zwei, fünf und zwei Blöcken aneinander. Zwischen den Stufen liegt jeweils ein Max-Pooling mit Stride 2. Der *Kopf* (engl. head) übernimmt den kompakten Aufbau von NiN. Auf Global-Average-Pooling folgt eine einzige lineare Klassifikationsschicht.

<!--en:e6d43d32-->
Die Originalarbeit gibt keine einfache Formel für die Breiten der einzelnen Zweige an. Sie wurden für die neun Inception-Blöcke von Hand abgestimmt. Wie bei VGG kodieren wir die gesamte Konfiguration als strukturierte Daten. Ein Tupel beschreibt die Stufen, jede Stufe enthält ein Konfigurationstupel pro Inception-Block:

<!--en:5edb9d08-->
Übergeben wir `arch` an `GoogLeNet`, setzt die Klasse das Netz zusammen. Auf den Eingangsteil folgen die drei durch Max-Pooling getrennten Inception-Stufen und der Kopf mit globalem Pooling.

<!--en:3e671361-->
GoogLeNet verringert die räumliche Auflösung um den Faktor 32. Dazu dienen fünf Downsampling-Schritte mit Stride 2. Drei liegen im Eingangsteil (die erste $7 \times 7$-Faltung und zweimal Max-Pooling), zwei zwischen den Inception-Stufen. Damit das Training auf Fashion-MNIST zügig läuft, skalieren wir die Eingaben auf $96 \times 96$. Die drei Inception-Stufen arbeiten dann mit $12 \times 12$, $6 \times 6$ und $3 \times 3$. Die Kanalzahl steigt von 192 nach dem Eingangsteil auf 480, 832 und 1024.

<!--en:a9a0d159-->
GoogLeNet ist die mit Abstand tiefste Architektur in diesem Abschnitt. Auf dem längsten Pfad liegen 22 parametrisierte Schichten, nämlich drei im Eingangsteil, je zwei in den neun Inception-Blöcken und eine lineare Schicht im Kopf. Wie bei NiN trainieren wir mit einer maximalen Lernrate von 0.03, weil die Genauigkeit bei 0.1 gelegentlich auf Zufallsniveau bleibt. Für den folgenden Vergleich ist die Eingabegröße wichtig. Wir trainieren mit $96 \times 96$ statt mit $224 \times 224$ Pixeln. Daher benötigt GoogLeNet bei 6.0 Millionen Parametern nur 276 Millionen Multiplikationsadditionen pro Bild.

<!--en:86c504cb-->
Nach 10 Epochen erreicht GoogLeNet rund 91% Validierungsgenauigkeit und liegt damit bei gleicher maximaler Lernrate gleichauf mit AlexNet. Der Trainingsverlust ist etwas niedriger (knapp unter 0.20), der Validierungsverlust etwa gleich hoch (rund 0.25).

<!--en:8c8caa4a-->
## Vergleich der Netze

<!--en:76ffc31a-->
Da der Kosinusplan die Lernrate stetig auf null senkt, ist die Validierungsgenauigkeit in der letzten Epoche bei allen vier Modellen stabil. So können wir sie direkt vergleichen:

<!--en:ec6e9425-->
:numref:`tab_blocks_scoreboard` stellt LeNet, AlexNet und alle vier Architekturen aus diesem Abschnitt gegenüber. Die Tabelle zeigt die Eingabeauflösung, die Parameteranzahl, die Multiplikationsadditionen pro Bild, die maximale Lernrate und die Validierungsgenauigkeit auf Fashion-MNIST.

<!--en:2bf8fec3-->
:Parameter, Multiplikationsadditionen und Validierungsgenauigkeit auf Fashion-MNIST für die Netze aus :numref:`sec_lenet`, :numref:`sec_alexnet` und diesem Abschnitt. Die Multiplikationsadditionen gelten pro Bild bei der im Training verwendeten Eingabegröße. Die Genauigkeiten geben die Spannweite über mehrere Läufe nach 10 Epochen mit der angegebenen maximalen Lernrate an. Ein Einzelwert bedeutet, dass die Läufe höchstens einen halben Prozentpunkt auseinanderliegen. Die Werte für LeNet gelten nach 30 Epochen bei konstanter Lernrate.

<!--en:802f11e3-->
| Netz | Eingabe | Parameter | Multiplikationsadditionen | Maximale Lernrate | Validierungsgenauigkeit |

<!--en:66e6fc81-->
| LeNet | $28 \times 28$ | 0.06 Mio. | 0.4 Mio. | 2.0 | 88–90% (30 Epochen) |

<!--en:4595ffc4-->
| AlexNet | $224 \times 224$ | 46.8 Mio. | 938 Mio. | 0.03 | 91% |

<!--en:59c5d0fe-->
| VGG mit einem Viertel der Breite | $224 \times 224$ | 43.1 Mio. | 512 Mio. | 0.1 | 93% |

<!--en:8209b159-->
| VGG mit einem Viertel der Breite | $224 \times 224$ | 43.1 Mio. | 512 Mio. | 0.03 | 91–92% |

<!--en:fe2927cf-->
| VGG mit einem Viertel der Breite und Pooling-Kopf | $224 \times 224$ | 0.6 Mio. | 470 Mio. | 0.03 | 89–90% |

<!--en:addf48a5-->
| NiN | $224 \times 224$ | 3.4 Mio. | 798 Mio. | 0.03 | 89% |

<!--en:8b2c2564-->
| GoogLeNet | $96 \times 96$ | 6.0 Mio. | 276 Mio. | 0.03 | 91% |

<!--en:21d8f540-->
Aus :numref:`tab_blocks_scoreboard` lassen sich drei Schlüsse ziehen:

<!--en:4ea9280a-->
1. **Der Rechenaufwand allein bestimmt die Genauigkeit nicht:** VGG mit einem Viertel der Breite erreicht die höchste Genauigkeit und benötigt nur halb so viele Multiplikationsadditionen wie AlexNet. NiN rechnet fast so viel wie AlexNet und liegt dennoch zwei Prozentpunkte dahinter.

<!--en:0cf9f67e-->
2. **Flaschenhälse und Downsampling im Eingangsteil sparen Parameter und Pixel:** GoogLeNet erreicht wie AlexNet 91% Genauigkeit, benötigt aber $8\times$ weniger Parameter (6.0 Mio. gegenüber 46.8 Mio.) und arbeitet mit $96 \times 96$-Eingaben. Das ergibt 276 Millionen Multiplikationsadditionen. Bei $224 \times 224$ Pixeln benötigt GoogLeNet dagegen etwa 1.5 Milliarden Multiplikationsadditionen.

<!--en:24622eca-->
3. **Bei einem schmalen Faltungsteil spart Global-Average-Pooling viele Parameter auf Kosten eines kleinen Genauigkeitsverlusts:** Ersetzen wir den vollvernetzten Kopf von `VGG` durch Global-Average-Pooling (`PooledVGG`), entfallen 99% der Parameter (43.1 Mio. $\to$ 0.6 Mio.). Auf Fashion-MNIST kostet das rund 2 Prozentpunkte Genauigkeit. Auf ImageNet hingegen übertraf GoogLeNet mit seinem tieferen Hauptteil und Pooling-Kopf die VGG-Modelle mit 144 Millionen Parametern deutlich.

<!--en:af1f2655-->
Alle vier Netze stoßen zudem an ähnliche Grenzen. Keines normalisiert die Aktivierungen der Zwischenschichten. Ohne zusätzliche Köpfe oder stufenweises Pre-Training kommt keines über rund 22 Schichten hinaus. NiN, `PooledVGG` und GoogLeNet werden instabil, wenn die maximale Lernrate von 0.03 auf 0.1 steigt. In :numref:`sec_batch_norm` beheben wir diese Empfindlichkeit gegenüber der Skalierung direkt.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:87cbece4-->
Nach AlexNet verlagerte sich der Entwurf tiefer Netze für die Bildverarbeitung von der einzelnen, von Hand entworfenen Schicht auf die Kombination modularer Blöcke. VGG führte wiederholbare Blöcke aus gestapelten $3 \times 3$-Faltungen mit Padding und anschließendem Pooling ein. VGG zeigte, dass gestapelte kleine Faltungskerne große rezeptive Felder mit weniger Gewichten und mehr Nichtlinearitäten erfassen. Zugleich ließ sich die Netztopologie als Daten darstellen. NiN führte $1 \times 1$-Faltungen zur pixelweisen Kanalmischung ein. An die Stelle der großen vollvernetzten Klassifikationsköpfe, die bei AlexNet und VGG über 90% der Gewichte enthalten, trat parameterfreies Global-Average-Pooling. GoogLeNet kombinierte parallele Zweige für verschiedene räumliche Maßstäbe mit $1 \times 1$-Flaschenhälsen und gliederte das Netz klar in Eingangsteil, Hauptteil und Kopf. Auf Fashion-MNIST erreicht VGG mit einem Viertel der Breite bei gleichem Kosinusplan mit Aufwärmphase die höchste Genauigkeit. GoogLeNet liegt mit einem Achtel der Parameter gleichauf mit AlexNet. Ersetzen wir den vollvernetzten Kopf von VGG durch Global-Average-Pooling, sparen wir 99% der Parameter und verlieren rund zwei Prozentpunkte Genauigkeit.

<!--en:2294243a-->
## Übungen

<!--en:e377c9ef-->
1. **VGG-16 anhand seiner Konfiguration.** VGG-16 stapelt 13 Faltungsschichten in fünf Blöcken mit `arch=((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))` und drei vollständig verbundene Schichten mit 4096, 4096 und 1000 Einheiten. Die Eingaben sind $224 \times 224$-RGB-Bilder.
    1. Zählen Sie die Parameter und Multiplikationsadditionen von Hand, getrennt für den Faltungsteil und den vollvernetzten Kopf. Vergleichen Sie Ihre Gesamtzahl mit den 138 Millionen Parametern aus der Arbeit von :citet:`Simonyan.Zisserman.2014`.
    1. Welche einzelne Schicht enthält die meisten Parameter? Welche Schichten benötigen die meisten Multiplikationsadditionen und warum liegen mehrere Faltungsschichten gleichauf?

<!--en:3e0b32bf-->
1. [code] **Ursprüngliche Auflösung.** Fashion-MNIST-Bilder haben $28 \times 28$ Pixel. Das Upsampling auf $224 \times 224$ erhöht den Rechenaufwand, ohne Informationen hinzuzufügen.
    1. Trainieren Sie `VGG` mit einem Viertel der Breite (`arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))`, `lr=0.1`) auf Bildern, die Sie auf $56 \times 56$ beziehungsweise $84 \times 84$ skalieren. Geben Sie für jede Auflösung die räumlichen Abmessungen der Merkmalskarte am Eingang des vollvernetzten Kopfes an. Vergleichen Sie die Validierungsgenauigkeit und die Dauer einer Epoche mit dem Lauf bei $224 \times 224$ Pixeln.
    1. `GoogLeNet` kann in der vorliegenden Implementierung eine $28 \times 28$-Eingabe verarbeiten. Jeder Schritt mit Stride 2 nutzt Padding und halbiert so die Auflösung (aufgerundet). Berechnen Sie für jede der drei Inception-Stufen die räumliche Auflösung bei einer $28 \times 28$-Eingabe. Was berechnen die $3 \times 3$- und $5 \times 5$-Zweige in der letzten Stufe?

<!--en:f0cb83fb-->
1. **Woher kommen die Einsparungen?** ● Verwenden Sie die Zahlen für die einzelnen Schichten aus diesem Abschnitt und aus :numref:`tab_alexnet_cost`.
    1. Betrachten Sie VGG-11, `NiN` und `GoogLeNet` für ein einkanaliges $224 \times 224$-Bild und 10 Klassen. Schlüsseln Sie die Parameter und Multiplikationsadditionen nach Faltungsschichten und Klassifikationskopf auf.
    1. Spart `NiN` gegenüber VGG-11 mehr Parameter im Hauptteil oder im Kopf? Wie verhält es sich bei `GoogLeNet`? Beantworten Sie dieselben Fragen für die Multiplikationsadditionen. Geben Sie für jede Einsparung an, auf welche Entwurfsentscheidung sie zurückgeht.

<!--en:ab640df9-->
1. [code] **Inception-Ablation mit einem einzigen Zweig.** ● Bauen Sie eine Variante von `GoogLeNet`, deren Inception-Blöcke nur den $3 \times 3$-Zweig mit seiner $1 \times 1$-Reduktion behalten. Lassen Sie die $3 \times 3$-Faltung genauso viele Kanäle ausgeben wie der vollständige Block mit vier Zweigen. Wählen Sie die Breite der $1 \times 1$-Reduktion so, dass die Parameteranzahl jedes Blocks höchstens 10% vom Original abweicht. Trainieren Sie beide Netze auf Fashion-MNIST mit $96 \times 96$ Pixeln und vergleichen Sie die Validierungsgenauigkeit. Was sagt das Ergebnis über die Kombination verschiedener Kerngrößen innerhalb eines Blocks aus?

<!--en:432ec46c-->
[Diskussion](https://d2l.discourse.group/t/77)

<!--en:0d4deb7f-->
[Diskussion](https://d2l.discourse.group/t/78)

<!--en:f7aec69c-->
[Diskussion](https://d2l.discourse.group/t/277)

<!--en:fbf2271e-->
[Diskussion](https://d2l.discourse.group/t/18002)

<!--en:a4f10707-->
Blöcke, Flaschenhälse und Zweige in VGG, NiN und GoogLeNet<br>**wiederholbare $3\times 3$-Blöcke · Kanalmischung mit $1\times 1$-Faltungen · Global-Average-Pooling · Inception mit mehreren Zweigen**

<!--en:bd6fc5c8-->
::: {.slide title="Zwei strukturelle Schwächen beim schichtweisen Entwurf von AlexNet"}

<!--en:b8c6532e-->
[Motivation und Engpässe beim Entwurf]{.kicker}

<!--en:5399ca25-->
- **Einzeln gewählte Schichten:** Die schichtweise Wahl von $11 \times 11$-, $5 \times 5$- und $3 \times 3$-Faltungskernen liefert keinen systematischen Bauplan für Netze mit 20 oder mehr Schichten.

<!--en:1829027e-->
- **Vollvernetzter Kopf mit 43 Mio. Parametern:** Er enthält 92% der Gewichte von AlexNet und benötigt nur 5% seiner MACs.

<!--en:0d7a223e-->
- **Drei Lösungen:** **VGG** ($3 \times 3$-Blöcke), **NiN** ($1 \times 1$-Faltungen und Global-Average-Pooling) und **GoogLeNet** (mehrere Zweige mit $1 \times 1$-Flaschenhälsen in einem Netz aus Eingangsteil, Hauptteil und Kopf).

<!--en:58f86fd8-->
::: {.slide title="Warum mehrere kleine 3×3-Faltungen statt eines großen Faltungskerns?"}

<!--en:2a75b96a-->
[VGG und das rezeptive Feld pro FLOP]{.kicker}

<!--en:19b10211-->
- **Gleiches rezeptives Feld:** Zwei $3 \times 3$-Schichten erfassen $5 \times 5$, drei $3 \times 3$-Schichten erfassen $7 \times 7$ ($r = 2k+1$).

<!--en:5394399f-->
- **Weniger Gewichte und MACs:** Zwei $3 \times 3$-Schichten benötigen $18c^2$ statt $25c^2$ (**28% weniger**), drei $3 \times 3$-Schichten $27c^2$ statt $49c^2$ (**45% weniger**).

<!--en:0eb07bd0-->
- **Mehr Nichtlinearität:** Der Stapel enthält $k$ ReLU-Aktivierungen statt 1.

<!--en:229acbd5-->
::: {.slide title="Implementierung eines wiederverwendbaren VGG-Blocks" except="tensorflow"}

<!--en:d80dad6b-->
[VGG und der modulare Blockaufbau im Code]{.kicker}

<!--en:9808143d-->
Ein VGG-Block stapelt `num_convs` $3\times 3$-Faltungen mit Padding ($p=1$ erhält Höhe und Breite). Danach halbiert ein $2\times 2$-Max-Pooling mit Stride 2 beide räumlichen Dimensionen:

<!--en:59b4c445-->
::: {.slide title="Implementierung eines wiederverwendbaren VGG-Blocks" only="tensorflow"}

<!--en:d80dad6b-2-->
[VGG und der modulare Blockaufbau im Code]{.kicker}

<!--en:1a7057c8-->
Ein VGG-Block stapelt `num_convs` $3\times 3$-Faltungen mit Padding (`padding='same'` erhält Höhe und Breite). Danach halbiert ein $2\times 2$-Max-Pooling mit Stride 2 beide räumlichen Dimensionen:

<!--en:d51ddf7d-->
::: {.slide title="Netztopologie als Daten am Beispiel der VGG-Familie"}

<!--en:020c4f87-->
[VGG als parametrisierte Architektur]{.kicker}

<!--en:cbea8bf4-->
Ein `(num_convs, out_channels)`-Tupel legt jeden VGG-Block fest. Damit genügt eine einzige Klasse für die ganze Architekturfamilie. Die `arch`-Liste legt die gewünschte Architektur fest:

<!--en:3169cc16-->
- **Konfiguration von VGG-11:** `arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))` (8 Faltungsschichten + 3 vollständig verbundene Schichten = 11 Schichten).

<!--en:970c1bb9-->
- **Auflösung und Kanalzahl:** Fünf Pooling-Schritte mit Stride 2 halbieren die Auflösung von Stufe zu Stufe ($224$ $\to$ $112$ $\to$ $56$ $\to$ $28$ $\to$ $14$ $\to$ $7$). Zugleich verdoppelt sich die Kanalzahl ($64$ $\to$ $128$ $\to$ $256$ $\to$ $512$). So bleibt der Rechenaufwand $h w c^2$ pro Schicht über die Stufen hinweg annähernd konstant.

<!--en:202e9168-->
- **VGG-16 und VGG-19:** In `arch` ändern sich nur die Schichtzahlen der Blöcke auf `(2, 2, 3, 3, 3)` oder `(2, 2, 4, 4, 4)`. Der Klassencode bleibt gleich.

<!--en:8441c2d9-->
::: {.slide title="Tensorformen in VGG-11 und der Flaschenhals mit 103 Mio. Parametern im Kopf" layout="tall-output"}

<!--en:93979781-->
[Ressourcenbedarf von VGG]{.kicker}

<!--en:d2880844-->
Bei einer $224\times 224$-Eingabe zeigt sich, dass der Klassifikationskopf von VGG-11 trotz des modularen Faltungsteils noch größer ist als der von AlexNet:

<!--en:b49e37da-->
- **Faltungsteil:** 8 Schichten enthalten $9.2\text{M}$ Parameter (7%) und benötigen $7.43\text{G}$ Multiplikationsadditionen (98% des Rechenaufwands, $8\times$ so viel wie bei AlexNet).

<!--en:73437086-->
- **Vollvernetzter Kopf mit Vektoreingabe:** Die letzte Merkmalskarte der Größe $512 \times 7 \times 7$ ergibt einen Vektor mit $25{,}088$ Merkmalen. Die erste Gewichtsmatrix enthält daher $25{,}088 \times 4096 = 102.8\text{M}$ Gewichte (in den drei vollständig verbundenen Schichten zusammen sind es $119.6\text{M}$, also **93% der 128.8 Mio. Parameter von VGG-11**).

<!--en:87522563-->
::: {.slide title="Training von VGG-11 mit einem Viertel der Breite auf Fashion-MNIST"}

<!--en:2483d5f2-->
[Versuchsergebnisse für VGG]{.kicker}

<!--en:47015710-->
Mit einem Viertel der ursprünglichen Kanalzahlen, also `(16, 32, 64, 128, 128)`, sinkt der Rechenaufwand auf $512\text{M}$ Multiplikationsadditionen (die Hälfte von AlexNet). Das Training verläuft bei der maximalen Lernrate $\eta_0 = 0.1$ stabil:

<!--en:3ef4e79c-->
- **Genauigkeit und Rechenaufwand:** In 10 Epochen erreicht VGG mit einem Viertel der Breite **$\approx 93\%$** Validierungsgenauigkeit. Das sind bei halbem Rechenaufwand 2 Prozentpunkte mehr als bei AlexNet (VGG erreicht $91\text{--}92\%$ bei der maximalen Lernrate von AlexNet, $\eta_0 = 0.03$).

<!--en:ff585b8a-->
- **Der Kopf bleibt groß:** Im schmaleren Faltungsteil bleiben nur $0.6\text{M}$ Gewichte. Der vollvernetzte Kopf ($6272$ $\to$ $4096$ $\to$ $4096$ $\to$ $10$) enthält weiterhin **$42.5\text{M}$ der $43.1\text{M}$ Parameter (99%)**.

<!--en:ff1f6ff2-->
::: {.slide title="Pixelweise Kanalmischung durch 1×1-Faltungen in NiN"}

<!--en:5f7fef73-->
[Network in Network (NiN)]{.kicker}

<!--en:c9b91b19-->
- **Dasselbe MLP für jedes Pixel:** Auf jede räumliche $k \times k$-Faltung folgen zwei $1 \times 1$-Faltungen mit ReLU. Sie mischen die Kanäle an jeder Position $(x, y)$.

<!--en:a1d6cc2e-->
::: {.slide title="Global-Average-Pooling ersetzt den Kopf mit 43 Mio. Parametern"}

<!--en:cdcf1228-->
[Der parameterfreie Klassifikationskopf von NiN]{.kicker}

<!--en:dd7c2fa2-->
NiN projiziert den letzten Block direkt auf $K$ Klassenkanäle und mittelt jeden Kanal über alle Positionen. So entfällt die Umformung der Merkmalskarten in einen Vektor für große vollständig verbundene Schichten:

<!--en:a30a4c7e-->
- **Räumliche Aggregation ohne Parameter:** Der letzte NiN-Block gibt pro Klasse eine $h \times w$-Merkmalskarte aus ($K=10$). *Global-Average-Pooling* berechnet den Mittelwert jeder $5\times 5$-Karte und liefert die $10$ Klassen-Logits:
  $$z_k = \frac{1}{h w} \sum_{x=1}^h \sum_{y=1}^w X_{k, x, y}$$

<!--en:c3881a33-->
- **Eingebaute Translationsinvarianz:** Die Mittelung über alle Koordinaten $(x,y)$ macht die Klassen-Scores unabhängig von der Position des Objekts.

<!--en:a9d7a3bf-->
- **14× weniger Parameter:** Die Parameteranzahl sinkt von $46.8\text{M}$ bei AlexNet auf **$3.4\text{M}$** ($798\text{M}$ MACs).

<!--en:05cae7ad-->
::: {.slide title="Warum die letzten Logit-Karten keine ReLU vertragen"}

<!--en:ae734a52-->
[Eine Fehlerquelle bei Implementierung und Training von NiN]{.kicker}

<!--en:862014d0-->
Übergeben wir `num_classes` an den letzten `nin_block`, muss die ReLU nach der letzten $1\times 1$-Faltung entfallen, damit das Netz trainierbar bleibt:

<!--en:e84c07b7-->
- **Dauerhafter Ausfall einer Klasse:** Mit einer ReLU nach der letzten $1\times 1$-Schicht könnte ein Klassenkanal vollständig ausfallen. Würden alle Werte seiner $5\times 5$-Karte im gesamten Datensatz negativ, setzte die ReLU sie überall auf $0$. Das Logit $z_k = 0$ und der Gradient $\partial L / \partial W = 0$ wären dann identisch null. Die Klasse bliebe dauerhaft inaktiv.

<!--en:bd649a11-->
- **Training von NiN auf Fashion-MNIST:** Bei der maximalen Lernrate $\eta_0 = 0.03$ erreicht NiN $\approx 89\%$ Validierungsgenauigkeit (Trainingsverlust $\approx 0.30$, leichte Unteranpassung nach 10 Epochen ohne Überanpassung).

<!--en:d93000b9-->
::: {.slide title="Kontrollierte Ablation mit Global-Average-Pooling bei VGG"}

<!--en:b89d56ef-->
[Ablation des Kopfes bei VGG und PooledVGG]{.kicker}

<!--en:4895dc28-->
::: {.slide title="Kontrollierte Ablation mit Global-Average-Pooling bei VGG (Fortsetzung)"}

<!--en:b89d56ef-2-->
[Ablation des Kopfes bei VGG und PooledVGG]{.kicker}

<!--en:69982f19-->
Training von `VGG` mit vollvernetztem Kopf und maximaler Lernrate 0.03:

<!--en:c079cd65-->
::: {.slide title="Kontrollierte Ablation mit Global-Average-Pooling bei VGG (Ergebnis)"}

<!--en:b89d56ef-3-->
[Ablation des Kopfes bei VGG und PooledVGG]{.kicker}

<!--en:1953351b-->
Training von `PooledVGG` mit derselben Lernrate und anschließender Vergleich der endgültigen Validierungsgenauigkeiten beider Netze:

<!--en:8193bca6-->
::: {.slide title="GoogLeNet mit mehrskaligen parallelen Zweigen und Flaschenhälsen"}

<!--en:59573d68-->
[GoogLeNet und der Inception-Block]{.kicker}

<!--en:ef3a7052-->
In natürlichen Bildern erscheinen Objekte in ganz unterschiedlichen Größen. Der *Inception-Block* von GoogLeNet verarbeitet vier Zweige parallel und verkettet deren Ausgaben entlang der Kanalachse. So braucht er sich pro Schicht nicht auf eine einzige Kerngröße festzulegen:

<!--en:9026cd16-->
- **Vier räumliche Maßstäbe parallel:**
  1. $1\times 1$-Faltung ($c_1$ Kanäle) für pixelweise Kanalmerkmale.
  2. $1\times 1$-Reduktion $\to$ $3\times 3$-Faltung ($c_{2,1} \to c_{2,2}$ Kanäle).
  3. $1\times 1$-Reduktion $\to$ $5\times 5$-Faltung ($c_{3,1} \to c_{3,2}$ Kanäle).
  4. $3\times 3$-Max-Pooling mit Stride 1 $\to$ $1\times 1$-Faltung ($c_4$ Kanäle).

<!--en:87beead4-->
- **1×1-Flaschenhälse zur Dimensionsreduktion:** Sie projizieren die $c_{\textrm{i}}$ Eingabekanäle auf einen niedrigdimensionalen Unterraum, bevor die aufwendigen $3\times 3$- und $5\times 5$-Filter zum Einsatz kommen.

<!--en:beedde80-->
::: {.slide title="Implementierung des Inception-Blocks mit vier Zweigen"}

<!--en:e93e07a5-->
[Der Inception-Block im Code]{.kicker}

<!--en:39cf5573-->
Formerhaltendes Padding („same“) sorgt für gleich große Ausgaben aller vier Zweige. So lassen sie sich entlang der Kanalachse verketten:

<!--en:2e7f69b8-->
::: {.slide title="Wie 1×1-Flaschenhälse die Kosten räumlicher Filter um den Faktor zehn senken"}

<!--en:28a40e4d-->
[Rechenaufwand mit und ohne Flaschenhals]{.kicker}

<!--en:68a442e2-->
- **Direkter $5 \times 5$-Zweig ($192 \to 32$):** Er benötigt $25 \cdot 192 \cdot 32 = 153{,}600$ Gewichte ($120.4\textrm{M}$ MACs bei $28 \times 28$).

<!--en:5c6a6e7e-->
- **Mit $1 \times 1$-Flaschenhals ($192 \to 16 \to 32$):** Der Zweig benötigt $3{,}072 + 12{,}800 = 15{,}872$ Gewichte ($12.4\textrm{M}$ MACs). Parameteranzahl und Rechenaufwand **sinken um den Faktor $9.7\times$**.

<!--en:174728c3-->
::: {.slide title="Der Bauplan aus Eingangsteil, Hauptteil und Kopf"}

<!--en:d2d1987e-->
[Der Gesamtaufbau von GoogLeNet]{.kicker}

<!--en:764dd9cd-->
GoogLeNet etablierte einen dreiteiligen Bauplan, den ResNet, MobileNet, ConvNeXt und RegNet übernommen haben:

<!--en:4f9ebdce-->
- **Rasches Downsampling im Eingangsteil:** Eine $7\times 7$-Faltung mit Stride 2, Max-Pooling mit Stride 2, eine $1\times 1$-Faltung, eine $3\times 3$-Faltung und ein zweites Max-Pooling mit Stride 2 senken die räumliche Auflösung um den Faktor $8\times$. Dabei steigt die Kanalzahl auf 192.

<!--en:b3ed955f-->
- **Dreistufiger Inception-Hauptteil:** Die Stufen stapeln 2, 5 und 2 Inception-Blöcke. Dazwischen liegt Max-Pooling mit Stride 2 ($192$ $\to$ $480$ $\to$ $832$ $\to$ $1024$ Kanäle bei insgesamt 22 Schichten mit Gewichten).

<!--en:8bdd7497-->
- **Kopf mit Global-Average-Pooling:** Die Mittelwerte der $1024$ letzten Kanäle bilden einen Vektor der Länge 1024 für eine einzige lineare Schicht ($1024 \to 10$).

<!--en:3ae062cf-->
::: {.slide title="Training von GoogLeNet und Vergleich aller fünf Architekturen"}

<!--en:343fa0a9-->
[Empirischer Vergleich]{.kicker}

<!--en:ca5a43d2-->
Auf $96\times 96$ Pixel großen Fashion-MNIST-Bildern erreicht GoogLeNet bei $\eta_0 = 0.03$ dieselbe Validierungsgenauigkeit von 91% wie AlexNet und braucht $8\times$ weniger Parameter ($6.0\text{M}$ gegenüber $46.8\text{M}$):

<!--en:bb820883-->
| Modell | Eingabe | Parameter | MACs | Max. Lernrate $\eta_0$ | Validierungsgenauigkeit |

<!--en:f35a92ba-->
| LeNet (30 Epochen) | $28^2$ | 0.06 Mio. | 0.4 Mio. | 2.0 | 88–90% |

<!--en:9ddbff60-->
| AlexNet | $224^2$ | 46.8 Mio. | 938 Mio. | 0.03 | 91% |

<!--en:ac4046db-->
| VGG mit einem Viertel der Breite | $224^2$ | 43.1 Mio. | 512 Mio. | 0.1 / 0.03 | 93% / 91–92% |

<!--en:cda69361-->
| PooledVGG | $224^2$ | 0.6 Mio. | 470 Mio. | 0.03 | 89–90% |

<!--en:248691f0-->
| NiN | $224^2$ | 3.4 Mio. | 798 Mio. | 0.03 | 89% |

<!--en:a66d24a1-->
| GoogLeNet | $96^2$ | 6.0 Mio. | 276 Mio. | 0.03 | 91% |

<!--en:df4bc07a-->
::: {.slide title="Verständnisfragen zu Blockaufbau und Rechenaufwand von Flaschenhälsen"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:4b021d66-->
Das eigene Verständnis von Blöcken, Flaschenhälsen und Pooling-Köpfen prüfen:

<!--en:fb2aa204-->
1. **Gestapelte $3\times 3$-Faltungen oder eine $7\times 7$-Faltung:** Eine Stufe arbeitet mit $c = 256$ Kanälen. Wie viele Parameter benötigen drei gestapelte $3\times 3$-Faltungen im Vergleich zu einer einzelnen $7\times 7$-Faltung (ohne Bias-Terme)? Welches rezeptive Feld decken beide Varianten ab?

<!--en:b826971a-->
2. **Flaschenhals im $3\times 3$-Zweig von Inception:** Im ersten Inception-Block von GoogLeNet ($c_{\textrm{i}} = 192$) reduziert Zweig 2 die Kanalzahl ($192 \to 96$) mit einer $1\times 1$-Faltung. Danach erzeugt eine $3\times 3$-Faltung $128$ Kanäle. Um welchen Faktor senkt die $1\times 1$-Reduktion die Anzahl der Gewichte in diesem Zweig gegenüber einer direkten Abbildung $192 \to 128$ durch eine $3\times 3$-Faltung?

<!--en:557b98b6-->
**Lösungen:**

<!--en:fe0bdd62-->
1. Beide decken ein **rezeptives Feld der Größe $7\times 7$** ab. Drei $3\times 3$-Schichten benötigen $3 \cdot 9 \cdot 256^2 = 1{,}769{,}472$ Gewichte, gegenüber $49 \cdot 256^2 = 3{,}211{,}264$ bei einer $7\times 7$-Schicht. Sie sparen $1.44\text{M}$ Gewichte ($45\%$) und bringen zwei zusätzliche ReLU-Nichtlinearitäten ein.

<!--en:51ab2409-->
2. Direkte $3\times 3$-Faltung: $9 \cdot 192 \cdot 128 = 221{,}184$ Gewichte. Mit $1\times 1$-Flaschenhals: $192 \cdot 96 + 9 \cdot 96 \cdot 128 = 18{,}432 + 110{,}592 = 129{,}024$ Gewichte (**um den Faktor $1.71\times$ weniger**).

<!--en:1c83799f-->
::: {.slide title="Die Beiträge von VGG, NiN und GoogLeNet"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:6f02fc7f-->
- **VGG ($3 \times 3$-Blöcke):** Tupel der Form `(num_convs, c)` für jede Stufe parametrisieren die Netztopologie.

<!--en:7ca3dde1-->
- **NiN ($1 \times 1$ und Global-Average-Pooling):** Mischt die Kanäle an jedem Pixel und spart $>90\%$ der Gewichte, indem es vollvernetzte Köpfe durch Global-Average-Pooling ersetzt.

<!--en:a3997cb8-->
- **GoogLeNet (Eingangsteil, Hauptteil, Kopf):** Mehrskalige Inception-Zweige mit $1 \times 1$-Flaschenhälsen folgen dem allgemeinen dreiteiligen Bauplan für Grundnetze.
