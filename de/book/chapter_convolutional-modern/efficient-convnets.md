<!--en:d1b764e3-->
# Effiziente konvolutionäre Netze

<!--en:a3a1ebfa-->
Die bisher betrachteten Architekturen waren darauf ausgelegt, auf GPUs im Rechenzentrum Spitzenplätze in wissenschaftlichen Ranglisten für Genauigkeit zu erreichen. In der Praxis laufen sehr viele konvolutionäre Netze auf Smartphones, intelligenten Kameras, autonomen Fahrzeugen und eingebetteten Rechnern. Dort gelten feste Grenzen für Latenz, Speicherbedarf und Stromverbrauch. Überschreitet ein Modell eine dieser Grenzen, scheidet es für den Einsatz aus, unabhängig von seiner Genauigkeit im Benchmark. Betrachten wir ein Smartphone, das Kamerabilder in Echtzeit klassifiziert. VGG-16 benötigt für ein einziges Bild der Größe $224 \times 224$ bereits 15.3 Milliarden Multiplikationsadditionen. Bei 30 Bildern pro Sekunde sind das 460 Milliarden Multiplikationsadditionen pro Sekunde. Seine 138 Millionen Parameter in einfacher Genauigkeit belegen zudem 550 MB des Arbeitsspeichers, den sich alle Apps auf dem Gerät teilen. Wie können wir die Kosten eines konvolutionären Netzes um eine Größenordnung senken, und wie viel Genauigkeit kostet uns das?

<!--en:c3fd22ce-->
In diesem Abschnitt untersuchen wir drei grundlegende Ideen für den Entwurf effizienter konvolutionärer Netze. Die *Faktorisierung* zerlegt jede dichte Faltung in einen günstigen räumlichen Filter pro Kanal und eine $1 \times 1$-Faltung, die die Kanäle mischt. Bei *Skalierung und Suche* legen wir einen leistungsfähigen Grundblock fest und optimieren systematisch die Tiefen und Breiten der Stufen sowie die Eingabeauflösung. Bei der *strukturellen Reparametrisierung* trainieren wir eine Architektur mit mehreren Zweigen, die sich leicht optimieren lässt. Für den Einsatz fassen wir die Zweige algebraisch zu einer einzigen schnellen Faltung zusammen. Wir messen direkt, wie sich die Faktorisierung auf Genauigkeit und Laufzeit auswirkt. Dabei zeigt sich, dass die theoretische Anzahl der Multiplikationsadditionen die tatsächliche Geschwindigkeit nur bedingt erklärt. Ob die Faktorisierung mehr bringt als ein schmaleres dichtes Modell, hängt stark von der Größe der Aufgabe ab.

<!--en:df769c4e-->
## Netze mit tiefenweise separierbaren Faltungen

<!--en:76f7fabc-->
Dichte räumliche Faltungen machen den größten Teil des Rechenaufwands eines konvolutionären Netzes aus. Der direkteste Weg zu höherer Effizienz ist daher die Faktorisierung der Faltungsschicht selbst.

<!--en:03abecf8-->
### Tiefenweise separierbare Faltungen

<!--en:0423d65d-->
Der Aufwand einer gewöhnlichen dichten Faltung wächst proportional zu $c_\textrm{i} c_\textrm{o}$, weil jeder Ausgabekanal jeden Eingabekanal mit einem eigenen räumlichen Filter verarbeitet. In :numref:`subsec_resnext` hat die gruppierte Faltung (engl. grouped convolution) von ResNeXt diese vollständige Kopplung aufgehoben. Sie teilt die Kanäle in $g$ disjunkte Gruppen. Jede Ausgabegruppe greift nur auf die zugehörige Eingabegruppe zu. Dadurch sinkt sowohl die Anzahl der Gewichte als auch die der Multiplikationsadditionen um den Faktor $g$. Wie :numref:`fig_conv_groups` zeigt, erhält die zuvor dichte Verbindungsmatrix der Kanäle durch die Gruppierung eine Blockdiagonalstruktur.

<!--en:644d83f5-->
![Eingabekanäle, die jeder Ausgabekanal liest, bei 8 Eingabe- und 8 Ausgabekanälen. Jedes schattierte Feld steht für einen $k_\textrm{h}\times k_\textrm{w}$-Faltungskern. Bei $g$ Gruppen bleibt der Anteil $1/g$ der Kerne erhalten.](../img/conv-groups.svg)

<!--en:768dd916-->
Treiben wir die Gruppierung nun bis zum Grenzfall $g = c_\textrm{i} = c_\textrm{o}$. Jede Gruppe enthält dann einen einzigen Kanal. Diese *tiefenweise Faltung* (engl. depthwise convolution) wendet auf jeden Eingabekanal unabhängig einen räumlichen $k\times k$-Filter an, ohne Kanäle zu mischen. Sie ergänzt damit genau die $1\times 1$-Faltung (:numref:`subsec_1x1`), die an jedem Pixel alle Kanäle ohne räumliche Wechselwirkung mischt. Verketten wir eine tiefenweise $k\times k$-Faltung mit einer *punktweisen* $1\times 1$-Faltung, erhalten wir räumliche Filterung und Kanalmischung in zwei günstigen Schritten. Das ergibt eine *tiefenweise separierbare Faltung* (engl. depthwise-separable convolution) :cite:`chollet2017xception,howard2017mobilenet`. :numref:`fig_conv_depthwise` stellt diese zweistufige Faktorisierung der dichten Faltung gegenüber, die sie ersetzt.

<!--en:67555a41-->
![Drei Eingabe- und drei Ausgabekanäle mit $3\times 3$-Faltungskernen. Oben liest bei einer dichten Faltung jeder Ausgabekanal alle Eingabekanäle mit einem eigenen Stapel von Faltungskernen. Unten filtert eine tiefenweise Faltung jeden Kanal separat, und zwar einmal für alle Ausgabekanäle. Eine punktweise Faltung mischt die Kanäle mit einem Stapel von $1\times 1$-Faltungskernen pro Ausgabekanal.](../img/arch-conv-depthwise.svg)

<!--en:43a201dd-->
Berechnen wir die Einsparung. Bei einer Ausgabe der Größe $h\times w$ und einem $k\times k$-Faltungskern benötigt eine dichte Faltung $hw\cdot c_\textrm{i} c_\textrm{o} k^2$ Multiplikationsadditionen. Das tiefenweise separierbare Paar benötigt $hw\cdot c_\textrm{i} k^2$ für die räumliche Filterung und $hw\cdot c_\textrm{i} c_\textrm{o}$ für die punktweise Mischung. Das Verhältnis beträgt

<!--en:b265e527-->
und genau dasselbe Verhältnis gilt für die Parameteranzahl. In :numref:`fig_conv_depthwise` ($c_\textrm{i} = c_\textrm{o} = k = 3$) benötigt die dichte Schicht $3 \cdot 27 = 81$ Gewichte, das separierbare Paar hingegen $27 + 9 = 36$ (Verhältnis $1/3 + 1/9 = 4/9$). Ein weiterer Ausgabekanal kostet in der dichten Schicht 27 zusätzliche Gewichte, in der separierbaren nur 3. Für große $c_\textrm{o}$ und $k = 3$ nähert sich das Verhältnis dem Wert $1/9$. Damit sinken sowohl die Anzahl der Gewichte als auch der Rechenaufwand fast um den Faktor neun.

<!--en:363f2411-->
Diese Einsparungen haben ihren Preis. Sie beruhen auf einer strukturellen Nebenbedingung für den effektiven Filtertensor. Das separierbare Paar wendet einen einzigen räumlichen Filter $\mathbf{D}_c$ auf den Eingabekanal $c$ an. Anschließend kombiniert es die gefilterten Merkmalskarten linear mit skalaren punktweisen Gewichten $P_{o,c}$. Das entspricht einer dichten Faltung, deren Faltungskerne die faktorisierte Form $\mathbf{K}_{o,c} = P_{o,c}\,\mathbf{D}_c$ haben müssen. Für den Eingabekanal $c$ verwenden also alle Ausgabekanäle bis auf einen skalaren Faktor denselben räumlichen Filter. Eine dichte Faltung kann dagegen einen beliebigen, unabhängigen $k\times k$-Filter für jedes Paar $(o, c)$ lernen. Effiziente Netze nehmen diese Rang-1-Beschränkung in Kauf und nutzen die frei werdenden Ressourcen für tiefere oder breitere Stufen.

<!--en:d08249f7-->
Überprüfen wir das Verhältnis der Parameteranzahlen im Code für eine $3\times 3$-Schicht ohne Bias, die 128 Kanäle auf 128 Kanäle abbildet. Dafür sagt :eqref:`eq_depthwise_sep_ratio` eine Verringerung um den Faktor $(1/128 + 1/9)^{-1} \approx 8.4$ voraus.

<!--en:d3afba07-->
Keras-Schichten erwarten die Kanalachse der Eingabe an letzter Stelle. Daher hat `X` im folgenden Code die Form $(1, 32, 32, c_\textrm{i})$.

<!--en:b1dd828c-->
Flax-Schichten erwarten die Kanalachse der Eingabe an letzter Stelle. Daher hat `X` im folgenden Code die Form $(1, 32, 32, c_\textrm{i})$.

<!--en:9f0aa01c-->
Die dichte Schicht benötigt 147 456 Gewichte, das separierbare Paar nur 17 536. Das entspricht einem Faktor von rund $8.4\times$.

<!--en:55a23717-->
In Gluon macht `groups=c_i` aus `nn.Conv2D` eine tiefenweise Faltung.

<!--en:46aab459-->
In PyTorch macht `groups=c_i` aus `nn.Conv2d` eine tiefenweise Faltung. Jeder Wert von `groups`, der beide Kanalanzahlen teilt, ergibt eine gruppierte Faltung.

<!--en:02da8b14-->
Keras bietet für die tiefenweise Faltung eine eigene Schicht `DepthwiseConv2D`. Gruppierte Faltungen verwenden das Argument `groups` von `Conv2D`.

<!--en:87f5ad0a-->
In Flax macht `feature_group_count=c_i` aus `nnx.Conv` eine tiefenweise Faltung. Das Argument gibt die Anzahl der Gruppen $g$ an.

<!--en:692d2590-->
### MobileNet

<!--en:2e418b6b-->
*MobileNet* :cite:`howard2017mobilenet` baut aus tiefenweise separierbaren Faltungen ein vollständiges Basisnetz für Mobilgeräte auf. Auf eine einzelne dichte Faltung im Eingangsteil folgen ausschließlich Schichten mit dem Block aus :numref:`fig_dws_block`. Er besteht aus einer tiefenweisen $3 \times 3$-Faltung und einer anschließenden punktweisen $1 \times 1$-Faltung, jeweils mit Batchnormalisierung und ReLU. Tiefenweise Faltungen mit Stride 2 übernehmen direkt das räumliche Downsampling, sodass separate Pooling-Schichten entfallen. Das Modell endet wie NiN mit Global-Average-Pooling und einer einzigen linearen Schicht (:numref:`sec_nin`). Auf ImageNet kam MobileNet bis auf einen Prozentpunkt an VGG-16 heran (70.6% gegenüber 71.5% Top-1-Genauigkeit). Dafür benötigte es nur 0.57 Milliarden Multiplikationsadditionen und 4.2 Millionen Parameter, also $27\times$ weniger Rechenoperationen und $33\times$ weniger Parameter.

<!--en:7d025d59-->
![Tiefenweise separierbarer Block mit $c_\textrm{i}$ Eingabe- und $c_\textrm{o}$ Ausgabekanälen. Die tiefenweise $3 \times 3$-Faltung filtert jeden Eingabekanal separat. Die punktweise $1 \times 1$-Faltung mischt die Kanäle an jeder Position. Auf jede Faltung folgen Batchnormalisierung und ReLU.](../img/arch-dws-block.svg)

<!--en:4e6314e8-->
### Der invertierte Flaschenhals

<!--en:8f1af9d0-->
Erinnern wir uns an den klassischen Flaschenhals (engl. bottleneck) in ResNet oder ResNeXt (:numref:`subsec_resnext`). Er *komprimiert* die Kanäle vor der räumlichen Filterung. Eine $1 \times 1$-Schicht bildet eine breite Darstellung auf einen schmalen Unterraum ab. So bleibt die aufwendige dichte $3 \times 3$-Faltung bezahlbar. Eine zweite $1 \times 1$-Schicht stellt die ursprüngliche Breite wieder her. Arbeitet die $3 \times 3$-Schicht jedoch *tiefenweise*, kehrt sich diese Überlegung um. Eine tiefenweise Faltung ist selbst auf breiten Merkmalskarten günstig. Da sie pro Kanal nur einen räumlichen Filter lernt, profitiert sie sogar von einer großen Breite.

<!--en:a8fa9a28-->
*MobileNetV2* :cite:`sandler2018mobilenetv2` *invertiert* daher den Flaschenhals. Der Residualpfad bleibt schmal, zum Beispiel bei 24 Kanälen. Im Block erweitert eine $1 \times 1$-Faltung die Darstellung auf das Sechsfache, also auf 144 Kanäle. Eine tiefenweise $3 \times 3$-Faltung filtert in diesem erweiterten Raum, und eine lineare $1 \times 1$-Projektion komprimiert die Darstellung wieder auf 24 Kanäle. Entscheidend ist, dass auf die letzte Projektion keine ReLU folgt. An einem sehr schmalen Flaschenhals setzt ReLU negative Koordinaten auf null. Die dabei verlorene Information können spätere Schichten nicht wiederherstellen. :numref:`fig_inverted_bottleneck` stellt den klassischen und den invertierten Flaschenhals gegenüber.

<!--en:e017f202-->
![Bottleneck-Block von ResNet (links) und invertierter Flaschenhals von MobileNetV2 (rechts), jeweils mit der Eingabe unten. Der Flaschenhals verengt 256 Kanäle auf 64 für eine dichte $3 \times 3$-Faltung und erweitert sie anschließend wieder. Der invertierte Flaschenhals erweitert 24 Kanäle auf 144 für eine tiefenweise $3 \times 3$-Faltung, die einen Filter pro Kanal lernt. Anschließend projiziert er ohne ReLU auf 24 Kanäle zurück. Die Residualverbindung verbindet die schmalen Enden.](../img/arch-inverted-bottleneck.svg)

<!--en:9fe940b3-->
Da die Sprungverbindung die schmalen Enden und nicht die breiten inneren Darstellungen verbindet, müssen zwischen den Blöcken nur kompakte Tensoren im Speicher bleiben. Das ist ein großer Vorteil auf Mobilprozessoren, wo der Arbeitsspeicher für Aktivierungen oft knapper ist als der Flashspeicher für Gewichte. Dieselbe Anordnung mit invertiertem Flaschenhals begegnet uns bei ConvNeXt wieder, dort mit einem größeren tiefenweisen $7 \times 7$-Faltungskern am Anfang (:numref:`sec_convnext`).

<!--en:2137ad92-->
### Ein Mini-MobileNet

<!--en:92cf26f6-->
Um die tiefenweise separierbare Faktorisierung selbst auf Fashion-MNIST zu erproben, bauen wir ein kompaktes Netz nach dem Vorbild von MobileNetV1. Mit dem einfachen Block aus :numref:`fig_dws_block` bleibt die Faktorisierung im Mittelpunkt. Da auf jede Faltung eine Batchnormalisierung folgt, kommen alle Faltungen ohne Bias aus. Für die Gewichte aller Faltungen und linearen Schichten verwenden wir die He-Initialisierung (:numref:`sec_numerical_stability`).

<!--en:27895821-->
Wie bei VGG (:numref:`sec_vgg`) genügt zur Beschreibung der Architektur eine Liste von Paaren der Form `(out_channels, stride)`. Eine $3 \times 3$-Faltung mit Stride 2 im Eingangsteil bildet die einkanalige Eingabe auf 32 Kanäle mit halber Auflösung ab. In den folgenden sieben tiefenweise separierbaren Blöcken verdoppelt sich dreimal die Kanalanzahl, während sich jeweils die räumliche Auflösung halbiert. Global-Average-Pooling und ein linearer Kopf liefern die Logits für die zehn Klassen.

<!--en:2ce722e9-->
Hier verwenden wir außerdem SGD mit Momentum 0.9 als Optimierer. SGD mit Momentum bildet einen exponentiell gewichteten gleitenden Mittelwert früherer Gradienten. Dadurch erreicht die effektive Schrittweite bei gleicher Lernrate bis zum Zehnfachen eines gewöhnlichen SGD-Schritts (:numref:`sec_momentum`). Wegen dieses Wechsels des Optimierers sollten Sie die hier gemessenen Genauigkeiten nicht direkt mit den ResNet-18-Läufen mit gewöhnlichem SGD aus :numref:`sec_resnet` vergleichen. Wie zuvor sinkt die Lernrate entlang einer halben Kosinusperiode (:numref:`sec_alexnet`) über die `num_steps` Trainingsschritte stetig auf null (4690 Minibatches in 10 Epochen). Dazu verwenden wir in jedem Framework den integrierten Kosinus-Zeitplan ohne Aufwärmphase.

<!--en:ba2d0b68-->
Zählen wir die trainierbaren Parameter von `MiniMobileNet` und vergleichen wir ihre Anzahl mit der eines dichten Gegenstücks. Wir speichern `num_params` zur Wiederverwendung in späteren Abschnitten in `d2l`.

<!--en:3fc40ed4-->
`MiniMobileNet` enthält 542 474 trainierbare Parameter. Ersetzten wir jeden separierbaren Block durch eine gewöhnliche dichte $3 \times 3$-Faltung mit denselben Ein- und Ausgabebreiten, bräuchten wir allein in den sieben Blöcken $9 \sum_l c_{\textrm{i},l}\, c_{\textrm{o},l} = 4{,}663{,}296$ Gewichte. Das sind $8.8\times$ so viele wie die 530 528 Gewichte unserer separierbaren Schichten. Bei einer Kanalbreite von 512 nähert sich der Faktor bereits der asymptotischen Obergrenze von $9\times$.

<!--en:99381c15-->
### Rechenaufwand, Genauigkeit und Laufzeit

<!--en:e9057bcb-->
Wie viel Genauigkeit kostet die Faktorisierung, und wie viel Laufzeit spart sie tatsächlich? Dazu bauen wir zunächst `VGGSmall`, ein dichtes Netz nach dem Vorbild von VGG (:numref:`sec_vgg`) mit ungefähr derselben *Parameteranzahl* wie `MiniMobileNet`. Da jede dichte $3 \times 3$-Schicht etwa neunmal so viele Gewichte benötigt wie ein separierbarer Block, erreicht `VGGSmall` bei gleichem Parameterbudget höchstens 128 Kanäle, während `MiniMobileNet` auf 512 kommt. Beide Modelle verwenden Batchnormalisierung, He-Initialisierung, einen Kopf mit Global-Average-Pooling und genau denselben Trainingsplan. Beachten Sie, dass sich bei gleicher Parameteranzahl sowohl der Schichttyp als auch die Kanalbreite ändern.

<!--en:42b473c7-->
Zählen wir vor dem Training die Multiplikationsadditionen für Eingaben der Größe $96 \times 96$ genau. Bei „same“-Padding und Stride $s$ ergibt eine $h \times h$-Eingabe eine Ausgabe mit Seitenlänge $\lceil h/s \rceil$. Eine gruppierte $k \times k$-Faltung mit $g$ Gruppen benötigt $k^2 c_\textrm{i} / g$ Multiplikationsadditionen pro Ausgabeelement.

<!--en:9bb59df6-->
Trotz vergleichbarer Parameteranzahl benötigt `VGGSmall` pro Bild 385 Millionen Multiplikationsadditionen, `MiniMobileNet` dagegen nur 50 Millionen, also $7.7\times$ weniger Rechenoperationen. Der Unterschied liegt darin, *wo* die Modelle ihre Gewichte einsetzen. `VGGSmall` wendet dichte Filter auf hochauflösende Merkmalskarten von $96 \times 96$ bis $24 \times 24$ an. Dagegen liegen fast drei Viertel der Gewichte von `MiniMobileNet` in den letzten beiden punktweisen Schichten, die auf kleinen $6 \times 6$-Gittern arbeiten.

<!--en:f0b54cc4-->
Für den Einsatz auf Endgeräten ist jedoch der Vergleich bei *gleichem Rechenaufwand* entscheidend. Damit `VGGSmall` wie `MiniMobileNet` mit 50 Millionen Multiplikationsadditionen auskommt, teilen wir jede Kanalanzahl durch etwa 2.8. Das ergibt `(11, 23, 46, 46)` (oben `narrow`). Es bleiben nur 76 000 Parameter, ein Siebtel der Anzahl der Gewichte von `MiniMobileNet`. Trainieren wir alle drei Modelle für 10 Epochen auf Fashion-MNIST mit einer Auflösung von $96 \times 96$.

<!--en:5877a09f-->
Die Anzahl der Operationen erfasst den Rechenaufwand. Um ihre tatsächliche Latenz zu ermitteln, messen wir Vorwärtsdurchläufe bei der Inferenz auf der GPU mit Batches von 128 Bildern. Wir schalten einen Aufwärmdurchlauf vor und synchronisieren das Gerät, bevor wir die Laufzeit ablesen (:numref:`subsec_async_dispatch`).

<!--en:65b3d15b-->
:numref:`tab_dws_vs_dense` fasst die Ergebnisse mehrerer Läufe für alle drei Modelle zusammen. Neben der Validierungsgenauigkeit steht die Latenz des Vorwärtsdurchlaufs, jeweils bezogen auf `MiniMobileNet`.

<!--en:afc4bc70-->
:Mini-MobileNet im Vergleich mit zwei dichten Netzen nach dem Vorbild von VGG. Das erste hat ungefähr gleich viele Parameter, das zweite ist schmaler und benötigt ungefähr gleich viele Multiplikationsadditionen (Fashion-MNIST mit $96 \times 96$, 10 Epochen, Validierungsgenauigkeit und relative Laufzeit des Vorwärtsdurchlaufs über mehrere Läufe).

<!--en:3f3239aa-->
| Modell | Parameter | Multiplikationsadditionen | Validierungsgenauigkeit | Relative Laufzeit des Vorwärtsdurchlaufs |

<!--en:5accfe08-->
| Mini-MobileNet | 542 474 | 50.3 Mio. | 91–92% | 1 |

<!--en:84a10f56-->
| Nach VGG | 583 594 | 384.9 Mio. | 93.5–94% | 2–3.5 |

<!--en:e0caa43a-->
| Nach VGG, schmal | 75 854 | 49.1 Mio. | 93% | 0.5–0.8 |

<!--en:a54a2084-->
Aus :numref:`tab_dws_vs_dense` ergeben sich drei Erkenntnisse.

<!--en:47dec844-->
1. **Die gleiche Parameteranzahl kann sehr unterschiedlichen Rechenaufwand bedeuten.** Bei rund 550 000 Parametern übertrifft `VGGSmall` das Modell `MiniMobileNet` um etwa 2 Prozentpunkte, benötigt aber $7.7\times$ so viele Multiplikationsadditionen. Es behält in der ersten Stufe die volle Auflösung von $96 \times 96$ bei und lernt dichte Faltungskerne ohne Einschränkungen.

<!--en:350d827e-->
2. **Bei gleichem Rechenaufwand und kleinem Datensatz bringt eine geringere Breite mehr als die Faktorisierung.** Verkleinern wir `VGGSmall` auf 50 Millionen Multiplikationsadditionen (`VGG-style, narrow`, 76 000 Parameter), übertrifft es `MiniMobileNet` um rund einen Prozentpunkt *und* läuft schneller. Nach :eqref:`eq_depthwise_sep_ratio` lohnt sich die tiefenweise Faktorisierung vor allem bei vielen Kanälen (Einsparung um den Faktor $5\times$ bei $c = 11$ gegenüber fast $9\times$ bei $c = 512$). Für die 10 Klassen von Fashion-MNIST reichen schmale dichte Schichten gut aus. ImageNet mit 1000 Klassen braucht dagegen breite Stufen. Hier würde ein schmaleres dichtes Netz zu viel Kapazität verlieren.

<!--en:1e9eb4dd-->
3. **Die Anzahl der Multiplikationsadditionen sagt wenig über die tatsächliche GPU-Latenz aus.** Bei $7.7\times$ weniger Rechenoperationen steigt die Geschwindigkeit nur um den Faktor $2\text{--}3.5\times$. Eine dichte $3 \times 3$-Schicht führt pro gelesenem Eingabewert $9c_\textrm{o}$ Multiplikationsadditionen aus und erreicht damit eine hohe arithmetische Intensität. Eine tiefenweise $3 \times 3$-Schicht schafft nur 9 Multiplikationsadditionen pro gelesenem Eingabewert. Hier begrenzt die Speicherbandbreite die Geschwindigkeit der GPU.

<!--en:cd448cb3-->
## Skalierung und Architektursuche

<!--en:a3ca54f7-->
Bei MobileNet ist ein effizienter Grundblock vorgegeben. Die Breiten der Stufen werden von Hand gewählt. Daraus ergeben sich zwei Fragen. Wie sollten wir eine Grundarchitektur vergrößern, wenn mehr Rechenleistung zur Verfügung steht? Und kann eine automatische Optimierung Breiten, Tiefen und Auflösungen der Stufen besser wählen, als es durch Ausprobieren von Hand gelingt?

<!--en:af2c2a66-->
### Kombinierte Skalierung

<!--en:be09a483-->
Betrachten wir den Rechenaufwand, um die Frage der Skalierung zu klären. Er wächst bei einem konvolutionären Netz linear mit der Tiefe, quadratisch mit der Kanalbreite (über $c_\textrm{i} c_\textrm{o}$) und quadratisch mit der Eingabeauflösung (über $hw$). Vergrößern wir nur eine Dimension, nimmt der zusätzliche Nutzen schnell ab. Einem sehr tiefen Modell fehlen bei winzigen Vorschaubildern die feinen Details zur Analyse. Hochauflösende Bilder erfordern dagegen mehr Schichten, damit die rezeptiven Felder große Objekte erfassen können. Die *kombinierte Skalierung* (engl. compound scaling) :cite:`tan2019efficientnet` vergrößert alle drei Dimensionen gemeinsam. Jeder Schritt multipliziert die Tiefe mit 1.2, die Breite mit 1.1 und die Auflösung mit 1.15. Damit verdoppelt sich der gesamte Rechenaufwand annähernd ($1.2 \cdot 1.1^2 \cdot 1.15^2 \approx 1.92$). Eine einzige ganze Zahl legt so eine ganze Modellfamilie fest. Drei Schritte ergeben ein Netz mit $1.7\times$ so vielen Schichten und $1.3\times$ so vielen Kanälen auf Bildern mit $1.5\times$ so großer Seitenlänge. Es benötigt ungefähr $7\times$ so viele Rechenoperationen.

<!--en:2c425c8f-->
### Architektursuche

<!--en:147c0036-->
Für den Entwurf des Basisnetzes nutzte EfficientNet die *neuronale Architektursuche* (engl. neural architecture search, NAS). Sie untersuchte Folgen von Blöcken mit invertiertem Flaschenhals und optimierte eine Zielfunktion, die Genauigkeit und Anzahl der Multiplikationsadditionen gegeneinander abwog. Wie unsere GPU-Laufzeitmessung gezeigt hat, kann die theoretische Anzahl der Multiplikationsadditionen bei der Optimierung für reale Hardware in die Irre führen. *EfficientNetV2* :cite:`tan2021efficientnetv2` ersetzte diese Anzahl in der Zielfunktion durch die gemessene Trainingsgeschwindigkeit und die Parameteranzahl. Bemerkenswerterweise ersetzte die Suche in frühen, hochauflösenden Stufen die tiefenweisen Blöcke automatisch durch gewöhnliche dichte $3 \times 3$-Faltungen. Trotz des höheren Rechenaufwands laufen diese auf Beschleunigern schneller. In beiden Modellgenerationen entwarfen Menschen die Grundbausteine. Die automatische Suche legte für jede Stufe Breite, Tiefe und Auflösung fest.

<!--en:64310170-->
### Squeeze-and-Excitation-Gates

<!--en:41fc5d22-->
Die Blöcke mit invertiertem Flaschenhals, die EfficientNet bei der Suche berücksichtigt, enthalten direkt nach der tiefenweisen Faltung ein zusätzliches kleines Modul, ein *Squeeze-and-Excitation*-Gate (SE-Gate) :cite:`Hu.Shen.Sun.2018`. Eine Faltung fasst Informationen nur innerhalb eines lokalen räumlichen Fensters zusammen. Ein SE-Gate erlaubt dem Netz dagegen, ganze Merkmalskanäle anhand des globalen Bildkontexts dynamisch neu zu gewichten. Es *verdichtet* jeden der $c$ Kanäle durch Global-Average-Pooling zu einem Skalar. Die $c$-dimensionale Zusammenfassung durchläuft ein kleines zweischichtiges MLP mit Flaschenhals (verborgene Breite $c/r$ mit ReLU, Ausgabebreite $c$ mit Sigmoidfunktion, die *Excitation*). Anschließend skaliert das Modul jeden Kanal der Merkmalskarte mit seinem vorhergesagten Gate-Wert aus $(0, 1)$. Beim Reduktionsfaktor $r$ kommen durch das MLP nur etwa $2c^2/r$ Parameter hinzu. Es verarbeitet pro Bild nur einen einzelnen Vektor statt jedes einzelnen Pixels. Seine zusätzlichen Rechenkosten sind daher vernachlässigbar. SE wirkt als einfacher kanalweiser Attention-Mechanismus und nimmt damit die allgemeinen Attention-Schichten aus :numref:`chap_attention` vorweg.

<!--en:b40dde8f-->
SE-Gates verbessern die Genauigkeit zuverlässig bei nahezu unverändertem Rechenaufwand. Daher finden sie sich überall in MobileNetV3 und RegNetY (:numref:`sec_cnn-design`, wo Aufgabe :exref:`ex_cnn-design-regnety` RegNet um SE erweitert). Ein SE-Block lässt sich in wenigen Zeilen implementieren. Auf globales Pooling folgen zwei lineare Schichten und eine kanalweise Neuskalierung.

<!--en:d04b00ef-->
Für $c = 32$ und $r = 4$ kommen durch das SE-Modul nur $2 \cdot 32^2 / 4 = 512$ Gewichte und 40 Bias-Terme hinzu. Die Form des Tensors bleibt exakt erhalten. Das Modul lässt sich daher unmittelbar in jeden Block einsetzen.

<!--en:3816b7ca-->
## Strukturelle Reparametrisierung

<!--en:f5df6932-->
### Der RepVGG-Block

<!--en:a63a231f-->
ResNet (:numref:`sec_resnet`) hat gezeigt, dass sich Blöcke mit mehreren Zweigen und Sprungverbindungen erheblich leichter optimieren lassen als einfache Folgen von Schichten. Bei der Inferenz verursachen parallele Zweige jedoch zusätzlichen Aufwand. Jeder Zweig muss separat starten, und seine Ausgabe muss bis zur Summation im Speicher bleiben. Dadurch sinkt die Hardwareauslastung. Eine einfache Folge von $3 \times 3$-Faltungen eignet sich dagegen ideal für Beschleuniger. Pro Schicht genügt eine hochoptimierte Rechenroutine für die dichte Faltung, und nur ein Aktivierungspuffer muss im Speicher bleiben. *RepVGG* :cite:`ding2021repvgg` verbindet beide Vorteile, indem es für Training und Inferenz unterschiedliche Topologien nutzt. Im Training summiert jede Schicht vor der ReLU drei parallele Zweige, eine $3 \times 3$-Faltung, eine $1 \times 1$-Faltung und eine Identitätsabbildung, jeweils mit eigener Batchnormalisierung (:numref:`fig_repvgg_reparam`). Nach dem Training ist jeder Zweig eine affine Abbildung seiner Eingabe. Die drei Zweige lassen sich daher algebraisch *zu einer einzigen $3 \times 3$-Faltung fusionieren*, die exakt dieselbe Funktion berechnet. Wir trainieren ein Residualnetz mit mehreren Zweigen und setzen eine schnelle, einfache Folge von VGG-Schichten ein.

<!--en:224c6169-->
![Strukturelle Reparametrisierung. Im Training summiert jede RepVGG-Schicht drei Zweige, eine $3 \times 3$-Faltung, eine $1 \times 1$-Faltung und eine Identitätsabbildung, jeweils mit Batchnormalisierung. Für die Inferenz werden die Zweige zu einer $3 \times 3$-Faltung mit Bias zusammengefasst. Beide Netze berechnen dieselbe Funktion.](../img/arch-repvgg-reparam.svg)

<!--en:0eb460b2-->
### Fusion der Zweige

<!--en:cb473a65-->
Um alle drei Zweige zu einer Faltung zusammenzufassen, benötigen wir drei einfache Schritte aus der linearen Algebra. Erstens ist die Batchnormalisierung im Inferenzmodus pro Kanal eine feste affine Abbildung. Sie subtrahiert den gleitenden Mittelwert $\mu_o$, dividiert durch $\sqrt{\sigma_o^2 + \epsilon}$, multipliziert mit $\gamma_o$ und addiert $\beta_o$. Rechnen wir diese affine Abbildung direkt in die vorangehende Faltung mit Faltungskern $\mathbf{W}$ und Bias $b_o$ ein (indiziert nach Ausgabekanal $o$), erhalten wir einen äquivalenten fusionierten Faltungskern und den zugehörigen Bias:

<!--en:cde7613e-->
Der Faltungskern jedes Ausgabekanals wird mit dessen Normalisierungsfaktor skaliert. Der fusionierte Bias enthält alle von der Eingabe unabhängigen Verschiebungen (z. B. ergeben $\gamma_o = 2$, $\sigma_o^2 + \epsilon = 16$, $\mu_o = 1$, $\beta_o = 1$ und $b_o = 0$ den Faltungskern $\hat{\mathbf{W}}_o = \mathbf{W}_o / 2$ und den Bias $\hat{b}_o = 1 - 0.5 = 0.5$). Da die Faltungen von `RepVGGBlock` keinen Bias haben, implementiert `fuse_bn` im folgenden Code den Fall $b_o = 0$. Dieses Einrechnen der Batchnormalisierung ist allgemein anwendbar. Inferenzcompiler nutzen es routinemäßig für beliebige Paare der Form `Conv + BN`.

<!--en:3a411048-->
Zweitens drücken wir die beiden übrigen Zweige als $3 \times 3$-Faltungen aus. Aus einem $1 \times 1$-Faltungskern entsteht ein $3 \times 3$-Faltungskern, indem wir das zentrale Element mit einem ein Pixel breiten Nullrand umgeben. Die Identitätsabbildung entspricht einer $3 \times 3$-Faltung, deren Faltungskern $\mathbf{W}^{\textrm{id}}_{o,i}$ für $i = o$ in der Mitte den Wert 1 und sonst überall den Wert 0 hat. Drittens nutzen wir die Linearität der Faltung. Die Summe der Ausgaben dreier paralleler Faltungen derselben Eingabe entspricht genau einer einzigen Faltung. Ihr Faltungskern ist die Summe der drei umgerechneten Faltungskerne, ihr Bias die Summe der drei umgerechneten Bias-Terme (:numref:`fig_repvgg_kernels`).

<!--en:888fb74f-->
![Faltungskerne eines Ausgabekanals $o$ für den Eingabekanal $i = o$, nachdem die Batchnormalisierung in jeden Zweig eingerechnet wurde (Beispielwerte, der mittlere Eintrag des Identitätskerns beträgt danach $\gamma_o/\sqrt{\sigma_o^2+\epsilon}$ statt 1). Der mit Nullen erweiterte $1 \times 1$-Faltungskern und der Identitätskern sind nur in der Mitte ungleich null. Daher unterscheidet sich der fusionierte Faltungskern nur dort vom $3 \times 3$-Zweig. Für $i \neq o$ ist der Identitätskern null.](../img/arch-repvgg-kernels.svg)

<!--en:2359a2f7-->
Da die Fusion auf den gleitenden Statistiken des Inferenzmodus beruht, prüfen wir `fuse_block` gründlich. Zunächst schicken wir mehrere Trainingsbatches (Mittelwert 1, Standardabweichung

<!--en:eefd35e9-->
2) durch `RepVGGBlock`, damit die gleitenden Statistiken von 0 und 1 abweichen. Wir wählen $\gamma_o$ und $\beta_o$ zufällig, damit jeder Term in :eqref:`eq_bn_fold` zum Tragen kommt. Anschließend bestimmen wir auf neuen Daten die maximale absolute Abweichung zwischen dem verzweigten Block im Inferenzmodus und der einzelnen fusionierten Faltung.

<!--en:bb4f4ead-->
Auf neueren NVIDIA-GPUs führt TensorFlow float32-Faltungen standardmäßig im TF32-Format mit reduzierter Genauigkeit aus (:numref:`sec_numerics`). Wir schalten TF32 aus, damit wir den Effekt der Fusion unabhängig von der Rundung bei der Faltung messen.

<!--en:1dc72969-->
Die maximale Abweichung liegt in der Größenordnung von $10^{-6}$ und beruht allein auf Rundungsfehlern der Gleitkommarechnung mit einfacher Genauigkeit. Das bestätigt die exakte funktionale Äquivalenz. Die Anzahl der Gewichte sinkt zwar nur um ein Zehntel (von $10c^2$ auf $9c^2$), die Ausführung vereinfacht sich jedoch erheblich. Eine dichte Faltung und ein einziger Aktivierungstensor im Speicher ersetzen zwei Faltungen, drei Normalisierungen, zwei Additionen und drei gleichzeitig benötigte Merkmalspuffer.

<!--en:29e6fc49-->
### Quantisierung nach der Fusion

<!--en:d7545c91-->
Beim praktischen Einsatz tritt ein Problem auf, das bei der algebraischen Betrachtung mit Gleitkommazahlen verborgen bleibt. Um Energie und Speicherbandbreite zu sparen, quantisieren Beschleuniger auf Endgeräten Gewichte und Aktivierungen häufig auf 8-Bit-Ganzzahlen. Dabei bilden sie den Wertebereich jedes Tensors auf nur 256 gleichmäßig verteilte Stufen ab. Der fusionierte Faltungskern von RepVGG summiert drei unabhängig voneinander batchnormalisierte Zweige. Deren Skalierungsfaktoren können sich nach dem Einrechnen der Normalisierung um Größenordnungen unterscheiden. Daher weisen die fusionierten Gewichte und Aktivierungen oft Verteilungen mit weit auslaufenden Flanken auf, die 256 diskrete Stufen nur schlecht abbilden. Die Folgen können gravierend sein. Eine naive 8-Bit-Quantisierung nach dem Training senkt die Top-1-Genauigkeit eines fusionierten RepVGG-B0 auf ImageNet von 75.1% auf 40.2%. Ein gewöhnliches ResNet verliert dagegen nur einen Bruchteil eines Prozentpunkts :cite:`chu2024qarepvgg`. Eine Regularisierung der Skalierungsfaktoren der Zweige im Training löst das Problem. Für die Praxis folgt daraus, dass exakte algebraische Äquivalenz in FP32 bei einer Quantisierung mit geringer Genauigkeit verloren gehen kann.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:50fe9e47-->
Beim Entwurf effizienter konvolutionärer Netze sind Latenz, Speicherbedarf und Energieverbrauch ebenso wichtig wie Genauigkeit. Tiefenweise separierbare Faltungen trennen räumliche Filterung und Kanalmischung. Damit sinken die Kosten einer Schicht auf den Anteil $1/c_\textrm{o} + 1/k^2$ der Kosten einer dichten Faltung. Der invertierte Flaschenhals von MobileNetV2 hält den Residualpfad schmal, um den maximalen Speicherbedarf der Aktivierungen zu minimieren. Auf Fashion-MNIST liegt `MiniMobileNet` in der Genauigkeit rund zwei Prozentpunkte hinter einem dichten Netz mit vergleichbarer Parameteranzahl. Es benötigt $7.7\times$ weniger Multiplikationsadditionen, läuft aber nur $2\text{--}3.5\times$ schneller, weil die Speicherbandbreite die tiefenweisen Schichten mit ihrer geringen arithmetischen Intensität ausbremst. Verengen wir das dichte Netz auf dasselbe Rechenbudget, schlägt es `MiniMobileNet` bei dieser kleinen Aufgabe sowohl in der Genauigkeit als auch in der Geschwindigkeit. Über die Faktorisierung einzelner Schichten hinaus legen kombinierte Skalierung und eine Architektursuche unter Berücksichtigung der Latenz die Tiefe, Breite und Auflösung der Stufen fest. Squeeze-and-Excitation-Gates gewichten die Kanäle anhand globaler Informationen neu und verursachen kaum zusätzlichen Rechenaufwand. Bei der strukturellen Reparametrisierung trainieren wir verzweigte Blöcke, die sich für die Inferenz zu schnellen, einfachen Faltungen zusammenfassen lassen. Die anschließende Quantisierung verlangt allerdings Sorgfalt. Nachdem wir die Effizienz beim Entwurf von Hand und bei der Suche nach einem einzelnen Modell untersucht haben, wenden wir uns nun Trainingsverfahren, der systematischen Modernisierung von ResNet und den Entwurfsräumen für ganze Populationen von Architekturen zu.

<!--en:2294243a-->
## Übungen

<!--en:0e537f07-->
1. **Kanalpermutation zwischen Gruppen (ShuffleNet).** ● Bei gruppierten Faltungen lassen sich Kanäle auch ohne dichte $1 \times 1$-Gewichte mischen :cite:`Zhang.Zhou.Lin.ea.2018`. Lassen Sie auf eine gruppierte $k \times k$-Faltung ($c_\textrm{i} \to c_\textrm{o}$ Kanäle, $g$ Gruppen) eine feste Permutation der $c_\textrm{o}$ Zwischenkanäle und eine zweite gruppierte Faltung ($c_\textrm{o} \to c_\textrm{o}$, $g$ Gruppen) folgen. Bestimmen Sie die Bedingung an $g$ und $c_\textrm{o}$, unter der eine Permutation jedem Ausgabekanal der zweiten Schicht erlaubt, von allen $c_\textrm{i}$ Eingabekanälen abzuhängen. Konstruieren Sie diese Permutation explizit.

<!--en:b8d1a2e9-->
1. [code] **Breitenmultiplikator.** Ergänzen Sie einen Breitenmultiplikator $m$ in `MiniMobileNet`, der die Kanalanzahl jeder Stufe mit $m$ multipliziert :cite:`howard2017mobilenet`. Sagen Sie voraus, wie die Parameteranzahl und die Anzahl der Multiplikationsadditionen mit $m$ skalieren. Überprüfen Sie Ihre Vorhersage für $m = 0.5$ mit `num_params` und `mobilenet_macs`.
   :exlabel:`ex_efficient-convnets-width-multiplier`

<!--en:25356823-->
1. [code] **Batchnormalisierung in eine Faltung mit Bias einrechnen.** Erweitern Sie `fuse_bn` für eine vorangehende Faltung mit einem von null verschiedenen Bias $b_o$. Verwenden Sie dazu :eqref:`eq_bn_fold`. Prüfen Sie numerisch, ob das Paar `Conv + BN` im Inferenzmodus und die einzelne fusionierte Faltung bis auf $10^{-6}$ übereinstimmen. Warum können ein fester Faltungskern und ein fester Bias die Ausgabe von `Conv + BN` nicht reproduzieren, wenn die Batchnormalisierung im *Trainingsmodus* arbeitet?

<!--en:9393b933-->
1. [code] **Der invertierte Flaschenhals von MobileNetV2.** ● MobileNetV2 :cite:`sandler2018mobilenetv2` ersetzt den Block aus :numref:`fig_dws_block` durch einen invertierten Flaschenhals. Dieser besteht aus einer $1 \times 1$-Erweiterung um den Faktor $t = 6$, einer tiefenweisen $3 \times 3$-Faltung und einer *linearen* $1 \times 1$-Projektion (ohne ReLU). Stimmen Eingabe- und Ausgabeform überein, kommt eine Sprungverbindung hinzu.
    1. Implementieren Sie den Block mit invertiertem Flaschenhals für $t = 6$ und ersetzen Sie die Blöcke von `MiniMobileNet` durch diesen Block. Wählen Sie die Stufenbreiten so, dass die Gesamtzahl der Parameter bei etwa 542 474 bleibt. Trainieren Sie das Netz unter den Bedingungen aus :numref:`tab_dws_vs_dense` und vergleichen Sie Validierungsgenauigkeit und Latenz des Vorwärtsdurchlaufs mit `MiniMobileNet`.
    1. Fügen Sie nach der letzten $1 \times 1$-Projektion eine ReLU ein. Trainieren Sie die lineare Variante und die ReLU-Variante jeweils dreimal. Prüfen Sie, ob die lineare Projektion bei dieser Modellgröße besser abschneidet als die ReLU-Projektion.
    1. Erstellen Sie für eine $96 \times 96$-Eingabe eine Tabelle mit der Anzahl der Elemente im Eingabetensor, im Ausgabetensor und im erweiterten inneren Tensor jedes Blocks. Erklären Sie, warum eine Inferenzumgebung den erweiterten Tensor nie vollständig im RAM speichern muss, wenn sie Erweiterung und tiefenweise Filterung kanalweise fusioniert oder in Teilblöcken ausführt.

<!--en:0ed4abbf-->
1. [code] **Arithmetische Intensität und Batchgröße.** ● `MiniMobileNet` führt $7.7\times$ weniger Multiplikationsadditionen aus als `VGGSmall`. Dennoch ist der GPU-Vorwärtsdurchlauf in :numref:`tab_dws_vs_dense` nur $2\text{--}3.5\times$ schneller.
    1. Definieren Sie die arithmetische Intensität einer Schicht als Anzahl ihrer Multiplikationsadditionen geteilt durch die Gesamtzahl der gelesenen und geschriebenen Tensorelemente (Gewichte + Eingabeaktivierungen + Ausgabeaktivierungen). Berechnen Sie die arithmetische Intensität der tiefenweisen Faltung, der punktweisen Faltung und einer entsprechenden dichten $3 \times 3$-Faltung für jede Stufe von `MiniMobileNet`. In welchen Stufen ist die arithmetische Intensität der tiefenweisen Faltung am geringsten? Wie erklärt das, dass EfficientNetV2 :cite:`tan2021efficientnetv2` frühe tiefenweise Stufen durch dichte Faltungen ersetzt?
    1. Messen Sie `forward_ms` für `MiniMobileNet` und `VGGSmall` bei Batchgrößen $B \in \{1, 16, 128, 512\}$. Tragen Sie die Latenz pro Bild gegen $B$ auf. Wie ändert sich das Verhältnis der Geschwindigkeiten mit $B$ und warum?

<!--en:a8076f23-->
1. [code] **Training und Quantisierung von RepVGG-Blöcken.** ● Warum sollten wir einen `RepVGGBlock` mit drei Zweigen trainieren, wenn wir ihn für die Inferenz zu einer einzigen $3 \times 3$-Faltung fusionieren?
    1. Erweitern Sie `RepVGGBlock` und `fuse_block` um Blöcke, die bei einem Downsampling mit Stride 2 die Kanalbreite ändern. Behalten Sie bei verschiedenen Formen von Eingabe und Ausgabe nur die $3 \times 3$- und $1 \times 1$-Zweige bei :cite:`ding2021repvgg`. Bauen Sie ein kleines konvolutionäres Netz aus Blöcken vom Typ `RepVGGBlock` und ein entsprechendes einfaches konvolutionäres Netz, das im Training nur den einzelnen $3 \times 3$-Zweig enthält. Trainieren Sie beide auf Fashion-MNIST und vergleichen Sie ihre Konvergenz. Überprüfen Sie, dass die Fusion des trainierten RepVGG-Netzes seine Validierungsgenauigkeit unverändert lässt.
    1. Rechnen Sie die Batchnormalisierungen des einfachen Netzes mit `fuse_bn` in die Faltungen ein. Quantisieren Sie anschließend die Gewichte beider für die Inferenz vorbereiteten Netze auf 8-Bit-Ganzzahlen. Verwenden Sie eine symmetrische Skalierung pro Tensor ($\hat{w} = s \cdot \mathrm{round}(w / s)$ mit $s = \max |w| / 127$). Vergleichen Sie den Genauigkeitsverlust des fusionierten RepVGG-Netzes mit dem des einfachen Netzes. Wie hängt er mit der Analyse der Gewichtsverteilung von :citet:`chu2024qarepvgg` zusammen?

<!--en:2733dd2c-->
[Diskussion](https://d2l.discourse.group/t/36760)

<!--en:1b9d3dea-->
Effiziente konvolutionäre Netze<br>**Tiefenweise separierbare Faktorisierung · invertierte Flaschenhälse · FLOPs und Latenz · kombinierte Skalierung · RepVGG-Fusion**

<!--en:58314379-->
::: {.slide title="Grenzen auf Endgeräten: Warum 15 Milliarden Multiplikationsadditionen zu viel sind"}

<!--en:f0e670cc-->
[Motivation und Grenzen im Einsatz]{.kicker}

<!--en:62dd1f60-->
- **Rechenzentrum und Endgerät:** VGG-16 benötigt $15.3\textrm{B}$ MACs pro Bild und $550\textrm{ MB}$ für FP32-Gewichte. MobileNetV1 erreicht $70.6\%$ Top-1-Genauigkeit auf ImageNet ($-0.9\%$) mit **$27\times$ weniger MACs** ($0.57\textrm{B}$) und **$33\times$ weniger Gewichten** ($4.2\textrm{M}$).

<!--en:d96b6e16-->
- **Drei Grundideen:** tiefenweise separierbare Faktorisierung, kombinierte Skalierung mit SE-Gates und Fusion der Trainingszweige von RepVGG für die Inferenz.

<!--en:1b0f625e-->
::: {.slide title="Von der gruppierten Faltung zum tiefenweisen Grenzfall"}

<!--en:7aedd311-->
[Verbindungsmatrix der Kanäle]{.kicker}

<!--en:5e2de1a2-->
Der Aufwand einer gewöhnlichen $k \times k$-Faltung wächst quadratisch mit der Breite ($c_{\textrm{i}} c_{\textrm{o}} k^2$), weil jeder Ausgabekanal jeden Eingabekanal mit einem eigenen räumlichen Filter faltet.

<!--en:0b190196-->
- **Gruppierte Faltung ($g$ Gruppen):** Die Kanäle zerfallen in $g$ disjunkte Blöcke mit jeweils $c_{\textrm{i}}/g \to c_{\textrm{o}}/g$ Kanälen. Die Verbindungsmatrix wird blockdiagonal. Die Anzahl der Gewichte und der MACs sinkt auf jeweils $c_{\textrm{i}} c_{\textrm{o}} k^2 / g$.

<!--en:7acb2031-->
- **Tiefenweiser Grenzfall ($g = c_{\textrm{i}} = c_{\textrm{o}}$):** Jede Gruppe enthält **einen einzigen Kanal**. Jeder der $c_{\textrm{i}}$ Kanäle hat seinen eigenen $k \times k$-Faltungskern (insgesamt $c_{\textrm{i}} k^2$ Gewichte). Die Kanäle werden nicht gemischt.

<!--en:5149dafa-->
::: {.slide title="Tiefenweise separierbare Faltungen: räumliche Filterung und Kanalmischung"}

<!--en:17304bef-->
[Faktorisierung in zwei Schritten]{.kicker}

<!--en:ad551ea1-->
Ein tiefenweiser räumlicher $k \times k$-Filter und eine nachgeschaltete punktweise $1 \times 1$-Faltung zur Kanalmischung zerlegen eine dichte Faltung in zwei günstige Schritte.

<!--en:fdb7b3a7-->
- **Schritt 1 – tiefenweise ($k \times k$, $g = c_{\textrm{i}}$):** Räumliche Nachbarschaften werden in jedem Eingabekanal unabhängig gefiltert. Das kostet $h w c_{\textrm{i}} k^2$ Multiplikationsadditionen.

<!--en:415a4090-->
- **Schritt 2 – punktweise ($1 \times 1$, $c_{\textrm{i}} \to c_{\textrm{o}}$):** An jedem Pixel werden die $c_{\textrm{i}}$ gefilterten Kanäle linear kombiniert. Das kostet $h w c_{\textrm{i}} c_{\textrm{o}}$ Multiplikationsadditionen.

<!--en:5f6baf27-->
- **Eingeschränkte Ausdrucksfähigkeit (Rang-1-Faktorisierung des Faltungskerns):** Für jedes Paar $(o, c)$ muss der dichte $k\times k$-Filter die faktorisierte Form $\mathbf{K}_{o,c} = P_{o,c}\,\mathbf{D}_c$ haben.

<!--en:d87a183c-->
::: {.slide title="Verhältnis von FLOPs und Parameteranzahlen bei tiefenweise separierbarer Faltung"}

<!--en:aefbeeb4-->
[Einsparung in Zahlen]{.kicker}

<!--en:c70202d8-->
Das Verhältnis der Kosten der beiden separierbaren Schritte zu denen der gewöhnlichen dichten Faltung ergibt für Parameter und Multiplikationsadditionen dasselbe exakte Verhältnis.

<!--en:1145c70e-->
- **Asymptotische Beschleunigung um den Faktor $k^2$:** Für $c_{\textrm{o}} \gg k^2$ ist der Term $1/c_{\textrm{o}}$ vernachlässigbar. Eine separierbare $3\times 3$-Schicht benötigt **$\approx 1/9$** (also $8\text{--}9\times$ weniger) der Rechenoperationen einer dichten $3\times 3$-Schicht. Bei einer separierbaren $7\times 7$-Schicht (ConvNeXt) beträgt der Anteil $\approx 1/49$.

<!--en:d1fda345-->
- **Überprüfung für $c_{\textrm{i}} = c_{\textrm{o}} = 128, k = 3$:** Die Anzahl der Gewichte sinkt um den Faktor $(1/128 + 1/9)^{-1} = 147{,}456 / 17{,}536 \approx \mathbf{8.41\times}$.

<!--en:177d1957-->
::: {.slide title="MobileNetV1 und der tiefenweise separierbare Block"}

<!--en:9cb76810-->
[Architektur von MobileNet]{.kicker}

<!--en:599b5ec7-->
MobileNetV1 ersetzt jede gewöhnliche Faltungsschicht nach dem Eingangsteil durch einen Block der Form `Depthwise 3×3 -> BN -> ReLU -> Pointwise 1×1 -> BN -> ReLU`.

<!--en:ab3a7868-->
- **Tiefenweises Downsampling mit Stride:** Mit `stride=2` halbiert die tiefenweise $3\times 3$-Faltung die Auflösung ohne separate Pooling-Schichten.

<!--en:0e5df5cd-->
- **Mini-MobileNet auf Fashion-MNIST mit $96\times 96$:** Auf eine $3\times 3$-Eingangsschicht mit Stride 2 und 32 Kanälen folgen 7 separierbare Blöcke mit bis zu 512 Kanälen. Das Netz hat **$542{,}474$ Parameter** (gegenüber $4.66\text{M}$ Gewichten in den 7 Blöcken mit dichten $3\times 3$-Faltungen, eine Einsparung um den Faktor $8.8\times$).

<!--en:436878ba-->
::: {.slide title="Implementierung eines tiefenweise separierbaren Blocks"}

<!--en:2e26dadc-->
[Block von Mini-MobileNet]{.kicker}

<!--en:0a4ee8f1-->
Auf eine tiefenweise $3\times 3$-Faltung (ein Filter pro Kanal) folgt eine $1\times 1$-Faltung zur punktweisen Kanalmischung.

<!--en:74fbe6b0-->
::: {.slide title="Aufbau von Mini-MobileNet und Zählung der Parameter"}

<!--en:e7d7882e-->
[Basisnetz von Mini-MobileNet]{.kicker}

<!--en:76763efd-->
::: {.slide title="MobileNetV2: invertierter Flaschenhals und lineare Projektion"}

<!--en:3d34459c-->
[Invertierte Flaschenhälse]{.kicker}

<!--en:3cc26e9d-->
Ein tiefenweiser $3\times 3$-Filter ist bereits günstiger als ein dichter. Da er nur einen Filter pro Kanal lernt, braucht er für eine hohe Ausdrucksfähigkeit einen *breiten* Kanalraum.

<!--en:2f48a61d-->
- **Klassischer ResNet-Flaschenhals (links):** Die Abfolge $256 \xrightarrow{1\times 1} 64$ $\to$ $\text{dense } 3\times 3$ $\to$ $64 \xrightarrow{1\times 1} 256$ komprimiert die Darstellung für die aufwendige dichte $3\times 3$-Faltung auf wenige Kanäle.

<!--en:a093c0fc-->
- **Invertierter Flaschenhals (rechts, MobileNetV2):** Der Residualstrom bleibt schmal ($24$ Kanäle). Der Block *erweitert* ihn um den Faktor $6\times$ durch eine $1\times 1$-Faltung ($24 \to 144$), faltet tiefenweise mit $3\times 3$ auf $144$ Kanälen und projiziert zurück ($144 \to 24$).

<!--en:2d5c7a4c-->
- **Warum die letzte $1\times 1$-Projektion ohne ReLU auskommt:** ReLU löscht negative Koordinaten im niedrigdimensionalen Flaschenhals mit 24 Kanälen. Bleibt die schmale Projektion *linear*, erhält sie Information. Schmale Sprungverbindungen minimieren den SRAM-Datenverkehr zwischen den Blöcken.

<!--en:a19c00b2-->
::: {.slide title="Kontrollierter Vergleich bei gleicher Parameteranzahl oder gleicher FLOP-Anzahl"}

<!--en:fd01f7e0-->
[Vergleich dreier Modelle]{.kicker}

<!--en:75e96104-->
Vergleich von `MiniMobileNet` ($542\text{K}$ Parameter, $50.3\text{M}$ MACs) auf Fashion-MNIST bei $96\times 96$ mit zwei dichten Referenzmodellen nach VGG: `VGGSmall` hat ungefähr gleich viele *Parameter* ($584\text{K}$ Parameter, Breite bis 128). `VGG-style, narrow` benötigt ungefähr gleich viele *Multiplikationsadditionen* ($49.1\text{M}$ MACs, Breite $\div 2.8$, $76\text{K}$ Parameter).

<!--en:95ae5244-->
::: {.slide title="Warum 7.7× weniger FLOPs auf der GPU nur 2–3.5× schneller sind"}

<!--en:4103de9d-->
[Systemanalyse: arithmetische Intensität]{.kicker}

<!--en:7362d81c-->
- **Gleiche Parameteranzahl (`VGGSmall`, 385 Mio. MACs):** $7.7\times$ so viele MACs wie `MiniMobileNet` ($50.3\textrm{M}$), auf der GPU aber nur $2\text{--}3.5\times$ langsamer ($93.8\%$ gegenüber $91.5\%$ Genauigkeit).

<!--en:af9cadc9-->
- **Gleiche FLOP-Anzahl (`VGG-style, narrow`, 49.1 Mio. MACs):** Höhere Genauigkeit als `MiniMobileNet` ($93\%$) *und* kürzere Laufzeit (Faktor $0.5\text{--}0.8\times$).

<!--en:2bed1aa9-->
- **Engpass im Roofline-Modell:** Eine tiefenweise $3 \times 3$-Faltung benötigt nur $9$ MACs pro gelesenem Wert, eine dichte dagegen $9c_\textrm{o}$ bei $3 \times 3$.

<!--en:f772c259-->
::: {.slide title="Kombinierte Skalierung und Architektursuche unter Berücksichtigung der Latenz"}

<!--en:2f40a7f0-->
[EfficientNet und EfficientNetV2]{.kicker}

<!--en:4de6e61c-->
- **Kombinierte Skalierung (Tan & Le, 2019):** Jeder Schritt $\phi$ skaliert $(d, w, r)$ mit $(1.2^\phi, 1.1^\phi, 1.15^\phi)$. Der Rechenaufwand steigt um den Faktor $\approx 1.92^\phi$.

<!--en:f77313f2-->
- **Latenzbewusste NAS (EfficientNetV2):** In frühen Stufen mit hoher Auflösung ersetzen fusionierte dichte $3 \times 3$-Faltungen die tiefenweisen Faltungen und erhöhen die arithmetische Intensität auf der GPU.

<!--en:43835e8c-->
::: {.slide title="Squeeze-and-Excitation (SE): globale Attention über die Kanäle"}

<!--en:4bc9f364-->
[Squeeze-and-Excitation-Gates]{.kicker}

<!--en:bed3fb4c-->
- **Verdichten $\to$ Anregen $\to$ Neuskalieren:** Pooling bildet $(C, H, W) \to (C, 1, 1)$ ab. Ein Gate-MLP mit $C \to C/r \to C$ ($\sigma$) benötigt dafür $<0.03\%$ zusätzliche MACs.

<!--en:6359e329-->
::: {.slide title="RepVGG: mehrere Zweige im Training, ein Pfad bei der Inferenz"}

<!--en:86db09c5-->
[Strukturelle Reparametrisierung]{.kicker}

<!--en:2bc40b68-->
Residualnetze mit mehreren Zweigen lassen sich im Training leicht optimieren. Einfache Folgen von $3\times 3$-Faltungen mit nur einem Pfad nutzen Speicher und Rechenleistung bei der Inferenz optimal aus. *RepVGG* (Ding et al., 2021) verbindet diese Vorteile durch algebraische Fusion.

<!--en:d709b6fd-->
- **Block im Training:** Vor der ReLU summiert er drei parallele Zweige mit Batchnormalisierung:
  $$\mathbf{Y} = \textrm{ReLU}\bigl(\textrm{BN}_3(\textrm{Conv}_{3\times 3}(\mathbf{X})) + \textrm{BN}_1(\textrm{Conv}_{1\times 1}(\mathbf{X})) + \textrm{BN}_0(\mathbf{X})\bigr)$$

<!--en:76e6755d-->
- **Fusion für den Einsatz:** Bei der Inferenz wirken alle drei Zweige innerhalb von $\textrm{ReLU}(\cdot)$ als affine Operatoren. Sie lassen sich algebraisch zu **einer einzigen $3\times 3$-Faltung mit Bias** zusammenfassen. Zwei Zweige und zwei Aktivierungspuffer entfallen.

<!--en:b5dc86c5-->
::: {.slide title="Fusion der Zweige in drei Schritten der linearen Algebra"}

<!--en:d4659533-->
[Batchnormalisierung einrechnen und Faltungskerne addieren]{.kicker}

<!--en:e6870553-->
Ein trainierter `RepVGGBlock` lässt sich in eine einzige $3\times 3$-Schicht $(\hat{\mathbf{W}}, \hat{\mathbf{b}})$ überführen. Dazu sind pro Ausgabekanal $o$ drei exakte Umformungen nötig.

<!--en:3cb97bb7-->
- **1. Batchnormalisierung im Inferenzmodus in jeden Zweig einrechnen:** Mit festen gleitenden Statistiken $(\mu_o, \sigma_o^2)$ und Parametern $(\gamma_o, \beta_o)$ gilt
  $$\hat{\mathbf{W}}_o = \frac{\gamma_o}{\sqrt{\sigma_o^2 + \epsilon}}\,\mathbf{W}_o, \qquad \hat{b}_o = \beta_o + \frac{\gamma_o(b_o - \mu_o)}{\sqrt{\sigma_o^2 + \epsilon}}$$

<!--en:ba81a543-->
- **2. $1\times 1$-Faltungskern und Identität als $3\times 3$-Faltungskerne darstellen:** Den $1\times 1$-Faltungskern auf allen Seiten mit 1 Pixel Nullen umgeben. Die Identität auf Kanal $o$ als $3\times 3$-Faltungskern mit $1$ an der mittleren Position $(i=o, 1, 1)$ und $0$ an allen anderen Positionen schreiben.

<!--en:d66a0b6d-->
- **3. Linearität zur Summation nutzen:** $\hat{\mathbf{W}}_{\text{fused}} = \hat{\mathbf{W}}^{(3)} + \hat{\mathbf{W}}^{(1)} + \hat{\mathbf{W}}^{(0)}$ und $\hat{\mathbf{b}}_{\text{fused}} = \hat{\mathbf{b}}^{(3)} + \hat{\mathbf{b}}^{(1)} + \hat{\mathbf{b}}^{(0)}$.

<!--en:486821bc-->
::: {.slide title="Implementierung der Fusion mit `fuse_bn` und `fuse_block`"}

<!--en:880813c7-->
[Code zur RepVGG-Fusion]{.kicker}

<!--en:db01073f-->
In jedem Zweig die Batchnormalisierung in einen $3\times 3$-Faltungskern einrechnen, dann alle drei zu einer Faltung addieren.

<!--en:1f50b2a3-->
::: {.slide title="Exakte FP32-Äquivalenz und die Tücken der INT8-Quantisierung"}

<!--en:f02f2aab-->
[Überprüfung im Code und Risiken beim Einsatz]{.kicker}

<!--en:24eda2a4-->
Implementierung von `fuse_bn` und `fuse_block`, Prüfung der exakten Äquivalenz in `float32` und Untersuchung eines wesentlichen Risikos der Quantisierung nach dem Training.

<!--en:b18025f2-->
- **Exakt bis auf FP32-Rundung ($\sim 10^{-6}$):** Nach Aktualisierung der gleitenden Statistiken und zufälliger Wahl von $(\gamma, \beta)$ stimmt die fusionierte $3\times 3$-Schicht bis auf $10^{-6}$ mit dem Block aus drei Zweigen überein.

<!--en:e72d5886-->
- **Fallstrick – FP32-Äquivalenz $\neq$ INT8-Äquivalenz (Chu et al., 2024):** Der mittlere Eintrag von $\hat{\mathbf{W}}_{\text{fused}}$ summiert drei unabhängig normalisierte Zweige mit sehr unterschiedlichen Skalierungsfaktoren ($\gamma_o / \sqrt{\sigma_o^2 + \epsilon}$). Daher treten unter den fusionierten Gewichten starke Ausreißer auf. Bei naiver gleichmäßiger 8-Bit-Quantisierung (256 Stufen) bricht die Genauigkeit des fusionierten RepVGG-B0 auf ImageNet **von $75.1\%$ auf $40.2\%$** ein. QAT oder Regularisierung der Skalierungsfaktoren der Zweige kann diesen Verlust ausgleichen.

<!--en:09253856-->
::: {.slide title="Verständnisfragen: Einsparung durch Faktorisierung, SE-Kosten und Batchnormalisierung"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:f9fb00e6-->
Die Effizienz in den folgenden drei Fällen berechnen:

<!--en:f606a35e-->
1. **Einsparung bei separierbarer $7\times 7$-Faltung:** In ConvNeXt (§{sec_convnext}) verwendet ein Block eine tiefenweise $7\times 7$-Faltung, gefolgt von einer $1\times 1$-Faltung bei einer Breite von $c_{\textrm{i}} = c_{\textrm{o}} = 96$. Um welchen Faktor ist dieses tiefenweise separierbare Paar günstiger als eine dichte $7\times 7$-Faltung bei $c = 96$?

<!--en:9b68d009-->
2. **Zusatzaufwand von Squeeze-and-Excitation:** In einer Stufe mit $c = 256$ Kanälen bei einer Auflösung von $14\times 14$ die Anzahl der Multiplikationsadditionen eines SE-Gates ($r = 4$) mit der einer dichten $3\times 3$-Faltung ($256 \to 256$) vergleichen.

<!--en:c3c0d55f-->
3. **Batchnormalisierung von Hand einrechnen:** Ein Faltungskanal ohne Bias ($b_o = 0$) hat den Faltungskern $\mathbf{W}_o$. Darauf folgt eine Batchnormalisierung mit gleitendem Mittelwert $\mu_o = 0.5$, gleitender Varianz mit Stabilisierungsterm $\sigma_o^2 + \epsilon = 0.25$, Skalierungsfaktor $\gamma_o = 1$ und Verschiebung $\beta_o = 0$. Wie lauten der umgerechnete Faltungskern $\hat{\mathbf{W}}_o$ und der Bias $\hat{b}_o$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:81bc8be9-->
1. Kostenverhältnis: $1/c_{\textrm{o}} + 1/k^2 = 1/96 + 1/49 \approx 0.0308$. Die separierbare Faltung ist also **um den Faktor $32.4\times$ günstiger** (der Faktor nähert sich $49\times$, wenn $c_{\textrm{o}}$ wächst).

<!--en:968327b3-->
2. **SE-Gate:** $2 c^2 / r = 2(256^2)/4 = 32{,}768$ MACs (für einen durch Pooling gewonnenen $1\times 1$-Vektor). **Dichte $3\times 3$-Faltung:** $14^2 \cdot 9 \cdot 256^2 \approx 115.6\text{M}$ MACs. Das SE-Gate erhöht den Rechenaufwand um nur **$0.028\%$**.

<!--en:730748a7-->
3. Der Skalierungsfaktor beträgt $\gamma_o / \sqrt{\sigma_o^2 + \epsilon} = 1 / \sqrt{0.25} = 2$. Damit ergeben sich $\hat{\mathbf{W}}_o = 2\mathbf{W}_o$ und $\hat{b}_o = 0 + 2(0 - 0.5) = -1$.

<!--en:75d81987-->
::: {.slide title="Zusammenfassung: Erkenntnisse zu effizienten konvolutionären Netzen"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:a8e09fc7-->
| Verfahren | Mathematische oder strukturelle Form | Effizienzgewinn und praktische Einschränkung |

<!--en:88758c7b-->
| **Tiefenweise separierbare Faltung** | Tiefenweise $k\times k$ ($g=c_{\textrm{i}}$) + punktweise $1\times 1$ ($c_{\textrm{i}}\to c_{\textrm{o}}$) | Senkt die Anzahl der Gewichte und MACs auf den Anteil $1/c_{\textrm{o}} + 1/k^2$ ($\approx 1/9$ bei $3\times 3$). Der tiefenweise Schritt hat jedoch eine geringe arithmetische Intensität ($9$ MACs pro gelesenem Wert). |

<!--en:5c41767d-->
| **Invertierter Flaschenhals (MBConv)** | Schmaler Residualpfad ($c$) $\to$ $1\times 1$ ($tc$) $\to$ $\text{DW } 3\times 3$ ($tc$) $\to$ linear $1\times 1$ ($c$) | Hält die Tensoren zwischen den Blöcken im SRAM schmal. Ohne abschließende ReLU bleibt der Rang im schmalen Flaschenhals erhalten. |

<!--en:256d7589-->
| **Kombinierte Skalierung** | Skaliert $(d, w, r) \leftarrow (\alpha^\phi d, \beta^\phi w, \gamma^\phi r)$ mit $\alpha\beta^2\gamma^2 \approx 2$ | Stimmt rezeptives Feld, Kapazität der Merkmalsdarstellung und Bilddetail gleichzeitig aufeinander ab (EfficientNet). |

<!--en:62f9a8b9-->
| **Squeeze-and-Excitation (SE)** | $\mathbf{X} \odot \sigma(\mathbf{W}_2\,\textrm{ReLU}(\mathbf{W}_1\,\textrm{GAP}(\mathbf{X})))$ | Globale Attention über die Kanäle für $2c^2/r$ Gewichte und $<0.1\%$ zusätzliche FLOPs |

<!--en:cd2f0df5-->
| **RepVGG-Reparametrisierung** | $\textrm{BN}$ in die Zweige $3\times 3, 1\times 1, \mathbf{I}$ einrechnen und diese zu einer $3\times 3$-Faltung addieren | Leicht trainierbar mit mehreren Zweigen, schnelle FP32-Inferenz mit einem Pfad. INT8-Quantisierung erfordert Kontrolle der Skalierungsfaktoren der Zweige. |
