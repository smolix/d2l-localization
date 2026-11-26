<!--en:dd7e5485-->
# Entwurfsräume für konvolutionäre Netze

<!--en:85727b90-->
Die meisten Netze, die wir in diesem Kapitel gebaut haben, wurden von Hand entworfen. Für jede Stufe legte man einige wenige strukturelle Hyperparameter fest: Wie viele Blöcke folgen aufeinander? Wie viele Kanäle gibt die Stufe aus? Wie ist der Block aufgebaut? Woher kommen diese Zahlen, und welchen allgemeinen Prinzipien folgen sie? Ein Rückblick auf unsere Modelle gibt überraschend wenig Aufschluss. ResNet-18 verwendet $(2, 2, 2, 2)$ Blöcke pro Stufe, ResNet-50 verwendet $(3, 4, 6, 3)$ und ConvNeXt-T verwendet $(3, 3, 9, 3)$. ConvNeXt-T übernimmt diese Folge von Swin und prüft sie in einem einzigen Versuch. Erschwerend kommt hinzu, dass die Parameter eng zusammenhängen: Bei festem Rechenbudget muss eine tiefere Stufe schmaler sein. Gruppierte Faltungen geben Rechenkapazität für mehr Kanäle frei. Alle Möglichkeiten durchzuprobieren, kommt nicht infrage. Bei vier Zahlen für jede der vier Stufen ergeben schon zwei Möglichkeiten pro Zahl $2^{16} = 65{,}536$ verschiedene Netze, von denen jedes einen eigenen Trainingslauf erfordert. Und selbst wenn wir so ein überlegenes Netz fänden, verriete uns eine einzelne Punktschätzung weder, *welche* Entwurfsentscheidungen den Ausschlag gaben, noch, wie wir sie an ein neues Rechenbudget oder einen neuen Baustein anpassen können.

<!--en:b7db15b5-->
Die *neuronale Architektursuche* (engl. neural architecture search, NAS) :cite:`zoph2016neural,liu2018darts`, aus der EfficientNet hervorging (:numref:`sec_efficient_cnns`), automatisiert die Auswahl eines einzelnen Modells. Wie der Entwurf von Hand liefert sie eine einzelne Lösung, aus der sich keine wiederverwendbaren wissenschaftlichen Regeln ablesen lassen. :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` gingen grundlegend anders vor. Sie verglichen ganze *Entwurfsräume* (engl. design spaces), also Populationen von Netzen, die aus einem gemeinsamen Schema und wenigen Nebenbedingungen hervorgehen. Sie zogen Netze aus einem Entwurfsraum, trainierten sie kurz und prüften, ob eine vorgeschlagene Nebenbedingung die *Fehlerverteilung* der gesamten Population verbessert. So leiteten sie einfache, übertragbare Entwurfsregeln ab, die die *RegNet*-Familie definieren. Wir implementieren im Folgenden ihr AnyNet-Schema und untersuchen ihre Ergebnisse für ganze Populationen auf ImageNet (500 Modelle pro Raum). Die wirksamste Regel prüfen wir selbst an zwei kleinen Populationen auf Fashion-MNIST. Anschließend trainieren wir ein kompaktes RegNet zum direkten Vergleich mit ResNet-18.

<!--en:651c2162-->
## Der Entwurfsraum von AnyNet

<!--en:674b68e4-->
Im Laufe dieses Kapitels haben wir gesehen, wie sich bei modernen CNNs der dreiteilige Aufbau aus :numref:`sec_googlenet` durchgesetzt hat. Der *Eingangsteil* (engl. stem) verarbeitet die Rohpixel mit einer oder wenigen Faltungen und verringert meist sofort die Auflösung. Der *Hauptteil* (engl. body) leistet den Großteil der Rechenarbeit. Er besteht aus einer Folge von *Stufen* (engl. stages). Jede Stufe wiederholt denselben *Block* bei fester Auflösung. Am Übergang zur nächsten Stufe halbieren sich die räumlichen Abmessungen, und die Kanalzahl steigt. Der *Kopf* (engl. head) bildet den letzten Merkmalstensor durch Global-Average-Pooling und eine lineare Schicht auf die Klassenlogits ab. Die vier Stufen von ResNet geben Merkmale mit $1/4$, $1/8$, $1/16$ und $1/32$ der Eingabeauflösung aus. Diese mehrskalige Pyramide hat sich bis hin zu ConvNeXt gehalten, vor allem weil nachgeschaltete Objektdetektoren und Segmentierungsköpfe alle vier Auflösungen gleichzeitig nutzen (:numref:`sec_multiscale-object-detection`).

<!--en:15d398df-->
### Stufenparameter von Hand entworfener Netze

<!--en:2705380d-->
:numref:`tab_hand_designed_stages` vergleicht vier wegweisende Architekturen aus diesem Kapitel. In jeder Tabellenzeile verdoppelt sich die Kanalzahl von Stufe zu Stufe. Für die Stufentiefen ist hingegen keine Regel erkennbar.

<!--en:de7f42b3-->
:Stufentiefen und -breiten von vier von Hand entworfenen Netzen (ImageNet-Versionen).

<!--en:f6b660ee-->
| Netz | Blöcke pro Stufe | Ausgabekanäle pro Stufe | Block |

<!--en:b851f020-->
| ResNet-18 (:numref:`sec_resnet`) | 2, 2, 2, 2 | 64, 128, 256, 512 | zwei $3 \times 3$-Faltungen |

<!--en:17d80dc6-->
| ResNet-50 :cite:`He.Zhang.Ren.ea.2016` | 3, 4, 6, 3 | 256, 512, 1024, 2048 | Flaschenhals aus $1 \times 1$-, $3 \times 3$- und $1 \times 1$-Faltung |

<!--en:2c183258-->
| ResNeXt-50 (:numref:`subsec_resnext`) | 3, 4, 6, 3 | 256, 512, 1024, 2048 | Flaschenhals mit 32 Gruppen |

<!--en:d4ff45dc-->
| ConvNeXt-T (:numref:`sec_convnext`) | 3, 3, 9, 3 | 96, 192, 384, 768 | tiefenweise $7 \times 7$-Faltung, umgekehrter Flaschenhals |

<!--en:147fcec6-->
Der Entwurfsraum *AnyNet* von :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` gibt diesen Gesamtaufbau vor, lässt die Hyperparameter der Stufen aber frei (:numref:`fig_anynet_full`). Der Eingangsteil besteht aus einer einzigen $3 \times 3$-Faltung mit Stride 2 und $c_0 = 32$ Ausgabekanälen, gefolgt von Batchnormalisierung und ReLU. Auf ImageNet liest er RGB-Bilder ein, auf Fashion-MNIST einen einzelnen Grauwertkanal. Der Hauptteil umfasst vier Stufen. Stufe $i$ besteht aus $d_i$ aufeinanderfolgenden ResNeXt-Blöcken (:numref:`fig_resnext_block`). Eine $1 \times 1$-Faltung projiziert $c_i$ Kanäle auf $c_i / b_i$. Eine gruppierte $3 \times 3$-Faltung verarbeitet Gruppen von je $g_i$ Kanälen. Eine zweite $1 \times 1$-Faltung stellt die ursprüngliche Zahl von $c_i$ Kanälen wieder her. Auf jede Faltung folgt eine Batchnormalisierung, und eine Sprungverbindung überbrückt den Block. (Beachten Sie die Notation der Publikation: $b_i$ ist das *Flaschenhalsverhältnis* (engl. bottleneck ratio). Die Flaschenhalsbreite, die wir in :numref:`subsec_resnext` mit $b$ bezeichnet haben, beträgt hier also $c_i / b_i$.) Der erste Block jeder Stufe halbiert die räumliche Auflösung mit Stride 2. Eine $1 \times 1$-Faltung mit Stride 2 übernimmt die Projektion in der Sprungverbindung. Global-Average-Pooling und ein linearer Klassifikator bilden den Kopf.

<!--en:de87ac6d-->
![Der Entwurfsraum von AnyNet umfasst einen Eingangsteil, einen Hauptteil mit vier Stufen und einen Kopf. Stufe $\mathit{i}$ wiederholt einen ResNeXt-Block $\mathit{d_i}$-mal. Der erste Block jeder Stufe halbiert die Auflösung. Die Beschriftungen geben Kanalzahl und Auflösung nach jedem Teil für eine Eingabe der Größe $\mathit{r} \times \mathit{r}$ an. Der Kopf gibt $\mathit{n}$ Klassenscores aus. Frei wählbar sind für jede Stufe die Tiefe $\mathit{d_i}$, die Breite $\mathit{c_i}$, die Gruppenbreite $\mathit{g_i}$ und das Flaschenhalsverhältnis $\mathit{b_i}$.](../img/arch-anynet.svg)

<!--en:543f804e-->
Damit hat jede Stufe vier Freiheitsgrade: Tiefe $d_i$, Ausgabebreite $c_i$, Flaschenhalsverhältnis $b_i \geq 1$ und *Gruppenbreite* (engl. group width) $g_i$. Letztere gibt die Zahl der Kanäle pro Gruppe an. Davon zu unterscheiden ist die *Gruppenzahl* $g$ in :numref:`subsec_resnext`. Diese Bedeutung behält das Argument `groups` in `d2l.ResNeXtBlock`. Über vier Stufen ergeben sich 16 frei wählbare Zahlen. Wir ziehen Tiefen bis 16, Breiten bis 1024 in Schritten von 8, Flaschenhalsverhältnisse aus $\{1, 2, 4\}$ und Zweierpotenzen bis 32 als Gruppenbreiten. Daraus ergeben sich rund $(16 \cdot 128 \cdot 3 \cdot 6)^4 \approx 2 \times 10^{18}$ mögliche Architekturen. Sie alle aufzuzählen ist unmöglich, eine Zufallsstichprobe zu ziehen dagegen einfach. Bei ImageNet-Eingaben der Größe $224 \times 224$ halbieren der Eingangsteil und die vier Stufen die räumlichen Abmessungen insgesamt fünfmal. Die Ausgabe hat damit die Größe $224 / 2^5 = 7 \times 7$.

<!--en:8c3a694a-->
### Implementierung von AnyNet

<!--en:257d0a51-->
Um diesen Raum im Code zu untersuchen, schreiben wir einen allgemeinen Konstruktor für `AnyNet`. Tupel geben die Parameter der Stufen vor. Wir beginnen mit dem dreischichtigen Eingangsteil.

<!--en:0bd64da6-->
Jede Stufe enthält `depth` aufeinanderfolgende Instanzen von `d2l.ResNeXtBlock` (:numref:`subsec_resnext`). Ihre Parameter sind die Ausgabebreite $c_i$, die Gruppenzahl und `bot_mul` $= 1/b_i$. Damit beträgt die Gruppenbreite $(c_i/b_i)/\text{groups}$. Nur der erste Block verwendet Stride 2.

<!--en:f1c5f25e-->
Wir setzen den Eingangsteil, je eine Stufe pro Tupel `(depth, num_channels, groups, bot_mul)` in `arch` und den Kopf mit globalem Pooling zusammen. Damit ist `AnyNet` vollständig. Wie in :numref:`sec_resnet` verwenden wir für alle Gewichte der Faltungen und linearen Schichten die He-Initialisierung (:numref:`subsec_he_init`). Vor einer Batchnormalisierung verzichten wir bei den Faltungen auf Bias-Terme.

<!--en:8b298540-->
Wir müssen jedes zufällig gezogene Modell eines Entwurfsraums nach genau demselben Trainingsprotokoll beurteilen. Deshalb definieren wir den Optimierer direkt in `AnyNet`. Wir folgen dem Protokoll von :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` und Rezept A aus :numref:`sec_training_recipes`: SGD mit Momentum 0.9 (:numref:`sec_efficient_cnns`, :numref:`chap_optimization`), gekoppelter Gewichtszerfall (:numref:`sec_weight_decay`) und der im ganzen Kapitel verwendete Kosinusplan mit Aufwärmphase `d2l.cosine_warmup` (:numref:`sec_alexnet`). Während der ersten 10% von `num_steps` steigt die Lernrate auf `lr` und fällt dann entlang einer halben Kosinusperiode auf null. Die Publikation verwendet für 100 Epochen auf ImageNet einen Gewichtszerfall von $5 \times 10^{-5}$. Für unsere kürzeren Läufe auf Fashion-MNIST verwenden wir $10^{-3}$. Dieser Wert erwies sich bei der Abstimmung für `RegNetX32` (:numref:`subsec_regnet`) als der beste und funktionierte für `ResNet18` ebenso gut.

<!--en:6d5b44d4-->
Das Argument `weight_decay` eines Keras-Optimierers verkleinert die Gewichte direkt, außerhalb des Momentum-Puffers. `CoupledSGD` addiert wie in :numref:`sec_training_recipes` den Beitrag des Gewichtszerfalls zum Gradienten.

<!--en:b1969d0f-->
## Vergleich von Entwurfsräumen

<!--en:14c71ec1-->
Wie können wir zwei riesige Entwurfsräume vergleichen, ohne in einem der beiden die beste Architektur zu kennen? :citet:`radosavovic2019network` stützen sich auf vier Arbeitsannahmen. Unter diesen Annahmen kann schon eine mäßig große Stichprobe kurz trainierter Modelle einen ganzen Raum zuverlässig repräsentieren:

<!--en:be3dded9-->
1. **Gute Prinzipien liefern viele gute Netze.** Viele Netze, die sinnvollen Entwurfsregeln folgen, schneiden gut ab. Im Heuhaufen liegen also viele gute Nadeln. Deshalb lohnt es sich, eine *Verteilung* über Architekturen zu optimieren.

<!--en:a3a3da84-->
1. **Die Rangfolge nach kurzem Training bleibt aussagekräftig.** Die Validierungsgenauigkeit nach einem kurzen Trainingslauf korreliert gut mit der Genauigkeit nach der Konvergenz. Deshalb genügen in der Publikation 10 Epochen auf ImageNet und in unserem Versuch weiter unten 1 Epoche auf Fashion-MNIST.

<!--en:d4635cf1-->
1. **Ergebnisse aus kleinen Versuchen lassen sich übertragen.** Regeln, die man bei kleinem Rechenbudget findet, gelten auch für größere Modelle. Das überprüfen wir am Ende im größeren Maßstab.

<!--en:fe4e9df4-->
1. **Entwurfsentscheidungen sind weitgehend unabhängig voneinander.** Wir können strukturelle Nebenbedingungen nacheinander bewerten, ohne dass starke verborgene Wechselwirkungen das Ergebnis verfälschen.

<!--en:545673bb-->
Keine dieser vier Annahmen gilt von vornherein. Besonders weitreichend sind die zweite und die dritte. In den Übungen prüfen Sie die Annahmen direkt im kleinen Maßstab.

<!--en:76c030ba-->
Unter diesen Annahmen ziehen wir $n$ Netze zufällig aus einem Raum, trainieren jedes kurz und erfassen die Validierungsfehler $e_1, \ldots, e_n$. Die *empirische Verteilungsfunktion* (engl. empirical cumulative distribution function, CDF) gibt für jede Fehlerschwelle $e$ den Anteil der gezogenen Modelle an, deren Fehler höchstens $e$ beträgt:

<!--en:ddfa313d-->
Je höher und je weiter links die Verteilungsfunktion ansteigt, desto mehr genaue Netze enthält der Raum anteilig. Liegt die Verteilungsfunktion eines Raums durchgehend oberhalb der eines anderen, hält ein zufällig gezogenes Modell aus dem ersten Raum jede Fehlerschwelle mit höherer Wahrscheinlichkeit ein. Das gilt selbst dann, wenn beide Räume zufällig dasselbe beste Modell enthalten.

<!--en:1d6667d3-->
Mit diesem Ansatz zogen :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` aus jedem Raum 500 Netze mit je rund 400 Millionen Multiplikationsadditionen (400 MFLOPs). Sie trainierten jedes Netz 10 Epochen auf ImageNet und zeichneten die Verteilungsfunktionen in :numref:`fig_regnet-fig`. Ausgehend vom uneingeschränkten Raum AnyNetX$_A$ grenzten sie den Raum mit jeder weiteren Regel ein. Zunächst gaben sie allen vier Stufen dasselbe Flaschenhalsverhältnis $b$ vor (AnyNetX$_B$), anschließend auch dieselbe Gruppenbreite $g$ (AnyNetX$_C$). Die Fehlerverteilung blieb unverändert (mittlerer Fehler rund 49%). Schon das ist ein großer Gewinn: 6 der 16 freien Hyperparameter entfallen ohne Einbuße an Genauigkeit. Die größte Verbesserung bringt eine Nebenbedingung für die *Stufenbreiten*. Fordern wir monoton wachsende Breiten ($c_{i+1} \geq c_i$), sinkt der mittlere Fehler der Population um volle sechs Prozentpunkte auf rund 43%. Bei konstanten Breiten steigt er auf 56%, bei abnehmenden auf 63%. Monoton wachsende Tiefen ($d_{i+1} \geq d_i$) senken ihn um einen weiteren halben Prozentpunkt. Konstante oder abnehmende Tiefen schneiden schlechter ab.

<!--en:f3c273f3-->
![Verteilungsfunktionen der Fehler in den Entwurfsräumen, übernommen aus :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`. Aus jedem Raum wurden 500 Netze mit jeweils rund 400 Millionen Multiplikationsadditionen gezogen und je 10 Epochen auf ImageNet trainiert. Die Legenden geben den kleinsten und den mittleren Top-1-Fehler in Prozent an. Mit $w_i$ bezeichnet die Publikation die Breite $c_i$. Oben: gemeinsames Flaschenhalsverhältnis für alle Stufen (AnyNetX$_B$) im Vergleich zu frei wählbaren Verhältnissen (AnyNetX$_A$), danach zusätzlich eine gemeinsame Gruppenbreite (AnyNetX$_C$). Unten: zunehmende, konstante oder abnehmende Breiten von Stufe zu Stufe, danach die entsprechenden Verläufe für die Tiefen im Raum mit zunehmenden Breiten (AnyNetX$_D$).](../img/regnet-fig.png)

<!--en:01ed24d5-->
Empirische Regeln hängen vom Datensatz, vom Rechenbudget und vom Blocktyp ab. Untersuchen wir deshalb die wirksamste Regel, zunehmende gegenüber abnehmenden Breiten, mit genau derselben Methode an Populationen auf Fashion-MNIST. Wir definieren zwei Räume für zweistufige Netze mit einem Eingangsteil mit 32 Kanälen und ResNeXt-Blöcken ($b = 1$, Gruppenbreite 16). Wir ziehen Stufentiefen $d_i \in \{1, \ldots, 8\}$ und Breiten $c_i \in \{16, 32, \ldots, 256\}$ unter der Nebenbedingung $c_1 < c_2$ im einen Raum und $c_1 > c_2$ im anderen. Wie in der Publikation gleichen wir die Rechenbudgets ab. Dazu zählen wir die Multiplikationsadditionen im Eingangsteil, in jedem Block und in allen Sprungverbindungen mit Projektion (:numref:`sec_alexnet`).

<!--en:a3bf3cf4-->
Für jedes mögliche Modell ziehen wir zwei Tiefen und zwei verschiedene Breiten. Wir sortieren die Breiten gemäß der Nebenbedingung des Raums. Wir akzeptieren das Modell nur, wenn die Zahl der Multiplikationsadditionen um höchstens 20% von unserem Budget von 112 Millionen pro $96 \times 96$-Bild abweicht. Das Budget entspricht dem Modell `RegNetX32`, das wir in :numref:`subsec_regnet` trainieren.

<!--en:150c4d5a-->
Bei gleichem Rechenaufwand verwenden Modelle mit abnehmenden Breiten viele Kanäle für die frühen, hochaufgelösten Merkmalskarten. In den oberen Stufen haben sie entsprechend weniger Kanäle. Wir trainieren jede gezogene Architektur eine Epoche lang (ein Zehntel der vollständigen Trainingsdauer in :numref:`subsec_regnet`). Dazu verwenden wir den Optimierer von `AnyNet` und erfassen den Validierungsfehler.

<!--en:090eb88b-->
Wir ziehen und trainieren 12 Modelle pro Raum, insgesamt also 24 Trainingsläufe mit je einer Epoche.

<!--en:0a3a9764-->
Tragen wir die empirische Verteilungsfunktion :eqref:`eq_edf` auf, erhalten wir eine Treppenfunktion. An jedem der aufsteigend sortierten Validierungsfehler springt sie um $1/n$ nach oben.

<!--en:bab61b88-->
Hier überlappen die beiden Verteilungsfunktionen weitgehend. Bei wiederholten Ziehungen liegen die mittleren Fehler nahe 12% und unterscheiden sich um weniger als einen Prozentpunkt, meist zugunsten zunehmender Breiten. Die besten Modelle beider Räume liegen bei rund 11%. Eine Nebenbedingung, die auf ImageNet volle zwanzig Prozentpunkte Unterschied im mittleren Fehler ausmacht, bringt auf Fashion-MNIST weniger als einen Punkt. Mit 12 Modellen pro Raum lässt sich dieser kleine Unterschied kaum klar erkennen. Das Ergebnis ist ernüchternd, erklärt aber auch, wann Entwurfsregeln greifen: Auf einem einfachen Datensatz mit zehn Klassen erreicht fast jedes vernünftig entworfene CNN in einer einzigen Epoche rund 88% Genauigkeit. Architektonische Nebenbedingungen können die Populationen dann kaum voneinander abheben. Entwurfsregeln sind keine allgemeingültigen Gesetze. Ihre Wirkung hängt unmittelbar von der Komplexität der jeweiligen Aufgabe ab, an der man sie entdeckt hat.

<!--en:e3803591-->
## RegNet

<!--en:0fc767f0-->
### Entwurfsprinzipien

<!--en:c3b3c8cc-->
Aus den Nebenbedingungen, die sich auf ImageNet bewährt haben, ergibt sich der kompakte Entwurfsraum AnyNetX$_E$:

<!--en:079fb53d-->
* ein gemeinsames Flaschenhalsverhältnis $b$ für alle Stufen,

<!--en:186bb35a-->
* eine gemeinsame Gruppenbreite $g$ für alle Stufen,

<!--en:bf4495e3-->
* monoton wachsende Stufenbreiten, $c_{i} \leq c_{i+1}$,

<!--en:948823d2-->
* monoton wachsende Stufentiefen, $d_{i} \leq d_{i+1}$.

<!--en:7612532d-->
Bei den besten Netzen in AnyNetX$_E$ fällt ein weiteres Muster auf. Nummerieren wir die Blöcke von unten nach oben mit $j = 0, 1, \ldots$, dann wachsen die optimalen Blockbreiten linear gemäß

<!--en:76ca4139-->
mit der Anfangsbreite $u_0$ und der Steigung $a > 0$. Da alle Blöcke einer Stufe dieselbe Breite haben, quantisiert RegNet jedes $u_j$ auf einem geometrischen Gitter. Das ähnelt der Verdopplung der Breiten in den von Hand entworfenen Netzen aus :numref:`tab_hand_designed_stages`. Aufeinanderfolgende Blöcke mit derselben quantisierten Breite bilden eine Stufe. So entsteht eine Treppe. :numref:`fig_regnet_staircase` zeigt diese Treppe für ein RegNet. In :exref:`ex_cnn-design-quantized-widths` berechnen Sie eine weitere. Überraschenderweise stellten die Autoren auch fest, dass $b = 1$ (also ganz ohne interne Verengung) durchweg bessere Ergebnisse liefert als die Flaschenhälse mit $b = 4$ in ResNet-50 und ResNeXt-50. Die Kombination aus quantisierten linearen Breiten und $b = 1$ definiert die *RegNetX*-Familie. An die Stelle von 16 frei wählbaren Stufenparametern treten für jedes Rechenbudget wenige, leicht interpretierbare Größen.

<!--en:622424c3-->
![Quantisierte lineare Breiten für RegNetX-200MF, ein Netz mit 13 Blöcken. Die Breite jedes Blocks auf der Geraden $u_j = u_0 + a j$ (Punkte) wird der nächstgelegenen Stufe des geometrischen Gitters $u_0, u_0 m, u_0 m^2, \ldots$ (gestrichelte Linien) zugeordnet. Aufeinanderfolgende Blöcke auf gleicher Höhe bilden eine Stufe des Netzes. Mit $u_0 = 24$, $a = 36.44$ und $m = 2.49$ ergeben sich vier Stufen der Tiefen 1, 1, 4 und 7. Nach dem Runden auf Vielfache von 8 betragen ihre Breiten 24, 56, 152 und 368 Kanäle.](../img/arch-regnet-staircase.svg)

<!--en:cd355eb5-->
### RegNetX32 auf Fashion-MNIST

<!--en:b4a7cf95-->
Bauen wir nun ein kompaktes Netz nach den RegNet-Regeln und vergleichen es direkt mit ResNet-18. Die auf $96 \times 96$ vergrößerten Bilder von Fashion-MNIST hatten ursprünglich nur $28 \times 28$ Pixel. Vier Stufen mit insgesamt fünf Halbierungen würden die Gitter in den letzten Stufen auf winzige $6 \times 6$ und $3 \times 3$ verkleinern. Deshalb verwenden wir zwei Stufen mit Auflösungen von $24 \times 24$ und $12 \times 12$. In :exref:`ex_cnn-design-four-stages` erproben Sie vier Stufen. Nach einem Eingangsteil mit 32 Kanälen wählen wir von Hand zunehmende Tiefen und Breiten: für Stufe 1 die Werte $d_1 = 4$ und $c_1 = 64$, für Stufe 2 die Werte $d_2 = 6$ und $c_2 = 160$. Dazu wählen wir $b = 1$ und die gemeinsame Gruppenbreite $g = 16$. Damit ergeben sich $64/16 = 4$ und $160/16 = 10$ Gruppen. Eine Faltung im Eingangsteil, zehn Blöcke mit je drei Schichten und ein linearer Klassifikator ergeben zusammen 32 Schichten. Daher heißt das Netz `RegNetX32`.

<!--en:86f84044-->
Die Ausgabe von `layer_summary` bestätigt, dass die räumliche Auflösung stufenweise gemäß $96 \to 48 \to 24 \to 12$ sinkt, während die Kanalzahl steigt.

<!--en:3e8d7f3c-->
Zählen wir die Rechenoperationen wie in :numref:`sec_alexnet`, so benötigt `RegNetX32` 112 Millionen Multiplikationsadditionen pro $96 \times 96$-Bild. Das ist kaum mehr als ein Drittel der 319 Millionen von ResNet-18.

<!--en:5c0758b0-->
Trainieren wir `RegNetX32` auf $96 \times 96$ großen Bildern von Fashion-MNIST für 10 Epochen mit einer Batchgröße von 128. Wir verwenden den SGD-Optimierer mit Momentum aus `AnyNet` mit einer maximalen Lernrate von 0.1 und einer Aufwärmphase von einer Epoche.

<!--en:8a4c475c-->
Nach den 10 Trainingsepochen ermitteln wir mit `d2l.num_params(model)` die gesamte Parameteranzahl von `RegNetX32`. Wir geben sie zusammen mit der abschließenden Validierungsgenauigkeit `trainer.history["val_acc"][-1]` aus.

<!--en:0be8cdc7-->
Für einen kontrollierten Vergleich trainieren wir ResNet-18 mit genau demselben Optimierer aus `AnyNet`. Auch ResNet-18 gliedert sich in Eingangsteil, Hauptteil und Kopf. Deshalb leiten wir eine Klasse von `AnyNet` ab und überschreiben `stem` und `stage` mithilfe von `d2l.Residual` (:numref:`sec_resnet`). Für jede Stufe übergeben wir `(depth, width, stride)`.

<!--en:a941e187-->
Nun trainieren wir `ResNet18` für 10 Epochen auf genau denselben Daten.

<!--en:85bd46bd-->
Wie zuvor geben wir `d2l.num_params(model)` und die abschließende Validierungsgenauigkeit `trainer.history["val_acc"][-1]` aus. So können wir `ResNet18` direkt mit `RegNetX32` vergleichen.

<!--en:f652903e-->
Beide Architekturen erreichen am Ende eine Validierungsgenauigkeit von rund 94% (93.7–94.2% über mehrere Läufe). Dabei benötigt `RegNetX32` nur 519 658 Parameter gegenüber 11 175 370 bei ResNet-18. Das sind weniger als 5% der Gewichte und ein Drittel der Multiplikationsadditionen (112 Millionen gegenüber 319 Millionen). Für die Einordnung dieses Ergebnisses sind drei Punkte wichtig:

<!--en:77157e85-->
1. **Zusammenspiel von Architektur und Optimierer:** Bei einfachem SGD ohne Momentum und Gewichtszerfall (:numref:`sec_resnet`) fällt `RegNetX32` auf rund 91% zurück, während ResNet-18 rund 93% erreicht. Erst mit dem Protokoll mit Momentum und Gewichtszerfall, auf das der RegNet-Entwurfsraum abgestimmt wurde, zieht `RegNetX32` gleich.

<!--en:d62d4759-->
2. **Obergrenze durch den Datensatz:** Auch größere, vierstufige RegNets (RegNetX-400MF, -800MF, -1.6GF mit 4.8–8.3 Millionen Parametern) erreichen auf Fashion-MNIST dieselbe Obergrenze von 93–94%.

<!--en:4c1a0c24-->
3. **Rechenoperationen und GPU-Latenz:** Eine Epoche dauert bei `RegNetX32` etwa $1.5\times$ so lange wie bei ResNet-18, obwohl nur ein Drittel der Multiplikationsadditionen anfällt. Deren Zahl misst die Rechenarbeit und sagt wenig über die Laufzeit aus (:numref:`sec_efficient_cnns`). `RegNetX32` ist fast doppelt so tief, und seine vielen schmalen, gruppierten Schichten schreiben pro Bild etwa $2.5\times$ so viele Aktivierungen. Damit begrenzen die Speicherzugriffe die Geschwindigkeit. Dasselbe beobachteten die RegNet-Autoren auf ImageNet :cite:`Radosavovic.Kosaraju.Girshick.ea.2020`. Die GPU-Laufzeit hängt stärker von der Zahl der Aktivierungen ab als von der Zahl der Multiplikationsadditionen. RegNetX-Modelle sind bei gleicher Aktivierungszahl im Training und in der Ausführung etwa so schnell wie ResNets. Bei vergleichbarer Zahl von Multiplikationsadditionen sind RegNets bis zu $5\times$ schneller als EfficientNets, deren Aktivierungszahl linear mit der Zahl der Multiplikationsadditionen wächst.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:8710da22-->
Beim Vergleich von *Entwurfsräumen* untersuchen wir Architekturen anhand von Fehlerverteilungen ganzer Populationen. Dazu ziehen wir Netze aus einem parametrisierten Schema (`AnyNet`), statt uns auf einzelne Punktschätzungen zu stützen. Für jeden Raum wurden 500 Netze kurz auf ImageNet trainiert. Ein gemeinsames Flaschenhalsverhältnis $b$ und eine gemeinsame Gruppenbreite $g$ für alle Stufen kosten keine Genauigkeit. Fordern wir außerdem monoton wachsende Stufenbreiten ($u_j = u_0 + a j$, zu einer Treppe quantisiert) und Stufentiefen, sinkt der mittlere Fehler der Population von 49% auf 43%. So ergibt sich die RegNetX-Familie. Auf Fashion-MNIST erreicht `RegNetX32` wie ResNet-18 rund 94% Genauigkeit, benötigt dafür aber weniger als 5% seiner Parameter und ein Drittel seiner Multiplikationsadditionen. Das gesamte Kapitel zeigt, worauf moderne Grundnetze für die Bildverarbeitung aufbauen: auf wenigen strukturellen Bausteinen (wiederholbaren Blöcken, Normalisierung der Aktivierungen und Sprungverbindungen in Residualblöcken) und auf der konsequenten Unterscheidung zwischen Verbesserungen durch die Architektur und Verbesserungen durch das Trainingsrezept.

<!--en:2294243a-->
## Übungen

<!--en:fc80b192-->
1. [code] **Vier Stufen bei vergleichbarem Budget.** `RegNetX32` verwendet für $96 \times 96$ große Bilder von Fashion-MNIST zwei Stufen. Erweitern Sie das Netz auf die vier Stufen aus :numref:`fig_anynet_full`. Wählen Sie monoton wachsende Breiten und Tiefen und halten Sie die Abweichung der gesamten Parameteranzahl von derjenigen von `RegNetX32` innerhalb von 10%. Vergleichen Sie Validierungsgenauigkeit und Laufzeit pro Epoche mit `RegNetX32`. Geben Sie für jede der vier Stufen die räumliche Auflösung ihrer Merkmalskarten an.
   :exlabel:`ex_cnn-design-four-stages`

<!--en:a971943b-->
1. **Quantisierte lineare Breiten.** :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` parametrisieren die Blockbreiten eines Netzes mit $d$ Blöcken durch die Gerade $u_j = u_0 + a\, j$ ($j = 0, \ldots, d-1$) aus :numref:`subsec_regnet`. Anschließend ordnen sie jeden Block einem geometrischen Gitter mit dem Faktor $m$ zu:
   $$s_j = \operatorname{round}\left(\frac{\log(u_j / u_0)}{\log m}\right), \qquad \tilde{u}_j = u_0\, m^{s_j},$$
   Blöcke mit gleichem $\tilde{u}_j$ fassen sie zu einer Stufe zusammen.
    1. :numref:`fig_regnet_staircase` zeigt das Ergebnis für RegNetX-200MF. Wiederholen Sie die Berechnung für RegNetX-400MF mit $d = 22$, $u_0 = 24$, $a = 24.48$ und $m = 2.54$ (die Publikation verwendet $w_0, w_a, w_m$). Berechnen Sie die Anzahl der Stufen, die Tiefe $d_i$ jeder Stufe und die Stufenbreiten, gerundet auf das nächste Vielfache von 8.
    1. Wie verhalten sich die Anzahl der Stufen und die Treppennäherung $\tilde{u}_j \approx u_j$ für $m \to 1$ und für große Werte von $m$?
    1. Welche Ihrer gerundeten Stufenbreiten aus (a) sind durch die gemeinsame Gruppenbreite $g = 16$ nicht teilbar? Wie sollten Sie sie anpassen?
   :exlabel:`ex_cnn-design-quantized-widths`

<!--en:ae53ac4a-->
1. [code] **RegNetY mit Squeeze-and-Excitation.** RegNetY fügt das Squeeze-and-Excitation-Gate `SE` (:numref:`subsec_se`, mit Reduktionsverhältnis $r = 4$) in jeden `ResNeXtBlock` direkt nach der gruppierten $3 \times 3$-Faltung ein :cite:`Radosavovic.Kosaraju.Girshick.ea.2020`.
    1. Bauen Sie `RegNetY32` mit der Stufenkonfiguration von `RegNetX32`. Berechnen Sie mit der Formel $2c^2/r$ die Zahl der zusätzlichen Parameter und prüfen Sie das Ergebnis mit `d2l.num_params`.
    1. Trainieren Sie `RegNetY32` auf $96 \times 96$ großen Bildern von Fashion-MNIST. Vergleichen Sie die Validierungsgenauigkeit und die Laufzeit pro Epoche mit `RegNetX32`. Warum erhöht `SE` die Laufzeit messbar, obwohl fast keine zusätzlichen Multiplikationsadditionen anfallen?
   :exlabel:`ex_cnn-design-regnety`

<!--en:832b94b8-->
1. [code] **Verstöße gegen die RegNet-Regeln.** ● Implementieren Sie drei Varianten von `RegNetX32` mit ungefähr gleicher Parameteranzahl, die jeweils eine Regel von AnyNetX$_E$ (:numref:`subsec_regnet`) verletzen: (i) abnehmende Stufentiefen ($d_1 > d_2$), (ii) unterschiedliche Gruppenbreiten und (iii) ein Flaschenhalsverhältnis $b_i > 1$, das von Stufe zu Stufe variiert. Trainieren Sie jede Variante dreimal auf Fashion-MNIST. Vergleichen Sie die Ergebnisse mit `RegNetX32` und stellen Sie sie den Verteilungsfunktionen der Populationen in :numref:`fig_regnet-fig` gegenüber. Warum ist es weniger aussagekräftig, für jeden Regelverstoß ein einzelnes, von Hand ausgewähltes Netz zu testen, als zwei empirische Verteilungsfunktionen zu vergleichen?
   :exlabel:`ex_cnn-design-vionet`

<!--en:3c984cf9-->
1. [extended] **Ein Entwurfsraum für ConvNeXt.** ● Bauen Sie einen AnyNet-Entwurfsraum für zweistufige Netze mit `ConvNeXtBlock` (:numref:`sec_convnext`, ergänzen Sie das Expansionsverhältnis $t \in \{2, 4\}$ als Argument). Verwenden Sie dazu den Eingangsteil und die Downsampling-Schichten von `ConvNeXt`. Ziehen Sie Stufentiefen $d_i \in \{1, 2, 4\}$ und Breiten $c_i \in \{32, 64, 128\}$.
    1. Welche der vier Freiheitsgrade pro AnyNet-Stufe ($d_i, c_i, g_i, b_i$) haben eine direkte Entsprechung in `ConvNeXtBlock`? Welche sind durch die tiefenweise Separierbarkeit festgelegt?
    1. Ziehen Sie 16 Netze aus dem uneingeschränkten Raum, 16 mit gemeinsamem Expansionsverhältnis $t_1 = t_2$ und 16 unter den Nebenbedingungen $c_1 \le c_2$ und $d_1 \le d_2$. Trainieren Sie jedes Netz für 2 Epochen auf $96 \times 96$ großen Bildern von Fashion-MNIST. Zeichnen Sie die drei empirischen Verteilungsfunktionen der Fehler :eqref:`eq_edf` und prüfen Sie, welche RegNet-Prinzipien auch für ConvNeXt-Blöcke gelten.
   :exlabel:`ex_cnn-design-convnext-space`

<!--en:31f866eb-->
[Diskussion](https://d2l.discourse.group/t/7462)

<!--en:d418b2a7-->
[Diskussion](https://d2l.discourse.group/t/7463)

<!--en:d1c5dfb3-->
[Diskussion](https://d2l.discourse.group/t/8738)

<!--en:0e4642d5-->
[Diskussion](https://d2l.discourse.group/t/18009)

<!--en:ab210935-->
Entwurfsräume für konvolutionäre Netze (RegNet)<br>**AnyNet-Schema · Verteilungsfunktionen des Populationsfehlers · schrittweise Eingrenzung des Entwurfsraums · quantisierte lineare Breiten · RegNetX32**

<!--en:31761307-->
::: {.slide title="Vom Entwurf einzelner Modelle zum Entwurf ganzer Populationen"}

<!--en:a5c18c67-->
[Motivation und Grenzen der Einzelmodellsuche]{.kicker}

<!--en:148aec12-->
- **Grenzen der Suche nach Einzellösungen:** Manuelle Abstimmung und NAS suchen unter $1.8 \times 10^{18}$ Möglichkeiten ein einzelnes Modell für ein festes Budget. Stufentiefen wie $(3,4,6,3)$ oder $(3,3,9,3)$ bleiben dabei schwer nachvollziehbar.

<!--en:87454ec3-->
- **Regeln aus Entwurfsräumen ableiten (RegNet, Radosavovic et al., 2020):** Pro parametrisiertem Raum $n=500$ Modelle ziehen. Der Vergleich der empirischen Verteilungsfunktionen der Fehler $\hat{F}(e)$ liefert einfache Strukturregeln, die für alle FLOP-Budgets gelten.

<!--en:275ed874-->
::: {.slide title="Der Entwurfsraum von AnyNet mit 16 Freiheitsgraden"}

<!--en:9610a108-->
[Das Schema ohne Nebenbedingungen]{.kicker}

<!--en:5ae161b3-->
Das *AnyNet*-Schema gibt den Aufbau aus Eingangsteil, Hauptteil und Kopf vor. Für jede der vier Stufen $(i = 1, 2, 3, 4)$ bleiben vier Hyperparameter frei:

<!--en:5518916a-->
- **Fester Eingangsteil und Kopf:** $3\times 3$-Faltung mit Stride 2 im Eingangsteil ($c_0 = 32$ Kanäle, BN, ReLU), Global-Average-Pooling und lineare Schicht im Kopf.

<!--en:8799cfe4-->
- **Vier freie Zahlen pro Stufe ($d_i, c_i, b_i, g_i$):** Stufe $i$ umfasst $d_i$ ResNeXt-Blöcke (§{sec_resnet}) mit der Ausgabekanalzahl $c_i$, dem Flaschenhalsverhältnis $b_i \in \{1, 2, 4\}$ (Flaschenhalsbreite $c_i / b_i$) und der *Gruppenbreite* $g_i \in \{1, 2, \ldots, 32\}$ (Gruppenzahl $(c_i / b_i)/g_i$).

<!--en:edc510e6-->
- **$1.8 \times 10^{18}$ mögliche Modelle:** Für $d_i \le 16$ und $c_i \le 1024$ (Breiten als Vielfache von 8) ergeben sich $(16 \cdot 128 \cdot 3 \cdot 6)^4 \approx 1.8 \times 10^{18}$ verschiedene Architekturen.

<!--en:eed2b063-->
::: {.slide title="Implementierung des parametrisierten AnyNet-Grundnetzes"}

<!--en:30e13891-->
[Modulares Codeschema]{.kicker}

<!--en:82957f13-->
In `AnyNet` beschreibt ein 4-Tupel `(depth, num_channels, groups, bot_mul)` jede Stufe. Dabei gilt `bot_mul` $= 1/b_i$. Der erste Block jeder Stufe verringert die Auflösung mit `strides=2`:

<!--en:9cf07d72-->
::: {.slide title="Entwurfsräume anhand empirischer Verteilungsfunktionen der Fehler vergleichen"}

<!--en:2c0f30e4-->
[Statistische Methode]{.kicker}

<!--en:3f9ed38f-->
- **Stochastische Dominanz:** Liegt $\hat{F}_1(e)$ oberhalb und links von $\hat{F}_2(e)$, hält ein zufällig gezogenes Netz aus Raum 1 jede Fehlerschwelle $e$ mit höherer Wahrscheinlichkeit ein.

<!--en:b12ae4d6-->
- **Betrachtung ganzer Populationen:** Bewertet wird die gesamte Verteilung zulässiger Architekturen. Das beste Einzelmodell unterliegt Zufallsschwankungen.

<!--en:d4f49cf3-->
::: {.slide title="Schrittweise Eingrenzung von AnyNetXA bis AnyNetXE"}

<!--en:f9f0dc24-->
[Ergebnisse für Populationen auf ImageNet]{.kicker}

<!--en:1dcfb984-->
Pro Raum $n = 500$ Modelle mit je $\approx 400\text{M}$ Multiplikationsadditionen ziehen und 10 Epochen auf ImageNet trainieren. Daraus ergeben sich vier Strukturregeln:

<!--en:11e7c4b7-->
- **$\text{AnyNetX}_A \to \text{AnyNetX}_B$ (gemeinsames $b_i = b$):** Ein einheitliches Flaschenhalsverhältnis für alle Stufen erhält die Verteilungsfunktion ($49.0\% \to 49.2\%$ mittlerer Fehler). 3 freie Parameter entfallen.

<!--en:6b19ae37-->
- **$\text{AnyNetX}_B \to \text{AnyNetX}_C$ (gemeinsames $g_i = g$):** Eine einheitliche Gruppenbreite erhält die Verteilungsfunktion ebenfalls ($49.4\%$ mittlerer Fehler, 3 weitere freie Parameter entfallen).

<!--en:f07e0462-->
- **$\text{AnyNetX}_C \to \text{AnyNetX}_D$ (wachsende Breiten $c_{i+1} \ge c_i$):** Der mittlere Fehler sinkt um **6 Prozentpunkte auf $43.2\%$**. Bei abnehmenden Breiten $c_{i+1} < c_i$ steigt er auf $63.2\%$.

<!--en:2817b177-->
- **$\text{AnyNetX}_D \to \text{AnyNetX}_E$ (wachsende Tiefen $d_{i+1} \ge d_i$):** Der mittlere Fehler sinkt auf **$42.7\%$**. Unter diesen Nebenbedingungen bleiben weniger freie Stufenparameter ($16 \to 10$).

<!--en:ced90e17-->
::: {.slide title="Multiplikationsadditionen eines AnyNet-Modells zählen"}

<!--en:8365710d-->
[Bilanz des FLOP-Budgets]{.kicker}

<!--en:1aafb36f-->
- **Lineare Kosten der gruppierten $3 \times 3$-Faltung:** Bei $b=1$ und Gruppenbreite $g$ gilt Gruppenzahl $= c/g$. Damit gilt für die gruppierte $3 \times 3$-Faltung: Multiplikationsadditionen $= 9 r^2 g c$ pro Block.

<!--en:93056b14-->
::: {.slide title="Architekturen mit vergleichbarem Rechenaufwand ziehen"}

<!--en:f4579ce1-->
[Rejection-Sampling bei 112 Mio. Multiplikationsadditionen]{.kicker}

<!--en:f3933ad9-->
`sample_arch` zieht zufällige Tiefen $d_i \in \{1,\ldots,8\}$ und verschiedene Breiten $c_i \in \{16, 32, \ldots, 256\}$. Es sortiert die Breiten gemäß $c_1 < c_2$ (`increasing=True`) oder $c_1 > c_2$ (`increasing=False`) und akzeptiert Modelle mit einer Abweichung von $\pm 20\%$ vom Budget von $112\text{M}$ Multiplikationsadditionen:

<!--en:68553b1e-->
::: {.slide title="Zunehmende und abnehmende Breiten auf Fashion-MNIST im Vergleich"}

<!--en:d8398283-->
[Populationsexperiment im kleinen Maßstab]{.kicker}

<!--en:6743c5d1-->
Auswahl: $n = 12$ zweistufige Netze mit zunehmenden Breiten ($c_1 < c_2$) und $12$ mit abnehmenden Breiten ($c_1 > c_2$) bei $112\text{M} \pm 20\%$ Multiplikationsadditionen. Jedes Netz 1 Epoche auf $96\times 96$ großen Bildern von Fashion-MNIST trainieren:

<!--en:e9af2c80-->
- **Einfluss der Aufgabenkomplexität:** Auf ImageNet unterscheiden sich die Populationsmittelwerte bei $c_{i+1} \ge c_i$ und abnehmenden Breiten um **20 Prozentpunkte** ($43\%$ gegenüber $63\%$). Auf Fashion-MNIST mit 10 Klassen liegen die Fehler der beiden Verteilungen um **weniger als 1 Prozentpunkt** auseinander (Mittelwert $\approx 12\%$, bester Wert $\approx 11\%$).

<!--en:b1c9f1bd-->
- **Schlussfolgerung:** Auf einem einfachen Grauwertdatensatz erreicht fast jedes vernünftig entworfene CNN in einer Epoche $\approx 88\%$ Genauigkeit. Hochaufgelöste frühe Stufen lassen der letzten Stufe noch genug Kapazität für semantische Merkmale. Entwurfsregeln gelten nie unabhängig vom Datensatz.

<!--en:02e53641-->
::: {.slide title="Die RegNetX-Familie: Treppen aus quantisierten linearen Breiten"}

<!--en:7a2d70e6-->
[Von AnyNetXE zu RegNetX]{.kicker}

<!--en:81bd99ac-->
- **Treppe durch lineare Zunahme und Quantisierung:** Die Blockbreiten wachsen entlang einer Geraden $(u_0, a)$ und werden geometrisch gestaffelten Stufen $s_j = \operatorname{round}(\log(u_j/u_0)/\log m)$ zugeordnet.

<!--en:9d3a264c-->
- **Kein Flaschenhals ($b = 1$):** Aus 16 Hyperparametern der Stufen werden $(d, u_0, a, m, g)$.

<!--en:ecdf1c7f-->
::: {.slide title="Aufbau von RegNetX32 für Fashion-MNIST"}

<!--en:9bafa7c7-->
[Implementierung von RegNetX32]{.kicker}

<!--en:cc240e25-->
Nach den RegNetX-Regeln ($b = 1$, gemeinsame Gruppenbreite $g = 16$, zunehmende Tiefen $d_1 < d_2$, zunehmende Breiten $c_1 < c_2$) entsteht ein zweistufiges `RegNetX32` mit 32 Schichten ($1\text{ stem} + (4 + 6) \times 3\text{ block convs} + 1\text{ head} = 32\text{ layers}$):

<!--en:024e1392-->
::: {.slide title="Kontrollierter Vergleich von RegNetX32 und ResNet-18"}

<!--en:3078e51d-->
[Versuchsergebnisse]{.kicker}

<!--en:fa285b89-->
`RegNetX32` und `ResNet18` jeweils 10 Epochen auf $96\times 96$ großen Bildern von Fashion-MNIST mit demselben Optimierer aus `AnyNet` trainieren (SGD mit Momentum $0.9$, gekoppelter Gewichtszerfall $10^{-3}$, lineare Aufwärmphase über 1 Epoche bis $\eta_0 = 0.1$, dann Kosinus-Abfall):

<!--en:5fe60df9-->
| Modell ($96^2$, 10 Epochen) | Parameter | Multiplikationsadditionen | Validierungsgenauigkeit |

<!--en:b2c62a49-->
| **ResNet-18** (4 Stufen) | $11.18\text{M}$ | $319\text{M}$ | $\approx 94.0\%$ |

<!--en:a83336e5-->
| **RegNetX32** (2 Stufen) | **$0.52\text{M}$** | **$112\text{M}$** | **$\approx 94.0\%$** |

<!--en:bb08ea48-->
| **RegNetX-400MF..1.6GF** | $4.8\text{--}8.3\text{M}$ | $70\text{--}290\text{M}$ | $93\text{--}94\%$ |

<!--en:8eb721d8-->
- **$21\times$ weniger Gewichte, $3\times$ weniger Multiplikationsadditionen:** `RegNetX32` erreicht dieselbe Genauigkeit wie ResNet-18 ($\approx 94\%$) mit $<5\%$ von dessen Parametern.

<!--en:2f9ee3e7-->
::: {.slide title="Eine AnyNet-Unterklasse für den fairen Vergleich mit ResNet-18"}

<!--en:78c9556c-->
[Referenzmodell mit identischem Trainingsprotokoll]{.kicker}

<!--en:a0e85c98-->
Auch ResNet-18 besteht aus Eingangsteil, Hauptteil und Kopf. Daher eine Klasse von `AnyNet` ableiten und `stem` und `stage` mithilfe von `d2l.Residual` überschreiben. Beide Modelle erben so denselben SGD-Optimierer mit Momentum und Gewichtszerfall:

<!--en:30ce10d1-->
::: {.slide title="Drei Einflussfaktoren: Optimierer, Datensatz und Latenz"}

<!--en:d0da1ad0-->
[Fallstricke bei Implementierung und Experiment]{.kicker}

<!--en:e35aadea-->
- **1. Zusammenspiel von Architektur und Optimierer:** Ohne Momentum und Gewichtszerfall fällt `RegNetX32` auf $\approx 91\%$ Genauigkeit, während ResNet-18 $\approx 93\%$ erreicht.

<!--en:e64462a6-->
- **2. Obergrenze durch den Datensatz:** Auch größere RegNets mit 4 Stufen erreichen auf Fashion-MNIST dieselbe Obergrenze von $\approx 94\%$.

<!--en:c5af6b4e-->
- **3. GPU-Latenz:** `RegNetX32` (32 Schichten, gruppierte Faltungen) braucht pro Epoche etwa $1.5\times$ so lange wie ResNet-18 (18 Schichten, dichte Faltungen), bei einem Drittel der Multiplikationsadditionen. Die Laufzeit hängt stärker von der Aktivierungszahl ab als von den FLOPs. Bei vergleichbaren FLOPs sind RegNets **bis zu $5\times$ schneller** als EfficientNets.

<!--en:72e12ebf-->
::: {.slide title="Verständnisfragen: Quantisierung, Gruppenzahlen und empirische Verteilungsfunktionen"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:071b5283-->
Fragen zu Entwurfsräumen und zur Parametrisierung von RegNet:

<!--en:cb4d12bb-->
1. **Gruppenzahl und Gruppenbreite:** In `RegNetX32` hat Stufe 1 die Breite $c_1 = 64$, Stufe 2 die Breite $c_2 = 160$. Beide haben das Flaschenhalsverhältnis $b = 1$ und dieselbe *Gruppenbreite* $g = 16$. Wie viele Gruppen (`groups = (c_i / b) // g`) verwendet die $3\times 3$-Faltung in Stufe 1 und in Stufe 2?

<!--en:9cef72d6-->
2. **Treppe aus quantisierten linearen Breiten:** Ein RegNet habe die Anfangsbreite $u_0 = 24$, die Steigung $a = 36$ und den geometrischen Stufenfaktor $m = 2.5$. Wie groß ist für Block $j = 2$ die unquantisierte lineare Breite $u_2$, welchem Stufenindex $s_2 = \operatorname{round}(\log(u_2/u_0)/\log m)$ wird der Block zugeordnet, und welchen Wert hat $\tilde{u}_2 = u_0 m^{s_2}$?

<!--en:cb48bc48-->
3. **Empirische Verteilungsfunktionen lesen:** Im Raum $\mathcal{A}$ sei ein einzelnes Modell mit $18\%$ Fehler ein Ausreißer, der mittlere Fehler betrage aber $52\%$. Im Raum $\mathcal{B}$ betrage der kleinste Fehler $19\%$ und der mittlere Fehler $42\%$. Dabei gelte $\hat{F}_\mathcal{B}(e) > \hat{F}_\mathcal{A}(e)$ für alle $e > 20\%$. Warum schneidet $\mathcal{B}$ beim Vergleich der Entwurfsräume besser ab?

<!--en:f0f4dccf-->
::: {.slide title="Verständnisfragen: Quantisierung, Gruppenzahlen und empirische Verteilungsfunktionen (Lösungen)"}

<!--en:26a317d8-2-->
[Verständnisfragen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:d3a65c3a-->
1. **Stufe 1:** $64 / 16 = \mathbf{4\text{ groups}}$. **Stufe 2:** $160 / 16 = \mathbf{10\text{ groups}}$. Bei konstanter Gruppenbreite $g = 16$ wächst die Anzahl paralleler Gruppen (Kardinalität) proportional zur Stufenbreite.

<!--en:8b6e7bac-->
2. $u_2 = 24 + 36(2) = 96$. Wegen $\log(96/24) / \log(2.5) = \log(4) / \log(2.5) \approx 1.386 / 0.916 \approx 1.513$ ergibt das Runden den Stufenindex $s_2 = \mathbf{2}$ und die quantisierte Breite $\tilde{u}_2 = 24 \times 2.5^2 = \mathbf{150}$. Auf das nächste Vielfache von 8 gerundet beträgt sie $152$.

<!--en:0f27bf18-->
3. Ein einzelner Glückstreffer unterliegt Zufallsschwankungen und lässt sich nicht auf andere Rechenbudgets übertragen. Die stochastische Dominanz im betrachteten Schwellenbereich ($\hat{F}_\mathcal{B} > \hat{F}_\mathcal{A}$) zeigt, dass die strukturelle Nebenbedingung die Verteilung der zulässigen Netze systematisch zugunsten genauerer Modelle verschiebt.

<!--en:bcc25f39-->
::: {.slide title="Zusammenfassung: Entwurfsräume für neuronale Netze"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:d819240e-->
| Entwurfsraum oder Begriff | Nebenbedingung oder mathematische Regel | Wirkung auf die Population und Schlussfolgerung |

<!--en:e0e705a5-->
| **AnyNet ($\text{AnyNetX}_A$)** | 4 Stufen $\times$ 4 freie Größen $(d_i, c_i, b_i, g_i) \implies 16$ freie Parameter | Umfasst $\sim 1.8 \times 10^{18}$ Netze nach dem Vorbild von ResNeXt. Mittlerer Fehler nach 10 Epochen auf ImageNet: $49.0\%$ |

<!--en:f30c9ab7-->
| **Empirische Verteilungsfunktion der Fehler** | $\hat{F}(e) = \frac{1}{n}\sum_{j=1}^n \mathbf{1}(e_j \le e)$ für $n=500$ Modelle mit vergleichbaren FLOPs | Bewertet ganze Verteilungen von Architekturen. Das beste Einzelmodell unterliegt Zufallsschwankungen |

<!--en:d75d9eb7-->
| **Gemeinsame Werte für $b$ und $g$ ($\text{AnyNetX}_C$)** | $b_i = b$ und $g_i = g$ für alle 4 Stufen ($16 \to 10$ freie Parameter) | **Keine Einbuße an Genauigkeit** ($49.4\%$ mittlerer Fehler), halbiert die Komplexität der Hyperparametersuche |

<!--en:218037fe-->
| **Monotone Stufen ($\text{AnyNetX}_E$)** | Monoton wachsende Breiten $c_{i+1} \ge c_i$ und Tiefen $d_{i+1} \ge d_i$ | Senkung des mittleren ImageNet-Fehlers um **rund 6 Prozentpunkte** ($49\% \to 43\%$), Gewinn auf dem einfachen Fashion-MNIST-Datensatz $<1$ Prozentpunkt |

<!--en:f40fbf15-->
| **RegNetX-Familie** | $b = 1$ (kein Flaschenhals) und quantisierter linearer Anstieg $u_j = u_0 + a j$ | Reduziert 16 Stufenparameter auf $(d, u_0, a, m, g)$. `RegNetX32` erreicht die Genauigkeit von ResNet-18 ($\approx 94\%$) mit $21\times$ weniger Gewichten |
