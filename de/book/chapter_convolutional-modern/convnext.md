<!--en:30d49f41-->
# ConvNeXt: Schritt für Schritt zu einem modernen ResNet

<!--en:c5cc31ab-->
Alle Architekturen, die wir in diesem Kapitel bisher betrachtet haben, beruhen auf Faltungen. Bis 2020 führten konvolutionäre Netze alle wichtigen Ranglisten der Bildklassifikation an. Dann übernahmen Architekturen die Führung, die auf *Attention* beruhen und ohne eine einzige überlappende räumliche Faltung auskommen. Dem ersten Vision Transformer gelang das Ende 2020 mithilfe eines Datensatzes für das Pre-Training, der weit größer war als ImageNet. Der *Swin Transformer* :cite:`liu2021swin` kam 2021 ohne solche zusätzlichen Daten aus. Allein auf ImageNet trainiert erreichte er eine Top-1-Genauigkeit von 81.3%, ein ähnlich großes ResNet-50 dagegen 76.1%. War Attention selbst für diesen Sprung um fünf Prozentpunkte verantwortlich? Waren konvolutionäre Netze damit überholt?

<!--en:b3b4f1d4-->
Hier genügt ein kurzer Überblick über Vision Transformer. Ausführlich behandeln wir sie in :numref:`chap_transformers`. Ein *Vision Transformer* :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` zerlegt ein Bild in ein Raster aus Bild-Patches, bettet jeden Patch als Merkmalsvektor ein und verarbeitet die Sequenz mit einem Stapel identischer Blöcke. Jeder Block hat zwei Teilschichten. In der *Attention*-Schicht berechnet jeder Patch einen datenabhängig gewichteten Mittelwert anderer Patches. Danach verarbeitet ein punktweises mehrschichtiges Perzeptron (MLP) jeden Patch-Vektor unabhängig von den übrigen. Swin greift die hierarchische Struktur konvolutionärer Netze wieder auf. Es berechnet Attention in lokalen Fenstern der Größe $7 \times 7$ und verringert zwischen den Stufen die Auflösung durch Downsampling.

<!--en:b3e0ff6f-->
Nach :numref:`sec_training_recipes` haben wir Grund zur Skepsis, wenn ein Abstand von fünf Prozentpunkten allein der Attention zugeschrieben wird. Neben Attention unterscheidet sich Swin in mindestens einem Dutzend Punkten von einem ResNet aus dem Jahr 2015. Swin nutzt ein modernes Trainingsrezept über 300 Epochen, zerlegt Bilder schon im Eingangsteil in Patches, verteilt die Tiefe anders auf die Stufen und verwendet andere Aktivierungen und Normalisierungen. Um diese Einflüsse zu trennen, führten :citet:`liu2022convnet` ein wegweisendes kontrolliertes Experiment durch. Ausgehend von einem herkömmlichen ResNet-50 übernahmen sie eine Neuerung von Swin nach der anderen und maßen nach jeder Änderung die Top-1-Genauigkeit auf ImageNet, durchgehend ohne Attention. Das so entstandene rein konvolutionäre Netz *ConvNeXt* erreicht 82.0% und übertrifft damit knapp Swin-T. Seine Bausteine kennen wir bereits aus diesem Kapitel: moderne Trainingsrezepte (:numref:`sec_training_recipes`), tiefenweise Faltungen (engl. depthwise convolutions) und invertierte Flaschenhälse (engl. inverted bottlenecks) (:numref:`sec_efficient_cnns`) sowie Schichtnormalisierung (engl. layer normalization) (:numref:`sec_batch_norm`). Da jede Zeile dieser Ablationsstudie einen vollständigen Trainingslauf auf ImageNet erfordert, gehen wir die veröffentlichten Ergebnisse Schritt für Schritt durch. Dann implementieren wir ConvNeXt von Grund auf und modernisieren auf Fashion-MNIST selbst ein ResNet-18 in drei Schritten. Seine Parameterzahl haben wir für den Vergleich an die von ConvNeXt angeglichen.

<!--en:2b70bd1e-->
## Die Modernisierung Schritt für Schritt

<!--en:9e563797-->
:numref:`tab_convnext_roadmap` zeigt die genaue Abfolge der Änderungen, die :citet:`liu2022convnet` untersuchten. Jede Zeile enthält auch alle Änderungen aus den vorhergehenden Zeilen. Der Rechenaufwand bleibt dabei nahe dem von Swin-T mit rund 4.5 Milliarden Multiplikationsadditionen pro Bild (in der Arbeit als 4.5 GFLOPs angegeben).

<!--en:539829c1-->
:Schritte der Modernisierung zu ConvNeXt :cite:`liu2022convnet`. Die Tabelle zeigt die Top-1-Genauigkeit eines ResNet-50 auf ImageNet nach jeder weiteren Änderung. Jede Zeile schließt alle vorhergehenden Änderungen ein, bei annähernd konstanten Kosten. Swin-T, ein Transformer mit vergleichbaren Kosten, erreicht 81.3%.

<!--en:7514b4e5-->
| Änderung | Bedeutung | Top-1 |

<!--en:048500dc-->
| (Ausgangspunkt) | ResNet-50 mit dem Trainingsrezept von 2015 | 76.1% |

<!--en:ee1bdb96-->
| Modernes Trainingsrezept | 300 Epochen, AdamW, Mixup, Label-Smoothing, stochastische Tiefe (:numref:`sec_training_recipes`) | 78.8% |

<!--en:f7c6bf8e-->
| Verteilung auf die Stufen | Blöcke pro Stufe: $(3, 4, 6, 3) \to (3, 3, 9, 3)$ | 79.4% |

<!--en:a1c9b63f-->
| Patch-Eingangsteil | $7 \times 7$-Faltung mit Stride 2 und Pooling $\to$ eine $4 \times 4$-Faltung mit Stride 4 | 79.5% |

<!--en:ef74bc05-->
| Tiefenweise Faltungen | die $3 \times 3$-Faltung arbeitet nun tiefenweise, die Breite steigt von 64 auf 96 Kanäle | 80.5% |

<!--en:89578dea-->
| Invertierter Flaschenhals | im Blockinneren um den Faktor $4\times$ erweitern statt komprimieren | 80.6% |

<!--en:71f13a8d-->
| Große Faltungskerne | tiefenweise Faltung an den Blockanfang verlegen und den Faltungskern auf $7 \times 7$ vergrößern | 80.6% |

<!--en:aae0b946-->
| ReLU $\to$ GELU | glatte Aktivierungsfunktion wie bei Transformern | 80.6% |

<!--en:cd521e70-->
| Weniger Aktivierungen | eine Aktivierung pro Block statt drei | 81.3% |

<!--en:525058ea-->
| Weniger Normalisierungen | eine Normalisierung pro Block statt drei | 81.4% |

<!--en:34d162d0-->
| BN $\to$ LN | Schichtnormalisierung über die Kanäle an jeder Position | 81.5% |

<!--en:73b38d26-->
| Separates Downsampling | LN und anschließende $2 \times 2$-Faltung mit Stride 2 zwischen den Stufen | 82.0% |

<!--en:45a10c81-->
Schon die erste Änderung bringt den größten Einzelgewinn, ohne eine einzige Schicht des Netzes zu verändern. Wie in :numref:`sec_training_recipes` steigt die Genauigkeit des unveränderten ResNet-50 allein durch das neue Trainingsrezept um 2.7 Prozentpunkte. Von den insgesamt 5.9 Prozentpunkten zwischen dem ResNet-50 von 2015 und ConvNeXt entfallen bei dieser Reihenfolge 2.7 auf das Training und 3.2 auf die Architektur. Der Vergleich eines modernen Transformers mit einem nach dem Rezept von 2015 trainierten ResNet hat also die Einflüsse von Trainingsrezept und Architektur vermischt.

<!--en:bba128fc-->
### Aufbau des Gesamtnetzes

<!--en:831bbb6a-->
Die nächsten beiden Zeilen betreffen die Struktur des Gesamtnetzes. Swin-T verteilt die Blöcke im Verhältnis $1{:}1{:}3{:}1$ auf vier Stufen. Die größte Tiefe liegt damit in Stufe 3, wo die Merkmalskarten klein und die Berechnungen kostengünstig sind. Ändern wir die Verteilung im ResNet-50 von $(3, 4, 6, 3)$ auf $(3, 3, 9, 3)$, gewinnen wir 0.6 Prozentpunkte. Als Nächstes betrachten wir den Eingangsteil. ResNet beginnt mit einer überlappenden $7 \times 7$-Faltung mit Stride 2 und anschließendem Max-Pooling. Damit sinkt die Auflösung auf ein Viertel je Achse. Ein Vision Transformer zerlegt das Bild in disjunkte Patches und projiziert jeden mit derselben linearen Schicht. Für Patches aus vier mal vier Pixeln entspricht das algebraisch einer nicht überlappenden $4 \times 4$-Faltung mit Stride 4 (:numref:`sec_padding`). Mit diesem *Patch-Eingangsteil* (engl. patchify stem) bleibt die Genauigkeit praktisch gleich (79.5%). Wie bei einem Transformer entsteht pro $4 \times 4$-Patch ein Token-Vektor.

<!--en:e27dd1f8-->
### Tiefenweise Faltungen und der invertierte Flaschenhals

<!--en:6a1aeed8-->
Ein Transformer-Block teilt die Berechnung in zwei Schritte. Attention mischt Informationen *über räumliche Positionen hinweg*, das MLP mischt sie an jeder Position *über Kanäle hinweg*. Wie wir in :numref:`sec_depthwise_separable` gesehen haben, lässt sich die Berechnung in einem konvolutionären Netz genauso zerlegen. Eine *tiefenweise* Faltung mischt innerhalb jedes Kanals Informationen aus verschiedenen Positionen. $1 \times 1$-Faltungen mischen an jedem Pixel Informationen aus verschiedenen Kanälen. Ersetzen wir die dichte $3 \times 3$-Schicht im Flaschenhals von ResNet durch eine tiefenweise Faltung, sinkt der Rechenaufwand so weit, dass wir Stufe 1 von 64 auf 96 Kanäle verbreitern können, wie bei Swin-T. Die Genauigkeit steigt auf 80.5%.

<!--en:cc700527-->
Sobald die räumliche Faltung tiefenweise arbeitet, steigen ihre Kosten mit der Breite nur noch linear statt quadratisch. Damit entfällt der Grund, die Kanalzahl vor dem Filtern zu verringern. ConvNeXt übernimmt deshalb den *invertierten Flaschenhals* von MobileNetV2 (:numref:`sec_efficient_cnns`). Der Residualstrom bleibt schmal, während die Breite im Blockinneren um den Faktor $4\times$ wächst. Die Geometrie $1\times \to 4\times \to 1\times$ entspricht der eines Transformer-MLP. Die Genauigkeit steigt leicht auf 80.6%.

<!--en:818d022d-->
Da die räumliche Filterung jetzt wenig Rechenaufwand erfordert, lässt sich auch die Kerngröße leicht erhöhen. Zunächst wandert die tiefenweise Faltung an den Blockanfang. Dort wirkt sie auf $c$ Kanäle statt auf $4c$. Die Genauigkeit sinkt um 0.7 Prozentpunkte auf 79.9%, der einzige Rückgang in der Versuchsreihe. Vergrößern wir den Faltungskern von $3 \times 3$ auf $7 \times 7$, ist der Verlust sofort ausgeglichen. Ein $7 \times 7$-Filter hat dasselbe rezeptive Feld wie drei gestapelte $3 \times 3$-Schichten (:numref:`sec_padding`). Warum sollte ein einzelner großer Faltungskern helfen, wenn beide Varianten dasselbe Fenster abdecken? Der Stapel deckt das Fenster ungleichmäßig ab. Von einem zentralen Pixel führen weit mehr Pfade zur Ausgabe als von einem Randpixel. Das *effektive* rezeptive Feld des Stapels hat daher näherungsweise die Form einer Gauß-Verteilung und ist viel kleiner als das nominelle :cite:`Luo.Li.Urtasun.ea.2016`. Ein einziger $7 \times 7$-Faltungskern kann dagegen schon in einer Schicht für jede Position im Fenster ein beliebiges Gewicht lernen. Bei VGG (:numref:`sec_vgg`) waren dichte $7 \times 7$-Faltungskerne wegen ihrer Kosten ($k^2 c_\textrm{i} c_\textrm{o}$) kaum praktikabel. Bei einem tiefenweisen Faltungskern ($k^2 c$) erhöht der Wechsel von $3 \times 3$ auf $7 \times 7$ nur einen winzigen Teil des Rechenaufwands im Block um den Faktor $49/9 \approx 5$. Die Genauigkeit erreicht ihre Sättigung genau bei $7 \times 7$, der Fenstergröße der lokalen Attention in Swin.

<!--en:b33cc02e-->
### Aufbau der Blöcke

<!--en:98b3b5d6-->
Die letzten Änderungen betreffen einzelne Schichten im Block. Ersetzen wir ReLU durch *GELU* :cite:`Hendrycks.Gimpel.2016`, die bei Transformern bevorzugte glatte Aktivierungsfunktion, bleibt die Genauigkeit zunächst gleich (80.6%). Was die Genauigkeit *tatsächlich* erhöht, ist eine geringere *Anzahl* von Aktivierungen und Normalisierungen. Beim klassischen ResNet-Flaschenhals folgt auf jede der drei Faltungen eine Normalisierung und eine Aktivierung. Ein Transformer-Block verwendet dagegen nur eine Aktivierung im MLP und weniger Normalisierungen. Entfernen wir zwei der drei GELUs und behalten nur die zwischen den beiden $1 \times 1$-Schichten, steigt die Genauigkeit um 0.7 Prozentpunkte. Das ist der größte Einzelgewinn beim Umbau der Blöcke. (Warum das Entfernen zweier Nichtlinearitäten die Genauigkeit erhöht, ist theoretisch noch nicht schlüssig bewiesen. Zunächst ist dies ein klarer empirischer Befund.) Beschränken wir uns auf eine Normalisierung statt drei, gewinnen wir weitere 0.1 Prozentpunkte. Ersetzen wir diese letzte Batchnormalisierung durch eine *Schichtnormalisierung* über die Kanäle an jeder Position (:numref:`subsec_layer-normalization-in-bn`), kommen nochmals 0.1 Prozentpunkte hinzu (81.5%). :numref:`fig_resnet_vs_convnext_block` vergleicht den ursprünglichen ResNet-Flaschenhals mit dem so entstandenen ConvNeXt-Block. Dieser enthält auch die weiter unten beschriebene lernbare kanalweise Schichtskalierung (engl. layer scale) $\boldsymbol{\gamma}$.

<!--en:bf78d4e3-->
![Der Bottleneck-Block von ResNet-50 und der daraus entstandene ConvNeXt-Block. Von drei Normalisierungen und drei Aktivierungen bleibt jeweils eine übrig. Der Flaschenhals, der erst komprimiert, dann verarbeitet und schließlich erweitert, wird zu einem invertierten Flaschenhals mit einer tiefenweisen $7 \times 7$-Faltung am Anfang. Eine lernbare kanalweise Schichtskalierung $\boldsymbol{\gamma}$ skaliert den Zweig vor der Addition.](../img/arch-resnet-vs-convnext-block.svg)

<!--en:42161037-->
ResNet verringert die Auflösung im ersten Block jeder Stufe durch eine Faltung mit Stride 2. Swin fügt dafür eigene Schichten zwischen den Stufen ein, die Patches zusammenführen. Den letzten Schritt zu 82.0% bilden separate Downsampling-Schichten. Zwischen den Stufen steht jeweils eine Schichtnormalisierung mit anschließender $2 \times 2$-Faltung mit Stride 2. Weitere Normalisierungen nach dem Eingangsteil und vor dem linearen Kopf stabilisieren das Training. :numref:`fig_convnext` zeigt das vollständige Grundnetz von ConvNeXt-T. Es stapelt $(3, 3, 9, 3)$ Blöcke mit den Stufenbreiten $(96, 192, 384, 768)$ und folgt der klassischen Gliederung in Eingangsteil, Hauptteil und Kopf (:numref:`sec_googlenet`).

<!--en:c73b5217-->
![ConvNeXt-T. Ein Patch-Eingangsteil ersetzt die Kombination aus Faltung und Pooling. Separate Downsampling-Schichten aus LN und einer $2 \times 2$-Faltung mit Stride 2 verbinden die Stufen aus 3, 3, 9 und 3 Blöcken. Der gesamte Hauptteil kommt ohne Pooling aus.](../img/arch-convnext.svg)

<!--en:fb705e5d-->
Bei gleichem Rechenaufwand übertrifft ConvNeXt-T Swin-T ohne eine einzige Attention-Schicht. Daraus folgt keine grundsätzliche Überlegenheit von Faltungen gegenüber Attention. Vielmehr ging der Genauigkeitsunterschied von 2021 auf die Trainingsrezepte und die vielen Details des Architekturentwurfs zurück und ließ sich nicht allein durch Attention erklären. Bei vergleichbarer Abstimmung skalieren beide Ansätze ähnlich gut.

<!--en:50c31334-->
## Implementierung

<!--en:956d50c7-->
Bei der Modernisierung sind mehr Schichten weggefallen als hinzugekommen. Der ConvNeXt-Block lässt sich deshalb mit erstaunlich wenig Code implementieren.

<!--en:87a6e561-->
### Der ConvNeXt-Block

<!--en:fa3d9c60-->
Jeder Block beginnt mit einer tiefenweisen $7 \times 7$-Faltung und einer Schichtnormalisierung an jeder Position. Es folgen eine $1 \times 1$-Erweiterung ($c \to 4c$), eine GELU und eine $1 \times 1$-Projektion ($4c \to c$). Zum Ergebnis addiert der Block seine ursprüngliche Eingabe. Zwei Maßnahmen aus der Arbeit stabilisieren das Training. Die *Schichtskalierung* :cite:`touvron2021cait` multipliziert die Ausgabe des Residualzweigs kanalweise mit einem lernbaren Vektor $\boldsymbol{\gamma}$, dessen Einträge anfangs $10^{-6}$ betragen. Dieser Vektor ist vom Skalierungsfaktor der Batchnormalisierung zu unterscheiden. So bildet jeder Block anfangs nahezu die Identität ab, was den Einstieg in die Optimierung tiefer Netze erleichtert. Für einen Block mit Schichtnormalisierung hat das denselben Effekt wie beim ResNet die Initialisierung des letzten Skalierungsfaktors der Batchnormalisierung im Residualzweig mit null (:numref:`subsec_residual-blks`). Die *stochastische Tiefe* (engl. stochastic depth) lässt während des Trainings mit `d2l.drop_path` zufällig ganze Residualzweige aus (:numref:`sec_training_recipes`).

<!--en:99c3d4f8-->
Die Schichtnormalisierung arbeitet an jeder räumlichen Position über den Kanalvektor. Bei einem Tensor mit den Kanälen an letzter Stelle, `(batch, height, width, channels)`, genügt dafür die gewöhnliche `LayerNorm` über die letzte Achse. Die beiden $1 \times 1$-Faltungen sind dann gewöhnliche lineare Schichten über dieselbe Achse. Wie :citet:`liu2022convnet` ziehen wir die Anfangswerte aller Gewichte der Faltungsschichten und der linearen Schichten aus $\mathcal{N}(0, 0.02^2)$ und setzen alle Bias-Werte auf null.

<!--en:10cd6d69-->
Beim Speichern der Bilder steht die Kanalachse vor den räumlichen Achsen. Der Block normalisiert deshalb über `axis=1` und verwendet für die $1 \times 1$-Faltungen weiterhin Faltungsschichten.

<!--en:2a72720c-->
Beim Speichern der Bilder steht die Kanalachse vor den räumlichen Achsen. Deshalb verschiebt `LayerNorm2d` die Kanalachse für die Normalisierung an die letzte Stelle und danach wieder zurück. Dasselbe tut der Block vor und nach seinen beiden linearen Schichten.

<!--en:2e5c3cbe-->
Beim Speichern der Bilder steht die Kanalachse an letzter Stelle. Der Block nutzt deshalb direkt die Schichtnormalisierung und die vollständig verbundenen Schichten der Bibliothek.

<!--en:9076b3b8-->
### Das vollständige Netz

<!--en:d3acc2b1-->
Wie bei VGG (:numref:`sec_vgg`) durchläuft der Code für die gesamte Architektur ein Tupel, das jede Stufe durch `(depth, channels)` beschreibt. Am Anfang stehen der Patch-Eingangsteil und eine Schichtnormalisierung. Separate Downsampling-Schichten verbinden die Stufen. Der Hauptteil kommt ganz ohne räumliches Pooling aus. Die Wahrscheinlichkeit, dass die stochastische Tiefe einen Zweig auslässt, steigt linear von 0 im ersten Block auf `drop_path_max` im letzten. Für Fashion-MNIST übernehmen wir das kompakte Profil *ConvNeXt-atto* mit den Stufenbreiten $(40, 80, 160, 320)$ und den Stufentiefen $(2, 2, 6, 2)$. Wie bei ConvNeXt-T bleibt die dritte Stufe die tiefste. Da wir `norm` als Argument übergeben und `stem` als Methode definieren, können wir beide in der folgenden Ablationsstudie leicht austauschen.

<!--en:7ac5c063-->
Bei $64 \times 64$ großen Eingaben aus Fashion-MNIST erzeugt der Patch-Eingangsteil ein $16 \times 16$-Raster. Die drei Downsampling-Schichten verringern die Auflösung nacheinander auf $8 \times 8$, $4 \times 4$ und $2 \times 2$. Danach folgt das Global-Average-Pooling. Prüfen wir die Form der Ausgabe und zählen wir die trainierbaren Parameter.

<!--en:2bad9a37-->
Unser ConvNeXt-atto hat genau 3 376 450 Parameter, rund ein Drittel der 11.2 Millionen von ResNet-18 (:numref:`sec_resnet`). Jede falsch konfigurierte Breite oder Kerngröße verändert diese Zahl unmittelbar. Die genaue Parameteranzahl bietet daher eine zuverlässige Kontrolle für jede Neuimplementierung.

<!--en:d5dc597f-->
## Modernisierung auf Fashion-MNIST in drei Schritten

<!--en:fb403840-->
Jede Änderung in :numref:`tab_convnext_roadmap` wurde auf 1.3 Millionen ImageNet-Fotos der Größe $224 \times 224$ geprüft. Bewähren sich dieselben Architekturentscheidungen auch bei kleinen $64 \times 64$-Graustufenbildern von Kleidungsstücken? Dazu führen wir eine eigene kurze Versuchsreihe durch. Als Vergleich dient ein ResNet-18, dessen Grundbreite wir von 64 auf 35 Kanäle verringern. Mit 3.35 Millionen Parametern hat es etwa ebenso viele wie unser ConvNeXt. Dieses ResNet ändern wir in drei aufeinander aufbauenden Schritten. Wir trainieren alle Modelle 20 Epochen lang auf dem vollständigen Fashion-MNIST mit dem kompakten modernen Trainingsrezept aus :numref:`sec_training_recipes`. Es kombiniert AdamW mit entkoppeltem Gewichtszerfall, eine lineare Aufwärmphase über 10% der Trainingsschritte mit anschließendem Kosinus-Abfall sowie Label-Smoothing und Mixup. :numref:`chap_optimization` behandelt AdamW ausführlich. Um den Einfluss der Architektur zu isolieren, setzen wir die Rate der stochastischen Tiefe bei ConvNeXt auf 0. Damit ist die Regularisierung für beide Modellfamilien identisch.

<!--en:a3efe332-->
Wir fassen das Trainingsverfahren im Mixin `ModernRecipe` zusammen. So erbt jedes Modell in der Versuchsreihe denselben Optimierer, denselben Lernratenplan und denselben Trainingsschritt mit gemischten Beispielen. Als Validierungsverlust dient weiterhin die gewöhnliche Kreuzentropie.

<!--en:897adc24-->
Unser Vergleichsmodell `CompactResNet18` stapelt im Hauptteil `d2l.Residual`-Blöcke (:numref:`sec_resnet`) zu Stufen der Breiten 35, 70, 140 und 280. Davor stehen der übliche $7 \times 7$-Eingangsteil von ResNet mit Stride 2 und das Max-Pooling.

<!--en:ae3af592-->
Wir bündeln die Änderungen der Arbeit nach dem betroffenen Teil der Architektur in drei Schritten. (1) Wir ersetzen die Blöcke und Stufen von ResNet durch die von ConvNeXt: invertierte $7 \times 7$-Flaschenhälse, eine Blockverteilung von $(2, 2, 6, 2)$, je eine GELU und eine Normalisierung pro Block, Schichtskalierung und separates Downsampling. Eingangsteil und Batchnormalisierung von ResNet behalten wir bei. Damit decken wir acht Zeilen von :numref:`tab_convnext_roadmap` ab. (2) Wir ersetzen die Batchnormalisierung durch Schichtnormalisierung. (3) An die Stelle des ResNet-Eingangsteils tritt der $4 \times 4$-Patch-Eingangsteil. Damit ist ConvNeXt vollständig. (In der Originalarbeit wurde der Eingangsteil früher ausgetauscht. Die endgültige Architektur ist von der Reihenfolge unabhängig, auch wenn der Beitrag der einzelnen Schritte leicht variieren kann.) Indem wir `stem` in `ResNetStemConvNeXt` überschreiben und `norm` umschalten, können wir alle vier Modelle mit wenig Code implementieren.

<!--en:a96fac67-->
Trainieren wir die vier Modelle nacheinander und halten wir jeweils die Parameteranzahl und die abschließende Validierungsgenauigkeit fest.

<!--en:bdb0ba0d-->
Nun durchlaufen wir `results` und geben für jeden Schritt in `roadmap` die Parameteranzahl und die Validierungsgenauigkeit nach 20 Epochen aus.

<!--en:91750d9c-->
:Unsere Versuchsreihe auf Fashion-MNIST im Vergleich zur Originalarbeit. Die letzte Spalte zeigt die Validierungsgenauigkeit nach 20 Epochen mit dem modernen Trainingsrezept bei einer Auflösung von $64 \times 64$. Die Rundung berücksichtigt, wie stark die Ergebnisse bei wiederholten Läufen schwanken. Die ImageNet-Spalte summiert für jeden Schritt die Zugewinne an Top-1-Genauigkeit aus den zugehörigen Zeilen von :numref:`tab_convnext_roadmap`.

<!--en:dfbf7dc3-->
| Schritt | Zeilen der Originalarbeit | Zugewinn auf ImageNet | Parameter | Validierungsgenauigkeit |

<!--en:569dbb99-->
| Kompaktes ResNet-18 | (ResNet-50 mit dem modernen Trainingsrezept) | | 3.35 Mio. | ≈ 94% |

<!--en:da372328-->
| ConvNeXt-Blöcke und -Stufen, BN | von der Blockverteilung bis zu den reduzierten Normalisierungen, dazu separates Downsampling | +3.0 | 3.38 Mio. | ≈ 94% |

<!--en:bb2785fb-->
| BN $\to$ LN | BN $\to$ LN | +0.1 | 3.38 Mio. | ≈ 94% |

<!--en:7b866602-->
| Patch-Eingangsteil (unser ConvNeXt) | Patch-Eingangsteil | +0.1 | 3.38 Mio. | ≈ 92% |

<!--en:df64c61a-->
Die Ergebnisse in :numref:`tab_convnext_local` zeigen genau, welche Änderungen sich auf $64 \times 64$ große Bilder aus Fashion-MNIST übertragen lassen. Behalten wir den überlappenden ResNet-Eingangsteil bei, erreichen die modernisierten Blöcke von ConvNeXt mit ihrer Blockverteilung und Schichtnormalisierung dieselbe Genauigkeit wie unser `CompactResNet18` mit vergleichbarer Parameteranzahl (rund 94% bei etwa 43–46 Millionen Multiplikationsadditionen pro Bild). Der gesamte Rückgang um rund 2 Prozentpunkte auf etwa 92% geht auf einen einzigen Schritt zurück, der bei ImageNet mit $224 \times 224$ Pixeln praktisch keinen Einfluss hatte: Der nicht überlappende $4 \times 4$-Patch-Eingangsteil mit Stride 4 erzeugt mit einer einzigen linearen Projektion direkt ein grobes $16 \times 16$-Raster. Architekturentscheidungen für die ImageNet-Auflösung lassen sich also nicht ohne Weiteres auf Bilder mit niedriger Auflösung übertragen. Eine kumulative Ablationsstudie zeigt unmittelbar, welchen Baustein wir anpassen müssen.

<!--en:89c23ea2-->
## Grenzen und offene Fragen

<!--en:445b387f-->
ConvNeXt hat gezeigt, dass rein konvolutionäre Netze bei der überwachten Klassifikation mit hierarchischen Vision Transformern gleichziehen, wenn beide nach vergleichbaren Rezepten trainieren. Zwei Fragen stehen weiterhin im Mittelpunkt der Forschung:

<!--en:da5732b3-->
1. **Maskiertes selbstüberwachtes Pre-Training:** Bei einer Sequenz aus Patch-Token lassen sich wie bei maskierten Autoencodern :cite:`he2022masked` zufällig ausgewählte Bild-Patches weglassen. Dichte Faltungen mit gleitendem Fenster tragen dagegen Informationen über die Grenzen maskierter Bereiche hinweg. *ConvNeXt V2* :cite:`woo2023convnextv2` schließt diese Lücke mit dünn besetzten Faltungen und Global Response Normalization (:exref:`ex_convnext-global-response-normalization`).

<!--en:dacf6f37-->
2. **Globaler Kontext:** Selbst ein tiefenweiser $7 \times 7$-Filter wirkt nur lokal. Offen bleibt, wie tiefenweise Faltungen mit extrem großen Faltungskernen :cite:`ding2022replknet`, verformbare Faltungen und hybride Stapel aus Faltungs- und Attention-Schichten bei den größten heute untersuchten Modellgrößen im Vergleich abschneiden.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:1ca2bb4d-->
Ausgehend von einem herkömmlichen ResNet-50 übernahm ConvNeXt Schritt für Schritt Neuerungen nach dem Vorbild von Swin: ein modernes Trainingsrezept über 300 Epochen (+2.7 Prozentpunkte), eine Blockverteilung im Verhältnis $1{:}1{:}3{:}1$, einen $4 \times 4$-Patch-Eingangsteil mit Stride 4, eine tiefenweise $7 \times 7$-Faltung am Anfang eines invertierten Flaschenhalses mit Erweiterungsfaktor $4\times$, je eine GELU und eine LayerNorm pro Block, Schichtskalierung und separate $2 \times 2$-Downsampling-Schichten mit Stride 2. Zusammen erreichen diese Änderungen auf ImageNet 82.0% ohne Attention. Auf $64 \times 64$ großen Bildern aus Fashion-MNIST erzielen der ConvNeXt-Hauptteil und LayerNorm dieselbe Genauigkeit wie ein ResNet-18 mit vergleichbarer Parameteranzahl (etwa 94%). Der grobe $4 \times 4$-Patch-Eingangsteil kostet dagegen rund 2 Prozentpunkte.

<!--en:2294243a-->
## Übungen

<!--en:c936467e-->
1. **Rechenaufwand beim Verschieben der tiefenweisen Faltung und beim Vergrößern des Faltungskerns.** Unmittelbar vor der Zeile „Große Faltungskerne“ in :numref:`tab_convnext_roadmap` erweitert jeder Block die Kanalzahl gemäß $c \to 4c$ mit einer $1 \times 1$-Faltung, filtert die $4c$ Kanäle mit einer tiefenweisen $3 \times 3$-Faltung und projiziert gemäß $4c \to c$ mit einer zweiten $1 \times 1$-Faltung. Im Schritt „Große Faltungskerne“ rückt die tiefenweise Faltung zunächst an den *Anfang* des Blocks (wo sie auf $c$ Kanäle statt auf $4c$ wirkt). Danach wächst ihr Faltungskern auf $7 \times 7$.
    1. Berechnen Sie die Anzahl der Multiplikationsadditionen pro Pixel für die tiefenweise Schicht vor dem Verschieben ($3 \times 3$ auf $4c$), nach dem Verschieben ($3 \times 3$ auf $c$) und nach dem Vergrößern auf $7 \times 7$ bei $c$ Kanälen. Vergleichen Sie die Nettoänderung mit den zusätzlichen Kosten einer Vergrößerung auf $7 \times 7$ *an der ursprünglichen Stelle* bei $4c$ Kanälen.
    1. Welchen Anteil an allen Multiplikationsadditionen pro Pixel im `ConvNeXtBlock` ($49c + 8c^2$) hat die tiefenweise $7 \times 7$-Faltung bei $c = 96$ und bei $c = 384$?

<!--en:c3ea99ca-->
1. **Verteilung der Parameter in ConvNeXt.**
    1. Berechnen Sie für das Modell `ConvNeXt` aus diesem Abschnitt ($3{,}376{,}450$ Parameter) den genauen Anteil der Parameter, der jeweils auf folgende Komponenten entfällt: (i) die tiefenweisen $7 \times 7$-Faltungen, (ii) die $1 \times 1$-Schichten zur Erweiterung und Projektion, (iii) die Vektoren für Normalisierung und Schichtskalierung in den Blöcken, (iv) die Downsampling-Schichten sowie (v) Eingangsteil und Kopf.
    1. Geben Sie die Parameteranzahl eines `ConvNeXtBlock` und eines zweischichtigen `Residual`-Blocks (:numref:`sec_resnet`, ohne $1 \times 1$-Sprungverbindung) als Funktion der Breite $c$ an. Um welchen Faktor ist `ConvNeXtBlock` für große $c$ asymptotisch parametereffizienter und welche Entwurfsentscheidung erklärt diesen Faktor?

<!--en:d1bd7ee0-->
1. [code] **Schichtskalierung und stochastische Tiefe im Vergleich.** Verwenden Sie das Trainingsrezept aus :numref:`tab_convnext_local` und trainieren Sie `ConvNeXt` in drei Varianten: mit dem Standardanfangswert $\gamma_0 = 10^{-6}$ für die Schichtskalierung, mit $\gamma_0 = 1$ und ganz ohne Schichtskalierung. Vergleichen Sie die Trainingskurven. Sowohl die Schichtskalierung mit Anfangswert $10^{-6}$ als auch die stochastische Tiefe (:numref:`sec_training_recipes`) verringern zu Trainingsbeginn die effektive Tiefe des Netzes. Vergleichen Sie die Wirkungsweise der beiden Mechanismen. Wann ist der jeweilige Mechanismus beim Training aktiv und verändert er das Netz auch bei der Auswertung?

<!--en:cbfe078b-->
1. [code] **Global Response Normalization (ConvNeXt V2).** ConvNeXt V2 :cite:`woo2023convnextv2` verzichtet auf die Schichtskalierung und fügt stattdessen in jedem Block direkt nach der GELU eine Global Response Normalization (GRN) ein. Sei für eine Merkmalskarte $\mathbf{X}$ mit $C$ Kanälen $g_c = \|\mathbf{X}_c\|_2$ die räumliche $\ell_2$-Norm des Kanals $c$ und sei $\bar{g} = \frac{1}{C}\sum_{c=1}^C g_c$. Die GRN skaliert jeden Kanal anhand seiner relativen Antwort mit einer lernbaren affinen Transformation und einer Residualverbindung:
   $$\mathrm{GRN}(\mathbf{X})_c = \gamma_c \frac{g_c}{\bar{g} + \epsilon} \mathbf{X}_c + \beta_c + \mathbf{X}_c,$$
   Dabei gilt $\epsilon = 10^{-6}$. Die Vektoren $\boldsymbol{\gamma}, \boldsymbol{\beta}$ initialisieren wir mit null, sodass die GRN anfangs die Identität abbildet. Implementieren Sie die GRN in wenigen Codezeilen, fügen Sie sie anstelle der Schichtskalierung in `ConvNeXtBlock` ein und vergleichen Sie die Validierungsgenauigkeit mit dem Trainingsrezept aus :numref:`tab_convnext_local`.
   :exlabel:`ex_convnext-global-response-normalization`

<!--en:efc082d2-->
1. [code] **Einfluss der Kerngröße bei tiefenweisen Faltungen.** ● Verwenden Sie das Trainingsrezept aus :numref:`tab_convnext_local` und trainieren Sie `ConvNeXt` mit tiefenweisen Faltungskernen der Größen $3 \times 3$, $7 \times 7$ und $11 \times 11$ (Padding 1, 3 bzw. 5). Vergleichen Sie Parameteranzahl, Multiplikationsadditionen, Laufzeit pro Epoche und Validierungsgenauigkeit. Warum steigt die Laufzeit pro Epoche beim Wechsel von $7 \times 7$ auf $11 \times 11$ viel stärker, als die Gesamtzahl der Multiplikationsadditionen erwarten lässt?

<!--en:f083bf2d-->
1. [code] **2×2-Ablationsstudie zu Architektur und Trainingsrezept.** ● Die erste und die letzte Zeile von :numref:`tab_convnext_local` vergleichen `CompactResNet18` und `ModernConvNeXt` unter dem modernen Trainingsrezept mit AdamW, Mixup und Label-Smoothing. Vervollständigen Sie das $2 \times 2$-Schema, indem Sie beide Architekturen mit Rezept A aus :numref:`sec_training_recipes` trainieren (SGD mit Momentum, stufenweise Lernratensenkung, weder Mixup noch Label-Smoothing). Stimmen Sie dabei die maximale Lernrate neu ab. Steigert das moderne Trainingsrezept die Genauigkeit beider Architekturen gleichermaßen?

<!--en:edb361f4-->
[Diskussionen](https://d2l.discourse.group/t/36762)

<!--en:839b6fcc-->
ConvNeXt: Schritt für Schritt zu einem modernen ResNet<br>**Kontrollierte Ablationsstudie mit Swin-T als Vergleich · tiefenweise $7\times 7$-Faltung und invertierter Flaschenhals mit Erweiterungsfaktor $4\times$ · je eine GELU und LayerNorm · Fallstrick beim Patch-Eingangsteil**

<!--en:a413214a-->
::: {.slide title="Hat Self-Attention konvolutionäre Netze 2021 überflüssig gemacht?"}

<!--en:b8129b07-->
[Motivation und der Vergleich mit Swin-T]{.kicker}

<!--en:745bfbf6-->
- **Swin-T im Jahr 2021:** Swin-T ($4.5\textrm{G}$ MACs) erreichte auf ImageNet **$81.3\%$** Top-1-Genauigkeit, ResNet-50 von 2015 dagegen **$76.1\%$**.

<!--en:aa02b7da-->
- **Vermischte Einflüsse:** Swin-T änderte auch das Trainingsrezept (300 Epochen mit AdamW), die Blockverteilung auf die Stufen, den Eingangsteil, die Geometrie des Flaschenhalses und die Normalisierung.

<!--en:b0d0a715-->
- **ConvNeXt (Liu et al., 2022):** Durch schrittweise Modernisierung erreicht ResNet-50 **$82.0\%$** *ohne eine einzige Attention-Schicht*.

<!--en:18e6e279-->
::: {.slide title="In elf Schritten zu ConvNeXt"}

<!--en:df528262-->
[Kumulative Ablationsstudie auf ImageNet]{.kicker}

<!--en:b2d2e4e6-->
- **Schritt 1 – modernes Trainingsrezept mit 300 Epochen (+2.7%):** Genauigkeit des unveränderten ResNet-50: $76.1\% \to 78.8\%$ (46% des Gesamtgewinns).

<!--en:a72818cf-->
- **Schritte 2–6 – Gesamtaufbau und invertierter $7 \times 7$-Flaschenhals (+1.8%):** $(3,3,9,3)$ Blöcke pro Stufe, $4 \times 4_{s4}$-Patch-Eingangsteil, tiefenweise $7 \times 7$-Faltung am Anfang des invertierten Flaschenhalses mit $4\times$ Erweiterung ($80.6\%$).

<!--en:ce2447a8-->
- **Schritte 7–11 – Feinarbeit am Block (+1.4%):** 1 GELU, 1 LayerNorm, Schichtskalierung und separate $2 \times 2_{s2}$-Downsampling-Schichten (**$82.0\%$**).

<!--en:70abae3c-->
::: {.slide title="Gesamtaufbau: Trainingsrezept, Stufenverhältnis 1:1:3:1 und Patch-Eingangsteil"}

<!--en:971aa8b6-->
[Schritte 1–3: Aufbau des Gesamtnetzes]{.kicker}

<!--en:8e16de5b-->
- **Blockverteilung $(3,4,6,3) \to (3,3,9,3)$ (+0.6%):** 50% der Blöcke liegen in Stufe 3 (Auflösung $14 \times 14$).

<!--en:5433ef7d-->
- **Patch-Eingangsteil (+0.1% bei $224^2$):** Eine überlappende $7 \times 7_{s2}$-Faltung mit Max-Pooling weicht einer nicht überlappenden $4 \times 4$-Faltung mit Stride 4. Sie entspricht der Patch-Einbettung beim ViT, ist aber für $64 \times 64$-Miniaturbilder zu grob.

<!--en:0c3eb298-->
::: {.slide title="Blockaufbau: Tiefenweise und 1×1-Faltung wie beim Transformer"}

<!--en:c3eba31e-->
[Schritte 4–6: Faktorisierung und 7×7-Faltungskerne]{.kicker}

<!--en:90679531-->
- **Tiefenweise Faltung am Blockanfang:** Steht $\textrm{DWConv}_{7\times 7}$ *vor* der Erweiterung $c \to 4c$, filtert sie $c$ Kanäle statt $4c$ ($49c$ gegenüber $196c$ MACs/Pixel).

<!--en:629e58bd-->
- **Großer räumlicher Faltungskern für $<6\%$ der MACs im Block:** Insgesamt benötigt der Block $49c + 8c^2$ MACs pro Pixel. Bei $c=96$ entfallen auf den räumlichen $7 \times 7$-Filter nur **$6.0\%$** der MACs ($1.6\%$ bei $c=384$).

<!--en:6d4d09ac-->
::: {.slide title="Warum weniger Aktivierungen und Normalisierungen die Genauigkeit erhöhen"}

<!--en:c52f312f-->
[Schritte 7–11: Änderungen im Block]{.kicker}

<!--en:7f3fd738-->
Der Vergleich des Flaschenhalses von ResNet-50 mit dem daraus entstandenen ConvNeXt-Block zeigt, wie sich durch den Abbau überflüssiger Aktivierungen und Normalisierungen die Genauigkeit um $+1.4$ Prozentpunkte steigern ließ:

<!--en:92a62e2a-->
- **Eine GELU pro Block ($+0.7\%$):** Beim klassischen ResNet folgt auf jede Faltung eine ReLU (3 pro Block). Eine *einzige* GELU zwischen den beiden $1\times 1$-Schichten, wie im MLP eines Transformers, bringt den größten Einzelgewinn beim Umbau des Blocks.

<!--en:7b1301e4-->
- **Eine LayerNorm pro Block ($+0.2\%$):** Eine einzige `LayerNorm` normalisiert pro Pixel über die Kanäle, direkt nach der tiefenweisen $7\times 7$-Faltung.

<!--en:2901ed94-->
- **Schichtskalierung $\boldsymbol{\gamma} \in \mathbb{R}^c$ (Anfangswert $10^{-6}$):** Skaliert den Residualzweig vor der Addition. So liegen auch tiefe Blöcke anfangs nahe an der Identität.

<!--en:8de6e316-->
::: {.slide title="Separate Downsampling-Schichten und das vollständige ConvNeXt-T"}

<!--en:ca6f2a60-->
[Aufbau des gesamten Grundnetzes]{.kicker}

<!--en:c2f554c2-->
ResNet verringert die Auflösung durch eine Faltung mit Stride 2 *im* ersten Residualblock jeder Stufe. ConvNeXt führt Patches durch eigene Schichten aus $\textrm{LayerNorm} + 2\times 2\text{ stride-2 conv}$ *zwischen* den Stufen zusammen ($+0.5\%$):

<!--en:92f8dd11-->
- **Kein Pooling im Hauptteil:** Das Downsampling übernehmen ausschließlich nicht überlappende Faltungen mit Stride 4 ($4\times 4$ im Eingangsteil) und Stride 2 ($2\times 2$ zwischen den Stufen).

<!--en:07569a9f-->
- **Normalisierungen für stabiles Training:** `LayerNorm` vor jeder $2\times 2$-Downsampling-Faltung mit Stride 2, nach dem Eingangsteil und nach dem Global-Average-Pooling verhindert sprunghafte Anstiege der Aktivierungsstärke beim Zusammenführen räumlicher Token.

<!--en:aef0235d-->
- **Endergebnis auf ImageNet:** **82.0%** (82.1% mit abgestimmtem EMA), gegenüber 81.3% für Swin-T bei gleichen FLOPs und höherem Durchsatz.

<!--en:9525227f-->
::: {.slide title="Implementierung des ConvNeXt-Blocks"}

<!--en:a4fbb6b9-->
[Umsetzung im Code]{.kicker}

<!--en:79705590-->
::: {.slide title="Aufbau der vollständigen ConvNeXt-Klasse"}

<!--en:7fd9e4ea-->
[Implementierung des Grundnetzes]{.kicker}

<!--en:cdf6505c-->
`ConvNeXt` verringert die Auflösung zwischen den Stufen. Die Auslasswahrscheinlichkeit steigt von Block zu Block linear:

<!--en:979e2f41-->
::: {.slide title="Tensorformen und Parameteranzahl von ConvNeXt-Atto prüfen"}

<!--en:bfee07a2-->
[Plausibilitätsprüfung: 3 376 450 Parameter]{.kicker}

<!--en:48f0018e-->
- **Kompakte Atto-Pyramide:** Mit den Tiefen $(2, 2, 6, 2)$ und Breiten $(40, 80, 160, 320)$ ergeben sich $3.38\textrm{M}$ Parameter ($1/3$ der Parameteranzahl von ResNet-18).

<!--en:cfd6719c-->
::: {.slide title="Gemeinsames ModernRecipe-Mixin für eine kontrollierte Ablationsstudie"}

<!--en:d4c97741-->
[Einheitliches Trainingsverfahren]{.kicker}

<!--en:442240ba-->
`ModernRecipe` legt Optimierer und Regularisierung für alle Schritte der Versuchsreihe fest:

<!--en:52dc538f-->
::: {.slide title="Kontrollierte Ablationsstudie in drei Schritten auf Fashion-MNIST mit 64×64 Pixeln"}

<!--en:5b3f4289-->
[Schritte der eigenen Versuchsreihe]{.kicker}

<!--en:526101a8-->
Bei festem `ModernRecipe` wird aus einem `CompactResNet18` mit angeglichener Parameteranzahl in drei Schritten ein `ConvNeXt`:

<!--en:7baaece1-->
::: {.slide title="Wann Architekturentscheidungen für ImageNet bei kleinen Bildern schaden"}

<!--en:793b88f8-->
[Ablationsergebnisse und der Einfluss der Auflösung]{.kicker}

<!--en:958965a4-->
- **Gleichstand mit Hauptteil und LN bei $\approx 94.0\%$:** Mit überlappendem Eingangsteil erreicht das Modell dieselbe Genauigkeit wie `CompactResNet18`.

<!--en:b531a4ca-->
- **Nachteil des Patch-Eingangsteils ($\approx -2$ Prozentpunkte):** Der nicht überlappende $4 \times 4_{s4}$-Eingangsteil ist für $64 \times 64$-Miniaturbilder zu grob.

<!--en:32e76c34-->
::: {.slide title="Aktuelle Forschung: Maskiertes Pre-Training und Global Response Normalization"}

<!--en:638a8d0c-->
[ConvNeXt V2 und offene Fragen]{.kicker}

<!--en:b8c74fb7-->
- **MAE-Pre-Training mit dünn besetzten Faltungen:** Dünn besetzte Submannigfaltigkeitsfaltungen verhindern, dass Nullwerte aus maskierten Bereichen auf benachbarte Positionen übergreifen.

<!--en:79e05660-->
- **Global Response Normalization (GRN):** Normalisiert die L2-Normen der Kanäle $g_c = \|\mathbf{X}_c\|_2$ relativ zum Mittelwert $\bar{g}$ der Schicht. Das verhindert, dass Kanäle inaktiv werden oder kollabieren.

<!--en:a17b64a1-->
::: {.slide title="Verständnisfragen: FLOPs pro Block, Parameter und Schichtskalierung"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:fef1ccdc-->
- **1. Verteilung der MACs im Block ($c=96$):** Die tiefenweise $7 \times 7$-Faltung benötigt $49c = 4{,}704$ MACs pro Pixel (**nur 6.0%**), gegenüber $8c^2 = 73{,}728$ für die beiden $1 \times 1$-Schichten.

<!--en:28079747-->
- **2. Warum die tiefenweise Faltung an den Anfang rücken?** Wer $c$ Kanäle statt $4c$ filtert, verringert die Kosten um den Faktor $4\times$ ($49c$ gegenüber $196c$).

<!--en:ec52d9fe-->
- **3. LayerScale und DropPath:** LayerScale ($\gamma_0 = 10^{-6}$) multipliziert deterministisch und bleibt bei der Auswertung aktiv. DropPath schaltet Zweige zufällig und lässt bei der Auswertung alle passieren ($b \equiv 1$).

<!--en:b818de07-->
::: {.slide title="Zusammenfassung: Was ConvNeXt zeigt"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:aca28f12-->
| Entwurfsentscheidung | Klassisches ResNet-50 (2015) | Modernisiertes ConvNeXt-T (2022) | Auswirkung auf ImageNet und Erklärung |

<!--en:fa318c3e-->
| **Trainingsrezept** | 90 Epochen, SGD mit stufenweiser Lernratensenkung, Bildausschnitte und Spiegelungen | 300 Epochen, AdamW, Mixup/CutMix, Label-Smoothing, EMA | **$+2.7\%$** ($76.1\% \to 78.8\%$), fast die Hälfte des Abstands zum ViT |

<!--en:c3a556f1-->
| **Blockverteilung und Eingangsteil** | $(3, 4, 6, 3)$ Blöcke pro Stufe, $7\times 7_{s2}$-Faltung und Max-Pooling | $(3, 3, 9, 3)$ Blöcke pro Stufe, $4\times 4_{s4}$-Faltung zur Patch-Bildung | **$+0.7\%$** bei $224^2$ auf ImageNet (aber mit $4\times 4_{s4}$ sinkt die Genauigkeit um $\approx 2\%$ bei $64^2$-Miniaturbildern) |

<!--en:7cb7faf6-->
| **Räumliche und kanalübergreifende Mischung** | Dichte $3\times 3$-Faltung im um den Faktor $4\times$ komprimierten Flaschenhals | Tiefenweise $7\times 7$-Faltung am Anfang und invertierter Flaschenhals mit Erweiterungsfaktor $4\times$ | **$+1.1\%$**. Die $49c$ MACs der tiefenweisen Faltung machen $\approx 6\%$ der insgesamt $49c + 8c^2$ MACs des Blocks bei $c=96$ aus. |

<!--en:d29f0747-->
| **Aktivierungen und Normalisierungen** | 3 ReLUs und 3 Batchnormalisierungen pro Block | **1 GELU** und **1 Schichtnormalisierung pro Pixel** je Block | **$+0.9\%$**. Allein das Entfernen von 2 Nichtlinearitäten pro Block bringt $+0.7\%$. |

<!--en:9cdffe32-->
| **Downsampling und Skalierung** | Faltung mit Stride 2 in Block 1 der Stufe | Separate Schichten: $\textrm{LN} + 2\times 2_{s2}$-Faltung, Schichtskalierung $\boldsymbol{\gamma}=10^{-6}$ | **$+0.5\%$** (ohne Attention **$82.0\%$** gegenüber $81.3\%$ für Swin-T) |
