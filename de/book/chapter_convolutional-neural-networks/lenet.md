<!--en:6e254baa-->
# LeNet

<!--en:629af925-->
Wir kennen nun alle Bausteine eines konvolutionären neuronalen Netzes (engl. convolutional neural network, CNN): Faltungsschichten (engl. convolutional layers, :numref:`sec_conv_layer`), Padding und Stride (:numref:`sec_padding`), mehrere Kanäle (:numref:`sec_channels`) und Pooling (:numref:`sec_pooling`). Setzen wir sie zu einem vollständigen Bildklassifikator zusammen und prüfen wir unsere bisherige Argumentation. Erzielen wir tatsächlich bessere Ergebnisse als mit einem herkömmlichen MLP, wenn wir die zweidimensionale räumliche Struktur berücksichtigen?

<!--en:59500680-->
Als Vergleich dient das MLP aus :numref:`sec_mlp-implementation`. Auf Fashion-MNIST erreicht es nach 30 Epochen eine Validierungsgenauigkeit (engl. validation accuracy) von rund 87%. Das MLP formt jedes Bild zu einem Vektor aus 784 Zahlen ohne räumliche Ordnung um (:numref:`sec_why-conv`). Ein CNN hingegen erhält das Pixelgitter und verwendet an jeder Position dieselben lokalen Faltungskerne. Für einen fairen Vergleich trainieren wir unser CNN ebenfalls mit dem Kreuzentropieverlust (engl. cross-entropy loss) und dem SGD-Optimierer über 30 Epochen. Die Genauigkeit allein erklärt diesen Vergleich noch nicht vollständig. Deshalb untersuchen wir auch die Filter, die die erste Faltungsschicht gelernt hat. Mit einem kontrollierten Experiment an Bildern mit permutierten Pixeln prüfen wir, woher der Vorteil des CNN stammt.

<!--en:87a8f03e-->
Wir implementieren *LeNet*, benannt nach Yann LeCun, der das Netz mit Kollegen an den AT&T Bell Labs zur Erkennung handgeschriebener Ziffern entwickelte :cite:`LeCun.Bottou.Bengio.ea.1998`. Es krönte ein Jahrzehnt der Forschung. Am Anfang stand eines der ersten konvolutionären Netze, das mittels Backpropagation durchgängig trainiert wurde :cite:`LeCun.Boser.Denker.ea.1989`. 1998 erreichte LeNet-5 bei handgeschriebenen Ziffern Fehlerraten unter 1% und konnte mit den Support-Vector-Machines mithalten, die damals das maschinelle Lernen dominierten. Varianten von LeNet kamen kommerziell zum Einsatz, um handgeschriebene Beträge auf Bankschecks und bei Einzahlungen am Geldautomaten zu lesen. LeNet zeigte schon früh überzeugend, dass ein durchgängig trainiertes Netz sorgfältig von Hand entwickelte Verfahren zur Bildverarbeitung ersetzen kann.

<!--en:83a13027-->
## Die Architektur von LeNet

<!--en:3c5b9eb4-->
Wie :numref:`img_lenet` zeigt, besteht LeNet aus zwei klar unterscheidbaren Stufen. Auf einen *Faltungsencoder* (engl. convolutional encoder) aus zwei Faltungsblöcken folgt ein *vollständig verbundener Klassifikationskopf* (engl. dense classification head) aus drei vollständig verbundenen Schichten.

<!--en:b734da0c-->
![Datenfluss in LeNet für ein einzelnes $28\times28$-Bild aus Fashion-MNIST. Die Merkmalskarten sind maßstabsgetreu dargestellt. Jedes schattierte Fenster liefert den Wert der dunklen Einheit, auf die es in der nächsten Schicht zeigt. Die drei vollständig verbundenen Schichten haben 120, 84 und 10 Ausgaben.](../img/lenet.svg)

<!--en:9c8ccaf1-->
Jeder Faltungsblock kombiniert eine $5\times5$-Faltungsschicht und eine Sigmoidfunktion mit einer $2\times2$-Average-Pooling-Schicht mit Stride 2. Die erste Faltungsschicht liefert 6 Kanäle, die zweite 16.

<!--en:e1eac88f-->
Die Encoder-Ausgabe hat die Form `(batch_size, channels, height, width)`, die Kanalachse steht also vor den räumlichen Achsen.

<!--en:faa798a0-->
Die Encoder-Ausgabe hat die Form `(batch_size, height, width, channels)`, die Kanalachse steht also hinter den räumlichen Achsen.

<!--en:2385e032-->
Um den Faltungsencoder mit dem vollständig verbundenen Block zu verbinden, formen wir die räumlichen Merkmalskarten (engl. feature maps) jedes Beispiels zu einem einzigen Vektor um. So entsteht eine zweidimensionale Matrix mit einer Zeile pro Beispiel im Minibatch. Drei vollständig verbundene Schichten projizieren den Vektor über 120 und 84 versteckte Einheiten auf 10 Ausgabe-Logits, eines pro Klasse, genau wie bei der Softmax-Regression.

<!--en:918076a4-->
Wir verwenden hier die für die Lehre übliche Version von LeNet-5, die das historische Netz in vereinfachter Form nachbildet. Wie im Netz von 1998 wechseln sich Faltung und Pooling ab, und auch die Schichtbreiten (6, 16, 120 und 84) stimmen überein. Vereinfacht sind heute kaum noch gebräuchliche Details wie die skalierte $\tanh$-Aktivierung, lernbare Skalierungsfaktoren und Bias-Terme beim Pooling, die nur teilweise verbundenen Kanäle zwischen den beiden Faltungsschichten und Ausgabeeinheiten mit radialen Basisfunktionen. Ein historisches Detail ist im Code noch zu erkennen. Das ursprüngliche LeNet-5 erwartete $32\times32$-Eingaben. MNIST-Ziffern sind $28\times28$ Pixel groß. In der Arbeit von 1998 erhielten sie auf jeder Seite einen Rand aus zwei Hintergrundpixeln. So lagen auch Striche nahe dem Ziffernrand in den rezeptiven Feldern der obersten Merkmalsdetektoren. Zwei Pixel Padding in unserer ersten Faltungsschicht erfüllen denselben Zweck. Die erste $5\times5$-Faltung liefert damit wieder eine $28\times28$-Ausgabe.

<!--en:0d5281e2-->
Bevor wir das Netz definieren, schreiben wir `init_cnn` für die Xavier-Initialisierung aller Gewichte der Faltungsschichten und linearen Schichten mit einer Gleichverteilung (:numref:`subsec_xavier`).

<!--en:a66f8205-->
Mit modernen Deep-Learning-Frameworks genügt für `LeNet` ein einziger `Sequential`-Block namens `self.net`. Er verbindet die beiden Stufen aus Faltung und Pooling mit den drei vollständig verbundenen Schichten. Alle Bias-Terme setzen wir auf null.

<!--en:f459d050-->
## Tensorformen in den einzelnen Schichten

<!--en:b0274796-->
Schon ein falscher Padding-Wert oder eine fehlende Schicht zum Umformen in einen Vektor verändert die Tensorformen aller folgenden Schichten. Deshalb empfiehlt es sich, die Zwischenformen vor dem Training zu prüfen. :numref:`img_lenet_vert` zeigt LeNet in kompakter Schreibweise mit der erwarteten Tensorform an jedem Übergang. Zum Abgleich geben wir einen einkanaligen $28 \times 28$-Testtensor in das Modell und geben nach jeder Schicht die Form der Ausgabe aus. Wir implementieren diese Prüfung als Methode `layer_summary` von `d2l.Classifier`. So können wir sie im nächsten Kapitel auch für tiefere Architekturen verwenden.

<!--en:1867a293-->
![Kompakte Darstellung von LeNet. Neben jedem Pfeil steht die Form des weitergereichten Tensors als Kanäle@Höhe×Breite.](../img/lenet-vert.svg)

<!--en:2122c23a-->
Die ausgegebenen Formen stimmen genau mit :numref:`img_lenet_vert` überein und folgen unseren Regeln für die Ausgabeform aus :numref:`sec_padding`. Ohne Padding verringert ein $5\times5$-Faltungskern Höhe und Breite jeweils um 4. Unsere erste Faltungsschicht erhält mit 2 Pixeln Padding auf jeder Seite die Auflösung von $28\times28$. Die zweite Schicht kommt ohne Padding aus und verkleinert $14\times14$ auf $10\times10$. Jede $2\times2$-Average-Pooling-Schicht halbiert beide räumlichen Dimensionen ($28 \to 14$ und $10 \to 5$). Gleichzeitig steigt die Kanalzahl von 1 über 6 auf 16. Mit sinkender räumlicher Auflösung kann der Encoder also mehr Merkmale darstellen. Den letzten Tensor der Form $16 \times 5 \times 5$ formen wir in einen 400-dimensionalen Vektor um. Der vollständig verbundene Block verdichtet ihn schrittweise zu 10 Klassen-Logits.

<!--en:f6e676b8-->
## Parameter und Multiplikationsadditionen

<!--en:713fe72e-->
Nachdem wir die Ausgabeform jeder Schicht geprüft haben, können wir den Speicherbedarf der Parameter und den Rechenaufwand bestimmen. Wie wir aus :numref:`subsec_multi-output-channels` wissen, speichert eine Faltungsschicht mit $c_\textrm{i}$ Eingabekanälen, $c_\textrm{o}$ Ausgabekanälen und einem $k\times k$-Faltungskern insgesamt $c_\textrm{i} c_\textrm{o} k^2$ Gewichte und $c_\textrm{o}$ Bias-Terme. Jedes Gewicht kommt an jeder räumlichen Position der Ausgabe einmal zum Einsatz. Eine vollständig verbundene Schicht verwendet jedes Gewicht dagegen nur einmal pro Beispiel. :numref:`tab_lenet_cost` schlüsselt diese Zahlen für LeNet nach Schichten auf.

<!--en:2bc4477b-->
:Parameter und Multiplikationsadditionen pro Beispiel für die einzelnen Schichten von LeNet. Pooling- und Aktivierungsschichten haben keine Parameter.

<!--en:ca5784dc-->
| Schicht | Ausgabe | Parameter | Multiplikationsadditionen |

<!--en:bc4602f5-->
| $5\times5$-Faltung, von 1 auf 6 Kanäle | 6@28×28 | $150 + 6 = 156$ | $150 \times 784 = 117{,}600$ |

<!--en:6c059c52-->
| $5\times5$-Faltung, von 6 auf 16 Kanäle | 16@10×10 | $2{,}400 + 16 = 2{,}416$ | $2{,}400 \times 100 = 240{,}000$ |

<!--en:8d330626-->
| vollständig verbunden, von 400 auf 120 | 120 | $48{,}000 + 120 = 48{,}120$ | $48{,}000$ |

<!--en:bcec0073-->
| vollständig verbunden, von 120 auf 84 | 84 | $10{,}080 + 84 = 10{,}164$ | $10{,}080$ |

<!--en:26c8e770-->
| vollständig verbunden, von 84 auf 10 | 10 | $840 + 10 = 850$ | $840$ |

<!--en:4dfec91a-->
| Gesamt | | $61{,}706$ | $416{,}520$ |

<!--en:12fbcf12-->
Die Aufschlüsselung zeigt, wie unterschiedlich sich Speicherbedarf und Rechenaufwand auf die Schichten verteilen. Dieses Muster findet sich in vielen CNN-Architekturen. Auf die beiden Faltungsschichten entfallen nur 4% der Parameter von LeNet, aber 86% der Multiplikationsadditionen, weil jedes Gewicht eines Faltungskerns auf dem gesamten räumlichen Gitter zum Einsatz kommt. Umgekehrt enthält allein die erste vollständig verbundene Schicht 78% der Parameter und trägt nur 12% zum Rechenaufwand bei. Unser früheres MLP für Fashion-MNIST hat 203 530 Parameter und benötigt rund 203 000 Multiplikationsadditionen pro Bild. LeNet kommt mit weniger als einem Drittel der Parameter aus, benötigt aber etwa doppelt so viele Rechenoperationen.

<!--en:4723154f-->
## Training

<!--en:775533ff-->
Wie beim MLP minimieren wir den Kreuzentropieverlust mit Minibatch-SGD über 30 Epochen. Die Minibatchgröße beträgt 128, halb so viel wie beim MLP mit 256 Beispielen. Einen Hyperparameter müssen wir jedoch deutlich ändern: die Lernrate. Da die Ableitung der Sigmoidfunktion höchstens $1/4$ beträgt (:numref:`subsec_activation-functions`), schwächen vier aufeinanderfolgende Sigmoidfunktionen die Gradienten auf dem Rückweg zur ersten Faltungsschicht ab. Mit der Lernrate des MLP von 0.1 kommt das Training nur langsam voran. Die Validierungsgenauigkeit bleibt in den ersten zwei bis vier Epochen nahe dem Zufallsniveau und erreicht nach 10 Epochen erst etwa 73%. Deshalb erhöhen wir die Lernrate auf 2.0. Der empirisch ermittelte Faktor zwanzig trägt den kleineren Gradienten Rechnung.

<!--en:ffc3de5f-->
Wie in :numref:`subsec_device_aware_trainer` verlagert `d2l.Trainer` das Training automatisch auf eine GPU, sofern eine verfügbar ist.

<!--en:b2055964-->
Nach 30 Epochen erreicht LeNet eine Validierungsgenauigkeit von rund 88–90% und übertrifft damit die rund 87% des MLP, obwohl es mit weniger als einem Drittel der Parameter auskommt.

<!--en:225296ea-->
Betrachten Sie auch den Anfang der Trainingskurve genauer. Allein von der Zufallsinitialisierung hängt ab, ob die Genauigkeit sofort steigt oder zunächst einige Epochen auf Zufallsniveau (10%) verharrt. Im zweiten Fall bleibt der Verlust nahe $\ln 10 \approx 2.3$, bevor die Genauigkeit plötzlich steigt. Auf diesem anfänglichen Plateau sagt das Netz für jedes Bild nahezu dieselbe Gleichverteilung über die Klassen vorher. Nach dem Durchlauf durch vier Sigmoidschichten sind die eingabeabhängigen Unterschiede zwischen den Logits anfangs zu gering, um die Klassen zu trennen. In einer der folgenden Übungen bestimmen wir, wie stark dieses Signal abklingt. Dasselbe Plateau begegnet uns im Experiment mit permutierten Pixeln wieder.

<!--en:3b95e448-->
### Die gelernten Merkmale

<!--en:8652eca9-->
Bei der Softmax-Regression konnten wir die Gewichte jeder Klasse direkt als $28\times28$-Bildvorlage darstellen (:numref:`sec_softmax_scratch`). Auch im MLP gewichtete jede versteckte Einheit feste Pixelpositionen (:numref:`sec_mlp-implementation`). Ein Faltungskern ist dagegen an kein bestimmtes Pixel gebunden. Er besteht aus einem $5\times5$-Feld von Gewichten, das für sich genommen schwer zu deuten ist. Was hat die erste Faltungsschicht gelernt? Dazu entnehmen wir einem Validierungsminibatch das Bild eines Turnschuhs (Klasse 7) und das eines Hemds (Klasse 6) und fassen beide im Tensor `X` zusammen.

<!--en:eebfd86c-->
Nun geben wir `X` in die erste Faltungsschicht und ihre Sigmoidfunktion. So erhalten wir die sechs $28 \times 28$-Merkmalskarten in `maps` sowie die gelernten $5 \times 5$-Faltungskerne in `kernels`.

<!--en:2a6b26c1-->
Wir können jetzt die sechs Faltungskerne aus `kernels` in der obersten Zeile anzeigen. Darunter stehen die Eingabebilder aus `images` und ihre jeweils sechs Merkmalskarten aus `maps` nach der Sigmoidfunktion. Jedes Teilbild skalieren wir unabhängig von seinem Minimum (schwarz) bis zu seinem Maximum (weiß).

<!--en:84a317cb-->
Über mehrere Trainingsläufe hinweg bilden sich stets zwei Arten von Merkmalskarten heraus. Einige Kanäle reagieren auf die lokale Helligkeit oder erkennen den Vordergrund. Sie geben die ausgefüllte Silhouette des Kleidungsstücks hell auf dunklem Grund oder mit umgekehrter Helligkeit wieder. Die übrigen Kanäle leuchten gezielt entlang der Konturen auf, besonders am linken oder rechten Rand des Kleidungsstücks. Ein Blick auf ihre $5\times5$-Faltungskerne verrät, wie das gelingt. Helle und dunkle Spalten liegen nebeneinander. So entstehen gelernte, mehrzeilige Varianten des von Hand entworfenen Kerns $(1, -1)$ für vertikale Kanten aus :numref:`subsec_edge_detection`. Damals mussten wir eine Kantenkarte als Ziel vorgeben, damit der Gradientenabstieg einen Kantenfilter lernen konnte. Hier erhält LeNet nur die rohen Pixelwerte der Kleidungsstücke und ihre Klassenlabel. Es entdeckt selbst, dass Kanten und Silhouetten nützliche Bausteine für die Bildklassifikation sind. (Welche Rolle die einzelnen Kanäle übernehmen, unterscheidet sich von Lauf zu Lauf.)

<!--en:4a8cca01-->
### Permutierte Pixel

<!--en:b6768043-->
Wie aussagekräftig ist ein Vorsprung von 1 bis 3 Prozentpunkten gegenüber dem MLP? Für sich genommen könnte dieser Abstand auf Zufallsschwankungen beruhen. Die Validierungsgenauigkeit des MLP schwankt von Epoche zu Epoche um rund einen Prozentpunkt (:numref:`sec_mlp-implementation`). Außerdem unterscheidet sich LeNet auch in der Tiefe, der Aktivierungsfunktion und der Lernrate vom MLP. In einem aussagekräftigeren kontrollierten Experiment entfernen wir genau die Information, die nur die Faltungen ausnutzen: die räumliche Nachbarschaft. Greifen wir das Gedankenexperiment aus der Kapiteleinleitung wieder auf. Wir ziehen einmal eine zufällige Permutation der 784 Pixelpositionen und ordnen die Pixel jedes Trainings- und Validierungsbilds nach genau diesem Muster um.

<!--en:bfb39243-->
Nun wenden wir `shuffle_pixels` mit derselben Permutation `perm` auf jedes Bild in den Trainings- und Validierungsdaten von `data_shuffled` an.

<!--en:fa4a499d-->
Betrachten wir einen Validierungsminibatch aus `data_shuffled`, sehen wir, dass durch die Permutation der 784 Pixelkoordinaten alle erkennbaren Konturen der Kleidungsstücke verschwinden. Für das menschliche Auge sehen die verbleibenden Muster wie reines Bildrauschen aus.

<!--en:d50b5121-->
Für das MLP spielt diese Permutation keine Rolle. Warum das so ist, lässt sich sogar ohne erneutes Training zeigen. Eine Permutation der Eingabekoordinaten entspricht einer Umordnung der Zeilen der ersten Gewichtsmatrix $\mathbf{W}^{(1)}$. Da wir alle Gewichte unabhängig voneinander aus derselben Verteilung initialisieren, ist das Training mit permutierten Pixeln statistisch identisch mit dem Training auf den ursprünglichen Bildern. Es ergibt dieselbe Genauigkeit von rund 87%. Für LeNet geht durch die Permutation jedoch die räumliche Lokalität verloren. Jeder $5\times5$-Faltungskern erfasst nun 25 Pixel, die im ursprünglichen Bild zufällig über das Kleidungsstück verstreut liegen. Trainieren wir LeNet mit genau denselben Einstellungen auf dem Datensatz mit permutierten Pixeln erneut und stellen wir die beiden Validierungskurven einander gegenüber.

<!--en:8e6425cd-->
Bei permutierten Pixeln sinkt die Genauigkeit von LeNet in einem typischen Lauf um mehrere Prozentpunkte auf rund 85–86%. Damit liegt sie auf oder knapp unter dem Niveau des MLP. Sobald die räumliche Nachbarschaft verloren geht, verschwindet der Vorteil von LeNet. Das bestätigt, dass sein Vorsprung gegenüber dem MLP unmittelbar auf dem induktiven Bias der Faltung beruht.

<!--en:313a1350-->
Auch hier kann die Genauigkeit zu Beginn auf Zufallsniveau verharren. Wann das geschieht, ist ebenso wenig vorhersehbar wie zuvor. Bei Bildern mit geordneten Pixeln dauert dieses Plateau selten länger als einige Epochen. Bei permutierten Pixeln kann das Modell durch eine ungünstige Initialisierung mitunter einen Großteil der 30 Epochen nahe dem Zufallsniveau bleiben. Das senkt die am Ende erreichte Genauigkeit noch weiter. Die Instabilität geht auf die vier aufeinanderfolgenden Sigmoidfunktionen zurück und ist unabhängig von der Pixelreihenfolge. Selbst wenn LeNet das Plateau sofort verlässt, übertrifft es bei permutierten Pixeln das MLP nie.

<!--en:eab348fa-->
Kehren wir mit LeNet zu den Symmetrieprinzipien aus :numref:`sec_why-conv` zurück. Ein idealisierter Bildklassifikator hätte streng translationsäquivariante Schichten für die Merkmalsextraktion und eine streng translationsinvariante Ausgabe. LeNet erfüllt beide Eigenschaften nur näherungsweise. Faltungen mit Padding verletzen die Äquivarianz am Rand. Pooling gleicht nur Verschiebungen innerhalb eines Pooling-Blocks aus (:numref:`sec_pooling`). Formen wir die letzten $5\times5$-Merkmalskarten in einen Vektor um, erhält jede räumliche Zelle eigene Gewichte in der vollständig verbundenen Schicht. LeNet toleriert Verschiebungen daher nur bedingt. Für Fashion-MNIST mit seinen sorgfältig zentrierten Kleidungsstücken reicht das aus, für Szenen ohne solche Einschränkungen jedoch nicht.

<!--en:783e6248-->
Zusammen zeigen unsere beiden Experimente, dass ein CNN mit weniger als einem Drittel der Parameter eines MLP dessen Genauigkeit erreichen und sogar übertreffen kann. Dank der Gewichtsteilung (engl. weight sharing) muss jeder Faltungskern ein lokales Bildmuster nur einmal lernen und erkennt es dann überall auf dem Gitter. Bei winzigen $28\times28$-Bildern ist diese Effizienz ein kleiner Vorteil. Bei den Megapixelfotos aus :numref:`sec_why-conv` braucht schon eine einzige vollständig verbundene Schicht eine Milliarde Gewichte. Dort entscheidet dasselbe Prinzip darüber, ob sich ein Modell überhaupt bauen lässt. Die nächste Herausforderung besteht darin, tiefe konvolutionäre Netze auch in großem Maßstab *leicht trainierbar* zu machen. Die langsame und unregelmäßige Konvergenz von LeNet mit seinen vier Sigmoidschichten lässt bereits erahnen, warum moderne Architekturen weit über den Stand von 1998 hinausgehen mussten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:48f23353-->
LeNet verbindet einen Faltungsencoder mit einem dreischichtigen, vollständig verbundenen Klassifikationskopf. Im Encoder nimmt die räumliche Auflösung von Stufe zu Stufe ab, während mehr Kanäle für die Darstellung der Merkmale hinzukommen. Die Faltungsschichten enthalten nur 4% der Parameter, benötigen aber 86% der Multiplikationsadditionen. Allein auf die erste vollständig verbundene Schicht entfallen dagegen 78% der Parameter. Wie das MLP aus :numref:`sec_mlp-implementation` trainieren wir LeNet mit SGD über 30 Epochen. Auf Fashion-MNIST erreicht LeNet mit weniger als einem Drittel der Parameter eine Validierungsgenauigkeit von rund 88–90% (MLP: rund 87%). Allein anhand der Klassenlabel lernt es selbstständig Filter für Silhouetten und vertikale Kanten. Permutieren wir die Pixelpositionen, verschwindet der Vorsprung. Damit lässt sich der Vorteil auf die räumliche Lokalität zurückführen. Zugleich erfordert die Rückwärtsausbreitung der Gradienten durch vier Sigmoidschichten eine wesentlich höhere Lernrate (2.0 statt 0.1) und verursacht unvorhersehbare Plateaus zu Trainingsbeginn.

<!--en:c044f774-->
Der Aufbau aus Encoder und Klassifikationskopf ist nach wie vor grundlegend, doch fast jeder einzelne Baustein von LeNet hat inzwischen einen modernen Nachfolger. ReLUs ersetzen die Sigmoidfunktionen, Max-Pooling und Faltungen mit Stride das Average-Pooling. An die Stelle der parameterreichen vollständig verbundenen Köpfe tritt Global-Average-Pooling mit einer einzigen nachgeschalteten linearen Schicht. Normalisierungsschichten stabilisieren das Training tiefer Netze. Im nächsten Kapitel verfolgen wir diese Fortschritte Schritt für Schritt, beginnend mit AlexNet (:numref:`sec_alexnet`). Dabei nehmen Tiefe, Breite und Auflösung der Netze zu.

<!--en:2294243a-->
## Übungen

<!--en:13b58d32-->
1. [code] **Moderne Bausteine.** Trainieren Sie `LeNet` mit den folgenden Änderungen und den Trainingseinstellungen dieses Abschnitts erneut. Vergleichen Sie die Validierungsgenauigkeit jeder Variante nach 30 Epochen mit der von `LeNet`.
    1. Ersetzen Sie Average-Pooling durch Max-Pooling.
    1. Ersetzen Sie die Sigmoidfunktionen durch ReLU. Falls der Verlust divergiert oder die Genauigkeit auf Zufallsniveau bleibt, senken Sie die Lernrate, bis das Training gelingt. Geben Sie die verwendete Lernrate an.
    1. Kombinieren Sie beide Änderungen und bezeichnen Sie das Ergebnis als modernisiertes LeNet. Welche Änderung erklärt den größten Teil des Unterschieds in der Genauigkeit?
    1. ● Verwenden Sie die Gewichte zu Trainingsbeginn und messen Sie für `LeNet` und das modernisierte LeNet zwei Größen: die Norm des Gradienten des Verlusts bezüglich des ersten Faltungskerns und die Standardabweichung der Logits über einen Minibatch. Die Herleitung der Xavier-Initialisierung in :numref:`subsec_xavier` setzt Eingaben mit Mittelwert null und das Fehlen einer Nichtlinearität voraus. Erklären Sie anhand des Mittelwerts der Ausgabe der Sigmoidfunktion und ihrer Steigung $\sigma'(0) = 1/4$ Ihre Messwerte, die von `LeNet` benötigte Lernrate und die Epochen auf Zufallsniveau zu Trainingsbeginn.
   :exlabel:`ex_lenet-modern-components`

<!--en:01889a98-->
2. [code] **Der ursprüngliche MNIST-Datensatz.** Schreiben Sie eine Datenklasse `MNIST`, indem Sie `d2l.FashionMNIST` anpassen. Trainieren Sie darauf sowohl `LeNet` als auch das MLP aus :numref:`sec_mlp-implementation` mit den Einstellungen, die das jeweilige Modell für Fashion-MNIST verwendet. :citet:`LeCun.Bottou.Bengio.ea.1998` berichten für LeNet-5 auf MNIST eine Testfehlerrate unter 1%.
    1. Vergleichen Sie die Validierungsfehlerrate von `LeNet` mit diesem Wert. Beide Fehlerraten beziehen sich auf dieselben Bilder. Wie bei Fashion-MNIST verwenden wir den offiziellen Testdatensatz als Validierungsdatensatz.
    1. Berechnen Sie für jeden Datensatz die Differenz und das Verhältnis der Validierungsfehlerraten beider Netze. Laut :numref:`sec_fashion_mnist` macht MNIST Unterschiede zwischen Modellklassen weniger deutlich sichtbar. Stützen beide Maße diese Aussage?

<!--en:acca566c-->
3. [code] **Merkmalskarten tieferer Schichten.** Im Text haben wir die Merkmalskarten der ersten Faltungsschicht dargestellt.
    1. Zeigen Sie die 16 Merkmalskarten der zweiten Faltungsschicht des trainierten `LeNet` nach der Sigmoidfunktion. Verwenden Sie Validierungsbilder zweier leicht zu verwechselnder Klassen, zum Beispiel Pullover und Mäntel. Können Sie noch erkennen, worauf ein Kanal reagiert? Welche Kanäle unterscheiden sich zwischen den beiden Klassen am stärksten?
    1. Trainieren Sie das modernisierte LeNet aus Aufgabe :exref:`ex_lenet-modern-components`. Messen Sie für jede der beiden Faltungsschichten die maximale Aktivierung nach der ReLU. Verwenden Sie dafür Validierungsbilder aus Fashion-MNIST, Fotos außerhalb der Trainingsverteilung, etwa von einer Katze oder einem Auto, sowie gleichverteiltes Zufallsrauschen. Wandeln Sie jedes Foto in ein $28 \times 28$-Graustufenbild um. Halten Sie alle Eingaben im Intervall $[0, 1]$, dem Pixelwertebereich von `d2l.FashionMNIST`. Liegen die Maxima für Fotos und Rauschen innerhalb oder deutlich außerhalb des Bereichs, den Sie bei den Validierungsbildern beobachten?

<!--en:6490782e-->
4. [code] **Verschobene Kleidungsstücke.** Im Text haben wir begründet, warum `LeNet` Verschiebungen nur teilweise toleriert.
    1. Verschieben Sie jedes Validierungsbild um ein, zwei und vier Pixel nach rechts und füllen Sie die frei gewordenen Spalten mit Nullen. Messen Sie für jeden der verschobenen Datensätze die Validierungsgenauigkeit des trainierten `LeNet`. Welche Verschiebungen toleriert das Netz? Erklären Sie Ihre Beobachtungen anhand der beiden Pooling-Schichten und des vollständig verbundenen Kopfs.
    1. Erklären Sie, warum das Training von `LeNet` mit verschobenen Kopien der Trainingsbilder die Genauigkeit auf verschobenen Validierungsbildern verbessern kann, obwohl seine Faltungsschichten an allen Positionen dieselben Gewichte verwenden.

    *Nach Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Kapitel 6.*

<!--en:f636dc67-->
5. [code] **Der vollständig verbundene Kopf.** ● :numref:`tab_lenet_cost` zeigt, dass die erste vollständig verbundene Schicht 78% der Parameter von LeNet enthält.
    1. Ersetzen Sie den gesamten vollständig verbundenen Kopf vom Umformen zum Vektor bis zur Ausgabeschicht durch Global-Average-Pooling mit einer anschließenden linearen Schicht von 16 auf 10 Einheiten. Wie viele Parameter hat das Netz jetzt? Schätzen Sie vor dem Training, ob sich die Validierungsgenauigkeit gegenüber `LeNet` um mehr als einen Prozentpunkt ändert. Trainieren Sie das Netz dann erneut und vergleichen Sie das Ergebnis mit Ihrer Vorhersage.
    1. Verbinden Sie stattdessen die zum 400-dimensionalen Vektor umgeformte Ausgabe der Faltungsschichten direkt mit einer linearen Ausgabeschicht für zehn Klassen. Wie viele Parameter hat dieses Netz? Trainieren Sie es erneut. Vergleichen Sie die Genauigkeit der beiden verkleinerten Köpfe miteinander und mit Ihrer Vorhersage. Was sagt dieser Vergleich darüber aus, wo LeNet seine Modellkapazität nutzt?

    *Nach Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Kapitel 6.*

<!--en:882b13af-->
[Diskussion](https://d2l.discourse.group/t/73)

<!--en:c3a0473c-->
[Diskussion](https://d2l.discourse.group/t/74)

<!--en:5d436cd1-->
[Diskussion](https://d2l.discourse.group/t/275)

<!--en:65c2bcd0-->
[Diskussion](https://d2l.discourse.group/t/18000)

<!--en:d77a1722-->
Konvolutionäre Netze (LeNet)<br>**Faltungsencoder · vollständig verbundener Kopf · Verteilung von Parametern und FLOPs · gelernte Filter · Test mit permutierten Pixeln**

<!--en:492c0a27-->
::: {.slide title="Ein CNN auf dem Pixelgitter im Vergleich zu unserem MLP mit Vektoreingabe"}

<!--en:81cd2726-->
[Motivation und Vergleich auf Fashion-MNIST]{.kicker}

<!--en:c4ed34fa-->
- **MLP aus §{num:chap_perceptrons} ($784 \to 256 \to 10$):** Formt $28 \times 28$-Bilder in Vektoren um und erreicht mit **$203{,}530$ Parametern** eine Validierungsgenauigkeit von $\approx 87\%$.

<!--en:b1c34e5d-->
- **CNN LeNet-5:** Erhält die zweidimensionale Lokalität und verwendet überall dieselben $5 \times 5$-Faltungskerne. Erreicht **$\approx 88\text{--}90\%$** mit **um den Faktor $3.3\times$ weniger Parametern** ($61{,}706$).

<!--en:8759dcc1-->
- **Kontrollierter Test:** Alle 784 Pixel permutieren, um den räumlichen induktiven Bias zu isolieren

<!--en:e50f452a-->
::: {.slide title="Die LeNet-Architektur mit Encoder und vollständig verbundenem Kopf"}

<!--en:c5f70411-->
[Zwei Faltungsblöcke, dann drei vollständig verbundene Schichten]{.kicker}

<!--en:22a096cd-->
LeNet besteht aus einem **Faltungsencoder** (räumliche Auflösung $28 \to 14 \to 10 \to 5$, dafür wachsende Kanalzahl $1 \to 6 \to 16$) und einem **vollständig verbundenen Klassifikationskopf** ($400 \to 120 \to 84 \to 10$):

<!--en:bb36daae-->
- **Faltungsblock 1:** $5 \times 5$-Faltung ($1 \to 6$ Kanäle, $p=2, s=1$) $\to$ Sigmoidfunktion $\to$ $2 \times 2$-Average-Pooling ($s=2$).

<!--en:04c80968-->
- **Faltungsblock 2:** $5 \times 5$-Faltung ($6 \to 16$ Kanäle, $p=0, s=1$) $\to$ Sigmoidfunktion $\to$ $2 \times 2$-Average-Pooling ($s=2$).

<!--en:11db2e6e-->
- **Vollständig verbundener Kopf:** Umformen zum Vektor ($16 \times 5 \times 5 = 400$) $\to$ vollständig verbunden ($120$) $\to$ Sigmoidfunktion $\to$ vollständig verbunden ($84$) $\to$ Sigmoidfunktion $\to$ vollständig verbunden ($10$).

<!--en:307e5639-->
::: {.slide title="Das historische LeNet-5 (1998) und unsere Lehrbuchversion"}

<!--en:9b1af922-->
[Warum die erste Schicht für 28×28-Bilder Padding mit p = 2 verwendet]{.kicker}

<!--en:fcb91f01-->
- **Padding $p = 2$ in Conv1:** Erweitert $28 \times 28$-Bilder aus Fashion-MNIST auf $32 \times 32$ Pixel, passend zum $32 \times 32$-Eingabegitter der Arbeit von 1998.

<!--en:7781f7c1-->
- **Vereinfachte Bausteine:** Gewöhnliche Sigmoidfunktion $\sigma(x)$, parameterfreies $2 \times 2$-Average-Pooling, vollständige Kanalverbindung $6 \to 16$ in Conv2 und eine lineare Ausgabe mit Softmax-Funktion und Kreuzentropieverlust anstelle euklidischer RBF-Einheiten.

<!--en:bacd5e5b-->
::: {.slide title="LeNet als sequenzielles Modul implementieren"}

<!--en:e3a9c3b8-->
[Xavier-Initialisierung mit Gleichverteilung und sequenzielle Anordnung der Schichten]{.kicker}

<!--en:e20fcc30-->
Alle Gewichte nach Xavier initialisieren, Encoder und vollständig verbundenen Kopf verbinden. Dazu genügt ein einziger `Sequential`-Block:

<!--en:b6da03f3-->
::: {.slide title="Tensorformen auf dem Weg durch LeNet" layout="tall-output"}

<!--en:4f849f5f-->
[Von Schicht zu Schicht sinkt die räumliche Auflösung und steigt die Kanalzahl]{.kicker}

<!--en:dc839126-->
::: {.slide title="Welche Schichten speichern die Parameter, welche benötigen die meisten FLOPs?"}

<!--en:6b0789e3-->
[Speicherbedarf und Rechenaufwand sind in CNNs unterschiedlich verteilt]{.kicker}

<!--en:6fe2c2f9-->
- **Faltungen ($4.2\%$ der Parameter, $85.9\%$ der MACs):** Conv1 ($156$ Parameter) und Conv2 ($2{,}416$ Parameter) verwenden jedes Gewicht an $784$ bzw. $100$ räumlichen Positionen ($357{,}600$ MACs).

<!--en:b2f786a6-->
- **Vollständig verbundener Kopf ($95.8\%$ der Parameter, $14.1\%$ der MACs):** `Dense1` ($400 \to 120$) enthält **$78\%$ aller Parameter** ($48{,}120$), wird aber nur einmal pro Bild ausgewertet.

<!--en:0cef38df-->
::: {.slide title="Vier Sigmoidfunktionen und das Problem verschwindender Gradienten"}

<!--en:4f83174a-->
[Warum LeNet die Lernrate 2.0 braucht und das MLP mit 0.1 auskommt]{.kicker}

<!--en:e11c7a95-->
- **Obergrenze der Ableitung $\sigma'(x) \le \tfrac{1}{4}$:** Der Gradient von Conv1 enthält vier Faktoren $\sigma'$. Ihr Produkt beträgt höchstens $(1/4)^4 = 1/256$.

<!--en:2080985e-->
- **Anfängliches Plateau bei $\ln 10 \approx 2.30$:** Die Ausgaben der Sigmoidfunktionen liegen nahe bei $0.5$. Dadurch ist die Varianz der Logits zu Trainingsbeginn sehr klein.

<!--en:3ff26384-->
- **Ausgleich:** Das klassische LeNet benötigt $\eta = 2.0$ ($20\times$ so hoch wie die Lernrate des ReLU-MLP von $\eta = 0.1$).

<!--en:0a3764ba-->
::: {.slide title="Training von LeNet auf Fashion-MNIST"}

<!--en:07c0ec53-->
[30 Epochen Minibatch-SGD (Batchgröße 128, Lernrate 2.0)]{.kicker}

<!--en:2c3d92d5-->
- **Trainingsverlauf:** Nach einem sigmoidbedingten Plateau von 1–3 Epochen bei $\mathcal{L} \approx 2.30$ erreicht die Validierungsgenauigkeit **$\approx 88\text{--}90\%$** (um den Faktor $3.3\times$ weniger Parameter als beim MLP).

<!--en:57cf3c2b-->
::: {.slide title="Faltungskerne und Merkmalskarten der ersten Schicht"}

<!--en:b780b779-->
[Conv1 am Beispiel eines Turnschuhs und eines Hemds aus den Validierungsdaten]{.kicker}

<!--en:dbb9208b-->
Eine vollständig verbundene Schicht lernt globale $28 \times 28$-Vorlagen, Conv1 dagegen sechs lokale $5 \times 5$-Faltungskerne. Was erkennt jeder Kanal? Dazu die $6$ gelernten $5 \times 5$-Faltungskerne und ihre $28 \times 28$-Merkmalskarten nach der Sigmoidfunktion für einen Turnschuh und ein Hemd auslesen:

<!--en:056731f1-->
::: {.slide title="Die gelernten Kanten- und Silhouettenfilter von LeNet"}

<!--en:22ed13d5-->
[Richtungsabhängige Kantendetektoren allein aus Klassenlabeln]{.kicker}

<!--en:443367df-->
::: {.slide title="Kontrolliertes Experiment mit permutierten Pixeln"}

<!--en:942bfcb7-->
[Eine feste Zufallspermutation hebt die räumliche Nachbarschaft auf]{.kicker}

<!--en:b2578795-->
- **Invarianz des MLP:** $(\mathbf{X}\mathbf{P})\mathbf{W}^{(1)} = \mathbf{X}(\mathbf{P}\mathbf{W}^{(1)})$ permutiert lediglich die anfangs unabhängig und identisch verteilten Zeilen ($\approx 87\%$).

<!--en:710cd6d1-->
- **Verlust der Lokalität beim CNN:** Jedes $5 \times 5$-Fenster erfasst 25 verstreute Pixel.

<!--en:4b2870ea-->
::: {.slide title="Fashion-MNIST-Bilder mit permutierten Pixeln"}

<!--en:8561407b-->
[Gleicher Informationsgehalt ohne lokale Geometrie]{.kicker}

<!--en:e6fdf370-->
Jedes permutierte Bild enthält dieselben 784 Pixelintensitäten wie zuvor. Die neue Anordnung ist für alle Bilder gleich, für das menschliche Auge sehen sie dennoch wie weißes Rauschen aus:

<!--en:39c8f583-->
::: {.slide title="Geordnete und permutierte Pixel im Vergleich: die räumliche Vorannahme"}

<!--en:a08e4c47-->
[Erneutes Training von LeNet auf Fashion-MNIST mit permutierten Pixeln]{.kicker}

<!--en:8a230a85-->
::: {.slide title="Geordnete und permutierte Pixel im Vergleich: die räumliche Vorannahme (Forts.)"}

<!--en:a08e4c47-2-->
[Erneutes Training von LeNet auf Fashion-MNIST mit permutierten Pixeln]{.kicker}

<!--en:2338f6dd-->
- **Induktiver Bias isoliert:** Ohne zweidimensionale Lokalität sinkt die Genauigkeit von LeNet um $3\text{--}4$ Prozentpunkte (auf $\approx 85\text{--}86\%$). Der Vorsprung gegenüber dem MLP verschwindet.

<!--en:e114d365-->
::: {.slide title="Grenzen der Translationsinvarianz von LeNet"}

<!--en:7de759d8-->
[Randeffekte, lokale Pooling-Blöcke und Vektoreingabe des vollständig verbundenen Kopfs]{.kicker}

<!--en:66464491-->
- **1. Ränder mit Padding:** Bei einer Verschiebung nahe dem Bildrand ändert sich die Anzahl der Nullen im Fenster von Conv1 ($p=2$).

<!--en:98278698-->
- **2. Lokale Pooling-Blöcke:** Beim $2 \times 2$-Average-Pooling ändert sich die Ausgabe sprunghaft, wenn ein Maximum eine Blockgrenze überschreitet.

<!--en:8b5d249d-->
- **3. Vollständig verbundener Kopf mit Vektoreingabe ($400 \to 120$):** Jede der $5 \times 5$ räumlichen Zellen erhält eigene Gewichte.

<!--en:a07c408a-->
::: {.slide title="Dreißig Jahre Entwicklung von LeNet zu modernen CNNs"}

<!--en:68b00a4d-->
[Gleicher Aufbau aus Encoder und Kopf mit moderneren Bausteinen]{.kicker}

<!--en:749d9fe0-->
- **Signalausbreitung:** Moderne CNNs ersetzen Sigmoidfunktion und Xavier-Initialisierung durch **ReLU/GELU, He-Initialisierung, BatchNorm/LayerNorm und Residualverbindungen**. Damit entfällt die Abschwächung der Gradienten um den Faktor $(1/4)^L$.

<!--en:f466586f-->
- **Downsampling und Kopf:** Faltungen mit Stride 2 und **Global-Average-Pooling** ersetzen Average-Pooling und den vollständig verbundenen Kopf mit 3 Schichten.

<!--en:65ec60bc-->
::: {.slide title="Verständnisfragen zu Architektur und Rechenaufwand von LeNet"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:99d3198b-->
- **1. Kopf mit Global-Average-Pooling:** Ersetzt man `Flatten + 3 Dense` ($59{,}134$ Parameter) durch Global-Average-Pooling ($16 \times 5 \times 5 \to 16$) und eine lineare Schicht ($16 \to 10$, $170$ Parameter), sinkt die Gesamtzahl der Parameter von LeNet um den Faktor $22.5\times$ ($61{,}706 \to 2{,}742$).

<!--en:16a674c6-->
- **2. Warum ReLU eine niedrigere Lernrate braucht:** ReLU hat die Steigung eins ($\sigma' = 1$ für $x>0$). Die vier Faktoren $\sigma' \le \tfrac{1}{4}$ entfallen, sodass das Training mit $\eta = 2.0$ divergiert.

<!--en:de9914b4-->
::: {.slide title="Zusammenfassung des Vergleichs von MLP und LeNet"}

<!--en:c686688d-->
[Abschnitt und Kapitel im Überblick]{.kicker}

<!--en:f04be9cc-->
| Kenngröße / Baustein | MLP aus §{num:chap_perceptrons} ($784 \to 256 \to 10$) | Klassisches LeNet (Architektur von 1998) | Moderne CNNs (Ausblick auf §{num:chap_modern_cnn}) |

<!--en:57607b42-->
| **Parameter** | $203{,}530$ ($98.6\%$ in Schicht 1) | **$61{,}706$** ($4.2\%$ in Faltungsschichten, $95.8\%$ in vollständig verbundenen Schichten) | Flaschenhälse und Kopf mit Global-Average-Pooling |

<!--en:9bb9d68b-->
| **Multiplikationsadditionen pro Bild** | $\approx 203{,}000$ | **$416{,}520$** ($85.9\%$ in Faltungsschichten) | Überwiegend in räumlichen Faltungsschichten |

<!--en:ba7a8c1b-->
| **Aktivierung und Lernrate** | ReLU, $\eta = 0.1$ | Sigmoidfunktion ($\sigma' \le 1/4$), $\eta = 2.0$ | ReLU / GELU und BatchNorm / LayerNorm |

<!--en:408688a4-->
| **Downsampling und Kopf** | Entfällt (Eingabe gleich zu Beginn zum Vektor umgeformt) | $2 \times 2$-Average-Pooling und vollständig verbundener Kopf mit 3 Schichten | Faltung mit Stride / Max-Pooling und Global-Average-Pooling |

<!--en:dd2223d0-->
| **Fashion-MNIST (geordnete Pixel)** | $\approx 87\%$ | **$\approx 88\text{--}90\%$** (um den Faktor $3.3\times$ weniger Parameter) | $> 92\%$ mit tieferen Residualstufen |

<!--en:10033a07-->
| **Fashion-MNIST (permutierte Pixel)** | $\approx 87\%$ (permutationsinvariant) | **$\approx 85\text{--}86\%$** (Lokalität aufgehoben) | Starker Rückgang ohne räumliches Gitter |
