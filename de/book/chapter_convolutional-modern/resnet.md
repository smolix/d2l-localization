<!--en:63e3e111-->
# Residualnetze

<!--en:d871a8ba-->
Bislang ging auf ImageNet größere Tiefe stets mit höherer Genauigkeit einher, von den 8 Schichten in AlexNet bis zu den 22 in GoogLeNet. Mit der Batchnormalisierung (:numref:`sec_batch_norm`) schien auch das letzte Hindernis für noch tiefere Netze überwunden. Man würde also erwarten, dass jede weitere Schicht den Trainingsfehler weiter senkt, zumindest bis eine Überanpassung (engl. overfitting) einsetzt. Um diese Vermutung zu prüfen, trainierten :citet:`He.Zhang.Ren.ea.2016` immer tiefere *unverzweigte* Netze nach dem Vorbild von VGG, in denen auf jede Faltung (engl. convolution) Batchnormalisierung und ReLU folgen. Sie beobachteten das genaue Gegenteil. Auf CIFAR-10 hatte ein unverzweigtes Netz mit 56 Schichten am Ende einen deutlich höheren *Trainingsfehler* als eines mit 20 Schichten. Da das tiefere Netz sogar den Trainingsdatensatz schlechter abbildet, liegt die Ursache dieses *Degradationsproblems* (engl. degradation problem) bei der Optimierung. Überanpassung scheidet aus. Das ist erstaunlich: Das Netz mit 56 Schichten könnte die Lösung des Netzes mit 20 Schichten übernehmen und die übrigen 36 Schichten als Identitätsabbildung wählen. Damit wäre sein Trainingsverlust mindestens ebenso niedrig. Warum findet der stochastische Gradientenabstieg (engl. stochastic gradient descent, SGD) eine solche Lösung nicht? Wie können wir den Baustein so umgestalten, dass dies gelingt?

<!--en:9a2a4300-->
*Residualnetze* (engl. residual networks, ResNets) lösen das Paradox durch eine überraschend einfache Umparametrisierung. Ein Residualblock muss die Transformation von der Eingabe zur Ausgabe nicht vollständig neu lernen. Er addiert eine gelernte Korrektur zur Eingabe. Sobald die Korrektur null ist, ergibt sich die Identitätsabbildung. Diese eine Sprungverbindung (engl. shortcut) ermöglichte bislang unerreichte Tiefen. Auf ImageNet blieb ein unverzweigtes Netz mit 34 Schichten hinter einem mit 18 Schichten zurück. Ein ResNet mit 34 Schichten dagegen übertraf sein Gegenstück mit 18 Schichten um fast drei Prozentpunkte. ResNets mit 152 Schichten gewannen 2015 den ImageNet-Wettbewerb.

<!--en:d3cad837-->
Wir untersuchen zunächst anhand geschachtelter Funktionenklassen, was zusätzliche Tiefe bewirkt, und entwerfen den Residualblock. Dann machen wir die Degradation auf Fashion-MNIST sichtbar. Wir trainieren unverzweigte Netze und Residualnetze mit jeweils 20 und 56 Schichten und beobachten im direkten Vergleich, wie die Degradation auftritt und wieder verschwindet. Anschließend bauen und trainieren wir ResNet-18. Zum Schluss betrachten wir zwei einflussreiche Nachfolger: *ResNeXt* verbreitert seine Blöcke bei gleichem Rechenaufwand durch gruppierte Faltungen. *DenseNet* verkettet Merkmalskarten (engl. feature maps), statt sie zu addieren.

<!--en:27aff704-->
## Funktionenklassen und Tiefe

<!--en:2398692e-->
Betrachten wir die *Funktionenklasse* (engl. function class) $\mathcal{F}$ einer Architektur. Sie umfasst alle Abbildungen von Eingaben auf Ausgaben, die sich durch Variation ihrer Parameter darstellen lassen. Wann enthält die Funktionenklasse $\mathcal{F}'$ einer größeren Architektur garantiert eine Lösung, deren Trainingsverlust mindestens so niedrig ist wie der beste in $\mathcal{F}$? Immer dann, wenn die Klassen *geschachtelt* sind, also $\mathcal{F} \subseteq \mathcal{F}'$ gilt (:numref:`fig_functionclasses`). Gilt hingegen $\mathcal{F} \not\subseteq \mathcal{F}'$, kann sich eine größere Klasse von der Zielfunktion $f^*$ entfernen.

<!--en:7724569d-->
![Funktionenklassen zunehmender Größe. Links sind die Klassen nicht geschachtelt. Eine größere Klasse muss weder die Funktionen einer kleineren enthalten noch dem Ziel $f^*$ näher kommen. Rechts enthält jede Klasse alle vorherigen. Mit wachsender Klasse kann die beste verfügbare Funktion das Ziel also nur besser oder gleich gut annähern.](../img/functionclasses.svg)

<!--en:9205c011-->
Fügen wir eine Schicht hinzu, so sind die Funktionenklassen geschachtelt, sofern die zusätzliche Schicht die Identitätsabbildung $f(\mathbf{x}) = \mathbf{x}$ darstellen kann. Ein unverzweigter Block $\mathbf{x} \mapsto \textrm{ReLU}(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}))$ *kann* die Identität auf nichtnegativen Eingaben darstellen. Unverzweigte Netze entsprechen also bereits dem rechten Teil von :numref:`fig_functionclasses` mit seinen geschachtelten Klassen. Ihre Kapazität reicht aus. Die Degradation entsteht bei der Optimierung, weil es SGD schwerfällt, die Identität über Dutzende nichtlinearer Schichten hinweg anzunähern. Ein Residualblock parametrisiert die Schicht so um, dass die Identität im Ursprung $g(\mathbf{x}) = 0$ liegt. Dort sind die Gewichte des Residualzweigs null.

<!--en:413d44c4-->
## Residualblöcke

<!--en:fc8e39f3-->
Anstatt eine Zielabbildung $f(\mathbf{x})$ direkt zu lernen, lernt ein Residualblock (:numref:`fig_residual_block`) das *Residuum* (engl. residual) $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$. Über eine *Sprungverbindung* (auch *Skip-Verbindung*) addiert er die Eingabe wieder dazu und gibt $\mathbf{x} + g(\mathbf{x})$ aus. Damit der Block die Identität berechnet, genügt es, die letzte Schicht von $g$ auf null zu setzen.

<!--en:9c8f5f4b-->
![Ein herkömmlicher Block (links) muss die Abbildung $f(\mathbf{x})$ direkt lernen. Ein Residualblock (rechts) lernt die Residualabbildung $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$ und addiert die Eingabe wieder dazu. Der Identitätsabbildung $f(\mathbf{x}) = \mathbf{x}$ entspricht somit $g(\mathbf{x}) = 0$.](../img/arch-residual-concept.svg)

<!--en:b6ceb837-->
Sprungverbindungen verändern auch den Weg der Gradienten. Betrachten wir zwei aufeinandergestapelte Blöcke $\mathbf{x}_1 = \mathbf{x}_0 + g_1(\mathbf{x}_0)$ und $\mathbf{x}_2 = \mathbf{x}_1 + g_2(\mathbf{x}_1)$. Multiplizieren wir das Produkt in ihrer Jacobi-Matrix $\partial \mathbf{x}_2 / \partial \mathbf{x}_0 = (\mathbf{I} + \partial g_2 / \partial \mathbf{x}_1)(\mathbf{I} + \partial g_1 / \partial \mathbf{x}_0)$ aus, erhalten wir die Einheitsmatrix $\mathbf{I}$ plus Terme aus den Residualzweigen. Gradienten des Verlusts erreichen also frühe Blöcke, ohne dass sie jede dazwischenliegende Gewichtsmatrix durchlaufen müssen. (In kontinuierlicher Zeit betrachtet entspricht $\mathbf{x}_{l+1} = \mathbf{x}_l + g_l(\mathbf{x}_l)$ einer Euler-Diskretisierung mit Schrittweite eins für die gewöhnliche Differentialgleichung $\dot{\mathbf{x}} = g(\mathbf{x}, t)$, siehe :numref:`sec_mdl-neural-odes-cnf`.)

<!--en:9d828a45-->
Es liegt nahe, die Degradation unverzweigter Netze auf verschwindende Gradienten zurückzuführen. Doch :citet:`He.Zhang.Ren.ea.2016` zeigten, dass die Gradientennormen in unverzweigten Netzen mit Batchnormalisierung in jeder Tiefe ausreichend groß bleiben. Die Ursache liegt tiefer. Spätere Studien liefern zwei Erklärungen, die einander ergänzen. In einem tiefen unverzweigten Netz verlieren die Gradienten an benachbarten Eingaben mit zunehmender Tiefe ihre Korrelation, bis sie weißem Rauschen gleichen. SGD fehlt dann eine beständige Richtung. Mit Sprungverbindungen nimmt die Korrelation viel langsamer ab :cite:`Balduzzi.Frean.Leary.ea.2017`. Betrachtet man den Verlust als Funktion der Gewichte, zeigt ein tiefes unverzweigtes Netz eine raue, chaotische Verlustlandschaft. Sprungverbindungen halten sie in der Umgebung des Minimierers glatt :cite:`Li.Xu.Taylor.ea.2018`. Die Jacobi-Matrix oben legt eine Erklärung nahe. Multiplizieren wir das Produkt über $L$ Blöcke mit den Faktoren $\mathbf{I} + \partial g_l / \partial \mathbf{x}_{l-1}$ aus, erhalten wir eine Summe über $2^L$ Pfade. Der größte Teil des Gradienten fließt entlang der kurzen Pfade. Ein ResNet verhält sich beim Training daher wie ein Ensemble recht flacher Netze :cite:`Veit.Wilber.Belongie.2016`. Entfernt man aus einem solchen Ensemble einen einzelnen Block, schadet das kaum. Die stochastische Tiefe aus :numref:`sec_training_recipes` nutzt dies zur Regularisierung.

<!--en:d7c8289b-->
Der grundlegende `Residual`-Block von ResNet enthält zwei $3 \times 3$-Faltungen ohne Bias mit Batchnormalisierung und einer ReLU dazwischen. Zur Ausgabe der zweiten Batchnormalisierung addiert er über die Sprungverbindung die Eingabe $\mathbf{x}$ und wendet abschließend eine ReLU an. Setzen wir in der zweiten Batchnormalisierung $\gamma = \beta = 0$, gibt der Residualzweig null aus. Für nichtnegative Eingaben gilt dann wieder $\textrm{ReLU}(\mathbf{x}) = \mathbf{x}$. Beim Training großer Netze nutzt man dies zur Initialisierung. Der letzte Parameter $\gamma$ jedes Residualzweigs startet bei 0. Im ersten Schritt berechnet somit jeder Block die Identität. Das tiefe Netz beginnt das Training wie ein flaches Netz, dessen Residualzweige nach und nach wirksam werden :cite:`Goyal.Dollar.Girshick.ea.2017,He.Zhang.Zhang.ea.2019`. Unsere Netze sind flach genug, um mit dem Standardwert $\gamma = 1$ zu trainieren. Wir behalten ihn bei. Ändern `use_1x1conv=True` oder `strides != 1` die Kanalzahl oder die räumliche Auflösung, nutzt die Sprungverbindung eine $1 \times 1$-Faltung mit passendem Stride und eine Batchnormalisierung. Damit ist die Addition wohldefiniert.

<!--en:6e7fd0da-->
NNX-Schichten benötigen beim Erstellen die Anzahl der Eingabekanäle. Daher hat `Residual` in JAX das Argument `in_channels` mit dem Standardwert `num_channels`.

<!--en:a886ef60-->
:numref:`fig_resnet_block` zeigt beide Varianten. Mit `use_1x1conv=True` oder einem Stride größer als 1 aktivieren wir die Sprungverbindung mit Projektion auf der rechten Seite.

<!--en:a7125e94-->
![Die beiden Varianten des ResNet-Blocks. Stimmen die Formen von Eingabe und Ausgabe überein, genügt ein Identitäts-Shortcut (links). Andernfalls passen eine $1 \times 1$-Faltung und eine Batchnormalisierung in der Sprungverbindung Kanalzahl und Auflösung an, sodass die Addition definiert ist (rechts).](../img/arch-resnet-block.svg)

<!--en:c74c34c3-->
Prüfen wir die Ausgabeformen. Bei gleicher Zahl von Eingabe- und Ausgabekanälen und Stride 1 behält der Tensor im Block seine Form `(4, 3, 6, 6)`.

<!--en:30a7a02f-->
Mit `num_channels=6`, `use_1x1conv=True` und `strides=2` verdoppelt sich die Kanalzahl, während sich Höhe und Breite halbieren. Genau dieser Downsampling-Block eröffnet weiter unten die Stufen 2, 3 und 4 von ResNet.

<!--en:186dc197-->
## Unverzweigte Netze und Residualnetze im Vergleich

<!--en:ca806a63-->
Können wir das Degradationsproblem direkt auf Fashion-MNIST reproduzieren und beobachten, wie Sprungverbindungen es beheben? Für einen kontrollierten Versuch bauen wir zwei Netzfamilien, die sich *nur* darin unterscheiden, ob sie Sprungverbindungen besitzen. Wir passen den CIFAR-10-Entwurf von :citet:`He.Zhang.Ren.ea.2016` an die ursprüngliche Bildgröße von Fashion-MNIST an. Mit $28 \times 28$ Pixeln lassen sich auch tiefe Netze schnell trainieren. Ein *unverzweigter Block* entspricht Schicht für Schicht einem `Residual`-Block, lässt aber die Addition über die Sprungverbindung weg. Er enthält zwei $3 \times 3$-Faltungen, jeweils gefolgt von Batchnormalisierung und ReLU.

<!--en:cac4eb24-->
Jedes Netz beginnt mit einer $3 \times 3$-Faltung mit 16 Kanälen, Batchnormalisierung und ReLU. Darauf folgen drei Stufen mit je $n$ Blöcken und 16, 32 beziehungsweise 64 Kanälen. Der erste Block der Stufen 2 und 3 verringert mit Stride 2 die Auflösung. Den Abschluss bilden Global-Average-Pooling und ein linearer Klassifikator. Zählen wir den Eingangsteil, zwei Faltungen pro Block und die lineare Ausgabeschicht, erhalten wir insgesamt $6n + 2$ Schichten: 20 für $n = 3$ und 56 für $n = 9$. Wir übergeben entweder `plain_block` oder `Residual` als `block` und erhalten so das Netzpaar für den Vergleich. Wie zuvor verwenden wir für alle Gewichte die He-Initialisierung (:numref:`subsec_he_init`) und trainieren mit `d2l.cosine_warmup` (:numref:`sec_alexnet`) bei einer maximalen Lernrate von 0.1. Diese Lernrate machte das Training unnormalisierter Netze in :numref:`sec_blocks` instabil. Hier verläuft es dank der Batchnormalisierung nach jeder Faltung stabil (:numref:`sec_batch_norm`).

<!--en:40b85248-->
`init.MSRAPrelu('in', 0)` ist in MXNet die Bezeichnung für die He-Initialisierung. Das Netz besteht aus `HybridBlock`-Bausteinen. Mit `hybridize()` kompilieren wir es zu einem einzigen Graphen. Diesen führt MXNet um ein Mehrfaches schneller aus als die einzelnen Operatoren nacheinander.

<!--en:18c9df80-->
Wie bei LeNet in :numref:`sec_lenet` ruft der Trainer die Modellmethode `weight_init` auf, sobald der erste Minibatch die Formen der verzögert initialisierten Schichten festgelegt hat. Die Methode nutzt die He-Initialisierung (`kaiming_normal_`) für alle Faltungsschichten und vollständig verbundenen Schichten, auch innerhalb der `Residual`-Blöcke.

<!--en:b98f6e97-->
`VarianceScaling(2.0, 'fan_in', 'untruncated_normal')` entspricht der He-Initialisierung. Verwenden mehrere Schichten dieselbe Instanz eines Keras-Initialisierers ohne Startwert für den Zufallsgenerator, erhalten sie dieselben Werte. Daher bekommt jede Schicht eine eigene Instanz. `Residual` verfährt intern genauso.

<!--en:4da82e81-->
NNX-Schichten ziehen ihre Gewichte bereits beim Erstellen. Daher erhält jede Schicht, auch innerhalb von `Residual`, die He-Initialisierung (`variance_scaling(2.0, 'fan_in', 'normal')`) als `kernel_init`.

<!--en:26832912-->
Wir trainieren alle vier Konfigurationen (unverzweigte Netze und Residualnetze mit jeweils 20 und 56 Schichten) fünf Epochen lang mit Batchgröße 128 und zeichnen die vollständigen Trainingsverläufe auf.

<!--en:dd7231ea-->
Tragen wir alle vier Kurven des Trainingsverlusts in dasselbe Diagramm mit logarithmischer y-Achse ein (blau für 20 Schichten, rot für 56, gestrichelt für unverzweigte Netze, durchgezogen für Residualnetze).

<!--en:5bed84df-->
Betrachten wir für alle vier Konfigurationen mit 20 und 56 Schichten auch die Werte von `train_loss` und `val_acc`, die `history` für die letzte Epoche enthält.

<!--en:734feddf-->
Bei den unverzweigten Netzen zeigt sich die Degradation deutlich. Das Netz mit 56 Schichten erreicht am Ende einen Trainingsverlust von etwa 0.25 bis 0.30, das mit 20 Schichten rund 0.20. Die Kurve des tieferen Netzes liegt in jeder einzelnen Epoche strikt *über* der des flacheren. Da bereits der Trainingsverlust höher ist, scheidet Überanpassung aus. Mit Sprungverbindungen kehrt sich die Reihenfolge um. Der Verlust des Residualnetzes mit 56 Schichten fällt auf etwa 0.15 bis 0.17. Es übertrifft damit beide Netze mit 20 Schichten, die mit und ohne Sprungverbindungen bei etwa 0.19 enden. Bei 20 Schichten machen sich die Sprungverbindungen kaum bemerkbar. Bei 56 Schichten entscheiden sie darüber, ob die 36 zusätzlichen Schichten die Optimierung erschweren oder verbessern. Da wir in diesen vier Läufen nur Tiefe und Sprungverbindungen variieren, lässt sich die Umkehr eindeutig den Sprungverbindungen zuschreiben. Bei der Validierungsgenauigkeit zeigt sich dasselbe Muster abgeschwächt. Das unverzweigte Netz mit 56 Schichten liegt nach fünf Epochen mit 88–90% zurück. Die übrigen drei liegen mit etwa 90.5–91.5% eng beieinander.

<!--en:3c0403d7-->
## ResNet-18

<!--en:d43a6c8c-->
### Eingangsteil, Stufen und Kopf

<!--en:35145558-->
Für ImageNet in voller Auflösung übernimmt ResNet die Gliederung von GoogLeNet in Eingangsteil, Hauptteil und Kopf (:numref:`sec_googlenet`). Der *Eingangsteil* (engl. stem) beginnt mit einer $7 \times 7$-Faltung mit 64 Kanälen und Stride 2, gefolgt von Batchnormalisierung, ReLU und $3 \times 3$-Max-Pooling mit Stride 2. Höhe und Breite sinken so gleich zu Beginn auf ein Viertel. Im folgenden Code baut `ResNet.stem` dieses Eingangsmodul.

<!--en:77b9068d-->
Der *Hauptteil* (engl. body) besteht aus vier *Stufen* (engl. stages). Jede enthält `num_residuals` aufeinandergestapelte Residualblöcke mit konstanter Kanalzahl. Da der Eingangsteil Höhe und Breite bereits geviertelt hat, behält Stufe 1 sowohl die Auflösung als auch die 64 Kanäle bei. Jede weitere Stufe beginnt mit einem Projektionsblock mit Stride 2. Er halbiert Höhe und Breite und verdoppelt die Kanalzahl. Die Methode `stage` baut eine solche Stufe.

<!--en:8345e519-->
Den Abschluss bildet der *Kopf* (engl. head) nach dem Vorbild von NiN und GoogLeNet. Auf Global-Average-Pooling folgt eine einzige lineare Schicht zur Klassifikation. Wie bei `VGG` gibt das Argument `arch` für jede Stufe `(num_residuals, num_channels)` vor. Alle Gewichte der Faltungen und linearen Schichten erhalten eine He-Initialisierung. Der einzige Bias liegt in der Ausgabeschicht und startet bei null.

<!--en:4b5dad34-->
Mit zwei Residualblöcken in jeder der vier Stufen erhalten wir *ResNet-18*. Vier Stufen $\times$ vier $3 \times 3$-Faltungen (ohne die $1 \times 1$-Projektionen der Sprungverbindungen) ergeben zusammen mit der $7 \times 7$-Faltung im Eingangsteil und der letzten linearen Schicht 18 Schichten mit Parametern. ResNet-34 verwendet `(3, 4, 6, 3)` Blöcke in den einzelnen Stufen. Tiefere Varianten (ResNet-50, -101 und -152) nutzen stattdessen dreischichtige *Bottleneck-Blöcke* ($1 \times 1 \to 3 \times 3 \to 1 \times 1$). Sie entsprechen dem weiter unten vorgestellten ResNeXt-Block mit nur einer Gruppe. :numref:`fig_resnet18` zeigt das gesamte ResNet-18.

<!--en:dbec9ebe-->
![ResNet-18 besteht aus einem Eingangsteil mit $7\times 7$-Faltung, vier Stufen mit je zwei Residualblöcken und einem Kopf aus Global-Average-Pooling und einer vollständig verbundenen Schicht. Der erste Block der Stufen 2, 3 und 4 halbiert die Auflösung und verdoppelt die Kanalzahl. Die angegebenen Formen gelten für eine Eingabe der Größe $96 \times 96$.](../img/arch-resnet18.svg)

<!--en:6aafeea6-->
Verfolgen wir die Tensorformen eines einkanaligen Bildes der Größe $96 \times 96$ durch ResNet-18. Auf dem Weg zum Global-Average-Pooling sinkt die Auflösung stufenweise, während die Kanalzahl zunimmt.

<!--en:70d292d9-->
### Training

<!--en:ce91f088-->
Für das Training von `ResNet18` übernehmen wir zunächst aus `DeepNet` den Kosinus-Lernratenplan mit Aufwärmphase für SGD. Dazu verwenden wir `configure_optimizers`.

<!--en:b890539e-->
Nun trainieren wir `ResNet18` auf Fashion-MNIST. Wir skalieren die Bilder auf $96 \times 96$ und trainieren 10 Epochen lang mit Batchgröße 128 und maximaler Lernrate `lr=0.1`.

<!--en:9974eeb8-->
Der letzte Eintrag in `trainer.history['val_acc']` liefert die Validierungsgenauigkeit am Ende des Kosinus-Lernratenplans über 10 Epochen.

<!--en:c451ee49-->
Nach 10 Epochen erreicht ResNet-18 etwa 93% Validierungsgenauigkeit. In den letzten drei Epochen schwankt sie um weniger als einen Prozentpunkt. Damit erreicht ResNet-18 die Genauigkeit der VGG-Variante mit einem Viertel der Kanalzahl aus :numref:`sec_blocks`. Es benötigt dazu nur ein Viertel ihrer Parameter und nutzt Bilder mit $96 \times 96$ statt $224 \times 224$ Pixeln. Wie immer sollten wir aus einem einzelnen Vergleich zwischen Modellfamilien keine zu weitreichenden Schlüsse ziehen: Architektur, Normalisierung und Eingabeauflösung haben sich zugleich geändert. Den Nutzen der Residualverbindung belegt der kontrollierte Vergleich der Netze mit 20 und 56 Schichten in :numref:`subsec_plain_residual`.

<!--en:2d188fba-->
Beachten wir auch ein typisches Anzeichen für Überanpassung. ResNet-18 senkt den Trainingsverlust auf unter 0.01, während der Validierungsverlust nach den ersten Epochen bei etwa 0.27 stagniert. Mit 11.2 Millionen Parametern fällt es dem Netz leicht, die 60 000 Graustufenbilder von Kleidungsstücken im Trainingsdatensatz auswendig zu lernen. Um diese Generalisierungslücke zu schließen, benötigen wir stärkere Datenaugmentierung und Regularisierung. Beides untersuchen wir systematisch in :numref:`sec_training_recipes`.

<!--en:6792a7b4-->
## ResNeXt und DenseNet

<!--en:b73dedb5-->
Residualverbindungen legen fest, *wie* ein Block seine Eingabe weitergibt: Er addiert sie zum Ergebnis der gelernten Transformation. Zwei bedeutende Nachfolger greifen die Idee der Sprungverbindung auf und verändern den Block selbst. *ResNeXt* gestaltet die Faltungen im Block so, dass die Kanalzahl ohne zusätzlichen Rechenaufwand steigt. *DenseNet* ersetzt die Addition in den Sprungverbindungen durch eine Verkettung entlang der Kanalachse. Auf den ResNeXt-Block kommen wir zurück, wenn wir in :numref:`sec_cnn-design` den Entwurfsraum für Netze systematisch untersuchen.

<!--en:6c893887-->
### ResNeXt

<!--en:a173d36b-->
Breitere Schichten lernen reichhaltigere Darstellungen. Der Aufwand einer herkömmlichen Faltung von $c_\textrm{i}$ Kanälen auf $c_\textrm{o}$ Kanäle wächst jedoch mit $c_\textrm{i} c_\textrm{o}$ (:numref:`sec_channels`). Bei doppelter Breite vervierfachen sich also sowohl die Anzahl der Gewichte als auch der Rechenaufwand. *ResNeXt* :cite:`Xie.Girshick.Dollar.ea.2017` greift die Idee der verzweigten Inception-Blöcke aus GoogLeNet auf (:numref:`sec_googlenet`) und vereinheitlicht die Struktur vollständig: Jeder Zweig hat genau dieselbe Topologie.

<!--en:8c7b6b5b-->
Eine *gruppierte Faltung* (engl. grouped convolution) teilt die $c_\textrm{i}$ Eingabekanäle in $g$ disjunkte Gruppen mit je $c_\textrm{i}/g$ Kanälen. Für jede Gruppe berechnet sie unabhängig durch Faltung $c_\textrm{o}/g$ Ausgabekanäle. Dabei setzen wir voraus, dass $g$ sowohl $c_\textrm{i}$ als auch $c_\textrm{o}$ teilt. Jede Gruppe benötigt einen Faltungskern der Größe $(c_\textrm{i}/g) \times (c_\textrm{o}/g) \times k \times k$. Über alle $g$ Gruppen hinweg sinkt die Anzahl der Parameter und der Multiplikationsadditionen um den Faktor $g$ von $c_\textrm{i} c_\textrm{o} k^2$ auf $c_\textrm{i} c_\textrm{o} k^2 / g$. (In der Praxis beträgt der Geschwindigkeitsgewinn meist weniger als das $g$-Fache, weil die übertragenen Datenmengen langsamer sinken als der Rechenaufwand. Diesen Unterschied untersuchen wir in :numref:`sec_efficient_cnns`.) Gruppierte Faltungen kamen erstmals in AlexNet zum Einsatz :cite:`Krizhevsky.Sutskever.Hinton.2012`. Drei seiner fünf Faltungsschichten waren in je zwei Gruppen aufgeteilt, damit sich das Netz auf zwei GPUs verteilen ließ.

<!--en:0cdd3ac3-->
Unabhängige Gruppen können von sich aus keine Informationen austauschen. Deshalb liegt im ResNeXt-Block (:numref:`fig_resnext_block`) eine gruppierte $3 \times 3$-Faltung zwischen zwei dichten $1 \times 1$-Faltungen. Die erste projiziert $c$ Eingabekanäle auf $b$ Kanäle im *Flaschenhals*. Die zweite projiziert zurück auf $c$ Kanäle und mischt dabei die Informationen aus allen Gruppen. Die beiden $1 \times 1$-Schichten benötigen $2cb$ Gewichte, die gruppierte $3 \times 3$-Schicht benötigt $9b^2/g$. Beim Downsampling verwendet die $3 \times 3$-Faltung den Stride 2, während die erste $1 \times 1$-Projektion die Auflösung beibehält. Die Sprungverbindung entspricht der in `Residual`.

<!--en:d785bcfb-->
Was „breiter bei gleichem Aufwand“ in Zahlen bedeutet, zeigt der Vergleich eines herkömmlichen Bottleneck-Blocks von ResNet-50 ($c = 256$, $b = 64$, $g = 1$) mit $2 \cdot 256 \cdot 64 + 9 \cdot 64^2 = 69{,}632$ Gewichten und des ResNeXt-Blocks in :numref:`fig_resnext_block` ($b = 128$, $g = 32$) mit $2 \cdot 256 \cdot 128 + 9 \cdot 128^2/32 = 70{,}144$ Gewichten. Die innere Breite des Flaschenhalses verdoppelt sich bei weniger als 1% zusätzlichen Gewichten. Auf ImageNet sinkt der Top-1-Fehler von ResNet-50 durch diese Änderung von 23.9% auf 22.2% :cite:`Xie.Girshick.Dollar.ea.2017`.

<!--en:b595fbd7-->
![Im ResNeXt-Block liegt eine gruppierte $3 \times 3$-Faltung mit $\mathit{g}$ Gruppen zwischen zwei $1 \times 1$-Faltungen, die Informationen über die Gruppen hinweg mischen. Der gezeigte Block hat $\mathit{c} = 256$ Kanäle, $\mathit{b} = 128$ Kanäle im Flaschenhals und $\mathit{g} = 32$ Gruppen zu je vier Kanälen.](../img/arch-resnext-block.svg)

<!--en:f08337d9-->
Im folgenden Code erhält `ResNeXtBlock` als Argumente die Ausgabebreite `num_channels` ($c$), die Gruppenanzahl `groups` ($g$) und das Breitenverhältnis des Flaschenhalses `bot_mul` ($b/c$). Wie bei `Residual` halbiert `strides=2` sowohl Höhe als auch Breite und aktiviert automatisch die $1 \times 1$-Projektion in der Sprungverbindung.

<!--en:ce969008-->
Prüfen wir, ob ein Block mit 32 Kanälen, 16 Gruppen und `bot_mul=1` die Form seines Eingabetensors beibehält.

<!--en:43536b51-->
### DenseNet

<!--en:307785f8-->
Ein Residualblock addiert die Ausgabe seines Residualzweigs zur Eingabe ($\mathbf{x}_k = \mathbf{x}_{k-1} + g_k(\mathbf{x}_{k-1})$). Dabei bleibt die Kanalzahl konstant, und alte und neue Merkmale vermischen sich. *DenseNet* (engl. dense convolutional network) :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` hingegen *verkettet* die Ausgabe jeder Schicht mit ihrer Eingabe entlang der Kanalachse:

<!--en:88f97c2c-->
Dabei bezeichnet $[\cdot, \cdot]$ die Verkettung entlang der Kanalachse. Alle Werte bleiben erhalten: $\mathbf{x}_k$ enthält die ursprüngliche Blockeingabe zusammen mit den Ausgaben aller $k$ vorangegangenen Schichten. Jede Schicht ist also mit jeder früheren Schicht des Blocks direkt verbunden. :numref:`fig_densenet_block` zeigt einen dichten Block mit zwei Schichten.

<!--en:0759dfd9-->
![Ein dichter Block mit zwei Schichten und Wachstumsrate 32. Jede Schicht hängt ihre 32 Ausgabekanäle an die Eingabe an. Aus 64 Eingabekanälen werden so zunächst 96 und dann 128 Kanäle. Jede Schicht erhält die Ausgaben aller vorangegangenen Schichten.](../img/arch-densenet-block.svg)

<!--en:faec3e00-->
Für jede Faltung im dichten Block gilt die Reihenfolge der Voraktivierung (Batchnormalisierung $\to$ ReLU $\to$ Faltung ohne Bias). In den Übungen unten untersuchen wir diese Reihenfolge für ResNet. Die Anzahl neuer Kanäle, die jede Schicht anhängt (32 in :numref:`fig_densenet_block`), heißt *Wachstumsrate* (engl. growth rate). Da die Kanalzahl in jedem dichten Block linear wächst, fügt DenseNet zwischen aufeinanderfolgenden Blöcken eine *Übergangsschicht* (engl. transition layer) ein. Eine $1 \times 1$-Faltung verringert darin die Kanalzahl, und Average-Pooling mit Stride 2 halbiert Höhe und Breite. Ein vollständiges DenseNet enthält entsprechend den vier Stufen von ResNet-18 vier dichte Blöcke mit Übergangsschichten dazwischen. In Aufgabe :exref:`ex_resnet-densenet` bauen und trainieren wir ein solches Netz.

<!--en:4ef70d83-->
DenseNet erreichte auf ImageNet mit weniger Parametern dieselbe Genauigkeit wie ResNet. Die Autoren führten diesen Erfolg darauf zurück, dass das Netz Merkmale vielfach wiederverwendet. Ein Nachteil in der Praxis ist jedoch der Aktivierungsspeicher. Zwar wächst die Zahl der *verschiedenen* Merkmalskarten linear mit der Blocktiefe, doch eine direkte Implementierung legt in jeder Schicht einen neuen, breiteren Tensor für die Verkettung und eine normalisierte Kopie an. Sie speichert beides für die Backpropagation. Dadurch steigt der Speicherbedarf *quadratisch* mit der Tiefe, sofern keine spezialisierte Implementierung die Zwischentensoren im Rückwärtsdurchlauf neu berechnet :cite:`pleiss2017memory`. Additive Residualverbindungen vermeiden diesen zusätzlichen Speicherbedarf vollständig. Das ist ein wesentlicher Grund dafür, dass additive Skip-Verbindungen im gesamten Deep Learning zum Standard wurden.

<!--en:c29e608b-->
Skip-Verbindungen mit Verkettung sind jedoch für eine bedeutende Architekturfamilie zentral: das *U-Net* :cite:`Ronneberger.Fischer.Brox.2015` zur Bildsegmentierung. Es verringert die Auflösung des Bildes stufenweise und erhöht sie anschließend wieder bis zur vollen Auflösung. Jede hochskalierte Merkmalskarte verkettet es mit der passenden hochaufgelösten Karte aus dem Encoder-Pfad. Die Verkettung findet nur einmal pro Auflösungsstufe statt. Dadurch bleibt der Speicherbedarf gering. Dieselbe U-Net-Grundarchitektur diente später als Basis für moderne Diffusionsmodelle für Bilder (:numref:`sec_diffusion-images`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:6208eea3-->
Überraschenderweise können zusätzliche Schichten in einem unverzweigten Netz mit Batchnormalisierung den Trainingsverlust gegenüber einem flacheren Modell *erhöhen*. Dabei umfasst die Funktionenklasse der tieferen Architektur die des flacheren Modells als echte Teilmenge. Residualblöcke beheben dieses Degradationsproblem, indem sie eine additive Korrektur lernen: $\mathbf{x} + g(\mathbf{x})$. Die Identitätsabbildung liegt nun bei $g = 0$. Dank der Sprungverbindung bleiben die Gradienten auch bei zunehmender Tiefe kohärent und die Verlustlandschaft glatt. Auf Fashion-MNIST erzielt ein unverzweigtes Netz mit 56 Schichten beim Training schlechtere Ergebnisse als eines mit 20 Schichten. Mit Sprungverbindungen schneidet das Modell mit 56 Schichten besser ab. Ordnen wir Residualblöcke mit Batchnormalisierung in vier Stufen an, erhalten wir ResNet-18 (etwa 93% Validierungsgenauigkeit auf Fashion-MNIST in 10 Epochen). Der Ansatz lässt sich problemlos auf mehr als 150 Schichten erweitern. ResNeXt verbreitert Bottleneck-Blöcke bei gleichem Aufwand durch gruppierte Faltungen. DenseNet ersetzt additive Sprungverbindungen durch die Verkettung von Merkmalen und nimmt dafür einen höheren Speicherbedarf in Kauf. Residualverbindungen bilden inzwischen ein unverzichtbares Grundelement fast jeder modernen Architektur, auch der Transformer in :numref:`chap_transformers`.

<!--en:2294243a-->
## Übungen

<!--en:1cfa2051-->
1. **Inception- und Residualblöcke im Vergleich.** In der Architektur `arch` von GoogLeNet (:numref:`sec_googlenet`) hat der zweite Block der mittleren Stufe die Zweigbreiten $c_1 = 160$, $c_2 = (112, 224)$, $c_3 = (24, 64)$ und $c_4 = 64$. Er bildet 512 Eingabekanäle auf 512 Ausgabekanäle ab. Vergleichen Sie ihn mit einem `Residual`-Block mit 512 Kanälen und einem Identitäts-Shortcut:
    1. Bestimmen Sie für beide Blöcke die Anzahl der Faltungsgewichte und der Multiplikationsadditionen pro Ausgabepixel. Vernachlässigen Sie Bias-Terme und Parameter der Batchnormalisierung.
    1. Gibt der Residualzweig eines `Residual`-Blocks null aus, berechnet der Block für nichtnegative Eingaben die Identität. Solche Eingaben liegen immer dann vor, wenn der Block auf eine ReLU folgt. Prüfen Sie, was `bn2` im Trainings- und Vorhersagemodus ausgibt, wenn Sie `conv2` auf null setzen. Geben Sie an, mit welchen Parametern von `bn2` der Block in beiden Modi exakt die Identität berechnet. Gibt es eine Parameterwahl, mit der der vierzweigige Inception-Block auf nichtnegativen Eingaben die Identität berechnet?

<!--en:3d66c141-->
1. **Reihenfolge bei der Voraktivierung.** :citet:`He.Zhang.Ren.ea.2016*1` verlegten Batchnormalisierung und ReLU *vor* jede Faltung ($\textrm{BN} \to \textrm{ReLU} \to \textrm{Conv}$) und entfernten die ReLU nach der Addition. Jeder Block berechnet somit $\mathbf{x}_k = \mathbf{x}_{k-1} + g_k(\mathbf{x}_{k-1})$, und der direkte Identitätspfad bleibt durchgängig erhalten. Dieselbe Reihenfolge verwenden die DenseNet-Blöcke in :numref:`sec_densenet`.
    1. Drücken Sie für $K$ aufeinandergestapelte Blöcke mit Voraktivierung $\mathbf{x}_K$ durch $\mathbf{x}_0$ und die Ausgaben der Residualzweige aus. Zeigen Sie, dass die Jacobi-Matrix $\partial \mathbf{x}_K / \partial \mathbf{x}_0$ die Einheitsmatrix $\mathbf{I}$ exakt als additiven Term enthält.
    1. Betrachten Sie $K$ herkömmliche `Residual`-Blöcke mit Nachaktivierung, die jeweils nach der Addition eine ReLU anwenden. Welcher Faktor ersetzt $\mathbf{I}$ in $\partial \mathbf{x}_K / \partial \mathbf{x}_0$?

<!--en:75ba313b-->
1. [code] **Gradienten unverzweigter Netze.** ● :citet:`He.Zhang.Ren.ea.2016` beobachteten, dass in unverzweigten Netzen mit Batchnormalisierung trotz der Degradation bei zunehmender Tiefe keine verschwindenden Gradienten auftreten. Überprüfen Sie dies an den vier Netzen mit 20 und 56 Schichten aus :numref:`subsec_plain_residual`:
    1. Berechnen Sie bei der Initialisierung und nach 1 Trainingsepoche den Gradienten des Verlusts auf einem Minibatch. Tragen Sie für alle vier Netze die Frobeniusnorm des Gewichtsgradienten jeder Faltungsschicht gegen den Schichtindex auf. Verschwinden die Gradienten des unverzweigten Netzes mit 56 Schichten zur Eingabe hin?
    1. Trainieren Sie das unverzweigte Netz mit 56 Schichten über 20 statt 5 Epochen. Erreicht es den Trainingsverlust des unverzweigten Netzes mit 20 Schichten?

<!--en:c8b7e14f-->
1. [code] **ResNet-34 und die Bottleneck-Blöcke von ResNet-50.** ● Tabelle 1 in :citet:`He.Zhang.Ren.ea.2016` definiert ResNet-18 bis ResNet-152 anhand der Tiefe der einzelnen Stufen und des Blocktyps.
    1. Bauen Sie ResNet-34 (`arch=((3, 64), (4, 128), (6, 256), (3, 512))`) mithilfe von `ResNet` und vergleichen Sie seine Parameteranzahl mit der von `ResNet18`.
    1. ResNet-50 ersetzt den zweischichtigen `Residual`-Block durch einen dreischichtigen Bottleneck-Block: Eine $1 \times 1$-Faltung verringert die Breite um den Faktor $4\times$. Auf eine $3 \times 3$-Faltung folgt eine $1 \times 1$-Faltung, die die ursprüngliche Breite wiederherstellt. Dies implementiert `ResNeXtBlock` mit `groups=1, bot_mul=0.25`. Die Stufenausgaben sind $4\times$ so breit und haben die Kanalzahlen `(256, 512, 1024, 2048)`. Vergleichen Sie die Parameteranzahl und die Anzahl der Multiplikationsadditionen für einen Bottleneck-Block von ResNet-50 der Breite $4c$ und einen `Residual`-Block von ResNet-34 der Breite $c$. Überschreiben Sie dann `stage` in einer Unterklasse von `ResNet`, um ResNet-50 zu bauen, und prüfen Sie die Gesamtzahl seiner Parameter.

<!--en:0bb65924-->
1. [code] **Sprungverbindung für das Downsampling in ResNet-D.** In einem `Residual`-Block mit Stride 2 erfasst die $1 \times 1$-Faltung der Sprungverbindung mit Stride 2 nur eine von vier räumlichen Positionen. Die übrigen drei verwirft sie. :citet:`He.Zhang.Zhang.ea.2019` ersetzen sie durch $2 \times 2$-Average-Pooling mit Stride 2, gefolgt von einer $1 \times 1$-Faltung mit Stride 1 und Batchnormalisierung. Implementieren Sie diese Sprungverbindung aus ResNet-D in einer Unterklasse von `Residual`. Bestätigen Sie, dass Ausgabeformen und Parameteranzahl gleich bleiben, und vergleichen Sie die Validierungsgenauigkeit auf Fashion-MNIST nach 10 Epochen über mehrere Läufe mit der von `ResNet18`.

<!--en:9e4e8c78-->
1. [code] **ResNet-18 mit Voraktivierung.** ● Implementieren Sie den Block mit Voraktivierung von :citet:`He.Zhang.Ren.ea.2016*1` als Variante von `Residual`. Überschreiben Sie `stage` in einer Unterklasse von `ResNet`, um diese Blöcke zu stapeln, und vergleichen Sie die Trainingskurve mit der von `ResNet18`.

<!--en:fecf63c0-->
1. [code] **DenseNet.** ● Bauen und trainieren Sie das DenseNet aus :numref:`sec_densenet`. Bestimmen Sie anschließend seine Parameteranzahl und seinen Bedarf an Aktivierungsspeicher.
    1. Implementieren Sie einen `DenseBlock` aus `num_convs` Einheiten der Form $(\textrm{BN} \to \textrm{ReLU} \to \textrm{Conv}_{3\times 3}\text{ without bias})$. Verketten Sie jede Ausgabe mit `growth_rate` Kanälen entlang der Kanalachse mit ihrer Eingabe. Implementieren Sie auch eine Übergangsschicht der Form $(\textrm{BN} \to \textrm{ReLU} \to \textrm{Conv}_{1\times 1}\text{ without bias} \to \textrm{AvgPool}_{2\times 2, s=2})$. Prüfen Sie an einer Eingabe mit 3 Kanälen, dass ein dichter Block mit 2 Faltungen und Wachstumsrate 10 insgesamt $3 + 10 + 10 = 23$ Kanäle ausgibt. Prüfen Sie außerdem, dass eine Übergangsschicht mit 10 Kanälen die Kanalzahl auf 10 reduziert und dabei Höhe und Breite halbiert.
    1. Bauen Sie ein DenseNet mit dem Eingangsteil von `ResNet` und vier dichten Blöcken aus je 4 Faltungen mit Wachstumsrate 32. Fügen Sie zwischen aufeinanderfolgenden dichten Blöcken Übergangsschichten ein, die die Kanalzahl halbieren, und setzen Sie $\textrm{BN} \to \textrm{ReLU}$ vor den Kopf von `ResNet`. Trainieren Sie das Netz auf $96 \times 96$ großen Fashion-MNIST-Bildern und vergleichen Sie Parameteranzahl, Validierungsgenauigkeit und Zeit pro Epoche mit den Werten von `ResNet18`.
    1. Betrachten Sie einen dichten Block, dessen $n$ Faltungen jeweils $r$ Kanäle ausgeben und dessen Eingabe $c$ Kanäle hat. Vergleichen Sie ihn mit $n$ `Residual`-Blöcken konstanter Breite $c$. Eine naive Implementierung hält alle verketteten Eingabetensoren für den Rückwärtsdurchlauf vor. Wie viele Kanäle liest die $i$-te Faltung, und wie groß ist die Summe über $i = 1, \ldots, n$? Vergleichen Sie diese Summe mit den $2nc$ Kanälen, die $n$ Residualblöcke vorhalten, und mit den $c + nr$ verschiedenen Kanälen, die :citet:`pleiss2017memory` speichern. Verwenden Sie dazu $c = 64, r = 32, n = 6$.
   :exlabel:`ex_resnet-densenet`

<!--en:a60bc53b-->
[Diskussion](https://d2l.discourse.group/t/85)

<!--en:3813142b-->
[Diskussion](https://d2l.discourse.group/t/86)

<!--en:8ff75b28-->
[Diskussion](https://d2l.discourse.group/t/8737)

<!--en:21ad2648-->
[Diskussion](https://d2l.discourse.group/t/18006)

<!--en:b6e21988-->
Residualnetze: ResNet, ResNeXt und DenseNet<br>**Degradationsparadox · geschachtelte Funktionenklassen · additive Sprungverbindungen · ResNet-18 · gruppierte Faltungen und Verkettung**

<!--en:2270986f-->
::: {.slide title="Das Degradationsparadox in tiefen unverzweigten Netzen"}

<!--en:0156804f-->
[Motivation und offene Frage zur Optimierung]{.kicker}

<!--en:2fdab455-->
- **Tiefere unverzweigte Netze passen schlechter zu den Trainingsdaten:** Ein unverzweigtes CNN mit 56 Schichten erreicht einen höheren *Trainingsverlust* ($\approx 0.28$) als eines mit 20 Schichten ($0.20$). Überanpassung scheidet damit aus.

<!--en:0abde548-->
- **Konstruktiver Existenzbeweis:** Wir kopieren das Netz mit 20 Schichten und setzen die übrigen 36 Schichten auf die Identität $f(\mathbf{x}) = \mathbf{x}$. Damit existiert im tieferen Netz eine Lösung mit Verlust $\le 0.20$. SGD gelingt es jedoch nicht, $f(\mathbf{x}) = \mathbf{x}$ über unverzweigte nichtlineare Schichten hinweg anzunähern.

<!--en:312c253c-->
::: {.slide title="Geschachtelte Funktionenklassen garantieren monotone Verbesserung"}

<!--en:5e8a4952-->
[Theoretische Einordnung]{.kicker}

<!--en:d926bdde-->
Betrachten wir eine Folge von Modellfamilien $\mathcal{F}_1, \mathcal{F}_2, \ldots, \mathcal{F}_K$, mit denen wir eine Zielabbildung $f^*$ annähern wollen:

<!--en:148e4bc2-->
- **Nicht geschachtelte Klassen ($\mathcal{F} \not\subseteq \mathcal{F}'$):** Erweitern wir eine Schicht, ohne die bisherigen Abbildungen zu erhalten, kann sich die gesamte Klasse von $f^*$ entfernen.

<!--en:e5cc70df-->
- **Geschachtelte Klassen ($\mathcal{F} \subseteq \mathcal{F}'$):** Der kleinstmögliche Verlust kann mit zunehmender Tiefe nur sinken oder gleich bleiben:
  $$\mathcal{F} \subseteq \mathcal{F}' \implies \min_{f \in \mathcal{F}'} L(f) \le \min_{f \in \mathcal{F}} L(f)$$

<!--en:87f897b9-->
- **Lücke bei der Optimierung:** Eine unverzweigte Schicht $\textrm{ReLU}(\textrm{BN}(\mathbf{W}\mathbf{x}))$ *kann* die Eingabe $\mathbf{x}$ für $\mathbf{x} \ge 0$ unverändert ausgeben. Theoretisch sind die Klassen also geschachtelt. Die Identität erfordert jedoch eine ungünstige Gewichtskonfiguration im Inneren des Parameterraums, die SGD nicht erreicht.

<!--en:6d68295f-->
::: {.slide title="Residualblöcke lernen Korrekturen zur Identität"}

<!--en:836521cf-->
[Der Residualblock]{.kicker}

<!--en:352c74d5-->
Ein herkömmlicher Block muss die gewünschte Abbildung $f(\mathbf{x})$ direkt parametrisieren. Ein Residualblock lernt die Korrektur, das *Residuum* $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$, und addiert über eine Sprungverbindung die Eingabe wieder dazu:

<!--en:cba7621a-->
- **Die Identität liegt im Ursprung ($g(\mathbf{x}) = \mathbf{0}$):**
  $$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x})$$

<!--en:19a9f8bd-->
- **Die Identität ist leicht zu erreichen:** Damit der Block $\mathbf{x}$ ausgibt, genügt es, die Gewichte (oder die Parameter $\gamma_2, \beta_2$ der letzten Batchnormalisierung) des Residualzweigs $g$ gegen null zu führen.

<!--en:f107f1ae-->
- **Vorkonditionierung um die Identität:** Selbst für $g(\mathbf{x}) \neq \mathbf{0}$ ist das Lernen kleiner Korrekturen um $\mathbf{x}$ wesentlich besser konditioniert als das vollständige Erlernen einer Koordinatentransformation, die fast der Identität entspricht.

<!--en:ede30e22-->
::: {.slide title="Die Identität als direkter Gradientenpfad durch alle Schichten"}

<!--en:45a1a20f-->
[Jacobi-Matrix im Rückwärtsdurchlauf]{.kicker}

<!--en:958253b1-->
- **Additiver Term $\mathbf{I}$:** Durch Auflösen der Rekursion ergibt sich $\mathbf{x}_L = \mathbf{x}_\ell + \sum g_k$. So gelangen die zurückfließenden Gradienten $\partial L / \partial \mathbf{x}_L$ direkt zu frühen Schichten $\mathbf{x}_\ell$, ohne Multiplikation mit den dazwischenliegenden Gewichtsmatrizen.

<!--en:dbc5ca70-->
::: {.slide title="Sprungverbindungen mit Identität oder 1×1-Projektion"}

<!--en:c92c7fa9-->
[Aufbau des Blocks]{.kicker}

<!--en:5d018d98-->
Der grundlegende `Residual`-Block von ResNet enthält zwei $3\times 3$-Faltungen ohne Bias mit Batchnormalisierung. Er addiert die Eingabe über die Sprungverbindung vor der zweiten ReLU und nutzt bei einem Dimensionswechsel eine $1\times 1$-Projektion:

<!--en:e249fb2e-->
- **Identitäts-Shortcut (links, gleiche Form):** Bei $c_{\textrm{o}} = c_{\textrm{i}}$ und Stride $s = 1$ addiert der Block $\mathbf{x}$ direkt, **ganz ohne zusätzliche Parameter**.

<!--en:d69c0764-->
- **Sprungverbindung mit Projektion (rechts, Downsampling):** Bei $c_{\textrm{o}} \neq c_{\textrm{i}}$ oder Stride $s = 2$ passt eine Faltung mit Stride $s$ und Kerngröße $1\times 1$ zusammen mit einer Batchnormalisierung die Eingabe $\mathbf{x}$ an räumliche Größe und Kanalzahl an.

<!--en:878951c7-->
- **Nullausgabe des Residualzweigs:** Mit $\gamma_2 = \beta_2 = 0$ in der zweiten Batchnormalisierung liefert der Residualzweig sowohl im Trainings- als auch im Auswertungsmodus null. Damit gilt wieder $\textrm{ReLU}(\mathbf{x}) = \mathbf{x}$ für $\mathbf{x} \ge 0$.

<!--en:b75c468c-->
::: {.slide title="Implementierung des Residualblocks" except="jax"}

<!--en:a4fbb6b9-->
[Implementierung]{.kicker}

<!--en:7238f9e5-->
Mit `use_1x1conv` oder `strides != 1` aktivieren wir die $1\times 1$-Projektion der Sprungverbindung aus `conv3` und `bn3`:

<!--en:cb5c1623-->
::: {.slide title="Implementierung des Residualblocks: Faltungen" only="jax"}

<!--en:a4fbb6b9-2-->
[Implementierung]{.kicker}

<!--en:7238f9e5-2-->
Mit `use_1x1conv` oder `strides != 1` aktivieren wir die $1\times 1$-Projektion der Sprungverbindung aus `conv3` und `bn3`:

<!--en:376dc88d-->
::: {.slide title="Implementierung des Residualblocks: Sprungverbindung und Vorwärtsdurchlauf" only="jax"}

<!--en:fd54fe25-->
::: {.slide title="Formprüfung bei gleichbleibender Auflösung und bei Downsampling mit Stride 2"}

<!--en:b1e57915-->
[Prüfung der Tensorformen im Block]{.kicker}

<!--en:f80f8026-->
::: {.slide title="Kontrollierter Versuch: unverzweigte Netze und ResNets mit 20 und 56 Schichten"}

<!--en:609d293c-->
[Reproduktion der Degradation]{.kicker}

<!--en:a6240732-->
Für den kontrollierten Versuch auf $28\times 28$ großen Fashion-MNIST-Bildern bauen wir `DeepNet` mit $6n+2$ Schichten: eine $3\times 3$-Faltung mit 16 Kanälen im Eingangsteil, drei Stufen mit je $n$ Blöcken bei 16, 32 und 64 Kanälen sowie ein linearer Kopf mit Pooling. Wir vergleichen `plain_block` und `Residual` für $n=3$ (20 Schichten) und $n=9$ (56 Schichten):

<!--en:9489ec14-->
- **Degradation in unverzweigten Netzen (Verlust: $56 > 20$):** Die Kurve für 56 Schichten (rot gestrichelt) liegt in jeder Epoche strikt *über* der für 20 Schichten (blau gestrichelt). Der Trainingsverlust beträgt am Ende **$0.25\text{--}0.30$ gegenüber $0.20$**.

<!--en:4b933554-->
- **Sprungverbindungen kehren die Reihenfolge um (Verlust: $56 < 20$):** Ohne einen einzigen zusätzlichen Parameter im Hauptzweig sinkt der Verlust des ResNets mit 56 Schichten (rot durchgezogen) auf **$0.15\text{--}0.17$** und liegt damit unter dem Verlust beider Modelle mit 20 Schichten ($\approx 0.19$).

<!--en:bd41cc28-->
::: {.slide title="ResNet-18: Eingangsteil, vier Stufen und Kopf mit Pooling"}

<!--en:1cb9acb5-->
[Architektur von ResNet-18]{.kicker}

<!--en:4d7d3831-->
Für Bilder in voller Auflösung übernimmt ResNet die Gliederung von GoogLeNet in Eingangsteil, Hauptteil und Kopf. Der Hauptteil enthält vier Residualstufen:

<!--en:cb9e0c1b-->
- **Eingangsteil mit 4× Downsampling (1 Schicht mit Gewichten):** Eine $7\times 7$-Faltung mit 64 Kanälen und Stride 2 ($p=3$), Batchnormalisierung, ReLU und $3\times 3$-Max-Pooling mit Stride 2 ($p=1$) verkleinern die Auflösung: $96\times 96 \to 24\times 24$.

<!--en:7402d91d-->
- **Vier Residualstufen (16 Schichten mit Gewichten):** Jede Stufe enthält 2 `Residual`-Blöcke, die Kanalzahlen betragen $(64, 128, 256, 512)$. Stufe 1 behält $24\times 24$ bei. Die Stufen 2–4 beginnen mit einem Projektionsblock mit Stride 2 ($24$ $\to$ $12$ $\to$ $6$ $\to$ $3$).

<!--en:f029ef7a-->
- **Kopf mit Pooling (1 Schicht mit Gewichten):** Global-Average-Pooling und eine lineare Schicht $512 \to 10$ bilden den Kopf. Das Netz hat insgesamt $1 + 16 + 1 = 18$ Schichten mit **11.18 Mio. Parametern**.

<!--en:9bfa17c0-->
::: {.slide title="Aufbau der Stufen und Verlauf der Tensorformen in ResNet-18" layout="tall-output"}

<!--en:a1b2534c-->
[Stufenaufbau und Tensorformen]{.kicker}

<!--en:d50d524c-->
In `ResNet.stage` halbiert der erste Block der Stufen 2, 3 und 4 (`not first_block`) sowohl Höhe als auch Breite und verdoppelt die Kanalzahl:

<!--en:906819fe-->
::: {.slide title="Training von ResNet-18 auf Fashion-MNIST"}

<!--en:3078e51d-->
[Versuchsergebnisse]{.kicker}

<!--en:e48c6213-->
Dank der Batchnormalisierung nach jeder Faltung trainiert ResNet-18 zuverlässig mit maximaler Lernrate $\eta_0 = 0.1$ über 10 Epochen auf $96\times 96$ großen Fashion-MNIST-Bildern:

<!--en:653412df-->
- **Hohe Genauigkeit mit $4\times$ weniger Gewichten als VGG:** Mit **$\approx 93\%$** Validierungsgenauigkeit bei $11.2\text{M}$ Parametern ($319\text{M}$ Multiplikationsadditionen bei $96\times 96$ Pixeln) erreicht ResNet-18 die Genauigkeit der VGG-Variante mit einem Viertel der Kanalzahl ($43.1\text{M}$ Parameter bei $224\times 224$ Pixeln).

<!--en:de20ae78-->
- **Auswendiglernen ohne moderne Regularisierung:** ResNet-18 senkt den Trainingsverlust auf unter $0.01$, während der Validierungsverlust bei etwa $0.27$ stagniert. Das motiviert die Verfahren zur Datenaugmentierung und Regularisierung in §{sec_training_recipes}.

<!--en:3f3fd74a-->
::: {.slide title="ResNeXt: breitere Flaschenhälse durch gruppierte Faltungen"}

<!--en:ce700238-->
[ResNeXt und Mächtigkeit]{.kicker}

<!--en:fb7cd37a-->
Verdoppeln wir die Kanalzahl, vervierfacht sich der Aufwand einer dichten Faltung ($c_{\textrm{i}} c_{\textrm{o}} k^2$). *ResNeXt* (Xie et al., 2017) verbreitert den internen Flaschenhals bei gleichem Aufwand. Dazu zerlegt es die $3\times 3$-Faltung in $g$ unabhängige Gruppen (*Mächtigkeit*):

<!--en:63b8636d-->
- **Ersparnis durch gruppierte $3\times 3$-Faltungen:** Teilen wir $b$ Kanäle im Flaschenhals in $g$ Gruppen zu je $b/g$ Kanälen auf, sinkt die Anzahl der $3\times 3$-Gewichte um den Faktor $g$ von $9b^2$ auf $9(b/g)^2 \cdot g = 9b^2/g$.

<!--en:dbeabe0f-->
- **Alle Kanäle durch $1\times 1$-Faltungen mischen:** Die gruppierte $3\times 3$-Schicht liegt zwischen zwei dichten $1\times 1$-Faltungen ($c \to b$ und $b \to c$ mit insgesamt $2cb$ Gewichten). Dadurch tauschen die Gruppen in jedem Block Informationen aus.

<!--en:60e6ed72-->
- **Breiter bei gleichem Aufwand:** ResNet-50 ($c=256, b=64, g=1$) benötigt $2(256)(64) + 9(64^2) = 69{,}632$ Gewichte. ResNeXt-50 ($b=128, g=32$) verdoppelt die Breite des Flaschenhalses auf $128$ und benötigt $2(256)(128) + 9(128^2)/32 = 70{,}144$ Gewichte (+0.7%). Der Top-1-Fehler auf ImageNet sinkt von **23.9% auf 22.2%**.

<!--en:498b2b8d-->
::: {.slide title="Implementierung des ResNeXt-Blocks" except="jax,tensorflow"}

<!--en:9b7d005a-->
[ResNeXt-Code]{.kicker}

<!--en:3a085d5d-->
`ResNeXtBlock` verallgemeinert `Residual` und den Bottleneck-Block von ResNet-50 zu einem wiederverwendbaren Modul:

<!--en:ab635393-->
::: {.slide title="Implementierung des ResNeXt-Blocks: Faltungen" only="jax"}

<!--en:9b7d005a-2-->
[ResNeXt-Code]{.kicker}

<!--en:3a085d5d-2-->
`ResNeXtBlock` verallgemeinert `Residual` und den Bottleneck-Block von ResNet-50 zu einem wiederverwendbaren Modul:

<!--en:ff43aa48-->
::: {.slide title="Implementierung des ResNeXt-Blocks: Sprungverbindung und Vorwärtsdurchlauf" only="jax"}

<!--en:9bbd7673-->
::: {.slide title="Implementierung des ResNeXt-Blocks: Faltungen" only="tensorflow"}

<!--en:9b7d005a-3-->
[ResNeXt-Code]{.kicker}

<!--en:3a085d5d-3-->
`ResNeXtBlock` verallgemeinert `Residual` und den Bottleneck-Block von ResNet-50 zu einem wiederverwendbaren Modul:

<!--en:f2c5645c-->
::: {.slide title="Implementierung des ResNeXt-Blocks: Sprungverbindung und Vorwärtsdurchlauf" only="tensorflow"}

<!--en:d23096d6-->
::: {.slide title="DenseNet: Verkettung und quadratisch wachsender Aktivierungsspeicher"}

<!--en:f60f1971-->
[DenseNet und die Anordnung der Sprungverbindungen]{.kicker}

<!--en:8cd321fd-->
Anstelle der Summe $\mathbf{x}_k = \mathbf{x}_{k-1} + g_k(\mathbf{x}_{k-1})$ verwendet *DenseNet* (Huang et al., 2017) die Verkettung: Jede Schicht hängt ihre Ausgabekanäle entsprechend der *Wachstumsrate* $r$ an alle vorherigen Merkmalskarten an:

<!--en:7394b097-->
- **Explizite Wiederverwendung von Merkmalen:**
  $$\mathbf{x}_k = [\mathbf{x}_{k-1},\; f_k(\mathbf{x}_{k-1})]$$
  Nach $n$ Schichten mit anfangs $c$ Eingabekanälen hat der Tensor $c + n r$ Kanäle. Übergangsschichten ($1\times 1$-Faltung und $2\times 2$-Average-Pooling) verringern die Kanalzahl zwischen dichten Blöcken.

<!--en:074c3fd7-->
- **Warum sich additive Sprungverbindungen durchgesetzt haben (Speicherbedarf):** Ein naiv implementiertes DenseNet reserviert einen neuen Puffer der Breite $c + (i-1)r$ für die Verkettung in jeder Schicht $i=1,\ldots,n$ und hält ihn für die Backpropagation vor. Das erfordert $\sum_{i=1}^n (c + (i-1)r) = nc + \frac{n(n-1)}{2}r = \Theta(n^2 r)$ Aktivierungsspeicher (z. B. $10{,}368$ Kanäle gegenüber $3{,}072$ bei ResNet für $n=24, c=64, r=32$).

<!--en:939e954f-->
::: {.slide title="ResNets mit Voraktivierung und Skip-Verbindungen im U-Net"}

<!--en:274d47c8-->
[Varianten und Erweiterungen]{.kicker}

<!--en:80eceb16-->
- **ResNet mit Voraktivierung (He et al., 2016b):** Verwendet $\textrm{BN} \to \textrm{ReLU} \to \textrm{Conv}$ innerhalb von $g(\mathbf{x})$ und lässt die ReLU nach der Addition weg. Damit ist die Jacobi-Matrix entlang des direkten Pfads bei jeder Tiefe exakt $\mathbf{I}$.

<!--en:c258091b-->
- **Sprungverbindung in ResNet-D (He et al., 2019):** Ein $2 \times 2$-Average-Pooling mit Stride 2 vor einer $1 \times 1$-Faltung mit Stride 1 erfasst 100% der Pixel statt 25%.

<!--en:207e858a-->
::: {.slide title="Verständnisfragen: Sprungverbindungen, Rechenaufwand bei ResNeXt und Speicherbedarf bei DenseNet"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:475a3a2c-->
Führen Sie die folgenden drei Berechnungen zum Netzentwurf durch:

<!--en:2d7fb78f-->
1. **Problem bei Sprungverbindungen mit Stride 2 (ResNet-D):** Ein `Residual`-Block mit Stride 2 verarbeitet eine $24\times 24$-Merkmalskarte. Darin ist `conv3` eine $1\times 1$-Faltung mit Stride 2. Welchen Anteil der $24\times 24$ Eingabepositionen liest `conv3` tatsächlich, und wie behebt ResNet-D das Problem?

<!--en:25b2759a-->
2. **FLOPs der Blöcke von ResNet-50 und ResNeXt-50:** Prüfen Sie für $c = 512$, dass ein Bottleneck-Block von ResNet-50 ($b = 128, g = 1$) und ein ResNeXt-50-Block ($b = 256, g = 32$) nahezu dieselbe Parameteranzahl haben.

<!--en:24782c56-->
3. **Kanalwachstum bei DenseNet:** Ein dichter Block erhält $c = 64$ Kanäle und enthält $n = 4$ Schichten mit Wachstumsrate $r = 32$. Wie viele Ausgabekanäle entstehen, und wie viele Eingabekanäle lesen alle 4 Faltungen zusammen?

<!--en:557b98b6-->
**Antworten:**

<!--en:17dd4b9e-->
1. Ein $1\times 1$-Faltungskern mit Stride 2 liest nur Pixel mit geradem Zeilen- und Spaltenindex, also **$1/4$ (25%) aller Pixel**. Die übrigen $75\%$ verwirft er. ResNet-D verwendet zunächst eine $2\times 2$-Average-Pooling-Schicht mit Stride 2 und ohne Parameter. Darauf folgt eine $1\times 1$-Faltung mit Stride 1. So tragen alle Pixel bei.

<!--en:f2aecbad-->
2. **ResNet-50:** $2(512)(128) + 9(128^2) = 131{,}072 + 147{,}456 = 278{,}528$ Gewichte. **ResNeXt-50:** $2(512)(256) + 9(256^2)/32 = 262{,}144 + 18{,}432 = 280{,}576$ Gewichte ($<0.8\%$ Unterschied bei $2\times$ so breitem Flaschenhals).

<!--en:c55fdc02-->
3. Die Ausgabebreite beträgt $64 + 4(32) = 192$ Kanäle. Die 4 Schichten lesen insgesamt $64 + 96 + 128 + 160 = 448$ verkettete Eingabekanäle.

<!--en:3f4e4f9a-->
::: {.slide title="Residualnetze im Überblick"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:277a8607-->
- **ResNet ($\mathbf{x} + g(\mathbf{x})$):** Die Identität liegt bei $g=0$ ($\mathcal{F} \subseteq \mathcal{F}'$). Das behebt die Degradation tiefer unverzweigter Netze bei einem Aktivierungsspeicher von $O(n \cdot c)$.

<!--en:08978f77-->
- **ResNeXt ($g$ Gruppen):** Die Anzahl der $3 \times 3$-Gewichte sinkt auf $9b^2/g$. Bei gleichen FLOPs verdoppelt sich die Breite $b$ des Flaschenhalses.

<!--en:7fab2417-->
- **DenseNet ($[\mathbf{x}, f(\mathbf{x})]$):** Verkettet Merkmale mit Wachstumsrate $r$ und benötigt bei naiver Implementierung $O(n^2 r)$ Aktivierungsspeicher.
