<!--en:c3e528a8-->
# Moderne konvolutionäre Netze

<!--en:fcbebf97-->
Aus :numref:`chap_cnn` kennen wir die Funktionsweise von Faltungsschichten (engl. convolutional layers) und LeNet. Nun verfolgen wir ein Jahrzehnt der Architekturentwicklung bei konvolutionären neuronalen Netzen (engl. convolutional neural networks, CNNs). Es reicht vom Durchbruch von AlexNet auf ImageNet im Jahr 2012 bis zur wechselseitigen Beeinflussung von konvolutionären Netzen und Vision Transformern in den frühen 2020er-Jahren. Bei jedem wegweisenden Modell fragen wir, welcher Engpass den nächsten Entwicklungsschritt ausgelöst hat. So bauen wir im Laufe des Kapitels auf den Entwurfsmethoden der vorangegangenen Abschnitte auf.

<!--en:b5e7c447-->
Die ersten Abschnitte führen von einzeln von Hand verschalteten Schichten zu modularen Bausteinen. Am Beispiel von AlexNet (:numref:`sec_alexnet`) sehen wir, wie das End-to-End-Lernen von Merkmalen die von Hand entworfenen Verfahren zur Bildverarbeitung übertraf, sobald große Datensätze mit Labels und GPUs gemeinsam zur Verfügung standen. Hier lernen wir ReLU-Aktivierungen, Dropout und Datenaugmentierung kennen. Auch der Lernratenplan (engl. learning rate schedule) mit Aufwärmphase und anschließendem Kosinusverlauf kommt im Kapitel immer wieder zum Einsatz. Bei AlexNet fällt ein deutliches Ungleichgewicht auf. Die Faltungen erfordern den Großteil der Rechenarbeit, während die meisten Parameter im vollvernetzten Kopf stecken. VGG, Network in Network (NiN) und GoogLeNet (:numref:`sec_blocks`) begegnen diesem Ungleichgewicht mit verschiedenen Bausteinen. Sie gliedern Netze in wiederholbare $3 \times 3$-Blöcke, mischen die Kanäle an jedem Pixel mit $1 \times 1$-Faltungen und ersetzen große vollvernetzte Köpfe durch Global-Average-Pooling. In einer Architektur aus Eingangsteil, Hauptteil und Kopf kombinieren sie parallele Zweige für mehrere Größenskalen.

<!--en:4e79ff87-->
In den mittleren Abschnitten untersuchen wir, wie sich tiefere Netze bauen und die Hardware effizienter nutzen lassen. Normalisierungsschichten (:numref:`sec_batch_norm`) standardisieren die Aktivierungen der Zwischenschichten über den Minibatch hinweg oder innerhalb einzelner Beispiele. Dadurch werden versteckte Schichten unempfindlich gegenüber der Skalierung der Gewichte, und der Bereich nutzbarer Lernraten erweitert sich um Größenordnungen. Residualnetze (:numref:`sec_resnet`) lösen ein Optimierungsproblem, das bei der einfachen Verkettung von Schichten mit zunehmender Tiefe auftritt. Dazu verwenden sie additive Sprungverbindungen (engl. shortcuts) der Form $\mathbf{x} + g(\mathbf{x})$. Bei $g = 0$ bleibt die Identitätsabbildung erhalten. ResNeXt erweitert den Block durch gruppierte Faltungen, DenseNet durch die Verkettung von Merkmalen. Beim Entwurf effizienter konvolutionärer Netze (:numref:`sec_efficient_cnns`) zählen neben der Genauigkeit auch Latenz, Speicherbedarf und Energieverbrauch. Dazu betrachten wir tiefenweise separierbare Faltungen, invertierte Flaschenhälse, kombinierte Skalierung (engl. compound scaling), Squeeze-and-Excitation-Gates und strukturelle Reparametrisierung. Wir sehen, warum die Anzahl der Rechenoperationen oft wenig über die tatsächliche Laufzeit auf der GPU aussagt.

<!--en:e931cb7c-->
In den letzten Abschnitten untersuchen wir getrennt, welchen Einfluss Trainingsverfahren und Architektur haben. Moderne Trainingsrezepte (:numref:`sec_training_recipes`) zeigen, wie viel heutige Verfahren zur Optimierung und Regularisierung bewirken. Mit AdamW, einem Kosinusplan mit Aufwärmphase, Label-Smoothing, Mixup, CutMix, stochastischer Tiefe und der Mittelung der Gewichte steigt die Genauigkeit eines unveränderten ResNet auf ImageNet um mehrere Prozentpunkte. Bei ConvNeXt (:numref:`sec_convnext`) bleibt das Trainingsrezept fest. Die Architektur eines ResNet wird Schritt für Schritt modernisiert, bis sie ohne Attention die Genauigkeit hierarchischer Vision Transformer erreicht. Für verschiedene Entwurfsräume konvolutionärer Netze (:numref:`sec_cnn-design`) ziehen wir Stichproben von Architekturen und vergleichen ihre empirischen Fehlerverteilungen. Daraus leiten wir die einfachen Strukturregeln der RegNet-Familie ab.

<!--en:e0422d3d-->
Moderne konvolutionäre Netze<br>**AlexNet · Blöcke · Normalisierung · Sprungverbindungen · effiziente konvolutionäre Netze · Trainingsrezepte · ConvNeXt · RegNet**

<!--en:c9a3fb38-->
::: {.slide title="Warum zusätzliche LeNet-Schichten in der Praxis scheitern"}

<!--en:acc8203e-->
[Das zentrale Problem]{.kicker}

<!--en:8e2134cc-->
- **Schwächere Signale und schlechtere Optimierung:** Sigmoidfunktionen verkleinern Gradienten um einen Faktor von $\le 1/4^L$. Selbst mit ReLU und BN erzielt ein Netz ohne Sprungverbindungen mit 56 Schichten einen höheren Trainingsverlust als eines mit 20 ($\approx 0.28$ gegenüber $0.20$).

<!--en:42c9fb0c-->
- **92% im vollvernetzten Kopf und FLOPs im Vergleich zur Latenz:** Bei AlexNet und VGG liegen $>92\%$ der Gewichte im Kopf. Tiefenweise Faltungen benötigen $8\times$ weniger MACs, stoßen aber an die Grenzen der GPU-Speicherbandbreite.

<!--en:8dc95df8-->
- **Vermischte Effekte von Architektur und Trainingsrezept:** Allein ein modernes Trainingsrezept (LAMB, BCE-Verlust, RandAugment/Mixup/CutMix, 600 Epochen) bringt einem unveränderten ResNet-50 $4.3$ Prozentpunkte mehr Genauigkeit ($76.1\% \to 80.4\%$).

<!--en:38b564de-->
::: {.slide title="Bausteine für den Architekturentwurf"}

<!--en:99e0030a-->
[Kapitelüberblick]{.kicker}

<!--en:a7bdbd8d-->
- **§§{sec_alexnet}–§{num:sec_blocks} (AlexNet, VGG, NiN, GoogLeNet):** ReLU, Dropout, $3 \times 3$-Blöcke, $1 \times 1$-Flaschenhälse und Global-Average-Pooling in einer Architektur aus Eingangsteil, Hauptteil und Kopf.

<!--en:c5d48b77-->
- **§§{sec_batch_norm}–§{num:sec_resnet} (Normalisierung und Residualblöcke):** BatchNorm, LayerNorm, GroupNorm, additive Sprungverbindungen $\mathbf{x} + g(\mathbf{x})$, gruppierte ResNeXt-Blöcke und die Verkettung von Merkmalen in DenseNet.

<!--en:136cab62-->
- **§§{sec_efficient_cnns}–§{num:sec_cnn-design} (Effizienz, Trainingsrezepte, ConvNeXt, RegNet):** tiefenweise Faktorisierung, Fusion der RepVGG-Zweige, AdamW/Mixup/EMA, $7 \times 7$-Blöcke in ConvNeXt und kumulative Verteilungsfunktionen der Fehler je Modellpopulation.

<!--en:a2a396a1-->
::: {.slide title="Drei gemeinsame Entwurfsprinzipien"}

<!--en:acd261d6-->
[Grundlegende Denkmodelle]{.kicker}

<!--en:c98fd784-->
- **1. Rezeptives Feld pro FLOP und GAP:** Gestapelte $3 \times 3$-Faltungen ergeben ein rezeptives Feld von $5 \times 5$ ($18c^2$ statt $25c^2$) oder $7 \times 7$ ($27c^2$ statt $49c^2$) mit mehr Nichtlinearitäten. Mit GAP entfallen Köpfe mit 43 Mio. Parametern.

<!--en:b524125d-->
- **2. Skaleninvarianz und geschachtelte Funktionsklassen:** Es gilt $\textrm{BN}(a\mathbf{w}^\top\mathbf{h}) = \textrm{BN}(\mathbf{w}^\top\mathbf{h})$. Bei $\mathbf{x} + g(\mathbf{x})$ bleibt für $g=0$ die Identitätsabbildung erhalten ($\mathcal{F} \subseteq \mathcal{F}'$).

<!--en:7d7f0815-->
- **3. Faktorisierung und Rechenintensität:** Tiefenweise separierbare Faltungen senken die Anzahl der MACs auf den Anteil $1/c_\textrm{o} + 1/k^2 \approx 1/9$, erreichen aber nur $9$ statt $9c_\textrm{o}$ MACs pro Lesezugriff.

<!--en:e47006f5-->
::: {.slide title="Durchgängiger Vergleich auf Fashion-MNIST unter kontrollierten Bedingungen"}

<!--en:343fa0a9-->
[Empirischer Vergleich]{.kicker}

<!--en:e8dd206b-->
- **AlexNet $\to$ Quarter-VGG / ResNet-18:** Die Genauigkeit steigt von $91\%$ auf $93\text{--}94\%$. Die Parameteranzahl sinkt von $46.8\textrm{M}$ auf $11.2\textrm{M}$ (mit GAP auf $0.6\textrm{M}$).

<!--en:9616121d-->
- **BNLeNet und ResNet-56:** BatchNorm erweitert den nutzbaren Lernratenbereich um den Faktor $300\times$ ($\eta_0 \in [0.03, 10]$). Sprungverbindungen kehren die Verschlechterung bei 56 Schichten um.

<!--en:684beff0-->
- **ConvNeXt und RegNetX32:** Beide erreichen die Genauigkeit von ResNet-18 ($\approx 94\%$) mit $3.38\textrm{M}$ bzw. $0.52\textrm{M}$ Parametern (bei RegNetX32 also $21\times$ weniger).
