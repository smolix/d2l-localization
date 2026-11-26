<!--en:7fa7f22f-->
# Vision Transformer (ViT)

<!--en:c6fa6769-->
Konvolutionäre neuronale Netze (engl. convolutional neural network, CNN) (:numref:`chap_cnn`, :numref:`chap_modern_cnn`) verankern zwei geometrische Annahmen über Bilder direkt in ihrer Architektur. Die erste ist **räumliche Lokalität** (engl. spatial locality): Ein Faltungskern der Größe $k \times k$ verknüpft nur benachbarte Pixel. Die zweite ist **Translationsäquivarianz** (engl. translation equivariance): An jeder Position $(r, c)$ kommen dieselben Kerngewichte zum Einsatz. Ein herkömmlicher bidirektionaler Transformer-Encoder (:numref:`sec_transformer`) hat *keinen* eingebauten Begriff von Nachbarschaft im 2D-Gitter oder von Translationsäquivarianz. Kann er Bilder ähnlich gut klassifizieren?

<!--en:53060e18-->
Wenden wir Self-Attention direkt auf einzelne Pixel an, wächst der Aufwand quadratisch mit der Pixelanzahl. Schon ein Bild der Größe $96 \times 96$ hat $9{,}216$ Pixel ($9216^2 \approx 8.5 \times 10^7$ Attention-Paare pro Kopf). :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` führten den **Vision Transformer (ViT)** (:numref:`fig_vit`) ein. Er zerlegt das Bild in ein Gitter aus nicht überlappenden **Patches** der Größe $p \times p$. Aus einem Bild der Größe $96 \times 96$ wird bei einer Patch-Größe von $p = 16$ eine kurze Sequenz von $m = (96 / 16)^2 = 36$ Patch-Token. Das Modell projiziert jeden Patch auf die Breite $d$, stellt ein lernbares `<cls>`-Token voran und addiert gelernte eindimensionale Positionseinbettungen (engl. position embedding). Die 37 Token durchlaufen $L$ bidirektionale Transformer-Blöcke mit vorgeschalteter Normalisierung.

<!--en:87b0b119-->
## Patches als Token (`PatchEmbedding`)

<!--en:a07ebd71-->
Wir teilen einen Bildtensor der Form $(B, C, H, W)$ in nicht überlappende Patches der Größe $p \times p$ auf. Auf jeden Patch wenden wir dieselbe lineare Projektion $\mathbb{R}^{C p^2} \to \mathbb{R}^d$ an. Das entspricht einer **2D-Faltung mit Kerngröße $p \times p$, Stride $p$ und $d$ Ausgabekanälen** (engl. convolution). Anschließend formen wir das räumliche Gitter der Höhe $H/p$ und Breite $W/p$ zu einer Sequenz von $m = (H/p)(W/p)$ Tokenvektoren um:

<!--en:3ac60a92-->
Für einen Minibatch aus vier einkanaligen Bildern der Größe $96 \times 96$ mit Patch-Größe $p = 16$ liefert `PatchEmbedding` pro Bild $(96 / 16)^2 = 36$ Patch-Token der Breite $d = 512$:

<!--en:14fec1c0-->
## Architektur des Vision Transformers

<!--en:4bc4497f-->
![Der Vision Transformer für ein Bild der Größe $96 \times 96$. Eine Faltung mit Stride wandelt das $6 \times 6$-Gitter aus Patches der Größe $16 \times 16$ in 36 Token um. Das Modell stellt ein gelerntes „&lt;cls&gt;“-Token voran und addiert gelernte Positionseinbettungen. Anschließend verarbeiten $L$ Transformer-Blöcke mit vorgeschalteter Normalisierung alle 37 Token. Ein linearer Kopf klassifiziert die letzte Repräsentation des „&lt;cls&gt;“-Tokens.](../img/mdl-transformers-vit.svg)

<!--en:c5faa4f9-->
### Der Encoder-Block (`ViTBlock`)

<!--en:a7eda102-->
Nach dem Vorbild von :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` leiten wir `ViTBlock` von unserem konfigurierbaren `d2l.TransformerBlock` (:numref:`sec_transformer-block`) ab. Wir verwenden eine vorgeschaltete `LayerNorm` und ein mehrschichtiges Perzeptron (MLP) mit zwei Schichten und `GELU`-Aktivierung. Die Breite der verborgenen Schicht legen wir mit `mlp_num_hiddens` fest. Die Attention-Projektionen haben standardmäßig keinen Bias (`use_bias=False`). Die beiden linearen Schichten des MLP und die `LayerNorm` behalten ihren Bias:

<!--en:147edaf3-->
### Das `<cls>`-Token, die Positionseinbettungen und `ViT`

<!--en:47ddf104-->
Die $m$ Patch-Repräsentationen sollen einen einzigen Vektor für das gesamte Bild ergeben, ohne eine Position zu bevorzugen. Dazu stellt `ViT` einen lernbaren Parametervektor `cls_token` der Form $(1, 1, d)$ an den Index $0$. Für die Positionen addiert das Modell eine lernbare Einbettungstabelle `pos_embedding` der Form $(1, m + 1, d)$, die alle $m + 1$ Positionen abdeckt. Es durchläuft $L$ Schichten vom Typ `ViTBlock` und normalisiert die Ausgabe mit `LayerNorm`. Den Vektor am Index $0$ (`X[:, 0]`) übergibt es einem linearen Klassifikationskopf:

<!--en:f22fafe5-->
## Training auf Fashion-MNIST

<!--en:4f512446-->
### Ein erster Trainingslauf

<!--en:ff8ef1e7-->
Wir trainieren einen `ViT` mit 2 Blöcken ($d = 512, d_{\textrm{mlp}} = 2048, H = 8, L = 2, p = 16$, $\approx 6.5\textrm{M}$ Parameter) auf Bildern der Größe $96 \times 96$ aus Fashion-MNIST. Für die 10 Epochen verwenden wir AdamW ($\lambda = 0.05$) mit linearer Aufwärmphase und anschließendem Kosinus-Abfall der Lernrate:

<!--en:865738b7-->
Vor dem Training erzeugen wir mit `model = vit()` eine Modellinstanz und speichern eine Kopie der zufällig initialisierten Positionseinbettungen der Patches in `pos_init`. So können wir später messen, wie viel räumliche Struktur durch das Training hinzukommt:

<!--en:c6cefe6e-->
Trainieren wir `model` für 10 Epochen mit `d2l.Trainer`, erreichen wir rund $89\%$ Validierungsgenauigkeit auf den auf $96 \times 96$ hochskalierten Bildern aus Fashion-MNIST:

<!--en:952c0308-->
### Die Rolle der räumlichen Position und die erlernte Topologie des 2D-Gitters

<!--en:6073c5ae-->
Ohne `pos_embedding` ist die Ausgabe von `ViT` **permutationsinvariant** bezüglich der $m = 36$ Patches. Jede Permutation der 36 Patches ordnet die Zeilen und Spalten $1 \dots 36$ jeder Attention-Matrix entsprechend um. In Zeile $0$ (`<cls>`) ändern sich dadurch zwar die Positionen der Gewichte, der resultierende Ausgabevektor bleibt aber exakt gleich. Mit gelerntem `pos_embedding` erhält Patch $k$ den Positionsvektor $\mathbf{p}_k$ des Platzes $k$, den er einnimmt. Wie stark stützt sich der trainierte `ViT` auf die räumliche Anordnung? Beim Test ordnen wir die 36 Patches jedes Validierungsbildes nach 5 zufälligen Permutationen um (`shuffled_accuracy`):

<!--en:7f33e8c4-->
Wir werten `shuffled_accuracy` für fünf feste Zufallspermutationen der $36$ Patches aus. So messen wir, wie stark die Genauigkeit sinkt, wenn wir beim Test die räumliche Anordnung verändern:

<!--en:3d7fc488-->
Mischen wir die Patches im $6 \times 6$-Gitter, sinkt die Validierungsgenauigkeit je nach Permutation von etwa 0.89 auf rund 0.25–0.45. Für den trainierten `ViT` ist also entscheidend, wo die einzelnen Patches liegen. Die Genauigkeit bleibt dennoch weit über dem Zufallsniveau von 0.1. Schon lokale Muster in den $16 \times 16$ Pixel großen Patches, etwa Schuhabsätze oder Kragenformen, lassen viele grobe Kleidungsklassen erkennen, selbst wenn das Bild nur noch eine ungeordnete Sammlung von Patches ist.

<!--en:72056fdd-->
Lernt die unstrukturierte 1D-Positionstabelle `pos_embedding` ($36 \times d$) tatsächlich die 2D-Geometrie des $6 \times 6$-Bildgitters? Dazu berechnen wir die mittlere Kosinusähnlichkeit der Positionsvektoren von Patch-Paaren als Funktion ihres Gitterabstands $\max(|r - r'|, |c - c'|)$ im $6 \times 6$-Gitter (`grid_similarity`). Als Vergleich berechnen wir dieselbe Kurve für eine sinusförmige 2D-Kodierung, die das Gitter konstruktionsbedingt abbildet, und für Zufallsvektoren, denen diese Struktur fehlt:

<!--en:8c504088-->
Nach zehn Epochen ist die Gitterstruktur kaum zu erkennen. Nur benachbarte Patches (Abstand 1) sind einander etwas ähnlicher als Zufallsvektoren. Ihre mittlere Kosinusähnlichkeit liegt unter 0.1, bei Zufallsvektoren beträgt sie etwa $-0.03$. Bei größeren Abständen verläuft die Kurve der gelernten Vektoren flach nahe null. Erst bei Abstand 5 wird der Beitrag des Trainings gegenüber dem anfänglichen Rauschen wieder positiv. Dann liegen die beiden Patches an gegenüberliegenden Bildrändern. Bei der sinusförmigen 2D-Kodierung fällt die Ähnlichkeit dagegen von etwa $0.75$ bei Abstand 1 auf deutlich unter null. Die klare 2D-Struktur, die :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` in den Positionseinbettungen des ViT finden, entsteht durch wesentlich längeres Training auf wesentlich größeren Datensätzen.

<!--en:03bd98aa-->
## Vergleich mit einem konvolutionären Netz gleicher Parameteranzahl

<!--en:ce0809f3-->
Wie schneidet unser `ViT` mit 6.5 Millionen Parametern auf diesem Datensatz aus 60 000 Bildern ab? Als Vergleich dient ein konvolutionäres Netz nach dem Vorbild von ResNet mit gleicher Parameteranzahl (`CompactResNet`, $\approx 6.5\textrm{M}$ Parameter). Wir trainieren es ebenfalls 10 Epochen mit AdamW und genau demselben Kosinusplan.

<!--en:dc9fe370-->
Wir definieren die Fabrikfunktion `compact_cnn` und erzeugen damit eine Instanz `cnn` mit der maximalen Lernrate $10^{-3}$:

<!--en:5a68eccd-->
Wir trainieren `CompactResNet` für 10 Epochen. Um zu prüfen, wie stabil die Ergebnisse zwischen Läufen sind, trainieren wir sowohl `CompactResNet` als auch `ViT` mit einem zweiten Zufallsstartwert. So vergleichen wir beide Architekturen bei demselben Budget von $6.5\textrm{M}$ Parametern:

<!--en:bbbe1ed4-->
Tragen wir die Validierungsgenauigkeit von `ViT` und `CompactResNet` über alle 10 Epochen im selben Diagramm auf, sehen wir den Unterschied in der Dateneffizienz während des gesamten Trainings:

<!--en:72ac6143-->
Auf einem kleinen Datensatz mit 60 000 Bildern, ohne Datenaugmentierung oder umfangreiches Pre-Training, konvergiert `CompactResNet` schneller und liegt am Ende rund 5 Prozentpunkte vor `ViT`. Das veranschaulicht den grundlegenden Zielkonflikt, den :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` beschrieben haben. Die fest eingebaute 2D-Lokalität und Translationsäquivarianz geben einem CNN bei wenigen Daten einen starken **induktiven Bias**. Ein Vision Transformer lernt räumliche Beziehungen dagegen flexibel aus den Daten. Auf kleinen Datensätzen ohne Augmentierung bleibt er hinter CNNs zurück. Nach einem Pre-Training auf mehreren zehn Millionen Bildern erreicht oder übertrifft er sie. Das gelingt auch mit starker Augmentierung, mit Destillation :cite:`touvron2021training` oder mit der Rekonstruktion maskierter Patches :cite:`he2022masked`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:a0fe5dc4-->
**Vision Transformer (ViT)** wandeln ein Bild der Form $(C, H, W)$ in eine Sequenz aus $m = (H/p)(W/p)$ Token um. Dazu verwenden sie eine Faltung mit Stride $p$ (`PatchEmbedding`). Sie stellen ein `<cls>`-Token voran, addieren gelernte 1D-Positionseinbettungen und wenden einen bidirektionalen Transformer-Encoder mit vorgeschalteter Normalisierung an (:numref:`fig_vit`). Ohne Positionseinbettungen ist die Ausgabe eines ViT streng permutationsinvariant bezüglich seiner Patches. Mit gelernten 1D-Positionseinbettungen hängt sie stark von der Patch-Anordnung ab, obwohl sich das 2D-Gitter nach zehn Epochen auf Fashion-MNIST nur schwach in der Positionstabelle abzeichnet. Dem ViT fehlen die eingebaute 2D-Lokalität und die Translationsäquivarianz der Faltung. Beim Training von Grund auf mit kleinen Datensätzen ohne Augmentierung schneidet ein ResNet mit gleicher Parameteranzahl deshalb besser ab. Beim Pre-Training mit großen Datenmengen überholt der ViT die CNNs. Zudem bietet er eine einheitliche Tokenschnittstelle für multimodale Modelle.

<!--en:2294243a-->
## Übungen

<!--en:189817e5-->
1. **Patch-Größe $p$, Parameteranzahl und FLOPs der Attention.** Für Bilder der Größe $96 \times 96$ verwendet unser `ViT` die Patch-Größe $p = 16$ ($m = 36$ Patches).
   1. Wie ändern sich die Parameteranzahl von `PatchEmbedding`, die von `pos_embedding` und die der $L$ `ViTBlock`-Schichten, wenn Sie die Patch-Größe auf $p = 8$ halbieren ($m = 144$) oder auf $p = 4$ vierteln ($m = 576$)?
   2. Um welchen Faktor wächst die Größe $(m + 1)^2$ der Matrix der Attention-Scores pro Schicht beim Übergang von $p = 16$ auf $p = 8$ und auf $p = 4$? Trainieren Sie für $p = 8$ den `ViT` über 10 Epochen. Vergleichen Sie die tatsächlich verstrichene Trainingszeit und die Validierungsgenauigkeit mit den Werten für $p = 16$.
   :exlabel:`ex_vision-transformer-parameters-compute`

<!--en:a373a571-->
2. **Auslesen des `<cls>`-Tokens und Global-Average-Pooling im Vergleich.** Ersetzen Sie das Auslesen des `<cls>`-Tokens in `ViT` durch Global-Average-Pooling über die $m = 36$ Patch-Token (`self.head(X[:, 1:].mean(1))`, oder entfernen Sie `cls_token` ganz und mitteln Sie über alle 36 Token). Trainieren Sie das Modell 10 Epochen auf Fashion-MNIST. Vergleichen Sie die Validierungsgenauigkeit mit der beim Auslesen des `<cls>`-Tokens.

<!--en:d14dd4ce-->
3. **Exakte Permutationsinvarianz ohne Positionseinbettungen.** Setzen Sie alle Einträge von `model.pos_embedding` auf null und schalten Sie `model` in den Auswertungsmodus (damit ist Dropout deaktiviert). Beweisen Sie die folgende Aussage algebraisch und überprüfen Sie sie numerisch an einem Validierungsbatch mit einer Toleranz von $10^{-6}$. Die Anwendung von `shuffle_patches(X, perm)` mit einer beliebigen Permutation `perm` von $\{0, \dots, 35\}$ lässt die Ausgabe-Logits `model(X)` exakt unverändert.
   :exlabel:`ex_vision-transformer-position-information`

<!--en:dd9df7c7-->
4. **Statistischer Permutationstest zum Nachweis des 2D-Gitters.** Sei $\Delta$ die Differenz der Werte von `grid_similarity` für benachbarte Patches (Gitterabstand 1) und für Patches mit Gitterabstand $\ge 4$, jeweils für die gelernte Tabelle `model.pos_embedding`. Berechnen Sie die Nullverteilung von $\Delta$ anhand von 1000 zufälligen Permutationen der 36 Zeilen von `pos_embedding` und geben Sie den empirischen $p$-Wert an. Lässt sich die schwache Gitterstruktur der gelernten Positionseinbettungen vom Zufall unterscheiden?
   :exlabel:`ex_vision-transformer-grid-test`

<!--en:676e510d-->
5. **Empfindlichkeit gegenüber Verschiebungen um ganze und halbe Patch-Breiten.** Verschieben wir das Eingabebild eines CNN um $s$ Pixel, verschieben sich auch die internen Merkmalskarten um $s$ Pixel (bis auf den Stride beim Pooling). In `ViT` ($p = 16$) verschiebt sich bei einer Verschiebung des Bildes um $s = 16$ Pixel (eine ganze Patch-Breite) jeder Patch nur auf den Nachbarplatz. Eine Verschiebung um $s = 8$ Pixel (einen halben Patch) schneidet dagegen durch alle Patches der Größe $16 \times 16$. Bestimmen Sie die Validierungsgenauigkeit unserer trainierten Modelle `ViT` und `CompactResNet` auf Validierungsbildern, die Sie um $s \in \{0, 2, 4, 8, 12, 16\}$ Pixel nach rechts verschieben. Füllen Sie den linken Rand mit Nullen auf. Warum fällt die Genauigkeit von `ViT` bei $s = 8$ ab und steigt bei $s = 16$ wieder an?

<!--en:2f521364-->
6. **Mittlerer Attention-Abstand in Abhängigkeit von der Tiefe.** :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` definieren den **mittleren Attention-Abstand** eines Kopfes als den mit Attention-Gewichten gemittelten euklidischen 2D-Abstand in Pixeln $\sum_{i,j=1}^{36} \alpha_{ij} \, d_{\textrm{pixel}}(i, j) / 36$ zwischen Abfrage-Patch $i$ und Schlüssel-Patch $j$. Berechnen Sie den mittleren Attention-Abstand für jeden der 8 Köpfe in Block 1 und Block 2 unseres trainierten `ViT`. Konzentrieren sich die Köpfe in frühen Schichten stärker auf ihre lokale Umgebung als die in tieferen Schichten, analog zum wachsenden rezeptiven Feld eines CNN?

<!--en:28963ddf-->
[Diskussion](https://d2l.discourse.group/t/36778)

<!--en:8981109a-->
Transformer für die Bildverarbeitung (ViT)<br>**Patches als Token · Faltung mit Stride am Eingang · `<cls>` und Positionseinbettungen · Analyse des 2D-Gitters · Vergleich mit einem gleich großen CNN**

<!--en:108e2d3e-->
::: {.slide title="Warum der Aufwand für Self-Attention auf Pixelebene quadratisch wächst"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:e72d1820-->
- **Pixel als Token ($96 \times 96$):** $m = 9{,}216$ Token $\to$ $m^2 = 84.9\text{M}$ Attention-Paare pro Kopf und Schicht ($2.52\text{B}$ Paare bei ImageNet-Bildern der Größe $224 \times 224$).

<!--en:c5de31ae-->
- **Patch-Token der Größe $16 \times 16$ (`ViT`):** Nicht überlappende Patches der Größe $p \times p$ verkürzen die Sequenz um den Faktor $p^2$ ($256\times$). Die Anzahl der Attention-Scores sinkt um den Faktor $p^4$ (**$65{,}536\times$ weniger Paare**).

<!--en:89db0573-->
::: {.slide title="Induktiver Bias bei CNNs und ViTs: Lokalität oder globaler Austausch zwischen Patches"}

<!--en:2cd89814-->
[Zielkonflikt beim Architekturentwurf]{.kicker}

<!--en:6441d104-->
- **Konvolutionäre Netze (CNNs):** 2D-Lokalität (gleitende Filter der Größe $k \times k$) und Translationsäquivarianz sind fest eingebaut. Das rezeptive Feld wächst von Schicht zu Schicht.

<!--en:3bf3359f-->
- **Vision Transformer (ViTs):** Schon ab Schicht 1 umfasst das rezeptive Feld alle Patch-Paare (Attention der Größe $m \times m$). Ohne Positionseinbettungen fehlt jede eingebaute Annahme über das 2D-Gitter.

<!--en:5799703e-->
::: {.slide title="Aufbau des Vision Transformers (ViT)"}

<!--en:c12b0ef7-->
[Architektur im Überblick]{.kicker}

<!--en:6f578e53-->
- **1. Patch-Eingangsteil (`PatchEmbedding`):** Aus einem Bild der Größe $96 \times 96$ mit Patch-Größe $p = 16$ entsteht ein $6 \times 6$-Gitter aus $m = 36$ Patch-Vektoren im Raum $\mathbb{R}^d$.

<!--en:724be03b-->
- **2. `<cls>`-Token und Positionstabelle:** 1 lernbarer `<cls>`-Vektor kommt an den Anfang. An allen $m + 1 = 37$ Positionen werden gelernte 1D-Positionseinbettungen addiert.

<!--en:eed762ac-->
- **3. Bidirektionaler Encoder und Kopf:** Die 37 Token durchlaufen $L$ unmaskierte `ViTBlock`-Schichten mit vorgeschalteter Normalisierung. Der Kopf klassifiziert den letzten `<cls>`-Vektor `X[:, 0]`.

<!--en:36eb8456-->
::: {.slide title="Patch-Zerlegung und Projektion in einer Faltung mit Stride"}

<!--en:621fc398-->
[Eingangsteil für die Patch-Einbettung]{.kicker}

<!--en:61dc082b-->
- Ein Bild der Form $(H, W, C)$ in $m = (H/p)(W/p)$ nicht überlappende Patches der Größe $p \times p$ zerlegen und jeden Patch mit $\mathbf{W}_E \in \mathbb{R}^{C p^2 \times d}$ multiplizieren: Das entspricht einer **2D-Faltung mit `kernel_size=p, stride=p`**.

<!--en:fd210403-->
- Das Umformen der Faltungsausgabe von $(B, d, H/p, W/p)$ in $(B, m, d)$ liefert die Tokensequenz in einem einzigen fusionierten GPU-Kernel.

<!--en:d081e713-->
::: {.slide title="Implementierung und Prüfung von PatchEmbedding"}

<!--en:65ee4e26-->
[Implementierung des Eingangsteils]{.kicker}

<!--en:64cf8649-->
`PatchEmbedding` prüft `96 % 16 == 0` und bildet vier Bilder der Größe $96 \times 96$ auf die Form `(4, 36, 512)` ab:

<!--en:bee85e49-->
::: {.slide title="Prüfung von PatchEmbedding an vier Bildern"}

<!--en:65ee4e26-2-->
[Implementierung des Eingangsteils]{.kicker}

<!--en:319f2d20-->
::: {.slide title="Der ViT-Encoder-Block (ViTBlock)"}

<!--en:96ecf918-->
[Encoder-Block]{.kicker}

<!--en:afff07f5-->
Nach Dosovitskiy et al. (2021) konfiguriert `ViTBlock` unseren `d2l.TransformerBlock` (§{sec_transformer-block}) mit vorgeschalteter `LayerNorm`, unmaskierter Multi-Head-Self-Attention und einem zweischichtigen MLP mit exakter `GELU`-Aktivierung:

<!--en:ea95b201-->
Warum verwendet ViT `LayerNorm` und `GELU` anstelle von `RMSNorm` und `SwiGLU`? Damit folgt das Modell der ViT-Originalarbeit von 2020. Moderne Grundnetze für die Bildverarbeitung wie SigLIP 2, DINOv2 und AIMv2 verwenden wie LLMs auch `RMSNorm`, `SwiGLU` und zweidimensionales `RoPE`.

<!--en:a1032097-->
::: {.slide title="Das Auslesen des `<cls>`-Tokens und der Vorwärtsdurchlauf im ViT" only="pytorch"}

<!--en:90f2d159-->
[Auslesen und Vorwärtsdurchlauf]{.kicker}

<!--en:f3140bb0-->
Um $m = 36$ Patch-Token ohne Bevorzugung einer Position zu einer Bildrepräsentation zusammenzufassen, stellt `ViT` den Vektor `self.cls_token` am Index $0$ voran. Den Vektor `X[:, 0]` liest das Modell nach $L$ Blöcken aus:

<!--en:49827e40-->
Global-Average-Pooling über die $m$ Patch-Ausgaben (`X[:, 1:].mean(dim=1)`) funktioniert in der Praxis ähnlich gut. Das `<cls>`-Token ist eine einfache, von BERT übernommene Konvention.

<!--en:3c54726a-->
::: {.slide title="Das Auslesen des `<cls>`-Tokens und der Vorwärtsdurchlauf im ViT" only="jax"}

<!--en:90f2d159-2-->
[Auslesen und Vorwärtsdurchlauf]{.kicker}

<!--en:f3140bb0-2-->
Um $m = 36$ Patch-Token ohne Bevorzugung einer Position zu einer Bildrepräsentation zusammenzufassen, stellt `ViT` den Vektor `self.cls_token` am Index $0$ voran. Den Vektor `X[:, 0]` liest das Modell nach $L$ Blöcken aus:

<!--en:29290362-->
Global-Average-Pooling über die $m$ Patch-Ausgaben (`X[:, 1:].mean(axis=1)`) funktioniert in der Praxis ähnlich gut. Das `<cls>`-Token ist eine einfache, von BERT übernommene Konvention.

<!--en:c9a0f98c-->
::: {.slide title="Kodierung der 2D-Position in einer 1D-Sequenz"}

<!--en:2f80ed13-->
[Positionskodierungen für Gitter]{.kicker}

<!--en:9df0869b-->
- **Gelernte 1D-Tabelle (ursprünglicher ViT):** Ein freier Vektor $\mathbf{p}_k \in \mathbb{R}^d$ für jeden Rasterindex $k \in \{0, \dots, m\}$. Die Topologie des 2D-Gitters muss das Modell aus den Daten lernen.

<!--en:d64d0b17-->
- **Sinusförmige 2D-Kodierung:** Verkettet die $d/2$-dimensionale Zeilenkodierung $\mathbf{s}_r$ mit der $d/2$-dimensionalen Spaltenkodierung $\mathbf{s}_c$.

<!--en:e0268631-->
- **Axiales 2D-RoPE:** Rotiert die Hälfte der Kanäle von Abfragen und Schlüsseln gemäß Zeile $r$, die andere Hälfte gemäß Spalte $c$ (in DINOv2 und SigLIP 2).

<!--en:b93b7db6-->
::: {.slide title="Wahl des Optimierers: Warum ViTs AdamW benötigen"}

<!--en:821003f6-->
[Optimierungsverfahren]{.kicker}

<!--en:8e1f839b-->
- ResNets lassen sich mit SGD und Momentum zuverlässig trainieren. ViTs schneiden mit SGD deutlich schlechter ab, weil die Gradienten der Attention-Projektionen ($\mathbf{W}_q, \mathbf{W}_k$) und der MLPs sehr unterschiedliche Größenordnungen haben.

<!--en:c6a60229-->
- **Trainingsrezept für ViT:** AdamW ($\eta_{\max} = 3 \times 10^{-4}$), selektiver Gewichtszerfall ($\lambda = 0.05$ nur für 2D- und 4D-Gewichte), lineare Aufwärmphase über 1 Epoche und anschließend Kosinus-Abfall.

<!--en:7086b512-->
::: {.slide title="Training unseres ViT mit 6.5 Millionen Parametern auf Fashion-MNIST"}

<!--en:14def771-->
[Erster Trainingslauf]{.kicker}

<!--en:1e6ec4fe-->
- **Modellkonfiguration:** Bilder der Größe $96 \times 96$, Patch-Größe $p = 16$ ($m = 36$ Patches), Breite $d = 512$, MLP-Breite $d_{\text{mlp}} = 2{,}048$, $H = 8$ Köpfe, $L = 2$ Blöcke ($\mathbf{6.46\text{M}}$ Parameter).

<!--en:f90e7eca-->
- **Verlauf über 10 Epochen:** Die Validierungsgenauigkeit steigt gleichmäßig auf **$\approx 89\%$** (je nach Zufallsstartwert $0.888\text{--}0.892$). Das Training umfasst $10$ Epochen mit Minibatches aus je $128$ Bildern.

<!--en:d249e1a4-->
::: {.slide title="Wie wichtig ist die Patch-Anordnung für den trainierten ViT?"}

<!--en:9ba06f49-->
[Test mit räumlichen Permutationen]{.kicker}

<!--en:45fd7828-->
Wie stark hängt der trainierte `ViT` von der räumlichen Anordnung im 2D-Gitter ab? Dazu die $36$ Patches jedes Validierungsbildes beim Test nach fünf Zufallspermutationen umordnen (`shuffle_patches`):

<!--en:dd06c2da-->
- **Starke Abhängigkeit von der Anordnung ($89\% \to 25\%\text{--}45\%$):** Eine zufällige Anordnung der Patches senkt die Genauigkeit um rund **45–65 Prozentpunkte**. Der Klassifikator stützt sich also stark auf die Patch-Positionen.

<!--en:af71198f-->
- **Genauigkeit bei ungeordneten Patches ($\gg 10\%$ beim Raten):** Selbst mit gemischten Patches liegt die Genauigkeit um den Faktor $2.5\text{--}4.5\times$ über dem Zufallsniveau von $10\%$. Lokale Muster der Größe $16 \times 16$ (Stiefelsohlen, Kragenausschnitte) kennzeichnen auch ohne globale Geometrie mehrere Kleidungsklassen.

<!--en:c112abed-->
::: {.slide title="Bilden gelernte Positionseinbettungen das 2D-Gitter ab?"}

<!--en:175f74a3-->
[Geometrie der Positionsvektoren]{.kicker}

<!--en:8fe65723-->
- **Kosinusähnlichkeit nach Chebyshev-Abstand $k = \max(|r - r'|, |c - c'|)$ im Gitter:** Als Vergleich für die gelernte Positionstabelle der Größe $36 \times 512$ dienen ein fest vorgegebenes sinusförmiges 2D-Gitter und Zufallsvektoren.

<!--en:731a05c5-->
- **Schwache lokale Struktur nach 10 Epochen:** Nur benachbarte Patches ($k = 1$) zeigen eine erhöhte Ähnlichkeit ($+0.05$ gegenüber $-0.03$ bei Zufallsrauschen). Bei der sinusförmigen 2D-Kodierung fällt die Ähnlichkeit deutlich von $+0.76$ auf $-0.46$.

<!--en:a39b3e9f-->
- **Warum?** Um die gesamte Topologie des 2D-Gitters aus 1D-Rasterindizes zu lernen, braucht das Modell große Datensätze und Hunderte von Epochen (Dosovitskiy et al., 2021).

<!--en:1d907085-->
::: {.slide title="Ähnlichkeit der Positionsvektoren in Abhängigkeit vom Gitterabstand"}

<!--en:175f74a3-2-->
[Geometrie der Positionsvektoren]{.kicker}

<!--en:d367ee5d-->
::: {.slide title="Kontrollierter Vergleich zwischen ViT und einem gleich großen ResNet"}

<!--en:0663da23-->
[Versuchsaufbau]{.kicker}

<!--en:b1718875-->
- Um den induktiven Bias der Faltung isoliert zu untersuchen, gelten für `CompactResNet` (7 Residualblöcke) und `ViT` ($L = 2, d = 512, p = 16$) in allen vier Vergleichsgrößen dieselben Versuchsbedingungen:

<!--en:0c453e25-->
- **Gleiches Budget:** $6.46\text{M}$ gegenüber $6.45\text{M}$ Parametern, $0.48$ gegenüber $0.55\text{ GFLOPs/image}$, 10 Epochen auf Fashion-MNIST mit Bildgröße $96 \times 96$ und AdamW ($\lambda = 0.05$) mit Aufwärmphase und Kosinus-Abfall.

<!--en:50877f48-->
::: {.slide title="Bei wenigen Daten liegt das CNN 5 Prozentpunkte vorn"}

<!--en:436d3895-->
[Ergebnisse für ViT und CNN]{.kicker}

<!--en:4194623b-->
- **Stabiler Vorsprung des CNN um 5 Prozentpunkte:** In unabhängigen Läufen (Streuung über Zufallsstartwerte $\le 0.1\%$) erreicht `CompactResNet` eine Validierungsgenauigkeit von **$\approx 94\%$**, gegenüber **$\approx 89\%$** für `ViT`.

<!--en:d180e70b-->
- **Schnellere Konvergenz zu Beginn:** Durch seine eingebaute 2D-Lokalität und Translationsäquivarianz liegt das CNN schon in Epoche 1 mehrere Prozentpunkte vorn. Der Vorsprung hält bis Epoche 10 an.

<!--en:cdbbe9b0-->
::: {.slide title="Wann überholt ViT die CNNs? Datenmenge und Pre-Training"}

<!--en:058a0b2b-->
[Skalierungsgesetze für die Bildverarbeitung]{.kicker}

<!--en:7592c382-->
- **Abhängigkeit von der Datensatzgröße (Dosovitskiy et al. 2021):** Unter $10\text{M}$ Bildern führen ResNets. Bei $30\text{M}\text{--}300\text{M}$ Bildern ziehen ViTs vorbei, weil globale Attention dann mehr leistet als feste lokale Filter.

<!--en:2bc5aec2-->
- **Einheitliche multimodale Schnittstelle:** ViT wandelt Bilder in gewöhnliche Tokenvektoren um. Dadurch lassen sich Bildpatches **ohne jede Architekturänderung** an ein LLM oder einen Perceiver-Resampler übergeben.

<!--en:be2dcdbe-->
::: {.slide title="Verständnisfragen zu Patch-Größe, FLOPs und Invarianz"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:175ccaf7-->
Drei quantitative Fragen beantworten:

<!--en:88c71ef3-->
1. **Halbierung der Patch-Größe:** Für Bilder der Größe $96 \times 96$ kommt unser `ViT` ($d = 512, L = 2$) zum Einsatz. Die Patch-Größe sinkt von $p = 16$ auf $p = 8$. Wie ändern sich (a) die Anzahl der Patch-Token $m$, (b) die Parameteranzahl der $L$ `ViTBlock`-Schichten und (c) die FLOPs pro Bild im Vorwärtsdurchlauf?

<!--en:c5aee1f6-->
2. **Tokenanzahl für ImageNet:** Wie viele Patch-Token $m$ und wie viele Attention-Scores $m^2$ pro Kopf ergeben sich für ein ImageNet-Bild der Größe $224 \times 224$ bei $p = 16$ und bei $p = 8$?

<!--en:c7c1d11e-->
3. **Positionseinbettungen auf null:** Beim Test `model.pos_embedding.zero_()` aufrufen und `shuffled_accuracy(model, data, perm)` für 5 verschiedene Zufallspermutationen `perm` auswerten. Wie stark unterscheiden sich die 5 Genauigkeitswerte?

<!--en:09b92175-->
::: {.slide title="Verständnisfragen zu Patch-Größe, FLOPs und Invarianz (Lösungen)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:afeed5a0-->
1. (a) Die Anzahl der Patch-Token vervierfacht sich: **$m = 36 \to 144$** ($12 \times 12$-Gitter). (b) Die Parameteranzahl der `ViTBlock`-Schichten ($12 L d^2$) bleibt **genau gleich**. Die Gesamtzahl der Parameter sinkt um $<1\%$, da der Faltungskern im Eingangsteil von $16 \times 16$ auf $8 \times 8$ schrumpft. (c) Die FLOPs im Vorwärtsdurchlauf **vervierfachen sich** ($0.48 \to 1.92\text{ GFLOPs}$).

<!--en:135cbdc3-->
2. Bei $p = 16$ ergeben sich **$m = 14^2 = 196$ Token** ($38{,}416$ Scores pro Kopf). Bei $p = 8$ sind es **$m = 28^2 = 784$ Token** ($614{,}656$ Scores pro Kopf, **$16\times$ so viele Attention-Scores**).

<!--en:55a51afa-->
3. **Kein Unterschied (bis auf Rundung bei der Gleitkommarechnung):** Ohne Positionseinbettungen ist Self-Attention streng permutationsäquivariant und das Auslesen des `<cls>`-Tokens streng permutationsinvariant.

<!--en:55f782ca-->
::: {.slide title="Zusammenfassung: Vision Transformer und konvolutionäre Netze im Vergleich"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:19eacaf6-->
- **Patch-Eingangsteil mit Stride $p$:** Eine Faltungsschicht erzeugt gemäß $(C, H, W) \to m = HW / p^2$ die Token. Die Parameteranzahl der Blöcke ($12 L d^2$) bleibt unabhängig von $m$.

<!--en:dcd36d6f-->
- **Räumlicher induktiver Bias:** Ohne Positionseinbettungen ist ViT gegenüber Permutationen der Patches invariant. Mit gelernten Positionseinbettungen sinkt beim Mischen der Patches im Test die Genauigkeit von $89\%$ auf $25\%\text{--}45\%$.

<!--en:04007229-->
- **Kleine und große Datenmengen:** Beim Training von Grund auf mit wenigen Daten liegt das gleich große ResNet um $5\%$ vorn. Beim Pre-Training mit den Datenmengen großer Basismodelle dominieren ViTs.
