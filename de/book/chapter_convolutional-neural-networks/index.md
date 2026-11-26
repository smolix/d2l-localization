<!--en:ef761b00-->
# Konvolutionäre Netze

<!--en:029944cc-->
Alle Bildklassifikatoren, die wir bisher gebaut haben, ignorieren die räumliche Anordnung der Pixel. Sie formen das zweidimensionale Pixelraster in einen eindimensionalen Vektor um. Das mehrschichtige Perzeptron (engl. multilayer perceptron, MLP) aus :numref:`chap_perceptrons` behandelt ein Fashion-MNIST-Foto der Größe $28 \times 28$ wie eine ungeordnete Liste von 784 Zahlen. Würden wir die Pixel in jedem Bild nach derselben festen Zufallspermutation umordnen und das MLP neu trainieren, würde es genau dieselbe Genauigkeit erreichen. Für das menschliche Auge sähen die Bilder dagegen wie reines Rauschen aus. Wir Menschen erkennen einen Turnschuh an seinem Umriss und seinen Texturen. Diese Muster entstehen aus *benachbarten* Pixeln. Ein Umriss hat überall im Bild dieselbe Bedeutung.

<!--en:9838d1c2-->
Einen frühen Hinweis lieferten die Neurowissenschaften. In klassischen Versuchen mit Katzen und Affen entdeckten Hubel und Wiesel, dass Neuronen in frühen Arealen des visuellen Kortex auf Kanten und Balken bestimmter Orientierung reagieren. Jedes Neuron reagiert dabei auf Reize in einem begrenzten Ausschnitt des Gesichtsfelds, seinem *rezeptiven Feld* (engl. receptive field) :cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968`. Später modellierte :citet:`Field.1987` diese Reaktionen auf natürliche Bilder mit einer kleinen Bank lokaler Filter, die er auf jede Bildposition anwandte (:numref:`field_visual`).

<!--en:fb4f9a8d-->
![Sechs Filter für rezeptive Felder und ihre Antworten auf ein natürliches Bild, nach :citet:`Field.1987`. Links sehen wir die Filter, in der Mitte das Eingabebild und rechts die Filterantworten. Die Abtastabstände sind proportional zur Größe des jeweiligen Filters. Verschiedene Filter reagieren auf unterschiedliche lokale Kantenmuster.](../img/field-visual.png)

<!--en:b6ce82d4-->
Angeregt durch diese Befunde ordnete Fukushima in seinem Neocognitron :cite:`Fukushima.1980` lokale Detektoren in hierarchischen Schichten an, in denen sich die Detektoren über den Raum hinweg wiederholen. LeCun und sein Team zeigten, wie sich solche Architekturen mit Backpropagation von der Eingabe bis zur Ausgabe darauf trainieren lassen, handgeschriebene Postleitzahlen zu erkennen :cite:`LeCun.Boser.Denker.ea.1989,LeCun.Jackel.Bottou.ea.1995`. Die Repräsentationen in tieferen Schichten trainierter Netze zur Bildverarbeitung ähneln sogar denen in höheren Stufen der Sehbahn von Primaten :cite:`Kuzovkin.Vicente.Petton.ea.2018`. Die Biologie lieferte allerdings nur die Anregung für die Architektur, keinen Bauplan für die Berechnung auf Siliziumchips.

<!--en:65cf35a0-->
Im Zentrum dieser Architekturen steht die *Faltungsschicht* (engl. convolutional layer). Sie schiebt einen kleinen Faltungskern (engl. kernel) mit denselben Gewichten über jede Nachbarschaft im Bild. Jede Ausgabeeinheit hängt nur von Pixeln in ihrer Nähe ab. Verschieben wir die Eingabe, verschiebt sich die Ausgabe entsprechend. So sind Lokalität und Translationsäquivarianz direkt im Netz verankert, und die Parameteranzahl sinkt um Größenordnungen. Diese *konvolutionären neuronalen Netze* (engl. convolutional neural network, CNN) ermöglichten den entscheidenden Durchbruch auf ImageNet :cite:`Deng.Dong.Socher.ea.2009,Krizhevsky.Sutskever.Hinton.2012`. Neben Vision Transformern bilden sie bis heute eine Grundlage der Computer Vision (:numref:`chap_transformers`).

<!--en:6c94ff74-->
In diesem Kapitel leiten wir CNNs in drei Schritten aus grundlegenden Prinzipien her und bauen sie auf. Zuerst zeigen wir, was aus einer vollständig verbundenen Schicht wird, wenn wir Lokalität und Translationsäquivarianz fordern. Als einzige lineare Operation bleibt die Faltung (engl. convolution) übrig (:numref:`sec_why-conv`). Dann implementieren wir die zweidimensionale Kreuzkorrelation und lernen mit dem Gradientenabstieg einen Faltungskern zur Kantenerkennung. Wir formulieren die Faltung als Matrizenmultiplikation (*im2col*) und verfolgen, wie sich die rezeptiven Felder mit jeder weiteren Schicht vergrößern (:numref:`sec_conv_layer`). Im zweiten Schritt steuern wir die räumliche Auflösung, das Sichtfeld und die Anzahl der Merkmale, die das Netz darstellen kann. Dazu nutzen wir Padding, Stride und Dilatation (:numref:`sec_padding`), mehrkanalige Filterbänke und $1 \times 1$-Flaschenhälse (:numref:`sec_channels`) sowie lokales und globales Pooling (:numref:`sec_pooling`). Zum Schluss setzen wir diese Bausteine zu LeNet zusammen (:numref:`sec_lenet`). Wir vergleichen LeNet auf Fashion-MNIST mit unserem MLP und untersuchen die gelernten Filter der ersten Schicht. Um den Beitrag des räumlichen induktiven Bias (engl. inductive bias) zu isolieren, testen wir beide Modelle auf Bildern mit zufällig umgeordneten Pixeln. Moderne tiefe Faltungsarchitekturen behandeln wir in :numref:`chap_modern_cnn`.

<!--en:5dd576ec-->
## Materialien und weiterführende Literatur {.unnumbered}

<!--en:a64691d2-->
Bei jedem der folgenden Materialien ist angegeben, welche Abschnitte dieses Kapitels es ergänzt. Soweit nicht anders vermerkt, sind alle Materialien online frei zugänglich.

<!--en:f3e47eb0-->
**Bücher**

<!--en:e68aa4cd-->
- [Deep Learning (Goodfellow, Bengio & Courville)](https://www.deeplearningbook.org/) ist kostenlos als HTML-Version verfügbar. Kapitel 9 behandelt dünn besetzte Wechselwirkungen, Gewichtsteilung und Äquivarianz im Detail (:numref:`sec_why-conv`, :numref:`sec_pooling`).

<!--en:e398d41c-->
- [Understanding Deep Learning (Simon J. D. Prince)](https://udlbook.github.io/udlbook/) ist kostenlos als PDF verfügbar. Kapitel 10 erläutert Faltungen, Stride, Dilatation und rezeptive Felder anhand anschaulicher geometrischer Abbildungen (:numref:`sec_conv_layer`, :numref:`sec_padding`).

<!--en:2ca5481e-->
- [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/chap6.html) ist online frei zugänglich. Kapitel 6 führt von Grund auf in Faltungsschichten, gemeinsam genutzte Gewichte und Pooling ein (:numref:`sec_conv_layer`, :numref:`sec_pooling`).

<!--en:51311a3d-->
**Kurse und Videovorlesungen**

<!--en:3911fa8a-->
- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/convolutional-networks/) bietet ein frei zugängliches Skript. Das Modul „Convolutional Networks“ ist eine häufig genutzte Online-Ergänzung zu diesem Kapitel. Es erläutert, wie man Schichtdimensionen berechnet und Parameter zählt, und zeigt typische Schichtfolgen (:numref:`sec_padding`, :numref:`sec_channels`).

<!--en:ffafa570-->
- [Michigan EECS 498-007: Deep Learning for Computer Vision (Justin Johnson)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/) bietet kostenlose Vorlesungsvideos. Sie führen im selben Tempo wie dieses Kapitel durch die Berechnungen bei Faltung und Pooling (:numref:`sec_conv_layer`, :numref:`sec_pooling`).

<!--en:b24c3731-->
**Tutorials, Skripte und interaktive Angebote**

<!--en:40460908-->
- [A guide to convolution arithmetic for deep learning (Dumoulin & Visin, 2016)](https://arxiv.org/abs/1603.07285) ist frei zugänglich. Zusammen mit den begleitenden [Animationen](https://github.com/vdumoulin/conv_arithmetic) dient der Beitrag als visuelles Nachschlagewerk für Padding, Stride, Dilatation und transponierte Faltungen (:numref:`sec_padding`).

<!--en:058c280d-->
- [CNN Explainer (Wang et al., Polo Club)](https://poloclub.github.io/cnn-explainer/) bietet ein interaktives CNN, mit dem sich Aktivierungen, Faltungskerne und rezeptive Felder im Browser untersuchen lassen (:numref:`sec_channels`, :numref:`sec_lenet`).

<!--en:a492243e-->
- [Computing Receptive Fields of Convolutional Neural Networks (Araujo, Norris & Sim, 2019, *Distill*)](https://distill.pub/2019/computing-receptive-fields/) leitet die in :numref:`sec_conv_layer` eingeführten geschlossenen Formeln für rezeptive Felder her. Die Herleitung umfasst auch Faltungen mit größerem Stride und Architekturen mit mehreren Pfaden.

<!--en:6e881d9e-->
- [Conv Nets: A Modular Perspective (Chris Olah)](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) ist ein kurzer, klassischer Beitrag, der die Faltung als strukturierte Form der Gewichtsteilung veranschaulicht (:numref:`sec_why-conv`).

<!--en:f65c5934-->
- [Image Kernels Explained Visually (Victor Powell)](https://setosa.io/ev/image-kernels/) ergänzt :numref:`sec_conv_layer` um ein interaktives Werkzeug, mit dem sich von Hand entworfene Faltungskerne über ein Bild schieben lassen.

<!--en:0c736337-->
- [Feature Visualization (Olah, Mordvintsev & Schubert, 2017, *Distill*)](https://distill.pub/2017/feature-visualization/) veranschaulicht, worauf einzelne Einheiten eines CNN reagieren, und ergänzt damit :numref:`sec_lenet`.

<!--en:d63f4e40-->
**Grundlegende Forschungsarbeiten**

<!--en:d85d5677-->
- [Gradient-Based Learning Applied to Document Recognition (LeCun, Bottou, Bengio & Haffner, 1998, *Proc. IEEE*)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) ist als kostenloses PDF verfügbar. Die Arbeit stellt LeNet-5 vor und bildet die Grundlage für :numref:`sec_lenet`.

<!--en:d46d3469-->
- [Backpropagation Applied to Handwritten Zip Code Recognition (LeCun et al., 1989, *Neural Computation*)](https://doi.org/10.1162/neco.1989.1.4.541) beschreibt das erste CNN, das mit Backpropagation von der Eingabe bis zur Ausgabe trainiert wurde (kostenpflichtig) (:numref:`sec_lenet`).

<!--en:9410a7b8-->
- [Neocognitron (Fukushima, 1980, *Biological Cybernetics*)](https://doi.org/10.1007/BF00344251) beschreibt einen Vorläufer der Architekturen mit abwechselnden Faltungs- und Poolingstufen aus der Zeit vor der Backpropagation (kostenpflichtig) (:numref:`sec_pooling`).

<!--en:fe39f3f3-->
- [Receptive fields, binocular interaction and functional architecture in the cat's visual cortex (Hubel & Wiesel, 1962, *J. Physiology*)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/) ist über PMC frei zugänglich. Die Arbeit beschreibt die biologischen Grundlagen lokaler rezeptiver Felder und der hierarchischen Merkmalserkennung (:numref:`sec_why-conv`).

<!--en:49541e54-->
Konvolutionäre neuronale Netze: Schichten für die räumliche Struktur von Bildern<br>**Lokalität · Translationsäquivarianz · Kreuzkorrelation · Padding · Kanäle · Pooling · LeNet**

<!--en:67402ace-->
::: {.slide title="Beim Umformen zum Vektor geht die Bildgeometrie verloren"}

<!--en:ae166c83-->
[Motivation und induktiver Bias]{.kicker}

<!--en:d1c67e2f-->
- **Blind für Permutationen:** Ordnen wir alle 784 Pixel jedes Fashion-MNIST-Bildes nach derselben festen Permutation um, bleibt die Genauigkeit eines MLP gleich ($\approx 87\%$).

<!--en:874a0bc3-->
- **Parameterexplosion bei großen Bildern:** Für ein RGB-Bild mit 1 Megapixel benötigt eine vollständig verbundene Schicht $(3 \times 10^6)^2 = 9 \times 10^{12}$ Gewichte (**36 TB** in `float32`).

<!--en:bdbc2d1f-->
- **Geometrische Vorannahme:** Fordern wir **Lokalität** und **Translationsäquivarianz**, genügen statt Billionen von Gewichten einige Hundert Parameter in einem gemeinsam genutzten Faltungskern.

<!--en:c2ffd101-->
::: {.slide title="Rezeptive Felder beim biologischen und maschinellen Sehen"}

<!--en:3ae29bb0-->
[Vom visuellen Kortex zu konvolutionären Netzen]{.kicker}

<!--en:49025cdd-->
Hubel und Wiesel entdeckten, dass Neuronen in frühen Arealen des visuellen Kortex auf Kanten bestimmter Orientierung in ihren lokalen *rezeptiven Feldern* reagieren. Das gab den Anstoß für hierarchische Filterbänke, die über das ganze Bild gleiten:

<!--en:02cc4043-->
- **Lokale Filterbank (links $\to$ rechts):** Die Faltung eines natürlichen Bildes mit sechs gerichteten Kantenfiltern ergibt dünn besetzte Aktivierungskarten. Sie heben lokale Konturen unterschiedlicher Orientierung und Größe hervor.

<!--en:1169bf6b-->
- **End-to-End-Lernen:** Konvolutionäre neuronale Netze (CNNs) lernen die Gewichte der Faltungskerne in allen Schichten gleichzeitig durch Backpropagation. Von Hand entworfene Filterbänke entfallen.

<!--en:39d2dfa9-->
::: {.slide title="Mathematik und Architekturen in Kapitel §{num:chap_cnn}"}

<!--en:00be04d1-->
[Aufbau des Kapitels]{.kicker}

<!--en:6e6c72e3-->
- **§§{sec_why-conv}–§{num:sec_conv_layer} (Symmetrie und Faltungsschicht):** Aus Translationsäquivarianz und Lokalität folgt die 2D-Kreuzkorrelation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$. Wir berechnen sie mit `im2col` und GEMM.

<!--en:e2d5e914-->
- **§§{sec_padding}–§{num:sec_channels} (Geometrie und Kanäle):** Padding $p$, Stride $s$, Dilatation $d$, 4D-Filterbänke $(c_\textrm{o}, c_\textrm{i}, k_\textrm{h}, k_\textrm{w})$ und $1 \times 1$-Flaschenhälse.

<!--en:67976edf-->
- **§§{sec_pooling}–§{num:sec_lenet} (Pooling und LeNet):** Parameterfreies Pooling und Überprüfung des gesamten LeNet im Vergleich mit unserem MLP.

<!--en:d8bc79ea-->
::: {.slide title="Kennzahlen aus Kapitel §{num:chap_cnn}"}

<!--en:418d9f93-->
[Quantitative Ergebnisse]{.kicker}

<!--en:f9c20798-->
- **Gewichte unabhängig von der Auflösung:** Eine $3 \times 3$-Eingangsschicht von RGB auf 64 Kanäle benötigt bei jeder Bildauflösung $1{,}792$ Parameter. $1 \times 1$-Flaschenhälse senken die Kosten tiefer $3 \times 3$-Schichten um den Faktor $8.5\times$.

<!--en:f4667bb8-->
- **Reichweite und Verschiebungstoleranz:** Strides $(2,2,1)$ oder Dilatationen $(1,2,4)$ vergrößern die Reichweite von 3 Schichten von $r=7$ auf $r=15$, also auf etwa das Doppelte. $2 \times 2$-Pooling fängt $50\%$ der Verschiebungen um 1 Pixel ab.

<!--en:e3569e2a-->
- **Ressourcenverteilung in LeNet:** Die Faltungsschichten enthalten $4.2\%$ der Gewichte, führen aber $85.9\%$ der Multiplikations-Akkumulations-Operationen (MACs) aus.

<!--en:2caa5954-->
::: {.slide title="Verständnisfragen und Lernstrategie für das Kapitel"}

<!--en:6a3f295c-->
[Wissen prüfen und Zusammenhänge erkennen]{.kicker}

<!--en:3ed99e5d-->
- **1. Höhere Auflösung ($224^2 \to 448^2$):** Wie ändern sich Parameteranzahl, Zahl der MACs und Speicherbedarf der Aktivierungen, wenn sich Höhe und Breite verdoppeln?

<!--en:81f49b72-->
- **2. Test auf induktiven Bias:** Warum übertrifft ein CNN ein MLP auf natürlichen Bildern, verliert diesen Vorteil aber bei einer festen Zufallspermutation der Pixel?

<!--en:e6db9f74-->
- **Antwort:** Die Parameteranzahl bleibt bei **$1\times$** (festgelegt durch $c_\textrm{o}, c_\textrm{i}, k$). Für MACs und Aktivierungen beträgt der Faktor **$4\times$**, da ihre Anzahl proportional zur Fläche $H \times W$ wächst. Das Umordnen der Pixel zerstört die zweidimensionale Lokalität.
