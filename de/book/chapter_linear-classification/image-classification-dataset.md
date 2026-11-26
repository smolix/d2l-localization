<!--en:b7e57356-->
# Daten für die Bildklassifikation

<!--en:58c1a818-->
Um die Softmax-Regression zu trainieren und zu testen, brauchen wir einen Benchmark für die Bildklassifikation (engl. image classification). Er soll anspruchsvoll genug sein, um schwache von starken Modellen zu unterscheiden, und so kompakt, dass das Training in einem interaktiven Notebook nur Sekunden dauert. Der klassische [MNIST](https://en.wikipedia.org/wiki/MNIST_database)-Datensatz mit handgeschriebenen Ziffern :cite:`LeCun.Bottou.Bengio.ea.1998` war in den 1990er-Jahren ein wegweisender Benchmark. Er umfasst $70{,}000$ Graustufenbilder der Größe $28 \times 28$ in 10 Ziffernklassen. Heute erreichen selbst triviale Modelle eine Genauigkeit (engl. accuracy) von über $97\%$, konvolutionäre Netze (engl. convolutional neural networks, CNNs) sogar über $99\%$. Verbesserungen an der Architektur lassen sich daran kaum noch erkennen. Am anderen Ende des Spektrums steht ImageNet :cite:`Deng.Dong.Socher.ea.2009` mit $1.2$ Millionen Farbfotos in $1{,}000$ Kategorien. Hier erfordert jeder Durchlauf mehrere Stunden GPU-Rechenzeit.

<!--en:8e26b233-->
*Fashion-MNIST* :cite:`Xiao.Rasul.Vollgraf.2017` bietet einen guten Mittelweg. Der Datensatz hat denselben Umfang wie MNIST: $60{,}000$ Trainingsbilder und $10{,}000$ Testbilder, alle in Graustufen, mit jeweils $28 \times 28$ Pixeln und gleichmäßig auf $10$ Klassen verteilt. An die Stelle der Ziffern treten Modeartikel (T-Shirt, Hose, Pullover, Kleid, Mantel, Sandale, Hemd, Sneaker, Tasche und Stiefelette). Auf Fashion-MNIST erreicht die lineare Softmax-Regression eine Genauigkeit von rund $83\%$, konvolutionäre Netze kommen auf über $90\%$. In diesem Abschnitt entwickeln wir die Klasse `FashionMNIST(d2l.DataModule)`, die unsere Bildklassifikatoren in den nächsten Kapiteln mit Daten versorgt.

<!--en:b17ecc77-->
## Laden des Datensatzes

<!--en:176884aa-->
Fashion-MNIST enthält $6{,}000$ Trainingsbilder und $1{,}000$ Testbilder für jede der $10$ Kategorien (insgesamt $60{,}000$ Trainingsbilder und $10{,}000$ Testbilder). Die Pixelwerte sind als vorzeichenlose 8-Bit-Ganzzahlen im Intervall $[0, 255]$ gespeichert ($0$ = dunkler Hintergrund). Gemäß unserem objektorientierten Entwurf (:numref:`sec_oo-design`) ist `FashionMNIST` eine Unterklasse von `d2l.DataModule`. Sie erwartet `batch_size` und optional ein Tupel `resize` für die Bildgröße (Standardwert $(28, 28)$):

<!--en:87d23308-->
`torchvision.datasets.FashionMNIST` lädt den Datensatz herunter und hält alle Bilder eines Teildatensatzes in einem einzigen `uint8`-Tensor. Der Datenlader (engl. data loader), den wir weiter unten definieren, liest diesen Tensor und wandelt jeweils ganze Minibatches um. Die Pipeline `transforms.Compose` passt die Größe eines einzelnen Bildes an und skaliert seine Pixelintensitäten. Das Ergebnis ist ein float32-Tensor mit Werten in $[0, 1]$. Die Pipeline kommt zum Einsatz, wenn wir wie in `data.train[i]` über einen Index direkt auf den Datensatz zugreifen oder `data.train` an einen eigenen Datenlader übergeben.

<!--en:ec960caa-->
Gluons integrierte Hilfsfunktion für Fashion-MNIST weicht von der offiziellen Aufteilung in Trainings- und Testdaten ab. Daher lädt `FashionMNIST.read` die mit gzip komprimierten offiziellen IDX-Binärdateien herunter und liest sie direkt ein. Dabei überspringt die Methode die Kopfdaten (16 Byte für Bilder, 8 Byte für Labels). Anschließend skaliert `to_dataset` die Pixelwerte auf das Intervall $[0, 1]$ und gibt sie im Format float32 aus.

<!--en:39dafe97-->
`tf.keras.datasets.fashion_mnist.load_data()` liefert die offiziellen Trainings- und Testdaten jeweils als Paar `(images, labels)` von NumPy-Arrays.

<!--en:34e29cd6-->
Erzeugen wir eine Instanz des Datensatzes und prüfen wir die Größe der beiden Teildatensätze.

<!--en:6bf891a9-->
Fashion-MNIST besteht aus nur zwei Teildatensätzen. Daher verwenden wir den Testdatensatz mit seinen $10{,}000$ Bildern während des Trainings als Validierungsdatensatz (engl. validation set). Die statistischen Folgen der wiederholten Auswertung auf dem Testdatensatz untersuchen wir in :numref:`chap_classification_generalization`. Ergänzen wir nun die Methode `text_labels`, die den ganzzahligen Klassenindizes $0, \ldots, 9$ lesbare Bezeichnungen für die Modeartikel zuordnet:

<!--en:c7bbce38-->
## Einlesen von Minibatches

<!--en:246f68c9-->
Betrachten wir einen Minibatch der Größe $n$. Jedes Bild hat $c$ Kanäle ($c = 1$ für Graustufen, $c = 3$ für RGB-Farbbilder), die Höhe $h$ und die Breite $w$. Je nach Deep-Learning-Framework hat der Minibatch die Achsenreihenfolge *Channel-First* $(n, c, h, w)$ oder *Channel-Last* $(n, h, w, c)$.

<!--en:78e754d8-->
MXNet speichert Bilder in der Achsenreihenfolge Channel-First $(n, c, h, w)$.

<!--en:81fbd3bc-->
PyTorch speichert Bilder in der Achsenreihenfolge Channel-First $(n, c, h, w)$.

<!--en:2becfc30-->
TensorFlow speichert Bilder in der Achsenreihenfolge Channel-Last $(n, h, w, c)$.

<!--en:3977e682-->
JAX speichert Bilder in der Achsenreihenfolge Channel-Last $(n, h, w, c)$. Wir kapseln `tf.data.Dataset` in einer schlanken Hilfsklasse `TensorFlowDataLoader`. Sie stellt `.as_numpy_iterator()` und `__len__` bereit, sodass wir in jeder Epoche direkt NumPy-Minibatches erhalten:

<!--en:8b8388c8-->
Nun ergänzen wir die Methode `get_dataloader(self, train)` in der Klasse `FashionMNIST`. Sie nutzt den Datenlader des Frameworks und mischt die Trainingsdaten in jeder Epoche neu. Den letzten, unvollständigen Minibatch behält sie bei ($60{,}000 = 937 \times 64 + 32$). So kommt jedes Bild genau einmal pro Epoche an die Reihe:

<!--en:14da29fe-->
Die Hilfsmethode `batchify` passt beim Einlesen jedes Minibatchs die Bildgröße an.

<!--en:c7d69d37-->
Die Hilfsmethode `batchify` skaliert beim Einlesen die Pixelwerte jedes Minibatchs auf float32-Werte in $[0, 1]$ und passt die Bildgröße an. Eine Operation für den ganzen Minibatch ist wesentlich günstiger als eine separate Operation für jedes Bild. Deshalb überspringt der Datenlader die Transformation für Einzelbilder.

<!--en:bf3b82c3-->
Holen wir einen Minibatch `(X, y)` von `train_dataloader()`. Betrachten wir die Achsenreihenfolge eines einzelnen Bildes sowie die Formen und Datentypen der Tensoren für den gesamten Minibatch:

<!--en:e7d36e8d-->
Fassen wir Beispiele zu einem Minibatch zusammen, erhalten die Bilder eine zusätzliche Batchachse an erster Stelle. Die Labels bilden einen eindimensionalen Vektor mit einem Eintrag pro Bild.

<!--en:c4629ba6-->
Bei $64 \times 28 \times 28 \times 4 \approx 200\textrm{ KB}$ pro Minibatch (und $\sim 190\textrm{ MB}$ für alle $60{,}000$ Trainingsbilder im Format float32) passt der gesamte Datensatz problemlos in den Arbeitsspeicher. Trotzdem sollten wir die Geschwindigkeit des Datenladers stets separat messen. Dazu stoppen wir die Zeit für einen vollständigen Durchlauf von `train_dataloader()` ohne Modell. Keine Trainingsepoche kann schneller sein als ihre Datenpipeline.

<!--en:2683844a-->
## Sichtprüfung der Bilder

<!--en:60b66265-->
Sehen Sie sich vor jedem Modelltraining eine Stichprobe von Bildern mit ihren Labels an. Falsch zugeordnete Labels, invertierte Helligkeitswerte oder eine Umformung, die Pixel durcheinanderbringt, machen das Training zunichte, ohne einen Laufzeitfehler auszulösen. Wir ergänzen die Methode `visualize` in der Klasse `FashionMNIST`. Sie entfernt die Kanalachse der Länge eins und zeigt die Graustufenbilder in einem Raster an, jeweils mit dem Text des Labels als Titel:

<!--en:a6c48d84-->
Zeigen wir die ersten drei Validierungsbilder jeder Klasse, mit einer Spalte pro Kategorie. Dabei nutzen wir aus, dass der Datenlader für die Validierung die Bilder stets in derselben Reihenfolge liefert:

<!--en:1dc92377-->
Schon auf den ersten Blick erkennen wir, wo unser linearer Klassifikator Schwierigkeiten haben dürfte. Hosen und Taschen haben unverwechselbare Umrisse. Sandalen, Sneaker und Stiefeletten bilden als Schuhe eine klar erkennbare Gruppe. Die vier Oberteile (T-Shirt, Pullover, Mantel und Hemd) haben am Rumpf fast denselben kastenförmigen Umriss. Sie unterscheiden sich nur in feinen Details an Kragen, Ärmeln und Knöpfen, die bei einer Auflösung von $28 \times 28$ kaum zu erkennen sind.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:4ff99acb-->
Fashion-MNIST umfasst $60{,}000$ Trainingsbilder und $10{,}000$ Testbilder in Graustufen mit jeweils $28 \times 28$ Pixeln, gleichmäßig verteilt auf $10$ Kategorien von Modeartikeln. Der Datensatz ist anspruchsvoll genug, um lineare Modelle ($\sim 83\%$) von konvolutionären Netzen ($>90\%$) zu unterscheiden, und klein genug für ein Training in Sekundenschnelle. Unsere Pipeline `FashionMNIST(d2l.DataModule)` skaliert die Pixelwerte vom Rohformat `uint8` auf `float32` im Intervall $[0, 1]$ und kann bei Bedarf die Bildgröße anpassen. Sie liefert neu gemischte Minibatches mit ganzzahligen Labels und verwendet die für das jeweilige Framework übliche Achsenreihenfolge. Aus der Laufzeit des Datenladers allein ergibt sich eine Untergrenze für die Dauer einer Epoche. Eine Sichtprüfung der beschrifteten Bilder deckt unbemerkte Fehler in Tensorformen oder Labelzuordnungen schon vor dem Training auf.

<!--en:2294243a-->
## Übungen

<!--en:1e747f1f-->
1. [code] **Klassenmittelwerte.** Berechnen Sie für jede Klasse das mittlere Bild über den Trainingsdatensatz. Stellen Sie die zehn Mittelwerte als Bilder in einem Raster dar.
    1. Berechnen Sie für jedes Paar von Klassenmittelwerten den euklidischen Abstand. Stellen Sie die Abstände in einer Tabelle der Größe $10 \times 10$ dar. Welche Paare liegen am nächsten beieinander und welche Klasse ist von allen anderen am weitesten entfernt?
    1. Sagen Sie anhand der Tabelle vorher, welche Klassen ein linearer Klassifikator am häufigsten und welche er fast nie miteinander verwechselt. Halten Sie Ihre Vorhersage fest. In :numref:`sec_softmax_scratch` berechnen wir die Antwort.
    1. Nennen Sie zwei Klassen, die ein Mensch leicht auseinanderhalten kann, obwohl ihre Mittelwerte nahe beieinanderliegen. Geben Sie an, an welchen Pixeln der Unterschied zu erkennen ist.

<!--en:b30b2618-->
2. [code] **Durchsatz des Datenladers.** Die Dauer eines Durchlaufs durch den Trainingsdatensatz hängt von der Minibatchgröße ab und davon, wie viel Arbeit die Pipeline parallel erledigt.
    1. Messen Sie die Dauer eines vollständigen Durchlaufs von `train_dataloader` für `FashionMNIST` mit den Werten 1, 16, 64, 256 und 1024 für `batch_size`. Tragen Sie den Durchsatz in Bildern pro Sekunde gegen `batch_size` auf. Erklären Sie den Kurvenverlauf anhand des Verwaltungsaufwands pro Batch und des Rechenaufwands pro Bild.
    1. Die `tf.data`-Pipeline in `get_dataloader` verwendet das Attribut `num_workers` von `DataModule` nicht. Die Parallelität wird für jede Stufe einzeln festgelegt. Messen Sie zunächst die Dauer eines Durchlaufs mit dem vorliegenden Code. Wiederholen Sie die Messung mit `num_parallel_calls=tf.data.AUTOTUNE` im Aufruf von `map` und einem anschließenden `.prefetch(tf.data.AUTOTUNE)`. Geben Sie an, unter welchen Bedingungen zusätzliche Parallelität keinen Zeitgewinn mehr bringt.

<!--en:b30b2618-2-->
2. [code] **Durchsatz des Datenladers.** Die Dauer eines Durchlaufs durch den Trainingsdatensatz hängt von der Minibatchgröße ab und davon, wie viel Arbeit die Pipeline parallel erledigt.
    1. Messen Sie die Dauer eines vollständigen Durchlaufs von `train_dataloader` für `FashionMNIST` mit den Werten 1, 16, 64, 256 und 1024 für `batch_size`. Tragen Sie den Durchsatz in Bildern pro Sekunde gegen `batch_size` auf. Erklären Sie den Kurvenverlauf anhand des Verwaltungsaufwands pro Batch und des Rechenaufwands pro Bild.
    1. Die `tf.data`-Pipeline in `get_dataloader` verwendet das Attribut `num_workers` von `DataModule` nicht. Die Parallelität wird für jede Stufe einzeln festgelegt. Messen Sie zunächst die Dauer eines Durchlaufs mit dem vorliegenden Code. Wiederholen Sie die Messung mit `num_parallel_calls=tf.data.AUTOTUNE` im Aufruf von `map` und einem anschließenden `.prefetch(tf.data.AUTOTUNE)`. Geben Sie an, unter welchen Bedingungen zusätzliche Parallelität keinen Zeitgewinn mehr bringt.

<!--en:2c202c24-->
2. [code] **Durchsatz des Datenladers.** Die Dauer eines Durchlaufs durch den Trainingsdatensatz hängt von der Minibatchgröße und von der Größe der Bilder ab, die der Datenlader liefert.
    1. Messen Sie die Dauer eines vollständigen Durchlaufs von `train_dataloader` für `FashionMNIST` mit den Werten 1, 16, 64, 256 und 1024 für `batch_size`. Tragen Sie den Durchsatz in Bildern pro Sekunde gegen `batch_size` auf. Erklären Sie den Kurvenverlauf anhand des Verwaltungsaufwands pro Batch und des Rechenaufwands pro Bild.
    1. `batchify` passt beim Einlesen jedes Minibatchs die Bildgröße an. Messen Sie die Dauer eines Durchlaufs von `train_dataloader` für `FashionMNIST(resize=(s, s))` mit $s$ = 28, 56, 112 und 224. Tragen Sie die Zeit gegen die Anzahl der Pixel pro Bild auf. Welcher Anteil der Zeit wächst mit der Pixelzahl und welcher bleibt davon unabhängig?

<!--en:834d5fa5-->
3. [code] **Upsampling und lineare Modelle.** ● Fassen Sie die Vergrößerung von $28 \times 28$ auf $32 \times 32$ als lineare Transformation $\mathbf{x}' = \mathbf{A}\mathbf{x}$ von den 784 ursprünglichen Pixeln auf die 1024 Pixel des vergrößerten Bildes auf. Die Spalten von $\mathbf{A} \in \mathbb{R}^{1024 \times 784}$ seien linear unabhängig.
    1. Zeigen Sie, dass die Softmax-Regression auf $\mathbf{x}'$ genau dieselben Klassifikatoren darstellen kann wie die Softmax-Regression auf $\mathbf{x}$.
    1. Trainieren Sie nach :numref:`sec_softmax_scratch` das Modell `SoftmaxRegressionScratch` mit Daten aus `FashionMNIST()` und aus `FashionMNIST(resize=(32, 32))`. Setzen Sie dabei `num_inputs` auf die Anzahl der Pixel pro Bild. Vergleichen Sie die Genauigkeiten auf dem Validierungsdatensatz und erklären Sie etwaige Unterschiede anhand der vorherigen Teilaufgabe.

<!--en:303e9696-->
[Diskussion](https://d2l.discourse.group/t/48)

<!--en:8ea5b6a1-->
[Diskussion](https://d2l.discourse.group/t/49)

<!--en:e3e1e385-->
[Diskussion](https://d2l.discourse.group/t/224)

<!--en:2445e033-->
[Diskussion](https://d2l.discourse.group/t/17980)

<!--en:654df33f-->
Der Datensatz für die Bildklassifikation: Fashion-MNIST und Datenpipelines<br>**Fashion-MNIST als Benchmark · API von `DataModule` · Achsenreihenfolge in Tensoren · Minibatchlader · Durchsatz der Pipeline · visuelle Fehlersuche**

<!--en:242af7f7-->
::: {.slide title="Fashion-MNIST als durchgängiger Benchmark für die Bildklassifikation"}

<!--en:fb220909-->
[Wahl des Benchmarks]{.kicker}

<!--en:6bcee1fe-->
Ein Benchmark für die Modellentwicklung muss schwache Modelle von starken Architekturen unterscheiden. Die Daten müssen in den Arbeitsspeicher passen und das Training darf nur Sekunden dauern:

<!--en:a5a1209d-->
- **MNIST (1998) ist zu einfach:** Lineare Modelle erreichen $\approx 92\%$, kleine CNNs übertreffen $99\%$. Damit bleibt kaum Spielraum, um neue Ideen zu erproben.

<!--en:5ee54d1d-->
- **ImageNet-1K (ILSVRC-2012) ist zu aufwendig:** $1.2\text{M}$ Farbfotos in $1{,}000$ Klassen erfordern pro Experiment mehrere Stunden Rechenzeit auf mehreren GPUs.

<!--en:95042594-->
- **Fashion-MNIST (2017) bietet den Mittelweg:** Mit $70{,}000$ Bildern ($28\times 28$) direkt als Ersatz für MNIST einsetzbar. Die lineare Softmax-Regression erreicht $\sim 83\%$, CNNs übertreffen $90\%$.

<!--en:1f31cb57-->
::: {.slide title="Zehn gleich stark besetzte Kategorien von Modeartikeln"}

<!--en:bf1f79d1-->
[Einteilung der Klassen]{.kicker}

<!--en:4365e2e5-->
Fashion-MNIST enthält $6{,}000$ Trainingsbilder und $1{,}000$ Testbilder für jede der $q = 10$ Kategorien (Indizierung ab $0$, Indizes $0, \ldots, 9$):

<!--en:28263c77-->
- **Gruppen nach Silhouette:** $4$ kastenförmige Oberteile (`0, 2, 4, 6`), $3$ Arten von Schuhen (`5, 7, 9`) und $3$ unverwechselbare Formen (`1` Hose, `3` Kleid, `8` Tasche)

<!--en:eebb18ff-->
::: {.slide title="Kapselung von Fashion-MNIST in einem DataModule"}

<!--en:ab1af787-->
[Objektorientierte Datenpipeline]{.kicker}

<!--en:c905e05f-->
`FashionMNIST` ist eine Unterklasse von `d2l.DataModule`. Mit `resize=(h, w)` lässt sich die Auflösung ohne Änderungen am Trainingscode anpassen:

<!--en:daa6aa4a-->
::: {.slide title="Skalierung der Pixelwerte im Datenmodul"}

<!--en:ab1af787-2-->
[Objektorientierte Datenpipeline]{.kicker}

<!--en:5e2e9ef9-->
- **Normalisierung vor der Modelleingabe:** Bevor der Datenlader einen Minibatch ausgibt, skaliert er die 8-Bit-Rohwerte (`uint8` in $[0, 255]$, dunkler Hintergrund $= 0$) auf `float32` in $[0, 1]$.

<!--en:a3dc4c9b-->
- **Änderung an einer Stelle:** Da das `DataModule` die Pixelwerte skaliert, erhält jedes Modell für Fashion-MNIST dieselben Eingaben im Intervall $[0, 1]$.

<!--en:5d22d1a4-->
::: {.slide title="Prüfung der offiziellen Aufteilung für Training und Validierung"}

<!--en:7b706305-->
[Aufteilung des Datensatzes und mögliche Fehler]{.kicker}

<!--en:b7b0ad97-->
Beim Erzeugen einer Instanz lädt `FashionMNIST()` die offiziellen $60{,}000$ Trainingsbilder und $10{,}000$ Bilder für die Auswertung in den Arbeitsspeicher:

<!--en:aa5474af-->
- **Geringer Speicherbedarf:** Bei $784$ Pixeln pro Bild belegen $60{,}000$ Trainingsbilder $\approx 47\text{ MB}$ im Format `uint8` oder $\approx 188\text{ MB}$ im Format `float32`.

<!--en:d75004c7-->
- **Mehrfache Nutzung des Testdatensatzes:** Fashion-MNIST besteht aus nur zwei Teildatensätzen. Daher verwendet `data.val` die $10\text{k}$ Testbilder (§{chap_classification_generalization} beziffert, wie stark die Mehrfachnutzung die gemessene Genauigkeit erhöht).

<!--en:0c406a0a-->
::: {.slide title="Warum MXNet die IDX-Binärdateien direkt einliest" only="mxnet"}

<!--en:2e837f54-->
[Reproduzierbarkeit über Frameworks hinweg]{.kicker}

<!--en:8ca07960-->
Warum liest MXNet die offiziellen `*-ubyte.gz`-Dateien direkt ein, statt `gluon.data.vision.FashionMNIST` aufzurufen?

<!--en:2b5cce62-->
- **Abweichende Aufteilung in Gluon:** Die integrierte Hilfsfunktion von Gluon bezieht rund $80\%$ ihrer $10{,}000$ „Testbilder“ aus den veröffentlichten **Trainingsdaten**.

<!--en:392e5ab8-->
- **Vergleichbarkeit der Frameworks:** Direktes Einlesen der offiziellen IDX-Binärdateien (ohne die Kopfdaten von $16$ Byte für Bilder und $8$ Byte für Labels) garantiert in allen Frameworks dieselbe Aufteilung in $60\text{k} + 10\text{k}$ Bilder.

<!--en:75dae6bc-->
::: {.slide title="Aufbau des Minibatchladers"}

<!--en:072b7a25-->
[Mischen und Batching]{.kicker}

<!--en:03c4cfa2-->
Die Methode `get_dataloader(train)` kapselt den Datensatz in einem iterierbaren Minibatchlader:

<!--en:4e70ebd1-->
::: {.slide title="Neumischen je Epoche und kurzer letzter Batch"}

<!--en:072b7a25-2-->
[Mischen und Batching]{.kicker}

<!--en:b613bdf8-->
- **Neumischen je Epoche (`shuffle=train`):** Beim Training werden alle $60{,}000$ Indizes in jeder Epoche neu gemischt. Bei der Validierung bleibt die Reihenfolge fest.

<!--en:c61d2324-->
- **Kurzen letzten Minibatch beibehalten:** Bei `batch_size=64` ergeben sich wegen $60{,}000 = 937 \times 64 + 32$ genau $938$ Batches pro Epoche. Kein Beispiel entfällt.

<!--en:3ab9e463-->
::: {.slide title="Achsenreihenfolge im Einzelbild: Channel-First und Channel-Last" only="pytorch,mxnet"}

<!--en:89169841-->
[Tensorkonventionen der Frameworks]{.kicker}

<!--en:c59b6981-->
Ein einzelnes Graustufenbild ($c = 1$, $h = 28$, $w = 28$) liegt als 3D-Tensor vor. Die Position der Kanalachse hängt vom Framework ab:

<!--en:2017e56d-->
- **Channel-First $(c, h, w) = (1, 28, 28)$:** In diesem Framework steht die Kanalachse an erster Stelle (`NCHW` für einen Minibatch).

<!--en:5f8d4a20-->
- **Die andere Konvention:** Channel-Last $(h, w, c)$ (`NHWC`) speichert die Kanäle jedes Pixels zusammen. Beide Anordnungen ergeben beim Abflachen dieselben $784$ Merkmale.

<!--en:79ff072e-->
::: {.slide title="Achsenreihenfolge im Einzelbild: Channel-First und Channel-Last" only="tensorflow,jax"}

<!--en:89169841-2-->
[Tensorkonventionen der Frameworks]{.kicker}

<!--en:c59b6981-2-->
Ein einzelnes Graustufenbild ($c = 1$, $h = 28$, $w = 28$) liegt als 3D-Tensor vor. Die Position der Kanalachse hängt vom Framework ab:

<!--en:5297f67f-->
- **Channel-Last $(h, w, c) = (28, 28, 1)$:** In diesem Framework steht die Kanalachse an letzter Stelle (`NHWC` für einen Minibatch).

<!--en:ee88cca1-->
- **Die andere Konvention:** Channel-First $(c, h, w)$ (`NCHW`) speichert jeden Kanal als zusammenhängende Ebene. Beide Anordnungen ergeben beim Abflachen dieselben $784$ Merkmale.

<!--en:45502e6e-->
::: {.slide title="Bilder und Labels eines vollständigen Minibatchs"}

<!--en:250b91a1-->
[4D-Bildtensoren und 1D-Labelvektoren]{.kicker}

<!--en:dfc87e63-->
Beim Zusammenfassen von $n = 64$ Bildern erhält `X` eine zusätzliche Batchachse an erster Stelle. Dazu gehört ein eindimensionaler ganzzahliger Vektor `y`:

<!--en:ec5ae56b-->
- **Speicherbedarf eines Minibatchs:** $64 \times 1 \times 28 \times 28 \times 4\text{ B} \approx 200\text{ KB}$ ($\approx 803\text{ KB}$ bei $n = 256$)

<!--en:76a108cb-->
- **Abflachen für lineare Modelle:** Die Umformung von `X` in `(-1, 784)` ergibt $\mathbf{X} \in \mathbb{R}^{n \times 784}$, sowohl bei `NCHW` als auch bei `NHWC`.

<!--en:6778a1a6-->
::: {.slide title="Separate Messung des Durchsatzes beim Laden"}

<!--en:be6df23a-->
[Analyse von Flaschenhälsen im System]{.kicker}

<!--en:0e92e41a-->
Vor dem Anschluss eines neuronalen Netzes stets die Dauer eines vollständigen Durchlaufs von `train_dataloader()` allein messen:

<!--en:88677add-->
- **Feste Untergrenze für die Epochendauer:** Datenlader und Modell arbeiten als Erzeuger und Verbraucher in einer Pipeline. Daher kann keine Trainingsepoche schneller ablaufen als der Datenlader allein.

<!--en:531fcaa9-->
::: {.slide title="Flaschenhälse in der Datenpipeline"}

<!--en:4449fc1d-->
[Zeitliche Überlappung von Erzeugen und Verbrauchen]{.kicker}

<!--en:ade3b69c-->
Wenn CPU-Arbeitsprozesse im Hintergrund Batch $b + 1$ vorab laden, während die GPU Batch $b$ verarbeitet, gilt für die tatsächlich verstrichene Zeit pro Epoche:

<!--en:432b519e-->
- **Kleine Batchgrößen ($n = 1$ oder $16$):** Der Verwaltungsaufwand für die Python-Aufrufe pro Batch überwiegt (`T_load >> T_compute`). Die GPU muss auf Daten warten.

<!--en:953472f5-->
- **Anzahl der Arbeitsprozesse:** Bei `num_workers=0` blockiert das serielle Laden zwischen den Schritten. Sobald $T_{\text{load}} < T_{\text{compute}}$ gilt, bringen zusätzliche Arbeitsprozesse keinen Zeitgewinn.

<!--en:55577406-->
::: {.slide title="Sichtprüfung von Minibatches mit Labels vor dem Training"}

<!--en:ed2a54a4-->
[Vorbeugende Datenprüfung]{.kicker}

<!--en:0ed0c665-->
Verschobene Labelzuordnungen, vertauschte räumliche Achsen oder durcheinandergeratene Kanäle lösen keine Syntaxfehler aus. Zuerst immer eine Bildübersicht mit Labels anzeigen:

<!--en:c9708f00-->
- **Kanalachse entfernen:** Die Umformung von `X` in `(-1, *self.resize)` entfernt die Kanalachse der Länge $1$ sowohl aus `(n, 1, h, w)` als auch aus `(n, h, w, 1)`.

<!--en:b3690aa4-->
::: {.slide title="Was die Silhouetten über Modellfehler verraten"}

<!--en:a2cb3498-->
[Fehler anhand der Bilder vorhersagen]{.kicker}

<!--en:b1db23f1-->
Die ersten drei Validierungsbilder jeder Kategorie (eine Spalte pro Klasse) zeigen sofort, wo ein linearer Klassifikator mit erlernten Schablonen Schwierigkeiten hat:

<!--en:6550ae41-->
**Gruppen nach Silhouette:** Hosen (`1`) und Taschen (`8`) haben unverwechselbare Umrisse. Sandalen (`5`), Sneaker (`7`) und Stiefeletten (`9`) bilden eine Gruppe von Schuhen. Die vier Oberteile (**T-Shirt `0`, Pullover `2`, Mantel `4`, Hemd `6`**) haben bei einer Auflösung von $28 \times 28$ nahezu identische kastenförmige Silhouetten.

<!--en:12288d86-->
::: {.slide title="Verständnisfragen zu Formen, Batches und visueller Fehlersuche"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:33f6c107-->
Drei praktische Fragen zur `FashionMNIST`-Pipeline:

<!--en:b16f38c9-->
1. **Anzahl der Minibatches:** Wie viele Minibatches ergeben sich bei der Umstellung auf `batch_size=256` in §{sec_softmax_scratch} pro Trainingsepoche ($60{,}000$ Bilder)? Wie viele Bilder enthält der letzte Minibatch?

<!--en:0d67d6d5-->
2. **Sichtbare und numerische Fehler:** Welche der vier Fehler fallen in einer Bildübersicht von `data.visualize` sofort ins Auge und welche bleiben unbemerkt? *(a)* Um $+1$ verschobene Labels, *(b)* transponierte Bilder $(h \leftrightarrow w)$, *(c)* unnormalisierte Pixelwerte in $[0, 255.0]$, *(d)* jedes Bild kommt pro Epoche doppelt vor.

<!--en:1602c854-->
3. **Invarianz bei Upsampling:** Erhöht `resize=(32, 32)` ($1{,}024$ Pixel durch bilineare Interpolation $\mathbf{x}' = \mathbf{A}\mathbf{x}$) die Ausdrucksstärke eines linearen Softmax-Klassifikators?

<!--en:074c68ab-->
**Lösungen:** **(1)** Aus $60{,}000 = 234 \times 256 + 96$ folgen $235$ Minibatches pro Epoche mit $96$ Bildern im letzten Batch (Validierung: $39 \times 256 + 16 \to 40$ Batches). **(2)** *(a)* und *(b)* sind direkt sichtbar (falsche Titel, seitlich gedrehte Kleidung). *(c)* bleibt verborgen, weil `imshow` die Anzeige anhand von `[vmin, vmax]` automatisch skaliert. Für *(d)* ist `len(dataloader)` zu prüfen. **(3)** Nein. Jeder lineare Score $\mathbf{w}'^\top(\mathbf{A}\mathbf{x}) + b = (\mathbf{A}^\top\mathbf{w}')^\top\mathbf{x} + b$ ist bereits ein linearer Score auf den ursprünglichen $784$ Pixeln.

<!--en:604fe64c-->
::: {.slide title="Zusammenfassung: Datenpipeline für Fashion-MNIST"}

<!--en:27de615d-->
[Zusammenfassung des Abschnitts]{.kicker}

<!--en:a132a410-->
| Stufe der Pipeline | Spezifikation / Form | Bedeutung für System und Modell |

<!--en:29bb5206-->
| **Aufteilung des Datensatzes** | $60{,}000$ Trainingsbilder / $10{,}000$ Validierungsbilder ($6\text{k} / 1\text{k}$ je Klasse) | Gleich viele Bilder in allen $10$ Kategorien, $\approx 188\text{ MB}$ im Format `float32` |

<!--en:530280f7-->
| **Pixelkodierung** | Rohdaten `uint8` $\in [0, 255]$ $\to$ `float32` $\in [0, 1]$ | Dunkler Hintergrund $= 0$. Die Skalierung hält die anfänglichen linearen Logits $\mathbf{W}^\top\mathbf{x}$ nahe bei $0$ |

<!--en:afbfee75-->
| **Minibatchtensor `X`** | Je nach Framework `(n, 1, 28, 28)` (Channel-First) oder `(n, 28, 28, 1)` (Channel-Last) | $n = 64 \to 200\text{ KB}$, $n = 256 \to 803\text{ KB}$. Für lineare Schichten auf `(n, 784)` abgeflacht |

<!--en:9b179642-->
| **Labelvektor `y`** | Eindimensionaler ganzzahliger Tensor `(n,)` mit Werten in $\{0, \ldots, 9\}$ | Enthält direkt den Klassenindex mit Basis $0$. `text_labels` ordnet den Indizes die Namen der Modeartikel zu |

<!--en:9467e747-->
| **Durchlauf einer Epoche** | $938$ Batches bei $n=64$ ($235$ bei $n=256$) | Trainingsdaten in jeder Epoche neu gemischt, letzter unvollständiger Batch beibehalten, alle Daten berücksichtigt |

<!--en:afc26c27-->
| **Durchsatzgrenze** | $T_{\text{epoch}} \ge \max(T_{\text{load}}, T_{\text{compute}})$ | Datenlader stets separat messen, `batch_size` und Parallelität der Arbeitsprozesse anpassen |
