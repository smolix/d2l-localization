<!--en:eff35701-->
# Attention-Scores und Maskierung

<!--en:79bcc1ee-->
In :numref:`sec_attention-pooling` bestimmte ein fester Gauß-Kern (engl. Gaussian kernel) die Attention-Gewichte anhand des euklidischen Abstands im Eingaberaum. In einem Sequenzmodell muss das Modell die Ähnlichkeit jedoch lernen. Sucht ein Verb sein Subjekt, braucht es das Wort mit dieser syntaktischen Rolle. Dessen ursprüngliche Einbettung liegt der des Verbs nicht unbedingt am nächsten. Wollen wir den weichen Zugriff (engl. soft lookup) als trainierbare Schicht eines Netzes umsetzen, stellen sich zwei Fragen. Welche Score-Funktion $a(\mathbf{q}, \mathbf{k})$ soll die Schicht berechnen? Welche Paare aus Abfrage (engl. query) und Schlüssel (engl. key) sollen vom Zugriff ausgeschlossen sein?

<!--en:765f696d-->
Liest ein Attention-Kopf $n = 1{,}024$ Token, wertet er pro Sequenz $n^2 = 1{,}048{,}576$ Abfrage-Schlüssel-Paare aus. Der Score muss sich daher mit wenig Aufwand auf Hardware für Matrizenmultiplikation berechnen lassen. Außerdem müssen wir ungültige Paare vor der Normalisierung ausschließen, damit sie Gewicht null erhalten. Dazu gehören Paare mit Padding-Token in einem Minibatch aus unterschiedlich langen Sequenzen oder mit zukünftigen Token in einem autoregressiven Sprachmodell.

<!--en:360b0c97-->
Wir leiten die *skalierte Skalarprodukt-Attention* (engl. scaled dot-product attention) mit dem Score $\mathbf{q}^\top\mathbf{k}/\sqrt{d}$ her und erklären, wie der Faktor $1/\sqrt{d}$ bei der Initialisierung eine Sättigung der Softmax-Funktion verhindert. Wir implementieren boolesche Masken, verknüpfen sie miteinander und zeigen, wie Attention allein aus den Zielausgaben Alignments zwischen Sequenzen lernt.

<!--en:cc1a6ca7-->
## Skalarprodukt-Attention

<!--en:e084fc70-->
Betrachten wir den Gauß-Kern mit Bandbreite eins aus :numref:`sec_attention-pooling`. Beim Ausmultiplizieren des negativen quadrierten Abstands erhalten wir ein inneres Produkt und zwei Terme mit quadrierten Normen:

<!--en:2d46eee9-->
Da $-\tfrac{1}{2}\|\mathbf{q}\|^2$ für alle Schlüssel $i$ konstant ist, fällt der Term bei der Softmax-Normalisierung weg. Haben alle Schlüssel dieselbe Norm, fällt auch $-\tfrac{1}{2}\|\mathbf{k}_i\|^2$ weg. Lassen wir beide Normterme auch bei unterschiedlichen Schlüsselnormen weg, ergibt sich der *Skalarprodukt-Score* $\mathbf{q}^\top
\mathbf{k}_i$. Der Gauß-Kern bestraft große Werte von $\|\mathbf{k}_i\|$. Beim Skalarprodukt wirkt die Schlüsselnorm dagegen als multiplikativer A-priori-Faktor. Jede gleichgerichtete Abfrage weist einem längeren Schlüssel mehr Gewicht zu (Übung :exref:`ex_queries-keys-values-average-two-values`c).

<!--en:eedb06e3-->
Frühe neuronale Modelle für maschinelle Übersetzung verwendeten stattdessen ein mehrschichtiges Perzeptron (engl. multilayer perceptron, MLP) mit einer versteckten Schicht. Dieses Verfahren heißt *additive Attention* :cite:`Bahdanau.Cho.Bengio.2014`:

<!--en:8f9827da-->
mit den gelernten Matrizen $\mathbf{W}_q \in \mathbb{R}^{r \times d_q}$ und $\mathbf{W}_k \in \mathbb{R}^{r \times d_k}$ sowie dem Gewichtsvektor $\mathbf{w}_v \in \mathbb{R}^{r}$. Additive Scores und Skalarprodukt-Scores benötigen asymptotisch ähnlich viele Rechenoperationen ($O(nmr)$ gegenüber $O(nmd)$ für $n$ Abfragen und $m$ Schlüssel). Die additive Attention muss jedoch die versteckten Aktivierungen in einem Tensor der Form $n \times m
\times r$ speichern und $\tanh$ elementweise anwenden. Die Skalarprodukt-Attention berechnet alle $n \times m$ Scores mit einer einzigen Matrizenmultiplikation $\mathbf{Q}\mathbf{K}^\top$. Ein bilinearer Score ist weniger allgemein als ein MLP. In der Praxis fällt diese Einschränkung jedoch kaum ins Gewicht, wenn gelernte Projektionen die Abfragen und Schlüssel erzeugen: $(\mathbf{W}_q \mathbf{q})^\top (\mathbf{W}_k \mathbf{k})$ (siehe auch Übung :exref:`ex_attention-scoring-additive-near-zero`).

<!--en:71931340-->
Ein entscheidender Skalierungsfaktor fehlt noch. Nehmen wir an, die Koordinaten von $\mathbf{q},
\mathbf{k} \in \mathbb{R}^d$ seien unabhängige Zufallsvariablen mit Mittelwert $0$ und Varianz $1$. Dann hat jeder Term $q_j k_j$ den Mittelwert $0$ und die Varianz $1$. Die Summe $\mathbf{q}^\top \mathbf{k}$ hat somit den Mittelwert $0$ und die Varianz $d$ (Standardabweichung $8$ bei $d = 64$). Für große $d$ können große Skalarprodukte die Softmax-Funktion in Bereiche mit verschwindenden Gradienten treiben. Deshalb teilen :citet:`Vaswani.Shazeer.Parmar.ea.2017` durch $\sqrt{d}$. Damit beträgt die Varianz wieder eins, und wir erhalten das *skalierte Skalarprodukt* als Score:

<!--en:f42d7a71-->
Hier übernimmt $1/\sqrt{d}$ die Rolle der inversen Temperatur $1/\sigma^2$ aus :numref:`sec_attention-pooling`. Der Faktor passt sich automatisch an die Vektorbreite $d$ an. Die Softmax-Normalisierung liefert dann die Attention-Gewichte

<!--en:1adb1c4c-->
und die Ausgabe $\sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\, \mathbf{v}_i$.

<!--en:7bf4dd3f-->
### Ein Rechenbeispiel

<!--en:5b1b13f5-->
Betrachten wir $d = 4$ (also $1/\sqrt{d} = 1/2$) mit zwei Abfragen und drei Schlüssel-Wert-Paaren:

<!--en:f0c0bafc-->
1. *Scores:* Für $\mathbf{q}_1$ ergeben sich mit den drei Schlüsseln die unskalierten Skalarprodukte $(4, 0, 0)$, für $\mathbf{q}_2$ entsprechend $(0, 2, 2)$.

<!--en:505fc8f6-->
2. *Skalierung:* Die Division durch $\sqrt{4} = 2$ ergibt $(2, 0, 0)$ und $(0, 1, 1)$.

<!--en:fdae2e2b-->
3. *Softmax-Funktion:* $\mathbf{q}_1$ weist den Schlüsseln die Gewichte $\big(e^2/(e^2 + 2),\, 1/(e^2 + 2),\, 1/(e^2 + 2)\big) \approx
   (0.787, 0.107, 0.107)$ zu, $\mathbf{q}_2$ die Gewichte $\big(1/(1 + 2e),\, e/(1 + 2e),\, e/(1 + 2e)\big) \approx
   (0.155, 0.422, 0.422)$.

<!--en:3b6ef542-->
4. *Gewichtete Summe:* $\mathbf{o}_1 = 0.787\,\mathbf{v}_1 + 0.107\,\mathbf{v}_2
   + 0.107\,\mathbf{v}_3 \approx (0.893, 0.213)$ und $\mathbf{o}_2 \approx
   (0.578, 0.845)$.

<!--en:916f53f1-->
Selbst ein orthogonaler Schlüssel erhält ein Softmax-Gewicht ungleich null ($0.107$). Stimmen die Schlüssel auf den aktiven Koordinaten einer Abfrage überein ($\mathbf{k}_2$ und $\mathbf{k}_3$ bei $\mathbf{q}_2$), teilen sie das verbleibende Gewicht gleichmäßig unter sich auf. Ohne den Faktor $1/2$ stiege das Gewicht, das $\mathbf{q}_1$ dem Schlüssel $\mathbf{k}_1$ zuweist, von $0.787$ auf $e^4/(e^4 + 2) \approx 0.96$.

<!--en:7502f5a3-->
### Varianz der Scores und der Faktor $1/\sqrt{d}$

<!--en:31410242-->
Übertrifft ein Score alle anderen bei Weitem, nähert sich der Gewichtsvektor $\boldsymbol{\alpha}$ einem One-Hot-Vektor und die Softmax-Funktion *sättigt*. Die Jacobi-Matrix von $\boldsymbol{\alpha} = \mathrm{softmax}(\mathbf{a})$,

<!--en:0e391e57-->
konvergiert gegen die Nullmatrix, wenn sich $\boldsymbol{\alpha}$ einem One-Hot-Vektor nähert (Übung :exref:`ex_attention-scoring-softmax-jacobian`). In einer gesättigten Zeile verschwinden bei der Backpropagation die Gradienten bezüglich der Scores. Dadurch gewinnen nicht ausgewählte Schlüssel beim Training nur schwer an Gewicht.

<!--en:347a7a9e-->
Diesen Effekt können wir an standardnormalverteilten Abfragen messen ($2{,}048$ Abfragen mit je $64$ zufälligen Schlüsseln). Für die Dimensionen $d \in
\{4, \ldots, 512\}$ vergleichen wir unskalierte Scores mit Scores, die wir mit $1/\sqrt{d}$ skalieren. Bei gleichmäßiger Verteilung über $64$ Schlüssel beträgt die Entropie der Gewichte $\log 64 \approx
4.16$ Nat, bei einer One-Hot-Verteilung dagegen $0$.

<!--en:8c243bad-->
Tragen wir die Kopfbreite $d$ von $4$ bis $512$ auf einer logarithmischen Achse auf, sehen wir, wie rasch die Entropie bei unskalierter Skalarprodukt-Attention mit wachsendem $d$ einbricht. Mit dem Faktor $1/\sqrt{d}$ bleibt die Entropie der Attention-Gewichte dagegen über alle Breiten nahezu konstant:

<!--en:3ca69b1c-->
Die folgende Tabelle zeigt für fünf Kopfbreiten die Entropie, das mittlere Maximalgewicht und die mittlere Frobeniusnorm der Jacobi-Matrix der Softmax-Funktion.

<!--en:72db801b-->
Mit dem Faktor $1/\sqrt{d}$ bleibt die Entropie über alle Breiten nahe $\log 64 - \tfrac{1}{2}
\approx 3.66$ Nat (Übung :exref:`ex_attention-scoring-entropy-of-random-scores`). Das maximale Gewicht beträgt $\approx 0.11$, die Norm der Jacobi-Matrix $\approx 0.18$. Ohne Skalierung wächst die Standardabweichung der Scores wie $\sqrt{d}$. Bei $d = 512$ sinkt die Entropie auf $\approx 0.2$ Nat. Schon bei der Zufallsinitialisierung erhält der stärkste Schlüssel über $90\%$ des Gewichts, und die Norm der Jacobi-Matrix halbiert sich. Der Faktor $1/\sqrt{d}$ entkoppelt die anfängliche Schärfe der Softmax-Verteilung von der Kopfbreite. Die gelernten Projektionen können die Gewichte im Training dennoch stärker konzentrieren. Sobald Abfragen und Schlüssel gleichgerichtet sind, wachsen die Skalarprodukte wie $O(d)$. Das motiviert die Skalierung mit $1/d$ bei der Maximal Update Parametrization (:numref:`sec_scaling`). Viele neuere Modelle regeln die Temperatur ganz unabhängig von den Projektionen. Sie normieren Abfragen und Schlüssel auf Länge eins und lernen den Skalierungsfaktor für die Scores als eigenen Parameter. Dieses Verfahren heißt *QK-Norm* und begegnet uns in :numref:`sec_transformer-block` wieder.

<!--en:a2344bb8-->
Durch die Skalierung wird die Schärfe unabhängig von der Breite $d$. Sie hängt aber weiterhin von der Anzahl der Schlüssel $m$ ab. Haben die Scores Varianz eins, beträgt die Entropie etwa $\log m -
\tfrac{1}{2}$. Das Gewicht verteilt sich also auf rund $0.6\,m$ Schlüssel, einen festen Anteil des Kontexts, unabhängig von dessen Länge. Auch beim Zugriff auf einen bestimmten Schlüssel verteilt sich das Gewicht zunehmend auf andere Schlüssel. Übertrifft der passende Schlüssel jeden der übrigen $m - 1$ Schlüssel um den Score-Abstand $g$, erhält er das Gewicht $1/(1 + (m-1)e^{-g})$. Bei einem Abstand von $g = 9$ beträgt das Gewicht etwa $0.9$ unter $1{,}000$ Schlüsseln und weniger als $0.1$ unter $100{,}000$ Schlüsseln. Damit der Zugriff selektiv bleibt, muss der Abstand wie $\log m$ wachsen. Ein bei einer bestimmten Kontextlänge trainiertes Modell verteilt seine Attention bei längerem Kontext daher breiter. Einige Modelle erhöhen deshalb den Skalierungsfaktor für die Scores mit der Kontextlänge (:numref:`sec_attention-further-reading`).

<!--en:c4d98b0a-->
## Maskierung

<!--en:919f31c0-->
Zwei typische Bedingungen schließen bestimmte Abfrage-Schlüssel-Paare von der Attention aus:

<!--en:41702b19-->
1. **Padding:** Füllen wir unterschiedlich lange Sequenzen mit `<blank>`-Token auf, um einen Minibatch zu bilden, darf keine Abfrage auf Padding-Positionen zugreifen.

<!--en:1ba4e8fa-->
2. **Kausalität:** Bei der autoregressiven Sprachmodellierung darf die Abfrage an Position $t$ nur auf Schlüssel an Positionen $j \le t$ zugreifen.

<!--en:9e038af0-->
Beide Nebenbedingungen lassen sich durch eine boolesche *Maske* darstellen. Ihr Tensor enthält genau dort `True`, wo Abfrage $i$ auf Schlüssel $j$ zugreifen darf.

<!--en:f30dd2a1-->
### Die maskierte Softmax-Funktion

<!--en:f87fde03-->
Um die Form dichter Tensoren beizubehalten, ersetzt die *maskierte Softmax-Funktion* ausgeschlossene Scores (`~mask`) vor dem Aufruf von `softmax` durch eine große negative Zahl. Bei der Wahl des Füllwerts müssen wir das Gleitkommaformat berücksichtigen:

<!--en:441091d6-->
* Eine fest vorgegebene Konstante wie $-10^6$ führt in float16 zum Überlauf. Der größte darstellbare endliche Betrag ist $65{,}504$.

<!--en:dad43285-->
* Der Füllwert $-\infty$ ergibt `NaN`, sobald eine ganze Abfragezeile maskiert ist. Die numerisch stabile Softmax-Funktion zieht nämlich das Zeilenmaximum $-\infty$ ab. Dabei entsteht $-\infty - (-\infty) = \mathrm{NaN}$.

<!--en:d0d9c4e6-->
* Verwenden wir den kleinsten *endlichen* Wert des Tensordatentyps (`finfo(dtype).min`), ergibt die Exponentialfunktion für ausgeschlossene Einträge durch Unterlauf exakt null, und zwar in float16, bfloat16 und float32, ohne dass `NaN` entsteht. Für jede Abfragezeile, in der alle Schlüssel maskiert sind (`~mask.any(-1)`), setzt `masked_softmax` die resultierenden Gewichte explizit auf null.

<!--en:4991cabf-->
Da `mask` nur durch Broadcasting mit `X` kompatibel sein muss, genügt für eine gemeinsame Padding-Maske aller Abfragen einer Sequenz die Form `(batch, 1, keys)`.

<!--en:774c6325-->
Betrachten wir einen Batch aus zwei Sequenzen der Länge $4$ mit gültigen Längen $2$ und $3$. Sind alle Scores null, verteilt `masked_softmax` das Gewicht jeder Zeile gleichmäßig über deren gültiges Präfix:

<!--en:7e834401-->
### Kausale Maskierung

<!--en:a39c5cd9-->
Eine kausale Maske erlaubt Abfrage $i$ nur dann den Zugriff auf Schlüssel $j$, wenn $j \le i$ gilt. So entsteht eine untere Dreiecksmatrix, die $n(n-1)/2$ der $n^2$ Paare ausschließt ($523{,}776$ Paare bei $n = 1{,}024$). Gilt `num_queries < num_keys`, wie bei der Generierung mit Cache, ordnet `causal_mask` die Abfragen den letzten Positionen der Sequenz zu.

<!--en:d6d880d9-->
### Verknüpfung von Masken

<!--en:bec7b2ad-->
Gelten mehrere Nebenbedingungen zugleich, ist ein Paar nur gültig, wenn jede von ihnen es zulässt. Boolesche Masken lassen sich daher durch logisches UND (`&`) mit Broadcasting verknüpfen (:numref:`fig_mask-algebra`).

<!--en:cc20a6ce-->
![Masken werden durch logisches UND verknüpft. Für eine Sequenz der Länge 6 mit gültiger Länge 3 speichert die Padding-Maske eine Zeile mit den Schlüsseln. Broadcasting wiederholt sie für jede Abfrage (blass dargestellt). Die kausale Maske bildet ein unteres Dreieck. Die UND-Verknüpfung lässt ein Paar nur zu, wenn beide Masken es zulassen. Die Zeilen stehen für Abfragen, die Spalten für Schlüssel und die ausgefüllten Felder für zulässige Paare.](../img/mdl-attention-mask-algebra.svg)

<!--en:2eb80812-->
Verknüpfen wir bei gleichen Scores eine Padding-Maske (gültige Längen $6$ und $3$) mit einer kausalen Maske, ergibt sich für die erste Sequenz das vollständige kausale Dreieck. In der zweiten Sequenz enden alle Zeilen spätestens bei Schlüssel $3$:

<!--en:ae1132a4-->
In der kürzeren Sequenz liegen die Abfragen $4$ bis $6$ selbst im Padding. Dennoch hat jede gültige Schlüssel, denn ein kausales Dreieck reicht stets bis zu Schlüssel $1$ zurück. Das liegt an der Form der Maske. Bei einer UND-Verknüpfung ist das nicht generell der Fall. Verknüpfen wir die Padding-Maske etwa mit einem *gleitenden Fenster* (engl. sliding window), sieht jede Abfrage nur ihre wenigen jüngsten Schlüssel. Liegt ihr Fenster ganz im Padding, bleibt kein gültiger Schlüssel übrig. Zwei Masken, die für sich genommen unproblematisch sind, können also zusammen eine leere Zeile ergeben. Genau diese Zeilen setzt `masked_softmax` auf null.

<!--en:8ad5fdf3-->
Dieselbe boolesche Algebra eignet sich für *gepackte Sequenzen* (engl. packed sequences). Dabei verketten wir mehrere kurze Dokumente zu einer einzigen Zeile, um den Mehraufwand durch Padding zu vermeiden. Der Ausdruck `doc[:, None] == doc[None, :]` zeigt an, ob zwei Positionen zum selben Dokument gehören. Seine UND-Verknüpfung mit `causal_mask` ergibt eine blockdiagonale kausale Maske:

<!--en:03ac08ca-->
## Attention für einen Minibatch

<!--en:041d0186-->
Fassen wir $n$ Abfragen in $\mathbf{Q} \in \mathbb{R}^{n \times d}$, $m$ Schlüssel in $\mathbf{K} \in \mathbb{R}^{m \times d}$ und $m$ Werte in $\mathbf{V} \in \mathbb{R}^{m \times d_v}$ zusammen, lässt sich die skalierte Skalarprodukt-Attention mit zwei Matrizenmultiplikationen und einer zeilenweisen Softmax-Funktion berechnen (:numref:`fig_attention-shapes`):

<!--en:6e5479e4-->
![Tensorformen bei der skalierten Skalarprodukt-Attention. $\mathbf{Q}\mathbf{K}^\top/\sqrt{d}$ enthält die Scores aller $n \times m$ Abfrage-Schlüssel-Paare. Maskierung und zeilenweise Anwendung der Softmax-Funktion ergeben die Gewichte $\mathbf{A}$ (hier mit einer kausalen Maske, die das obere Dreieck leer lässt). Die Multiplikation mit $\mathbf{V}$ liefert für jede Abfrage eine gewichtete Summe der Werte.](../img/mdl-attention-shapes.svg)

<!--en:c9b7dd2c-->
Bei Eingaben mit vorangestellten Batch- und gegebenenfalls Kopfachsen arbeitet das Matrixprodukt `@` auf den letzten beiden Dimensionen. Die vorangestellten Achsen gleicht es automatisch durch Broadcasting an. Für einen Kopf von GPT-2 Small ($n = m = 1{,}024$, $d = d_v =
64$) enthält die Score-Matrix $1{,}048{,}576$ Gleitkommazahlen (4 MiB in fp32). Die beiden Produkte kosten $4n^2 d \approx 0.27$ GFLOPs. Sowohl Speicherbedarf als auch Rechenaufwand wachsen quadratisch mit $n$.

<!--en:e41b5f81-->
### Die Klasse DotProductAttention

<!--en:639f1439-->
`DotProductAttention` implementiert :eqref:`eq_softmax_QK_V` mit optionaler Maskierung und Dropout für die Attention-Gewichte (Standardwert $0.0$). Die zuletzt berechneten Attention-Gewichte speichert die Klasse zur späteren Untersuchung in `attention_weights`.

<!--en:e3a903ad-->
Wenden wir `DotProductAttention` auf die Vektoren aus :numref:`subsec_attention-worked-example` an, erhalten wir dasselbe Ergebnis wie bei der Rechnung von Hand:

<!--en:375cc14a-->
### Zuordnung der Buchstaben zweier Wörter

<!--en:2cc0b42d-->
Was kann ein rein inhaltlicher Vergleich ohne Projektion unterscheiden? Dazu weisen wir jedem unterschiedlichen Buchstaben in `"attention"` und `"translation"` eine zufällige standardnormalverteilte Einbettung $\mathbf{e} \in \mathbb{R}^{32}$ zu. Die Buchstaben von `"attention"` dienen als Abfragen an die Buchstaben von `"translation"`. Für gleiche Buchstaben beträgt der erwartete skalierte Score $\mathbb{E}[\|\mathbf{e}\|^2]/\sqrt{32} = \sqrt{32}
\approx 5.7$. Für unterschiedliche Buchstaben hat der Score den Erwartungswert $0$ und die Varianz eins.

<!--en:a795a189-->
Drei Fälle fallen auf:

<!--en:01396aaa-->
1. Bei einem eindeutigen Treffer (`i`, `o`) entfällt fast das gesamte Gewicht auf den passenden Buchstaben.

<!--en:0eab7449-->
2. Bei mehreren Treffern (`a`, `t`, `n`) verteilt sich das Gewicht gleichmäßig auf alle Vorkommen des Buchstabens in `"translation"`. Rein inhaltliche Schlüssel enthalten keine Information über die Position.

<!--en:4c759120-->
3. Abfragen ohne Treffer (`e`) verteilen ihre Wahrscheinlichkeitsmasse auf zufällige Schlüssel. Die Softmax-Gewichte müssen sich auch dann zu eins summieren, wenn es keinen relevanten Schlüssel gibt. In trainierten Sprachmodellen nimmt oft das erste Token als *Attention-Sink* diese ungenutzte Masse auf (:cite:`Xiao.Tian.Chen.ea.2024`).

<!--en:63642c6b-->
## Attention als gelerntes Alignment

<!--en:c9e9ee37-->
Lernbare Attention entstand in der neuronalen maschinellen Übersetzung :cite:`Bahdanau.Cho.Bengio.2014`. Sie sollte den Flaschenhals überwinden, der in Sequenz-zu-Sequenz-RNNs durch einen Vektor fester Größe entsteht (:numref:`sec_seq2seq`). Bei jedem Dekodierschritt fragte der Decoder alle Encoder-Zustände ab. Dabei bildeten die gelernten Attention-Gewichte von selbst Alignments zwischen Wörtern der Ausgangs- und Zielsprache ab. Es zeigte sich eine monoton verlaufende Diagonale mit Kreuzungen abseits der Diagonalen, wo sich die Wortstellung unterschied (etwa englisch *„black cat“* und französisch *„chat noir“*).

<!--en:3c0d6ec8-->
Wie ein Alignment entsteht, können wir am Minimalmodell `ToyAligner` beobachten. Jede Quellsequenz besteht aus $8$ Token aus einem Vokabular der Größe $16$. Die Zielsequenz entsteht durch eine feste Permutation $\pi$, die die Positionen $3$ und $4$ vertauscht. Jede Ausgabeposition $j$ hat einen gelernten Abfragevektor. Für jeden Schlüssel verketten wir die One-Hot-Kodierungen von Quelltoken und Quellposition und projizieren das Ergebnis linear. Als Werte dienen die One-Hot-kodierten Quelltoken.

<!--en:6c8b229e-->
Wir trainieren `ToyAligner` mit Adam über 100 Schritte und minimieren dabei die negative Log-Likelihood der permutierten Zieltoken.

<!--en:16659da1-->
Wir vergleichen ein untrainiertes Modell mit drei unabhängigen Trainingsläufen auf $1{,}000$ Testsequenzen. Mit Gradientenabstieg erreicht das Modell zuverlässig eine Genauigkeit von $100\%$. Im Mittel über die Ausgabepositionen legt es mehr als $99\%$ des Gewichts auf $\pi(j)$:

<!--en:866a051d-->
An der Matrix der Attention-Gewichte für die erste Testsequenz vor und nach dem Training erkennen wir, wie `ToyAligner` aus einer anfangs unstrukturierten Verteilung die genaue Permutationsmatrix von $\pi$ lernt. Die Vertauschung an den Positionen $3$ und $4$ ist an den Einträgen außerhalb der Diagonalen sichtbar:

<!--en:4372bdd8-->
Die Heatmap der trainierten Attention-Gewichte gibt die Permutationsmatrix von $\pi$ genau wieder, selbst wenn Token in der Quellsequenz mehrfach vorkommen. Die Schlüssel kodieren nämlich neben dem Token auch dessen Position. Bei der Sequenzmodellierung im Allgemeinen müssen die Abfragen jedoch aus der Sequenz selbst hervorgehen. Eine einzelne Position muss unter Umständen mehrere verschiedene Werte gleichzeitig abrufen (:numref:`sec_multihead-attention`), und Positionskodierungen müssen auf andere Sequenzlängen verallgemeinern (:numref:`sec_positional-information`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:a38a4d0f-->
Das skalierte Skalarprodukt $a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k} / \sqrt{d}$ verzichtet auf die quadratischen Normterme des Gauß-Kerns und berechnet die Scores aller Paare mit einer einzigen Matrizenmultiplikation $\mathbf{Q}\mathbf{K}^\top$. Die Division durch $\sqrt{d}$ sorgt dafür, dass die Scores bei Eingaben mit Varianz eins ebenfalls die Varianz eins haben. So verhindern wir bei wachsender Kopfbreite $d$ eine Sättigung der Softmax-Funktion und verschwindende Score-Gradienten. Strukturelle Nebenbedingungen wie Padding von Sequenzen und kausales Dekodieren setzen wir mit booleschen Masken um, die wir durch logisches UND verknüpfen. Mit dem Füllwert `finfo(dtype).min` und dem expliziten Nullsetzen leerer Zeilen vermeiden wir einen Überlauf in `float16` und die Ausbreitung von `NaN`. Rein inhaltliche Schlüssel verteilen die Attention bei mehrfach vorkommenden Token gleichmäßig über deren Positionen. Schlüssel mit Positionsinformation ermöglichen dagegen das End-to-End-Lernen von Sequenzalignments.

<!--en:2294243a-->
## Übungen

<!--en:e148c7aa-->
1. **Invarianzen des Scores.** Der Gauß-Score :eqref:`eq_gaussian-expansion` und das skalierte Skalarprodukt :eqref:`eq_dot_product_attention` bestimmen die Attention-Gewichte jeweils über die Softmax-Funktion :eqref:`eq_attn-scoring-alpha`.
    1. Bestimmen Sie für beide Scores, welche der folgenden Transformationen die Gewichte unverändert lassen. Addieren Sie denselben Vektor $\mathbf{c}$ zu jedem Schlüssel. Addieren Sie $\mathbf{c}$ sowohl zur Abfrage als auch zu jedem Schlüssel. Multiplizieren Sie die Abfrage und jeden Schlüssel mit derselben orthogonalen Matrix.
    1. Abfragen und Schlüssel entstehen meist durch affine Abbildungen der Eingaben, $\mathbf{q} = \mathbf{W}_q \mathbf{x} + \mathbf{b}_q$ und $\mathbf{k}_i = \mathbf{W}_k \mathbf{y}_i + \mathbf{b}_k$. Multiplizieren Sie $\mathbf{q}^\top \mathbf{k}_i$ aus, sodass vier Terme entstehen. Zeigen Sie, dass die Gewichte nicht von $\mathbf{b}_k$ abhängen. Welchen Gradienten hat der Verlust bezüglich $\mathbf{b}_k$?
    1. Über welchen der vier Terme beeinflusst $\mathbf{b}_q$ die Gewichte? Welche Präferenz für bestimmte Schlüssel lässt sich damit ausdrücken, die ein Score ohne Bias nicht erfassen kann?

<!--en:32df9b25-->
1. [code] **Jacobi-Matrix der Softmax-Funktion.** `jacobian_norm` misst die Frobeniusnorm der Jacobi-Matrix :eqref:`eq_softmax-jacobian`.
    1. Leiten Sie :eqref:`eq_softmax-jacobian` aus :eqref:`eq_attn-scoring-alpha` her. Zeigen Sie, dass der Einsvektor bei endlichen Scores den Nullraum der Jacobi-Matrix aufspannt. Prüfen Sie die Formel für einen zufälligen Score-Vektor durch automatisches Differenzieren.
    1. Berechnen Sie die Norm, wenn sich das Gewicht gleichmäßig auf $k$ der $m$ Schlüssel verteilt. Bestimmen Sie das $k$, für das die Norm maximal ist. Ziehen Sie zufällige Gewichtsvektoren und prüfen Sie, dass in keiner Stichprobe die Norm dieses Maximum überschreitet. Erklären Sie damit, warum die Norm ohne Skalierung in der Tabelle von $d = 4$ bis $d = 16$ ansteigt und dann fällt.
   :exlabel:`ex_attention-scoring-softmax-jacobian`

<!--en:d9a976bb-->
1. **Vollständig maskierte Zeilen.** `masked_softmax` überschreibt ausgeschlossene Scores mit dem kleinsten endlichen Wert des Datentyps. Anschließend setzt die Funktion die Zeilen der Abfragen ohne gültigen Schlüssel auf null.
    1. Erstellen Sie eine Zeile mit vier Scores, in der alle Schlüssel ausgeschlossen sind. Füllen Sie die Zeile einmal mit $-10^6$ in float16, einmal mit $-\infty$ in float32 und einmal mit `finfo(dtype).min` in float32. Wenden Sie jeweils die Softmax-Funktion des Frameworks an und lassen Sie das abschließende Nullsetzen von `masked_softmax` weg. Geben Sie die Ergebnisse an und erklären Sie jedes anhand der Subtraktion des Zeilenmaximums.
    1. Betrachten Sie eine Self-Attention-Schicht, deren Abfragen, Schlüssel und Werte alle aus derselben Sequenz stammen. Eine Abfrage an einer Padding-Position habe keinen gültigen Schlüssel, wie es bei den verknüpften Masken aus :numref:`subsec_composing-masks` vorkommen kann. Die ausgeschlossenen Scores werden mit $-\infty$ überschrieben, sodass die Ausgabe an dieser Position NaN ist. Diese Ausgabe wird verworfen. In der nächsten Attention-Schicht schließt jede gültige Abfrage die Padding-Position als Schlüssel aus. Erklären Sie, warum der NaN-Wert dennoch in der nächsten Schicht die Ausgaben an allen gültigen Positionen der Sequenz verfälscht.

<!--en:f8004b14-->
1. **Verknüpfte Masken.** In :numref:`subsec_composing-masks` verknüpfen wir Masken durch logisches UND. Selbst wenn jede einzelne Maske für sich unproblematisch ist, kann bei der Verknüpfung eine Abfrage ohne gültigen Schlüssel bleiben. Ein kausales gleitendes Fenster der Breite $w$ erlaubt der Abfrage an Position $t$ den Zugriff auf die Schlüssel an den Positionen $t - w + 1, \ldots,
   t$.
    1. Schreiben Sie für eine Sequenz der Länge $6$ mit gültiger Länge $3$ die Padding-Maske und das kausale gleitende Fenster der Breite $2$ als boolesche Matrizen der Form $6 \times 6$ auf. Prüfen Sie, dass jede Maske für jede Abfrage mindestens einen Schlüssel zulässt. Bestimmen Sie die Abfragen, deren Zeile nach der UND-Verknüpfung leer ist.
    1. Zeigen Sie, dass die UND-Verknüpfung keine leere Zeile enthält, wenn jede Maske jeder Abfrage den Zugriff auf ihre eigene Position erlaubt. Welche der Masken für Padding, Kausalität, gleitende Fenster und gepackte Sequenzen erfüllen diese Bedingung? Was sagt das über die gefundenen leeren Zeilen aus?
    1. Manche Implementierungen addieren eine Bias-Matrix mit Einträgen $0$ für zulässige und $c < 0$ für ausgeschlossene Paare. Zur Verknüpfung der Masken addieren sie deren Bias-Terme. Wählen Sie für $c$ den kleinsten endlichen Wert in einfacher Genauigkeit, etwa $-3.4 \times 10^{38}$. Berechnen Sie die Summe zweier solcher Bias-Terme in einfacher Genauigkeit. Bestimmen Sie die Ausgabe der Softmax-Funktion für eine Zeile, in der beide Masken jeden Schlüssel ausschließen. Bestimmen Sie die Ausgabe auch für eine Zeile, in der alle Schlüssel ausgeschlossen sind, einige jedoch nur durch eine der Masken.
   :exlabel:`ex_attention-scoring-composite-masks`

<!--en:ef81a0c6-->
1. **Entropie zufälliger Scores.** ● Die oben dargestellte Entropiekurve für skalierte Scores bleibt nahe $3.7$ Nat, unterhalb des Werts $\log 64 \approx 4.16$ für die Gleichverteilung. Halten Sie eine Abfrage $\mathbf{q} \in \mathbb{R}^d$ fest und ziehen Sie $m$ Schlüssel mit unabhängigen standardnormalverteilten Koordinaten, wie es `softmax_weights` tut.
    1. Zeigen Sie, dass die skalierten Scores $a_i = \mathbf{q}^\top \mathbf{k}_i /
       \sqrt{d}$ unabhängige normalverteilte Zufallsvariablen mit Mittelwert $0$ und Varianz $s^2 = \|\mathbf{q}\|^2 / d$ sind. Berechnen Sie den Erwartungswert von $s^2$, wenn Sie $\mathbf{q}$ wie in `softmax_weights` ziehen.
    1. Die Entropie der Gewichte beträgt $H = \log Z - \sum_i \alpha_i a_i$ mit $Z = \sum_j \exp(a_j)$. Ersetzen Sie für große $m$ die Größen $Z/m$ und $\sum_i \exp(a_i)\, a_i / m$ durch ihre Erwartungswerte und leiten Sie folgende Näherung her:
       $$H \approx \log m - \frac{s^2}{2}.$$
       Werten Sie die Näherung für $m = 64$ und $s = 1$ aus.
    1. Die Näherung setzt voraus, dass sich $Z$ um seinen Mittelwert konzentriert. Berechnen Sie die relative Standardabweichung von $Z$ als Funktion von $s$ und $m$. Leiten Sie daraus eine Bedingung an $s^2$ ab, unter der die Näherung gilt. Für unskalierte Scores gilt $s^2 = \|\mathbf{q}\|^2$ mit Erwartungswert $d$. Berechnen Sie die relative Standardabweichung für $m = 64$ bei $s^2 = 4$ und $s^2 = 8$. Vergleichen Sie die Näherung mit den oben dargestellten Entropien für unskalierte Scores bei $d = 4$ und $d = 8$.
   :exlabel:`ex_attention-scoring-entropy-of-random-scores`

<!--en:3b849c69-->
1. [code] **Additive Scores nahe null.** ● Beim additiven Score :eqref:`eq_additive-attn` steht die Summe aus projizierter Abfrage und projiziertem Schlüssel im Argument von $\tanh$.
    1. Ersetzen Sie $\tanh$ durch die Identität und zeigen Sie, dass die Attention-Gewichte dann nicht von der Abfrage abhängen.
    1. Entwickeln Sie $\tanh$ bis zur dritten Ordnung um null. Bestimmen Sie die Terme niedrigster Ordnung, über die die Gewichte von der Abfrage abhängen. Geben Sie an, wie diese Abhängigkeit skaliert, wenn Sie $\mathbf{W}_q$ und $\mathbf{W}_k$ mit demselben kleinen Faktor $\epsilon$ multiplizieren.
    1. Implementieren Sie den additiven Score für $3$ Abfragen und $6$ Schlüssel in $\mathbb{R}^8$ mit standardnormalverteilten Einträgen und einer versteckten Schicht der Breite $r = 16$. Ziehen Sie für $\mathbf{W}_q$, $\mathbf{W}_k$ und $\mathbf{w}_v$ standardnormalverteilte Einträge und teilen Sie diese jeweils durch die Quadratwurzel der zugehörigen Eingabebreite. Multiplizieren Sie $\mathbf{W}_q$ und $\mathbf{W}_k$ mit $\epsilon \in \{1, 0.3, 0.1, 0.03\}$. Messen Sie die größte Differenz zwischen den Gewichtszeilen der drei Abfragen und mitteln Sie diese über viele Ziehungen. Prüfen Sie damit die hergeleitete Skalierung. Bei welchen Faktoren hängen die Gewichte merklich von der Abfrage ab?
   :exlabel:`ex_attention-scoring-additive-near-zero`

<!--en:c83eee72-->
1. [code] **Varianten des Alignments.** ● `ToyAligner` aus :numref:`subsec_learned-alignment` lernt die Permutation $\pi$ mithilfe einer gelernten Abfrage je Ausgabeposition und mit Schlüsseln, die Token und Position kodieren.
    1. Entfernen Sie die One-Hot-Kodierung der Position aus den Schlüsseln, sodass diese nur vom Token abhängen. Sagen Sie vor dem Training voraus, ob die Ausgabeposition $j$ ihr Gewicht noch auf $\pi(j)$ konzentrieren kann und welche Genauigkeit das Modell ungefähr erreicht. Trainieren Sie dann das Modell und vergleichen Sie das Ergebnis mit Ihrer Vorhersage. Setzen Sie es in Beziehung zu den Buchstaben in :numref:`subsec_cross-attention-alignment`.
    1. Ersetzen Sie den Score durch die additive Regel :eqref:`eq_additive-attn`. Vergleichen Sie die Anzahl der Schritte, die jedes Modell bis zu einer Genauigkeit von $99\%$ benötigt. Sagen Sie vor dem Training anhand von Übung :exref:`ex_attention-scoring-additive-near-zero` voraus, wie sich das additive Modell bei der Initialisierung seiner Projektionen mit kleinen Werten anfangs verhält. Prüfen Sie Ihre Vorhersage.
    1. Geben Sie den Abfragen und Schlüsseln die Breite $d \in \{16, 256, 1024\}$ und initialisieren Sie sie mit standardnormalverteilten Werten. Trainieren Sie mit und ohne den Faktor $1/\sqrt{d}$. Setzen Sie die Trainingskurven in Beziehung zur Entropie der Gewichte bei der Initialisierung.
    1. Zeigen Sie, dass `ToyAligner` keine Umordnung lernen kann, die von den Token abhängt. Ein Beispiel wäre das Vertauschen zweier Nachbarn nur dann, wenn der erste zu einer vorgegebenen Teilmenge des Vokabulars gehört. (Hinweis: Der Score ist die Summe aus einem für alle Positionen gemeinsamen Tokenterm und einem Positionsterm. Vergleichen Sie die Scores der beiden Nachbarn.) Geben Sie an, wovon die Abfrage oder die Schlüssel abhängen müssten.

<!--en:647c3ed8-->
[Diskussion](https://d2l.discourse.group/t/36769)

<!--en:6eabebea-->
Score-Funktionen für Attention, skaliertes Skalarprodukt und Maskierung<br>**Skalarprodukt und additive Scores · Varianzskalierung mit 1/√d · Jacobi-Matrix der Softmax-Funktion · Algebra boolescher Masken · gelerntes Alignment**

<!--en:12f3d0f5-->
::: {.slide title="Zwei Anforderungen an eine trainierbare Attention-Schicht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:823659a1-->
- **1. Schnelle, lernbare Score-Funktion ($n^2$ Paare):** Ein einzelner Attention-Kopf liest $n = 1{,}024$ Token und bewertet pro Schicht $1{,}024^2 = 1{,}048{,}576$ Abfrage-Schlüssel-Paare. Die Score-Funktion $a(\mathbf{q}, \mathbf{k})$ muss sich direkt auf GPU-Hardware für dichte Matrizenmultiplikation abbilden lassen.

<!--en:466d31de-->
- **2. Ausschluss ungültiger Paare (Maskierung):** Aufgefüllte Minibatches enthalten `<blank>`-Token. Kausale Sprachmodelle müssen $n(n-1)/2 = 523{,}776$ Paare mit zukünftigen Schlüsseln sperren, damit Position $t$ keine Token an Positionen $> t$ lesen kann.

<!--en:3bb14460-->
::: {.slide title="Vom Abstand im Gauß-Kern zum Skalarprodukt-Score"}

<!--en:74227c94-->
[Herleitung des Scores]{.kicker}

<!--en:a63b8a29-->
- **Abstand im Gauß-Kern ausmultipliziert:**
  $$-\frac{1}{2}\|\mathbf{q} - \mathbf{k}_i\|^2 = \mathbf{q}^\top \mathbf{k}_i - \frac{1}{2}\|\mathbf{k}_i\|^2 - \frac{1}{2}\|\mathbf{q}\|^2$$

<!--en:19f1b5e3-->
- **Die Norm der Abfrage fällt weg:** $-\frac{1}{2}\|\mathbf{q}\|^2$ ist für alle Schlüssel $i$ konstant. Der Term fällt wegen der Verschiebungsinvarianz der Softmax-Funktion weg.

<!--en:6c88bd63-->
- **Schlüssellänge als A-priori-Faktor:** Ohne $-\frac{1}{2}\|\mathbf{k}_i\|^2$ ergibt sich $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$. Damit wirkt $\|\mathbf{k}_i\|$ als multiplikativer A-priori-Faktor für die Relevanz.

<!--en:920268df-->
::: {.slide title="Additive Scores (MLP) und Skalarprodukt-Scores"}

<!--en:5c3066c9-->
[Hardwareeffizienz]{.kicker}

<!--en:aed405b2-->
- **Additive Attention (Bahdanau et al., 2014):** $a_{\textrm{add}}(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k})$ benötigt als Zwischenergebnis einen dreidimensionalen Aktivierungstensor der Form $(n, m, r)$.

<!--en:21b28dfd-->
- **Problem bei kleinen Gewichten:** Nahe $0$ ist $\tanh(u) \approx u$ näherungsweise linear. Die Abfrageterme $\mathbf{W}_q\mathbf{q}$ fallen daher in der Softmax-Funktion beim Vergleich der Schlüssel näherungsweise weg.

<!--en:1284c47d-->
- **Skaliertes Skalarprodukt ($\mathbf{Q}\mathbf{K}^\top / \sqrt{d}$):** Eine einzige GEMM-Operation auf Tensor Cores berechnet alle $n \times m$ Scores. Sie koppelt $\mathbf{q}$ und $\mathbf{k}$ bei jeder Skalierung bilinear.

<!--en:be5ae169-->
::: {.slide title="Warum unskalierte Skalarprodukte wie √d wachsen"}

<!--en:5b01b6a8-->
[Kalibrierung der Varianz]{.kicker}

<!--en:8cae2414-->
- **Lineares Wachstum der Varianz:** Für unabhängige, zentrierte Koordinaten $q_j, k_j$ mit Varianz eins gilt:
  $$\mathbb{E}[\mathbf{q}^\top \mathbf{k}] = 0, \qquad \textrm{Var}(\mathbf{q}^\top \mathbf{k}) = d$$

<!--en:744c09b4-->
- **Zunehmende Streuung:** Unskalierte Skalarprodukte haben die Standardabweichung $\sqrt{d}$ ($8$ bei $d = 64$ und $22.6$ bei $d = 512$). Dadurch gerät die Softmax-Funktion in Sättigung.

<!--en:665ea0f3-->
- **Automatische Temperaturanpassung ($1/\sqrt{d}$):** Die Division durch $\sqrt{d}$ stellt für alle Kopfbreiten $d$ die Varianz eins der Scores wieder her.

<!--en:a8c64602-->
::: {.slide title="Rechenbeispiel zum skalierten Skalarprodukt"}

<!--en:525a8730-->
[Schritt für Schritt]{.kicker}

<!--en:219d3499-->
- **Ausgangslage ($d = 4$, Skalierungsfaktor $1/\sqrt{4} = 1/2$):** Die Abfragen $\mathbf{q}_1 = (2,2,0,0)$ und $\mathbf{q}_2 = (0,0,2,0)$ greifen auf die Schlüssel $\mathbf{k}_1 = (1,1,0,0)$, $\mathbf{k}_2 = (0,0,1,1)$ und $\mathbf{k}_3 = (0,0,1,-1)$ zu.

<!--en:085448f3-->
- **Auch orthogonale Schlüssel erhalten Gewicht:** Die Schlüssel $\mathbf{k}_2, \mathbf{k}_3$ haben das Skalarprodukt $0$ mit $\mathbf{q}_1$. Wegen $e^0 = 1$ erhält dennoch jeder das Gewicht $10.7\%$.

<!--en:d7a137ef-->
- **Wirkung der Skalierung:** Ohne den Faktor $1/2$ steigt das Gewicht von $\mathbf{q}_1$ für $\mathbf{k}_1$ von $0.787$ auf $0.965$.

<!--en:20344e8d-->
::: {.slide title="Sättigung und Jacobi-Matrix der Softmax-Funktion"}

<!--en:1edfba85-->
[Gradientendynamik]{.kicker}

<!--en:8dd8a272-->
- **Symmetrische Jacobi-Matrix der Softmax-Funktion:**
  $$\frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{a}} = \mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha}\boldsymbol{\alpha}^\top$$

<!--en:2ec8c15e-->
- **Verschwindende Einträge an One-Hot-Eckpunkten:** Für $\boldsymbol{\alpha} \to \mathbf{e}_{i^*}$ gilt auf der Diagonale $\alpha_i(1 - \alpha_i) \to 0$ und außerhalb der Diagonale $-\alpha_i \alpha_j \to 0$.

<!--en:bd314f0b-->
- **Folge für das Training:** In einer gesättigten Zeile verschwinden bei der Backpropagation die Score-Gradienten $\partial \mathcal{L} / \partial \mathbf{a}$. Nicht ausgewählte Schlüssel erhalten so keine Aktualisierung.

<!--en:06d3586d-->
::: {.slide title="Einbruch der Entropie bei zunehmender Kopfbreite"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:06c69923-->
- **Versuchsaufbau:** $2{,}048$ standardnormalverteilte Abfragen greifen auf $m = 64$ standardnormalverteilte Schlüssel zu. Die Breiten sind $d \in \{4, \ldots, 512\}$.

<!--en:0fcbacf4-->
- **Konstante Entropie mit Skalierung:** Mit $1/\sqrt{d}$ bleibt die Entropie der Gewichte bei allen Breiten nahe $\log 64 - \frac{1}{2} \approx 3.66$ Nat. Das entspricht effektiv $\approx 39$ Schlüsseln.

<!--en:fd6ad26a-->
- **Einbruch ohne Skalierung:** Ohne $1/\sqrt{d}$ fällt die Entropie auf $\approx 0.2$ Nat bei $d = 512$. Schon vor dem ersten Gradientenschritt liegt fast das gesamte Gewicht auf einem Schlüssel.

<!--en:5f2ccfe6-->
::: {.slide title="Maximalgewicht und Norm der Jacobi-Matrix in Abhängigkeit von d"}

<!--en:2a7ab2ed-->
[Quantitativer Vergleich]{.kicker}

<!--en:7e57ca84-->
Der Vergleich von mittlerem Maximalgewicht $\max_i \alpha_i$ und mittlerer Frobeniusnorm $\|\mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha}\boldsymbol{\alpha}^\top\|_F$ bestätigt, dass $1/\sqrt{d}$ die Gradienten stabilisiert:

<!--en:1a0f193b-->
Warum steigt die Norm der Jacobi-Matrix ohne Skalierung von $d = 4$ ($\approx 0.26$) auf $d = 16$ ($\approx 0.29$), bevor sie sinkt? $\|\mathbf{J}\|_F$ ist maximal, wenn sich das Gewicht gleichmäßig auf $k = 2$ konkurrierende Schlüssel verteilt ($\|\mathbf{J}\|_F = 1/2$). Dieser Fall tritt vorübergehend beim Übergang von der Gleichverteilung ($m = 64$) zur One-Hot-Verteilung ($k = 1$) auf.

<!--en:393e1afa-->
::: {.slide title="Warum Sequenzmodelle boolesche Attention-Masken brauchen"}

<!--en:f9fefd89-->
[Attention-Maskierung]{.kicker}

<!--en:1c74b1e4-->
Die dichte Matrizenmultiplikation auf der GPU wertet alle $n \times m$ Abfrage-Schlüssel-Paare gleichzeitig aus. Um strukturelle Nebenbedingungen einzuhalten, schließt eine Maske unzulässige Paare vor der Softmax-Normalisierung aus:

<!--en:fd603f17-->
- **1. Padding-Masken (Batches mit unterschiedlichen Sequenzlängen):** Sequenzen der Längen $2, 3, 4$ lassen sich mit `<blank>`-Token zu einem gemeinsamen Tensor der Form `(batch, 4)` auffüllen. Keine Abfrage darf einen `<blank>`-Schlüssel lesen.

<!--en:cd085010-->
- Durch Broadcasting der Form `(batch, 1, keys)` gilt für jede Abfrage einer Sequenz dasselbe Präfix gültiger Schlüssel.

<!--en:f26f02fa-->
- **2. Kausale Masken (autoregressives Dekodieren):** Bei der Vorhersage von Token $t + 1$ aus den Positionen $\le t$ darf Abfrage $i$ nur dann auf Schlüssel $j$ zugreifen, wenn $j \le i$ gilt.

<!--en:1386f54c-->
- Eine boolesche untere Dreiecksmatrix der Form `(queries, keys)` schließt alle $n(n-1)/2$ Paare mit zukünftigen Schlüsseln im oberen Dreieck aus.

<!--en:9d84bd89-->
::: {.slide title="Wahl des Füllwerts für die maskierte Softmax-Funktion"}

<!--en:15f6abbb-->
[Fehlerquellen bei Gleitkommazahlen]{.kicker}

<!--en:8aa8c519-->
- **Fest vorgegebenes `-1e6`:** Bei `float16` entsteht ein Überlauf. Der größte endliche Betrag ist $65{,}504$.

<!--en:c273622f-->
- **Der Füllwert `-inf`:** In jeder vollständig maskierten Zeile (`all False`) beträgt das Zeilenmaximum $-\infty$. Beim Abziehen des Maximums ergibt sich $-\infty - (-\infty) = \mathrm{NaN}$. In der GEMM-Operation der nächsten Schicht breitet sich dieser Wert auf alle Token aus.

<!--en:8cc88e4e-->
- **`finfo(dtype).min` und explizites Nullsetzen:** Bei ausgeschlossenen Einträgen führt ein Unterlauf der Exponentialfunktion zu `0.0` in `float16`, `bfloat16` und `float32`. Zusätzlich setzt `where(mask.any(-1))` leere Zeilen auf null.

<!--en:62edf891-->
::: {.slide title="Numerisch sichere maskierte Softmax-Funktion und Padding-Masken"}

<!--en:ce0f4436-->
[Implementierung]{.kicker}

<!--en:adab0fa1-->
`masked_softmax` setzt ausgeschlossene Einträge auf `finfo(X.dtype).min` und die Gewichte jeder Abfragezeile ohne gültigen Schlüssel explizit auf null:

<!--en:cb017247-->
::: {.slide title="Verknüpfung boolescher Masken durch logisches UND"}

<!--en:89b6052d-->
[Verknüpfung von Masken]{.kicker}

<!--en:650106e1-->
- **Regel für die Konjunktion:** Bei mehreren gleichzeitig geltenden Nebenbedingungen darf Abfrage $i$ genau dann auf Schlüssel $j$ zugreifen, wenn *jede* Nebenbedingung $(i, j)$ zulässt:
  $$\texttt{mask} = \texttt{padding\_mask} \;\mathbin{\&}\; \texttt{causal\_mask}$$

<!--en:d060c208-->
- **Schnittmenge mit Broadcasting:** Eine Padding-Maske der Form `(batch, 1, n)` und eine kausale Maske der Form `(n, n)` ergeben unter `&` durch Broadcasting eine Maske der Form `(batch, n, n)`.

<!--en:aebfc18b-->
![Padding-Maske und kausale Maske werden durch logisches UND mit Broadcasting verknüpft.](../img/mdl-attention-mask-algebra.svg){.nostretch width=100%}

<!--en:029f94bf-->
::: {.slide title="Kausale Maske mit Padding im Bild"}

<!--en:3450d0fe-->
[Heatmaps der Masken]{.kicker}

<!--en:c9078cc3-->
- **Padding `&` Kausalität:** Sind alle Scores null, mittelt Zeile $i$ über $\min(i+1, \ell_{\textrm{valid}})$ Schlüssel. In der zweiten Sequenz reicht keine Zeile über Schlüssel $3$ hinaus.

<!--en:b8c656b8-->
::: {.slide title="Maske für gepackte Dokumente im Bild"}

<!--en:3450d0fe-2-->
[Heatmaps der Masken]{.kicker}

<!--en:6c7e3f1b-->
- **Gepackte Dokumente:** Kurze Dokumente `doc = [0,0,0, 1,1, 2]` zu einer Zeile verketten. Der Ausdruck `(doc[:, None] == doc[None, :]) & causal_mask(6)` ergibt eine blockdiagonale kausale Maske, die jeden Informationsfluss zwischen Dokumenten verhindert.

<!--en:a2a0418b-->
::: {.slide title="Attention im Batch mit zwei Matrizenmultiplikationen"}

<!--en:a276c29b-->
[Matrixdarstellung]{.kicker}

<!--en:497641d8-->
- Stehen $n$ Abfragen in $\mathbf{Q} \in \mathbb{R}^{n \times d}$, $m$ Schlüssel in $\mathbf{K} \in \mathbb{R}^{m \times d}$ und $m$ Werte in $\mathbf{V} \in \mathbb{R}^{m \times d_v}$, genügen für alle $n$ Zugriffe zwei GEMM-Operationen:
  $$\textrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{n \times d_v}$$

<!--en:7cbf99a2-->
- **Ein Kopf von GPT-2 Small ($n=m=1{,}024, d=d_v=64$):** Die Score-Matrix enthält $1{,}024^2 = 1{,}048{,}576$ Gleitkommazahlen ($4\text{ MiB}$ in `float32`). Die beiden Produkte kosten $4n^2 d \approx 0.27\text{ GFLOPs}$.

<!--en:045d84f0-->
![Tensorformen bei skalierter Skalarprodukt-Attention im Batch.](../img/mdl-attention-shapes.svg){.nostretch width=100%}

<!--en:1b29bfcf-->
::: {.slide title="Das Modul DotProductAttention"}

<!--en:f18ec381-->
[Implementierung des Moduls]{.kicker}

<!--en:b6700a0e-->
Auf den letzten beiden Achsen berechnet `DotProductAttention` die Attention und behält alle vorangestellten Batch- und Kopfachsen bei:

<!--en:724d1b22-->
::: {.slide title="Überprüfung der Handrechnung im Code"}

<!--en:a7f9a231-->
[Plausibilitätsprüfung]{.kicker}

<!--en:9f5cbd64-->
Für die beiden $4\text{D}$-Abfragen und die drei $4\text{D}$-Schlüssel aus dem Rechenbeispiel liefert `DotProductAttention` genau die von Hand berechneten Gewichte und Ausgaben:

<!--en:0713de01-->
Eigene Module für Attention und Maskierung vor einem großen Trainingslauf stets an kleinen Eingaben prüfen, für die die Ausgaben der Softmax-Funktion in geschlossener Form bekannt sind.

<!--en:f38af194-->
::: {.slide title="Möglichkeiten und Grenzen des rein inhaltlichen Vergleichs"}

<!--en:3e2f79c9-->
[Rein inhaltlicher Zugriff]{.kicker}

<!--en:4ddfd593-->
- Jedem unterschiedlichen Buchstaben in `"attention"` und `"translation"` eine zufällige Einbettung $\mathbf{e} \in \mathbb{R}^{32}$ zuweisen. Gleiche Buchstaben erhalten den Score $\approx \sqrt{32} \approx 5.7$, unterschiedliche Buchstaben den Score $\approx 0 \pm 1$:

<!--en:edecf18a-->
- **Eindeutige Treffer (`i`, `o`):** Fast $100\%$ des Gewichts entfallen auf den passenden Buchstaben.

<!--en:6ca907e1-->
- **Mehrere Treffer (`a`, `t`, `n`):** Das Gewicht verteilt sich *gleichmäßig* über alle Vorkommen. Der Inhalt allein unterscheidet die Positionen nicht.

<!--en:21bc5566-->
- **Abfrage ohne Treffer (`e`):** Die Gewichte müssen sich zu $1$ summieren. Die Masse verteilt sich daher auf zufällige Schlüssel. Dies erklärt die Rolle von **Attention-Sinks** in großen Sprachmodellen.

<!--en:0c81e89e-->
::: {.slide title="End-to-End-Lernen von Sequenzalignments"}

<!--en:70791179-->
[Gelerntes Alignment]{.kicker}

<!--en:830e82e7-->
Kann Gradientenabstieg allein aus den Zielausgaben eine Umstellung der Wortfolge lernen? In `ToyAligner` muss die Ausgabeposition $j$ das Quelltoken an Position $\pi(j)$ kopieren. Dabei vertauscht $\pi = (0, 1, 2, 4, 3, 5, 6, 7)$ die Positionen $3$ und $4$:

<!--en:309f8fff-->
- **Modellarchitektur:** Jede Ausgabeposition $j$ hat eine trainierbare Abfrage $\mathbf{q}_j \in \mathbb{R}^{16}$.

<!--en:4e94faa9-->
- **Schlüssel mit Positionsinformation:** Für Quellposition $i$ werden die One-Hot-Kodierungen von Token ($16\text{D}$) und Position ($8\text{D}$) verkettet. Anschließend folgt die Multiplikation mit $\mathbf{W}_k \in \mathbb{R}^{24 \times 16}$.

<!--en:8d3af678-->
- **Werte:** One-Hot-kodierte Quelltoken. Die Attention-Ausgabe liefert damit direkt die vorhergesagte Tokenverteilung.

<!--en:4dc63da9-->
::: {.slide title="End-to-End-Lernen von Sequenzalignments (Fortsetzung)"}

<!--en:70791179-2-->
[Gelerntes Alignment]{.kicker}

<!--en:19c8173f-->
`ToyAligner` erzeugt die Schlüssel mit Positionsinformation. Anschließend liest `DotProductAttention` die One-Hot-kodierten Werte aus:

<!--en:4c9df242-->
::: {.slide title="Entstehung der Permutationsmatrix in ToyAligner"}

<!--en:cba18bb6-->
[Trainingsergebnisse]{.kicker}

<!--en:4317ee1a-->
- Nach $100$ Adam-Schritten zur Minimierung der negativen Log-Likelihood auf zufälligen Sequenzen erreichen alle drei Läufe eine Testgenauigkeit von $100\%$. Im Mittel entfallen mehr als $99\%$ des Gewichts auf $\pi(j)$:

<!--en:7ed93b49-->
- Auch bei wiederholten Quelltoken führt die gelernte Schlüsselprojektion jede Abfrage zu $\pi(j)$. Das zeigt sich an der Vertauschung der Positionen $3$ und $4$ (ab $0$ gezählt) außerhalb der Diagonalen.

<!--en:ee26445c-->
::: {.slide title="Verständnisfragen zu Skalierung, Maskierung und Invarianzen"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:5cdb4b26-->
Fragen zu skalierten Skalarprodukt-Scores und booleschen Masken:

<!--en:8d24e785-->
1. **Gewichtsverhältnis ohne Skalierung:** Bei einer Kopfbreite von $d = 64$ unterscheiden sich die unskalierten Skalarprodukt-Scores zweier Schlüssel um eine Standardabweichung ($\Delta = \sqrt{64} = 8$). In welchem Verhältnis stehen ihre Softmax-Gewichte ohne $1/\sqrt{d}$ und mit $1/\sqrt{d}$?

<!--en:fa4d3979-->
2. **Invarianz gegenüber dem Schlüssel-Bias:** Eine affine Schicht erzeuge die Schlüssel $\mathbf{k}_i = \mathbf{W}_k \mathbf{y}_i + \mathbf{b}_k$. Warum hat der Bias-Vektor $\mathbf{b}_k$ bei Skalarprodukt-Attention den Gradienten null ($\nabla_{\mathbf{b}_k} \mathcal{L} = \mathbf{0}$)?

<!--en:d823655e-->
3. **Gleitendes Fenster in Verbindung mit Padding:** Sequenz 2 habe die gültige Länge $3$ in einem auf Länge $6$ aufgefüllten Tensor. Was geschieht bei Anwendung von `padding_mask & sliding_window(w=2)` mit der Maskenzeile für die Abfrage an Position $i = 5$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:d8029ce2-->
1. **Unskaliert:** $e^8 \approx 2{,}981 : 1$. **Skaliert mit $1/\sqrt{64} = 1/8$:** $\Delta' = 1$, also $e^1 \approx 2.72 : 1$.

<!--en:74402cec-->
2. Ausmultiplizieren von $\mathbf{q}^\top(\mathbf{W}_k\mathbf{y}_i + \mathbf{b}_k) = \mathbf{q}^\top\mathbf{W}_k\mathbf{y}_i + \mathbf{q}^\top\mathbf{b}_k$ zeigt, dass $\mathbf{q}^\top\mathbf{b}_k$ unabhängig vom Schlüsselindex $i$ ist. Wegen der Verschiebungsinvarianz der Softmax-Funktion fällt dieser Term vollständig weg.

<!--en:2aa6f810-->
3. **Zeile $i = 5$ ist vollständig leer (`all False`).** Das Fenster erlaubt die Schlüssel $\{4, 5\}$, die Padding-Maske erlaubt $\{0, 1, 2\}$. Die Schnittmenge ist $\emptyset$. Deshalb muss `masked_softmax` leere Zeilen zuverlässig auf null setzen.

<!--en:deddbddb-->
::: {.slide title="Zusammenfassung zu Attention-Scores und Maskierung"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:0f5d90b3-->
- **Skaliertes Skalarprodukt ($\mathbf{q}^\top\mathbf{k} / \sqrt{d}$):** Eine einzige GEMM-Operation berechnet alle $n \times m$ Paare ($2nmd$ FLOPs). Dabei bleibt $\textrm{Var}(a) = 1$ erhalten, sodass die Jacobi-Matrix der Softmax-Funktion $\mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha}\boldsymbol{\alpha}^\top$ bei der Initialisierung mit zunehmender Kopfbreite nicht verschwindet.

<!--en:a9850760-->
- **Additive Scores:** Bei $d_q \neq d_k$ liefert $\mathbf{w}_v^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ die Scores. Der Speicherbedarf beträgt $O(nmr)$.

<!--en:e82be382-->
- **Numerisch sichere Maskenverknüpfung:** Das boolesche `&` verknüpft Padding-Masken und kausale Masken. Ausgeschlossene Scores erhalten vor der Softmax-Funktion einen Füllwert.
