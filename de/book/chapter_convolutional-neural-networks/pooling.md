<!--en:b363e5df-->
# Pooling

<!--en:310052b4-->
Faltungsschichten (engl. convolutional layers) beantworten lokale Fragen, etwa ob am Pixel $(i, j)$ eine Kante liegt. Ein Klassifikator muss dagegen eine globale Frage zum gesamten Bild beantworten. Der Schritt von lokalen Merkmalskarten (engl. feature maps) zu einer globalen Vorhersage erfordert sowohl eine große räumliche Reichweite (:numref:`subsec_stride_rf`) als auch Robustheit gegenüber kleinen Verschiebungen. Ein Kamerawackler um ein Pixel soll die Vorhersage nicht ändern.

<!--en:35e7fd65-->
Eine *Pooling-Schicht* (engl. pooling layer) erfüllt beide Ziele ohne zusätzliche lernbare Parameter. Sie verschiebt ein Fenster über jeden Kanal unabhängig von den anderen und fasst jeden Ausschnitt durch sein *Maximum* oder seinen *Mittelwert* zusammen. Das Maximum eines kleinen Blocks bleibt gleich, solange die stärkste Antwort innerhalb des Blocks bleibt. Lokales Pooling sorgt so für *Toleranz* gegenüber kleinen Verschiebungen. *Globales Pooling* über die gesamte Merkmalskarte fasst alle räumlichen Positionen zu einem einzigen Vektor mit einem Eintrag pro Kanal zusammen.

<!--en:70a2ce9e-->
## Max-Pooling und Average-Pooling

<!--en:96f50ea3-->
Wie eine Faltungsschicht verschiebt ein Pooling-Operator ein Fenster fester Form (das *Pooling-Fenster*) über die Eingabe und berechnet an jeder Position einen einzelnen Skalar als Ausgabe. Anders als bei einer Faltung gibt es in einer Pooling-Schicht jedoch keine lernbaren Gewichte für den Faltungskern. Eine feste deterministische Operation reduziert die Elemente im Fenster auf einen Wert. Beim *Maximum-Pooling* (kurz *Max-Pooling*) ist das Ergebnis der größte Eintrag im Fenster, beim *Average-Pooling* das arithmetische Mittel aller Einträge.

<!--en:aacb14ad-->
![Max-Pooling mit einem Pooling-Fenster der Form $2\times 2$. Schattiert sind das erste Ausgabeelement und die Elemente des Eingabetensors, aus denen es sich berechnet: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)

<!--en:e1debe05-->
In :numref:`fig_pooling` gleitet ein $2\times2$-Fenster mit Stride 1 über eine $3\times3$-Eingabe, von links nach rechts und von oben nach unten. Entlang jeder Achse passt das Fenster an zwei Positionen. Die Ausgabe hat daher die Form $2\times2$, und jeder Eintrag ist das Maximum des zugehörigen Fensters:

<!--en:0b6c9c44-->
Fünf der neun Eingabeeinträge (0, 1, 2, 3 und 6) sind in keinem Fenster das Maximum. Ändern wir sie geringfügig, bleibt die Ausgabe völlig unverändert, denn beim Max-Pooling zählt in jedem Fenster nur der größte Eintrag. Dasselbe Verfahren mit gleitendem Fenster funktioniert für jedes Pooling-Fenster der Form $k_\textrm{h} \times k_\textrm{w}$.

<!--en:a0a332c0-->
Den Vorwärtsdurchlauf für zweidimensionales Pooling implementieren wir in `pool2d`. Die verschachtelte Schleife ist wie in `corr2d` aus :numref:`sec_conv_layer` aufgebaut. An die Stelle des gewichteten inneren Produkts tritt entweder `max` oder `mean`. Außerdem nehmen wir ein Argument `stride` auf, damit wir als Nächstes das Downsampling untersuchen können.

<!--en:1969ade8-->
Ein Test von `pool2d` am $3\times3$-Tensor aus :numref:`fig_pooling` bestätigt, dass Max-Pooling die Werte 4, 5, 7 und 8 liefert.

<!--en:37f39316-->
Wechseln wir bei derselben Eingabe zum Average-Pooling, erhalten wir die vier Fenstermittelwerte 2, 3, 5 und 6.

<!--en:058c2abb-->
## Toleranz gegenüber kleinen Verschiebungen

<!--en:f35d56eb-->
Wie weit darf sich ein Merkmal verschieben, ohne dass sich die Pooling-Ausgabe ändert? Testen wir den einfachsten Fall. Eine $2\times2$-Pooling-Schicht mit Stride 2 unterteilt die Eingabe in überlappungsfreie $2\times2$-Blöcke und fasst jeden Block zu einem Wert zusammen. Betrachten wir eine $4\times4$-Merkmalskarte, auf der nur eine einzige Antwort von null verschieden ist. Sie hat den Wert 1, wie bei einem Kantendetektor, der auf eine isolierte Kante anspricht. Wir verschieben diese Antwort pixelweise entlang der Zeile 1. Im Folgenden vergleichen wir für drei aufeinanderfolgende Spaltenpositionen die Ausgaben von Max-Pooling und Average-Pooling.

<!--en:09dd1ba4-->
Verschieben wir die Antwort von Spalte 0 nach Spalte 1, bleiben beide Pooling-Ausgaben völlig unverändert, denn beide Positionen liegen im selben $2\times2$-Block. Verschieben wir sie um ein weiteres Pixel nach Spalte 2, überschreitet sie eine Blockgrenze. Die Antwort springt in der Ausgabe in die Nachbarzelle. Eine horizontale Verschiebung um ein Pixel bleibt also ohne Wirkung, wenn die Antwort innerhalb ihres Blocks bleibt. Das gilt für die Hälfte aller Ausgangsspalten. Andernfalls verschiebt sich die Ausgabe um eine Position. Durch lokales Pooling wird eine Darstellung also *tolerant* gegenüber bestimmten kleinen Verschiebungen, aber nicht streng invariant. :numref:`fig_pooling_shift` zeigt die Ausgaben des Max-Poolings für alle vier Spaltenpositionen.

<!--en:2d8be369-->
![Eine einzelne Antwort durchläuft die vier Spalten einer $4\times4$-Merkmalskarte (oben). Darunter stehen die Karte nach $2\times2$-Max-Pooling mit Stride 2 (Mitte) und das globale Maximum (unten). Dicke Linien markieren die Pooling-Blöcke. Solange die Antwort in ihrem Block bleibt, behält sie nach dem Pooling ihre Position. Beim Wechsel in den nächsten Block springt sie. Das globale Maximum bleibt stets an derselben Stelle.](../img/pooling-shift.svg)

<!--en:40e6884e-->
Max-Pooling und Average-Pooling fassen den Block auf unterschiedliche Weise zusammen. Max-Pooling erhält unabhängig von der Position im Block die volle Stärke der Aktivierung (1). Es beantwortet damit die Frage, ob das Merkmal irgendwo in dieser Nachbarschaft auftritt. Average-Pooling liefert 0.25, weil das Merkmal nur eine der vier Zellen belegt. Es misst also, welchen Anteil der Fläche das Merkmal einnimmt. Die Nullen in den Nachbarzellen schwächen eine starke isolierte Antwort bei der Mittelung ab.

<!--en:7df4381e-->
Bei der Unterabtastung (engl. subsampling) mit Stride 2 tritt außerdem ein leicht zu übersehendes Problem der Signalverarbeitung auf, sobald ein Merkmal eine Abtastgrenze überschreitet. Nehmen wir im Extremfall von einem schnell wechselnden Signal $1, 0, 1, 0, \ldots$ nur jeden zweiten Abtastwert, erhalten wir ausschließlich Einsen oder ausschließlich Nullen. Eine Verschiebung um ein einziges Pixel entscheidet, welcher Fall eintritt. Diese Verzerrung heißt *Aliasing*. Sie tritt auf, wenn sich ein Signal schneller ändert, als die Abtastrate auflösen kann, und betrifft jedes Downsampling mit Stride 2, ob durch Pooling oder Faltung. Ein kleiner Glättungsfilter vor der Unterabtastung mildert das Aliasing :cite:`zhang2019making`. Dieses Verfahren untersuchen wir in den Übungen.

<!--en:42aa14cd-->
Ursprünglich war die Toleranz gegenüber Verschiebungen das Hauptmotiv für Pooling. In Fukushimas Neocognitron :cite:`Fukushima.1980` gehörten zu jeder Schicht aus merkmalsdetektierenden Zellen weitere Zellen mit einer ähnlichen Funktion wie beim Pooling. Sie feuerten, sobald ein Merkmal irgendwo in einer lokalen Nachbarschaft auftrat. Eine frühe Form des Max-Poolings kam in der Spracherkennung zum Einsatz :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`. Später schlug die rechnergestützte Neurowissenschaft Max-Pooling als Modell dafür vor, wie der visuelle Kortex Hinweise hierarchisch zusammenführt :cite:`Riesenhuber.Poggio.1999`.

<!--en:0c7ed5da-->
## Globales Pooling

<!--en:81a09bc1-->
Wie erreichen wir die vollständige Translationsinvarianz (engl. translation invariance), die wir in :numref:`sec_why-conv` für die Frage brauchten, ob Walter *irgendwo* in einem Bild zu sehen ist? Wie wir gerade gesehen haben, kann ein kleines Fenster fester Größe diese Invarianz nicht garantieren. Das gelingt, wenn das Pooling-Fenster die *gesamte* Fläche jeder Merkmalskarte abdeckt. Maximum oder Mittelwert über alle räumlichen Positionen ergeben unabhängig vom Ort der Antwort exakt denselben Skalar (untere Zeile in :numref:`fig_pooling_shift`). Diese Operation heißt *globales Pooling* und reduziert jeden Kanal auf eine einzige Zahl.

<!--en:b33d7dbc-->
In einem vollständigen CNN folgt globales Pooling jedoch auf Faltungsschichten mit Null-Padding. Diese sind nur in ausreichendem Abstand zum Bildrand streng translationsäquivariant (:numref:`sec_why-conv`). Wie wir in :numref:`subsec_padding_values` gesehen haben, geht ein Pixel im Inneren in alle neun Fenster eines $3 \times 3$-Faltungskerns ein, ein Pixel am Rand dagegen nur in sechs und ein Eckpixel nur in vier. Das globale Maximum einer isolierten Antwort bleibt daher über die Spalten hinweg konstant, während der globale Mittelwert sinkt, wenn ein Merkmal an den mit Padding versehenen Rand rückt. Eine Folge von Faltungsschichten mit Stride 1 und abschließendem globalem Pooling ist also streng invariant gegenüber Verschiebungen, bei denen markante Bildinhalte vom Rand entfernt bleiben. Andernfalls ist sie näherungsweise invariant. Schichten mit größerem Stride schwächen diese Garantie ab. Wegen des zuvor beschriebenen Aliasings gilt die Invarianz dann exakt nur für Verschiebungen um ein Vielfaches des Produkts aller Strides.

<!--en:89e11729-->
In den modernen Architekturen aus :numref:`chap_modern_cnn` sind die beiden Aufgaben des Poolings häufig getrennt. Das räumliche Downsampling zwischen den Stufen übernehmen oft Faltungen mit Stride 2. Sie lernen, wie die Einträge jedes Fensters zu gewichten sind, statt Maximum oder Mittelwert fest vorzugeben. Am Ende des Netzes steht vor dem linearen Klassifikator ein *Global-Average-Pooling* :cite:`Lin.Chen.Yan.2013`. LeNet, die klassische Architektur, die wir im nächsten Abschnitt aufbauen, entstand vor beiden Konventionen.

<!--en:c93fd141-->
## Padding und Stride

<!--en:033a3b60-->
Beim Pooling gleitet ein Fenster genauso über das räumliche Gitter wie bei der Faltung. Deep-Learning-Frameworks verwenden für Pooling-Schichten daher dieselben Argumente für Fensterform, Padding und Stride. Für die Ausgabeform gilt dieselbe Formel wie in :numref:`sec_padding`. Bei einem Padding von $p$ Pixeln auf jeder Seite einer Achse der Länge $n$, einer Fenstergröße von $k$ und einem Stride von $s$ erhalten wir entlang dieser Achse $\lfloor(n+2p-k)/s\rfloor+1$ Ausgabewerte. Ein feiner Unterschied liegt darin, welcher *Wert* den zusätzlichen Rand füllt. Um dies an eingebauten Pooling-Schichten zu untersuchen, erzeugen wir zunächst einen vierdimensionalen Tensor. Er enthält eine einzige $4\times4$-Merkmalskarte mit den Einträgen 0 bis 15.

<!--en:7f76be93-->
Pooling fasst üblicherweise überlappungsfreie Bereiche zusammen. Deshalb wählen die meisten Frameworks als Stride standardmäßig die Größe des Pooling-Fensters. So überdecken die Fenster die Eingabe ohne Überlappung. Auf unsere $4\times4$-Eingabe passt ein $3\times3$-Fenster mit Stride 3 nur einmal. Es liefert das Maximum des oberen linken $3\times3$-Blocks (10). Da Pooling keine lernbaren Gewichte hat, benötigt die Schicht keine Initialisierung der Parameter.

<!--en:0a4a16a1-->
Die Funktion `nnx.max_pool` verwendet allerdings standardmäßig Stride 1. Deshalb übergeben wir `strides` ausdrücklich.

<!--en:adf5335d-->
Für überlappende Fenster oder explizites Padding können wir diese Vorgaben ändern. Setzen wir das Padding auf 1 und den Stride auf 2, hat die Ausgabe sowohl in der Höhe als auch in der Breite die Größe $\lfloor(4+2-3)/2\rfloor+1 = 2$.

<!--en:0bb2fdd0-->
Mit welchem Wert sollte eine Pooling-Schicht den zusätzlichen Rand füllen? Bei einer Faltung bietet sich null als neutrales Element an, denn ein Produkt mit null trägt nichts zur gewichteten Summe bei. Beim Max-Pooling ist das neutrale Element dagegen $-\infty$. Damit kann keine zusätzliche Randzelle eine echte Aktivierung übertreffen. Jeder Ausgabewert ist also das Maximum der ursprünglichen Eingabeeinträge im jeweiligen Fenster. In unserem Beispiel erfassen die Fenster in jeder Richtung die ursprünglichen Zeilen oder Spalten $\{0, 1\}$ beziehungsweise $\{1, 2, 3\}$. Das ergibt 5, 7, 13 und 15. Padding mit $-\infty$ erhält außerdem die Äquivarianz gegenüber einer Verschiebung der *Werte*. Addieren wir zu jedem Eingabeeintrag dieselbe Konstante, erhöht sich jeder Ausgabewert nach dem Pooling um genau diese Konstante. Das können wir an `X - 16` überprüfen, dessen Einträge alle strikt negativ sind.

<!--en:728b7c1d-->
Wie erwartet sinkt jeder Ausgabewert um genau 16. Hätte die Schicht den Rand mit Nullen gefüllt, hätten die drei Fenster am Rand den Wert 0 geliefert. Das Ergebnis wäre allein deshalb verfälscht, weil 0 größer ist als alle negativen Eingabewerte.

<!--en:c490c62c-->
Für Average-Pooling gibt es keine entsprechende neutrale Konstante. Zählen die zusätzlichen Nullen im Nenner mit, verzerrt das die Mittelwerte am Rand in Richtung 0. Sauberer ist es, nur durch die Anzahl der ursprünglichen Eingabeeinträge im jeweiligen Fenster zu teilen.

<!--en:4a3217eb-->
Hier zählt Average-Pooling die zusätzlichen Nullen standardmäßig mit (`count_include_pad=True`). Mit `count_include_pad=False` gehen nur die ursprünglichen Einträge in den Mittelwert ein.

<!--en:70d6b165-->
`tf.keras.layers.AvgPool2D` mittelt bei `padding='same'` nur die ursprünglichen Einträge im jeweiligen Fenster.

<!--en:ec7f4e81-->
## Mehrere Kanäle

<!--en:f45817df-->
Eine Faltungsschicht verknüpft Informationen aus allen Eingabekanälen, indem sie über diese Kanäle summiert (:numref:`sec_channels`). Eine Pooling-Schicht verarbeitet dagegen jeden Kanal unabhängig. Die Ausgabe hat deshalb immer genau so viele Kanäle wie die Eingabe. Zur Überprüfung verketten wir `X` und `X + 1` entlang der Kanalachse zu einem Tensor mit zwei Kanälen.

<!--en:179c9726-->
Unsere Pooling-Operation von vorhin (Padding 1, Stride 2) liefert zwei Ausgabekanäle. Im ersten stehen wieder 5, 7, 13 und 15, im zweiten sind alle Werte um 1 größer.

<!--en:1fddc623-->
Im Ergebnis ist die Kanalachse die letzte Achse. Dort stehen die beiden Kanäle nebeneinander.

<!--en:ef68dcb8-->
Da Pooling keine lernbaren Gewichte hat, kann es selbst keine Muster entdecken. Es fasst nur die Merkmale zusammen, die vorhergehende Faltungsschichten erkannt haben. Mit Faltungen, Padding, Stride, Dilatation, mehrkanaligen Merkmalskarten und Pooling kennen wir nun alle nötigen Bausteine. Damit können wir ein vollständiges konvolutionäres neuronales Netz aufbauen, trainieren und direkt mit unseren früheren MLPs vergleichen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:dd1d2041-->
Pooling-Schichten fassen jedes räumliche Fenster einer Eingabemerkmalskarte durch sein Maximum oder seinen Mittelwert zusammen. Sie haben keine lernbaren Parameter und verarbeiten jeden Kanal unabhängig. Für die Ausgabeform bei gegebenem Padding und Stride gelten genau dieselben Regeln wie bei Faltungen. Max-Pooling füllt den Rand mit $-\infty$, damit zusätzliche Positionen das Maximum nie beeinflussen. Eine $2\times2$-Pooling-Schicht mit Stride 2 halbiert die räumliche Auflösung entlang jeder Achse. Kleine Verschiebungen innerhalb eines Blocks gleicht sie aus. Überschreitet eine Antwort die Blockgrenze, springt sie dagegen auch in der Ausgabe. Max-Pooling erhält die Spitzenaktivierung in einer Nachbarschaft. Beim Average-Pooling schwächt die Mittelung über das Fenster isolierte Spitzen ab. Globales Pooling schließlich reduziert jede ganze Merkmalskarte auf einen Skalar. Nach Schichten mit Stride 1 ergibt das vollständige Translationsinvarianz für Muster im Inneren und näherungsweise Invarianz nahe den mit Padding versehenen Rändern. Bei Schichten mit größerem Stride gilt die Invarianz exakt nur für Verschiebungen um ein Vielfaches des Produkts aller Strides.

<!--en:2294243a-->
## Übungen

<!--en:2a8148cd-->
1. **Alternativen beim Pooling.**
    1. Brauchen wir für Minimum-Pooling eine eigene Schicht, oder lässt es sich durch eine Operation aus diesem Abschnitt ausdrücken?
    1. *Softmax-Pooling* mit der Temperatur $\tau > 0$ ersetzt die Fenstereinträge $x_1, \ldots, x_m$ durch ihren mit der Softmax-Funktion gewichteten Mittelwert
       $$\sum_{i=1}^m x_i \frac{\exp(x_i / \tau)}{\sum_{j=1}^m \exp(x_j / \tau)}.$$
       Bestimmen Sie die Grenzwerte für $\tau \to 0$ und für $\tau \to \infty$. Nennen Sie einen zusätzlichen Aufwand, der beim Softmax-Pooling gegenüber dem Max-Pooling entsteht.

<!--en:726decd1-->
2. **Aliasing und BlurPool.** Beim einfachen Downsampling mit Stride 2 bleibt jeder zweite Eingabeeintrag erhalten. Messen Sie die Frequenz in Perioden pro Abtastwert der Eingabe. Die eindimensionalen Signale $(1, 0, 1, 0, 1, 0)$ und $(0, 1, 0, 1, 0, 1)$ haben dann beide die Frequenz $\tfrac{1}{2}$.
    1. Wenden Sie das einfache Downsampling mit Stride 2 auf beide Signale an. Sie unterscheiden sich durch eine Verschiebung um eine Position. Vergleichen Sie die beiden Ausgaben. Warum heißt dieses Verhalten *Aliasing*?
    1. Welche Eingabefrequenzen kann die Unterabtastung mit Stride 2 unverfälscht wiedergeben?
    1. BlurPool :cite:`zhang2019making` wirkt Aliasing entgegen, indem es vor der Unterabtastung einen kleinen Mittelwertfilter anwendet. Berechnen Sie für die beiden obigen Signale das Ergebnis von BlurPool mit dem Mittelwertfilter $(\tfrac{1}{2}, \tfrac{1}{2})$ der Länge zwei. Mit welcher Pooling-Operation aus diesem Abschnitt stimmt es überein?

<!--en:786ccf6e-->
3. [code] **Rückwärtsdurchlauf des Max-Poolings.** Leiten Sie den Gradienten des zweidimensionalen Max-Poolings bezüglich seiner Eingabe für ein $k_\textrm{h} \times k_\textrm{w}$-Fenster her. Der Stride entspricht der Fenstergröße. Implementieren Sie den Gradienten ohne automatisches Differenzieren. Berechnen Sie zum Vergleich auf einem kleinen Zufallstensor den Gradienten durch automatisches Differenzieren der eingebauten Max-Pooling-Operation aus diesem Abschnitt. Prüfen Sie, ob der relative Fehler unter $10^{-4}$ liegt.

    *Nach Stanford CS231n, [Aufgabe 2](https://cs231n.github.io/assignments2024/assignment2/), ConvolutionalNetworks, einfacher Rückwärtsdurchlauf des Max-Poolings.*

<!--en:aa5dc1e5-->
4. **Max-Pooling mit ReLU.** ● Eine Faltung ist linear und kann daher allein kein Maximum berechnen.
    1. Drücken Sie $\max(a, b)$ ausschließlich durch affine Operationen und $\textrm{ReLU}(x) = \max(0, x)$ aus.
    1. Implementieren Sie damit Max-Pooling als Folge von Faltungen und ReLU-Schichten.
    1. Zählen Sie für ein $2 \times 2$-Fenster und ein $3 \times 3$-Fenster, wie viele Kanäle pro Schicht und wie viele ReLU-Schichten Ihre Konstruktion verwendet. Wie wächst die Anzahl der ReLU-Schichten mit der Anzahl der Einträge im Fenster?

<!--en:ca3b0d45-->
[Diskussion](https://d2l.discourse.group/t/71)

<!--en:a34e1b18-->
[Diskussion](https://d2l.discourse.group/t/72)

<!--en:886c76c6-->
[Diskussion](https://d2l.discourse.group/t/274)

<!--en:427755e2-->
[Diskussion](https://d2l.discourse.group/t/17999)

<!--en:a8baa677-->
Pooling-Schichten<br>**Max- und Average-Pooling · lokale Verschiebungstoleranz · Aliasing · globales Pooling · Padding mit −∞**

<!--en:a65bc89b-->
::: {.slide title="Warum äquivariante Merkmalskarten räumliche Aggregation brauchen"}

<!--en:3c60aaa5-->
[Warum die Empfindlichkeit gegenüber Verschiebungen ein Problem ist]{.kicker}

<!--en:e1508917-->
- **Pixelgenaue Äquivarianz:** Verschiebt sich ein Eingabebild um 1 Pixel, verschiebt sich jede Kantenaktivierung in jeder Merkmalskarte um 1 Pixel.

<!--en:b71ac0b2-->
- **Toleranz und Downsampling:** Klassifikatoren brauchen Stabilität gegenüber lokalen Verschiebungen und eine schrittweise gröbere räumliche Auflösung.

<!--en:222e735f-->
- **Pooling ohne Parameter:** Jeder $k \times k$-Ausschnitt wird pro Kanal durch sein **Maximum** oder seinen **Mittelwert** zusammengefasst ($0$ lernbare Gewichte).

<!--en:471a5a3b-->
::: {.slide title="Max-Pooling und Average-Pooling"}

<!--en:9a9125e4-->
[Deterministische Reduktion im gleitenden Fenster]{.kicker}

<!--en:9387a0db-->
Ein Pooling-Fenster der Form $k_\textrm{h} \times k_\textrm{w}$ gleitet wie bei `corr2d` über das Eingabegitter. An die Stelle des gewichteten inneren Produkts tritt eine feste Reduktion (`max` oder `mean`):

<!--en:97bd32d4-->
- **$2 \times 2$-Max-Pooling ($s=1$) auf einer $3 \times 3$-Eingabe:**
  $\max(0,1,3,4) = 4$, $\max(1,2,4,5) = 5$, $\max(3,4,6,7) = 7$, $\max(4,5,7,8) = 8$.

<!--en:cc6074fe-->
- **Gradientenverteilung bei der Backpropagation:** Max-Pooling leitet 100% des ankommenden Gradienten an den größten Eintrag jedes Fensters weiter. Average-Pooling verteilt ihn gleichmäßig ($1 / (k_\textrm{h} k_\textrm{w})$).

<!--en:2869a7aa-->
::: {.slide title="Eigene Implementierung von 2D-Pooling"}

<!--en:819c2547-->
[Maximum oder Mittelwert anstelle des inneren Produkts mit dem Faltungskern]{.kicker}

<!--en:cffee92f-->
Die Funktion `pool2d` ist wie `corr2d` aufgebaut. Sie verschiebt die linke obere Ecke um `stride` und reduziert jedes $k_\textrm{h} \times k_\textrm{w}$-Fenster mit `.max()` oder `.mean()`:

<!--en:ef77fef7-->
::: {.slide title="Ausgaben von Max-Pooling und Average-Pooling im Vergleich"}

<!--en:44bb56d5-->
[Erhalt der Spitzenaktivierung und räumliche Mittelung]{.kicker}

<!--en:6861c156-->
::: {.slide title="Verschiebungstoleranz am Beispiel eines wandernden Impulses"}

<!--en:97844879-->
[Reaktion von 2×2-Pooling mit Stride 2 auf Verschiebungen um 1 Pixel]{.kicker}

<!--en:dc8e9cf4-->
- **Verschiebung innerhalb des Blocks ($j=0 \to 1$):** Die Ausgabe bleibt zu $100\%$ unverändert.

<!--en:91f16320-->
- **Sprung über die Blockgrenze ($j=1 \to 2$):** Die Spitze wechselt in den rechten $2 \times 2$-Block.

<!--en:3f1cb0f0-->
::: {.slide title="Lokale Verschiebungstoleranz und globale Invarianz"}

<!--en:8531bc94-->
[Warum lokales Pooling die Hälfte der Verschiebungen um 1 Pixel ausgleicht]{.kicker}

<!--en:f5e5c8dc-->
Ein Impuls durchläuft alle vier Spalten des $4 \times 4$-Gitters. Die Mitte zeigt lokales $2 \times 2$-Pooling mit Stride 2, die untere Zeile globales Pooling über das ganze Gitter:

<!--en:bb265c92-->
- **Lokales Pooling ($k \times k$, Stride $k$):** Eine horizontale Verschiebung um 1 Pixel bleibt ohne Wirkung, wenn die Spitze vor der Verschiebung nicht in der letzten Spalte ihres Blocks liegt. Das gilt für einen Anteil von $(k - 1)/k$ der Ausgangsspalten ($50\%$ für $k = 2$, $67\%$ für $k = 3$).

<!--en:2aaeb04a-->
- **Signalstärke bei Maximum und Mittelwert:** Max-Pooling erhält überall im Block die volle Impulsamplitude ($1.0$) (*„Ist das Merkmal aufgetreten?“*). Average-Pooling schwächt eine isolierte Antwort auf das $1/k^2 = 0.25$-Fache ab (*„Welchen Anteil des Blocks hat das Merkmal bedeckt?“*).

<!--en:e0e8ce9f-->
::: {.slide title="Aliasing bei der Unterabtastung"}

<!--en:59b20ed5-->
[Warum Downsampling mit Stride 2 bei hohen Frequenzen das Ergebnis umkehren kann]{.kicker}

<!--en:39119cd4-->
- **Aliasing bei Verletzung des Nyquist-Kriteriums:** Für $\mathbf{x} = (1,0,1,0,1,0)$ liefert einfache Unterabtastung mit Stride 2 die Ausgabe $(1,1,1)$. Eine Verschiebung um 1 Pixel auf $T_1 \mathbf{x} = (0,1,0,1,0,1)$ kehrt das Ergebnis zu $(0,0,0)$ um.

<!--en:083ee9ea-->
- **Abhilfe durch BlurPool (Zhang, 2019):** Ein Mittelwertfilter $[\tfrac{1}{2}, \tfrac{1}{2}]$ als Tiefpass vor der Unterabtastung mit Stride 2 (d. h. $2 \times 2$-Average-Pooling) bildet beide Signale auf $(0.5, 0.5, 0.5)$ ab.

<!--en:dbe2f66d-->
::: {.slide title="Globales Pooling und Randeffekte"}

<!--en:818eee33-->
[Ein Skalar pro Merkmalskarte]{.kicker}

<!--en:97f0696b-->
- **Globales Pooling ($H \times W \to 1 \times 1$):** Reduziert jeden Kanal auf einen Skalar (insgesamt ein $c_\textrm{o}$-Vektor). Damit entfallen die positionsabhängigen Gewichte im vollständig verbundenen Ausgabeteil.

<!--en:6ec45b91-->
- **Exakte Invarianz im Inneren:** Das globale Maximum beträgt unabhängig von der Position des Impulses $1$.

<!--en:94f337bf-->
- **Randeffekt bei Null-Padding:** In den vorhergehenden Faltungen geht ein Pixel im Inneren in 9 Fenster ein, ein Randpixel in 6 (Mittelwert $9/16$ bzw. $6/16$).

<!--en:27c279ca-->
::: {.slide title="Eingebaute Pooling-Schichten mit überlappungsfreien Fenstern als Standard"}

<!--en:beb0ee8b-->
[Standardwerte der Frameworks und vierdimensionale Eingabetensoren]{.kicker}

<!--en:f5e29b7a-->
::: {.slide title="Eigene Werte für Padding und Stride in Pooling-Schichten"}

<!--en:643e4e39-->
[Dieselbe Formel für die Ausgabeform wie bei Faltungen]{.kicker}

<!--en:fd1bd384-->
Mit `padding=1` und `stride=2` anstelle der Standardwerte ergibt sich für ein $3 \times 3$-Fenster die Ausgabegröße $\lfloor (n + 2p - k)/s \rfloor + 1 = \lfloor (4 + 2 - 3)/2 \rfloor + 1 = 2 \times 2$:

<!--en:c1c9ad15-->
::: {.slide title="Warum Max-Pooling den Rand mit −∞ füllt"}

<!--en:07d6c6b2-->
[Neutrale Elemente und Äquivarianz bei Verschiebung der Aktivierungswerte]{.kicker}

<!--en:5d00534b-->
- **Neutrales Element $-\infty$:** Zusätzliche Randzellen können auch negative Aktivierungen der Eingabe nie übertreffen ($\max(\mathbf{X}-16) = \max(\mathbf{X})-16$).

<!--en:0daf6482-->
- **Option für AvgPool:** Mit `count_include_pad=False` nur durch die Anzahl der ursprünglichen Pixel teilen

<!--en:9dcbe32a-->
::: {.slide title="Unabhängigkeit der Kanäle in Pooling-Schichten"}

<!--en:bd620028-->
[Eine räumliche Reduktion pro Kanal ohne Mischung der Kanäle]{.kicker}

<!--en:61d8fd10-->
::: {.slide title="Faltungen mit größerem Stride und Global-Average-Pooling"}

<!--en:603d9a1a-->
[Wie moderne Architekturen Downsampling und Invarianz trennen]{.kicker}

<!--en:29ac716d-->
- **Downsampling zwischen Stufen:** Moderne CNNs (ResNet, ConvNeXt) ersetzen festes lokales Pooling zwischen den Stufen durch **Faltungen mit Stride 2**, die Gewichte für das Downsampling lernen.

<!--en:e4d95220-->
- **Aggregation im Ausgabeteil:** **Global-Average-Pooling** ($c \times h \times w \to c$) mit einer einzigen linearen Schicht ersetzt mehrschichtige, vollständig verbundene Ausgabeteile mit abgeflachter Eingabe. Das spart $>75\%$ der Parameter ein.

<!--en:0e2f10cd-->
::: {.slide title="Verständnisfragen zu Pooling und Invarianz"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:ec87df8e-->
Ausgabeform, Rechenaufwand und Verschiebungstoleranz beim Pooling bestimmen:

<!--en:a593fc8c-->
1. **Erste Pooling-Schicht in LeNet:** Auf eine Merkmalskarte der Form $6 \times 28 \times 28$ folgt $2 \times 2$-Pooling mit Stride $2$ und Padding $0$. Welche Form hat die Ausgabe? Wie groß sind die Parameteranzahl und die Gesamtzahl der paarweisen Vergleiche beim Max-Pooling?

<!--en:e74475df-->
2. **Wahrscheinlichkeit einer unveränderten Ausgabe:** Eine isolierte Spitzenaktivierung liegt an einem zufällig und gleichverteilt gewählten Pixel $(i, j)$ einer großen Merkmalskarte. Sie verschiebt sich diagonal um $+1$ Zeile und $+1$ Spalte. Bei welchem Anteil der Ausgangspositionen bleibt die Ausgabe einer $3 \times 3$-Max-Pooling-Schicht mit Stride 3 unverändert?

<!--en:62ca5ac3-->
3. **Minimum-Pooling und Softmax-Pooling ausdrücken:** Wie lässt sich Minimum-Pooling $\min_{x \in W} x$ allein mit Max-Pooling berechnen? Welche Operationen ergeben sich in den Temperaturgrenzfällen $\tau \to 0^+$ und $\tau \to \infty$ des Softmax-Poolings $\sum_i x_i e^{x_i/\tau} / \sum_j e^{x_j/\tau}$?

<!--en:942b6cee-->
**Lösungen:**
(1) Ausgabeform: $6 \times 14 \times 14$ ($1{,}176$ Werte). Parameter: **$0$**. Vergleiche: Jedes $2 \times 2$-Fenster mit 4 Zahlen erfordert $3$ Vergleiche. Insgesamt sind das $1{,}176 \times 3 = 3{,}528$ Vergleiche (gegenüber $117{,}600$ MACs in der vorhergehenden Faltungsschicht).
(2) Die Spitze bleibt genau dann in ihrem $3 \times 3$-Block, wenn ihre Zeile in $\{0, 1\} \pmod 3$ und ihre Spalte in $\{0, 1\} \pmod 3$ liegt. Das gilt für $(2/3) \times (2/3) = 4/9 \approx 44.4\%$ der Positionen.
(3) $\min(\mathbf{X}) = -\max(-\mathbf{X})$. Für $\tau \to 0^+$ konzentrieren sich die Softmax-Gewichte auf den größten Eintrag (**Max-Pooling**). Für $\tau \to \infty$ nähern sich alle Gewichte dem Wert $1/m$ (**Average-Pooling**).

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und zentrale Ergebnisse"}

<!--en:ce12e85f-->
[Der Abschnitt im Überblick]{.kicker}

<!--en:1df0d637-->
| Pooling-Variante | Reduktion im Fenster | Padding-Regel | Parameter und Kanäle | Rolle in der Architektur |

<!--en:284410c6-->
| **Max-Pooling ($k \times k$)** | $\max_{(a,b) \in W} X_{i\cdot s+a, j\cdot s+b}$ | Padding mit $-\infty$ | $0$ Parameter, $c_\textrm{o} = c_\textrm{i}$ (kanalweise) | Erhält die Spitzenaktivierung eines Merkmals und leitet den Gradienten an den größten Eintrag weiter |

<!--en:8aaeb1a1-->
| **Average-Pooling ($k \times k$)** | $\frac{1}{|W|}\sum_{(a,b) \in W} X_{i\cdot s+a, j\cdot s+b}$ | Division durch die Anzahl der ursprünglichen Einträge (`count_include_pad=False`) | $0$ Parameter, $c_\textrm{o} = c_\textrm{i}$ (kanalweise) | Mildert Aliasing bei hohen Frequenzen und misst den belegten Flächenanteil |

<!--en:9cfcc27f-->
| **Lokales Pooling mit Stride $k$** | Überlappungsfreie $k \times k$-Blöcke | Ausgabegröße $\lfloor (n+2p-k)/k \rfloor + 1$ | Verringert die Anzahl räumlicher Positionen um den Faktor $k^2$ | Gleicht pro Achse einen Anteil von $(k-1)/k$ der Verschiebungen um 1 Pixel aus (lokale Toleranz) |

<!--en:090b3f84-->
| **Globales Pooling ($H \times W$)** | $\max$ oder $\textrm{mean}$ über alle $(i,j)$ | Kein Padding nötig | Ausgabeform $(c_\textrm{i}, 1, 1) \to \mathbb{R}^{c_\textrm{i}}$ | Ersetzt den vollständig verbundenen Ausgabeteil mit abgeflachter Eingabe. Invariant gegenüber Verschiebungen im Inneren |

<!--en:fd8eb11c-->
**Ausblick (§{sec_lenet}):** Faltungen, Padding, mehrkanalige Merkmalskarten und Pooling bilden ein vollständiges, durchgängig trainierbares CNN: **LeNet**. Es tritt gegen das bisherige MLP an, auf Fashion-MNIST-Bildern mit ursprünglicher und zufällig gemischter Pixelanordnung.
