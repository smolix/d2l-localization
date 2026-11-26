<!--en:d4c58907-->
# Mehrere Eingabe- und Ausgabekanäle

<!--en:7ff92f27-->
Bisher haben unsere Faltungen (engl. convolutions) ein einzelnes 2D-Gitter auf ein einzelnes 2D-Gitter abgebildet. Bei realen Bildern und versteckten Darstellungen treten auf beiden Seiten mehrere *Kanäle* (engl. channels) auf. Ein RGB-Foto ist ein Tensor der Form $3 \times h \times w$. Um eine Grenze zwischen Rot und Grün bei gleicher Leuchtdichte zu erkennen, müssen wir die Kanäle miteinander vergleichen. Eine versteckte Schicht extrahiert parallel Dutzende oder Hunderte verschiedener Merkmalskarten (engl. feature maps).

<!--en:117785bc-->
Wie in :numref:`sec_why-conv` hergeleitet, besitzt eine Schicht mit $c_\textrm{i}$ Eingabekanälen und $c_\textrm{o}$ Ausgabekanälen für jeden Ausgabekanal einen $c_\textrm{i} \times k_\textrm{h} \times k_\textrm{w}$-Faltungskern (engl. convolutional kernel). Sie berechnet die Kreuzkorrelation für jeden Eingabekanal und summiert die Ergebnisse. In diesem Abschnitt implementieren wir Mehrkanalfaltungen und bestimmen ihren Aufwand anhand der Parameteranzahl, der Gleitkommaoperationen (FLOP) und des Aktivierungsspeichers (engl. activation memory). Für die Skalierung spielt das Produkt $c_\textrm{i} c_\textrm{o}$ eine zentrale Rolle. Außerdem führen wir $1 \times 1$-Faltungen als Flaschenhälse (engl. bottlenecks) ein.

<!--en:c36b8301-->
## Mehrere Eingabekanäle

<!--en:df7b8e0b-->
Beginnen wir mit der Eingabe. Hat sie $c_\textrm{i}$ Kanäle, so benötigt der Faltungskern für jeden Eingabekanal einen $k_\textrm{h}\times k_\textrm{w}$-Filter. Die Filter bilden einen Tensor dritter Ordnung mit der Form $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$. Für jeden Eingabekanal $\mathbf{X}_c$ und den zugehörigen Slice $\mathbf{K}_c$ des Faltungskerns berechnen wir die zweidimensionale Kreuzkorrelation (:numref:`sec_conv_layer`, implementiert durch `corr2d`). Dann summieren wir die $c_\textrm{i}$ resultierenden Gitter:

<!--en:acfd3e45-->
Jeder Eintrag der zweidimensionalen Ausgabe ist also eine gewichtete Summe über dasselbe räumliche Fenster in allen Eingabekanälen. In diese Summe gehen $c_\textrm{i} k_\textrm{h} k_\textrm{w}$ Produkte skalarer Werte ein. :numref:`fig_conv_multi_in` zeigt die Rechnung für zwei Eingabekanäle. Für den schattierten Ausgabeeintrag oben links ergibt sich $(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

<!--en:9bf57c84-->
![Kreuzkorrelation mit zwei Eingabekanälen. Auf die Kreuzkorrelation je Kanal folgt die Summation.](../img/conv-multi-in.svg)

<!--en:2cb504fd-->
Erst die Summe über die Kanäle erlaubt es einem Faltungskern, Informationen aus verschiedenen Kanälen zu vergleichen, etwa Farbinformationen. Betrachten wir einen $1\times 1$-Faltungskern mit dem Gewicht $+1$ für den Rotkanal und $-1$ für den Grünkanal. An jedem Pixel zieht er den Grünwert vom Rotwert ab. Das ergibt positive Werte in roten Bereichen, negative in grünen und null in neutralgrauen Bereichen jeder Helligkeit. Wechselt das Vorzeichen zwischen benachbarten Pixeln, trifft dort ein roter auf einen grünen Bereich. Das gilt auch bei identischer Leuchtdichte, wenn ein Helligkeitsfilter im Graustufenbild keine Kante erkennen könnte.

<!--en:28706580-->
Für die Kreuzkorrelation mit mehreren Eingabekanälen setzen wir die Formel direkt in Python um. Wir rufen `corr2d` einmal pro Kanal auf und summieren die Ergebnisse.

<!--en:c6e41198-->
Prüfen wir die Implementierung mit dem Eingabetensor und dem Faltungskern aus :numref:`fig_conv_multi_in`. Der Ausgabeeintrag oben links sollte 56 betragen.

<!--en:cbcc9b57-->
## Mehrere Ausgabekanäle

<!--en:f137bf78-->
Unabhängig von der Anzahl der Eingabekanäle erzeugt ein einzelner Faltungskern genau eine zweidimensionale *Merkmalskarte* (:numref:`subsec_why-conv-channels`). Um gleichzeitig $c_\textrm{o}$ verschiedene Merkmalskarten zu extrahieren, verwendet eine Faltungsschicht $c_\textrm{o}$ solcher Faltungskerne mit jeweils der Form $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$. Stapeln wir sie entlang einer neuen Achse, erhalten wir einen Tensor vierter Ordnung mit der Form $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$. Für den Ausgabekanal $o$ gilt

<!--en:2c0b4ddd-->
Jeder Ausgabekanal ist mit *jedem* Eingabekanal über einen eigenen $k_\textrm{h}\times k_\textrm{w}$-Slice $\mathbf{K}_{o,c}$ verbunden. Diese vollständige Vernetzung der Kanäle bestimmt maßgeblich den Rechenaufwand der Schicht. Mit einem zusätzlichen skalaren Bias $u_o$ je Ausgabekanal erhalten wir die in :numref:`sec_why-conv` hergeleitete Mehrkanal-Faltungsschicht. In typischen CNN-Architekturen wächst die Kanalzahl, während die räumliche Auflösung durch Strides oder Pooling sinkt. Räumliche Details gehen dabei verloren, doch an jeder Position entsteht eine reichhaltigere Beschreibung durch gemeinsam gelernte Merkmale.

<!--en:7cadb75c-->
Gluon speichert Faltungskerne mit den Kanalachsen am Anfang: $(c_\textrm{o}, c_\textrm{i}, k_\textrm{h}, k_\textrm{w})$.

<!--en:c92f279b-->
PyTorch speichert Faltungskerne mit den Kanalachsen am Anfang: $(c_\textrm{o}, c_\textrm{i}, k_\textrm{h}, k_\textrm{w})$.

<!--en:eda768f2-->
Keras speichert Faltungskerne mit den Kanalachsen am Ende: $(k_\textrm{h}, k_\textrm{w}, c_\textrm{i}, c_\textrm{o})$.

<!--en:034fcca0-->
Flax speichert Faltungskerne mit den Kanalachsen am Ende: $(k_\textrm{h}, k_\textrm{w}, c_\textrm{i}, c_\textrm{o})$.

<!--en:95ef245a-->
Für die Kreuzkorrelation mit mehreren Ausgabekanälen durchlaufen wir die Faltungskerne der Ausgabekanäle. Auf jeden wenden wir `corr2d_multi_in` an und stapeln die resultierenden Merkmalskarten entlang der Achse 0.

<!--en:09d93357-->
Testen wir diese Funktion. Dazu stapeln wir `K`, `K + 1` und `K + 2` zu einem Tensor von Faltungskernen für drei Ausgabekanäle.

<!--en:560c5457-->
Wenden wir `corr2d_multi_in_out` auf unsere zweikanalige Eingabe `X` und den gestapelten $3 \times 2 \times 2 \times 2$-Faltungskern `K` an, erhalten wir einen Ausgabetensor mit drei Kanälen und der Form $(3, 2, 2)$.

<!--en:51ee9067-->
Der erste Ausgabekanal stimmt mit unserem früheren Ergebnis überein, denn sein Faltungskern ist das ursprüngliche `K`. Den zweiten Kanal können wir im Kopf überprüfen. Erhöhen wir jedes Gewicht des Faltungskerns um 1, wächst jeder Ausgabeeintrag um die Summe aller Elemente im zugehörigen Fenster über beide Eingabekanäle. Für das Fenster oben links beträgt diese Summe 20. Der zweite Kanal beginnt also mit $56 + 20 = 76$.

<!--en:b34f2951-->
### Parameter und Kosten

<!--en:0c564b8d-->
Mit zusätzlichen Kanälen kann ein Netz mehr Zusammenhänge darstellen. Allerdings steigen Speicherbedarf und Rechenaufwand rasch. Zählen wir einen skalaren Bias pro Ausgabekanal mit, so hat eine Faltungsschicht

<!--en:8d9a71a1-->
lernbare Parameter. Nahe der Eingabe des Netzes ist diese Zahl überschaubar. Die $3\times 3$-Schicht, die 3 RGB-Kanäle auf 64 Kanäle abbildet, hat in :numref:`sec_why-conv` nur 1792 Parameter. Tiefer im Netz wächst jedoch die Kanalzahl. Eine $3\times 3$-Schicht mit 256 Eingabe- und 256 Ausgabekanälen benötigt 589 824 Gewichte und 256 Bias-Terme.

<!--en:2827b201-->
Betrachten wir nun den Rechenaufwand. Jedes Ausgabeelement benötigt eine Multiplikationsaddition (engl. multiply-add) für jedes Gewicht seines Ausgabekanals, insgesamt also $c_\textrm{i} k_\textrm{h} k_\textrm{w}$. Bei Merkmalskarten der räumlichen Größe $h_\textrm{o}\times w_\textrm{o}$ und $c_\textrm{o}$ Ausgabekanälen benötigt die Schicht

<!--en:4acec263-->
Multiplikationsadditionen. Für eine $56\times 56$-Merkmalskarte benötigt unsere Schicht mit 256 Eingabe- und 256 Ausgabekanälen $56^2\cdot 589{,}824 \approx 1.8\times 10^9$ Multiplikationsadditionen pro Bild. Sowohl die Parameteranzahl als auch der Rechenaufwand wachsen proportional zum Produkt $c_\textrm{i} c_\textrm{o}$. Verdoppeln wir die Breite jeder Schicht, vervierfachen sich beide Größen. Da die Schicht dieselben Gewichte an jeder räumlichen Position verwendet, wächst der Rechenaufwand auch proportional zur Ausgabefläche $h_\textrm{o} w_\textrm{o}$.

<!--en:25258d98-->
Beim Training fällt eine dritte, ebenso wichtige Größe leicht unter den Tisch: der *Aktivierungsspeicher*. Um die Gradienten im Rückwärtsdurchlauf zu berechnen, muss das Framework die Ausgabeaktivierungen jeder Schicht aufbewahren. Das sind $h_\textrm{o} w_\textrm{o} c_\textrm{o}$ Zahlen pro Trainingsbeispiel (:numref:`subsec_memory_during_training`). Anders als der Speicherbedarf für die Parameter ist der Aktivierungsspeicher in den *ersten* Schichten am größten. Dort ist die räumliche Auflösung am höchsten, obwohl diese Schichten nur wenige Parameter haben. Erzeugt zum Beispiel eine Schicht mit 3 Eingabe- und 64 Ausgabekanälen eine $224\times 224$-Ausgabe, muss sie $224^2\cdot 64 \approx 3.2\times 10^6$ Aktivierungen pro Beispiel zwischenspeichern, fast zweitausendmal so viele Zahlen wie ihre 1792 Parameter.

<!--en:b8045b4b-->
## Die $1\times 1$-Faltung

<!--en:f9996044-->
Auf den ersten Blick mag eine $1\times 1$-Faltung ($k_\textrm{h} = k_\textrm{w} = 1$) sinnlos erscheinen. Ihr räumliches Fenster umfasst nur ein Pixel. Daher kann sie weder Kanten noch Ecken oder andere Muster erkennen, die sich über benachbarte Positionen erstrecken. Bei mehreren Kanälen erfüllt sie jedoch eine wichtige Aufgabe, wie schon unser Farbfilter für die Differenz zwischen Rot und Grün gezeigt hat. An jeder Position $(i, j)$ multipliziert eine $1\times 1$-Faltung den $c_\textrm{i}$-dimensionalen Kanalvektor mit derselben $c_\textrm{o}\times c_\textrm{i}$-Gewichtsmatrix. So entsteht ein $c_\textrm{o}$-dimensionaler Ausgabevektor. Die Faltung wirkt also wie eine vollständig verbundene Schicht (engl. fully connected layer), die an jedem Pixel unabhängig mit denselben Gewichten arbeitet. :numref:`fig_conv_1x1` zeigt eine $1\times 1$-Faltung, die 3 Eingabekanäle auf 2 Ausgabekanäle abbildet. Höhe und Breite bleiben dabei erhalten.

<!--en:331016e1-->
![Eine $1\times 1$-Faltung mit drei Eingabekanälen und zwei Ausgabekanälen. Jedes Ausgabeelement ist eine Linearkombination der Eingabeelemente an derselben Position.](../img/conv-1x1.svg)

<!--en:18b53dd9-->
Da an jedem Pixel dieselbe lineare Transformation wirkt, genügt eine einzige Matrizenmultiplikation für die gesamte $1\times 1$-Faltung. Wir formen die Eingabe zu einer $c_\textrm{i}\times hw$-Matrix um, multiplizieren sie von links mit der $c_\textrm{o}\times c_\textrm{i}$-Matrix der Faltungsgewichte und bringen die Ausgabe wieder in die Form $(c_\textrm{o}, h, w)$.

<!--en:acf19fca-->
Prüfen wir mit zufälligen Eingaben, ob diese Formulierung als Matrizenmultiplikation mit `corr2d_multi_in_out` übereinstimmt.

<!--en:cb5fbe75-->
Wie erwartet stimmen beide Ergebnisse bis auf Rundungsfehler der Gleitkommaarithmetik überein.

<!--en:3bc0704b-->
Warum kommen $1\times 1$-Faltungen in modernen Architekturen so häufig vor? Mit ihnen lässt sich die Kanalzahl mit geringem Aufwand ändern. So können aufwendige räumliche Faltungen mit schmaleren Darstellungen arbeiten. Betrachten wir noch einmal unsere $3\times 3$-Schicht mit 256 Eingabe- und 256 Ausgabekanälen. Statt $3\times 3$-Faltungskerne direkt auf alle 256 Kanäle anzuwenden, projizieren wir die 256 Kanäle mit einer $1\times 1$-Faltung zunächst auf 64. Auf diese 64 Kanäle wenden wir die $3\times 3$-Faltung an. Eine zweite $1\times 1$-Faltung erhöht die Kanalzahl wieder auf 256. Zusammen benötigen die drei Schichten nur $16{,}384 + 36{,}864 + 16{,}384 = 69{,}632$ Gewichte statt $589{,}824$. Die Anzahl der Parameter und der Multiplikationsadditionen sinkt jeweils um einen Faktor von rund $8.5\times$. Ein solcher *Flaschenhals* beschränkt die Kombination benachbarter Positionen in der mittleren Schicht auf einen Unterraum mit 64 Kanälen. Architekturen wie ResNet (:numref:`sec_resnet`) nehmen diese Einschränkung in Kauf und gewinnen dadurch erheblich an Tiefe. Mit einer nachgeschalteten Aktivierungsfunktion wird eine $1\times 1$-Faltung zu einem kleinen nichtlinearen MLP, das an jedem Pixel ausgewertet wird :cite:`Lin.Chen.Yan.2013`. Ohne Nichtlinearität ließen sich aufeinanderfolgende lineare Faltungen zu einer einzigen linearen Transformation zusammenfassen.

<!--en:83ff5562-->
Zwei weitere Verfahren verkleinern den Faktor $c_\textrm{i} c_\textrm{o}$, indem sie einschränken, welche Eingabekanäle zu jedem Ausgabekanal beitragen. Das eine sind *gruppierte Faltungen* (engl. grouped convolutions), die durch ResNeXt (:numref:`subsec_resnext`) bekannt wurden. Das andere sind *tiefenweise separierbare Faltungen* (engl. depthwise-separable convolutions), auf denen kompakte Modelle für mobile Geräte beruhen (:numref:`sec_depthwise_separable`). Im nächsten Kapitel stellen wir beide Verfahren vor, wenn wir die entsprechenden Architekturen aufbauen.

<!--en:83bd0b3f-->
Auch mit mehreren Kanälen bleibt eine grundlegende Eigenschaft der Faltung erhalten. Verschiebt sich die Eingabe, verschiebt sich die Ausgabe um genauso viele Pixel. In :numref:`sec_pooling` führen wir Pooling-Schichten ein, die lokale Nachbarschaften zusammenfassen und das Netz unempfindlicher gegenüber kleinen räumlichen Verschiebungen machen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ecfa5438-->
Bei Eingaben mit mehreren Kanälen berechnet eine Faltungsschicht die Kreuzkorrelation jedes Eingabekanals mit seinem eigenen räumlichen Faltungskern und summiert die Ergebnisse. So kann ein einzelner Filter Informationen aus verschiedenen Kanälen verknüpfen, etwa Farbinformationen. Für $c_\textrm{o}$ Merkmalskarten in der Ausgabe stapelt die Schicht $c_\textrm{o}$ solcher Mehrkanal-Faltungskerne zu einem Tensor der Form $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$. Dafür benötigt sie $c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$ Parameter und $h_\textrm{o} w_\textrm{o} c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w}$ Multiplikationsadditionen. Beide Anzahlen wachsen mit dem Produkt $c_\textrm{i} c_\textrm{o}$. Beim Training muss die Schicht außerdem $h_\textrm{o} w_\textrm{o} c_\textrm{o}$ Aktivierungen pro Beispiel speichern. Eine $1\times 1$-Faltung wirkt an jedem Pixel wie dieselbe vollständig verbundene Schicht. Damit können Architekturen Kanäle mischen und Flaschenhälse bilden, die mit wenigen Parametern und geringem Rechenaufwand auskommen.

<!--en:2294243a-->
## Übungen

<!--en:4c55fece-->
1. **Verkettung von Faltungen.** Zwei Faltungsschichten mit Stride 1 und ohne Padding haben Faltungskerne der Formen $c_\textrm{m} \times c_\textrm{i} \times k_1 \times k_1$ und $c_\textrm{o} \times c_\textrm{m} \times k_2 \times k_2$. Zwischen den Schichten liegt keine Nichtlinearität.
    1. Beweisen Sie, dass die Verkettung beider Schichten eine einzige Faltung ergibt, und geben Sie die Form ihres Faltungskerns an.
    1. Vergleichen Sie die Anzahl der Gewichte beider Schichten mit der Anzahl bei der einzelnen Faltung. Für welche Breite $c_\textrm{m}$ der Zwischendarstellung benötigt das Schichtenpaar weniger Gewichte?
    1. Wie :numref:`sec_conv_layer` gezeigt hat, haben drei gestapelte einkanalige $3 \times 3$-Faltungskerne ein rezeptives Feld der Größe $7 \times 7$. Entscheiden Sie, ob sich jeder einkanalige $7 \times 7$-Faltungskern als Verkettung dreier $3 \times 3$-Faltungskerne darstellen lässt. Begründen Sie Ihre Antwort.

<!--en:47dc79e1-->
1. **Drei Kosten, drei Schichten.** Ein Netz für RGB-Bilder der Größe $224\times 224$ beginnt mit vier $3\times 3$-Faltungen mit Padding 1. Die erste bildet 3 auf 64 Kanäle ab (Stride 1), die zweite 64 auf 128 (Stride 2), die dritte 128 auf 128 (Stride 1) und die vierte 128 auf 256 (Stride 2).
    1. Geben Sie für jede Schicht die Ausgabeform, die Anzahl der Gewichte und die Anzahl der Multiplikationsadditionen pro Bild an. Wie viele Ausgabewerte muss jede Schicht beim Training für den Rückwärtsdurchlauf speichern?
    1. Welche Schicht hat die meisten Gewichte, welche benötigt die meisten Multiplikationsadditionen und welche speichert die meisten Ausgabewerte? Erklären Sie, warum es sich um drei verschiedene Schichten handelt.
    1. Ersetzen Sie die Bilder durch Bilder der Größe $448\times 448$. Welche der drei Kosten ändern sich und um welchen Faktor?

<!--en:4917601d-->
1. [code] **im2col mit mehreren Kanälen.** Erweitern Sie die Ausschnittsmatrix aus :numref:`subsec_conv_matmul` auf mehrere Eingabe- und Ausgabekanäle. Erstellen Sie die zweikanalige Eingabe `X` aus :numref:`fig_conv_multi_in` und den Faltungskern `K` mit drei Ausgabekanälen aus :numref:`subsec_multi-output-channels` erneut, da spätere Codezellen dieses Abschnitts beide überschreiben. Bauen Sie die Ausschnittsmatrix auf. Jede Zeile enthält einen Eingabeausschnitt der Form $c_\textrm{i} \times k_\textrm{h} \times k_\textrm{w}$, den Sie in genau dieser Achsenreihenfolge zu einem Vektor abflachen. Formen Sie `K` zu einer Matrix der Faltungsgewichte mit der Form $c_\textrm{i} k_\textrm{h} k_\textrm{w} \times c_\textrm{o}$ um. Ihre Zeilen sollen derselben Reihenfolge folgen. Prüfen Sie, ob das Produkt beider Matrizen nach dem Umformen in die Ausgabeform das Ergebnis von `corr2d_multi_in_out(X, K)` liefert.

    *Nach Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), Aufgaben 10.6 und 10.7.*

<!--en:23645f35-->
[Diskussion](https://d2l.discourse.group/t/69)

<!--en:b7d28c4f-->
[Diskussion](https://d2l.discourse.group/t/70)

<!--en:3df6a369-->
[Diskussion](https://d2l.discourse.group/t/273)

<!--en:d46791c7-->
[Diskussion](https://d2l.discourse.group/t/17998)

<!--en:adcd2a94-->
Mehrere Eingabe- und Ausgabekanäle<br>**Mehrkanal-Faltungskerne · Merkmalskarten · Skalierung von Rechenaufwand und Speicherbedarf · 1×1-Flaschenhälse**

<!--en:eb89328a-->
::: {.slide title="Warum Netze für die Bildverarbeitung Mehrkanaltensoren brauchen"}

<!--en:0f618e37-->
[Warum der Aufwand mit der Kanalzahl quadratisch wächst]{.kicker}

<!--en:483bd06e-->
- **Mehrkanaleingaben ($c_\textrm{i}$):** RGB-Bilder haben 3 Kanäle pro Pixel, versteckte Stufen liefern $64\text{--}512$ parallele Merkmalskarten ($c_\textrm{o}$).

<!--en:a3b9493c-->
- **Skalierungsgrenze durch $c_\textrm{i} c_\textrm{o}$:** Jeder Ausgabekanal ist mit jedem Eingabekanal verbunden. Eine einzige Schicht von $256 \to 256$ Kanälen mit $3 \times 3$-Filtern benötigt daher **$590{,}080$ Parameter** und **$1.85 \times 10^9$ MACs** bei $56 \times 56$.

<!--en:6263b39a-->
- **Lösungsansatz:** $1 \times 1$-Flaschenhalsprojektionen senken beides um den Faktor $8.5\times$.

<!--en:c449b1cb-->
::: {.slide title="Kreuzkorrelation mit mehreren Eingabekanälen"}

<!--en:ffdeab82-->
[Ein räumlicher Filter-Slice pro Eingabekanal, anschließend summieren]{.kicker}

<!--en:cafb904b-->
Hat die Eingabe $\mathsf{X}$ insgesamt $c_\textrm{i}$ Kanäle und die Form $(c_\textrm{i}, n_\textrm{h}, n_\textrm{w})$, so braucht der Faltungskern $\mathsf{K}$ für einen Ausgabekanal die Form $(c_\textrm{i}, k_\textrm{h}, k_\textrm{w})$. Für jeden Kanal $\mathbf{X}_c$ die Kreuzkorrelation mit seinem Filter-Slice $\mathbf{K}_c$ berechnen und über alle $c_\textrm{i}$ Kanäle summieren:

<!--en:d23a6827-->
- **Produkte skalarer Werte pro Ausgabepixel:** Jeder Eintrag $[\mathbf{Y}]_{i,j}$ summiert $c_\textrm{i} k_\textrm{h} k_\textrm{w}$ Produkte über Positionen und Kanäle.

<!--en:c4a50dc7-->
- **Schattierter Ausgabeeintrag oben links $(0,0)$ ($c_\textrm{i}=2$):**
  - Kanal 0: $0{\cdot}0 + 1{\cdot}1 + 3{\cdot}2 + 4{\cdot}3 = 19$.
  - Kanal 1: $1{\cdot}1 + 2{\cdot}2 + 4{\cdot}3 + 5{\cdot}4 = 37$.
  - Summe: $19 + 37 = 56$.

<!--en:3fe01a6f-->
::: {.slide title="Wie die Summe über Kanäle Gegenfarben erfasst"}

<!--en:aba6d6f4-->
[Warum räumliche Helligkeitsfilter Kanten bei gleicher Leuchtdichte übersehen]{.kicker}

<!--en:0e97af97-->
- **Gewichte für Gegenfarben:** Mit $+1$ für Rot und $-1$ für Grün ergibt sich $+1$ für reines Rot, $-1$ für reines Grün und $0$ für neutrales Grau jeder Helligkeit.

<!--en:f30c84ce-->
- **Isoluminante Kanten:** Rot-Grün-Grenzen bleiben selbst dort erkennbar, wo das Graustufenbild eine konstante Leuchtdichte hat.

<!--en:c84b01ea-->
::: {.slide title="Implementierung der Kreuzkorrelation mit mehreren Eingabekanälen"}

<!--en:486329f7-->
[2D-Kreuzkorrelationen entlang der Kanalachse summieren]{.kicker}

<!--en:723cfe5e-->
In Python bildet `corr2d_multi_in` Paare aus den Kanal-Slices von `X` und `K`, ruft für jedes Paar `d2l.corr2d(x, k)` auf und summiert die resultierenden 2D-Gitter:

<!--en:599bb492-->
::: {.slide title="Mehrere Ausgabekanäle und der 4D-Tensor der Faltungskerne"}

<!--en:ff61ad3f-->
[c_o Merkmalskarten parallel extrahieren]{.kicker}

<!--en:b4e99e35-->
- **4D-Filterbank $\mathsf{K} \in \mathbb{R}^{c_\textrm{o} \times c_\textrm{i} \times k_\textrm{h} \times k_\textrm{w}}$:** Aus $c_\textrm{o}$ unabhängigen, gestapelten 3D-Faltungskernen entstehen $c_\textrm{o}$ Merkmalskarten.

<!--en:cd7c35c1-->
- **Speicheranordnung:** Je nach Framework steht die Kanalachse unmittelbar nach der Batchachse, $(B, C, H, W)$, oder am Ende, $(B, H, W, C)$. Die Rechnung ist dieselbe.

<!--en:f35c6f53-->
::: {.slide title="Implementierung der Faltung mit mehreren Eingabe- und Ausgabekanälen"}

<!--en:fe4131c7-->
[Merkmalskarten der Ausgabe entlang der Achse 0 stapeln]{.kicker}

<!--en:f63ff0cd-->
Die Funktion `corr2d_multi_in(X, k)` auf jeden 3D-Faltungskern `k` im 4D-Tensor `K` anwenden. Stapeln der Ergebnisse entlang der Achse 0 liefert den 3D-Ausgabetensor $(c_\textrm{o}, h_\textrm{o}, w_\textrm{o})$:

<!--en:392772ee-->
::: {.slide title="Mehrkanalausgaben im Kopf überprüfen"}

<!--en:46657185-->
[Warum +1 für jedes Gewicht des Faltungskerns den Ausgabewert an Position (0,0) um 20 erhöht]{.kicker}

<!--en:01da81d9-->
- **Linearität im Faltungskern:** Steigt jedes Gewicht in $\mathsf{K}$ um $+c$, erhöht sich die Ausgabe um das $c$-Fache der Summe aller Eingabeelemente im Fenster.

<!--en:f1b73f6a-->
- **Summe im $2 \times 2$-Fenster oben links:** $(0+1+3+4) + (1+2+4+5) = 20$.

<!--en:b84b2e43-->
- **Ausgabewerte an Position $(0,0)$:** Kanal 0 liefert $56$, Kanal 1 ($\mathsf{K}+1$) liefert $56 + 20 = 76$ und Kanal 2 ($\mathsf{K}+2$) liefert $56 + 40 = 96$.

<!--en:a17c1d9b-->
::: {.slide title="Drei Kosten einer Faltungsschicht"}

<!--en:d11dd2d7-->
[Parameter, Multiplikationsadditionen (MACs) und Aktivierungsspeicher]{.kicker}

<!--en:f000c0f0-->
- **1. Parameter:** $c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$ (unabhängig von $H \times W$)

<!--en:884729d6-->
- **2. Multiplikationsadditionen (MACs):** $h_\textrm{o} w_\textrm{o} \cdot c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w}$ (proportional zu Ausgabefläche $\times$ Anzahl der Gewichte)

<!--en:505e0f46-->
- **3. Aktivierungsspeicher:** $B \cdot h_\textrm{o} w_\textrm{o} c_\textrm{o}$ Werte im Vorwärtsdurchlauf für die Backpropagation zwischenspeichern

<!--en:c9d0b865-->
::: {.slide title="Welche Schichten Speicher und Rechenzeit beanspruchen"}

<!--en:a1fec68d-->
[Frühe Eingangsschichten und tiefe Schichten mit vielen Kanälen]{.kicker}

<!--en:eca5f817-->
- **Eingangsstufe ($3 \to 64$ bei $224^2$):** Wenig Speicher für Gewichte ($1{,}792$ Parameter = $7\textrm{ KB}$), viel für zwischengespeicherte Aktivierungen ($3.21\textrm{M}$ = **$12.8\textrm{ MB/image}**).

<!--en:93b55dbe-->
- **Tiefe Stufe ($256 \to 256$ bei $56^2$):** Viel Speicher für Gewichte ($590\textrm{K}$ Parameter = **$2.36\textrm{ MB}**), $4\times$ weniger Speicher für Aktivierungen ($3.2\textrm{ MB/image}$).

<!--en:0e5dca4e-->
- **Faustregel für Speicherengpässe beim Training:** Den GPU-Speicherbedarf bestimmt vor allem das Produkt **Batchgröße $\times$ Größe der frühen, hochauflösenden Merkmalskarten**.

<!--en:e1143667-->
::: {.slide title="Die 1×1-Faltung als MLP über die Kanäle jedes Pixels"}

<!--en:e3fcb98d-->
[Kanäle mischen, ohne räumlich zu aggregieren]{.kicker}

<!--en:cfb5d8de-->
Bei $k_\textrm{h} = k_\textrm{w} = 1$ umfasst das räumliche Fenster nur ein Pixel $(i, j)$. Was berechnet eine $1 \times 1$-Faltung dann?

<!--en:cdce3af1-->
- **Dieselbe vollständig verbundene Schicht an jeder Position:** An jedem Pixel $(i, j)$ multipliziert sie den $c_\textrm{i}$-Vektor der Eingabekanäle mit derselben $c_\textrm{o} \times c_\textrm{i}$-Gewichtsmatrix $\mathbf{K}$.

<!--en:d96a2a05-->
- **Aufgaben:** Merkmale über Kanäle hinweg mischen, die Kanalzahl durch Projektion erhöhen oder senken ($c_\textrm{i} \to c_\textrm{o}$) und mit nachgeschalteter ReLU eine Nichtlinearität pro Pixel einführen

<!--en:519a917e-->
::: {.slide title="Die 1×1-Faltung als einzelnes Matrixprodukt implementieren"}

<!--en:dd506663-->
[Von (c_i, H, W) nach (c_i, H*W) umformen, ohne Werte wie bei im2col zu duplizieren]{.kicker}

<!--en:7073f7c0-->
::: {.slide title="1×1-Flaschenhälse senken Parameteranzahl und MACs um den Faktor 8.5"}

<!--en:8243e380-->
[Den Aufwand breiter 3×3-Faltungen durch ein kleineres Produkt c_i * c_o senken]{.kicker}

<!--en:458f4642-->
- **Direkte $3 \times 3$-Faltung ($256 \to 256$):** $589{,}824$ Gewichte und $1.85 \times 10^9$ MACs bei $56 \times 56$

<!--en:fa0351f2-->
- **Flaschenhals ($256 \xrightarrow{1\times 1} 64 \xrightarrow{3\times 3} 64 \xrightarrow{1\times 1} 256$):** Nur $16{,}384 + 36{,}864 + 16{,}384 = 69{,}632$ Gewichte (**Reduktion um den Faktor $8.5\times$**)

<!--en:f6ba3d9c-->
- **Notwendige Nichtlinearitäten:** ReLU zwischen den 3 linearen Schichten verhindert, dass sie zu einer einzigen Matrix vom Rang 64 zusammenfallen.

<!--en:ec1d20c1-->
::: {.slide title="Neben 1×1-Flaschenhälsen: gruppierte und tiefenweise Faltungen"}

<!--en:7f00cbed-->
[Die vollständige Vernetzung der Kanäle in modernen CNNs faktorisieren]{.kicker}

<!--en:73317876-->
- **Gruppierte Faltung ($g$ Gruppen):** Die Aufteilung der Kanäle in $g$ unabhängige Gruppen senkt die Anzahl der Gewichte und MACs um den Faktor $g\times$ (ResNeXt, RegNet).

<!--en:7ff32900-->
- **Tiefenweise separierbare Faltung ($g = c_\textrm{i}$ + $1 \times 1$):** Räumliche Filterung ($c_\textrm{i} k^2$) und Kanalmischung ($c_\textrm{o} c_\textrm{i}$) laufen getrennt ab. Der Aufwand sinkt um den Faktor $\approx 8\text{--}9\times$ bei $3 \times 3$-Schichten (MobileNet, ConvNeXt).

<!--en:fa526291-->
::: {.slide title="Verständnisfragen zu Mehrkanalfaltungen und ihrer Skalierung"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:fe04b7e4-->
Die folgenden drei Aufgaben lösen:

<!--en:911e1641-->
1. **Kreuzkorrelation von Hand:** Für die zweikanalige $3 \times 3$-Eingabe und den $2 \times 2$-Faltungskern aus dem Beispiel ergab sich $[\mathbf{Y}]_{0,0} = 19 + 37 = 56$. Den Eintrag $[\mathbf{Y}]_{0,1}$ in der ersten Zeile und zweiten Spalte berechnen.

<!--en:b55b74ef-->
2. **Zweite Faltungsschicht von LeNet:** Sie bildet $c_\textrm{i} = 6$ auf $c_\textrm{o} = 16$ Kanäle ab, mit $5 \times 5$-Faltungskernen ohne Padding auf einer $14 \times 14$-Eingabe. Welche Ausgabeform, Parameteranzahl und Anzahl an Multiplikationsadditionen ergeben sich?

<!--en:020b07fd-->
3. **Verdopplung der Auflösung:** Die Bildauflösung der Eingabe steigt von $224 \times 224$ auf $448 \times 448$. Um welchen Faktor ändern sich bei einer Faltungsschicht (a) die Parameteranzahl, (b) die Anzahl der Multiplikationsadditionen und (c) der Aktivierungsspeicher?

<!--en:2bc32333-->
**Antworten:**
(1) Fenster oben rechts in Kanal 0: $1{\cdot}0 + 2{\cdot}1 + 4{\cdot}2 + 5{\cdot}3 = 25$. Dasselbe Fenster in Kanal 1: $2{\cdot}1 + 3{\cdot}2 + 5{\cdot}3 + 6{\cdot}4 = 47$. Summe: $25 + 47 = 72$.
(2) Ausgabeform: $16 \times 10 \times 10$. Parameter: $16 \times 6 \times 25 + 16 = 2{,}416$. MACs: $10^2 \times 2{,}400 = 240{,}000$.
(3) (a) Parameter: **$1\times$** (unverändert). (b) Multiplikationsadditionen: **$4\times$** ($h_\textrm{o} w_\textrm{o}$ vervierfacht sich). (c) Aktivierungsspeicher: **$4\times$**.

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und Kernaussagen"}

<!--en:ce12e85f-->
[Der Abschnitt im Überblick]{.kicker}

<!--en:ec7742dc-->
| Konzept oder Verfahren | Mathematische Form oder Tensorform | Skalierung der Kosten | Funktion in der Architektur |

<!--en:e0bc0b35-->
| **Faltung mit mehreren Eingabekanälen** | $\mathbf{Y} = \sum_{c=1}^{c_\textrm{i}} \mathbf{X}_c \star \mathbf{K}_c$ | $\mathsf{K} \in \mathbb{R}^{c_\textrm{i} \times k_\textrm{h} \times k_\textrm{w}}$ | Verknüpft Farbinformationen oder mehrere Merkmale über Kanäle hinweg |

<!--en:54409a32-->
| **Faltung mit mehreren Ausgabekanälen** | $\mathbf{Y}_o = u_o + \sum_{c=1}^{c_\textrm{i}} \mathbf{X}_c \star \mathbf{K}_{o,c}$ | $c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$ Parameter | Extrahiert $c_\textrm{o}$ Merkmalskarten parallel. Die Kanalzahl wächst, wenn $H, W$ abnehmen |

<!--en:e86127d9-->
| **Rechenaufwand und Speicher** | $\textrm{MACs} = h_\textrm{o} w_\textrm{o} c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w}$ | $\textrm{Activations} = h_\textrm{o} w_\textrm{o} c_\textrm{o}$ | Frühe Schichten benötigen den meisten Aktivierungsspeicher, tiefe und breite Schichten die meisten Parameter |

<!--en:fab47780-->
| **$1 \times 1$-Faltung** | $\mathbf{Y} = \textrm{reshape}(\mathbf{K}\,\textrm{reshape}(\mathsf{X}))$ | $c_\textrm{o} c_\textrm{i} + c_\textrm{o}$ Parameter | MLP über die Kanäle jedes Pixels. Flaschenhalsprojektionen senken den Aufwand um den Faktor $8.5\times$ |

<!--en:59564370-->
**Ausblick (§{sec_pooling}):** Wie **Pooling-Schichten** lokale Nachbarschaften zusammenfassen, kleine räumliche Verschiebungen abfangen und räumliche Merkmalskarten zu translationsinvarianten Vektoren verdichten.
