<!--en:e207b0fd-->
# Padding, Stride und Dilatation

<!--en:2a517bb8-->
Eine $k \times k$-Faltung (engl. convolution) ohne Padding verkleinert die Eingabe entlang jeder Achse um $k - 1$ Pixel (:numref:`sec_conv_layer`). Nach zehn $5 \times 5$-Schichten schrumpft ein Bild von $240 \times 240$ auf $200 \times 200$ Pixel. An den Rändern verliert es dabei 30% seiner Fläche. Behalten wir mit Stride 1 in jeder Schicht die volle Auflösung bei, entsteht das umgekehrte Problem. Jede $3 \times 3$-Schicht erweitert das rezeptive Feld (engl. receptive field) einer Einheit um nur zwei Pixel. Um ein ganzes $224 \times 224$-Bild zu erfassen, brauchen wir also über hundert Schichten. Bei Aufgaben mit dichter Vorhersage (engl. dense prediction) wie der semantischen Segmentierung brauchen wir *beides*, ein großes Sichtfeld und die volle räumliche Auflösung, ohne den quadratischen Rechenaufwand riesiger Faltungskerne (engl. convolutional kernels).

<!--en:d0e92128-->
Drei geometrische Größen bestimmen diese Abwägung. *Padding* ergänzt die Eingabe um einen Rand. So können wir die Faltungskerne auf Randpixeln zentrieren und die räumlichen Abmessungen erhalten. Mit einem *Stride* größer als eins verschieben wir das Fenster um mehrere Pixel. Dadurch sinkt die Auflösung der Merkmalskarten (engl. feature maps) und das rezeptive Feld wächst schneller. Die *Dilatation* vergrößert die Abstände zwischen den Gewichten des Faltungskerns. So erfasst auch ein kleiner Kern bei voller Auflösung einen größeren Bereich.

<!--en:d305fe05-->
## Padding

<!--en:4924faaf-->
### Randpixel

<!--en:b78ebdc5-->
Schon bevor aufeinanderfolgende Schichten den Bildrand abtragen, nutzen sie die Pixel in Randnähe nur wenig. :numref:`img_conv_reuse` zeigt für jedes Pixel, wie viele Positionen des gleitenden Fensters es bei einer Faltung ohne Padding überdecken. Bei einem $k\times k$-Faltungskern liegt jedes innere Pixel, das von allen Rändern mindestens $k-1$ Schritte entfernt ist, in $k^2$ Fenstern. Ein Randpixel liegt dagegen in höchstens $k$ Fenstern und ein Eckpixel in nur einem. Bei einem $5\times 5$-Faltungskern trägt ein inneres Pixel zu 25 Ausgabepositionen bei, ein Eckpixel nur zu einer. Ein Objekt in der Nähe einer Ecke macht sich in der nächsten Schicht also kaum bemerkbar. Ohne Gegenmaßnahme bezieht jede Schicht ihre Information überproportional aus der Mitte ihrer Eingabe.

<!--en:ab749495-->
![Anzahl der Fenster, die jedes Pixel enthalten, wenn ein $1\times 1$-, $2\times 2$- oder $3\times 3$-Faltungskern ohne Padding über eine große Eingabe gleitet. Innere Pixel liegen in $k^2$ Fenstern, Eckpixel in einem. Jede gestreckte Zelle steht für alle Pixel im jeweiligen Abstand vom Rand, unabhängig von ihrer Anzahl.](../img/conv-reuse.svg)

<!--en:1bcae6d2-->
### Zero-Padding

<!--en:c54c2692-->
Die einfachste Abhilfe besteht darin, die Eingabe um zusätzliche Pixel (meist Nullen) zu ergänzen, bevor wir den Faltungskern darüber verschieben. Dieses Verfahren heißt *Padding*. In :numref:`img_conv_pad` fügen wir oben und unten je eine Zeile sowie links und rechts je eine Spalte aus Nullen hinzu. Damit wächst die $3\times 3$-Eingabe zu einem $5\times 5$-Gitter. Ein $2\times 2$-Faltungskern liefert dann eine Ausgabe der Größe $4\times 4$ statt $2\times 2$. Das farbig markierte Ausgabeelement oben links kombiniert drei Nullen aus dem Padding mit dem Eckpixel der ursprünglichen Eingabe. Das ergibt $0\times0+0\times1+0\times2+0\times3=0$.

<!--en:b2d4bab4-->
![Zweidimensionale Kreuzkorrelation mit einem Pixel Zero-Padding auf jeder Seite.](../img/conv-pad.svg)

<!--en:9bddf95d-->
Wie in Deep-Learning-Frameworks üblich, geben wir das Padding pro Seite an. Wir ergänzen $p_\textrm{h}$ Zeilen oberhalb und $p_\textrm{h}$ Zeilen unterhalb der Eingabe sowie jeweils $p_\textrm{w}$ Spalten links und rechts. Die Eingabe mit Padding hat dann die Form $(n_\textrm{h}+2p_\textrm{h})\times(n_\textrm{w}+2p_\textrm{w})$. Für die Ausgabe ergibt sich die Form

<!--en:22b35f8b-->
Jede zusätzliche Padding-Zeile oben oder unten erweitert die Ausgabe um eine Zeile, jede zusätzliche Spalte links oder rechts um eine Spalte. Der Faltungskern verkleinert die Ausgabe um $k_\textrm{h}-1$ Zeilen und $k_\textrm{w}-1$ Spalten. Wählen wir $p_\textrm{h}=(k_\textrm{h}-1)/2$ und $p_\textrm{w}=(k_\textrm{w}-1)/2$, bleiben Höhe und Breite der Eingabe exakt erhalten. Das ist in der Praxis mit Abstand die häufigste Wahl. In einem Stapel formerhaltender Schichten bleibt die räumliche Auflösung in jeder Tiefe gleich. Das erleichtert es erheblich, Netzarchitekturen zu entwerfen und zu verstehen.

<!--en:ca96b71a-->
Die Division von $k-1$ durch zwei erklärt, warum CNNs fast immer ungerade Kerngrößen wie 1, 3, 5 oder 7 mit einem Padding von entsprechend 0, 1, 2 oder 3 kombinieren. Ein Faltungskern ungerader Größe hat ein eindeutiges mittleres Pixel. Bei symmetrischem Padding liegt die Mitte des Fensters für den Ausgabewert `Y[i, j]` genau auf dem Eingabepixel `X[i, j]`. Eingabe- und Ausgabegitter sind damit aufeinander ausgerichtet. Bei gerader Kerngröße ist $k_\textrm{h}-1$ ungerade. Um die räumlichen Abmessungen zu erhalten, brauchen wir dann asymmetrisches Padding mit einer Zeile mehr auf einer Seite als auf der anderen. Dadurch verschiebt sich das Ausgabegitter gegenüber der Eingabe um ein halbes Pixel.

<!--en:e1dd1834-->
Um diese Formeln für die Ausgabeform im Code zu überprüfen, definieren wir zunächst eine kleine Hilfsfunktion. Die eingebauten Faltungsschichten erwarten vierdimensionale Eingaben mit einer Batch- und einer Kanalachse. Deshalb ergänzt `comp_conv2d` eine zweidimensionale Eingabe um zwei Dimensionen der Länge eins, wendet die Schicht an und entfernt diese Dimensionen wieder aus dem Ergebnis.

<!--en:d7eff6ba-->
Bei einem $3\times 3$-Faltungskern erfordert formerhaltendes Padding die Breite $p = (3-1)/2 = 1$ (im Code `padding=1`). Wenden wir diese Schicht auf eine $8\times 8$-Eingabe an, sollte die räumliche Form $8\times 8$ erhalten bleiben.

<!--en:12bf77dd-->
In Keras akzeptiert `Conv2D` die Einstellungen `padding='valid'` (ohne Padding) und `padding='same'` (Ausgabegröße $\lceil n/s \rceil$, bei Stride 1 entspricht das einem symmetrischen Padding der Breite $(k-1)/2$ für ungerades $k$). Für eigene Padding-Breiten verwenden wir `ZeroPadding2D` vor `padding='valid'`. Bei der Eingabe steht die Kanalachse an letzter Stelle (`(batch, height, width, channel)`).

<!--en:8b11f2c1-->
`nnx.Conv` verwendet standardmäßig `padding='SAME'` und akzeptiert auch `'VALID'`, eine ganze Zahl (oder ein Tupel mit einem Wert pro Achse), `'REFLECT'` oder `'CIRCULAR'`. Bei der Eingabe steht die Kanalachse an letzter Stelle (`(batch, height, width, channel)`).

<!--en:093e8cb5-->
### Padding-Werte

<!--en:8a2b4aa3-->
Zero-Padding setzt implizit voraus, dass schwarze Pixel das Bild außerhalb seines Rahmens umgeben. Formal legt das Padding eine *Randbedingung* (engl. boundary condition) fest. Sie gibt vor, welche Werte wir jenseits des Bildrands einsetzen, wo keine Messwerte vorliegen. Untersuchen wir die Wirkung von Zero-Padding auf die Ausgabe. Dazu ergänzen wir ein gleichförmiges $5\times 5$-Bild aus Einsen auf jeder Seite um einen ein Pixel breiten Rand aus Nullen. Dann berechnen wir die Kreuzkorrelation mit einem $3\times 3$-Faltungskern aus Einsen. Dazu verwenden wir `corr2d` aus :numref:`sec_conv_layer`. Jeder Ausgabewert gibt nun an, wie viele Pixel des ursprünglichen Bildes in seinem Fenster liegen.

<!--en:ea31b813-->
Im Inneren beträgt der Ausgabewert 9, am Rand sinkt er auf 6 und in den Ecken auf 4. Obwohl das ursprüngliche Bild völlig gleichförmig ist, sieht die Schicht einen dunklen Rahmen. Dieser künstliche Rahmen breitet sich in den folgenden Schichten nach innen aus. Die Schichten können lernen, ihn als impliziten Hinweis auf die Position zu nutzen. Zugleich können Randartefakte in Randnähe blinde Flecken erzeugen, an denen das Netz nur schwach reagiert :cite:`Alsallakh.Kokhlikyan.Miglani.ea.2020`. Zwei gängige Alternativen kommen ohne zusätzliche Nullen aus. *Spiegelndes Padding* (engl. reflect padding) spiegelt die Pixel am Rand. *Padding durch Wiederholung* (engl. replicate padding) setzt die Randwerte nach außen fort. Beide Verfahren bilden ein konstantes Bild auf eine konstante Ausgabe ab. Deshalb sind sie in Netzen zur Bilderzeugung und Bildrestaurierung beliebt, bei denen dunkle Ränder direkt in den erzeugten Pixeln sichtbar wären. In Klassifikationsnetzen bleibt Zero-Padding die übliche Wahl. Es ist einfach, und schnelle Kernel können es implizit anwenden, ohne Speicher für einen größeren Tensor mit Padding zu reservieren.

<!--en:d3ceace8-->
## Stride

<!--en:03c515ae-->
### Faltung mit Stride

<!--en:27f9c895-->
Ein $224\times 224$-Bild im gesamten Netz mit voller Auflösung zu verarbeiten, kostet viel Rechenzeit. Außerdem dauert es lange, bis eine Einheit den globalen Kontext erfasst. Jeder Kanal enthält in jeder Schicht 50 176 räumliche Positionen. Da jede $3\times 3$-Faltung das Sichtfeld einer Einheit um nur zwei Pixel erweitert, bräuchten wir mehr als 100 Schichten, bis eine einzelne Einheit das ganze Bild sehen könnte. Verschieben wir das Faltungsfenster in größeren Schritten über die Eingabe, lösen wir beide Probleme zugleich. Die räumliche Auflösung sinkt, und es genügen weniger Schichten, um den globalen Kontext zu erfassen.

<!--en:e072585d-->
Bisher haben wir das Fenster in jeder Zeile um jeweils ein Pixel nach rechts und in jeder Spalte um jeweils ein Pixel nach unten verschoben. Die Schrittweite entlang der Höhe und der Breite heißt *Stride*. Bei vertikalem Stride $s_\textrm{h}$ und horizontalem Stride $s_\textrm{w}$ wertet die Schicht den Faltungskern nur in jeder $s_\textrm{h}$-ten Zeile und jeder $s_\textrm{w}$-ten Spalte aus. Dadurch sinkt die Anzahl der Ausgabeelemente ungefähr um den Faktor $s_\textrm{h} s_\textrm{w}$. :numref:`img_conv_stride` zeigt einen vertikalen Stride von 3 und einen horizontalen Stride von 2 für die Eingabe mit Padding aus :numref:`img_conv_pad`. Durch diese asymmetrische Wahl lassen sich die Achsen leicht unterscheiden. Jeder farbige Ausgabewert hat die Farbe des Fensters, aus dem er stammt. Der blaue Eintrag beträgt $0\times0+0\times1+1\times2+2\times3=8$. Sein Fenster liegt zwei Spalten rechts vom ersten. Der orangefarbene Eintrag beträgt $0\times0+6\times1+0\times2+0\times3=6$. Sein Fenster liegt drei Zeilen tiefer. Ein weiterer Schritt um zwei Spalten nach rechts würde das Fenster über den rechten Rand der Eingabe mit Padding hinausschieben. Es bleiben also zwei Ausgabewerte pro Zeile.

<!--en:cc8e31e5-->
![Kreuzkorrelation mit einem vertikalen Stride von 3 und einem horizontalen Stride von 2. Jeder farbige Ausgabewert hat die Farbe des Fensters, aus dem er stammt.](../img/conv-stride.svg)

<!--en:fa2e28db-->
Um die Ausgabehöhe allgemein zu bestimmen, betrachten wir die möglichen Fensterpositionen. Das erste Fenster beginnt in Zeile 0 der Eingabe mit Padding. Das gesamte Fenster muss innerhalb der $n_\textrm{h}+2p_\textrm{h}$ Zeilen dieser Eingabe liegen. Als Anfangszeilen kommen $0, s_\textrm{h}, 2s_\textrm{h}, \ldots$ bis höchstens $n_\textrm{h}+2p_\textrm{h}-k_\textrm{h}$ infrage. Die gleiche Überlegung für die Breite liefert die Ausgabeform

<!--en:e7574d4d-->
Der abgerundete Term zählt, wie viele ganze Schritte das Fenster nach seiner Platzierung machen kann. Das $+1$ zählt die Ausgangsposition mit. Für :numref:`img_conv_stride` erhalten wir $\lfloor 3/3\rfloor+1=2$ Zeilen und $\lfloor 3/2\rfloor+1=2$ Spalten. Setzen wir $s_\textrm{h}=s_\textrm{w}=1$, ergibt sich wieder unsere frühere Formel für das Padding. Kombinieren wir Stride mit formerhaltendem Padding ($2p_\textrm{h}=k_\textrm{h}-1$), vereinfacht sich die Ausgabehöhe zu $\lfloor (n_\textrm{h}-1)/s_\textrm{h}\rfloor+1=\lceil n_\textrm{h}/s_\textrm{h}\rceil$. Bei einem Stride von $s$ teilen wir also die räumliche Auflösung durch $s$ und runden auf.

<!--en:c84b7c96-->
Eine in der gesamten ResNet-Familie (:numref:`sec_resnet`) verbreitete Schicht für das Downsampling ist die $3\times 3$-Faltung mit Padding 1 und Stride 2. Bei unserer $8\times 8$-Eingabe halbiert sie beide räumlichen Abmessungen. Die Ausgabe hat die Form $4\times 4$.

<!--en:b8c52bcd-->
Bei Stride 2 und einer Eingabe gerader Größe ergänzt `padding='same'` unten eine Zeile und rechts eine Spalte. Oben und links kommt nichts hinzu. Das Fenster für die Ausgabe $i$ ist also auf dem Eingabepixel $2i+1$ statt auf $2i$ zentriert. Mit `ZeroPadding2D(1)` gefolgt von `padding='valid'` ergänzen wir auf beiden Seiten jeweils 1 Pixel.

<!--en:1fa12c06-->
Für Höhe und Breite können wir unterschiedliche Kerngrößen, Padding-Breiten und Strides wählen. Die Formel für die Ausgabeform gilt für jede Achse unabhängig.

<!--en:1efa53f7-->
Ein wichtiger Sonderfall ergibt sich, wenn der Stride gleich der Kerngröße ist und das Padding entfällt. Die Fenster zerlegen die Eingabe dann in überlappungsfreie Bildausschnitte. Eine $16\times 16$-Faltung mit Stride 16 zerlegt zum Beispiel ein $224\times 224$-Bild in ein $14\times 14$-Gitter aus Bildausschnitten (engl. patches). Jeden Bildausschnitt projiziert sie auf einen Vektor mit den Ausgabewerten der Kanäle. Mit dieser *Patchify*-Eingangsschicht wandeln Vision Transformer Bilder in Tokensequenzen um. Auch mehrere moderne CNNs beginnen mit dieser Operation.

<!--en:3615417b-->
Wenn für Höhe und Breite dieselben Einstellungen gelten, lassen wir die Indizes weg und schreiben $p$ und $s$. Sofern nicht anders angegeben, verwendet eine Faltungsschicht Stride 1 und kein Padding ($p=0$).

<!--en:0dd739bf-->
Bei `nnx.Conv` ist allerdings `padding='SAME'` voreingestellt.

<!--en:1754b416-->
### Stride und rezeptives Feld

<!--en:f1d9d49c-->
In :numref:`sec_conv_layer` haben wir gesehen, dass bei Stride 1 jede $k\times k$-Schicht das rezeptive Feld pro Achse um $k-1$ Pixel erweitert. Mit der Tiefe wächst es also nur linear. Ein größerer Stride beschleunigt dieses Wachstum erheblich. Nach einer Schicht mit Stride 2 entsprechen benachbarte Ausgabeeinträge Positionen, die im ursprünglichen Bild zwei Pixel auseinanderliegen. Ein nachfolgender $3\times 3$-Faltungskern erweitert das rezeptive Feld daher um $(3-1)\cdot 2=4$ Eingabepixel statt um

<!--en:ed79abf2-->
2. Allgemein entspricht ein einzelner Schritt in der Eingabe von Schicht $i$ einer Strecke von $\prod_{j<i} s_j$ Pixeln im ursprünglichen Bild. Für einen Stapel aus $L$ Schichten mit Kerngrößen $k_i$ und Strides $s_i$ hat das rezeptive Feld einer Einheit in der obersten Schicht also die Breite

<!--en:0575ddb9-->
Dabei hat das leere Produkt für $i = 1$ definitionsgemäß den Wert 1. Jede Schicht trägt die Spannweite ihres Faltungskerns bei, gemessen in Pixeln des ursprünglichen Bildes. Setzen wir alle Strides auf 1, vereinfacht sich jeder Summand zu $k_i-1$. Wir erhalten wieder das lineare Wachstum aus :numref:`sec_conv_layer`. Das Padding kommt in der Formel nicht vor. Es bestimmt, welche Ausgabepositionen es gibt. Die Breite des Bildausschnitts, den eine Einheit sieht, hängt davon nicht ab. Vergleichen wir zwei Stapel aus je drei $3\times 3$-Faltungen. Mit den Strides $(1, 1, 1)$ ergibt sich $r = 1+2+2+2 = 7$. Mit den Strides $(2, 2, 1)$ wächst die Reichweite auf mehr als das Doppelte, nämlich auf $r = 1+2+4+8 = 15$.

<!--en:a1f9471a-->
Jeder Stride vervielfacht die effektive Reichweite aller darüberliegenden Schichten. Solange das Downsampling fortgesetzt wird, wächst das rezeptive Feld daher geometrisch. Fünf Stufen mit Stride 2 sind typisch für Bildklassifikatoren mit $224\times 224$-Eingaben. Danach entspricht ein einzelner Schritt in der Merkmalskarte 32 Eingabepixeln. Einheiten in den obersten Schichten können so schon nach wenigen Schichten das ganze Bild erfassen.

<!--en:551ff1e1-->
## Dilatation

<!--en:6059ce21-->
### Dilatierte Faltung

<!--en:8bcefe89-->
Bei Faltungen mit Stride erkaufen wir ein großes rezeptives Feld mit geringerer räumlicher Auflösung. Für manche Aufgaben brauchen wir jedoch sowohl einen weiten Kontext als auch eine Ausgabe an jedem Eingabepixel. Ein größerer Faltungskern würde das Sichtfeld erweitern, aber Parameteranzahl und Rechenaufwand wachsen quadratisch mit der Kerngröße. Padding und Stride bestimmen, *wo* wir den Faltungskern ansetzen. Die Dilatation verändert die *Abstände im Faltungskern selbst*. Eine *dilatierte Faltung* (engl. dilated convolution) :cite:`yu2016dilated` lässt zwischen benachbarten Gewichten Lücken, sodass ihre Abtastpunkte $d$ Pixel auseinanderliegen. Dabei bezeichnet $d$ die *Dilatationsrate*. Ein $k\times k$-Faltungskern mit Dilatation $d$ enthält weiterhin nur $k^2$ lernbare Gewichte. Er erreicht aber eine *effektive Kerngröße* von

<!--en:74641642-->
Pixeln pro Achse und überspringt zwischen zwei aufeinanderfolgenden Abtastpunkten jeweils $d-1$ Pixel. Mit $d=1$ erhalten wir wieder die gewöhnliche dichte Faltung. :numref:`fig_conv_dilation` vergleicht einen $3\times 3$-Faltungskern bei $d=1$ und $d=2$. Bei Dilatation 2 erfassen dieselben neun Gewichte einen $5\times 5$-Bereich, indem sie entlang jeder Achse jedes zweite Pixel abtasten. Pro Ausgabeposition genügen weiterhin neun Multiplikationen.

<!--en:23b2b719-->
![Ein $3\times 3$-Faltungskern bei Dilatation 1 und Dilatation 2. Bei Dilatation 2 erstrecken sich seine neun Gewichte über einen $5\times 5$-Bereich.](../img/arch-conv-dilation.svg)

<!--en:51ef77b5-->
Für die Abmessungen der Ausgabe verhält sich ein dilatierter Faltungskern genau wie ein dichter Kern der Größe $k'$. Ersetzen wir in der Formel für die Faltung mit Stride die Kerngröße $k$ durch $k' = d(k-1)+1$, erhalten wir einen einheitlichen Ausdruck für die Ausgabegröße entlang einer Achse bei beliebigem Padding, Stride und beliebiger Dilatation:

<!--en:86ea3af7-->
Die Formel gilt unabhängig für Höhe und Breite mit den jeweiligen Parametern $n$, $k$, $p$, $s$ und $d$. Mit $d=1$ erhalten wir die Formel für die Faltung mit Stride. Setzen wir zusätzlich $s=1$ und $p=0$, ergibt sich wieder $n-k+1$. Für unsere $8\times 8$-Eingabe sollte ein $3\times 3$-Faltungskern mit Dilatation 2 daher dieselbe $4\times 4$-Ausgabeform liefern wie ein dichter $5\times 5$-Faltungskern.

<!--en:91f42d5c-->
### Dilatation und rezeptives Feld

<!--en:b3e58ac7-->
Bei Stride 1 erweitert eine dilatierte Schicht das rezeptive Feld um $k'-1 = d(k-1)$ Pixel. Somit gilt :eqref:`eq_receptive_field` weiterhin, wenn wir jede Kerngröße $k_i$ durch ihre effektive Größe $k'_i$ ersetzen. Verdoppeln wir die Dilatationsrate in jeder Schicht nach dem Schema $d_i = 1, 2, 4, \ldots, 2^{L-1}$, so erreicht ein Stapel aus $L$ Schichten mit $3\times 3$-Faltungskernen ein rezeptives Feld der Breite $1 + 2(1 + 2 + \cdots + 2^{L-1}) = 2^{L+1}-1$. Wie bei einem Stapel mit Stride wächst das rezeptive Feld exponentiell mit der Tiefe. Dabei behält jede Schicht die volle räumliche Auflösung der Eingabe bei.

<!--en:f861b4b8-->
:numref:`fig_conv_rf_towers` stellt die beiden Strategien einander gegenüber. Teil (a) zeigt unseren Stapel mit den Strides $(2, 2, 1)$. Die drei Schichten erweitern das Sichtfeld der obersten Einheit um $2$, $4$ und $8$ Pixel auf insgesamt 15 Eingabepixel. Teil (b) erreicht mit Stride 1 und den Dilatationsraten $(1, 2, 4)$ ein rezeptives Feld von genau denselben 15 Pixeln. Das Muster der Verbindungen zwischen der obersten Einheit und der Eingabe ist identisch. Allein die räumliche Dichte unterscheidet sich. Der Stapel mit Stride behält nur jede vierte Position pro Achse bei, der dilatierte Stapel wertet alle Positionen aus. Ohne Padding und bei gleichen Gewichten entspricht die Ausgabe des Stapels mit Stride genau jedem vierten Eintrag pro Achse in der Ausgabe des dilatierten Stapels. Das gilt selbst dann, wenn nach jeder Schicht eine elementweise Nichtlinearität folgt. Dilatierte Faltungen eignen sich deshalb besonders für Aufgaben mit *dichter Vorhersage* wie die semantische Segmentierung. Dort müssen wir jedem Pixel ein Klassenlabel zuweisen und dabei einen großen Bereich der Umgebung berücksichtigen :cite:`yu2016dilated`. Das kostet Rechenzeit und Speicher, weil jede Schicht Merkmalskarten mit voller Auflösung verarbeiten muss. Alternativ können wir die Auflösung mit Strides verringern und sie später mit lernbaren Upsampling-Schichten wiederherstellen, etwa mit der transponierten Faltung aus :numref:`sec_transposed_conv`.

<!--en:5f14ab5c-->
![Rezeptives Feld der obersten Einheit von drei aufeinanderfolgenden Schichten mit Kerngröße 3 in einer Dimension. Die Zellen zeigen, welche Positionen jede Schicht berechnet. Von den farbig markierten Zellen hängt die oberste Einheit ab. Die Linien verbinden die Abtastpunkte der Faltungskerne mit den zugehörigen Ausgaben. (a) Bei Strides von 2, 2, 1 behält Schicht 1 jede zweite Position bei, Schicht 2 und 3 jede vierte. (b) Bei Stride 1 und Dilatationsraten von 1, 2, 4 behält jede Schicht alle Positionen bei. Beide obersten Einheiten sehen dieselben 15 Eingabepixel.](../img/conv-rf-towers.svg)

<!--en:282a444e-->
Bisher haben wir in allen Beispielen dieses Abschnitts eine einkanalige Eingabe mit einem einzelnen Faltungskern verarbeitet. Reale Bilder enthalten mehrere Farbkanäle, und versteckte Schichten erkennen in der Praxis Dutzende oder Hunderte verschiedener Muster gleichzeitig. In :numref:`sec_channels` heben wir beide Einschränkungen auf.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:68ea0e27-->
Ohne Padding verkleinert eine Faltung die Eingabe entlang jeder Achse um $k-1$ Pixel. Randpixel gehen dabei in weit weniger Fenster ein als innere Pixel. Bei ungerader Kerngröße genügt ein Padding von $(k-1)/2$ Pixeln auf jeder Seite der Eingabe, um deren Form zu erhalten. Das Fenster für jede Ausgabe bleibt auf dem zugehörigen Eingabepixel zentriert. Zero-Padding führt allerdings eine Randbedingung ein, die tiefe Schichten erkennen können. Bei einem Stride $s > 1$ werten wir nur jede $s$-te Fensterposition aus. Dadurch sinken räumliche Auflösung und Rechenaufwand pro Achse ungefähr um den Faktor $s$, während das rezeptive Feld höherer Schichten geometrisch wächst. Die Dilatation $d$ verteilt die Gewichte eines Faltungskerns über eine effektive Breite von $d(k-1)+1$. Mit schichtweise verdoppelter Dilatation wächst das rezeptive Feld bei voller Auflösung exponentiell, ohne dass Parameter hinzukommen. Eine einzige Formel für die Ausgabeform erfasst alle drei Mechanismen (:eqref:`eq_conv_output_shape`).

<!--en:2294243a-->
## Übungen

<!--en:a4611816-->
1. [code] **Spiegelndes Padding.** Beim spiegelnden Padding (auch Reflect-Padding genannt) erweitern wir einen Tensor, indem wir seine Einträge am Rand spiegeln, ohne den Randeintrag zu wiederholen. Ergänzen wir den Vektor $(1, 2, 3)$ auf beiden Seiten um je einen Eintrag, erhalten wir $(2, 1, 2, 3, 2)$.
    1. Implementieren Sie spiegelndes Padding mit je einem zusätzlichen Eintrag auf jeder Seite eines zweidimensionalen Tensors. Überprüfen Sie das Ergebnis von Hand an einem $3 \times 3$-Tensor mit lauter verschiedenen Einträgen.
    1. Wiederholen Sie die Rechnung aus :numref:`subsec_padding_values` mit spiegelndem Padding anstelle von Zero-Padding und erklären Sie das Ergebnis.
    1. Betrachten Sie die $5\times 5$-Rampe $x_{ij} = j$ und einen $3\times 3$-Faltungskern, dessen Einträge alle gleich $\frac{1}{9}$ sind. Zeigen Sie, dass die Ausgabe im Inneren gleich der Eingabe ist. Berechnen Sie die Ausgaben in der ersten und letzten Spalte für Zero-Padding und für spiegelndes Padding. Finden Sie ein Padding, das die Rampe exakt reproduziert.

<!--en:5b3a3c91-->
2. **Dilatation und Gitterartefakte.** Ein Netz besteht aus vier aufeinanderfolgenden $3 \times 3$-Faltungen mit Stride 1. Verdoppeln wir die Dilatationsrate nach dem Schema $(1, 2, 4, 8)$, hat das rezeptive Feld eines Ausgabeelements die Seitenlänge $31$. Vergleichen Sie diese Folge mit der konstanten Folge $(4, 4, 4, 4)$.
    1. Berechnen Sie mit :eqref:`eq_receptive_field` das rezeptive Feld für die konstante Folge. Setzen Sie für jeden Faltungskern seine effektive Größe ein.
    1. Bestimmen Sie für beide Folgen, von welchen Pixeln innerhalb des rezeptiven Feldes die Ausgabe tatsächlich abhängt, und zählen Sie diese Pixel. Hängt die Ausgabe nur von den Punkten eines groben Gitters innerhalb ihres rezeptiven Feldes ab, spricht man von *Gitterartefakten*.
    1. Nennen Sie einen Grund, weshalb Gitterartefakte bei Aufgaben mit dichter Vorhersage wie der Segmentierung schaden.
    1. ● Geben Sie für die Dilatationsraten $d_1, \ldots, d_L$ eines Stapels aus $3 \times 3$-Faltungen mit Stride 1 eine Bedingung an, die garantiert, dass die Ausgabe von jedem Pixel ihres rezeptiven Feldes abhängt. Zeigen Sie, dass die Bedingung ausreicht.

<!--en:47f1579b-->
3. **Strides und Dilatationsraten im Vergleich.** Betrachten Sie die beiden eindimensionalen Schichtstapel aus :numref:`fig_conv_rf_towers` ohne Padding. Beide verwenden dieselben Faltungskerne $\mathbf{w}_1, \mathbf{w}_2, \mathbf{w}_3$ der Größe 3 und nach jeder Schicht eine elementweise Nichtlinearität.
    1. Schreiben Sie für jeden Stapel die Ausgabe an Position $i$ als Funktion der Eingabe. Zeigen Sie, dass die Ausgabe des Stapels mit Stride an Position $i$ gleich der Ausgabe des dilatierten Stapels an Position $4i$ ist.
    1. Zählen Sie für eine Eingabe der Länge $n$ die Multiplikationen, die jeder Stapel ausführt. Vernachlässigen Sie dabei den Rand.
    1. Verallgemeinern Sie den ersten Teil. Welche Dilatationsraten bilden einen Stapel mit den Strides $s_1, \ldots, s_L$ nach?

<!--en:f87c6e0f-->
4. **Ausgabeformen der Eingangsschichten.** Ein $224\times 224$-Bild durchläuft vier Schichten. Auf eine $7\times 7$-Faltung mit Padding 3 und Stride 2 folgen eine $3\times 3$-Faltung mit Padding 1 und Stride 2 und zwei $3\times 3$-Faltungen mit Padding 1 und Stride 1.
    1. Geben Sie für jede Schicht die Höhe der Ausgabe an.
    1. Berechnen Sie mit :eqref:`eq_receptive_field` das rezeptive Feld einer Einheit der letzten Schicht. Welchen Anteil der Bildbreite sieht sie?
    1. Ersetzen Sie die ersten beiden Schichten durch eine einzige $4\times 4$-Faltung mit Stride 4 ohne Padding, also durch die Patchify-Eingangsschicht aus diesem Abschnitt. Geben Sie erneut die Ausgabehöhe und das rezeptive Feld an. Überlappen sich benachbarte Fenster der ersten Schicht noch?

<!--en:9c565944-->
[Diskussion](https://d2l.discourse.group/t/67)

<!--en:da98f5a3-->
[Diskussion](https://d2l.discourse.group/t/68)

<!--en:5f479630-->
[Diskussion](https://d2l.discourse.group/t/272)

<!--en:3dee25a1-->
[Diskussion](https://d2l.discourse.group/t/17997)

<!--en:6d38b3e3-->
Padding, Stride und Dilatation<br>**Randbehandlung · räumliches Downsampling · erweiterter Kernbereich · Wachstum des rezeptiven Feldes**

<!--en:3dea6ae0-->
::: {.slide title="Drei geometrische Zielkonflikte bei einfachen Faltungen"}

<!--en:bf18c1e6-->
[Motivation und räumliche Zielkonflikte]{.kicker}

<!--en:3bb2a0d7-->
- **Randverlust $\to$ Padding ($p$):** Zehn $5 \times 5$-Schichten ohne Padding verkleinern ein Bild von $240 \times 240$ auf $200 \times 200$ Pixel (30% Flächenverlust). Padding mit $p = (k-1)/2$ erhält die Auflösung.

<!--en:70fb56da-->
- **Langsames lineares Wachstum des rezeptiven Feldes $\to$ Stride ($s$):** Bei Stride 1 brauchen wir für $3 \times 3$-Faltungen $>110$ Schichten, um $224 \times 224$ Pixel zu erfassen. Mit Stride $s > 1$ wächst die Reichweite geometrisch.

<!--en:bc39ba1f-->
- **Kontext bei voller Auflösung $\to$ Dilatation ($d$):** Die Abtastpunkte des Faltungskerns liegen $d$ Pixel auseinander. Die Parameteranzahl bleibt gleich.

<!--en:f589c777-->
::: {.slide title="Warum Faltungen ohne Padding Randpixel seltener abtasten"}

<!--en:9339a7d9-->
[Nutzung der Randpixel]{.kicker}

<!--en:06442400-->
Schon in einer einzigen $k \times k$-Schicht ohne Padding liegen Pixel nahe am Bildrand in weit weniger Fenstern als innere Pixel.

<!--en:4eb9a971-->
- **Inneres und Ecken:** Jedes innere Pixel im Abstand $\ge k-1$ vom Rand liegt in $k^2$ Fenstern ($25$ Fenster bei einem $5 \times 5$-Faltungskern), ein Randpixel in höchstens $k$ Fenstern und ein Eckpixel in **nur 1 Fenster**.

<!--en:6d4d3d27-->
- **Zunehmendes Übergewicht der Bildmitte:** Über mehrere Schichten hinweg nimmt das Gewicht der Information nahe dem Bildrand bei Faltungen ohne Padding exponentiell ab.

<!--en:f3365d64-->
::: {.slide title="Zero-Padding erhält Bildränder und Ausgabegröße"}

<!--en:0d99853e-->
[Symmetrisches Padding pro Seite]{.kicker}

<!--en:117a0d0a-->
Ergänzen wir oben und unten je $p_\textrm{h}$ Zeilen aus Nullen und links und rechts je $p_\textrm{w}$ Spalten aus Nullen, wächst das effektive Eingabegitter auf $(n_\textrm{h} + 2p_\textrm{h}) \times (n_\textrm{w} + 2p_\textrm{w})$.

<!--en:977dd02e-->
- **Formerhaltendes Padding („same“):** Mit $p_\textrm{h} = (k_\textrm{h}-1)/2$ und $p_\textrm{w} = (k_\textrm{w}-1)/2$ gleichen wir die Verkleinerung um $k - 1$ aus. Die Ausgabe hat dann die Form $n_\textrm{h} \times n_\textrm{w}$.

<!--en:5646658d-->
- **Beispiel ($3 \times 3$-Eingabe, $p=1$, $2 \times 2$-Faltungskern):** Durch Padding wächst die Eingabe ($3 \times 3 \to 5 \times 5$). Die Ausgabe hat dann die Form $4 \times 4$ statt $2 \times 2$.

<!--en:93818bda-->
::: {.slide title="Warum Faltungskerne fast immer eine ungerade Größe haben"}

<!--en:bb86dc3f-->
[Ausrichtung der Gitter bei symmetrischem Padding]{.kicker}

<!--en:7b285dec-->
- **Ungerades $k \in \{1, 3, 5, 7\}$:** $k - 1$ ist gerade. Mit $p = (k-1)/2$ ergänzen wir beide Seiten gleichmäßig und zentrieren jedes Fenster auf dem zugehörigen Eingabepixel $(i, j)$.

<!--en:6e568516-->
- **Schwierigkeit bei geradem $k$:** $k - 1$ ist ungerade. Das erzwingt asymmetrisches Padding ($p_{\textrm{left}} \ne p_{\textrm{right}}$) und verschiebt die Koordinaten zwischen den Schichten um ein halbes Pixel.

<!--en:557484f9-->
::: {.slide title="Formerhaltendes Padding im Code überprüfen"}

<!--en:5c731266-->
[Hilfsfunktion für Batch- und Kanalachse]{.kicker}

<!--en:62ed7b52-->
Die eingebauten 2D-Faltungsschichten erwarten 4D-Tensoren mit Batch- und Kanalachse. Unsere Hilfsfunktion `comp_conv2d` ergänzt eine 2D-Matrix der Form $(H, W)$ um eine Batch- und eine Kanalachse der Länge eins und gibt die räumliche Ausgabe als 2D-Matrix zurück.

<!--en:ced56228-->
::: {.slide title="Wie Zero-Padding einen dunklen Rahmen erzeugt"}

<!--en:7fd77227-->
[Implizite Positionshinweise, spiegelndes Padding und Padding durch Wiederholung]{.kicker}

<!--en:aac66a79-->
- **Dunkler Rand ($9 \to 6 \to 4$):** Zero-Padding erzeugt einen künstlichen dunklen Rahmen, der tiefen Schichten die absolute Position im Bild verrät.

<!--en:35f2275d-->
- **Abhilfe bei der Bilderzeugung:** Spiegelndes Padding oder Padding durch Wiederholung erhält eine konstante Intensität bis zum Rand.

<!--en:df9ca757-->
::: {.slide title="Faltungen mit Stride für das Downsampling des Gitters"}

<!--en:29df5332-->
[Verschiebung des Fensters um s > 1 Pixel pro Schritt]{.kicker}

<!--en:b707d505-->
Bei $s = 1$ werten wir den Faltungskern an jedem Pixel aus. Mit vertikalem Stride $s_\textrm{h}$ und horizontalem Stride $s_\textrm{w}$ verschieben wir das Fenster pro Schritt um $s_\textrm{h}$ Zeilen und $s_\textrm{w}$ Spalten.

<!--en:1a429866-->
- **Beispiel ($s_\textrm{h} = 3$, $s_\textrm{w} = 2$ bei einer $3 \times 3$-Eingabe mit Padding):**
  - Fenster oben links (grau): $0\cdot 0 + 0\cdot 1 + 0\cdot 2 + 0\cdot 3 = 0$.
  - Um 2 Spalten nach rechts verschoben (blau): $0\cdot 0 + 0\cdot 1 + 1\cdot 2 + 2\cdot 3 = 8$.
  - Um 3 Zeilen nach unten verschoben (orange): $0\cdot 0 + 6\cdot 1 + 0\cdot 2 + 0\cdot 3 = 6$.

<!--en:e66b8254-->
- **Rechenersparnis:** Werten wir nur jede $s$-te Zeile und Spalte aus, sinkt die Zahl der Ausgabeaktivierungen und Multiplikationsadditionen um den Faktor $s_\textrm{h} s_\textrm{w}$.

<!--en:9dd2c78f-->
::: {.slide title="Ausgabeform bei Faltungen mit Stride herleiten"}

<!--en:03440481-->
[Zulässige Schritte und Ausgangsposition zählen]{.kicker}

<!--en:aaa04f58-->
- **Schritte zählen:** Das erste Fenster der Größe $k$ beginnt am Index $0$. Auf den verbleibenden $n + 2p - k$ Pixeln ist Platz für $\lfloor (n+2p-k)/s \rfloor$ Schritte der Weite $s$.

<!--en:1319c338-->
- **Mit formerhaltendem Padding ($2p = k - 1$):** Die Ausgabehöhe vereinfacht sich zu $h_\textrm{o} = \lceil n / s \rceil$. Die Auflösung wird durch $s$ geteilt und aufgerundet.

<!--en:b3420b9f-->
::: {.slide title="Downsampling mit Stride im Code und beim Netzentwurf"}

<!--en:1987095f-->
[Drei typische Kombinationen von Padding und Stride]{.kicker}

<!--en:c6bd4b18-->
- **Drei Standardfälle:** Auflösung erhalten ($k=3, p=1, s=1$), beim Übergang zur nächsten Stufe halbieren ($k=3, p=1, s=2$) oder die Eingabe in überlappungsfreie Bildausschnitte für die Token zerlegen ($p=0, s=k$).

<!--en:02004b37-->
::: {.slide title="Geometrisches Wachstum des rezeptiven Feldes durch Stride"}

<!--en:648e90dc-->
[Wie frühe Strides die Reichweite aller nachfolgenden Faltungskerne vervielfachen]{.kicker}

<!--en:67c223c5-->
- **Gesamtfaktor $\prod_{j < i} s_j$:** Jeder frühe Stride vervielfacht die räumliche Reichweite aller darüberliegenden Schichten, unabhängig vom Padding $p$.

<!--en:438b240d-->
- **Lineares und geometrisches Wachstum:** Drei $3 \times 3$-Schichten erreichen $r = 7$ bei $s=(1,1,1)$ und $r = 15$ bei $s=(2,2,1)$.

<!--en:305a4fe8-->
::: {.slide title="Dilatierte Faltungen erweitern den erfassten Bereich bei Stride 1"}

<!--en:bb3c4876-->
[Abstand der Abtastpunkte im Faltungskern bei Dilatationsrate d]{.kicker}

<!--en:1b40ed45-->
Manche Aufgaben erfordern ein großes rezeptives Feld *ohne* Verlust an räumlicher Auflösung, etwa die pixelweise semantische Segmentierung. Durch **Dilatation** bleiben zwischen benachbarten Abtastpunkten des Faltungskerns jeweils $d - 1$ Pixel frei.

<!--en:512b62ba-->
- **Effektive Kerngröße $k'$:** Ein $k \times k$-Faltungskern mit Dilatationsrate $d$ erreicht folgende Spannweite:
  $$k' = k + (k - 1)(d - 1) = d(k - 1) + 1$$

<!--en:f173637d-->
- **Keine zusätzlichen Parameter oder FLOPs:** Bei $d = 2$ erfasst ein $3 \times 3$-Faltungskern einen $5 \times 5$-Bereich ($k' = 2(2)+1 = 5$). Dafür genügen **$9$ Gewichte und $9$ Multiplikationsadditionen** pro Pixel statt $25$.

<!--en:85bc43f9-->
::: {.slide title="Gemeinsame Formel für die Ausgabeform bei Padding, Stride und Dilatation"}

<!--en:4618cf50-->
[Einsetzen der effektiven Kerngröße k' = d(k - 1) + 1]{.kicker}

<!--en:e0579832-->
Ersetzen wir $k$ durch $k' = d(k - 1) + 1$ in der Formel für die Ausgabeform bei einer Faltung mit Stride, erhalten wir eine allgemeine Gleichung für jede Faltungsschicht entlang einer Achse:

<!--en:fc33cfa0-->
- **Plausibilitätsprüfung:** Bei einer $8 \times 8$-Eingabe ohne Padding ($p=0, s=1$) hat ein $3 \times 3$-Faltungskern mit Dilatation $d=2$ die effektive Größe $k' = 5$. Er liefert genau dieselbe $4 \times 4$-Ausgabeform wie ein dichter $5 \times 5$-Faltungskern.

<!--en:4d83de10-->
::: {.slide title="Schichtstapel mit Stride und mit Dilatation im Vergleich"}

<!--en:beaa7f85-->
[Reichweite und Dichte bei Strides (2, 2, 1) und Dilatationsraten (1, 2, 4)]{.kicker}

<!--en:d81317fb-->
Ersetzen wir $k_i - 1$ durch $k'_i - 1 = d_i(k_i - 1)$ in der Formel für das rezeptive Feld, sehen wir, wie sich eine Verdopplung der Dilatationsraten gemäß $d_i = 2^{i-1}$ bei Stride 1 auswirkt. Das rezeptive Feld wächst **exponentiell** ($r = 2^{L+1} - 1$ für $3 \times 3$-Faltungskerne).

<!--en:3e852a96-->
- **(a) Stapel mit Stride ($s = 2, 2, 1$, $d = 1, 1, 1$):** Erfasst $r = 1 + 2 + 4 + 8 = 15$ Eingabepixel. Die Unterabtastung behält pro Achse nur jede vierte Position bei (in 2D $16\times$ weniger Rechenaufwand).

<!--en:36154d31-->
- **(b) Stapel mit Dilatation ($s = 1, 1, 1$, $d = 1, 2, 4$):** Erfasst über identische Verbindungen genau dieselben $r = 1 + 2(1) + 2(2) + 2(4) = 15$ Eingabepixel und wertet dabei **jede räumliche Position** bei voller Auflösung aus.

<!--en:652eda86-->
::: {.slide title="Gitterartefakte bei dilatierten Faltungen"}

<!--en:9567807f-->
[Warum konstante Dilatationsraten Lücken im Abtastgitter lassen]{.kicker}

<!--en:c8829717-->
- **Gefahr von Gitterartefakten:** Bei konstanter Dilatation $d = (4, 4, 4, 4)$ liegen die Abtastpunkte ausschließlich an den Positionen $(i+4a, j+4b)$. Sie erfassen nur $81$ von $1{,}089$ Pixeln ($7.4\%$) im $33 \times 33$-Bereich.

<!--en:1f0f86cf-->
- **Blind für feine Strukturen:** Kleine Objekte zwischen den Gitterlinien im Abstand von $4$ Pixeln erfasst keines der Gewichte.

<!--en:b7405c20-->
- **Abhilfe durch exponentielles Wachstum:** Verdoppeln wir die Rate nach dem Schema $d = (1, 2, 4, 8)$, schließen sich alle Lücken. Die Abdeckung beträgt $100\%$ ($r = 31$).

<!--en:8c9c2da1-->
::: {.slide title="Verständnisfragen zu Ausgabeformen und rezeptiven Feldern"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:a52334f7-->
Überprüfen Sie, wie sicher Sie mit den allgemeinen Gleichungen für die Ausgabeform und das rezeptive Feld umgehen.

<!--en:93bcee8a-->
1. **Stapel ohne und mit Padding:** Welche Ausgabeform ergibt sich, wenn ein $28 \times 28$-Bild drei $5 \times 5$-Schichten ohne Padding mit Stride 1 durchläuft? Welches Padding $p$ pro Seite erhält die Form $28 \times 28$ über alle drei Schichten hinweg?

<!--en:7a44956c-->
2. **Eingangsschichten wie bei ResNet:** Ein $224 \times 224$-Bild durchläuft eine $7 \times 7$-Faltung ($p=3, s=2$) und anschließend eine $3 \times 3$-Faltung ($p=1, s=2$). Welche räumliche Form hat die Ausgabe nach jeder Schicht? Wie groß ist das rezeptive Feld $r$ einer Einheit nach der zweiten Schicht?

<!--en:06207387-->
3. **Formerhaltendes Padding bei Dilatation:** Für einen $3 \times 3$-Faltungskern mit Dilatation $d = 4$ und Stride $s = 1$: Welches Padding $p$ pro Seite erhält die Höhe und Breite der Eingabe?

<!--en:d7036035-->
**Antworten:**
(1) Jede $5 \times 5$-Schicht ohne Padding verkürzt jede Achse um $4$ Pixel: $28 \to 24 \to 20 \to 16$ ($16 \times 16$). Mit Padding $p = (5-1)/2 = 2$ auf jeder Seite bleibt die Form $28 \times 28$ erhalten.
(2) Ausgabe von Schicht 1: $\lfloor(224+6-7)/2\rfloor + 1 = 112 \times 112$. Ausgabe von Schicht 2: $\lfloor(112+2-3)/2\rfloor + 1 = 56 \times 56$. Rezeptives Feld nach Schicht 2: $r = 1 + (7-1)(1) + (3-1)(2) = 1 + 6 + 4 = 11$ Pixel.
(3) Die effektive Kerngröße beträgt $k' = 4(3-1)+1 = 9$. Formerhaltendes Padding erfordert daher $p = (k'-1)/2 = 4$ Pixel pro Seite.

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und wichtigste Ergebnisse"}

<!--en:ce12e85f-->
[Überblick über den Abschnitt]{.kicker}

<!--en:5c38ee2c-->
| Hyperparameter | Einfluss auf die Ausgabegröße $n_\textrm{out}$ | Einfluss auf das rezeptive Feld $r$ | Rolle für Berechnung und Architektur |

<!--en:b8c6b292-->
| **Padding ($p$ pro Seite)** | Verlängert die effektive Eingabe um $2p$ | Keiner ($r$ ist unabhängig von $p$) | Verhindert Randverlust. $p = (k-1)/2$ erhält $H \times W$ bei $s=1$ |

<!--en:b8219f6e-->
| **Stride ($s$)** | Teilt die Ausgabelänge durch $s$ ($\approx \lceil n/s \rceil$) | Multipliziert die Reichweite aller folgenden Schichten mit $s$ | Senkt die Anzahl der Aktivierungen und Multiplikationsadditionen um den Faktor $s^2$. Das rezeptive Feld wächst geometrisch |

<!--en:53776224-->
| **Dilatation ($d$)** | Vergrößert den Faltungskern gemäß $k \to k' = d(k-1)+1$ | Addiert $d(k-1) \prod_{j<i} s_j$ zur Feldbreite $r$ | Exponentielles Wachstum des rezeptiven Feldes bei voller Auflösung ($0$ zusätzliche Parameter) |

<!--en:5c6bc7dd-->
| **Allgemeine Formel** | $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$ | $1 + \sum_{i=1}^L d_i(k_i - 1)\prod_{j<i} s_j$ | Gilt für alle Faltungs- und Pooling-Schichten in 1D, 2D und 3D |

<!--en:5ec9d0bc-->
**Ausblick (§{sec_channels}):** Wir erweitern Faltungen von einkanaligen 2D-Gittern auf 3D-Stapel mit Eingabe- und Ausgabekanälen, berücksichtigen den mit $c_\textrm{i} c_\textrm{o}$ skalierenden Rechen- und Speicheraufwand und bauen $1 \times 1$-Flaschenhalsschichten.
