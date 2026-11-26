<!--en:33d3b96b-->
# Faltungen für Bilder

<!--en:852f5868-->
In :numref:`sec_why-conv` haben wir die Faltungsschicht (engl. convolutional layer) auf dem Papier aus zwei geometrischen Prinzipien hergeleitet: Translationsäquivarianz und Lokalität. Nun sehen wir uns an, was diese Formel für ein Bild berechnet und ob der Gradientenabstieg (engl. gradient descent) geeignete Faltungskerne (engl. kernels) selbstständig aus Daten lernen kann.

<!--en:55f0a41b-->
Wie finden wir vertikale Kanten, an denen ein heller Bereich an einen dunklen grenzt? Ziehen wir vom Wert jedes Pixels den seines rechten Nachbarn ab ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), erhalten wir in gleichmäßig hellen Bereichen null, beim Übergang von Weiß zu Schwarz $+1$ und beim Übergang von Schwarz zu Weiß $-1$. Schieben wir den Detektor mit den beiden Gewichten $(1, -1)$ über alle gültigen Bildpositionen, entsteht eine *Kantenkarte* (engl. edge map). In diesem Abschnitt implementieren wir die zweidimensionale Kreuzkorrelation und erproben unseren von Hand entworfenen Kantendetektor. Dann lassen wir den Gradientenabstieg $(1, -1)$ direkt aus Daten *lernen*. Wir zeigen, wie sich eine Faltung (engl. convolution) als eine einzige Matrizenmultiplikation schreiben lässt (*im2col*), und leiten her, wie sich das rezeptive Feld (engl. receptive field) einer Einheit mit jeder weiteren Schicht vergrößert.

<!--en:47958ed8-->
## Kreuzkorrelation

<!--en:5e4296fe-->
### Die Rechenoperation

<!--en:ae7e906d-->
Lassen wir die Kanäle zunächst außer Acht. Die Eingabe $\mathbf{X}$ sei eine Matrix der Form $n_\textrm{h} \times n_\textrm{w}$, der Faltungskern $\mathbf{K}$ eine Matrix der Form $k_\textrm{h} \times k_\textrm{w}$. Der Faltungskern legt ein gleitendes *Fenster* der Form $k_\textrm{h} \times k_\textrm{w}$ fest. Wir setzen die linke obere Ecke des Fensters auf $(i, j)$, multiplizieren den überdeckten Eingabeausschnitt elementweise mit $\mathbf{K}$ und addieren die Produkte. So erhalten wir den Ausgabewert an der Position $(i, j)$:

<!--en:3688b38b-->
Das ist die *zweidimensionale Kreuzkorrelation* von $\mathbf{X}$ mit $\mathbf{K}$. Wir schreiben $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ und zählen die Indizes wie im Code ab null. In :numref:`sec_why-conv` lagen die Versätze symmetrisch zwischen $-\Delta$ und $\Delta$. Verwenden wir die linke obere Ecke als Bezugspunkt, ändert sich die Lage nur um eine feste Verschiebung von $\Delta$. So lassen sich auch Faltungskerne mit gerader Seitenlänge verwenden, etwa ein $2 \times 2$-Faltungskern oder unser $1 \times 2$-Kantendetektor, die kein mittleres Pixel besitzen.

<!--en:fa1b7da9-->
![Zweidimensionale Kreuzkorrelation. Schattiert sind das erste Ausgabeelement und die Elemente von Eingabe und Faltungskern, aus denen es sich berechnet: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb-->
In :numref:`fig_correlation` schieben wir einen $2 \times 2$-Faltungskern pixelweise über eine $3 \times 3$-Eingabe, von links nach rechts und von oben nach unten. An vier Positionen liegt das Fenster vollständig in der Eingabe. Daraus ergibt sich die $2 \times 2$-Ausgabe:

<!--en:4741e012-->
Die Ausgabe ist entlang jeder Achse etwas kleiner als die Eingabe, weil der Faltungskern vollständig in die Eingabe passen muss. Für ein Fenster der Breite $k$ gibt es auf einer Strecke der Länge $n$ genau $n - k + 1$ gültige Positionen. Die Ausgabe hat also die Form

<!--en:12d81c14-->
Eine $3 \times 3$-Eingabe mit einem $2 \times 2$-Faltungskern liefert somit eine Ausgabe der Form $2 \times 2$. Bei einem $6 \times 8$-Bild mit einem $1 \times 2$-Faltungskern ergibt sich $6 \times 7$, bei einem $28 \times 28$-Bild aus Fashion-MNIST mit einem $5 \times 5$-Faltungskern ergibt sich $24 \times 24$. Wollen wir die räumlichen Abmessungen erhalten, umgeben wir die Eingabe mit einem Rand aus Nullen (*Padding*). In :numref:`sec_padding` betrachten wir Padding zusammen mit Stride und Dilatation.

<!--en:869d223f-->
### Implementierung

<!--en:6a869f7c-->
Wir setzen :eqref:`eq_corr2d` direkt in `corr2d` um. Zwei verschachtelte Schleifen durchlaufen alle Ausgabekoordinaten.

<!--en:81e1f383-->
Um `corr2d` mit unserer Rechnung aus :numref:`fig_correlation` zu vergleichen, legen wir den $3 \times 3$-Eingabetensor `X` und den $2 \times 2$-Faltungskern `K` an und berechnen `corr2d(X, K)`.

<!--en:f567e1b2-->
Die Ausgabe stimmt Eintrag für Eintrag mit :numref:`fig_correlation` überein. Wie wir in :numref:`sec_why-conv` gesehen haben, lässt sich mit der mathematischen Faltung und einem um 180° gedrehten Faltungskern $\mathbf{K}'$ genau dieselbe Familie von Schichten lernen. Wir folgen daher dem Sprachgebrauch im Deep Learning und nennen die Kreuzkorrelation „Faltung“.

<!--en:bfe11c62-->
## Faltungsschichten

<!--en:2cdc020d-->
### Aufbau der Schicht

<!--en:c6371ece-->
Eine zweidimensionale Faltungsschicht berechnet die Kreuzkorrelation ihrer Eingabe mit einem gelernten Faltungskern und addiert einen skalaren Bias. Wie bei den MLP-Gewichten in :numref:`chap_perceptrons` ziehen wir die Anfangswerte des Faltungskerns aus $\mathcal{N}(0, 0.01^2)$ und setzen den Bias auf null. Im folgenden Code registriert `Conv2D` beide Parameter und ruft im Vorwärtsdurchlauf `corr2d` auf. Hat der Faltungskern die Höhe $h$ und die Breite $w$, sprechen wir von einer $h \times w$-Faltungsschicht.

<!--en:29cd9061-->
Anders als bei einer vollständig verbundenen Schicht ist die Eingabeauflösung bei einer Faltungsschicht frei wählbar, denn ihre Parameter hängen nur von der Kerngröße ab. Dieselbe $2 \times 2$-Schicht mit insgesamt 5 Parametern verarbeitet sowohl unseren $3 \times 3$-Tensor `X` als auch ein $6 \times 8$-Bild. Die Ausgaben haben die Formen $2 \times 2$ und $5 \times 7$:

<!--en:c9e5aa07-->
### Kantenerkennung

<!--en:3d9a357b-->
Wir testen nun den Detektor für vertikale Kanten aus der Einleitung an einem synthetischen $6 \times 8$-Bild. Die mittleren vier Spalten sind schwarz (`0`), die äußeren weiß (`1`).

<!--en:68513ff0-->
Wir definieren den $1 \times 2$-Faltungskern

<!--en:eb45803b-->
An jeder Position $(i, j)$ berechnet er die horizontale *endliche Differenz* $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$. Bis auf das Vorzeichen nähert sie die horizontale Ableitung der Helligkeit diskret an. In gleichmäßig hellen Bereichen ist sie null, an Helligkeitssprüngen schlägt sie aus.

<!--en:e810b3a9-->
Für das $6 \times 8$-Streifenbild `X` und den $1 \times 2$-Faltungskern `K` für die horizontale Differenz berechnet `corr2d(X, K)` die Kreuzkorrelation. Das Ergebnis ist die $6 \times 7$-Kantenkarte `Y`:

<!--en:568358e4-->
Wie erwartet hat `Y` an der Grenze von Weiß zu Schwarz den Wert `+1`, an der Grenze von Schwarz zu Weiß den Wert `-1` und überall sonst den Wert `0`. Wenden wir denselben Faltungskern auf das transponierte Bild an, wird aus dem vertikalen Streifen ein horizontaler und wir erhalten nur Nullen:

<!--en:71c591a4-->
Der Faltungskern `[1, -1]` reagiert hier also nur auf die vertikalen Kanten. Für horizontale Kanten, Diagonalen, Ecken oder Texturen benötigen wir andere Faltungskerne.

<!--en:f0af4887-->
### Lernen eines Faltungskerns

<!--en:61cacfaf-->
Wenn wir das gesuchte Merkmal genau kennen, ist es leicht, `[1, -1]` von Hand zu entwerfen. Für die Zwischenschichten eines tiefen Netzes kann jedoch niemand Tausende von Filtern von Hand vorgeben. Kann der Gradientenabstieg den Faltungskern `[1, -1]` allein aus dem Eingabe-Ausgabe-Paar `(X, Y)` rekonstruieren? Wir testen das mit der integrierten 2D-Faltungsschicht jedes Frameworks, ohne Bias und mit Anfangsgewichten aus $\mathcal{N}(0, 0.01^2)$. Wir minimieren die Summe der quadratischen Fehler über alle 42 Ausgabeeinträge mit 10 Gradientenschritten bei einer Lernrate von $0.03$. Die integrierten Schichten erwarten vierdimensionale Tensoren mit einer Batch- und einer Kanalachse (:numref:`sec_channels`). Daher bringen wir `X` und `Y` in die passende Form.

<!--en:c29c5d61-->
Anfangs liegt der Verlust (engl. loss) nahe bei 12: Der zufällig initialisierte Faltungskern hat Gewichte nahe null und verfehlt daher bei allen 12 Kantenpixeln den Zielwert $\pm 1$ um etwa 1. Schon nach neun Schritten liegt der Verlust unter 0.01. Ein Blick auf den gelernten Faltungskern zeigt das Ergebnis:

<!--en:67d25b0b-->
Beide Einträge weichen um höchstens etwa 0.02 von den entsprechenden Werten in `[1, -1]` ab. Die Schleife enthielt keinerlei Vorwissen über Kanten. Allein anhand des Verlustsignals hat der Gradientenabstieg den Filter gefunden. Hier haben wir die einzelne Schicht direkt mit `Y` als Zielvorgabe trainiert. In einem tiefen CNN lernt das Netz mithilfe der Backpropagation die Faltungskerne aller Schichten durchgängig allein aus dem Verlust für die endgültige Aufgabe.

<!--en:0b85caf9-->
## Faltung als Matrizenmultiplikation

<!--en:5bdffcd7-->
Die verschachtelten Schleifen in `corr2d` machen die Rechnung nachvollziehbar. Für praktische Anwendungen ist es jedoch viel zu langsam, jedes Ausgabepixel einzeln in Python zu berechnen. Eine klassische Lösung, die Bibliotheken bis heute einsetzen, fasst die Kreuzkorrelation zu einer einzigen großen Matrizenmultiplikation zusammen. Gerade bei dieser Operation erreichen GPUs ihre höchste Rechengeschwindigkeit. Jeder Ausgabeeintrag in :numref:`fig_correlation` ist ein Skalarprodukt: Wir schreiben die $k_\textrm{h} k_\textrm{w}$ Elemente des Faltungskerns und das zugehörige Eingabefenster jeweils als Vektor. Extrahieren wir alle $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ Fenster und schreiben jedes als Zeile in eine Matrix, erhalten wir die Ausschnittsmatrix. Multiplizieren wir diese mit dem Faltungskern in Vektorform, ergibt sich die gesamte Kreuzkorrelation in einem Schritt. Diese übliche Umordnung der Daten heißt *im2col* („image to columns“, also vom Bild zu Spalten, hier zu Zeilen). :numref:`fig_conv_im2col` zeigt die Umordnung am Beispiel aus :numref:`fig_correlation`.

<!--en:107554a5-->
![Die Kreuzkorrelation aus :numref:`fig_correlation` als ein einziges Matrizenprodukt. Jedes der vier $2 \times 2$-Fenster der Eingabe wird zu einer Zeile der Ausschnittsmatrix. Die Multiplikation mit dem Faltungskern in Vektorform liefert die Ausgabe als Vektor. Die schattierten Zellen zeigen den Weg vom ersten Fenster über seine Matrixzeile zum Ausgabewert 19. Das mittlere Pixel mit dem Wert 4 liegt in allen vier Fenstern und erscheint daher in jeder Zeile.](../img/conv-im2col.svg)

<!--en:76713479-->
Dafür benötigen wir zusätzlichen Arbeitsspeicher, denn jedes innere Pixel erscheint in bis zu $k_\textrm{h} k_\textrm{w}$ Ausschnitten. Am Beispiel aus :numref:`fig_correlation` prüfen wir, dass die eine Matrizenmultiplikation genau dasselbe Ergebnis liefert wie `corr2d`:

<!--en:3bcd7d8a-->
Die Abweichung ist überall null: Ein einziges Matrizenprodukt liefert genau das Ergebnis von `corr2d`.

<!--en:c30f2dea-->
Gleichwertig dazu können wir anstelle der Eingabe den *Faltungskern* zu einer dünn besetzten Bandmatrix $\mathbf{W}$ entfalten. Wir multiplizieren sie mit dem Bild in Vektorform. Das entspricht im zweidimensionalen Fall genau Teilbild (c) in :numref:`fig_why_conv_weight_sharing`: In jeder Zeile stehen dieselben wenigen Gewichte des Faltungskerns. Interpretieren wir die Faltung als vollständig verbundene Schicht mit strukturierter Gewichtsmatrix, gelangen wir wieder zur Herleitung aus :numref:`sec_why-conv`. In den Übungen konstruieren wir $\mathbf{W}$ explizit.

<!--en:ad4fc956-->
## Merkmalskarten und rezeptive Felder

<!--en:e7983df8-->
Eine einzelne Faltungsschicht erfasst nur einen Ausschnitt von der Größe ihres Faltungskerns. Um ganze Objekte zu erkennen, muss das Netz Informationen aus wesentlich größeren Bereichen zusammenführen. Wie wir aus :numref:`subsec_why-conv-channels` wissen, heißt die Ausgabe einer Faltungsschicht *Merkmalskarte* (engl. feature map). Ein Beispiel ist unsere Karte der vertikalen Kanten `Y`. Das *rezeptive Feld* eines Elements $x$ in einer Merkmalskarte umfasst den Bereich des Eingabebildes (oder einer früheren Schicht), dessen Werte $x$ im Vorwärtsdurchlauf beeinflussen können. Der Begriff stammt aus der Neurophysiologie (siehe :numref:`field_visual`).

<!--en:b36f221e-->
### Aufeinanderfolgende Schichten

<!--en:24a25ec4-->
In :numref:`fig_correlation` hat der schattierte Ausgabewert `19` ein rezeptives Feld der Größe $2 \times 2$. Es umfasst die vier Pixel links oben in $\mathbf{X}$. Setzen wir nun eine *zweite* $2 \times 2$-Faltungsschicht auf die $2 \times 2$-Ausgabe $\mathbf{Y}$, erhalten wir einen einzigen Skalar $z$. Dieser Wert $z$ hängt von allen vier Elementen von $\mathbf{Y}$ ab. Diese hängen zusammen von allen neun Elementen von $\mathbf{X}$ ab. Mit jeder weiteren Schicht wächst also das rezeptive Feld.

<!--en:f95c9394-->
Wie schnell wächst es? Jede $k \times k$-Faltungsschicht erfasst entlang jeder Achse $k - 1$ Pixel mehr als ein einzelnes Element ihrer Eingabe. Für $L$ aufeinanderfolgende Schichten mit den Kerngrößen $k_1, \ldots, k_L$ hat das rezeptive Feld somit die Seitenlänge

<!--en:b302e3ef-->
Bei $L$ aufeinanderfolgenden $3 \times 3$-Schichten nimmt die Seitenlänge mit jeder Schicht um 2 Pixel zu. Das rezeptive Feld umfasst $(2L + 1) \times (2L + 1)$ Eingabepixel: $5 \times 5$ nach zwei Schichten und $7 \times 7$ nach drei (:numref:`fig_conv_receptive_field`). Dieses lineare Wachstum ist ein Vorteil. Wir brauchen keine riesigen Faltungskerne: Drei aufeinanderfolgende $3 \times 3$-Schichten erreichen dasselbe rezeptive Feld der Größe $7 \times 7$ wie eine einzelne $7 \times 7$-Schicht. Dabei benötigen sie deutlich weniger Parameter ($3 \times 9 = 27$ Gewichte statt 49 pro Kanalpaar) und kombinieren drei Nichtlinearitäten statt nur einer.

<!--en:39917513-->
![Rezeptives Feld einer Einheit nach drei eindimensionalen Schichten mit Kerngröße 3. Schattiert sind die Einheiten, von denen die oberste Einheit abhängt. Die Linien verbinden jede dieser Einheiten mit ihren Eingaben. Jede Schicht verbreitert das rezeptive Feld um 2, bis es 7 Eingabeelemente umfasst.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc-->
### Effektive rezeptive Felder

<!--en:77d044db-->
Die Größe $r$ beschreibt das *theoretische* rezeptive Feld. Es umfasst jedes Eingabepixel, von dem *irgendein* Pfad zu einer Ausgabeeinheit führt. Wie stark jedes Pixel tatsächlich beiträgt, bleibt dabei offen. Durch direkte Messung der Eingabegradienten zeigten :citet:`Luo.Li.Urtasun.ea.2016`, dass der Einfluss im *effektiven* rezeptiven Feld vom Zentrum nach außen stark abnimmt und annähernd einer gaußschen Glockenkurve folgt. Das gilt für untrainierte ebenso wie für trainierte Netze. Einheiten in tiefen Schichten können theoretisch weit entfernte Bildbereiche erfassen, reagieren aber vor allem auf die Umgebung des Zentrums.

<!--en:eed4e2a0-->
Die Glockenform lässt sich erklären, indem wir Pfade zählen. Von einem Eingabeelement am Rand des rezeptiven Feldes führt nur eine einzige Kette von Verbindungen zur Ausgabeeinheit. Vom mittleren Eingabeelement führen viele Wege durch die dazwischenliegenden Schichten. :numref:`fig_conv_rf_paths` zeigt die Anzahl der Pfade für die Schichtenfolge aus :numref:`fig_conv_receptive_field`: Vom mittleren Eingabeelement führen sieben Pfade zur obersten Einheit, von den beiden äußeren jeweils einer. Sind die Beträge aller Gewichte ähnlich, wächst der Einfluss eines Eingabeelements mit der Anzahl seiner Pfade.

<!--en:b92775ee-->
![Anzahl der Pfade von jeder Einheit zur obersten Einheit aus :numref:`fig_conv_receptive_field` (drei eindimensionale Schichten mit Kerngröße 3). Die Anzahl für eine Zelle ergibt sich als Summe der Anzahlen für alle Einheiten, an die sie ihren Wert weitergibt. Dunklere Zellen haben mehr Pfade als die anderen Zellen derselben Schicht. Die Balken zeigen nochmals die Anzahlen der Eingabezeile: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8-->
Unsere einfache Schicht hat noch zwei praktische Einschränkungen. Jede Faltung ohne Padding verkleinert die räumlichen Abmessungen und berücksichtigt Randpixel seltener. Außerdem erkennt ein einzelner Faltungskern nur ein Muster. Mit den räumlichen Abmessungen und den Rändern befassen wir uns als Nächstes in :numref:`sec_padding`, mit mehrkanaligen Merkmalskarten in :numref:`sec_channels`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:92688a0c-->
Eine Faltungsschicht schiebt einen gelernten Faltungskern über alle gültigen Fenster der Eingabe. Für jedes Fenster berechnet sie die zweidimensionale Kreuzkorrelation und addiert einen skalaren Bias. Der einfache $1 \times 2$-Faltungskern `[1, -1]` berechnet eine endliche Differenz und erkennt vertikale Kanten an jeder Stelle eines Bildes. Der Gradientenabstieg findet genau diese Gewichte allein aus Eingabe-Ausgabe-Daten, so wie tiefe CNNs ganze Hierarchien von Filtern durchgängig lernen. Bei der mathematischen Faltung ist der Faltungskern um 180° gedreht. Sobald das Netz die Faltungskerne lernt, spielt das keine Rolle. Eine Faltung lässt sich als eine einzige Matrizenmultiplikation auswerten (*im2col*). Bei $L$ aufeinanderfolgenden Faltungsschichten wächst die Seitenlänge des rezeptiven Feldes jeder Einheit linear um $k_i - 1$ pro Schicht.

<!--en:2294243a-->
## Übungen

<!--en:6f4ddec6-->
1. [code] **Von Hand entworfene Faltungskerne.** Entwerfen Sie einige Faltungskerne von Hand.
    1. Gegeben sei ein Richtungsvektor $\mathbf{v} = (v_1, v_2)$. Leiten Sie einen Faltungskern her, der Kanten orthogonal zu $\mathbf{v}$ erkennt, also Kanten in Richtung $(v_2, -v_1)$.
    1. Leiten Sie einen Differenzenoperator für die zweite Ableitung her. Bei welchen Bildstrukturen liefert er den stärksten Ausschlag?
    1. Wie groß muss ein Faltungskern mindestens sein, um eine Ableitung der Ordnung $d$ zu berechnen?
    1. Wie würden Sie einen Faltungskern zum Weichzeichnen entwerfen? Wofür könnten Sie ihn einsetzen?
    1. Laden Sie ein Foto, wandeln Sie es in ein einkanaliges Bild um und verkleinern Sie es auf etwa $100 \times 100$ Pixel (`corr2d` verwendet Schleifen in Python). Wenden Sie den Kantendetektor, den Weichzeichner und den Schärfungsfilter „zweimal die Identität minus Weichzeichner“ an. Zeigen Sie jedes Ergebnis neben dem Original. Was zeigt die Kantenkarte eines Fotos, das die Kantenkarte des schwarzen Streifens nicht zeigen konnte?

    *Nach Stanford CS231n, [Aufgabenblatt 2](https://cs231n.github.io/assignments2024/assignment2/), Notebook ConvolutionalNetworks.*

<!--en:ec8621f8-->
2. [code] **Faltung als Bandmatrix.** Für die Ausschnittsmatrix aus :numref:`subsec_conv_matmul` entfalten wir die Eingabe. Bei der umgekehrten Konstruktion entfalten wir den Faltungskern. Es sei $\mathbf{X}$ eine $4 \times 4$-Eingabe, $\mathbf{K}$ ein $3 \times 3$-Faltungskern und $\textrm{vec}(\cdot)$ die Operation, die eine Matrix zeilenweise als Vektor schreibt.
    1. Konstruieren Sie die $4 \times 16$-Matrix $\mathbf{W}$ mit $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$. Dabei ist $\mathbf{Y}$ die Kreuzkorrelation von $\mathbf{X}$ mit $\mathbf{K}$. Wie viele von null verschiedene Einträge hat jede Zeile? Wie viele verschiedene Werte treten in $\mathbf{W}$ auf?
    1. Konstruieren Sie $\mathbf{W}$ im Code aus `K` und prüfen Sie, ob das Produkt das Ergebnis von `corr2d(X, K)` liefert.
    1. Betrachten Sie eine $n \times n$-Eingabe und einen $k \times k$-Faltungskern. Wie viele Einträge speichern Sie für $\mathbf{W}$ als dichtes Array? Wie viele Einträge von $\mathbf{W}$ sind von null verschieden, und wie viele Einträge hat die Ausschnittsmatrix? Wie wächst jede dieser Anzahlen mit $n$?
    1. Behalten Sie das Besetzungsmuster von $\mathbf{W}$ bei, geben Sie aber jeder Zeile $k^2$ eigene Werte. Das Ergebnis ist eine lokal verbundene Schicht. Ihr entspricht der Tensor vierter Ordnung $\mathsf{V}$ aus :numref:`sec_why-conv` in der Gleichung $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$, beschränkt auf ein $k \times k$-Fenster von Versätzen $(a, b)$, bevor die Translationsäquivarianz die Gewichte aneinander bindet. Zählen Sie für eine $n \times n$-Eingabe die Parameter dieser Schicht, einer vollständig verbundenen Schicht mit denselben Eingabe- und Ausgabeformen und der Faltung.

    *Nach CMU 11-785, [Hausaufgabe 2, Teil 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

<!--en:54bcee26-->
3. [code] **Effektives rezeptives Feld.** ● Das theoretische rezeptive Feld $r$ aus diesem Abschnitt gibt an, welche Eingabeelemente eine Ausgabe beeinflussen *können*. Das effektive rezeptive Feld misst, wie stark sie dies tatsächlich tun :cite:`Luo.Li.Urtasun.ea.2016`. Schalten Sie $L$ einkanalige $3 \times 3$-Faltungsschichten ohne Padding, Bias oder Nichtlinearität für eine $32 \times 32$-Eingabe hintereinander. Verwenden Sie dazu die integrierte Schicht aus :numref:`subsec_learning_kernel`. Ziehen Sie jeden Eintrag jedes Faltungskerns unabhängig aus einer Standardnormalverteilung.
    1. Ziehen Sie für $L = 3$ einmal die Faltungskerne. Setzen Sie den Gradienten an einer Ausgabeeinheit nahe der Mitte der letzten Schicht auf eins und propagieren Sie ihn rückwärts bis zur Eingabe. Stellen Sie den Betrag des Eingabegradienten grafisch dar. Vergleichen Sie seinen Träger mit dem theoretischen rezeptiven Feld der Größe $7 \times 7$ und geben Sie an, wo der Betrag am größten ist.
    1. Mitteln Sie den quadrierten Eingabegradienten über 500 Ziehungen der Faltungskerne, einmal für $L = 3$ und einmal für $L = 10$. Summieren Sie jede gemittelte Karte über ihre Zeilen. Normieren Sie das resultierende Spaltenprofil so, dass sich seine Einträge zu eins summieren. Wie ändert sich das Verhältnis der Standardabweichung des Profils zur theoretischen Seitenlänge $2L + 1$ mit der Tiefe?
    1. :numref:`fig_conv_rf_paths` zeigt die Anzahl der Pfade für $L = 3$ in einer Dimension. Zeigen Sie für beliebiges $L$, dass der über die Zeilen summierte Erwartungswert des quadrierten Eingabegradienten proportional zur Faltung von $L$ Kopien des Vektors $(1, 1, 1)$ ist. Leiten Sie her, wie die Standardabweichung des normierten Profils mit $L$ wächst.

<!--en:6ab5b67c-->
4. [code] **Rückwärtsdurchlauf einer Faltung.** ● Es sei $\mathbf{Y}$ die einkanalige Kreuzkorrelation einer Eingabe $\mathbf{X}$ mit einem Faltungskern $\mathbf{K}$, wie sie `corr2d` berechnet. Der skalare Verlust $\ell$ hänge von $\mathbf{X}$ und $\mathbf{K}$ nur über $\mathbf{Y}$ ab.
    1. Drücken Sie $\partial \ell/\partial \mathbf{K}$ und $\partial \ell/\partial \mathbf{X}$ durch $\partial \ell/\partial \mathbf{Y}$ aus. Orientieren Sie sich bei der Herleitung an den Backpropagation-Gleichungen aus :numref:`sec_backprop`.
    1. Implementieren Sie Ihren Ausdruck für $\partial \ell/\partial \mathbf{K}$ ohne automatisches Differenzieren für den Verlust $\ell = \sum_{i,j} G_{ij} Y_{ij}$. Dabei ist $\mathbf{G}$ eine feste Zufallsmatrix mit derselben Form wie $\mathbf{Y}$. Vergleichen Sie das Ergebnis mit dem Gradienten aus zentrierten endlichen Differenzen mit der Schrittweite $10^{-2}$. Verwenden Sie kleine, zufällig erzeugte `X` und `K` und geben Sie den größten relativen Fehler an.
    1. Überprüfen Sie Ihren Ausdruck für $\partial \ell/\partial \mathbf{K}$ anhand des Gradienten des Faltungskerns, den das automatische Differenzieren im Experiment aus :numref:`subsec_learning_kernel` liefert.

    *Nach Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html), Kapitel 6, und Stanford CS231n, [Aufgabenblatt 2](https://cs231n.github.io/assignments2024/assignment2/), Notebook ConvolutionalNetworks.*

<!--en:657a396a-->
[Diskussion](https://d2l.discourse.group/t/65)

<!--en:63c44b48-->
[Diskussion](https://d2l.discourse.group/t/66)

<!--en:00b73d0b-->
[Diskussion](https://d2l.discourse.group/t/271)

<!--en:b79af65b-->
[Diskussion](https://d2l.discourse.group/t/17996)

<!--en:91b0010d-->
Faltungen für Bilder<br>**2D-Kreuzkorrelation · Kantenerkennung · gelernte Faltungskerne · im2col · rezeptive Felder**

<!--en:1d94295a-->
::: {.slide title="Lokale Kanten im gesamten Bild erkennen"}

<!--en:687ed302-->
[Motivation und Grundidee]{.kicker}

<!--en:05fceed0-->
- **Synthetisches $6 \times 8$-Testbild:** Weiße äußere Spalten (`1`) umgeben einen 4 Spalten breiten schwarzen vertikalen Streifen (`0`).

<!--en:71e700b3-->
- **Horizontale endliche Differenz:** Der gleitende Faltungskern $\mathbf{K} = [1, -1]$ berechnet $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$. Das ergibt `+1` an Kanten von Weiß zu Schwarz und `-1` an Kanten von Schwarz zu Weiß.

<!--en:957dc033-->
- **Lernen oder von Hand entwerfen:** Tiefe CNNs lernen Tausende solcher Faltungskerne automatisch durch Gradientenabstieg.

<!--en:cb1045e1-->
::: {.slide title="Berechnung der 2D-Kreuzkorrelation"}

<!--en:17bcece5-->
[Innere Produkte im gleitenden Fenster]{.kicker}

<!--en:8d45aae8-->
Gegeben sind eine $n_\textrm{h} \times n_\textrm{w}$-Eingabematrix $\mathbf{X}$ und ein $k_\textrm{h} \times k_\textrm{w}$-Faltungskern $\mathbf{K}$. Für die 2D-Kreuzkorrelation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ die linke obere Ecke des Fensters auf $(i, j)$ setzen und die elementweisen Produkte summieren:

<!--en:e484380c-->
- **Bezugspunkt links oben:** Die bei null beginnenden Indizes $a, b \ge 0$ entsprechen der Array-Indizierung im Code und eignen sich auch für Faltungskerne mit gerader Seitenlänge (etwa $2 \times 2$ oder $1 \times 2$).

<!--en:137c0980-->
- **Schattierter Ausgabewert links oben an Position $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea-->
::: {.slide title="Gültige Fensterpositionen und räumliche Form der Ausgabe"}

<!--en:7eda4531-->
[Warum Faltungen ohne Padding das Gitter verkleinern]{.kicker}

<!--en:891243ae-->
- **Gültige Fensterpositionen:** Ein $k_\textrm{h} \times k_\textrm{w}$-Fenster passt in ein $n_\textrm{h} \times n_\textrm{w}$-Gitter an $n - k + 1$ Positionen pro Achse.

<!--en:55cc8c32-->
- **Beispiele für Ausgabeformen:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ (Werte $19, 25, 37, 43$) und $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da-->
::: {.slide title="Implementierung der 2D-Kreuzkorrelation"}

<!--en:07566992-->
[Direkte Umsetzung mit verschachtelten Schleifen]{.kicker}

<!--en:202aa5d3-->
Die Funktion `corr2d` legt den Ausgabetensor der Form $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ an und berechnet jeden Eintrag als Summe der Elemente von `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba-->
Jeder Schleifendurchlauf wählt ein $k_\textrm{h} \times k_\textrm{w}$-Fenster aus `X`, multipliziert es elementweise mit `K` und summiert die Produkte mit `sum`.

<!--en:4ee9f6b1-->
::: {.slide title="Prüfung von corr2d und Äquivalenz bei gedrehtem Faltungskern"}

<!--en:2b0be423-->
[Kreuzkorrelation und mathematische Faltung im Vergleich]{.kicker}

<!--en:74b9b745-->
- **Um 180° gedrehter Faltungskern $\mathbf{K}'$:** Die Kreuzkorrelation mit $\mathbf{K}$ entspricht der Faltung mit $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd-->
- **Gleiche Klasse lernbarer Schichten:** Frameworks berechnen die Kreuzkorrelation und nennen sie Faltung.

<!--en:a69e29b5-->
::: {.slide title="Aufbau eines Moduls für eine 2D-Faltungsschicht"}

<!--en:55763603-->
[Lernbare Gewichte des Faltungskerns und skalarer Bias]{.kicker}

<!--en:5b8bad24-->
Eine Faltungsschicht kombiniert `corr2d` mit einer lernbaren $k_\textrm{h} \times k_\textrm{w}$-Gewichtsmatrix (Anfangswerte aus $\mathcal{N}(0, 0.01^2)$) und einem skalaren Bias (Anfangswert $0$):

<!--en:c16d97d6-->
::: {.slide title="Unabhängigkeit der Faltungsschicht von der Bildauflösung"}

<!--en:a7d1e48f-->
[Dieselben Parameter für beliebige Eingabegrößen]{.kicker}

<!--en:53a2bce7-->
- **Unabhängig von $H \times W$:** Dieselben 5 Parameter ($2 \times 2$-Faltungskern + 1 Bias) verarbeiten Eingaben der Größen $3 \times 3$ und $6 \times 8$ ebenso wie Megapixelbilder. Die Gewichte behalten ihre Form.

<!--en:017fec24-->
::: {.slide title="Ein synthetisches Bild mit vertikalen Kanten"}

<!--en:896e641c-->
[Ein binäres 6×8-Testmuster]{.kicker}

<!--en:0a4b456b-->
Das $6 \times 8$-Testbild `X` mit Einsen am linken und rechten Rand und Nullen in den mittleren vier Spalten erzeugen (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3-->
- **Abschnittsweise konstante Spaltenwerte:** Die Intensität bleibt konstant (`1` oder `0`) innerhalb der Spaltenbereiche $0\text{--}1$, $2\text{--}5$ und $6\text{--}7$. Sprünge treten nur zwischen den Spalten $1 \to 2$ und $5 \to 6$ auf.

<!--en:3ce31cbe-->
::: {.slide title="Anwendung des Faltungskerns für horizontale endliche Differenzen"}

<!--en:b37a1029-->
[Vertikale Kanten mit K = [1, -1] erkennen]{.kicker}

<!--en:9be85019-->
Die Kreuzkorrelation von `X` mit dem $1 \times 2$-Faltungskern $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ berechnet für jedes horizontale Pixelpaar die Differenz $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$:

<!--en:f649cfae-->
- **Kantenkarte mit Vorzeichen:** Spalte 1 hat den Wert `+1` (Weiß nach Schwarz), Spalte 5 den Wert `-1` (Schwarz nach Weiß). Innerhalb der konstanten Bereiche ergibt sich `0`.

<!--en:e6d155be-->
::: {.slide title="Richtungsabhängigkeit von Faltungskernen zur Kantenerkennung"}

<!--en:b525cde4-->
[Warum ein Faltungskern allein nicht alle Richtungen erkennt]{.kicker}

<!--en:38159de6-->
- **Unempfindlich für horizontale Kanten:** Nach dem Transponieren von `X` (`X.T`) haben horizontale Nachbarn denselben Wert. Daher liefert $[1, -1]$ nur Nullen.

<!--en:f0f30f76-->
- **Filterbänke (§{sec_channels}):** Schichten zur Bildverarbeitung lernen parallele Kanäle für horizontale, vertikale und diagonale Strukturen.

<!--en:b755e31b-->
::: {.slide title="Lernen eines Kantendetektors durch Gradientenabstieg"}

<!--en:86bbeea7-->
[Überwachtes Lernen mit einer integrierten Faltungsschicht und (X, Y)]{.kicker}

<!--en:a569780c-->
- **4D-Tensoren im Framework:** `X` und `Y` als 4D-Tensoren darstellen (ein Beispiel, ein Kanal) und $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ mit $\eta = 0.03$ minimieren

<!--en:2489ce6e-->
::: {.slide title="Die gelernten Gewichte des Faltungskerns"}

<!--en:e49eaf11-->
[Warum der Verlust bei 12 beginnt und die Gewichte gegen [1, -1] konvergieren]{.kicker}

<!--en:9cd1c645-->
- **Warum der Verlust bei $12.0$ beginnt:** Gewichte nahe null liefern die Vorhersage $\hat{Y} \approx 0$. Bei den zwölf Kantenpixeln mit den Werten $\pm 1$ in `Y` liegen sie daher daneben ($12 \times 1^2 = 12$).

<!--en:04d69b68-->
- **Konvergenz in 10 Schritten:** Der Gradientenabstieg lernt $[0.99, -0.99]$ automatisch aus den Daten.

<!--en:3ba929d2-->
::: {.slide title="Schnelle Berechnung der Faltung durch Matrizenmultiplikation"}

<!--en:1711f812-->
[Die im2col-Transformation vom Bild zu Spalten]{.kicker}

<!--en:a040f3d7-->
- **Eine einzige dichte Matrizenmultiplikation (GEMM):** Jedes $k_\textrm{h} \times k_\textrm{w}$-Fenster als Zeile von `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ schreiben und die Matrix mit $\textrm{vec}(\mathbf{K})$ multiplizieren

<!--en:1fb18ec4-->
- **Speicherbedarf:** Die Tensor Cores der GPU erreichen ihre Höchstgeschwindigkeit. Dafür steht jedes Pixel im Speicher für die Ausschnitte bis zu $k_\textrm{h} k_\textrm{w}\times$.

<!--en:0d757cf6-->
::: {.slide title="Alternative Darstellung des Faltungskerns als Bandmatrix"}

<!--en:7f0defaa-->
[Multiplikation mit einer Gewichtsmatrix in Toeplitz-Form]{.kicker}

<!--en:5898ba57-->
Statt das Eingabebild in überlappende Ausschnitte zu zerlegen, lässt sich der *Faltungskern* zu einer dünn besetzten Bandmatrix $\mathbf{W}$ entfalten. Ihre Multiplikation mit der Eingabe in Vektorform $\textrm{vec}(\mathbf{X})$ ergibt die Ausgabe:

<!--en:dae51741-->
- **Matrix-Vektor-Form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, wobei $\mathbf{W}$ genau $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ Zeilen und $n_\textrm{h} n_\textrm{w}$ Spalten hat.

<!--en:f0e86ee3-->
- **Bezug zu §{sec_why-conv} und zur Backpropagation:** Jede Zeile von $\mathbf{W}$ enthält dieselben $k_\textrm{h} k_\textrm{w}$ Einträge des Faltungskerns, jeweils um einige Spalten versetzt (Teilbild c). Der Rückwärtsdurchlauf bezüglich $\mathbf{X}$ ist daher eine Multiplikation mit $\mathbf{W}^\top$ (eine transponierte Faltung).

<!--en:af8cf19a-->
::: {.slide title="Merkmalskarten und rezeptive Felder mit zunehmender Tiefe"}

<!--en:4eb965f3-->
[Lineares Wachstum des rezeptiven Feldes über mehrere Schichten]{.kicker}

<!--en:5599a6e9-->
Das **rezeptive Feld** einer Einheit umfasst den Bereich des ursprünglichen Eingabebildes, dessen Pixel die Aktivierung dieser Einheit im Vorwärtsdurchlauf beeinflussen können:

<!--en:eba74f75-->
- **Eine Schicht ($L=1$):** Ein $k_1 \times k_1$-Faltungskern erfasst einen $k_1 \times k_1$-Ausschnitt der Eingabe.

<!--en:01c175bf-->
- **$L$ aufeinanderfolgende Schichten mit Stride 1:** Schicht $i$ mit Kerngröße $k_i$ erfasst entlang jeder Achse $k_i - 1$ Pixel mehr als ein einzelnes Element der Schicht $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822-->
- **$L$ aufeinanderfolgende $3 \times 3$-Schichten:** Jede Schicht fügt $3 - 1 = 2$ Pixel hinzu. Damit ergibt sich die Seitenlänge $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a-->
::: {.slide title="Vorteile mehrerer 3×3-Faltungskerne gegenüber einem großen Faltungskern"}

<!--en:95e46e70-->
[Weniger Parameter und höhere Ausdrucksstärke durch Nichtlinearität]{.kicker}

<!--en:bd1d7cc0-->
- **Gleiches rezeptives Feld der Größe $7 \times 7$:** Drei aufeinanderfolgende $3 \times 3$-Schichten ($r = 1 + 2 + 2 + 2 = 7$) erfassen genau denselben $7 \times 7$-Ausschnitt wie eine $7 \times 7$-Schicht.

<!--en:755ac11a-->
- **45% weniger Parameter:** $3 \times 9 c^2 = 27 c^2$ Gewichte statt $49 c^2$ bei $c$ Kanälen.

<!--en:e3aa1086-->
- **Höhere Ausdrucksstärke:** Die Schichten verknüpfen **3 nichtlineare Aktivierungen** statt nur 1 Aktivierung (das grundlegende Entwurfsprinzip von VGG in §{sec_blocks}).

<!--en:9ec83bf6-->
::: {.slide title="Theoretische und effektive rezeptive Felder"}

<!--en:72a52c3b-->
[Gaußförmige Abnahme der Empfindlichkeit vom Zentrum zum Rand]{.kicker}

<!--en:33bb377d-->
- **Theoretisches rezeptives Feld ($r = 2L+1$):** Begrenzungsrahmen aller Pixel mit *mindestens einem* Pfad zur Ausgabeeinheit.

<!--en:293436eb-->
- **Zentraler Grenzwertsatz:** Von den mittleren Pixeln führen exponentiell viele Pfade zur Ausgabeeinheit, von jedem Eckpixel nur **1 Pfad**.

<!--en:f47d181e-->
- **Effektives rezeptives Feld (Luo et al., 2016):** Der Betrag des Eingabegradienten $\partial y / \partial \mathbf{X}$ fällt gemäß einer gaußschen Glockenkurve mit Standardabweichung $\sigma \propto \sqrt{L}$ ab.

<!--en:fac792d0-->
::: {.slide title="Verständnisfragen zu Kreuzkorrelation und rezeptiven Feldern"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:8fe554e5-->
Fragen zu Ausgabeformen, `im2col` und zur Berechnung rezeptiver Felder:

<!--en:281c7c78-->
1. **Matrixdimensionen bei im2col:** Ein einkanaliger $3 \times 3$-Faltungskern ohne Padding wirkt auf ein $10 \times 10$-Bild. Die Berechnung nutzt `im2col`. Welche räumliche Form hat die Ausgabe? Welche Dimensionen haben die Matrix `patches` und der Faltungskern in Vektorform genau?

<!--en:16976d18-->
2. **Erforderliche Tiefe für Fashion-MNIST:** Wie viele aufeinanderfolgende $3 \times 3$-Schichten mit Stride 1 sind nötig, damit das theoretische rezeptive Feld einer Einheit ein ganzes $28 \times 28$-Bild abdeckt? Wie viele $5 \times 5$-Schichten wären nötig?

<!--en:ab2e0d6a-->
3. **Verlust bei verschobener Kante:** Warum lag der Verlust in unserem Lernexperiment mit vertikalen Kanten im $6 \times 8$-Bild anfangs bei $\approx 12$? Wie groß wäre der anfängliche Verlust, wenn der schwarze Streifen bis zum rechten Rand reichte (`X[:, 2:8] = 0`)?

<!--en:250d6877-->
**Lösungen:**
(1) Die Ausgabe hat die Form $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ Ausgabewerte). Die Matrix `patches` hat die Form $(64, 9)$. Der rechte Faktor ist der Faltungskern in Form eines $(9, 1)$-Vektors.
(2) Für $3 \times 3$ gilt $2L + 1 \ge 28 \implies L = 14$ Schichten. Für $5 \times 5$ gilt $4L + 1 \ge 28 \implies L = 7$ Schichten.
(3) In jeder der 6 Zeilen lagen zwei Kanten mit den Werten $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Reicht der schwarze Streifen bis zum rechten Rand, entfällt die rechte Kante von Schwarz zu Weiß. Es bleibt eine Kante pro Zeile. Der anfängliche Verlust halbiert sich auf $\approx 6$.

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und Kernaussagen"}

<!--en:ce12e85f-->
[Überblick über den Abschnitt]{.kicker}

<!--en:ebc1b04f-->
- **2D-Kreuzkorrelation und Lernen eines Kantendetektors:** $\mathbf{K}$ durchläuft alle gültigen Fenster ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). Der Gradientenabstieg lernt $[1, -1]$ in 10 Schritten.

<!--en:09efe4f0-->
- **Matrizenmultiplikation mit im2col:** Die bis zu $k_\textrm{h} k_\textrm{w}\times$ wiederholte Speicherung der Pixel in den Ausschnitten ermöglicht maximalen GEMM-Durchsatz auf der GPU.

<!--en:3af8acd0-->
- **Rezeptive Felder bei aufeinanderfolgenden Schichten:** Die Seitenlänge beträgt $r = 1 + \sum (k_i - 1) = 2L+1$ bei $3 \times 3$-Schichten. Die Empfindlichkeit ist gaußförmig und im Zentrum am größten.
