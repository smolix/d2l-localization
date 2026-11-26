<!--en:de6402d5-->
# Von vollständig verbundenen Schichten zu Faltungen

<!--en:eef99ec7-->
Die mehrschichtigen Perzeptrons (engl. multilayer perceptrons, MLP) aus :numref:`chap_perceptrons` formen jedes Bild zu einem Zahlenvektor ohne räumliche Struktur um. Für tabellarische Daten ist das sinnvoll. Bei Bildern geht dabei jedoch die Information verloren, welche Pixel nebeneinanderliegen. Für ein Graustufenfoto mit einem Megapixel ($1000 \times 1000$) benötigt schon eine versteckte Schicht mit nur 1000 Einheiten $10^6 \times 10^3 = 10^9$ Gewichte (4 GB in fp32). Für eine versteckte Karte in voller Auflösung von $1000 \times 1000$ sind es sogar $10^{12}$ Gewichte. Natürliche Bilder unterscheiden sich jedoch grundlegend von beliebigen Vektoren mit einer Million Dimensionen. Benachbarte Pixel sind stark korreliert (*Lokalität*), und ein visuelles Merkmal (engl. feature) wie ein Schnurrhaar hat überall im Bild dieselbe Bedeutung (*Translationsäquivarianz*).

<!--en:9ac408b5-->
Was geschieht, wenn wir diese beiden geometrischen Prinzipien als Nebenbedingungen für eine vollständig verbundene Schicht (engl. fully connected layer) vorgeben, die ein zweidimensionales Pixelgitter auf ein zweidimensionales Gitter versteckter Einheiten abbildet? Allein durch Translationsäquivarianz sinkt die Zahl der Gewichte von $10^{12}$ auf rund $4 \times 10^6$. Beschränken wir jede versteckte Einheit auf eine kleine lokale Umgebung, bleiben nur wenige Hundert Gewichte. Eine Schicht, die beide Nebenbedingungen erfüllt, ist eine *Faltungsschicht* (engl. convolutional layer).

<!--en:11b632d3-->
## Äquivarianz und Invarianz

<!--en:12014b14-->
Das Kindersuchspiel „Wo ist Walter?“ (:numref:`img_waldo`) veranschaulicht eine zentrale Schwierigkeit der visuellen Wahrnehmung. In jeder Szene wimmelt es von Ablenkungen, und Walter versteckt sich an einer unerwarteten Stelle. Doch *wie Walter aussieht*, hängt nicht davon ab, *wo er steht*. Wir können also mit einem einzigen Walter-Detektor das ganze Bild absuchen. An jeder Position bewerten wir nach derselben Regel, wie gut der jeweilige Bildausschnitt zu Walter passt. Moderne Verfahren zur Objekterkennung :cite:`Girshick.Donahue.Darrell.ea.2014` und zur semantischen Segmentierung :cite:`Long.Shelhamer.Darrell.2015` bauen direkt auf dieser Idee auf.

<!--en:9adfb91b-->
![Wo ist Walter (Bild mit freundlicher Genehmigung von William Murphy (Infomatique))?](../img/waldo-football.jpg)

<!--en:78a61406-->
Daraus ergeben sich zwei Leitprinzipien für die ersten Schichten eines Netzes zur Bildverarbeitung:

<!--en:0d9b406b-->
1. Auf denselben Bildausschnitt sollen sie überall gleich reagieren. Verschiebt sich die Eingabe, so verschiebt sich die Ausgabe um denselben Vektor.

<!--en:57981b65-->
1. Sie sollen nur kleine lokale Umgebungen berücksichtigen und entfernte Bildbereiche außer Acht lassen. Das ist das Prinzip der *Lokalität*.

<!--en:cff4ba40-->
Nachfolgende Schichten können diese lokalen Antworten zu immer größeren, abstrakteren Mustern zusammensetzen. Daraus ergibt sich schließlich eine Vorhersage für das ganze Bild.

<!--en:32911ddc-->
Um das erste Prinzip zu formalisieren, bezeichnen wir mit $T_v$ den Verschiebungsoperator, der ein Bild um den Vektor $v = (v_1, v_2)$ verschiebt. Das Pixel $(i, j)$ gelangt an die Position $(i + v_1, j + v_2)$, sodass $[T_v \mathbf{X}]_{i,j} = [\mathbf{X}]_{i-v_1, j-v_2}$ gilt. Eine Abbildung $f$ von Bildern auf Bilder heißt *translationsäquivariant*, wenn sich bei einer Verschiebung der Eingabe die Ausgabe um genau denselben Vektor verschiebt. Eine Abbildung von Bildern auf Labels heißt *translationsinvariant*, wenn ihre Ausgabe bei einer Verschiebung der Eingabe unverändert bleibt:

<!--en:a03a79ee-->
:numref:`fig_why_conv_equivariance` veranschaulicht die Translationsäquivarianz in einem kommutativen Diagramm. Hier vergleicht $f$ eine kleine Schablone mit jedem Fenster des Bildes. Genau diese Operation leiten wir gleich her. Beide Reihenfolgen ergeben dieselbe Karte: erst verschieben und dann vergleichen oder erst vergleichen und dann verschieben.

<!--en:4bcbf106-->
![Translationsäquivarianz. Die Eingabe enthält einen $2\times 2$-Ausschnitt aus Einsen. Die Funktion $f$ vergleicht jedes $2\times 2$-Fenster mit diesem Ausschnitt. Die Antwort beträgt 4, wo das Fenster den gesamten Ausschnitt abdeckt. Eine Verschiebung der Eingabe um $v = (3, 2)$ mit anschließender Anwendung von $f$ (untere Zeile) ergibt dieselbe Karte wie die Anwendung von $f$ mit anschließender Verschiebung der Antwort (rechte Spalte).](../img/why-conv-equivariance.svg)

<!--en:62983fa5-->
Die beiden Eigenschaften erfüllen unterschiedliche Aufgaben. *Äquivarianz* erhält die Information darüber, *wo* ein Muster auftritt. Bei *Invarianz* spielt der Ort keine Rolle mehr. Ein Walter-Detektor sollte äquivariant sein, damit seine Score-Karte Walters Position in der Szene verfolgt. Die Antwort auf die Ja-Nein-Frage „Ist Walter auf diesem Bild?“ sollte dagegen invariant sein. Das erreichen wir, indem wir am Ende die Werte der äquivarianten Karte über alle Positionen zusammenfassen, etwa indem wir ihr Maximum nehmen.

<!--en:173ce326-->
## Nebenbedingungen für die vollständig verbundene Schicht

<!--en:687768a8-->
In einer Dimension können wir die gesamte Gewichtsmatrix zeichnen. So lassen sich beide Nebenbedingungen besonders gut erkennen. Betrachten wir ein Signal aus acht Abtastwerten $x_0, \ldots, x_7$. Eine vollständig verbundene Schicht bildet es auf acht versteckte Einheiten ab, gemäß $h_i = u_i + \sum_k W_{i,k} x_k$. Ihre $8 \times 8$-Gewichtsmatrix hat 64 unabhängige Einträge, wie Teil (a) von :numref:`fig_why_conv_weight_sharing` zeigt.

<!--en:7c15e517-->
![Gewichte einer linearen Schicht, die ein eindimensionales Signal mit 8 Abtastwerten (Spalten) auf 8 versteckte Einheiten (Zeilen) abbildet. Gleiche Farben stehen für gleiche Gewichte, Weiß steht für null. (a) Vollständig verbunden: Jedes Gewicht ist frei wählbar. (b) Translationsäquivariant: Ein Gewicht hängt nur vom Versatz zwischen Eingabe und Ausgabe ab. Entlang jeder Diagonalen ist die Matrix daher konstant. (c) Äquivariant und lokal: Nur für die Versätze $-1$, $0$ und $1$ sind die Gewichte von null verschieden.](../img/why-conv-weight-sharing.svg)

<!--en:50da4510-->
Was folgt aus der Translationsäquivarianz für $\mathbf{W}$? Lassen wir die Bias-Terme zunächst außer Acht (unten zeigen wir, dass sie alle übereinstimmen müssen). Geben wir einen Einheitsimpuls mit einer einzigen 1 an der Stelle $k$ ein, so gibt die Schicht die Spalte $k$ von $\mathbf{W}$ aus. Verschieben wir den Impuls nach $k + 1$, so gibt die Schicht die Spalte $k + 1$ aus. Äquivarianz verlangt, dass die Spalte $k + 1$ mit der um eine Zeile nach unten verschobenen Spalte $k$ übereinstimmt. Es gilt also $W_{i+1, k+1} = W_{i, k}$. Folglich muss $\mathbf{W}$ entlang jeder Diagonalen konstant sein wie in Teil (b). Eine solche Matrix heißt *Toeplitz*-Matrix. Jede Diagonale entspricht einem relativen Versatz $a = k - i$. Von 64 freien Gewichten bleiben damit nur 15 (für die Versätze von $-7$ bis $7$).

<!--en:d1306670-->
Durch Lokalität fallen fast alle der verbleibenden 15 Gewichte weg. Beschränken wir jede versteckte Einheit auf Nachbarn im Abstand von höchstens 1, so setzen wir die Gewichte für alle Versätze außerhalb von $\{-1, 0, 1\}$ auf null. Es bleiben nur 3 gemeinsam genutzte Gewichte, die sich in jeder Zeile von Teil (c) wiederfinden. Die Schicht verschiebt nun eine kleine Schablone mit 3 Gewichten über das gesamte Signal. Damit haben wir die ganze Herleitung im Kleinen schon gesehen. Bei zweidimensionalen Bildern kommt nur ein zweiter räumlicher Index hinzu.

<!--en:c40f013b-->
Bei Bildern sind sowohl die Eingabe $\mathbf{X}$ als auch die versteckte Darstellung $\mathbf{H}$ zweidimensionale Gitter. Das Eingabepixel $[\mathbf{X}]_{k,l}$ liegt an der Position $(k, l)$, die versteckte Einheit $[\mathbf{H}]_{i,j}$ an der Position $(i, j)$. Verbinden wir jede versteckte Einheit mit jedem Pixel, so bilden die Gewichte einen Tensor vierter Ordnung $\mathsf{W}$. Hinzu kommt eine Bias-Matrix $\mathbf{U}$:

<!--en:509475b9-->
Für beide Nebenbedingungen ist entscheidend, wo ein Pixel *relativ* zur versteckten Einheit $(i, j)$ liegt. Daher führen wir die Versätze $a = k - i$ und $b = l - j$ als neue Indizes ein und definieren $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$:

<!--en:82c1d42a-->
Dabei nehmen $a$ und $b$ sowohl positive als auch negative Werte an. Bisher hat sich nur die Schreibweise geändert. Wir lesen $\mathsf{W}$ entlang der Diagonalen ab, die Parameteranzahl bleibt gleich. Für ein Bild der Größe $1000 \times 1000$ und ein verstecktes Gitter der Größe $1000 \times 1000$ benötigt diese eine Schicht weiterhin $10^6 \times 10^6 = 10^{12}$ Gewichte.

<!--en:c60a638e-->
### Translationsäquivarianz

<!--en:4b50c132-->
Wie in einer Dimension ist eine Schicht der Form :eqref:`eq_mlp_fourth_order` genau dann translationsäquivariant, wenn ihre Gewichte und ihr Bias unabhängig von der Position $(i, j)$ sind. Dass gemeinsam genutzte Gewichte Äquivarianz ergeben, lässt sich in einer Zeile zeigen. Erst die Umkehrung erklärt, warum wir zwangsläufig auf die Faltung (engl. convolution) kommen: Unter allen linearen Abbildungen erfüllt nur sie die Äquivarianzbedingung.

<!--en:4758f4ad-->
Verschieben wir ein endliches Bild, können Pixel über den Rand hinauswandern. Daher betten wir das Bild in ein unendliches Gitter ein und füllen es außerhalb seines Trägers mit Nullen auf. So verlässt keine Information das Gitter. Bei endlichen Bildern gilt die Äquivarianz streng genommen nur abseits der Ränder. Darauf gehen wir in :numref:`sec_padding` ein.

<!--en:227468d0-->
**Satz.** Die Eingabe habe nur endlich viele von null verschiedene Pixel. Sowohl die Eingabe als auch die versteckte Darstellung seien auf dem unendlichen Gitter definiert. Die Schicht :eqref:`eq_mlp_fourth_order` ist genau dann für jede Verschiebung $v$ translationsäquivariant, wenn es einen Skalar $u$ und eine Matrix $\mathbf{K}$ gibt, sodass $[\mathbf{U}]_{i,j} = u$ und $[\mathsf{V}]_{i,j,a,b} = [\mathbf{K}]_{a,b}$ für alle $i, j, a, b$ gelten.

<!--en:80c83d83-->
**Beweis.** Hängen weder die Gewichte noch der Bias von $(i, j)$ ab, so verschiebt eine Verschiebung der Eingabe um $v$ auch die Ausgabe um $v$:

<!--en:df55eda6-->
Nehmen wir umgekehrt an, die Schicht sei äquivariant. Das Nullbild ist unter jeder Verschiebung invariant. Die Schicht bildet es auf $\mathbf{U}$ ab. Daher gilt $\mathbf{U} = T_v \mathbf{U}$ für alle $v$, der Bias muss also eine Konstante $u$ sein. Ziehen wir $u$ auf beiden Seiten ab, sehen wir, dass auch der lineare Anteil für sich genommen äquivariant ist. Geben wir ein Bild mit einer einzigen 1 an der Position $(k, l)$ und Nullen an allen anderen Positionen ein. Die Antwort an der Position $(i, j)$ lautet $[\mathsf{W}]_{i,j,k,l}$. Eine Verschiebung um $v$ bringt den Impuls an die Position $(k + v_1, l + v_2)$. Aus der Äquivarianz folgt $[\mathsf{W}]_{i, j, k+v_1, l+v_2} = [\mathsf{W}]_{i-v_1, j-v_2, k, l}$. Setzen wir $(k, l) = (a, b)$ und $v = (i, j)$, so ergibt sich $[\mathsf{V}]_{i,j,a,b} = [\mathsf{W}]_{i,j,i+a,j+b} = [\mathsf{W}]_{0,0,a,b}$. Dieser Wert ist unabhängig von $(i, j)$. $\blacksquare$

<!--en:159b4f9b-->
Ein einziges helles Pixel als Testeingabe zeigt also, dass die Gewichte an allen Positionen exakt übereinstimmen müssen. Damit vereinfacht sich die Schicht zu

<!--en:f4f8e981-->
Jede versteckte Einheit wendet nun dasselbe Muster $\mathbf{K}$ an, mit ihrem eigenen Gitterpunkt als Mittelpunkt. Bei einem mit Nullen aufgefüllten Bild der Größe $1000 \times 1000$ können nur Versätze $a, b \in [-999, 999]$ ein von null verschiedenes Pixel erreichen. So sinkt die Zahl der Gewichte von $10^{12}$ auf $1999^2 \approx 4 \times 10^6$.

<!--en:5d2de8a7-->
### Lokalität

<!--en:c55c08b0-->
Vier Millionen Gewichte sind für einen einzigen Filter immer noch viel zu viele. Die meisten davon verbinden eine versteckte Einheit mit weit entfernten Pixeln. Fordern wir *Lokalität*, so darf $[\mathbf{H}]_{i,j}$ nur von Pixeln abhängen, deren Abstand entlang jeder Achse höchstens $\Delta$ beträgt. Wir setzen also $[\mathbf{K}]_{a, b} = 0$, sobald $|a| > \Delta$ oder $|b| > \Delta$ gilt. Damit ergibt sich

<!--en:76bc2a3e-->
Jede versteckte Einheit berechnet nun eine gewichtete Summe über ein lokales Fenster der Größe $(2\Delta+1) \times (2\Delta+1)$ mit Mittelpunkt $(i, j)$. Dafür genügen $(2\Delta+1)^2$ gemeinsam genutzte Gewichte und ein Bias. Das sind 9 Gewichte für $\Delta = 1$ (ein $3 \times 3$-Fenster) oder 121 für $\Delta = 5$. Da $\Delta$ meist deutlich unter 10 liegt, treten wenige Dutzend oder Hundert Gewichte an die Stelle von $4 \times 10^6$. Die volle räumliche Auflösung der versteckten Karte bleibt erhalten.

<!--en:cc0521a8-->
Gleichung :eqref:`eq_conv-layer` definiert eine *Faltungsschicht*, den Baustein *konvolutionärer neuronaler Netze* (engl. convolutional neural network, CNN). Die gemeinsam genutzte Matrix $\mathbf{K}$ heißt *Faltungskern* (engl. convolution kernel) oder *Filter*. Zu den frühen Beispielen räumlicher und zeitlicher Gewichtsteilung (engl. weight sharing) gehören neuronale Netze mit Zeitverzögerung zur Spracherkennung :cite:`Waibel.Hanazawa.Hinton.ea.1989` und die verschiebungsinvarianten Netze von :citet:`Zhang.ea.1988`.

<!--en:1eac75f5-->
Die Parameteranzahl ist von $10^{12}$ auf $(2\Delta + 1)^2$ gesunken, weil wir die Hypothesenklasse auf lokale, translationsäquivariante Abbildungen beschränkt haben. Dieser eingebaute *induktive Bias* (:numref:`sec_generalization_deep`) verbessert die Dateneffizienz (engl. sample efficiency) erheblich, wenn Merkmale wie Kanten und Texturen lokal und positionsunabhängig sind. Eine einzelne Schicht kann allerdings identische Bildausschnitte nicht allein anhand ihrer absoluten Position unterscheiden (etwa ein Auge im oberen Teil eines Passfotos von einem Nasenloch weiter unten). Stapeln wir mehrere Schichten mit Nichtlinearitäten, können tiefer liegende Einheiten Informationen aus immer größeren Bereichen zusammenführen.

<!--en:e5caaa8c-->
## Faltung und Kreuzkorrelation

<!--en:2685cbd7-->
Streng genommen ist die Bezeichnung „Faltung“ im Deep Learning etwas ungenau. In der Mathematik ist die diskrete *Faltung* zweier eindimensionaler Funktionen durch $(f * g)(i) = \sum_a f(a) g(i-a)$ gegeben (auf $\mathbb{R}^d$ mit einem Integral). Sie misst die Überlappung von $f$ mit einer *gespiegelten* und verschobenen Kopie von $g$. In zwei Dimensionen gilt

<!--en:6d41e24f-->
Vergleichen wir dies mit :eqref:`eq_conv-layer`. Dort greifen wir auf $\mathbf{X}$ an der Position $(i + a, j + b)$ statt an der Position $(i - a, j - b)$ zu. Setzen wir $-a$ für $a$ und $-b$ für $b$ ein, so erkennen wir: :eqref:`eq_conv-layer` beschreibt die mathematische Faltung von $\mathbf{X}$ mit dem gespiegelten Faltungskern $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$:

<!--en:dc12f185-->
Die Spiegelung des Faltungskerns kehrt die Reihenfolge seiner Gewichte entlang beider Achsen um. Das Gewicht oben links liegt danach unten rechts (:numref:`fig_conv_flipped_kernel`).

<!--en:80c9ec72-->
![Ein $3 \times 3$-Faltungskern $\mathbf{K}$ und seine gespiegelte Kopie $\mathbf{K}'$. Die Versätze $a$ und $b$ indizieren Zeilen und Spalten. Die Spiegelung kehrt beide Achsen um: Die beiden schattierten Gewichte wandern von den Positionen $(-1, -1)$ und $(-1, 0)$ nach $(1, 1)$ und $(1, 0)$. Die Kreuzkorrelation mit $\mathbf{K}$ entspricht der Faltung mit $\mathbf{K}'$.](../img/conv-flipped-kernel.svg)

<!--en:5ee1bcc7-->
Die Operation ohne Spiegelung heißt genau genommen *Kreuzkorrelation* (engl. cross-correlation). Da CNNs ihre Faltungskerne aus Daten lernen, hat der Unterschied keinen Einfluss auf die Ausdrucksstärke. Eine Schicht, die eine Kreuzkorrelation mit $\mathbf{K}$ berechnet, kann genau dieselbe Familie von Funktionen lernen wie eine Schicht, die eine Faltung mit $\mathbf{K}'$ berechnet. Deep-Learning-Frameworks berechnen die Kreuzkorrelation und nennen sie Faltung. Dieser üblichen Konvention folgen wir im gesamten Buch.

<!--en:a2c36647-->
## Kanäle

<!--en:2da7a27a-->
Bisher haben wir sowohl die Eingaben als auch die versteckten Karten als einzelne Grauwertgitter behandelt. Farbfotos haben pro Pixel drei Intensitätswerte (Rot, Grün und Blau). Ein Bild ist also ein Tensor dritter Ordnung $\mathsf{X}$ mit der Form $3 \times \textrm{height} \times \textrm{width}$ und den Einträgen $[\mathsf{X}]_{c,i,j}$. Die erste Achse gibt den *Kanal* (engl. channel) an, sodass zu jeder Pixelposition ein Merkmalsvektor gehört. Ein Faltungskern summiert sowohl über das lokale räumliche Fenster als auch über alle Eingabekanäle.

<!--en:054e1f79-->
Ebenso wichtig ist, dass ein einzelner Faltungskern nur eine Antwortkarte erzeugt (wie die Score-Karte für Walter in :numref:`fig_waldo_mask`). Er erfasst also nur ein visuelles Muster. Für eine aussagekräftige Bilddarstellung müssen wir viele Muster parallel erkennen, etwa Kanten verschiedener Ausrichtung, Ecken, Farbkontraste und Texturen. Daher wird auch die versteckte Darstellung $\mathsf{H}$ zu einem Tensor dritter Ordnung: Die Karten liegen entlang der Achse für den *Ausgabekanal* (engl. output channel) übereinander. Jeder Ausgabekanal $o$ hat einen eigenen gelernten Slice des Faltungskerns und einen eigenen Bias $u_o$. Damit wird $\mathsf{K}$ zu einem Tensor vierter Ordnung:

<!--en:65beb6c1-->
Hier bezeichnet $c$ den Index des Eingabekanals und $o$ den des Ausgabekanals. Im gesamten Kapitel schreiben wir den Ausgabekanal vor dem Eingabekanal. Jeder Ausgabekanal zeigt, wo auf dem Gitter ein bestimmtes gelerntes Merkmal auftritt. Daher heißen versteckte Kanäle auch *Merkmalskarten* (engl. feature maps). Der gesamte Stapel bildet die Eingabe für die nächste Schicht. Bei $c_\textrm{i}$ Eingabekanälen und $c_\textrm{o}$ Ausgabekanälen benötigt die Schicht $(2\Delta+1)^2 c_\textrm{i} c_\textrm{o}$ Gewichte und $c_\textrm{o}$ Bias-Terme. Eine $3 \times 3$-Schicht, die ein RGB-Bild auf 64 Merkmalskarten abbildet, hat also nur $9 \cdot 3 \cdot 64 + 64 = 1{,}792$ Parameter, unabhängig von der Bildauflösung.

<!--en:be185e73-->
![Ein Walter-Detektor: Die Antwort eines auf Walter abgestimmten Faltungskerns erreicht an Walters Position ihr Maximum (Bild mit freundlicher Genehmigung von William Murphy (Infomatique)).](../img/waldo-mask.jpg)

<!--en:daa61dce-->
Die Herleitung gilt in beliebig vielen Dimensionen. Dieselbe Überlegung ergibt eindimensionale Faltungen über Audioabtastwerte oder Text-Token sowie dreidimensionale Faltungen über die Einzelbilder eines Videos oder die Volumendaten medizinischer Aufnahmen.

<!--en:904f7cf3-->
Gleichung :eqref:`eq_conv-layer-channels` bildet die Grundlage für den Rest dieses Kapitels. Da wir sie auf einem idealisierten unendlichen Gitter hergeleitet haben, bleiben drei praktische Fragen offen. Wie behandeln wir die Ränder endlicher Bilder? Wie vergrößern wir den Bildbereich, den tiefer liegende Einheiten erfassen? Und wie fassen wir räumliche Merkmalskarten zu einer Entscheidung für das ganze Bild zusammen, etwa darüber, ob Walter *irgendwo* in der Szene zu sehen ist? Zunächst setzen wir jedoch :eqref:`eq_conv-layer` in Code um.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:e36b09e5-->
Formen wir ein Bild zu einem unstrukturierten Vektor um, werden vollständig verbundene Schichten unpraktikabel groß. Um ein Megapixelbild auf ein gleich großes verstecktes Gitter abzubilden, brauchen sie $10^{12}$ Gewichte. Zwei naheliegende Annahmen über Bilder verringern diese Zahl auf wenige Hundert. *Translationsäquivarianz* erzwingt an allen Positionen denselben Faltungskern, wie wir auf dem unendlichen Gitter bewiesen haben. *Lokalität* beschränkt diesen Kern auf ein kleines Fenster. Die resultierende Schicht berechnet eine Kreuzkorrelation, die im Deep Learning allgemein Faltung heißt. Mit mehreren gestapelten Eingabe- und Ausgabekanälen kann eine einzige Schicht viele visuelle Muster gleichzeitig erkennen. Dabei bleibt die Parameteranzahl unabhängig von der Bildgröße. Diese Vorteile folgen direkt aus dem induktiven Bias: Eine Faltungsschicht verzichtet auf beliebige positionsabhängige Abbildungen und generalisiert dadurch gut auf räumlich strukturierten Daten.

<!--en:2294243a-->
## Übungen

<!--en:994fc53e-->
1. **Gewichtsteilung auf einem Ring.** Übertragen Sie die Überlegung aus :numref:`fig_why_conv_weight_sharing` auf ein Signal mit vier Abtastwerten auf einem periodischen Gitter. Auf den letzten Abtastwert folgt wieder der erste, sodass bei einer Verschiebung kein Wert verloren geht.
    1. Geben Sie der Schicht nacheinander an jeder Abtaststelle eine einzelne 1 als Eingabe. Schreiben Sie die $4 \times 4$-Gewichtsmatrix einer translationsäquivarianten Schicht auf. Wie viele freie Gewichte hat sie? Vergleichen Sie das Ergebnis mit den $2 \cdot 4 - 1 = 7$ Versätzen bei vier von Nullen umgebenen Abtastwerten.
    1. Fordern Sie Lokalität mit den Versätzen $-1$, $0$ und $1$. Welche Matrixeinträge sind von null verschieden, und welche davon haben keine Entsprechung in Teil (c) der Abbildung?

<!--en:21deb4f8-->
2. **Kommutativität der Faltung.**
    1. Beweisen Sie, dass die Faltung aus :eqref:`eq_2d-conv-discrete` kommutativ ist, dass also $f * g = g * f$ gilt.
    1. Definieren Sie die Kreuzkorrelation $(f \star g)(i, j) = \sum_a \sum_b f(i+a, j+b)\, g(a, b)$. Das ist die Operation aus :eqref:`eq_conv-layer` mit der Eingabe $f$ und dem Faltungskern $g$. Drücken Sie $(g \star f)(i, j)$ durch Werte von $f \star g$ aus und entscheiden Sie anhand dieses Ausdrucks, ob die Kreuzkorrelation kommutativ ist.

<!--en:7773c061-->
3. **Faltungen für Audiodaten.** Audiodaten liegen häufig als eindimensionale Sequenz von Abtastwerten vor.
    1. Leiten Sie das eindimensionale Gegenstück zu :eqref:`eq_conv-layer` her.
    1. Nennen Sie eine Aufgabe der Audioverarbeitung, für die sich eine zeitlich lokale und translationsäquivariante Schicht gut eignet, und eine, für die sie sich nicht eignet. Begründen Sie jeweils Ihre Wahl.
    1. Ein Spektrogramm stellt eine Aufnahme als zweidimensionales Feld mit den Achsen Zeit und Frequenz dar. Die Frequenzachse ist linear in Hertz skaliert. Geben Sie für jede Achse an, ob Translationsäquivarianz eine sinnvolle Annahme ist. Begründen Sie Ihre Antwort damit, wie sich eine Verschiebung entlang der jeweiligen Achse auf den Klang auswirkt. Beantworten Sie dieselbe Frage anschließend für eine logarithmisch skalierte Frequenzachse.

<!--en:68319090-->
4. **Äquivarianz und Invarianz.** Ein Klassifikator berechnet Merkmalskarten $h(\mathsf{X})$ mit mehreren aufeinanderfolgenden Faltungsschichten der Form :eqref:`eq_conv-layer-channels` und dazwischenliegenden Nichtlinearitäten. Anschließend berechnet er die Vorhersage $g(h(\mathsf{X}))$ mit einem Ausgabekopf $g$.
    1. Nehmen Sie ein periodisches Gitter an, auf dem nach der letzten Spalte wieder die erste folgt. Nach der Begründung des Satzes ist $h$ auf diesem Gitter translationsäquivariant. Der Ausgabekopf $g$ berechne für jeden Kanal von $h(\mathsf{X})$ den Mittelwert über alle Positionen. Zeigen Sie mithilfe von :eqref:`eq_equivariance_invariance`, dass die Vorhersage translationsinvariant ist.
    1. Ein Bild zeigt eine einzelne Ziffer in einer der neun Zellen eines $3 \times 3$-Gitters. Die Aufgabe besteht darin, die Zelle zu benennen. Erklären Sie, warum der Klassifikator aus dem ersten Teil diese Aufgabe nicht lösen kann. Beschreiben Sie einen Ausgabekopf $g$, der sie anhand derselben Merkmalskarten $h(\mathsf{X})$ löst.

<!--en:a755c6d5-->
5. **Rotationsäquivarianz.** ● Im Satz dieses Abschnitts haben wir die Gewichtsteilung aus der Äquivarianz gegenüber Verschiebungen hergeleitet. Untersuchen Sie nun Drehungen unter demselben Gesichtspunkt. Auf dem unendlichen Gitter des Satzes drehe $R$ ein Bild um $90^\circ$ um den Ursprung, sodass $[R\mathbf{X}]_{i,j} = [\mathbf{X}]_{-j,i}$ gilt.
    1. Zeigen Sie, dass die Faltungsschicht :eqref:`eq_conv-layer` die Bedingung $f(R\mathbf{X}) = R f(\mathbf{X})$ für jedes Bild $\mathbf{X}$ genau dann erfüllt, wenn $[\mathbf{K}]_{a,b} = [\mathbf{K}]_{-b,a}$ für alle Versätze $(a, b)$ gilt.
    1. Wie viele freie Gewichte hat ein solcher Faltungskern für $\Delta = 1$? Zeigen Sie, dass sich seine Antwort auf ein Bild nicht ändert, wenn Sie das Bild um $90^\circ$ um das gerade ausgewertete Pixel drehen. Er kann daher eine vertikale Kante nicht von einer horizontalen unterscheiden.
    1. Verwenden Sie anstelle eines eingeschränkten Faltungskerns vier Ausgabekanäle. Ihre Faltungskerne seien Kopien eines frei wählbaren Faltungskerns $\mathbf{K}$, gedreht um $0^\circ$, $90^\circ$, $180^\circ$ und $270^\circ$. Zeigen Sie, dass eine Drehung der Eingabe jede Ausgabekarte dreht und die vier Kanäle zyklisch vertauscht. Welche Information erhält diese Schicht, die der eingeschränkte Faltungskern aus dem ersten Teil verliert?

<!--en:ad409fba-->
[Diskussionen](https://d2l.discourse.group/t/64)

<!--en:23eb38f8-->
Von vollständig verbundenen Schichten zu Faltungen<br>**Translationsäquivarianz · Lokalität · Gewichtsteilung · Kreuzkorrelation · Kanäle**

<!--en:3b89280e-->
::: {.slide title="Warum vollständig verbundene Schichten an hochauflösenden Bildern scheitern"}

<!--en:c1155a99-->
[Grenzen der Skalierbarkeit]{.kicker}

<!--en:3a97f384-->
- **Quadratisches Wachstum bei vollständiger Verbindung:** Die Abbildung eines $1000 \times 1000$-Bildes auf ein gleich großes verstecktes Gitter benötigt $10^{12}$ Gewichte (**4 TB** für Graustufen, **36 TB** für RGB).

<!--en:87917efb-->
- **Selbst schmale Projektionen sind zu groß:** Für die Projektion eines RGB-Bildes (1 MP) auf nur 300 versteckte Einheiten sind $9 \times 10^8$ Gewichte nötig (**3.6 GB**).

<!--en:a642b4ad-->
- **Ursache:** Vollständig verbundene Schichten haben ein eigenes Gewicht für jedes Paar von Eingabe- und Ausgabepositionen. Räumliche Nähe im zweidimensionalen Bild bleibt unberücksichtigt.

<!--en:aae17c5f-->
::: {.slide title="Zwei räumliche Symmetrien der visuellen Wahrnehmung"}

<!--en:50ccdc3d-->
[Das Beispiel „Wo ist Walter?“]{.kicker}

<!--en:5183f8a0-->
Die Suche nach einem Muster in einem Wimmelbild veranschaulicht zwei geometrische Prinzipien für die ersten Schichten der Bildverarbeitung:

<!--en:51cf2a23-->
- **Positionsunabhängiges Aussehen:** Walter sieht überall gleich aus. Eine einzige lokale Schablone genügt, um jeden Bildausschnitt abzusuchen.

<!--en:fa9adf73-->
- **Translationsäquivarianz:** Eine Verschiebung des Eingabebildes verschiebt die Antwortkarte des Detektors um genau denselben Vektor.

<!--en:33378b1f-->
- **Lokalität:** Frühe Merkmalsdetektoren betrachten nur eine kleine Umgebung jedes Pixels. Tiefere Schichten erfassen den globalen Kontext.

<!--en:cc3cedc3-->
::: {.slide title="Translationsäquivarianz und Translationsinvarianz"}

<!--en:00a942ee-->
[Räumliche Symmetrien in Formeln]{.kicker}

<!--en:6794dc13-->
- **Verschiebungsoperator $T_v$:** Er verschiebt ein Bild um $v = (v_1, v_2)$, sodass $[T_v \mathbf{X}]_{i,j} = [\mathbf{X}]_{i-v_1, j-v_2}$ gilt.

<!--en:05db6145-->
- **Äquivarianz ($f(T_v \mathbf{X}) = T_v f(\mathbf{X})$):** Merkmalskarten verschieben sich mit der Eingabe. So bleibt für die Lokalisierung erhalten, *wo* Muster auftreten.

<!--en:20bc109d-->
- **Invarianz ($f(T_v \mathbf{X}) = f(\mathbf{X})$):** Klassifikatoren für ganze Bilder fassen äquivariante Karten über alle Positionen zusammen. Das vorhergesagte Label ist damit unabhängig von der Verschiebung.

<!--en:0a7c2355-->
::: {.slide title="Äquivarianz als kommutatives Diagramm"}

<!--en:8b462a97-->
[Verschieben und Filtern in beliebiger Reihenfolge]{.kicker}

<!--en:b2560476-->
Vergleicht $f$ jedes $2 \times 2$-Fenster mit einem $2 \times 2$-Ausschnitt aus Einsen, ergeben beide Wege durch das Diagramm dasselbe Ausgabegitter:

<!--en:a41b1c52-->
**Vertauschbarkeit:** Zuerst die Eingabe um $v = (3, 2)$ verschieben und dann den Filter $f$ anwenden (unterer Pfad) ergibt dieselbe Antwortkarte wie die umgekehrte Reihenfolge: zuerst $f$ anwenden und dann die Ausgabe um $v$ verschieben (rechter Pfad).

<!--en:bcda1502-->
::: {.slide title="Ein eindimensionales Beispiel: von 64 Gewichten auf 15 und dann auf 3"}

<!--en:5df8652b-->
[Toeplitz-Struktur und Lokalität in einer Dimension]{.kicker}

<!--en:8371d424-->
Ein eindimensionales Signal mit 8 Abtastwerten wird auf 8 versteckte Einheiten abgebildet ($h_i = u_i + \sum_k W_{i,k} x_k$). Jede geometrische Nebenbedingung schränkt die $8 \times 8$-Gewichtsmatrix $\mathbf{W}$ weiter ein:

<!--en:9fc3fee7-->
- **(a) Vollständig verbunden (64 Gewichte):** Jeder Eintrag $W_{i,k}$ ist unabhängig.

<!--en:f27032e2-->
- **(b) Äquivariant / Toeplitz (15 Gewichte):** Wandert ein Einheitsimpuls von $k$ nach $k+1$, verschiebt sich die Ausgabe um eine Zeile nach unten ($W_{i+1,k+1} = W_{i,k}$). Damit ist $\mathbf{W}$ entlang jeder Diagonalen $a = k - i \in \{-7,\dots,7\}$ konstant.

<!--en:19279bc2-->
- **(c) Äquivariant und lokal (3 Gewichte):** Betrachtet jede Einheit nur Nachbarn bis zum Abstand $\Delta = 1$, verschwinden alle Diagonalen mit $|a| > 1$. In jeder Zeile steht derselbe Faltungskern mit 3 Gewichten an der jeweils passenden Position.

<!--en:80c82f1a-->
::: {.slide title="Relative Versätze als Indizes einer vollständig verbundenen 2D-Schicht"}

<!--en:b43a1599-->
[Vom vierdimensionalen Gewichtstensor zu räumlichen Versätzen]{.kicker}

<!--en:a1a9adfb-->
- **Variablentransformation:** Den relativen Versatz $(a, b) = (k - i, l - j)$ einsetzen und $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$ definieren

<!--en:7f68aafc-->
- **Noch gleich viele Parameter:** Die neuen Indizes lesen $\mathsf{W}$ entlang räumlicher Diagonalen ab. Es bleiben $10^{12}$ freie Gewichte, bis die Abhängigkeit von $(i,j)$ entfällt.

<!--en:20da5483-->
::: {.slide title="Satz: Gewichtsteilung ergibt Äquivarianz"}

<!--en:6ad2acb3-->
[Hinreichende Bedingung auf dem unendlichen Gitter]{.kicker}

<!--en:786a2fa5-->
- **Unendliches, mit Nullen aufgefülltes Gitter:** Bei einer Verschiebung geht kein von null verschiedenes Pixel am Rand verloren.

<!--en:d019a06a-->
- **Hinreichende Bedingung:** Gelten überall $[\mathbf{U}]_{i,j} = u$ und $[\mathsf{V}]_{i,j,a,b} = [\mathbf{K}]_{a,b}$, so kommutiert $f$ mit jeder Verschiebung $T_v$.

<!--en:30b86525-->
::: {.slide title="Beweis der Umkehrung: Äquivarianz erzwingt Gewichtsteilung"}

<!--en:a1745b09-->
[Nachweis der Notwendigkeit mit einem Einheitsimpuls]{.kicker}

<!--en:f574c4a1-->
- **1. Konstanter Bias:** Aus $T_v \mathbf{0} = \mathbf{0} \implies \mathbf{U} = T_v \mathbf{U}$ folgt $[\mathbf{U}]_{i,j} = u$.

<!--en:46b9cb0b-->
- **2. Einheitsimpuls als Test:** Eine einzelne $1$ an der Position $(k,l)$ und eine Verschiebung um $v = (i,j)$ erzwingen $[\mathsf{V}]_{i,j,a,b} = [\mathsf{W}]_{0,0,a,b} \equiv [\mathbf{K}]_{a,b}$.

<!--en:15bdabe2-->
- **Äquivarianzsatz:** Eine lineare Schicht kommutiert mit Verschiebungen $\iff$ sie nutzt überall denselben Faltungskern $\mathbf{K}$ und denselben skalaren Bias $u$ ($10^{12} \to 4 \times 10^6$ Gewichte).

<!--en:f2e54e5e-->
::: {.slide title="Von der Lokalität zur Faltungsschicht"}

<!--en:531bb03c-->
[Beschränkung der Versätze auf ein $(2\Delta+1) \times (2\Delta+1)$-Fenster]{.kicker}

<!--en:962baced-->
- **Lokales Fenster $|a|, |b| \le \Delta$:** Gewichte für größere Versätze sind null. Jede Einheit betrachtet nur eine Umgebung der Größe $(2\Delta+1) \times (2\Delta+1)$.

<!--en:c8865c5f-->
- **Rückgang um 11 Größenordnungen:** Für $\Delta = 1$ ($3 \times 3$-Faltungskern) werden aus $10^{12}$ Gewichten (**4 TB**) nur **9 gemeinsam genutzte Gewichte** (**36 B**).

<!--en:10e6b66f-->
::: {.slide title="Induktiver Bias: Dateneffizienz und Ausdrucksstärke"}

<!--en:096cde0b-->
[Gewinne und Einschränkungen]{.kicker}

<!--en:5902303e-->
- **Dateneffizienz:** Die gemeinsame Nutzung von Beobachtungen an allen $H \times W$ Positionen verhindert Überanpassung an lokale, translationsäquivariante Bildstrukturen.

<!--en:35076420-->
- **Grenze einer einzelnen Schicht:** Bei geteilten Gewichten lassen sich identische Bildausschnitte nicht anhand ihrer absoluten Position unterscheiden (etwa ein Auge von einem Nasenloch).

<!--en:3060526f-->
- **Vorteil tiefer Netze:** Aufeinanderfolgende Schichten mit Nichtlinearitäten vergrößern das rezeptive Feld, bis es die räumliche Anordnung eines ganzen Objekts erfasst.

<!--en:17ef3020-->
::: {.slide title="Mathematische Faltung und Kreuzkorrelation"}

<!--en:e31fd85a-->
[Warum Deep-Learning-Bibliotheken den Faltungskern nicht spiegeln]{.kicker}

<!--en:d171eefe-->
- **Vorzeichenkonvention:** Die mathematische Faltung verwendet $(i-a, j-b)$, die Kreuzkorrelation $(i+a, j+b)$.

<!--en:1e766698-->
- **Identische Hypothesenklasse:** Da $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ die gelernten Gewichte um 180° dreht, implementieren Frameworks die Kreuzkorrelation und nennen die Schicht Faltungsschicht.

<!--en:21c579c9-->
::: {.slide title="Vom Grauwertgitter zum dreidimensionalen Tensor mit Kanälen"}

<!--en:b139de3c-->
[Mehrkanalige Eingaben und Merkmalskarten]{.kicker}

<!--en:8e9fdf8c-->
Reale Bilder enthalten mehrere Werte pro Pixel (z. B. Rot, Grün und Blau). Versteckte Schichten müssen Dutzende verschiedener visueller Muster parallel erkennen:

<!--en:a13814a6-->
- **Eingabetensor $\mathsf{X} \in \mathbb{R}^{c_\textrm{i} \times H \times W}$:** Er ordnet einen $c_\textrm{i}$-dimensionalen Vektor jeder Pixelposition $(i, j)$ zu ($c_\textrm{i} = 3$ für RGB).

<!--en:d78a6236-->
- **Grenze eines einzelnen Kanals:** Ein Faltungskern erzeugt nur eine Score-Karte (wie die einzelne Aktivierungskarte für Walter rechts).

<!--en:a19d9807-->
- **Ausgabetensor $\mathsf{H} \in \mathbb{R}^{c_\textrm{o} \times H \times W}$:** Er stapelt $c_\textrm{o}$ parallele **Merkmalskarten**. Jede erkennt ein anderes gelerntes Muster (gerichtete Kanten, Farbkontraste, Texturen).

<!--en:be6bd6b3-->
::: {.slide title="Der vierdimensionale Faltungskern und seine Parameteranzahl"}

<!--en:0a4cbd46-->
[Vollständige Verbindung der Kanäle bei räumlicher Gewichtsteilung]{.kicker}

<!--en:e8a33ab5-->
- **Vierdimensionaler Faltungskern $\mathsf{K} \in \mathbb{R}^{c_\textrm{o} \times c_\textrm{i} \times k_\textrm{h} \times k_\textrm{w}}$:** Parameteranzahl $= c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$

<!--en:b7d8816e-->
- **Unabhängigkeit von der Auflösung:** Eine $3 \times 3$-Schicht mit RGB-Eingabe und 64 Ausgabekanälen benötigt $1{,}792$ Parameter, für $28 \times 28$-Bilder ebenso wie für $4000 \times 4000$-Bilder.

<!--en:4ea517b7-->
::: {.slide title="Faltungen über Bilddaten hinaus: eine, zwei und drei Dimensionen"}

<!--en:6e5465b7-->
[Dasselbe Symmetrieprinzip für verschiedene Datenarten]{.kicker}

<!--en:978fc8f2-->
- **Gitter in 1D, 2D und 3D:** Äquivarianz und Lokalität ergeben 1D-Faltungen über die Zeit $(c_\textrm{i}, T)$, 2D-Faltungen über Bilder $(c_\textrm{i}, H, W)$ und 3D-Faltungen über Video- oder Volumendaten $(c_\textrm{i}, D, H, W)$.

<!--en:773ee4fb-->
- **Grenzen der Äquivarianz:** In einem Spektrogramm mit linearer Hertz-Skala verändert eine Frequenzverschiebung die Frequenzverhältnisse der Obertöne. Soll eine Verschiebung bei der 2D-Faltung eine Transposition abbilden, braucht es eine logarithmische Frequenzachse (näherungsweise eine Mel-Skala).

<!--en:a4a422e7-->
::: {.slide title="Verständnisfragen: Faltungen herleiten und Gewichte zählen"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:e09e9983-->
Die Antworten auf diese drei Fragen aus den Grundprinzipien herleiten:

<!--en:ee05fe7a-->
1. **Weniger Gewichte bei 28×28 Pixeln:** Ein einkanaliges $28 \times 28$-Bild wird auf ein $28 \times 28$-Gitter abgebildet. Wie viele Gewichte hat eine vollständig verbundene Schicht? Wie viele bleiben bei Translationsäquivarianz (Nullen außerhalb des Bildes) und wie viele bei zusätzlicher Beschränkung auf ein lokales $5 \times 5$-Fenster ($\Delta = 2$)?

<!--en:80e529a3-->
2. **Auflösung und Parameteranzahl:** Eine $3 \times 3$-Schicht bildet $c_\textrm{i}=3$ Eingabekanäle auf $c_\textrm{o}=64$ Ausgabekanäle ab. Wie viele Parameter hat sie bei einem $224 \times 224$-Bild, wie viele bei einem $1024 \times 1024$-Bild? Mit den $200{,}704$ Gewichten der ersten Schicht des Fashion-MNIST-MLP ($784 \to 256$) vergleichen.

<!--en:dcfaf507-->
3. **Periodische Randbedingung auf einem Ring:** Auf einem eindimensionalen Ring mit 4 Abtastwerten entspricht Index 4 wieder Index 0. Wie viele freie Gewichte hat eine äquivariante $4 \times 4$-Matrix, bevor Lokalität gefordert wird?

<!--en:382d96cc-->
**Lösungen:**
(1) Vollständig verbunden: $784^2 = 614{,}656$. Äquivariant (Versätze $-27 \dots 27$): $55^2 = 3{,}025$. Lokales $5 \times 5$-Fenster: $25$ Gewichte ($+1$ Bias).
(2) Beide Auflösungen benötigen $64 \times 3 \times 9 + 64 = 1{,}792$ Parameter ($< 0.9\%$ der Parameter der ersten MLP-Schicht).
(3) Genau $4$ freie Gewichte (zirkulante Matrix, jede Zeile ist gegenüber der vorherigen um 1 zyklisch verschoben, gegenüber $2(4)-1 = 7$ Gewichten bei einer mit Nullen aufgefüllten Folge).

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und zentrale Ergebnisse"}

<!--en:ce12e85f-->
[Der Abschnitt im Überblick]{.kicker}

<!--en:39e6da9f-->
| Prinzip / Mechanismus | Mathematische Form | Parameteranzahl ($c_\textrm{i} \to c_\textrm{o}$, Gitter $H \times W$) | Rolle im Deep Learning |

<!--en:f859eddc-->
| **Vollständig verbundene Schicht auf einem 2D-Gitter** | $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ | $c_\textrm{o} c_\textrm{i} H^2 W^2 + c_\textrm{o} HW$ | Beliebige räumliche Verbindungen, bei hoher Auflösung nicht praktikabel |

<!--en:271fa928-->
| **Translationsäquivarianz** | $f(T_v \mathbf{X}) = T_v f(\mathbf{X}) \iff [\mathsf{V}]_{i,j,a,b} = [\mathbf{K}]_{a,b}$ | $c_\textrm{o} c_\textrm{i} (2H-1)(2W-1) + c_\textrm{o}$ | Gewichtsteilung über alle Positionen, sodass sich Merkmalskarten mit der Eingabe verschieben |

<!--en:33d2f4b2-->
| **Lokalität ($\|a\|, \|b\| \le \Delta$)** | $[\mathbf{K}]_{a,b} = 0$ außerhalb des Fensters der Größe $(2\Delta+1) \times (2\Delta+1)$ | $c_\textrm{o} c_\textrm{i} (2\Delta+1)^2 + c_\textrm{o}$ | Beschränkung jeder Einheit auf eine lokale Umgebung, unabhängig von $H, W$ |

<!--en:81a41628-->
| **Kreuzkorrelation und Faltung** | $(\mathbf{X} \star \mathbf{K})_{i,j} = (\mathbf{K}' * \mathbf{X})_{i,j}$ | Identisch | Indizierung $(i+a, j+b)$ ohne Spiegelung in allen Frameworks |

<!--en:1d647a79-->
**Ausblick (§{sec_conv_layer}):** 2D-Kreuzkorrelation in Code umsetzen, einen Faltungskern zur Kantenerkennung von Hand entwerfen und per Gradientenabstieg *lernen*, Faltung als Matrizenmultiplikation (*im2col*) ausdrücken und rezeptive Felder über mehrere Schichten analysieren.
