<!--en:daab1405-->
# Mehrschichtige Perzeptrons

<!--en:b74a442d-->
In :numref:`chap_classification` berechnete die Softmax-Regression für jede Kleidungskategorie einen Score aus der gewichteten Summe der Pixelwerte und einem Bias: $o_j = \sum_{i=1}^d x_i w_{ij} + b_j$. Diese lineare Form bringt zwei wesentliche Einschränkungen mit sich:

<!--en:f0655a23-->
1. **Monotonie:** Erhöhen wir den Wert des Merkmals $x_i$, so ändert sich $o_j$ stets in dieselbe Richtung (nach oben für $w_{ij} > 0$, nach unten für $w_{ij} < 0$). Viele reale Zusammenhänge sind jedoch nicht monoton: Weicht die Körpertemperatur eines Menschen von $37^\circ\textrm{C}$ ab, steigt das Gesundheitsrisiko – und zwar in *beiden* Richtungen.

<!--en:e544564f-->
2. **Keine Wechselwirkungen zwischen Merkmalen:** Erhöhen wir $x_i$ um eine Einheit, ändert sich $o_j$ um $w_{ij}$, unabhängig von den Werten aller anderen Koordinaten. In einem Bild hat ein helles Pixel an der Position $(13, 17)$ jedoch eine ganz andere Bedeutung, je nachdem, ob seine Nachbarn einen Kragen, einen Ärmel oder einen Taschenriemen bilden.

<!--en:165d6340-->
:numref:`fig_mdl-mlp-temperature-risk` zeigt das Beispiel der Körpertemperatur. Selbst der am besten angepasste lineare Score steigt über den gesamten Temperaturbereich. Damit schreibt er einem Menschen mit Unterkühlung das geringste Risiko zu.

<!--en:8b60c68f-->
![Gesundheitsrisiko in Abhängigkeit von der Körpertemperatur. Das Risiko ist in der Nähe von $37^\circ\textrm{C}$ am niedrigsten und steigt zu beiden Seiten an. Ein linearer Score kann dagegen nur steigen oder nur fallen. Die gestrichelte Gerade wurde im dargestellten Bereich mit der Methode der kleinsten Quadrate angepasst.](../img/mdl-mlp-temperature-risk.svg)

<!--en:15108105-->
Ein *mehrschichtiges Perzeptron* (engl. multilayer perceptron, MLP) überwindet beide Einschränkungen durch eine oder mehrere *versteckte Schichten* (engl. hidden layers). Jede dieser Schichten wendet erst eine affine Abbildung und dann eine nichtlineare *Aktivierungsfunktion* an. So entstehen aus den Rohdaten erlernte Zwischenrepräsentationen, aus denen eine lineare Ausgabeschicht die endgültige Vorhersage berechnet. In diesem Abschnitt beweisen wir, dass kein lineares Modell das XOR-Problem mit vier Punkten lösen kann. Anschließend konstruieren wir mit zwei versteckten ReLU-Einheiten eine explizite Lösung. Wir untersuchen, welche Funktionen flache und tiefe Netze darstellen können, und vergleichen gebräuchliche Aktivierungsfunktionen im modernen Deep Learning.

<!--en:4ea3043f-->
## Grenzen linearer Modelle: das XOR-Problem

<!--en:bc5dfcbc-->
Besonders deutlich zeigt sich die Wechselwirkung zwischen Merkmalen an nur vier Datenpunkten: den Ecken des Einheitsquadrats $\{0, 1\}^2$. Ihr Label ergibt sich aus dem *exklusiven Oder* (XOR) ihrer Koordinaten. Es gilt genau dann $y = 1$, wenn *genau eine* Koordinate den Wert $1$ hat:

<!--en:72e5ed0e-->
Hier zeigt sich die Wechselwirkung: Schalten wir $x_1$ von $0$ auf $1$ um, steigt $y$ für $x_2 = 0$, doch $y$ sinkt für $x_2 = 1$.

<!--en:d2974410-->
**Proposition.** Für alle $w_1, w_2, b \in \mathbb{R}$ ordnet die lineare Entscheidungsregel „Klasse 1 vorhersagen, wenn $w_1 x_1 + w_2 x_2 + b > 0$“ mindestens einen der vier XOR-Punkte der falschen Klasse zu.

<!--en:da2de205-->
**Beweis.** Nehmen wir an, es gäbe $w_1, w_2, b$, die alle vier Ecken korrekt klassifizieren. Für die beiden Punkte $(0, 1)$ und $(1, 0)$ der Klasse 1 müssen $w_2 + b > 0$ und $w_1 + b > 0$ gelten. Addition ergibt $w_1 + w_2 + 2b > 0$. Für die beiden Punkte $(0, 0)$ und $(1, 1)$ der Klasse 0 müssen dagegen $b \leq 0$ und $w_1 + w_2 + b \leq 0$ gelten. Addition ergibt $w_1 + w_2 + 2b \leq 0$, einen unmittelbaren Widerspruch. $\blacksquare$

<!--en:a68fed3e-->
Geometrisch lässt sich das am gemeinsamen Mittelpunkt $(1/2, 1/2)$ der beiden Diagonalen des Einheitsquadrats erkennen (:numref:`fig_mdl-mlp-xor-diagonals`). Dort ist der Wert jeder affinen Funktion gleich dem Mittelwert ihrer Werte an den Endpunkten jeder der beiden Diagonalen. Daher kann keine Gerade die beiden Diagonalen trennen.

<!--en:615b49f2-->
![Die vier XOR-Punkte mit ihren Klassenlabels. Jede Diagonale verbindet die beiden Punkte einer Klasse. Da sich die Diagonalen in $(1/2, 1/2)$ schneiden, trennt keine Gerade die Klassen.](../img/mdl-mlp-xor-diagonals.svg)

<!--en:042165d3-->
## Versteckte Schichten

<!--en:61903b5f-->
### Das Modell

<!--en:11de8134-->
Um die Linearität zu überwinden, fügt ein MLP zwischen den Eingabemerkmalen und der Ausgabeschicht eine oder mehrere *versteckte Schichten* ein (:numref:`fig_mlp`). Betrachten wir die Entwurfsmatrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ eines Minibatchs mit $n$ Beispielen und $d$ Eingabemerkmalen. Bei einer einzigen versteckten Schicht der Breite $h$ hat diese die Gewichte $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ und die Bias-Terme $\mathbf{b}^{(1)} \in \mathbb{R}^h$. Die Ausgabeschicht hat die Gewichte $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ und die Bias-Terme $\mathbf{b}^{(2)} \in \mathbb{R}^q$. Das zweischichtige MLP mit einer versteckten Schicht berechnet

<!--en:10a4ad8a-->
Dabei addieren wir die Bias-Terme per Broadcasting zu jeder Zeile. Die nichtlineare *Aktivierungsfunktion* $\phi$ wirkt elementweise auf die *Präaktivierungen* $\mathbf{Z} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$ und liefert die versteckten *Aktivierungen* $\mathbf{H} \in \mathbb{R}^{n \times h}$. Die Parameteranzahl beträgt $(d + 1)h + (h + 1)q$. Für Fashion-MNIST ($d = 784, q = 10$) hat ein MLP mit einer versteckten Schicht aus $h = 256$ Einheiten $785 \times 256 + 257 \times 10 = 203{,}530$ Parameter, also 26-mal so viele wie die Softmax-Regression mit ihren $7{,}850$ Parametern.

<!--en:016af59b-->
![Ein MLP mit 4 Eingaben, 1 versteckten Schicht aus 5 versteckten Einheiten und 3 Ausgaben. Beide parametrisierten Schichten sind vollständig verbunden.](../img/mlp.svg)

<!--en:b9cb8e41-->
### Die Rolle der Nichtlinearität

<!--en:92f084c4-->
Warum ist die nichtlineare Aktivierung $\phi$ unverzichtbar? Verzichten wir auf $\phi$, indem wir $\phi(z) = z$ setzen, so gilt $\mathbf{H} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$. Setzen wir $\mathbf{H}$ in die Ausgabegleichung ein, reduzieren sich die beiden Schichten wieder auf eine einzige affine Transformation:

<!--en:3dba74a7-->
Ohne Nichtlinearität können auch beliebig viele aufeinanderfolgende vollständig verbundene Schichten nur Funktionen darstellen, die eine einzige lineare Schicht darstellen kann. Ist eine versteckte Schicht schmaler als $\min(d, q)$, beschränkt sie $\mathbf{W}$ sogar auf Matrizen von niedrigem Rang (siehe Aufgabe :exref:`ex_mlp-linear-networks`). Eine elementweise Nichtlinearität wie die *ReLU* (engl. rectified linear unit),

<!--en:627baa15-->
verhindert diese algebraische Vereinfachung.

<!--en:a8ea99a7-->
### Lösung des XOR-Problems mit zwei ReLU-Einheiten

<!--en:4329186c-->
Mit ReLU-Aktivierungen genügen einer versteckten Schicht bereits $h = 2$ Einheiten, um XOR zu lösen :cite:`Goodfellow.Bengio.Courville.2016`. Wir setzen

<!--en:4ed0c703-->
Beide versteckten Einheiten hängen von der Koordinatensumme $s = x_1 + x_2$ ab. Einheit 1 berechnet $h_1 = \mathrm{ReLU}(s) = s$ und Einheit 2 berechnet $h_2 = \mathrm{ReLU}(s - 1)$. Die zweite Einheit ist *nur* dann aktiv, wenn $s = 2$ gilt, also am Punkt $(1, 1)$. Dort zieht die Ausgabefunktion $o = h_1 - 2h_2$ den Wert $2$ ab, sodass die Ausgabe wieder auf $0$ fällt:

<!--en:89b7f5f3-->
Betrachten wir die Geometrie in :numref:`fig_mdl-mlp-xor`. Die versteckte Schicht bildet *beide* Ecken $(0, 1)$ und $(1, 0)$ der Klasse 1 auf denselben Punkt $(1, 0)$ ab. Der Geraden $h_1 - 2h_2 = 1/2$ im Raum der versteckten Aktivierungen (rechts) entspricht deshalb im Eingaberaum (links) die Grenze des diagonalen Streifens $1/2 < x_1 + x_2 < 3/2$. Dieser Streifen ist kein Halbraum.

<!--en:6615ddbc-->
![Lösung des XOR-Problems mit zwei versteckten ReLU-Einheiten. Links: Im Eingaberaum $(x_1, x_2)$ sagt das Netz innerhalb des diagonalen Streifens $1/2 < x_1 + x_2 < 3/2$ die Klasse 1 vorher. Rechts: Im Raum der versteckten Aktivierungen $(h_1, h_2)$ fallen die beiden Punkte der Klasse 1 im Punkt $(1, 0)$ zusammen. Eine einzige Gerade $h_1 - 2h_2 = 1/2$ trennt dort die Klassen.](../img/mdl-mlp-xor.svg)

<!--en:1495041a-->
### Tiefere Netze

<!--en:f87c329f-->
Allgemein folgen in einem MLP mit $K$ Schichten insgesamt $K - 1$ versteckte Schichten auf die Eingabe $\mathbf{H}^{(0)} = \mathbf{X}$:

<!--en:940dfe47-->
So entsteht durch Verkettung eine Hierarchie von Repräsentationen, die von den Rohdaten bis zu den Klassen-Logits reicht.

<!--en:44232d7e-->
## Ausdrucksstärke

<!--en:11613b98-->
### Universelle Approximation

<!--en:5bdd0f99-->
Welche Funktionen kann eine einzige versteckte Schicht ($K = 2$) berechnen? Auf der reellen Achse ($d = 1$) liefert jede versteckte ReLU-Einheit $a_j\,\mathrm{ReLU}(x - t_j)$ eine *Knickfunktion* (engl. hinge). Links von ihrem Knoten $t_j$ verläuft sie waagerecht, rechts davon ändert sich die Steigung um $a_j$ (:numref:`fig_mdl-mlp-uat-hinges`). Addieren wir $h$ Knickfunktionen und eine Konstante, ergibt sich ein stetiger Polygonzug mit $h$ inneren Knoten. Bei geeigneter Anfangssteigung lässt sich so jeder solche Polygonzug darstellen. Auf einem abgeschlossenen, beschränkten Intervall $[a, b]$ lässt sich jede stetige Funktion durch einen hinreichend feinen Polygonzug gleichmäßig mit einem Fehler unter jeder vorgegebenen Schranke $\epsilon > 0$ approximieren. Daher kann eine einzige, hinreichend breite versteckte ReLU-Schicht jede stetige Funktion einer Variablen auf diesem Intervall beliebig genau approximieren.

<!--en:ec348c07-->
![Approximation einer glatten Kurve auf einem Intervall durch eine Summe von ReLU-Knickfunktionen $a_j\,\mathrm{ReLU}(x - t_j)$. Jeder Steigungsänderung entspricht eine versteckte Einheit.](../img/mdl-mlp-uat-hinges.svg)

<!--en:95678bcc-->
Der *universelle Approximationssatz* verallgemeinert diese Aussage auf beliebige Eingabedimensionen $d$. Ein MLP mit einer einzigen versteckten Schicht kann *jede* stetige Funktion auf jeder kompakten Teilmenge von $\mathbb{R}^d$ gleichmäßig mit beliebiger Genauigkeit approximieren. Dazu eignet sich *jede* stetige, nichtpolynomiale Aktivierungsfunktion $\phi$ :cite:`Cybenko.1989,Leshno.Lin.Pinkus.ea.1993`. Einen Beweis für zwei Dimensionen mit Ridge-Funktionen und dem Satz von Stone-Weierstrass behandelt Aufgabe :exref:`ex_mlp-one-hidden-layer`.

<!--en:ea3212a1-->
### Tiefe und Raumfaltung

<!--en:85c8124e-->
Wenn eine versteckte Schicht bereits alles approximieren kann, warum bauen wir dann *tiefe* Netze? Die universelle Approximation sagt nichts darüber aus, *wie viele Einheiten* eine einzige Schicht braucht. Ihre Anzahl kann exponentiell mit $d$ wachsen. Ebenso bleibt offen, ob der Gradientenabstieg aus endlich vielen Daten eine Lösung lernen kann und ob diese generalisiert.

<!--en:8e740a37-->
Tiefe Netze können mit exponentiell weniger Parametern auskommen, indem sie Abbildungen *verketten*, die den Raum falten :cite:`Telgarsky.2016`. Betrachten wir die eindimensionale Zeltfunktion auf $[0, 1]$, die zwei ReLU-Einheiten berechnen:

<!--en:851128a6-->
Sie steigt von $0$ auf $1$ im Intervall $[0, 1/2]$ und fällt wieder auf $0$ im Intervall $[1/2, 1]$ (:numref:`fig_mdl-mlp-depth-folding`). Verketten wir $g$ mit sich selbst, faltet jede Schicht das Intervall $[0, 1]$ in der Mitte. So hat $g \circ g$ insgesamt $4$ lineare Abschnitte und $g \circ g \circ g$ bereits $8$. Die $m$-fache Verkettung $g^{\circ m}$ lässt sich mit $m$ versteckten Schichten aus je $2$ ReLU-Einheiten (insgesamt $2m$ Einheiten) berechnen. Sie erzeugt eine Sägezahnkurve mit $2^m$ linearen Abschnitten. Um dieselbe Kurve mit $2^m$ Abschnitten in einer *einzigen* versteckten Schicht darzustellen, sind mindestens $2^m - 1$ versteckte Einheiten nötig (für eine Schicht $1{,}023$ Einheiten gegenüber $20$ Einheiten in $10$ Schichten bei $m = 10$).

<!--en:76aa9b96-->
![Bei der Verkettung der Zeltfunktion $g$ aus zwei ReLU-Einheiten mit sich selbst verdoppelt jede Schicht die Anzahl der linearen Abschnitte. Die $m$-fache Verkettung erzeugt so $2^m$ Abschnitte mit nur $2m$ versteckten Einheiten.](../img/mdl-mlp-depth-folding.svg)

<!--en:675c3560-->
## Aktivierungsfunktionen

<!--en:9cf309ab-->
Für die universelle Approximation genügt fast jede nichtpolynomiale Aktivierungsfunktion $\phi$. Für die Optimierung durch Gradientenabstieg ist dagegen die Ableitung $\phi'(z)$ entscheidend. Betrachten wir eine versteckte Einheit $j$ mit Präaktivierung $z_j = \sum_i x_i w^{(1)}_{ij} + b^{(1)}_j$ und Aktivierung $h_j = \phi(z_j)$. Nach der Kettenregel skaliert die lokale Steigung $\phi'(z_j)$ den Gradienten jedes eingehenden Gewichts:

<!--en:c0464431-->
Wo $\phi'(z_j) \approx 0$ gilt, stockt der Gradientenfluss (engl. gradient flow). :numref:`fig_mdl-mlp-activations` vergleicht die vier wichtigsten Aktivierungsfunktionen und ihre Ableitungen.

<!--en:9eff52d6-->
![Vier Aktivierungsfunktionen $\phi(z)$ (links) und ihre Ableitungen $\phi'(z)$ (rechts): oben ReLU und GELU, unten die Sigmoidfunktion und $\tanh$.](../img/mdl-mlp-activations.svg)

<!--en:9ac318dd-->
- **ReLU und Leaky ReLU:** $\mathrm{ReLU}(z) = \max(z, 0)$ hat die Ableitung $1$ für $z > 0$ und $0$ für $z < 0$ (vereinbarungsgemäß $0$ bei $z = 0$). Auf der gesamten positiven Halbachse ist ihre Steigung konstant $1$. Aktive ReLU-Einheiten schwächen Gradienten daher auch bei wachsendem $z$ nicht ab :cite:`Nair.Hinton.2010,Glorot.Bordes.Bengio.2011`. Verschieben sich die Gewichte einer Einheit jedoch so, dass $z < 0$ für jede Trainingseingabe gilt, verschwindet ihr Gradient dauerhaft. Man spricht von einer *toten ReLU* (engl. dead ReLU). Die *Leaky ReLU*, $\max(z, \alpha z)$ mit $\alpha \approx 0.01$, behält für $z < 0$ eine kleine positive Steigung. Dadurch können inaktive Einheiten wieder aktiv werden.

<!--en:1a5fba5c-->
- **GELU und SiLU (Swish):** Die *Gaussian Error Linear Unit* $\mathrm{GELU}(z) = z\,\Phi(z)$ (mit der Verteilungsfunktion $\Phi$ der Standardnormalverteilung) und die verwandte Funktion $\mathrm{SiLU}(z) = z\,\sigma(z)$ glätten den Knick der ReLU im Ursprung. Beide verlaufen durch $z = 0$ mit Steigung $1/2$. Die GELU hat nahe $z \approx -0.75$ eine flache Senke im negativen Bereich. In modernen Transformern sind beide Funktionen als Aktivierungen üblich :cite:`Hendrycks.Gimpel.2016`.

<!--en:841d5af1-->
- **Sigmoidfunktion und Tanh:** Die logistische Sigmoidfunktion $\sigma(z) = 1/(1 + e^{-z}) \in (0, 1)$ hat die Ableitung $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \leq 1/4$. Der zum Ursprung punktsymmetrische hyperbolische Tangens $\tanh(z) = (1 - e^{-2z})/(1 + e^{-2z}) = 2\sigma(2z) - 1 \in (-1, 1)$ (Aufgabe :exref:`ex_mlp-sigmoid-tanh`) hat die Ableitung $1 - \tanh^2(z) \leq 1$. Beide Funktionen nähern sich in der Sättigung horizontalen Asymptoten ($\phi'(z) \to 0$), sobald $|z|$ über $\sim 3$ hinauswächst. Multiplizieren sich die Ableitungen $\sigma'(z) \leq 1/4$ über $K$ Schichten, bleibt höchstens der Anteil $4^{-K}$ des Gradienten erhalten (*verschwindende Gradienten*, engl. vanishing gradients, siehe :numref:`sec_numerical_stability`). Heute dienen die Sigmoidfunktion und $\tanh$ in rekurrenten Architekturen vor allem als beschränkte *Gates* und zur Begrenzung der Zustandswerte (:numref:`chap_modern_rnn`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:901c964b-->
Lineare Modelle sind in jeder Koordinate monoton und können Wechselwirkungen zwischen Merkmalen wie bei XOR nicht erfassen. Mehrschichtige Perzeptrons überwinden beide Einschränkungen, indem sie vor der linearen Ausgabeschicht versteckte Schichten $\mathbf{H}^{(k)} = \phi(\mathbf{H}^{(k-1)}\mathbf{W}^{(k)} + \mathbf{b}^{(k)})$ einfügen. Ohne eine nichtlineare Aktivierung $\phi$ reduzieren sich aufeinanderfolgende affine Transformationen auf eine einzige affine Abbildung. Mit jeder stetigen, nichtpolynomialen Aktivierungsfunktion kann eine einzige, hinreichend breite versteckte Schicht jede stetige Funktion auf einer kompakten Menge approximieren. Verketten wir dagegen $m$ versteckte Schichten, können sie den Raum wiederholt falten und so $2^m$ lineare Bereiche mit $O(m)$ Einheiten erzeugen. Gleichrichtende Aktivierungsfunktionen wie ReLU ($\max(z, 0)$) und GELU ($z\Phi(z)$) haben für große positive Eingaben eine Steigung von eins oder nahezu eins. Damit vermeiden sie in diesem Bereich die Sättigung, die Gradienten in tiefen Netzen mit Sigmoid- oder $\tanh$-Einheiten abschwächt.

<!--en:2294243a-->
## Übungen

<!--en:6d21cc54-->
1. **Lineare Netze.** Ohne Aktivierungsfunktionen verkettet ein mehrschichtiges Netz affine Abbildungen zu einer weiteren affinen Abbildung. Betrachten Sie ein solches Netz mit Eingabedimension $d$, Ausgabedimension $q$ und versteckten Schichten der Breiten $h_1, \ldots, h_{K-1}$. Zeigen Sie, dass es genau diejenigen affinen Abbildungen $\mathbf{X}\mathbf{W} + \mathbf{b}$ darstellt, deren Gewichtsmatrix $\mathbf{W}$ höchstens den Rang $\min(h_1, \ldots, h_{K-1})$ hat. Geben Sie mithilfe dieses Ergebnisses ein Beispiel an, bei dem eine zusätzliche versteckte Schicht die Funktionenklasse eines linearen Modells echt verkleinert.
   :exlabel:`ex_mlp-linear-networks`

<!--en:4e88943e-->
1. **XOR von Hand.** :numref:`fig_mdl-mlp-xor` zeigt ein ReLU-Netz mit zwei Einheiten, das XOR berechnet. Es bildet die beiden Eingaben der Klasse 1 auf denselben Punkt im Raum der versteckten Aktivierungen ab.
    1. Finden Sie Gewichte für ein ReLU-Netz mit zwei Einheiten, das ebenfalls XOR berechnet, die beiden Eingaben der Klasse 1 jedoch auf *verschiedene* Punkte im Raum der versteckten Aktivierungen abbildet. Prüfen Sie Ihre Lösung an allen vier Eingaben.
    1. Zeigen Sie, dass kein Netz mit einer einzigen versteckten ReLU-Einheit und einer linearen Ausgabe XOR korrekt klassifiziert, unabhängig vom Schwellenwert für die Ausgabe.

<!--en:593b0620-->
1. **Sigmoidfunktion und Tanh.** Die beiden begrenzenden Funktionen sind eng miteinander verwandt.
    1. Zeigen Sie, dass $\tanh(z) = 2\sigma(2z) - 1$ gilt.
    1. Zeigen Sie, dass es zu jedem MLP mit Tanh-Aktivierungen in den versteckten Schichten ein MLP derselben Architektur mit Sigmoid-Aktivierungen gibt, das dieselbe Funktion berechnet, und umgekehrt. Geben Sie an, wie sich Gewichte und Bias-Terme ändern.
   :exlabel:`ex_mlp-sigmoid-tanh`

<!--en:14d854db-->
1. **Tiefe und lineare Abschnitte.** ● Auf der reellen Achse hat ein ReLU-Netz mit einer einzigen versteckten Schicht aus $h$ Einheiten höchstens $h + 1$ lineare Abschnitte. Jede Verkettung mit der Zeltfunktion aus zwei Einheiten in :numref:`fig_mdl-mlp-depth-folding` verdoppelt dagegen die Anzahl der Abschnitte. In dieser Aufgabe bestimmen Sie für ein Netz mit drei Einheiten pro Schicht genau, wie viel die Tiefe bringt.
    1. Betrachten Sie drei versteckte Einheiten $\mathrm{ReLU}(w_j x + b_j)$ mit den Eingabegewichten $w_1 = w_2 = 1$ und $w_3 = -1$. Die Knoten $-b_j/w_j$ liegen bei $1/6$, $2/6$ und $4/6$. Wählen Sie die Ausgabegewichte und den Ausgabe-Bias so, dass die Ausgabe $f$ die Werte $0, 1, 0, 1, 0$ an den Stellen $x = 0, 1/6, 2/6, 4/6, 1$ annimmt.
    1. Zählen Sie die linearen Abschnitte von $f \circ f$ auf $[0, 1]$ und die der $m$-fachen Verkettung $f \circ \cdots \circ f$. Erklären Sie anhand der Faltung des Graphen bei der Verkettung mit $f$, warum jede Verkettung die Anzahl der Abschnitte multiplikativ statt additiv erhöht.
    1. Ein Netz mit $m$ versteckten Schichten zu je drei Einheiten berechnet die $m$-fache Verkettung. Wie viele versteckte Einheiten bräuchte ein Netz mit nur einer versteckten Schicht, um dieselbe Funktion darzustellen? Vergleichen Sie die Parameteranzahl der beiden Netze.

    *Nach Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), Aufgabe 4.8.*

<!--en:796303ed-->
1. **Eine versteckte Schicht genügt.** ● Auf der reellen Achse approximiert eine einzige versteckte Schicht eine stetige Funktion durch einen Polygonzug mit einer Knickfunktion pro Knoten. In der Ebene gibt es keine solchen Knoten zu platzieren. Hier braucht der Beweis eine andere Idee: Funktionen, die sich nur entlang einer Richtung ändern.
    1. Sei $\psi$ stetig auf $\mathbb{R}$ und sei $\mathbf{w} \in \mathbb{R}^2$. Zeigen Sie, dass eine einzige versteckte Schicht die *Ridge-Funktion* $\psi(\mathbf{w}^\top \mathbf{x})$ auf jeder kompakten Teilmenge der Ebene gleichmäßig approximieren kann. Die Funktion ist auf jeder Geraden orthogonal zu $\mathbf{w}$ konstant.
    1. Der Satz von Stone-Weierstrass besagt Folgendes: Eine Menge stetiger Funktionen auf einer kompakten Menge sei unter Summen, Produkten und skalaren Vielfachen abgeschlossen. Sie enthalte die konstanten Funktionen und für je zwei verschiedene Punkte eine Funktion, die dort unterschiedliche Werte annimmt. Dann ist jede stetige Funktion auf der kompakten Menge gleichmäßiger Grenzwert einer Folge von Funktionen aus der betrachteten Funktionenmenge. Zeigen Sie, dass die endlichen Linearkombinationen von $\cos(\boldsymbol{\omega}^\top \mathbf{x})$ und $\sin(\boldsymbol{\omega}^\top \mathbf{x})$ mit $\boldsymbol{\omega} \in \mathbb{R}^2$ eine solche Menge bilden.
    1. Folgern Sie daraus, dass eine einzige versteckte Schicht jede stetige Funktion auf einer kompakten Teilmenge der Ebene approximieren kann.
   :exlabel:`ex_mlp-one-hidden-layer`

    *Nach Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap4.html), Kapitel 4.*

<!--en:5eb13fea-->
[Diskussion](https://d2l.discourse.group/t/90)

<!--en:28d2bc2f-->
[Diskussion](https://d2l.discourse.group/t/91)

<!--en:efa9eb92-->
[Diskussion](https://d2l.discourse.group/t/226)

<!--en:69a8331d-->
[Diskussion](https://d2l.discourse.group/t/17984)

<!--en:f744bb21-->
Mehrschichtige Perzeptrons: versteckte Schichten und Nichtlinearität<br>**XOR und Wechselwirkungen zwischen Merkmalen · versteckte Repräsentationen · universelle Approximation · Raumfaltung durch Tiefe · Ableitungen der Aktivierungsfunktionen**

<!--en:a72d7daf-->
::: {.slide title="Zwei strukturelle Grenzen linearer Modelle"}

<!--en:777ade46-->
[Warum Linearität nicht ausreicht]{.kicker}

<!--en:5cf723dc-->
- **Monotonie in jeder Koordinate:** Steigt der Wert des Merkmals $x_i$, ändert sich $o_j = \sum_i x_i w_{ij} + b_j$ stets in dieselbe Richtung. Das Gesundheitsrisiko steigt dagegen auf *beiden* Seiten einer Körpertemperatur von $37^\circ\textrm{C}$.

<!--en:1530da0d-->
- **Keine Wechselwirkungen zwischen Merkmalen:** Steigt $x_i$ um $1$, ändert sich $o_j$ um $w_{ij}$, unabhängig von allen anderen Merkmalen. Die Bedeutung eines hellen Pixels hängt dagegen davon ab, ob seine Nachbarn einen Kragen oder einen Ärmel bilden.

<!--en:40147df7-->
- **Erlernte Zwischenmerkmale:** Bei $d = 784$ Pixeln lassen sich die Wechselwirkungen mehrerer Pixel kaum von Hand aufzählen. Versteckte Schichten lernen sie direkt aus den Daten.

<!--en:6df5945f-->
::: {.slide title="Das XOR-Problem: Vier Punkte, die keine Gerade trennen kann"}

<!--en:76b62535-->
[Wechselwirkungen zwischen Merkmalen]{.kicker}

<!--en:1a53cc02-->
- **Richtungswechsel auf $\{0, 1\}^2$:** Ein Wechsel bei $x_1$ von $0$ auf $1$ *erhöht* $y = x_1 \oplus x_2$ für $x_2 = 0$, *senkt* aber $y$ für $x_2 = 1$.

<!--en:55d8e2ec-->
- **Algebraischer Unmöglichkeitsbeweis:** Klasse $1$ an den Punkten $(0, 1), (1, 0)$ erfordert $w_1 + w_2 + 2b > 0$, Klasse $0$ an den Punkten $(0, 0), (1, 1)$ dagegen $w_1 + w_2 + 2b \le 0$. Das ist ein Widerspruch. $\blacksquare$

<!--en:17e81d52-->
- **Gemeinsamer Mittelpunkt der Diagonalen:** Beide Diagonalen schneiden sich in $(1/2, 1/2)$. Jeder lineare Klassifikator ordnet daher mindestens eine Ecke der falschen Klasse zu.

<!--en:45b028ee-->
::: {.slide title="Versteckte Schichten: affine Abbildung mit anschließender Nichtlinearität"}

<!--en:ee7c2c3b-->
[Architektur des MLP]{.kicker}

<!--en:b55afcd2-->
Ein MLP überwindet die Linearität durch eine oder mehrere *versteckte Schichten*. Sie wandeln vor der Ausgabeschicht einen Minibatch $\mathbf{X} \in \mathbb{R}^{n \times d}$ in erlernte Repräsentationen $\mathbf{H} \in \mathbb{R}^{n \times h}$ um:

<!--en:f3394ccd-->
- **Parameter der ersten Schicht:** Aus den Gewichten $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ und dem per Broadcasting zu jeder Zeile addierten Bias $\mathbf{b}^{(1)} \in \mathbb{R}^h$ ergeben sich die Präaktivierungen $\mathbf{Z} \in \mathbb{R}^{n \times h}$.

<!--en:2975f9f5-->
- **Elementweise Aktivierung $\phi$:** Auf jeden Eintrag wirkt eine skalare Nichtlinearität wie $\mathrm{ReLU}(z) = \max(z, 0)$.

<!--en:892ac331-->
- **Ausgabeschicht:** $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ und $\mathbf{b}^{(2)} \in \mathbb{R}^q$ bilden die versteckten Merkmale $\mathbf{H}$ auf die Klassen-Logits $\mathbf{O} \in \mathbb{R}^{n \times q}$ ab.

<!--en:d4744a44-->
::: {.slide title="Warum lineare Schichten ohne φ zu einer Schicht werden"}

<!--en:592b0463-->
[Die Rolle der Nichtlinearität]{.kicker}

<!--en:519ca1dc-->
- **Kein Gewinn an Ausdrucksstärke:** Ohne $\phi$ ergibt das Ausmultiplizieren beliebig vieler linearer Schichten eine einzige affine Abbildung mit einer Gewichtsmatrix der Größe $d \times q$. Auch sie kann XOR nicht lösen.

<!--en:0c358352-->
- **Niedriger Rang als Flaschenhals:** Aus $h_k < \min(d, q)$ folgt $\mathrm{rank}(\mathbf{W}) \le h_k$. Die lineare Modellklasse wird dadurch *echt kleiner*.

<!--en:2c1b984d-->
- **Grenze der Assoziativität:** Bei einer elementweisen Aktivierung $\mathrm{ReLU}(z) = \max(z, 0)$ zwischen den Matrixprodukten lässt sich $\mathrm{ReLU}(\mathbf{X}\mathbf{W}^{(1)})\mathbf{W}^{(2)}$ nicht mehr zu $\mathbf{X}(\mathbf{W}^{(1)}\mathbf{W}^{(2)})$ umklammern.

<!--en:37bf730c-->
::: {.slide title="Explizite Lösung des XOR-Problems mit zwei versteckten ReLU-Einheiten"}

<!--en:d9f29e36-->
[Konstruktives Beispiel]{.kicker}

<!--en:3ca8bbe7-->
Mit $h = 2$ versteckten ReLU-Einheiten lässt sich XOR in geschlossener Form lösen. Beide Einheiten verwenden die Koordinatensumme $s = x_1 + x_2$, mit Schwellenwerten bei $0$ und $1$:

<!--en:00930b57-->
- **Einheit 1 ($h_1 = \mathrm{ReLU}(x_1 + x_2)$):** Gibt an allen vier Ecken die Summe $s \in \{0, 1, 2\}$ aus.

<!--en:0c635cba-->
- **Einheit 2 ($h_2 = \mathrm{ReLU}(x_1 + x_2 - 1)$):** Bleibt gleich $0$ an den Punkten $(0, 0), (0, 1), (1, 0)$. Sie wird *nur* an einer Ecke aktiv: $h_2 = 1$ gilt am Punkt $(1, 1)$. Deshalb zieht die Ausgabe $o = h_1 - 2h_2$ den Wert $2$ ab und fällt an $(1, 1)$ wieder auf $0$.

<!--en:bb3c51a6-->
::: {.slide title="Alternative Gewichte für XOR und Grenzen einer einzelnen Einheit"}

<!--en:eab2b661-->
[Konstruktives Beispiel · Eindeutigkeit]{.kicker}

<!--en:32fd6f9a-->
- **Verschiedene Punkte im Raum der versteckten Aktivierungen:** Mit $h_1 = \mathrm{ReLU}(x_1 - x_2)$ und $h_2 = \mathrm{ReLU}(x_2 - x_1)$ gilt $(1, 0) \mapsto (1, 0)$ und $(0, 1) \mapsto (0, 1)$. Die beiden Ecken $(0, 0)$ und $(1, 1)$ der Klasse 0 fallen dagegen in $(0, 0)$ zusammen. Damit löst $o = h_1 + h_2$ das XOR-Problem.

<!--en:a0d650b2-->
- **Warum 1 versteckte ReLU-Einheit für XOR nicht ausreicht:** Eine einzelne Einheit $h_1 = \mathrm{ReLU}(w_1 x_1 + w_2 x_2 + b)$ ist entlang $(w_1, w_2)$ monoton und kann nur einen Halbraum abgrenzen. Daher sind mindestens zwei Einheiten nötig ($h \ge 2$).

<!--en:c1a071b6-->
::: {.slide title="Geometrie von XOR: Raumfaltung bis zur linearen Trennbarkeit"}

<!--en:5630e7d3-->
[Repräsentationslernen]{.kicker}

<!--en:3caa76ff-->
Der Vergleich des Eingaberaums $(x_1, x_2)$ mit dem erlernten Raum der versteckten Aktivierungen $(h_1, h_2)$ zeigt, wie versteckte Schichten linear untrennbare Punktmengen klassifizieren:

<!--en:7691ec0c-->
- **Eingaberaum (links):** Der diagonale Streifen $1/2 < x_1 + x_2 < 3/2$ ergibt sich als Schnitt zweier Halbräume.

<!--en:b5591653-->
- **Raum der versteckten Aktivierungen (rechts):** Die Abbildung $(x_1, x_2) \mapsto (h_1, h_2)$ faltet das Einheitsquadrat entlang $x_1 + x_2 = 1$. Dadurch fallen *beide* Ecken $(0, 1)$ und $(1, 0)$ der Klasse 1 in einem einzigen Punkt $(1, 0)$ zusammen.

<!--en:3ece3ca8-->
- **Lineare Ausgabe:** Im Raum der versteckten Aktivierungen trennt eine einzige Gerade $h_1 - 2h_2 = 1/2$ die Klasse $1$ von der Klasse $0$.

<!--en:5838a916-->
::: {.slide title="Parameteranzahl und Hierarchien in tiefen Netzen"}

<!--en:fafe39f2-->
[Von 2 Schichten zu K Schichten]{.kicker}

<!--en:1558fef5-->
- **Parameteranzahl bei 1 versteckten Schicht:** $(d + 1)h + (h + 1)q = dh + h + hq + q$ Parameter und $dh + hq$ Multiplikationen mit anschließender Addition pro Beispiel

<!--en:c306c86e-->
- **Erste Schicht dominiert bei Fashion-MNIST ($d = 784, h = 256, q = 10$):** Auf $\mathbf{W}^{(1)}, \mathbf{b}^{(1)}$ entfallen $785 \times 256 = 200{,}960$ der $203{,}530$ Parameter, also fast $99\%$ aller Modellparameter.

<!--en:bfd7bde0-->
- **Zweite versteckte Schicht ($256 \to 256$):** Sie fügt $257 \times 256 = 65{,}792$ Parameter hinzu (insgesamt $269{,}322$).

<!--en:75d00da2-->
::: {.slide title="Universelle Approximation in einer Dimension durch eine Summe von ReLU-Knickfunktionen"}

<!--en:6f77220b-->
[Ausdrucksstärke · Breite]{.kicker}

<!--en:6ba146f7-->
Welche Funktionen kann eine einzige versteckte Schicht ($K = 2$) darstellen? Auf der reellen Achse ($d = 1$) liefert jede ReLU-Einheit $a_j\,\mathrm{ReLU}(x - t_j)$ eine *Knickfunktion* mit Knoten $t_j$:

<!--en:7a38b314-->
- **Stückweise lineare Polygonzüge:** Einheit $j$ ist gleich $0$ für $x \le t_j$ und ändert die Gesamtsteigung um $a_j$ für $x > t_j$. Die Summe von $h$ Knickfunktionen und einem Ausgabe-Bias ergibt bei geeigneter Anfangssteigung *jeden* stetigen Polygonzug mit $h$ inneren Knoten ($h + 1$ lineare Abschnitte).

<!--en:a63f14d0-->
- **Gleichmäßige Approximation auf $[a, b]$:** Ein hinreichend feiner Polygonzug approximiert jede stetige Funktion auf einem abgeschlossenen, beschränkten Intervall mit einem Fehler unter jeder vorgegebenen Schranke $\epsilon > 0$. Daher genügt für jede stetige Funktion einer Variablen eine einzige, hinreichend breite ReLU-Schicht.

<!--en:1d80d496-->
::: {.slide title="Der universelle Approximationssatz und seine Grenzen"}

<!--en:f067cafb-->
[Ausdrucksstärke · Garantien und Grenzen]{.kicker}

<!--en:5d6319af-->
- **Universelle Approximation (Cybenko, 1989, Leshno et al., 1993):** Für jede stetige, nichtpolynomiale Aktivierungsfunktion $\phi$ und jede stetige Funktion $f$ auf einer kompakten Menge $K \subset \mathbb{R}^d$ gibt es ein MLP $g(\mathbf{x}) = \sum_{j=1}^h v_j\phi(\mathbf{w}_j^\top\mathbf{x} + b_j) + b^{(2)}$ mit 1 versteckten Schicht, für das $\sup_{\mathbf{x} \in K}|f(\mathbf{x}) - g(\mathbf{x})| < \epsilon$ gilt.

<!--en:14428f82-->
- **Exponentiell wachsende Breite:** Um oszillierende Funktionen im $\mathbb{R}^d$ mit einer einzigen versteckten Schicht zu approximieren, können $h = \Omega(\epsilon^{-d})$ Einheiten nötig sein.

<!--en:ef37f232-->
- **Keine Garantie für die Lernbarkeit:** Der Satz beweist, dass gute Gewichte *existieren*. Ob SGD sie findet und das Modell anhand endlicher Stichproben generalisiert, bleibt offen.

<!--en:99e8375d-->
::: {.slide title="Warum Tiefe zählt: Jede weitere Faltung vervielfacht die Anzahl der Abschnitte"}

<!--en:4cf6aebb-->
[Ausdrucksstärke · Effizienz durch Tiefe]{.kicker}

<!--en:4dbd856f-->
Jede zusätzliche Einheit in einer *einzigen* versteckten Schicht fügt einen Knick hinzu ($h + 1$ lineare Abschnitte). Aufeinanderfolgende Schichten *verketten* dagegen Abbildungen, die den Raum falten. Dadurch **vervielfacht** sich die Anzahl der linearen Abschnitte (Telgarsky, 2016):

<!--en:9e3085d6-->
- **Zeltabbildung mit 2 ReLU-Einheiten:** Auf $[0, 1]$ berechnen zwei ReLU-Einheiten die symmetrische Zeltfunktion:
  $$g(z) = 2\,\mathrm{ReLU}(z) - 4\,\mathrm{ReLU}(z - 1/2)$$

<!--en:34bd0548-->
- **Exponentielles Wachstum durch Faltung:** Jede Verkettung mit $g$ faltet $[0, 1]$ in der Mitte und verdoppelt die Anzahl der linearen Abschnitte. So hat $g^{\circ m}$ insgesamt $2^m$ lineare Abschnitte bei nur $m$ versteckten Schichten mit je $2$ Einheiten (insgesamt $2m$ Einheiten).

<!--en:811c3435-->
- **10 Schichten gegenüber 1 Schicht:** Für $m = 10$ erzeugt ein tiefes Netz mit $20$ ReLU-Einheiten $2^{10} = 1{,}024$ Abschnitte. Eine einzige versteckte Schicht benötigt dafür $2^{10} - 1 = 1{,}023$ Einheiten.

<!--en:46b23265-->
::: {.slide title="Aktivierungsfunktionen und Gradientenfluss"}

<!--en:59a66cec-->
[Aktivierungsfunktionen · Überblick]{.kicker}

<!--en:8c5f46ca-->
Für die universelle Approximation genügt jede stetige, nichtpolynomiale Aktivierungsfunktion $\phi$. Beim Training mit Gradientenabstieg kommt es dagegen auf die Ableitung $\phi'(z)$ an. Nach der Kettenregel skaliert sie den Gradienten jedes eingehenden Gewichts:

<!--en:83ad9e2c-->
- **Gleichrichtende Aktivierungen (oben):** ReLU hat für positive Präaktivierungen die Steigung $1$. Bei GELU liegt die Steigung nahe $1$, sobald $z \gtrsim 1$ gilt. So können Gradienten auch viele Schichten ohne nennenswerte Abschwächung durchlaufen.

<!--en:0eb4ee53-->
- **Begrenzende Aktivierungen (unten):** Die Sigmoidfunktion und $\tanh$ bilden $\mathbb{R}$ auf beschränkte Intervalle ab. Daher gehen ihre Ableitungen gegen null ($\phi'(z) \to 0$), sobald die Funktionen in den beiden Randbereichen sättigen ($|z| \gtrsim 3$).

<!--en:c948752f-->
::: {.slide title="Gleichrichtende Aktivierungen: ReLU, Leaky ReLU und GELU"}

<!--en:51bfab6e-->
[Aktivierungsfunktionen · Ohne Sättigung]{.kicker}

<!--en:0eb29c6f-->
Moderne tiefe Netze verwenden in versteckten Schichten meist gleichrichtende Nichtlinearitäten, weil diese für positive Eingaben nicht sättigen:

<!--en:4c46b30a-->
- **ReLU ($\mathrm{ReLU}(z) = \max(z, 0)$):** Die Ableitung beträgt $1$ für $z > 0$ und $0$ für $z < 0$ (vereinbarungsgemäß $0$ bei $z = 0$). Ein einziger Vergleich genügt zur Auswertung. Aktive Einheiten geben Gradienten ungeschwächt weiter.

<!--en:a3affb0c-->
- **Tote ReLU und Leaky ReLU:** Verschiebt ein großer Gradientenschritt die Gewichte einer Einheit so, dass $z_j < 0$ für *jedes* Trainingsbeispiel gilt, bleiben ihre Gewichtsgradienten wegen $\phi'(z_j) = 0$ dauerhaft null. *Leaky ReLU*, $\max(z, \alpha z)$ mit $\alpha \approx 0.01$, behält auf der negativen Halbachse eine kleine positive Steigung. So können inaktive Einheiten wieder aktiv werden.

<!--en:f2e3aa81-->
- **GELU ($z\,\Phi(z)$) und SiLU ($z\,\sigma(z)$):** Sie gewichten $z$ mit der Verteilungsfunktion $\Phi(z)$ der Standardnormalverteilung bzw. der logistischen Sigmoidfunktion $\sigma(z)$. Das glättet den Knick der ReLU bei $z = 0$ (mit $\mathrm{GELU}'(0) = 1/2$) und ergibt nahe $z \approx -0.75$ eine flache Senke im negativen Bereich. Beide sind in modernen Transformern üblich.

<!--en:78491279-->
::: {.slide title="Begrenzende Aktivierungen: Sigmoidfunktion, Tanh und Sättigung"}

<!--en:2555b79b-->
[Aktivierungsfunktionen · Beschränkte Gates]{.kicker}

<!--en:00deba3d-->
Klassische neuronale Netze verwendeten glatte S-Kurven, die $\mathbb{R}$ auf $(0, 1)$ oder $(-1, 1)$ abbilden. Über viele Schichten schwächen diese Funktionen Gradienten jedoch stark ab:

<!--en:b3822f9b-->
- **Logistische Sigmoidfunktion:** $\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$ hat die Ableitung $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 1/4$ (Maximum bei $z = 0$). Selbst nahe $z = 0$ bleibt nach $K = 10$ Sigmoidschichten höchstens der Anteil $4^{-10} \approx 10^{-6}$ des Gradienten erhalten.

<!--en:88fc9e3c-->
- **Hyperbolischer Tangens:** $\tanh(z) = \frac{1 - e^{-2z}}{1 + e^{-2z}} = 2\sigma(2z) - 1 \in (-1, 1)$ ist punktsymmetrisch zum Ursprung. Seine Ableitung $1 - \tanh^2(z) \le 1$ erreicht bei $z = 0$ ihr Maximum. So entfällt nahe dem Ursprung die Abschwächung auf $1/4$.

<!--en:c1458931-->
- **Sättigung bei $|z| = 4$:** Bei $z = \pm 4$ gelten $\sigma'(\pm 4) \approx 0.018$ und $\tanh'(\pm 4) \approx 0.0013$. Beide schwächen Gradienten um $>98\%$ ab. Heute dienen $\sigma$ und $\tanh$ vor allem als Wahrscheinlichkeitsausgaben oder multiplikative *Gates* in LSTMs und GRUs. In versteckten Schichten tiefer Netze sind sie kaum noch üblich.

<!--en:5f9e5edd-->
::: {.slide title="Quantitativer Vergleich der Ableitungen von Aktivierungsfunktionen"}

<!--en:3962e33b-->
[Aktivierungsfunktionen · Direkter Vergleich]{.kicker}

<!--en:c1c560ad-->
- **Positiver Randbereich ($z = +4$):** ReLU, Leaky ReLU und GELU haben hier eine Steigung von rund eins ($\phi'(+4) = 1.000$). Die Sigmoidfunktion ($0.018$) und $\tanh$ ($0.0013$) schwächen Gradienten dagegen um $>98\%$ ab.

<!--en:0a999595-->
- **Ursprung ($z = 0$):** Die Ableitung der Sigmoidfunktion erreicht hier ihr Maximum $\sigma'(0) = 0.25$ (Abschwächung um den Faktor $4\times$ pro Schicht, selbst ohne Sättigung). Außerdem gelten $\tanh'(0) = 1.00$ und $\mathrm{GELU}'(0) = 0.50$.

<!--en:675eeaad-->
- **Negativer Randbereich ($z = -4$):** Die Ableitung der ReLU beträgt $0$ (die Einheit ist tot, wenn $z < 0$ für alle Eingaben gilt). Bei Leaky ReLU erlaubt die Steigung $0.01$ eine Erholung.

<!--en:a448b9bb-->
::: {.slide title="Verständnisfragen zu Parametern, linearen Abschnitten und Gradienten"}

<!--en:ffcb46a8-->
[Selbsttest]{.kicker}

<!--en:cb0b984a-->
Drei kurze Rechenaufgaben zur Funktionsweise eines MLP:

<!--en:e9dbe500-->
1. **Parameteranzahl bei zwei versteckten Schichten:** Wie viele Parameter hat ein MLP für Fashion-MNIST ($d = 784, q = 10$) mit zwei versteckten Schichten der Breite $256$ ($784 \to 256 \to 256 \to 10$, mit Bias-Termen)? Was ändert sich, wenn alle Aktivierungsfunktionen $\phi$ entfallen?

<!--en:ea5b00c8-->
2. **Einfluss von Tiefe und Breite:** Wie viele lineare Abschnitte können $12$ ReLU-Einheiten bei eindimensionaler Ein- und Ausgabe erzeugen, wenn sie auf $1$ versteckte Schicht mit $12$ Einheiten oder auf $6$ versteckte Schichten mit je $2$ Einheiten verteilt sind?

<!--en:1a9521bd-->
3. **Produkt der Steigungen über 5 Schichten:** Wie viel vom ankommenden Gradienten bleibt in einer Kette aus $5$ versteckten Einheiten mit Einheitsgewichten ($w = 1$) erhalten, wenn jede Einheit die Sigmoidfunktion bei $z = 0$ bzw. die ReLU bei $z = 4$ verwendet?

<!--en:557b98b6-->
**Lösungen:**

<!--en:a352614e-->
1. **Parameter:** $785 \times 256 + 257 \times 256 + 257 \times 10 = 200{,}960 + 65{,}792 + 2{,}570 = 269{,}322$. Ohne $\phi$ reduziert sich das gesamte Netz auf eine einzige affine Abbildung der Dimension $784 \times 10$ ($7{,}850$ effektive Parameter).

<!--en:b14b19c9-->
2. **Lineare Abschnitte:** Eine Schicht mit $12$ Einheiten erzeugt höchstens $12 + 1 = 13$ Abschnitte. Mit $6$ Schichten zu je $2$ Einheiten ($g^{\circ 6}$) entstehen $2^6 = 64$ Abschnitte.

<!--en:3625fc60-->
3. **Verbleibender Gradient:** Bei der Sigmoidfunktion und $z = 0$ bleiben $(1/4)^5 = 1/1024 \approx 0.1\%$ erhalten, bei der ReLU und $z = 4$ sind es $1^5 = 100\%$.

<!--en:c8be603c-->
::: {.slide title="Zusammenfassung: mehrschichtige Perzeptrons und Aktivierungsfunktionen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:e933cf29-->
| Begriff oder Operation | Mathematische Form | Eigenschaft oder Komplexität | Rolle im Deep Learning |

<!--en:04c2e87e-->
| **MLP mit 1 versteckten Schicht** | $\mathbf{H} = \phi(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$, $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ | $(d + 1)h + (h + 1)q$ Parameter | Lernt nichtlineare Wechselwirkungen zwischen Merkmalen (z. B. XOR, Bildteile) |

<!--en:34ab9199-->
| **Zusammenfassung linearer Schichten ($\phi = \mathrm{id}$)** | $\mathbf{O} = \mathbf{X}(\mathbf{W}^{(1)}\mathbf{W}^{(2)}) + \mathbf{b}$ | Rang $\le \min(d, h, q)$ | Ohne Nichtlinearität $\phi$ lässt sich eine versteckte Schicht mit der nächsten zusammenfassen |

<!--en:a6318691-->
| **Universelle Approximation** | $\sum_{j=1}^h v_j\,\phi(\mathbf{w}_j^\top\mathbf{x} + b_j) + b^{(2)}$ | Approximiert jede stetige Funktion $f$ auf einer kompakten Menge $K \subset \mathbb{R}^d$ | Garantiert die Existenz für $K = 2$, doch die erforderliche Breite $h$ kann wie $\Omega(\epsilon^{-d})$ wachsen |

<!--en:bc7e9144-->
| **Raumfaltung durch Tiefe** | $g^{\circ m}(z)$ mit Zeltfunktion $g$ aus 2 ReLU-Einheiten | $2^m$ Abschnitte mit $2m$ Einheiten ($O(m)$ gegenüber $O(2^m)$) | Erklärt, warum tiefe Verkettungen exponentiell weniger Parameter benötigen können |

<!--en:6e74a524-->
| **ReLU / GELU** | $\max(z, 0)$ / $z\,\Phi(z)$ | $\phi'(z) = 1$ für $z \gg 0$, $\phi'(z) = 0$ für $z \ll 0$ | Übliche Aktivierungen versteckter Schichten, verhindern die Sättigung des Gradienten im positiven Randbereich |

<!--en:45b6af7f-->
| **Sigmoidfunktion / Tanh** | $\frac{1}{1+e^{-z}}$ / $2\sigma(2z) - 1$ | $\sigma'(0) = 1/4$, $\tanh'(0) = 1$ und $\phi'(\pm 4) < 0.02$ | Beschränkte Gates, binäre Wahrscheinlichkeiten und Begrenzung rekurrenter Zustände |
