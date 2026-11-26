<!--en:321a9f76-->
# Automatisches Differenzieren

<!--en:0648fb11-->
Im gesamten Grundlagenteil dieses Buches haben wir jeden Gradienten mit einem einzigen Aufruf des Frameworks zum automatischen Differenzieren berechnet. Im vorigen Abschnitt haben wir gesehen, was dieser Aufruf mathematisch berechnet. Die Ableitung eines Netzes ist ein Produkt von Jacobi-Matrizen, eine pro Schicht. Die Reihenfolge, in der wir diese Matrizen multiplizieren, bestimmt die Kosten. Ein echtes Programm besteht allerdings aus einer Folge von Additionen, Multiplikationen und Funktionsaufrufen. Wie gewinnt ein Framework daraus die einzelnen Matrixfaktoren, und wie multipliziert es sie in der günstigsten Reihenfolge?

<!--en:1631e647-->
An einem modernen Netz wird deutlich, wie viel Rechenzeit die günstige Reihenfolge spart. Betrachten wir ein Modell mit $10^8$ Parametern, das einen einzigen skalaren Verlust $\ell$ ausgibt. Jeder Trainingsschritt benötigt die Ableitung von $\ell$ nach allen $10^8$ Parametern. Bei der direkten Berechnung mit endlichen Differenzen ändern wir jeweils einen Parameter geringfügig, werten das Netz erneut aus und teilen die Änderung von $\ell$ durch die Parameteränderung. Das erfordert $10^8$ Durchläufe pro Schritt. Bei 10 ms pro Durchlauf würde ein einziger Gradient $10^6$ s kosten, also fast zwölf Tage. Hinzu kämen Abbruch- und Rundungsfehler, deren Größe von der Schrittweite abhängt. Die Backpropagation liefert alle $10^8$ Ableitungen bis auf Gleitkommarundung exakt und kostet etwa so viel wie drei Vorwärtsdurchläufe, also rund 30 ms. In diesem Abschnitt klären wir, wie sich der Faktor $10^8$ einsparen lässt.

<!--en:ed16a9dd-->
Die Grundidee ist überraschend einfach. Ein Produkt von Jacobi-Matrizen lässt sich von beiden Enden her auswerten. Beginnen wir bei der Eingabe, so verfolgen wir pro Durchlauf eine Eingaberichtung vorwärts. Für $10^8$ Eingaben brauchen wir also $10^8$ Durchläufe. Beginnen wir beim Verlust, so verfolgen wir pro Durchlauf eine Ausgaberichtung rückwärts. Ein skalarer Verlust hat nur eine Ausgabe.

<!--en:625eb165-->
In :numref:`sec_autograd` haben wir gesehen, wie wir beide Modi in einem Deep-Learning-Framework aufrufen. Jetzt untersuchen wir, wie sie funktionieren. Wir implementieren den Vorwärtsmodus und den Rückwärtsmodus in wenigen Dutzend Zeilen Python und beweisen, dass sie die tatsächlichen Ableitungen liefern. Wir rechnen einen Rückwärtsdurchlauf von Hand durch und messen die Kosten beider Modi. Anschließend kombinieren wir die Modi, um die Krümmung zweiter Ordnung entlang einer Richtung zu berechnen und die Lösung einer Gleichung abzuleiten, ohne die Schritte des Lösers zu differenzieren. Zwei Einsichten sind besonders wichtig. Erstens liefert ein Programm automatisch Ableitungen, wenn wir es mit *dualen Zahlen* auswerten. Das beweisen wir in der einzigen Proposition dieses Abschnitts. Zweitens kostet jeder Durchlauf nur ein kleines Vielfaches einer Programmauswertung. Der Vorwärtsmodus braucht einen Durchlauf pro Eingaberichtung, der Rückwärtsmodus einen pro Ausgaberichtung. Bei der ersten Lektüre können Sie :numref:`subsec_mdl-composing-products` überspringen und später darauf zurückkommen.

<!--en:1f40d413-->
Drei kleine Beispiele begleiten uns durch den Abschnitt. Das erste ist die Log-Sum-Exp-Funktion $\ell(\mathbf x)=\log\sum_j e^{x_j}$ aus diesem Kapitel. Ihr Gradient ist die Softmax-Funktion. Das zweite ist die Abbildung der Ebene $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ aus dem vorigen Abschnitt. Das dritte ist der Polynomausdruck $y=(uv+u)^2$, der sich vollständig von Hand durchrechnen lässt.

<!--en:857f9054-->
Für den gesamten Abschnitt verwenden wir die folgenden Importe.

<!--en:7286bb23-->
## Vorwärtsmodus und duale Zahlen

<!--en:12712f02-->
### Duale Zahlen

<!--en:c9c7c82f-->
Wir möchten in einem einzigen Durchlauf sowohl den Wert eines Programms als auch seine Ableitung erhalten, ohne eine Schrittweite abzustimmen oder vorab eine symbolische Formel für $f'$ anzugeben. Dafür genügt eine kleine Erweiterung der reellen Zahlen. Wir fügen ein neues Symbol $\varepsilon$ hinzu, das von null verschieden ist, dessen Quadrat aber verschwindet:

<!--en:227df8ba-->
Mit dieser einen zusätzlichen Regel rechnen wir nach den üblichen Regeln der Algebra mit Ausdrücken der Form $a+b\varepsilon$, den *dualen Zahlen*. Bei den Ableitungsregeln in :numref:`sec_mdl-single_variable_calculus` fallen Terme zweiter Ordnung weg. Indem wir $\varepsilon^2=0$ setzen, bauen wir diesen Schritt direkt in die Arithmetik ein.

<!--en:0b154256-->
Wenden wir diese Regel auf $3+\varepsilon$ an. Für die zweite und dritte Potenz erhalten wir

<!--en:fb75e3df-->
Betrachten wir die Koeffizienten von $\varepsilon$. Die Werte $6$ und $27$ sind die Ableitungen von $x^2$ und $x^3$ an der Stelle $x=3$. Dafür haben wir keine Ableitungsregel angewandt. Die Ableitungen ergeben sich direkt aus der Multiplikation. Das gilt für jedes Produkt. Multiplizieren wir zwei duale Zahlen und streichen den Term mit $\varepsilon^2$, erhalten wir

<!--en:197effbc-->
Sind $b$ und $d$ die Ableitungen zweier Größen mit den Werten $a$ und $c$, so ist $ad+bc$ die Ableitung ihres Produkts. Die Multiplikation dualer Zahlen setzt also die Produktregel um.

<!--en:37da98d4-->
Hier könnte man einwenden, dass eine von null verschiedene Zahl unmöglich das Quadrat null haben kann. Für reelle Zahlen trifft das zu, für Matrizen jedoch nicht. Betrachten wir $\varepsilon=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$. Dann gilt $\varepsilon^2=\mathbf 0$. Die duale Zahl $a+b\varepsilon$ entspricht der $2\times2$-Matrix $\left(\begin{smallmatrix}a&b\\0&a\end{smallmatrix}\right)$, und die gewöhnliche Matrizenmultiplikation $\left(\begin{smallmatrix}a&b\\0&a\end{smallmatrix}\right)\left(\begin{smallmatrix}c&d\\0&c\end{smallmatrix}\right)=\left(\begin{smallmatrix}ac&ad+bc\\0&ac\end{smallmatrix}\right)$ ergibt genau :eqref:`eq_mdl-dual-mul`. Duale Zahlen sind so konkret wie $2\times2$-Matrizen, ähnlich wie komplexe Zahlen so konkret sind wie Drehungen und Skalierungen der Ebene.

<!--en:741a785b-->
Programme rufen auch elementare Funktionen wie $\sin$, $\exp$ und $\log$ auf. Sei eine Funktion $\varphi$ durch ihre Taylor-Reihe um $a$ gegeben. Setzen wir $a+b\varepsilon$ in die Reihe ein, verschwinden alle Terme ab $(b\varepsilon)^2$. Übrig bleibt die lineare Näherung:

<!--en:ca831bcd-->
Damit folgen $e^{a+b\varepsilon}=e^a+e^a b\,\varepsilon$ und $\sin(a+b\varepsilon)=\sin a+(\cos a)\,b\,\varepsilon$ direkt aus der Algebra. Für eine Grundfunktion, die an der Stelle $a$ nur differenzierbar und nicht analytisch ist, definiert die obige Formel ihre Wirkung auf duale Zahlen. In beiden Fällen folgt der $\varepsilon$-Anteil der Kettenregel. Die eingehende Ableitung $b$ des Arguments wird mit der lokalen Ableitung $\varphi'(a)$ multipliziert.

<!--en:6b5f4848-->
**Proposition (Duale Zahlen liefern Ableitungen).** *Sei $f$ ein Ausdruck aus Konstanten, der Identität, Summen, Produkten und Grundfunktionen $\varphi$ (etwa $\sin$, $\exp$, $\log$). Jede Grundfunktion sei an dem Argument differenzierbar, das sie bei der Auswertung von $f$ an der Stelle $x$ erhält (für $\log$ also an einem positiven Argument). Werten wir denselben Ausdruck an der dualen Zahl $x+\varepsilon$ aus, so gilt*

<!--en:e49bb84e-->
**Beweis.** Wir führen eine strukturelle Induktion über den Ausdruck. Eine Konstante $c$ stellen wir als $c+0\varepsilon$ dar, die Identität als $x+1\varepsilon$. Ihre $\varepsilon$-Anteile sind also ihre Ableitungen $0$ und $1$. Nun gelte für zwei Teilausdrücke $g(x+\varepsilon)=g(x)+g'(x)\varepsilon$ und $h(x+\varepsilon)=h(x)+h'(x)\varepsilon$. Ihre Summe hat den $\varepsilon$-Anteil $g'+h'=(g+h)'$. Nach :eqref:`eq_mdl-dual-mul` hat ihr Produkt den $\varepsilon$-Anteil $g'h+gh'=(gh)'$. Wenden wir schließlich eine Grundfunktion $\varphi$ auf $g$ an, ergibt die obige Regel mit $a=g(x)$ und $b=g'(x)$ den $\varepsilon$-Anteil $\varphi'(g(x))\,g'(x)=(\varphi\circ g)'(x)$. Die Eigenschaft bleibt bei jeder möglichen Verknüpfung von Teilausdrücken erhalten. Also erfüllt der gesamte Ausdruck :eqref:`eq_mdl-dual-eval`. $\blacksquare$

<!--en:6948668f-->
Um ein Programm zu differenzieren, führen wir es also mit $x+\varepsilon$ aus und lesen den $\varepsilon$-Anteil der Ausgabe ab. Das funktioniert auch bei Schleifen und Verzweigungen automatisch. Für jede feste Eingabe führt das Programm eine konkrete Folge elementarer Operationen aus. Diese bildet einen Ausdruck der beschriebenen Form, und der Durchlauf mit dualen Zahlen differenziert genau diesen Ausführungspfad (:numref:`subsec_autograd_control_flow` zeigt das für die Frameworks). Die Ableitung des ausgeführten Pfads stimmt allerdings nur dann garantiert mit $f'(x)$ überein, wenn alle benachbarten Eingaben dieselben Zweige durchlaufen. Ein Programm, das `x` zurückgibt, wenn `x != 0` gilt, und sonst `0.0`, berechnet die Identität. An der Stelle $x=0$ führt es jedoch den konstanten Zweig aus. Die Rechnung mit dualen Zahlen liefert die Steigung $0$. Diesem Problem sind wir bei $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)$ in :numref:`subsec_mdl-nonsmooth-training` begegnet. Den $\varepsilon$-Anteil der Eingabe, hier $1$, nennen wir *Startwert*. Wir konstruieren keinen symbolischen Ausdruck für $f'$ und brauchen keine Schrittweite. Der einzige Fehler entsteht durch Gleitkommarundung.

<!--en:eacd806e-->
### Implementierung des Vorwärtsmodus

<!--en:7d348b44-->
Durch Überladen von Operatoren lässt sich diese Algebra direkt in Code übersetzen. Die folgende Klasse `Dual` speichert einen Wert $a$ zusammen mit seinem Ableitungsanteil $b$. Diesen Anteil nennen wir *Tangente* und schreiben dafür $\dot v$, wenn die zugehörige Größe $v$ heißt. Die Klasse implementiert :eqref:`eq_mdl-dual-mul` für `*` und verwendet für `sin`, `exp` und `log` die Regel für Grundfunktionen. Der Klassenzähler `ops` zählt die ausgeführten elementaren Operationen. Damit messen wir in :numref:`subsec_mdl-cost-asymmetry` die Kosten.

<!--en:574827a6-->
Testen wir `Dual` an der Funktion $f(x)=\sin(x^2)+e^x$ bei $x=1.3$. Wir übergeben $1.3+\varepsilon$ und vergleichen den $\varepsilon$-Anteil des Ergebnisses mit der von Hand berechneten Ableitung $f'(x)=2x\cos(x^2)+e^x$:

<!--en:a240c9e2-->
Ein einziger Durchlauf liefert den Funktionswert $4.6622$ und die Ableitung $3.360101$. Beide stimmen in allen ausgegebenen Stellen mit der analytischen Formel überein.

<!--en:90a10706-->
An einer einzigen Variablen wird die wichtigste Einschränkung des Vorwärtsmodus allerdings nicht sichtbar. Betrachten wir Log-Sum-Exp für die dreidimensionale Eingabe $\mathbf x=(1,-0.5,2)$. Der Gradient ist die Softmax-Funktion $\partial\ell/\partial x_j=e^{x_j}/\sum_k e^{x_k}=p_j$. Wie viele Durchläufe braucht der Vorwärtsmodus für den vollständigen Gradienten? Geben wir der Eingabe $j$ den Startwert $1$ und allen übrigen Eingaben den Startwert $0$, erhalten wir nur die partielle Ableitung $\partial\ell/\partial x_j$ als $\varepsilon$-Anteil der Ausgabe.

<!--en:4f9c4181-->
Wir brauchen drei Durchläufe. Jeder davon berechnet erneut denselben Funktionswert $2.371539$ und liefert nur einen Eintrag der Softmax-Funktion: $0.253716$, $0.056612$ oder $0.689672$. Allgemein erfordert der Gradient für $n$ Eingaben $n$ Durchläufe. Für das Netz mit $10^8$ Parametern aus der Einleitung sind wir damit wieder bei $10^8$ Programmdurchläufen. Im Vorwärtsmodus entfällt die Schrittweite der endlichen Differenzen, die Zahl der Durchläufe bleibt jedoch gleich.

<!--en:63962663-->
### Vektorwertige Tangenten und Spalten der Jacobi-Matrix

<!--en:7c11e887-->
Betrachten wir nun eine vektorwertige Abbildung $\mathbf f:\mathbb R^n\to\mathbb R^m$ mit der $m\times n$-Jacobi-Matrix $\mathbf J$. Was berechnet ein einzelner Vorwärtsdurchlauf? Als Startwerte wählen wir die Komponenten einer beliebigen Richtung $\mathbf v$. Dazu setzen wir die Eingabe $j$ auf $x_j+v_j\varepsilon$. Wenden wir unsere Proposition auf den eindimensionalen Schnitt $t\mapsto f_i(\mathbf x+t\mathbf v)$ an, erhalten wir für die Ausgabe $i$ die duale Zahl $f_i(\mathbf x)+(\nabla f_i\cdot\mathbf v)\,\varepsilon$. Die $\varepsilon$-Anteile aller $m$ Ausgaben bilden also den Vektor $\mathbf J\mathbf v$, das *Jacobi-Vektor-Produkt* (JVP). Ist $\mathbf v$ der $j$-te Standardbasisvektor $\mathbf e_j$, liefert das JVP die Spalte $j$ von $\mathbf J$. Ein Vorwärtsdurchlauf verfolgt eine Eingaberichtung durch die gesamte Abbildung.

<!--en:cc2575ea-->
Überprüfen wir das an der Abbildung der Ebene $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ an der Stelle $(0.5,0.5)$. In Zeile $i$ ihrer Jacobi-Matrix stehen die partiellen Ableitungen der Ausgabe $i$. Die Matrix lautet

<!--en:7c90f53d-->
Zwei Vorwärtsdurchläufe mit den Startvektoren $\mathbf e_1$ und $\mathbf e_2$ liefern nacheinander die beiden Spalten.

<!--en:df092509-->
Die Tangente in `Dual` muss kein Skalar sein. Weisen wir $x$ die vektorwertige Tangente $(1,0)$ zu und $y$ die vektorwertige Tangente $(0,1)$, gibt jede Operation beide Eingaberichtungen gleichzeitig weiter. Die Tangente der Ausgabe $i$ ist dann die Zeile $i$ von $\mathbf J$. Lässt sich so die gesamte Matrix in einem Durchlauf berechnen?

<!--en:5da44a95-->
Ja, doch die Ersparnis ist nur scheinbar. Jede Tangentenoperation verarbeitet nun Arrays der Länge $n$. Der einzelne Durchlauf erfordert damit etwa $n$-mal so viele Rechenoperationen wie ein skalarer Durchlauf. Die Zahl der Durchläufe allein sagt also wenig über den Rechenaufwand aus. In :numref:`subsec_mdl-cost-asymmetry` messen wir diesen Aufwand direkt. Bei einem skalaren Verlust mit $10^8$ Eingaben braucht der Vorwärtsmodus entweder $10^8$ skalare Durchläufe oder einen Durchlauf mit Tangenten der Länge $10^8$. In beiden Fällen wächst der Gesamtaufwand wie der von $n$ Programmdurchläufen.

<!--en:bc687f26-->
## Rückwärtsmodus und Tape

<!--en:85c480ff-->
### Zwei Klammerungen desselben Produkts

<!--en:3c0d36da-->
Um den Faktor $n$ einzusparen, müssen wir die Ableitungen in umgekehrter Reihenfolge miteinander multiplizieren. Wie wir in :numref:`sec_mdl-matrix-calculus` gesehen haben, ergibt sich die Jacobi-Matrix eines Programms aus $K$ Schritten, $\mathbf f=\mathbf f_K\circ\cdots\circ\mathbf f_1$, als Produkt der Jacobi-Matrizen der einzelnen Schritte:

<!--en:eaba6426-->
Die Klammerung dieses Produkts bestimmt die Kosten. Betrachten wir drei Schichten der Breite $1000$ und multiplizieren das Produkt von links mit einem Zeilenvektor. Von links geklammert braucht die Rechnung $3\times10^6$ Multiplikationen, von rechts geklammert dagegen $2\times10^9$ (:numref:`fig_mdl-cal-jacobian-chain-order`). Beim automatischen Differenzieren bauen wir diese Jacobi-Matrizen nie explizit auf. Stattdessen geben wir einen Startvektor von Schritt zu Schritt weiter. Der Vorwärtsmodus beginnt die Multiplikation von rechts mit einem Richtungsvektor $\mathbf v$ in Spaltenform, der Rückwärtsmodus von links mit einem Zeilenvektor $\mathbf u^\top$:

<!--en:3e2a583c-->
Da der Startvektor von Anfang an beteiligt ist, genügt bei jedem Schritt ein Matrix-Vektor-Produkt. Ein Produkt zweier Matrizen ist nie nötig. Im linken Ausdruck läuft eine Eingaberichtung vorwärts und liefert das JVP $\mathbf J\mathbf v$. Im rechten Ausdruck läuft eine Ausgaberichtung rückwärts und liefert das *Vektor-Jacobi-Produkt* (VJP) $\mathbf u^\top\mathbf J$. Es ist eine Linearkombination der Zeilen von $\mathbf J$ mit den Gewichten aus $\mathbf u$. Setzen wir $\mathbf u=\mathbf e_i$, erhalten wir die Zeile $i$ von $\mathbf J$.

<!--en:5f2e9d3f-->
Aus dieser Symmetrie folgt die grundlegende Kostenregel für das automatische Differenzieren. Der Vorwärtsmodus braucht einen Durchlauf pro Eingaberichtung, der Rückwärtsmodus einen pro Ausgaberichtung. Jeder Durchlauf kostet nur ein kleines konstantes Vielfaches einer Programmauswertung. Für Programme aus den Operationen $+,-,\times,\div$ haben Baur und Strassen eine obere Schranke bewiesen. Alle partiellen Ableitungen zusammen kosten höchstens ein festes Vielfaches der Operationen des Programms. Je nach Zählweise liegt der Faktor zwischen drei und fünf :cite:`Baur.Strassen.1983`. Ein skalarer Verlust hat eine $1\times n$-Jacobi-Matrix, also eine einzige Zeile. Ein Rückwärtsdurchlauf mit dem Startwert $1$ liefert daher den gesamten Gradienten auf einmal. :numref:`fig_mdl-cal-jacobian-shapes` veranschaulicht diese Regel anhand der Matrixformen.

<!--en:2081d8d7-->
![Die Form der Jacobi-Matrix bestimmt den günstigeren Modus. Ein skalarer Verlust hat eine $1\times n$-Jacobi-Matrix: eine Zeile, ein VJP, ein Rückwärtsdurchlauf. Eine Abbildung mit einer Eingabe hat eine $m\times 1$-Jacobi-Matrix: eine Spalte, ein JVP, ein Vorwärtsdurchlauf. Die vollständige $m\times n$-Jacobi-Matrix kostet im günstigeren Modus $\min(m,n)$ Durchläufe.](../img/mdl-cal-jacobian-shapes.svg)

<!--en:256bd843-->
Beide Modi werten dasselbe Matrixprodukt von entgegengesetzten Enden her aus. Welches Ende günstiger ist, hängt allein von der Form von $\mathbf J$ ab. Wir wählen den Modus, dessen Anzahl an Startvektoren der kürzeren Matrixdimension entspricht. Beim Verlust eines neuronalen Netzes mit einer skalaren Ausgabe und Millionen oder Milliarden von Parametern unterscheiden sich die beiden Dimensionen besonders stark.

<!--en:4d903979-->
### Das Tape

<!--en:f2777096-->
Vorwärtsmodus und Rückwärtsmodus geben grundverschiedene Größen weiter. Betrachten wir ein Programm mit Eingabe $x$ und Ausgabe $y$. Die Tangente $\dot v$ einer Zwischenvariablen $v$ misst, wie schnell sich $v$ ändert, wenn wir die Eingabe in Richtung $\dot x$ bewegen. Die *adjungierte Variable* $\bar v$ aus :numref:`sec_mdl-chain-rule-graphs` gibt dagegen die totale Ableitung der endgültigen Ausgabe $y$ nach $v$ an. In sie gehen alle Pfade von $v$ nach $y$ ein. Eine Tangente hängt nur von den Schritten *vor* $v$ ab, eine adjungierte Variable von den Schritten *nach* $v$. Für $y=v^2$ mit $v=3x$ erhalten wir zum Beispiel $\dot v=3\dot x$ von unten und $\bar v=2v$ von oben.

<!--en:96343d67-->
Diese Asymmetrie bestimmt die Ausführung der beiden Modi (:numref:`fig_mdl-cal-fwd-vs-rev`). Tangenten durchlaufen das Programm in derselben Richtung wie die Werte. Der Vorwärtsmodus kommt deshalb mit einem einzigen Durchlauf zusammen mit der Programmauswertung aus. `Dual` muss sich keine früheren Zwischenwerte merken. Adjungierte Variablen laufen dagegen entgegen der Ausführungsrichtung. Für die adjungierte Variable eines Knotens brauchen wir die adjungierten Variablen aller Knoten, die seinen Wert verwenden. Sie stehen erst zur Verfügung, nachdem das Programm seine Ausgabe erreicht hat. Der Rückwärtsmodus arbeitet deshalb in zwei Phasen. Zuerst führt er das Programm vorwärts aus und zeichnet jede elementare Operation auf. Dann arbeitet er die Aufzeichnung rückwärts ab. Diese Aufzeichnung heißt *Tape* oder Wengert-Liste, nach der Liste elementarer Operationen, die :citet:`Wengert.1964` für den Vorwärtsmodus einführte. Durch Umkehren der Liste erhalten wir den von :citet:`Linnainmaa.1970` eingeführten Rückwärtsmodus.

<!--en:a626b35d-->
![Vorwärts- und Rückwärtsmodus auf einer Kette $x\to a\to b\to\ell$. Beide werten die Kette von links nach rechts aus (grau). Der Vorwärtsmodus (oben) gibt die Tangenten $\dot v$ zusammen mit den Werten weiter, eine Eingaberichtung pro Durchlauf. Der Rückwärtsmodus (unten) wertet zuerst die Kette aus. Dann gibt er die adjungierten Variablen $\bar v$, die totalen Ableitungen von $\ell$ nach $v$, vom Startwert $\bar\ell=1$ aus rückwärts weiter. Das ergibt eine Ausgaberichtung pro Durchlauf, bei einem skalaren Verlust also den gesamten Gradienten.](../img/mdl-cal-fwd-vs-rev.svg)

<!--en:6767a7ab-->
Für eine funktionsfähige Implementierung des Rückwärtsmodus genügen etwa vierzig Zeilen Python. Jeder Knoten auf dem Tape speichert seinen Vorwärtswert, seine direkten Vorgänger und eine kleine Rückwärtsfunktion als Closure. Diese erhält die adjungierte Variable $g$ des Knotens und addiert seinen Beitrag zur adjungierten Variablen jedes direkten Vorgängers. Für $c=a+b$ addiert sie $g$ bei beiden Vorgängern. Für $c=ab$ addiert sie $b\,g$ zu $\bar a$ und $a\,g$ zu $\bar b$. Die Funktion `backprop` sortiert den Graphen topologisch, sodass jeder Knoten nach seinen direkten Vorgängern erscheint. Sie setzt die adjungierte Variable der Ausgabe auf $1$ und ruft die Closures in umgekehrter Reihenfolge auf. Wie `Dual` führt die Klasse einen Zähler `ops`, der sowohl Vorwärts- als auch Rückwärtsschritte erfasst.

<!--en:02af19cd-->
Die Klasse `Var` implementiert nur `+` und `*`. Das genügt bereits für unser drittes Beispiel $y=(uv+u)^2$. Rechnen wir vor dem Aufruf des Codes den gesamten Rückwärtsdurchlauf für $u=2$ und $v=-3$ von Hand durch. Der Vorwärtsdurchlauf zeichnet drei elementare Operationen auf: $t=uv=-6$, dann $r=t+u=-4$ und schließlich $y=r\cdot r=16$. Der Rückwärtsdurchlauf besucht die Knoten in einer umgekehrten topologischen Reihenfolge wie $y,r,t,u,v$. Damit kommt jeder Knoten erst nach allen Knoten an die Reihe, die von ihm abhängen (`backprop` besucht $v$ vor $u$, was genauso zulässig ist). Jede adjungierte Variable ist die Summe der Beiträge über die ausgehenden Kanten. Jeder Beitrag ist das Produkt aus der adjungierten Variablen des jeweiligen direkten Nachfolgers und der lokalen Ableitung:

<!--en:c7df6e3e-->
| Knoten | Ausgehende Kanten und ihre Beiträge | Adjungierte Variable |

<!--en:c12abaa9-->
| $y$ | keine, daher der Startwert | $\bar y=1$ |

<!--en:bc60dad5-->
| $r$ | beide Argumente von $y=r\cdot r$: $\bar y\,r+\bar y\,r$ | $\bar r=-8$ |

<!--en:5029bc4a-->
| $u$ | $r=t+u$ und $t=uv$: $\bar r\cdot1+\bar t\cdot v$ | $\bar u=-8+24=16$ |

<!--en:965a9a41-->
Direktes Ableiten von $y=(uv+u)^2$ bestätigt beide Ergebnisse: $\partial y/\partial u=2(uv+u)(v+1)=2(-4)(-2)=16$ und $\partial y/\partial v=2(uv+u)\,u=2(-4)(2)=-16$. An zwei Stellen im Graphen ist das Aufsummieren entscheidend. Der Knoten $r$ liefert beide Argumente des letzten Produkts, und $u$ geht sowohl in $t$ als auch in $r$ ein (:numref:`fig_mdl-cal-tape-dag`). Würden wir die Werte überschreiben, statt sie zu addieren, erhielten wir $\bar r=-4$. Selbst bei korrektem $\bar r$ hätte $\bar u$ entweder den Wert $-8$ oder $24$, je nachdem, welche Closure zuletzt ausgeführt wird. Die adjungierte Variable summiert über alle ausgehenden Kanten, wie es die Pfadsumme der Kettenregel verlangt. Deshalb muss jede Rückwärtsfunktion mit `+=` aufsummieren.

<!--en:e64de405-->
![Das Tape für $y=(uv+u)^2$ bei $u=2$ und $v=-3$. Jedes Kästchen zeigt den Vorwärtswert des Knotens. In Orange steht seine adjungierte Variable, die Ableitung von $y$ nach diesem Knoten. Der Graph bildet eine Raute statt einer Kette. Der Knoten $u$ geht sowohl in $t=uv$ als auch in $r=t+u$ ein, und $r$ liefert beide Argumente von $y=r\cdot r$. Daher setzen sich die adjungierten Variablen von $r$ und $u$ jeweils aus zwei Beiträgen zusammen.](../img/mdl-cal-tape-dag.svg)

<!--en:ab1c3beb-->
Zeichnen wir denselben Ausdruck mit `Var` auf und führen einen einzigen Rückwärtsdurchlauf aus, erhalten wir:

<!--en:20528c20-->
Die Werte stimmen mit $\bar u$ und $\bar v$ aus unserer Rechnung von Hand überein. Ein Deep-Learning-Framework für den praktischen Einsatz verwendet denselben Algorithmus auf Tensorebene und liefert dieselben beiden Ableitungen:

<!--en:78224e75-->
In der Praxis unterscheiden sich Frameworks in drei Punkten von unserer kleinen Implementierung mit `Var`. Der zugrunde liegende Algorithmus bleibt dabei gleich. Erstens enthalten die Knoten ganze Tensoren statt Skalare. Jede Rückwärtsfunktion implementiert also das VJP einer ganzen Tensoroperation, etwa $\bar{\mathbf X}=\bar{\mathbf Y}\mathbf W^\top$ und $\bar{\mathbf W}=\mathbf X^\top\bar{\mathbf Y}$ (:numref:`subsec_mdl-layer-vjp`) für die Matrizenmultiplikation. Zweitens ist die Bibliothek elementarer Operationen viel größer. Jede hat eine von Hand optimierte VJP-Regel. Drittens bauen Frameworks den Berechnungsgraphen auf eine von zwei Arten auf. Ein *dynamisches Tape* zeichnet Operationen während der Programmausführung auf, genau wie `Var`. Ein *Tracer* führt das Programm dagegen einmal mit abstrakten Platzhaltern als Eingaben aus und erfasst die Folge der Operationen. Er bildet für jeden Schritt die linearisierte Regel des Vorwärtsmodus, transponiert die einzelnen Schritte und kehrt ihre Reihenfolge um. Beide Verfahren liefern denselben Rückwärtsdurchlauf. Beim Deep Learning heißt dieser Durchlauf im Rückwärtsmodus für ein neuronales Netz *Backpropagation* :cite:`Rumelhart.Hinton.Williams.ea.1988`.

<!--en:7e8a283f-->
PyTorch zeichnet die Operationen während der Ausführung auf. Jede Operation auf einem Tensor mit `requires_grad=True` fügt dem Graphen einen Knoten hinzu, und `backward()` arbeitet diese Knoten in umgekehrter Reihenfolge ab.

<!--en:a7dafd20-->
MXNet zeichnet die Operationen innerhalb von `autograd.record()` während der Ausführung auf. Anschließend arbeitet `backward()` den aufgezeichneten Graphen rückwärts ab.

<!--en:f1393209-->
TensorFlow zeichnet die Operationen innerhalb eines `GradientTape`-Kontexts während der Ausführung auf. Anschließend arbeitet `tape.gradient` das Tape rückwärts ab.

<!--en:96ec8a30-->
JAX verwendet einen Tracer. Die Funktion `jax.grad` erfasst die Operationen der Funktion und gibt eine neue Python-Funktion zurück, die den transponierten Rückwärtsdurchlauf ausführt.

<!--en:3e2fc1b5-->
Auch für die Frameworks gilt die Voraussetzung unserer Proposition über duale Zahlen. Jede Grundfunktion muss an ihrem jeweiligen Argument differenzierbar sein. In der Praxis scheitert diese Voraussetzung am häufigsten beim Maskieren. Angenommen, wir summieren $\log x_i$ über die ursprünglichen Einträge eines durch Padding verlängerten Vektors und setzen die aufgefüllten Einträge mit `where` auf null. Da `where` zwischen zwei bereits berechneten Werten wählt, wertet das Programm $\log$ auch an den aufgefüllten Stellen aus, hier bei $x_2=0$. Im Vorwärtsdurchlauf stört das nicht. Der Wert $-\infty$ wird verworfen, der Verlust beträgt $\log 2\approx0.6931$. Im Rückwärtsdurchlauf gibt `where` jedoch die adjungierte Variable $0$ an den verworfenen Zweig weiter. Das VJP von $\log$ multipliziert sie mit der lokalen Ableitung $1/0=\infty$. In der Gleitkommaarithmetik ergibt $0\cdot\infty$ statt $0$ den Wert NaN.

<!--en:8de4e867-->
Die erste Zeile liefert den Gradienten $(0.5, \textrm{NaN})$, obwohl der Verlust überhaupt nicht vom aufgefüllten Eintrag abhängt und die tatsächliche Ableitung dort $0$ ist. Ein einziges NaN bleibt nicht auf diese Stelle beschränkt. Es erreicht jeden Parameter, von dem $x_2$ abhängt. Die zweite Zeile zeigt die übliche Abhilfe, das *doppelte where*. Ein inneres `where` ersetzt die maskierten Einträge vor Auswertung der Grundfunktion durch einen unproblematischen Wert, hier $1$. So entsteht keine unendliche lokale Ableitung, und der Gradient ist $(0.5, 0)$. Ebenso kann man das Argument so begrenzen, dass es vom Rand des Definitionsbereichs entfernt bleibt. Dasselbe Problem tritt bei $\sqrt{x}$ an der Stelle $0$ auf und damit auch bei einer als $\sqrt{\sum_i x_i^2}$ berechneten Norm an der Stelle $\mathbf x=\mathbf 0$.

<!--en:ed05ecae-->
### Parametergradienten

<!--en:97c95277-->
Bisher haben wir mit dem Produkt der Jacobi-Matrizen in :eqref:`eq_mdl-ad-jacobian-product` die Ableitungen nach der Eingabe des Netzes verfolgt. Für das Training brauchen wir jedoch die Ableitungen nach seinen Gewichten. Die Gewichtsgradienten ergeben sich im selben Rückwärtsdurchlauf aus der VJP-Regel für die Schicht (:numref:`subsec_mdl-layer-vjp`). Für eine lineare Schicht $\mathbf h_k=\mathbf W_k\mathbf h_{k-1}$ mit der Gewichtsmatrix $\mathbf W_k$ der Form $d_k\times d_{k-1}$ lautet diese Regel

<!--en:c4e74c98-->
also das äußere Produkt aus der von oben eintreffenden adjungierten Variablen und der gespeicherten Aktivierung von unten. Anschaulich wächst der Einfluss eines Gewichts proportional zu zwei Größen: dem Einfluss seiner Ausgabe auf den Verlust und der Größe seiner Eingabeaktivierung. Für unsere Kostenanalyse interessiert uns, wie viele Rechenoperationen diese Regel erfordert.

<!--en:a6a36849-->
Das äußere Produkt $\bar{\mathbf h}_k\,\mathbf h_{k-1}^\top$ erfordert $d_kd_{k-1}$ Multiplikationen, genauso viele wie das Matrix-Vektor-Produkt $\mathbf W_k\mathbf h_{k-1}$ im Vorwärtsdurchlauf. Die Weitergabe der adjungierten Variablen an die vorherige Schicht gemäß $\bar{\mathbf h}_{k-1}^\top=\bar{\mathbf h}_k^\top\mathbf W_k$ kostet weitere $d_kd_{k-1}$ Multiplikationen. Gemessen an den Multiplikationen kostet der Rückwärtsdurchlauf also etwa doppelt so viel wie ein Vorwärtsdurchlauf. Zusammen kosten beide etwa so viel wie drei Vorwärtsdurchläufe, entsprechend den 30 ms aus der Einleitung. Der Vorwärtsmodus müsste dagegen jedes einzelne Gewicht als eigene Eingaberichtung behandeln. Schon für eine einzige Schicht wären $d_kd_{k-1}$ Durchläufe nötig. Deshalb trainiert kein Framework neuronale Netze im Vorwärtsmodus.

<!--en:5f7e7738-->
### Kosten der beiden Modi

<!--en:8998dd7d-->
Da wir beide Modi selbst implementiert haben, können wir die Operationen direkt zählen und sind nicht auf asymptotische Argumente angewiesen. Betrachten wir $f(\mathbf x)=(\sum_j x_j^2)^2$ mit $n=200$ Eingaben $x_j=0.01\,j$ für $j=1,\ldots,200$. Wir wählen ein Polynom, das nur aus $+$ und $\times$ besteht. So kann `Var` es aufzeichnen, ohne dass wir Closures für `exp` und `log` hinzufügen müssen (sie entsprächen den Methoden von `Dual`). Der Gradient ist $\nabla f=4\bigl(\sum_jx_j^2\bigr)\mathbf x$. Eine gewöhnliche Auswertung braucht $2n+1=401$ elementare Operationen: $n$ Quadrierungen, $n$ Additionen und eine abschließende Quadrierung. Der Vorwärtsmodus benötigt also $200$ Durchläufe mit je $401$ Operationen. Schätzen Sie vor dem Lesen der Ausgabe, wie viele Operationen der Rückwärtsmodus braucht.

<!--en:7964c23b-->
Der Vorwärtsmodus führte $80{,}200=n(2n+1)$ Operationen aus, der Rückwärtsmodus nur $802=2(2n+1)$. Davon zeichnete er $401$ im Vorwärtsdurchlauf auf und führte $401$ Rückwärtsfunktionen im Rückwärtsdurchlauf aus. Das Verhältnis beträgt $100=n/2$. Die beiden Gradienten stimmen in allen ausgegebenen Stellen überein. Bei unseren Implementierungen unterscheiden sich die Zahlen der Durchläufe um den Faktor $n$ und die Zahlen der Operationen um den Faktor $n/2$. Für $n=10^8$ ergibt das bis auf dieselbe kleine Konstante den Faktor aus der Einleitung.

<!--en:8e277217-->
Auf den ersten Blick scheint der gemessene Faktor zwei dem zuvor genannten Faktor drei zu widersprechen. Das liegt an der Zählweise. Für `Var.ops` zählt jede Rückwärtsfunktion als eine Operation, obwohl die Rückwärtsfunktion der Multiplikation selbst zwei Multiplikationen ausführt. Zählen wir die skalaren Multiplikationen direkt, erhalten wir $402$ im Rückwärtsdurchlauf und $201$ im Vorwärtsdurchlauf. Wert und Gradient zusammen kosten also wie bei der linearen Schicht dreimal so viel wie der Wert allein. Zählen wir die Additionen zum Aufsummieren der adjungierten Variablen mit, steigt der Aufwand des Rückwärtsdurchlaufs auf $1204$ Operationen. Der Gesamtfaktor beträgt damit vier und liegt weiterhin innerhalb der Schranke von Baur und Strassen.

<!--en:2a0a7821-->
Für diesen Geschwindigkeitsgewinn braucht der Rückwärtsmodus mehr Speicher. Das Tape muss alle $401$ aufgezeichneten Zwischenwerte behalten, bis der Rückwärtsdurchlauf sie verarbeitet. Der Vorwärtsmodus hält wie das ursprüngliche Programm nur die aktuellen Werte vor, ergänzt um ihre Tangenten.

<!--en:bddf54f6-->
### Aktivierungs-Checkpointing

<!--en:d69166c9-->
Eine Berechnung mit $T$ aufgezeichneten Schritten hält $T$ Zwischenwerte im Speicher, bis der Rückwärtsdurchlauf sie erreicht. In einem tiefen neuronalen Netz sind dies die Aktivierungen jeder Schicht für jedes Beispiel im Batch. Häufig begrenzt der Aktivierungsspeicher die Batchgröße oder die Tiefe des Netzes, noch bevor der Rechenaufwand zum Problem wird.

<!--en:0ef3f644-->
Die einfachste Abhilfe besteht darin, das Tape in Segmente mit je $s$ Schritten zu unterteilen. Im Vorwärtsdurchlauf speichern wir nur jeden $s$-ten Wert. Das sind unsere $T/s$ *Checkpoints*. Alles dazwischen verwerfen wir. Erreicht der Rückwärtsdurchlauf ein Segment, berechnet er dessen $s$ Schritte vom gespeicherten Checkpoint aus erneut und zeichnet sie auf einem temporären Tape auf. Er arbeitet das kurze Tape rückwärts ab und gibt es sofort wieder frei. Im Speicher liegen jederzeit nur die $T/s$ Checkpoints und ein aktives Segment der Länge $s$. Der maximale Speicherbedarf ist minimal, wenn wir $s$ so wählen, dass beide Terme gleich groß sind:

<!--en:00293899-->
Das Minimum ergibt sich aus der Bedingung $M'(s)=1-T/s^2=0$ oder direkt aus der Ungleichung zwischen arithmetischem und geometrischem Mittel $T/s+s\ge2\sqrt T$. Der maximale Speicherbedarf wächst nun nur noch mit der Quadratwurzel der Berechnungslänge statt linear. Da wir jeden Schritt genau einmal neu berechnen, kostet das lediglich einen zusätzlichen Vorwärtsdurchlauf. Für $T=400$ und die Segmentlängen $s=1,5,10,20,100,400$ erhalten wir den maximalen Speicherbedarf $M=401,85,50,40,104,401$ mit dem Minimum bei $s=20=\sqrt{400}$. Bei $T=10^4$ genügen mit Segmenten der Länge $s=100$ nur $200$ gespeicherte Werte anstelle von $10{,}000$ :cite:`Chen.Xu.Zhang.ea.2016`. :numref:`fig_mdl-cal-checkpointing` veranschaulicht diesen Ablauf.

<!--en:52736240-->
![Aktivierungs-Checkpointing auf einem Tape mit $T$ Schritten. Der gewöhnliche Rückwärtsmodus (oben) behält alle $T$ Werte, bis der Rückwärtsdurchlauf sie verwendet. Beim Checkpointing (unten) bleibt jeder $s$-te Wert gespeichert. Erreicht der Rückwärtsdurchlauf ein Segment, berechnet er es von seinem Checkpoint aus neu und arbeitet es dann rückwärts ab. Der Speicherbedarf beträgt $T/s+s$, also $2\sqrt T$ bei $s=\sqrt T$. Dafür genügt ein zusätzlicher Vorwärtsdurchlauf.](../img/mdl-cal-checkpointing.svg)

<!--en:e8451f9d-->
Checkpointing verändert Laufzeit und Speicherbedarf, liefert aber genau denselben Gradienten, sofern die Neuberechnung deterministisch verläuft. Dann sind die neu berechneten Zwischenwerte identisch, und die Ableitungen stimmen Bit für Bit mit denen des gewöhnlichen Rückwärtsmodus überein. Deep-Learning-Frameworks bieten diese Abwägung für ganze Netzblöcke an. Sie speichern nur die Eingabe eines solchen Blocks und führen ihn während der Backpropagation erneut aus. Dabei ist die Voraussetzung des Determinismus zu beachten. Ein erneut ausgeführter Block zieht eine neue Dropout-Maske, sofern der Zustand des Zufallszahlengenerators nicht wiederhergestellt wird. Die Batchnormalisierung aktualisiert ihre laufenden Statistiken ein zweites Mal. Nichtdeterministische GPU-Kernel liefern beim zweiten Durchlauf möglicherweise andere Bits. Wenden wir dieselbe Idee rekursiv innerhalb jedes Segments an, sinkt der Speicherbedarf bis auf $O(\log T)$. Der Aufwand für die Neuberechnungen beträgt dann $O(T\log T)$.

<!--en:86307fa2-->
## Verknüpfung von Produkten mit der Jacobi-Matrix

<!--en:c3a4f568-->
Beim automatischen Differenzieren stehen uns beide linearen Operatoren zur Verfügung, die wir gerade selbst implementiert haben: das JVP $\mathbf J\mathbf v$ im Vorwärtsmodus und das VJP $\mathbf u^\top\mathbf J$ im Rückwärtsmodus. Sobald wir diese Grundoperationen verknüpfen können, lässt sich die Krümmung entlang einer Richtung berechnen. Wir können auch die Ergebnisse numerischer Löser differenzieren oder bei Bedarf vollständige Jacobi-Matrizen aufbauen.

<!--en:2b19da33-->
Autograd in MXNet bietet nur das VJP im Rückwärtsmodus. Der folgende MXNet-Code verwendet deshalb einen Umweg für das fehlende JVP.

<!--en:4ecc4658-->
### Jacobi-Vektor-Produkte und Vektor-Jacobi-Produkte

<!--en:df07a826-->
In :numref:`subsec_autograd_fwd_rev` haben wir die Framework-Schnittstellen für beide Produkte kennengelernt. Werten wir sie für unsere Abbildung der Ebene bei $(0.5,0.5)$ mit $\mathbf v=\mathbf u=(1,2)$ aus. Aus der zuvor mit `Dual` berechneten Jacobi-Matrix erwarten wir $\mathbf J\mathbf v=(1+2\cdot0.8776,\ 0.5+2)=(2.7552,\ 2.5)$ und $\mathbf u^\top\mathbf J=(1+2\cdot0.5,\ 0.8776+2)=(2,\ 2.8776)$. Die Framework-Aufrufe liefern genau diese Werte.

<!--en:8ee9d5d7-->
Autograd in MXNet unterstützt nur den Rückwärtsmodus. Hier bauen wir $\mathbf J$ aus zwei VJPs auf, einem pro Zeile, und multiplizieren die Matrix dann mit $\mathbf v$. Für eine $2\times2$-Abbildung kostet das nicht mehr als zwei JVPs.

<!--en:a3e400f4-->
Sowohl ein JVP als auch ein VJP kostet einen einzigen Durchlauf. Beide geben die Ableitungen in einem einzigen Vektor weiter. Das VJP muss allerdings zusätzlich die Zwischenwerte des Vorwärtsdurchlaufs auf dem Tape speichern. Der Aufbau einer dicht besetzten $m\times n$-Jacobi-Matrix kostet im günstigeren Modus $\min(m,n)$ Durchläufe sowie Speicher für $mn$ Zahlen. Die vollständige Matrix aufzubauen lohnt sich, wenn $n$ klein ist, wie in unserem $2\times2$-Beispiel, oder wenn ein Algorithmus ihre Determinante braucht. Normalisierende Flüsse etwa transformieren Wahrscheinlichkeitsdichten mithilfe von $|\det\mathbf J|$.

<!--en:ef532a0e-->
### Hesse-Vektor-Produkte

<!--en:a7fa6dbb-->
Bei einem Modell mit $n=10^6$ Parametern hat die Hesse-Matrix $10^{12}$ Einträge. Im Format float32 würden sie 4 TB belegen. Optimierungsverfahren zweiter Ordnung benötigen die Matrix selbst jedoch nur selten. Das Newton-Verfahren braucht die Hesse-Matrix nur, um $\mathbf H\mathbf d=-\nabla\ell$ nach dem Aktualisierungsschritt $\mathbf d$ aufzulösen. Das Newton-CG-Verfahren (oder abgeschnittene Newton-Verfahren) löst dieses lineare Gleichungssystem mit dem Verfahren der konjugierten Gradienten. Dieser iterative Löser greift ausschließlich über Matrix-Vektor-Produkte auf $\mathbf H$ zu, also über $\mathbf H\mathbf v$ :cite:`Nocedal.Wright.2006,Martens.2010`. Können wir $\mathbf H\mathbf v$ berechnen, ohne $\mathbf H$ je aufzubauen?

<!--en:63508c37-->
Das gelingt, indem wir Rückwärtsmodus und Vorwärtsmodus verknüpfen. Die Hesse-Matrix ist die Jacobi-Matrix der Gradientenabbildung $\nabla\ell$. Das Produkt $\mathbf H\mathbf v$ ist daher ein JVP des Gradienten. Es misst, wie schnell sich der Gradient in Richtung $\mathbf v$ ändert:

<!--en:f447f6eb-->
Ein Rückwärtsdurchlauf berechnet $\nabla\ell$. Wenden wir auf diese Gradientenberechnung den Vorwärtsmodus mit der Anfangstangente $\mathbf v$ an, erhalten wir ihre Ableitung in Richtung $\mathbf v$. Diese Verknüpfung heißt *Vorwärts- über Rückwärtsmodus* :cite:`Pearlmutter.1994`. Unabhängig von der Dimension $n$ kostet sie nur ein kleines konstantes Vielfaches einer einzelnen Gradientenberechnung.

<!--en:0e4d455a-->
Log-Sum-Exp bietet einen einfachen Test mit einem Ergebnis in geschlossener Form. Der Gradient ist die Softmax-Funktion $\mathbf p$. Die Hesse-Matrix ist also die Jacobi-Matrix der Softmax-Funktion (siehe :numref:`sec_mdl-matrix-calculus`). Sie ist zugleich die Kovarianzmatrix eines One-Hot-Vektors $\mathbf e_J$, dessen Index $J$ wir aus der Verteilung $\mathbf p$ ziehen. Daher gilt

<!--en:874d19b5-->
Dabei bezeichnet $\odot$ die elementweise Multiplikation. Für die rechte Seite genügen $3n$ Multiplikationen, ohne dass wir eine $n\times n$-Matrix aufbauen müssen. Für $\mathbf x=(1,-0.5,2)$ und $\mathbf v=(1,0.5,-1)$ gilt $\mathbf p=(0.2537,0.0566,0.6897)$ und $\mathbf p^\top\mathbf v=-0.4077$. Damit erhalten wir

<!--en:90ad5fe4-->
Dieser Test ist aussagekräftiger als ein Test mit einer quadratischen Form $\tfrac12\mathbf x^\top\mathbf A\mathbf x$. Bei einer quadratischen Form stimmt der Gradient an der Stelle $\mathbf v$ zufällig mit $\mathbf H\mathbf v$ überein. Eine fehlerhafte Implementierung könnte den Test also bestehen. Bei Log-Sum-Exp dagegen hängt $\mathbf H$ von $\mathbf x$ ab. Der Gradient an der Stelle $\mathbf v$ beträgt $\operatorname{softmax}(\mathbf v)=(0.574,0.348,0.078)$ und weicht deutlich von $\mathbf H\mathbf v$ ab.

<!--en:b9dd75ad-->
Stehen beide Modi zur Verfügung, genügt eine Codezeile für den Vorwärts- über Rückwärtsmodus.

<!--en:0b19e3fd-->
MXNet bietet keinen Vorwärtsmodus, und Autograd unterstützt für `exp` keine Ableitung zweiter Ordnung. Deshalb werten wir hier :eqref:`eq_mdl-hvp` direkt über einen zentralen Differenzenquotienten des im Rückwärtsmodus berechneten Gradienten aus. Mit der Schrittweite $h=10^{-4}$ im Format float64 stimmt das Ergebnis auf etwa acht Stellen mit der analytischen Formel überein.

<!--en:86cf1417-->
Die berechneten Produkte stimmen in allen ausgegebenen Stellen mit :eqref:`eq_mdl-ad-lse-hessian` überein, ohne dass wir die $3\times3$-Hesse-Matrix aufbauen. In :exref:`ex_mdl-matrix-calculus-autodiff-hessian-vector-product` berechnen Sie dasselbe Produkt direkt mit unseren Klassen `Dual` und `Var` und vergleichen den Vorwärts- über Rückwärtsmodus mit dem Rückwärts- über Rückwärtsmodus.

<!--en:7c3fde0d-->
### Implizites Differenzieren

<!--en:4c2a45a6-->
Viele Größen, die uns interessieren, sind nur implizit als Lösung einer Gleichung gegeben. Ein iterativer Löser berechnet etwa den Fixpunkt einer Rekursion, das Gleichgewicht eines physikalischen Systems oder den Minimierer eines inneren Optimierungsproblems. Wir können zwanzig Newton-Iterationen auf einem Tape aufzeichnen und sie alle mit Backpropagation durchlaufen. Dazu müssen wir jedoch alle Zwischenwerte der Iteration speichern und differenzieren genau die Trajektorie, die der Löser genommen hat. Können wir die Ableitung allein aus der definierenden Gleichung bestimmen?

<!--en:0e8cd33d-->
Beginnen wir mit dem einfachsten Fall, einer einzigen Gleichung in einer Unbekannten, $g(x,\theta)=x^2-\theta=0$, mit der Lösung $x^\star=\sqrt\theta$. Statt die geschlossene Formel zu differenzieren, leiten wir die Identität $x^\star(\theta)^2-\theta\equiv0$ direkt nach $\theta$ ab. Das ergibt $2x^\star\,dx^\star/d\theta-1=0$. Lösen wir nach der Ableitung auf, erhalten wir

<!--en:ebcdcd34-->
Das ist tatsächlich die Ableitung von $\sqrt\theta$ (mit dem Wert $0.25$ bei $\theta=4$). Entscheidend ist, dass wir keine explizite Formel für $x^\star$ verwendet haben. Es genügen die Gleichung $g(x,\theta)=0$ und der Wert der Lösung selbst.

<!--en:efbb7563-->
Das allgemeine Ergebnis liefert der *Satz über implizite Funktionen* :cite:`Rudin.1976`, den wir ohne Beweis angeben. Die Funktion $\mathbf g(\mathbf x,\boldsymbol\theta)$ bilde $\mathbb R^d\times\mathbb R^p$ nach $\mathbb R^d$ ab. Dabei sei $\mathbf g$ in der Nähe einer Lösung $(\mathbf x^\star,\boldsymbol\theta)$ von $\mathbf g=\mathbf 0$ stetig differenzierbar, und die Jacobi-Matrix $\partial\mathbf g/\partial\mathbf x$ sei dort invertierbar. Dann ist die Lösung in einer Umgebung von $\boldsymbol\theta$ eine differenzierbare Funktion $\mathbf x^\star(\boldsymbol\theta)$. Die Kettenregel für die Identität $\mathbf g(\mathbf x^\star(\boldsymbol\theta),\boldsymbol\theta)\equiv\mathbf 0$ ergibt $\frac{\partial\mathbf g}{\partial\mathbf x}\frac{\partial\mathbf x^\star}{\partial\boldsymbol\theta}
+\frac{\partial\mathbf g}{\partial\boldsymbol\theta}=\mathbf 0$. Daraus folgt

<!--en:87c3eecb-->
Beide Jacobi-Matrizen auf der rechten Seite werten wir ausschließlich an der Lösung $(\mathbf x^\star,\boldsymbol\theta)$ aus. Für die vollständige Ableitungsmatrix müssen wir pro Parameter ein lineares Gleichungssystem mit der Matrix $\partial\mathbf g/\partial\mathbf x$ lösen, unabhängig von der Zahl der Iterationen des Lösers. In VJP-Form genügt sogar ein einziges lineares Gleichungssystem für alle Parameter zusammen. Es gilt $\mathbf u^\top\partial\mathbf x^\star/\partial\boldsymbol\theta
=-\mathbf a^\top\partial\mathbf g/\partial\boldsymbol\theta$, wobei der adjungierte Vektor $\mathbf a$ die Gleichung $(\partial\mathbf g/\partial\mathbf x)^\top\mathbf a=\mathbf u$ löst.

<!--en:5d2f2aea-->
Auf diesem Prinzip beruhen mehrere wichtige Verfahren des maschinellen Lernens. Dazu gehören die adjungierte Methode zum Differenzieren durch ODE-Löser (:numref:`sec_mdl-adjoint-method`) und tiefe Gleichgewichtsmodelle :cite:`Bai.Kolter.Koltun.2019`, die eine ganze Schicht als Fixpunkt $\mathbf z^\star=\mathbf f(\mathbf z^\star,\boldsymbol\theta)$ definieren und sie mithilfe von :eqref:`eq_mdl-implicit-grad` trainieren. Ein weiteres Beispiel ist die gradientenbasierte Hyperparameteroptimierung über eine innere Minimierung.

<!--en:7e035981-->
Testen wir :eqref:`eq_mdl-implicit-grad` an einem nichtlinearen $2\times2$-System mit den Gleichungen $g_1=x_1^2+\theta x_2-2=0$ und $g_2=x_1x_2+\theta-1=0$. Wir lösen es mit zwanzig Newton-Schritten und prüfen den impliziten Gradienten mit einem zentralen Differenzenquotienten der Ausgabe des Lösers.

<!--en:c1d8902e-->
Die beiden Vektoren stimmen bis zur vollen Genauigkeit der endlichen Differenz überein. Keine der zwanzig Newton-Iterationen taucht in der impliziten Formel auf. Da die Ableitung nur von $\mathbf g$ und der Lösung $\mathbf x^\star$ abhängt, liefert jeder Löser, der gegen dieselbe Nullstelle konvergiert, denselben Gradienten.

<!--en:1ec136f4-->
Der Satz über implizite Funktionen hat zwei klare Grenzen, die sich in der Praxis leicht beobachten lassen. Erstens muss $\partial\mathbf g/\partial\mathbf x$ an der Lösung invertierbar sein. In unserem skalaren Beispiel $x^2=\theta$ gilt bei $\theta=0$ die Gleichung $\partial g/\partial x=2x^\star=0$. Die Formel dividiert also durch null. Das entspricht der senkrechten Tangente von $\sqrt\theta$ im Ursprung. Zweitens setzt die Formel voraus, dass der Löser tatsächlich konvergiert ist. Untersuchen wir, was bei vorzeitigem Abbruch passiert. Dazu wenden wir das Newton-Verfahren auf $x^2=4$ an, starten bei $x=1$ und brechen nach $k=1,2,3$ Schritten ab. Wir vergleichen die implizite Formel $1/(2\tilde x_k)$ am noch nicht konvergierten Näherungswert $\tilde x_k$ mit der Ableitung des über $k$ Schritte entfalteten Lösers. Letztere schätzen wir durch einen zentralen Differenzenquotienten in $\theta$.

<!--en:15e7b68c-->
Nach einem, zwei und drei Schritten erreicht das Newton-Verfahren die Werte $2.5$, $2.05$ und $2.00061$. Die tatsächliche Ableitung beträgt $0.25$. Die implizite Formel liefert $0.2000$, $0.2439$ und $0.2499$, die Ableitung der entfalteten Iteration dagegen $0.5000$, $0.2900$ und $0.2509$. Vor der Konvergenz sind beide ungenau. In die implizite Formel geht der verbleibende Fehler des Lösers ein. Die Ableitung der entfalteten Iteration differenziert den vorzeitig abgebrochenen Iterationspfad und weicht hier noch stärker ab. Die Abhilfe liegt nahe: Wir lassen den Löser zuerst konvergieren.

<!--en:88dbf7c8-->
Ein grundlegendes Problem bleibt bestehen. Das automatische Differenzieren berechnet das Produkt $\mathbf J_K\cdots\mathbf J_1$ bis auf Gleitkommarundung exakt, kann dessen Wert jedoch nicht ändern. Über viele Schichten hinweg kann ein langes Produkt von Jacobi-Matrizen weiterhin geometrisch schrumpfen oder anwachsen (:numref:`sec_mdl-matrix-calculus`). Ein exakter Gradient der Größe $10^{-20}$ hilft beim Training kaum mehr als null. Um verschwindende und explodierende Gradienten zu beherrschen, müssen wir Initialisierung und Architektur sorgfältig wählen. Darauf gehen wir ab :numref:`sec_numerical_stability` ein.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:9533ec94-->
Die Ableitung eines Programms ist das Produkt der Jacobi-Matrizen seiner elementaren Schritte. Beim automatischen Differenzieren wenden wir dieses Produkt auf einen Startvektor an. Damit genügt in jedem Schritt ein Matrix-Vektor-Produkt anstelle eines Matrix-Matrix-Produkts. Der Vorwärtsmodus führt das Programm mit dualen Zahlen aus und berechnet pro Durchlauf ein Jacobi-Vektor-Produkt für eine Eingaberichtung. Der Rückwärtsmodus zeichnet im Vorwärtsdurchlauf ein Tape auf und arbeitet es rückwärts ab. So berechnet er pro Durchlauf ein Vektor-Jacobi-Produkt für eine Ausgaberichtung. Die Form der Jacobi-Matrix bestimmt, welcher Modus günstiger ist. Ein skalarer Verlust hat nur eine Ausgabe. Deshalb liefert ein Rückwärtsdurchlauf alle Parametergradienten für etwa das Dreifache der Kosten einer Verlustauswertung einschließlich des Vorwärtsdurchlaufs. Für $(\sum_j x_j^2)^2$ mit $n=200$ Eingaben brauchte der Rückwärtsmodus $802$ Operationen, der Vorwärtsmodus dagegen $80{,}200$. Der Speicherbedarf des Tapes wächst linear mit der Berechnungslänge $T$. Aktivierungs-Checkpointing senkt den maximalen Speicherbedarf auf $2\sqrt T$ und kostet einen zusätzlichen Vorwärtsdurchlauf. Mit dem Vorwärts- über Rückwärtsmodus lassen sich Hesse-Vektor-Produkte für ein kleines konstantes Vielfaches der Kosten einer Gradientenberechnung bestimmen. Mit dem Satz über implizite Funktionen lässt sich die Ausgabe eines Lösers ableiten, indem wir an der konvergierten Lösung ein einziges lineares Gleichungssystem lösen.

<!--en:2294243a-->
## Übungen

<!--en:a52fb2b6-->
1. [code] **Duale und komplexe Schritte.** Betrachten Sie $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ an der Stelle $\mathbf x_0=(0.5,0.5)$ und die zweite Spalte $\mathbf J\mathbf e_2=(\cos0.5,\ 1)$ der zugehörigen Jacobi-Matrix. Der zentrale Differenzenquotient $(\mathbf f(\mathbf x_0+t\mathbf e_2)-\mathbf f(\mathbf x_0-t\mathbf e_2))/(2t)$ hat einen Abbruchfehler der Ordnung $t^2$ und einen Rundungsfehler der Ordnung $u/t$. Dabei bezeichnet $u\approx10^{-16}$ die Rundungseinheit von float64.
    1. Sagen Sie die beste Schrittweite und den kleinsten Fehler voraus. Prüfen Sie Ihre Vorhersage für $t=10^{-1},10^{-2},\ldots,10^{-12}$ (Sie sollten einen Fehler von etwa $2\times10^{-11}$ in der Nähe von $t=10^{-5}$ finden). Vergleichen Sie das Ergebnis mit einem Durchlauf von `Dual` mit dem Startvektor $\mathbf e_2$. Warum geht in die Rechnung mit dualen Zahlen keine Schrittweite ein?
    1. Wiederholen Sie (a) für die erste Spalte $\mathbf J\mathbf e_1$. Erklären Sie, warum der zentrale Differenzenquotient hier für jedes $t$ bis auf Rundung exakt ist. (Hinweis: $\mathbf f$ ist ein Polynom zweiten Grades in $x$.)
    1. Ersetzt man $\varepsilon$ durch die imaginäre Einheit $i$, erhält man die Näherung mit komplexem Schritt $\operatorname{Im}\mathbf f(\mathbf x_0+it\mathbf e_2)/t$ :cite:`Squire.Trapp.1998`. Sie gilt für jede Funktion $\mathbf f$, die sich wie die hier betrachtete analytisch auf komplexe Argumente fortsetzen lässt. Entwickeln Sie $\mathbf f(\mathbf x_0+it\mathbf e_2)$ bis zur dritten Ordnung in $t$ und zeigen Sie, dass der Fehler der Näherung $O(t^2)$ beträgt. Warum müssen bei dieser Näherung keine fast gleich großen Zahlen subtrahiert werden?
    1. Werten Sie die Näherung mit komplexem Schritt für $t=10^{-1},10^{-2},\ldots,10^{-20}$ mit der komplexen Arithmetik von NumPy aus. Vergleichen Sie die Fehler mit denen aus (a). Welcher Unterschied zwischen $\varepsilon^2=0$ und $i^2=-1$ verursacht den Fehler der Ordnung $O(t^2)$ beim komplexen Schritt?

<!--en:170a9bf5-->
1. [code] **Hyperparametergradient.** Die Ridge-Regression mit einem Regularisierungsparameter $\lambda_j>0$ pro Merkmal bestimmt die Gewichte durch

   $$\mathbf w^\star(\boldsymbol\lambda)=\operatorname*{argmin}_{\mathbf w}\ \|\mathbf X\mathbf w-\mathbf y\|^2+\sum_{j=1}^d\lambda_jw_j^2$$

   Dabei sind $(\mathbf X,\mathbf y)$ die Trainingsdaten. Auf den Daten $(\mathbf X_{\text{v}},\mathbf y_{\text{v}})$ beträgt der Validierungsverlust $V(\boldsymbol\lambda)=\|\mathbf X_{\text{v}}\mathbf w^\star(\boldsymbol\lambda)-\mathbf y_{\text{v}}\|^2$.
    1. Zeigen Sie, dass $\mathbf w^\star$ die Gleichung $\mathbf g(\mathbf w,\boldsymbol\lambda)=(\mathbf X^\top\mathbf X+\operatorname{diag}(\boldsymbol\lambda))\,\mathbf w-\mathbf X^\top\mathbf y=\mathbf 0$ löst. Leiten Sie mithilfe von :eqref:`eq_mdl-implicit-grad` die Matrix $\partial\mathbf w^\star/\partial\boldsymbol\lambda$ her.
    1. Zeigen Sie, dass $\nabla_{\boldsymbol\lambda}V=-\mathbf a\odot\mathbf w^\star$ gilt, wobei $\mathbf a$ ein lineares Gleichungssystem mit der Matrix $\mathbf X^\top\mathbf X+\operatorname{diag}(\boldsymbol\lambda)$ löst. Geben Sie dessen rechte Seite an. Wie viele lineare Gleichungssysteme sind beim Vorwärtsverfahren über $\partial\mathbf w^\star/\partial\boldsymbol\lambda$ für $d$ Regularisierungsparameter zu lösen, und wie viele sind es beim adjungierten Verfahren?
    1. Implementieren Sie $V$ und den adjungierten Gradienten in NumPy für zufällige Daten mit $50$ Trainingsbeispielen, $50$ Validierungsbeispielen und $d=10$ Merkmalen. Prüfen Sie den Gradienten anhand zentraler Differenzenquotienten von $V$.

    *Nach Alan Edelman und Steven G. Johnson, [Matrix Calculus for Machine Learning and Beyond, Aufgabenblatt 2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_pset2.pdf), MIT 18.S096, IAP 2023, Aufgabe 1.*

<!--en:00f5d7af-->
1. [code] **Hesse-Vektor-Produkt.** ● Die Funktion $f(\mathbf x)=\bigl(\sum_j x_j^2\bigr)^2$ aus :numref:`subsec_mdl-cost-asymmetry` wird an den Eingaben `xs0` mit $n=200$ ausgewertet. Sie besteht nur aus Additionen und Multiplikationen, sodass sowohl `Dual` als auch `Var` sie auswerten können.
    1. Leiten Sie den Gradienten $\nabla f$ und die Hesse-Matrix $\mathbf H$ von $f$ her. Schreiben Sie $\mathbf H\mathbf v$ in einer Form, die $\mathbf H$ nie explizit aufbaut.
    1. Ziehen Sie eine zufällige Richtung $\mathbf v$, stellen Sie jede Eingabe $x_j$ als `Var(Dual(x_j, v_j))` dar und rufen Sie `backprop` auf. Zeigen Sie, dass die Wertanteile der resultierenden adjungierten Variablen den Gradienten $\nabla f$ bilden und ihre $\varepsilon$-Anteile das Produkt $\mathbf H\mathbf v$. Vergleichen Sie beide mit (a) und mit dem Hesse-Vektor-Produkt aus :numref:`subsec_mdl-hvp` für dieselbe Funktion $f$. Welche Operationen der Rückwärtsfunktionen rechnen mit dualen Zahlen? Warum entspricht die Berechnung damit dem Vorwärts- über Rückwärtsmodus im Sinne von :eqref:`eq_mdl-hvp`?
    1. Verschachteln Sie stattdessen das Tape mit sich selbst. Stellen Sie jede Eingabe als `Var(Var(x_j))` dar und rufen Sie `backprop` auf. Die adjungierten Variablen sind nun `Var`-Objekte. Bilden Sie daraus $\sum_j(\nabla f)_jv_j$ und wenden Sie `backprop` darauf an. So erhalten Sie $\mathbf H\mathbf v$ im Rückwärts- über Rückwärtsmodus. Zählen Sie für jedes Verfahren die erzeugten `Var`-Objekte, etwa mit einem Zähler in `Var.__init__`. Vergleichen Sie beide Anzahlen mit der Anzahl für einen gewöhnlichen Gradienten. Welches Verfahren zeichnet den Rückwärtsdurchlauf auf, und wie viel Speicher kostet das?
   :exlabel:`ex_mdl-matrix-calculus-autodiff-hessian-vector-product`

<!--en:f3e2a5f3-->
1. [code] **Checkpointing auf dem Tape.** ● Betrachten Sie $x_k=x_{k-1}-0.01\,x_{k-1}^2$ für $k=1,\ldots,T$ mit $T=400$ und $x_0=1$. Differenzieren Sie $x_T$ nach $x_0$. Da `Var` keine Subtraktion unterstützt, zeichnen Sie jeden Schritt als `x + (-0.01) * x * x` auf.
    1. Zeichnen Sie die Kette mit der gewöhnlichen Funktion `backprop` auf dem Tape auf. Zählen Sie die Knoten auf dem Tape zu Beginn des Rückwärtsdurchlaufs.
    1. Die Funktion `backprop` setzt die adjungierte Variable der Ausgabe auf $1$. Verallgemeinern Sie die Funktion so, dass sie jeden Startwert akzeptiert. Zeichnen Sie ein Segment $x_i\mapsto x_j$ auf, beginnend mit einem neuen Blatt für $x_i$. Zeigen Sie, dass die verallgemeinerte Funktion `backprop` bei einem Aufruf ab $x_j$ mit dem Startwert $\partial x_T/\partial x_j$ den Wert $\partial x_T/\partial x_i$ in der adjungierten Variablen des Blatts liefert.
    1. Implementieren Sie den Ablauf aus :numref:`fig_mdl-cal-checkpointing` mit Segmenten von je $s$ Schritten. Ein Vorwärtsdurchlauf mit gewöhnlichen Gleitkommazahlen speichert $x_0,x_s,x_{2s},\ldots$. Ein anschließender Rückwärtsdurchlauf zeichnet vom letzten bis zum ersten Segment jeweils ein Segment neu auf und wendet (b) darauf an. Prüfen Sie, dass das Ergebnis mit dem Gradienten aus (a) übereinstimmt.
    1. Geben Sie für $s\in\{1,5,10,20,100,400\}$ die Anzahl gespeicherter Checkpoints, die Länge des größten Tapes in Schritten und die Anzahl der Auswertungen des Schritts $x\mapsto x-0.01x^2$ einschließlich des ersten Vorwärtsdurchlaufs an. Bestätigen Sie den Speicherbedarf $T/s+s$ aus :eqref:`eq_mdl-ad-checkpoint-memory`, also $401,85,50,40,104,401$, und seinen Minimierer $s=\sqrt T=20$. Wiederholen Sie die Untersuchung für $T=100$ und $T=1600$ mit allen Teilern $s\le800$ von $T$ (ein einzelnes Segment aus $1600$ Schritten überschreitet in `backprop` das standardmäßige Rekursionslimit von Python). Wie wächst die optimale Segmentlänge $s$ mit $T$, und wie viele Schrittauswertungen benötigt der beste Ablauf?

<!--en:6fe9c6e5-->
1. [extended] **Gleichgewichtsschicht.** ●● Eine tiefe Gleichgewichtsschicht :cite:`Bai.Kolter.Koltun.2019` gibt den folgenden Fixpunkt aus:

   $$\mathbf z^\star=\tanh(\mathbf W\mathbf z^\star+\mathbf c),$$

   Dabei hängt $\mathbf c$ von der Eingabe der Schicht ab. Der Verlust sei $\ell=\tfrac12\|\mathbf z^\star-\mathbf y\|^2$. Wir schreiben $\mathbf D=\operatorname{diag}\bigl(1-(z^\star_1)^2,\ldots,1-(z^\star_d)^2\bigr)$ für die Ableitung von $\tanh$ am Fixpunkt.
    1. Zeigen Sie, dass die Abbildung $\mathbf z\mapsto\tanh(\mathbf W\mathbf z+\mathbf c)$ für $\|\mathbf W\|_2<1$ eine Kontraktion ist, sodass der Fixpunkt existiert und eindeutig ist. Zeigen Sie außerdem, dass $\mathbf I-\mathbf D\mathbf W$ dann invertierbar ist. Leiten Sie mithilfe von :eqref:`eq_mdl-implicit-grad` den Gradienten $\nabla_{\mathbf W}\ell$ her. Zeigen Sie, dass er gleich $(\mathbf D\mathbf a)(\mathbf z^\star)^\top$ ist, wobei die adjungierte Variable $\mathbf a$ die Gleichung $(\mathbf I-\mathbf D\mathbf W)^\top\mathbf a=\mathbf z^\star-\mathbf y$ löst.
    1. Wählen Sie $d=20$ und eine Matrix $\mathbf G\in\mathbb R^{d\times d}$ mit standardnormalverteilten Einträgen. Setzen Sie $\mathbf S=\mathbf G+\mathbf G^\top$ und $\mathbf W=\rho\,\mathbf S/\|\mathbf S\|_2$ für $\rho\in\{0.5,0.9,0.99\}$. Ziehen Sie die Einträge von $\mathbf c$ und $\mathbf y$ unabhängig aus $\mathcal N(0,0.1^2)$. Berechnen Sie in NumPy den Fixpunkt $\mathbf z^\star$ durch Fixpunktiteration und den Gradienten aus (a) durch Lösen eines einzigen linearen Gleichungssystems. Prüfen Sie einige seiner Einträge anhand zentraler Differenzenquotienten von $\ell$.
    1. Beim Differenzieren der entfalteten Iteration führen Sie $K$ Iterationen ab $\mathbf z_0=\mathbf 0$ aus und wenden Backpropagation auf die gesamte Folge an. Implementieren Sie diesen Rückwärtsdurchlauf von Hand mit einem VJP pro Iteration. Tragen Sie gegen $K$ den Fehler des so berechneten Gradienten gegenüber dem Gradienten aus (b) und den Vorwärtsfehler $\|\mathbf z_K-\mathbf z^\star\|$ auf. Vergleichen Sie die Abklingraten beider Fehler mit dem Spektralradius von $\mathbf D\mathbf W$. Wie verhält sich das Verhältnis der beiden Fehler mit wachsendem $K$ und warum? Wie wächst der Speicherbedarf der beiden Gradientenberechnungen mit $K$?
    1. Ersetzen Sie die Lösung des linearen Gleichungssystems durch $m$ Schritte der Iteration $\mathbf a\leftarrow(\mathbf z^\star-\mathbf y)+(\mathbf D\mathbf W)^\top\mathbf a$ mit dem Startwert $\mathbf a=\mathbf 0$. Das entspricht einer abgeschnittenen Neumann-Reihe, die :citet:`Geng.Zhang.Bai.ea.2021` als *Phantomgradienten* verwenden. Zeigen Sie, dass diese Näherung der adjungierten Variablen denselben Gradienten liefert wie Backpropagation durch $m$ Iterationen mit Start bei $\mathbf z^\star$. Tragen Sie für jedes $\rho$ den Kosinus des Winkels zwischen dem näherungsweisen und dem exakten Gradienten gegen $m$ auf. Für $m=1$ benötigt der Gradient kein lineares Gleichungssystem mehr. Das entspricht der Backpropagation ohne Jacobi-Matrix nach :citet:`Fung.Heaton.Li.ea.2022`. Liefert der negative Näherungsgradient noch eine Abstiegsrichtung für $\ell$?

<!--en:0fb82e53-->
1. [code] **Zweite Ableitungen in einem Vorwärtsdurchlauf.** Fügen Sie $\varepsilon$ mit $\varepsilon^3=0$ (aber $\varepsilon^2\neq0$) hinzu und schreiben Sie $a+b\varepsilon+c\varepsilon^2$ als Tripel $(a,b,c)$.
    1. Leiten Sie das Produkt $(a,b,c)(a',b',c')=(aa',\ ab'+ba',\ ac'+bb'+ca')$ her und zeigen Sie, dass für eine analytische Funktion $\varphi$ die Gleichung $\varphi(a+b\varepsilon+c\varepsilon^2)=\varphi(a)+\varphi'(a)\,b\,\varepsilon+\bigl(\varphi'(a)\,c+\tfrac12\varphi''(a)\,b^2\bigr)\varepsilon^2$ gilt.
    1. Implementieren Sie `Dual2` mit `__add__`, `__mul__`, `sin` und `exp` und werten Sie $f(x)=\sin(x^2)+e^x$ an der Stelle $(1.3,1,0)$ aus. Zeigen Sie, dass das Ergebnis $(f(1.3),\ f'(1.3),\ \tfrac12f''(1.3))$ lautet, und prüfen Sie $f''(1.3)\approx-3.2806$ anhand der von Hand hergeleiteten Formel $f''(x)=2\cos x^2-4x^2\sin x^2+e^x$.
    1. Betrachten Sie $\ell:\mathbb R^n\to\mathbb R$ mit Eingaben, deren Startwerte durch eine Richtung $\mathbf v$ gegeben sind. Was liefert der $\varepsilon^2$-Anteil? (Antwort: $\tfrac12\mathbf v^\top\mathbf H\mathbf v$, also eine Zahl.) Wie viele solcher Durchläufe brauchen Sie, um den Vektor $\mathbf H\mathbf v$ durch Polarisation zu erhalten, verglichen mit einem Durchlauf im Vorwärts- über Rückwärtsmodus?
    1. Stellen Sie $\varepsilon$ als $3\times3$-Matrix mit $\varepsilon^3=\mathbf 0$ dar und bestätigen Sie das Produkt aus (a) durch Matrizenmultiplikation.

<!--en:d09b7bf1-->
[Diskussion](https://d2l.discourse.group/t/36827)

<!--en:7cb316d6-->
Automatisches Differenzieren<br>**Duale Zahlen und Vorwärtsmodus · Rückwärtsmodus und Tape · Parametergradienten · Checkpointing · HVPs · implizites Differenzieren**

<!--en:bbc05a5a-->
::: {.slide title="12 Tage gegenüber 30 Millisekunden bei 10⁸ Parametern"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:651d8fa2-->
- **Endliche Differenzen ($10^8$ Durchläufe):** Bei 10 ms pro Durchlauf dauert es rund 11.6 Tage pro Schritt, $10^8$ Parameter einzeln zu variieren.

<!--en:c0a65159-->
- **Automatisches Differenzieren im Rückwärtsmodus (1 Durchlauf):** Alle $10^8$ exakten partiellen Ableitungen kosten so viel wie rund 3 Vorwärtsdurchläufe (etwa 30 ms).

<!--en:ae40f438-->
- **Spalten und Zeilen von $\mathbf{J} = \mathbf{J}_K \cdots \mathbf{J}_1$:** Der Vorwärtsmodus liefert pro Durchlauf 1 Spalte für eine Eingabe, der Rückwärtsmodus 1 Zeile für eine Ausgabe.

<!--en:2610faa3-->
::: {.slide title="Duale Zahlen brechen Taylor-Reihen algebraisch ab"}

<!--en:332313f5-->
[Algebra des Vorwärtsmodus]{.kicker}

<!--en:4f28132f-->
- **Rechnen mit dualen Zahlen ($a + b\varepsilon$ mit $\varepsilon^2 = 0$):** $(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon$ setzt die Produktregel automatisch um.

<!--en:6b04d306-->
- **Exakte Potenzen bei $x = 3$:** $(3 + \varepsilon)^2 = 9 + 6\varepsilon$ und $(3 + \varepsilon)^3 = 27 + 27\varepsilon$ ohne Schrittweitenfehler.

<!--en:f06cbc23-->
- **Isomorphismus zu $2 \times 2$-Matrizen:** $a + b\varepsilon \leftrightarrow \left(\begin{smallmatrix}a & b \\ 0 & a\end{smallmatrix}\right)$.

<!--en:8c27ac44-->
::: {.slide title="Warum duale Zahlen exakte Ableitungen liefern"}

<!--en:9dbe3b2b-->
[Strukturelle Induktion]{.kicker}

<!--en:095e02f6-->
- **Überladene Grundfunktionen:** Die Definition $\varphi(a + b\varepsilon) = \varphi(a) + \varphi'(a)b\varepsilon$ gewährleistet $f(x + 1\varepsilon) = f(x) + f'(x)\varepsilon$ für jedes aus diesen Operationen zusammengesetzte Programm.

<!--en:41439316-->
- **Beweis durch strukturelle Induktion:** Induktionsanfang ($c + 0\varepsilon$, $x + 1\varepsilon$) und Induktionsschritte ($+$, $\times$, $\varphi \circ g$) erhalten die exakten Ableitungen des Ausführungspfads auch in Schleifen und Verzweigungen.

<!--en:ef769299-->
::: {.slide title="Duale Zahlen durch Überladen von Operatoren"}

<!--en:19b3043c-->
[Implementierung des Vorwärtsmodus]{.kicker}

<!--en:b58b88bd-->
In Python überlädt eine kurze Klasse `Dual(a, b)` die Operationen `+`, `*`, `sin`, `exp` und `log`. Sie gibt den Wert `a` und die Tangente `b` ($\dot{v}$) gemeinsam weiter. Für $f(x) = \sin(x^2) + e^x$ liefert ein Aufruf mit `Dual(1.3, 1.0)` sowohl $f(1.3)$ als auch $f'(1.3) = 2x\cos(x^2) + e^x$:

<!--en:76955e2d-->
Funktionswert $4.6622$ und Ableitung $3.360101$ stimmen in allen ausgegebenen Stellen mit der analytischen Formel überein. Eine Schrittweite ist dafür nicht nötig.

<!--en:fb520941-->
::: {.slide title="Log-Sum-Exp im Vorwärtsmodus braucht einen Durchlauf pro Eingabe"}

<!--en:c192c596-->
[Der O(n)-Flaschenhals]{.kicker}

<!--en:4786f646-->
- **Eine partielle Ableitung pro Durchlauf:** Die Initialisierung der Eingabe $j$ mit `Dual(x_j, 1.0)` liefert nur $\partial\ell/\partial x_j = p_j$. Daher sind $n$ Durchläufe für $n$ Eingaben nötig.

<!--en:0e36160e-->
::: {.slide title="Jacobi-Vektor-Produkte (JVPs) und vektorwertige Tangenten"}

<!--en:a9290b1c-->
[Spaltenzahl und Rechenaufwand]{.kicker}

<!--en:40fe3780-->
- **JVP für eine Richtung:** Für die Eingaben $\mathbf{x} + \mathbf{v}\varepsilon$ ergibt sich $\mathbf{f}(\mathbf{x}) + (\mathbf{J}\mathbf{v})\varepsilon$ in einem Durchlauf.

<!--en:549b80fc-->
- **Scheinbare Ersparnis durch vektorwertige Tangenten:** Mit Tangenten der Länge $n$ aus $\mathbf{I}_n$ genügt ein Python-Aufruf. Jede elementare Operation erfordert jedoch weiterhin Vektorrechnungen mit Aufwand $O(n)$.

<!--en:a7b8faee-->
::: {.slide title="Zwei Klammerungen für JVP (von rechts) und VJP (von links)"}

<!--en:6eb31bba-->
[Regel für den Rechenaufwand]{.kicker}

<!--en:2a8139a2-->
- **Jacobi-Matrizen nie als Zwischenergebnis aufbauen:** Einen Startvektor schrittweise mit den Faktoren von $\mathbf{J} = \mathbf{J}_K \mathbf{J}_{K-1} \cdots \mathbf{J}_1$ multiplizieren, entweder vorwärts ($\mathbf{J}\mathbf{v} = \mathbf{J}_K(\cdots(\mathbf{J}_1\mathbf{v}))$) oder rückwärts:
  $$\mathbf{u}^\top\mathbf{J} = \bigl((\mathbf{u}^\top\mathbf{J}_K)\mathbf{J}_{K-1}\cdots\bigr)\mathbf{J}_1$$

<!--en:6452ac3f-->
- **Die kleinere Dimension von $\mathbf{J} \in \mathbb{R}^{m \times n}$ entscheidet:**
  - **Vorwärtsmodus:** $1$ Durchlauf pro Eingaberichtung ($n$ Durchläufe für die gesamte Matrix $\mathbf{J}$, ideal bei $n \ll m$).
  - **Rückwärtsmodus:** $1$ Durchlauf pro Ausgaberichtung ($m$ Durchläufe für die gesamte Matrix $\mathbf{J}$. Bei skalarem Verlust gilt $m = 1$, **ein Durchlauf** liefert den gesamten Gradienten).

<!--en:1d7d183c-->
::: {.slide title="Warum der Rückwärtsmodus ein Tape (Wengert-Liste) braucht"}

<!--en:b5685e3f-->
[Asymmetrie der Ausführung]{.kicker}

<!--en:2ee9baee-->
- **Tangenten ($\dot{v}$) und adjungierte Variablen ($\bar{v}$):**
  - Die Tangente $\dot{v} = \frac{\mathrm{d}v}{\mathrm{d}x}$ hängt nur von Operationen **vor** $v$ ab. Sie läuft mit den Werten vorwärts, ohne den bisherigen Verlauf zu speichern.
  - Die adjungierte Variable $\bar{v} = \frac{\mathrm{d}\ell}{\mathrm{d}v}$ hängt von jeder Verwendung **nach** $v$ ab ($\bar{v} = \sum_c \bar{c}\frac{\partial c}{\partial v}$) und läuft entgegen der Ausführungsrichtung zurück.

<!--en:71f2c2b4-->
- **Zwei Phasen (Wengert, 1964, Linnainmaa, 1970):** Der Rückwärtsmodus führt das Programm zuerst vorwärts aus und zeichnet jede Operation und jeden Zwischenwert auf einem **Tape** auf. Dann arbeitet er das Tape ausgehend von $\bar{\ell} = 1$ in umgekehrter topologischer Reihenfolge ab.

<!--en:147356e1-->
::: {.slide title="Der Rückwärtsmodus mit Tape in 40 Zeilen"}

<!--en:4676a2d3-->
[Implementierung des Rückwärtsmodus]{.kicker}

<!--en:cc945a28-->
- **1. DAG-Knoten (`Var`):** Speichert `.value`, `.grad` (anfangs `0.0`), `._parents` und `._backward(g)`.

<!--en:56ae4a61-->
- **2. Closures für lokale VJPs:** Die Addition addiert `g` bei beiden direkten Vorgängern. Die Multiplikation multipliziert `g` mit dem Vorwärtswert `.value` des jeweils *anderen* direkten Vorgängers.

<!--en:46eb9ca7-->
- **3. Topologische Sortierung in Nachordnung:** Beim Durchlaufen von `reversed(order)` ab `node.grad = 1.0` gehen alle Beiträge der direkten Nachfolger ein, bevor `v._backward(v.grad)` ausgeführt wird.

<!--en:6ad7674e-->
::: {.slide title="Ein rautenförmiges Tape von Hand durchrechnen: Warum += nötig ist"}

<!--en:96013fee-->
[Knoten mit mehreren Nachfolgern]{.kicker}

<!--en:dff78b7e-->
- **Vorwärts aufgezeichnetes Tape für $y = (uv + u)^2$ bei $(2, -3)$:** Es enthält $t = uv = -6$, $r = t + u = -4$ und $y = r \cdot r = 16$.

<!--en:373f8058-->
- **Rückwärtsdurchlauf ab $\bar{y} = 1$:**
  - Der Knoten $r$ liefert **beide** Faktoren von $y = r \cdot r$. Daher gilt $\bar{r} = \bar{y}r + \bar{y}r = -4 + (-4) = -8$ und $\bar{t} = \bar{r}(1) = -8$.
  - Die Eingabe $u$ geht **sowohl** in $r = t+u$ als auch in $t = uv$ ein. Daher gilt $\bar{u} = \bar{r}(1) + \bar{t}(v) = -8 + (-8)(-3) = 16$, während $\bar{v} = \bar{t}(u) = -16$ gilt.

<!--en:0da59514-->
- **Warum `+=` nötig ist:** Eine Zuweisung mit `=` an den Verzweigungen $r$ und $u$ verwirft Beiträge paralleler Pfade.

<!--en:74423e35-->
::: {.slide title="Unser einfaches Tape und Autograd im Framework"}

<!--en:f989df99-->
[Überprüfung des Codes]{.kicker}

<!--en:5d0bb75e-->
Das Tape mit `Var` in 40 Zeilen und Autograd im Framework (das `Var` auf Tensor-VJPs überträgt) liefern beide $\partial y/\partial u = 16$ und $\partial y/\partial v = -16$:

<!--en:1e6d563a-->
::: {.slide title="Warum Parametergradienten etwa 3 Vorwärtsdurchläufe kosten"}

<!--en:37a84155-->
[Rechenaufwand]{.kicker}

<!--en:525f526d-->
- **Drei Matrixprodukte pro Schicht ($\mathbf{h}_k = \mathbf{W}_k\mathbf{h}_{k-1}$):**
  1. Vorwärtsdurchlauf: $\mathbf{h}_k = \mathbf{W}_k\mathbf{h}_{k-1}$ ($d_k d_{k-1}$ Multiplikationen, speichert $\mathbf{h}_{k-1}$).
  2. Gewichtsgradient: $\nabla_{\mathbf{W}_k}\ell = \bar{\mathbf{h}}_k\mathbf{h}_{k-1}^\top$ ($d_k d_{k-1}$ Multiplikationen).
  3. Weitergabe der adjungierten Variablen: $\bar{\mathbf{h}}_{k-1} = \mathbf{W}_k^\top\bar{\mathbf{h}}_k$ ($d_k d_{k-1}$ Multiplikationen).

<!--en:b99048fe-->
- **Gesamtkosten:** $3\times$ die Kosten eines Vorwärtsdurchlaufs (innerhalb der Schranke von Baur und Strassen von $3\text{--}5\times$).

<!--en:6c3ac6bf-->
::: {.slide title="Gemessene Anzahl der Operationen: 80 200 gegenüber 802"}

<!--en:a55ebaca-->
[Empirischer Vergleich]{.kicker}

<!--en:80cf68d6-->
- **Testfunktion $f(\mathbf{x}) = (\sum_{j=1}^{200} x_j^2)^2$ ($2n+1 = 401$ Operationen):** Der Vorwärtsmodus braucht $200 \times 401 = 80{,}200$ Operationen, der Rückwärtsmodus $2 \times 401 = 802$ (Beschleunigung um den Faktor $100\times$).

<!--en:4f0b3537-->
::: {.slide title="Speicher als Flaschenhals und Aktivierungs-Checkpointing"}

<!--en:c67bfd6a-->
[Speicherbedarf und Rechenaufwand]{.kicker}

<!--en:8d42df11-->
- **Warum der Rückwärtsmodus speicherbegrenzt ist:** Eine Berechnung mit $T$ Schritten hält alle $T$ Zwischenaktivierungen auf dem Tape, bis der Rückwärtsdurchlauf sie verarbeitet.

<!--en:dd199fef-->
- **Ein Checkpoint nach jedem $s$-ten Schritt (Chen et al., 2016):** Im Vorwärtsdurchlauf nur $T/s$ Checkpoints speichern. Im Rückwärtsdurchlauf jedes Segment der Länge $s$ von seinem Checkpoint aus neu berechnen und auf einem lokalen Tape der Größe $s$ aufzeichnen.

<!--en:ce0e78af-->
- **Optimale Segmentlänge $s^\star = \sqrt{T}$:** Der maximale Speicherbedarf beträgt
  $$M(s) = \frac{T}{s} + s \ge 2\sqrt{T}$$
  Bei $T = 400$ ($s^\star = 20$) sinkt er um den Faktor $10\times$ ($401 \to 40$). Das kostet nur **einen zusätzlichen Vorwärtsdurchlauf**.

<!--en:8ef60dce-->
::: {.slide title="JVP und VJP als kombinierbare Grundoperationen in Frameworks" only="pytorch,tensorflow,jax"}

<!--en:3c39c8b8-->
[Framework-Schnittstellen]{.kicker}

<!--en:77f637f8-->
Moderne Frameworks stellen beide linearen Operatoren als kombinierbare Transformationen höherer Ordnung bereit: `jvp` ($\mathbf{J}\mathbf{v}$) im Vorwärtsmodus und `vjp` ($\mathbf{u}^\top\mathbf{J}$) im Rückwärtsmodus. Für $\mathbf{f}(x, y) = (x + \sin y,\ y + x^2/2)^\top$ an der Stelle $(0.5, 0.5)^\top$ mit $\mathbf{v} = \mathbf{u} = (1, 2)^\top$ ergibt sich:

<!--en:4be003e2-->
Die Produkte $\mathbf{J}\mathbf{v} = (2.7552, 2.5)^\top$ und $\mathbf{u}^\top\mathbf{J} = (2.0, 2.8776)$ benötigen jeweils nur einen Durchlauf und keinen Speicher für eine $m \times n$-Matrix. Die dicht besetzte Jacobi-Matrix $\mathbf{J}$ nur aufbauen, wenn $\min(m, n)$ klein oder $\det \mathbf{J}$ erforderlich ist (z. B. bei normalisierenden Flüssen).

<!--en:f154ddaf-->
::: {.slide title="JVP und VJP als kombinierbare Grundoperationen in Frameworks" only="mxnet"}

<!--en:3c39c8b8-2-->
[Framework-Schnittstellen]{.kicker}

<!--en:176a8091-->
Der Vorwärtsmodus liefert $\mathbf{J}\mathbf{v}$, der Rückwärtsmodus $\mathbf{u}^\top\mathbf{J}$. Für $\mathbf{f}(x, y) = (x + \sin y,\ y + x^2/2)^\top$ an der Stelle $(0.5, 0.5)^\top$ mit $\mathbf{v} = \mathbf{u} = (1, 2)^\top$ ergibt sich:

<!--en:2da4ced5-->
Autograd in MXNet unterstützt nur den Rückwärtsmodus. Zwei VJPs liefern je eine Zeile von $\mathbf{J}$, anschließend folgt die Multiplikation mit $\mathbf{v}$. Für eine $2\times2$-Abbildung kostet das höchstens so viel wie zwei JVPs. Die dicht besetzte Jacobi-Matrix $\mathbf{J}$ nur aufbauen, wenn $\min(m, n)$ klein oder $\det \mathbf{J}$ erforderlich ist (z. B. bei normalisierenden Flüssen).

<!--en:7e9d2325-->
::: {.slide title="Hesse-Vektor-Produkte (HVPs) ohne Aufbau von H"}

<!--en:5591a93f-->
[Automatisches Differenzieren zweiter Ordnung]{.kicker}

<!--en:2ef33b9a-->
- **Ohne die 4 TB große Hesse-Matrix ($n = 10^6$):** Verfahren zweiter Ordnung benötigen nur $\mathbf{H}\mathbf{v} = \mathbf{J}_{\nabla\ell}(\mathbf{x})\mathbf{v} = \left.\frac{d}{dt}\nabla\ell(\mathbf{x} + t\mathbf{v})\right|_{t=0}$.

<!--en:2a7f9002-->
- **Pearlmutters Vorwärts- über Rückwärtsmodus:** Ein JVP im Vorwärtsmodus mit der Anfangstangente $\mathbf{v}$ durchläuft die Gradientenberechnung im Rückwärtsmodus. Das liefert $\mathbf{H}\mathbf{v}$ mit Zeit- und Speicheraufwand $O(n)$ ohne zweites Tape.

<!--en:85b08e80-->
::: {.slide title="HVP im Vorwärts- über Rückwärtsmodus am Beispiel Log-Sum-Exp prüfen" only="pytorch,tensorflow,jax"}

<!--en:f5176dd2-->
[Vergleich mit der geschlossenen Formel]{.kicker}

<!--en:8f7befdf-->
Für Log-Sum-Exp $\ell(\mathbf{x}) = \log\sum_j e^{x_j}$ ist der Gradient $\mathbf{p} = \operatorname{softmax}(\mathbf{x})$ und die Hesse-Matrix $\mathbf{H} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$. Daraus folgt $\mathbf{H}\mathbf{v} = \mathbf{p} \odot \mathbf{v} - \mathbf{p}(\mathbf{p}^\top\mathbf{v})$ mit $3n$ skalaren Multiplikationen:

<!--en:3a726c4f-->
Bei $\mathbf{x} = (1, -0.5, 2)^\top$ und $\mathbf{v} = (1, 0.5, -1)^\top$ liefert der Vorwärts- über Rückwärtsmodus $(0.3571, 0.0514, -0.4085)^\top$. Das stimmt in jeder Stelle mit $\mathbf{p}\odot\mathbf{v} - \mathbf{p}(\mathbf{p}^\top\mathbf{v})$ überein und erfordert keine $3 \times 3$-Matrix.

<!--en:78e43d13-->
::: {.slide title="HVP am Beispiel Log-Sum-Exp prüfen" only="mxnet"}

<!--en:f5176dd2-2-->
[Vergleich mit der geschlossenen Formel]{.kicker}

<!--en:8f7befdf-2-->
Für Log-Sum-Exp $\ell(\mathbf{x}) = \log\sum_j e^{x_j}$ ist der Gradient $\mathbf{p} = \operatorname{softmax}(\mathbf{x})$ und die Hesse-Matrix $\mathbf{H} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$. Daraus folgt $\mathbf{H}\mathbf{v} = \mathbf{p} \odot \mathbf{v} - \mathbf{p}(\mathbf{p}^\top\mathbf{v})$ mit $3n$ skalaren Multiplikationen:

<!--en:b6f0d506-->
MXNet bietet keinen Vorwärtsmodus, und Autograd unterstützt für `exp` keine zweite Ableitung. Daher dient als Näherung für $\mathbf{H}\mathbf{v}$ ein zentraler Differenzenquotient des im Rückwärtsmodus berechneten Gradienten mit der Schrittweite $h=10^{-4}$ in float64. Bei $\mathbf{x} = (1, -0.5, 2)^\top$ und $\mathbf{v} = (1, 0.5, -1)^\top$ ergibt sich $(0.3571, 0.0514, -0.4085)^\top$. Das stimmt mit $\mathbf{p}\odot\mathbf{v} - \mathbf{p}(\mathbf{p}^\top\mathbf{v})$ überein und erfordert keine $3 \times 3$-Matrix.

<!--en:8d284edf-->
::: {.slide title="Implizites Differenzieren bei iterativen Lösern"}

<!--en:e6863620-->
[Satz über implizite Funktionen]{.kicker}

<!--en:fc9c9162-->
- **Gleichgewichtsschichten ($\mathbf{g}(\mathbf{x}^\star, \boldsymbol{\theta}) = \mathbf{0}$):** Ableiten an der Nullstelle ergibt $\frac{\partial\mathbf{x}^\star}{\partial\boldsymbol{\theta}} = -\bigl(\frac{\partial\mathbf{g}}{\partial\mathbf{x}}\bigr)^{-1}\frac{\partial\mathbf{g}}{\partial\boldsymbol{\theta}}$ mit Speicherbedarf $O(1)$ (kein Tape für die entfalteten Schritte des Lösers).

<!--en:9cd03012-->
- **Ein adjungiertes lineares Gleichungssystem:** $\bigl(\frac{\partial\mathbf{g}}{\partial\mathbf{x}}\bigr)^\top\mathbf{a} = \mathbf{u}$ einmal nach $\mathbf{a} \in \mathbb{R}^d$ auflösen, dann $\mathbf{u}^\top\frac{\partial\mathbf{x}^\star}{\partial\boldsymbol{\theta}} = -\mathbf{a}^\top\frac{\partial\mathbf{g}}{\partial\boldsymbol{\theta}}$ bilden.

<!--en:0a39f007-->
::: {.slide title="Impliziter Gradient und Entfaltung von 20 Newton-Schritten"}

<!--en:c43b303c-->
[Vom Löser unabhängige Gradienten]{.kicker}

<!--en:4d2a6e25-->
Für das nichtlineare $2 \times 2$-System $g_1(x, \theta) = x_1^2 + \theta x_2 - 2 = 0$, $g_2(x, \theta) = x_1 x_2 + \theta - 1 = 0$ bei $\theta = 0.3$ genügt es, $-\bigl(\frac{\partial \mathbf{g}}{\partial \mathbf{x}}\bigr)^{-1}\frac{\partial \mathbf{g}}{\partial \theta}$ einmal an der konvergierten Lösung $\mathbf{x}^\star$ zu berechnen. Das Ergebnis stimmt auf 8 Stellen mit zentralen Differenzenquotienten des Newton-Lösers mit 20 Schritten überein:

<!--en:7063b21f-->
Der Speicherbedarf beträgt $O(1)$ in Abhängigkeit von der Schrittzahl $K$ des Lösers. Jeder Löser (Newton-Verfahren, Anderson-Beschleunigung, Fixpunktiteration), der $\mathbf{x}^\star$ erreicht, liefert genau denselben impliziten Gradienten.

<!--en:7bc07111-->
::: {.slide title="Zwei Grenzen des impliziten Differenzierens"}

<!--en:de8134d3-->
[Voraussetzungen]{.kicker}

<!--en:b8430f07-->
- **1. Singuläre Jacobi-Matrix ($\partial g/\partial x = 0$):** Bei $\theta = 0$ in der Gleichung $x^2 - \theta = 0$ führt $1/(2x^\star)$ zur Division durch null.

<!--en:b573b9d4-->
- **2. Fehlende Konvergenz ($\tilde{x}_k \not\to x^\star$):** Bei $k = 1$ und $\theta = 4$ verfehlen sowohl die implizite Ableitung ($0.20$) als auch die Ableitung der entfalteten Iteration ($0.50$) den Wert $0.25$. Erst ab $k \ge 3$ kommen sie ihm nahe.

<!--en:7ad5fa4d-->
::: {.slide title="Schlecht konditionierte Produkte bleiben beim automatischen Differenzieren ein Problem"}

<!--en:9b2d7c33-->
[Ausblick auf die Architektur]{.kicker}

<!--en:eedf1d6a-->
- **Exakte Auswertung eines verschwindend kleinen Werts:** Skalieren 50 Schichten die Aktivierungen jeweils mit $0.5\times$, berechnet der Rückwärtsmodus $0.5^{50} \approx 8.9 \times 10^{-16}$ schnell und genau. Dennoch stockt das Training.

<!--en:d0f98d52-->
- **Abhilfe durch die Architektur:** Damit die Singulärwerte von $\mathbf{J}_K \cdots \mathbf{J}_1$ nahe bei $1$ bleiben, braucht es Xavier- oder He-Initialisierung, LayerNorm oder BatchNorm und residuale Skip-Verbindungen.

<!--en:6ba822ce-->
::: {.slide title="Verständnisfragen zum automatischen Differenzieren"}

<!--en:b6d9a92f-->
[Wiederholung und Übung]{.kicker}

<!--en:2fa9c400-->
Verständnisfragen zu Vorwärts- und Rückwärtsmodus, Checkpointing und implizitem Differenzieren:

<!--en:247cb8b0-->
1. **Wahl des günstigeren Modus:** Ein Encoder $\mathbf{f} : \mathbb{R}^{3} \to \mathbb{R}^{1000}$ bildet 3 Parameter der Kamerapose auf eine gerenderte Spalte mit 1000 Pixeln ab. Gesucht ist seine vollständige $1000 \times 3$-Jacobi-Matrix $\mathbf{J}$. Wie viele Durchläufe brauchen Vorwärts- und Rückwärtsmodus jeweils?

<!--en:b1d28632-->
2. **Speicherbedarf beim Aktivierungs-Checkpointing:** Ein tiefes rekurrentes Netz ist über $T = 10{,}000$ Zeitschritte entfaltet. Wie viele Zwischenzustände muss der gewöhnliche Rückwärtsmodus speichern? Welchen minimalen Spitzenspeicherbedarf $2\sqrt{T}$ erreicht optimales einstufiges Checkpointing?

<!--en:d1992386-->
3. **Adjungiertes Gleichungssystem beim impliziten Differenzieren:** Bei der Ridge-Regression $\mathbf{w}^\star(\boldsymbol{\lambda})$ mit $d = 5000$ Regularisierungsparametern $\lambda_j$ erfordert die direkte Berechnung von $\partial\mathbf{w}^\star/\partial\boldsymbol{\lambda}$ die Lösung von $5000$ linearen Gleichungssystemen. Wie viele lineare Gleichungssysteme sind beim adjungierten Verfahren mit VJP nötig, um den Gradienten des skalaren Validierungsverlusts $V(\mathbf{w}^\star(\boldsymbol{\lambda}))$ nach allen $5000$ Regularisierungsparametern zu berechnen?

<!--en:27050d7c-->
::: {.slide title="Verständnisfragen zum automatischen Differenzieren (Fortsetzung)"}

<!--en:b6d9a92f-2-->
[Wiederholung und Übung]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:98a9757b-->
1. Hier gilt $n = 3 \ll m = 1000$. **Der Vorwärtsmodus braucht nur $3$ JVP-Durchläufe**, einen pro Eingabespalte $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ und ohne Speicher für ein Tape. Der Rückwärtsmodus bräuchte $1000$ VJP-Durchläufe.

<!--en:25ded5b3-->
2. Der gewöhnliche Rückwärtsmodus speichert alle $10{,}000$ Zustände. Ein Checkpoint alle $s^\star = \sqrt{10{,}000} = 100$ Schritte hält höchstens $T/s + s = 100 + 100 = 200$ Zustände im Speicher (**um den Faktor $50\times$ weniger Speicher** für einen zusätzlichen Vorwärtsdurchlauf).

<!--en:0bf63e8e-->
3. **Nur $1$ lineares Gleichungssystem:** Einmal $(\partial\mathbf{g}/\partial\mathbf{w})^\top \mathbf{a} = \nabla_{\mathbf{w}} V$ nach dem adjungierten Vektor $\mathbf{a} \in \mathbb{R}^{5000}$ auflösen. Alle $5000$ Hyperparameterableitungen ergeben sich aus $-\mathbf{a}^\top(\partial\mathbf{g}/\partial\boldsymbol{\lambda})$.

<!--en:f38cfc42-->
::: {.slide title="Zusammenfassung: Modi und Kosten des automatischen Differenzierens"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:060e1ce6-->
| Verfahren | Mathematisches Prinzip | Rechenaufwand und Speicherbedarf |

<!--en:93e98a4f-->
| **Vorwärtsmodus (JVP)** | Duale Zahlen $f(\mathbf{x}+\mathbf{v}\varepsilon) = f(\mathbf{x}) + (\mathbf{J}\mathbf{v})\varepsilon$ ($\varepsilon^2=0$) | $1$ Durchlauf pro Startvektor der Eingabe ($n$ Durchläufe für $\mathbb{R}^n \to \mathbb{R}^m$), zusätzlicher Speicher $O(1)$ |

<!--en:7d67e925-->
| **Rückwärtsmodus (VJP)** | Vorwärts aufgezeichnetes Tape und Rückwärtsdurchlauf $\bar{n} = \sum_c \bar{c}\frac{\partial c}{\partial n}$ | $1$ Durchlauf pro Startvektor der Ausgabe ($1$ Durchlauf für skalaren Verlust, $\sim 3\times$ so viele Gleitkommaoperationen wie vorwärts), Speicher für das Tape $O(T)$ |

<!--en:8ee8d5ac-->
| **Gewichtsgradient einer Schicht** | $\nabla_{\mathbf{W}_k}\ell = \bar{\mathbf{h}}_k \mathbf{h}_{k-1}^\top$, $\bar{\mathbf{h}}_{k-1} = \mathbf{W}_k^\top \bar{\mathbf{h}}_k$ | $2 d_k d_{k-1}$ Multiplikationen rückwärts, $1 d_k d_{k-1}$ vorwärts |

<!--en:644e5d60-->
| **Aktivierungs-Checkpointing** | Jeden $s$-ten Schritt speichern, Segmente der Länge $s$ neu berechnen | Speicher für das Tape sinkt von $T$ auf $T/s + s = 2\sqrt{T}$ bei $s = \sqrt{T}$ (1 zusätzlicher Vorwärtsdurchlauf) |

<!--en:4107acde-->
| **HVP im Vorwärts- über Rückwärtsmodus** | $\mathbf{H}\mathbf{v} = \mathbf{J}_{\nabla\ell}\mathbf{v} = \frac{d}{dt}\nabla\ell(\mathbf{x}+t\mathbf{v})\big|_{t=0}$ | Exaktes Hesse-Vektor-Produkt in $O(1)$ Durchläufen ohne den Speicherbedarf $O(n^2)$ der Hesse-Matrix |

<!--en:4e8aa0fe-->
| **Implizites Differenzieren** | $\mathbf{g}(\mathbf{x}^\star, \boldsymbol{\theta})=\mathbf{0}$ $\Rightarrow$ $(\frac{\partial\mathbf{g}}{\partial\mathbf{x}})^\top\mathbf{a} = \mathbf{u}$, $\nabla_{\boldsymbol{\theta}}\ell = -\mathbf{a}^\top\frac{\partial\mathbf{g}}{\partial\boldsymbol{\theta}}$ | Ableitung an der konvergierten Nullstelle: $1$ lineares Gleichungssystem, Speicherbedarf $O(1)$ |
