<!--en:f4c72ea5-->
# Matrixdifferentialrechnung

<!--en:73790c9f-->
In den letzten beiden Abschnitten haben wir den Verlust (engl. loss) als Funktion skalarer Variablen betrachtet. In :numref:`sec_mdl-multivariable_calculus` haben wir seine partiellen Ableitungen zu einer Spalte zusammengefasst, dem Gradienten. In :numref:`sec_mdl-chain-rule-graphs` haben wir diese Spalte durch einen Rückwärtsdurchlauf (engl. backward pass) im Graphen berechnet, mit einer Zahl pro Knoten. Ein neuronales Netz schreiben wir jedoch in anderer Form. Jede Schicht bildet einen Vektor auf einen Vektor ab, und ihre Gewichte bilden eine Matrix. Was ist die Ableitung eines Verlusts nach einem Vektor oder einer Matrix? Wie berechnen wir sie für eine ganze Schicht auf einmal?

<!--en:a9950bb3-->
An einem Netz, das wir von Hand durchrechnen können, sehen wir, was wir brauchen. Betrachten wir ein Netz mit zwei Eingaben, zwei versteckten ReLU-Einheiten und einer Ausgabe:

<!--en:3d87584d-->
Dabei wendet $\boldsymbol\varphi$ die Funktion $\max(0,\cdot)$ auf jeden Eintrag an. An der Stelle $\mathbf x=(1,1)$ sind die Präaktivierungen $\mathbf W_1\mathbf x+\mathbf b=(3,-2)$. Die zweite versteckte Einheit ist also inaktiv, der Vektor der versteckten Aktivierungen ist $(3,0)$ und die Ausgabe beträgt $g=6$. Für einen Trainingsschritt stellen sich drei Fragen zu dieser Ausgabe. Erstens: Ändert sich die Eingabe um ein kleines $\boldsymbol\delta$, so ändert sich die Ausgabe um $2\delta_1+4\delta_2$. Die beiden Koeffizienten bilden eine Zeile, die Transponierte des als Spalte geschriebenen Gradienten. Zweitens: Ändert sich die Gewichtsmatrix um $\Delta\mathbf W_1$, so ändert sich die Ausgabe um $2\,\Delta W_{11}+2\,\Delta W_{12}$. Die zweite Zeile trägt nichts dazu bei. Die Ableitung nach einer Matrix ist also selbst eine Matrix, hier $\left(\begin{smallmatrix}2&2\\0&0\end{smallmatrix}\right)$. Drittens: Die versteckte Schicht $\mathbf x\mapsto\boldsymbol\varphi(\mathbf W_1\mathbf x+\mathbf b)$ bildet einen Vektor auf einen Vektor ab. Weder eine skalare Steigung noch ein Gradientenvektor beschreibt ihre Ableitung. Dabei hängt die Ausgabe nur über diese Schicht von $\mathbf W_1$ ab.

<!--en:9896c009-->
Bei solchen Ableitungen unterlaufen einem von Hand überraschend leicht Fehler. Vergisst man bei einer quadratischen Gewichtsmatrix die Transposition, passen die Dimensionen weiterhin zusammen. Eine Fehlermeldung bleibt aus, und das Training läuft in die falsche Richtung. Formelsammlungen für Matrixableitungen umfassen Hunderte von Einträgen. Gegen solche Versehen hilft das Nachschlagen aber wenig.

<!--en:1fb10f31-->
Alle drei Fragen lassen sich mit einem einzigen mathematischen Objekt und einem einzigen Prinzip beantworten. Die Ableitung einer vektorwertigen Abbildung ist eine Matrix, die *Jacobi-Matrix* (engl. Jacobian). Wir definieren sie über die beste lineare Approximation der Abbildung. Verketten wir Abbildungen, multiplizieren sich ihre Jacobi-Matrizen. Die Ableitung eines Netzes ist somit ein Produkt aus je einer Matrix pro Schicht. Für alle Gradienten, die eine Schicht benötigt, genügt dann dieselbe Methode, ob für lineare und quadratische Formen, die Methode der kleinsten Quadrate, Matrixprodukte oder die Softmax-Funktion. Wir schreiben die Änderung der Funktion in erster Ordnung auf und lesen den Gradienten als ihren Koeffizienten ab. So erhalten wir den Rückwärtsdurchlauf jeder Schicht aus :numref:`sec_backprop` und auch $\mathbf p-\mathbf y$, den Gradienten des Klassifikators aus :numref:`subsec_softmax_and_derivatives`.

<!--en:da468fe6-->
Drei Ergebnisse sollten Sie sich merken: das Produkt der Jacobi-Matrizen :eqref:`eq_mdl-jacobian-product`, die Regel, dass ein Gradient immer die Form seiner Variablen hat, und die Rückwärtsregel :eqref:`eq_mdl-matmul-vjp` für ein Matrixprodukt. Den Beweis der Kettenregel können Sie beim ersten Lesen überspringen. Das Rechnen mit Differentialen (:numref:`subsec_mdl-differential-method`) brauchen Sie jedoch für alles Weitere.

<!--en:857f9054-->
Im gesamten Abschnitt verwenden wir die folgenden Importe.

<!--en:85828374-->
## Die Jacobi-Matrix

<!--en:034d0cbc-->
### Die beste lineare Approximation

<!--en:e8b70472-->
In der Nähe von $\mathbf x=(1,1)$ ist die versteckte Schicht unseres Netzes affin. Ihre lineare Approximation ist dort exakt, es gibt also keinen Fehler zu untersuchen. Beginnen wir deshalb mit einer nichtlinearen Abbildung, die uns hier und im nächsten Abschnitt als durchgängiges Beispiel dient. Sobald die Definition steht, kehren wir zum Netz zurück. Unsere Abbildung lautet:

<!--en:57c9a288-->
An der Stelle $\mathbf x_0$ hat die Abbildung den Wert $\mathbf f(\mathbf x_0)=(0.9794,\ 0.6250)$. Jede der beiden Ausgaben hat zwei partielle Ableitungen: $\partial f_1/\partial x=1$, $\partial f_1/\partial y=\cos y=0.8776$, $\partial f_2/\partial x=x=0.5$ und $\partial f_2/\partial y=1$. Wir ordnen diese vier Zahlen in einer Matrix an, mit einer Zeile je Ausgabe und einer Spalte je Eingabe:

<!--en:fec0a449-->
Spalte $j$ gibt an, wie sich beide Ausgaben ändern, wenn sich die Eingabe $j$ ändert. Diese Matrix ist die Ableitung von $\mathbf f$ an der Stelle $\mathbf x_0$ in folgendem Sinn: Es gilt $\mathbf f(\mathbf x_0+\boldsymbol\delta)\approx\mathbf f(\mathbf x_0)+\mathbf J\boldsymbol\delta$ für kleine $\boldsymbol\delta$, und keine andere Matrix approximiert $\mathbf f$ so gut.

<!--en:5b6eb18a-->
Bei einer Variablen ist die Ableitung die Steigung der Geraden, die die Funktion lokal am besten annähert: $f(x+\delta)\approx f(x)+f'(x)\,\delta$. Die Verallgemeinerung auf vektorwertige Funktionen beruht auf derselben lokalen Approximation. Wir nennen $\mathbf f:\mathbb R^n\to\mathbb R^m$ an der Stelle $\mathbf x$ *differenzierbar*, wenn es eine Matrix $\mathbf J\in\mathbb R^{m\times n}$ gibt, für die gilt:

<!--en:d2d1660b-->
Das heißt, der Fehler des linearen Modells $\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta$, geteilt durch $\|\boldsymbol\delta\|$, geht für $\boldsymbol\delta\to\mathbf 0$ gegen null. Existiert eine solche Matrix, so ist sie eindeutig. Wir nennen sie die *Jacobi-Matrix* $\mathbf J_{\mathbf f}(\mathbf x)$. Ihre Einträge sind die partiellen Ableitungen:

<!--en:03a5d172-->
Um das zu sehen, setzen wir die Störung $\boldsymbol\delta=t\,\mathbf e_j$ (einen kleinen Schritt entlang der Koordinatenachse $j$) in :eqref:`eq_mdl-jacobian-linearization` ein und betrachten die Komponente $i$. Teilen wir durch $t$ und betrachten den Grenzübergang $t\to0$, so erhalten wir $\partial f_i/\partial x_j$ als Eintrag an der Stelle $(i,j)$, unabhängig von der Matrix $\mathbf J$, von der wir ausgegangen sind. Die partiellen Ableitungen liefern also die Einträge von $\mathbf J$. Definiert ist die Matrix über die lineare Approximation.

<!--en:5c101401-->
Die Existenz partieller Ableitungen allein garantiert noch keine Differenzierbarkeit. Betrachten wir die Funktion $f(x,y)=2xy/(x^2+y^2)$ mit $f(0,0)=0$ aus :numref:`subsec_mdl-gradient`. Beide partiellen Ableitungen sind im Ursprung gleich $0$. Doch $f$ hat den Wert $1$ auf der gesamten Diagonalen $x=y$ mit Ausnahme des Ursprungs. Als Jacobi-Matrix kommt nur der Nullzeilenvektor infrage. Sein Approximationsfehler entlang der Diagonalen bleibt $1$, wie klein $\boldsymbol\delta$ auch ist. Daher hat $f$ im Ursprung keine Jacobi-Matrix. Stetige partielle Ableitungen schließen ein solches Verhalten aus. Existiert jede partielle Ableitung $\partial f_i/\partial x_j$ in einer Umgebung von $\mathbf x$ und ist sie an der Stelle $\mathbf x$ stetig, dann ist $\mathbf f$ an der Stelle $\mathbf x$ differenzierbar :cite:`Rudin.1976`.

<!--en:d6346454-->
:numref:`fig_mdl-cal-jacobian-ellipse` veranschaulicht die Definition geometrisch. In der Nähe von $\mathbf x_0$ wirkt die Abbildung wie ihre Jacobi-Matrix. Das Bild eines Kreises mit Radius $r$ um $\mathbf x_0$ liegt nahe bei der Ellipse $\mathbf f(\mathbf x_0)+\mathbf J\boldsymbol\delta$ ($\|\boldsymbol\delta\|=r$). Deren Halbachsen sind $r$-mal so lang wie die Singulärwerte von $\mathbf J$. Wie :numref:`sec_mdl-svd-low-rank` zeigt, bildet jede Matrix den Einheitskreis auf eine Ellipse mit den entsprechenden Singulärwerten als Halbachsenlängen ab. Hier sind die Singulärwerte $1.706$ und $0.329$. Aus dem Kreis mit Radius $0.35$ wird also näherungsweise eine Ellipse mit Halbachsen der Längen $0.60$ und $0.12$. Die größte Abweichung des tatsächlichen Bildes von der Ellipse beträgt $0.061$ bei $r=0.35$ und $0.015$ bei $r=0.175$. Halbieren wir den Radius, so sinkt die Abweichung auf ein Viertel.

<!--en:6e36b754-->
![Lokal verhält sich eine differenzierbare Abbildung wie ihre Jacobi-Matrix. Links zwei Kreise mit den Radien $0.35$ und $0.175$ um $\mathbf x_0=(0.5,0.5)$, rechts ihre Bilder unter $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ (blau) im Vergleich zu den Ellipsen $\mathbf f(\mathbf x_0)+\mathbf J(\mathbf x_0)\boldsymbol\delta$ (orange gestrichelt). Die Halbachsenlängen jeder Ellipse ergeben sich aus dem jeweiligen Radius mal den Singulärwerten $1.71$ und $0.33$ von $\mathbf J$. Bei halbem Radius sinkt die größte Abweichung zwischen Bild und Ellipse von $0.061$ auf $0.015$. Sie entspricht dem Restglied $o(\|\boldsymbol\delta\|)$.](../img/mdl-cal-jacobian-ellipse.svg)

<!--en:52fa1559-->
Die Definition garantiert, dass der Fehler des linearen Modells geteilt durch $\|\boldsymbol\delta\|$ gegen null geht. Wie schnell das geschieht, lässt sie offen. Verkleinern wir die Störung wiederholt um den Faktor zehn und betrachten wir diesen Quotienten.

<!--en:c916787b-->
Der Quotient fällt von $0.09$ auf $0.009$, $0.0009$ und $0.00009$. Er geht gegen null und bestätigt damit das Restglied $o(\|\boldsymbol\delta\|)$ aus der Definition. Verkleinern wir $\|\boldsymbol\delta\|$ um den Faktor zehn, fällt auch der Quotient auf ein Zehntel. Der Fehler selbst sinkt also auf ein Hundertstel und ist von der Ordnung $O(\|\boldsymbol\delta\|^2)$. Die Definition garantiert nur die erste Eigenschaft. Das schnellere quadratische Abklingen gilt, weil $\mathbf f$ stetige zweite Ableitungen hat.

<!--en:aaee95f0-->
### Gradienten, Hesse-Matrizen und die Form einer Ableitung

<!--en:99eab6cd-->
Sowohl der Gradient als auch die Hesse-Matrix (engl. Hessian) aus :numref:`sec_mdl-multivariable_calculus` sind Spezialfälle der Jacobi-Matrix. Aus dieser Sicht klärt sich eine Frage nach der Form, die bei jeder Herleitung von Hand auftritt: Ist die Ableitung eine Zeile oder eine Spalte, und wie ordnen wir die Ableitung nach einer Matrix an?

<!--en:3c0fda6b-->
Für eine skalarwertige Funktion $f:\mathbb R^n\to\mathbb R$ ($m=1$) besteht die Jacobi-Matrix aus einer einzigen Zeile, $\partial f/\partial\mathbf x=[\partial f/\partial x_1,\ldots,\partial f/\partial x_n]$. Damit wird :eqref:`eq_mdl-jacobian-linearization` zur Entwicklung erster Ordnung $f(\mathbf x+\boldsymbol\delta)\approx f(\mathbf x)+(\partial f/\partial\mathbf x)\,\boldsymbol\delta$. In :numref:`sec_mdl-multivariable_calculus` haben wir den Gradienten als Spaltenvektor geschrieben. Die beiden Formen sind zueinander transponiert:

<!--en:a2b3c086-->
In diesem Buch verwenden wir beide Formen, jede für ihren eigenen Zweck. Jacobi-Matrizen vektorwertiger Abbildungen haben die Form $m\times n$ mit einer Zeile pro Ausgabe. Damit kommt die folgende Kettenregel beim Verketten ohne Transpositionen aus. Gradienten skalarer Funktionen schreiben wir als Spaltenvektoren, $\nabla_{\mathbf x}f$. Sie haben dieselbe Form wie $\mathbf x$, sodass ein Optimierer sie direkt subtrahieren kann. Im Netz vom Anfang ist $\partial g/\partial\mathbf x=(2,\ 4)$ eine $1\times2$-Jacobi-Matrix, während $\nabla_{\mathbf x}g=(2,4)^\top$ den Gradienten als Spalte angibt.

<!--en:09191df4-->
Für eine skalare Funktion einer Matrix $\mathbf W\in\mathbb R^{p\times q}$ brauchen wir eine weitere Konvention. Wir definieren ihren Gradienten $\nabla_{\mathbf W}f$ als die $p\times q$-Matrix mit den Einträgen $\partial f/\partial W_{ij}$. Er hat also dieselbe Form wie $\mathbf W$. Die Entwicklung erster Ordnung lautet dann

<!--en:6ed0c8ac-->
Dabei multipliziert das innere Produkt $\langle\cdot,\cdot\rangle$ zwei Matrizen gleicher Form eintragsweise und summiert die Produkte. Das entspricht dem gewöhnlichen Skalarprodukt, wenn wir die Matrizen als Vektoren schreiben. In Worten: Die Änderung von $f$ in erster Ordnung ergibt sich, indem wir „Empfindlichkeit mal Änderung“ über alle Einträge summieren. In unserem Eingangsbeispiel ist $\nabla_{\mathbf W_1}g$ die $2\times2$-Matrix $\left(\begin{smallmatrix}2&2\\0&0\end{smallmatrix}\right)$, und $\langle\nabla_{\mathbf W_1}g,\Delta\mathbf W_1\rangle=2\,\Delta W_{11}+2\,\Delta W_{12}$ ergibt wieder die zuvor berechnete Änderung. Damit können wir jede Formel in diesem Abschnitt auf Plausibilität prüfen: Ein Gradient hat immer die Form seiner Variablen. Ist das Ergebnis einer Herleitung von anderer Form, fehlt eine Transposition.

<!--en:485f2187-->
Leiten wir noch einmal ab, erhalten wir die Hesse-Matrix. Das Gradientenfeld $\nabla f:\mathbb R^n\to\mathbb R^n$ ist selbst eine vektorwertige Abbildung. Seine Jacobi-Matrix ist die Matrix der zweiten partiellen Ableitungen, die Hesse-Matrix $\mathbf H_f$ mit den Einträgen $[\mathbf H_f]_{ij}=\partial^2 f/\partial x_i\partial x_j$. Streng genommen stehen die beiden Ableitungen in $[\mathbf J_{\nabla f}]_{ij}=\partial^2 f/\partial x_j\partial x_i$ in umgekehrter Reihenfolge. Sind die gemischten partiellen Ableitungen stetig, stimmen beide Reihenfolgen überein und $\mathbf H_f$ ist symmetrisch (Satz von Schwarz, auch Satz von Clairaut genannt, aus :numref:`sec_mdl-multivariable_calculus`).

<!--en:c2908534-->
## Die Kettenregel für Jacobi-Matrizen

<!--en:97899d81-->
Ein neuronales Netz ist eine Verkettung von Schichten. In einem Graphen mit skalaren Knoten liefert die Kettenregel für jeden Pfad von einer Eingabe zu einer Ausgabe ein Produkt von Ableitungen und summiert diese Produkte über alle Pfade (:numref:`subsec_mdl-chain-rule-paths`). Sind die Schichten vektorwertige Abbildungen, so wird aus dieser Pfadsumme ein Matrixprodukt.

<!--en:98da1538-->
### Verkettung als Matrixprodukt

<!--en:58801f19-->
**Satz (Kettenregel).** *Ist $\mathbf f:\mathbb R^n\to\mathbb R^p$ an der Stelle $\mathbf x$ differenzierbar und ist $\mathbf g:\mathbb R^p\to\mathbb R^m$ an der Stelle $\mathbf f(\mathbf x)$ differenzierbar, so ist $\mathbf g\circ\mathbf f$ an der Stelle $\mathbf x$ differenzierbar, und es gilt*

<!--en:70d285b2-->
**Beweis.** Schreiben wir $\mathbf y=\mathbf f(\mathbf x)$, $\mathbf A=\mathbf
J_{\mathbf f}(\mathbf x)$ und $\mathbf B=\mathbf J_{\mathbf g}(\mathbf y)$. Wir setzen das lineare Modell für $\mathbf f$ in das lineare Modell für $\mathbf g$ ein. Nach :eqref:`eq_mdl-jacobian-linearization` gilt $\mathbf f(\mathbf x+\boldsymbol\delta)=\mathbf y+\mathbf A\boldsymbol\delta
+o(\|\boldsymbol\delta\|)$. Bezeichnen wir diese Verschiebung mit $\boldsymbol\eta=\mathbf A\boldsymbol\delta+o(\|\boldsymbol\delta\|)$ und setzen sie in $\mathbf g$ ein, so ergibt sich

<!--en:f9e388d4-->
Die zweite Gleichheit beruht auf zwei Tatsachen. Erstens bleibt ein Restglied der Ordnung $o(\|\boldsymbol\delta\|)$ nach Multiplikation mit der festen Matrix $\mathbf B$ von der Ordnung $o(\|\boldsymbol\delta\|)$. Zweitens zeigt $\|\boldsymbol\eta\|\le\|\mathbf A\|\,\|\boldsymbol\delta\|+o(\|\boldsymbol\delta\|)$, dass die Länge der Verschiebung durch ein konstantes Vielfaches von $\|\boldsymbol\delta\|$ beschränkt ist. Daher ist das Restglied $o(\|\boldsymbol\eta\|)$ aus $\mathbf g$ auch von der Ordnung $o(\|\boldsymbol\delta\|)$. Nur dieser Schritt geht über algebraische Umformungen hinaus. Der Koeffizient von $\boldsymbol\delta$ ist somit eine Matrix, die :eqref:`eq_mdl-jacobian-linearization` erfüllt. Wegen der Eindeutigkeit ist sie die Jacobi-Matrix der Verkettung: $\mathbf J_{\mathbf g\circ\mathbf f}=\mathbf B\mathbf A$. $\blacksquare$

<!--en:45fafdb2-->
An einem einzelnen Eintrag des Produkts erkennen wir die skalare Pfadsumme wieder. Der Ausdruck $[\mathbf B\mathbf A]_{ik}=\sum_j (\partial g_i/\partial y_j)(\partial f_j/\partial x_k)$ summiert über alle Zwischenvariablen $y_j$ das Produkt der Ableitungen entlang des jeweiligen Pfads von $x_k$ nach $g_i$.

<!--en:8b504874-->
### Die Jacobi-Matrix eines Netzes

<!--en:68938a3f-->
Kehren wir zu unserem Eingangsbeispiel zurück. Das Netz verkettet drei Abbildungen: die affine Transformation $\mathbf a=\mathbf W_1\mathbf x+\mathbf b$, die ReLU-Aktivierung $\mathbf h=\boldsymbol\varphi(\mathbf a)$ und die lineare Ausgabe $g=\mathbf w_2^\top\mathbf h$. Die erste Abbildung ist bereits affin, ihre Jacobi-Matrix ist daher $\mathbf W_1$. Der Bias verschiebt den Wert, ohne die Steigung zu ändern. Die ReLU-Funktion wirkt auf jeden Eintrag einzeln, mit Steigung $1$ für $a_i>0$ und $0$ für $a_i<0$. Ihre Jacobi-Matrix ist daher die Diagonalmatrix $\operatorname{diag}(\mathbf m)$, gebildet aus der *Maske* $\mathbf m$ dieser Steigungen. Die Ausgabeabbildung hat schließlich die Jacobi-Matrix $\mathbf w_2^\top$. An der Stelle $\mathbf x=(1,1)$ gilt $\mathbf a=(3,-2)$ und $\mathbf m=(1,0)$. Die Kettenregel multipliziert die drei Faktoren:

<!--en:1ead6a0f-->
Die Maske setzt den zweiten Eintrag von $\mathbf w_2^\top$ auf null, sodass die zweite Zeile von $\mathbf W_1$ die Ausgabe nie erreicht. Die Gewichte einer inaktiven Einheit tragen nichts zur Ableitung bei (:numref:`fig_mdl-cal-relu-net-jacobian`). Die resultierende Zeile $(2,4)$ entspricht der Änderung $2\delta_1+4\delta_2$ aus dem Eingangsbeispiel. Bei einer anderen Eingabe kann sich die Maske und damit die Jacobi-Matrix ändern. An der Stelle $\mathbf x=(2,-3)$ sind die Präaktivierungen beispielsweise $(-4,5)$, die Maske wechselt zu $(0,1)$ und es gilt $\mathbf J_g=(0,\ -1)\,\mathbf W_1=(-3,\ 1)$. Da die ReLU-Funktion stückweise linear ist, liefert ein Differenzenquotient entlang jeder Achse diese Einträge exakt, solange die Schrittweite klein genug ist, dass die Maske gleich bleibt (Aufgabe :exref:`ex_mdl-matrix-calculus-relu-jacobian`).

<!--en:e51639d5-->
![Die Jacobi-Matrix des 2-2-1-ReLU-Netzes an der Stelle $\mathbf x=(1,1)$ ist das Produkt dreier Matrizen. Die zweite Einheit hat die Präaktivierung $-2$ und damit den Maskeneintrag $0$. Die zweite Zeile von $\mathbf W_1$ erreicht die Ausgabe daher nie: $\mathbf J_g=\mathbf w_2^\top\operatorname{diag}(1,0)\,\mathbf W_1=(2,\ 4)$.](../img/mdl-cal-relu-net-jacobian.svg)

<!--en:20bd6e44-->
Jedes mehrschichtige Perzeptron besteht aus denselben zwei Arten von Faktoren. Eine vollständig verbundene Schicht $\mathbf x\mapsto\mathbf W\mathbf x$ hat die Jacobi-Matrix $\mathbf W$. Eine elementweise Aktivierung $\boldsymbol\varphi(\mathbf x)=(\varphi(x_1),\ldots,\varphi(x_n))$ koppelt verschiedene Koordinaten nicht miteinander. Ihre Jacobi-Matrix ist daher diagonal: $\mathbf J_{\boldsymbol\varphi}(\mathbf x)=\operatorname{diag}\bigl(\varphi'(x_1),\ldots,\varphi'(x_n)\bigr)$. Das setzt voraus, dass $\varphi$ an jeder Stelle $x_i$ differenzierbar ist. Die ReLU-Funktion hat bei $0$ keine Ableitung. Dort setzen die Frameworks nach Konvention einen festen Wert ein (:numref:`sec_mdl-tangent-fails`). Ein Netz der Tiefe $K$ ist eine Verkettung $\mathbf f=\mathbf f_K\circ\cdots\circ\mathbf f_1$. Wenden wir :eqref:`eq_mdl-jacobian-chain` wiederholt an, erhalten wir für seine Ableitung ein einziges Produkt:

<!--en:407c25d6-->
Dabei werten wir jeden Faktor an der Eingabe seiner jeweiligen Schicht aus. Wechseln sich vollständig verbundene Schichten und Aktivierungen ab, ergibt sich $\mathbf J=\operatorname{diag}(\boldsymbol\varphi_K')\,\mathbf W_K\cdots
\operatorname{diag}(\boldsymbol\varphi_1')\,\mathbf W_1$. Gewichtsmatrizen wechseln sich also mit Diagonalmatrizen ab. Die Multiplikation eines Vektors mit einer Diagonalmatrix skaliert jeden Eintrag einzeln. Jeder Aktivierungsfaktor kostet daher nur $O(n)$ statt $O(n^2)$ Operationen.

<!--en:d9a0f71e-->
Definieren wir nun das 2-2-1-Netz im Code. Wir verwenden es später in diesem Abschnitt erneut. Vergleichen wir das Produkt $\mathbf w_2^\top\operatorname{diag}(\mathbf m)\mathbf W_1$ mit dem Gradienten, den automatisches Differenzieren liefert.

<!--en:d7a68f3f-->
Beide Zeilen geben wie erwartet $(2,4)$ aus.

<!--en:2baa70c8-->
Das Produkt :eqref:`eq_mdl-jacobian-product` zeigt auch, wie stark eine Ableitung in einem tiefen Netz wachsen oder schrumpfen kann. Ist jeder Faktor dieselbe Matrix $\mathbf W$, vereinfacht sich das Produkt zu $\mathbf W^K$. Das gilt etwa für ein rekurrentes Netz, das in jedem Schritt dieselbe Gewichtsmatrix anwendet, wenn wir die Aktivierungen außer Acht lassen. Für eine diagonalisierbare Matrix $\mathbf W=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$ gilt $\mathbf W^K=\mathbf V\boldsymbol\Lambda^K\mathbf V^{-1}$. Für große $K$ wächst oder schrumpft die Norm dieser Potenz daher wie $|\lambda_{\max}|^K$, wobei $\lambda_{\max}$ der betragsgrößte Eigenwert ist. So gilt $0.9^{50}\approx0.005$, aber $1.1^{50}\approx117$. Diese exponentielle Abhängigkeit führt in tiefen und rekurrenten Netzen zum Problem verschwindender und explodierender Gradienten (:numref:`sec_numerical_stability`, :numref:`sec_bptt`). Dabei beschreibt $|\lambda_{\max}|^K$ nur die langfristige Rate. Sind die Eigenvektoren in $\mathbf V$ weit davon entfernt, orthogonal zu sein, kann das Produkt viele Schritte lang wachsen, bevor es schließlich abklingt (:numref:`subsec_mdl-transient-growth`). Bei einem Produkt verschiedener Faktoren sind die Singulärwerte anstelle der Eigenwerte maßgeblich (:numref:`subsec_mdl-jacobian-products`).

<!--en:9b2e77ec-->
Für das Training hat :eqref:`eq_mdl-jacobian-product` allerdings eine wichtige Einschränkung: Die Formel liefert die Ableitung nach der *Eingabe*. Die Gewichtsmatrizen $\mathbf W_k$ gehen als feste Faktoren ein. Wie sich die Ausgabe ändert, wenn sich ein Gewicht ändert, können wir daraus noch nicht ablesen. Für die Gewichtsgradienten brauchen wir für jede Schicht eine weitere Regel. In :numref:`subsec_mdl-layer-vjp` leiten wir sie her und wenden sie auf dasselbe Netz an.

<!--en:79443745-->
### Die Reihenfolge der Multiplikation

<!--en:a3fdaf4e-->
Die Matrizenmultiplikation ist assoziativ. Wir können das Produkt :eqref:`eq_mdl-jacobian-product` also beliebig klammern und erhalten immer dieselbe Matrix. Die Kosten verschiedener Klammerungen unterscheiden sich jedoch um einen Faktor, der mit der Breite des Netzes wächst und sehr groß werden kann. Beim Training suchen wir den Gradienten eines skalaren Verlusts $\ell$. Die Kette beginnt links daher mit einem Zeilenvektor, der $1\times h$-Jacobi-Matrix $\mathbf u^\top=\partial\ell/\partial\mathbf h_K$ des Verlusts bezüglich der Ausgabe $\mathbf h_K$ der letzten Schicht. Multiplizieren wir von links, bleibt jedes Zwischenergebnis ein Zeilenvektor. Jeder Schritt ist also ein Matrix-Vektor-Produkt. Multiplizieren wir von rechts, entstehen bis zum letzten Schritt teure Matrix-Matrix-Produkte. Erst dann kommt der Zeilenvektor ins Spiel. Günstig ist die Reihenfolge, bei der jeder Schritt ein Matrix-Vektor-Produkt bleibt.

<!--en:4106138b-->
Zählen wir die Operationen, wird der Unterschied greifbar. Betrachten wir $\mathbf u^\top\mathbf J_3\mathbf J_2\mathbf J_1$ mit den Formen $1\times h$, $h\times h$, $h\times h$ und $h\times n$ für $h=n=1000$. Zur Erinnerung: Das Produkt einer $p\times q$-Matrix mit einer $q\times r$-Matrix kostet $pqr$ Multiplikationen. Von links berechnen wir drei Matrix-Vektor-Produkte mit insgesamt $h^2+h^2+hn=3\times10^6$ Multiplikationen. Von rechts kosten $\mathbf J_2\mathbf J_1$ und anschließend $\mathbf J_3(\mathbf J_2\mathbf J_1)$ jeweils $h^2n=10^9$ Multiplikationen. Das letzte Produkt mit $\mathbf u^\top$ erfordert weitere $hn=10^6$, insgesamt also $2.001\times10^9$ Multiplikationen. Drei Millionen gegenüber zwei Milliarden entsprechen einem Faktor von rund $670$ beziehungsweise $2h/3$ (:numref:`fig_mdl-cal-jacobian-chain-order`).

<!--en:c504c6ba-->
![Zwei Reihenfolgen für $\mathbf u^\top\mathbf J_3\mathbf J_2\mathbf J_1$ mit $h=n=1000$. Von links ist jedes Zwischenergebnis eine $1\times h$-Zeile und jeder Schritt ein Matrix-Vektor-Produkt. Das erfordert $3\times10^6$ Multiplikationen. Von rechts sind die ersten beiden Schritte Matrix-Matrix-Produkte mit je $10^9$ Multiplikationen. Bei gleicher Tiefe entscheiden die Formen über die Kosten.](../img/mdl-cal-jacobian-chain-order.svg)

<!--en:c4f1a8a7-->
Diese beiden Reihenfolgen entsprechen zwei Richtungen durch das Netz. Die Multiplikation von links propagiert die Empfindlichkeit der Ausgabe rückwärts durch die Schichten und kostet pro Schicht ein Matrix-Vektor-Produkt. Steht am rechten Ende der Kette ein Spaltenvektor $\mathbf v$, ist die Multiplikation von rechts günstig. Sie propagiert eine einzelne Eingaberichtung vorwärts und liefert $\mathbf J\mathbf v$. Die Algorithmen, die diese beiden Reihenfolgen automatisieren, heißen automatisches Differenzieren im Rückwärtsmodus beziehungsweise im Vorwärtsmodus (:numref:`sec_mdl-matrix-calculus-autodiff`).

<!--en:38e2c0b9-->
Bei der Zählung haben wir Schichten gleicher Breite angenommen. Haben die Faktoren unterschiedliche Formen, kann eine andere Reihenfolge günstiger sein als beide durchgängigen Richtungen. Für eine Kette lässt sich die günstigste Klammerung durch dynamische Programmierung bestimmen (Aufgabe :exref:`ex_mdl-matrix-calculus-chain-order`). Für einen allgemeinen Berechnungsgraphen ist das Problem NP-vollständig :cite:`Naumann.2008`.

<!--en:0029658b-->
## Ableitungen nach Vektoren und Matrizen

<!--en:3de79ba0-->
Beim Training einer Schicht brauchen wir den Gradienten eines skalaren Verlusts sowohl nach Vektoren als auch nach Matrizen: nach der Eingabe, den Gewichten und dem Bias der Schicht. In diesem Abschnitt leiten wir alle benötigten Gradienten mit einer einzigen Methode her. Jedes Ergebnis hat dabei automatisch die Form seiner Variablen.

<!--en:43032a0d-->
### Rechnen mit Differentialen

<!--en:0fef6d9f-->
Schreiben wir $d\mathbf x$ für eine kleine Störung der Variablen und $df$ für den Anteil der resultierenden Änderung von $f$, der linear in $d\mathbf x$ ist. Das ist der Term $\mathbf J\boldsymbol\delta$ aus :eqref:`eq_mdl-jacobian-linearization`, in dem wir $\boldsymbol\delta$ in $d\mathbf x$ umbenennen. Nach :eqref:`eq_mdl-jacobian-linearization` und :eqref:`eq_mdl-matrix-gradient` ist dieser lineare Anteil ein inneres Produkt mit dem Gradienten:

<!--en:039cd052-->
Das gilt sowohl für Vektorvariablen ($\langle\mathbf a,\mathbf b\rangle=\mathbf a^\top\mathbf b$) als auch für Matrixvariablen. Wir können :eqref:`eq_mdl-differential-rule` auf zwei Arten lesen. Kennen wir den Gradienten, so erfahren wir, wie $f$ auf eine kleine Änderung reagiert. Kennen wir $f$, so können wir umgekehrt direkt den Gradienten bestimmen. Dazu berechnen wir $df$ mit den gewöhnlichen Ableitungsregeln, bringen es in die Form $\langle\mathbf M,\,d\mathbf x\rangle$ und lesen $\nabla f=\mathbf M$ ab. Das Ergebnis ist eindeutig, denn eine lineare Funktion von $d\mathbf x$ lässt sich nur für eine einzige Matrix als $\langle\mathbf M,d\mathbf x\rangle$ schreiben. Um diese Matrix $\mathbf M$ zu bestimmen, setzen wir für $d\mathbf x$ eine Matrix ein, die eine einzelne $1$ an der Stelle $(i,j)$ und sonst nur Nullen enthält. So erhalten wir den Eintrag $M_{ij}$.

<!--en:d7cf8408-->
Vier vertraute Regeln reichen aus, um $df$ zu berechnen. Erstens ist das Differential linear. Zweitens gilt die Produktregel $d(\mathbf A\mathbf B)=d\mathbf A\,\mathbf B+\mathbf A\,d\mathbf B$, denn beim Ausmultiplizieren von $(\mathbf A+d\mathbf A)(\mathbf B+d\mathbf B)$ ergibt sich gegenüber $\mathbf A\mathbf B+d\mathbf A\,\mathbf B+\mathbf A\,d\mathbf B$ nur der zusätzliche Term zweiter Ordnung $d\mathbf A\,d\mathbf B$. Drittens sind Differentialbildung und Transposition vertauschbar: $d(\mathbf A^\top)=(d\mathbf A)^\top$. Viertens gilt bei einer eintragsweise angewandten skalaren Funktion die skalare Kettenregel für jeden Eintrag: $d\varphi(\mathbf X)=\varphi'(\mathbf X)\odot d\mathbf X$. Dabei bezeichnet $\odot$ die elementweise Multiplikation.

<!--en:4c3fabeb-->
Um $df$ in die Form $\langle\mathbf M,d\mathbf x\rangle$ zu bringen, brauchen wir noch ein Hilfsmittel. Mit der *Verschiebungsregel* (engl. move rule) bringen wir einen Faktor, der neben $d\mathbf x$ steht, auf die andere Seite des inneren Produkts:

<!--en:61e38222-->
Das folgt direkt aus der zyklischen Invarianz der Spur, $\operatorname{tr}(\mathbf X\mathbf Y)=\operatorname{tr}(\mathbf Y\mathbf X)$: $\operatorname{tr}(\mathbf A^\top\mathbf B\mathbf C)=\operatorname{tr}\bigl((\mathbf B^\top\mathbf A)^\top\mathbf C\bigr)
=\operatorname{tr}(\mathbf C\mathbf A^\top\mathbf B)=\operatorname{tr}\bigl((\mathbf A\mathbf C^\top)^\top\mathbf B\bigr)$. In Worten: Ein Faktor links von der Variablen erscheint auf der anderen Seite als seine Transponierte wieder links. Ein Faktor rechts von der Variablen erscheint dort als seine Transponierte wieder rechts. Ein elementweiser Faktor wechselt unverändert die Seite, $\langle\mathbf A,\mathbf B\odot\mathbf C\rangle=\langle\mathbf A\odot\mathbf B,\mathbf C\rangle$, denn beide Seiten sind gleich $\sum_{ij}A_{ij}B_{ij}C_{ij}$.

<!--en:e93aad74-->
Die beiden einfachsten Fälle brauchen jeweils nur eine Zeile. Für eine lineare Form gilt $d(\mathbf a^\top\mathbf x)=\mathbf a^\top d\mathbf x=\langle\mathbf a,d\mathbf x\rangle$, also $\nabla_{\mathbf x}\,\mathbf a^\top\mathbf x=\mathbf a$. Bei einer quadratischen Form erfasst die Produktregel beide Vorkommen von $\mathbf x$:

<!--en:5ebf0cf9-->
Bei der zweiten Gleichheit haben wir eine $1\times1$-Matrix transponiert. Der Ausdruck $d\mathbf x^\top\mathbf A\mathbf x$ ist ein Skalar und daher gleich seiner eigenen Transponierten $\mathbf x^\top\mathbf A^\top d\mathbf x$. Am Koeffizienten lesen wir $\nabla_{\mathbf x}\,\mathbf x^\top\mathbf A\mathbf x=(\mathbf A+\mathbf A^\top)\mathbf x$ ab. Das vereinfacht sich zu $2\mathbf A\mathbf x$, wenn $\mathbf A$ symmetrisch ist.

<!--en:259581d6-->
Eine zweite Herleitung ergibt sich mit Indizes. Eintragsweise lautet dieselbe Ableitung $\partial\bigl(\sum_{ij}x_iA_{ij}x_j\bigr)/\partial x_k=\sum_jA_{kj}x_j+\sum_ix_iA_{ik}$. Das ist wieder der $k$-te Eintrag von $(\mathbf A+\mathbf A^\top)\mathbf x$. Herleitungen mit Indizes sind länger, lassen aber keinen Zweifel an den Transpositionen. In :numref:`sec_backprop` leiten wir so die Regel für das Matrixprodukt her.

<!--en:a9834374-->
Eine Matrixidentität lässt sich rasch prüfen, indem wir allen Matrizen die Form $1\times1$ geben. Dann muss sich die Formel auf die bekannte skalare Regel reduzieren. Unser Gradient der quadratischen Form besteht diese Prüfung, denn $\frac{d}{dx}(ax^2)=2ax=(a+a)x$. Die *$1\times1$-Prüfung* deckt einen falschen Koeffizienten oder einen fehlenden Term zuverlässig auf. Transpositionen und die Reihenfolge der Faktoren erfasst sie jedoch nicht, weil $1\times1$-Matrizen kommutieren und mit ihren Transponierten übereinstimmen. Die Prüfung der Form deckt die meisten Transpositionsfehler auf. Bei quadratischen Faktoren versagt sie allerdings, wie schon am Anfang dieses Abschnitts erwähnt. Hier erkennt erst eine numerische Prüfung den Fehler. Betrachten wir $\mathbf A=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$ und $\mathbf x=(1,2)$. Dann gilt $\mathbf x^\top\mathbf A\mathbf x=x_1x_2$. Der tatsächliche Gradient $(x_2,x_1)=(2,1)$ stimmt mit $(\mathbf A+\mathbf A^\top)\mathbf x$ überein. Der naheliegende Ansatz $2\mathbf A\mathbf x=(4,0)$, den die skalare Regel $2ax$ nahelegt, besteht die $1\times1$-Prüfung und hat die richtige Form, ist aber dennoch falsch.

<!--en:67c6a4bd-->
Mit Differentialen behandeln wir Matrixvariablen ohne zusätzlichen Aufwand. Betrachten wir den Verlust der kleinsten Quadrate $\ell=\|\mathbf W\mathbf x-\mathbf y\|^2$ mit dem Residuum $\mathbf r=\mathbf W\mathbf x-\mathbf y$. Variieren wir $\mathbf W$, ergibt sich $d\mathbf r=d\mathbf W\,\mathbf x$. Die Verschiebungsregel bringt $\mathbf x$ auf die andere Seite: $d\ell=2\mathbf r^\top d\mathbf r=\langle 2\mathbf r,\ d\mathbf W\,\mathbf x\rangle
=\langle2\mathbf r\mathbf x^\top,\ d\mathbf W\rangle$. Damit folgt

<!--en:714d5ea5-->
Dies ist ein äußeres Produkt aus Residuum und Eingabe vom Rang eins, das automatisch die Form von $\mathbf W$ hat. Variieren wir stattdessen $\mathbf x$, gilt $d\mathbf r=\mathbf W\,d\mathbf x$ und $\nabla_{\mathbf x}\ell=2\mathbf W^\top\mathbf r$. In beiden Fällen erhalten wir den Gradienten, indem wir das doppelte Residuum mit der Transponierten des Faktors multiplizieren, der an der Variablen steht.

<!--en:8706aec4-->
### Rückwärtsregeln einer Schicht

<!--en:fa1d68a4-->
In einem Netz ist eine Schicht $\mathbf Y=\mathbf F(\mathbf X)$ ein Schritt in einer langen Berechnung, die in einem skalaren Verlust $\ell$ endet. Ihre vollständige Jacobi-Matrix zu bilden kommt kaum infrage. Wenden wir eine $1000\times1000$-Gewichtsmatrix auf einen Minibatch mit $1000$ Beispielen an, so entstehen $10^6$ Ausgaben aus $10^6$ Gewichten. Die Jacobi-Matrix der Ausgaben bezüglich der Gewichte hätte also $10^{12}$ Einträge. Die Backpropagation benötigt zum Glück weit weniger. Eine Schicht muss lediglich aus dem Gradienten des Verlusts nach ihrer Ausgabe die Gradienten nach ihren Eingaben und Gewichten berechnen. Wie in :numref:`subsec_mdl-backpropagation` schreiben wir $\bar{\mathbf Y}=\nabla_{\mathbf Y}\ell$ für die adjungierte Variable zu $\mathbf Y$. Sie ist nun eine Matrix derselben Form wie $\mathbf Y$ und enthält die totale Ableitung des Verlusts nach jedem Eintrag von $\mathbf Y$.

<!--en:04dfef0b-->
**Satz (Rückwärtsregel einer Schicht).** *Sei $\mathbf Y=\mathbf F(\mathbf X)$ an der Stelle $\mathbf X$ differenzierbar und sei $\ell$ eine differenzierbare skalare Funktion von $\mathbf Y$. Dann ist die adjungierte Variable $\bar{\mathbf X}$ der Verkettung $\ell(\mathbf F(\mathbf X))$ die eindeutig bestimmte Matrix, für die gilt:*

<!--en:601c1b10-->
*Dabei ist $d\mathbf Y$ die Änderung erster Ordnung von $\mathbf F$, die durch $d\mathbf X$ entsteht.*

<!--en:4c2f99ea-->
**Beweis.** Nach der Kettenregel :eqref:`eq_mdl-jacobian-chain` können wir die Matrizen als Vektoren auffassen und den linearen Anteil der Verkettung berechnen, indem wir den linearen Anteil von $\ell$ auf den von $\mathbf F$ anwenden: $d\ell=\langle\bar{\mathbf Y},d\mathbf Y\rangle$. Da $d\mathbf Y$ linear in $d\mathbf X$ ist, gilt dies auch für $d\ell$. Wir können dieses Differential daher als $\langle\mathbf M,d\mathbf X\rangle$ mit einer eindeutig bestimmten Matrix $\mathbf M$ schreiben. Nach :eqref:`eq_mdl-differential-rule` gilt $\mathbf M=\bar{\mathbf X}$. $\blacksquare$

<!--en:cafb61c0-->
In der Praxis genügt zur Anwendung des Satzes die Verschiebungsregel. Für eine vektorwertige Schicht mit $d\mathbf y=\mathbf J\,d\mathbf x$ liefert sie $\langle\bar{\mathbf y},\mathbf J\,d\mathbf x\rangle=\langle\mathbf J^\top\bar{\mathbf y},d\mathbf x\rangle$, also $\bar{\mathbf x}=\mathbf J^\top\bar{\mathbf y}$. Die Rückwärtsregel einer Schicht ist somit die Transponierte ihrer Linearisierung. Die Transponierte einer linearen Abbildung heißt auch ihre Adjungierte. Daher stammt die Bezeichnung. Schreiben wir die Regel als $\bar{\mathbf x}^\top=\bar{\mathbf y}^\top\mathbf J$, erkennen wir einen Schritt der Multiplikation von links nach rechts aus :eqref:`eq_mdl-jacobian-product`. Das Produkt eines Zeilenvektors mit einer Jacobi-Matrix heißt *Vektor-Jacobi-Produkt* (engl. vector–Jacobian product, VJP). Der Satz erlaubt uns, es für Matrixvariablen zu berechnen, ohne die Jacobi-Matrix aufzustellen.

<!--en:0eaf380a-->
Die häufigste Ableitung im Deep Learning ist die eines Matrixprodukts. Je nachdem, ob wir einen Minibatch oder einen einzelnen Vektor verarbeiten, stehen die Gewichte rechts oder links von den Daten. Leiten wir beide Anordnungen zusammen her. Ein Minibatch $\mathbf X\in\mathbb R^{n\times d}$ enthält nach der Konvention des Haupttexts ein Beispiel pro Zeile. Multiplizieren wir ihn mit der Gewichtsmatrix $\mathbf W\in\mathbb R^{d\times q}$, erhalten wir $\mathbf Y=\mathbf X\mathbf W$. Die Produktregel liefert $d\mathbf Y=d\mathbf X\,\mathbf W+\mathbf X\,d\mathbf W$. Mit der Verschiebungsregel zerlegen wir das innere Produkt in einen Term pro Variable: $\langle\bar{\mathbf Y},d\mathbf X\,\mathbf W\rangle+\langle\bar{\mathbf Y},\mathbf X\,d\mathbf W\rangle
=\langle\bar{\mathbf Y}\mathbf W^\top,d\mathbf X\rangle+\langle\mathbf X^\top\bar{\mathbf Y},d\mathbf W\rangle$. An den beiden Koeffizienten lesen wir ab:

<!--en:7b52796b-->
Für ein einzelnes Beispiel in Spaltenform, $\mathbf y=\mathbf W\mathbf x$, ergibt dieselbe Rechnung

<!--en:77b78cc0-->
Das entspricht dem Muster aus :eqref:`eq_mdl-lstsq-grad`. Der Gewichtsgradient ist das äußere Produkt der adjungierten Variablen der Ausgabe mit der Eingabe. Die beiden Anordnungen sind zueinander transponiert. Sind $\mathbf x_i^\top$ und $\bar{\mathbf y}_i^\top$ die jeweiligen Zeilen von $\mathbf X$ und $\bar{\mathbf Y}$, so summiert die Minibatch-Formel $\mathbf X^\top\bar{\mathbf Y}=\sum_i\mathbf x_i\bar{\mathbf y}_i^\top$ je ein solches äußeres Produkt pro Beispiel.

<!--en:7309c379-->
Der Rückwärtsdurchlauf einer linearen Schicht besteht somit aus zwei Matrixprodukten, die jeweils ebenso viele Multiplikationen benötigen wie das Vorwärtsprodukt. Der Vorwärtsdurchlauf kostet $ndq$ Multiplikationen, der Rückwärtsdurchlauf $2ndq$. Die Formen bestätigen beide Transpositionen: $\bar{\mathbf Y}\mathbf W^\top$ hat die Form $n\times d$ wie $\mathbf X$, und $\mathbf X^\top\bar{\mathbf Y}$ hat die Form $d\times q$ wie $\mathbf W$. Auch die Rückwärtsregeln für aufwendigere Matrixoperationen wie Inversion, Determinantenbildung und Zerlegung ergeben sich mit derselben Methode :cite:`Giles.2008b`.

<!--en:b65c8f43-->
Zwei weitere Schichten vervollständigen ein mehrschichtiges Perzeptron. Für eine elementweise Aktivierung $\mathbf Y=\boldsymbol\varphi(\mathbf X)$ gilt $d\mathbf Y=\varphi'(\mathbf X)\odot d\mathbf X$. Verschieben wir $\varphi'(\mathbf X)$ auf die andere Seite des inneren Produkts, erhalten wir $\bar{\mathbf X}=\varphi'(\mathbf X)\odot\bar{\mathbf Y}$. Das ist die diagonale Jacobi-Matrix von vorhin, direkt als Maske angewandt. Für die Addition eines Bias $\mathbf y=\mathbf z+\mathbf b$ gilt $d\mathbf y=d\mathbf b$, also $\bar{\mathbf b}=\bar{\mathbf y}$. In einem Minibatch verteilt $\mathbf Z=\mathbf X\mathbf W+\mathbf 1\mathbf b^\top$ den Bias durch Broadcasting auf alle Zeilen. Das ist ein weiteres Matrixprodukt und liefert $\bar{\mathbf b}=\bar{\mathbf Z}^\top\mathbf 1$, die Summe der adjungierten Variablen über alle Beispiele. Aus diesen drei lokalen Regeln setzen wir in :numref:`sec_backprop` den vollständigen Rückwärtsdurchlauf eines mehrschichtigen Perzeptrons zusammen.

<!--en:416a7444-->
Mit diesen Regeln können wir auch die Matrix der Gewichtsgradienten aus dem Eingangsbeispiel herleiten. Durchlaufen wir das Netz aus :numref:`fig_mdl-cal-relu-net-jacobian` ausgehend von $\bar g=1$ Schicht für Schicht rückwärts, so ergibt sich an der Stelle $\mathbf x=(1,1)$:

<!--en:039dbbd0-->
1. Die Ausgabe $g=\mathbf w_2^\top\mathbf h$ liefert $\bar{\mathbf h}=\mathbf w_2=(2,-1)$ und $\bar{\mathbf w}_2=\mathbf h=(3,0)$.

<!--en:f0ee916a-->
2. Die ReLU-Funktion liefert $\bar{\mathbf a}=\mathbf m\odot\bar{\mathbf h}=(2,0)$, der Bias $\bar{\mathbf b}=\bar{\mathbf a}=(2,0)$.

<!--en:032d0326-->
3. Die affine Abbildung $\mathbf a=\mathbf W_1\mathbf x+\mathbf b$ liefert $\bar{\mathbf W}_1=\bar{\mathbf a}\,\mathbf x^\top=\left(\begin{smallmatrix}2&2\\0&0\end{smallmatrix}\right)$ und $\bar{\mathbf x}=\mathbf W_1^\top\bar{\mathbf a}=(2,4)$.

<!--en:f6f4346c-->
Dabei ist $\bar{\mathbf x}$ die Transponierte der zuvor gefundenen Zeile $(2,4)$ der Jacobi-Matrix. Die Matrix $\bar{\mathbf W}_1$ hatten wir im Eingangsbeispiel durch kleine Änderungen der Gewichte bestimmt. Leiten wir im Code `g` nach $\mathbf W_1$, $\mathbf b$ und $\mathbf w_2$ ab, so bestätigen sich alle drei Ergebnisse.

<!--en:c07086a7-->
Sehen wir uns an, wo Nullen auftreten. Die inaktive versteckte Einheit hinterlässt an drei Stellen Nullen: eine Nullzeile in $\bar{\mathbf W}_1$, eine Null in $\bar{\mathbf b}$ und eine Null in $\bar{\mathbf w}_2$ (denn mit ihrer Ausgabe $h_2=0$ wird $w_{2,2}$ multipliziert). Ein Trainingsschritt mit dieser Eingabe lässt alle vier Parameter der Einheit unverändert.

<!--en:ab481cff-->
Es fehlt noch der numerische Test, der eine falsch gesetzte Transposition bei einem quadratischen Faktor erkennt. Ein Test, der nicht fehlschlagen kann, beweist nichts. Wir wählen deshalb für jeden Faktor eine zufällige quadratische und damit fast sicher nicht symmetrische Matrix. Neben dem Ergebnis jeder korrekten Regel geben wir das eines plausiblen, aber falschen Ansatzes gleicher Form aus. Wir prüfen sowohl die quadratische Form als auch das Matrixprodukt. Für das Produkt differenzieren wir den Skalar $\langle\bar{\mathbf Y},\mathbf X\mathbf W\rangle$ für ein zufälliges $\bar{\mathbf Y}$. Nach unserem Satz über die Rückwärtsregel sind seine Gradienten nach $\mathbf X$ und $\mathbf W$ genau $\bar{\mathbf X}$ und $\bar{\mathbf W}$.

<!--en:74ac018c-->
Die hergeleiteten Regeln liefern in der Ausgabe `True`, die falschen Ansätze `False`. Hier ist der Aufbau des Tests noch wichtiger als sein Ergebnis. Bei rechteckigen Faktoren, etwa $\mathbf X$ der Form $3\times4$ und $\mathbf W$ der Form $4\times2$, ließen sich die beiden falschen Ansätze für das Produkt gar nicht ausführen. Eine falsch gesetzte Transposition würde bereits vor dem ersten Zahlenvergleich einen Dimensionsfehler auslösen. Genau das leistet die Prüfung der Form. Bei symmetrischem $\mathbf A$ wiederum hätte der falsche Ansatz $2\mathbf A\mathbf x$ mit dem tatsächlichen Gradienten übereingestimmt und den Test bestanden. Tückisch ist der Fall eines quadratischen, nicht symmetrischen Faktors. Solche Matrizen muss ein numerischer Test daher ziehen.

<!--en:d0ceddc4-->
Die Regel für das Matrixprodukt gilt ebenso für Verluste, bei denen beide Faktoren eines Produkts unbekannt sind, etwa für den Verlust der Matrixfaktorisierung $\|\mathbf X-\mathbf U\mathbf V\|_F^2$ in Aufgabe :exref:`ex_mdl-matrix-calculus-matmul-rules`. In jedem Fall multipliziert die Rückwärtsregel die eingehende adjungierte Variable mit den Transponierten der Faktoren, mit denen die Variable im Vorwärtsdurchlauf multipliziert wurde. Beachten wir auch, welche Werte aus dem Vorwärtsdurchlauf jede Regel braucht: $\mathbf X$ und $\mathbf W$ für das Matrixprodukt, $\varphi'(\mathbf X)$ für die Aktivierung und keinen Zwischenwert für den Bias. Diese Zwischenwerte müssen für den Rückwärtsdurchlauf gespeichert bleiben. Den Speicherbedarf dafür messen wir in :numref:`sec_mdl-matrix-calculus-autodiff` direkt.

<!--en:5acc5c1a-->
### Softmax-Funktion und Kreuzentropie

<!--en:d092ce06-->
Die Ableitung, die ein Klassifikator am häufigsten berechnet, ist die der Verkettung von Softmax-Funktion und Kreuzentropieverlust. Die Softmax-Funktion wandelt einen Vektor von Scores $\mathbf z$, die *Logits*, in einen Wahrscheinlichkeitsvektor um. Setzen wir $S=\sum_k e^{z_k}$, so gilt $p_i=e^{z_i}/S$. Dieselbe Quotientenregel, mit der wir in :numref:`sec_mdl-multivariable_calculus` die Hesse-Matrix der Log-Sum-Exp-Funktion für zwei Logits berechnet haben, liefert die Jacobi-Matrix der Softmax-Funktion für beliebig viele Klassen. Schreiben wir $\delta_{ij}=1$ für $i=j$ und sonst $0$, so gilt

<!--en:3d2707f7-->
Erhöhen wir also das Logit $i$, steigt $p_i$ mit der Rate $p_i(1-p_i)$. Jede andere Wahrscheinlichkeit $p_j$ sinkt mit der Rate $p_ip_j$. Zur Kontrolle gilt $\mathbf J\mathbf 1=\mathbf p-\mathbf p(\mathbf p^\top\mathbf 1)=\mathbf 0$, weil sich die Wahrscheinlichkeiten zu eins summieren. Das muss so sein, denn die Addition derselben Konstanten zu jedem Logit lässt $\mathbf p$ unverändert. Die Jacobi-Matrix der Softmax-Funktion ist daher immer singulär.

<!--en:9bcc8f9a-->
Beachten wir, dass $\mathbf J_{\operatorname{softmax}}$ auch symmetrisch ist. Das gilt nicht für jede Jacobi-Matrix. Hier liegt es daran, dass sie zugleich eine Hesse-Matrix ist. Die Softmax-Funktion ist nämlich der Gradient der Log-Sum-Exp-Funktion $\operatorname{LSE}(\mathbf z)=\log\sum_k e^{z_k}$, unseres durchgängigen Kapitelbeispiels, denn es gilt $\partial\operatorname{LSE}/\partial z_i=e^{z_i}/S=p_i$. Die Jacobi-Matrix des Gradienten ist die Hesse-Matrix der Funktion. Über die Formel für zwei Logits hinaus liefert :eqref:`eq_mdl-softmax-jacobian` eine statistische Deutung der Matrix. Sie ist die Kovarianzmatrix eines One-Hot-Vektors, den wir mit den Wahrscheinlichkeiten $\mathbf p$ ziehen. Für jedes $\mathbf z$ und jeden Vektor $\mathbf v$ ist die quadratische Form $\mathbf v^\top\mathbf J\mathbf v=\sum_ip_iv_i^2-\bigl(\sum_ip_iv_i\bigr)^2$ also die Varianz der Einträge von $\mathbf v$ unter der Verteilung $\mathbf p$. Da eine Varianz nie negativ ist, ist diese Hesse-Matrix überall positiv semidefinit. Damit ist die Konvexität der Log-Sum-Exp-Funktion bewiesen. Ein konstanter Vektor $\mathbf v$ hat die Varianz null. So finden wir erneut den Vektor $\mathbf 1$ im Kern der Matrix.

<!--en:a37d3975-->
Verketten wir nun die Softmax-Funktion mit dem Kreuzentropieverlust $\ell=-\sum_i y_i\log p_i$ bezüglich einer Zielverteilung $\mathbf y$. Die Kettenregel multipliziert die Zeile $\partial\ell/\partial\mathbf p=(-y_i/p_i)_i$ mit :eqref:`eq_mdl-softmax-jacobian`. Dabei vereinfacht sich der Ausdruck erheblich:

<!--en:d8179c49-->
Der Gradient nach den Logits ist die vorhergesagte Wahrscheinlichkeit abzüglich des Zielwerts. Das ist die Formel $\hat{\mathbf y}-\mathbf y$ aus :numref:`subsec_softmax_and_derivatives`. Da wir im letzten Schritt nur $\sum_iy_i=1$ verwendet haben, gilt das Ergebnis für jeden Wahrscheinlichkeitsvektor $\mathbf y$, also über One-Hot-Labels hinaus. Dieselbe Formel gilt damit auch für geglättete Labels und für Zielwerte bei der Destillation. Summieren sich die Zielwerte nicht zu eins, lautet der Gradient hingegen $(\sum_iy_i)\,\mathbf p-\mathbf y$. Bei $\mathbf z=(1,-0.5,2)$ erhalten wir zum Beispiel $\mathbf p=(0.254,\ 0.057,\ 0.690)$. Ist die zweite Klasse das Ziel, lautet der Gradient $(0.254,\ -0.943,\ 0.690)$.

<!--en:14f97036-->
In der Praxis fassen Bibliotheken die Softmax-Funktion und die Kreuzentropie zur numerischen Stabilität in einer einzigen Operation zusammen (:numref:`subsec_softmax-implementation-revisited`, :numref:`subsec_mdl-stable-softmax`). Im folgenden Code berechnen wir die Jacobi-Matrix der Softmax-Funktion an der Stelle $\mathbf z$ durch automatisches Differenzieren und geben ihre Eigenwerte aus. Nach unserem Varianzargument sind sie nichtnegativ und einer davon ist null. Anschließend differenzieren wir den zusammengefassten Verlust sowohl für den obigen One-Hot-Zielvektor als auch für den Two-Hot-Zielvektor $\mathbf y=(1,1,0)$, dessen Einträge sich nicht zu eins summieren.

<!--en:45ba0539-->
Die Eigenwerte sind $0$, $0.079$ und $0.378$. Der Eigenwert null gehört zur Richtung $\mathbf 1$, die beiden anderen Werte sind positiv. Für den One-Hot-Zielvektor ist der Gradient des zusammengefassten Verlusts $\mathbf p-\mathbf y=(0.2537,-0.9434,0.6897)$, wie von Hand berechnet. Für den Two-Hot-Zielvektor liefert die Berechnung hingegen $(-0.4926,-0.8868,1.3793)$. Das entspricht $2\mathbf p-\mathbf y$ und weicht von dem daneben ausgegebenen $\mathbf p-\mathbf y$ ab. „Vorhersage minus Zielwert“ setzt voraus, dass sich die Zielwerte zu eins summieren. Aus der Verlustfunktion allein folgt diese Formel nicht.

<!--en:2673f521-->
Hier haben wir $\mathbf J_K\cdots\mathbf J_1$ für eine einfache Kette aus drei Schichten von Hand multipliziert. In der Praxis muss ein Programm beliebige Berechnungsgraphen verarbeiten, in denen sich Pfade verzweigen und wieder zusammenlaufen. Es muss die Werte aus dem Vorwärtsdurchlauf speichern, die jede Rückwärtsregel benötigt. Und es muss alle Gradienten mit Kosten berechnen, die unabhängig von der Parameterzahl nur ein kleines konstantes Vielfaches einer Netzauswertung betragen. Ein solches Programm übernimmt das automatische Differenzieren, das Thema von :numref:`sec_mdl-matrix-calculus-autodiff`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:e9db935f-->
Die Jacobi-Matrix einer Abbildung $\mathbf f:\mathbb R^n\to\mathbb R^m$ ist die $m\times n$-Matrix ihrer besten linearen Approximation mit den Einträgen $[\mathbf J_{\mathbf f}]_{ij}=\partial f_i/\partial x_j$. Der Gradient (eine einzelne Zeile, als Spalte geschrieben) und die Hesse-Matrix (die Jacobi-Matrix des Gradienten) sind Spezialfälle. Die Ableitung eines Netzes nach seiner Eingabe ist das Produkt der Jacobi-Matrizen seiner Schichten, $\mathbf J_K\cdots\mathbf J_1$. Die Reihenfolge der Multiplikation bestimmt die Kosten: Bei drei Schichten der Breite $1000$ stehen drei Millionen Multiplikationen zwei Milliarden gegenüber. Mit der Differentialregel $df=\langle\nabla f,d\mathbf x\rangle$ erhalten wir jeden Gradienten, den eine Schicht benötigt, direkt in der Form seiner Variablen. So ergeben sich die Rückwärtsregeln $\bar{\mathbf X}=\bar{\mathbf Y}\mathbf W^\top$ und $\bar{\mathbf W}=\mathbf X^\top\bar{\mathbf Y}$ für das Matrixprodukt. Ebenso vereinfacht sich der Gradient bei der Verkettung von Softmax-Funktion und Kreuzentropie zu $\mathbf p-\mathbf y$. Die Prüfung der Form und die $1\times1$-Prüfung decken die meisten Fehler beim Herleiten von Hand auf. Eine falsch gesetzte Transposition bei einem quadratischen Faktor entgeht beiden Prüfungen und erfordert einen numerischen Test an einer nicht symmetrischen Matrix.

<!--en:2294243a-->
## Übungen

<!--en:ccf39f10-->
1. **Jacobi-Matrix eines ReLU-Netzes.** Sei $\mathbf g(\mathbf x)=\mathbf W_2\,\boldsymbol\varphi(\mathbf W_1\mathbf x+\mathbf b)$ mit $\mathbf W_1\in\mathbb R^{h\times n}$, $\mathbf b\in\mathbb R^h$, $\mathbf W_2\in\mathbb R^{m\times h}$ und der elementweisen ReLU-Funktion $\boldsymbol\varphi$. Kein Eintrag der Präaktivierung $\mathbf W_1\mathbf x+\mathbf b$ sei null. Nach :eqref:`eq_mdl-jacobian-chain` gilt $\mathbf J_{\mathbf g}(\mathbf x)=\mathbf W_2\mathbf M(\mathbf x)\mathbf W_1$. Dabei enthält die Diagonalmatrix $\mathbf M(\mathbf x)$ den Eintrag $1$ für jede aktive versteckte Einheit und $0$ für jede inaktive.
    1. Zeigen Sie, dass $\mathbf g(\mathbf x+\boldsymbol\delta)=\mathbf g(\mathbf x)+\mathbf J_{\mathbf g}(\mathbf x)\,\boldsymbol\delta$ für jedes hinreichend kleine $\boldsymbol\delta$ exakt gilt. Das Restglied in :eqref:`eq_mdl-jacobian-linearization` verschwindet also identisch. Bestimmen Sie für das 2-2-1-Netz dieses Abschnitts an der Stelle $\mathbf x=(1,1)$ das größte $r$, für das dies für alle $\boldsymbol\delta$ mit $\|\boldsymbol\delta\|_\infty<r$ gilt. Wie unterscheidet sich für $n=m=2$ das Bild eines kleinen Kreises unter $\mathbf g$ von dem in :numref:`fig_mdl-cal-jacobian-ellipse`?
    1. Es seien $k$ versteckte Einheiten an der Stelle $\mathbf x$ aktiv. Zeigen Sie, dass $\operatorname{rank}\mathbf J_{\mathbf g}(\mathbf x)\le k$ gilt. Was ist für $n=m=2$ und $k=1$ das Bild eines kleinen Kreises um $\mathbf x$?
    1. Beschreiben Sie die Menge der Eingaben, bei denen mindestens eine Präaktivierung null ist. Was geschieht mit $\mathbf J_{\mathbf g}$, wenn $\mathbf x$ diese Menge durchquert?
   :exlabel:`ex_mdl-matrix-calculus-relu-jacobian`

<!--en:2d87440b-->
1. [code] **Günstigste Multiplikationsreihenfolge einer Kette.** ● Nach :eqref:`eq_mdl-jacobian-product` ist die Jacobi-Matrix einer Kette aus $K$ Abbildungen gleich $\mathbf J_K\cdots\mathbf J_1$ mit $\mathbf J_k\in\mathbb R^{d_k\times d_{k-1}}$. Das Produkt einer $p\times q$-Matrix mit einer $q\times r$-Matrix kostet $pqr$ Multiplikationen.
    1. Geben Sie die Kosten für die Vorwärtsreihenfolge (Multiplikation von rechts) und die Rückwärtsreihenfolge (Multiplikation von links) als Funktionen von $d_0,\ldots,d_K$ an. Berechnen Sie beide für $(d_0,\ldots,d_4)=(1000,10,1000,10,1000)$, also für eine Abbildung mit abwechselnd breiten und schmalen Schichten. Vergleichen Sie die Kosten mit der Anzahl der Multiplikationen, die wir in diesem Abschnitt für $(1\times h)(h\times h)(h\times h)(h\times n)$ ermittelt haben.
    1. Schreiben Sie ein Programm, das durch dynamische Programmierung über die Teilketten $\mathbf J_j\cdots\mathbf J_i$ die günstigste Klammerung bestimmt. Führen Sie es für die Dimensionen aus (a) aus. Welche Reihenfolge wählt es, und wie viel günstiger ist diese als die bessere der beiden Reihenfolgen aus (a)?
    1. Vergleichen Sie für eine skalare Ausgabe, $d_K=1$, die optimale Reihenfolge mit der Rückwärtsreihenfolge an einigen Tausend zufälligen Dimensionsvektoren. Wählen Sie $K$ zwischen $2$ und $6$ und ziehen Sie jede der Dimensionen $d_0,\ldots,d_{K-1}$ gleichverteilt aus $\{1,\ldots,10\}$. Charakterisieren Sie die Vektoren, für die eine andere Reihenfolge günstiger ist.
   :exlabel:`ex_mdl-matrix-calculus-chain-order`

<!--en:9ab95190-->
3. [code] **Rückwärtsregeln eines Matrixprodukts.**
    1. Leiten Sie für $\mathbf Y=\mathbf X\mathbf W$ die Regeln :eqref:`eq_mdl-matmul-vjp` durch Rechnen mit Differentialen her. Leiten Sie sie anschließend mit Indizes her, um die Transpositionen zu überprüfen.
    1. Prüfen Sie beide Regeln mithilfe des Vektor-Jacobi-Produkts `backward(head_grad)` für $\mathbf X\in\mathbb R^{3\times4}$, $\mathbf W\in\mathbb R^{4\times2}$ und eine zufällige adjungierte Variable $\bar{\mathbf Y}$. Zählen Sie die Multiplikationen für das Vorwärtsprodukt und die beiden Rückwärtsprodukte.
    1. Prüfen Sie für den Faktorisierungsverlust $\|\mathbf X-\mathbf U\mathbf V\|_F^2$ mit $\mathbf R=\mathbf X-\mathbf U\mathbf V$ die Formeln $\nabla_{\mathbf U}=-2\mathbf R\mathbf V^\top$ und $\nabla_{\mathbf V}=-2\mathbf U^\top\mathbf R$ numerisch. Formulieren Sie dann den Schritt der alternierenden Methode der kleinsten Quadrate, der einen Gradienten bei festem anderen Faktor gleich null setzt. Zeigen Sie, dass jeder Schritt genau ein lineares Gleichungssystem löst.
   :exlabel:`ex_mdl-matrix-calculus-matmul-rules`

<!--en:099dd688-->
3. [code] **Rückwärtsregeln eines Matrixprodukts.**
    1. Leiten Sie für $\mathbf Y=\mathbf X\mathbf W$ die Regeln :eqref:`eq_mdl-matmul-vjp` durch Rechnen mit Differentialen her. Leiten Sie sie anschließend mit Indizes her, um die Transpositionen zu überprüfen.
    1. Prüfen Sie beide Regeln mithilfe des Vektor-Jacobi-Produkts `torch.func.vjp` für $\mathbf X\in\mathbb R^{3\times4}$, $\mathbf W\in\mathbb R^{4\times2}$ und eine zufällige adjungierte Variable $\bar{\mathbf Y}$. Zählen Sie die Multiplikationen für das Vorwärtsprodukt und die beiden Rückwärtsprodukte.
    1. Prüfen Sie für den Faktorisierungsverlust $\|\mathbf X-\mathbf U\mathbf V\|_F^2$ mit $\mathbf R=\mathbf X-\mathbf U\mathbf V$ die Formeln $\nabla_{\mathbf U}=-2\mathbf R\mathbf V^\top$ und $\nabla_{\mathbf V}=-2\mathbf U^\top\mathbf R$ numerisch. Formulieren Sie dann den Schritt der alternierenden Methode der kleinsten Quadrate, der einen Gradienten bei festem anderen Faktor gleich null setzt. Zeigen Sie, dass jeder Schritt genau ein lineares Gleichungssystem löst.
   :exlabel:`ex_mdl-matrix-calculus-matmul-rules`

<!--en:c364a51f-->
3. [code] **Rückwärtsregeln eines Matrixprodukts.**
    1. Leiten Sie für $\mathbf Y=\mathbf X\mathbf W$ die Regeln :eqref:`eq_mdl-matmul-vjp` durch Rechnen mit Differentialen her. Leiten Sie sie anschließend mit Indizes her, um die Transpositionen zu überprüfen.
    1. Prüfen Sie beide Regeln mithilfe des Vektor-Jacobi-Produkts `tape.gradient(..., output_gradients=...)` für $\mathbf X\in\mathbb R^{3\times4}$, $\mathbf W\in\mathbb R^{4\times2}$ und eine zufällige adjungierte Variable $\bar{\mathbf Y}$. Zählen Sie die Multiplikationen für das Vorwärtsprodukt und die beiden Rückwärtsprodukte.
    1. Prüfen Sie für den Faktorisierungsverlust $\|\mathbf X-\mathbf U\mathbf V\|_F^2$ mit $\mathbf R=\mathbf X-\mathbf U\mathbf V$ die Formeln $\nabla_{\mathbf U}=-2\mathbf R\mathbf V^\top$ und $\nabla_{\mathbf V}=-2\mathbf U^\top\mathbf R$ numerisch. Formulieren Sie dann den Schritt der alternierenden Methode der kleinsten Quadrate, der einen Gradienten bei festem anderen Faktor gleich null setzt. Zeigen Sie, dass jeder Schritt genau ein lineares Gleichungssystem löst.
   :exlabel:`ex_mdl-matrix-calculus-matmul-rules`

<!--en:c0ec462d-->
3. [code] **Rückwärtsregeln eines Matrixprodukts.**
    1. Leiten Sie für $\mathbf Y=\mathbf X\mathbf W$ die Regeln :eqref:`eq_mdl-matmul-vjp` durch Rechnen mit Differentialen her. Leiten Sie sie anschließend mit Indizes her, um die Transpositionen zu überprüfen.
    1. Prüfen Sie beide Regeln mithilfe des Vektor-Jacobi-Produkts `jax.vjp` für $\mathbf X\in\mathbb R^{3\times4}$, $\mathbf W\in\mathbb R^{4\times2}$ und eine zufällige adjungierte Variable $\bar{\mathbf Y}$. Zählen Sie die Multiplikationen für das Vorwärtsprodukt und die beiden Rückwärtsprodukte.
    1. Prüfen Sie für den Faktorisierungsverlust $\|\mathbf X-\mathbf U\mathbf V\|_F^2$ mit $\mathbf R=\mathbf X-\mathbf U\mathbf V$ die Formeln $\nabla_{\mathbf U}=-2\mathbf R\mathbf V^\top$ und $\nabla_{\mathbf V}=-2\mathbf U^\top\mathbf R$ numerisch. Formulieren Sie dann den Schritt der alternierenden Methode der kleinsten Quadrate, der einen Gradienten bei festem anderen Faktor gleich null setzt. Zeigen Sie, dass jeder Schritt genau ein lineares Gleichungssystem löst.
   :exlabel:`ex_mdl-matrix-calculus-matmul-rules`

<!--en:871cc9b3-->
4. **Vektor-Jacobi-Produkt der Softmax-Funktion.** Ein Rückwärtsdurchlauf stellt die Jacobi-Matrix :eqref:`eq_mdl-softmax-jacobian` der Softmax-Funktion nie auf. Er bildet die adjungierte Variable $\mathbf u=\bar{\mathbf p}$ der Wahrscheinlichkeiten auf die adjungierte Variable $\bar{\mathbf z}=\mathbf J_{\operatorname{softmax}}^\top\mathbf u$ der Logits ab.
    1. Zeigen Sie, dass $\mathbf J_{\operatorname{softmax}}^\top\mathbf u=\mathbf p\odot\bigl(\mathbf u-(\mathbf p^\top\mathbf u)\,\mathbf 1\bigr)$ gilt, wobei $\odot$ das elementweise Produkt bezeichnet. Zählen Sie für $n$ Klassen die Operationen dieser Formel. Zählen Sie zum Vergleich die Operationen, die nötig sind, um $\mathbf J_{\operatorname{softmax}}$ zu bilden und mit $\mathbf u$ zu multiplizieren.
    1. Self-Attention wendet die Softmax-Funktion auf jede Zeile einer $n\times n$-Score-Matrix $\mathbf S$ an und liefert die Gewichte $\mathbf P$. Wie viele Einträge hat die Jacobi-Matrix von $\mathbf P$ bezüglich $\mathbf S$, und wie viele davon können von null verschieden sein? Vergleichen Sie für $n=4096$ Token im Format float32 den Speicherbedarf der vollständigen Jacobi-Matrix, ihrer von null verschiedenen Einträge und der adjungierten Variablen von $\mathbf S$, die die Formel aus (a) bei zeilenweiser Anwendung liefert.

<!--en:cefb1804-->
1. [code] **Die Jacobi-Matrix von LayerNorm.** Die Schichtnormalisierung :cite:`Ba.Kiros.Hinton.2016` bildet $\mathbf x\in\mathbb R^n$ auf $\hat{\mathbf x}=(\mathbf x-\mu\mathbf 1)/\sigma$ ab, mit $\mu=\frac1n\sum_ix_i$ und $\sigma^2=\frac1n\sum_i(x_i-\mu)^2>0$ (ohne erlernten Skalierungsfaktor und ohne $\epsilon$).
    1. Zeigen Sie durch Rechnen mit Differentialen, dass $\mathbf J=\frac1\sigma\bigl(\mathbf I-\frac1n\mathbf 1\mathbf 1^\top-\frac1n\hat{\mathbf x}\hat{\mathbf x}^\top\bigr)$ gilt. *Hinweis:* Es gilt $d\mu=\frac1n\mathbf 1^\top d\mathbf x$ und $d\sigma=\frac1n\hat{\mathbf x}^\top d\mathbf x$.
    1. Zeigen Sie, dass $\mathbf 1$ und $\hat{\mathbf x}$ orthogonal sind und dass $\|\hat{\mathbf x}\|^2=n$ gilt. Folgern Sie, dass $\sigma\mathbf J$ die orthogonale Projektion auf das orthogonale Komplement von $\operatorname{span}\{\mathbf 1,\hat{\mathbf x}\}$ ist und dass $\operatorname{rank}\mathbf J=n-2$ für $n\ge2$ gilt. Welche zwei Änderungen der Eingabe kann LayerNorm nicht erkennen und warum?
    1. Prüfen Sie (a) und die beiden Vektoren im Kern anhand der Jacobi-Matrix, die Sie für ein zufälliges $\mathbf x\in\mathbb R^5$ durch automatisches Differenzieren berechnen. Geben Sie die Eigenwerte von $\mathbf J$ aus.

<!--en:d89dd310-->
1. **Gradient des Logarithmus des Determinantenbetrags.** ● Ein normalisierender Fluss :cite:`rezende2015variational` führt einen Zufallsvektor $\mathbf X$ durch invertierbare Schichten. Für eine Schicht $\mathbf Y=\mathbf g(\mathbf X)$ ergibt die Formel zur Variablentransformation für Dichten die Log-Dichte von $\mathbf Y$ an der Stelle $\mathbf g(\mathbf x)$ als $\log p_{\mathbf X}(\mathbf x)-\log|\det\mathbf J_{\mathbf g}(\mathbf x)|$. Für eine lineare Schicht $\mathbf x\mapsto\mathbf A\mathbf x$ lautet der Korrekturterm $-\log|\det\mathbf A|$. Beim Maximum-Likelihood-Training leiten wir daher $\log|\det\mathbf A|$ nach $\mathbf A$ ab.
    1. Entwickeln Sie $\det(\mathbf I+t\mathbf M)$ bis zur ersten Ordnung in $t$ und verwenden Sie $\det(\mathbf A+d\mathbf A)=\det\mathbf A\,\det(\mathbf I+\mathbf A^{-1}d\mathbf A)$, um die Jacobi-Formel $d\det\mathbf A=\det\mathbf A\,\operatorname{tr}(\mathbf A^{-1}d\mathbf A)$ für invertierbare Matrizen $\mathbf A$ herzuleiten. Folgern Sie mit :eqref:`eq_mdl-differential-rule`, dass $\nabla_{\mathbf A}\log|\det\mathbf A|=\mathbf A^{-\top}$ gilt. Warum spielt das Vorzeichen von $\det\mathbf A$ keine Rolle?
    1. Prüfen Sie die Identität an einer zufälligen invertierbaren $3\times3$-Matrix mit zentralen Differenzenquotienten von `np.linalg.slogdet`.
    1. Wie verhält sich die Norm des Gradienten, wenn sich $\mathbf A$ einer singulären Matrix nähert? Was bedeutet das für eine Schicht eines normalisierenden Flusses, deren Gewichtsmatrix fast singulär ist?

<!--en:7e8e86cf-->
1. **Gradient der Spektralnorm.** ● Die spektrale Normalisierung :cite:`Miyato.Kataoka.Koyama.ea.2018` teilt jede Gewichtsmatrix $\mathbf W$ eines Netzes durch ihren größten Singulärwert $\sigma_1(\mathbf W)$. Beim Training wird daher $\sigma_1$ differenziert.
    1. Sei $\mathbf S$ eine reelle symmetrische Matrix mit einem einfachen Eigenwert $\lambda$ und einem zugehörigen normierten Eigenvektor $\mathbf q$. Betrachten Sie die gestörte Matrix $\mathbf S+t\mathbf E$ mit symmetrischem $\mathbf E$. Setzen Sie als gegeben voraus, dass $\lambda(t)$ und $\mathbf q(t)$, definiert durch $(\mathbf S+t\mathbf E)\mathbf q=\lambda\mathbf q$ und $\mathbf q^\top\mathbf q=1$, in der Nähe von $t=0$ differenzierbar sind (das folgt aus dem Satz über implizite Funktionen, weil $\lambda$ einfach ist). Differenzieren Sie beide Gleichungen an der Stelle $t=0$, um den Satz von Hellmann und Feynman $\lambda'(0)=\mathbf q^\top\mathbf E\,\mathbf q$ zu zeigen. Geben Sie $\nabla_{\mathbf S}\lambda$ bezüglich des inneren Produkts $\operatorname{tr}(\mathbf A^\top\mathbf B)$ an.
    1. Wenden Sie (a) auf $\sigma_1(\mathbf W)^2=\lambda_{\max}(\mathbf W^\top\mathbf W)$ an, um zu zeigen, dass $\nabla_{\mathbf W}\sigma_1=\mathbf u_1\mathbf v_1^\top$ gilt, wenn $\sigma_1>0$ ein einfacher Singulärwert ist. Dabei sind $\mathbf u_1$ und $\mathbf v_1$ die zum größten Singulärwert gehörenden linken und rechten Singulärvektoren. Warum ist $\sigma_1>0$ nötig? Betrachten Sie $\mathbf W=\mathbf w\in\mathbb R^{n\times1}$ in der Nähe von $\mathbf w=\mathbf 0$. Prüfen Sie die Formel mit zentralen Differenzenquotienten an einer zufälligen $5\times3$-Matrix in NumPy.
    1. Berechnen Sie $\sigma_1(\mathbf I+t\operatorname{diag}(1,-1))$ für kleine $t$ beider Vorzeichen. Ist $\sigma_1$ an der $2\times2$-Einheitsmatrix differenzierbar, und welche Voraussetzung aus (a) ist dort verletzt?

    *Nach Alan Edelman und Steven G. Johnson, [Matrixdifferentialrechnung für maschinelles Lernen und darüber hinaus, Aufgabenblatt 2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_pset2.pdf), MIT 18.S096, IAP 2023, Aufgabe 3.*

<!--en:37006531-->
[Diskussionen](https://d2l.discourse.group/t/36857)

<!--en:e1f80a3e-->
Matrixdifferentialrechnung<br>**Jacobi-Matrix · Kettenregel und Multiplikationsreihenfolge · Rechnen mit Differentialen · Rückwärtsregeln für Schichten (VJPs) · Softmax-Funktion und Kreuzentropie**

<!--en:d0f87d05-->
::: {.slide title="Von skalaren Knoten zu Schichten mit Vektoren und Matrizen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:96efe379-->
- **2-2-1-ReLU-Netz bei $\mathbf{x} = (1, 1)^\top$:** Aus $g(\mathbf{x}) = \mathbf{w}_2^\top\boldsymbol{\varphi}(\mathbf{W}_1\mathbf{x} + \mathbf{b})$ folgen $\mathbf{a} = (3, -2)^\top$, $\mathbf{h} = (3, 0)^\top$ und $g = 6$.

<!--en:64181478-->
- **Drei Ableitungen mit unterschiedlichen Formen:** Jacobi-Matrix bezüglich der Eingabe $\partial g/\partial\mathbf{x} \in \mathbb{R}^{1 \times 2}$, Gewichtsgradient $\nabla_{\mathbf{W}_1}g \in \mathbb{R}^{2 \times 2}$ und Jacobi-Matrix der versteckten Schicht $\partial\mathbf{h}/\partial\mathbf{x} \in \mathbb{R}^{2 \times 2}$.

<!--en:48cd7aad-->
- **Fehlende Transposition:** Bei einer quadratischen Gewichtsmatrix verrät die Form den Fehler nicht. Rechnen mit Differentialen beugt ihm vor.

<!--en:13c16d9b-->
::: {.slide title="Die Jacobi-Matrix als beste lineare Approximation einer Vektorfunktion"}

<!--en:d4577a42-->
[Definition und Einträge]{.kicker}

<!--en:ad547e4e-->
- **Linearisierung einer Vektorfunktion in erster Ordnung:** $\mathbf{f}(\mathbf{x} + \boldsymbol{\delta}) = \mathbf{f}(\mathbf{x}) + \mathbf{J}_{\mathbf{f}}(\mathbf{x})\boldsymbol{\delta} + o(\|\boldsymbol{\delta}\|)$ mit $[\mathbf{J}_{\mathbf{f}}(\mathbf{x})]_{ij} = \frac{\partial f_i}{\partial x_j}(\mathbf{x}) \in \mathbb{R}^{m \times n}$.

<!--en:132eb3c0-->
- **Zeilen und Spalten:** Zeile $i$ ist $\nabla f_i(\mathbf{x})^\top$, Spalte $j$ ist $\partial\mathbf{f}/\partial x_j$.

<!--en:d8265172-->
- **Nichtlineare Abbildung $(x+\sin y, y+x^2/2)^\top$ an der Stelle $(0.5, 0.5)^\top$:** $\mathbf{J}_{\mathbf{f}} = \left(\begin{smallmatrix}1 & \cos 0.5 \\ 0.5 & 1\end{smallmatrix}\right)$.

<!--en:4642dbb5-->
::: {.slide title="Geometrie der Jacobi-Matrix: Kreise werden zu Ellipsen"}

<!--en:7dff711d-->
[Verzerrung durch die Singulärwerte]{.kicker}

<!--en:1b9d1321-->
- **Lokale affine Verzerrung:** In der Nähe von $\mathbf{x}_0 = (0.5, 0.5)^\top$ verhält sich $\mathbf{f}(\mathbf{x}_0 + \boldsymbol{\delta})$ wie die affine Abbildung $\mathbf{f}(\mathbf{x}_0) + \mathbf{J}(\mathbf{x}_0)\boldsymbol{\delta}$.

<!--en:d5894f06-->
- **Singulärwerte bestimmen die Halbachsen:** Jede Matrix bildet einen Kreis mit Radius $r$ um den Ursprung auf eine Ellipse mit Halbachsen der Längen $r\sigma_1$ und $r\sigma_2$ ab. Dabei sind $\sigma_1 \approx 1.706$ und $\sigma_2 \approx 0.329$ die Singulärwerte von $\mathbf{J}(\mathbf{x}_0)$.

<!--en:fb9bbdf0-->
- **Quadratische Konvergenz der Abweichung gegen null:** Beim Radius $r = 0.35$ weicht das tatsächliche Bild (blau) um höchstens $0.061$ von der Ellipse der Jacobi-Matrix ab (orange gestrichelt). Beim halben Radius $r = 0.175$ sinkt die Abweichung auf ein Viertel, auf $0.015$.

<!--en:03a44dab-->
::: {.slide title="Quadratische Abnahme des Restglieds der Linearisierung überprüfen"}

<!--en:7484a6ca-->
[Numerische Prüfung]{.kicker}

<!--en:3f8fcdd9-->
Verkleinern wir den Schritt $\boldsymbol{\delta}$ wiederholt um den Faktor zehn, so sehen wir, dass der relative Fehler der linearen Approximation $\|\mathbf{f}(\mathbf{x}_0 + \boldsymbol{\delta}) - \mathbf{f}(\mathbf{x}_0) - \mathbf{J}(\mathbf{x}_0)\boldsymbol{\delta}\| / \|\boldsymbol{\delta}\|$ linear mit $\|\boldsymbol{\delta}\|$ gegen null geht:

<!--en:5c404f30-->
**Hinreichende Bedingung für eine Jacobi-Matrix:** Wie in §{sec_mdl-multivariable_calculus} genügt die Existenz partieller Ableitungen allein nicht ($2xy/(x^2+y^2)$ hat bei $\mathbf{0}$ keine Jacobi-Matrix). Sind alle partiellen Ableitungen $\partial f_i / \partial x_j$ in der Nähe von $\mathbf{x}$ **stetig**, so ist $\mathbf{f}$ an der Stelle $\mathbf{x}$ differenzierbar. Für glatte ($C^2$) Abbildungen hat der Fehler die Ordnung $O(\|\boldsymbol{\delta}\|^2)$.

<!--en:7ef510b6-->
::: {.slide title="Zähler- und Nennerkonvention und die Form der Ableitung"}

<!--en:f2a77ed4-->
[Formkonventionen]{.kicker}

<!--en:3ec2624f-->
- **Jacobi-Matrix $\mathbf{J}_{\mathbf{f}} \in \mathbb{R}^{m \times n}$ (Zählerkonvention):** Die Verkettung ergibt direkt $\mathbf{J}_{\mathbf{g}}\mathbf{J}_{\mathbf{f}}$, ohne Transpositionen.

<!--en:c94acb1a-->
- **Gradient $\nabla_{\mathbf{X}} f$ (Grundregel für die Form):** Er hat stets dieselbe Form wie $\mathbf{X}$ (Spalte $\mathbb{R}^{n \times 1}$ oder Matrix $\mathbb{R}^{p \times q}$). Das folgt aus $df = \langle \nabla_{\mathbf{X}} f, d\mathbf{X}\rangle$.

<!--en:0f5e69ed-->
- **Hesse-Matrix $\mathbf{H}_f = \mathbf{J}_{\nabla f} \in \mathbb{R}^{n \times n}$:** Jacobi-Matrix des Gradientenfelds.

<!--en:9297b85f-->
::: {.slide title="Die Kettenregel für Jacobi-Matrizen als Matrizenmultiplikation"}

<!--en:4bd86744-->
[Verkettung von Abbildungen]{.kicker}

<!--en:992e4c41-->
- **Kettenregel für Matrizen:** Sind $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^p$ und $\mathbf{g} : \mathbb{R}^p \to \mathbb{R}^m$ differenzierbar, so gilt $\mathbf{J}_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}) = \mathbf{J}_{\mathbf{g}}(\mathbf{f}(\mathbf{x}))\,\mathbf{J}_{\mathbf{f}}(\mathbf{x})$.

<!--en:7f494752-->
- **Beweis in drei Zeilen:** $\boldsymbol{\eta} = \mathbf{A}\boldsymbol{\delta} + o(\|\boldsymbol{\delta}\|) \implies \mathbf{g}(\mathbf{y}+\boldsymbol{\eta}) = \mathbf{g}(\mathbf{y}) + \mathbf{B}\mathbf{A}\boldsymbol{\delta} + o(\|\boldsymbol{\delta}\|)$.

<!--en:b168b3d1-->
- **Zusammenhang mit der Pfadsumme:** $[\mathbf{B}\mathbf{A}]_{ik} = \sum_{j=1}^p \frac{\partial g_i}{\partial y_j}\frac{\partial f_j}{\partial x_k}$.

<!--en:12f6e8d1-->
::: {.slide title="Jacobi-Matrix eines mehrschichtigen Perzeptrons"}

<!--en:1ab75ad2-->
[Ein Faktor pro Schicht]{.kicker}

<!--en:f631a960-->
- **Drei verkettete Schichten in unserem 2-2-1-Netz mit $\mathrm{ReLU}$:**
  1. Die affine Schicht $\mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}$ hat die Jacobi-Matrix $\mathbf{J}_1 = \mathbf{W}_1$.
  2. Die elementweise $\mathrm{ReLU}$-Funktion $\mathbf{h} = \boldsymbol{\varphi}(\mathbf{a})$ entkoppelt die Koordinaten. Ihre Jacobi-Matrix ist daher diagonal: $\mathbf{J}_2 = \operatorname{diag}(\mathbf{m})$ mit der Indikatormaske $m_i = \mathbf{1}(a_i > 0)$.
  3. Die Ausgabeschicht $g = \mathbf{w}_2^\top \mathbf{h}$ hat die $1 \times 2$-Jacobi-Matrix $\mathbf{J}_3 = \mathbf{w}_2^\top$ in Zeilenform.

<!--en:25c61bfd-->
- **An der Stelle $\mathbf{x} = (1, 1)^\top$ ($\mathbf{a} = (3, -2)^\top$, $\mathbf{m} = (1, 0)^\top$):** Das Produkt ergibt $(2, 4)$:
  $$\mathbf{J}_g(\mathbf{x}) = \mathbf{w}_2^\top \operatorname{diag}(\mathbf{m})\,\mathbf{W}_1$$

<!--en:0fbfc3f8-->
::: {.slide title="Die Jacobi-Matrix des 2-2-1-Netzes im Code prüfen"}

<!--en:0b41a912-->
[Prüfung mit Autograd]{.kicker}

<!--en:05c5b73e-->
Für ein Netz mit $K$ Schichten $\mathbf{f} = \mathbf{f}_K \circ \cdots \circ \mathbf{f}_1$ ergibt wiederholtes Anwenden der Kettenregel $\mathbf{J} = \mathbf{J}_K \mathbf{J}_{K-1} \cdots \mathbf{J}_1$. Gewichtsmatrizen $\mathbf{W}_k$ wechseln sich mit diagonalen Aktivierungsmasken ab, deren Anwendung $O(n)$ Operationen kostet: $\operatorname{diag}(\boldsymbol{\varphi}_k')$.

<!--en:f64209b0-->
::: {.slide title="Produkte von Jacobi-Matrizen in tiefen Netzen: verschwindende und explodierende Normen"}

<!--en:5695923a-->
[Skalierung durch das Spektrum]{.kicker}

<!--en:f992edef-->
- **Exponentielle Abhängigkeit von der Tiefe ($\mathbf{J} = \mathbf{J}_K \cdots \mathbf{J}_1$):** Bei wiederholter Multiplikation mit $\mathbf{W} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ skaliert die Norm von $\mathbf{W}^K = \mathbf{V}\boldsymbol{\Lambda}^K\mathbf{V}^{-1}$ wie $|\lambda_{\max}|^K$.

<!--en:439bb8d8-->
- **Bei Tiefe $K = 50$:** $|\lambda_{\max}| = 0.9 \implies 0.9^{50} \approx 0.005$ (verschwindend), $|\lambda_{\max}| = 1.1 \implies 1.1^{50} \approx 117$ (explodierend).

<!--en:89772c56-->
- **Gradienten nach Eingaben und Gewichten:** $\mathbf{J}_K \cdots \mathbf{J}_1$ liefert $\partial g/\partial\mathbf{x}$. Das Training benötigt $\nabla_{\mathbf{W}_k}\ell$.

<!--en:72f7e9ca-->
::: {.slide title="Assoziativität und der Kostenfaktor 670"}

<!--en:a688e324-->
[Multiplikationsreihenfolge]{.kicker}

<!--en:db9d17e9-->
- **Gleiches Produkt, verschiedene Klammerungen:** Für die Ableitung eines skalaren Verlusts $\ell$ multiplizieren wir den $1 \times h$-Zeilenvektor $\mathbf{u}^\top = \partial\ell/\partial\mathbf{h}_3$ mit $\mathbf{J}_3\mathbf{J}_2\mathbf{J}_1$ (mit $h = n = 1000$).

<!--en:d649d55e-->
- **Von links nach rechts (Rückwärtsmodus / VJP):** Bei $(((\mathbf{u}^\top \mathbf{J}_3)\mathbf{J}_2)\mathbf{J}_1)$ ist jedes Zwischenergebnis ein $1 \times h$-Zeilenvektor. Drei Matrix-Vektor-Produkte kosten $h^2 + h^2 + hn = 3 \times 10^6$ Multiplikationen.

<!--en:a6cb1ef8-->
- **Von rechts nach links (teure Matrix-Matrix-Produkte):** Berechnen wir zuerst $\mathbf{J}_2\mathbf{J}_1$ und $\mathbf{J}_3(\mathbf{J}_2\mathbf{J}_1)$, kostet das gesamte Produkt $2h^2n + hn = 2.001 \times 10^9$ Multiplikationen. Der **Mehraufwand beträgt den Faktor $667\times$ ($2h/3$)**.

<!--en:764cda10-->
::: {.slide title="Rechnen mit Differentialen: Eine Regel ersetzt Formeltabellen"}

<!--en:13e64b06-->
[Differentiale erster Ordnung]{.kicker}

<!--en:324d75e8-->
- **Schritt 1: $df$ entwickeln.** Wir nutzen die Linearität, die Produktregel für Matrizen $d(\mathbf{A}\mathbf{B}) = (d\mathbf{A})\mathbf{B} + \mathbf{A}(d\mathbf{B})$, die Transposition $d(\mathbf{A}^\top) = (d\mathbf{A})^\top$ und die elementweise Regel $d\varphi(\mathbf{X}) = \varphi'(\mathbf{X})\odot d\mathbf{X}$.

<!--en:4a59db87-->
- **Schritt 2: In die Standardform des inneren Produkts bringen.** Wir formen zu $df = \langle \mathbf{M}, d\mathbf{X}\rangle = \operatorname{tr}(\mathbf{M}^\top d\mathbf{X})$ um und lesen $\nabla_{\mathbf{X}} f = \mathbf{M}$ ab.

<!--en:47121926-->
::: {.slide title="Die Spurregel zum Freistellen von dX"}

<!--en:9b4b7f38-->
[Rechnen mit adjungierten Variablen]{.kicker}

<!--en:4aef26e1-->
- **Zyklische Invarianz der Spur ($\operatorname{tr}(\mathbf{U}\mathbf{V}) = \operatorname{tr}(\mathbf{V}\mathbf{U})$):**
  - Linker Faktor: $\langle \mathbf{A}, \mathbf{B}\mathbf{C}\rangle = \langle \mathbf{B}^\top\mathbf{A}, \mathbf{C}\rangle$.
  - Rechter Faktor: $\langle \mathbf{A}, \mathbf{B}\mathbf{C}\rangle = \langle \mathbf{A}\mathbf{C}^\top, \mathbf{B}\rangle$.
  - Hadamard-Maske: $\langle \mathbf{A}, \mathbf{B} \odot \mathbf{C}\rangle = \langle \mathbf{A} \odot \mathbf{B}, \mathbf{C}\rangle$.

<!--en:f450828a-->
::: {.slide title="Gradienten linearer und quadratischer Formen und der kleinsten Quadrate"}

<!--en:e0bca90d-->
[Beispielrechnungen]{.kicker}

<!--en:8808a9f1-->
- **1. Lineare Form $\mathbf{a}^\top\mathbf{x}$:** $df = \langle \mathbf{a}, d\mathbf{x}\rangle \implies \nabla_{\mathbf{x}}(\mathbf{a}^\top\mathbf{x}) = \mathbf{a}$.

<!--en:a20088bb-->
- **2. Quadratische Form $\mathbf{x}^\top\mathbf{A}\mathbf{x}$:** $df = \langle (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}, d\mathbf{x}\rangle \implies \nabla_{\mathbf{x}}(\mathbf{x}^\top\mathbf{A}\mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ ($= 2\mathbf{A}\mathbf{x}$ nur für $\mathbf{A} = \mathbf{A}^\top$).

<!--en:7c4f3b2d-->
- **3. Kleinste Quadrate $\|\mathbf{W}\mathbf{x} - \mathbf{y}\|_2^2$:** $d\ell = \langle 2\mathbf{r}\mathbf{x}^\top, d\mathbf{W}\rangle \implies \nabla_{\mathbf{W}}\ell = 2(\mathbf{W}\mathbf{x} - \mathbf{y})\mathbf{x}^\top$.

<!--en:15c5760f-->
::: {.slide title="Plausibilitätsprüfungen: Form, Skalarfall 1×1 und Test ohne Symmetrie"}

<!--en:2bbfdc94-->
[Fehlersuche bei Matrixgradienten]{.kicker}

<!--en:58ec089c-->
- **1. Form prüfen:** Deckt fehlende Transpositionen bei rechteckigen Matrizen ($p \neq q$) auf, versagt aber bei quadratischen Matrizen ($p = q$).

<!--en:4d887d7a-->
- **2. Skalarfall $1 \times 1$ prüfen:** Deckt fehlende Faktoren $2$ und Vorzeichenfehler auf, aber keine Transpositionsfehler.

<!--en:c7c6cd46-->
- **3. Numerischer Test ohne Symmetrie:** Der Vergleich mit Autograd an einer zufälligen, nicht symmetrischen quadratischen Matrix deckt *jeden* dieser Fehler auf.

<!--en:4c25d777-->
::: {.slide title="Die allgemeine Rückwärtsregel (VJP) einer Schicht"}

<!--en:03a307b1-->
[Adjungierte Variablen einer Schicht]{.kicker}

<!--en:9e452ada-->
- **Keine Jacobi-Matrizen mit $10^{12}$ Einträgen für Tensoren:** Die Schicht $\mathbf{Y} = \mathbf{F}(\mathbf{X})$ bildet die eingehende adjungierte Variable $\bar{\mathbf{Y}} = \nabla_{\mathbf{Y}}\ell$ direkt auf $\bar{\mathbf{X}} = \nabla_{\mathbf{X}}\ell$ ab.

<!--en:7e16b360-->
- **Dualität der adjungierten Variablen:** $\bar{\mathbf{X}}$ ist die einzige Matrix, für die $\langle \bar{\mathbf{X}}, d\mathbf{X}\rangle = \langle \bar{\mathbf{Y}}, d\mathbf{Y}\rangle$ für alle $d\mathbf{X}$ gilt.

<!--en:a3d8b867-->
- **VJP für Vektoren:** Gilt $d\mathbf{y} = \mathbf{J}\,d\mathbf{x}$, so folgt $\bar{\mathbf{x}} = \mathbf{J}^\top\bar{\mathbf{y}}$.

<!--en:61c352a7-->
::: {.slide title="Rückwärtsregeln für die Matrizenmultiplikation im Batch"}

<!--en:7d71c639-->
[Grundoperationen eines MLP]{.kicker}

<!--en:4748213d-->
- **Matrixprodukt im Batch ($\mathbf{Y} = \mathbf{X}\mathbf{W}$):** Aus $\langle \bar{\mathbf{Y}}, (d\mathbf{X})\mathbf{W} + \mathbf{X}(d\mathbf{W})\rangle$ folgen direkt $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}^\top$ und $\bar{\mathbf{W}} = \mathbf{X}^\top\bar{\mathbf{Y}}$.

<!--en:0cd524c4-->
- **Regeln für elementweise Operationen und Bias:** $\mathbf{Y} = \boldsymbol{\varphi}(\mathbf{X}) \implies \bar{\mathbf{X}} = \varphi'(\mathbf{X}) \odot \bar{\mathbf{Y}}$ und $\mathbf{Z} = \mathbf{Y} + \mathbf{1}\mathbf{b}^\top \implies \bar{\mathbf{b}} = \bar{\mathbf{Z}}^\top\mathbf{1}$.

<!--en:4ed495b8-->
::: {.slide title="Transpositionsfehler bei quadratischen Matrizen im Code erkennen"}

<!--en:1b718c47-->
[Numerische Prüfung]{.kicker}

<!--en:53ac6452-->
Wir prüfen die Rückwärtsregeln für das Matrixprodukt direkt. Dazu differenzieren wir die skalare Testfunktion $s = \langle \bar{\mathbf{Y}}, \mathbf{X}\mathbf{W}\rangle$ an zufälligen **quadratischen, nicht symmetrischen** $3 \times 3$-Matrizen und vergleichen mit plausiblen Transpositionsfehlern gleicher Form:

<!--en:16ad2ee2-->
Für jede hergeleitete Regel erscheint `True`, für jeden plausiblen, aber falschen Ansatz (`2 A x`, `Ybar W` und `(X^T Ybar)^T`) erscheint `False`. Prüfen Sie eigene Rückwärtsregeln immer an nicht symmetrischen Matrizen.

<!--en:9df08b40-->
::: {.slide title="Alle Gewichtsgradienten des 2-2-1-Netzes von Hand berechnen"}

<!--en:94e42837-->
[Rückwärts durch alle Schichten]{.kicker}

<!--en:fab30781-->
Wenden wir die drei lokalen VJP-Regeln ausgehend von $\bar{g} = 1$ an der Stelle $\mathbf{x} = (1, 1)^\top$ rückwärts an (mit $\mathbf{a} = (3, -2)^\top$, der Maske $\mathbf{m} = (1, 0)^\top$ und $\mathbf{h} = (3, 0)^\top$), erhalten wir alle Parametergradienten des 2-2-1-Netzes:

<!--en:e6d335c1-->
1. **Ausgabeschicht $g = \mathbf{w}_2^\top\mathbf{h}$:** $\bar{\mathbf{h}} = \bar{g}\mathbf{w}_2 = (2, -1)^\top$ und $\bar{\mathbf{w}}_2 = \bar{g}\mathbf{h} = (3, 0)^\top$.

<!--en:aee07695-->
2. **$\mathrm{ReLU}$ und Bias $\mathbf{h} = \boldsymbol{\varphi}(\mathbf{a})$, $\mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}$:** $\bar{\mathbf{a}} = \mathbf{m} \odot \bar{\mathbf{h}} = (2, 0)^\top$ und $\bar{\mathbf{b}} = \bar{\mathbf{a}} = (2, 0)^\top$.

<!--en:4bc45c8f-->
3. **Erste lineare Schicht:** $\bar{\mathbf{W}}_1 = \bar{\mathbf{a}}\mathbf{x}^\top$ $= \begin{pmatrix} 2 & 2 \\ 0 & 0 \end{pmatrix}$ und $\bar{\mathbf{x}} = \mathbf{W}_1^\top\bar{\mathbf{a}} = (2, 4)^\top$.

<!--en:dfa56fb2-->
::: {.slide title="Jacobi-Matrix der Softmax-Funktion und Konvexität von Log-Sum-Exp"}

<!--en:c389af16-->
[Ableitungen des Klassifikators]{.kicker}

<!--en:235473c4-->
- **Jacobi-Matrix der Softmax-Funktion:** Ableiten von $p_i = e^{z_i}/\sum_k e^{z_k}$ ergibt $\mathbf{J}_{\operatorname{softmax}}(\mathbf{z}) = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$. Wegen der Verschiebungsinvarianz liegt $\mathbf{1}$ im Kern.

<!--en:d23277f6-->
- **Statistischer Beweis der Konvexität von Log-Sum-Exp:** Wegen $\mathbf{p} = \nabla\operatorname{LSE}(\mathbf{z})$ gilt überall $\mathbf{v}^\top\mathbf{H}_{\operatorname{LSE}}\mathbf{v} = \operatorname{Var}_{I \sim \mathbf{p}}(v_I) \ge 0$.

<!--en:a74927d0-->
::: {.slide title="Softmax-Funktion und Kreuzentropie ergeben den Gradienten p − y"}

<!--en:09e7377e-->
[Gradient nach den Logits]{.kicker}

<!--en:4d9220bf-->
- **Kürzen von $1/p_i$:** Multiplizieren wir $\frac{\partial\ell}{\partial p_i} = -\frac{y_i}{p_i}$ mit $\mathbf{J}_{\operatorname{softmax}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$, kürzt sich $p_i$ im Nenner heraus.

<!--en:6eaf1fbd-->
- **Gradient nach den Logits:** $\nabla_{\mathbf{z}}\ell = \bigl(\sum_{i=1}^C y_i\bigr)\mathbf{p} - \mathbf{y}$ vereinfacht sich zu $\mathbf{p} - \mathbf{y}$, wenn $\sum_{i=1}^C y_i = 1$.

<!--en:80eb705c-->
::: {.slide title="Softmax-Spektrum prüfen und Fehler bei Two-Hot-Zielvektoren erkennen"}

<!--en:f989df99-->
[Prüfung im Code]{.kicker}

<!--en:513c7091-->
Bei den Logits $\mathbf{z} = (1, -0.5, 2)^\top$ hat die Jacobi-Matrix der Softmax-Funktion einen Eigenwert null (in Richtung $\mathbf{1}$) und zwei positive Eigenwerte. Für den unnormierten Two-Hot-Zielvektor $\mathbf{y} = (1, 1, 0)^\top$ ($\sum_i y_i = 2$) ergibt sich der Gradient $2\mathbf{p} - \mathbf{y}$ statt $\mathbf{p} - \mathbf{y}$:

<!--en:43215298-->
::: {.slide title="Verständnisfragen zur Matrixdifferentialrechnung"}

<!--en:b6d9a92f-->
[Wiederholung und Übung]{.kicker}

<!--en:712482ed-->
Beantworten Sie die drei Fragen durch Rechnen mit Differentialen, mithilfe der Verschiebungsregel für die Spur und der Kettenregel für Jacobi-Matrizen:

<!--en:dbe7b769-->
1. **Gradient bei der Matrixfaktorisierung:** Gegeben sei der Verlust $\ell(\mathbf{U}, \mathbf{V}) = \tfrac{1}{2}\|\mathbf{U}\mathbf{V} - \mathbf{X}\|_F^2$ mit dem Residuum $\mathbf{R} = \mathbf{U}\mathbf{V} - \mathbf{X} \in \mathbb{R}^{m \times n}$ sowie $\mathbf{U} \in \mathbb{R}^{m \times r}$ und $\mathbf{V} \in \mathbb{R}^{r \times n}$. Bestimmen Sie mithilfe von $d\ell = \langle \mathbf{R}, (d\mathbf{U})\mathbf{V} + \mathbf{U}(d\mathbf{V})\rangle$ die Gradienten $\nabla_{\mathbf{U}}\ell$ und $\nabla_{\mathbf{V}}\ell$.

<!--en:1e08faa9-->
2. **Aufwand des VJP der Softmax-Funktion:** Bei Self-Attention über $C$ Klassen oder Token erfordert das Bilden von $\mathbf{J}_{\operatorname{softmax}} \in \mathbb{R}^{C \times C}$ und die Multiplikation mit $\mathbf{u}$ insgesamt $O(C^2)$ Operationen. Wie schnell lässt sich $\mathbf{J}_{\operatorname{softmax}}^\top\mathbf{u} = (\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)\mathbf{u}$ auswerten, ohne $\mathbf{J}$ zu bilden?

<!--en:043b5c49-->
3. **Parameter einer toten ReLU-Einheit:** In unserem 2-2-1-Netz mit $\mathrm{ReLU}$ hatte die versteckte Einheit 2 an der Stelle $\mathbf{x} = (1, 1)^\top$ die Präaktivierung $a_2 = -2 < 0$. Für welche Parameter der Einheit 2 war der Gradient an diesem Beispiel von null verschieden?

<!--en:557b98b6-->
**Antworten:**

<!--en:bef67de6-->
1. Die Verschiebungsregel für die Spur ergibt $d\ell = \langle \mathbf{R}\mathbf{V}^\top, d\mathbf{U}\rangle + \langle \mathbf{U}^\top\mathbf{R}, d\mathbf{V}\rangle$. Also gelten $\nabla_{\mathbf{U}}\ell = \mathbf{R}\mathbf{V}^\top \in \mathbb{R}^{m \times r}$ und $\nabla_{\mathbf{V}}\ell = \mathbf{U}^\top\mathbf{R} \in \mathbb{R}^{r \times n}$. Beide Gradienten haben die Form ihrer jeweiligen Variablen.

<!--en:16ff99d8-->
2. Die Umformung $(\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)\mathbf{u}$ $= \mathbf{p} \odot \bigl(\mathbf{u} - (\mathbf{p}^\top\mathbf{u})\mathbf{1}\bigr)$ erfordert nur ein Skalarprodukt $\mathbf{p}^\top\mathbf{u}$, eine Subtraktion mit Broadcasting und eine elementweise Multiplikation. **Zeit- und Speicherbedarf betragen $O(C)$** statt $O(C^2)$.

<!--en:208a945b-->
3. **Für keinen:** Zeile 2 von $\bar{\mathbf{W}}_1$ ist $(0, 0)$, es gilt $\bar{b}_2 = 0$ (wegen $m_2 = 0$) und $\bar{w}_{2,2} = h_2 = 0$. Hat eine $\mathrm{ReLU}$-Einheit im gesamten Trainingsdatensatz negative Präaktivierungen, aktualisiert der Gradientenabstieg ihre Gewichte nie („tote $\mathrm{ReLU}$“).

<!--en:1f9deb9b-->
::: {.slide title="Zusammenfassung: Matrixdifferentialrechnung und Rückwärtsregeln für Schichten"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:58d4cc85-->
| Operation oder Begriff | Formel im Vorwärtsdurchlauf | Jacobi-Matrix oder Rückwärtsregel (VJP) |

<!--en:0ba85676-->
| **Linearisierung einer Vektorabbildung** | $\mathbf{y} = \mathbf{f}(\mathbf{x}) : \mathbb{R}^n \to \mathbb{R}^m$ | $\mathbf{f}(\mathbf{x}+\boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}_{\mathbf{f}}\boldsymbol{\delta}$, $[\mathbf{J}_{\mathbf{f}}]_{ij} = \frac{\partial f_i}{\partial x_j}$ |

<!--en:f36a3421-->
| **Kettenregel und VJP-Reihenfolge** | $\mathbf{f} = \mathbf{f}_K \circ \cdots \circ \mathbf{f}_1$ | $\mathbf{u}^\top\mathbf{J} = ((\mathbf{u}^\top\mathbf{J}_K)\cdots)\mathbf{J}_1$ ($O(Kh^2)$ gegenüber $O(Kh^3)$ von rechts nach links) |

<!--en:9008e8e7-->
| **Differential und Verschiebungsregel** | $df = \langle \nabla_{\mathbf{X}}f, d\mathbf{X}\rangle = \operatorname{tr}((\nabla_{\mathbf{X}}f)^\top d\mathbf{X})$ | $\langle \mathbf{A}, \mathbf{B}\mathbf{C}\rangle = \langle \mathbf{B}^\top\mathbf{A}, \mathbf{C}\rangle$ $= \langle \mathbf{A}\mathbf{C}^\top, \mathbf{B}\rangle$ |

<!--en:3c8a69e8-->
| **Quadratische Form** | $f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}\mathbf{x}$ | $\nabla_{\mathbf{x}} f = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ ($= 2\mathbf{A}\mathbf{x}$ genau dann, wenn $\mathbf{A} = \mathbf{A}^\top$) |

<!--en:3b908d6a-->
| **Lineare Schicht im Batch** | $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^\top$ | $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}^\top$, $\bar{\mathbf{W}} = \mathbf{X}^\top\bar{\mathbf{Y}}$, $\bar{\mathbf{b}} = \bar{\mathbf{Y}}^\top\mathbf{1}$ |

<!--en:a29b74ac-->
| **Elementweise Aktivierung** | $\mathbf{Y} = \boldsymbol{\varphi}(\mathbf{X})$ | $\bar{\mathbf{X}} = \varphi'(\mathbf{X}) \odot \bar{\mathbf{Y}}$ (diagonale Maske mit $O(n)$ Operationen) |

<!--en:f93de165-->
| **Softmax-Funktion und Kreuzentropie** | $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$, $\ell = -\mathbf{y}^\top\log\mathbf{p}$ | $\mathbf{J}_{\operatorname{softmax}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ und $\nabla_{\mathbf{z}}\ell = \mathbf{p} - \mathbf{y}$ (wenn $\sum_i y_i = 1$) |
