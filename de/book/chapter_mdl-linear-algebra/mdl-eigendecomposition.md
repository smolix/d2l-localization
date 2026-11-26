<!--en:57266ae0-->
# Eigenwertzerlegungen

<!--en:f32cc06a-->
In :numref:`sec_mdl-linear-maps` haben wir gefragt, wie eine Matrix den Raum bei einmaliger Anwendung verändert. Bei vielen Berechnungen im maschinellen Lernen wenden wir dieselbe Matrix jedoch immer wieder an. Ein tiefes lineares Netz mit identischen Schichten berechnet nach 50 Schichten $\mathbf A^{50}\mathbf x$. Beim Gradientenabstieg (engl. gradient descent) für eine quadratische Zielfunktion multipliziert sich der Fehler in jedem Schritt mit derselben festen Matrix. In beiden Fällen wollen wir wissen, ob das Signal wächst, abklingt oder sich stabilisiert und wie schnell das geschieht. Wenn wir die Matrixpotenzen ausmultiplizieren, erhalten wir Zahlen, aber keine Erklärung.

<!--en:0639cfe4-->
Im einfachsten Fall stellt sich diese Frage bei einer Irrfahrt (engl. random walk) zwischen zwei Webseiten. Nehmen wir an, eine Nutzerin ist auf Seite 1. Sie bleibt auf Seite 1 oder klickt auf Seite 2, jeweils mit Wahrscheinlichkeit $1/2$. Auf Seite 2 bleibt sie mit Wahrscheinlichkeit $3/4$ und kehrt mit Wahrscheinlichkeit $1/4$ auf Seite 1 zurück. Der Vektor $\mathbf p_k$ enthalte für jede Seite die Wahrscheinlichkeit, nach $k$ Klicks dort zu sein. Bei jedem Klick multiplizieren wir diesen Vektor mit einer festen Matrix,

<!--en:2c3b1c80-->
deren Spalten die Übergangswahrscheinlichkeiten von der jeweiligen Seite angeben. Die Einträge jeder Spalte summieren sich daher zu eins. Startet die Nutzerin auf Seite 1 mit $\mathbf p_0=(1,0)$, so lauten die ersten Verteilungen $(0.5,0.5)$, $(0.375,0.625)$, $(0.344,0.656)$ und $(0.336,0.664)$. Die Änderungen werden mit jedem Klick kleiner, als würde sich die Irrfahrt auf eine feste Verteilung einpendeln. Wo befindet sich die Nutzerin nach 50 Klicks und wie schnell konvergiert die Verteilung? PageRank stellt dieselbe Frage für das gesamte Web.

<!--en:7d044185-->
Dazu drücken wir den Startvektor in den Koordinaten einiger besonderer Richtungen aus, der Eigenvektoren. Entlang jeder dieser Richtungen wirkt die Matrix wie die Multiplikation mit einer Zahl. In diesen Koordinaten entwickelt sich jede Komponente unabhängig von den anderen gemäß einer skalaren Potenz $\lambda^k$.

<!--en:b70522ee-->
In :numref:`sec_linear-algebra` haben wir Eigenvektoren als Richtungen kennengelernt, entlang derer eine Matrix lediglich streckt. Die symmetrische Matrix $\mathbf S=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ streckt $(1,1)$ um den Faktor 3 und $(1,-1)$ um den Faktor 1. Für jede symmetrische Matrix galt $\mathbf A^k\mathbf x=\sum_i c_i\lambda_i^k\mathbf v_i$. Hier lassen wir die Symmetriebedingung fallen. Wir bestimmen die Eigenwerte (engl. eigenvalues) einer beliebigen quadratischen Matrix und untersuchen, wann es eine Basis aus Eigenvektoren gibt und was an ihre Stelle tritt, wenn sie fehlt. Anschließend beweisen wir, warum Symmetrie so viel mehr garantiert.

<!--en:7eca0eaa-->
Drei Ergebnisse sind besonders wichtig: die Entwicklung :eqref:`eq_mdl-eig-power-expansion` von $\mathbf A^k\mathbf x$ nach Eigenvektoren, der Spektralsatz (engl. spectral theorem) und die Darstellung der quadratischen Form als gewichtete Summe von Quadraten in :eqref:`eq_mdl-quadform`. Wer es eilig hat, kann von :numref:`subsec_mdl-diagonalization` direkt zu :numref:`subsec_mdl-spectral-theorem` springen. Zwei praktische Fragen behandeln wir in :numref:`sec_mdl-spectra-in-practice`: Wie lassen sich Eigenwerte großer Matrizen eingrenzen und berechnen? Und wann können Eigenwerte in die Irre führen? In diesem Abschnitt rechnen wir durchgehend mit NumPy. Die Funktion `eig` liefert auch die komplexen Eigenwerte einer reellen Matrix, sofern sie welche besitzt.

<!--en:9e830aa9-->
## Eigenwerte und Diagonalisierung

<!--en:bbd9d45a-->
### Eigenvektoren und Eigenwerte

<!--en:3fb5db39-->
Bestimmte Eingabevektoren skaliert $\mathbf P$ lediglich, als wäre die Matrix eine einzelne Zahl. Wenden wir $\mathbf P$ auf $(1,2)$ an, so erhalten wir $\mathbf P(1,2)=(\tfrac12+\tfrac12,\ \tfrac12+\tfrac32)=(1,2)$. Der Vektor bleibt also unverändert. Wenden wir dagegen $\mathbf P$ auf $(1,-1)$ an, ergibt sich $(\tfrac12-\tfrac14,\ \tfrac12-\tfrac34)=\tfrac14(1,-1)$. Entlang der ersten Richtung multipliziert $\mathbf P$ mit $1$, entlang der zweiten mit $1/4$.

<!--en:7547fdec-->
Allgemein heißt ein von null verschiedener Vektor $\mathbf v$ *Eigenvektor* einer quadratischen Matrix $\mathbf A$ zum *Eigenwert* $\lambda$, wenn

<!--en:8f490631-->
Geometrisch gesehen streckt $\mathbf A$ einen Eigenvektor lediglich um den Faktor $\lambda$. Für $\lambda<0$ kehrt sich dabei seine Richtung um. Für die Matrix $\mathbf P$ der Nutzerin lauten die Eigenpaare $(\lambda,\mathbf v)$ also $(1,(1,2))$ und $(\tfrac14,(1,-1))$. Den Nullvektor schließen wir als Eigenvektor aus, da er :eqref:`eq_mdl-eigpair` für jedes $\lambda$ erfüllt. Alle Lösungen von $\mathbf A\mathbf v=\lambda\mathbf v$ zu einem festen $\lambda$ bilden zusammen mit $\mathbf 0$ einen linearen Unterraum, den *Eigenraum* (engl. eigenspace) zu $\lambda$.

<!--en:1ad31c5d-->
Der Eigenwert $\lambda=0$ ist dagegen zulässig. Er bedeutet, dass $\mathbf A\mathbf v=\mathbf 0$ für ein von null verschiedenes $\mathbf v$ gilt. Eine quadratische Matrix ist also genau dann singulär, wenn $0$ zu ihren Eigenwerten gehört. Eine reelle Matrix muss auch keine reellen Eigenvektoren besitzen. Eine Drehung um $90^\circ$ ändert beispielsweise die Richtung jedes vom Nullvektor verschiedenen Vektors in der Ebene. Ihre Eigenwerte sind komplex. Diesen Fall betrachten wir in :numref:`subsec_mdl-complex-rotation`.

<!--en:21f79e0e-->
Eigenvektoren kann man leicht mit den Achsen der Ellipse verwechseln, die entsteht, wenn eine invertierbare $2\times2$-Matrix auf den Einheitskreis wirkt. :numref:`fig_mdl-la-eig-ellipse` zeigt den Unterschied. Bei der symmetrischen Matrix $\mathbf S$ liegen die Eigenvektoren tatsächlich auf den Ellipsenachsen. Bei der oberen Dreiecksmatrix $\left(\begin{smallmatrix}1&1\\0&2\end{smallmatrix}\right)$ ist das anders. Ihre Eigenvektoren sind $(1,0)$ zu $\lambda=1$ und $(1,1)$ zu $\lambda=2$, während die Ellipsenachsen Winkel von $58^\circ$ und $-32^\circ$ bilden (ihre Berechnung zeigt :numref:`sec_mdl-svd-low-rank`). Die Ellipsenachsen geben die Richtungen der größten und kleinsten Streckung im Bild an. Die Eigenvektoren kennzeichnen die Geraden, die bei der Abbildung ihre Richtung behalten. Bei symmetrischen Matrizen stimmen diese Richtungen stets überein, wie der Spektralsatz in :numref:`subsec_mdl-spectral-theorem` zeigt. Für allgemeine Matrizen gilt das nicht unbedingt.

<!--en:a18eae54-->
![Der Einheitskreis (gestrichelt) und sein Bild (blau). Links liegen die normierten Eigenvektoren (grün) der symmetrischen Matrix $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ auf den Ellipsenachsen. Ihre Bilder (rot) sind um die Faktoren 3 und 1 skaliert (bei $\lambda=1$ fallen beide Pfeile zusammen). Rechts liegen die Eigenvektoren der oberen Dreiecksmatrix $\left(\begin{smallmatrix}1&1\\0&2\end{smallmatrix}\right)$, nämlich $(1,0)$ und $(1,1)$, nicht auf den Ellipsenachsen (grau).](../img/mdl-la-eig-ellipse.svg)

<!--en:be5929af-->
### Das charakteristische Polynom

<!--en:a3f12851-->
Die Eigenvektoren von $\mathbf P$ konnten wir direkt ablesen. Bei größeren Matrizen kommen wir damit nicht weit. Um die Eigenwerte einer beliebigen quadratischen Matrix systematisch zu bestimmen, bringen wir $\lambda\mathbf v$ auf die linke Seite von :eqref:`eq_mdl-eigpair` und klammern $\mathbf v$ aus:

<!--en:187a344a-->
Eine von null verschiedene Lösung $\mathbf v$ existiert genau dann, wenn $\mathbf A-\lambda\mathbf I$ einen nichttrivialen Nullraum besitzt. Das ist gleichbedeutend damit, dass die Matrix singulär ist, also $\det(\mathbf A-\lambda\mathbf I)=0$ gilt. Entwickeln wir diese Determinante, erhalten wir ein Polynom vom Grad $n$ in $\lambda$, das *charakteristische Polynom* $p(\lambda)=\det(\mathbf A-\lambda\mathbf I)$. Seine Nullstellen sind genau die Eigenwerte von $\mathbf A$. Kennen wir eine Nullstelle $\lambda$, erhalten wir ihren Eigenraum durch Lösen des homogenen linearen Gleichungssystems :eqref:`eq_mdl-eigvalue_der`.

<!--en:7ae7175b-->
Wenden wir dieses Verfahren auf die Matrix der Nutzerin an:

<!--en:4c5c9a7c-->
Die Nullstellen sind $1$ und $1/4$, genau die beiden Skalierungsfaktoren, die uns zuvor aufgefallen sind. Da $\mathbf P-\lambda\mathbf I$ an jeder Nullstelle singulär ist, genügt hier eine einzige Zeile, um einen Eigenvektor zu finden. Für $\lambda=1$ liefert die erste Zeile von $\mathbf P-\mathbf I$ die Gleichung $-\tfrac12v_1+\tfrac14v_2=0$. Also gilt $v_2=2v_1$ und wir können $\mathbf v=(1,2)$ wählen. Für $\lambda=\tfrac14$ ergibt die erste Zeile von $\mathbf P-\tfrac14\mathbf I$ die Gleichung $\tfrac14v_1+\tfrac14v_2=0$ und damit $\mathbf v=(1,-1)$. Jeder Eigenvektor ist nur bis auf einen von null verschiedenen skalaren Faktor bestimmt. Ist $\mathbf v$ ein Eigenvektor, so ist auch $c\mathbf v$ für jedes $c\neq0$ ein Eigenvektor, da sich beide Seiten von :eqref:`eq_mdl-eigpair` mit $c$ multiplizieren.

<!--en:73bc5548-->
In der Praxis überlassen wir diese Rechnung einer numerischen Bibliothek. Die NumPy-Funktion `eig` liefert die Eigenwerte sowie eine Matrix, deren Spalten die zugehörigen Eigenvektoren sind. Die letzte Zeile unten prüft :eqref:`eq_mdl-eigpair` für alle Paare zugleich.

<!--en:df74775f-->
Reihenfolge, Skalierung und Vorzeichen hängen von der Bibliothek ab. Hier liefert `eig` die Eigenwerte in der Reihenfolge $1/4$ und $1$. Die Spalten sind Einheitsvektoren parallel zu $(1,-1)$ beziehungsweise $(1,2)$.

<!--en:af6b5808-->
Numerische Bibliotheken stellen das charakteristische Polynom nicht explizit auf, um seine Nullstellen zu bestimmen. Für $n\ge5$ gibt es keine allgemeine geschlossene Formel, die alle Nullstellen durch Wurzelausdrücke angibt. Schon für mäßig große $n$ reagieren die Nullstellen äußerst empfindlich auf kleinste Rundungsfehler in den Koeffizienten. Stattdessen bringt `eig` die Matrix durch eine Folge orthogonaler Basiswechsel schrittweise auf Dreiecksform, ohne ihre Eigenwerte zu ändern. Bei einer Dreiecksmatrix stehen die Eigenwerte auf der Diagonale: Subtrahieren wir $\lambda\mathbf I$, bleibt die Dreiecksform erhalten, und die Determinante einer Dreiecksmatrix ist das Produkt ihrer Diagonaleinträge (:numref:`subsec_mdl-determinant-general`). Bei Matrizen, die für eine Zerlegung zu groß sind, bestimmt die Potenziteration (:numref:`subsec_mdl-power-iteration`) dominante Eigenwerte allein mit Matrix-Vektor-Produkten.

<!--en:29539ed5-->
### Diagonalisierung

<!--en:194a2070-->
Nun können wir unsere Ausgangsfrage zu $\mathbf P^{50}\mathbf p_0$ beantworten. Wir schreiben die beiden Eigenvektoren von $\mathbf P$ als Spalten einer Matrix $\mathbf V$ und setzen ihre Eigenwerte auf die Diagonale von $\boldsymbol\Lambda$:

<!--en:5af99181-->
Schreiben wir die beiden Eigenvektorgleichungen $\mathbf P\mathbf v_i=\lambda_i\mathbf v_i$ als Spalten nebeneinander, so erhalten wir die kompakte Gleichung $\mathbf P\mathbf V=\mathbf V\boldsymbol\Lambda$. Wegen $\det\mathbf V=-1-2=-3\neq0$ ist die Matrix $\mathbf V$ invertierbar und es gilt $\mathbf V^{-1}=\tfrac13\left(\begin{smallmatrix}1&1\\2&-1\end{smallmatrix}\right)$. Multiplikation von rechts mit $\mathbf V^{-1}$ ergibt

<!--en:c013e877-->
Von rechts nach links gelesen beschreibt diese Zerlegung drei Schritte: Wir wechseln zu Koordinaten in der Eigenbasis ($\mathbf V^{-1}$), skalieren jede Koordinate unabhängig mit ihrem Eigenwert ($\boldsymbol\Lambda$) und wechseln zurück zu den ursprünglichen Koordinaten ($\mathbf V$). Eine quadratische Matrix $\mathbf A$ heißt *diagonalisierbar*, wenn sie sich als $\mathbf A=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$ mit einer invertierbaren Matrix $\mathbf V$ und einer Diagonalmatrix $\boldsymbol\Lambda$ schreiben lässt. Die Zerlegung selbst heißt *Eigenwertzerlegung* (engl. eigendecomposition). Sie existiert genau dann, wenn $\mathbf A$ insgesamt $n$ linear unabhängige Eigenvektoren besitzt, die wir als Spalten von $\mathbf V$ verwenden können.

<!--en:9c027785-->
Potenzen einer diagonalisierbaren Matrix lassen sich nun leicht berechnen. In einem Produkt aus $k$ Faktoren $\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$ kürzt sich jedes benachbarte Paar $\mathbf V^{-1}\mathbf V$ heraus:

<!--en:f1a41cf9-->
Potenzen einer diagonalisierbaren Matrix lassen sich also auf Potenzen von Skalaren zurückführen. Wenden wir $\mathbf A^k$ auf einen Vektor $\mathbf x$ an und schreiben $\mathbf c=\mathbf V^{-1}\mathbf x$ für die Koordinaten von $\mathbf x$ in der Eigenbasis (also $\mathbf x=\sum_i c_i\mathbf v_i$), so erhalten wir

<!--en:062341cf-->
Jede Koordinate in der Eigenbasis folgt unabhängig von den anderen der geometrischen Folge $c_i\lambda_i^k$. Startet die Nutzerin auf Seite 1, gilt $\mathbf c=\mathbf V^{-1}\mathbf p_0=\tfrac13(1,2)=(\tfrac13,\tfrac23)$. Damit erhalten wir eine geschlossene Formel für die Verteilung nach einer beliebigen Anzahl von Klicks $k$:

<!--en:15192317-->
Damit ist unsere Ausgangsfrage vollständig beantwortet. Mit wachsendem $k$ verschwindet der zweite Term und die Verteilung nähert sich $\boldsymbol\pi=(1/3,2/3)$. Auf lange Sicht verbringt die Nutzerin doppelt so viel Zeit auf Seite 2 wie auf Seite 1. Der verbleibende Fehler hat in jedem Eintrag den Betrag $\tfrac23\,4^{-k}$ und zeigt stets entlang des Eigenvektors $(1,-1)$. Jeder Klick teilt den Fehler durch $4$. Nach 10 Klicks ist er auf $6\times10^{-7}$ gesunken, nach 50 Klicks liegt er unter $10^{-30}$. Zur Kontrolle setzen wir $k=3$ und erhalten $(\tfrac13+\tfrac1{96},\ \tfrac23-\tfrac1{96})=(0.34375,0.65625)$, genau wie bei unserem dritten Iterationsschritt zu Beginn. :numref:`fig_mdl-la-markov-iterates` veranschaulicht die Geometrie. Da sich die Einträge jedes Vektors $\mathbf p_k$ zu eins summieren, bewegen sich die Iterierten auf der Geraden $x+y=1$ zu dem Punkt, an dem der Eigenvektor zum Eigenwert $\lambda=1$ die Gerade trifft.

<!--en:2bf48901-->
![Die Verteilungen $\mathbf p_k$ der Nutzerin bewegen sich auf $x+y=1$ auf den Punkt $\boldsymbol\pi=(1/3,2/3)$ zu. Dort trifft der Eigenvektor $(1,2)$ zum Eigenwert $\lambda=1$ die Gerade. Der Fehler zeigt entlang des Eigenvektors $(1,-1)$ zum Eigenwert $\lambda=1/4$ und schrumpft mit jedem Klick um den Faktor 4.](../img/mdl-la-markov-iterates.svg)

<!--en:85118567-->
Prüfen wir, ob fünfzig Matrizenmultiplikationen und die Entwicklung nach den beiden Eigenvektoren dasselbe $\mathbf p_{50}$ ergeben. Bestimmen wir außerdem das Verhältnis aufeinanderfolgender Fehler.

<!--en:2c5b619d-->
Beide Rechnungen stimmen überein, und das Verhältnis aufeinanderfolgender Fehler beträgt stets $0.25$. Der Eigenwert $1$ bestimmt den Anteil von $\mathbf p_0$, der erhalten bleibt, und der Eigenwert $1/4$ den Anteil, der abklingt.

<!--en:27981af0-->
Dasselbe Verfahren eignet sich für andere Funktionen einer diagonalisierbaren Matrix. Ist kein Eigenwert null, gilt $\mathbf A^{-1}=\mathbf V\boldsymbol\Lambda^{-1}\mathbf V^{-1}$. Wir bilden dazu den Kehrwert jedes Eigenwerts auf der Diagonale. Analog erhalten wir das Matrixexponential $e^{\mathbf A t}=\mathbf V e^{\boldsymbol\Lambda t}\mathbf V^{-1}$, indem wir die Exponentialfunktion auf jeden Eigenwert mit dem Zeitfaktor anwenden. Es liefert die Lösung der linearen Differentialgleichung $\dot{\mathbf x}=\mathbf A\mathbf x$. Einzelheiten behandelt :numref:`sec_mdl-odes-solvers`.

<!--en:c69ce60d-->
**Ähnlichkeit.** Die Eigenwertzerlegung zeigt, dass $\mathbf P$ und $\boldsymbol\Lambda$ dieselbe lineare Abbildung in zwei verschiedenen Basen darstellen. Allgemein heißen zwei Matrizen *ähnlich*, wenn $\mathbf A'=\mathbf B\mathbf A\mathbf B^{-1}$ für eine invertierbare Matrix $\mathbf B$ gilt. Der folgende Satz bestätigt, dass ein Basiswechsel die Eigenwerte einer Matrix erhält. Er beruht auf der Multiplikativität der Determinante aus :numref:`sec_mdl-linear-maps`,

<!--en:cb744fdd-->
woraus insbesondere $\det(\mathbf B^{-1})=1/\det\mathbf B$ folgt, wenn wir $\mathbf B\mathbf B^{-1}=\mathbf I$ einsetzen.

<!--en:1b5c7c3c-->
**Satz (ähnliche Matrizen haben dasselbe charakteristische Polynom).** *Für jede quadratische Matrix $\mathbf A$ und jede invertierbare Matrix $\mathbf B$ gilt*

<!--en:8d49e44c-->
**Beweis.** Wir schreiben $\lambda\mathbf I=\mathbf B(\lambda\mathbf I)\mathbf B^{-1}$ und klammern $\mathbf B$ und $\mathbf B^{-1}$ aus: $\det(\mathbf B\mathbf A\mathbf B^{-1}-\lambda\mathbf I)
=\det\bigl(\mathbf B(\mathbf A-\lambda\mathbf I)\mathbf B^{-1}\bigr)
=\det\mathbf B\,\det(\mathbf A-\lambda\mathbf I)\,\det\mathbf B^{-1}
=\det(\mathbf A-\lambda\mathbf I)$. $\blacksquare$

<!--en:e4ad40da-->
Ähnliche Matrizen besitzen somit dieselben Eigenwerte mit denselben Vielfachheiten als Nullstellen. Jede Größe, die das charakteristische Polynom festlegt, heißt *Ähnlichkeitsinvariante*. Die beiden nützlichsten Invarianten sind die Determinante und die Spur (engl. trace). Für jede diagonalisierbare Matrix gilt

<!--en:a2aa7212-->
Die Determinante ist somit das Produkt der Eigenwerte. Jede Volumenänderung durch $\mathbf V$ hebt $\mathbf V^{-1}$ wieder auf. Übrig bleibt nur die Skalierung der Achsen durch $\boldsymbol\Lambda$. Für die Spur nutzen wir ihre zyklische Eigenschaft (:numref:`subsec_mdl-einsum`),

<!--en:fd940d24-->
die für alle Matrizen $\mathbf X$ und $\mathbf Y$ gilt, deren Produkte quadratisch sind. Setzen wir $\mathbf X=\mathbf V$ und $\mathbf Y=\boldsymbol\Lambda\mathbf V^{-1}$, erhalten wir

<!--en:743470f9-->
Damit ist die Spur die Summe der Eigenwerte. An der Matrix der Nutzerin können wir beide Identitäten sofort prüfen: $\operatorname{tr}\mathbf P=\tfrac12+\tfrac34=\tfrac54=1+\tfrac14$ und $\det\mathbf P=\tfrac38-\tfrac18=\tfrac14=1\cdot\tfrac14$. Für jede $2\times2$-Matrix legen diese beiden Zahlen die Eigenwerte unmittelbar als Nullstellen von $\lambda^2-(\operatorname{tr})\lambda+\det$ fest.

<!--en:7ffdcb0d-->
Tatsächlich gelten beide Identitäten für *jede* quadratische Matrix, auch wenn sie nicht diagonalisierbar ist. Dazu zählen wir die Eigenwerte über $\mathbb C$ mit ihren Vielfachheiten als Nullstellen. Nach dem *Fundamentalsatz der Algebra* (den wir voraussetzen) zerfällt jedes Polynom vom Grad $n$ über $\mathbb C$ vollständig in Linearfaktoren:

<!--en:d37bb527-->
Setzen wir $\lambda=0$, folgt unmittelbar $\det\mathbf A=\prod_i\lambda_i$. Für die Spur vergleichen wir die Koeffizienten von $\lambda^{n-1}$ auf beiden Seiten. Rechts lautet der Koeffizient $(-1)^{n-1}\sum_i\lambda_i$. Links erreicht nur das Produkt der Diagonaleinträge von $\mathbf A-\lambda\mathbf I$ den Grad $n-1$. Daraus ergibt sich der Koeffizient $(-1)^{n-1}\sum_i a_{ii}$. Also gilt $\operatorname{tr}\mathbf A=\sum_i\lambda_i$. Da nichtreelle Eigenwerte einer reellen Matrix in komplex konjugierten Paaren auftreten, bleiben sowohl die Summe als auch das Produkt reell.

<!--en:a27d3c47-->
Schließlich entspricht der Rang einer diagonalisierbaren Matrix der Anzahl ihrer von null verschiedenen Eigenwerte, jeweils mit Vielfachheit gezählt. Die Multiplikation mit den invertierbaren Matrizen $\mathbf V$ und $\mathbf V^{-1}$ erhält die Dimensionen, also gilt $\operatorname{rank}\mathbf A=\operatorname{rank}\boldsymbol\Lambda$. Übung :exref:`ex_mdl-eigendecomposition-rank-zero-eigenvalues` untersucht den Fall einer nicht diagonalisierbaren Matrix.

<!--en:446c9fb2-->
### Algebraische und geometrische Vielfachheit

<!--en:6bea9831-->
Die Matrix der Nutzerin hatte zwei verschiedene Eigenwerte und lieferte zwei linear unabhängige Eigenvektoren. Das ist kein Zufall: Eigenvektoren zu verschiedenen Eigenwerten sind stets linear unabhängig.

<!--en:eee4e989-->
**Satz (Eigenvektoren zu verschiedenen Eigenwerten sind linear unabhängig).** *Eigenvektoren zu paarweise verschiedenen Eigenwerten sind linear unabhängig. Insbesondere ist jede $n\times n$-Matrix mit $n$ verschiedenen Eigenwerten diagonalisierbar.*

<!--en:9d7aed42-->
**Beweis.** Nehmen wir zum Widerspruch an, einige Eigenvektoren $\mathbf{v}_1,\ldots,\mathbf{v}_m$ zu verschiedenen Eigenwerten $\lambda_1,\ldots,\lambda_m$ seien linear abhängig. Unter allen nichttrivialen Linearkombinationen, die null ergeben, wählen wir eine mit möglichst wenigen von null verschiedenen Koeffizienten, $\sum_{i=1}^{m} c_i\mathbf{v}_i = \mathbf 0$. Bei Bedarf nummerieren wir so um, dass $c_m\neq0$ gilt. Mindestens zwei Koeffizienten müssen von null verschieden sein, denn aus $c_m\mathbf{v}_m=\mathbf 0$ allein würde $\mathbf{v}_m=\mathbf 0$ folgen. Wir wenden nun $\mathbf{A}$ auf die Gleichung an und subtrahieren das $\lambda_m$-Fache der ursprünglichen Gleichung:

<!--en:19e8c82c-->
Der Term mit $\mathbf{v}_m$ fällt weg. Die verbleibende Gleichung enthält weniger Terme und ist dennoch nichttrivial. Jeder andere von null verschiedene Koeffizient $c_i$ ($i<m$) bleibt als $c_i(\lambda_i-\lambda_m)\neq0$ erhalten, weil die Eigenwerte verschieden sind. Das widerspricht der Minimalität unserer Wahl. Besitzt $\mathbf A$ insgesamt $n$ verschiedene Eigenwerte, bilden die $n$ linear unabhängigen Eigenvektoren eine Basis von $\mathbb C^n$ (oder von $\mathbb R^n$, wenn alle Eigenwerte reell sind). Also ist $\mathbf{V}$ invertierbar. $\blacksquare$

<!--en:678684c9-->
Bei mehrfachen Eigenwerten müssen wir zwei Vielfachheiten unterscheiden. Die *algebraische Vielfachheit* (engl. algebraic multiplicity) eines Eigenwerts $\lambda$ gibt an, wie oft er als Nullstelle des charakteristischen Polynoms vorkommt. Wir bezeichnen sie mit $a(\lambda)$. Die Dimension seines Eigenraums, also die Anzahl der linear unabhängigen Eigenvektoren, die er beisteuert, heißt *geometrische Vielfachheit* (engl. geometric multiplicity) $g(\lambda)$.

<!--en:d92953d6-->
**Satz.** *Für jeden Eigenwert $\lambda$ einer quadratischen Matrix gilt $1\le g(\lambda)\le a(\lambda)$.*

<!--en:39d7ac7d-->
**Beweis.** Da $\mathbf A-\lambda\mathbf I$ singulär ist, besitzt diese Matrix einen nichttrivialen Nullraum. Daraus folgt $g\ge1$. Für die obere Schranke wählen wir eine Basis $\mathbf v_1,\ldots,\mathbf v_g$ des Eigenraums, ergänzen sie zu einer Basis von $\mathbb C^n$ und schreiben die Basisvektoren als Spalten von $\mathbf B$. Die ersten $g$ Spalten von $\mathbf A\mathbf B$ sind gerade $\lambda\mathbf v_1,\ldots,\lambda\mathbf v_g$. In der neuen Basis hat die Matrix daher obere Blockdreiecksform:

<!--en:e8b175ab-->
Nach :eqref:`eq_mdl-similar-charpoly` besitzt diese Matrix dasselbe charakteristische Polynom $p$ wie $\mathbf A$. Wir bezeichnen die Polynomvariable mit $t$, da $\lambda$ nun den festen Eigenwert bezeichnet. Beim Subtrahieren von $t\mathbf I$ bleibt die obere Blockdreiecksform erhalten, mit den Diagonalblöcken $(\lambda-t)\mathbf I_g$ und $\mathbf Y-t\mathbf I$. Die Determinante einer oberen Blockdreiecksmatrix ist das Produkt der Determinanten ihrer Diagonalblöcke (:numref:`subsec_mdl-determinant-general`). Damit folgt $p(t)=(\lambda-t)^g\det(\mathbf Y-t\mathbf I)$. Also teilt $(\lambda-t)^g$ das charakteristische Polynom. Das bedeutet $a\ge g$. $\blacksquare$

<!--en:3832abe5-->
Ein Eigenwert kann weniger linear unabhängige Eigenvektoren liefern, als seine Vielfachheit als Nullstelle angibt, aber niemals mehr. Wählen wir für jeden Eigenraum eine Basis und fassen wir alle gewählten Vektoren zusammen. Gemeinsam sind sie linear unabhängig. Ergibt eine Linearkombination dieser Vektoren null, so gruppieren wir die Terme nach Eigenwert. Jede Gruppe ergibt entweder null oder einen Eigenvektor. Nach dem Satz über verschiedene Eigenwerte muss daher jede Gruppe null ergeben. Eine Matrix besitzt somit höchstens $\sum_\lambda g(\lambda)$ linear unabhängige Eigenvektoren. Über $\mathbb C$ summieren sich die algebraischen Vielfachheiten zu $n$, und es gilt $g(\lambda)\le a(\lambda)$ für jedes $\lambda$. Daraus folgt ein einfaches Kriterium: *Die Eigenvektoren von $\mathbf A$ spannen $\mathbb C^n$ genau dann auf, wenn $g(\lambda)=a(\lambda)$ für jeden komplexen Eigenwert $\lambda$ gilt.*

<!--en:510fa46a-->
Damit eine reelle Matrix eine Basis aus *reellen* Eigenvektoren besitzt, müssen außerdem alle Eigenwerte reell sein. Die Drehung um $90^\circ$ mit der Matrix $\mathbf R_{90}=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$ zeigt, warum diese zusätzliche Bedingung nötig ist. Ihre Eigenwerte $\pm i$ erfüllen jeweils $g=a=1$. Sie besitzt also eine vollständige komplexe Eigenbasis, aber keinen einzigen reellen Eigenvektor.

<!--en:8eec5fab-->
Bei mehrfachen Eigenwerten kann die Diagonalisierbarkeit ganz verloren gehen. Das einfachste Gegenbeispiel ist die *Scherung* aus :numref:`subsec_mdl-composition`:

<!--en:414e3129-->
Ihr charakteristisches Polynom lautet $(1-\lambda)^2$. Der Eigenwert $\lambda=1$ hat also die algebraische Vielfachheit $2$. Die Gleichung $(\mathbf{T}-\mathbf{I})\mathbf{v}=\mathbf 0$ lautet jedoch $\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)\mathbf v=\mathbf 0$, woraus $v_2=0$ folgt. Der Eigenraum besteht nur aus der Geraden, die $(1,0)$ aufspannt. Die geometrische Vielfachheit beträgt also $1$. Es gibt keine Eigenbasis, und die Scherung ist nicht diagonalisierbar. Eine Matrix, für die bei mindestens einem Eigenwert $g(\lambda)<a(\lambda)$ gilt, heißt *defekt*. :numref:`fig_mdl-la-defective-shear` zeigt, warum ein Eigenvektor fehlt. Die Scherung verschiebt jede horizontale Gerade seitlich um einen ihrer Höhe proportionalen Betrag. Nur die horizontale Achse behält als Ursprungsgerade ihre Richtung bei. (Eine diagonalisierbare $2\times2$-Matrix mit doppeltem Eigenwert $\lambda$ müsste dagegen gleich $\lambda\mathbf I$ sein und würde jede Richtung erhalten.)

<!--en:abf8defe-->
![Die defekte Scherung in der Ebene. Jede horizontale Gerade verschiebt sich proportional zu ihrer Höhe nach rechts, sodass das Gitter kippt. Die $x$-Achse (grün) geht in sich selbst über. Es gilt $\mathbf{T}\mathbf{e}_1=\mathbf{e}_1$. Die horizontale Achse ist die einzige Eigenrichtung, mit $\lambda=1$. Der andere Basisvektor erhält eine horizontale Komponente, $\mathbf{T}\mathbf{e}_2=(1,1)$. Es bleibt keine zweite Richtung erhalten: Die geometrische Vielfachheit ist 1, die algebraische 2.](../img/mdl-la-defective-shear.svg)

<!--en:70272054-->
Wiederholen wir die Scherung, sehen wir, wie sich der Defekt auf die Matrixpotenzen auswirkt. Es gilt $\mathbf T^k=\left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$. Damit wächst $\mathbf T^k\mathbf e_2=(k,1)$ unbeschränkt, obwohl der einzige Eigenwert $1$ ist. Eine Entwicklung allein nach Eigenvektoren kann den linearen Faktor $k$ nicht erklären. Der nächste Unterabschnitt zeigt, woher er kommt.

<!--en:1d90e9b8-->
### Die jordansche Normalform

<!--en:9a1af67d-->
Ist die geometrische Vielfachheit zu klein, gibt es keine Eigenbasis. Welche Struktur tritt an ihre Stelle? Für den weiteren Verlauf des Buches ist vor allem eine Folge wichtig: Bei einem defekten Eigenwert tritt neben $\lambda^k$ ein Polynom in $k$ als Faktor auf. Über den komplexen Zahlen ist jede quadratische Matrix ähnlich zu einer Blockdiagonalmatrix aus *Jordanblöcken*

<!--en:3066d4b8-->
der Größe $m\times m$. Auf der Diagonale jedes Blocks steht ein einziger Eigenwert $\lambda$, auf der ersten Diagonale darüber steht überall $1$. Diese Zerlegung heißt *jordansche Normalform* :cite:`Horn.Johnson.2012`. Wir verwenden sie ohne Beweis. Nach :eqref:`eq_mdl-similar-charpoly` besitzt sie dieselben Eigenwerte wie die ursprüngliche Matrix.

<!--en:a1e2deb3-->
Jeder Jordanblock liefert genau einen linear unabhängigen Eigenvektor, nämlich seinen ersten Basisvektor. Die geometrische Vielfachheit von $\lambda$ ist daher die Anzahl der Blöcke zum Eigenwert $\lambda$, seine algebraische Vielfachheit die Summe ihrer Größen. Insbesondere ist eine Matrix genau dann diagonalisierbar, wenn jeder Jordanblock die Größe $1\times1$ hat. Die Scherung $\mathbf T$ aus :eqref:`eq_mdl-defective-shear` besteht aus dem einzelnen Block $\mathbf{J}_2(1)$.

<!--en:0f68a56c-->
Bei wiederholter Anwendung bewirkt ein Jordanblock mehr als eine reine Skalierung. Schreiben wir $\mathbf{J}_m(\lambda)=\lambda\mathbf{I}+\mathbf{N}$. Dabei hat $\mathbf{N}$ direkt oberhalb der Diagonale den Eintrag $1$ und sonst nur Nullen. Geometrisch wirkt $\mathbf N$ als Verschiebung: Jeder Basisvektor geht in seinen Vorgänger über, der erste in den Nullvektor. Daher gilt $\mathbf N^m=\mathbf 0$ (eine Matrix, von der eine Potenz verschwindet, heißt *nilpotent*). Da $\mathbf{I}$ und $\mathbf{N}$ kommutieren, können wir die Potenzen mit dem binomischen Lehrsatz entwickeln. Für $m=2$ enthalten alle Terme nach den ersten beiden den Faktor $\mathbf{N}^2=\mathbf 0$. Es bleibt

<!--en:3222bd81-->
Langfristig entscheidet der Eigenwert weiterhin über Wachstum oder Abklingen. Jetzt kommt jedoch ein Polynom in $k$ als Faktor hinzu (vom Grad $m-1$ bei einem Block der Größe $m\times m$). Für $|\lambda|=1$ driften die Iterierten wie bei der Scherung linear. Liegt $|\lambda|$ knapp unter $1$, geschieht etwas Überraschendes: Der Eintrag oben rechts, $k\lambda^{k-1}$, wächst zunächst, erreicht sein Maximum nahe $k\approx-1/\log|\lambda|$ und fällt erst dann gegen null. Für $\lambda=0.99$ liegt das Maximum nahe $k\approx100$, wo der Eintrag $100\cdot0.99^{99}\approx37$ erreicht. Prüfen wir diese Vorhersage numerisch.

<!--en:3cd4939b-->
Alle Eigenwerte von $\mathbf{J}$ sind gleich $0.99$. Dennoch steigt die Norm der Potenzen auf etwa $37$ bei $k=100$ und beträgt noch $27$ bei $k=200$, bevor schließlich der Exponentialfaktor $0.99^k$ überwiegt. Die Eigenwerte allein lassen dieses transiente Wachstum nicht erkennen. In :numref:`subsec_mdl-transient-growth` kommen wir darauf zurück.

<!--en:fc74a323-->
Jordanblöcke zeigen einen Grund, weshalb einer reellen Matrix eine reelle Eigenbasis fehlen kann: Sie hat zu wenige linear unabhängige Eigenvektoren. Der zweite Grund ist anderer Natur. Wenn eine reelle Matrix eine Ebene *dreht*, kann es in dieser Ebene keinen einzigen reellen Eigenvektor geben.

<!--en:d56e2042-->
### Komplexe Eigenwerte

<!--en:61406e46-->
Betrachten wir die Drehung in der Ebene um den Winkel $\theta$ (:numref:`subsec_mdl-orthogonal-matrices`),

<!--en:2218bb3a-->
Ist $\theta$ kein Vielfaches von $\pi$, dreht $\mathbf{R}_\theta$ jeden von null verschiedenen reellen Vektor. Es gibt also keinen reellen Eigenvektor. Das charakteristische Polynom $\lambda^2-2\cos\theta\,\lambda+1$ hat die komplex konjugierten Nullstellen $\lambda=\cos\theta\pm i\sin\theta=e^{\pm i\theta}$. Der Betrag $|\lambda|=1$ drückt aus, dass $\mathbf{R}_\theta$ Längen erhält. Das Argument $\pm\theta$ gibt den Drehwinkel an.

<!--en:12b2c821-->
Betrachten wir nun die $2\times2$-Matrix, die uns durch dieses Kapitel begleitet. An ihren Einträgen lässt sich keine Drehung unmittelbar erkennen:

<!--en:4c1e32b4-->
Ihr charakteristisches Polynom $\lambda^2-4\lambda+5$ hat die Nullstellen $\lambda=2\pm i=\sqrt5\,e^{\pm i\theta}$ mit $\theta=\arctan\tfrac12\approx26.6^\circ$. Um einen Eigenvektor zu $2+i$ zu finden, lesen wir die zweite Zeile von $\mathbf A-(2+i)\mathbf I$ ab. Sie ergibt $-w_1+(1-i)w_2=0$. Setzen wir $w_2=1$, erhalten wir

<!--en:4709f305-->
Wo steckt die Drehung? Betrachten wir eine beliebige reelle Matrix $\mathbf{A}$ mit einem komplexen Eigenpaar $\mathbf{A}\mathbf{w}=\lambda\mathbf{w}$, wobei $\lambda=a+ib$ und $b\neq0$ gelten. Zerlegen wir den Eigenvektor in Real- und Imaginärteil, $\mathbf{w}=\mathbf{u}+i\mathbf{v}$. Multiplizieren wir $\mathbf{A}(\mathbf{u}+i\mathbf{v})=(a+ib)(\mathbf{u}+i\mathbf{v})$ aus und vergleichen die Real- und Imaginärteile, so erhalten wir

<!--en:28826672-->
Beide Bildvektoren bleiben in $\operatorname{span}\{\mathbf{u},\mathbf{v}\}$. Dieser Unterraum ist also unter $\mathbf{A}$ invariant und bildet tatsächlich eine zweidimensionale Ebene. Wären $\mathbf u$ und $\mathbf v$ linear abhängig, so würden sie eine einzige Gerade aufspannen (wegen $\mathbf w\neq\mathbf 0$ nicht den Nullraum). Diese Gerade ginge in sich selbst über, und $\mathbf A$ würde auf ihr wie die Multiplikation mit einem reellen Skalar $\mu$ wirken. Daraus würde $\mathbf A\mathbf w=\mu\mathbf w$ folgen. Damit wäre $\lambda=\mu$ reell, im Widerspruch zu $b\neq0$. Schreiben wir $\mathbf{v}$ und $\mathbf{u}$ als Spalten von $\mathbf{B}=[\mathbf{v}\;\;\mathbf{u}]$, so ergeben die beiden Identitäten $\mathbf{A}\mathbf{B}=\mathbf{B}\left(\begin{smallmatrix}a & -b\\ b & \phantom{-}a\end{smallmatrix}\right)$ oder gleichwertig

<!--en:d6afe2c1-->
In der Basis $(\mathbf v,\mathbf u)$ der invarianten Ebene wirkt $\mathbf{A}$ als Drehstreckung mit dem Block $r\mathbf R_\theta$. Er skaliert um den Betrag $r$ und dreht um das Argument $\theta$. Da wir bei der Herleitung nur $\mathbf A\mathbf w=\lambda\mathbf w$ verwendet haben, hängt dieser Block weder von der Skalierung noch von der Phase von $\mathbf w$ ab. Der Zusammenhang gilt auch umgekehrt. Identifizieren wir $(x,y)\in\mathbb{R}^2$ mit der komplexen Zahl $x+iy$, bildet der Block $(x,y)$ auf $(ax-by,\,bx+ay)$ ab. Das entspricht genau der komplexen Multiplikation mit $a+ib$.

<!--en:292387ad-->
Für unsere Beispielmatrix $\mathbf A$ gilt $\mathbf B=\left(\begin{smallmatrix}-1&1\\0&1\end{smallmatrix}\right)$ und $\mathbf B^{-1}\mathbf A\mathbf B=\left(\begin{smallmatrix}2&-1\\1&2\end{smallmatrix}\right)$. Der Block skaliert um $\sqrt5$ und dreht um $26.6^\circ$. Man könnte erwarten, dass jeder Schritt einen Vektor um den Faktor $\sqrt5\approx2.24$ verlängert. Berechnen wir den Block aus dem Eigenvektor, den `eig` liefert, und verfolgen wir die tatsächlichen Längen der Iterierten $\mathbf A^k(1,0)$.

<!--en:4d4942de-->
Unabhängig davon, welchen Eigenvektor `eig` wählt, ergibt sich der Block $\left(\begin{smallmatrix}2&-1\\1&2\end{smallmatrix}\right)$. Dennoch wachsen die Vektorlängen pro Schritt *nicht* um einen festen Faktor $2.24$. Entlang der Trajektorie $(1,0)\to(1,-1)\to(-1,-4)\to(-9,-11)\to(-31,-24)$ schwankt das Verhältnis aufeinanderfolgender Normen zwischen etwa $1.4$ und $3.5$. Die Skalierung ist nur in den Koordinaten $(\mathbf v,\mathbf u)$ gleichmäßig, und die zugehörige Basis ist nicht orthonormal. In Standardkoordinaten bewegen sich die Iterierten auf elliptisch verzerrten Spiralbahnen nach außen. Etwa alle $2\pi/\theta\approx13.6$ Schritte durchlaufen sie eine volle Umdrehung. Erst das langfristige geometrische Mittel $\|\mathbf A^k\mathbf x\|^{1/k}$ nähert sich $\sqrt5$ (es erreicht $2.25$ bei $k=20$). In Standardkoordinaten ist die Bewegung nur dann eine starre Drehung mit gleichmäßiger Skalierung, wenn die Matrix *normal* ist. Diese Eigenschaft definieren wir in :numref:`subsec_mdl-normal-matrices`.

<!--en:43b636e4-->
Bis auf einen reellen Basiswechsel setzt sich jede reelle quadratische Matrix aus den drei untersuchten Grundformen zusammen, einzeln oder in Kombination: Streckung entlang einer Geraden, Scherung in einem Jordanblock und Drehstreckung in einer Ebene. Diese Zerlegung heißt *reelle jordansche Normalform* :cite:`Horn.Johnson.2012`.

<!--en:0c27975a-->
## Symmetrische Matrizen und quadratische Formen

<!--en:e72e701b-->
### Der Spektralsatz

<!--en:17238fe8-->
Welche Matrizen besitzen garantiert eine reelle Orthonormalbasis aus Eigenvektoren? Symmetrie beseitigt beide Hindernisse, auf die wir gestoßen sind: Sie schließt nichtreelle Eigenwerte ebenso aus wie einen Mangel an Eigenvektoren. Die symmetrische Matrix $\mathbf S$ aus :numref:`sec_linear-algebra` zeigt das Ergebnis. Ihre normierten Eigenvektoren $(1,1)/\sqrt2$ und $(1,-1)/\sqrt2$ stehen senkrecht aufeinander. Damit erhalten wir

<!--en:f42cda1b-->
Da die Eigenvektormatrix $\mathbf V$ orthogonal ist ($\mathbf V^\top\mathbf V=\mathbf I$), ist ihre Inverse gleich ihrer Transponierten. Der Spektralsatz garantiert dies für *jede* reelle symmetrische Matrix. Er ist der zentrale Satz dieses Kapitels. Die positive Definitheit, der Rayleigh-Quotient und der gesamte Abschnitt :numref:`sec_mdl-svd-low-rank` bauen darauf auf.

<!--en:87c242cf-->
**Satz (Spektralsatz).** *Jede reelle symmetrische Matrix $\mathbf{A}$ besitzt reelle Eigenwerte und eine Orthonormalbasis aus reellen Eigenvektoren. Gleichwertig gilt*

<!--en:aa852b09-->
*wobei $\boldsymbol{\Lambda}=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$ reell und $\mathbf{V}$ orthogonal ist.*

<!--en:29312ef2-->
Wir beweisen den Satz in drei Schritten. Erstens ist jeder Eigenwert reell. Zweitens sind Eigenvektoren zu verschiedenen Eigenwerten orthogonal. Drittens existiert stets eine vollständige Orthonormalbasis aus Eigenvektoren. Für komplexe Vektoren verwenden wir das hermitesche innere Produkt $\langle\mathbf x,\mathbf y\rangle=\mathbf x^*\mathbf y$. Dabei bezeichnet $\mathbf x^*$ die konjugierte Transponierte. Für reelle Vektoren ergibt sich das übliche Skalarprodukt. Wir setzen nur voraus, dass das charakteristische Polynom mindestens eine komplexe Nullstelle hat (Fundamentalsatz der Algebra). Damit existiert mindestens ein Eigenpaar über $\mathbb{C}$. Für die Teile (i) und (ii) genügen jeweils zwei Zeilen. Der größte Teil der Arbeit steckt in Teil (iii).

<!--en:2231843f-->
**Beweis, Teil (i): Die Eigenwerte sind reell.** Es gelte $\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ mit $\mathbf{v}\neq\mathbf 0$. Wir lassen zunächst komplexe Einträge zu. Da $\mathbf{A}$ reell und symmetrisch ist, gilt $\mathbf A^*=\mathbf A^\top=\mathbf A$ und damit

<!--en:55fb82c3-->
Wegen $\mathbf{v}^*\mathbf{v}=\|\mathbf v\|^2>0$ können wir beide Seiten durch diesen Ausdruck teilen und erhalten $\lambda=\bar\lambda$. Der Eigenwert ist also reell. Auch seinen Eigenvektor können wir reell wählen, denn er löst das reelle homogene Gleichungssystem $(\mathbf A-\lambda\mathbf I)\mathbf v=\mathbf 0$. $\blacksquare$

<!--en:7a35d955-->
**Beweis, Teil (ii): Eigenvektoren zu verschiedenen Eigenwerten sind orthogonal.** Es gelte $\mathbf{A}\mathbf{u}=\lambda\mathbf{u}$ und $\mathbf{A}\mathbf{v}=\mu\mathbf{v}$ mit $\lambda\neq\mu$. Nach Teil (i) sind beide Eigenwerte reell und wir können beide Eigenvektoren reell wählen. Wegen $\mathbf{A}=\mathbf{A}^\top$ dürfen wir $\mathbf{A}$ im inneren Produkt auf die andere Seite ziehen und erhalten

<!--en:ef744ba9-->
Also gilt $(\lambda-\mu)\langle\mathbf{u},\mathbf{v}\rangle=0$. Wegen $\lambda\neq\mu$ folgt $\langle\mathbf{u},\mathbf{v}\rangle=0$. $\blacksquare$

<!--en:4ca5174c-->
**Beweis, Teil (iii): Eine Orthonormalbasis aus Eigenvektoren.** Wir führen den Beweis durch Induktion über $n$. Für $n=1$ ist die Behauptung unmittelbar klar. Für $n>1$ liefert Teil (i) ein reelles Eigenpaar $(\lambda_1,\mathbf{v}_1)$ mit $\|\mathbf{v}_1\|=1$. Betrachten wir das $(n-1)$-dimensionale orthogonale Komplement $U=\mathbf{v}_1^{\perp}$. Aufgrund der Symmetrie bildet $\mathbf{A}$ den Unterraum $U$ in sich ab. Für jedes $\mathbf{x}\in U$ gilt nämlich

<!--en:d5425b69-->
Wir wählen eine Orthonormalbasis von $U$ und schreiben ihre Vektoren als Spalten von $\mathbf Q\in\mathbb R^{n\times(n-1)}$. Da jede Spalte von $\mathbf A\mathbf Q$ in $U$ liegt, können wir $\mathbf A\mathbf Q=\mathbf Q\mathbf C$ schreiben. Dabei stellt $\mathbf C=\mathbf Q^\top\mathbf A\mathbf Q$ die Einschränkung von $\mathbf A$ auf $U$ dar. Die Matrix $\mathbf C$ ist symmetrisch, denn $\mathbf C^\top=\mathbf Q^\top\mathbf A^\top\mathbf Q=\mathbf C$, und hat die Größe $(n-1)\times(n-1)$. Nach Induktionsvoraussetzung gilt $\mathbf C=\mathbf V'\boldsymbol\Lambda'\mathbf V'^\top$ für eine orthogonale Matrix $\mathbf V'$. Betrachten wir nun die Spalten von $\mathbf Q\mathbf V'$. Wegen $(\mathbf Q\mathbf V')^\top\mathbf Q\mathbf V'=\mathbf V'^\top\mathbf V'=\mathbf I$ sind sie orthonormal. Sie liegen in $U$ und stehen daher senkrecht auf $\mathbf v_1$. Außerdem sind sie Eigenvektoren von $\mathbf A$, denn $\mathbf A\mathbf Q\mathbf V'=\mathbf Q\mathbf C\mathbf V'=\mathbf Q\mathbf V'\boldsymbol\Lambda'$. Fügen wir sie rechts neben $\mathbf v_1$ ein, erhalten wir die orthogonale Matrix $\mathbf V=[\mathbf v_1\;\mathbf Q\mathbf V']$ mit $\mathbf A\mathbf V=\mathbf V\boldsymbol\Lambda$. $\blacksquare$

<!--en:6d6fb659-->
Geometrisch zerlegt :eqref:`eq_mdl-spectral` jede symmetrische lineare Abbildung in drei Schritte. Zunächst richtet $\mathbf V^\top$ die orthonormalen Eigenvektoren an den Koordinatenachsen aus. Das ist eine reine Drehung, gegebenenfalls kombiniert mit einer Spiegelung (für die obige Matrix $\mathbf S$ gilt $\det\mathbf V=-1$). Dann streckt $\boldsymbol\Lambda$ jede Koordinatenachse um ihren Eigenwert. Schließlich dreht $\mathbf V$ die Achsen zurück. Damit ist für jede symmetrische Matrix bewiesen, was das linke Teilbild von :numref:`fig_mdl-la-eig-ellipse` für $\mathbf S$ veranschaulicht hat.

<!--en:ae3d4ae4-->
Jede Matrix $\mathbf A$, ob rechteckig oder quadratisch, liefert eine symmetrische Matrix $\mathbf A^\top\mathbf A$. Ihre quadratische Form $\mathbf x^\top\mathbf A^\top\mathbf A\mathbf x=\|\mathbf A\mathbf x\|^2$ kann nie negativ sein. Im Folgenden benennen, prüfen und nutzen wir diese Eigenschaft.

<!--en:8a59a4a6-->
### Positive Definitheit

<!--en:f45197f6-->
Kann die quadratische Form $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ negative Werte annehmen? Diese Frage spielt eine entscheidende Rolle dabei, ob ein kritischer Punkt einer Verlustlandschaft ein Minimum ist, ob eine Matrix eine gültige Kovarianzmatrix darstellt und ob ein Problem der kleinsten Quadrate eine eindeutige Lösung hat. Eine symmetrische Matrix $\mathbf{A}$ heißt *positiv semidefinit* (engl. positive semidefinite, PSD), geschrieben $\mathbf{A}\succeq0$, wenn

<!--en:3cbccb37-->
und *positiv definit* (engl. positive definite, PD), geschrieben $\mathbf{A}\succ0$, wenn die Ungleichung für jeden von null verschiedenen Vektor $\mathbf{x}$ strikt gilt. Beispielsweise ist $\mathbf x^\top\mathbf S\mathbf x=2x_1^2+2x_1x_2+2x_2^2=x_1^2+x_2^2+(x_1+x_2)^2$ strikt positiv, sobald $\mathbf x\neq\mathbf 0$ gilt. Somit gilt $\mathbf S\succ0$. Dagegen ist $\left(\begin{smallmatrix}1&0\\0&-1\end{smallmatrix}\right)$ weder PD noch PSD, denn $\mathbf e_2$ liefert den Wert $-1$. In höheren Dimensionen wird die quadratische Ergänzung von Hand schnell unpraktisch. Die Eigenwerte entscheiden die Frage für jede symmetrische Matrix.

<!--en:159f020a-->
**Satz (Definitheit anhand der Eigenwerte).** *Für eine symmetrische Matrix $\mathbf{A}$ gilt*

<!--en:0dbf8bc1-->
**Beweis.** Nach dem Spektralsatz gilt $\mathbf{A}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$. Wir wechseln zu den Koordinaten $\mathbf{y}=\mathbf{V}^\top\mathbf{x}$ in der Eigenbasis. Dieser orthogonale Basiswechsel erhält die Längen. Damit folgt

<!--en:fa60b274-->
In der Eigenbasis ist die quadratische Form eine gewichtete Summe von Quadraten mit den Eigenwerten als Gewichten. Gilt für jeden Eigenwert $\lambda_i\ge0$, ist diese Summe nichtnegativ. Umgekehrt erhalten wir durch Einsetzen von $\mathbf{x}=\mathbf{v}_j$ die Beziehungen $\mathbf y=\mathbf e_j$ und $\mathbf{v}_j^\top\mathbf{A}\mathbf{v}_j=\lambda_j$. Daraus folgt $\lambda_j\ge0$. Für positive Definitheit gilt dasselbe Argument mit strikten Ungleichungen, da $\mathbf y\neq\mathbf 0$ gilt, sobald $\mathbf x\neq\mathbf 0$ ist. $\blacksquare$

<!--en:a8965fb8-->
Prüfen wir :eqref:`eq_mdl-quadform` für $\mathbf S$ mit $\mathbf x=\mathbf e_1$. In der Eigenbasis lauten die Koordinaten $\mathbf y=\mathbf V^\top\mathbf e_1=(1/\sqrt2,1/\sqrt2)$. Die gewichtete Summe ergibt $3\cdot\tfrac12+1\cdot\tfrac12=2$, in Übereinstimmung mit $\mathbf e_1^\top\mathbf S\mathbf e_1=s_{11}$. Drei Folgerungen erklären, warum uns positive Definitheit im Deep Learning so häufig begegnet:

<!--en:bfdc9812-->
* **Positiv definite Matrizen sind invertierbar.** Gelten $\mathbf{A}\succ0$ und $\mathbf{A}\mathbf{x}=\mathbf 0$, so folgt $\mathbf{x}^\top\mathbf{A}\mathbf{x}=0$ und damit zwingend $\mathbf{x}=\mathbf 0$.

<!--en:68167778-->
* **Gram-Matrizen und Kovarianzmatrizen sind PSD.** Für jede Matrix $\mathbf{X}$ gilt $\mathbf{x}^\top\mathbf{X}^\top\mathbf{X}\mathbf{x}=\|\mathbf{X}\mathbf{x}\|^2\ge0$ und somit $\mathbf X^\top\mathbf X\succeq0$. Die Gram-Matrix ist genau dann strikt positiv definit, wenn $\mathbf{X}$ vollen Spaltenrang besitzt (sodass aus $\mathbf{X}\mathbf{x}=\mathbf 0$ stets $\mathbf{x}=\mathbf 0$ folgt). Die Stichprobenkovarianz (engl. sample covariance) zentrierter Daten ist eine Gram-Matrix geteilt durch die Anzahl der Samples und damit stets PSD.

<!--en:0919117f-->
* **Eine positiv definite Hesse-Matrix garantiert ein striktes lokales Minimum.** Sei $f$ zweimal stetig differenzierbar mit $\nabla f(\mathbf x^*)=\mathbf 0$. Ihre Hesse-Matrix $\nabla^2 f(\mathbf x^*)$ (die symmetrische Matrix der zweiten partiellen Ableitungen) sei positiv definit. Für jeden kleinen Schritt $\boldsymbol\delta$ lautet die Taylor-Entwicklung zweiter Ordnung (:numref:`subsec_mdl-second-order-taylor`) dann $f(\mathbf x^*+\boldsymbol\delta)=f(\mathbf x^*)+\tfrac12\boldsymbol\delta^\top\nabla^2 f(\mathbf x^*)\boldsymbol\delta+o(\|\boldsymbol\delta\|^2)$. Nach :eqref:`eq_mdl-quadform` beträgt der quadratische Term für jeden von null verschiedenen Schritt mindestens $\tfrac12\lambda_{\min}\|\boldsymbol\delta\|^2>0$. Damit ist $\mathbf x^*$ ein striktes lokales Minimum. Diese Bedingung ist hinreichend, aber nicht notwendig. Beispielsweise besitzt $f(x)=x^4$ bei $0$ ein striktes Minimum, obwohl die zweite Ableitung dort $0$ ist.

<!--en:c565d01d-->
Positive Definitheit lässt sich auch mit einem schnellen Algorithmus prüfen, der *Cholesky-Zerlegung*. Es gilt genau dann $\mathbf{A}\succ0$, wenn $\mathbf{A}=\mathbf{L}\mathbf{L}^\top$ für eine untere Dreiecksmatrix $\mathbf{L}$ mit positiven Diagonaleinträgen gilt. Eine Richtung folgt sofort: Ein solches $\mathbf{L}$ ist invertierbar, also gilt $\mathbf{x}^\top\mathbf{A}\mathbf{x}=\|\mathbf{L}^\top\mathbf{x}\|^2>0$ für jedes von null verschiedene $\mathbf{x}$. Der Versuch, eine Cholesky-Zerlegung zu berechnen, ist das gängige numerische Verfahren zum Prüfen positiver Definitheit und deutlich günstiger als die Berechnung aller Eigenwerte. Derselbe Faktor $\mathbf{L}$ überführt zudem unabhängige standardnormalverteilte Samples $\mathbf z$ in Samples $\mathbf L\mathbf z$ mit Kovarianz $\mathbf L\mathbf L^\top=\mathbf A$.

<!--en:11c70960-->
Geometrisch bestimmen die Vorzeichen der Eigenwerte die Form der Fläche $z=\mathbf{x}^\top\mathbf{A}\mathbf{x}$ (:numref:`fig_mdl-la-psd`). Sind alle Eigenwerte positiv, entsteht eine Schüssel. Bei nichtnegativen Eigenwerten mit Nulleigenwerten entsteht eine Rinne, die in jeder Richtung zum Eigenwert null flach ist. Bei gemischten Vorzeichen entsteht ein Sattel.

<!--en:0c0093ad-->
![Die Fläche $z=\mathbf{x}^\top\mathbf{A}\mathbf{x}$ für drei Diagonalmatrizen. Die Vorzeichen der Eigenwerte bestimmen die Form: eine Schüssel, wenn alle $\lambda_i>0$ sind (positiv definit), eine Rinne, wenn ein Eigenwert $\lambda_i=0$ erfüllt (positiv semidefinit), und ein Sattel bei gemischten Vorzeichen (indefinit).](../img/mdl-la-psd.svg)

<!--en:25f08722-->
Die Rangbedingung für Gram-Matrizen können wir direkt an ihren Eigenwerten ablesen. Im Folgenden betrachten wir eine Matrix $\mathbf{X}$ mit mehr Zeilen als Spalten und linear unabhängigen Spalten. Für sie sollte $\mathbf X^\top\mathbf X$ positiv definit sein. Zum Vergleich betrachten wir eine Matrix, deren zweite Spalte das Doppelte der ersten ist. Sie sollte eine positiv semidefinite Gram-Matrix mit einem Eigenwert null liefern.

<!--en:c1ef47ea-->
Bei vollem Rang sind beide Eigenwerte strikt positiv. Bei linear abhängigen Spalten fällt einer auf null. Die quadratische Form verschwindet entlang $(2,-1)$, genau der Richtung, die $\mathbf X$ auf null abbildet.

<!--en:9b214841-->
### Der Rayleigh-Quotient

<!--en:9880eb1d-->
Mit der Identität :eqref:`eq_mdl-quadform` können wir mehr als Vorzeichen prüfen. Sie charakterisiert den größten und den kleinsten Eigenwert als Lösungen von Optimierungsproblemen. Für eine symmetrische Matrix $\mathbf{A}$ definieren wir den *Rayleigh-Quotienten* durch

<!--en:e0393cc2-->
Er misst die quadratische Form bezogen auf das Quadrat der Länge. Ist $\mathbf{x}$ ein Eigenvektor, stimmt $R(\mathbf{x})$ mit seinem Eigenwert überein. In diesem Unterabschnitt ordnen wir die Eigenwerte nach ihrem vorzeichenbehafteten Wert, $\lambda_1\ge\cdots\ge\lambda_n$, und nicht nach ihrem Betrag.

<!--en:b3fc5281-->
**Satz (Rayleigh).** *Für eine symmetrische Matrix $\mathbf{A}$ mit Eigenwerten $\lambda_1\ge\cdots\ge\lambda_n$ gilt*

<!--en:5bc1af9b-->
*wobei der Quotient an den zugehörigen Eigenvektoren sein Maximum beziehungsweise sein Minimum erreicht.*

<!--en:0b420877-->
**Beweis.** Skalieren wir $\mathbf{x}$, bleibt $R(\mathbf{x})$ unverändert. Wir können uns daher auf Einheitsvektoren beschränken. Der Basiswechsel $\mathbf{y}=\mathbf{V}^\top\mathbf{x}$ erhält die Norm ($\|\mathbf y\|=\|\mathbf x\|=1$) und überführt :eqref:`eq_mdl-quadform` in $R(\mathbf x)=\sum_i\lambda_i y_i^2$. Dies ist ein gewichteter Mittelwert der Eigenwerte mit nichtnegativen Gewichten $y_i^2$, die sich zu $1$ summieren. Jeder solche Mittelwert liegt zwischen dem kleinsten und dem größten Wert, also gilt $\lambda_n\le R(\mathbf x)\le\lambda_1$. Konzentrieren wir das gesamte Gewicht auf eine Koordinate, bei $\mathbf x=\mathbf v_1$ oder $\mathbf x=\mathbf v_n$, erreichen wir die beiden Schranken. $\blacksquare$

<!--en:34f6061a-->
Für unsere symmetrische Beispielmatrix $\mathbf S$ durchläuft $R(\mathbf x)$ also das Intervall $[1,3]$. Das Maximum liegt entlang des Eigenvektors bei $45^\circ$, das Minimum entlang des Eigenvektors bei $135^\circ$. In jeder Richtung dazwischen mischen sich die Eigenwerte $1$ und $3$ im Verhältnis der quadrierten Koordinaten in der Eigenbasis.

<!--en:9fcb735f-->
Durch Maximieren über Unterräume erhalten wir auch alle dazwischenliegenden Eigenwerte. Der *Satz von Courant und Fischer* besagt, dass $\lambda_j=\max_{\dim U=j}\ \min_{\mathbf 0\neq\mathbf x\in U}R(\mathbf x)$ gilt. Dies ist der höchste garantierte Rayleigh-Quotient unter allen $j$-dimensionalen Unterräumen $U$ (Übung :exref:`ex_mdl-eigendecomposition-computed-eigenvalue-accuracy`d führt durch den Beweis).

<!--en:77548198-->
Zwei Anwendungen der extremalen Eigenwerte begegnen uns in diesem Buch immer wieder. Erstens ist für jede Matrix $\mathbf A$ der größte Rayleigh-Quotient von $\mathbf A^\top\mathbf A$ gleich $\max_{\mathbf x\neq\mathbf 0}\|\mathbf A\mathbf x\|^2/\|\mathbf x\|^2$. Das ist das Quadrat des größten Faktors, um den $\mathbf A$ einen Vektor strecken kann. Zweitens betrachten wir den Gradientenabstieg für die konvexe quadratische Funktion $f(\mathbf x)=\tfrac12\mathbf x^\top\mathbf A\mathbf x$ mit $\mathbf A\succ0$ und der Aktualisierung $\mathbf x\leftarrow(\mathbf I-\eta\mathbf A)\mathbf x$. Nach :eqref:`eq_mdl-eig-power-expansion` multipliziert sich jede Koordinate in der Eigenbasis bei jedem Schritt mit $1-\eta\lambda_i$. Die Iteration konvergiert also von jedem Startpunkt aus genau dann zum Minimum, wenn $|1-\eta\lambda_i|<1$ für alle $i$ gilt, das heißt $0<\eta<2/\lambda_{\max}$. Der größte Eigenwert begrenzt somit die Schrittweite (er ist die Glattheitskonstante $L$ aus :numref:`sec_mdl-gradient-based-optimization`). Bei Schrittweiten $\eta\le1/\lambda_{\max}$ bestimmt der kleinste Eigenwert den langsamsten Kontraktionsfaktor $1-\eta\lambda_{\min}$. Ihr Verhältnis $\lambda_{\max}/\lambda_{\min}$ ist die Konditionszahl von $\mathbf A$. In :numref:`subsec_mdl-condition-number` verallgemeinern wir sie auf beliebige Matrizen.

<!--en:5f778ebe-->
### Stichprobenkovarianz und Marchenko-Pastur-Gesetz

<!--en:1a59b3bd-->
Wenn wir in hochdimensionalen Daten nach Struktur suchen, betrachten wir die größten Eigenwerte der Stichprobenkovarianzmatrix. Um zu beurteilen, welche Eigenwerte wirklich groß sind, brauchen wir einen Vergleichsmaßstab. Wie sehen die Eigenwerte aus, wenn die Daten keinerlei Struktur enthalten? Wir ziehen $n$ Samples aus reinem Rauschen in $d$ Dimensionen. Ihre Koordinaten seien unabhängig mit Mittelwert $0$ und Varianz $1$ (die wahre Kovarianzmatrix ist also $\mathbf I$). Wir schreiben die Samples als Zeilen von $\mathbf X\in\mathbb R^{n\times d}$. Häufen sich die Eigenwerte der Stichprobenkovarianz $\mathbf C=\tfrac1n\mathbf X^\top\mathbf X$ in der Nähe von $1$? Für jeden Einheitsvektor $\mathbf w$ ist der Rayleigh-Quotient $\mathbf w^\top\mathbf C\mathbf w$ die Stichprobenvarianz der auf $\mathbf w$ projizierten Daten. Die extremalen Eigenwerte von $\mathbf C$ sind somit die Stichprobenvarianzen entlang der Richtungen mit der größten und der kleinsten Varianz. Genau diese Größen deutet die Hauptkomponentenanalyse als Struktur (:numref:`subsec_mdl-pca`).

<!--en:7d4bb3e7-->
Überraschenderweise häufen sie sich *nicht* bei $1$. Es gelte $d,n\to\infty$ bei festem *Dimensionsverhältnis* $c=d/n\le1$. Dann konvergiert das Histogramm der Eigenwerte von $\mathbf C$ gegen eine deterministische Dichte mit Träger auf dem Intervall

<!--en:76069afe-->
das als *Marchenko-Pastur-Intervall* (engl. Marchenko–Pastur bulk) bekannt ist :cite:`Marchenko.Pastur.1967`. Hat das Rauschen ein endliches viertes Moment, konvergieren die extremalen Eigenwerte gegen die Intervallgrenzen. Für $c>1$ hat die Matrix $\mathbf C$ höchstens den Rang $n<d$. Dann sind neben den Eigenwerten im Intervall mindestens $d-n$ Eigenwerte exakt null. Selbst bei vier Samples pro Dimension ($c=1/4$) verteilen sich die Eigenwerte reinen Rauschens über das gesamte Intervall $[0.25,2.25]$. In der zufälligen Richtung mit der größten Varianz beträgt die Stichprobenvarianz mehr als das Doppelte der wahren Varianz, in der Richtung mit der kleinsten nur ein Viertel. Schon für $n=4000$ und $d=1000$ ist dieser asymptotische Grenzwert bemerkenswert genau. Prüfen wir das.

<!--en:96dd23cd-->
Bei jedem Durchlauf weichen der kleinste und der größte Eigenwert nur um wenige Hundertstel von den vorhergesagten Grenzen $0.25$ und $2.25$ ab. Ein Eigenwert innerhalb dieses Intervalls lässt sich nicht von reinem Rauschen unterscheiden. Erst ein Eigenwert deutlich oberhalb der Obergrenze liefert einen Hinweis auf tatsächliche Struktur. Übung :exref:`ex_mdl-eigendecomposition-spiked-covariance` untersucht, wie stark ein Signal sein muss, damit sein Eigenwert das Intervall verlässt.

<!--en:293d28b6-->
In diesem Abschnitt haben wir zweierlei vorausgesetzt: dass wir die Eigenwerte einer Matrix tatsächlich berechnen können und dass uns diese Eigenwerte über die Wirkung weniger Iterationsschritte ebenso viel verraten wie über die Wirkung vieler Schritte. Bei großen Matrizen und bei Matrizen, die stark von der Symmetrie abweichen, können wir uns auf keine der beiden Annahmen verlassen. :numref:`sec_mdl-spectra-in-practice` behandelt beide Fragen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:fbf997b0-->
In den Koordinaten ihrer Eigenbasis wirkt eine diagonalisierbare Matrix durch Multiplikation mit einzelnen Skalaren: $\mathbf A^k\mathbf x=\sum_i c_i\lambda_i^k\mathbf v_i$. Das erklärt die wiederholte Anwendung einer Matrix in Markov-Ketten, in tiefen linearen Netzen und beim Gradientenabstieg für quadratische Funktionen. Eine vollständige Eigenbasis über $\mathbb C$ existiert genau dann, wenn die geometrische Vielfachheit jedes Eigenwerts mit seiner algebraischen Vielfachheit übereinstimmt. Andernfalls kommen durch Jordanblöcke polynomiale Faktoren in $k$ hinzu, die transientes Wachstum verursachen können. Komplex konjugierte Eigenwertpaare beschreiben eine Drehstreckung in einer invarianten Ebene. Bei reellen symmetrischen Matrizen treten beide Schwierigkeiten nicht auf. Sie besitzen stets reelle Eigenwerte und eine Orthonormalbasis aus Eigenvektoren. Ihre quadratischen Formen werden zu gewichteten Summen von Quadraten. Deshalb entscheiden die Vorzeichen der Eigenwerte über die positive Definitheit, und die extremalen Eigenwerte sind die Extremwerte des Rayleigh-Quotienten. Die Eigenwerte der Stichprobenkovarianz von reinem Rauschen verteilen sich über das Marchenko-Pastur-Intervall. Erst Eigenwerte oberhalb der oberen Intervallgrenze deuten auf tatsächliche Struktur hin.

<!--en:2294243a-->
## Übungen

<!--en:bd59a684-->
1. **Spur, Determinante und Definitheit.** Die Identitäten :eqref:`eq_mdl-det-eig` und :eqref:`eq_mdl-tr-eig` legen die beiden Eigenwerte einer $2\times2$-Matrix über ihre Summe und ihr Produkt fest.
    1. Bestimmen Sie die Eigenwerte von $\left(\begin{smallmatrix}4&1\\1&4\end{smallmatrix}\right)$ allein aus Spur und Determinante, ohne das charakteristische Polynom aufzustellen. Bestimmen Sie anschließend die Eigenvektoren. Gehen Sie für die Übergangsmatrix $\mathbf P$ der Nutzerin ebenso vor. Ihre Spur beträgt $5/4$, ihre Determinante $1/4$.
    1. Zeigen Sie, dass eine symmetrische $2\times2$-Matrix genau dann positiv definit ist, wenn Spur und Determinante beide positiv sind. Welche Vorzeichenbedingungen für Spur und Determinante charakterisieren positive Semidefinitheit?
    1. Jede der Matrizen $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$, $\left(\begin{smallmatrix}1&2\\2&1\end{smallmatrix}\right)$ und $\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$ ist die Hesse-Matrix der jeweiligen quadratischen Funktion $f(\mathbf{x})=\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ an ihrem kritischen Punkt $\mathbf{0}$. Entscheiden Sie anhand von (b) für jedes $f$, ob der Ursprung ein striktes Minimum, ein nichtstriktes Minimum oder ein Sattelpunkt ist. Geben Sie für den Sattelpunkt eine Richtung an, entlang derer $f$ abnimmt.
    1. Zeigen Sie, dass der Cholesky-Faktor $\mathbf{L}$ von $\left(\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\right)$ die Diagonaleinträge $\sqrt a$ und $\sqrt{(ac-b^2)/a}$ hat. Warum stimmen die Existenzbedingungen für einen Faktor mit positiver Diagonale mit dem Kriterium aus (b) überein?
   :exlabel:`ex_mdl-eigendecomposition-trace-determinant`

<!--en:3db8ce94-->
1. **Rang und Eigenwerte null.** Wie dieser Abschnitt zeigt, entspricht der Rang einer diagonalisierbaren Matrix der Anzahl ihrer von null verschiedenen Eigenwerte, jeweils mit Vielfachheit gezählt.
    1. Bestimmen Sie den Rang und die Eigenwerte von $\mathbf{N}=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$. Welcher Schritt im Argument für diagonalisierbare Matrizen scheitert bei $\mathbf{N}$?
    1. Sei $g_0$ die geometrische Vielfachheit des Eigenwerts $0$ einer $n\times n$-Matrix $\mathbf{A}$. Setzen Sie $g_0=0$, falls $0$ kein Eigenwert ist. Zeigen Sie, dass $\operatorname{rank}\mathbf{A}=n-g_0$ für jede quadratische Matrix gilt. Folgern Sie, dass der Rang mindestens so groß ist wie die Anzahl der mit algebraischer Vielfachheit gezählten Eigenwerte ungleich null und dass Gleichheit genau dann gilt, wenn der Eigenwert $0$ nicht defekt ist oder gar nicht auftritt.
    1. Die *Singulärwerte* einer Matrix sind $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$, also die Quadratwurzeln der Eigenwerte der positiv semidefiniten Matrix $\mathbf A^\top\mathbf A$. Bestimmen Sie die Singulärwerte von $\mathbf{N}$ und des Jordanblocks $\mathbf{J}_m(0)$ aus :numref:`subsec_mdl-jordan`. Vergleichen Sie die Anzahl der von null verschiedenen Singulärwerte mit dem Rang und mit der Anzahl der von null verschiedenen Eigenwerte.
   :exlabel:`ex_mdl-eigendecomposition-rank-zero-eigenvalues`

<!--en:5aa1e4c1-->
1. **Die Zentrierungsmatrix.** Sei $\mathbf{1}\in\mathbb{R}^n$ der Vektor, dessen Einträge alle eins sind, und sei $\mathbf{H}=\mathbf{I}-\tfrac1n\mathbf{1}\mathbf{1}^\top$.
    1. Zeigen Sie, dass $n\mathbf{I}-\mathbf{1}\mathbf{1}^\top$ positiv semidefinit ist, indem Sie $\mathbf{x}^\top(n\mathbf{I}-\mathbf{1}\mathbf{1}^\top)\mathbf{x}$ als Summe von Quadraten der Differenzen $x_i-x_j$ schreiben. Bestimmen Sie anschließend alle Eigenwerte von $n\mathbf{I}-\mathbf{1}\mathbf{1}^\top$ und die zugehörigen Eigenvektoren.
    1. Zeigen Sie, dass $\mathbf{H}$ symmetrisch ist und $\mathbf{H}^2=\mathbf{H}$ erfüllt. Leiten Sie aus (a) die Eigenwerte und den Rang ab.
    1. Die Zeilen von $\mathbf{X}\in\mathbb{R}^{n\times d}$ seien $n$ Samples. Ziehen wir den Stichprobenmittelwert von jeder Zeile ab, erhalten wir $\mathbf{H}\mathbf{X}$. Damit ist die Stichprobenkovarianz $\mathbf{C}=\tfrac1n\mathbf{X}^\top\mathbf{H}\mathbf{X}$. Zeigen Sie, dass $\mathbf{C}\succeq0$ und $\operatorname{rank}\mathbf{C}\le n-1$ gelten. Wie viele Eigenwerte null besitzt die Stichprobenkovarianz von $512$-dimensionalen Einbettungen mindestens, wenn Sie sie aus einem Minibatch von $256$ Beispielen berechnen? Das entspricht dem Fall $c=d/n>1$ in :numref:`subsec_mdl-sample-covariance-mp`. Welche Operationen mit $\mathbf{C}$ sind damit ausgeschlossen?

   *Nach Gilbert Strang, [Übungsblätter zum MIT-Kurs 18.065](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/cd4c57b7e61b8ca9fdd3511a30aca052_MIT18_065S18PSets.pdf), Aufgaben zu Vorlesung 13, Aufgabe 4.*
   :exlabel:`ex_mdl-eigendecomposition-centering-matrix`

<!--en:be95b9c1-->
1. [code] **Genauigkeit berechneter Eigenwerte.** ● Der Algorithmus hinter `np.linalg.eigvals` ist rückwärtsstabil. Er liefert die exakten Eigenwerte einer Matrix $\mathbf{A}+\mathbf{E}$, wobei $\|\mathbf{E}\|_2$ in der Größenordnung von $\epsilon_{\textrm{mach}}\|\mathbf{A}\|_2$ liegt. Bei doppelter Genauigkeit gilt $\epsilon_{\textrm{mach}}\approx2.2\times10^{-16}$ :cite:`Golub.Van-Loan.1996`. Eine kleine Störung kann einen defekten Eigenwert um eine gebrochene Potenz von $\varepsilon$ verschieben.
    1. Seien $\mathbf{e}_1,\ldots,\mathbf{e}_m$ die Standardbasisvektoren. Zeigen Sie, dass $\mathbf{J}_m(\lambda)+\varepsilon\,\mathbf{e}_m\mathbf{e}_1^\top$, also der Jordanblock mit dem Eintrag $\varepsilon$ links unten, die Eigenwerte $\lambda+\varepsilon^{1/m}\omega$ besitzt. Dabei durchläuft $\omega$ die $m$-ten Einheitswurzeln.
    1. Bilden Sie für $m=1,\ldots,4$ jeweils $\mathbf{Q}\mathbf{J}_m(2)\mathbf{Q}^\top$. Verwenden Sie dazu eine orthogonale Matrix $\mathbf{Q}$ aus der QR-Zerlegung einer gaußschen Zufallsmatrix. Bestimmen Sie für $20$ Ziehungen den Median von $\max_i|\hat\lambda_i-2|$ für die berechneten Eigenwerte $\hat\lambda_i$. Vergleichen Sie ihn mit $\epsilon_{\textrm{mach}}^{1/m}$. Wiederholen Sie den Versuch mit der symmetrischen Matrix $\mathbf{Q}(2\mathbf{I})\mathbf{Q}^\top$.
    1. Seien $\mathbf{A}$ und $\mathbf{E}$ symmetrisch. Zeigen Sie mit dem Satz von Rayleigh aus :numref:`subsec_mdl-rayleigh`, dass $\lambda_{\max}(\mathbf{A}+\mathbf{E})\le\lambda_{\max}(\mathbf{A})+\|\mathbf{E}\|_2$ und $\lambda_{\min}(\mathbf{A}+\mathbf{E})\ge\lambda_{\min}(\mathbf{A})-\|\mathbf{E}\|_2$ gelten. Wie erklärt das das Ergebnis aus (b) im symmetrischen Fall?
    1. Beweisen Sie den Satz von Courant und Fischer, $\lambda_j=\max_{\dim U=j}\ \min_{\mathbf 0\neq\mathbf x\in U}R(\mathbf x)$, für symmetrisches $\mathbf A$ mit $\lambda_1\ge\cdots\ge\lambda_n$. Schneiden Sie für die obere Schranke den Unterraum $U$ mit $\operatorname{span}(\mathbf v_j,\ldots,\mathbf v_n)$. Letzterer hat die Dimension $n-j+1$, sodass der Schnitt mindestens eine Gerade enthält. Wählen Sie für die untere Schranke $U=\operatorname{span}(\mathbf v_1,\ldots,\mathbf v_j)$. Leiten Sie daraus die Weyl-Ungleichung $|\lambda_j(\mathbf A+\mathbf E)-\lambda_j(\mathbf A)|\le\|\mathbf E\|_2$ für jedes $j$ ab. Damit erweitern Sie (c) auf alle Eigenwerte.
   :exlabel:`ex_mdl-eigendecomposition-computed-eigenvalue-accuracy`

<!--en:cb98fd12-->
1. [code] **Ausreißer oberhalb des Marchenko-Pastur-Intervalls.** ● In :numref:`subsec_mdl-sample-covariance-mp` beschreibt das Marchenko-Pastur-Intervall den Bereich der Eigenwerte bei reinem Rauschen. Nehmen Sie an, die Daten enthalten eine Richtung mit echtem Signal. Die $n$ Samples besitzen unabhängige normalverteilte Koordinaten mit Varianz $1$, außer der ersten Koordinate, deren Varianz $1+\beta$ beträgt. Für $n\to\infty$ bei festem $c=d/n$ zeigen :citet:`Baik.Silverstein.2006`, dass der größte Eigenwert von $\mathbf{C}=\tfrac1n\mathbf{X}^\top\mathbf{X}$ gegen folgenden Wert konvergiert:

   $$
   \lambda_{\max}\to\begin{cases}(1+\sqrt c)^2, & \beta\le\sqrt c,\\
   (1+\beta)\left(1+c/\beta\right), & \beta>\sqrt c.\end{cases}
   $$

    1. Berechnen Sie für $n=2000$ und $d=500$ den größten Eigenwert von $\mathbf{C}$ für $\beta\in\{0,0.3,0.5,0.7,1,2\}$. Mitteln Sie über fünf Ziehungen und vergleichen Sie mit der Vorhersage.
    1. Halten Sie $d=500$ und $\beta=0.4$ fest. Bestimmen Sie mit der Formel den Stichprobenumfang $n$, oberhalb dessen der größte Eigenwert das Intervall verlässt. Erhöhen Sie dann die Anzahl der Samples gemäß $n\in\{500,1000,2000,4000,8000\}$. Vergleichen Sie den größten Eigenwert mit der oberen Intervallgrenze, mit der Vorhersage und mit dem größten Eigenwert für $\beta=0$ bei gleichem $n$. Warum ist der Abstand bei $n=4000$ schwer zu erkennen?
    1. In einer Anwendung bleiben nur die Hauptkomponenten erhalten, deren Eigenwerte die obere Intervallgrenze überschreiten. Wie groß muss die Varianz einer Richtung bei $c=1/4$ sein, damit sie erhalten bleibt? Wie viele Samples pro Dimension sind nötig, um eine Richtung zu erkennen, deren Varianz die der anderen um $10\%$ übersteigt?
   :exlabel:`ex_mdl-eigendecomposition-spiked-covariance`

<!--en:74216e99-->
[Diskussionen](https://d2l.discourse.group/t/411)

<!--en:91071da4-->
[Diskussionen](https://d2l.discourse.group/t/1086)

<!--en:d25e7816-->
[Diskussionen](https://d2l.discourse.group/t/1087)

<!--en:d25e7816-2-->
[Diskussionen](https://d2l.discourse.group/t/1087)

<!--en:0ad021f7-->
Eigenwertzerlegungen<br>**Eigenvektoren · Diagonalisierung und Matrixpotenzen · defekte und komplexe Spektren · Spektralsatz · quadratische Formen**

<!--en:07e092c7-->
::: {.slide title="Wiederholte Matrixanwendung: Wohin führt P⁵⁰?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:51515f10-->
- **Wiederholte lineare Abbildungen im maschinellen Lernen:** Tiefe lineare Netze wenden $\mathbf{W}^{50}\mathbf{x}$ an. Der Gradientenabstieg für quadratische Funktionen aktualisiert Fehler mit $(\mathbf{I} - \eta\mathbf{H})^k$. Markov-Ketten iterieren Übergangsmatrizen $\mathbf{p}_{k+1} = \mathbf{P}\mathbf{p}_k$.

<!--en:b931c8b8-->
- **Zufällige Klicks auf zwei Seiten:** Von Seite 1 geht es mit Wahrscheinlichkeit $1/2$ auf Seite 1 bzw. 2. Von Seite 2 geht es mit Wahrscheinlichkeit $1/4$ zurück auf Seite 1, mit $3/4$ bleibt man auf Seite 2:
  $$\mathbf{P} = \begin{bmatrix}1/2 & 1/4 \\ 1/2 & 3/4\end{bmatrix}$$

<!--en:a5bc4ceb-->
- **Schnelle Konvergenz:** Vom Startpunkt $\mathbf{p}_0 = (1, 0)$ aus folgen die Iterierten $(0.5, 0.5)$ $\to$ $(0.375, 0.625)$ $\to$ $(0.344, 0.656)$. Eigenvektoren erklären Grenzwert und genaue Konvergenzrate.

<!--en:d1b9821f-->
::: {.slide title="Eigenvektoren: Richtungen, in denen eine Matrix nur streckt"}

<!--en:55623f5d-->
[Grunddefinition]{.kicker}

<!--en:16d1c86a-->
- **Eigenpaargleichung ($\mathbf{v} \neq \mathbf{0}$):** Ein von null verschiedener Vektor $\mathbf{v}$ ist ein *Eigenvektor* von $\mathbf{A}$ zum *Eigenwert* $\lambda$, wenn $\mathbf{A}$ auf $\mathbf{v}$ wie eine skalare Multiplikation wirkt:
  $$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

<!--en:5700a82a-->
- **Eigenpaare der Übergangsmatrix:** $\mathbf{P}(1, 2) = 1\cdot(1, 2)$ (stationäre Richtung, $\lambda_1 = 1$) und $\mathbf{P}(1, -1) = \tfrac{1}{4}(1, -1)$ (Richtung des abklingenden Fehlers, $\lambda_2 = 1/4$).

<!--en:66b1f7ef-->
- **Eigenrichtungen und Achsen maximaler Streckung:** Eigenvektoren kennzeichnen Richtungen, die sich *nicht drehen*. Bei symmetrischen Matrizen liegen sie auf den Ellipsenachsen des abgebildeten Einheitskreises, bei allgemeinen Matrizen muss das nicht gelten.

<!--en:fc67d781-->
::: {.slide title="Eigenwerte mit det(A − λI) = 0 bestimmen"}

<!--en:a7d96a6e-->
[Charakteristisches Polynom]{.kicker}

<!--en:9220199f-->
- **Singularitätsbedingung:** Die Umformung von $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ zu $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$ zeigt: Ein von null verschiedener Eigenvektor $\mathbf{v}$ existiert genau dann, wenn $\mathbf{A} - \lambda\mathbf{I}$ singulär ist:
  $$p(\lambda) = \det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

<!--en:9ed3f91b-->
- **Beispiel der $2 \times 2$-Übergangsmatrix:** $\det(\mathbf{P} - \lambda\mathbf{I}) = (\tfrac{1}{2}-\lambda)(\tfrac{3}{4}-\lambda) - \tfrac{1}{8} = \lambda^2 - \tfrac{5}{4}\lambda + \tfrac{1}{4} = (\lambda - 1)(\lambda - \tfrac{1}{4})$ mit den Nullstellen $\lambda_1 = 1$ und $\lambda_2 = 1/4$.

<!--en:7a873051-->
- **Eigenräume bestimmen:** Einsetzen von $\lambda_1 = 1$ in $(\mathbf{P} - \mathbf{I})\mathbf{v} = \mathbf{0}$ ergibt $-\tfrac{1}{2}v_1 + \tfrac{1}{4}v_2 = 0 \implies \mathbf{v}_1 = (1, 2)$. Für $\lambda_2 = 1/4$ folgt $\tfrac{1}{4}v_1 + \tfrac{1}{4}v_2 = 0 \implies \mathbf{v}_2 = (1, -1)$.

<!--en:5c8470bc-->
::: {.slide title="Wie numerische Bibliotheken Eigenwerte berechnen"}

<!--en:6b28b2b8-->
[Numerische Eigenwertverfahren]{.kicker}

<!--en:dc9d14ac-->
- **$p(\lambda) = 0$ nie direkt lösen:** Für $n \ge 5$ gibt es keine allgemeine Formel mit Wurzelausdrücken. Nullstellen reagieren extrem empfindlich auf Rundungsfehler in den Koeffizienten.

<!--en:827cdce7-->
- **Iterative orthogonale Ähnlichkeitstransformation (QR/Schur):** Die Iteration $\mathbf{A}_{k+1} = \mathbf{Q}_k^\top\mathbf{A}_k\mathbf{Q}_k$ lässt Einträge unter der Diagonale gegen null gehen und erhält die Eigenwerte.

<!--en:b023b24c-->
- **Eigenwerte ablesen:** In oberer Dreiecksform zeigt $\det(\mathbf{U} - \lambda\mathbf{I}) = \prod_i (u_{ii} - \lambda)$, dass alle Eigenwerte auf der Diagonale stehen.

<!--en:91f67ff8-->
::: {.slide title="Diagonalisierung führt Matrixpotenzen auf skalare Potenzen zurück"}

<!--en:68039e91-->
[Diagonalisierung]{.kicker}

<!--en:75bae96a-->
- **Zerlegung in der Eigenbasis ($\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$):** Mit $n$ linear unabhängigen Eigenvektoren als Spalten von $\mathbf{V}$ gilt $\mathbf{A}\mathbf{V} = \mathbf{V}\boldsymbol{\Lambda}$.

<!--en:b4aabba2-->
- **Kürzen im Produkt:** Die inneren Faktoren $\mathbf{V}^{-1}\mathbf{V} = \mathbf{I}$ kürzen sich in $\mathbf{A}^k = \mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1}$. Damit folgt $\mathbf{A}^k\mathbf{x} = \sum_{i=1}^n c_i\lambda_i^k\mathbf{v}_i$ ($\mathbf{c} = \mathbf{V}^{-1}\mathbf{x}$).

<!--en:c67927b8-->
- **Geschlossene Formel für die Klickverteilung:** $\mathbf{p}_k = \tfrac{1}{3}(1, 2) + \tfrac{2}{3}(\tfrac{1}{4})^k(1, -1) \to (1/3, 2/3)$, da sich der Fehler mit jedem Klick durch $4$ teilt.

<!--en:dd43fd2c-->
::: {.slide title="Die geschlossene Formel in der Eigenbasis überprüfen"}

<!--en:5efedb90-->
[Diagonalisierung im Code]{.kicker}

<!--en:4b09123a-->
Fünfzig Matrizenmultiplikationen $\mathbf{P}^{50}\mathbf{p}_0$ und die Formel mit skalaren Potenzen $\sum_i c_i \lambda_i^{50}\mathbf{v}_i$ stimmen bis auf Maschinengenauigkeit überein. Der Fehler teilt sich in jedem Schritt durch $4$:

<!--en:e940118e-->
**Allgemeine Matrixfunktionen:** Dasselbe Verfahren in der Eigenbasis liefert inverse Matrizen $\mathbf{A}^{-1} = \mathbf{V}\boldsymbol{\Lambda}^{-1}\mathbf{V}^{-1}$ (wenn alle $\lambda_i \neq 0$ gelten) und Matrixexponentiale $e^{\mathbf{A}t} = \mathbf{V}e^{\boldsymbol{\Lambda}t}\mathbf{V}^{-1}$ für Differentialgleichungen in kontinuierlicher Zeit $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$.

<!--en:e9430aeb-->
::: {.slide title="Ähnlichkeitsinvarianten: Spur und Determinante"}

<!--en:4aaadf4b-->
[Ähnlichkeit und Invarianten]{.kicker}

<!--en:df7b6c5b-->
- **Unabhängigkeit von der Basis:** Ähnliche Matrizen $\mathbf{A}' = \mathbf{B}\mathbf{A}\mathbf{B}^{-1}$ besitzen dasselbe charakteristische Polynom $\det(\mathbf{A} - \lambda\mathbf{I}) = \prod_{i=1}^n (\lambda_i - \lambda)$.

<!--en:a9366444-->
- **Identitäten für Spur und Determinante:** $\operatorname{tr}\mathbf{A} = \sum_{i=1}^n \lambda_i$ und $\det\mathbf{A} = \prod_{i=1}^n \lambda_i$ gelten für jede quadratische Matrix.

<!--en:4ec40e25-->
- **Kurzformel für $2 \times 2$-Matrizen:** $p(\lambda) = \lambda^2 - (\operatorname{tr}\mathbf{A})\lambda + \det\mathbf{A} = 0$.

<!--en:57c0eb7c-->
::: {.slide title="Verschiedene Eigenwerte garantieren lineare Unabhängigkeit"}

<!--en:9852fda4-->
[Vielfachheit und Diagonalisierbarkeit]{.kicker}

<!--en:7aac4cfe-->
- **Verschiedene Eigenwerte $\implies$ linear unabhängige Eigenvektoren:** Durch Anwendung von $\mathbf{A} - \lambda_m\mathbf{I}$ auf eine kürzeste Abhängigkeitsrelation zeigt man: Bei $n$ verschiedenen Eigenwerten ist $\mathbf{A}$ stets diagonalisierbar.

<!--en:76573358-->
- **Schranke für die Vielfachheit ($1 \le g(\lambda) \le a(\lambda)$):** Die geometrische Vielfachheit $g(\lambda) = \dim\ker(\mathbf{A} - \lambda\mathbf{I})$ übersteigt nie die Vielfachheit $a(\lambda)$ als Nullstelle.

<!--en:989bd8ce-->
- **Kriterium für Diagonalisierbarkeit:** $\mathbf{A}$ ist über $\mathbb{C}$ genau dann diagonalisierbar, wenn $g(\lambda) = a(\lambda)$ für jeden Eigenwert $\lambda$ gilt.

<!--en:6da75ed7-->
::: {.slide title="Defekte Matrizen: Der Scherung fehlt eine Eigenbasis"}

<!--en:230650df-->
[Grenzen der Diagonalisierung]{.kicker}

<!--en:4f08bac6-->
- **Doppelte Nullstelle ($\lambda = 1$ tritt zweimal auf):** Für die horizontale Scherung $\mathbf{T} = \left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ gilt $\det(\mathbf{T} - \lambda\mathbf{I}) = (1 - \lambda)^2 = 0$. Die algebraische Vielfachheit ist also $a(1) = 2$.

<!--en:78bf7c71-->
- **Eine einzige Eigenrichtung ($g(1) = 1$):** Aus $(\mathbf{T} - \mathbf{I})\mathbf{v} = \left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)\mathbf{v} = \mathbf{0}$ folgt $v_2 = 0$. Es bleibt nur die horizontale Achse $\operatorname{span}\{(1, 0)\}$.

<!--en:d8a2e493-->
- **Lineare Drift bei Potenzierung:** Wegen $g(1) < a(1)$ ist $\mathbf{T}$ *defekt*. Die Potenzen $\mathbf{T}^k = \left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$ wachsen linear in $k$, obwohl $\lambda = 1$ gilt.

<!--en:227feb56-->
::: {.slide title="Jordanblöcke multiplizieren λᵏ mit einem Polynom in k"}

<!--en:0f2d91f9-->
[Jordansche Normalform]{.kicker}

<!--en:abd93422-->
- **Jordanblock ($\mathbf{J}_m(\lambda) = \lambda\mathbf{I} + \mathbf{N}$):** Aus der Nilpotenz der Verschiebung ($\mathbf{N}^m = \mathbf{0}$) folgt $\mathbf{J}_2(\lambda)^k = \lambda^k\mathbf{I} + k\lambda^{k-1}\mathbf{N}$.

<!--en:cd5fe572-->
- **Transiente Verstärkung:** Für $\lambda = 0.99$ wächst $k(0.99)^{k-1}$ zunächst auf $37.0$ bei $k = 100$ und klingt anschließend ab.

<!--en:123cc1fe-->
::: {.slide title="Jordanblöcke multiplizieren λᵏ mit einem Polynom in k (Fortsetzung)"}

<!--en:0f2d91f9-2-->
[Jordansche Normalform]{.kicker}

<!--en:b8d52b9d-->
::: {.slide title="Komplexe Eigenwerte beschreiben Drehung und Skalierung"}

<!--en:7a8e3bc2-->
[Komplexe Spektren reeller Matrizen]{.kicker}

<!--en:98e9804b-->
- **Reelle invariante Ebene:** Die Zerlegung von $\mathbf{A}(\mathbf{u} + i\mathbf{v}) = (a + ib)(\mathbf{u} + i\mathbf{v})$ in Real- und Imaginärteil liefert $\mathbf{B}^{-1}\mathbf{A}\mathbf{B} = r\mathbf{R}_\theta$ in der Basis $\mathbf{B} = [\mathbf{v}\;\;\mathbf{u}]$.

<!--en:7ad9bf2b-->
- **Betrag und Winkel:** Der Block skaliert um den Faktor $r = \sqrt{a^2 + b^2}$ und dreht um den Winkel $\theta = \arg(a + ib)$.

<!--en:c1f34002-->
- **Beispielmatrix $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$:** $\lambda = 2 \pm i = \sqrt{5}\,e^{\pm i\,26.6^\circ}$.

<!--en:1da9503e-->
::: {.slide title="Elliptische Spiralen der Beispielmatrix A"}

<!--en:241031c3-->
[Komplexe Spektren im Code]{.kicker}

<!--en:001bb5f3-->
Real- und Imaginärteil des komplexen Eigenvektors bilden die Basis, in der $\mathbf{A}$ genau den Block $\left(\begin{smallmatrix}2&-1\\1&2\end{smallmatrix}\right)$ ergibt. In Standardkoordinaten schwanken die Normverhältnisse von Schritt zu Schritt:

<!--en:a23ab865-->
::: {.slide title="Elliptische Spiralen der Beispielmatrix A (Fortsetzung)"}

<!--en:241031c3-2-->
[Komplexe Spektren im Code]{.kicker}

<!--en:23094045-->
**Warum die Normverhältnisse zwischen $1.4$ und $3.5$ schwanken:** Die Drehstreckung ist nur in der nichtorthogonalen Basis $(\mathbf{v}, \mathbf{u})$ gleichmäßig. In Standardkoordinaten folgen die Iterierten einer Spirale auf geneigten Ellipsen. Das geometrische Mittel der Wachstumsfaktoren nähert sich $\sqrt{5} \approx 2.236$.

<!--en:ae095b18-->
::: {.slide title="Drei Grundformen der Wirkung reeller quadratischer Matrizen"}

<!--en:52fed4ce-->
[Überblick zur reellen jordanschen Normalform]{.kicker}

<!--en:a5f7d86b-->
- **1. Verschiedene reelle Eigenwerte:** Rein exponentielle Skalierung $\lambda_i^k$ entlang reeller Achsen (Übergangsmatrix $\mathbf{P}$, symmetrische Matrix $\mathbf{S}$).

<!--en:8d9b3577-->
- **2. Defekter Jordanblock:** Ein Polynom mal einer Exponentialfunktion, $k\lambda^{k-1}$, bewirkt Drift oder transientes Wachstum (Scherung $\mathbf{T}$, $\mathbf{J}_2(0.99)$).

<!--en:366a9ee5-->
- **3. Komplex konjugiertes Paar:** Drehstreckung $r^k\mathbf{R}_{k\theta}$ in einer invarianten Ebene (Beispielmatrix $\mathbf{A}$).

<!--en:31c29a8e-->
::: {.slide title="Der Spektralsatz für reelle symmetrische Matrizen"}

<!--en:c572779f-->
[Symmetrische Matrizen]{.kicker}

<!--en:934f22cc-->
- **Spektralsatz ($\mathbf{A} = \mathbf{A}^\top \implies \mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$):** Reelle Eigenwerte $\boldsymbol{\Lambda}$ und orthonormale Eigenvektoren $\mathbf{V}^\top\mathbf{V} = \mathbf{I}$.

<!--en:2831ea0a-->
- **Teil (i), reelle $\lambda$:** $\lambda\|\mathbf{v}\|^2 = \mathbf{v}^*\mathbf{A}\mathbf{v} = \bar{\lambda}\|\mathbf{v}\|^2 \implies \lambda \in \mathbb{R}$.

<!--en:c65b9c45-->
- **Teil (ii), orthogonale $\mathbf{v}_i$:** $(\lambda - \mu)\mathbf{u}^\top\mathbf{v} = 0 \implies \mathbf{u} \perp \mathbf{v}$ für $\lambda \neq \mu$.

<!--en:15632d1d-->
::: {.slide title="Induktionsbeweis und Geometrie des Spektralsatzes"}

<!--en:c572779f-2-->
[Symmetrische Matrizen]{.kicker}

<!--en:061efea5-->
- **Beweis von Teil (iii) durch Induktion im orthogonalen Komplement:** Sei $\mathbf{v}_1$ ein normierter Eigenvektor von $\mathbf{A}$. Wegen der Symmetrie bleibt der $(n-1)$-dimensionale Unterraum $U = \mathbf{v}_1^\perp$ invariant, denn $\mathbf{x} \perp \mathbf{v}_1 \implies (\mathbf{A}\mathbf{x})^\top\mathbf{v}_1 = \mathbf{x}^\top(\mathbf{A}\mathbf{v}_1) = \lambda_1(\mathbf{x}^\top\mathbf{v}_1) = 0$.

<!--en:dc98e144-->
- **Einschränkung auf $U$:** Eine Orthonormalbasis $\mathbf{Q} \in \mathbb{R}^{n \times (n-1)}$ von $U$ liefert eine kleinere symmetrische $(n-1) \times (n-1)$-Matrix $\mathbf{C} = \mathbf{Q}^\top\mathbf{A}\mathbf{Q}$. Ihre Orthonormalbasis aus Eigenvektoren vervollständigt $\mathbf{V} = [\mathbf{v}_1\;\;\mathbf{Q}\mathbf{V}']$.

<!--en:3a880094-->
- **Drehen, skalieren und zurückdrehen:** Wegen $\mathbf{V}^{-1} = \mathbf{V}^\top$ dreht eine symmetrische Matrix $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ die orthonormalen Eigenachsen auf die Standardachsen ($\mathbf{V}^\top$), streckt jede Achse um den Faktor $\lambda_i$ ($\boldsymbol{\Lambda}$) und dreht zurück ($\mathbf{V}$).

<!--en:de6b2b71-->
- **Beispiel der symmetrischen Matrix $\mathbf{S} = \left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$:** Die normierten Eigenvektoren $(1, 1)/\sqrt{2}$ ($\lambda_1 = 3$) und $(1, -1)/\sqrt{2}$ ($\lambda_2 = 1$) stehen senkrecht aufeinander und liegen auf den Halbachsen der Bildellipse.

<!--en:328c3b8e-->
::: {.slide title="Quadratische Formen und positive Definitheit"}

<!--en:12555027-->
[Quadratische Formen]{.kicker}

<!--en:38094e0a-->
- **Quadratsumme in der Eigenbasis:** Durch Einsetzen von $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ und $\mathbf{y} = \mathbf{V}^\top\mathbf{x}$ zerfällt jede symmetrische quadratische Form in einzelne Terme:
  $$\mathbf{x}^\top\mathbf{A}\mathbf{x} = \mathbf{y}^\top\boldsymbol{\Lambda}\mathbf{y} = \sum_{i=1}^n \lambda_i\,y_i^2 = \sum_{i=1}^n \lambda_i (\mathbf{v}_i^\top\mathbf{x})^2$$

<!--en:472b6319-->
- **Definitheit aus den Vorzeichen der Eigenwerte:**
  - *Positiv definit* ($\mathbf{A} \succ 0$, Schüssel): $\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0$ für alle $\mathbf{x} \neq \mathbf{0} \iff$ alle $\lambda_i > 0$.
  - *Positiv semidefinit* ($\mathbf{A} \succeq 0$, Rinne): $\mathbf{x}^\top\mathbf{A}\mathbf{x} \ge 0$ für alle $\mathbf{x} \iff$ alle $\lambda_i \ge 0$.
  - *Indefinit* (Sattel): Eigenwerte mit verschiedenen Vorzeichen $\lambda_i$.

<!--en:bb2fce15-->
::: {.slide title="Gram-Matrizen, Hesse-Matrizen und der Cholesky-Test"}

<!--en:1e9896d3-->
[Positive Definitheit im maschinellen Lernen]{.kicker}

<!--en:1337be84-->
- **Positiv definite und positiv semidefinite Matrizen im Deep Learning:**
  - *Gram-Matrizen und Kovarianzmatrizen:* Für jedes $\mathbf{X}$ gilt $\mathbf{x}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{x} = \|\mathbf{X}\mathbf{x}\|^2 \ge 0$, also $\mathbf{X}^\top\mathbf{X} \succeq 0$. Es gilt auch: $\mathbf{X}^\top\mathbf{X} \succ 0 \iff \mathbf{X}$ hat vollen Spaltenrang.
  - *Strikte lokale Minima:* An einem kritischen Punkt $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ liefert eine positiv definite Hesse-Matrix $\nabla^2 f(\mathbf{x}^\star) \succ 0$ die untere Schranke $\tfrac{1}{2}\lambda_{\min}\|\boldsymbol{\delta}\|^2 > 0$ für den Taylor-Term zweiter Ordnung.
  - *Cholesky-Zerlegung ($\mathbf{A} = \mathbf{L}\mathbf{L}^\top$):* Der Nachweis eines unteren Cholesky-Faktors $\mathbf{L}$ mit positiver Diagonale ist der schnellste numerische Test für $\mathbf{A} \succ 0$. Mit diesem Cholesky-Faktor lassen sich korrelierte normalverteilte Zufallsvektoren $\mathbf{L}\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{A})$ erzeugen.

<!--en:ba99387b-->
::: {.slide title="Schranken für die quadratische Streckung durch den Rayleigh-Quotienten"}

<!--en:d2aaf5d0-->
[Variationscharakterisierung]{.kicker}

<!--en:89916bcc-->
- **Konvexkombination der Eigenwerte:** Für einen Einheitsvektor $\mathbf{x}$ ($\mathbf{y} = \mathbf{V}^\top\mathbf{x}$) gilt $R(\mathbf{x}) = \frac{\mathbf{x}^\top\mathbf{A}\mathbf{x}}{\mathbf{x}^\top\mathbf{x}} = \sum_{i=1}^n \lambda_i y_i^2 \in [\lambda_{\min}, \lambda_{\max}]$.

<!--en:16caa032-->
- **Maximale Streckung:** Für $\mathbf{A}^\top\mathbf{A}$ gilt $\max_{\|\mathbf{x}\|=1}\|\mathbf{A}\mathbf{x}\|^2 = \lambda_{\max}(\mathbf{A}^\top\mathbf{A}) = \|\mathbf{A}\|_2^2$.

<!--en:5a86c1fe-->
- **Schrittweite beim Gradientenabstieg:** Für $\tfrac{1}{2}\mathbf{x}^\top\mathbf{A}\mathbf{x}$ erfordert $|1 - \eta\lambda_i| < 1$ die Bedingung $0 < \eta < 2/\lambda_{\max}$.

<!--en:8604b65e-->
::: {.slide title="Stichprobenkovarianz bei Rauschen: Marchenko-Pastur-Gesetz"}

<!--en:05948379-->
[Hochdimensionale Statistik]{.kicker}

<!--en:23b0ae88-->
- **Marchenko-Pastur-Intervall:** Für $n$ Samples aus reinem Rauschen in $\mathbb{R}^d$ ($c = d/n$) verteilen sich die Eigenwerte von $\mathbf{C} = \tfrac{1}{n}\mathbf{X}^\top\mathbf{X}$ über $[(1 - \sqrt{c})^2, (1 + \sqrt{c})^2]$ ($[0.25, 2.25]$ bei $c = 1/4$).

<!--en:96172acd-->
- **Signalerkennung:** Erst Eigenwerte deutlich oberhalb von $(1 + \sqrt{c})^2$ deuten auf tatsächliche Struktur in den Daten hin.

<!--en:ccc614e8-->
::: {.slide title="Verständnisfragen: Eigenwerte, Potenzen und Definitheit"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:097cb772-->
1. **Eigenwerte aus Spur und Determinante:** Für eine $2 \times 2$-Matrix gilt $\operatorname{tr}\mathbf{A} = 8$ und $\det\mathbf{A} = 15$. Wie lauten ihre Eigenwerte?

<!--en:2ec0b02b-->
2. **Zwei Tests auf positive Definitheit:** Ohne Berechnung der Nullstellen von $p(\lambda)$ auf zwei unabhängigen Wegen zeigen, dass $\mathbf{S} = \left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ strikt positiv definit ist.

<!--en:0e8baef2-->
3. **Rangdefizit bei $d > n$:** Eine $512 \times 512$-Stichprobenkovarianzmatrix $\mathbf{C}$ entsteht aus einem Minibatch von $n = 256$ zentrierten Einbettungen. Wie viele Eigenwerte von $\mathbf{C}$ sind mindestens null?

<!--en:349c6fac-->
**Lösungen:**
**1.** Die Nullstellen von $\lambda^2 - 8\lambda + 15 = (\lambda - 3)(\lambda - 5) = 0$ liefern $\lambda_1 = 5$ und $\lambda_2 = 3$.
**2.** (i) Es gilt $\operatorname{tr}\mathbf{S} = 4 > 0$ und $\det\mathbf{S} = 3 > 0$, also sind beide reellen Eigenwerte positiv. (ii) Quadratische Ergänzung ergibt $\mathbf{x}^\top\mathbf{S}\mathbf{x} = x_1^2 + x_2^2 + (x_1 + x_2)^2 > 0$ für alle $\mathbf{x} \neq \mathbf{0}$.
**3.** Nach dem Zentrieren von $256$ Zeilen beträgt der Rang höchstens $n - 1 = 255$. Also hat $\mathbf{C}$ mindestens $512 - 255 = 257$ Eigenwerte null und lässt sich ohne Regularisierung nicht invertieren.

<!--en:b878817b-->
::: {.slide title="Zusammenfassung: Eigenwertzerlegungen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4617f689-->
| Begriff | Mathematische Form | Eigenschaft und Bedeutung im Deep Learning |

<!--en:b33c887e-->
| **Eigenpaar und charakteristisches Polynom** | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$, $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ | Skalierung invarianter Richtungen um $\lambda$. Es gilt $\operatorname{tr}\mathbf{A} = \sum_i \lambda_i$ und $\det\mathbf{A} = \prod_i \lambda_i$ |

<!--en:bbd9fc94-->
| **Diagonalisierung und Potenzen** | $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$, $\mathbf{A}^k\mathbf{x} = \sum_i c_i \lambda_i^k \mathbf{v}_i$ | Existiert genau dann, wenn $g(\lambda) = a(\lambda)$ für alle $\lambda$ gilt. Zerlegt die Dynamik über $k$ Schritte in unabhängige skalare Potenzen |

<!--en:69f87fcb-->
| **Defekte und komplexe Eigenwerte** | $\mathbf{J}_2(\lambda)^k = \lambda^k\mathbf{I} + k\lambda^{k-1}\mathbf{N}$, $\lambda = r e^{\pm i\theta}$ | Fehlende Eigenvektoren verursachen transientes Wachstum gemäß $k\lambda^{k-1}$. Komplex konjugierte Paare bewirken eine Drehstreckung in einer Ebene |

<!--en:ce6541e9-->
| **Spektralsatz** | $\mathbf{A} = \mathbf{A}^\top \implies \mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ | Jede reelle symmetrische Matrix besitzt reelle Eigenwerte und eine Orthonormalbasis aus Eigenvektoren ($\mathbf{V}^{-1} = \mathbf{V}^\top$) |

<!--en:4ce070cf-->
| **Quadratische Form und Rayleigh-Quotient** | $\mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_i \lambda_i y_i^2$, $R(\mathbf{x}) \in [\lambda_{\min}, \lambda_{\max}]$ | $\mathbf{A} \succ 0 \iff$ alle $\lambda_i > 0$. Schranke für die Schrittweite beim Gradientenabstieg: $\eta < 2/\lambda_{\max}$ |

<!--en:f8d35536-->
| **Marchenko-Pastur-Intervall** | $\lambda(\tfrac{1}{n}\mathbf{X}^\top\mathbf{X}) \in [(1-\sqrt{c})^2, (1+\sqrt{c})^2]$ | Die Streuung der Eigenwerte bei reinem Rauschen hängt von $c = d/n$ ab. Nur Ausreißer oberhalb von $(1+\sqrt{c})^2$ weisen auf ein Signal hin |
