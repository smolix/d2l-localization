<!--en:e1e173bc-->
# Spektren in der Praxis

<!--en:3780cc39-->
In :numref:`sec_mdl-eigendecompositions` waren unsere Matrizen klein genug, um sie von Hand zu zerlegen. Ihre Eigenwerte hielten, was sie versprachen. Mit jedem Klick sank der Fehler des Surfers auf einen Anteil von $1/4$ seines vorherigen Werts, genau wie es der zweite Eigenwert vorhersagte. Außerhalb solcher Lehrbuchbeispiele entfallen beide Vorteile. Die Linkmatrix des Webs hat Milliarden von Zeilen. Niemand stellt dafür das charakteristische Polynom auf oder berechnet eine vollständige Eigenwertzerlegung (engl. eigendecomposition). Und selbst bekannte Eigenwerte sagen weniger aus, als es zunächst scheint. Die Potenzen eines Jordan-Blocks mit zwei Eigenwerten von jeweils $0.99$ wuchsen zunächst bis zur Norm 37 an, bevor sie abklangen (:numref:`subsec_mdl-jordan`). Bei der Backpropagation durch ein rekurrentes Netz multiplizieren wir einen Gradienten mit fünfzig *verschiedenen* Matrizen, statt dieselbe Matrix fünfzigmal anzuwenden.

<!--en:195de91f-->
Betrachten wir diese beiden Probleme nacheinander. Was lässt sich erstens über die Eigenwerte einer Matrix sagen, die für eine Zerlegung viel zu groß ist? Erstaunlich viel. Schon die Einträge grenzen alle Eigenwerte auf einige Kreisscheiben in der komplexen Ebene ein. Durch wiederholte Matrix-Vektor-Multiplikation finden wir den betragsgrößten Eigenwert und seinen Eigenvektor. Diese eine Operation bleibt auch bei großen Matrizen günstig. So wurde PageRank ursprünglich berechnet. Was verraten uns Eigenwerte zweitens über eine endliche Zahl von Schritten oder über ein Produkt wechselnder Matrizen? Hier fällt die Antwort ernüchternd aus. Eigenwerte bestimmen das langfristige Verhalten der Potenzen einer einzelnen Matrix. Für alles Weitere brauchen wir Matrixnormen und schließlich die Singulärwerte aus :numref:`sec_mdl-svd-low-rank`.

<!--en:4ef0bfe2-->
Die folgenden Themen spielen im Deep Learning unterschiedlich große Rollen. Drei Ideen begegnen uns im Buch immer wieder. Mit der Potenziteration (engl. power iteration, :numref:`subsec_mdl-power-iteration`) schätzen wir die größte Streckung einer Schicht. Darauf beruht die spektrale Normalisierung (:numref:`subsec_mdl-spectrum-control`). Transientes Wachstum (engl. transient growth) und Produkte von Jacobi-Matrizen (:numref:`subsec_mdl-transient-growth`, :numref:`subsec_mdl-jacobian-products`) liefern die mathematische Grundlage für die explodierenden und verschwindenden Gradienten aus :numref:`sec_bptt`. Die Zufallsmatrizen aus :numref:`subsec_mdl-random-matrices` dienen als Modell für die Gewichtsmatrizen aus :numref:`sec_numerical_stability` bei der Initialisierung. Der Satz von Gerschgorin und PageRank sind klassische Ergebnisse, die im Deep Learning seltener vorkommen. Wenn Sie wenig Zeit haben, können Sie beide Themen überfliegen. Auch die Pseudospektren (:numref:`subsec_mdl-pseudospectra`) können Sie beim ersten Lesen überspringen. Sie sind ein spezialisiertes Diagnosewerkzeug. Alle folgenden numerischen Beispiele verwenden NumPy.

<!--en:cfbcdcaf-->
## Eingrenzung und Berechnung von Eigenwerten

<!--en:97d3f7a4-->
### Gerschgorin-Kreise

<!--en:58f1ed31-->
Können wir die Lage der Eigenwerte einer Matrix grob eingrenzen, ohne sie zu berechnen? Bei einer Diagonalmatrix sind die Eigenwerte gerade die Diagonaleinträge. Sind die Einträge außerhalb der Diagonalen klein, so sollten die Eigenwerte in der Nähe der Diagonaleinträge bleiben. Der folgende Satz präzisiert diese Vermutung. Sein Beweis braucht nur drei Zeilen.

<!--en:13b558b5-->
**Satz (Gerschgorin** :cite:`Gershgorin.1931`**).** *Sei $\mathbf{A}=(a_{ij})$ eine beliebige $n\times n$-Matrix. Für jede Zeile $i$ sei $r_i=\sum_{j\neq i}|a_{ij}|$ die Summe der Beträge ihrer Einträge außerhalb der Diagonalen. Sei $\mathcal{D}_i$ die abgeschlossene Kreisscheibe in der komplexen Ebene mit Mittelpunkt $a_{ii}$ und Radius $r_i$. Dann liegt jeder Eigenwert von $\mathbf{A}$ in der Vereinigung $\bigcup_i\mathcal{D}_i$.*

<!--en:f0b641ba-->
**Beweis.** Es gelte $\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ mit $\mathbf{v}\neq\mathbf 0$. Wählen wir einen Index $i$, an dem $\mathbf{v}$ einen betragsgrößten Eintrag hat. Dann gilt $|v_i|\ge|v_j|$ für alle $j$ und $v_i\neq0$. Aus Zeile $i$ von $\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ folgt

<!--en:49a45144-->
Wir dividieren beide Seiten durch $v_i$ und bilden die Beträge. Wegen $|v_j/v_i|\le 1$ für jedes $j$ gilt

<!--en:b20da78d-->
und damit $\lambda\in\mathcal{D}_i$. $\blacksquare$

<!--en:79e0d74d-->
Der Beweis sagt uns sogar, in welcher Kreisscheibe ein Eigenwert liegt, nämlich in derjenigen zur betragsgrößten Komponente seines Eigenvektors. Da wir diese Zeile im Voraus nicht kennen, garantiert der Satz lediglich, dass jeder Eigenwert in der Vereinigung aller $n$ Kreisscheiben liegt. Betrachten wir dazu die nichtsymmetrische Matrix

<!--en:871f4793-->
Ihre drei Kreisscheiben haben die Mittelpunkte $4$, $4$ und $1$ und die zugehörigen Radien $1$, $1.5$ und $0.5$. Die beiden Kreisscheiben um $4$ überlappen sich. Die eine liegt sogar ganz in der anderen. Die Kreisscheibe um $1$ ist von beiden getrennt, denn $|4-1|=3>1.5+0.5$. Die tatsächlichen Eigenwerte sind $3.99\pm0.96i$ und $1.03$. Das komplex konjugierte Paar liegt in beiden Kreisscheiben um $4$, der reelle Eigenwert in der isolierten Kreisscheibe um $1$. :numref:`fig_mdl-la-gershgorin` zeigt die Kreisscheiben zusammen mit den Eigenwerten.

<!--en:70ab8624-->
![Gerschgorin-Kreise der $3\times3$-Matrix aus dem Text und ihre Eigenwerte (Kreuze). Die Mittelpunkte sind die Diagonaleinträge, die Radien die Summen der Beträge aller übrigen Einträge der jeweiligen Zeile. Die beiden überlappenden Kreisscheiben um 4 enthalten zusammen zwei Eigenwerte, das komplexe Paar. Die isolierte Kreisscheibe um 1 enthält den dritten.](../img/mdl-la-gershgorin.svg)

<!--en:f5d358d4-->
Dass zwei Eigenwerte in den überlappenden Kreisscheiben liegen und einer in der isolierten Kreisscheibe, ist kein Zufall. *Ist die Vereinigung von $m$ Kreisscheiben zusammenhängend und zu allen übrigen Kreisscheiben disjunkt, so enthält sie genau $m$ Eigenwerte*, mit Vielfachheit gezählt. Das folgt aus der Stetigkeit. Erhöhen wir den Skalierungsfaktor der Einträge außerhalb der Diagonalen von null auf eins, so starten die Eigenwerte bei den Diagonaleinträgen und bewegen sich stetig. Sie können daher die Lücke zwischen getrennten Gruppen von Kreisscheiben nicht überspringen :cite:`Horn.Johnson.2012`. Überprüfen wir die Mittelpunkte, Radien und Eigenwerte numerisch.

<!--en:6df61825-->
Hier liegen die beiden komplexen Eigenwerte im Abstand $0.96$ von $4$, also dicht am Rand der inneren Kreisscheibe mit Radius $1$. Die Schranke von Gerschgorin ist hier also nahezu scharf.

<!--en:89731b45-->
Eine unmittelbare Folgerung kommt ganz ohne Rechnung aus. Ist $\mathbf{A}$ *streng diagonaldominant* mit positiven Diagonaleinträgen, gilt also $a_{ii}>r_i$ für jedes $i$, so liegt jede Kreisscheibe ganz in der offenen rechten Halbebene. Null kann kein Eigenwert sein, also ist $\mathbf{A}$ invertierbar. Ist $\mathbf{A}$ zudem symmetrisch, so sind ihre Eigenwerte reell und strikt positiv. Damit ist sie positiv definit.

<!--en:b5c7190c-->
### Potenziteration

<!--en:12cb0c67-->
Gerschgorin-Kreise grenzen die Lage der Eigenwerte nur grob ein. Was tun wir, wenn wir einen Eigenwert einer Matrix mit einer Milliarde Zeilen genau bestimmen wollen? Eine Zerlegung kommt bei dieser Größe nicht infrage. Ist die Matrix jedoch dünn besetzt, können wir sie mit geringem Aufwand mit einem Vektor multiplizieren. Was passiert, wenn wir mit einem Vektor $\mathbf{v}_0$ beginnen und ihn immer wieder mit $\mathbf{A}$ multiplizieren?

<!--en:0fec5fef-->
Die Entwicklung $\mathbf{A}^k\mathbf{x} = \sum_i c_i\lambda_i^k\mathbf{v}_i$ aus :numref:`subsec_mdl-diagonalization` liefert die Antwort. Sei $\mathbf{A}$ diagonalisierbar mit Eigenvektoren $\mathbf{v}_i$, geordnet nach absteigendem Betrag ihrer Eigenwerte. Der betragsgrößte Eigenwert sei *streng dominant*, das heißt, es gelte $|\lambda_1|>|\lambda_2|\ge\cdots\ge|\lambda_n|$. (Ist $\mathbf{A}$ reell, so muss bei strenger Dominanz auch $\lambda_1$ reell sein. Ein nichtreeller Eigenwert hätte denselben Betrag wie sein komplex konjugierter Partner.) Setzen wir $\mathbf{v}_0=\sum_i c_i\mathbf{v}_i$ ein und klammern $\lambda_1^k$ aus, so erhalten wir

<!--en:5a0cda42-->
Jeder Quotient $|\lambda_i/\lambda_1|$ mit $i\ge 2$ ist strikt kleiner als $1$. Daher verschwindet jeder der hinteren Terme mit wachsendem $k$. Sofern $c_1\neq 0$ gilt, konvergiert der Klammerausdruck gegen $c_1\mathbf{v}_1$. Die Richtung von $\mathbf{A}^k\mathbf{v}_0$ nähert sich also der des dominanten Eigenvektors $\mathbf{v}_1$, wobei der Fehler wie $|\lambda_2/\lambda_1|^k$ abnimmt. Die Länge ändert sich bei jedem Schritt um einen Faktor, der gegen $|\lambda_1|$ strebt. Um Über- und Unterläufe zu vermeiden, normieren wir den Vektor nach jeder Multiplikation wieder auf Länge eins. Diese Schleife heißt *Potenziteration* :cite:`Golub.Van-Loan.1996`. Ist $\mathbf{A}$ symmetrisch, können wir den Eigenwert selbst noch genauer schätzen. Der Rayleigh-Quotient der aktuellen Iterierten konvergiert gegen $\lambda_1$, und zwar mit der quadrierten Rate $|\lambda_2/\lambda_1|^{2k}$. Da die Eigenvektoren einer symmetrischen Matrix orthogonal gewählt werden können, verändert ein Richtungsfehler der Größe $\delta$ den Rayleigh-Quotienten nur um $O(\delta^2)$ (Aufgabe :exref:`ex_mdl-eigendecomposition-power-iteration-rates`).

<!--en:358f4c83-->
Beide Raten lassen sich an einem $2\times 2$-Beispiel beobachten. In :numref:`sec_linear-algebra` haben wir die Iterierten von $\mathbf{S}=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ mit Startvektor $\mathbf{e}_1$ berechnet. Wir erhielten $(2,1)$, $(5,4)$, $(14,13)$ und allgemein $\bigl((3^k+1)/2,\,(3^k-1)/2\bigr)$. Für die normierten Eigenvektoren $\mathbf{v}_1=(1,1)/\sqrt2$ und $\mathbf{v}_2=(1,-1)/\sqrt2$ gilt $\mathbf{e}_1=(\mathbf{v}_1+\mathbf{v}_2)/\sqrt2$, also

<!--en:89e04bde-->
Mit $|\lambda_2/\lambda_1|=1/3$ sinkt der Richtungsfehler bei jedem Schritt um den Faktor $3$. Der Sinus des Winkels zu $\mathbf{v}_1$ beträgt $0.316$, $0.110$ und $0.037$ für $k=1,2,3$. Der Fehler des Rayleigh-Quotienten sinkt dagegen pro Schritt etwa um den Faktor $9$ und nimmt die Werte $0.2$, $0.024$ und $0.0027$ an. :numref:`fig_mdl-la-power-iter` zeigt beide Kurven in logarithmischer Skala.

<!--en:d5557aff-->
![Potenziteration für $\mathbf S=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ mit Startvektor $\mathbf e_1$. Der Richtungsfehler sinkt pro Schritt um den Faktor 3, der Fehler der Eigenwertschätzung um den Faktor 9. Die quadrierte Rate ergibt sich aus der Orthogonalität der Eigenvektoren. Die gestrichelten Linien zeigen $3^{-k}$ und $2\cdot9^{-k}$.](../img/mdl-la-power-iter.svg)

<!--en:60bc8016-->
Wann kann die Potenziteration scheitern? Zwei Voraussetzungen in :eqref:`eq_mdl-power-iter` sind wesentlich. Erstens braucht der Startvektor eine von null verschiedene Komponente in Richtung $\mathbf{v}_1$ ($c_1\neq 0$). Bei zufälliger Wahl von $\mathbf{v}_0$ ist das mit Wahrscheinlichkeit eins der Fall. Zweitens muss der betragsgrößte Eigenwert streng dominant sein. Haben zwei komplex konjugierte Eigenwerte denselben maximalen Betrag, so dreht sich die dominante Komponente der Iterierten immer weiter und nähert sich keiner einzelnen Geraden. Der Quotient aufeinanderfolgender Normen oszilliert dann fortwährend, wie bei unserer Beispielmatrix aus :numref:`subsec_mdl-complex-rotation`. Doch selbst dann konvergiert das geometrische Mittel $\|\mathbf A^k\mathbf v_0\|^{1/k}$ gegen den größten Eigenwertbetrag.

<!--en:57f7afb6-->
Bei Matrizen mit strikt positiven Einträgen tritt dieses Problem gar nicht erst auf. Wie wir in :numref:`subsec_mdl-pagerank` sehen, garantiert der Satz von Perron-Frobenius einen reellen, einfachen, streng dominanten Eigenwert. Prüfen wir das an einer zufälligen $5\times 5$-Matrix mit gleichverteilten Einträgen aus $[0,1]$. Wir beginnen mit einem Zufallsvektor, führen 200 Schritte der Potenziteration aus und vergleichen den letzten Normquotienten mit dem tatsächlichen größten Eigenwertbetrag.

<!--en:aa1feedf-->
Bei jeder Ziehung stimmt der Normquotient in allen ausgegebenen Stellen mit dem größten Eigenwertbetrag überein. Um zu prüfen, wie typisch diese schnelle Konvergenz ist, berechnen wir das Betragsverhältnis $|\lambda_2/\lambda_1|$ für 10 000 solcher positiven Matrizen. Zum Vergleich ziehen wir auch 10 000 Matrizen mit standardnormalverteilten Einträgen. Diese sind nicht zwingend positiv. Wir messen, wie oft ihre dominanten Eigenwerte ein komplex konjugiertes Paar bilden.

<!--en:28a95d0e-->
Bei gleichverteilten positiven Einträgen liegt das Betragsverhältnis in neun von zehn Ziehungen zwischen etwa $0.14$ und $0.37$. Nach 200 Iterationen liegt der Richtungsfehler damit weit unter der Gleitkommagenauigkeit. Bei normalverteilten Einträgen ergibt sich dagegen in etwa vier von zehn Ziehungen ein dominantes komplexes Paar. Für diese Matrizen konvergiert der Normquotient nicht.

<!--en:cd04626c-->
### PageRank

<!--en:bdcc93ad-->
Bei Matrizen mit nichtnegativen Einträgen interessiert uns oft gerade der dominante Eigenvektor. Ihn berechnen wir mit der Potenziteration. Um anzugeben, wann dieser Eigenvektor eindeutig bestimmt ist und welche Eigenschaften er hat, nennen wir eine nichtnegative Matrix *irreduzibel*, wenn in ihrem gerichteten Graphen jeder Knoten von jedem anderen aus erreichbar ist. Der Graph hat dabei eine Kante $j\to i$, wann immer $a_{ij}>0$ gilt. Nach dem *Satz von Perron-Frobenius* :cite:`Horn.Johnson.2012` hat jede irreduzible nichtnegative Matrix einen reellen, positiven Eigenwert, der gleich dem größten Eigenwertbetrag ist. Dieser Eigenwert ist einfach, und sein Eigenvektor lässt sich mit strikt positiven Einträgen wählen. Ist jeder Matrixeintrag strikt positiv, so ist dieser Eigenwert auch streng dominant. Jeder andere Eigenwert hat dann einen strikt kleineren Betrag.

<!--en:e70e0e8c-->
Googles ursprünglicher PageRank-Algorithmus :cite:`Brin.Page.1998` führt die Rangordnung von Webseiten auf genau ein solches Eigenvektorproblem zurück. Betrachten wir ein kleines Web aus drei Seiten mit den Links $1\to2$, $2\to1$ und $2\to3$. Ein Surfer wählt zufällig einen der ausgehenden Links, alle mit gleicher Wahrscheinlichkeit. Seite $3$ hat keine ausgehenden Links. Wir beheben diese Sackgasse, indem wir von Seite $3$ aus jede Seite mit gleicher Wahrscheinlichkeit als Sprungziel zulassen. Die so korrigierte Übergangsmatrix $\mathbf{P}$ ist spaltenstochastisch. Spalte $j$ enthält die Klickwahrscheinlichkeiten für die Sprünge von Seite $j$:

<!--en:97973814-->
Langfristig bilden die Zeitanteile, die der Surfer auf den einzelnen Seiten verbringt, die stationäre Verteilung $\boldsymbol{\pi}$ mit $\mathbf{P}\boldsymbol{\pi}=\boldsymbol{\pi}$. Diese Gleichung lässt sich von Hand lösen. Die Zeilen $1$ und $3$ ergeben zusammen $\pi_1=\tfrac12\pi_2+\tfrac13\pi_3=\pi_3$. Setzen wir $\pi_1=\pi_3$ wieder in Zeile $1$ ein, so folgt $\pi_2=\tfrac43\pi_3$. Normieren wir die Einträge auf die Summe $1$, erhalten wir $\boldsymbol{\pi}=(0.3,0.4,0.3)$. Seite $2$ steht an erster Stelle, weil ihr Seite $1$ alle Klicks schickt. Seite $2$ hingegen verteilt ihre Klicks auf die Seiten $1$ und $3$. Die Eigenwerte von $\mathbf{P}$ sind $1$, $-\tfrac23$ und $0$. Die Potenziteration konvergiert daher von jeder Anfangsverteilung aus mit der Rate $2/3$ pro Schritt.

<!--en:4dedcb62-->
Tatsächlich hat *jede* spaltenstochastische Matrix $1$ als betragsgrößten Eigenwert. Erstens ergibt jede Spaltensumme $1$, also gilt $\mathbf{1}^\top\mathbf{P}=\mathbf{1}^\top$. Damit ist $1$ ein Eigenwert von $\mathbf{P}^\top$ und folglich auch von $\mathbf{P}$. (Eine Matrix und ihre Transponierte haben dasselbe charakteristische Polynom, denn $\det\mathbf{X}^\top=\det\mathbf{X}$.) Zweitens können wir den Satz von Gerschgorin auf $\mathbf{P}^\top$ anwenden, deren Zeilen die Spalten von $\mathbf{P}$ sind. Jeder Eigenwert liegt dann in einer Kreisscheibe mit Mittelpunkt $p_{ii}\ge 0$ und Radius $1-p_{ii}$. Jede solche Kreisscheibe liegt ganz in der Einheitskreisscheibe.

<!--en:5a014a71-->
Das reale Web ist allerdings verwickelter als unser Beispiel mit drei Seiten. Es enthält Zyklen von Links und isolierte Seitengruppen, aus denen der Surfer nicht entkommen kann. Die korrigierte Matrix $\mathbf{P}$ kann dann reduzibel sein oder mehrere Eigenwerte vom Betrag $1$ haben. Ein einfacher Zyklus aus vier Seiten hat beispielsweise die Eigenwerte $\pm 1$ und $\pm i$, weshalb die ungedämpfte Potenziteration auf diesem Zyklus nicht zur Ruhe kommt (Aufgabe :exref:`ex_mdl-eigendecomposition-pagerank-damping`). PageRank löst beide Probleme zugleich durch Iteration mit der *gedämpften* Matrix

<!--en:59a39ac2-->
Diese Matrix beschreibt einen Surfer, der mit Wahrscheinlichkeit $0.85$ einem zufälligen Link folgt und mit Wahrscheinlichkeit $0.15$ zu einer zufällig und gleichverteilt gewählten Seite springt. Da jeder Eintrag von $\mathbf{G}$ strikt positiv ist, gilt der Satz von Perron-Frobenius bereits aufgrund der Konstruktion. Darüber hinaus zeigen :citet:`Haveliwala.Kamvar.2003`, dass $|\lambda_2(\mathbf{G})|\le\alpha$ gilt. Der Fehler der Potenziteration sinkt also bei jedem Schritt auf höchstens das $0.85$-Fache, unabhängig davon, wie verwickelt der Linkgraph ist. Diese eine Schranke machte PageRank für das gesamte Web praktikabel. Fünfzig Durchläufe verringern den Fehler um mehr als den Faktor $3000$ ($0.85^{50}\approx 3\times 10^{-4}$), unabhängig davon, wie viele Milliarden Seiten der Graph enthält.

<!--en:3df2dfa3-->
Für das gesamte Web speichern wir $\mathbf{P}$ dünn besetzt mit einem Eintrag pro Link. Die dicht besetzte Matrix $\mathbf{G}$ stellen wir nie explizit auf. Für jeden Wahrscheinlichkeitsvektor $\mathbf{p}$ (mit $\mathbf{1}^\top\mathbf{p}=1$) lässt sich $\mathbf{G}\mathbf{p}=\alpha\mathbf{P}\mathbf{p}+(1-\alpha)\tfrac1n\mathbf{1}$ mit einem einzigen Matrix-Vektor-Produkt für die dünn besetzte Matrix und einer konstanten Verschiebung berechnen. Die Korrektur für Seiten ohne ausgehende Links ist ein Rang-eins-Term, den wir genauso behandeln.

<!--en:e6957210-->
## Iteration und Stabilität

<!--en:833db468-->
### Der Spektralradius

<!--en:bb43f978-->
Bei der Potenziteration kam es auf eine Zahl an, den größten Betrag unter den Eigenwerten. Diese Zahl heißt *Spektralradius*, $\rho(\mathbf{A})=\max_i|\lambda_i|$. Sie bestimmt die langfristige Wachstumsrate der Potenzen einer festen Matrix. Die Formel von Gelfand präzisiert das. Für jede quadratische Matrix $\mathbf{A}$ und jede Matrixnorm gilt

<!--en:039e84e6-->
Die einfachere Hälfte dieser Aussage, nämlich $\rho(\mathbf{A})\le\|\mathbf{A}^k\|^{1/k}$ für jedes $k\ge 1$, ist Gegenstand von Aufgabe :exref:`ex_mdl-eigendecomposition-spectral-radius-norms`.

<!--en:b72bdc5c-->
Was die Formel von Gelfand *nicht* verrät, ist das Verhalten bei einem festen, endlichen Schritt $k$. Wir kennen bereits zwei Matrizen, bei denen dieser Unterschied ins Gewicht fällt. Die Scherung $\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ hat $\rho(\mathbf{T})=1$, doch ihre Potenzen $\mathbf{T}^k=\left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$ wachsen unbegrenzt. Bei $\rho\le 1$ bleiben die Iterierten also nicht einmal zwangsläufig beschränkt. Für den Jordan-Block $\mathbf{J}_2(0.99)$ gilt $\rho<1$, sodass seine Potenzen schließlich gegen null gehen. Zuvor steigen sie jedoch bis zur Norm $37$ an. Auch $\rho<1$ schließt also starkes transientes Wachstum nicht aus. Wann bestimmen die Eigenwerte das Verhalten in *jedem* Schritt und nicht nur im Grenzfall?

<!--en:30ac121b-->
### Normale Matrizen

<!--en:709d24c4-->
Unsere beiden Gegenbeispiele, die Scherung und der Jordan-Block, sind defekt. Ihnen fehlen linear unabhängige Eigenvektoren, um eine Basis zu bilden. Betrachten wir gleich das andere Extrem, nämlich Matrizen mit einer *orthonormalen* Basis aus Eigenvektoren. Eine solche Matrix wirkt unabhängig entlang paarweise senkrechter Richtungen. Ihre Eigenwerte sollten ihr Verhalten daher in jedem Schritt vollständig beschreiben.

<!--en:f288ab87-->
Eine reelle Matrix heißt *normal*, wenn sie mit ihrer Transponierten kommutiert, also $\mathbf{A}^\top\mathbf{A}=\mathbf{A}\mathbf{A}^\top$ gilt. (Bei einer komplexen Matrix verwenden wir die konjugiert Transponierte $\mathbf{A}^*$ und fordern $\mathbf{A}^*\mathbf{A}=\mathbf{A}\mathbf{A}^*$.) Symmetrische Matrizen, orthogonale Matrizen und die Drehstreckungsblöcke $r\mathbf{R}_\theta$ aus :numref:`subsec_mdl-complex-rotation` sind alle normal. Unsere Beispielmatrix $\mathbf{A}=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ ist dagegen nicht normal. Es gilt $\mathbf{A}\mathbf{A}^\top=\left(\begin{smallmatrix}5&5\\5&10\end{smallmatrix}\right)$, aber $\mathbf{A}^\top\mathbf{A}=\left(\begin{smallmatrix}2&-1\\-1&13\end{smallmatrix}\right)$. Weil die beiden Matrizen nicht kommutieren, folgten die Iterierten einer spiralförmigen Bahn entlang von Ellipsen statt Kreisen.

<!--en:8c40641f-->
Der *komplexe Spektralsatz* :cite:`Horn.Johnson.2012` charakterisiert diese Klasse vollständig. Eine Matrix ist genau dann normal, wenn sie eine Orthonormalbasis aus Eigenvektoren über $\mathbb{C}$ besitzt. Es gilt dann $\mathbf{A}=\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^*$ mit einer unitären Matrix $\mathbf{U}$ ($\mathbf{U}^*\mathbf{U}=\mathbf{I}$). Der Beweis folgt demselben Induktionsargument wie beim reellen Spektralsatz in :numref:`subsec_mdl-spectral-theorem`.

<!--en:fc5aaf26-->
Um die größtmögliche Verstärkung in einem Schritt zu messen, schreiben wir $\|\mathbf{X}\|_2=\max_{\|\mathbf{x}\|=1}\|\mathbf{X}\mathbf{x}\|$ für die *Spektralnorm* von $\mathbf{X}$. Sie ist der größte Faktor, um den $\mathbf{X}$ einen Vektor strecken kann. Da ein unitärer Basiswechsel euklidische Längen erhält, gilt für jede Potenz einer normalen Matrix

<!--en:da05fbcc-->
Bei einer normalen Matrix bestimmen die Eigenwerte die größtmögliche Verstärkung also in jedem einzelnen Schritt $k$ und nicht nur asymptotisch.

<!--en:f7227730-->
Was passiert, wenn $\mathbf{A}$ diagonalisierbar ist, also $\mathbf{A}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ gilt, die Matrix aber nicht normal ist? Die Eigenvektormatrix $\mathbf{V}$ ist dann nicht mehr orthogonal. Die Submultiplikativität der Spektralnorm liefert nur die obere Schranke

<!--en:01376cd7-->
Der Vorfaktor $\kappa_2(\mathbf{V})$ ist die Konditionszahl (:numref:`subsec_mdl-condition-number`) der Eigenvektormatrix. Für orthonormale Eigenvektoren ist er gleich $1$. Zeigen zwei Eigenvektoren in fast dieselbe Richtung, kann er beliebig groß werden. Das transiente Wachstum, das die Schranke zulässt, hat einen geometrischen Grund. Die Komponenten löschen sich gegenseitig aus. Sind Eigenvektoren fast parallel, kann die Darstellung eines gewöhnlichen Einheitsvektors sehr große Komponenten entlang der Eigenrichtungen erfordern, die sich fast aufheben. Wenden wir $\mathbf{A}$ wiederholt an, so klingen die Komponenten unterschiedlich schnell ab. Die Auslöschung lässt nach, und die Länge der Iterierten wächst stark an, bevor schließlich der gemeinsame exponentielle Abfall überwiegt.

<!--en:dd19c715-->
### Transientes Wachstum

<!--en:ae2ff35d-->
Die einfachste Matrix, an der wir dieses transiente Wachstum beobachten können, ist die obere $2\times 2$-Dreiecksmatrix

<!--en:ece013d5-->
Ihre Eigenwerte sind die Diagonaleinträge $0.9$ und $0.8$. Beide liegen deutlich innerhalb des Einheitskreises. Da sie verschieden sind, ist $\mathbf{M}$ diagonalisierbar. Berechnen wir ihre Eigenvektoren von Hand und stellen wir den Einheitsvektor $\mathbf{e}_2=(0,1)$ in dieser Basis dar. Für $\lambda=0.9$ lautet die zweite Zeile von $\mathbf{M}-0.9\mathbf{I}$ in der Eigenvektorgleichung $-0.1v_2=0$. Das ergibt $\mathbf{v}_1=(1,0)$. Für $\lambda=0.8$ lautet die erste Zeile von $\mathbf{M}-0.8\mathbf{I}$ entsprechend $0.1v_1+4v_2=0$. Wir erhalten $\mathbf{v}_2=(-40,1)$. Die beiden Eigenrichtungen liegen bemerkenswert dicht beieinander. Ihr spitzer Winkel beträgt nur $\arctan(1/40)\approx 1.4^\circ$. Für $\mathbf{e}_2$ ergibt sich in dieser Basis aus fast parallelen Vektoren die Darstellung $\mathbf{e}_2=40\mathbf{v}_1+\mathbf{v}_2$. Die Entwicklung der Matrixpotenz in Eigenvektoren (:numref:`subsec_mdl-diagonalization`) liefert

<!--en:8a4b4d40-->
Bei $k=0$ entsteht ein Vektor der Länge $1$ als Differenz zweier Komponenten entlang der Eigenrichtungen, die jeweils etwa die Länge $40$ haben. Beide Komponenten klingen bei jedem Schritt ab. Da aber $0.8^k$ schneller fällt als $0.9^k$, heben sich die Komponenten immer weniger auf, und ihre Summe wächst zunächst. Die erste Koordinate $40(0.9^k-0.8^k)$ erreicht ihr Maximum dort, wo ihre Ableitung nach $k$ verschwindet, also bei $(9/8)^k=\ln 0.8/\ln 0.9$. Das ist bei $k\approx 6.4$ der Fall, mit einer Höhe von etwa $10.8$. :numref:`fig_mdl-la-transient-cancel` zeigt die beiden Komponenten bei $k=0$ und bei $k=6$. Zum Vergleich liefert die allgemeine Schranke :eqref:`eq_mdl-nonnormal-power-bound` für normierte Eigenvektoren die Konditionszahl $\kappa_2(\mathbf{V})=80$ und damit $\|\mathbf{M}^6\|_2\le 80\cdot 0.9^6\approx 42.5$. Die Schranke gilt, liegt aber etwa um den Faktor vier über dem tatsächlichen Höchstwert von $10.8$.

<!--en:6a84b1a9-->
![Wachstum von $\mathbf M^k\mathbf e_2$ vor dem Abklingen. Links entsteht bei $k=0$ der Einheitsvektor $\mathbf e_2$ als Differenz zweier Komponenten der Länge etwa 40 entlang der Eigenrichtungen, die sich fast aufheben. Rechts, sechs Schritte später, sind die Komponenten auf das $0.9^6$- bzw. $0.8^6$-Fache abgeklungen. Die Auslöschung lässt nach, und der resultierende Vektor ist mehr als zehnmal so lang. Die vertikale Achse ist etwa 30-fach gestreckt. Bei maßstabsgetreuer Darstellung bilden die Richtungen von $\mathbf v_1$ und $\mathbf v_2$ einen Winkel von nur $1.4^\circ$.](../img/mdl-la-transient-cancel.svg)

<!--en:1e27a491-->
Prüfen wir die Rechnung numerisch und vergleichen wir $\mathbf{M}$ mit der Diagonalmatrix $\boldsymbol{\Lambda}=\operatorname{diag}(0.9,0.8)$. Sie hat genau dieselben Eigenwerte, ist aber normal.

<!--en:aadf0122-->
In der Basis der normierten Eigenvektoren hat $\mathbf{e}_2$ zwei Koeffizienten vom Betrag etwa $40$. Das entspricht den Längen von $40\mathbf{v}_1$ und $\mathbf{v}_2$. Die Spektralnorm $\|\mathbf{M}^k\|_2$ steigt auf $10.8$ bei $k=6$, während $\|\boldsymbol{\Lambda}^k\|_2$ nie über ihren Anfangswert $1$ steigt. Obwohl beide Matrizen dieselben Eigenwerte haben, schrumpfen die Potenzen der normalen Matrix vom ersten Schritt an. Bei der nichtnormalen Matrix wachsen sie auf mehr als das Zehnfache an, bevor sie abklingen. Das linke Teilbild von :numref:`fig_mdl-la-transient-growth` zeigt beide Verläufe in logarithmischer Skala. Nach dem Maximum fallen beide Kurven schließlich mit derselben Steigung $\log 0.9$, genau wie es die Formel von Gelfand garantiert.

<!--en:9b516838-->
![Links sind $\|\mathbf M^k\|_2$ und $\|\boldsymbol\Lambda^k\|_2$ für dieselben Eigenwerte 0.9 und 0.8 dargestellt. Die Norm der Potenzen von $\mathbf M$ steigt bei $k=6$ auf 10.8, bevor beide Kurven mit der Steigung $\log 0.9$ abfallen. Rechts sind die $\varepsilon$-Pseudospektren für $\varepsilon=0.05,0.1,0.2,0.4$ zu sehen. Bei der normalen Matrix $\boldsymbol\Lambda$ sind es Kreisscheiben mit Radius $\varepsilon$ um die Eigenwerte. Bei $\mathbf M$ wölben sie sich weit über den Einheitskreis hinaus.](../img/mdl-la-transient-growth.svg)

<!--en:25473ad4-->
### Pseudospektren

<!--en:57f5db21-->
Dieser Unterabschnitt führt ein Diagnoseverfahren für nichtnormale Matrizen ein. Das Phänomen selbst kennen wir bereits, und Sie können den Unterabschnitt beim ersten Lesen überspringen. Neben transientem Wachstum ist eine extreme Empfindlichkeit der Eigenwerte gegenüber kleinen Störungen ein zweites Anzeichen für Nichtnormalität. Das *$\varepsilon$-Pseudospektrum* macht diese Empfindlichkeit sichtbar. Es umfasst alle komplexen Zahlen, die als Eigenwert von $\mathbf{A}+\mathbf{E}$ für eine Störung $\mathbf{E}$ mit Spektralnorm höchstens $\varepsilon$ auftreten. Dieselbe Menge lässt sich direkt aus der *Resolvente* $(z\mathbf{I}-\mathbf{A})^{-1}$ berechnen:

<!--en:dd534921-->
An den Eigenwerten von $\mathbf{A}$ selbst setzen wir die Norm der Resolvente gleich unendlich. Die Äquivalenz beider Beschreibungen beweisen Sie in Aufgabe :exref:`ex_mdl-eigendecomposition-pseudospectra-transient-growth`. Bei einer normalen Matrix ist die Resolventennorm an der Stelle $z$ der Kehrwert des Abstands von $z$ zum nächstgelegenen Eigenwert. Daher ist $\Lambda_\varepsilon(\mathbf{A})$ die Vereinigung der Kreisscheiben mit Radius $\varepsilon$ um die Eigenwerte. Bei einer nichtnormalen Matrix wie $\mathbf{M}$ kann die Resolventennorm wesentlich größer sein. Prüfen wir das am Punkt $z=1$ auf dem Einheitskreis.

<!--en:d13b722a-->
Für $\boldsymbol{\Lambda}$ beträgt die Resolventennorm $10$. Das ist der Kehrwert des Abstands $0.1$ von $1$ zum Eigenwert $0.9$. Für $\mathbf{M}$ steigt sie auf $200.3$. Schon eine winzige Störung der Größe $1/200.3\approx 0.005$ verschiebt einen Eigenwert von $\mathbf{M}$ auf den Einheitskreis. Bei $\boldsymbol{\Lambda}$ ist dafür eine zwanzigmal so große Störung der Größe $0.1$ nötig (rechtes Teilbild von :numref:`fig_mdl-la-transient-growth`). Rundung in Gleitkommaarithmetik wirkt wie eine kleine Störung der Matrixeinträge. Daher können numerisch berechnete Eigenwerte einer stark nichtnormalen Matrix überraschend ungenau sein. :numref:`sec_mdl-linear-algebra-further-reading` führt diesen Gedanken mit dem Matrixsatz von Kreiss weiter.

<!--en:430b18b7-->
### Produkte von Jacobi-Matrizen

<!--en:0f25ecc4-->
Bisher haben wir einen Vektor wiederholt mit *derselben* Matrix multipliziert. Bei rekurrenten neuronalen Netzen (:numref:`sec_bptt`) entfällt selbst diese Voraussetzung. Eine rekurrente Schicht aktualisiert ihren versteckten Zustand gemäß $\mathbf{h}_t=\phi(\mathbf{W}\mathbf{h}_{t-1}+\cdots)$ und verwendet in jedem Schritt dieselbe Gewichtsmatrix $\mathbf{W}$. Bei der Backpropagation durch die Zeit wird der Gradient jedoch nacheinander mit den transponierten Jacobi-Matrizen der einzelnen Schritte multipliziert. Für $\mathbf{D}_t=\partial\mathbf{h}_t/\partial\mathbf{h}_{t-1}$ gilt:

<!--en:374b355b-->
Da wir die Nichtlinearität $\phi$ in jedem Zeitschritt an einem anderen versteckten Zustand auswerten, unterscheiden sich die Jacobi-Matrizen $\mathbf{D}_1,\ldots,\mathbf{D}_T$. Die Theorie der Potenzen einer einzelnen Matrix greift hier nicht mehr.

<!--en:959ff8d5-->
Tatsächlich sagen die Spektralradien der einzelnen Faktoren allein nichts über ihr Produkt aus. Betrachten wir die beiden $2\times 2$-Matrizen

<!--en:c04b4aa4-->
Sowohl $\mathbf{D}_1$ als auch $\mathbf{D}_2$ sind nilpotent mit $\rho=0$. Die Potenzen beider Matrizen verschwinden daher schon ab dem zweiten Schritt. Ihr Produkt hat jedoch $\rho=4$. Die Matrix $\mathbf{D}_1$ kopiert die zweite Koordinate in die erste und verdoppelt sie dabei. Anschließend kopiert $\mathbf{D}_2$ sie zurück und verdoppelt sie erneut.

<!--en:a8f2515a-->
Bei der Untersuchung iterierter linearer Abbildungen gilt es daher, drei Größen auseinanderzuhalten. Der *Spektralradius* gibt die asymptotische Wachstumsrate der Potenzen einer festen Matrix an. Die *Spektralnorm* liefert eine Schranke für jedes endliche Produkt wechselnder Matrizen, denn aus der Submultiplikativität folgt $\|\mathbf{D}_T\cdots\mathbf{D}_1\|_2\le\prod_t\|\mathbf{D}_t\|_2$ (in unserem Beispiel $4\le 2\cdot 2$). Der *größte Lyapunov-Exponent* schließlich

<!--en:f2ae87c8-->
misst die mittlere logarithmische Wachstumsrate pro Schritt für ein langes Produkt wechselnder Matrizen. Sind die Faktoren unabhängige und identisch verteilte Zufallsmatrizen, so existiert dieser Grenzwert unter schwachen Voraussetzungen mit Wahrscheinlichkeit eins :cite:`Furstenberg.Kesten.1960`.

<!--en:fbe5d67d-->
Durchläuft ein Gradient rückwärts eine Folge von Jacobi-Matrizen mit $\mu>0$, so wächst er exponentiell und explodiert. Gilt $\mu<0$, so klingt er exponentiell ab und verschwindet :cite:`bengio1994learning,Pascanu.Mikolov.Bengio.2013`. Die üblichen Gegenmaßnahmen im Deep Learning setzen an den Normen und Jacobi-Matrizen an. Bei der orthogonalen Initialisierung :cite:`Saxe.McClelland.Ganguli.2014` wählen wir eine orthogonale Gewichtsmatrix. Damit erhält eine lineare Rekurrenz in jedem Schritt die euklidische Länge jedes Vektors. Das ist eine wesentlich stärkere Garantie als $\rho=1$. Gradienten-Clipping begrenzt die Norm des Gradienten direkt nach der Backpropagation. Architekturen mit Gates wie LSTMs und GRUs verändern die Jacobi-Matrizen selbst. So entstehen Pfade, auf denen ausgewählte Richtungen mit Faktoren nahe $1$ weitergegeben werden (:numref:`subsec_bptt-gradient-pathologies`).

<!--en:3f1f985e-->
### Zufallsmatrizen bei der Initialisierung

<!--en:8c5e8c7e-->
Wie groß sind Spektralradius und Spektralnorm der zufälligen Gewichtsmatrizen eines frisch initialisierten neuronalen Netzes? Betrachten wir eine $n\times n$-Matrix $\mathbf{A}$ mit unabhängigen standardnormalverteilten Einträgen. Dieses Modell heißt reelles *Ginibre-Ensemble* :cite:`Ginibre.1965`. Für $n\to\infty$ verteilen sich die mit $1/\sqrt{n}$ skalierten Eigenwerte gleichmäßig über die Einheitskreisscheibe in der komplexen Ebene. Das ist das *Kreisgesetz* (engl. circular law). :citet:`Tao.Vu.2010` bewiesen es allgemein für unabhängige, identisch verteilte Einträge mit Mittelwert $0$ und Varianz $1$. Das Kreisgesetz beschreibt die Lage fast aller Eigenwerte, schließt aber für sich genommen einzelne Ausreißer nicht aus. Ein gesonderter Satz schließt auch Ausreißer aus, die im Grenzwert außerhalb des Einheitskreises bleiben. Damit gilt $\rho(\mathbf{A})/\sqrt{n}\to 1$ mit Wahrscheinlichkeit eins :cite:`Geman.1986`.

<!--en:7a366850-->
Die Spektralnorm verhält sich ganz anders. Es gilt $\|\mathbf{A}\|_2/\sqrt{n}\to 2$ :cite:`Yin.Bai.Krishnaiah.1988`. Das ist ein Spezialfall des Marchenko-Pastur-Gesetzes (:numref:`subsec_mdl-sample-covariance-mp`) für das Seitenverhältnis $c=1$. Der größte Eigenwert von $\tfrac1n\mathbf{A}^\top\mathbf{A}$ konvergiert gegen den oberen Rand $(1+\sqrt{c})^2=4$, und $\|\mathbf{A}\|_2^2$ ist das $n$-Fache dieses Eigenwerts. Schon bei $n=1000$ liefert eine einzelne Ziehung Werte dicht an beiden asymptotischen Grenzwerten.

<!--en:aac88fc8-->
Für große $n$ weicht der Spektralradius nur um wenige Prozent von $\sqrt{n}$ ab, während die größte Streckung nahe bei $2\sqrt{n}$ liegt. Eine gaußsche Zufallsmatrix streckt manche Vektoren also doppelt so stark, wie ihre Eigenwerte vermuten lassen. Gängige Initialisierungsverfahren teilen die zufälligen Gewichte aus einem ganz anderen Grund durch $\sqrt{\textrm{fan-in}}$. Haben die Einträge Varianz $1/n$, so gilt $\mathbb{E}\|\mathbf{A}\mathbf{x}\|^2=\|\mathbf{x}\|^2$ für jeden festen Vektor $\mathbf{x}$. Die quadrierten Längen bleiben also im Mittel erhalten. Ein Spektralradius nahe $1$ ist nur ein Nebeneffekt. Wie wir in :numref:`subsec_mdl-jacobian-products` gesehen haben, bestimmt er das Verhalten eines Produkts verschiedener Matrizen nicht. Überraschenderweise *verkürzt* ein langes Produkt unabhängiger Matrizen mit genau dieser Skalierung einen typischen Vektor. Es gilt also $\mu<0$, obwohl jeder Faktor für sich die quadrierte Länge im Mittel erhält. Der Grund ist die Konkavität des Logarithmus. Für einen Einheitsvektor $\mathbf{x}$ liegt der Erwartungswert von $\log\|\mathbf{A}\mathbf{x}\|^2$ strikt unter dem Logarithmus des Erwartungswerts von $\|\mathbf{A}\mathbf{x}\|^2$ (also unter $\log 1 = 0$). In einem Produkt mit vielen Faktoren addieren sich aber gerade die Logarithmen (Aufgabe :exref:`ex_mdl-eigendecomposition-lyapunov-random-products`).

<!--en:167805fb-->
Die nichtnormale Matrix $\mathbf{M}$ und die gaußsche Zufallsmatrix führen zum selben Schluss. Eigenwerte sagen wenig darüber aus, wie stark eine Matrix Vektoren in endlich vielen Schritten streckt. Für nichtquadratische Matrizen sind Eigenwerte nicht einmal definiert. Um die Streckung für Matrizen beliebiger Form direkt zu messen, wenden wir uns als Nächstes den Singulärwerten aus :numref:`sec_mdl-svd-low-rank` zu.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3e86da7d-->
Gerschgorin-Kreise grenzen die Eigenwerte einer Matrix direkt anhand ihrer Einträge ein, ganz ohne Zerlegung. Die Potenziteration bestimmt das dominante Eigenpaar mit der geometrischen Rate $|\lambda_2/\lambda_1|$. Darauf beruht PageRank. Der Spektralradius $\rho(\mathbf{A})$ bestimmt nur die langfristige Wachstumsrate der Potenzen einer festen Matrix. Bei einer nichtnormalen Matrix können die Potenzen vor dem Abklingen stark anwachsen. Bei wechselnden Matrizen sagen die Eigenwerte der einzelnen Faktoren nichts über ihr Produkt aus. Für Produkte wechselnder Abbildungen liefert die Spektralnorm eine Schranke für endlich viele Schritte, der größte Lyapunov-Exponent die mittlere langfristige Wachstumsrate. Eine gaußsche $n\times n$-Zufallsmatrix bei der Initialisierung verdeutlicht den Unterschied zwischen Eigenwerten und Streckung. Ihr Spektralradius liegt nahe bei $\sqrt{n}$, ihre Spektralnorm dagegen nahe bei $2\sqrt{n}$.

<!--en:2294243a-->
## Übungen

<!--en:360736b9-->
1. **Spektralradius und Normen.** Der Spektralradius liefert eine untere Schranke für die Norm jeder Matrixpotenz. Bei normalen Matrizen ist sie zugleich eine obere Schranke.
    1. Zeigen Sie, dass $\rho(\mathbf{A})\le\|\mathbf{A}^k\|_2^{1/k}$ für jede quadratische Matrix $\mathbf{A}$ und jedes $k\ge1$ gilt.
    1. Die Gleichungskette :eqref:`eq_mdl-normal-power-norm` verwendet zwei Aussagen ohne Beweis. Zeigen Sie, dass $\|\mathbf{U}\mathbf{X}\mathbf{U}^*\|_2=\|\mathbf{X}\|_2$ für jede unitäre Matrix $\mathbf{U}$ und jede quadratische Matrix $\mathbf{X}$ gilt. Zeigen Sie außerdem, dass die Spektralnorm einer Diagonalmatrix der größte Betrag ihrer Diagonaleinträge ist.
    1. Zeigen Sie, dass die Umkehrung von :eqref:`eq_mdl-normal-power-norm` nicht gilt. Finden Sie eine nichtnormale $3\times3$-Matrix, für die $\|\mathbf{A}^k\|_2=\rho(\mathbf{A})^k$ für jedes $k\ge1$ gilt.
    1. Konstruieren Sie zu gegebenem $t>0$ eine diagonalisierbare $2\times2$-Matrix mit Spektralradius $1/2$ und $\|\mathbf{A}\|_2>t$.
   :exlabel:`ex_mdl-eigendecomposition-spectral-radius-norms`

<!--en:97b33b61-->
1. [code] **Konvergenzraten der Potenziteration.** Sei $\mathbf{S}$ die Matrix aus :numref:`fig_mdl-la-power-iter`.
    1. Nach :eqref:`eq_mdl-power-iter` nimmt die Richtungsabweichung der Iterierten von $\mathbf{v}_1$ mit der Rate $|\lambda_2/\lambda_1|=1/3$ ab. Wie viele Iterationen sind nötig, um sie um den Faktor $100$ zu verringern?
    1. Sei $\mathbf{A}$ symmetrisch mit normiertem Eigenvektor $\mathbf{v}_1$ zum Eigenwert $\lambda_1$. Sei ferner $\mathbf{x}=\mathbf{v}_1+\delta\mathbf{u}$ mit einem Einheitsvektor $\mathbf{u}$ orthogonal zu $\mathbf{v}_1$. Zeigen Sie, dass für den Rayleigh-Quotienten aus :numref:`subsec_mdl-rayleigh` die Beziehung $R(\mathbf{x})-\lambda_1=O(\delta^2)$ gilt, ohne linearen Term in $\delta$. Mit welcher Rate konvergiert also der Rayleigh-Quotient der Iterierten gegen $\lambda_1$? Wo geht die Symmetrie von $\mathbf A$ in das Argument ein?
    1. Wiederholen Sie das Experiment aus :numref:`fig_mdl-la-power-iter` mit zufälligem Startvektor für $\left(\begin{smallmatrix}1&0.1\\0.1&1\end{smallmatrix}\right)$. Diese Matrix hat dieselben Eigenvektoren wie $\mathbf S$ und die Eigenwerte 1.1 und 0.9. Plotten Sie den Sinus des Winkels zu $\mathbf v_1$ und den Fehler des Rayleigh-Quotienten für 30 Schritte in logarithmischer Skala. Sagen Sie beide Steigungen und die Zahl der Schritte vorher, die jeder Fehler benötigt, um um den Faktor 100 zu sinken. Erklären Sie, warum die Potenziteration für diese Matrix so viel langsamer konvergiert als für $\mathbf S$.
   :exlabel:`ex_mdl-eigendecomposition-power-iteration-rates`

<!--en:0a32180b-->
1. [code] **Pseudospektren und transientes Wachstum.** ● Sei $\mathbf{M}$ die nichtnormale Matrix aus :eqref:`eq_mdl-nonnormal-example` und $\boldsymbol\Lambda=\operatorname{diag}(0.9,0.8)$ die normale Matrix mit denselben Eigenwerten wie in :numref:`subsec_mdl-transient-growth`.
    1. Beweisen Sie, dass die beiden Mengen in :eqref:`eq_mdl-pseudospectrum` gleich sind. Zeigen Sie zunächst $\|(z\mathbf I-\mathbf A)^{-1}\|_2=1/\sigma_{\min}(z\mathbf I-\mathbf A)$, wobei $\sigma_{\min}(\mathbf X)=\min_{\|\mathbf x\|=1}\|\mathbf X\mathbf x\|$ gilt. Wählen Sie für die eine Inklusion Einheitsvektoren $\mathbf u$ und $\mathbf v$ mit $(z\mathbf{I}-\mathbf{A})\mathbf{v}=\sigma\mathbf{u}$ und $\sigma=\sigma_{\min}(z\mathbf I-\mathbf A)$. Zeigen Sie, dass $z$ ein Eigenwert von $\mathbf{A}+\sigma\mathbf{u}\mathbf{v}^*$ ist. Verwenden Sie für die andere Inklusion die Gleichung $(z\mathbf I-\mathbf A)\mathbf x=\mathbf E\mathbf x$ für einen Eigenvektor $\mathbf x$ von $\mathbf A+\mathbf E$. Lesen Sie die Resolventennormen aus :numref:`subsec_mdl-pseudospectra` ab. Wie groß muss eine Störung sein, damit $z=1$ ein Eigenwert der gestörten Matrix $\mathbf{M}$ wird? Wie groß muss sie bei $\boldsymbol\Lambda$ sein?
    1. Es gelte $\|\mathbf{A}^k\|_2\le K_0$ für alle $k\ge0$. Zeigen Sie, dass für $|z|>1$ die Reihe $\sum_{k\ge0}\mathbf{A}^k/z^{k+1}$ gegen $(z\mathbf{I}-\mathbf{A})^{-1}$ konvergiert. Folgern Sie daraus

       $$
       \sup_{k\ge0}\|\mathbf{A}^k\|_2\;\ge\;K(\mathbf{A})=\sup_{|z|>1}\,(|z|-1)\,\|(z\mathbf{I}-\mathbf{A})^{-1}\|_2 .
       $$

       Die Größe $K(\mathbf{A})$ heißt Kreiss-Konstante. Formulieren Sie die Schranke mithilfe von Pseudospektren. Wenn $\Lambda_\varepsilon(\mathbf{A})$ einen Punkt außerhalb des Einheitskreises im Abstand $\delta$ vom Kreis enthält, wie groß muss dann die Norm mindestens einer Potenz von $\mathbf{A}$ sein?
    1. Berechnen Sie $K(\mathbf{M})$ numerisch auf einem Gitter von Punkten $z$ mit $|z|>1$. Vergleichen Sie das Ergebnis mit dem größten Wert von $\|\mathbf{M}^k\|_2$ aus :numref:`subsec_mdl-transient-growth`. Bestimmen Sie $K(\boldsymbol\Lambda)$ von Hand.
   :exlabel:`ex_mdl-eigendecomposition-pseudospectra-transient-growth`

<!--en:3196912e-->
1. [code] **Dämpfung bei PageRank.** ● Sei $\mathbf{P}$ eine spaltenstochastische $n\times n$-Matrix und $\mathbf{G}=\alpha\mathbf{P}+(1-\alpha)\tfrac1n\mathbf{1}\mathbf{1}^\top$ mit $0<\alpha<1$ die gedämpfte Matrix aus :numref:`subsec_mdl-pagerank`.
    1. Es gelte $\mathbf{P}\mathbf{x}=\lambda\mathbf{x}$. Zeigen Sie, dass $\mathbf{1}^\top\mathbf{x}=0$ für $\lambda\neq1$ gilt und dass $\mathbf{G}\mathbf{x}=\alpha\lambda\mathbf{x}$ aus $\mathbf{1}^\top\mathbf{x}=0$ folgt.
    1. Der Eigenwert $1$ von $\mathbf{P}$ kann mehrfach auftreten, etwa wenn die Links in voneinander getrennte Gruppen zerfallen. Sie dürfen verwenden, dass jede spaltenstochastische Matrix $\mathbf P$ einen nichtnegativen Eigenvektor zum Eigenwert 1 besitzt (eine stationäre Verteilung). Zeigen Sie, dass die Vektoren $\mathbf{x}$ dieses Eigenraums mit $\mathbf{1}^\top\mathbf{x}=0$ einen Unterraum bilden, dessen Dimension um eins kleiner ist als die des Eigenraums. Folgern Sie, dass für diagonalisierbares $\mathbf{P}$ die Eigenwerte von $\mathbf{G}$ durch $1$ und $\alpha\lambda_2(\mathbf{P}),\ldots,\alpha\lambda_n(\mathbf{P})$ gegeben sind. Daraus ergibt sich die Schranke $|\lambda_2(\mathbf{G})|\le\alpha$ von :citet:`Haveliwala.Kamvar.2003`.
    1. Vier Seiten sind in einem Zyklus verlinkt, $1\to2\to3\to4\to1$. Bestimmen Sie die Eigenwerte ihrer Linkmatrix $\mathbf{P}$. Warum konvergiert die ungedämpfte Potenziteration nie, wenn sie auf Seite $1$ startet?
    1. Verwenden Sie $\alpha=0.85$ und den Zyklus aus (c). Führen Sie die Potenziteration für $\mathbf{G}$ mit Start auf Seite $1$ aus. Zählen Sie die Iterationen, bis der $\ell_1$-Abstand zur stationären Verteilung unter $10^{-8}$ fällt. Vergleichen Sie mit der Vorhersage $0.85^k\le10^{-8}$ aus (b).
    1. Prüfen Sie am Beispiel der drei Seiten aus :numref:`subsec_mdl-pagerank` numerisch, dass die Eigenwerte von $\mathbf G$ bis auf den Eigenwert 1 das $\alpha$-Fache der Eigenwerte von $\mathbf P$ betragen. Prüfen Sie außerdem, dass sich die stationäre Verteilung von $(0.3,0.4,0.3)$ auf etwa $(0.303,0.394,0.303)$ verschiebt. Warum verschiebt die Dämpfung sie in Richtung der Gleichverteilung?
   :exlabel:`ex_mdl-eigendecomposition-pagerank-damping`

<!--en:693ae722-->
1. [code] **Lyapunov-Exponenten zufälliger Produkte.** ●● Seien $\mathbf{D}_1,\mathbf{D}_2,\ldots$ unabhängige $n\times n$-Matrizen mit unabhängigen $\mathcal{N}(0,1/n)$-verteilten Einträgen. Sie dienen als lineares Modell für die Jacobi-Matrizen eines tiefen Netzes bei Fan-in-Initialisierung. Sei $\mathbf{x}_0$ ein Einheitsvektor. Nach :numref:`subsec_mdl-random-matrices` hat jeder Faktor einen Spektralradius nahe $1$ für große $n$. In :numref:`subsec_mdl-jacobian-products` beschreiben wir das Wachstum des Produkts durch seinen größten Lyapunov-Exponenten

   $$
   \mu=\lim_{T\to\infty}\frac1T\log\|\mathbf{D}_T\cdots\mathbf{D}_1\mathbf{x}_0\|,
   $$

   Dieser existiert nach dem Satz von Furstenberg und Kesten mit Wahrscheinlichkeit eins.
    1. Zeigen Sie, dass $\mathbb{E}\|\mathbf{D}\mathbf{x}\|^2=\|\mathbf{x}\|^2$ für jeden festen Vektor $\mathbf{x}$ gilt. Diese Skalierung erhält also die quadrierten Normen im Mittel.
    1. Setzen Sie $\mathbf{u}_0=\mathbf{x}_0$ und $\mathbf{u}_t=\mathbf{x}_t/\|\mathbf{x}_t\|$ für die Iterierten $\mathbf{x}_t=\mathbf{D}_t\cdots\mathbf{D}_1\mathbf{x}_0$. Welche Verteilung hat bei gegebenen $\mathbf{D}_1,\ldots,\mathbf{D}_{t-1}$ die Größe $n\|\mathbf{D}_t\mathbf{u}_{t-1}\|^2$? Stellen Sie mit Ihrer Antwort $\log\|\mathbf{D}_T\cdots\mathbf{D}_1\mathbf{x}_0\|$ als Summe unabhängiger Terme dar. Verwenden Sie $\mathbb{E}\log\chi^2_n=\psi(n/2)+\log2$, wobei $\psi$ die Digammafunktion bezeichnet, und leiten Sie die geschlossene Formel von :citet:`Cohen.Newman.1984` her:

       $$
       \mu=\tfrac12\bigl(\psi(n/2)+\log2-\log n\bigr).
       $$

    1. Zeigen Sie, dass trotz (a) die Ungleichung $\mu<0$ für jedes $n$ gilt. Berechnen Sie $\mu$ in geschlossener Form für $n=1$. Verwenden Sie dazu $\psi(1/2)=-\gamma_{\textrm{E}}-2\log2$ mit der Euler-Konstanten $\gamma_{\textrm{E}}\approx0.5772$. Zeigen Sie mithilfe von $\psi(x)=\log x-\tfrac1{2x}+O(x^{-2})$, dass $\mu\approx-1/(2n)$ für große $n$ gilt.
    1. Simulieren Sie $T=10^4$ Schritte mit erneuter Normierung nach jedem Schritt für $n\in\{1,2,5,50\}$. Vergleichen Sie den gemessenen Exponenten mit (b) sowie mit den Logarithmen des Spektralradius und des größten Singulärwerts einzelner Faktoren. Um welchen Faktor schrumpft eine typische Eingabe bei dieser Initialisierung in einem linearen Netz der Breite $50$ und Tiefe $100$? Was sagt der Vergleich über die drei Größen aus :numref:`subsec_mdl-jacobian-products` aus?
   :exlabel:`ex_mdl-eigendecomposition-lyapunov-random-products`

<!--en:fb71d175-->
[Diskussionen](https://d2l.discourse.group/t/36876)

<!--en:6a644630-->
Spektren in der Praxis<br>**Gerschgorin-Kreise · Potenziteration und PageRank · transientes Wachstum bei nichtnormalen Matrizen · Produkte von Jacobi-Matrizen · Zufallsmatrizen**

<!--en:363792c5-->
::: {.slide title="Matrizen mit Milliarden Zeilen und trügerische Eigenwerte"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:02f75663-->
- **Zerlegung mit Aufwand $O(n^3)$ zu teuer:** Bei den Größenordnungen des Webs oder eines LLM sind nur Matrix-Vektor-Produkte $\mathbf{A}\mathbf{v}$ bezahlbar.

<!--en:faea6a4b-->
- **Transientes Wachstum bei endlich vielen Schritten:** Selbst wenn für alle Eigenwerte $|\lambda_i| < 1$ gilt, können die Normen $\|\mathbf{M}^k\|_2$ nichtnormaler Matrixpotenzen zunächst auf das Zehnfache anwachsen.

<!--en:5518b10c-->
- **Wechselnde Jacobi-Matrizen bei BPTT:** Die rekurrente Backpropagation multipliziert $T$ verschiedene Matrizen $\mathbf{D}_1^\top\cdots\mathbf{D}_T^\top$. Die einzelnen Eigenwerte liefern keine Garantie.

<!--en:75657658-->
::: {.slide title="Eingrenzung der Eigenwerte durch Gerschgorin-Kreise aus den Matrixeinträgen"}

<!--en:f76e120b-->
[Eingrenzung der Eigenwerte]{.kicker}

<!--en:debe8bad-->
- **Kreisscheiben aus Zeilensummen:** Zu jeder Zeile $i$ von $\mathbf{A}$ gehört die Kreisscheibe $\mathcal{D}_i$ mit Mittelpunkt $a_{ii}$ und Radius $r_i = \sum_{j \neq i} |a_{ij}|$.

<!--en:3b253ada-->
- **Beweis in drei Zeilen:** Gelten $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ und $|v_i| = \max_j |v_j| > 0$, so ergibt Zeile $i$ die Gleichung $(\lambda - a_{ii})v_i = \sum_{j \neq i} a_{ij}v_j$. Daraus folgt
  $$|\lambda - a_{ii}| \le \sum_{j \neq i} |a_{ij}|\,\frac{|v_j|}{|v_i|} \le r_i$$

<!--en:4068d52a-->
- **Zusammenhangskomponenten und Diagonaldominanz:** Eine von den übrigen isolierte Gruppe aus $m$ Kreisscheiben enthält $m$ Eigenwerte. Gilt $a_{ii} > r_i$ für alle $i$, so folgt $0 \notin \bigcup_i \mathcal{D}_i$. Damit ist $\mathbf{A}$ invertierbar und im symmetrischen Fall positiv definit.

<!--en:6b25c62d-->
::: {.slide title="Gerschgorin-Kreise am Beispiel einer 3×3-Matrix"}

<!--en:f76e120b-2-->
[Eingrenzung der Eigenwerte]{.kicker}

<!--en:2d7278cd-->
Für $\mathbf{A} = \left(\begin{smallmatrix}4&1&0\\-1&4&0.5\\0.5&0&1\end{smallmatrix}\right)$ enthalten die beiden überlappenden Kreisscheiben um $4$ das komplexe Paar $3.99 \pm 0.96i$. Die isolierte Kreisscheibe um $1$ enthält $1.03$:

<!--en:21d294f6-->
**Nahezu scharfe Schranke:** Die komplexen Eigenwerte liegen im Abstand $0.96$ vom Mittelpunkt $4$, fast am Rand der ersten Kreisscheibe mit Radius $1$.

<!--en:94379014-->
::: {.slide title="Das dominante Eigenpaar durch Potenziteration"}

<!--en:69eb0e75-->
[Eigenwerte großer Matrizen]{.kicker}

<!--en:6d55e295-->
- **Wiederholte Multiplikation und Normierung:** Gilt $|\lambda_1| > |\lambda_2| \ge \cdots \ge |\lambda_n|$ und hat $\mathbf{v}_0 = \sum_i c_i\mathbf{v}_i$ einen Koeffizienten $c_1 \neq 0$, so folgt durch Ausklammern von $\lambda_1^k$
  $$\mathbf{A}^k\mathbf{v}_0 = \lambda_1^k\Bigl(c_1\mathbf{v}_1 + \sum_{i \ge 2} c_i\Bigl(\frac{\lambda_i}{\lambda_1}\Bigr)^k \mathbf{v}_i\Bigr)$$

<!--en:905f40ba-->
- **Lineare Konvergenz der Richtung ($|\lambda_2/\lambda_1|^k$):** Durch die Normierung $\mathbf{v}_{k+1} = \mathbf{A}\mathbf{v}_k / \|\mathbf{A}\mathbf{v}_k\|$ nähert sich die Richtung von $\mathbf{v}_k$ der von $\mathbf{v}_1$ mit der geometrischen Rate $|\lambda_2/\lambda_1|$ ($1/3$ pro Schritt bei $\mathbf{S} = \left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$).

<!--en:9d267347-->
- **Quadrierte Rate des Rayleigh-Quotienten ($|\lambda_2/\lambda_1|^{2k}$):** Bei symmetrischem $\mathbf{A}$ bewirken orthogonale Eigenvektoren, dass ein Winkelfehler $\delta$ den Quotienten $R(\mathbf{v}_k)$ nur um $O(\delta^2)$ verschiebt ($1/9$ pro Schritt bei $\mathbf{S}$).

<!--en:d9168e09-->
::: {.slide title="Potenziteration im Code und ihre Grenzen"}

<!--en:9a5869a5-->
[Potenziteration]{.kicker}

<!--en:0617b8d9-->
Bei einer positiven $5 \times 5$-Matrix liefern 200 Schritte der Potenziteration den Betrag $|\lambda_1|$ auf 10 Nachkommastellen genau. Bei einem dominanten komplex konjugierten Paar konvergiert die Richtung dagegen nicht:

<!--en:2c28ae53-->
**Positive und gaußsche Matrizen:** Unter $10{,}000$ zufälligen $5 \times 5$-Matrizen mit gleichverteilten positiven Einträgen haben $100\%$ einen reellen, streng dominanten Eigenwert (Betragsverhältnis $0.14$–$0.37$). Bei standardnormalverteilten Einträgen haben $\sim 40\%$ ein dominantes komplexes Paar, sodass der Normquotient dauerhaft oszilliert.

<!--en:27aac536-->
::: {.slide title="PageRank und der Satz von Perron-Frobenius"}

<!--en:4c92166b-->
[Eigenvektoren für das gesamte Web]{.kicker}

<!--en:e579a3e5-->
- **Perron-Frobenius:** Jede strikt positive Matrix hat einen eindeutigen reellen, streng dominanten Eigenwert $\lambda_1 = \rho(\mathbf{A})$ mit positivem Eigenvektor.

<!--en:00a3ec43-->
- **Gedämpfte Google-Matrix ($\alpha = 0.85$):** Die Mischung der spaltenstochastischen Matrix $\mathbf{P}$ mit gleichverteilter Teleportation ergibt $\mathbf{G} = \alpha\mathbf{P} + (1-\alpha)\tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$. Sie ermöglicht Sprünge aus Sackgassen und Zyklen und garantiert $|\lambda_2(\mathbf{G})| \le 0.85$.

<!--en:a0b7fd87-->
- **Durchlauf mit dünn besetzter Matrix:** Auswertung von $\mathbf{G}\mathbf{p} = \alpha\mathbf{P}\mathbf{p} + \tfrac{1-\alpha}{n}\mathbf{1}$ in $O(\text{edges})$

<!--en:a9f1a7cf-->
::: {.slide title="Der Spektralradius bestimmt die asymptotische Rate"}

<!--en:9248b6ab-->
[Iteration und Stabilität]{.kicker}

<!--en:4888cdff-->
- **Formel von Gelfand:** $\rho(\mathbf{A}) = \max_i |\lambda_i| = \lim_{k\to\infty} \|\mathbf{A}^k\|^{1/k}$ bestimmt nur die asymptotische Rate für $k \to \infty$.

<!--en:235ccd88-->
- **Grenzen von $\rho \le 1$ bei endlichem $k$:** Für die Scherung $\mathbf{T}$ gilt $\rho = 1$ und dennoch $\|\mathbf{T}^k\|_2 \sim k \to \infty$. Für $\mathbf{J}_2(0.99)$ gilt $\rho = 0.99 < 1$, doch die Norm ihrer Potenzen steigt auf $37$ bei $k = 100$.

<!--en:614a4031-->
::: {.slide title="Normale Matrizen und Verstärkung bei nichtnormalen Matrizen"}

<!--en:77049e3b-->
[Normale und nichtnormale Abbildungen]{.kicker}

<!--en:1f9bb4bd-->
- **Normal ($\mathbf{A}^\top\mathbf{A} = \mathbf{A}\mathbf{A}^\top$):** Eine unitäre Eigenbasis in $\mathbf{A} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^*$ garantiert $\|\mathbf{A}^k\|_2 = \rho(\mathbf{A})^k$ in jedem Schritt $k$.

<!--en:9f12ec49-->
- **Schranke für nichtnormale Matrizen:** Bei nichtorthogonaler Eigenbasis gilt $\|\mathbf{A}^k\|_2 \le \kappa_2(\mathbf{V})\rho(\mathbf{A})^k$.

<!--en:35e37ed0-->
- **Nachlassende Auslöschung:** Bei fast parallelen Eigenvektoren ($\kappa_2(\mathbf{V}) \gg 1$) heben sich sehr große Komponenten bei $k = 0$ fast auf. Bei endlichem $k$ lässt diese Auslöschung nach.

<!--en:6118cb1f-->
::: {.slide title="Transientes Wachstum durch fast parallele Eigenvektoren"}

<!--en:dbba610a-->
[Nichtnormale Dynamik]{.kicker}

<!--en:ee57fef8-->
- **Verschiedene Eigenwerte im Einheitskreis:** $\mathbf{M} = \left(\begin{smallmatrix}0.9&4\\0&0.8\end{smallmatrix}\right)$ hat die Eigenwerte $\lambda_1 = 0.9$ und $\lambda_2 = 0.8$. Die Richtungen der Eigenvektoren $\mathbf{v}_1 = (1, 0)$ und $\mathbf{v}_2 = (-40, 1)$ bilden einen spitzen Winkel von nur $1.4^\circ$.

<!--en:8e39f269-->
- **Nachlassende Auslöschung:** Aus der Darstellung $\mathbf{e}_2 = (0, 1) = 40\mathbf{v}_1 + \mathbf{v}_2$ folgt
  $$\mathbf{M}^k\mathbf{e}_2 = \begin{bmatrix}40(0.9^k - 0.8^k) \\ 0.8^k\end{bmatrix}$$

<!--en:47f4a7ef-->
- **Mehr als zehnfache Verstärkung bei $k = 6$:** Da $0.8^k$ schneller abklingt als $0.9^k$, wächst die Differenz zunächst. Die Norm erreicht mit $\|\mathbf{M}^6\|_2 \approx 10.8$ ihren Höchstwert und sinkt danach.

<!--en:8a6d6298-->
::: {.slide title="Nichtnormales M und normales Λ im Codevergleich"}

<!--en:dbba610a-2-->
[Nichtnormale Dynamik]{.kicker}

<!--en:5172015d-->
Obwohl $\mathbf{M} = \left(\begin{smallmatrix}0.9&4\\0&0.8\end{smallmatrix}\right)$ und $\boldsymbol{\Lambda} = \operatorname{diag}(0.9, 0.8)$ genau dieselben Eigenwerte haben, nimmt die Norm von $\boldsymbol{\Lambda}^k$ ab Schritt 1 monoton ab, während die Norm von $\mathbf{M}^k$ auf das Zehnfache anwächst:

<!--en:5b4b919d-->
::: {.slide title="Pseudospektren und Empfindlichkeit der Eigenwerte"}

<!--en:f3abd0d1-->
[Diagnose der Nichtnormalität]{.kicker}

<!--en:39d21b5e-->
- **$\varepsilon$-Pseudospektrum $\Lambda_\varepsilon(\mathbf{A})$:** Menge aller $z \in \mathbb{C}$, die als Eigenwert von $\mathbf{A} + \mathbf{E}$ für eine Störung mit $\|\mathbf{E}\|_2 \le \varepsilon$ auftreten. Gleichwertig ist die Bedingung an die Resolventennorm
  $$\|(z\mathbf{I} - \mathbf{A})^{-1}\|_2 \ge \frac{1}{\varepsilon}$$

<!--en:cb394462-->
- **Kreisscheiben im normalen Fall, Ausbuchtung im nichtnormalen Fall:** Bei $z = 1$ gilt $\|(\mathbf{I} - \boldsymbol{\Lambda})^{-1}\|_2 = 1/(1 - 0.9) = 10$, aber $\|(\mathbf{I} - \mathbf{M})^{-1}\|_2 = 200.3$. Schon eine Störung der Größe $\varepsilon \approx 0.005$ verschiebt einen Eigenwert von $\mathbf{M}$ auf den Einheitskreis.

<!--en:0c305e28-->
::: {.slide title="Produkte wechselnder Jacobi-Matrizen in rekurrenten Netzen"}

<!--en:6e8ced2d-->
[Backpropagation durch die Zeit]{.kicker}

<!--en:c6bc1682-->
- **Wechselnde Jacobi-Matrizen bei BPTT:** Die Backpropagation multipliziert die $T$ verschiedenen Matrizen der einzelnen Zeitschritte gemäß $\nabla_{\mathbf{h}_0}\mathcal{L} = \mathbf{D}_1^\top\mathbf{D}_2^\top\cdots\mathbf{D}_T^\top\nabla_{\mathbf{h}_T}\mathcal{L}$.

<!--en:51a52ce8-->
- **Wachstum trotz Spektralradius null:** Die nilpotenten Matrizen $\mathbf{D}_1 = \left(\begin{smallmatrix}0&2\\0&0\end{smallmatrix}\right)$ und $\mathbf{D}_2 = \left(\begin{smallmatrix}0&0\\2&0\end{smallmatrix}\right)$ haben beide $\rho = 0$. Ihr Produkt $\mathbf{D}_2\mathbf{D}_1 = \operatorname{diag}(0, 4)$ hat jedoch $\rho = 4$, und bei wiederholter Anwendung wächst die Norm wie $4^{T/2}$.

<!--en:2a03068a-->
::: {.slide title="Drei Größen für iterierte lineare Abbildungen"}

<!--en:b054dd38-->
[Stabilität]{.kicker}

<!--en:0dd4b945-->
- **Spektralradius $\rho(\mathbf{A})$:** Asymptotische Wachstumsrate für $k \to \infty$ bei Potenzen *einer festen* Matrix

<!--en:f330e47e-->
- **Spektralnorm $\|\mathbf{A}\|_2 = \sigma_1$:** Maximale Streckung in einem Schritt und Schranke für endliche Produkte gemäß $\|\mathbf{D}_T\cdots\mathbf{D}_1\|_2 \le \prod_t \|\mathbf{D}_t\|_2$

<!--en:362b268a-->
- **Größter Lyapunov-Exponent $\mu$:** Langfristige exponentielle Wachstumsrate $\lim_{T\to\infty}\frac{1}{T}\log\|\mathbf{D}_T\cdots\mathbf{D}_1\mathbf{x}_0\|$ für Produkte wechselnder Matrizen

<!--en:5fd65042-->
::: {.slide title="Zufällige Gewichtsmatrizen bei der Initialisierung"}

<!--en:35911c1c-->
[Theorie der Zufallsmatrizen]{.kicker}

<!--en:40ce8d1d-->
- **Kreisgesetz und $2\times$ so große Streckung:** Eine gaußsche $n \times n$-Matrix hat $\rho(\mathbf{A}) \approx \sqrt{n}$, ihre maximale Streckung beträgt jedoch $\|\mathbf{A}\|_2 \approx 2\sqrt{n}$.

<!--en:bcc5f510-->
- **Paradoxon langer Produkte ($\mu < 0$):** Die Fan-in-Skalierung $\mathcal{N}(0, 1/n)$ erhält die quadrierte Norm im Mittel gemäß $\mathbb{E}\|\mathbf{D}\mathbf{x}\|^2 = \|\mathbf{x}\|^2$. Wegen der Konkavität von $\log$ verkürzen lange Produkte typische Vektoren mit der Rate $\mu \approx -1/(2n)$.

<!--en:7f1e7b86-->
::: {.slide title="Singulärwerte als nächster Schritt im Deep Learning"}

<!--en:fc9c4e1c-->
[Übergang zur Singulärwertzerlegung]{.kicker}

<!--en:5718d64b-->
- **Vier Grenzen der Eigenwertanalyse:** Nichtquadratische Schichten ($m \neq n$), defekte Schermatrizen, transientes Wachstum bei nichtnormalen Matrizen und die $2\times$ so große Streckung bei gaußschen Zufallsgewichten

<!--en:1a3806a6-->
- **Singulärwertzerlegung (§{sec_mdl-svd-low-rank}):** Mit getrennten Orthonormalbasen für Eingabe ($\mathbf{V}$) und Ausgabe ($\mathbf{U}$) lässt sich *jede* Matrix durch nichtnegative Streckfaktoren $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ darstellen.

<!--en:c1b2f971-->
::: {.slide title="Verständnisfragen zu Kreisscheiben, Potenziteration und Stabilität"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:548ea2c8-->
1. **Invertierbarkeit nach Gerschgorin:** Anhand der Gerschgorin-Kreise erklären, warum $\mathbf{A} = \left(\begin{smallmatrix}5&-1&1\\2&6&-1\\0&1&4\end{smallmatrix}\right)$ invertierbar sein muss. Eine Berechnung der Determinante ist dafür überflüssig.

<!--en:492f2992-->
2. **Rate der Potenziteration:** Gegeben sei eine symmetrische Matrix $\mathbf{S}$ mit Eigenwerten $\lambda_1 = 3$ und $\lambda_2 = 1$. Um welchen Faktor sinkt der Fehler $3 - R(\mathbf{v}_k)$ des Rayleigh-Quotienten pro Schritt der Potenziteration?

<!--en:a6dbd879-->
3. **Spektralradius und Norm:** Garantiert $\rho(\mathbf{A}) \le 1$, dass $\|\mathbf{A}^k\|_2$ für alle $k \ge 1$ beschränkt bleibt? Ein $2 \times 2$-Gegenbeispiel angeben.

<!--en:32f0878e-->
**Lösungen:** **1.** Die Kreisscheiben der drei Zeilen haben die Mittelpunkte $5, 6, 4$ und die Radien $2, 3, 1$. Da in jeder Zeile $a_{ii} > r_i$ gilt, liegen alle Kreisscheiben ganz in der rechten Halbebene, hier sogar im Bereich $\operatorname{Re}(z) \ge 3 > 0$. Also kann $0$ kein Eigenwert sein, und $\mathbf{A}$ ist invertierbar. **2.** Der Richtungsfehler sinkt pro Schritt mit dem Faktor $|\lambda_2/\lambda_1| = 1/3$, der Fehler des Rayleigh-Quotienten mit dem quadrierten Faktor $(1/3)^2 = 1/9$. **3.** Nein. Für die defekte Schermatrix $\mathbf{T} = \left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ gilt $\rho(\mathbf{T}) = 1$, doch ihre Potenzen $\mathbf{T}^k = \left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$ erfüllen $\|\mathbf{T}^k\|_2 \sim k \to \infty$.

<!--en:00ace665-->
::: {.slide title="Zusammenfassung zu Spektren in der Praxis"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:e125323f-->
- **Eigenwerte eingrenzen und bestimmen:** Gerschgorin-Kreise grenzen Spektren mit Aufwand $O(n^2)$ ein. Die Potenziteration konvergiert mit der Rate $|\lambda_2/\lambda_1|^k$, und die Dämpfung bei PageRank garantiert $|\lambda_2(\mathbf{G})| \le 0.85$.

<!--en:abed2638-->
- **Stabilität jenseits der Eigenwerte:** Für die Potenzen diagonalisierbarer nichtnormaler Matrizen gilt die obere Schranke $\kappa_2(\mathbf{V})\rho(\mathbf{A})^k$. Bei wechselnden Jacobi-Matrizen in BPTT spielen Spektralnormen und Gates eine Rolle. Gaußsche Matrizen strecken um bis zu etwa $2\sqrt{n}$.
