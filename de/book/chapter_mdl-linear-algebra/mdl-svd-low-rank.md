<!--en:7def598f-->
# Singulärwertzerlegung und niedrigrangige Approximation

<!--en:215fd002-->
In :numref:`sec_mdl-eigendecompositions` haben wir gesehen, wie die Eigenwertzerlegung eine quadratische Matrix anhand der Richtungen beschreibt, die sie lediglich streckt. Potenzen der Matrix lassen sich dadurch auf Potenzen einzelner Zahlen zurückführen. Die meisten Matrizen im Deep Learning haben jedoch eine andere Gestalt. Eine Gewichtsmatrix bildet Eingaben einer Breite auf Ausgaben einer anderen Breite ab. Eine Datenmatrix enthält eine Zeile pro Beispiel und eine Spalte pro Merkmal. Keine der beiden wenden wir wiederholt auf ihre eigene Ausgabe an, und keine muss quadratisch sein. Selbst wenn eine Matrix quadratisch *ist*, helfen ihre Eigenwerte mitunter wenig. Die Scherung $\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ ist *defekt*. Sie hat nur eine Eigenvektorrichtung, obwohl wir für die Ebene zwei brauchen. In :numref:`sec_mdl-spectra-in-practice` haben wir eine Matrix kennengelernt, deren sechste Potenz einen Vektor auf das Zehnfache streckt, obwohl ihre beiden Eigenwerte kleiner als 1 sind. Wir suchen eine Zerlegung, die für *jede* Matrix existiert, die Streckung direkt misst und zeigt, wie wenige Zahlen für eine gute Approximation einer großen Matrix genügen.

<!--en:20b036c5-->
An einem Foto lässt sich die letzte Anforderung nachvollziehen. Ein Graustufenbild einer Katze mit $400\times500$ Pixeln ist eine Matrix aus 200 000 Zahlen. Wie jede Matrix lässt sie sich als Summe von Matrizen vom Rang eins schreiben. Jede davon ist das äußere Produkt einer Spalte und einer Zeile. Schon zwanzig geschickt gewählte Summanden (18 020 Zahlen oder 9% der ursprünglichen Daten) geben das Foto mit einem relativen Fehler von nur 8% wieder (:numref:`fig_mdl-la-eckart-young`). Welche zwanzig Summanden leisten das? Könnten andere zwanzig ein besseres Ergebnis liefern? Diese Frage begegnet uns im gesamten Deep Learning. Ihre Antwort zeigt, wie viele Parameter ein Fine-Tuning-Adapter für ein großes Modell braucht und welche Richtungen in einem Datensatz es sich zu erhalten lohnt.

<!--en:ed208837-->
Die entscheidende Idee besteht darin, für Eingaben und Ausgaben verschiedene Basen zuzulassen. Wählen wir je eine Orthonormalbasis für den Eingaberaum und den Ausgaberaum, so reduziert sich *jede* Matrix auf eine Liste unabhängiger Streckungen mit nichtnegativen Faktoren $\sigma_1\ge\sigma_2\ge\cdots\ge0$. Daraus lassen sich der Rang einer Matrix und ihre beste niedrigrangige Approximation (engl. low-rank approximation) ablesen. Dieselbe Liste liefert die Hauptkomponentenanalyse (engl. principal component analysis, PCA) und Lösungen nach der Methode der kleinsten Quadrate. Sie zeigt auch, wie empfindlich ein lineares Gleichungssystem auf Messfehler reagiert. Aus :numref:`sec_linear-algebra` kennen wir zwei Maße für die Größe einer Matrix. Die Frobeniusnorm $\|\mathbf{A}\|_F$ ist die Quadratwurzel aus der Summe der quadrierten Einträge. Die Spektralnorm $\|\mathbf{A}\|_2$ ist der größte Faktor, um den $\mathbf{A}$ einen Einheitsvektor strecken kann. In diesem Abschnitt ergeben sich beide Normen als einfache Funktionen der Singulärwerte $\sigma_i$.

<!--en:e200037c-->
Zwei Ergebnisse stehen im Mittelpunkt. Die Singulärwertzerlegung (engl. singular value decomposition, SVD) existiert für jede Matrix. Der Satz von Eckart und Young garantiert, dass wir durch Abschneiden der Zerlegung für jeden Rang die beste Approximation erhalten. Daraus ergeben sich die Pseudoinverse und die Konditionszahl in wenigen Zeilen. Der letzte Unterabschnitt stellt die Verbindung zum modernen Deep Learning her. Sie können ihn beim ersten Lesen überspringen. Für alle folgenden numerischen Experimente verwenden wir NumPy. Soweit nicht anders angegeben, bezeichnet $\mathbf{T}$ die Schermatrix und $\mathbf{A}$ eine allgemeine Matrix.

<!--en:606d4afb-->
## Die Singulärwertzerlegung

<!--en:77b308ce-->
### Die Zerlegung und ihre Geometrie

<!--en:cbeef8ae-->
In :numref:`sec_linear-algebra` haben wir gesehen, dass die symmetrische Matrix $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ den Einheitskreis auf eine Ellipse abbildet. Ihre Halbachsen liegen in Richtung der Eigenvektoren $(1,1)$ und $(-1,1)$ und haben die Längen 3 und 1. Auch eine *nicht symmetrische* Matrix bildet den Einheitskreis auf eine Ellipse ab (oder, wenn sie singulär ist, auf eine Strecke), wie wir gleich beweisen. Der Unterschied liegt in den orthogonalen Eingaberichtungen, deren Bilder auf den Ellipsenachsen liegen. Diese Richtungen stimmen im Allgemeinen nicht mehr mit den Achsen selbst überein. Es gibt zwei Orthonormalbasen, eine für den Eingaberaum und eine für den Ausgaberaum. Die Singulärwertzerlegung gibt beide an.

<!--en:58ba3010-->
**Satz (Singulärwertzerlegung).** *Jede reelle $m\times n$-Matrix $\mathbf{A}$ lässt sich schreiben als*

<!--en:065d86f6-->
*Dabei sind $\mathbf{U}\in\mathbb{R}^{m\times m}$ und $\mathbf{V}\in\mathbb{R}^{n\times n}$ orthogonal ($\mathbf{U}^\top\mathbf{U}=\mathbf{I}$, $\mathbf{V}^\top\mathbf{V}=\mathbf{I}$). Alle Einträge von $\boldsymbol{\Sigma}\in\mathbb{R}^{m\times n}$ sind null, mit Ausnahme der nichtnegativen Einträge $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_{\min(m,n)}\ge0$ auf der Hauptdiagonalen.* Die Spalten $\mathbf{v}_i$ von $\mathbf{V}$ heißen *rechte Singulärvektoren*, die Spalten $\mathbf{u}_i$ von $\mathbf{U}$ heißen *linke Singulärvektoren* und die Diagonaleinträge $\sigma_i$ heißen *Singulärwerte*.

<!--en:5901abc9-->
Geometrisch beschreibt *jede* Matrix unabhängig von ihrer Form eine orthogonale Transformation, gefolgt von einer Streckung entlang der Koordinatenachsen und einer zweiten orthogonalen Transformation. Schreiben wir $\mathbf{A}\mathbf{V}=\mathbf{U}\boldsymbol{\Sigma}$ und lesen Spalte $i$ ab, erkennen wir die Wirkung auf jede einzelne Richtung:

<!--en:c94a78ec-->
Jede Eingaberichtung $\mathbf{v}_i$ geht in die zugehörige Ausgaberichtung $\mathbf{u}_i$ über, mit dem Streckfaktor $\sigma_i$. Für $n>m$ fallen die zusätzlichen Eingaberichtungen $\mathbf{v}_{m+1},\ldots,\mathbf{v}_n$ auf den Nullvektor. Lesen wir :eqref:`eq_mdl-svd` von rechts nach links, so zerfällt die Anwendung von $\mathbf{A}$ auf einen Vektor $\mathbf{x}$ in drei Schritte:

<!--en:63fa7c68-->
1. $\mathbf{V}^\top$ dreht die Eingabe so, dass jeder rechte Singulärvektor $\mathbf{v}_i$ auf die Koordinatenachse $\mathbf{e}_i$ fällt.

<!--en:5db33b8b-->
2. $\boldsymbol{\Sigma}$ streckt die Achse $i$ um den Faktor $\sigma_i$. Für $m\ne n$ ergänzt die Abbildung Nullkoordinaten oder lässt überzählige Koordinaten weg, bis die Ausgabedimension stimmt.

<!--en:82aeabb5-->
3. $\mathbf{U}$ dreht die gestreckten Achsen so, dass die Achse $i$ in Richtung $\mathbf{u}_i$ zeigt.

<!--en:5d3fac5f-->
Mit „Drehung“ bezeichnen wir hier kurz eine orthogonale Abbildung, die auch eine Spiegelung enthalten kann.

<!--en:ab918194-->
:numref:`fig_mdl-la-svd-action` veranschaulicht die drei Schritte für die Scherung $\mathbf{T}$. Ihre Singulärwerte sind $\sigma_1=1.618$ und $\sigma_2=0.618$. Der erste rechte Singulärvektor $\mathbf{v}_1$ bildet einen Winkel von $58.3^\circ$ mit der positiven $x$-Achse, seine Bildrichtung $\mathbf{u}_1$ einen Winkel von $31.7^\circ$. Da sich Eingabe- und Ausgabebasis unterscheiden, gilt $\mathbf{U}\ne\mathbf{V}$. Diese Werte leiten wir in :numref:`subsec_mdl-svd-via-ata` her.

<!--en:a0bb0486-->
![Wirkung der Scherung $\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ auf den Einheitskreis, von links nach rechts. $\mathbf{V}^\top$ dreht um $-58.3^\circ$, $\boldsymbol{\Sigma}$ streckt um den Faktor $\varphi=1.618$ entlang der $x$-Achse und um $1/\varphi=0.618$ entlang der $y$-Achse. Anschließend dreht $\mathbf{U}$ um $31.7^\circ$. Die Eingabebasis $\{\mathbf{v}_i\}$ und die Ausgabebasis $\{\mathbf{u}_i\}$ unterscheiden sich.](../img/mdl-la-svd-action.svg)

<!--en:23e082e8-->
In $n$ Dimensionen verformen dieselben drei Schritte die Einheitssphäre zu einem Ellipsoid mit den Halbachsenvektoren $\sigma_i\mathbf{u}_i$. Haben nur $r<n$ Eingaberichtungen positive Singulärwerte, so fallen die übrigen $n-r$ Richtungen auf null. Das Bild ist dann ein flaches $r$-dimensionales Ellipsoid im Spann von $\mathbf{u}_1,\ldots,\mathbf{u}_r$.

<!--en:7671f562-->
**Kompakte und dünne Form.** Sei $r$ die Anzahl der positiven Singulärwerte (in :numref:`subsec_mdl-svd-subspaces` zeigen wir, dass $r$ genau der Rang von $\mathbf{A}$ ist). In $\boldsymbol{\Sigma}$ sind nach den ersten $r$ Zeilen und Spalten alle weiteren null. Die zugehörigen Singulärvektoren tragen daher nichts zum Produkt bei. Lassen wir diese Nulleinträge weg, erhalten wir die *kompakte SVD*, $\mathbf{A}=\mathbf{U}_r\boldsymbol{\Sigma}_r\mathbf{V}_r^\top$. Dabei enthalten $\mathbf{U}_r\in\mathbb{R}^{m\times r}$ und $\mathbf{V}_r\in\mathbb{R}^{n\times r}$ jeweils die ersten $r$ Singulärvektoren, und es gilt $\boldsymbol{\Sigma}_r=\operatorname{diag}(\sigma_1,\ldots,\sigma_r)$. Zwischen der vollständigen und der kompakten Form liegt die *dünne SVD*. Sie ist die übliche reduzierte Form, die numerische Bibliotheken auf Anfrage liefern. Sie behält $\min(m,n)$ Spalten in $\mathbf{U}$ und $\mathbf{V}$ bei und enthält für $r<\min(m,n)$ auch die Singulärwerte null. Bei einer $3\times2$-Matrix vom Rang 2 haben die vollständigen Faktoren die Formen $3\times3$, $3\times2$ und $2\times2$, die dünnen Faktoren dagegen $3\times2$, $2\times2$ und $2\times2$ (hier wie bei der kompakten SVD, da die Matrix vollen Spaltenrang hat). Die vollständige und die dünne Form unterscheiden sich bei jeder nicht quadratischen Matrix. Die dünne und die kompakte Form unterscheiden sich bei jeder rangdefizienten Matrix. :numref:`fig_mdl-la-svd-forms` zeigt eine $6\times4$-Matrix vom Rang 3, bei der alle drei Formen verschieden sind.

<!--en:3e459e71-->
![Formen der SVD-Faktoren einer $m\times n$-Matrix vom Rang $r$, maßstabsgetreu für $m=6$, $n=4$ und $r=3$. (a) Vollständige, (b) dünne und (c) kompakte Form, (d) dyadische Summe aus $r$ Summanden vom Rang eins. Ausgefüllte Quadrate auf der Diagonalen von $\boldsymbol{\Sigma}$ stehen für positive Singulärwerte, das gestrichelte Quadrat für einen Singulärwert null. Die gestrichelten Teile von $\mathbf{U}$ und $\mathbf{V}^\top$ werden nur mit Nullen aus $\boldsymbol{\Sigma}$ multipliziert. Jede kleinere Form kann daher einige davon weglassen, ohne das Produkt zu ändern.](../img/mdl-la-svd-forms.svg)

<!--en:d0680536-->
**Die dyadische Summe.** Multiplizieren wir bei der kompakten SVD jeweils eine Spalte mit einer Zeile, so erhalten wir $\mathbf{A}$ als gewichtete Summe von Matrizen vom Rang eins, die oft *Dyaden* heißen:

<!--en:2ecf1d3f-->
Jeder Term $\mathbf{u}_i\mathbf{v}_i^\top$ ist das äußere Produkt einer Ausgabe- und einer Eingaberichtung. In die Summe geht er mit seinem Singulärwert als Gewicht ein. Die Terme sind bereits nach absteigendem Gewicht geordnet. Für die Scherung gilt $\mathbf{T}=\varphi\,\mathbf{u}_1\mathbf{v}_1^\top+\varphi^{-1}\,\mathbf{u}_2\mathbf{v}_2^\top$. Die Matrix zerfällt also in einen Anteil mit Gewicht $1.618$ und einen zweiten mit Gewicht $0.618$. Nur die Anteile mit den größten Gewichten beizubehalten, ist die Idee der niedrigrangigen Approximation in :numref:`subsec_mdl-eckart-young`.

<!--en:6c4b9b93-->
### Existenz

<!--en:4670fa6f-->
Woher kommen $\sigma_i$, $\mathbf{v}_i$ und $\mathbf{u}_i$, und weshalb sind die Ausgaberichtungen orthonormal? Nach dem Spektralsatz aus :numref:`subsec_mdl-spectral-theorem` hat jede symmetrische Matrix eine Orthonormalbasis aus Eigenvektoren mit reellen Eigenwerten. Wir wenden ihn auf die *Gram-Matrix* $\mathbf{A}^\top\mathbf{A}$ an. Diese $n\times n$-Matrix enthält die Skalarprodukte zwischen den Spalten von $\mathbf{A}$. Sie ist symmetrisch und positiv semidefinit (:numref:`subsec_mdl-psd`), denn es gilt $\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}=\|\mathbf{A}\mathbf{x}\|^2\ge0$ für jeden Vektor $\mathbf{x}$.

<!--en:7a0e69fe-->
**Beweis von :eqref:`eq_mdl-svd` (Existenz).** Nach dem Spektralsatz hat $\mathbf{A}^\top\mathbf{A}$ eine Orthonormalbasis aus Eigenvektoren $\mathbf{v}_1,\ldots,\mathbf{v}_n$ mit Eigenwerten $\lambda_1\ge\cdots\ge\lambda_n$. Jeder Eigenwert ist nichtnegativ, denn es gilt $\lambda_i=\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_i=\|\mathbf{A}\mathbf{v}_i\|^2$. Wir definieren

<!--en:41a4e115-->
Die Singulärwerte sind somit die Quadratwurzeln der Eigenwerte der Gram-Matrix, und $r$ zählt die positiven Werte. Für jedes $i\le r$ setzen wir $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$. Entscheidend ist, dass die so gewonnenen Bildvektoren automatisch orthonormal sind. Denn $\mathbf{v}_j$ ist ein Eigenvektor von $\mathbf{A}^\top\mathbf{A}$, und die $\mathbf{v}_i$ sind orthonormal:

<!--en:0ec35a98-->
Die Orthonormalität der Eingabebasis überträgt sich also über $\mathbf{A}^\top\mathbf{A}$ direkt auf die Ausgabebasis. Prüfen wir nun die Dimensionen. Da die $r$ orthonormalen Vektoren $\mathbf{u}_1,\ldots,\mathbf{u}_r$ in $\mathbb{R}^m$ liegen, gilt $r\le m$. Da es insgesamt $n$ Eigenwerte gibt, gilt außerdem $r\le n$. Damit gilt $r\le\min(m,n)$ und folglich $\lambda_i=0$ für jedes $i>r$. Wir ergänzen $\mathbf{u}_1,\ldots,\mathbf{u}_r$ zu einer Orthonormalbasis $\mathbf{u}_1,\ldots,\mathbf{u}_m$ von $\mathbb{R}^m$ (zum Beispiel durch das Gram-Schmidt-Verfahren nach einer beliebigen Basisergänzung). Für $r<i\le n$ gilt $\|\mathbf{A}\mathbf{v}_i\|^2=\lambda_i=0$, also $\mathbf{A}\mathbf{v}_i=\mathbf{0}$. Vergleichen wir $\mathbf{A}\mathbf{V}$ und $\mathbf{U}\boldsymbol{\Sigma}$ spaltenweise. Spalte $i$ von $\mathbf{U}\boldsymbol{\Sigma}$ ist gleich $\sigma_i\mathbf{u}_i$ für $i\le\min(m,n)$ (dieser Vektor verschwindet für $i>r$) und gleich $\mathbf{0}$ für $i>m$ (dort hat $\boldsymbol{\Sigma}$ keinen Diagonaleintrag). In jedem Fall stimmt Spalte $i$ von $\mathbf{A}\mathbf{V}$ mit Spalte $i$ von $\mathbf{U}\boldsymbol{\Sigma}$ überein. Somit gilt $\mathbf{A}\mathbf{V}=\mathbf{U}\boldsymbol{\Sigma}$. Multiplikation von rechts mit $\mathbf{V}^\top=\mathbf{V}^{-1}$ ergibt $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. $\blacksquare$

<!--en:9d92fdeb-->
Der Beweis gibt zugleich ein Rechenverfahren an. Wir diagonalisieren die symmetrische Matrix $\mathbf{A}^\top\mathbf{A}$ und ziehen die Quadratwurzeln ihrer Eigenwerte. Das liefert die Singulärwerte. Die linken Singulärvektoren erhalten wir, indem wir die Eigenvektoren mit $\mathbf{A}$ abbilden.

<!--en:44f04639-->
Rechnen wir jeden Schritt für die Scherung von Hand durch. Für $\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ lautet die Gram-Matrix $\mathbf{T}^\top\mathbf{T}=\left(\begin{smallmatrix}1&1\\1&2\end{smallmatrix}\right)$. Sie hat Spur 3 und Determinante 1. Ihr charakteristisches Polynom ist also $\lambda^2-3\lambda+1$, und ihre Eigenwerte sind $(3\pm\sqrt5)/2$. Mit dem Goldenen Schnitt $\varphi=(1+\sqrt5)/2$ und der Beziehung $\varphi^2=\varphi+1$ vereinfachen sich diese Eigenwerte zu $\varphi^2$ und $\varphi^{-2}$. Die Singulärwerte sind daher $\sigma_1=\varphi\approx1.618$ und $\sigma_2=1/\varphi\approx0.618$. Ihr Produkt ist $1=|\det\mathbf{T}|$.

<!--en:d8f7618b-->
Bestimmen wir nun die Singulärvektoren. Für den ersten rechten Singulärvektor $(x,y)$ liefert die erste Zeile von $(\mathbf{T}^\top\mathbf{T}-\varphi^2\mathbf{I})(x,y)=\mathbf{0}$ die Gleichung $(1-\varphi^2)x+y=0$. Mit $\varphi^2-1=\varphi$ folgt $y=\varphi x$ und damit $\mathbf{v}_1\propto(1,\varphi)$. Anwenden von $\mathbf{T}$ ergibt $\mathbf{T}(1,\varphi)=(1+\varphi,\varphi)=\varphi\,(\varphi,1)$, also $\mathbf{u}_1\propto(\varphi,1)$. Beide Vektoren haben die Länge $\sqrt{1+\varphi^2}=\sqrt{\varphi+2}$. Nach dem Normieren erhalten wir $\mathbf{v}_1\approx(0.526,0.851)$ unter dem Winkel $\arctan\varphi=58.3^\circ$ und $\mathbf{u}_1\approx(0.851,0.526)$ unter dem Winkel $31.7^\circ$. Das zweite Paar ergibt sich aus der Orthogonalität: $\mathbf{v}_2=(-\varphi,1)/\sqrt{\varphi+2}$ und $\mathbf{u}_2=(-1,\varphi)/\sqrt{\varphi+2}$. Tatsächlich gilt $\mathbf{T}\mathbf{v}_2=(1-\varphi,1)/\sqrt{\varphi+2}=\varphi^{-1}\mathbf{u}_2$, denn $(1-\varphi)\varphi=-1$. Sowohl $\mathbf{V}$ als auch $\mathbf{U}$ sind Drehungen der Form $\mathbf{R}_\theta=\left(\begin{smallmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{smallmatrix}\right)$. Die Scherung zerfällt somit in die Faktoren

<!--en:47393a35-->
Geometrisch dreht die Scherung die Ebene um $-58.3^\circ$, streckt sie um den Faktor $1.618$ entlang der $x$-Achse und um $0.618$ entlang der $y$-Achse und dreht sie um nur $31.7^\circ$ zurück. Die beiden Winkelbeträge ergeben zusammen $90^\circ$, denn $\mathbf{u}_1$ entsteht aus $\mathbf{v}_1$ durch Vertauschen der Koordinaten. Darin zeigt sich die Symmetrie $\mathbf{T}^\top=\mathbf{J}\mathbf{T}\mathbf{J}$ mit der Vertauschungsmatrix $\mathbf{J}=\left(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right)$.

<!--en:36678fdf-->
Stimmt das Ergebnis von `np.linalg.svd` aus NumPy mit unserer Handrechnung überein? Singulärvektoren sind nur bis auf das Vorzeichen bestimmt (:numref:`subsec_mdl-svd-uniqueness`). Die folgende Codezelle speichert die Scherung als `A` und wählt zunächst die Vorzeichen so, dass $\mathbf{u}_1$ im ersten Quadranten liegt und $\mathbf{U}$ eine reine Drehung ist.

<!--en:1e2c6b65-->
Die Ergebnisse stimmen überein. Die Bibliothek liefert $\varphi$ und $1/\varphi$ zusammen mit den beiden Drehungen aus :eqref:`eq_mdl-shear-svd`.

<!--en:88c7593a-->
Dieselbe Konstruktion funktioniert unverändert für eine rechteckige Matrix. Betrachten wir die $3\times2$-Matrix mit den Zeilen $(3,1)$, $(1,3)$ und $(0,2)$. Ihre Gram-Matrix ist $\left(\begin{smallmatrix}10&6\\6&14\end{smallmatrix}\right)$ mit den Eigenwerten $12\pm\sqrt{40}$. Die nächste Codezelle setzt die drei Schritte des Beweises direkt um (Eigenvektoren von $\mathbf{A}^\top\mathbf{A}$, Quadratwurzeln und $\mathbf{u}_i=\mathbf{A}\mathbf{v}_i/\sigma_i$) und vergleicht das Ergebnis mit `np.linalg.svd`.

<!--en:7c62d6a9-->
Beide Wege liefern $\sigma_1=\sqrt{12+\sqrt{40}}\approx4.281$ und $\sigma_2=\sqrt{12-\sqrt{40}}\approx2.382$. Wie im Beweis sind die Bildvektoren $\mathbf{u}_i$ orthonormal. Sowohl $\mathbf{U}$ als auch $\mathbf{V}$ stimmen bis auf die Vorzeichen mit der Ausgabe der Bibliothek überein.

<!--en:a244dc02-->
### Eindeutigkeit und maximale Streckung

<!--en:a52a07e5-->
Ist die SVD einer Matrix eindeutig? Die *Singulärwerte* sind stets eindeutig, denn sie sind die Quadratwurzeln der Eigenwerte von $\mathbf{A}^\top\mathbf{A}$. Diese Matrix ist durch $\mathbf{A}$ vollständig bestimmt. Bei den *Singulärvektoren* gibt es dagegen drei Freiheiten:

<!--en:ceb306e0-->
* Bei einem einfachen (nicht mehrfachen) Singulärwert $\sigma_i>0$ können wir lediglich beide Vorzeichen gemeinsam wechseln: $(\mathbf{u}_i,\mathbf{v}_i)\to(-\mathbf{u}_i,-\mathbf{v}_i)$. Wechselten wir nur ein Vorzeichen, wäre $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ verletzt.

<!--en:9390dce3-->
* Bei einem mehrfachen Singulärwert $\sigma>0$ mit Indexmenge $\mathcal{I}$ lassen sich die Spalten $\mathbf{V}_{\mathcal I}$ durch jede orthogonale Matrix $\mathbf{Q}$ passender Größe in $\mathbf{V}_{\mathcal I}\mathbf{Q}$ überführen. Dabei müssen wir $\mathbf{U}_{\mathcal I}$ mit *derselben* Matrix $\mathbf{Q}$ drehen, damit $\mathbf{U}_{\mathcal I}=\mathbf{A}\mathbf{V}_{\mathcal I}/\sigma$ erhalten bleibt.

<!--en:e4ef4b12-->
* Für $\sigma=0$ können wir die Vektoren $\mathbf{v}_{r+1},\ldots,\mathbf{v}_n$ im Nullraum und die Vektoren $\mathbf{u}_{r+1},\ldots,\mathbf{u}_m$ im linken Nullraum unabhängig voneinander drehen, da sie in der dyadischen Summe :eqref:`eq_mdl-svd-dyadic` nicht auftreten.

<!--en:c52651fe-->
Diese Freiheit erklärt, weshalb zwei numerische Bibliotheken oder sogar dieselbe Bibliothek auf verschiedenen Rechnern Singulärvektoren mit entgegengesetzten Vorzeichen liefern können. Deshalb legen unsere Codezellen vor der Ausgabe der Vektoren eine Vorzeichenkonvention fest.

<!--en:96cb4a17-->
Über die Zerlegung von $\mathbf{A}$ hinaus hat der größte Singulärwert eine unmittelbare geometrische Bedeutung. Er ist der größte Faktor, um den $\mathbf{A}$ einen Einheitsvektor strecken kann:

<!--en:544416cb-->
Das folgt direkt aus der Proposition über den Rayleigh-Quotienten in :numref:`subsec_mdl-rayleigh`. Unter allen Einheitsvektoren $\mathbf{x}$ maximiert der Eigenvektor zum größten Eigenwert die quadratische Form $\mathbf{x}^\top\mathbf{S}\mathbf{x}$ einer symmetrischen Matrix $\mathbf{S}$. Der Maximalwert ist dieser Eigenwert. Mit $\mathbf{S}=\mathbf{A}^\top\mathbf{A}$ sehen wir, dass $\|\mathbf{A}\mathbf{x}\|^2=\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}$ bei $\mathbf{x}=\mathbf{v}_1$ den Maximalwert $\lambda_1=\sigma_1^2$ annimmt. Damit ist $\sigma_1$ genau die Spektralnorm $\|\mathbf{A}\|_2$ aus :numref:`sec_linear-algebra`. Bei der Scherung wird kein Einheitsvektor um mehr als $\varphi=1.618$ gestreckt, und $\mathbf{v}_1$ erreicht diese Streckung. Die übrigen Singulärwerte lösen dasselbe Maximierungsproblem, wenn wir jeweils die bisherigen rechten Singulärvektoren durch Orthogonalität ausschließen. Dieses Verfahren heißt *Deflation*: $\sigma_k=\max\{\|\mathbf{A}\mathbf{x}\| : \|\mathbf{x}\|=1,\ \mathbf{x}\perp\mathbf{v}_1,\ldots,\mathbf{v}_{k-1}\}$. Aufgabe :exref:`ex_mdl-svd-low-rank-norms-singular-values`a kehrt diesen Gedankengang um und konstruiert die gesamte SVD allein aus der maximalen Streckung, ohne den Spektralsatz zu verwenden.

<!--en:f6f86406-->
### Singulärwerte und Eigenwerte

<!--en:4d90fcfb-->
Wie hängen Singulärwerte und Eigenwerte zusammen? Setzen wir :eqref:`eq_mdl-svd` in die beiden Gram-Matrizen ein und nutzen $\mathbf{U}^\top\mathbf{U}=\mathbf{I}$ und $\mathbf{V}^\top\mathbf{V}=\mathbf{I}$, so erhalten wir

<!--en:d41030c0-->
Beide Ausdrücke sind symmetrische Eigenwertzerlegungen. Die rechten Singulärvektoren sind Eigenvektoren von $\mathbf{A}^\top\mathbf{A}$, die linken Singulärvektoren sind Eigenvektoren von $\mathbf{A}\mathbf{A}^\top$, und die Quadrate $\sigma_i^2$ der positiven Singulärwerte sind die gemeinsamen Eigenwerte ungleich null. Ist $\mathbf{A}$ symmetrisch und positiv semidefinit, so stimmen SVD und Eigenwertzerlegung überein, und es gilt $\mathbf{U}=\mathbf{V}$.

<!--en:e2f37a27-->
Bei anderen Matrizen können sich die beiden Zerlegungen jedoch unterscheiden. Ein häufiger Fehler ist die Annahme, die Singulärwerte seien gleich den Beträgen der Eigenwerte. Betrachten wir die Beispielmatrix $\mathbf{A}=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ aus diesem Kapitel. Ihre Eigenwerte $2\pm i$ haben beide den Betrag $\sqrt5\approx2.236$ (:numref:`sec_mdl-eigendecompositions`). Ihre Gram-Matrix $\mathbf{A}^\top\mathbf{A}=\left(\begin{smallmatrix}2&-1\\-1&13\end{smallmatrix}\right)$ hat Spur 15 und Determinante 25. Daraus folgt $\sigma_{1,2}^2=(15\pm5\sqrt5)/2$ und damit $\sigma_{1,2}=(5\pm\sqrt5)/2\approx3.618,\,1.382$. Diese Werte liegen weit von $2.236$ entfernt. Für jede quadratische Matrix verbindet jedoch eine einfache Identität beide Listen:

<!--en:ed9397d1-->
Die erste Gleichheit gilt, weil orthogonale Matrizen die Determinante $\pm1$ haben. Die zweite gilt, weil die Determinante das Produkt aller Eigenwerte ist. Für eine $2\times2$-Matrix mit zwei konjugierten Eigenwerten vom Betrag $|\lambda|$ folgt daraus $|\lambda|=\sqrt{\sigma_1\sigma_2}$. In unserem Beispiel gilt tatsächlich $\sqrt{3.618\cdot1.382}=\sqrt5$. Die Identität liefert auch die Volumendeutung der Determinante aus :numref:`sec_mdl-linear-maps`. Drehungen erhalten den Flächeninhalt. Daher geht die gesamte Flächenänderung auf die diagonale Streckung $\boldsymbol{\Sigma}$ zurück. Unsere Beispielmatrix bildet das Einheitsquadrat auf ein Parallelogramm mit dem Flächeninhalt $3.618\cdot1.382=5$ ab.

<!--en:94ab5a4e-->
**Berechnung der SVD in der Praxis.** Unser Existenzbeweis konstruiert die SVD aus $\mathbf{A}^\top\mathbf{A}$, doch numerische Bibliotheken bilden dieses Produkt nie. Beim Bilden von $\mathbf{A}^\top\mathbf{A}$ quadriert sich die Konditionszahl (:numref:`subsec_mdl-condition-number`). In der Gleitkommaarithmetik können dadurch kleine Singulärwerte verloren gehen. Bibliotheken wenden deshalb orthogonale Transformationen direkt auf $\mathbf{A}$ an, um die Matrix auf Bidiagonalform zu bringen (Bidiagonalisierung nach Golub und Kahan). Anschließend diagonalisieren sie die bidiagonale Matrix :cite:`Golub.Van-Loan.1996`.

<!--en:a148c0ea-->
### Die vier fundamentalen Unterräume

<!--en:739aeb42-->
Welche Eingaben bildet $\mathbf{A}$ auf null ab, und welche Ausgaben kann die Matrix erreichen? In :numref:`sec_mdl-linear-maps` haben wir diese Mengen als Nullraum und Spaltenraum bezeichnet und den Rang als Anzahl linear unabhängiger Spalten definiert. Da $\mathbf{U}$ und $\mathbf{V}$ invertierbar sind, bleiben bei der Multiplikation mit ihnen alle Dimensionen erhalten. Somit gilt $\operatorname{rank}\mathbf{A}=\operatorname{rank}\boldsymbol{\Sigma}$. Wir brauchen also nur die von null verschiedenen Diagonaleinträge zu zählen:

<!--en:37dfcf75-->
Der Rang zählt somit die Richtungen, die bei der Abbildung erhalten bleiben. Mit einer Rundungstoleranz (siehe unten) ist das Zählen der Singulärwerte das numerisch zuverlässigste Verfahren zur Rangbestimmung. Es ist aufwendiger als das gaußsche Eliminationsverfahren, aber wesentlich stabiler. Eine Störung $\mathbf{E}$, etwa ein Rundungsfehler in der Gleitkommaarithmetik, ändert jeden Singulärwert um höchstens $\|\mathbf{E}\|_2$ (Weyl-Ungleichung in :numref:`subsec_mdl-eckart-young`). Auf eine deutliche Lücke zwischen großen und sehr kleinen Singulärwerten können wir uns daher verlassen.

<!--en:8c328207-->
Teilen wir die Singulärvektoren am Index $r$ in zwei Gruppen, erhalten wir sogar explizite Orthonormalbasen für alle vier fundamentalen Unterräume von $\mathbf{A}$ direkt aus der dyadischen Summe :eqref:`eq_mdl-svd-dyadic`:

<!--en:b076611e-->
* Der **Spaltenraum** (Bildraum) von $\mathbf{A}$ ist $\operatorname{span}\{\mathbf{u}_1,\ldots,\mathbf{u}_r\}$. Jede Ausgabe $\mathbf{A}\mathbf{x}=\sum_{i\le r}\sigma_i(\mathbf{v}_i^\top\mathbf{x})\mathbf{u}_i$ ist eine Linearkombination dieser Vektoren, und jeden einzelnen erhalten wir über $\mathbf{u}_i=\mathbf{A}(\mathbf{v}_i/\sigma_i)$.

<!--en:0e5550ad-->
* Der **Nullraum** von $\mathbf{A}$ ist $\operatorname{span}\{\mathbf{v}_{r+1},\ldots,\mathbf{v}_n\}$. Nach derselben Formel gilt $\mathbf{A}\mathbf{x}=\mathbf{0}$ genau dann, wenn $\mathbf{v}_i^\top\mathbf{x}=0$ für jedes $i\le r$ gilt (die $\mathbf{u}_i$ sind linear unabhängig und die $\sigma_i$ positiv). Das bedeutet, dass $\mathbf{x}$ orthogonal zu $\mathbf{v}_1,\ldots,\mathbf{v}_r$ ist.

<!--en:fa36ab35-->
* Der **Zeilenraum** (Spaltenraum von $\mathbf{A}^\top$) ist $\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_r\}$, und der **linke Nullraum** (Nullraum von $\mathbf{A}^\top$) ist $\operatorname{span}\{\mathbf{u}_{r+1},\ldots,\mathbf{u}_m\}$. Dazu wenden wir dieselben beiden Argumente auf $\mathbf{A}^\top=\mathbf{V}\boldsymbol{\Sigma}^\top\mathbf{U}^\top$ an, wodurch sich die Rollen der $\mathbf{u}_i$ und $\mathbf{v}_i$ vertauschen.

<!--en:d299833e-->
Daraus ergibt sich ein einfaches Bild jeder linearen Abbildung. Der Eingaberaum zerfällt in zwei orthogonale Teile, den $r$-dimensionalen Zeilenraum und den $(n-r)$-dimensionalen Nullraum. Ebenso zerfällt der Ausgaberaum in den $r$-dimensionalen Spaltenraum und den $(m-r)$-dimensionalen linken Nullraum. Auf dem Zeilenraum bildet $\mathbf{A}$ jeden Vektor $\mathbf{v}_i$ auf $\sigma_i\mathbf{u}_i$ ab. Das ergibt eine Bijektion auf den Spaltenraum. Der gesamte Nullraum geht in den Nullvektor über.

<!--en:cfd22fc9-->
Betrachten wir als einfachsten nichttrivialen Fall die singuläre Matrix $\mathbf{B}=\left(\begin{smallmatrix}2&1\\4&2\end{smallmatrix}\right)$ aus :numref:`sec_mdl-linear-maps`. Ihre Gram-Matrix $\left(\begin{smallmatrix}20&10\\10&5\end{smallmatrix}\right)$ hat die Eigenwerte 25 und 0. Also gilt $\sigma_1=5$, $\sigma_2=0$ und $r=1$. Die vier Unterräume sind die Geraden, die folgende Vektoren aufspannen:

<!--en:32931c56-->
Hier ist der Spaltenraum die Gerade $y=2x$ in Richtung $\mathbf{u}_1$ und der linke Nullraum die dazu senkrechte Gerade in Richtung $\mathbf{u}_2$. In der Eingabeebene verläuft der Nullraum entlang $(1,-2)$, senkrecht zur Zeile $(2,1)$, die den Zeilenraum aufspannt. :numref:`fig_mdl-la-svd-subspaces` veranschaulicht die allgemeine Geometrie. Wie wir in :numref:`subsec_mdl-pseudoinverse` sehen, kehrt die Pseudoinverse die Bijektion vom Zeilenraum auf den Spaltenraum um.

<!--en:5f330662-->
![Die vier fundamentalen Unterräume. $\mathbf{A}$ bildet den Zeilenraum $\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_r\}$ bijektiv auf den Spaltenraum $\operatorname{span}\{\mathbf{u}_1,\dots,\mathbf{u}_r\}$ ab, und zwar durch $\mathbf{v}_i\mapsto\sigma_i\mathbf{u}_i$. Den Nullraum bildet sie auf null ab. Die Pseudoinverse $\mathbf{A}^{+}$ (gestrichelt) kehrt die Abbildung um, $\sigma_i\mathbf{u}_i\mapsto\mathbf{v}_i$, und bildet den linken Nullraum auf null ab.](../img/mdl-la-svd-subspaces.svg)

<!--en:4cc5fccf-->
**Numerischer Rang.** In der Gleitkommaarithmetik hat eine mathematisch rangdefiziente Matrix nicht zuverlässig Singulärwerte von exakt null. Rundungsfehler hinterlassen meist winzige Werte der Größenordnung $\epsilon_{\text{mach}}\sigma_1$. In der Praxis bestimmen wir den Rang daher mit einem Schwellenwert. Die NumPy-Funktion `np.linalg.matrix_rank` zählt die Singulärwerte oberhalb von $\sigma_1\max(m,n)\,\epsilon_{\text{mach}}$. Diese Toleranz skaliert sowohl mit dem größten Singulärwert als auch mit den Dimensionen der Matrix. Sehen wir uns die „verschwindenden“ Singulärwerte einer $4\times4$-Matrix vom Rang 2 an.

<!--en:f4dd954c-->
Zwei Singulärwerte haben die Größenordnung eins, die übrigen beiden entstehen allein durch Rundungsfehler. Mit $10^{-15}$ oder weniger liegen sie weit unter der Toleranz. Einer von ihnen kann durch Rundung genau null sein, doch selten gilt das für beide. Das Zählen exakter Nullen ergäbe daher einen falschen Rang. Mit dem Schwellenwert erhalten wir den tatsächlichen Rang 2.

<!--en:0074b762-->
## Niedrigrangige Approximation

<!--en:a314113a-->
### Beste niedrigrangige Approximation

<!--en:d86b51f8-->
Kehren wir zum Katzenfoto vom Anfang des Abschnitts zurück. Welche zwanzig Anteile vom Rang eins sollten wir behalten? Die dyadische Summe :eqref:`eq_mdl-svd-dyadic` ordnet die Anteile von $\mathbf{A}$ bereits nach absteigendem Gewicht. Es liegt also nahe, die ersten $k$ Summanden zu behalten. So erhalten wir die *abgeschnittene SVD* (engl. truncated SVD),

<!--en:aa89522a-->
Die Matrix $\mathbf{A}_k$ hat höchstens Rang $k$. Um sie zu speichern, genügen $k$ Singulärwerte, $k$ Spalten der Länge $m$ und $k$ Spalten der Länge $n$, insgesamt also $k(m+n+1)$ statt $mn$ Zahlen. Für unser Foto mit $400\times500$ Pixeln und $k=20$ sind das 18 020 von 200 000 Zahlen. Wie groß ist der Approximationsfehler, und könnte eine geschickter gewählte Matrix vom Rang $k$ noch besser abschneiden?

<!--en:93f91be8-->
**Matrixnormen aus Singulärwerten.** Um den Approximationsfehler zu messen, brauchen wir eine Matrixnorm. Für die Frobeniusnorm aus :numref:`sec_linear-algebra` gilt $\|\mathbf{A}\|_F^2=\sum_{ij}a_{ij}^2$. Für das Normquadrat summieren wir die quadrierten Längen der Spalten von $\mathbf{A}$. Diese stehen auf der Diagonalen von $\mathbf{A}^\top\mathbf{A}$. Somit gilt $\|\mathbf{A}\|_F^2=\operatorname{tr}(\mathbf{A}^\top\mathbf{A})$. Setzen wir die SVD ein und verschieben wir mithilfe der zyklischen Eigenschaft der Spur (:numref:`sec_mdl-linear-maps`) den Faktor $\mathbf{V}$ auf die andere Seite, so erhalten wir

<!--en:66067072-->
Die Frobeniusnorm ist also die Quadratwurzel aus der Summe der quadrierten Singulärwerte. Daher bleibt sie bei Multiplikation mit einer orthogonalen Matrix von links oder rechts unverändert. Für die Scherung gilt $1+1+0+1=3=\varphi^2+\varphi^{-2}$. Die Spektralnorm wiederum ist der größte Singulärwert, $\|\mathbf{A}\|_2=\sigma_1$, wie :eqref:`eq_mdl-sigma1-variational` zeigt.

<!--en:c18b13f4-->
Mit diesen Ausdrücken für beide Normen können wir den zentralen Approximationssatz der linearen Algebra formulieren.

<!--en:93d32511-->
**Satz (Eckart, Young und Mirsky).** *Sei $\sigma_i=0$ für $i>r$. Für jedes $k\ge0$ hat der kleinste Fehler, den eine Matrix vom Rang höchstens $k$ in der Spektralnorm erreichen kann, den Wert*

<!--en:2208ecb1-->
*und der kleinste quadrierte Fehler in der Frobeniusnorm beträgt $\min_{\operatorname{rank}\mathbf{B}\le k}\|\mathbf{A}-\mathbf{B}\|_F^2=\sum_{i>k}\sigma_i^2$. Die abgeschnittene SVD $\mathbf{B}=\mathbf{A}_k$ erreicht beide Minima.*

<!--en:d4bf4831-->
Keine Matrix vom Rang $k$ kann die abgeschnittene SVD in einer der beiden Normen übertreffen. Bei der besten Approximation vom Rang $k$ ist der Fehler in der Spektralnorm gleich dem ersten ausgelassenen Singulärwert. In der Frobeniusnorm bestimmt die gesamte Energie der ausgelassenen Singulärwerte den Fehler. Eckart und Young bewiesen den Satz für die Frobeniusnorm :cite:`Eckart.Young.1936`. Mirsky erweiterte ihn auf jede Norm, die bei orthogonaler Multiplikation von beiden Seiten invariant bleibt :cite:`Mirsky.1960`. Wir beweisen zuerst die Schranke für die Spektralnorm. Die Schranke für die Frobeniusnorm folgt dann mithilfe der Weyl-Ungleichung, die sich aus demselben Argument ergibt.

<!--en:fcfbd92f-->
Betrachten wir vor dem allgemeinen Beweis den Fall $n=3$ und $k=1$. Jede Matrix $\mathbf{B}$ vom Rang eins bildet eine ganze Ebene von Eingaben auf null ab. Die ersten beiden rechten Singulärvektoren von $\mathbf{A}$ spannen eine weitere Ebene auf. In $\mathbb{R}^3$ schneiden sich zwei Ursprungsebenen mindestens in einer Geraden. Auf dieser Geraden liefert $\mathbf{B}$ null, während $\mathbf{A}$ jeden Vektor mindestens um den Faktor $\sigma_2$ streckt. Daher muss $\mathbf{A}-\mathbf{B}$ einen Einheitsvektor mindestens um den Faktor $\sigma_2$ strecken (:numref:`fig_mdl-la-svd-dimension-count`). Der folgende Beweis überträgt dieses Dimensionsargument auf $n$ Dimensionen.

<!--en:c04b9e2f-->
![Dimensionsargument für den Satz von Eckart und Young bei $n=3$, $k=1$. Die Ebene $\ker\mathbf{B}$ und die von den ersten beiden rechten Singulärvektoren aufgespannte Ebene haben mindestens eine Gerade gemeinsam. Auf dieser Geraden verschwindet $\mathbf{B}$, während $\mathbf{A}$ noch mindestens um den Faktor $\sigma_2$ streckt.](../img/mdl-la-svd-dimension-count.svg)

<!--en:900e34e7-->
**Beweis (Spektralnorm).** *Erstens erreicht die abgeschnittene SVD den Fehler $\sigma_{k+1}$.* Das Residuum $\mathbf{A}-\mathbf{A}_k=\sum_{i>k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ liegt bereits in kompakter SVD-Form mit größtem Singulärwert $\sigma_{k+1}$ vor. Somit gilt $\|\mathbf{A}-\mathbf{A}_k\|_2=\sigma_{k+1}$ nach :eqref:`eq_mdl-sigma1-variational`. Ebenso liefert :eqref:`eq_mdl-frobenius-singular` die Gleichung $\|\mathbf{A}-\mathbf{A}_k\|_F^2=\sum_{i>k}\sigma_i^2$.

<!--en:7357527d-->
*Zweitens kann keine Matrix vom Rang höchstens $k$ einen kleineren Fehler erreichen.* Für $k\ge r$ ist der Fehler null und die Aussage bewiesen. Nehmen wir also $k<r$ an und sei $\operatorname{rank}\mathbf{B}\le k$. Nach dem Rangsatz (:numref:`sec_mdl-linear-maps`) hat der Nullraum $\mathcal{K}=\ker\mathbf{B}$ mindestens die Dimension $n-k$. Der Spann $\mathcal{V}=\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_{k+1}\}$ der ersten $k+1$ rechten Singulärvektoren hat die Dimension $k+1$. Da die Summe der Dimensionen größer als $n$ ist, enthalten beide Unterräume einen gemeinsamen Vektor ungleich null:

<!--en:adafd794-->
Die Gleichheit ist die Dimensionsformel für zwei Unterräume. Die Ungleichung folgt aus $\dim(\mathcal{K}+\mathcal{V})\le n$. (Zum Beweis der Dimensionsformel ergänzen wir eine Basis von $\mathcal{K}\cap\mathcal{V}$ getrennt zu einer Basis von $\mathcal{K}$ und einer Basis von $\mathcal{V}$. Die vereinigten Basisvektoren sind linear unabhängig, spannen $\mathcal{K}+\mathcal{V}$ auf und haben die Anzahl $\dim\mathcal{K}+\dim\mathcal{V}-\dim(\mathcal{K}\cap\mathcal{V})$.) Wir wählen einen Einheitsvektor $\mathbf{x}\in\mathcal{K}\cap\mathcal{V}$ und schreiben ihn als $\mathbf{x}=\sum_{i\le k+1}c_i\mathbf{v}_i$ mit $\sum_i c_i^2=1$. Dann gilt $\mathbf{B}\mathbf{x}=\mathbf{0}$. Mit $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ und der Orthonormalität der $\mathbf{u}_i$ erhalten wir

<!--en:45c3494d-->
Somit gilt $\|\mathbf{A}-\mathbf{B}\|_2\ge\|(\mathbf{A}-\mathbf{B})\mathbf{x}\|=\|\mathbf{A}\mathbf{x}\|\ge\sigma_{k+1}$. $\blacksquare$

<!--en:f49f1112-->
Geometrisch bedeutet das: Jede Vergleichsmatrix vom Rang $k$, etwa $\mathbf{B}$, muss mindestens eine Richtung im Spann der ersten $k+1$ rechten Singulärvektoren auf null abbilden. Jede Richtung in diesem Spann streckt $\mathbf{A}$ aber mindestens um den Faktor $\sigma_{k+1}$. Für die Scherung $\mathbf{T}$ mit $k=1$ hat die abgeschnittene SVD $\mathbf{T}_1=\varphi\,\mathbf{u}_1\mathbf{v}_1^\top=\tfrac{1}{\sqrt5}\left(\begin{smallmatrix}\varphi&\varphi^2\\1&\varphi\end{smallmatrix}\right)\approx\left(\begin{smallmatrix}0.724&1.171\\0.447&0.724\end{smallmatrix}\right)$ in beiden Normen den Fehler $1/\varphi=0.618$ (das Residuum hat Rang eins, daher stimmen seine Spektralnorm und seine Frobeniusnorm überein). Keine Matrix vom Rang eins kommt der Scherung näher.

<!--en:6c64e862-->
Um den Satz auch für die Frobeniusnorm zu beweisen, leiten wir aus der soeben bewiesenen Schranke für die Spektralnorm zunächst eine allgemeine Störungsabschätzung ab.

<!--en:0f64085d-->
**Lemma (Weyl-Ungleichung für Singulärwerte).** *Für zwei beliebige Matrizen $\mathbf{X},\mathbf{Y}$ derselben Form und Indizes $i,j\ge1$ mit $i+j-1\le\min(m,n)$ gilt*

<!--en:f2c571f5-->
**Beweis.** Seien $\mathbf{X}_{i-1}$ und $\mathbf{Y}_{j-1}$ die abgeschnittenen SVDs :eqref:`eq_mdl-truncated-svd` (mit $\mathbf{X}_0=\mathbf{0}$). Ihre Summe $\mathbf{X}_{i-1}+\mathbf{Y}_{j-1}$ hat höchstens Rang $i+j-2$. Wenden wir die untere Schranke für die Spektralnorm auf $\mathbf{X}+\mathbf{Y}$ mit $k=i+j-2$ an, folgt $\sigma_{i+j-1}(\mathbf{X}+\mathbf{Y})\le\|\mathbf{X}+\mathbf{Y}-\mathbf{X}_{i-1}-\mathbf{Y}_{j-1}\|_2$. Nach der Dreiecksungleichung und dem Teil des Satzes von Eckart und Young, der die Erreichbarkeit zeigt, beträgt die rechte Seite höchstens $\|\mathbf{X}-\mathbf{X}_{i-1}\|_2+\|\mathbf{Y}-\mathbf{Y}_{j-1}\|_2=\sigma_i(\mathbf{X})+\sigma_j(\mathbf{Y})$. $\blacksquare$

<!--en:b4337401-->
Addieren wir $\mathbf{Y}$ zu $\mathbf{X}$, so kann der $(i+j-1)$-te Singulärwert der Summe den $i$-ten Singulärwert von $\mathbf{X}$ höchstens um $\sigma_j(\mathbf{Y})$ übersteigen. Setzen wir $j=1$ und wenden die Ungleichung in beiden Richtungen an, so sehen wir, dass eine Störung um $\mathbf{Y}$ keinen Singulärwert um mehr als $\|\mathbf{Y}\|_2$ verschiebt :cite:`Horn.Johnson.2012`. Die Einzelheiten untersuchen Sie in Aufgabe :exref:`ex_mdl-svd-low-rank-weyl-perturbation`.

<!--en:d75344c7-->
**Beweis (Frobeniusnorm).** Wir wenden das Lemma von Weyl mit $\mathbf{X}=\mathbf{A}-\mathbf{B}$, $\mathbf{Y}=\mathbf{B}$ und $j=k+1$ an. Wegen $\operatorname{rank}\mathbf{B}\le k$ gilt $\sigma_{k+1}(\mathbf{B})=0$. Daraus folgt $\sigma_{k+i}(\mathbf{A})\le\sigma_i(\mathbf{A}-\mathbf{B})$ für jedes $i\ge1$. Wir quadrieren, summieren über $i$ und wenden :eqref:`eq_mdl-frobenius-singular` an:

<!--en:5abcff95-->
Diese Schranke erreicht die abgeschnittene SVD $\mathbf{A}_k$. $\blacksquare$

<!--en:1cadbd57-->
**Eindeutigkeit der optimalen Approximation.** Die beiden Normen unterscheiden sich in einem Punkt. In der Spektralnorm ist die beste Approximation vom Rang $k$ nie eindeutig, wenn $k\ge1$ und $\sigma_{k+1}>0$ gelten. Für $\mathbf{A}=\operatorname{diag}(3,2,1)$ und $k=1$ liefert jede Matrix $\mathbf{B}=\operatorname{diag}(3+t,0,0)$ mit $|t|\le1$ den Fehler $\|\mathbf{A}-\mathbf{B}\|_2=\max(|t|,2,1)=2=\sigma_2$. In der Frobeniusnorm dagegen ist $\mathbf{A}_k$ die *einzige* optimale Approximation, wenn $\sigma_k>\sigma_{k+1}$ oder $\sigma_{k+1}=0$ gilt (im letzten Fall ist $\mathbf{A}_k=\mathbf{A}$) :cite:`Golub.Van-Loan.1996`. Diese Eindeutigkeit ist ein Grund dafür, die nun folgende Hauptkomponentenanalyse mit der Frobeniusnorm zu formulieren.

<!--en:c4480424-->
Prüfen wir den Satz an unserem Katzenfoto mit $400\times500$ Pixeln. Der Satz von Eckart und Young gibt für jede abgeschnittene SVD den Fehler $\sigma_{k+1}$ in der Spektralnorm an. Er sagt jedoch nichts darüber aus, wie rasch die Singulärwerte eines echten Bildes abfallen oder wie viel schlechter eine naheliegende Alternative vom Rang $k$ abschneidet. Die folgende Codezelle lädt `img/cat1.jpg` und wandelt das Bild mit den üblichen Luminanzgewichten für Rot, Grün und Blau in Graustufen um. Die abgeschnittenen Zerlegungen berechnet sie mit `np.einsum`. Diese Funktion multipliziert ihre Eingaben und summiert über alle Indizes, die nach dem Pfeil fehlen. Da Zeile $i$ von `Vt` gleich $\mathbf{v}_i^\top$ ist, berechnet `'i,mi,in->mn'` die dyadische Summe.

<!--en:ae5b5bb0-->
Der Fehler in der Spektralnorm ist wie vorausgesagt bei jedem Rang gleich $\sigma_{k+1}$. Bei Rang 20 speichern wir nur 9% der Zahlen bei einem relativen Fehler von 8%. In :numref:`fig_mdl-la-eckart-young` erscheint die Katze bei Rang 1 als heller Fleck. Bei Rang 5 erkennen wir eine Silhouette mit vier Pfoten, bei Rang 20 eindeutig eine auf dem Rücken liegende Katze. Links sehen wir, weshalb so wenige Anteile genügen. Über die ersten hundert Indizes fallen die Singulärwerte annähernd nach einem Potenzgesetz ab, $\sigma_i\propto i^{-1.1}$.

<!--en:af0703e5-->
![Der Satz von Eckart und Young an einem Foto mit $400\times500$ Pixeln. Links fallen die Singulärwerte annähernd nach einem Potenzgesetz ab (doppelt logarithmische Achsen, schattiert sind die zwanzig für das Rang-20-Bild beibehaltenen Werte). Rechts die abgeschnittenen Zerlegungen $\mathbf{A}_k$ mit relativem Fehler in der Frobeniusnorm und dem Anteil $k(m+n+1)/mn$ der ursprünglichen Zahlen, den sie speichern.](../img/mdl-la-eckart-young.svg)

<!--en:b59687a6-->
**Energieanteil.** Die Fehlerformel für die Frobeniusnorm liefert eine praktische Regel für die Wahl von $k$. Von der gesamten quadratischen „Energie“ $\sum_i\sigma_i^2$ behält die auf Rang $k$ abgeschnittene SVD folgenden Anteil:

<!--en:89a75956-->
Der relative Fehler in der Frobeniusnorm ist also die Quadratwurzel aus eins minus dem Energieanteil. Bei 95% erhaltener Energie beträgt der relative Fehler somit $\sqrt{0.05}\approx22\%$, bei 99% sinkt er auf 10%. Diesen Anteil müssen wir jedoch sorgfältig interpretieren. Beim Foto entfallen bereits 89% der Energie auf $\sigma_1^2$ allein. Ein konstantes Bild hat Rang eins (alle Zeilen sind identisch). Da die mittlere Helligkeit des Fotos groß ist, erfasst die erste Dyade überwiegend diesen Mittelwert. Schon Rang 4 erhält 95% der Energie (bei 21% relativem Fehler). Ziehen wir zunächst den mittleren Pixelwert ab, brauchen wir für 95% dagegen Rang 8 statt 4 (wie wir in :numref:`subsec_mdl-spectrum-control` nachprüfen). Prüfen Sie daher stets, ob ein von null verschiedener Mittelwert $\sigma_1$ dominiert, oder ziehen Sie wie bei der Hauptkomponentenanalyse zuerst den Mittelwert ab.

<!--en:36b18338-->
Wie viel gewinnen wir mit dem Satz von Eckart und Young gegenüber einem einfachen Rang-20-Verfahren? Die letzten Zeilen der Codezelle wählen die 20 Spalten des Fotos mit den größten Normen und projizieren jede Spalte auf deren Spann. Auf dieser Idee beruhen CUR-Zerlegungen, die eine Matrix durch eine Auswahl ihrer eigenen Zeilen und Spalten approximieren. Der relative Fehler beträgt 0.31 und ist damit fast viermal so groß wie der Fehler von 0.08 bei der abgeschnittenen SVD.

<!--en:d2716bb4-->
**Rauschunterdrückung durch Abschneiden.** Neben der Kompression ist die Rauschunterdrückung ein weiterer Grund, eine SVD abzuschneiden. Eine beobachtete $m\times n$-Matrix ($m\le n$) bestehe aus einem niedrigrangigen Signal und additivem Zufallsrauschen, $\mathbf{A}=\mathbf{B}+\mathbf{N}$. Die Einträge von $\mathbf{N}$ seien unabhängig mit Mittelwert null und Standardabweichung $\sigma_{\text{noise}}$. Wenden wir das Marchenko-Pastur-Gesetz aus :numref:`subsec_mdl-sample-covariance-mp` auf $\mathbf{N}\mathbf{N}^\top/n$ an, so konzentrieren sich die Singulärwerte einer großen Rauschmatrix im Intervall $[(\sqrt n-\sqrt m)\sigma_{\text{noise}},(\sqrt n+\sqrt m)\sigma_{\text{noise}}]$. Signalkomponenten mit Singulärwerten deutlich über der oberen Intervallgrenze heben sich klar vom Rauschen ab, schwächere gehen darin unter. Für quadratische Matrizen mit bekanntem Rauschpegel bewiesen Gavish und Donoho, dass der asymptotisch optimale harte Schwellenwert $\tfrac{4}{\sqrt3}\sqrt n\,\sigma_{\text{noise}}\approx2.309\sqrt n\,\sigma_{\text{noise}}$ beträgt. Er liegt etwas über der oberen Rauschgrenze $2\sqrt n\,\sigma_{\text{noise}}$ :cite:`Gavish.Donoho.2014`. In Aufgabe :exref:`ex_mdl-svd-low-rank-optimal-hard-threshold` leiten Sie die Konstante $4/\sqrt3$ her.

<!--en:d5af3c15-->
### Hauptkomponentenanalyse

<!--en:713127de-->
Betrachten wir einen Datensatz, dessen Beispiele die Zeilen von $\mathbf{X}\in\mathbb{R}^{n\times d}$ bilden, mit $n$ Datenpunkten und $d$ Merkmalen. Oft liegt der größte Teil der Variation in nur wenigen Richtungen. Welche Einheitsrichtung erfasst die größte Varianz der Daten? Welcher $k$-dimensionale affine Unterraum approximiert die Punkte mit dem kleinsten quadratischen Fehler? Überraschenderweise liefert die abgeschnittene SVD der zentrierten Daten auf beide Fragen dieselbe Antwort.

<!--en:b9629589-->
**Zentrierung.** Wir wählen zunächst einen beliebigen $k$-dimensionalen Unterraum mit orthogonalem Projektor $\mathbf{P}$. Welche Verschiebung $\mathbf{c}$ minimiert den gesamten quadratischen Rekonstruktionsfehler $\sum_i\|(\mathbf{I}-\mathbf{P})(\mathbf{x}_i-\mathbf{c})\|^2$? Die Zielfunktion ist eine konvexe quadratische Funktion von $\mathbf{c}$. Ihr Gradient $-2(\mathbf{I}-\mathbf{P})\sum_i(\mathbf{x}_i-\mathbf{c})$ verschwindet beim Stichprobenmittelwert $\mathbf{c}=\bar{\mathbf{x}}$, unabhängig von $\mathbf{P}$. Der beste affine Unterraum verläuft also stets durch den Mittelwert. Deshalb *zentrieren* wir zuerst die Daten: $\tilde{\mathbf{X}}=\mathbf{X}-\mathbf{1}\bar{\mathbf{x}}^\top$. Ihre empirische Kovarianzmatrix ist die symmetrische, positiv semidefinite Matrix $\mathbf{C}=\tfrac1n\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$. (Der Faktor $\tfrac1{n-1}$ für eine erwartungstreue Schätzung skaliert alle Eigenwerte gleich. Die Hauptrichtungen und der von ihnen erklärte Varianzanteil bleiben erhalten.)

<!--en:8c725c8f-->
**Proposition (PCA über den Rayleigh-Quotienten).** *Unter allen Einheitsvektoren $\mathbf{w}$ erreicht die projizierte Varianz $\tfrac1n\|\tilde{\mathbf{X}}\mathbf{w}\|^2=\mathbf{w}^\top\mathbf{C}\mathbf{w}$ ihr Maximum beim Eigenvektor zum größten Eigenwert von $\mathbf{C}$. Dieser Vektor ist der erste rechte Singulärvektor $\mathbf{v}_1$ von $\tilde{\mathbf{X}}$. Die maximale Varianz beträgt $\sigma_1^2/n$.*

<!--en:bb618499-->
**Beweis.** Die Projektion eines zentrierten Punkts auf $\mathbf{w}$ liefert die skalare Koordinate $\tilde{\mathbf{x}}^\top\mathbf{w}$ mit empirischer Varianz $\tfrac1n\sum_i(\tilde{\mathbf{x}}_i^\top\mathbf{w})^2=\tfrac1n\|\tilde{\mathbf{X}}\mathbf{w}\|^2=\mathbf{w}^\top\mathbf{C}\mathbf{w}$. Nach der Proposition über den Rayleigh-Quotienten aus :numref:`subsec_mdl-rayleigh` maximiert der erste Eigenvektor von $\mathbf{C}$ diese quadratische Form. Ihr Maximalwert ist $\lambda_1(\mathbf{C})$. Nach :eqref:`eq_mdl-svd-gram` gilt $\mathbf{C}=\tfrac1n\mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top$. Die Eigenvektoren von $\mathbf{C}$ sind also die rechten Singulärvektoren $\mathbf{v}_i$ von $\tilde{\mathbf{X}}$, und die Eigenwerte sind $\sigma_i^2/n$. $\blacksquare$

<!--en:f94c8024-->
Unter den Einheitsvektoren orthogonal zu $\mathbf{v}_1$ maximiert $\mathbf{v}_2$ die Varianz. Setzen wir dies induktiv fort, erhalten wir die ersten $k$ *Hauptrichtungen* $\mathbf{v}_1,\ldots,\mathbf{v}_k$. Dabei erklärt die Komponente $i$ die Varianz $\sigma_i^2/n$. Betrachten wir nun den Rekonstruktionsfehler. Alle zentrierten Punkte auf $\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$ zu projizieren entspricht der Multiplikation von $\tilde{\mathbf{X}}$ mit $\mathbf{V}_k\mathbf{V}_k^\top$ von rechts. Da $\mathbf{V}^\top\mathbf{V}_k$ die ersten $k$ Spalten der Einheitsmatrix auswählt, gilt

<!--en:e55917e2-->
Die Projektion auf die ersten $k$ Hauptrichtungen entspricht also genau der auf Rang $k$ abgeschnittenen SVD der zentrierten Daten. Nach dem Satz von Eckart und Young für die Frobeniusnorm kann keine andere Projektion, ja keine andere Matrix vom Rang $k$, die zentrierten Punkte mit kleinerem quadratischem Fehler rekonstruieren. Die verworfene Varianz $\sum_{i>k}\sigma_i^2/n$ ist gleich dem mittleren quadratischen Rekonstruktionsfehler. Der erklärte Varianzanteil ist der Energieanteil :eqref:`eq_mdl-energy-ratio` von $\tilde{\mathbf{X}}$.

<!--en:ace4706c-->
:numref:`fig_mdl-la-pca` zeigt eine Wolke aus 300 Punkten mit Standardabweichungen 3 und 0.8 entlang der um $\pi/5=36^\circ$ gedrehten Achsen sowie ihre beiden Hauptachsen. In der Grundgesamtheit erfasst die erste Hauptrichtung $9/9.64\approx93.4\%$ der Varianz und fällt mit der langen Achse zusammen. Die folgende Codezelle prüft, wie genau eine endliche Stichprobe mit 300 Punkten diese Werte wiedergibt.

<!--en:45926ed7-->
![Wolke aus 300 Punkten um $(1,0.6)$ mit Standardabweichungen 3 und 0.8 entlang der um $36^\circ$ gedrehten Achsen. Die beiden Hauptachsen sind vom Mittelwert aus mit einer Länge von jeweils zwei Standardabweichungen $2\sigma_i/\sqrt{n}$ eingezeichnet. Die erste Achse zeigt in die Richtung der größten Varianz.](../img/mdl-la-pca.svg)

<!--en:eb56a887-->
Bei wiederholten Zufallsstichproben erklärt die erste Hauptrichtung etwa 91% bis 95% der Varianz und weicht nur um wenige Grad von der tatsächlichen langen Achse ab. Verzichten wir jedoch auf die Zentrierung, schwenkt der erste Singulärvektor zum Mittelwert der Wolke, sobald dessen Abstand vom Ursprung die Streuung der Punkte übersteigt. Er beschreibt dann, *wo* die Wolke liegt, statt *wie* ihre Punkte streuen. In Aufgabe :exref:`ex_mdl-svd-low-rank-centering-pca` untersuchen Sie diesen Effekt quantitativ.

<!--en:cb16e640-->
## Lösung linearer Gleichungssysteme mit der SVD

<!--en:a8b1ed70-->
### Pseudoinverse und kleinste Quadrate

<!--en:4e9c3e4a-->
Ein lineares Gleichungssystem $\mathbf{A}\mathbf{x}=\mathbf{b}$ mit mehr Gleichungen als Unbekannten hat im Allgemeinen keine exakte Lösung. Sind es weniger Gleichungen als Unbekannte, gibt es im konsistenten Fall unendlich viele Lösungen. In beiden Fällen suchen wir die Lösung nach der Methode der kleinsten Quadrate aus :numref:`sec_mdl-geometry-linear-algebraic-ops` (für den Zielvektor schreiben wir $\mathbf{b}$ statt $\mathbf{y}$),

<!--en:95b8d1f1-->
und wenn mehrere $\mathbf{x}$ dasselbe minimale Residuum erreichen, wählen wir die Lösung mit der kleinsten Norm.

<!--en:5f298f1e-->
Betrachten wir zunächst das einfachste unterbestimmte Beispiel, eine einzige Gleichung in zwei Unbekannten, $x_1+x_2=2$. Ihre Lösungen bilden eine Gerade, auf der das Residuum überall null ist. Welchen Punkt auf der Geraden wählen wir? Hier gilt $\mathbf{A}=(1\;\;1)$. Der einzige Singulärwert ungleich null ist $\sigma_1=\sqrt2$, der rechte Singulärvektor ist $\mathbf{v}_1=(1,1)/\sqrt2$, und es gilt $u_1=1$. Die Gleichung legt die Koordinate von $\mathbf{x}$ entlang $\mathbf{v}_1$ auf $\sqrt2$ fest. Die Koordinate in der Nullraumrichtung $\mathbf{v}_2=(1,-1)/\sqrt2$ hat dagegen keinen Einfluss auf das Residuum. Setzen wir diese freie Koordinate auf null, erhalten wir $(1,1)$, den Lotfußpunkt vom Ursprung auf die Gerade, mit Norm $\sqrt2$. Jede andere Lösung $(1+t,1-t)$ hat eine strikt größere quadrierte Norm $2+2t^2$ (:numref:`fig_mdl-la-min-norm`).

<!--en:b8a0c51e-->
![Kleinste Quadrate bei einer Gleichung mit zwei Unbekannten. Jeder Punkt der Geraden löst $x_1+x_2=2$. Die Pseudoinverse liefert den Punkt mit dem kleinsten Abstand zum Ursprung, den Lotfußpunkt.](../img/mdl-la-min-norm.svg)

<!--en:1f796974-->
Die allgemeine Regel bildet den Kehrwert jedes von null verschiedenen Singulärwerts und transponiert die orthogonalen Faktoren. Die *Moore-Penrose-Pseudoinverse* von $\mathbf{A}$ lautet

<!--en:63f78361-->
Dabei hat $\boldsymbol{\Sigma}^{+}$ die Form $n\times m$, also die Form der Transponierten von $\boldsymbol{\Sigma}$. Geometrisch kehrt $\mathbf{A}^{+}$ jede umkehrbare Streckung um und bildet den linken Nullraum auf null ab. Für die Scherung gilt $\mathbf{T}^{+}=\mathbf{T}^{-1}=\left(\begin{smallmatrix}1&-1\\0&1\end{smallmatrix}\right)$. Für $\mathbf{A}=(1\;\;1)$ gilt $\mathbf{A}^{+}=\mathbf{v}_1\sigma_1^{-1}u_1=(\tfrac12,\tfrac12)^\top$, also $\mathbf{A}^{+}\cdot2=(1,1)$.

<!--en:81e3b766-->
**Proposition (Kleinste-Quadrate-Lösung minimaler Norm).** *Unter allen Minimierern von $\|\mathbf{A}\mathbf{x}-\mathbf{b}\|$ ist $\hat{\mathbf{x}}=\mathbf{A}^{+}\mathbf{b}$ die eindeutige Lösung mit der kleinsten Norm.*

<!--en:e0de77af-->
**Beweis.** Wir wechseln zu den Basen aus Singulärvektoren und setzen $\mathbf{y}=\mathbf{V}^\top\mathbf{x}$ und $\mathbf{c}=\mathbf{U}^\top\mathbf{b}$. Da $\mathbf{U}$ orthogonal ist, bleiben euklidische Normen erhalten. Somit gilt

<!--en:0e7f86a1-->
Durch die Drehung in die Singulärbasen zerfällt das System in unabhängige skalare Gleichungen. Die erste Summe minimieren wir gliedweise mit der Wahl $y_i=c_i/\sigma_i$ für $i\le r$. Die zweite Summe ist konstant. Sie entspricht der quadrierten Norm der Komponente von $\mathbf{b}$ außerhalb des Spaltenraums, die keine Wahl von $\mathbf{x}$ verkleinern kann. Die Koordinaten $y_i$ für $i>r$ gehen nicht in das Residuum ein. Wegen $\|\mathbf{x}\|=\|\mathbf{y}\|$ erhalten wir den eindeutigen Minimierer kleinster Norm, wenn wir sie auf null setzen. Diese Wahl entspricht $\mathbf{y}=\boldsymbol{\Sigma}^{+}\mathbf{c}$ und in den ursprünglichen Koordinaten $\hat{\mathbf{x}}=\mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top\mathbf{b}=\mathbf{A}^{+}\mathbf{b}$. $\blacksquare$

<!--en:35587565-->
In den Singulärbasen betreffen die Minimierung des Residuums und die Minimierung der Norm getrennte Koordinaten ($i\le r$ beziehungsweise $i>r$). Daher erreichen wir beide Ziele gleichzeitig. Ist $\mathbf{A}$ quadratisch und invertierbar, so ist jeder Singulärwert $\sigma_i$ positiv und es gilt $\mathbf{A}^{+}=\mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top=\mathbf{A}^{-1}$. Dieselbe Formel zeigt auch die größte Gefahr bei der Methode der kleinsten Quadrate. Teilen wir durch ein winziges $\sigma_i$, verstärken wir das Rauschen in $\mathbf{b}$ entlang $\mathbf{u}_i$ um den Faktor $1/\sigma_i$.

<!--en:a18ef6a0-->
Zwei gebräuchliche Verfahren begrenzen den Einfluss dieser instabilen Richtungen. Die *Ridge-Regression* minimiert $\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2+\lambda\|\mathbf{x}\|^2$ für $\lambda>0$ und liefert $\mathbf{x}_\lambda=(\mathbf{A}^\top\mathbf{A}+\lambda\mathbf{I})^{-1}\mathbf{A}^\top\mathbf{b}$. Die *abgeschnittene Pseudoinverse* $\mathbf{x}_k$ behält lediglich die ersten $k$ Terme der Singulärwertzerlegung von $\mathbf{A}^{+}\mathbf{b}$ bei. Setzen wir die SVD ein und beachten $\mathbf{A}^\top\mathbf{A}+\lambda\mathbf{I}=\mathbf{V}(\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}+\lambda\mathbf{I})\mathbf{V}^\top$, so erhalten wir für beide Lösungen eine Darstellung als Filter:

<!--en:3b731726-->
Beide Verfahren gehen von den Termen $\mathbf{u}_i^\top\mathbf{b}/\sigma_i$ der unregularisierten Pseudoinversen aus. Die Ridge-Regression skaliert die Komponente $i$ mit dem glatten Filterfaktor $\sigma_i^2/(\sigma_i^2+\lambda)$. Dieser bleibt für $\sigma_i^2\gg\lambda$ nahe bei 1 und nähert sich für $\sigma_i^2\ll\lambda$ der 0. Das Abschneiden verwendet eine Sprungfunktion mit den Werten 1 und 0. Für $\lambda=1$ erhalten die Singulärwerte $3$, $1$ und $0.3$ die Filterfaktoren $0.9$, $0.5$ und $0.083$. Beim Training eines linearen Modells mit quadratischem Verlust entspricht Gewichtszerfall (:numref:`sec_weight_decay`) genau der Ridge-Regression. Die Filterfaktoren zeigen, welche Richtungen der Gewichtszerfall dämpft.

<!--en:52285a07-->
**Die Normalgleichungen.** Der klassische Lehrbuchansatz für die Methode der kleinsten Quadrate löst die Normalgleichungen $\mathbf{A}^\top\mathbf{A}\mathbf{x}=\mathbf{A}^\top\mathbf{b}$. Für jede Matrix $\mathbf{A}$ sind ihre Lösungen genau die Kleinste-Quadrate-Minimierer, denn $\mathbf{A}^\top(\mathbf{A}\mathbf{x}-\mathbf{b})=\mathbf{0}$ besagt, dass das Residuum orthogonal zum Spaltenraum ist. Die Lösung ist eindeutig, wenn $\mathbf{A}$ vollen Spaltenrang hat. Numerisch ist es jedoch ungünstig, $\mathbf{A}^\top\mathbf{A}$ zu bilden, da sich dabei die Konditionszahl quadriert (:numref:`subsec_mdl-condition-number`). Moderne Bibliotheken lösen Kleinste-Quadrate-Probleme daher mit der SVD oder einer QR-Zerlegung.

<!--en:e3b138d3-->
Prüfen wir die Pseudoinverse an einem überbestimmten und einem unterbestimmten System. Im ersten passen wir eine Gerade $b\approx x_1+x_2 t$ an die vier Beobachtungen $\mathbf{b}=(6,5,7,10)$ an den Stellen $t=1,2,3,4$ an. Das zweite besteht aus unserer einzelnen Gleichung $x_1+x_2=2$.

<!--en:b54fea9d-->
Die Pseudoinverse und `np.linalg.lstsq` liefern dieselbe Ausgleichsgerade $3.5+1.4t$ mit Residuum $\sqrt{4.2}\approx2.049$. Für die einzelne Gleichung liefert die Pseudoinverse den Lotfußpunkt $(1,1)$.

<!--en:57bea96c-->
### Die Konditionszahl

<!--en:c66e5bf6-->
In der Praxis stammen die Einträge eines linearen Gleichungssystems aus verrauschten Messungen oder aus Rechnungen mit endlicher Genauigkeit und Rundung. Um eine berechnete Lösung beurteilen zu können, müssen wir wissen, wie stark sich eine kleine Änderung der Daten auf sie auswirkt. Schon ein $2\times2$-System zeigt, wie groß die Verstärkung sein kann. Betrachten wir $\mathbf{K}\mathbf{x}=\mathbf{b}$ mit $\mathbf{K}=\left(\begin{smallmatrix}1&1\\1&1.0001\end{smallmatrix}\right)$ und $\mathbf{b}=(2,2.0001)$. Die exakte Lösung lautet $\mathbf{x}=(1,1)$. Ändern wir nun $b_2$ um nur $10^{-4}$ auf $2.0002$, springt die Lösung auf $(0,2)$. Die relative Störung der Daten beträgt etwa drei Teile in $10^5$, und doch ändert sich die Lösung grundlegend. Die Singulärwerte von $\mathbf{K}$, ungefähr $2.00005$ und $5\times10^{-5}$, erklären diesen Effekt.

<!--en:086baca6-->
Die *Konditionszahl* einer Matrix $\mathbf{A}$ vom Rang $r$ ist das Verhältnis ihres größten zu ihrem kleinsten von null verschiedenen Singulärwert,

<!--en:7179f882-->
Geometrisch misst $\kappa(\mathbf{A})$, wie unterschiedlich stark $\mathbf{A}$ einzelne Richtungen streckt. Für $\mathbf{K}$ beträgt sie etwa $4\times10^4$, für die Scherung nur $\varphi^2\approx2.618$. Ist $\mathbf{A}$ symmetrisch und positiv definit, stimmen Singulärwerte und Eigenwerte überein. Dann gilt $\kappa=\lambda_{\max}/\lambda_{\min}$, und wir erhalten die Konditionszahl für quadratische Funktionen mit schüsselförmigem Graphen aus :numref:`subsec_mdl-rayleigh`. Der folgende Satz zeigt, dass $\kappa(\mathbf{A})$ beim Lösen eines linearen Gleichungssystems genau der größtmögliche Verstärkungsfaktor für relative Fehler ist.

<!--en:eccb035e-->
**Proposition (Fehlerverstärkung).** *Sei $\mathbf{A}$ quadratisch und invertierbar. Es gelte $\mathbf{A}\mathbf{x}=\mathbf{b}$ mit $\mathbf{b}\ne\mathbf{0}$, und eine Störung $\delta\mathbf{b}$ verschiebe die Lösung auf $\mathbf{x}+\delta\mathbf{x}$. Dann gilt*

<!--en:1622e42d-->
*und zu jeder Matrix $\mathbf{A}$ gibt es Vektoren $\mathbf{b}$ und $\delta\mathbf{b}$, für die Gleichheit gilt.*

<!--en:798f848f-->
**Beweis.** Aus $\mathbf{A}\,\delta\mathbf{x}=\delta\mathbf{b}$ folgt $\delta\mathbf{x}=\mathbf{A}^{-1}\delta\mathbf{b}$. Der größte Singulärwert von $\mathbf{A}^{-1}=\mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top$ ist $1/\sigma_n$. Daher liefert :eqref:`eq_mdl-sigma1-variational` die Schranke $\|\delta\mathbf{x}\|\le\sigma_n^{-1}\|\delta\mathbf{b}\|$. Ebenso ergibt $\mathbf{b}=\mathbf{A}\mathbf{x}$ die Schranke $\|\mathbf{b}\|\le\sigma_1\|\mathbf{x}\|$. Zusammen folgt aus den beiden Ungleichungen

<!--en:37472fe0-->
In beiden Schranken gilt gleichzeitig Gleichheit, wenn $\mathbf{b}=\mathbf{u}_1$ (dann ist $\mathbf{x}=\sigma_1^{-1}\mathbf{v}_1$ für eine rechte Seite der Norm eins möglichst kurz) und $\delta\mathbf{b}=\varepsilon\mathbf{u}_n$ (dann ist $\delta\mathbf{x}=\varepsilon\sigma_n^{-1}\mathbf{v}_n$ bei einer Störung der Größe $\varepsilon$ möglichst groß). $\blacksquare$

<!--en:d0aa2913-->
Der relative Fehler in den Daten verstärkt sich also höchstens um den Faktor $\kappa$. Bei $\kappa=10^4$ kann ein relativer Fehler von $10^{-8}$ in $\mathbf{b}$ zu einem Fehler von $10^{-4}$ in $\mathbf{x}$ anwachsen. Dabei gehen vier Stellen Genauigkeit verloren. In unserem $2\times2$-Beispiel beträgt die relative Änderung von $\mathbf{b}$ genau $10^{-4}/\|\mathbf{b}\|\approx3.5\times10^{-5}$. Die Schranke erlaubt somit eine relative Änderung um $1.41$ in $\mathbf{x}$. Tatsächlich ergibt sich $1.0$. Von der Länge der Störung $(0,10^{-4})$ entfallen $1/\sqrt2$ auf die am stärksten verstärkte Richtung $\mathbf{u}_2\approx(1,-1)/\sqrt2$, während $\mathbf{b}$ fast genau entlang $\mathbf{u}_1$ zeigt.

<!--en:e21deea8-->
Im günstigsten Fall ist die Matrix orthogonal. Dann sind alle Singulärwerte gleich 1 und es gilt $\kappa=1$. Drehungen und Spiegelungen verstärken Fehler also nie. Deshalb ist das Rechnen mit den orthogonalen Faktoren $\mathbf{U}$ und $\mathbf{V}$ numerisch unbedenklich. Aus demselben Grund erhält eine orthogonal initialisierte Gewichtsmatrix die Norm jedes Eingabevektors. Am anderen Extrem steht die Bildung der Gram-Matrix, bei der sich die Konditionszahl quadriert:

<!--en:37296b6e-->
Denn :eqref:`eq_mdl-svd-gram` ist die SVD von $\mathbf{A}^\top\mathbf{A}$ mit den Singulärwerten $\sigma_i^2$ (für jede Matrix $\mathbf{A}$ beliebigen Rangs). Bei der Scherung steigt $\kappa$ von $\varphi^2\approx2.62$ auf $\varphi^4\approx6.85$. Die folgende Codezelle prüft unser $2\times2$-Beispiel und vergleicht $\kappa(\mathbf{A})$ mit $\kappa(\mathbf{A}^\top\mathbf{A})$ für die obige Matrix zur Geradenanpassung.

<!--en:7cd5297c-->
Die Ausgabe bestätigt unsere Handrechnung. Die Lösung springt von $(1,1)$ auf $(0,2)$, mit $\kappa(\mathbf{K})\approx40{,}002$. Schon bei der kleinen $4\times2$-Matrix für die Geradenanpassung gilt $\kappa=7.47$. Die Normalgleichungen hätten daher bereits die Konditionszahl $\kappa^2=55.8$.

<!--en:482971da-->
**Kondition und Form der Niveaumengen.** Bei der Optimierung zeigt sich eine große Konditionszahl direkt an lang gestreckten Niveaumengen. Hat die Matrix einer Kleinste-Quadrate-Zielfunktion $\tfrac12\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2$ vollen Spaltenrang, so lautet die Hesse-Matrix $\mathbf{H}=\mathbf{A}^\top\mathbf{A}$. Die Niveaumengen der Zielfunktion sind Ellipsoide mit Halbachsenlängen proportional zu $1/\sigma_i$ in Richtung der $\mathbf{v}_i$. Das Achsenverhältnis der Konturen ist daher $\kappa(\mathbf{A})$, während die Hesse-Matrix selbst die Konditionszahl $\kappa(\mathbf{A})^2$ hat. In einem so schmalen Tal läuft der Gradientenabstieg im Zickzack zwischen den steilen Flanken hin und her und kommt entlang der Talsohle nur langsam voran (:numref:`fig_mdl-la-condition`). Die Konvergenzrate hängt von $\kappa(\mathbf{H})$ ab. Wir leiten sie in :numref:`sec_mdl-gradient-based-optimization` her.

<!--en:4ca3db05-->
![Gradientenabstieg bei einer gut konditionierten quadratischen Funktion ($\kappa(\mathbf{H})=1.3$, fast kreisförmige Höhenlinien, nahezu gerader Pfad) und einer schlecht konditionierten Funktion ($\kappa(\mathbf{H})=18$, Zickzackpfad quer durch das schmale Tal). Die Titel geben die Konditionszahl $\kappa(\mathbf{H})$ der Hesse-Matrix an. Das Achsenverhältnis der Höhenlinien ist ihre Quadratwurzel, im zweiten Teilbild etwa $4.2$.](../img/mdl-la-condition.svg)

<!--en:9fe4db5b-->
## Die SVD im Deep Learning

<!--en:cd0c9c2a-->
Drei wiederkehrende Fragen beim Training großer Modelle führen direkt auf die Singulärwerte einer Gewichtsmatrix. Wie viele Parameter braucht ein Fine-Tuning-Adapter? Wie konstruieren wir eine Aktualisierung, die jede Richtung im Gewichtsraum gleich behandelt? Wie stark kann eine einzelne Schicht ihre Eingabe strecken?

<!--en:ee942c44-->
### Niedrigrangige Aktualisierungen

<!--en:c3626c51-->
Beim Fine-Tuning eines großen vortrainierten Modells jeden Parameter zu aktualisieren kostet viel Speicher und Rechenzeit. LoRA :cite:`Hu.Shen.Wallis.ea.2021` vermeidet diese Kosten. Das Verfahren hält jede vortrainierte Gewichtsmatrix $\mathbf{W}\in\mathbb{R}^{m\times n}$ fest und lernt nur eine Korrektur vom Rang $k$ in der Form $\Delta\mathbf{W}=\mathbf{B}\mathbf{A}$ mit $\mathbf{B}\in\mathbb{R}^{m\times k}$ und $\mathbf{A}\in\mathbb{R}^{k\times n}$ (hier bezeichnet $\mathbf{A}$ wie in der Originalarbeit die $k\times n$-Projektion in den kleineren Raum, nicht die allgemeine Matrix der vorigen Unterabschnitte). Der Adapter benötigt $k(m+n)$ trainierbare Parameter statt $mn$. In einer $4096\times4096$-Schicht mit $k=8$ sinkt ihre Anzahl von $16{,}777{,}216$ auf $8\cdot8192=65{,}536$, also auf nur 0.39%. Wie :numref:`fig_mdl-la-lora` zeigt, wirken die feste Gewichtsmatrix und der parallele Pfad mit niedrigem Rang auf dieselbe Eingabe. Ihre Ausgaben werden addiert.

<!--en:f61d605c-->
![LoRA-Architektur für eine Schicht. Die vortrainierte Matrix $\mathbf{W}$ (alle $mn = 16.8$ Mio. Parameter für $m=n=4096$) bleibt fest. Ein paralleler Pfad komprimiert die Eingabe mit einer Projektion auf $k=8$ Dimensionen durch $\mathbf{A}$ und erweitert sie anschließend wieder mit $\mathbf{B}$. Damit sind nur $k(m+n) \approx 65.5$ Tausend Parameter (etwa $0.39\%$) trainierbar. Die Ausgaben beider Zweige addieren sich: $\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$. Nach dem Training lässt sich $\mathbf{B}\mathbf{A}$ in $\mathbf{W}$ integrieren, sodass beim Einsatz keine Zusatzkosten entstehen.](../img/mdl-la-lora.svg)

<!--en:dcbb3191-->
Was sagt der Satz von Eckart und Young hier aus, und was sagt er *nicht*? Angenommen, vollständiges Fine-Tuning ergäbe eine Aktualisierung mit den Singulärwerten $\sigma_1\ge\sigma_2\ge\cdots$. Dann garantiert der Satz, dass kein Adapter vom Rang $k$ diese Aktualisierung mit einem Fehler unter $\sigma_{k+1}$ in der Spektralnorm oder unter $(\sum_{i>k}\sigma_i^2)^{1/2}$ in der Frobeniusnorm approximieren kann. Er legt *nicht* fest, welchen Adapter der Gradientenabstieg tatsächlich lernt. LoRA ist erfolgreich, wenn *irgendeine* niedrigrangige Aktualisierung einen kleinen Verlust erreicht, auch wenn sie nicht die beste Approximation vom Rang $k$ an die vollständige Fine-Tuning-Aktualisierung ist. Dass solche niedrigrangigen Lösungen für viele nachgelagerte Aufgaben existieren, ist die empirische Entdeckung hinter LoRA. In Aufgabe :exref:`ex_mdl-svd-low-rank-weyl-perturbation` bestimmen Sie eine Schranke dafür, wie stark ein Adapter vom Rang $k$ die Singulärwerte von $\mathbf{W}$ verändern kann.

<!--en:5b3d41ca-->
### Orthogonalisierte Aktualisierungen

<!--en:035c65b5-->
Der Optimierer Muon :cite:`Jordan.Jin.Boza.ea.2024` aktualisiert eine Gewichtsmatrix anhand seiner Momentum-Matrix $\mathbf{M}$. Als Richtung verwendet er den *Polarfaktor* von $\mathbf{M}$. Diese Matrix behält die Singulärvektoren von $\mathbf{M}$ bei und setzt jeden von null verschiedenen Singulärwert auf 1. :numref:`sec_muon` erläutert die Motivation und entwickelt den vollständigen Optimierer. Hier erarbeiten wir seine beiden Grundlagen aus der linearen Algebra. Was ist der Polarfaktor, und wie berechnen wir ihn allein durch Matrixmultiplikationen?

<!--en:72c06407-->
**Die Polarzerlegung.** Für eine quadratische Matrix $\mathbf{A}$ fügen wir $\mathbf{V}^\top\mathbf{V}=\mathbf{I}$ in die SVD ein. So zerfällt die Matrix in einen orthogonalen Faktor und eine symmetrische Streckung:

<!--en:2fea36e8-->
Dabei ist $\mathbf{Q}=\mathbf{U}\mathbf{V}^\top$ als Produkt orthogonaler Matrizen orthogonal. Die Matrix $\mathbf{P}=\mathbf{V}\boldsymbol{\Sigma}\mathbf{V}^\top$ ist symmetrisch und positiv semidefinit mit den Eigenwerten $\sigma_i$. Jede quadratische Matrix beschreibt somit eine symmetrische Streckung gefolgt von einer orthogonalen Abbildung. Das entspricht der Polarform $z=e^{i\theta}|z|$ einer komplexen Zahl. Dabei muss $\mathbf{Q}$ keine reine Drehung sein. Ist $\mathbf{A}$ invertierbar, so gilt $\det\mathbf{Q}=\operatorname{sign}\det\mathbf{A}$. Also enthält $\mathbf{Q}$ eine Spiegelung, wenn $\det\mathbf{A}<0$ gilt. Für die Scherung $\mathbf{T}$ liefert :eqref:`eq_mdl-shear-svd` die Faktoren $\mathbf{Q}=\mathbf{R}_{31.7^\circ}\mathbf{R}_{58.3^\circ}^\top=\mathbf{R}_{-26.6^\circ}=\tfrac{1}{\sqrt5}\left(\begin{smallmatrix}2&1\\-1&2\end{smallmatrix}\right)$ (wegen $\tan26.6^\circ=\tfrac12$) und $\mathbf{P}=\tfrac{1}{\sqrt5}\left(\begin{smallmatrix}2&1\\1&3\end{smallmatrix}\right)$. Ausmultiplizieren bestätigt $\mathbf{Q}\mathbf{P}=\tfrac15\left(\begin{smallmatrix}5&5\\0&5\end{smallmatrix}\right)=\mathbf{T}$ und $\mathbf{P}^2=\left(\begin{smallmatrix}1&1\\1&2\end{smallmatrix}\right)=\mathbf{T}^\top\mathbf{T}$.

<!--en:fadf1e59-->
Für eine rechteckige Matrix $\mathbf{M}$ mit dünner SVD $\mathbf{M}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ ist der Polarfaktor ebenfalls $\mathbf{U}\mathbf{V}^\top$. Er hat dieselben Singulärvektoren wie $\mathbf{M}$, und alle seine Singulärwerte sind 1. Folglich sind seine Spalten (für $m\ge n$) oder seine Zeilen (für $m\le n$) orthonormal. Ist $\mathbf{M}$ rangdefizient ($r<\min(m,n)$), so ist der Polarfaktor nicht eindeutig. Die kanonische Wahl ist dann $\mathbf{U}_r\mathbf{V}_r^\top$ aus der kompakten SVD. In Aufgabe :exref:`ex_mdl-svd-low-rank-polar-procrustes` zeigen Sie, dass der Faktor $\mathbf{U}\mathbf{V}^\top$ auch eine der orthogonalen Matrizen mit dem kleinsten Abstand zur quadratischen Matrix $\mathbf{A}$ in der Frobeniusnorm ist.

<!--en:5b1191e1-->
**Ungerade Polynome einer Matrix.** Für jede Gewichtsmatrix in jedem Trainingsschritt eine vollständige SVD zu berechnen wäre zu langsam. Die Singulärwerte lassen sich jedoch allein durch Matrixmultiplikationen transformieren, wobei die Singulärvektoren erhalten bleiben.

<!--en:1356a816-->
**Proposition (Wirkung ungerader Polynome auf die Singulärwerte).** *Sei $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ eine dünne SVD und $p(\sigma)=\sum_j c_j\sigma^{2j+1}$ ein ungerades Polynom. Dann gilt*

<!--en:7310fd38-->
**Beweis.** Wegen $\mathbf{X}^\top\mathbf{X}=\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top$ erhalten wir durch Potenzieren $(\mathbf{X}^\top\mathbf{X})^j=\mathbf{V}\boldsymbol{\Sigma}^{2j}\mathbf{V}^\top$ und durch Multiplikation von links mit $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ den Ausdruck $\mathbf{X}(\mathbf{X}^\top\mathbf{X})^j=\mathbf{U}\boldsymbol{\Sigma}^{2j+1}\mathbf{V}^\top$. Summation über $j$ schließt den Beweis ab. $\blacksquare$

<!--en:9301d807-->
Jedes ungerade Matrixpolynom lässt also $\mathbf{U}$ und $\mathbf{V}$ unverändert und wendet die skalare Funktion $p$ einzeln auf jeden Singulärwert an. Für den Polarfaktor $\mathbf{U}\mathbf{V}^\top$ brauchen wir somit ein ungerades Polynom $p$, dessen wiederholte Anwendung jeden positiven Singulärwert gegen 1 führt.

<!--en:daa8383c-->
**Newton-Schulz-Verfahren.** Die klassische *Newton-Schulz-Iteration* $\mathbf{X}\leftarrow\tfrac12(3\mathbf{X}-\mathbf{X}\mathbf{X}^\top\mathbf{X})$ wendet das kubische Polynom $p(\sigma)=\tfrac12(3\sigma-\sigma^3)$ auf jeden Singulärwert an. Seine Fixpunkte sind $0$ und $\pm1$. Auf $(0,1]$ erhöht $p$ jeden Wert, ohne 1 zu überschreiten, denn es gilt $p(\sigma)-\sigma=\tfrac12\sigma(1-\sigma^2)\ge0$ und $p$ erreicht sein Maximum $p(1)=1$. Tatsächlich bildet $p$ das gesamte Intervall $(0,\sqrt3)$ in $(0,1]$ ab. Von jedem Startwert in diesem *unmittelbaren Einzugsgebiet* von 1 aus konvergiert die Iteration gegen 1. Größere Startwerte können sich unregelmäßig verhalten (Aufgabe :exref:`ex_mdl-svd-low-rank-newton-schulz-basin`). Skalieren wir die Anfangsmatrix gemäß $\mathbf{X}_0=\mathbf{M}/\|\mathbf{M}\|_F$, so beginnt jeder positive Singulärwert in $(0,1]$, denn es gilt $\|\mathbf{M}\|_F\ge\sigma_1(\mathbf{M})$. Singulärwerte von null bleiben null (wegen $p(0)=0$). Für eine rangdefiziente Matrix $\mathbf{M}$ konvergiert die Iteration daher gegen $\mathbf{U}_r\mathbf{V}_r^\top$.

<!--en:9fbfab92-->
Wie viele Iterationen braucht das Verfahren? Nahe 0 gilt $p(\sigma)\approx\tfrac32\sigma$. Ein kleiner Singulärwert $s$ wächst daher pro Schritt um den Faktor $3/2$ und erreicht nach ungefähr $\log_{3/2}(1/s)$ Schritten die Größenordnung eins. Nahe 1 ergibt die Entwicklung $p(1-e)=1-\tfrac32e^2+\tfrac12e^3$. Der Fehler quadriert sich also ungefähr bei jedem Schritt und sinkt in etwa vier weiteren Schritten von der Größenordnung eins auf $10^{-6}$. Für beide Phasen zusammen erwarten wir rund $\log_{3/2}(1/s_{\min})+4$ Schritte. Entscheidend ist der kleinste normierte Singulärwert $s_{\min}=\sigma_{\min}(\mathbf{M})/\|\mathbf{M}\|_F$. Prüfen wir diese grobe Schätzung an einer zufälligen $6\times4$-Matrix, die für das Momentum steht.

<!--en:099bde0f-->
Die tatsächliche Schrittzahl weicht um höchstens eine Iteration von der Vorhersage ab. Der Abstand des Ergebnisses zum exakten Polarfaktor aus einer vollständigen SVD beträgt höchstens $10^{-5}$. Da $s_{\min}$ von Stichprobe zu Stichprobe variiert, schwankt auch die Schrittzahl. Bei einer nahezu rangdefizienten Matrix $\mathbf{M}$ dauert die Iteration länger. In der Praxis ersetzt Muon das kubische Polynom durch ein abgestimmtes Polynom fünften Grades. Es verzichtet zugunsten höherer Geschwindigkeit auf exakte Konvergenz gegen 1 (:numref:`subsec_muon-newton-schulz`).

<!--en:e28da1fa-->
### Messung und Begrenzung des Spektrums

<!--en:09391158-->
Wie stark kann eine lineare Schicht ihre Eingabe strecken? Der größte Singulärwert beantwortet unsere dritte Frage direkt. Das gesamte Singulärwertspektrum zeigt zudem, ob sich eine Matrix gut durch eine Matrix niedrigen Rangs approximieren lässt. Moderne Trainingsverfahren begrenzen, untersuchen und approximieren daher Singulärwerte, ohne den Aufwand einer vollständigen SVD in Kauf zu nehmen.

<!--en:87012e6f-->
**Spektrale Normalisierung.** Nach :eqref:`eq_mdl-sigma1-variational` erfüllt eine lineare Schicht $\mathbf{x}\mapsto\mathbf{W}\mathbf{x}$ die Ungleichung $\|\mathbf{W}\mathbf{x}-\mathbf{W}\mathbf{x}'\|\le\sigma_1\|\mathbf{x}-\mathbf{x}'\|$. Gleichheit gilt, wenn $\mathbf{x}-\mathbf{x}'$ in Richtung $\mathbf{v}_1$ liegt. Die kleinste Konstante, die die Verstärkung von Eingabedifferenzen durch eine Abbildung beschränkt, heißt *Lipschitz-Konstante*. Hier ist die Lipschitz-Konstante also $\sigma_1=\|\mathbf{W}\|_2$. Teilen wir $\mathbf{W}$ durch $\sigma_1$, begrenzen wir die Streckung der Schicht auf 1. Das stabilisiert das Training von GAN-Diskriminatoren :cite:`Miyato.Kataoka.Koyama.ea.2018`. Anstelle einer vollständigen SVD schätzen wir $\sigma_1$ durch Potenziteration für $\mathbf{W}^\top\mathbf{W}$ (:numref:`subsec_mdl-power-iteration`). In der Praxis wechseln wir zwischen $\mathbf{u}\leftarrow\mathbf{W}\mathbf{v}$ und $\mathbf{v}\leftarrow\mathbf{W}^\top\mathbf{u}$ und normieren nach jedem Halbschritt. Zusammen entsprechen diese Schritte einer Multiplikation mit $\mathbf{W}^\top\mathbf{W}$, wobei $\|\mathbf{W}\mathbf{v}\|$ als Schätzwert für $\sigma_1$ dient. Mit jeder Runde nähert sich $\mathbf{v}$ weiter an $\mathbf{v}_1$ an. Dabei schrumpft der Fehler etwa um den Faktor $(\sigma_2/\sigma_1)^2$. :citet:`Miyato.Kataoka.Koyama.ea.2018` zeigten empirisch, dass eine Runde pro Trainingsschritt genügt, wenn wir mit dem Vektor aus dem vorigen Schritt beginnen. (Bei einer Faltungsschicht liefert das Umformen des Faltungskerns in eine zweidimensionale Matrix einen Näherungswert für die Spektralnorm der Faltung, jedoch im Allgemeinen nicht den exakten Wert.)

<!--en:c7346e11-->
**Das Spektrum messen.** Unsere Angabe von 0.39% für LoRA beruhte auf der Annahme, dass Rang 8 genügt. Wie gut eine niedrigrangige Approximation ist, hängt allein davon ab, wie rasch die Singulärwerte abfallen. Deshalb sollten wir sie vor dem Abschneiden messen. Die folgende Codezelle vergleicht zwei $400\times500$-Matrizen, eine gaußsche Zufallsmatrix ohne jede Struktur und unser Katzenfoto nach Abzug des mittleren Pixelwerts.

<!--en:894bccc4-->
Der Unterschied ist deutlich. Die Singulärwerte der gaußschen Matrix verteilen sich ohne steilen Abfall über das Marchenko-Pastur-Intervall $[\sqrt n-\sqrt m,\sqrt n+\sqrt m]\approx[2.4,42.4]$ aus :numref:`subsec_mdl-eckart-young`. Für 95% ihrer Energie brauchen wir etwa Rang 270 von 400. Ein Adapter mit niedrigem Rang hätte damit sogar *mehr* Parameter als die ursprüngliche Matrix. Beim zentrierten Foto erreichen wir 95% der Energie bereits bei Rang 8. Das entspricht einem Adapter mit weniger als 4% der Parameter. Keine der beiden Matrizen enthält trainierte Gewichte eines neuronalen Netzes. Das Paar veranschaulicht lediglich die beiden Extreme. Bei LoRA kommt es auf die aufgabenspezifische Aktualisierung an. Wie ihr Spektrum aussieht, müssen wir messen.

<!--en:7bff6add-->
**Randomisierte SVD.** Ist eine Matrix für eine direkte Zerlegung zu groß, so berechnet die *randomisierte SVD* :cite:`Halko.Martinsson.Tropp.2011` eine genaue Approximation vom Rang $k$ mit wenigen Matrix-Vektor-Produkten mit $\mathbf{A}$. Den Kern bildet ein *Verfahren zur Bestimmung des Bildraums* (engl. range finder). Wir multiplizieren $\mathbf{A}$ mit einer gaußschen Zufallsmatrix $\boldsymbol{\Omega}$ mit etwas mehr als $k$ Spalten, orthonormalisieren die Spalten von $\mathbf{A}\boldsymbol{\Omega}$ zu $\mathbf{Q}$ und berechnen die SVD der wesentlich kleineren Matrix $\mathbf{Q}^\top\mathbf{A}$. Fallen die Singulärwerte ab, dominieren die ersten Singulärrichtungen jede zufällige Linearkombination der Spalten von $\mathbf{A}$. Daher erfasst $\mathbf{Q}$ sie mit hoher Wahrscheinlichkeit. In Aufgabe :exref:`ex_mdl-svd-low-rank-randomized-range-finder` implementieren Sie diesen Algorithmus mit einer Verfeinerung durch Potenziteration für langsam abfallende Spektren. :numref:`sec_mdl-linear-algebra-further-reading` ordnet ihn in den größeren Zusammenhang randomisierter Skizzenverfahren ein.

<!--en:023afb51-->
Behalten wir zum Schluss drei Grenzen der SVD im Blick. Erstens erfordert eine vollständige SVD $O(mn\min(m,n))$ Operationen (ungefähr $4096^3\approx7\times10^{10}$ für eine einzelne $4096\times4096$-Gewichtsmatrix). Deshalb verwenden praktische Verfahren wie die Newton-Schulz-Iteration und die randomisierte Bestimmung des Bildraums nur Matrixprodukte. Zweitens hängt es von den Daten ab, ob ein niedriger Rang genügt. Wie die gaußsche Matrix zeigt, gibt es dafür keine allgemeine Garantie. Drittens erfasst die SVD ausschließlich *lineare* Struktur. Punkte auf einem Kreis in der Ebene bilden eine Datenmatrix vom Rang 2, obwohl der Kreis selbst eindimensional ist. Um gekrümmte Funktionsgraphen und nichtlineare Strukturen zu verstehen, brauchen wir Ableitungen. Ihnen wenden wir uns im nächsten Kapitel zu.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:4eca3bac-->
Für jede reelle Matrix $\mathbf{A}$ gibt es unabhängig von ihrer Form eine Orthonormalbasis $\{\mathbf{v}_i\}$ des Eingaberaums und eine Orthonormalbasis $\{\mathbf{u}_i\}$ des Ausgaberaums, zwischen denen sie durch nichtnegative Streckfaktoren wirkt. Es gilt $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ mit $\sigma_1\ge\sigma_2\ge\cdots\ge0$. Die Zerlegung folgt aus dem Spektralsatz für die Gram-Matrix $\mathbf{A}^\top\mathbf{A}$ mit $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$.

<!--en:4d7b1626-->
Aus dieser einen geordneten Liste von Singulärwerten ergeben sich fünf zentrale Größen. Der Rang ist die Anzahl der positiven $\sigma_i$. Spektralnorm und Frobeniusnorm ergeben sich aus $\|\mathbf{A}\|_2=\sigma_1$ und $\|\mathbf{A}\|_F^2=\sum_i\sigma_i^2$. Die optimale Approximation vom Rang $k$ liefert die abgeschnittene SVD $\mathbf{A}_k$ nach dem Satz von Eckart und Young. Bei zentrierten Daten entspricht das der Hauptkomponentenanalyse. Die Moore-Penrose-Pseudoinverse $\mathbf{A}^{+}$ liefert die Kleinste-Quadrate-Lösung mit der kleinsten Norm. Die Konditionszahl $\kappa=\sigma_1/\sigma_r$ begrenzt die Verstärkung relativer Fehler. Im Deep Learning bestimmt dasselbe Singulärwertspektrum die Größe von Fine-Tuning-Adaptern mit niedrigem Rang. Es definiert den von Muon verwendeten Polarfaktor und liefert die Lipschitz-Konstante, die wir mit der spektralen Normalisierung begrenzen.

<!--en:2294243a-->
## Übungen

<!--en:632be335-->
1. **Normen und maximale Streckung.** Im Text haben wir :eqref:`eq_mdl-sigma1-variational` aus dem Spektralsatz hergeleitet. Umgekehrt lässt sich die gesamte SVD allein aus der maximalen Streckung konstruieren.
    1. Sei $\sigma_1=\max_{\|\mathbf{x}\|=1}\|\mathbf{A}\mathbf{x}\|$. Nehmen Sie $\sigma_1>0$ an und konstruieren Sie eine SVD ohne den Spektralsatz. (i) Zeigen Sie, dass ein Einheitsvektor $\mathbf{v}_1$ das Maximum erreicht, und setzen Sie $\mathbf{u}_1=\mathbf{A}\mathbf{v}_1/\sigma_1$. (ii) Für einen Einheitsvektor $\mathbf{w}\perp\mathbf{v}_1$ hat die Funktion $t\mapsto\|\mathbf{A}(\mathbf{v}_1+t\mathbf{w})\|^2/(1+t^2)$ bei $t=0$ ein Maximum. Ihre Ableitung verschwindet dort also. Folgern Sie $\mathbf{u}_1^\top\mathbf{A}\mathbf{w}=0$. Somit bildet $\mathbf{A}$ den Raum $\mathbf{v}_1^\perp$ in $\mathbf{u}_1^\perp$ ab. (iii) Wenden Sie das Verfahren rekursiv auf die Einschränkung von $\mathbf{A}$ auf $\mathbf{v}_1^\perp$ an. Wo ging die Kompaktheit der Einheitssphäre ein?
    1. Zeigen Sie, dass $\|\mathbf{A}\|_2 \le \|\mathbf{A}\|_F \le \sqrt{r}\,\|\mathbf{A}\|_2$ mit $r = \operatorname{rank}\mathbf{A}$ gilt. Für welche Matrizen gilt links beziehungsweise rechts Gleichheit?
    1. Der Quotient $\|\mathbf{A}\|_F^2/\|\mathbf{A}\|_2^2$ heißt *stabiler Rang* von $\mathbf{A}$. Nach (b) liegt er zwischen 1 und $\operatorname{rank}\mathbf{A}$. Berechnen Sie ihn in geschlossener Form für das abfallende Spektrum $\sigma_i=e^{-i/12}$, $i=0,\ldots,255$ (eine geometrische Reihe). Zeigen Sie, dass Rang 18 nötig ist, um 95% der Energie dieses Spektrums zu erhalten. Schätzen Sie dann den stabilen Rang der gaußschen $400\times500$-Matrix aus :numref:`subsec_mdl-spectrum-control` mithilfe von $\|\mathbf{G}\|_F^2\approx mn$ und $\sigma_1\approx\sqrt m+\sqrt n$. Vergleichen Sie ihn mit dem Rang von etwa 270, der 95% ihrer Energie erhält. Warum kann der stabile Rang weit unter dem Rang liegen?
   :exlabel:`ex_mdl-svd-low-rank-norms-singular-values`

<!--en:acf48602-->
1. **Störungsschranke von Weyl.** Eine Aktualisierung $\boldsymbol{\Delta}$, etwa eine LoRA-Aktualisierung vom Rang $k$ der Form $\mathbf{B}\mathbf{A}$ (in der Notation der Originalarbeit), ändert die Singulärwerte einer Gewichtsmatrix $\mathbf{W}$. Die in der Weyl-Ungleichung :eqref:`eq_mdl-weyl-singular` angegebene Schranke aus :numref:`subsec_mdl-eckart-young` begrenzt diese Änderungen.
    1. Setzen Sie $j = 1$ in :eqref:`eq_mdl-weyl-singular` und wenden Sie die Ungleichung in beiden Richtungen an. Leiten Sie daraus $|\sigma_i(\mathbf{W}+\boldsymbol{\Delta})-\sigma_i(\mathbf{W})|\le\sigma_1(\boldsymbol{\Delta})$ für jedes $i$ her.
    1. Wählen Sie $\mathbf{W} = \operatorname{diag}(1, 2)$ und $\boldsymbol{\Delta} = 0.1\,\mathbf{1}\mathbf{1}^\top$. Berechnen Sie die Singulärwerte von $\mathbf{W}+\boldsymbol{\Delta}$ und vergleichen Sie jede Änderung mit $\sigma_1(\boldsymbol{\Delta})$.
    1. Nun gelte $\operatorname{rank}\boldsymbol{\Delta}\le k$. Der Beweis für die Frobeniusnorm in :numref:`subsec_mdl-eckart-young` wendet :eqref:`eq_mdl-weyl-singular` mit $j = k+1$ auf eine Matrix vom Rang höchstens $k$ an. Wenden Sie denselben Schritt auf $\boldsymbol{\Delta}$ an, um die Verzahnungsungleichungen $\sigma_{i+k}(\mathbf{W})\le\sigma_i(\mathbf{W}+\boldsymbol{\Delta})$ und $\sigma_{i+k}(\mathbf{W}+\boldsymbol{\Delta})\le\sigma_i(\mathbf{W})$ herzuleiten. Zeigen Sie damit, dass sich die Ränge von $\mathbf{W}+\boldsymbol{\Delta}$ und $\mathbf{W}$ um höchstens $k$ unterscheiden und höchstens $k$ Singulärwerte von $\mathbf{W}+\boldsymbol{\Delta}$ größer als $\sigma_1(\mathbf{W})$ sind, unabhängig davon, wie groß $\sigma_1(\boldsymbol{\Delta})$ ist. Warum folgen diese Aussagen nicht aus der Schranke in (a)?
   :exlabel:`ex_mdl-svd-low-rank-weyl-perturbation`

<!--en:e054d0e4-->
1. [code] **Zentrierung bei der PCA und Fashion-MNIST.** Die Zeilen von $\mathbf{X}\in\mathbb{R}^{n\times d}$ seien Datenpunkte mit mittlerer Zeile $\bar{\mathbf{x}}$ und Kovarianzmatrix $\mathbf{C}$ wie bei der Hauptkomponentenanalyse in :numref:`subsec_mdl-pca`.
    1. Zeigen Sie, dass $\tfrac1n\mathbf{X}^\top\mathbf{X} = \mathbf{C}+\bar{\mathbf{x}}\bar{\mathbf{x}}^\top$ gilt.
    1. Erzeugen Sie eine Punktwolke wie in :numref:`subsec_mdl-pca` (300 Punkte mit Standardabweichungen 3 und 0.8 entlang der um $\pi/5$ gedrehten Achsen). Verschieben Sie jeden Punkt um $c\,(1,-1)/\sqrt2$ für $c\in\{0, 1, 3, 10, 30\}$. Berechnen Sie für jedes $c$ den ersten rechten Singulärvektor der *unzentrierten* Daten, seinen Winkel zur Hauptachse $(\cos\frac\pi5, \sin\frac\pi5)$ und das Verhältnis $\sigma_1^2/\sum_i\sigma_i^2$.
    1. Sagen Sie mit (a) anhand der Eigenwerte von $\mathbf{C}$ vorher, bei welcher Verschiebung der erste Singulärvektor von der Hauptachse in Richtung des Mittelwerts schwenkt.
    1. Approximieren Sie mit (a) das Verhältnis $\sigma_1^2/\sum_i\sigma_i^2$ für große $c$ und bestimmen Sie seinen Grenzwert für $c\to\infty$. Bei $c = 10$ liegt das Verhältnis nahe beim erklärten Varianzanteil der zentrierten Daten. Liegt diese Übereinstimmung an den Daten oder an der gewählten Verschiebung?
    1. Laden Sie die Fashion-MNIST-Testbilder von T-Shirts und Hosen mit `d2l.FashionMNIST` (siehe :numref:`sec_mdl-geometry-linear-algebraic-ops`). Formen Sie die Bilder zu Vektoren der Länge 784 um, zentrieren Sie sie und berechnen Sie die ersten beiden rechten Singulärvektoren. Zeichnen Sie beide Klassen in der Ebene der ersten beiden Hauptkomponenten. Welchen Varianzanteil erfassen diese beiden Richtungen? Berechnen Sie den Winkel zwischen $\mathbf{v}_1$ und der Differenz der Klassenmittelwerte $\bar{\mathbf{x}}_{\text{trouser}}-\bar{\mathbf{x}}_{\text{t-shirt}}$. Erklären Sie, warum die Richtungen nahe beieinanderliegen, aber nicht übereinstimmen. Rekonstruieren Sie abschließend ein T-Shirt aus seinen ersten 5, 20 und 50 Komponenten und geben Sie jeweils den relativen Fehler an.
   :exlabel:`ex_mdl-svd-low-rank-centering-pca`

<!--en:81a78a47-->
1. [code] **Grenzfall ohne Ridge-Regularisierung und Gradientenabstieg.** Ein überparametrisiertes lineares Modell hat mehr Parameter als Gleichungen. Das Kleinste-Quadrate-Problem hat daher unendlich viele Minimierer. Teil (a) gilt für jede Form und jeden Rang. In (b) und (c) sei $\mathbf{A}\in\mathbb{R}^{m\times n}$ mit $m < n$.
    1. Zeigen Sie mit der Filterdarstellung :eqref:`eq_mdl-ridge-filter`, dass $(\mathbf{A}^\top\mathbf{A}+\lambda\mathbf{I})^{-1}\mathbf{A}^\top\to\mathbf{A}^{+}$ für $\lambda\to0^{+}$ gilt, und zwar für eine Matrix $\mathbf{A}$ beliebiger Form und beliebigen Rangs. Warum existiert $(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ für $m<n$ nicht?
    1. Beim Gradientenabstieg für $\tfrac12\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2$ mit Schrittweite $0 < \eta < 2/\sigma_1^2$ lautet die Aktualisierung $\mathbf{x}_{t+1} = \mathbf{x}_t-\eta\,\mathbf{A}^\top(\mathbf{A}\mathbf{x}_t-\mathbf{b})$. Zeigen Sie, dass $\mathbf{x}_t-\mathbf{x}_0$ im Zeilenraum von $\mathbf{A}$ bleibt und $\mathbf{x}_t$ gegen $\mathbf{A}^{+}\mathbf{b}+(\mathbf{I}-\mathbf{A}^{+}\mathbf{A})\mathbf{x}_0$ konvergiert. Welche Lösung wählt die Initialisierung $\mathbf{x}_0 = \mathbf{0}$ aus?
    1. Prüfen Sie (a) und (b) für eine gaußsche $20\times50$-Matrix $\mathbf{A}$ und einen gaußschen Vektor $\mathbf{b}$. Starten Sie den Gradientenabstieg einmal bei $\mathbf{x}_0 = \mathbf{0}$ und einmal bei einem zufälligen $\mathbf{x}_0$. Vergleichen Sie beide Grenzwerte und ihre Normen mit $\mathbf{A}^{+}\mathbf{b}$ und dessen Norm.
   :exlabel:`ex_mdl-svd-low-rank-ridgeless-gradient-descent`

<!--en:e3255a09-->
1. [code] **Einzugsgebiet der Newton-Schulz-Iteration.** Die Newton-Schulz-Iteration $\mathbf{X}\leftarrow\tfrac12(3\mathbf{X}-\mathbf{X}\mathbf{X}^\top\mathbf{X})$ aus :numref:`subsec_mdl-orthogonalized-updates` beginnt bei $\mathbf{X} = \mathbf{M}/\|\mathbf{M}\|_F$.
    1. Sei $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ eine dünne SVD. Zeigen Sie direkt, ohne :eqref:`eq_mdl-odd-polynomial`, dass ein Schritt $\mathbf{U}\,p(\boldsymbol{\Sigma})\,\mathbf{V}^\top$ ergibt, wobei $p(\sigma) = \tfrac12(3\sigma-\sigma^3)$ auf jeden Diagonaleintrag wirkt.
    1. Zeigen Sie, dass $(0,\sqrt3)$ das unmittelbare Einzugsgebiet von 1 ist. Dazu zeigen Sie, dass $p$ dieses Intervall in $(0, 1]$ abbildet und die Iterierten $p(p(\cdots p(\sigma_0)))$ für jedes $\sigma_0\in(0,\sqrt3)$ gegen 1 konvergieren. Zeigen Sie, dass einige Startwerte jenseits von $\sqrt3$ ebenfalls gegen 1 führen (versuchen Sie $2.15$), andere hingegen nicht (versuchen Sie 2). Entwickeln Sie $p(1-e)$ nach Potenzen von $e$. Wie schnell schrumpft der Abstand zu 1, sobald er klein ist?
    1. Zählen Sie für eine gaußsche $6\times4$-Matrix $\mathbf{M}$ die Schritte bis $\|\mathbf{X}-\mathbf{U}\mathbf{V}^\top\|_F < 10^{-6}$. Starten Sie die Iteration bei $\mathbf{M}/c$, einmal mit $c = \|\mathbf{M}\|_F$ und einmal mit $c = \sigma_1(\mathbf{M})$. Vergleichen Sie beide Schrittzahlen mit $\log_{3/2}(1/s_{\min})+4$, wobei $s_{\min}$ der kleinste Singulärwert nach der Skalierung ist. Warum dividiert Muon durch $\|\mathbf{M}\|_F$, obwohl $\sigma_1(\mathbf{M})$ einen Schritt spart?
    1. Starten Sie die Iteration bei $\mathbf{M}/c$ mit $c = \sigma_1(\mathbf{M})/2$ und mit $c = \sigma_1(\mathbf{M})/2.5$. Wogegen konvergiert sie jeweils? Erklären Sie beide Ergebnisse mit dem Polynom $p$ aus (b).
   :exlabel:`ex_mdl-svd-low-rank-newton-schulz-basin`

<!--en:62944835-->
1. [code] **Polarfaktor und Prokrustes-Problem.** ● Die Polarzerlegung :eqref:`eq_mdl-polar` stellt eine quadratische Matrix in der Form $\mathbf{A} = \mathbf{Q}\mathbf{P}$ dar. Dabei ist $\mathbf{Q}$ orthogonal und $\mathbf{P}$ symmetrisch und positiv semidefinit.
    1. Zeigen Sie, dass $\mathbf{P} = (\mathbf{A}^\top\mathbf{A})^{1/2}$ durch $\mathbf{A}$ bestimmt ist und dass $\mathbf{Q}$ durch $\mathbf{A}$ bestimmt ist, wenn $\mathbf{A}$ invertierbar ist. Geben Sie zwei verschiedene Polarzerlegungen von $\operatorname{diag}(1, 0)$ an.
    1. Zeigen Sie, dass $\mathbf{Q} = \mathbf{U}\mathbf{V}^\top$ eine orthogonale Matrix mit kleinstem Abstand zu $\mathbf{A}$ in der Frobeniusnorm ist. Führen Sie das Problem auf die Maximierung von $\operatorname{tr}(\mathbf{R}^\top\mathbf{A})$ über alle orthogonalen Matrizen $\mathbf{R}$ zurück.
    1. Zwei Einbettungsräume, die sich durch eine Drehung unterscheiden, lassen sich anhand von Punktpaaren $\mathbf{X},\mathbf{Y}\in\mathbb{R}^{d\times N}$ aufeinander ausrichten. Dazu minimieren wir $\|\mathbf{R}\mathbf{X}-\mathbf{Y}\|_F$ über alle orthogonalen Matrizen $\mathbf{R}$. Zeigen Sie mit (b), dass der Polarfaktor von $\mathbf{Y}\mathbf{X}^\top$ das Minimum erreicht. Testen Sie dies mit $d = 50$, $N = 1000$, einer zufälligen orthogonalen Matrix $\mathbf{R}$ (dem Faktor $\mathbf{Q}$ einer QR-Zerlegung einer gaußschen Matrix), einer gaußschen Matrix $\mathbf{X}$ und $\mathbf{Y} = \mathbf{R}\mathbf{X}+\mathbf{E}$. Die Einträge von $\mathbf{E}$ seien unabhängig und identisch nach $\mathcal{N}(0, s^2)$ verteilt, mit $s\in\{0, 0.1, 1, 3\}$. Vergleichen Sie $\|\hat{\mathbf{R}}-\mathbf{R}\|_F/\sqrt d$ für diese Schätzung und für die unbeschränkte Kleinste-Quadrate-Schätzung $\mathbf{Y}\mathbf{X}^{+}$.
   :exlabel:`ex_mdl-svd-low-rank-polar-procrustes`

<!--en:e4bbc2a2-->
1. [code] **Randomisierte Bestimmung des Bildraums.** ● Die randomisierte SVD aus :numref:`subsec_mdl-spectrum-control` multipliziert $\mathbf{A}$ mit einer gaußschen Matrix $\boldsymbol\Omega\in\mathbb{R}^{n\times(k+p)}$, orthonormalisiert die Spalten von $\mathbf{Y} = \mathbf{A}\boldsymbol\Omega$ zu $\mathbf{Q}$ und schneidet die SVD von $\mathbf{Q}^\top\mathbf{A}$ bei Rang $k$ ab.
    1. Implementieren Sie das Verfahren. Konstruieren Sie $\mathbf{A}\in\mathbb{R}^{1000\times500}$ als $\mathbf{U}\operatorname{diag}(\sigma_1,\ldots,\sigma_{500})\mathbf{V}^\top$ mit zufälligen Matrizen $\mathbf{U}$ und $\mathbf{V}$ mit orthonormalen Spalten. Verwenden Sie einmal den schnellen Abfall $\sigma_i = e^{-i/5}$ und einmal den langsamen Abfall $\sigma_i = 1/i$. Dividieren Sie für $k = 20$ und $p\in\{0, 5, 10\}$ den Fehler der Approximation vom Rang $k$ in der Spektralnorm durch $\sigma_{k+1}$, das Minimum aus :eqref:`eq_mdl-eckart-young`. Mitteln Sie über 10 Zufallsziehungen von $\boldsymbol\Omega$. Warum genügen zusätzliche Zufallsspalten allein für das eine Spektrum, aber nicht für das andere?
    1. Ersetzen Sie $\mathbf{Y}$ durch $(\mathbf{A}\mathbf{A}^\top)^q\mathbf{A}\boldsymbol\Omega$ für $q\in\{1, 2\}$. Wie ändern sich die Verhältnisse bei den beiden Spektren? Erklären Sie die Wirkung von $q$ anhand der Singulärwerte von $(\mathbf{A}\mathbf{A}^\top)^q\mathbf{A}$. Wie viele Produkte mit $\mathbf{A}$ oder $\mathbf{A}^\top$ benötigt jede Variante?
   :exlabel:`ex_mdl-svd-low-rank-randomized-range-finder`

<!--en:0cd07d81-->
1. [code] **Optimaler harter Schwellenwert.** ●● Der Schwellenwert $\tfrac{4}{\sqrt3}\sqrt n\,\sigma_{\text{noise}}$ von :citet:`Gavish.Donoho.2014` folgt aus dem Verhalten einer einzelnen Signalkomponente. Sei $\mathbf{A} = \theta\,\mathbf{u}\mathbf{v}^\top+\mathbf{N}$ eine $n\times n$-Matrix mit Einheitsvektoren $\mathbf{u},\mathbf{v}$, Signalstärke $\theta > 0$ und einer Rauschmatrix $\mathbf{N}$ mit unabhängigen, identisch gaußverteilten Einträgen $\mathcal{N}(0, \sigma_{\text{noise}}^2)$. Schreiben Sie $x = \theta/(\sqrt n\,\sigma_{\text{noise}})$. Sei $(\sigma_1, \hat{\mathbf{u}}, \hat{\mathbf{v}})$ das Tripel aus dem größten Singulärwert von $\mathbf{A}$ und seinen Singulärvektoren. Wählen Sie das Vorzeichen des Vektorpaares so, dass $\mathbf{u}^\top\hat{\mathbf{u}}\ge0$ gilt. Für $n\to\infty$ zeigen :citet:`Benaych-Georges.Nadakuditi.2012`, dass $\sigma_1$ gegen die obere Intervallgrenze $2\sqrt n\,\sigma_{\text{noise}}$ strebt und die Singulärvektoren im Grenzwert orthogonal zu $\mathbf{u}$ und $\mathbf{v}$ sind, wenn $x\le1$ gilt. Für $x > 1$ gilt

   $$\frac{\sigma_1}{\sqrt n\,\sigma_{\text{noise}}}\to x+\frac1x,\qquad
   (\mathbf{u}^\top\hat{\mathbf{u}})^2\to1-\frac1{x^2},\qquad
   (\mathbf{v}^\top\hat{\mathbf{v}})^2\to1-\frac1{x^2}.$$

   Der Übergang bei $x = 1$ ist das Gegenstück für Singulärwerte zum Übergang für Eigenwerte bei $\beta = \sqrt c$ in Aufgabe :exref:`ex_mdl-eigendecomposition-spiked-covariance` aus :numref:`sec_mdl-eigendecompositions`.

    1. Zwei Schätzer des Signals $\theta\,\mathbf{u}\mathbf{v}^\top$ sind die auf Rang eins abgeschnittene SVD $\sigma_1\hat{\mathbf{u}}\hat{\mathbf{v}}^\top$ und die Nullmatrix. Drücken Sie den quadrierten Fehler in der Frobeniusnorm für beide durch $\theta$, $\sigma_1$, $\mathbf{u}^\top\hat{\mathbf{u}}$ und $\mathbf{v}^\top\hat{\mathbf{v}}$ aus.
    1. Zeigen Sie mit den Grenzwerten, dass die abgeschnittene SVD genau für $x > \sqrt3$ den kleineren asymptotischen Fehler hat und dass der Singulärwert bei $x = \sqrt3$ gleich $\tfrac{4}{\sqrt3}\sqrt n\,\sigma_{\text{noise}}$ ist. Bei welchen Signalstärken liegt der Singulärwert über der Rauschgrenze und sollte dennoch verworfen werden?
    1. Prüfen Sie (a) und (b) für $n = 500$ und $\sigma_{\text{noise}} = 1/\sqrt n$ mit $x\in\{0.5, 1.2, 1.5, \sqrt3, 2, 3\}$ und jeweils 10 Zufallsziehungen. Vergleichen Sie $\sigma_1$, die quadrierten Kosinuswerte und beide Fehler mit ihren Grenzwerten.
   :exlabel:`ex_mdl-svd-low-rank-optimal-hard-threshold`

<!--en:a4371b20-->
[Diskussion](https://d2l.discourse.group/t/36826)

<!--en:534fc39e-->
Singulärwertzerlegung und niedrigrangige Approximation<br>**zwei Orthonormalbasen · vier Unterräume · Satz von Eckart und Young und PCA · Pseudoinverse und Kondition · LoRA und Muon**

<!--en:6349d81d-->
::: {.slide title="Warum eine Basis für Deep Learning nicht ausreicht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:c1455a7b-->
- **Rechteckige Schichten und Datentabellen:** Gewichtsmatrizen $\mathbf{W} \in \mathbb{R}^{m \times n}$ und Datentabellen $\mathbf{X} \in \mathbb{R}^{n \times d}$ sind im Allgemeinen nicht quadratisch und haben dann keine Eigenwerte.

<!--en:6cda47d3-->
- **Zwei Orthonormalbasen ($\mathbf{V}$ und $\mathbf{U}$):** Mit einer Eingabebasis $\mathbf{V}$ und einer Ausgabebasis $\mathbf{U}$ wirkt *jede* Matrix durch nichtnegative Streckfaktoren $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$.

<!--en:e9c0303f-->
- **Niedrigrangige Kompression:** Mit $k = 20$ Rang-1-Anteilen eines Katzenfotos mit $400 \times 500$ Pixeln genügen $9\%$ des Speicherplatzes bei $8\%$ relativem Fehler.

<!--en:5b7ae00f-->
::: {.slide title="Die SVD in drei Schritten: drehen, entlang der Achsen strecken, drehen"}

<!--en:c638361f-->
[Die Zerlegung]{.kicker}

<!--en:5c02b670-->
- **Satz ($\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$):** Jede reelle $m \times n$-Matrix zerfällt in eine orthogonale Matrix $\mathbf{U} \in \mathbb{R}^{m \times m}$ (linke Singulärvektoren $\mathbf{u}_i$), eine Diagonalmatrix $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ mit $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ und eine orthogonale Matrix $\mathbf{V} \in \mathbb{R}^{n \times n}$ (rechte Singulärvektoren $\mathbf{v}_i$).

<!--en:ef55f139-->
- **Wirkung auf einzelne Richtungen ($\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$):** Von rechts nach links gelesen dreht $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\mathbf{x}$ die Eingabebasis $\{\mathbf{v}_i\}$ auf die Koordinatenachsen ($\mathbf{V}^\top$), streckt Achse $i$ um $\sigma_i$ ($\boldsymbol{\Sigma}$) und dreht auf die Ausgabebasis $\{\mathbf{u}_i\}$ ($\mathbf{U}$).

<!--en:2721299c-->
::: {.slide title="Vollständige, dünne, kompakte und dyadische Form der SVD"}

<!--en:c5c8a7c2-->
[Darstellungen der SVD]{.kicker}

<!--en:0279e1f4-->
- **Vollständige und dünne SVD:** Die vollständige SVD behält die quadratischen orthogonalen Matrizen $\mathbf{U}_{m\times m}, \mathbf{V}_{n\times n}$, die dünne SVD jeweils $p = \min(m, n)$ Spalten (`full_matrices=False`).

<!--en:a8c3cac7-->
- **Kompakte SVD ($r = \operatorname{rank}\mathbf{A}$):** Nur positive Streckfaktoren bleiben erhalten: $\sigma_1 \ge \cdots \ge \sigma_r > 0$.

<!--en:fb35f186-->
- **Dyadische Summe:** $\mathbf{A} = \sum_{i=1}^r \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ ordnet die äußeren Produkte vom Rang 1 nach absteigendem Gewicht.

<!--en:33106fe2-->
::: {.slide title="Existenzbeweis mit dem Spektralsatz für AᵀA"}

<!--en:04d85825-->
[Warum die SVD immer existiert]{.kicker}

<!--en:1fbc43da-->
- **1. Positiv semidefinite Gram-Matrix $\mathbf{A}^\top\mathbf{A}$ diagonalisieren:** Die orthonormalen Eigenvektoren $\mathbf{v}_1, \ldots, \mathbf{v}_n$ haben die Eigenwerte $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$.

<!--en:0a5b5b14-->
- **2. Bildvektoren bestimmen ($\sigma_i = \sqrt{\lambda_i}$):** Für $i \le r$ den Vektor $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$ setzen. Für $i > r$ gilt $\|\mathbf{A}\mathbf{v}_i\|^2 = 0 \implies \mathbf{A}\mathbf{v}_i = \mathbf{0}$.

<!--en:3ebde1a3-->
- **3. Orthonormalität folgt direkt:** $\mathbf{u}_i^\top\mathbf{u}_j = \frac{\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j}{\sigma_i\sigma_j} = \delta_{ij}$.

<!--en:ac5ce1c0-->
::: {.slide title="Rechenbeispiel zur SVD der defekten Scherung T"}

<!--en:3edeee5b-->
[SVD von Hand und numerisch]{.kicker}

<!--en:1d57f2ef-->
- **Streckfaktoren und Goldener Schnitt:** Für $\mathbf{T} = \left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ hat $\mathbf{T}^\top\mathbf{T} = \left(\begin{smallmatrix}1&1\\1&2\end{smallmatrix}\right)$ die Kenngrößen $\operatorname{tr} = 3, \det = 1$ und die Eigenwerte $\lambda = (3 \pm \sqrt{5})/2 = \varphi^2, \varphi^{-2}$ mit $\varphi = (1+\sqrt{5})/2 \approx 1.618$. Daraus folgen $\sigma_1 = \varphi \approx 1.618$ und $\sigma_2 = 1/\varphi \approx 0.618$.

<!--en:ae7a0261-->
- **Winkel der Eingabe- und Ausgaberichtungen:** Aus $(\mathbf{T}^\top\mathbf{T} - \varphi^2\mathbf{I})\mathbf{v}_1 = \mathbf{0}$ folgen $\mathbf{v}_1 \propto (1, \varphi)$ bei $58.3^\circ$ und $\mathbf{u}_1 = \mathbf{T}\mathbf{v}_1/\varphi \propto (\varphi, 1)$ bei $31.7^\circ$. Damit gilt $\mathbf{T} = \mathbf{R}_{31.7^\circ}\operatorname{diag}(\varphi, 1/\varphi)\mathbf{R}_{58.3^\circ}^\top$:

<!--en:23abfc76-->
::: {.slide title="Rechenbeispiel zur SVD der defekten Scherung T (Fortsetzung)"}

<!--en:3edeee5b-2-->
[SVD von Hand und numerisch]{.kicker}

<!--en:5f55c37e-->
::: {.slide title="Eine rechteckige SVD aus AᵀA im Code konstruieren"}

<!--en:3edeee5b-3-->
[SVD von Hand und numerisch]{.kicker}

<!--en:d138db82-->
Für eine $3 \times 2$-Matrix ($\mathbf{A}^\top\mathbf{A} = \left(\begin{smallmatrix}10&6\\6&14\end{smallmatrix}\right)$, $\lambda = 12 \pm \sqrt{40}$) liefert das Verfahren aus dem Beweis dasselbe Ergebnis wie `np.linalg.svd`, bis auf gemeinsame Vorzeichenwechsel $(\mathbf{u}_i, \mathbf{v}_i) \mapsto (-\mathbf{u}_i, -\mathbf{v}_i)$:

<!--en:43def4c0-->
**Warum Bibliotheken die Bidiagonalisierung nach Golub und Kahan verwenden:** Mit $\mathbf{A}^\top\mathbf{A}$ lässt sich die Existenz beweisen. Beim Bilden von $\mathbf{A}^\top\mathbf{A}$ in Gleitkommaarithmetik steigt die Konditionszahl auf $\kappa(\mathbf{A})^2$, und kleine Singulärwerte gehen verloren. Numerische SVD-Verfahren wenden orthogonale Transformationen direkt auf $\mathbf{A}$ an.

<!--en:6786d638-->
::: {.slide title="Maximale Streckung, Normen und Zusammenhang mit den Eigenwerten"}

<!--en:1d0451bf-->
[Singulärwerte und Eigenwerte]{.kicker}

<!--en:895624a0-->
- **Normen aus Singulärwerten:** $\|\mathbf{A}\|_2 = \max_{\|\mathbf{x}\|=1}\|\mathbf{A}\mathbf{x}\| = \sigma_1$ (erreicht bei $\mathbf{v}_1$) und $\|\mathbf{A}\|_F^2 = \sum_{i=1}^r \sigma_i^2$.

<!--en:87b68b3b-->
- **Singulärwerte $\neq$ Eigenwertbeträge:** Die Beispielmatrix $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ hat $|\lambda_{1,2}| = \sqrt{5} \approx 2.236$, während $\sigma_1 \approx 3.618, \sigma_2 \approx 1.382$ gilt.

<!--en:8800d6ef-->
- **Volumenidentität:** $|\det\mathbf{A}| = \prod_i \sigma_i = \prod_i |\lambda_i| = 5$.

<!--en:e5633e34-->
::: {.slide title="Die vier fundamentalen Unterräume aus der SVD"}

<!--en:8d124865-->
[Geometrie der Unterräume]{.kicker}

<!--en:e24b019b-->
- **Exakter Rang ($r = \#\{\sigma_i > 0\}$):** Orthogonale Matrizen erhalten die Dimension. Daher gilt $\operatorname{rank}\mathbf{A} = \operatorname{rank}\boldsymbol{\Sigma} = r$.

<!--en:65b49f33-->
- **Orthonormalbasen für alle vier Unterräume:**
  - *Zeilenraum* $\operatorname{col}(\mathbf{A}^\top) = \operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_r\} \subseteq \mathbb{R}^n$
  - *Nullraum* $\ker(\mathbf{A}) = \operatorname{span}\{\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n\} \subseteq \mathbb{R}^n$
  - *Spaltenraum* $\operatorname{col}(\mathbf{A}) = \operatorname{span}\{\mathbf{u}_1, \ldots, \mathbf{u}_r\} \subseteq \mathbb{R}^m$
  - *Linker Nullraum* $\ker(\mathbf{A}^\top) = \operatorname{span}\{\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m\} \subseteq \mathbb{R}^m$

<!--en:bcdaf05a-->
- **Bijektion auf dem Zeilenraum:** $\mathbf{A}$ bildet $\mathbf{v}_i \mapsto \sigma_i\mathbf{u}_i$ ($i \le r$) bijektiv ab und schickt $\ker\mathbf{A}$ auf null.

<!--en:ab12972a-->
::: {.slide title="Numerischer Rang und Schwellenwert für Gleitkommazahlen"}

<!--en:af221f3a-->
[Geometrie der Unterräume in der Praxis]{.kicker}

<!--en:32f43893-->
- **Exakte Nullen sind unzuverlässig:** Rundungsfehler ändern Singulärwerte um bis zu $\|\mathbf{E}\|_2 \sim \epsilon_{\text{mach}}\sigma_1$ (Weyl-Ungleichung). Eine mathematische Null erscheint daher meist als winzige Zahl um $10^{-15}\sigma_1$ und nur gelegentlich als exakte $0$.

<!--en:bafa107f-->
- **Übliche Rangtoleranz:** `np.linalg.matrix_rank` zählt Singulärwerte oberhalb von $\text{tol} = \sigma_1 \max(m, n)\,\epsilon_{\text{mach}}$:

<!--en:0de1e335-->
::: {.slide title="Abgeschnittene SVD und Satz von Eckart und Young"}

<!--en:b543f5a2-->
[Beste niedrigrangige Approximation]{.kicker}

<!--en:ffd7cb7c-->
- **Auf Rang $k$ abgeschnittene SVD:** $\mathbf{A}_k = \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ speichert $k(m + n + 1)$ Zahlen anstelle von $mn$.

<!--en:2abd922f-->
- **Satz von Eckart, Young und Mirsky:** $\mathbf{A}_k$ minimiert $\|\mathbf{A} - \mathbf{B}\|$ über alle $\operatorname{rank}\mathbf{B} \le k$ sowohl in der Spektralnorm ($\sigma_{k+1}$) als auch in der Frobeniusnorm ($\sqrt{\sum_{i>k}\sigma_i^2}$).

<!--en:ff4cebcb-->
- **Eindeutigkeit:** In der Frobeniusnorm ist die Lösung eindeutig, wenn $\sigma_k > \sigma_{k+1}$.

<!--en:08e958b3-->
::: {.slide title="Beweis des Satzes von Eckart und Young durch Schnitt von Unterräumen"}

<!--en:446a2f57-->
[Beweis des Satzes von Eckart und Young]{.kicker}

<!--en:1c08f8b4-->
- **Dimensionsargument ($\dim\mathcal{K} + \dim\mathcal{V} > n$):** Jede Vergleichsmatrix $\mathbf{B}$ mit $\operatorname{rank}\mathbf{B} \le k$ hat einen Nullraum $\mathcal{K} = \ker\mathbf{B}$ der Dimension mindestens $n - k$, während $\mathcal{V} = \operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{k+1}\}$ die Dimension $k + 1$ hat.

<!--en:8aa8500f-->
- **Nichttrivialer Schnitt:** Wegen $(n - k) + (k + 1) = n + 1 > n$ enthält $\mathcal{K} \cap \mathcal{V}$ einen Einheitsvektor $\mathbf{x} = \sum_{i=1}^{k+1} c_i\mathbf{v}_i$ ($\sum c_i^2 = 1$), für den $\mathbf{B}\mathbf{x} = \mathbf{0}$ gilt.

<!--en:af7cdcdc-->
- **Untere Schranke:** Diesen Vektor $\mathbf{x}$ streckt $\mathbf{A}$ noch mindestens um den Faktor $\sigma_{k+1}$:
  $$\|\mathbf{A} - \mathbf{B}\|_2^2 \ge \|\mathbf{A}\mathbf{x}\|^2 = \sum_{i=1}^{k+1} c_i^2\sigma_i^2 \ge \sigma_{k+1}^2$$

<!--en:d2fe36ab-->
::: {.slide title="Weyl-Ungleichung und Beweis für die Frobeniusnorm"}

<!--en:bbe6e269-->
[Störung und Schranke in der Frobeniusnorm]{.kicker}

<!--en:c675117a-->
- **Lemma von Weyl:** $\sigma_{i+j-1}(\mathbf{X} + \mathbf{Y}) \le \sigma_i(\mathbf{X}) + \sigma_j(\mathbf{Y})$ ergibt sich durch Vergleich von $\mathbf{X} + \mathbf{Y}$ mit der auf Rang $(i+j-2)$ abgeschnittenen Matrix $\mathbf{X}_{i-1} + \mathbf{Y}_{j-1}$.

<!--en:301f86d1-->
- **1-Lipschitz-Stabilität ($j = 1$):** $|\sigma_i(\mathbf{A} + \mathbf{E}) - \sigma_i(\mathbf{A})| \le \|\mathbf{E}\|_2$.

<!--en:b02dc226-->
- **Beweis für die Frobeniusnorm ($j = k + 1$):** $\sigma_i(\mathbf{A} - \mathbf{B}) \ge \sigma_{k+i}(\mathbf{A}) \implies \|\mathbf{A} - \mathbf{B}\|_F^2 \ge \sum_{i>k}\sigma_i^2$.

<!--en:8325b921-->
::: {.slide title="Der Satz von Eckart und Young an einem Foto mit 400×500 Pixeln"}

<!--en:f4d7a406-->
[Niedrigrangige Kompression]{.kicker}

<!--en:0401e6b4-->
- **Abfall nach einem Potenzgesetz ($\sigma_i \propto i^{-1.1}$):** Bei natürlichen Bildern fallen die Singulärwerte rasch ab. Deshalb erfassen $k = 20$ Dyaden $99.4\%$ der Frobeniusenergie ($8\%$ relativer Fehler) mit nur $9\%$ der Parameter. Gegenüber der Projektion auf die 20 größten Bildspalten ($31\%$ Fehler) sinkt der Fehler fast um den Faktor $4\times$.

<!--en:ee83e44a-->
::: {.slide title="Energieanteil, Zentrierung und Rauschunterdrückung mit der SVD"}

<!--en:ac2363f5-->
[Wahl des Rangs k]{.kicker}

<!--en:fc7bc863-->
- **Energieanteil in der Frobeniusnorm:** Der bei Rang $k$ erhaltene Anteil der quadratischen Gesamtenergie bestimmt direkt den relativen Fehler in der Frobeniusnorm:
  $$\text{Energy}(k) = \frac{\sum_{i \le k} \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} = 1 - \frac{\|\mathbf{A} - \mathbf{A}_k\|_F^2}{\|\mathbf{A}\|_F^2}$$

<!--en:6bc9b36e-->
- **Einfluss des Mittelwerts und Rauschunterdrückung nach Gavish und Donoho:** Beim unzentrierten Foto erfasst $\sigma_1^2$ allein durch die mittlere Helligkeit $89\%$ der Energie. Bei additivem Rauschen $\mathbf{A} = \mathbf{B} + \mathbf{N}$ ($\sigma_{\text{noise}}$) verwirft das optimale Verfahren Singulärwerte unterhalb der harten Schwelle $\tfrac{4}{\sqrt{3}}\sqrt{n}\,\sigma_{\text{noise}} \approx 2.309\sqrt{n}\,\sigma_{\text{noise}}$.

<!--en:e8285937-->
::: {.slide title="Energieanteil, Zentrierung und Rauschunterdrückung mit der SVD (Fortsetzung)"}

<!--en:ac2363f5-2-->
[Wahl des Rangs k]{.kicker}

<!--en:12883fff-->
::: {.slide title="Hauptkomponentenanalyse mit der SVD zentrierter Daten"}

<!--en:01a7963e-->
[Dimensionsreduktion]{.kicker}

<!--en:71672a77-->
- **Zuerst zentrieren ($\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^\top$):** Der optimale $k$-dimensionale affine Unterraum verläuft stets durch den Stichprobenmittelwert $\bar{\mathbf{x}}$. Die Kovarianzmatrix lautet $\mathbf{C} = \tfrac{1}{n}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} = \tfrac{1}{n}\mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top$.

<!--en:870e8729-->
- **Maximale Varianz = minimaler Fehler:** Nach Rayleigh sowie Eckart und Young maximieren die ersten rechten Singulärvektoren $\mathbf{v}_1, \ldots, \mathbf{v}_k$ von $\tilde{\mathbf{X}}$ die projizierte Varianz ($\sigma_i^2/n$) und minimieren zugleich den Rekonstruktionsfehler:
  $$\tilde{\mathbf{X}}\mathbf{V}_k\mathbf{V}_k^\top = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^\top = \tilde{\mathbf{X}}_k$$

<!--en:1ea42704-->
::: {.slide title="PCA im Code: Varianz und Rekonstruktionsfehler"}

<!--en:653250e1-->
[PCA in der Praxis]{.kicker}

<!--en:e8327347-->
Bei einer um $36^\circ$ gedrehten Wolke aus 300 Punkten mit Standardabweichungen $3.0$ und $0.8$ in der Grundgesamtheit liefert die SVD der zentrierten Daten die Hauptachse. Der mittlere quadratische Fehler bei Rang 1 stimmt mit $\sigma_2^2/n$ überein:

<!--en:e463bb22-->
**Was geschieht ohne Zentrierung?** Ohne Abzug von `X.mean(0)` zeigt $\mathbf{v}_1$ vom Ursprung zum Schwerpunkt $\bar{\mathbf{x}}$ der Wolke statt entlang ihrer größten Streuung.

<!--en:aa02b9ce-->
::: {.slide title="Die Moore-Penrose-Pseudoinverse A⁺"}

<!--en:c3ab6d89-->
[Lösen linearer Gleichungssysteme]{.kicker}

<!--en:6941a537-->
- **Positive Streckfaktoren umkehren:** Für jedes $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \in \mathbb{R}^{m \times n}$ gilt für die *Moore-Penrose-Pseudoinverse* $\mathbf{A}^+ \in \mathbb{R}^{n \times m}$ die Darstellung
  $$\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$$
  mit $\boldsymbol{\Sigma}^+_{ii} = 1/\sigma_i$ für $\sigma_i > 0$ und $0$ sonst.

<!--en:584dadfc-->
- **Kleinste Quadrate bei minimaler Norm ($\hat{\mathbf{x}} = \mathbf{A}^+\mathbf{b}$):** In den Singulärkoordinaten $\mathbf{y} = \mathbf{V}^\top\mathbf{x}$ und $\mathbf{c} = \mathbf{U}^\top\mathbf{b}$ gilt $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 = \sum_{i \le r}(\sigma_i y_i - c_i)^2 + \sum_{i > r} c_i^2$.

<!--en:5a3a695e-->
- **Getrennte Aufgaben:** Die Wahl $y_i = c_i/\sigma_i$ ($i \le r$) minimiert das Residuum. Die Nullraumkoordinaten $y_i = 0$ ($i > r$) minimieren $\|\mathbf{x}\| = \|\mathbf{y}\|$.

<!--en:b89786e5-->
::: {.slide title="Ridge-Regression als Spektralfilter"}

<!--en:e5951b8f-->
[Regularisierte Methode der kleinsten Quadrate]{.kicker}

<!--en:88887837-->
- **Warum kleine Singulärwerte problematisch sind:** In $\mathbf{A}^+\mathbf{b} = \sum_{i \le r} \frac{\mathbf{u}_i^\top\mathbf{b}}{\sigma_i}\mathbf{v}_i$ verstärkt die Division durch ein winziges $\sigma_i$ das Messrauschen entlang $\mathbf{u}_i$ um den Faktor $1/\sigma_i$.

<!--en:dce07e2d-->
- **Filterfaktoren bei Ridge-Regression und abgeschnittener SVD:** Die Ridge-Regression $\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|^2$ (Gewichtszerfall) dämpft Richtungen mit $\sigma_i^2 \ll \lambda$ allmählich, statt sie abrupt abzuschneiden:
  $$\mathbf{x}_\lambda = (\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})^{-1}\mathbf{A}^\top\mathbf{b} = \sum_{i=1}^r \underbrace{\frac{\sigma_i^2}{\sigma_i^2 + \lambda}}_{\text{ridge filter}} \frac{\mathbf{u}_i^\top\mathbf{b}}{\sigma_i}\,\mathbf{v}_i$$

<!--en:476fa6b8-->
::: {.slide title="Ridge-Regression als Spektralfilter (Fortsetzung)"}

<!--en:e5951b8f-2-->
[Regularisierte Methode der kleinsten Quadrate]{.kicker}

<!--en:dd6280ee-->
::: {.slide title="Die Konditionszahl κ(A) = σ₁ / σᵣ"}

<!--en:7fee687a-->
[Empfindlichkeit und Kondition]{.kicker}

<!--en:951dbff6-->
- **Verhältnis der extremen Streckfaktoren:** $\kappa(\mathbf{A}) = \sigma_1/\sigma_r \ge 1$ ($\kappa = 1$ für orthogonale $\mathbf{Q}$, $\kappa = \infty$ für singuläre Matrizen).

<!--en:8952f9ff-->
- **Relative Fehlerverstärkung:** Bei einer Störung von $\mathbf{A}\mathbf{x} = \mathbf{b}$ um $\delta\mathbf{b}$ verstärkt sich der relative Fehler in der Lösung höchstens um den Faktor $\kappa(\mathbf{A})$: $\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(\mathbf{A})\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}$.

<!--en:334a9eb7-->
- **Quadrieren durch die Gram-Matrix:** $\kappa(\mathbf{A}^\top\mathbf{A}) = (\sigma_1/\sigma_r)^2 = \kappa(\mathbf{A})^2$.

<!--en:532e26c1-->
::: {.slide title="Fehlerverstärkung durch die Konditionszahl und schmale Täler"}

<!--en:8ac50f08-->
[Kondition im Code und in der Optimierung]{.kicker}

<!--en:1a3f826c-->
- **Zickzackpfad beim Gradientenabstieg:** Bei $\tfrac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ haben die Niveaumengen das Achsenverhältnis $\kappa(\mathbf{A})$ und die Hesse-Matrix $\mathbf{H} = \mathbf{A}^\top\mathbf{A}$ die Konditionszahl $\kappa(\mathbf{A})^2$. Der Gradientenabstieg springt daher zwischen den steilen Talflanken hin und her.

<!--en:1a4f4500-->
::: {.slide title="LoRA: Anpassung vortrainierter Gewichte mit niedrigem Rang"}

<!--en:c282af95-->
[SVD im modernen Deep Learning]{.kicker}

<!--en:66c5e62b-->
- **Paralleler Pfad vom Rang $k$:** LoRA hält eine vortrainierte Gewichtsmatrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ fest und lernt eine Aktualisierung vom Rang $k$ in der Form $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ ($\mathbf{B} \in \mathbb{R}^{m \times k}$, $\mathbf{A} \in \mathbb{R}^{k \times n}$):
  $$\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$$

<!--en:d88c4699-->
- **Parameterreduktion ($256\times$):** Bei einer $4096 \times 4096$-Transformer-Schicht ($16.8\text{M}$ Gewichte) genügen für Rang $k = 8$ genau $k(m+n) = 65{,}536$ trainierbare Parameter ($0.39\%$). Die Aktualisierung geht ohne Zusatzkosten bei der Inferenz in $\mathbf{W}$ ein.

<!--en:e7632779-->
- **Aussage des Satzes von Eckart und Young:** Jede vollständige Aktualisierung $\Delta\mathbf{W}_{\text{full}}$ lässt sich bis auf $\sigma_{k+1}(\Delta\mathbf{W}_{\text{full}})$ durch einen Adapter vom Rang $k$ approximieren. LoRA funktioniert, wenn *irgendeine* niedrigrangige Aktualisierung einen kleinen Verlust erreicht. Das ist ein empirischer Befund und folgt nicht aus dem Satz von Eckart und Young.

<!--en:bb7116ae-->
::: {.slide title="Polarzerlegung und Newton-Schulz-Iteration bei Muon"}

<!--en:3404bb3f-->
[Orthogonalisierte Matrixaktualisierungen]{.kicker}

<!--en:848c8ee2-->
- **Polarfaktor ($\mathbf{Q} = \mathbf{U}\mathbf{V}^\top$):** Nächstgelegene orthogonale Matrix ($\sigma_i \to 1$). Muon nutzt sie, um alle aktiven Richtungen mit dem Faktor eins zu aktualisieren.

<!--en:16ebe4d9-->
- **Newton-Schulz-Iteration mit Matrixmultiplikationen:** Die Iteration $\mathbf{X}_{t+1} = \tfrac{1}{2}(3\mathbf{X}_t - \mathbf{X}_t\mathbf{X}_t^\top\mathbf{X}_t)$ mit Startwert $\mathbf{M}/\|\mathbf{M}\|_F$ wendet $p(\sigma) = \tfrac{1}{2}(3\sigma - \sigma^3) \to 1$ an.

<!--en:01874494-->
::: {.slide title="Polarzerlegung und Newton-Schulz-Iteration bei Muon (Fortsetzung)"}

<!--en:3404bb3f-2-->
[Orthogonalisierte Matrixaktualisierungen]{.kicker}

<!--en:caa05cc5-->
::: {.slide title="Spektrale Normalisierung und Abfall der Singulärwerte"}

<!--en:12a018c4-->
[Spektren begrenzen und untersuchen]{.kicker}

<!--en:4324e5b6-->
- **Lipschitz-Konstante über $\sigma_1 = \|\mathbf{W}\|_2$ begrenzen:** Wegen $\|\mathbf{W}\mathbf{x} - \mathbf{W}\mathbf{x}'\| \le \sigma_1\|\mathbf{x} - \mathbf{x}'\|$ dividiert die spektrale Normalisierung $\mathbf{W}$ durch einen Schätzwert für $\sigma_1$. Pro Aktualisierung genügt 1 Schritt der alternierenden Potenziteration ($\mathbf{u} \leftarrow \mathbf{W}\mathbf{v}$, $\mathbf{v} \leftarrow \mathbf{W}^\top\mathbf{u}$), um die Schätzung fortzuschreiben.

<!--en:0cf125e8-->
- **Niedrigen Rang durch Messung prüfen:** Eine gaußsche $400 \times 500$-Rauschmatrix und unser zentriertes Katzenfoto zeigen im Vergleich, weshalb niedrigrangige Kompression ein abfallendes Spektrum voraussetzt:

<!--en:77589ef0-->
::: {.slide title="Verständnisfragen zu SVD, Satz von Eckart und Young und Kondition"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:f2fddd7e-->
1. **Singulärwerte und Eigenwerte:** Die Matrix $\mathbf{A} = \left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ hat die Eigenwerte $2 \pm i$ ($|\lambda| = \sqrt{5}$). Sind ihre Singulärwerte gleich $\sqrt{5}$? Wie hängen $\sigma_1, \sigma_2$ und $\det\mathbf{A} = 5$ zusammen?

<!--en:e4ee50ac-->
2. **Konditionszahl der Scherung:** Die Scherung $\mathbf{T} = \left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ hat die Singulärwerte $\sigma_1 = \varphi$ und $\sigma_2 = 1/\varphi$ ($\varphi \approx 1.618$). Wie groß sind $\kappa(\mathbf{T})$ und $\kappa(\mathbf{T}^\top\mathbf{T})$?

<!--en:56eaab30-->
3. **Pseudoinverse bei einem unterbestimmten System:** Welche Lösung $\hat{\mathbf{x}} = \mathbf{A}^+\mathbf{b}$ liefert die Pseudoinverse für die einzelne Gleichung $x_1 + x_2 = 2$ und warum?

<!--en:51647851-->
**Lösungen:**
**1.** Nein. $\mathbf{A}^\top\mathbf{A} = \left(\begin{smallmatrix}2&-1\\-1&13\end{smallmatrix}\right)$ hat die Eigenwerte $\sigma_{1,2}^2 = (15 \pm 5\sqrt{5})/2$. Daraus folgen $\sigma_1 \approx 3.618$ und $\sigma_2 \approx 1.382$ mit dem Produkt $\sigma_1\sigma_2 = 5 = |\det\mathbf{A}| = |\lambda_1\lambda_2|$.
**2.** $\kappa(\mathbf{T}) = \sigma_1/\sigma_2 = \varphi / (1/\varphi) = \varphi^2 \approx 2.618$. Beim Bilden der Gram-Matrix quadriert sich die Konditionszahl auf $\kappa(\mathbf{T}^\top\mathbf{T}) = \varphi^4 \approx 6.854$.
**3.** $\hat{\mathbf{x}} = (1, 1)$ ist der einzige Punkt auf der Geraden $x_1 + x_2 = 2$ mit minimalem Abstand zum Ursprung (Komponente null entlang des Nullraumvektors $(1, -1)$).

<!--en:bdfd89ce-->
::: {.slide title="Zusammenfassung zur Singulärwertzerlegung"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:90899ac6-->
| Begriff oder Verfahren | Mathematischer Ausdruck | Zentrale Eigenschaft und Rolle im Deep Learning |

<!--en:7e10e6bb-->
| **SVD und dyadische Summe** | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_{i=1}^r \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ | Existiert für jede $m \times n$-Matrix. Es gilt $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$ mit $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$. |

<!--en:a31b6905-->
| **Vier Unterräume und Normen** | $r = \#\{\sigma_i > 0\}$, $\|\mathbf{A}\|_2 = \sigma_1$, $\|\mathbf{A}\|_F^2 = \sum \sigma_i^2$ | $\mathbf{v}_{1..r}$ spannen den Zeilenraum auf, $\mathbf{v}_{r+1..n}$ den Nullraum, $\mathbf{u}_{1..r}$ den Spaltenraum und $\mathbf{u}_{r+1..m}$ den linken Nullraum. |

<!--en:6dcfc3e0-->
| **Satz von Eckart und Young und PCA** | $\|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}$, $\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i>k} \sigma_i^2$ | Die abgeschnittene SVD $\mathbf{A}_k$ ist eine optimale Approximation vom Rang $k$. Für zentrierte Daten liefert $\tilde{\mathbf{X}}_k$ die PCA. |

<!--en:46013541-->
| **Pseudoinverse und Ridge-Regression** | $\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$, Filter $\frac{\sigma_i^2}{\sigma_i^2 + \lambda}$ | $\hat{\mathbf{x}} = \mathbf{A}^+\mathbf{b}$ ist die Kleinste-Quadrate-Lösung minimaler Norm. Ridge-Regression dämpft Rauschen bei kleinen $\sigma_i$. |

<!--en:507f86d3-->
| **Konditionszahl $\kappa(\mathbf{A})$** | $\kappa(\mathbf{A}) = \sigma_1/\sigma_r$, $\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}$ | Beschränkt Fehlerverstärkung und Streckung der Höhenlinien. Es gilt $\kappa(\mathbf{A}^\top\mathbf{A}) = \kappa(\mathbf{A})^2$. |

<!--en:a15e9770-->
| **LoRA, Muon und Lipschitz-Konstante** | $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$, $\mathbf{Q} = \mathbf{U}\mathbf{V}^\top$, $\mathbf{W}/\sigma_1$ | LoRA komprimiert Adapter. Die Newton-Schulz-Iteration berechnet $\mathbf{U}\mathbf{V}^\top$ für Muon. $\sigma_1$ begrenzt die Streckung der Schicht. |
