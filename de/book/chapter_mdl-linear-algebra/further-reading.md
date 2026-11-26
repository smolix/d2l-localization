<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:45df84d7-->
Dieser Abschnitt verbindet die lineare Algebra aus diesem Kapitel mit der numerischen Mathematik, der Theorie der Zufallsmatrizen und Anwendungen im Deep Learning. Den Abschluss bildet eine Auswahl von Lehrbüchern und Kursen zu den Kernthemen des Kapitels. Wenn Sie sich vor allem für Deep Learning interessieren, beginnen Sie mit Zufallsprojektionen, iterativen Lösungsverfahren, Zufallsmatrizen und der trainierbaren Netztiefe sowie dem Sketching. Springen Sie anschließend zu den letzten beiden Beiträgen über Attention mit niedrigem Rang und Optimierer auf Basis der Spektralnorm. Der Rest des Buches setzt diesen Abschnitt nicht voraus.

<!--en:699aefbb-->
## Geometrie und lineare Abbildungen

<!--en:73d04265-->
### Kernel-Methoden mit anderen inneren Produkten

<!--en:82d3bfd8-->
Bei der Klassifikation nach dem nächsten Klassenmittelwert (:numref:`subsec_mdl-nearest-mean`) vergleichen wir ein Testbild über ein einziges Skalarprodukt mit den beiden Klassenmittelwerten. Damit ist die Entscheidungsgrenze zwangsläufig eine flache Hyperebene. Diese einfache Regel stößt an ihre Grenzen, wenn eine Klasse gekrümmt ist oder in mehrere Gruppen zerfällt, etwa bei der Unterscheidung von T-Shirts und Hemden. Kernel-Methoden behalten genau dieselbe geometrische Regel bei und ändern nur das innere Produkt. Wir ersetzen $\mathbf{x}\cdot\mathbf{x}_i$ durch eine Ähnlichkeitsfunktion $k(\mathbf{x},\mathbf{x}_i)$. Sie entspricht einem echten inneren Produkt in einem anderen, möglicherweise unendlichdimensionalen Merkmalsraum. Der Klassenmittelwert ist nun ein Mittelwert in diesem verborgenen Raum. Jedes Trainingsbeispiel erhält bei der Abstimmung das Gewicht $k(\mathbf{x},\mathbf{x}_i)$. Die Entscheidungsgrenze kann sich dadurch im ursprünglichen Eingaberaum frei krümmen. Die Frage, welche Trainingspunkte tatsächlich ein von null verschiedenes Gewicht brauchen, führt direkt zur Support Vector Machine. :citet:`Scholkopf.Smola.2002` beginnen ihr Buch mit genau diesem Klassifikator und entwickeln daraus die gesamte Theorie der Kernel-Methoden.

<!--en:d94e6320-->
### Zufallsprojektionen

<!--en:ce2bf907-->
In :numref:`sec_mdl-geometry-linear-algebraic-ops` haben wir gesehen, dass zufällige Richtungen im hochdimensionalen Raum $\mathbb{R}^d$ nahezu orthogonal zueinander sind. Erstaunlicherweise können wir dank dieser geometrischen Eigenschaft die Dimension eines Datensatzes verringern, ohne die Daten überhaupt anzusehen. Das Johnson-Lindenstrauss-Lemma :cite:`Johnson.Lindenstrauss.1984` garantiert für jede Menge von $n$ Punkten Folgendes. Multiplizieren wir die Punkte mit einer geeignet skalierten gaußschen Zufallsmatrix, die sie in $k$ Dimensionen der Größenordnung $\varepsilon^{-2}\log n$ abbildet, so bleiben mit hoher Wahrscheinlichkeit alle $\binom{n}{2}$ paarweisen Abstände bis auf einen Faktor $1\pm\varepsilon$ erhalten. Anders als die Hauptkomponentenanalyse (:numref:`subsec_mdl-pca`) ist eine Zufallsprojektion völlig unabhängig von den Daten. Nicht einmal ihre Zieldimension $k$ hängt von der Dimension $d$ des umgebenden Raums ab. Das Verfahren lohnt sich daher besonders, wenn $d$ sehr viel größer als $\varepsilon^{-2}\log n$ ist. :citet:`Dasgupta.Gupta.2003` geben einen bemerkenswert kurzen Beweis mit expliziten Konstanten. :citet:`Vershynin.2018` ordnet das Lemma in die allgemeinere Theorie der Konzentration von Zufallsvektoren ein.

<!--en:ae60100d-->
### Lösungsverfahren für große lineare Gleichungssysteme

<!--en:2b99d2ca-->
In :numref:`subsec_mdl-numerical-issues` haben wir lineare Gleichungssysteme mittels LU-Zerlegung gelöst. Dabei haben wir festgestellt, dass die Inverse einer großen, dünn besetzten Matrix meist dicht besetzt ist. Warum funktioniert die Gauß-Elimination mit partieller Pivotisierung in der Gleitkommaarithmetik so zuverlässig, obwohl es pathologische Gegenbeispiele gibt? :citet:`Trefethen.Bau.1997` erklären anschaulich, warum sich die Stabilität im ungünstigsten Fall so stark von der in der Praxis unterscheidet. :citet:`Higham.2002` liefert die maßgebliche Analyse der Rundungsfehler.

<!--en:aaacf8de-->
Bei großen dünn besetzten Systemen bilden die Lösungsverfahren nie $\mathbf{A}^{-1}$. Oft kommen sie sogar ganz ohne dicht besetzte LU-Faktoren aus. Direkte Verfahren für dünn besetzte Systeme ordnen die Unbekannten sorgfältig um, damit bei der Elimination möglichst wenige neue Einträge ungleich null entstehen. Iterative Verfahren gehen einen Schritt weiter und greifen ausschließlich über Matrix-Vektor-Produkte $\mathbf{A}\mathbf{x}$ auf die Matrix zu, ähnlich wie bei der Potenziteration (:numref:`subsec_mdl-power-iteration`). Ist $\mathbf{A}$ symmetrisch und positiv definit, so gilt für das Verfahren konjugierter Gradienten von :citet:`Hestenes.Stiefel.1952` folgende Schranke. Nach $k$ Schritten beträgt der Fehler höchstens das $2\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^k$-Fache des Anfangsfehlers, gemessen in der von $\mathbf{A}$ induzierten Norm. Dabei ist $\kappa$ die Konditionszahl aus :numref:`subsec_mdl-condition-number`. :citet:`Golub.Van-Loan.1996` behandeln das Verfahren konjugierter Gradienten und verwandte Verfahren ausführlich.

<!--en:e7fb8fcf-->
### Reihenfolge der Kontraktionen und Tensorzerlegungen

<!--en:f7838f1a-->
Die `einsum`-Notation aus :numref:`subsec_mdl-einsum` legt das mathematische Ergebnis der Kontraktion mehrerer Tensoren fest. Die Reihenfolge der paarweisen Kontraktionen bleibt offen. Diese Wahl kann den Rechenaufwand erheblich beeinflussen. Bei $n\times n$-Matrizen erfordert die Berechnung von $(\mathbf{A}\mathbf{B})\mathbf{x}$ etwa $n^3$ Multiplikationen, die von $\mathbf{A}(\mathbf{B}\mathbf{x})$ dagegen nur $2n^2$. Bei vielen Faktoren müssen wir unter den möglichen Klammerungen die günstigste suchen. `np.einsum(..., optimize=True)` und das Paket `opt_einsum` :cite:`Smith.Gray.2018` automatisieren diese Suche.

<!--en:efaab0e0-->
Die Idee der niedrigrangigen Approximation lässt sich von Matrizen auf Tensoren höherer Ordnung übertragen, allerdings mit einer ernüchternden Einschränkung. Einen Tensor dritter Ordnung können wir als Summe äußerer Produkte $\mathbf{a}_i\otimes\mathbf{b}_i\otimes\mathbf{c}_i$ vom Rang eins schreiben (CP-Zerlegung). Bei der Tucker-Zerlegung multiplizieren wir einen kleinen Kerntensor entlang jeder seiner Dimensionen mit einer Faktormatrix. Eine dritte Darstellung ist eine Kette von Kerntensoren dritter Ordnung, der Tensor Train nach :citet:`Oseledets.2011`. Für Tensoren gilt im Allgemeinen leider kein so einfacher Satz wie der von Eckart und Young. Den CP-Rang zu bestimmen ist NP-schwer, und selbst die Existenz einer besten Approximation mit vorgegebenem CP-Rang ist nicht garantiert. Der Übersichtsartikel von :citet:`Kolda.Bader.2009` ist das Standardwerk zu CP- und Tucker-Zerlegungen und den zugehörigen Algorithmen.

<!--en:058d0ee7-->
## Eigenwerte und Spektren

<!--en:e07275d5-->
### Berechnung von Eigenwerten in der Praxis

<!--en:192071c2-->
Für die Nullstellen eines allgemeinen Polynoms vom Grad fünf oder höher gibt es keine Lösungsformel in Radikalen. Daher muss jedes praktische Verfahren zur Eigenwertberechnung iterieren. Hinzu kommt, dass keine für den produktiven Einsatz bestimmte Bibliothek das charakteristische Polynom aus :numref:`subsec_mdl-finding-eigenvalues` bildet. Schon winzige Rundungsfehler in den Koeffizienten eines Polynoms können dessen Nullstellen extrem stark verschieben. Deshalb bringt `eig` die Matrix zunächst durch orthogonale Ähnlichkeitstransformationen auf Hessenberg-Form. Unterhalb der ersten Nebendiagonalen stehen dann nur Nullen. Das kostet $O(n^3)$ Operationen. Anschließend folgt der QR-Algorithmus mit Verschiebung. Wir berechnen die Zerlegung $\mathbf{H}-\mu\mathbf{I}=\mathbf{Q}\mathbf{R}$, bilden $\mathbf{R}\mathbf{Q}+\mu\mathbf{I}$ und wiederholen beide Schritte. Jeder Schritt ist eine Ähnlichkeitstransformation, sodass die Eigenwerte erhalten bleiben. Die Einträge unterhalb der Diagonalen gehen gegen null, und auf der Diagonalen zeichnet sich das Spektrum ab (bei reellen Matrizen bilden sich für komplex konjugierte Paare diagonale $2\times 2$-Blöcke). Bei symmetrischen Matrizen ist die Hessenberg-Form tridiagonal. Diese Struktur nutzt `eigh`, um die Rechnung erheblich zu beschleunigen. :citet:`Trefethen.Bau.1997` leiten den QR-Algorithmus anschaulich aus der Potenziteration auf Unterräumen her. :citet:`Golub.Van-Loan.1996` erläutern die technischen Einzelheiten der Implementierung für den praktischen Einsatz.

<!--en:60662639-->
### Pseudospektren und der Matrixsatz von Kreiss

<!--en:4f407776-->
In :numref:`subsec_mdl-pseudospectra` haben wir gesehen, dass bei einer nichtnormalen Matrix selbst Eigenwerte innerhalb der Einheitskreisscheibe ein starkes vorübergehendes Wachstum von $\|\mathbf{A}^k\|_2$ zulassen. Der Matrixsatz von Kreiss übersetzt das geometrische Bild der Pseudospektren in eine quantitative Schranke. Definieren wir die Kreiss-Konstante als $K(\mathbf{A})=\sup_{|z|>1}(|z|-1)\|(z\mathbf{I}-\mathbf{A})^{-1}\|_2$. Dann liefert der Satz für die größte Norm unter den Potenzen einer beliebigen $n\times n$-Matrix die Abschätzung $K(\mathbf{A})\le\sup_k\|\mathbf{A}^k\|_2\le e\,n\,K(\mathbf{A})$. Die untere Schranke können Sie in einer Übung in :numref:`sec_mdl-spectra-in-practice` beweisen. Die obere Schranke garantiert, dass das vorübergehende Wachstum beschränkt bleibt, wenn sich die Pseudospektren eng an die Einheitskreisscheibe schmiegen. :citet:`Trefethen.Embree.2005` entwickeln diese Theorie ausführlich. Sie behandeln auch die klassische Anwendung auf Strömungen, die in Turbulenz übergehen, obwohl alle Eigenwerte der Linearisierung strikt stabil sind.

<!--en:9172f9f3-->
### Mischzeiten und Spektrallücke

<!--en:5fbf65b9-->
Bei der Markov-Kette mit zwei Webseiten aus :numref:`sec_mdl-eigendecompositions` sinkt der Abstand zur stationären Verteilung mit jedem Klick um den Faktor 4, weil der zweite Eigenwert $1/4$ beträgt. Wie viele Schritte braucht eine allgemeine Markov-Kette zur Durchmischung? Für eine reversible Kette sei $\lambda_\star$ der größte Betrag unter allen Eigenwerten außer $1$. Wie viele Schritte braucht die Kette, bis ihr Totalvariationsabstand zur stationären Verteilung höchstens $\varepsilon$ beträgt? Diese Schrittzahl ist nach oben und unten durch Vielfache von $1/(1-\lambda_\star)$ beschränkt. Die Konstanten hängen von $\varepsilon$ und von der kleinsten stationären Wahrscheinlichkeit ab. Bei einer trägen Kette bleibt die Kette mit Wahrscheinlichkeit mindestens $1/2$ im aktuellen Zustand. Dann sind alle Eigenwerte nichtnegativ, und $\lambda_\star$ ist der zweite Eigenwert $\lambda_2$. Die Cheeger-Ungleichung liefert dann eine obere und eine untere Schranke für die Spektrallücke $1-\lambda_2$ in Abhängigkeit von der *Konduktanz* der Kette. Diese bezeichnet den kleinsten Wahrscheinlichkeitsfluss aus einer Teilmenge im Verhältnis zu ihrer stationären Masse. Die Konvergenzrate lässt sich also direkt an den Engpässen des Graphen ablesen. :citet:`Levin.Peres.2017` stellen diese Theorie besonders anschaulich dar. Sie behandeln auch die Konvergenzergebnisse, die :numref:`subsec_mdl-pagerank` aus :citet:`Horn.Johnson.2012` anführt.

<!--en:5b7bc8fd-->
### Zufallsmatrizen und trainierbare Netztiefe

<!--en:8132756a-->
In :numref:`subsec_mdl-jacobian-products` haben wir gesehen, dass die Spektralradien der Jacobi-Matrizen einzelner Schichten das Verhalten eines langen Produkts solcher Matrizen nicht bestimmen. Wovon hängt es dann ab, ob sich ein sehr tiefes Netz ausgehend von seiner Initialisierung trainieren lässt? :citet:`Saxe.McClelland.Ganguli.2014` lösen die Gleichungen für die Lerndynamik tiefer linearer Netze exakt. Sie zeigen, dass die Trainingszeit bei orthogonalen Anfangsgewichten unabhängig von der Tiefe bleibt. Bei skalierten gaußschen Anfangsgewichten verlangsamt sich das Lernen dagegen mit wachsender Tiefe. :citet:`Pennington.Schoenholz.Ganguli.2017` erweitern die Analyse auf nichtlineare Netze. Mithilfe der freien Wahrscheinlichkeitstheorie berechnen sie die gesamte Verteilung der Singulärwerte der Jacobi-Matrix der Abbildung von der Eingabe auf die Ausgabe. Sie untersuchen den Bereich der *dynamischen Isometrie*, in dem alle Singulärwerte dicht bei $1$ liegen. Sie zeigen, dass sich dieser Bereich mit orthogonalen Gewichten und $\tanh$-Aktivierungen erreichen lässt, mit ReLU hingegen nicht. Bei großer Tiefe lassen sich Netze wesentlich schneller trainieren, wenn ihre Initialisierung in diesem Bereich liegt.

<!--en:71964f0e-->
### Spektren trainierter Gewichte

<!--en:529080ae-->
Das Marchenko-Pastur-Gesetz aus :numref:`subsec_mdl-sample-covariance-mp` beschreibt das Singulärwertspektrum einer Matrix, deren Einträge aus reinem, unabhängigem Rauschen bestehen. So sehen die Gewichtsmatrizen eines frisch initialisierten Netzes aus. :citet:`Martin.Mahoney.2021` vergleichen die Spektren trainierter Gewichte neuronaler Netze mit dieser Zufallsmatrix als Referenz. Sie beobachten, dass beim Training sowohl deutliche Ausreißer als auch ein schwerer Verteilungsrand entstehen, der einem Potenzgesetz folgt. Den Exponenten am Verteilungsrand deuten sie als Kennzeichen impliziter Selbstregularisierung. Diese Abweichung vom Marchenko-Pastur-Gesetz zeigt sich in vielen Architekturen. Wer die Güte von Modellen anhand des Exponenten einstuft, ohne sie an Testdaten zu prüfen, stützt sich bislang auf eine empirische Korrelation. Einen mathematischen Beweis für diesen Zusammenhang gibt es bisher nicht.

<!--en:ca1c595c-->
## Niedriger Rang jenseits der abgeschnittenen Singulärwertzerlegung

<!--en:74a64460-->
### Matrixvervollständigung und Nuklearnorm

<!--en:89c3f7a7-->
Bei einer Matrix von Filmbewertungen sind womöglich nur wenige Prozent der Einträge bekannt. Der Satz von Eckart und Young (:numref:`subsec_mdl-eckart-young`) setzt dagegen voraus, dass wir von Anfang an die gesamte Matrix kennen. Die Matrixvervollständigung schließt diese Lücke. Wir minimieren die *Nuklearnorm* $\|\mathbf{Z}\|_*=\sum_i\sigma_i$ unter der Nebenbedingung, dass alle beobachteten Einträge erhalten bleiben. Die Nuklearnorm dient dabei als konvexer Ersatz für den Rang einer Matrix, so wie die $\ell_1$-Norm als Ersatz für die Sparsität eines Vektors dient. :citet:`Candes.Recht.2009` beweisen, dass diese konvexe Optimierung die zugrunde liegende Matrix niedrigen Ranges mit hoher Wahrscheinlichkeit exakt rekonstruiert. Dafür genügen weit weniger als $n^2$ zufällig beobachtete Einträge, sofern sich die Singulärvektoren über die Koordinaten verteilen und sich nicht auf wenige Spitzen konzentrieren. Soft-Impute :cite:`Mazumder.Hastie.Tibshirani.2010` löst die Variante dieses Problems mit Strafterm. Das Verfahren füllt abwechselnd die fehlenden Einträge auf und wendet das weiche Schwellenwertverfahren auf die Singulärwerte an.

<!--en:b8f801e1-->
### CUR-Zerlegungen und Sketching

<!--en:a57c7e9f-->
Die abgeschnittene Singulärwertzerlegung (SVD) minimiert zwar den Approximationsfehler, doch ihre Singulärvektoren vermischen sämtliche Zeilen und Spalten. Das erschwert ihre Interpretation. Zudem kostet die vollständige SVD einer $m\times n$-Matrix $O(mn\min(m,n))$ Operationen. Eine CUR-Zerlegung baut die Approximation aus Spalten $\mathbf{C}$ und Zeilen $\mathbf{R}$ der Datenmatrix selbst auf. Eine kleine Kernmatrix $\mathbf{U}$ verbindet beide. In :numref:`subsec_mdl-eckart-young` haben wir gesehen, dass die naheliegende Wahl der Spalten mit den größten Normen deutlich schlechter abschneidet als die abgeschnittene SVD. :citet:`Mahoney.Drineas.2009` zeigen, wie es besser geht. Beim Sampling von Spalten und Zeilen richten sich die Auswahlwahrscheinlichkeiten nach ihren *Hebelwerten*, den quadrierten Längen ihrer Projektionen auf den von den führenden Singulärvektoren aufgespannten Unterraum. Mit hoher Wahrscheinlichkeit übersteigt der Fehler dann den Fehler nach Eckart und Young höchstens um einen kleinen konstanten Faktor.

<!--en:b110febd-->
Das randomisierte Verfahren zur Bestimmung des Bildraums aus :numref:`subsec_mdl-spectrum-control` :cite:`Halko.Martinsson.Tropp.2011` ist ein Beispiel für die allgemeinere Idee des *Sketchings*. Wir multiplizieren eine große Matrix mit einer schmalen Zufallsmatrix und lösen das Problem anhand der wesentlich kleineren Ergebnismatrix, des sogenannten Sketchs. :citet:`Woodruff.2014` gibt einen Überblick darüber, wie dieselbe Idee die Lösung von Problemen der kleinsten Quadrate und die Regression beschleunigt. Ein Sketch mit weit weniger Zeilen als der ursprüngliche Datensatz liefert eine Lösung, deren Zielfunktionswert höchstens um den Faktor $1+\varepsilon$ über dem Optimum liegt.

<!--en:660c6dd9-->
### Adaptervarianten

<!--en:7c0a009a-->
LoRA (:numref:`subsec_mdl-low-rank-updates`) initialisiert die Aktualisierung mit niedrigem Rang $\mathbf{B}\mathbf{A}$ mit $\mathbf{B}=\mathbf{0}$ und einer zufälligen Matrix $\mathbf{A}$. Bei den ersten Gradientenschritten geht die Eingabe daher nur über einen zufälligen $k$-dimensionalen Unterraum ein, der keinen Bezug zur vortrainierten Gewichtsmatrix hat. Mehrere Varianten ändern diese Initialisierung oder die Parametrisierung auf einfache Weise. PiSSA :cite:`Meng.Wang.Zhang.2024` initialisiert $\mathbf{B}\mathbf{A}$ mit der auf Rang $k$ abgeschnittenen SVD der vortrainierten Gewichtsmatrix und hält das Residuum fest. So passt das Fine-Tuning von Anfang an die Hauptrichtungen der ursprünglichen Matrix an. DoRA :cite:`Liu.Wang.Yin.ea.2024` zerlegt jede Spalte der Gewichtsmatrix in einen skalaren Betrag und einen Einheitsvektor, der die Richtung angibt. Das Verfahren trainiert die Beträge direkt und wendet die Aktualisierung mit niedrigem Rang nur auf die Richtungen an.

<!--en:c20051e6-->
### Niedriger Rang bei Attention

<!--en:b677cf9a-->
Self-Attention bildet für eine Sequenz aus $n$ Token eine $n\times n$-Matrix der Attention-Scores. Mehrere effiziente Architekturen aus den Kapiteln über Attention ersetzen diese Matrix oder den zugehörigen KV-Cache durch eine niedrigrangige Faktorisierung. Lineare Attention (:numref:`sec_windowed-linear-attention`) drückt die Attention-Scores als Skalarprodukte von Merkmalsvektoren $\phi(\mathbf{q})\cdot\phi(\mathbf{k})$ aus. Dadurch ist der Rang der Score-Matrix höchstens so groß wie die Merkmalsdimension. Multi-Head Latent Attention (:numref:`subsec_kv-cache-low-rank`) :cite:`DeepSeek-AI.2024` komprimiert den Cache. Pro Token speichert sie einen einzigen niedrigdimensionalen latenten Vektor anstelle separater Schlüssel und Werte für jeden Kopf. Linformer :cite:`Wang.Li.Khabsa.ea.2020` geht von der empirischen Beobachtung aus, dass die Singulärwerte trainierter Attention-Matrizen rasch abfallen. Das Verfahren projiziert die Schlüssel und Werte entlang der Sequenzdimension auf eine feste, kürzere Länge.

<!--en:89b36b62-->
### Steilster Abstieg in anderen Normen

<!--en:48a7c0f1-->
In :numref:`subsec_mdl-orthogonalized-updates` haben wir den Polarfaktor $\mathbf{U}\mathbf{V}^\top$ einer Gradientenmatrix berechnet. Geometrisch gibt diese Matrix die Richtung des steilsten Abstiegs an, wenn wir die Schrittweite in der Spektralnorm statt in der euklidischen Norm messen. :citet:`Bernstein.Newhouse.2024` leiten aus dieser Beobachtung eine gemeinsame Sicht auf Optimierer für Deep Learning ab. Schalten wir die gleitenden Mittelwerte ab, reduziert sich Adam auf den Vorzeichenabstieg (den steilsten Abstieg in der $\ell_\infty$-Norm) und Shampoo auf den steilsten Abstieg in der Spektralnorm. Aus dieser Sicht besteht die zentrale Entscheidung beim Entwurf darin, für jede Schicht die passende Norm zu wählen. Die Aktualisierungsregel ergibt sich dann von selbst. In :numref:`sec_muon` entwickeln wir den auf dieser Idee beruhenden Optimierer für die Spektralnorm.

<!--en:4bf0c8ae-->
## Bücher, Kurse und Skripte

<!--en:f3e47eb0-->
**Bücher**

<!--en:da56012a-->
- [Mathematics for Machine Learning](https://mml-book.github.io/) von Deisenroth, Faisal und Ong ist ein frei zugängliches Lehrbuch. Die Kapitel 2 (lineare Algebra), 3 (analytische Geometrie) und 4 (Matrixzerlegungen) behandeln den Stoff dieses Kapitels aus der Sicht des maschinellen Lernens.

<!--en:b9fd7f74-->
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/) von Boyd und Vandenberghe ist ein frei zugängliches Buch mit Folien und Code. Im Mittelpunkt stehen die Methode der kleinsten Quadrate und die Arbeit mit Daten. Das Buch ergänzt :numref:`sec_mdl-geometry-linear-algebraic-ops` und :numref:`sec_mdl-svd-low-rank`.

<!--en:8a9663f0-->
- [Introduction to Linear Algebra, 6. Auflage](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html) und [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) von Gilbert Strang. Die klassische Einführung geht von der Spaltendarstellung und den vier fundamentalen Unterräumen aus. Der Begleitband stellt die SVD in den Mittelpunkt der Datenanalyse und Optimierung.

<!--en:1bdf237e-->
- [Linear Algebra Done Right, 4. Auflage](https://linear.axler.net/) von Sheldon Axler ist frei zugänglich und behandelt Eigenwerte, Operatoren und den Spektralsatz (:numref:`sec_mdl-eigendecompositions`) ohne Rückgriff auf Determinanten.

<!--en:0fc9888a-->
- [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html) von Trefethen und Bau erklärt verständlich die Konditionierung, die Stabilität, die QR-Zerlegung und die Gleitkommaalgorithmen hinter `solve`, `eig` und `svd`.

<!--en:6eeb657c-->
**Kurse und Tutorials**

<!--en:ab5ecc55-->
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) und [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) von Gilbert Strang sind vollständige Vorlesungen mit Skripten und Übungsblättern.

<!--en:72749a6f-->
- [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) von 3Blue1Brown und [Singular Value Decomposition](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv) von Steve Brunton veranschaulichen in Videovorlesungen lineare Abbildungen, Determinanten, Eigenvektoren und die SVD.

<!--en:f2a84ae3-->
- [Linear Algebra Review and Reference](https://cs229.stanford.edu/section/cs229-linalg.pdf) von Kolter und Do (Stanford CS229) und [Immersive Linear Algebra](http://immersivemath.com/ila/index.html) von Ström, Åström und Akenine-Möller.

<!--en:161de105-->
[Diskussion](https://d2l.discourse.group/t/36856)
