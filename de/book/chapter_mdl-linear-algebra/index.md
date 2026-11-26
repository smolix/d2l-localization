<!--en:5a7d784a-->
# Lineare Algebra

<!--en:0c6eb0f1-->
Jedes neuronale Netz in diesem Buch beruht auf Matrixprodukten. Eine lineare Schicht multipliziert ihre Eingabe mit einer Gewichtsmatrix. Bei der Attention werden Abfragen mit Schlüsseln multipliziert. Ein rekurrentes Netz multipliziert seinen versteckten Zustand in jedem Schritt mit derselben Matrix. In :numref:`sec_linear-algebra` haben wir diese Produkte berechnet, ohne zu fragen, was eine Matrix mit den Vektoren *macht*, auf die sie wirkt. In diesem optionalen Kapitel gehen wir dieser geometrischen Frage nach und führen auch die Beweise. Der Haupttext setzt das Kapitel an keiner Stelle voraus. Wer die Geometrie hinter den Matrixprodukten erkennt, sieht jedoch fast jedes Modell im Buch mit anderen Augen.

<!--en:11da4ad8-->
Betrachten wir ein Fashion-MNIST-Bild mit 784 Pixeln, das eine Schicht auf 256 Merkmale abbildet. Wie ähnlich sind sich zwei Bilder? Wie viel Information verwirft die Schicht? Was geschieht, wenn wir dieselbe lineare Transformation fünfzigmal hintereinander anwenden? In 784 Dimensionen können wir nicht zeichnen. Die Antworten liefert daher die Algebra, mit der wir die Geometrie der Ebene auf beliebig viele Dimensionen übertragen.

<!--en:cde83739-->
Selbst eine komplizierte Matrix wird verständlich, wenn wir sie entlang geeigneter Richtungen betrachten. Um 1810 passte Gauß die Bahn des Asteroiden Pallas mit der Methode der kleinsten Quadrate an Beobachtungsdaten an. Dabei hatte er mehr Beobachtungen als Unbekannte. Ein solches Problem lösen wir im ersten Abschnitt und greifen es im letzten wieder auf. In einem orthonormalen Koordinatensystem lassen sich Längen und Winkel auf Skalarprodukte zurückführen. In Richtung jedes Eigenvektors skaliert eine quadratische Matrix mit einem einzigen Skalar. Wählen wir für Ausgangs- und Zielraum jeweils eine Basis aus Singulärvektoren, so zerfällt *jede* Matrix in unabhängige Streckungen. Damit lässt sich erklären, warum die Attention ihre Scores durch $\sqrt{d}$ teilt (:numref:`sec_attention-scoring-functions`), wann die Gradienten in rekurrenten Netzen explodieren (:numref:`sec_bptt`) und was ein Adapter mit niedrigem Rang darstellen kann.

<!--en:f7f9125b-->
Wir gehen in zwei Schritten vor. Zuerst wenden wir eine Matrix ein einziges Mal an. In :numref:`sec_mdl-geometry-linear-algebraic-ops` übertragen wir Längen, Winkel und orthogonale Projektionen auf höhere Dimensionen. Die orthogonale Projektion liefert den nächstgelegenen Punkt in einem Unterraum. In :numref:`sec_mdl-linear-maps` untersuchen wir, welche Richtungen auf null abgebildet werden und wann eine Abbildung invertierbar ist. Im zweiten Schritt wählen wir Koordinaten, die zur Matrix passen. Wir betrachten Eigenvektoren und den Spektralsatz für symmetrische Matrizen (:numref:`sec_mdl-eigendecompositions`). Dann untersuchen wir das Verhalten nichtnormaler Matrizen, das sich an den Eigenwerten allein nicht ablesen lässt (:numref:`sec_mdl-spectra-in-practice`). Schließlich zeigen wir, wie Singulärvektoren für jede Matrixform eine optimale Approximation mit niedrigem Rang liefern, wenn wir die Zerlegung abschneiden (:numref:`sec_mdl-svd-low-rank`). An zwei kleinen Matrizen prüfen wir unsere Überlegungen im ganzen Kapitel. Die Matrix $\mathbf{A}=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ hat komplexe Eigenwerte. Die Eigenvektoren der Scherung $\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ spannen die Ebene nicht auf.

<!--en:3796f690-->
Bei der ersten Lektüre können Sie die Ausführungen zu $n$-dimensionalen Determinanten sowie die Erörterung von Gerschgorin-Kreisen, PageRank und Pseudospektren in :numref:`sec_mdl-spectra-in-practice` überfliegen. Einen Überblick über numerische Verfahren für Eigenwertprobleme und lineare Gleichungssysteme bietet :numref:`sec_mdl-linear-algebra-further-reading`.

<!--en:21595c10-->
Lineare Algebra: Geometrie, Spektren und Strukturen mit niedrigem Rang<br>**Projektionen · lineare Transformationen · Eigenwertzerlegungen · spektrale Stabilität · Singulärwertzerlegung**

<!--en:8e674e67-->
::: {.slide title="Warum Deep Learning die Geometrie der linearen Algebra braucht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:499aa373-->
- **Mehr als Ausmultiplizieren:** Die Geometrie von $\mathbf{X}\mathbf{W}$ zeigt, was eine Schicht in $\mathbb{R}^d$ erhält, auf null abbildet oder verstärkt.

<!--en:fe24de8c-->
- **Anschauung in hohen Dimensionen:** Mit Skalarprodukten lassen sich Längen, Winkel und orthogonale Projektionen auch für Bilder mit 784 Pixeln oder 4096-dimensionale Einbettungen bestimmen.

<!--en:b186d65d-->
- **Wahl natürlicher Achsen:** Orthonormalbasen von Unterräumen, Eigenvektoren oder Singulärvektoren machen die Wirkung einer Matrix anschaulich.

<!--en:e9eba208-->
::: {.slide title="In zwei Schritten von einzelnen Abbildungen zu natürlichen Achsen"}

<!--en:ddfd1d8f-->
[Kapitelaufbau]{.kicker}

<!--en:a8f4aa03-->
- **Schritt 1: Standardkoordinaten (§§{sec_mdl-geometry-linear-algebraic-ops}–§{num:sec_mdl-linear-maps}):** Ähnlichkeit messen, auf Unterräume projizieren ($\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$, $\mathbf{A} = \mathbf{Q}\mathbf{R}$) und bestimmen, wie viele Dimensionen erhalten bleiben und wie viele verloren gehen ($\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n$, $\det\mathbf{A}$)

<!--en:39983e36-->
- **Schritt 2: An die Matrix angepasste Koordinaten (§§{sec_mdl-eigendecompositions}–§{num:sec_mdl-svd-low-rank}):** Quadratische Matrizen für die wiederholte Anwendung diagonalisieren ($\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$), Potenziteration und Wachstum bei nichtnormalen Matrizen untersuchen und jede $m \times n$-Matrix mittels SVD zerlegen ($\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$)

<!--en:632b51b4-->
::: {.slide title="Drei zentrale Sätze aus Kapitel §{num:chap_mdl-linear-algebra}"}

<!--en:2c5478a1-->
[Theoretische Grundlagen]{.kicker}

<!--en:af4f3570-->
- **1. Orthogonale Projektion ($\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$):** Für den nächstgelegenen Punkt in $S$ erfüllt das Residuum die Beziehung $\mathbf{y} - \mathbf{P}\mathbf{y} \perp S$. So entsteht beim Problem der kleinsten Quadrate ein rechtwinkliges Dreieck.

<!--en:5fa31978-->
- **2. Spektralsatz ($\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$):** Jede reelle symmetrische Matrix hat reelle Eigenwerte und eine Orthonormalbasis aus Eigenvektoren.

<!--en:824da38e-->
- **3. Satz von Eckart–Young–Mirsky ($\mathbf{A}_k$):** Das Abschneiden der SVD bei Rang $k$ liefert die optimale Approximation vom Rang $k$ sowohl in der Spektralnorm als auch in der Frobeniusnorm.

<!--en:f71b0cda-->
::: {.slide title="Wiederkehrende Beispielmatrizen und empirische Vergleichswerte"}

<!--en:05adf916-->
[Roter Faden]{.kicker}

<!--en:3662cffc-->
- **Testmatrizen der Größe $2 \times 2$:** Die wiederkehrende Beispielmatrix $\mathbf{A}$ (komplexe Eigenwerte $\lambda = 2 \pm i$, $\det = 5$), die nicht diagonalisierbare Scherung $\mathbf{T}$ ($g = 1 < a = 2$, lineare Drift) und die singuläre Matrix $\mathbf{B}$ vom Rang 1

<!--en:14c9a2bf-->
- **Hochdimensionale Daten aus der Praxis:** Fashion-MNIST in $\mathbb{R}^{784}$ ($92\%$ Genauigkeit mit der Trennhyperebene zwischen den Klassenmittelwerten, nichtisotrope Kosinuswerte) und ein Katzenfoto mit $400 \times 500$ Pixeln ($9\%$ des ursprünglichen Speicherbedarfs für die SVD vom Rang 20 bei $8\%$ Fehler)

<!--en:15d021a2-->
::: {.slide title="Orientierung im Kapitel"}

<!--en:123f19cf-->
[Lesehinweise]{.kicker}

<!--en:8bad973b-->
- **Schwerpunkte bei der ersten Lektüre:** Orthogonale Projektionen und Skalierung der Attention-Scores mit $1/\sqrt{d}$ (§{sec_mdl-geometry-linear-algebraic-ops}), Spaltenraum, Nullraum und Rangsatz (§{sec_mdl-linear-maps}), Spektralsatz und positive Definitheit (§{sec_mdl-eigendecompositions}), Potenziteration und transientes Wachstum (§{sec_mdl-spectra-in-practice}) sowie das gesamte Vorgehen bei SVD, LoRA und der Untersuchung der Konditionierung (§{sec_mdl-svd-low-rank})

<!--en:17f74952-->
- **Zum Überfliegen bei der ersten Lektüre:** Die axiomatische Herleitung der $n$-dimensionalen Determinante über Permutationen (§{sec_mdl-linear-maps}), Gerschgorin-Kreise, PageRank und $\varepsilon$-Pseudospektren (§{sec_mdl-spectra-in-practice}) lassen sich überspringen, ohne den roten Faden zu verlieren.

<!--en:fc711756-->
- **Reelle und komplexe Räume:** Ausgangspunkt sind durchweg reelle Vektorräume $\mathbb{R}^d$. Komplexe Zahlen $\mathbb{C}$ kommen erst ins Spiel, wenn reelle Matrizen wie Drehmatrizen oder $\mathbf{A}$ komplexe Eigenwerte erzwingen.

<!--en:67b05a19-->
- **Numerische Praxis:** Jeder Abschnitt verbindet exakte Sätze mit den Bedingungen der Gleitkommaarithmetik. Daraus ergibt sich, warum es sinnvoll ist, Kosinuswerte auf $[-1, 1]$ zu begrenzen, QR oder SVD anstelle der Normalgleichungen zu verwenden, `solve` statt `inv` aufzurufen und sehr kleine Singulärwerte anhand eines Schwellenwerts auf null zu setzen.

<!--en:43c83f81-->
**Im nächsten Abschnitt (§{sec_mdl-geometry-linear-algebraic-ops} Vektoren, Winkel und Projektionen):** Wie das Skalarprodukt Längen, Winkel, orthogonale Projektionen und trennende Hyperebenen aus der Ebene auf 784 Dimensionen überträgt.
