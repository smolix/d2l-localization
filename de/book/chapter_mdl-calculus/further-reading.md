<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:d2ffd1d7-->
In diesem Kapitel haben wir die wichtigsten Ergebnisse unter vereinfachenden Annahmen bewiesen, an deren Grenzen wir beim Deep Learning regelmäßig stoßen. Ein ReLU-Netz hat zahlreiche Knickstellen, die Liste der Zwischenwerte im Rückwärtsmodus kann leicht den GPU-Speicher sprengen, und ein tausenddimensionales Integral lässt sich in der Praxis nicht auf einem Gitter auswerten. Die folgenden Anmerkungen fassen zusammen, was geschieht, wenn diese Annahmen nicht mehr gelten. Anschließend stellen wir ausgewählte Lehrbücher, Kurse und Tutorials zu den Kernthemen des Kapitels vor. Zwei Themen lohnen sich als Erstes, weil sie das Training unmittelbar betreffen. Konservative Felder (engl. conservative fields) erklären, in welchem Sinne Backpropagation auch an Knickstellen gültig bleibt. Rekursives Checkpointing begrenzt den Speicherbedarf bei langen Berechnungen. Kein anderer Teil des Buches setzt diesen Abschnitt voraus.

<!--en:6b973d86-->
## Ableitungen an Knickstellen

<!--en:aff6a25c-->
### Konservative Felder

<!--en:358b97f6-->
Wie :numref:`sec_mdl-single_variable_calculus` zeigt, führt die übliche Konvention $\mathrm{ReLU}'(0)=0$ zu einem merkwürdigen Ergebnis. Schreiben wir die Identitätsfunktion als $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$, so liefert Backpropagation die Steigung $0$ an der Stelle $x=0$. Dieser Wert ist weder die tatsächliche Ableitung noch ein gültiger Subgradient von $x$. Klassische Subdifferentiale können dieses Verhalten nicht erklären. Sobald die beteiligten Funktionen nicht mehr konvex sind, gelten ihre Summen- und Kettenregeln nur als Mengeninklusionen. Um diese Lücke zu schließen, führen :citet:`Bolte.Pauwels.2021` *konservative Felder* ein. Diese mengenwertigen Verallgemeinerungen der Ableitung erfüllen die Kettenregel bereits aufgrund ihrer Konstruktion. Außerhalb einer Nullmenge stimmen sie mit dem gewöhnlichen Gradienten überein. Legen wir an jeder Knickstelle eine beliebige Steigung fest, so liefert Backpropagation stets ein Element eines solchen Feldes. Warum ist die Ausnahmemenge der Knickstellen so klein, dass sie kein Problem darstellt? Nach dem Satz von Rademacher :cite:`Rademacher.1919` ist jede Lipschitz-stetige Funktion auf $\mathbb R^n$ fast überall differenzierbar. Die Knickstellen der Verlustfunktion eines ReLU-Netzes bilden also eine Nullmenge.

<!--en:303b85df-->
Dieselbe Arbeit klärt auch die für die Praxis wichtige Frage, ob der stochastische Gradientenabstieg (engl. stochastic gradient descent) mit diesen durch Backpropagation berechneten Werten weiterhin konvergiert. Betrachten wir Verlustfunktionen, die aus den üblichen Schichten neuronaler Netze aufgebaut sind. Sind die Iterierten beschränkt und gelten die üblichen Bedingungen an Schrittweiten und Gradientenrauschen, so konvergieren die Iterierten gegen die Menge der kritischen Punkte, an denen das konservative Feld die Null enthält. Diese Garantie gilt bei exakter Arithmetik. In Gleitkommaarithmetik mit endlicher Genauigkeit treffen die Iterierten tatsächlich auf Knickstellen. :numref:`sec_mdl-single_variable_calculus` gibt für drei Rechengenauigkeiten an, wie oft das geschieht. Die Zahlen stammen aus den Messungen von :citet:`Bertoin.Bolte.Gerchinovitz.ea.2021` und berücksichtigen deren Korrektur von 2023. Der Abschnitt zeigt auch, wie wenig sich die Wahl von $\mathrm{ReLU}'(0)$ letztlich auf das Training auswirkt.

<!--en:6e481674-->
### Subgradientenverfahren

<!--en:862dfc2d-->
Die glatte Optimierung stützt sich auf das Abstiegslemma (engl. descent lemma). Es garantiert, dass ein kleiner Schritt entgegen dem Gradienten den Zielfunktionswert senkt. Für nichtglatte Funktionen entfällt diese Garantie. Am Minimum von $|x|$ zeigt der gültige Subgradient $\tfrac12$ sogar bergauf, wie :numref:`sec_mdl-single_variable_calculus` zeigt. Bemerkenswerterweise konvergiert das Subgradientenverfahren dennoch. Für den Nachweis verfolgen wir den euklidischen Abstand zu einem Minimierer anstelle des Zielfunktionswerts. Bei abnehmenden Schrittweiten mit $\sum_t\eta_t=\infty$ und $\sum_t\eta_t^2<\infty$ konvergiert der bislang beste Zielfunktionswert gegen das Minimum. Für die optimale Konvergenzrate wählen wir die Schrittweite anders. Betrachten wir eine konvexe Funktion, deren Subgradienten höchstens die Norm $G$ haben. Wir starten im Abstand $R$ von einem Minimierer. Mit der konstanten Schrittweite $R/(G\sqrt T)$ liegt der beste Zielfunktionswert unter den ersten $T$ Iterierten höchstens $RG/\sqrt T$ über dem Minimum. Übersteigt die Dimension $T$, so kann kein Verfahren, das ausschließlich Subgradienten nutzt, diese Rate um mehr als einen konstanten Faktor verbessern :cite:`Nesterov.2018`. Eine systematische Einführung in die Subdifferentialrechnung gibt :citet:`Rockafellar.1970`. Dort finden sich auch die genauen Bedingungen, unter denen die Summenregel als Gleichheit gilt.

<!--en:44bd0467-->
## Ableitungen vektor- und matrixwertiger Abbildungen

<!--en:f29c7da0-->
### Der Mittelwertsatz für vektorwertige Abbildungen

<!--en:4a2725c0-->
In einer Dimension garantiert der Mittelwertsatz aus :numref:`sec_mdl-single_variable_calculus` einen Zwischenpunkt $\xi$ mit $f(b)-f(a)=f'(\xi)(b-a)$. Es liegt nahe, dieselbe Gleichung für vektorwertige Kurven $\boldsymbol\gamma:\mathbb R\to\mathbb R^n$ zu vermuten. Doch schon der Einheitskreis $\boldsymbol\gamma(t)=(\cos t,\sin t)$ liefert ein Gegenbeispiel. Nach einer vollen Umdrehung bei $t=2\pi$ erreicht die Kurve wieder ihren Ausgangspunkt, also gilt $\boldsymbol\gamma(2\pi)-\boldsymbol\gamma(0)=\mathbf 0$. Der Geschwindigkeitsvektor $\boldsymbol\gamma'(\xi)$ hat jedoch die Länge eins und ist damit ungleich $\mathbf 0$ für jedes $\xi$. In höheren Dimensionen bleibt nur die *Mittelwertungleichung* (engl. mean value inequality) $\|\boldsymbol\gamma(b)-\boldsymbol\gamma(a)\|\le\sup_\xi\|\boldsymbol\gamma'(\xi)\|\,|b-a|$ (Satz 5.19 in :citet:`Rudin.1976`). Für die Taylor-Schranken in :numref:`sec_mdl-multivariable_calculus` brauchen wir zum Glück keine vektorwertige Fassung des Satzes. Wir schränken dort eine skalarwertige Verlustfunktion auf eine Strecke ein. Auf dieser gilt der gewöhnliche Mittelwertsatz für eine Variable unmittelbar.

<!--en:9d2f8415-->
### Jacobi-Matrizen matrixwertiger Abbildungen

<!--en:792c9e0f-->
In :numref:`sec_mdl-matrix-calculus` differenzieren wir skalare Verlustfunktionen. Daher hat jeder Gradient praktischerweise dieselbe Form wie seine Matrixvariable. Leiten wir jedoch eine matrixwertige Funktion nach einer Eingabematrix ab, so erhalten wir als vollständige Ableitung einen Tensor mit vier Indizes. Um mit der vertrauten linearen Algebra weiterzurechnen, schreiben wir beide Matrizen mithilfe des Operators $\operatorname{vec}$ als Vektoren, indem wir ihre Spalten untereinander anordnen. So wird aus dem Tensor mit vier Indizes eine gewöhnliche Jacobi-Matrix. Das Matrixdifferential $d\mathbf F=\mathbf A\,d\mathbf X\,\mathbf B$ nimmt dabei die Form $d\operatorname{vec}\mathbf F=(\mathbf B^\top\otimes\mathbf A)\,d\operatorname{vec}\mathbf X$ an. Bei Anwendung der Kettenregel multiplizieren wir dann nur noch Kronecker-Produkte. Bei :citet:`Giles.2008b` finden sich die Ableitungsregeln für Matrixprodukte, Inverse, Determinanten und gängige Matrixzerlegungen zum Nachschlagen, jeweils für den Vorwärts- und den Rückwärtsmodus.

<!--en:0a27cc09-->
## Automatisches Differenzieren

<!--en:e876fc98-->
### Optimale Berechnung von Jacobi-Matrizen

<!--en:e9f3b586-->
Wie :numref:`sec_mdl-matrix-calculus` zeigt, kann allein die Reihenfolge, in der wir eine Kette von Jacobi-Matrizen multiplizieren, die Rechenkosten um einen Faktor verändern, der etwa der Schichtbreite entspricht. Bei einer unverzweigten Kette aus $K$ Faktoren ist die Suche nach der günstigsten Klammerung das klassische Problem der Matrixkettenmultiplikation. Dynamische Programmierung löst es in $O(K^3)$ Zeit. Bei einem allgemeinen Berechnungsgraphen mit Verzweigungen und Skip-Verbindungen gibt es weit mehr mögliche Eliminationsreihenfolgen. Wir können die Zwischenknoten in beliebiger Reihenfolge eliminieren, indem wir jeweils die partiellen Ableitungen an den ein- und ausgehenden Kanten multiplizieren. Die Suche nach der Eliminationsreihenfolge mit den wenigsten Multiplikationen erweist sich als NP-vollständig :cite:`Naumann.2008`. Deep-Learning-Frameworks verzichten deshalb auf eine globale Optimierung der Reihenfolge. Sie bieten den Vorwärtsmodus, den Rückwärtsmodus und deren Kombinationen an. Den Modus wählen die Anwender selbst. :citet:`Griewank.Walther.2008` behandeln die Elimination von Knoten, Kanten und Flächen in voller Allgemeinheit. Sie geben auch einen Überblick über die Heuristiken, mit denen Compiler zur Ableitungsberechnung arbeiten.

<!--en:a06624e4-->
### Rekursives Checkpointing

<!--en:094675ad-->
Das einstufige Checkpointing aus :numref:`sec_mdl-matrix-calculus-autodiff` senkt den maximalen Bedarf an Aktivierungsspeicher auf $T/s+s$ gespeicherte Zustände. Das Minimum von $2\sqrt T$ erreichen wir mit der optimalen Blockgröße $s$. Dafür ist ein zusätzlicher Vorwärtsdurchlauf nötig :cite:`Chen.Xu.Zhang.ea.2016`. Bei sehr langen Sequenzen oder entrollten iterativen Lösungsverfahren kann selbst ein Speicherbedarf von $O(\sqrt T)$ zu hoch sein. Wenden wir dieselbe Idee innerhalb jedes neu berechneten Segments rekursiv an, so sinkt der Speicherbedarf auf $O(\log T)$. Der Aufwand für die Neuberechnung beträgt dabei $O(T\log T)$ :cite:`Griewank.1992`. Wenn jeder Schritt gleich viel kostet, ist Griewanks binomialer Checkpointing-Plan für jedes feste Speicherbudget nachweislich optimal :cite:`Griewank.Walther.2008`. In der Praxis verwenden Checkpointing-Werkzeuge eine Stufe pro markiertem Block.

<!--en:17f669c1-->
PyTorch stellt `torch.utils.checkpoint` bereit.

<!--en:27148441-->
TensorFlow stellt `tf.recompute_grad` bereit.

<!--en:9ff8db7f-->
JAX stellt `jax.checkpoint` bereit. Durch Verschachteln dieser Blöcke lässt sich ein rekursiver Plan von Hand erstellen.

<!--en:ddd1af4b-->
### Taylor-Modus

<!--en:73310c4e-->
Eine Richtungsableitung der Ordnung $k$ durch $k$-faches Verschachteln des automatischen Differenzierens im Vorwärtsmodus zu berechnen, ist aufwendig. Jede weitere Ebene verdoppelt den Zustand, sodass wir für jeden Wert $2^k$ Komponenten mitführen. Der *Taylor-Modus* vermeidet dieses exponentielle Wachstum. Dazu ersetzt er duale Zahlen durch abgebrochene Potenzreihen $a_0+a_1\varepsilon+\cdots+a_k\varepsilon^k$ mit $\varepsilon^{k+1}=0$. Jede elementare Operation verarbeitet alle $k+1$ Koeffizienten in einem einzigen Durchlauf. Bei der Multiplikation werden die beiden Koeffizientenfolgen in $O(k^2)$ Operationen gefaltet. Für transzendente Funktionen wie $\exp$, $\log$ und $\sin$ ergeben sich die Koeffizienten aus Rekursionsformeln mit denselben Kosten von $O(k^2)$ :cite:`Griewank.Walther.2008`. Die Übung zur Berechnung zweiter Ableitungen in einem Vorwärtsdurchlauf in :numref:`sec_mdl-matrix-calculus-autodiff` behandelt den Fall $k=2$.

<!--en:1ffe76e6-->
`jax.experimental.jet` implementiert den Taylor-Modus für jede beliebige Ordnung $k$.

<!--en:d48bdc26-->
### Vorwärtsgradienten

<!--en:40b1e987-->
Beim Differenzieren im Rückwärtsmodus müssen wir eine Liste von Zwischenwerten speichern. Der Vorwärtsmodus kommt ohne diese Liste aus, benötigt aber einen Durchlauf pro Eingabeparameter. :citet:`Baydin.Pearlmutter.Syme.ea.2022` umgehen beide Kosten. Sie berechnen einen einzigen Vorwärtsdurchlauf entlang eines gaußverteilten zufälligen Tangentenvektors $\mathbf v\sim\mathcal N(\mathbf 0,\mathbf I)$ und verwenden $(\nabla\ell^\top\mathbf v)\,\mathbf v$ als stochastische Schätzung des Gradienten. Wegen $\mathbb{E}[\mathbf v\mathbf v^\top]=\mathbf I$ ist der Schätzer erwartungstreu. Bei $n$ Parametern beträgt sein mittlerer quadratischer Fehler $(n+1)\|\nabla\ell\|^2$. Die Liste der Zwischenwerte entfällt vollständig. Dafür beträgt der typische Schätzfehler etwa das $\sqrt n$-Fache der Norm des tatsächlichen Gradienten. Bei kleinen Modellen ist das vertretbar, bei der Größe moderner neuronaler Netze ist der Fehler jedoch zu groß.

<!--en:3484711e-->
## Integration

<!--en:76ccb530-->
### Das Lebesgue-Integral

<!--en:3afd9d00-->
Beim Riemann-Integral zerlegen wir den Definitionsbereich in Intervalle. Bei Funktionen mit ausgeprägten Unstetigkeiten stößt dieses Vorgehen an seine Grenzen. Wie :numref:`sec_mdl-integral_calculus` zeigt, haben die Riemann-Summen der Indikatorfunktion der rationalen Zahlen keinen von der Wahl der Stützstellen unabhängigen Grenzwert. Selbst der punktweise Grenzwert Riemann-integrierbarer Funktionen muss nicht Riemann-integrierbar sein. Lebesgues Idee war, den Wertebereich zu unterteilen. Für jede Höhe messen wir, wie groß die Menge der Punkte ist, an denen $f$ diese Höhe überschreitet, und summieren diese Maße auf. Nullmengen tragen nichts zum Integral bei. Dazu gehören die rationalen Zahlen ebenso wie der Strahl $\theta=0$ in Polarkoordinaten. Das Integral der Indikatorfunktion der rationalen Zahlen ist daher $0$. In :numref:`sec_mdl-integral_calculus` und :numref:`sec_mdl-expectation-gradients` verwenden wir drei grundlegende Sätze dieser Theorie ohne Beweis. Es sind der Satz von Fubini-Tonelli, der Transformationssatz für die $n$-dimensionale Variablentransformation und der Satz über das Differenzieren unter dem Integralzeichen auf einem unbeschränkten Integrationsbereich. Alle drei beruhen auf den Sätzen von der monotonen und der dominierten Konvergenz. Diese geben an, wann wir Grenzwert und Integral vertauschen dürfen. Gilt $f_k\to f$ punktweise und gilt $|f_k|\le g$ mit ein und derselben integrierbaren Funktion $g$, so folgt $\int f_k\to\int f$ :cite:`Folland.1999`.

<!--en:aa9afb07-->
### Quasi-Monte-Carlo-Verfahren

<!--en:380436fc-->
In :numref:`subsec_mdl-monte-carlo` haben wir die Gitterintegration zweiter Ordnung, deren Fehler wie $N^{-2/d}$ abnimmt, mit der Monte-Carlo-Integration verglichen. Bei dieser nimmt der Fehler unabhängig von der Dimension wie $N^{-1/2}$ ab. Bei reinen Zufallsstichproben entstehen jedoch zwangsläufig zufällige Häufungen und Lücken. Punktmengen mit *geringer Diskrepanz* (engl. low discrepancy) sind so konstruiert, dass sie den Einheitswürfel viel gleichmäßiger ausfüllen. Beispiele liefern Sobol-Folgen, Halton-Folgen und Gitterregeln. Für Integranden mit beschränkter Variation im Sinne von Hardy und Krause erzielen sie einen Fehler von $O(N^{-1}(\log N)^d)$. Als Garantie für den ungünstigsten Fall ist diese Schranke ernüchternd. Vernachlässigen wir konstante Faktoren, so fällt sie erst unter $N^{-1/2}$, wenn die Punktzahl mehrere Tausend für $d=2$ beziehungsweise $10^{15}$ für $d=5$ übersteigt. In der Praxis liefern Quasi-Monte-Carlo-Verfahren selbst in hohen Dimensionen häufig bessere Ergebnisse als gewöhnliche Monte-Carlo-Verfahren. Das liegt meist daran, dass bei realen Integranden jeweils nur wenige Koordinaten eine wesentliche Rolle spielen. Eine deterministische Punktmenge liefert von sich aus keinen Fehlerbalken. *Randomisierte Quasi-Monte-Carlo-Verfahren* verschieben die Punkte daher zufällig oder permutieren ihre Koordinatenziffern. Aus unabhängigen Wiederholungen ergibt sich so ein erwartungstreuer Varianzschätzer. :citet:`Owen.2013` behandelt in den Kapiteln 15–17 sowohl deterministische als auch randomisierte Verfahren.

<!--en:26a4906a-->
### Gradientenschätzer jenseits von Score-Funktionsschätzern und pfadweisen Schätzern

<!--en:aca1b809-->
In :numref:`sec_mdl-expectation-gradients` haben wir den Score-Funktionsschätzer und den pfadweisen Schätzer hergeleitet. Mit einer konstanten Baseline haben wir die Varianz des Score-Funktionsschätzers reduziert. Ausgefeiltere Kontrollvariaten nutzen die Struktur des Modells selbst. REBAR :cite:`Tucker.Mnih.Maddison.ea.2017` zieht den pfadweisen Gradienten einer stetigen Relaxierung vom Score-Funktionsschätzer ab und addiert den analytisch berechneten Erwartungswert des pfadweisen Gradienten wieder hinzu. Dadurch sinkt die Varianz, und der Schätzer bleibt für diskrete Zufallsvariablen erwartungstreu. Stetige Relaxierungen wie Gumbel-Softmax :cite:`Jang.Gu.Poole.2017,Maddison.Mnih.Teh.2017` und Straight-Through-Schätzer :cite:`Bengio.Leonard.Courville.2013` liefern selbst Schätzwerte mit geringer Varianz. Dafür führen sie eine Verzerrung gegenüber dem ursprünglichen diskreten Optimierungsziel ein. Ein dritter Ansatz verwendet *maßwertige Ableitungen* (engl. measure-valued derivatives). Dabei stellen wir die Ableitung einer Wahrscheinlichkeitsdichte als skalierte Differenz zweier Dichten dar. Den Gradienten schätzen wir, indem wir aus beiden Verteilungen Stichproben ziehen. :citet:`Mohamed.Rosca.Figurnov.ea.2020` fassen alle drei Familien in einem gemeinsamen Rahmen zusammen und geben für jede die genauen Anwendungsvoraussetzungen an.

<!--en:5fd245f8-->
## Bücher, Kurse und Tutorials

<!--en:f3e47eb0-->
**Bücher**

<!--en:68dc8bfa-->
- [Mathematics for Machine Learning](https://mml-book.github.io/) von Deisenroth, Faisal und Ong behandelt in Kapitel 5 („Vector Calculus“) Gradienten, Jacobi-Matrizen und die Kettenregel für maschinelles Lernen (:numref:`sec_mdl-single_variable_calculus` und :numref:`sec_mdl-multivariable_calculus`).

<!--en:22f58cb0-->
- [Calculus](https://ocw.mit.edu/courses/res-18-001-calculus-fall-2023/) von Gilbert Strang (MIT OpenCourseWare) ist ein frei zugängliches Lehrbuch. Die Kapitel über Integrale, den Hauptsatz der Differential- und Integralrechnung und Mehrfachintegrale ergänzen :numref:`sec_mdl-integral_calculus`.

<!--en:04840500-->
- [The Matrix Cookbook](https://archive.org/details/imm3274) von Petersen und Pedersen enthält eine Formelsammlung zu Matrixableitungen. Damit lassen sich Ergebnisse überprüfen, die wir mithilfe von Differentialen hergeleitet haben (:numref:`sec_mdl-matrix-calculus`).

<!--en:b4291163-->
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717761) von Griewank und Walther sowie [The Art of Differentiating Computer Programs](https://epubs.siam.org/doi/book/10.1137/1.9781611972078) von Naumann sind die Standardwerke zum Vorwärts- und Rückwärtsmodus, zu VJPs und JVPs, zum Checkpointing und zu Compilern zur Ableitungsberechnung (:numref:`sec_mdl-matrix-calculus-autodiff`).

<!--en:6eeb657c-->
**Kurse und Tutorials**

<!--en:0fa7dd2a-->
- [Essence of Calculus](https://www.3blue1brown.com/topics/calculus) von 3Blue1Brown sowie [Single Variable Calculus (18.01SC)](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) und [Multivariable Calculus (18.02SC)](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) auf MIT OpenCourseWare.

<!--en:fc48b701-->
- [Matrix Calculus for Machine Learning and Beyond (18.S096)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/) von Edelman und Johnson (MIT OCW), [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) von Parr und Howard sowie [matrixcalculus.org](https://www.matrixcalculus.org/) von Laue, Mitterreiter und Giesen.

<!--en:8c4697b5-->
- [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) von Christopher Olah, [Automatic Differentiation in Machine Learning: a Survey](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf) von Baydin et al. sowie [The Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) und [Autodidax](https://docs.jax.dev/en/latest/autodidax.html) in der JAX-Dokumentation (:numref:`sec_mdl-matrix-calculus-autodiff`). [Monte Carlo Gradient Estimation in Machine Learning](https://jmlr.org/papers/v21/19-346.html) von Mohamed et al. behandelt den Score-Funktionsschätzer (REINFORCE), die pfadweise Reparametrisierung und die Varianzreduktion mit Kontrollvariaten (:numref:`sec_mdl-expectation-gradients`).

<!--en:9b6632a7-->
[Diskussion](https://d2l.discourse.group/t/36858)
