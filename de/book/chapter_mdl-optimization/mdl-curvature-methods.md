<!--en:15e3bac5-->
# Krümmungsbasierte Verfahren: Newton-Verfahren, Quasi-Newton-Verfahren und Trust-Regionen

<!--en:01b52458-->
Der Gradientenabstieg (engl. gradient descent) misst die Krümmung (engl. curvature) der Funktion, die er minimiert, an keiner Stelle. Er bekommt nur ihre Folgen zu spüren. Die steilste Richtung begrenzt die Schrittweite, sodass das Verfahren in der flachsten Richtung kaum vorankommt. Die Zahl der Iterationen wächst mit der Konditionszahl $\kappa$, mit Momentum nur noch mit $\sqrt\kappa$ (:numref:`subsec_mdl-momentum-acceleration`). In unserem durchgängigen Beispiel $\tfrac12(x_1^2 + 1000\,x_2^2)$ brauchte der Gradientenabstieg $6{,}908$ Schritte, mit Momentum waren es $315$. Dabei sieht man die Lösung auf einen Blick: Teilen wir jede Koordinate des Gradienten durch die zugehörige Krümmung ($1$ oder $1000$), erreichen wir das Minimum in einem Schritt. Was kann ein Optimierer erreichen, wenn er die Krümmung gezielt nutzt?

<!--en:151ea6bb-->
Die klassische Antwort liefert das Newton-Verfahren (engl. Newton’s method). Es ist anderthalb Jahrhunderte älter als der Gradientenabstieg. An jeder Iterierten nähert es die Zielfunktion quadratisch an und springt direkt zum Minimierer dieser Näherung. Wir untersuchen, wie schnell das Verfahren konvergiert, was ein Schritt kostet und wo das Verfahren versagt. Wenn zweite Ableitungen fehlen, wie viel lässt sich dann allein aus Gradienten über die Krümmung erfahren? In der Nähe eines Minimums ist das Newton-Verfahren bemerkenswert schnell. Bei einem neuronalen Netz wäre es allerdings viel zu teuer, die Hesse-Matrix (engl. Hessian) zu bilden. Bei nichtkonvexen Funktionen ziehen Sattelpunkte das Verfahren ebenso an wie Minima. Anschließend betrachten wir zwei Ansätze zur Abhilfe. Quasi-Newton-Verfahren rekonstruieren die Krümmung aus Differenzen aufeinanderfolgender Gradienten. Trust-Region-Verfahren beschränken jeden Schritt auf eine Umgebung, in der das quadratische Modell die Funktion tatsächlich gut beschreibt.

<!--en:f655f1b2-->
Beim Training tiefer Netze kommen diese klassischen Verfahren viel seltener zum Einsatz als Optimierer erster Ordnung. Die Gründe dafür lassen sich klar benennen. Die vollständige Hesse-Matrix ist zu groß, um sie zu bilden. Minibatch-Rauschen verfälscht Krümmungsschätzungen aus Gradientendifferenzen. Dennoch begegnen uns die Grundideen im Deep Learning immer wieder. Die koordinatenweise Skalierung bei Adam übernimmt die Rolle einer diagonalen Näherung an die inverse Hesse-Matrix des Newton-Verfahrens (:numref:`sec_mdl-adaptive-stochastic-methods`). Die Dämpfung von Krümmungsmatrizen stabilisiert den Schritt ähnlich wie eine Trust-Region. Das zentrale Ergebnis ist die quadratische Konvergenz des Newton-Verfahrens in :eqref:`eq_mdl-opt-newton-error`. Den Unterabschnitt über Quasi-Newton-Verfahren können Sie beim ersten Lesen überspringen. Aus :numref:`sec_mdl-gradient-based-optimization` übernehmen wir die Armijo-Bedingung für die Backtracking-Liniensuche und die Testfunktion $\tfrac12 x^2 + \cos y$, an deren Sattelpunkt der Gradientenabstieg zum Stillstand kam.

<!--en:5e0e312d-->
## Newton-Verfahren

<!--en:d80462f5-->
Der Gradientenabstieg nähert $f$ am aktuellen Punkt linear an. Eine lineare Funktion hat kein Minimum. Deshalb kann der Algorithmus die Schrittlänge nicht selbst wählen und braucht eine vorgegebene Lernrate. Eine nach oben gekrümmte quadratische Näherung hat hingegen ein eindeutiges Minimum. Damit bestimmt die lokale Geometrie die Schrittlänge.

<!--en:7cce7f9c-->
### Der Newton-Schritt

<!--en:90912a05-->
Beim Newton-Verfahren bestimmt die lokale Krümmung sowohl die Richtung als auch die Länge des Schritts (:numref:`subsec_gd-newton`). Wir schreiben $\mathbf{g}_k = \nabla f(\mathbf{x}_k)$ und nähern $f$ in der Umgebung von $\mathbf{x}_k$ durch die Taylor-Entwicklung zweiter Ordnung in der Verschiebung $\mathbf{p}$ an:

<!--en:d7c7cc92-->
Ist die Hesse-Matrix $\nabla^2 f(\mathbf{x}_k)$ positiv definit, so hat das Modell $m_k$ die Form einer streng konvexen Schale mit eindeutigem Minimierer. Diesen finden wir, indem wir $\nabla m_k(\mathbf{p}) = \mathbf{g}_k + \nabla^2 f(\mathbf{x}_k)\,\mathbf{p} = \mathbf{0}$ setzen. Lösen wir nach $\mathbf{p}$ auf und führen die entsprechende Verschiebung aus, erhalten wir das **Newton-Verfahren**:

<!--en:fe5c3358-->
![Ein Newton-Schritt für $f(x) = \tfrac14 x^4 + \tfrac12 x^2$ vom Startpunkt $x_k = 1$ aus. Das Modell $m_k$ (gestrichelt) hat denselben Wert, dieselbe Steigung und dieselbe Krümmung wie $f$ an der Stelle $x_k$. Der Schritt $p$ führt zum Minimierer des Modells, $x_{k+1} = \tfrac12$. Der Minimierer von $f$ ist $x^\star = 0$.](../img/mdl-opt-newton-model.svg)

<!--en:6ac405c8-->
:numref:`fig_mdl-opt-newton-model` zeigt einen solchen Schritt in einer Dimension. In :eqref:`eq_mdl-opt-newton-step` kommt keine Schrittweite vor. Die inverse Hesse-Matrix skaliert jede Richtung entsprechend ihrer Krümmung. Allgemeiner gehört die Newton-Richtung zur Familie der vorkonditionierten Schritte $\mathbf{d} = -\mathbf{P}^{-1}\nabla f$ mit einer positiv definiten Matrix $\mathbf{P}$ (hier $\mathbf{P} = \nabla^2 f$). Jede solche Richtung ist eine Abstiegsrichtung, denn $\nabla f^\top \mathbf{d} = -\nabla f^\top \mathbf{P}^{-1}\nabla f < 0$. Messen wir Schrittlängen in der von der Hesse-Matrix induzierten Norm statt in der euklidischen Norm, so ist die Newton-Richtung gerade die Richtung des steilsten Abstiegs (Aufgabe :exref:`ex_mdl-gradient-based-optimization-show-that-is-de`).

<!--en:d5358fc2-->
Für eine positiv definite quadratische Funktion $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ ist das Taylor-Modell exakt. Deshalb führt ein einziger Newton-Schritt zum Minimierer: $\mathbf{x}_1 = \mathbf{x}_0 - \mathbf{A}^{-1}\mathbf{A}\mathbf{x}_0 = \mathbf{0}$, unabhängig von der Konditionszahl $\kappa$. Dafür gibt es einen tieferen Grund als das Kürzen der Matrizen. Das Newton-Verfahren ist **affininvariant**. Seine Trajektorie hängt also nicht von der Wahl der Koordinaten ab. Bei jeder invertierbaren linearen Variablentransformation $\mathbf{x} = \mathbf{T}\mathbf{y}$ liefert das Newton-Verfahren für $f(\mathbf{T}\mathbf{y})$ genau die transformierten Newton-Iterierten für $f$. Beim Gradientenabstieg ändert eine nichtorthogonale Transformation $\mathbf{T}$ im Allgemeinen den Verlauf. Ein skalares Vielfaches einer orthogonalen Matrix skaliert lediglich die effektive Schrittweite (Aufgabe :exref:`ex_mdl-gradient-based-optimization-newton-show-that-any`). Für jede positiv definite quadratische Funktion gibt es eine lineare Transformation, unter der ihre Niveauflächen zu Kugeln werden. Deshalb spielt die Konditionszahl für das Newton-Verfahren keine Rolle. Ein langes Tal ist für dieses Verfahren eine runde Schale auf gedehntem Papier.

<!--en:ba1c9b52-->
### Quadratische Konvergenz

<!--en:33bcb181-->
Für eine allgemeine, nichtquadratische Zielfunktion gilt das Taylor-Modell nur lokal. Ein Schritt reicht dann nicht mehr aus. Die bemerkenswerte Geschwindigkeit bleibt jedoch erhalten. Sobald die Iterierte in eine hinreichend kleine Umgebung eines Minimierers gelangt, konvergiert das Newton-Verfahren wesentlich schneller als geometrisch. Für den Beweis brauchen wir kaum mehr als den Hauptsatz der Differential- und Integralrechnung. Sei $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}^\star$ der Fehler in Iteration $k$.

<!--en:6e300e8f-->
**Proposition (quadratische Konvergenz).** *Es gelte $\nabla f(\mathbf{x}^\star) = \mathbf{0}$. In einer Kugel um $\mathbf{x}^\star$ erfülle die Hesse-Matrix die Bedingungen $\nabla^2 f(\mathbf{x}) \succeq \mu\mathbf{I}$ und $\|\nabla^2 f(\mathbf{x}) - \nabla^2 f(\mathbf{y})\| \le L_2\|\mathbf{x} - \mathbf{y}\|$. Liegt $\mathbf{x}_k$ in der Kugel, so gilt*

<!--en:191178d2-->
**Beweis.** Wir schreiben kurz $\nabla^2 f_k = \nabla^2 f(\mathbf{x}_k)$. Wegen $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ können wir $\mathbf{x}^\star$ von der Newton-Aktualisierung abziehen und erhalten $\mathbf{e}_{k+1} = \mathbf{e}_k - (\nabla^2 f_k)^{-1}\mathbf{g}_k = (\nabla^2 f_k)^{-1}\big[\nabla^2 f_k\,\mathbf{e}_k - (\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star))\big]$. Nach dem Hauptsatz der Differential- und Integralrechnung ist die Gradientendifferenz entlang der Strecke von $\mathbf{x}^\star$ nach $\mathbf{x}_k$ gleich $\int_0^1 \nabla^2 f(\mathbf{x}^\star + t\mathbf{e}_k)\,\mathbf{e}_k\,dt$. Der Ausdruck in eckigen Klammern ist also $\int_0^1 \big[\nabla^2 f_k - \nabla^2 f(\mathbf{x}^\star + t\mathbf{e}_k)\big]\mathbf{e}_k\,dt$. Die Punkte $\mathbf{x}_k$ und $\mathbf{x}^\star + t\mathbf{e}_k$ unterscheiden sich um $(1 - t)\mathbf{e}_k$. Nach der Lipschitz-Schranke für die Hesse-Matrix beträgt die Norm des Integrals daher höchstens $\int_0^1 L_2(1 - t)\|\mathbf{e}_k\|^2\,dt = \tfrac{L_2}{2}\|\mathbf{e}_k\|^2$. Zusammen mit $\|(\nabla^2 f_k)^{-1}\| \le 1/\mu$ folgt die Behauptung. $\blacksquare$

<!--en:058a3289-->
Was folgt aus :eqref:`eq_mdl-opt-newton-error`? Multiplizieren wir beide Seiten mit $L_2/(2\mu)$, so erhalten wir $\tfrac{L_2}{2\mu}\|\mathbf{e}_{k+1}\| \le \big(\tfrac{L_2}{2\mu}\|\mathbf{e}_k\|\big)^2$. In jedem Schritt ist der skalierte Fehler also höchstens so groß wie das Quadrat seines vorherigen Werts. Liegt sein Anfangswert unter $1$, nimmt er monoton ab und alle weiteren Iterierten bleiben in der Kugel. Sobald er unter $0.1$ sinkt, fällt er auf höchstens $10^{-2}$, $10^{-4}$ und $10^{-8}$. Die Zahl der korrekten Stellen verdoppelt sich also ungefähr mit jeder Iteration. In unserem durchgängigen Beispiel braucht der Gradientenabstieg dagegen etwa $1{,}150$ Schritte für eine einzige weitere korrekte Stelle. Die Einschränkung steht schon in den Voraussetzungen: Die quadratische Rate gilt nur lokal. Sie garantiert erst dann etwas, wenn die Iterierte bereits nahe bei $\mathbf{x}^\star$ liegt.

<!--en:00daebf2-->
Weit entfernt von $\mathbf{x}^\star$ kann der reine Newton-Schritt selbst bei positiv definiter Hesse-Matrix über das Ziel hinausschießen. Die übliche Absicherung ist die **Dämpfung** (engl. damping). Wir gehen zum Punkt $\mathbf{x}_k + \alpha_k\mathbf{p}_k$ entlang der Newton-Richtung $\mathbf{p}_k = -(\nabla^2 f(\mathbf{x}_k))^{-1}\mathbf{g}_k$. Durch schrittweises Verkleinern wählen wir $\alpha_k \le 1$ so, dass die Armijo-Bedingung $f(\mathbf{x}_k + \alpha\mathbf{p}_k) \le f(\mathbf{x}_k) + c\,\alpha\,\mathbf{g}_k^\top\mathbf{p}_k$ gilt (die Armijo-Regel aus :numref:`sec_mdl-gradient-based-optimization` für eine allgemeine Suchrichtung). In der Nähe eines Minimierers mit positiv definiter Hesse-Matrix erfüllt der volle Schritt $\alpha = 1$ diese Bedingung automatisch, sofern $c < \tfrac12$ gilt. Sobald die quadratische Konvergenz einsetzt, verursacht die Dämpfung also keinen Mehraufwand.

<!--en:7040e56b-->
Um diese Vorhersagen im Code zu prüfen, verwenden wir eine $\ell_2$-regularisierte logistische Regression mit $n = 200$ synthetischen Beispielen in $d = 5$ Dimensionen,

<!--en:bfbfcef0-->
wobei wir die binären Labels $y_i = \pm 1$ aus einem logistischen Modell ziehen (mit $\nu$ bezeichnen wir den Koeffizienten des Gewichtszerfalls, damit $\lambda$ für Eigenwerte frei bleibt). Die folgende Codezelle berechnet außerdem ein Referenzoptimum $\mathbf{w}^\star$ mit $4{,}000$ Gradientenschritten der Schrittweite $\eta = 0.5$. Diese liegt deutlich unter $2/L$, denn $L = \lambda_{\max}(\mathbf{X}^\top\mathbf{X})/(4n) + \nu$ beträgt hier ungefähr $0.4$ (Aufgabe :exref:`ex_mdl-gradient-based-optimization-smoothness-constants`).

<!--en:8280cebb-->
Nun können wir beide Vorhersagen direkt prüfen. Bei unserer quadratischen Funktion mit der Hesse-Matrix $\mathrm{diag}(1, 1000)$ sollte ein einziger Newton-Schritt die $\log 10^6/\log(1001/999) \approx 6{,}908$ Schritte ersetzen, die der Gradientenabstieg bei optimaler fester Schrittweite braucht (:numref:`subsec_mdl-momentum-acceleration`). Beim nichtquadratischen logistischen Verlust sollte sich der Exponent der Gradientennorm in jeder Iteration ungefähr verdoppeln, sobald die lokale Phase beginnt.

<!--en:37274c65-->
Beide Vorhersagen treffen zu. Bei der quadratischen Funktion fällt der Fehler mit einem Newton-Schritt auf null. Der Gradientenabstieg mit optimaler fester Schrittweite verringert ihn pro Schritt nur um den Faktor $0.998$ und braucht $6{,}908$ Iterationen. Beim logistischen Verlust liegt die Gradientennorm zunächst bei einigen Zehnteln und fällt bis zur fünften Iteration unter $10^{-8}$. Sobald die Norm klein ist, verdoppelt sich ihr Exponent ungefähr pro Iteration. Schon nach sechs Iterationen stimmt die Newton-Iterierte bis auf etwa $10^{-15}$ mit dem Referenzoptimum überein. Das entspricht der Genauigkeit der Referenz selbst. Das Lösen von sechs linearen Gleichungssystemen hat so viel erreicht wie $4{,}000$ Gradientenschritte.

<!--en:1313a13d-->
### Kosten eines Schritts

<!--en:28ecbc48-->
Ein Vergleich allein nach der Zahl der Iterationen lässt das Newton-Verfahren zu günstig erscheinen. Die Hesse-Matrix hat $d^2$ Einträge. Um das lineare Gleichungssystem in :eqref:`eq_mdl-opt-newton-step` zu lösen, brauchen wir im Allgemeinen $O(d^3)$ Operationen pro Schritt. Für $d = 5$ sind die Kosten vernachlässigbar. Doch schon bei einem neuronalen Netz mit vergleichsweise wenigen Parametern, etwa $10^8$, hat die Hesse-Matrix $10^{16}$ Einträge. Damit scheidet das exakte Newton-Verfahren schon vor dem ersten Schritt aus (:numref:`subsec_gd-newton`). Praktikable Alternativen nähern die Hesse-Matrix aus Gradienten an, wie die folgenden Quasi-Newton-Verfahren, oder nutzen kostengünstige diagonale und strukturierte Vorkonditionierer (:numref:`sec_mdl-adaptive-stochastic-methods`).

<!--en:fc55be48-->
### Sattelpunkte

<!--en:af739f3f-->
Die Rechenkosten sind weder der einzige noch der aufschlussreichste Nachteil des Newton-Verfahrens. Der Newton-Schritt löst $\nabla m_k = \mathbf{0}$ und springt deshalb zum stationären Punkt des quadratischen Modells, ob Minimum, Maximum oder Sattelpunkt. Betrachten wir wieder die Funktion $f(x, y) = \tfrac12 x^2 + \cos y$ aus :numref:`sec_mdl-gradient-based-optimization`. Sie hat einen Sattelpunkt im Ursprung und ein Minimum mit dem Wert $-1$ bei $(0, \pi)$. Wir starten bei $(0.5, 0.3)$. Dort ist der Gradient $(0.5, -\sin 0.3) = (0.5, -0.296)$ und die Hesse-Matrix $\mathrm{diag}(1, -\cos 0.3) = \mathrm{diag}(1, -0.955)$ ist indefinit. Die Fläche ist in $x$-Richtung nach oben und in $y$-Richtung nach unten gekrümmt. In welche Richtung bewegt ein Newton-Schritt von hier aus die Koordinate $y$, und in welche Richtung bewegt sie ein gewöhnlicher Gradientenschritt mit Schrittweite $\eta = 1$?

<!--en:22e08fab-->
Der erste Newton-Schritt setzt $x$ auf $0$ und $y$ auf $0.3 - \tan 0.3 = -0.0093$. Der zweite setzt $y$ auf $2.7 \times 10^{-7}$. Hier zeigt sich die schnelle lokale Konvergenz des Newton-Verfahrens (wegen $y - \tan y \approx -y^3/3$ ist sie hier sogar kubisch). Sie führt die Iterierte direkt zum Sattelpunkt $(0, 0)$ mit $f = 1$. Der Gradientenabstieg bewegt $y$ in die entgegengesetzte Richtung, über $0.60$, $1.16$, $2.07$, $2.95$, $3.140$ und $3.1416$ bis zum Minimum $(0, \pi)$ mit $f = -1$. Das genauere Modell hat zum schlechteren Punkt geführt. Weil die Krümmung in $y$-Richtung negativ ist, hat das quadratische Modell einen Sattelpunkt als stationären Punkt. Das Newton-Verfahren kann einen Sattelpunkt nicht von einem Minimum unterscheiden. Eine Trust-Region (:numref:`subsec_mdl-trust-region`) behebt dieses Problem.

<!--en:600a83e4-->
## Quasi-Newton-Verfahren

<!--en:ac09dcdb-->
Das Newton-Verfahren braucht explizite zweite Ableitungen. Können wir die Krümmung auch aus den Gradienten erschließen, die wir ohnehin berechnen? Ändert sich der Gradient entlang eines Schritts rasch, muss die Zielfunktion in dieser Richtung eine große Krümmung aufweisen. Ein Optimierer kann diese Änderungen festhalten und daraus nach und nach ein Modell der Hesse-Matrix aufbauen, ohne je eine zweite Ableitung auszuwerten.

<!--en:32d76912-->
### Das Sekantenverfahren

<!--en:e901e2b9-->
In einer Dimension ist die Idee unmittelbar einsichtig. Verbinden wir die letzten beiden Gradientenwerte durch eine Gerade, so schätzt deren Steigung die zweite Ableitung $f''$:

<!--en:cd360d48-->
Das ist das **Sekantenverfahren** (engl. secant method). Es ersetzt im Newton-Schritt $f''(x_k)$ durch einen Differenzenquotienten der Gradienten. Betrachten wir $f(x) = e^x - 2x$ mit dem Minimierer $\log 2 \approx 0.6931$ und den Startwerten $x_0 = 0$ und $x_1 = 1$. Die erste Sekantensteigung beträgt $h_1 = (e - 2) - (1 - 2) = e - 1 \approx 1.718$. Damit erhalten wir $x_2 = 1 - (e - 2)/1.718 \approx 0.582$. Von da an sinken die Fehler auf $1.1 \times 10^{-1}$, $1.6 \times 10^{-2}$, $9.3 \times 10^{-4}$, $7.7 \times 10^{-6}$ und $3.6 \times 10^{-9}$. Jeder Fehler beträgt ungefähr die Hälfte des Produkts der beiden vorherigen Fehler (der Faktor ein Halb ergibt sich aus $f'''/(2f'')$ am Minimierer). Die Zahl der korrekten Stellen wächst pro Schritt um einen Faktor, der sich $1.6$ nähert. Das ist etwas langsamer als beim Newton-Verfahren mit seinem Faktor $2$. Dafür brauchen wir pro Iteration nur eine Gradientenauswertung und keine zweite Ableitung. Für eine eindimensionale quadratische Funktion $\tfrac{\lambda}{2}x^2$ ist die Sekantensteigung genau $\lambda$ und der Schritt von $x_1$ führt direkt zum Minimierer.

<!--en:41130dd9-->
In $d$ Dimensionen liefert die entsprechende Sekantenbedingung eine einzige Vektorgleichung. Das reicht bei Weitem nicht aus, um eine $d \times d$-Matrix zu bestimmen. Ein Schritt verrät uns nur die Krümmung in einer Richtung. Für die übrigen Richtungen brauchen wir eine begründete Regel.

<!--en:84a54ba0-->
### BFGS

<!--en:84c00ad4-->
Ein **Quasi-Newton-Verfahren** verwendet eine Näherung $\mathbf{M}_k$ an die inverse Hesse-Matrix. Nach jedem Schritt aktualisiert es diese Näherung anhand der beobachteten Änderung des Gradienten. Nach dem Schritt von $\mathbf{x}_k$ nach $\mathbf{x}_{k+1}$ halten wir das Sekantenpaar fest:

<!--en:f143abb1-->
Für eine quadratische Funktion $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ erfüllt die Gradientendifferenz die Gleichung $\mathbf{y}_k = \mathbf{A}\mathbf{s}_k$. Die tatsächliche inverse Hesse-Matrix bildet also $\mathbf{y}_k$ wieder auf $\mathbf{s}_k$ ab. Wir fordern, dass die aktualisierte Näherung $\mathbf{M}_{k+1}$ dieselbe **Sekantengleichung** erfüllt:

<!--en:5d315373-->
Unter allen symmetrischen Matrizen, die :eqref:`eq_mdl-opt-secant` erfüllen, wählt **BFGS** (benannt nach Broyden, Fletcher, Goldfarb und Shanno) diejenige mit dem kleinsten Abstand zur bisherigen Schätzung $\mathbf{M}_k$. Den Abstand messen wir in einer geeignet gewichteten Frobeniusnorm:

<!--en:93da3cbe-->
Wie :eqref:`eq_mdl-opt-bfgs` wirkt, lässt sich auch ohne Herleitung ablesen. Der Übersicht halber lassen wir beim Sekantenpaar den Index $k$ weg. Der rechte Faktor $\mathbf{I} - \gamma\mathbf{y}\mathbf{s}^\top$ bildet $\mathbf{y}$ auf null ab, denn es gilt $\gamma\,\mathbf{s}^\top\mathbf{y} = 1$. Jeden zu $\mathbf{s}$ orthogonalen Vektor lässt er unverändert. Der erste Term löscht damit die bisherige Vorhersage von $\mathbf{M}_k$ in Richtung $\mathbf{y}$. Der zweite Term trägt die neue Beobachtung ein: $\mathbf{M}_{k+1}\mathbf{y} = \gamma\,\mathbf{s}\,(\mathbf{s}^\top\mathbf{y}) = \mathbf{s}$. Damit ist die Sekantengleichung erfüllt.

<!--en:b0eff1b1-->
Ebenso wichtig ist, dass die Aktualisierung die positive Definitheit erhält. Die nächste Suchrichtung $-\mathbf{M}_{k+1}\nabla f(\mathbf{x}_{k+1})$ bleibt also eine Abstiegsrichtung. Für einen beliebigen Vektor $\mathbf{z}$ ungleich null setzen wir $\mathbf{w} = (\mathbf{I} - \gamma\mathbf{y}\mathbf{s}^\top)\mathbf{z}$. Dann gilt $\mathbf{z}^\top\mathbf{M}_{k+1}\mathbf{z} = \mathbf{w}^\top\mathbf{M}_k\mathbf{w} + \gamma(\mathbf{s}^\top\mathbf{z})^2$. Für $\mathbf{M}_k \succ 0$ und $\gamma > 0$ sind beide Terme nichtnegativ und können nicht gleichzeitig verschwinden. Denn aus $\mathbf{w} = \mathbf{0}$ folgt $\mathbf{z} = \gamma(\mathbf{s}^\top\mathbf{z})\,\mathbf{y}$, was $\mathbf{s}^\top\mathbf{z} \neq 0$ voraussetzt. Die Forderung $\gamma > 0$ ist die **Krümmungsbedingung** $\mathbf{y}_k^\top\mathbf{s}_k > 0$. Sie besagt, dass die Steigung in Schrittrichtung entlang des Schritts zugenommen hat. Für jede streng konvexe Zielfunktion gilt sie automatisch. Bei allgemeinen Zielfunktionen erzwingt die Liniensuche sie durch eine zweite Prüfung, die Wolfe-Bedingung :cite:`Nocedal.Wright.2006`.

<!--en:96d48c84-->
Wie schnell lässt sich die tatsächliche Krümmung aus Sekantenpaaren rekonstruieren? Starten wir BFGS bei $(1, 1)$ für die zweidimensionale quadratische Funktion mit Hesse-Matrix $\mathrm{diag}(1, 10)$. Wir beginnen mit $\mathbf{M}_0 = \mathbf{I}$ und verwenden die exakte Liniensuche $\alpha = -\mathbf{g}^\top\mathbf{p}/(\mathbf{p}^\top\mathbf{A}\mathbf{p})$. Nach jedem Schritt betrachten wir $\mathbf{x}$ und $\mathbf{M}$.

<!--en:cd58327d-->
Schon nach einem Schritt kommt $\mathbf{M}_1 \approx \mathrm{diag}(1.009, 0.100)$ der Matrix $\mathbf{A}^{-1}$ in der gerade untersuchten steilen Richtung sehr nahe. Nach dem zweiten Schritt gilt bis auf Gleitkomma-Rundungsfehler $\mathbf{M}_2 = \mathbf{A}^{-1}$ und $\mathbf{x}_2 = \mathbf{0}$. Dieses Verhalten lässt sich verallgemeinern. Bei jeder $d$-dimensionalen positiv definiten quadratischen Funktion mit exakter Liniensuche bleibt jede Sekantengleichung auch nach allen weiteren Aktualisierungen erfüllt. Die Schritte sind paarweise $\mathbf{A}$-konjugiert ($\mathbf{s}_i^\top\mathbf{A}\mathbf{s}_j = 0$ für $i \neq j$). Nach $d$ Schritten gilt $\mathbf{M}_d\mathbf{A} = \mathbf{I}$ und BFGS erreicht den exakten Minimierer (Aufgabe :exref:`ex_mdl-gradient-based-optimization-bfgs-quadratic`).

<!--en:ffc8f0b3-->
Der Abschluss nach endlich vielen Schritten ist allerdings nur für quadratische Funktionen mit exakter Liniensuche garantiert. Bei nichtquadratischen Zielfunktionen ist die Liniensuche in der Praxis ungenau, die tatsächliche Hesse-Matrix ändert sich von Schritt zu Schritt und stochastische Gradienten verfälschen $\mathbf{y}_k$. Die Differenz zweier Minibatch-Gradienten spiegelt oft vor allem die Zufallsschwankungen der Stichprobe wider. Die Krümmung geht darin unter. Außerdem speichert das vollständige BFGS-Verfahren eine dicht besetzte $d \times d$-Matrix. **L-BFGS** spart Speicher, indem es nur die letzten $m$ Sekantenpaare behält. Die daraus gebildete Näherung an die inverse Hesse-Matrix besteht aus einer skalierten Einheitsmatrix und einer Korrektur mit Rang höchstens $2m$. Sie lässt sich mit einem Zeit- und Speicherbedarf von $O(md)$ anwenden :cite:`Liu.Nocedal.1989`. Bei deterministischen Problemen mittlerer Größe funktioniert das sehr gut. Schwierig bleibt der Einsatz, wenn das Rauschen neuer Minibatches die Gradientendifferenzen unzuverlässig macht.

<!--en:7422edf9-->
## Trust-Regionen

<!--en:68dacd73-->
Bei $(0.5, 0.3)$ lief das Newton-Verfahren in die falsche Richtung, weil es sich im gesamten Raum auf sein lokales quadratisches Modell verließ. Das Modell beruht auf Ableitungen an einem einzigen Punkt und beschreibt $f$ nur in dessen Nähe genau. Ein Trust-Region-Verfahren trägt dieser Einschränkung Rechnung. Es nutzt die gesamte Krümmungsinformation einschließlich der Richtungen negativer Krümmung und beschränkt den Schritt auf ein Gebiet, in dem das Modell verlässlich ist.

<!--en:241c8585-->
### Das Teilproblem

<!--en:1764bdfd-->
Im Punkt $(0.5, 0.3)$ hat das quadratische Modell von $\tfrac12 x^2 + \cos y$ die Krümmung $-0.955$ in $y$-Richtung. In dieser Richtung fällt es unbegrenzt ab. Nach einem Minimum ohne Nebenbedingungen zu suchen, hat also keinen Sinn. Ein **Trust-Region-Verfahren** minimiert das quadratische Modell innerhalb einer Kugel mit Radius $\Delta_k$. Der Radius gibt an, wie weit wir der Näherung vertrauen:

<!--en:304c8522-->
Dabei ist $\mathbf{B}_k$ die exakte Hesse-Matrix oder eine symmetrische Näherung daran. Beschränken wir $\mathbf{p}$ auf eine abgeschlossene Kugel, so existiert auch dann ein Minimierer, wenn $\mathbf{B}_k$ negative Eigenwerte hat. Denn jede stetige Funktion nimmt auf einer kompakten Menge ein Minimum an. Die exakten Lösungen dieses Problems mit Nebenbedingung lassen sich sogar genau charakterisieren.

<!--en:30837a6e-->
**Proposition (Lösung des Teilproblems).** *Ein Schritt $\mathbf{p}$ mit $\|\mathbf{p}\| \le \Delta$ löst :eqref:`eq_mdl-opt-trust-region` genau dann, wenn es ein $\tau \ge 0$ gibt mit*

<!--en:ec7fd0da-->
Dieses Ergebnis ist Satz 4.1 bei :citet:`Nocedal.Wright.2006`. In Aufgabe :exref:`ex_mdl-gradient-based-optimization-trust-region-saddle` beweisen Sie, dass die Bedingungen hinreichen. Geometrisch gesehen addiert die Trust-Region denselben Betrag $\tau$ zu jedem Eigenwert von $\mathbf{B}$. Diese Verschiebung reicht gerade aus, um das quadratische Modell konvex zu machen. Wir unterscheiden zwei Fälle. Ist $\mathbf{B}$ positiv definit und liegt der unbeschränkte Newton-Schritt bereits in der Kugel, so gilt $\tau = 0$ und wir wählen den Newton-Schritt. Andernfalls liegt die Lösung auf dem Rand $\|\mathbf{p}\| = \Delta$. Die Verschiebung $\tau$ löst die skalare Gleichung $\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \Delta$ im Bereich $\tau > -\lambda_{\min}(\mathbf{B})$, in dem die linke Seite mit wachsendem $\tau$ streng monoton fällt.

<!--en:004edda7-->
Nur im entarteten *schwierigen Fall* (engl. hard case) kann diese skalare Gleichung ohne Lösung bleiben. Dabei steht $\mathbf{g}$ senkrecht auf den Eigenvektoren zum kleinsten Eigenwert von $\mathbf{B}$ (siehe Teil d von Aufgabe :exref:`ex_mdl-gradient-based-optimization-trust-region-saddle`). Wie wir in :numref:`sec_mdl-constrained-optimization-duality` sehen, ist die Verschiebung $\tau$ der Lagrange-Multiplikator, der die Nebenbedingung $\tfrac12\|\mathbf{p}\|^2 \le \tfrac12\Delta^2$ durchsetzt.

<!--en:4b290ee5-->
Zwischen den Iterationen passen wir den Radius $\Delta_k$ daran an, wie genau das Modell den tatsächlichen Rückgang von $f$ vorhergesagt hat. Nach dem Lösen von :eqref:`eq_mdl-opt-trust-region` bilden wir das Verhältnis der tatsächlichen zur vorhergesagten Abnahme:

<!--en:eada3770-->
Ein Verhältnis nahe $1$ zeigt, dass das quadratische Modell die Zielfunktion gut beschrieben hat. In der folgenden Implementierung akzeptieren wir den Schritt, wenn $r_k > 0.1$ gilt. Unabhängig davon verkleinern wir den Radius gemäß $\Delta \leftarrow \Delta/4$, wenn $r_k < \tfrac14$ gilt. Wir verdoppeln ihn, wenn $r_k > \tfrac34$ gilt und der Schritt den Rand $\|\mathbf{p}_k\| = \Delta_k$ erreicht hat.

<!--en:0cb55930-->
### Der Trust-Region-Schritt an einem Sattelpunkt

<!--en:e8888092-->
Berechnen wir einen Trust-Region-Schritt im Punkt $(0.5, 0.3)$ mit $\Delta = 0.5$ von Hand. Hier gelten $\mathbf{g} = (0.5, -0.296)$ und $\mathbf{B} = \mathrm{diag}(1, -0.955)$. Damit $\mathbf{B} + \tau\mathbf{I}$ positiv semidefinit ist, muss $\tau \ge 0.955$ gelten. Also scheidet $\tau = 0$ aus und der Schritt muss auf dem Rand der Kugel liegen. Die Optimalitätsbedingungen liefern $(1 + \tau)p_1 = -0.5$ und $(\tau - 0.955)p_2 = 0.296$ unter der Nebenbedingung $p_1^2 + p_2^2 = 0.25$. Die Lösung lautet $\tau \approx 1.60$ und $\mathbf{p} \approx (-0.19, 0.46)$. Beim Schritt nach $(0.31, 0.76)$ fällt $f$ von $1.080$ auf $0.771$. Das Modell sagt eine Abnahme um $0.316$ voraus, tatsächlich beträgt sie $0.309$. Das Verhältnis ist also $r = 0.98$. Deshalb akzeptieren wir den Schritt und verdoppeln den Radius. Entscheidend ist, dass der Trust-Region-Schritt vom Sattelpunkt *wegführt*. Er folgt der Richtung negativer Krümmung, in der das Modell abfällt. Der unbeschränkte Newton-Schritt $(-0.5, -0.31)$ führte direkt auf den Sattelpunkt zu.

<!--en:c3ed8ad5-->
Um zu sehen, wohin die Lösung des Teilproblems über mehrere Iterationen führt, implementieren wir zunächst `tr_step(g, B, Delta)`. Die Funktion gibt den unbeschränkten Newton-Schritt zurück, wenn $\mathbf{B} \succ \mathbf{0}$ und $\|\mathbf{B}^{-1}\mathbf{g}\| \le \Delta$ gelten. Andernfalls bestimmt sie $\tau$ durch Bisektion, bis $\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \Delta$ gilt:

<!--en:d307f7bd-->
Wir starten bei $\mathbf{x} = (0.5, 0.3)$ mit $\Delta = 0.5$ und führen sechs Iterationen des Trust-Region-Verfahrens für $f_s$ aus. Bei jedem Schritt geben wir die Iterierte, den Zielfunktionswert, die Schrittnorm $\|\mathbf{p}\|$, die Verschiebung $\tau$, das Verhältnis $r$ der tatsächlichen zur vorhergesagten Abnahme und den Radius $\Delta$ aus:

<!--en:11ee997e-->
Die erste Zeile bestätigt unsere Rechnung von Hand. Der Radius verdoppelt sich von $0.5$ auf $1$ und dann auf $2$. Den dritten Schritt (Verhältnis $0.41$) akzeptiert das Verfahren bei unverändertem Radius. Mit der dritten Iterierten $(0.18, 1.75)$ hat es das Einzugsgebiet erreicht, in dem die Hesse-Matrix positiv definit ist. Ab der nächsten Iterierten passt der unbeschränkte Newton-Schritt mit $\tau = 0$ in die Trust-Region. Damit geht das Verfahren nahtlos in das Newton-Verfahren über und konvergiert quadratisch zum Minimum $(0, \pi)$ mit $f = -1$. Sechs Iterationen genügen. :numref:`fig_mdl-opt-trust-region-saddle` zeigt beide Trajektorien im Vergleich.

<!--en:e65c793e-->
![Newton-Verfahren und Trust-Region-Verfahren vom Startpunkt $(0.5,0.3)$ für $\tfrac12 x^2 + \cos y$. Dort ist die Hesse-Matrix $\mathrm{diag}(1,-0.96)$. (a) Das lokale quadratische Modell bildet eine Sattelfläche. Der Newton-Schritt führt zu ihrem stationären Punkt. Bei der Minimierung des Modells auf der Kugel mit Radius $\Delta = 0.5$, führt der Schritt in der Richtung negativer Krümmung vom Sattelpunkt weg. (b) Das Newton-Verfahren konvergiert quadratisch zum Sattelpunkt $(0,0)$ mit $f = 1$. Das Trust-Region-Verfahren (mit eingezeichnetem Anfangsradius) erreicht das Minimum $(0,\pi)$ mit $f = -1$ in sechs Schritten. Dabei wächst der Radius gemäß der Folge $0.5, 1, 2$.](../img/mdl-opt-trust-region-saddle.svg)

<!--en:03fa4ae5-->
Um das Trust-Region-Teilproblem exakt zu lösen, müssen wir $\mathbf{B}_k + \tau\mathbf{I}$ für jeden versuchsweise gewählten Wert von $\tau$ faktorisieren. Implementierungen für große Probleme lösen es näherungsweise mit Hesse-Vektor-Produkten. Jedes davon kostet ungefähr so viel wie eine Gradientenauswertung (:numref:`sec_mdl-matrix-calculus-autodiff`). Auch mit dieser Beschleunigung ersetzen weder BFGS noch Trust-Region-Verfahren den stochastischen Gradientenabstieg bei großen, verrauschten Zielfunktionen im Deep Learning.

<!--en:9d50f73c-->
Der Sattelpunkt von $\tfrac12 x^2 + \cos y$ hat nun zwei Optimierern Schwierigkeiten bereitet. Beim Start auf der Geraden $y = 0$ blieb der Gradientenabstieg dort stecken. Das Newton-Verfahren lief auch von einem allgemeinen Startpunkt aus direkt auf ihn zu. Ein Trust-Region-Verfahren entkommt ihm, muss dazu aber jedes lokale Modell mit der tatsächlichen Zielfunktion vergleichen. Es gibt jedoch eine große Klasse von Funktionen, bei denen all diese Vorkehrungen entfallen. Sie haben weder Sattelpunkte noch irreführende Einzugsgebiete und jeder stationäre Punkt ist ein globales Minimum. Wie wir solche Funktionen erkennen, untersuchen wir in :numref:`sec_mdl-convexity`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:0300390b-->
Das Newton-Verfahren minimiert an jeder Iterierten ein lokales quadratisches Modell von $f$. Bei jeder positiv definiten quadratischen Funktion erreicht es unabhängig von der Konditionszahl in einem Schritt den Minimierer. In der Nähe eines Minimierers mit positiv definiter, Lipschitz-stetiger Hesse-Matrix fällt der Fehler in jeder Iteration quadratisch. Dieser Geschwindigkeit stehen zwei Hindernisse gegenüber. Jeder Schritt braucht $O(d^2)$ Speicher und $O(d^3)$ Operationen, weit mehr, als für ein neuronales Netz tragbar ist. Außerdem ziehen Sattelpunkte den unbeschränkten Schritt ebenso stark an wie Minima. BFGS kommt ohne explizite zweite Ableitungen aus. Es rekonstruiert die inverse Hesse-Matrix aus bisherigen Schritten und Gradientendifferenzen, und zwar vollständig in $d$ Schritten bei einer $d$-dimensionalen quadratischen Funktion. Trust-Region-Verfahren erlauben es, jedes quadratische Modell zuverlässig einzusetzen. Sie beschränken jeden Schritt auf eine Kugel, vergleichen die vorhergesagte mit der tatsächlichen Abnahme von $f$ und vergrößern oder verkleinern die Kugel entsprechend.

<!--en:2294243a-->
## Übungen

<!--en:0cf47e1b-->
1. **Reparametrisierung und Vorkonditionierung.** Sei $\mathbf{T}$ eine invertierbare Matrix und sei $h(\mathbf{y}) = f(\mathbf{T}\mathbf{y})$, sodass $\mathbf{x} = \mathbf{T}\mathbf{y}$ einen Koordinatenwechsel beschreibt.
    1. Drücken Sie $\nabla h$ und $\nabla^2 h$ durch die Ableitungen von $f$ aus. Zeigen Sie, dass die Newton-Iterierten gemäß :eqref:`eq_mdl-opt-newton-step` für $h$ mit Startpunkt $\mathbf{y}_0 = \mathbf{T}^{-1}\mathbf{x}_0$ die Beziehung $\mathbf{y}_k = \mathbf{T}^{-1}\mathbf{x}_k$ erfüllen. Dabei sind $\mathbf{x}_k$ die Newton-Iterierten für $f$.
    1. Wenden Sie den Gradientenabstieg mit Schrittweite $\eta$ auf $h$ an und bilden Sie jede Iterierte gemäß $\mathbf{x}_k = \mathbf{T}\mathbf{y}_k$ zurück. Zeigen Sie, dass die zurücktransformierten Iterierten der Vorschrift $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \mathbf{P}^{-1} \nabla f(\mathbf{x}_k)$ für eine positiv definite Matrix $\mathbf{P}$ folgen, und bestimmen Sie $\mathbf{P}$. Nach Aufgabe :exref:`ex_mdl-gradient-based-optimization-show-that-is-de`b ist dies die Richtung des steilsten Abstiegs in der Norm $\|\cdot\|_{\mathbf{P}}$. Wie lautet $\|\mathbf{x}\|_{\mathbf{P}}$ in den Koordinaten $\mathbf{y}$?
    1. Betrachten Sie $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top \mathbf{A}\,\mathbf{x}$ mit $\mathbf{A} = \mathrm{diag}(1, 1000)$, die quadratische Funktion mit $\kappa = 1000$ aus dem Unterabschnitt zum Newton-Verfahren. Bestimmen Sie eine Diagonalmatrix $\mathbf{T}$, für die der Gradientenabstieg auf $h$ mit $\eta = 1$ den Minimierer in einem Schritt erreicht. Was braucht man, um eine solche Matrix $\mathbf{T}$ für eine beliebige positiv definite Matrix $\mathbf{A}$ zu konstruieren?
   :exlabel:`ex_mdl-gradient-based-optimization-newton-show-that-any`

<!--en:69a8e86d-->
1. [code] **BFGS für eine quadratische Funktion.** ● Sei $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x} - \mathbf{b}^\top\mathbf{x}$ mit positiv definiter Matrix $\mathbf{A}$. Führen Sie BFGS gemäß :eqref:`eq_mdl-opt-bfgs` mit exakter Liniensuche und Anfangsmatrix $\mathbf{M}_0 = \mathbf{I}$ aus.
    1. Zeigen Sie, dass eine exakte Liniensuche entlang $\mathbf{p}_k$ die Gleichung $\nabla f(\mathbf{x}_{k+1})^\top\mathbf{p}_k = 0$ liefert und dass $\mathbf{y}_k = \mathbf{A}\mathbf{s}_k$ gilt.
    1. Zeigen Sie die Vererbungseigenschaft: Gilt $\mathbf{M}_k\mathbf{y}_j = \mathbf{s}_j$ für alle $j < k$ und sind die Schritte $\mathbf{s}_0, \ldots, \mathbf{s}_{k-1}$, $\mathbf{s}_k$ paarweise $\mathbf{A}$-konjugiert, so gilt $\mathbf{M}_{k+1}\mathbf{y}_j = \mathbf{s}_j$ für alle $j \le k$. (Hinweis: Es gilt $\mathbf{s}_j^\top\mathbf{y}_k = \mathbf{s}_j^\top\mathbf{A}\mathbf{s}_k = 0$ für $j < k$.) Folgern Sie, dass nach $d$ linear unabhängigen Schritten $\mathbf{M}_d\mathbf{A} = \mathbf{I}$ gilt.
    1. Führen Sie die BFGS-Schleife aus dem Text in $d = 6$ Dimensionen für eine zufällig gedrehte quadratische Funktion mit Eigenwerten von $1$ bis $1000$ aus. Geben Sie nach jedem Schritt $\max_{j \le k}\|\mathbf{M}_{k+1}\mathbf{y}_j - \mathbf{s}_j\|$ aus. Bestätigen Sie, dass diese Größe im Bereich der Rundungsfehler bleibt und dass $\mathbf{M}_6 = \mathbf{A}^{-1}$ gilt. Wie weit ist der Gradientenabstieg mit optimaler fester Schrittweite nach ebenfalls sechs Gradientenauswertungen vom Minimierer entfernt?
    1. Die symmetrische Rang-1-Aktualisierung $\mathbf{M}_{k+1} = \mathbf{M}_k + \frac{(\mathbf{s} - \mathbf{M}_k\mathbf{y})(\mathbf{s} - \mathbf{M}_k\mathbf{y})^\top}{(\mathbf{s} - \mathbf{M}_k\mathbf{y})^\top\mathbf{y}}$ erfüllt ebenfalls die Sekantengleichung. Zeigen Sie anhand eines $2 \times 2$-Beispiels, dass die Matrix selbst für $\mathbf{y}^\top\mathbf{s} > 0$ ihre positive Definitheit verlieren kann. Geben Sie an, welche der beiden Prüfungen im Text bei dieser Aktualisierung fehlschlägt.
   :exlabel:`ex_mdl-gradient-based-optimization-bfgs-quadratic`

<!--en:d83e27a0-->
1. [code] **Trust-Region an einem Sattelpunkt.** ● Betrachten Sie das Modell $m(\mathbf{p}) = \mathbf{g}^\top\mathbf{p} + \tfrac12\mathbf{p}^\top\mathbf{B}\mathbf{p}$ mit $\mathbf{B} = \mathrm{diag}(1, -1)$ und $\mathbf{g} = (\tfrac12, -\tfrac{3}{10})$.
    1. Zeigen Sie, dass $m$ auf $\mathbb{R}^2$ nach unten unbeschränkt ist und dass der Newton-Punkt $-\mathbf{B}^{-1}\mathbf{g} = (-\tfrac12, -\tfrac{3}{10})$ ein Sattelpunkt von $m$ und kein Minimum ist.
    1. Beweisen Sie, dass die Bedingungen aus der Proposition zur Lösung des Teilproblems hinreichen: Gilt $(\mathbf{B} + \tau\mathbf{I})\mathbf{p} = -\mathbf{g}$ mit $\tau \ge 0$, $\mathbf{B} + \tau\mathbf{I} \succeq \mathbf{0}$ und $\tau(\Delta - \|\mathbf{p}\|) = 0$, so gilt für jedes $\|\mathbf{q}\| \le \Delta$ die Ungleichung $m(\mathbf{q}) \ge m(\mathbf{p}) + \tfrac{\tau}{2}(\|\mathbf{p}\|^2 - \|\mathbf{q}\|^2) \ge m(\mathbf{p})$. (Hinweis: Entwickeln Sie $m(\mathbf{q}) - m(\mathbf{p})$ um $\mathbf{p}$ und nutzen Sie $\mathbf{g} = -(\mathbf{B} + \tau\mathbf{I})\mathbf{p}$.)
    1. Bestimmen Sie für $\Delta = \tfrac12$ den Wert $\tau \ge 1$, indem Sie die Gleichung $\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \Delta$ durch Bisektion lösen (das Ergebnis ist $\tau \approx 1.65$, $\mathbf{p} \approx (-0.19, 0.46)$). Prüfen Sie, dass der Schritt die Koordinate in Richtung negativer Krümmung vergrößert, während der Newton-Schritt sie verkleinert.
    1. Ersetzen Sie $\mathbf{g}$ durch $(\tfrac12, 0)$. Zeigen Sie, dass $\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \tfrac12/(1 + \tau) \le \tfrac14$ für alle $\tau > 1$ gilt. Damit hat die Randgleichung keine Lösung mit $\tau > 1$. Bestimmen Sie die Minimierer auf der Kugel direkt (für sie gelten $\tau = 1$ und $p_2 = \pm\sqrt{\Delta^2 - p_1^2}$). Erklären Sie in einem Satz, wie sich die Beschränkung auf die Kugel in einer Richtung negativer Krümmung auswirkt, in der die Gradientenkomponente verschwindet.
   :exlabel:`ex_mdl-gradient-based-optimization-trust-region-saddle`

<!--en:72a34797-->
[Diskussion](https://d2l.discourse.group/t/36879)

<!--en:171a968a-->
Krümmungsbasierte Verfahren: Newton-Verfahren, Quasi-Newton-Verfahren und Trust-Regionen<br>**Newton-Schritt · affine Invarianz · quadratische Konvergenz · BFGS · Trust-Region-Teilproblem**

<!--en:3cdc8bb0-->
::: {.slide title="Einfluss des Krümmungsmodells auf die Schrittzahl"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:2d402d5c-->
Bei $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ mit Startpunkt $(1, 1)$ hängt die Zahl der Iterationen davon ab, wie das Verfahren die Krümmung berücksichtigt:

<!--en:46a3c113-->
- **Gradientenabstieg ($6{,}908$ Schritte):** Eine einzige skalare Schranke $L = 1000$ gilt für beide Achsen. Die steile Achse erzwingt $\eta < 2/L$, auf der flachen Achse kommt das Verfahren kaum voran.

<!--en:0db2f273-->
- **Heavy-Ball-Momentum ($315$ Schritte):** Mit $\mu = 1, L = 1000$ klingen die Moden gleich schnell ab. Die benötigte Schrittzahl skaliert mit $\sqrt{\kappa}$.

<!--en:ead6e19d-->
- **Newton-Verfahren ($1$ Schritt):** Die vollständige Hesse-Matrix $\nabla^2 f = \mathrm{diag}(1, 1000)$ skaliert jede Koordinate entsprechend ihrer exakten Krümmung.

<!--en:b06b5fe4-->
::: {.slide title="Taylor-Modell zweiter Ordnung und Newton-Schritt"}

<!--en:e5b15e13-->
[Newton-Verfahren]{.kicker}

<!--en:073b4e08-->
Das **Newton-Verfahren** minimiert die lokale quadratische Taylor-Näherung $m_k(\mathbf{p})$ um $\mathbf{x}_k$:

<!--en:8e8c69f3-->
- **Eindeutiges Minimum der Schale:** Gilt $\nabla^2 f(\mathbf{x}_k) \succ \mathbf{0}$, so folgt aus $\nabla m_k(\mathbf{p}) = \mathbf{0}$ die Iterationsvorschrift $\mathbf{x}_{k+1} = \mathbf{x}_k - (\nabla^2 f(\mathbf{x}_k))^{-1}\nabla f(\mathbf{x}_k)$.

<!--en:166bbdf5-->
- **Natürliche Schrittweite eins:** Die inverse Hesse-Matrix bestimmt sowohl die Richtung als auch die physikalischen Einheiten des Schritts ($\eta = 1$).

<!--en:e55b00e6-->
::: {.slide title="Geometrie des vorkonditionierten steilsten Abstiegs"}

<!--en:e5b15e13-2-->
[Newton-Verfahren]{.kicker}

<!--en:0e7a74c4-->
Für jeden symmetrischen, positiv definiten Vorkonditionierer $\mathbf{P} \succ \mathbf{0}$ führt der Schritt $\mathbf{d} = -\mathbf{P}^{-1}\nabla f(\mathbf{x})$ zum steilsten Abstieg in der Mahalanobis-Norm $\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P}\mathbf{d}}$:

<!--en:40cefbac-->
- **Verbindung zwischen Optimierungsverfahren:** $\mathbf{P} = \mathbf{I}$ ergibt den Gradientenabstieg, eine diagonale Matrix $\mathbf{P}$ liefert AdaGrad oder Adam und $\mathbf{P} = \nabla^2 f(\mathbf{x}_k)$ das Newton-Verfahren.

<!--en:a8ecd584-->
::: {.slide title="Dank affiner Invarianz spielt die Konditionszahl keine Rolle"}

<!--en:e5b15e13-3-->
[Newton-Verfahren]{.kicker}

<!--en:40d8837e-->
Bei jedem invertierbaren Koordinatenwechsel $\mathbf{x} = \mathbf{T}\mathbf{y}$ mit $h(\mathbf{y}) = f(\mathbf{T}\mathbf{y})$ transformieren sich Gradient und Hesse-Matrix gemäß $\nabla h = \mathbf{T}^\top \nabla f$ und $\nabla^2 h = \mathbf{T}^\top \nabla^2 f\,\mathbf{T}$. Daraus folgt:

<!--en:92314ebd-->
- **Vollständige $\kappa$-Invarianz:** Mit $\mathbf{T} = \mathbf{A}^{-1/2}$ wird jedes langgestreckte quadratische Tal $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ zu einer rotationssymmetrischen Schale mit kugelförmigen Niveauflächen ($\kappa = 1$). Die Trajektorie des Newton-Verfahrens ist daher unabhängig von der Konditionszahl.

<!--en:0fe5252c-->
::: {.slide title="Lokale quadratische Konvergenz: Zerlegung des Fehlers"}

<!--en:92320639-->
[Konvergenzbeweis]{.kicker}

<!--en:ff9d50c3-->
Sei der betrachtete Punkt $\mathbf{x}^\star$ ein Minimierer einer nichtquadratischen Funktion mit $\nabla f(\mathbf{x}^\star) = \mathbf{0}$. Wie schnell schrumpft der Fehler $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}^\star$ in seiner Nähe?

<!--en:7b2701b3-->
- **Schritt 1: Neuen Fehler mit der Hesse-Matrix bei $\mathbf{x}_k$ ausdrücken:** $\mathbf{x}^\star$ von der Newton-Aktualisierung abziehen und $(\nabla^2 f_k)^{-1}$ ausklammern:
  $$\mathbf{e}_{k+1} = \mathbf{e}_k - (\nabla^2 f_k)^{-1}\mathbf{g}_k = (\nabla^2 f_k)^{-1}\Big[\nabla^2 f_k\,\mathbf{e}_k - \big(\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star)\big)\Big]$$

<!--en:ea1e47b4-->
- **Schritt 2: Hauptsatz der Differential- und Integralrechnung anwenden:** Die Gradientendifferenz als Integral der Hesse-Matrix entlang der Strecke $\mathbf{x}^\star + t\mathbf{e}_k$ schreiben:
  $$\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star) = \int_0^1 \nabla^2 f(\mathbf{x}^\star + t\mathbf{e}_k)\,\mathbf{e}_k\,dt$$

<!--en:0d5d7ed3-->
- **Schritt 3: Terme unter dem Integral zusammenfassen:** Das Residuum in der eckigen Klammer ist $\int_0^1 \big[\nabla^2 f(\mathbf{x}_k) - \nabla^2 f(\mathbf{x}^\star + t\mathbf{e}_k)\big]\mathbf{e}_k\,dt$.

<!--en:dd09816c-->
::: {.slide title="Lokale quadratische Konvergenz: doppelt so viele korrekte Stellen"}

<!--en:92320639-2-->
[Konvergenzbeweis]{.kicker}

<!--en:8b88cef0-->
Gilt $\nabla^2 f(\mathbf{x}) \succeq \mu\mathbf{I}$ und ist $\nabla^2 f$ $L_2$-Lipschitz-stetig in einer Umgebung von $\mathbf{x}^\star$, so beträgt die Norm des Residuums in Integraldarstellung höchstens $\tfrac{L_2}{2}\|\mathbf{e}_k\|^2$:

<!--en:d82b1cf1-->
- **Verdopplung der korrekten Stellen:** Sobald für den skalierten Fehler $\frac{L_2}{2\mu}\|\mathbf{e}_k\| < 10^{-1}$ gilt, sinkt er mit jedem Newton-Schritt mindestens auf das Quadrat seines vorherigen Werts ($10^{-1} \to 10^{-2} \to 10^{-4} \to 10^{-8} \to 10^{-16}$).

<!--en:6597604e-->
- **Gedämpftes Newton-Verfahren:** Fern von $\mathbf{x}^\star$ sichert die Armijo-Backtracking-Liniensuche entlang $\mathbf{p}_k$ den Abstieg. Sie akzeptiert automatisch $\alpha_k = 1$ in der Nähe von $\mathbf{x}^\star$.

<!--en:d6437345-->
::: {.slide title="Newton-Verfahren im langgestreckten Tal und beim logistischen Verlust"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:0451e1a4-->
Test des Newton-Verfahrens an einer quadratischen Funktion mit Konditionszahl $\kappa = 1000$ und Hesse-Matrix $\mathrm{diag}(1, 1000)$ sowie an einer $\ell_2$-regularisierten logistischen Regression ($n = 200$ Beispiele, $d = 5$ Gewichte, $\nu = 0.1$):

<!--en:09fd8e35-->
- Im Tal mit $\kappa = 1000$ ersetzt $1$ Newton-Schritt $6{,}908$ Schritte des Gradientenabstiegs.

<!--en:df6d91a3-->
- Beim nichtquadratischen logistischen Verlust sinkt $\|\nabla f\|$ in jedem Schritt ungefähr auf das Quadrat des vorherigen Werts ($10^{-1}$ $\to$ $10^{-2}$ $\to$ $10^{-3}$ $\to$ $10^{-6}$ $\to$ $10^{-11}$ $\to$ $10^{-17}$). Das Ergebnis von $4{,}000$ Schritten des Gradientenabstiegs ergibt sich durch Lösen von $6$ linearen Gleichungssystemen.

<!--en:9313d397-->
::: {.slide title="Warum das exakte Newton-Verfahren bei großen Netzen scheitert"}

<!--en:bd602953-->
[Rechenkosten]{.kicker}

<!--en:fe17f1b1-->
- **Quadratischer Speicherbedarf, kubischer Rechenaufwand:** Für $\nabla^2 f(\mathbf{x}_k)$ sind $O(d^2)$ Speicher nötig. Das Lösen von $(\nabla^2 f_k)\mathbf{p}_k = -\mathbf{g}_k$ kostet $O(d^3)$ FLOPs pro Schritt.

<!--en:99761b6d-->
- **Grenzen der Skalierbarkeit:** Für eine lineare Schicht mit $d = 10^4$ ist der Aufwand tragbar ($400\text{ MB}$). Ein ConvNet oder ViT mit $d = 10^8$ braucht jedoch $40\text{ PB}$, ein LLM mit $d = 10^{11}$ sogar $10^{33}\text{ FLOPs}$.

<!--en:f41441c1-->
- **Abhilfe:** Speichersparende Quasi-Newton-Verfahren (L-BFGS) sowie diagonale Vorkonditionierer und Kronecker-Vorkonditionierer.

<!--en:4c7e7f6a-->
::: {.slide title="Anziehung durch Sattelpunkte auf nichtkonvexen Flächen"}

<!--en:4585c242-->
[Schwierigkeiten bei fehlender Konvexität]{.kicker}

<!--en:c1828f25-->
Das Newton-Verfahren springt zu einem Punkt mit $\nabla m_k(\mathbf{p}) = \mathbf{0}$. Deshalb ziehen auch Sattelpunkte und Maxima das Verfahren an, wenn $\nabla^2 f$ negative Eigenwerte hat:

<!--en:9852ee79-->
- **Schritt in die falsche Richtung:** Bei $f(x,y) = \tfrac12 x^2 + \cos y$ im Punkt $(0.5, 0.3)$ kehrt die Division von $-\sin y$ durch $f_{yy} = -\cos y < 0$ das Vorzeichen des Schritts um. Das Newton-Verfahren läuft bergauf zum Sattelpunkt $(0, 0)$.

<!--en:89279411-->
::: {.slide title="Newton-Verfahren und Gradientenabstieg nahe einem Sattelpunkt"}

<!--en:4585c242-2-->
[Schwierigkeiten bei fehlender Konvexität]{.kicker}

<!--en:561ee039-->
Beide Verfahren starten bei $(0.5, 0.3)$ auf der Funktion $f(x, y) = \tfrac12 x^2 + \cos y$. Dabei zeigt sich, wie ein genaueres lokales Modell zu einem schlechteren stationären Punkt führen kann:

<!--en:a6534960-->
Das Newton-Verfahren erreicht den Sattelpunkt $(0, 0)$ mit $f = 1$ in drei Schritten. Der Gradientenabstieg ($\eta = 1$) folgt dem negativen Gradienten bergab zum globalen Minimum $(0, \pi)$ mit $f = -1$.

<!--en:96f4d0c5-->
::: {.slide title="Krümmung aus Gradienten schätzen: die Sekantenregel"}

<!--en:0666593d-->
[Quasi-Newton-Verfahren]{.kicker}

<!--en:8d95ab26-->
In einer Dimension schätzt das **Sekantenverfahren** $f''(x_k)$ aus den letzten beiden Gradienten:

<!--en:5edd0033-->
- **Superlineare Konvergenz:** Für die Fehler gilt $e_{k+1} \approx C e_k e_{k-1}$. Die Zahl der korrekten Stellen wächst pro Schritt um den Faktor des Goldenen Schnitts $\approx 1.618$. Dazu genügen erste Ableitungen.

<!--en:cc0e7763-->
::: {.slide title="Die mehrdimensionale Sekantengleichung und BFGS"}

<!--en:0666593d-2-->
[Quasi-Newton-Verfahren]{.kicker}

<!--en:bd38ed44-->
In $d$ Dimensionen verwendet ein **Quasi-Newton-Verfahren** eine symmetrische, positiv definite Näherung $\mathbf{M}_k \approx (\nabla^2 f(\mathbf{x}_k))^{-1}$. Es aktualisiert sie anhand des beobachteten Schritts und der Gradientendifferenz:

<!--en:d1377c59-->
- **Sekantengleichung:** Für die quadratische Funktion $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ gilt $\mathbf{y}_k = \mathbf{A}\mathbf{s}_k$. Gefordert ist also, dass $\mathbf{M}_{k+1}$ den Vektor $\mathbf{y}_k$ wieder auf $\mathbf{s}_k$ abbildet:
  $$\mathbf{M}_{k+1}\,\mathbf{y}_k = \mathbf{s}_k$$

<!--en:b4415e0d-->
- **BFGS-Aktualisierung:** Mit $d$ Gleichungen ist eine symmetrische $d \times d$-Matrix nicht eindeutig bestimmt. **BFGS** wählt die symmetrische Matrix, die in einer gewichteten Frobeniusnorm den kleinsten Abstand zu $\mathbf{M}_k$ hat und dabei $\mathbf{M}_{k+1}\mathbf{y}_k = \mathbf{s}_k$ erfüllt:
  $$\mathbf{M}_{k+1} = \big(\mathbf{I} - \gamma_k\mathbf{s}_k\mathbf{y}_k^\top\big)\,\mathbf{M}_k\,\big(\mathbf{I} - \gamma_k\mathbf{y}_k\mathbf{s}_k^\top\big) + \gamma_k\mathbf{s}_k\mathbf{s}_k^\top, \qquad \gamma_k = \frac{1}{\mathbf{y}_k^\top \mathbf{s}_k}$$

<!--en:03b83d92-->
::: {.slide title="Wie BFGS die positive Definitheit erhält"}

<!--en:0666593d-3-->
[Quasi-Newton-Verfahren]{.kicker}

<!--en:d7345a7e-->
Mit $\mathbf{w} = (\mathbf{I} - \gamma_k\mathbf{y}_k\mathbf{s}_k^\top)\mathbf{z}$ und $\gamma_k = 1/(\mathbf{y}_k^\top\mathbf{s}_k)$ hat die quadratische Form von BFGS folgende Gestalt:

<!--en:cd00b45e-->
- **Löschen und neu schreiben:** $(\mathbf{I} - \gamma_k\mathbf{y}_k\mathbf{s}_k^\top)$ bildet $\mathbf{y}_k$ auf null ab. Der Term $+\gamma_k\mathbf{s}_k\mathbf{s}_k^\top$ sorgt dann für $\mathbf{M}_{k+1}\mathbf{y}_k = \mathbf{s}_k$.

<!--en:b9af28c9-->
- **Krümmungsbedingung:** Gelten $\mathbf{M}_k \succ \mathbf{0}$ und $\mathbf{y}_k^\top\mathbf{s}_k > 0$ (Letzteres erzwingt die Wolfe-Liniensuche), so gilt garantiert auch $\mathbf{M}_{k+1} \succ \mathbf{0}$.

<!--en:fd7c2ef7-->
::: {.slide title="BFGS rekonstruiert die inverse Hesse-Matrix exakt in d Schritten"}

<!--en:0666593d-4-->
[Quasi-Newton-Verfahren]{.kicker}

<!--en:cc93c9a1-->
Für $\mathbf{A} = \mathrm{diag}(1, 10)$ mit Startpunkt $(1, 1)$, Anfangsmatrix $\mathbf{M}_0 = \mathbf{I}$ und exakter Liniensuche erkundet BFGS in Schritt 1 die steile Richtung. Die inverse Hesse-Matrix $\mathbf{A}^{-1} = \mathrm{diag}(1, 0.1)$ rekonstruiert es in Schritt $d = 2$ exakt:

<!--en:85c20af9-->
- **L-BFGS für große $d$:** Eine dicht besetzte $d \times d$-Matrix $\mathbf{M}_k$ braucht nach wie vor $O(d^2)$ Speicher. **L-BFGS** speichert nur die letzten $m$ Paare $\{(\mathbf{s}_j, \mathbf{y}_j)\}$. Damit lässt sich $\mathbf{M}_k\mathbf{g}_k$ mit Zeit- und Speicherbedarf $O(md)$ berechnen. Das eignet sich besonders für deterministische Probleme, ist aber anfällig für Minibatch-Rauschen in $\mathbf{y}_k$.

<!--en:7ec0f921-->
::: {.slide title="Trust-Region-Verfahren: begrenzter Gültigkeitsbereich des Modells"}

<!--en:7e615035-->
[Trust-Regionen]{.kicker}

<!--en:780aac47-->
Das Newton-Verfahren lief zum Sattelpunkt von $\tfrac12 x^2 + \cos y$, weil es seiner lokalen Taylor-Näherung auf ganz $\mathbb{R}^d$ vertraute. Ein **Trust-Region-Verfahren** behält das vollständige quadratische Modell bei, einschließlich der Richtungen negativer Krümmung. Es beschränkt den Schritt $\mathbf{p}$ auf eine Kugel mit Radius $\Delta_k$:

<!--en:64d48c32-->
- **Existenz eines Minimierers:** Die abgeschlossene Kugel $\{\mathbf{p} : \|\mathbf{p}\| \le \Delta_k\}$ ist kompakt und $m_k$ ist stetig. Deshalb hat das Teilproblem immer einen globalen Minimierer, selbst wenn $\mathbf{B}_k = \nabla^2 f(\mathbf{x}_k)$ indefinit oder negativ definit ist.

<!--en:7321c6b9-->
- **Dem Sattelpunkt entkommen:** Entlang jedes Eigenvektors von $\mathbf{B}_k$ mit negativem Eigenwert ist das quadratische Modell nach unten gekrümmt. Bei der Minimierung von $m_k$ auf der Kugel führt der Schritt $\mathbf{p}$ daher vom Sattelpunkt weg bis zum Rand $\|\mathbf{p}\| = \Delta_k$.

<!--en:1c175282-->
::: {.slide title="Optimalitätsbedingungen auf der Trust-Region-Kugel"}

<!--en:7e615035-2-->
[Trust-Regionen]{.kicker}

<!--en:f761d83d-->
Ein Vektor mit $\|\mathbf{p}\| \le \Delta$ minimiert $m_k(\mathbf{p})$ auf der Kugel genau dann global, wenn es ein $\tau \ge 0$ gibt, das die folgenden Bedingungen erfüllt:

<!--en:e254b72c-->
- **Inneres ($\tau = 0$):** Gelten $\mathbf{B} \succ \mathbf{0}$ und $\|\mathbf{B}^{-1}\mathbf{g}\| \le \Delta$, genügt der reine Newton-Schritt.

<!--en:88b4d0f1-->
- **Rand ($\tau > -\lambda_{\min}(\mathbf{B})$):** Alle Eigenwerte nach oben verschieben, bis $\mathbf{B} + \tau\mathbf{I} \succ \mathbf{0}$ und $\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \Delta$ gelten. Dadurch kehrt sich der Schritt in Richtungen negativer Krümmung um und führt vom Sattelpunkt weg.

<!--en:2b44a7b4-->
::: {.slide title="Mit einem Trust-Region-Schritt aus dem Sattelpunkt entkommen"}

<!--en:7e615035-3-->
[Trust-Regionen]{.kicker}

<!--en:8d156676-->
Das Trust-Region-Teilproblem im Punkt $\mathbf{x}_0 = (0.5, 0.3)$ für $f(x,y) = \tfrac12 x^2 + \cos y$ mit Radius $\Delta = 0.5$ von Hand lösen:

<!--en:d5e5b3dc-->
- **Lokale Ableitungen:** $\mathbf{g} = (0.5, -0.296)$ und $\mathbf{B} = \mathrm{diag}(1, -0.955)$.

<!--en:f8fa9537-->
- **Schranke für positive Semidefinitheit:** Aus $\mathbf{B} + \tau\mathbf{I} = \mathrm{diag}(1 + \tau,\, \tau - 0.955) \succeq \mathbf{0}$ folgt $\tau \ge 0.955 > 0$. Der Schritt muss also auf dem Rand $\|\mathbf{p}\| = 0.5$ liegen.

<!--en:80912560-->
- **Gleichungssystem am Rand:** Zu lösen sind $(1 + \tau)p_1 = -0.5$ und $(\tau - 0.955)p_2 = 0.296$ unter der Nebenbedingung $p_1^2 + p_2^2 = 0.25$:
  $$\tau \approx 1.60, \qquad \mathbf{p} \approx (-0.19,\, +0.46), \qquad \mathbf{x}_1 = \mathbf{x}_0 + \mathbf{p} \approx (0.31,\, 0.76)$$

<!--en:72787237-->
- **Vergleich mit dem Newton-Verfahren:** Der unbeschränkte Newton-Schritt $(-0.5, -0.31)$ führt *zum* Sattelpunkt $(0, 0)$. Die verschobene Matrix $(\mathbf{B} + \tau\mathbf{I}) \succ \mathbf{0}$ kehrt das Vorzeichen von $p_2$ um. Der Schritt führt vom Sattelpunkt *weg* und senkt $f$ von $1.080$ auf $0.771$.

<!--en:d23d7de4-->
::: {.slide title="Radiusanpassung anhand der tatsächlichen und der vorhergesagten Abnahme"}

<!--en:7e615035-4-->
[Trust-Regionen]{.kicker}

<!--en:1124b561-->
Nach der Berechnung des vorgeschlagenen Schritts $\mathbf{p}_k$ vergleicht der Algorithmus die tatsächliche Abnahme des Verlusts mit der Vorhersage des quadratischen Modells:

<!--en:01c8610b-->
- Gilt am Rand $r_k > 0.75$, verdoppelt sich $\Delta_k$ ($0.5 \to 1 \to 2$). Sobald der Newton-Schritt in die Kugel passt ($k \ge 3$), gilt $\tau = 0$. Das Verfahren geht in das reine Newton-Verfahren über und erreicht $(0, \pi)$ in $6$ Schritten.

<!--en:be2f2547-->
::: {.slide title="Trajektorien des Newton-Verfahrens und des Trust-Region-Verfahrens"}

<!--en:7e615035-5-->
[Trust-Regionen]{.kicker}

<!--en:ce0b7a3d-->
- **Teilbild (a): lokale Sattelfläche:** Bei $(0.5, 0.3)$ bildet das quadratische Modell $m_0(\mathbf{p})$ eine hyperbolische Sattelfläche. Das reine Newton-Verfahren springt zu ihrem stationären Punkt nahe dem Ursprung. Die Minimierung von $m_0$ auf der Kugel $\|\mathbf{p}\| \le 0.5$ folgt der nach unten gekrümmten Fläche in $y$-Richtung.

<!--en:fc66f655-->
- **Teilbild (b): globale Trajektorie:** Das Trust-Region-Verfahren verlässt in zwei Schritten das Gebiet mit indefiniter Hesse-Matrix, vergrößert $\Delta$ und macht ab dem vierten Schritt reine Newton-Schritte ($\tau = 0$). Es konvergiert quadratisch zum globalen Minimum $(0, \pi)$.

<!--en:a4e42d60-->
::: {.slide title="Verständnisfragen zu Krümmung, Kosten und Dämpfung"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:ab3740d8-->
Fragen zu Verfahren zweiter Ordnung und Quasi-Newton-Verfahren:

<!--en:e7825b8e-->
1. Das Newton-Verfahren hat den Bereich quadratischer Konvergenz mit skaliertem Fehler $\frac{L_2}{2\mu}\|\mathbf{e}_k\| = 10^{-1}$ erreicht. Wie viele Newton-Schritte braucht es bis $10^{-8}$? Wie viele Schritte braucht der Gradientenabstieg bei $\kappa = 10^3$?

<!--en:2a2faf18-->
2. Ein Trust-Region-Teilproblem habe $\mathbf{B} = \mathrm{diag}(1, 4)$, $\mathbf{g} = (1, 4)$ und Radius $\Delta = 2$. Wie lauten der optimale Multiplikator $\tau$ und der optimale Schritt $\mathbf{p}$?

<!--en:1358f6eb-->
3. Warum braucht BFGS die Krümmungsbedingung $\mathbf{y}_k^\top \mathbf{s}_k > 0$?

<!--en:a1c61764-->
**Lösungen:** **(1) 3 Newton-Schritte:** In jedem Schritt sinkt der skalierte Fehler mindestens auf das Quadrat seines vorherigen Werts ($10^{-1}$ $\to$ $10^{-2}$ $\to$ $10^{-4}$ $\to$ $10^{-8}$). Der Gradientenabstieg bei $\kappa = 10^3$ braucht $\approx 1{,}150$ Schritte pro Stelle, also etwa **$8{,}050$ Schritte** für sieben Stellen. **(2) $\tau = 0$ und $\mathbf{p} = (-1, -1)$:** Hier gilt $\mathbf{B} \succ \mathbf{0}$. Der unbeschränkte Newton-Schritt $-\mathbf{B}^{-1}\mathbf{g} = (-1, -1)$ hat die Norm $\sqrt{2} \approx 1.414 < \Delta = 2$ und liegt damit im Inneren der Kugel. **(3) Damit $\mathbf{M}_{k+1} \succ \mathbf{0}$ gilt:** Bei der quadratischen Form $\mathbf{z}^\top\mathbf{M}_{k+1}\mathbf{z} = \mathbf{w}^\top\mathbf{M}_k\mathbf{w} + (\mathbf{s}_k^\top\mathbf{z})^2 / (\mathbf{y}_k^\top\mathbf{s}_k)$ bleibt die positive Definitheit genau dann erhalten, wenn $\mathbf{y}_k^\top\mathbf{s}_k > 0$ gilt.

<!--en:9526d5ca-->
::: {.slide title="Zusammenfassung: Verfahren zweiter Ordnung und krümmungsbasierte Verfahren"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:250c6568-->
| Verfahren | Schritt / Teilproblem | Konvergenzrate | Speicher- und Rechenbedarf pro Schritt | Einsatz und Grenzen im Deep Learning |

<!--en:92089457-->
| **Reines Newton-Verfahren** | $\mathbf{p}_k = -(\nabla^2 f_k)^{-1}\mathbf{g}_k$ | Quadratisch ($\|\mathbf{e}_{k+1}\| \le \frac{L_2}{2\mu}\|\mathbf{e}_k\|^2$), affininvariant | $O(d^2)$ Speicher, $O(d^3)$ zum Lösen des Gleichungssystems | Für große $d$ zu teuer, wird von Sattelpunkten angezogen |

<!--en:18679a90-->
| **Gedämpftes Newton-Verfahren** | $\mathbf{x}_k + \alpha_k \mathbf{p}_k$ mit Armijo-Liniensuche | Globaler Abstieg bei $\nabla^2 f_k \succ \mathbf{0}$, lokal quadratisch ($\alpha_k = 1$) | $O(d^2)$ Speicher, $O(d^3)$ zum Lösen des Gleichungssystems | Verhindert zu weite Schritte fern von $\mathbf{x}^\star$, setzt weiterhin $\nabla^2 f_k \succ \mathbf{0}$ voraus |

<!--en:052258c2-->
| **BFGS / L-BFGS** | Rang-2-Aktualisierung erfüllt $\mathbf{M}_{k+1}\mathbf{y}_k = \mathbf{s}_k$ | Superlinear ($d$ Schritte bei $d$-dimensionaler quadratischer Funktion) | BFGS: $O(d^2)$, L-BFGS: $O(md)$ | Standard für deterministische Modellanpassungen mittlerer Größe, empfindlich gegenüber Minibatch-Rauschen |

<!--en:f6373c9a-->
| **Trust-Region-Verfahren** | $\min_{\|\mathbf{p}\| \le \Delta_k} m_k(\mathbf{p})$ $\iff$ $(\mathbf{B}_k + \tau\mathbf{I})\mathbf{p} = -\mathbf{g}_k$ | Entkommt Sattelpunkten in Richtungen negativer Krümmung, lokal quadratisch | Ein Teilproblem pro Schritt lösen | Vorbild für die spektrale Dämpfung ($\mathbf{B} + \tau\mathbf{I}$) bei Vorkonditionierern zweiter Ordnung |
