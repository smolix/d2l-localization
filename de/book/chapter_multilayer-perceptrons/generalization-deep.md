<!--en:641953e1-->
# Generalisierung im Deep Learning

<!--en:d6e3203f-->
In :numref:`sec_generalization_basics` interpolierte ein Polynom 19. Grades mit $20$ Parametern $20$ verrauschte Datenpunkte. Auf den Testdaten nahm es extrem große Werte an. Nach dieser klassischen Vorstellung müssten wir bei unserem Fashion-MNIST-MLP aus :numref:`sec_mlp-implementation` mit einer starken Überanpassung (engl. overfitting) rechnen: Es hat $203{,}530$ Parameter für $60{,}000$ Trainingsbilder ($3.4$ Parameter pro Bild). Das sollte besonders dann ins Gewicht fallen, wenn wir so lange trainieren, bis das Modell jedes einzelne Trainingsbild interpoliert. Dennoch erreichte es auf ungesehenen Bildern eine Genauigkeit von $\sim 87\%$. Wie wir weiter unten sehen, klassifiziert dieselbe Architektur selbst mit nur $1{,}000$ Trainingsbildern und einer Trainingsgenauigkeit von $100\%$ noch rund $80\%$ der ungesehenen Bilder richtig.

<!--en:9eb3bf99-->
Warum stark überparametrisierte Netze generalisieren, ist eine der zentralen offenen Fragen des modernen maschinellen Lernens. Wir verbinden kontrollierte Experimente mit exakten Ergebnissen für mathematisch zugängliche Modelle. Vier Beobachtungen tragen zur Erklärung bei:

<!--en:252d352b-->
1. **Auswendiglernen zufälliger Labels:** Herkömmliche MLPs erreichen bei völlig zufälligen Labels mühelos eine Trainingsgenauigkeit von $100\%$ :cite:`zhang2021understanding`. Klassische verteilungsfreie Kapazitätsschranken, etwa solche auf Basis der VC-Dimension, können ihre Generalisierung bei echten Labels folglich nicht erklären.

<!--en:2960a683-->
2. **Doppelter Abstieg:** Erhöhen wir die Modellbreite über die Interpolationsschwelle $h = n$ hinaus, zeigt der Testfehler bei $h = n$ eine ausgeprägte Spitze. Dort ist die Merkmalsmatrix fast singulär. Danach folgt ein *zweiter Abstieg*, weil die zusätzlichen Parameter Interpolanten mit kleinerer Norm erlauben :cite:`Belkin.Hsu.Ma.ea.2019`.

<!--en:749683d0-->
3. **Impliziter Bias des Gradientenabstiegs:** Unter unendlich vielen Lösungen mit Verlust null wählt der Gradientenabstieg bei quadratischem Verlust automatisch den Interpolanten kleinster Norm. Bei logistischem Verlust wählt er die Maximum-Margin-Trennebene :cite:`Soudry.Hoffer.Nacson.ea.2018`.

<!--en:847ab152-->
4. **Zuerst die unverfälschten Muster lernen und rechtzeitig aufhören:** Bei verrauschten Datensätzen passt der Gradientenabstieg das Modell an die unverfälschten, strukturierten Muster an, lange bevor es verfälschte Labels auswendig lernt :cite:`Arpit.Jastrzebski.Ballas.ea.2017`. Deshalb ist *Early Stopping* eine wirksame Regularisierung.

<!--en:0083e8bb-->
## Auswendiglernen und Kapazität

<!--en:3ed959a8-->
### Zufällige Labels

<!--en:ff98899d-->
Prüfen wir wie :citet:`zhang2021understanding`, wie viel das Modell allein durch Auswendiglernen erfassen kann. Wir ziehen $2{,}000$ Trainingsbilder aus Fashion-MNIST ($1{,}000$ für das Training, $1{,}000$ für die Validierung) sowie $2{,}000$ Testbilder. Wir definieren die Hilfsklassen `ArrayData` und `Tracker` und ein `MLP` mit 256 Einheiten und He-Initialisierung. Dann trainieren wir jeweils 200 Epochen, einmal mit den wahren Labels (`fraction=0`) und einmal mit völlig zufälligen Labels (`fraction=1`):

<!--en:6639dbaf-->
Zwei kleine Hilfsklassen vereinfachen die folgenden Experimente. `ArrayData` fasst NumPy-Arrays zu zufällig gemischten Minibatches zusammen. `Tracker` ist eine Unterklasse von `Trainer` und nimmt ein beliebiges Wörterbuch benannter Datensätze entgegen und protokolliert für jeden Datensatz die Genauigkeit nach jeder Epoche.

<!--en:bee31f7e-->
Wir verwenden dasselbe ReLU-MLP mit 256 Einheiten wie in :numref:`sec_mlp-implementation`. Mit der He-Initialisierung (:numref:`subsec_he_init`) anstelle von `std=0.01` macht das Training schon ab Epoche 1 Fortschritte. Die Bias-Terme sind anfangs null, die Eingaben liegen bereits als flache Vektoren vor.

<!--en:48189174-->
Nun definieren wir `corrupt(y, fraction, rng)`. Für jedes Trainingslabel zieht die Funktion die Ersatzklasse gleichverteilt aus $\{0, \ldots, 9\}$. Die Wahrscheinlichkeit für das Ersetzen beträgt `fraction`. Wir trainieren `MLP(num_hiddens=256, lr=0.1)` jeweils $200$ Epochen mit `fraction = 0` (wahre Labels) und `fraction = 1` (rein zufällige Labels) und zeichnen den Verlauf der Trainings- und Testgenauigkeit auf:

<!--en:03e58df0-->
Was zeigen die beiden Trainingsläufe?

<!--en:18ad0d53-->
- Mit **wahren Labels** erreicht die Trainingsgenauigkeit innerhalb von 60 bis 90 Epochen $100\%$. Die Testgenauigkeit pendelt sich oberhalb von $80\%$ ein, selbst bei nur $1{,}000$ Trainingsbildern.

<!--en:fd3c32cc-->
- Mit **völlig zufälligen Labels** erreichen dieselbe Architektur und derselbe Optimierer ebenfalls $100\%$ Trainingsgenauigkeit. Sie brauchen dafür etwas länger ($\sim 120$–$150$ Epochen). Die Testgenauigkeit bleibt mit $10\%$ auf Zufallsniveau.

<!--en:43366802-->
### Grenzen von Kapazitätsschranken und induktiver Bias

<!--en:a90b446d-->
Bei zufälligen Labels erreicht unser MLP einen Trainingsfehler von null ($\epsilon_{\mathcal{S}}(f) = 0$), während der wahre Populationsfehler $\epsilon(f) = 0.90$ beträgt. Für *jede* verteilungsfreie Schranke der Form $\epsilon(f) \leq \epsilon_{\mathcal{S}}(f) + B(\mathcal{F}, n, \delta)$, die bei gegebener Stichprobengröße und Konfidenz nur von der Hypothesenklasse $\mathcal{F}$ abhängt, muss daher $B(\mathcal{F}, n, \delta) \geq 0.90$ gelten. Damit liefert die Schranke auch für echte Labels keine aussagekräftige Garantie (Übung :exref:`ex_generalization-deep-complexity`).

<!--en:f8ea95b9-->
Nach dem *No-Free-Lunch-Theorem* :cite:`Wolpert.1996` kann kein Lernverfahren über alle Verteilungen hinweg generalisieren. Ob es generalisiert, hängt davon ab, ob sein *induktiver Bias* zur Struktur der realen Daten passt, also davon, welche der unendlich vielen interpolierenden Funktionen es bevorzugt. So interpoliert ein 1-Nächster-Nachbar-Klassifikator $100\%$ seiner Trainingsdaten und generalisiert dennoch gut, wenn benachbarte Eingaben dasselbe Label tragen :cite:`Cover.Hart.1967`.

<!--en:60598fcf-->
## Doppelter Abstieg

<!--en:2b6cd1b4-->
### Interpolationsschwelle und Kondition

<!--en:c79c8e74-->
Was geschieht, wenn wir die Modellbreite $h$ über den Stichprobenumfang $n$ hinaus erhöhen? Untersuchen wir ein mathematisch zugängliches Modell mit *zufälligen Merkmalen* (engl. random features) :cite:`Rahimi.Recht.2007`. Wir halten die erste Schicht $\mathbf{W}^{(1)} \in \mathbb{R}^{784 \times 10000}$ nach der He-Initialisierung fest. Damit berechnen wir für unsere $n = 1{,}000$ zentrierten Trainingsbilder und die zugehörigen Testdaten vorab $10{,}000$ zufällige ReLU-Merkmale (`H` und `H_test`). Wir kodieren die Labels als One-Hot-Matrix `Y` und definieren `error(H, W2, y)`:

<!--en:63bf0de5-->
Für die Breite `h` der versteckten Schicht durchlaufen wir nun 15 Werte von $50$ bis $10{,}000$. Wir bestimmen die Ausgabegewichte $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times 10}$ aus den ersten `h` Merkmalsspalten `H[:, :h]`. Dazu berechnen wir mit der Methode der kleinsten Quadrate die Lösung kleinster Norm (`np.linalg.lstsq`). Dabei protokollieren wir den kleinsten Singulärwert `s_min[h]` und die Frobeniusnorm `norm[h]`:

<!--en:d3163073-->
Die Kurve zeigt den bekannten *doppelten Abstieg* (engl. double descent) :cite:`Belkin.Hsu.Ma.ea.2019`:

<!--en:693834c8-->
1. **Klassische U-Kurve ($h < n$):** Der Testfehler sinkt von $\sim 30\%$ bei $h = 50$ auf $\sim 23\%$ in der Nähe von $h = 200$. Dann schnellt er auf $\sim 90\%$ hoch, genau an der *Interpolationsschwelle* $h = n = 1{,}000$.

<!--en:702d0898-->
2. **Zweiter Abstieg ($h > n$):** Wächst $h$ über $n = 1{,}000$ hinaus bis auf $10{,}000$, interpoliert jedes Modell den Trainingsdatensatz. Dennoch fällt der Testfehler wieder auf $\sim 20\%$ und liegt damit unter dem des besten unterparametrisierten Modells.

<!--en:806231fe-->
Warum steigt der Testfehler bei $h = n$ so steil an und sinkt für $h \gg n$ wieder? Betrachten wir den kleinsten Singulärwert $s_{\min}$ von $\mathbf{H}_{:, :h}$ und die Frobeniusnorm $\|\mathbf{W}^{(2)}\|_{\textrm{F}}$:

<!--en:f2d17f43-->
Bei $h = n = 1{,}000$ ist $\mathbf{H}$ eine quadratische Matrix der Größe $1000 \times 1000$ mit einem winzigen kleinsten Singulärwert $s_{\min}$. In der eindeutigen exakten Lösung $\mathbf{H}^{-1}\mathbf{Y}$ wird das Labelrauschen in der zugehörigen Richtung durch $s_{\min}$ geteilt. Dadurch wächst $\|\mathbf{W}^{(2)}\|_{\textrm{F}}$ auf mehrere Hundert oder mehr. Für $h \gg n$ eröffnen die zusätzlichen Spalten einen $(h - n)$-dimensionalen affinen Unterraum exakter Interpolanten. Die Lösung *kleinster Norm* kann dadurch schlecht konditionierten Richtungen ausweichen, und $\|\mathbf{W}^{(2)}\|_{\textrm{F}}$ sinkt bis auf $\sim 2$. Halten wir nun $h = 1{,}000$ fest und verdoppeln den Trainingsdatensatz von $n = 500$ auf $n = 1{,}000$, trifft das unregularisierte Modell genau die Schwelle $h = n$. Der Testfehler *steigt* von $\sim 30\%$ auf $85\text{--}90\%$:

<!--en:4cab65f5-->
## Impliziter Bias des Gradientenabstiegs

<!--en:407a21a4-->
Der zweite Abstieg kam oben dadurch zustande, dass `lstsq` unter unendlich vielen Lösungen mit Verlust null den Interpolanten *kleinster Norm* auswählt. Bemerkenswerterweise hat der Gradientenabstieg von sich aus genau diese Präferenz. Dieser *implizite Bias* (oder diese *implizite Regularisierung*) wirkt auch ohne expliziten Strafterm.

<!--en:e6918419-->
### Interpolation mit kleinster Norm bei quadratischem Verlust

<!--en:f0ff2f79-->
Für ein überparametrisiertes lineares Gleichungssystem $\mathbf{H}\mathbf{w} = \mathbf{y}$ ($h > n$, voller Zeilenrang) ist $\mathbf{w}^\star = \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{y}$ der eindeutige Interpolant kleinster Norm.

<!--en:6f79f12a-->
**Satz.** Der Gradientenabstieg für $L(\mathbf{w}) = \frac{1}{2}\|\mathbf{H}\mathbf{w} - \mathbf{y}\|_2^2$ mit Startwert $\mathbf{w}_0 = \mathbf{0}$ und einer Schrittweite, die Konvergenz gewährleistet, konvergiert zum Interpolanten kleinster Norm $\mathbf{w}^\star$.

<!--en:62e29c84-->
**Beweis.** Jeder Gradient $\nabla L(\mathbf{w}) = \mathbf{H}^\top(\mathbf{H}\mathbf{w} - \mathbf{y})$ liegt im Zeilenraum von $\mathbf{H}$. Starten wir bei $\mathbf{w}_0 = \mathbf{0}$, bleibt jede Iterierte $\mathbf{w}_t$ und damit auch der Grenzwert $\mathbf{w}_{\infty} = \mathbf{H}^\top\mathbf{a}$ im Zeilenraum von $\mathbf{H}$ (:numref:`fig_mdl-mlp-min-norm`). Aus $\mathbf{H}\mathbf{w}_{\infty} = \mathbf{H}\mathbf{H}^\top\mathbf{a} = \mathbf{y}$ folgen $\mathbf{a} = (\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{y}$ und $\mathbf{w}_{\infty} = \mathbf{w}^\star$. Für jeden anderen Interpolanten $\mathbf{w}^\star + \mathbf{v}$ gilt $\mathbf{H}\mathbf{v} = \mathbf{0}$ und damit $\mathbf{v} \perp \mathbf{w}^\star$. Seine Norm ist also strikt größer, denn ihr Quadrat beträgt $\|\mathbf{w}^\star\|_2^2 + \|\mathbf{v}\|_2^2$. $\blacksquare$

<!--en:3127daa2-->
![Bei einer unterbestimmten linearen Gleichung $\mathbf{h}^\top\mathbf{w} = y$ bewegt sich der Gradientenabstieg stets senkrecht zur Lösungshyperebene. Vom Startpunkt $\mathbf{0}$ (blau) gelangt er daher zur Lösung kleinster Norm $\mathbf{w}^\star$. Beim Start in $\mathbf{w}_0 \neq \mathbf{0}$ (orange) bleibt die Komponente von $\mathbf{w}_0$ im Kern erhalten.](../img/mdl-mlp-min-norm.svg)

<!--en:b74f1d51-->
Überprüfen wir diesen Satz numerisch anhand von $20$ Gleichungen mit $100$ Unbekannten. Dazu vergleichen wir den Start bei $\mathbf{w}_0 = \mathbf{0}$ mit einer Zufallsinitialisierung:

<!--en:f71ad080-->
### Maximum-Margin-Klassifikation bei logistischem Verlust

<!--en:13a028a2-->
Für linear trennbare Daten zur binären Klassifikation ($y_i \in \{-1, +1\}$) und den logistischen Verlust (Kreuzentropie) $\sum_i \log(1 + \exp(-y_i \mathbf{w}^\top\mathbf{x}_i))$ gilt $\|\mathbf{w}_t\|_2 \to \infty$ für $t \to \infty$. Wie :citet:`Soudry.Hoffer.Nacson.ea.2018` bewiesen haben, konvergiert die *Richtung* $\mathbf{w}_t / \|\mathbf{w}_t\|_2$ mit der Rate $1/\log t$ gegen die Normale der *Maximum-Margin-Trennebene* der Hard-Margin-SVM (:numref:`fig_mdl-mlp-max-margin`). Der Grund ist, dass die Stützvektoren mit den kleinsten Margins im exponentiell abklingenden Teil von $\log(1 + e^{-m})$ dominieren. Wie die Umskalierung einzelner Schichten die Bestimmung von Margins anhand der Parameternorm bei mehrschichtigen ReLU-Netzen erschwert, zeigt Übung :exref:`ex_generalization-deep-implicit-bias`.

<!--en:15bbbbb2-->
![Bei der logistischen Regression mit linear trennbaren Daten dreht der Gradientenabstieg $\mathbf{w}_t / \|\mathbf{w}_t\|_2$ in Richtung der Normalen der Maximum-Margin-Hyperebene (links). Diese Drehung setzt sich mit logarithmischer Konvergenzrate fort, lange nachdem die Trainingsgenauigkeit $100\%$ erreicht hat (rechts).](../img/mdl-mlp-max-margin.svg)

<!--en:99fdd486-->
## Early Stopping

<!--en:fcf4a76f-->
Selbst wenn die Kapazität eines MLP ausreicht, um verfälschte Labels auswendig zu lernen, lernt der Gradientenabstieg *zuerst die unverfälschten, strukturierten Muster*. Das Modell lernt das *Labelrauschen erst viel später* auswendig :cite:`Arpit.Jastrzebski.Ballas.ea.2017`. Ersetzen wir $20\%$ der Labels in unserem Trainingsdatensatz von $1{,}000$ Bildern durch zufällige Labels. Über 200 Epochen verfolgen wir die Genauigkeit getrennt für die unverfälschten Trainingsbeispiele, die falsch beschrifteten Trainingsbeispiele (gemessen an ihren verfälschten Labels) und die $1{,}000$ unverfälschten Validierungsbilder:

<!--en:e81d015c-->
Bis zur 20. Epoche klassifiziert das MLP über $80\%$ der unverfälschten Trainingsbilder richtig, hat aber weniger als $20\%$ der verfälschten Labels auswendig gelernt. In dieser frühen Phase erreicht die Validierungsgenauigkeit ihren Höchstwert. In späteren Epochen sinkt sie, weil das Modell das Rauschen auswendig lernt. Beim *Early Stopping* beenden wir das Training, sobald sich die Validierungsgüte für eine vorgegebene *Wartezeit* (engl. patience) nicht mehr verbessert (Übung :exref:`ex_generalization-deep-patience`). Dann stellen wir den Checkpoint mit der besten Validierungsgüte wieder her. So gewinnen wir mehrere Prozentpunkte Testgenauigkeit zurück und sparen Rechenaufwand. Bei linearen Problemen der kleinsten Quadrate wirkt Early Stopping ähnlich wie $\ell_2$-Gewichtszerfall (siehe Übung :exref:`ex_generalization-deep-early-stopping-ridge`):

<!--en:bb6d2653-->
Eine bekannte Ausnahme vom Nutzen des Early Stopping ist das *Grokking* :cite:`Power.Burda.Edwards.ea.2022` (:numref:`fig_grokking`, Übung :exref:`ex_generalization-deep-grokking`). Trainieren wir ein Netz mit Gewichtszerfall auf kleinen algorithmischen Aufgaben wie der modularen Addition, lernt es die Trainingstabelle rasch auswendig. Die Validierungsgenauigkeit bleibt $100\times$ so lange nahe dem Zufallsniveau und springt dann unvermittelt auf $100\%$.

<!--en:e316326b-->
![Grokking (:citet:`Power.Burda.Edwards.ea.2022`): Bei modularer Arithmetik erreicht die Trainingsgenauigkeit innerhalb von $\sim 10^3$ Schritten ihren Höchstwert. Erst nach $\sim 10^5$ Schritten steigt die Validierungsgenauigkeit sprunghaft an: Das Netz generalisiert.](../img/mdl-mlp-grokking.svg)

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:65b03958-->
Überparametrisierte neuronale Netze können völlig zufällige Labels auswendig lernen und bei strukturierten Daten gut generalisieren. Verteilungsfreie Kapazitätsschranken liefern daher keine aussagekräftige Garantie. Eine Erklärung muss den induktiven Bias berücksichtigen. Bei der Methode der kleinsten Quadrate mit zufälligen Merkmalen steigt der Testfehler an der Interpolationsschwelle $h = n$ steil an. Dort treibt der winzige kleinste Singulärwert $s_{\min}(\mathbf{H})$ die Norm $\|\mathbf{W}^{(2)}\|$ in die Höhe. Für $h \gg n$ sinkt der Testfehler ein zweites Mal, weil der Interpolant kleinster Norm die Gewichte auf redundante Merkmale verteilt. Bemerkenswerterweise findet der bei null gestartete Gradientenabstieg bei quadratischem Verlust diesen Interpolanten automatisch. Bei logistischem Verlust konvergiert seine Richtung gegen die Normale der Maximum-Margin-Trennebene. Außerdem lernt der Gradientenabstieg zusammenhängende Muster, bevor das Modell verfälschte Labels auswendig lernt. Deshalb bietet Early Stopping anhand eines Validierungsdatensatzes eine einfache und wirksame Regularisierung.

<!--en:2294243a-->
## Übungen

<!--en:6e0d4f0a-->
1. **Grenzen von Komplexitätsmaßen.** Betrachten Sie eine Generalisierungsschranke, die für jede Datenverteilung gilt. Mit Wahrscheinlichkeit mindestens $1 - \delta$ über die zufällige Ziehung von $n$ Trainingsbeispielen erfüllt jeder Klassifikator $f$ aus einer Klasse $\mathcal{F}$ die Ungleichung

    $$\epsilon(f) \leq \epsilon_\mathcal{S}(f) + B(\mathcal{F}, n, \delta).$$

    Dabei ist $\epsilon(f)$ der Populationsfehler und $\epsilon_\mathcal{S}(f)$ der Trainingsfehler. Der Term $B$ hängt nur von $\mathcal{F}$, $n$ und $\delta$ ab, ebenso wie der Term $c\sqrt{(v + \log(1/\delta))/n}$ in der VC-Schranke. Im Text haben wir die Aussagekraft einer solchen Schranke anhand von 1000 Bildern mit zufälligen Labels untersucht.
    1. Angenommen, das Training findet auch für alle 60 000 Trainingsbilder von Fashion-MNIST einen Klassifikator aus $\mathcal{F}$ mit Trainingsfehler null, wenn die Labels gleichverteilt zufällig gezogen werden. Dies haben :citet:`zhang2021understanding` bei Bildklassifikatoren beobachtet. Welche untere Schranke für $B(\mathcal{F}, 60{,}000, \delta)$ folgt daraus? Was garantiert die Schranke dann für das Netz, das mit den wahren Labels trainiert wurde?
    1. Bei vielen später entwickelten Schranken hängt der zweite Term vom Klassifikator ab, den das Training liefert, etwa von der Norm seiner Gewichte oder von seinen Margins auf den Trainingsbeispielen. Welcher Schritt des Arguments gilt dann nicht mehr? Wie muss sich eine solche Größe beim Training mit zufälligen Labels verhalten, damit die Schranke bei wahren Labels aussagekräftig sein kann?
   :exlabel:`ex_generalization-deep-complexity`

<!--en:6c2fa552-->
1. [code] **Abbruch nach einer Wartezeit.** `FashionMNIST` verwendet den Testdatensatz von Fashion-MNIST zur Validierung. Legen Sie stattdessen 10 000 Trainingsbilder als Validierungsdatensatz beiseite. Trainieren Sie wie in :numref:`sec_mlp-implementation` ein MLP mit einer versteckten Schicht aus 256 Einheiten auf den übrigen Bildern. Wählen Sie eine feste Anzahl von Epochen, die groß genug ist, dass Überanpassung eintritt. Zeichnen Sie nach jeder Epoche den Validierungsverlust und die Testgenauigkeit auf. Die Abbruchregel hat eine Toleranz $\theta$ und eine Wartezeit $k$. Sie beendet das Training, wenn eine Verbesserung des bisherigen Bestwerts um mehr als $\theta$ für $k$ aufeinanderfolgende Epochen ausbleibt. Sie behält die Parameter aus der Epoche mit dem bisher niedrigsten Validierungsverlust bei.
    1. Halten Sie $\theta$ fest und wenden Sie die Regel für $k \in \{1, 5, 20\}$ auf die aufgezeichneten Kurven an. Geben Sie für jedes $k$ die Abbruchepoche, die beibehaltene Epoche und die Testgenauigkeit in beiden Epochen an.
    1. Wie hängt die Testgenauigkeit in der beibehaltenen Epoche von $k$ ab? Setzen Sie diese Abhängigkeit in Beziehung zu den Schwankungen des Validierungsverlusts von Epoche zu Epoche.

    *Nach Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap3.html), Kapitel 3.*
   :exlabel:`ex_generalization-deep-patience`

<!--en:31c4eb74-->
1. [code] **Gewichtszerfall an der Interpolationsschwelle.** Ersetzen Sie im Experiment zum doppelten Abstieg die Lösung kleinster Norm durch die Ridge-Lösung $\mathbf{W}^{(2)} = (\frac{1}{n}\mathbf{H}^\top\mathbf{H} + \lambda
   \mathbf{I})^{-1}\frac{1}{n}\mathbf{H}^\top\mathbf{Y}$ aus :numref:`sec_weight_decay`. Wie dort wird der quadratische Fehler über die $n$ Beispiele gemittelt.
    1. Tragen Sie den Testfehler für $\lambda \in \{10^{-5},
       10^{-3}, 10^{-1}\}$ gegen die Breite auf. Was geschieht mit der Spitze bei $h = n$?
    1. Zeigen Sie, dass sich die Ridge-Lösung für $\lambda \to 0$ bei $h > n$ der Lösung kleinster Norm nähert. Hinweis: Schreiben Sie $(\frac{1}{n}\mathbf{H}^\top\mathbf{H} + \lambda
       \mathbf{I})^{-1}\frac{1}{n}\mathbf{H}^\top =
       \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top + n\lambda \mathbf{I})^{-1}$.
    1. Wählen Sie für jede Breite $\lambda$ anhand der 1000 Validierungsbilder. Fällt die Kurve des so abgestimmten Modells weiterhin zweimal?

<!--en:2d7b934c-->
1. **Early Stopping und Gewichtszerfall.** ● Early Stopping begrenzt die Zahl der Trainingsepochen, Gewichtszerfall die Gewichte. Für lineare Probleme der kleinsten Quadrate lässt sich dieser Zusammenhang präzisieren. Die Matrix $\mathbf{X}^\top\mathbf{X}/n$ habe die Eigenwerte $\mu_i > 0$, und $\mathbf{w}^*$ minimiere $L(\mathbf{w}) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$. Führen Sie für $L$ einen Gradientenabstieg vom Startwert $\mathbf{w}_0 = \mathbf{0}$ mit Lernrate $\eta < 1/\max_i \mu_i$ aus.
    1. Zeigen Sie, dass nach $t$ Schritten die Koordinate von $\mathbf{w}_t$ in Richtung des $i$-ten Eigenvektors das $1 - (1 - \eta\mu_i)^t$-Fache der entsprechenden Koordinate von $\mathbf{w}^*$ beträgt.
    1. Der Minimierer von $L(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$ skaliert dieselbe Koordinate mit dem Faktor $\mu_i/(\mu_i + \lambda)$ (:numref:`subsec_wd-shrinkage`). Für welches $\lambda$ als Funktion von $\eta$ und $t$ stimmen die beiden Faktoren näherungsweise überein, sowohl in den Richtungen mit $\eta\mu_i t \ll 1$ als auch in denen mit $\eta\mu_i t \gg 1$?

    *Nach Ian Goodfellow, Yoshua Bengio und Aaron Courville, [Deep Learning](https://www.deeplearningbook.org/contents/regularization.html), Abschnitt 7.8.*
   :exlabel:`ex_generalization-deep-early-stopping-ridge`

<!--en:f73ebc79-->
1. **Impliziter Bias des logistischen Verlusts.** ● Bei linear trennbaren Daten konvergiert die Richtung des Gradientenabstiegs für den logistischen Verlust gegen die Normale der Maximum-Margin-Trennebene :cite:`Soudry.Hoffer.Nacson.ea.2018`. Betrachten Sie Labels $y_i \in \{-1, +1\}$.
    1. Zeigen Sie für einen linearen Prädiktor $f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x}$, dass bei trennbaren Daten der Verlust $\sum_i \log\left(1 + \exp(-y_i \mathbf{w}^\top\mathbf{x}_i)\right)$ für jedes $\mathbf{w}$ positiv ist, aber das Infimum null hat. Folgern Sie, dass $\|\mathbf{w}_t\|$ unbeschränkt wächst, wenn der Gradientenabstieg den Verlust gegen null treibt. Dann kann nur die Richtung $\mathbf{w}_t/\|\mathbf{w}_t\|$ konvergieren.
    1. Betrachten Sie ein zweischichtiges ReLU-Netz $f(\mathbf{x}) = \sum_j v_j \max(0, \mathbf{w}_j^\top\mathbf{x})$. Zeigen Sie, dass das Ersetzen von $(\mathbf{w}_j, v_j)$ durch $(c_j\mathbf{w}_j, v_j/c_j)$ mit $c_j > 0$ die Funktion $f$ unverändert lässt. Bestimmen Sie den kleinsten Wert von $\sum_j (\|\mathbf{w}_j\|^2 + v_j^2)$ über alle diese Umskalierungen. Erklären Sie, warum eine anhand der Parameternorm des Netzes gemessene Margin, anders als eine mit $\|\mathbf{w}\|$ beim linearen Prädiktor gemessene Margin, keine Eigenschaft der Funktion $f$ allein ist.
   :exlabel:`ex_generalization-deep-implicit-bias`

<!--en:17eab2c6-->
1. [extended] **Grokking.** ●● Stellen Sie das Experiment von :citet:`Power.Burda.Edwards.ea.2022` nach. Trainieren Sie ein kleines Netz darauf, $c = (a + b) \bmod 97$ aus dem Paar $(a, b)$ vorherzusagen. Die Autoren verwenden einen zweischichtigen Transformer, aber auch ein breites MLP mit One-Hot-kodierten Paaren eignet sich. Wählen Sie zufällig 50% aller Paare zum Training aus und trainieren Sie mit Gewichtszerfall für mindestens $10^5$ Schritte. Tragen Sie die Trainings- und Validierungsgenauigkeit gegen den Logarithmus der Schrittzahl auf und vergleichen Sie mit :numref:`fig_grokking`. Wie hängt die Zeit bis zur Generalisierung vom Anteil der Trainingsdaten und von der Stärke des Gewichtszerfalls ab?
   :exlabel:`ex_generalization-deep-grokking`

<!--en:d3ba6d68-->
[Diskussionen](https://d2l.discourse.group/t/7473)

<!--en:9329a875-->
Generalisierung im Deep Learning<br>**Auswendiglernen zufälliger Labels · doppelter Abstieg · impliziter Bias zugunsten kleinster Norm und maximaler Margin · Early Stopping und Grokking**

<!--en:55236420-->
::: {.slide title="Das Paradox der Überparametrisierung im Deep Learning"}

<!--en:553a1a5a-->
[Klassische und moderne Sicht]{.kicker}

<!--en:ee76a4d4-->
- **Klassische Schwelle ($p / n = 1.0$):** Ein Polynom 19. Grades ($p = 20$), das $n = 20$ verrauschte Punkte interpoliert, oszilliert stark (mittlerer quadratischer Testfehler $\geq 10^5$ gegenüber $0.01$ beim wahren kubischen Polynom).

<!--en:a290b741-->
- **Überparametrisierte MLPs ($p \gg n$):** Das MLP mit 256 Einheiten ($p = 203{,}530$) erreicht $\sim 87\%$ Testgenauigkeit bei $n = 60{,}000$ Trainingsbildern ($p/n = 3.4$) und $\approx 80\%$ Testgenauigkeit bei $n = 1{,}000$ Trainingsbildern ($p/n = 203.5$) bei einer Trainingsgenauigkeit von $100\%$.

<!--en:4194577f-->
- **Die offene Frage:** Warum generalisiert das Modell bei $p/n = 203.5$? Dazu sind seine Kapazität, die Kondition bei $h = n$ und der implizite Bias von SGD zu untersuchen.

<!--en:15f29f2f-->
::: {.slide title="Experiment 1: Auswendiglernen zufälliger und wahrer Labels"}

<!--en:7c5e26a4-->
[Auswendiglernen und Kapazität]{.kicker}

<!--en:38a596cb-->
Wie bei Zhang et al. (2017) wird das MLP mit He-Initialisierung und 256 Einheiten ($203{,}530$ Parameter) für $200$ Epochen auf $n = 1{,}000$ Fashion-MNIST-Bildern trainiert, einmal mit wahren und einmal mit gleichverteilt gezogenen Labels:

<!--en:45396116-->
- **Wahre Labels (`fraction = 0`):** Die Trainingsgenauigkeit erreicht innerhalb von 60 bis 90 Epochen $100\%$. Die Testgenauigkeit pendelt sich oberhalb von $80\%$ ein.

<!--en:d944e4ea-->
- **Zufällige Labels (`fraction = 1`):** Dasselbe Netz erreicht mit derselben SGD-Lernrate ebenfalls **$100\%$ Trainingsgenauigkeit** (in $\sim 120$–$150$ Epochen). Die Testgenauigkeit bleibt mit $10\%$ auf Zufallsniveau.

<!--en:3bc6fc35-->
::: {.slide title="Warum verteilungsfreie Kapazitätsschranken keine Aussagekraft haben"}

<!--en:d950441c-->
[Auswendiglernen und Kapazität · Theorie]{.kicker}

<!--en:551d9ac6-->
- **Die Parameteranzahl allein hilft wenig:** Mit der Näherung $v \approx p = 203{,}530$ für die VC-Dimension ergibt sich bei $n = 60{,}000$ der Wert $\sqrt{p/n} \approx 1.84 > 1$.

<!--en:ca0aa3db-->
- **Zufällige Labels entkräften *jede* klassenbezogene Schranke, die Labels außer Acht lässt:** $B(\mathcal{F}, n, \delta)$ hängt nur von der Funktionenklasse $\mathcal{F}$ und dem Stichprobenumfang $n$ ab. Erreicht das Modell $\epsilon_{\mathcal{S}}(f) = 0$ bei zufälligen Labels ($\epsilon(f) = 0.90$), muss daher auch bei wahren Labels $B(\mathcal{F}, n, \delta) \ge 0.90$ gelten.

<!--en:0c3fa671-->
::: {.slide title="No Free Lunch und die Rolle des induktiven Bias"}

<!--en:b7e2d2c7-->
[Induktiver Bias]{.kicker}

<!--en:16b4b38c-->
- **No Free Lunch (Wolpert, 1996):** Mittelt man gleichmäßig über alle möglichen Labelzuweisungen für ungesehene Eingaben, hat jeder Lernalgorithmus genau denselben erwarteten Testfehler.

<!--en:3fc66d3e-->
- **Interpolation $\neq$ Überanpassung:** Der 1-Nächster-Nachbar-Klassifikator erreicht $100\%$ Trainingsgenauigkeit. Dank der lokalen Glattheit in den Voronoi-Zellen ist sein asymptotischer Testfehler dennoch $\le 2\epsilon^\star$.

<!--en:0c91e187-->
- **Induktiver Bias im Deep Learning:** Bei überparametrisierten MLPs wählt SGD Interpolanten mit kleinster Norm bzw. maximaler Margin. Sie erfassen das zusammenhängende Signal vor dem Rauschen.

<!--en:7d6c645e-->
::: {.slide title="Von der klassischen U-Kurve aus Verzerrung und Varianz zur Überparametrisierung"}

<!--en:3103dfad-->
[Von der Unter- zur Überparametrisierung]{.kicker}

<!--en:5b849fbd-->
Im klassischen unterparametrisierten Bereich ($h < n$) sinkt der Trainingsfehler mit wachsender Kapazität. Der Testfehler beschreibt eine U-Kurve:

<!--en:05bb43c4-->
- **Unteranpassung (linker Ast):** Zu wenige Merkmale, um das wahre Signal zu erfassen

<!--en:0424cf8d-->
- **Überanpassung nahe $h = n$ (rechter Ast):** Reichen die Parameter gerade aus, um alle $n$ verrauschten Punkte zu interpolieren, sind starke Oszillationen zwischen den Trainingsbeispielen unvermeidlich.

<!--en:3a5469f7-->
- **Jenseits des dargestellten Bereichs:** Was passiert, wenn die Breite $h$ *über* $n$ *hinaus* wächst ($h \gg n$)? Dann gibt es unendlich viele fehlerfreie Interpolanten, von denen der mit der **kleinsten Norm** gewählt wird.

<!--en:dee8e5b3-->
::: {.slide title="Doppelter Abstieg bei kleinsten Quadraten mit zufälligen Merkmalen"}

<!--en:b009b6bf-->
[Doppelter Abstieg · Experiment]{.kicker}

<!--en:c567aad3-->
Die zufällige ReLU-Schicht $\mathbf{H} = \mathrm{ReLU}((\mathbf{X} - \bar{\mathbf{x}})\mathbf{W}^{(1)}) \in \mathbb{R}^{n \times h}$ bleibt nach der He-Initialisierung für $n = 1{,}000$ Bilder fest. Für die Ausgabegewichte $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times 10}$ liefert die Methode der kleinsten Quadrate die Lösung kleinster Norm (`np.linalg.lstsq`), jeweils für Breiten $h \in [50, 10{,}000]$:

<!--en:ca5a9411-->
- **Klassische U-Kurve ($h < 1{,}000$):** Der Testfehler sinkt auf $\sim 23\%$ bei $h = 200$ und steigt dann steil auf $\sim 90\%$, genau an der **Interpolationsschwelle** $h = n = 1{,}000$.

<!--en:2bd65166-->
- **Zweiter Abstieg ($h > 1{,}000$):** Jenseits von $h = n$ bleibt der Trainingsfehler bei $0$. Der Testfehler fällt auf $\sim 20\%$ bei $h = 10{,}000$ und liegt damit unter dem des besten Modells mit $h < n$.

<!--en:b8facbfc-->
::: {.slide title="Warum der Testfehler bei h = n steil ansteigt: schlechte Kondition"}

<!--en:80d3b2cd-->
[Doppelter Abstieg · Kondition]{.kicker}

<!--en:0f3bdc78-->
Der kleinste Singulärwert $s_{\min}(\mathbf{H})$ und die Norm der Gewichte $\|\mathbf{W}^{(2)}\|_{\textrm{F}}$ erklären sowohl den steilen Anstieg bei $h = n$ als auch den Rückgang bei $h \gg n$:

<!--en:bb639f5f-->
- **Bei $h = n = 1{,}000$:** $\mathbf{H}$ ist eine quadratische Matrix der Größe $1000 \times 1000$ mit eindeutiger exakter Lösung $\mathbf{W}^{(2)} = \mathbf{H}^{-1}\mathbf{Y}$. Die zufälligen Merkmale haben nahezu kollineare Richtungen ($s_{\min} \sim 10^{-4}\text{--}10^{-2}$). Beim Invertieren von $\mathbf{H}$ wird das Labelrauschen in der zugehörigen Richtung durch $s_{\min}$ geteilt. Dadurch wächst $\|\mathbf{W}^{(2)}\|_{\textrm{F}}$ stark an.

<!--en:9aeadd02-->
- **Bei $h = 10{,}000 \gg n$:** Zusätzliche Merkmale eröffnen einen $9{,}000$-dimensionalen affinen Unterraum exakter Interpolanten. Die Lösung kleinster Norm kann die Gewichte auf gut konditionierte Richtungen verteilen ($\|\mathbf{W}^{(2)}\|_{\textrm{F}} \approx 2.2\text{--}2.4$).

<!--en:29673f34-->
::: {.slide title="Wann zusätzliche Trainingsdaten die Testgenauigkeit senken"}

<!--en:7b7a8540-->
[Doppelter Abstieg · Stichprobenumfang]{.kicker}

<!--en:3d8e85c3-->
Entlang der Diagonalen $h = n$ ist das Problem schlecht konditioniert. Ein größerer Stichprobenumfang $n$ kann ein überparametrisiertes Modell bei *fester* Modellbreite $h = 1{,}000$ genau an die Interpolationsschwelle bringen:

<!--en:c6b4cb7e-->
Bei fester Breite $h = 1{,}000$ liegt das Modell mit $n = 500$ Trainingsbildern deutlich im überparametrisierten Bereich ($h = 2n$, Testfehler $\sim 28\text{--}30\%$). Verdoppelt sich der Datensatz auf $n = 1{,}000$ Bilder, erreicht das Modell genau die Schwelle $h = n$. Der Testfehler steigt auf $85\text{--}89\%$. Regularisierung durch einen $\ell_2$-Ridge-Strafterm dämpft $1/s_{\min}$ und beseitigt die Spitze.

<!--en:dc0cbaec-->
::: {.slide title="Impliziter Bias 1: Lösung kleinster Norm bei kleinsten Quadraten"}

<!--en:66eaebe6-->
[Impliziter Bias · Quadratischer Verlust]{.kicker}

<!--en:aa3f1bc5-->
Warum generalisiert die von `lstsq` bevorzugte Lösung kleinster Norm für $h \gg n$? **Der Gradientenabstieg vom Startwert $\mathbf{w}_0 = \mathbf{0}$ findet den Interpolanten kleinster Norm automatisch**, auch ohne explizite Regularisierung:

<!--en:6b37eda1-->
- **Invarianz des Zeilenraums:** Jeder Gradientenschritt ist eine Linearkombination der Zeilen von $\mathbf{H}$. Beim Start in $\mathbf{w}_0 = \mathbf{0}$ bleibt jede Iterierte $\mathbf{w}_t = \mathbf{H}^\top\mathbf{a}_t$ orthogonal zu $\mathrm{null}(\mathbf{H})$.

<!--en:f32160e9-->
- **Optimalität nach Pythagoras:** Bei Konvergenz ($\mathbf{H}\mathbf{w}^\star = \mathbf{y}$) gilt $\mathbf{w}^\star = \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{y}$. Jeder andere Interpolant $\mathbf{w}^\star + \mathbf{v}$ ($\mathbf{H}\mathbf{v} = \mathbf{0}$) hat eine strikt größere Norm. Ihr Quadrat beträgt $\|\mathbf{w}^\star\|_2^2 + \|\mathbf{v}\|_2^2$.

<!--en:fd37da4b-->
::: {.slide title="Numerische Überprüfung des impliziten Bias zur kleinsten Norm"}

<!--en:8b4ea807-->
[Impliziter Bias · Überprüfung]{.kicker}

<!--en:159f438f-->
$2{,}000$ Schritte des unregularisierten Gradientenabstiegs bei einem unterbestimmten $20 \times 100$-Gleichungssystem bestätigen die Rolle der Initialisierung $\mathbf{w}_0$:

<!--en:bc89d438-->
- **Start bei $\mathbf{w}_0 = \mathbf{0}$:** Der Gradientenabstieg konvergiert zur exakten Lösung kleinster Norm $\mathbf{w}^\star$ (`distance to w* = 0.0`, Norm unter $1$).

<!--en:e8a032cd-->
- **Zufallsinitialisierung $\mathbf{w}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:** Die Lösung interpoliert die Daten ebenfalls (`residual` in der Größenordnung des Rundungsfehlers). Da die anfängliche Komponente $\mathbf{v}_0 \in \mathrm{null}(\mathbf{H})$ im Kern erhalten bleibt, ist die Norm der Gewichte am Ende mehr als $10\times$ so groß.

<!--en:cf945634-->
::: {.slide title="Impliziter Bias 2: Maximum-Margin-Richtung bei der Klassifikation"}

<!--en:71c999ed-->
[Impliziter Bias · Klassifikation]{.kicker}

<!--en:5c3129a8-->
Welchen impliziten Bias hat der Gradientenabstieg bei der Klassifikation linear trennbarer Daten ($y_i \in \{-1, +1\}$) mit dem logistischen Verlust (Kreuzentropie) $\sum_i \log(1 + e^{-y_i \mathbf{w}^\top\mathbf{x}_i})$?

<!--en:04fe0a0e-->
- **Die Norm divergiert, die Richtung konvergiert:** Da $\log(1 + e^{-m}) > 0$ für alle endlichen Margins gilt, folgt $\|\mathbf{w}_t\|_2 \to \infty$. Dabei dreht sich $\mathbf{w}_t / \|\mathbf{w}_t\|_2$ in Richtung der Normalen der **Maximum-Margin-Trennebene** der Hard-Margin-SVM (Soudry et al., 2018).

<!--en:29a93567-->
- **Warum die Margins nach $100\%$ Trainingsgenauigkeit weiter wachsen:** Beispiele nahe der Entscheidungsgrenze haben die kleinsten Margins und dominieren den Gradienten $\propto e^{-y_i\mathbf{w}^\top\mathbf{x}_i}$. So sinkt der Testfehler weiter, lange nachdem der Trainingsfehler $0$ erreicht hat.

<!--en:9db6eafb-->
::: {.slide title="Zeitlicher Bias: SGD lernt unverfälschte Strukturen vor dem Rauschen"}

<!--en:a79ae95a-->
[Early Stopping · Dynamik]{.kicker}

<!--en:8444ac05-->
Bei verrauschten Datensätzen zeigt der Gradientenabstieg eine auffällige zeitliche Abfolge (Arpit et al., 2017). Nach zufälliger Neuzuweisung von $20\%$ der $1{,}000$ Fashion-MNIST-Labels ($\sim 18\%$ falsch) lässt sich die Genauigkeit getrennt für korrekt und falsch beschriftete Trainingsbeispiele sowie für unverfälschte Validierungsbilder verfolgen:

<!--en:8d9a2fe5-->
- **Epochen 1–20 (zuerst die unverfälschten Beispiele):** Gleichgerichtete Gradienten der unverfälschten Bilder heben deren Trainingsgenauigkeit über $80\%$. Das Netz hat weniger als $20\%$ der falschen Labels auswendig gelernt. Die Validierungsgenauigkeit erreicht hier ihren Höchstwert.

<!--en:77032a5d-->
- **Epochen 20–200 (Auswendiglernen):** Die Residuen der unverfälschten Beispiele schrumpfen. SGD passt das Modell zu $\sim 100\%$ an die widersprüchlichen falschen Labels an. Dadurch sinkt die Validierungsgenauigkeit um mehrere Prozentpunkte.

<!--en:f4790bfe-->
::: {.slide title="Early Stopping und der Ausnahmefall Grokking"}

<!--en:33b8ba30-->
[Early Stopping und Grokking]{.kicker}

<!--en:aecaebc9-->
Ein Trainingsabbruch beim Checkpoint mit der besten Validierungsgüte (*Early Stopping*) verhindert, dass das Netz in späten Epochen Rauschen auswendig lernt. Bei algorithmischen Aufgaben mit Gewichtszerfall kann das gegenteilige Phänomen auftreten (*Grokking*, Power et al., 2022):

<!--en:12d0dddf-->
- **Early Stopping als implizite Ridge-Regularisierung:** Bei linearen Problemen der kleinsten Quadrate wirkt ein Abbruch nach $t$ Schritten mit Schrittweite $\eta$ wie $\ell_2$-Gewichtszerfall mit $\lambda \approx 1/(\eta t)$.

<!--en:682d8142-->
- **Ausnahmefall Grokking (rechts):** Bei der modularen Addition $(a + b) \bmod 97$ mit Gewichtszerfall erreicht die Trainingsgenauigkeit $100\%$ innerhalb von $10^3$ Schritten. Die Validierungsgenauigkeit bleibt auf Zufallsniveau und springt dann auf $100\%$, und zwar nach $10^5$ Schritten.

<!--en:54ad670e-->
::: {.slide title="Verständnisfragen zu Kapazität, Kondition und implizitem Bias"}

<!--en:ffcb46a8-->
[Verständnisfragen]{.kicker}

<!--en:aad3af10-->
Fragen zur Generalisierung moderner Modelle:

<!--en:7244ade2-->
1. **Rechnen mit verfälschten Labels:** $20\%$ von $1{,}000$ Labels aus $10$ Klassen werden gleichverteilt neu gezogen. In Epoche 200 hat das Netz $100\%$ des verfälschten Trainingsdatensatzes auswendig gelernt. Welche Genauigkeit erreicht es gemessen an den *wahren*, unverfälschten Trainingslabels?

<!--en:5316a23b-->
2. **Gradientenabstieg mit Startwert ungleich null:** Gegeben ist die lineare Gleichung $w_1 + 2w_2 = 4$ ($\mathbf{h} = (1, 2)^\top$). Gegen welchen Punkt konvergiert der Gradientenabstieg beim Start in $\mathbf{w}_0 = (0, 0)$ bzw. $\mathbf{w}_0 = (1, 1)$?

<!--en:0a9321f8-->
3. **Die Spitze beim doppelten Abstieg beseitigen:** Warum beseitigt ein zusätzlicher $\ell_2$-Ridge-Strafterm $\lambda\mathbf{I}$ die Spitze des Testfehlers an der Interpolationsschwelle $h = n$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:343e5323-->
1. **$82\%$ Genauigkeit bei wahren Labels:** Mit Wahrscheinlichkeit $1/10$ entspricht ein neu gezogenes Label zufällig der wahren Klasse. Damit sind $0.20 \times (9/10) = 18\%$ der Trainingslabels tatsächlich falsch und auswendig gelernt.

<!--en:955deb99-->
2. **Projektion auf $w_1 + 2w_2 = 4$:** Vom Startpunkt $(0, 0)$ aus erreicht der Gradientenabstieg den Punkt kleinster Norm $\mathbf{w}^\star = \frac{4}{5}(1, 2) = (0.8, 1.6)$ (Norm $1.79$). Von $(1, 1)$ aus gelangt er entlang $\mathbf{h}$ zum Punkt $(1, 1) + \frac{4 - 3}{5}(1, 2) = (1.2, 1.4)$ (Norm $1.84$).

<!--en:6ef97a64-->
3. **Dämpfung bei $s_{\min}$:** Die Ridge-Regularisierung ersetzt die Inversion der Singulärwerte $1/s_i$ durch $s_i / (s_i^2 + n\lambda)$. Dadurch begrenzt sie die Verstärkung in Richtungen mit winzigen Singulärwerten $s_{\min} \approx 0$ bei $h = n$.

<!--en:30c7ea67-->
::: {.slide title="Zusammenfassung: Generalisierung im Deep Learning"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:a9db3c5f-->
- **Kapazität und doppelter Abstieg:** Das Auswendiglernen zufälliger Labels macht VC-Schranken ohne Labelbezug aussagelos. Der Testfehler steigt bei $h = n$ steil an ($s_{\min}(\mathbf{H}) \approx 0$) und fällt für $h \gg n$.

<!--en:b350eb56-->
- **Räumlicher impliziter Bias:** Der Gradientenabstieg von $\mathbf{w}_0 = \mathbf{0}$ aus findet bei quadratischem Verlust den Interpolanten mit kleinster $\ell_2$-Norm. Bei logistischem Verlust dreht sich der Gewichtsvektor in Richtung der Normalen der Maximum-Margin-Trennebene der SVM.

<!--en:bfa7a552-->
- **Zeitlicher impliziter Bias:** SGD passt das Modell in $\sim 20$ Epochen an zusammenhängende, unverfälschte Labels an, bevor es Rauschen auswendig lernt. Early Stopping wirkt dadurch wie eine Ridge-Regularisierung mit $\lambda \approx 1/(\eta t)$.
