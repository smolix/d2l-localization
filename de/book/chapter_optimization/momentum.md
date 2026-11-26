<!--en:69ea020b-->
# Momentum

<!--en:8604647f-->
In :numref:`sec_gd` haben wir gesehen, warum der einfache Gradientenabstieg (engl. gradient descent) in einem schlecht konditionierten Tal nur langsam vorankommt. Die stärkste Krümmung $\lambda_{\max}$ gibt die Stabilitätsgrenze $\eta < 2/\lambda_{\max}$ vor. In der flachsten Richtung $\lambda_{\min}$ schrumpft der Fehler daher pro Schritt höchstens um einen Anteil der Größenordnung $O(1/\kappa)$. Dabei ist $\kappa = \lambda_{\max}/\lambda_{\min}$ die Konditionszahl (engl. condition number). Der einfache Gradientenabstieg ist langsam, weil jeder Schritt allein vom aktuellen Gradienten $\mathbf{g}_t$ abhängt. Die letzten Gradienten verraten jedoch viel über die Geometrie des Tals. Entlang der Talsohle zeigen sie beständig in dieselbe Richtung. Quer zu den steilen Talwänden wechseln sie das Vorzeichen, weil die Iterierten abwechselnd zu beiden Seiten über die Talsohle hinausschießen.

<!--en:06950f0c-->
:citet:`Polyak.1964` schlug vor, die Schritte an einer exponentiell gewichteten Summe bisheriger Gradienten auszurichten, der **Geschwindigkeit** (engl. velocity). So verstärken sich gleichgerichtete Komponenten entlang flacher Richtungen, während sich wechselnde Komponenten in steilen Richtungen gegenseitig aufheben. Das **Heavy-Ball-Momentum** verhält sich wie eine schwere Kugel, die unter Reibung eine Rinne hinabrollt. Bei einer quadratischen Funktion mit schüsselförmigem Graphen sinkt die Zahl der nötigen Iterationen im ungünstigsten Fall von $O(\kappa)$ auf $O(\sqrt{\kappa})$. Dazu genügt es, für jeden Parameter einen Geschwindigkeitswert zu speichern.

<!--en:36e5beac-->
In diesem Abschnitt untersuchen wir die Rekursion für die Geschwindigkeit, die daraus folgende effektive Schrittweite (engl. effective step size) und die Zeitskala des Gedächtnisses. Wir zeigen, wie sich das $\sqrt{\kappa}$-Beschleunigungsgesetz aus der kritischen Dämpfung (engl. critical damping) ergibt. Anschließend testen wir Momentum bei der Softmax-Regression auf dem gesamten Trainingssatz und beim Training von `FashionCNN` mit Minibatches. Wir führen das vorausschauende Nesterov-Momentum ein und leiten praktische Regeln für die Wahl von $\beta$ und $\eta$ ab.

<!--en:dfc60fe6-->
## Das Momentum-Verfahren

<!--en:46752dd1-->
### Geschwindigkeit und effektive Schrittweite

<!--en:5b5be529-->
Sei $\mathbf{g}_t$ der Gradient im Schritt $t$, entweder exakt berechnet oder anhand eines Minibatchs geschätzt. Das Heavy-Ball-Momentum führt einen Geschwindigkeitsvektor $\mathbf{v}_t$ mit. Es initialisiert ihn mit $\mathbf{v}_0 = \mathbf{0}$ und aktualisiert Geschwindigkeit und Position wie folgt:

<!--en:e4811ab0-->
Dabei ist $\beta \in [0, 1)$ der Momentum-Koeffizient (für $\beta = 0$ erhalten wir den einfachen Gradientenabstieg). Durch wiederholtes Einsetzen erhalten wir die Geschwindigkeit als exponentiell gewichtete Summe aller bisherigen Gradienten:

<!--en:7f63dedf-->
Betrachten wir, wie diese gewichtete Summe auf die beiden Koordinaten des Tals $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ ($\lambda_1 = 0.2, \lambda_2 = 4$) wirkt. Mit $\eta = 0.6$ überschreiten wir die Stabilitätsgrenze $2/\lambda_2 = 0.5$ des einfachen Gradientenabstiegs. Dieser divergiert entlang von $x_2$ mit dem Faktor $1 - 4(0.6) = -1.4$ pro Schritt. Mit Momentum $\beta = 0.5$ wird die Trajektorie stabil. Beide Koordinaten nähern sich dem Ursprung bis auf $0.01$, und zwar in $20$ Schritten:

<!--en:ae08f7d7-->
Zwei Grenzfälle erklären, warum Momentum den Abstieg in der flachen Richtung beschleunigt und ihn zugleich in der steilen Richtung stabilisiert:

<!--en:6c72fa63-->
* **Gleichgerichtete Gradienten (flache Talsohle):** Bleiben aufeinanderfolgende Gradienten nahezu konstant ($\mathbf{g}_{t-k} \approx \mathbf{g}$), so wächst die Geschwindigkeit auf $\mathbf{v}_t \approx (1 + \beta + \beta^2 + \cdots)\mathbf{g} = \frac{1}{1-\beta}\mathbf{g}$. Daraus ergibt sich die **effektive Schrittweite**
  $$\eta_\textrm{eff} = \frac{\eta}{1-\beta}.$$

<!--en:cb56143c-->
* **Alternierende Gradienten (steile Talwände):** Wechseln aufeinanderfolgende Gradienten das Vorzeichen ($\mathbf{g}_{t-k} \approx (-1)^k \mathbf{g}$), ergibt die geometrische Reihe $1 - \beta + \beta^2 - \cdots = \frac{1}{1+\beta}$. Die effektive Schrittweite sinkt dadurch auf $\frac{\eta}{1+\beta}$.

<!--en:3436c174-->
:numref:`fig_opt_momentum_gain` zeigt beide Fälle für Gradienten vom Betrag eins.

<!--en:4da29d1d-->
![Geschwindigkeit $v_t = \beta v_{t-1} + g_t$ (blau) für Gradienten $g_t$ vom Betrag eins (graue Balken) bei $\beta = 0.5$. Links addieren sich Gradienten mit gleichem Vorzeichen, und $v_t$ nähert sich $1/(1-\beta) = 2$. Rechts heben sich Gradienten mit wechselndem Vorzeichen weitgehend auf, und $v_t$ schwingt sich auf die Amplitude $1/(1+\beta) = 2/3$ ein.](../img/mdl-opt-momentum-gain.svg)

<!--en:e013621c-->
Bei $\eta = 0.6$ und $\beta = 0.5$ erhöht Momentum die effektive Schrittweite entlang der Talsohle auf $\eta_{\text{eff}} = 0.6 / 0.5 = 1.2$. Zugleich dämpft es die Schwingungen quer zur Talwand, denn dort beträgt die effektive Schrittweite nur $0.6 / 1.5 = 0.4 < 0.5$. Für die Krümmung $\lambda$ steigt die Stabilitätsgrenze damit von $\eta\lambda < 2$ auf $\eta\lambda < 2(1 + \beta)$ (Übung :exref:`ex_momentum-stability`).

<!--en:4f710858-->
### Die Zeitskala von $\beta$ und das Minibatch-Rauschen

<!--en:97ad41ef-->
Die Gewichte $1, \beta, \beta^2, \ldots$ summieren sich zu $1/(1-\beta)$. Die Geschwindigkeit hat daher ein effektives Gedächtnisfenster (engl. memory window) von $\tau = 1/(1-\beta)$ Schritten und eine Halbwertszeit von $\log(1/2)/\log\beta$ Schritten:

<!--en:cec865dd-->
Erhöhen wir $\beta$ von $0.9$ auf $0.99$, verzehnfachen sich sowohl die Länge des Gedächtnisfensters als auch die effektive Schrittweite $\eta_{\text{eff}} = \eta/(1-\beta)$. Das Fenster wächst von $10$ auf $100$ Schritte.

<!--en:756f357f-->
Betrachten wir stochastische Gradienten $\mathbf{g}_t = \nabla f(\mathbf{x}_t) + \boldsymbol{\xi}_t$. Verkleinert die Mittelung über $1/(1-\beta)$ Schritte die stationäre Kugel um das Minimum (engl. noise ball) aus :numref:`sec_sgd`? Nein. Jede Störung $\boldsymbol{\xi}_t$ geht in $\mathbf{v}_t$ mit dem Gewicht $1$ ein. In den folgenden Schritten klingt ihr Gewicht wie $\beta^k$ ab. Insgesamt trägt sie damit eine Verschiebung von $\eta(1 + \beta + \beta^2 + \cdots)\boldsymbol{\xi}_t = \eta_{\text{eff}}\boldsymbol{\xi}_t$ bei. Bei gleicher effektiver Schrittweite $\eta_{\text{eff}} = \eta/(1-\beta)$ erreicht der stochastische Gradientenabstieg (engl. stochastic gradient descent, SGD) mit und ohne Momentum einen stationären Zustand. In beiden Fällen streuen die Iterierten mit nahezu gleicher Varianz um das Minimum (Übung :exref:`ex_momentum-momentum-under-minibatch-noise`). Erhöhen wir $\beta$ in einem rauschbegrenzten Lauf, sollten wir daher $\eta$ proportional zu $1 - \beta$ wählen, damit $\eta_{\text{eff}}$ konstant bleibt.

<!--en:3834eb8f-->
## Beschleunigung und kritische Dämpfung

<!--en:7ccb4310-->
Welche Wahl von $(\eta, \beta)$ ist für eine quadratische Funktion mit den Eigenwerten $\lambda_{\min} \le \cdots \le \lambda_{\max}$ und der Konditionszahl $\kappa = \lambda_{\max}/\lambda_{\min}$ optimal? Nach dem Beweis von :citet:`Polyak.1964` liefern die Parameter

<!--en:e5dcb3f6-->
im ungünstigsten Fall pro Schritt den Kontraktionsfaktor

<!--en:fa1b1dbe-->
Damit sinkt die Zahl der Schritte bis zur relativen Genauigkeit $\epsilon$ von $\frac{\kappa}{2}\log(1/\epsilon)$ auf $\frac{\sqrt{\kappa}}{2}\log(1/\epsilon)$ (:numref:`subsec_mdl-momentum-acceleration`).

<!--en:997bd433-->
Warum ist der Kontraktionsfaktor gleich $\sqrt{\beta^\star}$? Setzen wir $\eta v_t = x_t - x_{t+1}$ in die eindimensionale Geschwindigkeitsgleichung entlang eines Eigenvektors mit Krümmung $\lambda$ ein. So erhalten wir die lineare Rekurrenz zweiter Ordnung

<!--en:c97db829-->
mit der charakteristischen Gleichung $z^2 - (1 + \beta - \eta\lambda)z + \beta = 0$. Nach dem Satz von Vieta gilt für das Produkt der beiden Wurzeln $z_1 z_2 = \beta$. Die Diskriminante $\Delta = (1 + \beta - \eta\lambda)^2 - 4\beta$ ist für alle $\beta \ge \beta_{\text{c}}(\lambda) = (1 - \sqrt{\eta\lambda})^2$ kleiner oder gleich null. Die Wurzeln sind dann komplex konjugiert und haben denselben Betrag $|z_1| = |z_2| = \sqrt{\beta}$, unabhängig von der Krümmung $\lambda$ (:numref:`fig_opt_critical_damping`).

<!--en:4b4a7fe1-->
![Kontraktionsfaktor pro Schritt beim Heavy-Ball-Momentum in einer Richtung mit Krümmung $\lambda$ als Funktion von $\beta$. Am schnellsten ist die Konvergenz an der Schwelle zur kritischen Dämpfung $\beta_\textrm{c} = (1 - \sqrt{\eta\lambda})^2$. Unterhalb von $\beta_\textrm{c}$ ist die Eigenmode überdämpft. Oberhalb von $\beta_\textrm{c}$ schwingt sie mit einem Abklingfaktor vom Betrag $\sqrt{\beta}$.](../img/mdl-opt-critical-damping.svg)

<!--en:952b3f5a-->
Mit den optimalen Parametern aus :eqref:`eq_momentum-optimal` fallen die kritischen Schwellen für die flachste und die steilste Richtung zusammen: $(1 - \sqrt{\eta^\star\lambda_{\min}})^2 = (\sqrt{\eta^\star\lambda_{\max}} - 1)^2 = \beta^\star$. Beide extremen Eigenmoden sind damit kritisch gedämpft. Zu jedem dazwischenliegenden Eigenwert $\lambda \in [\lambda_{\min}, \lambda_{\max}]$ gehört eine unterdämpfte Eigenmode mit komplexen Wurzeln und dem Kontraktionsfaktor $\sqrt{\beta^\star}$.

<!--en:204174d9-->
Für unser Tal ($\kappa = 20$) liefert :eqref:`eq_momentum-optimal` die Parameter $\eta^\star \approx 0.668, \beta^\star \approx 0.403$. Der asymptotische Kontraktionsfaktor sinkt von $0.905$ auf $\sqrt{\beta^\star} \approx 0.635$. Bis zu einem relativen Fehler von $10^{-6}$ brauchen wir statt $139$ nur $40$ Schritte:

<!--en:9245e22c-->
Betrachten wir die vierte Zeile ($\eta = 0.3, \beta = 0.8$). Hier liegt $\beta$ weit über der kritischen Schwelle $\beta_{\text{c}}$. Beide Koordinaten sind unterdämpft, der Betrag der Wurzeln ist $\sqrt{0.8} \approx 0.894$. Die Trajektorie schießt über den Ursprung hinaus und schwingt um ihn:

<!--en:0981e9c1-->
## Momentum beim Training mit Daten

<!--en:2c4af4d7-->
Unsere Analyse sagt für das Training mit Daten zwei unterschiedliche Fälle voraus:

<!--en:72ed1371-->
1. **Krümmungsbegrenztes Regime (genaue Gradienten, großes $\kappa$):** Momentum erlaubt eine größere effektive Schrittweite $\eta_{\text{eff}}$ und senkt die Zahl der nötigen Iterationen um einen großen Faktor.

<!--en:06afa5d5-->
2. **Rauschbegrenztes Regime (kleine Minibatches):** Die stationäre Rauschgrenze hängt von $\eta_{\text{eff}} = \eta/(1-\beta)$ ab. Mit $\beta = 0.9$ sinkt die optimale Lernrate auf das $1 - \beta = 0.1$-Fache, während die Konvergenzgeschwindigkeit vergleichbar bleibt.

<!--en:d73ceef7-->
Wir überprüfen beide Fälle auf Fashion-MNIST und vergleichen dazu $\beta = 0$ mit $\beta = 0.9$. Die beiden Versuche sind (1) die Softmax-Regression auf dem gesamten Trainingssatz ($b = 60{,}000$, exakte Gradienten, schlechte Konditionierung aufgrund der nichtnegativen Pixelintensitäten) und (2) das Training von `FashionCNN` (:numref:`subsec_fashioncnn`) mit kleinen Minibatches ($b = 32$). Zunächst laden wir die normalisierten Fashion-MNIST-Tensoren auf den Beschleuniger. Dort hält der Datenlader `Batches` sie im Speicher:

<!--en:62d3802f-->
Als Nächstes definieren wir das mit Nullen initialisierte lineare Modell `softmax_regression` und den SGD-Konstruktor `make_sgd(beta)`, der den Momentum-Koeffizienten als Argument erhält:

<!--en:e63032a0-->
Mit `d2l.lr_sweep` durchsuchen wir nun die Lernratenraster `grids` für beide Modelle bei $\beta \in \{0.0, 0.9\}$. Für jede Konfiguration halten wir die beste Lernrate, die Zahl der Schritte bis zum Zielverlust und den Verlust am Ende des Trainings fest:

<!--en:6f291a4e-->
Beide Vorhersagen bestätigen sich auch zahlenmäßig:

<!--en:a42eefb1-->
* Bei der **Softmax-Regression** auf dem gesamten Trainingssatz erhöht Momentum mit $\beta = 0.9$ die optimale Lernrate von $\eta = 0.125$ auf $\eta = 0.4$. Damit beträgt die effektive Schrittweite $\eta_{\text{eff}} = 4.0$ und ist $32\times$ so groß wie zuvor. Bis zu einem Verlust von $0.5$ sinkt die Schrittzahl um einen Faktor von mehr als $6\times$ ($625 \to 100$ Schritte). Der Endverlust sinkt von $0.472$ auf $0.365$.

<!--en:25d3708d-->
* Bei **`FashionCNN`** mit kleinen Minibatches ($b = 32$) begrenzt das Rauschen den Trainingsfortschritt. Die optimale Lernrate liegt im Bereich $\eta \approx 0.1$–$0.2$ bei $\beta = 0$ und sinkt auf $\eta \approx 0.02$–$0.04$ bei $\beta = 0.9$. Das entspricht einem Faktor von $5$ bis $10$. Im Raster mit jeweils verdoppelter Lernrate weicht die effektive Schrittweite $\eta_{\text{eff}}$ damit höchstens um eine Rasterstufe von ihrem Wert bei $\beta = 0$ ab. Beide Konfigurationen erreichen einen Verlust von $0.3$ und Endverluste von $0.26$–$0.30$. Die Unterschiede liegen innerhalb der in :numref:`sec_test_models` beschriebenen Schwankungsbreite für einen einzelnen Startwert des Zufallsgenerators.

<!--en:b493bcd5-->
## Nesterov-Momentum

<!--en:768cf354-->
Im unterdämpften Lauf ($\eta = 0.3, \beta = 0.8$) schoss das Heavy-Ball-Momentum über das Minimum hinaus. Es berechnete den Gradienten an der aktuellen Position $\mathbf{x}_t$, obwohl die Geschwindigkeit den nächsten Schritt bereits vorgab. :citet:`Nesterov.1983` schlug vor, den Gradienten am **vorausberechneten Punkt** (engl. look-ahead point) $\mathbf{x}_t - \eta\beta\mathbf{v}_{t-1}$ auszuwerten (:numref:`fig_opt_nesterov_lookahead`):

<!--en:65de7c87-->
![Je ein Schritt mit Heavy-Ball-Momentum und Nesterov-Momentum bei gleicher Ausgangsposition und Geschwindigkeit. Das Heavy-Ball-Verfahren berechnet den Gradienten bei $\mathbf{x}_t$ und schießt über das Ziel hinaus. Das Nesterov-Verfahren berechnet den Gradienten am vorausberechneten Punkt $\mathbf{x}_t - \eta\beta\mathbf{v}_{t-1}$ und korrigiert die Bewegung unmittelbar in Gegenrichtung.](../img/mdl-opt-nesterov-lookahead.svg)

<!--en:814d3202-->
Die Auswertung des Gradienten am vorausberechneten Punkt dämpft die Bewegung vorausschauend und unterdrückt Schwingungen in steilen Richtungen. Auch die Konvergenzgarantien reichen weiter als beim Heavy-Ball-Momentum. Mit dem konstanten Koeffizienten $\beta = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ erreicht das Nesterov-Momentum die $\sqrt{\kappa}$-Rate für jede glatte, stark konvexe Funktion, also auch über den quadratischen Fall hinaus. Mit der ansteigenden Folge $\beta_t = (t-1)/(t+2)$ erreicht es die optimale Konvergenzrate $O(1/t^2)$ für allgemeine glatte konvexe Zielfunktionen :cite:`Nesterov.2018` (:numref:`subsec_mdl-momentum-acceleration`). Hier halten wir $\beta$ konstant:

<!--en:db0ef59f-->
In der Praxis ist $\beta = 0.9$ für die meisten Architekturen ein robuster Standardwert. Die Lernrate stimmen wir über die effektive Schrittweite $\eta_{\text{eff}} = \eta/(1-\beta)$ ab. Momentum verwendet jedoch für alle Koordinaten dieselbe skalare Schrittweite. Unterscheiden sich die Gradienten wie bei Transformern von Schicht zu Schicht um Größenordnungen, brauchen wir eine adaptive Skalierung für jede einzelne Koordinate (:numref:`sec_adam`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:c05f895f-->
Das Heavy-Ball-Momentum aktualisiert die Parameter anhand der Geschwindigkeit $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \mathbf{g}_t$. Darin klingen frühere Gradienten exponentiell ab, mit einem Gedächtnisfenster von $1/(1-\beta)$ Schritten. Bei gleichgerichteten Gradienten steigt die effektive Schrittweite auf $\eta_{\text{eff}} = \eta/(1-\beta)$. Bei wechselndem Vorzeichen sinkt sie auf $\eta/(1+\beta)$. Damit steigt die Stabilitätsgrenze auf $\eta\lambda < 2(1+\beta)$. Bei einer schüsselförmigen quadratischen Funktion mit Konditionszahl $\kappa$ dämpft die Wahl von $(\eta^\star, \beta^\star)$ die beiden extremen Eigenmoden kritisch. Die Zahl der benötigten Iterationen sinkt von $O(\kappa)$ auf $O(\sqrt{\kappa})$. Das Nesterov-Momentum wertet den Gradienten am vorausberechneten Punkt $\mathbf{x}_t - \eta\beta\mathbf{v}_{t-1}$ aus. Diese vorausschauende Dämpfung unterdrückt das Nachschwingen bei Unterdämpfung und erweitert die Beschleunigung auf allgemeine glatte konvexe Funktionen. Bei Minibatch-Rauschen bleibt die stationäre Kugel um das Minimum mit Momentum gleich groß, sofern die effektive Schrittweite $\eta_{\text{eff}} = \eta/(1-\beta)$ gleich bleibt. Bei krümmungsbegrenztem Training ermöglicht Momentum eine starke Beschleunigung. Bei rauschbegrenztem Training muss $\eta \propto 1 - \beta$ gelten.

<!--en:2294243a-->
## Übungen

<!--en:7c83754d-->
1. [code] **Raster für Momentum und Lernrate.** Führen Sie `momentum_2d` auf `f_valley` für alle Kombinationen von $\beta \in \{0, 0.25, 0.5, 0.8\}$ und $\eta \in \{0.1, 0.3, 0.6, 1.0\}$ mit jeweils bis zu $1{,}000$ Schritten aus. Halten Sie in einer Tabelle fest, wie viele Schritte bis $\|\mathbf{x}_t\| \le 10^{-3}$ nötig sind oder wann das Verfahren divergiert. Vergleichen Sie das empirische Optimum für jedes $\eta$ mit $\beta^\star$ aus :eqref:`eq_momentum-optimal` und mit $\beta_{\text{c}}(\lambda) = (1 - \sqrt{\eta\lambda})^2$.
   :exlabel:`ex_momentum-momentum-learning-rate-together`

<!--en:41e3027e-->
1. **Verhältnis von Rauschen zu Signal in der Geschwindigkeit.** Betrachten Sie $\mathbf{g}_t = \mathbf{g} + \boldsymbol{\xi}_t$ mit unabhängig und identisch verteiltem Rauschen, Mittelwert null und Varianz $\sigma^2$ pro Koordinate. Zeigen Sie, dass $\mathbf{v}_t$ aus :eqref:`eq_momentum` den stationären Mittelwert $\mathbf{g}/(1-\beta)$ und die stationäre Varianz $\sigma^2/(1-\beta^2)$ hat. Damit sinkt das Verhältnis von Rauschen zu Signal in der Geschwindigkeit auf das $\sqrt{(1-\beta)/(1+\beta)}$-Fache.
   :exlabel:`ex_momentum-velocity-noise`

<!--en:1e1a713e-->
1. **Äquivalente Formen des Nesterov-Momentums und Stabilität.**
    1. Zeigen Sie durch Induktion, dass die in Bibliotheken verwendete Form $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \nabla f(\mathbf{y}_t)$, $\mathbf{y}_{t+1} = \mathbf{y}_t - \eta(\nabla f(\mathbf{y}_t) + \beta\mathbf{v}_t)$ genau dieselben Geschwindigkeiten liefert wie :eqref:`eq_nesterov` mit $\mathbf{y}_t = \mathbf{x}_t - \eta\beta\mathbf{v}_{t-1}$.
       :exlabel:`ex_momentum-two-forms-nesterov-momentum`
    1. Beweisen Sie anhand von :eqref:`eq_momentum-recurrence`, dass beide Wurzeln von $z^2 - (1 + \beta - \eta\lambda)z + \beta = 0$ für $0 \le \beta < 1$ genau dann strikt innerhalb des Einheitskreises liegen, wenn $0 < \eta\lambda < 2(1+\beta)$ gilt.
       :exlabel:`ex_momentum-stability`

<!--en:9dc3878e-->
1. **Momentum mit exponentieller Mittelung.** Vergleichen Sie :eqref:`eq_momentum` mit der normierten Form des exponentiell gleitenden Mittelwerts (EMA) $\mathbf{m}_t = \beta\mathbf{m}_{t-1} + (1-\beta)\mathbf{g}_t$, $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha\mathbf{m}_t$, die Adam verwendet (:numref:`sec_adam`). Zeigen Sie, dass $\mathbf{m}_t = (1-\beta)\mathbf{v}_t$ gilt und die beiden Trajektorien daher für $\alpha = \eta/(1-\beta) = \eta_{\text{eff}}$ übereinstimmen.
   :exlabel:`ex_momentum-averaged-form`

<!--en:7487ec2a-->
1. [code] **Momentum bei Minibatch-Rauschen und verschiedenen Batchgrößen.** ●
    1. Betrachten Sie $f(x) = \frac{\lambda}{2}x^2$ mit dem verrauschten Gradienten $g_t = \lambda x_t + \xi_t$. Vergleichen Sie die stationäre Varianz der Iterierten beim einfachen SGD mit Schrittweite $\alpha$ und beim SGD mit Momentum ($\beta = 0.9$) und Schrittweite $\eta = (1-\beta)\alpha$ für $\alpha\lambda \in \{0.05, 0.2\}$.
    1. Trainieren Sie `d2l.FashionCNN` für $1{,}000$ Schritte mit den Batchgrößen $b \in \{32, 500, 6000\}$ und den Momentum-Koeffizienten $\beta \in \{0, 0.9\}$ bei gleicher effektiver Schrittweite $\eta/(1-\beta) = 0.2$. Variieren Sie anschließend $\eta$ bei $b = 6000$. Überprüfen Sie so, dass Momentum das Training beschleunigt, sobald der Batch groß genug für das krümmungsbegrenzte Regime ist.
   :exlabel:`ex_momentum-momentum-under-minibatch-noise`

<!--en:e6b11b68-->
[Diskussion](https://d2l.discourse.group/t/1070)

<!--en:3183d2d1-->
[Diskussion](https://d2l.discourse.group/t/1071)

<!--en:93649e9b-->
Momentum und Nesterovs beschleunigter Gradientenabstieg<br>**Geschwindigkeitsrekursion · effektive Schrittweite · Zeitskala des Gedächtnisses · kritische Dämpfung und √κ-Beschleunigung · Nesterovs vorausschauender Schritt**

<!--en:67a6068e-->
::: {.slide title="Warum Gradientenabstieg ohne Gedächtnis in Tälern langsam ist"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:f8aa5735-->
- **Ohne Gedächtnis sind $O(\kappa)$ Schritte nötig:** Für $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$ ($\kappa = 20$) braucht der einfache Gradientenabstieg $47$ Schritte bis zu einem Fehler von $1\%$ ($139$ Schritte bis $10^{-6}$). Er nutzt die Ausrichtung aufeinanderfolgender Gradienten nicht:
  - **Entlang der flachen Talsohle ($x_1$):** Gradienten zeigen **beständig in dieselbe Richtung**.
  - **Quer zu den steilen Talwänden ($x_2$):** Gradienten **wechseln bei jedem Schritt das Vorzeichen**.

<!--en:f6ba3323-->
- **Polyaks Idee der schweren Kugel (1964):** In der Geschwindigkeitssumme klingen ältere Beiträge exponentiell ab. Entlang der Talsohle verstärken sich die Gradienten, quer zu den Talwänden heben sich die Schwingungen auf.

<!--en:cbe9da60-->
::: {.slide title="Heavy-Ball-Momentum: Rekursion für die Geschwindigkeit"}

<!--en:90578f94-->
[Aktualisierungsgleichungen]{.kicker}

<!--en:6eb13c6c-->
- **Aktualisierung der Geschwindigkeit ($\beta \in [0, 1)$, $\mathbf{v}_0 = \mathbf{0}$):**
  $$\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + \mathbf{g}_t, \qquad \mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\mathbf{v}_t$$

<!--en:e57229d9-->
- **Bisherige Gradienten als geometrisch gewichtete Summe:**
  $$\mathbf{v}_t = \sum_{k=0}^{t-1} \beta^k \mathbf{g}_{t-k}$$

<!--en:6b6f9187-->
- **Physikalische Analogie:** Eine schwere Kugel rollt auf $f(\mathbf{x})$ bergab, mit Trägheit $\beta$, viskoser Reibung $1 - \beta$ und Gedächtnisfenster $\tau = \frac{1}{1-\beta}$.

<!--en:c0c59793-->
::: {.slide title="Momentum stabilisiert den Abstieg bei η = 0.6"}

<!--en:06892055-->
[Trajektorie im Tal]{.kicker}

<!--en:2751cbb9-->
- Bei $\eta = 0.6 > 2/\lambda_2 = 0.5$ divergiert der einfache Gradientenabstieg quer zur Talwand rasch. Die Koordinate $x_2$ ändert sich bei jedem Schritt um den Faktor $1 - 4(0.6) = -1.4$ ($1.4^{20} \approx 837$).

<!--en:c4df96af-->
- Heavy-Ball-Momentum mit **$\beta = 0.5$** dämpft bei *genau derselben Lernrate* $\eta = 0.6$ die Schwingungen quer zur Talwand. Beide Koordinaten erreichen einen Abstand von höchstens $1\%$ ihres jeweiligen Anfangsabstands zum Ursprung, und zwar in **$20$ Schritten**.

<!--en:28f1efb2-->
- **Scheinbarer Widerspruch:** Die Summe bisheriger Gradienten vergrößert die Schrittweite entlang der Talsohle. Wie kann sie zugleich den Abstieg an der steilen Wand *stabilisieren*?

<!--en:7f9bd9ad-->
::: {.slide title="Verstärkung bei gleichem Vorzeichen, Auslöschung bei Vorzeichenwechsel"}

<!--en:40756f41-->
[Die effektive Schrittweite]{.kicker}

<!--en:3e80f0a4-->
- **Gleichgerichtete Gradienten entlang der Talsohle ($\mathbf{g}_{t-k} \approx \mathbf{g}$):** Die geometrische Summe $\sum_{k \ge 0}\beta^k = \frac{1}{1-\beta}$ ergibt die effektive Schrittweite **$\eta_{\text{eff}} = \frac{\eta}{1-\beta}$** ($1.2$ bei $\eta=0.6, \beta=0.5$).

<!--en:69850254-->
- **Alternierende Gradienten quer zur Talwand ($\mathbf{g}_{t-k} \approx (-1)^k\mathbf{g}$):** Die alternierende Summe $\sum_{k \ge 0}(-\beta)^k = \frac{1}{1+\beta}$ dämpft die Schrittweite auf **$\eta_{\text{alt}} = \frac{\eta}{1+\beta}$** ($0.4 < 0.5$, stabil).

<!--en:fe9c9156-->
- **Höhere Stabilitätsgrenze:** Nach $\eta\lambda_{\max} < 2(1+\beta)$ dürfen die Schritte entlang der Talsohle um den Faktor $\frac{1+\beta}{1-\beta}$ größer ausfallen als beim einfachen Gradientenabstieg.

<!--en:5edae217-->
::: {.slide title="Wie Momentum die Stabilitätsgrenze anhebt"}

<!--en:d67b5088-->
[Stabilität und maximale Beschleunigung]{.kicker}

<!--en:b5beaac6-->
Für eine Eigenmode mit wechselndem Vorzeichen $x_t \propto (-1)^t$ gilt die gedämpfte Schrittweite $\frac{\eta}{1+\beta}$. Nach dem Stabilitätskriterium $\frac{\eta}{1+\beta}\lambda < 2$ steigt die zulässige Schrittweite um den Faktor $1 + \beta$:

<!--en:e9fabbcd-->
| Optimierer | Schrittweite bei Vorzeichenwechsel | Stabilitätsgrenze bei Krümmung $\lambda$ | Grenze im Tal ($\lambda_{\max} = 4$) | Maximale stabile Schrittweite $\eta_{\text{eff}}$ entlang der Talsohle |

<!--en:5e33461b-->
| **Einfacher Gradientenabstieg ($\beta = 0$)** | $\eta$ | $\eta\lambda < 2$ | $\eta < 0.50$ | $0.50$ |

<!--en:a3595806-->
| **Heavy-Ball-Momentum ($\beta$)** | $\frac{\eta}{1 + \beta}$ | **$\eta\lambda < 2(1 + \beta)$** | $\eta < 0.75$ (bei $\beta = 0.5$) | **$\frac{2(1+\beta)}{(1-\beta)\lambda_{\max}} = 1.50$** (bei $\beta = 0.5$) |

<!--en:38ec5240-->
- Bei $\beta = 0.5$ kann die effektive Schrittweite entlang der Talsohle mit Momentum bis zu $\frac{1+\beta}{1-\beta} = 3\times$ so groß sein wie die maximale stabile Schrittweite des einfachen Gradientenabstiegs.

<!--en:eb90c4db-->
::: {.slide title="Gedächtnisfenster und Halbwertszeit bei β"}

<!--en:69562191-->
[Exponentiell abklingende Gewichte]{.kicker}

<!--en:63e6d8af-->
- Die geometrischen Gewichte $1, \beta, \beta^2, \ldots$ summieren sich zu $\frac{1}{1-\beta}$. Die Geschwindigkeit mittelt die Gradienten daher über ein charakteristisches **Gedächtnisfenster**:
  $$\tau = \frac{1}{1 - \beta} \text{ steps}$$

<!--en:a0ba2b60-->
- **Halbwertszeit:** Das Gewicht $\beta^k$ eines Gradienten fällt auf $0.5$, sobald $k_{1/2} = \frac{\ln 0.5}{\ln\beta} \approx 0.693\,\tau$ Schritte vergangen sind. Bei den betrachteten Koeffizienten entfallen auf die jüngsten $\tau$ Schritte jeweils $1 - \beta^{1/(1-\beta)} \approx 1 - e^{-1} \approx 63\text{--}75\%$ des Gesamtgewichts.

<!--en:1aa8dffe-->
- Eine Erhöhung von $\beta$ gemäß $0.9 \to 0.99$ **verzehnfacht** sowohl $\tau$ als auch $\eta_{\text{eff}}$ ($10 \to 100$ Schritte).

<!--en:02b8cece-->
::: {.slide title="Momentum bei Minibatch-Rauschen: gleiche Rauschgrenze"}

<!--en:9ba355d1-->
[Zusammenspiel mit stochastischem Rauschen]{.kicker}

<!--en:600f704d-->
- **Gesamtwirkung eines einzelnen Rauschimpulses $\boldsymbol{\xi}_t$:** Er wirkt in $\mathbf{v}_{t+k}$ mit dem Gewicht $\beta^k$ nach. Über alle Schritte ergibt sich die Verschiebung
  $$-\eta\,(1 + \beta + \beta^2 + \cdots)\,\boldsymbol{\xi}_t = -\frac{\eta}{1 - \beta}\,\boldsymbol{\xi}_t = -\eta_{\text{eff}}\,\boldsymbol{\xi}_t$$

<!--en:bd79a7bc-->
- **Keine kleinere Kugel um das Minimum bei gleichem $\eta_{\text{eff}}$:** Momentum verteilt jeden Rauschimpuls auf $\tau = \frac{1}{1-\beta}$ Schritte. Die stationäre Varianz entspricht der des einfachen SGD mit Schrittweite $\eta_{\text{eff}}$.

<!--en:4b82e2b8-->
- **Faustregel:** Beim Zuschalten von $\beta = 0.9$ in einem rauschbegrenzten Lauf die Lernrate $\eta$ mit $1 - \beta = 0.1$ multiplizieren.

<!--en:f346caa3-->
::: {.slide title="Optimale Beschleunigung für quadratische Funktionen: von O(κ) auf O(√κ)"}

<!--en:b8b673ae-->
[Polyaks Beschleunigungsgesetz]{.kicker}

<!--en:e23f921f-->
- **Polyaks optimale Parameterwahl (1964):**
  $$\eta^\star = \frac{4}{\left(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}}\right)^2}, \qquad \beta^\star = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2$$

<!--en:f4d030ad-->
- **Von $O(\kappa)$ auf $O(\sqrt{\kappa})$ Schritte:** Der Kontraktionsfaktor sinkt von $\frac{\kappa-1}{\kappa+1} \approx 1 - \frac{2}{\kappa}$ (einfacher Gradientenabstieg) auf **$\sqrt{\beta^\star} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \approx 1 - \frac{2}{\sqrt{\kappa}}$** (Schrittzahl um den Faktor $37\times$ kleiner bei $\kappa = 1{,}400$).

<!--en:6a7365e2-->
- **Warum $\sqrt{\beta}$?** Für $\beta \ge \beta_{\text{c}}(\lambda) = (1-\sqrt{\eta\lambda})^2$ sind die charakteristischen Wurzeln $z_1, z_2$ komplex konjugiert. Daher gilt $z_1 z_2 = \beta \implies |z| = \sqrt{\beta}$.

<!--en:235f7ca8-->
::: {.slide title="Rekurrenz zweiter Ordnung und Betrag √β der komplexen Wurzeln"}

<!--en:8a09bacf-->
[Warum der Kontraktionsfaktor √β beträgt]{.kicker}

<!--en:b5780800-->
Warum fallen *alle* dazwischenliegenden Eigenmoden mit genau demselben Faktor $\sqrt{\beta^\star}$ ab, unabhängig von ihrer Krümmung $\lambda$?

<!--en:72a44218-->
- Entlang eines Eigenvektors mit Krümmung $\lambda$ ($g_t = \lambda x_t$) ergibt das Einsetzen von $\eta v_{t-1} = x_{t-1} - x_t$ in $x_{t+1} = x_t - \eta(\beta v_{t-1} + \lambda x_t)$ eine lineare Rekurrenz zweiter Ordnung:
  $$x_{t+1} - (1 + \beta - \eta\lambda)\,x_t + \beta\,x_{t-1} = 0$$

<!--en:ca33077d-->
- **Charakteristisches Polynom:** Der Ansatz $x_t = z^t$ liefert $z^2 - (1 + \beta - \eta\lambda)z + \beta = 0$. Die beiden Wurzeln erfüllen die Produktformel von Vieta **$z_1 z_2 = \beta$**.

<!--en:28a82da9-->
- **Komplex konjugierte Wurzeln:** Für die Diskriminante gilt $\Delta = (1 + \beta - \eta\lambda)^2 - 4\beta \le 0$, sobald $\beta \ge \beta_{\text{c}}(\lambda) = (1 - \sqrt{\eta\lambda})^2$. Dann sind die Wurzeln $z_1, z_2 = z_1^*$ komplex konjugiert und haben denselben Betrag:
  $$|z_1| = |z_2| = \sqrt{z_1 z_2} = \sqrt{\beta}$$

<!--en:e212d813-->
::: {.slide title="Überdämpfte, kritisch gedämpfte und unterdämpfte Eigenmoden"}

<!--en:688a5219-->
[Geometrie der kritischen Dämpfung]{.kicker}

<!--en:17ef1ef4-->
- **Drei Dämpfungsfälle für eine Eigenmode mit Krümmung $\lambda$:**
  - **Überdämpft ($\beta < \beta_{\text{c}}$):** Zwei verschiedene reelle Wurzeln. Der Beitrag der betragsgrößeren Wurzel klingt langsam gegen null ab.
  - **Kritisch gedämpft ($\beta = \beta_{\text{c}} = (1 - \sqrt{\eta\lambda})^2$):** Die Wurzeln fallen zusammen (für die flache Richtung bei $z_1 = z_2 = \sqrt{\beta_{\text{c}}}$). Die Eigenmode kontrahiert so schnell wie möglich.
  - **Unterdämpft ($\beta > \beta_{\text{c}}$):** Die komplexen Wurzeln führen zu Schwingungen. Deren Einhüllende klingt mit dem Faktor $\sqrt{\beta}$ ab.

<!--en:b6822252-->
- **Polyaks optimale Parameterwahl $(\eta^\star, \beta^\star)$** bringt die kritischen Schwellen für $\lambda_{\min}$ und $\lambda_{\max}$ zur Deckung: $(1 - \sqrt{\eta^\star\lambda_{\min}})^2 = (\sqrt{\eta^\star\lambda_{\max}} - 1)^2 = \beta^\star$.

<!--en:6cf99f82-->
::: {.slide title="Quantitative Prüfung am Tal mit κ = 20"}

<!--en:adf60e20-->
[40 statt 139 Schritte]{.kicker}

<!--en:64d83061-->
- Für das betrachtete Tal ($\lambda_1 = 0.2, \lambda_2 = 4.0, \kappa = 20$) gilt:
  - Die optimale Parameterwahl ergibt $\eta^\star \approx 0.668$ und $\beta^\star \approx 0.403$.
  - Der Kontraktionsfaktor pro Schritt sinkt von $0.905$ (einfacher Gradientenabstieg) auf $\sqrt{\beta^\star} \approx 0.635$.
  - Bis zu einem relativen Fehler von $10^{-6}$ sinkt die Schrittzahl um den Faktor **$3.5\times$**, von **$139$ auf $40$ Schritte**.

<!--en:72f9cc3a-->
::: {.slide title="Nachschwingen bei β oberhalb der kritischen Schwelle"}

<!--en:6c368af7-->
[Gefahr des Überschwingens]{.kicker}

<!--en:287e88b7-->
- Was geschieht bei $\beta = 0.8$ und $\eta = 0.3$, wenn der Momentum-Koeffizient damit weit über den kritischen Schwellen beider Koordinaten liegt ($\beta_{\text{c}}(\lambda_1) = 0.57, \beta_{\text{c}}(\lambda_2) = 0.01$)?

<!--en:aec81681-->
- Beide Koordinaten sind stark unterdämpft. Die komplexen Wurzeln haben den Betrag $\sqrt{0.8} \approx 0.894$ (kaum schnellerer Abfall als beim einfachen Gradientenabstieg mit $0.905$).

<!--en:09ac7d43-->
- In der Geschwindigkeit wirkt der Antrieb der letzten $\tau = \frac{1}{1-0.8} = 5$ Schritte bergab nach. Die Trajektorie schießt über den Ursprung hinaus und schwingt entlang beider Achsen hin und her.

<!--en:8ff6b868-->
::: {.slide title="Zwei Regime auf realen Daten: Krümmung und Rauschen"}

<!--en:eb9c79b3-->
[Empirische Prüfung auf Fashion-MNIST]{.kicker}

<!--en:5c602bb3-->
Die Theorie sagt für den Vergleich von $\beta = 0$ und $\beta = 0.9$ auf Fashion-MNIST zwei unterschiedliche Ergebnisse voraus:

<!--en:7e278928-->
- **1. Krümmungsbegrenzt (Softmax-Regression auf dem gesamten Trainingssatz, $b = 60{,}000$):** Exakte Gradienten, schlecht konditionierte Kovarianzmatrix der Pixel. Mit $\beta = 0.9$ ist die optimale effektive Schrittweite $32\times$ so groß ($\eta_{\text{eff}}: 0.125 \to 4.0$). Bis zu einem Verlust von $0.5$ sinkt die Schrittzahl um den Faktor **$6.25\times$** ($625 \to 100$ Schritte).

<!--en:19986031-->
- **2. Rauschbegrenzt (`FashionCNN`, kleiner Batch mit $b = 32$):** Die optimale Lernrate $\eta$ sinkt um den Faktor $\sim 10\times$ ($0.2 \to 0.02$). So bleibt $\eta_{\text{eff}} = \frac{\eta}{1-\beta}$ gleich. Schrittzahl und Endverlust sind vergleichbar.

<!--en:ead647db-->
::: {.slide title="Nesterovs beschleunigter Gradientenabstieg: vorausschauende Korrektur"}

<!--en:f3f0260d-->
[Vorausschauende Dämpfung]{.kicker}

<!--en:8081036d-->
- Warum schoss das unterdämpfte Heavy-Ball-Momentum über das Minimum hinaus? Es berechnete $\nabla f(\mathbf{x}_t)$ an der aktuellen Position und berücksichtigte dabei noch nicht, wohin der Momentum-Schritt $-\eta\beta\mathbf{v}_{t-1}$ führen würde.

<!--en:8ada0986-->
- **Nesterov-Momentum (1983)** wertet den Gradienten am **vorausberechneten Punkt** $\tilde{\mathbf{x}}_t = \mathbf{x}_t - \eta\beta\,\mathbf{v}_{t-1}$ aus:
  $$\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + \nabla f(\mathbf{x}_t - \eta\beta\,\mathbf{v}_{t-1})$$

<!--en:3036a06c-->
- Die Auswertung am vorausberechneten Punkt berücksichtigt die Krümmung und bremst die Bewegung *vor* dem Überschwingen. Mit der Folge $\beta_t = (t-1)/(t+2)$ erreicht das Verfahren die optimale Rate $O(1/t^2)$ für allgemeine glatte konvexe Funktionen.

<!--en:3af6bc90-->
::: {.slide title="Nesterovs vorausschauender Schritt unterdrückt das Nachschwingen"}

<!--en:3f8bb317-->
[Heavy-Ball- und Nesterov-Momentum im Vergleich]{.kicker}

<!--en:d454abae-->
- Bei genau denselben Hyperparametern wie im unterdämpften Lauf ($\eta = 0.3, \beta = 0.8$) zeigt sich im Tal sofort die vorausschauende Dämpfung des Nesterov-Momentums:
  - In der steilen Richtung $x_2$ ist die Schwingung so weit abgeklungen, dass $|x_2| < 10^{-4}$ gilt, und zwar innerhalb von $20$ Schritten (gegenüber $-0.25$ beim Heavy-Ball-Momentum).
  - In Längsrichtung entlang von $x_1$ fällt das Überschwingen fast halb so groß aus.

<!--en:48251e3d-->
- In der Praxis ist $\beta = 0.9$ für Heavy-Ball- und Nesterov-Momentum ein üblicher Standardwert. Die Lernrate lässt sich über $\eta_{\text{eff}} = \frac{\eta}{1-\beta}$ abstimmen.

<!--en:60ba1718-->
::: {.slide title="Verständnisfragen: effektive Schrittweite, Zeitskalen und √κ"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:5cce3512-->
Für jedes Beispiel mit Momentum die Antwort zahlenmäßig bestimmen:

<!--en:44281f24-->
1. **Anpassung von $\eta$ bei Änderung von $\beta$:** Minibatch-SGD trainiert mit $\eta = 0.02$ und $\beta = 0.9$ erfolgreich. Der Momentum-Koeffizient steigt nun auf $\beta = 0.98$. Welche Lernrate $\eta$ hält die effektive Schrittweite $\eta_{\text{eff}}$ und die Größe der stationären Kugel um das Minimum konstant? Wie ändert sich das Gedächtnisfenster $\tau$?

<!--en:0e4a1be0-->
2. **$\sqrt{\kappa}$-Beschleunigung:** Für eine quadratische Funktion mit Konditionszahl $\kappa = 10{,}000$: Welcher Momentum-Koeffizient $\beta^\star$ ist optimal? Um welchen Faktor sinkt mit optimalem Momentum die Zahl der Schritte, die den Fehler auf das $10^{-2}$-Fache reduzieren, verglichen mit optimal eingestelltem einfachem Gradientenabstieg?

<!--en:9b831269-->
3. **Übergang bei wachsender Batchgröße:** Warum sank mit $\beta = 0.9$ die Schrittzahl um den Faktor $6\times$ bei der Softmax-Regression auf dem gesamten Trainingssatz ($b = 60{,}000$), aber nicht bei `FashionCNN` mit $b = 32$?

<!--en:a5a47016-->
::: {.slide title="Verständnisfragen: effektive Schrittweite, Zeitskalen und √κ (Lösungen)"}

<!--en:929b7671-2-->
[Wissensprüfung]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:50b29342-->
1. Bei $\beta = 0.9$ gelten $\eta_{\text{eff}} = \frac{0.02}{1 - 0.9} = 0.2$ und ein Gedächtnisfenster von $\tau = 10$ Schritten. Für $\beta = 0.98$ ($1 - \beta = 0.02$) bleibt $\eta_{\text{eff}} = 0.2$ konstant, wenn **$\eta = 0.2 \times 0.02 = 0.004$** gilt (um den Faktor $5\times$ kleiner). Das Gedächtnisfenster wächst **um den Faktor $5\times$ ($10 \to 50$ Schritte)**.

<!--en:b4b6c3b9-->
2. Aus $\sqrt{\kappa} = 100$ folgt **$\beta^\star = (99/101)^2 \approx 0.9608$**. Die Zahl der nötigen Schritte sinkt um den Faktor **$\frac{\kappa}{\sqrt{\kappa}} = \sqrt{\kappa} = 100\times$** ($\approx 230$ statt $23{,}000$ Schritte).

<!--en:0dedd414-->
3. Bei $b = 32$ ist das Training **rauschbegrenzt** ($b \ll b_{\text{crit}}$). Die Größe der Kugel um das Minimum begrenzt daher $\eta_{\text{eff}}$. Beim Training auf dem gesamten Trainingssatz ($b = 60{,}000$) ist das Gradientenrauschen null, das Training also **krümmungsbegrenzt**. Mit Momentum ist dann eine $32\times$ so große effektive Schrittweite $\eta_{\text{eff}}$ möglich.

<!--en:5126529b-->
::: {.slide title="Zusammenfassung: Heavy-Ball- und Nesterov-Momentum"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:0382e49d-->
| Begriff oder Regel | Mathematischer Ausdruck | Wesentliche Eigenschaft | Faustregel |

<!--en:90440225-->
| **Geschwindigkeit beim Heavy-Ball-Momentum** | $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \mathbf{g}_t,\; \mathbf{x}_{t+1} = \mathbf{x}_t - \eta\mathbf{v}_t$ | Wiederholtes Einsetzen ergibt $\sum_{k=0}^{t-1} \beta^k \mathbf{g}_{t-k}$ | Standardwert $\beta = 0.9$, zusätzlich 1 Zustandswert pro Parameter speichern |

<!--en:71cf2ec9-->
| **Effektive Schrittweite und Gedächtnisfenster** | $\eta_{\text{eff}} = \frac{\eta}{1-\beta},\quad \tau = \frac{1}{1-\beta}$ | Dämpft Schritte bei wechselndem Vorzeichen auf $\frac{\eta}{1+\beta}$. Stabilitätsgrenze: $\eta\lambda < 2(1+\beta)$ | Bei Rauschen $\eta \propto (1-\beta)$ wählen, wenn sich $\beta$ ändert |

<!--en:ce449060-->
| **Kritische Dämpfung und $\sqrt{\kappa}$** | $\beta_{\text{c}}(\lambda) = (1 - \sqrt{\eta\lambda})^2,\; \rho^\star = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ | Für $\beta \ge \beta_{\text{c}}$ haben die komplexen Wurzeln den Betrag $|z| = \sqrt{\beta}$ | Senkt die Schrittzahl im krümmungsbegrenzten Fall von $O(\kappa)$ auf $O(\sqrt{\kappa})$ |

<!--en:fc134325-->
| **Nesterovs vorausschauender Schritt** | $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \nabla f(\mathbf{x}_t - \eta\beta\mathbf{v}_{t-1})$ | Berücksichtigt die Krümmung an der vorausberechneten Position $\tilde{\mathbf{x}}_t$ | Unterdrückt Nachschwingen bei Unterdämpfung und bildet die Grundlage von Muon (§{sec_muon}) |
